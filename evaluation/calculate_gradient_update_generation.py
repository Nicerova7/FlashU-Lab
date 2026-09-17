# coding=utf-8
"""Collect loss-aware gradient x update scores for generation tokens.

For each prompt, Show-o2 first generates a reference image without gradients. The
generated latent then becomes the target for the flow-matching gradient pass.

Run from the repository root:
  python -m evaluation.calculate_gradient_update_generation \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    generation_data_file=prompts/generation_counting_spatial_40.json \
    num_prompts=1 layers=all steps=all output_dir=gradient_update_generation
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
import gzip
import logging
import math
import sys
import time
import uuid

import torch
import torch.nn.functional as F

from evaluation.calculate_token_importance_generation import (
    create_generation_sampler,
    load_generation_examples,
    prepare_generation_inputs,
)
from evaluation.calculate_token_importance_understanding import (
    atomic_json,
    atomic_torch_save,
    file_sha256,
    find_repository_root,
    load_config,
    load_model_resources,
    prepare_output_directory,
    repo_commit,
    seed_example,
    select_device_and_dtype,
    select_layers,
)

SCHEMA_VERSION = 1
SCORE_NAME = "token_gradient_update_flow_loss_v1"
LOGGER = logging.getLogger(__name__)


def experiment_settings(config):
    count = int(config.get("num_inference_steps", config.transport.num_inference_steps))
    if count < 2:
        raise ValueError("num_inference_steps must be at least 2")
    steps = select_layers(config.get("steps", "all"), count - 1)
    times = torch.linspace(0.0, 1.0, count)[:-1]
    reference_guidance = float(
        config.get("reference_guidance_scale", config.transport.guidance_scale)
    )
    if not math.isfinite(reference_guidance) or reference_guidance < 0:
        raise ValueError("reference_guidance_scale must be finite and nonnegative")
    return {
        "num_time_grid_points": count,
        "recorded_steps": steps,
        "timesteps": [float(times[index]) for index in steps],
        "guidance_scale": 0.0,
        "reference_guidance_scale": reference_guidance,
        "atol": float(config.transport.atol),
        "rtol": float(config.transport.rtol),
        "time_shifting_factor": config.get("time_shifting_factor", None),
    }


def run_decoder(core, input_embeds, attention_mask, layer_ids):
    hidden_states = input_embeds
    cache_position = torch.arange(input_embeds.shape[1], device=input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = core.rotary_emb(hidden_states, position_ids)
    selected = set(layer_ids)
    records = []

    for layer_index, layer in enumerate(core.layers):
        hidden_in = None
        if layer_index in selected:
            hidden_in = hidden_states[0].detach().clone()
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
        if layer_index in selected:
            records.append((layer_index, hidden_in, hidden_states))
    return core.norm(hidden_states), records


class BackboneRecorder:
    def __init__(self, resources):
        self.resources = resources
        self.records = None

    def forward(self, *, inputs_embeds, attention_mask, output_hidden_states=True, **kwargs):
        final_hidden, self.records = run_decoder(
            self.resources.model.showo.model,
            inputs_embeds,
            attention_mask,
            self.resources.layer_ids,
        )
        return {"logits": None, "hidden_states": (final_hidden,)}


@contextmanager
def measured_backbone(model, recorder):
    original = model.showo.forward
    model.showo.forward = recorder.forward
    try:
        yield
    finally:
        model.showo.forward = original


def score_step(resources, inputs, image_latents, target_velocity, timestep):
    recorder = BackboneRecorder(resources)
    image_latents = image_latents.detach().requires_grad_(True)
    t = torch.tensor([timestep], device=resources.device, dtype=image_latents.dtype)

    with measured_backbone(resources.model, recorder):
        _, velocity = resources.model(
            text_tokens=inputs["text_tokens"],
            image_latents=image_latents,
            t=t,
            attention_mask=inputs["attention_mask"],
            modality_positions=inputs["modality_positions"],
            max_seq_len=inputs["sequence_length"],
            guidance_scale=0.0,
            output_hidden_states=True,
        )

    loss = F.mse_loss(velocity.float(), target_velocity.float())
    gradients = torch.autograd.grad(loss, [record[2] for record in recorder.records])
    layers = {}
    for (layer_index, hidden_in, hidden_out), gradient in zip(recorder.records, gradients):
        update = hidden_out[0].detach().float() - hidden_in.float()
        score = (gradient[0].detach().float() * update).sum(-1).abs()
        layers[str(layer_index)] = {
            "input_features": hidden_in.cpu(),
            "gradient_update": score.cpu(),
        }
    return layers, float(loss.detach().cpu())


def save_step(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with gzip.open(temporary, "wb", compresslevel=6) as stream:
            torch.save(payload, stream)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def generate_reference(record_index, inputs, settings, resources, output_dir):
    from PIL import Image
    from utils import denorm

    patch_size = int(resources.model.config.patch_size)
    noise = torch.randn(
        (
            1,
            int(resources.model.config.image_latent_dim),
            inputs["grid_height"] * patch_size,
            inputs["grid_width"] * patch_size,
        ),
        device=resources.device,
        dtype=resources.dtype,
    )
    sampler = create_generation_sampler(settings)
    sampler_noise = noise
    if settings["reference_guidance_scale"] > 0:
        sampler_noise = torch.cat([noise, noise])
    trajectory = sampler(
        sampler_noise,
        resources.model.t2i_generate,
        text_tokens=inputs["text_tokens"],
        attention_mask=inputs["attention_mask"],
        modality_positions=inputs["modality_positions"],
        max_seq_len=inputs["sequence_length"],
        guidance_scale=settings["reference_guidance_scale"],
    )
    reference_latents = trajectory[-1, :1].to(
        device=resources.device, dtype=resources.dtype
    )
    pixels = resources.vae.batch_decode(reference_latents.unsqueeze(2)).squeeze(2)
    image_path = output_dir / f"sample_{record_index:06d}_reference.png"
    Image.fromarray(denorm(pixels)[0]).save(image_path)
    return reference_latents, noise, image_path


def measure_one(record, record_index, config, settings, resources, output_dir, run, writer):
    started = time.perf_counter()
    seed = run["base_seed"] + record_index
    seed_example(seed, resources.device)
    inputs = prepare_generation_inputs(record, resources, config, settings)
    reference_inputs = prepare_generation_inputs(
        record,
        resources,
        config,
        {"guidance_scale": settings["reference_guidance_scale"]},
    )
    reference_latents, noise, image_path = generate_reference(
        record_index, reference_inputs, settings, resources, output_dir
    )
    target_velocity = reference_latents - noise
    layout = inputs["branches"]["conditional"]
    matrices = []
    losses = []

    for step, timestep in zip(settings["recorded_steps"], settings["timesteps"]):
        latent = (1.0 - timestep) * noise + timestep * reference_latents
        layers, loss = score_step(resources, inputs, latent, target_velocity, timestep)
        indices = sorted(map(int, layers))
        matrix = torch.stack([layers[str(index)]["gradient_update"] for index in indices])
        payload = {
            "schema_version": SCHEMA_VERSION,
            "score_name": SCORE_NAME,
            "run_id": run["run_id"],
            "sample": {
                "sample_id": f"{run['dataset_sha256'][:12]}:{record_index}",
                "record_index": record_index,
                "source_record": record,
                "task": "generation",
                "phase": "flow_matching",
                "step_index": step,
                "denoising_timestep": timestep,
                "cfg_branch": "conditional",
                "task_loss": loss,
            },
            "tokens": layout["tokens"],
            "valid_token_mask": layout["valid_token_mask"],
            "token_positions": torch.arange(inputs["sequence_length"]),
            "visual_token_positions": layout["visual_token_positions"],
            "attention_span": layout["attention_span"],
            "layer_indices": indices,
            "layers": layers,
            "gradient_update_matrix": matrix,
            "image_latents": latent.detach().cpu(),
        }
        path = output_dir / f"sample_{record_index:06d}_step_{step:04d}_conditional.pt.gz"
        save_step(path, payload)
        valid = layout["valid_token_mask"]
        for layer_index, score in zip(indices, matrix):
            values = score[valid]
            writer.writerow({
                "record_index": record_index,
                "step_index": step,
                "denoising_timestep": timestep,
                "layer": layer_index,
                "task_loss": loss,
                "num_valid_tokens": len(values),
                "min": values.min().item(),
                "mean": values.mean().item(),
                "median": values.median().item(),
                "max": values.max().item(),
            })
        matrices.append(matrix)
        losses.append(loss)
        LOGGER.info("Sample %d | step %d | loss %.6f", record_index, step, loss)

    sample = {
        "sample_id": f"{run['dataset_sha256'][:12]}:{record_index}",
        "record_index": record_index,
        "source_record": record,
        "reference_image_path": str(image_path.resolve()),
        "reference_image_sha256": file_sha256(image_path),
        "reference_source": "generated_by_showo2",
        "reference_guidance_scale": settings["reference_guidance_scale"],
        "seed": seed,
        "task": "generation",
        "phase": "flow_matching",
        "collection_seconds": time.perf_counter() - started,
    }
    atomic_torch_save(output_dir / f"sample_{record_index:06d}.pt", {
        "schema_version": SCHEMA_VERSION,
        "score_name": SCORE_NAME,
        "run_id": run["run_id"],
        "sample": sample,
        "layer_indices": resources.layer_ids,
        "step_indices": settings["recorded_steps"],
        "denoising_timesteps": settings["timesteps"],
        "task_losses": losses,
        "score_axes": ["layer", "step", "token"],
        "branches": {
            "conditional": {
                **layout,
                "gradient_update_tensor": torch.stack(matrices, dim=1),
            }
        },
        "initial_noise": noise.detach().cpu(),
        "reference_latents": reference_latents.detach().cpu(),
    })


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return

    config = load_config()
    root = find_repository_root()
    settings = experiment_settings(config)
    data_file, records, start = load_generation_examples(config)
    if "output_dir" not in config:
        config.output_dir = "gradient_update_generation"
    output_dir = prepare_output_directory(config)
    device, dtype = select_device_and_dtype(config)
    run = {
        "schema_version": SCHEMA_VERSION,
        "score_name": SCORE_NAME,
        "run_id": str(uuid.uuid4()),
        "status": "running",
        "task": "generation",
        "label_definition": "abs(dL/dh_out dot (h_out-h_in))",
        "loss": "conditional_linear_path_flow_matching_mse_self_generated_reference",
        "loss_status": "self_generated_pseudo_target_not_ground_truth_flow_matching_loss",
        "base_seed": int(config.get("seed", 42)),
        "dataset_path": str(data_file.resolve()),
        "dataset_sha256": file_sha256(data_file),
        "actual_repo_commit": repo_commit(root),
        "effective_sampling": settings,
        "num_requested_samples": len(records),
        "num_saved_samples": 0,
    }
    atomic_json(output_dir / "run.json", run)
    started = time.perf_counter()

    try:
        resources = load_model_resources(config, device, dtype)
        resources.model.requires_grad_(False)
        resources.model.eval()
        run.update(
            selected_layers=resources.layer_ids,
            model_config=dict(resources.model.config),
            backbone_config=resources.model.showo.model.config.to_dict(),
        )
        atomic_json(output_dir / "run.json", run)
        fields = [
            "record_index", "step_index", "denoising_timestep", "layer", "task_loss",
            "num_valid_tokens", "min", "mean", "median", "max",
        ]
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record_index, record in enumerate(records, start):
                measure_one(
                    record, record_index, config, settings, resources, output_dir, run, writer
                )
                stream.flush()
                run["num_saved_samples"] += 1
                atomic_json(output_dir / "run.json", run)
        run["status"] = "complete"
    except Exception as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output_dir / "run.json", run)


if __name__ == "__main__":
    main()
