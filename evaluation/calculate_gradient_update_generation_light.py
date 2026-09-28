# coding=utf-8
r"""Collect generation Gradient x Update targets without saving hidden vectors.

Uses the same reference generation, conditional loss, layers, steps and score
definition as calculate_gradient_update_generation.py. Writes one score aggregate
and one reference image per prompt; no *_step_*_conditional.pt.gz feature files.

Run from the repository root (with a NEW output folder):
  python -m evaluation.calculate_gradient_update_generation_light \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    generation_data_file=prompts/generation_counting_spatial_40.json \
    start_sample_index=12 num_prompts=28 layers=all steps=all \
    num_inference_steps=50 batch_size=1 device=cuda dtype=bfloat16 seed=42 \
    output_dir=gradient_update_generation_light

This reduces disk storage, not the number of gradient passes. These files are
evaluation targets, NOT drop-in training data for train_gradient_update_mlp.ipynb.
replay_step_features() recovers inputs later with a gradient-free forward pass
using the original frozen model/checkpoint, dtype and attention implementation.
"""

from __future__ import annotations

import csv
import importlib.metadata
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

import torch
import torch.nn.functional as F

from evaluation.calculate_gradient_update_generation import (
    BackboneRecorder,
    SCORE_NAME,
    experiment_settings,
    generate_reference,
    measured_backbone,
)
from evaluation.calculate_token_importance_generation import (
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
)

SCHEMA_VERSION = 1
STORAGE_FORMAT = "gradient_update_generation_light_v1"
LOGGER = logging.getLogger(__name__)


def score_step(resources, inputs, image_latents, target_velocity, timestep):
    """Return FP32 scores [selected layer, token], without CPU feature copies."""
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
    scores = {}
    for (layer_index, hidden_in, hidden_out), gradient in zip(recorder.records, gradients):
        update = hidden_out[0].detach().float() - hidden_in.float()
        score = (gradient[0].detach().float() * update).sum(-1).abs()
        scores[layer_index] = score.cpu()
    return torch.stack([scores[index] for index in sorted(scores)]), float(loss.detach().cpu())


def reconstruct_latent(payload, step_index, device="cpu"):
    """Recover a recorded linear-path state, preserving stored precision.

    step_index is the original step ID, not its position in a selected-step list.
    These states belong to the scoring path, not the reference sampler trajectory.
    """
    if payload.get("storage_format") != STORAGE_FORMAT:
        raise ValueError(f"Expected a {STORAGE_FORMAT} aggregate")
    try:
        position = payload["step_indices"].index(step_index)
    except ValueError as error:
        raise ValueError(f"Step {step_index} was not collected") from error
    timestep = payload["denoising_timesteps"][position]
    noise = payload["initial_noise"].to(device=device)
    reference = payload["reference_latents"].to(device=device)
    latent = (1.0 - timestep) * noise + timestep * reference
    return latent, timestep


@torch.no_grad()
def replay_step_features(resources, payload, step_index):
    """Recover block-input vectors for one step without calculating gradients.

    Returns the same per-layer input_features/gradient_update structure used by
    the full collector, in CPU memory only. Consume it and release it before the
    next step. Use the original model weights/config and backend recorded in
    run.json. Matching hardware/backend is needed for tight numerical agreement.
    """
    latent, timestep = reconstruct_latent(payload, step_index, resources.device)
    if resources.dtype != latent.dtype:
        raise ValueError("Replay dtype must match the saved reference/noise dtype")
    if resources.model.training:
        raise ValueError("Replay requires model.eval()")
    inputs = payload["replay_inputs"]
    recorder = BackboneRecorder(SimpleNamespace(
        model=resources.model, layer_ids=payload["layer_indices"],
    ))
    with measured_backbone(resources.model, recorder):
        resources.model(
            text_tokens=inputs["text_tokens"].to(resources.device),
            image_latents=latent,
            t=torch.tensor([timestep], device=resources.device, dtype=latent.dtype),
            attention_mask=inputs["attention_mask"].to(resources.device),
            modality_positions=inputs["modality_positions"].to(resources.device),
            max_seq_len=inputs["max_seq_len"],
            guidance_scale=0.0,
            output_hidden_states=True,
        )
    position = payload["step_indices"].index(step_index)
    targets = payload["branches"]["conditional"]["gradient_update_tensor"][:, position]
    target_by_layer = dict(zip(payload["layer_indices"], targets))
    return {
        str(index): {
            "input_features": hidden_in.cpu(),
            "gradient_update": target_by_layer[index].cpu(),
        }
        for index, hidden_in, _ in recorder.records
    }


def measure_one(record, record_index, config, settings, resources, output_dir, run, writer):
    started = time.perf_counter()
    seed = run["base_seed"] + record_index
    seed_example(seed, resources.device)
    inputs = prepare_generation_inputs(record, resources, config, settings)
    reference_inputs = prepare_generation_inputs(
        record, resources, config,
        {"guidance_scale": settings["reference_guidance_scale"]},
    )
    reference_latents, noise, image_path = generate_reference(
        record_index, reference_inputs, settings, resources, output_dir,
    )
    target_velocity = reference_latents - noise
    layout = inputs["branches"]["conditional"]
    indices = sorted(resources.layer_ids)
    matrices, losses = [], []

    for step, timestep in zip(settings["recorded_steps"], settings["timesteps"]):
        latent = (1.0 - timestep) * noise + timestep * reference_latents
        matrix, loss = score_step(resources, inputs, latent, target_velocity, timestep)
        valid = layout["valid_token_mask"]
        for layer_index, score in zip(indices, matrix):
            values = score[valid]
            writer.writerow({
                "record_index": record_index, "step_index": step,
                "denoising_timestep": timestep, "layer": layer_index,
                "task_loss": loss, "num_valid_tokens": len(values),
                "min": values.min().item(), "mean": values.mean().item(),
                "median": values.median().item(), "max": values.max().item(),
            })
        matrices.append(matrix)
        losses.append(loss)
        LOGGER.info("Sample %d | step %d | loss %.6f", record_index, step, loss)

    sample = {
        "sample_id": f"{run['dataset_sha256'][:12]}:{record_index}",
        "record_index": record_index, "source_record": record,
        "reference_image_path": str(image_path.resolve()),
        "reference_image_sha256": file_sha256(image_path),
        "reference_source": "generated_by_showo2",
        "reference_guidance_scale": settings["reference_guidance_scale"],
        "seed": seed, "task": "generation", "phase": "flow_matching",
        "collection_seconds": time.perf_counter() - started,
    }
    # Keep the two original-dtype endpoints and exact model inputs ONCE. This
    # permits later feature replay without storing a vector for every layer/step.
    payload = {
        "schema_version": SCHEMA_VERSION,
        "storage_format": STORAGE_FORMAT,
        "input_features_saved": False,
        "dtype": str(noise.dtype).removeprefix("torch."),
        "score_name": SCORE_NAME,
        "run_id": run["run_id"],
        "sample": sample,
        "layer_indices": indices,
        "step_indices": settings["recorded_steps"],
        "denoising_timesteps": settings["timesteps"],
        "task_losses": losses,
        "score_axes": ["layer", "step", "token"],
        "branches": {
            "conditional": {
                **layout,
                "gradient_update_tensor": torch.stack(matrices, dim=1),
            },
        },
        "initial_noise": noise.detach().cpu().clone(),
        "reference_latents": reference_latents.detach().cpu().clone(),
        "replay_inputs": {
            "text_tokens": inputs["text_tokens"].detach().cpu().clone(),
            "modality_positions": inputs["modality_positions"].detach().cpu().clone(),
            "attention_mask": inputs["attention_mask"].detach().cpu().clone(),
            "max_seq_len": inputs["sequence_length"],
            "guidance_scale": 0.0,
        },
    }
    destination = output_dir / f"sample_{record_index:06d}.pt"
    atomic_torch_save(destination, payload)
    LOGGER.info("Saved %s: %.2f MiB; no hidden-state feature files",
                destination.name, destination.stat().st_size / 2**20)


def runtime_metadata(resources):
    versions = {"torch": str(torch.__version__), "cuda": torch.version.cuda}
    for name in ("transformers", "tokenizers", "torchdiffeq"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    core = resources.model.showo.model
    return {
        "selected_layers": sorted(resources.layer_ids),
        "model_config": dict(resources.model.config),
        "backbone_config": core.config.to_dict(),
        "dtype": str(resources.dtype),
        "device": str(resources.device),
        "attention_implementation": getattr(core.config, "_attn_implementation", None),
        "versions": versions,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return
    from omegaconf import OmegaConf

    config = load_config()
    root = find_repository_root()
    settings = experiment_settings(config)
    data_file, records, start = load_generation_examples(config)
    if "output_dir" not in config:
        config.output_dir = "gradient_update_generation_light"
    output_dir = prepare_output_directory(config)
    device, dtype = select_device_and_dtype(config)
    source_paths = [
        Path(__file__).resolve(),
        root / "evaluation/calculate_gradient_update_generation.py",
        root / "evaluation/calculate_token_importance_generation.py",
        root / "evaluation/calculate_token_importance_understanding.py",
        root / "models/modeling_showo2_qwen2_5.py",
        root / "models/qwen2.py",
    ]
    run = {
        "schema_version": SCHEMA_VERSION, "storage_format": STORAGE_FORMAT,
        "input_features_saved": False, "score_name": SCORE_NAME,
        "run_id": str(uuid.uuid4()), "status": "running", "task": "generation",
        "label_definition": "abs(dL/dh_out dot (h_out-h_in))",
        "loss": "conditional_linear_path_flow_matching_mse_self_generated_reference",
        "loss_status": "self_generated_pseudo_target_not_ground_truth_flow_matching_loss",
        "base_seed": int(config.get("seed", 42)),
        "dataset_path": str(data_file.resolve()), "dataset_sha256": file_sha256(data_file),
        "actual_repo_commit": repo_commit(root),
        "source_sha256": {str(path.relative_to(root)): file_sha256(path) for path in source_paths},
        "effective_sampling": settings,
        "effective_config": OmegaConf.to_container(config, resolve=True),
        "num_requested_samples": len(records), "num_saved_samples": 0,
        "replay_note": "Use original frozen weights, config, dtype and attention backend; "
                       "replay conditional linear-path scoring states, not the sampler trajectory.",
    }
    atomic_json(output_dir / "run.json", run)
    started = time.perf_counter()
    try:
        resources = load_model_resources(config, device, dtype)
        resources.model.requires_grad_(False)
        resources.model.eval()
        run.update(runtime_metadata(resources))
        atomic_json(output_dir / "run.json", run)
        fields = [
            "record_index", "step_index", "denoising_timestep", "layer", "task_loss",
            "num_valid_tokens", "min", "mean", "median", "max",
        ]
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record_index, record in enumerate(records, start):
                measure_one(record, record_index, config, settings, resources, output_dir, run, writer)
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
