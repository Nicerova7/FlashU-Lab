# coding=utf-8
# Generation preparation follows NUS Show Lab / FlashU-Lab (Apache-2.0).
r"""Record token updates during a complete Show-o2 text-to-image trajectory.

Start at main(), then measure_one_example(). GenerationRecorder.evaluate() runs
one denoising evaluation; run_decoder_layers() contains the explicit layer loop.
The native model still assembles text/visual/time embeddings and runs its flow head.

I_gen(layer, step, token) = L2(float32(block_out) - float32(block_in)).
These are candidate update-magnitude labels, not validated removal importance.
No pruning, answer decoding, or MLP training is performed.

Run from the repository root:
  python -m evaluation.calculate_token_importance_generation \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    generation_data_file=prompts/generation_counting_spatial_40.json \
    num_prompts=1 layers=all steps=all output_dir=token_generation_pilot

Options: num_inference_steps=50; cfg_branches=conditional (or both);
save_input_features=true; save_hidden_out=false; truncate_prompt=false.
Euler only: 50 time-grid points produce 49 model evaluations, indexed 0..48.
steps=[0,24,48] records only those evaluations but still finishes generation.
See TOKEN_IMPORTANCE_GENERATION_README.md for the schema and storage estimate.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import sys
import time
import uuid

import torch

from evaluation.calculate_token_importance_understanding import (
    atomic_json, atomic_torch_save, compute_update_labels, file_sha256,
    find_repository_root, load_config, load_model_resources,
    prepare_output_directory, repo_commit, seed_example,
    select_device_and_dtype, select_layers,
)

SCHEMA_VERSION = 3
SCORE_NAME = "token_decoder_update_l2_v2"  # Same equation as understanding.
LOGGER = logging.getLogger(__name__)


# Helpers for steps 1-3: inputs, settings, and provenance

def load_generation_examples(config):
    """Read a JSON prompt list or one prompt per line, preserving source indices."""
    data_file = Path(config.get("generation_data_file", "prompts/generation_counting_spatial_40.json"))
    contents = data_file.read_text(encoding="utf-8")
    records = json.loads(contents) if data_file.suffix.lower() == ".json" else contents.splitlines()
    start = int(config.get("start_sample_index", 0))
    count = int(config.get("num_prompts", 1))
    if not isinstance(records, list) or start < 0 or count < 1 or start + count > len(records):
        raise ValueError("Request a nonempty prompt range inside the dataset")
    selected = []
    for index, record in enumerate(records[start:start + count], start):
        record = {"prompt": record} if isinstance(record, str) else record
        if not isinstance(record, dict) or not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"Record {index} needs a nonempty prompt")
        selected.append(record)
    return data_file, selected, start


def generation_settings(config):
    """Use the demo's Euler schedule; distinguish grid points from evaluations."""
    transport = config.transport
    count = int(config.get("num_inference_steps", transport.num_inference_steps))
    if count < 2 or transport.sampling_method != "euler":
        raise ValueError("This collector requires Euler with num_inference_steps >= 2")
    if transport.path_type != "Linear" or transport.prediction != "velocity" or transport.get("reverse", False):
        raise ValueError("This pilot supports forward Linear/velocity text-to-image generation")
    guidance = float(config.get("guidance_scale", transport.guidance_scale))
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    branches = str(config.get("cfg_branches", "conditional"))
    if branches not in {"conditional", "both"} or (branches == "both" and guidance == 0):
        raise ValueError("cfg_branches must be conditional, or both with positive guidance_scale")
    # The demo does not pass the YAML time-shift settings to sample_ode().
    # A top-level override here explicitly enables a different, recorded schedule.
    shift = config.get("time_shifting_factor", None)
    if shift is not None and (not math.isfinite(float(shift)) or float(shift) <= 0):
        raise ValueError("time_shifting_factor must be positive when explicitly supplied")
    try:
        steps = select_layers(config.get("steps", "all"), count - 1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"steps must select Euler evaluation indices 0..{count - 2}") from error
    return {
        "num_time_grid_points": count, "expected_model_evaluations": count - 1,
        "recorded_steps": steps, "sampling_method": "euler", "guidance_scale": guidance,
        "recorded_cfg_branches": ["conditional", "unconditional"] if branches == "both" else ["conditional"],
        "time_shifting_factor": None if shift is None else float(shift),
        "do_shift": False, "reverse": False,
        "atol": float(transport.atol), "rtol": float(transport.rtol),
        "save_input_features": bool(config.get("save_input_features", True)),
        "save_hidden_out": bool(config.get("save_hidden_out", False)),
    }


def create_run_metadata(config, settings, root, data_file, records, device, dtype):
    """Record the actual generation protocol separately from the supplied YAML."""
    from omegaconf import OmegaConf

    sources = [
        "evaluation/calculate_token_importance_understanding.py", "inference_t2i.py",
        "models/modeling_showo2_qwen2_5.py", "models/qwen2.py", "models/misc.py",
        "models/omni_attention.py", "transport/__init__.py", "transport/transport.py", "transport/integrators.py",
    ]
    versions = {}
    for name in ["torch", "transformers", "torchdiffeq", "omegaconf", "torchvision"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": SCHEMA_VERSION, "score_name": SCORE_NAME,
        "run_id": str(uuid.uuid4()), "status": "running",
        "task": "generation", "phase": "denoising", "text_decode_steps": 0,
        "label_definition": "I_gen(layer, step, token) = L2(float32(block_out)-float32(block_in))",
        "label_status": "candidate_local_update_magnitude_not_validated_removal_damage",
        "token_scope": "all_sequence_positions; padding identified and excluded from summaries",
        "reference_pipeline": "inference_t2i.py -> native t2i_generate -> native multimodal forward",
        "effective_sampling": settings,
        "base_seed": int(config.get("seed", 42)),
        "device": str(device), "dtype": str(dtype).removeprefix("torch."),
        "versions": versions,
        "dataset_path": str(data_file.resolve()), "dataset_sha256": file_sha256(data_file),
        "actual_repo_commit": repo_commit(root), "script_sha256": file_sha256(__file__),
        "source_sha256": {name: file_sha256(root / name) for name in sources},
        "config": OmegaConf.to_container(config, resolve=True),
        "num_requested_samples": len(records), "num_saved_samples": 0,
        "split_guidance": "Keep each prompt and all its seeds/steps/branches together across dataset splits",
    }


# Helpers for step 4: native generation sequence and sampler

def prepare_generation_inputs(record, resources, config, settings):
    """Build the native padded sequence and branch-specific token metadata."""
    from models import omni_attn_mask_naive
    from models.misc import prepare_gen_input

    model, tokenizer, ids = resources.model, resources.tokenizer, resources.token_ids
    height = int(config.dataset.preprocessing.latent_height)
    width = int(config.dataset.preprocessing.latent_width)
    num_visual = height * width
    add_time = bool(model.config.add_time_embeds)
    num_image_slots = num_visual + int(add_time)
    max_seq = int(config.dataset.preprocessing.max_seq_length)
    max_text = max_seq - num_image_slots - 4  # BOS, BOI, EOI, EOS.
    if max_text < 1 or num_visual != int(config.dataset.preprocessing.num_t2i_image_tokens):
        raise ValueError("Check max_seq_length, latent grid, and num_t2i_image_tokens (visual patches only)")
    # Native forward uses its fixed positional-ID grid at this checkpoint size.
    native_grid = (int(model.config.image_latent_height), int(model.config.image_latent_width))
    if (height, width) != native_grid or model.image_position_ids.shape[-1] != num_visual:
        raise ValueError("Use the checkpoint's native latent grid for this first generation experiment")
    original_count = len(tokenizer(record["prompt"], add_special_tokens=False)["input_ids"])
    if original_count > max_text and not config.get("truncate_prompt", False):
        raise ValueError(f"Prompt has {original_count} tokens; budget is {max_text}. Set truncate_prompt=true explicitly")

    # TEXT TOKENS: BOS -> prompt -> BOI -> image placeholders -> EOI -> EOS -> padding.
    conditional, unconditional, pos_cond, pos_uncond = prepare_gen_input(
        [record["prompt"]], tokenizer, num_image_slots,
        ids["bos_id"], ids["eos_id"], ids["boi_id"], ids["eoi_id"],
        tokenizer.pad_token_id, ids["img_pad_id"], max_text, resources.device,
    )
    guided = settings["guidance_scale"] > 0
    text_tokens = torch.cat([conditional, unconditional]) if guided else conditional
    modality_positions = torch.cat([pos_cond, pos_uncond]) if guided else pos_cond
    attention_mask = omni_attn_mask_naive(
        text_tokens.shape[0], max_seq, modality_positions, resources.device,
    ).to(resources.dtype)
    branches = ["conditional", "unconditional"] if guided else ["conditional"]
    metadata = {}
    for batch_index, branch in enumerate(branches):
        offset, length = modality_positions[batch_index, 0].tolist()
        tokens = describe_generation_tokens(
            text_tokens[batch_index].tolist(), tokenizer, offset, length, add_time, width,
        )
        metadata[branch] = {
            "batch_index": batch_index, "tokens": tokens,
            "valid_token_mask": torch.tensor([token["token_type"] != "padding" for token in tokens]),
            "visual_token_positions": torch.tensor([token["sequence_position"] for token in tokens
                                                     if token["token_type"] == "visual"]),
            "attention_span": {"offset": offset, "length": length},
            "prompt_token_count": offset - 2,
        }
    return {
        "text_tokens": text_tokens, "attention_mask": attention_mask,
        "modality_positions": modality_positions, "branches": metadata,
        "sequence_length": max_seq, "grid_height": height, "grid_width": width,
        "num_visual_tokens": num_visual, "original_prompt_token_count": original_count,
        "used_prompt_token_count": min(original_count, max_text),
        "prompt_truncated": original_count > max_text,
    }


def describe_generation_tokens(token_ids, tokenizer, offset, length, add_time, grid_width):
    """Type by sequence structure, so EOS/padding sharing an ID is unambiguous."""
    first_visual = offset + int(add_time)
    tokens = []
    for position, token_id in enumerate(token_ids):
        patch = None
        if position == 0 or position == offset + length + 1:
            kind = "chat_format"  # Native BOS/EOS; no understanding system prompt.
        elif position in {offset - 1, offset + length}:
            kind = "image_boundary"
        elif add_time and position == offset:
            kind = "time"
        elif first_visual <= position < offset + length:
            kind, patch = "visual", position - first_visual
        elif position > offset + length + 1:
            kind = "padding"
        else:
            kind = "prompt_text"
        continuous = kind in {"visual", "time"}
        tokens.append({
            "sequence_position": position, "token_type": kind,
            "token_id": None if continuous else token_id,
            "token_piece": None if continuous else tokenizer.convert_ids_to_tokens(token_id),
            "patch_index": patch, "patch_row": None if patch is None else patch // grid_width,
            "patch_col": None if patch is None else patch % grid_width,
        })
    return tokens


def create_generation_sampler(settings):
    """Use the repository's solver, including its float32 latent integration."""
    from transport import Sampler, create_transport

    transport = create_transport(path_type="Linear", prediction="velocity")
    return Sampler(transport).sample_ode(
        sampling_method="euler", num_steps=settings["num_time_grid_points"],
        atol=settings["atol"], rtol=settings["rtol"],
        time_shifting_factor=settings["time_shifting_factor"],
    )


# Helpers for step 5: denoising evaluations and the explicit decoder loop

@torch.no_grad()
def run_decoder_layers(core, input_embeds, attention_mask, layer_ids, branch_indices,
                       save_input_features=True, save_hidden_out=False):
    """Execute every layer, record selected boundaries, then apply final RMSNorm."""
    if core.training or input_embeds.ndim != 3:
        raise ValueError("Require model.eval() and backbone inputs [batch, tokens, hidden]")
    if attention_mask.shape[-2:] != (input_embeds.shape[1], input_embeds.shape[1]):
        raise ValueError("Attention mask does not match the backbone sequence")
    results = {branch: {} for branch in branch_indices}
    selected_layers = set(layer_ids)
    hidden_states = input_embeds  # FINAL BACKBONE INPUT: native text + time + visual embeddings.
    cache_position = torch.arange(input_embeds.shape[1], device=input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = core.rotary_emb(hidden_states, position_ids)

    for layer_index, layer in enumerate(core.layers):
        # --- A. Keep this layer's incoming token vectors ---
        record = layer_index in selected_layers and bool(branch_indices)
        if record:
            hidden_in = {branch: hidden_states[index].detach().clone()
                         for branch, index in branch_indices.items()}

        # --- B. Execute the layer for all active guidance branches ---
        hidden_states = layer(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=None, output_attentions=False, use_cache=False,
            cache_position=cache_position, position_embeddings=position_embeddings,
        )[0]
        if hidden_states.shape != input_embeds.shape:
            raise ValueError(f"Token sequence shape changed at layer {layer_index}")

        # --- C. Score each token's update through this layer ---
        if record:
            for branch, batch_index in branch_indices.items():
                hidden_out = hidden_states[batch_index].detach()
                # change = token_after_layer - token_before_layer (cast to float32 first)
                # score  = sqrt(change[1]^2 + ... + change[d]^2)
                labels = compute_update_labels(hidden_in[branch], hidden_out)
                if not torch.isfinite(hidden_in[branch]).all() or not torch.isfinite(labels).all():
                    raise ValueError(f"Non-finite {branch} features/scores at layer {layer_index}")
                data = {"update_l2": labels.cpu()}
                if save_input_features:
                    data["input_features"] = hidden_in[branch].cpu()
                if save_hidden_out:
                    data["hidden_out"] = hidden_out.clone().cpu()
                results[branch][str(layer_index)] = data

    # Scores stop BEFORE final RMSNorm; the native flow head needs normalized states.
    return core.norm(hidden_states), results


@contextmanager
def use_measured_backbone(model, recorder):
    """Temporarily replace only the LM wrapper; restore it even if collection fails."""
    had_override = "forward" in model.showo.__dict__
    original_override = model.showo.__dict__.get("forward")
    model.showo.forward = recorder.backbone_forward
    try:
        yield
    finally:
        if had_override:
            model.showo.forward = original_override
        else:
            del model.showo.forward


class GenerationRecorder:
    """Connect a solver evaluation to its layer measurements and saved step files."""

    def __init__(self, resources, inputs, settings, sample, output_dir, run_id, writer, manifest):
        self.resources, self.inputs, self.settings = resources, inputs, settings
        self.sample, self.output_dir, self.run_id = sample, output_dir, run_id
        self.writer, self.manifest = writer, manifest
        self.evaluation_index = 0
        self.layer_data = None
        self.backbone_calls = 0
        self.record_current = False
        self.steps, self.timesteps = [], []
        self.matrices = {branch: [] for branch in settings["recorded_cfg_branches"]}

    def backbone_forward(self, *, inputs_embeds, attention_mask, output_hidden_states=True, **kwargs):
        """Receive the native assembled embeddings; run the measured decoder."""
        branches = {branch: self.inputs["branches"][branch]["batch_index"]
                    for branch in self.settings["recorded_cfg_branches"]} if self.record_current else {}
        final_hidden, self.layer_data = run_decoder_layers(
            self.resources.model.showo.model, inputs_embeds, attention_mask,
            self.resources.layer_ids, branches,
            self.settings["save_input_features"], self.settings["save_hidden_out"],
        )
        self.backbone_calls += 1
        # Native Show-o2 uses the normalized states for its diffusion head.
        # Its vocabulary logits are discarded by t2i_generate(), so omit that projection.
        return {"logits": None, "hidden_states": (final_hidden,)}

    @torch.no_grad()
    def evaluate(self, image_latents, t, **kwargs):
        """One solver call: native fusion -> measured layers -> native flow head/CFG."""
        index = self.evaluation_index
        self.record_current = index in self.settings["recorded_steps"]
        self.backbone_calls = 0
        # VISUAL TOKENS: native Show-o2.forward fuses patches of these current latents.
        # It replaces the time/visual placeholders before calling backbone_forward().
        velocity = self.resources.model.t2i_generate(image_latents=image_latents, t=t, **kwargs)
        if self.backbone_calls != 1 or not torch.isfinite(velocity).all():
            raise ValueError("Expected one finite native backbone/flow-head evaluation")
        if self.record_current:
            if not torch.all(t == t[0]):
                raise ValueError("This single-image experiment requires one timestep across CFG branches")
            timestep = float(t[0].item())
            for branch, layers in self.layer_data.items():
                payload = build_step_payload(
                    self.sample, self.run_id, self.inputs, branch, layers,
                    index, timestep, image_latents,
                )
                save_step(self.output_dir, payload, self.writer, self.manifest)
                self.matrices[branch].append(payload["update_l2_matrix"])
            self.steps.append(index)
            self.timesteps.append(timestep)
        self.layer_data = None  # Release features before the next denoising evaluation.
        self.evaluation_index += 1
        LOGGER.info("Sample %d | evaluation %d/%d | t=%.6f | recorded=%s",
                    self.sample["record_index"], index, self.settings["expected_model_evaluations"] - 1,
                    t[0].item(), self.record_current)
        return velocity


# Helpers for steps 5-6: aligned files and final image

def build_step_payload(sample, run_id, inputs, branch, layers, step, timestep, latents):
    """Each file has [layer, token] scores for one sample, timestep, and CFG branch."""
    layout = inputs["branches"][branch]
    indices = sorted(int(index) for index in layers)
    return {
        "schema_version": SCHEMA_VERSION, "score_name": SCORE_NAME, "run_id": run_id,
        "sample": {**sample, "step_index": step, "model_evaluation_index": step,
                   "denoising_timestep": timestep, "cfg_branch": branch,
                   "branch_prompt_token_count": layout["prompt_token_count"]},
        "tokens": layout["tokens"], "valid_token_mask": layout["valid_token_mask"],
        "token_positions": torch.arange(inputs["sequence_length"]),
        "visual_token_positions": layout["visual_token_positions"],
        "attention_span": layout["attention_span"],
        "layer_indices": indices, "layers": layers,
        "update_l2_matrix": torch.stack([layers[str(index)]["update_l2"] for index in indices]),
        "image_latents": latents[layout["batch_index"]:layout["batch_index"] + 1].detach().cpu().clone(),
    }


def save_step(output_dir, payload, writer, manifest):
    """Stream features to disk; layer statistics exclude right-padding rows."""
    sample = payload["sample"]
    filename = f"sample_{sample['record_index']:06d}_step_{sample['step_index']:04d}_{sample['cfg_branch']}.pt"
    atomic_torch_save(output_dir / filename, payload)
    for layer, data in payload["layers"].items():
        scores = data["update_l2"][payload["valid_token_mask"]]
        writer.writerow({
            "record_index": sample["record_index"], "step_index": sample["step_index"],
            "denoising_timestep": sample["denoising_timestep"], "cfg_branch": sample["cfg_branch"],
            "layer": layer, "num_valid_tokens": len(scores),
            "min": scores.min().item(), "mean": scores.mean().item(),
            "median": scores.median().item(), "max": scores.max().item(),
        })
    manifest.write(json.dumps({"file": filename, **sample}, ensure_ascii=False) + "\n")
    manifest.flush()


@torch.no_grad()
def measure_one_example(record, record_index, config, settings, resources, output_dir, run, writer, manifest):
    """Run the complete generation trajectory while streaming selected measurements."""
    started = time.perf_counter()
    sample_seed = run["base_seed"] + record_index
    seed_example(sample_seed, resources.device)

    # --- 5a. Prepare text tokens, masks, and one initial noise sample ---
    inputs = prepare_generation_inputs(record, resources, config, settings)
    patch_size = int(resources.model.config.patch_size)
    noise = torch.randn(
        (1, int(resources.model.config.image_latent_dim),
         inputs["grid_height"] * patch_size, inputs["grid_width"] * patch_size),
        device=resources.device, dtype=resources.dtype,
    )
    initial_noise = noise.cpu().clone()
    if settings["guidance_scale"] > 0:
        noise = torch.cat([noise, noise])  # Same image state; conditional then unconditional.
    sample = {
        "sample_id": f"{run['dataset_sha256'][:12]}:{record_index}", "record_index": record_index,
        "source_record": record, "seed": sample_seed, "task": "generation", "phase": "denoising",
        "sequence_length": inputs["sequence_length"], "num_visual_tokens": inputs["num_visual_tokens"],
        "grid_height": inputs["grid_height"], "grid_width": inputs["grid_width"],
        "original_prompt_token_count": inputs["original_prompt_token_count"],
        "used_prompt_token_count": inputs["used_prompt_token_count"], "prompt_truncated": inputs["prompt_truncated"],
        "guidance_scale": settings["guidance_scale"],
    }
    recorder = GenerationRecorder(resources, inputs, settings, sample, output_dir, run["run_id"], writer, manifest)
    sample_fn = create_generation_sampler(settings)

    # --- 5b. Install the measured layer loop, run denoising, then restore the wrapper ---
    with use_measured_backbone(resources.model, recorder):
        trajectory = sample_fn(
            noise, recorder.evaluate,
            text_tokens=inputs["text_tokens"], attention_mask=inputs["attention_mask"],
            modality_positions=inputs["modality_positions"], max_seq_len=inputs["sequence_length"],
            guidance_scale=settings["guidance_scale"],
        )
    if recorder.evaluation_index != settings["expected_model_evaluations"] or recorder.steps != settings["recorded_steps"]:
        raise ValueError("Solver evaluation count differs from the recorded Euler protocol")

    # --- 5c. Decode the final image with the native VAE ---
    from PIL import Image
    from utils import denorm

    final_latents = trajectory[-1, :1]  # Conditional copy when CFG is enabled.
    pixels = resources.vae.batch_decode(final_latents.unsqueeze(2)).squeeze(2)
    image_path = output_dir / f"sample_{record_index:06d}.png"
    Image.fromarray(denorm(pixels)[0]).save(image_path)

    # --- 5d. Save a small trajectory index with I_gen[layer, recorded_step, token] ---
    aggregate = {}
    for branch, matrices in recorder.matrices.items():
        aggregate[branch] = {
            **inputs["branches"][branch],
            "update_l2_tensor": torch.stack(matrices, dim=1),
        }
    atomic_torch_save(output_dir / f"sample_{record_index:06d}.pt", {
        "schema_version": SCHEMA_VERSION, "score_name": SCORE_NAME, "run_id": run["run_id"],
        "sample": {**sample, "image_path": str(image_path.resolve()), "image_sha256": file_sha256(image_path),
                   "collection_seconds": time.perf_counter() - started},
        "layer_indices": resources.layer_ids, "step_indices": recorder.steps,
        "denoising_timesteps": recorder.timesteps, "score_axes": ["layer", "step", "token"],
        "branches": aggregate, "initial_noise": initial_noise, "final_latents": final_latents.cpu(),
    })


# START HERE: the complete experiment, in execution order

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return

    # --- 1. Load configuration and select prompts, layers, and denoising evaluations ---
    config = load_config()
    root = find_repository_root()
    settings = generation_settings(config)
    data_file, records, start = load_generation_examples(config)
    base_seed = int(config.get("seed", 42))
    if base_seed < 0 or base_seed + start + len(records) > 2**32 - 1:
        raise ValueError("seed plus record index must fit in an unsigned 32-bit integer")

    # --- 2. Set the output directory, device, and precision ---
    if "output_dir" not in config:
        config.output_dir = "token_importance_generation"
    output_dir = prepare_output_directory(config)
    device, dtype = select_device_and_dtype(config)

    # --- 3. Save the run protocol ---
    run = create_run_metadata(config, settings, root, data_file, records, device, dtype)
    atomic_json(output_dir / "run.json", run)
    LOGGER.info("Euler: %d time-grid points, %d evaluations; recording steps %s",
                settings["num_time_grid_points"], settings["expected_model_evaluations"], settings["recorded_steps"])
    LOGGER.info("Recorded branches: %s; effective time_shifting_factor=%s (demo default: None)",
                settings["recorded_cfg_branches"], settings["time_shifting_factor"])
    started = time.perf_counter()
    try:
        # --- 4. Load the same Show-o2 checkpoint, tokenizer, and VAE ---
        resources = load_model_resources(config, device, dtype)
        run.update(selected_layers=resources.layer_ids,
                   model_config=dict(resources.model.config),
                   backbone_config=resources.model.showo.model.config.to_dict(),
                   vae_sha256=file_sha256(config.model.vae_model.pretrained_model_path))
        atomic_json(output_dir / "run.json", run)
        vectors = int(settings["save_input_features"]) + int(settings["save_hidden_out"])
        estimate = (vectors * len(resources.layer_ids) * len(settings["recorded_steps"])
                    * len(settings["recorded_cfg_branches"]) * int(config.dataset.preprocessing.max_seq_length)
                    * resources.model.showo.model.config.hidden_size * torch.empty((), dtype=dtype).element_size())
        LOGGER.info("Feature storage per prompt: approximately %.2f GiB; scores and latents are additional", estimate / 2**30)

        # --- 5. Generate each image and stream its token measurements ---
        fields = ["record_index", "step_index", "denoising_timestep", "cfg_branch", "layer",
                  "num_valid_tokens", "min", "mean", "median", "max"]
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summaries, \
             (output_dir / "steps.jsonl").open("w", encoding="utf-8") as manifest:
            writer = csv.DictWriter(summaries, fieldnames=fields)
            writer.writeheader()
            for index, record in enumerate(records, start):
                measure_one_example(record, index, config, settings, resources, output_dir, run, writer, manifest)
                summaries.flush()
                run["num_saved_samples"] += 1
                atomic_json(output_dir / "run.json", run)

        # --- 6. Mark the completed run ---
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
