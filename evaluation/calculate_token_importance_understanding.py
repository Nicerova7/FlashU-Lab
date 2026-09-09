# coding=utf-8
# Understanding preparation adapted from NUS Show Lab / FlashU-Lab.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
r"""Measure every token's updates during Show-o2 understanding prefill.

Start at main(), then measure_one_example(). Input assembly is in
assemble_prefill(); the explicit layer loop is in collect_token_updates().

Prefill runs the complete input through every decoder layer once. All token types
are scored; all layers are recorded by default (28 for Show-o2-1.5B).
No answer decoding, pruning, or MLP training. Saved input vectors and L2 update
scores support a future predictor; the scores are candidate labels, not Hessian
OBD or validated removal damage. Equations are in compute_update_labels().

Preparation follows evaluation/calculate_obd_cache_understanding.py at FlashU-Lab
develop commit d30ae0d4f34fa9bce1121329db4c1747f30db737.

Run from the FlashU-Lab repository root:
  python -m evaluation.calculate_token_importance_understanding \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    understanding_data_file=prompts/understanding_counting_20.json \
    num_prompts=2 layers=all output_dir=token_prefill_pilot

See TOKEN_IMPORTANCE_README.md for output fields and limitations.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import NamedTuple
import uuid

import torch

SOURCE_COMMIT = "d30ae0d4f34fa9bce1121329db4c1747f30db737"
SCHEMA_VERSION = 2
SCORE_NAME = "token_decoder_update_l2_v2"
LOGGER = logging.getLogger(__name__)


def measure_and_save_examples(config, records, start_index, resources, output_dir, run):
    """Measure and save examples one at a time."""
    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as scores_file, \
         (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file, \
         (output_dir / "samples.jsonl").open("w", encoding="utf-8") as manifest:
        score_writer, summary_writer = create_csv_writers(scores_file, summary_file)

        for record_index, record in enumerate(records, start_index):
            payload = measure_one_example(config, record, record_index, resources, run)
            filename = f"sample_{record_index:06d}.pt"
            save_sample_files(
                output_dir / filename, payload, score_writer, summary_writer, manifest,
                resources.layer_ids,
            )
            manifest.flush()
            scores_file.flush()
            summary_file.flush()

            run["num_saved_samples"] += 1
            atomic_json(output_dir / "run.json", run)
            sample = payload["sample"]
            LOGGER.info(
                "Saved %s: %d tokens, %d layers, %.2fs",
                filename, sample["sequence_length"], len(resources.layer_ids),
                sample["collection_seconds"],
            )
            del payload


@torch.no_grad()
def measure_one_example(config, record, record_index, resources, run):
    """Encode an image/question pair, run prefill, and package its scores."""
    from models import omni_attn_mask_naive
    from models.misc import interpolate_pos_encoding

    # --- 5a. Seed and encode the image ---
    sample_started = time.perf_counter()
    sample_seed = run["base_seed"] + record_index
    seed_example(sample_seed, resources.device)
    # VAE image grid; converted to visual-token vectors in assemble_prefill().
    image_latents, original_size = encode_image(record["image_path"], resources, config)

    # --- 5b. Assemble the backbone input ---
    input_embeds, attention_mask, visual_positions, layout = assemble_prefill(
        resources.model, resources.tokenizer, resources.token_ids, image_latents,
        record["prompt"], config.dataset.preprocessing.max_seq_length,
        omni_attn_mask_naive, interpolate_pos_encoding,
        truncate_prompt=bool(config.get("truncate_prompt", False)),
    )

    # --- 5c. Run one prefill and score every token at the selected layers ---
    layer_data = collect_token_updates(
        resources.model.showo.model, input_embeds, attention_mask,
        resources.layer_ids, save_hidden_out=bool(config.get("save_hidden_out", False)),
    )

    # --- 5d. Package features, scores, and metadata ---
    if resources.device.type == "cuda":
        torch.cuda.synchronize(resources.device)
    return build_sample_payload(
        record, record_index, sample_seed, sample_started, original_size,
        image_latents, visual_positions, layout, layer_data, resources, run,
    )


# Helpers for steps 1-2: configuration, examples, and run setup

def print_help():
    """Show usage without importing the model-loading dependencies."""
    print(__doc__)
    print(
        "Options: layers=all (default) or layers=[13]; num_prompts=1; start_sample_index=0;\n"
        "output_dir=token_prefill_pilot; seed=42; device=cuda; dtype=bfloat16;\n"
        "save_hidden_out=false; truncate_prompt=false; vae_deterministic=false.\n"
        "batch_size must be 1. Output directory must be empty; no implicit resume."
    )


def load_config():
    """Load YAML settings and apply CLI key=value overrides."""
    from omegaconf import OmegaConf

    cli = OmegaConf.from_cli()
    if "config" not in cli:
        raise ValueError("Pass config=configs/showo2_1.5b_demo_432x432.yaml")
    config = OmegaConf.merge(OmegaConf.load(cli.config), cli)
    if int(config.get("batch_size", 1)) != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This pilot is single-process with batch_size=1; run with python")
    return config


def find_repository_root():
    """Find FlashU-Lab and make its local model imports available."""
    root = Path(__file__).resolve().parent.parent
    if not (root / "models" / "qwen2.py").is_file():
        root = Path.cwd()
    if not (root / "models" / "qwen2.py").is_file():
        raise FileNotFoundError("Run from FlashU-Lab root or place this file in its evaluation/ directory")
    sys.path.insert(0, str(root))
    return root


def load_examples(config):
    """Read the requested JSON slice and check its prompts and local image paths."""

    data_file = Path(config.get("understanding_data_file", "prompts/understanding_calibration.json"))
    records = json.loads(data_file.read_text(encoding="utf-8"))
    start_index = int(config.get("start_sample_index", 0))
    count = int(config.get("num_prompts", 1))

    if not isinstance(records, list) or count <= 0 or start_index < 0 or start_index + count > len(records):
        raise ValueError("Require a JSON list and a nonempty requested range within that list")
    
    selected = records[start_index:start_index + count]
    for index, record in enumerate(selected, start_index):
        if not isinstance(record, dict) or not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"Record {index} needs a nonempty prompt")
        if not isinstance(record.get("image_path"), str) or not Path(record["image_path"]).is_file():
            raise FileNotFoundError(f"Record {index}: image_path must identify an existing local image")
    return data_file, selected, start_index


def prepare_output_directory(config):
    """Create an empty results folder and send progress to its analysis.log."""
    output_dir = Path(config.get("output_dir", "token_importance_understanding"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty. Choose a new output_dir to avoid mixing runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "analysis.log", encoding="utf-8")],
        force=True,
    )
    return output_dir


def select_device_and_dtype(config):
    """Choose CPU/CUDA and the model's floating-point precision."""
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(config.get("device", default_device)))
    default_dtype = "bfloat16" if device.type == "cuda" else "float32"
    dtype_name = str(config.get("dtype", default_dtype))
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError("dtype must be bfloat16, float16 or float32")
    return device, dtypes[dtype_name]


# Helpers for step 4: load the model and image preprocessing once

class ModelResources(NamedTuple):
    """Loaded objects and settings reused per example."""

    model: torch.nn.Module
    tokenizer: object
    token_ids: dict
    vae: object
    image_transform: object
    layer_ids: list[int]
    device: torch.device
    dtype: torch.dtype
    resolution: int


def load_model_resources(config, device, dtype):
    """Load Show-o2, tokenizer, WanVAE, and image preprocessing."""
    from torchvision import transforms
    from models import Showo2Qwen2_5, WanVAE
    from models.misc import get_text_tokenizer
    from utils import path_to_llm_name, load_state_dict

    # --- 4a. Load tokenizer and special token IDs ---
    llm_path = config.model.showo.llm_model_path
    tokenizer, token_ids = get_text_tokenizer(
        llm_path, add_showo_tokens=True, return_showo_token_ids=True,
        llm_name=path_to_llm_name[llm_path],
    )

    # --- 4b. Load weights and select layers ---
    if config.model.showo.get("load_from_showo", False):
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path, use_safetensors=False
        )
    else:
        constructor = dict(config.model.showo)
        constructor["llm_vocab_size"] = len(tokenizer)
        model = Showo2Qwen2_5(**constructor)
        model.load_state_dict(load_state_dict(config.model_path))
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    core = model.showo.model
    if len(tokenizer) > core.embed_tokens.num_embeddings:
        raise ValueError("Tokenizer has more tokens than the pretrained embedding table")
    if core.config._attn_implementation == "flash_attention_2":
        raise ValueError("The source uses a 4D additive mask. Use its SDPA/eager checkpoint configuration")
    layer_ids = select_layers(config.get("layers"), len(core.layers))

    # --- 4c. Load VAE and Resize/CenterCrop/Normalize ---
    if config.model.vae_model.type != "wan21":
        raise ValueError("This source-compatible pilot supports WanVAE only")
    vae = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path, dtype=dtype, device=device
    )
    resolution = int(config.dataset.preprocessing.resolution)
    image_transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    return ModelResources(
        model, tokenizer, token_ids, vae, image_transform, layer_ids, device, dtype, resolution
    )


def select_layers(requested, num_layers):
    """Accept layers=[...], one index, or 'all'; default to every layer."""
    if requested is None:
        return list(range(num_layers))
    if isinstance(requested, str) and requested == "all":
        return list(range(num_layers))
    if isinstance(requested, int) and not isinstance(requested, bool):
        requested = [requested]
    result = list(requested)
    if not result or any(isinstance(index, bool) or not isinstance(index, int) for index in result):
        raise ValueError("layers must be an integer list, e.g. layers=[0,13,27], or layers=all")
    if len(set(result)) != len(result) or any(index < 0 or index >= num_layers for index in result):
        raise ValueError(f"layers must be unique zero-based indices in [0, {num_layers - 1}]")
    return sorted(result)


def seed_example(sample_seed, device):
    """Use the recorded seed for Python, NumPy, PyTorch, and CUDA sampling."""
    import numpy as np

    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(sample_seed)
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def encode_image(image_path, resources, config):
    """Return one image's VAE latents [1,C,H,W] and its original width/height."""
    from PIL import Image

    with Image.open(image_path) as image:
        original_size = list(image.size)
        pixels = resources.image_transform(image.convert("RGB")).unsqueeze(0)
        pixels = pixels.to(resources.device)
    # WanVAE expects a time dimension: [B,C,H,W] -> [B,C,1,H,W].
    image_latents = resources.vae.sample(
        pixels.unsqueeze(2), deterministic=bool(config.get("vae_deterministic", False))
    )
    return image_latents.squeeze(2).to(resources.dtype), original_size


# Helpers for step 5b: assemble the understanding prefill

@torch.no_grad()
def assemble_prefill(model, tokenizer, token_ids, image_latents, prompt, max_seq_length,
                     omni_mask_fn, interpolate_fn, truncate_prompt=False):
    """Return backbone embeddings, attention mask, visual positions, and metadata.
    
    Order: chat prefix -> BOI -> optional time -> visual tokens -> EOI
    -> question -> assistant prefix. Scores cover every position.
    """
    device = next(model.parameters()).device
    dtype = model.showo.model.embed_tokens.weight.dtype

    # --- A. Count visual patches in the latent grid ---
    if image_latents.ndim != 4 or image_latents.shape[0] != 1:
        raise ValueError("Expected one image latent tensor [1,C,H,W]")
    patch_size = int(model.config.patch_size)
    if image_latents.shape[-2] % patch_size or image_latents.shape[-1] % patch_size:
        raise ValueError("Latent grid must be divisible by the model's patch size")
    grid_height = image_latents.shape[-2] // patch_size
    grid_width = image_latents.shape[-1] // patch_size
    num_visual = grid_height * grid_width
    add_time = bool(model.config.add_time_embeds)

    # --- B. Embed text tokens and visual tokens ---
    # TEXT TOKENS: chat prefix, question, and assistant prefix (integer IDs).
    (prefix_ids, question_ids, assistant_ids,
     original_question_length, prefix_types) = prepare_chat_tokens(
        tokenizer, token_ids, prompt, max_seq_length, num_visual, add_time, truncate_prompt
    )
    suffix_ids = [token_ids["boi_id"], token_ids["eoi_id"]] + question_ids + assistant_ids
    embed_tokens = model.showo.model.embed_tokens
    # TEXT EMBEDDINGS: prefix and suffix, including BOI/EOI markers.
    prefix_embeds = embed_tokens(torch.tensor([prefix_ids], device=device, dtype=torch.long))
    suffix_embeds = embed_tokens(torch.tensor([suffix_ids], device=device, dtype=torch.long))
    # VISUAL TOKENS: fused image-patch vectors [1, N_visual, d].
    visual_embeds = fuse_image_embeddings(
        model, image_latents.to(device=device, dtype=dtype), grid_height, grid_width,
        interpolate_fn,
    )

    # --- C. Assemble the sequence ---
    parts = [prefix_embeds, suffix_embeds[:, :1]]  # prefix, BOI
    if add_time:
        time_embeds = model.time_embed(torch.ones(1, device=device), dtype)
        if hasattr(model, "time_embed_proj"):
            time_embeds = model.time_embed_proj(time_embeds)
        parts.append(time_embeds.unsqueeze(1))
    parts.extend([visual_embeds, suffix_embeds[:, 1:]])
    input_embeds = torch.cat(parts, dim=1)  # FINAL BACKBONE INPUT: [1, N_sequence, d].
    first_visual = len(prefix_ids) + 1 + int(add_time)
    # Indices of visual tokens in the combined backbone input.
    visual_positions = torch.arange(
        first_visual, first_visual + num_visual, dtype=torch.long, device=device
    )
    tokens = describe_prefill_tokens(
        tokenizer, token_ids, prefix_ids, prefix_types, question_ids, assistant_ids,
        num_visual, grid_width, add_time,
    )
    if len(tokens) != input_embeds.shape[1]:
        raise ValueError("Token metadata must match the complete backbone input")

    # --- D. Build the source mask; verify question visibility ---
    # With time enabled, the mask span includes EOI and excludes the time token.
    span_length = num_visual + int(add_time)
    modalities = torch.tensor([[[first_visual, span_length]]], device=device)
    attention_mask = omni_mask_fn(
        B=1, LEN=input_embeds.shape[1], modalities=modalities, device=device, inverted=True
    ).to(dtype)
    question_start = first_visual + num_visual + 1
    visual_can_see_question = bool(
        (attention_mask[0, 0, visual_positions, question_start:] == 0).any().item()
    )
    if visual_can_see_question:
        raise ValueError("Attention layout changed: visual tokens unexpectedly attend to question/suffix")

    metadata = {
        "sequence_length": input_embeds.shape[1], "num_visual_tokens": num_visual,
        "tokens": tokens,
        "token_type_counts": {
            kind: sum(token["token_type"] == kind for token in tokens)
            for kind in sorted({token["token_type"] for token in tokens})
        },
        "grid_height": grid_height, "grid_width": grid_width,
        "grid_coordinates_refer_to": "VAE patch grid of resized and center-cropped image",
        "add_time_embeds": add_time, "understanding_time_value": 1.0 if add_time else None,
        "question_token_ids": question_ids, "original_question_token_count": original_question_length,
        "used_question_token_count": len(question_ids),
        "prompt_truncated": original_question_length != len(question_ids),
        "question_start_position": question_start,
        "attention_span": [first_visual, span_length],
        "attention_layout": "source_understanding_obd_v1",
        "visual_tokens_can_attend_to_question": visual_can_see_question,
    }
    return input_embeds, attention_mask, visual_positions, metadata


def prepare_chat_tokens(tokenizer, token_ids, prompt, max_seq_length, num_visual,
                        add_time, truncate_prompt):
    """Tokenize the chat template and fit the question into the token budget."""
    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    # Offsets label the system's content without changing its tokenization.
    system_text = "You are a helpful assistant."
    system_header = "system\n"
    system = tokenizer(
        system_header + system_text + "<|im_end|>",
        add_special_tokens=False, return_offsets_mapping=True,
    )
    content_start = len(system_header)
    content_end = content_start + len(system_text)
    system_types = [
        "system_text" if end > content_start and start < content_end else "chat_format"
        for start, end in system["offset_mapping"]
    ]
    user_ids = encode("\n<|im_start|>user\n")
    prefix_ids = [token_ids["bos_id"]] + system["input_ids"] + user_ids
    prefix_types = ["chat_format"] + system_types + ["chat_format"] * len(user_ids)
    assistant_ids = encode("\n<|im_start|>assistant\n")
    question_ids = encode(prompt)
    original_question_length = len(question_ids)
    # Reserve image, time, BOI/EOI, and fixed chat positions.
    question_budget = (
        int(max_seq_length) - len(prefix_ids) - num_visual - int(add_time)
        - 2 - len(assistant_ids)
    )
    if question_budget <= 0:
        raise ValueError("max_seq_length cannot fit the image and chat template")
    if len(question_ids) > question_budget:
        if not truncate_prompt:
            raise ValueError(
                f"Question has {len(question_ids)} tokens but only {question_budget} fit; "
                "increase max_seq_length or explicitly set truncate_prompt=true"
            )
        LOGGER.warning("Truncating question from %d to %d tokens", len(question_ids), question_budget)
        question_ids = question_ids[:question_budget]
    return prefix_ids, question_ids, assistant_ids, original_question_length, prefix_types


def describe_prefill_tokens(tokenizer, token_ids, prefix_ids, prefix_types,
                           question_ids, assistant_ids, num_visual, grid_width, add_time):
    """Describe each sequence position; image patches and time have no vocabulary ID."""
    sequence_ids = prefix_ids + [token_ids["boi_id"]]
    token_types = prefix_types + ["image_boundary"]
    if add_time:
        sequence_ids += [None]
        token_types += ["time"]
    first_visual = len(sequence_ids)
    sequence_ids += [None] * num_visual + [token_ids["eoi_id"]] + question_ids + assistant_ids
    token_types += (
        ["visual"] * num_visual + ["image_boundary"]
        + ["question_text"] * len(question_ids) + ["chat_format"] * len(assistant_ids)
    )

    tokens = []
    for position, (token_id, token_type) in enumerate(zip(sequence_ids, token_types)):
        patch_index = position - first_visual if token_type == "visual" else None
        tokens.append({
            "sequence_position": position, "token_type": token_type, "token_id": token_id,
            "token_piece": tokenizer.convert_ids_to_tokens(token_id) if token_id is not None else None,
            "patch_index": patch_index,
            "patch_row": patch_index // grid_width if patch_index is not None else None,
            "patch_col": patch_index % grid_width if patch_index is not None else None,
        })
    return tokens


def fuse_image_embeddings(model, image_latents, grid_height, grid_width, interpolate_fn):
    """Fuse both image branches into visual-token vectors for the backbone."""
    num_visual = grid_height * grid_width
    understanding_embeds = model.image_embedder_und(image_latents)
    generation_embeds = model.image_embedder_gen(image_latents)
    if understanding_embeds.shape[1] != num_visual or generation_embeds.shape[1] != num_visual:
        raise ValueError("Visual embedding count does not match the latent patch grid")

    if model.position_embedding.weight.shape[0] == num_visual:
        image_positions = torch.arange(num_visual, device=image_latents.device).unsqueeze(0)
        understanding_embeds = understanding_embeds + model.position_embedding(image_positions)
    else:
        understanding_embeds = understanding_embeds + interpolate_fn(
            model.config.clip_latent_dim, model.position_embedding, grid_height, grid_width, 1
        )
    understanding_embeds = model.und_trans(understanding_embeds)["last_hidden_state"]
    return model.fusion_proj(torch.cat([understanding_embeds, generation_embeds], dim=-1))


# Helpers for step 5c: observe layer inputs/outputs and calculate the scores

@torch.no_grad()
def collect_token_updates(core, input_embeds, attention_mask, layer_ids=None,
                          save_hidden_out=False):
    """Run every decoder layer once on the prefill input; score all token positions."""
    # --- A. Validate inputs and choose which layers to record ---
    validate_collection_inputs(core, input_embeds, attention_mask)
    layer_ids = set(select_layers(layer_ids, len(core.layers)))
    layer_results = {}

    # --- B. Prepare positions and RoPE as in Qwen2Model.forward ---
    hidden_states = input_embeds
    cache_position = torch.arange(input_embeds.shape[1], device=input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = core.rotary_emb(hidden_states, position_ids)

    # --- C. PREFILL: pass the same sequence through all decoder layers ---
    for layer_index, layer in enumerate(core.layers):
        # 1. Capture the input vectors for every token in this example.
        if layer_index in layer_ids:
            hidden_in = hidden_states[0].detach().clone()

        # 2. Execute this layer; its output becomes the next layer's input.
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
        if hidden_states.shape != input_embeds.shape:
            raise ValueError(f"Token sequence shape changed at layer {layer_index}")

        # 3. Calculate and save I_und(layer, token) for the recorded layers.
        if layer_index in layer_ids:
            hidden_out = hidden_states[0].detach()
            labels = compute_update_labels(hidden_in, hidden_out)
            if not torch.isfinite(labels).all():
                raise ValueError(f"Non-finite features or labels at layer {layer_index}")
            layer_result = {
                "input_features": hidden_in.cpu(),
                "update_l2": labels.cpu(),
            }
            if save_hidden_out:
                layer_result["hidden_out"] = hidden_out.clone().cpu()
            layer_results[str(layer_index)] = layer_result

    # --- D. STOP: no final RMSNorm, vocabulary head, or answer decoding ---
    return layer_results


def validate_collection_inputs(core, input_embeds, attention_mask):
    """Reject inputs that would make token identity or the measured phase ambiguous."""
    if core.training:
        raise ValueError("Call model.eval() before collecting labels")
    if input_embeds.ndim != 3 or input_embeds.shape[0] != 1:
        raise ValueError("This pilot collects one example per forward: input shape [1,N,d]")
    sequence_length = input_embeds.shape[1]
    if sequence_length == 0:
        raise ValueError("The prefill sequence must contain at least one token")
    if attention_mask.shape[-2:] != (sequence_length, sequence_length):
        raise ValueError("Attention mask must match the complete prefill sequence")


def compute_update_labels(hidden_in, hidden_out):
    """Compare each token before (hidden_in) and after (hidden_out) a layer.
    
    Inputs: [N, d]; scores: [N]. N is the full sequence length; d is the vector width.
    In the equations, ^2 means squared.
    I_und(layer, token) is this L2 update magnitude.
    """
    # --- Equation 1: change = token_after_layer - token_before_layer ---
    # Cast before subtraction to avoid BF16/FP16 rounding.
    changes = hidden_out.float() - hidden_in.float()

    # --- Equation 2: score = sqrt(change[1]^2 + ... + change[d]^2) ---
    # ord=2: sqrt(sum of squares); dim=-1: components of each token vector.
    return torch.linalg.vector_norm(changes, ord=2, dim=-1)


# Helpers for steps 3, 5d, and 6: run metadata and saved dataset files

def create_run_metadata(config, repo_root, data_file, count, device, dtype, base_seed):
    """Describe the score, input dataset, settings, and source code in run.json."""
    from omegaconf import OmegaConf

    versions = {}
    for name in ["torch", "transformers", "torchvision", "omegaconf", "timm", "accelerate"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    provenance_files = [
        "evaluation/calculate_obd_cache_understanding.py", "models/qwen2.py",
        "models/modeling_showo2_qwen2_5.py", "models/omni_attention.py", "models/misc.py",
    ]
    source_hashes = {
        name: file_sha256(repo_root / name)
        for name in provenance_files if (repo_root / name).is_file()
    }
    return {
        "schema_version": SCHEMA_VERSION, "run_id": str(uuid.uuid4()), "status": "running",
        "score_name": SCORE_NAME,
        "label_definition": "I_und(layer, token) = L2(float32(block_out)-float32(block_in)) per prefill token",
        "label_status": "candidate_local_update_magnitude_not_validated_removal_damage",
        "token_scope": "all_prefill_positions",
        "task": "understanding", "phase": "prefill", "decode_steps": 0,
        "based_on_commit": SOURCE_COMMIT, "actual_repo_commit": repo_commit(repo_root),
        "script_sha256": file_sha256(__file__), "dataset_path": str(data_file.resolve()),
        "dataset_sha256": file_sha256(data_file), "source_sha256": source_hashes,
        "config": OmegaConf.to_container(config, resolve=True), "versions": versions,
        "device": str(device), "dtype": str(dtype).removeprefix("torch."), "base_seed": base_seed,
        "num_requested_samples": count, "num_saved_samples": 0,
        "question_dependence": "Visual tokens precede and cannot attend to the question in the preserved mask",
        "split_guidance": "Group future train/validation/test splits by image_sha256; do not split token rows randomly",
    }


def record_model_metadata(run, resources, config):
    """Add the loaded model dimensions, selected layers, and VAE identity."""
    core = resources.model.showo.model
    device = resources.device
    run.update(
        selected_layers=resources.layer_ids,
        hidden_size=core.config.hidden_size,
        num_decoder_layers=len(core.layers),
        model_config=dict(resources.model.config),
        backbone_config=core.config.to_dict(),
        vae_sha256=file_sha256(config.model.vae_model.pretrained_model_path),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )
    LOGGER.info("Layers %s; score %s; one dense prefill per sample", resources.layer_ids, SCORE_NAME)
    LOGGER.info("Preserving original understanding attention layout; scoring all prefill positions")
    LOGGER.info("These visual updates cannot depend on the later question text")


def build_sample_payload(record, record_index, sample_seed, sample_started, original_size,
                         image_latents, visual_positions, layout, layer_data, resources, run):
    """Package the aligned token rows and metadata for one sample_XXXXXX.pt file."""
    image_path = Path(record["image_path"])
    device = resources.device
    layout = layout.copy()
    tokens = layout.pop("tokens")
    sample = {
        "sample_id": f"{run['dataset_sha256'][:12]}:{record_index}",
        "record_index": record_index,
        "image_path": str(image_path.resolve()), "image_sha256": file_sha256(image_path),
        "original_image_size_wh": original_size, "processed_resolution": resources.resolution,
        "source_record": record, "seed": sample_seed, "task": "understanding", "phase": "prefill",
        "step_index": 0, "step_kind": "prefill", "denoising_timestep": None,
        **layout,
        "collection_seconds": time.perf_counter() - sample_started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    # Matrix rows follow layer_indices; columns follow tokens and token_positions.
    layer_indices = sorted(int(index) for index in layer_data)
    update_l2_matrix = torch.stack([
        layer_data[str(index)]["update_l2"] for index in layer_indices
    ])
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run["run_id"], "score_name": SCORE_NAME,
        "sample": sample, "visual_token_positions": visual_positions.cpu(),
        "token_positions": torch.arange(layout["sequence_length"], dtype=torch.long),
        "tokens": tokens, "layer_indices": layer_indices, "update_l2_matrix": update_l2_matrix,
        "image_latents": image_latents.detach().cpu(), "layers": layer_data,
    }


def create_csv_writers(scores_file, summary_file):
    """Write token-score and layer-summary CSV headers."""
    score_fields = [
        "sample_id", "record_index", "layer", "sequence_position", "token_type",
        "token_id", "token_piece", "patch_index", "patch_row", "patch_col", "update_l2",
    ]
    summary_fields = ["sample_id", "layer", "num_tokens", "num_visual_tokens", "min", "mean", "median", "max"]
    score_writer = csv.DictWriter(scores_file, fieldnames=score_fields)
    summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
    score_writer.writeheader()
    summary_writer.writeheader()
    return score_writer, summary_writer


def save_sample_files(path, payload, score_writer, summary_writer, manifest, layer_ids):
    """Save tensors, append CSV rows, and link the sample in samples.jsonl."""
    atomic_torch_save(path, payload)
    sample = payload["sample"]
    for layer_index, layer_result in payload["layers"].items():
        labels = layer_result["update_l2"]
        if len(payload["tokens"]) != len(labels):
            raise ValueError("Every saved token must have one score per layer")
        for token, label in zip(payload["tokens"], labels.tolist()):
            score_writer.writerow({
                "sample_id": sample["sample_id"], "record_index": sample["record_index"],
                "layer": int(layer_index), **token, "update_l2": label,
            })
        summary_writer.writerow({
            "sample_id": sample["sample_id"], "layer": int(layer_index),
            "num_tokens": len(labels), "num_visual_tokens": sample["num_visual_tokens"],
            "min": labels.min().item(), "mean": labels.mean().item(),
            "median": labels.median().item(), "max": labels.max().item(),
        })
    manifest.write(
        json.dumps({**sample, "file": path.name, "layers": layer_ids}, ensure_ascii=False) + "\n"
    )


def file_sha256(path):
    """Fingerprint a dataset, image, checkpoint, or source file for reproducibility."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    """Write JSON atomically using a temporary file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path, payload):
    """Save tensors atomically using a temporary file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def repo_commit(root):
    """Read the current commit for run metadata, when Git is available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# START HERE: the complete experiment, in execution order

def main():
    """Run the six experiment steps."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        return

    # --- 1. Load configuration and examples ---
    config = load_config()
    repo_root = find_repository_root()
    data_file, records, start_index = load_examples(config)

    # --- 2. Set output, logging, device, precision, and seed ---
    output_dir = prepare_output_directory(config)
    device, dtype = select_device_and_dtype(config)
    base_seed = int(config.get("seed", 42))
    if base_seed < 0 or base_seed + start_index + len(records) > 2**32 - 1:
        raise ValueError("seed plus record index must fit in an unsigned 32-bit integer")

    # --- 3. Record run metadata ---
    run = create_run_metadata(
        config, repo_root, data_file, len(records), device, dtype, base_seed
    )
    atomic_json(output_dir / "run.json", run)
    started = time.perf_counter()

    try:
        # --- 4. Load models and tokenizer ---
        resources = load_model_resources(config, device, dtype)
        record_model_metadata(run, resources, config)
        atomic_json(output_dir / "run.json", run)

        # --- 5. Run prefill and save all token scores ---
        measure_and_save_examples(
            config, records, start_index, resources, output_dir, run
        )

        # --- 6. Save final run status ---
        run["status"] = "complete"
    except Exception as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output_dir / "run.json", run)



# Entry point: run main() after defining the helpers.
if __name__ == "__main__":
    main()
