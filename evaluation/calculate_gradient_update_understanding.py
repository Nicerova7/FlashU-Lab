# coding=utf-8
"""Collect loss-aware gradient x update scores for understanding tokens.

Run from the repository root:
  python -m evaluation.calculate_gradient_update_understanding \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    understanding_data_file=prompts/understanding_counting_20.json \
    num_prompts=1 layers=all output_dir=gradient_update_understanding
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
import sys
import time
import uuid

import torch
import torch.nn.functional as F

from evaluation.calculate_token_importance_understanding import (
    assemble_prefill,
    atomic_json,
    atomic_torch_save,
    encode_image,
    file_sha256,
    find_repository_root,
    load_config,
    load_examples,
    load_model_resources,
    prepare_output_directory,
    repo_commit,
    seed_example,
    select_device_and_dtype,
)

SCHEMA_VERSION = 1
SCORE_NAME = "token_gradient_update_answer_loss_v1"
LOGGER = logging.getLogger(__name__)


def collect_scores(resources, input_embeds, attention_mask, labels, recorded_length):
    core = resources.model.showo.model
    input_embeds = input_embeds.detach().requires_grad_(True)
    hidden_states = input_embeds
    cache_position = torch.arange(input_embeds.shape[1], device=resources.device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = core.rotary_emb(hidden_states, position_ids)
    selected = set(resources.layer_ids)
    records = []

    for layer_index, layer in enumerate(core.layers):
        hidden_in = None
        if layer_index in selected:
            hidden_in = hidden_states[0, :recorded_length].detach().clone()
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

    final_hidden = core.norm(hidden_states)
    target_mask = labels[:, 1:] != -100
    prediction_hidden = final_hidden[:, :-1][target_mask]
    targets = labels[:, 1:][target_mask]
    logits = resources.model.showo.lm_head(prediction_hidden)
    loss = F.cross_entropy(logits.float(), targets)
    gradients = torch.autograd.grad(loss, [record[2] for record in records])

    layers = {}
    for (layer_index, hidden_in, hidden_out), gradient in zip(records, gradients):
        update = hidden_out[0, :recorded_length].detach().float() - hidden_in.float()
        score = (gradient[0, :recorded_length].detach().float() * update).sum(-1).abs()
        layers[str(layer_index)] = {
            "input_features": hidden_in.cpu(),
            "gradient_update": score.cpu(),
        }
    return layers, float(loss.detach().cpu())


def measure_one(record, record_index, config, resources, run):
    started = time.perf_counter()
    seed = run["base_seed"] + record_index
    seed_example(seed, resources.device)
    image_latents, original_size = encode_image(record["image_path"], resources, config)

    from models import omni_attn_mask_naive
    from models.misc import interpolate_pos_encoding

    answer = str(record.get("expected_answer", "")).strip()
    answer_ids = resources.tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError("Each record needs a tokenizable expected_answer")
    max_length = int(config.dataset.preprocessing.max_seq_length)
    input_embeds, _, visual_positions, layout = assemble_prefill(
        resources.model,
        resources.tokenizer,
        resources.token_ids,
        image_latents,
        record["prompt"],
        max_length - len(answer_ids),
        omni_attn_mask_naive,
        interpolate_pos_encoding,
        truncate_prompt=bool(config.get("truncate_prompt", False)),
    )
    recorded_length = input_embeds.shape[1]
    answer_tensor = torch.tensor([answer_ids], device=resources.device, dtype=torch.long)
    input_embeds = torch.cat(
        [input_embeds, resources.model.showo.model.embed_tokens(answer_tensor)], dim=1
    )
    first_visual, span_length = layout["attention_span"]
    modalities = torch.tensor([[[first_visual, span_length]]], device=resources.device)
    attention_mask = omni_attn_mask_naive(
        B=1,
        LEN=input_embeds.shape[1],
        modalities=modalities,
        device=resources.device,
        inverted=True,
    ).to(input_embeds.dtype)
    labels = torch.full(
        (1, input_embeds.shape[1]), -100, device=resources.device, dtype=torch.long
    )
    labels[0, recorded_length:] = answer_tensor[0]

    layers, loss = collect_scores(
        resources, input_embeds, attention_mask, labels, recorded_length
    )
    tokens = layout.pop("tokens")
    layer_indices = sorted(map(int, layers))
    sample = {
        "sample_id": f"{run['dataset_sha256'][:12]}:{record_index}",
        "record_index": record_index,
        "image_path": str(Path(record["image_path"]).resolve()),
        "image_sha256": file_sha256(record["image_path"]),
        "source_record": record,
        "seed": seed,
        "task": "understanding",
        "phase": "answer_teacher_forcing",
        "task_loss": loss,
        "answer_token_ids": answer_ids,
        "original_image_size_wh": original_size,
        "collection_seconds": time.perf_counter() - started,
        **layout,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "score_name": SCORE_NAME,
        "run_id": run["run_id"],
        "sample": sample,
        "tokens": tokens,
        "token_positions": torch.arange(recorded_length),
        "visual_token_positions": visual_positions.cpu(),
        "layer_indices": layer_indices,
        "gradient_update_matrix": torch.stack(
            [layers[str(index)]["gradient_update"] for index in layer_indices]
        ),
        "image_latents": image_latents.detach().cpu(),
        "layers": layers,
    }


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return

    config = load_config()
    root = find_repository_root()
    data_file, records, start = load_examples(config)
    if "output_dir" not in config:
        config.output_dir = "gradient_update_understanding"
    output_dir = prepare_output_directory(config)
    device, dtype = select_device_and_dtype(config)
    run = {
        "schema_version": SCHEMA_VERSION,
        "score_name": SCORE_NAME,
        "run_id": str(uuid.uuid4()),
        "status": "running",
        "task": "understanding",
        "label_definition": "abs(dL/dh_out dot (h_out-h_in))",
        "loss": "cross_entropy_on_expected_answer_tokens",
        "base_seed": int(config.get("seed", 42)),
        "dataset_path": str(data_file.resolve()),
        "dataset_sha256": file_sha256(data_file),
        "actual_repo_commit": repo_commit(root),
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

        fields = ["record_index", "layer", "num_tokens", "min", "mean", "median", "max"]
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record_index, record in enumerate(records, start):
                payload = measure_one(record, record_index, config, resources, run)
                atomic_torch_save(output_dir / f"sample_{record_index:06d}.pt", payload)
                for layer, data in payload["layers"].items():
                    score = data["gradient_update"]
                    writer.writerow({
                        "record_index": record_index,
                        "layer": layer,
                        "num_tokens": len(score),
                        "min": score.min().item(),
                        "mean": score.mean().item(),
                        "median": score.median().item(),
                        "max": score.max().item(),
                    })
                stream.flush()
                run["num_saved_samples"] += 1
                atomic_json(output_dir / "run.json", run)
                LOGGER.info("Saved understanding sample %d", record_index)
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
