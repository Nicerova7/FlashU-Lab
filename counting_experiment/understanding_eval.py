"""Scorers, FFN token routing, and counting evaluation used by the Colab notebook."""

import hashlib
import json
import math
import re
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch import nn


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coco_ids(record):
    ids = set()
    if record.get("coco_image_id") is not None:
        ids.add(int(record["coco_image_id"]))
    for key in ("image_path", "source_url"):
        ids.update(int(value) for value in re.findall(
            r"(?<!\d)(\d{12})(?!\d)", str(record.get(key, ""))))
    return ids


def load_test_cases(test_dir, expected_images=40, image_dir=None):
    """Read reference answers from the repository and JPEGs from Drive."""
    test_dir = Path(test_dir)
    image_dir = Path(image_dir) if image_dir is not None else test_dir / "images"
    rows = json.loads((test_dir / "understanding.json").read_text(encoding="utf-8"))
    assert len(rows) == expected_images, f"Expected {expected_images} test images."
    assert all(row["review_status"] == "verified" for row in rows), "Review the reference counts first."
    cases = []
    for row in rows:
        image = image_dir / Path(row["image_path"]).name
        cases.append(dict(row, resolved_image=str(image), image_sha256=sha256_file(image)))
    return cases


def check_held_out(cases, used_records):
    old_ids, old_hashes = set(), set()
    for row in used_records:
        old_ids.update(coco_ids(row.get("source_record", {})))
        if row.get("image_sha256"):
            old_hashes.add(row["image_sha256"])
    overlaps = [x["id"] for x in cases if coco_ids(x) & old_ids
                or x["image_sha256"] in old_hashes]
    if overlaps:
        raise ValueError(f"Test images overlap the checkpoint's U collection: {overlaps}")
    return dict(checked_u_collection_images=len(used_records), test_images=len(cases),
                overlap_images=0, compared_by=["COCO image ID", "image SHA-256"])


def read_checkpoint_u_records(saved, u_root):
    """Inspect only U records named in the checkpoint's original collection."""
    marker = "/gradient_update_understanding/"
    paths = [Path(u_root) / str(stamp[0]).replace("\\", "/").split(marker, 1)[1]
             for stamp in saved["settings"]["files"]
             if marker in str(stamp[0]).replace("\\", "/") and str(stamp[0]).endswith(".pt")]
    assert paths, "Checkpoint settings contain no U collection records."
    records = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        sample = payload["sample"]
        records.append(dict(source_record=sample["source_record"],
                            image_sha256=sample["image_sha256"]))
        if len(records) == 1:
            features = payload["layers"]["0"]["input_features"]
            width = features.shape[1]
            dtype = str(features.dtype).removeprefix("torch.")
            del features
        del payload
    return records, width, dtype


class TokenMLP(nn.Module):
    """Score each token using its hidden state and layer index."""

    def __init__(self, feature_dim, hidden_width=128, num_layers=28):
        super().__init__()
        self.feature_dim, self.num_layers = feature_dim, num_layers
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1 + num_layers, hidden_width),
            nn.GELU(), nn.Linear(hidden_width, 1),
        )

    def forward(self, hidden, layer):
        hidden = hidden.float()
        rms = hidden.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        layer_code = nn.functional.one_hot(layer.long(), self.num_layers).float()
        return self.net(torch.cat([hidden / rms, rms.log(), layer_code], 1)).squeeze(1)


def load_scorers(weights, seed, device="cuda"):
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    settings = saved["settings"]
    scorers = {}
    for name in ("f", "u"):
        scorer = TokenMLP(settings["feature_dim"], settings["train"]["hidden_width"])
        scorer.load_state_dict(saved["models"][seed][name])
        scorers[name] = scorer.to(device).eval().requires_grad_(False)
    return scorers, saved


class UnderstandingFFNRouter:
    """Select image-token FFN rows for one prefill; attention stays dense."""

    def __init__(self, core, scorers, mode, eligible_mask, keep_fraction=.90):
        self.core, self.scorers, self.mode = core, scorers, mode
        self.keep_fraction = keep_fraction
        device = next(core.parameters()).device
        mask = eligible_mask[0].to(device)
        self.image_rows = mask.nonzero(as_tuple=True)[0]
        self.text_rows = (~mask).nonzero(as_tuple=True)[0]
        self.keep = math.ceil(keep_fraction * len(self.image_rows))
        self.stats = dict(calls=0, rows_total=0, rows_computed=0)
        self.by_layer, self.hooks, self.originals, self.selected = {}, [], [], {}

    def __enter__(self):
        self.core._depth_router_active = self
        if self.keep_fraction == 1:
            return self
        for index, layer in enumerate(self.core.layers[:-1]):
            self.hooks.append(layer.register_forward_pre_hook(self.selector(index), with_kwargs=True))
            original = layer.mlp.forward
            self.originals.append((layer.mlp, original))
            layer.mlp.forward = self.sparse_ffn(index, original)
        return self

    def selector(self, index):
        @torch.no_grad()
        def select(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            features = hidden[0].index_select(0, self.image_rows)
            layers = torch.full((len(self.image_rows),), index, device=hidden.device, dtype=torch.long)
            scores = self.scorers["f"](features, layers)
            if self.mode == "specialized":
                scores = scores + self.scorers["u"](features, layers)
            order = torch.argsort(scores, descending=True, stable=True)[:self.keep]
            self.selected[index] = torch.cat([self.text_rows, self.image_rows[order]])
        return select

    def sparse_ffn(self, index, original):
        def forward(hidden):
            rows = self.selected.pop(index)
            flat = hidden.reshape(-1, hidden.shape[-1])
            result = torch.zeros_like(flat)
            result.index_copy_(0, rows, original(flat.index_select(0, rows)))
            counts = dict(calls=1, rows_total=len(flat), rows_computed=len(rows))
            self.by_layer[index] = counts
            for key, value in counts.items():
                self.stats[key] += value
            return result.reshape_as(hidden)
        return forward

    def __exit__(self, *exc):
        for module, original in self.originals:
            module.forward = original
        for hook in self.hooks:
            hook.remove()
        self.originals.clear()
        self.hooks.clear()
        self.selected.clear()
        del self.core._depth_router_active


def parse_count(answer):
    """Accept one unambiguous integer, written as digits or an English number."""
    text = str(answer).casefold().strip()
    # Do not interpret part of a decimal/fraction/negative answer as a count.
    if re.search(r"-\s*\d|\d\s*[./]\s*\d", text):
        return None
    words = dict(zip(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        range(21)))
    if re.search(r"\b(thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b", text):
        return None
    values = []
    for token in re.findall(r"\b\d+\b|\b[a-z]+\b", text):
        if token.isdigit():
            values.append(int(token))
        elif token in words:
            values.append(words[token])
    return values[0] if values and len(set(values)) == 1 else None


@torch.no_grad()
def infer_understanding(case, image, resources, config, scorers, mode,
                        keep_fraction=.90, max_new_tokens=16):
    """Encode a preloaded RGB image, route prefill, then decode densely."""
    from transformers.cache_utils import DynamicCache
    from models import omni_attn_mask_naive
    from models.misc import interpolate_pos_encoding
    from evaluation.calculate_token_importance_understanding import assemble_prefill

    pixels = resources.image_transform(image).unsqueeze(0).to(resources.device)
    latents = resources.vae.sample(
        pixels.unsqueeze(2), deterministic=bool(config.get("vae_deterministic", False))
    ).squeeze(2).to(resources.dtype)
    embeds, attention, visual, _ = assemble_prefill(
        resources.model, resources.tokenizer, resources.token_ids, latents,
        case["prompt"], int(config.dataset.preprocessing.max_seq_length),
        omni_attn_mask_naive, interpolate_pos_encoding, truncate_prompt=False,
    )
    router = None
    if mode != "dense":
        eligible = torch.zeros((1, embeds.shape[1]), dtype=torch.bool)
        eligible[0, visual.detach().cpu()] = True
        router = UnderstandingFFNRouter(resources.model.showo.model, scorers,
                                        mode, eligible, keep_fraction)
    with router if router is not None else nullcontext():
        output = resources.model.showo(
            inputs_embeds=embeds, attention_mask=attention, past_key_values=DynamicCache(),
            use_cache=True, return_dict=True, num_logits_to_keep=1,
        )

    stop_ids = {value for value in [resources.tokenizer.eos_token_id,
                resources.token_ids.get("eos_id")] if value is not None}
    im_end = resources.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end >= 0 and im_end != resources.tokenizer.unk_token_id:
        stop_ids.add(im_end)
    generated, ended_with_stop = [], False
    decode_mask = torch.cat([attention[:, :, -1:, :],
                             torch.zeros_like(attention[:, :, -1:, :1])], dim=-1)
    for index in range(max_new_tokens):
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
        token_id = int(next_token.item())
        if token_id in stop_ids:
            ended_with_stop = True
            break
        generated.append(token_id)
        if index + 1 == max_new_tokens:
            break
        output = resources.model.showo(
            input_ids=next_token, attention_mask=decode_mask,
            past_key_values=output.past_key_values, use_cache=True,
            return_dict=True, num_logits_to_keep=1,
        )
        decode_mask = torch.cat([decode_mask, torch.zeros_like(decode_mask[..., :1])], dim=-1)
    result = dict(answer=resources.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                  token_ids=generated, generated_tokens=len(generated),
                  hit_token_limit=not ended_with_stop and len(generated) == max_new_tokens,
                  image_tokens=int(visual.numel()), prefill_tokens=int(embeds.shape[1]))
    if router is not None:
        result.update(router_stats=router.stats, router_by_layer=router.by_layer)
    return result


def timed_inference(case, image, resources, config, scorers, mode, seed,
                    keep_fraction=.90, max_new_tokens=16):
    from evaluation.calculate_token_importance_understanding import seed_example
    seed_example(seed, resources.device)
    torch.cuda.synchronize(resources.device)
    started = time.perf_counter()
    result = infer_understanding(case, image, resources, config, scorers, mode,
                                 keep_fraction, max_new_tokens)
    torch.cuda.synchronize(resources.device)
    elapsed = time.perf_counter() - started
    # Reference labels enter scoring only after model inference.
    predicted_count = parse_count(result["answer"])
    return dict(result, seconds=elapsed, predicted_count=predicted_count,
                correct=predicted_count == case["target_count"])


def summarize_evaluation(rows, modes=("dense", "shared", "specialized")):
    table = pd.DataFrame(rows)
    per_image = []
    for (case_id, mode), section in table.groupby(["id", "mode"], sort=False):
        if section["answer"].nunique(dropna=False) != 1:
            raise ValueError(f"Repeated answers differ for {case_id}/{mode}; inspect per_call.csv.")
        first = section.iloc[0]
        per_image.append(dict(id=case_id, mode=mode, target_count=int(first["target_count"]),
                              answer=first["answer"], predicted_count=first["predicted_count"],
                              correct=bool(first["correct"]), median_seconds=section["seconds"].median(),
                              generated_tokens=int(first["generated_tokens"]),
                              hit_token_limit=bool(first["hit_token_limit"]), repeats=len(section)))
    per_image = pd.DataFrame(per_image)
    expected_ids = set(per_image.loc[per_image["mode"] == "dense", "id"])
    if not expected_ids or any(set(per_image.loc[per_image["mode"] == mode, "id"]) != expected_ids
                               for mode in modes):
        raise ValueError("Every mode must cover the same test images.")
    summary = []
    for mode in modes:
        section = per_image[per_image["mode"] == mode]
        summary.append(dict(mode=mode, images=len(section), correct=int(section["correct"].sum()),
                            accuracy=section["correct"].mean(),
                            mean_seconds=section["median_seconds"].mean(),
                            median_seconds=section["median_seconds"].median(),
                            unparseable=int(section["predicted_count"].isna().sum()),
                            token_limit_hits=int(section["hit_token_limit"].sum()),
                            mean_answer_tokens=section["generated_tokens"].mean()))
    summary = pd.DataFrame(summary)
    dense = summary.loc[summary["mode"] == "dense"].iloc[0]
    summary["speedup_vs_dense"] = dense["mean_seconds"] / summary["mean_seconds"]
    summary["accuracy_change_pp"] = 100 * (summary["accuracy"] - dense["accuracy"])
    paired = per_image.pivot(index="id", columns="mode", values="correct")
    summary["dense_correct_routed_wrong"] = [int((paired["dense"] & ~paired[mode]).sum()) for mode in modes]
    summary["dense_wrong_routed_correct"] = [int((~paired["dense"] & paired[mode]).sum()) for mode in modes]
    return per_image, summary


def run_evaluation(cases, resources, config, scorers, results_dir, seed=1001,
                   keep_fraction=.90, max_new_tokens=16, repeats=3):
    """Warm each mode, then measure matched cases with rotating mode order."""
    modes = ("dense", "shared", "specialized")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=False)
    with Image.open(cases[0]["resolved_image"]) as source:
        image = source.convert("RGB")
        for mode in modes:
            result = timed_inference(cases[0], image, resources, config, scorers, mode,
                                     seed, keep_fraction, max_new_tokens)
            print("Warm-up", mode, "answer:", result["answer"])
    rows = []
    for index, case in enumerate(cases):
        with Image.open(case["resolved_image"]) as source:
            image = source.convert("RGB")
            for repeat in range(repeats):
                offset = (index + repeat) % len(modes)
                order = modes[offset:] + modes[:offset]
                for mode in order:
                    result = timed_inference(case, image, resources, config, scorers, mode,
                                             seed + index, keep_fraction, max_new_tokens)
                    row = {key: value for key, value in result.items()
                           if key not in ("token_ids", "router_stats", "router_by_layer")}
                    row.update({f"ffn_{key}": value for key, value in result.get("router_stats", {}).items()})
                    rows.append(dict(id=case["id"], mode=mode, repeat=repeat,
                                     target_count=case["target_count"], **row))
        pd.DataFrame(rows).to_csv(results_dir / "per_call.csv", index=False)
        print(f"{index + 1}/{len(cases)}: {case['id']}", flush=True)
    return rows
