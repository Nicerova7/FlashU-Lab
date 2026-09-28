"""Essential CPU checks for the standalone depth-router notebook."""

import ast
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import nbformat
import numpy as np
import torch


NOTEBOOK = Path(__file__).resolve().parents[1] / "evaluation/train_gradient_update_mlp_depth_router.ipynb"
CATEGORIES = ["counting", "spatial_relations", "attribute_combinations", "visual_state_traffic_lights"]


def notebook_functions():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    required = {"data_functions", "training_functions", "metric_functions", "routing_functions"}
    found = set()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        ast.parse(cell.source)
        tags = required.intersection(cell.metadata.get("tags", []))
        if tags:
            exec(compile(cell.source, f"notebook/{next(iter(tags))}", "exec"), namespace)
            found.update(tags)
    if found != required:
        raise AssertionError(f"Missing inline notebook definitions: {required - found}")
    return namespace


def write_dataset(root):
    u_root, g_root = root / "U", root / "G"
    for task, folder, count in [("U", u_root / name, 10) for name in CATEGORIES] + [("G", g_root, 40)]:
        folder.mkdir(parents=True)
        score_name = f"token_gradient_update_{'answer' if task == 'U' else 'flow'}_loss_v1"
        run = dict(run_id=folder.name, status="complete", score_name=score_name,
                   backbone_config={"hidden_size": 2}, dtype="bfloat16",
                   effective_sampling={"recorded_steps": [0], "timesteps": [0.5]})
        (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
        for index in range(count):
            source = dict(prompt=f"prompt {index}", expected_answer="one",
                          image_path=f"/{folder.name}/{index}.jpg")
            sample = dict(record_index=index, source_record=source,
                          image_sha256=f"{folder.name}/{index}")
            payload = dict(schema_version=1, run_id=folder.name, score_name=score_name,
                           sample=sample, layer_indices=list(range(28)))
            if task == "U":
                payload.update(tokens=["a", "b"], layers={str(layer): {
                    "input_features": torch.ones(2, 2, dtype=torch.bfloat16),
                    "gradient_update": torch.ones(2),
                } for layer in range(28)})
            else:
                payload.update(storage_format="gradient_update_generation_light_v1",
                               step_indices=[0], denoising_timesteps=[0.5],
                               initial_noise=torch.zeros(1, dtype=torch.bfloat16),
                               branches={"conditional": {
                                   "valid_token_mask": torch.tensor([True, False]),
                                   "gradient_update_tensor": torch.ones(28, 1, 2),
                               }})
            torch.save(payload, folder / f"sample_{index:06d}.pt")
    return u_root, g_root


class FakeSource:
    def __init__(self):
        self.calls = []

    def get(self, example, step):
        self.calls.append((example["split"], example["uid"]))
        hidden = torch.arange(1, 13).reshape(-1, 1).to(torch.bfloat16)
        hidden[10:] = float("nan")
        # Raw rankings must survive the logarithm's floor used only for training.
        raw = torch.arange(1, 13).float() * 1e-20
        raw[10:] = 999
        return {"valid": torch.tensor([True] * 10 + [False] * 2), "layers": {
            str(layer): {"input_features": hidden, "gradient_update": raw}
            for layer in (0, 14)
        }}


class LinearScore(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden, layer):
        return hidden[:, 0].float() * self.scale


class CountingScore(LinearScore):
    def __init__(self, scale):
        super().__init__(scale)
        self.calls = []

    def forward(self, hidden, layer):
        self.calls.append((hidden.clone(), layer.clone()))
        return super().forward(hidden, layer)


class CountingFFN(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1, bias=False)
        self.weight.data.fill_(1)
        self.rows = []

    def forward(self, hidden):
        self.rows.append(hidden.detach().reshape(-1, 1).clone())
        return super().forward(hidden)


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = CountingFFN()
        self.attention_rows = []

    def forward(self, hidden_states, **kwargs):
        self.input = hidden_states.detach().clone()
        self.attention_rows.append(hidden_states.shape[0] * hidden_states.shape[1])
        post_attention = -hidden_states
        self.output = post_attention + self.mlp(post_attention)
        return self.output


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyDecoder() for _ in range(28)])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden_states=hidden)
        return hidden


class NotebookTests(unittest.TestCase):
    def test_original_splits_and_no_leakage(self):
        discover = notebook_functions()["discover_data"]
        with tempfile.TemporaryDirectory() as temporary:
            u_root, g_root = write_dataset(Path(temporary))
            examples = discover(u_root, g_root, CATEGORIES)
            counts = Counter((x["task"], x["split"]) for x in examples)
            self.assertEqual(counts, {("U", "train"): 16, ("U", "val"): 4, ("U", "test"): 20,
                                      ("G", "train"): 8, ("G", "val"): 2, ("G", "test"): 30})
            for category, (val, test) in zip(CATEGORIES, [(1, 0), (3, 5), (0, 1), (5, 1)]):
                split = {x["index"]: x["split"] for x in examples if x["category"] == f"U/{category}"}
                self.assertEqual((split[val], split[test]), ("val", "test"))
            self.assertEqual([x["index"] for x in examples if x["task"] == "G" and x["split"] == "val"], [2, 3])
            self.assertEqual([x["index"] for x in examples if x["task"] == "G" and x["cohort"] == "original" and x["split"] == "test"], [1, 6])
            self.assertTrue(all(x["split"] == "test" for x in examples if x["cohort"] == "new"))
            for path, field, duplicate in [
                (u_root / "counting/sample_000006.pt", "image_sha256", "counting/2"),
                (g_root / "sample_000012.pt", "prompt", "  PROMPT   0 "),
            ]:
                payload = torch.load(path, weights_only=True)
                target = payload["sample"] if field == "image_sha256" else payload["sample"]["source_record"]
                original, target[field] = target[field], duplicate
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "[Ll]eakage"):
                    discover(u_root, g_root, CATEGORIES)
                target[field] = original
                torch.save(payload, path)

    def test_balanced_training_and_frozen_residual_target(self):
        ns = notebook_functions()
        banks = {task: {"hidden": torch.randn(8, 2).bfloat16(),
                         "layer": torch.tensor([0, 14] * 4), "group": torch.arange(8) // 2,
                         "score": torch.full((8,), value)} for task, value in [("U", 2.), ("G", 3.)]}
        samples, original_sample = [], ns["_sample_rows"]
        def record_sample(bank, count, generator):
            samples.append((float(bank["score"][0]), count))
            return original_sample(bank, count, generator)
        ns["_sample_rows"] = record_sample
        settings = dict(hidden_width=4, batch_size=8, max_epochs=1, updates=2)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                core, _ = ns["train_scorer"]("f", 42, banks, banks, 2, **settings)
                self.assertEqual(samples, [(2., 4), (3., 4)] * 2)
                before = {key: value.clone() for key, value in core.state_dict().items()}
                for task in ("u", "g"):
                    _, history = ns["train_scorer"](task, 42, banks, banks, 2, core=core, **settings)
                    self.assertTrue(np.isfinite(history[0]["val_loss"]))
                self.assertTrue(all(torch.equal(value, core.state_dict()[key]) for key, value in before.items()))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in core.parameters()))
                h, layer, raw = torch.ones(3, 2), torch.tensor([0, 14, 14]), torch.tensor([0., 2., 4.])
                actual = ns["_training_target"](raw, h, layer, core)
                torch.testing.assert_close(actual, raw.clamp_min(1e-12).log() - core(h, layer))
        finally:
            torch.set_num_threads(threads)

    def test_compact_training_full_token_evaluation_and_depth_gate(self):
        ns, source = notebook_functions(), FakeSource()
        examples = [dict(uid=f"{split}/{task}", task=task, category=task, split=split, cohort="new",
                         steps=[-1] if task == "U" else [0, 1], feature_dim=1,
                         dtype="bfloat16", n_valid_tokens=10)
                    for split in ("train", "val", "test") for task in ("U", "G")]
        with contextlib.redirect_stdout(io.StringIO()):
            train, val = ns["build_banks"](examples, source, tokens_per_group=3, layers=(0, 14))
        self.assertFalse(any(split == "test" for split, _ in source.calls))
        for banks in (train, val):
            for bank in banks.values():
                self.assertEqual(bank["hidden"].dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(bank["hidden"]).all())
                self.assertTrue((bank["score"] < 1e-18).all())
                self.assertTrue(torch.unique(bank["group"], return_counts=True)[1].eq(3).all())
        models = {42: {"f": LinearScore(-1), "u": LinearScore(2), "g": LinearScore(2)}}
        details = ns["evaluate"](examples, source, models, split="test", layers=(0, 14), progress=None)
        self.assertTrue(details["n_tokens"].eq(10).all())
        self.assertTrue(details["k"].eq(2).all())
        self.assertTrue(details["chance"].eq(.2).all())
        for mode, expected in [("Shared everywhere", 0.), ("Specialized everywhere", 1.)]:
            self.assertTrue(details.loc[details["mode"] == mode, "overlap"].eq(expected).all())
        depth = details[details["mode"] == "Depth-dependent"]
        np.testing.assert_array_equal(depth["overlap"], (depth["layer"] >= 14).astype(float))
        summary = ns["summarize"](details)
        self.assertTrue(summary["n_examples"].eq(1).all())
        self.assertTrue(summary.loc[summary["mode"] == "Depth-dependent", "mean_overlap"].eq(.5).all())

    def test_actual_ffn_skipping_preserves_attention_and_restores_model(self):
        Router = notebook_functions()["FFNRouter"]
        core = TinyBackbone().eval()
        scorers = {name: CountingScore(scale).eval() for name, scale in
                   [("f", 1), ("u", -2), ("g", -2)]}
        hidden = torch.arange(1, 23).float().reshape(2, 11, 1)
        eligible = torch.zeros(2, 11, dtype=torch.bool)
        eligible[0, :10] = True
        originals = [layer.mlp.forward for layer in core.layers]
        with torch.inference_mode(), Router(core, scorers, "U", "depth", eligible) as router:
            core(hidden)
            self.assertEqual(router.stats, dict(calls=27, rows_total=27 * 22,
                rows_computed=27 * 20, eligible_rows=270, selected_eligible=216))
            self.assertTrue(all(layer.attention_rows == [22] for layer in core.layers))
            self.assertTrue(all(len(layer.mlp.rows[0]) == 20 for layer in core.layers[:-1]))
            self.assertEqual(len(core.layers[-1].mlp.rows[0]), 22)
            # Router reads pre-attention positives; FFN receives post-attention negatives.
            torch.testing.assert_close(scorers["f"].calls[0][0], hidden[0, :10])
            expected = torch.cat([-hidden[0, 2:], -hidden[1]]).flatten().sort().values
            torch.testing.assert_close(core.layers[0].mlp.rows[0].flatten().sort().values, expected)
            torch.testing.assert_close(core.layers[0].output[0, :2], -hidden[0, :2])
            torch.testing.assert_close(core.layers[0].output[0, 2:], -2 * hidden[0, 2:])
            torch.testing.assert_close(core.layers[0].output[1], -2 * hidden[1])
            self.assertEqual([int(layer[0]) for _, layer in scorers["u"].calls], list(range(14, 27)))
            self.assertEqual(scorers["g"].calls, [])
        self.assertTrue(all(layer.mlp.forward == original for layer, original in zip(core.layers, originals)))
        self.assertTrue(all(not layer._forward_pre_hooks for layer in core.layers))
        calls = len(scorers["f"].calls)
        with torch.inference_mode():
            dense = core(hidden)
            with Router(core, scorers, "U", "dense", eligible) as router:
                torch.testing.assert_close(core(hidden), dense, rtol=0, atol=0)
                self.assertEqual(router.stats, dict(calls=27, rows_total=27 * 22,
                    rows_computed=27 * 22, eligible_rows=270, selected_eligible=270))
            self.assertEqual(len(scorers["f"].calls), calls)
            with Router(core, scorers, "U", "depth", eligible, keep_fraction=1) as router:
                torch.testing.assert_close(core(hidden), dense, rtol=0, atol=0)
                self.assertEqual(router.stats["calls"], 0)
            self.assertEqual(len(scorers["f"].calls), calls)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with Router(core, scorers, "U", "shared", eligible):
                raise RuntimeError("intentional")
        self.assertFalse(hasattr(core, "_depth_router_active"))
        self.assertTrue(all(layer.mlp.forward == original for layer, original in zip(core.layers, originals)))
        self.assertTrue(all(not layer._forward_pre_hooks for layer in core.layers))


if __name__ == "__main__":
    unittest.main()
