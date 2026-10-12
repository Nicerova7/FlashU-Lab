"""CPU checks for the fresh-image understanding evaluation notebook."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "counting_experiment/understanding_counting_eval.ipynb"
spec = importlib.util.spec_from_file_location(
    "depth_eval_fixtures", ROOT / "tests/test_depth_router_notebook.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def notebook_functions():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    spec = importlib.util.spec_from_file_location(
        "understanding_eval", ROOT / "counting_experiment/understanding_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


class UnderstandingEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = notebook_functions()

    def run_router(self, mode, keep_fraction=.90):
        backbone = fixtures.TinyBackbone().eval()
        scorers = {name: fixtures.CountingScore(scale).eval()
                   for name, scale in [("f", 1), ("u", -2)]}
        hidden = torch.arange(1, 13).float().reshape(1, 12, 1)
        eligible = torch.tensor([[False] + [True] * 10 + [False]])
        router = self.ns["UnderstandingFFNRouter"](
            backbone, scorers, mode, eligible, keep_fraction)
        with torch.no_grad(), router:
            result = backbone(hidden)
        return result, backbone, scorers, router

    def test_routing_keeps_attention_text_and_final_layer_dense(self):
        _, core, scorers, router = self.run_router("specialized")
        self.assertEqual(len(scorers["f"].calls), 27)
        self.assertEqual(len(scorers["u"].calls), 27)
        self.assertEqual(router.stats["rows_total"], 27 * 12)
        self.assertEqual(router.stats["rows_computed"], 27 * 11)
        self.assertTrue(all(layer.attention_rows == [12] for layer in core.layers))
        self.assertEqual(core.layers[-1].mlp.rows[0].shape[0], 12)
        # u reverses the ranking: first image row is kept, last image row bypassed.
        actual = core.layers[0].mlp.rows[0].flatten().sort().values
        expected = -torch.tensor([12., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.])
        torch.testing.assert_close(actual, expected)
        self.assertFalse(hasattr(core, "_depth_router_active"))
        self.assertTrue(all(not layer._forward_pre_hooks for layer in core.layers))

    def test_shared_does_not_call_u_and_full_keep_is_unmodified(self):
        _, _, scorers, _ = self.run_router("shared")
        self.assertEqual(scorers["u"].calls, [])
        result, core, scorers, _ = self.run_router("specialized", keep_fraction=1)
        self.assertEqual(scorers["f"].calls, [])
        baseline = fixtures.TinyBackbone().eval()
        with torch.no_grad():
            expected = baseline(torch.arange(1, 13).float().reshape(1, 12, 1))
        torch.testing.assert_close(result, expected)
        self.assertTrue(all(layer.mlp.rows[0].shape[0] == 12 for layer in core.layers))

    def test_hooks_are_restored_after_inference_failure(self):
        core = fixtures.TinyBackbone().eval()
        scorers = {"f": fixtures.LinearScore(1).eval()}
        def fail(hidden):
            raise RuntimeError("inference failed")
        core.layers[3].mlp.forward = fail
        original = [layer.mlp.forward for layer in core.layers]
        router = self.ns["UnderstandingFFNRouter"](
            core, scorers, "shared", torch.ones(1, 10, dtype=torch.bool), .90)
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with router:
                core(torch.arange(10.).reshape(1, 10, 1))
        self.assertEqual([layer.mlp.forward for layer in core.layers], original)
        self.assertTrue(all(not layer._forward_pre_hooks for layer in core.layers))
        self.assertFalse(hasattr(core, "_depth_router_active"))

    def test_mlp_is_compatible_with_existing_checkpoint_architecture(self):
        previous = nbformat.read(ROOT / "counting_experiment/router_training_and_overlap.ipynb", as_version=4)
        cell = next(c for c in previous.cells if "training_functions" in c.metadata.get("tags", []))
        node = next(x for x in ast.parse(cell.source).body
                    if isinstance(x, ast.ClassDef) and x.name == "TokenMLP")
        old_ns = {"torch": torch, "nn": torch.nn}
        exec(ast.get_source_segment(cell.source, node), old_ns)
        old = old_ns["TokenMLP"](2, 4).eval()
        new = self.ns["TokenMLP"](2, 4).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        hidden, layer = torch.randn(4, 2), torch.tensor([0, 14, 24, 26])
        torch.testing.assert_close(old(hidden, layer), new(hidden, layer), rtol=0, atol=0)

    def test_count_parser_accepts_clear_answers_and_rejects_ambiguity(self):
        parse = self.ns["parse_count"]
        for answer, expected in [("two", 2), ("There are 3 dogs.", 3), ("Two (2).", 2),
                                 ("zero", 0), ("No visible objects: 0", 0), ("12", 12)]:
            self.assertEqual(parse(answer), expected, answer)
        for answer in ["one or two", "three cats and one dog", "unknown", "someone", "-1",
                       "2.0", "1/2", "twenty one", "one hundred", ""]:
            self.assertIsNone(parse(answer), answer)

    def test_count_parser_reads_totals_before_object_descriptions(self):
        parse = self.ns["parse_count"]
        for answer, expected in [
            ("There are two cats in the image. One cat is lying on the floor,", 2),
            ("There are two dogs in the image. One dog is black and white, and", 2),
            ("There are two zebras in the image. One zebra is standing with its", 2),
            ("There are three bottles in the image. Two of them are dark brown glass bottles", 3),
            ("There are four individuals visible in the image. One person is standing and appears to", 4),
            ("There are two birds in the image. One bird is perched on a stone", 2),
        ]:
            self.assertEqual(parse(answer), expected, answer)
        for answer in ["Maybe three giraffes.", "There are two or three cats.",
                       "There are three cats. Actually, there are two.",
                       "There are three cats. There are two cats.",
                       "There are two cats and three dogs.", "There are not three cats."]:
            self.assertIsNone(parse(answer), answer)

    def test_manifest_review_and_holdout_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images/000000123456.jpg"
            image.parent.mkdir()
            image.write_bytes(b"test-image")
            manifest = [{"id": "U_0000", "review_status": "verified", "target_count": 2,
                         "prompt": "How many cats?", "image_path": "images/000000123456.jpg",
                         "coco_image_id": 123456}]
            (root / "understanding.json").write_text(json.dumps(manifest), encoding="utf-8")
            cases = self.ns["load_test_cases"](root, expected_images=1)
            check = self.ns["check_held_out"]
            self.assertEqual(check(cases, [{"source_record": {"coco_image_id": 999}}])["overlap_images"], 0)
            for used in [{"source_record": {"image_path": "/old/000000123456.jpg"}},
                         {"source_record": {}, "image_sha256": cases[0]["image_sha256"]}]:
                with self.assertRaisesRegex(ValueError, "overlap"):
                    check(cases, [used])
            manifest[0]["review_status"] = "pending"
            (root / "understanding.json").write_text(json.dumps(manifest), encoding="utf-8")
            pending = self.ns["load_test_cases"](root, 1)
            self.assertEqual(pending[0]["review_status"], "pending")
            with self.assertRaisesRegex(AssertionError, "Review the cropped reference counts"):
                self.ns["run_evaluation"](pending, None, None, {}, root / "results")
            self.assertFalse((root / "results").exists())

    def test_images_can_live_in_drive_separate_from_repository_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir, image_dir = root / "repo/test", root / "drive/counting_test_images"
            manifest_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            (image_dir / "000000123456.jpg").write_bytes(b"test-image-in-drive")
            row = {"id": "U_0000", "review_status": "verified", "target_count": 2,
                   "prompt": "How many cats?", "image_path": "images/000000123456.jpg"}
            manifest = manifest_dir / "understanding.json"
            manifest.write_text(json.dumps([row]), encoding="utf-8")
            load = self.ns["load_test_cases"]
            cases = load(manifest_dir, expected_images=1, image_dir=image_dir)
            self.assertEqual(Path(cases[0]["resolved_image"]), image_dir / "000000123456.jpg")
            self.assertFalse((manifest_dir / "images").exists())
            row["image_path"] = "images/000000999999.jpg"
            manifest.write_text(json.dumps([row]), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load(manifest_dir, expected_images=1, image_dir=image_dir)

    def test_reads_provenance_from_checkpoint_u_files_without_g_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            u_root = Path(temporary)
            stamps = []
            for category in fixtures.CATEGORIES:
                folder = u_root / category
                folder.mkdir()
                for index in range(10):
                    path = folder / f"sample_{index:06d}.pt"
                    torch.save(dict(score_name="token_gradient_update_answer_loss_v1",
                                    sample={"source_record": {"image_path": f"{index}.jpg"},
                                            "image_sha256": f"{category}/{index}"},
                                    layers={"0": {"input_features": torch.ones(2, 3).bfloat16()}},
                                    image_latents=torch.zeros(1, 16, 27, 27)), path)
                    stamps.append((f"/content/drive/MyDrive/intership/gradient_update_understanding/{category}/{path.name}",
                                   path.stat().st_size, 0))
            stamps.append(("/unavailable/gradient_update_generation/sample_000000.pt", 999, 0))
            checkpoint = {"settings": {"files": stamps}}
            rows, width, dtype = self.ns["read_checkpoint_u_records"](checkpoint, u_root)
            self.assertEqual(len(rows), 40)
            self.assertEqual((width, dtype), (3, "bfloat16"))
            self.assertEqual(len({row["image_sha256"] for row in rows}), 40)

    def test_summary_counts_images_once_and_exposes_paired_errors(self):
        rows = []
        for case_id in ["a", "b"]:
            for mode, seconds in [("dense", 2.), ("shared", 1.), ("specialized", 1.5)]:
                correct = (case_id == "a") if mode != "specialized" else case_id == "b"
                for repeat in range(3):
                    rows.append(dict(id=case_id, mode=mode, repeat=repeat, target_count=2,
                                     answer="two" if correct else "unknown",
                                     predicted_count=2 if correct else None, correct=correct,
                                     seconds=seconds, generated_tokens=1, hit_token_limit=False))
        images, summary = self.ns["summarize_evaluation"](rows)
        self.assertEqual(len(images), 6)
        self.assertEqual(summary["images"].tolist(), [2, 2, 2])
        self.assertEqual(summary["accuracy"].tolist(), [.5, .5, .5])
        self.assertEqual(summary["unparseable"].tolist(), [1, 1, 1])
        self.assertEqual(summary["dense_correct_routed_wrong"].tolist(), [0, 0, 1])
        self.assertEqual(summary["dense_wrong_routed_correct"].tolist(), [0, 0, 1])
        self.assertEqual(summary.loc[summary["mode"] == "shared", "speedup_vs_dense"].iloc[0], 2.)
        rows[0]["answer"] = "changed"
        with self.assertRaisesRegex(ValueError, "Repeated answers differ"):
            self.ns["summarize_evaluation"](rows)

    def test_synchronized_timer_excludes_reference_scoring(self):
        from types import SimpleNamespace
        events = []
        ns = self.ns
        resources = SimpleNamespace(device=torch.device("cpu"))
        def fake_inference(*args):
            events.append("inference")
            return {"answer": "two"}
        def clock():
            events.append("clock")
            return {1: 10., 2: 11.5}[events.count("clock")]
        with patch.dict(ns, {"infer_understanding": fake_inference,
                             "parse_count": lambda _: events.append("score") or 2}), \
             patch.object(ns["time"], "perf_counter", clock), \
             patch.object(torch.cuda, "synchronize", lambda _: events.append("sync")), \
             patch("evaluation.calculate_token_importance_understanding.seed_example",
                   lambda *args: events.append("seed")):
            result = ns["timed_inference"]({"target_count": 2}, None, resources, None, {}, "dense", 123)
        self.assertEqual(events, ["seed", "sync", "clock", "inference", "sync", "clock", "score"])
        self.assertEqual(result["seconds"], 1.5)
        self.assertTrue(result["correct"])

    def test_evaluation_excludes_warmup_and_matches_seeds_across_modes(self):
        from PIL import Image
        calls = []

        def inference(case, image, resources, config, scorers, mode, seed, *args):
            calls.append((case["id"], mode, seed))
            return dict(answer="two", predicted_count=2, correct=True, seconds=1.,
                        generated_tokens=1, hit_token_limit=False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "test.jpg"
            Image.new("RGB", (2, 2)).save(image)
            cases = [dict(id=name, target_count=2, resolved_image=str(image), review_status="verified")
                     for name in ("a", "b")]
            with patch.dict(self.ns, {"timed_inference": inference}):
                rows = self.ns["run_evaluation"](cases, None, None, {}, root / "results")
            self.assertEqual(len(calls), 21)
            self.assertEqual(len(rows), 18)
            self.assertEqual(len(pd.read_csv(root / "results/per_call.csv")), 18)
            self.assertEqual(calls[:3], [("a", mode, 1001) for mode in ("dense", "shared", "specialized")])
            measured = calls[3:]
            self.assertEqual({seed for name, _, seed in measured if name == "a"}, {1001})
            self.assertEqual({seed for name, _, seed in measured if name == "b"}, {1002})
            self.assertEqual([mode for _, mode, _ in measured[:9]],
                             ["dense", "shared", "specialized", "shared", "specialized", "dense",
                              "specialized", "dense", "shared"])
            _, summary = self.ns["summarize_evaluation"](rows)
            self.assertEqual(summary["images"].tolist(), [2, 2, 2])
            self.assertEqual(summary["accuracy"].tolist(), [1., 1., 1.])

    def test_fresh_image_prefill_is_routed_but_cached_decode_is_dense(self):
        from types import ModuleType, SimpleNamespace
        cache_module = ModuleType("transformers.cache_utils")
        cache_module.DynamicCache = lambda: object()
        model_module, misc_module = ModuleType("models"), ModuleType("models.misc")
        model_module.omni_attn_mask_naive = object()
        misc_module.interpolate_pos_encoding = object()

        class FakeLLM(torch.nn.Module):
            def __init__(self, test):
                super().__init__()
                self.model = fixtures.TinyBackbone().eval()
                self.test, self.calls = test, 0

            def forward(self, **kwargs):
                self.calls += 1
                if "inputs_embeds" in kwargs:
                    self.model(kwargs["inputs_embeds"])
                    token = 2
                else:
                    self.test.assertFalse(hasattr(self.model, "_depth_router_active"))
                    self.test.assertTrue(all(not layer._forward_pre_hooks for layer in self.model.layers))
                    self.test.assertEqual(kwargs["attention_mask"].shape[-1], 13)
                    self.model(kwargs["input_ids"].float().unsqueeze(-1))
                    token = 0
                logits = torch.zeros(1, 1, 3)
                logits[..., token] = 1
                return SimpleNamespace(logits=logits, past_key_values=object())

        def assemble(model, tokenizer, token_ids, latents, prompt, *args, **kwargs):
            self.assertEqual(prompt, "How many cats are there?\nAnswer with only a single integer.")
            return (torch.arange(1, 13).float().reshape(1, 12, 1), torch.zeros(1, 1, 12, 12),
                    torch.arange(1, 11), {})

        for mode in ["dense", "shared", "specialized"]:
            with self.subTest(mode=mode):
                llm = FakeLLM(self)
                resources = SimpleNamespace(
                    device=torch.device("cpu"), dtype=torch.float32,
                    model=SimpleNamespace(showo=llm), token_ids={"eos_id": 0},
                    image_transform=lambda _: torch.ones(3, 2, 2),
                    vae=SimpleNamespace(sample=lambda *args, **kwargs: torch.zeros(1, 1, 1, 2, 2)),
                    tokenizer=SimpleNamespace(eos_token_id=0, unk_token_id=-1,
                                              convert_tokens_to_ids=lambda _: 0,
                                              decode=lambda ids, **kwargs: "two"))
                config = SimpleNamespace(dataset=SimpleNamespace(
                    preprocessing=SimpleNamespace(max_seq_length=1024)), get=lambda key, default: default)
                scorers = {"f": fixtures.CountingScore(1).eval(), "u": fixtures.CountingScore(-2).eval()}
                with patch.dict(sys.modules, {"models": model_module, "models.misc": misc_module,
                                               "transformers.cache_utils": cache_module}), \
                     patch("evaluation.calculate_token_importance_understanding.assemble_prefill", assemble):
                    result = self.ns["infer_understanding"](
                        {"prompt": "How many cats are there?"}, None, resources, config, scorers,
                        mode)
                self.assertEqual(result["answer"], "two")
                self.assertEqual(result["generated_tokens"], 1)
                self.assertFalse(result["hit_token_limit"])
                self.assertEqual(llm.calls, 2)
                self.assertEqual(len(scorers["f"].calls), 0 if mode == "dense" else 27)
                self.assertEqual(len(scorers["u"].calls), 27 if mode == "specialized" else 0)


if __name__ == "__main__":
    unittest.main()
