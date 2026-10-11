"""CPU checks for the counting notebook's explicit U specialization mask."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "counting_experiment/experiment.ipynb"
spec = importlib.util.spec_from_file_location(
    "depth_router_fixtures", ROOT / "tests/test_depth_router_notebook.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class UnderstandingMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(fixtures, "NOTEBOOK", NOTEBOOK):
            cls.ns = fixtures.notebook_functions()

    def run_router(self, task, mode, selected_layers):
        core = fixtures.TinyBackbone().eval()
        scorers = {name: fixtures.CountingScore(scale).eval()
                   for name, scale in [("f", 1), ("u", -2), ("g", -2)]}
        hidden = torch.arange(1, 11).float().reshape(1, 10, 1)
        eligible = torch.ones(1, 10, dtype=torch.bool)
        Router = self.ns["FFNRouter"]
        with torch.inference_mode(), Router(
            core, scorers, task, mode, eligible, keep_fraction=.8,
            depth_start=24, u_specialized_layers=selected_layers,
        ):
            result = core(hidden)
        return result, core, scorers

    def test_noncontiguous_mask_changes_actual_ffn_token_selection(self):
        _, core, scorers = self.run_router("U", "depth", (0, 13, 26))
        self.assertEqual([int(layer[0]) for _, layer in scorers["u"].calls], [0, 13, 26])
        self.assertEqual(scorers["g"].calls, [])
        # f + u reverses rankings at layer 0; its FFN processes tokens 1–8.
        torch.testing.assert_close(
            core.layers[0].mlp.rows[0].flatten().sort().values,
            -torch.arange(8, 0, -1).float(), rtol=0, atol=0)
        self.assertTrue(all(layer.attention_rows == [10] for layer in core.layers))
        self.assertTrue(all(not layer._forward_pre_hooks for layer in core.layers))

    def test_empty_and_full_masks_match_shared_and_specialized(self):
        empty, _, scorers = self.run_router("U", "depth", ())
        shared, _, _ = self.run_router("U", "shared", (0, 13, 26))
        torch.testing.assert_close(empty, shared, rtol=0, atol=0)
        self.assertEqual(scorers["u"].calls, [])
        full, _, scorers = self.run_router("U", "depth", tuple(range(27)))
        specialized, _, _ = self.run_router("U", "specialized", ())
        torch.testing.assert_close(full, specialized, rtol=0, atol=0)
        self.assertEqual(len(scorers["u"].calls), 27)

    def test_u_mask_does_not_change_generation_boundary(self):
        result, _, scorers = self.run_router("G", "depth", (0, 13, 26))
        self.assertEqual([int(layer[0]) for _, layer in scorers["g"].calls], [24, 25, 26])
        self.assertEqual(scorers["u"].calls, [])
        other, _, _ = self.run_router("G", "depth", ())
        torch.testing.assert_close(result, other, rtol=0, atol=0)

    def test_rank_evaluation_uses_same_u_mask(self):
        examples = [dict(uid=task, task=task, category=task, split="test", steps=[-1])
                    for task in ("U", "G")]
        models = {42: {"f": fixtures.LinearScore(-1),
                       "u": fixtures.LinearScore(2), "g": fixtures.LinearScore(2)}}
        details = self.ns["evaluate"](
            examples, fixtures.FakeSource(), models, split="test", layers=(0, 14),
            depth_start=14, u_specialized_layers=(0,), progress=None)
        depth = details[details["mode"] == "Depth-dependent"]
        for task, expected in [("U", [1., 0.]), ("G", [0., 1.])]:
            rows = depth[depth["task"] == task].sort_values("layer")
            self.assertEqual(rows["overlap"].tolist(), expected)
            self.assertEqual(rows["specialization_gate"].tolist(), expected)

    def test_invalid_mask_layers_are_rejected(self):
        mask = self.ns["make_u_layer_mask"]
        self.assertEqual(mask(()), (0,) * 27)
        self.assertEqual(mask((24, 25, 26)), (0,) * 24 + (1,) * 3)
        for value in [(-1,), (27,), ("24",), (True,)]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                mask(value)


if __name__ == "__main__":
    unittest.main()
