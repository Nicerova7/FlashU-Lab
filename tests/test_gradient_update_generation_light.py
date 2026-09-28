"""CPU regression checks; no Show-o checkpoint, VAE, or GPU is required.

Run from the repository root:
    python -m unittest discover -s tests -p test_gradient_update_generation_light.py
"""

import gzip
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from evaluation import calculate_gradient_update_generation as original
from evaluation import calculate_gradient_update_generation_light as light


class TinyBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer(
            "weight",
            scale * torch.tensor([[0.2, 0.1, -0.3], [-0.2, 0.4, 0.1], [0.3, -0.1, 0.2]]),
        )

    def forward(self, hidden_states, **kwargs):
        # Mix positions as well as channels so each recorded output contributes
        # to the eventual flow loss, as it does through downstream attention.
        update = torch.tanh(hidden_states @ self.weight + hidden_states.mean(1, keepdim=True))
        return (hidden_states + update,)


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(0.7), TinyBlock(1.1), TinyBlock(1.4)])
        self.norm = nn.Identity()

    def rotary_emb(self, hidden_states, position_ids):
        return None


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyCore()

    def forward(self, **kwargs):
        raise AssertionError("The collector should install its measured backbone")


class TinyShowo(nn.Module):
    def __init__(self):
        super().__init__()
        self.showo = TinyBackbone()

    def forward(self, *, image_latents, t, attention_mask, **kwargs):
        hidden = image_latents + t.view(-1, 1, 1) * 0.13
        outputs = self.showo(inputs_embeds=hidden, attention_mask=attention_mask)
        final = outputs["hidden_states"][-1]
        return None, final * 0.23 + final.mean(1, keepdim=True) * 0.07


class RowCollector:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(dict(row))


def make_resources():
    return SimpleNamespace(
        model=TinyShowo().eval(), device=torch.device("cpu"), dtype=torch.float32,
        # Deliberately unordered: saved rows must match sorted layer metadata.
        layer_ids=[2, 0],
    )


def make_inputs():
    layout = {
        "batch_index": 0,
        "tokens": [
            {"token_type": "text", "position": 0},
            {"token_type": "visual", "position": 1},
            {"token_type": "visual", "position": 2},
            {"token_type": "padding", "position": 3},
        ],
        "valid_token_mask": torch.tensor([True, True, True, False]),
        "visual_token_positions": torch.tensor([1, 2]),
        "attention_span": {"offset": 1, "length": 2},
    }
    return {
        "text_tokens": torch.tensor([[5, 6, 7, 0]]),
        "attention_mask": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        "modality_positions": torch.tensor([[[1, 2]]]),
        "sequence_length": 4,
        "grid_height": 1,
        "grid_width": 2,
        "branches": {"conditional": layout},
    }


def assert_no_features(test, value):
    if isinstance(value, dict):
        for key, item in value.items():
            test.assertNotIn(key, {"input_features", "hidden_in", "hidden_out", "image_latents"})
            assert_no_features(test, item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_features(test, item)


class GenerationLightTests(unittest.TestCase):
    def test_score_matches_original_autograd_and_restores_backbone(self):
        resources = make_resources()
        inputs = make_inputs()
        latents = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3) / 13 - 0.4
        target = latents.flip(1) * 0.2
        before = resources.model.showo.forward

        layers, expected_loss = original.score_step(resources, inputs, latents, target, 0.37)
        matrix, loss = light.score_step(resources, inputs, latents, target, 0.37)
        expected = torch.stack([layers[str(index)]["gradient_update"] for index in [0, 2]])

        self.assertEqual(resources.model.showo.forward, before)
        self.assertEqual(matrix.shape, (2, 4))
        self.assertEqual(matrix.device.type, "cpu")
        self.assertEqual(matrix.dtype, torch.float32)
        self.assertFalse(matrix.requires_grad)
        self.assertTrue(bool((matrix > 0).all()))
        torch.testing.assert_close(matrix, expected, rtol=0, atol=0)
        self.assertEqual(loss, expected_loss)

    def test_collection_preserves_scores_and_replay_without_feature_archives(self):
        resources = make_resources()
        inputs = make_inputs()
        settings = {
            "recorded_steps": [1, 4], "timesteps": [0.2, 0.8],
            "reference_guidance_scale": 4.0,
        }
        run = {"base_seed": 42, "run_id": "cpu-test", "dataset_sha256": "abcdef1234567890"}
        record = {"prompt": "Two objects next to each other"}
        noise = torch.linspace(-1, 1, 12).reshape(1, 4, 3)
        reference = noise.flip(1) * 0.4 + 0.1

        def fake_reference(record_index, reference_inputs, settings, resources, output_dir):
            path = output_dir / f"sample_{record_index:06d}_reference.png"
            path.write_bytes(b"identical mock reference image for both collectors")
            return reference.clone(), noise.clone(), path

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_dir, new_dir = root / "original", root / "light"
            old_dir.mkdir()
            new_dir.mkdir()
            writers = []
            for module, directory in [(original, old_dir), (light, new_dir)]:
                writer = RowCollector()
                writers.append(writer)
                with patch.object(module, "prepare_generation_inputs", return_value=inputs), \
                     patch.object(module, "generate_reference", side_effect=fake_reference):
                    module.measure_one(record, 12, {}, settings, resources, directory, run, writer)

            old_payload = torch.load(old_dir / "sample_000012.pt", map_location="cpu", weights_only=True)
            payload = torch.load(new_dir / "sample_000012.pt", map_location="cpu", weights_only=True)
            self.assertEqual(
                sorted(path.name for path in new_dir.iterdir()),
                ["sample_000012.pt", "sample_000012_reference.png"],
            )
            self.assertEqual(payload["storage_format"], "gradient_update_generation_light_v1")
            self.assertFalse(payload["input_features_saved"])
            self.assertEqual(payload["dtype"], "float32")
            self.assertEqual(payload["layer_indices"], [0, 2])
            self.assertEqual(payload["step_indices"], [1, 4])
            self.assertEqual(payload["score_axes"], ["layer", "step", "token"])
            self.assertEqual(payload["sample"]["record_index"], 12)
            self.assertEqual(payload["sample"]["seed"], 54)
            self.assertEqual(payload["sample"]["source_record"], record)
            self.assertEqual(payload["task_losses"], old_payload["task_losses"])
            self.assertEqual(writers[0].rows, writers[1].rows)
            torch.testing.assert_close(
                payload["branches"]["conditional"]["gradient_update_tensor"],
                old_payload["branches"]["conditional"]["gradient_update_tensor"],
                rtol=0, atol=0,
            )
            torch.testing.assert_close(payload["initial_noise"], noise, rtol=0, atol=0)
            torch.testing.assert_close(payload["reference_latents"], reference, rtol=0, atol=0)
            for key in ["text_tokens", "modality_positions"]:
                torch.testing.assert_close(payload["replay_inputs"][key], inputs[key], rtol=0, atol=0)
            self.assertEqual(payload["replay_inputs"]["max_seq_len"], 4)
            assert_no_features(self, payload)

            for step, expected_time in zip(settings["recorded_steps"], settings["timesteps"]):
                with gzip.open(old_dir / f"sample_000012_step_{step:04d}_conditional.pt.gz", "rb") as stream:
                    old_step = torch.load(stream, map_location="cpu", weights_only=True)
                latent, timestep = light.reconstruct_latent(payload, step, device="cpu")
                self.assertEqual(timestep, expected_time)
                torch.testing.assert_close(latent, old_step["image_latents"], rtol=0, atol=0)
                self.assertFalse(latent.requires_grad)
                replayed = light.replay_step_features(resources, payload, step)
                self.assertEqual(set(replayed), {"0", "2"})
                for layer_index in replayed:
                    for field in ["input_features", "gradient_update"]:
                        torch.testing.assert_close(
                            replayed[layer_index][field], old_step["layers"][layer_index][field],
                            rtol=0, atol=0,
                        )
                        self.assertFalse(replayed[layer_index][field].requires_grad)

            with self.assertRaises((ValueError, KeyError)):
                light.reconstruct_latent(payload, 0)

    def test_replay_preserves_low_precision_arithmetic(self):
        # Casting to float32 before interpolation would not replay bfloat16 input
        # rounding exactly; the saved noise and reference dtype must be retained.
        payload = {
            "storage_format": "gradient_update_generation_light_v1",
            "step_indices": [3, 8],
            "denoising_timesteps": [0.17, 0.83],
            "initial_noise": torch.tensor([[[0.31, -0.72, 1.83]]], dtype=torch.bfloat16),
            "reference_latents": torch.tensor([[[-0.21, 0.42, 0.63]]], dtype=torch.bfloat16),
        }
        for step, time in zip(payload["step_indices"], payload["denoising_timesteps"]):
            actual, actual_time = light.reconstruct_latent(payload, step)
            expected = (1.0 - time) * payload["initial_noise"] + time * payload["reference_latents"]
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertEqual(actual_time, time)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
