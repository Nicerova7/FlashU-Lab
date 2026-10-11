# Step 0 — Counting test

This is a fixed pilot for comparing dense, shared, specialized and depth-dependent
FFN routing. No inference or training has been run on these inputs.

`evaluation/counting_test.zip` includes the manifests and all 40 images for transfer
to Drive/Colab. The image files and ZIP are excluded from Git; a repository clone
alone will not include the images.

| Task | Inputs per configuration | Performance measure |
|---|---|---|
| Understanding | 40 COCO validation images: 10 each for counts 1–4 | Exact integer count accuracy |
| Generation | 10 official GenEval counting prompts × 4 fixed seeds = 40 images | GenEval counting correct rate |
| Generation quality | 10 matched dense/routed pairs | Blind A/B/tie visual review |

## 0.1 Inputs prepared

- `understanding.json`: image paths, questions and provisional COCO reference counts.
- `images/`: the 40 U images; resolve relative paths against this directory.
- `generation.json`: selected prompts and their original GenEval metadata/index.
- `generation_cases.json`: fixed prompt/seed pairs and output filenames.
- `evaluation_metadata.jsonl`: selected metadata in GenEval format.
- `human_audit_cases.json`: ten cases chosen before viewing outputs, one per prompt.
- `protocol.json`: selection seeds, source hashes, exclusions and evaluation rules.
- `geneval_metadata.jsonl`: the original downloaded prompt pool for reproduction.

Previous inputs in `prompts/understanding_*.json` and
`prompts/generation_counting_spatial_40.json` were excluded. U exclusions use COCO
image IDs; G exclusions use normalized prompts. Each U image occurs once.
This establishes separation from the repository's known router inputs, not from
Show-o2's pretraining data.

## 0.2 Finish the test checks

1. Visually check every U image/question against `target_count`. COCO annotations
   provide initial labels; crowd cases and tiny instances were excluded, but
   annotations can still omit an object. Set `review_status` to `verified` only
   after checking. Resolve ambiguous cases before running any configuration.
   Use `pending` for an unreviewed case and `replace` for an incorrect or ambiguous
   case. Replacements retain the original case ID and target count, but their
   image and question may change. Review the new image before marking it verified.
   Replacement history is recorded in `protocol.json`.
2. Check against the actual Colab collection records. The local repository does
   not contain the Drive feature archives. In the existing notebook, after 1.3:

   ```python
   (OUTPUT_DIR / "used_records.json").write_text(json.dumps([
       {"task": x["task"], "source_record": x["source_record"]}
       for x in examples
   ], indent=2))
   ```

   The preparation script accepts this file via `--used-records`. If those records
   contain additional inputs, prepare a new test directory before inference.
   Include any other records previously used for router training or tuning.
3. Freeze the router checkpoint and all inference settings. Apply the same
   keep fraction and dense edge-step schedule to all three routed modes.
   The dense mode computes all FFNs. Use identical images/prompts/seeds across
   modes. Do not choose a depth mask after viewing this test's scores.

`protocol.json` remains pending until these checks are complete. This pilot can
measure a quality/speed trade-off; 40 cases cannot establish that quality is
universally preserved. The G set is a counting subset, not the full GenEval score.

## 0.3 Human audit

After generation, use the same ten cases for each dense/routed comparison. Hide
mode labels and randomize left/right placement. Record object/count correctness
separately from visual preference (anatomy/shape, exposure/colors and artifacts).
Allow ties. Do not replace cases because a result looks unusual.

## What already exists / what comes next

- **Reuse:** trained f/u/g scorers, `FFNRouter`, U decode, G sampler and VAE.
- **Modify next:** inference functions currently require saved Gradient × Update
  payloads. Add raw-image/raw-prompt inputs for these fresh cases; no new gradient
  collection or scorer training is required.
- **Add next:** U count scoring and the official GenEval evaluator. Keep GenEval's
  detector environment separate from Show-o2 if their dependencies conflict.
- **Timing next:** `evaluation/inference_dpg_accelerated.py` already synchronizes
  CUDA and excludes the first prompt as warm-up. Its outer timing includes PNG
  saving and uses FlashU acceleration, so it is not a depth-router latency result.
  Measure each mode after warm-up, including router/selection/gather/scatter,
  U answer decoding and G VAE decoding; exclude loading, saving and evaluation.
  Repeat paired runs with alternating mode order. FFN row reductions alone are
  not measured speedups.

The existing depth-router notebook was left untouched.

## Reproduce preparation

Uses Python's standard library; no GPU or Show-o2 dependencies. Obtain
`instances_val2017.json` from the [official COCO annotations](https://cocodataset.org/#download).
The downloaded GenEval prompt pool is included here.

```bash
python evaluation/prepare_counting_test.py \
  --coco-annotations /path/to/instances_val2017.json \
  --geneval-metadata evaluation/counting_test/geneval_metadata.jsonl \
  --used-records /path/to/used_records.json \
  --output /path/to/new_counting_test \
  --download-images
```

Preparation refuses to overwrite an existing protocol. If image downloading is
interrupted, resume the frozen image list without selecting new cases:

```python
import json
from pathlib import Path
from evaluation.prepare_counting_test import download_images

test_dir = Path("evaluation/counting_test")
download_images(json.loads((test_dir / "understanding.json").read_text()), test_dir)
```

GenEval metadata, detector setup and output format:
[official GenEval repository](https://github.com/djghosh13/geneval).
COCO image terms: [COCO terms of use](https://cocodataset.org/#termsofuse).
