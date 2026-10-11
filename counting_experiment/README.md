# Counting evaluation

Open **`understanding_counting_eval.ipynb`** for the new understanding evaluation.
It compares dense inference, shared routing (`f`), and specialized routing (`f + u`)
on the same 40 reviewed counting images. It reuses the trained scorers and reports
count accuracy and inference time. `router_training_and_overlap.ipynb` remains the earlier router
training and overlap experiment.

## Files needed in Colab

| Item | Default location |
|---|---|
| Questions and reference answers | Cloned repository: `counting_experiment/test/understanding.json` |
| All 40 test JPEGs | Drive: `MyDrive/intership/counting_test_images/` (JPEGs directly inside) |
| Existing scorer checkpoint | `MyDrive/intership/gradient_mlp_results/depth_router_minimal_v2/routers.pt` |
| Original U collection, for held-out-image checks | `MyDrive/intership/gradient_update_understanding/` |
| VAE weights, downloaded if absent | `MyDrive/intership/model_weights/Wan2.1_VAE.pth` |

The notebook imports its scorer, router, and evaluation functions from
`understanding_eval.py`. It clones the configured repository branch for that
helper, the Show-o2 model, image preparation, and the test manifest.
It does not need the training notebook, generation
feature archives, or new scorer training. JPEGs remain excluded from Git. Upload
the local `test/images/` folder into `MyDrive/intership/`, then rename the uploaded
folder to `counting_test_images`. Do not add another `images/` level inside it.
You do not need to upload the test JSON to Drive.

## Run

Push the notebook, `understanding_eval.py`, manifest, and repository code to the
configured GitHub branch. Open the notebook from GitHub in a fresh Colab GPU runtime.
Run its five cells in order:

1. Install dependencies, mount Drive, and clone the repository.
2. Check `IMAGE_DIR` and review the grid of 40 **model-input crops** against the
   reference counts. Resolve any crop/label mismatch before inference.
3. Load the scorers, check that test images were absent from the previous U
   collection, and load Show-o2.
4. Run dense, shared, and specialized inference.
5. Compare counting accuracy and time; save results.

Cases marked `pending` can be previewed in step 2; step 4 requires every case to be
`verified`. Confirm counts from the cropped images before changing their status.
The current ten replacement JPEGs are bundled in `crop_replacements.zip` for
manual upload to the existing Drive image folder. The ZIP and JPEGs are ignored by Git.

Routing skips 10% of image-token FFN rows (rounded to an integer keep count) during
prefill in layers 0–26. Attention, text, layer 27, and answer decoding stay dense.
Specialized routing applies `u` at every routed layer. The default runs three timing
repeats per image and mode with identical seeds and rotating mode order.

## Outputs

Each run creates a new `results/understanding_<timestamp>/` directory:

- `summary.csv`: count accuracy, measured speedup, and paired correctness changes.
- `per_image.csv`: answer, correctness, and median latency per image/mode.
- `per_call.csv`: individual timing repeats.
- `run.json`: checkpoint and manifest provenance, settings, and GPU.

Timing includes image preprocessing, VAE encoding, prompt assembly, routing,
prefill, and answer decoding. It excludes model loading, image file reads, output
saving, reference scoring, and warm-up. Unparseable answers count as incorrect;
raw answers remain available for review. The 40-image pilot measures a trade-off
on this counting set and does not establish unchanged quality on every task.
