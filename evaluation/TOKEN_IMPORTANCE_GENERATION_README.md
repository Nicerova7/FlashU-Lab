# Generation: token-update dataset

**Current research framing:** see [Research direction](RESEARCH_DIRECTION.md).
Treat the L2 score as representation change / an importance proxy pending validation.
Preserve the step dimension for initial U–G analysis; model form and token
correspondence remain open questions.

`calculate_token_importance_generation.py` measures every sequence position at
selected decoder layers and denoising evaluations during a complete Show-o2
text-to-image run. The score is the same candidate metric used for understanding:

```text
change = float32(token_after_layer) - float32(token_before_layer)
score  = sqrt(change[1]^2 + ... + change[d]^2)
I_gen(layer, step, token) = score
```

Each score measures one token's update through one decoder block **at the current
denoising evaluation**. It does not measure the difference between denoising steps.
Scores exclude final backbone RMSNorm and diffusion-head updates. Final RMSNorm,
the diffusion head, guidance, the solver, and VAE decoding still execute to generate
the image. Update magnitude is not validated task importance or removal damage.

## Run in Colab

Use the same Show-o2 setup and WanVAE weights as the understanding collector.
Generation additionally needs the repository's transport dependencies; if absent:

```python
%pip install -q torchdiffeq scipy
```

From `/content/FlashU-Lab`, with this file and the existing understanding collector
present in `evaluation/`:

```python
output_dir = "/content/drive/MyDrive/token_generation_pilot_01"  # Choose a fresh folder.

!python -m evaluation.calculate_token_importance_generation \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    generation_data_file=prompts/generation_counting_spatial_40.json \
    num_prompts=1 \
    layers=all \
    steps=all \
    num_inference_steps=50 \
    batch_size=1 \
    device=cuda \
    dtype=bfloat16 \
    seed=42 \
    cfg_branches=conditional \
    save_input_features=true \
    save_hidden_out=false \
    output_dir="{output_dir}"
```

The default input is the first JSON record: `a photo of one dog`.
`start_sample_index=1` selects the second record. JSON lists of prompt objects or
strings, and `.txt` files with one prompt per line, are supported. Seeds are
`seed + source_record_index`; source records and dataset fingerprints are saved.

**Storage:** with 28 layers, 1024 positions, width 1536, BF16 and all 49 evaluations,
uncompressed input features occupy about **4.02 GiB per prompt** for one CFG branch, plus scores
and latents. `cfg_branches=both` doubles this; `save_hidden_out=true` adds another
copy of the vectors. For a smaller first dataset use `steps=[0,24,48]` (about
252 MiB of uncompressed input features). Gzip reduces the on-disk size by a
data-dependent amount. The complete generation trajectory still runs.
`save_input_features=false` keeps scores and metadata but omits input representations.

## What runs, and where

The entry point `main()` has six numbered steps. Helpers precede it.

1. `prepare_generation_inputs()` calls the native `prepare_gen_input()` and builds
   its attention mask and token metadata.
2. `GenerationRecorder.evaluate()` calls the native `model.t2i_generate()` once per
   Euler evaluation. Native `Showo2Qwen2_5.forward()` fuses the current noisy image
   latents and inserts time/visual embeddings into the sequence.
3. `run_decoder_layers()` runs every backbone layer in an explicit loop, saving
   selected block inputs and L2 updates. Its `input_embeds` argument is the final
   assembled backbone input. It then applies final RMSNorm.
4. The native diffusion head and classifier-free guidance produce the velocity.
   The repository's ODE solver advances the image latents. The VAE decodes the final
   image after the complete trajectory.

`use_measured_backbone()` temporarily replaces only the language-model wrapper's
`forward` during sampling and restores it in `finally`. Decoder layers and model
weights are unchanged. The replacement omits the unused vocabulary projection;
native `t2i_generate()` discards those logits. No hooks, pruning, persistent patches,
or source-model edits are required.

The reference is `inference_t2i.py`, rather than the older
`calculate_obd_cache_new.py`: that OBD script starts from placeholder token-ID
embeddings and does not run the native image-denoising trajectory.

## Tokens, guidance, and time

Native generation uses this sequence:

```text
BOS -> prompt text -> BOI -> optional time -> visual patches -> EOI -> EOS -> padding
```

Token types are `prompt_text`, `chat_format` (BOS/EOS), `image_boundary`, `time`,
`visual`, and `padding`. This template has no understanding system prompt or
question/assistant prefix. Visual tokens can attend to the preceding prompt.
Metadata uses the actual layout rather than token ID alone because EOS and padding
may share an ID. Time/visual slots have no vocabulary token ID after replacement.

All positions, including padding, retain their native computation and are scored.
`valid_token_mask` marks non-padding rows for analysis and future labels;
`summary.csv` excludes padding. No new padding mask changes the native inference.

With positive guidance, the batch is `[conditional, unconditional]`. Both branches
execute, using the same evolving image latents. By default only conditional scores
and vectors are saved. `cfg_branches=both` saves the branches separately; it requires
positive guidance. Scores from the two branches are never averaged or combined
using the guidance formula. Patch coordinates align across branches, but sequence
positions differ because their text-prefix lengths differ.

This first collector supports forward **Linear/velocity with Euler**. The solver
uses `num_inference_steps` as the number of time-grid points. Thus 50 points give
49 model evaluations, indexed 0 through 48; the final endpoint is an integrated
latent state, not an extra backbone evaluation. `steps` selects evaluation indices,
not arbitrary continuous time values. The actual model timestep is saved in every
step file. The code verifies the expected evaluation count.

The default schedule follows the effective `inference_t2i.py` call: **unshifted**.
That demo does not pass the YAML's `transport.do_shift` or
`transport.time_shifting_factor` into the sampler. To explicitly change this
collector's schedule, pass a top-level `time_shifting_factor=3.0`; its actual value
is recorded in `run.json` under `effective_sampling`. This is an intentional change
from the demo default. Other generation scripts may use different sampling settings.

## Saved data

The generation schema is version **3**; the existing understanding-only visualizer
does not accept it. This file collects data; it does not create heatmaps.

| File | Contents |
|---|---|
| `sample_000000_step_0000_conditional.pt.gz` | Losslessly compressed selected-layer inputs, L2 labels, optional outputs, token metadata, current latents, actual timestep and CFG branch |
| `sample_000000.pt` | Small score aggregate, layer/step indices, branch layouts, initial noise, final latents, generated-image path/hash |
| `sample_000000.png` | Final generated image |
| `steps.jsonl` | One record per saved step/branch tensor file, including gzip filename and compressed size in bytes |
| `summary.csv` | Non-padding score statistics per sample, step, branch and layer |
| `run.json`, `analysis.log` | Configuration, effective schedule, provenance and completion status |

```python
import gzip
import torch

sample = torch.load(f"{output_dir}/sample_000000.pt", map_location="cpu", weights_only=True)
I_gen = sample["branches"]["conditional"]["update_l2_tensor"]  # [L, S, N]
print(I_gen.shape)  # Defaults: [28, 49, 1024]
# Tensor axis indices map through layer_indices and step_indices.
print(sample["denoising_timesteps"])

with gzip.open(f"{output_dir}/sample_000000_step_0000_conditional.pt.gz", "rb") as file:
    step = torch.load(file, map_location="cpu", weights_only=True)
valid = step["valid_token_mask"]
X = step["layers"]["13"]["input_features"][valid]  # [N_valid, d]
y = step["layers"]["13"]["update_l2"][valid]       # [N_valid]
# These aligned vectors and scores support representation and proxy-validation analysis.
# Retain layer, timestep, task, branch, sample identity, and token metadata.
```

Per-step files are compressed directly while saving using Python's built-in
`gzip` at level 6. No extra dependency or uncompressed disk copy is needed.
Compression preserves every tensor's values, dtype, shape, and metadata; the
schema remains version 3. The aggregate `sample_000000.pt` and visualization
commands are unchanged. Older uncompressed step files still load with ordinary
`torch.load(path, map_location="cpu", weights_only=True)`.

The feature-storage estimate in the log is **before compression**. Actual savings
depend on the activations; no fixed ratio is assumed. Gzip adds CPU work during
saving/loading. `steps.jsonl` records each completed file's compressed size.
This applies to new collection runs; existing `.pt` files are not converted.
Offline checks verify exact BF16/FP32 tensor and metadata round trips and cleanup
of incomplete compressed writes using synthetic data.

Token positions and patch coordinates persist within a branch's trajectory; their
representations change across layers and steps. They are not automatically aligned
with positions in an understanding sample. Cross-task transfer and safe bypass must
be evaluated separately. The prompt precedes the visual/time span, so prompt-token
updates cannot depend on the later image/time tokens and should stay constant across
steps apart from numerical effects. Understanding uses a different order and mask;
cross-task differences can reflect layout and latent state as well as task.
Keep related prompts, seeds, steps, and branches together
when splitting data for any future model evaluation.

Results stream to disk rather than retaining all feature tensors in memory. A
nonempty output directory is rejected; interrupted runs retain partial files and
are marked `failed`, with no implicit resume. A saved image is an inspection aid,
not a task-quality evaluation or evidence that low-scoring tokens can be skipped.

## Development validation

Fourteen offline checks cover sequence metadata, padding, truncation, grid guards,
guidance branches, Euler evaluation indices, sparse recording, feature/label
alignment, file loading, aggregation, and restoration after errors. Numerical
checks use the repository's real Qwen decoder with tiny random weights and unchanged
native Show-o2 forward/guidance/unpatchify methods, with synthetic image/time/head
modules. They compare native and recorded velocities, layer boundaries, final
normalization, and short trajectories using the real repository sampler and
`torchdiffeq` Euler; differences were zero in CPU float32 checks. The complete
writer/image-output path was checked with a synthetic VAE.

These checks used Transformers 4.46.3 and torchdiffeq 0.2.5. They do not replace a
full pretrained Show-o2/WanVAE run on CUDA; that run has not been performed locally.

## Visualize a saved generation trajectory

`visualize_token_importance_generation.py` reads the small aggregate
`sample_XXXXXX.pt`. It does not load the large per-step feature files, run the model,
or calculate new importance labels.

```python
!python -m evaluation.visualize_token_importance_generation \
    "{output_dir}/sample_000000.pt"
```

The three main figures default to **nine panels in a 3 x 3 grid** when enough
layers/steps were recorded. Defaults select up to nine evenly spaced entries of
each saved axis. With 49 steps and 28 layers, these are:

- Steps: `0 6 12 18 24 30 36 42 48`.
- Layers: `0 3 7 10 14 17 20 24 27`.

Sparse datasets use only recorded IDs; missing measurements are never generated.
Fewer than nine recorded entries produce fewer panels. More than nine explicitly
selected IDs add rows with at most three columns. Each step/token or trace panel
contains all recorded steps, regardless of the layer/token panel selection.

Selections are independent and customizable:

```python
!python -m evaluation.visualize_token_importance_generation \
    "{output_dir}/sample_000000.pt" \
    --steps 0 6 12 18 24 30 36 42 48 \
    --layers 0 3 7 10 14 17 20 24 27 \
    --patch-steps 0 24 48 \
    --patch-layers 0 14 27
```

These are saved **IDs**, not tensor offsets. Explicit IDs must exist in the data.
Patch maps independently default to the first, middle, and last saved IDs on each
axis: three layers x three steps. Increasing the main grids to nine panels does
not expand the patch figure to 81 panels.

The default branch is conditional. Use `--branch unconditional` only if that
branch was saved during collection. Branches are displayed separately.

| Output | Content |
|---|---|
| `layer_token_selected_steps.png` | One panel per selected step: all layers x non-padding tokens |
| `step_token_selected_layers.png` | One panel per selected layer: all recorded steps x non-padding tokens |
| `token_traces_selected_layers.png` | One panel per selected layer: fixed token positions through all recorded steps |
| `layer_token_mean_over_steps.png` | All layers x non-padding tokens; arithmetic mean over all recorded steps |
| `step_token_mean_over_layers.png` | All recorded steps x non-padding tokens; arithmetic mean over all recorded layers |
| `layer_step_visual_mean.png` | All layers x all recorded steps; arithmetic mean over visual tokens only |
| `visual_patch_heatmaps.png` | Individual visual-token scores at selected patch layers/steps, arranged by spatial coordinates |
| `plot_settings.json` | Source hash, branch, independent selections, averaging definitions, color limits, and output filenames |

Every run writes **seven PNG files**, including when fewer panels are available.
Start U–G analysis with the step-resolved figures and full `[L, S, N]` tensor.
The averaged figures are secondary summaries and do not replace the step dimension.
All averages use the entire recorded axes, not just the displayed selections:

```text
layer_token_mean[layer, token] = mean over recorded steps of score[layer, step, token]
step_token_mean[step, token]   = mean over recorded layers of score[layer, step, token]
visual_mean[layer, step]      = mean over visual tokens of score[layer, step, token]
```

The first two means preserve individual non-padding token positions, including
text and special tokens. The third includes only visual tokens. Means operate on
raw L2 scores before log-color mapping. Recorded entries receive equal weight;
the step average is not a time-weighted integral for shifted or sparse schedules.
No missing steps/layers are inferred. Averaging can hide isolated spikes.

All non-padding token heatmaps, including the two averaged counterparts, share
one color scale calibrated over the entire recorded branch's individual scores.
Patch maps use a separate scale over all recorded visual-token scores. The visual
layer/step mean uses its own scale over all its cells. There is no per-panel,
per-layer, or percentile normalization. Log display is linear from 0 to 1 and
logarithmic above 1, so zero remains representable. Colors are not automatically
calibrated between different branches or samples.

Trace panels use the same token colors and shared Y-axis limits, calibrated over
those token positions in all recorded layers and steps. Default trace tokens are
the first non-padding position of each available type, not the highest scores.
Original token positions appear on token-map axes. Heatmap step rows/columns
have equal size, with saved step IDs and actual timestep labels. Curves use actual
timestep spacing, including shifted schedules, and connect only recorded points.

Additional arguments:

```text
--tokens 0 2 20        # Non-padding sequence positions for curves; example values only.
--color-scale linear  # Heatmap colors; default: log.
--trace-scale linear  # Curve y-axis; default: log.
--image-path PATH     # Relocated final generated image, verified against its saved hash.
--output-dir PATH     # Override the default plot folder.
```

Images are saved to `visualizations/<sample name>/<branch>/` next to the aggregate.
Rerunning replaces matching plot filenames; source tensor files are never modified.
Old per-step/per-layer PNGs from earlier versions are left in place.
`plot_settings.json` lists outputs from the current run. Use a separate output
folder to keep multiple configurations or avoid mixing old and new files.

Patch maps place each saved token L2 score at its patch row/column, without
averaging. They measure change through a decoder layer at one timestep, not
attention or change between denoising steps. The displayed image is the **final
generated image, for context only**; early maps describe noisy states. No
intermediate images are decoded. A missing image permits patch-only plots;
a supplied image must match the saved hash. An adjacent `sample_XXXXXX.png` is
detected when moving results from Colab. The image is not resized or cropped.

These remain update-magnitude visualizations. Bright regions do not establish
task relevance or safe token removal. Positions do not automatically correspond
to the same content across understanding/generation samples.

Offline checks cover raw values and arithmetic means, sparse axes, panel ordering,
independent patch selections, shared scales and trace limits, padding, coordinate
mapping, image identity, and a single-layer/single-step/all-zero run. Figure layouts
are checked with explicitly labelled synthetic data; these previews are software
checks, not model research results.
