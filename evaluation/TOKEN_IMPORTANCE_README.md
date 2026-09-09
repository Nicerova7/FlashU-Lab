# Understanding prefill: token importance dataset

This script runs one unchanged Show-o2 understanding prefill per example and
saves one candidate label per input token at each decoder layer. All token types
are included, and all layers are recorded by default (28 for Show-o2-1.5B).
The backbone runs once through the complete prefill sequence and stops before
answer decoding.

`collect_token_updates()` uses an explicit loop over `core.layers`. At each layer
it captures the input vectors, executes the layer, calculates the token scores,
and saves the selected layer's data. The normal output feeds the next layer.
Position IDs and rotary embeddings follow the repository's `Qwen2Model.forward`;
the prepared attention mask is passed through unchanged. No hooks are used.

For one token, compare its hidden-state vector immediately before and
after the selected decoder layer. The score is the distance between these two
vectors:

```text
change = token_after_layer - token_before_layer
score  = sqrt(change[1]^2 + ... + change[d]^2)
I_und(layer, token) = score
```

Here, `d` is the number of components in the vector; `^2` means squared.
The calculation uses float32 for both vectors before subtraction.

Tiny example with two components:

```text
Before the layer: [1.0, 2.0]
After the layer:  [1.3, 2.4]
Change:           [0.3, 0.4]
Score: sqrt(0.3^2 + 0.4^2) = sqrt(0.25) = 0.5
```

A larger score means the layer changed this token more. It does not by itself
prove that the token is more important for answering the question.

The label answers **“How much did this layer update this token?”** It is not a
keep/drop decision, a Hessian OBD quantity, or demonstrated downstream removal
damage. There is no pruning, bypassing, answer decoding, neuron scoring, or MLP
training in this script.

It adapts the image preparation, chat template, visual fusion, and attention layout
from `evaluation/calculate_obd_cache_understanding.py` on FlashU-Lab **develop** at
commit `d30ae0d4f34fa9bce1121329db4c1747f30db737`. The original file is unchanged.
That source's existing neuron/KV-group measurements are squared reconstruction
errors averaged across examples; this new token label is the individual L2 norm
defined above. They are different quantities and should not be mixed.

## Token coverage

The input order is:

```text
system message -> user header -> image-start -> optional time
-> visual patches -> image-end -> question -> assistant header
```

Every position receives a score and a type label:

| `token_type` | Meaning |
|---|---|
| `system_text` | The content `You are a helpful assistant.` |
| `question_text` | The supplied question, after any explicit truncation |
| `chat_format` | Template role names, separators, newlines, and chat markers |
| `image_boundary` | The image-start and image-end markers |
| `visual` | Fused image-patch embeddings |
| `time` | The optional timestep embedding, set to 1.0 for understanding |

System text is text with a different role. The type labels describe where a token
comes from; they do not change the input, mask, score formula, or computation.
System-content boundaries use the fast tokenizer's character offsets, preserving
the original tokenization. There are no generated answer tokens and no padding.

## Run in a Show-o2 environment

The script `calculate_token_importance_understanding.py` is in the repository's
`evaluation/` directory. Run from the repository root with the required Show-o2
dependencies and model/VAE weights. The source revision listed above specifies
`transformers==4.46.3` and `tokenizers==0.20.3`; the collector also uses PyTorch,
OmegaConf, Pillow, torchvision, and the existing model imports.

Start with two examples and all layers (layer indices are zero-based):

```bash
python -m evaluation.calculate_token_importance_understanding \
  config=configs/showo2_1.5b_demo_432x432.yaml \
  understanding_data_file=prompts/understanding_counting_20.json \
  num_prompts=2 \
  start_sample_index=0 \
  layers=all \
  batch_size=1 \
  seed=42 \
  save_hidden_out=true \
  output_dir=token_prefill_counting_pilot
```

These commands use Bash/Colab line continuation. In PowerShell, place the command
on one line. The dataset's `image_path` values and configured VAE file must exist
on the machine running the script. The checked-in datasets use Colab/Drive paths;
the script does not download or relocate their images.

Each input record uses the following JSON format:

```json
[
  {
    "image_path": "/path/to/image.jpg",
    "prompt": "How many dogs are there?",
    "expected_answer": "two"
  }
]
```

`expected_answer` and any other record fields are preserved as metadata. They are
not used to calculate this label and are not predictor input features.

If `layers` is omitted, the script records every decoder layer. Use `layers=[13]`
or `layers=[0,13,27]` to record a subset. It checks indices against the loaded
model rather than assuming a fixed layer count. `num_prompts=20` collects more examples.

The pilot intentionally processes one example at a time, so different question
lengths produce different sequence lengths without padding or cross-example
averaging. Every decoder layer executes normally even when only one is recorded.

## Saved files

| File | Contents |
|---|---|
| `sample_000000.pt`, etc. | Features, labels, the layer-by-token score matrix, token types/IDs, visual patch coordinates, encoded image latents, and sample metadata |
| `scores.csv` | One row per sample/layer/token, including its type, vocabulary ID/piece, position, patch coordinates when applicable, and raw L2 label |
| `summary.csv` | Per-sample, per-layer statistics over all input tokens, with total and visual token counts |
| `samples.jsonl` | One record per saved example, linking its tensor file to prompt, image hash, layout and timing |
| `run.json` | Score definition, completion status, configuration, selected layers, versions, source hashes and model configuration |
| `analysis.log` | Progress and errors |

For each layer in a sample file:

- `input_features`: `[N, d]`, in the model's dtype. These are the token
  vectors entering that decoder layer, before its input normalization.
- `update_l2`: `[N]`, FP32. These are the **candidate MLP labels**.
- `hidden_out`: `[N, d]`, only when `save_hidden_out=true`. This is optional
  verification data, not a proposed MLP input.

Here `N` is the complete prefill sequence length for that example; `d` is the
hidden-vector width. Rows in all three tensors align with `token_positions` and
the `tokens` list, in sequence order. Each `tokens[i]` contains `sequence_position`,
`token_type`, `token_id`, `token_piece`, `patch_index`, `patch_row`, and `patch_col`.
Vocabulary IDs/pieces are null for visual and time embeddings. Patch fields are
null for every non-visual position (blank in CSV). `token_piece` is a tokenizer
vocabulary piece, which may contain whitespace encodings rather than a whole word.

`visual_token_positions` remains available as the visual subset of sequence
positions. Its patch coordinates are in the corresponding `tokens` entries and
refer to the resized, center-cropped VAE patch grid, not original image pixels.

`update_l2_matrix` is `[L, N]`, FP32: row `j` belongs to `layer_indices[j]`.
With all 28 layers selected, this is the **28 x N** matrix `I_und(layer, token)`.
Each example has its own matrix; token counts can vary between examples.
The collector stops at the last decoder block, before final RMSNorm and the
vocabulary head. Even the last layer's label measures only that block's update.

The optional `hidden_out` roughly doubles the feature storage. For an example with
800 total tokens and hidden width 1536, BF16 input features occupy about 2.34 MiB
per layer, or 65.6 MiB across 28 layers, before serialization overhead.
Raw image latents are saved once per sample to support later replay.

This output uses schema version 2 and score name `token_decoder_update_l2_v2`.
Version 1 files contain only visual rows; do not mix their row indexing with this
schema. Patch metadata now lives in the per-token records.

## Read labels for the future MLP

```python
import torch

sample = torch.load("token_prefill_counting_pilot/sample_000000.pt",
                    map_location="cpu", weights_only=True)
layer = sample["layers"]["13"]
X = layer["input_features"].float()  # [N, d], every input token
y = layer["update_l2"]              # [N]
I_und = sample["update_l2_matrix"]   # [L, N]; L=28 for the full 1.5B backbone

# Row j of X and element j of y refer to exactly the same token.
j = 0
print(sample["sample"]["sample_id"], sample["tokens"][j], y[j])

# Inspect question and visual rows while retaining all token types in the file.
content_rows = [i for i, token in enumerate(sample["tokens"])
                if token["token_type"] in {"question_text", "visual"}]
X_content, y_content = X[content_rows], y[content_rows]

# Optional verification if save_hidden_out=true was used:
measured = (layer["hidden_out"].float() - X).norm(dim=-1)
torch.testing.assert_close(measured, y)
```

No normalization or keep/drop threshold is imposed. Preserve these raw labels.
If a later experiment learns across layers, decide normalization using training
data only. Split future train/validation/test data by `image_sha256` (and related
sample groups), not by random token rows: the same image may appear under multiple
questions, layers, or runs.

## Interpretation and reproducibility

**The source prefill layout is image-before-question.** Its causal multimodal mask
lets question tokens attend to image tokens, but image tokens cannot attend to the
later question. Consequently the visual-token update labels describe image and
prefix processing, not question-conditioned visual usefulness. Different questions
for the same encoded image should not be interpreted as independently different
visual-importance evidence. Question-token updates can reflect both the image and
the preceding question text. Recording these rows adds that information, but
update magnitude still needs validation against downstream task performance.

The original script's attention span begins at the first visual token and has
length `N_visual + int(add_time_embeds)`. With time embeddings enabled, it includes
EOI and excludes the time token. This collector preserves that convention for
baseline comparability. The **label domain is independent of that attention span**:
every input position is scored, including BOI, EOI, time, and all text/template positions.
The saved `attention_span` makes this behavior inspectable.

Images use the source Resize/CenterCrop/Normalize pipeline. Seeding is per example
(`seed + record_index`) and recorded. `vae_deterministic=false` follows the source
sampling default; `vae_deterministic=true` is an explicit alternative. Saved latents
allow later experiments to reuse the same encoded input. Exact reproduction across
different hardware or dependency versions is not guaranteed.

Overlong questions fail by default. Use `truncate_prompt=true` only if intended;
the used token IDs and truncation counts are recorded. The script derives visual
counts from actual latent/embedding shapes and handles variable sequence lengths.
The configured transform still resizes/crops all images to its specified resolution.

Choose a fresh `output_dir` for each run. Existing nonempty directories are rejected
to prevent accidental mixing of model versions, examples or score definitions.
The script writes individual tensor files atomically and updates `run.json` after
each example. On failure, already written files remain and the run is marked
`failed`; there is no implicit resume. Trust complete runs for downstream analysis.

Timing includes feature collection and CPU transfer and is analysis cost, not a
measurement of inference speedup. The collector executes the unchanged decoder
layers without a vocabulary projection, which is sufficient to observe these layer
boundaries. It does not evaluate answers or establish safe removal.

## Validation before training a predictor

After collecting candidate labels, compare low-score bypass with random bypass at
the same budget using an actual understanding-quality measure. This script prepares
the dataset for that next experiment; it does not establish the labels' predictive
validity or train an MLP predictor.

Development checks cover known numerical labels, saved feature/label and matrix
alignment, all token types, optional time inclusion, variable sequence lengths,
truncation, non-finite data rejection, unchanged input assembly/visual scores,
and the file-writing pipeline. Numerical checks compare the explicit loop's
per-layer inputs, outputs, and scores against a normal `core(...)` forward pass,
including all 28 layers and selective recording. Integration checks use
the repository's actual Qwen2Model with tiny random weights and Transformers 4.46.3;
the output-pipeline test used synthetic image/VAE/model-loading fixtures. Full
pretrained Show-o2 collection requires model/VAE weights and real images; a full
collection run was not performed during this validation.
