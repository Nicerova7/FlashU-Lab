"""Plot one saved generation trajectory; no model inference or new importance scores.

Run from the repository root:
    python -m evaluation.visualize_token_importance_generation \
        token_generation_pilot/sample_000000.pt --steps 0 24 48 --layers 0 14 27

Use the aggregate sample_XXXXXX.pt, not an individual step/feature file.
Steps and layers are saved IDs, not tensor-axis offsets. Defaults select the first,
middle, and last recorded IDs. Scores remain raw L2 update magnitudes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from PIL import Image
import torch

from evaluation.visualize_token_importance_understanding import (
    make_color_norm, save_figure, token_label,
)

TYPE_COLORS = {
    "prompt_text": "#009E73", "chat_format": "#777777",
    "image_boundary": "#D55E00", "time": "#CC79A7", "visual": "#E69F00",
    "padding": "#DDDDDD",
}
SCORE_LABEL = "Layer-update L2 magnitude (raw)"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_file", type=Path, help="Generation aggregate sample_XXXXXX.pt")
    parser.add_argument("--steps", type=int, nargs="+", help="Saved step IDs for layer/token and patch maps")
    parser.add_argument("--layers", type=int, nargs="+", help="Saved layer IDs for step/token maps, patches, and curves")
    parser.add_argument("--tokens", type=int, nargs="+", help="Non-padding sequence positions for curves")
    parser.add_argument("--branch", choices=["conditional", "unconditional"], default="conditional")
    parser.add_argument("--output-dir", type=Path, help="Default: visualizations/<sample name>/<branch>")
    parser.add_argument("--image-path", type=Path, help="Final generated image location if it moved")
    parser.add_argument("--color-scale", choices=["log", "linear"], default="log")
    parser.add_argument("--trace-scale", choices=["log", "linear"], default="log")
    return parser.parse_args()


# Helpers for steps 1-2: read and select saved data

def load_sample(path, branch="conditional"):
    """Check axes and token alignment before interpreting any saved values."""
    sample = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(sample, dict) or sample.get("schema_version") != 3
            or sample.get("score_name") != "token_decoder_update_l2_v2"):
        raise ValueError("Expected schema-v3 generation token-update data")
    if "branches" not in sample or sample.get("score_axes") != ["layer", "step", "token"]:
        raise ValueError("Use aggregate sample_XXXXXX.pt, not an individual step/feature file")
    metadata = sample["sample"]
    if metadata.get("task") != "generation" or metadata.get("phase") != "denoising":
        raise ValueError("Expected a generation denoising trajectory")
    if branch not in sample["branches"]:
        raise ValueError(f"Branch {branch!r} was not recorded. Available: {list(sample['branches'])}")
    for name in ["layer_indices", "step_indices"]:
        values = sample[name]
        if (not isinstance(values, list) or not values
                or any(type(value) is not int or value < 0 for value in values)
                or values != sorted(set(values))):
            raise ValueError(f"{name} must contain unique increasing nonnegative IDs")
    times = np.asarray(sample["denoising_timesteps"], dtype=float)
    if (times.shape != (len(sample["step_indices"]),) or not np.isfinite(times).all()
            or (times < 0).any() or (times > 1).any() or (np.diff(times) <= 0).any()):
        raise ValueError("Timesteps must match the recorded forward trajectory in increasing order")

    data = sample["branches"][branch]
    values = data["update_l2_tensor"]
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise ValueError("Expected a floating-point score tensor")
    scores = values.float().numpy()
    tokens = data["tokens"]
    if not tokens or scores.shape != (len(sample["layer_indices"]), len(times), len(tokens)):
        raise ValueError("Scores must have shape [recorded layers, recorded steps, token positions]")
    if metadata["sequence_length"] != len(tokens):
        raise ValueError("Sequence length does not match token metadata")
    if [token["sequence_position"] for token in tokens] != list(range(len(tokens))):
        raise ValueError("Token metadata must preserve the original sequence-position order")
    if any(token["token_type"] not in TYPE_COLORS for token in tokens):
        raise ValueError("Unknown generation token type")
    valid = data["valid_token_mask"]
    expected = torch.tensor([token["token_type"] != "padding" for token in tokens])
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or not torch.equal(valid, expected):
        raise ValueError("valid_token_mask must identify exactly the non-padding positions")
    if not valid.any() or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Require non-padding tokens and finite nonnegative L2 scores")
    visual = [token["sequence_position"] for token in tokens if token["token_type"] == "visual"]
    if data["visual_token_positions"].tolist() != visual or len(visual) != metadata["num_visual_tokens"]:
        raise ValueError("Visual positions must agree with the token metadata")
    return sample, scores


def choose_indices(saved, requested, option):
    """Arguments are recorded IDs, even when the saved tensor has sparse axes."""
    selected = requested if requested is not None else [saved[0], saved[len(saved) // 2], saved[-1]]
    if any(value not in saved for value in selected):
        raise ValueError(f"{option} must select recorded IDs: {saved}")
    return list(dict.fromkeys(selected))


def choose_tokens(tokens, requested):
    """Default curves show the first position of each type, not a top-score ranking."""
    valid = [token["sequence_position"] for token in tokens if token["token_type"] != "padding"]
    if requested is not None:
        if any(position not in valid for position in requested):
            raise ValueError("--tokens must select saved non-padding sequence positions")
        return list(dict.fromkeys(requested))
    first_per_type = {}
    for position in valid:
        first_per_type.setdefault(tokens[position]["token_type"], position)
    return list(first_per_type.values())


def load_final_image(sample_file, metadata, override=None):
    """Use the exact generated image; also support a relocated adjacent PNG."""
    if override is not None:
        candidates = [override]
    else:
        candidates = [sample_file.with_suffix(".png")]
        if metadata.get("image_path"):
            candidates.append(Path(metadata["image_path"]))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        if override is not None:
            raise FileNotFoundError(override)
        print("Final image unavailable: saving patch maps alone. Use --image-path to locate it.")
        return None
    for path in existing:
        if hashlib.sha256(path.read_bytes()).hexdigest() == metadata.get("image_sha256"):
            with Image.open(path) as image:
                if image.width * metadata["grid_height"] != image.height * metadata["grid_width"]:
                    raise ValueError("Generated image aspect ratio differs from the saved patch grid")
                return image.convert("RGB")
    raise ValueError("Image hash differs from this generated sample; select the matching final image")


def visual_patch_grids(scores, tokens, metadata):
    """Map [layer, step, token] to [layer, step, patch row, patch column]."""
    height, width = metadata["grid_height"], metadata["grid_width"]
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("Require a positive integer patch-grid height and width")
    visual = [token for token in tokens if token["token_type"] == "visual"]
    coordinates = [(token["patch_row"], token["patch_col"]) for token in visual]
    if len(visual) != height * width or set(coordinates) != {(r, c) for r in range(height) for c in range(width)}:
        raise ValueError("Visual tokens must cover each saved patch coordinate exactly once")
    grids = np.empty((*scores.shape[:2], height, width), dtype=scores.dtype)
    for token in visual:
        grids[:, :, token["patch_row"], token["patch_col"]] = scores[:, :, token["sequence_position"]]
    return grids


# Helpers for steps 3-5: the four plot types

def figure_title(metadata, branch, detail):
    prompt = metadata.get("source_record", {}).get("prompt", "")
    prompt = textwrap.shorten(" ".join(prompt.split()), width=130, placeholder="...")
    return f"Generation | {branch} | {detail}\nPrompt: {prompt}"


def plot_token_heatmap(matrix, row_labels, tokens, positions, ylabel, title, norm):
    """The same plot structure supports layers x tokens and steps x tokens."""
    fig, (type_ax, ax) = plt.subplots(
        2, 1, figsize=(17, 8), sharex=True,
        gridspec_kw={"height_ratios": [0.3, 7]}, layout="constrained",
    )
    kinds = list(TYPE_COLORS)
    type_ids = [kinds.index(tokens[position]["token_type"]) for position in positions]
    type_ax.imshow([type_ids], aspect="auto", interpolation="nearest",
                   cmap=ListedColormap(list(TYPE_COLORS.values())), vmin=-0.5, vmax=len(kinds) - 0.5)
    type_ax.set(yticks=[], ylabel="Type")
    type_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    present = {tokens[position]["token_type"] for position in positions}
    type_ax.legend(handles=[Patch(color=TYPE_COLORS[kind], label=kind) for kind in kinds if kind in present],
                   loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False)
    heatmap = ax.imshow(matrix[:, positions], aspect="auto", interpolation="nearest", cmap="viridis", norm=norm)
    row_ticks = np.unique(np.linspace(0, len(row_labels) - 1, min(28, len(row_labels)), dtype=int))
    ax.set_yticks(row_ticks, labels=[row_labels[index] for index in row_ticks])
    ticks = np.unique(np.linspace(0, len(positions) - 1, min(12, len(positions)), dtype=int))
    ax.set_xticks(ticks, labels=[positions[index] for index in ticks])
    ax.set(xlabel="Token sequence position (padding excluded)", ylabel=ylabel)
    fig.colorbar(heatmap, ax=[type_ax, ax], label=SCORE_LABEL, pad=0.02, format=FormatStrFormatter("%g"))
    fig.suptitle(title, fontsize=13)
    return fig


def plot_token_traces(scores, steps, times, tokens, positions, title, scale):
    """Track fixed positions through all recorded steps of one layer."""
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    for position in positions:
        ax.plot(times, scores[:, position], marker="o", markersize=3, label=token_label(tokens[position]))
    if scale == "log":
        ax.set_yscale("symlog", linthresh=1.0)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    largest = float(scores[:, positions].max())
    ax.set_ylim(0, largest * 1.15 if largest > 0 else 1.0)
    ticks = np.unique(np.linspace(0, len(steps) - 1, min(9, len(steps)), dtype=int))
    ax.set_xticks([times[index] for index in ticks], labels=[f"{steps[index]}\n{times[index]:.3f}" for index in ticks])
    ax.set(xlabel="Recorded step / model timestep t (horizontal spacing follows t)", ylabel=SCORE_LABEL)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.suptitle(title, fontsize=13)
    return fig


def plot_visual_patches(grids, layers, steps, times, selected_layers, selected_steps, image, title, norm):
    """Compare patch maps; the optional image is final-output context, not an intermediate state."""
    offset = int(image is not None)
    fig = plt.figure(figsize=(4 * (len(selected_steps) + offset), 3.6 * len(selected_layers)), layout="constrained")
    layout = fig.add_gridspec(len(selected_layers), len(selected_steps) + offset)
    if image is not None:
        image_ax = fig.add_subplot(layout[:, 0])
        image_ax.imshow(image)
        image_ax.set_title("Final generated image\n(context only)")
        image_ax.set_axis_off()
    axes = []
    for row, layer in enumerate(selected_layers):
        for column, step in enumerate(selected_steps):
            ax = fig.add_subplot(layout[row, column + offset])
            si = steps.index(step)
            ax.imshow(grids[layers.index(layer), si], cmap="viridis", norm=norm, interpolation="nearest")
            ax.set(title=f"Layer {layer} | step {step}\nt = {times[si]:.4f}", xticks=[], yticks=[])
            axes.append(ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes, label=SCORE_LABEL,
                 shrink=0.8, format=FormatStrFormatter("%g"))
    fig.suptitle(title + "\nMaps describe the recorded noisy states; one shared visual-score scale.", fontsize=12)
    return fig


def main():
    # --- 1. Read one saved trajectory and select its guidance branch ---
    args = parse_arguments()
    sample, scores = load_sample(args.sample_file, args.branch)
    data, metadata = sample["branches"][args.branch], sample["sample"]
    layers, steps, times = sample["layer_indices"], sample["step_indices"], sample["denoising_timesteps"]
    tokens = data["tokens"]
    print(f"Loaded {args.branch}: {scores.shape} = [layers, recorded steps, token positions]")

    # --- 2. Select IDs, exclude padding, and prepare shared color scales ---
    selected_steps = choose_indices(steps, args.steps, "--steps")
    selected_layers = choose_indices(layers, args.layers, "--layers")
    positions = np.flatnonzero(data["valid_token_mask"].numpy()).tolist()
    trace_positions = choose_tokens(tokens, args.tokens)
    grids = visual_patch_grids(scores, tokens, metadata)
    image = load_final_image(args.sample_file, metadata, args.image_path)
    # Stable scales across the complete recorded branch, including non-displayed steps.
    token_norm = make_color_norm(scores[:, :, positions], args.color_scale)
    visual_norm = make_color_norm(grids, args.color_scale)
    output_dir = args.output_dir or args.sample_file.parent / "visualizations" / args.sample_file.stem / args.branch
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "text.parse_math": False})
    files = []

    # --- 3. Plot layers x tokens once per selected denoising step ---
    for step in selected_steps:
        si = steps.index(step)
        title = figure_title(metadata, args.branch, f"Layer x token | step {step}, t={times[si]:.4f} | {args.color_scale} colors")
        fig = plot_token_heatmap(scores[:, si, :], layers, tokens, positions, "Decoder layer index", title, token_norm)
        filename = f"layer_token_step_{step:04d}.png"
        save_figure(fig, output_dir, filename)
        files.append(filename)

    # --- 4. Plot all recorded steps and token curves at each selected layer ---
    for layer in selected_layers:
        li = layers.index(layer)
        title = figure_title(metadata, args.branch, f"Step x token | layer {layer} | {args.color_scale} colors")
        labels = [f"{step} | {t:.3f}" for step, t in zip(steps, times)]
        fig = plot_token_heatmap(scores[li], labels, tokens, positions, "Recorded step | timestep t", title, token_norm)
        filename = f"step_token_layer_{layer:03d}.png"
        save_figure(fig, output_dir, filename)
        files.append(filename)
        title = figure_title(metadata, args.branch, f"Fixed-token curves | layer {layer} | {args.trace_scale} y-axis")
        fig = plot_token_traces(scores[li], steps, times, tokens, trace_positions, title, args.trace_scale)
        filename = f"token_traces_layer_{layer:03d}.png"
        save_figure(fig, output_dir, filename)
        files.append(filename)

    # --- 5. Compare visual patch maps across selected layers and steps ---
    title = figure_title(metadata, args.branch, f"Visual patch updates | {args.color_scale} colors")
    fig = plot_visual_patches(grids, layers, steps, times, selected_layers, selected_steps, image, title, visual_norm)
    save_figure(fig, output_dir, "visual_patch_heatmaps.png")
    files.append("visual_patch_heatmaps.png")

    # --- 6. Record the selections and plotting scales ---
    settings = {
        "sample_file": str(args.sample_file.resolve()), "sample_sha256": hashlib.sha256(args.sample_file.read_bytes()).hexdigest(),
        "branch": args.branch, "selected_steps": selected_steps, "selected_layers": selected_layers,
        "trace_token_positions": trace_positions, "padding_excluded": True,
        "recorded_step_indices": steps, "recorded_timesteps": times,
        "color_scale": args.color_scale, "trace_scale": args.trace_scale,
        "log_linear_threshold": 1.0, "token_color_max": token_norm.vmax, "visual_color_max": visual_norm.vmax,
        "scale_domain": "entire recorded branch; token maps exclude padding, patch maps use visual tokens only",
        "final_image_shown": image is not None, "files": files,
    }
    (output_dir / "plot_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
