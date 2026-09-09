"""Plot saved understanding-prefill scores for one example; no model inference.

Run from the repository root:
    python -m evaluation.visualize_token_importance_understanding \
        token_prefill_pilot/sample_000000.pt

All figures use the saved raw layer-update L2 magnitudes. A larger update
does not by itself establish a token's effect on answer quality.
"""

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures in Colab or a terminal without a display.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import torch


TYPE_COLORS = {
    "system_text": "#0072B2",
    "question_text": "#009E73",
    "chat_format": "#777777",
    "image_boundary": "#D55E00",
    "visual": "#E69F00",
    "time": "#CC79A7",
}
SCORE_LABEL = "Layer-update L2 magnitude (raw)"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_file", type=Path, help="One schema-v2 sample_XXXXXX.pt file")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: beside the sample, in visualizations/<sample name>")
    parser.add_argument("--layers", type=int, nargs="+",
                        help="Layer IDs for distributions/patch maps; default: first, middle, last")
    parser.add_argument("--tokens", type=int, nargs="+",
                        help="Sequence positions to track; default: first token of each type")
    parser.add_argument("--image-path", type=Path, help="Original image location if it moved since collection")
    return parser.parse_args()


def load_sample(path):
    """Read saved scores and check their alignment with the layer/token metadata."""
    sample = torch.load(path, map_location="cpu", weights_only=True)
    if (sample.get("schema_version") != 2
            or sample.get("score_name") != "token_decoder_update_l2_v2"):
        raise ValueError("Expected schema-v2 all-token layer-update scores")
    if sample["sample"]["phase"] != "prefill" or sample["sample"]["task"] != "understanding":
        raise ValueError("Expected understanding-prefill data")
    scores = sample["update_l2_matrix"].float().numpy()
    layers, tokens = sample["layer_indices"], sample["tokens"]
    if not layers or not tokens or scores.shape != (len(layers), len(tokens)):
        raise ValueError("Score matrix must have one row per layer and one column per token")
    if layers != sorted(set(layers)):
        raise ValueError("Saved layer indices must be unique and increasing")
    positions = list(range(len(tokens)))
    metadata_positions = [token["sequence_position"] for token in tokens]
    if sample["token_positions"].tolist() != positions or metadata_positions != positions:
        raise ValueError("Token metadata is not in sequence-position order")
    if any(t["token_type"] not in TYPE_COLORS for t in tokens):
        raise ValueError("Unknown token type")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Scores must be finite, nonnegative L2 magnitudes")
    return sample, scores


def choose_positions(tokens, requested):
    """Choose stable sequence positions; defaults are examples, not a ranking."""
    if requested is not None:
        if any(position < 0 or position >= len(tokens) for position in requested):
            raise ValueError(f"--tokens must be sequence positions between 0 and {len(tokens) - 1}")
        return list(dict.fromkeys(requested))
    first_per_type = {}
    for token in tokens:
        first_per_type.setdefault(token["token_type"], token["sequence_position"])
    return list(first_per_type.values())


def choose_layers(saved_layers, requested):
    selected = requested
    if selected is None:
        selected = [saved_layers[0], saved_layers[len(saved_layers) // 2], saved_layers[-1]]
    if any(layer not in saved_layers for layer in selected):
        raise ValueError(f"--layers must be recorded layer IDs: {saved_layers}")
    return list(dict.fromkeys(selected))


def token_label(token):
    position, kind = token["sequence_position"], token["token_type"]
    if kind == "visual":
        detail = f"patch ({token['patch_row']}, {token['patch_col']})"
    else:
        detail = repr(token["token_piece"] or kind)
    return f"pos {position} | {kind} | {detail[:45]}"


def save_figure(fig, output_dir, filename):
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_layer_token_heatmap(scores, layers, tokens):
    """Rows are saved layers; columns retain the complete input sequence order."""
    fig, (type_ax, ax) = plt.subplots(
        2, 1, figsize=(17, 9), sharex=True,
        gridspec_kw={"height_ratios": [0.35, 8]}, layout="constrained",
    )
    kinds = list(TYPE_COLORS)
    type_ids = [kinds.index(token["token_type"]) for token in tokens]
    type_ax.imshow([type_ids], aspect="auto", interpolation="nearest",
                   cmap=ListedColormap(list(TYPE_COLORS.values())), vmin=-0.5, vmax=len(kinds) - 0.5)
    type_ax.set_yticks([])
    type_ax.set_ylabel("Type")
    type_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    present = {token["token_type"] for token in tokens}
    legend = [Patch(color=TYPE_COLORS[kind], label=kind) for kind in kinds if kind in present]
    type_ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.3),
                   ncol=3, frameon=False)
    heatmap = ax.imshow(scores, aspect="auto", interpolation="nearest", cmap="viridis",
                        vmin=0, vmax=max(float(scores.max()), 1e-12))
    ax.set_yticks(range(len(layers)), labels=layers)
    ax.set_xticks(np.unique(np.linspace(0, len(tokens) - 1, min(12, len(tokens)), dtype=int)))
    ax.set(xlabel="Token sequence position", ylabel="Decoder layer index")
    fig.colorbar(heatmap, ax=[type_ax, ax], label=SCORE_LABEL, pad=0.02)
    fig.suptitle("Understanding prefill | Layer x token", fontsize=15)
    return fig


def plot_token_traces(scores, layers, tokens, positions):
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    for position in positions:
        ax.plot(layers, scores[:, position], marker="o", markersize=3,
                label=token_label(tokens[position]))
    ax.set(xlabel="Decoder layer index", ylabel=SCORE_LABEL,
           title="Same token positions across recorded layers", ylim=(0, None))
    ax.set_xticks(layers)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    return fig


def plot_type_distributions(scores, layers, tokens, selected_layers):
    """Summarize existing scores within each type and layer; no new labels."""
    kinds = [kind for kind in TYPE_COLORS if any(t["token_type"] == kind for t in tokens)]
    positions_by_type = [
        [i for i, token in enumerate(tokens) if token["token_type"] == kind]
        for kind in kinds
    ]
    fig, axes = plt.subplots(len(selected_layers), 1, figsize=(11, 3.5 * len(selected_layers)),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for ax, layer in zip(axes[:, 0], selected_layers):
        row = scores[layers.index(layer)]
        boxes = ax.boxplot([row[positions] for positions in positions_by_type], patch_artist=True,
                           medianprops={"color": "black", "linewidth": 1.4},
                           flierprops={"markersize": 3, "alpha": 0.4})
        for box, kind in zip(boxes["boxes"], kinds):
            box.set_facecolor(TYPE_COLORS[kind])
            box.set_alpha(0.65)
        ax.set(title=f"Layer {layer}", ylabel="Raw L2 magnitude")
        ax.grid(axis="y", alpha=0.2)
    # Set the shared limit after including every displayed layer.
    largest = float(scores[[layers.index(layer) for layer in selected_layers]].max())
    axes[0, 0].set_ylim(0, largest * 1.08 if largest > 0 else 1.0)
    labels = [f"{kind}\n(n={len(positions)})"
              for kind, positions in zip(kinds, positions_by_type)]
    axes[-1, 0].set_xticks(range(1, len(kinds) + 1), labels=labels)
    fig.suptitle("Scores by token type | Boxes: middle 50%; line: median", fontsize=14)
    return fig


def load_processed_image(metadata, override_path=None):
    """Match the collector's PIL Resize(BICUBIC) and CenterCrop geometry."""
    path = override_path if override_path is not None else Path(metadata["image_path"])
    if not path.is_file():
        if override_path is not None:
            raise FileNotFoundError(path)
        print("Original image unavailable: saving patch maps without an overlay. "
              "Use --image-path to locate it.")
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["image_sha256"]:
        raise ValueError("Image hash differs from the collected example; choose the original image")
    size = int(metadata["processed_resolution"])
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= height:
            resized = (size, int(size * height / width))
        else:
            resized = (int(size * width / height), size)
        image = image.resize(resized, Image.Resampling.BICUBIC)
        left = int(round((image.width - size) / 2.0))
        top = int(round((image.height - size) / 2.0))
        return image.crop((left, top, left + size, top + size))


def visual_patch_grids(scores, tokens, metadata):
    """Place saved visual scores using patch coordinates, not sequence offsets."""
    height, width = metadata["grid_height"], metadata["grid_width"]
    visual = [token for token in tokens if token["token_type"] == "visual"]
    coordinates = [(token["patch_row"], token["patch_col"]) for token in visual]
    expected = {(row, col) for row in range(height) for col in range(width)}
    if len(visual) != height * width or set(coordinates) != expected:
        raise ValueError("Visual tokens must cover every saved patch coordinate exactly once")
    grids = np.empty((len(scores), height, width), dtype=scores.dtype)
    for token in visual:
        grids[:, token["patch_row"], token["patch_col"]] = scores[:, token["sequence_position"]]
    return grids


def plot_visual_patches(grids, layers, selected_layers, image):
    rows = [layers.index(layer) for layer in selected_layers]
    norm = Normalize(vmin=0, vmax=max(float(grids[rows].max()), 1e-12))
    count = len(rows) + int(image is not None)
    fig, axes = plt.subplots(1, count, figsize=(4.5 * count, 4.8),
                             squeeze=False, layout="constrained")
    axes = axes[0]
    offset = int(image is not None)
    height, width = grids.shape[1:]
    extent = (-0.5, width - 0.5, height - 0.5, -0.5)
    if image is not None:
        axes[0].imshow(image, extent=extent)
        axes[0].set_title("Resized and center-cropped input")
    for ax, row, layer in zip(axes[offset:], rows, selected_layers):
        if image is not None:
            ax.imshow(image, extent=extent)
        ax.imshow(grids[row], cmap="viridis", norm=norm, interpolation="nearest", extent=extent,
                  alpha=0.55 if image is not None else 1.0)
        ax.set_title(f"Layer {layer}")
    for ax in axes:
        ax.set(xticks=[], yticks=[])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes[offset:]),
                 label=SCORE_LABEL, shrink=0.8)
    fig.suptitle("Visual tokens | Shared raw score scale", fontsize=14)
    return fig


def main():
    # --- 1. Read the saved scores ---
    args = parse_arguments()
    sample, scores = load_sample(args.sample_file)
    layers, tokens = sample["layer_indices"], sample["tokens"]
    print(f"Loaded {args.sample_file}: {scores.shape[0]} layers x {scores.shape[1]} tokens")

    # --- 2. Select layers/tokens and locate the original image ---
    selected_layers = choose_layers(layers, args.layers)
    positions = choose_positions(tokens, args.tokens)
    image = load_processed_image(sample["sample"], args.image_path)
    grids = visual_patch_grids(scores, tokens, sample["sample"])
    output_dir = args.output_dir or args.sample_file.parent / "visualizations" / args.sample_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "text.parse_math": False})

    # --- 3. Plot every saved layer and token ---
    fig = plot_layer_token_heatmap(scores, layers, tokens)
    save_figure(fig, output_dir, "layer_token_heatmap.png")

    # --- 4. Track selected token positions through the layers ---
    fig = plot_token_traces(scores, layers, tokens, positions)
    save_figure(fig, output_dir, "token_traces.png")

    # --- 5. Compare token types within selected layers ---
    fig = plot_type_distributions(scores, layers, tokens, selected_layers)
    save_figure(fig, output_dir, "token_type_distributions.png")

    # --- 6. Map visual scores onto image-patch positions ---
    fig = plot_visual_patches(grids, layers, selected_layers, image)
    save_figure(fig, output_dir, "visual_patch_heatmaps.png")


if __name__ == "__main__":
    main()
