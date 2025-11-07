import os
import numpy as np
import openslide
import torch
from PIL import Image
from PIL import ImageChops
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List
import joblib
from heatmap_utils import (
    build_scoremap,
    get_display_image,
    get_tile,
    load_trained_model,
    predict_attention_matrix,
    read_data,
    scale_rectangles,
    scoremap_to_heatmap,
    standardize_scores,
)


def tight_crop_image(img: Image.Image, bg_color=(255, 255, 255, 0)) -> Image.Image:
    """
    Crops out uniform background from a transparent or white-padded image. Works for RGBA or RGB heatmaps.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    bg = Image.new("RGBA", img.size, bg_color)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def create_composite_image(
    heatmap: Image.Image,
    top_tiles: List[Image.Image],
    predicted_class: str,
    slide_id: str,
    output_dir: str,
    figsize=(16, 32),
):
    """
    Generates a composite image containing the attention heatmap, top-10 most important tiles,
    and classification results for a given slide.
    """
    if len(top_tiles) < 10:
        print(
            f"Warning: expected 10 tiles for composite image but received {len(top_tiles)}. "
            "The layout will contain fewer tiles."
        )

    os.makedirs(output_dir, exist_ok=True)

    cropped_heatmap = tight_crop_image(heatmap)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(35, 5, figure=fig)

    if predicted_class == "POLARIX-mut":
        recommendation = "Perform Confirmatory Testing"
    else:
        recommendation = "Rule-out Case"

    ax_text = fig.add_subplot(gs[0:4, :])
    ax_text.axis("off")

    # Slide ID
    ax_text.text(
        0.5,
        1.0,  # Top-center
        f"Slide ID: {slide_id}",
        fontsize=40,
        ha="center",
        va="top",
        weight="bold",
        transform=ax_text.transAxes,
    )

    label_fontsize = 36
    spacing = 0.2  # vertical spacing

    lines = [("Prediction", f"{predicted_class}"), ("Recommendation", recommendation)]

    y_start = 0.70

    for i, (label, value) in enumerate(lines):
        ax_text.text(
            0.02,
            y_start - i * spacing,
            f"{label:<15}:  {value}",
            fontsize=label_fontsize,
            ha="left",
            va="top",
            transform=ax_text.transAxes,
        )

    ax_heatmap_title = fig.add_subplot(gs[5, :])
    ax_heatmap_title.axis("off")
    ax_heatmap_title.text(
        0.02,
        1.0,
        "Attention Heatmap",
        fontsize=30,
        ha="left",
        va="bottom",
        transform=ax_heatmap_title.transAxes,
    )

    # Section B: Cropped heatmap (rows 2–12)
    ax_heatmap = fig.add_subplot(gs[6:25, :])
    ax_heatmap.imshow(cropped_heatmap)
    ax_heatmap.axis("off")

    # Section C: Top tiles (2 rows × 5 columns, rows 14–19)
    ax_tiles_title = fig.add_subplot(gs[26, :])
    ax_tiles_title.axis("off")
    ax_tiles_title.text(
        0.02,
        1.0,
        "Top-10 Most Important Tiles",
        fontsize=30,
        ha="left",
        va="bottom",
        transform=ax_tiles_title.transAxes,
    )

    for i, tile in enumerate(top_tiles):
        row = 27 + (i // 5) * 3  # Tiles stacked closer
        col = i % 5
        ax_tile = fig.add_subplot(gs[row : row + 3, col])
        ax_tile.imshow(tile)
        ax_tile.axis("off")

    # Adjust layout: white margin around figure
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Margin on all sides
    out_path = os.path.join(output_dir, f"{slide_id}_composite_image.jpeg")
    fig.savefig(out_path, format="jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite image saved: {out_path}")


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    print(f"Predicting attention map for {args.input_slide}")

    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
    slide_dir = os.path.join(output_root, slide_id)
    os.makedirs(slide_dir, exist_ok=True)

    # Load the trained POLARIX model
    if not os.path.isfile(args.checkpoint_POLARIX_model):
        raise Exception(f"checkpoint {args.checkpoint_POLARIX_model} is not a file")
    print("loading checkpoints '{}'".format(args.checkpoint_POLARIX_model))

    model = load_trained_model(device, args.checkpoint_POLARIX_model)

    wsi = openslide.open_slide(args.input_slide)

    # Get the display image and scale factor
    display_level = min(args.display_level, len(wsi.level_dimensions) - 1)
    display_image, scale_factor = get_display_image(wsi, display_level)
    if not os.path.isfile(args.features_path):
        raise Exception(f"feature bag {args.features_path} is not a file")

    features, coords = read_data(args.features_path)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()

    device = next(model.parameters()).device
    features = features.to(device)

    # Predict the attention matrix, probability, and logits
    A_raw, Y_prob, logits = predict_attention_matrix(model, features)

    # Attention
    assert A_raw.shape[1] == len(coords)
    assert (
        A_raw.shape[1] == coords.shape[0]
    ), "Number of attention score sets is not the same as the number of tiles in the batch"

    # Generate and save the attention heatmap
    raw_attn = A_raw[0]
    scaled_rects = scale_rectangles(coords, scale_factor)
    z_scores = standardize_scores(raw_attn)
    scoremap = build_scoremap(display_image, scaled_rects, z_scores)
    overlay = scoremap_to_heatmap(scoremap)
    display_image = display_image.convert("RGBA")
    result = Image.alpha_composite(display_image, overlay)
    outpath = os.path.join(slide_dir, f"{slide_id}.png")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    print(f"Exporting {outpath}")
    result_rgba = result.convert("RGBA")
    result_rgba.save(outpath)

    print("Finished Full attention map, now creating top-ten tiles...")

    # Extract and save the top-10 most important tiles
    top_indices = np.argsort(raw_attn)[::-1][:10]

    regions = []
    for i, idx in enumerate(top_indices):
        rect = coords[idx]

        if isinstance(rect, Polygon):
            rect = rect.bounds
        try:
            region = get_tile(wsi, rect).convert("RGB")
            outpath = os.path.join(slide_dir, f"{slide_id}_top_tile_{i+1}.png")
            print(f"Exporting {outpath}")
            region.save(outpath)
            regions.append(region)
        except Exception as e:
            print(f"Skipping tile {i+1} due to error: {e}")

    print("Finished exporting top 10 attention tiles.")

    # Classify the slide and apply Platt scaling for calibrated probabilities
    threshold_classifier = args.threshold_classifier
    print("Threshold classifier: ", threshold_classifier)
    print("Y_prob: ", Y_prob)

    # Platt scaling
    platt_scaler = joblib.load(args.checkpoint_platt_model)
    logits_cpu = logits.detach().cpu().numpy().reshape(-1, 1)
    calprobs = platt_scaler.predict_proba(logits_cpu)[:, 1]
    print("Y_prob calibrated: ", calprobs)

    calibrated_prob = float(calprobs[0]) if calprobs.size else float("nan")
    if not np.isfinite(calibrated_prob):
        raise ValueError(
            "Calibrated probability is undefined (NaN or inf). "
            "Check Platt scaler inputs before proceeding."
        )

    if calibrated_prob < threshold_classifier:
        predicted_class = "POLARIX-wt"
    else:
        predicted_class = "POLARIX-mut"

    print("Predicted class: ", predicted_class)

    # Create a new high-resolution composite image
    create_composite_image(
        heatmap=result_rgba,
        top_tiles=regions,
        predicted_class=predicted_class,
        slide_id=slide_id,
        output_dir=slide_dir,
    )
    print("All done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attention heatmap generation script")
    parser.add_argument(
        "--input_slide",
        type=str,
        help="Path to input WSI file",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
        default="./results",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        help="Path to the precomputed feature bag (.h5) for this slide",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_POLARIX_model",
        type=str,
        help="Attention model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_platt_model",
        type=str,
        help="Platt model pkl",
        required=True,
    )
    parser.add_argument(
        "--display_level",
        help="Control the resolution of the heatmap by selecting the level of the slide used for the background of the overlay",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--threshold_classifier",
        help="Threshold for determining the class the slide is classified to.",
        type=float,
        default=0.005,
    )

    args = parser.parse_args()
    main(args)
