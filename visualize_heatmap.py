import os
import cv2
import numpy as np
import openslide
import torch
from PIL import Image
from shapely.geometry import Polygon
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


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    print(f"Predicting attention map for {args.input_slide}")

    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
    slide_dir = os.path.join(output_root, slide_id)
    os.makedirs(slide_dir, exist_ok=True)

    if not os.path.isfile(args.checkpoint_POLARIX_model):
        raise Exception(f"checkpoint {args.checkpoint_POLARIX_model} is not a file")
    print("loading checkpoints '{}'".format(args.checkpoint_POLARIX_model))

    if not os.path.isfile(args.features_path):
        raise Exception(f"feature bag {args.features_path} is not a file")

    model = load_trained_model(device, args.checkpoint_POLARIX_model)

    model = model.to(device)

    wsi = openslide.open_slide(args.input_slide)

    display_level = min(args.display_level, len(wsi.level_dimensions) - 1)
    display_image, scale_factor = get_display_image(wsi, display_level)
    features, coords = read_data(args.features_path)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()

    device = next(model.parameters()).device
    features = features.to(device)

    A_raw, Y_prob, logits = predict_attention_matrix(model, features)
    print("Y_prob:", Y_prob)

    assert A_raw.shape[1] == len(coords)
    assert (
        A_raw.shape[1] == coords.shape[0]
    ), "Number of attention score sets is not the same as the number of tiles in the batch"

    raw_attn = A_raw[0]
    scaled_rects = scale_rectangles(coords, scale_factor)
    z_scores = standardize_scores(raw_attn)
    scoremap = build_scoremap(display_image, scaled_rects, z_scores)
    print(np.min(scoremap), np.max(scoremap))

    # apply Gaussian blur, kernel size depends on tile size and desired smoothness
    sigma = 180 / 8
    blurred_scoremap = cv2.GaussianBlur(
        scoremap, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma
    )
    # normalize again to [0,1]
    blurred_scoremap = (blurred_scoremap - blurred_scoremap.min()) / (
        blurred_scoremap.max() - blurred_scoremap.min()
    )

    colorset = cv2.COLORMAP_JET

    overlay = scoremap_to_heatmap(blurred_scoremap, colorset)
    display_image = display_image.convert("RGBA")
    result = Image.alpha_composite(display_image, overlay)
    outpath = os.path.join(slide_dir, f"{slide_id}_jet_blur.bmp")
    print(f"Exporting {outpath}")
    result_rgba = result.convert("RGBA")
    result_rgba.save(outpath)

    print("Finished Full attention map, now creating top-10 tiles...")

    top_indices = np.argsort(raw_attn)[::-1][:10]

    regions = []
    for i, idx in enumerate(top_indices):
        rect = coords[idx]

        if isinstance(rect, Polygon):
            rect = rect.bounds
        try:
            region = get_tile(wsi, rect).convert("RGB")
            top_tiles_dir = os.path.join(slide_dir, "top_tiles")
            os.makedirs(top_tiles_dir, exist_ok=True)
            outpath = os.path.join(top_tiles_dir, f"{slide_id}_top_tile_{i+1}.bmp")
            print(f"Exporting {outpath}")
            region.save(outpath)
            regions.append(region)
        except Exception as e:
            print(f"Skipping tile {i+1} due to error: {e}")

    print("Finished exporting top 10 attention tiles.")


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
        "--display_level",
        help="Control the resolution of the heatmap by selecting the level of the slide used for the background of the overlay",
        type=int,
        default=4,
    )

    args = parser.parse_args()
    main(args)
