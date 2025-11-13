import json
import os
import cv2
import numpy as np
import openslide
import torch
from PIL import Image
from shapely.geometry import Polygon, box, mapping
from heatmap_utils import (
    build_scoremap,
    get_display_image,
    get_tile,
    predict_attention_matrix,
    read_data,
    scale_rectangles,
    scoremap_to_heatmap,
    standardize_scores,
)
from utils import load_trained_model


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    print(f"Predicting attention map for {args.slide}")

    slide_id, _ = os.path.splitext(os.path.basename(args.slide))
    slide_dir = os.path.join(output_root, slide_id)
    os.makedirs(slide_dir, exist_ok=True)

    if not os.path.isfile(args.checkpoint):
        raise Exception(f"checkpoint {args.checkpoint} is not a file")
    print("Loading checkpoint '{}'".format(args.checkpoint))

    if not os.path.isfile(args.features):
        raise Exception(f"feature bag {args.features} is not a file")

    model = load_trained_model(device, args.checkpoint)

    wsi = openslide.open_slide(args.slide)

    display_level = min(args.display_level, len(wsi.level_dimensions) - 1)
    display_image, scale_factor = get_display_image(wsi, display_level)
    features, coords = read_data(args.features)

    def rect_to_polygon(rect_like):
        """Ensure tile coordinates are represented as a shapely polygon."""
        if isinstance(rect_like, Polygon):
            return rect_like
        minx, miny, maxx, maxy = rect_like
        return box(minx, miny, maxx, maxy)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()

    features = features.to(device)

    print("Running inference on feature bag...")
    A_raw, Y_prob, _ = predict_attention_matrix(model, features)
    print("Y_prob:", Y_prob)

    assert A_raw.shape[1] == len(coords)
    assert (
        A_raw.shape[1] == coords.shape[0]
    ), "Number of attention score sets is not the same as the number of tiles in the batch"

    print("Normalizing attention scores...")
    raw_attn = A_raw[0]
    scaled_rects = scale_rectangles(coords, scale_factor)
    normed_attn = standardize_scores(raw_attn)
    scoremap = build_scoremap(display_image, scaled_rects, normed_attn)

    # apply Gaussian blur, kernel size depends on tile size and desired smoothness
    sigma = 180 / 8
    blurred_scoremap = cv2.GaussianBlur(
        scoremap, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma
    )
    # normalize again to [0,1]
    blurred_scoremap = (blurred_scoremap - blurred_scoremap.min()) / (
        blurred_scoremap.max() - blurred_scoremap.min()
    )

    print("Building heatmap...")
    overlay = scoremap_to_heatmap(blurred_scoremap, cv2.COLORMAP_JET)
    display_image = display_image.convert("RGBA")
    result = Image.alpha_composite(display_image, overlay)
    outpath = os.path.join(slide_dir, f"{slide_id}_jet_blur.png")
    print(f"Exporting {outpath}")
    result_rgba = result.convert("RGBA")
    result_rgba.save(outpath)

    geojson_path = os.path.join(slide_dir, f"{slide_id}_tiles.jsonl")
    print(f"Exporting {geojson_path}")
    with open(geojson_path, "w") as tiles_file:
        for attention, normed_attention, rect in zip(
            raw_attn.tolist(), normed_attn.tolist(), coords
        ):
            tile_polygon = rect_to_polygon(rect)
            feature = {
                "type": "Feature",
                "geometry": mapping(tile_polygon),
                "properties": {
                    "raw_attention": float(attention),
                    "normed_attention": float(normed_attention),
                },
            }
            tiles_file.write(json.dumps(feature) + "\n")

    print("Finished Full attention map, now creating top-10 tiles...")

    top_indices = np.argsort(raw_attn)[::-1][:10]

    regions = []
    for i, idx in enumerate(top_indices):
        tile_polygon = rect_to_polygon(coords[idx])
        rect_bounds = tile_polygon.bounds
        region = get_tile(wsi, rect_bounds).convert("RGB")
        top_tiles_dir = os.path.join(slide_dir, "top_tiles")
        os.makedirs(top_tiles_dir, exist_ok=True)
        outpath = os.path.join(top_tiles_dir, f"{slide_id}_top_tile_{i+1}.png")
        print(f"Exporting {outpath}")
        region.save(outpath)
        regions.append(region)

    print("Finished exporting top 10 attention tiles.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attention heatmap generation script")
    parser.add_argument(
        "--slide",
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
        "--features",
        type=str,
        help="Path to the precomputed feature bag (.h5) for this slide",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
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
