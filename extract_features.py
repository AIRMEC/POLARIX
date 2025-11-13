import time
import os
import h5py
import numpy as np
import openslide
import torch
import timm
from PIL import ImageDraw
from shapely.geometry import Polygon
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from huggingface_hub import login

from heatmap_utils import create_tissue_mask


def generate_tiles(
    tile_width_pix, tile_height_pix, img_width, img_height, offsets=[(0, 0)]
):
    range_stop_width = int(np.ceil(img_width + tile_width_pix))
    range_stop_height = int(np.ceil(img_height + tile_height_pix))

    rects = []
    for xmin, ymin in offsets:
        cols = range(int(np.floor(xmin)), range_stop_width, tile_width_pix)
        rows = range(int(np.floor(ymin)), range_stop_height, tile_height_pix)
        for x in cols:
            for y in rows:
                rect = Polygon(
                    [
                        (x, y),
                        (x + tile_width_pix, y),
                        (x + tile_width_pix, y - tile_height_pix),
                        (x, y - tile_height_pix),
                    ]
                )
                rects.append(rect)
    return rects


def make_tile_QC_fig(tiles, slide, level, line_width_pix=1, extra_tiles=None):
    """
    Creates a quality control image with tile boundaries drawn on the slide.
    """
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    downsample = 1 / slide.level_downsamples[level]

    draw = ImageDraw.Draw(img, "RGBA")
    for tile in tiles:
        bbox = tuple(np.array(tile.bounds) * downsample)
        draw.rectangle(bbox, outline="lightgreen", width=line_width_pix)

    if extra_tiles:
        for tile in extra_tiles:
            bbox = tuple(np.array(tile.bounds) * downsample)
            draw.rectangle(bbox, outline="blue", width=line_width_pix + 1)

    return img


def create_tissue_tiles(
    wsi, tissue_mask_scaled, tile_size_microns, offsets_micron=None
):
    print(f"tile size is {tile_size_microns} um")
    assert (
        openslide.PROPERTY_NAME_MPP_X in wsi.properties
    ), "microns per pixel along X-dimension not available"
    assert (
        openslide.PROPERTY_NAME_MPP_Y in wsi.properties
    ), "microns per pixel along Y-dimension not available"

    mpp_x = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
    mpp_y = float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y])
    mpp_scale_factor = min(mpp_x, mpp_y)
    if mpp_x != mpp_y:
        print(
            f"mpp_x of {mpp_x} and mpp_y of {mpp_y} are not the same. Using smallest value: {mpp_scale_factor}"
        )

    tile_size_pix = round(tile_size_microns / mpp_scale_factor)
    tissue_margin_pix = tile_size_pix * 2
    minx, miny, maxx, maxy = tissue_mask_scaled.bounds
    min_offset_x = minx - tissue_margin_pix
    min_offset_y = miny - tissue_margin_pix
    offsets = [(min_offset_x, min_offset_y)]

    if offsets_micron is not None:
        assert (
            len(offsets_micron) > 0
        ), "offsets_micron needs to contain at least one value"
        offset_pix = [round(o / mpp_scale_factor) for o in offsets_micron]
        offsets = [(o + min_offset_x, o + min_offset_y) for o in offset_pix]

    all_tiles = generate_tiles(
        tile_size_pix,
        tile_size_pix,
        maxx + tissue_margin_pix,
        maxy + tissue_margin_pix,
        offsets=offsets,
    )

    filtered_tiles = [rect for rect in all_tiles if tissue_mask_scaled.intersects(rect)]

    return filtered_tiles


def tile_is_not_empty(tile, threshold_white=20):
    """
    Checks if a tile is not empty by analyzing its color histogram.
    """
    histogram = tile.histogram()
    whiteness_check = [0, 0, 0]
    for channel_id in (0, 1, 2):
        whiteness_check[channel_id] = np.median(
            histogram[256 * channel_id : 256 * (channel_id + 1)][100:200]
        )

    if all(c <= threshold_white for c in whiteness_check):
        return False
    return True


def crop_rect_from_slide(slide, rect):
    """
    Crops a rectangular region from a slide.
    """
    minx, miny, maxx, maxy = rect.bounds
    top_left_coords = (int(minx), int(miny))
    return slide.read_region(top_left_coords, 0, (int(maxx - minx), int(maxy - miny)))


class BagOfTiles(Dataset):
    """
    Dataset for loading and transforming tiles from a whole-slide image.
    """

    def __init__(self, wsi, tiles, transform):
        self.wsi = wsi
        self.tiles = tiles
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        img = crop_rect_from_slide(self.wsi, tile)
        is_tile_kept = tile_is_not_empty(img, threshold_white=20)
        img = img.convert("RGB")
        width, height = img.size
        assert width == height, "input image is not a square"

        img = self.transform(img).unsqueeze(0)
        coord = tile.bounds
        return img, coord, is_tile_kept


def collate_features(batch):
    """
    Collates features from a batch of tiles, filtering out empty ones.
    """
    imgs = [item[0] for item in batch if item[2]]
    coords = [item[1] for item in batch if item[2]]
    if len(imgs) == 0:
        return None, None
    img = torch.cat(imgs, dim=0)
    coords = np.stack(coords, axis=0)
    return img, coords


def write_to_h5(file, asset_dict):
    for key, val in asset_dict.items():
        if key not in file:
            maxshape = (None,) + val.shape[1:]
            dset = file.create_dataset(
                key, shape=val.shape, maxshape=maxshape, dtype=val.dtype
            )
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + val.shape[0], axis=0)
            dset[-val.shape[0] :] = val


def load_encoder(device, hf_token=None):
    """
    Loads the feature extraction model from Hugging Face Hub.
    """
    if hf_token is None:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    if hf_token:
        login(token=hf_token, add_to_git_credential=True)

    print("Loading model...")

    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    model.to(device)
    model.eval()

    return model


def extract_features(model, device, wsi, filtered_tiles, workers, batch_size):
    """
    Extracts features from tiles using the provided model.
    """
    using_cuda = device.type == "cuda"
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )
    print(transform)

    loader = DataLoader(
        dataset=BagOfTiles(wsi, filtered_tiles, transform),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate_features,
        pin_memory=using_cuda,
        persistent_workers=(workers > 0),
        prefetch_factor=2 if workers > 0 else None,
    )

    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=using_cuda
    ):
        with torch.inference_mode():
            for batch, coords in loader:
                if batch is None:
                    continue
                batch = batch.to(device, non_blocking=True)
                output = model(batch).cpu().numpy()
                assert output.shape == (batch.shape[0], 1536), output.shape
                assert coords.shape == (batch.shape[0], 4), coords.shape
                yield output, coords


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing script")

    parser.add_argument(
        "--slide",
        type=str,
        help="Path to input WSI file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token. If omitted, the HUGGINGFACE_TOKEN environment variable or existing login will be used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--tile_size",
        help="Desired tile size in microns (should be the same value as used in feature extraction model).",
        type=int,
        default=180
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loader. Only relevant when using a GPU.",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    # PREPARE OUTPUT FILE PATHS AND CHECK EXISTENCE
    slide_id, _ = os.path.splitext(os.path.basename(args.slide))
    wip_file_path = os.path.join(args.output_dir, slide_id + "_wip.h5")
    output_file_path = os.path.join(args.output_dir, slide_id + "_features.h5")
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(output_file_path):
        raise Exception(f"{output_file_path} already exists")

    # OPEN SLIDE AND SELECT SEGMENTATION LEVEL
    wsi = openslide.open_slide(args.slide)
    seg_level = wsi.get_best_level_for_downsample(64)

    # SEGMENTATION AND TILING PROCEDURE
    start_time = time.time()
    tissue_mask_scaled = create_tissue_mask(wsi, seg_level)
    filtered_tiles = create_tissue_tiles(wsi, tissue_mask_scaled, args.tile_size)

    # QUALITY CONTROL (QC) FIGURE GENERATION
    qc_img = make_tile_QC_fig(filtered_tiles, wsi, seg_level, 2)
    qc_img_target_width = 1920
    qc_img = qc_img.resize(
        (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
    )
    print(
        f"Finished creating {len(filtered_tiles)} tissue tiles in {time.time() - start_time}s"
    )

    # FEATURE EXTRACTION MODEL INITIALIZATION
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Are we using CPU or GPU? --> {device}")

    model = load_encoder(device=device, hf_token=args.hf_token)

    generator = extract_features(
        model, device, wsi, filtered_tiles, args.workers, args.batch_size
    )

    start_time = time.time()
    count_features = 0
    with h5py.File(wip_file_path, "w") as file:
        for i, (features, coords) in enumerate(generator):
            count_features += features.shape[0]
            write_to_h5(file, {"features": features, "coords": coords})
            print(
                f"Processed batch {i}. Extracted features from {count_features}/{len(filtered_tiles)} tiles in {(time.time() - start_time):.2f}s."
            )

    # FINALIZE AND SAVE OUTPUT
    os.rename(wip_file_path, output_file_path)
    qc_img_file_path = os.path.join(
        args.output_dir, f"{slide_id}_{count_features}_features_QC.png"
    )
    qc_img.save(qc_img_file_path)
    print(
        f"Finished extracting {count_features} features in {(time.time() - start_time):.2f}s"
    )
