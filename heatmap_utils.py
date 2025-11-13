import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from shapely.affinity import scale
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from POLARIX import INPUT_FEATURE_SIZE, POLARIX


def segment_tissue(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mthresh = 7
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    _, img_prepped = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_prepped = cv2.morphologyEx(img_prepped, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        img_prepped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy


def detect_foreground(contours, hierarchy):
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    foreground_contours = [contours[cont_idx] for cont_idx in hierarchy_1]

    all_holes = []
    for cont_idx in hierarchy_1:
        all_holes.append(np.flatnonzero(hierarchy[:, 1] == cont_idx))

    hole_contours = []
    for hole_ids in all_holes:
        holes = [contours[idx] for idx in hole_ids]
        hole_contours.append(holes)

    return foreground_contours, hole_contours


def construct_polygon(foreground_contours, hole_contours, min_area):
    polys = []
    for foreground, holes in zip(foreground_contours, hole_contours):
        if len(foreground) < 3:
            continue
        poly = Polygon(np.squeeze(foreground))
        if poly.area < min_area:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        for hole_contour in holes:
            if len(hole_contour) < 3:
                continue
            hole = Polygon(np.squeeze(hole_contour))
            if not hole.is_valid:
                continue

            if hole.area < min_area:
                continue
            poly = poly.difference(hole)
        polys.append(poly)
    if len(polys) == 0:
        raise Exception("Raw tissue mask consists of 0 polygons")
    return unary_union(polys)


def create_tissue_mask(wsi, seg_level):
    level_dims = wsi.level_dimensions[seg_level]
    img = np.array(wsi.read_region((0, 0), seg_level, level_dims))
    level_area = level_dims[0] * level_dims[1]
    min_area = level_area / 500

    contours, hierarchy = segment_tissue(img)
    foreground_contours, hole_contours = detect_foreground(contours, hierarchy)
    tissue_mask = construct_polygon(foreground_contours, hole_contours, min_area)
    scale_factor = wsi.level_downsamples[seg_level]
    tissue_mask_scaled = scale(
        tissue_mask, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)
    )

    return tissue_mask_scaled


def get_display_image(wsi, display_level):
    assert display_level <= (len(wsi.level_dimensions) - 1)
    display_image = wsi.read_region(
        (0, 0), display_level, wsi.level_dimensions[display_level]
    )

    scale_factor = 1 / wsi.level_downsamples[display_level]
    return display_image, scale_factor


def predict_attention_matrix(model, feature_flattened):
    with torch.inference_mode():
        logits, Y_prob, A_raw, m = model(feature_flattened)
    return A_raw.cpu().numpy(), Y_prob, logits


def standardize_scores(raw):
    z_scores = (raw - np.mean(raw)) / np.std(raw)
    z_scores_s = z_scores + np.abs(np.min(z_scores))
    z_scores_s /= np.max(z_scores_s)
    return z_scores_s


def scale_rectangles(raw_rect_bounds, scale_factor):
    rects = []
    for coords in raw_rect_bounds:
        minx, miny, maxx, maxy = coords
        rect = box(minx, miny, maxx, maxy)
        rect_scaled = scale(
            rect, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)
        )
        rects.append(rect_scaled)
    return rects


def build_scoremap(src_img, rect_shapes, scores):
    h, w, _ = np.asarray(src_img).shape
    score_map = np.zeros(dtype=np.float32, shape=(h, w))

    for rect, score in zip(rect_shapes, scores):
        minx, miny, maxx, maxy = rect.bounds
        score_map[round(miny) : round(maxy), round(minx) : round(maxx)] = score

    return score_map


def scoremap_to_heatmap(score_map, colorset=cv2.COLORMAP_JET):
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * score_map), colorset)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGBA)
    heatmap[..., 3] = 60
    heatmap[np.where(score_map == 0)] = (255, 255, 255, 0)
    return Image.fromarray(heatmap, mode="RGBA")


def get_tile(slide, rect):
    minx, miny, maxx, maxy = rect
    tile = slide.read_region(
        (int(minx), int(miny)), 0, (int(maxx - minx), int(maxy - miny))
    )
    return tile


def read_data(features_path):
    with h5py.File(features_path, "r") as hdf5_file:
        features = hdf5_file["features"][:]
        coords = hdf5_file["coords"][:]

    print("Feature shape: ", features.shape)
    return features, coords
