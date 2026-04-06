"""
Horizontal-Vertical (HV) map generation for nuclei instance segmentation.

HV maps encode the normalized distance of each nuclei pixel from its instance center.
This enables separation of touching nuclei during post-processing, inspired by Hover-Net.

For each nucleus instance:
  - Horizontal map: normalized x-distance from centroid, in [-1, 1]
  - Vertical map: normalized y-distance from centroid, in [-1, 1]

Reference: Graham et al., "Hover-Net", Med Image Anal, 2019.
"""
import json
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def rasterize_geojson_instances(geojson_path, class_map, image_size=(1024, 1024),
                                coord_offset=(0, 0)):
    """Rasterize GeoJSON nuclei annotations preserving individual instances.

    Unlike the standard rasterize_geojson which only produces a class mask,
    this function also produces an instance mask where each nucleus gets a
    unique integer ID. This is required for HV map generation.

    Args:
        geojson_path: Path to GeoJSON file with nuclei annotations.
        class_map: Dict mapping class names to class indices.
        image_size: (H, W) of the output masks.
        coord_offset: (offset_x, offset_y) for context ROI coordinate mapping.

    Returns:
        instance_mask: [H, W] int32, each nucleus has a unique ID (0 = background).
        class_mask: [H, W] int32, semantic class per pixel (0 = background).
    """
    instance_mask = np.zeros(image_size, dtype=np.int32)
    class_mask = np.zeros(image_size, dtype=np.int32)

    with open(geojson_path, 'r') as f:
        data = json.load(f)

    instance_id = 0

    for feature in data.get('features', []):
        geometry = feature.get('geometry', {})
        classification = feature.get('properties', {}).get('classification', {})
        class_name = classification.get('name', '')

        class_idx = class_map.get(class_name, None)
        if class_idx is None or class_idx == 0:
            continue

        geom_type = geometry.get('type', '')
        coords_list = geometry.get('coordinates', [])

        if geom_type == 'Polygon':
            instance_id += 1
            _fill_instance(instance_mask, class_mask, coords_list,
                           instance_id, class_idx, coord_offset)
        elif geom_type == 'MultiPolygon':
            for polygon_coords in coords_list:
                instance_id += 1
                _fill_instance(instance_mask, class_mask, polygon_coords,
                               instance_id, class_idx, coord_offset)

    return instance_mask, class_mask


def _fill_instance(instance_mask, class_mask, coords_list, instance_id,
                   class_idx, coord_offset=(0, 0)):
    """Fill a single polygon instance into both masks."""
    if not coords_list or not CV2_AVAILABLE:
        return

    exterior = np.array(coords_list[0], dtype=np.int32)
    if coord_offset != (0, 0):
        exterior = exterior + np.array(coord_offset, dtype=np.int32)

    cv2.fillPoly(instance_mask, [exterior], instance_id)
    cv2.fillPoly(class_mask, [exterior], class_idx)

    # Handle holes
    for hole in coords_list[1:]:
        hole_pts = np.array(hole, dtype=np.int32)
        if coord_offset != (0, 0):
            hole_pts = hole_pts + np.array(coord_offset, dtype=np.int32)
        cv2.fillPoly(instance_mask, [hole_pts], 0)
        cv2.fillPoly(class_mask, [hole_pts], 0)


def generate_hv_map(instance_mask):
    """Generate horizontal and vertical gradient maps from an instance mask.

    For each nucleus pixel, computes the normalized distance from the
    instance centroid:
      - Channel 0 (horizontal): (x - cx) / (half_width), clipped to [-1, 1]
      - Channel 1 (vertical):   (y - cy) / (half_height), clipped to [-1, 1]

    Args:
        instance_mask: [H, W] int32, each instance has a unique ID.

    Returns:
        hv_map: [2, H, W] float32, horizontal and vertical gradient maps.
    """
    H, W = instance_mask.shape
    hv_map = np.zeros((2, H, W), dtype=np.float32)

    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]

    for inst_id in instance_ids:
        inst_pixels = (instance_mask == inst_id)
        ys, xs = np.where(inst_pixels)

        if len(ys) == 0:
            continue

        # Centroid
        cy = np.mean(ys).astype(np.float32)
        cx = np.mean(xs).astype(np.float32)

        # Bounding box dimensions for normalization
        y_range = max(ys.max() - ys.min(), 1)
        x_range = max(xs.max() - xs.min(), 1)

        # Normalized distance from centroid
        hv_map[0, ys, xs] = (xs - cx) / (x_range / 2.0 + 1e-6)  # horizontal
        hv_map[1, ys, xs] = (ys - cy) / (y_range / 2.0 + 1e-6)  # vertical

    return np.clip(hv_map, -1.0, 1.0)


def generate_np_map(instance_mask):
    """Generate binary nuclei-pixel map from instance mask.

    Args:
        instance_mask: [H, W] int32.

    Returns:
        np_map: [H, W] int32, 1 = nuclei pixel, 0 = background.
    """
    return (instance_mask > 0).astype(np.int32)
