"""
Post-processing pipeline for PUMA panoptic segmentation.

Two-stage post-processing:
  1. Instance separation: Use HV maps + NP map to separate touching nuclei
     via watershed on the HV gradient energy landscape.
  2. Tissue-guided reclassification: Use tissue segmentation predictions
     to refine nuclei classes based on biological priors.

Tissue-guided rules (from paper findings):
  - Nucleus inside epidermis tissue -> reclassify as epithelial nucleus
  - Nucleus inside blood vessel tissue -> reclassify as endothelial nucleus
  - These rules significantly boost F1 for epithelium (0.04 -> 0.52 in paper)
"""
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from scipy.ndimage import label as scipy_label
    from scipy.ndimage import measurements
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def process_hv_maps(np_map, hv_map, thresh_np=0.5, thresh_k=0.4, min_size=10):
    """Separate touching nuclei using HV gradient maps + watershed.

    Algorithm:
      1. Threshold NP map to get binary nuclei mask
      2. Compute gradient magnitude of HV maps
      3. Find markers (low-gradient regions = nuclei centers)
      4. Apply watershed to separate touching instances

    Args:
        np_map: [H, W] float, nuclei probability map (0-1).
        hv_map: [2, H, W] float, horizontal and vertical gradient maps.
        thresh_np: Threshold for nuclei detection.
        thresh_k: Threshold for marker generation (energy landscape).
        min_size: Minimum nucleus size in pixels.

    Returns:
        instance_map: [H, W] int, each nucleus has a unique ID.
    """
    H, W = np_map.shape

    # Step 1: Binary nuclei mask
    binary_mask = (np_map > thresh_np).astype(np.uint8)

    if binary_mask.sum() == 0:
        return np.zeros((H, W), dtype=np.int32)

    # Step 2: Compute HV gradient energy
    h_grad = hv_map[0]  # horizontal
    v_grad = hv_map[1]  # vertical

    # Sobel on HV maps to detect boundaries between instances
    if CV2_AVAILABLE:
        sobelh_h = cv2.Sobel(h_grad, cv2.CV_64F, 1, 0, ksize=3)
        sobelh_v = cv2.Sobel(h_grad, cv2.CV_64F, 0, 1, ksize=3)
        sobelv_h = cv2.Sobel(v_grad, cv2.CV_64F, 1, 0, ksize=3)
        sobelv_v = cv2.Sobel(v_grad, cv2.CV_64F, 0, 1, ksize=3)

        # Energy = gradient magnitude (high at instance boundaries)
        energy = np.sqrt(sobelh_h**2 + sobelh_v**2 + sobelv_h**2 + sobelv_v**2)
        energy = energy / (energy.max() + 1e-6)
    else:
        # Fallback: simple gradient magnitude
        energy = np.sqrt(h_grad**2 + v_grad**2)
        energy = energy / (energy.max() + 1e-6)

    # Step 3: Generate markers (low energy = nucleus centers)
    marker_mask = ((1 - energy) > thresh_k) & (binary_mask > 0)
    marker_mask = marker_mask.astype(np.uint8)

    # Remove small connected components
    if CV2_AVAILABLE:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, kernel)

    # Label connected components as markers
    if SCIPY_AVAILABLE:
        markers, num_markers = scipy_label(marker_mask)
    elif CV2_AVAILABLE:
        num_markers, markers = cv2.connectedComponents(marker_mask)
        num_markers -= 1  # exclude background
    else:
        return binary_mask.astype(np.int32)

    if num_markers == 0:
        # No markers found, return connected components of binary mask
        if SCIPY_AVAILABLE:
            instance_map, _ = scipy_label(binary_mask)
        else:
            _, instance_map = cv2.connectedComponents(binary_mask)
        return instance_map.astype(np.int32)

    # Step 4: Watershed
    if CV2_AVAILABLE:
        # Prepare for OpenCV watershed (requires 3-channel image)
        energy_uint8 = (energy * 255).astype(np.uint8)
        energy_3ch = cv2.cvtColor(energy_uint8, cv2.COLOR_GRAY2BGR)

        # Markers: background = 0, unknown = -1 is not needed for our approach
        # Use markers + 1 so background = 1, then subtract after
        markers_ws = markers.copy().astype(np.int32)
        markers_ws[binary_mask == 0] = -1  # mark background
        markers_ws += 1  # shift: bg=-1 -> 0, actual markers stay positive

        # OpenCV watershed expects background as 0
        # Actually, let's use a simpler approach with distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        dist_transform = dist_transform / (dist_transform.max() + 1e-6)

        # Combine distance transform with HV energy for better separation
        combined = dist_transform * (1 - energy * 0.5)

        # Use markers from low-energy regions
        markers_final = markers.astype(np.int32)
        markers_final[binary_mask == 0] = 0

        # Watershed on combined energy
        markers_ws = markers_final.copy()
        markers_ws[markers_ws == 0] = 0
        unknown = (binary_mask > 0) & (markers_ws == 0)
        markers_ws[unknown] = 0

        # Use cv2.watershed
        img_3ch = np.stack([
            (combined * 255).astype(np.uint8),
            (binary_mask * 255).astype(np.uint8),
            energy_uint8,
        ], axis=-1)

        markers_ws32 = markers_ws.copy()
        markers_ws32[binary_mask == 0] = -1
        markers_ws32 += 1

        cv2.watershed(img_3ch, markers_ws32)
        markers_ws32 -= 1
        markers_ws32[markers_ws32 <= 0] = 0

        instance_map = markers_ws32.astype(np.int32)
    else:
        instance_map = markers.astype(np.int32)

    # Remove small instances
    if min_size > 0:
        instance_ids = np.unique(instance_map)
        for inst_id in instance_ids:
            if inst_id == 0:
                continue
            if (instance_map == inst_id).sum() < min_size:
                instance_map[instance_map == inst_id] = 0

    return instance_map


def get_instance_centers(instance_map):
    """Extract center coordinates for each nucleus instance.

    Args:
        instance_map: [H, W] int, instance segmentation map.

    Returns:
        centers: dict {instance_id: (center_y, center_x)}
    """
    centers = {}
    instance_ids = np.unique(instance_map)

    for inst_id in instance_ids:
        if inst_id == 0:
            continue
        ys, xs = np.where(instance_map == inst_id)
        centers[inst_id] = (float(np.mean(ys)), float(np.mean(xs)))

    return centers


def classify_instances(instance_map, nc_map):
    """Assign class to each instance based on majority vote of NC predictions.

    Args:
        instance_map: [H, W] int, instance segmentation map.
        nc_map: [H, W] int, per-pixel nuclei classification (argmax of NC head).

    Returns:
        instance_classes: dict {instance_id: class_idx}
    """
    instance_classes = {}
    instance_ids = np.unique(instance_map)

    for inst_id in instance_ids:
        if inst_id == 0:
            continue
        inst_pixels = (instance_map == inst_id)
        pixel_classes = nc_map[inst_pixels]

        # Exclude background class (0) from voting
        fg_classes = pixel_classes[pixel_classes > 0]
        if len(fg_classes) == 0:
            instance_classes[inst_id] = 0
        else:
            # Majority vote
            counts = np.bincount(fg_classes)
            instance_classes[inst_id] = int(np.argmax(counts))

    return instance_classes


def tissue_guided_reclassification(instance_map, instance_classes,
                                   tissue_mask, nuclei_class_names):
    """Refine nuclei classes using tissue context.

    Biological rules:
      - Nucleus center inside epidermis (tissue class 3) -> epithelial nucleus
      - Nucleus center inside blood vessel (tissue class 4) -> endothelial nucleus

    This heuristic significantly improves detection of rare cell types
    (epithelium F1: 0.04 -> 0.52 in PUMA paper baseline).

    Args:
        instance_map: [H, W] int, instance segmentation.
        instance_classes: dict {instance_id: class_idx} from NC head.
        tissue_mask: [H, W] int, tissue segmentation (6 classes).
        nuclei_class_names: list of nuclei class names for the current track.

    Returns:
        refined_classes: dict {instance_id: class_idx} after tissue refinement.
    """
    # Find class indices for epithelium and endothelium in nuclei classes
    epithelium_idx = None
    endothelium_idx = None
    for i, name in enumerate(nuclei_class_names):
        if 'epithelium' in name.lower() or 'epidermal' in name.lower():
            epithelium_idx = i
        if 'endothelium' in name.lower() or 'endothelial' in name.lower():
            endothelium_idx = i

    # Tissue class indices (from constants.py)
    TISSUE_EPIDERMIS = 3
    TISSUE_BLOOD_VESSEL = 4

    centers = get_instance_centers(instance_map)
    refined = dict(instance_classes)

    for inst_id, (cy, cx) in centers.items():
        if inst_id not in refined:
            continue

        # Get tissue class at nucleus center
        iy, ix = int(round(cy)), int(round(cx))
        iy = max(0, min(iy, tissue_mask.shape[0] - 1))
        ix = max(0, min(ix, tissue_mask.shape[1] - 1))
        tissue_class = tissue_mask[iy, ix]

        # Apply tissue-guided rules
        if tissue_class == TISSUE_EPIDERMIS and epithelium_idx is not None:
            refined[inst_id] = epithelium_idx
        elif tissue_class == TISSUE_BLOOD_VESSEL and endothelium_idx is not None:
            refined[inst_id] = endothelium_idx

    return refined


def full_pipeline(np_pred, hv_pred, nc_pred, tissue_pred,
                  nuclei_class_names, thresh_np=0.5, thresh_k=0.4,
                  min_size=10):
    """Full post-processing pipeline: instance separation + tissue refinement.

    Args:
        np_pred: [H, W] float, nuclei probability.
        hv_pred: [2, H, W] float, HV gradient predictions.
        nc_pred: [H, W] int, nuclei classification (argmax).
        tissue_pred: [H, W] int, tissue segmentation (argmax).
        nuclei_class_names: list of nuclei class names.
        thresh_np: NP threshold.
        thresh_k: Marker generation threshold.
        min_size: Minimum nucleus size.

    Returns:
        instance_map: [H, W] int, instance segmentation.
        instance_classes: dict {instance_id: class_idx} after refinement.
        centers: dict {instance_id: (cy, cx)}.
    """
    # Stage 1: Instance separation
    instance_map = process_hv_maps(np_pred, hv_pred, thresh_np, thresh_k, min_size)

    # Stage 2: Per-instance classification
    instance_classes = classify_instances(instance_map, nc_pred)

    # Stage 3: Tissue-guided refinement
    instance_classes = tissue_guided_reclassification(
        instance_map, instance_classes, tissue_pred, nuclei_class_names,
    )

    # Centers for evaluation
    centers = get_instance_centers(instance_map)

    return instance_map, instance_classes, centers
