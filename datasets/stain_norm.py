"""
Macenko Stain Normalization for H&E histopathology images.

Normalizes staining variation across slides from different scanners/labs
by decomposing each image into Hematoxylin and Eosin stain concentrations
and reconstructing with a reference stain matrix.

Algorithm (Macenko et al., 2009):
  1. Convert RGB to Optical Density (OD)
  2. Remove background pixels (low OD)
  3. Compute stain vectors via SVD of OD values
  4. Project pixel OD onto stain vectors to get concentrations
  5. Normalize concentration percentiles to match reference
  6. Reconstruct RGB from normalized concentrations + reference stain matrix

Reference:
  M. Macenko et al., "A method for normalizing histology slides for
  quantitative analysis," ISBI 2009.

Usage:
    normalizer = MacenkoNormalizer()
    normalized_img = normalizer.normalize(img)  # img: uint8 [H, W, 3]
"""
import numpy as np


# Default reference stain matrix for H&E (Hematoxylin, Eosin)
# Derived from a typical well-stained H&E slide in OD space
# Each column is one stain vector [R_od, G_od, B_od]
DEFAULT_REF_STAIN_MATRIX = np.array([
    [0.5626, 0.2159],   # R channel OD
    [0.7201, 0.8012],   # G channel OD
    [0.4062, 0.5581],   # B channel OD
])

# Default reference max concentrations (99th percentile)
DEFAULT_REF_MAX_CONC = np.array([1.9705, 1.0308])


class MacenkoNormalizer:
    """Macenko stain normalization for H&E images.

    Fits a reference stain profile (either from a target image or defaults),
    then normalizes new images to match that profile.

    Args:
        luminosity_threshold: OD threshold to separate tissue from background.
            Pixels with max OD below this are considered background.
        percentile: Percentile for robust concentration range estimation.
    """

    def __init__(self, luminosity_threshold=0.15, percentile=99):
        self.luminosity_threshold = luminosity_threshold
        self.percentile = percentile

        # Reference stain profile (use defaults or fit from target image)
        self.ref_stain_matrix = DEFAULT_REF_STAIN_MATRIX.copy()
        self.ref_max_conc = DEFAULT_REF_MAX_CONC.copy()

    def fit(self, target_image):
        """Fit reference stain profile from a target image.

        Call this once with a well-stained reference image.
        If not called, default H&E reference values are used.

        Args:
            target_image: [H, W, 3] uint8 RGB image.
        """
        stain_matrix, max_conc = self._extract_stain_profile(target_image)
        if stain_matrix is not None:
            self.ref_stain_matrix = stain_matrix
            self.ref_max_conc = max_conc

    def normalize(self, image):
        """Normalize a single H&E image to match the reference stain profile.

        Args:
            image: [H, W, 3] uint8 RGB image.

        Returns:
            normalized: [H, W, 3] uint8 RGB image with standardized staining.
        """
        h, w, c = image.shape
        if c != 3:
            return image

        # Extract source stain profile
        src_stain_matrix, src_max_conc = self._extract_stain_profile(image)
        if src_stain_matrix is None:
            # Image is mostly background, return as-is
            return image

        # Convert to OD
        od = self._rgb_to_od(image.reshape(-1, 3).astype(np.float64))

        # Get stain concentrations in source space
        # OD = stain_matrix @ concentrations  →  conc = pinv(stain_matrix) @ OD
        concentrations = od @ np.linalg.pinv(src_stain_matrix).T

        # Normalize concentration ranges to match reference
        for i in range(2):
            if src_max_conc[i] > 1e-6:
                concentrations[:, i] *= self.ref_max_conc[i] / src_max_conc[i]

        # Reconstruct in reference stain space
        od_normalized = concentrations @ self.ref_stain_matrix.T
        rgb_normalized = self._od_to_rgb(od_normalized)

        return rgb_normalized.reshape(h, w, 3)

    def _extract_stain_profile(self, image):
        """Extract stain matrix and max concentrations from an image.

        Returns:
            stain_matrix: [3, 2] matrix (columns = H stain, E stain in OD)
            max_conc: [2] 99th percentile concentrations
            Returns (None, None) if image has insufficient tissue.
        """
        # Convert to OD
        pixels = image.reshape(-1, 3).astype(np.float64)
        od = self._rgb_to_od(pixels)

        # Filter background: keep pixels with sufficient OD
        od_max = od.max(axis=1)
        tissue_mask = od_max > self.luminosity_threshold
        od_tissue = od[tissue_mask]

        if od_tissue.shape[0] < 100:
            return None, None

        # SVD to find principal stain directions
        # Center the OD values
        od_centered = od_tissue - od_tissue.mean(axis=0)

        try:
            _, _, Vt = np.linalg.svd(od_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, None

        # Top 2 singular vectors span the stain plane
        plane = Vt[:2, :]  # [2, 3]

        # Project tissue OD onto the plane
        proj = od_tissue @ plane.T  # [N, 2]

        # Find the angle of each projected point
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        # Robust min/max angles (percentile to avoid outliers)
        min_angle = np.percentile(angles, 1)
        max_angle = np.percentile(angles, 99)

        # Stain vectors are the extreme angle directions projected back to OD
        v1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ plane
        v2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ plane

        # Ensure both vectors point in positive OD direction
        if v1[0] < 0:
            v1 = -v1
        if v2[0] < 0:
            v2 = -v2

        # Convention: Hematoxylin has higher blue component in OD
        # (appears purple/blue), Eosin has higher red (appears pink)
        # In OD space: H has higher OD in blue channel (index 2)
        if v1[2] < v2[2]:
            # v2 is more blue → v2 = Hematoxylin, v1 = Eosin
            stain_matrix = np.column_stack([v2, v1])
        else:
            stain_matrix = np.column_stack([v1, v2])

        # Normalize stain vectors to unit length
        for i in range(2):
            norm = np.linalg.norm(stain_matrix[:, i])
            if norm > 1e-6:
                stain_matrix[:, i] /= norm

        # Get concentration range
        concentrations = od_tissue @ np.linalg.pinv(stain_matrix).T
        max_conc = np.percentile(concentrations, self.percentile, axis=0)
        max_conc = np.clip(max_conc, 0.1, None)  # avoid division by ~0

        return stain_matrix, max_conc

    @staticmethod
    def _rgb_to_od(rgb):
        """Convert RGB values (0-255 range) to Optical Density.

        OD = -log10(I / I_0), where I_0 = 255 (max intensity).
        """
        rgb = np.clip(rgb / 255.0, 1e-6, 1.0)
        return -np.log(rgb)

    @staticmethod
    def _od_to_rgb(od):
        """Convert Optical Density back to RGB uint8."""
        rgb = np.exp(-od)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb
