from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

UPPER_TEETH_LABEL_ID = 3
LOWER_TEETH_LABEL_ID = 4


@dataclass(frozen=True)
class ToothSegmentationResult:
    tooth_masks: dict[int, np.ndarray]
    warnings: list[str]


def _component_masks(mask: np.ndarray, minimum_voxels: int) -> list[np.ndarray]:
    structure = ndi.generate_binary_structure(mask.ndim, mask.ndim)
    components, count = ndi.label(mask, structure=structure)
    if count == 0:
        return []

    sizes = np.bincount(components.ravel())
    parts: list[np.ndarray] = []
    for component_id in range(1, len(sizes)):
        if sizes[component_id] < minimum_voxels:
            continue
        parts.append(components == component_id)
    return parts


def _watershed_split(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    minimum_voxels: int,
    target_max_teeth: int,
) -> list[np.ndarray]:
    distance = ndi.distance_transform_edt(mask, sampling=spacing_xyz[::-1])
    if distance.max() <= 0.0:
        return _component_masks(mask, minimum_voxels)

    smoothed = ndi.gaussian_filter(distance, sigma=1.0)
    peak_coords = peak_local_max(
        smoothed,
        min_distance=4,
        threshold_abs=float(np.percentile(smoothed[mask], 40.0)),
        labels=mask,
        num_peaks=max(1, target_max_teeth),
    )
    if len(peak_coords) == 0:
        return _component_masks(mask, minimum_voxels)

    markers = np.zeros(mask.shape, dtype=np.int32)
    for marker_id, coord in enumerate(peak_coords, start=1):
        markers[tuple(coord)] = marker_id

    split_labels = watershed(-smoothed, markers=markers, mask=mask)
    components: list[np.ndarray] = []
    for component_id in np.unique(split_labels):
        if component_id == 0:
            continue
        part = split_labels == component_id
        if int(part.sum()) >= minimum_voxels:
            components.append(part)

    if not components:
        return _component_masks(mask, minimum_voxels)
    return components


def _split_arch_components(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    minimum_tooth_volume_mm3: float,
    expected_min_teeth: int,
    expected_max_teeth: int,
) -> list[np.ndarray]:
    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    minimum_voxels = max(1, int(np.ceil(minimum_tooth_volume_mm3 / voxel_volume_mm3)))

    parts = _component_masks(mask, minimum_voxels)
    if expected_min_teeth <= len(parts) <= expected_max_teeth:
        return parts

    # Fallback for merged teeth: use distance-based watershed to introduce cuts.
    split_parts = _watershed_split(
        mask, spacing_xyz, minimum_voxels, expected_max_teeth
    )
    if len(split_parts) > len(parts):
        return split_parts
    return parts


def _centroid_xyz(
    mask: np.ndarray, spacing_xyz: tuple[float, float, float]
) -> np.ndarray:
    coords_zyx = np.argwhere(mask)
    if len(coords_zyx) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Index order is z, y, x. Convert to x, y, z in mm.
    spacing_zyx = np.array(spacing_xyz[::-1], dtype=np.float64)
    centroid_zyx = coords_zyx.mean(axis=0) * spacing_zyx
    return centroid_zyx[::-1]


def _assign_fdi_for_arch(
    parts: list[np.ndarray],
    spacing_xyz: tuple[float, float, float],
    arch: str,
) -> tuple[dict[int, np.ndarray], list[str]]:
    warnings: list[str] = []
    if not parts:
        return {}, warnings

    centroids = [(_centroid_xyz(part, spacing_xyz), part) for part in parts]
    x_values = np.array([centroid[0] for centroid, _ in centroids], dtype=np.float64)
    mid_x = float(np.median(x_values))

    right = [(c, p) for c, p in centroids if c[0] >= mid_x]
    left = [(c, p) for c, p in centroids if c[0] < mid_x]

    right.sort(key=lambda item: abs(item[0][0] - mid_x))
    left.sort(key=lambda item: abs(item[0][0] - mid_x))

    result: dict[int, np.ndarray] = {}
    if arch == "upper":
        right_base, left_base = 11, 21
    else:
        left_base, right_base = 31, 41

    for index, (_, part) in enumerate(right[:8]):
        result[right_base + index] = part
    for index, (_, part) in enumerate(left[:8]):
        result[left_base + index] = part

    dropped = max(0, len(right) - 8) + max(0, len(left) - 8)
    if dropped:
        warnings.append(
            f"{arch}: dropped {dropped} extra tooth components beyond FDI range"
        )

    return result, warnings


def segment_teeth(
    segmentation: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    minimum_tooth_volume_mm3: float = 80.0,
    expected_min_teeth_per_arch: int = 6,
    expected_max_teeth_per_arch: int = 16,
) -> ToothSegmentationResult:
    warnings: list[str] = []
    tooth_masks: dict[int, np.ndarray] = {}

    arch_specs = (
        ("upper", UPPER_TEETH_LABEL_ID),
        ("lower", LOWER_TEETH_LABEL_ID),
    )
    for arch, label_id in arch_specs:
        arch_mask = segmentation == label_id
        if not arch_mask.any():
            warnings.append(f"{arch}: no tooth voxels found for label {label_id}")
            continue

        parts = _split_arch_components(
            arch_mask,
            spacing_xyz,
            minimum_tooth_volume_mm3,
            expected_min_teeth_per_arch,
            expected_max_teeth_per_arch,
        )
        if len(parts) < expected_min_teeth_per_arch:
            warnings.append(
                f"{arch}: only {len(parts)} tooth components found after split"
            )

        fdi_map, arch_warnings = _assign_fdi_for_arch(parts, spacing_xyz, arch)
        tooth_masks.update(fdi_map)
        warnings.extend(arch_warnings)

    return ToothSegmentationResult(tooth_masks=tooth_masks, warnings=warnings)
