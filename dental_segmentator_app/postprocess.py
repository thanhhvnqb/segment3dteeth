from __future__ import annotations

import numpy as np
from scipy.ndimage import generate_binary_structure, label as connected_components

from .contracts import DEFAULT_MINIMUM_ISLAND_SIZE_MM3, POSTPROCESS_LABEL_IDS


def remove_small_islands(
    segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    minimum_island_size_mm3: float = DEFAULT_MINIMUM_ISLAND_SIZE_MM3,
    label_ids: tuple[int, ...] = POSTPROCESS_LABEL_IDS,
) -> np.ndarray:
    cleaned = segmentation.copy()
    voxel_volume_mm3 = float(np.prod(spacing))
    minimum_voxels = max(1, int(np.ceil(minimum_island_size_mm3 / voxel_volume_mm3)))
    structure = generate_binary_structure(segmentation.ndim, segmentation.ndim)

    for label_id in label_ids:
        mask = cleaned == label_id
        if not mask.any():
            continue

        components, component_count = connected_components(mask, structure=structure)
        if component_count == 0:
            continue

        component_sizes = np.bincount(components.ravel())
        keep_mask = np.zeros_like(mask, dtype=bool)
        for component_index, size in enumerate(component_sizes):
            if component_index == 0 or size < minimum_voxels:
                continue
            keep_mask |= components == component_index

        cleaned[(mask & ~keep_mask)] = 0

    return cleaned
