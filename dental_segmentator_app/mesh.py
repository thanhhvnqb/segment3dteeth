from __future__ import annotations

import colorsys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from .contracts import LABEL_TO_COLOR_HEX, POSTPROCESS_LABEL_IDS

TRANSPARENT_LABEL_ALPHA = {
    1: 72,
    2: 72,
}


def _coerce_visibility_id(value: str | int) -> str:
    if isinstance(value, int):
        return f"label:{value}"
    text = str(value)
    if ":" in text:
        return text
    if text.isdigit():
        return f"label:{text}"
    return text


def _visibility_id_for_node(node_name: str) -> str | None:
    if node_name.startswith("label_"):
        try:
            label_id = int(node_name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            return None
        return f"label:{label_id}"

    if node_name.startswith("tooth_"):
        try:
            tooth_id = int(node_name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            return None
        # Keep UI toggles grouped as the original 4 labels.
        return "label:3" if tooth_id < 30 else "label:4"

    return None


def _iter_scene_node_geometry_pairs(
    scene: trimesh.Scene,
) -> list[tuple[str, str]]:
    nodes_geometry = scene.graph.nodes_geometry
    if hasattr(nodes_geometry, "items"):
        return [
            (str(node_name), str(geometry_name))
            for node_name, geometry_name in nodes_geometry.items()
        ]

    return [
        (str(node_name), str(scene.graph[node_name][1]))
        for node_name in nodes_geometry
        if scene.graph[node_name][1] is not None
    ]


def _label_color_rgba(label_id: int) -> tuple[int, int, int, int]:
    hex_color = LABEL_TO_COLOR_HEX[label_id].lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    alpha = TRANSPARENT_LABEL_ALPHA.get(label_id, 255)
    return red, green, blue, alpha


def _tooth_color_rgba(tooth_id: int) -> tuple[int, int, int, int]:
    # Deterministic hue spread so each tooth keeps a stable distinct color.
    hue = (tooth_id * 0.61803398875) % 1.0
    saturation = 0.65
    value = 0.92
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(red * 255), int(green * 255), int(blue * 255), 255


def _mask_to_mesh(
    mask: np.ndarray, spacing_zyx: tuple[float, float, float]
) -> trimesh.Trimesh | None:
    if not mask.any():
        return None
    vertices, faces, _, _ = marching_cubes(
        mask.astype(np.uint8, copy=False), level=0.5, spacing=spacing_zyx
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def build_scene_from_segmentation(
    segmentation: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    tooth_masks: dict[int, np.ndarray] | None = None,
) -> trimesh.Scene:
    spacing_zyx = tuple(float(value) for value in spacing_xyz[::-1])
    scene = trimesh.Scene()
    excluded_teeth_labels = {3, 4} if tooth_masks else set()

    for label_id in POSTPROCESS_LABEL_IDS:
        if label_id in excluded_teeth_labels:
            continue
        mesh = _mask_to_mesh(segmentation == label_id, spacing_zyx)
        if mesh is None:
            continue
        mesh.visual.vertex_colors = np.tile(
            np.array(_label_color_rgba(label_id), dtype=np.uint8),
            (len(mesh.vertices), 1),
        )
        scene.add_geometry(mesh, node_name=f"label_{label_id}")

    for tooth_id, tooth_mask in sorted((tooth_masks or {}).items()):
        mesh = _mask_to_mesh(tooth_mask, spacing_zyx)
        if mesh is None:
            continue
        mesh.visual.vertex_colors = np.tile(
            np.array(_tooth_color_rgba(tooth_id), dtype=np.uint8),
            (len(mesh.vertices), 1),
        )
        scene.add_geometry(mesh, node_name=f"tooth_{tooth_id}")
    return scene


def export_mesh_bundle(
    segmentation: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    output_directory: str | Path,
    tooth_masks: dict[int, np.ndarray] | None = None,
) -> dict[str, Path]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    scene = build_scene_from_segmentation(
        segmentation, spacing_xyz, tooth_masks=tooth_masks
    )
    combined_mesh = (
        trimesh.util.concatenate([geometry for geometry in scene.geometry.values()])
        if scene.geometry
        else None
    )

    outputs: dict[str, Path] = {}
    if combined_mesh is not None:
        outputs["stl"] = output_directory / "segmentation.stl"
        outputs["obj"] = output_directory / "segmentation.obj"
        combined_mesh.export(outputs["stl"])
        combined_mesh.export(outputs["obj"])

    outputs["gltf"] = output_directory / "segmentation.glb"
    scene.export(outputs["gltf"])
    return outputs


def export_preview_and_bundle(
    segmentation: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    output_directory: str | Path,
    tooth_masks: dict[int, np.ndarray] | None = None,
) -> dict[str, Path]:
    return export_mesh_bundle(
        segmentation,
        spacing_xyz,
        output_directory,
        tooth_masks=tooth_masks,
    )


def export_filtered_preview(
    source_preview_path: str | Path,
    output_preview_path: str | Path,
    visible_label_ids: Iterable[str | int],
) -> Path:
    source_preview_path = Path(source_preview_path)
    output_preview_path = Path(output_preview_path)
    visible_ids = {_coerce_visibility_id(label_id) for label_id in visible_label_ids}

    scene = trimesh.load(source_preview_path, force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError(f"Expected scene in {source_preview_path}")

    node_geometry_pairs = _iter_scene_node_geometry_pairs(scene)

    filtered_scene = trimesh.Scene()
    matched_nodes = 0
    for node_name, geometry_name in node_geometry_pairs:
        visibility_id = _visibility_id_for_node(str(node_name))
        if visibility_id is None:
            continue
        if visibility_id not in visible_ids:
            continue

        transform, _ = scene.graph[node_name]
        geometry = scene.geometry[geometry_name].copy()
        filtered_scene.add_geometry(
            geometry,
            node_name=node_name,
            geom_name=geometry_name,
            transform=transform,
        )
        matched_nodes += 1

    # If node metadata is unavailable, keep the original preview so UI still works.
    if matched_nodes == 0:
        return source_preview_path

    output_preview_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_scene.export(output_preview_path)
    return output_preview_path


def list_preview_visibility_ids(source_preview_path: str | Path) -> list[str]:
    source_preview_path = Path(source_preview_path)
    scene = trimesh.load(source_preview_path, force="scene")
    if not isinstance(scene, trimesh.Scene):
        return []

    visible_ids = {
        visibility_id
        for node_name, _ in _iter_scene_node_geometry_pairs(scene)
        for visibility_id in [_visibility_id_for_node(node_name)]
        if visibility_id is not None
    }
    return sorted(visible_ids)


def write_bundle_zip(
    segmentation: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    output_directory: str | Path,
    nifti_path: str | Path | None = None,
    tooth_masks: dict[int, np.ndarray] | None = None,
) -> Path:
    import shutil

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        export_mesh_bundle(
            segmentation,
            spacing_xyz,
            temp_path,
            tooth_masks=tooth_masks,
        )
        if nifti_path is not None:
            shutil.copy2(Path(nifti_path), temp_path / Path(nifti_path).name)
        archive_base = output_directory / "dental_segmentator_export"
        shutil.make_archive(str(archive_base), "zip", temp_path)
        return archive_base.with_suffix(".zip")
