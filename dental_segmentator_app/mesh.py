from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from .contracts import LABEL_TO_COLOR_HEX, POSTPROCESS_LABEL_IDS


def _label_color_rgba(label_id: int) -> tuple[int, int, int, int]:
    hex_color = LABEL_TO_COLOR_HEX[label_id].lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return red, green, blue, 255


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
    segmentation: np.ndarray, spacing_xyz: tuple[float, float, float]
) -> trimesh.Scene:
    spacing_zyx = tuple(float(value) for value in spacing_xyz[::-1])
    scene = trimesh.Scene()
    for label_id in POSTPROCESS_LABEL_IDS:
        mesh = _mask_to_mesh(segmentation == label_id, spacing_zyx)
        if mesh is None:
            continue
        mesh.visual.vertex_colors = np.tile(
            np.array(_label_color_rgba(label_id), dtype=np.uint8),
            (len(mesh.vertices), 1),
        )
        scene.add_geometry(mesh, node_name=f"label_{label_id}")
    return scene


def export_mesh_bundle(
    segmentation: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    output_directory: str | Path,
) -> dict[str, Path]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    scene = build_scene_from_segmentation(segmentation, spacing_xyz)
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
) -> dict[str, Path]:
    return export_mesh_bundle(segmentation, spacing_xyz, output_directory)


def export_filtered_preview(
    source_preview_path: str | Path,
    output_preview_path: str | Path,
    visible_label_ids: Iterable[int],
) -> Path:
    source_preview_path = Path(source_preview_path)
    output_preview_path = Path(output_preview_path)
    visible_ids = {int(label_id) for label_id in visible_label_ids}

    scene = trimesh.load(source_preview_path, force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError(f"Expected scene in {source_preview_path}")

    nodes_geometry = scene.graph.nodes_geometry
    if hasattr(nodes_geometry, "items"):
        node_geometry_pairs = list(nodes_geometry.items())
    else:
        node_geometry_pairs = [
            (node_name, scene.graph[node_name][1])
            for node_name in nodes_geometry
            if scene.graph[node_name][1] is not None
        ]

    filtered_scene = trimesh.Scene()
    matched_nodes = 0
    for node_name, geometry_name in node_geometry_pairs:
        node_text = str(node_name)
        if not node_text.startswith("label_"):
            continue

        try:
            label_id = int(node_text.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue

        if label_id not in visible_ids:
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


def write_bundle_zip(
    segmentation: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    output_directory: str | Path,
    nifti_path: str | Path | None = None,
) -> Path:
    import shutil

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        export_mesh_bundle(segmentation, spacing_xyz, temp_path)
        if nifti_path is not None:
            shutil.copy2(Path(nifti_path), temp_path / Path(nifti_path).name)
        archive_base = output_directory / "dental_segmentator_export"
        shutil.make_archive(str(archive_base), "zip", temp_path)
        return archive_base.with_suffix(".zip")
