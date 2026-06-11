from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import gradio as gr

from .contracts import LABEL_TO_NAME, POSTPROCESS_LABEL_IDS
from .io import case_to_nnunet_inputs, load_volume
from .mesh import export_filtered_preview, export_preview_and_bundle
from .postprocess import remove_small_islands
from .triton_client import TritonConnection, TritonSegmentationClient

CSS = """
:root {
  --background-fill-primary: #f3efe4;
  --block-background-fill: #fffaf0;
  --block-border-color: #c9b79d;
  --body-text-color: #1f1d1a;
}
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(227, 221, 144, 0.35), transparent 30%),
    radial-gradient(circle at top right, rgba(212, 161, 230, 0.18), transparent 26%),
    linear-gradient(180deg, #f7f3eb 0%, #efe4d3 100%);
}
.hero-card {
  border: 1px solid rgba(80, 63, 34, 0.15);
  border-radius: 24px;
  background: rgba(255, 250, 240, 0.8);
  box-shadow: 0 20px 60px rgba(72, 52, 20, 0.12);
  padding: 24px;
}
.label-chip {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  margin-right: 6px;
  margin-bottom: 6px;
  background: rgba(217, 199, 160, 0.35);
}
"""


def _default_cache_root() -> Path:
    configured = os.environ.get("DENTAL_SEGMENTATOR_CACHE")
    if configured:
        return Path(configured)

    container_cache = Path("/dental_segmentation_cache")
    if container_cache.exists():
        return container_cache

    # Local fallback when running outside Docker from the repository workspace.
    return Path(__file__).resolve().parent.parent / "dental_segmentation_cache"


CACHE_ROOT = _default_cache_root()
INPUT_CACHE_DIR = CACHE_ROOT / "input"
OUTPUT_CACHE_DIR = CACHE_ROOT / "output"
VISIBLE_PART_CHOICES = [
    (LABEL_TO_NAME[label_id], str(label_id)) for label_id in POSTPROCESS_LABEL_IDS
]
DEFAULT_VISIBLE_PARTS = [str(label_id) for label_id in POSTPROCESS_LABEL_IDS]


def _ensure_cache_dirs() -> None:
    INPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_file(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_unique_cached_name(target_name: str, source_path: Path) -> Path:
    candidate = INPUT_CACHE_DIR / target_name
    if not candidate.exists():
        return candidate

    if _hash_file(candidate) == _hash_file(source_path):
        return candidate

    stem = Path(target_name).stem
    suffix = Path(target_name).suffix or ".zip"
    source_hash = _hash_file(source_path)[:8]
    return INPUT_CACHE_DIR / f"{stem}_{source_hash}{suffix}"


def _cache_uploaded_zip(zip_path: str | Path) -> Path:
    source = Path(zip_path)
    target_name = (
        source.name if source.suffix.lower() == ".zip" else f"{source.name}.zip"
    )
    cached_zip = _ensure_unique_cached_name(target_name, source)
    cached_zip.parent.mkdir(parents=True, exist_ok=True)
    if not cached_zip.exists():
        shutil.copy2(source, cached_zip)
    return cached_zip


def _output_paths(cached_zip: Path) -> tuple[Path, Path, Path]:
    output_dir = OUTPUT_CACHE_DIR / cached_zip.stem
    preview_path = output_dir / f"{cached_zip.stem}.glb"
    hash_marker_path = output_dir / "input.sha256"
    return output_dir, preview_path, hash_marker_path


def _filtered_preview_path(output_dir: Path, visible_label_ids: list[int]) -> Path:
    if not visible_label_ids:
        return output_dir / "preview_none.glb"
    key = "-".join(str(label_id) for label_id in sorted(set(visible_label_ids)))
    return output_dir / f"preview_labels_{key}.glb"


def _list_cached_zip_names() -> list[str]:
    _ensure_cache_dirs()
    return sorted(path.name for path in INPUT_CACHE_DIR.glob("*.zip"))


def _refresh_cached_zip_dropdown() -> Any:
    choices = _list_cached_zip_names()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _zip_to_dicom_directory(zip_path: str | Path, work_dir: str | Path) -> Path:
    extracted_dir = Path(work_dir) / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extracted_dir)

    dicom_dir = Path(work_dir) / "dicom_series"
    dicom_dir.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    for source_file in extracted_dir.rglob("*"):
        if not source_file.is_file():
            continue
        target_file = dicom_dir / f"{copied_files:06d}_{source_file.name}"
        shutil.copy2(source_file, target_file)
        copied_files += 1

    if copied_files == 0:
        raise ValueError("The uploaded zip does not contain any files.")
    return dicom_dir


def _build_client() -> TritonSegmentationClient | None:
    connection_url = os.environ.get("TRITON_URL")
    if not connection_url:
        connection_file = Path("/tmp/dental_segmentator_triton_url.txt")
        if connection_file.exists():
            connection_url = connection_file.read_text().strip()
        else:
            connection_url = "localhost:8001"
    try:
        return TritonSegmentationClient(TritonConnection(connection_url))
    except Exception:
        return None


def _load_and_predict(
    zip_path: str, client: TritonSegmentationClient | None
) -> tuple[str, str]:
    _ensure_cache_dirs()
    cached_zip = _cache_uploaded_zip(zip_path)
    output_dir, preview_path, hash_marker_path = _output_paths(cached_zip)

    current_hash: str | None = None
    if preview_path.exists() and hash_marker_path.exists():
        current_hash = _hash_file(cached_zip)
        stored_hash = hash_marker_path.read_text().strip()
        if stored_hash == current_hash:
            return str(preview_path), cached_zip.name

    if client is None:
        raise RuntimeError("Triton client is not configured")

    with tempfile.TemporaryDirectory(
        prefix=f"dental_segmentator_{cached_zip.stem[:24]}_"
    ) as temp_dir:
        dicom_dir = _zip_to_dicom_directory(cached_zip, temp_dir)
        case = load_volume(dicom_dir)

    image, payload = case_to_nnunet_inputs(case)
    segmentation = client.infer(image, tuple(payload["spacing"]))
    segmentation = segmentation.astype("uint8", copy=False)
    segmentation = remove_small_islands(segmentation, case.spacing)

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_outputs = export_preview_and_bundle(segmentation, case.spacing, output_dir)
    generated_preview = mesh_outputs.get("gltf")
    if generated_preview is None:
        raise RuntimeError("Failed to generate 3D preview")
    if generated_preview != preview_path:
        Path(generated_preview).replace(preview_path)

    if current_hash is None:
        current_hash = _hash_file(cached_zip)
    hash_marker_path.write_text(current_hash)

    return str(preview_path), cached_zip.name


def _build_preview_for_visible_parts(
    cached_zip_name: str | None, visible_parts: list[str] | None
) -> str:
    if not cached_zip_name:
        raise gr.Error("Run segmentation first before toggling visible parts.")

    output_dir = OUTPUT_CACHE_DIR / Path(cached_zip_name).stem
    base_preview_path = output_dir / f"{Path(cached_zip_name).stem}.glb"
    if not base_preview_path.exists():
        raise gr.Error("Preview file is missing. Please run segmentation again.")

    visible_parts = visible_parts or []
    visible_label_ids = [int(value) for value in visible_parts]
    if not visible_label_ids:
        raise gr.Error("Select at least one part to display.")

    if set(visible_label_ids) == set(POSTPROCESS_LABEL_IDS):
        return str(base_preview_path)

    filtered_preview_path = _filtered_preview_path(output_dir, visible_label_ids)
    if filtered_preview_path.exists():
        return str(filtered_preview_path)

    generated_path = export_filtered_preview(
        source_preview_path=base_preview_path,
        output_preview_path=filtered_preview_path,
        visible_label_ids=visible_label_ids,
    )
    return str(generated_path)


def _resolve_zip_path(
    source_mode: str | None, uploaded_zip: str | None, cached_zip_name: str | None
) -> str | None:
    if source_mode == "Upload zip":
        return uploaded_zip
    if source_mode == "Use cached zip" and cached_zip_name:
        return str(INPUT_CACHE_DIR / cached_zip_name)

    # Backward-compatible fallback when mode is missing.
    if uploaded_zip:
        return uploaded_zip
    if cached_zip_name:
        return str(INPUT_CACHE_DIR / cached_zip_name)
    return None


def _toggle_input_source(source_mode: str) -> tuple[Any, Any]:
    return (
        gr.update(visible=source_mode == "Upload zip"),
        gr.update(visible=source_mode == "Use cached zip"),
    )


def build_demo() -> gr.Blocks:
    client = _build_client()
    _ensure_cache_dirs()

    with gr.Blocks(title="Dental Segmentator", css=CSS) as demo:
        gr.Markdown("""
            <div class="hero-card">
              <h1 style="margin:0 0 8px 0;">Dental Segmentator</h1>
              <p style="margin:0; max-width: 70ch;">
                Upload one zip file containing DICOM files or choose a zip already available
                in cache/input. Uploaded files are stored by filename, and hash is computed
                only when needed to validate whether an existing 3D result can be reused.
              </p>
            </div>
            """)

        source_mode = gr.Radio(
            label="Input source",
            choices=["Upload zip", "Use cached zip"],
            value="Upload zip",
            info="Choose where the input zip comes from.",
        )

        with gr.Row():
            zip_input = gr.File(
                label="Upload zip file",
                file_types=[".zip"],
                type="filepath",
                visible=True,
            )
            cached_zip_input = gr.Dropdown(
                label="Select from cache/input",
                choices=_list_cached_zip_names(),
                value=None,
                interactive=True,
                visible=False,
            )

        refresh_button = gr.Button("Refresh", variant="secondary")
        run_button = gr.Button("Run segmentation", variant="primary")

        visible_parts_input = gr.CheckboxGroup(
            label="Visible segmented parts",
            choices=VISIBLE_PART_CHOICES,
            value=DEFAULT_VISIBLE_PARTS,
            info="Toggle parts on/off and the 3D view updates without rerunning inference.",
        )

        preview_output = gr.Model3D(label="3D preview")
        cached_zip_state = gr.State(value=None)

        def _run(selected_source, uploaded_zip, cached_zip_name):
            zip_path = _resolve_zip_path(selected_source, uploaded_zip, cached_zip_name)
            if not zip_path:
                raise gr.Error(
                    "Provide an uploaded zip or choose one file from cache/input."
                )
            preview, cached_name = _load_and_predict(zip_path, client)
            choices = _list_cached_zip_names()
            selected = (
                cached_name
                if cached_name in choices
                else (choices[0] if choices else None)
            )
            preview = _build_preview_for_visible_parts(
                cached_name, DEFAULT_VISIBLE_PARTS
            )
            return (
                preview,
                gr.update(choices=choices, value=selected),
                cached_name,
                gr.update(value=DEFAULT_VISIBLE_PARTS),
            )

        def _update_preview_visibility(visible_parts, cached_zip_name):
            return _build_preview_for_visible_parts(cached_zip_name, visible_parts)

        source_mode.change(
            _toggle_input_source,
            inputs=[source_mode],
            outputs=[zip_input, cached_zip_input],
        )

        refresh_button.click(
            _refresh_cached_zip_dropdown,
            inputs=[],
            outputs=[cached_zip_input],
        )

        run_button.click(
            _run,
            inputs=[source_mode, zip_input, cached_zip_input],
            outputs=[
                preview_output,
                cached_zip_input,
                cached_zip_state,
                visible_parts_input,
            ],
        )

        visible_parts_input.change(
            _update_preview_visibility,
            inputs=[visible_parts_input, cached_zip_state],
            outputs=[preview_output],
        )

    return demo


def launch() -> None:
    build_demo().launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(CACHE_ROOT), str(OUTPUT_CACHE_DIR)],
    )
