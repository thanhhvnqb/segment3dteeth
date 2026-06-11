from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class VolumeCase:
    image: sitk.Image
    array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    source_path: str


def _read_dicom_series(directory: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))
    if not series_ids:
        raise ValueError(f"No DICOM series found in {directory}")
    series_files = reader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
    reader.SetFileNames(series_files)
    return reader.Execute()


def load_volume(source: str | Path) -> VolumeCase:
    source_path = Path(source)
    if source_path.is_dir():
        image = _read_dicom_series(source_path)
    else:
        image = sitk.ReadImage(str(source_path))

    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    return VolumeCase(
        image=image,
        array=array[None],
        spacing=tuple(float(value) for value in image.GetSpacing()),
        origin=tuple(float(value) for value in image.GetOrigin()),
        direction=tuple(float(value) for value in image.GetDirection()),
        source_path=str(source_path),
    )


def case_to_nnunet_inputs(case: VolumeCase) -> tuple[np.ndarray, dict]:
    return case.array, {"spacing": list(case.spacing[::-1])}


def save_segmentation_nifti(
    segmentation: np.ndarray, reference_case: VolumeCase, output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = sitk.GetImageFromArray(
        segmentation.astype(
            np.uint8 if segmentation.max() < 255 else np.uint16, copy=False
        )
    )
    output_image.SetSpacing(reference_case.spacing)
    output_image.SetOrigin(reference_case.origin)
    output_image.SetDirection(reference_case.direction)
    sitk.WriteImage(output_image, str(output_path), True)
    return output_path


def copy_directory_files(files: Iterable[str | Path], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        file_path = Path(file_path)
        target = destination / file_path.name
        target.write_bytes(file_path.read_bytes())
    return destination
