from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelSpec:
    label_id: int
    name: str
    color_hex: str


LABELS: tuple[LabelSpec, ...] = (
    LabelSpec(0, "Background", "#000000"),
    LabelSpec(1, "Maxilla & Upper Skull", "#E3DD90"),
    LabelSpec(2, "Mandible", "#D4A1E6"),
    LabelSpec(3, "Upper Teeth", "#DC9565"),
    LabelSpec(4, "Lower Teeth", "#EBDFB4"),
    LabelSpec(5, "Mandibular canal", "#D8654F"),
)

LABEL_TO_NAME = {label.label_id: label.name for label in LABELS}
LABEL_TO_COLOR_HEX = {label.label_id: label.color_hex for label in LABELS}
LABEL_IDS = tuple(label.label_id for label in LABELS)
POSTPROCESS_LABEL_IDS = (1, 2, 3, 4)
DEFAULT_MINIMUM_ISLAND_SIZE_MM3 = 60.0
