from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TritonConnection:
    url: str
    model_name: str = "dental_segmentator"
    model_version: str = "1"


class TritonSegmentationClient:
    def __init__(self, connection: TritonConnection):
        self.connection = connection

    def infer(
        self, image: np.ndarray, spacing_xyz: tuple[float, float, float]
    ) -> np.ndarray:
        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient(
            url=self.connection.url, verbose=False
        )
        if not client.is_server_live():
            raise RuntimeError(f"Triton server {self.connection.url} is not live")

        image_input = grpcclient.InferInput("image", image.shape, "FP32")
        spacing_input = grpcclient.InferInput("spacing", (3,), "FP32")
        image_input.set_data_from_numpy(image.astype(np.float32, copy=False))
        spacing_input.set_data_from_numpy(np.asarray(spacing_xyz, dtype=np.float32))

        output = grpcclient.InferRequestedOutput("segmentation")
        response = client.infer(
            model_name=self.connection.model_name,
            model_version=self.connection.model_version,
            inputs=[image_input, spacing_input],
            outputs=[output],
        )
        return response.as_numpy("segmentation")


def serialize_metadata(metadata: dict) -> str:
    return json.dumps(metadata, sort_keys=True)
