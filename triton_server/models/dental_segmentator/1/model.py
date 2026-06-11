from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError as exc:  # pragma: no cover - only executed inside Triton
    pb_utils = None
    _IMPORT_ERROR = exc


class TritonPythonModel:
    def initialize(self, args):
        if pb_utils is None:
            raise RuntimeError(
                "triton_python_backend_utils is unavailable"
            ) from _IMPORT_ERROR

        model_config = json.loads(args["model_config"])
        self.model_dir = Path(args["model_repository"]) / args["model_name"]
        self.device_id = int(
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0
        )
        self.output_name = pb_utils.get_output_config_by_name(
            model_config, "segmentation"
        )["name"]

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch

        model_root = Path(
            os.environ.get(
                "DENTAL_SEGMENTATOR_MODEL_PATH",
                "/model_weight/DentalSegmentator_v100/",
            )
        )
        checkpoint = os.environ.get(
            "DENTAL_SEGMENTATOR_CHECKPOINT", "checkpoint_final.pth"
        )

        self.predictor = nnUNetPredictor(
            device=torch.device(f"cuda:{self.device_id}"),
            perform_everything_on_device=True,
        )
        self.predictor.initialize_from_trained_model_folder(
            str(model_root), use_folds=(0,), checkpoint_name=checkpoint
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            spacing = (
                pb_utils.get_input_tensor_by_name(request, "spacing")
                .as_numpy()
                .astype(np.float32, copy=False)
            )

            if image.ndim == 3:
                image = image[None]
            if image.ndim != 4:
                raise ValueError(
                    f"Expected image with shape [C, Z, Y, X], got {image.shape}"
                )
            if image.shape[0] != 1:
                raise ValueError(
                    f"This scaffold expects a single CT channel, got {image.shape[0]}"
                )

            segmentation = self.predictor.predict_single_npy_array(
                image,
                {"spacing": spacing.tolist()},
                segmentation_previous_stage=None,
                output_file_truncated=None,
                save_or_return_probabilities=False,
            )

            output_tensor = pb_utils.Tensor(
                self.output_name, np.asarray(segmentation, dtype=np.uint8)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses
