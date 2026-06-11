# Dental Segmentator Gradio Scaffold

This workspace now contains a first-pass implementation of the plan in `plan.md`:

- `dental_segmentator_app/` provides the shared I/O, post-processing, mesh export, Triton client, and Gradio UI.
- `triton_repo/` contains a Triton Python backend model that loads the nnU-Net checkpoint and returns a segmentation volume.
- `Dockerfile.triton`, `Dockerfile.gradio`, and `docker-compose.yml` provide the deployment skeleton.

The implementation follows the Slicer contract:

- labels 1-5 match the source mapping
- small-island cleanup applies to labels 1-4 only
- NIfTI and DICOM series are supported as inputs
- exports include NIfTI, STL, OBJ, and GLB/glTF-compatible preview output

To run the Gradio app locally, install the Python dependencies and launch `python -m dental_segmentator_app`.