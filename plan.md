## Plan: Gradio Dental Segmentation via Triton

Mục tiêu là dựng web app Gradio nhận NIfTI + DICOM, gửi inference sang Triton (Python backend chạy nnUNet), hiển thị mesh 3D kết quả, và export đủ NIfTI/STL/OBJ/GLTF. Kế hoạch này bám sát hành vi từ module SlicerDentalSegmentator và contract nnUNet hiện có.

**Những gì đã khóa từ source**
1. Mapping segment và hậu xử lý theo Slicer:
- Tên/màu segment và quy trình hậu xử lý nằm ở [SlicerDentalSegmentator/DentalSegmentator/DentalSegmentatorLib/SegmentationWidget.py](SlicerDentalSegmentator/DentalSegmentator/DentalSegmentatorLib/SegmentationWidget.py#L391) và [SlicerDentalSegmentator/DentalSegmentator/DentalSegmentatorLib/SegmentationWidget.py](SlicerDentalSegmentator/DentalSegmentator/DentalSegmentatorLib/SegmentationWidget.py#L422).
- Remove small islands cho segment 1-4, bỏ qua mandibular canal ở [SlicerDentalSegmentator/DentalSegmentator/DentalSegmentatorLib/SegmentationWidget.py](SlicerDentalSegmentator/DentalSegmentator/DentalSegmentatorLib/SegmentationWidget.py#L454).
2. Contract model nnUNet:
- Labels/file ending ở [Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json](Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json#L1).
- Plans, transpose, spacing, patch size ở [Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json](Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json#L291).
- Predictor pipeline ở [nnUNet/nnunetv2/inference/predict_from_raw_data.py](nnUNet/nnunetv2/inference/predict_from_raw_data.py#L1).
- Resample logits về shape gốc ở [nnUNet/nnunetv2/inference/export_prediction.py](nnUNet/nnunetv2/inference/export_prediction.py#L14).
3. Deployment hiện trạng:
- Workspace chưa có Dockerfile, docker-compose, config.pbtxt hay Triton client code.

**Steps**
1. Phase 1: Chốt hợp đồng I/O Gradio ↔ Triton.
- Input web: NIfTI hoặc DICOM series.
- Tiền xử lý tạo tensor + properties theo nnUNet plans.
- Output từ Triton: segmentation label volume đã đưa về không gian ảnh gốc.
2. Phase 2: Thiết kế Triton model repository với Python backend.
- Initialize load dataset/plans/checkpoint fold_0 từ model path đã mount.
- Execute nhận tensor/properties, gọi predictor + convert to segmentation.
- Cấu hình model config batch 1, instance GPU, timeout.
3. Phase 3: Thiết kế service Gradio.
- Upload NIfTI và DICOM->NIfTI.
- Gọi Triton qua gRPC.
- Ánh xạ tên/màu 5 lớp giống Slicer.
- Hậu xử lý islands cho segment 1-4.
- Dựng mesh 3D và hiển thị.
- Export NIfTI/STL/OBJ/GLTF.
4. Phase 4: Docker hóa.
- Image Triton có nnUNet dependencies.
- Image web Gradio có tritonclient + stack xử lý mesh.
- Docker Compose 2 services: Triton dùng GPU, Gradio CPU.
- Mount model path và thư mục input/output.
5. Phase 5: Verification.
- Smoke test NIfTI và DICOM.
- Contract test shape/dtype/label range.
- Visual test mesh đủ 5 lớp.
- Export test đủ 4 định dạng mở được.

**Scope đã xác nhận**
1. Input: NIfTI + DICOM series.
2. Triton backend: Python backend với nnUNetPredictor.
3. 3D web: Mesh 3D.
4. Export: đủ NIfTI/STL/OBJ/GLTF.
5. GPU allocation: chỉ Triton dùng GPU, Gradio chạy CPU.
