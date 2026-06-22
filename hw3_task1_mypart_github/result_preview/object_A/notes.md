# Object A notes

Method: Real multi-view reconstruction + COLMAP + 3DGS

Input: phone video, extracted frames in data/object_A_raw/input

COLMAP notes:
- COLMAP GPU SIFT failed in the headless cloud terminal.
- Final successful COLMAP conversion used --no_gpu CPU mode.
- Registered images: 48, according to the successful conversion log.

3DGS notes:
- Training iterations: 7000
- Training hardware: NVIDIA GeForce RTX 5090
- Training log: logs/A_train_7000.log
- GPU log: logs/gpu/A_train_gpu.csv

Output:
- 3DGS model directory

Recommended file for teammate:
- A_3dgs_model/point_cloud/iteration_7000/point_cloud.ply
- Or the whole A_3dgs_model directory

Known issues:
- Early COLMAP attempts failed due to conda/time/Qt/OpenGL issues.
- Early 3DGS training attempt failed because CUDA extensions were not installed in the correct hw3_3dgs environment.
