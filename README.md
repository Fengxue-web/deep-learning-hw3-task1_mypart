# HW3 Task 1：三类 3D 物体资产准备

本仓库是课程“深度学习与空间智能”HW3 Task 1 中本人负责部分的公开提交材料，主要包含三类独立 3D 物体资产的准备流程、复现脚本、输入数据说明、结果预览和方法比较材料。

## 1. 任务范围

本仓库只覆盖 Task 1 中的 3D Asset Preparation 个人部分：

- Object A：真实物体多视角重建，路线为手机视频/多视角图像 -> COLMAP -> 3D Gaussian Splatting。
- Object B：文本到 3D 生成，路线为 text prompt -> threestudio / LatentNeRF 风格优化 -> 3D object。
- Object C：单图到 3D 生成，路线为单张前景图 -> Stable Zero123 风格方法 -> 3D object。

本仓库不包含背景场景重建、场景融合渲染、最终漫游视频，也不包含 HW3 Task 2。

## 2. hw3_task1_mypart_github的具体结构

```text
.
├── README.md
├── .gitignore
├── data/
│   └── object_C_raw/
│       ├── input.jpg
│       └── object_C_rgba.png
├── scripts/
│   ├── 00_check_server.sh
│   ├── 01_setup_system.sh
│   ├── 02_setup_3dgs.sh
│   ├── 03_setup_threestudio.sh
│   ├── run_object_A_3dgs.sh
│   ├── run_object_B_text3d.sh
│   ├── run_object_C_image3d.sh
│   ├── collect_final_assets.sh
│   └── gpu_monitor.sh
├── result_preview/
│   ├── object_A/
│   ├── object_B/
│   └── object_C/
└── report_materials/
    └── README_report.md
```

其中：

- `scripts/`：用于在云端 GPU 环境中复现实验或检查环境的 shell 脚本。
- `data/object_C_raw/`：Object C 的原始输入图和处理后的 RGBA 前景图。
- `result_preview/`：三类 Object 的最终资产、预览图、日志和配置记录。
- `report_materials/README_report.md`：三类方法的结果说明、运行记录和对比材料。

## 3. 数据说明

Object A 原始输入是围绕真实马克杯拍摄的手机视频。由于视频文件体积较大，未直接放入 Git 仓库，已上传到 Google Drive：

```text
https://drive.google.com/drive/folders/1xt6MCu9QGwL-X8EUKWptdIcvAv1mtiWG?usp=sharing
```

Object C 的输入数据已放入仓库：

```text
data/object_C_raw/input.jpg
data/object_C_raw/object_C_rgba.png
```

Object B 不使用真实图像或视频，唯一语义输入是文本 prompt：

```text
a small blue ceramic coffee mug with a large handle, full object, centered
```

## 4. 实验环境

实际实验在 AutoDL RTX 5090 32GB 云端环境中完成。主要工具包括：

- COLMAP
- official 3D Gaussian Splatting
- threestudio
- Stable Diffusion v1-5
- Stable Zero123 风格单图生成流程

本仓库中的脚本假设云端工作目录为：

```bash
/root/autodl-tmp/hw3_task1_assets
```

建议先在新的云端机器上进行基础检查：

```bash
bash scripts/00_check_server.sh
```

再按需要配置系统工具、3DGS 和 threestudio：

```bash
bash scripts/01_setup_system.sh
bash scripts/02_setup_3dgs.sh
bash scripts/03_setup_threestudio.sh
```

## 5. 复现命令

### Object A：多视角重建 + 3DGS

先将 Object A 原始视频下载到云端：

```text
/root/autodl-tmp/hw3_task1_assets/data/object_A_raw/video.mp4
```

然后按顺序运行：

```bash
bash scripts/run_object_A_3dgs.sh check
bash scripts/run_object_A_3dgs.sh frames
bash scripts/run_object_A_3dgs.sh convert
bash scripts/run_object_A_3dgs.sh train
bash scripts/run_object_A_3dgs.sh render
bash scripts/run_object_A_3dgs.sh handoff
```

最终资产目录：

```text
final_assets/object_A/A_3dgs_model/
```

### Object B：文本到 3D 生成

Object B 只需要文本 prompt。最终采用的 prompt 为：

```text
a small blue ceramic coffee mug with a large handle, full object, centered
```

推荐运行顺序：

```bash
bash scripts/run_object_B_text3d.sh preflight
bash scripts/run_object_B_text3d.sh train
bash scripts/run_object_B_text3d.sh export-latest
```

如果要整理本次采用的最终结果，可运行：

```bash
bash scripts/run_object_B_text3d.sh collect-adopted
bash scripts/run_object_B_text3d.sh sanity-final
```

最终资产目录：

```text
final_assets/object_B/object_B_latentnerf_p2_blue_20260613_final/
```

### Object C：单图到 3D 生成

Object C 使用 `object_C_rgba.png` 作为前景图输入。推荐运行顺序：

```bash
bash scripts/run_object_C_image3d.sh prepare-input
bash scripts/run_object_C_image3d.sh preflight
bash scripts/run_object_C_image3d.sh timed-final-1200-threshold20
```

最终资产目录：

```text
final_assets/object_C/object_C_timed_1200_threshold20_20260613_081839/
```

如需检查最终资产汇总，可运行：

```bash
bash scripts/collect_final_assets.sh check
```

## 6. 结果概览

| Object | 方法 | 推荐资产 | 结果说明 | 最终采用耗时 |
|---|---|---|---|---:|
| Object A | COLMAP + 3DGS | `result_preview/object_A/A_3dgs_model/point_cloud/iteration_7000/point_cloud.ply` | 最接近真实物体，但包含部分地面/背景 | 核心重建 5:27.54 |
| Object B | threestudio LatentNeRF | `result_preview/object_B/object_B_latentnerf_p2_blue_20260613_final/asset/model.obj` | 文本生成的蓝色马克杯状资产，主体轮廓和把手相关结构可辨 | 4:19.06 |
| Object C | Stable Zero123 风格单图生成 | `result_preview/object_C/object_C_timed_1200_threshold20_20260613_081839/model.obj` | 单图生成的黑色马克杯资产，杯身、杯柄和横向纹理带可辨 | 4:21.31 |

更详细的方法比较见：

```text
report_materials/README_report.md
```

## 7. 已知限制

- Object A 的输入视频包含地面/背景，因此 3DGS 结果不是干净分割的单物体模型。
- Object B 是文本生成资产，几何连接和纹理层次仍较简化。
- Object C 只使用一张前景图，不可见侧面需要由模型补全。
- 本仓库不包含大型预训练模型权重，也不包含 Object A 原始视频文件本体。

## 8. 文件大小说明

仓库中保留了最终资产和预览证据，包括 `.ply`、`.obj`、`.mtl`、`.png`、`.jpg`、小型 `.mp4` 预览、日志和配置文件。Object A 原始视频未放入仓库，需通过上方 Google Drive 链接获取。

如需重新运行完整实验，请参考 `scripts/` 中的脚本和 `report_materials/README_report.md` 中的实验记录。
