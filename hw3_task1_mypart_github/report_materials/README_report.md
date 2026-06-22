# HW3 Task 1 物体资产结果与方法比较

本文档说明仓库中三类 3D 物体资产的实验结果、文件位置和方法对比。三类资产分别对应 HW3 Task 1 要求的三种技术路线：

- Object A：真实物体多视角重建，使用 COLMAP 和 3D Gaussian Splatting。
- Object B：文本到 3D 生成，使用 threestudio / LatentNeRF 风格优化。
- Object C：单图到 3D 生成，使用 Stable Zero123 风格流程。

Object A 原始手机视频体积较大，未直接放入 Git 仓库；主 `README.md` 中提供了外部网盘链接。

## 1. 资产总览

| Object | 技术路线 | 主要输入 | 主要输出 | 仓库内证据 |
|---|---|---|---|---|
| Object A | 多视角重建 | 真实马克杯的 360 度手机视频 | 3DGS 模型目录，核心文件为 `point_cloud/iteration_7000/point_cloud.ply` | `result_preview/object_A/A_3dgs_model/` |
| Object B | 文本到 3D 生成 | 文本 prompt | OBJ 网格、MTL 材质和贴图 | `result_preview/object_B/object_B_latentnerf_p2_blue_20260613_final/` |
| Object C | 单图到 3D 生成 | 一张 RGBA 前景图 | OBJ 网格、MTL 材质和贴图 | `result_preview/object_C/object_C_timed_1200_threshold20_20260613_081839/` |

Object A 和 Object C 使用真实拍摄马克杯相关输入；Object B 使用纯文本 prompt 生成独立的虚拟马克杯资产。

## 2. Object A：多视角重建

Object A 使用真实多视角重建路线。实验流程为手机视频抽帧、COLMAP 相机位姿估计、3D Gaussian Splatting 训练和多视角渲染预览。

最终资产目录：

```text
result_preview/object_A/A_3dgs_model/
```

核心文件：

```text
result_preview/object_A/A_3dgs_model/point_cloud/iteration_7000/point_cloud.ply
```

该路线在已观察视角上保留了真实拍摄外观。训练视角渲染图中可以辨认杯身、杯柄、杯口、深色外观和 logo。由于输入视频包含地面/背景，最终 3DGS 资产中也保留了部分背景信息。

记录的运行时间：

| 阶段 | 耗时 | 资源记录 |
|---|---:|---|
| COLMAP 转换，使用 `--no_gpu` | 3:14.13 | CPU 阶段，成功注册 48 张图像 |
| 3DGS 训练，7000 iterations | 2:13.41 | 采样记录中的 GPU 峰值约 1551 MiB |
| 渲染预览 | 0:28.86 | 生成 48 张训练视角 PNG 预览图 |

Object A 核心重建耗时按 COLMAP 转换加 3DGS 训练计算，为 **5:27.54**。如果把渲染预览计入，总耗时为 **5:56.40**。

## 3. Object B：文本到 3D 生成

Object B 使用纯文本生成路线。最终采用 threestudio 的 `configs/latentnerf.yaml`，结合本地缓存的 Stable Diffusion v1-5 和 LatentNeRF / SDS 风格优化生成 3D 物体。

最终 prompt：

```text
a small blue ceramic coffee mug with a large handle, full object, centered
```

最终资产目录：

```text
result_preview/object_B/object_B_latentnerf_p2_blue_20260613_final/
```

mesh 文件组合：

```text
asset/model.obj
asset/model.mtl
asset/texture_kd.jpg
```

Object B 导出的 OBJ 在 3D 查看器中呈现为蓝色马克杯状资产，主体轮廓和把手相关结构可辨，符合文本 prompt 中对蓝色陶瓷马克杯的目标描述。局部几何连接、边界细节和纹理层次仍较简化，这是该 text-to-3D 结果的主要视觉限制。

记录的运行时间：

| 阶段 | 耗时 | 资源记录 |
|---|---:|---|
| 1000-step LatentNeRF 训练 | 2:45.00 | 采样记录中的 GPU 峰值约 7739 MiB |
| Mesh 导出 | 1:34.06 | 采样记录中的 GPU 峰值约 5199 MiB |

Object B 最终采用耗时为 **4:19.06**，统计口径为训练加 mesh 导出。

## 4. Object C：单图到 3D 生成

Object C 使用单图生成 3D 路线。输入是一张经过背景移除后的 RGBA 前景图。最终采用 Stable Zero123 风格的 threestudio 流程，使用低显存 PyTorch encoding 配置，并用固定 threshold 导出 mesh。

仓库内输入文件：

```text
data/object_C_raw/input.jpg
data/object_C_raw/object_C_rgba.png
```

最终资产目录：

```text
result_preview/object_C/object_C_timed_1200_threshold20_20260613_081839/
```

mesh 文件组合：

```text
model.obj
model.mtl
texture_kd.jpg
```

Object C 导出的 OBJ 在 3D 查看器中可以看到黑色杯身、杯柄以及横向暖色纹理带，整体轮廓与输入马克杯较一致。由于该路线只使用单张前景图，不可见侧面的结构和纹理由模型推断；杯口、局部表面细节和背面纹理仍存在一定简化。

记录的运行时间：

| 阶段 | 耗时 | 资源记录 |
|---|---:|---|
| 1200-step continuation | 2:58.42 | 采样记录中的 GPU 峰值约 6745 MiB |
| threshold 20.0 mesh 导出 | 1:22.89 | 采样记录中的 GPU 峰值约 1513 MiB |

Object C 最终采用耗时为 **4:21.31**，统计口径为 1200-step continuation 加 threshold 20 mesh 导出。

## 5. 方法比较

| 方法 | Object | 输入要求 | 几何表现 | 纹理表现 | 最终采用耗时 |
|---|---|---|---|---|---:|
| 多视角重建 + 3DGS | A | 需要手机视频或多张图片，并需要 COLMAP 位姿估计 | 多视角观测约束最强，杯身和杯柄清晰；同时保留部分背景 | 使用真实拍摄外观 | 核心重建 5:27.54 |
| 文本到 3D 生成 | B | 只需要文本 prompt | 蓝色马克杯主体轮廓和把手相关结构可辨，局部细节较简化 | 生成纹理，颜色目标明确，细节层次有限 | 4:19.06 |
| 单图到 3D 生成 | C | 需要一张干净的 RGBA 前景图 | 黑色杯身和杯柄可辨，不可见区域由模型推断 | 黑色外观和横向暖色纹理带可见，局部细节较弱 | 4:21.31 |

三种方法的输入条件差异明显。Object A 使用多视角真实观测，几何约束最充分；Object B 只依赖文本 prompt，生成自由度最高；Object C 使用单张真实前景图，在输入成本和真实外观约束之间取得折中。

## 6. 实现说明与局限

| Object | 实现说明与局限 |
|---|---|
| Object A | COLMAP 最终采用 CPU `--no_gpu` 模式完成位姿估计；3DGS 输出中包含输入视频里的部分地面/背景。 |
| Object B | 文本生成结果完成 OBJ/MTL/texture 导出；主体形态与颜色可辨，局部几何连接和纹理细节仍较简化。 |
| Object C | 单图生成结果完成 OBJ/MTL/texture 导出；杯身、杯柄和横向纹理带可见，不可见侧面由单图生成模型补全。 |

以上耗时和显存均来自成功运行记录。显存数值是 `nvidia-smi` 采样峰值，不代表理论最大显存占用。
