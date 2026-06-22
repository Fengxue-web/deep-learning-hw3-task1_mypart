# Object B Final Asset Notes

## Method

Object B was generated with threestudio using a text-to-3D pipeline. The selected run used LatentNeRF-style text-to-3D optimization with a pretrained Stable Diffusion v1-5 2D diffusion model as guidance and SDS-style score distillation.

## Selected Prompt

a small blue ceramic coffee mug with a large handle, full object, centered

## Selected Run

- Object: Object B
- Route: text-to-3D generation
- Framework: threestudio
- Config: configs/latentnerf.yaml
- Seed: 11
- Steps: 1000
- Selected trial: object_B_mug_20260613_latentnerf_sweep_p2_blue_full_handle_1000
- Main exported asset: asset/model.obj
- Material file: asset/model.mtl

## Preview Files

- previews/preview_it1000.png
- previews/preview_turntable_it1000.mp4
- previews/test_views/

## Known Visual Issues

The result is recognizable as a mug-like generated object, but the geometry is not clean. The handle is imperfect and there are visible side artifacts / Janus-like ambiguity. The texture is blurry and generated rather than physically faithful. This should be reported honestly as a limitation of text-to-3D generation under the available time and compute budget.

## Handoff Recommendation

Use asset/model.obj together with asset/model.mtl and any texture files in the asset folder. If mesh import is unstable in the fusion pipeline, use the checkpoint and preview files as backup evidence, but the preferred handoff asset is the exported OBJ package.

## Runtime and Logs

See report_materials/runtime_summary.txt, logs/, and gpu/ for command/runtime/GPU records.
