## RM75 单臂夹爪训练

本节说明如何把 `data/org_data/0/episode_*` 形式的演示数据转换为训练数据，并完成 RM75 单臂夹爪策略的配置、训练和检查。以下命令均在项目根目录执行。模型使用一帧夹爪相机 RGB 图像及目标 mask（`rgbm: 1×4×294×518`）、7 维关节加 1 维夹爪状态（`right_state: 1×8`），预测 64 步、每步 8 维的动作。

### 1. 准备环境与权重

先安装 `uv`，再创建独立的 Python 3.10 环境。安装脚本使用锁定依赖和 PyTorch 2.7.1 / CUDA 12.8，安装 Cutie v1.0 源码，并执行 CUDA、BF16 和关键模块检查：

```bash
bash scripts/setup_rm75_env.sh
uv run --no-project --python .venv-rm75/bin/python scripts/check_rm75_env.py
```

默认环境位于 `.venv-rm75/`。后续命令使用 `uv run --no-project --python .venv-rm75/bin/python` 指定该环境；`--no-project` 避免启动时按其他项目配置重新同步依赖。已有 Cutie v1.0 源码时，可在执行安装脚本前设置 `RM75_CUTIE_SOURCE=/path/to/Cutie`。

权重放在项目根目录的 `weights/`，该目录不由 Git 跟踪。RM75 训练需要 DINOv2 ViT-B/14；生成 mask 需要 Cutie base mega。分别从 [DINOv2 官方模型列表](https://github.com/facebookresearch/dinov2#pretrained-models)和 [Cutie v1.0 发布页](https://github.com/hkchengrex/Cutie/releases/tag/v1.0)下载：

```bash
mkdir -p weights/dinov2 weights/tracking
curl -fL --retry 3 -o weights/dinov2/dinov2_vitb14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
curl -fL --retry 3 -o weights/tracking/cutie-base-mega.pth \
  https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth
```

当前代码默认使用上述两个本地路径。可用 SHA-256 核对下载文件：

```bash
cat <<'SHA256' | sha256sum -c -
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73  weights/dinov2/dinov2_vitb14_pretrain.pth
9c05402ee36d3a356fb72715d263ba7e1ea06ad3bada48c1306491792da43023  weights/tracking/cutie-base-mega.pth
SHA256
```

DINOv2 ViT-L/14 仅供切换其他模型配置时使用，可选下载并校验：

```bash
curl -fL --retry 3 -o weights/dinov2/dinov2_vitl14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
echo 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428  weights/dinov2/dinov2_vitl14_pretrain.pth' | sha256sum -c -
```

### 2. 从原始演示生成 Zarr

原始数据按 episode 分目录。每段至少需要 `proprio.hdf5` 和 `gripper.mp4`；生成 mask 后，每段还应有 `gripper.json` 和 `mask_gripper.mp4`：

```text
data/org_data/0/
├── episode_0/
│   ├── proprio.hdf5
│   ├── gripper.mp4
│   ├── gripper.json
│   └── mask_gripper.mp4
├── episode_1/
│   └── ...
└── ...
```

`proprio.hdf5` 包含关节动作、夹爪动作、关节状态、夹爪状态，以及各自时间戳和 `observations/images/cam_gripper_timestamp`。转换脚本以 `action/joint_action/timestamp` 为基准，在 50 ms 内最近邻对齐其他数据流；按 `assets/rm75/rm_75.urdf` 中的关节限位及夹爪 `[0,1]` 范围归一化，并保留参考数据的末尾夹爪动作裁剪规则。

若原始 episode 已有有效的 `gripper.json`，可跳过首帧提取和标注回填。否则先提取首帧：

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/extract_first_frames.py \
  --src data/org_data/0 --dst data/org_data/0_first_frames
```

脚本会得到类似 `data/org_data/0_first_frames/episode_0/gripper.jpg` 的图片。用 Labelme 或能导出相同 JSON 格式的工具，在每张图片上标注**一个目标多边形**，保存为同目录的 `gripper.json`。标注读取器会忽略标签名包含 `plate` 的形状，并要求剩余形状恰好有一个有效目标多边形。

复制标注前，可在图形桌面会话中分页检查图片和多边形。默认每页显示 4×4 张；`--rows` 和 `--cols` 可调整行列数。缺少图片或 JSON、空标注及无效多边形会在对应格子显示红色提示。按右键或下键查看下一页，左键或上键返回上一页，按 `q` 退出：

```bash
uv run --no-project --python .venv-rm75/bin/python \
  scripts/rm75/view_gripper_annotations.py \
  --src data/org_data/0_first_frames --rows 4 --cols 4
```

检查完成后，把标注复制回原始 episode：

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/copy_json_back.py \
  --src data/org_data/0_first_frames --dst data/org_data/0
```

然后用 Cutie 跟踪首帧目标，生成与 `gripper.mp4` 等帧数的 `mask_gripper.mp4`：

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/video_extract_masks_gripper.py \
  --src data/org_data/0 --weights weights/tracking/cutie-base-mega.pth --workers 1
```

已有 mask 默认跳过。需要重新生成时添加 `--overwrite`。整目录转换要求**每个**待处理 episode 都有可用 mask；当前 `data/org_data/0` 中有部分 episode 尚缺标注或 mask，需先补齐。可用下列命令列出缺失项：

```bash
for episode in data/org_data/0/episode_*; do
  test -f "$episode/gripper.json" || echo "缺少标注: $episode"
  test -f "$episode/mask_gripper.mp4" || echo "缺少 mask: $episode"
done
```

最后转换为训练用 Zarr：

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/trans_hdf5_to_zarr_gripper.py \
  --src data/org_data/0 --dst data/gripper_zarr --workers 4
```

转换结果包含 `data/gripper_zarr/data/{action,state,gripper_image_paths,mask_image_paths}` 和 `data/gripper_zarr/meta/episode_ends`。`action`、`state` 都是 8 维；视频解码帧默认保存在同级的 `data/gripper_frames/source_000/`，Zarr 中记录相对于 Zarr 目录的图像路径。移动数据时应一起保留 Zarr 和帧缓存。多个来源可在 `--src` 后连续列出，例如 `--src data/org_data/0 data/org_data/1`；不同来源的同名 episode 会进入不同的 `source_XXX` 子目录。

转换时如缺少 mask、视频与时间戳帧数不一致、无有效对齐帧或数据为空，脚本会报错。已有非空输出默认不会覆盖；确认需要重建时才添加 `--overwrite`。也可通过 `--frames-cache-dir` 和 `--urdf` 指定其他帧缓存及 URDF 路径。

用下面的检查确认 Zarr 及其图像路径可读取：

```bash
uv run --no-project --python .venv-rm75/bin/python python - <<'PY'
from pathlib import Path
import zarr

dataset_path = Path("data/gripper_zarr")
root = zarr.open(str(dataset_path), mode="r")
frames = int(root["meta/episode_ends"][-1])
assert frames > 0
assert root["data/action"].shape == root["data/state"].shape == (frames, 8)
for name in ("gripper_image_paths", "mask_image_paths"):
    paths = root[f"data/{name}"]
    assert len(paths) == frames
    assert (dataset_path / str(paths[0])).is_file()
print(f"数据可用：{len(root['meta/episode_ends'])} episodes，{frames} frames")
PY
```

### 3. 配置与启动训练

RM75 专用配置为 `controller/config/train_dexgraspvla_controller_workspace_rm75.yaml` 和 `controller/config/task/grasp_rm75.yaml`。`train_rm75.sh` 内部通过 `uv run` 使用 `.venv-rm75/`，默认单进程、BF16、训练 batch size 16、验证 batch size 4、各 4 个数据加载进程，训练 50 个 epoch；学习率为 `1e-4`。每轮保存 `latest.ckpt` 和按训练 loss 选出的最优 checkpoint。默认不开启验证集划分（`val_ratio: 0`），日志使用离线模式。

`RM75_DATASET` 指向转换后的 Zarr；`RM75_DINO_WEIGHTS` 可覆盖默认权重路径。省略 `RM75_DINO_SOURCE` 时，PyTorch Hub 会获取 DINOv2 源码；离线训练时应把它设为本地 DINOv2 源码目录。

```bash
RM75_DATASET=data/gripper_zarr bash train_rm75.sh
```

如需直接使用 `uv` 启动，等价命令为：

```bash
RM75_DATASET=data/gripper_zarr XFORMERS_DISABLED=1 \
  uv run --no-project --python .venv-rm75/bin/python accelerate launch \
  --num_processes 1 --main_process_port 25000 train.py \
  --config-name train_dexgraspvla_controller_workspace_rm75
```

启动脚本把后续参数原样交给 Hydra，例如调整 epoch、batch size、数据加载进程和验证集比例：

```bash
RM75_DATASET=data/gripper_zarr bash train_rm75.sh \
  training.num_epochs=20 dataloader.batch_size=2 dataloader.num_workers=2 \
  task.dataset.val_ratio=0.1 training.val_every=1
```

只有划出验证 episode 且执行验证轮次时才会产生 `val_loss`。多卡启动可设置 `RM75_NUM_PROCESSES` 和 `RM75_PORT`，例如两卡使用 `RM75_NUM_PROCESSES=2 RM75_PORT=25000`。`dataloader.batch_size` 是每卡的 batch size。训练产物默认写入 `data/outputs/<日期>/<运行名>/`。

两张 RTX 5090 的 NCCL 实测配置如下。当前环境将 `NCCL_NTHREADS` 设为 `256` 可避免训练时的 CUDA 非法内存访问。每卡 batch 64、全局 batch 128，已完成五轮训练并每轮验证：

```bash
NCCL_NTHREADS=256 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=4 \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
RM75_DATASET=data/gripper_zarr RM75_DINO_SOURCE=.deps/dinov2 \
RM75_NUM_PROCESSES=2 RM75_DISTRIBUTED_BACKEND=nccl bash train_rm75.sh \
  training.num_epochs=5 dataloader.batch_size=64 dataloader.num_workers=2 \
  val_dataloader.num_workers=2 task.dataset.val_ratio=0.1 training.val_every=1 \
  training.lr_scheduler=constant_with_warmup training.lr_warmup_steps=100
```

这套配置的第五轮 `train_loss` 为 0.02407，`val_loss` 为 0.02699，训练过程中未出现 CUDA 非法内存访问。

若读取参考项目生成的旧 Zarr，其中图像路径仍为旧机器上的绝对路径，可通过 `path_prefix_map` 映射到当前机器；新转换的数据无需设置：

```bash
RM75_DATASET=/path/to/old/gripper_zarr bash train_rm75.sh \
  '++task.dataset.path_prefix_map={/old/image/root:/current/image/root}'
```

### 4. 短跑验收与测试

先用一个已完成 mask 的数据集执行 1 个 epoch、最多 3 个训练 batch。该命令同时触发一次仅依赖观测的动作采样，并在指定输出目录保存 checkpoint；重复执行时请换一个输出目录：

```bash
RM75_DATASET=data/gripper_zarr bash train_rm75.sh \
  hydra.run.dir=data/outputs/rm75_check \
  training.num_epochs=1 training.max_train_steps=3 \
  training.sample_every=1 training.checkpoint_every=1
```

训练退出后，检查最后一条训练 loss 为有限数、归一化器存在，并确认 `latest.ckpt` 能重新读取：

```bash
uv run --no-project --python .venv-rm75/bin/python python - <<'PY'
from pathlib import Path
import json
import math
import dill
import torch

run_dir = Path("data/outputs/rm75_check")
records = [json.loads(line) for line in (run_dir / "logs.json.txt").read_text().splitlines()]
assert records and math.isfinite(records[-1]["train_loss"])
assert (run_dir / "normalizer.pkl").is_file()
checkpoint = run_dir / "checkpoints/latest.ckpt"
payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
assert "model" in payload["state_dicts"]
print(f"训练正常：loss={records[-1]['train_loss']:.6f}，checkpoint={checkpoint}")
PY
```

训练进程顺利完成 `sample_every=1` 的采样阶段，说明策略的 `predict_action(obs_dict)` 调用已执行。运行 RM75 专项测试，覆盖时间对齐、裁剪与归一化、路径映射、序列补齐、颜色通道、缺失文件、多来源重名和启动参数：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 XFORMERS_DISABLED=1 \
  uv run --no-project --python .venv-rm75/bin/python -m pytest tests/test_rm75_pipeline.py -q
```
