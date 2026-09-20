## RM75 单臂夹爪训练

本文说明如何把 `data/org_data/0/episode_*` 演示数据转换为训练数据，并完成 RM75 单臂夹爪策略的环境配置、数据预处理、训练与验收。除特别说明外，命令均在项目根目录执行。

| 类型 | 名称 | 形状 | 含义 |
| --- | --- | --- | --- |
| 输入 | `rgbm` | `1×4×392×518` | 一帧夹爪相机 RGB 图像及目标 mask |
| 输入 | `right_state` | `1×8` | 7 维关节状态及 1 维夹爪状态 |
| 输出 | `action` | `64×8` | 未来 64 步、每步 8 维的动作 |

### 1. 准备环境与权重

#### 1.1 检查 uv

**命令**

```bash
uv --version
```

**含义**：确认系统已安装 `uv`。若命令不存在，请先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装。

| 项目 | 说明 |
| --- | --- |
| 参数 | 无 |
| 输入 | 系统中的 `uv` 可执行程序 |
| 输出 | `uv` 版本号 |

#### 1.2 创建 RM75 Python 环境

**命令**

```bash
bash scripts/setup_rm75_env.sh
```

已有 Cutie v1.0 源码或需要自定义环境目录时：

```bash
RM75_VENV=/path/to/venv RM75_CUTIE_SOURCE=/path/to/Cutie \
  bash scripts/setup_rm75_env.sh
```

**含义**：创建 Python 3.10 环境，同步锁定依赖和 PyTorch 2.7.1 / CUDA 12.8 依赖，安装 Cutie v1.0 源码，并自动执行环境检查。

| 名称 | 类型 | 默认值 | 输入/输出及含义 |
| --- | --- | --- | --- |
| `RM75_VENV` | 环境变量 | `.venv-rm75` | 输出虚拟环境目录 |
| `RM75_CUTIE_SOURCE` | 环境变量 | `.deps/Cutie` | Cutie 输入目录；不存在时自动克隆到该位置 |
| `requirements-rm75.lock` | 输入文件 | 固定 | 锁定的 Python 依赖 |
| `.venv-rm75/` | 输出目录 | — | 默认可运行环境 |

后续命令使用 `uv run --no-project --python .venv-rm75/bin/python`：`--python` 指定解释器，`--no-project` 避免按其他项目配置重新同步依赖。

#### 1.3 检查运行环境

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/check_rm75_env.py
```

**含义**：检查 CUDA、BF16 和关键模块。安装脚本已自动执行一次；环境变化或排障时可单独复查。

| 项目 | 说明 |
| --- | --- |
| 输入 | RM75 虚拟环境、CUDA 运行时及 GPU |
| 输出 | 各项检查结果；必要条件不满足时返回非零状态 |
| 文件修改 | 无 |

#### 1.4 下载必需权重

**命令**

```bash
mkdir -p weights/dinov2 weights/tracking
curl -fL --retry 3 -o weights/dinov2/dinov2_vitb14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
curl -fL --retry 3 -o weights/tracking/cutie-base-mega.pth \
  https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth
```

**含义**：下载训练使用的 DINOv2 ViT-B/14 和生成 mask 使用的 Cutie base mega 权重。`weights/` 不由 Git 跟踪。

| 参数/项目 | 说明 |
| --- | --- |
| `mkdir -p` | 递归创建目录，目录已存在时不报错 |
| `curl -fL` | HTTP 失败时报错，并跟随重定向 |
| `--retry 3` | 失败时最多重试 3 次 |
| `-o <文件>` | 指定输出文件 |
| 输入 | [DINOv2 官方权重](https://github.com/facebookresearch/dinov2#pretrained-models)、[Cutie v1.0 权重](https://github.com/hkchengrex/Cutie/releases/tag/v1.0) |
| 输出 | `weights/dinov2/dinov2_vitb14_pretrain.pth`、`weights/tracking/cutie-base-mega.pth` |

#### 1.5 校验必需权重

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/check_rm75_weights.py
```

**含义**：对比文件 SHA-256，排除下载不完整或文件被修改。

| 项目 | 说明 |
| --- | --- |
| `--dinov2 <文件>` | 可选；DINOv2 ViT-B/14 权重，默认 `weights/dinov2/dinov2_vitb14_pretrain.pth` |
| `--cutie <文件>` | 可选；Cutie base mega 权重，默认 `weights/tracking/cutie-base-mega.pth` |
| 输入 | 两个权重文件；预期摘要保存在验证脚本中 |
| 输出 | 每个文件显示 `[通过]`；缺失或摘要不一致时返回非零状态 |

#### 1.6 可选：下载 ViT-L/14 权重

**命令**

```bash
curl -fL --retry 3 -o weights/dinov2/dinov2_vitl14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/check_rm75_weights.py --vitl14 weights/dinov2/dinov2_vitl14_pretrain.pth
```

**含义**：下载并校验 ViT-L/14 权重。默认 RM75 配置使用 ViT-B/14，仅切换模型配置时需要本步骤。

| 项目 | 说明 |
| --- | --- |
| 输入 | DINOv2 ViT-L/14 官方权重地址，以及步骤 1.4 下载的默认权重 |
| 输出 | `weights/dinov2/dinov2_vitl14_pretrain.pth` 及三个权重的校验结果 |
| `--vitl14 <文件>` | 可选；提供时除默认权重外一并校验 ViT-L/14 权重 |

### 2. 从原始演示生成 Zarr

#### 2.1 检查 episode 目录

每段 episode 至少需要 `proprio.hdf5` 和 `gripper.mp4`；完成标注与 mask 生成后，应具有以下结构：

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

**命令**

```bash
find data/org_data/0 -maxdepth 2 -type f | sort
```

**含义**：列出来源目录下两层以内的文件，核对 episode 结构。

| 参数/项目 | 说明 |
| --- | --- |
| `data/org_data/0` | 输入的来源根目录 |
| `-maxdepth 2` | 最多遍历两层 |
| `-type f` | 只列出普通文件 |
| 输出 | 排序后的文件路径列表；不修改文件 |

`proprio.hdf5` 应包含关节动作、夹爪动作、关节状态、夹爪状态、各自时间戳和 `observations/images/cam_gripper_timestamp`。转换脚本以 `action/joint_action/timestamp` 为基准，在 50 ms 内最近邻对齐其他数据流，并按 URDF 关节限位和夹爪 `[0,1]` 范围归一化。

#### 2.2 提取视频首帧

所有 episode 已有有效 `gripper.json` 时，可跳过步骤 2.2 至 2.5。

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/extract_first_frames.py \
  --src data/org_data/0 --dst data/org_data/0_first_frames
```

**含义**：递归查找 `gripper.mp4`，提取第一帧，并保持原目录结构。

| 参数 | 必填 | 默认值 | 输入/输出及含义 |
| --- | --- | --- | --- |
| `--src <目录>` | 是 | 无 | 输入 episode 根目录 |
| `--dst <目录>` | 是 | 无 | 输出首帧根目录 |
| `--ext <后缀>` | 否 | `.jpg` | 输出图片后缀，可写 `.png` 或 `png` |
| `--skip-existing` | 否 | 开启 | 跳过已有首帧 |
| `--overwrite` | 否 | 关闭 | 覆盖已有首帧，优先于跳过逻辑 |
| 输出 | — | — | `data/org_data/0_first_frames/episode_*/gripper.jpg` 及处理统计 |

#### 2.3 标注候选目标

**命令**

```bash
labelme data/org_data/0_first_frames
```

**含义**：用 Labelme 打开首帧，在每张图片上绘制一个或多个候选目标多边形，并把 JSON 保存到图片所在目录。也可使用能导出相同 JSON 格式的工具。

| 项目 | 说明 |
| --- | --- |
| 路径参数 | 输入首帧根目录 |
| 输入 | `episode_*/gripper.jpg` |
| 约束 | 每张图片至少一个有效目标多边形；标签名含 `plate` 的形状会被忽略 |
| 输出 | `episode_*/gripper.json` |

#### 2.4 分页检查标注

本步骤需要图形桌面会话。

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python \
  scripts/rm75/view_gripper_annotations.py \
  --src data/org_data/0_first_frames --rows 4 --cols 4
```

**含义**：分页显示首帧和候选多边形；缺少图片或 JSON、空标注和无效多边形会显示红色提示。

| 参数/项目 | 必填 | 默认值 | 输入/输出及含义 |
| --- | --- | --- | --- |
| `--src <目录>` | 是 | 无 | 输入图片和同名 JSON 的根目录 |
| `--rows <正整数>` | 否 | `4` | 每页行数 |
| `--cols <正整数>` | 否 | `4` | 每页列数 |
| 交互 | — | — | 右键/下键：下一页；左键/上键：上一页；`q`：退出 |
| 输出 | — | — | 图形检查窗口；不修改图片或 JSON |

#### 2.5 复制标注回原始目录

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/copy_json_back.py \
  --src data/org_data/0_first_frames --dst data/org_data/0
```

**含义**：保持相对结构，把标注 JSON 复制回对应原始 episode。

| 参数 | 必填 | 默认值 | 输入/输出及含义 |
| --- | --- | --- | --- |
| `--src <目录>` | 是 | 无 | 输入标注根目录 |
| `--dst <目录>` | 是 | 无 | 输出原始 episode 根目录 |
| `--overwrite` | 否 | 关闭 | 覆盖已存在 JSON；默认跳过 |
| 输出 | — | — | `data/org_data/0/episode_*/gripper.json` 及复制统计 |

#### 2.6 生成 mask 视频

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/video_extract_masks_gripper.py \
  --src data/org_data/0 --weights weights/tracking/cutie-base-mega.pth --workers 1
```

**含义**：用 Cutie 跟踪首帧目标，生成与 `gripper.mp4` 等帧数的 `mask_gripper.mp4`。已有 mask 默认跳过。

多个候选目标时，脚本会依次弹出选择窗口；点击多边形或右侧列表，或按 `1`–`9` 选择，再按 Enter 确认。全部选择后才开始 GPU 跟踪；使用 `--overwrite` 重跑会重新选择。

| 参数 | 必填 | 默认值 | 输入/输出及含义 |
| --- | --- | --- | --- |
| `--src <目录...>` | 是 | 无 | 输入一个或多个 episode 根目录 |
| `--weights <文件>` | 否 | `weights/tracking/cutie-base-mega.pth` | 输入 Cutie v1.0 权重 |
| `--workers <整数>` | 否 | `1` | GPU 进程数，范围为 1 到可用 GPU 数 |
| `--overwrite` | 否 | 关闭 | 重建已有 mask |
| 其他输入 | — | — | `gripper.mp4`、`gripper.json` 和 CUDA GPU |
| 输出 | — | — | `episode_*/mask_gripper.mp4` |

#### 2.7 递归检查必需文件

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/check_episode_files.py \
  --root /data/scl_data/haier_train/data
```

**含义**：从 `--root` 开始递归查找任意层级中的 `episode_*` 目录，并检查每个 episode 的必需文件是否存在、非空且可读。路径中包含空格时也能正常检查；发现异常或没有找到 episode 时返回非零状态。

| 参数/项目 | 说明 |
| --- | --- |
| `--root <目录>` | 必填；要递归检查的根目录，也可使用 `data/org_data/0` |
| `--required-files <文件名...>` | 可选；自定义每个 episode 的必需文件，默认检查 `proprio.hdf5`、`gripper.mp4`、`gripper.json`、`mask_gripper.mp4` |
| 输入 | 根目录下任意层级中的所有 `episode_*` 目录 |
| 输出 | 逐项打印 `[缺失]`、`[空文件]` 或 `[不可读]`，最后打印汇总 |
| 退出码 | 全部通过为 `0`；存在异常或未找到 episode 为非零 |
| 文件修改 | 无 |

该脚本检查目录结构与文件基本状态，不解析 HDF5、JSON 或视频内容；文件内容和时间戳一致性会在步骤 2.8 转换 Zarr 时继续验证。
#### 2.8 转换为训练 Zarr

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/trans_hdf5_to_zarr_gripper.py \
  --src data/org_data/0 --dst data/gripper_zarr --workers 4
```

多个来源可连续输入：

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/trans_hdf5_to_zarr_gripper.py \
  --src data/org_data/0 data/org_data/1 --dst data/gripper_zarr --workers 4
```

**含义**：对齐 HDF5 时间戳，解码 RGB 与 mask，归一化动作和状态，并写入 Zarr。不同来源的同名 episode 会进入不同 `source_XXX` 帧目录。

| 参数 | 必填 | 默认值 | 输入/输出及含义 |
| --- | --- | --- | --- |
| `--src <目录...>` | 是 | 无 | 输入一个或多个 episode 根目录 |
| `--dst <目录>` | 否 | `data/gripper_zarr` | 输出 Zarr 目录 |
| `--frames-cache-dir <目录>` | 否 | 同级 `gripper_frames` | 输出解码帧缓存 |
| `--urdf <文件>` | 否 | `assets/rm75/rm_75.urdf` | 输入关节限位 |
| `--workers <整数>` | 否 | `4` | 并行进程数 |
| `--overwrite` | 否 | 关闭 | 覆盖已有非空输出 |
| 其他输入 | — | — | `proprio.hdf5`、`gripper.mp4`、`mask_gripper.mp4` |
| Zarr 输出 | — | — | `data/{action,state,gripper_image_paths,mask_image_paths}`、`meta/episode_ends` |
| 帧输出 | — | — | `data/gripper_frames/source_XXX/` |

Zarr 记录相对于 Zarr 目录的图像路径，移动数据时应一起保留 Zarr 和帧缓存。缺少 mask、视频与时间戳帧数不一致、无有效对齐帧或数据为空时，脚本会报错。

#### 2.9 验证 Zarr

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/check_gripper_zarr.py \
  --dataset data/gripper_zarr
```

**含义**：只读检查 episode 边界、动作与状态形状、路径数量以及全部 RGB/mask 文件路径。

| 参数/项目 | 说明 |
| --- | --- |
| `--dataset <目录>` | 必填；待检查的 Zarr 输入目录 |
| 输出 | episode 数和总帧数 |
| 失败 | 数组缺失、形状或边界错误、图像路径不存在时返回非零状态 |
| 文件修改 | 无，Zarr 以 `mode="r"` 打开 |

### 3. 配置与启动训练

#### 3.1 查看默认配置

**命令**

```bash
sed -n '1,180p' controller/config/train_dexgraspvla_controller_workspace_rm75.yaml
sed -n '1,120p' controller/config/task/grasp_rm75.yaml
```

**含义**：只读查看 RM75 主训练配置和数据集配置。

| 配置 | 默认值/输出含义 |
| --- | --- |
| 训练进程数 | `1` |
| BF16 | 开启 |
| `dataloader.batch_size` | `16`，多卡时为每卡 batch |
| `val_dataloader.batch_size` | `4` |
| 训练/验证 `num_workers` | 各 `4` |
| `training.num_epochs` | `50` |
| `optimizer.lr` | `1e-4` |
| `task.dataset.val_ratio` | `0`，默认不划验证集 |
| 输出 | 离线日志、每轮 `latest.ckpt`、按 `train_loss` 选出的最优 checkpoint |

#### 3.2 使用单张 RTX 5090 训练

**命令**

```bash
RM75_DATASET=data/gripper_zarr ./train_rm75_5090.sh
```

**含义**：使用一张 RTX 5090 和 BF16 启动训练。脚本固定启动 1 个训练进程，并使用已验证的生产参数。

| 配置 | 默认值 |
| --- | --- |
| 训练进程数 / GPU 数 | `1 / 1` |
| 每卡 batch / 全局 batch | `64 / 64` |
| 训练轮数 | `20` |
| 学习率 | `1e-4` |
| 训练/验证加载进程数 | 各 `2` |
| 验证集比例 | `0.1` |
| 验证、采样频率 | 每 `2` 轮 |
| checkpoint 频率 | 每轮 |
| 学习率调度 | `constant_with_warmup`，warmup 500 step |

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `RM75_DATASET` | `data/gripper_zarr` | 训练 Zarr 路径 |
| `RM75_VENV` | `.venv-rm75` | Python 环境 |
| `RM75_DINO_WEIGHTS` | ViT-B/14 本地权重 | DINOv2 权重 |
| `RM75_DINO_SOURCE` | 自动检测 `.deps/dinov2` | DINOv2 本地源码 |
| `RM75_PORT` | `25000` | accelerate 通信端口 |
| 输出 | `data/outputs/<日期>/<运行名>/` | 日志、归一化器和 checkpoint |

#### 3.3 使用两张 RTX 5090 训练

**命令**

```bash
RM75_DATASET=data/gripper_zarr ./train_rm75_5090x2.sh
```

**含义**：使用两张 RTX 5090、两个训练进程和 NCCL 启动训练。每卡 batch 64，全局 batch 128。

| 配置 | 默认值及含义 |
| --- | --- |
| 训练进程数 / GPU 数 | `2 / 2` |
| 每卡 batch / 全局 batch | `64 / 128` |
| `NCCL_NTHREADS` | `256` |
| `NCCL_MIN_NCHANNELS/MAX_NCHANNELS` | `1/4` |
| `NCCL_P2P_DISABLE`、`NCCL_IB_DISABLE` | `1`，适配当前无 NVLink 的 NODE 拓扑 |
| 其他训练参数 | 与步骤 3.2 相同 |
| 输出 | 由主进程写入训练目录 |

两个脚本都会自动使用 `.deps/dinov2` 本地源码，避免 PyTorch Hub 访问 GitHub 触发限流；显式设置 `RM75_DINO_SOURCE` 时优先使用指定目录。

#### 3.4 覆盖训练参数

**命令**

```bash
RM75_DATASET=data/gripper_zarr ./train_rm75_5090.sh \
  training.num_epochs=20 dataloader.batch_size=2 dataloader.num_workers=2 \
  task.dataset.val_ratio=0.1 training.val_every=1
```

**含义**：把脚本后的参数原样传给 Hydra；示例训练 20 轮、每卡 batch 2、2 个加载进程，划分 10% 验证 episode 并每轮验证。

| 参数 | 输入类型 | 输出影响 |
| --- | --- | --- |
| `training.num_epochs` | 正整数 | 训练轮数 |
| `dataloader.batch_size` | 正整数 | 每卡 batch 和显存占用 |
| `dataloader.num_workers` | 非负整数 | 训练读取吞吐 |
| `val_dataloader.num_workers` | 非负整数 | 验证读取吞吐 |
| `task.dataset.val_ratio` | 通常为 `[0,1)` | 验证 episode 比例 |
| `training.val_every` | 正整数 | 验证频率 |
| `training.lr_scheduler` | 字符串 | 学习率变化方式 |
| `training.lr_warmup_steps` | 非负整数 | warmup 步数 |

只有划出验证 episode 且执行验证轮次时才会产生 `val_loss`。

#### 3.5 启动多卡训练

**命令**

```bash
RM75_DATASET=data/gripper_zarr RM75_PORT=25000 ./train_rm75_5090x2.sh
```

**含义**：启动两个训练进程，通常对应两张 GPU。全局 batch 为“每卡 batch × 进程数 × 梯度累积步数”。

| 项目 | 说明 |
| --- | --- |
| `train_rm75_5090x2.sh` | 固定启动两个训练进程，每个进程对应一张 GPU |
| `RM75_PORT=25000` | 输入通信端口；冲突时更换 |
| 其他输入 | 两张可用 GPU、训练数据、源码和权重 |
| 输出 | 与单卡相同，由主进程写入训练目录 |

#### 3.6 使用两张 RTX 5090 实测配置

**命令**

```bash
RM75_DATASET=data/gripper_zarr ./train_rm75_5090x2.sh training.num_epochs=5 training.val_every=1 training.lr_warmup_steps=100
```

**含义**：使用双卡脚本内置的稳定 NCCL 参数训练五轮并逐轮验证。`NCCL_NTHREADS=256` 可避免当前环境中的 CUDA 非法内存访问。

| 参数 | 值及含义 |
| --- | --- |
| `NCCL_NTHREADS` | 脚本内置 `256`，设置每个通信块的线程数 |
| `NCCL_MIN_NCHANNELS/MAX_NCHANNELS` | 脚本内置 `1/4`，限制通信通道数 |
| `NCCL_P2P_DISABLE`、`NCCL_IB_DISABLE` | 脚本内置 `1`，禁用 P2P 和 InfiniBand |
| `RM75_DISTRIBUTED_BACKEND` | 脚本内置 `nccl` GPU 分布式后端 |
| `dataloader.batch_size` | 每卡 64，全局 128 |
| 输出 | 五轮日志与 checkpoint；实测第 5 轮 `train_loss=0.02407`、`val_loss=0.02699` |

#### 3.7 映射旧 Zarr 绝对路径

新转换的数据使用相对路径，无需执行本步骤。
**命令**

```bash
RM75_DATASET=/path/to/old/gripper_zarr ./train_rm75_5090.sh \
  '++task.dataset.path_prefix_map={/old/image/root:/current/image/root}'
```

**含义**：运行时把旧数据中的绝对路径前缀映射到当前机器，不改写原 Zarr。

| 参数/项目 | 输入/输出及含义 |
| --- | --- |
| `/path/to/old/gripper_zarr` | 输入旧 Zarr |
| `/old/image/root` | 输入旧路径前缀 |
| `/current/image/root` | 输入当前路径前缀 |
| `++task.dataset.path_prefix_map` | 新增或强制覆盖 Hydra 映射字典 |
| 输出 | 正常训练产物 |

### 4. 短跑验收与测试

#### 4.1 执行三步短跑

**命令**

```bash
RM75_DATASET=data/gripper_zarr ./train_rm75_5090.sh \
  hydra.run.dir=data/outputs/rm75_check \
  training.num_epochs=1 training.max_train_steps=3 \
  training.sample_every=1 training.checkpoint_every=1
```

**含义**：执行 1 个 epoch、最多 3 个训练 batch，触发动作采样并保存 checkpoint。重复执行时请更换输出目录。

| 参数 | 值及输入/输出含义 |
| --- | --- |
| `hydra.run.dir` | 输出到 `data/outputs/rm75_check` |
| `training.num_epochs` | 输入 `1` 轮 |
| `training.max_train_steps` | 每轮最多输入 3 个 batch |
| `training.sample_every` | 每轮执行采样 |
| `training.checkpoint_every` | 每轮保存 checkpoint |
| 输出 | `logs.json.txt`、`normalizer.pkl`、`checkpoints/latest.ckpt` |

#### 4.2 检查短跑输出

**命令**

```bash
uv run --no-project --python .venv-rm75/bin/python scripts/rm75/check_training_output.py \
  --run-dir data/outputs/rm75_check
```

**含义**：检查最终 loss、归一化器，并在 CPU 上重读 checkpoint。步骤 4.1 完成采样也说明 `predict_action(obs_dict)` 已成功执行。

| 参数/项目 | 说明 |
| --- | --- |
| `--run-dir <目录>` | 必填；步骤 4.1 的训练输出目录 |
| `--checkpoint <相对路径>` | 可选；相对于 `run-dir` 的 checkpoint，默认 `checkpoints/latest.ckpt` |
| 输入 | 日志、归一化器和 checkpoint |
| 输出 | 最终 loss 和 checkpoint 路径 |
| 失败 | loss 非有限值、文件或模型状态缺失时返回非零状态 |

#### 4.3 运行专项测试

**命令**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 XFORMERS_DISABLED=1 \
  uv run --no-project --python .venv-rm75/bin/python -m pytest tests/test_rm75_pipeline.py -q
```

**含义**：测试时间对齐、裁剪与归一化、路径映射、序列补齐、颜色通道、缺失文件、多来源重名及启动参数。

| 参数/项目 | 输入/输出及含义 |
| --- | --- |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` | 禁止自动加载第三方 pytest 插件 |
| `XFORMERS_DISABLED=1` | 禁用 xFormers 路径 |
| `-m pytest` | 通过当前 Python 运行 pytest |
| `-q` | 简洁输出 |
| 输入 | `tests/test_rm75_pipeline.py` 及被测代码 |
| 输出 | 测试统计；失败时返回非零状态 |
