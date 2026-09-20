#!/usr/bin/env bash
# 使用两张 RTX 5090 启动 RM75 训练，附加参数原样传给 Hydra。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path="${RM75_VENV:-${repo_root}/.venv-rm75}"
port="${RM75_PORT:-25000}"

cd "${repo_root}"
export XFORMERS_DISABLED=1
export RM75_DISTRIBUTED_BACKEND=nccl
if [[ -z "${RM75_DINO_SOURCE:-}" && -f "${repo_root}/.deps/dinov2/hubconf.py" ]]; then
    export RM75_DINO_SOURCE="${repo_root}/.deps/dinov2"
fi

# 当前双卡为 NODE 拓扑；以下参数是已通过完整训练和短跑验证的稳定设置。
export NCCL_NTHREADS=256
export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

exec uv run --no-project --python "${venv_path}/bin/python" accelerate launch \
    --num_processes 2 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no \
    --main_process_port "${port}" \
    "${repo_root}/train.py" --config-name train_dexgraspvla_controller_workspace_rm75 \
    training.num_epochs=20 \
    optimizer.lr=1e-4 \
    dataloader.batch_size=64 \
    dataloader.num_workers=2 \
    val_dataloader.batch_size=16 \
    val_dataloader.num_workers=2 \
    task.dataset.val_ratio=0.1 \
    training.val_every=2 \
    training.sample_every=2 \
    training.checkpoint_every=1 \
    training.lr_scheduler=constant_with_warmup \
    training.lr_warmup_steps=500 \
    "$@"
