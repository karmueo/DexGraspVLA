#!/usr/bin/env bash
# 在单卡或多卡环境启动 RM75 夹爪训练，附加参数直接传给 Hydra。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path="${RM75_VENV:-${repo_root}/.venv-rm75}"
processes="${RM75_NUM_PROCESSES:-1}"
port="${RM75_PORT:-25000}"
cd "${repo_root}"
export XFORMERS_DISABLED=1
uv run --no-project --python "${venv_path}/bin/python" accelerate launch \
    --num_processes "${processes}" --main_process_port "${port}" \
    "${repo_root}/train.py" --config-name train_dexgraspvla_controller_workspace_rm75 "$@"
