#!/usr/bin/env bash
# 为 RTX 5090 创建 RM75 训练与 mask 预处理共用的 uv 环境。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${RM75_VENV:-${repo_root}/.venv-rm75}"
cutie_ref="v1.0"
cutie_source="${RM75_CUTIE_SOURCE:-${repo_root}/.deps/Cutie}"

if [[ ! -x "${venv_path}/bin/python" ]]; then
    uv venv --python 3.10 "${venv_path}"
fi
uv pip sync --python "${venv_path}/bin/python" --torch-backend cu128 "${repo_root}/requirements-rm75.lock"
if [[ ! -f "${cutie_source}/cutie/config/eval_config.yaml" ]]; then
    mkdir -p "$(dirname "${cutie_source}")"
    git clone --branch "${cutie_ref}" --depth 1 https://github.com/hkchengrex/Cutie.git "${cutie_source}"
fi
if [[ "$(git -C "${cutie_source}" describe --tags --exact-match 2>/dev/null)" != "${cutie_ref}" ]]; then
    echo "Cutie 源码必须处于 ${cutie_ref} 标签: ${cutie_source}" >&2
    exit 1
fi
"${venv_path}/bin/python" - "${cutie_source}" <<'PY'
"""将固定版本 Cutie 源码加入独立环境，避开上游 v1.0 的 wheel 打包问题。"""
import pathlib
import site
import sys

source = pathlib.Path(sys.argv[1]).resolve()
pth = pathlib.Path(site.getsitepackages()[0]) / "rm75-cutie-source.pth"
pth.write_text(f"{source}\n", encoding="utf-8")
PY
"${venv_path}/bin/python" "${repo_root}/scripts/check_rm75_env.py"
