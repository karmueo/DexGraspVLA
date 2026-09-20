#!/usr/bin/env python3
"""检查 RM75 短跑训练的日志、归一化器和 checkpoint。"""

import argparse
import json
import math
from pathlib import Path

import dill
import torch


def read_log_records(path: Path) -> list[dict]:
    """读取 JSON Lines 日志并返回非空记录。"""
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"日志第 {line_number} 行不是有效 JSON: {path}") from exc
    return records


def main() -> int:
    """解析参数并验证训练输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="短跑训练输出目录")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latest.ckpt",
        help="相对于 run-dir 的 checkpoint 路径",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"训练输出目录不存在或不是目录: {run_dir}")

    log_path = run_dir / "logs.json.txt"
    normalizer_path = run_dir / "normalizer.pkl"
    checkpoint_path = run_dir / args.checkpoint
    for path in (log_path, normalizer_path, checkpoint_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"训练产物缺失或为空: {path}")

    records = read_log_records(log_path)
    losses = [record["train_loss"] for record in records if "train_loss" in record]
    if not losses or not math.isfinite(float(losses[-1])):
        raise ValueError("日志中没有有限的最终 train_loss")

    payload = torch.load(checkpoint_path, map_location="cpu", pickle_module=dill)
    state_dicts = payload.get("state_dicts")
    if not isinstance(state_dicts, dict) or "model" not in state_dicts:
        raise ValueError("checkpoint 中缺少 state_dicts.model")

    print(f"[通过] loss={float(losses[-1]):.6f}，checkpoint={checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
