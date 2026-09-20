#!/usr/bin/env python3
"""检查 RM75 训练 Zarr 的数组形状、episode 边界和全部图像路径。"""

import argparse
from pathlib import Path

import numpy as np
import zarr


def require_array(root, key: str):
    """读取必需数组，并在缺失时给出明确错误。"""
    try:
        return root[key]
    except KeyError as exc:
        raise ValueError(f"缺少 Zarr 数组: {key}") from exc


def main() -> int:
    """解析参数并验证 Zarr 数据集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="待检查的 Zarr 目录")
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        parser.error(f"Zarr 目录不存在或不是目录: {dataset}")

    root = zarr.open(str(dataset), mode="r")
    action = require_array(root, "data/action")
    state = require_array(root, "data/state")
    episode_ends = np.asarray(require_array(root, "meta/episode_ends")[:])

    if episode_ends.ndim != 1 or len(episode_ends) == 0:
        raise ValueError("meta/episode_ends 必须是一维非空数组")
    if np.any(episode_ends <= 0) or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("meta/episode_ends 必须为严格递增的正整数")

    frames = int(episode_ends[-1])
    expected_shape = (frames, 8)
    if action.shape != expected_shape:
        raise ValueError(f"data/action 形状错误: {action.shape}，期望 {expected_shape}")
    if state.shape != expected_shape:
        raise ValueError(f"data/state 形状错误: {state.shape}，期望 {expected_shape}")

    missing_paths: list[Path] = []
    for name in ("gripper_image_paths", "mask_image_paths"):
        paths = require_array(root, f"data/{name}")
        if len(paths) != frames:
            raise ValueError(f"data/{name} 长度错误: {len(paths)}，期望 {frames}")
        for value in paths[:]:
            path = Path(str(value))
            resolved = path if path.is_absolute() else dataset / path
            if not resolved.is_file():
                missing_paths.append(resolved)

    if missing_paths:
        for path in missing_paths:
            print(f"[缺失] {path}")
        print(f"[失败] 共发现 {len(missing_paths)} 个图像路径不存在")
        return 1

    print(f"[通过] {len(episode_ends)} episodes，{frames} frames，全部图像路径可读取")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
