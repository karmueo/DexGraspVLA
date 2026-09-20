#!/usr/bin/env python3
"""递归检查 RM75 episode 的必需文件是否存在、非空且可读。"""

import argparse
import os
from pathlib import Path


DEFAULT_REQUIRED_FILES = (
    "proprio.hdf5",
    "gripper.mp4",
    "gripper.json",
    "mask_gripper.mp4",
)


def find_episode_dirs(root: Path) -> list[Path]:
    """递归返回名称匹配 episode_* 的目录。"""
    return sorted(path for path in root.rglob("episode_*") if path.is_dir())


def validate_file(path: Path) -> str | None:
    """返回文件异常原因；文件满足基本要求时返回 None。"""
    if not path.is_file():
        return "缺失"
    if path.stat().st_size == 0:
        return "空文件"
    if not os.access(path, os.R_OK):
        return "不可读"
    return None


def main() -> int:
    """解析参数并打印递归检查结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="递归检查的根目录")
    parser.add_argument(
        "--required-files",
        nargs="+",
        default=list(DEFAULT_REQUIRED_FILES),
        metavar="NAME",
        help="每个 episode 必须包含的文件名",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"检查根目录不存在或不是目录: {root}")

    episodes = find_episode_dirs(root)
    if not episodes:
        print(f"[失败] 未在 {root} 下找到 episode_* 目录")
        return 1

    errors = 0
    for episode in episodes:
        for filename in args.required_files:
            path = episode / filename
            reason = validate_file(path)
            if reason is not None:
                print(f"[{reason}] {path}")
                errors += 1

    if errors:
        print(f"[失败] 共检查 {len(episodes)} 个 episode，发现 {errors} 个异常文件")
        return 1
    print(f"[通过] 共检查 {len(episodes)} 个 episode，所有必需文件均存在、非空且可读")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
