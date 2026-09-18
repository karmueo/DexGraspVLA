#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将标注目录中的 json 文件复制回原始数据目录，保持相同的文件夹结构。

示例:
    python scripts/rm75/copy_json_back.py --src data/first_frames --dst data/raw
"""

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def find_json_files(src_dir: Path):
    """递归查找标注目录中的 JSON 文件。"""
    json_files = sorted(src_dir.rglob("*.json"))
    return json_files


def main():
    """将标注复制回同名 episode，保护已存在的输出文件。"""
    parser = argparse.ArgumentParser(description="将 json 标注文件复制回原始数据目录")
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="包含 json 的源目录",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="原始数据目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖目标目录中已存在的 json 文件",
    )
    args = parser.parse_args()

    src_dir = args.src.resolve()
    dst_dir = args.dst.resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")
    if not dst_dir.exists():
        raise FileNotFoundError(f"目标目录不存在: {dst_dir}")

    json_files = find_json_files(src_dir)
    if not json_files:
        raise ValueError(f"未在 {src_dir} 中找到 gripper.json 标注")

    print(f"源目录: {src_dir}")
    print(f"目标目录: {dst_dir}")
    print(f"共找到 {len(json_files)} 个 json 文件")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for json_path in tqdm(json_files, desc="复制 json"):
        rel_path = json_path.relative_to(src_dir)
        output_path = dst_dir / rel_path

        if output_path.exists() and not args.overwrite:
            skip_count += 1
            continue

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_path, output_path)
            success_count += 1
        except OSError as e:
            print(f"\n失败: {json_path} -> {output_path}, 错误: {e}")
            fail_count += 1

    print("\n" + "=" * 50)
    print("处理完成")
    print(f"总 json 数: {len(json_files)}")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
