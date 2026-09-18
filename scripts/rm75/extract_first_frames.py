#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从指定目录递归提取所有视频的第一帧，保存到新目录并保持原有文件夹结构。

示例:
    python scripts/rm75/extract_first_frames.py --src data/raw --dst data/first_frames
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

VIDEO_NAME = "gripper.mp4"


def extract_first_frame(video_path: Path):
    """读取视频首帧，失败时返回 None。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None
    return frame


def find_videos(src_dir: Path):
    """仅查找用于标注的 gripper.mp4。"""
    return sorted(src_dir.rglob(VIDEO_NAME))


def main():
    """按 episode 结构批量保存首帧。"""
    parser = argparse.ArgumentParser(description="提取目录下全部视频的第一帧，并保持文件夹结构")
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="源视频目录",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="输出目录，结构与 src 保持一致",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".jpg",
        help="输出图片后缀，默认 .jpg",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="跳过已存在的输出文件（默认开启）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件",
    )
    args = parser.parse_args()

    src_dir = args.src.resolve()
    dst_dir = args.dst.resolve()
    output_ext = args.ext if args.ext.startswith(".") else f".{args.ext}"
    skip_existing = args.skip_existing and not args.overwrite

    if not src_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")

    videos = find_videos(src_dir)
    if not videos:
        raise ValueError(f"未在 {src_dir} 中找到 gripper.mp4")

    print(f"源目录: {src_dir}")
    print(f"输出目录: {dst_dir}")
    print(f"共找到 {len(videos)} 个视频")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for video_path in tqdm(videos, desc="提取第一帧"):
        rel_path = video_path.relative_to(src_dir)
        output_path = dst_dir / rel_path.with_suffix(output_ext)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and output_path.exists():
            skip_count += 1
            continue

        frame = extract_first_frame(video_path)
        if frame is None:
            print(f"\n失败: 无法读取 {video_path}")
            fail_count += 1
            continue

        if not cv2.imwrite(str(output_path), frame):
            print(f"\n失败: 无法保存 {output_path}")
            fail_count += 1
            continue

        success_count += 1

    print("\n" + "=" * 50)
    print("处理完成")
    print(f"总视频数: {len(videos)}")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
