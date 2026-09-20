#!/usr/bin/env python3
"""按数据集目录结构提取指定类别中每个 MP4 的首帧 JPEG 图像。

可作为脚本执行：
``python scripts/extract_first_video_frames.py --categories CATEGORY ... --output OUTPUT``。
``--root`` 指定类别目录所在的数据集根目录，默认为
``dataset/h5dy_data/data``；``--overwrite`` 允许覆盖已有图像。程序将处理结果
写入标准输出、错误写入标准错误，并以非零状态码表示存在处理失败。

也可导入 ``output_path_for_video`` 和 ``extract_first_frame``，分别计算输出路径和
处理单个视频；后者返回 ``FirstFrameResult``。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import cv2

if __package__:
    from .extract_video_frames import (
        DEFAULT_ROOT,
        JPEG_QUALITY,
        find_mp4_files,
        validate_categories,
    )
else:
    from extract_video_frames import (
        DEFAULT_ROOT,
        JPEG_QUALITY,
        find_mp4_files,
        validate_categories,
    )


@dataclass(frozen=True)
class FirstFrameResult:
    """单个视频成功处理后的源视频与生成图像路径。"""

    video: Path
    image: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 待解析的参数序列；为 ``None`` 时读取实际命令行参数。

    Returns:
        包含数据集根目录、类别名、输出根目录与覆盖选项的命名空间。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract the first frame from each MP4 in one or more dataset "
            "categories while preserving their directory structure."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"dataset root containing category directories (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        metavar="CATEGORY",
        help="one or more direct child category names under --root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="root directory for extracted category image directories",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite JPEG files that already exist",
    )
    return parser.parse_args(argv)


def output_path_for_video(video: Path, category: Path, output_root: Path) -> Path:
    """计算视频对应的 JPEG 输出路径，并保留其在类别目录内的层级。

    Args:
        video: 位于 ``category`` 内的源 MP4 文件。
        category: 源视频所属的类别目录。
        output_root: 所有提取图像的输出根目录。

    Returns:
        ``output_root/category.name`` 下、扩展名替换为 ``.jpg`` 的目标路径。
    """
    relative_video = video.relative_to(category)
    return output_root / category.name / relative_video.with_suffix(".jpg")


def extract_first_frame(
    video: Path,
    category: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> FirstFrameResult:
    """解码一个视频的第一帧，并保存为 JPEG 图像。

    Args:
        video: 待读取的源视频。
        category: 源视频所属的类别目录。
        output_root: 类别图像输出目录的共同根路径。
        overwrite: 是否允许覆盖已存在的目标图像。

    Returns:
        记录源视频和成功写入图像路径的处理结果。

    Raises:
        FileExistsError: 目标图像已存在且未启用覆盖。
        ValueError: 视频无法读取或不含可解码帧。
        OSError: JPEG 图像写入失败。
    """
    destination = output_path_for_video(video, category, output_root)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output image already exists: {destination}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"video is not readable: {video}")

    try:
        readable, frame = capture.read()
    finally:
        capture.release()
    if not readable:
        raise ValueError(f"video contains no readable frames: {video}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.imwrite(
        str(destination),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not written:
        raise OSError(f"failed to write image: {destination}")
    return FirstFrameResult(video=video, image=destination)


def main(argv: Sequence[str] | None = None) -> int:
    """批量提取指定类别视频的首帧，并返回进程退出状态。"""
    args = parse_args(argv)
    try:
        categories = validate_categories(args.root, args.categories)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    # 所有类别图像目录的绝对输出根路径。
    output_root = args.output.resolve()
    # 分别统计成功提取的源视频数和失败项数。
    successful_videos = 0
    failed_videos = 0

    for category in categories:
        videos = find_mp4_files(category)
        if not videos:
            print(f"ERROR {category.name}: no MP4 videos found", file=sys.stderr)
            failed_videos += 1
            continue

        for video in videos:
            relative_video = video.relative_to(category)
            try:
                result = extract_first_frame(
                    video,
                    category,
                    output_root,
                    overwrite=args.overwrite,
                )
            except (FileExistsError, OSError, ValueError) as error:
                print(
                    f"ERROR {category.name}/{relative_video}: {error}",
                    file=sys.stderr,
                )
                failed_videos += 1
                continue

            successful_videos += 1
            print(
                f"OK {category.name}/{relative_video}: "
                f"image={result.image.relative_to(output_root)}"
            )

    print(
        "Summary: "
        f"success={successful_videos}, failed={failed_videos}, "
        f"images={successful_videos}"
    )
    return 1 if failed_videos else 0


if __name__ == "__main__":
    raise SystemExit(main())
