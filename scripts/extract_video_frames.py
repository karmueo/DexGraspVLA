#!/usr/bin/env python3
"""按固定时间间隔从指定数据集类别的 MP4 视频中提取 JPEG 帧。

可作为脚本执行：
``python scripts/extract_video_frames.py --categories CATEGORY ... --output OUTPUT
--interval-seconds SECONDS``。``--root`` 指定类别目录所在的数据集根目录，默认值为
``dataset/h5dy_data/data``；``--overwrite`` 允许覆盖已有图像。程序将处理结果写入
标准输出、错误写入标准错误，并以非零状态码表示参数错误或存在处理失败。

也可导入 ``validate_categories``、``find_mp4_files`` 与 ``extract_video``，分别用于
校验类别、枚举视频和处理单个视频；最后一个入口返回 ``ExtractionResult``。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Sequence

import cv2


# 默认数据集根目录，类别目录必须是其直接子目录。
DEFAULT_ROOT = Path("dataset/h5dy_data/data")
# 传给 OpenCV JPEG 编码器的质量值。
JPEG_QUALITY = 95


@dataclass(frozen=True)
class ExtractionResult:
    """单个视频成功处理后的源视频路径与已写入图像数量。"""

    video: Path
    images: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 待解析的参数序列；为 ``None`` 时读取实际命令行参数。

    Returns:
        包含数据集根目录、类别名、输出根目录、采样间隔与覆盖选项的命名空间。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Recursively extract JPEG frames from MP4 videos in one or more "
            "dataset categories."
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
        "--interval-seconds",
        type=float,
        required=True,
        metavar="SECONDS",
        help="sampling interval in seconds, for example 0.2, 0.5, 1, 2, or 5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite JPEG files that already exist",
    )
    return parser.parse_args(argv)


def validate_categories(root: Path, category_names: Sequence[str]) -> list[Path]:
    """解析并校验数据集根目录下的直接子类别目录。

    Args:
        root: 数据集根目录。
        category_names: 待处理的类别目录名；每项必须是直接子目录名且不可重复。

    Returns:
        解析后的类别目录绝对路径列表，顺序与输入一致。

    Raises:
        ValueError: 根目录或类别目录不存在，类别名无效、重复或越出根目录时抛出。
    """
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")

    # 保留调用方给定的类别顺序。
    categories: list[Path] = []
    # 防止同一类别被重复处理。
    seen: set[str] = set()
    for name in category_names:
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"category must be a direct child name: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate category: {name}")
        seen.add(name)

        category = (root / name).resolve()
        if category.parent != root:
            raise ValueError(f"category is outside the dataset root: {name}")
        if not category.is_dir():
            raise ValueError(f"category is not a directory: {name}")
        categories.append(category)
    return categories


def find_mp4_files(category: Path) -> list[Path]:
    """按相对路径稳定排序，返回类别目录下递归发现的 MP4 文件。

    Args:
        category: 搜索 MP4 文件的类别目录。

    Returns:
        按相对 POSIX 路径升序排列的文件路径列表。
    """
    return sorted(
        (
            path
            for path in category.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mp4"
        ),
        key=lambda path: path.relative_to(category).as_posix(),
    )


def _output_stem(video: Path, category: Path) -> str:
    """将类别内视频的相对路径编码为无歧义的文件名前缀。"""
    relative = video.relative_to(category)
    return "".join(f"{len(part)}_{part}" for part in relative.parts)


def _frame_filename(stem: str, frame_index: int, fps: float) -> str:
    """生成包含帧索引与按帧率换算时间戳的 JPEG 文件名。"""
    timestamp_ms = int(round(frame_index * 1000.0 / fps))
    return f"{stem}__f{frame_index:08d}__t{timestamp_ms:012d}ms.jpg"


def extract_video(
    video: Path,
    category: Path,
    output_directory: Path,
    interval_seconds: float,
    *,
    overwrite: bool = False,
) -> ExtractionResult:
    """顺序解码一个视频，并保存最接近各采样时刻的帧。

    Args:
        video: 待解码的源 MP4 文件。
        category: 源视频所属的类别目录，用于生成输出文件名前缀。
        output_directory: 当前类别图像的输出目录。
        interval_seconds: 相邻采样目标时刻的间隔（秒），必须为有限正数。
        overwrite: 是否允许覆盖已存在的目标图像。

    Returns:
        记录源视频路径与已写入图像数量的处理结果。

    Raises:
        FileExistsError: 目标图像已存在且未启用覆盖。
        ValueError: 采样间隔无效、视频无法读取、帧率无效或没有可解码帧。
        OSError: JPEG 图像写入失败。
    """
    if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be a finite positive number")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"video is not readable: {video}")

    # 视频报告的每秒帧数，用于把时间采样间隔换算为帧索引。
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise ValueError(f"video has invalid FPS: {video}")

    output_directory.mkdir(parents=True, exist_ok=True)
    # 对每个路径分量加入长度前缀，避免目录分隔符与文件名字符串冲突。
    stem = _output_stem(video, category)
    # 间隔不大于一帧时，视频中的每个可读帧都应保存。
    save_every_frame = interval_seconds * fps <= 1.0
    # target_number 是待采样时刻的序号，target_frame 是其对应的最近帧索引。
    target_number = 0
    target_frame = 0
    # frame_index 从零开始记录已读取帧的位置；image_count 统计成功写入数。
    frame_index = 0
    image_count = 0

    try:
        while True:
            readable, frame = capture.read()
            if not readable:
                break

            should_save = save_every_frame or frame_index == target_frame
            if should_save:
                destination = output_directory / _frame_filename(
                    stem, frame_index, fps
                )
                if destination.exists() and not overwrite:
                    raise FileExistsError(f"output image already exists: {destination}")
                written = cv2.imwrite(
                    str(destination),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
                if not written:
                    raise OSError(f"failed to write image: {destination}")
                image_count += 1

                if not save_every_frame:
                    target_number += 1
                    target_frame = int(
                        math.floor(target_number * interval_seconds * fps + 0.5)
                    )
            frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        raise ValueError(f"video contains no readable frames: {video}")
    return ExtractionResult(video=video, images=image_count)


def main(argv: Sequence[str] | None = None) -> int:
    """批量提取指定类别视频的定间隔帧，并返回进程退出状态。"""
    args = parse_args(argv)
    if not math.isfinite(args.interval_seconds) or args.interval_seconds <= 0.0:
        print(
            "Error: --interval-seconds must be a finite positive number",
            file=sys.stderr,
        )
        return 2

    try:
        categories = validate_categories(args.root, args.categories)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    # 所有类别图像目录的绝对输出根路径。
    output_root = args.output.resolve()
    # 分别统计成功处理的视频数、失败项数和写入的图像总数。
    successful_videos = 0
    failed_videos = 0
    total_images = 0

    for category in categories:
        videos = find_mp4_files(category)
        if not videos:
            print(f"ERROR {category.name}: no MP4 videos found", file=sys.stderr)
            failed_videos += 1
            continue

        category_output = output_root / category.name
        for video in videos:
            relative_video = video.relative_to(category)
            try:
                result = extract_video(
                    video,
                    category,
                    category_output,
                    args.interval_seconds,
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
            total_images += result.images
            print(f"OK {category.name}/{relative_video}: images={result.images}")

    print(
        "Summary: "
        f"success={successful_videos}, failed={failed_videos}, images={total_images}"
    )
    return 1 if failed_videos else 0


if __name__ == "__main__":
    raise SystemExit(main())
