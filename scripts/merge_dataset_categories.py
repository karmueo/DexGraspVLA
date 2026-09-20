#!/usr/bin/env python3
"""将多个数据集类别中的 episode 顺序合并到首个类别。

作为脚本运行时，使用 ``python scripts/merge_dataset_categories.py CATEGORY [CATEGORY ...]``：
首个 ``CATEGORY`` 是目标类别，其余类别中的 ``episode_<非负整数>`` 目录会复制到目标
类别，并按目标中现有的最大编号继续编号。``--root`` 指定包含各类别目录的数据集根目录，
默认为 ``dataset/h5dy_data/data``。脚本逐条输出复制记录和汇总信息；参数或数据校验失败时
向标准错误输出原因并以状态码 2 退出，复制失败时以状态码 1 退出。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Sequence


# 默认数据集根目录；其直接子目录应为数据集类别。
DEFAULT_ROOT = Path("dataset/h5dy_data/data")
# 合法 episode 目录名：编号为不带前导零的非负十进制整数（零本身除外）。
EPISODE_PATTERN = re.compile(r"episode_(0|[1-9]\d*)\Z")


@dataclass(frozen=True)
class Episode:
    """已校验的 episode 目录及其用于排序和重新编号的数值索引。"""

    path: Path
    index: int


@dataclass(frozen=True)
class CopyOperation:
    """一项经预检后可执行的 episode 复制操作。"""

    source_category: str
    source: Path
    destination: Path
    destination_index: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析类别列表和可选的数据集根目录参数。

    ``argv`` 为 ``None`` 时读取命令行参数；返回的 ``categories`` 保持用户指定的顺序，
    以便首项作为合并目标。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "categories",
        nargs="+",
        metavar="CATEGORY",
        help=(
            "categories to merge; the first is the destination and all "
            "remaining categories are copied into it"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"dataset root containing category directories (default: {DEFAULT_ROOT})",
    )
    return parser.parse_args(argv)


def _validate_category_name(name: str) -> None:
    """拒绝空名称、路径遍历及非根目录直接子项的类别名称。

    不返回结果；名称不满足约束时抛出 ``ValueError``。
    """
    path = Path(name)
    if name in {"", ".", ".."} or path.is_absolute() or path.name != name:
        raise ValueError(f"category must be a direct child name: {name!r}")


def validate_categories(root: Path, names: Sequence[str]) -> list[Path]:
    """解析并校验有序的目标/源类别目录列表。

    ``names`` 至少包含一个目标和一个源类别，首项对应目标目录。返回位于 ``root`` 下的
    已解析目录路径；根目录、类别目录或名称约束不满足时抛出 ``ValueError``。
    """
    if len(names) < 2:
        raise ValueError(
            "provide a destination category and at least one source category"
        )
    if len(set(names)) != len(names):
        raise ValueError("category names must not be repeated")

    # 使用解析后的根目录，避免符号链接绕过“直接子目录”约束。
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {resolved_root}")

    categories: list[Path] = []
    for name in names:
        _validate_category_name(name)
        category = resolved_root / name
        if not category.is_dir():
            raise ValueError(f"category is not a directory: {category}")
        resolved_category = category.resolve()
        if resolved_category.parent != resolved_root:
            raise ValueError(f"category must remain inside the dataset root: {name!r}")
        categories.append(resolved_category)
    return categories


def find_episodes(category: Path) -> list[Episode]:
    """查找类别的直接子 episode 目录，并按数值索引升序返回。

    任何以 ``episode_`` 开头但名称不合法、不是目录或编号重复的条目都会导致
    ``ValueError``，以避免生成含歧义的合并计划。
    """
    episodes: list[Episode] = []
    seen_indices: set[int] = set()

    for entry in category.iterdir():
        if not entry.name.startswith("episode_"):
            continue
        match = EPISODE_PATTERN.fullmatch(entry.name)
        if match is None:
            raise ValueError(f"invalid episode name: {entry}")
        if not entry.is_dir():
            raise ValueError(f"episode is not a directory: {entry}")

        index = int(match.group(1))
        if index in seen_indices:
            raise ValueError(f"duplicate episode index {index} in {category}")
        seen_indices.add(index)
        episodes.append(Episode(path=entry, index=index))

    episodes.sort(key=lambda episode: episode.index)
    return episodes


def plan_merge(categories: Sequence[Path]) -> list[CopyOperation]:
    """校验全部 episode，并构建完整且有序的复制计划。

    首个类别是目标；后续类别按给定顺序处理，每个源类别内按 episode 编号处理。目标编号
    从其现有最大编号加一开始。遇到空源类别、非法 episode 或目标路径已存在时抛出
    ``ValueError``，且尚未执行任何复制。
    """
    destination_category = categories[0]
    destination_episodes = find_episodes(destination_category)
    # 目标目录为空时从 0 开始，否则紧接现有最大编号。
    next_index = max(
        (episode.index for episode in destination_episodes), default=-1
    ) + 1

    operations: list[CopyOperation] = []
    for source_category in categories[1:]:
        source_episodes = find_episodes(source_category)
        if not source_episodes:
            raise ValueError(f"source category contains no episodes: {source_category}")
        for source_episode in source_episodes:
            destination = destination_category / f"episode_{next_index}"
            if destination.exists():
                raise ValueError(f"destination already exists: {destination}")
            operations.append(
                CopyOperation(
                    source_category=source_category.name,
                    source=source_episode.path,
                    destination=destination,
                    destination_index=next_index,
                )
            )
            next_index += 1
    return operations


def create_temporary_directory(destination: Path) -> Path:
    """为 ``destination`` 创建并返回本次复制独占的同级临时目录。"""
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.merge_tmp.",
            dir=destination.parent,
        )
    )


def copy_episode(operation: CopyOperation) -> None:
    """经同级临时目录复制一个 episode，并在完成后重命名发布。

    若复制过程中发生异常，会尝试删除临时目录后重新抛出原异常；目标目录在开始或发布前
    已存在时抛出 ``FileExistsError``。
    """
    temporary = create_temporary_directory(operation.destination)
    try:
        if operation.destination.exists():
            raise FileExistsError(
                f"destination already exists: {operation.destination}"
            )
        shutil.copytree(operation.source, temporary, dirs_exist_ok=True)
        if operation.destination.exists():
            raise FileExistsError(
                f"destination appeared during copy: {operation.destination}"
            )
        temporary.rename(operation.destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """执行合并并返回进程状态码。

    成功时返回 0；参数或合并计划校验失败时返回 2；任一复制操作失败时返回 1。每次成功
    复制会输出源、目标及最终编号范围。
    """
    args = parse_args(argv)
    try:
        categories = validate_categories(args.root, args.categories)
        operations = plan_merge(categories)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    destination_category = categories[0]
    # 已成功发布到目标类别的 episode 数量。
    copied = 0
    for operation in operations:
        try:
            copy_episode(operation)
        except OSError as error:
            print(
                f"Error: failed to copy {operation.source} to "
                f"{operation.destination}: {error}",
                file=sys.stderr,
            )
            return 1
        copied += 1
        print(
            f"COPY {operation.source_category}/{operation.source.name} -> "
            f"{destination_category.name}/{operation.destination.name}"
        )

    first_index = operations[0].destination_index
    last_index = operations[-1].destination_index
    print(
        f"Summary: copied={copied}, destination={destination_category}, "
        f"new_range={first_index}..{last_index}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
