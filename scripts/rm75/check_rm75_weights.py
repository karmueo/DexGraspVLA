#!/usr/bin/env python3
"""校验 RM75 默认 DINOv2 与 Cutie 权重的 SHA-256。"""

import argparse
import hashlib
from pathlib import Path


EXPECTED_SHA256 = {
    "dinov2": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    "cutie": "9c05402ee36d3a356fb72715d263ba7e1ea06ad3bada48c1306491792da43023",
    "vitl14": "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
}


def sha256(path: Path) -> str:
    """分块计算文件摘要，避免把权重整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """解析权重路径并校验默认权重及可选权重。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dinov2",
        type=Path,
        default=Path("weights/dinov2/dinov2_vitb14_pretrain.pth"),
        help="DINOv2 ViT-B/14 权重路径",
    )
    parser.add_argument(
        "--cutie",
        type=Path,
        default=Path("weights/tracking/cutie-base-mega.pth"),
        help="Cutie base mega 权重路径",
    )
    parser.add_argument(
        "--vitl14",
        type=Path,
        default=None,
        help="可选的 DINOv2 ViT-L/14 权重路径；提供时一并校验",
    )
    args = parser.parse_args()

    failed = False
    targets = [("dinov2", args.dinov2), ("cutie", args.cutie)]
    if args.vitl14 is not None:
        targets.append(("vitl14", args.vitl14))
    for name, path in targets:
        path = path.expanduser().resolve()
        if not path.is_file():
            print(f"[缺失] {path}")
            failed = True
            continue
        actual = sha256(path)
        expected = EXPECTED_SHA256[name]
        if actual != expected:
            print(f"[失败] {path}\n  期望: {expected}\n  实际: {actual}")
            failed = True
        else:
            print(f"[通过] {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
