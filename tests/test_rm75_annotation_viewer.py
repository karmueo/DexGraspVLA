"""验证首帧标注浏览器的文件发现、绘制和分页行为。"""

import json
from pathlib import Path

from PIL import Image

from scripts.rm75.view_gripper_annotations import AnnotationItem, clamp_page, discover_items, render_tile


def write_image(path: Path) -> None:
    """创建用于检查多边形覆盖效果的纯白测试图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), "white").save(path)


def write_annotation(path: Path, shapes: list[dict], image_size: tuple[int, int] = (20, 20)) -> None:
    """创建最小的 Labelme 格式测试标注。"""
    path.write_text(json.dumps({"imageWidth": image_size[0], "imageHeight": image_size[1],
                                "shapes": shapes}), encoding="utf-8")


def polygon(label: str = "target") -> dict:
    """返回覆盖图片中部的目标多边形。"""
    return {"label": label, "shape_type": "polygon",
            "points": [[5, 5], [15, 5], [15, 15], [5, 15]]}


def test_discovery_pairs_and_natural_order(tmp_path: Path) -> None:
    """递归配对同名文件，保留缺失项并按 episode 数值排序。"""
    for name in ("episode_10", "episode_2", "episode_1"):
        episode = tmp_path / "nested" / name
        write_image(episode / "gripper.jpg")
    write_annotation(tmp_path / "nested/episode_1/gripper.json", [polygon()])
    only_json = tmp_path / "nested/episode_3/gripper.json"
    only_json.parent.mkdir(parents=True)
    write_annotation(only_json, [polygon()])

    items = discover_items(tmp_path)
    assert [item.label for item in items] == [
        "nested/episode_1/gripper.jpg", "nested/episode_2/gripper.jpg",
        "nested/episode_3/gripper.json", "nested/episode_10/gripper.jpg",
    ]
    assert items[0].json_path is not None
    assert items[1].json_path is None
    assert items[2].image_path is None


def test_overlay_and_annotation_errors(tmp_path: Path) -> None:
    """仅绘制非 plate 多边形，并将多目标作为可选择提示。"""
    image_path = tmp_path / "gripper.png"
    json_path = tmp_path / "gripper.json"
    write_image(image_path)
    item = AnnotationItem("gripper.png", image_path, json_path)

    plate = {"label": "plate", "shape_type": "polygon",
             "points": [[0, 0], [19, 0], [19, 19], [0, 19]]}
    write_annotation(json_path, [plate, polygon()])
    result = render_tile(item, (40, 40))
    assert not result.errors
    assert result.image.getpixel((20, 20)) != (255, 255, 255)
    assert result.image.getpixel((12, 12)) == (255, 255, 255)

    write_annotation(json_path, [])
    assert "数量为 0" in render_tile(item, (40, 40)).errors[0]
    other = {"label": "other", "shape_type": "polygon",
             "points": [[1, 1], [4, 1], [4, 4], [1, 4]]}
    write_annotation(json_path, [polygon(), other])
    result = render_tile(item, (40, 40))
    assert not result.errors and "2 个候选目标" in result.notices[0]
    assert result.image.getpixel((5, 5)) != (255, 255, 255)
    json_path.write_text("{invalid", encoding="utf-8")
    assert "无法读取标注" in render_tile(item, (40, 40)).errors[0]
    json_path.unlink()
    assert "缺少同名 JSON" in render_tile(AnnotationItem(item.label, image_path, None), (40, 40)).errors
    assert "缺少同名图片" in render_tile(AnnotationItem("gripper.json", None, json_path), (40, 40)).errors


def test_scaled_coordinates_and_page_boundaries(tmp_path: Path) -> None:
    """标注尺寸变化时缩放坐标，翻页在首尾停留。"""
    image_path = tmp_path / "gripper.png"
    json_path = tmp_path / "gripper.json"
    write_image(image_path)
    write_annotation(json_path, [{"label": "target", "shape_type": "polygon",
                                  "points": [[10, 10], [30, 10], [30, 30], [10, 30]]}], (40, 40))
    result = render_tile(AnnotationItem("gripper.png", image_path, json_path), (40, 40))
    assert result.notices and not result.errors
    assert result.image.getpixel((20, 20)) != (255, 255, 255)
    assert result.image.getpixel((12, 12)) == (255, 255, 255)
    assert clamp_page(0, -1, 50, 16) == 0
    assert clamp_page(0, 1, 50, 16) == 1
    assert clamp_page(3, 1, 50, 16) == 3
