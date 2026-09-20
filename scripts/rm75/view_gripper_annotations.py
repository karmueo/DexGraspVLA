"""分页查看首帧图片及同名 Labelme JSON 中的目标多边形。"""

import argparse
import json
import math
import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class AnnotationItem:
    """记录一张图片及其同名标注；缺失的一侧用 None 表示。"""

    label: str
    image_path: Path | None
    json_path: Path | None


@dataclass
class TileResult:
    """保存单格缩略图，以及需要显示的错误和尺寸提示。"""

    image: Image.Image
    errors: list[str]
    notices: list[str]


def natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """将路径中的数字按数值排序，使 episode_2 排在 episode_10 前。"""
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r"(\d+)", value))


def discover_items(root: Path) -> list[AnnotationItem]:
    """递归收集图片与同名 JSON，同时保留只有 JSON 的条目。"""
    if not root.is_dir():
        raise ValueError(f"根目录不存在或不是目录: {root}")

    # 以完整路径去掉后缀作为配对键，避免不同 episode 的同名文件混淆。
    json_files = {path.with_suffix(""): path for path in root.rglob("*")
                  if path.is_file() and path.suffix.lower() == ".json"}
    image_files = [path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    items = []
    matched_json = set()
    for image_path in image_files:
        json_path = json_files.get(image_path.with_suffix(""))
        if json_path is not None:
            matched_json.add(json_path)
        items.append(AnnotationItem(image_path.relative_to(root).as_posix(), image_path, json_path))
    for json_path in json_files.values():
        if json_path not in matched_json:
            items.append(AnnotationItem(json_path.relative_to(root).as_posix(), None, json_path))
    return sorted(items, key=lambda item: natural_key(item.label))


def clamp_page(current: int, delta: int, total: int, page_size: int) -> int:
    """限制翻页结果在第一页和最后一页之间。"""
    if page_size < 1:
        raise ValueError("每页图片数必须大于 0")
    last_page = max(0, (total - 1) // page_size)
    return min(max(current + delta, 0), last_page)


def read_polygons(json_path: Path, image_size: tuple[int, int]) -> tuple[list[list[tuple[float, float]]], tuple[float, float], list[str], list[str]]:
    """读取非 plate 目标多边形，返回标注尺寸及可供浏览的异常信息。"""
    with json_path.open(encoding="utf-8") as stream:
        annotation = json.load(stream)
    if not isinstance(annotation, dict):
        raise ValueError("JSON 顶层应为对象")

    errors = []
    notices = []
    annotated_width = annotation.get("imageWidth", image_size[0])
    annotated_height = annotation.get("imageHeight", image_size[1])
    if (not isinstance(annotated_width, (int, float)) or isinstance(annotated_width, bool)
            or not math.isfinite(annotated_width) or annotated_width <= 0
            or not isinstance(annotated_height, (int, float)) or isinstance(annotated_height, bool)
            or not math.isfinite(annotated_height) or annotated_height <= 0):
        errors.append("标注图片尺寸无效")
        annotated_width, annotated_height = image_size
    elif (annotated_width, annotated_height) != image_size:
        notices.append(f"尺寸不同：标注 {annotated_width:g}×{annotated_height:g}，图片 {image_size[0]}×{image_size[1]}")

    shapes = annotation.get("shapes")
    if not isinstance(shapes, list):
        return [], (annotated_width, annotated_height), errors + ["shapes 缺失或格式错误"], notices

    # 与 mask 生成脚本保持相同的 plate 标签过滤规则。
    targets = [shape for shape in shapes if isinstance(shape, dict)
               and isinstance(shape.get("label", ""), str)
               and "plate" not in shape.get("label", "")]
    if not targets:
        errors.append("目标多边形数量为 0，至少应有 1 个")
    elif len(targets) > 1:
        notices.append(f"检测到 {len(targets)} 个候选目标，生成 mask 时需选择一个")
    if any(not isinstance(shape, dict) or not isinstance(shape.get("label", ""), str)
           for shape in shapes):
        errors.append("形状或标签格式错误")

    polygons = []
    for shape in targets:
        points = shape.get("points")
        if shape.get("shape_type") != "polygon" or not isinstance(points, list) or len(points) < 3:
            errors.append("目标形状不是有效多边形")
            continue
        if any(not isinstance(point, (list, tuple)) or len(point) != 2
               or any(not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool)
                      or not math.isfinite(coordinate) for coordinate in point)
               for point in points):
            errors.append("多边形坐标格式错误")
            continue
        polygons.append([(float(x), float(y)) for x, y in points])
    return polygons, (annotated_width, annotated_height), errors, notices


def render_tile(item: AnnotationItem, size: tuple[int, int]) -> TileResult:
    """只读取当前页的图片与标注，并生成带目标轮廓的缩略图。"""
    preview = Image.new("RGB", size, "#e8e8e8")
    errors = []
    notices = []
    if item.image_path is None:
        return TileResult(preview, ["缺少同名图片"], notices)
    try:
        with Image.open(item.image_path) as image:
            image.load()
            image_size = image.size
            thumbnail = image.convert("RGB")
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    except (OSError, ValueError) as exc:
        return TileResult(preview, [f"无法读取图片：{exc}"], notices)

    # 把原图等比缩放后居中放入固定大小的缩略图格子。
    offset = ((size[0] - thumbnail.width) // 2, (size[1] - thumbnail.height) // 2)
    preview.paste(thumbnail, offset)
    if item.json_path is None:
        return TileResult(preview, ["缺少同名 JSON"], notices)
    try:
        polygons, annotated_size, errors, notices = read_polygons(item.json_path, image_size)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return TileResult(preview, [f"无法读取标注：{exc}"], notices)

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for polygon in polygons:
        scaled = [(offset[0] + x * thumbnail.width / annotated_size[0],
                   offset[1] + y * thumbnail.height / annotated_size[1])
                  for x, y in polygon]
        draw.polygon(scaled, fill=(50, 220, 80, 65))
        draw.line(scaled + [scaled[0]], fill=(0, 0, 0, 255), width=4, joint="curve")
        draw.line(scaled + [scaled[0]], fill=(60, 255, 70, 255), width=2, joint="curve")
    preview = Image.alpha_composite(preview.convert("RGBA"), overlay).convert("RGB")
    return TileResult(preview, errors, notices)


class AnnotationViewer:
    """在单个 Tk 窗口中按网格分页显示标注检查结果。"""

    def __init__(self, items: list[AnnotationItem], rows: int, cols: int) -> None:
        """创建窗口、计算屏幕内的格子大小并绑定方向键。"""
        self.items = items
        self.rows = rows
        self.cols = cols
        self.page_size = rows * cols
        self.page = 0
        self.root = tk.Tk()
        self.root.title("首帧目标多边形检查")
        self.tile_size = (max(100, min(400, (self.root.winfo_screenwidth() - 80) // cols - 12)),
                          max(80, min(300, (self.root.winfo_screenheight() - 180) // rows - 45)))
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(padx=8, pady=8)
        self.footer = tk.Label(self.root)
        self.footer.pack(pady=(0, 8))
        self.photos: list[ImageTk.PhotoImage] = []
        for key in ("<Right>", "<Down>"):
            self.root.bind(key, lambda _event: self.change_page(1))
        for key in ("<Left>", "<Up>"):
            self.root.bind(key, lambda _event: self.change_page(-1))
        self.root.bind("<KeyPress-q>", lambda _event: self.root.destroy())
        self.show_page()
        self.root.focus_set()

    def change_page(self, delta: int) -> None:
        """响应方向键，只在页码变化时重新读取并绘制当前页。"""
        next_page = clamp_page(self.page, delta, len(self.items), self.page_size)
        if next_page != self.page:
            self.page = next_page
            self.show_page()

    def show_page(self) -> None:
        """重建本页格子，释放上一页图片以控制内存占用。"""
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.photos.clear()
        first = self.page * self.page_size
        current_items = self.items[first:first + self.page_size]
        for index, item in enumerate(current_items):
            result = render_tile(item, self.tile_size)
            photo = ImageTk.PhotoImage(result.image, master=self.root)
            self.photos.append(photo)
            frame = tk.Frame(self.grid_frame, borderwidth=1, relief="solid")
            frame.grid(row=index // self.cols, column=index % self.cols, padx=3, pady=3)
            tk.Label(frame, text=item.label, anchor="w", width=max(15, self.tile_size[0] // 8)).pack(fill="x")
            tk.Label(frame, image=photo).pack()
            if result.errors:
                color = "#b00020"
            elif result.notices:
                color = "#9a5b00"
            else:
                color = "#176b2d"
            status = "；".join(result.errors + result.notices) or "标注正常"
            tk.Label(frame, text=status, fg=color, anchor="w", wraplength=self.tile_size[0],
                     height=2).pack(fill="x")
        last_page = (len(self.items) - 1) // self.page_size + 1
        self.footer.config(text=f"第 {self.page + 1}/{last_page} 页 · {first + 1}–{first + len(current_items)}/{len(self.items)} 张"
                                "    ←/↑ 上一页    →/↓ 下一页    q 退出")

    def run(self) -> None:
        """启动交互式窗口事件循环。"""
        self.root.mainloop()


def positive_int(value: str) -> int:
    """校验网格行列参数为正整数。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def main() -> None:
    """解析命令行参数并启动首帧标注浏览窗口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="首帧图片和同名 JSON 的根目录")
    parser.add_argument("--rows", type=positive_int, default=4, help="每页行数，默认 4")
    parser.add_argument("--cols", type=positive_int, default=4, help="每页列数，默认 4")
    args = parser.parse_args()
    try:
        items = discover_items(args.src)
    except ValueError as exc:
        parser.error(str(exc))
    if not items:
        parser.error(f"未找到图片或 JSON：{args.src}")
    try:
        AnnotationViewer(items, args.rows, args.cols).run()
    except tk.TclError as exc:
        parser.exit(1, f"无法打开图形窗口，请在有桌面显示会话的环境运行：{exc}\n")


if __name__ == "__main__":
    main()
