"""解析 gripper 多边形标注，并在多目标时提供首帧交互选择。"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class TargetPolygon:
    """保存候选目标的标签及其在标注图片坐标系中的多边形。"""

    label: str
    points: tuple[tuple[float, float], ...]


class TargetSelectionCancelled(RuntimeError):
    """用户取消了多目标选择。"""


def read_target_polygons(
    json_path: Path, frame_shape: tuple[int, ...]
) -> tuple[list[TargetPolygon], tuple[int, int]]:
    """读取并校验非 plate 候选目标，返回多边形及标注图片尺寸。"""
    with json_path.open(encoding="utf-8") as stream:
        annotation = json.load(stream)
    if not isinstance(annotation, dict):
        raise ValueError(f"JSON 顶层应为对象: {json_path}")

    height, width = frame_shape[:2]
    try:
        annotated_height = int(annotation.get("imageHeight", height))
        annotated_width = int(annotation.get("imageWidth", width))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"标注图片尺寸无效: {json_path}") from exc
    if annotated_height <= 0 or annotated_width <= 0:
        raise ValueError(f"标注图片尺寸无效: {json_path}")

    shapes = annotation.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError(f"shapes 缺失或格式错误: {json_path}")
    targets = []
    for shape in shapes:
        if not isinstance(shape, dict) or not isinstance(shape.get("label", ""), str):
            raise ValueError(f"形状或标签格式错误: {json_path}")
        label = shape.get("label", "")
        if "plate" in label:
            continue
        points = shape.get("points")
        if (
            not isinstance(points, list)
            or len(points) < 3
            or any(
                not isinstance(point, (list, tuple))
                or len(point) != 2
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for value in point
                )
                for point in points
            )
        ):
            raise ValueError(f"目标不是有效多边形: {json_path}")
        targets.append(TargetPolygon(label, tuple((float(x), float(y)) for x, y in points)))
    if not targets:
        raise ValueError(f"需要至少一个有效的目标多边形: {json_path}")
    return targets, (annotated_width, annotated_height)


def load_init_mask(
    json_path: Path, frame_shape: tuple[int, ...], target_index: int | None = None
) -> np.ndarray:
    """读取所选非 plate 多边形并生成 0/1 mask。"""
    targets, (annotated_width, annotated_height) = read_target_polygons(json_path, frame_shape)
    if target_index is None:
        if len(targets) != 1:
            raise ValueError(f"存在 {len(targets)} 个目标，需要先选择一个: {json_path}")
        target_index = 0
    if (
        isinstance(target_index, bool)
        or not isinstance(target_index, int)
        or not 0 <= target_index < len(targets)
    ):
        raise ValueError(f"目标索引超出范围: {target_index}, 共 {len(targets)} 个: {json_path}")

    polygon = np.asarray(targets[target_index].points, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros((annotated_height, annotated_width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    height, width = frame_shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if not np.any(mask):
        raise ValueError(f"目标多边形为空: {json_path}")
    return mask


def find_target_at_point(
    targets: list[TargetPolygon], point: tuple[float, float]
) -> int | None:
    """返回覆盖点击位置且面积最小的目标索引，未命中时返回 None。"""
    matches = []
    for index, target in enumerate(targets):
        polygon = np.asarray(target.points, dtype=np.float32)
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            matches.append((abs(cv2.contourArea(polygon)), index))
    return min(matches)[1] if matches else None


def target_index_from_key(character: str, keysym: str, target_count: int) -> int | None:
    """将主键盘或数字小键盘的 1–9 转为有效目标索引。"""
    digit = character if len(character) == 1 and character in "123456789" else ""
    keypad_digit = keysym[3:] if keysym.startswith("KP_") else ""
    if not digit and len(keypad_digit) == 1 and keypad_digit in "123456789":
        digit = keypad_digit
    index = int(digit) - 1 if digit else -1
    return index if 0 <= index < target_count else None


def select_target(
    first_frame: np.ndarray,
    targets: list[TargetPolygon],
    annotated_size: tuple[int, int],
    title: str,
) -> int:
    """弹出首帧多边形选择窗口并返回所选目标索引。"""
    try:
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageTk
    except ImportError as exc:
        raise RuntimeError(f"缺少多目标选择窗口依赖: {title}") from exc
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            f"无法打开多目标选择窗口，请在有桌面显示会话的环境运行: {title}"
        ) from exc

    root.title(f"选择跟踪目标 - {title}")
    frame_height, frame_width = first_frame.shape[:2]
    scale = min(
        1.0,
        max(320, root.winfo_screenwidth() - 380) / frame_width,
        max(240, root.winfo_screenheight() - 160) / frame_height,
    )
    display_size = (max(1, round(frame_width * scale)), max(1, round(frame_height * scale)))
    rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    base_image = Image.fromarray(rgb).resize(display_size, Image.Resampling.LANCZOS)
    colors = [
        (255, 70, 70),
        (65, 180, 255),
        (80, 210, 110),
        (255, 180, 55),
        (190, 90, 255),
        (40, 210, 205),
        (255, 105, 180),
        (180, 180, 60),
    ]
    selected: int | None = None
    result: int | None = None
    photo = None

    body = tk.Frame(root)
    body.pack(padx=10, pady=10)
    canvas = tk.Canvas(
        body, width=display_size[0], height=display_size[1], highlightthickness=0
    )
    canvas.grid(row=0, column=0)
    image_item = canvas.create_image(0, 0, anchor="nw")
    side = tk.Frame(body)
    side.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
    tk.Label(
        side,
        text="请选择要跟踪的目标\n数字键 1–9 选择，Enter 确认",
        anchor="w",
        justify="left",
    ).pack(fill="x", pady=(0, 6))
    target_list = tk.Listbox(
        side, width=32, height=min(max(len(targets), 4), 18), exportselection=False
    )
    target_list.pack(fill="both", expand=True)
    for index, target in enumerate(targets):
        target_list.insert("end", f"{index + 1}. {target.label or '(无标签)'}")
    confirm_button = tk.Button(side, text="确认", state=tk.DISABLED)
    confirm_button.pack(fill="x", pady=(10, 4))
    cancel_button = tk.Button(side, text="取消")
    cancel_button.pack(fill="x")

    def render() -> None:
        """根据当前选择重绘覆盖层。"""
        nonlocal photo
        overlay = Image.new("RGBA", display_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        annotated_width, annotated_height = annotated_size
        for index, target in enumerate(targets):
            color = colors[index % len(colors)]
            points = [
                (
                    x * display_size[0] / annotated_width,
                    y * display_size[1] / annotated_height,
                )
                for x, y in target.points
            ]
            draw.polygon(points, fill=(*color, 105 if index == selected else 55))
            draw.line(
                points + [points[0]],
                fill=(*color, 255),
                width=5 if index == selected else 3,
                joint="curve",
            )
            x, y = points[0]
            draw.text(
                (x + 4, y + 4),
                str(index + 1),
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
            )
        preview = Image.alpha_composite(base_image.convert("RGBA"), overlay).convert("RGB")
        photo = ImageTk.PhotoImage(preview, master=root)
        canvas.itemconfigure(image_item, image=photo)

    def set_selected(index: int) -> None:
        """同步图片高亮、列表选项和确认按钮。"""
        nonlocal selected
        if selected == index:
            return
        selected = index
        target_list.selection_clear(0, "end")
        target_list.selection_set(index)
        target_list.see(index)
        confirm_button.config(state=tk.NORMAL)
        render()

    def on_canvas_click(event: object) -> None:
        annotated_width, annotated_height = annotated_size
        point = (
            event.x * annotated_width / display_size[0],
            event.y * annotated_height / display_size[1],
        )
        index = find_target_at_point(targets, point)
        if index is not None:
            set_selected(index)

    def on_list_select(_event: object) -> None:
        indices = target_list.curselection()
        if indices:
            set_selected(indices[0])

    def on_key_press(event: object) -> None:
        index = target_index_from_key(event.char, event.keysym, len(targets))
        if index is not None:
            set_selected(index)

    def confirm() -> None:
        nonlocal result
        if selected is not None:
            result = selected
            root.destroy()

    def cancel() -> None:
        root.destroy()

    canvas.bind("<Button-1>", on_canvas_click)
    target_list.bind("<<ListboxSelect>>", on_list_select)
    confirm_button.config(command=confirm)
    cancel_button.config(command=cancel)
    root.bind("<KeyPress>", on_key_press)
    root.bind("<Return>", lambda _event: confirm())
    root.bind("<Escape>", lambda _event: cancel())
    root.protocol("WM_DELETE_WINDOW", cancel)
    render()
    root.focus_force()
    root.mainloop()
    if result is None:
        raise TargetSelectionCancelled(f"已取消多目标选择: {title}")
    return result


def resolve_video_target(
    video_path: Path,
    selector: Callable[
        [np.ndarray, list[TargetPolygon], tuple[int, int], str], int
    ] = select_target,
) -> int:
    """读取视频首帧；单目标自动选择，多目标交由图形窗口选择。"""
    annotation_path = video_path.with_name("gripper.json")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"缺少首帧标注: {annotation_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        valid, first_frame = capture.read()
        if not valid:
            raise ValueError(f"视频没有帧: {video_path}")
    finally:
        capture.release()

    targets, annotated_size = read_target_polygons(annotation_path, first_frame.shape)
    if len(targets) == 1:
        return 0
    target_index = selector(first_frame, targets, annotated_size, str(video_path.parent))
    if not 0 <= target_index < len(targets):
        raise ValueError(f"选择器返回无效目标索引: {target_index}, 共 {len(targets)} 个")
    return target_index
