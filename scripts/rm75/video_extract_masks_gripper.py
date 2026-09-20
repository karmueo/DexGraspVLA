"""根据首帧多边形标注，用 Cutie 为 RM75 夹爪视频生成逐帧目标 mask。"""

import argparse
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tqdm import tqdm


try:
    from .gripper_target_selection import (
        TargetSelectionCancelled,
        load_init_mask,
        resolve_video_target,
    )
except ImportError:
    from gripper_target_selection import (
        TargetSelectionCancelled,
        load_init_mask,
        resolve_video_target,
    )


class Matting:
    """持有单个 GPU 上的 Cutie 模型与逐视频重置的推理状态。"""

    def __init__(self, cutie_weights: Path, device: torch.device) -> None:
        """从已安装的 Cutie v1.0 源码配置加载本地模型权重。"""
        import cutie
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg
        from cutie.model.cutie import CUTIE

        config_dir = Path(next(iter(cutie.__path__))).resolve() / "config"
        with initialize_config_dir(version_base="1.3.2", config_dir=str(config_dir)):
            cfg = compose(config_name="eval_config")
        get_dataset_cfg(cfg)
        cfg.weights = str(cutie_weights)
        self.device = device
        self.cutie = CUTIE(cfg).to(device).eval()
        self.cutie.load_weights(torch.load(cutie_weights, map_location="cpu", weights_only=True))
        self.processor = InferenceCore(self.cutie, cfg=cfg)
        self.processor.max_internal_size = -1

    def start_video(self, mask: np.ndarray) -> None:
        """清除上一段视频的记忆并注册首帧目标。"""
        self.processor.clear_memory()
        self.mask = torch.from_numpy(mask).to(self.device)
        self.initialized = False

    @torch.inference_mode()
    def process_frame(self, bgr_image: np.ndarray) -> np.ndarray:
        """将 OpenCV 的 BGR 帧转为 Cutie 所需 RGB，返回 0/1 mask。"""
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb_image).to(self.device).permute(2, 0, 1).float() / 255.0
        if not self.initialized:
            probability = self.processor.step(image, self.mask, objects=[1])
            self.initialized = True
        else:
            probability = self.processor.step(image)
        return probability.argmax(dim=0).cpu().numpy().astype(np.uint8)


def process_video(
    video_path: Path,
    matting: Matting,
    overwrite: bool = False,
    target_index: int | None = None,
) -> Path:
    """生成与原视频等帧数的灰度 mask 视频，成功后替换临时文件。"""
    output_path = video_path.with_name("mask_gripper.mp4")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"mask 视频已存在: {output_path}")
    annotation_path = video_path.with_name("gripper.json")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"缺少首帧标注: {annotation_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    temporary_path = video_path.with_name("mask_gripper.part.mp4")
    writer = None
    try:
        valid, first_frame = capture.read()
        if not valid:
            raise ValueError(f"视频没有帧: {video_path}")
        height, width = first_frame.shape[:2]
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        if frame_rate <= 0:
            raise ValueError(f"视频帧率无效: {video_path}")
        matting.start_video(load_init_mask(annotation_path, first_frame.shape, target_index))
        writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height), isColor=False)
        if not writer.isOpened():
            raise RuntimeError(f"无法创建 mask 视频: {temporary_path}")
        count = 0
        frame = first_frame
        while True:
            mask = matting.process_frame(frame)
            if mask.shape != (height, width):
                raise ValueError(f"mask 尺寸不匹配: {video_path}: {mask.shape}")
            writer.write(mask * 255)
            count += 1
            valid, frame = capture.read()
            if not valid:
                break
        writer.release()
        writer = None
        check = cv2.VideoCapture(str(temporary_path))
        decoded_count = 0
        while check.read()[0]:
            decoded_count += 1
        check.release()
        if decoded_count != count:
            raise ValueError(f"mask 帧数不匹配: {video_path}: {count} vs {decoded_count}")
        temporary_path.replace(output_path)
        return output_path
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        temporary_path.unlink(missing_ok=True)


def _run_worker(task: tuple[Path, Path, int, bool, int]) -> Path:
    """在指定 GPU 上初始化 Cutie 并处理单段视频。"""
    video_path, weights, gpu_id, overwrite, target_index = task
    torch.cuda.set_device(gpu_id)
    return process_video(
        video_path,
        Matting(weights, torch.device(f"cuda:{gpu_id}")),
        overwrite,
        target_index,
    )


def main() -> None:
    """查找单目视频并生成缺失的 mask，可按 GPU 数并行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, nargs="+", required=True, help="原始 episode 根目录")
    default_weights = Path(__file__).resolve().parents[2] / "weights/tracking/cutie-base-mega.pth"
    parser.add_argument("--weights", type=Path, default=default_weights, help="Cutie v1.0 权重")
    parser.add_argument("--workers", type=int, default=1, help="GPU 工作进程数，默认 1")
    parser.add_argument("--overwrite", action="store_true", help="重建已存在的 mask 视频")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if not torch.cuda.is_available():
        raise RuntimeError("Cutie mask 生成需要 CUDA GPU")
    if args.workers < 1 or args.workers > torch.cuda.device_count():
        raise ValueError("workers 必须在 1 和 GPU 数量之间")
    videos = sorted({video for root in args.src for video in root.rglob("gripper.mp4")})
    if not videos:
        raise ValueError("未找到 gripper.mp4")
    pending_videos = [
        video
        for video in videos
        if args.overwrite or not video.with_name("mask_gripper.mp4").exists()
    ]
    if not pending_videos:
        print("所有 mask 视频均已存在")
        return
    try:
        target_indices = [resolve_video_target(video) for video in pending_videos]
    except TargetSelectionCancelled as exc:
        parser.exit(1, f"{exc}\n")
    tasks = [
        (
            video,
            args.weights,
            index % torch.cuda.device_count(),
            args.overwrite,
            target_index,
        )
        for index, (video, target_index) in enumerate(zip(pending_videos, target_indices))
    ]
    if args.workers == 1:
        matting = Matting(args.weights, torch.device("cuda:0"))
        for video, _, _, overwrite, target_index in tqdm(tasks, desc="生成 mask"):
            print(process_video(video, matting, overwrite, target_index))
    else:
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            for output in tqdm(pool.imap(_run_worker, tasks), total=len(tasks), desc="生成 mask"):
                print(output)


if __name__ == "__main__":
    main()
