"""验证 RM75 数据预处理优化不改变模型输入。"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from controller.dataset.mask_image_dataset_videos_rm75 import MaskImageDataset


def _reference_process(
    dataset: MaskImageDataset,
    rgb_images: np.ndarray,
    mask_images: np.ndarray,
) -> np.ndarray:
    """保留优化前的处理步骤，作为逐元素对照。"""
    rgb = torch.from_numpy(rgb_images).float().permute(0, 3, 1, 2)
    mask = torch.from_numpy(mask_images).float().permute(0, 3, 1, 2)
    org_h, org_w = rgb.shape[2:]
    dst_w = dataset.image_size[0]
    dst_h = round(org_h / org_w * dst_w / 14.0) * 14
    rgb = F.interpolate(
        rgb / 255.0,
        size=(dst_h, dst_w),
        mode="bilinear",
        align_corners=False,
    )
    mask = F.interpolate(mask, size=(dst_h, dst_w), mode="nearest")
    combined = torch.cat([rgb, (mask > 200).float()], dim=1)
    return F.interpolate(
        combined,
        size=dataset.output_size,
        mode="bilinear",
        align_corners=False,
    ).numpy()


def test_mask_loader_preserves_reference_blue_channel(tmp_path: Path) -> None:
    """灰度图和 RGB 图都只解码参考实现最终保留的通道。"""
    gray = np.arange(48, dtype=np.uint8).reshape(6, 8)
    color = np.stack([gray, gray + 20, gray + 40], axis=-1)
    Image.fromarray(gray).save(tmp_path / "gray.png")
    Image.fromarray(color).save(tmp_path / "color.png")

    dataset = object.__new__(MaskImageDataset)
    dataset.path_prefix_map = {}
    result = dataset._load_mask_images_from_paths(
        ["gray.png", "color.png"], tmp_path
    )

    assert result.shape == (2, 6, 8, 1)
    np.testing.assert_array_equal(result[0, ..., 0], gray)
    np.testing.assert_array_equal(result[1, ..., 0], color[..., 2])


def test_optimized_processing_matches_reference_for_same_and_smaller_size() -> None:
    """跳过同尺寸插值时结果不变，缩小输出时仍沿用原路径。"""
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(1, 96, 128, 3), dtype=np.uint8)
    mask = rng.integers(0, 256, size=(1, 96, 128, 1), dtype=np.uint8)
    dataset = object.__new__(MaskImageDataset)
    dataset.image_size = (518, 518)

    for output_size in ((392, 518), (294, 518)):
        dataset.output_size = output_size
        expected = _reference_process(dataset, rgb, mask)
        actual = dataset._process_mask_image_batch(rgb, mask)
        np.testing.assert_array_equal(actual, expected)
