"""验证 RM75 转换、路径迁移和观测编码的关键行为。"""

import os
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch
import zarr

from controller.dataset.mask_image_dataset_videos_rm75 import MaskImageDataset
from controller.model.vision.obs_encoder_rm75 import ObsEncoder
from scripts.rm75.trans_hdf5_to_zarr_gripper import (
    align_timestamps,
    extract_episode_frames,
    find_first_gripper_loosen,
    minmax_normalize,
    trans_items,
)
from scripts.rm75.video_extract_masks_gripper import load_init_mask


def test_alignment_boundary_and_reference_trim() -> None:
    """50 ms 边界内匹配，边界外丢弃，夹爪裁剪与参考一致。"""
    aligned = align_timestamps(np.array([1.0, 2.0]), [np.array([1.049, 2.051])])
    assert aligned[1] == [1.049, None]
    assert find_first_gripper_loosen([[0] * 7 + [0.5], [0] * 7 + [1.0]]) == 1
    assert np.array_equal(minmax_normalize(np.array([[0.0, 0.5]]), [-1, 0], [1, 1]), [[0, 0]])


def test_missing_mask_and_invalid_annotation(tmp_path: Path) -> None:
    """缺少 mask 视频或有效多边形时立即失败。"""
    episode = tmp_path / "episode_0"
    episode.mkdir()
    (episode / "gripper.mp4").touch()
    with pytest.raises(FileNotFoundError):
        extract_episode_frames(episode, tmp_path, tmp_path / "frames")
    annotation = episode / "gripper.json"
    annotation.write_text('{"shapes": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_init_mask(annotation, (32, 32, 3))


def _make_episode(root: Path, color: tuple[int, int, int]) -> None:
    """创建三帧可对齐的 HDF5 与真实可解码视频。"""
    episode = root / "episode_0"
    episode.mkdir(parents=True)
    timestamps = np.array([1.0, 1.02, 1.04])
    for name, monochrome in (("gripper.mp4", False), ("mask_gripper.mp4", True)):
        writer = cv2.VideoWriter(str(episode / name), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32), isColor=not monochrome)
        assert writer.isOpened()
        for _ in timestamps:
            frame = np.full((32, 32), 255, np.uint8) if monochrome else np.full((32, 32, 3), color, np.uint8)
            writer.write(frame)
        writer.release()
    with h5py.File(episode / "proprio.hdf5", "w") as stream:
        values = {
            "observations/images/cam_gripper_timestamp": timestamps,
            "action/joint_action/position": np.zeros((3, 7), np.float32),
            "action/joint_action/timestamp": timestamps,
            "action/gripper_action/position": np.ones((3, 1), np.float32),
            "action/gripper_action/timestamp": timestamps,
            "observations/joint_state/qpos": np.zeros((3, 7), np.float32),
            "observations/joint_state/timestamp": timestamps,
            "observations/gripper_state/position": np.ones((3, 1), np.float32),
            "observations/gripper_state/timestamp": timestamps,
        }
        for key, value in values.items():
            stream.create_dataset(key, data=value)


def test_frame_cache_refreshes_when_mask_video_changes(tmp_path: Path) -> None:
    """mask 视频保持帧数不变时，重新提取更新后的内容。"""
    _make_episode(tmp_path, (40, 80, 120))
    episode = tmp_path / "episode_0"
    cache_dir = tmp_path / "frames"
    assert extract_episode_frames(episode, tmp_path, cache_dir) == 3
    cached_mask = cache_dir / "episode_0/mask/frame_000000.jpg"
    assert cv2.imread(str(cached_mask), cv2.IMREAD_GRAYSCALE).mean() > 200

    mask_video = episode / "mask_gripper.mp4"
    previous_mtime = mask_video.stat().st_mtime_ns
    writer = cv2.VideoWriter(str(mask_video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32), isColor=False)
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((32, 32), np.uint8))
    writer.release()
    os.utime(mask_video, ns=(previous_mtime + 1_000_000_000, previous_mtime + 1_000_000_000))

    assert extract_episode_frames(episode, tmp_path, cache_dir) == 3
    assert cv2.imread(str(cached_mask), cv2.IMREAD_GRAYSCALE).mean() < 10


def test_multi_source_relative_paths_and_padding(tmp_path: Path) -> None:
    """重名 episode 保持独立，图像可读取且尾部动作窗口可补齐。"""
    sources = [tmp_path / "source_a", tmp_path / "source_b"]
    for source, color in zip(sources, ((255, 0, 0), (0, 0, 255))):
        _make_episode(source, color)
    dst = tmp_path / "gripper_zarr"
    trans_items(sources, dst, tmp_path / "frames", Path("assets/rm75/rm_75.urdf"), workers=2)
    root = zarr.open(dst, mode="r")
    assert root["meta/episode_ends"][:].tolist() == [3, 6]
    assert "source_000" in root["data/gripper_image_paths"][0]
    assert "source_001" in root["data/gripper_image_paths"][3]
    assert not Path(root["data/gripper_image_paths"][0]).is_absolute()
    dataset = MaskImageDataset([str(dst)], horizon=4, pad_after=3)
    sample = dataset[2]
    assert sample["action"].shape == (4, 8)
    assert sample["obs"]["rgbm"].shape == (1, 4, 294, 518)
    assert torch.equal(sample["action"][0], sample["action"][1])
    assert sample["obs"]["rgbm"].max() <= 1
    with pytest.raises(FileExistsError):
        trans_items(sources, dst, tmp_path / "frames", Path("assets/rm75/rm_75.urdf"), workers=1)


def test_normalizer_excludes_validation_episodes(tmp_path: Path) -> None:
    """验证 episode 的独有极值不参与训练归一化。"""
    root = zarr.open(str(tmp_path / "normalizer.zarr"), mode="w")
    root.create_dataset("meta/episode_ends", data=np.array([2, 4], dtype=np.int32))
    values = np.repeat(np.array([[0.0], [1.0], [100.0], [101.0]], dtype=np.float32), 8, axis=1)
    root.create_dataset("data/action", data=values)
    root.create_dataset("data/state", data=values * 2)
    paths = np.array(["unused"] * 4, dtype="U6")
    root.create_dataset("data/gripper_image_paths", data=paths)
    root.create_dataset("data/mask_image_paths", data=paths)

    dataset = MaskImageDataset([str(tmp_path / "normalizer.zarr")], val_ratio=0.5)
    train_episode = int(np.flatnonzero(dataset.train_masks[0])[0])
    expected_actions = values[train_episode * 2:(train_episode + 1) * 2]
    stats = dataset.get_normalizer().get_input_stats()
    np.testing.assert_array_equal(stats["action"]["min"].numpy(), expected_actions.min(axis=0))
    np.testing.assert_array_equal(stats["action"]["max"].numpy(), expected_actions.max(axis=0))
    np.testing.assert_array_equal(stats["right_state"]["max"].numpy(), expected_actions.max(axis=0) * 2)


def test_old_absolute_image_path_mapping(tmp_path: Path) -> None:
    """旧 Zarr 中绝对路径可显式映射，缺失图像报出实际解析路径。"""
    actual = tmp_path / "images"
    actual.mkdir()
    image = actual / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
    dataset = object.__new__(MaskImageDataset)
    dataset.path_prefix_map = {Path("/old/images"): actual}
    assert dataset._resolve_image_path("/old/images/frame.jpg", tmp_path) == image
    with pytest.raises(FileNotFoundError):
        dataset._resolve_image_path("/old/images/missing.jpg", tmp_path)


def test_encoder_uses_rgb_and_observation_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """策略接口无需真实动作，DINO 输入是 RGB 且骨干被冻结。"""
    class DummyDino(torch.nn.Module):
        """记录编码器送入的色彩通道。"""

        def __init__(self) -> None:
            """建立可冻结的单参数骨干。"""
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.observed = None

        def get_intermediate_layers(self, image: torch.Tensor, n: int):
            """模拟两个图像 patch 的 DINO 特征。"""
            self.observed = image.detach()
            return [torch.zeros((len(image), 4, 768), device=image.device)]

    dino = DummyDino()
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: dino)
    shape_meta = {"obs": {"rgbm": {"shape": [4, 28, 28], "horizon": 1}}}
    encoder = ObsEncoder(shape_meta, {"head": {"model_type": "dinov2_vitb14", "source_dir": None, "local_weights_path": None}})
    rgbm = torch.zeros((1, 1, 4, 28, 28))
    rgbm[:, :, 2] = 1
    output = encoder({"rgbm": rgbm, "right_state": torch.zeros((1, 1, 8))}, training=False)
    assert output.shape == (1, 5, 768)
    assert encoder.output_shape() == ((1, 5, 768), [4, 1])
    assert dino.observed[:, 0].mean() > dino.observed[:, 2].mean()
    assert not dino.weight.requires_grad
    output.sum().backward()
    assert encoder.state_net[0].weight.grad is not None


def test_multigpu_launcher_passes_hydra_arguments(tmp_path: Path) -> None:
    """启动脚本用 uv 指定独立环境，并透传多卡及 Hydra 参数。"""
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "RM75_VENV": str(tmp_path),
        "RM75_NUM_PROCESSES": "2",
        "RM75_PORT": "27000",
    }
    result = subprocess.run(["bash", "train_rm75.sh", "training.num_epochs=1"],
                            check=True, capture_output=True, text=True, env=env)
    arguments = result.stdout.splitlines()
    assert arguments[:9] == [
        "run", "--no-project", "--python", str(tmp_path / "bin/python"),
        "accelerate", "launch", "--num_processes", "2", "--main_process_port",
    ]
    assert arguments[9] == "27000"
    assert arguments[-1] == "training.num_epochs=1"
