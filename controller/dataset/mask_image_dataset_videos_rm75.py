"""读取 RM75 单相机 Zarr 演示并转换为控制器的 RGBM 观测。"""

from typing import Dict
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from controller.common.pytorch_util import dict_apply
from controller.common.streaming_replay_buffer import StreamingReplayBuffer
from controller.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from controller.dataset.base_dataset import BaseImageDataset
from controller.model.common.normalizer import LinearNormalizer


class MaskImageDataset(BaseImageDataset):
    """单臂单相机 gripper zarr 数据集，由 trans_hdf5_to_zarr_gripper.py 生成。"""

    def __init__(
        self,
        zarr_paths,
        horizon=1,
        n_obs_steps=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        image_size=(518, 518),
        output_size=(294, 518),
        path_prefix_map=None,
    ):
        """建立序列采样器，并为旧数据配置可选的图像路径前缀替换。"""
        super().__init__()
        self.image_size = tuple(int(x) for x in image_size)
        self.output_size = tuple(int(x) for x in output_size)
        self.path_prefix_map = {
            Path(source): Path(target)
            for source, target in (path_prefix_map or {}).items()
        }

        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        self.zarr_roots = []

        for zarr_path in zarr_paths:
            self.zarr_roots.append(Path(zarr_path))
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path,
                keys=[
                    "gripper_image_paths",
                    "mask_image_paths",
                    "state",
                    "action",
                ],
            )
            self.replay_buffers.append(replay_buffer)

            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask, max_n=max_train_episodes, seed=seed
            )
            self.train_masks.append(train_mask)

            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                key_first_k=dict(
                    gripper_image_paths=n_obs_steps,
                    mask_image_paths=n_obs_steps,
                ),
            )
            self.samplers.append(sampler)
            self.sampler_lens.append(len(sampler))

        if not any(self.sampler_lens):
            raise ValueError(f"RM75 数据集没有可训练序列: {zarr_paths}")

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        """复用 Zarr 引用，创建互斥的验证采样器。"""
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []

        for i, replay_buffer in enumerate(self.replay_buffers):
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_masks[i],
                key_first_k=dict(
                    gripper_image_paths=self.n_obs_steps,
                    mask_image_paths=self.n_obs_steps,
                ),
            )
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
        return val_set

    def _resolve_image_path(self, path_str, zarr_root):
        """将相对路径或旧机器路径映射到当前机器上的图像。"""
        path = Path(path_str)
        if path.is_absolute():
            for source, target in sorted(
                self.path_prefix_map.items(), key=lambda pair: len(pair[0].parts), reverse=True
            ):
                if path == source or source in path.parents:
                    path = target / path.relative_to(source)
                    break
        else:
            path = zarr_root / path
        if not path.is_file():
            raise FileNotFoundError(f"RM75 图像不存在: {path}")
        return path

    def _load_images_from_paths(self, image_paths, zarr_root):
        """按原参考实现将 RGB 图像读取为 BGR 数组。"""
        images = []
        for path_str in image_paths:
            full_path = self._resolve_image_path(path_str, zarr_root)
            with Image.open(full_path) as opened:
                img = np.array(opened.convert("RGB"))
            img = img[:, :, ::-1].copy()
            images.append(img)
        return np.array(images)

    def _process_mask_image_batch(self, rgb_images, mask_images):
        """复现参考数据集的图像缩放、mask 阈值和最终插值。"""
        rgb = torch.from_numpy(rgb_images).float().permute(0, 3, 1, 2)
        mask = torch.from_numpy(mask_images).float().permute(0, 3, 1, 2)

        org_h, org_w = rgb.shape[2], rgb.shape[3]
        dst_shape_w = self.image_size[0]
        dst_shape_h = round(org_h / org_w * dst_shape_w / 14.0) * 14

        rgb = F.interpolate(
            rgb / 255.0,
            size=(dst_shape_h, dst_shape_w),
            mode="bilinear",
            align_corners=False,
        )
        mask = F.interpolate(mask, size=(dst_shape_h, dst_shape_w), mode="nearest")
        mask = (mask > 200).float()

        combined = torch.cat([rgb, mask], dim=1)
        combined = F.interpolate(
            combined,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        return combined.numpy()

    def _sample_to_data(self, sample, zarr_root):
        """生成模型使用的 RGBM、8 维状态与 8 维动作。"""
        state = sample["state"].astype(np.float32)
        t_slice = slice(self.n_obs_steps)

        gripper_images = self._load_images_from_paths(
            sample["gripper_image_paths"][t_slice], zarr_root
        )
        mask_images = self._load_images_from_paths(
            sample["mask_image_paths"][t_slice], zarr_root
        )
        mask_images = mask_images[:, :, :, :1]

        mask_processed_frames = self._process_mask_image_batch(
            gripper_images, mask_images
        )

        return {
            "obs": {
                "rgbm": mask_processed_frames,
                "right_state": state[t_slice],
            },
            "action": sample["action"].astype(np.float32),
        }

    def get_normalizer(self, mode="limits", **kwargs):
        """仅从训练 episode 拟合动作及状态的线性归一化器。"""
        actions = []
        states = []
        for replay_buffer, train_mask in zip(self.replay_buffers, self.train_masks):
            episode_ends = replay_buffer.episode_ends
            for index in np.flatnonzero(train_mask):
                start = 0 if index == 0 else int(episode_ends[index - 1])
                end = int(episode_ends[index])
                actions.append(replay_buffer["action"][start:end])
                states.append(replay_buffer["state"][start:end])

        data = {
            "action": np.concatenate(actions, axis=0),
            "right_state": np.concatenate(states, axis=0),
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """按全局索引获取一个补齐后的动作窗口。"""
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        curr_idx = idx
        sampler_idx = 0
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                sampler_idx = i
                break
            curr_idx -= length

        data = self._sample_to_data(sample, self.zarr_roots[sampler_idx])
        return dict_apply(data, torch.from_numpy)

    def __len__(self):
        """返回所有 Zarr 中训练窗口的总数。"""
        return sum(self.sampler_lens)
