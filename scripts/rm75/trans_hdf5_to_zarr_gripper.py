#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单臂单相机 gripper 数据集 zarr 转换脚本。

数据格式 (org_data/0):
    episode_*/proprio.hdf5
    episode_*/gripper.mp4
    episode_*/mask_gripper.mp4

时间戳以 action/joint_action/timestamp 为参考对齐，其余流（视频、夹爪、state）向其对齐。
action/state 维度: 7 关节 + 1 夹爪 = 8
min/max 从 rm_75.urdf 读取，夹爪固定 [0, 1]。

示例:
    python scripts/rm75/trans_hdf5_to_zarr_gripper.py --src data/raw --dst data/gripper_zarr
"""

import argparse
import bisect
import json
import os
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor as EpisodeExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr
from tqdm import tqdm

CHUNK_SIZE = 1000
ALIGN_THRESHOLD_S = 0.05
ACTION_DIM = 8


def parse_urdf_limits(urdf_path: Path):
    """读取 RM75 的七个旋转关节限位，并附加夹爪 [0,1] 限位。"""
    root = ET.parse(urdf_path).getroot()
    lowers, uppers = [], []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lowers.append(float(limit.get("lower")))
        uppers.append(float(limit.get("upper")))
    if len(lowers) != 7:
        raise ValueError(f"期望 7 个 revolute 关节，实际 {len(lowers)}: {urdf_path}")
    lowers.append(0.0)
    uppers.append(1.0)
    return lowers, uppers


def minmax_normalize(x, min_vals, max_vals):
    """按参考实现将关节与夹爪数值线性映射到 [-1,1]。"""
    min_vals = np.array(min_vals).reshape(1, -1)
    max_vals = np.array(max_vals).reshape(1, -1)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return 2 * (x - min_vals) / range_vals - 1


def align_timestamps(ref_timestamps, query_timestamps):
    """按最近时间戳匹配各数据流，超过 50 ms 时返回 None。"""
    aligned = [ref_timestamps.tolist()]
    for query_ts in query_timestamps:
        curr = []
        for t_ref in ref_timestamps:
            pos = bisect.bisect_left(query_ts, t_ref)
            candidates = []
            if pos > 0:
                candidates.append(query_ts[pos - 1])
            if pos < len(query_ts):
                candidates.append(query_ts[pos])
            if not candidates:
                curr.append(None)
                continue
            closest = min(candidates, key=lambda x: abs(x - t_ref))
            curr.append(None if abs(closest - t_ref) > ALIGN_THRESHOLD_S else closest)
        aligned.append(curr)
    return aligned


def episode_relative(episode_dir: Path, input_root: Path):
    """返回当前输入根目录下 episode 的相对路径。"""
    rel = episode_dir.relative_to(input_root)
    return rel.as_posix()


def video_signature(path: Path):
    """记录源视频身份和修改时间，用于识别需要重建的帧缓存。"""
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def extract_episode_frames(episode_dir: Path, input_root: Path, frames_cache_dir: Path, overwrite=False):
    """同时解码相机及 mask 视频，检查二者帧数并保存到缓存。"""
    rel = episode_relative(episode_dir, input_root)
    cache_dir = frames_cache_dir / rel
    gripper_dir = cache_dir / "gripper"
    mask_dir = cache_dir / "mask"
    metadata_path = cache_dir / "source_videos.json"

    video_path = episode_dir / "gripper.mp4"
    mask_path = episode_dir / "mask_gripper.mp4"
    if not video_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"缺少 gripper.mp4 或 mask_gripper.mp4: {episode_dir}")

    source_signatures = {
        "gripper": video_signature(video_path),
        "mask": video_signature(mask_path),
    }
    gripper_frames = list(gripper_dir.glob("frame_*.jpg")) if gripper_dir.exists() else []
    mask_frames = list(mask_dir.glob("frame_*.jpg")) if mask_dir.exists() else []
    try:
        cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache_metadata = None
    if (not overwrite and gripper_frames and len(gripper_frames) == len(mask_frames)
            and cache_metadata == {"sources": source_signatures, "frame_count": len(gripper_frames)}):
        return len(gripper_frames)

    metadata_path.unlink(missing_ok=True)
    shutil.rmtree(gripper_dir, ignore_errors=True)
    shutil.rmtree(mask_dir, ignore_errors=True)

    gripper_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not cv2.imwrite(str(gripper_dir / f"frame_{frame_idx:06d}.jpg"), frame):
            raise OSError(f"图像写入失败: {gripper_dir}")
        frame_idx += 1
    cap.release()

    video_frames = frame_idx
    cap = cv2.VideoCapture(str(mask_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 mask 视频: {mask_path}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not cv2.imwrite(str(mask_dir / f"frame_{frame_idx:06d}.jpg"), frame):
            raise OSError(f"mask 写入失败: {mask_dir}")
        frame_idx += 1
    cap.release()
    if frame_idx != video_frames or len(list(gripper_dir.glob("frame_*.jpg"))) != video_frames:
        raise ValueError(f"相机和 mask 视频帧数不匹配: {episode_dir}: {video_frames} vs {frame_idx}")
    metadata_path.write_text(
        json.dumps({"sources": source_signatures, "frame_count": video_frames}), encoding="utf-8"
    )
    return video_frames


def extract_frames_worker(args):
    """线程任务包装，返回 HDF5 路径与已解码帧数。"""
    cv2.setNumThreads(0)
    h5file, input_root, frames_cache_dir, overwrite = args
    episode_dir = Path(h5file).parent
    count = extract_episode_frames(episode_dir, Path(input_root), Path(frames_cache_dir), overwrite)
    return h5file, count


def find_first_gripper_loosen(joint_action):
    """维持参考训练集的末尾夹爪动作裁剪点。"""
    for i in range(len(joint_action) - 1, 0, -1):
        if joint_action[i - 1][7] < 0.95:
            return i
    return len(joint_action)


def sync_action_state(f, h5file, input_root, frames_cache_dir):
    """按动作时间戳对齐视频、夹爪与状态，返回 8 维序列。"""
    episode_dir = Path(h5file).parent
    rel = episode_relative(episode_dir, Path(input_root))
    gripper_dir = frames_cache_dir / rel / "gripper"
    mask_dir = frames_cache_dir / rel / "mask"

    if not gripper_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"缺少解码图像目录: {episode_dir}")

    video_ts = f["observations/images/cam_gripper_timestamp"][()]
    action_joint = f["action/joint_action/position"][()]
    action_joint_ts = f["action/joint_action/timestamp"][()]
    action_gripper = f["action/gripper_action/position"][()]
    action_gripper_ts = f["action/gripper_action/timestamp"][()]
    state_joint = f["observations/joint_state/qpos"][()]
    state_joint_ts = f["observations/joint_state/timestamp"][()]
    state_gripper = f["observations/gripper_state/position"][()]
    state_gripper_ts = f["observations/gripper_state/timestamp"][()]

    if action_gripper.ndim == 1:
        action_gripper = action_gripper[:, None]
    if state_gripper.ndim == 1:
        state_gripper = state_gripper[:, None]

    frame_count = len(list(gripper_dir.glob("frame_*.jpg")))
    if frame_count == 0:
        raise ValueError(f"无视频帧: {h5file}")
    if frame_count != len(video_ts):
        raise ValueError(f"视频帧数与时间戳不一致: {h5file}, frames={frame_count}, ts={len(video_ts)}")

    ref_ts = action_joint_ts
    aligned = align_timestamps(
        ref_ts,
        [video_ts, action_gripper_ts, state_joint_ts, state_gripper_ts],
    )

    ts_maps = {
        "video": {ts: i for i, ts in enumerate(video_ts)},
        "action_gripper": {ts: i for i, ts in enumerate(action_gripper_ts)},
        "state_joint": {ts: i for i, ts in enumerate(state_joint_ts)},
        "state_gripper": {ts: i for i, ts in enumerate(state_gripper_ts)},
    }

    joint_action, joint_state, gripper_paths, mask_paths = [], [], [], []
    skipped = 0

    for ref_idx in range(len(ref_ts)):
        ts_video = aligned[1][ref_idx]
        ts_action_gripper = aligned[2][ref_idx]
        ts_state_joint = aligned[3][ref_idx]
        ts_state_gripper = aligned[4][ref_idx]

        if any(ts is None for ts in [ts_video, ts_action_gripper, ts_state_joint, ts_state_gripper]):
            skipped += 1
            continue

        video_frame_idx = ts_maps["video"][ts_video]
        gripper_path = gripper_dir / f"frame_{video_frame_idx:06d}.jpg"
        mask_path = mask_dir / f"frame_{video_frame_idx:06d}.jpg"
        if not gripper_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"帧文件缺失: {h5file}, frame_{video_frame_idx:06d}")

        joint_action.append(
            np.hstack([
                action_joint[ref_idx],
                action_gripper[ts_maps["action_gripper"][ts_action_gripper]].reshape(-1),
            ])
        )
        joint_state.append(
            np.hstack([
                state_joint[ts_maps["state_joint"][ts_state_joint]],
                state_gripper[ts_maps["state_gripper"][ts_state_gripper]].reshape(-1),
            ])
        )
        gripper_paths.append(str(gripper_path))
        mask_paths.append(str(mask_path))

    if not joint_action:
        raise ValueError(f"无有效对齐帧: {h5file}, 跳过 {skipped}/{len(aligned[0])} 帧")
    if skipped > 0:
        print(f"跳过 {skipped} 帧未对齐: {h5file}, 保留 {len(joint_action)} 帧")

    end_idx = find_first_gripper_loosen(joint_action)
    return {
        "joint_action": joint_action[:end_idx],
        "joint_state": joint_state[:end_idx],
        "gripper": gripper_paths[:end_idx],
        "mask": mask_paths[:end_idx],
        "frame_count": end_idx,
    }


def process_h5_file(h5file, input_root, frames_cache_dir):
    """读取单段 HDF5 并按时间戳对齐到解码图像。"""
    with h5py.File(h5file, "r") as stream:
        return sync_action_state(stream, h5file, Path(input_root), Path(frames_cache_dir))


def trans_items(src_paths, dst_path, frames_cache_dir, urdf_path, workers, overwrite=False):
    """从一个或多个原始目录创建兼容参考训练的相对路径 Zarr。"""
    if isinstance(src_paths, (str, Path)):
        src_paths = [src_paths]
    src_paths = [Path(source).resolve() for source in src_paths]
    dst_path = Path(dst_path).resolve()
    frames_cache_dir = Path(frames_cache_dir).resolve()
    if workers < 1:
        raise ValueError("workers 必须大于零")
    if dst_path.exists() and any(dst_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Zarr 已存在，需显式 --overwrite: {dst_path}")
    if frames_cache_dir == dst_path or dst_path in frames_cache_dir.parents:
        raise ValueError("帧缓存不能放在 Zarr 目录内")
    action_lower, action_upper = parse_urdf_limits(Path(urdf_path))
    episode_specs = []
    for source_index, source in enumerate(src_paths):
        if not source.is_dir():
            raise FileNotFoundError(source)
        cache_root = frames_cache_dir / f"source_{source_index:03d}"
        for h5file in sorted(source.rglob("proprio.hdf5")):
            episode_specs.append((h5file, source, cache_root))
    if not episode_specs:
        raise ValueError("没有找到 proprio.hdf5")

    extraction_args = [(str(h5file), str(source), str(cache_root), overwrite)
                       for h5file, source, cache_root in episode_specs]
    with EpisodeExecutor(max_workers=workers) as executor:
        extracted = list(tqdm(executor.map(extract_frames_worker, extraction_args),
                              total=len(episode_specs), desc="提取视频帧"))
    rows = []
    for (h5file, source, cache_root), (_, frame_count) in tqdm(
        zip(episode_specs, extracted), total=len(episode_specs), desc="对齐时间戳"
    ):
        with h5py.File(h5file, "r") as stream:
            timestamp_count = len(stream["observations/images/cam_gripper_timestamp"])
        if frame_count != timestamp_count:
            raise ValueError(f"视频帧数与时间戳不一致: {h5file}: {frame_count} vs {timestamp_count}")
        data = process_h5_file(h5file, source, cache_root)
        if data["frame_count"] == 0:
            raise ValueError(f"裁剪后 episode 为空: {h5file}")
        rows.append((h5file, source, cache_root, data))

    dst_path.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(dst_path))
    root = zarr.group(store=store, overwrite=True)
    total_frames = sum(data["frame_count"] for _, _, _, data in rows)
    compressed = zarr.Blosc(cname="zstd", clevel=3)
    actions = root.create_dataset("data/action", shape=(total_frames, ACTION_DIM),
                                  chunks=(CHUNK_SIZE, ACTION_DIM), dtype="float32", compressor=compressed)
    states = root.create_dataset("data/state", shape=(total_frames, ACTION_DIM),
                                 chunks=(CHUNK_SIZE, ACTION_DIM), dtype="float32", compressor=compressed)
    relative_paths = [os.path.relpath(path, dst_path)
                      for _, _, _, data in rows for path in data["gripper"] + data["mask"]]
    path_width = max(map(len, relative_paths))
    image_arrays = {
        name: root.create_dataset(f"data/{name}", shape=(total_frames,), chunks=(CHUNK_SIZE,),
                                  dtype=f"U{path_width}", compressor=compressed)
        for name in ("gripper_image_paths", "mask_image_paths")
    }
    episode_ends = root.create_dataset("meta/episode_ends", shape=(len(rows),),
                                       chunks=(10000,), dtype="int32", compressor=compressed)
    folders = root.create_dataset("meta/episode_folders", shape=(len(rows),),
                                  chunks=(10000,), dtype="U200", compressor=compressed)
    files = root.create_dataset("meta/episode_files", shape=(len(rows),),
                                chunks=(10000,), dtype="U200", compressor=compressed)
    last_pos = 0
    for episode_index, (h5file, source, cache_root, data) in enumerate(tqdm(rows, desc="写入 Zarr")):
        frame_count = data["frame_count"]
        span = slice(last_pos, last_pos + frame_count)
        actions[span] = minmax_normalize(np.asarray(data["joint_action"]), action_lower, action_upper)
        states[span] = minmax_normalize(np.asarray(data["joint_state"]), action_lower, action_upper)
        image_arrays["gripper_image_paths"][span] = [os.path.relpath(path, dst_path) for path in data["gripper"]]
        image_arrays["mask_image_paths"][span] = [os.path.relpath(path, dst_path) for path in data["mask"]]
        last_pos += frame_count
        episode_ends[episode_index] = last_pos
        folders[episode_index] = f"{cache_root.name}/{h5file.parent.relative_to(source)}"
        files[episode_index] = h5file.name
    root.attrs["action_lower"] = action_lower
    root.attrs["action_upper"] = action_upper
    root.attrs["action_dim"] = ACTION_DIM
    print(f"完成: {len(rows)} episodes, {total_frames} frames -> {dst_path}")


def main():
    """解析输入目录、帧缓存与 Zarr 路径并启动转换。"""
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="单臂单相机 gripper 数据集转 zarr")
    parser.add_argument("--src", type=Path, nargs="+", required=True)
    parser.add_argument("--dst", type=Path, default=repo_root / "data/gripper_zarr")
    parser.add_argument("--frames-cache-dir", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=repo_root / "assets/rm75/rm_75.urdf")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cache_dir = args.frames_cache_dir or args.dst.parent / "gripper_frames"
    trans_items(args.src, args.dst, cache_dir, args.urdf, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
