"""检查 RM75 环境的依赖、Blackwell GPU 和 BF16 运算。"""

import importlib

import torch


def main() -> None:
    """在 GPU 上执行一次 BF16 矩阵乘法并检查必要模块。"""
    for module_name in ("accelerate", "diffusers", "timm", "hydra", "zarr", "h5py", "cv2", "cutie"):
        importlib.import_module(module_name)
    if not torch.cuda.is_available():
        raise RuntimeError("RM75 训练需要可用的 CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16")
    device = torch.device("cuda")
    # 小矩阵用于检查实际 CUDA 算子能执行。
    matrix = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
    result = matrix @ matrix
    torch.cuda.synchronize()
    if not torch.all(result == 16):
        raise RuntimeError("BF16 矩阵乘法结果异常")
    print(f"RM75 环境可用：torch={torch.__version__} CUDA={torch.version.cuda} GPU={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
