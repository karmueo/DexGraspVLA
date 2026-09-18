"""RM75 单目 RGBM 与关节状态编码器，提供扩散策略所需的观测 token。"""

import logging

import torch
import torch.nn as nn
import torchvision.transforms as T
from einops.layers.torch import Rearrange

from controller.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

feature_dim_dict = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

OBS_DIM = 8
DINO_PATCH_SIZE = 14


class ObsEncoder(ModuleAttrMixin):
    """RM75 单臂单相机 obs encoder，适配 mask_image_dataset_videos_rm75 输出。"""

    def __init__(self, shape_meta: dict, model_config: dict):
        """构造冻结的 DINOv2 骨干和可训练的 mask、状态融合网络。"""
        super().__init__()

        head_cfg = model_config["head"]
        source_dir = head_cfg.get("source_dir")
        weights_path = head_cfg.get("local_weights_path")
        self.dino_head = torch.hub.load(
            source_dir or "facebookresearch/dinov2",
            head_cfg["model_type"],
            pretrained=weights_path is None,
            force_reload=False,
            trust_repo=True,
            source="local" if source_dir else "github",
        )
        if weights_path is not None:
            self.dino_head.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

        self.dino_head.eval()
        for param in self.dino_head.parameters():
            param.requires_grad = False

        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        self.dino_transform = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        head_feature_dim = feature_dim_dict[head_cfg["model_type"]]
        self.feature_dim = head_feature_dim

        self.mask_process_net = nn.ModuleDict(
            {
                "patch_embed": nn.Sequential(
                    nn.Conv2d(1, head_feature_dim, kernel_size=14, stride=14),
                    nn.Flatten(2),
                    Rearrange("b c n -> b n c"),
                    nn.LayerNorm(head_feature_dim),
                ),
                "transformer": nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(
                        d_model=head_feature_dim,
                        nhead=8,
                        dim_feedforward=head_feature_dim * 4,
                        dropout=0.0,
                        activation=nn.GELU(),
                        batch_first=True,
                        norm_first=True,
                    ),
                    num_layers=4,
                ),
            }
        )

        self.head_net = nn.Sequential(
            nn.Linear(head_feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

        self.state_net = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

        self.shape_meta = shape_meta
        rgbm_shape = shape_meta["obs"]["rgbm"]["shape"]
        self.rgbm_h = rgbm_shape[1]
        self.rgbm_w = rgbm_shape[2]
        if self.rgbm_h % DINO_PATCH_SIZE or self.rgbm_w % DINO_PATCH_SIZE:
            raise ValueError(f"RGBM 尺寸必须是 14 的倍数: {rgbm_shape}")
        self.num_patches = (self.rgbm_h // DINO_PATCH_SIZE) * (self.rgbm_w // DINO_PATCH_SIZE)

        logger.info(
            "RM75 ObsEncoder params: %e, rgbm=%sx%s, patches=%d",
            sum(p.numel() for p in self.parameters()),
            self.rgbm_h,
            self.rgbm_w,
            self.num_patches,
        )

    def forward_head(self, rgbm_data, training=True):
        """提取冻结的 DINO 图像 token 并与 mask token 融合。"""
        # rgbm_data: B,T,4,H,W
        b, t = rgbm_data.shape[:2]
        rgb_data = rgbm_data[:, :, :3]
        rgb_data = rgb_data[:, :, [2, 1, 0], ...]
        mask_data = rgbm_data[:, :, 3:]

        rgb_data = rgb_data.reshape(b * t, *rgb_data.shape[2:])
        if training:
            rgb_data = self.color_jitter(rgb_data)
        rgb_data = self.dino_transform(rgb_data)
        with torch.no_grad():
            self.dino_head.eval()
            rgb_feature = self.dino_head.get_intermediate_layers(rgb_data, n=1)[0]

        mask_data = mask_data.reshape(b * t, *mask_data.shape[2:])
        mask_feature = self.mask_process_net["patch_embed"](mask_data)
        mask_feature = self.mask_process_net["transformer"](mask_feature)

        combined_feature = torch.cat([rgb_feature, mask_feature], dim=-1)
        head_feature = self.head_net(combined_feature)
        return head_feature.reshape(b, t * head_feature.shape[1], head_feature.shape[-1])

    def forward_state(self, state_data):
        """将 8 维机械臂与夹爪状态编码为单个 token。"""
        b, t = state_data.shape[:2]
        state_data = state_data.reshape(b * t, -1)
        state_feature = self.state_net(state_data)
        return state_feature.reshape(b, t, state_feature.shape[-1])

    def forward(self, obs_dict, training=True):
        """从 RGBM 和状态构造观测 token，不读取真实动作。"""
        embeddings = [
            self.forward_head(obs_dict["rgbm"], training),
            self.forward_state(obs_dict["right_state"]),
        ]
        return torch.cat(embeddings, dim=1)

    @torch.no_grad()
    def output_shape(self):
        """返回 token 总形状与图像、状态各自的长度。"""
        n_obs = self.shape_meta["obs"]["rgbm"]["horizon"]
        head_tokens = n_obs * self.num_patches
        state_tokens = n_obs
        total_tokens = head_tokens + state_tokens
        return (1, total_tokens, self.feature_dim), [head_tokens, state_tokens]
