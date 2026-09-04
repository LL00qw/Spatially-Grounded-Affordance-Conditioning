import math

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet_util import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
)


class TextGuidedPointAttention(nn.Module):
    """Point-level text-guided re-grounding used in Eq. (15) of the paper.

    Given per-point features p_i and a text embedding e_d:
        a_i = softmax_i((W_q e_d)^T (W_k p_i) / sqrt(d_a))
        p_tilde_i = p_i + a_i W_v p_i

    The softmax is over points, so the text query produces one scalar support
    weight per target point. This is intentionally *not* point-to-point
    self-attention.
    """

    def __init__(self, point_feat_dim=512, text_feat_dim=512, attn_dim=256):
        super().__init__()
        self.attn_dim = attn_dim
        self.q_proj = nn.Linear(text_feat_dim, attn_dim, bias=False)
        self.k_proj = nn.Conv1d(point_feat_dim, attn_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(point_feat_dim, point_feat_dim, kernel_size=1, bias=False)

    def forward(self, point_feats, text_feats):
        """
        Args:
            point_feats: [B, C, N], C=512 in AGPENet.
            text_feats:  [B, 512].

        Returns:
            grounded_feats: [B, C, N]
            attention:      [B, N]
        """
        q = self.q_proj(text_feats)          # [B, d_a]
        k = self.k_proj(point_feats)         # [B, d_a, N]
        v = self.v_proj(point_feats)         # [B, C, N]

        logits = torch.einsum('bd,bdn->bn', q, k) / math.sqrt(self.attn_dim)
        attention = F.softmax(logits, dim=-1)  # normalize over target points
        grounded_feats = point_feats + attention.unsqueeze(1) * v
        return grounded_feats, attention


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embedding for a diffusion time step."""

    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        time = time * self.scale
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1 + 1e-5)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def __len__(self):
        return self.dim


class TimeNet(nn.Module):
    """Learned time embedding."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.net(t)


class TextEncoder(nn.Module):
    """Frozen CLIP text encoder used to map canonical affordance text to e_d."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=self.device,
        )

    def forward(self, texts):
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer(texts).to(self.device)
        return self.clip_model.encode_text(tokens).to(self.device)


class PointNetPlusPlus(nn.Module):
    """PointNet++ encoder + target-frame text re-grounding.

    The PointNet++ hierarchy matches the paper:
      - 512 centroids with radii (0.1, 0.2, 0.4)
      - 128 centroids with radii (0.4, 0.8)
      - group-all 1024-D global cloud feature c_X
      - feature propagation to a 512-D descriptor p_i per input point

    Text-guided point attention is then applied to those 512-D descriptors.
    """

    def __init__(self, use_point_attention=True, point_attn_dim=256):
        super().__init__()
        self.use_point_attention = use_point_attention

        self.sa1 = PointNetSetAbstractionMsg(
            512,
            [0.1, 0.2, 0.4],
            [32, 64, 128],
            3,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128,
            [0.4, 0.8],
            [64, 128],
            128 + 128 + 64,
            [[128, 128, 256], [128, 196, 256]],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=512 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128])

        # Produce p_i in the 512-D space used by CLIP matching.
        self.final_conv = nn.Sequential(
            nn.Conv1d(128, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )

        self.point_attention = TextGuidedPointAttention(
            point_feat_dim=512,
            text_feat_dim=512,
            attn_dim=point_attn_dim,
        )

    def forward(self, xyz, text_feats=None):
        """
        Args:
            xyz: [B, N, 3]
            text_feats: [B, 512] canonical-affordance CLIP features.

        Returns:
            point_features: [B, 512, N], p_tilde_i when attention is enabled.
            c_x: [B, 1024], geometry-only global point-cloud feature.
        """
        xyz = xyz.contiguous().transpose(1, 2)
        l0_xyz = xyz
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Geometry-only global feature c_X. Do not concatenate text here.
        c_x = l3_points.squeeze(-1)  # [B, 1024]

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(
            l0_xyz,
            l1_xyz,
            torch.cat([l0_xyz, l0_points], dim=1),
            l1_points,
        )

        point_features = self.final_conv(l0_points)  # p_i: [B, 512, N]

        if self.use_point_attention and text_feats is not None:
            point_features, _ = self.point_attention(point_features, text_feats)

        return point_features, c_x


class PoseNet(nn.Module):
    """U-shaped MLP diffusion denoiser with hierarchical modality conditioning.

    The pose path contracts 7 -> 6 -> 4 -> 2 and expands 2 -> 4 -> 6 -> 7.
    Cloud/text/time features are produced at s in {2, 4, 6}. For the Full
    model, cloud and text influence networks produce complementary per-channel
    weights via a softmax over the two modalities.
    """

    def __init__(self, conditioning_mode="adaptive_hierarchical"):
        super().__init__()
        if conditioning_mode not in {"adaptive_hierarchical", "hierarchical_fixed"}:
            raise ValueError(
                "conditioning_mode must be 'adaptive_hierarchical' or 'hierarchical_fixed'"
            )
        self.conditioning_mode = conditioning_mode

        # c_X is geometry-only and is exactly 1024-D in the paper.
        self.cloud_net0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32),
        )
        self.cloud_net3 = nn.Sequential(
            nn.Linear(32, 16), nn.GroupNorm(4, 16), nn.GELU(), nn.Linear(16, 6)
        )
        self.cloud_net2 = nn.Sequential(
            nn.Linear(32, 16), nn.GroupNorm(4, 16), nn.GELU(), nn.Linear(16, 4)
        )
        self.cloud_net1 = nn.Sequential(
            nn.Linear(32, 16), nn.GroupNorm(4, 16), nn.GELU(), nn.Linear(16, 2)
        )

        self.text_net0 = nn.Sequential(
            nn.Linear(512, 256),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 32),
        )
        self.text_net3 = nn.Sequential(
            nn.Linear(32, 16), nn.GroupNorm(4, 16), nn.GELU(), nn.Linear(16, 6)
        )
        self.text_net2 = nn.Sequential(
            nn.Linear(32, 16), nn.GroupNorm(4, 16), nn.GELU(), nn.Linear(16, 4)
        )
        self.text_net1 = nn.Sequential(
            nn.Linear(32, 16), nn.GroupNorm(4, 16), nn.GELU(), nn.Linear(16, 2)
        )

        # The influence networks are symmetric across cloud/text. Each sees the
        # current modality feature, the noisy 7-D pose, the time feature, and
        # the complementary modality feature. A 2-way softmax below makes the
        # resulting weights complementary channel by channel.
        self.cloud_influence_net3 = nn.Sequential(
            nn.Linear(6 + 7 + 6 + 6, 6), nn.GELU(), nn.Linear(6, 6)
        )
        self.cloud_influence_net2 = nn.Sequential(
            nn.Linear(4 + 7 + 4 + 4, 4), nn.GELU(), nn.Linear(4, 4)
        )
        self.cloud_influence_net1 = nn.Sequential(
            nn.Linear(2 + 7 + 2 + 2, 16), nn.GELU(), nn.Linear(16, 2)
        )

        self.text_influence_net3 = nn.Sequential(
            nn.Linear(6 + 7 + 6 + 6, 6), nn.GELU(), nn.Linear(6, 6)
        )
        self.text_influence_net2 = nn.Sequential(
            nn.Linear(4 + 7 + 4 + 4, 4), nn.GELU(), nn.Linear(4, 4)
        )
        self.text_influence_net1 = nn.Sequential(
            nn.Linear(2 + 7 + 2 + 2, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )

        self.time_net3 = TimeNet(dim=6)
        self.time_net2 = TimeNet(dim=4)
        self.time_net1 = TimeNet(dim=2)

        self.down1 = nn.Sequential(nn.Linear(7, 6), nn.GELU(), nn.Linear(6, 6))
        self.down2 = nn.Sequential(nn.Linear(6, 4), nn.GELU(), nn.Linear(4, 4))
        self.down3 = nn.Sequential(nn.Linear(4, 2), nn.GELU(), nn.Linear(2, 2))

        self.up1 = nn.Sequential(nn.Linear(2 + 4, 4), nn.GELU(), nn.Linear(4, 4))
        self.up2 = nn.Sequential(nn.Linear(4 + 6, 6), nn.GELU(), nn.Linear(6, 6))
        self.up3 = nn.Sequential(nn.Linear(6 + 7, 7), nn.GELU(), nn.Linear(7, 7))

    def _fuse(self, c_s, t_s, g, time_s, cloud_net, text_net):
        if self.conditioning_mode == "hierarchical_fixed":
            return 0.5 * c_s + 0.5 * t_s

        cloud_logits = cloud_net(torch.cat((c_s, g, time_s, t_s), dim=1))
        text_logits = text_net(torch.cat((t_s, g, time_s, c_s), dim=1))
        weights = F.softmax(
            torch.stack((cloud_logits, text_logits), dim=1),
            dim=1,
        )
        # w_X + w_d = 1 for every channel.
        return c_s * weights[:, 0, :] + t_s * weights[:, 1, :]

    def forward(self, g, c, t, context_mask, _t):
        """
        Args:
            g: [B, 7] noisy pose.
            c: [B, 1024] geometry-only global point-cloud feature.
            t: [B, 512] affordance text feature.
            context_mask: [B, 1], same mask jointly drops c and t for CFG.
            _t: [B], normalized diffusion timestep t / T.
        """
        c = c * context_mask
        t = t * context_mask

        c0 = self.cloud_net0(c)
        c1, c2, c3 = self.cloud_net1(c0), self.cloud_net2(c0), self.cloud_net3(c0)

        t0 = self.text_net0(t)
        t1, t2, t3 = self.text_net1(t0), self.text_net2(t0), self.text_net3(t0)

        time_scalar = _t.unsqueeze(1)
        time1 = self.time_net1(time_scalar)
        time2 = self.time_net2(time_scalar)
        time3 = self.time_net3(time_scalar)

        g = g.float()
        g_down1 = self.down1(g)           # 6
        g_down2 = self.down2(g_down1)     # 4
        g_down3 = self.down3(g_down2)     # 2

        h1 = self._fuse(c1, t1, g, time1, self.cloud_influence_net1, self.text_influence_net1)
        up1 = self.up1(torch.cat((g_down3 * h1 + time1, g_down2), dim=1))

        h2 = self._fuse(c2, t2, g, time2, self.cloud_influence_net2, self.text_influence_net2)
        up2 = self.up2(torch.cat((up1 * h2 + time2, g_down1), dim=1))

        h3 = self._fuse(c3, t3, g, time3, self.cloud_influence_net3, self.text_influence_net3)
        up3 = self.up3(torch.cat((up2 * h3 + time3, g), dim=1))
        return up3
