import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import PointNetPlusPlus, PoseNet, TextEncoder


text_encoder = TextEncoder(device=torch.device('cuda'))


def linear_diffusion_schedule(betas, T):
    """Linear beta schedule indexed exactly by paper timesteps t=1,...,T.

    Index 0 is an identity sentinel. This avoids the training/sampling
    off-by-one mismatch in the stale code.
    """
    beta_t = torch.zeros(T + 1, dtype=torch.float32)
    beta_t[1:] = torch.linspace(betas[0], betas[1], T, dtype=torch.float32)

    alpha_t = 1.0 - beta_t
    alphabar_t = torch.cumprod(alpha_t, dim=0)

    sqrt_beta_t = torch.sqrt(beta_t)
    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(torch.clamp(1.0 - alphabar_t, min=0.0))
    oneover_sqrta = 1.0 / torch.sqrt(alpha_t)

    mab_over_sqrtmab = torch.zeros_like(beta_t)
    mab_over_sqrtmab[1:] = beta_t[1:] / torch.clamp(sqrtmab[1:], min=1e-12)

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }


class DetectionDiffusion(nn.Module):
    """AGPENet: point-level re-grounding + conditional 6-DoF pose diffusion."""

    def __init__(
        self,
        betas,
        n_T,
        device,
        background_text,
        drop_prob=0.1,
        use_point_attention=True,
        point_attn_dim=256,
        conditioning_mode="adaptive_hierarchical",
    ):
        super().__init__()
        self.posenet = PoseNet(conditioning_mode=conditioning_mode)
        self.pointnetplusplus = PointNetPlusPlus(
            use_point_attention=use_point_attention,
            point_attn_dim=point_attn_dim,
        )

        # Store log(gamma) so the learned inverse temperature is positive.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))

        for k, v in linear_diffusion_schedule(betas, n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.background_text = background_text
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    @staticmethod
    def _canonicalize_quaternion(q):
        """Normalize q and choose a unique sign using scalar-last [x,y,z,w]."""
        q = F.normalize(q, p=2, dim=-1, eps=1e-8)
        sign = torch.where(q[..., 3:4] < 0, -torch.ones_like(q[..., 3:4]), torch.ones_like(q[..., 3:4]))
        return q * sign

    @staticmethod
    def _project_quaternion_subvector(g):
        """Project the quaternion subvector of a 7-D pose back to S^3."""
        q = F.normalize(g[..., :4], p=2, dim=-1, eps=1e-8)
        return torch.cat((q, g[..., 4:]), dim=-1)

    @classmethod
    def _canonicalize_clean_pose(cls, g):
        return torch.cat((cls._canonicalize_quaternion(g[..., :4]), g[..., 4:]), dim=-1)

    def _encode_text_pair(self, text, batch_size):
        with torch.no_grad():
            foreground = text_encoder(text)
            background = text_encoder([self.background_text] * batch_size)
        return foreground, background

    def _affordance_logits(self, point_features, foreground_text_features, background_text_features):
        text_features = torch.stack((background_text_features, foreground_text_features), dim=1)  # [B,2,512]
        text_norm = F.normalize(text_features, p=2, dim=-1, eps=1e-8)
        point_norm = F.normalize(point_features, p=2, dim=1, eps=1e-8)
        gamma = self.logit_scale.exp().clamp(max=100.0)
        return gamma * torch.einsum('bkc,bcn->bkn', text_norm, point_norm)

    def forward(self, xyz, text, affordance_label, g):
        """Training forward pass implementing Eqs. (15)-(21)."""
        B = xyz.shape[0]
        foreground_text_features, background_text_features = self._encode_text_pair(text, B)

        # PointNet++ returns re-grounded point descriptors and geometry-only c_X.
        point_features, c_x = self.pointnetplusplus(xyz, foreground_text_features)

        logits = self._affordance_logits(
            point_features,
            foreground_text_features,
            background_text_features,
        )
        affordance_loss = F.cross_entropy(logits, affordance_label)

        # Paper: canonicalize quaternion signs before corruption.
        g0 = self._canonicalize_clean_pose(g)

        # Uniform t in {1,...,T}, with the same schedule index used at sampling.
        ts = torch.randint(1, self.n_T + 1, (B,), device=self.device)
        noise = torch.randn_like(g0)
        g_t = self.sqrtab[ts, None] * g0 + self.sqrtmab[ts, None] * noise

        # Paper: Euclidean corruption followed by projection of q back to S^3.
        g_t = self._project_quaternion_subvector(g_t)

        # Joint condition dropout: the same mask drops both cloud and text.
        context_mask = torch.bernoulli(
            torch.full((B, 1), 1.0 - self.drop_prob, device=self.device)
        )

        pred_noise = self.posenet(
            g_t,
            c_x,
            foreground_text_features,
            context_mask,
            ts.float() / self.n_T,
        )
        pose_loss = self.loss_mse(noise, pred_noise)
        return affordance_loss, pose_loss

    def detect_and_sample(self, xyz, text, n_sample, guide_w):
        """Predict point affordance labels and sample n_sample 6-DoF poses."""
        g_i = torch.randn(n_sample, 7, device=self.device)

        foreground_text_features, background_text_features = self._encode_text_pair(text, 1)
        point_features, c_x = self.pointnetplusplus(xyz, foreground_text_features)

        logits = self._affordance_logits(
            point_features,
            foreground_text_features,
            background_text_features,
        )
        affordance_prediction = torch.argmax(logits, dim=1)  # [1,N]

        c_i = c_x.repeat(n_sample, 1)
        t_i = foreground_text_features.repeat(n_sample, 1)
        context_mask = torch.ones((n_sample, 1), dtype=torch.float32, device=self.device)

        # Batch conditional and unconditional evaluations for classifier-free guidance.
        c_i = c_i.repeat(2, 1)
        t_i = t_i.repeat(2, 1)
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample:] = 0.0

        for i in range(self.n_T, 0, -1):
            time = torch.full(
                (2 * n_sample,),
                i / self.n_T,
                dtype=torch.float32,
                device=self.device,
            )
            g_batch = g_i.repeat(2, 1)

            eps_all = self.posenet(g_batch, c_i, t_i, context_mask, time)
            eps_cond = eps_all[:n_sample]
            eps_uncond = eps_all[n_sample:]
            eps_hat = (1.0 + guide_w) * eps_cond - guide_w * eps_uncond

            z = torch.randn(n_sample, 7, device=self.device) if i > 1 else torch.zeros(n_sample, 7, device=self.device)
            g_i = (
                self.oneover_sqrta[i]
                * (g_i - eps_hat * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            # Paper: renormalize the quaternion subvector after every reverse update.
            g_i = self._project_quaternion_subvector(g_i)

        # A unique output sign avoids q/-q ambiguity in downstream Euclidean checks.
        g_i = self._canonicalize_clean_pose(g_i)
        return affordance_prediction.cpu().numpy(), g_i.cpu().numpy()
