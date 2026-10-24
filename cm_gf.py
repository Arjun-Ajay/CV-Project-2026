"""
Cross-Modal Guided Filter (CM-GF) / Adaptive Guided Fusion Module (AGFM)

Implemented for Open-Vocabulary Semantic Segmentation refinement (SED CVPR 2024 extension).
Uses shallow visual features as guidance to sharpen blurry semantic cost maps via closed-form,
differentiable local linear transformations with O(N) box filtering.

Phase 2 Standalone Deliverables:
1. FastBoxFilter: Efficient nn.Module box filtering layer using F.avg_pool2d
2. GuidedFilterCore: Deterministic parameter-free closed-form guided filter
3. CMGFModule: Adaptive multi-mode guided filter with residual gating
4. AGFM: Composite Adaptive Guided Fusion Module for end-to-end skip connection fusion
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def fast_box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Computes 2D box filter (local window mean) with O(N) complexity using F.avg_pool2d.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        radius: Radius of the filter window. Window size will be 2 * radius + 1.
        
    Returns:
        Filtered tensor of shape (B, C, H, W) with preserved spatial dimensions.
    """
    kernel_size = 2 * radius + 1
    # count_include_pad=False ensures correct boundary averaging without dark border artifacts
    return F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
        count_include_pad=False
    )


class FastBoxFilter(nn.Module):
    """
    Box filter nn.Module wrapper around fast_box_filter.
    Enables parameterized placement in sequential networks.
    """

    def __init__(self, radius: int = 2):
        super().__init__()
        self.radius = radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fast_box_filter(x, self.radius)

    def extra_repr(self) -> str:
        return f"radius={self.radius}, window_size={2 * self.radius + 1}"


class GuidedFilterCore(nn.Module):
    """
    Deterministic, parameter-free closed-form Guided Filter.
    Computes:
        q_i = a_bar_i * I_i + b_bar_i
    where (a_k, b_k) are analytical regression coefficients minimizing:
        E(a_k, b_k) = sum_{i in w_k} ((a_k * I_i + b_k - P_i)^2 + eps * a_k^2)
    """

    def __init__(self, radius: int = 2, eps: float = 1e-3):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def forward(self, guidance: torch.Tensor, cost_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            guidance (I): Guidance feature map (B, C, H, W) or (B, 1, H, W)
            cost_map (P): Input cost / semantic probability map to be refined (B, C, H, W)
            
        Returns:
            Refined cost map (q) of shape (B, C, H, W) with aligned physical edges.
        """
        # 1. Compute local window statistics via O(N) box filtering
        mean_I = fast_box_filter(guidance, self.radius)
        mean_P = fast_box_filter(cost_map, self.radius)

        corr_I = fast_box_filter(guidance * guidance, self.radius)
        corr_IP = fast_box_filter(guidance * cost_map, self.radius)

        # 2. Local variance and cross-covariance with non-negative clamping
        var_I = corr_I - mean_I * mean_I
        var_I = torch.clamp(var_I, min=0.0)  # Numerical safeguard against tiny negative FP precision errors
        cov_IP = corr_IP - mean_I * mean_P

        # 3. Closed-form linear coefficients (a_k, b_k)
        # eps prevents division by zero in homogeneous/flat regions
        a = cov_IP / (var_I + self.eps)
        b = mean_P - a * mean_I

        # 4. Overlapping window averaging
        mean_a = fast_box_filter(a, self.radius)
        mean_b = fast_box_filter(b, self.radius)

        # 5. Linear reconstruction: q = a_bar * I + b_bar
        q = mean_a * guidance + mean_b
        return q

    def extra_repr(self) -> str:
        return f"radius={self.radius}, eps={self.eps}"


class CMGFModule(nn.Module):
    """
    Cross-Modal Guided Filter Module (CMGFModule) with channel projection and residual gating.
    
    Can handle different channel dimensions between guidance (shallow visual features)
    and cost map (upsampled semantic features), and incorporates a learnable residual gate
    so the network can gracefully interpolate between original features and filtered edges.
    """

    def __init__(
        self,
        guidance_channels: int,
        cost_map_channels: int,
        radius: int = 2,
        eps: float = 1e-3,
        mode: str = "channelwise",
        use_residual_gate: bool = True
    ):
        """
        Args:
            guidance_channels: Number of channels in guidance feature map I.
            cost_map_channels: Number of channels in cost map P.
            radius: Radius of the filtering window (kernel size = 2 * radius + 1).
            eps: Regularization parameter penalizing large linear coefficients.
            mode: 'channelwise' (full projection to cost_map_channels) or 
                  'spatial' (ultra-lightweight 1-channel edge guidance map, ideal for low GPU power).
            use_residual_gate: If True, uses learnable gamma to smoothly blend original and refined maps.
        """
        super().__init__()
        self.radius = radius
        self.eps = eps
        self.mode = mode
        self.use_residual_gate = use_residual_gate

        if mode == "channelwise":
            if guidance_channels != cost_map_channels:
                self.guidance_proj = nn.Sequential(
                    nn.Conv2d(guidance_channels, cost_map_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(cost_map_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                self.guidance_proj = nn.Identity()
        elif mode == "spatial":
            # Ultra-lightweight: compresses guidance into 1 spatial edge channel
            self.guidance_proj = nn.Sequential(
                nn.Conv2d(guidance_channels, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'channelwise' or 'spatial'.")

        self.filter_core = GuidedFilterCore(radius=radius, eps=eps)

        if self.use_residual_gate:
            # Learnable scalar gating parameter initialized to 0.5 (balanced prior)
            self.gamma = nn.Parameter(torch.tensor([0.5]))
        else:
            self.gamma = None

    def forward(self, guidance: torch.Tensor, cost_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            guidance (I): Shallow backbone features of shape (B, C_in_guide, H_g, W_g)
            cost_map (P): Upsampled semantic cost map of shape (B, C_in_cost, H_p, W_p)
            
        Returns:
            Refined feature map of shape (B, C_in_cost, H_p, W_p)
        """
        # Ensure spatial resolutions match
        if guidance.shape[-2:] != cost_map.shape[-2:]:
            guidance = F.interpolate(
                guidance,
                size=cost_map.shape[-2:],
                mode="bilinear",
                align_corners=True
            )

        # Project guidance to appropriate channel representation
        I_proj = self.guidance_proj(guidance)

        # Apply closed-form guided filtering
        q = self.filter_core(guidance=I_proj, cost_map=cost_map)

        # Residual gating: q_final = cost_map + gamma * (q - cost_map)
        if self.use_residual_gate:
            q_final = cost_map + self.gamma * (q - cost_map)
        else:
            q_final = q

        return q_final


class AGFM(nn.Module):
    """
    Adaptive Guided Fusion Module (AGFM).
    Combines Cross-Modal Guided Filter (CM-GF) with dual-branch residual feature stitching.
    Acts as a drop-in replacement for standard concatenation-convolution blocks in SED.
    """

    def __init__(
        self,
        in_cost_channels: int,
        guidance_channels: int,
        out_channels: int,
        radius: int = 2,
        eps: float = 1e-3,
        mode: str = "channelwise"
    ):
        super().__init__()
        self.cm_gf = CMGFModule(
            guidance_channels=guidance_channels,
            cost_map_channels=in_cost_channels,
            radius=radius,
            eps=eps,
            mode=mode,
            use_residual_gate=True
        )

        # Dual-branch fusion convolution (Pointwise + Depthwise-Separable for low FLOPs)
        total_in = in_cost_channels + guidance_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, cost_map: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_map: Upsampled deep feature map (B, C_cost, H, W)
            guidance: Shallow visual feature map (B, C_guide, H, W)
        Returns:
            Edge-sharpened fused representation (B, C_out, H, W)
        """
        refined_cost = self.cm_gf(guidance=guidance, cost_map=cost_map)
        if guidance.shape[-2:] != cost_map.shape[-2:]:
            guidance = F.interpolate(
                guidance,
                size=cost_map.shape[-2:],
                mode="bilinear",
                align_corners=True
            )
        fused = self.fusion(torch.cat([refined_cost, guidance], dim=1))
        return fused
