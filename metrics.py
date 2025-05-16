import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn, Tensor
from typing import Tuple 


    
def rand_proj(input_dim, num_proj, seed, requires_grad=False):
    """ Generates a normalized random projection matrix.  """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device='cuda') 
    generator.manual_seed(seed)
    proj = torch.randn([input_dim, num_proj], device=device, 
                        requires_grad=requires_grad, generator=generator)
    norm_constant = (torch.norm(proj, dim=-2, keepdim=True) + 1e-9) 
    return proj / norm_constant

def calc_quantile_diff(y_pred, y_real, proj_mat): 
    sorted_y_pred, _ = torch.sort(torch.matmul(y_pred, proj_mat), dim=-1) # torch.einsum('...e, en->...n'
    sorted_y_real, _ = torch.sort(torch.matmul(y_real, proj_mat), dim=-1)
    return torch.abs(sorted_y_pred - sorted_y_real) 
 
def calc_swd_distance(y_pred, y_real, proj_mat):
    diff = calc_quantile_diff(y_pred, y_real, proj_mat) 
    swd_distance = torch.dot(diff.view(-1), diff.view(-1)) / diff.numel()   
    return swd_distance


        
class SWDMetric(nn.Module):
    """ Compute the Sliced Wasserstein Distance between two sets of data. """ 
    def __init__(self, dim: int, num_proj: int, seed: int):
        super().__init__()
        self.seed = seed
        self.dim = dim
        self.num_proj = num_proj


    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> Tensor:
        proj_mat = rand_proj(self.dim, self.num_proj, 
            seed=self.seed,  requires_grad=False)
        y_pred = y_pred.permute(0, 2, 1)
        y_real = y_real.permute(0, 2, 1)
        diff = calc_quantile_diff(y_pred, y_real, proj_mat) 
        swd = torch.dot(diff.view(-1), diff.view(-1)) / diff.numel()  
        return swd


class TargetStdMetric(nn.Module):
    """Compute the standard deviation of target data."""
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        # y_real shape: [batch_size, seq_len, channels] or [batch_size, channels, seq_len]
        return torch.std(y_real, dim=-1).mean()  # Average across batch and channels

class EPTMetric(nn.Module):
    """
    Effective Prediction Time:
        • global_std: 1‑D tensor [D]  (per‑channel threshold)
        • y_pred, y_true: [B, S, D]
    Returns a scalar: mean T_{b,d} over all sequences and channels.
    """
    def __init__(self, global_std: torch.Tensor):
        super().__init__()
        
        # shape to [1, D, 1] so it broadcasts against [B, D, S]
        # print(f"EPT global_std shape: {global_std.shape}")
        self.register_buffer("thr", global_std.to(torch.float32)[None, :, None])

    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.permute(0, 2, 1)
        y_real = y_real.permute(0, 2, 1)
        # print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_real.shape}, thr shape: {self.thr.shape}")
        # print(f"self.thr: {self.thr}")
        err = (y_pred - y_real).abs()                 # [B,D,S]
        crossed = err > self.thr                      # [B,D,S] bool
        # first index along S where crossed is True; if never, returns 0
        first = (crossed.float()).argmax(-1)          # [B,D] int64
        # first index along S where value==1 (PyTorch argmax returns first max)
        never = (~crossed.any(-1))                    # [B,D] bool
        # If any time‑step is True, it returns True; otherwise False. → [B,D].
        first[never] = y_pred.size(-1)                # set to S when never crossed
        return first.float().mean()                   # scalar like MSE/MAE

class EPTHingeLoss(nn.Module):
    """
    Effective Prediction Time:
        • global_std: 1‑D tensor [D]  (per‑channel threshold)
        • y_pred, y_true: [B, S, D]
    Returns a scalar: mean T_{b,d} over all sequences and channels.
    """
    def __init__(self, global_std: torch.Tensor):
        super().__init__()
        
        # shape to [1, D, 1] so it broadcasts against [B, D, S]
        # print(f"EPT global_std shape: {global_std.shape}")
        self.register_buffer("thr", global_std.to(torch.float32)[None, :, None])

    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.permute(0, 2, 1)
        y_real = y_real.permute(0, 2, 1)
        # print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_real.shape}, thr shape: {self.thr.shape}")
        # print(f"self.thr: {self.thr}")
        S = y_pred.size(-1)
        err = (y_pred - y_real).abs().mean(dim=1)                # [B,S]
        sigma = float(self.thr.mean())  
        mask = torch.relu(err - sigma)   # [B, S]
        weights = torch.arange(1, S+1, device=err.device).float() / S
        L_ept = (mask * weights).mean()               # scalar
        return -L_ept                 # scalar like MSE/MAE

class EPTHingeLossSmooth(nn.Module):
    """
    Effective Prediction Time:
        • global_std: 1‑D tensor [D]  (per‑channel threshold)
        • y_pred, y_true: [B, S, D]
    Returns a scalar: mean T_{b,d} over all sequences and channels.
    """
    def __init__(self, global_std: torch.Tensor):
        super().__init__()
        
        # shape to [1, D, 1] so it broadcasts against [B, D, S]
        # print(f"EPT global_std shape: {global_std.shape}")
        self.register_buffer("thr", global_std.to(torch.float32)[None, :, None])
        self.alpha = 2.0

    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        """
        Soft EPT via smooth “first breach” estimator.
        Args:
        y_pred : [B, D, S] tensor of predictions
        y_true : [B, D, S] tensor of ground truth
        epsilon: scalar threshold
        alpha  : steepness for sigmoid
        Returns:
        T_soft : [B] tensor of soft breach times
        """
        # 1) per-step average MAE over channels → [B, S]
        err = torch.mean(torch.abs(y_pred - y_real), dim=1)

        # 2) soft exceedance signal s_t ∈ (0,1) → [B, S]
        epsilon = float(self.thr.mean())  
        s = torch.sigmoid(self.alpha * (err - epsilon))

        # 3) “probability” no breach before t → Π_{u<t}(1−s_u)
        #    we compute cumulative product along time dimension:
        no_breach_before = torch.cumprod(1.0 - s, dim=1)  
        #    at time t this is ∏_{u=0..t} (1−s[u])

        # 4) weight for “first breach at t”: s_t * (no breach before t−1)
        #    shift no_breach_before right by 1 to exclude current step:
        pad = torch.ones_like(no_breach_before[:, :1])
        alive = torch.cat([pad, no_breach_before[:, :-1]], dim=1)
        first_breach_prob = s * alive  # [B, S]

        # 5) expected breach time: sum t * p(first breach at t)
        #    t indices go 1…S
        S = err.size(1)
        idxs = torch.arange(1, S+1, device=err.device, dtype=err.dtype)
        T_soft = torch.sum(first_breach_prob * idxs, dim=1).mean()  # [B]

        return -T_soft