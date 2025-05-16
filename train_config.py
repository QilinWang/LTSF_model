import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass, field
from torch.nn import MSELoss, L1Loss, HuberLoss
from metrics import SWDMetric, TargetStdMetric 
from abc import ABC

class BaseTrainingConfig(ABC):
    """Abstract base for all training configs."""
    pass

@dataclass
class TransformPairConfig:
    """Configuration for scale-shift transform pairs used in CoefGenerator."""
    in_dim: int
    out_dim: int
    hid_dim: int
    channels: int

    activations_scale_shift: List[str] = field(default_factory=lambda: ["relu6", "dynamic_tanh"]) # nn.ReLU6, DynamicTanh
    spectral_flags_scale_shift: List[bool] = field(default_factory=lambda: [True, False]) 
    
    enable_magnitudes: List[bool] = field(default_factory=lambda: [False, True])
    spectral_flags_magnitudes: List[bool] = field(default_factory=lambda: [False, False])
    single_magnitude_for_shift: bool = False

    scale_zeroing_threshold: float = 0.0 # 5e-4 Threshold for stability in scale
    scale_clip_upper_bound: float = 0.0 #  10        # Upper bound for scale

    # hidden layer activation function
    activations_hidden_layers: List[str] = field(default_factory=lambda: [nn.ELU, nn.LogSigmoid])
    spectral_flags_hidden_layers: List[bool] = field(default_factory=lambda: [False, False])

@dataclass(kw_only=True)
class FlatACLConfig(BaseTrainingConfig):
    """FLAT configuration for ACL model. Contains ALL base and specific fields."""
    # --- Meta ---
    model_type: str = field(default='ACL', init=False) # Type identifier

    # --- Essential Identifiers (from BaseTrainingConfig) ---
    seq_len: int
    pred_len: int
    channels: int
    batch_size: int

    # --- Core Training Hyperparameters (from BaseTrainingConfig) ---
    learning_rate: float
    epochs: int = 50
    patience: int = 5
    device: str = 'cuda'
    num_proj_swd: int = 500
    seeds: Optional[List[int]] = None # Set per run by MultiSeedExperiment

    # --- Losses (from BaseTrainingConfig) ---
    losses : List[str] = field(default_factory=lambda: ['mse', 'mae', 'huber', 'swd', 'ept'])
    loss_backward_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])
    loss_validate_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])

    # --- ACL Specific Fields ---
    dim_augment: int
    dim_hidden: int 
    x_to_z_delay: Optional[TransformPairConfig] = None # Will be set in __post_init__
    x_to_z_deri: Optional[TransformPairConfig] = None # Will be set in __post_init__
    z_to_x_main: Optional[TransformPairConfig] = None # Will be set in __post_init__
    mixing_strategy: str = "convex" # delay_only, deriv_only
    z_to_y_main: Optional[TransformPairConfig] = None # Will be set in __post_init__
    z_push_to_z: Optional[TransformPairConfig] = None # Will be set in __post_init__
    
    householder_reflects_latent: int = 4
    householder_reflects_data: int = 8
    second_delay_use_shift: bool = True

    # Ablations
    ablate_no_koopman: bool = False
    ablate_no_rotation: bool = False
    use_complex_eigenvalues: bool = True
    single_magnitude_for_shift: bool = False
    ablate_rotate_back_Koopman: bool = False
    ablate_shift_only: bool = False
    ablate_scale_only: bool = False
    ablate_shift_inside_scale: bool = False
    ablate_cascaded_model:bool = False
    noise_y0: bool = False 
    ablate_single_encoding_layer: bool = False
    ablate_deterministic_y0: bool = False
    ablate_no_shift_in_z_push: bool = False

    # --- EPT ---
    ept_global_std: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Sets up the TransformPairConfig instances after initialization."""
        # Note: All required fields (seq_len, pred_len, channels, dim_augment, dim_hidden)
        # are now direct attributes of `self`.

        if self.x_to_z_delay is None:
            self.x_to_z_delay = TransformPairConfig(
                in_dim=self.seq_len,
                out_dim=self.dim_augment,
                hid_dim=self.dim_hidden,
                channels = self.channels,
                scale_zeroing_threshold=0, # 1e-4,
                scale_clip_upper_bound=10.0, # CORRECTED
                single_magnitude_for_shift=self.single_magnitude_for_shift,
            )
        if self.x_to_z_deri is None:
            self.x_to_z_deri = TransformPairConfig(
                in_dim=self.seq_len,
                out_dim=self.dim_augment,
                hid_dim=self.dim_hidden,
                channels = self.channels,
                scale_zeroing_threshold=0, # 1e-4,
                scale_clip_upper_bound=10.0, # CORRECTED
                single_magnitude_for_shift=self.single_magnitude_for_shift,
            )
        if self.z_to_x_main is None:
            self.z_to_x_main = TransformPairConfig(
                in_dim=self.dim_augment,
                out_dim=self.seq_len,
                hid_dim=self.dim_hidden,
                channels = self.channels,
                scale_zeroing_threshold=0, # 1e-4,
                scale_clip_upper_bound=10.0, # CORRECTED
                single_magnitude_for_shift=self.single_magnitude_for_shift,
            )
        if self.z_to_y_main is None:
            self.z_to_y_main = TransformPairConfig(
                in_dim=self.dim_augment,
                out_dim=self.pred_len,
                hid_dim=self.dim_hidden,
                channels = self.channels,
                scale_zeroing_threshold = 0,
                scale_clip_upper_bound  = 10.0, # CORRECTED
                single_magnitude_for_shift=self.single_magnitude_for_shift,
            )
        if self.z_push_to_z is None:
            self.z_push_to_z = TransformPairConfig(
                in_dim=self.dim_augment,
                out_dim=self.dim_augment,
                hid_dim=self.dim_hidden,
                channels=self.channels,
                scale_zeroing_threshold = 0,
                scale_clip_upper_bound  = 10.0, # CORRECTED
                single_magnitude_for_shift=self.single_magnitude_for_shift,
            )
         
            
@dataclass(kw_only=True)
class FlatPatchTSTConfig(BaseTrainingConfig):
    """FLAT configuration for PatchTST model."""
    # --- Meta ---
    model_type: str = field(default='PatchTST', init=False)

    # --- Essential Identifiers (from BaseTrainingConfig) ---
    seq_len: int
    pred_len: int
    channels: int
    batch_size: int
    enc_in: int  
    dec_in: int 
    c_out: int 
    label_len: int = 0

    # --- Core Training Hyperparameters (from BaseTrainingConfig) ---
    learning_rate: float
    epochs: int = 50
    patience: int = 5
    device: str = 'cuda'
    num_proj_swd: int = 500
    seeds: Optional[List[int]] = None
    task_name: str = 'long_term_forecast'

    # --- Losses (from BaseTrainingConfig) ---
    losses : List[str] = field(default_factory=lambda: ['mse', 'mae', 'huber', 'swd', 'ept'])
    loss_backward_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])
    loss_validate_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])

    # --- PatchTST Specific Fields ---
    d_model: int = 128
    e_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    activation: str = 'gelu'
    patch_len: int = 16
    stride: int = 8
    factor: int = 3

    # --- EPT ---
    ept_global_std: Optional[torch.Tensor] = None

@dataclass(kw_only=True)
class FlatDLinearConfig(BaseTrainingConfig):
    """FLAT configuration for DLinear model."""
    # --- Meta ---
    model_type: str = field(default='DLinear', init=False)

    # --- Essential Identifiers (from BaseTrainingConfig) ---
    seq_len: int
    pred_len: int
    channels: int
    batch_size: int


    # --- Core Training Hyperparameters (from BaseTrainingConfig) ---
    learning_rate: float
    epochs: int = 50
    patience: int = 5
    device: str = 'cuda'
    num_proj_swd: int = 500
    seeds: Optional[List[int]] = None

    # --- Losses (from BaseTrainingConfig) ---
    losses : List[str] = field(default_factory=lambda: ['mse', 'mae', 'huber', 'swd', 'ept'])
    loss_backward_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])
    loss_validate_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])

    # --- DLinear Specific Fields ---
    individual: bool = True

    # --- EPT ---
    ept_global_std: Optional[torch.Tensor] = None

@dataclass(kw_only=True)
class FlatTimeMixerConfig(BaseTrainingConfig):
    """FLAT configuration for TimeMixer model."""
    # --- Meta ---
    model_type: str = field(default='TimeMixer', init=False)

    # --- Essential Identifiers (from BaseTrainingConfig) ---
    seq_len: int
    pred_len: int
    channels: int
    batch_size: int
    enc_in: int  
    dec_in: int 
    c_out: int 
    label_len: int = 0

    # --- Core Training Hyperparameters (from BaseTrainingConfig) ---
    learning_rate: float
    epochs: int = 50
    patience: int = 5
    device: str = 'cuda'
    num_proj_swd: int = 500
    seeds: Optional[List[int]] = None
    task_name: str = 'long_term_forecast'
    embed: Optional[str] = None # must keep None
    freq: Optional[str] = None # must keep None
    use_norm: Optional[bool] = None # must keep None
    # --- Losses (from BaseTrainingConfig) ---
    losses : List[str] = field(default_factory=lambda: ['mse', 'mae', 'huber', 'swd', 'ept'])
    loss_backward_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])
    loss_validate_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])

    # --- TimeMixer Specific Fields ---
    channel_independence: bool = True
    e_layers: int = 2
    down_sampling_layers: int = 3
    down_sampling_window: int = 2
    d_model: int = 16
    d_ff: int = 32
    dropout: float = 0.1
    decomp_method: str = 'moving_avg'
    moving_avg: int = 25
    down_sampling_method: str = 'avg'
    # --- EPT ---
    ept_global_std: Optional[torch.Tensor] = None

@dataclass(kw_only=True)
class FlatNaiveConfig(BaseTrainingConfig):
    """FLAT configuration for Naive baseline model."""
    # --- Meta ---
    model_type: str = field(default='naive', init=False)

    # --- Essential Identifiers (from BaseTrainingConfig) ---
    seq_len: int
    pred_len: int
    channels: int
    batch_size: int

    # --- Core Training Hyperparameters (from BaseTrainingConfig) ---
    learning_rate: float
    epochs: int = 50
    patience: int = 5
    device: str = 'cuda'
    num_proj_swd: int = 500
    seeds: Optional[List[int]] = None

    # --- Losses (from BaseTrainingConfig) ---
    losses : List[str] = field(default_factory=lambda: ['mse', 'mae', 'huber', 'swd', 'ept'])
    loss_backward_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])
    loss_validate_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0])

    # --- Naive Specific Fields ---
    # None needed for repeat-last-value baseline

    # --- EPT ---
    ept_global_std: Optional[torch.Tensor] = None




    
    

    

    


    
