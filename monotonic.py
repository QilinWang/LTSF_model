import torch
from abc import ABC, abstractmethod  
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Callable, Dict
from torch import Tensor  
import torch.nn.utils.parametrizations as parametrizations
import torch.nn.utils.parametrize as parametrize
from enum import Enum, auto
from dataclasses import dataclass, field
from metrics import rand_proj
import einops
from einops import rearrange
from typing import Optional
import re
import torch.nn.utils.spectral_norm as spectral_norm
from torch.cuda.amp import autocast
from train_config import TransformPairConfig, FlatACLConfig

class DynamicTanh(nn.Module):
    def __init__(self, dim: int, channels: int):
        super().__init__() 
        self.dim = dim 
        self.a = nn.Parameter(torch.ones(channels, self.dim))
        self.c = nn.Parameter(torch.ones(channels, self.dim))
        self.d = nn.Parameter(torch.zeros(channels, self.dim))   
        # self.b = nn.Parameter(torch.zeros(channels,self.dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.a * x) * self.c + self.d

class CoefGenerator(nn.Module): 
    """Enhanced coefficient generator with flexible transform pairs."""
    def __init__(self, transform_config: TransformPairConfig, top_config: FlatACLConfig) -> None:
        super().__init__()   
        self.transform_cfg  = transform_config
        self.top_cfg = top_config

        self._build_hidden_network()
        self._build_param_networks()
         
    def _build_hidden_network(self):
        self.hidden_in_net =  nn.Linear(self.transform_cfg.in_dim, self.transform_cfg.hid_dim) 
        layers = [] 
        for i, (activation, use_spec_norm) in enumerate(zip(self.transform_cfg.activations_hidden_layers, 
                                                            self.transform_cfg.spectral_flags_hidden_layers)):
            linear = nn.Linear(self.transform_cfg.hid_dim, self.transform_cfg.hid_dim, bias=True)
            if use_spec_norm:
                linear = spectral_norm(linear)

            if isinstance(activation, nn.Module):
                act_mod = activation       # already built
            elif callable(activation):
                act_mod = activation()     # build it now
            else:
                raise TypeError(f"Bad activation entry: {activation!r}")
            # if isinstance(activation, DynamicTanh):
            #     layers.extend([linear, activation])
            # else:
            #     layers.extend([linear, activation()])
            layers.extend([linear, act_mod])
            # print(f"activation: {activation}")
        self.hidden_main_net  = nn.Sequential(*layers)

    def _build_output_head(self, in_dim, out_dim, activation_cls, sn_flag, channels):
        acti_map = {
            "identity": nn.Identity,
            "relu": nn.ReLU,
            "relu6": nn.ReLU6,
            "tanh": nn.Tanh,
            "logsigmoid": nn.LogSigmoid,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "mish": nn.Mish,
            "dynamic_tanh": DynamicTanh,
            "leaky_relu": nn.LeakyReLU,
        }
        layer = nn.Linear(in_dim, out_dim, bias=True)
        if sn_flag:
            layer = spectral_norm(layer)

        
        if channels is None:
            channels = self.transform_cfg.channels
            assert channels is not None, "You must set cfg.channels before building the networks."
        if activation_cls == "dynamic_tanh":
            activation = DynamicTanh(out_dim, channels=channels)
        elif activation_cls in acti_map:
            activation = acti_map[activation_cls]()
        else:
            raise ValueError(f"Invalid activation function: {activation_cls}")
        
        return nn.Sequential(layer, activation)
    
    def _build_magnitude_block(self, use_flag):
        if not self.transform_cfg.single_magnitude_for_shift:
            layer = nn.Linear(self.transform_cfg.in_dim, self.transform_cfg.out_dim, bias=True)
        else:
            layer = nn.Linear(self.transform_cfg.in_dim, 1, bias=True)
        if self.transform_cfg.spectral_flags_magnitudes:
            layer = spectral_norm(layer)
        return nn.Sequential(layer, nn.ReLU6()) if use_flag else None
          

    def _build_param_networks(self):
        acts = self.transform_cfg.activations_scale_shift
        specs = self.transform_cfg.spectral_flags_scale_shift 

        # assert len(acts) == len(specs) == len(biases), "All lists must have the same length"
        self.scale_net = self._build_output_head(self.transform_cfg.hid_dim, self.transform_cfg.out_dim, acts[0], 
            specs[0],   self.transform_cfg.channels)
        self.shift_net = self._build_output_head(self.transform_cfg.hid_dim, self.transform_cfg.out_dim, acts[1], 
            specs[1],   self.transform_cfg.channels)
        
        self.scale_mag = self._build_magnitude_block(self.transform_cfg.enable_magnitudes[0])
        self.shift_mag = self._build_magnitude_block(self.transform_cfg.enable_magnitudes[1])

    def forward(self, x: torch.Tensor, update: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the coefficient generator.
        Args:
            x: Input tensor
            update: Whether to compute gradients
        Returns:
            Dictionary containing h (hidden representation) and all scales/shifts
        """
        with torch.set_grad_enabled(update):   
            h = self.hidden_in_net(x)  
            h = self.hidden_main_net(h)

            raw_scale = self.scale_net(h)
            raw_shift = self.shift_net(h)

            scale_mag = self.scale_mag(x) if self.scale_mag else None
            shift_mag = self.shift_mag(x) if self.shift_mag else None
            
            scale = raw_scale * scale_mag if self.scale_mag else raw_scale
            shift = raw_shift * shift_mag if self.shift_mag else raw_shift
            
            # scale = F.relu6(scale)
            if self.transform_cfg.scale_zeroing_threshold > 0:
                scale = F.threshold(scale, self.transform_cfg.scale_zeroing_threshold, 0)
            if self.transform_cfg.scale_clip_upper_bound > 0:
                scale = torch.clamp(scale, max=self.transform_cfg.scale_clip_upper_bound)
            
            # scale = scale + 1e-5
            
        return CoefOutput(scale=scale, shift=shift, 
                          scale_mag=scale_mag, shift_mag=shift_mag, h=h,
                          raw_scale=raw_scale, raw_shift=raw_shift
            )

     
class ACL(nn.Module):
    def __init__(self, cfg: FlatACLConfig): # FullACLConfig
        super().__init__() 
        self.cfg = cfg 
        self.encoder = XZCouplingStage(self.cfg)
        self.decoder = ZtoYStage(self.cfg)
        
    def forward(self, x_bsd: torch.Tensor, update: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_bsd: Input tensor of shape [B, D, S]
            update: Whether to compute gradients
        Returns:
            Dictionary containing all intermediate and final results
        """
        x_bds = x_bsd.permute(0, 2, 1)
        FullCouplingResults = self.encoder(x_bds, update=update )   
        ZtoYResults = self.decoder(x_bds, FullCouplingResults, update=update ) 
        y_bsd_pred = ZtoYResults.pred.permute(0, 2, 1)
        # print(f"Type of object causing error: {type(ZtoYResults)}")
        # print(f"Type of object causing error: {type(FullCouplingResults)}")
        # print(f"""
        #       FullCouplingResults.x_to_z_results_1.mixed_coefs.scale: {FullCouplingResults.x_to_z_results_1.mixed_coefs.scale[0,0,0:3]}
        # FullCouplingResults.x_to_z_results_1.mixed_coefs.shift: {FullCouplingResults.x_to_z_results_1.mixed_coefs.shift[0,0,0:3]}
        # FullCouplingResults.x_to_z_results_1.mixed_coefs.h: {FullCouplingResults.x_to_z_results_1.mixed_coefs.h[0,0,0:3]}
        # """)
        # print(f"""
        # FullCouplingResults.x_to_z_results_1.mixed_coefs.scale: {FullCouplingResults.x_to_z_results_1.mixed_coefs.scale[0,0,0:3]}
        # ZtoYResults.coefs.scale: {ZtoYResults.coefs.scale[0,0,0:3]}
        # ZtoYResults.coefs.shift: {ZtoYResults.coefs.shift[0,0,0:3]}
        # ZtoYResults.coefs.h: {ZtoYResults.coefs.h[0,0,0:3]}
        # ZtoYResults.pred: {ZtoYResults.pred[0,0,0:3]}
        # ZtoYResults.coefs.scale_mag: {ZtoYResults.coefs.scale_mag[0,0,0:3]}
        # ZtoYResults.coefs.shift_mag: {ZtoYResults.coefs.shift_mag[0,0,0:3]}
        # ZtoYResults.coefs.raw_scale: {ZtoYResults.coefs.raw_scale[0,0,0:3]}
        # ZtoYResults.coefs.raw_shift: {ZtoYResults.coefs.raw_shift[0,0,0:3]}
        # """)
        return {
            "encoder": FullCouplingResults,
            "decoder": ZtoYResults,
            "pred": y_bsd_pred,
        }

@dataclass
class CoefOutput:
    scale: torch.Tensor # Shape [B, D, S]
    shift: torch.Tensor # Shape [B, D, S]
    h: torch.Tensor # Latent Representation Shape [B, D, S]
    scale_mag: Optional[torch.Tensor] = None
    shift_mag: Optional[torch.Tensor] = None
    raw_scale: Optional[torch.Tensor] = None
    raw_shift: Optional[torch.Tensor] = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    def as_dict(self):
        return {"scale_coef": self.scale, "shift": self.shift, "scale_mag": self.scale_mag, "shift_mag": self.shift_mag, "h": self.h}

def apply_affine_transform(data: torch.Tensor, coef_output: CoefOutput):
    scale = coef_output.scale
    shift = coef_output.shift
    return data * scale + shift

def apply_convex_combination(input_1: torch.Tensor, input_2: torch.Tensor, weight: torch.Tensor):
    return weight * input_1 + (1 - weight) * input_2

@dataclass
class XtoZResults:
    z_transformed: torch.Tensor
    mixed_coefs: CoefOutput
    weight: torch.Tensor
    delay_coefs: Optional[CoefOutput] = None
    deriv_coefs: Optional[CoefOutput] = None

@dataclass
class ZtoXResults:
    data: torch.Tensor
    coefs: CoefOutput

@dataclass
class FullCouplingResults:
    x_to_z_results_1: XtoZResults
    x_to_z_results_2: XtoZResults
    z_to_x_results: ZtoXResults


class XZCouplingStage(nn.Module):
    """
    Implemented x->z->x->z, a three step ANF style affine coupling block encoding information of x into z.
    """
    def __init__(self, cfg: FlatACLConfig):
        super().__init__() 
        self.cfg = cfg

        self.x_to_z_delay = CoefGenerator(transform_config=self.cfg.x_to_z_delay, top_config=self.cfg)
        self.x_to_z_deri = CoefGenerator(transform_config=self.cfg.x_to_z_deri, top_config=self.cfg)
        self.z_to_x_main = CoefGenerator(transform_config=self.cfg.z_to_x_main, top_config=self.cfg)
 
        self.mixing_network = nn.Sequential(
            # nn.Linear(cfg.seq_len, cfg.seq_len, bias=True),
            # nn.GELU(), 
            nn.Linear(cfg.seq_len, 1, bias=True), 
            nn.Sigmoid(),  
        ) 
    def encode_x_to_z (self, data: torch.Tensor, aug: torch.Tensor, update: bool = True)->XtoZResults:
        if not self.cfg.mixing_strategy=="deriv_only":
            delay_coefs = self.x_to_z_delay(data, update=update)
        else:
            delay_coefs = CoefOutput(scale=torch.ones_like(aug), 
                shift=torch.zeros_like(aug), h=torch.zeros_like(aug), 
                scale_mag=None, shift_mag=None)
        
        if not self.cfg.mixing_strategy=="delay_only":
            if self.cfg.second_delay_use_shift:
                deri_coefs = self.x_to_z_deri(shift_embed(data, order=1) , update=update) 
            else:
                deri_coefs = self.x_to_z_deri(deri_embed(data, order=1), update=update) 
        else:
            deri_coefs = CoefOutput(scale=torch.zeros_like(aug), 
                shift=torch.zeros_like(aug), h=torch.zeros_like(aug), 
                scale_mag=None, shift_mag=None)
        
        if self.cfg.mixing_strategy == "convex":
            weight = self.mixing_network(data) 
            scale = apply_convex_combination(delay_coefs.scale, deri_coefs.scale, weight)
            shift = apply_convex_combination(delay_coefs.shift, deri_coefs.shift, weight)
            h = apply_convex_combination(delay_coefs.h, deri_coefs.h, weight)
        elif self.cfg.mixing_strategy == "delay_only":
            weight = torch.ones_like(data[..., :1])
            scale = delay_coefs.scale
            shift = delay_coefs.shift
            h = delay_coefs.h
        elif self.cfg.mixing_strategy == "deriv_only":
            weight = torch.zeros_like(data[..., :1])
            scale = deri_coefs.scale
            shift = deri_coefs.shift
            h = deri_coefs.h
        else: 
            raise ValueError(f"Invalid mixing strategy: {self.cfg.mixing_strategy}")
        
        aug_transformed = aug * scale + shift
        
        return XtoZResults(z_transformed=aug_transformed, 
                             mixed_coefs=CoefOutput(scale=scale, shift=shift, h=h, 
                                scale_mag=None, shift_mag=None), 
                             weight=weight, 
                             delay_coefs=delay_coefs, 
                             deriv_coefs=deri_coefs)
    
    def update_x_with_z (self, data, aug_tranformed: torch.Tensor, update: bool = True):
        decode_coefs = self.z_to_x_main(aug_tranformed, update=update)
        updated_data  = apply_affine_transform(data, decode_coefs)
        return ZtoXResults(data=updated_data, coefs=decode_coefs)
    
    def forward(self, x: torch.Tensor, update: bool = True) -> Dict[str, torch.Tensor]:
        """ 
        Args:
            data: Input tensor
            update: Whether to compute gradients
        Returns:
            Dictionary containing encoded data and intermediate results
        """
        z_shape = list(x.shape[:-1]) + [self.cfg.dim_augment]
        with torch.set_grad_enabled(update):
            z_01 = torch.randn(z_shape, device=x.device, dtype=x.dtype)
            z_02 = torch.randn(z_shape, device=x.device, dtype=x.dtype)
            
            
            
            if not self.cfg.ablate_single_encoding_layer:
                x_to_z_result_1 = self.encode_x_to_z(x, z_01, update=update)
                z_to_x_result  = self.update_x_with_z(x, 
                    x_to_z_result_1.z_transformed, update=update)
                x_to_z_result_2 = self.encode_x_to_z(z_to_x_result.data, z_02, update=update)
            else:
                x_to_z_result_1 = torch.zeros_like(x)
                z_to_x_result = torch.zeros_like(x)
                x_to_z_result_2 = self.encode_x_to_z(x, z_02, update=update)
        return FullCouplingResults(
            x_to_z_results_1=x_to_z_result_1,
            z_to_x_results=z_to_x_result,
            x_to_z_results_2=x_to_z_result_2
        )

def apply_block_diagonal(x, a, b):
    """
    Apply a block-diagonal transformation with 2×2 blocks to a vector.
    
    Args:
        x: Input tensor of shape [..., dim]
        a: Scaling parameters of shape [..., dim//2]
        b: Rotation parameters of shape [..., dim//2]
        
    Returns:
        Transformed tensor of same shape as x
    """
    dim = x.shape[-1]
    assert dim % 2 == 0, "Dimension must be even for 2×2 blocks" # Ensure even dimension
    
    # For example, if x has shape [32, 128], it becomes [32, 64, 2], 64 is the number of 2×2 blocks, 2 is the number of elements per block
    # x_reshaped = x.view(*batch_shape, dim//2, 2)
    x_reshaped = einops.rearrange(x, 'b d (sn e) -> b d sn e', sn=dim//2, e=2) # Reshape tensor to group adjacent pairs [..., dim//2, 2]
    x_even = x_reshaped[..., 0]  # [..., dim//2]
    x_odd = x_reshaped[..., 1]   # [..., dim//2]
    
    # Apply block-diagonal transformation
    # [a  b] [x_even]
    # [-b a] [x_odd ]
    # print(f"shapes: {x_even.shape}, {x_odd.shape}, {a.shape}, {b.shape}")
    y_even = a * x_even + b * x_odd       # [..., dim//2]
    y_odd = -b * x_even + a * x_odd       # [..., dim//2]
    
    return einops.rearrange(torch.stack([y_even, y_odd], dim=-1), 'b d sn e -> b d (sn e)')
    # y = torch.stack([y_even, y_odd], dim=-1)  # [..., dim//2, 2]
    # return y.view(*batch_shape, dim)

class Rotation(nn.Module):
    """ Data-dependent rotation matrix via Householder reflections """
    def __init__(self, num_reflects: int, 
            in_dim: int, out_dim: int, cfg: FlatACLConfig):
        super().__init__() 
        self.cfg = cfg
        self.num_reflects = num_reflects       
        self.direction_generating_nets  = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False),
                DynamicTanh(out_dim, channels=self.cfg.channels),
            ) for i in range(num_reflects)] )   
    
    def apply_reflection(self, normed_reflect_vec: torch.Tensor, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Apply single Householder reflection: H = I - 2vv^T."""   
        with torch.set_grad_enabled(update):
            dot_prod = torch.sum(normed_reflect_vec * x, dim=-1, keepdim=True) # (v/||v||)ᵀx ; [B,D,1]
        return x - 2 * normed_reflect_vec * dot_prod # x - 2(v/||v||)((v/||v||)ᵀx) ; [B,D,S]
    
    def generate_rotation_basis(self, source: torch.Tensor) -> torch.Tensor:
        reflect_vectors = []  # data_copy = data.clone() # data_proj = self.proj(source)
        for i, net in enumerate(self.direction_generating_nets ):    
            reflect_vector = net(source)  
            normed_reflect_vec = F.normalize(reflect_vector, p=2, dim=-1) 
            reflect_vectors.append(normed_reflect_vec)
        return reflect_vectors
    
    def rotate(self, data: torch.Tensor, reflect_vectors: List[torch.Tensor], update: bool = True) -> torch.Tensor: # apply eg H5(H4(H3(H2(H1(x))))) 
        """Apply sequence of Householder reflections H_n(...H_2(H_1(x)))."""   
        with torch.set_grad_enabled(update):
            for i, normed_reflect_vec in enumerate(reflect_vectors):   
                data = self.apply_reflection(normed_reflect_vec=normed_reflect_vec, x=data, update=update)  
        return data 
    
    def rotate_back(self, input: torch.Tensor, reflect_vectors: List[torch.Tensor], update: bool = True) -> torch.Tensor: # Reverse order of Householder reflections: H1(H2(...Hn(x))))
        """Apply reverse sequence H_1(H_2(...H_n(x)))."""  
        for i, normed_reflect_vec in enumerate(reversed(reflect_vectors)):  # Reversed Householder vectors   
            input = self.apply_reflection(normed_reflect_vec=normed_reflect_vec, x=input, update=update)  # Negate each vi for the inverse
        return input 
    


class ZtoYStage(nn.Module):
    """
    Decoder component of the Affine Coupling Layer.
    Transforms augmented space back to data space.
    """
    def __init__(self, cfg: FlatACLConfig):
        super().__init__()  
        self.cfg = cfg 
        self.z_to_y_main = CoefGenerator(transform_config=cfg.z_to_y_main, top_config=cfg)
        self.z_push_to_z = CoefGenerator(transform_config=cfg.z_push_to_z, top_config=cfg)
        if cfg.ablate_cascaded_model:
            self.z_to_y_main_2 = CoefGenerator(transform_config=cfg.z_to_y_main, top_config=cfg)
             
        self.rotation_manager = Rotation(
            num_reflects=cfg.householder_reflects_data, 
            in_dim=cfg.dim_hidden, out_dim=cfg.pred_len, cfg=cfg
        )  

        self.rotation_manager_aug = Rotation(
            num_reflects=cfg.householder_reflects_latent, 
            in_dim=cfg.seq_len, out_dim=cfg.dim_augment, cfg=cfg
        )  
        if not self.cfg.ablate_deterministic_y0:
            self.latent_to_y0_map = nn.Sequential(
                nn.Linear(cfg.dim_augment, cfg.pred_len, bias=True),
                nn.Mish(), 
                nn.Linear(cfg.pred_len, cfg.pred_len, bias=True)
            ) 
        else:
            self.latent_to_y0_map = nn.Sequential(
                nn.Linear(cfg.seq_len, cfg.seq_len, bias=True),
                nn.Mish(), 
                nn.Linear(cfg.seq_len, cfg.pred_len, bias=True)
            ) 

        self.fixed_real_scale = nn.Parameter(torch.ones(1, cfg.dim_augment))
        self.fixed_shift = nn.Parameter(torch.zeros(1, cfg.dim_augment))

        # NEW: Parameters for complex eigenvalues in block-diagonal structure
        self.fixed_complex_a = nn.Parameter(torch.ones(cfg.dim_augment // 2)) # real part (scaling/damping) 
        self.fixed_complex_b = nn.Parameter(torch.zeros(cfg.dim_augment // 2)) # imaginary part (rotation/oscillation)
         

    def forward(self, data: torch.Tensor, FullCouplingResults: FullCouplingResults, 
                update: bool = True) -> Dict[str, torch.Tensor]:
        """ 
        Args:
            data: Original input tensor
            enc_rec: Encoded data from encoder
            update: Whether to compute gradients
        Returns:
            Dictionary containing decoded data and intermediate results
        """
        with torch.set_grad_enabled(update):
            z = FullCouplingResults.x_to_z_results_2.z_transformed # The normally distributed latent 

            #---- Latent side: Eigenfunction: Householder rotation of z, i.e. x |-> Q(x)z(x) -----
            coefs_push = self.z_push_to_z(z, update=update)

            if not self.cfg.ablate_no_koopman:
                reflect_vectors_for_z = self.rotation_manager_aug.generate_rotation_basis(data)
                
                z = self.rotation_manager_aug.rotate(
                    z, #+ coefs_push.shift, 
                    reflect_vectors_for_z
                )
                # #----- Fixed operation as Koopman operator pushing on Qz -----
                if not self.cfg.use_complex_eigenvalues:
                    z = self.fixed_real_scale * z 
                else: # complex eigenvalues
                    z = apply_block_diagonal(z, self.fixed_complex_a, self.fixed_complex_b) 
                if not self.cfg.ablate_rotate_back_Koopman:
                    z = z + coefs_push.shift # No need to rotate back
                else:
                    if not self.cfg.ablate_no_shift_in_z_push:
                        z = self.rotation_manager_aug.rotate_back(
                            z,  reflect_vectors_for_z ) + coefs_push.shift
                    else:
                        z = self.rotation_manager_aug.rotate_back(
                            z,  reflect_vectors_for_z ) 
            
            else:
                if not self.cfg.ablate_no_shift_in_z_push:
                    z = z + coefs_push.shift
                else:
                    z = z 
                reflect_vectors_for_z = None

            #---- Data side: EigenACL as Optiaml Transport -----
            y_size = list(z.shape[:-1]) + [self.cfg.pred_len]

            coefs_decode = self.z_to_y_main(z, update=update)
            if self.cfg.ablate_cascaded_model:
                coefs_decode_2 = self.z_to_y_main_2(z, update=update)
             

            
            if self.cfg.noise_y0:
                y0 = torch.randn(y_size, device=z.device, dtype=z.dtype) 
            elif self.cfg.ablate_deterministic_y0:
                y0 = self.latent_to_y0_map(data)
            else:
                y0 = self.latent_to_y0_map(z)
           

            if not self.cfg.ablate_no_rotation:
                reflect_vectors_for_y0 = self.rotation_manager.generate_rotation_basis(coefs_decode.h)
                Qx = self.rotation_manager.rotate(data=y0, reflect_vectors=reflect_vectors_for_y0)
                accumulator_vec = Qx

                if self.cfg.ablate_cascaded_model:
                    accumulator_vec = (
                        Qx + 
                        accumulator_vec * coefs_decode_2.scale + 
                        coefs_decode_2.shift
                    )
                
                if self.cfg.ablate_shift_only:
                    y_pred = y0 + coefs_decode.shift

                elif self.cfg.ablate_scale_only:
                    y_pred = self.rotation_manager.rotate_back(
                        input = accumulator_vec * coefs_decode.scale, 
                        reflect_vectors=reflect_vectors_for_y0
                    )
                elif self.cfg.ablate_shift_inside_scale:
                    y_pred = self.rotation_manager.rotate_back(
                        input = accumulator_vec * coefs_decode.scale + coefs_decode.shift, 
                        reflect_vectors=reflect_vectors_for_y0
                    )
                    
                else:
                    y_pred = self.rotation_manager.rotate_back(
                        input = accumulator_vec * coefs_decode.scale, 
                        reflect_vectors=reflect_vectors_for_y0
                    ) + coefs_decode.shift
             

            else:
                y_pred = y0*coefs_decode.scale + coefs_decode.shift
                reflect_vectors_for_y0 = None
                reflect_vectors_for_z = None
            
            

         
        return ZtoYResults(pred=y_pred, coefs=coefs_decode, y0=y0, 
            reflect_vectors_for_y0=reflect_vectors_for_y0, 
            reflect_vectors_for_z=reflect_vectors_for_z, 
            rotation_manager_z=self.rotation_manager, 
            rotation_manager_y0=self.rotation_manager_aug)
 
@dataclass
class ZtoYResults:
    pred: torch.Tensor
    coefs: CoefOutput
    y0: torch.Tensor
    reflect_vectors_for_y0: Optional[List[torch.Tensor]] = None
    reflect_vectors_for_z: Optional[List[torch.Tensor]] = None
    rotation_manager_z: Optional[Rotation] = None
    rotation_manager_y0: Optional[Rotation] = None

def compute_divergences(s1, t1, s2=1.0, t2=0.0, output_type: str = "w2_squared"):
    """ Compute divergences between N(t1, diag(s1²)) and N(t2, diag(s2²)) """  
    if output_type == "kl":
        # output = 0.5 * (s1.pow(2) + t1.pow(2) - 1 - s1.pow(2).log()) #TODO: this is positive kl 
        # Proper KL between two Gaussians
        var_ratio = (s1/s2).pow(2)
        t1_diff = t1 - t2
        kl = 0.5 * (var_ratio + t1_diff.pow(2)/s2.pow(2) - 1 - var_ratio.log())
        output = torch.sum(kl, dim=-1)
    if output_type == "wd":
        w2_squared = (torch.linalg.vector_norm(t1-t2).pow(2) + torch.sum((s1-s2).pow(2)))
        output = w2_squared 
    return output
 

       
def patch_embed(data, patch_len, stride):
    data = torch.nn.functional.pad(data, ( 0,  patch_len - stride,  ),  'replicate') # self.padding
    return data.unfold(dimension=-1, size=patch_len, step=stride)

def shift_embed(x, order=1):
    x = torch.nn.functional.pad(x, (order,0),  'replicate')
    return x[..., :-order]

def deri_embed(x, order=1):
    """ Compute discrete derivative of order n using rolling differences.  """
    x = torch.nn.functional.pad(x, (order,0),  'replicate')
    y = x.roll(-order, dims=-1)
    return (y - x)[..., :-order]
 
##### Parametrizations 

class SymmetricParametrization(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)
 
class MatrixExponentialParametrization(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)
    
class CayleyMapParametrization(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X): # (I + X)(I - X)^{-1}
        Id = self.Id.to(X.device)
        return torch.linalg.solve(Id - X, Id + X) 
    
class SkewParametrization(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)
    
class LowerTriangularParametrization(nn.Module):
    def forward(self, X):
        return torch.tril(X, diagonal=-1) + torch.eye(X.size(-1), device=X.device)

