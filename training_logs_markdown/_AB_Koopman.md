# data


```python
import torch 
import importlib
import monotonic
import data_manager
import metrics
import utils
importlib.reload(utils)
import train as Train
from train import execute_model_evaluation
import train_config
from data_manager import DatasetManager
from train_config import FlatACLConfig, FlatDLinearConfig, FlatNaiveConfig, FlatPatchTSTConfig, FlatTimeMixerConfig
from dataclasses import replace

%load_ext autoreload
%autoreload 2
modules_to_reload_list = [
    data_manager,
    Train,
    train_config,
    monotonic,
    # data_manager, # Reloaded only once even if listed twice
    utils,
    # train_config, # Reloaded only once even if listed twice
    metrics
]

# Initialize the data manager
data_mgr = DatasetManager(device='cuda')

# Load a synthetic dataset
data_mgr.load_trajectory('lorenz', steps=24999, dt=1e-2, ) # 51999 36999
# SCALE = False
```

    LorenzSystem initialized with method: rk4 on device: cuda
    
    ==================================================
    Dataset: lorenz (synthetic)
    ==================================================
    Shape: torch.Size([25000, 3])
    Channels: 3
    Length: 25000
    Parameters: {'steps': 24999, 'dt': 0.01}
    
    Sample data (first 2 rows):
    tensor([[1.0000, 0.9800, 1.1000],
            [1.0106, 1.2389, 1.0820]], device='cuda:0')
    ==================================================
    




    <data_manager.DatasetManager at 0x223795d38c0>



# Exp - Lorenz - 19000 - ablation

## baseline


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128, 
    ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=True, 
    ablate_shift_inside_scale=False,
    householder_reflects_latent = 2,
    householder_reflects_data = 4,
    mixing_strategy='delay_only', 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 74.4717, mae: 6.5322, huber: 6.0521, swd: 42.9372, ept: 32.6197
    Epoch [1/50], Val Losses: mse: 59.6087, mae: 5.8431, huber: 5.3661, swd: 26.8070, ept: 38.4709
    Epoch [1/50], Test Losses: mse: 55.4091, mae: 5.4901, huber: 5.0164, swd: 26.7390, ept: 37.7025
      Epoch 1 composite train-obj: 6.052078
            Val objective improved inf → 5.3661, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.2988, mae: 4.8663, huber: 4.4009, swd: 17.2347, ept: 87.1235
    Epoch [2/50], Val Losses: mse: 50.1042, mae: 5.0190, huber: 4.5525, swd: 14.8758, ept: 84.2962
    Epoch [2/50], Test Losses: mse: 46.4008, mae: 4.6995, huber: 4.2369, swd: 15.3488, ept: 77.5370
      Epoch 2 composite train-obj: 4.400894
            Val objective improved 5.3661 → 4.5525, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.0256, mae: 4.1310, huber: 3.6781, swd: 9.5066, ept: 132.7567
    Epoch [3/50], Val Losses: mse: 43.0674, mae: 4.3850, huber: 3.9294, swd: 8.9794, ept: 114.0095
    Epoch [3/50], Test Losses: mse: 40.0224, mae: 4.1567, huber: 3.7030, swd: 9.4554, ept: 115.2078
      Epoch 3 composite train-obj: 3.678081
            Val objective improved 4.5525 → 3.9294, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 34.8529, mae: 3.6757, huber: 3.2314, swd: 5.8119, ept: 168.4728
    Epoch [4/50], Val Losses: mse: 45.1078, mae: 4.3324, huber: 3.8801, swd: 6.5226, ept: 139.1567
    Epoch [4/50], Test Losses: mse: 38.6800, mae: 3.9682, huber: 3.5182, swd: 6.7174, ept: 139.0107
      Epoch 4 composite train-obj: 3.231413
            Val objective improved 3.9294 → 3.8801, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.0636, mae: 3.4134, huber: 2.9762, swd: 4.6921, ept: 183.3263
    Epoch [5/50], Val Losses: mse: 40.2229, mae: 4.0557, huber: 3.6061, swd: 5.8727, ept: 144.6233
    Epoch [5/50], Test Losses: mse: 35.1236, mae: 3.7061, huber: 3.2611, swd: 5.9484, ept: 154.5843
      Epoch 5 composite train-obj: 2.976231
            Val objective improved 3.8801 → 3.6061, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.7938, mae: 3.2280, huber: 2.7951, swd: 3.9501, ept: 192.1844
    Epoch [6/50], Val Losses: mse: 38.5966, mae: 3.8961, huber: 3.4504, swd: 4.5329, ept: 153.2357
    Epoch [6/50], Test Losses: mse: 33.2245, mae: 3.5671, huber: 3.1240, swd: 4.7659, ept: 156.9653
      Epoch 6 composite train-obj: 2.795125
            Val objective improved 3.6061 → 3.4504, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.0224, mae: 3.0905, huber: 2.6606, swd: 3.3716, ept: 198.1145
    Epoch [7/50], Val Losses: mse: 38.6302, mae: 3.8133, huber: 3.3698, swd: 4.2561, ept: 161.4422
    Epoch [7/50], Test Losses: mse: 30.9504, mae: 3.3467, huber: 2.9097, swd: 4.0855, ept: 172.7421
      Epoch 7 composite train-obj: 2.660628
            Val objective improved 3.4504 → 3.3698, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 25.7389, mae: 2.8834, huber: 2.4605, swd: 2.8519, ept: 207.8556
    Epoch [8/50], Val Losses: mse: 39.4300, mae: 3.8302, huber: 3.3899, swd: 3.9284, ept: 164.5912
    Epoch [8/50], Test Losses: mse: 29.7514, mae: 3.2667, huber: 2.8318, swd: 3.6652, ept: 177.5161
      Epoch 8 composite train-obj: 2.460533
            No improvement (3.3899), counter 1/5
    Epoch [9/50], Train Losses: mse: 24.5200, mae: 2.8076, huber: 2.3848, swd: 2.6165, ept: 212.1876
    Epoch [9/50], Val Losses: mse: 36.1324, mae: 3.6397, huber: 3.1988, swd: 3.7789, ept: 171.5737
    Epoch [9/50], Test Losses: mse: 29.1206, mae: 3.2294, huber: 2.7923, swd: 3.3339, ept: 183.1656
      Epoch 9 composite train-obj: 2.384766
            Val objective improved 3.3698 → 3.1988, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.3015, mae: 2.7098, huber: 2.2897, swd: 2.3517, ept: 217.3866
    Epoch [10/50], Val Losses: mse: 38.5680, mae: 3.7487, huber: 3.3073, swd: 3.3875, ept: 168.7725
    Epoch [10/50], Test Losses: mse: 30.4456, mae: 3.2955, huber: 2.8592, swd: 3.3314, ept: 181.2381
      Epoch 10 composite train-obj: 2.289651
            No improvement (3.3073), counter 1/5
    Epoch [11/50], Train Losses: mse: 21.7114, mae: 2.5885, huber: 2.1718, swd: 2.0772, ept: 221.7873
    Epoch [11/50], Val Losses: mse: 38.6587, mae: 3.6710, huber: 3.2329, swd: 2.9046, ept: 179.4020
    Epoch [11/50], Test Losses: mse: 28.3105, mae: 3.0879, huber: 2.6569, swd: 2.7085, ept: 197.2829
      Epoch 11 composite train-obj: 2.171835
            No improvement (3.2329), counter 2/5
    Epoch [12/50], Train Losses: mse: 20.2711, mae: 2.4716, huber: 2.0581, swd: 1.8772, ept: 227.2980
    Epoch [12/50], Val Losses: mse: 35.7608, mae: 3.4500, huber: 3.0163, swd: 2.4089, ept: 190.3063
    Epoch [12/50], Test Losses: mse: 26.3279, mae: 2.9237, huber: 2.4978, swd: 2.1932, ept: 208.3584
      Epoch 12 composite train-obj: 2.058147
            Val objective improved 3.1988 → 3.0163, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.6772, mae: 2.3353, huber: 1.9270, swd: 1.6482, ept: 233.1424
    Epoch [13/50], Val Losses: mse: 39.4328, mae: 3.6300, huber: 3.1954, swd: 2.3789, ept: 186.5176
    Epoch [13/50], Test Losses: mse: 28.9320, mae: 3.0592, huber: 2.6310, swd: 2.4312, ept: 204.5124
      Epoch 13 composite train-obj: 1.926951
            No improvement (3.1954), counter 1/5
    Epoch [14/50], Train Losses: mse: 18.3938, mae: 2.3204, huber: 1.9115, swd: 1.6160, ept: 234.7459
    Epoch [14/50], Val Losses: mse: 39.7576, mae: 3.6242, huber: 3.1895, swd: 2.6238, ept: 190.9158
    Epoch [14/50], Test Losses: mse: 27.9504, mae: 3.0314, huber: 2.6018, swd: 2.4990, ept: 203.4434
      Epoch 14 composite train-obj: 1.911461
            No improvement (3.1895), counter 2/5
    Epoch [15/50], Train Losses: mse: 17.4962, mae: 2.2386, huber: 1.8326, swd: 1.4879, ept: 239.0218
    Epoch [15/50], Val Losses: mse: 35.8527, mae: 3.3735, huber: 2.9453, swd: 2.6410, ept: 195.1966
    Epoch [15/50], Test Losses: mse: 24.5848, mae: 2.7727, huber: 2.3523, swd: 2.2517, ept: 212.6323
      Epoch 15 composite train-obj: 1.832627
            Val objective improved 3.0163 → 2.9453, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.5352, mae: 2.1589, huber: 1.7566, swd: 1.3602, ept: 242.4407
    Epoch [16/50], Val Losses: mse: 34.8165, mae: 3.2573, huber: 2.8326, swd: 2.1494, ept: 207.4208
    Epoch [16/50], Test Losses: mse: 23.0503, mae: 2.6437, huber: 2.2266, swd: 1.8238, ept: 223.7524
      Epoch 16 composite train-obj: 1.756594
            Val objective improved 2.9453 → 2.8326, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.3086, mae: 2.0499, huber: 1.6517, swd: 1.2294, ept: 248.6760
    Epoch [17/50], Val Losses: mse: 39.6649, mae: 3.4872, huber: 3.0603, swd: 2.3609, ept: 205.4483
    Epoch [17/50], Test Losses: mse: 29.2050, mae: 2.9530, huber: 2.5322, swd: 2.1323, ept: 216.3663
      Epoch 17 composite train-obj: 1.651688
            No improvement (3.0603), counter 1/5
    Epoch [18/50], Train Losses: mse: 14.9610, mae: 2.0171, huber: 1.6206, swd: 1.2014, ept: 251.1886
    Epoch [18/50], Val Losses: mse: 38.9293, mae: 3.4758, huber: 3.0483, swd: 2.4443, ept: 198.9698
    Epoch [18/50], Test Losses: mse: 26.1839, mae: 2.8169, huber: 2.3962, swd: 2.0627, ept: 215.6455
      Epoch 18 composite train-obj: 1.620588
            No improvement (3.0483), counter 2/5
    Epoch [19/50], Train Losses: mse: 15.4014, mae: 2.0540, huber: 1.6553, swd: 1.2488, ept: 250.6757
    Epoch [19/50], Val Losses: mse: 37.6479, mae: 3.3907, huber: 2.9645, swd: 2.0696, ept: 206.8464
    Epoch [19/50], Test Losses: mse: 26.0310, mae: 2.7488, huber: 2.3301, swd: 1.7758, ept: 228.3549
      Epoch 19 composite train-obj: 1.655286
            No improvement (2.9645), counter 3/5
    Epoch [20/50], Train Losses: mse: 13.9194, mae: 1.9188, huber: 1.5271, swd: 1.0805, ept: 256.2429
    Epoch [20/50], Val Losses: mse: 37.6045, mae: 3.3638, huber: 2.9413, swd: 2.1079, ept: 206.5557
    Epoch [20/50], Test Losses: mse: 24.9455, mae: 2.6672, huber: 2.2546, swd: 1.7582, ept: 227.5794
      Epoch 20 composite train-obj: 1.527124
            No improvement (2.9413), counter 4/5
    Epoch [21/50], Train Losses: mse: 13.7002, mae: 1.8852, huber: 1.4955, swd: 1.0544, ept: 259.8782
    Epoch [21/50], Val Losses: mse: 37.3940, mae: 3.4400, huber: 3.0108, swd: 2.4402, ept: 199.5756
    Epoch [21/50], Test Losses: mse: 24.0942, mae: 2.7243, huber: 2.3048, swd: 2.1181, ept: 218.1151
      Epoch 21 composite train-obj: 1.495529
    Epoch [21/50], Test Losses: mse: 23.0515, mae: 2.6438, huber: 2.2267, swd: 1.8239, ept: 223.7528
    Best round's Test MSE: 23.0503, MAE: 2.6437, SWD: 1.8238
    Best round's Validation MSE: 34.8165, MAE: 3.2573, SWD: 2.1494
    Best round's Test verification MSE : 23.0515, MAE: 2.6438, SWD: 1.8239
    Time taken: 76.05 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 74.9860, mae: 6.5212, huber: 6.0414, swd: 43.3840, ept: 34.4640
    Epoch [1/50], Val Losses: mse: 58.6281, mae: 5.7031, huber: 5.2289, swd: 25.1048, ept: 48.8319
    Epoch [1/50], Test Losses: mse: 53.7552, mae: 5.3205, huber: 4.8494, swd: 25.6747, ept: 47.8742
      Epoch 1 composite train-obj: 6.041425
            Val objective improved inf → 5.2289, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.0125, mae: 4.7628, huber: 4.2986, swd: 15.3710, ept: 96.8786
    Epoch [2/50], Val Losses: mse: 49.1243, mae: 4.9104, huber: 4.4461, swd: 12.9426, ept: 90.6375
    Epoch [2/50], Test Losses: mse: 45.5511, mae: 4.6168, huber: 4.1553, swd: 13.9522, ept: 89.6520
      Epoch 2 composite train-obj: 4.298633
            Val objective improved 5.2289 → 4.4461, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.8942, mae: 4.0406, huber: 3.5885, swd: 8.2801, ept: 142.8682
    Epoch [3/50], Val Losses: mse: 45.7949, mae: 4.5340, huber: 4.0753, swd: 8.2812, ept: 114.3149
    Epoch [3/50], Test Losses: mse: 41.1846, mae: 4.2024, huber: 3.7459, swd: 8.4849, ept: 123.6740
      Epoch 3 composite train-obj: 3.588481
            Val objective improved 4.4461 → 4.0753, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 34.6805, mae: 3.6605, huber: 3.2158, swd: 5.6158, ept: 169.0447
    Epoch [4/50], Val Losses: mse: 43.1798, mae: 4.2793, huber: 3.8257, swd: 6.9380, ept: 133.7176
    Epoch [4/50], Test Losses: mse: 38.7192, mae: 3.9879, huber: 3.5366, swd: 7.3794, ept: 144.0006
      Epoch 4 composite train-obj: 3.215848
            Val objective improved 4.0753 → 3.8257, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 31.6822, mae: 3.3956, huber: 2.9579, swd: 4.5580, ept: 184.2869
    Epoch [5/50], Val Losses: mse: 40.0546, mae: 4.0449, huber: 3.5960, swd: 5.8536, ept: 147.5896
    Epoch [5/50], Test Losses: mse: 35.0112, mae: 3.7054, huber: 3.2592, swd: 6.0943, ept: 154.1722
      Epoch 5 composite train-obj: 2.957924
            Val objective improved 3.8257 → 3.5960, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.7763, mae: 3.2311, huber: 2.7976, swd: 3.8446, ept: 192.3106
    Epoch [6/50], Val Losses: mse: 41.3366, mae: 4.1012, huber: 3.6532, swd: 5.6205, ept: 149.4874
    Epoch [6/50], Test Losses: mse: 33.9429, mae: 3.6247, huber: 3.1807, swd: 5.2156, ept: 159.9798
      Epoch 6 composite train-obj: 2.797601
            No improvement (3.6532), counter 1/5
    Epoch [7/50], Train Losses: mse: 27.9508, mae: 3.0871, huber: 2.6568, swd: 3.3392, ept: 198.2041
    Epoch [7/50], Val Losses: mse: 42.8429, mae: 4.0812, huber: 3.6346, swd: 4.5963, ept: 154.8228
    Epoch [7/50], Test Losses: mse: 33.3336, mae: 3.5113, huber: 3.0704, swd: 4.2864, ept: 168.3324
      Epoch 7 composite train-obj: 2.656785
            No improvement (3.6346), counter 2/5
    Epoch [8/50], Train Losses: mse: 26.2731, mae: 2.9519, huber: 2.5248, swd: 2.8957, ept: 203.6560
    Epoch [8/50], Val Losses: mse: 38.1223, mae: 3.8372, huber: 3.3925, swd: 4.0961, ept: 158.2303
    Epoch [8/50], Test Losses: mse: 30.6678, mae: 3.3488, huber: 2.9098, swd: 4.1013, ept: 174.5141
      Epoch 8 composite train-obj: 2.524826
            Val objective improved 3.5960 → 3.3925, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.5241, mae: 2.8097, huber: 2.3868, swd: 2.5841, ept: 209.1949
    Epoch [9/50], Val Losses: mse: 37.1755, mae: 3.7614, huber: 3.3175, swd: 3.5628, ept: 167.4498
    Epoch [9/50], Test Losses: mse: 29.3107, mae: 3.2446, huber: 2.8079, swd: 3.5463, ept: 178.1322
      Epoch 9 composite train-obj: 2.386791
            Val objective improved 3.3925 → 3.3175, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.3211, mae: 2.7248, huber: 2.3034, swd: 2.3781, ept: 213.0604
    Epoch [10/50], Val Losses: mse: 38.0670, mae: 3.8070, huber: 3.3627, swd: 3.8372, ept: 160.2422
    Epoch [10/50], Test Losses: mse: 31.3314, mae: 3.3818, huber: 2.9432, swd: 3.7234, ept: 172.9676
      Epoch 10 composite train-obj: 2.303405
            No improvement (3.3627), counter 1/5
    Epoch [11/50], Train Losses: mse: 21.5077, mae: 2.5770, huber: 2.1602, swd: 2.1060, ept: 220.4831
    Epoch [11/50], Val Losses: mse: 38.7634, mae: 3.7220, huber: 3.2849, swd: 3.5994, ept: 172.2634
    Epoch [11/50], Test Losses: mse: 28.7670, mae: 3.1109, huber: 2.6802, swd: 3.2561, ept: 191.6377
      Epoch 11 composite train-obj: 2.160157
            Val objective improved 3.3175 → 3.2849, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.4659, mae: 2.4918, huber: 2.0778, swd: 1.9292, ept: 225.0461
    Epoch [12/50], Val Losses: mse: 34.8785, mae: 3.4804, huber: 3.0457, swd: 2.9476, ept: 184.5870
    Epoch [12/50], Test Losses: mse: 29.2950, mae: 3.0985, huber: 2.6691, swd: 2.4362, ept: 198.9836
      Epoch 12 composite train-obj: 2.077790
            Val objective improved 3.2849 → 3.0457, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 19.0544, mae: 2.3762, huber: 1.9656, swd: 1.7346, ept: 230.2625
    Epoch [13/50], Val Losses: mse: 38.7402, mae: 3.6244, huber: 3.1897, swd: 2.3673, ept: 180.4288
    Epoch [13/50], Test Losses: mse: 27.0863, mae: 2.9535, huber: 2.5270, swd: 1.9698, ept: 203.7760
      Epoch 13 composite train-obj: 1.965563
            No improvement (3.1897), counter 1/5
    Epoch [14/50], Train Losses: mse: 18.4665, mae: 2.3338, huber: 1.9238, swd: 1.6517, ept: 233.0189
    Epoch [14/50], Val Losses: mse: 38.3153, mae: 3.6073, huber: 3.1710, swd: 2.5169, ept: 184.3080
    Epoch [14/50], Test Losses: mse: 29.3239, mae: 3.0451, huber: 2.6171, swd: 2.5076, ept: 205.5672
      Epoch 14 composite train-obj: 1.923791
            No improvement (3.1710), counter 2/5
    Epoch [15/50], Train Losses: mse: 17.4731, mae: 2.2452, huber: 1.8390, swd: 1.5485, ept: 238.3653
    Epoch [15/50], Val Losses: mse: 34.5335, mae: 3.3747, huber: 2.9430, swd: 2.5940, ept: 192.0730
    Epoch [15/50], Test Losses: mse: 26.6831, mae: 2.8749, huber: 2.4514, swd: 2.3341, ept: 210.5420
      Epoch 15 composite train-obj: 1.839017
            Val objective improved 3.0457 → 2.9430, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.7596, mae: 2.1831, huber: 1.7791, swd: 1.4485, ept: 241.5293
    Epoch [16/50], Val Losses: mse: 34.9507, mae: 3.3901, huber: 2.9593, swd: 2.6889, ept: 188.5570
    Epoch [16/50], Test Losses: mse: 25.9469, mae: 2.8086, huber: 2.3872, swd: 2.1070, ept: 214.5743
      Epoch 16 composite train-obj: 1.779086
            No improvement (2.9593), counter 1/5
    Epoch [17/50], Train Losses: mse: 15.4846, mae: 2.0690, huber: 1.6702, swd: 1.3002, ept: 246.6622
    Epoch [17/50], Val Losses: mse: 31.1430, mae: 3.1889, huber: 2.7602, swd: 2.3580, ept: 192.0410
    Epoch [17/50], Test Losses: mse: 26.7978, mae: 2.8618, huber: 2.4399, swd: 2.1386, ept: 209.7544
      Epoch 17 composite train-obj: 1.670213
            Val objective improved 2.9430 → 2.7602, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.0546, mae: 2.0320, huber: 1.6344, swd: 1.2533, ept: 249.7268
    Epoch [18/50], Val Losses: mse: 35.1641, mae: 3.2908, huber: 2.8669, swd: 2.2631, ept: 202.6992
    Epoch [18/50], Test Losses: mse: 24.1084, mae: 2.6318, huber: 2.2181, swd: 1.6580, ept: 227.3047
      Epoch 18 composite train-obj: 1.634352
            No improvement (2.8669), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.2877, mae: 1.9588, huber: 1.5647, swd: 1.1651, ept: 253.3060
    Epoch [19/50], Val Losses: mse: 33.8288, mae: 3.2317, huber: 2.8071, swd: 2.3838, ept: 205.1492
    Epoch [19/50], Test Losses: mse: 25.9831, mae: 2.7509, huber: 2.3344, swd: 2.0422, ept: 224.1496
      Epoch 19 composite train-obj: 1.564737
            No improvement (2.8071), counter 2/5
    Epoch [20/50], Train Losses: mse: 13.5861, mae: 1.8907, huber: 1.4997, swd: 1.1051, ept: 257.1673
    Epoch [20/50], Val Losses: mse: 34.1141, mae: 3.2710, huber: 2.8469, swd: 2.3322, ept: 207.3784
    Epoch [20/50], Test Losses: mse: 23.7812, mae: 2.6392, huber: 2.2253, swd: 1.8233, ept: 224.1165
      Epoch 20 composite train-obj: 1.499658
            No improvement (2.8469), counter 3/5
    Epoch [21/50], Train Losses: mse: 14.1826, mae: 1.9496, huber: 1.5549, swd: 1.1558, ept: 255.6296
    Epoch [21/50], Val Losses: mse: 37.0295, mae: 3.5017, huber: 3.0733, swd: 3.0192, ept: 193.9375
    Epoch [21/50], Test Losses: mse: 28.0795, mae: 2.8616, huber: 2.4431, swd: 1.9321, ept: 216.9593
      Epoch 21 composite train-obj: 1.554923
            No improvement (3.0733), counter 4/5
    Epoch [22/50], Train Losses: mse: 13.6246, mae: 1.8995, huber: 1.5076, swd: 1.0781, ept: 257.7962
    Epoch [22/50], Val Losses: mse: 34.9946, mae: 3.2872, huber: 2.8640, swd: 2.0995, ept: 207.7540
    Epoch [22/50], Test Losses: mse: 25.7905, mae: 2.7259, huber: 2.3102, swd: 1.6764, ept: 227.4660
      Epoch 22 composite train-obj: 1.507625
    Epoch [22/50], Test Losses: mse: 26.7953, mae: 2.8617, huber: 2.4398, swd: 2.1387, ept: 209.7696
    Best round's Test MSE: 26.7978, MAE: 2.8618, SWD: 2.1386
    Best round's Validation MSE: 31.1430, MAE: 3.1889, SWD: 2.3580
    Best round's Test verification MSE : 26.7953, MAE: 2.8617, SWD: 2.1387
    Time taken: 84.12 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.2624, mae: 6.5101, huber: 6.0304, swd: 44.3738, ept: 35.1315
    Epoch [1/50], Val Losses: mse: 59.1496, mae: 5.7427, huber: 5.2671, swd: 26.6024, ept: 48.1855
    Epoch [1/50], Test Losses: mse: 53.9087, mae: 5.3350, huber: 4.8633, swd: 27.1466, ept: 47.4498
      Epoch 1 composite train-obj: 6.030425
            Val objective improved inf → 5.2671, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.2940, mae: 4.7751, huber: 4.3104, swd: 16.8015, ept: 100.5673
    Epoch [2/50], Val Losses: mse: 50.1518, mae: 4.9488, huber: 4.4836, swd: 13.5682, ept: 92.5905
    Epoch [2/50], Test Losses: mse: 45.9288, mae: 4.6300, huber: 4.1664, swd: 14.6739, ept: 88.5283
      Epoch 2 composite train-obj: 4.310419
            Val objective improved 5.2671 → 4.4836, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.6232, mae: 4.0963, huber: 3.6428, swd: 9.4508, ept: 140.7967
    Epoch [3/50], Val Losses: mse: 47.6867, mae: 4.6965, huber: 4.2352, swd: 9.5586, ept: 109.0016
    Epoch [3/50], Test Losses: mse: 41.8238, mae: 4.2717, huber: 3.8142, swd: 10.6748, ept: 115.1333
      Epoch 3 composite train-obj: 3.642842
            Val objective improved 4.4836 → 4.2352, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.3835, mae: 3.7346, huber: 3.2884, swd: 6.7951, ept: 161.1146
    Epoch [4/50], Val Losses: mse: 43.7580, mae: 4.2932, huber: 3.8392, swd: 6.8107, ept: 132.7532
    Epoch [4/50], Test Losses: mse: 37.8513, mae: 3.9302, huber: 3.4793, swd: 7.2115, ept: 134.9980
      Epoch 4 composite train-obj: 3.288383
            Val objective improved 4.2352 → 3.8392, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.3811, mae: 3.4542, huber: 3.0150, swd: 5.3482, ept: 177.2833
    Epoch [5/50], Val Losses: mse: 41.0448, mae: 4.1268, huber: 3.6764, swd: 6.6987, ept: 137.3645
    Epoch [5/50], Test Losses: mse: 36.9298, mae: 3.8273, huber: 3.3805, swd: 6.4719, ept: 147.8209
      Epoch 5 composite train-obj: 3.015002
            Val objective improved 3.8392 → 3.6764, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.3494, mae: 3.2847, huber: 2.8495, swd: 4.4392, ept: 184.4730
    Epoch [6/50], Val Losses: mse: 38.7300, mae: 3.9473, huber: 3.4995, swd: 5.7238, ept: 145.4306
    Epoch [6/50], Test Losses: mse: 33.3815, mae: 3.5903, huber: 3.1453, swd: 5.6551, ept: 151.8247
      Epoch 6 composite train-obj: 2.849527
            Val objective improved 3.6764 → 3.4995, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.4830, mae: 3.0489, huber: 2.6201, swd: 3.6770, ept: 195.0120
    Epoch [7/50], Val Losses: mse: 40.7846, mae: 3.9601, huber: 3.5153, swd: 5.1690, ept: 153.5188
    Epoch [7/50], Test Losses: mse: 32.2978, mae: 3.4324, huber: 2.9917, swd: 4.3163, ept: 164.9773
      Epoch 7 composite train-obj: 2.620087
            No improvement (3.5153), counter 1/5
    Epoch [8/50], Train Losses: mse: 25.9740, mae: 2.9313, huber: 2.5044, swd: 3.2054, ept: 200.4061
    Epoch [8/50], Val Losses: mse: 37.2341, mae: 3.7172, huber: 3.2749, swd: 4.0957, ept: 160.8869
    Epoch [8/50], Test Losses: mse: 28.9508, mae: 3.2086, huber: 2.7715, swd: 3.7368, ept: 178.3502
      Epoch 8 composite train-obj: 2.504365
            Val objective improved 3.4995 → 3.2749, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.3343, mae: 2.8088, huber: 2.3846, swd: 2.8231, ept: 205.8823
    Epoch [9/50], Val Losses: mse: 41.1436, mae: 3.9097, huber: 3.4653, swd: 4.7360, ept: 158.9152
    Epoch [9/50], Test Losses: mse: 29.8688, mae: 3.2965, huber: 2.8580, swd: 4.1901, ept: 172.4198
      Epoch 9 composite train-obj: 2.384619
            No improvement (3.4653), counter 1/5
    Epoch [10/50], Train Losses: mse: 22.7459, mae: 2.6872, huber: 2.2659, swd: 2.5296, ept: 211.1636
    Epoch [10/50], Val Losses: mse: 41.2430, mae: 3.8407, huber: 3.4008, swd: 4.3508, ept: 164.6021
    Epoch [10/50], Test Losses: mse: 29.3860, mae: 3.2020, huber: 2.7683, swd: 3.8697, ept: 179.1476
      Epoch 10 composite train-obj: 2.265879
            No improvement (3.4008), counter 2/5
    Epoch [11/50], Train Losses: mse: 21.6341, mae: 2.5962, huber: 2.1778, swd: 2.3089, ept: 216.4084
    Epoch [11/50], Val Losses: mse: 38.7518, mae: 3.6786, huber: 3.2392, swd: 3.6091, ept: 171.1008
    Epoch [11/50], Test Losses: mse: 27.7456, mae: 3.0636, huber: 2.6316, swd: 2.8091, ept: 182.8886
      Epoch 11 composite train-obj: 2.177808
            Val objective improved 3.2749 → 3.2392, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 19.4942, mae: 2.4123, huber: 2.0007, swd: 1.9591, ept: 225.8293
    Epoch [12/50], Val Losses: mse: 36.4618, mae: 3.5371, huber: 3.0998, swd: 3.0373, ept: 181.1326
    Epoch [12/50], Test Losses: mse: 28.3466, mae: 3.0809, huber: 2.6493, swd: 2.6977, ept: 195.2504
      Epoch 12 composite train-obj: 2.000741
            Val objective improved 3.2392 → 3.0998, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 19.0743, mae: 2.3786, huber: 1.9675, swd: 1.8616, ept: 228.2778
    Epoch [13/50], Val Losses: mse: 40.4504, mae: 3.6910, huber: 3.2567, swd: 3.5215, ept: 175.2632
    Epoch [13/50], Test Losses: mse: 29.5476, mae: 3.1050, huber: 2.6754, swd: 2.9213, ept: 190.5348
      Epoch 13 composite train-obj: 1.967492
            No improvement (3.2567), counter 1/5
    Epoch [14/50], Train Losses: mse: 17.8837, mae: 2.2770, huber: 1.8699, swd: 1.6984, ept: 233.6776
    Epoch [14/50], Val Losses: mse: 39.4406, mae: 3.6456, huber: 3.2091, swd: 3.2930, ept: 178.3535
    Epoch [14/50], Test Losses: mse: 29.9926, mae: 3.1148, huber: 2.6856, swd: 2.7982, ept: 194.8105
      Epoch 14 composite train-obj: 1.869931
            No improvement (3.2091), counter 2/5
    Epoch [15/50], Train Losses: mse: 17.0573, mae: 2.2084, huber: 1.8042, swd: 1.5695, ept: 237.3453
    Epoch [15/50], Val Losses: mse: 36.2258, mae: 3.4727, huber: 3.0418, swd: 3.1952, ept: 188.2991
    Epoch [15/50], Test Losses: mse: 24.4609, mae: 2.7923, huber: 2.3718, swd: 2.7505, ept: 211.6368
      Epoch 15 composite train-obj: 1.804173
            Val objective improved 3.0998 → 3.0418, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.5193, mae: 2.1589, huber: 1.7562, swd: 1.5204, ept: 240.7404
    Epoch [16/50], Val Losses: mse: 34.3234, mae: 3.3425, huber: 2.9120, swd: 2.6875, ept: 190.7381
    Epoch [16/50], Test Losses: mse: 26.4790, mae: 2.8684, huber: 2.4453, swd: 2.3559, ept: 207.4360
      Epoch 16 composite train-obj: 1.756223
            Val objective improved 3.0418 → 2.9120, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.6887, mae: 2.0816, huber: 1.6825, swd: 1.3979, ept: 245.6139
    Epoch [17/50], Val Losses: mse: 34.0268, mae: 3.2766, huber: 2.8513, swd: 3.1646, ept: 197.4361
    Epoch [17/50], Test Losses: mse: 25.2758, mae: 2.7357, huber: 2.3204, swd: 2.2556, ept: 220.7585
      Epoch 17 composite train-obj: 1.682487
            Val objective improved 2.9120 → 2.8513, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.7950, mae: 2.0870, huber: 1.6874, swd: 1.3943, ept: 246.3937
    Epoch [18/50], Val Losses: mse: 36.0267, mae: 3.3899, huber: 2.9608, swd: 2.6759, ept: 189.5255
    Epoch [18/50], Test Losses: mse: 25.5016, mae: 2.8046, huber: 2.3837, swd: 2.2088, ept: 211.9406
      Epoch 18 composite train-obj: 1.687435
            No improvement (2.9608), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.6959, mae: 1.9922, huber: 1.5965, swd: 1.2947, ept: 251.0507
    Epoch [19/50], Val Losses: mse: 35.7698, mae: 3.3222, huber: 2.8966, swd: 2.8657, ept: 200.5733
    Epoch [19/50], Test Losses: mse: 25.1453, mae: 2.7175, huber: 2.3012, swd: 1.9650, ept: 222.3462
      Epoch 19 composite train-obj: 1.596485
            No improvement (2.8966), counter 2/5
    Epoch [20/50], Train Losses: mse: 15.0458, mae: 2.0070, huber: 1.6114, swd: 1.3068, ept: 252.4757
    Epoch [20/50], Val Losses: mse: 40.0234, mae: 3.5129, huber: 3.0843, swd: 2.4198, ept: 195.4707
    Epoch [20/50], Test Losses: mse: 27.3230, mae: 2.8714, huber: 2.4493, swd: 2.0937, ept: 212.7967
      Epoch 20 composite train-obj: 1.611420
            No improvement (3.0843), counter 3/5
    Epoch [21/50], Train Losses: mse: 14.2971, mae: 1.9537, huber: 1.5599, swd: 1.2315, ept: 254.6326
    Epoch [21/50], Val Losses: mse: 34.4700, mae: 3.2157, huber: 2.7929, swd: 2.4690, ept: 203.3592
    Epoch [21/50], Test Losses: mse: 24.0258, mae: 2.6055, huber: 2.1937, swd: 1.8248, ept: 230.3986
      Epoch 21 composite train-obj: 1.559888
            Val objective improved 2.8513 → 2.7929, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.2870, mae: 1.8580, huber: 1.4692, swd: 1.1098, ept: 259.5629
    Epoch [22/50], Val Losses: mse: 34.2990, mae: 3.2028, huber: 2.7827, swd: 2.7368, ept: 201.5798
    Epoch [22/50], Test Losses: mse: 25.7565, mae: 2.6839, huber: 2.2730, swd: 2.1315, ept: 227.2282
      Epoch 22 composite train-obj: 1.469228
            Val objective improved 2.7929 → 2.7827, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 13.0899, mae: 1.8320, huber: 1.4453, swd: 1.0917, ept: 260.6456
    Epoch [23/50], Val Losses: mse: 32.2109, mae: 3.1391, huber: 2.7163, swd: 2.5869, ept: 200.8301
    Epoch [23/50], Test Losses: mse: 24.6125, mae: 2.6440, huber: 2.2309, swd: 1.9623, ept: 226.6328
      Epoch 23 composite train-obj: 1.445301
            Val objective improved 2.7827 → 2.7163, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 12.5364, mae: 1.7791, huber: 1.3950, swd: 1.0238, ept: 263.3201
    Epoch [24/50], Val Losses: mse: 33.3506, mae: 3.1281, huber: 2.7098, swd: 2.4314, ept: 211.7318
    Epoch [24/50], Test Losses: mse: 24.5682, mae: 2.6333, huber: 2.2245, swd: 1.8660, ept: 227.4394
      Epoch 24 composite train-obj: 1.395010
            Val objective improved 2.7163 → 2.7098, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 13.1415, mae: 1.8329, huber: 1.4454, swd: 1.0854, ept: 261.1581
    Epoch [25/50], Val Losses: mse: 33.9535, mae: 3.2353, huber: 2.8096, swd: 2.3278, ept: 198.0551
    Epoch [25/50], Test Losses: mse: 25.6330, mae: 2.7590, huber: 2.3434, swd: 1.9477, ept: 216.4805
      Epoch 25 composite train-obj: 1.445351
            No improvement (2.8096), counter 1/5
    Epoch [26/50], Train Losses: mse: 11.7543, mae: 1.7148, huber: 1.3329, swd: 0.9701, ept: 267.1975
    Epoch [26/50], Val Losses: mse: 30.5676, mae: 2.9814, huber: 2.5676, swd: 2.1676, ept: 210.9253
    Epoch [26/50], Test Losses: mse: 23.6585, mae: 2.5424, huber: 2.1371, swd: 1.6659, ept: 233.6276
      Epoch 26 composite train-obj: 1.332859
            Val objective improved 2.7098 → 2.5676, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 10.8947, mae: 1.6257, huber: 1.2503, swd: 0.8797, ept: 270.9426
    Epoch [27/50], Val Losses: mse: 39.1185, mae: 3.4015, huber: 2.9819, swd: 2.4322, ept: 205.9317
    Epoch [27/50], Test Losses: mse: 23.9477, mae: 2.5943, huber: 2.1840, swd: 1.8802, ept: 230.2733
      Epoch 27 composite train-obj: 1.250289
            No improvement (2.9819), counter 1/5
    Epoch [28/50], Train Losses: mse: 11.0939, mae: 1.6431, huber: 1.2663, swd: 0.8927, ept: 271.0624
    Epoch [28/50], Val Losses: mse: 33.8291, mae: 3.2081, huber: 2.7869, swd: 2.3432, ept: 204.8083
    Epoch [28/50], Test Losses: mse: 25.5040, mae: 2.6930, huber: 2.2809, swd: 1.7732, ept: 227.7269
      Epoch 28 composite train-obj: 1.266335
            No improvement (2.7869), counter 2/5
    Epoch [29/50], Train Losses: mse: 11.8619, mae: 1.7089, huber: 1.3284, swd: 0.9527, ept: 268.7330
    Epoch [29/50], Val Losses: mse: 32.5470, mae: 3.0917, huber: 2.6755, swd: 2.2414, ept: 209.8713
    Epoch [29/50], Test Losses: mse: 29.3549, mae: 2.8078, huber: 2.3983, swd: 1.8828, ept: 229.0991
      Epoch 29 composite train-obj: 1.328434
            No improvement (2.6755), counter 3/5
    Epoch [30/50], Train Losses: mse: 10.8696, mae: 1.6247, huber: 1.2486, swd: 0.8625, ept: 271.1207
    Epoch [30/50], Val Losses: mse: 31.5934, mae: 3.0014, huber: 2.5884, swd: 2.1731, ept: 221.1198
    Epoch [30/50], Test Losses: mse: 24.0708, mae: 2.5391, huber: 2.1343, swd: 1.6043, ept: 239.2503
      Epoch 30 composite train-obj: 1.248612
            No improvement (2.5884), counter 4/5
    Epoch [31/50], Train Losses: mse: 10.5023, mae: 1.5829, huber: 1.2097, swd: 0.8360, ept: 273.9696
    Epoch [31/50], Val Losses: mse: 38.1460, mae: 3.3161, huber: 2.8994, swd: 2.2642, ept: 214.3215
    Epoch [31/50], Test Losses: mse: 24.7323, mae: 2.5744, huber: 2.1697, swd: 1.6301, ept: 240.0954
      Epoch 31 composite train-obj: 1.209740
    Epoch [31/50], Test Losses: mse: 23.6303, mae: 2.5411, huber: 2.1359, swd: 1.6712, ept: 233.6618
    Best round's Test MSE: 23.6585, MAE: 2.5424, SWD: 1.6659
    Best round's Validation MSE: 30.5676, MAE: 2.9814, SWD: 2.1676
    Best round's Test verification MSE : 23.6303, MAE: 2.5411, SWD: 1.6712
    Time taken: 119.51 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1522)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 24.5022 ± 1.6421
      mae: 2.6826 ± 0.1333
      huber: 2.2679 ± 0.1270
      swd: 1.8761 ± 0.1965
      ept: 222.3781 ± 9.7945
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 32.1757 ± 1.8821
      mae: 3.1425 ± 0.1173
      huber: 2.7201 ± 0.1118
      swd: 2.2250 ± 0.0943
      ept: 203.4624 ± 8.2019
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 281.26 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1522
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

## Ab data lengths: 

length 25000 -> 52000: longer, better geometry

Experiment Summary (ACL_lorenz_seq336_pred336_20250514_0254)
Number of runs: 3
Seeds: [1955, 7, 20]

Test Performance at Best Validation (mean ± std):
  mse: 11.4145 ± 1.2282
  mae: 1.6146 ± 0.1313
  huber: 1.2538 ± 0.1206
  swd: 0.9656 ± 0.1587
  ept: 274.5805 ± 7.4334
  count: 36.0000 ± 0.0000

# Ab core components

## Ab: no rotation


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128, 
    ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=True, 
    ablate_shift_inside_scale=False,
    householder_reflects_latent = 2,
    householder_reflects_data = 4,
    mixing_strategy='delay_only', 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
    ablate_no_rotation=True, ### HERE
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 124.7099, mae: 8.2677, huber: 7.7850, swd: 60.7525, ept: 13.6469
    Epoch [1/50], Val Losses: mse: 97.9352, mae: 7.2435, huber: 6.7647, swd: 34.8426, ept: 15.1067
    Epoch [1/50], Test Losses: mse: 93.7106, mae: 6.8865, huber: 6.4103, swd: 34.4916, ept: 15.5017
      Epoch 1 composite train-obj: 7.785026
            Val objective improved inf → 6.7647, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 75.1776, mae: 5.9506, huber: 5.4821, swd: 20.0889, ept: 24.3271
    Epoch [2/50], Val Losses: mse: 65.1510, mae: 5.5488, huber: 5.0820, swd: 13.9334, ept: 28.1277
    Epoch [2/50], Test Losses: mse: 61.3655, mae: 5.2352, huber: 4.7707, swd: 13.8166, ept: 39.9763
      Epoch 2 composite train-obj: 5.482070
            Val objective improved 6.7647 → 5.0820, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 49.0697, mae: 4.5047, huber: 4.0503, swd: 8.3324, ept: 56.2923
    Epoch [3/50], Val Losses: mse: 57.0550, mae: 5.0211, huber: 4.5610, swd: 7.4572, ept: 60.0849
    Epoch [3/50], Test Losses: mse: 47.2635, mae: 4.4596, huber: 4.0021, swd: 7.5083, ept: 77.3517
      Epoch 3 composite train-obj: 4.050287
            Val objective improved 5.0820 → 4.5610, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.2370, mae: 3.7777, huber: 3.3324, swd: 5.1769, ept: 106.2269
    Epoch [4/50], Val Losses: mse: 46.6304, mae: 4.4180, huber: 3.9654, swd: 5.8671, ept: 97.3666
    Epoch [4/50], Test Losses: mse: 38.3493, mae: 3.8785, huber: 3.4294, swd: 5.4865, ept: 114.9188
      Epoch 4 composite train-obj: 3.332403
            Val objective improved 4.5610 → 3.9654, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 30.6680, mae: 3.3346, huber: 2.8966, swd: 3.7442, ept: 172.6064
    Epoch [5/50], Val Losses: mse: 46.5378, mae: 4.2924, huber: 3.8411, swd: 4.4209, ept: 152.5179
    Epoch [5/50], Test Losses: mse: 36.6559, mae: 3.7372, huber: 3.2888, swd: 4.4805, ept: 167.5359
      Epoch 5 composite train-obj: 2.896566
            Val objective improved 3.9654 → 3.8411, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 27.4364, mae: 3.0741, huber: 2.6422, swd: 2.9693, ept: 207.5503
    Epoch [6/50], Val Losses: mse: 46.6213, mae: 4.2584, huber: 3.8086, swd: 3.9430, ept: 161.1402
    Epoch [6/50], Test Losses: mse: 34.4238, mae: 3.5555, huber: 3.1110, swd: 3.7663, ept: 177.9581
      Epoch 6 composite train-obj: 2.642246
            Val objective improved 3.8411 → 3.8086, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 25.1424, mae: 2.8883, huber: 2.4613, swd: 2.5584, ept: 217.4520
    Epoch [7/50], Val Losses: mse: 46.0052, mae: 4.1398, huber: 3.6928, swd: 3.8091, ept: 173.6244
    Epoch [7/50], Test Losses: mse: 34.5427, mae: 3.4974, huber: 3.0554, swd: 3.4911, ept: 185.1395
      Epoch 7 composite train-obj: 2.461278
            Val objective improved 3.8086 → 3.6928, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 23.0720, mae: 2.7127, huber: 2.2916, swd: 2.2283, ept: 226.2323
    Epoch [8/50], Val Losses: mse: 44.9124, mae: 4.0385, huber: 3.5943, swd: 3.5892, ept: 173.6041
    Epoch [8/50], Test Losses: mse: 35.2271, mae: 3.4622, huber: 3.0234, swd: 3.0366, ept: 193.4345
      Epoch 8 composite train-obj: 2.291561
            Val objective improved 3.6928 → 3.5943, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 21.7274, mae: 2.6013, huber: 2.1835, swd: 2.0464, ept: 232.9582
    Epoch [9/50], Val Losses: mse: 43.4741, mae: 3.9770, huber: 3.5346, swd: 3.5837, ept: 179.6349
    Epoch [9/50], Test Losses: mse: 34.4478, mae: 3.4195, huber: 2.9834, swd: 2.8994, ept: 197.3189
      Epoch 9 composite train-obj: 2.183454
            Val objective improved 3.5943 → 3.5346, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 20.5142, mae: 2.5001, huber: 2.0856, swd: 1.8850, ept: 237.8641
    Epoch [10/50], Val Losses: mse: 44.4899, mae: 3.9622, huber: 3.5204, swd: 3.2318, ept: 182.1034
    Epoch [10/50], Test Losses: mse: 34.7545, mae: 3.3905, huber: 2.9561, swd: 2.6906, ept: 203.7205
      Epoch 10 composite train-obj: 2.085608
            Val objective improved 3.5346 → 3.5204, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 19.4845, mae: 2.4076, huber: 1.9968, swd: 1.7178, ept: 243.5146
    Epoch [11/50], Val Losses: mse: 45.1387, mae: 3.9417, huber: 3.5019, swd: 2.6475, ept: 188.5768
    Epoch [11/50], Test Losses: mse: 32.3787, mae: 3.2617, huber: 2.8295, swd: 2.3904, ept: 207.9697
      Epoch 11 composite train-obj: 1.996763
            Val objective improved 3.5204 → 3.5019, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 18.7758, mae: 2.3378, huber: 1.9301, swd: 1.6440, ept: 248.1857
    Epoch [12/50], Val Losses: mse: 44.5295, mae: 3.8856, huber: 3.4521, swd: 3.0022, ept: 188.1506
    Epoch [12/50], Test Losses: mse: 33.3355, mae: 3.2432, huber: 2.8162, swd: 2.4543, ept: 211.1890
      Epoch 12 composite train-obj: 1.930091
            Val objective improved 3.5019 → 3.4521, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 17.7985, mae: 2.2565, huber: 1.8519, swd: 1.5348, ept: 251.9465
    Epoch [13/50], Val Losses: mse: 41.1662, mae: 3.7754, huber: 3.3389, swd: 2.9656, ept: 191.7715
    Epoch [13/50], Test Losses: mse: 32.8408, mae: 3.2434, huber: 2.8131, swd: 2.5711, ept: 210.5105
      Epoch 13 composite train-obj: 1.851894
            Val objective improved 3.4521 → 3.3389, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 17.5071, mae: 2.2183, huber: 1.8161, swd: 1.4648, ept: 254.3921
    Epoch [14/50], Val Losses: mse: 43.3090, mae: 3.8282, huber: 3.3952, swd: 3.1183, ept: 191.5918
    Epoch [14/50], Test Losses: mse: 31.1315, mae: 3.1735, huber: 2.7452, swd: 2.5715, ept: 210.1814
      Epoch 14 composite train-obj: 1.816112
            No improvement (3.3952), counter 1/5
    Epoch [15/50], Train Losses: mse: 16.7637, mae: 2.1517, huber: 1.7518, swd: 1.3928, ept: 258.1838
    Epoch [15/50], Val Losses: mse: 43.2192, mae: 3.8066, huber: 3.3733, swd: 2.8576, ept: 194.1580
    Epoch [15/50], Test Losses: mse: 32.2217, mae: 3.1463, huber: 2.7204, swd: 2.2089, ept: 218.1910
      Epoch 15 composite train-obj: 1.751808
            No improvement (3.3733), counter 2/5
    Epoch [16/50], Train Losses: mse: 16.4159, mae: 2.1187, huber: 1.7204, swd: 1.3539, ept: 259.9652
    Epoch [16/50], Val Losses: mse: 37.9497, mae: 3.5848, huber: 3.1533, swd: 3.1064, ept: 199.3945
    Epoch [16/50], Test Losses: mse: 32.1153, mae: 3.1874, huber: 2.7605, swd: 2.7108, ept: 218.7652
      Epoch 16 composite train-obj: 1.720375
            Val objective improved 3.3389 → 3.1533, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.8297, mae: 2.0622, huber: 1.6667, swd: 1.2853, ept: 263.0574
    Epoch [17/50], Val Losses: mse: 42.9780, mae: 3.7852, huber: 3.3522, swd: 2.9597, ept: 200.3600
    Epoch [17/50], Test Losses: mse: 30.9522, mae: 3.1144, huber: 2.6891, swd: 2.3002, ept: 220.6640
      Epoch 17 composite train-obj: 1.666669
            No improvement (3.3522), counter 1/5
    Epoch [18/50], Train Losses: mse: 15.3386, mae: 2.0150, huber: 1.6216, swd: 1.2263, ept: 265.2973
    Epoch [18/50], Val Losses: mse: 41.8788, mae: 3.7212, huber: 3.2917, swd: 2.7855, ept: 202.5914
    Epoch [18/50], Test Losses: mse: 29.6743, mae: 2.9952, huber: 2.5777, swd: 2.1528, ept: 230.1609
      Epoch 18 composite train-obj: 1.621566
            No improvement (3.2917), counter 2/5
    Epoch [19/50], Train Losses: mse: 14.8752, mae: 1.9677, huber: 1.5768, swd: 1.1922, ept: 267.9484
    Epoch [19/50], Val Losses: mse: 40.5992, mae: 3.6792, huber: 3.2464, swd: 2.7138, ept: 201.1954
    Epoch [19/50], Test Losses: mse: 32.7303, mae: 3.1544, huber: 2.7306, swd: 1.9976, ept: 218.8717
      Epoch 19 composite train-obj: 1.576836
            No improvement (3.2464), counter 3/5
    Epoch [20/50], Train Losses: mse: 14.6182, mae: 1.9476, huber: 1.5574, swd: 1.1572, ept: 268.9228
    Epoch [20/50], Val Losses: mse: 42.5810, mae: 3.7276, huber: 3.2956, swd: 2.4341, ept: 206.9637
    Epoch [20/50], Test Losses: mse: 30.6194, mae: 3.0013, huber: 2.5803, swd: 1.7813, ept: 229.9887
      Epoch 20 composite train-obj: 1.557366
            No improvement (3.2956), counter 4/5
    Epoch [21/50], Train Losses: mse: 14.0038, mae: 1.8919, huber: 1.5047, swd: 1.0843, ept: 271.5706
    Epoch [21/50], Val Losses: mse: 39.2977, mae: 3.5635, huber: 3.1370, swd: 2.5675, ept: 209.7388
    Epoch [21/50], Test Losses: mse: 29.2106, mae: 2.9626, huber: 2.5455, swd: 2.0714, ept: 230.2080
      Epoch 21 composite train-obj: 1.504678
            Val objective improved 3.1533 → 3.1370, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.7863, mae: 1.8665, huber: 1.4803, swd: 1.0880, ept: 273.1483
    Epoch [22/50], Val Losses: mse: 40.0643, mae: 3.6557, huber: 3.2247, swd: 2.6149, ept: 206.7524
    Epoch [22/50], Test Losses: mse: 28.6652, mae: 2.9438, huber: 2.5238, swd: 2.0367, ept: 229.5937
      Epoch 22 composite train-obj: 1.480318
            No improvement (3.2247), counter 1/5
    Epoch [23/50], Train Losses: mse: 13.6609, mae: 1.8527, huber: 1.4676, swd: 1.0646, ept: 273.9919
    Epoch [23/50], Val Losses: mse: 37.7157, mae: 3.5336, huber: 3.1049, swd: 2.5540, ept: 209.4538
    Epoch [23/50], Test Losses: mse: 28.3822, mae: 2.9209, huber: 2.5042, swd: 2.0920, ept: 226.7275
      Epoch 23 composite train-obj: 1.467580
            Val objective improved 3.1370 → 3.1049, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 12.9927, mae: 1.7877, huber: 1.4064, swd: 0.9902, ept: 276.7423
    Epoch [24/50], Val Losses: mse: 38.7431, mae: 3.5536, huber: 3.1236, swd: 2.5526, ept: 206.2428
    Epoch [24/50], Test Losses: mse: 29.1042, mae: 2.9319, huber: 2.5134, swd: 1.9463, ept: 231.6437
      Epoch 24 composite train-obj: 1.406358
            No improvement (3.1236), counter 1/5
    Epoch [25/50], Train Losses: mse: 12.6584, mae: 1.7563, huber: 1.3763, swd: 0.9715, ept: 278.5496
    Epoch [25/50], Val Losses: mse: 36.9833, mae: 3.4913, huber: 3.0650, swd: 2.6831, ept: 207.9913
    Epoch [25/50], Test Losses: mse: 31.3910, mae: 3.0607, huber: 2.6431, swd: 2.0483, ept: 227.2469
      Epoch 25 composite train-obj: 1.376262
            Val objective improved 3.1049 → 3.0650, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 12.9661, mae: 1.7823, huber: 1.4008, swd: 0.9832, ept: 277.7898
    Epoch [26/50], Val Losses: mse: 40.4175, mae: 3.6101, huber: 3.1832, swd: 2.3615, ept: 210.2257
    Epoch [26/50], Test Losses: mse: 28.7672, mae: 2.8985, huber: 2.4829, swd: 1.9071, ept: 233.6401
      Epoch 26 composite train-obj: 1.400776
            No improvement (3.1832), counter 1/5
    Epoch [27/50], Train Losses: mse: 12.0139, mae: 1.6939, huber: 1.3178, swd: 0.9033, ept: 281.2615
    Epoch [27/50], Val Losses: mse: 41.8268, mae: 3.6250, huber: 3.1994, swd: 2.2694, ept: 213.4479
    Epoch [27/50], Test Losses: mse: 27.4625, mae: 2.8449, huber: 2.4313, swd: 1.8002, ept: 233.8488
      Epoch 27 composite train-obj: 1.317791
            No improvement (3.1994), counter 2/5
    Epoch [28/50], Train Losses: mse: 12.2590, mae: 1.7137, huber: 1.3363, swd: 0.9118, ept: 281.2266
    Epoch [28/50], Val Losses: mse: 38.8393, mae: 3.5110, huber: 3.0875, swd: 2.5361, ept: 211.6723
    Epoch [28/50], Test Losses: mse: 28.9800, mae: 2.8912, huber: 2.4781, swd: 1.8297, ept: 231.4973
      Epoch 28 composite train-obj: 1.336264
            No improvement (3.0875), counter 3/5
    Epoch [29/50], Train Losses: mse: 12.2385, mae: 1.7063, huber: 1.3292, swd: 0.9177, ept: 281.9898
    Epoch [29/50], Val Losses: mse: 37.8844, mae: 3.4931, huber: 3.0655, swd: 2.4363, ept: 212.1040
    Epoch [29/50], Test Losses: mse: 29.5887, mae: 2.9403, huber: 2.5243, swd: 1.9199, ept: 232.7476
      Epoch 29 composite train-obj: 1.329176
            No improvement (3.0655), counter 4/5
    Epoch [30/50], Train Losses: mse: 12.5169, mae: 1.7218, huber: 1.3445, swd: 0.9115, ept: 282.2241
    Epoch [30/50], Val Losses: mse: 37.7088, mae: 3.4758, huber: 3.0492, swd: 2.4925, ept: 213.7222
    Epoch [30/50], Test Losses: mse: 27.6782, mae: 2.8690, huber: 2.4535, swd: 1.9039, ept: 235.0543
      Epoch 30 composite train-obj: 1.344498
            Val objective improved 3.0650 → 3.0492, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 11.6721, mae: 1.6507, huber: 1.2774, swd: 0.8622, ept: 284.4236
    Epoch [31/50], Val Losses: mse: 36.4173, mae: 3.4547, huber: 3.0308, swd: 2.6787, ept: 212.6937
    Epoch [31/50], Test Losses: mse: 29.9222, mae: 2.9864, huber: 2.5716, swd: 1.9960, ept: 232.0231
      Epoch 31 composite train-obj: 1.277357
            Val objective improved 3.0492 → 3.0308, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 12.2437, mae: 1.6956, huber: 1.3197, swd: 0.8901, ept: 283.5085
    Epoch [32/50], Val Losses: mse: 40.9891, mae: 3.6696, huber: 3.2428, swd: 2.5322, ept: 206.5674
    Epoch [32/50], Test Losses: mse: 29.3927, mae: 2.9864, huber: 2.5709, swd: 1.9482, ept: 226.7852
      Epoch 32 composite train-obj: 1.319699
            No improvement (3.2428), counter 1/5
    Epoch [33/50], Train Losses: mse: 11.7161, mae: 1.6514, huber: 1.2780, swd: 0.8546, ept: 285.0776
    Epoch [33/50], Val Losses: mse: 38.9139, mae: 3.5402, huber: 3.1142, swd: 2.4705, ept: 211.0093
    Epoch [33/50], Test Losses: mse: 27.8965, mae: 2.8576, huber: 2.4451, swd: 1.9480, ept: 237.4038
      Epoch 33 composite train-obj: 1.278020
            No improvement (3.1142), counter 2/5
    Epoch [34/50], Train Losses: mse: 11.2455, mae: 1.5989, huber: 1.2295, swd: 0.8214, ept: 287.2845
    Epoch [34/50], Val Losses: mse: 38.2141, mae: 3.5415, huber: 3.1122, swd: 2.4513, ept: 210.9785
    Epoch [34/50], Test Losses: mse: 27.7584, mae: 2.8525, huber: 2.4383, swd: 1.7912, ept: 235.6345
      Epoch 34 composite train-obj: 1.229510
            No improvement (3.1122), counter 3/5
    Epoch [35/50], Train Losses: mse: 10.6772, mae: 1.5454, huber: 1.1789, swd: 0.7632, ept: 289.1303
    Epoch [35/50], Val Losses: mse: 40.0865, mae: 3.5347, huber: 3.1121, swd: 2.3036, ept: 215.3687
    Epoch [35/50], Test Losses: mse: 27.6931, mae: 2.7890, huber: 2.3820, swd: 1.6355, ept: 242.5365
      Epoch 35 composite train-obj: 1.178940
            No improvement (3.1121), counter 4/5
    Epoch [36/50], Train Losses: mse: 11.5631, mae: 1.6283, huber: 1.2560, swd: 0.8318, ept: 286.7468
    Epoch [36/50], Val Losses: mse: 37.9874, mae: 3.4969, huber: 3.0669, swd: 2.2538, ept: 214.4599
    Epoch [36/50], Test Losses: mse: 27.2506, mae: 2.8668, huber: 2.4498, swd: 1.8857, ept: 235.4820
      Epoch 36 composite train-obj: 1.255975
    Epoch [36/50], Test Losses: mse: 29.9214, mae: 2.9864, huber: 2.5716, swd: 1.9965, ept: 232.0463
    Best round's Test MSE: 29.9222, MAE: 2.9864, SWD: 1.9960
    Best round's Validation MSE: 36.4173, MAE: 3.4547, SWD: 2.6787
    Best round's Test verification MSE : 29.9214, MAE: 2.9864, SWD: 1.9965
    Time taken: 112.62 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 116.4537, mae: 7.9202, huber: 7.4388, swd: 52.7966, ept: 12.0245
    Epoch [1/50], Val Losses: mse: 93.6468, mae: 6.9770, huber: 6.4998, swd: 24.8970, ept: 13.5967
    Epoch [1/50], Test Losses: mse: 89.6298, mae: 6.6468, huber: 6.1717, swd: 25.2075, ept: 13.6636
      Epoch 1 composite train-obj: 7.438782
            Val objective improved inf → 6.4998, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 75.0511, mae: 5.8569, huber: 5.3901, swd: 16.8935, ept: 16.9560
    Epoch [2/50], Val Losses: mse: 67.1757, mae: 5.6151, huber: 5.1486, swd: 12.4848, ept: 25.0938
    Epoch [2/50], Test Losses: mse: 62.8156, mae: 5.2500, huber: 4.7867, swd: 12.8155, ept: 28.0425
      Epoch 2 composite train-obj: 5.390069
            Val objective improved 6.4998 → 5.1486, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 51.1079, mae: 4.5693, huber: 4.1144, swd: 8.3969, ept: 41.4063
    Epoch [3/50], Val Losses: mse: 54.6584, mae: 4.8947, huber: 4.4354, swd: 7.9312, ept: 44.2150
    Epoch [3/50], Test Losses: mse: 48.4297, mae: 4.4575, huber: 4.0016, swd: 7.9300, ept: 53.2767
      Epoch 3 composite train-obj: 4.114448
            Val objective improved 5.1486 → 4.4354, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 38.0662, mae: 3.8010, huber: 3.3567, swd: 5.1925, ept: 72.2991
    Epoch [4/50], Val Losses: mse: 45.4883, mae: 4.3930, huber: 3.9391, swd: 5.9998, ept: 110.3198
    Epoch [4/50], Test Losses: mse: 39.5859, mae: 3.9720, huber: 3.5208, swd: 5.8181, ept: 125.9782
      Epoch 4 composite train-obj: 3.356667
            Val objective improved 4.4354 → 3.9391, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 31.0125, mae: 3.3399, huber: 2.9029, swd: 3.7526, ept: 161.1652
    Epoch [5/50], Val Losses: mse: 42.8295, mae: 4.1760, huber: 3.7254, swd: 5.2106, ept: 139.9454
    Epoch [5/50], Test Losses: mse: 36.0368, mae: 3.6883, huber: 3.2426, swd: 4.8145, ept: 153.0162
      Epoch 5 composite train-obj: 2.902872
            Val objective improved 3.9391 → 3.7254, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 28.3001, mae: 3.1285, huber: 2.6959, swd: 3.0998, ept: 189.3211
    Epoch [6/50], Val Losses: mse: 46.5055, mae: 4.2290, huber: 3.7808, swd: 3.9052, ept: 157.7354
    Epoch [6/50], Test Losses: mse: 35.1256, mae: 3.5863, huber: 3.1438, swd: 3.7678, ept: 174.2808
      Epoch 6 composite train-obj: 2.695897
            No improvement (3.7808), counter 1/5
    Epoch [7/50], Train Losses: mse: 25.1187, mae: 2.8820, huber: 2.4558, swd: 2.5472, ept: 206.7485
    Epoch [7/50], Val Losses: mse: 46.8486, mae: 4.2004, huber: 3.7551, swd: 3.6186, ept: 164.5203
    Epoch [7/50], Test Losses: mse: 34.3154, mae: 3.4599, huber: 3.0218, swd: 3.4430, ept: 187.4965
      Epoch 7 composite train-obj: 2.455761
            No improvement (3.7551), counter 2/5
    Epoch [8/50], Train Losses: mse: 24.2828, mae: 2.8021, huber: 2.3779, swd: 2.3383, ept: 212.5644
    Epoch [8/50], Val Losses: mse: 44.5669, mae: 4.1237, huber: 3.6781, swd: 3.9246, ept: 163.0707
    Epoch [8/50], Test Losses: mse: 35.4867, mae: 3.5108, huber: 3.0725, swd: 3.5448, ept: 183.4078
      Epoch 8 composite train-obj: 2.377895
            Val objective improved 3.7254 → 3.6781, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 22.4018, mae: 2.6573, huber: 2.2381, swd: 2.1027, ept: 219.3339
    Epoch [9/50], Val Losses: mse: 43.7226, mae: 3.9817, huber: 3.5395, swd: 3.0849, ept: 169.1464
    Epoch [9/50], Test Losses: mse: 34.0705, mae: 3.3654, huber: 2.9320, swd: 3.1585, ept: 188.1513
      Epoch 9 composite train-obj: 2.238056
            Val objective improved 3.6781 → 3.5395, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 20.2281, mae: 2.4744, huber: 2.0614, swd: 1.8416, ept: 237.2363
    Epoch [10/50], Val Losses: mse: 39.6386, mae: 3.7241, huber: 3.2830, swd: 2.6584, ept: 185.8225
    Epoch [10/50], Test Losses: mse: 31.9438, mae: 3.2313, huber: 2.7990, swd: 2.6874, ept: 202.7955
      Epoch 10 composite train-obj: 2.061357
            Val objective improved 3.5395 → 3.2830, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 20.0174, mae: 2.4474, huber: 2.0351, swd: 1.7873, ept: 242.6770
    Epoch [11/50], Val Losses: mse: 43.3068, mae: 3.8918, huber: 3.4536, swd: 3.2256, ept: 185.9376
    Epoch [11/50], Test Losses: mse: 31.3604, mae: 3.1969, huber: 2.7676, swd: 3.1148, ept: 205.3569
      Epoch 11 composite train-obj: 2.035068
            No improvement (3.4536), counter 1/5
    Epoch [12/50], Train Losses: mse: 18.4441, mae: 2.3269, huber: 1.9187, swd: 1.6089, ept: 248.2199
    Epoch [12/50], Val Losses: mse: 42.1726, mae: 3.7889, huber: 3.3536, swd: 2.7567, ept: 192.7009
    Epoch [12/50], Test Losses: mse: 33.4110, mae: 3.2441, huber: 2.8179, swd: 2.6736, ept: 212.1315
      Epoch 12 composite train-obj: 1.918691
            No improvement (3.3536), counter 2/5
    Epoch [13/50], Train Losses: mse: 18.7898, mae: 2.3350, huber: 1.9272, swd: 1.6147, ept: 249.4496
    Epoch [13/50], Val Losses: mse: 42.4238, mae: 3.8521, huber: 3.4110, swd: 2.6976, ept: 189.3342
    Epoch [13/50], Test Losses: mse: 32.1422, mae: 3.2321, huber: 2.8006, swd: 2.7327, ept: 207.6349
      Epoch 13 composite train-obj: 1.927249
            No improvement (3.4110), counter 3/5
    Epoch [14/50], Train Losses: mse: 17.8409, mae: 2.2502, huber: 1.8462, swd: 1.5031, ept: 253.6697
    Epoch [14/50], Val Losses: mse: 40.2587, mae: 3.6746, huber: 3.2401, swd: 2.5433, ept: 195.2865
    Epoch [14/50], Test Losses: mse: 32.5599, mae: 3.1851, huber: 2.7595, swd: 2.4512, ept: 212.6827
      Epoch 14 composite train-obj: 1.846188
            Val objective improved 3.2830 → 3.2401, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 16.8596, mae: 2.1661, huber: 1.7654, swd: 1.4116, ept: 257.7231
    Epoch [15/50], Val Losses: mse: 39.4547, mae: 3.6901, huber: 3.2517, swd: 2.6736, ept: 199.2788
    Epoch [15/50], Test Losses: mse: 30.3426, mae: 3.0547, huber: 2.6281, swd: 2.1946, ept: 220.4537
      Epoch 15 composite train-obj: 1.765376
            No improvement (3.2517), counter 1/5
    Epoch [16/50], Train Losses: mse: 15.1746, mae: 2.0113, huber: 1.6182, swd: 1.2307, ept: 263.5243
    Epoch [16/50], Val Losses: mse: 38.5673, mae: 3.5357, huber: 3.1040, swd: 2.5024, ept: 201.9972
    Epoch [16/50], Test Losses: mse: 28.8571, mae: 2.9707, huber: 2.5477, swd: 2.2426, ept: 225.1440
      Epoch 16 composite train-obj: 1.618204
            Val objective improved 3.2401 → 3.1040, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 16.4853, mae: 2.1219, huber: 1.7230, swd: 1.3459, ept: 260.7367
    Epoch [17/50], Val Losses: mse: 39.8250, mae: 3.7059, huber: 3.2660, swd: 2.7095, ept: 197.7600
    Epoch [17/50], Test Losses: mse: 32.8635, mae: 3.2192, huber: 2.7870, swd: 2.4211, ept: 219.3142
      Epoch 17 composite train-obj: 1.722960
            No improvement (3.2660), counter 1/5
    Epoch [18/50], Train Losses: mse: 16.3911, mae: 2.1190, huber: 1.7198, swd: 1.3505, ept: 261.5348
    Epoch [18/50], Val Losses: mse: 38.7671, mae: 3.6428, huber: 3.2097, swd: 2.3856, ept: 198.4752
    Epoch [18/50], Test Losses: mse: 28.2498, mae: 2.9651, huber: 2.5443, swd: 2.2337, ept: 221.3083
      Epoch 18 composite train-obj: 1.719805
            No improvement (3.2097), counter 2/5
    Epoch [19/50], Train Losses: mse: 15.3964, mae: 2.0183, huber: 1.6243, swd: 1.2329, ept: 265.5426
    Epoch [19/50], Val Losses: mse: 36.5728, mae: 3.4559, huber: 3.0261, swd: 2.4517, ept: 207.2711
    Epoch [19/50], Test Losses: mse: 26.7247, mae: 2.8696, huber: 2.4502, swd: 2.1508, ept: 227.0885
      Epoch 19 composite train-obj: 1.624287
            Val objective improved 3.1040 → 3.0261, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 13.8851, mae: 1.8867, huber: 1.4991, swd: 1.0802, ept: 270.7590
    Epoch [20/50], Val Losses: mse: 35.7887, mae: 3.4054, huber: 2.9785, swd: 2.5227, ept: 208.0060
    Epoch [20/50], Test Losses: mse: 26.8879, mae: 2.8398, huber: 2.4230, swd: 2.0140, ept: 227.7591
      Epoch 20 composite train-obj: 1.499146
            Val objective improved 3.0261 → 2.9785, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 13.5664, mae: 1.8578, huber: 1.4715, swd: 1.0571, ept: 271.9645
    Epoch [21/50], Val Losses: mse: 37.2225, mae: 3.4634, huber: 3.0366, swd: 2.4206, ept: 213.2773
    Epoch [21/50], Test Losses: mse: 28.6316, mae: 2.9234, huber: 2.5052, swd: 2.0912, ept: 228.7821
      Epoch 21 composite train-obj: 1.471477
            No improvement (3.0366), counter 1/5
    Epoch [22/50], Train Losses: mse: 14.0025, mae: 1.8749, huber: 1.4889, swd: 1.0896, ept: 272.5050
    Epoch [22/50], Val Losses: mse: 39.3734, mae: 3.5342, huber: 3.1040, swd: 2.2007, ept: 207.0798
    Epoch [22/50], Test Losses: mse: 31.0860, mae: 3.0371, huber: 2.6154, swd: 2.1027, ept: 227.9758
      Epoch 22 composite train-obj: 1.488858
            No improvement (3.1040), counter 2/5
    Epoch [23/50], Train Losses: mse: 13.4142, mae: 1.8414, huber: 1.4558, swd: 1.0599, ept: 274.2592
    Epoch [23/50], Val Losses: mse: 37.8739, mae: 3.5058, huber: 3.0771, swd: 2.5126, ept: 212.4347
    Epoch [23/50], Test Losses: mse: 27.2554, mae: 2.8842, huber: 2.4660, swd: 2.0840, ept: 233.0554
      Epoch 23 composite train-obj: 1.455808
            No improvement (3.0771), counter 3/5
    Epoch [24/50], Train Losses: mse: 12.8784, mae: 1.7835, huber: 1.4014, swd: 0.9923, ept: 276.5170
    Epoch [24/50], Val Losses: mse: 36.2595, mae: 3.3982, huber: 2.9714, swd: 2.0543, ept: 216.5433
    Epoch [24/50], Test Losses: mse: 27.2583, mae: 2.8376, huber: 2.4221, swd: 1.8403, ept: 230.9371
      Epoch 24 composite train-obj: 1.401388
            Val objective improved 2.9785 → 2.9714, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 12.7854, mae: 1.7656, huber: 1.3846, swd: 0.9669, ept: 278.0116
    Epoch [25/50], Val Losses: mse: 38.7227, mae: 3.4950, huber: 3.0670, swd: 2.3285, ept: 214.7771
    Epoch [25/50], Test Losses: mse: 28.4759, mae: 2.9100, huber: 2.4917, swd: 2.0759, ept: 236.0568
      Epoch 25 composite train-obj: 1.384618
            No improvement (3.0670), counter 1/5
    Epoch [26/50], Train Losses: mse: 12.6220, mae: 1.7588, huber: 1.3775, swd: 0.9472, ept: 278.5593
    Epoch [26/50], Val Losses: mse: 39.4637, mae: 3.5295, huber: 3.1047, swd: 2.2283, ept: 216.8317
    Epoch [26/50], Test Losses: mse: 27.5758, mae: 2.8360, huber: 2.4229, swd: 1.9530, ept: 236.4246
      Epoch 26 composite train-obj: 1.377500
            No improvement (3.1047), counter 2/5
    Epoch [27/50], Train Losses: mse: 12.3511, mae: 1.7161, huber: 1.3390, swd: 0.9250, ept: 280.4510
    Epoch [27/50], Val Losses: mse: 41.6534, mae: 3.5804, huber: 3.1564, swd: 1.9406, ept: 215.7543
    Epoch [27/50], Test Losses: mse: 29.0289, mae: 2.9150, huber: 2.5013, swd: 1.8803, ept: 233.5273
      Epoch 27 composite train-obj: 1.339028
            No improvement (3.1564), counter 3/5
    Epoch [28/50], Train Losses: mse: 12.6131, mae: 1.7477, huber: 1.3678, swd: 0.9450, ept: 279.3381
    Epoch [28/50], Val Losses: mse: 39.9830, mae: 3.5214, huber: 3.0947, swd: 2.2741, ept: 215.5547
    Epoch [28/50], Test Losses: mse: 29.4645, mae: 2.9764, huber: 2.5569, swd: 2.0989, ept: 231.7938
      Epoch 28 composite train-obj: 1.367816
            No improvement (3.0947), counter 4/5
    Epoch [29/50], Train Losses: mse: 11.8360, mae: 1.6741, huber: 1.2983, swd: 0.8868, ept: 282.4248
    Epoch [29/50], Val Losses: mse: 38.0964, mae: 3.4842, huber: 3.0553, swd: 2.3179, ept: 215.8563
    Epoch [29/50], Test Losses: mse: 26.9400, mae: 2.8225, huber: 2.4072, swd: 2.0098, ept: 232.7045
      Epoch 29 composite train-obj: 1.298324
    Epoch [29/50], Test Losses: mse: 27.2636, mae: 2.8380, huber: 2.4225, swd: 1.8411, ept: 230.9245
    Best round's Test MSE: 27.2583, MAE: 2.8376, SWD: 1.8403
    Best round's Validation MSE: 36.2595, MAE: 3.3982, SWD: 2.0543
    Best round's Test verification MSE : 27.2636, MAE: 2.8380, SWD: 1.8411
    Time taken: 94.09 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 118.3525, mae: 8.0434, huber: 7.5614, swd: 58.0230, ept: 13.1611
    Epoch [1/50], Val Losses: mse: 90.1686, mae: 6.8994, huber: 6.4221, swd: 30.9165, ept: 16.2459
    Epoch [1/50], Test Losses: mse: 86.5304, mae: 6.5817, huber: 6.1067, swd: 30.7060, ept: 18.6140
      Epoch 1 composite train-obj: 7.561419
            Val objective improved inf → 6.4221, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 69.5540, mae: 5.6747, huber: 5.2086, swd: 18.1324, ept: 29.4434
    Epoch [2/50], Val Losses: mse: 62.5281, mae: 5.4168, huber: 4.9515, swd: 13.0161, ept: 39.2881
    Epoch [2/50], Test Losses: mse: 58.7126, mae: 5.1028, huber: 4.6403, swd: 13.1596, ept: 49.2426
      Epoch 2 composite train-obj: 5.208573
            Val objective improved 6.4221 → 4.9515, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 47.1420, mae: 4.3954, huber: 3.9429, swd: 8.1949, ept: 56.5158
    Epoch [3/50], Val Losses: mse: 50.2863, mae: 4.6769, huber: 4.2213, swd: 8.0847, ept: 74.2065
    Epoch [3/50], Test Losses: mse: 45.7415, mae: 4.3054, huber: 3.8542, swd: 8.1786, ept: 87.4888
      Epoch 3 composite train-obj: 3.942938
            Val objective improved 4.9515 → 4.2213, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.2545, mae: 3.7069, huber: 3.2638, swd: 5.4059, ept: 122.3645
    Epoch [4/50], Val Losses: mse: 47.2603, mae: 4.3887, huber: 3.9362, swd: 5.7882, ept: 124.9583
    Epoch [4/50], Test Losses: mse: 39.0507, mae: 3.8703, huber: 3.4229, swd: 5.4465, ept: 138.2467
      Epoch 4 composite train-obj: 3.263835
            Val objective improved 4.2213 → 3.9362, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 31.0860, mae: 3.3063, huber: 2.8720, swd: 3.9999, ept: 153.5209
    Epoch [5/50], Val Losses: mse: 46.9062, mae: 4.3071, huber: 3.8587, swd: 5.8397, ept: 130.2025
    Epoch [5/50], Test Losses: mse: 38.4902, mae: 3.7765, huber: 3.3336, swd: 5.4419, ept: 145.0298
      Epoch 5 composite train-obj: 2.871998
            Val objective improved 3.9362 → 3.8587, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 27.7452, mae: 3.0682, huber: 2.6386, swd: 3.3085, ept: 162.4901
    Epoch [6/50], Val Losses: mse: 49.5174, mae: 4.3473, huber: 3.9001, swd: 4.8203, ept: 140.1398
    Epoch [6/50], Test Losses: mse: 36.5603, mae: 3.6379, huber: 3.1966, swd: 4.6180, ept: 157.8262
      Epoch 6 composite train-obj: 2.638592
            No improvement (3.9001), counter 1/5
    Epoch [7/50], Train Losses: mse: 25.3084, mae: 2.8720, huber: 2.4471, swd: 2.7926, ept: 176.6814
    Epoch [7/50], Val Losses: mse: 42.8846, mae: 4.0013, huber: 3.5585, swd: 3.8492, ept: 154.7863
    Epoch [7/50], Test Losses: mse: 34.6410, mae: 3.4550, huber: 3.0198, swd: 3.6281, ept: 169.1793
      Epoch 7 composite train-obj: 2.447124
            Val objective improved 3.8587 → 3.5585, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 23.2043, mae: 2.7100, huber: 2.2892, swd: 2.4485, ept: 191.1174
    Epoch [8/50], Val Losses: mse: 45.5193, mae: 4.0539, huber: 3.6124, swd: 3.7565, ept: 156.2512
    Epoch [8/50], Test Losses: mse: 34.9339, mae: 3.4008, huber: 2.9667, swd: 3.2016, ept: 177.0868
      Epoch 8 composite train-obj: 2.289204
            No improvement (3.6124), counter 1/5
    Epoch [9/50], Train Losses: mse: 22.0949, mae: 2.6109, huber: 2.1935, swd: 2.2263, ept: 196.4489
    Epoch [9/50], Val Losses: mse: 46.3966, mae: 4.0605, huber: 3.6211, swd: 3.8974, ept: 160.0747
    Epoch [9/50], Test Losses: mse: 32.6592, mae: 3.2864, huber: 2.8548, swd: 3.3266, ept: 180.1959
      Epoch 9 composite train-obj: 2.193453
            No improvement (3.6211), counter 2/5
    Epoch [10/50], Train Losses: mse: 20.3155, mae: 2.4692, huber: 2.0561, swd: 2.0123, ept: 209.8767
    Epoch [10/50], Val Losses: mse: 46.6399, mae: 4.0473, huber: 3.6087, swd: 3.5069, ept: 168.2201
    Epoch [10/50], Test Losses: mse: 32.7730, mae: 3.2711, huber: 2.8412, swd: 3.2892, ept: 188.6489
      Epoch 10 composite train-obj: 2.056072
            No improvement (3.6087), counter 3/5
    Epoch [11/50], Train Losses: mse: 19.6571, mae: 2.4099, huber: 1.9988, swd: 1.9058, ept: 218.8178
    Epoch [11/50], Val Losses: mse: 43.9198, mae: 3.8723, huber: 3.4389, swd: 3.5900, ept: 173.9688
    Epoch [11/50], Test Losses: mse: 30.5634, mae: 3.1152, huber: 2.6908, swd: 2.9723, ept: 195.5641
      Epoch 11 composite train-obj: 1.998753
            Val objective improved 3.5585 → 3.4389, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 18.2420, mae: 2.2812, huber: 1.8759, swd: 1.7040, ept: 225.1435
    Epoch [12/50], Val Losses: mse: 41.4230, mae: 3.7404, huber: 3.3074, swd: 3.2363, ept: 184.3918
    Epoch [12/50], Test Losses: mse: 28.5638, mae: 3.0009, huber: 2.5768, swd: 2.6958, ept: 201.8098
      Epoch 12 composite train-obj: 1.875905
            Val objective improved 3.4389 → 3.3074, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 17.6005, mae: 2.2256, huber: 1.8219, swd: 1.6199, ept: 228.1068
    Epoch [13/50], Val Losses: mse: 39.5654, mae: 3.6441, huber: 3.2107, swd: 3.0212, ept: 181.8269
    Epoch [13/50], Test Losses: mse: 29.9741, mae: 3.0646, huber: 2.6399, swd: 2.6493, ept: 202.3838
      Epoch 13 composite train-obj: 1.821865
            Val objective improved 3.3074 → 3.2107, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 16.8284, mae: 2.1612, huber: 1.7602, swd: 1.5365, ept: 249.1993
    Epoch [14/50], Val Losses: mse: 38.9724, mae: 3.6188, huber: 3.1884, swd: 3.2615, ept: 200.5758
    Epoch [14/50], Test Losses: mse: 29.1743, mae: 3.0368, huber: 2.6141, swd: 2.9990, ept: 215.1096
      Epoch 14 composite train-obj: 1.760178
            Val objective improved 3.2107 → 3.1884, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 15.9504, mae: 2.0846, huber: 1.6871, swd: 1.4326, ept: 261.0417
    Epoch [15/50], Val Losses: mse: 40.6777, mae: 3.6901, huber: 3.2551, swd: 3.1343, ept: 201.9119
    Epoch [15/50], Test Losses: mse: 29.6813, mae: 3.0998, huber: 2.6716, swd: 2.5798, ept: 216.9784
      Epoch 15 composite train-obj: 1.687118
            No improvement (3.2551), counter 1/5
    Epoch [16/50], Train Losses: mse: 15.1889, mae: 2.0159, huber: 1.6212, swd: 1.3457, ept: 264.3123
    Epoch [16/50], Val Losses: mse: 37.3989, mae: 3.4841, huber: 3.0559, swd: 2.8066, ept: 208.7440
    Epoch [16/50], Test Losses: mse: 26.8814, mae: 2.8413, huber: 2.4240, swd: 2.3411, ept: 229.0660
      Epoch 16 composite train-obj: 1.621176
            Val objective improved 3.1884 → 3.0559, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.3560, mae: 2.0246, huber: 1.6293, swd: 1.3491, ept: 264.8538
    Epoch [17/50], Val Losses: mse: 38.6254, mae: 3.5877, huber: 3.1543, swd: 3.2952, ept: 206.5495
    Epoch [17/50], Test Losses: mse: 31.8341, mae: 3.1544, huber: 2.7278, swd: 2.7344, ept: 224.4624
      Epoch 17 composite train-obj: 1.629267
            No improvement (3.1543), counter 1/5
    Epoch [18/50], Train Losses: mse: 14.8174, mae: 1.9804, huber: 1.5864, swd: 1.2902, ept: 267.4831
    Epoch [18/50], Val Losses: mse: 38.3380, mae: 3.4896, huber: 3.0640, swd: 2.9928, ept: 208.1624
    Epoch [18/50], Test Losses: mse: 29.1175, mae: 2.9568, huber: 2.5385, swd: 2.5902, ept: 230.3630
      Epoch 18 composite train-obj: 1.586386
            No improvement (3.0640), counter 2/5
    Epoch [19/50], Train Losses: mse: 14.3836, mae: 1.9356, huber: 1.5446, swd: 1.2611, ept: 269.0588
    Epoch [19/50], Val Losses: mse: 40.4863, mae: 3.5844, huber: 3.1598, swd: 3.0135, ept: 208.7634
    Epoch [19/50], Test Losses: mse: 28.6378, mae: 2.9185, huber: 2.5028, swd: 2.5770, ept: 230.9052
      Epoch 19 composite train-obj: 1.544642
            No improvement (3.1598), counter 3/5
    Epoch [20/50], Train Losses: mse: 13.0765, mae: 1.8165, huber: 1.4315, swd: 1.1272, ept: 274.2791
    Epoch [20/50], Val Losses: mse: 40.5835, mae: 3.5561, huber: 3.1343, swd: 2.8605, ept: 212.2867
    Epoch [20/50], Test Losses: mse: 29.2214, mae: 2.9399, huber: 2.5251, swd: 2.5273, ept: 231.6449
      Epoch 20 composite train-obj: 1.431471
            No improvement (3.1343), counter 4/5
    Epoch [21/50], Train Losses: mse: 13.5488, mae: 1.8432, huber: 1.4570, swd: 1.1385, ept: 274.8510
    Epoch [21/50], Val Losses: mse: 40.3963, mae: 3.5753, huber: 3.1464, swd: 2.7840, ept: 206.1161
    Epoch [21/50], Test Losses: mse: 27.7383, mae: 2.8888, huber: 2.4699, swd: 2.3928, ept: 232.2799
      Epoch 21 composite train-obj: 1.457007
    Epoch [21/50], Test Losses: mse: 26.8874, mae: 2.8415, huber: 2.4242, swd: 2.3408, ept: 229.0923
    Best round's Test MSE: 26.8814, MAE: 2.8413, SWD: 2.3411
    Best round's Validation MSE: 37.3989, MAE: 3.4841, SWD: 2.8066
    Best round's Test verification MSE : 26.8874, MAE: 2.8415, SWD: 2.3408
    Time taken: 66.82 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1530)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 28.0206 ± 1.3534
      mae: 2.8885 ± 0.0693
      huber: 2.4726 ± 0.0700
      swd: 2.0591 ± 0.2093
      ept: 230.6754 ± 1.2213
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 36.6919 ± 0.5041
      mae: 3.4457 ± 0.0356
      huber: 3.0194 ± 0.0354
      swd: 2.5132 ± 0.3287
      ept: 212.6603 ± 3.1842
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 273.63 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1530
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

## AB: Koopman Components

### AB: No Koopman


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128, 
    # ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=True, 
    ablate_shift_inside_scale=False,
    householder_reflects_latent = 2,
    householder_reflects_data = 4,
    mixing_strategy='delay_only', 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
    ablate_no_koopman=True, ### HERE
    ablate_no_shift_in_z_push=True, ### HERE
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 76.7488, mae: 6.6491, huber: 6.1683, swd: 45.4359, ept: 30.7262
    Epoch [1/50], Val Losses: mse: 61.5423, mae: 5.9935, huber: 5.5153, swd: 29.4883, ept: 32.3976
    Epoch [1/50], Test Losses: mse: 57.0725, mae: 5.6041, huber: 5.1292, swd: 27.5431, ept: 36.3589
      Epoch 1 composite train-obj: 6.168337
            Val objective improved inf → 5.5153, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.6636, mae: 4.9890, huber: 4.5216, swd: 19.0372, ept: 84.4600
    Epoch [2/50], Val Losses: mse: 52.4617, mae: 5.1611, huber: 4.6931, swd: 16.9679, ept: 84.0272
    Epoch [2/50], Test Losses: mse: 48.2696, mae: 4.8156, huber: 4.3511, swd: 17.2022, ept: 80.7164
      Epoch 2 composite train-obj: 4.521572
            Val objective improved 5.5153 → 4.6931, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 42.5778, mae: 4.3268, huber: 3.8706, swd: 11.4934, ept: 127.1008
    Epoch [3/50], Val Losses: mse: 49.2826, mae: 4.8224, huber: 4.3602, swd: 11.2456, ept: 104.2680
    Epoch [3/50], Test Losses: mse: 43.5245, mae: 4.4153, huber: 3.9563, swd: 11.7836, ept: 104.7926
      Epoch 3 composite train-obj: 3.870605
            Val objective improved 4.6931 → 4.3602, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.1551, mae: 3.8668, huber: 3.4184, swd: 7.3959, ept: 152.0583
    Epoch [4/50], Val Losses: mse: 45.1998, mae: 4.4511, huber: 3.9950, swd: 8.3107, ept: 117.9338
    Epoch [4/50], Test Losses: mse: 40.2963, mae: 4.1072, huber: 3.6549, swd: 8.1421, ept: 122.8826
      Epoch 4 composite train-obj: 3.418412
            Val objective improved 4.3602 → 3.9950, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 34.0013, mae: 3.5997, huber: 3.1562, swd: 5.6372, ept: 167.6836
    Epoch [5/50], Val Losses: mse: 42.5223, mae: 4.2697, huber: 3.8147, swd: 7.5601, ept: 133.0058
    Epoch [5/50], Test Losses: mse: 38.0556, mae: 3.9258, huber: 3.4741, swd: 6.8111, ept: 138.1855
      Epoch 5 composite train-obj: 3.156244
            Val objective improved 3.9950 → 3.8147, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.9951, mae: 3.3388, huber: 2.9024, swd: 4.4366, ept: 181.7720
    Epoch [6/50], Val Losses: mse: 42.5542, mae: 4.2172, huber: 3.7649, swd: 6.4320, ept: 138.5536
    Epoch [6/50], Test Losses: mse: 36.4530, mae: 3.7783, huber: 3.3309, swd: 5.5881, ept: 145.3370
      Epoch 6 composite train-obj: 2.902391
            Val objective improved 3.8147 → 3.7649, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 29.0043, mae: 3.1823, huber: 2.7491, swd: 3.7804, ept: 188.7514
    Epoch [7/50], Val Losses: mse: 45.1175, mae: 4.3142, huber: 3.8612, swd: 5.5005, ept: 139.4229
    Epoch [7/50], Test Losses: mse: 34.7984, mae: 3.6685, huber: 3.2221, swd: 4.4480, ept: 152.8786
      Epoch 7 composite train-obj: 2.749071
            No improvement (3.8612), counter 1/5
    Epoch [8/50], Train Losses: mse: 27.1618, mae: 3.0310, huber: 2.6014, swd: 3.2912, ept: 195.6630
    Epoch [8/50], Val Losses: mse: 42.4448, mae: 4.0781, huber: 3.6317, swd: 4.7963, ept: 152.9180
    Epoch [8/50], Test Losses: mse: 33.7267, mae: 3.5248, huber: 3.0835, swd: 3.8420, ept: 164.2914
      Epoch 8 composite train-obj: 2.601401
            Val objective improved 3.7649 → 3.6317, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 25.4012, mae: 2.8918, huber: 2.4658, swd: 2.8998, ept: 202.0794
    Epoch [9/50], Val Losses: mse: 45.0026, mae: 4.1435, huber: 3.7001, swd: 4.7612, ept: 160.8904
    Epoch [9/50], Test Losses: mse: 32.5912, mae: 3.3952, huber: 2.9579, swd: 3.5032, ept: 175.8298
      Epoch 9 composite train-obj: 2.465821
            No improvement (3.7001), counter 1/5
    Epoch [10/50], Train Losses: mse: 23.8262, mae: 2.7641, huber: 2.3418, swd: 2.5857, ept: 208.1040
    Epoch [10/50], Val Losses: mse: 45.7401, mae: 4.2004, huber: 3.7535, swd: 4.2205, ept: 153.5150
    Epoch [10/50], Test Losses: mse: 33.7528, mae: 3.4958, huber: 3.0557, swd: 3.4408, ept: 170.5647
      Epoch 10 composite train-obj: 2.341811
            No improvement (3.7535), counter 2/5
    Epoch [11/50], Train Losses: mse: 22.6170, mae: 2.6746, huber: 2.2543, swd: 2.3527, ept: 213.0592
    Epoch [11/50], Val Losses: mse: 44.6166, mae: 4.1399, huber: 3.6926, swd: 4.4553, ept: 159.1268
    Epoch [11/50], Test Losses: mse: 33.5071, mae: 3.4591, huber: 3.0185, swd: 3.1598, ept: 174.6042
      Epoch 11 composite train-obj: 2.254349
            No improvement (3.6926), counter 3/5
    Epoch [12/50], Train Losses: mse: 21.4844, mae: 2.5846, huber: 2.1671, swd: 2.1652, ept: 217.1752
    Epoch [12/50], Val Losses: mse: 43.4535, mae: 3.9933, huber: 3.5531, swd: 4.1365, ept: 167.2809
    Epoch [12/50], Test Losses: mse: 31.8824, mae: 3.2887, huber: 2.8554, swd: 3.2494, ept: 185.9078
      Epoch 12 composite train-obj: 2.167110
            Val objective improved 3.6317 → 3.5531, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 20.3374, mae: 2.4849, huber: 2.0710, swd: 1.9925, ept: 222.2458
    Epoch [13/50], Val Losses: mse: 42.9192, mae: 3.9633, huber: 3.5225, swd: 3.9732, ept: 169.6275
    Epoch [13/50], Test Losses: mse: 31.2584, mae: 3.2516, huber: 2.8179, swd: 3.1636, ept: 188.6874
      Epoch 13 composite train-obj: 2.071045
            Val objective improved 3.5531 → 3.5225, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 19.7405, mae: 2.4357, huber: 2.0235, swd: 1.8631, ept: 225.0439
    Epoch [14/50], Val Losses: mse: 44.0568, mae: 3.9298, huber: 3.4908, swd: 3.4346, ept: 171.8614
    Epoch [14/50], Test Losses: mse: 30.8561, mae: 3.2080, huber: 2.7751, swd: 2.6660, ept: 188.2542
      Epoch 14 composite train-obj: 2.023491
            Val objective improved 3.5225 → 3.4908, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 18.7556, mae: 2.3577, huber: 1.9473, swd: 1.7383, ept: 229.4071
    Epoch [15/50], Val Losses: mse: 43.2194, mae: 3.9160, huber: 3.4784, swd: 4.0483, ept: 178.4994
    Epoch [15/50], Test Losses: mse: 29.6988, mae: 3.1124, huber: 2.6840, swd: 2.9653, ept: 197.9823
      Epoch 15 composite train-obj: 1.947304
            Val objective improved 3.4908 → 3.4784, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 17.9685, mae: 2.2882, huber: 1.8807, swd: 1.6190, ept: 233.0430
    Epoch [16/50], Val Losses: mse: 42.5871, mae: 3.8191, huber: 3.3839, swd: 3.5318, ept: 180.6194
    Epoch [16/50], Test Losses: mse: 30.2070, mae: 3.0992, huber: 2.6718, swd: 2.4194, ept: 198.6633
      Epoch 16 composite train-obj: 1.880653
            Val objective improved 3.4784 → 3.3839, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 17.1733, mae: 2.2111, huber: 1.8067, swd: 1.5120, ept: 237.7064
    Epoch [17/50], Val Losses: mse: 43.5633, mae: 3.8573, huber: 3.4226, swd: 3.5231, ept: 183.7300
    Epoch [17/50], Test Losses: mse: 30.4348, mae: 3.0654, huber: 2.6402, swd: 2.4088, ept: 206.8530
      Epoch 17 composite train-obj: 1.806727
            No improvement (3.4226), counter 1/5
    Epoch [18/50], Train Losses: mse: 16.2725, mae: 2.1335, huber: 1.7328, swd: 1.4275, ept: 242.1550
    Epoch [18/50], Val Losses: mse: 42.8818, mae: 3.8044, huber: 3.3698, swd: 3.2943, ept: 183.4845
    Epoch [18/50], Test Losses: mse: 29.9513, mae: 3.0448, huber: 2.6200, swd: 2.3614, ept: 205.2584
      Epoch 18 composite train-obj: 1.732848
            Val objective improved 3.3839 → 3.3698, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 15.8331, mae: 2.0993, huber: 1.6993, swd: 1.3421, ept: 244.0304
    Epoch [19/50], Val Losses: mse: 43.2341, mae: 3.8252, huber: 3.3904, swd: 3.1569, ept: 182.6900
    Epoch [19/50], Test Losses: mse: 32.7548, mae: 3.1747, huber: 2.7490, swd: 2.3149, ept: 202.4830
      Epoch 19 composite train-obj: 1.699316
            No improvement (3.3904), counter 1/5
    Epoch [20/50], Train Losses: mse: 15.2754, mae: 2.0471, huber: 1.6493, swd: 1.3026, ept: 247.2984
    Epoch [20/50], Val Losses: mse: 44.5602, mae: 3.8306, huber: 3.4004, swd: 3.2885, ept: 186.0821
    Epoch [20/50], Test Losses: mse: 31.1836, mae: 3.0317, huber: 2.6117, swd: 2.2091, ept: 213.0477
      Epoch 20 composite train-obj: 1.649334
            No improvement (3.4004), counter 2/5
    Epoch [21/50], Train Losses: mse: 14.6942, mae: 1.9944, huber: 1.5990, swd: 1.2305, ept: 250.7353
    Epoch [21/50], Val Losses: mse: 41.4632, mae: 3.6755, huber: 3.2438, swd: 3.1130, ept: 192.6785
    Epoch [21/50], Test Losses: mse: 29.4594, mae: 2.9690, huber: 2.5463, swd: 2.0721, ept: 212.9681
      Epoch 21 composite train-obj: 1.599025
            Val objective improved 3.3698 → 3.2438, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 14.3786, mae: 1.9578, huber: 1.5641, swd: 1.1987, ept: 253.9247
    Epoch [22/50], Val Losses: mse: 41.1378, mae: 3.6531, huber: 3.2233, swd: 3.3180, ept: 197.1092
    Epoch [22/50], Test Losses: mse: 28.6697, mae: 2.9200, huber: 2.4984, swd: 2.0868, ept: 218.6128
      Epoch 22 composite train-obj: 1.564066
            Val objective improved 3.2438 → 3.2233, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 14.0386, mae: 1.9342, huber: 1.5415, swd: 1.1788, ept: 254.7113
    Epoch [23/50], Val Losses: mse: 42.1157, mae: 3.7141, huber: 3.2838, swd: 3.5017, ept: 192.6994
    Epoch [23/50], Test Losses: mse: 30.3540, mae: 3.0079, huber: 2.5874, swd: 2.5641, ept: 219.7250
      Epoch 23 composite train-obj: 1.541488
            No improvement (3.2838), counter 1/5
    Epoch [24/50], Train Losses: mse: 13.6281, mae: 1.8859, huber: 1.4967, swd: 1.1210, ept: 257.7573
    Epoch [24/50], Val Losses: mse: 40.1954, mae: 3.5804, huber: 3.1533, swd: 3.2894, ept: 198.4938
    Epoch [24/50], Test Losses: mse: 27.3893, mae: 2.8129, huber: 2.3967, swd: 2.0875, ept: 223.4404
      Epoch 24 composite train-obj: 1.496692
            Val objective improved 3.2233 → 3.1533, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 13.0142, mae: 1.8326, huber: 1.4455, swd: 1.0768, ept: 260.4156
    Epoch [25/50], Val Losses: mse: 40.7144, mae: 3.6099, huber: 3.1846, swd: 3.0904, ept: 193.5973
    Epoch [25/50], Test Losses: mse: 31.4293, mae: 2.9749, huber: 2.5604, swd: 2.1687, ept: 222.7292
      Epoch 25 composite train-obj: 1.445483
            No improvement (3.1846), counter 1/5
    Epoch [26/50], Train Losses: mse: 12.9194, mae: 1.8096, huber: 1.4248, swd: 1.0502, ept: 262.2584
    Epoch [26/50], Val Losses: mse: 38.7366, mae: 3.5283, huber: 3.1010, swd: 3.2327, ept: 196.2254
    Epoch [26/50], Test Losses: mse: 28.9535, mae: 2.8859, huber: 2.4693, swd: 2.1437, ept: 223.2512
      Epoch 26 composite train-obj: 1.424768
            Val objective improved 3.1533 → 3.1010, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 12.6541, mae: 1.8020, huber: 1.4161, swd: 1.0392, ept: 263.3623
    Epoch [27/50], Val Losses: mse: 41.4541, mae: 3.6466, huber: 3.2187, swd: 3.0795, ept: 195.2209
    Epoch [27/50], Test Losses: mse: 27.9814, mae: 2.8374, huber: 2.4207, swd: 2.0818, ept: 223.1580
      Epoch 27 composite train-obj: 1.416084
            No improvement (3.2187), counter 1/5
    Epoch [28/50], Train Losses: mse: 12.4950, mae: 1.7807, huber: 1.3962, swd: 1.0085, ept: 264.6324
    Epoch [28/50], Val Losses: mse: 39.7675, mae: 3.5370, huber: 3.1135, swd: 3.0342, ept: 201.0653
    Epoch [28/50], Test Losses: mse: 27.6961, mae: 2.7631, huber: 2.3526, swd: 1.8919, ept: 229.5799
      Epoch 28 composite train-obj: 1.396240
            No improvement (3.1135), counter 2/5
    Epoch [29/50], Train Losses: mse: 12.1139, mae: 1.7508, huber: 1.3674, swd: 0.9723, ept: 265.8443
    Epoch [29/50], Val Losses: mse: 39.2447, mae: 3.4868, huber: 3.0633, swd: 3.0132, ept: 203.0051
    Epoch [29/50], Test Losses: mse: 27.8710, mae: 2.7845, huber: 2.3719, swd: 1.9806, ept: 229.8288
      Epoch 29 composite train-obj: 1.367431
            Val objective improved 3.1010 → 3.0633, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 11.9862, mae: 1.7315, huber: 1.3498, swd: 0.9639, ept: 267.8800
    Epoch [30/50], Val Losses: mse: 42.6103, mae: 3.7158, huber: 3.2848, swd: 2.7990, ept: 189.4555
    Epoch [30/50], Test Losses: mse: 29.0026, mae: 2.8921, huber: 2.4740, swd: 1.7638, ept: 219.7368
      Epoch 30 composite train-obj: 1.349762
            No improvement (3.2848), counter 1/5
    Epoch [31/50], Train Losses: mse: 11.5553, mae: 1.7031, huber: 1.3220, swd: 0.9108, ept: 268.3649
    Epoch [31/50], Val Losses: mse: 39.7175, mae: 3.5539, huber: 3.1285, swd: 2.9631, ept: 196.8580
    Epoch [31/50], Test Losses: mse: 28.8892, mae: 2.8611, huber: 2.4465, swd: 1.9483, ept: 221.6254
      Epoch 31 composite train-obj: 1.322014
            No improvement (3.1285), counter 2/5
    Epoch [32/50], Train Losses: mse: 11.3513, mae: 1.6721, huber: 1.2937, swd: 0.8966, ept: 270.5410
    Epoch [32/50], Val Losses: mse: 40.1961, mae: 3.5469, huber: 3.1212, swd: 2.7438, ept: 199.4756
    Epoch [32/50], Test Losses: mse: 26.9825, mae: 2.7396, huber: 2.3271, swd: 1.8002, ept: 229.7608
      Epoch 32 composite train-obj: 1.293726
            No improvement (3.1212), counter 3/5
    Epoch [33/50], Train Losses: mse: 11.0869, mae: 1.6426, huber: 1.2664, swd: 0.8710, ept: 272.2393
    Epoch [33/50], Val Losses: mse: 39.7675, mae: 3.5225, huber: 3.0997, swd: 3.0898, ept: 201.2978
    Epoch [33/50], Test Losses: mse: 27.1804, mae: 2.7444, huber: 2.3328, swd: 1.9826, ept: 229.8652
      Epoch 33 composite train-obj: 1.266383
            No improvement (3.0997), counter 4/5
    Epoch [34/50], Train Losses: mse: 10.8825, mae: 1.6261, huber: 1.2502, swd: 0.8707, ept: 273.3993
    Epoch [34/50], Val Losses: mse: 37.2046, mae: 3.4116, huber: 2.9896, swd: 2.9877, ept: 203.4492
    Epoch [34/50], Test Losses: mse: 28.1607, mae: 2.7762, huber: 2.3674, swd: 1.9148, ept: 229.6915
      Epoch 34 composite train-obj: 1.250222
            Val objective improved 3.0633 → 2.9896, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 10.6194, mae: 1.5962, huber: 1.2227, swd: 0.8361, ept: 274.8312
    Epoch [35/50], Val Losses: mse: 39.2558, mae: 3.4886, huber: 3.0650, swd: 3.0517, ept: 203.5673
    Epoch [35/50], Test Losses: mse: 27.9053, mae: 2.7339, huber: 2.3245, swd: 1.6878, ept: 234.8305
      Epoch 35 composite train-obj: 1.222705
            No improvement (3.0650), counter 1/5
    Epoch [36/50], Train Losses: mse: 10.5385, mae: 1.5903, huber: 1.2170, swd: 0.8261, ept: 275.0942
    Epoch [36/50], Val Losses: mse: 39.3860, mae: 3.4740, huber: 3.0525, swd: 2.6240, ept: 202.3048
    Epoch [36/50], Test Losses: mse: 26.9095, mae: 2.7043, huber: 2.2953, swd: 1.6468, ept: 230.7987
      Epoch 36 composite train-obj: 1.217050
            No improvement (3.0525), counter 2/5
    Epoch [37/50], Train Losses: mse: 10.1096, mae: 1.5488, huber: 1.1780, swd: 0.7895, ept: 277.3457
    Epoch [37/50], Val Losses: mse: 38.7409, mae: 3.4552, huber: 3.0348, swd: 2.7847, ept: 202.2789
    Epoch [37/50], Test Losses: mse: 27.9300, mae: 2.7336, huber: 2.3274, swd: 1.8938, ept: 231.9117
      Epoch 37 composite train-obj: 1.178043
            No improvement (3.0348), counter 3/5
    Epoch [38/50], Train Losses: mse: 10.1642, mae: 1.5501, huber: 1.1795, swd: 0.7889, ept: 277.4779
    Epoch [38/50], Val Losses: mse: 36.4610, mae: 3.3435, huber: 2.9223, swd: 2.6310, ept: 206.8658
    Epoch [38/50], Test Losses: mse: 29.2339, mae: 2.8104, huber: 2.4022, swd: 1.7207, ept: 229.1525
      Epoch 38 composite train-obj: 1.179532
            Val objective improved 2.9896 → 2.9223, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 9.6955, mae: 1.5108, huber: 1.1426, swd: 0.7572, ept: 279.3059
    Epoch [39/50], Val Losses: mse: 38.3095, mae: 3.4723, huber: 3.0456, swd: 2.7279, ept: 199.0575
    Epoch [39/50], Test Losses: mse: 27.1327, mae: 2.7333, huber: 2.3205, swd: 1.6443, ept: 230.1391
      Epoch 39 composite train-obj: 1.142596
            No improvement (3.0456), counter 1/5
    Epoch [40/50], Train Losses: mse: 9.9592, mae: 1.5243, huber: 1.1560, swd: 0.7666, ept: 278.9173
    Epoch [40/50], Val Losses: mse: 37.5559, mae: 3.3503, huber: 2.9327, swd: 2.5755, ept: 205.9350
    Epoch [40/50], Test Losses: mse: 26.6263, mae: 2.6487, huber: 2.2452, swd: 1.7189, ept: 238.6820
      Epoch 40 composite train-obj: 1.156012
            No improvement (2.9327), counter 2/5
    Epoch [41/50], Train Losses: mse: 10.1050, mae: 1.5418, huber: 1.1715, swd: 0.8022, ept: 279.5968
    Epoch [41/50], Val Losses: mse: 37.7240, mae: 3.4112, huber: 2.9906, swd: 2.6480, ept: 200.0210
    Epoch [41/50], Test Losses: mse: 29.0568, mae: 2.8065, huber: 2.3980, swd: 1.7703, ept: 227.6381
      Epoch 41 composite train-obj: 1.171524
            No improvement (2.9906), counter 3/5
    Epoch [42/50], Train Losses: mse: 9.8818, mae: 1.5326, huber: 1.1620, swd: 0.7780, ept: 279.1603
    Epoch [42/50], Val Losses: mse: 40.5907, mae: 3.6113, huber: 3.1859, swd: 3.1546, ept: 192.8296
    Epoch [42/50], Test Losses: mse: 26.1595, mae: 2.7314, huber: 2.3182, swd: 1.8435, ept: 227.5146
      Epoch 42 composite train-obj: 1.161991
            No improvement (3.1859), counter 4/5
    Epoch [43/50], Train Losses: mse: 9.7292, mae: 1.5048, huber: 1.1371, swd: 0.7584, ept: 280.7013
    Epoch [43/50], Val Losses: mse: 39.8888, mae: 3.5337, huber: 3.1082, swd: 2.5826, ept: 196.9184
    Epoch [43/50], Test Losses: mse: 26.5067, mae: 2.7114, huber: 2.3007, swd: 1.5917, ept: 228.7726
      Epoch 43 composite train-obj: 1.137096
    Epoch [43/50], Test Losses: mse: 29.2353, mae: 2.8104, huber: 2.4023, swd: 1.7199, ept: 229.1634
    Best round's Test MSE: 29.2339, MAE: 2.8104, SWD: 1.7207
    Best round's Validation MSE: 36.4610, MAE: 3.3435, SWD: 2.6310
    Best round's Test verification MSE : 29.2353, MAE: 2.8104, SWD: 1.7199
    Time taken: 187.19 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 78.2055, mae: 6.7089, huber: 6.2281, swd: 46.7759, ept: 29.2573
    Epoch [1/50], Val Losses: mse: 60.8827, mae: 5.8913, huber: 5.4145, swd: 28.1914, ept: 40.4435
    Epoch [1/50], Test Losses: mse: 56.0469, mae: 5.4994, huber: 5.0260, swd: 28.5573, ept: 39.2415
      Epoch 1 composite train-obj: 6.228071
            Val objective improved inf → 5.4145, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.7536, mae: 5.0108, huber: 4.5428, swd: 18.5020, ept: 83.9885
    Epoch [2/50], Val Losses: mse: 52.1846, mae: 5.1683, huber: 4.6998, swd: 15.6105, ept: 76.6997
    Epoch [2/50], Test Losses: mse: 47.4024, mae: 4.8111, huber: 4.3462, swd: 17.2048, ept: 72.9983
      Epoch 2 composite train-obj: 4.542824
            Val objective improved 5.4145 → 4.6998, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 41.5117, mae: 4.2907, huber: 3.8336, swd: 9.8889, ept: 128.1478
    Epoch [3/50], Val Losses: mse: 49.4664, mae: 4.8563, huber: 4.3932, swd: 10.1115, ept: 103.0201
    Epoch [3/50], Test Losses: mse: 42.9027, mae: 4.4135, huber: 3.9530, swd: 10.8627, ept: 107.0717
      Epoch 3 composite train-obj: 3.833600
            Val objective improved 4.6998 → 4.3932, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.0955, mae: 3.9005, huber: 3.4505, swd: 6.7722, ept: 151.5218
    Epoch [4/50], Val Losses: mse: 43.9520, mae: 4.4052, huber: 3.9490, swd: 7.9991, ept: 122.2567
    Epoch [4/50], Test Losses: mse: 40.8019, mae: 4.1401, huber: 3.6865, swd: 7.9304, ept: 127.2535
      Epoch 4 composite train-obj: 3.450528
            Val objective improved 4.3932 → 3.9490, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.8886, mae: 3.6131, huber: 3.1697, swd: 5.3266, ept: 167.7480
    Epoch [5/50], Val Losses: mse: 42.1899, mae: 4.2791, huber: 3.8230, swd: 6.9780, ept: 129.6817
    Epoch [5/50], Test Losses: mse: 37.6033, mae: 3.9257, huber: 3.4734, swd: 6.5384, ept: 138.4183
      Epoch 5 composite train-obj: 3.169662
            Val objective improved 3.9490 → 3.8230, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.7242, mae: 3.4210, huber: 2.9821, swd: 4.4783, ept: 177.7776
    Epoch [6/50], Val Losses: mse: 41.9473, mae: 4.1538, huber: 3.7029, swd: 6.3295, ept: 140.4748
    Epoch [6/50], Test Losses: mse: 36.3723, mae: 3.7695, huber: 3.3224, swd: 5.6343, ept: 149.2967
      Epoch 6 composite train-obj: 2.982132
            Val objective improved 3.8230 → 3.7029, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 29.7809, mae: 3.2615, huber: 2.8258, swd: 3.8057, ept: 185.5805
    Epoch [7/50], Val Losses: mse: 41.4530, mae: 4.0906, huber: 3.6412, swd: 4.9182, ept: 146.4325
    Epoch [7/50], Test Losses: mse: 34.7588, mae: 3.6665, huber: 3.2209, swd: 4.3643, ept: 157.8317
      Epoch 7 composite train-obj: 2.825839
            Val objective improved 3.7029 → 3.6412, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 28.3514, mae: 3.1391, huber: 2.7066, swd: 3.3346, ept: 191.5136
    Epoch [8/50], Val Losses: mse: 44.9443, mae: 4.2267, huber: 3.7773, swd: 4.9945, ept: 149.1260
    Epoch [8/50], Test Losses: mse: 35.5173, mae: 3.6649, huber: 3.2197, swd: 4.3166, ept: 160.6628
      Epoch 8 composite train-obj: 2.706553
            No improvement (3.7773), counter 1/5
    Epoch [9/50], Train Losses: mse: 26.4994, mae: 2.9924, huber: 2.5630, swd: 2.9833, ept: 197.7583
    Epoch [9/50], Val Losses: mse: 40.2269, mae: 3.9491, huber: 3.5022, swd: 4.2118, ept: 156.5066
    Epoch [9/50], Test Losses: mse: 33.1132, mae: 3.5029, huber: 3.0608, swd: 3.9467, ept: 167.0378
      Epoch 9 composite train-obj: 2.563022
            Val objective improved 3.6412 → 3.5022, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 25.2449, mae: 2.8897, huber: 2.4634, swd: 2.7234, ept: 202.1684
    Epoch [10/50], Val Losses: mse: 42.2721, mae: 3.9956, huber: 3.5513, swd: 4.4848, ept: 158.9756
    Epoch [10/50], Test Losses: mse: 33.2878, mae: 3.5090, huber: 3.0690, swd: 4.4752, ept: 167.5384
      Epoch 10 composite train-obj: 2.463353
            No improvement (3.5513), counter 1/5
    Epoch [11/50], Train Losses: mse: 24.2442, mae: 2.8062, huber: 2.3822, swd: 2.5407, ept: 206.2801
    Epoch [11/50], Val Losses: mse: 42.9849, mae: 4.0166, huber: 3.5734, swd: 4.5194, ept: 158.9260
    Epoch [11/50], Test Losses: mse: 32.2957, mae: 3.3713, huber: 2.9345, swd: 3.5421, ept: 176.5329
      Epoch 11 composite train-obj: 2.382191
            No improvement (3.5734), counter 2/5
    Epoch [12/50], Train Losses: mse: 22.6670, mae: 2.6772, huber: 2.2571, swd: 2.2985, ept: 211.9165
    Epoch [12/50], Val Losses: mse: 41.6102, mae: 3.9506, huber: 3.5065, swd: 4.1607, ept: 160.2855
    Epoch [12/50], Test Losses: mse: 32.4399, mae: 3.3811, huber: 2.9438, swd: 3.1953, ept: 176.0867
      Epoch 12 composite train-obj: 2.257072
            No improvement (3.5065), counter 3/5
    Epoch [13/50], Train Losses: mse: 21.4049, mae: 2.5751, huber: 2.1584, swd: 2.1106, ept: 216.2365
    Epoch [13/50], Val Losses: mse: 43.2339, mae: 3.9887, huber: 3.5468, swd: 4.0890, ept: 163.3328
    Epoch [13/50], Test Losses: mse: 31.6928, mae: 3.3101, huber: 2.8762, swd: 3.0533, ept: 179.9299
      Epoch 13 composite train-obj: 2.158433
            No improvement (3.5468), counter 4/5
    Epoch [14/50], Train Losses: mse: 21.0719, mae: 2.5527, huber: 2.1362, swd: 2.0338, ept: 217.7855
    Epoch [14/50], Val Losses: mse: 41.7127, mae: 3.8625, huber: 3.4227, swd: 3.9864, ept: 168.3882
    Epoch [14/50], Test Losses: mse: 33.2471, mae: 3.3586, huber: 2.9248, swd: 3.4031, ept: 182.5904
      Epoch 14 composite train-obj: 2.136230
            Val objective improved 3.5022 → 3.4227, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 20.0678, mae: 2.4595, huber: 2.0466, swd: 1.9138, ept: 223.6814
    Epoch [15/50], Val Losses: mse: 40.7595, mae: 3.7892, huber: 3.3520, swd: 3.5302, ept: 172.0990
    Epoch [15/50], Test Losses: mse: 33.2569, mae: 3.3272, huber: 2.8952, swd: 2.8976, ept: 186.8371
      Epoch 15 composite train-obj: 2.046586
            Val objective improved 3.4227 → 3.3520, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 18.9462, mae: 2.3630, huber: 1.9540, swd: 1.7547, ept: 228.4116
    Epoch [16/50], Val Losses: mse: 39.2493, mae: 3.6695, huber: 3.2339, swd: 3.4973, ept: 178.2735
    Epoch [16/50], Test Losses: mse: 30.2756, mae: 3.1380, huber: 2.7087, swd: 2.6304, ept: 192.4708
      Epoch 16 composite train-obj: 1.954017
            Val objective improved 3.3520 → 3.2339, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 18.3700, mae: 2.3150, huber: 1.9077, swd: 1.6718, ept: 231.2090
    Epoch [17/50], Val Losses: mse: 38.8750, mae: 3.6444, huber: 3.2099, swd: 3.3994, ept: 178.6321
    Epoch [17/50], Test Losses: mse: 30.7054, mae: 3.1429, huber: 2.7160, swd: 2.8922, ept: 195.1273
      Epoch 17 composite train-obj: 1.907692
            Val objective improved 3.2339 → 3.2099, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 17.8615, mae: 2.2744, huber: 1.8681, swd: 1.6081, ept: 233.5030
    Epoch [18/50], Val Losses: mse: 41.5887, mae: 3.7455, huber: 3.3107, swd: 3.3004, ept: 180.6750
    Epoch [18/50], Test Losses: mse: 29.8090, mae: 3.0518, huber: 2.6263, swd: 2.4024, ept: 197.2260
      Epoch 18 composite train-obj: 1.868068
            No improvement (3.3107), counter 1/5
    Epoch [19/50], Train Losses: mse: 17.0822, mae: 2.2028, huber: 1.8000, swd: 1.5043, ept: 237.0831
    Epoch [19/50], Val Losses: mse: 40.5881, mae: 3.6706, huber: 3.2388, swd: 3.2359, ept: 184.9629
    Epoch [19/50], Test Losses: mse: 29.9655, mae: 3.0668, huber: 2.6426, swd: 2.3881, ept: 200.5801
      Epoch 19 composite train-obj: 1.799987
            No improvement (3.2388), counter 2/5
    Epoch [20/50], Train Losses: mse: 16.9527, mae: 2.2105, huber: 1.8061, swd: 1.5118, ept: 237.3603
    Epoch [20/50], Val Losses: mse: 39.8141, mae: 3.6444, huber: 3.2117, swd: 3.1649, ept: 183.4909
    Epoch [20/50], Test Losses: mse: 29.4856, mae: 3.0497, huber: 2.6257, swd: 2.5953, ept: 200.2360
      Epoch 20 composite train-obj: 1.806098
            No improvement (3.2117), counter 3/5
    Epoch [21/50], Train Losses: mse: 16.7095, mae: 2.1544, huber: 1.7540, swd: 1.4382, ept: 241.0735
    Epoch [21/50], Val Losses: mse: 41.6244, mae: 3.7217, huber: 3.2901, swd: 3.3476, ept: 184.4124
    Epoch [21/50], Test Losses: mse: 30.5273, mae: 3.0489, huber: 2.6261, swd: 2.3374, ept: 208.3505
      Epoch 21 composite train-obj: 1.754005
            No improvement (3.2901), counter 4/5
    Epoch [22/50], Train Losses: mse: 15.8948, mae: 2.0897, huber: 1.6916, swd: 1.3723, ept: 244.6265
    Epoch [22/50], Val Losses: mse: 46.5179, mae: 3.9664, huber: 3.5313, swd: 3.6359, ept: 180.4221
    Epoch [22/50], Test Losses: mse: 30.6645, mae: 3.0746, huber: 2.6498, swd: 2.7129, ept: 206.5806
      Epoch 22 composite train-obj: 1.691647
    Epoch [22/50], Test Losses: mse: 30.7055, mae: 3.1429, huber: 2.7160, swd: 2.8923, ept: 195.1337
    Best round's Test MSE: 30.7054, MAE: 3.1429, SWD: 2.8922
    Best round's Validation MSE: 38.8750, MAE: 3.6444, SWD: 3.3994
    Best round's Test verification MSE : 30.7055, MAE: 3.1429, SWD: 2.8923
    Time taken: 78.30 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 76.6699, mae: 6.5762, huber: 6.0963, swd: 45.9637, ept: 34.3122
    Epoch [1/50], Val Losses: mse: 60.5245, mae: 5.8363, huber: 5.3601, swd: 27.6595, ept: 41.2454
    Epoch [1/50], Test Losses: mse: 55.6340, mae: 5.4289, huber: 4.9564, swd: 27.4832, ept: 45.4713
      Epoch 1 composite train-obj: 6.096290
            Val objective improved inf → 5.3601, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.2886, mae: 4.8538, huber: 4.3883, swd: 18.2875, ept: 95.7068
    Epoch [2/50], Val Losses: mse: 53.3127, mae: 5.2068, huber: 4.7389, swd: 17.0418, ept: 82.5361
    Epoch [2/50], Test Losses: mse: 48.0425, mae: 4.8026, huber: 4.3389, swd: 17.5866, ept: 79.2706
      Epoch 2 composite train-obj: 4.388263
            Val objective improved 5.3601 → 4.7389, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 41.4930, mae: 4.2439, huber: 3.7887, swd: 11.3614, ept: 130.4693
    Epoch [3/50], Val Losses: mse: 50.0728, mae: 4.8138, huber: 4.3516, swd: 11.3038, ept: 107.1579
    Epoch [3/50], Test Losses: mse: 43.6392, mae: 4.3580, huber: 3.8997, swd: 11.4682, ept: 106.0794
      Epoch 3 composite train-obj: 3.788697
            Val objective improved 4.7389 → 4.3516, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.7676, mae: 3.8224, huber: 3.3754, swd: 7.9441, ept: 153.9234
    Epoch [4/50], Val Losses: mse: 46.1140, mae: 4.4900, huber: 4.0339, swd: 8.5868, ept: 120.5661
    Epoch [4/50], Test Losses: mse: 40.1119, mae: 4.0580, huber: 3.6067, swd: 8.4321, ept: 125.9710
      Epoch 4 composite train-obj: 3.375440
            Val objective improved 4.3516 → 4.0339, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.4411, mae: 3.5471, huber: 3.1054, swd: 6.0280, ept: 167.8504
    Epoch [5/50], Val Losses: mse: 46.4884, mae: 4.4724, huber: 4.0176, swd: 8.1622, ept: 125.2825
    Epoch [5/50], Test Losses: mse: 36.9379, mae: 3.8770, huber: 3.4279, swd: 7.6069, ept: 135.6876
      Epoch 5 composite train-obj: 3.105385
            Val objective improved 4.0339 → 4.0176, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.5394, mae: 3.3101, huber: 2.8734, swd: 4.8610, ept: 178.5716
    Epoch [6/50], Val Losses: mse: 43.1425, mae: 4.1917, huber: 3.7400, swd: 6.2139, ept: 139.1131
    Epoch [6/50], Test Losses: mse: 35.4922, mae: 3.7038, huber: 3.2562, swd: 5.5928, ept: 147.3776
      Epoch 6 composite train-obj: 2.873442
            Val objective improved 4.0176 → 3.7400, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.1601, mae: 3.1252, huber: 2.6923, swd: 4.0303, ept: 186.2840
    Epoch [7/50], Val Losses: mse: 45.0281, mae: 4.2597, huber: 3.8091, swd: 6.5963, ept: 143.1566
    Epoch [7/50], Test Losses: mse: 34.1154, mae: 3.5713, huber: 3.1272, swd: 4.9288, ept: 156.1684
      Epoch 7 composite train-obj: 2.692273
            No improvement (3.8091), counter 1/5
    Epoch [8/50], Train Losses: mse: 25.5338, mae: 2.9134, huber: 2.4861, swd: 3.3390, ept: 195.1359
    Epoch [8/50], Val Losses: mse: 44.1322, mae: 4.1569, huber: 3.7082, swd: 5.8741, ept: 144.6585
    Epoch [8/50], Test Losses: mse: 34.1384, mae: 3.5324, huber: 3.0899, swd: 4.6066, ept: 155.7027
      Epoch 8 composite train-obj: 2.486093
            Val objective improved 3.7400 → 3.7082, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 23.6433, mae: 2.7709, huber: 2.3466, swd: 2.9153, ept: 201.5670
    Epoch [9/50], Val Losses: mse: 41.1220, mae: 3.9122, huber: 3.4689, swd: 5.2636, ept: 156.5196
    Epoch [9/50], Test Losses: mse: 31.3031, mae: 3.2972, huber: 2.8613, swd: 4.0717, ept: 170.9642
      Epoch 9 composite train-obj: 2.346638
            Val objective improved 3.7082 → 3.4689, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 22.3511, mae: 2.6678, huber: 2.2461, swd: 2.6173, ept: 207.1143
    Epoch [10/50], Val Losses: mse: 44.2347, mae: 4.0410, huber: 3.5962, swd: 4.9051, ept: 154.6536
    Epoch [10/50], Test Losses: mse: 31.7181, mae: 3.3450, huber: 2.9072, swd: 3.8251, ept: 170.5295
      Epoch 10 composite train-obj: 2.246133
            No improvement (3.5962), counter 1/5
    Epoch [11/50], Train Losses: mse: 20.8763, mae: 2.5483, huber: 2.1306, swd: 2.3246, ept: 211.4397
    Epoch [11/50], Val Losses: mse: 43.4241, mae: 3.9210, huber: 3.4823, swd: 4.9703, ept: 162.0624
    Epoch [11/50], Test Losses: mse: 32.4719, mae: 3.2821, huber: 2.8511, swd: 3.5290, ept: 181.4183
      Epoch 11 composite train-obj: 2.130579
            No improvement (3.4823), counter 2/5
    Epoch [12/50], Train Losses: mse: 19.8040, mae: 2.4672, huber: 2.0512, swd: 2.1441, ept: 215.6253
    Epoch [12/50], Val Losses: mse: 43.7626, mae: 3.9308, huber: 3.4903, swd: 4.4393, ept: 162.8210
    Epoch [12/50], Test Losses: mse: 30.9723, mae: 3.1970, huber: 2.7649, swd: 2.8908, ept: 184.3399
      Epoch 12 composite train-obj: 2.051236
            No improvement (3.4903), counter 3/5
    Epoch [13/50], Train Losses: mse: 19.0696, mae: 2.4086, huber: 1.9943, swd: 2.0158, ept: 219.7135
    Epoch [13/50], Val Losses: mse: 43.0155, mae: 3.9012, huber: 3.4612, swd: 4.5622, ept: 164.4257
    Epoch [13/50], Test Losses: mse: 33.2830, mae: 3.3042, huber: 2.8715, swd: 3.4477, ept: 184.1774
      Epoch 13 composite train-obj: 1.994301
            Val objective improved 3.4689 → 3.4612, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.0328, mae: 2.3118, huber: 1.9016, swd: 1.8312, ept: 225.1262
    Epoch [14/50], Val Losses: mse: 42.2633, mae: 3.8619, huber: 3.4184, swd: 4.4182, ept: 166.7273
    Epoch [14/50], Test Losses: mse: 31.9270, mae: 3.2098, huber: 2.7758, swd: 2.8910, ept: 190.0130
      Epoch 14 composite train-obj: 1.901600
            Val objective improved 3.4612 → 3.4184, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 17.2668, mae: 2.2401, huber: 1.8325, swd: 1.6878, ept: 229.9050
    Epoch [15/50], Val Losses: mse: 40.9931, mae: 3.7559, huber: 3.3194, swd: 4.3827, ept: 171.9529
    Epoch [15/50], Test Losses: mse: 28.5924, mae: 3.0553, huber: 2.6270, swd: 3.1548, ept: 188.7675
      Epoch 15 composite train-obj: 1.832481
            Val objective improved 3.4184 → 3.3194, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.5144, mae: 2.1684, huber: 1.7642, swd: 1.6234, ept: 234.3430
    Epoch [16/50], Val Losses: mse: 40.1385, mae: 3.6931, huber: 3.2563, swd: 3.9982, ept: 175.7026
    Epoch [16/50], Test Losses: mse: 29.4263, mae: 3.0403, huber: 2.6120, swd: 2.5672, ept: 198.7114
      Epoch 16 composite train-obj: 1.764224
            Val objective improved 3.3194 → 3.2563, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.8326, mae: 2.1072, huber: 1.7055, swd: 1.5142, ept: 238.8300
    Epoch [17/50], Val Losses: mse: 39.1864, mae: 3.6355, huber: 3.2011, swd: 4.0666, ept: 175.6971
    Epoch [17/50], Test Losses: mse: 30.1715, mae: 3.1025, huber: 2.6742, swd: 2.9325, ept: 195.1197
      Epoch 17 composite train-obj: 1.705462
            Val objective improved 3.2563 → 3.2011, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.0783, mae: 2.0318, huber: 1.6345, swd: 1.4179, ept: 242.6354
    Epoch [18/50], Val Losses: mse: 41.3267, mae: 3.6961, huber: 3.2628, swd: 3.6068, ept: 179.5240
    Epoch [18/50], Test Losses: mse: 28.4086, mae: 2.9429, huber: 2.5187, swd: 2.4429, ept: 206.0157
      Epoch 18 composite train-obj: 1.634504
            No improvement (3.2628), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.9848, mae: 2.0330, huber: 1.6341, swd: 1.3771, ept: 243.6182
    Epoch [19/50], Val Losses: mse: 43.5395, mae: 3.8062, huber: 3.3707, swd: 3.9030, ept: 177.5757
    Epoch [19/50], Test Losses: mse: 27.5989, mae: 2.9307, huber: 2.5058, swd: 2.6472, ept: 201.7969
      Epoch 19 composite train-obj: 1.634051
            No improvement (3.3707), counter 2/5
    Epoch [20/50], Train Losses: mse: 14.1563, mae: 1.9405, huber: 1.5477, swd: 1.2864, ept: 249.3057
    Epoch [20/50], Val Losses: mse: 42.5318, mae: 3.7623, huber: 3.3307, swd: 3.6046, ept: 180.6568
    Epoch [20/50], Test Losses: mse: 29.3416, mae: 3.0014, huber: 2.5784, swd: 2.6266, ept: 203.1923
      Epoch 20 composite train-obj: 1.547730
            No improvement (3.3307), counter 3/5
    Epoch [21/50], Train Losses: mse: 14.2986, mae: 1.9624, huber: 1.5676, swd: 1.2947, ept: 250.0414
    Epoch [21/50], Val Losses: mse: 42.0042, mae: 3.7521, huber: 3.3162, swd: 3.5315, ept: 182.3207
    Epoch [21/50], Test Losses: mse: 29.9649, mae: 3.0288, huber: 2.6031, swd: 2.5643, ept: 205.9710
      Epoch 21 composite train-obj: 1.567567
            No improvement (3.3162), counter 4/5
    Epoch [22/50], Train Losses: mse: 13.7365, mae: 1.9101, huber: 1.5177, swd: 1.2257, ept: 254.1095
    Epoch [22/50], Val Losses: mse: 40.0676, mae: 3.5684, huber: 3.1406, swd: 3.6609, ept: 194.0556
    Epoch [22/50], Test Losses: mse: 27.8587, mae: 2.8312, huber: 2.4149, swd: 2.3901, ept: 220.4207
      Epoch 22 composite train-obj: 1.517729
            Val objective improved 3.2011 → 3.1406, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 12.9258, mae: 1.8283, huber: 1.4409, swd: 1.1390, ept: 258.3262
    Epoch [23/50], Val Losses: mse: 40.1870, mae: 3.5945, huber: 3.1639, swd: 3.5849, ept: 189.8280
    Epoch [23/50], Test Losses: mse: 26.5766, mae: 2.8172, huber: 2.3966, swd: 2.2905, ept: 215.4869
      Epoch 23 composite train-obj: 1.440869
            No improvement (3.1639), counter 1/5
    Epoch [24/50], Train Losses: mse: 12.7972, mae: 1.8213, huber: 1.4333, swd: 1.1183, ept: 259.5703
    Epoch [24/50], Val Losses: mse: 39.1062, mae: 3.5430, huber: 3.1151, swd: 3.6648, ept: 190.4795
    Epoch [24/50], Test Losses: mse: 26.2101, mae: 2.7813, huber: 2.3638, swd: 2.4330, ept: 217.9150
      Epoch 24 composite train-obj: 1.433278
            Val objective improved 3.1406 → 3.1151, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 12.2136, mae: 1.7565, huber: 1.3730, swd: 1.0651, ept: 263.4860
    Epoch [25/50], Val Losses: mse: 41.0091, mae: 3.6298, huber: 3.1991, swd: 3.3652, ept: 185.4881
    Epoch [25/50], Test Losses: mse: 27.4027, mae: 2.8655, huber: 2.4447, swd: 2.4129, ept: 209.3007
      Epoch 25 composite train-obj: 1.373045
            No improvement (3.1991), counter 1/5
    Epoch [26/50], Train Losses: mse: 12.1274, mae: 1.7565, huber: 1.3722, swd: 1.0634, ept: 263.5749
    Epoch [26/50], Val Losses: mse: 39.4533, mae: 3.5640, huber: 3.1365, swd: 3.4975, ept: 191.5485
    Epoch [26/50], Test Losses: mse: 29.5695, mae: 2.9239, huber: 2.5062, swd: 2.3416, ept: 214.0178
      Epoch 26 composite train-obj: 1.372213
            No improvement (3.1365), counter 2/5
    Epoch [27/50], Train Losses: mse: 12.1738, mae: 1.7602, huber: 1.3756, swd: 1.0641, ept: 264.0342
    Epoch [27/50], Val Losses: mse: 39.7630, mae: 3.5376, huber: 3.1097, swd: 3.3754, ept: 195.6787
    Epoch [27/50], Test Losses: mse: 26.5001, mae: 2.7884, huber: 2.3715, swd: 2.4864, ept: 219.4147
      Epoch 27 composite train-obj: 1.375639
            Val objective improved 3.1151 → 3.1097, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 12.2489, mae: 1.7605, huber: 1.3765, swd: 1.0595, ept: 264.9584
    Epoch [28/50], Val Losses: mse: 40.6332, mae: 3.6015, huber: 3.1723, swd: 3.4484, ept: 196.3388
    Epoch [28/50], Test Losses: mse: 28.2933, mae: 2.8572, huber: 2.4403, swd: 2.4190, ept: 219.6345
      Epoch 28 composite train-obj: 1.376505
            No improvement (3.1723), counter 1/5
    Epoch [29/50], Train Losses: mse: 12.0054, mae: 1.7359, huber: 1.3531, swd: 1.0201, ept: 266.7470
    Epoch [29/50], Val Losses: mse: 40.1924, mae: 3.5296, huber: 3.1073, swd: 3.3974, ept: 195.7110
    Epoch [29/50], Test Losses: mse: 27.8283, mae: 2.7958, huber: 2.3855, swd: 2.3328, ept: 223.4346
      Epoch 29 composite train-obj: 1.353148
            Val objective improved 3.1097 → 3.1073, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 11.5582, mae: 1.6935, huber: 1.3136, swd: 0.9780, ept: 268.4750
    Epoch [30/50], Val Losses: mse: 41.2839, mae: 3.6104, huber: 3.1834, swd: 3.2197, ept: 194.6484
    Epoch [30/50], Test Losses: mse: 26.4538, mae: 2.7806, huber: 2.3658, swd: 2.3149, ept: 218.6425
      Epoch 30 composite train-obj: 1.313592
            No improvement (3.1834), counter 1/5
    Epoch [31/50], Train Losses: mse: 10.7079, mae: 1.5969, huber: 1.2238, swd: 0.9138, ept: 273.1943
    Epoch [31/50], Val Losses: mse: 41.8916, mae: 3.6143, huber: 3.1887, swd: 3.2311, ept: 196.1933
    Epoch [31/50], Test Losses: mse: 25.8196, mae: 2.6900, huber: 2.2786, swd: 2.0333, ept: 226.6437
      Epoch 31 composite train-obj: 1.223831
            No improvement (3.1887), counter 2/5
    Epoch [32/50], Train Losses: mse: 10.9167, mae: 1.6256, huber: 1.2497, swd: 0.9265, ept: 272.1870
    Epoch [32/50], Val Losses: mse: 39.9174, mae: 3.5094, huber: 3.0873, swd: 3.0706, ept: 196.5126
    Epoch [32/50], Test Losses: mse: 25.0242, mae: 2.6488, huber: 2.2405, swd: 2.1874, ept: 227.1089
      Epoch 32 composite train-obj: 1.249704
            Val objective improved 3.1073 → 3.0873, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 10.6735, mae: 1.6071, huber: 1.2324, swd: 0.8894, ept: 272.7891
    Epoch [33/50], Val Losses: mse: 41.2569, mae: 3.6030, huber: 3.1796, swd: 3.2136, ept: 198.6433
    Epoch [33/50], Test Losses: mse: 27.1372, mae: 2.8152, huber: 2.4018, swd: 2.2120, ept: 216.5897
      Epoch 33 composite train-obj: 1.232402
            No improvement (3.1796), counter 1/5
    Epoch [34/50], Train Losses: mse: 10.1727, mae: 1.5493, huber: 1.1788, swd: 0.8502, ept: 276.0192
    Epoch [34/50], Val Losses: mse: 39.5694, mae: 3.4833, huber: 3.0578, swd: 3.0671, ept: 199.2021
    Epoch [34/50], Test Losses: mse: 25.7039, mae: 2.7020, huber: 2.2888, swd: 2.1797, ept: 227.7997
      Epoch 34 composite train-obj: 1.178797
            Val objective improved 3.0873 → 3.0578, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 10.0675, mae: 1.5418, huber: 1.1712, swd: 0.8284, ept: 276.3708
    Epoch [35/50], Val Losses: mse: 39.4548, mae: 3.4794, huber: 3.0579, swd: 3.4620, ept: 198.5236
    Epoch [35/50], Test Losses: mse: 25.9307, mae: 2.6832, huber: 2.2750, swd: 2.1322, ept: 228.0369
      Epoch 35 composite train-obj: 1.171198
            No improvement (3.0579), counter 1/5
    Epoch [36/50], Train Losses: mse: 9.9127, mae: 1.5270, huber: 1.1576, swd: 0.8364, ept: 277.6227
    Epoch [36/50], Val Losses: mse: 38.9775, mae: 3.4235, huber: 3.0038, swd: 3.0524, ept: 204.7891
    Epoch [36/50], Test Losses: mse: 25.0186, mae: 2.6092, huber: 2.2037, swd: 1.9522, ept: 234.7158
      Epoch 36 composite train-obj: 1.157639
            Val objective improved 3.0578 → 3.0038, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 9.5960, mae: 1.4997, huber: 1.1312, swd: 0.8052, ept: 278.6465
    Epoch [37/50], Val Losses: mse: 38.4216, mae: 3.4043, huber: 2.9834, swd: 3.0367, ept: 203.3012
    Epoch [37/50], Test Losses: mse: 25.1421, mae: 2.6392, huber: 2.2306, swd: 2.0860, ept: 231.5003
      Epoch 37 composite train-obj: 1.131196
            Val objective improved 3.0038 → 2.9834, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 9.7866, mae: 1.5163, huber: 1.1470, swd: 0.8151, ept: 278.5907
    Epoch [38/50], Val Losses: mse: 39.6885, mae: 3.4811, huber: 3.0610, swd: 3.4149, ept: 196.8790
    Epoch [38/50], Test Losses: mse: 26.0936, mae: 2.6677, huber: 2.2625, swd: 2.2111, ept: 231.7411
      Epoch 38 composite train-obj: 1.146962
            No improvement (3.0610), counter 1/5
    Epoch [39/50], Train Losses: mse: 9.9164, mae: 1.5359, huber: 1.1642, swd: 0.8263, ept: 278.0877
    Epoch [39/50], Val Losses: mse: 38.8496, mae: 3.4277, huber: 3.0086, swd: 3.2506, ept: 203.8361
    Epoch [39/50], Test Losses: mse: 25.7788, mae: 2.6589, huber: 2.2524, swd: 2.1286, ept: 231.8011
      Epoch 39 composite train-obj: 1.164161
            No improvement (3.0086), counter 2/5
    Epoch [40/50], Train Losses: mse: 9.3921, mae: 1.4854, huber: 1.1173, swd: 0.7671, ept: 279.5881
    Epoch [40/50], Val Losses: mse: 38.5129, mae: 3.3860, huber: 2.9680, swd: 3.1601, ept: 202.2652
    Epoch [40/50], Test Losses: mse: 22.8840, mae: 2.5006, huber: 2.0976, swd: 1.9784, ept: 235.9959
      Epoch 40 composite train-obj: 1.117302
            Val objective improved 2.9834 → 2.9680, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 9.5692, mae: 1.4916, huber: 1.1242, swd: 0.7821, ept: 279.9683
    Epoch [41/50], Val Losses: mse: 39.5874, mae: 3.4894, huber: 3.0662, swd: 3.1901, ept: 201.5537
    Epoch [41/50], Test Losses: mse: 27.5877, mae: 2.7715, huber: 2.3612, swd: 2.0825, ept: 227.0566
      Epoch 41 composite train-obj: 1.124160
            No improvement (3.0662), counter 1/5
    Epoch [42/50], Train Losses: mse: 9.2268, mae: 1.4667, huber: 1.1001, swd: 0.7661, ept: 281.3700
    Epoch [42/50], Val Losses: mse: 38.7086, mae: 3.3984, huber: 2.9803, swd: 2.8789, ept: 203.1564
    Epoch [42/50], Test Losses: mse: 25.6779, mae: 2.6080, huber: 2.2058, swd: 1.9564, ept: 236.5820
      Epoch 42 composite train-obj: 1.100072
            No improvement (2.9803), counter 2/5
    Epoch [43/50], Train Losses: mse: 8.9203, mae: 1.4287, huber: 1.0656, swd: 0.7349, ept: 283.3911
    Epoch [43/50], Val Losses: mse: 39.0977, mae: 3.4435, huber: 3.0225, swd: 3.0830, ept: 202.3257
    Epoch [43/50], Test Losses: mse: 25.9082, mae: 2.6535, huber: 2.2472, swd: 2.0366, ept: 232.6967
      Epoch 43 composite train-obj: 1.065560
            No improvement (3.0225), counter 3/5
    Epoch [44/50], Train Losses: mse: 8.7859, mae: 1.4137, huber: 1.0513, swd: 0.7121, ept: 283.8842
    Epoch [44/50], Val Losses: mse: 39.2485, mae: 3.4569, huber: 3.0375, swd: 3.2263, ept: 196.9609
    Epoch [44/50], Test Losses: mse: 25.5204, mae: 2.6508, huber: 2.2450, swd: 2.1315, ept: 228.6325
      Epoch 44 composite train-obj: 1.051265
            No improvement (3.0375), counter 4/5
    Epoch [45/50], Train Losses: mse: 8.8811, mae: 1.4283, huber: 1.0644, swd: 0.7277, ept: 283.6209
    Epoch [45/50], Val Losses: mse: 38.6499, mae: 3.4133, huber: 2.9903, swd: 3.0669, ept: 201.8746
    Epoch [45/50], Test Losses: mse: 24.1891, mae: 2.5876, huber: 2.1793, swd: 1.8220, ept: 230.8400
      Epoch 45 composite train-obj: 1.064366
    Epoch [45/50], Test Losses: mse: 22.8848, mae: 2.5007, huber: 2.0977, swd: 1.9784, ept: 235.9774
    Best round's Test MSE: 22.8840, MAE: 2.5006, SWD: 1.9784
    Best round's Validation MSE: 38.5129, MAE: 3.3860, SWD: 3.1601
    Best round's Test verification MSE : 22.8848, MAE: 2.5007, SWD: 1.9784
    Time taken: 143.58 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1722)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 27.6077 ± 3.3938
      mae: 2.8180 ± 0.2623
      huber: 2.4053 ± 0.2525
      swd: 2.1971 ± 0.5027
      ept: 220.0919 ± 17.8723
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 37.9496 ± 1.0630
      mae: 3.4580 ± 0.1329
      huber: 3.0334 ± 0.1262
      swd: 3.0635 ± 0.3210
      ept: 195.9211 ± 12.3686
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 410.81 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1722
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### AB: No Koopman but shift in z_push
Shift term in z alone *can* offer the increase in performance of full Koopman.


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128, 
    # ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=True, 
    ablate_shift_inside_scale=False,
    householder_reflects_latent = 2,
    householder_reflects_data = 4,
    mixing_strategy='delay_only', 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
    ablate_no_koopman=True, ### HERE
    ablate_no_shift_in_z_push=False, ### HERE
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 74.8636, mae: 6.5581, huber: 6.0778, swd: 43.2033, ept: 32.2648
    Epoch [1/50], Val Losses: mse: 60.1865, mae: 5.8990, huber: 5.4218, swd: 28.1588, ept: 35.8416
    Epoch [1/50], Test Losses: mse: 55.8140, mae: 5.5282, huber: 5.0538, swd: 27.4124, ept: 36.9981
      Epoch 1 composite train-obj: 6.077780
            Val objective improved inf → 5.4218, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.6036, mae: 4.8951, huber: 4.4295, swd: 17.6546, ept: 87.5457
    Epoch [2/50], Val Losses: mse: 49.8220, mae: 4.9589, huber: 4.4945, swd: 15.4406, ept: 86.8151
    Epoch [2/50], Test Losses: mse: 46.7554, mae: 4.7044, huber: 4.2424, swd: 16.1154, ept: 80.4440
      Epoch 2 composite train-obj: 4.429486
            Val objective improved 5.4218 → 4.4945, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.6598, mae: 4.1928, huber: 3.7384, swd: 10.1384, ept: 127.9885
    Epoch [3/50], Val Losses: mse: 44.9405, mae: 4.5041, huber: 4.0446, swd: 9.2550, ept: 110.5080
    Epoch [3/50], Test Losses: mse: 41.3374, mae: 4.2346, huber: 3.7771, swd: 8.9971, ept: 113.1896
      Epoch 3 composite train-obj: 3.738444
            Val objective improved 4.4945 → 4.0446, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 34.8656, mae: 3.6969, huber: 3.2516, swd: 6.1240, ept: 164.0673
    Epoch [4/50], Val Losses: mse: 40.9449, mae: 4.1592, huber: 3.7059, swd: 7.0028, ept: 131.1907
    Epoch [4/50], Test Losses: mse: 37.2239, mae: 3.9466, huber: 3.4948, swd: 7.6165, ept: 133.7267
      Epoch 4 composite train-obj: 3.251619
            Val objective improved 4.0446 → 3.7059, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.1811, mae: 3.4584, huber: 3.0187, swd: 4.7379, ept: 180.1870
    Epoch [5/50], Val Losses: mse: 39.8319, mae: 4.0537, huber: 3.6027, swd: 5.6567, ept: 144.6835
    Epoch [5/50], Test Losses: mse: 37.0962, mae: 3.8499, huber: 3.4005, swd: 5.6101, ept: 146.4449
      Epoch 5 composite train-obj: 3.018676
            Val objective improved 3.7059 → 3.6027, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.0300, mae: 3.2622, huber: 2.8277, swd: 3.9643, ept: 190.6235
    Epoch [6/50], Val Losses: mse: 39.0471, mae: 3.9308, huber: 3.4845, swd: 4.9605, ept: 153.5061
    Epoch [6/50], Test Losses: mse: 34.1570, mae: 3.6098, huber: 3.1659, swd: 4.5372, ept: 161.4381
      Epoch 6 composite train-obj: 2.827667
            Val objective improved 3.6027 → 3.4845, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.9446, mae: 3.0846, huber: 2.6545, swd: 3.3420, ept: 198.6717
    Epoch [7/50], Val Losses: mse: 38.9308, mae: 3.8877, huber: 3.4402, swd: 4.4967, ept: 154.4662
    Epoch [7/50], Test Losses: mse: 33.7402, mae: 3.5242, huber: 3.0812, swd: 3.7932, ept: 169.1296
      Epoch 7 composite train-obj: 2.654488
            Val objective improved 3.4845 → 3.4402, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 26.3039, mae: 2.9466, huber: 2.5201, swd: 2.8518, ept: 204.7647
    Epoch [8/50], Val Losses: mse: 39.2971, mae: 3.8552, huber: 3.4117, swd: 4.2700, ept: 163.5464
    Epoch [8/50], Test Losses: mse: 32.0686, mae: 3.4049, huber: 2.9668, swd: 3.9587, ept: 175.3927
      Epoch 8 composite train-obj: 2.520135
            Val objective improved 3.4402 → 3.4117, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 25.0831, mae: 2.8517, huber: 2.4273, swd: 2.5824, ept: 209.6870
    Epoch [9/50], Val Losses: mse: 39.4348, mae: 3.7995, huber: 3.3570, swd: 3.9134, ept: 171.0871
    Epoch [9/50], Test Losses: mse: 29.9309, mae: 3.2346, huber: 2.7989, swd: 3.5683, ept: 186.2827
      Epoch 9 composite train-obj: 2.427320
            Val objective improved 3.4117 → 3.3570, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.8361, mae: 2.7588, huber: 2.3369, swd: 2.3381, ept: 214.1998
    Epoch [10/50], Val Losses: mse: 38.3864, mae: 3.7494, huber: 3.3082, swd: 3.3239, ept: 167.7868
    Epoch [10/50], Test Losses: mse: 31.8259, mae: 3.3560, huber: 2.9181, swd: 2.8819, ept: 177.4723
      Epoch 10 composite train-obj: 2.336914
            Val objective improved 3.3570 → 3.3082, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.4520, mae: 2.5579, huber: 2.1428, swd: 1.9742, ept: 223.7372
    Epoch [11/50], Val Losses: mse: 39.4561, mae: 3.6845, huber: 3.2457, swd: 2.8832, ept: 182.1164
    Epoch [11/50], Test Losses: mse: 27.6072, mae: 3.0503, huber: 2.6186, swd: 2.4132, ept: 197.0637
      Epoch 11 composite train-obj: 2.142829
            Val objective improved 3.3082 → 3.2457, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.9428, mae: 2.5201, huber: 2.1053, swd: 1.8529, ept: 226.6334
    Epoch [12/50], Val Losses: mse: 38.1526, mae: 3.6169, huber: 3.1813, swd: 2.9849, ept: 178.8120
    Epoch [12/50], Test Losses: mse: 30.4769, mae: 3.1407, huber: 2.7120, swd: 2.6875, ept: 196.9493
      Epoch 12 composite train-obj: 2.105332
            Val objective improved 3.2457 → 3.1813, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 19.1826, mae: 2.3707, huber: 1.9616, swd: 1.6396, ept: 233.4312
    Epoch [13/50], Val Losses: mse: 38.7860, mae: 3.5912, huber: 3.1589, swd: 2.4970, ept: 188.1985
    Epoch [13/50], Test Losses: mse: 28.1579, mae: 2.9788, huber: 2.5537, swd: 2.2007, ept: 207.4694
      Epoch 13 composite train-obj: 1.961558
            Val objective improved 3.1813 → 3.1589, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 19.1519, mae: 2.3748, huber: 1.9644, swd: 1.6011, ept: 233.6180
    Epoch [14/50], Val Losses: mse: 38.1819, mae: 3.5828, huber: 3.1491, swd: 2.9430, ept: 185.1802
    Epoch [14/50], Test Losses: mse: 29.8111, mae: 3.0723, huber: 2.6439, swd: 2.5177, ept: 203.0482
      Epoch 14 composite train-obj: 1.964358
            Val objective improved 3.1589 → 3.1491, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 18.0416, mae: 2.2716, huber: 1.8651, swd: 1.4728, ept: 239.3185
    Epoch [15/50], Val Losses: mse: 37.1272, mae: 3.4431, huber: 3.0144, swd: 2.4496, ept: 198.2973
    Epoch [15/50], Test Losses: mse: 25.7535, mae: 2.7829, huber: 2.3629, swd: 1.9705, ept: 218.7558
      Epoch 15 composite train-obj: 1.865097
            Val objective improved 3.1491 → 3.0144, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.9751, mae: 2.1832, huber: 1.7804, swd: 1.3531, ept: 242.8539
    Epoch [16/50], Val Losses: mse: 37.8479, mae: 3.4409, huber: 3.0151, swd: 2.3144, ept: 200.1914
    Epoch [16/50], Test Losses: mse: 25.9923, mae: 2.7806, huber: 2.3619, swd: 1.8827, ept: 219.0454
      Epoch 16 composite train-obj: 1.780432
            No improvement (3.0151), counter 1/5
    Epoch [17/50], Train Losses: mse: 15.9117, mae: 2.0914, huber: 1.6926, swd: 1.2412, ept: 247.3746
    Epoch [17/50], Val Losses: mse: 38.4632, mae: 3.5038, huber: 3.0757, swd: 2.2666, ept: 197.4443
    Epoch [17/50], Test Losses: mse: 27.9041, mae: 2.9235, huber: 2.5024, swd: 2.0473, ept: 213.2110
      Epoch 17 composite train-obj: 1.692591
            No improvement (3.0757), counter 2/5
    Epoch [18/50], Train Losses: mse: 15.6015, mae: 2.0716, huber: 1.6725, swd: 1.2053, ept: 249.6658
    Epoch [18/50], Val Losses: mse: 35.8406, mae: 3.3251, huber: 2.8990, swd: 2.1412, ept: 201.1758
    Epoch [18/50], Test Losses: mse: 27.5726, mae: 2.8671, huber: 2.4460, swd: 1.8095, ept: 217.4038
      Epoch 18 composite train-obj: 1.672511
            Val objective improved 3.0144 → 2.8990, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 15.1220, mae: 2.0216, huber: 1.6250, swd: 1.1564, ept: 253.0126
    Epoch [19/50], Val Losses: mse: 38.2994, mae: 3.4858, huber: 3.0584, swd: 2.2654, ept: 195.8898
    Epoch [19/50], Test Losses: mse: 30.9430, mae: 3.0575, huber: 2.6331, swd: 2.0247, ept: 212.6465
      Epoch 19 composite train-obj: 1.624990
            No improvement (3.0584), counter 1/5
    Epoch [20/50], Train Losses: mse: 14.5094, mae: 1.9672, huber: 1.5733, swd: 1.0922, ept: 256.2366
    Epoch [20/50], Val Losses: mse: 37.9880, mae: 3.4298, huber: 3.0017, swd: 2.1094, ept: 198.6565
    Epoch [20/50], Test Losses: mse: 29.3770, mae: 2.9490, huber: 2.5273, swd: 1.8940, ept: 220.7298
      Epoch 20 composite train-obj: 1.573333
            No improvement (3.0017), counter 2/5
    Epoch [21/50], Train Losses: mse: 14.8586, mae: 1.9956, huber: 1.5998, swd: 1.1291, ept: 255.8607
    Epoch [21/50], Val Losses: mse: 33.0758, mae: 3.1946, huber: 2.7713, swd: 2.1225, ept: 203.6404
    Epoch [21/50], Test Losses: mse: 29.4650, mae: 2.9953, huber: 2.5738, swd: 1.8541, ept: 213.2567
      Epoch 21 composite train-obj: 1.599766
            Val objective improved 2.8990 → 2.7713, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.7286, mae: 1.8866, huber: 1.4975, swd: 1.0282, ept: 260.7170
    Epoch [22/50], Val Losses: mse: 35.5953, mae: 3.2499, huber: 2.8267, swd: 2.3166, ept: 207.6742
    Epoch [22/50], Test Losses: mse: 26.3995, mae: 2.7509, huber: 2.3357, swd: 1.9199, ept: 228.3053
      Epoch 22 composite train-obj: 1.497487
            No improvement (2.8267), counter 1/5
    Epoch [23/50], Train Losses: mse: 13.3723, mae: 1.8576, huber: 1.4691, swd: 0.9972, ept: 262.4224
    Epoch [23/50], Val Losses: mse: 35.8657, mae: 3.2466, huber: 2.8261, swd: 2.1655, ept: 210.3127
    Epoch [23/50], Test Losses: mse: 25.3106, mae: 2.6758, huber: 2.2621, swd: 1.8385, ept: 230.8557
      Epoch 23 composite train-obj: 1.469128
            No improvement (2.8261), counter 2/5
    Epoch [24/50], Train Losses: mse: 12.7523, mae: 1.7912, huber: 1.4072, swd: 0.9340, ept: 265.4731
    Epoch [24/50], Val Losses: mse: 34.0531, mae: 3.1435, huber: 2.7256, swd: 2.0427, ept: 212.1471
    Epoch [24/50], Test Losses: mse: 24.7761, mae: 2.6625, huber: 2.2505, swd: 1.6851, ept: 225.5693
      Epoch 24 composite train-obj: 1.407153
            Val objective improved 2.7713 → 2.7256, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 13.2138, mae: 1.8124, huber: 1.4283, swd: 0.9508, ept: 265.5022
    Epoch [25/50], Val Losses: mse: 33.8774, mae: 3.1646, huber: 2.7463, swd: 2.0203, ept: 220.0253
    Epoch [25/50], Test Losses: mse: 29.6757, mae: 2.8493, huber: 2.4369, swd: 1.7072, ept: 235.3982
      Epoch 25 composite train-obj: 1.428310
            No improvement (2.7463), counter 1/5
    Epoch [26/50], Train Losses: mse: 12.7450, mae: 1.7874, huber: 1.4024, swd: 0.9155, ept: 267.3986
    Epoch [26/50], Val Losses: mse: 33.4448, mae: 3.1798, huber: 2.7595, swd: 2.3207, ept: 207.3557
    Epoch [26/50], Test Losses: mse: 24.9809, mae: 2.6638, huber: 2.2511, swd: 1.9322, ept: 231.7260
      Epoch 26 composite train-obj: 1.402361
            No improvement (2.7595), counter 2/5
    Epoch [27/50], Train Losses: mse: 12.1467, mae: 1.7331, huber: 1.3519, swd: 0.8836, ept: 268.9973
    Epoch [27/50], Val Losses: mse: 33.2954, mae: 3.0514, huber: 2.6390, swd: 1.8074, ept: 223.2520
    Epoch [27/50], Test Losses: mse: 24.5866, mae: 2.5654, huber: 2.1599, swd: 1.5948, ept: 240.7011
      Epoch 27 composite train-obj: 1.351893
            Val objective improved 2.7256 → 2.6390, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 12.3958, mae: 1.7411, huber: 1.3599, swd: 0.8791, ept: 269.3679
    Epoch [28/50], Val Losses: mse: 35.0600, mae: 3.2357, huber: 2.8166, swd: 2.2677, ept: 209.9666
    Epoch [28/50], Test Losses: mse: 28.8119, mae: 2.8612, huber: 2.4453, swd: 2.1028, ept: 221.6509
      Epoch 28 composite train-obj: 1.359898
            No improvement (2.8166), counter 1/5
    Epoch [29/50], Train Losses: mse: 12.9729, mae: 1.8024, huber: 1.4169, swd: 0.9357, ept: 266.7239
    Epoch [29/50], Val Losses: mse: 34.6809, mae: 3.1236, huber: 2.7110, swd: 2.0377, ept: 219.6604
    Epoch [29/50], Test Losses: mse: 25.8530, mae: 2.6420, huber: 2.2341, swd: 1.7400, ept: 236.4090
      Epoch 29 composite train-obj: 1.416887
            No improvement (2.7110), counter 2/5
    Epoch [30/50], Train Losses: mse: 11.5517, mae: 1.6572, huber: 1.2814, swd: 0.8255, ept: 273.7409
    Epoch [30/50], Val Losses: mse: 35.9588, mae: 3.2008, huber: 2.7842, swd: 1.7050, ept: 219.0643
    Epoch [30/50], Test Losses: mse: 26.3971, mae: 2.6776, huber: 2.2693, swd: 1.5664, ept: 234.1170
      Epoch 30 composite train-obj: 1.281367
            No improvement (2.7842), counter 3/5
    Epoch [31/50], Train Losses: mse: 11.2666, mae: 1.6360, huber: 1.2611, swd: 0.7936, ept: 274.1552
    Epoch [31/50], Val Losses: mse: 31.3193, mae: 2.9888, huber: 2.5764, swd: 2.0128, ept: 225.0583
    Epoch [31/50], Test Losses: mse: 24.9371, mae: 2.5634, huber: 2.1583, swd: 1.7237, ept: 243.6963
      Epoch 31 composite train-obj: 1.261100
            Val objective improved 2.6390 → 2.5764, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 10.9567, mae: 1.6105, huber: 1.2366, swd: 0.7661, ept: 275.1022
    Epoch [32/50], Val Losses: mse: 36.9822, mae: 3.2525, huber: 2.8348, swd: 1.7193, ept: 216.8145
    Epoch [32/50], Test Losses: mse: 27.6763, mae: 2.7657, huber: 2.3538, swd: 1.5138, ept: 228.4327
      Epoch 32 composite train-obj: 1.236611
            No improvement (2.8348), counter 1/5
    Epoch [33/50], Train Losses: mse: 11.1536, mae: 1.6231, huber: 1.2484, swd: 0.7746, ept: 275.2406
    Epoch [33/50], Val Losses: mse: 36.7819, mae: 3.1998, huber: 2.7875, swd: 1.9159, ept: 220.7708
    Epoch [33/50], Test Losses: mse: 26.1516, mae: 2.6110, huber: 2.2068, swd: 1.5649, ept: 243.6640
      Epoch 33 composite train-obj: 1.248402
            No improvement (2.7875), counter 2/5
    Epoch [34/50], Train Losses: mse: 10.2761, mae: 1.5307, huber: 1.1632, swd: 0.7149, ept: 279.1639
    Epoch [34/50], Val Losses: mse: 32.6837, mae: 3.0513, huber: 2.6367, swd: 1.7947, ept: 224.0761
    Epoch [34/50], Test Losses: mse: 28.5135, mae: 2.7708, huber: 2.3610, swd: 1.6411, ept: 231.3540
      Epoch 34 composite train-obj: 1.163200
            No improvement (2.6367), counter 3/5
    Epoch [35/50], Train Losses: mse: 10.9645, mae: 1.6040, huber: 1.2306, swd: 0.7789, ept: 276.9513
    Epoch [35/50], Val Losses: mse: 36.2282, mae: 3.1944, huber: 2.7794, swd: 1.8157, ept: 220.0441
    Epoch [35/50], Test Losses: mse: 27.6117, mae: 2.7199, huber: 2.3107, swd: 1.6675, ept: 238.0534
      Epoch 35 composite train-obj: 1.230594
            No improvement (2.7794), counter 4/5
    Epoch [36/50], Train Losses: mse: 11.2248, mae: 1.6210, huber: 1.2469, swd: 0.7902, ept: 275.8525
    Epoch [36/50], Val Losses: mse: 34.2492, mae: 3.1965, huber: 2.7755, swd: 2.2248, ept: 216.7713
    Epoch [36/50], Test Losses: mse: 27.2161, mae: 2.7382, huber: 2.3258, swd: 1.9542, ept: 236.4904
      Epoch 36 composite train-obj: 1.246857
    Epoch [36/50], Test Losses: mse: 24.9418, mae: 2.5635, huber: 2.1585, swd: 1.7228, ept: 243.7139
    Best round's Test MSE: 24.9371, MAE: 2.5634, SWD: 1.7237
    Best round's Validation MSE: 31.3193, MAE: 2.9888, SWD: 2.0128
    Best round's Test verification MSE : 24.9418, MAE: 2.5635, SWD: 1.7228
    Time taken: 119.07 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 74.9865, mae: 6.5224, huber: 6.0427, swd: 43.4665, ept: 34.4261
    Epoch [1/50], Val Losses: mse: 58.7095, mae: 5.7164, huber: 5.2414, swd: 24.8846, ept: 49.1569
    Epoch [1/50], Test Losses: mse: 53.9555, mae: 5.3362, huber: 4.8649, swd: 26.8128, ept: 46.8910
      Epoch 1 composite train-obj: 6.042699
            Val objective improved inf → 5.2414, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.2681, mae: 4.7868, huber: 4.3224, swd: 15.5492, ept: 96.0385
    Epoch [2/50], Val Losses: mse: 48.4449, mae: 4.9034, huber: 4.4374, swd: 12.8581, ept: 84.1658
    Epoch [2/50], Test Losses: mse: 45.0719, mae: 4.6019, huber: 4.1395, swd: 13.4827, ept: 85.3874
      Epoch 2 composite train-obj: 4.322384
            Val objective improved 5.2414 → 4.4374, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.0694, mae: 4.0602, huber: 3.6075, swd: 8.1883, ept: 140.3817
    Epoch [3/50], Val Losses: mse: 43.8718, mae: 4.4546, huber: 3.9963, swd: 8.5432, ept: 114.2031
    Epoch [3/50], Test Losses: mse: 40.5474, mae: 4.1850, huber: 3.7294, swd: 9.0388, ept: 121.4289
      Epoch 3 composite train-obj: 3.607516
            Val objective improved 4.4374 → 3.9963, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 34.9855, mae: 3.6951, huber: 3.2497, swd: 5.7267, ept: 165.2924
    Epoch [4/50], Val Losses: mse: 41.5341, mae: 4.1752, huber: 3.7234, swd: 6.6342, ept: 136.0800
    Epoch [4/50], Test Losses: mse: 38.4650, mae: 3.9589, huber: 3.5087, swd: 7.0936, ept: 140.5957
      Epoch 4 composite train-obj: 3.249653
            Val objective improved 3.9963 → 3.7234, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.1593, mae: 3.4397, huber: 3.0007, swd: 4.6847, ept: 182.4695
    Epoch [5/50], Val Losses: mse: 42.1281, mae: 4.1619, huber: 3.7115, swd: 6.1185, ept: 146.5759
    Epoch [5/50], Test Losses: mse: 35.1058, mae: 3.7085, huber: 3.2612, swd: 5.6465, ept: 153.7393
      Epoch 5 composite train-obj: 3.000653
            Val objective improved 3.7234 → 3.7115, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.2580, mae: 3.2846, huber: 2.8493, swd: 3.9996, ept: 188.9860
    Epoch [6/50], Val Losses: mse: 37.5390, mae: 3.8417, huber: 3.3942, swd: 4.6530, ept: 157.9123
    Epoch [6/50], Test Losses: mse: 32.9268, mae: 3.5291, huber: 3.0857, swd: 4.4376, ept: 163.8578
      Epoch 6 composite train-obj: 2.849307
            Val objective improved 3.7115 → 3.3942, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.9097, mae: 3.0876, huber: 2.6576, swd: 3.3643, ept: 198.0163
    Epoch [7/50], Val Losses: mse: 39.4443, mae: 3.8640, huber: 3.4213, swd: 4.3084, ept: 163.6418
    Epoch [7/50], Test Losses: mse: 30.4767, mae: 3.3694, huber: 2.9307, swd: 4.3553, ept: 166.1742
      Epoch 7 composite train-obj: 2.657609
            No improvement (3.4213), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.1963, mae: 2.9522, huber: 2.5253, swd: 2.9391, ept: 203.1641
    Epoch [8/50], Val Losses: mse: 40.8758, mae: 3.9514, huber: 3.5053, swd: 3.8700, ept: 157.7998
    Epoch [8/50], Test Losses: mse: 31.6601, mae: 3.3894, huber: 2.9492, swd: 3.3844, ept: 173.9015
      Epoch 8 composite train-obj: 2.525260
            No improvement (3.5053), counter 2/5
    Epoch [9/50], Train Losses: mse: 24.5376, mae: 2.8176, huber: 2.3939, swd: 2.5811, ept: 209.2320
    Epoch [9/50], Val Losses: mse: 40.0774, mae: 3.9251, huber: 3.4782, swd: 3.9609, ept: 163.9550
    Epoch [9/50], Test Losses: mse: 31.5740, mae: 3.3936, huber: 2.9523, swd: 3.9422, ept: 168.7757
      Epoch 9 composite train-obj: 2.393859
            No improvement (3.4782), counter 3/5
    Epoch [10/50], Train Losses: mse: 23.3652, mae: 2.7257, huber: 2.3044, swd: 2.3665, ept: 212.7271
    Epoch [10/50], Val Losses: mse: 37.6986, mae: 3.7363, huber: 3.2968, swd: 3.8376, ept: 171.0177
    Epoch [10/50], Test Losses: mse: 29.9994, mae: 3.2512, huber: 2.8170, swd: 3.5830, ept: 183.6158
      Epoch 10 composite train-obj: 2.304381
            Val objective improved 3.3942 → 3.2968, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 22.1596, mae: 2.6282, huber: 2.2097, swd: 2.1774, ept: 217.7601
    Epoch [11/50], Val Losses: mse: 41.7645, mae: 3.8991, huber: 3.4587, swd: 3.8227, ept: 171.3845
    Epoch [11/50], Test Losses: mse: 31.3800, mae: 3.2617, huber: 2.8281, swd: 3.0919, ept: 186.5657
      Epoch 11 composite train-obj: 2.209684
            No improvement (3.4587), counter 1/5
    Epoch [12/50], Train Losses: mse: 21.3755, mae: 2.5705, huber: 2.1530, swd: 2.0286, ept: 220.3194
    Epoch [12/50], Val Losses: mse: 40.2001, mae: 3.7835, huber: 3.3431, swd: 2.9184, ept: 173.7461
    Epoch [12/50], Test Losses: mse: 28.0826, mae: 3.0839, huber: 2.6518, swd: 2.5915, ept: 193.7944
      Epoch 12 composite train-obj: 2.153001
            No improvement (3.3431), counter 2/5
    Epoch [13/50], Train Losses: mse: 20.0969, mae: 2.4553, huber: 2.0424, swd: 1.8450, ept: 225.9822
    Epoch [13/50], Val Losses: mse: 37.6482, mae: 3.5639, huber: 3.1324, swd: 3.2394, ept: 185.1521
    Epoch [13/50], Test Losses: mse: 27.3380, mae: 2.9555, huber: 2.5321, swd: 2.6203, ept: 200.7308
      Epoch 13 composite train-obj: 2.042439
            Val objective improved 3.2968 → 3.1324, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.5397, mae: 2.3193, huber: 1.9117, swd: 1.6537, ept: 231.9507
    Epoch [14/50], Val Losses: mse: 41.0606, mae: 3.7488, huber: 3.3133, swd: 2.8571, ept: 180.4989
    Epoch [14/50], Test Losses: mse: 29.4632, mae: 3.0536, huber: 2.6267, swd: 2.2858, ept: 200.3999
      Epoch 14 composite train-obj: 1.911700
            No improvement (3.3133), counter 1/5
    Epoch [15/50], Train Losses: mse: 17.6988, mae: 2.2498, huber: 1.8445, swd: 1.5595, ept: 235.7648
    Epoch [15/50], Val Losses: mse: 37.1421, mae: 3.5512, huber: 3.1175, swd: 3.3252, ept: 185.2974
    Epoch [15/50], Test Losses: mse: 29.0116, mae: 3.0192, huber: 2.5936, swd: 2.6254, ept: 203.6832
      Epoch 15 composite train-obj: 1.844507
            Val objective improved 3.1324 → 3.1175, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.9358, mae: 2.1886, huber: 1.7857, swd: 1.4601, ept: 237.7434
    Epoch [16/50], Val Losses: mse: 37.3594, mae: 3.5226, huber: 3.0927, swd: 2.8758, ept: 185.6769
    Epoch [16/50], Test Losses: mse: 27.6608, mae: 2.9435, huber: 2.5217, swd: 2.3398, ept: 201.2468
      Epoch 16 composite train-obj: 1.785695
            Val objective improved 3.1175 → 3.0927, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 16.0315, mae: 2.0978, huber: 1.6993, swd: 1.3511, ept: 242.7154
    Epoch [17/50], Val Losses: mse: 36.9096, mae: 3.4336, huber: 3.0047, swd: 2.2858, ept: 192.5028
    Epoch [17/50], Test Losses: mse: 27.1490, mae: 2.8854, huber: 2.4639, swd: 1.9007, ept: 208.9503
      Epoch 17 composite train-obj: 1.699346
            Val objective improved 3.0927 → 3.0047, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.9508, mae: 2.1011, huber: 1.7009, swd: 1.3191, ept: 243.7698
    Epoch [18/50], Val Losses: mse: 40.2320, mae: 3.5977, huber: 3.1651, swd: 2.4744, ept: 195.5218
    Epoch [18/50], Test Losses: mse: 26.0444, mae: 2.8376, huber: 2.4157, swd: 2.1063, ept: 211.2025
      Epoch 18 composite train-obj: 1.700932
            No improvement (3.1651), counter 1/5
    Epoch [19/50], Train Losses: mse: 15.5996, mae: 2.0717, huber: 1.6722, swd: 1.2709, ept: 246.1304
    Epoch [19/50], Val Losses: mse: 38.8630, mae: 3.5066, huber: 3.0792, swd: 2.3795, ept: 193.4112
    Epoch [19/50], Test Losses: mse: 26.8933, mae: 2.8469, huber: 2.4268, swd: 1.9843, ept: 213.6337
      Epoch 19 composite train-obj: 1.672235
            No improvement (3.0792), counter 2/5
    Epoch [20/50], Train Losses: mse: 14.9909, mae: 2.0114, huber: 1.6150, swd: 1.2133, ept: 249.3234
    Epoch [20/50], Val Losses: mse: 37.0665, mae: 3.3839, huber: 2.9586, swd: 2.3985, ept: 200.6692
    Epoch [20/50], Test Losses: mse: 26.3298, mae: 2.7476, huber: 2.3308, swd: 1.7314, ept: 225.1439
      Epoch 20 composite train-obj: 1.615048
            Val objective improved 3.0047 → 2.9586, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 15.0227, mae: 1.9983, huber: 1.6041, swd: 1.2006, ept: 251.1513
    Epoch [21/50], Val Losses: mse: 40.2951, mae: 3.5327, huber: 3.1074, swd: 2.2328, ept: 200.6523
    Epoch [21/50], Test Losses: mse: 26.5923, mae: 2.7763, huber: 2.3604, swd: 1.6897, ept: 218.9444
      Epoch 21 composite train-obj: 1.604060
            No improvement (3.1074), counter 1/5
    Epoch [22/50], Train Losses: mse: 14.8688, mae: 1.9832, huber: 1.5893, swd: 1.1936, ept: 253.5628
    Epoch [22/50], Val Losses: mse: 37.5455, mae: 3.4523, huber: 3.0230, swd: 2.1031, ept: 195.6907
    Epoch [22/50], Test Losses: mse: 27.3661, mae: 2.8602, huber: 2.4395, swd: 1.7980, ept: 214.5808
      Epoch 22 composite train-obj: 1.589321
            No improvement (3.0230), counter 2/5
    Epoch [23/50], Train Losses: mse: 13.5086, mae: 1.8718, huber: 1.4826, swd: 1.0787, ept: 257.7071
    Epoch [23/50], Val Losses: mse: 36.5374, mae: 3.3621, huber: 2.9384, swd: 2.4127, ept: 203.3474
    Epoch [23/50], Test Losses: mse: 26.9607, mae: 2.8230, huber: 2.4064, swd: 1.9838, ept: 217.6931
      Epoch 23 composite train-obj: 1.482586
            Val objective improved 2.9586 → 2.9384, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 13.2963, mae: 1.8489, huber: 1.4610, swd: 1.0401, ept: 259.4140
    Epoch [24/50], Val Losses: mse: 39.3665, mae: 3.4771, huber: 3.0509, swd: 2.1857, ept: 198.3262
    Epoch [24/50], Test Losses: mse: 28.2294, mae: 2.8939, huber: 2.4749, swd: 1.8404, ept: 215.5151
      Epoch 24 composite train-obj: 1.461042
            No improvement (3.0509), counter 1/5
    Epoch [25/50], Train Losses: mse: 13.4973, mae: 1.8525, huber: 1.4648, swd: 1.0383, ept: 260.8603
    Epoch [25/50], Val Losses: mse: 37.5592, mae: 3.3590, huber: 2.9387, swd: 2.4387, ept: 208.1158
    Epoch [25/50], Test Losses: mse: 25.9484, mae: 2.6771, huber: 2.2659, swd: 1.7096, ept: 229.7747
      Epoch 25 composite train-obj: 1.464835
            No improvement (2.9387), counter 2/5
    Epoch [26/50], Train Losses: mse: 13.0574, mae: 1.8156, huber: 1.4300, swd: 1.0085, ept: 262.1414
    Epoch [26/50], Val Losses: mse: 35.2928, mae: 3.2231, huber: 2.8055, swd: 1.9521, ept: 213.6383
    Epoch [26/50], Test Losses: mse: 25.8108, mae: 2.6702, huber: 2.2625, swd: 1.7732, ept: 231.2959
      Epoch 26 composite train-obj: 1.430042
            Val objective improved 2.9384 → 2.8055, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 11.9859, mae: 1.7109, huber: 1.3319, swd: 0.9223, ept: 266.7841
    Epoch [27/50], Val Losses: mse: 39.3827, mae: 3.4099, huber: 2.9890, swd: 1.9643, ept: 211.1179
    Epoch [27/50], Test Losses: mse: 27.3809, mae: 2.7476, huber: 2.3346, swd: 1.4831, ept: 229.4343
      Epoch 27 composite train-obj: 1.331946
            No improvement (2.9890), counter 1/5
    Epoch [28/50], Train Losses: mse: 12.2984, mae: 1.7443, huber: 1.3619, swd: 0.9194, ept: 266.3590
    Epoch [28/50], Val Losses: mse: 35.5052, mae: 3.3017, huber: 2.8774, swd: 2.2809, ept: 204.3639
    Epoch [28/50], Test Losses: mse: 28.3924, mae: 2.8858, huber: 2.4688, swd: 1.8905, ept: 216.8494
      Epoch 28 composite train-obj: 1.361925
            No improvement (2.8774), counter 2/5
    Epoch [29/50], Train Losses: mse: 12.2452, mae: 1.7399, huber: 1.3582, swd: 0.9251, ept: 266.7208
    Epoch [29/50], Val Losses: mse: 35.3518, mae: 3.2562, huber: 2.8298, swd: 2.0947, ept: 211.1828
    Epoch [29/50], Test Losses: mse: 26.7613, mae: 2.7683, huber: 2.3494, swd: 1.5239, ept: 224.4346
      Epoch 29 composite train-obj: 1.358160
            No improvement (2.8298), counter 3/5
    Epoch [30/50], Train Losses: mse: 11.4450, mae: 1.6598, huber: 1.2831, swd: 0.8630, ept: 270.5286
    Epoch [30/50], Val Losses: mse: 37.1722, mae: 3.2464, huber: 2.8319, swd: 1.9806, ept: 218.8137
    Epoch [30/50], Test Losses: mse: 24.5686, mae: 2.5488, huber: 2.1453, swd: 1.5082, ept: 239.9298
      Epoch 30 composite train-obj: 1.283144
            No improvement (2.8319), counter 4/5
    Epoch [31/50], Train Losses: mse: 11.2723, mae: 1.6430, huber: 1.2672, swd: 0.8394, ept: 271.3586
    Epoch [31/50], Val Losses: mse: 37.4454, mae: 3.2859, huber: 2.8692, swd: 2.1823, ept: 214.2440
    Epoch [31/50], Test Losses: mse: 26.8451, mae: 2.6514, huber: 2.2439, swd: 1.5187, ept: 239.5812
      Epoch 31 composite train-obj: 1.267218
    Epoch [31/50], Test Losses: mse: 25.8104, mae: 2.6702, huber: 2.2625, swd: 1.7732, ept: 231.2722
    Best round's Test MSE: 25.8108, MAE: 2.6702, SWD: 1.7732
    Best round's Validation MSE: 35.2928, MAE: 3.2231, SWD: 1.9521
    Best round's Test verification MSE : 25.8104, MAE: 2.6702, SWD: 1.7732
    Time taken: 106.00 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.2669, mae: 6.5109, huber: 6.0312, swd: 44.4572, ept: 34.8653
    Epoch [1/50], Val Losses: mse: 58.9617, mae: 5.7270, huber: 5.2517, swd: 26.6241, ept: 45.9662
    Epoch [1/50], Test Losses: mse: 53.8084, mae: 5.3161, huber: 4.8446, swd: 27.0711, ept: 44.4353
      Epoch 1 composite train-obj: 6.031158
            Val objective improved inf → 5.2517, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.3294, mae: 4.7816, huber: 4.3168, swd: 16.7082, ept: 97.7545
    Epoch [2/50], Val Losses: mse: 50.9911, mae: 5.0015, huber: 4.5356, swd: 13.9828, ept: 88.9820
    Epoch [2/50], Test Losses: mse: 46.1740, mae: 4.6341, huber: 4.1712, swd: 14.6855, ept: 84.0878
      Epoch 2 composite train-obj: 4.316810
            Val objective improved 5.2517 → 4.5356, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.5400, mae: 4.0903, huber: 3.6372, swd: 9.4079, ept: 141.1290
    Epoch [3/50], Val Losses: mse: 48.2505, mae: 4.6883, huber: 4.2274, swd: 9.5234, ept: 111.4870
    Epoch [3/50], Test Losses: mse: 42.1318, mae: 4.3056, huber: 3.8465, swd: 9.5533, ept: 116.0425
      Epoch 3 composite train-obj: 3.637176
            Val objective improved 4.5356 → 4.2274, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.1816, mae: 3.7019, huber: 3.2573, swd: 6.5736, ept: 164.0044
    Epoch [4/50], Val Losses: mse: 44.0171, mae: 4.3448, huber: 3.8890, swd: 6.7916, ept: 127.7174
    Epoch [4/50], Test Losses: mse: 39.5603, mae: 4.0406, huber: 3.5870, swd: 7.3077, ept: 133.1526
      Epoch 4 composite train-obj: 3.257251
            Val objective improved 4.2274 → 3.8890, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.5531, mae: 3.4684, huber: 3.0290, swd: 5.1154, ept: 176.5052
    Epoch [5/50], Val Losses: mse: 40.6973, mae: 4.0893, huber: 3.6397, swd: 6.8994, ept: 139.4667
    Epoch [5/50], Test Losses: mse: 35.5229, mae: 3.7291, huber: 3.2832, swd: 6.3339, ept: 151.5635
      Epoch 5 composite train-obj: 3.029014
            Val objective improved 3.8890 → 3.6397, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.1935, mae: 3.2533, huber: 2.8199, swd: 4.2465, ept: 186.8028
    Epoch [6/50], Val Losses: mse: 38.5942, mae: 3.9378, huber: 3.4906, swd: 6.1784, ept: 147.1087
    Epoch [6/50], Test Losses: mse: 33.3937, mae: 3.5991, huber: 3.1536, swd: 5.9469, ept: 155.6398
      Epoch 6 composite train-obj: 2.819851
            Val objective improved 3.6397 → 3.4906, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.2795, mae: 3.1105, huber: 2.6803, swd: 3.7217, ept: 193.8314
    Epoch [7/50], Val Losses: mse: 43.6418, mae: 4.0917, huber: 3.6456, swd: 4.7802, ept: 150.8397
    Epoch [7/50], Test Losses: mse: 32.8358, mae: 3.4921, huber: 3.0502, swd: 4.5531, ept: 164.9113
      Epoch 7 composite train-obj: 2.680309
            No improvement (3.6456), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.5217, mae: 2.9637, huber: 2.5372, swd: 3.2735, ept: 201.4359
    Epoch [8/50], Val Losses: mse: 41.4036, mae: 3.9353, huber: 3.4946, swd: 4.3464, ept: 157.3780
    Epoch [8/50], Test Losses: mse: 31.5637, mae: 3.3346, huber: 2.8998, swd: 3.9339, ept: 173.9341
      Epoch 8 composite train-obj: 2.537235
            No improvement (3.4946), counter 2/5
    Epoch [9/50], Train Losses: mse: 24.9065, mae: 2.8409, huber: 2.4177, swd: 2.8755, ept: 206.6702
    Epoch [9/50], Val Losses: mse: 40.6284, mae: 3.9212, huber: 3.4781, swd: 4.4362, ept: 158.9055
    Epoch [9/50], Test Losses: mse: 29.2119, mae: 3.2533, huber: 2.8176, swd: 4.0070, ept: 176.9627
      Epoch 9 composite train-obj: 2.417656
            Val objective improved 3.4906 → 3.4781, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.3671, mae: 2.7170, huber: 2.2973, swd: 2.5874, ept: 213.3657
    Epoch [10/50], Val Losses: mse: 38.7093, mae: 3.6709, huber: 3.2364, swd: 3.7559, ept: 173.0439
    Epoch [10/50], Test Losses: mse: 28.5991, mae: 3.1140, huber: 2.6846, swd: 3.3357, ept: 191.1093
      Epoch 10 composite train-obj: 2.297311
            Val objective improved 3.4781 → 3.2364, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 22.1034, mae: 2.6254, huber: 2.2074, swd: 2.3325, ept: 218.6547
    Epoch [11/50], Val Losses: mse: 39.7507, mae: 3.7208, huber: 3.2840, swd: 3.5463, ept: 172.5999
    Epoch [11/50], Test Losses: mse: 29.5724, mae: 3.1376, huber: 2.7095, swd: 3.4045, ept: 192.4520
      Epoch 11 composite train-obj: 2.207427
            No improvement (3.2840), counter 1/5
    Epoch [12/50], Train Losses: mse: 20.2281, mae: 2.4718, huber: 2.0591, swd: 2.0439, ept: 225.8547
    Epoch [12/50], Val Losses: mse: 39.5632, mae: 3.7049, huber: 3.2692, swd: 3.0145, ept: 175.8331
    Epoch [12/50], Test Losses: mse: 27.9109, mae: 3.0311, huber: 2.6025, swd: 2.5805, ept: 195.6958
      Epoch 12 composite train-obj: 2.059074
            No improvement (3.2692), counter 2/5
    Epoch [13/50], Train Losses: mse: 20.0809, mae: 2.4629, huber: 2.0495, swd: 1.9820, ept: 227.4822
    Epoch [13/50], Val Losses: mse: 36.1870, mae: 3.5169, huber: 3.0815, swd: 3.4601, ept: 181.7920
    Epoch [13/50], Test Losses: mse: 27.1798, mae: 3.0282, huber: 2.5976, swd: 3.1817, ept: 196.7405
      Epoch 13 composite train-obj: 2.049540
            Val objective improved 3.2364 → 3.0815, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.2677, mae: 2.3099, huber: 1.9026, swd: 1.7156, ept: 233.9511
    Epoch [14/50], Val Losses: mse: 39.1232, mae: 3.6708, huber: 3.2346, swd: 3.3820, ept: 179.4849
    Epoch [14/50], Test Losses: mse: 28.5838, mae: 3.0911, huber: 2.6603, swd: 2.7937, ept: 194.0366
      Epoch 14 composite train-obj: 1.902611
            No improvement (3.2346), counter 1/5
    Epoch [15/50], Train Losses: mse: 17.4162, mae: 2.2321, huber: 1.8274, swd: 1.5899, ept: 237.7963
    Epoch [15/50], Val Losses: mse: 41.7112, mae: 3.7064, huber: 3.2746, swd: 3.2130, ept: 188.5768
    Epoch [15/50], Test Losses: mse: 26.5706, mae: 2.8804, huber: 2.4564, swd: 2.4122, ept: 211.0858
      Epoch 15 composite train-obj: 1.827442
            No improvement (3.2746), counter 2/5
    Epoch [16/50], Train Losses: mse: 17.3006, mae: 2.2125, huber: 1.8088, swd: 1.5505, ept: 238.9827
    Epoch [16/50], Val Losses: mse: 35.2900, mae: 3.3707, huber: 2.9413, swd: 3.0712, ept: 187.6143
    Epoch [16/50], Test Losses: mse: 27.0894, mae: 2.8928, huber: 2.4700, swd: 2.3696, ept: 209.8052
      Epoch 16 composite train-obj: 1.808793
            Val objective improved 3.0815 → 2.9413, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 16.0401, mae: 2.1121, huber: 1.7123, swd: 1.4016, ept: 243.1123
    Epoch [17/50], Val Losses: mse: 36.5041, mae: 3.4283, huber: 2.9984, swd: 3.1150, ept: 193.2539
    Epoch [17/50], Test Losses: mse: 27.5450, mae: 2.8993, huber: 2.4755, swd: 2.4820, ept: 213.8143
      Epoch 17 composite train-obj: 1.712332
            No improvement (2.9984), counter 1/5
    Epoch [18/50], Train Losses: mse: 15.9892, mae: 2.0998, huber: 1.7003, swd: 1.4051, ept: 245.1990
    Epoch [18/50], Val Losses: mse: 40.8650, mae: 3.6985, huber: 3.2633, swd: 2.8298, ept: 182.9150
    Epoch [18/50], Test Losses: mse: 26.8390, mae: 2.9591, huber: 2.5300, swd: 2.3685, ept: 201.0985
      Epoch 18 composite train-obj: 1.700349
            No improvement (3.2633), counter 2/5
    Epoch [19/50], Train Losses: mse: 14.9257, mae: 2.0101, huber: 1.6146, swd: 1.2842, ept: 249.5333
    Epoch [19/50], Val Losses: mse: 35.2562, mae: 3.2826, huber: 2.8589, swd: 2.7420, ept: 201.5887
    Epoch [19/50], Test Losses: mse: 26.1505, mae: 2.7561, huber: 2.3391, swd: 1.8627, ept: 221.4689
      Epoch 19 composite train-obj: 1.614610
            Val objective improved 2.9413 → 2.8589, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 14.3568, mae: 1.9603, huber: 1.5668, swd: 1.2140, ept: 252.1623
    Epoch [20/50], Val Losses: mse: 39.6882, mae: 3.5053, huber: 3.0829, swd: 2.7366, ept: 199.6562
    Epoch [20/50], Test Losses: mse: 25.7685, mae: 2.7467, huber: 2.3323, swd: 2.0869, ept: 219.7214
      Epoch 20 composite train-obj: 1.566820
            No improvement (3.0829), counter 1/5
    Epoch [21/50], Train Losses: mse: 14.4016, mae: 1.9558, huber: 1.5630, swd: 1.1899, ept: 252.4966
    Epoch [21/50], Val Losses: mse: 37.4981, mae: 3.3742, huber: 2.9530, swd: 2.6962, ept: 203.4257
    Epoch [21/50], Test Losses: mse: 24.3773, mae: 2.6453, huber: 2.2312, swd: 1.8576, ept: 228.0422
      Epoch 21 composite train-obj: 1.562966
            No improvement (2.9530), counter 2/5
    Epoch [22/50], Train Losses: mse: 13.6381, mae: 1.8758, huber: 1.4872, swd: 1.1213, ept: 257.7591
    Epoch [22/50], Val Losses: mse: 33.8183, mae: 3.1899, huber: 2.7684, swd: 2.7305, ept: 199.8492
    Epoch [22/50], Test Losses: mse: 24.0826, mae: 2.6595, huber: 2.2441, swd: 2.1231, ept: 224.1761
      Epoch 22 composite train-obj: 1.487244
            Val objective improved 2.8589 → 2.7684, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 13.9337, mae: 1.9068, huber: 1.5165, swd: 1.1455, ept: 255.9415
    Epoch [23/50], Val Losses: mse: 37.5448, mae: 3.4120, huber: 2.9873, swd: 2.6317, ept: 201.5064
    Epoch [23/50], Test Losses: mse: 24.9985, mae: 2.7492, huber: 2.3308, swd: 2.0942, ept: 215.2309
      Epoch 23 composite train-obj: 1.516454
            No improvement (2.9873), counter 1/5
    Epoch [24/50], Train Losses: mse: 13.5398, mae: 1.8775, huber: 1.4883, swd: 1.1101, ept: 257.3150
    Epoch [24/50], Val Losses: mse: 38.2210, mae: 3.3861, huber: 2.9642, swd: 2.2820, ept: 202.8447
    Epoch [24/50], Test Losses: mse: 24.3764, mae: 2.6440, huber: 2.2311, swd: 1.8159, ept: 225.3619
      Epoch 24 composite train-obj: 1.488324
            No improvement (2.9642), counter 2/5
    Epoch [25/50], Train Losses: mse: 12.7879, mae: 1.7968, huber: 1.4123, swd: 1.0389, ept: 262.2384
    Epoch [25/50], Val Losses: mse: 33.5164, mae: 3.1807, huber: 2.7596, swd: 2.4372, ept: 201.1259
    Epoch [25/50], Test Losses: mse: 23.8167, mae: 2.6275, huber: 2.2154, swd: 1.9668, ept: 223.7199
      Epoch 25 composite train-obj: 1.412284
            Val objective improved 2.7684 → 2.7596, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 12.1884, mae: 1.7420, huber: 1.3614, swd: 0.9824, ept: 264.0226
    Epoch [26/50], Val Losses: mse: 35.3097, mae: 3.2338, huber: 2.8160, swd: 2.5525, ept: 207.1936
    Epoch [26/50], Test Losses: mse: 23.1288, mae: 2.5426, huber: 2.1345, swd: 1.7227, ept: 230.6495
      Epoch 26 composite train-obj: 1.361432
            No improvement (2.8160), counter 1/5
    Epoch [27/50], Train Losses: mse: 11.9487, mae: 1.7226, huber: 1.3423, swd: 0.9450, ept: 265.4862
    Epoch [27/50], Val Losses: mse: 39.6942, mae: 3.4638, huber: 3.0425, swd: 2.2859, ept: 205.4098
    Epoch [27/50], Test Losses: mse: 24.6579, mae: 2.6273, huber: 2.2155, swd: 1.6534, ept: 227.6176
      Epoch 27 composite train-obj: 1.342321
            No improvement (3.0425), counter 2/5
    Epoch [28/50], Train Losses: mse: 12.1264, mae: 1.7454, huber: 1.3626, swd: 0.9667, ept: 264.7698
    Epoch [28/50], Val Losses: mse: 34.8256, mae: 3.2205, huber: 2.7994, swd: 1.9478, ept: 213.9415
    Epoch [28/50], Test Losses: mse: 24.6684, mae: 2.6626, huber: 2.2473, swd: 1.7042, ept: 226.3086
      Epoch 28 composite train-obj: 1.362642
            No improvement (2.7994), counter 3/5
    Epoch [29/50], Train Losses: mse: 11.3407, mae: 1.6586, huber: 1.2826, swd: 0.8912, ept: 268.5527
    Epoch [29/50], Val Losses: mse: 36.7813, mae: 3.2824, huber: 2.8647, swd: 2.2760, ept: 211.2836
    Epoch [29/50], Test Losses: mse: 24.7229, mae: 2.5910, huber: 2.1830, swd: 1.7904, ept: 237.1321
      Epoch 29 composite train-obj: 1.282570
            No improvement (2.8647), counter 4/5
    Epoch [30/50], Train Losses: mse: 11.2260, mae: 1.6480, huber: 1.2725, swd: 0.8789, ept: 268.9613
    Epoch [30/50], Val Losses: mse: 34.9211, mae: 3.2437, huber: 2.8217, swd: 2.3355, ept: 205.4692
    Epoch [30/50], Test Losses: mse: 24.4197, mae: 2.6482, huber: 2.2379, swd: 1.7752, ept: 225.7957
      Epoch 30 composite train-obj: 1.272490
    Epoch [30/50], Test Losses: mse: 23.8167, mae: 2.6275, huber: 2.2153, swd: 1.9665, ept: 223.7073
    Best round's Test MSE: 23.8167, MAE: 2.6275, SWD: 1.9668
    Best round's Validation MSE: 33.5164, MAE: 3.1807, SWD: 2.4372
    Best round's Test verification MSE : 23.8167, MAE: 2.6275, SWD: 1.9665
    Time taken: 107.50 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1641)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 24.8548 ± 0.8162
      mae: 2.6204 ± 0.0439
      huber: 2.2120 ± 0.0426
      swd: 1.8212 ± 0.1049
      ept: 232.9040 ± 8.2342
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 33.3762 ± 1.6252
      mae: 3.1309 ± 0.1019
      huber: 2.7138 ± 0.0990
      swd: 2.1340 ± 0.2158
      ept: 213.2742 ± 9.7737
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 332.65 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1641
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### AB: Koopman but with no shift in z_push
Yet, Koopman alone *is* useful, it is just that shift overrides it.
Basically, you have a free interpretability enhancement,
without hurting the performance gain by shift term.  


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128, 
    # ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=True, 
    ablate_shift_inside_scale=False,
    householder_reflects_latent = 2,
    householder_reflects_data = 4,
    mixing_strategy='delay_only', 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
    ablate_no_koopman=False, ### HERE
    ablate_no_shift_in_z_push=True, ### HERE
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 76.5795, mae: 6.6441, huber: 6.1634, swd: 45.2190, ept: 31.0791
    Epoch [1/50], Val Losses: mse: 61.1763, mae: 5.9559, huber: 5.4783, swd: 27.7987, ept: 34.6733
    Epoch [1/50], Test Losses: mse: 56.7164, mae: 5.5811, huber: 5.1067, swd: 27.8736, ept: 36.5755
      Epoch 1 composite train-obj: 6.163448
            Val objective improved inf → 5.4783, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.9553, mae: 5.0243, huber: 4.5563, swd: 19.3773, ept: 81.3188
    Epoch [2/50], Val Losses: mse: 53.3488, mae: 5.2062, huber: 4.7384, swd: 18.7098, ept: 79.8762
    Epoch [2/50], Test Losses: mse: 48.5573, mae: 4.8582, huber: 4.3929, swd: 18.8054, ept: 76.9520
      Epoch 2 composite train-obj: 4.556300
            Val objective improved 5.4783 → 4.7384, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 43.1237, mae: 4.3759, huber: 3.9189, swd: 12.2448, ept: 124.1504
    Epoch [3/50], Val Losses: mse: 48.9216, mae: 4.8481, huber: 4.3850, swd: 12.6293, ept: 103.9862
    Epoch [3/50], Test Losses: mse: 44.0814, mae: 4.4724, huber: 4.0132, swd: 13.0831, ept: 105.0816
      Epoch 3 composite train-obj: 3.918861
            Val objective improved 4.7384 → 4.3850, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 38.2546, mae: 3.9489, huber: 3.4998, swd: 8.5828, ept: 150.6731
    Epoch [4/50], Val Losses: mse: 45.7790, mae: 4.5429, huber: 4.0842, swd: 8.9655, ept: 114.6723
    Epoch [4/50], Test Losses: mse: 42.0060, mae: 4.2572, huber: 3.8014, swd: 9.2537, ept: 118.2550
      Epoch 4 composite train-obj: 3.499774
            Val objective improved 4.3850 → 4.0842, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 34.3395, mae: 3.6302, huber: 3.1866, swd: 6.0857, ept: 165.3747
    Epoch [5/50], Val Losses: mse: 44.3668, mae: 4.3866, huber: 3.9309, swd: 7.6304, ept: 122.6461
    Epoch [5/50], Test Losses: mse: 38.5439, mae: 3.9874, huber: 3.5365, swd: 7.3758, ept: 130.0599
      Epoch 5 composite train-obj: 3.186609
            Val objective improved 4.0842 → 3.9309, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.6476, mae: 3.4048, huber: 2.9661, swd: 4.8661, ept: 177.1705
    Epoch [6/50], Val Losses: mse: 44.9633, mae: 4.3585, huber: 3.9024, swd: 6.0160, ept: 131.1744
    Epoch [6/50], Test Losses: mse: 36.5384, mae: 3.7910, huber: 3.3413, swd: 5.3905, ept: 141.7861
      Epoch 6 composite train-obj: 2.966111
            Val objective improved 3.9309 → 3.9024, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 29.4011, mae: 3.2225, huber: 2.7880, swd: 4.0036, ept: 185.1688
    Epoch [7/50], Val Losses: mse: 45.5221, mae: 4.3301, huber: 3.8766, swd: 5.1216, ept: 135.3189
    Epoch [7/50], Test Losses: mse: 36.0573, mae: 3.7358, huber: 3.2878, swd: 4.6284, ept: 148.3161
      Epoch 7 composite train-obj: 2.787951
            Val objective improved 3.9024 → 3.8766, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 27.3949, mae: 3.0468, huber: 2.6173, swd: 3.3686, ept: 192.8791
    Epoch [8/50], Val Losses: mse: 43.9016, mae: 4.1434, huber: 3.6977, swd: 4.8974, ept: 148.1132
    Epoch [8/50], Test Losses: mse: 34.6838, mae: 3.5817, huber: 3.1408, swd: 4.3434, ept: 160.2615
      Epoch 8 composite train-obj: 2.617267
            Val objective improved 3.8766 → 3.6977, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 25.9587, mae: 2.9410, huber: 2.5137, swd: 3.0594, ept: 197.7190
    Epoch [9/50], Val Losses: mse: 44.7067, mae: 4.1477, huber: 3.7019, swd: 4.5818, ept: 150.5427
    Epoch [9/50], Test Losses: mse: 34.4944, mae: 3.5025, huber: 3.0636, swd: 3.5096, ept: 165.2294
      Epoch 9 composite train-obj: 2.513705
            No improvement (3.7019), counter 1/5
    Epoch [10/50], Train Losses: mse: 24.5128, mae: 2.8273, huber: 2.4029, swd: 2.7274, ept: 202.7759
    Epoch [10/50], Val Losses: mse: 44.6095, mae: 4.1314, huber: 3.6869, swd: 4.5723, ept: 152.9790
    Epoch [10/50], Test Losses: mse: 33.3071, mae: 3.4403, huber: 3.0028, swd: 3.7353, ept: 170.8390
      Epoch 10 composite train-obj: 2.402928
            Val objective improved 3.6977 → 3.6869, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 22.9118, mae: 2.6885, huber: 2.2689, swd: 2.4405, ept: 209.7333
    Epoch [11/50], Val Losses: mse: 43.9134, mae: 4.0419, huber: 3.5983, swd: 4.1460, ept: 160.1362
    Epoch [11/50], Test Losses: mse: 33.9338, mae: 3.4370, huber: 3.0003, swd: 3.4545, ept: 177.0532
      Epoch 11 composite train-obj: 2.268850
            Val objective improved 3.6869 → 3.5983, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 21.9817, mae: 2.6225, huber: 2.2041, swd: 2.2650, ept: 213.1861
    Epoch [12/50], Val Losses: mse: 42.3636, mae: 3.9275, huber: 3.4856, swd: 4.4315, ept: 163.6681
    Epoch [12/50], Test Losses: mse: 32.4486, mae: 3.3273, huber: 2.8921, swd: 3.5477, ept: 180.4398
      Epoch 12 composite train-obj: 2.204065
            Val objective improved 3.5983 → 3.4856, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 20.7929, mae: 2.5149, huber: 2.1008, swd: 2.1249, ept: 219.5540
    Epoch [13/50], Val Losses: mse: 40.4670, mae: 3.8030, huber: 3.3645, swd: 3.7182, ept: 168.4677
    Epoch [13/50], Test Losses: mse: 32.8674, mae: 3.3064, huber: 2.8741, swd: 3.2129, ept: 188.1896
      Epoch 13 composite train-obj: 2.100816
            Val objective improved 3.4856 → 3.3645, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 19.8987, mae: 2.4452, huber: 2.0328, swd: 1.9777, ept: 223.4796
    Epoch [14/50], Val Losses: mse: 41.0540, mae: 3.8087, huber: 3.3711, swd: 3.5467, ept: 171.8095
    Epoch [14/50], Test Losses: mse: 32.4053, mae: 3.2566, huber: 2.8262, swd: 3.2192, ept: 191.2032
      Epoch 14 composite train-obj: 2.032821
            No improvement (3.3711), counter 1/5
    Epoch [15/50], Train Losses: mse: 19.4275, mae: 2.4122, huber: 2.0003, swd: 1.8570, ept: 224.9067
    Epoch [15/50], Val Losses: mse: 40.5763, mae: 3.7387, huber: 3.3025, swd: 3.0912, ept: 178.1626
    Epoch [15/50], Test Losses: mse: 31.7043, mae: 3.1554, huber: 2.7271, swd: 2.5137, ept: 198.0278
      Epoch 15 composite train-obj: 2.000283
            Val objective improved 3.3645 → 3.3025, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 18.1989, mae: 2.3032, huber: 1.8956, swd: 1.7040, ept: 230.4172
    Epoch [16/50], Val Losses: mse: 43.8193, mae: 3.8496, huber: 3.4132, swd: 3.2855, ept: 179.3702
    Epoch [16/50], Test Losses: mse: 31.0415, mae: 3.1317, huber: 2.7030, swd: 2.5737, ept: 198.2874
      Epoch 16 composite train-obj: 1.895623
            No improvement (3.4132), counter 1/5
    Epoch [17/50], Train Losses: mse: 17.9999, mae: 2.2848, huber: 1.8777, swd: 1.6724, ept: 232.3968
    Epoch [17/50], Val Losses: mse: 40.7711, mae: 3.7095, huber: 3.2764, swd: 3.3514, ept: 180.0736
    Epoch [17/50], Test Losses: mse: 32.1088, mae: 3.1583, huber: 2.7326, swd: 2.5910, ept: 203.4859
      Epoch 17 composite train-obj: 1.877731
            Val objective improved 3.3025 → 3.2764, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 17.0281, mae: 2.2006, huber: 1.7970, swd: 1.5551, ept: 236.5181
    Epoch [18/50], Val Losses: mse: 40.8358, mae: 3.6692, huber: 3.2365, swd: 2.9552, ept: 185.9591
    Epoch [18/50], Test Losses: mse: 30.6303, mae: 3.0489, huber: 2.6235, swd: 2.2529, ept: 204.2666
      Epoch 18 composite train-obj: 1.796992
            Val objective improved 3.2764 → 3.2365, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 16.1437, mae: 2.1189, huber: 1.7189, swd: 1.4565, ept: 241.1771
    Epoch [19/50], Val Losses: mse: 40.4896, mae: 3.6801, huber: 3.2466, swd: 2.8389, ept: 185.1727
    Epoch [19/50], Test Losses: mse: 31.0679, mae: 3.1088, huber: 2.6827, swd: 2.4355, ept: 201.5930
      Epoch 19 composite train-obj: 1.718918
            No improvement (3.2466), counter 1/5
    Epoch [20/50], Train Losses: mse: 15.7871, mae: 2.0924, huber: 1.6933, swd: 1.4036, ept: 242.7182
    Epoch [20/50], Val Losses: mse: 39.7547, mae: 3.6132, huber: 3.1846, swd: 3.3493, ept: 186.7461
    Epoch [20/50], Test Losses: mse: 28.7622, mae: 2.9464, huber: 2.5250, swd: 2.3666, ept: 204.6623
      Epoch 20 composite train-obj: 1.693280
            Val objective improved 3.2365 → 3.1846, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 15.1636, mae: 2.0276, huber: 1.6323, swd: 1.3418, ept: 246.3801
    Epoch [21/50], Val Losses: mse: 39.3160, mae: 3.5793, huber: 3.1504, swd: 3.0712, ept: 186.3222
    Epoch [21/50], Test Losses: mse: 29.7259, mae: 3.0011, huber: 2.5800, swd: 2.4002, ept: 205.9176
      Epoch 21 composite train-obj: 1.632271
            Val objective improved 3.1846 → 3.1504, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 15.0877, mae: 2.0351, huber: 1.6382, swd: 1.3087, ept: 246.2454
    Epoch [22/50], Val Losses: mse: 38.5318, mae: 3.5186, huber: 3.0909, swd: 2.7972, ept: 192.4325
    Epoch [22/50], Test Losses: mse: 30.2971, mae: 2.9935, huber: 2.5733, swd: 2.1041, ept: 207.1954
      Epoch 22 composite train-obj: 1.638188
            Val objective improved 3.1504 → 3.0909, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 14.4121, mae: 1.9707, huber: 1.5772, swd: 1.2439, ept: 249.2134
    Epoch [23/50], Val Losses: mse: 39.2298, mae: 3.5713, huber: 3.1426, swd: 3.0271, ept: 190.1579
    Epoch [23/50], Test Losses: mse: 28.4059, mae: 2.9129, huber: 2.4931, swd: 2.3714, ept: 209.0497
      Epoch 23 composite train-obj: 1.577243
            No improvement (3.1426), counter 1/5
    Epoch [24/50], Train Losses: mse: 13.6526, mae: 1.9025, huber: 1.5121, swd: 1.1763, ept: 253.0494
    Epoch [24/50], Val Losses: mse: 35.3524, mae: 3.3516, huber: 2.9256, swd: 3.1155, ept: 192.9998
    Epoch [24/50], Test Losses: mse: 27.7061, mae: 2.8415, huber: 2.4239, swd: 2.3258, ept: 213.0304
      Epoch 24 composite train-obj: 1.512141
            Val objective improved 3.0909 → 2.9256, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 13.4971, mae: 1.8836, huber: 1.4943, swd: 1.1597, ept: 254.4719
    Epoch [25/50], Val Losses: mse: 36.8374, mae: 3.4208, huber: 2.9951, swd: 2.7832, ept: 194.5499
    Epoch [25/50], Test Losses: mse: 31.0293, mae: 2.9768, huber: 2.5589, swd: 2.1212, ept: 214.8920
      Epoch 25 composite train-obj: 1.494298
            No improvement (2.9951), counter 1/5
    Epoch [26/50], Train Losses: mse: 13.5682, mae: 1.8928, huber: 1.5027, swd: 1.1455, ept: 254.6500
    Epoch [26/50], Val Losses: mse: 37.9473, mae: 3.4623, huber: 3.0349, swd: 3.0086, ept: 195.2212
    Epoch [26/50], Test Losses: mse: 28.1651, mae: 2.8626, huber: 2.4440, swd: 2.2570, ept: 215.9315
      Epoch 26 composite train-obj: 1.502742
            No improvement (3.0349), counter 2/5
    Epoch [27/50], Train Losses: mse: 12.7639, mae: 1.8129, huber: 1.4275, swd: 1.0739, ept: 258.9667
    Epoch [27/50], Val Losses: mse: 37.5062, mae: 3.4049, huber: 2.9824, swd: 2.7674, ept: 199.5741
    Epoch [27/50], Test Losses: mse: 27.5148, mae: 2.7736, huber: 2.3605, swd: 1.9681, ept: 223.0650
      Epoch 27 composite train-obj: 1.427477
            No improvement (2.9824), counter 3/5
    Epoch [28/50], Train Losses: mse: 12.6076, mae: 1.7892, huber: 1.4058, swd: 1.0663, ept: 260.0811
    Epoch [28/50], Val Losses: mse: 36.0549, mae: 3.3625, huber: 2.9396, swd: 2.8116, ept: 196.4741
    Epoch [28/50], Test Losses: mse: 28.1531, mae: 2.8317, huber: 2.4166, swd: 2.1792, ept: 217.8048
      Epoch 28 composite train-obj: 1.405770
            No improvement (2.9396), counter 4/5
    Epoch [29/50], Train Losses: mse: 12.4523, mae: 1.7800, huber: 1.3961, swd: 1.0472, ept: 260.5386
    Epoch [29/50], Val Losses: mse: 37.5473, mae: 3.4138, huber: 2.9906, swd: 2.8554, ept: 200.4036
    Epoch [29/50], Test Losses: mse: 29.8186, mae: 2.9025, huber: 2.4877, swd: 2.2796, ept: 220.3782
      Epoch 29 composite train-obj: 1.396124
    Epoch [29/50], Test Losses: mse: 27.7068, mae: 2.8416, huber: 2.4240, swd: 2.3258, ept: 213.0009
    Best round's Test MSE: 27.7061, MAE: 2.8415, SWD: 2.3258
    Best round's Validation MSE: 35.3524, MAE: 3.3516, SWD: 3.1155
    Best round's Test verification MSE : 27.7068, MAE: 2.8416, SWD: 2.3258
    Time taken: 108.01 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 77.8024, mae: 6.6796, huber: 6.1990, swd: 46.6331, ept: 30.3637
    Epoch [1/50], Val Losses: mse: 60.8261, mae: 5.8881, huber: 5.4116, swd: 28.4712, ept: 40.9055
    Epoch [1/50], Test Losses: mse: 56.0379, mae: 5.5025, huber: 5.0297, swd: 29.1335, ept: 39.8192
      Epoch 1 composite train-obj: 6.198967
            Val objective improved inf → 5.4116, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.8358, mae: 5.0156, huber: 4.5477, swd: 18.6171, ept: 83.6814
    Epoch [2/50], Val Losses: mse: 52.6788, mae: 5.2103, huber: 4.7414, swd: 15.7165, ept: 74.8842
    Epoch [2/50], Test Losses: mse: 48.0217, mae: 4.8473, huber: 4.3823, swd: 16.8141, ept: 71.4224
      Epoch 2 composite train-obj: 4.547739
            Val objective improved 5.4116 → 4.7414, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 42.2477, mae: 4.3461, huber: 3.8889, swd: 10.7796, ept: 120.8380
    Epoch [3/50], Val Losses: mse: 49.2209, mae: 4.8133, huber: 4.3509, swd: 10.4369, ept: 95.4693
    Epoch [3/50], Test Losses: mse: 43.8376, mae: 4.4262, huber: 3.9670, swd: 10.2811, ept: 100.3289
      Epoch 3 composite train-obj: 3.888929
            Val objective improved 4.7414 → 4.3509, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.2294, mae: 3.8859, huber: 3.4372, swd: 7.0093, ept: 147.4106
    Epoch [4/50], Val Losses: mse: 44.1721, mae: 4.4234, huber: 3.9674, swd: 7.8783, ept: 116.6895
    Epoch [4/50], Test Losses: mse: 40.2264, mae: 4.1304, huber: 3.6762, swd: 7.6943, ept: 120.4291
      Epoch 4 composite train-obj: 3.437241
            Val objective improved 4.3509 → 3.9674, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 34.1982, mae: 3.6413, huber: 3.1969, swd: 5.5915, ept: 161.9847
    Epoch [5/50], Val Losses: mse: 44.0418, mae: 4.3716, huber: 3.9148, swd: 6.4301, ept: 123.1395
    Epoch [5/50], Test Losses: mse: 38.3321, mae: 3.9740, huber: 3.5196, swd: 6.1967, ept: 134.1694
      Epoch 5 composite train-obj: 3.196939
            Val objective improved 3.9674 → 3.9148, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.3169, mae: 3.3852, huber: 2.9475, swd: 4.4450, ept: 176.9147
    Epoch [6/50], Val Losses: mse: 43.2313, mae: 4.2247, huber: 3.7736, swd: 6.4311, ept: 135.3204
    Epoch [6/50], Test Losses: mse: 35.8367, mae: 3.7706, huber: 3.3244, swd: 5.7133, ept: 149.4436
      Epoch 6 composite train-obj: 2.947483
            Val objective improved 3.9148 → 3.7736, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.9729, mae: 3.1959, huber: 2.7620, swd: 3.7028, ept: 187.0989
    Epoch [7/50], Val Losses: mse: 39.6728, mae: 3.9579, huber: 3.5106, swd: 4.5749, ept: 148.3097
    Epoch [7/50], Test Losses: mse: 33.8864, mae: 3.6052, huber: 3.1612, swd: 4.6944, ept: 159.0576
      Epoch 7 composite train-obj: 2.762039
            Val objective improved 3.7736 → 3.5106, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 27.4266, mae: 3.0739, huber: 2.6427, swd: 3.2326, ept: 193.1289
    Epoch [8/50], Val Losses: mse: 43.5358, mae: 4.1391, huber: 3.6911, swd: 5.1155, ept: 149.2369
    Epoch [8/50], Test Losses: mse: 34.8915, mae: 3.6013, huber: 3.1587, swd: 4.4581, ept: 161.1069
      Epoch 8 composite train-obj: 2.642699
            No improvement (3.6911), counter 1/5
    Epoch [9/50], Train Losses: mse: 25.5886, mae: 2.9109, huber: 2.4845, swd: 2.8061, ept: 201.4315
    Epoch [9/50], Val Losses: mse: 41.8728, mae: 3.9923, huber: 3.5490, swd: 4.5575, ept: 155.2605
    Epoch [9/50], Test Losses: mse: 32.5977, mae: 3.4383, huber: 2.9997, swd: 3.6869, ept: 166.9187
      Epoch 9 composite train-obj: 2.484472
            No improvement (3.5490), counter 2/5
    Epoch [10/50], Train Losses: mse: 24.2663, mae: 2.8131, huber: 2.3888, swd: 2.5294, ept: 205.5732
    Epoch [10/50], Val Losses: mse: 43.4620, mae: 4.1062, huber: 3.6600, swd: 4.6449, ept: 156.8764
    Epoch [10/50], Test Losses: mse: 34.9149, mae: 3.5599, huber: 3.1196, swd: 3.9304, ept: 173.5183
      Epoch 10 composite train-obj: 2.388839
            No improvement (3.6600), counter 3/5
    Epoch [11/50], Train Losses: mse: 23.0878, mae: 2.7100, huber: 2.2889, swd: 2.3308, ept: 212.2873
    Epoch [11/50], Val Losses: mse: 43.4690, mae: 3.9946, huber: 3.5525, swd: 4.0967, ept: 165.8489
    Epoch [11/50], Test Losses: mse: 31.2684, mae: 3.2857, huber: 2.8507, swd: 3.2246, ept: 183.6673
      Epoch 11 composite train-obj: 2.288857
            No improvement (3.5525), counter 4/5
    Epoch [12/50], Train Losses: mse: 21.5615, mae: 2.5834, huber: 2.1660, swd: 2.0708, ept: 218.0116
    Epoch [12/50], Val Losses: mse: 42.1539, mae: 3.9425, huber: 3.5010, swd: 4.4999, ept: 168.8105
    Epoch [12/50], Test Losses: mse: 32.1196, mae: 3.3349, huber: 2.8998, swd: 3.4230, ept: 186.9662
      Epoch 12 composite train-obj: 2.165972
            Val objective improved 3.5106 → 3.5010, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 20.8958, mae: 2.5404, huber: 2.1238, swd: 2.0023, ept: 220.2246
    Epoch [13/50], Val Losses: mse: 41.8325, mae: 3.8478, huber: 3.4086, swd: 3.9162, ept: 174.9347
    Epoch [13/50], Test Losses: mse: 30.3769, mae: 3.1624, huber: 2.7299, swd: 2.8882, ept: 193.0339
      Epoch 13 composite train-obj: 2.123760
            Val objective improved 3.5010 → 3.4086, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 19.7362, mae: 2.4408, huber: 2.0275, swd: 1.8018, ept: 225.2158
    Epoch [14/50], Val Losses: mse: 40.3169, mae: 3.7517, huber: 3.3128, swd: 3.9992, ept: 175.3175
    Epoch [14/50], Test Losses: mse: 31.5550, mae: 3.2118, huber: 2.7805, swd: 3.0464, ept: 192.8158
      Epoch 14 composite train-obj: 2.027491
            Val objective improved 3.4086 → 3.3128, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 18.6974, mae: 2.3466, huber: 1.9374, swd: 1.6666, ept: 230.2279
    Epoch [15/50], Val Losses: mse: 43.9741, mae: 3.8923, huber: 3.4561, swd: 3.3202, ept: 177.0834
    Epoch [15/50], Test Losses: mse: 32.0996, mae: 3.2364, huber: 2.8054, swd: 2.7270, ept: 192.6783
      Epoch 15 composite train-obj: 1.937377
            No improvement (3.4561), counter 1/5
    Epoch [16/50], Train Losses: mse: 17.8464, mae: 2.2673, huber: 1.8611, swd: 1.5643, ept: 234.1019
    Epoch [16/50], Val Losses: mse: 43.1980, mae: 3.8698, huber: 3.4320, swd: 3.7373, ept: 172.8780
    Epoch [16/50], Test Losses: mse: 31.2779, mae: 3.1842, huber: 2.7541, swd: 2.6452, ept: 192.3833
      Epoch 16 composite train-obj: 1.861089
            No improvement (3.4320), counter 2/5
    Epoch [17/50], Train Losses: mse: 17.2165, mae: 2.2070, huber: 1.8039, swd: 1.4789, ept: 236.9872
    Epoch [17/50], Val Losses: mse: 40.3131, mae: 3.6771, huber: 3.2441, swd: 3.4028, ept: 180.2021
    Epoch [17/50], Test Losses: mse: 30.1157, mae: 3.0591, huber: 2.6333, swd: 2.4400, ept: 201.8579
      Epoch 17 composite train-obj: 1.803939
            Val objective improved 3.3128 → 3.2441, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 17.0287, mae: 2.1945, huber: 1.7910, swd: 1.4526, ept: 238.4121
    Epoch [18/50], Val Losses: mse: 39.1649, mae: 3.5923, huber: 3.1604, swd: 3.2331, ept: 183.2234
    Epoch [18/50], Test Losses: mse: 28.9375, mae: 3.0050, huber: 2.5805, swd: 2.4639, ept: 200.1507
      Epoch 18 composite train-obj: 1.791043
            Val objective improved 3.2441 → 3.1604, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 16.2645, mae: 2.1129, huber: 1.7137, swd: 1.3841, ept: 243.1217
    Epoch [19/50], Val Losses: mse: 43.1639, mae: 3.7746, huber: 3.3427, swd: 3.1941, ept: 183.5577
    Epoch [19/50], Test Losses: mse: 29.1625, mae: 2.9870, huber: 2.5640, swd: 2.2627, ept: 206.1021
      Epoch 19 composite train-obj: 1.713653
            No improvement (3.3427), counter 1/5
    Epoch [20/50], Train Losses: mse: 16.1179, mae: 2.1025, huber: 1.7039, swd: 1.3455, ept: 243.5637
    Epoch [20/50], Val Losses: mse: 40.5983, mae: 3.6867, huber: 3.2540, swd: 3.7383, ept: 182.2046
    Epoch [20/50], Test Losses: mse: 31.3498, mae: 3.1401, huber: 2.7124, swd: 2.6666, ept: 199.3488
      Epoch 20 composite train-obj: 1.703887
            No improvement (3.2540), counter 2/5
    Epoch [21/50], Train Losses: mse: 16.0498, mae: 2.0922, huber: 1.6938, swd: 1.3369, ept: 245.0331
    Epoch [21/50], Val Losses: mse: 39.1576, mae: 3.5482, huber: 3.1205, swd: 3.3811, ept: 189.2291
    Epoch [21/50], Test Losses: mse: 30.2193, mae: 3.0245, huber: 2.6031, swd: 2.4791, ept: 205.0923
      Epoch 21 composite train-obj: 1.693785
            Val objective improved 3.1604 → 3.1205, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 14.9683, mae: 2.0010, huber: 1.6066, swd: 1.2388, ept: 248.7887
    Epoch [22/50], Val Losses: mse: 44.7816, mae: 3.7846, huber: 3.3564, swd: 3.1272, ept: 186.5791
    Epoch [22/50], Test Losses: mse: 30.3272, mae: 2.9941, huber: 2.5739, swd: 2.2465, ept: 211.1418
      Epoch 22 composite train-obj: 1.606569
            No improvement (3.3564), counter 1/5
    Epoch [23/50], Train Losses: mse: 14.4300, mae: 1.9460, huber: 1.5547, swd: 1.1886, ept: 252.3119
    Epoch [23/50], Val Losses: mse: 42.0806, mae: 3.6829, huber: 3.2528, swd: 3.4773, ept: 193.4160
    Epoch [23/50], Test Losses: mse: 28.9389, mae: 2.9412, huber: 2.5174, swd: 2.1704, ept: 215.1774
      Epoch 23 composite train-obj: 1.554721
            No improvement (3.2528), counter 2/5
    Epoch [24/50], Train Losses: mse: 13.9973, mae: 1.9023, huber: 1.5132, swd: 1.1486, ept: 255.0166
    Epoch [24/50], Val Losses: mse: 40.3875, mae: 3.5758, huber: 3.1484, swd: 3.0934, ept: 194.7746
    Epoch [24/50], Test Losses: mse: 29.5418, mae: 2.9279, huber: 2.5080, swd: 1.9176, ept: 217.4331
      Epoch 24 composite train-obj: 1.513210
            No improvement (3.1484), counter 3/5
    Epoch [25/50], Train Losses: mse: 13.8188, mae: 1.8816, huber: 1.4940, swd: 1.1255, ept: 256.5161
    Epoch [25/50], Val Losses: mse: 41.4070, mae: 3.5995, huber: 3.1743, swd: 2.9800, ept: 193.6061
    Epoch [25/50], Test Losses: mse: 27.4681, mae: 2.8398, huber: 2.4228, swd: 2.1137, ept: 214.6868
      Epoch 25 composite train-obj: 1.493955
            No improvement (3.1743), counter 4/5
    Epoch [26/50], Train Losses: mse: 13.6895, mae: 1.8666, huber: 1.4799, swd: 1.1147, ept: 257.7347
    Epoch [26/50], Val Losses: mse: 38.1590, mae: 3.4750, huber: 3.0473, swd: 2.9079, ept: 198.4602
    Epoch [26/50], Test Losses: mse: 27.2271, mae: 2.8460, huber: 2.4253, swd: 2.0733, ept: 216.2701
      Epoch 26 composite train-obj: 1.479946
            Val objective improved 3.1205 → 3.0473, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 13.4956, mae: 1.8460, huber: 1.4604, swd: 1.0958, ept: 259.5347
    Epoch [27/50], Val Losses: mse: 40.6431, mae: 3.5622, huber: 3.1372, swd: 3.2236, ept: 193.4969
    Epoch [27/50], Test Losses: mse: 30.4551, mae: 2.9725, huber: 2.5546, swd: 2.0909, ept: 213.2227
      Epoch 27 composite train-obj: 1.460430
            No improvement (3.1372), counter 1/5
    Epoch [28/50], Train Losses: mse: 12.6833, mae: 1.7810, huber: 1.3984, swd: 1.0220, ept: 261.6388
    Epoch [28/50], Val Losses: mse: 41.5291, mae: 3.6731, huber: 3.2432, swd: 3.2031, ept: 190.2101
    Epoch [28/50], Test Losses: mse: 29.3030, mae: 2.9547, huber: 2.5353, swd: 2.2033, ept: 210.8487
      Epoch 28 composite train-obj: 1.398434
            No improvement (3.2432), counter 2/5
    Epoch [29/50], Train Losses: mse: 12.9460, mae: 1.8120, huber: 1.4268, swd: 1.0620, ept: 260.2032
    Epoch [29/50], Val Losses: mse: 42.8248, mae: 3.6595, huber: 3.2356, swd: 2.8707, ept: 195.4694
    Epoch [29/50], Test Losses: mse: 28.1507, mae: 2.8415, huber: 2.4265, swd: 2.0389, ept: 217.2772
      Epoch 29 composite train-obj: 1.426766
            No improvement (3.2356), counter 3/5
    Epoch [30/50], Train Losses: mse: 12.3310, mae: 1.7388, huber: 1.3592, swd: 1.0018, ept: 264.7572
    Epoch [30/50], Val Losses: mse: 41.7741, mae: 3.6164, huber: 3.1934, swd: 3.1454, ept: 195.7955
    Epoch [30/50], Test Losses: mse: 28.7930, mae: 2.8308, huber: 2.4190, swd: 1.9775, ept: 223.6164
      Epoch 30 composite train-obj: 1.359181
            No improvement (3.1934), counter 4/5
    Epoch [31/50], Train Losses: mse: 12.5405, mae: 1.7613, huber: 1.3795, swd: 0.9956, ept: 263.7607
    Epoch [31/50], Val Losses: mse: 42.1048, mae: 3.6074, huber: 3.1841, swd: 2.7325, ept: 200.3916
    Epoch [31/50], Test Losses: mse: 27.5739, mae: 2.7755, huber: 2.3643, swd: 1.9510, ept: 225.1175
      Epoch 31 composite train-obj: 1.379549
    Epoch [31/50], Test Losses: mse: 27.2280, mae: 2.8460, huber: 2.4253, swd: 2.0730, ept: 216.2551
    Best round's Test MSE: 27.2271, MAE: 2.8460, SWD: 2.0733
    Best round's Validation MSE: 38.1590, MAE: 3.4750, SWD: 2.9079
    Best round's Test verification MSE : 27.2280, MAE: 2.8460, SWD: 2.0730
    Time taken: 122.33 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 76.4461, mae: 6.5595, huber: 6.0797, swd: 45.8521, ept: 34.6354
    Epoch [1/50], Val Losses: mse: 60.9300, mae: 5.8806, huber: 5.4041, swd: 26.6548, ept: 39.6464
    Epoch [1/50], Test Losses: mse: 56.6494, mae: 5.5192, huber: 5.0455, swd: 26.9410, ept: 41.3897
      Epoch 1 composite train-obj: 6.079688
            Val objective improved inf → 5.4041, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.4168, mae: 4.8618, huber: 4.3961, swd: 18.2263, ept: 91.9024
    Epoch [2/50], Val Losses: mse: 52.9369, mae: 5.1840, huber: 4.7158, swd: 17.9082, ept: 76.7447
    Epoch [2/50], Test Losses: mse: 47.3190, mae: 4.7855, huber: 4.3204, swd: 18.5499, ept: 72.7266
      Epoch 2 composite train-obj: 4.396053
            Val objective improved 5.4041 → 4.7158, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 41.7214, mae: 4.2581, huber: 3.8027, swd: 11.4651, ept: 124.6721
    Epoch [3/50], Val Losses: mse: 49.7253, mae: 4.8175, huber: 4.3549, swd: 10.9010, ept: 100.5646
    Epoch [3/50], Test Losses: mse: 44.2284, mae: 4.4486, huber: 3.9887, swd: 11.9197, ept: 97.6766
      Epoch 3 composite train-obj: 3.802686
            Val objective improved 4.7158 → 4.3549, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.9975, mae: 3.8477, huber: 3.3999, swd: 7.8428, ept: 147.4906
    Epoch [4/50], Val Losses: mse: 46.3886, mae: 4.5234, huber: 4.0637, swd: 7.6844, ept: 113.9824
    Epoch [4/50], Test Losses: mse: 41.0002, mae: 4.1487, huber: 3.6934, swd: 7.9753, ept: 116.8708
      Epoch 4 composite train-obj: 3.399883
            Val objective improved 4.3549 → 4.0637, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.5878, mae: 3.5563, huber: 3.1143, swd: 5.7982, ept: 165.4735
    Epoch [5/50], Val Losses: mse: 42.8115, mae: 4.2233, huber: 3.7722, swd: 7.7637, ept: 123.4673
    Epoch [5/50], Test Losses: mse: 37.5745, mae: 3.8958, huber: 3.4469, swd: 7.6401, ept: 127.9868
      Epoch 5 composite train-obj: 3.114322
            Val objective improved 4.0637 → 3.7722, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.9515, mae: 3.3280, huber: 2.8914, swd: 4.6812, ept: 177.6007
    Epoch [6/50], Val Losses: mse: 43.0796, mae: 4.1681, huber: 3.7179, swd: 6.2374, ept: 134.3103
    Epoch [6/50], Test Losses: mse: 35.4906, mae: 3.6889, huber: 3.2435, swd: 5.6192, ept: 140.3633
      Epoch 6 composite train-obj: 2.891370
            Val objective improved 3.7722 → 3.7179, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.5056, mae: 3.1476, huber: 2.7139, swd: 3.9406, ept: 185.1714
    Epoch [7/50], Val Losses: mse: 46.6847, mae: 4.3460, huber: 3.8934, swd: 5.8955, ept: 134.0434
    Epoch [7/50], Test Losses: mse: 36.1034, mae: 3.7081, huber: 3.2613, swd: 5.1917, ept: 142.4753
      Epoch 7 composite train-obj: 2.713861
            No improvement (3.8934), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.3108, mae: 2.9665, huber: 2.5376, swd: 3.3390, ept: 193.2308
    Epoch [8/50], Val Losses: mse: 42.3490, mae: 4.0595, huber: 3.6103, swd: 5.3650, ept: 145.8062
    Epoch [8/50], Test Losses: mse: 34.1753, mae: 3.5304, huber: 3.0875, swd: 4.6876, ept: 159.7441
      Epoch 8 composite train-obj: 2.537605
            Val objective improved 3.7179 → 3.6103, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.5750, mae: 2.8307, huber: 2.4051, swd: 2.9292, ept: 199.5049
    Epoch [9/50], Val Losses: mse: 43.3423, mae: 4.0636, huber: 3.6148, swd: 4.7128, ept: 142.7258
    Epoch [9/50], Test Losses: mse: 34.2754, mae: 3.5325, huber: 3.0901, swd: 4.2002, ept: 157.5029
      Epoch 9 composite train-obj: 2.405131
            No improvement (3.6148), counter 1/5
    Epoch [10/50], Train Losses: mse: 23.1723, mae: 2.7155, huber: 2.2931, swd: 2.6391, ept: 204.8701
    Epoch [10/50], Val Losses: mse: 43.0299, mae: 3.9659, huber: 3.5227, swd: 4.2918, ept: 152.5968
    Epoch [10/50], Test Losses: mse: 33.7345, mae: 3.4382, huber: 3.0005, swd: 4.0855, ept: 163.7321
      Epoch 10 composite train-obj: 2.293080
            Val objective improved 3.6103 → 3.5227, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.8182, mae: 2.6192, huber: 2.1992, swd: 2.4134, ept: 208.8881
    Epoch [11/50], Val Losses: mse: 42.9364, mae: 3.9341, huber: 3.4915, swd: 4.2681, ept: 155.8896
    Epoch [11/50], Test Losses: mse: 32.8828, mae: 3.3583, huber: 2.9219, swd: 3.6681, ept: 172.6446
      Epoch 11 composite train-obj: 2.199245
            Val objective improved 3.5227 → 3.4915, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.4093, mae: 2.5006, huber: 2.0846, swd: 2.1588, ept: 215.2779
    Epoch [12/50], Val Losses: mse: 42.4192, mae: 3.9174, huber: 3.4730, swd: 3.6753, ept: 160.4512
    Epoch [12/50], Test Losses: mse: 33.1905, mae: 3.3864, huber: 2.9474, swd: 3.1863, ept: 178.1612
      Epoch 12 composite train-obj: 2.084623
            Val objective improved 3.4915 → 3.4730, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 19.3390, mae: 2.4048, huber: 1.9920, swd: 1.9646, ept: 220.2975
    Epoch [13/50], Val Losses: mse: 42.1775, mae: 3.8266, huber: 3.3879, swd: 3.7646, ept: 167.1015
    Epoch [13/50], Test Losses: mse: 32.7358, mae: 3.2723, huber: 2.8405, swd: 3.2141, ept: 182.1256
      Epoch 13 composite train-obj: 1.992002
            Val objective improved 3.4730 → 3.3879, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.4845, mae: 2.3327, huber: 1.9228, swd: 1.8277, ept: 225.2445
    Epoch [14/50], Val Losses: mse: 43.1609, mae: 3.8680, huber: 3.4257, swd: 3.7352, ept: 167.5029
    Epoch [14/50], Test Losses: mse: 35.3293, mae: 3.3958, huber: 2.9601, swd: 2.9569, ept: 182.8424
      Epoch 14 composite train-obj: 1.922756
            No improvement (3.4257), counter 1/5
    Epoch [15/50], Train Losses: mse: 18.0951, mae: 2.3023, huber: 1.8927, swd: 1.7514, ept: 227.6704
    Epoch [15/50], Val Losses: mse: 46.8134, mae: 4.0774, huber: 3.6347, swd: 4.1417, ept: 166.5905
    Epoch [15/50], Test Losses: mse: 33.0144, mae: 3.3043, huber: 2.8697, swd: 2.8846, ept: 182.3942
      Epoch 15 composite train-obj: 1.892709
            No improvement (3.6347), counter 2/5
    Epoch [16/50], Train Losses: mse: 17.6114, mae: 2.2623, huber: 1.8538, swd: 1.6862, ept: 230.9000
    Epoch [16/50], Val Losses: mse: 45.4683, mae: 3.9365, huber: 3.4978, swd: 3.4117, ept: 174.9262
    Epoch [16/50], Test Losses: mse: 32.7642, mae: 3.2387, huber: 2.8075, swd: 2.5966, ept: 191.7718
      Epoch 16 composite train-obj: 1.853783
            No improvement (3.4978), counter 3/5
    Epoch [17/50], Train Losses: mse: 17.3852, mae: 2.2320, huber: 1.8256, swd: 1.6525, ept: 233.7384
    Epoch [17/50], Val Losses: mse: 44.2717, mae: 3.8816, huber: 3.4433, swd: 3.7389, ept: 177.5613
    Epoch [17/50], Test Losses: mse: 33.2663, mae: 3.2429, huber: 2.8126, swd: 2.8345, ept: 196.6187
      Epoch 17 composite train-obj: 1.825641
            No improvement (3.4433), counter 4/5
    Epoch [18/50], Train Losses: mse: 15.8878, mae: 2.1057, huber: 1.7043, swd: 1.4694, ept: 239.5934
    Epoch [18/50], Val Losses: mse: 39.1948, mae: 3.5892, huber: 3.1551, swd: 3.4022, ept: 187.2240
    Epoch [18/50], Test Losses: mse: 29.5788, mae: 3.0528, huber: 2.6259, swd: 2.9634, ept: 197.4467
      Epoch 18 composite train-obj: 1.704339
            Val objective improved 3.3879 → 3.1551, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 15.5000, mae: 2.0562, huber: 1.6580, swd: 1.4118, ept: 243.4108
    Epoch [19/50], Val Losses: mse: 40.6532, mae: 3.6421, huber: 3.2087, swd: 3.5145, ept: 189.0812
    Epoch [19/50], Test Losses: mse: 31.2774, mae: 3.1473, huber: 2.7187, swd: 2.7549, ept: 200.9936
      Epoch 19 composite train-obj: 1.657984
            No improvement (3.2087), counter 1/5
    Epoch [20/50], Train Losses: mse: 14.8512, mae: 2.0133, huber: 1.6154, swd: 1.3550, ept: 245.8880
    Epoch [20/50], Val Losses: mse: 42.8026, mae: 3.7515, huber: 3.3157, swd: 3.4096, ept: 185.4163
    Epoch [20/50], Test Losses: mse: 32.6970, mae: 3.1909, huber: 2.7621, swd: 2.7349, ept: 197.9174
      Epoch 20 composite train-obj: 1.615448
            No improvement (3.3157), counter 2/5
    Epoch [21/50], Train Losses: mse: 14.0990, mae: 1.9361, huber: 1.5430, swd: 1.2657, ept: 249.2158
    Epoch [21/50], Val Losses: mse: 39.2843, mae: 3.5420, huber: 3.1117, swd: 3.1132, ept: 190.5124
    Epoch [21/50], Test Losses: mse: 30.6144, mae: 3.0343, huber: 2.6125, swd: 2.4108, ept: 204.2803
      Epoch 21 composite train-obj: 1.542979
            Val objective improved 3.1551 → 3.1117, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.8398, mae: 1.9146, huber: 1.5219, swd: 1.2262, ept: 251.4017
    Epoch [22/50], Val Losses: mse: 42.8709, mae: 3.7435, huber: 3.3084, swd: 3.6159, ept: 186.8772
    Epoch [22/50], Test Losses: mse: 31.1964, mae: 3.0796, huber: 2.6536, swd: 2.6445, ept: 206.1564
      Epoch 22 composite train-obj: 1.521926
            No improvement (3.3084), counter 1/5
    Epoch [23/50], Train Losses: mse: 13.5490, mae: 1.8980, huber: 1.5054, swd: 1.2052, ept: 252.4182
    Epoch [23/50], Val Losses: mse: 41.2721, mae: 3.6393, huber: 3.2101, swd: 3.2840, ept: 191.7265
    Epoch [23/50], Test Losses: mse: 29.2799, mae: 2.9628, huber: 2.5413, swd: 2.3499, ept: 209.1105
      Epoch 23 composite train-obj: 1.505367
            No improvement (3.2101), counter 2/5
    Epoch [24/50], Train Losses: mse: 13.4214, mae: 1.8719, huber: 1.4818, swd: 1.1717, ept: 254.2532
    Epoch [24/50], Val Losses: mse: 40.4185, mae: 3.5839, huber: 3.1547, swd: 2.8657, ept: 191.6585
    Epoch [24/50], Test Losses: mse: 29.2771, mae: 2.9838, huber: 2.5616, swd: 2.1716, ept: 206.0960
      Epoch 24 composite train-obj: 1.481837
            No improvement (3.1547), counter 3/5
    Epoch [25/50], Train Losses: mse: 12.5127, mae: 1.7776, huber: 1.3933, swd: 1.0678, ept: 259.6250
    Epoch [25/50], Val Losses: mse: 36.8758, mae: 3.3796, huber: 2.9547, swd: 2.9338, ept: 194.9299
    Epoch [25/50], Test Losses: mse: 31.8576, mae: 3.0500, huber: 2.6305, swd: 2.4377, ept: 208.4773
      Epoch 25 composite train-obj: 1.393337
            Val objective improved 3.1117 → 2.9547, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 12.5131, mae: 1.7891, huber: 1.4030, swd: 1.0756, ept: 259.4795
    Epoch [26/50], Val Losses: mse: 38.6207, mae: 3.4292, huber: 3.0063, swd: 2.9294, ept: 197.6993
    Epoch [26/50], Test Losses: mse: 29.3852, mae: 2.8973, huber: 2.4832, swd: 2.1858, ept: 215.9784
      Epoch 26 composite train-obj: 1.403008
            No improvement (3.0063), counter 1/5
    Epoch [27/50], Train Losses: mse: 12.2881, mae: 1.7562, huber: 1.3722, swd: 1.0582, ept: 262.0459
    Epoch [27/50], Val Losses: mse: 37.9709, mae: 3.4622, huber: 3.0342, swd: 3.2132, ept: 197.0030
    Epoch [27/50], Test Losses: mse: 29.6791, mae: 2.9196, huber: 2.5018, swd: 2.3314, ept: 214.6471
      Epoch 27 composite train-obj: 1.372207
            No improvement (3.0342), counter 2/5
    Epoch [28/50], Train Losses: mse: 12.2274, mae: 1.7534, huber: 1.3697, swd: 1.0317, ept: 262.2212
    Epoch [28/50], Val Losses: mse: 38.1333, mae: 3.4224, huber: 2.9983, swd: 2.8881, ept: 203.6840
    Epoch [28/50], Test Losses: mse: 27.9818, mae: 2.8302, huber: 2.4153, swd: 2.0104, ept: 218.7406
      Epoch 28 composite train-obj: 1.369711
            No improvement (2.9983), counter 3/5
    Epoch [29/50], Train Losses: mse: 11.5940, mae: 1.6881, huber: 1.3083, swd: 0.9662, ept: 266.1605
    Epoch [29/50], Val Losses: mse: 37.6543, mae: 3.3732, huber: 2.9509, swd: 2.8886, ept: 199.0136
    Epoch [29/50], Test Losses: mse: 29.5941, mae: 2.8809, huber: 2.4671, swd: 2.0333, ept: 219.5215
      Epoch 29 composite train-obj: 1.308284
            Val objective improved 2.9547 → 2.9509, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 11.9585, mae: 1.7301, huber: 1.3469, swd: 1.0110, ept: 264.4546
    Epoch [30/50], Val Losses: mse: 38.1056, mae: 3.3944, huber: 2.9705, swd: 2.9713, ept: 204.3169
    Epoch [30/50], Test Losses: mse: 29.4199, mae: 2.8785, huber: 2.4639, swd: 2.0701, ept: 221.3543
      Epoch 30 composite train-obj: 1.346857
            No improvement (2.9705), counter 1/5
    Epoch [31/50], Train Losses: mse: 11.0587, mae: 1.6312, huber: 1.2556, swd: 0.9244, ept: 269.0283
    Epoch [31/50], Val Losses: mse: 39.8416, mae: 3.4601, huber: 3.0362, swd: 2.8759, ept: 202.4754
    Epoch [31/50], Test Losses: mse: 29.1135, mae: 2.8532, huber: 2.4397, swd: 2.0609, ept: 222.4008
      Epoch 31 composite train-obj: 1.255600
            No improvement (3.0362), counter 2/5
    Epoch [32/50], Train Losses: mse: 10.8210, mae: 1.6190, huber: 1.2421, swd: 0.8997, ept: 270.2443
    Epoch [32/50], Val Losses: mse: 36.1834, mae: 3.3128, huber: 2.8893, swd: 2.8286, ept: 205.4935
    Epoch [32/50], Test Losses: mse: 25.5981, mae: 2.6997, huber: 2.2878, swd: 1.8559, ept: 224.3288
      Epoch 32 composite train-obj: 1.242106
            Val objective improved 2.9509 → 2.8893, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 10.7345, mae: 1.6017, huber: 1.2268, swd: 0.8754, ept: 271.1224
    Epoch [33/50], Val Losses: mse: 39.6929, mae: 3.4653, huber: 3.0434, swd: 2.9682, ept: 200.3878
    Epoch [33/50], Test Losses: mse: 29.4868, mae: 2.9117, huber: 2.4974, swd: 2.2524, ept: 214.9741
      Epoch 33 composite train-obj: 1.226833
            No improvement (3.0434), counter 1/5
    Epoch [34/50], Train Losses: mse: 10.5900, mae: 1.5864, huber: 1.2119, swd: 0.8813, ept: 272.7828
    Epoch [34/50], Val Losses: mse: 38.2487, mae: 3.4084, huber: 2.9836, swd: 2.8329, ept: 200.0442
    Epoch [34/50], Test Losses: mse: 29.6573, mae: 2.8986, huber: 2.4826, swd: 2.0075, ept: 221.7481
      Epoch 34 composite train-obj: 1.211882
            No improvement (2.9836), counter 2/5
    Epoch [35/50], Train Losses: mse: 10.7672, mae: 1.6143, huber: 1.2377, swd: 0.8841, ept: 272.1086
    Epoch [35/50], Val Losses: mse: 38.3329, mae: 3.3482, huber: 2.9299, swd: 2.6503, ept: 208.1958
    Epoch [35/50], Test Losses: mse: 29.0877, mae: 2.8428, huber: 2.4324, swd: 1.8229, ept: 226.7108
      Epoch 35 composite train-obj: 1.237654
            No improvement (2.9299), counter 3/5
    Epoch [36/50], Train Losses: mse: 9.7892, mae: 1.5188, huber: 1.1487, swd: 0.7972, ept: 275.4657
    Epoch [36/50], Val Losses: mse: 39.4985, mae: 3.4706, huber: 3.0442, swd: 2.7632, ept: 198.4453
    Epoch [36/50], Test Losses: mse: 27.8776, mae: 2.8190, huber: 2.4040, swd: 1.8124, ept: 221.5552
      Epoch 36 composite train-obj: 1.148682
            No improvement (3.0442), counter 4/5
    Epoch [37/50], Train Losses: mse: 10.0001, mae: 1.5293, huber: 1.1590, swd: 0.8158, ept: 276.0800
    Epoch [37/50], Val Losses: mse: 38.7026, mae: 3.4455, huber: 3.0209, swd: 2.6370, ept: 203.2669
    Epoch [37/50], Test Losses: mse: 28.8829, mae: 2.8900, huber: 2.4747, swd: 1.9610, ept: 217.5616
      Epoch 37 composite train-obj: 1.158972
    Epoch [37/50], Test Losses: mse: 25.5978, mae: 2.6998, huber: 2.2880, swd: 1.8574, ept: 224.3301
    Best round's Test MSE: 25.5981, MAE: 2.6997, SWD: 1.8559
    Best round's Validation MSE: 36.1834, MAE: 3.3128, SWD: 2.8286
    Best round's Test verification MSE : 25.5978, MAE: 2.6998, SWD: 1.8574
    Time taken: 174.40 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1709)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 26.8437 ± 0.9023
      mae: 2.7957 ± 0.0680
      huber: 2.3790 ± 0.0645
      swd: 2.0850 ± 0.1920
      ept: 217.8764 ± 4.7503
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 36.5649 ± 1.1771
      mae: 3.3798 ± 0.0691
      huber: 2.9541 ± 0.0676
      swd: 2.9507 ± 0.1210
      ept: 198.9845 ± 5.1140
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 404.84 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1709
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### AB: Koopman, but do not rotate back
The actual transformation on z-space is no longer normal as UAUT(z).
A general theme is that the more structure it is, the better the performance.
A general K is worse than complex or real-valued fixed parameter. (Not tested here)


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128, 
    # ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    # ablate_rotate_back_Koopman=True, 
    ablate_shift_inside_scale=False,
    householder_reflects_latent = 2,
    householder_reflects_data = 4,
    mixing_strategy='delay_only', 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
    ablate_no_koopman=False, ### HERE
    ablate_no_shift_in_z_push=False, ### HERE
    ablate_rotate_back_Koopman=False,
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.6958, mae: 6.3962, huber: 5.9170, swd: 40.6140, ept: 34.8023
    Epoch [1/50], Val Losses: mse: 59.5880, mae: 5.7859, huber: 5.3100, swd: 23.9736, ept: 43.3834
    Epoch [1/50], Test Losses: mse: 55.5043, mae: 5.4536, huber: 4.9807, swd: 24.9552, ept: 44.0182
      Epoch 1 composite train-obj: 5.917010
            Val objective improved inf → 5.3100, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.8759, mae: 4.8373, huber: 4.3716, swd: 16.7557, ept: 89.5355
    Epoch [2/50], Val Losses: mse: 50.8745, mae: 5.0326, huber: 4.5664, swd: 15.5785, ept: 81.3474
    Epoch [2/50], Test Losses: mse: 47.4693, mae: 4.7441, huber: 4.2815, swd: 16.9234, ept: 74.6335
      Epoch 2 composite train-obj: 4.371566
            Val objective improved 5.3100 → 4.5664, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 42.0157, mae: 4.2672, huber: 3.8121, swd: 11.5411, ept: 123.1545
    Epoch [3/50], Val Losses: mse: 49.6673, mae: 4.8308, huber: 4.3678, swd: 11.6859, ept: 97.5400
    Epoch [3/50], Test Losses: mse: 44.9435, mae: 4.5003, huber: 4.0416, swd: 13.0282, ept: 91.7609
      Epoch 3 composite train-obj: 3.812088
            Val objective improved 4.5664 → 4.3678, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.7110, mae: 3.9058, huber: 3.4567, swd: 8.2980, ept: 141.1565
    Epoch [4/50], Val Losses: mse: 47.6251, mae: 4.6479, huber: 4.1866, swd: 8.4990, ept: 107.4032
    Epoch [4/50], Test Losses: mse: 42.1988, mae: 4.2894, huber: 3.8316, swd: 9.1055, ept: 107.6471
      Epoch 4 composite train-obj: 3.456667
            Val objective improved 4.3678 → 4.1866, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.2833, mae: 3.5511, huber: 3.1074, swd: 5.4950, ept: 161.8404
    Epoch [5/50], Val Losses: mse: 46.0552, mae: 4.4189, huber: 3.9632, swd: 7.2828, ept: 120.5729
    Epoch [5/50], Test Losses: mse: 39.9586, mae: 4.0145, huber: 3.5622, swd: 7.3559, ept: 127.2661
      Epoch 5 composite train-obj: 3.107403
            Val objective improved 4.1866 → 3.9632, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.8490, mae: 3.2679, huber: 2.8305, swd: 4.2563, ept: 180.1243
    Epoch [6/50], Val Losses: mse: 45.6213, mae: 4.3252, huber: 3.8701, swd: 5.1838, ept: 134.9265
    Epoch [6/50], Test Losses: mse: 36.5107, mae: 3.7622, huber: 3.3134, swd: 5.2505, ept: 143.6647
      Epoch 6 composite train-obj: 2.830497
            Val objective improved 3.9632 → 3.8701, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.2516, mae: 3.0666, huber: 2.6337, swd: 3.4885, ept: 191.4707
    Epoch [7/50], Val Losses: mse: 50.3930, mae: 4.4867, huber: 4.0312, swd: 4.7123, ept: 138.2077
    Epoch [7/50], Test Losses: mse: 37.0442, mae: 3.7725, huber: 3.3223, swd: 4.7160, ept: 150.8676
      Epoch 7 composite train-obj: 2.633688
            No improvement (4.0312), counter 1/5
    Epoch [8/50], Train Losses: mse: 25.1227, mae: 2.8916, huber: 2.4633, swd: 2.9572, ept: 200.7506
    Epoch [8/50], Val Losses: mse: 48.0810, mae: 4.2717, huber: 3.8222, swd: 4.7908, ept: 145.6209
    Epoch [8/50], Test Losses: mse: 34.4623, mae: 3.5588, huber: 3.1146, swd: 4.5754, ept: 160.9273
      Epoch 8 composite train-obj: 2.463250
            Val objective improved 3.8701 → 3.8222, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 23.5396, mae: 2.7686, huber: 2.3432, swd: 2.5793, ept: 207.1825
    Epoch [9/50], Val Losses: mse: 46.8203, mae: 4.1793, huber: 3.7320, swd: 4.4994, ept: 159.0016
    Epoch [9/50], Test Losses: mse: 33.7176, mae: 3.4517, huber: 3.0107, swd: 3.6290, ept: 173.7453
      Epoch 9 composite train-obj: 2.343186
            Val objective improved 3.8222 → 3.7320, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 21.9673, mae: 2.6312, huber: 2.2102, swd: 2.2741, ept: 214.7464
    Epoch [10/50], Val Losses: mse: 47.2989, mae: 4.1406, huber: 3.6960, swd: 4.4367, ept: 159.5983
    Epoch [10/50], Test Losses: mse: 32.5242, mae: 3.3521, huber: 2.9152, swd: 3.7043, ept: 179.6794
      Epoch 10 composite train-obj: 2.210235
            Val objective improved 3.7320 → 3.6960, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 20.8709, mae: 2.5473, huber: 2.1287, swd: 2.0581, ept: 218.4515
    Epoch [11/50], Val Losses: mse: 44.1164, mae: 4.0072, huber: 3.5618, swd: 3.9728, ept: 166.9489
    Epoch [11/50], Test Losses: mse: 33.2337, mae: 3.3808, huber: 2.9431, swd: 3.4038, ept: 178.8585
      Epoch 11 composite train-obj: 2.128739
            Val objective improved 3.6960 → 3.5618, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 19.9038, mae: 2.4615, huber: 2.0456, swd: 1.9197, ept: 223.4862
    Epoch [12/50], Val Losses: mse: 43.9512, mae: 3.8877, huber: 3.4481, swd: 3.6183, ept: 169.5787
    Epoch [12/50], Test Losses: mse: 32.0161, mae: 3.2278, huber: 2.7954, swd: 2.9714, ept: 185.8078
      Epoch 12 composite train-obj: 2.045630
            Val objective improved 3.5618 → 3.4481, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.7850, mae: 2.3623, huber: 1.9507, swd: 1.7352, ept: 228.5541
    Epoch [13/50], Val Losses: mse: 42.4748, mae: 3.8102, huber: 3.3717, swd: 3.5986, ept: 175.6250
    Epoch [13/50], Test Losses: mse: 32.4118, mae: 3.2347, huber: 2.8035, swd: 3.0806, ept: 190.7855
      Epoch 13 composite train-obj: 1.950711
            Val objective improved 3.4481 → 3.3717, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.1776, mae: 2.2918, huber: 1.8838, swd: 1.6690, ept: 233.6616
    Epoch [14/50], Val Losses: mse: 41.8629, mae: 3.7635, huber: 3.3265, swd: 3.5933, ept: 177.2038
    Epoch [14/50], Test Losses: mse: 31.7209, mae: 3.1924, huber: 2.7627, swd: 2.9890, ept: 189.5027
      Epoch 14 composite train-obj: 1.883829
            Val objective improved 3.3717 → 3.3265, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 17.0689, mae: 2.2076, huber: 1.8016, swd: 1.5390, ept: 238.1730
    Epoch [15/50], Val Losses: mse: 41.7564, mae: 3.7289, huber: 3.2931, swd: 3.3567, ept: 182.6093
    Epoch [15/50], Test Losses: mse: 32.0400, mae: 3.1376, huber: 2.7103, swd: 2.5535, ept: 202.7162
      Epoch 15 composite train-obj: 1.801597
            Val objective improved 3.3265 → 3.2931, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.6740, mae: 2.1688, huber: 1.7648, swd: 1.4755, ept: 240.7615
    Epoch [16/50], Val Losses: mse: 40.7653, mae: 3.6710, huber: 3.2358, swd: 3.3786, ept: 183.0328
    Epoch [16/50], Test Losses: mse: 29.6778, mae: 3.0350, huber: 2.6091, swd: 2.6611, ept: 201.0733
      Epoch 16 composite train-obj: 1.764824
            Val objective improved 3.2931 → 3.2358, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.9280, mae: 2.1006, huber: 1.6989, swd: 1.3932, ept: 245.4657
    Epoch [17/50], Val Losses: mse: 41.4364, mae: 3.6470, huber: 3.2147, swd: 3.0382, ept: 186.6530
    Epoch [17/50], Test Losses: mse: 30.3484, mae: 3.0132, huber: 2.5907, swd: 2.2871, ept: 205.6554
      Epoch 17 composite train-obj: 1.698873
            Val objective improved 3.2358 → 3.2147, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.1564, mae: 2.0271, huber: 1.6296, swd: 1.2803, ept: 249.1415
    Epoch [18/50], Val Losses: mse: 42.5890, mae: 3.6866, huber: 3.2553, swd: 3.2632, ept: 188.0241
    Epoch [18/50], Test Losses: mse: 30.3022, mae: 3.0032, huber: 2.5812, swd: 2.2714, ept: 209.8784
      Epoch 18 composite train-obj: 1.629570
            No improvement (3.2553), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.6602, mae: 1.9712, huber: 1.5767, swd: 1.2463, ept: 252.7095
    Epoch [19/50], Val Losses: mse: 42.1351, mae: 3.6667, huber: 3.2342, swd: 2.8431, ept: 192.3350
    Epoch [19/50], Test Losses: mse: 33.1246, mae: 3.1508, huber: 2.7250, swd: 2.2069, ept: 207.0004
      Epoch 19 composite train-obj: 1.576742
            No improvement (3.2342), counter 2/5
    Epoch [20/50], Train Losses: mse: 14.1318, mae: 1.9291, huber: 1.5367, swd: 1.1964, ept: 254.9148
    Epoch [20/50], Val Losses: mse: 40.4008, mae: 3.5797, huber: 3.1498, swd: 3.0110, ept: 193.7545
    Epoch [20/50], Test Losses: mse: 28.3533, mae: 2.8751, huber: 2.4570, swd: 2.2140, ept: 216.2685
      Epoch 20 composite train-obj: 1.536667
            Val objective improved 3.2147 → 3.1498, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 13.2603, mae: 1.8390, huber: 1.4519, swd: 1.1060, ept: 259.5789
    Epoch [21/50], Val Losses: mse: 37.9487, mae: 3.4414, huber: 3.0137, swd: 3.1008, ept: 199.7699
    Epoch [21/50], Test Losses: mse: 29.3268, mae: 2.9509, huber: 2.5307, swd: 2.4754, ept: 211.7188
      Epoch 21 composite train-obj: 1.451896
            Val objective improved 3.1498 → 3.0137, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.1607, mae: 1.8336, huber: 1.4460, swd: 1.0939, ept: 260.4274
    Epoch [22/50], Val Losses: mse: 40.7322, mae: 3.5526, huber: 3.1271, swd: 2.9088, ept: 199.0687
    Epoch [22/50], Test Losses: mse: 29.6884, mae: 2.9079, huber: 2.4921, swd: 2.1700, ept: 217.9728
      Epoch 22 composite train-obj: 1.446020
            No improvement (3.1271), counter 1/5
    Epoch [23/50], Train Losses: mse: 13.1624, mae: 1.8420, huber: 1.4534, swd: 1.1211, ept: 260.6262
    Epoch [23/50], Val Losses: mse: 38.9505, mae: 3.4616, huber: 3.0360, swd: 2.9495, ept: 199.5699
    Epoch [23/50], Test Losses: mse: 28.3111, mae: 2.8532, huber: 2.4368, swd: 2.0768, ept: 216.7073
      Epoch 23 composite train-obj: 1.453367
            No improvement (3.0360), counter 2/5
    Epoch [24/50], Train Losses: mse: 12.6251, mae: 1.7778, huber: 1.3935, swd: 1.0398, ept: 263.6671
    Epoch [24/50], Val Losses: mse: 36.9341, mae: 3.3740, huber: 2.9470, swd: 2.6902, ept: 200.0509
    Epoch [24/50], Test Losses: mse: 28.3809, mae: 2.8413, huber: 2.4259, swd: 1.9495, ept: 221.4627
      Epoch 24 composite train-obj: 1.393462
            Val objective improved 3.0137 → 2.9470, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 12.1753, mae: 1.7338, huber: 1.3521, swd: 0.9966, ept: 266.1510
    Epoch [25/50], Val Losses: mse: 36.8408, mae: 3.3710, huber: 2.9428, swd: 2.6705, ept: 200.4196
    Epoch [25/50], Test Losses: mse: 31.9458, mae: 3.0122, huber: 2.5923, swd: 1.9314, ept: 219.1976
      Epoch 25 composite train-obj: 1.352066
            Val objective improved 2.9470 → 2.9428, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 12.0852, mae: 1.7264, huber: 1.3451, swd: 0.9762, ept: 266.5518
    Epoch [26/50], Val Losses: mse: 37.6336, mae: 3.4504, huber: 3.0229, swd: 3.0185, ept: 193.6727
    Epoch [26/50], Test Losses: mse: 29.0350, mae: 2.9181, huber: 2.4990, swd: 2.2937, ept: 213.5193
      Epoch 26 composite train-obj: 1.345105
            No improvement (3.0229), counter 1/5
    Epoch [27/50], Train Losses: mse: 11.9240, mae: 1.7150, huber: 1.3336, swd: 0.9781, ept: 267.7804
    Epoch [27/50], Val Losses: mse: 38.1520, mae: 3.3824, huber: 2.9619, swd: 2.9403, ept: 204.3487
    Epoch [27/50], Test Losses: mse: 29.1977, mae: 2.8547, huber: 2.4418, swd: 2.0921, ept: 223.6316
      Epoch 27 composite train-obj: 1.333636
            No improvement (2.9619), counter 2/5
    Epoch [28/50], Train Losses: mse: 11.5610, mae: 1.6755, huber: 1.2964, swd: 0.9568, ept: 270.4164
    Epoch [28/50], Val Losses: mse: 40.8600, mae: 3.6206, huber: 3.1933, swd: 3.2270, ept: 194.0817
    Epoch [28/50], Test Losses: mse: 31.4564, mae: 3.0387, huber: 2.6195, swd: 2.1905, ept: 211.7341
      Epoch 28 composite train-obj: 1.296438
            No improvement (3.1933), counter 3/5
    Epoch [29/50], Train Losses: mse: 11.9851, mae: 1.7162, huber: 1.3352, swd: 0.9660, ept: 268.4740
    Epoch [29/50], Val Losses: mse: 37.7519, mae: 3.3605, huber: 2.9387, swd: 2.8113, ept: 207.1360
    Epoch [29/50], Test Losses: mse: 29.7229, mae: 2.8544, huber: 2.4410, swd: 2.0403, ept: 225.9657
      Epoch 29 composite train-obj: 1.335209
            Val objective improved 2.9428 → 2.9387, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 11.4287, mae: 1.6510, huber: 1.2752, swd: 0.9142, ept: 271.8138
    Epoch [30/50], Val Losses: mse: 38.2606, mae: 3.4144, huber: 2.9931, swd: 2.6991, ept: 200.3736
    Epoch [30/50], Test Losses: mse: 31.2003, mae: 2.9286, huber: 2.5169, swd: 1.9335, ept: 222.6633
      Epoch 30 composite train-obj: 1.275226
            No improvement (2.9931), counter 1/5
    Epoch [31/50], Train Losses: mse: 11.2075, mae: 1.6533, huber: 1.2741, swd: 0.9220, ept: 272.0506
    Epoch [31/50], Val Losses: mse: 37.2536, mae: 3.3334, huber: 2.9123, swd: 2.9239, ept: 207.3988
    Epoch [31/50], Test Losses: mse: 26.1827, mae: 2.6903, huber: 2.2798, swd: 2.0553, ept: 227.3704
      Epoch 31 composite train-obj: 1.274149
            Val objective improved 2.9387 → 2.9123, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 10.3789, mae: 1.5561, huber: 1.1858, swd: 0.8289, ept: 275.8628
    Epoch [32/50], Val Losses: mse: 39.0098, mae: 3.4420, huber: 3.0201, swd: 2.6739, ept: 199.7260
    Epoch [32/50], Test Losses: mse: 28.1522, mae: 2.8001, huber: 2.3886, swd: 1.8074, ept: 219.5427
      Epoch 32 composite train-obj: 1.185828
            No improvement (3.0201), counter 1/5
    Epoch [33/50], Train Losses: mse: 10.6411, mae: 1.5831, huber: 1.2103, swd: 0.8647, ept: 275.1324
    Epoch [33/50], Val Losses: mse: 38.1520, mae: 3.3795, huber: 2.9609, swd: 2.5998, ept: 203.9715
    Epoch [33/50], Test Losses: mse: 28.9100, mae: 2.7992, huber: 2.3915, swd: 1.9397, ept: 223.7862
      Epoch 33 composite train-obj: 1.210325
            No improvement (2.9609), counter 2/5
    Epoch [34/50], Train Losses: mse: 10.4938, mae: 1.5680, huber: 1.1961, swd: 0.8363, ept: 276.1502
    Epoch [34/50], Val Losses: mse: 36.8197, mae: 3.3482, huber: 2.9257, swd: 2.6848, ept: 204.7392
    Epoch [34/50], Test Losses: mse: 27.9685, mae: 2.7735, huber: 2.3630, swd: 1.9706, ept: 225.7693
      Epoch 34 composite train-obj: 1.196144
            No improvement (2.9257), counter 3/5
    Epoch [35/50], Train Losses: mse: 10.0259, mae: 1.5257, huber: 1.1567, swd: 0.8034, ept: 277.5007
    Epoch [35/50], Val Losses: mse: 36.5181, mae: 3.3200, huber: 2.9003, swd: 2.7390, ept: 201.9851
    Epoch [35/50], Test Losses: mse: 28.0045, mae: 2.7946, huber: 2.3843, swd: 1.9788, ept: 221.5131
      Epoch 35 composite train-obj: 1.156674
            Val objective improved 2.9123 → 2.9003, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 9.6759, mae: 1.4876, huber: 1.1213, swd: 0.7699, ept: 279.5143
    Epoch [36/50], Val Losses: mse: 37.0667, mae: 3.3304, huber: 2.9103, swd: 2.6501, ept: 207.8129
    Epoch [36/50], Test Losses: mse: 26.0757, mae: 2.6830, huber: 2.2756, swd: 1.7594, ept: 227.2645
      Epoch 36 composite train-obj: 1.121275
            No improvement (2.9103), counter 1/5
    Epoch [37/50], Train Losses: mse: 9.5657, mae: 1.4733, huber: 1.1086, swd: 0.7472, ept: 280.0399
    Epoch [37/50], Val Losses: mse: 36.6178, mae: 3.3312, huber: 2.9113, swd: 2.7058, ept: 201.3404
    Epoch [37/50], Test Losses: mse: 26.8867, mae: 2.7444, huber: 2.3347, swd: 1.7748, ept: 223.4320
      Epoch 37 composite train-obj: 1.108590
            No improvement (2.9113), counter 2/5
    Epoch [38/50], Train Losses: mse: 9.8377, mae: 1.5022, huber: 1.1349, swd: 0.7888, ept: 279.5890
    Epoch [38/50], Val Losses: mse: 36.5795, mae: 3.2821, huber: 2.8656, swd: 2.3600, ept: 208.8653
    Epoch [38/50], Test Losses: mse: 26.8585, mae: 2.6908, huber: 2.2853, swd: 1.6682, ept: 229.5125
      Epoch 38 composite train-obj: 1.134941
            Val objective improved 2.9003 → 2.8656, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 9.2598, mae: 1.4478, huber: 1.0840, swd: 0.7355, ept: 281.5343
    Epoch [39/50], Val Losses: mse: 35.2031, mae: 3.2233, huber: 2.8065, swd: 2.6521, ept: 208.7957
    Epoch [39/50], Test Losses: mse: 27.8565, mae: 2.7343, huber: 2.3289, swd: 1.8430, ept: 229.0089
      Epoch 39 composite train-obj: 1.083957
            Val objective improved 2.8656 → 2.8065, saving checkpoint.
    Epoch [40/50], Train Losses: mse: 9.3455, mae: 1.4564, huber: 1.0920, swd: 0.7585, ept: 281.4877
    Epoch [40/50], Val Losses: mse: 37.5665, mae: 3.3320, huber: 2.9125, swd: 2.2622, ept: 204.0277
    Epoch [40/50], Test Losses: mse: 30.2575, mae: 2.8733, huber: 2.4634, swd: 1.5926, ept: 224.6783
      Epoch 40 composite train-obj: 1.092043
            No improvement (2.9125), counter 1/5
    Epoch [41/50], Train Losses: mse: 9.6156, mae: 1.4827, huber: 1.1164, swd: 0.7529, ept: 280.8122
    Epoch [41/50], Val Losses: mse: 35.6080, mae: 3.2400, huber: 2.8207, swd: 2.6845, ept: 205.2489
    Epoch [41/50], Test Losses: mse: 28.0982, mae: 2.7891, huber: 2.3781, swd: 1.9010, ept: 224.2143
      Epoch 41 composite train-obj: 1.116397
            No improvement (2.8207), counter 2/5
    Epoch [42/50], Train Losses: mse: 9.1704, mae: 1.4366, huber: 1.0732, swd: 0.7245, ept: 283.3339
    Epoch [42/50], Val Losses: mse: 37.6536, mae: 3.3122, huber: 2.8971, swd: 2.5842, ept: 208.2755
    Epoch [42/50], Test Losses: mse: 26.6777, mae: 2.6499, huber: 2.2477, swd: 1.9291, ept: 234.3617
      Epoch 42 composite train-obj: 1.073213
            No improvement (2.8971), counter 3/5
    Epoch [43/50], Train Losses: mse: 8.9181, mae: 1.4016, huber: 1.0416, swd: 0.6910, ept: 285.1201
    Epoch [43/50], Val Losses: mse: 34.4544, mae: 3.1723, huber: 2.7570, swd: 2.3989, ept: 207.6851
    Epoch [43/50], Test Losses: mse: 25.6616, mae: 2.6062, huber: 2.2058, swd: 1.7583, ept: 234.0410
      Epoch 43 composite train-obj: 1.041634
            Val objective improved 2.8065 → 2.7570, saving checkpoint.
    Epoch [44/50], Train Losses: mse: 8.3277, mae: 1.3594, huber: 1.0011, swd: 0.6505, ept: 285.9977
    Epoch [44/50], Val Losses: mse: 33.4903, mae: 3.1266, huber: 2.7116, swd: 2.2960, ept: 212.1389
    Epoch [44/50], Test Losses: mse: 25.5590, mae: 2.5948, huber: 2.1953, swd: 1.7023, ept: 237.6405
      Epoch 44 composite train-obj: 1.001097
            Val objective improved 2.7570 → 2.7116, saving checkpoint.
    Epoch [45/50], Train Losses: mse: 8.4202, mae: 1.3614, huber: 1.0033, swd: 0.6473, ept: 286.6477
    Epoch [45/50], Val Losses: mse: 36.4735, mae: 3.2916, huber: 2.8730, swd: 2.4349, ept: 208.2032
    Epoch [45/50], Test Losses: mse: 26.9520, mae: 2.6833, huber: 2.2789, swd: 1.6278, ept: 233.6577
      Epoch 45 composite train-obj: 1.003337
            No improvement (2.8730), counter 1/5
    Epoch [46/50], Train Losses: mse: 8.4326, mae: 1.3700, huber: 1.0104, swd: 0.6665, ept: 286.1067
    Epoch [46/50], Val Losses: mse: 35.2522, mae: 3.2270, huber: 2.8101, swd: 2.4981, ept: 210.8449
    Epoch [46/50], Test Losses: mse: 27.2231, mae: 2.7039, huber: 2.3001, swd: 1.7644, ept: 229.7999
      Epoch 46 composite train-obj: 1.010436
            No improvement (2.8101), counter 2/5
    Epoch [47/50], Train Losses: mse: 8.0514, mae: 1.3162, huber: 0.9629, swd: 0.6251, ept: 288.5775
    Epoch [47/50], Val Losses: mse: 35.6771, mae: 3.2395, huber: 2.8262, swd: 2.6918, ept: 213.0951
    Epoch [47/50], Test Losses: mse: 27.4686, mae: 2.6699, huber: 2.2702, swd: 1.7339, ept: 235.5103
      Epoch 47 composite train-obj: 0.962875
            No improvement (2.8262), counter 3/5
    Epoch [48/50], Train Losses: mse: 8.4232, mae: 1.3471, huber: 0.9921, swd: 0.6657, ept: 288.1082
    Epoch [48/50], Val Losses: mse: 36.5057, mae: 3.3097, huber: 2.8920, swd: 2.4596, ept: 201.3722
    Epoch [48/50], Test Losses: mse: 26.3388, mae: 2.6736, huber: 2.2709, swd: 1.6948, ept: 227.6708
      Epoch 48 composite train-obj: 0.992060
            No improvement (2.8920), counter 4/5
    Epoch [49/50], Train Losses: mse: 8.5810, mae: 1.3702, huber: 1.0123, swd: 0.6590, ept: 286.8581
    Epoch [49/50], Val Losses: mse: 32.6899, mae: 3.0833, huber: 2.6697, swd: 2.6409, ept: 213.2152
    Epoch [49/50], Test Losses: mse: 28.2606, mae: 2.6972, huber: 2.2964, swd: 1.6893, ept: 235.5884
      Epoch 49 composite train-obj: 1.012280
            Val objective improved 2.7116 → 2.6697, saving checkpoint.
    Epoch [50/50], Train Losses: mse: 8.1015, mae: 1.3274, huber: 0.9718, swd: 0.6243, ept: 289.3380
    Epoch [50/50], Val Losses: mse: 33.5260, mae: 3.0843, huber: 2.6743, swd: 2.5216, ept: 213.9392
    Epoch [50/50], Test Losses: mse: 27.0323, mae: 2.6271, huber: 2.2303, swd: 1.7106, ept: 237.0731
      Epoch 50 composite train-obj: 0.971844
            No improvement (2.6743), counter 1/5
    Epoch [50/50], Test Losses: mse: 28.2586, mae: 2.6971, huber: 2.2963, swd: 1.6894, ept: 235.6068
    Best round's Test MSE: 28.2606, MAE: 2.6972, SWD: 1.6893
    Best round's Validation MSE: 32.6899, MAE: 3.0833, SWD: 2.6409
    Best round's Test verification MSE : 28.2586, MAE: 2.6971, SWD: 1.6894
    Time taken: 196.91 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 73.0989, mae: 6.3741, huber: 5.8953, swd: 41.5272, ept: 35.6458
    Epoch [1/50], Val Losses: mse: 58.0510, mae: 5.6893, huber: 5.2140, swd: 24.5062, ept: 44.2537
    Epoch [1/50], Test Losses: mse: 53.1977, mae: 5.3007, huber: 4.8289, swd: 25.8022, ept: 44.7237
      Epoch 1 composite train-obj: 5.895314
            Val objective improved inf → 5.2140, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.1669, mae: 4.7806, huber: 4.3158, swd: 16.2375, ept: 90.8503
    Epoch [2/50], Val Losses: mse: 50.6252, mae: 4.9956, huber: 4.5298, swd: 15.0351, ept: 82.6046
    Epoch [2/50], Test Losses: mse: 47.0245, mae: 4.7034, huber: 4.2415, swd: 16.1053, ept: 74.3492
      Epoch 2 composite train-obj: 4.315830
            Val objective improved 5.2140 → 4.5298, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.9589, mae: 4.2088, huber: 3.7536, swd: 10.5142, ept: 123.7460
    Epoch [3/50], Val Losses: mse: 47.6369, mae: 4.7045, huber: 4.2431, swd: 11.5467, ept: 101.9867
    Epoch [3/50], Test Losses: mse: 44.3041, mae: 4.4540, huber: 3.9955, swd: 12.3792, ept: 97.1042
      Epoch 3 composite train-obj: 3.753646
            Val objective improved 4.5298 → 4.2431, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.2575, mae: 3.7954, huber: 3.3473, swd: 7.0383, ept: 147.5519
    Epoch [4/50], Val Losses: mse: 46.2159, mae: 4.4898, huber: 4.0329, swd: 7.4923, ept: 114.3207
    Epoch [4/50], Test Losses: mse: 40.3697, mae: 4.0997, huber: 3.6462, swd: 7.9497, ept: 118.0214
      Epoch 4 composite train-obj: 3.347331
            Val objective improved 4.2431 → 4.0329, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.6152, mae: 3.4969, huber: 3.0541, swd: 5.1726, ept: 166.4852
    Epoch [5/50], Val Losses: mse: 47.1218, mae: 4.4248, huber: 3.9700, swd: 6.5779, ept: 127.5564
    Epoch [5/50], Test Losses: mse: 38.7274, mae: 3.9438, huber: 3.4922, swd: 6.4269, ept: 133.5660
      Epoch 5 composite train-obj: 3.054065
            Val objective improved 4.0329 → 3.9700, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.5755, mae: 3.2598, huber: 2.8218, swd: 4.0732, ept: 180.3691
    Epoch [6/50], Val Losses: mse: 45.9534, mae: 4.3149, huber: 3.8613, swd: 5.7172, ept: 135.9636
    Epoch [6/50], Test Losses: mse: 36.3350, mae: 3.7584, huber: 3.3091, swd: 5.4568, ept: 149.1466
      Epoch 6 composite train-obj: 2.821807
            Val objective improved 3.9700 → 3.8613, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 26.9266, mae: 3.0496, huber: 2.6165, swd: 3.3327, ept: 189.8619
    Epoch [7/50], Val Losses: mse: 45.0236, mae: 4.1663, huber: 3.7175, swd: 4.7674, ept: 149.6353
    Epoch [7/50], Test Losses: mse: 33.3377, mae: 3.5364, huber: 3.0929, swd: 4.9330, ept: 158.5573
      Epoch 7 composite train-obj: 2.616534
            Val objective improved 3.8613 → 3.7175, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 25.0109, mae: 2.8959, huber: 2.4669, swd: 2.9004, ept: 198.3647
    Epoch [8/50], Val Losses: mse: 45.5780, mae: 4.1298, huber: 3.6823, swd: 4.1079, ept: 154.1498
    Epoch [8/50], Test Losses: mse: 33.6201, mae: 3.4852, huber: 3.0443, swd: 3.8764, ept: 162.7740
      Epoch 8 composite train-obj: 2.466907
            Val objective improved 3.7175 → 3.6823, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 23.4208, mae: 2.7699, huber: 2.3438, swd: 2.5591, ept: 205.4354
    Epoch [9/50], Val Losses: mse: 45.5993, mae: 4.1007, huber: 3.6554, swd: 4.1473, ept: 158.7788
    Epoch [9/50], Test Losses: mse: 34.6341, mae: 3.4971, huber: 3.0579, swd: 3.8034, ept: 168.9101
      Epoch 9 composite train-obj: 2.343800
            Val objective improved 3.6823 → 3.6554, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 21.6846, mae: 2.6291, huber: 2.2070, swd: 2.2693, ept: 213.7705
    Epoch [10/50], Val Losses: mse: 44.6596, mae: 4.0082, huber: 3.5646, swd: 3.3186, ept: 168.9163
    Epoch [10/50], Test Losses: mse: 33.3504, mae: 3.3736, huber: 2.9365, swd: 3.0290, ept: 181.1967
      Epoch 10 composite train-obj: 2.206965
            Val objective improved 3.6554 → 3.5646, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 20.7252, mae: 2.5547, huber: 2.1346, swd: 2.0840, ept: 217.8688
    Epoch [11/50], Val Losses: mse: 47.1663, mae: 4.0719, huber: 3.6303, swd: 3.4842, ept: 168.2256
    Epoch [11/50], Test Losses: mse: 33.1711, mae: 3.3309, huber: 2.8970, swd: 3.0055, ept: 184.4075
      Epoch 11 composite train-obj: 2.134610
            No improvement (3.6303), counter 1/5
    Epoch [12/50], Train Losses: mse: 19.6269, mae: 2.4557, huber: 2.0390, swd: 1.8872, ept: 224.1873
    Epoch [12/50], Val Losses: mse: 45.5296, mae: 3.9873, huber: 3.5457, swd: 3.4406, ept: 172.0210
    Epoch [12/50], Test Losses: mse: 32.8175, mae: 3.3011, huber: 2.8669, swd: 2.9180, ept: 186.7281
      Epoch 12 composite train-obj: 2.038992
            Val objective improved 3.5646 → 3.5457, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.7049, mae: 2.3610, huber: 1.9483, swd: 1.7977, ept: 230.3965
    Epoch [13/50], Val Losses: mse: 41.9524, mae: 3.8236, huber: 3.3828, swd: 3.4615, ept: 176.8192
    Epoch [13/50], Test Losses: mse: 30.7019, mae: 3.1457, huber: 2.7146, swd: 2.6957, ept: 194.8635
      Epoch 13 composite train-obj: 1.948334
            Val objective improved 3.5457 → 3.3828, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 17.5956, mae: 2.2736, huber: 1.8638, swd: 1.6095, ept: 234.7195
    Epoch [14/50], Val Losses: mse: 42.1184, mae: 3.8267, huber: 3.3879, swd: 3.3107, ept: 177.1910
    Epoch [14/50], Test Losses: mse: 32.8650, mae: 3.2324, huber: 2.8028, swd: 2.6885, ept: 195.5787
      Epoch 14 composite train-obj: 1.863850
            No improvement (3.3879), counter 1/5
    Epoch [15/50], Train Losses: mse: 16.8520, mae: 2.2076, huber: 1.8005, swd: 1.5220, ept: 239.3641
    Epoch [15/50], Val Losses: mse: 39.9145, mae: 3.6545, huber: 3.2204, swd: 3.2336, ept: 186.7531
    Epoch [15/50], Test Losses: mse: 31.8539, mae: 3.1623, huber: 2.7343, swd: 2.5141, ept: 201.8012
      Epoch 15 composite train-obj: 1.800507
            Val objective improved 3.3828 → 3.2204, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.3625, mae: 2.1530, huber: 1.7488, swd: 1.4322, ept: 242.6910
    Epoch [16/50], Val Losses: mse: 41.9228, mae: 3.7424, huber: 3.3036, swd: 3.2141, ept: 181.2394
    Epoch [16/50], Test Losses: mse: 31.4437, mae: 3.1625, huber: 2.7292, swd: 2.4368, ept: 195.5682
      Epoch 16 composite train-obj: 1.748752
            No improvement (3.3036), counter 1/5
    Epoch [17/50], Train Losses: mse: 15.6613, mae: 2.0881, huber: 1.6863, swd: 1.3544, ept: 246.6242
    Epoch [17/50], Val Losses: mse: 40.5220, mae: 3.6407, huber: 3.2076, swd: 2.9295, ept: 185.8016
    Epoch [17/50], Test Losses: mse: 30.0192, mae: 3.0335, huber: 2.6096, swd: 2.5177, ept: 206.3812
      Epoch 17 composite train-obj: 1.686343
            Val objective improved 3.2204 → 3.2076, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.7118, mae: 2.0949, huber: 1.6924, swd: 1.3665, ept: 247.2857
    Epoch [18/50], Val Losses: mse: 40.3494, mae: 3.6502, huber: 3.2151, swd: 3.0725, ept: 188.3415
    Epoch [18/50], Test Losses: mse: 29.6850, mae: 3.0528, huber: 2.6248, swd: 2.3885, ept: 199.8032
      Epoch 18 composite train-obj: 1.692375
            No improvement (3.2151), counter 1/5
    Epoch [19/50], Train Losses: mse: 15.8061, mae: 2.0953, huber: 1.6933, swd: 1.3503, ept: 247.4669
    Epoch [19/50], Val Losses: mse: 39.4981, mae: 3.5682, huber: 3.1375, swd: 3.4135, ept: 191.2542
    Epoch [19/50], Test Losses: mse: 31.2189, mae: 3.0893, huber: 2.6638, swd: 2.8042, ept: 207.5517
      Epoch 19 composite train-obj: 1.693252
            Val objective improved 3.2076 → 3.1375, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 14.4935, mae: 1.9873, huber: 1.5894, swd: 1.2186, ept: 253.3073
    Epoch [20/50], Val Losses: mse: 39.1616, mae: 3.6173, huber: 3.1836, swd: 3.2596, ept: 183.8533
    Epoch [20/50], Test Losses: mse: 29.2981, mae: 3.0221, huber: 2.5970, swd: 2.5593, ept: 201.1853
      Epoch 20 composite train-obj: 1.589433
            No improvement (3.1836), counter 1/5
    Epoch [21/50], Train Losses: mse: 14.2561, mae: 1.9622, huber: 1.5659, swd: 1.1855, ept: 254.9840
    Epoch [21/50], Val Losses: mse: 38.2432, mae: 3.4291, huber: 3.0002, swd: 2.6752, ept: 201.7075
    Epoch [21/50], Test Losses: mse: 27.5964, mae: 2.8439, huber: 2.4243, swd: 2.1577, ept: 219.1898
      Epoch 21 composite train-obj: 1.565867
            Val objective improved 3.1375 → 3.0002, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.3246, mae: 1.8625, huber: 1.4719, swd: 1.1016, ept: 260.5354
    Epoch [22/50], Val Losses: mse: 41.1160, mae: 3.6024, huber: 3.1716, swd: 2.5889, ept: 195.7695
    Epoch [22/50], Test Losses: mse: 28.4927, mae: 2.9203, huber: 2.4991, swd: 2.3120, ept: 212.6738
      Epoch 22 composite train-obj: 1.471913
            No improvement (3.1716), counter 1/5
    Epoch [23/50], Train Losses: mse: 12.9741, mae: 1.8335, huber: 1.4433, swd: 1.0756, ept: 263.0781
    Epoch [23/50], Val Losses: mse: 38.9559, mae: 3.4575, huber: 3.0307, swd: 2.7310, ept: 201.4819
    Epoch [23/50], Test Losses: mse: 28.4657, mae: 2.8758, huber: 2.4563, swd: 2.0782, ept: 219.8167
      Epoch 23 composite train-obj: 1.443330
            No improvement (3.0307), counter 2/5
    Epoch [24/50], Train Losses: mse: 12.6424, mae: 1.8082, huber: 1.4190, swd: 1.0357, ept: 263.9323
    Epoch [24/50], Val Losses: mse: 37.6723, mae: 3.3992, huber: 2.9756, swd: 2.5292, ept: 199.9533
    Epoch [24/50], Test Losses: mse: 26.7447, mae: 2.7799, huber: 2.3649, swd: 2.0924, ept: 222.2827
      Epoch 24 composite train-obj: 1.418967
            Val objective improved 3.0002 → 2.9756, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 12.5982, mae: 1.7882, huber: 1.4011, swd: 1.0270, ept: 265.7571
    Epoch [25/50], Val Losses: mse: 38.6730, mae: 3.5255, huber: 3.0959, swd: 3.0668, ept: 194.7418
    Epoch [25/50], Test Losses: mse: 28.5862, mae: 2.9285, huber: 2.5063, swd: 2.3405, ept: 213.1067
      Epoch 25 composite train-obj: 1.401118
            No improvement (3.0959), counter 1/5
    Epoch [26/50], Train Losses: mse: 12.4912, mae: 1.7859, huber: 1.3991, swd: 1.0166, ept: 265.4324
    Epoch [26/50], Val Losses: mse: 38.7410, mae: 3.4998, huber: 3.0718, swd: 2.8585, ept: 199.6680
    Epoch [26/50], Test Losses: mse: 28.0409, mae: 2.8608, huber: 2.4429, swd: 2.1396, ept: 218.5957
      Epoch 26 composite train-obj: 1.399080
            No improvement (3.0718), counter 2/5
    Epoch [27/50], Train Losses: mse: 11.9965, mae: 1.7238, huber: 1.3406, swd: 0.9580, ept: 269.2074
    Epoch [27/50], Val Losses: mse: 39.2263, mae: 3.4988, huber: 3.0684, swd: 3.0964, ept: 200.8032
    Epoch [27/50], Test Losses: mse: 30.0454, mae: 2.9679, huber: 2.5440, swd: 2.6265, ept: 221.8672
      Epoch 27 composite train-obj: 1.340591
            No improvement (3.0684), counter 3/5
    Epoch [28/50], Train Losses: mse: 12.0383, mae: 1.7329, huber: 1.3482, swd: 0.9738, ept: 269.3889
    Epoch [28/50], Val Losses: mse: 40.0586, mae: 3.5870, huber: 3.1552, swd: 3.0486, ept: 196.6227
    Epoch [28/50], Test Losses: mse: 29.6991, mae: 2.9601, huber: 2.5389, swd: 2.2626, ept: 216.3673
      Epoch 28 composite train-obj: 1.348240
            No improvement (3.1552), counter 4/5
    Epoch [29/50], Train Losses: mse: 11.5243, mae: 1.6890, huber: 1.3075, swd: 0.9029, ept: 270.4413
    Epoch [29/50], Val Losses: mse: 40.0360, mae: 3.5179, huber: 3.0903, swd: 2.6156, ept: 201.8542
    Epoch [29/50], Test Losses: mse: 28.6726, mae: 2.8495, huber: 2.4327, swd: 1.9185, ept: 221.8542
      Epoch 29 composite train-obj: 1.307461
    Epoch [29/50], Test Losses: mse: 26.7475, mae: 2.7803, huber: 2.3653, swd: 2.0920, ept: 222.2821
    Best round's Test MSE: 26.7447, MAE: 2.7799, SWD: 2.0924
    Best round's Validation MSE: 37.6723, MAE: 3.3992, SWD: 2.5292
    Best round's Test verification MSE : 26.7475, MAE: 2.7803, SWD: 2.0920
    Time taken: 114.18 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 74.3501, mae: 6.4290, huber: 5.9501, swd: 42.8356, ept: 36.4574
    Epoch [1/50], Val Losses: mse: 59.0182, mae: 5.6973, huber: 5.2225, swd: 23.0536, ept: 44.4701
    Epoch [1/50], Test Losses: mse: 53.8327, mae: 5.3090, huber: 4.8382, swd: 25.2841, ept: 43.8123
      Epoch 1 composite train-obj: 5.950055
            Val objective improved inf → 5.2225, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.4205, mae: 4.7897, huber: 4.3249, swd: 16.9677, ept: 91.8635
    Epoch [2/50], Val Losses: mse: 50.4432, mae: 4.9997, huber: 4.5334, swd: 15.7382, ept: 81.8143
    Epoch [2/50], Test Losses: mse: 46.4648, mae: 4.6987, huber: 4.2349, swd: 17.4754, ept: 74.5127
      Epoch 2 composite train-obj: 4.324936
            Val objective improved 5.2225 → 4.5334, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 41.3028, mae: 4.2298, huber: 3.7746, swd: 11.7195, ept: 125.0598
    Epoch [3/50], Val Losses: mse: 50.1244, mae: 4.8589, huber: 4.3948, swd: 11.0648, ept: 99.5752
    Epoch [3/50], Test Losses: mse: 44.3072, mae: 4.4429, huber: 3.9835, swd: 12.1846, ept: 98.6550
      Epoch 3 composite train-obj: 3.774599
            Val objective improved 4.5334 → 4.3948, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.1992, mae: 3.8708, huber: 3.4220, swd: 8.4524, ept: 145.2700
    Epoch [4/50], Val Losses: mse: 48.4617, mae: 4.6457, huber: 4.1857, swd: 9.0862, ept: 110.7003
    Epoch [4/50], Test Losses: mse: 41.2318, mae: 4.1820, huber: 3.7257, swd: 9.0030, ept: 113.3057
      Epoch 4 composite train-obj: 3.422005
            Val objective improved 4.3948 → 4.1857, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.3979, mae: 3.5584, huber: 3.1158, swd: 6.2564, ept: 161.7636
    Epoch [5/50], Val Losses: mse: 46.3949, mae: 4.4227, huber: 3.9672, swd: 7.6989, ept: 124.2241
    Epoch [5/50], Test Losses: mse: 39.5926, mae: 4.0157, huber: 3.5638, swd: 7.3987, ept: 127.9976
      Epoch 5 composite train-obj: 3.115818
            Val objective improved 4.1857 → 3.9672, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.5762, mae: 3.3369, huber: 2.8978, swd: 4.8924, ept: 176.2894
    Epoch [6/50], Val Losses: mse: 44.1295, mae: 4.2404, huber: 3.7881, swd: 6.6808, ept: 137.2037
    Epoch [6/50], Test Losses: mse: 36.1259, mae: 3.7591, huber: 3.3110, swd: 6.3299, ept: 146.8229
      Epoch 6 composite train-obj: 2.897791
            Val objective improved 3.9672 → 3.7881, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.8456, mae: 3.1169, huber: 2.6830, swd: 3.9763, ept: 188.2564
    Epoch [7/50], Val Losses: mse: 45.5152, mae: 4.2968, huber: 3.8438, swd: 6.7901, ept: 139.8009
    Epoch [7/50], Test Losses: mse: 35.8011, mae: 3.7258, huber: 3.2776, swd: 5.9413, ept: 145.5575
      Epoch 7 composite train-obj: 2.683048
            No improvement (3.8438), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.1387, mae: 2.9854, huber: 2.5547, swd: 3.4513, ept: 194.3746
    Epoch [8/50], Val Losses: mse: 44.7193, mae: 4.1401, huber: 3.6937, swd: 6.0563, ept: 153.3092
    Epoch [8/50], Test Losses: mse: 35.0894, mae: 3.5764, huber: 3.1352, swd: 5.0985, ept: 159.6921
      Epoch 8 composite train-obj: 2.554704
            Val objective improved 3.7881 → 3.6937, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.2770, mae: 2.8385, huber: 2.4113, swd: 3.0311, ept: 202.9849
    Epoch [9/50], Val Losses: mse: 46.8976, mae: 4.2581, huber: 3.8080, swd: 5.0387, ept: 151.4705
    Epoch [9/50], Test Losses: mse: 35.7322, mae: 3.6149, huber: 3.1710, swd: 4.0460, ept: 164.2036
      Epoch 9 composite train-obj: 2.411321
            No improvement (3.8080), counter 1/5
    Epoch [10/50], Train Losses: mse: 22.7172, mae: 2.7113, huber: 2.2875, swd: 2.6833, ept: 210.8951
    Epoch [10/50], Val Losses: mse: 42.0910, mae: 3.9378, huber: 3.4932, swd: 4.7291, ept: 162.4958
    Epoch [10/50], Test Losses: mse: 32.3384, mae: 3.3777, huber: 2.9382, swd: 3.7296, ept: 178.9252
      Epoch 10 composite train-obj: 2.287506
            Val objective improved 3.6937 → 3.4932, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.2771, mae: 2.5809, huber: 2.1618, swd: 2.3853, ept: 216.9094
    Epoch [11/50], Val Losses: mse: 42.9518, mae: 3.9212, huber: 3.4810, swd: 4.8037, ept: 172.2585
    Epoch [11/50], Test Losses: mse: 33.0559, mae: 3.3655, huber: 2.9314, swd: 3.9121, ept: 180.5104
      Epoch 11 composite train-obj: 2.161776
            Val objective improved 3.4932 → 3.4810, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.0516, mae: 2.4790, huber: 2.0630, swd: 2.1827, ept: 222.7195
    Epoch [12/50], Val Losses: mse: 42.6725, mae: 3.8730, huber: 3.4345, swd: 4.4508, ept: 174.6642
    Epoch [12/50], Test Losses: mse: 33.5754, mae: 3.3450, huber: 2.9119, swd: 3.3703, ept: 185.6559
      Epoch 12 composite train-obj: 2.062966
            Val objective improved 3.4810 → 3.4345, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.8486, mae: 2.3669, huber: 1.9553, swd: 1.9781, ept: 228.8239
    Epoch [13/50], Val Losses: mse: 40.8741, mae: 3.7672, huber: 3.3294, swd: 4.4547, ept: 180.1535
    Epoch [13/50], Test Losses: mse: 33.0296, mae: 3.3075, huber: 2.8750, swd: 3.3892, ept: 188.1180
      Epoch 13 composite train-obj: 1.955279
            Val objective improved 3.4345 → 3.3294, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.6846, mae: 2.3676, huber: 1.9551, swd: 1.9090, ept: 229.9622
    Epoch [14/50], Val Losses: mse: 43.1378, mae: 3.8578, huber: 3.4173, swd: 4.1627, ept: 179.6994
    Epoch [14/50], Test Losses: mse: 33.4210, mae: 3.3338, huber: 2.8975, swd: 2.9231, ept: 188.9595
      Epoch 14 composite train-obj: 1.955142
            No improvement (3.4173), counter 1/5
    Epoch [15/50], Train Losses: mse: 18.1209, mae: 2.3023, huber: 1.8934, swd: 1.8164, ept: 234.1150
    Epoch [15/50], Val Losses: mse: 46.0368, mae: 4.0579, huber: 3.6163, swd: 4.5445, ept: 172.3732
    Epoch [15/50], Test Losses: mse: 31.0393, mae: 3.1870, huber: 2.7543, swd: 2.7709, ept: 191.3950
      Epoch 15 composite train-obj: 1.893394
            No improvement (3.6163), counter 2/5
    Epoch [16/50], Train Losses: mse: 18.1002, mae: 2.3075, huber: 1.8976, swd: 1.8161, ept: 234.4532
    Epoch [16/50], Val Losses: mse: 42.1631, mae: 3.8796, huber: 3.4393, swd: 4.7880, ept: 173.0492
    Epoch [16/50], Test Losses: mse: 32.0276, mae: 3.2677, huber: 2.8341, swd: 3.2601, ept: 188.9894
      Epoch 16 composite train-obj: 1.897638
            No improvement (3.4393), counter 3/5
    Epoch [17/50], Train Losses: mse: 16.0567, mae: 2.1155, huber: 1.7145, swd: 1.5797, ept: 244.1594
    Epoch [17/50], Val Losses: mse: 40.8993, mae: 3.7255, huber: 3.2895, swd: 3.9807, ept: 186.1479
    Epoch [17/50], Test Losses: mse: 32.3880, mae: 3.2099, huber: 2.7802, swd: 2.9293, ept: 197.2787
      Epoch 17 composite train-obj: 1.714548
            Val objective improved 3.3294 → 3.2895, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 16.3662, mae: 2.1493, huber: 1.7461, swd: 1.5820, ept: 243.3118
    Epoch [18/50], Val Losses: mse: 39.1864, mae: 3.5987, huber: 3.1670, swd: 3.9354, ept: 190.8791
    Epoch [18/50], Test Losses: mse: 29.0717, mae: 2.9779, huber: 2.5553, swd: 2.7120, ept: 210.0089
      Epoch 18 composite train-obj: 1.746058
            Val objective improved 3.2895 → 3.1670, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 16.0459, mae: 2.1147, huber: 1.7133, swd: 1.5680, ept: 245.8239
    Epoch [19/50], Val Losses: mse: 41.4566, mae: 3.6627, huber: 3.2327, swd: 3.9411, ept: 194.7631
    Epoch [19/50], Test Losses: mse: 31.2895, mae: 3.0711, huber: 2.6483, swd: 2.4632, ept: 210.1685
      Epoch 19 composite train-obj: 1.713298
            No improvement (3.2327), counter 1/5
    Epoch [20/50], Train Losses: mse: 15.1022, mae: 2.0235, huber: 1.6266, swd: 1.4175, ept: 250.3378
    Epoch [20/50], Val Losses: mse: 41.4697, mae: 3.6714, huber: 3.2412, swd: 3.8533, ept: 193.7549
    Epoch [20/50], Test Losses: mse: 31.5416, mae: 3.0896, huber: 2.6670, swd: 2.9461, ept: 212.4274
      Epoch 20 composite train-obj: 1.626561
            No improvement (3.2412), counter 2/5
    Epoch [21/50], Train Losses: mse: 14.4355, mae: 1.9634, huber: 1.5694, swd: 1.3824, ept: 253.6737
    Epoch [21/50], Val Losses: mse: 40.6781, mae: 3.6730, huber: 3.2420, swd: 3.9786, ept: 186.8508
    Epoch [21/50], Test Losses: mse: 30.4942, mae: 3.0352, huber: 2.6143, swd: 2.5412, ept: 207.5296
      Epoch 21 composite train-obj: 1.569407
            No improvement (3.2420), counter 3/5
    Epoch [22/50], Train Losses: mse: 14.0752, mae: 1.9347, huber: 1.5419, swd: 1.3165, ept: 255.2888
    Epoch [22/50], Val Losses: mse: 38.3247, mae: 3.5415, huber: 3.1118, swd: 3.8427, ept: 191.0741
    Epoch [22/50], Test Losses: mse: 29.6427, mae: 2.9835, huber: 2.5621, swd: 2.5753, ept: 210.6356
      Epoch 22 composite train-obj: 1.541909
            Val objective improved 3.1670 → 3.1118, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 13.8702, mae: 1.9110, huber: 1.5193, swd: 1.2789, ept: 257.1260
    Epoch [23/50], Val Losses: mse: 41.4156, mae: 3.7031, huber: 3.2698, swd: 4.3385, ept: 189.9310
    Epoch [23/50], Test Losses: mse: 30.8772, mae: 3.0552, huber: 2.6311, swd: 2.6610, ept: 211.1741
      Epoch 23 composite train-obj: 1.519331
            No improvement (3.2698), counter 1/5
    Epoch [24/50], Train Losses: mse: 14.2096, mae: 1.9356, huber: 1.5428, swd: 1.2886, ept: 256.6452
    Epoch [24/50], Val Losses: mse: 40.8921, mae: 3.5965, huber: 3.1681, swd: 3.4593, ept: 199.6703
    Epoch [24/50], Test Losses: mse: 27.1076, mae: 2.8225, huber: 2.4049, swd: 2.1243, ept: 218.5340
      Epoch 24 composite train-obj: 1.542796
            No improvement (3.1681), counter 2/5
    Epoch [25/50], Train Losses: mse: 12.8704, mae: 1.8137, huber: 1.4275, swd: 1.1692, ept: 262.2123
    Epoch [25/50], Val Losses: mse: 40.5903, mae: 3.6160, huber: 3.1875, swd: 3.7049, ept: 192.6651
    Epoch [25/50], Test Losses: mse: 27.1576, mae: 2.8182, huber: 2.4010, swd: 2.3024, ept: 218.8192
      Epoch 25 composite train-obj: 1.427511
            No improvement (3.1875), counter 3/5
    Epoch [26/50], Train Losses: mse: 12.3961, mae: 1.7817, huber: 1.3961, swd: 1.1079, ept: 262.8422
    Epoch [26/50], Val Losses: mse: 39.8379, mae: 3.5103, huber: 3.0851, swd: 3.3536, ept: 202.8393
    Epoch [26/50], Test Losses: mse: 29.1014, mae: 2.8759, huber: 2.4605, swd: 2.2751, ept: 221.4715
      Epoch 26 composite train-obj: 1.396061
            Val objective improved 3.1118 → 3.0851, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 12.3895, mae: 1.7737, huber: 1.3889, swd: 1.0850, ept: 264.3114
    Epoch [27/50], Val Losses: mse: 39.3327, mae: 3.5836, huber: 3.1526, swd: 3.5414, ept: 192.6773
    Epoch [27/50], Test Losses: mse: 30.0206, mae: 2.9782, huber: 2.5570, swd: 2.3373, ept: 215.6806
      Epoch 27 composite train-obj: 1.388949
            No improvement (3.1526), counter 1/5
    Epoch [28/50], Train Losses: mse: 12.6170, mae: 1.7893, huber: 1.4040, swd: 1.1096, ept: 263.7579
    Epoch [28/50], Val Losses: mse: 36.7496, mae: 3.3639, huber: 2.9375, swd: 3.2361, ept: 205.9066
    Epoch [28/50], Test Losses: mse: 29.5051, mae: 2.9080, huber: 2.4911, swd: 2.2890, ept: 222.1879
      Epoch 28 composite train-obj: 1.404031
            Val objective improved 3.0851 → 2.9375, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 11.5269, mae: 1.6781, huber: 1.2998, swd: 1.0059, ept: 268.7310
    Epoch [29/50], Val Losses: mse: 40.8620, mae: 3.6263, huber: 3.1999, swd: 3.7624, ept: 199.4112
    Epoch [29/50], Test Losses: mse: 28.7898, mae: 2.8674, huber: 2.4539, swd: 2.2719, ept: 221.1782
      Epoch 29 composite train-obj: 1.299823
            No improvement (3.1999), counter 1/5
    Epoch [30/50], Train Losses: mse: 11.8136, mae: 1.7132, huber: 1.3323, swd: 1.0147, ept: 267.2662
    Epoch [30/50], Val Losses: mse: 40.1039, mae: 3.5620, huber: 3.1370, swd: 3.9199, ept: 200.0780
    Epoch [30/50], Test Losses: mse: 28.1130, mae: 2.8552, huber: 2.4403, swd: 2.2647, ept: 220.1588
      Epoch 30 composite train-obj: 1.332277
            No improvement (3.1370), counter 2/5
    Epoch [31/50], Train Losses: mse: 11.4251, mae: 1.6608, huber: 1.2839, swd: 0.9755, ept: 270.1707
    Epoch [31/50], Val Losses: mse: 39.1525, mae: 3.4800, huber: 3.0549, swd: 3.2224, ept: 202.4152
    Epoch [31/50], Test Losses: mse: 29.0509, mae: 2.8791, huber: 2.4631, swd: 2.0195, ept: 220.5990
      Epoch 31 composite train-obj: 1.283899
            No improvement (3.0549), counter 3/5
    Epoch [32/50], Train Losses: mse: 11.5760, mae: 1.6769, huber: 1.2989, swd: 1.0036, ept: 269.7505
    Epoch [32/50], Val Losses: mse: 37.4264, mae: 3.3757, huber: 2.9537, swd: 3.2586, ept: 206.7222
    Epoch [32/50], Test Losses: mse: 28.4980, mae: 2.8351, huber: 2.4230, swd: 2.0722, ept: 225.3806
      Epoch 32 composite train-obj: 1.298947
            No improvement (2.9537), counter 4/5
    Epoch [33/50], Train Losses: mse: 11.0304, mae: 1.6193, huber: 1.2452, swd: 0.9262, ept: 272.0842
    Epoch [33/50], Val Losses: mse: 36.0095, mae: 3.3548, huber: 2.9333, swd: 3.5359, ept: 201.0669
    Epoch [33/50], Test Losses: mse: 26.2317, mae: 2.7525, huber: 2.3410, swd: 2.3210, ept: 224.7684
      Epoch 33 composite train-obj: 1.245215
            Val objective improved 2.9375 → 2.9333, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 10.6371, mae: 1.5944, huber: 1.2210, swd: 0.9055, ept: 273.0085
    Epoch [34/50], Val Losses: mse: 41.3444, mae: 3.5354, huber: 3.1121, swd: 3.1728, ept: 205.8207
    Epoch [34/50], Test Losses: mse: 27.1806, mae: 2.7165, huber: 2.3081, swd: 1.8958, ept: 231.5749
      Epoch 34 composite train-obj: 1.220988
            No improvement (3.1121), counter 1/5
    Epoch [35/50], Train Losses: mse: 10.4802, mae: 1.5621, huber: 1.1914, swd: 0.8752, ept: 275.3484
    Epoch [35/50], Val Losses: mse: 40.1619, mae: 3.5363, huber: 3.1105, swd: 3.1404, ept: 205.6894
    Epoch [35/50], Test Losses: mse: 28.0950, mae: 2.8196, huber: 2.4052, swd: 2.0572, ept: 226.4568
      Epoch 35 composite train-obj: 1.191369
            No improvement (3.1105), counter 2/5
    Epoch [36/50], Train Losses: mse: 9.8805, mae: 1.5150, huber: 1.1468, swd: 0.8316, ept: 277.1281
    Epoch [36/50], Val Losses: mse: 38.3164, mae: 3.4112, huber: 2.9914, swd: 2.9448, ept: 204.3986
    Epoch [36/50], Test Losses: mse: 27.1218, mae: 2.7497, huber: 2.3419, swd: 2.0285, ept: 227.5565
      Epoch 36 composite train-obj: 1.146795
            No improvement (2.9914), counter 3/5
    Epoch [37/50], Train Losses: mse: 10.2947, mae: 1.5450, huber: 1.1758, swd: 0.8490, ept: 276.2369
    Epoch [37/50], Val Losses: mse: 39.4842, mae: 3.4683, huber: 3.0494, swd: 3.2891, ept: 207.0595
    Epoch [37/50], Test Losses: mse: 25.6720, mae: 2.6703, huber: 2.2638, swd: 2.0668, ept: 231.0513
      Epoch 37 composite train-obj: 1.175797
            No improvement (3.0494), counter 4/5
    Epoch [38/50], Train Losses: mse: 9.7901, mae: 1.5074, huber: 1.1393, swd: 0.7911, ept: 277.4286
    Epoch [38/50], Val Losses: mse: 38.7823, mae: 3.4309, huber: 3.0086, swd: 3.2321, ept: 207.6785
    Epoch [38/50], Test Losses: mse: 27.0264, mae: 2.7159, huber: 2.3065, swd: 1.9309, ept: 230.9051
      Epoch 38 composite train-obj: 1.139300
    Epoch [38/50], Test Losses: mse: 26.2296, mae: 2.7524, huber: 2.3410, swd: 2.3217, ept: 224.7966
    Best round's Test MSE: 26.2317, MAE: 2.7525, SWD: 2.3210
    Best round's Validation MSE: 36.0095, MAE: 3.3548, SWD: 3.5359
    Best round's Test verification MSE : 26.2296, MAE: 2.7524, SWD: 2.3217
    Time taken: 144.95 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1735)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 27.0790 ± 0.8613
      mae: 2.7432 ± 0.0344
      huber: 2.3341 ± 0.0284
      swd: 2.0342 ± 0.2611
      ept: 227.5465 ± 5.7763
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 35.4572 ± 2.0712
      mae: 3.2791 ± 0.1396
      huber: 2.8595 ± 0.1354
      swd: 2.9020 ± 0.4505
      ept: 204.7451 ± 6.0065
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 457.42 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1735
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### AB: Koopman, but with real valued eigenvalues
It is slightly better; yet in previous AB studies (not presented here), it has 
a negative impact. Worth investigating more.


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128, 
    # ablate_no_koopman=False,
    # use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    # ablate_rotate_back_Koopman=True, 
    ablate_shift_inside_scale=False,
    householder_reflects_latent = 2,
    householder_reflects_data = 4,
    mixing_strategy='delay_only', 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
    ablate_no_koopman=False, ### HERE
    ablate_no_shift_in_z_push=False, ### HERE
    ablate_rotate_back_Koopman=True, ### HERE
    use_complex_eigenvalues=False, ###HERE
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 74.3266, mae: 6.5198, huber: 6.0397, swd: 42.9020, ept: 32.9033
    Epoch [1/50], Val Losses: mse: 60.8405, mae: 5.9287, huber: 5.4517, swd: 27.2381, ept: 34.2182
    Epoch [1/50], Test Losses: mse: 55.6754, mae: 5.5203, huber: 5.0467, swd: 27.6208, ept: 34.0200
      Epoch 1 composite train-obj: 6.039745
            Val objective improved inf → 5.4517, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.6015, mae: 4.8881, huber: 4.4223, swd: 17.3547, ept: 85.8840
    Epoch [2/50], Val Losses: mse: 50.7871, mae: 5.0623, huber: 4.5946, swd: 15.7120, ept: 83.2584
    Epoch [2/50], Test Losses: mse: 46.9575, mae: 4.7351, huber: 4.2712, swd: 15.5202, ept: 78.6747
      Epoch 2 composite train-obj: 4.422280
            Val objective improved 5.4517 → 4.5946, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.7560, mae: 4.1885, huber: 3.7342, swd: 9.7724, ept: 134.1006
    Epoch [3/50], Val Losses: mse: 45.9539, mae: 4.5771, huber: 4.1176, swd: 9.4135, ept: 114.2753
    Epoch [3/50], Test Losses: mse: 41.3179, mae: 4.2635, huber: 3.8068, swd: 9.5715, ept: 114.5604
      Epoch 3 composite train-obj: 3.734175
            Val objective improved 4.5946 → 4.1176, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.4091, mae: 3.7140, huber: 3.2692, swd: 6.0326, ept: 166.9897
    Epoch [4/50], Val Losses: mse: 41.0206, mae: 4.1574, huber: 3.7049, swd: 7.3098, ept: 133.6185
    Epoch [4/50], Test Losses: mse: 38.2218, mae: 3.9750, huber: 3.5237, swd: 7.5845, ept: 137.5728
      Epoch 4 composite train-obj: 3.269218
            Val objective improved 4.1176 → 3.7049, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.4492, mae: 3.4490, huber: 3.0110, swd: 4.6940, ept: 183.9340
    Epoch [5/50], Val Losses: mse: 40.7349, mae: 4.0666, huber: 3.6169, swd: 5.9938, ept: 142.2775
    Epoch [5/50], Test Losses: mse: 36.2283, mae: 3.7762, huber: 3.3301, swd: 6.2941, ept: 149.8832
      Epoch 5 composite train-obj: 3.010978
            Val objective improved 3.7049 → 3.6169, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.5400, mae: 3.2733, huber: 2.8401, swd: 4.0815, ept: 193.1966
    Epoch [6/50], Val Losses: mse: 39.1984, mae: 3.8895, huber: 3.4462, swd: 4.5231, ept: 155.8054
    Epoch [6/50], Test Losses: mse: 34.4997, mae: 3.5887, huber: 3.1484, swd: 4.5950, ept: 161.2892
      Epoch 6 composite train-obj: 2.840090
            Val objective improved 3.6169 → 3.4462, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.2835, mae: 3.0991, huber: 2.6702, swd: 3.4484, ept: 200.5586
    Epoch [7/50], Val Losses: mse: 40.1269, mae: 3.9659, huber: 3.5184, swd: 4.1937, ept: 152.9909
    Epoch [7/50], Test Losses: mse: 33.0236, mae: 3.5785, huber: 3.1334, swd: 4.1890, ept: 163.1578
      Epoch 7 composite train-obj: 2.670241
            No improvement (3.5184), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.8518, mae: 2.9833, huber: 2.5566, swd: 3.0538, ept: 206.3926
    Epoch [8/50], Val Losses: mse: 37.7465, mae: 3.7595, huber: 3.3166, swd: 3.9020, ept: 166.4583
    Epoch [8/50], Test Losses: mse: 31.2003, mae: 3.3678, huber: 2.9285, swd: 3.7314, ept: 176.9584
      Epoch 8 composite train-obj: 2.556628
            Val objective improved 3.4462 → 3.3166, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.7505, mae: 2.8168, huber: 2.3948, swd: 2.5924, ept: 212.8740
    Epoch [9/50], Val Losses: mse: 35.6154, mae: 3.6014, huber: 3.1619, swd: 3.3875, ept: 176.1649
    Epoch [9/50], Test Losses: mse: 29.7020, mae: 3.2486, huber: 2.8126, swd: 2.9998, ept: 184.7546
      Epoch 9 composite train-obj: 2.394814
            Val objective improved 3.3166 → 3.1619, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.2635, mae: 2.7023, huber: 2.2827, swd: 2.2947, ept: 217.7071
    Epoch [10/50], Val Losses: mse: 37.8870, mae: 3.6086, huber: 3.1742, swd: 3.1004, ept: 180.2769
    Epoch [10/50], Test Losses: mse: 30.3372, mae: 3.2140, huber: 2.7823, swd: 2.9620, ept: 192.4904
      Epoch 10 composite train-obj: 2.282682
            No improvement (3.1742), counter 1/5
    Epoch [11/50], Train Losses: mse: 22.0910, mae: 2.5931, huber: 2.1777, swd: 2.0967, ept: 223.6440
    Epoch [11/50], Val Losses: mse: 36.2513, mae: 3.4824, huber: 3.0506, swd: 2.9085, ept: 188.1596
    Epoch [11/50], Test Losses: mse: 31.3933, mae: 3.2399, huber: 2.8088, swd: 2.9694, ept: 193.7067
      Epoch 11 composite train-obj: 2.177736
            Val objective improved 3.1619 → 3.0506, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.7274, mae: 2.4956, huber: 2.0822, swd: 1.9097, ept: 227.6853
    Epoch [12/50], Val Losses: mse: 34.8161, mae: 3.3921, huber: 2.9614, swd: 2.5679, ept: 186.2485
    Epoch [12/50], Test Losses: mse: 28.5568, mae: 3.0741, huber: 2.6451, swd: 2.5966, ept: 198.6912
      Epoch 12 composite train-obj: 2.082221
            Val objective improved 3.0506 → 2.9614, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.6994, mae: 2.3308, huber: 1.9236, swd: 1.6514, ept: 235.0746
    Epoch [13/50], Val Losses: mse: 39.3380, mae: 3.6277, huber: 3.1937, swd: 2.9567, ept: 182.5036
    Epoch [13/50], Test Losses: mse: 30.6891, mae: 3.1782, huber: 2.7479, swd: 2.7083, ept: 198.7215
      Epoch 13 composite train-obj: 1.923555
            No improvement (3.1937), counter 1/5
    Epoch [14/50], Train Losses: mse: 19.0405, mae: 2.3620, huber: 1.9524, swd: 1.6622, ept: 234.6604
    Epoch [14/50], Val Losses: mse: 33.6193, mae: 3.3728, huber: 2.9400, swd: 2.4984, ept: 190.9339
    Epoch [14/50], Test Losses: mse: 28.9890, mae: 3.0659, huber: 2.6397, swd: 2.3790, ept: 202.8406
      Epoch 14 composite train-obj: 1.952393
            Val objective improved 2.9614 → 2.9400, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 17.4988, mae: 2.2425, huber: 1.8366, swd: 1.4808, ept: 240.3474
    Epoch [15/50], Val Losses: mse: 32.9754, mae: 3.2477, huber: 2.8202, swd: 2.1554, ept: 198.2841
    Epoch [15/50], Test Losses: mse: 26.4837, mae: 2.8577, huber: 2.4347, swd: 2.0025, ept: 215.8321
      Epoch 15 composite train-obj: 1.836575
            Val objective improved 2.9400 → 2.8202, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.5528, mae: 2.1526, huber: 1.7513, swd: 1.3584, ept: 244.7646
    Epoch [16/50], Val Losses: mse: 37.1404, mae: 3.3561, huber: 2.9321, swd: 2.1229, ept: 203.4865
    Epoch [16/50], Test Losses: mse: 26.1690, mae: 2.8029, huber: 2.3832, swd: 1.9870, ept: 221.0302
      Epoch 16 composite train-obj: 1.751314
            No improvement (2.9321), counter 1/5
    Epoch [17/50], Train Losses: mse: 16.5777, mae: 2.1538, huber: 1.7512, swd: 1.4165, ept: 247.3971
    Epoch [17/50], Val Losses: mse: 38.8720, mae: 3.4753, huber: 3.0461, swd: 2.1319, ept: 197.1099
    Epoch [17/50], Test Losses: mse: 28.8549, mae: 2.9354, huber: 2.5119, swd: 1.9235, ept: 215.6396
      Epoch 17 composite train-obj: 1.751202
            No improvement (3.0461), counter 2/5
    Epoch [18/50], Train Losses: mse: 15.6044, mae: 2.0628, huber: 1.6651, swd: 1.2366, ept: 250.6819
    Epoch [18/50], Val Losses: mse: 35.8374, mae: 3.3388, huber: 2.9138, swd: 1.9453, ept: 202.4967
    Epoch [18/50], Test Losses: mse: 26.3425, mae: 2.7982, huber: 2.3787, swd: 1.7103, ept: 219.7038
      Epoch 18 composite train-obj: 1.665127
            No improvement (2.9138), counter 3/5
    Epoch [19/50], Train Losses: mse: 14.9335, mae: 1.9889, huber: 1.5958, swd: 1.1637, ept: 255.0146
    Epoch [19/50], Val Losses: mse: 33.4877, mae: 3.1888, huber: 2.7666, swd: 2.0834, ept: 204.0374
    Epoch [19/50], Test Losses: mse: 26.1458, mae: 2.7919, huber: 2.3743, swd: 1.8877, ept: 221.2153
      Epoch 19 composite train-obj: 1.595809
            Val objective improved 2.8202 → 2.7666, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 14.0515, mae: 1.9113, huber: 1.5211, swd: 1.0870, ept: 259.8037
    Epoch [20/50], Val Losses: mse: 34.9569, mae: 3.2392, huber: 2.8156, swd: 1.8298, ept: 207.1324
    Epoch [20/50], Test Losses: mse: 25.0443, mae: 2.7178, huber: 2.2996, swd: 1.5724, ept: 221.5359
      Epoch 20 composite train-obj: 1.521125
            No improvement (2.8156), counter 1/5
    Epoch [21/50], Train Losses: mse: 14.1180, mae: 1.9196, huber: 1.5284, swd: 1.0616, ept: 259.0815
    Epoch [21/50], Val Losses: mse: 33.7754, mae: 3.1825, huber: 2.7597, swd: 1.9140, ept: 207.6688
    Epoch [21/50], Test Losses: mse: 26.2723, mae: 2.7678, huber: 2.3511, swd: 1.7603, ept: 224.8714
      Epoch 21 composite train-obj: 1.528408
            Val objective improved 2.7666 → 2.7597, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.4834, mae: 1.8550, huber: 1.4676, swd: 1.0145, ept: 262.7431
    Epoch [22/50], Val Losses: mse: 34.3981, mae: 3.1949, huber: 2.7743, swd: 1.7884, ept: 213.3123
    Epoch [22/50], Test Losses: mse: 29.1827, mae: 2.9012, huber: 2.4844, swd: 1.7233, ept: 224.8941
      Epoch 22 composite train-obj: 1.467632
            No improvement (2.7743), counter 1/5
    Epoch [23/50], Train Losses: mse: 13.6131, mae: 1.8770, huber: 1.4874, swd: 1.0249, ept: 262.2178
    Epoch [23/50], Val Losses: mse: 34.2893, mae: 3.1505, huber: 2.7326, swd: 1.7736, ept: 215.5535
    Epoch [23/50], Test Losses: mse: 23.7267, mae: 2.5699, huber: 2.1598, swd: 1.5139, ept: 239.6638
      Epoch 23 composite train-obj: 1.487395
            Val objective improved 2.7597 → 2.7326, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 12.4415, mae: 1.7562, huber: 1.3746, swd: 0.9073, ept: 267.8709
    Epoch [24/50], Val Losses: mse: 35.4716, mae: 3.2167, huber: 2.7963, swd: 1.9141, ept: 214.1343
    Epoch [24/50], Test Losses: mse: 24.8571, mae: 2.6705, huber: 2.2564, swd: 1.4791, ept: 226.6517
      Epoch 24 composite train-obj: 1.374588
            No improvement (2.7963), counter 1/5
    Epoch [25/50], Train Losses: mse: 12.3347, mae: 1.7435, huber: 1.3625, swd: 0.8950, ept: 268.8013
    Epoch [25/50], Val Losses: mse: 34.6898, mae: 3.1830, huber: 2.7637, swd: 1.7357, ept: 215.0268
    Epoch [25/50], Test Losses: mse: 24.8211, mae: 2.6202, huber: 2.2069, swd: 1.5695, ept: 235.7881
      Epoch 25 composite train-obj: 1.362542
            No improvement (2.7637), counter 2/5
    Epoch [26/50], Train Losses: mse: 12.2092, mae: 1.7384, huber: 1.3567, swd: 0.8740, ept: 269.1488
    Epoch [26/50], Val Losses: mse: 36.2543, mae: 3.2320, huber: 2.8128, swd: 1.9171, ept: 215.4656
    Epoch [26/50], Test Losses: mse: 25.5972, mae: 2.6572, huber: 2.2457, swd: 1.6977, ept: 234.3789
      Epoch 26 composite train-obj: 1.356720
            No improvement (2.8128), counter 3/5
    Epoch [27/50], Train Losses: mse: 11.8868, mae: 1.7029, huber: 1.3241, swd: 0.8562, ept: 270.7296
    Epoch [27/50], Val Losses: mse: 34.9130, mae: 3.1604, huber: 2.7455, swd: 1.7446, ept: 216.9882
    Epoch [27/50], Test Losses: mse: 25.5855, mae: 2.6249, huber: 2.2173, swd: 1.5119, ept: 239.5129
      Epoch 27 composite train-obj: 1.324085
            No improvement (2.7455), counter 4/5
    Epoch [28/50], Train Losses: mse: 11.7081, mae: 1.6907, huber: 1.3115, swd: 0.8514, ept: 271.9721
    Epoch [28/50], Val Losses: mse: 32.2639, mae: 2.9949, huber: 2.5843, swd: 1.6392, ept: 222.4172
    Epoch [28/50], Test Losses: mse: 23.1632, mae: 2.4673, huber: 2.0646, swd: 1.3768, ept: 245.0402
      Epoch 28 composite train-obj: 1.311472
            Val objective improved 2.7326 → 2.5843, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 11.2462, mae: 1.6359, huber: 1.2614, swd: 0.7962, ept: 274.3197
    Epoch [29/50], Val Losses: mse: 34.6862, mae: 3.0940, huber: 2.6851, swd: 1.7540, ept: 221.7319
    Epoch [29/50], Test Losses: mse: 23.9625, mae: 2.4899, huber: 2.0904, swd: 1.3787, ept: 246.8369
      Epoch 29 composite train-obj: 1.261417
            No improvement (2.6851), counter 1/5
    Epoch [30/50], Train Losses: mse: 11.4658, mae: 1.6542, huber: 1.2774, swd: 0.8299, ept: 274.6665
    Epoch [30/50], Val Losses: mse: 35.2164, mae: 3.1396, huber: 2.7219, swd: 1.5917, ept: 219.0921
    Epoch [30/50], Test Losses: mse: 23.8942, mae: 2.5316, huber: 2.1230, swd: 1.3215, ept: 242.9406
      Epoch 30 composite train-obj: 1.277371
            No improvement (2.7219), counter 2/5
    Epoch [31/50], Train Losses: mse: 10.7563, mae: 1.5839, huber: 1.2127, swd: 0.7578, ept: 276.7207
    Epoch [31/50], Val Losses: mse: 33.1683, mae: 3.0823, huber: 2.6702, swd: 1.9950, ept: 218.5190
    Epoch [31/50], Test Losses: mse: 26.7395, mae: 2.6411, huber: 2.2361, swd: 1.6357, ept: 240.6410
      Epoch 31 composite train-obj: 1.212657
            No improvement (2.6702), counter 3/5
    Epoch [32/50], Train Losses: mse: 10.5016, mae: 1.5568, huber: 1.1870, swd: 0.7443, ept: 278.5720
    Epoch [32/50], Val Losses: mse: 33.7398, mae: 3.0845, huber: 2.6711, swd: 1.5765, ept: 223.4517
    Epoch [32/50], Test Losses: mse: 23.9459, mae: 2.5510, huber: 2.1428, swd: 1.3357, ept: 238.0749
      Epoch 32 composite train-obj: 1.187040
            No improvement (2.6711), counter 4/5
    Epoch [33/50], Train Losses: mse: 10.2287, mae: 1.5415, huber: 1.1717, swd: 0.7179, ept: 279.4264
    Epoch [33/50], Val Losses: mse: 32.2039, mae: 3.0447, huber: 2.6302, swd: 1.8180, ept: 218.3158
    Epoch [33/50], Test Losses: mse: 22.5342, mae: 2.4599, huber: 2.0577, swd: 1.5357, ept: 239.4903
      Epoch 33 composite train-obj: 1.171695
    Epoch [33/50], Test Losses: mse: 23.1632, mae: 2.4673, huber: 2.0647, swd: 1.3773, ept: 245.0418
    Best round's Test MSE: 23.1632, MAE: 2.4673, SWD: 1.3768
    Best round's Validation MSE: 32.2639, MAE: 2.9949, SWD: 1.6392
    Best round's Test verification MSE : 23.1632, MAE: 2.4673, SWD: 1.3773
    Time taken: 120.25 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.0249, mae: 6.5246, huber: 6.0449, swd: 43.4490, ept: 34.7536
    Epoch [1/50], Val Losses: mse: 60.8113, mae: 5.8480, huber: 5.3718, swd: 24.3675, ept: 48.6099
    Epoch [1/50], Test Losses: mse: 55.8742, mae: 5.4785, huber: 5.0054, swd: 25.9125, ept: 46.9696
      Epoch 1 composite train-obj: 6.044864
            Val objective improved inf → 5.3718, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.3781, mae: 4.7957, huber: 4.3309, swd: 15.6306, ept: 97.1788
    Epoch [2/50], Val Losses: mse: 48.8996, mae: 4.9309, huber: 4.4638, swd: 12.5495, ept: 88.4974
    Epoch [2/50], Test Losses: mse: 45.2327, mae: 4.6101, huber: 4.1465, swd: 13.2672, ept: 90.3412
      Epoch 2 composite train-obj: 4.330918
            Val objective improved 5.3718 → 4.4638, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.2520, mae: 4.0804, huber: 3.6276, swd: 8.4074, ept: 140.5960
    Epoch [3/50], Val Losses: mse: 46.4856, mae: 4.5446, huber: 4.0869, swd: 8.4841, ept: 115.9734
    Epoch [3/50], Test Losses: mse: 40.6838, mae: 4.1569, huber: 3.7009, swd: 8.4649, ept: 122.5011
      Epoch 3 composite train-obj: 3.627595
            Val objective improved 4.4638 → 4.0869, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 34.6786, mae: 3.6750, huber: 3.2302, swd: 5.6789, ept: 167.8313
    Epoch [4/50], Val Losses: mse: 42.9450, mae: 4.2484, huber: 3.7956, swd: 6.5862, ept: 130.3816
    Epoch [4/50], Test Losses: mse: 38.2986, mae: 3.9389, huber: 3.4878, swd: 6.3732, ept: 140.9978
      Epoch 4 composite train-obj: 3.230158
            Val objective improved 4.0869 → 3.7956, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 31.9575, mae: 3.4245, huber: 2.9861, swd: 4.5543, ept: 183.7331
    Epoch [5/50], Val Losses: mse: 38.5493, mae: 4.0353, huber: 3.5846, swd: 6.0670, ept: 141.4824
    Epoch [5/50], Test Losses: mse: 34.3889, mae: 3.7148, huber: 3.2670, swd: 5.9796, ept: 153.1048
      Epoch 5 composite train-obj: 2.986117
            Val objective improved 3.7956 → 3.5846, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.8282, mae: 3.2445, huber: 2.8105, swd: 3.7813, ept: 192.1139
    Epoch [6/50], Val Losses: mse: 40.8475, mae: 4.0440, huber: 3.5964, swd: 4.7509, ept: 148.3251
    Epoch [6/50], Test Losses: mse: 34.4576, mae: 3.6376, huber: 3.1930, swd: 4.4912, ept: 163.8145
      Epoch 6 composite train-obj: 2.810494
            No improvement (3.5964), counter 1/5
    Epoch [7/50], Train Losses: mse: 27.9745, mae: 3.0954, huber: 2.6653, swd: 3.3183, ept: 197.6670
    Epoch [7/50], Val Losses: mse: 41.8996, mae: 4.0711, huber: 3.6231, swd: 4.5407, ept: 152.8639
    Epoch [7/50], Test Losses: mse: 33.6411, mae: 3.5490, huber: 3.1068, swd: 4.5223, ept: 168.0782
      Epoch 7 composite train-obj: 2.665264
            No improvement (3.6231), counter 2/5
    Epoch [8/50], Train Losses: mse: 26.4679, mae: 2.9719, huber: 2.5442, swd: 2.9913, ept: 202.5170
    Epoch [8/50], Val Losses: mse: 41.0210, mae: 3.9555, huber: 3.5118, swd: 4.4114, ept: 157.3474
    Epoch [8/50], Test Losses: mse: 32.1969, mae: 3.4136, huber: 2.9744, swd: 3.9757, ept: 171.7541
      Epoch 8 composite train-obj: 2.544218
            Val objective improved 3.5846 → 3.5118, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.4632, mae: 2.8138, huber: 2.3906, swd: 2.5799, ept: 208.8354
    Epoch [9/50], Val Losses: mse: 38.9550, mae: 3.8645, huber: 3.4191, swd: 3.7190, ept: 162.6820
    Epoch [9/50], Test Losses: mse: 31.6938, mae: 3.3907, huber: 2.9515, swd: 3.7012, ept: 174.1115
      Epoch 9 composite train-obj: 2.390563
            Val objective improved 3.5118 → 3.4191, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.2274, mae: 2.7136, huber: 2.2931, swd: 2.3703, ept: 213.1987
    Epoch [10/50], Val Losses: mse: 38.3227, mae: 3.7350, huber: 3.2957, swd: 3.4424, ept: 169.2123
    Epoch [10/50], Test Losses: mse: 33.3558, mae: 3.3912, huber: 2.9555, swd: 3.1050, ept: 186.7027
      Epoch 10 composite train-obj: 2.293060
            Val objective improved 3.4191 → 3.2957, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 22.0169, mae: 2.6129, huber: 2.1952, swd: 2.1504, ept: 218.3974
    Epoch [11/50], Val Losses: mse: 38.9808, mae: 3.7175, huber: 3.2815, swd: 3.5980, ept: 170.6604
    Epoch [11/50], Test Losses: mse: 31.1011, mae: 3.2096, huber: 2.7802, swd: 3.1761, ept: 186.3926
      Epoch 11 composite train-obj: 2.195248
            Val objective improved 3.2957 → 3.2815, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.2300, mae: 2.4723, huber: 2.0591, swd: 1.8736, ept: 223.9360
    Epoch [12/50], Val Losses: mse: 39.9153, mae: 3.7724, huber: 3.3337, swd: 3.5921, ept: 173.2154
    Epoch [12/50], Test Losses: mse: 28.6683, mae: 3.0805, huber: 2.6507, swd: 2.5688, ept: 194.4961
      Epoch 12 composite train-obj: 2.059106
            No improvement (3.3337), counter 1/5
    Epoch [13/50], Train Losses: mse: 20.1970, mae: 2.4798, huber: 2.0651, swd: 1.8471, ept: 224.4571
    Epoch [13/50], Val Losses: mse: 39.7232, mae: 3.7308, huber: 3.2937, swd: 2.9049, ept: 179.4092
    Epoch [13/50], Test Losses: mse: 29.7883, mae: 3.1187, huber: 2.6901, swd: 2.5178, ept: 193.8506
      Epoch 13 composite train-obj: 2.065081
            No improvement (3.2937), counter 2/5
    Epoch [14/50], Train Losses: mse: 18.6625, mae: 2.3389, huber: 1.9305, swd: 1.6528, ept: 231.6436
    Epoch [14/50], Val Losses: mse: 38.2264, mae: 3.5775, huber: 3.1455, swd: 2.7342, ept: 185.1294
    Epoch [14/50], Test Losses: mse: 30.1125, mae: 3.0624, huber: 2.6373, swd: 2.3860, ept: 201.3283
      Epoch 14 composite train-obj: 1.930512
            Val objective improved 3.2815 → 3.1455, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 17.8090, mae: 2.2626, huber: 1.8569, swd: 1.5290, ept: 236.5874
    Epoch [15/50], Val Losses: mse: 40.0519, mae: 3.7114, huber: 3.2758, swd: 3.1416, ept: 178.2798
    Epoch [15/50], Test Losses: mse: 29.7834, mae: 3.1178, huber: 2.6882, swd: 2.5620, ept: 197.8011
      Epoch 15 composite train-obj: 1.856931
            No improvement (3.2758), counter 1/5
    Epoch [16/50], Train Losses: mse: 17.1164, mae: 2.1958, huber: 1.7931, swd: 1.4402, ept: 240.3209
    Epoch [16/50], Val Losses: mse: 38.9868, mae: 3.6284, huber: 3.1936, swd: 2.8086, ept: 179.1658
    Epoch [16/50], Test Losses: mse: 30.3849, mae: 3.1267, huber: 2.6979, swd: 2.4710, ept: 195.5486
      Epoch 16 composite train-obj: 1.793115
            No improvement (3.1936), counter 2/5
    Epoch [17/50], Train Losses: mse: 16.3275, mae: 2.1399, huber: 1.7388, swd: 1.3408, ept: 242.9301
    Epoch [17/50], Val Losses: mse: 35.9094, mae: 3.4144, huber: 2.9866, swd: 3.0377, ept: 194.6977
    Epoch [17/50], Test Losses: mse: 26.5070, mae: 2.8558, huber: 2.4362, swd: 2.5899, ept: 208.5335
      Epoch 17 composite train-obj: 1.738827
            Val objective improved 3.1455 → 2.9866, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 16.5848, mae: 2.1512, huber: 1.7497, swd: 1.3696, ept: 243.7245
    Epoch [18/50], Val Losses: mse: 38.1251, mae: 3.5349, huber: 3.1075, swd: 3.0606, ept: 192.6192
    Epoch [18/50], Test Losses: mse: 26.4709, mae: 2.7929, huber: 2.3770, swd: 2.3935, ept: 215.2215
      Epoch 18 composite train-obj: 1.749684
            No improvement (3.1075), counter 1/5
    Epoch [19/50], Train Losses: mse: 15.6748, mae: 2.0722, huber: 1.6738, swd: 1.2811, ept: 247.9877
    Epoch [19/50], Val Losses: mse: 36.0685, mae: 3.3832, huber: 2.9568, swd: 2.5286, ept: 198.2183
    Epoch [19/50], Test Losses: mse: 28.0782, mae: 2.9111, huber: 2.4913, swd: 2.1677, ept: 213.2683
      Epoch 19 composite train-obj: 1.673789
            Val objective improved 2.9866 → 2.9568, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 14.9658, mae: 2.0068, huber: 1.6118, swd: 1.2006, ept: 251.2356
    Epoch [20/50], Val Losses: mse: 36.4945, mae: 3.4379, huber: 3.0094, swd: 2.6167, ept: 195.7034
    Epoch [20/50], Test Losses: mse: 26.0249, mae: 2.7731, huber: 2.3553, swd: 2.1594, ept: 218.7903
      Epoch 20 composite train-obj: 1.611781
            No improvement (3.0094), counter 1/5
    Epoch [21/50], Train Losses: mse: 14.9340, mae: 2.0081, huber: 1.6120, swd: 1.2082, ept: 251.9649
    Epoch [21/50], Val Losses: mse: 35.9418, mae: 3.3671, huber: 2.9426, swd: 2.4318, ept: 197.9958
    Epoch [21/50], Test Losses: mse: 27.9252, mae: 2.8301, huber: 2.4137, swd: 1.8929, ept: 220.8118
      Epoch 21 composite train-obj: 1.612031
            Val objective improved 2.9568 → 2.9426, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 14.4839, mae: 1.9605, huber: 1.5673, swd: 1.1465, ept: 254.8443
    Epoch [22/50], Val Losses: mse: 41.2200, mae: 3.6673, huber: 3.2348, swd: 2.5858, ept: 190.6564
    Epoch [22/50], Test Losses: mse: 27.1387, mae: 2.8845, huber: 2.4625, swd: 2.0677, ept: 208.9677
      Epoch 22 composite train-obj: 1.567260
            No improvement (3.2348), counter 1/5
    Epoch [23/50], Train Losses: mse: 13.9555, mae: 1.9066, huber: 1.5168, swd: 1.1001, ept: 258.0220
    Epoch [23/50], Val Losses: mse: 36.4860, mae: 3.3675, huber: 2.9424, swd: 2.2840, ept: 205.6855
    Epoch [23/50], Test Losses: mse: 26.1350, mae: 2.7431, huber: 2.3278, swd: 1.8597, ept: 224.6578
      Epoch 23 composite train-obj: 1.516808
            Val objective improved 2.9426 → 2.9424, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 13.4874, mae: 1.8550, huber: 1.4683, swd: 1.0516, ept: 260.7443
    Epoch [24/50], Val Losses: mse: 37.3673, mae: 3.4045, huber: 2.9768, swd: 2.0928, ept: 202.4667
    Epoch [24/50], Test Losses: mse: 26.3202, mae: 2.7552, huber: 2.3374, swd: 1.7553, ept: 221.8605
      Epoch 24 composite train-obj: 1.468267
            No improvement (2.9768), counter 1/5
    Epoch [25/50], Train Losses: mse: 13.7429, mae: 1.8827, huber: 1.4933, swd: 1.0922, ept: 261.1605
    Epoch [25/50], Val Losses: mse: 36.3385, mae: 3.3485, huber: 2.9246, swd: 2.3789, ept: 204.3901
    Epoch [25/50], Test Losses: mse: 25.9134, mae: 2.7000, huber: 2.2866, swd: 1.8225, ept: 227.7705
      Epoch 25 composite train-obj: 1.493269
            Val objective improved 2.9424 → 2.9246, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 12.2676, mae: 1.7439, huber: 1.3630, swd: 0.9381, ept: 267.0928
    Epoch [26/50], Val Losses: mse: 36.7796, mae: 3.3306, huber: 2.9096, swd: 2.1108, ept: 207.4577
    Epoch [26/50], Test Losses: mse: 25.5897, mae: 2.6622, huber: 2.2514, swd: 1.6335, ept: 228.8900
      Epoch 26 composite train-obj: 1.362977
            Val objective improved 2.9246 → 2.9096, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 12.3425, mae: 1.7581, huber: 1.3754, swd: 0.9469, ept: 266.8965
    Epoch [27/50], Val Losses: mse: 35.4860, mae: 3.2692, huber: 2.8498, swd: 2.3319, ept: 208.2084
    Epoch [27/50], Test Losses: mse: 26.1262, mae: 2.6946, huber: 2.2842, swd: 1.8020, ept: 232.1299
      Epoch 27 composite train-obj: 1.375405
            Val objective improved 2.9096 → 2.8498, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 12.2042, mae: 1.7411, huber: 1.3593, swd: 0.9463, ept: 268.5734
    Epoch [28/50], Val Losses: mse: 37.3274, mae: 3.3722, huber: 2.9494, swd: 2.3590, ept: 206.2434
    Epoch [28/50], Test Losses: mse: 25.8930, mae: 2.7013, huber: 2.2895, swd: 1.8289, ept: 228.5849
      Epoch 28 composite train-obj: 1.359300
            No improvement (2.9494), counter 1/5
    Epoch [29/50], Train Losses: mse: 12.2025, mae: 1.7387, huber: 1.3577, swd: 0.9285, ept: 268.7199
    Epoch [29/50], Val Losses: mse: 36.0292, mae: 3.2925, huber: 2.8734, swd: 2.3190, ept: 206.0258
    Epoch [29/50], Test Losses: mse: 25.8525, mae: 2.6699, huber: 2.2606, swd: 1.8514, ept: 231.2897
      Epoch 29 composite train-obj: 1.357686
            No improvement (2.8734), counter 2/5
    Epoch [30/50], Train Losses: mse: 11.6367, mae: 1.6734, huber: 1.2967, swd: 0.8700, ept: 272.0462
    Epoch [30/50], Val Losses: mse: 35.0253, mae: 3.2327, huber: 2.8136, swd: 2.0893, ept: 209.5609
    Epoch [30/50], Test Losses: mse: 25.9436, mae: 2.6453, huber: 2.2370, swd: 1.7516, ept: 235.2905
      Epoch 30 composite train-obj: 1.296712
            Val objective improved 2.8498 → 2.8136, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 11.7654, mae: 1.6944, huber: 1.3156, swd: 0.8895, ept: 270.9404
    Epoch [31/50], Val Losses: mse: 36.7734, mae: 3.3374, huber: 2.9129, swd: 2.3231, ept: 210.5411
    Epoch [31/50], Test Losses: mse: 25.2310, mae: 2.6468, huber: 2.2328, swd: 1.9347, ept: 238.1896
      Epoch 31 composite train-obj: 1.315644
            No improvement (2.9129), counter 1/5
    Epoch [32/50], Train Losses: mse: 11.7699, mae: 1.6923, huber: 1.3131, swd: 0.8848, ept: 272.0492
    Epoch [32/50], Val Losses: mse: 35.6619, mae: 3.2562, huber: 2.8369, swd: 2.2290, ept: 212.8851
    Epoch [32/50], Test Losses: mse: 26.4313, mae: 2.6956, huber: 2.2860, swd: 1.8106, ept: 234.5058
      Epoch 32 composite train-obj: 1.313053
            No improvement (2.8369), counter 2/5
    Epoch [33/50], Train Losses: mse: 10.8657, mae: 1.6057, huber: 1.2326, swd: 0.8160, ept: 275.7443
    Epoch [33/50], Val Losses: mse: 34.8495, mae: 3.1961, huber: 2.7779, swd: 2.1106, ept: 217.2203
    Epoch [33/50], Test Losses: mse: 24.5353, mae: 2.5549, huber: 2.1480, swd: 1.4785, ept: 241.6006
      Epoch 33 composite train-obj: 1.232569
            Val objective improved 2.8136 → 2.7779, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 10.9012, mae: 1.6053, huber: 1.2323, swd: 0.7924, ept: 276.1385
    Epoch [34/50], Val Losses: mse: 36.2529, mae: 3.2935, huber: 2.8736, swd: 2.4423, ept: 209.1349
    Epoch [34/50], Test Losses: mse: 25.7063, mae: 2.6807, huber: 2.2711, swd: 2.0282, ept: 231.1920
      Epoch 34 composite train-obj: 1.232250
            No improvement (2.8736), counter 1/5
    Epoch [35/50], Train Losses: mse: 11.4932, mae: 1.6536, huber: 1.2779, swd: 0.8343, ept: 274.7057
    Epoch [35/50], Val Losses: mse: 32.4062, mae: 3.0808, huber: 2.6643, swd: 2.1082, ept: 215.8242
    Epoch [35/50], Test Losses: mse: 25.0878, mae: 2.5652, huber: 2.1606, swd: 1.7222, ept: 240.7049
      Epoch 35 composite train-obj: 1.277939
            Val objective improved 2.7779 → 2.6643, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 10.8472, mae: 1.5947, huber: 1.2227, swd: 0.8022, ept: 277.0574
    Epoch [36/50], Val Losses: mse: 38.9806, mae: 3.4301, huber: 3.0103, swd: 2.3585, ept: 209.5045
    Epoch [36/50], Test Losses: mse: 25.2203, mae: 2.5929, huber: 2.1865, swd: 1.5476, ept: 236.2267
      Epoch 36 composite train-obj: 1.222711
            No improvement (3.0103), counter 1/5
    Epoch [37/50], Train Losses: mse: 10.8221, mae: 1.5904, huber: 1.2185, swd: 0.7963, ept: 277.0593
    Epoch [37/50], Val Losses: mse: 35.4978, mae: 3.2541, huber: 2.8379, swd: 2.1540, ept: 210.3464
    Epoch [37/50], Test Losses: mse: 26.2301, mae: 2.6425, huber: 2.2376, swd: 1.5660, ept: 234.0727
      Epoch 37 composite train-obj: 1.218519
            No improvement (2.8379), counter 2/5
    Epoch [38/50], Train Losses: mse: 10.4209, mae: 1.5562, huber: 1.1866, swd: 0.7557, ept: 278.7954
    Epoch [38/50], Val Losses: mse: 36.8835, mae: 3.2887, huber: 2.8715, swd: 1.9369, ept: 212.0289
    Epoch [38/50], Test Losses: mse: 25.0527, mae: 2.5559, huber: 2.1518, swd: 1.5078, ept: 241.1151
      Epoch 38 composite train-obj: 1.186641
            No improvement (2.8715), counter 3/5
    Epoch [39/50], Train Losses: mse: 10.8751, mae: 1.6005, huber: 1.2273, swd: 0.7838, ept: 276.9229
    Epoch [39/50], Val Losses: mse: 34.8707, mae: 3.2603, huber: 2.8396, swd: 2.2837, ept: 212.6095
    Epoch [39/50], Test Losses: mse: 25.8176, mae: 2.6496, huber: 2.2401, swd: 1.5584, ept: 238.3488
      Epoch 39 composite train-obj: 1.227340
            No improvement (2.8396), counter 4/5
    Epoch [40/50], Train Losses: mse: 10.4372, mae: 1.5465, huber: 1.1771, swd: 0.7442, ept: 280.1167
    Epoch [40/50], Val Losses: mse: 37.6667, mae: 3.3335, huber: 2.9189, swd: 2.0815, ept: 212.7913
    Epoch [40/50], Test Losses: mse: 25.5223, mae: 2.5779, huber: 2.1770, swd: 1.6195, ept: 240.4948
      Epoch 40 composite train-obj: 1.177129
    Epoch [40/50], Test Losses: mse: 25.0944, mae: 2.5654, huber: 2.1607, swd: 1.7199, ept: 240.7924
    Best round's Test MSE: 25.0878, MAE: 2.5652, SWD: 1.7222
    Best round's Validation MSE: 32.4062, MAE: 3.0808, SWD: 2.1082
    Best round's Test verification MSE : 25.0944, MAE: 2.5654, SWD: 1.7199
    Time taken: 149.46 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.2622, mae: 6.5124, huber: 6.0327, swd: 44.5486, ept: 35.0690
    Epoch [1/50], Val Losses: mse: 59.5260, mae: 5.7914, huber: 5.3150, swd: 27.9920, ept: 43.8516
    Epoch [1/50], Test Losses: mse: 54.7983, mae: 5.4046, huber: 4.9322, swd: 28.2532, ept: 42.0580
      Epoch 1 composite train-obj: 6.032653
            Val objective improved inf → 5.3150, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.3497, mae: 4.7799, huber: 4.3152, swd: 16.8924, ept: 100.8427
    Epoch [2/50], Val Losses: mse: 50.4073, mae: 4.9581, huber: 4.4926, swd: 14.2412, ept: 91.2357
    Epoch [2/50], Test Losses: mse: 46.9079, mae: 4.6647, huber: 4.2014, swd: 14.5067, ept: 90.1038
      Epoch 2 composite train-obj: 4.315182
            Val objective improved 5.3150 → 4.4926, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.8077, mae: 4.1063, huber: 3.6534, swd: 9.4370, ept: 140.4670
    Epoch [3/50], Val Losses: mse: 47.6973, mae: 4.6089, huber: 4.1507, swd: 9.3769, ept: 113.5596
    Epoch [3/50], Test Losses: mse: 41.4174, mae: 4.2195, huber: 3.7631, swd: 10.5680, ept: 117.1048
      Epoch 3 composite train-obj: 3.653406
            Val objective improved 4.4926 → 4.1507, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.6205, mae: 3.7502, huber: 3.3040, swd: 6.9596, ept: 161.7962
    Epoch [4/50], Val Losses: mse: 42.2209, mae: 4.2548, huber: 3.8003, swd: 6.4399, ept: 132.2777
    Epoch [4/50], Test Losses: mse: 38.2296, mae: 3.9659, huber: 3.5135, swd: 6.6360, ept: 136.0270
      Epoch 4 composite train-obj: 3.303973
            Val objective improved 4.1507 → 3.8003, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.4359, mae: 3.4661, huber: 3.0265, swd: 5.2900, ept: 176.3351
    Epoch [5/50], Val Losses: mse: 40.0087, mae: 4.0916, huber: 3.6399, swd: 6.9669, ept: 133.5922
    Epoch [5/50], Test Losses: mse: 36.4187, mae: 3.8372, huber: 3.3879, swd: 7.1941, ept: 141.8992
      Epoch 5 composite train-obj: 3.026484
            Val objective improved 3.8003 → 3.6399, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.1486, mae: 3.2681, huber: 2.8336, swd: 4.4000, ept: 185.8169
    Epoch [6/50], Val Losses: mse: 41.2922, mae: 4.0860, huber: 3.6376, swd: 6.1799, ept: 148.9188
    Epoch [6/50], Test Losses: mse: 32.4182, mae: 3.5649, huber: 3.1196, swd: 6.1205, ept: 157.0673
      Epoch 6 composite train-obj: 2.833614
            Val objective improved 3.6399 → 3.6376, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.8918, mae: 3.0802, huber: 2.6511, swd: 3.7360, ept: 194.9202
    Epoch [7/50], Val Losses: mse: 43.8283, mae: 4.1314, huber: 3.6843, swd: 4.9825, ept: 151.8098
    Epoch [7/50], Test Losses: mse: 32.8030, mae: 3.5014, huber: 3.0586, swd: 4.9087, ept: 162.5700
      Epoch 7 composite train-obj: 2.651071
            No improvement (3.6843), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.1350, mae: 2.9443, huber: 2.5178, swd: 3.2364, ept: 202.1964
    Epoch [8/50], Val Losses: mse: 43.0314, mae: 4.0269, huber: 3.5836, swd: 4.6716, ept: 161.9963
    Epoch [8/50], Test Losses: mse: 30.4868, mae: 3.2853, huber: 2.8487, swd: 3.9723, ept: 177.8600
      Epoch 8 composite train-obj: 2.517819
            Val objective improved 3.6376 → 3.5836, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.1447, mae: 2.7803, huber: 2.3591, swd: 2.8253, ept: 209.9460
    Epoch [9/50], Val Losses: mse: 43.2510, mae: 3.9663, huber: 3.5269, swd: 4.0007, ept: 168.0559
    Epoch [9/50], Test Losses: mse: 30.0941, mae: 3.2505, huber: 2.8160, swd: 4.0214, ept: 183.0258
      Epoch 9 composite train-obj: 2.359059
            Val objective improved 3.5836 → 3.5269, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 22.7535, mae: 2.6805, huber: 2.2616, swd: 2.5255, ept: 214.4755
    Epoch [10/50], Val Losses: mse: 39.4035, mae: 3.7552, huber: 3.3167, swd: 3.7610, ept: 168.9166
    Epoch [10/50], Test Losses: mse: 30.0448, mae: 3.2319, huber: 2.7980, swd: 3.5525, ept: 185.0833
      Epoch 10 composite train-obj: 2.261637
            Val objective improved 3.5269 → 3.3167, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.4281, mae: 2.5797, huber: 2.1630, swd: 2.2816, ept: 219.5377
    Epoch [11/50], Val Losses: mse: 40.2297, mae: 3.7714, huber: 3.3348, swd: 3.9017, ept: 175.8690
    Epoch [11/50], Test Losses: mse: 28.2543, mae: 3.0727, huber: 2.6427, swd: 3.1646, ept: 194.8933
      Epoch 11 composite train-obj: 2.163009
            No improvement (3.3348), counter 1/5
    Epoch [12/50], Train Losses: mse: 20.3476, mae: 2.4813, huber: 2.0685, swd: 2.0924, ept: 224.7815
    Epoch [12/50], Val Losses: mse: 40.8105, mae: 3.7619, huber: 3.3285, swd: 3.5217, ept: 180.3916
    Epoch [12/50], Test Losses: mse: 26.9232, mae: 2.9700, huber: 2.5442, swd: 2.9675, ept: 198.9977
      Epoch 12 composite train-obj: 2.068512
            No improvement (3.3285), counter 2/5
    Epoch [13/50], Train Losses: mse: 19.8023, mae: 2.4326, huber: 2.0213, swd: 1.9819, ept: 227.5140
    Epoch [13/50], Val Losses: mse: 38.3618, mae: 3.6761, huber: 3.2404, swd: 3.9786, ept: 175.0583
    Epoch [13/50], Test Losses: mse: 27.4994, mae: 3.0473, huber: 2.6182, swd: 3.2583, ept: 193.0292
      Epoch 13 composite train-obj: 2.021344
            Val objective improved 3.3167 → 3.2404, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.1677, mae: 2.3035, huber: 1.8963, swd: 1.7647, ept: 233.0925
    Epoch [14/50], Val Losses: mse: 35.7389, mae: 3.4586, huber: 3.0274, swd: 2.9426, ept: 186.0716
    Epoch [14/50], Test Losses: mse: 26.0301, mae: 2.8765, huber: 2.4537, swd: 2.4086, ept: 208.2248
      Epoch 14 composite train-obj: 1.896314
            Val objective improved 3.2404 → 3.0274, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 17.4313, mae: 2.2388, huber: 1.8340, swd: 1.6245, ept: 236.8050
    Epoch [15/50], Val Losses: mse: 39.9865, mae: 3.6458, huber: 3.2119, swd: 2.7225, ept: 189.9046
    Epoch [15/50], Test Losses: mse: 27.3745, mae: 2.9247, huber: 2.4988, swd: 2.1696, ept: 211.3172
      Epoch 15 composite train-obj: 1.833961
            No improvement (3.2119), counter 1/5
    Epoch [16/50], Train Losses: mse: 17.5722, mae: 2.2506, huber: 1.8448, swd: 1.6351, ept: 237.1443
    Epoch [16/50], Val Losses: mse: 37.6737, mae: 3.5101, huber: 3.0807, swd: 2.9542, ept: 189.6134
    Epoch [16/50], Test Losses: mse: 26.0209, mae: 2.8461, huber: 2.4238, swd: 2.2578, ept: 211.9877
      Epoch 16 composite train-obj: 1.844782
            No improvement (3.0807), counter 2/5
    Epoch [17/50], Train Losses: mse: 15.9760, mae: 2.1199, huber: 1.7192, swd: 1.4567, ept: 243.0108
    Epoch [17/50], Val Losses: mse: 34.6105, mae: 3.3791, huber: 2.9462, swd: 2.8833, ept: 194.9181
    Epoch [17/50], Test Losses: mse: 24.7398, mae: 2.8160, huber: 2.3908, swd: 2.2380, ept: 210.4485
      Epoch 17 composite train-obj: 1.719204
            Val objective improved 3.0274 → 2.9462, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.1067, mae: 2.0360, huber: 1.6394, swd: 1.3549, ept: 247.7926
    Epoch [18/50], Val Losses: mse: 37.7757, mae: 3.4946, huber: 3.0684, swd: 2.6821, ept: 195.6948
    Epoch [18/50], Test Losses: mse: 24.8284, mae: 2.7512, huber: 2.3327, swd: 2.0859, ept: 216.0196
      Epoch 18 composite train-obj: 1.639440
            No improvement (3.0684), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.3148, mae: 1.9656, huber: 1.5721, swd: 1.2573, ept: 251.8487
    Epoch [19/50], Val Losses: mse: 34.3545, mae: 3.2790, huber: 2.8552, swd: 2.7851, ept: 205.2759
    Epoch [19/50], Test Losses: mse: 22.7973, mae: 2.6026, huber: 2.1875, swd: 1.9437, ept: 226.6036
      Epoch 19 composite train-obj: 1.572084
            Val objective improved 2.9462 → 2.8552, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 14.6079, mae: 1.9923, huber: 1.5967, swd: 1.2877, ept: 252.2730
    Epoch [20/50], Val Losses: mse: 33.8686, mae: 3.2298, huber: 2.8080, swd: 2.7178, ept: 203.4122
    Epoch [20/50], Test Losses: mse: 23.7522, mae: 2.6404, huber: 2.2267, swd: 1.9436, ept: 225.8692
      Epoch 20 composite train-obj: 1.596665
            Val objective improved 2.8552 → 2.8080, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 14.4415, mae: 1.9564, huber: 1.5639, swd: 1.2477, ept: 254.6264
    Epoch [21/50], Val Losses: mse: 35.3917, mae: 3.2992, huber: 2.8787, swd: 2.9725, ept: 206.1319
    Epoch [21/50], Test Losses: mse: 23.8045, mae: 2.6325, huber: 2.2213, swd: 2.0937, ept: 227.0662
      Epoch 21 composite train-obj: 1.563856
            No improvement (2.8787), counter 1/5
    Epoch [22/50], Train Losses: mse: 13.7725, mae: 1.8966, huber: 1.5067, swd: 1.1897, ept: 258.1787
    Epoch [22/50], Val Losses: mse: 33.5583, mae: 3.1966, huber: 2.7766, swd: 2.6443, ept: 202.6647
    Epoch [22/50], Test Losses: mse: 23.2880, mae: 2.5979, huber: 2.1852, swd: 1.9722, ept: 228.5344
      Epoch 22 composite train-obj: 1.506653
            Val objective improved 2.8080 → 2.7766, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 12.8870, mae: 1.8158, huber: 1.4305, swd: 1.0922, ept: 261.8030
    Epoch [23/50], Val Losses: mse: 34.0838, mae: 3.1732, huber: 2.7548, swd: 2.5049, ept: 212.9116
    Epoch [23/50], Test Losses: mse: 23.8501, mae: 2.5934, huber: 2.1828, swd: 1.8921, ept: 234.3139
      Epoch 23 composite train-obj: 1.430546
            Val objective improved 2.7766 → 2.7548, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 12.7622, mae: 1.8101, huber: 1.4248, swd: 1.0697, ept: 262.0047
    Epoch [24/50], Val Losses: mse: 36.9337, mae: 3.4278, huber: 3.0024, swd: 3.1958, ept: 200.4525
    Epoch [24/50], Test Losses: mse: 24.4362, mae: 2.6819, huber: 2.2657, swd: 1.9167, ept: 225.4830
      Epoch 24 composite train-obj: 1.424753
            No improvement (3.0024), counter 1/5
    Epoch [25/50], Train Losses: mse: 12.5672, mae: 1.7962, huber: 1.4108, swd: 1.0711, ept: 263.0432
    Epoch [25/50], Val Losses: mse: 33.3944, mae: 3.1773, huber: 2.7548, swd: 2.5955, ept: 213.9472
    Epoch [25/50], Test Losses: mse: 21.4865, mae: 2.5016, huber: 2.0891, swd: 1.8552, ept: 232.4586
      Epoch 25 composite train-obj: 1.410795
            Val objective improved 2.7548 → 2.7548, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 12.2974, mae: 1.7695, huber: 1.3853, swd: 1.0327, ept: 264.9876
    Epoch [26/50], Val Losses: mse: 34.7149, mae: 3.2517, huber: 2.8320, swd: 2.6802, ept: 208.6838
    Epoch [26/50], Test Losses: mse: 23.1113, mae: 2.5732, huber: 2.1629, swd: 1.9529, ept: 229.3548
      Epoch 26 composite train-obj: 1.385308
            No improvement (2.8320), counter 1/5
    Epoch [27/50], Train Losses: mse: 11.7858, mae: 1.7101, huber: 1.3308, swd: 0.9564, ept: 267.6473
    Epoch [27/50], Val Losses: mse: 35.0236, mae: 3.2486, huber: 2.8305, swd: 2.9940, ept: 211.8571
    Epoch [27/50], Test Losses: mse: 23.3288, mae: 2.5535, huber: 2.1451, swd: 1.9169, ept: 231.3256
      Epoch 27 composite train-obj: 1.330802
            No improvement (2.8305), counter 2/5
    Epoch [28/50], Train Losses: mse: 11.8322, mae: 1.7173, huber: 1.3369, swd: 0.9667, ept: 267.6532
    Epoch [28/50], Val Losses: mse: 29.9818, mae: 2.9250, huber: 2.5145, swd: 2.3385, ept: 219.3296
    Epoch [28/50], Test Losses: mse: 23.1535, mae: 2.5187, huber: 2.1135, swd: 1.7621, ept: 237.2632
      Epoch 28 composite train-obj: 1.336873
            Val objective improved 2.7548 → 2.5145, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 12.1499, mae: 1.7407, huber: 1.3589, swd: 1.0035, ept: 267.3569
    Epoch [29/50], Val Losses: mse: 37.3791, mae: 3.3601, huber: 2.9379, swd: 2.4900, ept: 206.6218
    Epoch [29/50], Test Losses: mse: 25.3269, mae: 2.7023, huber: 2.2863, swd: 2.0314, ept: 227.4008
      Epoch 29 composite train-obj: 1.358868
            No improvement (2.9379), counter 1/5
    Epoch [30/50], Train Losses: mse: 10.6428, mae: 1.6075, huber: 1.2337, swd: 0.8676, ept: 272.6089
    Epoch [30/50], Val Losses: mse: 30.7713, mae: 2.9970, huber: 2.5814, swd: 2.0683, ept: 216.3985
    Epoch [30/50], Test Losses: mse: 20.7692, mae: 2.3872, huber: 1.9836, swd: 1.5730, ept: 242.2284
      Epoch 30 composite train-obj: 1.233696
            No improvement (2.5814), counter 2/5
    Epoch [31/50], Train Losses: mse: 11.1041, mae: 1.6434, huber: 1.2670, swd: 0.9040, ept: 271.9154
    Epoch [31/50], Val Losses: mse: 32.2911, mae: 3.1073, huber: 2.6888, swd: 2.1885, ept: 211.4450
    Epoch [31/50], Test Losses: mse: 21.0802, mae: 2.4065, huber: 2.0007, swd: 1.5718, ept: 239.7860
      Epoch 31 composite train-obj: 1.267043
            No improvement (2.6888), counter 3/5
    Epoch [32/50], Train Losses: mse: 10.8218, mae: 1.6235, huber: 1.2480, swd: 0.8791, ept: 273.1685
    Epoch [32/50], Val Losses: mse: 36.5473, mae: 3.2754, huber: 2.8580, swd: 2.7183, ept: 210.7185
    Epoch [32/50], Test Losses: mse: 21.2143, mae: 2.4080, huber: 2.0025, swd: 1.8398, ept: 241.1108
      Epoch 32 composite train-obj: 1.248037
            No improvement (2.8580), counter 4/5
    Epoch [33/50], Train Losses: mse: 10.3154, mae: 1.5662, huber: 1.1946, swd: 0.8295, ept: 275.9235
    Epoch [33/50], Val Losses: mse: 34.5246, mae: 3.1152, huber: 2.7052, swd: 2.3474, ept: 223.5797
    Epoch [33/50], Test Losses: mse: 23.4558, mae: 2.5041, huber: 2.1015, swd: 1.7463, ept: 242.6897
      Epoch 33 composite train-obj: 1.194644
    Epoch [33/50], Test Losses: mse: 23.1505, mae: 2.5185, huber: 2.1134, swd: 1.7624, ept: 237.2833
    Best round's Test MSE: 23.1535, MAE: 2.5187, SWD: 1.7621
    Best round's Validation MSE: 29.9818, MAE: 2.9250, SWD: 2.3385
    Best round's Test verification MSE : 23.1505, MAE: 2.5185, SWD: 1.7624
    Time taken: 166.16 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1750)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 23.8015 ± 0.9096
      mae: 2.5171 ± 0.0400
      huber: 2.1129 ± 0.0392
      swd: 1.6204 ± 0.1730
      ept: 241.0028 ± 3.1819
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 31.5507 ± 1.1108
      mae: 3.0002 ± 0.0637
      huber: 2.5877 ± 0.0612
      swd: 2.0286 ± 0.2910
      ept: 219.1903 ± 2.6934
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 436.01 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1750
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    


