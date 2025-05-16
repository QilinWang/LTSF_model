# Data


```python
%load_ext autoreload
%autoreload 2
import importlib
from importlib import reload  
  
import monotonic
import utils
from train import execute_model_evaluation
from train_config import FlatACLConfig
import train_config
import data_manager
from data_manager import DatasetManager
import metrics
from dataclasses import replace

reload(utils)
reload(monotonic)
reload(train_config)


%load_ext autoreload
%autoreload 2 
data_mgr = DatasetManager(device='cuda')
 
data_mgr.load_csv('ettm2', './ettm2.csv')
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    
    ==================================================
    Dataset: ettm2 (csv)
    ==================================================
    Shape: torch.Size([69680, 7])
    Channels: 7
    Length: 69680
    Source: ./ettm2.csv
    
    Sample data (first 2 rows):
    tensor([[41.1300, 12.4810, 36.5360,  9.3550,  4.4240,  1.3110, 38.6620],
            [39.6220, 11.3090, 35.5440,  8.5510,  3.2090,  1.2580, 38.2230]])
    ==================================================
    




    <data_manager.DatasetManager at 0x1718d4fa840>



# Exp-ETTm2

## Seq=96

### EigenACL

#### pred=96

##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    global_std.shape: torch.Size([7])
    Global Std for ettm2: tensor([10.2434,  6.0312, 13.0618,  4.3690,  6.1544,  6.0135, 11.8865],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 380
    Validation Batches: 53
    Test Batches: 108
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 33.7631, mae: 2.7994, huber: 2.3856, swd: 23.7697, ept: 81.7366
    Epoch [1/50], Val Losses: mse: 13.9235, mae: 2.3118, huber: 1.9116, swd: 6.8322, ept: 83.7469
    Epoch [1/50], Test Losses: mse: 10.4821, mae: 2.0564, huber: 1.6414, swd: 4.9650, ept: 86.2638
      Epoch 1 composite train-obj: 2.385568
            Val objective improved inf → 1.9116, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.8756, mae: 1.9845, huber: 1.5808, swd: 6.2814, ept: 89.0543
    Epoch [2/50], Val Losses: mse: 13.3165, mae: 2.2385, huber: 1.8414, swd: 6.7101, ept: 83.9617
    Epoch [2/50], Test Losses: mse: 10.2917, mae: 2.0142, huber: 1.6013, swd: 5.0099, ept: 86.1418
      Epoch 2 composite train-obj: 1.580810
            Val objective improved 1.9116 → 1.8414, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.4338, mae: 1.9308, huber: 1.5301, swd: 6.0635, ept: 89.3017
    Epoch [3/50], Val Losses: mse: 13.1345, mae: 2.2103, huber: 1.8148, swd: 6.8538, ept: 83.6850
    Epoch [3/50], Test Losses: mse: 10.1905, mae: 1.9983, huber: 1.5861, swd: 5.0865, ept: 86.2087
      Epoch 3 composite train-obj: 1.530061
            Val objective improved 1.8414 → 1.8148, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.1173, mae: 1.8883, huber: 1.4893, swd: 5.9015, ept: 89.5668
    Epoch [4/50], Val Losses: mse: 13.3639, mae: 2.2274, huber: 1.8322, swd: 7.1951, ept: 83.5682
    Epoch [4/50], Test Losses: mse: 10.9624, mae: 2.0809, huber: 1.6641, swd: 5.8254, ept: 85.4830
      Epoch 4 composite train-obj: 1.489313
            No improvement (1.8322), counter 1/5
    Epoch [5/50], Train Losses: mse: 10.9422, mae: 1.8701, huber: 1.4719, swd: 5.8290, ept: 89.6402
    Epoch [5/50], Val Losses: mse: 13.0935, mae: 2.2033, huber: 1.8106, swd: 7.2338, ept: 83.9640
    Epoch [5/50], Test Losses: mse: 10.1419, mae: 2.0307, huber: 1.6149, swd: 5.3699, ept: 87.0157
      Epoch 5 composite train-obj: 1.471907
            Val objective improved 1.8148 → 1.8106, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.7679, mae: 1.8485, huber: 1.4515, swd: 5.7218, ept: 89.7794
    Epoch [6/50], Val Losses: mse: 13.0420, mae: 2.2042, huber: 1.8119, swd: 6.8955, ept: 83.1405
    Epoch [6/50], Test Losses: mse: 10.5887, mae: 2.0204, huber: 1.6111, swd: 5.5305, ept: 85.1098
      Epoch 6 composite train-obj: 1.451481
            No improvement (1.8119), counter 1/5
    Epoch [7/50], Train Losses: mse: 10.6673, mae: 1.8372, huber: 1.4410, swd: 5.6471, ept: 89.8585
    Epoch [7/50], Val Losses: mse: 13.1655, mae: 2.1929, huber: 1.8037, swd: 6.9533, ept: 83.3403
    Epoch [7/50], Test Losses: mse: 10.2996, mae: 1.9698, huber: 1.5636, swd: 5.3120, ept: 85.7250
      Epoch 7 composite train-obj: 1.440992
            Val objective improved 1.8106 → 1.8037, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.4828, mae: 1.8155, huber: 1.4206, swd: 5.5196, ept: 89.9687
    Epoch [8/50], Val Losses: mse: 12.8475, mae: 2.1665, huber: 1.7761, swd: 6.7102, ept: 83.4457
    Epoch [8/50], Test Losses: mse: 9.9729, mae: 1.9682, huber: 1.5577, swd: 5.0623, ept: 85.9801
      Epoch 8 composite train-obj: 1.420573
            Val objective improved 1.8037 → 1.7761, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 10.4419, mae: 1.8140, huber: 1.4193, swd: 5.5102, ept: 90.0047
    Epoch [9/50], Val Losses: mse: 12.9665, mae: 2.1853, huber: 1.7947, swd: 6.9493, ept: 83.6038
    Epoch [9/50], Test Losses: mse: 9.6589, mae: 1.9506, huber: 1.5398, swd: 4.9025, ept: 86.6473
      Epoch 9 composite train-obj: 1.419267
            No improvement (1.7947), counter 1/5
    Epoch [10/50], Train Losses: mse: 10.3650, mae: 1.8029, huber: 1.4088, swd: 5.4616, ept: 90.0822
    Epoch [10/50], Val Losses: mse: 12.8176, mae: 2.1618, huber: 1.7746, swd: 6.7205, ept: 83.4067
    Epoch [10/50], Test Losses: mse: 9.7233, mae: 1.9213, huber: 1.5155, swd: 4.8785, ept: 85.8326
      Epoch 10 composite train-obj: 1.408817
            Val objective improved 1.7761 → 1.7746, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 10.2897, mae: 1.7961, huber: 1.4029, swd: 5.4185, ept: 90.1071
    Epoch [11/50], Val Losses: mse: 14.2298, mae: 2.2795, huber: 1.8904, swd: 7.8293, ept: 82.1863
    Epoch [11/50], Test Losses: mse: 11.3348, mae: 2.0776, huber: 1.6684, swd: 6.2373, ept: 84.4870
      Epoch 11 composite train-obj: 1.402878
            No improvement (1.8904), counter 1/5
    Epoch [12/50], Train Losses: mse: 10.2297, mae: 1.7885, huber: 1.3958, swd: 5.3767, ept: 90.1508
    Epoch [12/50], Val Losses: mse: 13.2165, mae: 2.2066, huber: 1.8196, swd: 7.2103, ept: 83.2966
    Epoch [12/50], Test Losses: mse: 9.7377, mae: 1.9560, huber: 1.5470, swd: 4.9726, ept: 86.3652
      Epoch 12 composite train-obj: 1.395776
            No improvement (1.8196), counter 2/5
    Epoch [13/50], Train Losses: mse: 10.1164, mae: 1.7746, huber: 1.3827, swd: 5.2969, ept: 90.2289
    Epoch [13/50], Val Losses: mse: 12.9225, mae: 2.1676, huber: 1.7798, swd: 6.7057, ept: 83.3539
    Epoch [13/50], Test Losses: mse: 9.7912, mae: 1.9443, huber: 1.5354, swd: 4.9060, ept: 85.9480
      Epoch 13 composite train-obj: 1.382654
            No improvement (1.7798), counter 3/5
    Epoch [14/50], Train Losses: mse: 10.0784, mae: 1.7700, huber: 1.3782, swd: 5.2742, ept: 90.2668
    Epoch [14/50], Val Losses: mse: 13.3268, mae: 2.2097, huber: 1.8222, swd: 7.0235, ept: 83.0905
    Epoch [14/50], Test Losses: mse: 10.0491, mae: 1.9523, huber: 1.5457, swd: 5.1437, ept: 85.3700
      Epoch 14 composite train-obj: 1.378179
            No improvement (1.8222), counter 4/5
    Epoch [15/50], Train Losses: mse: 9.9703, mae: 1.7569, huber: 1.3661, swd: 5.1966, ept: 90.3295
    Epoch [15/50], Val Losses: mse: 13.6061, mae: 2.2304, huber: 1.8414, swd: 7.2491, ept: 82.9327
    Epoch [15/50], Test Losses: mse: 10.3516, mae: 1.9855, huber: 1.5750, swd: 5.3504, ept: 85.4141
      Epoch 15 composite train-obj: 1.366063
    Epoch [15/50], Test Losses: mse: 9.7233, mae: 1.9213, huber: 1.5155, swd: 4.8785, ept: 85.8324
    Best round's Test MSE: 9.7233, MAE: 1.9213, SWD: 4.8785
    Best round's Validation MSE: 12.8176, MAE: 2.1618, SWD: 6.7205
    Best round's Test verification MSE : 9.7233, MAE: 1.9213, SWD: 4.8785
    Time taken: 163.74 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 33.3367, mae: 2.8017, huber: 2.3871, swd: 23.3579, ept: 82.3376
    Epoch [1/50], Val Losses: mse: 14.0518, mae: 2.3174, huber: 1.9150, swd: 6.8287, ept: 83.8247
    Epoch [1/50], Test Losses: mse: 10.4410, mae: 2.0652, huber: 1.6475, swd: 4.7322, ept: 86.3445
      Epoch 1 composite train-obj: 2.387135
            Val objective improved inf → 1.9150, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.0105, mae: 1.9928, huber: 1.5888, swd: 6.2072, ept: 89.0057
    Epoch [2/50], Val Losses: mse: 13.4033, mae: 2.2592, huber: 1.8584, swd: 6.7106, ept: 84.0406
    Epoch [2/50], Test Losses: mse: 10.1568, mae: 2.0427, huber: 1.6225, swd: 4.7959, ept: 86.9633
      Epoch 2 composite train-obj: 1.588826
            Val objective improved 1.9150 → 1.8584, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.3131, mae: 1.9115, huber: 1.5116, swd: 5.8160, ept: 89.4144
    Epoch [3/50], Val Losses: mse: 13.3622, mae: 2.2295, huber: 1.8337, swd: 6.8316, ept: 83.2075
    Epoch [3/50], Test Losses: mse: 10.4983, mae: 2.0211, huber: 1.6090, swd: 5.1714, ept: 85.8164
      Epoch 3 composite train-obj: 1.511634
            Val objective improved 1.8584 → 1.8337, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.1382, mae: 1.8966, huber: 1.4975, swd: 5.7741, ept: 89.5326
    Epoch [4/50], Val Losses: mse: 12.7765, mae: 2.1738, huber: 1.7807, swd: 6.4868, ept: 83.9515
    Epoch [4/50], Test Losses: mse: 9.7972, mae: 1.9725, huber: 1.5602, swd: 4.7009, ept: 86.6854
      Epoch 4 composite train-obj: 1.497518
            Val objective improved 1.8337 → 1.7807, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.7623, mae: 1.8448, huber: 1.4482, swd: 5.5091, ept: 89.8329
    Epoch [5/50], Val Losses: mse: 13.0414, mae: 2.1879, huber: 1.7954, swd: 6.5237, ept: 83.5136
    Epoch [5/50], Test Losses: mse: 10.0966, mae: 1.9699, huber: 1.5615, swd: 4.8525, ept: 85.9173
      Epoch 5 composite train-obj: 1.448247
            No improvement (1.7954), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.7567, mae: 1.8514, huber: 1.4546, swd: 5.5293, ept: 89.8033
    Epoch [6/50], Val Losses: mse: 12.6467, mae: 2.1525, huber: 1.7612, swd: 6.2901, ept: 84.3532
    Epoch [6/50], Test Losses: mse: 9.6476, mae: 1.9519, huber: 1.5397, swd: 4.6388, ept: 87.0239
      Epoch 6 composite train-obj: 1.454624
            Val objective improved 1.7807 → 1.7612, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 10.6276, mae: 1.8325, huber: 1.4373, swd: 5.4348, ept: 89.8829
    Epoch [7/50], Val Losses: mse: 12.9006, mae: 2.1697, huber: 1.7794, swd: 6.3991, ept: 83.8284
    Epoch [7/50], Test Losses: mse: 10.0258, mae: 1.9667, huber: 1.5568, swd: 4.8361, ept: 85.9681
      Epoch 7 composite train-obj: 1.437278
            No improvement (1.7794), counter 1/5
    Epoch [8/50], Train Losses: mse: 10.4687, mae: 1.8158, huber: 1.4213, swd: 5.3239, ept: 89.9942
    Epoch [8/50], Val Losses: mse: 12.8808, mae: 2.1706, huber: 1.7816, swd: 6.5320, ept: 84.1070
    Epoch [8/50], Test Losses: mse: 9.9048, mae: 1.9816, huber: 1.5701, swd: 4.8805, ept: 86.8447
      Epoch 8 composite train-obj: 1.421312
            No improvement (1.7816), counter 2/5
    Epoch [9/50], Train Losses: mse: 10.4754, mae: 1.8220, huber: 1.4273, swd: 5.3466, ept: 89.9696
    Epoch [9/50], Val Losses: mse: 12.9782, mae: 2.1900, huber: 1.8011, swd: 6.4487, ept: 83.7292
    Epoch [9/50], Test Losses: mse: 9.9045, mae: 1.9553, huber: 1.5460, swd: 4.7236, ept: 85.8485
      Epoch 9 composite train-obj: 1.427350
            No improvement (1.8011), counter 3/5
    Epoch [10/50], Train Losses: mse: 10.3988, mae: 1.8072, huber: 1.4138, swd: 5.2950, ept: 90.0461
    Epoch [10/50], Val Losses: mse: 13.0160, mae: 2.1828, huber: 1.7965, swd: 6.5232, ept: 83.4730
    Epoch [10/50], Test Losses: mse: 9.7665, mae: 1.9355, huber: 1.5284, swd: 4.6525, ept: 86.0064
      Epoch 10 composite train-obj: 1.413767
            No improvement (1.7965), counter 4/5
    Epoch [11/50], Train Losses: mse: 10.3115, mae: 1.7972, huber: 1.4045, swd: 5.2462, ept: 90.0952
    Epoch [11/50], Val Losses: mse: 12.7715, mae: 2.1676, huber: 1.7818, swd: 6.3338, ept: 83.8581
    Epoch [11/50], Test Losses: mse: 9.7046, mae: 1.9258, huber: 1.5192, swd: 4.6644, ept: 86.2965
      Epoch 11 composite train-obj: 1.404478
    Epoch [11/50], Test Losses: mse: 9.6480, mae: 1.9519, huber: 1.5397, swd: 4.6391, ept: 87.0223
    Best round's Test MSE: 9.6476, MAE: 1.9519, SWD: 4.6388
    Best round's Validation MSE: 12.6467, MAE: 2.1525, SWD: 6.2901
    Best round's Test verification MSE : 9.6480, MAE: 1.9519, SWD: 4.6391
    Time taken: 120.38 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 36.2459, mae: 2.8699, huber: 2.4555, swd: 23.2686, ept: 82.1593
    Epoch [1/50], Val Losses: mse: 14.0553, mae: 2.3179, huber: 1.9156, swd: 6.4033, ept: 83.6235
    Epoch [1/50], Test Losses: mse: 10.7190, mae: 2.0856, huber: 1.6686, swd: 4.6574, ept: 86.1139
      Epoch 1 composite train-obj: 2.455533
            Val objective improved inf → 1.9156, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.7909, mae: 1.9737, huber: 1.5705, swd: 5.6611, ept: 89.1992
    Epoch [2/50], Val Losses: mse: 13.2245, mae: 2.2308, huber: 1.8319, swd: 6.0161, ept: 83.9923
    Epoch [2/50], Test Losses: mse: 10.2932, mae: 2.0176, huber: 1.6026, swd: 4.5350, ept: 86.5687
      Epoch 2 composite train-obj: 1.570491
            Val objective improved 1.9156 → 1.8319, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.1479, mae: 1.8994, huber: 1.5002, swd: 5.3248, ept: 89.5658
    Epoch [3/50], Val Losses: mse: 14.0782, mae: 2.3154, huber: 1.9163, swd: 6.8492, ept: 82.2299
    Epoch [3/50], Test Losses: mse: 11.3866, mae: 2.1200, huber: 1.7030, swd: 5.5052, ept: 84.7278
      Epoch 3 composite train-obj: 1.500171
            No improvement (1.9163), counter 1/5
    Epoch [4/50], Train Losses: mse: 10.9085, mae: 1.8730, huber: 1.4752, swd: 5.1963, ept: 89.6752
    Epoch [4/50], Val Losses: mse: 13.1130, mae: 2.1921, huber: 1.7974, swd: 6.4387, ept: 83.9670
    Epoch [4/50], Test Losses: mse: 9.9741, mae: 1.9891, huber: 1.5744, swd: 4.5905, ept: 87.0806
      Epoch 4 composite train-obj: 1.475186
            Val objective improved 1.8319 → 1.7974, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.7078, mae: 1.8495, huber: 1.4528, swd: 5.0910, ept: 89.8133
    Epoch [5/50], Val Losses: mse: 13.7054, mae: 2.2599, huber: 1.8644, swd: 6.6078, ept: 82.8278
    Epoch [5/50], Test Losses: mse: 10.6453, mae: 2.0241, huber: 1.6143, swd: 4.9954, ept: 85.2947
      Epoch 5 composite train-obj: 1.452804
            No improvement (1.8644), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.5214, mae: 1.8244, huber: 1.4294, swd: 4.9909, ept: 89.9509
    Epoch [6/50], Val Losses: mse: 12.7862, mae: 2.1610, huber: 1.7707, swd: 6.0434, ept: 84.0007
    Epoch [6/50], Test Losses: mse: 9.7387, mae: 1.9400, huber: 1.5318, swd: 4.3832, ept: 86.4589
      Epoch 6 composite train-obj: 1.429411
            Val objective improved 1.7974 → 1.7707, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 10.4635, mae: 1.8187, huber: 1.4242, swd: 4.9616, ept: 89.9882
    Epoch [7/50], Val Losses: mse: 12.7886, mae: 2.1614, huber: 1.7719, swd: 5.9683, ept: 84.0593
    Epoch [7/50], Test Losses: mse: 9.6417, mae: 1.9382, huber: 1.5290, swd: 4.2738, ept: 86.4980
      Epoch 7 composite train-obj: 1.424231
            No improvement (1.7719), counter 1/5
    Epoch [8/50], Train Losses: mse: 10.3207, mae: 1.7983, huber: 1.4051, swd: 4.8704, ept: 90.0873
    Epoch [8/50], Val Losses: mse: 12.9821, mae: 2.1703, huber: 1.7794, swd: 6.0175, ept: 83.7646
    Epoch [8/50], Test Losses: mse: 9.9901, mae: 1.9661, huber: 1.5537, swd: 4.4992, ept: 85.9554
      Epoch 8 composite train-obj: 1.405059
            No improvement (1.7794), counter 2/5
    Epoch [9/50], Train Losses: mse: 10.2984, mae: 1.7998, huber: 1.4065, swd: 4.8747, ept: 90.0821
    Epoch [9/50], Val Losses: mse: 12.8372, mae: 2.1577, huber: 1.7702, swd: 5.9961, ept: 83.8965
    Epoch [9/50], Test Losses: mse: 9.8046, mae: 1.9338, huber: 1.5265, swd: 4.4538, ept: 86.1619
      Epoch 9 composite train-obj: 1.406519
            Val objective improved 1.7707 → 1.7702, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 10.1652, mae: 1.7791, huber: 1.3875, swd: 4.7834, ept: 90.1970
    Epoch [10/50], Val Losses: mse: 13.6870, mae: 2.2317, huber: 1.8421, swd: 6.5752, ept: 83.3407
    Epoch [10/50], Test Losses: mse: 10.2465, mae: 1.9821, huber: 1.5716, swd: 4.7092, ept: 85.5529
      Epoch 10 composite train-obj: 1.387536
            No improvement (1.8421), counter 1/5
    Epoch [11/50], Train Losses: mse: 10.1140, mae: 1.7752, huber: 1.3839, swd: 4.7518, ept: 90.2027
    Epoch [11/50], Val Losses: mse: 13.2826, mae: 2.2054, huber: 1.8174, swd: 6.2766, ept: 83.2497
    Epoch [11/50], Test Losses: mse: 9.9308, mae: 1.9441, huber: 1.5374, swd: 4.4809, ept: 85.8866
      Epoch 11 composite train-obj: 1.383937
            No improvement (1.8174), counter 2/5
    Epoch [12/50], Train Losses: mse: 10.1628, mae: 1.7861, huber: 1.3942, swd: 4.8031, ept: 90.1916
    Epoch [12/50], Val Losses: mse: 13.7276, mae: 2.2303, huber: 1.8426, swd: 6.5763, ept: 83.3015
    Epoch [12/50], Test Losses: mse: 10.0327, mae: 1.9563, huber: 1.5479, swd: 4.5295, ept: 85.9103
      Epoch 12 composite train-obj: 1.394239
            No improvement (1.8426), counter 3/5
    Epoch [13/50], Train Losses: mse: 9.9958, mae: 1.7602, huber: 1.3698, swd: 4.6957, ept: 90.2888
    Epoch [13/50], Val Losses: mse: 13.4311, mae: 2.2084, huber: 1.8216, swd: 6.3449, ept: 83.1438
    Epoch [13/50], Test Losses: mse: 10.2928, mae: 1.9771, huber: 1.5679, swd: 4.7440, ept: 85.5071
      Epoch 13 composite train-obj: 1.369833
            No improvement (1.8216), counter 4/5
    Epoch [14/50], Train Losses: mse: 9.9472, mae: 1.7549, huber: 1.3647, swd: 4.6671, ept: 90.3065
    Epoch [14/50], Val Losses: mse: 13.6448, mae: 2.2392, huber: 1.8515, swd: 6.4951, ept: 83.0491
    Epoch [14/50], Test Losses: mse: 10.6033, mae: 2.0043, huber: 1.5949, swd: 4.9715, ept: 85.0305
      Epoch 14 composite train-obj: 1.364675
    Epoch [14/50], Test Losses: mse: 9.8046, mae: 1.9338, huber: 1.5265, swd: 4.4539, ept: 86.1628
    Best round's Test MSE: 9.8046, MAE: 1.9338, SWD: 4.4538
    Best round's Validation MSE: 12.8372, MAE: 2.1577, SWD: 5.9961
    Best round's Test verification MSE : 9.8046, MAE: 1.9338, SWD: 4.4539
    Time taken: 154.16 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred96_20250512_1606)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 9.7251 ± 0.0641
      mae: 1.9357 ± 0.0125
      huber: 1.5272 ± 0.0099
      swd: 4.6570 ± 0.1738
      ept: 86.3395 ± 0.5023
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 12.7672 ± 0.0856
      mae: 2.1574 ± 0.0038
      huber: 1.7686 ± 0.0056
      swd: 6.3355 ± 0.2975
      ept: 83.8855 ± 0.3865
      count: 53.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 439.95 seconds
    
    Experiment complete: ACL_ettm2_seq96_pred96_20250512_1606
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

##### ab: rotations (8,4)


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128,
    ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=False, 
    ablate_shift_inside_scale=False,
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.enable_magnitudes = [False, True]
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 380
    Validation Batches: 53
    Test Batches: 108
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.9950, mae: 2.7920, huber: 2.3785, swd: 24.5491, target_std: 20.3267
    Epoch [1/50], Val Losses: mse: 14.2782, mae: 2.3332, huber: 1.9322, swd: 7.4713, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.4285, mae: 2.0526, huber: 1.6357, swd: 5.0073, target_std: 18.4467
      Epoch 1 composite train-obj: 2.378509
            Val objective improved inf → 1.9322, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.1776, mae: 2.0072, huber: 1.6034, swd: 6.5179, target_std: 20.3264
    Epoch [2/50], Val Losses: mse: 13.5401, mae: 2.2468, huber: 1.8483, swd: 7.0217, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 10.6299, mae: 2.0591, huber: 1.6418, swd: 5.2910, target_std: 18.4467
      Epoch 2 composite train-obj: 1.603398
            Val objective improved 1.9322 → 1.8483, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.4281, mae: 1.9244, huber: 1.5243, swd: 6.0760, target_std: 20.3277
    Epoch [3/50], Val Losses: mse: 12.8790, mae: 2.1806, huber: 1.7899, swd: 6.7978, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 10.1537, mae: 1.9787, huber: 1.5697, swd: 5.1016, target_std: 18.4467
      Epoch 3 composite train-obj: 1.524342
            Val objective improved 1.8483 → 1.7899, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.0653, mae: 1.8849, huber: 1.4868, swd: 5.8922, target_std: 20.3274
    Epoch [4/50], Val Losses: mse: 12.9007, mae: 2.1626, huber: 1.7732, swd: 6.8717, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 10.0219, mae: 1.9687, huber: 1.5596, swd: 5.0672, target_std: 18.4467
      Epoch 4 composite train-obj: 1.486768
            Val objective improved 1.7899 → 1.7732, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.8830, mae: 1.8604, huber: 1.4637, swd: 5.7913, target_std: 20.3259
    Epoch [5/50], Val Losses: mse: 13.2470, mae: 2.1936, huber: 1.8028, swd: 7.1211, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 10.0063, mae: 1.9493, huber: 1.5428, swd: 5.0228, target_std: 18.4467
      Epoch 5 composite train-obj: 1.463656
            No improvement (1.8028), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.7245, mae: 1.8428, huber: 1.4471, swd: 5.6861, target_std: 20.3284
    Epoch [6/50], Val Losses: mse: 13.0816, mae: 2.1763, huber: 1.7872, swd: 6.8951, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 10.2162, mae: 1.9728, huber: 1.5644, swd: 5.2592, target_std: 18.4467
      Epoch 6 composite train-obj: 1.447080
            No improvement (1.7872), counter 2/5
    Epoch [7/50], Train Losses: mse: 10.6032, mae: 1.8264, huber: 1.4318, swd: 5.5952, target_std: 20.3272
    Epoch [7/50], Val Losses: mse: 13.1182, mae: 2.1796, huber: 1.7889, swd: 6.8472, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.9211, mae: 1.9386, huber: 1.5310, swd: 4.9309, target_std: 18.4467
      Epoch 7 composite train-obj: 1.431838
            No improvement (1.7889), counter 3/5
    Epoch [8/50], Train Losses: mse: 10.5144, mae: 1.8162, huber: 1.4223, swd: 5.5516, target_std: 20.3255
    Epoch [8/50], Val Losses: mse: 13.4061, mae: 2.2056, huber: 1.8151, swd: 6.9701, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 10.3251, mae: 1.9755, huber: 1.5680, swd: 5.2436, target_std: 18.4467
      Epoch 8 composite train-obj: 1.422280
            No improvement (1.8151), counter 4/5
    Epoch [9/50], Train Losses: mse: 10.4526, mae: 1.8116, huber: 1.4181, swd: 5.5164, target_std: 20.3272
    Epoch [9/50], Val Losses: mse: 13.1802, mae: 2.1985, huber: 1.8098, swd: 6.8128, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.9435, mae: 1.9514, huber: 1.5439, swd: 4.9962, target_std: 18.4467
      Epoch 9 composite train-obj: 1.418074
    Epoch [9/50], Test Losses: mse: 10.0219, mae: 1.9687, huber: 1.5596, swd: 5.0671, target_std: 18.4467
    Best round's Test MSE: 10.0219, MAE: 1.9687, SWD: 5.0672
    Best round's Validation MSE: 12.9007, MAE: 2.1626
    Best round's Test verification MSE : 10.0219, MAE: 1.9687, SWD: 5.0671
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.8698, mae: 2.8071, huber: 2.3933, swd: 22.8207, target_std: 20.3271
    Epoch [1/50], Val Losses: mse: 15.0001, mae: 2.3859, huber: 1.9846, swd: 7.3174, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 11.5377, mae: 2.1451, huber: 1.7284, swd: 5.4209, target_std: 18.4467
      Epoch 1 composite train-obj: 2.393314
            Val objective improved inf → 1.9846, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.1366, mae: 2.0035, huber: 1.5998, swd: 6.2682, target_std: 20.3275
    Epoch [2/50], Val Losses: mse: 13.9834, mae: 2.2756, huber: 1.8818, swd: 7.0440, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 10.8126, mae: 2.0533, huber: 1.6398, swd: 5.2484, target_std: 18.4467
      Epoch 2 composite train-obj: 1.599817
            Val objective improved 1.9846 → 1.8818, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.5271, mae: 1.9362, huber: 1.5352, swd: 5.9689, target_std: 20.3275
    Epoch [3/50], Val Losses: mse: 13.2589, mae: 2.2187, huber: 1.8237, swd: 6.8014, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.8946, mae: 1.9755, huber: 1.5648, swd: 4.7029, target_std: 18.4467
      Epoch 3 composite train-obj: 1.535210
            Val objective improved 1.8818 → 1.8237, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.1976, mae: 1.8991, huber: 1.4998, swd: 5.7886, target_std: 20.3270
    Epoch [4/50], Val Losses: mse: 13.5984, mae: 2.2448, huber: 1.8495, swd: 6.7868, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 10.6032, mae: 2.0166, huber: 1.6071, swd: 5.2092, target_std: 18.4467
      Epoch 4 composite train-obj: 1.499844
            No improvement (1.8495), counter 1/5
    Epoch [5/50], Train Losses: mse: 10.9338, mae: 1.8698, huber: 1.4721, swd: 5.6169, target_std: 20.3277
    Epoch [5/50], Val Losses: mse: 13.3070, mae: 2.2100, huber: 1.8169, swd: 6.6699, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 10.4150, mae: 1.9923, huber: 1.5826, swd: 5.1600, target_std: 18.4467
      Epoch 5 composite train-obj: 1.472065
            Val objective improved 1.8237 → 1.8169, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.7310, mae: 1.8445, huber: 1.4483, swd: 5.4972, target_std: 20.3261
    Epoch [6/50], Val Losses: mse: 13.3418, mae: 2.2179, huber: 1.8258, swd: 6.5945, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 10.1326, mae: 1.9724, huber: 1.5638, swd: 4.8937, target_std: 18.4467
      Epoch 6 composite train-obj: 1.448344
            No improvement (1.8258), counter 1/5
    Epoch [7/50], Train Losses: mse: 10.6507, mae: 1.8358, huber: 1.4404, swd: 5.4560, target_std: 20.3262
    Epoch [7/50], Val Losses: mse: 13.0493, mae: 2.1873, huber: 1.7963, swd: 6.4635, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.9020, mae: 1.9388, huber: 1.5320, swd: 4.7410, target_std: 18.4467
      Epoch 7 composite train-obj: 1.440421
            Val objective improved 1.8169 → 1.7963, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.6285, mae: 1.8369, huber: 1.4418, swd: 5.4640, target_std: 20.3267
    Epoch [8/50], Val Losses: mse: 13.4218, mae: 2.2259, huber: 1.8352, swd: 6.6800, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 10.0693, mae: 1.9495, huber: 1.5434, swd: 4.8438, target_std: 18.4467
      Epoch 8 composite train-obj: 1.441804
            No improvement (1.8352), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.4873, mae: 1.8140, huber: 1.4201, swd: 5.3693, target_std: 20.3276
    Epoch [9/50], Val Losses: mse: 13.5340, mae: 2.2237, huber: 1.8331, swd: 6.8540, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 10.2978, mae: 1.9775, huber: 1.5685, swd: 5.0872, target_std: 18.4467
      Epoch 9 composite train-obj: 1.420066
            No improvement (1.8331), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.4021, mae: 1.8062, huber: 1.4128, swd: 5.3174, target_std: 20.3263
    Epoch [10/50], Val Losses: mse: 14.0213, mae: 2.2692, huber: 1.8782, swd: 7.0370, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 10.5932, mae: 2.0207, huber: 1.6115, swd: 5.2059, target_std: 18.4467
      Epoch 10 composite train-obj: 1.412756
            No improvement (1.8782), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.3022, mae: 1.7948, huber: 1.4018, swd: 5.2549, target_std: 20.3263
    Epoch [11/50], Val Losses: mse: 13.5370, mae: 2.2149, huber: 1.8247, swd: 6.8217, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.8687, mae: 1.9372, huber: 1.5305, swd: 4.6429, target_std: 18.4467
      Epoch 11 composite train-obj: 1.401847
            No improvement (1.8247), counter 4/5
    Epoch [12/50], Train Losses: mse: 10.2011, mae: 1.7832, huber: 1.3909, swd: 5.1847, target_std: 20.3280
    Epoch [12/50], Val Losses: mse: 13.2145, mae: 2.2050, huber: 1.8148, swd: 6.6387, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 9.7672, mae: 1.9420, huber: 1.5331, swd: 4.7001, target_std: 18.4467
      Epoch 12 composite train-obj: 1.390861
    Epoch [12/50], Test Losses: mse: 9.9020, mae: 1.9388, huber: 1.5320, swd: 4.7410, target_std: 18.4467
    Best round's Test MSE: 9.9020, MAE: 1.9388, SWD: 4.7410
    Best round's Validation MSE: 13.0493, MAE: 2.1873
    Best round's Test verification MSE : 9.9020, MAE: 1.9388, SWD: 4.7410
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.5144, mae: 2.7932, huber: 2.3792, swd: 21.5470, target_std: 20.3266
    Epoch [1/50], Val Losses: mse: 14.0932, mae: 2.3191, huber: 1.9196, swd: 6.4871, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.4540, mae: 2.0686, huber: 1.6524, swd: 4.4037, target_std: 18.4467
      Epoch 1 composite train-obj: 2.379221
            Val objective improved inf → 1.9196, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.7370, mae: 1.9682, huber: 1.5655, swd: 5.6349, target_std: 20.3264
    Epoch [2/50], Val Losses: mse: 12.9916, mae: 2.1941, huber: 1.7983, swd: 6.1266, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.9807, mae: 2.0037, huber: 1.5876, swd: 4.4160, target_std: 18.4467
      Epoch 2 composite train-obj: 1.565480
            Val objective improved 1.9196 → 1.7983, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.2637, mae: 1.9145, huber: 1.5143, swd: 5.4108, target_std: 20.3271
    Epoch [3/50], Val Losses: mse: 13.3207, mae: 2.2077, huber: 1.8127, swd: 6.2817, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 10.0173, mae: 1.9813, huber: 1.5682, swd: 4.3524, target_std: 18.4467
      Epoch 3 composite train-obj: 1.514279
            No improvement (1.8127), counter 1/5
    Epoch [4/50], Train Losses: mse: 10.9318, mae: 1.8666, huber: 1.4687, swd: 5.2483, target_std: 20.3277
    Epoch [4/50], Val Losses: mse: 12.9397, mae: 2.1775, huber: 1.7840, swd: 6.0697, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.9427, mae: 1.9494, huber: 1.5416, swd: 4.4533, target_std: 18.4467
      Epoch 4 composite train-obj: 1.468731
            Val objective improved 1.7983 → 1.7840, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.8267, mae: 1.8583, huber: 1.4610, swd: 5.1887, target_std: 20.3280
    Epoch [5/50], Val Losses: mse: 13.0310, mae: 2.1765, huber: 1.7845, swd: 6.1698, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.7532, mae: 1.9348, huber: 1.5275, swd: 4.3663, target_std: 18.4467
      Epoch 5 composite train-obj: 1.460977
            No improvement (1.7845), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.7148, mae: 1.8433, huber: 1.4469, swd: 5.1261, target_std: 20.3274
    Epoch [6/50], Val Losses: mse: 13.0403, mae: 2.1790, huber: 1.7860, swd: 6.1046, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 10.0334, mae: 1.9621, huber: 1.5518, swd: 4.5208, target_std: 18.4467
      Epoch 6 composite train-obj: 1.446903
            No improvement (1.7860), counter 2/5
    Epoch [7/50], Train Losses: mse: 10.6310, mae: 1.8311, huber: 1.4360, swd: 5.0872, target_std: 20.3273
    Epoch [7/50], Val Losses: mse: 12.9786, mae: 2.1723, huber: 1.7813, swd: 6.0913, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.8832, mae: 1.9323, huber: 1.5253, swd: 4.4810, target_std: 18.4467
      Epoch 7 composite train-obj: 1.435961
            Val objective improved 1.7840 → 1.7813, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.4904, mae: 1.8148, huber: 1.4206, swd: 5.0019, target_std: 20.3282
    Epoch [8/50], Val Losses: mse: 13.4208, mae: 2.1985, huber: 1.8096, swd: 6.3468, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.9610, mae: 1.9425, huber: 1.5362, swd: 4.4538, target_std: 18.4467
      Epoch 8 composite train-obj: 1.420580
            No improvement (1.8096), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.4480, mae: 1.8097, huber: 1.4161, swd: 4.9675, target_std: 20.3248
    Epoch [9/50], Val Losses: mse: 13.2726, mae: 2.2166, huber: 1.8245, swd: 6.4063, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 10.0035, mae: 2.0035, huber: 1.5892, swd: 4.6642, target_std: 18.4467
      Epoch 9 composite train-obj: 1.416116
            No improvement (1.8245), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.3320, mae: 1.7990, huber: 1.4058, swd: 4.9053, target_std: 20.3277
    Epoch [10/50], Val Losses: mse: 13.7284, mae: 2.2271, huber: 1.8377, swd: 6.5906, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 10.5575, mae: 1.9794, huber: 1.5729, swd: 4.9865, target_std: 18.4467
      Epoch 10 composite train-obj: 1.405789
            No improvement (1.8377), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.2077, mae: 1.7802, huber: 1.3880, swd: 4.8267, target_std: 20.3265
    Epoch [11/50], Val Losses: mse: 13.1978, mae: 2.2026, huber: 1.8131, swd: 6.3120, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.5503, mae: 1.9236, huber: 1.5164, swd: 4.2771, target_std: 18.4467
      Epoch 11 composite train-obj: 1.388009
            No improvement (1.8131), counter 4/5
    Epoch [12/50], Train Losses: mse: 10.1508, mae: 1.7766, huber: 1.3846, swd: 4.8034, target_std: 20.3277
    Epoch [12/50], Val Losses: mse: 13.3335, mae: 2.1991, huber: 1.8100, swd: 6.3967, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 9.7167, mae: 1.9378, huber: 1.5284, swd: 4.3709, target_std: 18.4467
      Epoch 12 composite train-obj: 1.384646
    Epoch [12/50], Test Losses: mse: 9.8831, mae: 1.9323, huber: 1.5254, swd: 4.4811, target_std: 18.4467
    Best round's Test MSE: 9.8832, MAE: 1.9323, SWD: 4.4810
    Best round's Validation MSE: 12.9786, MAE: 2.1723
    Best round's Test verification MSE : 9.8831, MAE: 1.9323, SWD: 4.4811
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred96_20250430_1834)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 9.9357 ± 0.0615
      mae: 1.9466 ± 0.0158
      huber: 1.5390 ± 0.0148
      swd: 4.7631 ± 0.2398
      target_std: 18.4467 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 12.9762 ± 0.0607
      mae: 2.1741 ± 0.0101
      huber: 1.7836 ± 0.0096
      swd: 6.4755 ± 0.3187
      target_std: 20.5171 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm2_seq96_pred96_20250430_1834
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128,
    ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=False, 
    ablate_shift_inside_scale=False,
    householder_reflects_latent=4,
    householder_reflects_data=8,
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, False]  
cfg.x_to_z_delay.spectral_flags_scale_shift = [False, False]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, True]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, False]
cfg.x_to_z_deri.spectral_flags_scale_shift = [False, False]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, True]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, False]
cfg.z_to_x_main.spectral_flags_scale_shift = [False, False]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, True]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, False]
cfg.z_to_y_main.spectral_flags_scale_shift = [False, False]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, True]
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 380
    Validation Batches: 53
    Test Batches: 108
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 31.0808, mae: 2.7461, huber: 2.3323, swd: 22.9196, target_std: 20.3272
    Epoch [1/50], Val Losses: mse: 14.0093, mae: 2.3149, huber: 1.9140, swd: 6.9430, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.8275, mae: 2.0918, huber: 1.6757, swd: 5.1668, target_std: 18.4467
      Epoch 1 composite train-obj: 2.332289
            Val objective improved inf → 1.9140, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.7566, mae: 1.9676, huber: 1.5646, swd: 6.2688, target_std: 20.3262
    Epoch [2/50], Val Losses: mse: 13.8587, mae: 2.2787, huber: 1.8807, swd: 7.1804, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 11.0532, mae: 2.0847, huber: 1.6708, swd: 5.6021, target_std: 18.4467
      Epoch 2 composite train-obj: 1.564565
            Val objective improved 1.9140 → 1.8807, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.2309, mae: 1.9070, huber: 1.5073, swd: 5.9582, target_std: 20.3268
    Epoch [3/50], Val Losses: mse: 12.8534, mae: 2.1760, huber: 1.7815, swd: 6.7133, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 10.1180, mae: 1.9853, huber: 1.5747, swd: 5.0685, target_std: 18.4467
      Epoch 3 composite train-obj: 1.507316
            Val objective improved 1.8807 → 1.7815, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 10.9805, mae: 1.8783, huber: 1.4802, swd: 5.8314, target_std: 20.3276
    Epoch [4/50], Val Losses: mse: 12.7140, mae: 2.1866, huber: 1.7910, swd: 6.8028, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.9335, mae: 2.0087, huber: 1.5935, swd: 5.0686, target_std: 18.4467
      Epoch 4 composite train-obj: 1.480159
            No improvement (1.7910), counter 1/5
    Epoch [5/50], Train Losses: mse: 10.7558, mae: 1.8509, huber: 1.4544, swd: 5.6852, target_std: 20.3278
    Epoch [5/50], Val Losses: mse: 12.6207, mae: 2.1487, huber: 1.7544, swd: 6.5429, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.6691, mae: 1.9232, huber: 1.5158, swd: 4.7154, target_std: 18.4467
      Epoch 5 composite train-obj: 1.454404
            Val objective improved 1.7815 → 1.7544, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.6517, mae: 1.8359, huber: 1.4402, swd: 5.6229, target_std: 20.3268
    Epoch [6/50], Val Losses: mse: 12.6957, mae: 2.1591, huber: 1.7661, swd: 6.6349, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.8550, mae: 1.9595, huber: 1.5493, swd: 4.9004, target_std: 18.4467
      Epoch 6 composite train-obj: 1.440247
            No improvement (1.7661), counter 1/5
    Epoch [7/50], Train Losses: mse: 10.5706, mae: 1.8298, huber: 1.4350, swd: 5.5778, target_std: 20.3274
    Epoch [7/50], Val Losses: mse: 12.5496, mae: 2.1329, huber: 1.7404, swd: 6.6092, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.7333, mae: 1.9338, huber: 1.5239, swd: 4.8904, target_std: 18.4467
      Epoch 7 composite train-obj: 1.435017
            Val objective improved 1.7544 → 1.7404, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.4007, mae: 1.8059, huber: 1.4122, swd: 5.4611, target_std: 20.3269
    Epoch [8/50], Val Losses: mse: 12.6526, mae: 2.1557, huber: 1.7648, swd: 6.5371, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.8831, mae: 1.9449, huber: 1.5377, swd: 4.9363, target_std: 18.4467
      Epoch 8 composite train-obj: 1.412229
            No improvement (1.7648), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.3410, mae: 1.8013, huber: 1.4080, swd: 5.4363, target_std: 20.3279
    Epoch [9/50], Val Losses: mse: 13.1423, mae: 2.1771, huber: 1.7863, swd: 7.0448, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.9303, mae: 1.9376, huber: 1.5311, swd: 5.0022, target_std: 18.4467
      Epoch 9 composite train-obj: 1.407990
            No improvement (1.7863), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.2403, mae: 1.7871, huber: 1.3947, swd: 5.3712, target_std: 20.3264
    Epoch [10/50], Val Losses: mse: 12.7135, mae: 2.1435, huber: 1.7542, swd: 6.6428, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.9113, mae: 1.9493, huber: 1.5408, swd: 4.9926, target_std: 18.4467
      Epoch 10 composite train-obj: 1.394668
            No improvement (1.7542), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.1557, mae: 1.7770, huber: 1.3850, swd: 5.3147, target_std: 20.3270
    Epoch [11/50], Val Losses: mse: 13.4672, mae: 2.2118, huber: 1.8201, swd: 7.1834, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 10.7898, mae: 2.0133, huber: 1.6050, swd: 5.7506, target_std: 18.4467
      Epoch 11 composite train-obj: 1.385029
            No improvement (1.8201), counter 4/5
    Epoch [12/50], Train Losses: mse: 10.1467, mae: 1.7810, huber: 1.3887, swd: 5.3320, target_std: 20.3264
    Epoch [12/50], Val Losses: mse: 12.8636, mae: 2.1527, huber: 1.7628, swd: 6.6900, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 10.1212, mae: 1.9426, huber: 1.5373, swd: 5.1111, target_std: 18.4467
      Epoch 12 composite train-obj: 1.388662
    Epoch [12/50], Test Losses: mse: 9.7330, mae: 1.9337, huber: 1.5239, swd: 4.8903, target_std: 18.4467
    Best round's Test MSE: 9.7333, MAE: 1.9338, SWD: 4.8904
    Best round's Validation MSE: 12.5496, MAE: 2.1329
    Best round's Test verification MSE : 9.7330, MAE: 1.9337, SWD: 4.8903
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 33.9107, mae: 2.8199, huber: 2.4064, swd: 24.2256, target_std: 20.3276
    Epoch [1/50], Val Losses: mse: 14.2211, mae: 2.3128, huber: 1.9120, swd: 7.2514, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.5701, mae: 2.0860, huber: 1.6661, swd: 5.0248, target_std: 18.4467
      Epoch 1 composite train-obj: 2.406359
            Val objective improved inf → 1.9120, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.0500, mae: 2.0005, huber: 1.5962, swd: 6.1959, target_std: 20.3263
    Epoch [2/50], Val Losses: mse: 14.2807, mae: 2.3122, huber: 1.9142, swd: 7.3228, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 10.6727, mae: 2.0563, huber: 1.6409, swd: 5.1781, target_std: 18.4467
      Epoch 2 composite train-obj: 1.596223
            No improvement (1.9142), counter 1/5
    Epoch [3/50], Train Losses: mse: 11.5913, mae: 1.9472, huber: 1.5456, swd: 5.9927, target_std: 20.3263
    Epoch [3/50], Val Losses: mse: 14.7229, mae: 2.3460, huber: 1.9469, swd: 7.6199, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 11.1153, mae: 2.0778, huber: 1.6640, swd: 5.5009, target_std: 18.4467
      Epoch 3 composite train-obj: 1.545646
            No improvement (1.9469), counter 2/5
    Epoch [4/50], Train Losses: mse: 11.2016, mae: 1.9042, huber: 1.5043, swd: 5.7787, target_std: 20.3280
    Epoch [4/50], Val Losses: mse: 12.8854, mae: 2.1703, huber: 1.7769, swd: 6.4537, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.7845, mae: 1.9585, huber: 1.5467, swd: 4.7153, target_std: 18.4467
      Epoch 4 composite train-obj: 1.504349
            Val objective improved 1.9120 → 1.7769, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.9754, mae: 1.8787, huber: 1.4802, swd: 5.6607, target_std: 20.3278
    Epoch [5/50], Val Losses: mse: 13.0092, mae: 2.1991, huber: 1.8038, swd: 6.4485, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 10.0478, mae: 1.9785, huber: 1.5672, swd: 4.9052, target_std: 18.4467
      Epoch 5 composite train-obj: 1.480220
            No improvement (1.8038), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.6881, mae: 1.8455, huber: 1.4487, swd: 5.4474, target_std: 20.3279
    Epoch [6/50], Val Losses: mse: 13.4048, mae: 2.2133, huber: 1.8196, swd: 6.7726, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 10.1137, mae: 1.9606, huber: 1.5522, swd: 4.9284, target_std: 18.4467
      Epoch 6 composite train-obj: 1.448661
            No improvement (1.8196), counter 2/5
    Epoch [7/50], Train Losses: mse: 10.6351, mae: 1.8395, huber: 1.4433, swd: 5.4167, target_std: 20.3267
    Epoch [7/50], Val Losses: mse: 13.1282, mae: 2.2133, huber: 1.8191, swd: 6.5072, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.9933, mae: 1.9755, huber: 1.5634, swd: 4.7722, target_std: 18.4467
      Epoch 7 composite train-obj: 1.443300
            No improvement (1.8191), counter 3/5
    Epoch [8/50], Train Losses: mse: 10.5293, mae: 1.8289, huber: 1.4334, swd: 5.3698, target_std: 20.3267
    Epoch [8/50], Val Losses: mse: 13.5685, mae: 2.2157, huber: 1.8215, swd: 6.9556, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 10.1543, mae: 1.9799, huber: 1.5667, swd: 5.0044, target_std: 18.4467
      Epoch 8 composite train-obj: 1.433392
            No improvement (1.8215), counter 4/5
    Epoch [9/50], Train Losses: mse: 10.4288, mae: 1.8175, huber: 1.4223, swd: 5.3064, target_std: 20.3275
    Epoch [9/50], Val Losses: mse: 12.9959, mae: 2.1812, huber: 1.7896, swd: 6.5437, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.8420, mae: 1.9399, huber: 1.5318, swd: 4.6923, target_std: 18.4467
      Epoch 9 composite train-obj: 1.422264
    Epoch [9/50], Test Losses: mse: 9.7845, mae: 1.9584, huber: 1.5467, swd: 4.7152, target_std: 18.4467
    Best round's Test MSE: 9.7845, MAE: 1.9585, SWD: 4.7153
    Best round's Validation MSE: 12.8854, MAE: 2.1703
    Best round's Test verification MSE : 9.7845, MAE: 1.9584, SWD: 4.7152
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 33.5951, mae: 2.8221, huber: 2.4087, swd: 22.1707, target_std: 20.3248
    Epoch [1/50], Val Losses: mse: 14.2772, mae: 2.3281, huber: 1.9297, swd: 6.8440, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.5775, mae: 2.0853, huber: 1.6690, swd: 4.7025, target_std: 18.4467
      Epoch 1 composite train-obj: 2.408726
            Val objective improved inf → 1.9297, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.7390, mae: 1.9744, huber: 1.5710, swd: 5.5739, target_std: 20.3277
    Epoch [2/50], Val Losses: mse: 13.7943, mae: 2.2718, huber: 1.8755, swd: 6.4006, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 10.8920, mae: 2.0697, huber: 1.6558, swd: 4.9436, target_std: 18.4467
      Epoch 2 composite train-obj: 1.570966
            Val objective improved 1.9297 → 1.8755, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.2964, mae: 1.9174, huber: 1.5172, swd: 5.4095, target_std: 20.3265
    Epoch [3/50], Val Losses: mse: 13.1594, mae: 2.2084, huber: 1.8151, swd: 6.4274, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.7742, mae: 1.9804, huber: 1.5669, swd: 4.3552, target_std: 18.4467
      Epoch 3 composite train-obj: 1.517181
            Val objective improved 1.8755 → 1.8151, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.0708, mae: 1.8884, huber: 1.4900, swd: 5.3093, target_std: 20.3277
    Epoch [4/50], Val Losses: mse: 13.1765, mae: 2.1972, huber: 1.8056, swd: 6.2210, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.9094, mae: 1.9674, huber: 1.5569, swd: 4.4193, target_std: 18.4467
      Epoch 4 composite train-obj: 1.490009
            Val objective improved 1.8151 → 1.8056, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.8801, mae: 1.8644, huber: 1.4673, swd: 5.2083, target_std: 20.3278
    Epoch [5/50], Val Losses: mse: 13.1006, mae: 2.2034, huber: 1.8086, swd: 5.9039, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 10.0031, mae: 1.9735, huber: 1.5623, swd: 4.4171, target_std: 18.4467
      Epoch 5 composite train-obj: 1.467294
            No improvement (1.8086), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.6649, mae: 1.8364, huber: 1.4406, swd: 5.0799, target_std: 20.3283
    Epoch [6/50], Val Losses: mse: 13.2936, mae: 2.1989, huber: 1.8066, swd: 6.0008, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.8479, mae: 1.9340, huber: 1.5270, swd: 4.3546, target_std: 18.4467
      Epoch 6 composite train-obj: 1.440602
            No improvement (1.8066), counter 2/5
    Epoch [7/50], Train Losses: mse: 10.6341, mae: 1.8372, huber: 1.4415, swd: 5.0749, target_std: 20.3262
    Epoch [7/50], Val Losses: mse: 13.1850, mae: 2.1968, huber: 1.8037, swd: 5.9471, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.5820, mae: 1.9329, huber: 1.5232, swd: 4.2035, target_std: 18.4467
      Epoch 7 composite train-obj: 1.441550
            Val objective improved 1.8056 → 1.8037, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.4773, mae: 1.8163, huber: 1.4214, swd: 4.9863, target_std: 20.3261
    Epoch [8/50], Val Losses: mse: 13.2449, mae: 2.2118, huber: 1.8193, swd: 5.9756, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.9712, mae: 1.9527, huber: 1.5457, swd: 4.4902, target_std: 18.4467
      Epoch 8 composite train-obj: 1.421392
            No improvement (1.8193), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.4238, mae: 1.8125, huber: 1.4182, swd: 4.9550, target_std: 20.3278
    Epoch [9/50], Val Losses: mse: 13.3904, mae: 2.2099, huber: 1.8191, swd: 6.1515, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.8895, mae: 1.9372, huber: 1.5306, swd: 4.4710, target_std: 18.4467
      Epoch 9 composite train-obj: 1.418154
            No improvement (1.8191), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.5572, mae: 1.8241, huber: 1.4295, swd: 5.0429, target_std: 20.3277
    Epoch [10/50], Val Losses: mse: 14.1646, mae: 2.2718, huber: 1.8803, swd: 6.6844, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 10.8003, mae: 2.0193, huber: 1.6119, swd: 4.9605, target_std: 18.4467
      Epoch 10 composite train-obj: 1.429508
            No improvement (1.8803), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.2981, mae: 1.7954, huber: 1.4021, swd: 4.8837, target_std: 20.3272
    Epoch [11/50], Val Losses: mse: 13.1703, mae: 2.1794, huber: 1.7892, swd: 5.9735, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.5653, mae: 1.9107, huber: 1.5036, swd: 4.2329, target_std: 18.4467
      Epoch 11 composite train-obj: 1.402054
            Val objective improved 1.8037 → 1.7892, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 10.2636, mae: 1.7903, huber: 1.3973, swd: 4.8613, target_std: 20.3281
    Epoch [12/50], Val Losses: mse: 13.3162, mae: 2.2057, huber: 1.8164, swd: 6.0727, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 10.2569, mae: 1.9778, huber: 1.5706, swd: 4.6985, target_std: 18.4467
      Epoch 12 composite train-obj: 1.397286
            No improvement (1.8164), counter 1/5
    Epoch [13/50], Train Losses: mse: 10.2112, mae: 1.7855, huber: 1.3928, swd: 4.8365, target_std: 20.3279
    Epoch [13/50], Val Losses: mse: 13.5279, mae: 2.2213, huber: 1.8312, swd: 6.2420, target_std: 20.5171
    Epoch [13/50], Test Losses: mse: 10.4490, mae: 1.9717, huber: 1.5635, swd: 4.8591, target_std: 18.4467
      Epoch 13 composite train-obj: 1.392805
            No improvement (1.8312), counter 2/5
    Epoch [14/50], Train Losses: mse: 10.1529, mae: 1.7798, huber: 1.3872, swd: 4.8137, target_std: 20.3268
    Epoch [14/50], Val Losses: mse: 14.1978, mae: 2.2772, huber: 1.8866, swd: 6.6589, target_std: 20.5171
    Epoch [14/50], Test Losses: mse: 10.5316, mae: 1.9956, huber: 1.5886, swd: 4.8683, target_std: 18.4467
      Epoch 14 composite train-obj: 1.387166
            No improvement (1.8866), counter 3/5
    Epoch [15/50], Train Losses: mse: 10.0971, mae: 1.7735, huber: 1.3810, swd: 4.7776, target_std: 20.3263
    Epoch [15/50], Val Losses: mse: 14.5730, mae: 2.3173, huber: 1.9242, swd: 7.0841, target_std: 20.5171
    Epoch [15/50], Test Losses: mse: 10.7832, mae: 2.0314, huber: 1.6218, swd: 5.1322, target_std: 18.4467
      Epoch 15 composite train-obj: 1.381011
            No improvement (1.9242), counter 4/5
    Epoch [16/50], Train Losses: mse: 9.9684, mae: 1.7548, huber: 1.3637, swd: 4.6928, target_std: 20.3276
    Epoch [16/50], Val Losses: mse: 13.5087, mae: 2.2171, huber: 1.8261, swd: 6.3357, target_std: 20.5171
    Epoch [16/50], Test Losses: mse: 10.0952, mae: 1.9577, huber: 1.5480, swd: 4.6220, target_std: 18.4467
      Epoch 16 composite train-obj: 1.363651
    Epoch [16/50], Test Losses: mse: 9.5653, mae: 1.9107, huber: 1.5036, swd: 4.2329, target_std: 18.4467
    Best round's Test MSE: 9.5653, MAE: 1.9107, SWD: 4.2329
    Best round's Validation MSE: 13.1703, MAE: 2.1794
    Best round's Test verification MSE : 9.5653, MAE: 1.9107, SWD: 4.2329
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred96_20250501_2101)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 9.6944 ± 0.0936
      mae: 1.9343 ± 0.0195
      huber: 1.5247 ± 0.0176
      swd: 4.6129 ± 0.2780
      target_std: 18.4467 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 12.8684 ± 0.2537
      mae: 2.1608 ± 0.0201
      huber: 1.7688 ± 0.0207
      swd: 6.3455 ± 0.2706
      target_std: 20.5171 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm2_seq96_pred96_20250501_2101
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    


```python
importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
    ablate_shift_inside_scale=True,
    householder_reflects_latent = 4,
    householder_reflects_data = 8,
    mixing_strategy='delay_only',
    # single_magnitude_for_shift=True,

)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 380
    Validation Batches: 53
    Test Batches: 108
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 33.1488, mae: 2.8008, huber: 2.3866, swd: 24.0352, target_std: 20.3266
    Epoch [1/50], Val Losses: mse: 14.0162, mae: 2.3118, huber: 1.9106, swd: 6.9357, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.6645, mae: 2.0811, huber: 1.6637, swd: 5.1143, target_std: 18.4467
      Epoch 1 composite train-obj: 2.386561
            Val objective improved inf → 1.9106, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.9428, mae: 1.9930, huber: 1.5892, swd: 6.4188, target_std: 20.3267
    Epoch [2/50], Val Losses: mse: 13.3337, mae: 2.2409, huber: 1.8442, swd: 6.7039, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 10.7222, mae: 2.0540, huber: 1.6398, swd: 5.4268, target_std: 18.4467
      Epoch 2 composite train-obj: 1.589205
            Val objective improved 1.9106 → 1.8442, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.3335, mae: 1.9240, huber: 1.5232, swd: 6.0613, target_std: 20.3261
    Epoch [3/50], Val Losses: mse: 13.1047, mae: 2.2085, huber: 1.8134, swd: 6.8908, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 10.1224, mae: 1.9841, huber: 1.5724, swd: 5.0383, target_std: 18.4467
      Epoch 3 composite train-obj: 1.523162
            Val objective improved 1.8442 → 1.8134, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 10.9506, mae: 1.8719, huber: 1.4735, swd: 5.8167, target_std: 20.3270
    Epoch [4/50], Val Losses: mse: 12.6944, mae: 2.1514, huber: 1.7586, swd: 6.7413, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 10.0251, mae: 1.9771, huber: 1.5647, swd: 5.1286, target_std: 18.4467
      Epoch 4 composite train-obj: 1.473460
            Val objective improved 1.8134 → 1.7586, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.7727, mae: 1.8560, huber: 1.4584, swd: 5.7278, target_std: 20.3274
    Epoch [5/50], Val Losses: mse: 13.8495, mae: 2.2471, huber: 1.8539, swd: 7.5154, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 11.0920, mae: 2.0673, huber: 1.6557, swd: 5.8890, target_std: 18.4467
      Epoch 5 composite train-obj: 1.458365
            No improvement (1.8539), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.7267, mae: 1.8507, huber: 1.4538, swd: 5.7289, target_std: 20.3277
    Epoch [6/50], Val Losses: mse: 12.8769, mae: 2.1803, huber: 1.7898, swd: 6.9119, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.6741, mae: 1.9364, huber: 1.5279, swd: 4.8060, target_std: 18.4467
      Epoch 6 composite train-obj: 1.453841
            No improvement (1.7898), counter 2/5
    Epoch [7/50], Train Losses: mse: 10.5091, mae: 1.8215, huber: 1.4264, swd: 5.5600, target_std: 20.3289
    Epoch [7/50], Val Losses: mse: 13.6306, mae: 2.2435, huber: 1.8519, swd: 7.3975, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 10.4757, mae: 2.0014, huber: 1.5923, swd: 5.5108, target_std: 18.4467
      Epoch 7 composite train-obj: 1.426353
            No improvement (1.8519), counter 3/5
    Epoch [8/50], Train Losses: mse: 10.5123, mae: 1.8256, huber: 1.4301, swd: 5.5869, target_std: 20.3268
    Epoch [8/50], Val Losses: mse: 13.1596, mae: 2.2007, huber: 1.8116, swd: 7.0860, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.8946, mae: 1.9768, huber: 1.5664, swd: 5.0975, target_std: 18.4467
      Epoch 8 composite train-obj: 1.430141
            No improvement (1.8116), counter 4/5
    Epoch [9/50], Train Losses: mse: 10.3328, mae: 1.8009, huber: 1.4071, swd: 5.4536, target_std: 20.3270
    Epoch [9/50], Val Losses: mse: 13.8806, mae: 2.2438, huber: 1.8569, swd: 7.4576, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 10.6082, mae: 1.9942, huber: 1.5877, swd: 5.5616, target_std: 18.4467
      Epoch 9 composite train-obj: 1.407119
    Epoch [9/50], Test Losses: mse: 10.0269, mae: 1.9773, huber: 1.5648, swd: 5.1303, target_std: 18.4467
    Best round's Test MSE: 10.0251, MAE: 1.9771, SWD: 5.1286
    Best round's Validation MSE: 12.6944, MAE: 2.1514
    Best round's Test verification MSE : 10.0269, MAE: 1.9773, SWD: 5.1303
    Time taken: 130.60 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 36.2562, mae: 2.8869, huber: 2.4724, swd: 25.7010, target_std: 20.3268
    Epoch [1/50], Val Losses: mse: 14.1103, mae: 2.3235, huber: 1.9214, swd: 6.9367, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.4170, mae: 2.0661, huber: 1.6480, swd: 4.8039, target_std: 18.4467
      Epoch 1 composite train-obj: 2.472429
            Val objective improved inf → 1.9214, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.0733, mae: 2.0001, huber: 1.5962, swd: 6.2266, target_std: 20.3289
    Epoch [2/50], Val Losses: mse: 14.5663, mae: 2.3508, huber: 1.9509, swd: 7.1986, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 11.5125, mae: 2.1335, huber: 1.7184, swd: 5.5810, target_std: 18.4467
      Epoch 2 composite train-obj: 1.596201
            No improvement (1.9509), counter 1/5
    Epoch [3/50], Train Losses: mse: 11.3917, mae: 1.9276, huber: 1.5268, swd: 5.8217, target_std: 20.3256
    Epoch [3/50], Val Losses: mse: 12.9678, mae: 2.2017, huber: 1.8048, swd: 6.4070, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 10.0129, mae: 1.9893, huber: 1.5756, swd: 4.7088, target_std: 18.4467
      Epoch 3 composite train-obj: 1.526769
            Val objective improved 1.9214 → 1.8048, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.0149, mae: 1.8819, huber: 1.4830, swd: 5.6489, target_std: 20.3263
    Epoch [4/50], Val Losses: mse: 12.8454, mae: 2.1955, huber: 1.7996, swd: 6.5281, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 10.0149, mae: 2.0113, huber: 1.5951, swd: 4.8454, target_std: 18.4467
      Epoch 4 composite train-obj: 1.482951
            Val objective improved 1.8048 → 1.7996, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.7815, mae: 1.8546, huber: 1.4569, swd: 5.5087, target_std: 20.3263
    Epoch [5/50], Val Losses: mse: 12.8623, mae: 2.1818, huber: 1.7876, swd: 6.4710, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.7112, mae: 1.9460, huber: 1.5359, swd: 4.5510, target_std: 18.4467
      Epoch 5 composite train-obj: 1.456875
            Val objective improved 1.7996 → 1.7876, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.6814, mae: 1.8417, huber: 1.4450, swd: 5.4713, target_std: 20.3264
    Epoch [6/50], Val Losses: mse: 12.5133, mae: 2.1423, huber: 1.7526, swd: 6.3679, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.5624, mae: 1.9410, huber: 1.5318, swd: 4.6000, target_std: 18.4467
      Epoch 6 composite train-obj: 1.444985
            Val objective improved 1.7876 → 1.7526, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 10.5679, mae: 1.8291, huber: 1.4332, swd: 5.3986, target_std: 20.3273
    Epoch [7/50], Val Losses: mse: 12.8762, mae: 2.1637, huber: 1.7738, swd: 6.5278, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.7581, mae: 1.9218, huber: 1.5162, swd: 4.6340, target_std: 18.4467
      Epoch 7 composite train-obj: 1.433166
            No improvement (1.7738), counter 1/5
    Epoch [8/50], Train Losses: mse: 10.4440, mae: 1.8125, huber: 1.4181, swd: 5.3240, target_std: 20.3268
    Epoch [8/50], Val Losses: mse: 12.5867, mae: 2.1477, huber: 1.7584, swd: 6.3244, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.5316, mae: 1.9318, huber: 1.5217, swd: 4.5210, target_std: 18.4467
      Epoch 8 composite train-obj: 1.418085
            No improvement (1.7584), counter 2/5
    Epoch [9/50], Train Losses: mse: 10.3809, mae: 1.8055, huber: 1.4116, swd: 5.2679, target_std: 20.3266
    Epoch [9/50], Val Losses: mse: 12.8885, mae: 2.1687, huber: 1.7786, swd: 6.4506, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.9813, mae: 1.9490, huber: 1.5426, swd: 4.8121, target_std: 18.4467
      Epoch 9 composite train-obj: 1.411557
            No improvement (1.7786), counter 3/5
    Epoch [10/50], Train Losses: mse: 10.3546, mae: 1.8037, huber: 1.4104, swd: 5.2745, target_std: 20.3278
    Epoch [10/50], Val Losses: mse: 12.9215, mae: 2.1571, huber: 1.7717, swd: 6.4917, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.9220, mae: 1.9401, huber: 1.5350, swd: 4.8342, target_std: 18.4467
      Epoch 10 composite train-obj: 1.410352
            No improvement (1.7717), counter 4/5
    Epoch [11/50], Train Losses: mse: 10.2480, mae: 1.7922, huber: 1.3995, swd: 5.2104, target_std: 20.3272
    Epoch [11/50], Val Losses: mse: 13.2968, mae: 2.2013, huber: 1.8134, swd: 6.6925, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 10.0887, mae: 1.9542, huber: 1.5485, swd: 4.8697, target_std: 18.4467
      Epoch 11 composite train-obj: 1.399523
    Epoch [11/50], Test Losses: mse: 9.5624, mae: 1.9410, huber: 1.5317, swd: 4.6000, target_std: 18.4467
    Best round's Test MSE: 9.5624, MAE: 1.9410, SWD: 4.6000
    Best round's Validation MSE: 12.5133, MAE: 2.1423
    Best round's Test verification MSE : 9.5624, MAE: 1.9410, SWD: 4.6000
    Time taken: 169.56 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 34.7616, mae: 2.8434, huber: 2.4289, swd: 22.9827, target_std: 20.3273
    Epoch [1/50], Val Losses: mse: 14.2766, mae: 2.3317, huber: 1.9287, swd: 6.5708, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.5731, mae: 2.0719, huber: 1.6543, swd: 4.5578, target_std: 18.4467
      Epoch 1 composite train-obj: 2.428906
            Val objective improved inf → 1.9287, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.1898, mae: 1.9995, huber: 1.5955, swd: 5.9637, target_std: 20.3271
    Epoch [2/50], Val Losses: mse: 13.4811, mae: 2.2525, huber: 1.8543, swd: 6.3614, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 10.1049, mae: 2.0221, huber: 1.6061, swd: 4.5031, target_std: 18.4467
      Epoch 2 composite train-obj: 1.595545
            Val objective improved 1.9287 → 1.8543, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.4820, mae: 1.9300, huber: 1.5294, swd: 5.5203, target_std: 20.3271
    Epoch [3/50], Val Losses: mse: 13.1236, mae: 2.2027, huber: 1.8091, swd: 6.2533, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.9966, mae: 1.9706, huber: 1.5605, swd: 4.4478, target_std: 18.4467
      Epoch 3 composite train-obj: 1.529435
            Val objective improved 1.8543 → 1.8091, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.1579, mae: 1.8960, huber: 1.4971, swd: 5.3847, target_std: 20.3278
    Epoch [4/50], Val Losses: mse: 12.9133, mae: 2.1818, huber: 1.7896, swd: 6.1070, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 10.1428, mae: 1.9809, huber: 1.5706, swd: 4.6581, target_std: 18.4467
      Epoch 4 composite train-obj: 1.497074
            Val objective improved 1.8091 → 1.7896, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.8861, mae: 1.8645, huber: 1.4671, swd: 5.2179, target_std: 20.3263
    Epoch [5/50], Val Losses: mse: 13.1356, mae: 2.2065, huber: 1.8138, swd: 6.0699, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 10.4513, mae: 2.0207, huber: 1.6090, swd: 4.6922, target_std: 18.4467
      Epoch 5 composite train-obj: 1.467054
            No improvement (1.8138), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.7886, mae: 1.8549, huber: 1.4579, swd: 5.1746, target_std: 20.3277
    Epoch [6/50], Val Losses: mse: 12.8566, mae: 2.1778, huber: 1.7861, swd: 6.0925, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.9372, mae: 1.9771, huber: 1.5664, swd: 4.5317, target_std: 18.4467
      Epoch 6 composite train-obj: 1.457928
            Val objective improved 1.7896 → 1.7861, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 10.5762, mae: 1.8286, huber: 1.4332, swd: 5.0407, target_std: 20.3270
    Epoch [7/50], Val Losses: mse: 12.9431, mae: 2.1760, huber: 1.7843, swd: 6.1242, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.9379, mae: 1.9424, huber: 1.5357, swd: 4.4571, target_std: 18.4467
      Epoch 7 composite train-obj: 1.433181
            Val objective improved 1.7861 → 1.7843, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.5343, mae: 1.8243, huber: 1.4292, swd: 5.0269, target_std: 20.3265
    Epoch [8/50], Val Losses: mse: 13.8698, mae: 2.2682, huber: 1.8753, swd: 6.8060, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 11.2266, mae: 2.0912, huber: 1.6786, swd: 5.4661, target_std: 18.4467
      Epoch 8 composite train-obj: 1.429220
            No improvement (1.8753), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.4209, mae: 1.8112, huber: 1.4170, swd: 4.9470, target_std: 20.3273
    Epoch [9/50], Val Losses: mse: 13.1437, mae: 2.1773, huber: 1.7882, swd: 6.4314, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.8922, mae: 1.9619, huber: 1.5511, swd: 4.5035, target_std: 18.4467
      Epoch 9 composite train-obj: 1.416962
            No improvement (1.7882), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.3561, mae: 1.8032, huber: 1.4095, swd: 4.9084, target_std: 20.3282
    Epoch [10/50], Val Losses: mse: 13.1490, mae: 2.1812, huber: 1.7938, swd: 6.4662, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.9107, mae: 1.9474, huber: 1.5411, swd: 4.5761, target_std: 18.4467
      Epoch 10 composite train-obj: 1.409473
            No improvement (1.7938), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.4143, mae: 1.8144, huber: 1.4197, swd: 4.9575, target_std: 20.3265
    Epoch [11/50], Val Losses: mse: 13.1561, mae: 2.1934, huber: 1.8066, swd: 6.2313, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.9731, mae: 1.9521, huber: 1.5452, swd: 4.5491, target_std: 18.4467
      Epoch 11 composite train-obj: 1.419726
            No improvement (1.8066), counter 4/5
    Epoch [12/50], Train Losses: mse: 10.2302, mae: 1.7894, huber: 1.3964, swd: 4.8373, target_std: 20.3276
    Epoch [12/50], Val Losses: mse: 14.2671, mae: 2.3077, huber: 1.9175, swd: 6.8835, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 11.0962, mae: 2.0687, huber: 1.6596, swd: 5.2575, target_std: 18.4467
      Epoch 12 composite train-obj: 1.396402
    Epoch [12/50], Test Losses: mse: 9.9378, mae: 1.9424, huber: 1.5357, swd: 4.4570, target_std: 18.4467
    Best round's Test MSE: 9.9379, MAE: 1.9424, SWD: 4.4571
    Best round's Validation MSE: 12.9431, MAE: 2.1760
    Best round's Test verification MSE : 9.9378, MAE: 1.9424, SWD: 4.4570
    Time taken: 188.64 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred96_20250503_1513)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 9.8418 ± 0.2007
      mae: 1.9535 ± 0.0167
      huber: 1.5440 ± 0.0147
      swd: 4.7286 ± 0.2888
      target_std: 18.4467 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 12.7169 ± 0.1762
      mae: 2.1566 ± 0.0142
      huber: 1.7652 ± 0.0138
      swd: 6.4112 ± 0.2538
      target_std: 20.5171 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 488.89 seconds
    
    Experiment complete: ACL_ettm2_seq96_pred96_20250503_1513
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196

##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    global_std.shape: torch.Size([7])
    Global Std for ettm2: tensor([10.2434,  6.0312, 13.0618,  4.3690,  6.1544,  6.0135, 11.8865],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 379
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 379
    Validation Batches: 53
    Test Batches: 107
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 39.2176, mae: 3.1513, huber: 2.7315, swd: 28.0282, ept: 151.9135
    Epoch [1/50], Val Losses: mse: 19.6974, mae: 2.7273, huber: 2.3164, swd: 10.3422, ept: 153.4657
    Epoch [1/50], Test Losses: mse: 12.9790, mae: 2.2831, huber: 1.8622, swd: 6.1008, ept: 161.4444
      Epoch 1 composite train-obj: 2.731548
            Val objective improved inf → 2.3164, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.1140, mae: 2.3288, huber: 1.9184, swd: 9.2350, ept: 168.5852
    Epoch [2/50], Val Losses: mse: 19.5761, mae: 2.7049, huber: 2.2953, swd: 10.1133, ept: 150.9968
    Epoch [2/50], Test Losses: mse: 14.4532, mae: 2.3954, huber: 1.9719, swd: 7.3441, ept: 157.6760
      Epoch 2 composite train-obj: 1.918360
            Val objective improved 2.3164 → 2.2953, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 16.5799, mae: 2.2837, huber: 1.8754, swd: 8.9062, ept: 169.4712
    Epoch [3/50], Val Losses: mse: 18.2024, mae: 2.5904, huber: 2.1846, swd: 9.4634, ept: 154.8770
    Epoch [3/50], Test Losses: mse: 12.9352, mae: 2.2529, huber: 1.8333, swd: 6.2420, ept: 160.8585
      Epoch 3 composite train-obj: 1.875388
            Val objective improved 2.2953 → 2.1846, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 16.0769, mae: 2.2340, huber: 1.8273, swd: 8.5958, ept: 170.3484
    Epoch [4/50], Val Losses: mse: 17.8276, mae: 2.5834, huber: 2.1767, swd: 9.2245, ept: 154.7577
    Epoch [4/50], Test Losses: mse: 13.1562, mae: 2.2671, huber: 1.8489, swd: 6.4221, ept: 158.9686
      Epoch 4 composite train-obj: 1.827258
            Val objective improved 2.1846 → 2.1767, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 15.7802, mae: 2.2062, huber: 1.8004, swd: 8.4306, ept: 170.7974
    Epoch [5/50], Val Losses: mse: 17.3973, mae: 2.5489, huber: 2.1436, swd: 9.0390, ept: 155.3582
    Epoch [5/50], Test Losses: mse: 12.9195, mae: 2.2618, huber: 1.8419, swd: 6.3077, ept: 159.6793
      Epoch 5 composite train-obj: 1.800404
            Val objective improved 2.1767 → 2.1436, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 15.5447, mae: 2.1855, huber: 1.7808, swd: 8.2677, ept: 171.0932
    Epoch [6/50], Val Losses: mse: 17.2306, mae: 2.5488, huber: 2.1467, swd: 8.9574, ept: 155.4534
    Epoch [6/50], Test Losses: mse: 12.6863, mae: 2.2506, huber: 1.8318, swd: 6.0919, ept: 161.9222
      Epoch 6 composite train-obj: 1.780827
            No improvement (2.1467), counter 1/5
    Epoch [7/50], Train Losses: mse: 15.2941, mae: 2.1641, huber: 1.7600, swd: 8.1123, ept: 171.5482
    Epoch [7/50], Val Losses: mse: 17.9005, mae: 2.5813, huber: 2.1812, swd: 9.3100, ept: 153.0206
    Epoch [7/50], Test Losses: mse: 14.4148, mae: 2.3481, huber: 1.9317, swd: 7.5893, ept: 156.7835
      Epoch 7 composite train-obj: 1.760025
            No improvement (2.1812), counter 2/5
    Epoch [8/50], Train Losses: mse: 15.1284, mae: 2.1522, huber: 1.7487, swd: 7.9806, ept: 171.8265
    Epoch [8/50], Val Losses: mse: 17.6512, mae: 2.5576, huber: 2.1588, swd: 9.1923, ept: 154.1341
    Epoch [8/50], Test Losses: mse: 13.3557, mae: 2.2644, huber: 1.8485, swd: 6.6252, ept: 158.6421
      Epoch 8 composite train-obj: 1.748672
            No improvement (2.1588), counter 3/5
    Epoch [9/50], Train Losses: mse: 14.9920, mae: 2.1399, huber: 1.7369, swd: 7.8923, ept: 172.0266
    Epoch [9/50], Val Losses: mse: 17.7339, mae: 2.5766, huber: 2.1759, swd: 9.1821, ept: 153.5356
    Epoch [9/50], Test Losses: mse: 13.4157, mae: 2.2723, huber: 1.8560, swd: 6.6814, ept: 158.7956
      Epoch 9 composite train-obj: 1.736880
            No improvement (2.1759), counter 4/5
    Epoch [10/50], Train Losses: mse: 14.7820, mae: 2.1186, huber: 1.7168, swd: 7.7465, ept: 172.3571
    Epoch [10/50], Val Losses: mse: 17.5611, mae: 2.5692, huber: 2.1690, swd: 9.0914, ept: 154.3035
    Epoch [10/50], Test Losses: mse: 13.2944, mae: 2.2622, huber: 1.8464, swd: 6.6483, ept: 158.4479
      Epoch 10 composite train-obj: 1.716807
    Epoch [10/50], Test Losses: mse: 12.9193, mae: 2.2618, huber: 1.8419, swd: 6.3075, ept: 159.6800
    Best round's Test MSE: 12.9195, MAE: 2.2618, SWD: 6.3077
    Best round's Validation MSE: 17.3973, MAE: 2.5489, SWD: 9.0390
    Best round's Test verification MSE : 12.9193, MAE: 2.2618, SWD: 6.3075
    Time taken: 111.81 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 37.8055, mae: 3.0949, huber: 2.6760, swd: 27.5141, ept: 153.7432
    Epoch [1/50], Val Losses: mse: 19.6045, mae: 2.7107, huber: 2.3000, swd: 10.3056, ept: 153.5894
    Epoch [1/50], Test Losses: mse: 13.2562, mae: 2.3152, huber: 1.8923, swd: 6.3529, ept: 160.9777
      Epoch 1 composite train-obj: 2.675997
            Val objective improved inf → 2.3000, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.0406, mae: 2.3275, huber: 1.9166, swd: 9.5120, ept: 168.8749
    Epoch [2/50], Val Losses: mse: 19.3273, mae: 2.6740, huber: 2.2658, swd: 10.4736, ept: 152.7925
    Epoch [2/50], Test Losses: mse: 14.3868, mae: 2.3954, huber: 1.9721, swd: 7.6564, ept: 158.9874
      Epoch 2 composite train-obj: 1.916642
            Val objective improved 2.3000 → 2.2658, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 16.3101, mae: 2.2550, huber: 1.8472, swd: 9.0307, ept: 170.1215
    Epoch [3/50], Val Losses: mse: 17.9705, mae: 2.5959, huber: 2.1917, swd: 10.0827, ept: 156.3206
    Epoch [3/50], Test Losses: mse: 12.6047, mae: 2.2497, huber: 1.8295, swd: 6.4001, ept: 163.5802
      Epoch 3 composite train-obj: 1.847245
            Val objective improved 2.2658 → 2.1917, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 15.8777, mae: 2.2163, huber: 1.8098, swd: 8.7956, ept: 170.7809
    Epoch [4/50], Val Losses: mse: 20.4412, mae: 2.7752, huber: 2.3683, swd: 11.7122, ept: 148.9453
    Epoch [4/50], Test Losses: mse: 16.6414, mae: 2.5870, huber: 2.1641, swd: 9.8865, ept: 153.1879
      Epoch 4 composite train-obj: 1.809818
            No improvement (2.3683), counter 1/5
    Epoch [5/50], Train Losses: mse: 15.5854, mae: 2.1918, huber: 1.7859, swd: 8.5943, ept: 171.2633
    Epoch [5/50], Val Losses: mse: 17.1365, mae: 2.5343, huber: 2.1293, swd: 9.0764, ept: 156.1680
    Epoch [5/50], Test Losses: mse: 12.7145, mae: 2.2243, huber: 1.8074, swd: 6.3868, ept: 161.1387
      Epoch 5 composite train-obj: 1.785923
            Val objective improved 2.1917 → 2.1293, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 15.2883, mae: 2.1650, huber: 1.7602, swd: 8.3721, ept: 171.6142
    Epoch [6/50], Val Losses: mse: 17.7586, mae: 2.5977, huber: 2.1914, swd: 9.4106, ept: 153.3365
    Epoch [6/50], Test Losses: mse: 13.5991, mae: 2.3018, huber: 1.8838, swd: 7.0706, ept: 157.9538
      Epoch 6 composite train-obj: 1.760178
            No improvement (2.1914), counter 1/5
    Epoch [7/50], Train Losses: mse: 15.0695, mae: 2.1436, huber: 1.7398, swd: 8.2047, ept: 171.9894
    Epoch [7/50], Val Losses: mse: 17.2389, mae: 2.5597, huber: 2.1553, swd: 9.0663, ept: 154.4427
    Epoch [7/50], Test Losses: mse: 12.9643, mae: 2.2357, huber: 1.8206, swd: 6.5455, ept: 159.3424
      Epoch 7 composite train-obj: 1.739824
            No improvement (2.1553), counter 2/5
    Epoch [8/50], Train Losses: mse: 15.0408, mae: 2.1482, huber: 1.7442, swd: 8.2180, ept: 172.0021
    Epoch [8/50], Val Losses: mse: 17.1970, mae: 2.5473, huber: 2.1450, swd: 8.8739, ept: 154.9016
    Epoch [8/50], Test Losses: mse: 13.6328, mae: 2.2948, huber: 1.8777, swd: 7.0335, ept: 159.0305
      Epoch 8 composite train-obj: 1.744232
            No improvement (2.1450), counter 3/5
    Epoch [9/50], Train Losses: mse: 14.7987, mae: 2.1194, huber: 1.7172, swd: 8.0183, ept: 172.4296
    Epoch [9/50], Val Losses: mse: 17.6831, mae: 2.5916, huber: 2.1884, swd: 9.3452, ept: 153.7239
    Epoch [9/50], Test Losses: mse: 13.7564, mae: 2.3125, huber: 1.8924, swd: 7.0862, ept: 157.6881
      Epoch 9 composite train-obj: 1.717203
            No improvement (2.1884), counter 4/5
    Epoch [10/50], Train Losses: mse: 14.8068, mae: 2.1294, huber: 1.7269, swd: 8.0386, ept: 172.3716
    Epoch [10/50], Val Losses: mse: 17.8926, mae: 2.5948, huber: 2.1931, swd: 9.5178, ept: 153.5870
    Epoch [10/50], Test Losses: mse: 14.1764, mae: 2.3292, huber: 1.9133, swd: 7.5258, ept: 156.0064
      Epoch 10 composite train-obj: 1.726853
    Epoch [10/50], Test Losses: mse: 12.7150, mae: 2.2244, huber: 1.8075, swd: 6.3873, ept: 161.1408
    Best round's Test MSE: 12.7145, MAE: 2.2243, SWD: 6.3868
    Best round's Validation MSE: 17.1365, MAE: 2.5343, SWD: 9.0764
    Best round's Test verification MSE : 12.7150, MAE: 2.2244, SWD: 6.3873
    Time taken: 108.65 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 39.3785, mae: 3.1679, huber: 2.7479, swd: 24.4014, ept: 152.4429
    Epoch [1/50], Val Losses: mse: 20.3267, mae: 2.7867, huber: 2.3744, swd: 9.6932, ept: 153.1641
    Epoch [1/50], Test Losses: mse: 13.4902, mae: 2.3589, huber: 1.9339, swd: 5.8038, ept: 162.3366
      Epoch 1 composite train-obj: 2.747878
            Val objective improved inf → 2.3744, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.2302, mae: 2.3374, huber: 1.9260, swd: 8.2016, ept: 168.7696
    Epoch [2/50], Val Losses: mse: 19.5409, mae: 2.7134, huber: 2.3027, swd: 9.4746, ept: 153.5421
    Epoch [2/50], Test Losses: mse: 12.7187, mae: 2.2869, huber: 1.8630, swd: 5.3483, ept: 162.1599
      Epoch 2 composite train-obj: 1.925985
            Val objective improved 2.3744 → 2.3027, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 16.6427, mae: 2.2841, huber: 1.8755, swd: 7.8646, ept: 169.7749
    Epoch [3/50], Val Losses: mse: 18.6967, mae: 2.6263, huber: 2.2200, swd: 8.5449, ept: 153.9043
    Epoch [3/50], Test Losses: mse: 13.5099, mae: 2.2969, huber: 1.8782, swd: 5.8281, ept: 159.8041
      Epoch 3 composite train-obj: 1.875534
            Val objective improved 2.3027 → 2.2200, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 16.0911, mae: 2.2303, huber: 1.8238, swd: 7.5631, ept: 170.5962
    Epoch [4/50], Val Losses: mse: 17.8298, mae: 2.5679, huber: 2.1651, swd: 8.0692, ept: 155.5426
    Epoch [4/50], Test Losses: mse: 12.8863, mae: 2.2353, huber: 1.8185, swd: 5.4550, ept: 161.8452
      Epoch 4 composite train-obj: 1.823835
            Val objective improved 2.2200 → 2.1651, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 15.8628, mae: 2.2137, huber: 1.8080, swd: 7.4683, ept: 170.7948
    Epoch [5/50], Val Losses: mse: 17.6119, mae: 2.5525, huber: 2.1504, swd: 7.9739, ept: 154.9220
    Epoch [5/50], Test Losses: mse: 12.9231, mae: 2.2442, huber: 1.8267, swd: 5.4051, ept: 161.2040
      Epoch 5 composite train-obj: 1.807993
            Val objective improved 2.1651 → 2.1504, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 15.5903, mae: 2.1861, huber: 1.7814, swd: 7.3126, ept: 171.2039
    Epoch [6/50], Val Losses: mse: 18.0188, mae: 2.5774, huber: 2.1761, swd: 8.2112, ept: 153.8470
    Epoch [6/50], Test Losses: mse: 13.6816, mae: 2.3022, huber: 1.8838, swd: 6.0283, ept: 159.2032
      Epoch 6 composite train-obj: 1.781374
            No improvement (2.1761), counter 1/5
    Epoch [7/50], Train Losses: mse: 15.3055, mae: 2.1664, huber: 1.7623, swd: 7.1279, ept: 171.5589
    Epoch [7/50], Val Losses: mse: 17.3790, mae: 2.5599, huber: 2.1583, swd: 7.9523, ept: 156.3125
    Epoch [7/50], Test Losses: mse: 12.5066, mae: 2.2349, huber: 1.8162, swd: 5.2731, ept: 162.9139
      Epoch 7 composite train-obj: 1.762266
            No improvement (2.1583), counter 2/5
    Epoch [8/50], Train Losses: mse: 15.1375, mae: 2.1538, huber: 1.7503, swd: 7.0128, ept: 171.7796
    Epoch [8/50], Val Losses: mse: 17.7263, mae: 2.5735, huber: 2.1750, swd: 8.0683, ept: 153.2979
    Epoch [8/50], Test Losses: mse: 13.0460, mae: 2.2301, huber: 1.8158, swd: 5.4633, ept: 160.3647
      Epoch 8 composite train-obj: 1.750316
            No improvement (2.1750), counter 3/5
    Epoch [9/50], Train Losses: mse: 14.9867, mae: 2.1417, huber: 1.7387, swd: 6.9130, ept: 171.9913
    Epoch [9/50], Val Losses: mse: 17.1232, mae: 2.5335, huber: 2.1350, swd: 7.5747, ept: 154.8721
    Epoch [9/50], Test Losses: mse: 13.2778, mae: 2.2417, huber: 1.8282, swd: 5.7269, ept: 159.7877
      Epoch 9 composite train-obj: 1.738659
            Val objective improved 2.1504 → 2.1350, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 14.8801, mae: 2.1316, huber: 1.7296, swd: 6.8693, ept: 172.1443
    Epoch [10/50], Val Losses: mse: 17.3193, mae: 2.5468, huber: 2.1492, swd: 7.7935, ept: 154.9383
    Epoch [10/50], Test Losses: mse: 13.2848, mae: 2.2358, huber: 1.8219, swd: 5.7251, ept: 159.8495
      Epoch 10 composite train-obj: 1.729634
            No improvement (2.1492), counter 1/5
    Epoch [11/50], Train Losses: mse: 14.7245, mae: 2.1177, huber: 1.7162, swd: 6.7784, ept: 172.3586
    Epoch [11/50], Val Losses: mse: 17.6980, mae: 2.5709, huber: 2.1719, swd: 8.0733, ept: 154.2557
    Epoch [11/50], Test Losses: mse: 13.6613, mae: 2.2807, huber: 1.8642, swd: 6.0293, ept: 158.7967
      Epoch 11 composite train-obj: 1.716247
            No improvement (2.1719), counter 2/5
    Epoch [12/50], Train Losses: mse: 14.6940, mae: 2.1203, huber: 1.7187, swd: 6.7861, ept: 172.4160
    Epoch [12/50], Val Losses: mse: 17.5367, mae: 2.5574, huber: 2.1590, swd: 7.9740, ept: 154.3798
    Epoch [12/50], Test Losses: mse: 12.8999, mae: 2.2200, huber: 1.8049, swd: 5.4593, ept: 160.9774
      Epoch 12 composite train-obj: 1.718683
            No improvement (2.1590), counter 3/5
    Epoch [13/50], Train Losses: mse: 14.5180, mae: 2.0998, huber: 1.6997, swd: 6.6685, ept: 172.6383
    Epoch [13/50], Val Losses: mse: 17.7878, mae: 2.5786, huber: 2.1806, swd: 8.2166, ept: 154.0130
    Epoch [13/50], Test Losses: mse: 13.6830, mae: 2.2921, huber: 1.8740, swd: 6.0568, ept: 158.7811
      Epoch 13 composite train-obj: 1.699683
            No improvement (2.1806), counter 4/5
    Epoch [14/50], Train Losses: mse: 14.4997, mae: 2.0992, huber: 1.6989, swd: 6.6719, ept: 172.6662
    Epoch [14/50], Val Losses: mse: 17.5661, mae: 2.5511, huber: 2.1539, swd: 7.9369, ept: 154.7820
    Epoch [14/50], Test Losses: mse: 13.9960, mae: 2.3079, huber: 1.8906, swd: 6.3434, ept: 158.1165
      Epoch 14 composite train-obj: 1.698871
    Epoch [14/50], Test Losses: mse: 13.2778, mae: 2.2417, huber: 1.8282, swd: 5.7268, ept: 159.7900
    Best round's Test MSE: 13.2778, MAE: 2.2417, SWD: 5.7269
    Best round's Validation MSE: 17.1232, MAE: 2.5335, SWD: 7.5747
    Best round's Test verification MSE : 13.2778, MAE: 2.2417, SWD: 5.7268
    Time taken: 154.64 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred196_20250512_1613)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.9706 ± 0.2328
      mae: 2.2426 ± 0.0153
      huber: 1.8259 ± 0.0142
      swd: 6.1404 ± 0.2942
      ept: 160.2019 ± 0.6639
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 17.2190 ± 0.1262
      mae: 2.5389 ± 0.0071
      huber: 2.1360 ± 0.0059
      swd: 8.5634 ± 0.6993
      ept: 155.4661 ± 0.5345
      count: 53.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 375.20 seconds
    
    Experiment complete: ACL_ettm2_seq96_pred196_20250512_1613
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

##### ab: rotations (8,4)


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128,
    ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=False, 
    ablate_shift_inside_scale=False,
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.enable_magnitudes = [False, True]
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 379
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 379
    Validation Batches: 53
    Test Batches: 107
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 38.6175, mae: 3.1195, huber: 2.7009, swd: 27.5201, target_std: 20.3337
    Epoch [1/50], Val Losses: mse: 19.8781, mae: 2.7308, huber: 2.3208, swd: 10.2568, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 13.7294, mae: 2.3435, huber: 1.9216, swd: 6.5940, target_std: 18.4014
      Epoch 1 composite train-obj: 2.700925
            Val objective improved inf → 2.3208, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.1143, mae: 2.3168, huber: 1.9069, swd: 9.2110, target_std: 20.3340
    Epoch [2/50], Val Losses: mse: 18.6032, mae: 2.6446, huber: 2.2378, swd: 9.6767, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.9045, mae: 2.2814, huber: 1.8598, swd: 6.1595, target_std: 18.4014
      Epoch 2 composite train-obj: 1.906917
            Val objective improved 2.3208 → 2.2378, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 16.5859, mae: 2.2721, huber: 1.8637, swd: 8.8631, target_std: 20.3332
    Epoch [3/50], Val Losses: mse: 19.7704, mae: 2.7149, huber: 2.3068, swd: 10.5419, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 13.3760, mae: 2.3076, huber: 1.8861, swd: 6.5362, target_std: 18.4014
      Epoch 3 composite train-obj: 1.863719
            No improvement (2.3068), counter 1/5
    Epoch [4/50], Train Losses: mse: 16.2072, mae: 2.2342, huber: 1.8275, swd: 8.6722, target_std: 20.3338
    Epoch [4/50], Val Losses: mse: 18.8024, mae: 2.6459, huber: 2.2403, swd: 10.0010, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.7271, mae: 2.2348, huber: 1.8176, swd: 6.0664, target_std: 18.4014
      Epoch 4 composite train-obj: 1.827480
            No improvement (2.2403), counter 2/5
    Epoch [5/50], Train Losses: mse: 16.2628, mae: 2.2372, huber: 1.8308, swd: 8.7885, target_std: 20.3331
    Epoch [5/50], Val Losses: mse: 19.1842, mae: 2.6692, huber: 2.2662, swd: 10.1299, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.9900, mae: 2.2416, huber: 1.8255, swd: 6.2277, target_std: 18.4014
      Epoch 5 composite train-obj: 1.830844
            No improvement (2.2662), counter 3/5
    Epoch [6/50], Train Losses: mse: 15.7591, mae: 2.1861, huber: 1.7813, swd: 8.4229, target_std: 20.3332
    Epoch [6/50], Val Losses: mse: 18.2138, mae: 2.6093, huber: 2.2070, swd: 9.4319, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 13.2517, mae: 2.2662, huber: 1.8494, swd: 6.4882, target_std: 18.4014
      Epoch 6 composite train-obj: 1.781317
            Val objective improved 2.2378 → 2.2070, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 15.4988, mae: 2.1701, huber: 1.7660, swd: 8.2254, target_std: 20.3335
    Epoch [7/50], Val Losses: mse: 18.1043, mae: 2.6134, huber: 2.2110, swd: 9.2869, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 13.8396, mae: 2.3162, huber: 1.8983, swd: 6.9060, target_std: 18.4014
      Epoch 7 composite train-obj: 1.765995
            No improvement (2.2110), counter 1/5
    Epoch [8/50], Train Losses: mse: 15.4081, mae: 2.1681, huber: 1.7641, swd: 8.1764, target_std: 20.3332
    Epoch [8/50], Val Losses: mse: 18.0626, mae: 2.6580, huber: 2.2521, swd: 9.3606, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 13.1606, mae: 2.3209, huber: 1.8969, swd: 6.4326, target_std: 18.4014
      Epoch 8 composite train-obj: 1.764073
            No improvement (2.2521), counter 2/5
    Epoch [9/50], Train Losses: mse: 15.0912, mae: 2.1365, huber: 1.7336, swd: 7.9349, target_std: 20.3333
    Epoch [9/50], Val Losses: mse: 18.1596, mae: 2.6317, huber: 2.2309, swd: 9.4690, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 13.1219, mae: 2.2558, huber: 1.8405, swd: 6.3406, target_std: 18.4014
      Epoch 9 composite train-obj: 1.733636
            No improvement (2.2309), counter 3/5
    Epoch [10/50], Train Losses: mse: 14.9485, mae: 2.1250, huber: 1.7229, swd: 7.8352, target_std: 20.3340
    Epoch [10/50], Val Losses: mse: 18.1597, mae: 2.6221, huber: 2.2214, swd: 9.3020, target_std: 20.5407
    Epoch [10/50], Test Losses: mse: 13.9799, mae: 2.3119, huber: 1.8965, swd: 7.0824, target_std: 18.4014
      Epoch 10 composite train-obj: 1.722886
            No improvement (2.2214), counter 4/5
    Epoch [11/50], Train Losses: mse: 14.8499, mae: 2.1205, huber: 1.7186, swd: 7.7687, target_std: 20.3333
    Epoch [11/50], Val Losses: mse: 18.2546, mae: 2.6240, huber: 2.2222, swd: 9.4501, target_std: 20.5407
    Epoch [11/50], Test Losses: mse: 15.1309, mae: 2.4100, huber: 1.9923, swd: 8.1973, target_std: 18.4014
      Epoch 11 composite train-obj: 1.718603
    Epoch [11/50], Test Losses: mse: 13.2511, mae: 2.2661, huber: 1.8493, swd: 6.4882, target_std: 18.4014
    Best round's Test MSE: 13.2517, MAE: 2.2662, SWD: 6.4882
    Best round's Validation MSE: 18.2138, MAE: 2.6093
    Best round's Test verification MSE : 13.2511, MAE: 2.2661, SWD: 6.4882
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 39.3819, mae: 3.1493, huber: 2.7308, swd: 29.1708, target_std: 20.3332
    Epoch [1/50], Val Losses: mse: 20.3515, mae: 2.7351, huber: 2.3253, swd: 11.1670, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 13.6560, mae: 2.3260, huber: 1.9049, swd: 6.8456, target_std: 18.4014
      Epoch 1 composite train-obj: 2.730844
            Val objective improved inf → 2.3253, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.2689, mae: 2.3293, huber: 1.9191, swd: 9.5859, target_std: 20.3332
    Epoch [2/50], Val Losses: mse: 20.3445, mae: 2.7480, huber: 2.3397, swd: 11.2574, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 14.6002, mae: 2.4073, huber: 1.9858, swd: 7.8297, target_std: 18.4014
      Epoch 2 composite train-obj: 1.919096
            No improvement (2.3397), counter 1/5
    Epoch [3/50], Train Losses: mse: 16.5660, mae: 2.2661, huber: 1.8585, swd: 9.1574, target_std: 20.3334
    Epoch [3/50], Val Losses: mse: 20.6215, mae: 2.7556, huber: 2.3482, swd: 10.9346, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 14.3119, mae: 2.4052, huber: 1.9789, swd: 7.5028, target_std: 18.4014
      Epoch 3 composite train-obj: 1.858452
            No improvement (2.3482), counter 2/5
    Epoch [4/50], Train Losses: mse: 16.0308, mae: 2.2184, huber: 1.8124, swd: 8.8001, target_std: 20.3340
    Epoch [4/50], Val Losses: mse: 19.1211, mae: 2.6689, huber: 2.2638, swd: 10.0839, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 13.0186, mae: 2.2436, huber: 1.8272, swd: 6.4866, target_std: 18.4014
      Epoch 4 composite train-obj: 1.812413
            Val objective improved 2.3253 → 2.2638, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 15.6543, mae: 2.1836, huber: 1.7790, swd: 8.5943, target_std: 20.3335
    Epoch [5/50], Val Losses: mse: 18.3800, mae: 2.6278, huber: 2.2253, swd: 9.7279, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 13.1208, mae: 2.2471, huber: 1.8326, swd: 6.6265, target_std: 18.4014
      Epoch 5 composite train-obj: 1.779040
            Val objective improved 2.2638 → 2.2253, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 15.4231, mae: 2.1656, huber: 1.7615, swd: 8.4401, target_std: 20.3330
    Epoch [6/50], Val Losses: mse: 18.5008, mae: 2.6482, huber: 2.2450, swd: 10.1377, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.7174, mae: 2.2539, huber: 1.8346, swd: 6.3104, target_std: 18.4014
      Epoch 6 composite train-obj: 1.761500
            No improvement (2.2450), counter 1/5
    Epoch [7/50], Train Losses: mse: 15.2044, mae: 2.1445, huber: 1.7417, swd: 8.2926, target_std: 20.3334
    Epoch [7/50], Val Losses: mse: 18.6188, mae: 2.6382, huber: 2.2374, swd: 10.0145, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 13.5088, mae: 2.2782, huber: 1.8619, swd: 6.9526, target_std: 18.4014
      Epoch 7 composite train-obj: 1.741667
            No improvement (2.2374), counter 2/5
    Epoch [8/50], Train Losses: mse: 15.0633, mae: 2.1357, huber: 1.7333, swd: 8.1909, target_std: 20.3337
    Epoch [8/50], Val Losses: mse: 18.5059, mae: 2.6513, huber: 2.2482, swd: 9.9671, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 13.2970, mae: 2.2658, huber: 1.8503, swd: 6.6299, target_std: 18.4014
      Epoch 8 composite train-obj: 1.733329
            No improvement (2.2482), counter 3/5
    Epoch [9/50], Train Losses: mse: 14.9376, mae: 2.1265, huber: 1.7244, swd: 8.1058, target_std: 20.3334
    Epoch [9/50], Val Losses: mse: 18.2842, mae: 2.6304, huber: 2.2321, swd: 10.0962, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 13.0926, mae: 2.2724, huber: 1.8546, swd: 6.6465, target_std: 18.4014
      Epoch 9 composite train-obj: 1.724431
            No improvement (2.2321), counter 4/5
    Epoch [10/50], Train Losses: mse: 14.8229, mae: 2.1190, huber: 1.7173, swd: 8.0163, target_std: 20.3333
    Epoch [10/50], Val Losses: mse: 19.1432, mae: 2.6925, huber: 2.2923, swd: 10.4523, target_std: 20.5407
    Epoch [10/50], Test Losses: mse: 14.6379, mae: 2.3799, huber: 1.9625, swd: 7.9249, target_std: 18.4014
      Epoch 10 composite train-obj: 1.717266
    Epoch [10/50], Test Losses: mse: 13.1208, mae: 2.2471, huber: 1.8325, swd: 6.6266, target_std: 18.4014
    Best round's Test MSE: 13.1208, MAE: 2.2471, SWD: 6.6265
    Best round's Validation MSE: 18.3800, MAE: 2.6278
    Best round's Test verification MSE : 13.1208, MAE: 2.2471, SWD: 6.6266
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 40.6019, mae: 3.1756, huber: 2.7556, swd: 26.1312, target_std: 20.3336
    Epoch [1/50], Val Losses: mse: 19.7201, mae: 2.7107, huber: 2.2993, swd: 8.9834, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 13.7919, mae: 2.3386, huber: 1.9170, swd: 5.8998, target_std: 18.4014
      Epoch 1 composite train-obj: 2.755575
            Val objective improved inf → 2.2993, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 16.8888, mae: 2.2972, huber: 1.8877, swd: 7.9184, target_std: 20.3336
    Epoch [2/50], Val Losses: mse: 18.7349, mae: 2.6289, huber: 2.2204, swd: 8.5950, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 13.2204, mae: 2.2875, huber: 1.8672, swd: 5.5601, target_std: 18.4014
      Epoch 2 composite train-obj: 1.887683
            Val objective improved 2.2993 → 2.2204, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 16.3205, mae: 2.2460, huber: 1.8391, swd: 7.6530, target_std: 20.3338
    Epoch [3/50], Val Losses: mse: 17.9024, mae: 2.5778, huber: 2.1743, swd: 8.1489, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.7535, mae: 2.2443, huber: 1.8261, swd: 5.2701, target_std: 18.4014
      Epoch 3 composite train-obj: 1.839081
            Val objective improved 2.2204 → 2.1743, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 15.8427, mae: 2.1966, huber: 1.7916, swd: 7.3956, target_std: 20.3338
    Epoch [4/50], Val Losses: mse: 17.5739, mae: 2.5556, huber: 2.1511, swd: 7.9571, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.5384, mae: 2.2447, huber: 1.8240, swd: 5.2270, target_std: 18.4014
      Epoch 4 composite train-obj: 1.791610
            Val objective improved 2.1743 → 2.1511, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 15.6058, mae: 2.1812, huber: 1.7768, swd: 7.2791, target_std: 20.3331
    Epoch [5/50], Val Losses: mse: 17.6067, mae: 2.5717, huber: 2.1669, swd: 7.8098, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.7729, mae: 2.2560, huber: 1.8349, swd: 5.2919, target_std: 18.4014
      Epoch 5 composite train-obj: 1.776806
            No improvement (2.1669), counter 1/5
    Epoch [6/50], Train Losses: mse: 15.2846, mae: 2.1509, huber: 1.7477, swd: 7.0586, target_std: 20.3333
    Epoch [6/50], Val Losses: mse: 18.0197, mae: 2.5887, huber: 2.1852, swd: 7.9959, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.9158, mae: 2.2241, huber: 1.8100, swd: 5.4888, target_std: 18.4014
      Epoch 6 composite train-obj: 1.747722
            No improvement (2.1852), counter 2/5
    Epoch [7/50], Train Losses: mse: 15.1414, mae: 2.1417, huber: 1.7390, swd: 6.9867, target_std: 20.3340
    Epoch [7/50], Val Losses: mse: 17.9632, mae: 2.6055, huber: 2.2022, swd: 7.8486, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 13.1365, mae: 2.2548, huber: 1.8390, swd: 5.6083, target_std: 18.4014
      Epoch 7 composite train-obj: 1.738954
            No improvement (2.2022), counter 3/5
    Epoch [8/50], Train Losses: mse: 14.9833, mae: 2.1294, huber: 1.7274, swd: 6.9021, target_std: 20.3341
    Epoch [8/50], Val Losses: mse: 18.1030, mae: 2.6022, huber: 2.2006, swd: 7.9701, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 13.3718, mae: 2.2584, huber: 1.8441, swd: 5.8124, target_std: 18.4014
      Epoch 8 composite train-obj: 1.727412
            No improvement (2.2006), counter 4/5
    Epoch [9/50], Train Losses: mse: 14.8413, mae: 2.1196, huber: 1.7184, swd: 6.8277, target_std: 20.3336
    Epoch [9/50], Val Losses: mse: 18.7535, mae: 2.6442, huber: 2.2409, swd: 8.3781, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 14.8340, mae: 2.3709, huber: 1.9532, swd: 6.9138, target_std: 18.4014
      Epoch 9 composite train-obj: 1.718392
    Epoch [9/50], Test Losses: mse: 12.5386, mae: 2.2447, huber: 1.8240, swd: 5.2271, target_std: 18.4014
    Best round's Test MSE: 12.5384, MAE: 2.2447, SWD: 5.2270
    Best round's Validation MSE: 17.5739, MAE: 2.5556
    Best round's Test verification MSE : 12.5386, MAE: 2.2447, SWD: 5.2271
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred196_20250430_1905)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.9703 ± 0.3100
      mae: 2.2527 ± 0.0096
      huber: 1.8353 ± 0.0105
      swd: 6.1139 ± 0.6297
      target_std: 18.4014 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 18.0559 ± 0.3475
      mae: 2.5976 ± 0.0306
      huber: 2.1945 ± 0.0316
      swd: 9.0389 ± 0.7745
      target_std: 20.5407 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm2_seq96_pred196_20250430_1905
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336

##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    global_std.shape: torch.Size([7])
    Global Std for ettm2: tensor([10.2434,  6.0312, 13.0618,  4.3690,  6.1544,  6.0135, 11.8865],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 378
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 378
    Validation Batches: 52
    Test Batches: 106
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 43.4807, mae: 3.3641, huber: 2.9415, swd: 29.5177, ept: 239.3629
    Epoch [1/50], Val Losses: mse: 22.7637, mae: 2.9828, huber: 2.5664, swd: 10.3768, ept: 233.5045
    Epoch [1/50], Test Losses: mse: 16.1253, mae: 2.5391, huber: 2.1120, swd: 7.5365, ept: 249.5990
      Epoch 1 composite train-obj: 2.941487
            Val objective improved inf → 2.5664, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 21.6707, mae: 2.5947, huber: 2.1792, swd: 11.0346, ept: 265.2120
    Epoch [2/50], Val Losses: mse: 21.3181, mae: 2.9036, huber: 2.4887, swd: 9.2152, ept: 235.9887
    Epoch [2/50], Test Losses: mse: 15.8564, mae: 2.5123, huber: 2.0863, swd: 7.3265, ept: 248.7800
      Epoch 2 composite train-obj: 2.179224
            Val objective improved 2.5664 → 2.4887, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 20.8727, mae: 2.5410, huber: 2.1276, swd: 10.5113, ept: 267.2026
    Epoch [3/50], Val Losses: mse: 21.5807, mae: 2.9023, huber: 2.4889, swd: 9.2939, ept: 232.9115
    Epoch [3/50], Test Losses: mse: 17.7142, mae: 2.6469, huber: 2.2192, swd: 8.8713, ept: 241.9759
      Epoch 3 composite train-obj: 2.127572
            No improvement (2.4889), counter 1/5
    Epoch [4/50], Train Losses: mse: 20.2161, mae: 2.4943, huber: 2.0824, swd: 10.0752, ept: 268.6257
    Epoch [4/50], Val Losses: mse: 22.4300, mae: 2.9700, huber: 2.5563, swd: 9.9940, ept: 229.8167
    Epoch [4/50], Test Losses: mse: 20.4162, mae: 2.8807, huber: 2.4477, swd: 11.3557, ept: 233.3458
      Epoch 4 composite train-obj: 2.082396
            No improvement (2.5563), counter 2/5
    Epoch [5/50], Train Losses: mse: 19.7467, mae: 2.4719, huber: 2.0602, swd: 9.7689, ept: 269.3612
    Epoch [5/50], Val Losses: mse: 21.5636, mae: 2.9215, huber: 2.5096, swd: 9.5070, ept: 233.9526
    Epoch [5/50], Test Losses: mse: 17.9969, mae: 2.6514, huber: 2.2243, swd: 9.1663, ept: 241.9922
      Epoch 5 composite train-obj: 2.060193
            No improvement (2.5096), counter 3/5
    Epoch [6/50], Train Losses: mse: 19.2929, mae: 2.4409, huber: 2.0300, swd: 9.4609, ept: 270.4035
    Epoch [6/50], Val Losses: mse: 20.7894, mae: 2.8700, huber: 2.4605, swd: 9.0237, ept: 237.1192
    Epoch [6/50], Test Losses: mse: 16.8301, mae: 2.5502, huber: 2.1267, swd: 8.2429, ept: 245.5389
      Epoch 6 composite train-obj: 2.029960
            Val objective improved 2.4887 → 2.4605, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 18.9377, mae: 2.4194, huber: 2.0091, swd: 9.2300, ept: 271.1599
    Epoch [7/50], Val Losses: mse: 22.2081, mae: 2.9493, huber: 2.5409, swd: 9.9002, ept: 233.0904
    Epoch [7/50], Test Losses: mse: 19.2119, mae: 2.7339, huber: 2.3080, swd: 10.2357, ept: 237.5965
      Epoch 7 composite train-obj: 2.009100
            No improvement (2.5409), counter 1/5
    Epoch [8/50], Train Losses: mse: 18.6375, mae: 2.3965, huber: 1.9869, swd: 9.0415, ept: 271.8699
    Epoch [8/50], Val Losses: mse: 20.9233, mae: 2.8611, huber: 2.4546, swd: 9.2532, ept: 236.5095
    Epoch [8/50], Test Losses: mse: 17.2992, mae: 2.5595, huber: 2.1366, swd: 8.5657, ept: 245.7975
      Epoch 8 composite train-obj: 1.986937
            Val objective improved 2.4605 → 2.4546, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 18.5542, mae: 2.3947, huber: 1.9857, swd: 9.0594, ept: 271.7475
    Epoch [9/50], Val Losses: mse: 20.6435, mae: 2.8586, huber: 2.4514, swd: 8.9371, ept: 238.5629
    Epoch [9/50], Test Losses: mse: 17.3535, mae: 2.6120, huber: 2.1844, swd: 8.6613, ept: 245.3851
      Epoch 9 composite train-obj: 1.985654
            Val objective improved 2.4546 → 2.4514, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 18.2695, mae: 2.3699, huber: 1.9618, swd: 8.8916, ept: 272.6243
    Epoch [10/50], Val Losses: mse: 21.0364, mae: 2.8852, huber: 2.4802, swd: 9.3191, ept: 236.9701
    Epoch [10/50], Test Losses: mse: 17.2972, mae: 2.5965, huber: 2.1714, swd: 8.5785, ept: 244.8181
      Epoch 10 composite train-obj: 1.961800
            No improvement (2.4802), counter 1/5
    Epoch [11/50], Train Losses: mse: 18.1591, mae: 2.3652, huber: 1.9571, swd: 8.8761, ept: 272.6020
    Epoch [11/50], Val Losses: mse: 21.7386, mae: 2.9104, huber: 2.5049, swd: 9.9120, ept: 236.8200
    Epoch [11/50], Test Losses: mse: 16.8539, mae: 2.5215, huber: 2.1016, swd: 8.2450, ept: 245.2481
      Epoch 11 composite train-obj: 1.957101
            No improvement (2.5049), counter 2/5
    Epoch [12/50], Train Losses: mse: 17.9778, mae: 2.3488, huber: 1.9415, swd: 8.7565, ept: 273.3396
    Epoch [12/50], Val Losses: mse: 21.0746, mae: 2.8766, huber: 2.4718, swd: 9.4519, ept: 237.7153
    Epoch [12/50], Test Losses: mse: 17.6177, mae: 2.5912, huber: 2.1667, swd: 8.8712, ept: 246.3853
      Epoch 12 composite train-obj: 1.941459
            No improvement (2.4718), counter 3/5
    Epoch [13/50], Train Losses: mse: 17.8785, mae: 2.3399, huber: 1.9328, swd: 8.7177, ept: 273.5666
    Epoch [13/50], Val Losses: mse: 21.5015, mae: 2.8913, huber: 2.4867, swd: 9.6409, ept: 236.5300
    Epoch [13/50], Test Losses: mse: 18.3205, mae: 2.6414, huber: 2.2183, swd: 9.5807, ept: 240.8006
      Epoch 13 composite train-obj: 1.932843
            No improvement (2.4867), counter 4/5
    Epoch [14/50], Train Losses: mse: 17.7956, mae: 2.3361, huber: 1.9294, swd: 8.6921, ept: 273.5683
    Epoch [14/50], Val Losses: mse: 21.5494, mae: 2.8872, huber: 2.4828, swd: 9.6622, ept: 236.4095
    Epoch [14/50], Test Losses: mse: 17.6558, mae: 2.5640, huber: 2.1444, swd: 8.9644, ept: 245.8837
      Epoch 14 composite train-obj: 1.929377
    Epoch [14/50], Test Losses: mse: 17.3523, mae: 2.6120, huber: 2.1843, swd: 8.6603, ept: 245.3883
    Best round's Test MSE: 17.3535, MAE: 2.6120, SWD: 8.6613
    Best round's Validation MSE: 20.6435, MAE: 2.8586, SWD: 8.9371
    Best round's Test verification MSE : 17.3523, MAE: 2.6120, SWD: 8.6603
    Time taken: 154.96 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 43.8799, mae: 3.3940, huber: 2.9708, swd: 31.5840, ept: 237.3490
    Epoch [1/50], Val Losses: mse: 23.3619, mae: 3.0256, huber: 2.6064, swd: 11.0447, ept: 229.7181
    Epoch [1/50], Test Losses: mse: 16.4099, mae: 2.5603, huber: 2.1333, swd: 7.9648, ept: 247.0347
      Epoch 1 composite train-obj: 2.970832
            Val objective improved inf → 2.6064, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 21.9541, mae: 2.6251, huber: 2.2086, swd: 11.7676, ept: 263.8722
    Epoch [2/50], Val Losses: mse: 22.1878, mae: 2.9488, huber: 2.5341, swd: 10.2740, ept: 233.7322
    Epoch [2/50], Test Losses: mse: 15.9615, mae: 2.5224, huber: 2.0962, swd: 7.7608, ept: 246.9662
      Epoch 2 composite train-obj: 2.208617
            Val objective improved 2.6064 → 2.5341, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 20.9579, mae: 2.5539, huber: 2.1399, swd: 11.0686, ept: 266.8412
    Epoch [3/50], Val Losses: mse: 21.0641, mae: 2.8758, huber: 2.4622, swd: 9.3631, ept: 233.9703
    Epoch [3/50], Test Losses: mse: 16.7169, mae: 2.5634, huber: 2.1368, swd: 8.4854, ept: 245.0656
      Epoch 3 composite train-obj: 2.139897
            Val objective improved 2.5341 → 2.4622, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 20.4277, mae: 2.5224, huber: 2.1093, swd: 10.7899, ept: 267.8688
    Epoch [4/50], Val Losses: mse: 21.4752, mae: 2.8956, huber: 2.4850, swd: 9.6143, ept: 230.6565
    Epoch [4/50], Test Losses: mse: 18.3720, mae: 2.6778, huber: 2.2519, swd: 9.9615, ept: 240.5190
      Epoch 4 composite train-obj: 2.109349
            No improvement (2.4850), counter 1/5
    Epoch [5/50], Train Losses: mse: 19.6836, mae: 2.4597, huber: 2.0486, swd: 10.2002, ept: 269.8262
    Epoch [5/50], Val Losses: mse: 20.4395, mae: 2.8550, huber: 2.4453, swd: 9.0079, ept: 235.6988
    Epoch [5/50], Test Losses: mse: 16.9010, mae: 2.5852, huber: 2.1583, swd: 8.6695, ept: 247.1422
      Epoch 5 composite train-obj: 2.048648
            Val objective improved 2.4622 → 2.4453, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 19.4452, mae: 2.4511, huber: 2.0401, swd: 10.0637, ept: 270.0006
    Epoch [6/50], Val Losses: mse: 20.6716, mae: 2.8720, huber: 2.4639, swd: 9.2500, ept: 235.0309
    Epoch [6/50], Test Losses: mse: 16.8288, mae: 2.5394, huber: 2.1183, swd: 8.6279, ept: 244.4051
      Epoch 6 composite train-obj: 2.040063
            No improvement (2.4639), counter 1/5
    Epoch [7/50], Train Losses: mse: 19.2039, mae: 2.4351, huber: 2.0244, swd: 9.8999, ept: 270.6025
    Epoch [7/50], Val Losses: mse: 21.1274, mae: 2.8853, huber: 2.4792, swd: 9.7242, ept: 235.8381
    Epoch [7/50], Test Losses: mse: 16.8260, mae: 2.5482, huber: 2.1259, swd: 8.6040, ept: 245.7988
      Epoch 7 composite train-obj: 2.024426
            No improvement (2.4792), counter 2/5
    Epoch [8/50], Train Losses: mse: 18.9354, mae: 2.4150, huber: 2.0050, swd: 9.7214, ept: 271.0726
    Epoch [8/50], Val Losses: mse: 21.5491, mae: 2.9050, huber: 2.4991, swd: 10.0058, ept: 233.7379
    Epoch [8/50], Test Losses: mse: 18.5371, mae: 2.6916, huber: 2.2652, swd: 10.0927, ept: 239.0171
      Epoch 8 composite train-obj: 2.004994
            No improvement (2.4991), counter 3/5
    Epoch [9/50], Train Losses: mse: 18.6809, mae: 2.3949, huber: 1.9856, swd: 9.5848, ept: 271.8096
    Epoch [9/50], Val Losses: mse: 22.9153, mae: 2.9855, huber: 2.5810, swd: 11.0503, ept: 232.3725
    Epoch [9/50], Test Losses: mse: 20.0932, mae: 2.7836, huber: 2.3591, swd: 11.5440, ept: 236.0260
      Epoch 9 composite train-obj: 1.985624
            No improvement (2.5810), counter 4/5
    Epoch [10/50], Train Losses: mse: 18.4540, mae: 2.3799, huber: 1.9714, swd: 9.4621, ept: 272.3532
    Epoch [10/50], Val Losses: mse: 21.3186, mae: 2.8778, huber: 2.4752, swd: 9.7842, ept: 237.3700
    Epoch [10/50], Test Losses: mse: 17.4687, mae: 2.5877, huber: 2.1647, swd: 9.1951, ept: 242.9136
      Epoch 10 composite train-obj: 1.971372
    Epoch [10/50], Test Losses: mse: 16.9011, mae: 2.5852, huber: 2.1583, swd: 8.6696, ept: 247.1328
    Best round's Test MSE: 16.9010, MAE: 2.5852, SWD: 8.6695
    Best round's Validation MSE: 20.4395, MAE: 2.8550, SWD: 9.0079
    Best round's Test verification MSE : 16.9011, MAE: 2.5852, SWD: 8.6696
    Time taken: 110.71 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 43.9648, mae: 3.3983, huber: 2.9753, swd: 28.2121, ept: 235.2794
    Epoch [1/50], Val Losses: mse: 23.6000, mae: 3.0098, huber: 2.5917, swd: 10.2739, ept: 230.8918
    Epoch [1/50], Test Losses: mse: 16.8708, mae: 2.5929, huber: 2.1637, swd: 7.7399, ept: 247.5701
      Epoch 1 composite train-obj: 2.975317
            Val objective improved inf → 2.5917, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 21.8581, mae: 2.6121, huber: 2.1954, swd: 10.8698, ept: 264.9398
    Epoch [2/50], Val Losses: mse: 22.2444, mae: 2.9506, huber: 2.5345, swd: 9.3601, ept: 232.9212
    Epoch [2/50], Test Losses: mse: 17.1676, mae: 2.6184, huber: 2.1891, swd: 8.1354, ept: 243.9878
      Epoch 2 composite train-obj: 2.195450
            Val objective improved 2.5917 → 2.5345, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 20.8605, mae: 2.5326, huber: 2.1191, swd: 10.2119, ept: 267.6226
    Epoch [3/50], Val Losses: mse: 21.1032, mae: 2.8884, huber: 2.4750, swd: 8.7616, ept: 236.0311
    Epoch [3/50], Test Losses: mse: 16.2849, mae: 2.5502, huber: 2.1216, swd: 7.4738, ept: 247.2768
      Epoch 3 composite train-obj: 2.119086
            Val objective improved 2.5345 → 2.4750, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 20.4854, mae: 2.5199, huber: 2.1072, swd: 10.0423, ept: 267.4650
    Epoch [4/50], Val Losses: mse: 20.5584, mae: 2.8954, huber: 2.4810, swd: 8.6412, ept: 235.9819
    Epoch [4/50], Test Losses: mse: 15.9816, mae: 2.5448, huber: 2.1167, swd: 7.3148, ept: 248.8082
      Epoch 4 composite train-obj: 2.107201
            No improvement (2.4810), counter 1/5
    Epoch [5/50], Train Losses: mse: 19.7284, mae: 2.4579, huber: 2.0465, swd: 9.5115, ept: 270.1042
    Epoch [5/50], Val Losses: mse: 20.7317, mae: 2.8723, huber: 2.4607, swd: 8.4197, ept: 236.5187
    Epoch [5/50], Test Losses: mse: 16.5578, mae: 2.5428, huber: 2.1169, swd: 7.7040, ept: 248.0266
      Epoch 5 composite train-obj: 2.046493
            Val objective improved 2.4750 → 2.4607, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 19.4701, mae: 2.4467, huber: 2.0356, swd: 9.3673, ept: 270.3774
    Epoch [6/50], Val Losses: mse: 20.6969, mae: 2.8876, huber: 2.4783, swd: 8.5932, ept: 237.7433
    Epoch [6/50], Test Losses: mse: 16.8206, mae: 2.5966, huber: 2.1680, swd: 7.9125, ept: 246.9202
      Epoch 6 composite train-obj: 2.035592
            No improvement (2.4783), counter 1/5
    Epoch [7/50], Train Losses: mse: 19.1439, mae: 2.4234, huber: 2.0131, swd: 9.1383, ept: 271.1484
    Epoch [7/50], Val Losses: mse: 21.6875, mae: 2.9066, huber: 2.5009, swd: 9.2550, ept: 236.8830
    Epoch [7/50], Test Losses: mse: 17.0367, mae: 2.5845, huber: 2.1594, swd: 8.1857, ept: 244.3907
      Epoch 7 composite train-obj: 2.013110
            No improvement (2.5009), counter 2/5
    Epoch [8/50], Train Losses: mse: 18.8318, mae: 2.4006, huber: 1.9911, swd: 8.9271, ept: 271.8553
    Epoch [8/50], Val Losses: mse: 21.2294, mae: 2.8775, huber: 2.4743, swd: 9.1563, ept: 238.3843
    Epoch [8/50], Test Losses: mse: 17.2207, mae: 2.5694, huber: 2.1479, swd: 8.2682, ept: 244.2746
      Epoch 8 composite train-obj: 1.991113
            No improvement (2.4743), counter 3/5
    Epoch [9/50], Train Losses: mse: 18.6494, mae: 2.3918, huber: 1.9827, swd: 8.8474, ept: 272.1033
    Epoch [9/50], Val Losses: mse: 21.5460, mae: 2.8643, huber: 2.4610, swd: 9.3762, ept: 238.4410
    Epoch [9/50], Test Losses: mse: 17.4483, mae: 2.5888, huber: 2.1655, swd: 8.4726, ept: 243.8808
      Epoch 9 composite train-obj: 1.982669
            No improvement (2.4610), counter 4/5
    Epoch [10/50], Train Losses: mse: 18.3854, mae: 2.3700, huber: 1.9619, swd: 8.6992, ept: 272.7379
    Epoch [10/50], Val Losses: mse: 21.7814, mae: 2.9007, huber: 2.4976, swd: 9.4722, ept: 236.0867
    Epoch [10/50], Test Losses: mse: 17.6194, mae: 2.6093, huber: 2.1865, swd: 8.5621, ept: 241.7597
      Epoch 10 composite train-obj: 1.961868
    Epoch [10/50], Test Losses: mse: 16.5581, mae: 2.5428, huber: 2.1169, swd: 7.7043, ept: 248.0272
    Best round's Test MSE: 16.5578, MAE: 2.5428, SWD: 7.7040
    Best round's Validation MSE: 20.7317, MAE: 2.8723, SWD: 8.4197
    Best round's Test verification MSE : 16.5581, MAE: 2.5428, SWD: 7.7043
    Time taken: 108.60 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred336_20250512_1619)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 16.9374 ± 0.3259
      mae: 2.5800 ± 0.0285
      huber: 2.1532 ± 0.0278
      swd: 8.3449 ± 0.4532
      ept: 246.8513 ± 1.0978
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 20.6049 ± 0.1224
      mae: 2.8620 ± 0.0075
      huber: 2.4525 ± 0.0063
      swd: 8.7882 ± 0.2622
      ept: 236.9268 ± 1.2044
      count: 52.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 374.35 seconds
    
    Experiment complete: ACL_ettm2_seq96_pred336_20250512_1619
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

##### ab: rotations (8,4)


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128,
    ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=False, 
    ablate_shift_inside_scale=False,
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.enable_magnitudes = [False, True]
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 378
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 378
    Validation Batches: 52
    Test Batches: 106
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 44.9379, mae: 3.4150, huber: 2.9922, swd: 30.8457, target_std: 20.3414
    Epoch [1/50], Val Losses: mse: 22.9836, mae: 2.9926, huber: 2.5741, swd: 10.5197, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 16.4936, mae: 2.5864, huber: 2.1559, swd: 7.8090, target_std: 18.3689
      Epoch 1 composite train-obj: 2.992233
            Val objective improved inf → 2.5741, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 21.8228, mae: 2.5958, huber: 2.1808, swd: 11.0037, target_std: 20.3422
    Epoch [2/50], Val Losses: mse: 22.3769, mae: 2.9787, huber: 2.5653, swd: 9.5805, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 17.1354, mae: 2.6135, huber: 2.1859, swd: 8.0380, target_std: 18.3689
      Epoch 2 composite train-obj: 2.180780
            Val objective improved 2.5741 → 2.5653, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 21.0793, mae: 2.5544, huber: 2.1401, swd: 10.5271, target_std: 20.3414
    Epoch [3/50], Val Losses: mse: 21.5561, mae: 2.9435, huber: 2.5325, swd: 9.4728, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 16.1135, mae: 2.5294, huber: 2.1047, swd: 7.6428, target_std: 18.3689
      Epoch 3 composite train-obj: 2.140052
            Val objective improved 2.5653 → 2.5325, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 20.3348, mae: 2.4988, huber: 2.0862, swd: 10.0863, target_std: 20.3415
    Epoch [4/50], Val Losses: mse: 21.0862, mae: 2.9102, huber: 2.4983, swd: 9.2633, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 16.7008, mae: 2.5601, huber: 2.1349, swd: 8.0166, target_std: 18.3689
      Epoch 4 composite train-obj: 2.086196
            Val objective improved 2.5325 → 2.4983, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 19.8518, mae: 2.4730, huber: 2.0613, swd: 9.8300, target_std: 20.3408
    Epoch [5/50], Val Losses: mse: 21.0599, mae: 2.8804, huber: 2.4698, swd: 9.3133, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 17.2174, mae: 2.5902, huber: 2.1653, swd: 8.6635, target_std: 18.3689
      Epoch 5 composite train-obj: 2.061297
            Val objective improved 2.4983 → 2.4698, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 19.4398, mae: 2.4441, huber: 2.0333, swd: 9.5755, target_std: 20.3406
    Epoch [6/50], Val Losses: mse: 21.4652, mae: 2.9293, huber: 2.5210, swd: 9.6969, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 17.3595, mae: 2.6047, huber: 2.1804, swd: 8.4647, target_std: 18.3689
      Epoch 6 composite train-obj: 2.033348
            No improvement (2.5210), counter 1/5
    Epoch [7/50], Train Losses: mse: 19.0993, mae: 2.4175, huber: 2.0079, swd: 9.3974, target_std: 20.3408
    Epoch [7/50], Val Losses: mse: 21.7984, mae: 2.9708, huber: 2.5633, swd: 10.0805, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 16.7655, mae: 2.5642, huber: 2.1415, swd: 8.1138, target_std: 18.3689
      Epoch 7 composite train-obj: 2.007897
            No improvement (2.5633), counter 2/5
    Epoch [8/50], Train Losses: mse: 18.9036, mae: 2.4109, huber: 2.0012, swd: 9.2726, target_std: 20.3415
    Epoch [8/50], Val Losses: mse: 21.6888, mae: 2.9153, huber: 2.5077, swd: 9.9373, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 18.2180, mae: 2.6237, huber: 2.2016, swd: 9.4894, target_std: 18.3689
      Epoch 8 composite train-obj: 2.001162
            No improvement (2.5077), counter 3/5
    Epoch [9/50], Train Losses: mse: 18.6939, mae: 2.3929, huber: 1.9843, swd: 9.1721, target_std: 20.3410
    Epoch [9/50], Val Losses: mse: 21.5910, mae: 2.9079, huber: 2.4984, swd: 9.8748, target_std: 20.5279
    Epoch [9/50], Test Losses: mse: 17.4155, mae: 2.5797, huber: 2.1567, swd: 8.6905, target_std: 18.3689
      Epoch 9 composite train-obj: 1.984272
            No improvement (2.4984), counter 4/5
    Epoch [10/50], Train Losses: mse: 18.5660, mae: 2.3900, huber: 1.9814, swd: 9.1692, target_std: 20.3416
    Epoch [10/50], Val Losses: mse: 21.5000, mae: 2.9156, huber: 2.5098, swd: 10.0574, target_std: 20.5279
    Epoch [10/50], Test Losses: mse: 17.0870, mae: 2.5695, huber: 2.1474, swd: 8.3720, target_std: 18.3689
      Epoch 10 composite train-obj: 1.981384
    Epoch [10/50], Test Losses: mse: 17.2184, mae: 2.5903, huber: 2.1654, swd: 8.6644, target_std: 18.3689
    Best round's Test MSE: 17.2174, MAE: 2.5902, SWD: 8.6635
    Best round's Validation MSE: 21.0599, MAE: 2.8804
    Best round's Test verification MSE : 17.2184, MAE: 2.5903, SWD: 8.6644
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 44.3439, mae: 3.4078, huber: 2.9844, swd: 31.9206, target_std: 20.3418
    Epoch [1/50], Val Losses: mse: 23.6170, mae: 3.0311, huber: 2.6112, swd: 11.5689, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 16.2582, mae: 2.5807, huber: 2.1490, swd: 7.8958, target_std: 18.3689
      Epoch 1 composite train-obj: 2.984351
            Val objective improved inf → 2.6112, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 21.9263, mae: 2.5927, huber: 2.1783, swd: 11.6317, target_std: 20.3413
    Epoch [2/50], Val Losses: mse: 22.4177, mae: 2.9649, huber: 2.5501, swd: 10.6955, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 16.0650, mae: 2.5834, huber: 2.1520, swd: 8.0273, target_std: 18.3689
      Epoch 2 composite train-obj: 2.178285
            Val objective improved 2.6112 → 2.5501, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 21.1185, mae: 2.5389, huber: 2.1267, swd: 11.0577, target_std: 20.3416
    Epoch [3/50], Val Losses: mse: 23.2047, mae: 3.0196, huber: 2.6082, swd: 10.9118, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 16.6390, mae: 2.6042, huber: 2.1738, swd: 8.4276, target_std: 18.3689
      Epoch 3 composite train-obj: 2.126737
            No improvement (2.6082), counter 1/5
    Epoch [4/50], Train Losses: mse: 20.3419, mae: 2.4882, huber: 2.0776, swd: 10.5354, target_std: 20.3412
    Epoch [4/50], Val Losses: mse: 23.7590, mae: 3.0497, huber: 2.6387, swd: 11.5069, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 18.0542, mae: 2.6552, huber: 2.2315, swd: 9.7213, target_std: 18.3689
      Epoch 4 composite train-obj: 2.077600
            No improvement (2.6387), counter 2/5
    Epoch [5/50], Train Losses: mse: 19.7355, mae: 2.4497, huber: 2.0400, swd: 10.1512, target_std: 20.3413
    Epoch [5/50], Val Losses: mse: 23.1296, mae: 3.0724, huber: 2.6613, swd: 11.3643, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 17.1497, mae: 2.5994, huber: 2.1756, swd: 8.6021, target_std: 18.3689
      Epoch 5 composite train-obj: 2.040016
            No improvement (2.6613), counter 3/5
    Epoch [6/50], Train Losses: mse: 19.3694, mae: 2.4334, huber: 2.0239, swd: 9.9499, target_std: 20.3412
    Epoch [6/50], Val Losses: mse: 23.3438, mae: 3.0109, huber: 2.5996, swd: 11.3922, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 17.1975, mae: 2.5638, huber: 2.1403, swd: 8.6509, target_std: 18.3689
      Epoch 6 composite train-obj: 2.023882
            No improvement (2.5996), counter 4/5
    Epoch [7/50], Train Losses: mse: 19.0267, mae: 2.4140, huber: 2.0050, swd: 9.7449, target_std: 20.3420
    Epoch [7/50], Val Losses: mse: 24.6613, mae: 3.0468, huber: 2.6368, swd: 12.4558, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 17.9258, mae: 2.5941, huber: 2.1713, swd: 9.6077, target_std: 18.3689
      Epoch 7 composite train-obj: 2.005027
    Epoch [7/50], Test Losses: mse: 16.0650, mae: 2.5834, huber: 2.1520, swd: 8.0273, target_std: 18.3689
    Best round's Test MSE: 16.0650, MAE: 2.5834, SWD: 8.0273
    Best round's Validation MSE: 22.4177, MAE: 2.9649
    Best round's Test verification MSE : 16.0650, MAE: 2.5834, SWD: 8.0273
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 44.6024, mae: 3.4197, huber: 2.9957, swd: 29.6205, target_std: 20.3416
    Epoch [1/50], Val Losses: mse: 22.7062, mae: 2.9836, huber: 2.5654, swd: 9.7936, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 16.2284, mae: 2.5338, huber: 2.1080, swd: 7.3046, target_std: 18.3689
      Epoch 1 composite train-obj: 2.995719
            Val objective improved inf → 2.5654, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 21.5999, mae: 2.5898, huber: 2.1737, swd: 10.5404, target_std: 20.3410
    Epoch [2/50], Val Losses: mse: 22.1651, mae: 2.9492, huber: 2.5316, swd: 9.3272, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 16.9406, mae: 2.5680, huber: 2.1435, swd: 7.8890, target_std: 18.3689
      Epoch 2 composite train-obj: 2.173704
            Val objective improved 2.5654 → 2.5316, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 20.9117, mae: 2.5420, huber: 2.1280, swd: 10.1810, target_std: 20.3408
    Epoch [3/50], Val Losses: mse: 20.7162, mae: 2.8615, huber: 2.4486, swd: 8.4581, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 16.4774, mae: 2.5571, huber: 2.1288, swd: 7.6490, target_std: 18.3689
      Epoch 3 composite train-obj: 2.128042
            Val objective improved 2.5316 → 2.4486, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 20.0836, mae: 2.4860, huber: 2.0740, swd: 9.6241, target_std: 20.3414
    Epoch [4/50], Val Losses: mse: 20.4011, mae: 2.8331, huber: 2.4262, swd: 8.3618, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 16.4873, mae: 2.5123, huber: 2.0912, swd: 7.7191, target_std: 18.3689
      Epoch 4 composite train-obj: 2.074037
            Val objective improved 2.4486 → 2.4262, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 19.5764, mae: 2.4551, huber: 2.0442, swd: 9.2962, target_std: 20.3411
    Epoch [5/50], Val Losses: mse: 20.5482, mae: 2.8624, huber: 2.4547, swd: 8.3512, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 16.8934, mae: 2.5635, huber: 2.1404, swd: 7.9312, target_std: 18.3689
      Epoch 5 composite train-obj: 2.044159
            No improvement (2.4547), counter 1/5
    Epoch [6/50], Train Losses: mse: 19.1647, mae: 2.4286, huber: 2.0183, swd: 9.0406, target_std: 20.3413
    Epoch [6/50], Val Losses: mse: 21.3099, mae: 2.9083, huber: 2.5014, swd: 8.8540, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 16.8075, mae: 2.5682, huber: 2.1443, swd: 7.7050, target_std: 18.3689
      Epoch 6 composite train-obj: 2.018338
            No improvement (2.5014), counter 2/5
    Epoch [7/50], Train Losses: mse: 18.8232, mae: 2.4011, huber: 1.9919, swd: 8.8413, target_std: 20.3417
    Epoch [7/50], Val Losses: mse: 20.9620, mae: 2.8799, huber: 2.4752, swd: 8.6566, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 18.0520, mae: 2.6409, huber: 2.2177, swd: 8.8350, target_std: 18.3689
      Epoch 7 composite train-obj: 1.991933
            No improvement (2.4752), counter 3/5
    Epoch [8/50], Train Losses: mse: 18.6907, mae: 2.3952, huber: 1.9858, swd: 8.8169, target_std: 20.3409
    Epoch [8/50], Val Losses: mse: 22.5430, mae: 2.9625, huber: 2.5545, swd: 9.6488, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 19.5326, mae: 2.7833, huber: 2.3534, swd: 9.9330, target_std: 18.3689
      Epoch 8 composite train-obj: 1.985839
            No improvement (2.5545), counter 4/5
    Epoch [9/50], Train Losses: mse: 18.4305, mae: 2.3782, huber: 1.9695, swd: 8.6860, target_std: 20.3415
    Epoch [9/50], Val Losses: mse: 20.9736, mae: 2.8607, huber: 2.4535, swd: 8.8871, target_std: 20.5279
    Epoch [9/50], Test Losses: mse: 17.6740, mae: 2.6085, huber: 2.1834, swd: 8.6867, target_std: 18.3689
      Epoch 9 composite train-obj: 1.969509
    Epoch [9/50], Test Losses: mse: 16.4872, mae: 2.5123, huber: 2.0912, swd: 7.7190, target_std: 18.3689
    Best round's Test MSE: 16.4873, MAE: 2.5123, SWD: 7.7191
    Best round's Validation MSE: 20.4011, MAE: 2.8331
    Best round's Test verification MSE : 16.4872, MAE: 2.5123, SWD: 7.7190
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred336_20250430_1928)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 16.5899 ± 0.4760
      mae: 2.5620 ± 0.0353
      huber: 2.1362 ± 0.0323
      swd: 8.1367 ± 0.3932
      target_std: 18.3689 ± 0.0000
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 21.2929 ± 0.8396
      mae: 2.8928 ± 0.0545
      huber: 2.4821 ± 0.0513
      swd: 9.4569 ± 0.9581
      target_std: 20.5279 ± 0.0000
      count: 52.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm2_seq96_pred336_20250430_1928
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720

##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    global_std.shape: torch.Size([7])
    Global Std for ettm2: tensor([10.2434,  6.0312, 13.0618,  4.3690,  6.1544,  6.0135, 11.8865],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 375
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 375
    Validation Batches: 49
    Test Batches: 103
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 51.1276, mae: 3.7482, huber: 3.3191, swd: 33.2174, ept: 419.6088
    Epoch [1/50], Val Losses: mse: 28.3690, mae: 3.4615, huber: 3.0324, swd: 11.0795, ept: 372.9170
    Epoch [1/50], Test Losses: mse: 22.1246, mae: 3.0317, huber: 2.5945, swd: 10.3920, ept: 399.6266
      Epoch 1 composite train-obj: 3.319088
            Val objective improved inf → 3.0324, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 29.1377, mae: 3.0155, huber: 2.5917, swd: 14.1069, ept: 466.7330
    Epoch [2/50], Val Losses: mse: 25.5667, mae: 3.3085, huber: 2.8834, swd: 9.1887, ept: 395.6856
    Epoch [2/50], Test Losses: mse: 20.8571, mae: 2.9095, huber: 2.4744, swd: 9.7533, ept: 415.8786
      Epoch 2 composite train-obj: 2.591715
            Val objective improved 3.0324 → 2.8834, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 27.9360, mae: 2.9293, huber: 2.5079, swd: 13.2620, ept: 476.3963
    Epoch [3/50], Val Losses: mse: 25.9543, mae: 3.3407, huber: 2.9183, swd: 9.5114, ept: 398.9518
    Epoch [3/50], Test Losses: mse: 21.0627, mae: 2.9479, huber: 2.5112, swd: 9.9618, ept: 415.2988
      Epoch 3 composite train-obj: 2.507925
            No improvement (2.9183), counter 1/5
    Epoch [4/50], Train Losses: mse: 27.2139, mae: 2.8824, huber: 2.4624, swd: 12.8152, ept: 480.8090
    Epoch [4/50], Val Losses: mse: 26.1492, mae: 3.3193, huber: 2.8986, swd: 9.8436, ept: 395.9550
    Epoch [4/50], Test Losses: mse: 20.8296, mae: 2.9244, huber: 2.4881, swd: 9.8719, ept: 420.8357
      Epoch 4 composite train-obj: 2.462366
            No improvement (2.8986), counter 2/5
    Epoch [5/50], Train Losses: mse: 26.6654, mae: 2.8394, huber: 2.4203, swd: 12.4361, ept: 484.6983
    Epoch [5/50], Val Losses: mse: 27.5296, mae: 3.4120, huber: 2.9915, swd: 10.6816, ept: 386.6018
    Epoch [5/50], Test Losses: mse: 24.4683, mae: 3.1812, huber: 2.7440, swd: 12.9444, ept: 403.7819
      Epoch 5 composite train-obj: 2.420270
            No improvement (2.9915), counter 3/5
    Epoch [6/50], Train Losses: mse: 26.3907, mae: 2.8183, huber: 2.3995, swd: 12.2520, ept: 486.2369
    Epoch [6/50], Val Losses: mse: 26.0412, mae: 3.3579, huber: 2.9379, swd: 9.6559, ept: 396.2400
    Epoch [6/50], Test Losses: mse: 21.2560, mae: 2.9628, huber: 2.5277, swd: 10.1943, ept: 419.8352
      Epoch 6 composite train-obj: 2.399503
            No improvement (2.9379), counter 4/5
    Epoch [7/50], Train Losses: mse: 26.1289, mae: 2.8027, huber: 2.3842, swd: 12.0916, ept: 487.7580
    Epoch [7/50], Val Losses: mse: 25.8721, mae: 3.2882, huber: 2.8706, swd: 9.6905, ept: 396.6043
    Epoch [7/50], Test Losses: mse: 22.9740, mae: 3.0878, huber: 2.6516, swd: 11.7703, ept: 413.9982
      Epoch 7 composite train-obj: 2.384161
            Val objective improved 2.8834 → 2.8706, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 26.0196, mae: 2.7933, huber: 2.3751, swd: 12.0555, ept: 488.3527
    Epoch [8/50], Val Losses: mse: 28.0327, mae: 3.4291, huber: 3.0090, swd: 11.2238, ept: 388.3263
    Epoch [8/50], Test Losses: mse: 24.8753, mae: 3.2119, huber: 2.7745, swd: 13.3380, ept: 404.5636
      Epoch 8 composite train-obj: 2.375095
            No improvement (3.0090), counter 1/5
    Epoch [9/50], Train Losses: mse: 25.7653, mae: 2.7747, huber: 2.3569, swd: 11.8952, ept: 489.3114
    Epoch [9/50], Val Losses: mse: 25.7661, mae: 3.2768, huber: 2.8581, swd: 9.7634, ept: 399.2759
    Epoch [9/50], Test Losses: mse: 22.9334, mae: 3.0952, huber: 2.6575, swd: 11.6307, ept: 416.2934
      Epoch 9 composite train-obj: 2.356880
            Val objective improved 2.8706 → 2.8581, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 25.5662, mae: 2.7594, huber: 2.3422, swd: 11.7754, ept: 490.4844
    Epoch [10/50], Val Losses: mse: 25.8279, mae: 3.2997, huber: 2.8795, swd: 9.9623, ept: 399.3066
    Epoch [10/50], Test Losses: mse: 21.8441, mae: 3.0233, huber: 2.5861, swd: 10.8541, ept: 420.1616
      Epoch 10 composite train-obj: 2.342236
            No improvement (2.8795), counter 1/5
    Epoch [11/50], Train Losses: mse: 25.3264, mae: 2.7365, huber: 2.3199, swd: 11.6024, ept: 492.0550
    Epoch [11/50], Val Losses: mse: 27.9755, mae: 3.4206, huber: 2.9998, swd: 11.4614, ept: 389.6942
    Epoch [11/50], Test Losses: mse: 24.5654, mae: 3.1690, huber: 2.7343, swd: 13.2123, ept: 409.7595
      Epoch 11 composite train-obj: 2.319881
            No improvement (2.9998), counter 2/5
    Epoch [12/50], Train Losses: mse: 25.1800, mae: 2.7266, huber: 2.3102, swd: 11.5143, ept: 493.2397
    Epoch [12/50], Val Losses: mse: 26.9675, mae: 3.3882, huber: 2.9678, swd: 10.6129, ept: 395.7968
    Epoch [12/50], Test Losses: mse: 21.8080, mae: 2.9713, huber: 2.5400, swd: 10.7154, ept: 420.5361
      Epoch 12 composite train-obj: 2.310188
            No improvement (2.9678), counter 3/5
    Epoch [13/50], Train Losses: mse: 24.9923, mae: 2.7168, huber: 2.3006, swd: 11.3892, ept: 493.3942
    Epoch [13/50], Val Losses: mse: 27.4121, mae: 3.3877, huber: 2.9686, swd: 11.0095, ept: 393.9709
    Epoch [13/50], Test Losses: mse: 23.6108, mae: 3.1173, huber: 2.6818, swd: 12.3294, ept: 413.8452
      Epoch 13 composite train-obj: 2.300566
            No improvement (2.9686), counter 4/5
    Epoch [14/50], Train Losses: mse: 24.9337, mae: 2.7145, huber: 2.2983, swd: 11.3699, ept: 493.1188
    Epoch [14/50], Val Losses: mse: 26.4646, mae: 3.3365, huber: 2.9153, swd: 10.2959, ept: 399.3707
    Epoch [14/50], Test Losses: mse: 22.5420, mae: 3.0194, huber: 2.5869, swd: 11.5083, ept: 419.0052
      Epoch 14 composite train-obj: 2.298283
    Epoch [14/50], Test Losses: mse: 22.9325, mae: 3.0952, huber: 2.6575, swd: 11.6300, ept: 416.3204
    Best round's Test MSE: 22.9334, MAE: 3.0952, SWD: 11.6307
    Best round's Validation MSE: 25.7661, MAE: 3.2768, SWD: 9.7634
    Best round's Test verification MSE : 22.9325, MAE: 3.0952, SWD: 11.6300
    Time taken: 150.33 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 51.3504, mae: 3.7783, huber: 3.3491, swd: 29.8001, ept: 414.6269
    Epoch [1/50], Val Losses: mse: 28.2562, mae: 3.4636, huber: 3.0341, swd: 9.9497, ept: 376.4311
    Epoch [1/50], Test Losses: mse: 21.3464, mae: 2.9883, huber: 2.5518, swd: 8.8282, ept: 409.4589
      Epoch 1 composite train-obj: 3.349074
            Val objective improved inf → 3.0341, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 29.6089, mae: 3.0501, huber: 2.6254, swd: 13.1671, ept: 457.7555
    Epoch [2/50], Val Losses: mse: 26.3297, mae: 3.3899, huber: 2.9625, swd: 9.3708, ept: 396.3216
    Epoch [2/50], Test Losses: mse: 19.3411, mae: 2.8131, huber: 2.3802, swd: 7.8759, ept: 428.3693
      Epoch 2 composite train-obj: 2.625430
            Val objective improved 3.0341 → 2.9625, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 27.9923, mae: 2.9382, huber: 2.5162, swd: 12.2694, ept: 476.3912
    Epoch [3/50], Val Losses: mse: 24.9058, mae: 3.2828, huber: 2.8600, swd: 8.4015, ept: 404.4487
    Epoch [3/50], Test Losses: mse: 20.0601, mae: 2.8823, huber: 2.4458, swd: 8.6757, ept: 427.5245
      Epoch 3 composite train-obj: 2.516235
            Val objective improved 2.9625 → 2.8600, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 27.1875, mae: 2.8800, huber: 2.4596, swd: 11.7881, ept: 481.2348
    Epoch [4/50], Val Losses: mse: 25.7377, mae: 3.3017, huber: 2.8827, swd: 8.7901, ept: 400.3187
    Epoch [4/50], Test Losses: mse: 20.7818, mae: 2.8873, huber: 2.4553, swd: 9.0969, ept: 421.6884
      Epoch 4 composite train-obj: 2.459635
            No improvement (2.8827), counter 1/5
    Epoch [5/50], Train Losses: mse: 26.6582, mae: 2.8349, huber: 2.4160, swd: 11.4985, ept: 484.0154
    Epoch [5/50], Val Losses: mse: 25.6611, mae: 3.2869, huber: 2.8688, swd: 8.5717, ept: 398.7279
    Epoch [5/50], Test Losses: mse: 21.4356, mae: 2.9351, huber: 2.5029, swd: 9.5434, ept: 419.9744
      Epoch 5 composite train-obj: 2.416043
            No improvement (2.8688), counter 2/5
    Epoch [6/50], Train Losses: mse: 26.3478, mae: 2.8087, huber: 2.3905, swd: 11.3168, ept: 486.2335
    Epoch [6/50], Val Losses: mse: 26.7454, mae: 3.3443, huber: 2.9270, swd: 9.3184, ept: 392.4201
    Epoch [6/50], Test Losses: mse: 23.5891, mae: 3.0930, huber: 2.6606, swd: 11.3445, ept: 408.3994
      Epoch 6 composite train-obj: 2.390520
            No improvement (2.9270), counter 3/5
    Epoch [7/50], Train Losses: mse: 26.1387, mae: 2.7923, huber: 2.3747, swd: 11.2172, ept: 487.7558
    Epoch [7/50], Val Losses: mse: 25.1162, mae: 3.2801, huber: 2.8629, swd: 8.7494, ept: 403.7561
    Epoch [7/50], Test Losses: mse: 20.3247, mae: 2.8787, huber: 2.4470, swd: 8.9242, ept: 426.3760
      Epoch 7 composite train-obj: 2.374660
            No improvement (2.8629), counter 4/5
    Epoch [8/50], Train Losses: mse: 25.9163, mae: 2.7718, huber: 2.3547, swd: 11.0876, ept: 489.3315
    Epoch [8/50], Val Losses: mse: 25.2929, mae: 3.2707, huber: 2.8534, swd: 8.7809, ept: 402.3176
    Epoch [8/50], Test Losses: mse: 21.1255, mae: 2.9342, huber: 2.5014, swd: 9.3674, ept: 426.5130
      Epoch 8 composite train-obj: 2.354726
            Val objective improved 2.8600 → 2.8534, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 25.7531, mae: 2.7617, huber: 2.3448, swd: 10.9931, ept: 490.2278
    Epoch [9/50], Val Losses: mse: 25.3854, mae: 3.2959, huber: 2.8770, swd: 8.6857, ept: 402.0823
    Epoch [9/50], Test Losses: mse: 21.0256, mae: 2.9230, huber: 2.4910, swd: 9.4665, ept: 424.1177
      Epoch 9 composite train-obj: 2.344819
            No improvement (2.8770), counter 1/5
    Epoch [10/50], Train Losses: mse: 25.5101, mae: 2.7446, huber: 2.3279, swd: 10.8420, ept: 491.6415
    Epoch [10/50], Val Losses: mse: 25.6350, mae: 3.2807, huber: 2.8631, swd: 8.8866, ept: 403.1047
    Epoch [10/50], Test Losses: mse: 21.8241, mae: 2.9794, huber: 2.5463, swd: 10.0404, ept: 423.4648
      Epoch 10 composite train-obj: 2.327913
            No improvement (2.8631), counter 2/5
    Epoch [11/50], Train Losses: mse: 25.2872, mae: 2.7283, huber: 2.3121, swd: 10.6990, ept: 492.6964
    Epoch [11/50], Val Losses: mse: 26.6381, mae: 3.3479, huber: 2.9290, swd: 9.5165, ept: 395.3088
    Epoch [11/50], Test Losses: mse: 22.2113, mae: 2.9836, huber: 2.5522, swd: 10.3070, ept: 421.4591
      Epoch 11 composite train-obj: 2.312140
            No improvement (2.9290), counter 3/5
    Epoch [12/50], Train Losses: mse: 25.1635, mae: 2.7243, huber: 2.3082, swd: 10.6259, ept: 492.8919
    Epoch [12/50], Val Losses: mse: 27.3344, mae: 3.4004, huber: 2.9815, swd: 10.1906, ept: 396.3418
    Epoch [12/50], Test Losses: mse: 21.8273, mae: 2.9980, huber: 2.5629, swd: 9.9710, ept: 422.1196
      Epoch 12 composite train-obj: 2.308212
            No improvement (2.9815), counter 4/5
    Epoch [13/50], Train Losses: mse: 24.8710, mae: 2.7024, huber: 2.2869, swd: 10.4360, ept: 493.9159
    Epoch [13/50], Val Losses: mse: 27.1353, mae: 3.3627, huber: 2.9437, swd: 10.0060, ept: 396.0711
    Epoch [13/50], Test Losses: mse: 23.6679, mae: 3.1383, huber: 2.7009, swd: 11.5089, ept: 415.8358
      Epoch 13 composite train-obj: 2.286939
    Epoch [13/50], Test Losses: mse: 21.1254, mae: 2.9342, huber: 2.5014, swd: 9.3674, ept: 426.5006
    Best round's Test MSE: 21.1255, MAE: 2.9342, SWD: 9.3674
    Best round's Validation MSE: 25.2929, MAE: 3.2707, SWD: 8.7809
    Best round's Test verification MSE : 21.1254, MAE: 2.9342, SWD: 9.3674
    Time taken: 135.58 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 48.5912, mae: 3.6947, huber: 3.2660, swd: 31.5396, ept: 418.6569
    Epoch [1/50], Val Losses: mse: 28.1812, mae: 3.4468, huber: 3.0176, swd: 12.0077, ept: 380.5557
    Epoch [1/50], Test Losses: mse: 20.4641, mae: 2.9277, huber: 2.4917, swd: 9.7120, ept: 414.2374
      Epoch 1 composite train-obj: 3.266041
            Val objective improved inf → 3.0176, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 29.1764, mae: 3.0137, huber: 2.5895, swd: 14.8620, ept: 469.0142
    Epoch [2/50], Val Losses: mse: 25.6632, mae: 3.3096, huber: 2.8829, swd: 9.6552, ept: 393.2267
    Epoch [2/50], Test Losses: mse: 20.7748, mae: 2.9435, huber: 2.5047, swd: 10.1690, ept: 416.3657
      Epoch 2 composite train-obj: 2.589484
            Val objective improved 3.0176 → 2.8829, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 27.6327, mae: 2.9085, huber: 2.4872, swd: 13.6505, ept: 480.1172
    Epoch [3/50], Val Losses: mse: 25.1098, mae: 3.3015, huber: 2.8790, swd: 9.0996, ept: 398.5563
    Epoch [3/50], Test Losses: mse: 21.0890, mae: 2.9511, huber: 2.5145, swd: 10.3868, ept: 415.5037
      Epoch 3 composite train-obj: 2.487242
            Val objective improved 2.8829 → 2.8790, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 27.1749, mae: 2.8855, huber: 2.4654, swd: 13.4069, ept: 481.4742
    Epoch [4/50], Val Losses: mse: 25.3645, mae: 3.3428, huber: 2.9199, swd: 9.5782, ept: 401.1483
    Epoch [4/50], Test Losses: mse: 20.5150, mae: 2.9148, huber: 2.4781, swd: 10.0545, ept: 424.7527
      Epoch 4 composite train-obj: 2.465407
            No improvement (2.9199), counter 1/5
    Epoch [5/50], Train Losses: mse: 26.7836, mae: 2.8546, huber: 2.4353, swd: 13.1819, ept: 483.8508
    Epoch [5/50], Val Losses: mse: 25.3283, mae: 3.3222, huber: 2.9005, swd: 9.6859, ept: 402.5634
    Epoch [5/50], Test Losses: mse: 20.4758, mae: 2.9384, huber: 2.4992, swd: 9.9869, ept: 428.0705
      Epoch 5 composite train-obj: 2.435283
            No improvement (2.9005), counter 2/5
    Epoch [6/50], Train Losses: mse: 26.4002, mae: 2.8147, huber: 2.3962, swd: 12.9382, ept: 487.5949
    Epoch [6/50], Val Losses: mse: 25.6263, mae: 3.3571, huber: 2.9362, swd: 10.3951, ept: 406.5887
    Epoch [6/50], Test Losses: mse: 19.9555, mae: 2.9195, huber: 2.4820, swd: 9.6792, ept: 435.0919
      Epoch 6 composite train-obj: 2.396225
            No improvement (2.9362), counter 3/5
    Epoch [7/50], Train Losses: mse: 26.2092, mae: 2.8032, huber: 2.3849, swd: 12.8117, ept: 487.7864
    Epoch [7/50], Val Losses: mse: 25.2197, mae: 3.2438, huber: 2.8267, swd: 9.7045, ept: 402.7534
    Epoch [7/50], Test Losses: mse: 20.6723, mae: 2.9010, huber: 2.4679, swd: 10.3637, ept: 426.1329
      Epoch 7 composite train-obj: 2.384931
            Val objective improved 2.8790 → 2.8267, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 25.8731, mae: 2.7724, huber: 2.3550, swd: 12.5632, ept: 490.5069
    Epoch [8/50], Val Losses: mse: 24.2999, mae: 3.2029, huber: 2.7839, swd: 8.9740, ept: 405.7259
    Epoch [8/50], Test Losses: mse: 21.3217, mae: 2.9875, huber: 2.5508, swd: 11.0120, ept: 423.4051
      Epoch 8 composite train-obj: 2.355025
            Val objective improved 2.8267 → 2.7839, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 25.6984, mae: 2.7651, huber: 2.3480, swd: 12.4562, ept: 490.4345
    Epoch [9/50], Val Losses: mse: 24.5799, mae: 3.2445, huber: 2.8204, swd: 9.1204, ept: 404.7738
    Epoch [9/50], Test Losses: mse: 21.8930, mae: 3.0400, huber: 2.6009, swd: 11.5558, ept: 417.8114
      Epoch 9 composite train-obj: 2.347952
            No improvement (2.8204), counter 1/5
    Epoch [10/50], Train Losses: mse: 25.4523, mae: 2.7487, huber: 2.3317, swd: 12.2849, ept: 491.9910
    Epoch [10/50], Val Losses: mse: 25.3878, mae: 3.2654, huber: 2.8445, swd: 9.9194, ept: 401.2348
    Epoch [10/50], Test Losses: mse: 20.8584, mae: 2.9032, huber: 2.4697, swd: 10.4893, ept: 424.8793
      Epoch 10 composite train-obj: 2.331690
            No improvement (2.8445), counter 2/5
    Epoch [11/50], Train Losses: mse: 25.2632, mae: 2.7367, huber: 2.3203, swd: 12.1615, ept: 492.2960
    Epoch [11/50], Val Losses: mse: 25.8576, mae: 3.3431, huber: 2.9219, swd: 10.5103, ept: 408.8335
    Epoch [11/50], Test Losses: mse: 20.3372, mae: 2.9305, huber: 2.4931, swd: 10.1281, ept: 428.5623
      Epoch 11 composite train-obj: 2.320299
            No improvement (2.9219), counter 3/5
    Epoch [12/50], Train Losses: mse: 25.1402, mae: 2.7279, huber: 2.3116, swd: 12.0874, ept: 493.3135
    Epoch [12/50], Val Losses: mse: 26.0750, mae: 3.3265, huber: 2.9065, swd: 10.4042, ept: 404.1387
    Epoch [12/50], Test Losses: mse: 21.0703, mae: 2.9572, huber: 2.5212, swd: 10.6031, ept: 427.3564
      Epoch 12 composite train-obj: 2.311643
            No improvement (2.9065), counter 4/5
    Epoch [13/50], Train Losses: mse: 25.0657, mae: 2.7274, huber: 2.3114, swd: 12.0654, ept: 492.8638
    Epoch [13/50], Val Losses: mse: 26.5256, mae: 3.3348, huber: 2.9158, swd: 10.6979, ept: 398.3454
    Epoch [13/50], Test Losses: mse: 21.2424, mae: 2.9329, huber: 2.4999, swd: 10.7352, ept: 424.8751
      Epoch 13 composite train-obj: 2.311436
    Epoch [13/50], Test Losses: mse: 21.3217, mae: 2.9875, huber: 2.5508, swd: 11.0120, ept: 423.4086
    Best round's Test MSE: 21.3217, MAE: 2.9875, SWD: 11.0120
    Best round's Validation MSE: 24.2999, MAE: 3.2029, SWD: 8.9740
    Best round's Test verification MSE : 21.3217, MAE: 2.9875, SWD: 11.0120
    Time taken: 140.03 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred720_20250512_1626)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 21.7935 ± 0.8100
      mae: 3.0056 ± 0.0670
      huber: 2.5699 ± 0.0652
      swd: 10.6700 ± 0.9551
      ept: 422.0705 ± 4.2775
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 25.1196 ± 0.6110
      mae: 3.2501 ± 0.0335
      huber: 2.8318 ± 0.0339
      swd: 9.1728 ± 0.4250
      ept: 402.4398 ± 2.6347
      count: 49.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 426.03 seconds
    
    Experiment complete: ACL_ettm2_seq96_pred720_20250512_1626
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm2']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=128,
    dim_augment=128,
    ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=False, 
    ablate_shift_inside_scale=False,
)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.enable_magnitudes = [False, True]
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 375
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 375
    Validation Batches: 49
    Test Batches: 103
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 54.9925, mae: 3.8702, huber: 3.4408, swd: 36.7978, target_std: 20.3600
    Epoch [1/50], Val Losses: mse: 27.3710, mae: 3.4175, huber: 2.9910, swd: 11.3414, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.0399, mae: 2.8913, huber: 2.4551, swd: 9.0514, target_std: 18.3425
      Epoch 1 composite train-obj: 3.440846
            Val objective improved inf → 2.9910, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 29.0079, mae: 2.9864, huber: 2.5638, swd: 13.8922, target_std: 20.3589
    Epoch [2/50], Val Losses: mse: 26.0223, mae: 3.3300, huber: 2.9061, swd: 9.4097, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 21.0924, mae: 2.9383, huber: 2.5021, swd: 9.7919, target_std: 18.3425
      Epoch 2 composite train-obj: 2.563778
            Val objective improved 2.9910 → 2.9061, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 27.9099, mae: 2.9170, huber: 2.4953, swd: 13.0344, target_std: 20.3584
    Epoch [3/50], Val Losses: mse: 26.7633, mae: 3.4194, huber: 2.9971, swd: 10.2172, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.7724, mae: 2.8937, huber: 2.4625, swd: 9.8396, target_std: 18.3425
      Epoch 3 composite train-obj: 2.495334
            No improvement (2.9971), counter 1/5
    Epoch [4/50], Train Losses: mse: 27.2957, mae: 2.8827, huber: 2.4623, swd: 12.6672, target_std: 20.3584
    Epoch [4/50], Val Losses: mse: 25.8754, mae: 3.3496, huber: 2.9292, swd: 9.4327, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 21.6508, mae: 2.9497, huber: 2.5182, swd: 10.5378, target_std: 18.3425
      Epoch 4 composite train-obj: 2.462256
            No improvement (2.9292), counter 2/5
    Epoch [5/50], Train Losses: mse: 26.6415, mae: 2.8318, huber: 2.4132, swd: 12.2827, target_std: 20.3593
    Epoch [5/50], Val Losses: mse: 27.9172, mae: 3.4725, huber: 3.0525, swd: 11.1768, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 22.0864, mae: 3.0252, huber: 2.5905, swd: 10.8996, target_std: 18.3425
      Epoch 5 composite train-obj: 2.413181
            No improvement (3.0525), counter 3/5
    Epoch [6/50], Train Losses: mse: 26.4214, mae: 2.8168, huber: 2.3985, swd: 12.1656, target_std: 20.3582
    Epoch [6/50], Val Losses: mse: 29.0973, mae: 3.4711, huber: 3.0490, swd: 11.9812, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 22.5811, mae: 3.0186, huber: 2.5851, swd: 11.3279, target_std: 18.3425
      Epoch 6 composite train-obj: 2.398480
            No improvement (3.0490), counter 4/5
    Epoch [7/50], Train Losses: mse: 26.0578, mae: 2.7803, huber: 2.3635, swd: 11.9255, target_std: 20.3589
    Epoch [7/50], Val Losses: mse: 26.2900, mae: 3.3098, huber: 2.8918, swd: 10.0672, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 22.1448, mae: 3.0045, huber: 2.5707, swd: 11.0992, target_std: 18.3425
      Epoch 7 composite train-obj: 2.363497
            Val objective improved 2.9061 → 2.8918, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 25.9517, mae: 2.7715, huber: 2.3548, swd: 11.8851, target_std: 20.3595
    Epoch [8/50], Val Losses: mse: 27.6693, mae: 3.3911, huber: 2.9751, swd: 11.3199, target_std: 20.4999
    Epoch [8/50], Test Losses: mse: 21.1332, mae: 2.8729, huber: 2.4452, swd: 10.2618, target_std: 18.3425
      Epoch 8 composite train-obj: 2.354800
            No improvement (2.9751), counter 1/5
    Epoch [9/50], Train Losses: mse: 25.8453, mae: 2.7613, huber: 2.3453, swd: 11.8620, target_std: 20.3592
    Epoch [9/50], Val Losses: mse: 28.1840, mae: 3.4808, huber: 3.0610, swd: 11.2623, target_std: 20.4999
    Epoch [9/50], Test Losses: mse: 23.6006, mae: 3.1279, huber: 2.6929, swd: 12.0294, target_std: 18.3425
      Epoch 9 composite train-obj: 2.345279
            No improvement (3.0610), counter 2/5
    Epoch [10/50], Train Losses: mse: 25.6503, mae: 2.7499, huber: 2.3339, swd: 11.7507, target_std: 20.3584
    Epoch [10/50], Val Losses: mse: 27.7927, mae: 3.4107, huber: 2.9939, swd: 11.3780, target_std: 20.4999
    Epoch [10/50], Test Losses: mse: 23.6841, mae: 3.0873, huber: 2.6536, swd: 12.3165, target_std: 18.3425
      Epoch 10 composite train-obj: 2.333893
            No improvement (2.9939), counter 3/5
    Epoch [11/50], Train Losses: mse: 25.4829, mae: 2.7358, huber: 2.3205, swd: 11.6385, target_std: 20.3591
    Epoch [11/50], Val Losses: mse: 25.8798, mae: 3.3039, huber: 2.8876, swd: 10.1011, target_std: 20.4999
    Epoch [11/50], Test Losses: mse: 21.6004, mae: 2.9053, huber: 2.4759, swd: 10.8722, target_std: 18.3425
      Epoch 11 composite train-obj: 2.320490
            Val objective improved 2.8918 → 2.8876, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 25.4162, mae: 2.7337, huber: 2.3183, swd: 11.6297, target_std: 20.3587
    Epoch [12/50], Val Losses: mse: 27.8939, mae: 3.4208, huber: 3.0044, swd: 11.5149, target_std: 20.4999
    Epoch [12/50], Test Losses: mse: 23.0290, mae: 3.0390, huber: 2.6073, swd: 11.8821, target_std: 18.3425
      Epoch 12 composite train-obj: 2.318345
            No improvement (3.0044), counter 1/5
    Epoch [13/50], Train Losses: mse: 25.2704, mae: 2.7218, huber: 2.3069, swd: 11.5377, target_std: 20.3588
    Epoch [13/50], Val Losses: mse: 28.2401, mae: 3.4293, huber: 3.0118, swd: 11.8130, target_std: 20.4999
    Epoch [13/50], Test Losses: mse: 24.1203, mae: 3.1131, huber: 2.6800, swd: 12.7669, target_std: 18.3425
      Epoch 13 composite train-obj: 2.306921
            No improvement (3.0118), counter 2/5
    Epoch [14/50], Train Losses: mse: 25.0431, mae: 2.7052, huber: 2.2909, swd: 11.3757, target_std: 20.3583
    Epoch [14/50], Val Losses: mse: 26.6819, mae: 3.3523, huber: 2.9364, swd: 10.3541, target_std: 20.4999
    Epoch [14/50], Test Losses: mse: 23.0069, mae: 3.0705, huber: 2.6357, swd: 11.8026, target_std: 18.3425
      Epoch 14 composite train-obj: 2.290935
            No improvement (2.9364), counter 3/5
    Epoch [15/50], Train Losses: mse: 24.9710, mae: 2.7036, huber: 2.2893, swd: 11.3585, target_std: 20.3588
    Epoch [15/50], Val Losses: mse: 27.2482, mae: 3.4040, huber: 2.9883, swd: 11.0379, target_std: 20.4999
    Epoch [15/50], Test Losses: mse: 23.5370, mae: 3.0697, huber: 2.6369, swd: 12.3327, target_std: 18.3425
      Epoch 15 composite train-obj: 2.289321
            No improvement (2.9883), counter 4/5
    Epoch [16/50], Train Losses: mse: 24.7929, mae: 2.6976, huber: 2.2832, swd: 11.2370, target_std: 20.3584
    Epoch [16/50], Val Losses: mse: 26.5505, mae: 3.3494, huber: 2.9336, swd: 10.5689, target_std: 20.4999
    Epoch [16/50], Test Losses: mse: 21.3172, mae: 2.8921, huber: 2.4618, swd: 10.4253, target_std: 18.3425
      Epoch 16 composite train-obj: 2.283248
    Epoch [16/50], Test Losses: mse: 21.6005, mae: 2.9054, huber: 2.4759, swd: 10.8723, target_std: 18.3425
    Best round's Test MSE: 21.6004, MAE: 2.9053, SWD: 10.8722
    Best round's Validation MSE: 25.8798, MAE: 3.3039
    Best round's Test verification MSE : 21.6005, MAE: 2.9054, SWD: 10.8723
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 53.8866, mae: 3.8527, huber: 3.4233, swd: 32.8315, target_std: 20.3593
    Epoch [1/50], Val Losses: mse: 28.0381, mae: 3.4226, huber: 2.9951, swd: 10.0473, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 21.6086, mae: 2.9821, huber: 2.5453, swd: 9.1955, target_std: 18.3425
      Epoch 1 composite train-obj: 3.423315
            Val objective improved inf → 2.9951, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 29.1257, mae: 2.9929, huber: 2.5701, swd: 12.8994, target_std: 20.3587
    Epoch [2/50], Val Losses: mse: 25.6048, mae: 3.2906, huber: 2.8674, swd: 8.3610, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 21.2577, mae: 2.9279, huber: 2.4928, swd: 9.2920, target_std: 18.3425
      Epoch 2 composite train-obj: 2.570119
            Val objective improved 2.9951 → 2.8674, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 27.9241, mae: 2.9244, huber: 2.5030, swd: 12.0833, target_std: 20.3590
    Epoch [3/50], Val Losses: mse: 26.0192, mae: 3.3376, huber: 2.9195, swd: 8.6677, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.9356, mae: 2.9257, huber: 2.4913, swd: 9.0575, target_std: 18.3425
      Epoch 3 composite train-obj: 2.502956
            No improvement (2.9195), counter 1/5
    Epoch [4/50], Train Losses: mse: 27.2671, mae: 2.8891, huber: 2.4689, swd: 11.7062, target_std: 20.3589
    Epoch [4/50], Val Losses: mse: 25.4513, mae: 3.3052, huber: 2.8872, swd: 8.4567, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 22.1931, mae: 3.0565, huber: 2.6176, swd: 10.1416, target_std: 18.3425
      Epoch 4 composite train-obj: 2.468923
            No improvement (2.8872), counter 2/5
    Epoch [5/50], Train Losses: mse: 26.7353, mae: 2.8424, huber: 2.4234, swd: 11.4117, target_std: 20.3584
    Epoch [5/50], Val Losses: mse: 26.4411, mae: 3.3614, huber: 2.9404, swd: 9.0293, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 22.4240, mae: 3.0423, huber: 2.6074, swd: 10.2074, target_std: 18.3425
      Epoch 5 composite train-obj: 2.423373
            No improvement (2.9404), counter 3/5
    Epoch [6/50], Train Losses: mse: 26.4049, mae: 2.8152, huber: 2.3967, swd: 11.2323, target_std: 20.3581
    Epoch [6/50], Val Losses: mse: 26.1069, mae: 3.3346, huber: 2.9159, swd: 8.8727, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 22.2490, mae: 3.0222, huber: 2.5878, swd: 10.1611, target_std: 18.3425
      Epoch 6 composite train-obj: 2.396691
            No improvement (2.9159), counter 4/5
    Epoch [7/50], Train Losses: mse: 26.1726, mae: 2.7918, huber: 2.3740, swd: 11.1174, target_std: 20.3586
    Epoch [7/50], Val Losses: mse: 25.6650, mae: 3.3151, huber: 2.8971, swd: 8.6743, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 21.7376, mae: 2.9971, huber: 2.5624, swd: 9.7930, target_std: 18.3425
      Epoch 7 composite train-obj: 2.373988
    Epoch [7/50], Test Losses: mse: 21.2577, mae: 2.9279, huber: 2.4928, swd: 9.2920, target_std: 18.3425
    Best round's Test MSE: 21.2577, MAE: 2.9279, SWD: 9.2920
    Best round's Validation MSE: 25.6048, MAE: 3.2906
    Best round's Test verification MSE : 21.2577, MAE: 2.9279, SWD: 9.2920
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 53.5495, mae: 3.8379, huber: 3.4078, swd: 36.3864, target_std: 20.3580
    Epoch [1/50], Val Losses: mse: 27.1599, mae: 3.3947, huber: 2.9666, swd: 11.2330, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.1099, mae: 2.8810, huber: 2.4467, swd: 9.4686, target_std: 18.3425
      Epoch 1 composite train-obj: 3.407826
            Val objective improved inf → 2.9666, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 29.3309, mae: 3.0153, huber: 2.5914, swd: 14.8832, target_std: 20.3583
    Epoch [2/50], Val Losses: mse: 26.0302, mae: 3.3168, huber: 2.8891, swd: 9.6539, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 22.0883, mae: 3.0131, huber: 2.5746, swd: 11.0764, target_std: 18.3425
      Epoch 2 composite train-obj: 2.591423
            Val objective improved 2.9666 → 2.8891, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 27.8607, mae: 2.9273, huber: 2.5055, swd: 13.6856, target_std: 20.3582
    Epoch [3/50], Val Losses: mse: 26.2355, mae: 3.3979, huber: 2.9744, swd: 10.2794, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 19.4865, mae: 2.8115, huber: 2.3801, swd: 9.1271, target_std: 18.3425
      Epoch 3 composite train-obj: 2.505523
            No improvement (2.9744), counter 1/5
    Epoch [4/50], Train Losses: mse: 27.1849, mae: 2.8771, huber: 2.4567, swd: 13.2700, target_std: 20.3599
    Epoch [4/50], Val Losses: mse: 26.9468, mae: 3.4050, huber: 2.9817, swd: 10.5193, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 21.1006, mae: 2.9444, huber: 2.5076, swd: 10.4403, target_std: 18.3425
      Epoch 4 composite train-obj: 2.456676
            No improvement (2.9817), counter 2/5
    Epoch [5/50], Train Losses: mse: 26.7019, mae: 2.8362, huber: 2.4164, swd: 13.0067, target_std: 20.3590
    Epoch [5/50], Val Losses: mse: 26.3337, mae: 3.3685, huber: 2.9474, swd: 10.1627, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 20.9569, mae: 2.9437, huber: 2.5083, swd: 10.3550, target_std: 18.3425
      Epoch 5 composite train-obj: 2.416427
            No improvement (2.9474), counter 3/5
    Epoch [6/50], Train Losses: mse: 26.5561, mae: 2.8256, huber: 2.4065, swd: 12.9177, target_std: 20.3588
    Epoch [6/50], Val Losses: mse: 25.9631, mae: 3.3454, huber: 2.9237, swd: 10.1921, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 20.4628, mae: 2.9209, huber: 2.4839, swd: 9.9344, target_std: 18.3425
      Epoch 6 composite train-obj: 2.406466
            No improvement (2.9237), counter 4/5
    Epoch [7/50], Train Losses: mse: 26.2917, mae: 2.8002, huber: 2.3818, swd: 12.7661, target_std: 20.3587
    Epoch [7/50], Val Losses: mse: 26.1743, mae: 3.3268, huber: 2.9058, swd: 10.1900, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 21.6050, mae: 2.9894, huber: 2.5523, swd: 10.9382, target_std: 18.3425
      Epoch 7 composite train-obj: 2.381760
    Epoch [7/50], Test Losses: mse: 22.0881, mae: 3.0131, huber: 2.5745, swd: 11.0762, target_std: 18.3425
    Best round's Test MSE: 22.0883, MAE: 3.0131, SWD: 11.0764
    Best round's Validation MSE: 26.0302, MAE: 3.3168
    Best round's Test verification MSE : 22.0881, MAE: 3.0131, SWD: 11.0762
    
    ==================================================
    Experiment Summary (ACL_ettm2_seq96_pred720_20250430_1949)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 21.6488 ± 0.3408
      mae: 2.9488 ± 0.0464
      huber: 2.5144 ± 0.0431
      swd: 10.4136 ± 0.7974
      target_std: 18.3425 ± 0.0000
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 25.8383 ± 0.1762
      mae: 3.3037 ± 0.0107
      huber: 2.8814 ± 0.0099
      swd: 9.3720 ± 0.7378
      target_std: 20.4999 ± 0.0000
      count: 49.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm2_seq96_pred720_20250430_1949
    Model: ACL
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### Timemixer

#### pred=96


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm2']['channels'],
    enc_in=data_mgr.datasets['ettm2']['channels'],
    dec_in=data_mgr.datasets['ettm2']['channels'],
    c_out=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 380
    Validation Batches: 53
    Test Batches: 108
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.9658, mae: 1.9576, huber: 1.5688, swd: 7.0351, target_std: 20.3268
    Epoch [1/50], Val Losses: mse: 12.9541, mae: 2.1268, huber: 1.7394, swd: 6.9012, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 9.8838, mae: 1.9193, huber: 1.5167, swd: 5.0030, target_std: 18.4467
      Epoch 1 composite train-obj: 1.568791
            Val objective improved inf → 1.7394, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.6397, mae: 1.8321, huber: 1.4479, swd: 6.4388, target_std: 20.3275
    Epoch [2/50], Val Losses: mse: 12.8915, mae: 2.1184, huber: 1.7307, swd: 6.8398, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.6689, mae: 1.8863, huber: 1.4850, swd: 4.7873, target_std: 18.4467
      Epoch 2 composite train-obj: 1.447902
            Val objective improved 1.7394 → 1.7307, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.3187, mae: 1.8037, huber: 1.4203, swd: 6.2791, target_std: 20.3268
    Epoch [3/50], Val Losses: mse: 12.6120, mae: 2.0963, huber: 1.7094, swd: 6.7216, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.5624, mae: 1.8789, huber: 1.4771, swd: 4.8285, target_std: 18.4467
      Epoch 3 composite train-obj: 1.420276
            Val objective improved 1.7307 → 1.7094, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.1131, mae: 1.7862, huber: 1.4034, swd: 6.1645, target_std: 20.3282
    Epoch [4/50], Val Losses: mse: 12.7118, mae: 2.1083, huber: 1.7224, swd: 6.8034, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.6394, mae: 1.8804, huber: 1.4792, swd: 4.9153, target_std: 18.4467
      Epoch 4 composite train-obj: 1.403368
            No improvement (1.7224), counter 1/5
    Epoch [5/50], Train Losses: mse: 10.9636, mae: 1.7734, huber: 1.3910, swd: 6.0753, target_std: 20.3262
    Epoch [5/50], Val Losses: mse: 12.5246, mae: 2.0918, huber: 1.7058, swd: 6.7284, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.5698, mae: 1.8763, huber: 1.4748, swd: 4.9115, target_std: 18.4467
      Epoch 5 composite train-obj: 1.390985
            Val objective improved 1.7094 → 1.7058, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.8352, mae: 1.7626, huber: 1.3805, swd: 5.9943, target_std: 20.3257
    Epoch [6/50], Val Losses: mse: 12.6396, mae: 2.0959, huber: 1.7094, swd: 6.8214, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.4213, mae: 1.8627, huber: 1.4608, swd: 4.7713, target_std: 18.4467
      Epoch 6 composite train-obj: 1.380531
            No improvement (1.7094), counter 1/5
    Epoch [7/50], Train Losses: mse: 10.7206, mae: 1.7536, huber: 1.3717, swd: 5.9199, target_std: 20.3264
    Epoch [7/50], Val Losses: mse: 12.5676, mae: 2.0903, huber: 1.7045, swd: 6.7391, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.4601, mae: 1.8581, huber: 1.4571, swd: 4.7673, target_std: 18.4467
      Epoch 7 composite train-obj: 1.371706
            Val objective improved 1.7058 → 1.7045, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.6222, mae: 1.7454, huber: 1.3638, swd: 5.8553, target_std: 20.3282
    Epoch [8/50], Val Losses: mse: 12.6403, mae: 2.1000, huber: 1.7137, swd: 6.8095, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.5146, mae: 1.8661, huber: 1.4648, swd: 4.8172, target_std: 18.4467
      Epoch 8 composite train-obj: 1.363832
            No improvement (1.7137), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.5057, mae: 1.7365, huber: 1.3552, swd: 5.7667, target_std: 20.3265
    Epoch [9/50], Val Losses: mse: 12.5572, mae: 2.0939, huber: 1.7080, swd: 6.7343, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.5268, mae: 1.8664, huber: 1.4649, swd: 4.8603, target_std: 18.4467
      Epoch 9 composite train-obj: 1.355194
            No improvement (1.7080), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.4429, mae: 1.7315, huber: 1.3502, swd: 5.7268, target_std: 20.3257
    Epoch [10/50], Val Losses: mse: 12.5530, mae: 2.0949, huber: 1.7088, swd: 6.7263, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.4461, mae: 1.8629, huber: 1.4612, swd: 4.7822, target_std: 18.4467
      Epoch 10 composite train-obj: 1.350204
            No improvement (1.7088), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.3620, mae: 1.7256, huber: 1.3445, swd: 5.6681, target_std: 20.3276
    Epoch [11/50], Val Losses: mse: 12.7528, mae: 2.1099, huber: 1.7230, swd: 6.9205, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.5174, mae: 1.8674, huber: 1.4655, swd: 4.8191, target_std: 18.4467
      Epoch 11 composite train-obj: 1.344537
            No improvement (1.7230), counter 4/5
    Epoch [12/50], Train Losses: mse: 10.3026, mae: 1.7192, huber: 1.3382, swd: 5.6254, target_std: 20.3281
    Epoch [12/50], Val Losses: mse: 12.8008, mae: 2.1211, huber: 1.7339, swd: 6.9211, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 9.5667, mae: 1.8719, huber: 1.4697, swd: 4.8564, target_std: 18.4467
      Epoch 12 composite train-obj: 1.338234
    Epoch [12/50], Test Losses: mse: 9.4601, mae: 1.8581, huber: 1.4571, swd: 4.7673, target_std: 18.4467
    Best round's Test MSE: 9.4601, MAE: 1.8581, SWD: 4.7673
    Best round's Validation MSE: 12.5676, MAE: 2.0903
    Best round's Test verification MSE : 9.4601, MAE: 1.8581, SWD: 4.7673
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.1307, mae: 1.9728, huber: 1.5833, swd: 6.8762, target_std: 20.3263
    Epoch [1/50], Val Losses: mse: 13.0159, mae: 2.1376, huber: 1.7497, swd: 6.6891, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 9.7784, mae: 1.9114, huber: 1.5086, swd: 4.6822, target_std: 18.4467
      Epoch 1 composite train-obj: 1.583333
            Val objective improved inf → 1.7497, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.8154, mae: 1.8450, huber: 1.4604, swd: 6.2924, target_std: 20.3264
    Epoch [2/50], Val Losses: mse: 12.9024, mae: 2.1231, huber: 1.7360, swd: 6.6338, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.6977, mae: 1.9031, huber: 1.5002, swd: 4.6679, target_std: 18.4467
      Epoch 2 composite train-obj: 1.460429
            Val objective improved 1.7497 → 1.7360, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.5147, mae: 1.8186, huber: 1.4347, swd: 6.1674, target_std: 20.3270
    Epoch [3/50], Val Losses: mse: 12.7577, mae: 2.1085, huber: 1.7219, swd: 6.5783, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.6121, mae: 1.8926, huber: 1.4904, swd: 4.6382, target_std: 18.4467
      Epoch 3 composite train-obj: 1.434675
            Val objective improved 1.7360 → 1.7219, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.2898, mae: 1.7997, huber: 1.4164, swd: 6.0607, target_std: 20.3265
    Epoch [4/50], Val Losses: mse: 12.6289, mae: 2.0996, huber: 1.7135, swd: 6.5288, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.6008, mae: 1.8865, huber: 1.4848, swd: 4.6662, target_std: 18.4467
      Epoch 4 composite train-obj: 1.416412
            Val objective improved 1.7219 → 1.7135, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.1069, mae: 1.7847, huber: 1.4019, swd: 5.9629, target_std: 20.3264
    Epoch [5/50], Val Losses: mse: 12.5211, mae: 2.0964, huber: 1.7100, swd: 6.4747, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.6021, mae: 1.8912, huber: 1.4887, swd: 4.7181, target_std: 18.4467
      Epoch 5 composite train-obj: 1.401887
            Val objective improved 1.7135 → 1.7100, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.9616, mae: 1.7728, huber: 1.3903, swd: 5.8863, target_std: 20.3273
    Epoch [6/50], Val Losses: mse: 12.5453, mae: 2.0969, huber: 1.7103, swd: 6.4805, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.5556, mae: 1.8800, huber: 1.4780, swd: 4.6520, target_std: 18.4467
      Epoch 6 composite train-obj: 1.390290
            No improvement (1.7103), counter 1/5
    Epoch [7/50], Train Losses: mse: 10.8436, mae: 1.7632, huber: 1.3811, swd: 5.8193, target_std: 20.3269
    Epoch [7/50], Val Losses: mse: 12.5445, mae: 2.0986, huber: 1.7120, swd: 6.4291, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.6643, mae: 1.8794, huber: 1.4781, swd: 4.6989, target_std: 18.4467
      Epoch 7 composite train-obj: 1.381094
            No improvement (1.7120), counter 2/5
    Epoch [8/50], Train Losses: mse: 10.7217, mae: 1.7530, huber: 1.3712, swd: 5.7375, target_std: 20.3261
    Epoch [8/50], Val Losses: mse: 12.6484, mae: 2.1062, huber: 1.7191, swd: 6.5293, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.5875, mae: 1.8721, huber: 1.4704, swd: 4.6141, target_std: 18.4467
      Epoch 8 composite train-obj: 1.371237
            No improvement (1.7191), counter 3/5
    Epoch [9/50], Train Losses: mse: 10.6306, mae: 1.7461, huber: 1.3645, swd: 5.6811, target_std: 20.3260
    Epoch [9/50], Val Losses: mse: 12.5765, mae: 2.1068, huber: 1.7204, swd: 6.4706, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.6222, mae: 1.8808, huber: 1.4788, swd: 4.6702, target_std: 18.4467
      Epoch 9 composite train-obj: 1.364482
            No improvement (1.7204), counter 4/5
    Epoch [10/50], Train Losses: mse: 10.5622, mae: 1.7387, huber: 1.3574, swd: 5.6328, target_std: 20.3287
    Epoch [10/50], Val Losses: mse: 12.6081, mae: 2.1047, huber: 1.7191, swd: 6.5119, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.5767, mae: 1.8767, huber: 1.4749, swd: 4.6490, target_std: 18.4467
      Epoch 10 composite train-obj: 1.357413
    Epoch [10/50], Test Losses: mse: 9.6021, mae: 1.8912, huber: 1.4887, swd: 4.7181, target_std: 18.4467
    Best round's Test MSE: 9.6021, MAE: 1.8912, SWD: 4.7181
    Best round's Validation MSE: 12.5211, MAE: 2.0964
    Best round's Test verification MSE : 9.6021, MAE: 1.8912, SWD: 4.7181
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.2150, mae: 1.9790, huber: 1.5896, swd: 6.5526, target_std: 20.3280
    Epoch [1/50], Val Losses: mse: 13.1677, mae: 2.1483, huber: 1.7594, swd: 6.3478, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 9.8758, mae: 1.9149, huber: 1.5121, swd: 4.3832, target_std: 18.4467
      Epoch 1 composite train-obj: 1.589627
            Val objective improved inf → 1.7594, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.6516, mae: 1.8313, huber: 1.4470, swd: 5.8314, target_std: 20.3269
    Epoch [2/50], Val Losses: mse: 12.8251, mae: 2.1177, huber: 1.7296, swd: 6.1950, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.7263, mae: 1.9084, huber: 1.5048, swd: 4.4678, target_std: 18.4467
      Epoch 2 composite train-obj: 1.447011
            Val objective improved 1.7594 → 1.7296, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.3488, mae: 1.8045, huber: 1.4209, swd: 5.6950, target_std: 20.3272
    Epoch [3/50], Val Losses: mse: 12.6604, mae: 2.1116, huber: 1.7223, swd: 6.0813, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.5789, mae: 1.8879, huber: 1.4849, swd: 4.3869, target_std: 18.4467
      Epoch 3 composite train-obj: 1.420875
            Val objective improved 1.7296 → 1.7223, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.1495, mae: 1.7891, huber: 1.4058, swd: 5.5909, target_std: 20.3269
    Epoch [4/50], Val Losses: mse: 12.5668, mae: 2.1016, huber: 1.7139, swd: 6.0603, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.6920, mae: 1.8856, huber: 1.4844, swd: 4.4677, target_std: 18.4467
      Epoch 4 composite train-obj: 1.405784
            Val objective improved 1.7223 → 1.7139, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.9983, mae: 1.7759, huber: 1.3930, swd: 5.5063, target_std: 20.3261
    Epoch [5/50], Val Losses: mse: 12.6354, mae: 2.1049, huber: 1.7162, swd: 6.1149, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.5684, mae: 1.8799, huber: 1.4779, swd: 4.3787, target_std: 18.4467
      Epoch 5 composite train-obj: 1.393034
            No improvement (1.7162), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.8600, mae: 1.7648, huber: 1.3822, swd: 5.4310, target_std: 20.3266
    Epoch [6/50], Val Losses: mse: 12.5928, mae: 2.1030, huber: 1.7144, swd: 6.0925, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.5712, mae: 1.8690, huber: 1.4679, swd: 4.3771, target_std: 18.4467
      Epoch 6 composite train-obj: 1.382182
            No improvement (1.7144), counter 2/5
    Epoch [7/50], Train Losses: mse: 10.7603, mae: 1.7552, huber: 1.3729, swd: 5.3753, target_std: 20.3275
    Epoch [7/50], Val Losses: mse: 12.5866, mae: 2.1007, huber: 1.7122, swd: 6.1096, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.5219, mae: 1.8727, huber: 1.4705, swd: 4.3676, target_std: 18.4467
      Epoch 7 composite train-obj: 1.372875
            Val objective improved 1.7139 → 1.7122, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.6744, mae: 1.7493, huber: 1.3671, swd: 5.3229, target_std: 20.3256
    Epoch [8/50], Val Losses: mse: 12.5774, mae: 2.1012, huber: 1.7128, swd: 6.1052, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.5502, mae: 1.8752, huber: 1.4728, swd: 4.3969, target_std: 18.4467
      Epoch 8 composite train-obj: 1.367124
            No improvement (1.7128), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.6025, mae: 1.7409, huber: 1.3589, swd: 5.2804, target_std: 20.3268
    Epoch [9/50], Val Losses: mse: 12.7698, mae: 2.1149, huber: 1.7258, swd: 6.2257, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.5276, mae: 1.8696, huber: 1.4672, swd: 4.3380, target_std: 18.4467
      Epoch 9 composite train-obj: 1.358937
            No improvement (1.7258), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.5067, mae: 1.7334, huber: 1.3517, swd: 5.2195, target_std: 20.3266
    Epoch [10/50], Val Losses: mse: 12.6778, mae: 2.1077, huber: 1.7196, swd: 6.1689, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.5878, mae: 1.8698, huber: 1.4682, swd: 4.3874, target_std: 18.4467
      Epoch 10 composite train-obj: 1.351683
            No improvement (1.7196), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.4500, mae: 1.7279, huber: 1.3464, swd: 5.1882, target_std: 20.3266
    Epoch [11/50], Val Losses: mse: 12.6903, mae: 2.1114, huber: 1.7232, swd: 6.1997, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.5115, mae: 1.8682, huber: 1.4663, swd: 4.3530, target_std: 18.4467
      Epoch 11 composite train-obj: 1.346376
            No improvement (1.7232), counter 4/5
    Epoch [12/50], Train Losses: mse: 10.3675, mae: 1.7208, huber: 1.3395, swd: 5.1310, target_std: 20.3270
    Epoch [12/50], Val Losses: mse: 12.9997, mae: 2.1333, huber: 1.7448, swd: 6.4306, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 9.6600, mae: 1.8782, huber: 1.4762, swd: 4.4270, target_std: 18.4467
      Epoch 12 composite train-obj: 1.339478
    Epoch [12/50], Test Losses: mse: 9.5219, mae: 1.8727, huber: 1.4705, swd: 4.3676, target_std: 18.4467
    Best round's Test MSE: 9.5219, MAE: 1.8727, SWD: 4.3676
    Best round's Validation MSE: 12.5866, MAE: 2.1007
    Best round's Test verification MSE : 9.5219, MAE: 1.8727, SWD: 4.3676
    
    ==================================================
    Experiment Summary (TimeMixer_ettm2_seq96_pred96_20250430_1853)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 9.5280 ± 0.0582
      mae: 1.8740 ± 0.0135
      huber: 1.4721 ± 0.0129
      swd: 4.6177 ± 0.1780
      target_std: 18.4467 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 12.5584 ± 0.0275
      mae: 2.0958 ± 0.0043
      huber: 1.7089 ± 0.0032
      swd: 6.4411 ± 0.2581
      target_std: 20.5171 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: TimeMixer_ettm2_seq96_pred96_20250430_1853
    Model: TimeMixer
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm2']['channels'],
    enc_in=data_mgr.datasets['ettm2']['channels'],
    dec_in=data_mgr.datasets['ettm2']['channels'],
    c_out=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 379
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 379
    Validation Batches: 53
    Test Batches: 107
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 18.8136, mae: 2.3145, huber: 1.9180, swd: 10.5023, target_std: 20.3332
    Epoch [1/50], Val Losses: mse: 19.0712, mae: 2.5710, huber: 2.1736, swd: 10.6877, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 12.6381, mae: 2.1866, huber: 1.7754, swd: 6.2307, target_std: 18.4014
      Epoch 1 composite train-obj: 1.918037
            Val objective improved inf → 2.1736, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.4201, mae: 2.1984, huber: 1.8055, swd: 9.9437, target_std: 20.3332
    Epoch [2/50], Val Losses: mse: 18.8675, mae: 2.5605, huber: 2.1629, swd: 10.5802, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.6454, mae: 2.1804, huber: 1.7692, swd: 6.3103, target_std: 18.4014
      Epoch 2 composite train-obj: 1.805531
            Val objective improved 2.1736 → 2.1629, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 17.0907, mae: 2.1744, huber: 1.7819, swd: 9.7459, target_std: 20.3336
    Epoch [3/50], Val Losses: mse: 18.7149, mae: 2.5469, huber: 2.1493, swd: 10.4465, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.4514, mae: 2.1621, huber: 1.7511, swd: 6.1641, target_std: 18.4014
      Epoch 3 composite train-obj: 1.781852
            Val objective improved 2.1629 → 2.1493, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 16.8072, mae: 2.1539, huber: 1.7616, swd: 9.5654, target_std: 20.3336
    Epoch [4/50], Val Losses: mse: 18.5868, mae: 2.5453, huber: 2.1470, swd: 10.3669, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.5559, mae: 2.1679, huber: 1.7566, swd: 6.3027, target_std: 18.4014
      Epoch 4 composite train-obj: 1.761575
            Val objective improved 2.1493 → 2.1470, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 16.5724, mae: 2.1383, huber: 1.7462, swd: 9.4113, target_std: 20.3336
    Epoch [5/50], Val Losses: mse: 18.6033, mae: 2.5468, huber: 2.1483, swd: 10.3588, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.4931, mae: 2.1577, huber: 1.7464, swd: 6.2101, target_std: 18.4014
      Epoch 5 composite train-obj: 1.746176
            No improvement (2.1483), counter 1/5
    Epoch [6/50], Train Losses: mse: 16.3893, mae: 2.1250, huber: 1.7330, swd: 9.2797, target_std: 20.3338
    Epoch [6/50], Val Losses: mse: 18.4391, mae: 2.5370, huber: 2.1384, swd: 10.2900, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.5036, mae: 2.1630, huber: 1.7512, swd: 6.2231, target_std: 18.4014
      Epoch 6 composite train-obj: 1.733034
            Val objective improved 2.1470 → 2.1384, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 16.2005, mae: 2.1129, huber: 1.7211, swd: 9.1314, target_std: 20.3335
    Epoch [7/50], Val Losses: mse: 18.5136, mae: 2.5438, huber: 2.1448, swd: 10.3331, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 12.6118, mae: 2.1712, huber: 1.7586, swd: 6.3551, target_std: 18.4014
      Epoch 7 composite train-obj: 1.721106
            No improvement (2.1448), counter 1/5
    Epoch [8/50], Train Losses: mse: 16.0717, mae: 2.1040, huber: 1.7123, swd: 9.0387, target_std: 20.3342
    Epoch [8/50], Val Losses: mse: 18.6211, mae: 2.5565, huber: 2.1578, swd: 10.4535, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 12.5917, mae: 2.1739, huber: 1.7612, swd: 6.3356, target_std: 18.4014
      Epoch 8 composite train-obj: 1.712310
            No improvement (2.1578), counter 2/5
    Epoch [9/50], Train Losses: mse: 15.9020, mae: 2.0938, huber: 1.7023, swd: 8.9045, target_std: 20.3331
    Epoch [9/50], Val Losses: mse: 18.6878, mae: 2.5545, huber: 2.1559, swd: 10.4656, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 12.7178, mae: 2.1736, huber: 1.7615, swd: 6.4004, target_std: 18.4014
      Epoch 9 composite train-obj: 1.702341
            No improvement (2.1559), counter 3/5
    Epoch [10/50], Train Losses: mse: 15.7717, mae: 2.0849, huber: 1.6936, swd: 8.8010, target_std: 20.3339
    Epoch [10/50], Val Losses: mse: 18.8225, mae: 2.5771, huber: 2.1781, swd: 10.6088, target_std: 20.5407
    Epoch [10/50], Test Losses: mse: 13.1730, mae: 2.2147, huber: 1.8015, swd: 6.8111, target_std: 18.4014
      Epoch 10 composite train-obj: 1.693552
            No improvement (2.1781), counter 4/5
    Epoch [11/50], Train Losses: mse: 15.6397, mae: 2.0776, huber: 1.6864, swd: 8.6997, target_std: 20.3340
    Epoch [11/50], Val Losses: mse: 18.7819, mae: 2.5694, huber: 2.1706, swd: 10.5775, target_std: 20.5407
    Epoch [11/50], Test Losses: mse: 12.9838, mae: 2.1930, huber: 1.7807, swd: 6.6394, target_std: 18.4014
      Epoch 11 composite train-obj: 1.686397
    Epoch [11/50], Test Losses: mse: 12.5036, mae: 2.1630, huber: 1.7512, swd: 6.2231, target_std: 18.4014
    Best round's Test MSE: 12.5036, MAE: 2.1630, SWD: 6.2231
    Best round's Validation MSE: 18.4391, MAE: 2.5370
    Best round's Test verification MSE : 12.5036, MAE: 2.1630, SWD: 6.2231
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 18.5559, mae: 2.2988, huber: 1.9025, swd: 10.8642, target_std: 20.3333
    Epoch [1/50], Val Losses: mse: 19.0961, mae: 2.5736, huber: 2.1757, swd: 11.0588, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 12.5941, mae: 2.1792, huber: 1.7685, swd: 6.3760, target_std: 18.4014
      Epoch 1 composite train-obj: 1.902517
            Val objective improved inf → 2.1757, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.3561, mae: 2.1932, huber: 1.8002, swd: 10.2522, target_std: 20.3337
    Epoch [2/50], Val Losses: mse: 18.8688, mae: 2.5660, huber: 2.1658, swd: 10.8614, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.5488, mae: 2.1706, huber: 1.7591, swd: 6.3762, target_std: 18.4014
      Epoch 2 composite train-obj: 1.800198
            Val objective improved 2.1757 → 2.1658, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 17.0479, mae: 2.1702, huber: 1.7775, swd: 10.0695, target_std: 20.3338
    Epoch [3/50], Val Losses: mse: 18.7597, mae: 2.5510, huber: 2.1532, swd: 10.8970, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.5156, mae: 2.1681, huber: 1.7572, swd: 6.4227, target_std: 18.4014
      Epoch 3 composite train-obj: 1.777483
            Val objective improved 2.1658 → 2.1532, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 16.8219, mae: 2.1534, huber: 1.7609, swd: 9.9237, target_std: 20.3335
    Epoch [4/50], Val Losses: mse: 18.6139, mae: 2.5446, huber: 2.1459, swd: 10.7395, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.4739, mae: 2.1559, huber: 1.7454, swd: 6.3302, target_std: 18.4014
      Epoch 4 composite train-obj: 1.760937
            Val objective improved 2.1532 → 2.1459, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 16.6414, mae: 2.1403, huber: 1.7481, swd: 9.7982, target_std: 20.3333
    Epoch [5/50], Val Losses: mse: 18.6696, mae: 2.5566, huber: 2.1575, swd: 10.8015, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.6666, mae: 2.1723, huber: 1.7615, swd: 6.5487, target_std: 18.4014
      Epoch 5 composite train-obj: 1.748063
            No improvement (2.1575), counter 1/5
    Epoch [6/50], Train Losses: mse: 16.4616, mae: 2.1283, huber: 1.7363, swd: 9.6745, target_std: 20.3335
    Epoch [6/50], Val Losses: mse: 18.4405, mae: 2.5335, huber: 2.1351, swd: 10.6730, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.3422, mae: 2.1519, huber: 1.7405, swd: 6.3118, target_std: 18.4014
      Epoch 6 composite train-obj: 1.736274
            Val objective improved 2.1459 → 2.1351, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 16.2959, mae: 2.1166, huber: 1.7248, swd: 9.5486, target_std: 20.3337
    Epoch [7/50], Val Losses: mse: 18.5794, mae: 2.5556, huber: 2.1559, swd: 10.7258, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 12.5199, mae: 2.1726, huber: 1.7599, swd: 6.4393, target_std: 18.4014
      Epoch 7 composite train-obj: 1.724789
            No improvement (2.1559), counter 1/5
    Epoch [8/50], Train Losses: mse: 16.1176, mae: 2.1054, huber: 1.7137, swd: 9.4089, target_std: 20.3332
    Epoch [8/50], Val Losses: mse: 18.6295, mae: 2.5493, huber: 2.1510, swd: 10.7896, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 12.4998, mae: 2.1536, huber: 1.7425, swd: 6.4147, target_std: 18.4014
      Epoch 8 composite train-obj: 1.713736
            No improvement (2.1510), counter 2/5
    Epoch [9/50], Train Losses: mse: 15.9737, mae: 2.0948, huber: 1.7034, swd: 9.2861, target_std: 20.3339
    Epoch [9/50], Val Losses: mse: 18.7242, mae: 2.5643, huber: 2.1648, swd: 10.8071, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 12.6683, mae: 2.1691, huber: 1.7571, swd: 6.5405, target_std: 18.4014
      Epoch 9 composite train-obj: 1.703403
            No improvement (2.1648), counter 3/5
    Epoch [10/50], Train Losses: mse: 15.8150, mae: 2.0849, huber: 1.6937, swd: 9.1624, target_std: 20.3339
    Epoch [10/50], Val Losses: mse: 18.7370, mae: 2.5688, huber: 2.1696, swd: 10.8537, target_std: 20.5407
    Epoch [10/50], Test Losses: mse: 13.0274, mae: 2.1983, huber: 1.7855, swd: 6.8674, target_std: 18.4014
      Epoch 10 composite train-obj: 1.693663
            No improvement (2.1696), counter 4/5
    Epoch [11/50], Train Losses: mse: 15.6574, mae: 2.0751, huber: 1.6840, swd: 9.0200, target_std: 20.3337
    Epoch [11/50], Val Losses: mse: 18.7038, mae: 2.5613, huber: 2.1623, swd: 10.8202, target_std: 20.5407
    Epoch [11/50], Test Losses: mse: 12.8370, mae: 2.1743, huber: 1.7624, swd: 6.6935, target_std: 18.4014
      Epoch 11 composite train-obj: 1.684015
    Epoch [11/50], Test Losses: mse: 12.3422, mae: 2.1519, huber: 1.7405, swd: 6.3118, target_std: 18.4014
    Best round's Test MSE: 12.3422, MAE: 2.1519, SWD: 6.3118
    Best round's Validation MSE: 18.4405, MAE: 2.5335
    Best round's Test verification MSE : 12.3422, MAE: 2.1519, SWD: 6.3118
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 18.4824, mae: 2.2829, huber: 1.8875, swd: 9.2265, target_std: 20.3332
    Epoch [1/50], Val Losses: mse: 19.1600, mae: 2.5756, huber: 2.1787, swd: 9.4167, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 12.6878, mae: 2.1848, huber: 1.7740, swd: 5.4588, target_std: 18.4014
      Epoch 1 composite train-obj: 1.887453
            Val objective improved inf → 2.1787, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.4549, mae: 2.1974, huber: 1.8047, swd: 8.7106, target_std: 20.3341
    Epoch [2/50], Val Losses: mse: 18.9865, mae: 2.5636, huber: 2.1669, swd: 9.3895, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.5199, mae: 2.1707, huber: 1.7600, swd: 5.4289, target_std: 18.4014
      Epoch 2 composite train-obj: 1.804699
            Val objective improved 2.1787 → 2.1669, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 17.1072, mae: 2.1729, huber: 1.7807, swd: 8.5396, target_std: 20.3339
    Epoch [3/50], Val Losses: mse: 18.7908, mae: 2.5592, huber: 2.1620, swd: 9.2177, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.5584, mae: 2.1658, huber: 1.7553, swd: 5.4579, target_std: 18.4014
      Epoch 3 composite train-obj: 1.780651
            Val objective improved 2.1669 → 2.1620, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 16.8135, mae: 2.1532, huber: 1.7612, swd: 8.3805, target_std: 20.3339
    Epoch [4/50], Val Losses: mse: 18.5920, mae: 2.5441, huber: 2.1465, swd: 9.1552, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.4576, mae: 2.1604, huber: 1.7493, swd: 5.4730, target_std: 18.4014
      Epoch 4 composite train-obj: 1.761234
            Val objective improved 2.1620 → 2.1465, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 16.5822, mae: 2.1373, huber: 1.7454, swd: 8.2551, target_std: 20.3334
    Epoch [5/50], Val Losses: mse: 18.5802, mae: 2.5441, huber: 2.1468, swd: 9.1077, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.4827, mae: 2.1585, huber: 1.7479, swd: 5.4225, target_std: 18.4014
      Epoch 5 composite train-obj: 1.745422
            No improvement (2.1468), counter 1/5
    Epoch [6/50], Train Losses: mse: 16.4011, mae: 2.1240, huber: 1.7323, swd: 8.1466, target_std: 20.3339
    Epoch [6/50], Val Losses: mse: 18.6388, mae: 2.5555, huber: 2.1571, swd: 9.1384, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.7277, mae: 2.1747, huber: 1.7634, swd: 5.5927, target_std: 18.4014
      Epoch 6 composite train-obj: 1.732298
            No improvement (2.1571), counter 2/5
    Epoch [7/50], Train Losses: mse: 16.2363, mae: 2.1123, huber: 1.7207, swd: 8.0414, target_std: 20.3336
    Epoch [7/50], Val Losses: mse: 18.8019, mae: 2.5696, huber: 2.1713, swd: 9.2594, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 12.8458, mae: 2.1785, huber: 1.7673, swd: 5.7062, target_std: 18.4014
      Epoch 7 composite train-obj: 1.720675
            No improvement (2.1713), counter 3/5
    Epoch [8/50], Train Losses: mse: 16.1308, mae: 2.1040, huber: 1.7125, swd: 7.9799, target_std: 20.3334
    Epoch [8/50], Val Losses: mse: 18.7374, mae: 2.5621, huber: 2.1636, swd: 9.2078, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 12.7068, mae: 2.1704, huber: 1.7588, swd: 5.6070, target_std: 18.4014
      Epoch 8 composite train-obj: 1.712507
            No improvement (2.1636), counter 4/5
    Epoch [9/50], Train Losses: mse: 15.9770, mae: 2.0947, huber: 1.7034, swd: 7.8791, target_std: 20.3341
    Epoch [9/50], Val Losses: mse: 18.8551, mae: 2.5770, huber: 2.1784, swd: 9.3076, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 12.9349, mae: 2.1936, huber: 1.7814, swd: 5.8056, target_std: 18.4014
      Epoch 9 composite train-obj: 1.703406
    Epoch [9/50], Test Losses: mse: 12.4576, mae: 2.1604, huber: 1.7493, swd: 5.4730, target_std: 18.4014
    Best round's Test MSE: 12.4576, MAE: 2.1604, SWD: 5.4730
    Best round's Validation MSE: 18.5920, MAE: 2.5441
    Best round's Test verification MSE : 12.4576, MAE: 2.1604, SWD: 5.4730
    
    ==================================================
    Experiment Summary (TimeMixer_ettm2_seq96_pred196_20250430_1913)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.4345 ± 0.0679
      mae: 2.1584 ± 0.0047
      huber: 1.7470 ± 0.0047
      swd: 6.0027 ± 0.3763
      target_std: 18.4014 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 18.4905 ± 0.0718
      mae: 2.5382 ± 0.0044
      huber: 2.1400 ± 0.0048
      swd: 10.0394 ± 0.6445
      target_std: 20.5407 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: TimeMixer_ettm2_seq96_pred196_20250430_1913
    Model: TimeMixer
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm2']['channels'],
    enc_in=data_mgr.datasets['ettm2']['channels'],
    dec_in=data_mgr.datasets['ettm2']['channels'],
    c_out=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 378
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 378
    Validation Batches: 52
    Test Batches: 106
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.8516, mae: 2.6356, huber: 2.2326, swd: 13.4375, target_std: 20.3416
    Epoch [1/50], Val Losses: mse: 23.7308, mae: 2.9288, huber: 2.5248, swd: 12.2321, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 15.8591, mae: 2.4382, huber: 2.0221, swd: 7.7229, target_std: 18.3689
      Epoch 1 composite train-obj: 2.232647
            Val objective improved inf → 2.5248, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.3773, mae: 2.5197, huber: 2.1200, swd: 13.0077, target_std: 20.3421
    Epoch [2/50], Val Losses: mse: 23.4416, mae: 2.9171, huber: 2.5121, swd: 12.0237, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 15.7597, mae: 2.4274, huber: 2.0108, swd: 7.7181, target_std: 18.3689
      Epoch 2 composite train-obj: 2.120040
            Val objective improved 2.5248 → 2.5121, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.9451, mae: 2.4950, huber: 2.0951, swd: 12.7403, target_std: 20.3415
    Epoch [3/50], Val Losses: mse: 23.2228, mae: 2.9039, huber: 2.4981, swd: 11.8881, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 15.6019, mae: 2.4192, huber: 2.0023, swd: 7.6558, target_std: 18.3689
      Epoch 3 composite train-obj: 2.095129
            Val objective improved 2.5121 → 2.4981, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.6179, mae: 2.4765, huber: 2.0765, swd: 12.5244, target_std: 20.3413
    Epoch [4/50], Val Losses: mse: 23.0649, mae: 2.9005, huber: 2.4937, swd: 11.8350, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 15.5564, mae: 2.4195, huber: 2.0019, swd: 7.6128, target_std: 18.3689
      Epoch 4 composite train-obj: 2.076504
            Val objective improved 2.4981 → 2.4937, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 22.3245, mae: 2.4604, huber: 2.0605, swd: 12.3186, target_std: 20.3409
    Epoch [5/50], Val Losses: mse: 23.1806, mae: 2.9110, huber: 2.5040, swd: 11.8104, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 15.7833, mae: 2.4232, huber: 2.0062, swd: 7.6961, target_std: 18.3689
      Epoch 5 composite train-obj: 2.060480
            No improvement (2.5040), counter 1/5
    Epoch [6/50], Train Losses: mse: 22.1023, mae: 2.4451, huber: 2.0455, swd: 12.1608, target_std: 20.3428
    Epoch [6/50], Val Losses: mse: 23.0198, mae: 2.9073, huber: 2.4989, swd: 11.6908, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 15.6126, mae: 2.4199, huber: 2.0016, swd: 7.6135, target_std: 18.3689
      Epoch 6 composite train-obj: 2.045453
            No improvement (2.4989), counter 2/5
    Epoch [7/50], Train Losses: mse: 21.8706, mae: 2.4326, huber: 2.0331, swd: 11.9965, target_std: 20.3411
    Epoch [7/50], Val Losses: mse: 23.2615, mae: 2.9213, huber: 2.5132, swd: 11.9527, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 15.6263, mae: 2.4318, huber: 2.0128, swd: 7.7249, target_std: 18.3689
      Epoch 7 composite train-obj: 2.033064
            No improvement (2.5132), counter 3/5
    Epoch [8/50], Train Losses: mse: 21.6651, mae: 2.4215, huber: 2.0221, swd: 11.8379, target_std: 20.3419
    Epoch [8/50], Val Losses: mse: 23.1346, mae: 2.9167, huber: 2.5085, swd: 11.8484, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 15.6788, mae: 2.4339, huber: 2.0151, swd: 7.7650, target_std: 18.3689
      Epoch 8 composite train-obj: 2.022109
            No improvement (2.5085), counter 4/5
    Epoch [9/50], Train Losses: mse: 21.5189, mae: 2.4116, huber: 2.0124, swd: 11.7414, target_std: 20.3412
    Epoch [9/50], Val Losses: mse: 23.4421, mae: 2.9339, huber: 2.5253, swd: 11.8597, target_std: 20.5279
    Epoch [9/50], Test Losses: mse: 16.1262, mae: 2.4478, huber: 2.0294, swd: 7.9975, target_std: 18.3689
      Epoch 9 composite train-obj: 2.012358
    Epoch [9/50], Test Losses: mse: 15.5564, mae: 2.4195, huber: 2.0019, swd: 7.6128, target_std: 18.3689
    Best round's Test MSE: 15.5564, MAE: 2.4195, SWD: 7.6128
    Best round's Validation MSE: 23.0649, MAE: 2.9005
    Best round's Test verification MSE : 15.5564, MAE: 2.4195, SWD: 7.6128
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.2916, mae: 2.5912, huber: 2.1896, swd: 14.0912, target_std: 20.3419
    Epoch [1/50], Val Losses: mse: 23.6291, mae: 2.9238, huber: 2.5195, swd: 12.5697, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 15.9667, mae: 2.4391, huber: 2.0233, swd: 8.1542, target_std: 18.3689
      Epoch 1 composite train-obj: 2.189599
            Val objective improved inf → 2.5195, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.1490, mae: 2.5080, huber: 2.1083, swd: 13.5069, target_std: 20.3418
    Epoch [2/50], Val Losses: mse: 23.2613, mae: 2.9032, huber: 2.4977, swd: 12.3532, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 15.7709, mae: 2.4251, huber: 2.0088, swd: 8.0686, target_std: 18.3689
      Epoch 2 composite train-obj: 2.108318
            Val objective improved 2.5195 → 2.4977, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.7766, mae: 2.4862, huber: 2.0864, swd: 13.2605, target_std: 20.3409
    Epoch [3/50], Val Losses: mse: 23.0826, mae: 2.8981, huber: 2.4912, swd: 12.3078, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 15.6440, mae: 2.4176, huber: 2.0008, swd: 8.0750, target_std: 18.3689
      Epoch 3 composite train-obj: 2.086410
            Val objective improved 2.4977 → 2.4912, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.4802, mae: 2.4696, huber: 2.0698, swd: 13.0535, target_std: 20.3413
    Epoch [4/50], Val Losses: mse: 23.1129, mae: 2.9102, huber: 2.5011, swd: 12.1855, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 15.8912, mae: 2.4286, huber: 2.0113, swd: 8.2137, target_std: 18.3689
      Epoch 4 composite train-obj: 2.069763
            No improvement (2.5011), counter 1/5
    Epoch [5/50], Train Losses: mse: 22.2388, mae: 2.4556, huber: 2.0558, swd: 12.8572, target_std: 20.3415
    Epoch [5/50], Val Losses: mse: 23.1593, mae: 2.9107, huber: 2.5029, swd: 12.2895, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 15.7727, mae: 2.4201, huber: 2.0028, swd: 8.1890, target_std: 18.3689
      Epoch 5 composite train-obj: 2.055754
            No improvement (2.5029), counter 2/5
    Epoch [6/50], Train Losses: mse: 21.9975, mae: 2.4421, huber: 2.0425, swd: 12.6649, target_std: 20.3408
    Epoch [6/50], Val Losses: mse: 23.2416, mae: 2.9235, huber: 2.5154, swd: 12.3173, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 15.9594, mae: 2.4333, huber: 2.0157, swd: 8.2907, target_std: 18.3689
      Epoch 6 composite train-obj: 2.042488
            No improvement (2.5154), counter 3/5
    Epoch [7/50], Train Losses: mse: 21.7541, mae: 2.4282, huber: 2.0287, swd: 12.4572, target_std: 20.3416
    Epoch [7/50], Val Losses: mse: 23.1135, mae: 2.9145, huber: 2.5064, swd: 12.2517, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 15.9543, mae: 2.4332, huber: 2.0151, swd: 8.3255, target_std: 18.3689
      Epoch 7 composite train-obj: 2.028710
            No improvement (2.5064), counter 4/5
    Epoch [8/50], Train Losses: mse: 21.4967, mae: 2.4159, huber: 2.0165, swd: 12.2452, target_std: 20.3409
    Epoch [8/50], Val Losses: mse: 23.5243, mae: 2.9362, huber: 2.5284, swd: 12.4859, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 16.2284, mae: 2.4496, huber: 2.0311, swd: 8.5573, target_std: 18.3689
      Epoch 8 composite train-obj: 2.016456
    Epoch [8/50], Test Losses: mse: 15.6440, mae: 2.4176, huber: 2.0008, swd: 8.0750, target_std: 18.3689
    Best round's Test MSE: 15.6440, MAE: 2.4176, SWD: 8.0750
    Best round's Validation MSE: 23.0826, MAE: 2.8981
    Best round's Test verification MSE : 15.6440, MAE: 2.4176, SWD: 8.0750
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.2797, mae: 2.5841, huber: 2.1829, swd: 13.1038, target_std: 20.3426
    Epoch [1/50], Val Losses: mse: 23.8384, mae: 2.9358, huber: 2.5319, swd: 11.7670, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 16.0149, mae: 2.4457, huber: 2.0297, swd: 7.5441, target_std: 18.3689
      Epoch 1 composite train-obj: 2.182870
            Val objective improved inf → 2.5319, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.2654, mae: 2.5123, huber: 2.1129, swd: 12.5437, target_std: 20.3407
    Epoch [2/50], Val Losses: mse: 23.4920, mae: 2.9166, huber: 2.5119, swd: 11.5007, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 15.9392, mae: 2.4345, huber: 2.0184, swd: 7.5158, target_std: 18.3689
      Epoch 2 composite train-obj: 2.112918
            Val objective improved 2.5319 → 2.5119, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.9623, mae: 2.4925, huber: 2.0932, swd: 12.3791, target_std: 20.3417
    Epoch [3/50], Val Losses: mse: 23.2854, mae: 2.9051, huber: 2.4998, swd: 11.3685, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 15.7840, mae: 2.4209, huber: 2.0045, swd: 7.4382, target_std: 18.3689
      Epoch 3 composite train-obj: 2.093197
            Val objective improved 2.5119 → 2.4998, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.6254, mae: 2.4749, huber: 2.0757, swd: 12.1691, target_std: 20.3406
    Epoch [4/50], Val Losses: mse: 23.3151, mae: 2.9144, huber: 2.5085, swd: 11.4174, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 15.7664, mae: 2.4259, huber: 2.0087, swd: 7.5063, target_std: 18.3689
      Epoch 4 composite train-obj: 2.075657
            No improvement (2.5085), counter 1/5
    Epoch [5/50], Train Losses: mse: 22.3310, mae: 2.4583, huber: 2.0593, swd: 11.9750, target_std: 20.3409
    Epoch [5/50], Val Losses: mse: 23.2122, mae: 2.9130, huber: 2.5069, swd: 11.3522, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 15.7733, mae: 2.4284, huber: 2.0109, swd: 7.5044, target_std: 18.3689
      Epoch 5 composite train-obj: 2.059301
            No improvement (2.5069), counter 2/5
    Epoch [6/50], Train Losses: mse: 22.0763, mae: 2.4441, huber: 2.0452, swd: 11.8032, target_std: 20.3415
    Epoch [6/50], Val Losses: mse: 23.3486, mae: 2.9162, huber: 2.5101, swd: 11.4147, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 15.9799, mae: 2.4340, huber: 2.0165, swd: 7.6647, target_std: 18.3689
      Epoch 6 composite train-obj: 2.045173
            No improvement (2.5101), counter 3/5
    Epoch [7/50], Train Losses: mse: 21.7533, mae: 2.4283, huber: 2.0296, swd: 11.5645, target_std: 20.3413
    Epoch [7/50], Val Losses: mse: 23.3895, mae: 2.9247, huber: 2.5185, swd: 11.5074, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 15.9746, mae: 2.4375, huber: 2.0197, swd: 7.6864, target_std: 18.3689
      Epoch 7 composite train-obj: 2.029594
            No improvement (2.5185), counter 4/5
    Epoch [8/50], Train Losses: mse: 21.5092, mae: 2.4149, huber: 2.0164, swd: 11.3827, target_std: 20.3421
    Epoch [8/50], Val Losses: mse: 23.1774, mae: 2.9091, huber: 2.5030, swd: 11.2120, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 16.0155, mae: 2.4314, huber: 2.0139, swd: 7.6923, target_std: 18.3689
      Epoch 8 composite train-obj: 2.016364
    Epoch [8/50], Test Losses: mse: 15.7840, mae: 2.4209, huber: 2.0045, swd: 7.4382, target_std: 18.3689
    Best round's Test MSE: 15.7840, MAE: 2.4209, SWD: 7.4382
    Best round's Validation MSE: 23.2854, MAE: 2.9051
    Best round's Test verification MSE : 15.7840, MAE: 2.4209, SWD: 7.4382
    
    ==================================================
    Experiment Summary (TimeMixer_ettm2_seq96_pred336_20250430_1941)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 15.6615 ± 0.0938
      mae: 2.4193 ± 0.0013
      huber: 2.0024 ± 0.0016
      swd: 7.7087 ± 0.2687
      target_std: 18.3689 ± 0.0000
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 23.1443 ± 0.1000
      mae: 2.9012 ± 0.0029
      huber: 2.4949 ± 0.0036
      swd: 11.8371 ± 0.3835
      target_std: 20.5279 ± 0.0000
      count: 52.0000 ± 0.0000
    ==================================================
    
    Experiment complete: TimeMixer_ettm2_seq96_pred336_20250430_1941
    Model: TimeMixer
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm2']['channels'],
    enc_in=data_mgr.datasets['ettm2']['channels'],
    dec_in=data_mgr.datasets['ettm2']['channels'],
    c_out=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 375
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 375
    Validation Batches: 49
    Test Batches: 103
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 33.4274, mae: 3.0947, huber: 2.6828, swd: 17.8081, target_std: 20.3581
    Epoch [1/50], Val Losses: mse: 28.9628, mae: 3.3565, huber: 2.9413, swd: 13.7645, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.2586, mae: 2.7997, huber: 2.3771, swd: 9.8297, target_std: 18.3425
      Epoch 1 composite train-obj: 2.682831
            Val objective improved inf → 2.9413, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.9439, mae: 2.9857, huber: 2.5760, swd: 17.0657, target_std: 20.3585
    Epoch [2/50], Val Losses: mse: 28.7249, mae: 3.3481, huber: 2.9310, swd: 13.5408, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 20.1698, mae: 2.7889, huber: 2.3660, swd: 9.7457, target_std: 18.3425
      Epoch 2 composite train-obj: 2.575965
            Val objective improved 2.9413 → 2.9310, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 31.6209, mae: 2.9666, huber: 2.5567, swd: 16.8781, target_std: 20.3588
    Epoch [3/50], Val Losses: mse: 28.5582, mae: 3.3464, huber: 2.9282, swd: 13.2312, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.2500, mae: 2.7842, huber: 2.3615, swd: 9.7309, target_std: 18.3425
      Epoch 3 composite train-obj: 2.556685
            Val objective improved 2.9310 → 2.9282, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 31.2553, mae: 2.9461, huber: 2.5359, swd: 16.6220, target_std: 20.3590
    Epoch [4/50], Val Losses: mse: 28.6044, mae: 3.3530, huber: 2.9333, swd: 13.1302, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 20.4032, mae: 2.7888, huber: 2.3654, swd: 9.8534, target_std: 18.3425
      Epoch 4 composite train-obj: 2.535912
            No improvement (2.9333), counter 1/5
    Epoch [5/50], Train Losses: mse: 31.0107, mae: 2.9292, huber: 2.5189, swd: 16.4649, target_std: 20.3583
    Epoch [5/50], Val Losses: mse: 28.1793, mae: 3.3332, huber: 2.9126, swd: 13.0068, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 20.1292, mae: 2.7873, huber: 2.3630, swd: 9.8242, target_std: 18.3425
      Epoch 5 composite train-obj: 2.518896
            Val objective improved 2.9282 → 2.9126, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.7154, mae: 2.9118, huber: 2.5015, swd: 16.2363, target_std: 20.3597
    Epoch [6/50], Val Losses: mse: 28.3410, mae: 3.3424, huber: 2.9215, swd: 13.1225, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 20.1150, mae: 2.7870, huber: 2.3623, swd: 9.8653, target_std: 18.3425
      Epoch 6 composite train-obj: 2.501512
            No improvement (2.9215), counter 1/5
    Epoch [7/50], Train Losses: mse: 30.5125, mae: 2.8985, huber: 2.4883, swd: 16.0835, target_std: 20.3588
    Epoch [7/50], Val Losses: mse: 28.5717, mae: 3.3636, huber: 2.9428, swd: 13.2983, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 20.1242, mae: 2.7952, huber: 2.3702, swd: 9.9124, target_std: 18.3425
      Epoch 7 composite train-obj: 2.488320
            No improvement (2.9428), counter 2/5
    Epoch [8/50], Train Losses: mse: 30.2604, mae: 2.8849, huber: 2.4749, swd: 15.8739, target_std: 20.3593
    Epoch [8/50], Val Losses: mse: 29.4655, mae: 3.4177, huber: 2.9962, swd: 13.9549, target_std: 20.4999
    Epoch [8/50], Test Losses: mse: 20.7282, mae: 2.8293, huber: 2.4041, swd: 10.4685, target_std: 18.3425
      Epoch 8 composite train-obj: 2.474929
            No improvement (2.9962), counter 3/5
    Epoch [9/50], Train Losses: mse: 29.9933, mae: 2.8719, huber: 2.4620, swd: 15.6612, target_std: 20.3592
    Epoch [9/50], Val Losses: mse: 28.9736, mae: 3.3943, huber: 2.9734, swd: 13.4737, target_std: 20.4999
    Epoch [9/50], Test Losses: mse: 20.6420, mae: 2.8136, huber: 2.3890, swd: 10.3431, target_std: 18.3425
      Epoch 9 composite train-obj: 2.462007
            No improvement (2.9734), counter 4/5
    Epoch [10/50], Train Losses: mse: 29.7340, mae: 2.8593, huber: 2.4497, swd: 15.4401, target_std: 20.3590
    Epoch [10/50], Val Losses: mse: 29.0869, mae: 3.4025, huber: 2.9817, swd: 13.5248, target_std: 20.4999
    Epoch [10/50], Test Losses: mse: 20.8296, mae: 2.8179, huber: 2.3934, swd: 10.4526, target_std: 18.3425
      Epoch 10 composite train-obj: 2.449656
    Epoch [10/50], Test Losses: mse: 20.1292, mae: 2.7873, huber: 2.3630, swd: 9.8242, target_std: 18.3425
    Best round's Test MSE: 20.1292, MAE: 2.7873, SWD: 9.8242
    Best round's Validation MSE: 28.1793, MAE: 3.3332
    Best round's Test verification MSE : 20.1292, MAE: 2.7873, SWD: 9.8242
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.7875, mae: 3.0454, huber: 2.6347, swd: 16.1600, target_std: 20.3588
    Epoch [1/50], Val Losses: mse: 28.8655, mae: 3.3541, huber: 2.9395, swd: 12.3589, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.3318, mae: 2.7929, huber: 2.3709, swd: 9.1210, target_std: 18.3425
      Epoch 1 composite train-obj: 2.634691
            Val objective improved inf → 2.9395, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.7218, mae: 2.9697, huber: 2.5604, swd: 15.6519, target_std: 20.3592
    Epoch [2/50], Val Losses: mse: 28.6000, mae: 3.3451, huber: 2.9291, swd: 12.0747, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 20.4072, mae: 2.7925, huber: 2.3692, swd: 9.2329, target_std: 18.3425
      Epoch 2 composite train-obj: 2.560354
            Val objective improved 2.9395 → 2.9291, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 31.2596, mae: 2.9421, huber: 2.5327, swd: 15.3835, target_std: 20.3589
    Epoch [3/50], Val Losses: mse: 28.1543, mae: 3.3203, huber: 2.9042, swd: 12.0242, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.0840, mae: 2.7833, huber: 2.3596, swd: 9.1288, target_std: 18.3425
      Epoch 3 composite train-obj: 2.532707
            Val objective improved 2.9291 → 2.9042, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 30.9112, mae: 2.9224, huber: 2.5131, swd: 15.1497, target_std: 20.3590
    Epoch [4/50], Val Losses: mse: 28.9879, mae: 3.3793, huber: 2.9621, swd: 12.4654, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 20.6577, mae: 2.8096, huber: 2.3852, swd: 9.6068, target_std: 18.3425
      Epoch 4 composite train-obj: 2.513136
            No improvement (2.9621), counter 1/5
    Epoch [5/50], Train Losses: mse: 30.6214, mae: 2.9035, huber: 2.4943, swd: 14.9491, target_std: 20.3586
    Epoch [5/50], Val Losses: mse: 28.1773, mae: 3.3467, huber: 2.9296, swd: 11.9644, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 20.3244, mae: 2.8035, huber: 2.3786, swd: 9.3873, target_std: 18.3425
      Epoch 5 composite train-obj: 2.494314
            No improvement (2.9296), counter 2/5
    Epoch [6/50], Train Losses: mse: 30.3368, mae: 2.8868, huber: 2.4776, swd: 14.7421, target_std: 20.3586
    Epoch [6/50], Val Losses: mse: 28.8849, mae: 3.3856, huber: 2.9677, swd: 12.3427, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 20.9984, mae: 2.8314, huber: 2.4060, swd: 9.9285, target_std: 18.3425
      Epoch 6 composite train-obj: 2.477602
            No improvement (2.9677), counter 3/5
    Epoch [7/50], Train Losses: mse: 30.1630, mae: 2.8750, huber: 2.4658, swd: 14.6268, target_std: 20.3580
    Epoch [7/50], Val Losses: mse: 29.9852, mae: 3.4371, huber: 3.0184, swd: 13.0429, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 21.8346, mae: 2.8589, huber: 2.4338, swd: 10.5830, target_std: 18.3425
      Epoch 7 composite train-obj: 2.465779
            No improvement (3.0184), counter 4/5
    Epoch [8/50], Train Losses: mse: 29.9128, mae: 2.8628, huber: 2.4537, swd: 14.4496, target_std: 20.3593
    Epoch [8/50], Val Losses: mse: 29.6596, mae: 3.4226, huber: 3.0040, swd: 12.7973, target_std: 20.4999
    Epoch [8/50], Test Losses: mse: 22.1201, mae: 2.8676, huber: 2.4426, swd: 10.7589, target_std: 18.3425
      Epoch 8 composite train-obj: 2.453695
    Epoch [8/50], Test Losses: mse: 20.0840, mae: 2.7833, huber: 2.3596, swd: 9.1288, target_std: 18.3425
    Best round's Test MSE: 20.0840, MAE: 2.7833, SWD: 9.1288
    Best round's Validation MSE: 28.1543, MAE: 3.3203
    Best round's Test verification MSE : 20.0840, MAE: 2.7833, SWD: 9.1288
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.7696, mae: 3.0433, huber: 2.6329, swd: 18.3371, target_std: 20.3583
    Epoch [1/50], Val Losses: mse: 28.8748, mae: 3.3540, huber: 2.9377, swd: 14.0578, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.1900, mae: 2.7954, huber: 2.3727, swd: 10.3580, target_std: 18.3425
      Epoch 1 composite train-obj: 2.632852
            Val objective improved inf → 2.9377, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.8061, mae: 2.9753, huber: 2.5655, swd: 17.8050, target_std: 20.3589
    Epoch [2/50], Val Losses: mse: 28.6598, mae: 3.3496, huber: 2.9312, swd: 13.7836, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 20.1798, mae: 2.7916, huber: 2.3683, swd: 10.2866, target_std: 18.3425
      Epoch 2 composite train-obj: 2.565479
            Val objective improved 2.9377 → 2.9312, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 31.5022, mae: 2.9547, huber: 2.5444, swd: 17.6517, target_std: 20.3583
    Epoch [3/50], Val Losses: mse: 28.4276, mae: 3.3434, huber: 2.9221, swd: 13.6057, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.0668, mae: 2.7896, huber: 2.3653, swd: 10.3058, target_std: 18.3425
      Epoch 3 composite train-obj: 2.544404
            Val objective improved 2.9312 → 2.9221, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 31.2582, mae: 2.9371, huber: 2.5264, swd: 17.5029, target_std: 20.3588
    Epoch [4/50], Val Losses: mse: 28.0830, mae: 3.3253, huber: 2.9039, swd: 13.3193, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 19.8837, mae: 2.7780, huber: 2.3536, swd: 10.1984, target_std: 18.3425
      Epoch 4 composite train-obj: 2.526357
            Val objective improved 2.9221 → 2.9039, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 30.9678, mae: 2.9210, huber: 2.5101, swd: 17.2701, target_std: 20.3593
    Epoch [5/50], Val Losses: mse: 28.1448, mae: 3.3354, huber: 2.9142, swd: 13.5665, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 19.8783, mae: 2.7959, huber: 2.3705, swd: 10.3541, target_std: 18.3425
      Epoch 5 composite train-obj: 2.510061
            No improvement (2.9142), counter 1/5
    Epoch [6/50], Train Losses: mse: 30.7426, mae: 2.9071, huber: 2.4962, swd: 17.0957, target_std: 20.3601
    Epoch [6/50], Val Losses: mse: 27.9952, mae: 3.3236, huber: 2.9023, swd: 13.3250, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 19.9675, mae: 2.7847, huber: 2.3599, swd: 10.3762, target_std: 18.3425
      Epoch 6 composite train-obj: 2.496228
            Val objective improved 2.9039 → 2.9023, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 30.5077, mae: 2.8936, huber: 2.4829, swd: 16.9099, target_std: 20.3595
    Epoch [7/50], Val Losses: mse: 28.3225, mae: 3.3504, huber: 2.9283, swd: 13.4752, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 20.0364, mae: 2.7940, huber: 2.3682, swd: 10.4187, target_std: 18.3425
      Epoch 7 composite train-obj: 2.482915
            No improvement (2.9283), counter 1/5
    Epoch [8/50], Train Losses: mse: 30.2725, mae: 2.8818, huber: 2.4714, swd: 16.7206, target_std: 20.3586
    Epoch [8/50], Val Losses: mse: 28.7487, mae: 3.3728, huber: 2.9516, swd: 13.8115, target_std: 20.4999
    Epoch [8/50], Test Losses: mse: 20.1693, mae: 2.7923, huber: 2.3670, swd: 10.5078, target_std: 18.3425
      Epoch 8 composite train-obj: 2.471392
            No improvement (2.9516), counter 2/5
    Epoch [9/50], Train Losses: mse: 30.0597, mae: 2.8709, huber: 2.4608, swd: 16.5435, target_std: 20.3589
    Epoch [9/50], Val Losses: mse: 28.2260, mae: 3.3499, huber: 2.9294, swd: 13.4690, target_std: 20.4999
    Epoch [9/50], Test Losses: mse: 20.0061, mae: 2.7897, huber: 2.3644, swd: 10.4203, target_std: 18.3425
      Epoch 9 composite train-obj: 2.460792
            No improvement (2.9294), counter 3/5
    Epoch [10/50], Train Losses: mse: 29.8486, mae: 2.8604, huber: 2.4505, swd: 16.3726, target_std: 20.3597
    Epoch [10/50], Val Losses: mse: 28.7288, mae: 3.3820, huber: 2.9615, swd: 13.8279, target_std: 20.4999
    Epoch [10/50], Test Losses: mse: 20.2857, mae: 2.8119, huber: 2.3858, swd: 10.7080, target_std: 18.3425
      Epoch 10 composite train-obj: 2.450505
            No improvement (2.9615), counter 4/5
    Epoch [11/50], Train Losses: mse: 29.6430, mae: 2.8507, huber: 2.4411, swd: 16.2010, target_std: 20.3593
    Epoch [11/50], Val Losses: mse: 28.6433, mae: 3.3823, huber: 2.9621, swd: 13.5857, target_std: 20.4999
    Epoch [11/50], Test Losses: mse: 20.5755, mae: 2.8197, huber: 2.3943, swd: 10.8536, target_std: 18.3425
      Epoch 11 composite train-obj: 2.441097
    Epoch [11/50], Test Losses: mse: 19.9675, mae: 2.7847, huber: 2.3599, swd: 10.3762, target_std: 18.3425
    Best round's Test MSE: 19.9675, MAE: 2.7847, SWD: 10.3762
    Best round's Validation MSE: 27.9952, MAE: 3.3236
    Best round's Test verification MSE : 19.9675, MAE: 2.7847, SWD: 10.3762
    
    ==================================================
    Experiment Summary (TimeMixer_ettm2_seq96_pred720_20250430_1957)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 20.0602 ± 0.0681
      mae: 2.7851 ± 0.0016
      huber: 2.3608 ± 0.0015
      swd: 9.7764 ± 0.5103
      target_std: 18.3425 ± 0.0000
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 28.1096 ± 0.0815
      mae: 3.3257 ± 0.0055
      huber: 2.9064 ± 0.0045
      swd: 12.7853 ± 0.5537
      target_std: 20.4999 ± 0.0000
      count: 49.0000 ± 0.0000
    ==================================================
    
    Experiment complete: TimeMixer_ettm2_seq96_pred720_20250430_1957
    Model: TimeMixer
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### PatchTST

#### pred=96


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm2']['channels'],
    enc_in=data_mgr.datasets['ettm2']['channels'],
    dec_in=data_mgr.datasets['ettm2']['channels'],
    c_out=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 380
    Validation Batches: 53
    Test Batches: 108
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.6237, mae: 1.9469, huber: 1.5576, swd: 6.6636, target_std: 20.3268
    Epoch [1/50], Val Losses: mse: 13.0160, mae: 2.1382, huber: 1.7502, swd: 6.8723, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 9.6832, mae: 1.9069, huber: 1.5027, swd: 4.8020, target_std: 18.4467
      Epoch 1 composite train-obj: 1.557598
            Val objective improved inf → 1.7502, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.9272, mae: 1.8820, huber: 1.4949, swd: 6.3608, target_std: 20.3264
    Epoch [2/50], Val Losses: mse: 12.8546, mae: 2.1238, huber: 1.7349, swd: 6.9018, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.6179, mae: 1.9054, huber: 1.5007, swd: 4.9225, target_std: 18.4467
      Epoch 2 composite train-obj: 1.494942
            Val objective improved 1.7502 → 1.7349, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.6577, mae: 1.8551, huber: 1.4691, swd: 6.2085, target_std: 20.3295
    Epoch [3/50], Val Losses: mse: 12.8867, mae: 2.1359, huber: 1.7481, swd: 6.8148, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.5127, mae: 1.8873, huber: 1.4838, swd: 4.7834, target_std: 18.4467
      Epoch 3 composite train-obj: 1.469067
            No improvement (1.7481), counter 1/5
    Epoch [4/50], Train Losses: mse: 11.4550, mae: 1.8390, huber: 1.4535, swd: 6.0692, target_std: 20.3268
    Epoch [4/50], Val Losses: mse: 12.8071, mae: 2.1302, huber: 1.7417, swd: 6.8135, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.5911, mae: 1.8909, huber: 1.4874, swd: 4.8983, target_std: 18.4467
      Epoch 4 composite train-obj: 1.453464
            No improvement (1.7417), counter 2/5
    Epoch [5/50], Train Losses: mse: 11.3294, mae: 1.8275, huber: 1.4423, swd: 5.9874, target_std: 20.3271
    Epoch [5/50], Val Losses: mse: 12.7713, mae: 2.1122, huber: 1.7249, swd: 6.8699, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.4420, mae: 1.8715, huber: 1.4686, swd: 4.7937, target_std: 18.4467
      Epoch 5 composite train-obj: 1.442333
            Val objective improved 1.7349 → 1.7249, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.1670, mae: 1.8134, huber: 1.4287, swd: 5.8619, target_std: 20.3275
    Epoch [6/50], Val Losses: mse: 12.6098, mae: 2.1062, huber: 1.7179, swd: 6.7241, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.4335, mae: 1.8786, huber: 1.4751, swd: 4.7918, target_std: 18.4467
      Epoch 6 composite train-obj: 1.428670
            Val objective improved 1.7249 → 1.7179, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 11.0769, mae: 1.8049, huber: 1.4204, swd: 5.8100, target_std: 20.3281
    Epoch [7/50], Val Losses: mse: 12.7530, mae: 2.1080, huber: 1.7228, swd: 6.8148, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.3783, mae: 1.8673, huber: 1.4652, swd: 4.7131, target_std: 18.4467
      Epoch 7 composite train-obj: 1.420448
            No improvement (1.7228), counter 1/5
    Epoch [8/50], Train Losses: mse: 10.9695, mae: 1.7959, huber: 1.4117, swd: 5.7217, target_std: 20.3278
    Epoch [8/50], Val Losses: mse: 12.7729, mae: 2.1127, huber: 1.7256, swd: 6.8577, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.4183, mae: 1.8688, huber: 1.4659, swd: 4.7501, target_std: 18.4467
      Epoch 8 composite train-obj: 1.411713
            No improvement (1.7256), counter 2/5
    Epoch [9/50], Train Losses: mse: 10.8453, mae: 1.7863, huber: 1.4024, swd: 5.6236, target_std: 20.3268
    Epoch [9/50], Val Losses: mse: 12.5032, mae: 2.0999, huber: 1.7109, swd: 6.6756, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.3761, mae: 1.8762, huber: 1.4726, swd: 4.7962, target_std: 18.4467
      Epoch 9 composite train-obj: 1.402352
            Val objective improved 1.7179 → 1.7109, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 10.7537, mae: 1.7789, huber: 1.3951, swd: 5.5508, target_std: 20.3260
    Epoch [10/50], Val Losses: mse: 12.6825, mae: 2.1050, huber: 1.7188, swd: 6.7716, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.5058, mae: 1.8737, huber: 1.4713, swd: 4.8305, target_std: 18.4467
      Epoch 10 composite train-obj: 1.395077
            No improvement (1.7188), counter 1/5
    Epoch [11/50], Train Losses: mse: 10.7130, mae: 1.7744, huber: 1.3908, swd: 5.5262, target_std: 20.3287
    Epoch [11/50], Val Losses: mse: 12.8650, mae: 2.1230, huber: 1.7363, swd: 6.8890, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.4425, mae: 1.8712, huber: 1.4683, swd: 4.7409, target_std: 18.4467
      Epoch 11 composite train-obj: 1.390769
            No improvement (1.7363), counter 2/5
    Epoch [12/50], Train Losses: mse: 10.6065, mae: 1.7665, huber: 1.3829, swd: 5.4303, target_std: 20.3272
    Epoch [12/50], Val Losses: mse: 12.9650, mae: 2.1299, huber: 1.7415, swd: 7.0364, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 9.4843, mae: 1.8844, huber: 1.4805, swd: 4.8736, target_std: 18.4467
      Epoch 12 composite train-obj: 1.382909
            No improvement (1.7415), counter 3/5
    Epoch [13/50], Train Losses: mse: 10.5626, mae: 1.7623, huber: 1.3788, swd: 5.4092, target_std: 20.3255
    Epoch [13/50], Val Losses: mse: 12.7391, mae: 2.1075, huber: 1.7213, swd: 6.8473, target_std: 20.5171
    Epoch [13/50], Test Losses: mse: 9.4550, mae: 1.8702, huber: 1.4682, swd: 4.7613, target_std: 18.4467
      Epoch 13 composite train-obj: 1.378798
            No improvement (1.7213), counter 4/5
    Epoch [14/50], Train Losses: mse: 10.4787, mae: 1.7550, huber: 1.3717, swd: 5.3383, target_std: 20.3265
    Epoch [14/50], Val Losses: mse: 12.7389, mae: 2.1159, huber: 1.7269, swd: 6.8284, target_std: 20.5171
    Epoch [14/50], Test Losses: mse: 9.4898, mae: 1.8846, huber: 1.4803, swd: 4.8378, target_std: 18.4467
      Epoch 14 composite train-obj: 1.371665
    Epoch [14/50], Test Losses: mse: 9.3761, mae: 1.8762, huber: 1.4726, swd: 4.7962, target_std: 18.4467
    Best round's Test MSE: 9.3761, MAE: 1.8762, SWD: 4.7962
    Best round's Validation MSE: 12.5032, MAE: 2.0999
    Best round's Test verification MSE : 9.3761, MAE: 1.8762, SWD: 4.7962
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.6430, mae: 1.9454, huber: 1.5561, swd: 6.4423, target_std: 20.3275
    Epoch [1/50], Val Losses: mse: 13.0631, mae: 2.1502, huber: 1.7618, swd: 6.7058, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 9.5651, mae: 1.9029, huber: 1.4989, swd: 4.5612, target_std: 18.4467
      Epoch 1 composite train-obj: 1.556134
            Val objective improved inf → 1.7618, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.8861, mae: 1.8790, huber: 1.4919, swd: 6.1150, target_std: 20.3257
    Epoch [2/50], Val Losses: mse: 12.8134, mae: 2.1237, huber: 1.7359, swd: 6.5781, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.5048, mae: 1.8985, huber: 1.4941, swd: 4.5845, target_std: 18.4467
      Epoch 2 composite train-obj: 1.491898
            Val objective improved 1.7618 → 1.7359, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.6344, mae: 1.8553, huber: 1.4690, swd: 5.9736, target_std: 20.3272
    Epoch [3/50], Val Losses: mse: 12.7663, mae: 2.1205, huber: 1.7339, swd: 6.5281, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.5432, mae: 1.8856, huber: 1.4824, swd: 4.6145, target_std: 18.4467
      Epoch 3 composite train-obj: 1.469029
            Val objective improved 1.7359 → 1.7339, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.4316, mae: 1.8361, huber: 1.4505, swd: 5.8514, target_std: 20.3268
    Epoch [4/50], Val Losses: mse: 12.7933, mae: 2.1196, huber: 1.7334, swd: 6.5959, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.5139, mae: 1.8843, huber: 1.4810, swd: 4.6240, target_std: 18.4467
      Epoch 4 composite train-obj: 1.450485
            Val objective improved 1.7339 → 1.7334, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.2921, mae: 1.8226, huber: 1.4375, swd: 5.7659, target_std: 20.3271
    Epoch [5/50], Val Losses: mse: 12.6815, mae: 2.1122, huber: 1.7236, swd: 6.4764, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.4854, mae: 1.8795, huber: 1.4759, swd: 4.6275, target_std: 18.4467
      Epoch 5 composite train-obj: 1.437481
            Val objective improved 1.7334 → 1.7236, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.1802, mae: 1.8131, huber: 1.4283, swd: 5.6887, target_std: 20.3283
    Epoch [6/50], Val Losses: mse: 12.7199, mae: 2.1135, huber: 1.7259, swd: 6.5524, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.4744, mae: 1.8833, huber: 1.4792, swd: 4.6147, target_std: 18.4467
      Epoch 6 composite train-obj: 1.428265
            No improvement (1.7259), counter 1/5
    Epoch [7/50], Train Losses: mse: 11.0655, mae: 1.8035, huber: 1.4190, swd: 5.6133, target_std: 20.3273
    Epoch [7/50], Val Losses: mse: 12.6075, mae: 2.1101, huber: 1.7217, swd: 6.4494, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.3188, mae: 1.8601, huber: 1.4570, swd: 4.4817, target_std: 18.4467
      Epoch 7 composite train-obj: 1.419006
            Val objective improved 1.7236 → 1.7217, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.9590, mae: 1.7938, huber: 1.4095, swd: 5.5393, target_std: 20.3266
    Epoch [8/50], Val Losses: mse: 12.6958, mae: 2.1053, huber: 1.7184, swd: 6.6041, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.3399, mae: 1.8620, huber: 1.4595, swd: 4.4950, target_std: 18.4467
      Epoch 8 composite train-obj: 1.409493
            Val objective improved 1.7217 → 1.7184, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 10.8318, mae: 1.7846, huber: 1.4005, swd: 5.4411, target_std: 20.3288
    Epoch [9/50], Val Losses: mse: 12.8810, mae: 2.1212, huber: 1.7352, swd: 6.6947, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.4390, mae: 1.8717, huber: 1.4692, swd: 4.5833, target_std: 18.4467
      Epoch 9 composite train-obj: 1.400521
            No improvement (1.7352), counter 1/5
    Epoch [10/50], Train Losses: mse: 10.7907, mae: 1.7788, huber: 1.3949, swd: 5.4232, target_std: 20.3266
    Epoch [10/50], Val Losses: mse: 12.6813, mae: 2.1091, huber: 1.7219, swd: 6.5530, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.4874, mae: 1.8745, huber: 1.4713, swd: 4.6115, target_std: 18.4467
      Epoch 10 composite train-obj: 1.394860
            No improvement (1.7219), counter 2/5
    Epoch [11/50], Train Losses: mse: 10.6951, mae: 1.7716, huber: 1.3877, swd: 5.3420, target_std: 20.3264
    Epoch [11/50], Val Losses: mse: 12.5812, mae: 2.0997, huber: 1.7130, swd: 6.5087, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.4456, mae: 1.8702, huber: 1.4674, swd: 4.5632, target_std: 18.4467
      Epoch 11 composite train-obj: 1.387705
            Val objective improved 1.7184 → 1.7130, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 10.6226, mae: 1.7648, huber: 1.3812, swd: 5.2928, target_std: 20.3278
    Epoch [12/50], Val Losses: mse: 12.7394, mae: 2.1180, huber: 1.7302, swd: 6.5994, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 9.4218, mae: 1.8728, huber: 1.4696, swd: 4.5806, target_std: 18.4467
      Epoch 12 composite train-obj: 1.381162
            No improvement (1.7302), counter 1/5
    Epoch [13/50], Train Losses: mse: 10.5524, mae: 1.7605, huber: 1.3768, swd: 5.2393, target_std: 20.3272
    Epoch [13/50], Val Losses: mse: 12.7032, mae: 2.1171, huber: 1.7288, swd: 6.6045, target_std: 20.5171
    Epoch [13/50], Test Losses: mse: 9.5338, mae: 1.8864, huber: 1.4822, swd: 4.6577, target_std: 18.4467
      Epoch 13 composite train-obj: 1.376768
            No improvement (1.7288), counter 2/5
    Epoch [14/50], Train Losses: mse: 10.4555, mae: 1.7522, huber: 1.3687, swd: 5.1619, target_std: 20.3267
    Epoch [14/50], Val Losses: mse: 12.9677, mae: 2.1278, huber: 1.7396, swd: 6.7689, target_std: 20.5171
    Epoch [14/50], Test Losses: mse: 9.4751, mae: 1.8716, huber: 1.4693, swd: 4.5764, target_std: 18.4467
      Epoch 14 composite train-obj: 1.368677
            No improvement (1.7396), counter 3/5
    Epoch [15/50], Train Losses: mse: 10.4086, mae: 1.7475, huber: 1.3640, swd: 5.1332, target_std: 20.3264
    Epoch [15/50], Val Losses: mse: 12.9876, mae: 2.1334, huber: 1.7448, swd: 6.7478, target_std: 20.5171
    Epoch [15/50], Test Losses: mse: 9.4900, mae: 1.8697, huber: 1.4672, swd: 4.5650, target_std: 18.4467
      Epoch 15 composite train-obj: 1.364036
            No improvement (1.7448), counter 4/5
    Epoch [16/50], Train Losses: mse: 10.3370, mae: 1.7421, huber: 1.3588, swd: 5.0838, target_std: 20.3273
    Epoch [16/50], Val Losses: mse: 12.8380, mae: 2.1286, huber: 1.7403, swd: 6.6348, target_std: 20.5171
    Epoch [16/50], Test Losses: mse: 9.5910, mae: 1.8774, huber: 1.4748, swd: 4.6661, target_std: 18.4467
      Epoch 16 composite train-obj: 1.358767
    Epoch [16/50], Test Losses: mse: 9.4456, mae: 1.8702, huber: 1.4674, swd: 4.5632, target_std: 18.4467
    Best round's Test MSE: 9.4456, MAE: 1.8702, SWD: 4.5632
    Best round's Validation MSE: 12.5812, MAE: 2.0997
    Best round's Test verification MSE : 9.4456, MAE: 1.8702, SWD: 4.5632
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.6143, mae: 1.9461, huber: 1.5565, swd: 5.9986, target_std: 20.3279
    Epoch [1/50], Val Losses: mse: 13.2467, mae: 2.1652, huber: 1.7751, swd: 6.4768, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 9.6033, mae: 1.9166, huber: 1.5110, swd: 4.2653, target_std: 18.4467
      Epoch 1 composite train-obj: 1.556522
            Val objective improved inf → 1.7751, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.8986, mae: 1.8785, huber: 1.4914, swd: 5.7166, target_std: 20.3266
    Epoch [2/50], Val Losses: mse: 12.8726, mae: 2.1200, huber: 1.7319, swd: 6.2504, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.4220, mae: 1.8873, huber: 1.4835, swd: 4.2020, target_std: 18.4467
      Epoch 2 composite train-obj: 1.491381
            Val objective improved 1.7751 → 1.7319, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.6319, mae: 1.8538, huber: 1.4676, swd: 5.5857, target_std: 20.3266
    Epoch [3/50], Val Losses: mse: 12.6361, mae: 2.1059, huber: 1.7184, swd: 6.1046, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.3606, mae: 1.8769, huber: 1.4736, swd: 4.2076, target_std: 18.4467
      Epoch 3 composite train-obj: 1.467630
            Val objective improved 1.7319 → 1.7184, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.4400, mae: 1.8372, huber: 1.4514, swd: 5.4735, target_std: 20.3280
    Epoch [4/50], Val Losses: mse: 12.7993, mae: 2.1136, huber: 1.7267, swd: 6.2226, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.3131, mae: 1.8676, huber: 1.4646, swd: 4.1512, target_std: 18.4467
      Epoch 4 composite train-obj: 1.451428
            No improvement (1.7267), counter 1/5
    Epoch [5/50], Train Losses: mse: 11.3015, mae: 1.8229, huber: 1.4377, swd: 5.3937, target_std: 20.3267
    Epoch [5/50], Val Losses: mse: 12.6117, mae: 2.1059, huber: 1.7178, swd: 6.1580, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.3131, mae: 1.8781, huber: 1.4737, swd: 4.1991, target_std: 18.4467
      Epoch 5 composite train-obj: 1.437699
            Val objective improved 1.7184 → 1.7178, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.1655, mae: 1.8124, huber: 1.4276, swd: 5.3083, target_std: 20.3268
    Epoch [6/50], Val Losses: mse: 12.8828, mae: 2.1377, huber: 1.7480, swd: 6.2594, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.4153, mae: 1.8844, huber: 1.4794, swd: 4.2267, target_std: 18.4467
      Epoch 6 composite train-obj: 1.427566
            No improvement (1.7480), counter 1/5
    Epoch [7/50], Train Losses: mse: 11.0613, mae: 1.8019, huber: 1.4175, swd: 5.2420, target_std: 20.3282
    Epoch [7/50], Val Losses: mse: 12.5441, mae: 2.1020, huber: 1.7135, swd: 6.0775, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.2258, mae: 1.8670, huber: 1.4631, swd: 4.1342, target_std: 18.4467
      Epoch 7 composite train-obj: 1.417454
            Val objective improved 1.7178 → 1.7135, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.9395, mae: 1.7923, huber: 1.4080, swd: 5.1613, target_std: 20.3273
    Epoch [8/50], Val Losses: mse: 12.7687, mae: 2.1127, huber: 1.7252, swd: 6.1890, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.3918, mae: 1.8706, huber: 1.4676, swd: 4.2243, target_std: 18.4467
      Epoch 8 composite train-obj: 1.407964
            No improvement (1.7252), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.8513, mae: 1.7858, huber: 1.4016, swd: 5.0892, target_std: 20.3262
    Epoch [9/50], Val Losses: mse: 12.6844, mae: 2.1157, huber: 1.7269, swd: 6.1483, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.3622, mae: 1.8694, huber: 1.4655, swd: 4.2361, target_std: 18.4467
      Epoch 9 composite train-obj: 1.401608
            No improvement (1.7269), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.7599, mae: 1.7761, huber: 1.3923, swd: 5.0301, target_std: 20.3272
    Epoch [10/50], Val Losses: mse: 12.7780, mae: 2.1086, huber: 1.7221, swd: 6.2343, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.5251, mae: 1.8745, huber: 1.4720, swd: 4.3299, target_std: 18.4467
      Epoch 10 composite train-obj: 1.392252
            No improvement (1.7221), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.6804, mae: 1.7689, huber: 1.3853, swd: 4.9794, target_std: 20.3274
    Epoch [11/50], Val Losses: mse: 12.8506, mae: 2.1209, huber: 1.7339, swd: 6.2805, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.3715, mae: 1.8678, huber: 1.4649, swd: 4.2306, target_std: 18.4467
      Epoch 11 composite train-obj: 1.385278
            No improvement (1.7339), counter 4/5
    Epoch [12/50], Train Losses: mse: 10.5826, mae: 1.7622, huber: 1.3787, swd: 4.8985, target_std: 20.3266
    Epoch [12/50], Val Losses: mse: 12.9092, mae: 2.1315, huber: 1.7429, swd: 6.2708, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 9.6210, mae: 1.8815, huber: 1.4787, swd: 4.3878, target_std: 18.4467
      Epoch 12 composite train-obj: 1.378655
    Epoch [12/50], Test Losses: mse: 9.2258, mae: 1.8670, huber: 1.4631, swd: 4.1342, target_std: 18.4467
    Best round's Test MSE: 9.2258, MAE: 1.8670, SWD: 4.1342
    Best round's Validation MSE: 12.5441, MAE: 2.1020
    Best round's Test verification MSE : 9.2258, MAE: 1.8670, SWD: 4.1342
    
    ==================================================
    Experiment Summary (PatchTST_ettm2_seq96_pred96_20250430_1859)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 9.3492 ± 0.0917
      mae: 1.8711 ± 0.0038
      huber: 1.4677 ± 0.0039
      swd: 4.4979 ± 0.2742
      target_std: 18.4467 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 12.5428 ± 0.0319
      mae: 2.1005 ± 0.0010
      huber: 1.7124 ± 0.0011
      swd: 6.4206 ± 0.2520
      target_std: 20.5171 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: PatchTST_ettm2_seq96_pred96_20250430_1859
    Model: PatchTST
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm2']['channels'],
    enc_in=data_mgr.datasets['ettm2']['channels'],
    dec_in=data_mgr.datasets['ettm2']['channels'],
    c_out=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 379
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 379
    Validation Batches: 53
    Test Batches: 107
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 18.2840, mae: 2.2894, huber: 1.8924, swd: 10.1203, target_std: 20.3339
    Epoch [1/50], Val Losses: mse: 18.9543, mae: 2.5769, huber: 2.1767, swd: 10.5872, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 12.4143, mae: 2.1804, huber: 1.7677, swd: 6.1134, target_std: 18.4014
      Epoch 1 composite train-obj: 1.892438
            Val objective improved inf → 2.1767, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.4353, mae: 2.2236, huber: 1.8284, swd: 9.6817, target_std: 20.3338
    Epoch [2/50], Val Losses: mse: 18.7341, mae: 2.5748, huber: 2.1752, swd: 10.3070, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.5115, mae: 2.1727, huber: 1.7599, swd: 6.1566, target_std: 18.4014
      Epoch 2 composite train-obj: 1.828372
            Val objective improved 2.1767 → 2.1752, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 17.0627, mae: 2.1976, huber: 1.8028, swd: 9.4238, target_std: 20.3333
    Epoch [3/50], Val Losses: mse: 18.6593, mae: 2.5677, huber: 2.1656, swd: 10.2333, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.3718, mae: 2.1667, huber: 1.7536, swd: 6.0865, target_std: 18.4014
      Epoch 3 composite train-obj: 1.802782
            Val objective improved 2.1752 → 2.1656, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 16.8117, mae: 2.1777, huber: 1.7833, swd: 9.2434, target_std: 20.3338
    Epoch [4/50], Val Losses: mse: 18.7325, mae: 2.5663, huber: 2.1655, swd: 10.3226, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.5095, mae: 2.1786, huber: 1.7648, swd: 6.2323, target_std: 18.4014
      Epoch 4 composite train-obj: 1.783330
            Val objective improved 2.1656 → 2.1655, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 16.6042, mae: 2.1646, huber: 1.7706, swd: 9.0837, target_std: 20.3333
    Epoch [5/50], Val Losses: mse: 18.6640, mae: 2.5587, huber: 2.1601, swd: 10.3264, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.5509, mae: 2.1775, huber: 1.7651, swd: 6.3038, target_std: 18.4014
      Epoch 5 composite train-obj: 1.770628
            Val objective improved 2.1655 → 2.1601, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 16.4513, mae: 2.1531, huber: 1.7593, swd: 8.9648, target_std: 20.3340
    Epoch [6/50], Val Losses: mse: 18.6658, mae: 2.5634, huber: 2.1641, swd: 10.3401, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.6318, mae: 2.1803, huber: 1.7674, swd: 6.3999, target_std: 18.4014
      Epoch 6 composite train-obj: 1.759309
            No improvement (2.1641), counter 1/5
    Epoch [7/50], Train Losses: mse: 16.3044, mae: 2.1427, huber: 1.7492, swd: 8.8396, target_std: 20.3336
    Epoch [7/50], Val Losses: mse: 18.8025, mae: 2.5665, huber: 2.1668, swd: 10.4535, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 12.6856, mae: 2.1877, huber: 1.7746, swd: 6.4605, target_std: 18.4014
      Epoch 7 composite train-obj: 1.749173
            No improvement (2.1668), counter 2/5
    Epoch [8/50], Train Losses: mse: 16.1269, mae: 2.1320, huber: 1.7387, swd: 8.6822, target_std: 20.3333
    Epoch [8/50], Val Losses: mse: 18.8543, mae: 2.5735, huber: 2.1742, swd: 10.3884, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 12.8525, mae: 2.1988, huber: 1.7856, swd: 6.5358, target_std: 18.4014
      Epoch 8 composite train-obj: 1.738683
            No improvement (2.1742), counter 3/5
    Epoch [9/50], Train Losses: mse: 16.0202, mae: 2.1245, huber: 1.7314, swd: 8.6036, target_std: 20.3333
    Epoch [9/50], Val Losses: mse: 19.2917, mae: 2.5855, huber: 2.1866, swd: 10.7868, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 13.0106, mae: 2.1929, huber: 1.7803, swd: 6.6963, target_std: 18.4014
      Epoch 9 composite train-obj: 1.731379
            No improvement (2.1866), counter 4/5
    Epoch [10/50], Train Losses: mse: 15.9022, mae: 2.1156, huber: 1.7225, swd: 8.4970, target_std: 20.3336
    Epoch [10/50], Val Losses: mse: 19.4102, mae: 2.6112, huber: 2.2101, swd: 10.8881, target_std: 20.5407
    Epoch [10/50], Test Losses: mse: 13.3005, mae: 2.2258, huber: 1.8112, swd: 6.9633, target_std: 18.4014
      Epoch 10 composite train-obj: 1.722532
    Epoch [10/50], Test Losses: mse: 12.5509, mae: 2.1775, huber: 1.7651, swd: 6.3038, target_std: 18.4014
    Best round's Test MSE: 12.5509, MAE: 2.1775, SWD: 6.3038
    Best round's Validation MSE: 18.6640, MAE: 2.5587
    Best round's Test verification MSE : 12.5509, MAE: 2.1775, SWD: 6.3038
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 18.1890, mae: 2.2853, huber: 1.8884, swd: 10.4232, target_std: 20.3340
    Epoch [1/50], Val Losses: mse: 19.0821, mae: 2.5798, huber: 2.1818, swd: 11.0053, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 12.4975, mae: 2.1726, huber: 1.7612, swd: 6.3783, target_std: 18.4014
      Epoch 1 composite train-obj: 1.888426
            Val objective improved inf → 2.1818, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.3735, mae: 2.2222, huber: 1.8268, swd: 9.9789, target_std: 20.3336
    Epoch [2/50], Val Losses: mse: 18.7941, mae: 2.5796, huber: 2.1789, swd: 10.7184, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.4809, mae: 2.1761, huber: 1.7632, swd: 6.3616, target_std: 18.4014
      Epoch 2 composite train-obj: 1.826769
            Val objective improved 2.1818 → 2.1789, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 17.0769, mae: 2.1985, huber: 1.8037, swd: 9.7687, target_std: 20.3336
    Epoch [3/50], Val Losses: mse: 18.7031, mae: 2.5707, huber: 2.1703, swd: 10.6902, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.3995, mae: 2.1658, huber: 1.7538, swd: 6.3029, target_std: 18.4014
      Epoch 3 composite train-obj: 1.803718
            Val objective improved 2.1789 → 2.1703, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 16.8128, mae: 2.1807, huber: 1.7863, swd: 9.5720, target_std: 20.3336
    Epoch [4/50], Val Losses: mse: 18.7219, mae: 2.5700, huber: 2.1704, swd: 10.6731, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.4656, mae: 2.1724, huber: 1.7598, swd: 6.3971, target_std: 18.4014
      Epoch 4 composite train-obj: 1.786278
            No improvement (2.1704), counter 1/5
    Epoch [5/50], Train Losses: mse: 16.6227, mae: 2.1659, huber: 1.7719, swd: 9.4115, target_std: 20.3336
    Epoch [5/50], Val Losses: mse: 18.6831, mae: 2.5724, huber: 2.1724, swd: 10.6361, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.5470, mae: 2.1758, huber: 1.7633, swd: 6.4734, target_std: 18.4014
      Epoch 5 composite train-obj: 1.771902
            No improvement (2.1724), counter 2/5
    Epoch [6/50], Train Losses: mse: 16.4692, mae: 2.1541, huber: 1.7603, swd: 9.2879, target_std: 20.3335
    Epoch [6/50], Val Losses: mse: 18.9048, mae: 2.5847, huber: 2.1851, swd: 10.7761, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.6828, mae: 2.1799, huber: 1.7673, swd: 6.5667, target_std: 18.4014
      Epoch 6 composite train-obj: 1.760348
            No improvement (2.1851), counter 3/5
    Epoch [7/50], Train Losses: mse: 16.3333, mae: 2.1441, huber: 1.7507, swd: 9.1749, target_std: 20.3341
    Epoch [7/50], Val Losses: mse: 18.4558, mae: 2.5462, huber: 2.1486, swd: 10.5189, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 12.4505, mae: 2.1622, huber: 1.7504, swd: 6.3784, target_std: 18.4014
      Epoch 7 composite train-obj: 1.750702
            Val objective improved 2.1703 → 2.1486, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 16.2149, mae: 2.1358, huber: 1.7426, swd: 9.0772, target_std: 20.3338
    Epoch [8/50], Val Losses: mse: 18.4564, mae: 2.5545, huber: 2.1559, swd: 10.4537, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 12.3728, mae: 2.1647, huber: 1.7523, swd: 6.3202, target_std: 18.4014
      Epoch 8 composite train-obj: 1.742617
            No improvement (2.1559), counter 1/5
    Epoch [9/50], Train Losses: mse: 16.0676, mae: 2.1250, huber: 1.7321, swd: 8.9512, target_std: 20.3340
    Epoch [9/50], Val Losses: mse: 18.5843, mae: 2.5571, huber: 2.1585, swd: 10.5838, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 12.5790, mae: 2.1692, huber: 1.7569, swd: 6.5102, target_std: 18.4014
      Epoch 9 composite train-obj: 1.732123
            No improvement (2.1585), counter 2/5
    Epoch [10/50], Train Losses: mse: 15.9506, mae: 2.1186, huber: 1.7257, swd: 8.8401, target_std: 20.3338
    Epoch [10/50], Val Losses: mse: 18.9517, mae: 2.5868, huber: 2.1881, swd: 10.8264, target_std: 20.5407
    Epoch [10/50], Test Losses: mse: 12.7622, mae: 2.1798, huber: 1.7677, swd: 6.6589, target_std: 18.4014
      Epoch 10 composite train-obj: 1.725750
            No improvement (2.1881), counter 3/5
    Epoch [11/50], Train Losses: mse: 15.8542, mae: 2.1098, huber: 1.7172, swd: 8.7596, target_std: 20.3334
    Epoch [11/50], Val Losses: mse: 18.6632, mae: 2.5601, huber: 2.1621, swd: 10.6560, target_std: 20.5407
    Epoch [11/50], Test Losses: mse: 12.5210, mae: 2.1680, huber: 1.7557, swd: 6.4958, target_std: 18.4014
      Epoch 11 composite train-obj: 1.717199
            No improvement (2.1621), counter 4/5
    Epoch [12/50], Train Losses: mse: 15.7237, mae: 2.1016, huber: 1.7091, swd: 8.6415, target_std: 20.3332
    Epoch [12/50], Val Losses: mse: 18.7678, mae: 2.5682, huber: 2.1693, swd: 10.7489, target_std: 20.5407
    Epoch [12/50], Test Losses: mse: 12.5160, mae: 2.1685, huber: 1.7562, swd: 6.4882, target_std: 18.4014
      Epoch 12 composite train-obj: 1.709066
    Epoch [12/50], Test Losses: mse: 12.4505, mae: 2.1622, huber: 1.7504, swd: 6.3784, target_std: 18.4014
    Best round's Test MSE: 12.4505, MAE: 2.1622, SWD: 6.3784
    Best round's Validation MSE: 18.4558, MAE: 2.5462
    Best round's Test verification MSE : 12.4505, MAE: 2.1622, SWD: 6.3784
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 18.2028, mae: 2.2854, huber: 1.8883, swd: 8.8365, target_std: 20.3336
    Epoch [1/50], Val Losses: mse: 19.2995, mae: 2.6035, huber: 2.2013, swd: 9.3703, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 12.5384, mae: 2.1764, huber: 1.7639, swd: 5.3533, target_std: 18.4014
      Epoch 1 composite train-obj: 1.888319
            Val objective improved inf → 2.2013, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 17.4601, mae: 2.2265, huber: 1.8310, swd: 8.5140, target_std: 20.3340
    Epoch [2/50], Val Losses: mse: 18.8870, mae: 2.5762, huber: 2.1764, swd: 9.2883, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.4680, mae: 2.1792, huber: 1.7665, swd: 5.4490, target_std: 18.4014
      Epoch 2 composite train-obj: 1.830985
            Val objective improved 2.2013 → 2.1764, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 17.1433, mae: 2.2023, huber: 1.8075, swd: 8.3384, target_std: 20.3342
    Epoch [3/50], Val Losses: mse: 18.6410, mae: 2.5587, huber: 2.1596, swd: 9.0645, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.2665, mae: 2.1629, huber: 1.7504, swd: 5.2993, target_std: 18.4014
      Epoch 3 composite train-obj: 1.807470
            Val objective improved 2.1764 → 2.1596, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 16.8953, mae: 2.1835, huber: 1.7893, swd: 8.1822, target_std: 20.3335
    Epoch [4/50], Val Losses: mse: 18.6325, mae: 2.5611, huber: 2.1613, swd: 9.1081, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.2320, mae: 2.1596, huber: 1.7476, swd: 5.2951, target_std: 18.4014
      Epoch 4 composite train-obj: 1.789281
            No improvement (2.1613), counter 1/5
    Epoch [5/50], Train Losses: mse: 16.6620, mae: 2.1669, huber: 1.7731, swd: 8.0165, target_std: 20.3337
    Epoch [5/50], Val Losses: mse: 18.8981, mae: 2.5705, huber: 2.1727, swd: 9.3520, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.3658, mae: 2.1684, huber: 1.7566, swd: 5.4113, target_std: 18.4014
      Epoch 5 composite train-obj: 1.773094
            No improvement (2.1727), counter 2/5
    Epoch [6/50], Train Losses: mse: 16.4953, mae: 2.1549, huber: 1.7613, swd: 7.9104, target_std: 20.3339
    Epoch [6/50], Val Losses: mse: 18.7345, mae: 2.5659, huber: 2.1672, swd: 9.0695, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.2428, mae: 2.1559, huber: 1.7439, swd: 5.3134, target_std: 18.4014
      Epoch 6 composite train-obj: 1.761276
            No improvement (2.1672), counter 3/5
    Epoch [7/50], Train Losses: mse: 16.3020, mae: 2.1422, huber: 1.7488, swd: 7.7609, target_std: 20.3338
    Epoch [7/50], Val Losses: mse: 18.8762, mae: 2.5676, huber: 2.1699, swd: 9.2575, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 12.3344, mae: 2.1570, huber: 1.7457, swd: 5.3389, target_std: 18.4014
      Epoch 7 composite train-obj: 1.748753
            No improvement (2.1699), counter 4/5
    Epoch [8/50], Train Losses: mse: 16.1757, mae: 2.1335, huber: 1.7403, swd: 7.6719, target_std: 20.3342
    Epoch [8/50], Val Losses: mse: 18.7866, mae: 2.5797, huber: 2.1800, swd: 9.0857, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 12.4123, mae: 2.1611, huber: 1.7489, swd: 5.3915, target_std: 18.4014
      Epoch 8 composite train-obj: 1.740271
    Epoch [8/50], Test Losses: mse: 12.2665, mae: 2.1629, huber: 1.7504, swd: 5.2993, target_std: 18.4014
    Best round's Test MSE: 12.2665, MAE: 2.1629, SWD: 5.2993
    Best round's Validation MSE: 18.6410, MAE: 2.5587
    Best round's Test verification MSE : 12.2665, MAE: 2.1629, SWD: 5.2993
    
    ==================================================
    Experiment Summary (PatchTST_ettm2_seq96_pred196_20250430_1918)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.4226 ± 0.1178
      mae: 2.1675 ± 0.0070
      huber: 1.7553 ± 0.0070
      swd: 5.9939 ± 0.4921
      target_std: 18.4014 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 18.5869 ± 0.0932
      mae: 2.5545 ± 0.0059
      huber: 2.1561 ± 0.0053
      swd: 9.9699 ± 0.6450
      target_std: 20.5407 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: PatchTST_ettm2_seq96_pred196_20250430_1918
    Model: PatchTST
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm2']['channels'],
    enc_in=data_mgr.datasets['ettm2']['channels'],
    dec_in=data_mgr.datasets['ettm2']['channels'],
    c_out=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 378
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 378
    Validation Batches: 52
    Test Batches: 106
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.0951, mae: 2.5979, huber: 2.1947, swd: 13.1190, target_std: 20.3417
    Epoch [1/50], Val Losses: mse: 24.0751, mae: 2.9729, huber: 2.5651, swd: 12.4737, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 15.7252, mae: 2.4571, huber: 2.0386, swd: 7.8327, target_std: 18.3689
      Epoch 1 composite train-obj: 2.194676
            Val objective improved inf → 2.5651, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.1941, mae: 2.5389, huber: 2.1368, swd: 12.6372, target_std: 20.3411
    Epoch [2/50], Val Losses: mse: 23.5824, mae: 2.9492, huber: 2.5411, swd: 11.9854, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 15.6604, mae: 2.4303, huber: 2.0122, swd: 7.6352, target_std: 18.3689
      Epoch 2 composite train-obj: 2.136806
            Val objective improved 2.5651 → 2.5411, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.7646, mae: 2.5135, huber: 2.1119, swd: 12.3131, target_std: 20.3415
    Epoch [3/50], Val Losses: mse: 23.3378, mae: 2.9275, huber: 2.5211, swd: 11.8000, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 15.7319, mae: 2.4309, huber: 2.0130, swd: 7.6968, target_std: 18.3689
      Epoch 3 composite train-obj: 2.111918
            Val objective improved 2.5411 → 2.5211, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.4482, mae: 2.4948, huber: 2.0934, swd: 12.0532, target_std: 20.3410
    Epoch [4/50], Val Losses: mse: 23.4951, mae: 2.9407, huber: 2.5340, swd: 11.7035, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 15.7790, mae: 2.4336, huber: 2.0152, swd: 7.7074, target_std: 18.3689
      Epoch 4 composite train-obj: 2.093428
            No improvement (2.5340), counter 1/5
    Epoch [5/50], Train Losses: mse: 22.1602, mae: 2.4794, huber: 2.0783, swd: 11.8318, target_std: 20.3412
    Epoch [5/50], Val Losses: mse: 23.1629, mae: 2.9227, huber: 2.5144, swd: 11.5713, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 15.4741, mae: 2.4242, huber: 2.0058, swd: 7.5629, target_std: 18.3689
      Epoch 5 composite train-obj: 2.078273
            Val objective improved 2.5211 → 2.5144, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 21.9067, mae: 2.4659, huber: 2.0649, swd: 11.5955, target_std: 20.3410
    Epoch [6/50], Val Losses: mse: 23.8078, mae: 2.9655, huber: 2.5551, swd: 11.9665, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 15.7825, mae: 2.4459, huber: 2.0267, swd: 7.7933, target_std: 18.3689
      Epoch 6 composite train-obj: 2.064895
            No improvement (2.5551), counter 1/5
    Epoch [7/50], Train Losses: mse: 21.7183, mae: 2.4552, huber: 2.0544, swd: 11.4471, target_std: 20.3423
    Epoch [7/50], Val Losses: mse: 24.0461, mae: 2.9695, huber: 2.5613, swd: 12.0713, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 16.0058, mae: 2.4452, huber: 2.0263, swd: 7.8752, target_std: 18.3689
      Epoch 7 composite train-obj: 2.054395
            No improvement (2.5613), counter 2/5
    Epoch [8/50], Train Losses: mse: 21.4292, mae: 2.4405, huber: 2.0399, swd: 11.1909, target_std: 20.3412
    Epoch [8/50], Val Losses: mse: 23.5323, mae: 2.9494, huber: 2.5421, swd: 11.9038, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 15.6903, mae: 2.4343, huber: 2.0160, swd: 7.7542, target_std: 18.3689
      Epoch 8 composite train-obj: 2.039884
            No improvement (2.5421), counter 3/5
    Epoch [9/50], Train Losses: mse: 21.2721, mae: 2.4311, huber: 2.0306, swd: 11.0602, target_std: 20.3414
    Epoch [9/50], Val Losses: mse: 23.3941, mae: 2.9434, huber: 2.5356, swd: 11.7399, target_std: 20.5279
    Epoch [9/50], Test Losses: mse: 16.0695, mae: 2.4415, huber: 2.0229, swd: 7.9350, target_std: 18.3689
      Epoch 9 composite train-obj: 2.030569
            No improvement (2.5356), counter 4/5
    Epoch [10/50], Train Losses: mse: 21.1351, mae: 2.4240, huber: 2.0235, swd: 10.9454, target_std: 20.3417
    Epoch [10/50], Val Losses: mse: 23.3524, mae: 2.9435, huber: 2.5335, swd: 11.6473, target_std: 20.5279
    Epoch [10/50], Test Losses: mse: 15.9266, mae: 2.4418, huber: 2.0228, swd: 7.8970, target_std: 18.3689
      Epoch 10 composite train-obj: 2.023509
    Epoch [10/50], Test Losses: mse: 15.4741, mae: 2.4242, huber: 2.0058, swd: 7.5629, target_std: 18.3689
    Best round's Test MSE: 15.4741, MAE: 2.4242, SWD: 7.5629
    Best round's Validation MSE: 23.1629, MAE: 2.9227
    Best round's Test verification MSE : 15.4741, MAE: 2.4242, SWD: 7.5629
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.0639, mae: 2.5973, huber: 2.1940, swd: 13.7629, target_std: 20.3420
    Epoch [1/50], Val Losses: mse: 23.9708, mae: 2.9696, huber: 2.5597, swd: 12.6012, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 15.7940, mae: 2.4361, huber: 2.0181, swd: 8.1043, target_std: 18.3689
      Epoch 1 composite train-obj: 2.193955
            Val objective improved inf → 2.5597, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.1735, mae: 2.5399, huber: 2.1376, swd: 13.2219, target_std: 20.3414
    Epoch [2/50], Val Losses: mse: 23.2927, mae: 2.9331, huber: 2.5235, swd: 12.2392, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 15.4747, mae: 2.4241, huber: 2.0052, swd: 7.9064, target_std: 18.3689
      Epoch 2 composite train-obj: 2.137601
            Val objective improved 2.5597 → 2.5235, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.7465, mae: 2.5136, huber: 2.1119, swd: 12.8830, target_std: 20.3408
    Epoch [3/50], Val Losses: mse: 23.0327, mae: 2.9114, huber: 2.5045, swd: 12.1450, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 15.4126, mae: 2.4219, huber: 2.0033, swd: 7.9671, target_std: 18.3689
      Epoch 3 composite train-obj: 2.111876
            Val objective improved 2.5235 → 2.5045, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.5144, mae: 2.4996, huber: 2.0981, swd: 12.6645, target_std: 20.3414
    Epoch [4/50], Val Losses: mse: 22.9655, mae: 2.9108, huber: 2.5035, swd: 11.9832, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 15.5199, mae: 2.4187, huber: 2.0003, swd: 7.9614, target_std: 18.3689
      Epoch 4 composite train-obj: 2.098112
            Val objective improved 2.5045 → 2.5035, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 22.1351, mae: 2.4801, huber: 2.0790, swd: 12.3624, target_std: 20.3413
    Epoch [5/50], Val Losses: mse: 23.1002, mae: 2.9217, huber: 2.5143, swd: 11.9984, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 15.5891, mae: 2.4239, huber: 2.0052, swd: 8.0033, target_std: 18.3689
      Epoch 5 composite train-obj: 2.078958
            No improvement (2.5143), counter 1/5
    Epoch [6/50], Train Losses: mse: 21.9067, mae: 2.4690, huber: 2.0681, swd: 12.1545, target_std: 20.3422
    Epoch [6/50], Val Losses: mse: 22.6485, mae: 2.8996, huber: 2.4925, swd: 11.6556, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 15.6796, mae: 2.4255, huber: 2.0070, swd: 8.0910, target_std: 18.3689
      Epoch 6 composite train-obj: 2.068143
            Val objective improved 2.5035 → 2.4925, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 21.6641, mae: 2.4547, huber: 2.0540, swd: 11.9376, target_std: 20.3418
    Epoch [7/50], Val Losses: mse: 23.7372, mae: 2.9623, huber: 2.5530, swd: 12.3663, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 16.1773, mae: 2.4547, huber: 2.0352, swd: 8.4674, target_std: 18.3689
      Epoch 7 composite train-obj: 2.053989
            No improvement (2.5530), counter 1/5
    Epoch [8/50], Train Losses: mse: 21.4613, mae: 2.4455, huber: 2.0449, swd: 11.7605, target_std: 20.3417
    Epoch [8/50], Val Losses: mse: 23.5774, mae: 2.9486, huber: 2.5404, swd: 12.3152, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 16.0801, mae: 2.4495, huber: 2.0299, swd: 8.4547, target_std: 18.3689
      Epoch 8 composite train-obj: 2.044900
            No improvement (2.5404), counter 2/5
    Epoch [9/50], Train Losses: mse: 21.2796, mae: 2.4352, huber: 2.0347, swd: 11.6048, target_std: 20.3425
    Epoch [9/50], Val Losses: mse: 23.1524, mae: 2.9282, huber: 2.5209, swd: 12.1336, target_std: 20.5279
    Epoch [9/50], Test Losses: mse: 15.7334, mae: 2.4283, huber: 2.0101, swd: 8.1265, target_std: 18.3689
      Epoch 9 composite train-obj: 2.034675
            No improvement (2.5209), counter 3/5
    Epoch [10/50], Train Losses: mse: 21.1281, mae: 2.4260, huber: 2.0256, swd: 11.4698, target_std: 20.3410
    Epoch [10/50], Val Losses: mse: 23.8274, mae: 2.9714, huber: 2.5616, swd: 12.4348, target_std: 20.5279
    Epoch [10/50], Test Losses: mse: 16.3374, mae: 2.4672, huber: 2.0468, swd: 8.6939, target_std: 18.3689
      Epoch 10 composite train-obj: 2.025585
            No improvement (2.5616), counter 4/5
    Epoch [11/50], Train Losses: mse: 20.8894, mae: 2.4162, huber: 2.0158, swd: 11.2426, target_std: 20.3409
    Epoch [11/50], Val Losses: mse: 24.2630, mae: 2.9803, huber: 2.5725, swd: 12.8641, target_std: 20.5279
    Epoch [11/50], Test Losses: mse: 16.7960, mae: 2.4688, huber: 2.0499, swd: 8.9407, target_std: 18.3689
      Epoch 11 composite train-obj: 2.015799
    Epoch [11/50], Test Losses: mse: 15.6796, mae: 2.4255, huber: 2.0070, swd: 8.0910, target_std: 18.3689
    Best round's Test MSE: 15.6796, MAE: 2.4255, SWD: 8.0910
    Best round's Validation MSE: 22.6485, MAE: 2.8996
    Best round's Test verification MSE : 15.6796, MAE: 2.4255, SWD: 8.0910
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.0604, mae: 2.5971, huber: 2.1936, swd: 12.7442, target_std: 20.3416
    Epoch [1/50], Val Losses: mse: 23.7254, mae: 2.9492, huber: 2.5409, swd: 11.3407, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 15.8974, mae: 2.4473, huber: 2.0291, swd: 7.3894, target_std: 18.3689
      Epoch 1 composite train-obj: 2.193560
            Val objective improved inf → 2.5409, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.1516, mae: 2.5384, huber: 2.1362, swd: 12.2540, target_std: 20.3418
    Epoch [2/50], Val Losses: mse: 23.4477, mae: 2.9318, huber: 2.5247, swd: 11.5699, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 15.4201, mae: 2.4348, huber: 2.0165, swd: 7.3416, target_std: 18.3689
      Epoch 2 composite train-obj: 2.136214
            Val objective improved 2.5409 → 2.5247, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.7800, mae: 2.5158, huber: 2.1141, swd: 12.0043, target_std: 20.3412
    Epoch [3/50], Val Losses: mse: 23.3318, mae: 2.9316, huber: 2.5239, swd: 11.1260, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 15.7863, mae: 2.4316, huber: 2.0128, swd: 7.4630, target_std: 18.3689
      Epoch 3 composite train-obj: 2.114140
            Val objective improved 2.5247 → 2.5239, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.5101, mae: 2.4974, huber: 2.0962, swd: 11.7886, target_std: 20.3414
    Epoch [4/50], Val Losses: mse: 23.6280, mae: 2.9551, huber: 2.5468, swd: 11.3954, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 15.7738, mae: 2.4349, huber: 2.0161, swd: 7.4861, target_std: 18.3689
      Epoch 4 composite train-obj: 2.096170
            No improvement (2.5468), counter 1/5
    Epoch [5/50], Train Losses: mse: 22.1941, mae: 2.4814, huber: 2.0804, swd: 11.5409, target_std: 20.3409
    Epoch [5/50], Val Losses: mse: 23.1695, mae: 2.9259, huber: 2.5179, swd: 11.0773, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 15.5887, mae: 2.4220, huber: 2.0034, swd: 7.3498, target_std: 18.3689
      Epoch 5 composite train-obj: 2.080444
            Val objective improved 2.5239 → 2.5179, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 21.9522, mae: 2.4690, huber: 2.0682, swd: 11.3481, target_std: 20.3411
    Epoch [6/50], Val Losses: mse: 23.5729, mae: 2.9442, huber: 2.5364, swd: 11.4454, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 15.7338, mae: 2.4462, huber: 2.0268, swd: 7.4977, target_std: 18.3689
      Epoch 6 composite train-obj: 2.068231
            No improvement (2.5364), counter 1/5
    Epoch [7/50], Train Losses: mse: 21.7114, mae: 2.4563, huber: 2.0557, swd: 11.1696, target_std: 20.3412
    Epoch [7/50], Val Losses: mse: 23.3203, mae: 2.9257, huber: 2.5174, swd: 11.2382, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 15.6810, mae: 2.4321, huber: 2.0128, swd: 7.4765, target_std: 18.3689
      Epoch 7 composite train-obj: 2.055658
            Val objective improved 2.5179 → 2.5174, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 21.5360, mae: 2.4470, huber: 2.0465, swd: 11.0004, target_std: 20.3413
    Epoch [8/50], Val Losses: mse: 23.9081, mae: 2.9499, huber: 2.5417, swd: 11.5982, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 16.0353, mae: 2.4494, huber: 2.0304, swd: 7.7245, target_std: 18.3689
      Epoch 8 composite train-obj: 2.046508
            No improvement (2.5417), counter 1/5
    Epoch [9/50], Train Losses: mse: 21.3076, mae: 2.4374, huber: 2.0370, swd: 10.8295, target_std: 20.3413
    Epoch [9/50], Val Losses: mse: 24.1622, mae: 2.9675, huber: 2.5601, swd: 11.7035, target_std: 20.5279
    Epoch [9/50], Test Losses: mse: 16.1602, mae: 2.4500, huber: 2.0313, swd: 7.7421, target_std: 18.3689
      Epoch 9 composite train-obj: 2.037000
            No improvement (2.5601), counter 2/5
    Epoch [10/50], Train Losses: mse: 21.1855, mae: 2.4299, huber: 2.0295, swd: 10.7205, target_std: 20.3419
    Epoch [10/50], Val Losses: mse: 24.0621, mae: 2.9573, huber: 2.5498, swd: 11.7160, target_std: 20.5279
    Epoch [10/50], Test Losses: mse: 16.0442, mae: 2.4430, huber: 2.0245, swd: 7.7365, target_std: 18.3689
      Epoch 10 composite train-obj: 2.029508
            No improvement (2.5498), counter 3/5
    Epoch [11/50], Train Losses: mse: 20.9816, mae: 2.4183, huber: 2.0180, swd: 10.5613, target_std: 20.3422
    Epoch [11/50], Val Losses: mse: 23.0607, mae: 2.9152, huber: 2.5082, swd: 11.0444, target_std: 20.5279
    Epoch [11/50], Test Losses: mse: 15.7744, mae: 2.4311, huber: 2.0126, swd: 7.5320, target_std: 18.3689
      Epoch 11 composite train-obj: 2.017996
            Val objective improved 2.5174 → 2.5082, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.7823, mae: 2.4106, huber: 2.0104, swd: 10.3945, target_std: 20.3417
    Epoch [12/50], Val Losses: mse: 24.4352, mae: 2.9923, huber: 2.5836, swd: 11.8636, target_std: 20.5279
    Epoch [12/50], Test Losses: mse: 16.6201, mae: 2.4700, huber: 2.0511, swd: 8.1236, target_std: 18.3689
      Epoch 12 composite train-obj: 2.010430
            No improvement (2.5836), counter 1/5
    Epoch [13/50], Train Losses: mse: 20.7082, mae: 2.4050, huber: 2.0048, swd: 10.3436, target_std: 20.3421
    Epoch [13/50], Val Losses: mse: 24.0441, mae: 2.9616, huber: 2.5538, swd: 11.6490, target_std: 20.5279
    Epoch [13/50], Test Losses: mse: 16.3633, mae: 2.4566, huber: 2.0374, swd: 8.0233, target_std: 18.3689
      Epoch 13 composite train-obj: 2.004796
            No improvement (2.5538), counter 2/5
    Epoch [14/50], Train Losses: mse: 20.5282, mae: 2.3964, huber: 1.9962, swd: 10.2102, target_std: 20.3406
    Epoch [14/50], Val Losses: mse: 23.2201, mae: 2.9228, huber: 2.5150, swd: 11.2482, target_std: 20.5279
    Epoch [14/50], Test Losses: mse: 15.7300, mae: 2.4322, huber: 2.0134, swd: 7.5239, target_std: 18.3689
      Epoch 14 composite train-obj: 1.996217
            No improvement (2.5150), counter 3/5
    Epoch [15/50], Train Losses: mse: 20.3874, mae: 2.3882, huber: 1.9880, swd: 10.1175, target_std: 20.3415
    Epoch [15/50], Val Losses: mse: 24.3660, mae: 2.9876, huber: 2.5790, swd: 11.8976, target_std: 20.5279
    Epoch [15/50], Test Losses: mse: 16.7434, mae: 2.4905, huber: 2.0705, swd: 8.2966, target_std: 18.3689
      Epoch 15 composite train-obj: 1.987977
            No improvement (2.5790), counter 4/5
    Epoch [16/50], Train Losses: mse: 20.2861, mae: 2.3819, huber: 1.9818, swd: 10.0275, target_std: 20.3413
    Epoch [16/50], Val Losses: mse: 24.5078, mae: 2.9995, huber: 2.5905, swd: 12.0855, target_std: 20.5279
    Epoch [16/50], Test Losses: mse: 16.5927, mae: 2.4806, huber: 2.0607, swd: 8.1377, target_std: 18.3689
      Epoch 16 composite train-obj: 1.981768
    Epoch [16/50], Test Losses: mse: 15.7744, mae: 2.4311, huber: 2.0126, swd: 7.5320, target_std: 18.3689
    Best round's Test MSE: 15.7744, MAE: 2.4311, SWD: 7.5320
    Best round's Validation MSE: 23.0607, MAE: 2.9152
    Best round's Test verification MSE : 15.7744, MAE: 2.4311, SWD: 7.5320
    
    ==================================================
    Experiment Summary (PatchTST_ettm2_seq96_pred336_20250430_1946)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 15.6427 ± 0.1254
      mae: 2.4269 ± 0.0030
      huber: 2.0085 ± 0.0030
      swd: 7.7286 ± 0.2565
      target_std: 18.3689 ± 0.0000
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 22.9574 ± 0.2223
      mae: 2.9125 ± 0.0096
      huber: 2.5050 ± 0.0092
      swd: 11.4238 ± 0.2705
      target_std: 20.5279 ± 0.0000
      count: 52.0000 ± 0.0000
    ==================================================
    
    Experiment complete: PatchTST_ettm2_seq96_pred336_20250430_1946
    Model: PatchTST
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm2']['channels'],
    enc_in=data_mgr.datasets['ettm2']['channels'],
    dec_in=data_mgr.datasets['ettm2']['channels'],
    c_out=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 375
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 375
    Validation Batches: 49
    Test Batches: 103
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.7659, mae: 3.0579, huber: 2.6456, swd: 17.3444, target_std: 20.3592
    Epoch [1/50], Val Losses: mse: 28.9055, mae: 3.3682, huber: 2.9512, swd: 13.4528, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.1387, mae: 2.7846, huber: 2.3607, swd: 9.7392, target_std: 18.3425
      Epoch 1 composite train-obj: 2.645604
            Val objective improved inf → 2.9512, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.9529, mae: 3.0051, huber: 2.5936, swd: 16.8778, target_std: 20.3593
    Epoch [2/50], Val Losses: mse: 28.6245, mae: 3.3509, huber: 2.9337, swd: 13.5557, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 19.9508, mae: 2.7882, huber: 2.3639, swd: 9.7472, target_std: 18.3425
      Epoch 2 composite train-obj: 2.593568
            Val objective improved 2.9512 → 2.9337, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 31.5118, mae: 2.9817, huber: 2.5705, swd: 16.5679, target_std: 20.3578
    Epoch [3/50], Val Losses: mse: 28.5829, mae: 3.3579, huber: 2.9394, swd: 13.4775, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.0229, mae: 2.8013, huber: 2.3764, swd: 9.8860, target_std: 18.3425
      Epoch 3 composite train-obj: 2.570501
            No improvement (2.9394), counter 1/5
    Epoch [4/50], Train Losses: mse: 31.1171, mae: 2.9653, huber: 2.5541, swd: 16.2251, target_std: 20.3584
    Epoch [4/50], Val Losses: mse: 28.5280, mae: 3.3570, huber: 2.9384, swd: 13.3171, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 20.1005, mae: 2.8035, huber: 2.3788, swd: 9.9083, target_std: 18.3425
      Epoch 4 composite train-obj: 2.554090
            No improvement (2.9384), counter 2/5
    Epoch [5/50], Train Losses: mse: 30.7714, mae: 2.9493, huber: 2.5382, swd: 15.9336, target_std: 20.3592
    Epoch [5/50], Val Losses: mse: 29.0881, mae: 3.4061, huber: 2.9849, swd: 13.2472, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 20.4416, mae: 2.8086, huber: 2.3825, swd: 10.0815, target_std: 18.3425
      Epoch 5 composite train-obj: 2.538229
            No improvement (2.9849), counter 3/5
    Epoch [6/50], Train Losses: mse: 30.4254, mae: 2.9332, huber: 2.5223, swd: 15.6382, target_std: 20.3584
    Epoch [6/50], Val Losses: mse: 28.9659, mae: 3.3907, huber: 2.9713, swd: 13.2034, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 20.4013, mae: 2.8082, huber: 2.3825, swd: 10.0883, target_std: 18.3425
      Epoch 6 composite train-obj: 2.522274
            No improvement (2.9713), counter 4/5
    Epoch [7/50], Train Losses: mse: 30.1591, mae: 2.9194, huber: 2.5086, swd: 15.3919, target_std: 20.3594
    Epoch [7/50], Val Losses: mse: 28.6710, mae: 3.3863, huber: 2.9665, swd: 12.9070, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 20.4989, mae: 2.8186, huber: 2.3925, swd: 10.2387, target_std: 18.3425
      Epoch 7 composite train-obj: 2.508646
    Epoch [7/50], Test Losses: mse: 19.9508, mae: 2.7882, huber: 2.3639, swd: 9.7472, target_std: 18.3425
    Best round's Test MSE: 19.9508, MAE: 2.7882, SWD: 9.7472
    Best round's Validation MSE: 28.6245, MAE: 3.3509
    Best round's Test verification MSE : 19.9508, MAE: 2.7882, SWD: 9.7472
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.7320, mae: 3.0573, huber: 2.6449, swd: 15.9907, target_std: 20.3592
    Epoch [1/50], Val Losses: mse: 28.9959, mae: 3.3800, huber: 2.9602, swd: 12.4310, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.0199, mae: 2.7857, huber: 2.3614, swd: 9.0404, target_std: 18.3425
      Epoch 1 composite train-obj: 2.644886
            Val objective improved inf → 2.9602, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.7958, mae: 3.0019, huber: 2.5902, swd: 15.4741, target_std: 20.3589
    Epoch [2/50], Val Losses: mse: 29.1554, mae: 3.3969, huber: 2.9773, swd: 12.4463, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 20.1513, mae: 2.7976, huber: 2.3725, swd: 9.1599, target_std: 18.3425
      Epoch 2 composite train-obj: 2.590160
            No improvement (2.9773), counter 1/5
    Epoch [3/50], Train Losses: mse: 31.3189, mae: 2.9772, huber: 2.5658, swd: 15.1146, target_std: 20.3586
    Epoch [3/50], Val Losses: mse: 28.9992, mae: 3.3924, huber: 2.9729, swd: 12.2251, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.1957, mae: 2.8010, huber: 2.3757, swd: 9.1063, target_std: 18.3425
      Epoch 3 composite train-obj: 2.565844
            No improvement (2.9729), counter 2/5
    Epoch [4/50], Train Losses: mse: 30.9936, mae: 2.9607, huber: 2.5495, swd: 14.8742, target_std: 20.3584
    Epoch [4/50], Val Losses: mse: 28.9121, mae: 3.3880, huber: 2.9690, swd: 12.2057, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 20.1234, mae: 2.7963, huber: 2.3709, swd: 9.1405, target_std: 18.3425
      Epoch 4 composite train-obj: 2.549517
            No improvement (2.9690), counter 3/5
    Epoch [5/50], Train Losses: mse: 30.6841, mae: 2.9449, huber: 2.5339, swd: 14.6463, target_std: 20.3583
    Epoch [5/50], Val Losses: mse: 29.6462, mae: 3.4380, huber: 3.0181, swd: 12.5712, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 20.3917, mae: 2.8177, huber: 2.3914, swd: 9.3335, target_std: 18.3425
      Epoch 5 composite train-obj: 2.533872
            No improvement (3.0181), counter 4/5
    Epoch [6/50], Train Losses: mse: 30.3311, mae: 2.9291, huber: 2.5181, swd: 14.3591, target_std: 20.3586
    Epoch [6/50], Val Losses: mse: 29.0959, mae: 3.3965, huber: 2.9768, swd: 12.2745, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 20.4558, mae: 2.8145, huber: 2.3888, swd: 9.4641, target_std: 18.3425
      Epoch 6 composite train-obj: 2.518136
    Epoch [6/50], Test Losses: mse: 20.0199, mae: 2.7857, huber: 2.3614, swd: 9.0404, target_std: 18.3425
    Best round's Test MSE: 20.0199, MAE: 2.7857, SWD: 9.0404
    Best round's Validation MSE: 28.9959, MAE: 3.3800
    Best round's Test verification MSE : 20.0199, MAE: 2.7857, SWD: 9.0404
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.6568, mae: 3.0535, huber: 2.6410, swd: 18.0836, target_std: 20.3581
    Epoch [1/50], Val Losses: mse: 28.9667, mae: 3.3752, huber: 2.9539, swd: 14.3595, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.0265, mae: 2.8066, huber: 2.3818, swd: 10.4198, target_std: 18.3425
      Epoch 1 composite train-obj: 2.640987
            Val objective improved inf → 2.9539, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.8191, mae: 3.0020, huber: 2.5904, swd: 17.6064, target_std: 20.3582
    Epoch [2/50], Val Losses: mse: 28.7935, mae: 3.3651, huber: 2.9477, swd: 13.7881, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 20.1229, mae: 2.7913, huber: 2.3671, swd: 10.2893, target_std: 18.3425
      Epoch 2 composite train-obj: 2.590369
            Val objective improved 2.9539 → 2.9477, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 31.3264, mae: 2.9779, huber: 2.5667, swd: 17.2171, target_std: 20.3589
    Epoch [3/50], Val Losses: mse: 28.5654, mae: 3.3527, huber: 2.9357, swd: 13.8371, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.1208, mae: 2.8008, huber: 2.3764, swd: 10.5206, target_std: 18.3425
      Epoch 3 composite train-obj: 2.566681
            Val objective improved 2.9477 → 2.9357, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 30.9949, mae: 2.9615, huber: 2.5504, swd: 16.9182, target_std: 20.3586
    Epoch [4/50], Val Losses: mse: 29.0920, mae: 3.3988, huber: 2.9787, swd: 13.8919, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 20.6615, mae: 2.8219, huber: 2.3958, swd: 10.8526, target_std: 18.3425
      Epoch 4 composite train-obj: 2.550402
            No improvement (2.9787), counter 1/5
    Epoch [5/50], Train Losses: mse: 30.6758, mae: 2.9453, huber: 2.5343, swd: 16.6385, target_std: 20.3583
    Epoch [5/50], Val Losses: mse: 28.4290, mae: 3.3671, huber: 2.9478, swd: 13.3594, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 20.6715, mae: 2.8219, huber: 2.3956, swd: 10.8798, target_std: 18.3425
      Epoch 5 composite train-obj: 2.534253
            No improvement (2.9478), counter 2/5
    Epoch [6/50], Train Losses: mse: 30.3395, mae: 2.9309, huber: 2.5200, swd: 16.3397, target_std: 20.3597
    Epoch [6/50], Val Losses: mse: 28.7533, mae: 3.3718, huber: 2.9529, swd: 13.7003, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 20.5773, mae: 2.8135, huber: 2.3882, swd: 10.8472, target_std: 18.3425
      Epoch 6 composite train-obj: 2.520021
            No improvement (2.9529), counter 3/5
    Epoch [7/50], Train Losses: mse: 30.1460, mae: 2.9202, huber: 2.5095, swd: 16.1512, target_std: 20.3592
    Epoch [7/50], Val Losses: mse: 29.0117, mae: 3.3917, huber: 2.9724, swd: 13.9669, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 20.6897, mae: 2.8260, huber: 2.3998, swd: 10.9792, target_std: 18.3425
      Epoch 7 composite train-obj: 2.509492
            No improvement (2.9724), counter 4/5
    Epoch [8/50], Train Losses: mse: 29.7897, mae: 2.9053, huber: 2.4946, swd: 15.8228, target_std: 20.3587
    Epoch [8/50], Val Losses: mse: 28.6646, mae: 3.3672, huber: 2.9487, swd: 13.6958, target_std: 20.4999
    Epoch [8/50], Test Losses: mse: 20.2629, mae: 2.8031, huber: 2.3779, swd: 10.5679, target_std: 18.3425
      Epoch 8 composite train-obj: 2.494626
    Epoch [8/50], Test Losses: mse: 20.1208, mae: 2.8008, huber: 2.3764, swd: 10.5206, target_std: 18.3425
    Best round's Test MSE: 20.1208, MAE: 2.8008, SWD: 10.5206
    Best round's Validation MSE: 28.5654, MAE: 3.3527
    Best round's Test verification MSE : 20.1208, MAE: 2.8008, SWD: 10.5206
    
    ==================================================
    Experiment Summary (PatchTST_ettm2_seq96_pred720_20250430_2004)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 20.0305 ± 0.0698
      mae: 2.7916 ± 0.0066
      huber: 2.3672 ± 0.0066
      swd: 9.7694 ± 0.6045
      target_std: 18.3425 ± 0.0000
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 28.7286 ± 0.1905
      mae: 3.3612 ± 0.0133
      huber: 2.9432 ± 0.0120
      swd: 13.2746 ± 0.6075
      target_std: 20.4999 ± 0.0000
      count: 49.0000 ± 0.0000
    ==================================================
    
    Experiment complete: PatchTST_ettm2_seq96_pred720_20250430_2004
    Model: PatchTST
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### DLinear

#### pred=96


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 380
    Validation Batches: 53
    Test Batches: 108
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 30.9079, mae: 2.5426, huber: 2.1390, swd: 11.0383, target_std: 20.3271
    Epoch [1/50], Val Losses: mse: 14.2877, mae: 2.2261, huber: 1.8330, swd: 8.0126, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.1002, mae: 1.9578, huber: 1.5512, swd: 4.9503, target_std: 18.4467
      Epoch 1 composite train-obj: 2.139015
            Val objective improved inf → 1.8330, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.7052, mae: 1.8963, huber: 1.5048, swd: 7.0511, target_std: 20.3271
    Epoch [2/50], Val Losses: mse: 13.7129, mae: 2.1749, huber: 1.7838, swd: 7.5752, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.8649, mae: 1.9141, huber: 1.5113, swd: 4.8239, target_std: 18.4467
      Epoch 2 composite train-obj: 1.504847
            Val objective improved 1.8330 → 1.7838, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.4422, mae: 1.8769, huber: 1.4860, swd: 6.8445, target_std: 20.3276
    Epoch [3/50], Val Losses: mse: 13.4996, mae: 2.1617, huber: 1.7701, swd: 7.3932, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.8008, mae: 1.9055, huber: 1.5034, swd: 4.7934, target_std: 18.4467
      Epoch 3 composite train-obj: 1.486025
            Val objective improved 1.7838 → 1.7701, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.3647, mae: 1.8727, huber: 1.4820, swd: 6.7823, target_std: 20.3259
    Epoch [4/50], Val Losses: mse: 13.4225, mae: 2.1570, huber: 1.7648, swd: 7.3293, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.8093, mae: 1.9150, huber: 1.5123, swd: 4.8061, target_std: 18.4467
      Epoch 4 composite train-obj: 1.481966
            Val objective improved 1.7701 → 1.7648, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 12.3446, mae: 1.8742, huber: 1.4831, swd: 6.7593, target_std: 20.3274
    Epoch [5/50], Val Losses: mse: 13.4545, mae: 2.1611, huber: 1.7686, swd: 7.3298, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.8158, mae: 1.9078, huber: 1.5051, swd: 4.8091, target_std: 18.4467
      Epoch 5 composite train-obj: 1.483115
            No improvement (1.7686), counter 1/5
    Epoch [6/50], Train Losses: mse: 12.3137, mae: 1.8718, huber: 1.4807, swd: 6.7365, target_std: 20.3276
    Epoch [6/50], Val Losses: mse: 13.4600, mae: 2.1644, huber: 1.7711, swd: 7.3223, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.8085, mae: 1.9036, huber: 1.5017, swd: 4.7902, target_std: 18.4467
      Epoch 6 composite train-obj: 1.480739
            No improvement (1.7711), counter 2/5
    Epoch [7/50], Train Losses: mse: 12.3047, mae: 1.8714, huber: 1.4805, swd: 6.7282, target_std: 20.3265
    Epoch [7/50], Val Losses: mse: 13.4659, mae: 2.1684, huber: 1.7750, swd: 7.3228, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.8362, mae: 1.9091, huber: 1.5064, swd: 4.8091, target_std: 18.4467
      Epoch 7 composite train-obj: 1.480515
            No improvement (1.7750), counter 3/5
    Epoch [8/50], Train Losses: mse: 12.3016, mae: 1.8708, huber: 1.4801, swd: 6.7270, target_std: 20.3277
    Epoch [8/50], Val Losses: mse: 13.4219, mae: 2.1642, huber: 1.7706, swd: 7.2828, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.7987, mae: 1.9066, huber: 1.5044, swd: 4.7801, target_std: 18.4467
      Epoch 8 composite train-obj: 1.480053
            No improvement (1.7706), counter 4/5
    Epoch [9/50], Train Losses: mse: 12.2796, mae: 1.8698, huber: 1.4787, swd: 6.7055, target_std: 20.3279
    Epoch [9/50], Val Losses: mse: 13.3136, mae: 2.1553, huber: 1.7621, swd: 7.2289, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.7704, mae: 1.9136, huber: 1.5108, swd: 4.7763, target_std: 18.4467
      Epoch 9 composite train-obj: 1.478716
            Val objective improved 1.7648 → 1.7621, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 12.2686, mae: 1.8693, huber: 1.4784, swd: 6.7020, target_std: 20.3265
    Epoch [10/50], Val Losses: mse: 13.3321, mae: 2.1605, huber: 1.7668, swd: 7.2357, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.7288, mae: 1.9056, huber: 1.5033, swd: 4.7488, target_std: 18.4467
      Epoch 10 composite train-obj: 1.478393
            No improvement (1.7668), counter 1/5
    Epoch [11/50], Train Losses: mse: 12.2661, mae: 1.8696, huber: 1.4788, swd: 6.7004, target_std: 20.3285
    Epoch [11/50], Val Losses: mse: 13.4885, mae: 2.1688, huber: 1.7756, swd: 7.3261, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.8529, mae: 1.9052, huber: 1.5034, swd: 4.8157, target_std: 18.4467
      Epoch 11 composite train-obj: 1.478835
            No improvement (1.7756), counter 2/5
    Epoch [12/50], Train Losses: mse: 12.2555, mae: 1.8681, huber: 1.4772, swd: 6.6861, target_std: 20.3280
    Epoch [12/50], Val Losses: mse: 13.3441, mae: 2.1587, huber: 1.7651, swd: 7.2307, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 9.7419, mae: 1.9049, huber: 1.5024, swd: 4.7354, target_std: 18.4467
      Epoch 12 composite train-obj: 1.477176
            No improvement (1.7651), counter 3/5
    Epoch [13/50], Train Losses: mse: 12.2454, mae: 1.8680, huber: 1.4772, swd: 6.6830, target_std: 20.3272
    Epoch [13/50], Val Losses: mse: 13.4200, mae: 2.1680, huber: 1.7743, swd: 7.2695, target_std: 20.5171
    Epoch [13/50], Test Losses: mse: 9.7606, mae: 1.9019, huber: 1.5002, swd: 4.7302, target_std: 18.4467
      Epoch 13 composite train-obj: 1.477159
            No improvement (1.7743), counter 4/5
    Epoch [14/50], Train Losses: mse: 12.2385, mae: 1.8685, huber: 1.4777, swd: 6.6812, target_std: 20.3258
    Epoch [14/50], Val Losses: mse: 13.3354, mae: 2.1572, huber: 1.7638, swd: 7.2126, target_std: 20.5171
    Epoch [14/50], Test Losses: mse: 9.7486, mae: 1.9058, huber: 1.5031, swd: 4.7379, target_std: 18.4467
      Epoch 14 composite train-obj: 1.477653
    Epoch [14/50], Test Losses: mse: 9.7704, mae: 1.9136, huber: 1.5108, swd: 4.7763, target_std: 18.4467
    Best round's Test MSE: 9.7704, MAE: 1.9136, SWD: 4.7763
    Best round's Validation MSE: 13.3136, MAE: 2.1553
    Best round's Test verification MSE : 9.7704, MAE: 1.9136, SWD: 4.7763
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.4578, mae: 2.5495, huber: 2.1463, swd: 10.2751, target_std: 20.3276
    Epoch [1/50], Val Losses: mse: 14.4153, mae: 2.2304, huber: 1.8373, swd: 7.7961, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.1360, mae: 1.9545, huber: 1.5483, swd: 4.7771, target_std: 18.4467
      Epoch 1 composite train-obj: 2.146338
            Val objective improved inf → 1.8373, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.7237, mae: 1.8958, huber: 1.5044, swd: 6.7885, target_std: 20.3271
    Epoch [2/50], Val Losses: mse: 13.7000, mae: 2.1724, huber: 1.7814, swd: 7.2834, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.8498, mae: 1.9125, huber: 1.5096, swd: 4.6147, target_std: 18.4467
      Epoch 2 composite train-obj: 1.504394
            Val objective improved 1.8373 → 1.7814, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.4461, mae: 1.8766, huber: 1.4859, swd: 6.5832, target_std: 20.3279
    Epoch [3/50], Val Losses: mse: 13.5105, mae: 2.1635, huber: 1.7719, swd: 7.1206, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.7986, mae: 1.9078, huber: 1.5052, swd: 4.5918, target_std: 18.4467
      Epoch 3 composite train-obj: 1.485853
            Val objective improved 1.7814 → 1.7719, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.3645, mae: 1.8719, huber: 1.4813, swd: 6.5223, target_std: 20.3265
    Epoch [4/50], Val Losses: mse: 13.5096, mae: 2.1645, huber: 1.7719, swd: 7.0732, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.8479, mae: 1.9113, huber: 1.5086, swd: 4.6198, target_std: 18.4467
      Epoch 4 composite train-obj: 1.481281
            Val objective improved 1.7719 → 1.7719, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 12.3243, mae: 1.8721, huber: 1.4813, swd: 6.4894, target_std: 20.3267
    Epoch [5/50], Val Losses: mse: 13.4431, mae: 2.1619, huber: 1.7690, swd: 7.0282, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.7835, mae: 1.9040, huber: 1.5021, swd: 4.5760, target_std: 18.4467
      Epoch 5 composite train-obj: 1.481276
            Val objective improved 1.7719 → 1.7690, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 12.3068, mae: 1.8716, huber: 1.4806, swd: 6.4811, target_std: 20.3268
    Epoch [6/50], Val Losses: mse: 13.4034, mae: 2.1606, huber: 1.7675, swd: 7.0053, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.7814, mae: 1.9051, huber: 1.5035, swd: 4.5735, target_std: 18.4467
      Epoch 6 composite train-obj: 1.480644
            Val objective improved 1.7690 → 1.7675, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 12.3007, mae: 1.8699, huber: 1.4791, swd: 6.4726, target_std: 20.3295
    Epoch [7/50], Val Losses: mse: 13.4247, mae: 2.1639, huber: 1.7705, swd: 7.0065, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.7968, mae: 1.9015, huber: 1.4996, swd: 4.5762, target_std: 18.4467
      Epoch 7 composite train-obj: 1.479133
            No improvement (1.7705), counter 1/5
    Epoch [8/50], Train Losses: mse: 12.2893, mae: 1.8709, huber: 1.4799, swd: 6.4628, target_std: 20.3276
    Epoch [8/50], Val Losses: mse: 13.5638, mae: 2.1759, huber: 1.7823, swd: 7.0864, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.9067, mae: 1.9102, huber: 1.5087, swd: 4.6425, target_std: 18.4467
      Epoch 8 composite train-obj: 1.479901
            No improvement (1.7823), counter 2/5
    Epoch [9/50], Train Losses: mse: 12.2748, mae: 1.8700, huber: 1.4790, swd: 6.4523, target_std: 20.3272
    Epoch [9/50], Val Losses: mse: 13.3590, mae: 2.1584, huber: 1.7650, swd: 6.9644, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.7797, mae: 1.9086, huber: 1.5060, swd: 4.5717, target_std: 18.4467
      Epoch 9 composite train-obj: 1.479022
            Val objective improved 1.7675 → 1.7650, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 12.2784, mae: 1.8707, huber: 1.4798, swd: 6.4509, target_std: 20.3283
    Epoch [10/50], Val Losses: mse: 13.3700, mae: 2.1598, huber: 1.7668, swd: 6.9697, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.7558, mae: 1.9024, huber: 1.4999, swd: 4.5556, target_std: 18.4467
      Epoch 10 composite train-obj: 1.479779
            No improvement (1.7668), counter 1/5
    Epoch [11/50], Train Losses: mse: 12.2539, mae: 1.8693, huber: 1.4785, swd: 6.4340, target_std: 20.3270
    Epoch [11/50], Val Losses: mse: 13.3428, mae: 2.1613, huber: 1.7675, swd: 6.9506, target_std: 20.5171
    Epoch [11/50], Test Losses: mse: 9.7378, mae: 1.9070, huber: 1.5051, swd: 4.5342, target_std: 18.4467
      Epoch 11 composite train-obj: 1.478522
            No improvement (1.7675), counter 2/5
    Epoch [12/50], Train Losses: mse: 12.2541, mae: 1.8698, huber: 1.4788, swd: 6.4382, target_std: 20.3258
    Epoch [12/50], Val Losses: mse: 13.4050, mae: 2.1642, huber: 1.7705, swd: 6.9536, target_std: 20.5171
    Epoch [12/50], Test Losses: mse: 9.8068, mae: 1.9047, huber: 1.5030, swd: 4.5444, target_std: 18.4467
      Epoch 12 composite train-obj: 1.478764
            No improvement (1.7705), counter 3/5
    Epoch [13/50], Train Losses: mse: 12.2440, mae: 1.8682, huber: 1.4773, swd: 6.4245, target_std: 20.3276
    Epoch [13/50], Val Losses: mse: 13.2542, mae: 2.1499, huber: 1.7566, swd: 6.9213, target_std: 20.5171
    Epoch [13/50], Test Losses: mse: 9.7176, mae: 1.9106, huber: 1.5078, swd: 4.5538, target_std: 18.4467
      Epoch 13 composite train-obj: 1.477297
            Val objective improved 1.7650 → 1.7566, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 12.2695, mae: 1.8694, huber: 1.4785, swd: 6.4426, target_std: 20.3283
    Epoch [14/50], Val Losses: mse: 13.4588, mae: 2.1701, huber: 1.7764, swd: 7.0020, target_std: 20.5171
    Epoch [14/50], Test Losses: mse: 9.8448, mae: 1.9061, huber: 1.5041, swd: 4.5915, target_std: 18.4467
      Epoch 14 composite train-obj: 1.478535
            No improvement (1.7764), counter 1/5
    Epoch [15/50], Train Losses: mse: 12.2436, mae: 1.8689, huber: 1.4779, swd: 6.4250, target_std: 20.3277
    Epoch [15/50], Val Losses: mse: 13.4409, mae: 2.1706, huber: 1.7767, swd: 6.9819, target_std: 20.5171
    Epoch [15/50], Test Losses: mse: 9.8262, mae: 1.9041, huber: 1.5019, swd: 4.5648, target_std: 18.4467
      Epoch 15 composite train-obj: 1.477882
            No improvement (1.7767), counter 2/5
    Epoch [16/50], Train Losses: mse: 12.2436, mae: 1.8695, huber: 1.4785, swd: 6.4254, target_std: 20.3262
    Epoch [16/50], Val Losses: mse: 13.3237, mae: 2.1599, huber: 1.7662, swd: 6.9390, target_std: 20.5171
    Epoch [16/50], Test Losses: mse: 9.7341, mae: 1.9058, huber: 1.5036, swd: 4.5292, target_std: 18.4467
      Epoch 16 composite train-obj: 1.478515
            No improvement (1.7662), counter 3/5
    Epoch [17/50], Train Losses: mse: 12.2222, mae: 1.8680, huber: 1.4769, swd: 6.4156, target_std: 20.3259
    Epoch [17/50], Val Losses: mse: 13.4306, mae: 2.1690, huber: 1.7749, swd: 6.9700, target_std: 20.5171
    Epoch [17/50], Test Losses: mse: 9.8135, mae: 1.9061, huber: 1.5038, swd: 4.5617, target_std: 18.4467
      Epoch 17 composite train-obj: 1.476939
            No improvement (1.7749), counter 4/5
    Epoch [18/50], Train Losses: mse: 12.2256, mae: 1.8675, huber: 1.4766, swd: 6.4100, target_std: 20.3274
    Epoch [18/50], Val Losses: mse: 13.4191, mae: 2.1689, huber: 1.7750, swd: 6.9744, target_std: 20.5171
    Epoch [18/50], Test Losses: mse: 9.8212, mae: 1.9013, huber: 1.5002, swd: 4.5784, target_std: 18.4467
      Epoch 18 composite train-obj: 1.476647
    Epoch [18/50], Test Losses: mse: 9.7176, mae: 1.9106, huber: 1.5078, swd: 4.5538, target_std: 18.4467
    Best round's Test MSE: 9.7176, MAE: 1.9106, SWD: 4.5538
    Best round's Validation MSE: 13.2542, MAE: 2.1499
    Best round's Test verification MSE : 9.7176, MAE: 1.9106, SWD: 4.5538
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 30.1212, mae: 2.5219, huber: 2.1186, swd: 9.6352, target_std: 20.3265
    Epoch [1/50], Val Losses: mse: 14.3584, mae: 2.2298, huber: 1.8365, swd: 7.3257, target_std: 20.5171
    Epoch [1/50], Test Losses: mse: 10.0976, mae: 1.9608, huber: 1.5534, swd: 4.4807, target_std: 18.4467
      Epoch 1 composite train-obj: 2.118561
            Val objective improved inf → 1.8365, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.7206, mae: 1.8969, huber: 1.5054, swd: 6.3500, target_std: 20.3259
    Epoch [2/50], Val Losses: mse: 13.6698, mae: 2.1719, huber: 1.7807, swd: 6.8400, target_std: 20.5171
    Epoch [2/50], Test Losses: mse: 9.8234, mae: 1.9150, huber: 1.5114, swd: 4.3178, target_std: 18.4467
      Epoch 2 composite train-obj: 1.505393
            Val objective improved 1.8365 → 1.7807, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.4479, mae: 1.8761, huber: 1.4854, swd: 6.1585, target_std: 20.3266
    Epoch [3/50], Val Losses: mse: 13.5089, mae: 2.1626, huber: 1.7707, swd: 6.7019, target_std: 20.5171
    Epoch [3/50], Test Losses: mse: 9.7950, mae: 1.9068, huber: 1.5044, swd: 4.3160, target_std: 18.4467
      Epoch 3 composite train-obj: 1.485374
            Val objective improved 1.7807 → 1.7707, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.3735, mae: 1.8736, huber: 1.4828, swd: 6.1041, target_std: 20.3268
    Epoch [4/50], Val Losses: mse: 13.3974, mae: 2.1576, huber: 1.7650, swd: 6.6210, target_std: 20.5171
    Epoch [4/50], Test Losses: mse: 9.7943, mae: 1.9135, huber: 1.5113, swd: 4.3181, target_std: 18.4467
      Epoch 4 composite train-obj: 1.482765
            Val objective improved 1.7707 → 1.7650, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 12.3334, mae: 1.8727, huber: 1.4817, swd: 6.0703, target_std: 20.3263
    Epoch [5/50], Val Losses: mse: 13.3707, mae: 2.1566, huber: 1.7638, swd: 6.5810, target_std: 20.5171
    Epoch [5/50], Test Losses: mse: 9.7624, mae: 1.9059, huber: 1.5036, swd: 4.2946, target_std: 18.4467
      Epoch 5 composite train-obj: 1.481697
            Val objective improved 1.7650 → 1.7638, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 12.3083, mae: 1.8728, huber: 1.4817, swd: 6.0516, target_std: 20.3273
    Epoch [6/50], Val Losses: mse: 13.4127, mae: 2.1612, huber: 1.7681, swd: 6.5998, target_std: 20.5171
    Epoch [6/50], Test Losses: mse: 9.7711, mae: 1.9015, huber: 1.4997, swd: 4.2912, target_std: 18.4467
      Epoch 6 composite train-obj: 1.481689
            No improvement (1.7681), counter 1/5
    Epoch [7/50], Train Losses: mse: 12.2918, mae: 1.8702, huber: 1.4794, swd: 6.0389, target_std: 20.3271
    Epoch [7/50], Val Losses: mse: 13.3871, mae: 2.1633, huber: 1.7698, swd: 6.5960, target_std: 20.5171
    Epoch [7/50], Test Losses: mse: 9.7593, mae: 1.9041, huber: 1.5017, swd: 4.2970, target_std: 18.4467
      Epoch 7 composite train-obj: 1.479358
            No improvement (1.7698), counter 2/5
    Epoch [8/50], Train Losses: mse: 12.2772, mae: 1.8710, huber: 1.4798, swd: 6.0302, target_std: 20.3263
    Epoch [8/50], Val Losses: mse: 13.4288, mae: 2.1625, huber: 1.7692, swd: 6.5699, target_std: 20.5171
    Epoch [8/50], Test Losses: mse: 9.8314, mae: 1.9128, huber: 1.5101, swd: 4.3134, target_std: 18.4467
      Epoch 8 composite train-obj: 1.479823
            No improvement (1.7692), counter 3/5
    Epoch [9/50], Train Losses: mse: 12.2702, mae: 1.8682, huber: 1.4775, swd: 6.0235, target_std: 20.3265
    Epoch [9/50], Val Losses: mse: 13.3918, mae: 2.1654, huber: 1.7716, swd: 6.5496, target_std: 20.5171
    Epoch [9/50], Test Losses: mse: 9.7916, mae: 1.9101, huber: 1.5079, swd: 4.2760, target_std: 18.4467
      Epoch 9 composite train-obj: 1.477524
            No improvement (1.7716), counter 4/5
    Epoch [10/50], Train Losses: mse: 12.2634, mae: 1.8685, huber: 1.4778, swd: 6.0221, target_std: 20.3261
    Epoch [10/50], Val Losses: mse: 13.4458, mae: 2.1691, huber: 1.7754, swd: 6.5963, target_std: 20.5171
    Epoch [10/50], Test Losses: mse: 9.8245, mae: 1.9046, huber: 1.5029, swd: 4.3082, target_std: 18.4467
      Epoch 10 composite train-obj: 1.477785
    Epoch [10/50], Test Losses: mse: 9.7624, mae: 1.9059, huber: 1.5036, swd: 4.2946, target_std: 18.4467
    Best round's Test MSE: 9.7624, MAE: 1.9059, SWD: 4.2946
    Best round's Validation MSE: 13.3707, MAE: 2.1566
    Best round's Test verification MSE : 9.7624, MAE: 1.9059, SWD: 4.2946
    
    ==================================================
    Experiment Summary (DLinear_ettm2_seq96_pred96_20250430_1902)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 9.7502 ± 0.0233
      mae: 1.9100 ± 0.0032
      huber: 1.5074 ± 0.0030
      swd: 4.5416 ± 0.1968
      target_std: 18.4467 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 13.3128 ± 0.0476
      mae: 2.1539 ± 0.0029
      huber: 1.7608 ± 0.0031
      swd: 6.9104 ± 0.2646
      target_std: 20.5171 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: DLinear_ettm2_seq96_pred96_20250430_1902
    Model: DLinear
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 379
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 379
    Validation Batches: 53
    Test Batches: 107
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 38.1977, mae: 2.8733, huber: 2.4642, swd: 14.5614, target_std: 20.3337
    Epoch [1/50], Val Losses: mse: 20.3757, mae: 2.6503, huber: 2.2475, swd: 11.5192, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 12.9335, mae: 2.2096, huber: 1.7963, swd: 6.2881, target_std: 18.4014
      Epoch 1 composite train-obj: 2.464197
            Val objective improved inf → 2.2475, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 18.3956, mae: 2.2496, huber: 1.8494, swd: 10.6172, target_std: 20.3339
    Epoch [2/50], Val Losses: mse: 19.7326, mae: 2.6024, huber: 2.1992, swd: 11.1573, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.6666, mae: 2.1835, huber: 1.7717, swd: 6.1800, target_std: 18.4014
      Epoch 2 composite train-obj: 1.849356
            Val objective improved 2.2475 → 2.1992, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 18.1039, mae: 2.2315, huber: 1.8316, swd: 10.3880, target_std: 20.3335
    Epoch [3/50], Val Losses: mse: 19.6097, mae: 2.6019, huber: 2.1966, swd: 11.0525, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.6304, mae: 2.1848, huber: 1.7725, swd: 6.1550, target_std: 18.4014
      Epoch 3 composite train-obj: 1.831611
            Val objective improved 2.1992 → 2.1966, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 18.0083, mae: 2.2278, huber: 1.8276, swd: 10.3110, target_std: 20.3331
    Epoch [4/50], Val Losses: mse: 19.6669, mae: 2.6077, huber: 2.2015, swd: 11.0789, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.6810, mae: 2.1790, huber: 1.7679, swd: 6.1789, target_std: 18.4014
      Epoch 4 composite train-obj: 1.827598
            No improvement (2.2015), counter 1/5
    Epoch [5/50], Train Losses: mse: 17.9443, mae: 2.2260, huber: 1.8255, swd: 10.2477, target_std: 20.3338
    Epoch [5/50], Val Losses: mse: 19.5261, mae: 2.5980, huber: 2.1913, swd: 10.9482, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.6067, mae: 2.1735, huber: 1.7626, swd: 6.1296, target_std: 18.4014
      Epoch 5 composite train-obj: 1.825502
            Val objective improved 2.1966 → 2.1913, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 17.9381, mae: 2.2277, huber: 1.8271, swd: 10.2296, target_std: 20.3335
    Epoch [6/50], Val Losses: mse: 19.5527, mae: 2.6042, huber: 2.1971, swd: 10.9451, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.6036, mae: 2.1801, huber: 1.7681, swd: 6.0979, target_std: 18.4014
      Epoch 6 composite train-obj: 1.827051
            No improvement (2.1971), counter 1/5
    Epoch [7/50], Train Losses: mse: 17.8862, mae: 2.2235, huber: 1.8230, swd: 10.1925, target_std: 20.3325
    Epoch [7/50], Val Losses: mse: 19.5670, mae: 2.6068, huber: 2.1993, swd: 10.9287, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 12.6410, mae: 2.1795, huber: 1.7682, swd: 6.1176, target_std: 18.4014
      Epoch 7 composite train-obj: 1.823023
            No improvement (2.1993), counter 2/5
    Epoch [8/50], Train Losses: mse: 17.8730, mae: 2.2239, huber: 1.8233, swd: 10.1737, target_std: 20.3337
    Epoch [8/50], Val Losses: mse: 19.5655, mae: 2.6028, huber: 2.1955, swd: 10.9283, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 12.6493, mae: 2.1766, huber: 1.7655, swd: 6.1283, target_std: 18.4014
      Epoch 8 composite train-obj: 1.823349
            No improvement (2.1955), counter 3/5
    Epoch [9/50], Train Losses: mse: 17.8542, mae: 2.2253, huber: 1.8244, swd: 10.1575, target_std: 20.3335
    Epoch [9/50], Val Losses: mse: 19.5740, mae: 2.6088, huber: 2.2014, swd: 10.9546, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 12.5491, mae: 2.1689, huber: 1.7580, swd: 6.0404, target_std: 18.4014
      Epoch 9 composite train-obj: 1.824391
            No improvement (2.2014), counter 4/5
    Epoch [10/50], Train Losses: mse: 17.8332, mae: 2.2232, huber: 1.8226, swd: 10.1406, target_std: 20.3336
    Epoch [10/50], Val Losses: mse: 19.4998, mae: 2.6067, huber: 2.1990, swd: 10.8696, target_std: 20.5407
    Epoch [10/50], Test Losses: mse: 12.5342, mae: 2.1689, huber: 1.7580, swd: 6.0123, target_std: 18.4014
      Epoch 10 composite train-obj: 1.822639
    Epoch [10/50], Test Losses: mse: 12.6067, mae: 2.1735, huber: 1.7626, swd: 6.1296, target_std: 18.4014
    Best round's Test MSE: 12.6067, MAE: 2.1735, SWD: 6.1296
    Best round's Validation MSE: 19.5261, MAE: 2.5980
    Best round's Test verification MSE : 12.6067, MAE: 2.1735, SWD: 6.1296
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 35.3928, mae: 2.8328, huber: 2.4235, swd: 14.9655, target_std: 20.3337
    Epoch [1/50], Val Losses: mse: 20.3074, mae: 2.6463, huber: 2.2435, swd: 11.8353, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 12.8870, mae: 2.2119, huber: 1.7983, swd: 6.4410, target_std: 18.4014
      Epoch 1 composite train-obj: 2.423542
            Val objective improved inf → 2.2435, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 18.3678, mae: 2.2476, huber: 1.8476, swd: 10.9522, target_std: 20.3332
    Epoch [2/50], Val Losses: mse: 19.7867, mae: 2.6081, huber: 2.2048, swd: 11.5445, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.6802, mae: 2.1896, huber: 1.7773, swd: 6.3612, target_std: 18.4014
      Epoch 2 composite train-obj: 1.847612
            Val objective improved 2.2435 → 2.2048, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 18.1058, mae: 2.2316, huber: 1.8317, swd: 10.7294, target_std: 20.3335
    Epoch [3/50], Val Losses: mse: 19.6990, mae: 2.6066, huber: 2.2015, swd: 11.4213, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.6751, mae: 2.1826, huber: 1.7711, swd: 6.3342, target_std: 18.4014
      Epoch 3 composite train-obj: 1.831731
            Val objective improved 2.2048 → 2.2015, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 18.0011, mae: 2.2291, huber: 1.8287, swd: 10.6378, target_std: 20.3334
    Epoch [4/50], Val Losses: mse: 19.6518, mae: 2.6083, huber: 2.2020, swd: 11.3858, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.6807, mae: 2.1776, huber: 1.7667, swd: 6.3331, target_std: 18.4014
      Epoch 4 composite train-obj: 1.828700
            No improvement (2.2020), counter 1/5
    Epoch [5/50], Train Losses: mse: 17.9558, mae: 2.2259, huber: 1.8255, swd: 10.5952, target_std: 20.3337
    Epoch [5/50], Val Losses: mse: 19.5029, mae: 2.5983, huber: 2.1917, swd: 11.2902, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.5862, mae: 2.1722, huber: 1.7613, swd: 6.2698, target_std: 18.4014
      Epoch 5 composite train-obj: 1.825502
            Val objective improved 2.2015 → 2.1917, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 17.9209, mae: 2.2259, huber: 1.8253, swd: 10.5540, target_std: 20.3335
    Epoch [6/50], Val Losses: mse: 19.6671, mae: 2.6101, huber: 2.2024, swd: 11.3581, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.6992, mae: 2.1767, huber: 1.7658, swd: 6.3358, target_std: 18.4014
      Epoch 6 composite train-obj: 1.825323
            No improvement (2.2024), counter 1/5
    Epoch [7/50], Train Losses: mse: 17.8897, mae: 2.2238, huber: 1.8233, swd: 10.5332, target_std: 20.3337
    Epoch [7/50], Val Losses: mse: 19.5351, mae: 2.6055, huber: 2.1979, swd: 11.2871, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 12.5864, mae: 2.1743, huber: 1.7628, swd: 6.2520, target_std: 18.4014
      Epoch 7 composite train-obj: 1.823320
            No improvement (2.1979), counter 2/5
    Epoch [8/50], Train Losses: mse: 17.8531, mae: 2.2228, huber: 1.8222, swd: 10.5026, target_std: 20.3339
    Epoch [8/50], Val Losses: mse: 19.4784, mae: 2.5996, huber: 2.1920, swd: 11.2139, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 12.6052, mae: 2.1796, huber: 1.7682, swd: 6.2814, target_std: 18.4014
      Epoch 8 composite train-obj: 1.822232
            No improvement (2.1920), counter 3/5
    Epoch [9/50], Train Losses: mse: 17.8500, mae: 2.2231, huber: 1.8223, swd: 10.4961, target_std: 20.3337
    Epoch [9/50], Val Losses: mse: 19.6495, mae: 2.6133, huber: 2.2056, swd: 11.2969, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 12.6943, mae: 2.1788, huber: 1.7678, swd: 6.3152, target_std: 18.4014
      Epoch 9 composite train-obj: 1.822333
            No improvement (2.2056), counter 4/5
    Epoch [10/50], Train Losses: mse: 17.8145, mae: 2.2214, huber: 1.8207, swd: 10.4691, target_std: 20.3332
    Epoch [10/50], Val Losses: mse: 19.5176, mae: 2.6054, huber: 2.1975, swd: 11.2091, target_std: 20.5407
    Epoch [10/50], Test Losses: mse: 12.5742, mae: 2.1683, huber: 1.7576, swd: 6.2167, target_std: 18.4014
      Epoch 10 composite train-obj: 1.820708
    Epoch [10/50], Test Losses: mse: 12.5862, mae: 2.1722, huber: 1.7613, swd: 6.2698, target_std: 18.4014
    Best round's Test MSE: 12.5862, MAE: 2.1722, SWD: 6.2698
    Best round's Validation MSE: 19.5029, MAE: 2.5983
    Best round's Test verification MSE : 12.5862, MAE: 2.1722, SWD: 6.2698
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 36.7536, mae: 2.8535, huber: 2.4445, swd: 12.2727, target_std: 20.3334
    Epoch [1/50], Val Losses: mse: 20.2529, mae: 2.6411, huber: 2.2389, swd: 10.0815, target_std: 20.5407
    Epoch [1/50], Test Losses: mse: 12.8586, mae: 2.2155, huber: 1.8016, swd: 5.4932, target_std: 18.4014
      Epoch 1 composite train-obj: 2.444455
            Val objective improved inf → 2.2389, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 18.3923, mae: 2.2495, huber: 1.8494, swd: 9.2954, target_std: 20.3338
    Epoch [2/50], Val Losses: mse: 19.7518, mae: 2.6056, huber: 2.2025, swd: 9.7793, target_std: 20.5407
    Epoch [2/50], Test Losses: mse: 12.6614, mae: 2.1827, huber: 1.7708, swd: 5.3724, target_std: 18.4014
      Epoch 2 composite train-obj: 1.849433
            Val objective improved 2.2389 → 2.2025, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 18.1000, mae: 2.2329, huber: 1.8328, swd: 9.0808, target_std: 20.3336
    Epoch [3/50], Val Losses: mse: 19.5498, mae: 2.5988, huber: 2.1939, swd: 9.6886, target_std: 20.5407
    Epoch [3/50], Test Losses: mse: 12.6360, mae: 2.1914, huber: 1.7792, swd: 5.4125, target_std: 18.4014
      Epoch 3 composite train-obj: 1.832761
            Val objective improved 2.2025 → 2.1939, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 17.9953, mae: 2.2273, huber: 1.8271, swd: 8.9975, target_std: 20.3334
    Epoch [4/50], Val Losses: mse: 19.6944, mae: 2.6102, huber: 2.2041, swd: 9.6670, target_std: 20.5407
    Epoch [4/50], Test Losses: mse: 12.6939, mae: 2.1759, huber: 1.7653, swd: 5.3678, target_std: 18.4014
      Epoch 4 composite train-obj: 1.827103
            No improvement (2.2041), counter 1/5
    Epoch [5/50], Train Losses: mse: 17.9490, mae: 2.2262, huber: 1.8256, swd: 8.9576, target_std: 20.3341
    Epoch [5/50], Val Losses: mse: 19.5305, mae: 2.5985, huber: 2.1918, swd: 9.6427, target_std: 20.5407
    Epoch [5/50], Test Losses: mse: 12.5642, mae: 2.1764, huber: 1.7649, swd: 5.3353, target_std: 18.4014
      Epoch 5 composite train-obj: 1.825635
            Val objective improved 2.1939 → 2.1918, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 17.9087, mae: 2.2245, huber: 1.8241, swd: 8.9261, target_std: 20.3332
    Epoch [6/50], Val Losses: mse: 19.5552, mae: 2.6073, huber: 2.1998, swd: 9.5921, target_std: 20.5407
    Epoch [6/50], Test Losses: mse: 12.6082, mae: 2.1745, huber: 1.7635, swd: 5.3095, target_std: 18.4014
      Epoch 6 composite train-obj: 1.824082
            No improvement (2.1998), counter 1/5
    Epoch [7/50], Train Losses: mse: 17.8842, mae: 2.2248, huber: 1.8242, swd: 8.9028, target_std: 20.3336
    Epoch [7/50], Val Losses: mse: 19.5186, mae: 2.6022, huber: 2.1950, swd: 9.5623, target_std: 20.5407
    Epoch [7/50], Test Losses: mse: 12.5805, mae: 2.1711, huber: 1.7602, swd: 5.3092, target_std: 18.4014
      Epoch 7 composite train-obj: 1.824185
            No improvement (2.1950), counter 2/5
    Epoch [8/50], Train Losses: mse: 17.8573, mae: 2.2237, huber: 1.8230, swd: 8.8831, target_std: 20.3331
    Epoch [8/50], Val Losses: mse: 19.4936, mae: 2.6023, huber: 2.1949, swd: 9.5382, target_std: 20.5407
    Epoch [8/50], Test Losses: mse: 12.5414, mae: 2.1689, huber: 1.7579, swd: 5.2733, target_std: 18.4014
      Epoch 8 composite train-obj: 1.822983
            No improvement (2.1949), counter 3/5
    Epoch [9/50], Train Losses: mse: 17.8430, mae: 2.2235, huber: 1.8228, swd: 8.8673, target_std: 20.3332
    Epoch [9/50], Val Losses: mse: 19.4360, mae: 2.5994, huber: 2.1921, swd: 9.4837, target_std: 20.5407
    Epoch [9/50], Test Losses: mse: 12.5210, mae: 2.1702, huber: 1.7592, swd: 5.2585, target_std: 18.4014
      Epoch 9 composite train-obj: 1.822828
            No improvement (2.1921), counter 4/5
    Epoch [10/50], Train Losses: mse: 17.8268, mae: 2.2220, huber: 1.8214, swd: 8.8550, target_std: 20.3331
    Epoch [10/50], Val Losses: mse: 19.5343, mae: 2.6045, huber: 2.1971, swd: 9.5374, target_std: 20.5407
    Epoch [10/50], Test Losses: mse: 12.6503, mae: 2.1739, huber: 1.7629, swd: 5.3357, target_std: 18.4014
      Epoch 10 composite train-obj: 1.821448
    Epoch [10/50], Test Losses: mse: 12.5642, mae: 2.1764, huber: 1.7649, swd: 5.3353, target_std: 18.4014
    Best round's Test MSE: 12.5642, MAE: 2.1764, SWD: 5.3353
    Best round's Validation MSE: 19.5305, MAE: 2.5985
    Best round's Test verification MSE : 12.5642, MAE: 2.1764, SWD: 5.3353
    
    ==================================================
    Experiment Summary (DLinear_ettm2_seq96_pred196_20250430_1921)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.5857 ± 0.0174
      mae: 2.1740 ± 0.0018
      huber: 1.7629 ± 0.0015
      swd: 5.9116 ± 0.4115
      target_std: 18.4014 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 19.5198 ± 0.0121
      mae: 2.5983 ± 0.0002
      huber: 2.1916 ± 0.0002
      swd: 10.6270 ± 0.7099
      target_std: 20.5407 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: DLinear_ettm2_seq96_pred196_20250430_1921
    Model: DLinear
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 378
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 378
    Validation Batches: 52
    Test Batches: 106
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 42.2856, mae: 3.1344, huber: 2.7199, swd: 16.7474, target_std: 20.3413
    Epoch [1/50], Val Losses: mse: 24.5980, mae: 2.9813, huber: 2.5708, swd: 12.7529, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 16.0924, mae: 2.4689, huber: 2.0494, swd: 7.8488, target_std: 18.3689
      Epoch 1 composite train-obj: 2.719912
            Val objective improved inf → 2.5708, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 24.2548, mae: 2.5644, huber: 2.1570, swd: 13.6450, target_std: 20.3417
    Epoch [2/50], Val Losses: mse: 24.2539, mae: 2.9546, huber: 2.5420, swd: 12.5285, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 16.0447, mae: 2.4524, huber: 2.0342, swd: 7.8150, target_std: 18.3689
      Epoch 2 composite train-obj: 2.157050
            Val objective improved 2.5708 → 2.5420, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 24.0104, mae: 2.5497, huber: 2.1422, swd: 13.4574, target_std: 20.3410
    Epoch [3/50], Val Losses: mse: 24.1433, mae: 2.9504, huber: 2.5358, swd: 12.4931, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 15.9060, mae: 2.4455, huber: 2.0273, swd: 7.6883, target_std: 18.3689
      Epoch 3 composite train-obj: 2.142206
            Val objective improved 2.5420 → 2.5358, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 23.9014, mae: 2.5464, huber: 2.1386, swd: 13.3557, target_std: 20.3418
    Epoch [4/50], Val Losses: mse: 24.1735, mae: 2.9592, huber: 2.5428, swd: 12.4405, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 15.8965, mae: 2.4396, huber: 2.0216, swd: 7.6495, target_std: 18.3689
      Epoch 4 composite train-obj: 2.138613
            No improvement (2.5428), counter 1/5
    Epoch [5/50], Train Losses: mse: 23.8342, mae: 2.5433, huber: 2.1355, swd: 13.2946, target_std: 20.3414
    Epoch [5/50], Val Losses: mse: 24.1874, mae: 2.9612, huber: 2.5443, swd: 12.5076, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 15.7628, mae: 2.4288, huber: 2.0112, swd: 7.5767, target_std: 18.3689
      Epoch 5 composite train-obj: 2.135545
            No improvement (2.5443), counter 2/5
    Epoch [6/50], Train Losses: mse: 23.7832, mae: 2.5424, huber: 2.1342, swd: 13.2468, target_std: 20.3412
    Epoch [6/50], Val Losses: mse: 24.0522, mae: 2.9559, huber: 2.5382, swd: 12.3342, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 15.8038, mae: 2.4352, huber: 2.0167, swd: 7.5997, target_std: 18.3689
      Epoch 6 composite train-obj: 2.134233
            No improvement (2.5382), counter 3/5
    Epoch [7/50], Train Losses: mse: 23.7217, mae: 2.5411, huber: 2.1329, swd: 13.1977, target_std: 20.3415
    Epoch [7/50], Val Losses: mse: 24.0680, mae: 2.9582, huber: 2.5407, swd: 12.3119, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 15.7954, mae: 2.4307, huber: 2.0129, swd: 7.5419, target_std: 18.3689
      Epoch 7 composite train-obj: 2.132866
            No improvement (2.5407), counter 4/5
    Epoch [8/50], Train Losses: mse: 23.7040, mae: 2.5403, huber: 2.1320, swd: 13.1808, target_std: 20.3415
    Epoch [8/50], Val Losses: mse: 23.9896, mae: 2.9524, huber: 2.5350, swd: 12.2653, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 15.7141, mae: 2.4257, huber: 2.0085, swd: 7.5108, target_std: 18.3689
      Epoch 8 composite train-obj: 2.131999
            Val objective improved 2.5358 → 2.5350, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 23.6479, mae: 2.5389, huber: 2.1307, swd: 13.1302, target_std: 20.3416
    Epoch [9/50], Val Losses: mse: 23.9876, mae: 2.9559, huber: 2.5380, swd: 12.2629, target_std: 20.5279
    Epoch [9/50], Test Losses: mse: 15.6902, mae: 2.4330, huber: 2.0139, swd: 7.4887, target_std: 18.3689
      Epoch 9 composite train-obj: 2.130714
            No improvement (2.5380), counter 1/5
    Epoch [10/50], Train Losses: mse: 23.6375, mae: 2.5398, huber: 2.1313, swd: 13.1125, target_std: 20.3414
    Epoch [10/50], Val Losses: mse: 24.2356, mae: 2.9709, huber: 2.5521, swd: 12.3257, target_std: 20.5279
    Epoch [10/50], Test Losses: mse: 16.1061, mae: 2.4538, huber: 2.0352, swd: 7.7181, target_std: 18.3689
      Epoch 10 composite train-obj: 2.131340
            No improvement (2.5521), counter 2/5
    Epoch [11/50], Train Losses: mse: 23.5871, mae: 2.5362, huber: 2.1278, swd: 13.0713, target_std: 20.3410
    Epoch [11/50], Val Losses: mse: 24.0268, mae: 2.9600, huber: 2.5420, swd: 12.2260, target_std: 20.5279
    Epoch [11/50], Test Losses: mse: 15.7303, mae: 2.4257, huber: 2.0081, swd: 7.4899, target_std: 18.3689
      Epoch 11 composite train-obj: 2.127821
            No improvement (2.5420), counter 3/5
    Epoch [12/50], Train Losses: mse: 23.5465, mae: 2.5356, huber: 2.1273, swd: 13.0434, target_std: 20.3408
    Epoch [12/50], Val Losses: mse: 23.9998, mae: 2.9575, huber: 2.5399, swd: 12.2094, target_std: 20.5279
    Epoch [12/50], Test Losses: mse: 15.7119, mae: 2.4202, huber: 2.0031, swd: 7.4605, target_std: 18.3689
      Epoch 12 composite train-obj: 2.127315
            No improvement (2.5399), counter 4/5
    Epoch [13/50], Train Losses: mse: 23.5421, mae: 2.5357, huber: 2.1275, swd: 13.0371, target_std: 20.3416
    Epoch [13/50], Val Losses: mse: 24.0510, mae: 2.9631, huber: 2.5452, swd: 12.1893, target_std: 20.5279
    Epoch [13/50], Test Losses: mse: 15.8189, mae: 2.4274, huber: 2.0100, swd: 7.5024, target_std: 18.3689
      Epoch 13 composite train-obj: 2.127490
    Epoch [13/50], Test Losses: mse: 15.7141, mae: 2.4257, huber: 2.0085, swd: 7.5108, target_std: 18.3689
    Best round's Test MSE: 15.7141, MAE: 2.4257, SWD: 7.5108
    Best round's Validation MSE: 23.9896, MAE: 2.9524
    Best round's Test verification MSE : 15.7141, MAE: 2.4257, SWD: 7.5108
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 41.5647, mae: 3.1219, huber: 2.7072, swd: 17.6294, target_std: 20.3403
    Epoch [1/50], Val Losses: mse: 24.6413, mae: 2.9853, huber: 2.5748, swd: 13.3220, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 16.1481, mae: 2.4738, huber: 2.0536, swd: 8.2817, target_std: 18.3689
      Epoch 1 composite train-obj: 2.707240
            Val objective improved inf → 2.5748, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 24.2524, mae: 2.5645, huber: 2.1571, swd: 14.3332, target_std: 20.3408
    Epoch [2/50], Val Losses: mse: 24.3075, mae: 2.9586, huber: 2.5462, swd: 13.1256, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 15.9622, mae: 2.4429, huber: 2.0250, swd: 8.1365, target_std: 18.3689
      Epoch 2 composite train-obj: 2.157083
            Val objective improved 2.5748 → 2.5462, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 23.9936, mae: 2.5490, huber: 2.1414, swd: 14.1259, target_std: 20.3415
    Epoch [3/50], Val Losses: mse: 24.1941, mae: 2.9560, huber: 2.5412, swd: 13.0506, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 15.8821, mae: 2.4336, huber: 2.0162, swd: 8.0688, target_std: 18.3689
      Epoch 3 composite train-obj: 2.141441
            Val objective improved 2.5462 → 2.5412, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 23.9011, mae: 2.5461, huber: 2.1383, swd: 14.0337, target_std: 20.3403
    Epoch [4/50], Val Losses: mse: 24.1990, mae: 2.9606, huber: 2.5443, swd: 13.0093, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 15.8856, mae: 2.4377, huber: 2.0196, swd: 8.0406, target_std: 18.3689
      Epoch 4 composite train-obj: 2.138340
            No improvement (2.5443), counter 1/5
    Epoch [5/50], Train Losses: mse: 23.8418, mae: 2.5452, huber: 2.1372, swd: 13.9737, target_std: 20.3422
    Epoch [5/50], Val Losses: mse: 24.2042, mae: 2.9610, huber: 2.5440, swd: 12.9687, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 15.8825, mae: 2.4315, huber: 2.0143, swd: 8.0465, target_std: 18.3689
      Epoch 5 composite train-obj: 2.137156
            No improvement (2.5440), counter 2/5
    Epoch [6/50], Train Losses: mse: 23.7919, mae: 2.5433, huber: 2.1350, swd: 13.9302, target_std: 20.3411
    Epoch [6/50], Val Losses: mse: 24.0781, mae: 2.9560, huber: 2.5384, swd: 12.8815, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 15.7949, mae: 2.4329, huber: 2.0149, swd: 7.9744, target_std: 18.3689
      Epoch 6 composite train-obj: 2.135016
            Val objective improved 2.5412 → 2.5384, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 23.7251, mae: 2.5414, huber: 2.1332, swd: 13.8744, target_std: 20.3416
    Epoch [7/50], Val Losses: mse: 24.0666, mae: 2.9592, huber: 2.5414, swd: 12.8670, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 15.7908, mae: 2.4301, huber: 2.0128, swd: 7.9333, target_std: 18.3689
      Epoch 7 composite train-obj: 2.133168
            No improvement (2.5414), counter 1/5
    Epoch [8/50], Train Losses: mse: 23.6860, mae: 2.5397, huber: 2.1314, swd: 13.8364, target_std: 20.3414
    Epoch [8/50], Val Losses: mse: 24.0746, mae: 2.9578, huber: 2.5403, swd: 12.8331, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 15.8174, mae: 2.4310, huber: 2.0134, swd: 7.9622, target_std: 18.3689
      Epoch 8 composite train-obj: 2.131400
            No improvement (2.5403), counter 2/5
    Epoch [9/50], Train Losses: mse: 23.6383, mae: 2.5381, huber: 2.1299, swd: 13.7923, target_std: 20.3404
    Epoch [9/50], Val Losses: mse: 24.1571, mae: 2.9652, huber: 2.5475, swd: 12.8803, target_std: 20.5279
    Epoch [9/50], Test Losses: mse: 15.8178, mae: 2.4261, huber: 2.0089, swd: 7.9604, target_std: 18.3689
      Epoch 9 composite train-obj: 2.129918
            No improvement (2.5475), counter 3/5
    Epoch [10/50], Train Losses: mse: 23.6276, mae: 2.5380, huber: 2.1298, swd: 13.7849, target_std: 20.3415
    Epoch [10/50], Val Losses: mse: 24.0056, mae: 2.9582, huber: 2.5405, swd: 12.8541, target_std: 20.5279
    Epoch [10/50], Test Losses: mse: 15.6156, mae: 2.4225, huber: 2.0045, swd: 7.7949, target_std: 18.3689
      Epoch 10 composite train-obj: 2.129787
            No improvement (2.5405), counter 4/5
    Epoch [11/50], Train Losses: mse: 23.5780, mae: 2.5355, huber: 2.1273, swd: 13.7358, target_std: 20.3409
    Epoch [11/50], Val Losses: mse: 24.0992, mae: 2.9627, huber: 2.5452, swd: 12.7692, target_std: 20.5279
    Epoch [11/50], Test Losses: mse: 15.8482, mae: 2.4303, huber: 2.0126, swd: 7.9246, target_std: 18.3689
      Epoch 11 composite train-obj: 2.127316
    Epoch [11/50], Test Losses: mse: 15.7949, mae: 2.4329, huber: 2.0149, swd: 7.9744, target_std: 18.3689
    Best round's Test MSE: 15.7949, MAE: 2.4329, SWD: 7.9744
    Best round's Validation MSE: 24.0781, MAE: 2.9560
    Best round's Test verification MSE : 15.7949, MAE: 2.4329, SWD: 7.9744
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 41.9838, mae: 3.1309, huber: 2.7161, swd: 16.2757, target_std: 20.3415
    Epoch [1/50], Val Losses: mse: 24.6335, mae: 2.9835, huber: 2.5729, swd: 12.1762, target_std: 20.5279
    Epoch [1/50], Test Losses: mse: 16.1279, mae: 2.4676, huber: 2.0484, swd: 7.6144, target_std: 18.3689
      Epoch 1 composite train-obj: 2.716130
            Val objective improved inf → 2.5729, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 24.2512, mae: 2.5638, huber: 2.1566, swd: 13.2535, target_std: 20.3425
    Epoch [2/50], Val Losses: mse: 24.3082, mae: 2.9576, huber: 2.5452, swd: 12.0437, target_std: 20.5279
    Epoch [2/50], Test Losses: mse: 15.9424, mae: 2.4407, huber: 2.0231, swd: 7.4720, target_std: 18.3689
      Epoch 2 composite train-obj: 2.156568
            Val objective improved 2.5729 → 2.5452, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 24.0193, mae: 2.5509, huber: 2.1433, swd: 13.0732, target_std: 20.3429
    Epoch [3/50], Val Losses: mse: 24.2000, mae: 2.9583, huber: 2.5436, swd: 11.9892, target_std: 20.5279
    Epoch [3/50], Test Losses: mse: 15.8314, mae: 2.4336, huber: 2.0161, swd: 7.3815, target_std: 18.3689
      Epoch 3 composite train-obj: 2.143326
            Val objective improved 2.5452 → 2.5436, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 23.8992, mae: 2.5461, huber: 2.1383, swd: 12.9741, target_std: 20.3413
    Epoch [4/50], Val Losses: mse: 24.2892, mae: 2.9668, huber: 2.5506, swd: 12.0310, target_std: 20.5279
    Epoch [4/50], Test Losses: mse: 15.8975, mae: 2.4373, huber: 2.0194, swd: 7.3979, target_std: 18.3689
      Epoch 4 composite train-obj: 2.138259
            No improvement (2.5506), counter 1/5
    Epoch [5/50], Train Losses: mse: 23.8310, mae: 2.5440, huber: 2.1361, swd: 12.9096, target_std: 20.3414
    Epoch [5/50], Val Losses: mse: 24.1205, mae: 2.9544, huber: 2.5379, swd: 11.8837, target_std: 20.5279
    Epoch [5/50], Test Losses: mse: 15.7863, mae: 2.4282, huber: 2.0107, swd: 7.3314, target_std: 18.3689
      Epoch 5 composite train-obj: 2.136117
            Val objective improved 2.5436 → 2.5379, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 23.7818, mae: 2.5430, huber: 2.1348, swd: 12.8661, target_std: 20.3424
    Epoch [6/50], Val Losses: mse: 24.2599, mae: 2.9674, huber: 2.5500, swd: 11.8887, target_std: 20.5279
    Epoch [6/50], Test Losses: mse: 15.9596, mae: 2.4385, huber: 2.0205, swd: 7.4071, target_std: 18.3689
      Epoch 6 composite train-obj: 2.134777
            No improvement (2.5500), counter 1/5
    Epoch [7/50], Train Losses: mse: 23.7384, mae: 2.5415, huber: 2.1332, swd: 12.8311, target_std: 20.3420
    Epoch [7/50], Val Losses: mse: 24.0901, mae: 2.9564, huber: 2.5389, swd: 11.7855, target_std: 20.5279
    Epoch [7/50], Test Losses: mse: 15.8524, mae: 2.4372, huber: 2.0187, swd: 7.3282, target_std: 18.3689
      Epoch 7 composite train-obj: 2.133222
            No improvement (2.5389), counter 2/5
    Epoch [8/50], Train Losses: mse: 23.7144, mae: 2.5401, huber: 2.1319, swd: 12.8071, target_std: 20.3411
    Epoch [8/50], Val Losses: mse: 23.9261, mae: 2.9476, huber: 2.5301, swd: 11.7317, target_std: 20.5279
    Epoch [8/50], Test Losses: mse: 15.6758, mae: 2.4278, huber: 2.0098, swd: 7.2298, target_std: 18.3689
      Epoch 8 composite train-obj: 2.131890
            Val objective improved 2.5379 → 2.5301, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 23.6486, mae: 2.5385, huber: 2.1303, swd: 12.7497, target_std: 20.3406
    Epoch [9/50], Val Losses: mse: 23.9927, mae: 2.9539, huber: 2.5364, swd: 11.7040, target_std: 20.5279
    Epoch [9/50], Test Losses: mse: 15.6840, mae: 2.4280, huber: 2.0098, swd: 7.2264, target_std: 18.3689
      Epoch 9 composite train-obj: 2.130293
            No improvement (2.5364), counter 1/5
    Epoch [10/50], Train Losses: mse: 23.6199, mae: 2.5391, huber: 2.1306, swd: 12.7245, target_std: 20.3416
    Epoch [10/50], Val Losses: mse: 24.0040, mae: 2.9568, huber: 2.5391, swd: 11.6954, target_std: 20.5279
    Epoch [10/50], Test Losses: mse: 15.7556, mae: 2.4263, huber: 2.0093, swd: 7.2485, target_std: 18.3689
      Epoch 10 composite train-obj: 2.130597
            No improvement (2.5391), counter 2/5
    Epoch [11/50], Train Losses: mse: 23.5768, mae: 2.5359, huber: 2.1276, swd: 12.6932, target_std: 20.3408
    Epoch [11/50], Val Losses: mse: 23.9178, mae: 2.9498, huber: 2.5324, swd: 11.6867, target_std: 20.5279
    Epoch [11/50], Test Losses: mse: 15.6359, mae: 2.4236, huber: 2.0057, swd: 7.1747, target_std: 18.3689
      Epoch 11 composite train-obj: 2.127550
            No improvement (2.5324), counter 3/5
    Epoch [12/50], Train Losses: mse: 23.5490, mae: 2.5359, huber: 2.1277, swd: 12.6658, target_std: 20.3408
    Epoch [12/50], Val Losses: mse: 24.1139, mae: 2.9658, huber: 2.5479, swd: 11.6743, target_std: 20.5279
    Epoch [12/50], Test Losses: mse: 15.9465, mae: 2.4369, huber: 2.0198, swd: 7.3234, target_std: 18.3689
      Epoch 12 composite train-obj: 2.127662
            No improvement (2.5479), counter 4/5
    Epoch [13/50], Train Losses: mse: 23.5284, mae: 2.5357, huber: 2.1273, swd: 12.6487, target_std: 20.3411
    Epoch [13/50], Val Losses: mse: 23.8663, mae: 2.9524, huber: 2.5345, swd: 11.5959, target_std: 20.5279
    Epoch [13/50], Test Losses: mse: 15.6399, mae: 2.4218, huber: 2.0043, swd: 7.1378, target_std: 18.3689
      Epoch 13 composite train-obj: 2.127326
    Epoch [13/50], Test Losses: mse: 15.6758, mae: 2.4278, huber: 2.0098, swd: 7.2298, target_std: 18.3689
    Best round's Test MSE: 15.6758, MAE: 2.4278, SWD: 7.2298
    Best round's Validation MSE: 23.9261, MAE: 2.9476
    Best round's Test verification MSE : 15.6758, MAE: 2.4278, SWD: 7.2298
    
    ==================================================
    Experiment Summary (DLinear_ettm2_seq96_pred336_20250430_1923)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 15.7283 ± 0.0496
      mae: 2.4288 ± 0.0030
      huber: 2.0111 ± 0.0028
      swd: 7.5717 ± 0.3070
      target_std: 18.3689 ± 0.0000
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 23.9979 ± 0.0623
      mae: 2.9520 ± 0.0034
      huber: 2.5345 ± 0.0034
      swd: 12.2928 ± 0.4698
      target_std: 20.5279 ± 0.0000
      count: 52.0000 ± 0.0000
    ==================================================
    
    Experiment complete: DLinear_ettm2_seq96_pred336_20250430_1923
    Model: DLinear
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('ettm2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 375
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: ettm2
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 375
    Validation Batches: 49
    Test Batches: 103
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 51.6802, mae: 3.5814, huber: 3.1590, swd: 21.1751, target_std: 20.3586
    Epoch [1/50], Val Losses: mse: 29.8575, mae: 3.4003, huber: 2.9793, swd: 14.4451, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.7199, mae: 2.8375, huber: 2.4116, swd: 10.1284, target_std: 18.3425
      Epoch 1 composite train-obj: 3.159014
            Val objective improved inf → 2.9793, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 32.6827, mae: 3.0284, huber: 2.6115, swd: 17.7751, target_std: 20.3589
    Epoch [2/50], Val Losses: mse: 29.6665, mae: 3.3904, huber: 2.9657, swd: 14.3872, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 20.4716, mae: 2.8142, huber: 2.3885, swd: 9.9086, target_std: 18.3425
      Epoch 2 composite train-obj: 2.611489
            Val objective improved 2.9793 → 2.9657, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 32.4602, mae: 3.0171, huber: 2.5995, swd: 17.5882, target_std: 20.3591
    Epoch [3/50], Val Losses: mse: 29.4001, mae: 3.3768, huber: 2.9492, swd: 14.1273, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.4646, mae: 2.8110, huber: 2.3857, swd: 9.8953, target_std: 18.3425
      Epoch 3 composite train-obj: 2.599503
            Val objective improved 2.9657 → 2.9492, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 32.3414, mae: 3.0138, huber: 2.5957, swd: 17.4804, target_std: 20.3592
    Epoch [4/50], Val Losses: mse: 29.6982, mae: 3.4054, huber: 2.9759, swd: 14.1754, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 20.5887, mae: 2.8035, huber: 2.3790, swd: 9.8798, target_std: 18.3425
      Epoch 4 composite train-obj: 2.595705
            No improvement (2.9759), counter 1/5
    Epoch [5/50], Train Losses: mse: 32.2437, mae: 3.0101, huber: 2.5919, swd: 17.3873, target_std: 20.3584
    Epoch [5/50], Val Losses: mse: 29.4291, mae: 3.3899, huber: 2.9594, swd: 14.0458, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 20.3219, mae: 2.7921, huber: 2.3671, swd: 9.7225, target_std: 18.3425
      Epoch 5 composite train-obj: 2.591904
            No improvement (2.9594), counter 2/5
    Epoch [6/50], Train Losses: mse: 32.1814, mae: 3.0079, huber: 2.5896, swd: 17.3179, target_std: 20.3597
    Epoch [6/50], Val Losses: mse: 29.4046, mae: 3.3896, huber: 2.9591, swd: 13.9654, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 20.3258, mae: 2.7898, huber: 2.3655, swd: 9.6713, target_std: 18.3425
      Epoch 6 composite train-obj: 2.589609
            No improvement (2.9591), counter 3/5
    Epoch [7/50], Train Losses: mse: 32.1150, mae: 3.0070, huber: 2.5885, swd: 17.2504, target_std: 20.3596
    Epoch [7/50], Val Losses: mse: 29.1493, mae: 3.3754, huber: 2.9443, swd: 13.8733, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 20.1068, mae: 2.8068, huber: 2.3793, swd: 9.6673, target_std: 18.3425
      Epoch 7 composite train-obj: 2.588493
            Val objective improved 2.9492 → 2.9443, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 32.0392, mae: 3.0045, huber: 2.5860, swd: 17.1826, target_std: 20.3591
    Epoch [8/50], Val Losses: mse: 29.3114, mae: 3.3878, huber: 2.9571, swd: 13.9169, target_std: 20.4999
    Epoch [8/50], Test Losses: mse: 19.9468, mae: 2.7790, huber: 2.3544, swd: 9.4895, target_std: 18.3425
      Epoch 8 composite train-obj: 2.586022
            No improvement (2.9571), counter 1/5
    Epoch [9/50], Train Losses: mse: 31.9596, mae: 3.0024, huber: 2.5838, swd: 17.1179, target_std: 20.3584
    Epoch [9/50], Val Losses: mse: 29.4620, mae: 3.4062, huber: 2.9748, swd: 13.9867, target_std: 20.4999
    Epoch [9/50], Test Losses: mse: 20.0833, mae: 2.7890, huber: 2.3631, swd: 9.4528, target_std: 18.3425
      Epoch 9 composite train-obj: 2.583794
            No improvement (2.9748), counter 2/5
    Epoch [10/50], Train Losses: mse: 31.9023, mae: 3.0009, huber: 2.5823, swd: 17.0527, target_std: 20.3592
    Epoch [10/50], Val Losses: mse: 29.2211, mae: 3.3859, huber: 2.9550, swd: 13.7515, target_std: 20.4999
    Epoch [10/50], Test Losses: mse: 20.0482, mae: 2.7814, huber: 2.3559, swd: 9.4622, target_std: 18.3425
      Epoch 10 composite train-obj: 2.582274
            No improvement (2.9550), counter 3/5
    Epoch [11/50], Train Losses: mse: 31.8747, mae: 3.0010, huber: 2.5824, swd: 17.0310, target_std: 20.3590
    Epoch [11/50], Val Losses: mse: 29.3058, mae: 3.3942, huber: 2.9630, swd: 13.7620, target_std: 20.4999
    Epoch [11/50], Test Losses: mse: 20.1449, mae: 2.7799, huber: 2.3553, swd: 9.4600, target_std: 18.3425
      Epoch 11 composite train-obj: 2.582419
            No improvement (2.9630), counter 4/5
    Epoch [12/50], Train Losses: mse: 31.8125, mae: 2.9969, huber: 2.5785, swd: 16.9651, target_std: 20.3582
    Epoch [12/50], Val Losses: mse: 29.2918, mae: 3.3936, huber: 2.9625, swd: 13.7023, target_std: 20.4999
    Epoch [12/50], Test Losses: mse: 20.2371, mae: 2.7854, huber: 2.3609, swd: 9.5812, target_std: 18.3425
      Epoch 12 composite train-obj: 2.578458
    Epoch [12/50], Test Losses: mse: 20.1068, mae: 2.8068, huber: 2.3793, swd: 9.6673, target_std: 18.3425
    Best round's Test MSE: 20.1068, MAE: 2.8068, SWD: 9.6673
    Best round's Validation MSE: 29.1493, MAE: 3.3754
    Best round's Test verification MSE : 20.1068, MAE: 2.8068, SWD: 9.6673
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 50.8098, mae: 3.5689, huber: 3.1464, swd: 19.1910, target_std: 20.3589
    Epoch [1/50], Val Losses: mse: 29.9531, mae: 3.4086, huber: 2.9873, swd: 13.2168, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.7929, mae: 2.8482, huber: 2.4207, swd: 9.4397, target_std: 18.3425
      Epoch 1 composite train-obj: 3.146381
            Val objective improved inf → 2.9873, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 32.6763, mae: 3.0295, huber: 2.6124, swd: 16.4123, target_std: 20.3588
    Epoch [2/50], Val Losses: mse: 29.4808, mae: 3.3800, huber: 2.9550, swd: 13.0002, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 20.5579, mae: 2.8250, huber: 2.3990, swd: 9.3446, target_std: 18.3425
      Epoch 2 composite train-obj: 2.612351
            Val objective improved 2.9873 → 2.9550, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 32.4517, mae: 3.0159, huber: 2.5984, swd: 16.2418, target_std: 20.3584
    Epoch [3/50], Val Losses: mse: 29.5103, mae: 3.3869, huber: 2.9593, swd: 12.9875, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.4434, mae: 2.8114, huber: 2.3858, swd: 9.1833, target_std: 18.3425
      Epoch 3 composite train-obj: 2.598376
            No improvement (2.9593), counter 1/5
    Epoch [4/50], Train Losses: mse: 32.3601, mae: 3.0143, huber: 2.5963, swd: 16.1526, target_std: 20.3594
    Epoch [4/50], Val Losses: mse: 29.7285, mae: 3.4081, huber: 2.9785, swd: 12.9729, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 20.5061, mae: 2.7968, huber: 2.3725, swd: 9.1341, target_std: 18.3425
      Epoch 4 composite train-obj: 2.596319
            No improvement (2.9785), counter 2/5
    Epoch [5/50], Train Losses: mse: 32.2599, mae: 3.0119, huber: 2.5936, swd: 16.0616, target_std: 20.3583
    Epoch [5/50], Val Losses: mse: 29.3995, mae: 3.3898, huber: 2.9594, swd: 12.8327, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 20.1625, mae: 2.7856, huber: 2.3613, swd: 8.9873, target_std: 18.3425
      Epoch 5 composite train-obj: 2.593562
            No improvement (2.9594), counter 3/5
    Epoch [6/50], Train Losses: mse: 32.1813, mae: 3.0085, huber: 2.5901, swd: 15.9941, target_std: 20.3589
    Epoch [6/50], Val Losses: mse: 29.7414, mae: 3.4106, huber: 2.9800, swd: 12.8452, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 20.5460, mae: 2.7991, huber: 2.3748, swd: 9.1556, target_std: 18.3425
      Epoch 6 composite train-obj: 2.590144
            No improvement (2.9800), counter 4/5
    Epoch [7/50], Train Losses: mse: 32.1061, mae: 3.0056, huber: 2.5872, swd: 15.9294, target_std: 20.3585
    Epoch [7/50], Val Losses: mse: 29.2033, mae: 3.3772, huber: 2.9463, swd: 12.6290, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 20.1860, mae: 2.7966, huber: 2.3702, swd: 8.9770, target_std: 18.3425
      Epoch 7 composite train-obj: 2.587227
            Val objective improved 2.9550 → 2.9463, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 32.0494, mae: 3.0046, huber: 2.5862, swd: 15.8753, target_std: 20.3599
    Epoch [8/50], Val Losses: mse: 29.5901, mae: 3.4104, huber: 2.9793, swd: 12.8257, target_std: 20.4999
    Epoch [8/50], Test Losses: mse: 20.1351, mae: 2.7848, huber: 2.3593, swd: 8.8634, target_std: 18.3425
      Epoch 8 composite train-obj: 2.586173
            No improvement (2.9793), counter 1/5
    Epoch [9/50], Train Losses: mse: 31.9828, mae: 3.0038, huber: 2.5853, swd: 15.8167, target_std: 20.3596
    Epoch [9/50], Val Losses: mse: 29.4822, mae: 3.4023, huber: 2.9712, swd: 12.6550, target_std: 20.4999
    Epoch [9/50], Test Losses: mse: 20.3634, mae: 2.7915, huber: 2.3665, swd: 9.0245, target_std: 18.3425
      Epoch 9 composite train-obj: 2.585268
            No improvement (2.9712), counter 2/5
    Epoch [10/50], Train Losses: mse: 31.9200, mae: 2.9995, huber: 2.5810, swd: 15.7699, target_std: 20.3597
    Epoch [10/50], Val Losses: mse: 29.1738, mae: 3.3836, huber: 2.9525, swd: 12.4881, target_std: 20.4999
    Epoch [10/50], Test Losses: mse: 20.0662, mae: 2.7811, huber: 2.3561, swd: 8.8717, target_std: 18.3425
      Epoch 10 composite train-obj: 2.581035
            No improvement (2.9525), counter 3/5
    Epoch [11/50], Train Losses: mse: 31.8627, mae: 2.9998, huber: 2.5812, swd: 15.7189, target_std: 20.3584
    Epoch [11/50], Val Losses: mse: 29.2274, mae: 3.3878, huber: 2.9569, swd: 12.4817, target_std: 20.4999
    Epoch [11/50], Test Losses: mse: 20.1342, mae: 2.7788, huber: 2.3543, swd: 8.8472, target_std: 18.3425
      Epoch 11 composite train-obj: 2.581211
            No improvement (2.9569), counter 4/5
    Epoch [12/50], Train Losses: mse: 31.8230, mae: 2.9987, huber: 2.5800, swd: 15.6843, target_std: 20.3591
    Epoch [12/50], Val Losses: mse: 29.5234, mae: 3.4134, huber: 2.9823, swd: 12.6378, target_std: 20.4999
    Epoch [12/50], Test Losses: mse: 20.0848, mae: 2.7727, huber: 2.3488, swd: 8.7850, target_std: 18.3425
      Epoch 12 composite train-obj: 2.580044
    Epoch [12/50], Test Losses: mse: 20.1860, mae: 2.7966, huber: 2.3702, swd: 8.9770, target_std: 18.3425
    Best round's Test MSE: 20.1860, MAE: 2.7966, SWD: 8.9770
    Best round's Validation MSE: 29.2033, MAE: 3.3772
    Best round's Test verification MSE : 20.1860, MAE: 2.7966, SWD: 8.9770
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 50.9285, mae: 3.5704, huber: 3.1480, swd: 21.6211, target_std: 20.3586
    Epoch [1/50], Val Losses: mse: 29.8084, mae: 3.3957, huber: 2.9747, swd: 14.8357, target_std: 20.4999
    Epoch [1/50], Test Losses: mse: 20.7262, mae: 2.8469, huber: 2.4195, swd: 10.6876, target_std: 18.3425
      Epoch 1 composite train-obj: 3.148017
            Val objective improved inf → 2.9747, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 32.6831, mae: 3.0291, huber: 2.6120, swd: 18.5929, target_std: 20.3590
    Epoch [2/50], Val Losses: mse: 29.6287, mae: 3.3894, huber: 2.9642, swd: 14.6201, target_std: 20.4999
    Epoch [2/50], Test Losses: mse: 20.6431, mae: 2.8226, huber: 2.3961, swd: 10.5531, target_std: 18.3425
      Epoch 2 composite train-obj: 2.612042
            Val objective improved 2.9747 → 2.9642, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 32.4652, mae: 3.0167, huber: 2.5992, swd: 18.4137, target_std: 20.3595
    Epoch [3/50], Val Losses: mse: 29.4381, mae: 3.3794, huber: 2.9520, swd: 14.4725, target_std: 20.4999
    Epoch [3/50], Test Losses: mse: 20.5060, mae: 2.8091, huber: 2.3841, swd: 10.4552, target_std: 18.3425
      Epoch 3 composite train-obj: 2.599202
            Val objective improved 2.9642 → 2.9520, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 32.3458, mae: 3.0127, huber: 2.5948, swd: 18.2983, target_std: 20.3589
    Epoch [4/50], Val Losses: mse: 29.4825, mae: 3.3899, huber: 2.9603, swd: 14.4237, target_std: 20.4999
    Epoch [4/50], Test Losses: mse: 20.3749, mae: 2.7944, huber: 2.3700, swd: 10.3364, target_std: 18.3425
      Epoch 4 composite train-obj: 2.594822
            No improvement (2.9603), counter 1/5
    Epoch [5/50], Train Losses: mse: 32.2701, mae: 3.0126, huber: 2.5942, swd: 18.2100, target_std: 20.3590
    Epoch [5/50], Val Losses: mse: 29.4572, mae: 3.3945, huber: 2.9642, swd: 14.3693, target_std: 20.4999
    Epoch [5/50], Test Losses: mse: 20.2951, mae: 2.7889, huber: 2.3652, swd: 10.2407, target_std: 18.3425
      Epoch 5 composite train-obj: 2.594239
            No improvement (2.9642), counter 2/5
    Epoch [6/50], Train Losses: mse: 32.1661, mae: 3.0068, huber: 2.5885, swd: 18.1157, target_std: 20.3585
    Epoch [6/50], Val Losses: mse: 29.1821, mae: 3.3738, huber: 2.9433, swd: 14.1659, target_std: 20.4999
    Epoch [6/50], Test Losses: mse: 20.3758, mae: 2.8004, huber: 2.3754, swd: 10.3573, target_std: 18.3425
      Epoch 6 composite train-obj: 2.588542
            Val objective improved 2.9520 → 2.9433, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 32.0873, mae: 3.0047, huber: 2.5864, swd: 18.0370, target_std: 20.3580
    Epoch [7/50], Val Losses: mse: 29.3862, mae: 3.3961, huber: 2.9648, swd: 14.1856, target_std: 20.4999
    Epoch [7/50], Test Losses: mse: 20.3673, mae: 2.7978, huber: 2.3731, swd: 10.2258, target_std: 18.3425
      Epoch 7 composite train-obj: 2.586355
            No improvement (2.9648), counter 1/5
    Epoch [8/50], Train Losses: mse: 32.0481, mae: 3.0053, huber: 2.5868, swd: 17.9902, target_std: 20.3587
    Epoch [8/50], Val Losses: mse: 29.3510, mae: 3.3918, huber: 2.9607, swd: 14.1605, target_std: 20.4999
    Epoch [8/50], Test Losses: mse: 20.2883, mae: 2.7897, huber: 2.3651, swd: 10.1968, target_std: 18.3425
      Epoch 8 composite train-obj: 2.586778
            No improvement (2.9607), counter 2/5
    Epoch [9/50], Train Losses: mse: 31.9813, mae: 3.0035, huber: 2.5849, swd: 17.9247, target_std: 20.3586
    Epoch [9/50], Val Losses: mse: 29.4675, mae: 3.4042, huber: 2.9729, swd: 14.2313, target_std: 20.4999
    Epoch [9/50], Test Losses: mse: 20.1228, mae: 2.7760, huber: 2.3525, swd: 10.0063, target_std: 18.3425
      Epoch 9 composite train-obj: 2.584921
            No improvement (2.9729), counter 3/5
    Epoch [10/50], Train Losses: mse: 31.9167, mae: 3.0004, huber: 2.5819, swd: 17.8603, target_std: 20.3582
    Epoch [10/50], Val Losses: mse: 29.2649, mae: 3.3891, huber: 2.9581, swd: 14.0893, target_std: 20.4999
    Epoch [10/50], Test Losses: mse: 20.0424, mae: 2.7765, huber: 2.3519, swd: 9.9788, target_std: 18.3425
      Epoch 10 composite train-obj: 2.581876
            No improvement (2.9581), counter 4/5
    Epoch [11/50], Train Losses: mse: 31.8638, mae: 2.9988, huber: 2.5802, swd: 17.8119, target_std: 20.3594
    Epoch [11/50], Val Losses: mse: 29.3922, mae: 3.4018, huber: 2.9706, swd: 14.0986, target_std: 20.4999
    Epoch [11/50], Test Losses: mse: 20.2248, mae: 2.7863, huber: 2.3619, swd: 10.0566, target_std: 18.3425
      Epoch 11 composite train-obj: 2.580225
    Epoch [11/50], Test Losses: mse: 20.3758, mae: 2.8004, huber: 2.3754, swd: 10.3573, target_std: 18.3425
    Best round's Test MSE: 20.3758, MAE: 2.8004, SWD: 10.3573
    Best round's Validation MSE: 29.1821, MAE: 3.3738
    Best round's Test verification MSE : 20.3758, MAE: 2.8004, SWD: 10.3573
    
    ==================================================
    Experiment Summary (DLinear_ettm2_seq96_pred720_20250430_1925)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 20.2229 ± 0.1129
      mae: 2.8013 ± 0.0042
      huber: 2.3750 ± 0.0038
      swd: 9.6672 ± 0.5635
      target_std: 18.3425 ± 0.0000
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 29.1782 ± 0.0222
      mae: 3.3755 ± 0.0014
      huber: 2.9446 ± 0.0012
      swd: 13.5561 ± 0.6663
      target_std: 20.4999 ± 0.0000
      count: 49.0000 ± 0.0000
    ==================================================
    
    Experiment complete: DLinear_ettm2_seq96_pred720_20250430_1925
    Model: DLinear
    Dataset: ettm2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


