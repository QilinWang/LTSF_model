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
# Initialize the data manager
data_mgr = DatasetManager(device='cuda')

# Load a synthetic dataset
data_mgr.load_csv('ettm1', './ettm1.csv')
# SCALE = False
# trajectory = utils.generate_trajectory('lorenz',steps=52200, dt=1e-2) 
# trajectory = utils.generate_hyperchaotic_rossler(steps=12000, dt=1e-3)
# trajectory_2 = utils.generate_henon(steps=52000) 
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    
    ==================================================
    Dataset: ettm1 (csv)
    ==================================================
    Shape: torch.Size([69680, 7])
    Channels: 7
    Length: 69680
    Source: ./ettm1.csv
    
    Sample data (first 2 rows):
    tensor([[ 5.8270,  2.0090,  1.5990,  0.4620,  4.2030,  1.3400, 30.5310],
            [ 5.7600,  2.0760,  1.4920,  0.4260,  4.2640,  1.4010, 30.4600]])
    ==================================================
    




    <data_manager.DatasetManager at 0x1a305455a90>



# seq=96

### EigenACL



#### 96-96

#### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    global_std.shape: torch.Size([7])
    Global Std for ettm1: tensor([7.0829, 2.0413, 6.8291, 1.8072, 1.1741, 0.6004, 8.5648],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 7.8547, mae: 1.5181, huber: 1.1500, swd: 3.1883, ept: 61.0570
    Epoch [1/50], Val Losses: mse: 5.9294, mae: 1.2819, huber: 0.9302, swd: 1.2307, ept: 67.8175
    Epoch [1/50], Test Losses: mse: 8.4386, mae: 1.5038, huber: 1.1430, swd: 1.6213, ept: 56.4718
      Epoch 1 composite train-obj: 1.150014
            Val objective improved inf → 0.9302, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8799, mae: 1.2473, huber: 0.8939, swd: 1.4607, ept: 68.7232
    Epoch [2/50], Val Losses: mse: 5.7983, mae: 1.2671, huber: 0.9171, swd: 1.2885, ept: 68.1259
    Epoch [2/50], Test Losses: mse: 8.2434, mae: 1.4890, huber: 1.1292, swd: 1.7221, ept: 57.4250
      Epoch 2 composite train-obj: 0.893868
            Val objective improved 0.9302 → 0.9171, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7433, mae: 1.2212, huber: 0.8703, swd: 1.4232, ept: 69.8795
    Epoch [3/50], Val Losses: mse: 5.7727, mae: 1.2570, huber: 0.9080, swd: 1.2196, ept: 68.9871
    Epoch [3/50], Test Losses: mse: 8.1561, mae: 1.4710, huber: 1.1130, swd: 1.6556, ept: 58.5190
      Epoch 3 composite train-obj: 0.870292
            Val objective improved 0.9171 → 0.9080, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6066, mae: 1.1941, huber: 0.8460, swd: 1.3546, ept: 70.7452
    Epoch [4/50], Val Losses: mse: 5.7280, mae: 1.2691, huber: 0.9189, swd: 1.3677, ept: 68.7950
    Epoch [4/50], Test Losses: mse: 8.0600, mae: 1.4798, huber: 1.1207, swd: 1.8770, ept: 58.4265
      Epoch 4 composite train-obj: 0.846006
            No improvement (0.9189), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5273, mae: 1.1810, huber: 0.8342, swd: 1.3323, ept: 71.4017
    Epoch [5/50], Val Losses: mse: 5.7447, mae: 1.2848, huber: 0.9340, swd: 1.4741, ept: 70.0179
    Epoch [5/50], Test Losses: mse: 8.0684, mae: 1.4935, huber: 1.1343, swd: 2.0299, ept: 59.1080
      Epoch 5 composite train-obj: 0.834240
            No improvement (0.9340), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.4494, mae: 1.1679, huber: 0.8227, swd: 1.2995, ept: 71.8607
    Epoch [6/50], Val Losses: mse: 5.8880, mae: 1.2802, huber: 0.9325, swd: 1.3881, ept: 69.2865
    Epoch [6/50], Test Losses: mse: 8.0210, mae: 1.4762, huber: 1.1189, swd: 1.9256, ept: 58.9119
      Epoch 6 composite train-obj: 0.822655
            No improvement (0.9325), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.3830, mae: 1.1577, huber: 0.8135, swd: 1.2794, ept: 72.2362
    Epoch [7/50], Val Losses: mse: 5.9438, mae: 1.3146, huber: 0.9636, swd: 1.6248, ept: 68.7051
    Epoch [7/50], Test Losses: mse: 8.1214, mae: 1.5105, huber: 1.1517, swd: 2.2071, ept: 58.4121
      Epoch 7 composite train-obj: 0.813532
            No improvement (0.9636), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3217, mae: 1.1482, huber: 0.8049, swd: 1.2629, ept: 72.5155
    Epoch [8/50], Val Losses: mse: 5.9305, mae: 1.2894, huber: 0.9399, swd: 1.3096, ept: 70.5192
    Epoch [8/50], Test Losses: mse: 8.0179, mae: 1.4623, huber: 1.1051, swd: 1.6320, ept: 59.8166
      Epoch 8 composite train-obj: 0.804919
    Epoch [8/50], Test Losses: mse: 8.1554, mae: 1.4710, huber: 1.1130, swd: 1.6556, ept: 58.5159
    Best round's Test MSE: 8.1561, MAE: 1.4710, SWD: 1.6556
    Best round's Validation MSE: 5.7727, MAE: 1.2570, SWD: 1.2196
    Best round's Test verification MSE : 8.1554, MAE: 1.4710, SWD: 1.6556
    Time taken: 106.95 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.8903, mae: 1.5186, huber: 1.1507, swd: 3.2115, ept: 60.4415
    Epoch [1/50], Val Losses: mse: 6.0114, mae: 1.3036, huber: 0.9516, swd: 1.4906, ept: 65.4278
    Epoch [1/50], Test Losses: mse: 8.4750, mae: 1.5286, huber: 1.1676, swd: 2.0162, ept: 54.9859
      Epoch 1 composite train-obj: 1.150724
            Val objective improved inf → 0.9516, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8872, mae: 1.2447, huber: 0.8920, swd: 1.4376, ept: 68.7769
    Epoch [2/50], Val Losses: mse: 5.7777, mae: 1.2655, huber: 0.9156, swd: 1.3682, ept: 68.4782
    Epoch [2/50], Test Losses: mse: 8.2747, mae: 1.4883, huber: 1.1298, swd: 1.8389, ept: 57.1239
      Epoch 2 composite train-obj: 0.891957
            Val objective improved 0.9516 → 0.9156, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7394, mae: 1.2171, huber: 0.8671, swd: 1.3962, ept: 70.1557
    Epoch [3/50], Val Losses: mse: 5.7124, mae: 1.2740, huber: 0.9231, swd: 1.4561, ept: 67.7100
    Epoch [3/50], Test Losses: mse: 8.1288, mae: 1.4963, huber: 1.1371, swd: 1.9611, ept: 57.4810
      Epoch 3 composite train-obj: 0.867099
            No improvement (0.9231), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.6354, mae: 1.1992, huber: 0.8510, swd: 1.3572, ept: 70.9118
    Epoch [4/50], Val Losses: mse: 5.7562, mae: 1.2717, huber: 0.9194, swd: 1.2558, ept: 69.4646
    Epoch [4/50], Test Losses: mse: 8.2295, mae: 1.4832, huber: 1.1228, swd: 1.6977, ept: 58.9138
      Epoch 4 composite train-obj: 0.851035
            No improvement (0.9194), counter 2/5
    Epoch [5/50], Train Losses: mse: 4.5319, mae: 1.1816, huber: 0.8352, swd: 1.3162, ept: 71.4805
    Epoch [5/50], Val Losses: mse: 5.7861, mae: 1.2870, huber: 0.9367, swd: 1.4102, ept: 69.5018
    Epoch [5/50], Test Losses: mse: 8.0929, mae: 1.4789, huber: 1.1216, swd: 1.8499, ept: 58.7763
      Epoch 5 composite train-obj: 0.835219
            No improvement (0.9367), counter 3/5
    Epoch [6/50], Train Losses: mse: 4.4613, mae: 1.1700, huber: 0.8247, swd: 1.2922, ept: 71.8815
    Epoch [6/50], Val Losses: mse: 5.8074, mae: 1.2859, huber: 0.9350, swd: 1.3145, ept: 70.4041
    Epoch [6/50], Test Losses: mse: 8.1255, mae: 1.4787, huber: 1.1209, swd: 1.7757, ept: 59.4015
      Epoch 6 composite train-obj: 0.824733
            No improvement (0.9350), counter 4/5
    Epoch [7/50], Train Losses: mse: 4.3869, mae: 1.1581, huber: 0.8138, swd: 1.2677, ept: 72.2415
    Epoch [7/50], Val Losses: mse: 5.9620, mae: 1.2856, huber: 0.9372, swd: 1.3560, ept: 70.7937
    Epoch [7/50], Test Losses: mse: 8.1965, mae: 1.4838, huber: 1.1267, swd: 1.8471, ept: 59.2727
      Epoch 7 composite train-obj: 0.813848
    Epoch [7/50], Test Losses: mse: 8.2740, mae: 1.4883, huber: 1.1297, swd: 1.8384, ept: 57.1238
    Best round's Test MSE: 8.2747, MAE: 1.4883, SWD: 1.8389
    Best round's Validation MSE: 5.7777, MAE: 1.2655, SWD: 1.3682
    Best round's Test verification MSE : 8.2740, MAE: 1.4883, SWD: 1.8384
    Time taken: 101.75 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.1567, mae: 1.5384, huber: 1.1699, swd: 3.1085, ept: 60.2653
    Epoch [1/50], Val Losses: mse: 5.8802, mae: 1.2869, huber: 0.9351, swd: 1.3231, ept: 65.5908
    Epoch [1/50], Test Losses: mse: 8.4301, mae: 1.5128, huber: 1.1518, swd: 1.6579, ept: 54.8111
      Epoch 1 composite train-obj: 1.169915
            Val objective improved inf → 0.9351, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8485, mae: 1.2421, huber: 0.8897, swd: 1.3412, ept: 68.9951
    Epoch [2/50], Val Losses: mse: 5.6760, mae: 1.2673, huber: 0.9174, swd: 1.3549, ept: 67.7735
    Epoch [2/50], Test Losses: mse: 8.0091, mae: 1.4718, huber: 1.1134, swd: 1.6330, ept: 57.4807
      Epoch 2 composite train-obj: 0.889651
            Val objective improved 0.9351 → 0.9174, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.6859, mae: 1.2102, huber: 0.8611, swd: 1.2821, ept: 70.2584
    Epoch [3/50], Val Losses: mse: 5.7189, mae: 1.2577, huber: 0.9096, swd: 1.2575, ept: 68.5901
    Epoch [3/50], Test Losses: mse: 8.0274, mae: 1.4754, huber: 1.1179, swd: 1.6863, ept: 58.4061
      Epoch 3 composite train-obj: 0.861073
            Val objective improved 0.9174 → 0.9096, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.5794, mae: 1.1906, huber: 0.8435, swd: 1.2433, ept: 70.9273
    Epoch [4/50], Val Losses: mse: 5.8819, mae: 1.2752, huber: 0.9250, swd: 1.3905, ept: 69.4928
    Epoch [4/50], Test Losses: mse: 8.2158, mae: 1.4913, huber: 1.1328, swd: 1.8495, ept: 58.7143
      Epoch 4 composite train-obj: 0.843450
            No improvement (0.9250), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5003, mae: 1.1758, huber: 0.8302, swd: 1.2140, ept: 71.4992
    Epoch [5/50], Val Losses: mse: 5.8885, mae: 1.2651, huber: 0.9170, swd: 1.1977, ept: 69.6686
    Epoch [5/50], Test Losses: mse: 8.0568, mae: 1.4671, huber: 1.1104, swd: 1.5848, ept: 59.1329
      Epoch 5 composite train-obj: 0.830193
            No improvement (0.9170), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.4297, mae: 1.1642, huber: 0.8196, swd: 1.1870, ept: 71.8658
    Epoch [6/50], Val Losses: mse: 5.7710, mae: 1.2692, huber: 0.9211, swd: 1.3001, ept: 69.8092
    Epoch [6/50], Test Losses: mse: 7.9976, mae: 1.4756, huber: 1.1187, swd: 1.7438, ept: 59.1865
      Epoch 6 composite train-obj: 0.819636
            No improvement (0.9211), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.3964, mae: 1.1597, huber: 0.8157, swd: 1.1819, ept: 72.1975
    Epoch [7/50], Val Losses: mse: 6.1150, mae: 1.2898, huber: 0.9413, swd: 1.3368, ept: 70.1242
    Epoch [7/50], Test Losses: mse: 8.1550, mae: 1.4918, huber: 1.1349, swd: 1.8912, ept: 59.3079
      Epoch 7 composite train-obj: 0.815683
            No improvement (0.9413), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3208, mae: 1.1472, huber: 0.8044, swd: 1.1578, ept: 72.4668
    Epoch [8/50], Val Losses: mse: 6.0777, mae: 1.2854, huber: 0.9387, swd: 1.2584, ept: 70.1042
    Epoch [8/50], Test Losses: mse: 7.8841, mae: 1.4488, huber: 1.0941, swd: 1.5234, ept: 60.0283
      Epoch 8 composite train-obj: 0.804356
    Epoch [8/50], Test Losses: mse: 8.0278, mae: 1.4754, huber: 1.1179, swd: 1.6863, ept: 58.3889
    Best round's Test MSE: 8.0274, MAE: 1.4754, SWD: 1.6863
    Best round's Validation MSE: 5.7189, MAE: 1.2577, SWD: 1.2575
    Best round's Test verification MSE : 8.0278, MAE: 1.4754, SWD: 1.6863
    Time taken: 122.72 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred96_20250512_0203)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 8.1527 ± 0.1010
      mae: 1.4783 ± 0.0073
      huber: 1.1202 ± 0.0071
      swd: 1.7269 ± 0.0802
      ept: 58.0163 ± 0.6327
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.7564 ± 0.0266
      mae: 1.2601 ± 0.0038
      huber: 0.9111 ± 0.0033
      swd: 1.2818 ± 0.0630
      ept: 68.6851 ± 0.2183
      count: 53.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 333.36 seconds
    
    Experiment complete: ACL_ettm1_seq96_pred96_20250512_0203
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

##### Ab:  shift inside


```python
importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
cfg.x_to_z_delay.spectral_flags_hidden = [False, False]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden = [False, False]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden = [False, False]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden = [False, False]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden = [False, False]
exp = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 7.7064, mae: 1.5129, huber: 1.1438, swd: 3.2373, target_std: 6.5114
    Epoch [1/50], Val Losses: mse: 5.9298, mae: 1.2830, huber: 0.9310, swd: 1.2214, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.4338, mae: 1.5108, huber: 1.1489, swd: 1.6224, target_std: 4.7755
      Epoch 1 composite train-obj: 1.143793
            Val objective improved inf → 0.9310, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8754, mae: 1.2456, huber: 0.8922, swd: 1.4537, target_std: 6.5111
    Epoch [2/50], Val Losses: mse: 5.7301, mae: 1.2593, huber: 0.9086, swd: 1.1510, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.1271, mae: 1.4637, huber: 1.1047, swd: 1.4731, target_std: 4.7755
      Epoch 2 composite train-obj: 0.892158
            Val objective improved 0.9310 → 0.9086, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7067, mae: 1.2130, huber: 0.8630, swd: 1.3936, target_std: 6.5114
    Epoch [3/50], Val Losses: mse: 5.9090, mae: 1.2655, huber: 0.9153, swd: 1.2521, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.2284, mae: 1.4777, huber: 1.1191, swd: 1.6827, target_std: 4.7755
      Epoch 3 composite train-obj: 0.862962
            No improvement (0.9153), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.6208, mae: 1.1985, huber: 0.8502, swd: 1.3715, target_std: 6.5109
    Epoch [4/50], Val Losses: mse: 5.5856, mae: 1.2525, huber: 0.9036, swd: 1.3015, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 7.9406, mae: 1.4682, huber: 1.1103, swd: 1.7794, target_std: 4.7755
      Epoch 4 composite train-obj: 0.850193
            Val objective improved 0.9086 → 0.9036, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.5104, mae: 1.1784, huber: 0.8323, swd: 1.3233, target_std: 6.5113
    Epoch [5/50], Val Losses: mse: 5.6771, mae: 1.2751, huber: 0.9258, swd: 1.3176, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 7.9826, mae: 1.4702, huber: 1.1134, swd: 1.7558, target_std: 4.7755
      Epoch 5 composite train-obj: 0.832297
            No improvement (0.9258), counter 1/5
    Epoch [6/50], Train Losses: mse: 4.4318, mae: 1.1659, huber: 0.8210, swd: 1.2968, target_std: 6.5112
    Epoch [6/50], Val Losses: mse: 6.0318, mae: 1.2968, huber: 0.9470, swd: 1.3526, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.1825, mae: 1.4903, huber: 1.1312, swd: 1.7976, target_std: 4.7755
      Epoch 6 composite train-obj: 0.820982
            No improvement (0.9470), counter 2/5
    Epoch [7/50], Train Losses: mse: 4.3594, mae: 1.1547, huber: 0.8108, swd: 1.2694, target_std: 6.5115
    Epoch [7/50], Val Losses: mse: 5.8822, mae: 1.2957, huber: 0.9468, swd: 1.4774, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 7.9670, mae: 1.4739, huber: 1.1175, swd: 1.8440, target_std: 4.7755
      Epoch 7 composite train-obj: 0.810814
            No improvement (0.9468), counter 3/5
    Epoch [8/50], Train Losses: mse: 4.3038, mae: 1.1477, huber: 0.8044, swd: 1.2577, target_std: 6.5118
    Epoch [8/50], Val Losses: mse: 6.3382, mae: 1.3140, huber: 0.9656, swd: 1.5522, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.1327, mae: 1.4905, huber: 1.1339, swd: 1.9359, target_std: 4.7755
      Epoch 8 composite train-obj: 0.804384
            No improvement (0.9656), counter 4/5
    Epoch [9/50], Train Losses: mse: 4.2237, mae: 1.1337, huber: 0.7918, swd: 1.2207, target_std: 6.5113
    Epoch [9/50], Val Losses: mse: 6.1785, mae: 1.3179, huber: 0.9680, swd: 1.4552, target_std: 4.2867
    Epoch [9/50], Test Losses: mse: 8.1496, mae: 1.4838, huber: 1.1268, swd: 1.7870, target_std: 4.7755
      Epoch 9 composite train-obj: 0.791786
    Epoch [9/50], Test Losses: mse: 7.9406, mae: 1.4682, huber: 1.1103, swd: 1.7794, target_std: 4.7755
    Best round's Test MSE: 7.9406, MAE: 1.4682, SWD: 1.7794
    Best round's Validation MSE: 5.5856, MAE: 1.2525
    Best round's Test verification MSE : 7.9406, MAE: 1.4682, SWD: 1.7794
    Time taken: 122.93 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.9005, mae: 1.5218, huber: 1.1533, swd: 3.3758, target_std: 6.5116
    Epoch [1/50], Val Losses: mse: 5.8931, mae: 1.2878, huber: 0.9372, swd: 1.3176, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.3450, mae: 1.5049, huber: 1.1457, swd: 1.7321, target_std: 4.7755
      Epoch 1 composite train-obj: 1.153327
            Val objective improved inf → 0.9372, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9290, mae: 1.2518, huber: 0.8987, swd: 1.4701, target_std: 6.5116
    Epoch [2/50], Val Losses: mse: 5.8038, mae: 1.2560, huber: 0.9073, swd: 1.2059, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.1926, mae: 1.4700, huber: 1.1118, swd: 1.6214, target_std: 4.7755
      Epoch 2 composite train-obj: 0.898735
            Val objective improved 0.9372 → 0.9073, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7343, mae: 1.2154, huber: 0.8656, swd: 1.3893, target_std: 6.5110
    Epoch [3/50], Val Losses: mse: 5.6074, mae: 1.2563, huber: 0.9081, swd: 1.3461, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 7.9673, mae: 1.4703, huber: 1.1136, swd: 1.8425, target_std: 4.7755
      Epoch 3 composite train-obj: 0.865585
            No improvement (0.9081), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.6252, mae: 1.1965, huber: 0.8489, swd: 1.3558, target_std: 6.5113
    Epoch [4/50], Val Losses: mse: 5.6095, mae: 1.2556, huber: 0.9065, swd: 1.3725, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.1638, mae: 1.4938, huber: 1.1358, swd: 2.0148, target_std: 4.7755
      Epoch 4 composite train-obj: 0.848861
            Val objective improved 0.9073 → 0.9065, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.5343, mae: 1.1797, huber: 0.8338, swd: 1.3169, target_std: 6.5106
    Epoch [5/50], Val Losses: mse: 5.7308, mae: 1.2652, huber: 0.9163, swd: 1.2057, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.1004, mae: 1.4677, huber: 1.1113, swd: 1.7040, target_std: 4.7755
      Epoch 5 composite train-obj: 0.833817
            No improvement (0.9163), counter 1/5
    Epoch [6/50], Train Losses: mse: 4.4561, mae: 1.1670, huber: 0.8224, swd: 1.2846, target_std: 6.5110
    Epoch [6/50], Val Losses: mse: 5.7955, mae: 1.2678, huber: 0.9199, swd: 1.3291, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.0363, mae: 1.4677, huber: 1.1113, swd: 1.8507, target_std: 4.7755
      Epoch 6 composite train-obj: 0.822397
            No improvement (0.9199), counter 2/5
    Epoch [7/50], Train Losses: mse: 4.3905, mae: 1.1570, huber: 0.8133, swd: 1.2642, target_std: 6.5108
    Epoch [7/50], Val Losses: mse: 5.8788, mae: 1.2804, huber: 0.9330, swd: 1.3167, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 7.9187, mae: 1.4543, huber: 1.0984, swd: 1.6999, target_std: 4.7755
      Epoch 7 composite train-obj: 0.813296
            No improvement (0.9330), counter 3/5
    Epoch [8/50], Train Losses: mse: 4.3217, mae: 1.1457, huber: 0.8030, swd: 1.2368, target_std: 6.5111
    Epoch [8/50], Val Losses: mse: 6.1010, mae: 1.2992, huber: 0.9513, swd: 1.5015, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.0169, mae: 1.4799, huber: 1.1231, swd: 1.9282, target_std: 4.7755
      Epoch 8 composite train-obj: 0.803013
            No improvement (0.9513), counter 4/5
    Epoch [9/50], Train Losses: mse: 4.2612, mae: 1.1369, huber: 0.7950, swd: 1.2175, target_std: 6.5112
    Epoch [9/50], Val Losses: mse: 5.9284, mae: 1.2867, huber: 0.9391, swd: 1.4839, target_std: 4.2867
    Epoch [9/50], Test Losses: mse: 7.8931, mae: 1.4665, huber: 1.1106, swd: 1.8523, target_std: 4.7755
      Epoch 9 composite train-obj: 0.795027
    Epoch [9/50], Test Losses: mse: 8.1639, mae: 1.4938, huber: 1.1358, swd: 2.0148, target_std: 4.7755
    Best round's Test MSE: 8.1638, MAE: 1.4938, SWD: 2.0148
    Best round's Validation MSE: 5.6095, MAE: 1.2556
    Best round's Test verification MSE : 8.1639, MAE: 1.4938, SWD: 2.0148
    Time taken: 122.49 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.7939, mae: 1.5193, huber: 1.1502, swd: 3.0903, target_std: 6.5109
    Epoch [1/50], Val Losses: mse: 5.9118, mae: 1.2875, huber: 0.9357, swd: 1.2420, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.4411, mae: 1.5141, huber: 1.1528, swd: 1.6221, target_std: 4.7755
      Epoch 1 composite train-obj: 1.150247
            Val objective improved inf → 0.9357, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8814, mae: 1.2464, huber: 0.8934, swd: 1.3442, target_std: 6.5109
    Epoch [2/50], Val Losses: mse: 5.6724, mae: 1.2479, huber: 0.8993, swd: 1.1567, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.1241, mae: 1.4673, huber: 1.1089, swd: 1.4926, target_std: 4.7755
      Epoch 2 composite train-obj: 0.893388
            Val objective improved 0.9357 → 0.8993, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7065, mae: 1.2118, huber: 0.8624, swd: 1.2849, target_std: 6.5115
    Epoch [3/50], Val Losses: mse: 5.6884, mae: 1.2485, huber: 0.9008, swd: 1.2184, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.0975, mae: 1.4704, huber: 1.1132, swd: 1.6416, target_std: 4.7755
      Epoch 3 composite train-obj: 0.862395
            No improvement (0.9008), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.6137, mae: 1.1940, huber: 0.8466, swd: 1.2531, target_std: 6.5111
    Epoch [4/50], Val Losses: mse: 5.7809, mae: 1.2644, huber: 0.9146, swd: 1.1333, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.1502, mae: 1.4666, huber: 1.1086, swd: 1.4897, target_std: 4.7755
      Epoch 4 composite train-obj: 0.846613
            No improvement (0.9146), counter 2/5
    Epoch [5/50], Train Losses: mse: 4.5277, mae: 1.1783, huber: 0.8326, swd: 1.2154, target_std: 6.5109
    Epoch [5/50], Val Losses: mse: 5.9651, mae: 1.2800, huber: 0.9329, swd: 1.3377, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.0510, mae: 1.4682, huber: 1.1118, swd: 1.6866, target_std: 4.7755
      Epoch 5 composite train-obj: 0.832563
            No improvement (0.9329), counter 3/5
    Epoch [6/50], Train Losses: mse: 4.4568, mae: 1.1664, huber: 0.8220, swd: 1.1956, target_std: 6.5117
    Epoch [6/50], Val Losses: mse: 5.9702, mae: 1.2855, huber: 0.9372, swd: 1.3591, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.0825, mae: 1.4706, huber: 1.1137, swd: 1.6189, target_std: 4.7755
      Epoch 6 composite train-obj: 0.822039
            No improvement (0.9372), counter 4/5
    Epoch [7/50], Train Losses: mse: 4.4026, mae: 1.1570, huber: 0.8135, swd: 1.1752, target_std: 6.5115
    Epoch [7/50], Val Losses: mse: 6.1743, mae: 1.3078, huber: 0.9589, swd: 1.3978, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.1526, mae: 1.4788, huber: 1.1223, swd: 1.6135, target_std: 4.7755
      Epoch 7 composite train-obj: 0.813513
    Epoch [7/50], Test Losses: mse: 8.1241, mae: 1.4673, huber: 1.1089, swd: 1.4926, target_std: 4.7755
    Best round's Test MSE: 8.1241, MAE: 1.4673, SWD: 1.4926
    Best round's Validation MSE: 5.6724, MAE: 1.2479
    Best round's Test verification MSE : 8.1241, MAE: 1.4673, SWD: 1.4926
    Time taken: 100.87 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred96_20250503_1446)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 8.0762 ± 0.0972
      mae: 1.4764 ± 0.0123
      huber: 1.1183 ± 0.0124
      swd: 1.7623 ± 0.2135
      target_std: 4.7755 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.6225 ± 0.0366
      mae: 1.2520 ± 0.0031
      huber: 0.9031 ± 0.0030
      swd: 1.2769 ± 0.0898
      target_std: 4.2867 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 346.43 seconds
    
    Experiment complete: ACL_ettm1_seq96_pred96_20250503_1446
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

Note: outside is consistently better on ETTm1, so we choose to use it as main model.
The post A() adjustment is doing worse in Lorenz because A() itself was doing exellent.
This has something do to with the geometry of Lorenz: stable, predictable.


```python
importlib.reload(monotonic)
importlib.reload(train_config)
# utils.reload_modules([modules_to_reload_list])

cfg_ACL_96_96 = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
    batch_size=128,
    learning_rate=9e-4, 
    seeds=[1955, 7, 20],  
    epochs=50, 
    dim_hidden=96,
    dim_augment=96,
    ablate_no_koopman=False,
    use_complex_eigenvalues=True,
    second_delay_use_shift=True,
    ablate_rotate_back_Koopman=True, 
    ablate_shift_inside_scale=False,
)
cfg_ACL_96_96.x_to_z_delay.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_96.x_to_z_deri.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_96.z_to_x_main.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_96.z_to_y_main.use_magnitude_for_scale_and_translate = [False, True]
exp_ACL_96_96_ = execute_model_evaluation('ettm1', cfg_ACL_96_96, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 8.2080, mae: 1.5513, huber: 1.1810, swd: 3.6423, target_std: 6.5108
    Epoch [1/50], Val Losses: mse: 6.0440, mae: 1.3018, huber: 0.9502, swd: 1.3607, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.4867, mae: 1.5163, huber: 1.1568, swd: 1.7680, target_std: 4.7755
      Epoch 1 composite train-obj: 1.180971
            Val objective improved inf → 0.9502, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9376, mae: 1.2527, huber: 0.8993, swd: 1.4821, target_std: 6.5112
    Epoch [2/50], Val Losses: mse: 5.7842, mae: 1.2692, huber: 0.9188, swd: 1.3347, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.1627, mae: 1.4745, huber: 1.1158, swd: 1.6610, target_std: 4.7755
      Epoch 2 composite train-obj: 0.899341
            Val objective improved 0.9502 → 0.9188, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7771, mae: 1.2212, huber: 0.8709, swd: 1.4156, target_std: 6.5107
    Epoch [3/50], Val Losses: mse: 5.8067, mae: 1.2547, huber: 0.9055, swd: 1.1851, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.2034, mae: 1.4710, huber: 1.1126, swd: 1.5422, target_std: 4.7755
      Epoch 3 composite train-obj: 0.870926
            Val objective improved 0.9188 → 0.9055, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6687, mae: 1.2028, huber: 0.8544, swd: 1.3796, target_std: 6.5112
    Epoch [4/50], Val Losses: mse: 5.8440, mae: 1.2695, huber: 0.9211, swd: 1.3360, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.1647, mae: 1.4771, huber: 1.1202, swd: 1.7446, target_std: 4.7755
      Epoch 4 composite train-obj: 0.854399
            No improvement (0.9211), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5807, mae: 1.1879, huber: 0.8411, swd: 1.3416, target_std: 6.5109
    Epoch [5/50], Val Losses: mse: 6.0215, mae: 1.3089, huber: 0.9575, swd: 1.3436, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.0277, mae: 1.4554, huber: 1.0985, swd: 1.5175, target_std: 4.7755
      Epoch 5 composite train-obj: 0.841102
            No improvement (0.9575), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.5212, mae: 1.1767, huber: 0.8313, swd: 1.3220, target_std: 6.5107
    Epoch [6/50], Val Losses: mse: 6.0177, mae: 1.3048, huber: 0.9555, swd: 1.4546, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.0523, mae: 1.4669, huber: 1.1107, swd: 1.7140, target_std: 4.7755
      Epoch 6 composite train-obj: 0.831327
            No improvement (0.9555), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.4406, mae: 1.1636, huber: 0.8195, swd: 1.2892, target_std: 6.5106
    Epoch [7/50], Val Losses: mse: 6.0645, mae: 1.3066, huber: 0.9578, swd: 1.5140, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.0579, mae: 1.4801, huber: 1.1228, swd: 1.8392, target_std: 4.7755
      Epoch 7 composite train-obj: 0.819515
            No improvement (0.9578), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3839, mae: 1.1543, huber: 0.8113, swd: 1.2736, target_std: 6.5108
    Epoch [8/50], Val Losses: mse: 6.0544, mae: 1.2972, huber: 0.9492, swd: 1.4117, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.0829, mae: 1.4652, huber: 1.1088, swd: 1.6681, target_std: 4.7755
      Epoch 8 composite train-obj: 0.811314
    Epoch [8/50], Test Losses: mse: 8.2032, mae: 1.4709, huber: 1.1126, swd: 1.5422, target_std: 4.7755
    Best round's Test MSE: 8.2034, MAE: 1.4710, SWD: 1.5422
    Best round's Validation MSE: 5.8067, MAE: 1.2547
    Best round's Test verification MSE : 8.2032, MAE: 1.4709, SWD: 1.5422
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.0440, mae: 1.5495, huber: 1.1786, swd: 3.3522, target_std: 6.5109
    Epoch [1/50], Val Losses: mse: 6.0795, mae: 1.3263, huber: 0.9717, swd: 1.5068, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.5928, mae: 1.5468, huber: 1.1839, swd: 1.9539, target_std: 4.7755
      Epoch 1 composite train-obj: 1.178598
            Val objective improved inf → 0.9717, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9645, mae: 1.2604, huber: 0.9063, swd: 1.4788, target_std: 6.5114
    Epoch [2/50], Val Losses: mse: 5.7953, mae: 1.2830, huber: 0.9322, swd: 1.3651, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.2199, mae: 1.4896, huber: 1.1305, swd: 1.7268, target_std: 4.7755
      Epoch 2 composite train-obj: 0.906275
            Val objective improved 0.9717 → 0.9322, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7448, mae: 1.2196, huber: 0.8695, swd: 1.3968, target_std: 6.5109
    Epoch [3/50], Val Losses: mse: 5.9572, mae: 1.3012, huber: 0.9472, swd: 1.3690, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.4290, mae: 1.5205, huber: 1.1571, swd: 1.8304, target_std: 4.7755
      Epoch 3 composite train-obj: 0.869481
            No improvement (0.9472), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.6456, mae: 1.2014, huber: 0.8533, swd: 1.3678, target_std: 6.5114
    Epoch [4/50], Val Losses: mse: 5.9955, mae: 1.2961, huber: 0.9444, swd: 1.3701, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.2535, mae: 1.4956, huber: 1.1351, swd: 1.8098, target_std: 4.7755
      Epoch 4 composite train-obj: 0.853299
            No improvement (0.9444), counter 2/5
    Epoch [5/50], Train Losses: mse: 4.5594, mae: 1.1859, huber: 0.8395, swd: 1.3314, target_std: 6.5116
    Epoch [5/50], Val Losses: mse: 5.8989, mae: 1.2781, huber: 0.9295, swd: 1.2863, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.1189, mae: 1.4687, huber: 1.1110, swd: 1.7257, target_std: 4.7755
      Epoch 5 composite train-obj: 0.839493
            Val objective improved 0.9322 → 0.9295, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.4993, mae: 1.1748, huber: 0.8296, swd: 1.3151, target_std: 6.5112
    Epoch [6/50], Val Losses: mse: 5.7813, mae: 1.2783, huber: 0.9299, swd: 1.2665, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.0171, mae: 1.4639, huber: 1.1070, swd: 1.6382, target_std: 4.7755
      Epoch 6 composite train-obj: 0.829620
            No improvement (0.9299), counter 1/5
    Epoch [7/50], Train Losses: mse: 4.4068, mae: 1.1584, huber: 0.8150, swd: 1.2721, target_std: 6.5109
    Epoch [7/50], Val Losses: mse: 5.8677, mae: 1.2786, huber: 0.9309, swd: 1.3575, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.0822, mae: 1.4722, huber: 1.1149, swd: 1.7810, target_std: 4.7755
      Epoch 7 composite train-obj: 0.815033
            No improvement (0.9309), counter 2/5
    Epoch [8/50], Train Losses: mse: 4.3809, mae: 1.1567, huber: 0.8136, swd: 1.2799, target_std: 6.5112
    Epoch [8/50], Val Losses: mse: 6.0505, mae: 1.3030, huber: 0.9543, swd: 1.3959, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.2506, mae: 1.4921, huber: 1.1338, swd: 1.8336, target_std: 4.7755
      Epoch 8 composite train-obj: 0.813572
            No improvement (0.9543), counter 3/5
    Epoch [9/50], Train Losses: mse: 4.3083, mae: 1.1423, huber: 0.8009, swd: 1.2399, target_std: 6.5108
    Epoch [9/50], Val Losses: mse: 5.8910, mae: 1.2770, huber: 0.9317, swd: 1.3611, target_std: 4.2867
    Epoch [9/50], Test Losses: mse: 7.9976, mae: 1.4679, huber: 1.1127, swd: 1.7952, target_std: 4.7755
      Epoch 9 composite train-obj: 0.800868
            No improvement (0.9317), counter 4/5
    Epoch [10/50], Train Losses: mse: 4.2545, mae: 1.1353, huber: 0.7944, swd: 1.2260, target_std: 6.5115
    Epoch [10/50], Val Losses: mse: 5.9486, mae: 1.3002, huber: 0.9531, swd: 1.6402, target_std: 4.2867
    Epoch [10/50], Test Losses: mse: 8.1256, mae: 1.4914, huber: 1.1349, swd: 2.0485, target_std: 4.7755
      Epoch 10 composite train-obj: 0.794441
    Epoch [10/50], Test Losses: mse: 8.1188, mae: 1.4687, huber: 1.1110, swd: 1.7255, target_std: 4.7755
    Best round's Test MSE: 8.1189, MAE: 1.4687, SWD: 1.7257
    Best round's Validation MSE: 5.8989, MAE: 1.2781
    Best round's Test verification MSE : 8.1188, MAE: 1.4687, SWD: 1.7255
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.3459, mae: 1.5652, huber: 1.1938, swd: 3.3368, target_std: 6.5107
    Epoch [1/50], Val Losses: mse: 6.1632, mae: 1.3170, huber: 0.9617, swd: 1.2936, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.7099, mae: 1.5443, huber: 1.1792, swd: 1.6688, target_std: 4.7755
      Epoch 1 composite train-obj: 1.193845
            Val objective improved inf → 0.9617, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9412, mae: 1.2613, huber: 0.9058, swd: 1.3610, target_std: 6.5112
    Epoch [2/50], Val Losses: mse: 5.8670, mae: 1.2993, huber: 0.9468, swd: 1.4940, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.3696, mae: 1.5317, huber: 1.1689, swd: 1.9813, target_std: 4.7755
      Epoch 2 composite train-obj: 0.905801
            Val objective improved 0.9617 → 0.9468, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7401, mae: 1.2208, huber: 0.8696, swd: 1.2790, target_std: 6.5115
    Epoch [3/50], Val Losses: mse: 5.7482, mae: 1.2732, huber: 0.9229, swd: 1.2382, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.2168, mae: 1.4851, huber: 1.1265, swd: 1.5988, target_std: 4.7755
      Epoch 3 composite train-obj: 0.869650
            Val objective improved 0.9468 → 0.9229, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6294, mae: 1.1986, huber: 0.8499, swd: 1.2394, target_std: 6.5110
    Epoch [4/50], Val Losses: mse: 5.8531, mae: 1.2801, huber: 0.9300, swd: 1.3033, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.2128, mae: 1.4854, huber: 1.1274, swd: 1.6911, target_std: 4.7755
      Epoch 4 composite train-obj: 0.849949
            No improvement (0.9300), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5468, mae: 1.1845, huber: 0.8375, swd: 1.2226, target_std: 6.5112
    Epoch [5/50], Val Losses: mse: 5.8166, mae: 1.2831, huber: 0.9326, swd: 1.3115, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.1383, mae: 1.4770, huber: 1.1200, swd: 1.6934, target_std: 4.7755
      Epoch 5 composite train-obj: 0.837542
            No improvement (0.9326), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.4814, mae: 1.1728, huber: 0.8273, swd: 1.2009, target_std: 6.5119
    Epoch [6/50], Val Losses: mse: 5.8111, mae: 1.2743, huber: 0.9265, swd: 1.2142, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.1570, mae: 1.4757, huber: 1.1197, swd: 1.6185, target_std: 4.7755
      Epoch 6 composite train-obj: 0.827340
            No improvement (0.9265), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.4018, mae: 1.1600, huber: 0.8160, swd: 1.1784, target_std: 6.5110
    Epoch [7/50], Val Losses: mse: 5.7497, mae: 1.2894, huber: 0.9405, swd: 1.4331, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.0081, mae: 1.4872, huber: 1.1306, swd: 1.8367, target_std: 4.7755
      Epoch 7 composite train-obj: 0.816044
            No improvement (0.9405), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3444, mae: 1.1508, huber: 0.8078, swd: 1.1606, target_std: 6.5116
    Epoch [8/50], Val Losses: mse: 5.9416, mae: 1.2963, huber: 0.9480, swd: 1.2512, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.1177, mae: 1.4792, huber: 1.1221, swd: 1.5571, target_std: 4.7755
      Epoch 8 composite train-obj: 0.807823
    Epoch [8/50], Test Losses: mse: 8.2169, mae: 1.4851, huber: 1.1265, swd: 1.5988, target_std: 4.7755
    Best round's Test MSE: 8.2168, MAE: 1.4851, SWD: 1.5988
    Best round's Validation MSE: 5.7482, MAE: 1.2732
    Best round's Test verification MSE : 8.2169, MAE: 1.4851, SWD: 1.5988
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred96_20250429_1207)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 8.1797 ± 0.0433
      mae: 1.4749 ± 0.0073
      huber: 1.1167 ± 0.0070
      swd: 1.6222 ± 0.0767
      target_std: 4.7755 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.8179 ± 0.0620
      mae: 1.2687 ± 0.0101
      huber: 0.9193 ± 0.0101
      swd: 1.2366 ± 0.0413
      target_std: 4.2867 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm1_seq96_pred96_20250429_1207
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

##### Ab: shift_inside_scale 


```python
importlib.reload(monotonic)
importlib.reload(train_config)
# utils.reload_modules([modules_to_reload_list])

cfg_ACL_96_96 = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
)
cfg_ACL_96_96.x_to_z_delay.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_96.x_to_z_deri.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_96.z_to_x_main.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_96.z_to_y_main.use_magnitude_for_scale_and_translate = [False, True]
exp_ACL_96_96 = execute_model_evaluation('ettm1', cfg_ACL_96_96, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 7.6619, mae: 1.5372, huber: 1.1645, swd: 3.1028, target_std: 6.5106
    Epoch [1/50], Val Losses: mse: 6.0710, mae: 1.3251, huber: 0.9699, swd: 1.4462, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.5961, mae: 1.5381, huber: 1.1752, swd: 1.8054, target_std: 4.7755
      Epoch 1 composite train-obj: 1.164529
            Val objective improved inf → 0.9699, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9782, mae: 1.2659, huber: 0.9101, swd: 1.4815, target_std: 6.5119
    Epoch [2/50], Val Losses: mse: 5.7895, mae: 1.2898, huber: 0.9374, swd: 1.4178, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.3118, mae: 1.5009, huber: 1.1416, swd: 1.8205, target_std: 4.7755
      Epoch 2 composite train-obj: 0.910096
            Val objective improved 0.9699 → 0.9374, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7871, mae: 1.2290, huber: 0.8771, swd: 1.4210, target_std: 6.5115
    Epoch [3/50], Val Losses: mse: 5.7416, mae: 1.2771, huber: 0.9238, swd: 1.2266, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.2329, mae: 1.4763, huber: 1.1159, swd: 1.5260, target_std: 4.7755
      Epoch 3 composite train-obj: 0.877148
            Val objective improved 0.9374 → 0.9238, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6628, mae: 1.2054, huber: 0.8561, swd: 1.3661, target_std: 6.5111
    Epoch [4/50], Val Losses: mse: 5.9316, mae: 1.3092, huber: 0.9566, swd: 1.5150, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.3027, mae: 1.5019, huber: 1.1428, swd: 1.8758, target_std: 4.7755
      Epoch 4 composite train-obj: 0.856060
            No improvement (0.9566), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5685, mae: 1.1873, huber: 0.8400, swd: 1.3271, target_std: 6.5104
    Epoch [5/50], Val Losses: mse: 5.8809, mae: 1.3089, huber: 0.9551, swd: 1.4287, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.2942, mae: 1.4981, huber: 1.1375, swd: 1.7627, target_std: 4.7755
      Epoch 5 composite train-obj: 0.840016
            No improvement (0.9551), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.4988, mae: 1.1762, huber: 0.8301, swd: 1.3054, target_std: 6.5108
    Epoch [6/50], Val Losses: mse: 5.9742, mae: 1.3136, huber: 0.9628, swd: 1.5970, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.2528, mae: 1.5029, huber: 1.1444, swd: 1.9858, target_std: 4.7755
      Epoch 6 composite train-obj: 0.830118
            No improvement (0.9628), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.4537, mae: 1.1688, huber: 0.8236, swd: 1.3016, target_std: 6.5115
    Epoch [7/50], Val Losses: mse: 5.8665, mae: 1.2942, huber: 0.9435, swd: 1.4786, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.0869, mae: 1.4822, huber: 1.1240, swd: 1.8442, target_std: 4.7755
      Epoch 7 composite train-obj: 0.823593
            No improvement (0.9435), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3723, mae: 1.1549, huber: 0.8111, swd: 1.2662, target_std: 6.5109
    Epoch [8/50], Val Losses: mse: 6.1640, mae: 1.3218, huber: 0.9680, swd: 1.4223, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.3566, mae: 1.5072, huber: 1.1450, swd: 1.7734, target_std: 4.7755
      Epoch 8 composite train-obj: 0.811124
    Epoch [8/50], Test Losses: mse: 8.2342, mae: 1.4764, huber: 1.1160, swd: 1.5259, target_std: 4.7755
    Best round's Test MSE: 8.2329, MAE: 1.4763, SWD: 1.5260
    Best round's Validation MSE: 5.7416, MAE: 1.2771
    Best round's Test verification MSE : 8.2342, MAE: 1.4764, SWD: 1.5259
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.8386, mae: 1.5361, huber: 1.1645, swd: 3.2015, target_std: 6.5112
    Epoch [1/50], Val Losses: mse: 6.1184, mae: 1.3063, huber: 0.9517, swd: 1.1939, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.6994, mae: 1.5386, huber: 1.1734, swd: 1.6769, target_std: 4.7755
      Epoch 1 composite train-obj: 1.164519
            Val objective improved inf → 0.9517, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9421, mae: 1.2607, huber: 0.9053, swd: 1.4616, target_std: 6.5111
    Epoch [2/50], Val Losses: mse: 6.0475, mae: 1.2925, huber: 0.9366, swd: 1.1839, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.6480, mae: 1.5220, huber: 1.1561, swd: 1.6323, target_std: 4.7755
      Epoch 2 composite train-obj: 0.905342
            Val objective improved 0.9517 → 0.9366, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7484, mae: 1.2219, huber: 0.8708, swd: 1.3809, target_std: 6.5115
    Epoch [3/50], Val Losses: mse: 5.7234, mae: 1.2553, huber: 0.9057, swd: 1.2412, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.2320, mae: 1.4747, huber: 1.1158, swd: 1.6941, target_std: 4.7755
      Epoch 3 composite train-obj: 0.870786
            Val objective improved 0.9366 → 0.9057, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6426, mae: 1.2022, huber: 0.8534, swd: 1.3540, target_std: 6.5115
    Epoch [4/50], Val Losses: mse: 5.8188, mae: 1.2834, huber: 0.9326, swd: 1.4652, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.1518, mae: 1.5031, huber: 1.1423, swd: 2.0106, target_std: 4.7755
      Epoch 4 composite train-obj: 0.853390
            No improvement (0.9326), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5484, mae: 1.1858, huber: 0.8388, swd: 1.3144, target_std: 6.5113
    Epoch [5/50], Val Losses: mse: 5.9913, mae: 1.2780, huber: 0.9266, swd: 1.2026, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.3053, mae: 1.4866, huber: 1.1256, swd: 1.6691, target_std: 4.7755
      Epoch 5 composite train-obj: 0.838801
            No improvement (0.9266), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.4766, mae: 1.1723, huber: 0.8268, swd: 1.2936, target_std: 6.5112
    Epoch [6/50], Val Losses: mse: 5.7745, mae: 1.2684, huber: 0.9201, swd: 1.3348, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.1360, mae: 1.4837, huber: 1.1263, swd: 1.8465, target_std: 4.7755
      Epoch 6 composite train-obj: 0.826818
            No improvement (0.9201), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.4108, mae: 1.1628, huber: 0.8185, swd: 1.2790, target_std: 6.5114
    Epoch [7/50], Val Losses: mse: 5.7881, mae: 1.2647, huber: 0.9177, swd: 1.2249, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 7.9665, mae: 1.4568, huber: 1.1010, swd: 1.6550, target_std: 4.7755
      Epoch 7 composite train-obj: 0.818527
            No improvement (0.9177), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3443, mae: 1.1520, huber: 0.8087, swd: 1.2511, target_std: 6.5111
    Epoch [8/50], Val Losses: mse: 6.1651, mae: 1.2945, huber: 0.9454, swd: 1.3146, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.2686, mae: 1.4948, huber: 1.1363, swd: 1.8340, target_std: 4.7755
      Epoch 8 composite train-obj: 0.808707
    Epoch [8/50], Test Losses: mse: 8.2321, mae: 1.4746, huber: 1.1157, swd: 1.6941, target_std: 4.7755
    Best round's Test MSE: 8.2320, MAE: 1.4747, SWD: 1.6941
    Best round's Validation MSE: 5.7234, MAE: 1.2553
    Best round's Test verification MSE : 8.2321, MAE: 1.4746, SWD: 1.6941
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.4873, mae: 1.5105, huber: 1.1401, swd: 2.7771, target_std: 6.5112
    Epoch [1/50], Val Losses: mse: 5.9608, mae: 1.2988, huber: 0.9458, swd: 1.2671, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.4593, mae: 1.5216, huber: 1.1599, swd: 1.6439, target_std: 4.7755
      Epoch 1 composite train-obj: 1.140091
            Val objective improved inf → 0.9458, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9292, mae: 1.2560, huber: 0.9015, swd: 1.3611, target_std: 6.5109
    Epoch [2/50], Val Losses: mse: 5.7661, mae: 1.2658, huber: 0.9155, swd: 1.1654, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.2972, mae: 1.4884, huber: 1.1296, swd: 1.5574, target_std: 4.7755
      Epoch 2 composite train-obj: 0.901482
            Val objective improved 0.9458 → 0.9155, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7383, mae: 1.2189, huber: 0.8681, swd: 1.2889, target_std: 6.5112
    Epoch [3/50], Val Losses: mse: 5.8145, mae: 1.2795, huber: 0.9279, swd: 1.3647, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.1120, mae: 1.4891, huber: 1.1295, swd: 1.7957, target_std: 4.7755
      Epoch 3 composite train-obj: 0.868080
            No improvement (0.9279), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.6315, mae: 1.2010, huber: 0.8520, swd: 1.2636, target_std: 6.5108
    Epoch [4/50], Val Losses: mse: 5.9361, mae: 1.3005, huber: 0.9475, swd: 1.3353, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.2945, mae: 1.5030, huber: 1.1425, swd: 1.7812, target_std: 4.7755
      Epoch 4 composite train-obj: 0.852042
            No improvement (0.9475), counter 2/5
    Epoch [5/50], Train Losses: mse: 4.5142, mae: 1.1790, huber: 0.8324, swd: 1.2109, target_std: 6.5113
    Epoch [5/50], Val Losses: mse: 5.9820, mae: 1.2980, huber: 0.9463, swd: 1.4109, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.1793, mae: 1.5044, huber: 1.1448, swd: 1.9193, target_std: 4.7755
      Epoch 5 composite train-obj: 0.832355
            No improvement (0.9463), counter 3/5
    Epoch [6/50], Train Losses: mse: 4.4666, mae: 1.1708, huber: 0.8252, swd: 1.2026, target_std: 6.5107
    Epoch [6/50], Val Losses: mse: 5.9388, mae: 1.2967, huber: 0.9456, swd: 1.3044, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.1192, mae: 1.4843, huber: 1.1260, swd: 1.6956, target_std: 4.7755
      Epoch 6 composite train-obj: 0.825212
            No improvement (0.9456), counter 4/5
    Epoch [7/50], Train Losses: mse: 4.4093, mae: 1.1627, huber: 0.8179, swd: 1.1905, target_std: 6.5115
    Epoch [7/50], Val Losses: mse: 6.0606, mae: 1.3184, huber: 0.9664, swd: 1.5712, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.2997, mae: 1.5270, huber: 1.1662, swd: 2.0625, target_std: 4.7755
      Epoch 7 composite train-obj: 0.817934
    Epoch [7/50], Test Losses: mse: 8.2970, mae: 1.4884, huber: 1.1296, swd: 1.5574, target_std: 4.7755
    Best round's Test MSE: 8.2972, MAE: 1.4884, SWD: 1.5574
    Best round's Validation MSE: 5.7661, MAE: 1.2658
    Best round's Test verification MSE : 8.2970, MAE: 1.4884, SWD: 1.5574
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred96_20250429_1220)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 8.2540 ± 0.0305
      mae: 1.4798 ± 0.0061
      huber: 1.1204 ± 0.0065
      swd: 1.5925 ± 0.0729
      target_std: 4.7755 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.7437 ± 0.0175
      mae: 1.2661 ± 0.0089
      huber: 0.9150 ± 0.0074
      swd: 1.2111 ± 0.0328
      target_std: 4.2867 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm1_seq96_pred96_20250429_1220
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

##### Ab: Original


```python
cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
    householder_reflects_latent=4,
    householder_reflects_data=8,
)
cfg.x_to_z_delay.enable_magnitudes_scale_shift = [False, True]
cfg.x_to_z_delay.spetral_flags_for_magnitude_scale_shift = [False, False]  
cfg.x_to_z_delay.spectral_flags_for_scale_shift = [False, False]
cfg.x_to_z_delay.hidden_spectral_flags = [False, True]

cfg.x_to_z_deri.enable_magnitudes_scale_shift = [False, True]
cfg.x_to_z_deri.spetral_flags_for_magnitude_scale_shift = [False, False]
cfg.x_to_z_deri.spectral_flags_for_scale_shift = [False, False]
cfg.x_to_z_deri.hidden_spectral_flags = [False, True]

cfg.z_to_x_main.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_x_main.spetral_flags_for_magnitude_scale_shift = [False, False]
cfg.z_to_x_main.spectral_flags_for_scale_shift = [False, False]
cfg.z_to_x_main.hidden_spectral_flags = [False, True]

cfg.z_to_y_main.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_y_main.spetral_flags_for_magnitude_scale_shift = [False, False]
cfg.z_to_y_main.spectral_flags_for_scale_shift = [False, False]
cfg.z_to_y_main.hidden_spectral_flags = [False, True]
exp = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 7.6875, mae: 1.5172, huber: 1.1489, swd: 3.1952, target_std: 6.5113
    Epoch [1/50], Val Losses: mse: 5.8986, mae: 1.2822, huber: 0.9312, swd: 1.3105, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.4413, mae: 1.5008, huber: 1.1417, swd: 1.6654, target_std: 4.7755
      Epoch 1 composite train-obj: 1.148918
            Val objective improved inf → 0.9312, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9381, mae: 1.2512, huber: 0.8983, swd: 1.4990, target_std: 6.5109
    Epoch [2/50], Val Losses: mse: 5.8548, mae: 1.2730, huber: 0.9226, swd: 1.3098, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.2169, mae: 1.4814, huber: 1.1221, swd: 1.6663, target_std: 4.7755
      Epoch 2 composite train-obj: 0.898271
            Val objective improved 0.9312 → 0.9226, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7638, mae: 1.2201, huber: 0.8700, swd: 1.4286, target_std: 6.5113
    Epoch [3/50], Val Losses: mse: 5.6871, mae: 1.2469, huber: 0.8989, swd: 1.2091, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.0439, mae: 1.4545, huber: 1.0975, swd: 1.5864, target_std: 4.7755
      Epoch 3 composite train-obj: 0.869972
            Val objective improved 0.9226 → 0.8989, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6492, mae: 1.1999, huber: 0.8518, swd: 1.3821, target_std: 6.5107
    Epoch [4/50], Val Losses: mse: 5.7159, mae: 1.2523, huber: 0.9037, swd: 1.3191, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.1541, mae: 1.4758, huber: 1.1178, swd: 1.7912, target_std: 4.7755
      Epoch 4 composite train-obj: 0.851849
            No improvement (0.9037), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5662, mae: 1.1845, huber: 0.8381, swd: 1.3461, target_std: 6.5113
    Epoch [5/50], Val Losses: mse: 5.7384, mae: 1.2661, huber: 0.9166, swd: 1.3002, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.1008, mae: 1.4660, huber: 1.1089, swd: 1.7112, target_std: 4.7755
      Epoch 5 composite train-obj: 0.838092
            No improvement (0.9166), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.4945, mae: 1.1726, huber: 0.8275, swd: 1.3213, target_std: 6.5109
    Epoch [6/50], Val Losses: mse: 5.7017, mae: 1.2694, huber: 0.9220, swd: 1.4021, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 7.9092, mae: 1.4713, huber: 1.1141, swd: 1.8575, target_std: 4.7755
      Epoch 6 composite train-obj: 0.827502
            No improvement (0.9220), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.4629, mae: 1.1687, huber: 0.8240, swd: 1.3210, target_std: 6.5105
    Epoch [7/50], Val Losses: mse: 5.8070, mae: 1.2669, huber: 0.9191, swd: 1.3505, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.0912, mae: 1.4819, huber: 1.1246, swd: 1.8881, target_std: 4.7755
      Epoch 7 composite train-obj: 0.824004
            No improvement (0.9191), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3945, mae: 1.1575, huber: 0.8142, swd: 1.2996, target_std: 6.5107
    Epoch [8/50], Val Losses: mse: 5.8056, mae: 1.2894, huber: 0.9406, swd: 1.5356, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.0252, mae: 1.4839, huber: 1.1281, swd: 1.9827, target_std: 4.7755
      Epoch 8 composite train-obj: 0.814166
    Epoch [8/50], Test Losses: mse: 8.0441, mae: 1.4545, huber: 1.0976, swd: 1.5865, target_std: 4.7755
    Best round's Test MSE: 8.0439, MAE: 1.4545, SWD: 1.5864
    Best round's Validation MSE: 5.6871, MAE: 1.2469
    Best round's Test verification MSE : 8.0441, MAE: 1.4545, SWD: 1.5865
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.0897, mae: 1.5493, huber: 1.1795, swd: 3.4044, target_std: 6.5109
    Epoch [1/50], Val Losses: mse: 6.0773, mae: 1.3122, huber: 0.9591, swd: 1.5121, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.5739, mae: 1.5358, huber: 1.1727, swd: 1.9403, target_std: 4.7755
      Epoch 1 composite train-obj: 1.179525
            Val objective improved inf → 0.9591, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9041, mae: 1.2500, huber: 0.8969, swd: 1.4532, target_std: 6.5112
    Epoch [2/50], Val Losses: mse: 5.7892, mae: 1.2668, huber: 0.9174, swd: 1.3507, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.2073, mae: 1.4886, huber: 1.1292, swd: 1.8219, target_std: 4.7755
      Epoch 2 composite train-obj: 0.896876
            Val objective improved 0.9591 → 0.9174, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7368, mae: 1.2184, huber: 0.8686, swd: 1.4003, target_std: 6.5111
    Epoch [3/50], Val Losses: mse: 5.8345, mae: 1.2710, huber: 0.9209, swd: 1.2795, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.1450, mae: 1.4684, huber: 1.1101, swd: 1.6299, target_std: 4.7755
      Epoch 3 composite train-obj: 0.868562
            No improvement (0.9209), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.6107, mae: 1.1951, huber: 0.8475, swd: 1.3419, target_std: 6.5110
    Epoch [4/50], Val Losses: mse: 5.8426, mae: 1.2906, huber: 0.9395, swd: 1.3415, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.0907, mae: 1.4766, huber: 1.1191, swd: 1.7171, target_std: 4.7755
      Epoch 4 composite train-obj: 0.847490
            No improvement (0.9395), counter 2/5
    Epoch [5/50], Train Losses: mse: 4.5309, mae: 1.1812, huber: 0.8351, swd: 1.3141, target_std: 6.5109
    Epoch [5/50], Val Losses: mse: 5.8491, mae: 1.2935, huber: 0.9423, swd: 1.4886, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.1175, mae: 1.5008, huber: 1.1410, swd: 2.0167, target_std: 4.7755
      Epoch 5 composite train-obj: 0.835064
            No improvement (0.9423), counter 3/5
    Epoch [6/50], Train Losses: mse: 4.4580, mae: 1.1673, huber: 0.8229, swd: 1.2864, target_std: 6.5111
    Epoch [6/50], Val Losses: mse: 5.8551, mae: 1.2924, huber: 0.9433, swd: 1.5266, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.0692, mae: 1.4873, huber: 1.1304, swd: 1.9174, target_std: 4.7755
      Epoch 6 composite train-obj: 0.822882
            No improvement (0.9433), counter 4/5
    Epoch [7/50], Train Losses: mse: 4.3917, mae: 1.1569, huber: 0.8136, swd: 1.2637, target_std: 6.5110
    Epoch [7/50], Val Losses: mse: 5.9507, mae: 1.2895, huber: 0.9406, swd: 1.3955, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.2352, mae: 1.4903, huber: 1.1322, swd: 1.7846, target_std: 4.7755
      Epoch 7 composite train-obj: 0.813621
    Epoch [7/50], Test Losses: mse: 8.2074, mae: 1.4887, huber: 1.1292, swd: 1.8221, target_std: 4.7755
    Best round's Test MSE: 8.2073, MAE: 1.4886, SWD: 1.8219
    Best round's Validation MSE: 5.7892, MAE: 1.2668
    Best round's Test verification MSE : 8.2074, MAE: 1.4887, SWD: 1.8221
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.8859, mae: 1.5233, huber: 1.1551, swd: 3.0578, target_std: 6.5112
    Epoch [1/50], Val Losses: mse: 6.0839, mae: 1.3135, huber: 0.9617, swd: 1.4252, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.4408, mae: 1.5234, huber: 1.1633, swd: 1.7975, target_std: 4.7755
      Epoch 1 composite train-obj: 1.155144
            Val objective improved inf → 0.9617, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8962, mae: 1.2468, huber: 0.8944, swd: 1.3466, target_std: 6.5105
    Epoch [2/50], Val Losses: mse: 5.9027, mae: 1.2730, huber: 0.9233, swd: 1.2855, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.2293, mae: 1.4859, huber: 1.1271, swd: 1.6737, target_std: 4.7755
      Epoch 2 composite train-obj: 0.894395
            Val objective improved 0.9617 → 0.9233, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7360, mae: 1.2168, huber: 0.8671, swd: 1.2936, target_std: 6.5112
    Epoch [3/50], Val Losses: mse: 5.7129, mae: 1.2495, huber: 0.9009, swd: 1.1536, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.1354, mae: 1.4745, huber: 1.1166, swd: 1.5481, target_std: 4.7755
      Epoch 3 composite train-obj: 0.867110
            Val objective improved 0.9233 → 0.9009, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6210, mae: 1.1973, huber: 0.8495, swd: 1.2609, target_std: 6.5109
    Epoch [4/50], Val Losses: mse: 5.7969, mae: 1.2752, huber: 0.9247, swd: 1.3466, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.1617, mae: 1.4988, huber: 1.1392, swd: 1.8746, target_std: 4.7755
      Epoch 4 composite train-obj: 0.849470
            No improvement (0.9247), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5226, mae: 1.1796, huber: 0.8338, swd: 1.2248, target_std: 6.5112
    Epoch [5/50], Val Losses: mse: 5.9146, mae: 1.2794, huber: 0.9306, swd: 1.2760, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.0441, mae: 1.4738, huber: 1.1169, swd: 1.6907, target_std: 4.7755
      Epoch 5 composite train-obj: 0.833760
            No improvement (0.9306), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.4320, mae: 1.1647, huber: 0.8204, swd: 1.1915, target_std: 6.5109
    Epoch [6/50], Val Losses: mse: 5.9240, mae: 1.2855, huber: 0.9365, swd: 1.4067, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.2161, mae: 1.5040, huber: 1.1470, swd: 1.9657, target_std: 4.7755
      Epoch 6 composite train-obj: 0.820395
            No improvement (0.9365), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.3494, mae: 1.1519, huber: 0.8087, swd: 1.1663, target_std: 6.5110
    Epoch [7/50], Val Losses: mse: 5.9255, mae: 1.2876, huber: 0.9392, swd: 1.3595, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.0170, mae: 1.4849, huber: 1.1286, swd: 1.7490, target_std: 4.7755
      Epoch 7 composite train-obj: 0.808733
            No improvement (0.9392), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3007, mae: 1.1455, huber: 0.8030, swd: 1.1628, target_std: 6.5103
    Epoch [8/50], Val Losses: mse: 5.9184, mae: 1.2975, huber: 0.9482, swd: 1.4166, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 7.9469, mae: 1.4833, huber: 1.1259, swd: 1.7971, target_std: 4.7755
      Epoch 8 composite train-obj: 0.802978
    Epoch [8/50], Test Losses: mse: 8.1354, mae: 1.4745, huber: 1.1166, swd: 1.5481, target_std: 4.7755
    Best round's Test MSE: 8.1354, MAE: 1.4745, SWD: 1.5481
    Best round's Validation MSE: 5.7129, MAE: 1.2495
    Best round's Test verification MSE : 8.1354, MAE: 1.4745, SWD: 1.5481
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred96_20250501_2113)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 8.1289 ± 0.0669
      mae: 1.4726 ± 0.0140
      huber: 1.1145 ± 0.0130
      swd: 1.6522 ± 0.1211
      target_std: 4.7755 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.7297 ± 0.0433
      mae: 1.2544 ± 0.0088
      huber: 0.9057 ± 0.0083
      swd: 1.2378 ± 0.0830
      target_std: 4.2867 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm1_seq96_pred96_20250501_2113
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    







#### 96-196

#### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    global_std.shape: torch.Size([7])
    Global Std for ettm1: tensor([7.0829, 2.0413, 6.8291, 1.8072, 1.1741, 0.6004, 8.5648],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 7.8547, mae: 1.5181, huber: 1.1500, swd: 3.1883, ept: 61.0570
    Epoch [1/50], Val Losses: mse: 5.9294, mae: 1.2819, huber: 0.9302, swd: 1.2307, ept: 67.8175
    Epoch [1/50], Test Losses: mse: 8.4386, mae: 1.5038, huber: 1.1430, swd: 1.6213, ept: 56.4718
      Epoch 1 composite train-obj: 1.150014
            Val objective improved inf → 0.9302, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8799, mae: 1.2473, huber: 0.8939, swd: 1.4607, ept: 68.7232
    Epoch [2/50], Val Losses: mse: 5.7983, mae: 1.2671, huber: 0.9171, swd: 1.2885, ept: 68.1259
    Epoch [2/50], Test Losses: mse: 8.2434, mae: 1.4890, huber: 1.1292, swd: 1.7221, ept: 57.4250
      Epoch 2 composite train-obj: 0.893868
            Val objective improved 0.9302 → 0.9171, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7433, mae: 1.2212, huber: 0.8703, swd: 1.4232, ept: 69.8795
    Epoch [3/50], Val Losses: mse: 5.7727, mae: 1.2570, huber: 0.9080, swd: 1.2196, ept: 68.9871
    Epoch [3/50], Test Losses: mse: 8.1561, mae: 1.4710, huber: 1.1130, swd: 1.6556, ept: 58.5190
      Epoch 3 composite train-obj: 0.870292
            Val objective improved 0.9171 → 0.9080, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6066, mae: 1.1941, huber: 0.8460, swd: 1.3546, ept: 70.7452
    Epoch [4/50], Val Losses: mse: 5.7280, mae: 1.2691, huber: 0.9189, swd: 1.3677, ept: 68.7950
    Epoch [4/50], Test Losses: mse: 8.0600, mae: 1.4798, huber: 1.1207, swd: 1.8770, ept: 58.4265
      Epoch 4 composite train-obj: 0.846006
            No improvement (0.9189), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5273, mae: 1.1810, huber: 0.8342, swd: 1.3323, ept: 71.4017
    Epoch [5/50], Val Losses: mse: 5.7447, mae: 1.2848, huber: 0.9340, swd: 1.4741, ept: 70.0179
    Epoch [5/50], Test Losses: mse: 8.0684, mae: 1.4935, huber: 1.1343, swd: 2.0299, ept: 59.1080
      Epoch 5 composite train-obj: 0.834240
            No improvement (0.9340), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.4494, mae: 1.1679, huber: 0.8227, swd: 1.2995, ept: 71.8607
    Epoch [6/50], Val Losses: mse: 5.8880, mae: 1.2802, huber: 0.9325, swd: 1.3881, ept: 69.2865
    Epoch [6/50], Test Losses: mse: 8.0210, mae: 1.4762, huber: 1.1189, swd: 1.9256, ept: 58.9119
      Epoch 6 composite train-obj: 0.822655
            No improvement (0.9325), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.3830, mae: 1.1577, huber: 0.8135, swd: 1.2794, ept: 72.2362
    Epoch [7/50], Val Losses: mse: 5.9438, mae: 1.3146, huber: 0.9636, swd: 1.6248, ept: 68.7051
    Epoch [7/50], Test Losses: mse: 8.1214, mae: 1.5105, huber: 1.1517, swd: 2.2071, ept: 58.4121
      Epoch 7 composite train-obj: 0.813532
            No improvement (0.9636), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3217, mae: 1.1482, huber: 0.8049, swd: 1.2629, ept: 72.5155
    Epoch [8/50], Val Losses: mse: 5.9305, mae: 1.2894, huber: 0.9399, swd: 1.3096, ept: 70.5192
    Epoch [8/50], Test Losses: mse: 8.0179, mae: 1.4623, huber: 1.1051, swd: 1.6320, ept: 59.8166
      Epoch 8 composite train-obj: 0.804919
    Epoch [8/50], Test Losses: mse: 8.1554, mae: 1.4710, huber: 1.1130, swd: 1.6556, ept: 58.5159
    Best round's Test MSE: 8.1561, MAE: 1.4710, SWD: 1.6556
    Best round's Validation MSE: 5.7727, MAE: 1.2570, SWD: 1.2196
    Best round's Test verification MSE : 8.1554, MAE: 1.4710, SWD: 1.6556
    Time taken: 93.36 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.8903, mae: 1.5186, huber: 1.1507, swd: 3.2115, ept: 60.4415
    Epoch [1/50], Val Losses: mse: 6.0114, mae: 1.3036, huber: 0.9516, swd: 1.4906, ept: 65.4278
    Epoch [1/50], Test Losses: mse: 8.4750, mae: 1.5286, huber: 1.1676, swd: 2.0162, ept: 54.9859
      Epoch 1 composite train-obj: 1.150724
            Val objective improved inf → 0.9516, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8872, mae: 1.2447, huber: 0.8920, swd: 1.4376, ept: 68.7769
    Epoch [2/50], Val Losses: mse: 5.7777, mae: 1.2655, huber: 0.9156, swd: 1.3682, ept: 68.4782
    Epoch [2/50], Test Losses: mse: 8.2747, mae: 1.4883, huber: 1.1298, swd: 1.8389, ept: 57.1239
      Epoch 2 composite train-obj: 0.891957
            Val objective improved 0.9516 → 0.9156, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7394, mae: 1.2171, huber: 0.8671, swd: 1.3962, ept: 70.1557
    Epoch [3/50], Val Losses: mse: 5.7124, mae: 1.2740, huber: 0.9231, swd: 1.4561, ept: 67.7100
    Epoch [3/50], Test Losses: mse: 8.1288, mae: 1.4963, huber: 1.1371, swd: 1.9611, ept: 57.4810
      Epoch 3 composite train-obj: 0.867099
            No improvement (0.9231), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.6354, mae: 1.1992, huber: 0.8510, swd: 1.3572, ept: 70.9118
    Epoch [4/50], Val Losses: mse: 5.7562, mae: 1.2717, huber: 0.9194, swd: 1.2558, ept: 69.4646
    Epoch [4/50], Test Losses: mse: 8.2295, mae: 1.4832, huber: 1.1228, swd: 1.6977, ept: 58.9138
      Epoch 4 composite train-obj: 0.851035
            No improvement (0.9194), counter 2/5
    Epoch [5/50], Train Losses: mse: 4.5319, mae: 1.1816, huber: 0.8352, swd: 1.3162, ept: 71.4805
    Epoch [5/50], Val Losses: mse: 5.7861, mae: 1.2870, huber: 0.9367, swd: 1.4102, ept: 69.5018
    Epoch [5/50], Test Losses: mse: 8.0929, mae: 1.4789, huber: 1.1216, swd: 1.8499, ept: 58.7763
      Epoch 5 composite train-obj: 0.835219
            No improvement (0.9367), counter 3/5
    Epoch [6/50], Train Losses: mse: 4.4613, mae: 1.1700, huber: 0.8247, swd: 1.2922, ept: 71.8815
    Epoch [6/50], Val Losses: mse: 5.8074, mae: 1.2859, huber: 0.9350, swd: 1.3145, ept: 70.4041
    Epoch [6/50], Test Losses: mse: 8.1255, mae: 1.4787, huber: 1.1209, swd: 1.7757, ept: 59.4015
      Epoch 6 composite train-obj: 0.824733
            No improvement (0.9350), counter 4/5
    Epoch [7/50], Train Losses: mse: 4.3869, mae: 1.1581, huber: 0.8138, swd: 1.2677, ept: 72.2415
    Epoch [7/50], Val Losses: mse: 5.9620, mae: 1.2856, huber: 0.9372, swd: 1.3560, ept: 70.7937
    Epoch [7/50], Test Losses: mse: 8.1965, mae: 1.4838, huber: 1.1267, swd: 1.8471, ept: 59.2727
      Epoch 7 composite train-obj: 0.813848
    Epoch [7/50], Test Losses: mse: 8.2740, mae: 1.4883, huber: 1.1297, swd: 1.8384, ept: 57.1238
    Best round's Test MSE: 8.2747, MAE: 1.4883, SWD: 1.8389
    Best round's Validation MSE: 5.7777, MAE: 1.2655, SWD: 1.3682
    Best round's Test verification MSE : 8.2740, MAE: 1.4883, SWD: 1.8384
    Time taken: 81.91 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.1567, mae: 1.5384, huber: 1.1699, swd: 3.1085, ept: 60.2653
    Epoch [1/50], Val Losses: mse: 5.8802, mae: 1.2869, huber: 0.9351, swd: 1.3231, ept: 65.5908
    Epoch [1/50], Test Losses: mse: 8.4301, mae: 1.5128, huber: 1.1518, swd: 1.6579, ept: 54.8111
      Epoch 1 composite train-obj: 1.169915
            Val objective improved inf → 0.9351, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8485, mae: 1.2421, huber: 0.8897, swd: 1.3412, ept: 68.9951
    Epoch [2/50], Val Losses: mse: 5.6760, mae: 1.2673, huber: 0.9174, swd: 1.3549, ept: 67.7735
    Epoch [2/50], Test Losses: mse: 8.0091, mae: 1.4718, huber: 1.1134, swd: 1.6330, ept: 57.4807
      Epoch 2 composite train-obj: 0.889651
            Val objective improved 0.9351 → 0.9174, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.6859, mae: 1.2102, huber: 0.8611, swd: 1.2821, ept: 70.2584
    Epoch [3/50], Val Losses: mse: 5.7189, mae: 1.2577, huber: 0.9096, swd: 1.2575, ept: 68.5901
    Epoch [3/50], Test Losses: mse: 8.0274, mae: 1.4754, huber: 1.1179, swd: 1.6863, ept: 58.4061
      Epoch 3 composite train-obj: 0.861073
            Val objective improved 0.9174 → 0.9096, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.5794, mae: 1.1906, huber: 0.8435, swd: 1.2433, ept: 70.9273
    Epoch [4/50], Val Losses: mse: 5.8819, mae: 1.2752, huber: 0.9250, swd: 1.3905, ept: 69.4928
    Epoch [4/50], Test Losses: mse: 8.2158, mae: 1.4913, huber: 1.1328, swd: 1.8495, ept: 58.7143
      Epoch 4 composite train-obj: 0.843450
            No improvement (0.9250), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.5003, mae: 1.1758, huber: 0.8302, swd: 1.2140, ept: 71.4992
    Epoch [5/50], Val Losses: mse: 5.8885, mae: 1.2651, huber: 0.9170, swd: 1.1977, ept: 69.6686
    Epoch [5/50], Test Losses: mse: 8.0568, mae: 1.4671, huber: 1.1104, swd: 1.5848, ept: 59.1329
      Epoch 5 composite train-obj: 0.830193
            No improvement (0.9170), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.4297, mae: 1.1642, huber: 0.8196, swd: 1.1870, ept: 71.8658
    Epoch [6/50], Val Losses: mse: 5.7710, mae: 1.2692, huber: 0.9211, swd: 1.3001, ept: 69.8092
    Epoch [6/50], Test Losses: mse: 7.9976, mae: 1.4756, huber: 1.1187, swd: 1.7438, ept: 59.1865
      Epoch 6 composite train-obj: 0.819636
            No improvement (0.9211), counter 3/5
    Epoch [7/50], Train Losses: mse: 4.3964, mae: 1.1597, huber: 0.8157, swd: 1.1819, ept: 72.1975
    Epoch [7/50], Val Losses: mse: 6.1150, mae: 1.2898, huber: 0.9413, swd: 1.3368, ept: 70.1242
    Epoch [7/50], Test Losses: mse: 8.1550, mae: 1.4918, huber: 1.1349, swd: 1.8912, ept: 59.3079
      Epoch 7 composite train-obj: 0.815683
            No improvement (0.9413), counter 4/5
    Epoch [8/50], Train Losses: mse: 4.3208, mae: 1.1472, huber: 0.8044, swd: 1.1578, ept: 72.4668
    Epoch [8/50], Val Losses: mse: 6.0777, mae: 1.2854, huber: 0.9387, swd: 1.2584, ept: 70.1042
    Epoch [8/50], Test Losses: mse: 7.8841, mae: 1.4488, huber: 1.0941, swd: 1.5234, ept: 60.0283
      Epoch 8 composite train-obj: 0.804356
    Epoch [8/50], Test Losses: mse: 8.0278, mae: 1.4754, huber: 1.1179, swd: 1.6863, ept: 58.3889
    Best round's Test MSE: 8.0274, MAE: 1.4754, SWD: 1.6863
    Best round's Validation MSE: 5.7189, MAE: 1.2577, SWD: 1.2575
    Best round's Test verification MSE : 8.0278, MAE: 1.4754, SWD: 1.6863
    Time taken: 95.21 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred96_20250512_0238)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 8.1527 ± 0.1010
      mae: 1.4783 ± 0.0073
      huber: 1.1202 ± 0.0071
      swd: 1.7269 ± 0.0802
      ept: 58.0163 ± 0.6327
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.7564 ± 0.0266
      mae: 1.2601 ± 0.0038
      huber: 0.9111 ± 0.0033
      swd: 1.2818 ± 0.0630
      ept: 68.6851 ± 0.2183
      count: 53.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 270.60 seconds
    
    Experiment complete: ACL_ettm1_seq96_pred96_20250512_0238
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    


```python

```

##### Ab: ori


```python
importlib.reload(monotonic)
importlib.reload(train_config) 
# utils.reload_modules([modules_to_reload_list])

cfg_ACL_96_196 = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
)
cfg_ACL_96_196.x_to_z_delay.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_196.x_to_z_deri.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_196.z_to_x_main.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_196.z_to_y_main.use_magnitude_for_scale_and_translate = [False, True]
exp_ACL_96_196 = execute_model_evaluation('ettm1', cfg_ACL_96_196, data_mgr, scale=False)
```

    [autoreload of monotonic failed: Traceback (most recent call last):
      File "c:\proj\DL_notebook\study\.venv\Lib\site-packages\IPython\extensions\autoreload.py", line 280, in check
        elif self.deduper_reloader.maybe_reload_module(m):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "c:\proj\DL_notebook\study\.venv\Lib\site-packages\IPython\extensions\deduperreload\deduperreload.py", line 533, in maybe_reload_module
        new_source_code = f.read()
                          ^^^^^^^^
    UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position 15643: illegal multibyte sequence
    ]
    

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 379
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 9.1967, mae: 1.6759, huber: 1.2992, swd: 3.7098, target_std: 6.5129
    Epoch [1/50], Val Losses: mse: 7.3271, mae: 1.4622, huber: 1.1021, swd: 1.6772, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.6751, mae: 1.7309, huber: 1.3606, swd: 2.3179, target_std: 4.7691
      Epoch 1 composite train-obj: 1.299248
            Val objective improved inf → 1.1021, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.0775, mae: 1.4056, huber: 1.0434, swd: 1.8117, target_std: 6.5130
    Epoch [2/50], Val Losses: mse: 7.0212, mae: 1.4319, huber: 1.0744, swd: 1.6701, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.1090, mae: 1.6619, huber: 1.2955, swd: 2.1249, target_std: 4.7691
      Epoch 2 composite train-obj: 1.043393
            Val objective improved 1.1021 → 1.0744, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.8562, mae: 1.3663, huber: 1.0080, swd: 1.6917, target_std: 6.5128
    Epoch [3/50], Val Losses: mse: 7.1266, mae: 1.4421, huber: 1.0840, swd: 1.6552, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.1531, mae: 1.6629, huber: 1.2953, swd: 2.0690, target_std: 4.7691
      Epoch 3 composite train-obj: 1.007951
            No improvement (1.0840), counter 1/5
    Epoch [4/50], Train Losses: mse: 5.7266, mae: 1.3464, huber: 0.9901, swd: 1.6358, target_std: 6.5130
    Epoch [4/50], Val Losses: mse: 7.2003, mae: 1.4834, huber: 1.1230, swd: 1.9901, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 9.9929, mae: 1.6821, huber: 1.3134, swd: 2.3785, target_std: 4.7691
      Epoch 4 composite train-obj: 0.990057
            No improvement (1.1230), counter 2/5
    Epoch [5/50], Train Losses: mse: 5.6372, mae: 1.3327, huber: 0.9776, swd: 1.6052, target_std: 6.5127
    Epoch [5/50], Val Losses: mse: 7.5095, mae: 1.4970, huber: 1.1389, swd: 1.9888, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.0794, mae: 1.6663, huber: 1.3002, swd: 2.1717, target_std: 4.7691
      Epoch 5 composite train-obj: 0.977646
            No improvement (1.1389), counter 3/5
    Epoch [6/50], Train Losses: mse: 5.5476, mae: 1.3206, huber: 0.9665, swd: 1.5690, target_std: 6.5129
    Epoch [6/50], Val Losses: mse: 7.5413, mae: 1.4946, huber: 1.1371, swd: 2.0351, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 9.9558, mae: 1.6614, huber: 1.2947, swd: 2.1966, target_std: 4.7691
      Epoch 6 composite train-obj: 0.966497
            No improvement (1.1371), counter 4/5
    Epoch [7/50], Train Losses: mse: 5.4862, mae: 1.3115, huber: 0.9583, swd: 1.5470, target_std: 6.5131
    Epoch [7/50], Val Losses: mse: 7.7030, mae: 1.4853, huber: 1.1297, swd: 1.9980, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.1164, mae: 1.6613, huber: 1.2952, swd: 2.1200, target_std: 4.7691
      Epoch 7 composite train-obj: 0.958307
    Epoch [7/50], Test Losses: mse: 10.1090, mae: 1.6619, huber: 1.2955, swd: 2.1251, target_std: 4.7691
    Best round's Test MSE: 10.1090, MAE: 1.6619, SWD: 2.1249
    Best round's Validation MSE: 7.0212, MAE: 1.4319
    Best round's Test verification MSE : 10.1090, MAE: 1.6619, SWD: 2.1251
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.1402, mae: 1.6750, huber: 1.2983, swd: 3.7976, target_std: 6.5129
    Epoch [1/50], Val Losses: mse: 7.2101, mae: 1.4395, huber: 1.0791, swd: 1.4332, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.8556, mae: 1.7380, huber: 1.3653, swd: 2.0181, target_std: 4.7691
      Epoch 1 composite train-obj: 1.298301
            Val objective improved inf → 1.0791, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.0946, mae: 1.4091, huber: 1.0463, swd: 1.8454, target_std: 6.5129
    Epoch [2/50], Val Losses: mse: 7.1383, mae: 1.4421, huber: 1.0850, swd: 1.7140, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.0011, mae: 1.6571, huber: 1.2902, swd: 1.9902, target_std: 4.7691
      Epoch 2 composite train-obj: 1.046319
            No improvement (1.0850), counter 1/5
    Epoch [3/50], Train Losses: mse: 5.8760, mae: 1.3717, huber: 1.0127, swd: 1.7434, target_std: 6.5129
    Epoch [3/50], Val Losses: mse: 7.3012, mae: 1.4644, huber: 1.1065, swd: 2.0002, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.0674, mae: 1.6760, huber: 1.3086, swd: 2.3324, target_std: 4.7691
      Epoch 3 composite train-obj: 1.012734
            No improvement (1.1065), counter 2/5
    Epoch [4/50], Train Losses: mse: 5.7470, mae: 1.3514, huber: 0.9946, swd: 1.6908, target_std: 6.5130
    Epoch [4/50], Val Losses: mse: 7.5309, mae: 1.4723, huber: 1.1149, swd: 1.9023, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.1703, mae: 1.6627, huber: 1.2954, swd: 2.0071, target_std: 4.7691
      Epoch 4 composite train-obj: 0.994572
            No improvement (1.1149), counter 3/5
    Epoch [5/50], Train Losses: mse: 5.6599, mae: 1.3385, huber: 0.9829, swd: 1.6625, target_std: 6.5130
    Epoch [5/50], Val Losses: mse: 7.5674, mae: 1.4873, huber: 1.1300, swd: 2.1955, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.0272, mae: 1.6683, huber: 1.3017, swd: 2.2962, target_std: 4.7691
      Epoch 5 composite train-obj: 0.982920
            No improvement (1.1300), counter 4/5
    Epoch [6/50], Train Losses: mse: 5.5654, mae: 1.3245, huber: 0.9701, swd: 1.6173, target_std: 6.5128
    Epoch [6/50], Val Losses: mse: 7.9101, mae: 1.5107, huber: 1.1532, swd: 2.3404, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.1354, mae: 1.6654, huber: 1.2986, swd: 2.1417, target_std: 4.7691
      Epoch 6 composite train-obj: 0.970050
    Epoch [6/50], Test Losses: mse: 10.8568, mae: 1.7381, huber: 1.3654, swd: 2.0179, target_std: 4.7691
    Best round's Test MSE: 10.8556, MAE: 1.7380, SWD: 2.0181
    Best round's Validation MSE: 7.2101, MAE: 1.4395
    Best round's Test verification MSE : 10.8568, MAE: 1.7381, SWD: 2.0179
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.3114, mae: 1.6831, huber: 1.3060, swd: 3.5544, target_std: 6.5130
    Epoch [1/50], Val Losses: mse: 7.1618, mae: 1.4502, huber: 1.0902, swd: 1.5269, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.5014, mae: 1.7166, huber: 1.3457, swd: 2.0357, target_std: 4.7691
      Epoch 1 composite train-obj: 1.306035
            Val objective improved inf → 1.0902, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.0832, mae: 1.4088, huber: 1.0460, swd: 1.6573, target_std: 6.5130
    Epoch [2/50], Val Losses: mse: 7.0211, mae: 1.4271, huber: 1.0698, swd: 1.4363, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.1630, mae: 1.6653, huber: 1.2983, swd: 1.8819, target_std: 4.7691
      Epoch 2 composite train-obj: 1.046001
            Val objective improved 1.0902 → 1.0698, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.8534, mae: 1.3666, huber: 1.0080, swd: 1.5429, target_std: 6.5128
    Epoch [3/50], Val Losses: mse: 7.1323, mae: 1.4535, huber: 1.0949, swd: 1.5990, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.1226, mae: 1.6719, huber: 1.3037, swd: 1.9766, target_std: 4.7691
      Epoch 3 composite train-obj: 1.007982
            No improvement (1.0949), counter 1/5
    Epoch [4/50], Train Losses: mse: 5.7108, mae: 1.3446, huber: 0.9881, swd: 1.4853, target_std: 6.5131
    Epoch [4/50], Val Losses: mse: 7.2541, mae: 1.4581, huber: 1.1013, swd: 1.5287, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.1940, mae: 1.6601, huber: 1.2943, swd: 1.9228, target_std: 4.7691
      Epoch 4 composite train-obj: 0.988091
            No improvement (1.1013), counter 2/5
    Epoch [5/50], Train Losses: mse: 5.6200, mae: 1.3306, huber: 0.9755, swd: 1.4474, target_std: 6.5129
    Epoch [5/50], Val Losses: mse: 7.7525, mae: 1.5090, huber: 1.1505, swd: 1.8279, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.3401, mae: 1.6809, huber: 1.3120, swd: 1.9492, target_std: 4.7691
      Epoch 5 composite train-obj: 0.975466
            No improvement (1.1505), counter 3/5
    Epoch [6/50], Train Losses: mse: 5.5523, mae: 1.3220, huber: 0.9677, swd: 1.4291, target_std: 6.5128
    Epoch [6/50], Val Losses: mse: 8.1181, mae: 1.5368, huber: 1.1781, swd: 2.2604, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.1145, mae: 1.6756, huber: 1.3088, swd: 2.1523, target_std: 4.7691
      Epoch 6 composite train-obj: 0.967692
            No improvement (1.1781), counter 4/5
    Epoch [7/50], Train Losses: mse: 5.4597, mae: 1.3086, huber: 0.9554, swd: 1.3898, target_std: 6.5128
    Epoch [7/50], Val Losses: mse: 8.1873, mae: 1.5280, huber: 1.1720, swd: 2.1302, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.1654, mae: 1.6727, huber: 1.3069, swd: 2.0424, target_std: 4.7691
      Epoch 7 composite train-obj: 0.955353
    Epoch [7/50], Test Losses: mse: 10.1632, mae: 1.6654, huber: 1.2984, swd: 1.8820, target_std: 4.7691
    Best round's Test MSE: 10.1630, MAE: 1.6653, SWD: 1.8819
    Best round's Validation MSE: 7.0211, MAE: 1.4271
    Best round's Test verification MSE : 10.1632, MAE: 1.6654, SWD: 1.8820
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred196_20250429_1254)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.3759 ± 0.3400
      mae: 1.6884 ± 0.0351
      huber: 1.3197 ± 0.0323
      swd: 2.0083 ± 0.0994
      target_std: 4.7691 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.0841 ± 0.0891
      mae: 1.4328 ± 0.0051
      huber: 1.0744 ± 0.0038
      swd: 1.5132 ± 0.1109
      target_std: 4.2926 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm1_seq96_pred196_20250429_1254
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

##### Ab: shift_inside_scale


```python
importlib.reload(monotonic)
importlib.reload(train_config) 
# utils.reload_modules([modules_to_reload_list])

cfg_ACL_96_196 = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
)
cfg_ACL_96_196.x_to_z_delay.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_196.x_to_z_deri.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_196.z_to_x_main.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_196.z_to_y_main.use_magnitude_for_scale_and_translate = [False, True]
exp_ACL_96_196 = execute_model_evaluation('ettm1', cfg_ACL_96_196, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 9.1226, mae: 1.6854, huber: 1.3065, swd: 3.5565, target_std: 6.5129
    Epoch [1/50], Val Losses: mse: 7.2711, mae: 1.4555, huber: 1.0941, swd: 1.5469, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.7734, mae: 1.7357, huber: 1.3643, swd: 2.1617, target_std: 4.7691
      Epoch 1 composite train-obj: 1.306490
            Val objective improved inf → 1.0941, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.1255, mae: 1.4157, huber: 1.0521, swd: 1.8137, target_std: 6.5130
    Epoch [2/50], Val Losses: mse: 7.2312, mae: 1.4644, huber: 1.1053, swd: 1.7533, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.2905, mae: 1.6813, huber: 1.3136, swd: 2.1824, target_std: 4.7691
      Epoch 2 composite train-obj: 1.052100
            No improvement (1.1053), counter 1/5
    Epoch [3/50], Train Losses: mse: 5.9005, mae: 1.3783, huber: 1.0182, swd: 1.7037, target_std: 6.5128
    Epoch [3/50], Val Losses: mse: 7.3833, mae: 1.4726, huber: 1.1140, swd: 1.7196, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.2342, mae: 1.6681, huber: 1.3010, swd: 1.9885, target_std: 4.7691
      Epoch 3 composite train-obj: 1.018187
            No improvement (1.1140), counter 2/5
    Epoch [4/50], Train Losses: mse: 5.7565, mae: 1.3547, huber: 0.9971, swd: 1.6361, target_std: 6.5130
    Epoch [4/50], Val Losses: mse: 7.5130, mae: 1.5085, huber: 1.1491, swd: 2.1281, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.0445, mae: 1.6666, huber: 1.2994, swd: 2.1764, target_std: 4.7691
      Epoch 4 composite train-obj: 0.997125
            No improvement (1.1491), counter 3/5
    Epoch [5/50], Train Losses: mse: 5.6701, mae: 1.3425, huber: 0.9860, swd: 1.6115, target_std: 6.5127
    Epoch [5/50], Val Losses: mse: 7.8962, mae: 1.5503, huber: 1.1898, swd: 2.4515, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.2058, mae: 1.6885, huber: 1.3203, swd: 2.2743, target_std: 4.7691
      Epoch 5 composite train-obj: 0.986035
            No improvement (1.1898), counter 4/5
    Epoch [6/50], Train Losses: mse: 5.5603, mae: 1.3268, huber: 0.9717, swd: 1.5665, target_std: 6.5129
    Epoch [6/50], Val Losses: mse: 7.9500, mae: 1.5348, huber: 1.1769, swd: 2.3531, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.1325, mae: 1.6710, huber: 1.3049, swd: 2.1920, target_std: 4.7691
      Epoch 6 composite train-obj: 0.971672
    Epoch [6/50], Test Losses: mse: 10.7735, mae: 1.7357, huber: 1.3643, swd: 2.1618, target_std: 4.7691
    Best round's Test MSE: 10.7734, MAE: 1.7357, SWD: 2.1617
    Best round's Validation MSE: 7.2711, MAE: 1.4555
    Best round's Test verification MSE : 10.7735, MAE: 1.7357, SWD: 2.1618
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.9922, mae: 1.6736, huber: 1.2955, swd: 3.6909, target_std: 6.5129
    Epoch [1/50], Val Losses: mse: 7.1477, mae: 1.4310, huber: 1.0725, swd: 1.4565, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.6321, mae: 1.7071, huber: 1.3377, swd: 1.9952, target_std: 4.7691
      Epoch 1 composite train-obj: 1.295547
            Val objective improved inf → 1.0725, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.1072, mae: 1.4116, huber: 1.0485, swd: 1.8342, target_std: 6.5129
    Epoch [2/50], Val Losses: mse: 7.1209, mae: 1.4401, huber: 1.0818, swd: 1.5925, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.1942, mae: 1.6659, huber: 1.2987, swd: 2.0091, target_std: 4.7691
      Epoch 2 composite train-obj: 1.048516
            No improvement (1.0818), counter 1/5
    Epoch [3/50], Train Losses: mse: 5.8945, mae: 1.3777, huber: 1.0177, swd: 1.7456, target_std: 6.5129
    Epoch [3/50], Val Losses: mse: 7.3214, mae: 1.4586, huber: 1.0998, swd: 1.7382, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.3149, mae: 1.6763, huber: 1.3081, swd: 2.1295, target_std: 4.7691
      Epoch 3 composite train-obj: 1.017676
            No improvement (1.0998), counter 2/5
    Epoch [4/50], Train Losses: mse: 5.7557, mae: 1.3567, huber: 0.9987, swd: 1.6867, target_std: 6.5130
    Epoch [4/50], Val Losses: mse: 7.3966, mae: 1.4686, huber: 1.1103, swd: 1.7856, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.1406, mae: 1.6558, huber: 1.2887, swd: 1.9364, target_std: 4.7691
      Epoch 4 composite train-obj: 0.998651
            No improvement (1.1103), counter 3/5
    Epoch [5/50], Train Losses: mse: 5.6701, mae: 1.3438, huber: 0.9870, swd: 1.6615, target_std: 6.5130
    Epoch [5/50], Val Losses: mse: 7.5860, mae: 1.5068, huber: 1.1462, swd: 2.2280, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.1952, mae: 1.7014, huber: 1.3322, swd: 2.5023, target_std: 4.7691
      Epoch 5 composite train-obj: 0.987043
            No improvement (1.1462), counter 4/5
    Epoch [6/50], Train Losses: mse: 5.5543, mae: 1.3267, huber: 0.9714, swd: 1.6022, target_std: 6.5128
    Epoch [6/50], Val Losses: mse: 7.7859, mae: 1.5091, huber: 1.1509, swd: 2.1654, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.2795, mae: 1.6758, huber: 1.3085, swd: 2.1681, target_std: 4.7691
      Epoch 6 composite train-obj: 0.971407
    Epoch [6/50], Test Losses: mse: 10.6316, mae: 1.7071, huber: 1.3377, swd: 1.9950, target_std: 4.7691
    Best round's Test MSE: 10.6321, MAE: 1.7071, SWD: 1.9952
    Best round's Validation MSE: 7.1477, MAE: 1.4310
    Best round's Test verification MSE : 10.6316, MAE: 1.7071, SWD: 1.9950
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.8954, mae: 1.6635, huber: 1.2849, swd: 3.2818, target_std: 6.5130
    Epoch [1/50], Val Losses: mse: 7.1712, mae: 1.4400, huber: 1.0796, swd: 1.3637, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.5745, mae: 1.7090, huber: 1.3383, swd: 1.8432, target_std: 4.7691
      Epoch 1 composite train-obj: 1.284916
            Val objective improved inf → 1.0796, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.0803, mae: 1.4082, huber: 1.0451, swd: 1.6276, target_std: 6.5130
    Epoch [2/50], Val Losses: mse: 6.9355, mae: 1.3981, huber: 1.0427, swd: 1.3271, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.2058, mae: 1.6609, huber: 1.2938, swd: 1.8757, target_std: 4.7691
      Epoch 2 composite train-obj: 1.045074
            Val objective improved 1.0796 → 1.0427, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.8610, mae: 1.3703, huber: 1.0109, swd: 1.5405, target_std: 6.5128
    Epoch [3/50], Val Losses: mse: 7.0606, mae: 1.4303, huber: 1.0734, swd: 1.3889, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.1002, mae: 1.6623, huber: 1.2950, swd: 1.8300, target_std: 4.7691
      Epoch 3 composite train-obj: 1.010926
            No improvement (1.0734), counter 1/5
    Epoch [4/50], Train Losses: mse: 5.7133, mae: 1.3488, huber: 0.9915, swd: 1.4882, target_std: 6.5131
    Epoch [4/50], Val Losses: mse: 7.0847, mae: 1.4254, huber: 1.0691, swd: 1.3747, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.1817, mae: 1.6525, huber: 1.2859, swd: 1.8143, target_std: 4.7691
      Epoch 4 composite train-obj: 0.991487
            No improvement (1.0691), counter 2/5
    Epoch [5/50], Train Losses: mse: 5.6036, mae: 1.3325, huber: 0.9767, swd: 1.4387, target_std: 6.5129
    Epoch [5/50], Val Losses: mse: 7.4858, mae: 1.4703, huber: 1.1121, swd: 1.5169, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.2969, mae: 1.6714, huber: 1.3025, swd: 1.8040, target_std: 4.7691
      Epoch 5 composite train-obj: 0.976656
            No improvement (1.1121), counter 3/5
    Epoch [6/50], Train Losses: mse: 5.5167, mae: 1.3215, huber: 0.9667, swd: 1.4141, target_std: 6.5128
    Epoch [6/50], Val Losses: mse: 7.7434, mae: 1.5000, huber: 1.1436, swd: 1.9713, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.0739, mae: 1.6689, huber: 1.3027, swd: 2.1392, target_std: 4.7691
      Epoch 6 composite train-obj: 0.966650
            No improvement (1.1436), counter 4/5
    Epoch [7/50], Train Losses: mse: 5.4167, mae: 1.3081, huber: 0.9543, swd: 1.3768, target_std: 6.5128
    Epoch [7/50], Val Losses: mse: 8.0117, mae: 1.5000, huber: 1.1453, swd: 1.8992, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.2382, mae: 1.6726, huber: 1.3066, swd: 2.0279, target_std: 4.7691
      Epoch 7 composite train-obj: 0.954289
    Epoch [7/50], Test Losses: mse: 10.2056, mae: 1.6608, huber: 1.2938, swd: 1.8754, target_std: 4.7691
    Best round's Test MSE: 10.2058, MAE: 1.6609, SWD: 1.8757
    Best round's Validation MSE: 6.9355, MAE: 1.3981
    Best round's Test verification MSE : 10.2056, MAE: 1.6608, SWD: 1.8754
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred196_20250429_1337)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.5371 ± 0.2413
      mae: 1.7012 ± 0.0308
      huber: 1.3319 ± 0.0291
      swd: 2.0109 ± 0.1173
      target_std: 4.7691 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.1181 ± 0.1386
      mae: 1.4282 ± 0.0235
      huber: 1.0698 ± 0.0211
      swd: 1.4435 ± 0.0902
      target_std: 4.2926 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm1_seq96_pred196_20250429_1337
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 96-336

#### huber 


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    global_std.shape: torch.Size([7])
    Global Std for ettm1: tensor([7.0829, 2.0413, 6.8291, 1.8072, 1.1741, 0.6004, 8.5648],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 378
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 10.0494, mae: 1.7717, huber: 1.3901, swd: 3.8488, ept: 128.8588
    Epoch [1/50], Val Losses: mse: 8.4248, mae: 1.5583, huber: 1.1949, swd: 1.6466, ept: 143.4601
    Epoch [1/50], Test Losses: mse: 11.8428, mae: 1.8267, huber: 1.4520, swd: 2.1670, ept: 112.0409
      Epoch 1 composite train-obj: 1.390103
            Val objective improved inf → 1.1949, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.8254, mae: 1.5076, huber: 1.1387, swd: 1.9322, ept: 150.1205
    Epoch [2/50], Val Losses: mse: 8.4648, mae: 1.5640, huber: 1.2021, swd: 1.8468, ept: 150.0765
    Epoch [2/50], Test Losses: mse: 11.5524, mae: 1.8053, huber: 1.4309, swd: 2.3191, ept: 116.9105
      Epoch 2 composite train-obj: 1.138693
            No improvement (1.2021), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.6125, mae: 1.4752, huber: 1.1091, swd: 1.8351, ept: 155.6708
    Epoch [3/50], Val Losses: mse: 8.6457, mae: 1.6004, huber: 1.2378, swd: 2.0517, ept: 150.6595
    Epoch [3/50], Test Losses: mse: 11.4272, mae: 1.8064, huber: 1.4322, swd: 2.3999, ept: 117.3111
      Epoch 3 composite train-obj: 1.109120
            No improvement (1.2378), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.4426, mae: 1.4517, huber: 1.0877, swd: 1.7425, ept: 158.5620
    Epoch [4/50], Val Losses: mse: 9.2337, mae: 1.6521, huber: 1.2898, swd: 2.2224, ept: 154.1750
    Epoch [4/50], Test Losses: mse: 11.6508, mae: 1.8098, huber: 1.4366, swd: 2.3021, ept: 119.5826
      Epoch 4 composite train-obj: 1.087685
            No improvement (1.2898), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.3271, mae: 1.4363, huber: 1.0735, swd: 1.6965, ept: 160.4255
    Epoch [5/50], Val Losses: mse: 9.8327, mae: 1.7150, huber: 1.3518, swd: 2.5267, ept: 154.4067
    Epoch [5/50], Test Losses: mse: 11.6455, mae: 1.8206, huber: 1.4473, swd: 2.3296, ept: 118.6050
      Epoch 5 composite train-obj: 1.073478
            No improvement (1.3518), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.2049, mae: 1.4212, huber: 1.0593, swd: 1.6380, ept: 161.9904
    Epoch [6/50], Val Losses: mse: 9.7814, mae: 1.7249, huber: 1.3603, swd: 2.7679, ept: 150.6479
    Epoch [6/50], Test Losses: mse: 11.5326, mae: 1.8285, huber: 1.4548, swd: 2.5447, ept: 118.2961
      Epoch 6 composite train-obj: 1.059317
    Epoch [6/50], Test Losses: mse: 11.8429, mae: 1.8267, huber: 1.4520, swd: 2.1670, ept: 112.0348
    Best round's Test MSE: 11.8428, MAE: 1.8267, SWD: 2.1670
    Best round's Validation MSE: 8.4248, MAE: 1.5583, SWD: 1.6466
    Best round's Test verification MSE : 11.8429, MAE: 1.8267, SWD: 2.1670
    Time taken: 72.51 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.9570, mae: 1.7664, huber: 1.3849, swd: 4.0139, ept: 128.3615
    Epoch [1/50], Val Losses: mse: 8.4655, mae: 1.5551, huber: 1.1918, swd: 1.7454, ept: 144.1834
    Epoch [1/50], Test Losses: mse: 12.0466, mae: 1.8415, huber: 1.4661, swd: 2.4161, ept: 111.3227
      Epoch 1 composite train-obj: 1.384912
            Val objective improved inf → 1.1918, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.8602, mae: 1.5128, huber: 1.1432, swd: 2.0362, ept: 149.0074
    Epoch [2/50], Val Losses: mse: 8.4924, mae: 1.5813, huber: 1.2173, swd: 2.0747, ept: 147.0085
    Epoch [2/50], Test Losses: mse: 11.5501, mae: 1.8149, huber: 1.4393, swd: 2.4521, ept: 114.6214
      Epoch 2 composite train-obj: 1.143181
            No improvement (1.2173), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.6417, mae: 1.4816, huber: 1.1145, swd: 1.9259, ept: 153.6905
    Epoch [3/50], Val Losses: mse: 8.9803, mae: 1.6338, huber: 1.2694, swd: 2.2043, ept: 149.6225
    Epoch [3/50], Test Losses: mse: 11.6266, mae: 1.8154, huber: 1.4406, swd: 2.3512, ept: 116.4913
      Epoch 3 composite train-obj: 1.114500
            No improvement (1.2694), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.4933, mae: 1.4608, huber: 1.0956, swd: 1.8472, ept: 156.5315
    Epoch [4/50], Val Losses: mse: 9.5038, mae: 1.7176, huber: 1.3517, swd: 2.9014, ept: 148.3708
    Epoch [4/50], Test Losses: mse: 11.6794, mae: 1.8404, huber: 1.4647, swd: 2.6970, ept: 116.4417
      Epoch 4 composite train-obj: 1.095580
            No improvement (1.3517), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.3829, mae: 1.4463, huber: 1.0823, swd: 1.8004, ept: 158.5854
    Epoch [5/50], Val Losses: mse: 9.3607, mae: 1.6984, huber: 1.3329, swd: 2.7295, ept: 146.8656
    Epoch [5/50], Test Losses: mse: 11.5628, mae: 1.8434, huber: 1.4676, swd: 2.6953, ept: 115.6975
      Epoch 5 composite train-obj: 1.082341
            No improvement (1.3329), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.2313, mae: 1.4254, huber: 1.0630, swd: 1.7157, ept: 160.4913
    Epoch [6/50], Val Losses: mse: 9.8962, mae: 1.7327, huber: 1.3690, swd: 2.9280, ept: 152.6197
    Epoch [6/50], Test Losses: mse: 11.7677, mae: 1.8421, huber: 1.4687, swd: 2.6232, ept: 117.9683
      Epoch 6 composite train-obj: 1.062997
    Epoch [6/50], Test Losses: mse: 12.0467, mae: 1.8415, huber: 1.4661, swd: 2.4162, ept: 111.3119
    Best round's Test MSE: 12.0466, MAE: 1.8415, SWD: 2.4161
    Best round's Validation MSE: 8.4655, MAE: 1.5551, SWD: 1.7454
    Best round's Test verification MSE : 12.0467, MAE: 1.8415, SWD: 2.4162
    Time taken: 72.62 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.8055, mae: 1.7565, huber: 1.3754, swd: 3.6316, ept: 129.1294
    Epoch [1/50], Val Losses: mse: 8.3265, mae: 1.5430, huber: 1.1808, swd: 1.6633, ept: 142.5677
    Epoch [1/50], Test Losses: mse: 11.9930, mae: 1.8435, huber: 1.4687, swd: 2.5138, ept: 110.8781
      Epoch 1 composite train-obj: 1.375424
            Val objective improved inf → 1.1808, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.8250, mae: 1.5082, huber: 1.1391, swd: 1.9238, ept: 150.4758
    Epoch [2/50], Val Losses: mse: 8.4483, mae: 1.5513, huber: 1.1905, swd: 1.7383, ept: 151.7434
    Epoch [2/50], Test Losses: mse: 11.5095, mae: 1.7946, huber: 1.4207, swd: 2.2425, ept: 117.5576
      Epoch 2 composite train-obj: 1.139110
            No improvement (1.1905), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.5737, mae: 1.4708, huber: 1.1049, swd: 1.7836, ept: 155.2167
    Epoch [3/50], Val Losses: mse: 8.9894, mae: 1.6207, huber: 1.2580, swd: 1.9351, ept: 152.5948
    Epoch [3/50], Test Losses: mse: 11.8620, mae: 1.8241, huber: 1.4493, swd: 2.2562, ept: 116.7064
      Epoch 3 composite train-obj: 1.104919
            No improvement (1.2580), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.4044, mae: 1.4471, huber: 1.0832, swd: 1.7039, ept: 157.9341
    Epoch [4/50], Val Losses: mse: 9.1365, mae: 1.6498, huber: 1.2865, swd: 2.1784, ept: 151.6970
    Epoch [4/50], Test Losses: mse: 11.3949, mae: 1.8014, huber: 1.4277, swd: 2.3881, ept: 118.1366
      Epoch 4 composite train-obj: 1.083193
            No improvement (1.2865), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.3159, mae: 1.4377, huber: 1.0745, swd: 1.6831, ept: 159.2772
    Epoch [5/50], Val Losses: mse: 9.5457, mae: 1.7070, huber: 1.3420, swd: 2.6020, ept: 147.2188
    Epoch [5/50], Test Losses: mse: 11.4343, mae: 1.8313, huber: 1.4564, swd: 2.7511, ept: 116.1674
      Epoch 5 composite train-obj: 1.074524
            No improvement (1.3420), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.1624, mae: 1.4177, huber: 1.0558, swd: 1.6111, ept: 161.4234
    Epoch [6/50], Val Losses: mse: 10.0504, mae: 1.7283, huber: 1.3662, swd: 2.7630, ept: 153.4848
    Epoch [6/50], Test Losses: mse: 11.8491, mae: 1.8369, huber: 1.4634, swd: 2.5740, ept: 117.9796
      Epoch 6 composite train-obj: 1.055822
    Epoch [6/50], Test Losses: mse: 11.9933, mae: 1.8436, huber: 1.4687, swd: 2.5138, ept: 110.9102
    Best round's Test MSE: 11.9930, MAE: 1.8435, SWD: 2.5138
    Best round's Validation MSE: 8.3265, MAE: 1.5430, SWD: 1.6633
    Best round's Test verification MSE : 11.9933, MAE: 1.8436, SWD: 2.5138
    Time taken: 74.85 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred336_20250512_0243)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.9608 ± 0.0863
      mae: 1.8372 ± 0.0075
      huber: 1.4623 ± 0.0073
      swd: 2.3656 ± 0.1460
      ept: 111.4139 ± 0.4790
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 8.4056 ± 0.0584
      mae: 1.5522 ± 0.0066
      huber: 1.1892 ± 0.0060
      swd: 1.6851 ± 0.0432
      ept: 143.4037 ± 0.6608
      count: 52.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 220.08 seconds
    
    Experiment complete: ACL_ettm1_seq96_pred336_20250512_0243
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

##### Ab: ori


```python
importlib.reload(monotonic)
importlib.reload(train_config)
# utils.reload_modules([modules_to_reload_list])

cfg_ACL_96_336 = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
)
cfg_ACL_96_336.x_to_z_delay.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_336.x_to_z_deri.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_336.z_to_x_main.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_336.z_to_y_main.use_magnitude_for_scale_and_translate = [False, True]
exp_ACL_96_336 = execute_model_evaluation('ettm1', cfg_ACL_96_336, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 10.4521, mae: 1.8073, huber: 1.4238, swd: 4.1406, target_std: 6.5151
    Epoch [1/50], Val Losses: mse: 8.7350, mae: 1.6190, huber: 1.2506, swd: 1.8918, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.1419, mae: 1.8766, huber: 1.4981, swd: 2.4017, target_std: 4.7640
      Epoch 1 composite train-obj: 1.423826
            Val objective improved inf → 1.2506, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.0019, mae: 1.5370, huber: 1.1653, swd: 1.9937, target_std: 6.5147
    Epoch [2/50], Val Losses: mse: 8.8145, mae: 1.6223, huber: 1.2569, swd: 1.9800, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 11.8561, mae: 1.8377, huber: 1.4611, swd: 2.2783, target_std: 4.7640
      Epoch 2 composite train-obj: 1.165339
            No improvement (1.2569), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.6980, mae: 1.4888, huber: 1.1215, swd: 1.8333, target_std: 6.5148
    Epoch [3/50], Val Losses: mse: 9.1185, mae: 1.6572, huber: 1.2926, swd: 2.2520, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 11.7302, mae: 1.8396, huber: 1.4625, swd: 2.4270, target_std: 4.7640
      Epoch 3 composite train-obj: 1.121498
            No improvement (1.2926), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.5454, mae: 1.4663, huber: 1.1010, swd: 1.7666, target_std: 6.5147
    Epoch [4/50], Val Losses: mse: 9.2749, mae: 1.6656, huber: 1.3027, swd: 2.3125, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 11.6661, mae: 1.8315, huber: 1.4561, swd: 2.4049, target_std: 4.7640
      Epoch 4 composite train-obj: 1.101020
            No improvement (1.3027), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.4063, mae: 1.4476, huber: 1.0838, swd: 1.7040, target_std: 6.5147
    Epoch [5/50], Val Losses: mse: 9.6757, mae: 1.7155, huber: 1.3518, swd: 2.6478, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 11.8222, mae: 1.8421, huber: 1.4672, swd: 2.4106, target_std: 4.7640
      Epoch 5 composite train-obj: 1.083786
            No improvement (1.3518), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.3107, mae: 1.4353, huber: 1.0725, swd: 1.6684, target_std: 6.5148
    Epoch [6/50], Val Losses: mse: 9.3864, mae: 1.6965, huber: 1.3318, swd: 2.5042, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 11.6280, mae: 1.8436, huber: 1.4686, swd: 2.5637, target_std: 4.7640
      Epoch 6 composite train-obj: 1.072536
    Epoch [6/50], Test Losses: mse: 12.1421, mae: 1.8767, huber: 1.4981, swd: 2.4015, target_std: 4.7640
    Best round's Test MSE: 12.1419, MAE: 1.8766, SWD: 2.4017
    Best round's Validation MSE: 8.7350, MAE: 1.6190
    Best round's Test verification MSE : 12.1421, MAE: 1.8767, SWD: 2.4015
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.2172, mae: 1.7924, huber: 1.4093, swd: 4.1281, target_std: 6.5152
    Epoch [1/50], Val Losses: mse: 8.6505, mae: 1.5847, huber: 1.2182, swd: 1.6504, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.4085, mae: 1.8784, huber: 1.4998, swd: 2.3058, target_std: 4.7640
      Epoch 1 composite train-obj: 1.409268
            Val objective improved inf → 1.2182, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.9655, mae: 1.5308, huber: 1.1598, swd: 2.0560, target_std: 6.5147
    Epoch [2/50], Val Losses: mse: 8.5058, mae: 1.5722, huber: 1.2090, swd: 1.7995, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 11.8149, mae: 1.8233, huber: 1.4477, swd: 2.3045, target_std: 4.7640
      Epoch 2 composite train-obj: 1.159794
            Val objective improved 1.2182 → 1.2090, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.7136, mae: 1.4910, huber: 1.1239, swd: 1.9216, target_std: 6.5149
    Epoch [3/50], Val Losses: mse: 8.7152, mae: 1.6300, huber: 1.2636, swd: 2.2939, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 11.7133, mae: 1.8422, huber: 1.4660, swd: 2.7118, target_std: 4.7640
      Epoch 3 composite train-obj: 1.123891
            No improvement (1.2636), counter 1/5
    Epoch [4/50], Train Losses: mse: 6.5423, mae: 1.4659, huber: 1.1009, swd: 1.8380, target_std: 6.5148
    Epoch [4/50], Val Losses: mse: 9.1095, mae: 1.6643, huber: 1.3000, swd: 2.4502, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 11.8204, mae: 1.8432, huber: 1.4669, swd: 2.5919, target_std: 4.7640
      Epoch 4 composite train-obj: 1.100935
            No improvement (1.3000), counter 2/5
    Epoch [5/50], Train Losses: mse: 6.4108, mae: 1.4481, huber: 1.0845, swd: 1.7755, target_std: 6.5149
    Epoch [5/50], Val Losses: mse: 9.4750, mae: 1.6903, huber: 1.3273, swd: 2.5693, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 11.7877, mae: 1.8341, huber: 1.4601, swd: 2.4777, target_std: 4.7640
      Epoch 5 composite train-obj: 1.084501
            No improvement (1.3273), counter 3/5
    Epoch [6/50], Train Losses: mse: 6.2904, mae: 1.4327, huber: 1.0701, swd: 1.7252, target_std: 6.5147
    Epoch [6/50], Val Losses: mse: 10.2896, mae: 1.7689, huber: 1.4066, swd: 3.1945, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 12.0292, mae: 1.8543, huber: 1.4807, swd: 2.6309, target_std: 4.7640
      Epoch 6 composite train-obj: 1.070139
            No improvement (1.4066), counter 4/5
    Epoch [7/50], Train Losses: mse: 6.2237, mae: 1.4243, huber: 1.0624, swd: 1.7215, target_std: 6.5150
    Epoch [7/50], Val Losses: mse: 10.3110, mae: 1.7653, huber: 1.4029, swd: 3.0564, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 11.9370, mae: 1.8507, huber: 1.4771, swd: 2.5707, target_std: 4.7640
      Epoch 7 composite train-obj: 1.062376
    Epoch [7/50], Test Losses: mse: 11.8154, mae: 1.8233, huber: 1.4477, swd: 2.3047, target_std: 4.7640
    Best round's Test MSE: 11.8149, MAE: 1.8233, SWD: 2.3045
    Best round's Validation MSE: 8.5058, MAE: 1.5722
    Best round's Test verification MSE : 11.8154, MAE: 1.8233, SWD: 2.3047
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.9333, mae: 1.7780, huber: 1.3950, swd: 3.8383, target_std: 6.5145
    Epoch [1/50], Val Losses: mse: 8.6336, mae: 1.6170, huber: 1.2493, swd: 2.1041, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 11.9552, mae: 1.8707, huber: 1.4929, swd: 2.7504, target_std: 4.7640
      Epoch 1 composite train-obj: 1.395023
            Val objective improved inf → 1.2493, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.8927, mae: 1.5233, huber: 1.1525, swd: 1.9277, target_std: 6.5146
    Epoch [2/50], Val Losses: mse: 8.7881, mae: 1.6618, huber: 1.2930, swd: 2.4604, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 11.5142, mae: 1.8581, huber: 1.4792, swd: 2.9570, target_std: 4.7640
      Epoch 2 composite train-obj: 1.152513
            No improvement (1.2930), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.6091, mae: 1.4782, huber: 1.1114, swd: 1.7786, target_std: 6.5148
    Epoch [3/50], Val Losses: mse: 9.2486, mae: 1.6781, huber: 1.3127, swd: 2.3592, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 11.6037, mae: 1.8318, huber: 1.4550, swd: 2.4146, target_std: 4.7640
      Epoch 3 composite train-obj: 1.111426
            No improvement (1.3127), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.4503, mae: 1.4563, huber: 1.0916, swd: 1.7174, target_std: 6.5148
    Epoch [4/50], Val Losses: mse: 9.8195, mae: 1.7235, huber: 1.3596, swd: 2.7111, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 11.7085, mae: 1.8377, huber: 1.4620, swd: 2.5320, target_std: 4.7640
      Epoch 4 composite train-obj: 1.091590
            No improvement (1.3596), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.3317, mae: 1.4406, huber: 1.0773, swd: 1.6733, target_std: 6.5150
    Epoch [5/50], Val Losses: mse: 10.6054, mae: 1.8137, huber: 1.4499, swd: 3.4514, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 11.9256, mae: 1.8633, huber: 1.4891, swd: 2.7886, target_std: 4.7640
      Epoch 5 composite train-obj: 1.077257
            No improvement (1.4499), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.2256, mae: 1.4264, huber: 1.0643, swd: 1.6357, target_std: 6.5148
    Epoch [6/50], Val Losses: mse: 10.0932, mae: 1.7344, huber: 1.3716, swd: 2.8710, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 11.9734, mae: 1.8558, huber: 1.4813, swd: 2.7280, target_std: 4.7640
      Epoch 6 composite train-obj: 1.064255
    Epoch [6/50], Test Losses: mse: 11.9552, mae: 1.8707, huber: 1.4930, swd: 2.7505, target_std: 4.7640
    Best round's Test MSE: 11.9552, MAE: 1.8707, SWD: 2.7504
    Best round's Validation MSE: 8.6336, MAE: 1.6170
    Best round's Test verification MSE : 11.9552, MAE: 1.8707, SWD: 2.7505
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred336_20250429_1404)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.9707 ± 0.1339
      mae: 1.8569 ± 0.0239
      huber: 1.4796 ± 0.0226
      swd: 2.4855 ± 0.1915
      target_std: 4.7640 ± 0.0000
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 8.6248 ± 0.0938
      mae: 1.6027 ± 0.0216
      huber: 1.2363 ± 0.0193
      swd: 1.9318 ± 0.1275
      target_std: 4.2820 ± 0.0000
      count: 52.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm1_seq96_pred336_20250429_1404
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

##### Ab: shift_inside_scale


```python
importlib.reload(monotonic)
importlib.reload(train_config)
# utils.reload_modules([modules_to_reload_list])

cfg_ACL_96_336 = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
)
cfg_ACL_96_336.x_to_z_delay.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_336.x_to_z_deri.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_336.z_to_x_main.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_336.z_to_y_main.use_magnitude_for_scale_and_translate = [False, True]
exp_ACL_96_336 = execute_model_evaluation('ettm1', cfg_ACL_96_336, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 10.0549, mae: 1.7902, huber: 1.4058, swd: 3.7585, target_std: 6.5151
    Epoch [1/50], Val Losses: mse: 8.6195, mae: 1.5956, huber: 1.2279, swd: 1.7099, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.1962, mae: 1.8778, huber: 1.4984, swd: 2.2555, target_std: 4.7640
      Epoch 1 composite train-obj: 1.405764
            Val objective improved inf → 1.2279, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.9695, mae: 1.5307, huber: 1.1596, swd: 1.9323, target_std: 6.5147
    Epoch [2/50], Val Losses: mse: 9.3050, mae: 1.6981, huber: 1.3313, swd: 2.4537, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 11.8389, mae: 1.8459, huber: 1.4693, swd: 2.3614, target_std: 4.7640
      Epoch 2 composite train-obj: 1.159581
            No improvement (1.3313), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.7218, mae: 1.4922, huber: 1.1246, swd: 1.8239, target_std: 6.5148
    Epoch [3/50], Val Losses: mse: 9.2077, mae: 1.6641, huber: 1.2997, swd: 2.2450, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 11.9566, mae: 1.8499, huber: 1.4731, swd: 2.3485, target_std: 4.7640
      Epoch 3 composite train-obj: 1.124636
            No improvement (1.2997), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.5701, mae: 1.4709, huber: 1.1052, swd: 1.7639, target_std: 6.5147
    Epoch [4/50], Val Losses: mse: 9.4354, mae: 1.7026, huber: 1.3377, swd: 2.5609, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 11.7390, mae: 1.8475, huber: 1.4711, swd: 2.5033, target_std: 4.7640
      Epoch 4 composite train-obj: 1.105187
            No improvement (1.3377), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.4278, mae: 1.4511, huber: 1.0869, swd: 1.6919, target_std: 6.5147
    Epoch [5/50], Val Losses: mse: 9.9927, mae: 1.7506, huber: 1.3864, swd: 2.9771, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 11.8943, mae: 1.8537, huber: 1.4782, swd: 2.5154, target_std: 4.7640
      Epoch 5 composite train-obj: 1.086914
            No improvement (1.3864), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.3455, mae: 1.4410, huber: 1.0777, swd: 1.6747, target_std: 6.5148
    Epoch [6/50], Val Losses: mse: 9.5898, mae: 1.7080, huber: 1.3445, swd: 2.5759, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 11.6652, mae: 1.8403, huber: 1.4650, swd: 2.4848, target_std: 4.7640
      Epoch 6 composite train-obj: 1.077691
    Epoch [6/50], Test Losses: mse: 12.1975, mae: 1.8779, huber: 1.4984, swd: 2.2557, target_std: 4.7640
    Best round's Test MSE: 12.1962, MAE: 1.8778, SWD: 2.2555
    Best round's Validation MSE: 8.6195, MAE: 1.5956
    Best round's Test verification MSE : 12.1975, MAE: 1.8779, SWD: 2.2557
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.8557, mae: 1.7751, huber: 1.3910, swd: 3.7665, target_std: 6.5152
    Epoch [1/50], Val Losses: mse: 8.8992, mae: 1.6195, huber: 1.2520, swd: 1.8769, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.5481, mae: 1.8932, huber: 1.5140, swd: 2.3700, target_std: 4.7640
      Epoch 1 composite train-obj: 1.391042
            Val objective improved inf → 1.2520, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.9455, mae: 1.5251, huber: 1.1546, swd: 2.0150, target_std: 6.5147
    Epoch [2/50], Val Losses: mse: 8.5996, mae: 1.5913, huber: 1.2272, swd: 2.0069, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 11.8521, mae: 1.8315, huber: 1.4563, swd: 2.3643, target_std: 4.7640
      Epoch 2 composite train-obj: 1.154647
            Val objective improved 1.2520 → 1.2272, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.7135, mae: 1.4893, huber: 1.1222, swd: 1.8983, target_std: 6.5149
    Epoch [3/50], Val Losses: mse: 8.6429, mae: 1.6177, huber: 1.2517, swd: 2.2352, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 11.8025, mae: 1.8459, huber: 1.4696, swd: 2.5766, target_std: 4.7640
      Epoch 3 composite train-obj: 1.122199
            No improvement (1.2517), counter 1/5
    Epoch [4/50], Train Losses: mse: 6.5652, mae: 1.4687, huber: 1.1034, swd: 1.8304, target_std: 6.5148
    Epoch [4/50], Val Losses: mse: 9.2084, mae: 1.6793, huber: 1.3147, swd: 2.6071, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 11.9108, mae: 1.8516, huber: 1.4756, swd: 2.5911, target_std: 4.7640
      Epoch 4 composite train-obj: 1.103401
            No improvement (1.3147), counter 2/5
    Epoch [5/50], Train Losses: mse: 6.4382, mae: 1.4511, huber: 1.0873, swd: 1.7675, target_std: 6.5149
    Epoch [5/50], Val Losses: mse: 9.4798, mae: 1.6870, huber: 1.3232, swd: 2.6006, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 12.0702, mae: 1.8537, huber: 1.4780, swd: 2.4523, target_std: 4.7640
      Epoch 5 composite train-obj: 1.087300
            No improvement (1.3232), counter 3/5
    Epoch [6/50], Train Losses: mse: 6.3407, mae: 1.4390, huber: 1.0760, swd: 1.7370, target_std: 6.5147
    Epoch [6/50], Val Losses: mse: 10.0149, mae: 1.7404, huber: 1.3778, swd: 3.0141, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 11.9897, mae: 1.8486, huber: 1.4745, swd: 2.5701, target_std: 4.7640
      Epoch 6 composite train-obj: 1.075988
            No improvement (1.3778), counter 4/5
    Epoch [7/50], Train Losses: mse: 6.2663, mae: 1.4303, huber: 1.0679, swd: 1.7282, target_std: 6.5150
    Epoch [7/50], Val Losses: mse: 10.1078, mae: 1.7317, huber: 1.3685, swd: 2.8140, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 12.2095, mae: 1.8620, huber: 1.4867, swd: 2.4496, target_std: 4.7640
      Epoch 7 composite train-obj: 1.067905
    Epoch [7/50], Test Losses: mse: 11.8512, mae: 1.8315, huber: 1.4563, swd: 2.3641, target_std: 4.7640
    Best round's Test MSE: 11.8521, MAE: 1.8315, SWD: 2.3643
    Best round's Validation MSE: 8.5996, MAE: 1.5913
    Best round's Test verification MSE : 11.8512, MAE: 1.8315, SWD: 2.3641
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.9453, mae: 1.7863, huber: 1.4012, swd: 3.7279, target_std: 6.5145
    Epoch [1/50], Val Losses: mse: 8.4381, mae: 1.5889, huber: 1.2217, swd: 1.8671, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 11.9244, mae: 1.8611, huber: 1.4834, swd: 2.5072, target_std: 4.7640
      Epoch 1 composite train-obj: 1.401213
            Val objective improved inf → 1.2217, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.9165, mae: 1.5246, huber: 1.1538, swd: 1.9195, target_std: 6.5146
    Epoch [2/50], Val Losses: mse: 8.4865, mae: 1.6008, huber: 1.2349, swd: 2.0914, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 11.5686, mae: 1.8449, huber: 1.4678, swd: 2.8034, target_std: 4.7640
      Epoch 2 composite train-obj: 1.153824
            No improvement (1.2349), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.6569, mae: 1.4841, huber: 1.1170, swd: 1.7918, target_std: 6.5148
    Epoch [3/50], Val Losses: mse: 8.6638, mae: 1.6036, huber: 1.2399, swd: 1.9659, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 11.5507, mae: 1.8207, huber: 1.4438, swd: 2.3510, target_std: 4.7640
      Epoch 3 composite train-obj: 1.117031
            No improvement (1.2399), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.4987, mae: 1.4631, huber: 1.0978, swd: 1.7312, target_std: 6.5148
    Epoch [4/50], Val Losses: mse: 9.6980, mae: 1.7277, huber: 1.3628, swd: 2.6268, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 11.9267, mae: 1.8624, huber: 1.4857, swd: 2.6335, target_std: 4.7640
      Epoch 4 composite train-obj: 1.097806
            No improvement (1.3628), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.3740, mae: 1.4472, huber: 1.0831, swd: 1.6829, target_std: 6.5150
    Epoch [5/50], Val Losses: mse: 10.1797, mae: 1.7777, huber: 1.4136, swd: 3.0664, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 11.9554, mae: 1.8628, huber: 1.4873, swd: 2.7762, target_std: 4.7640
      Epoch 5 composite train-obj: 1.083107
            No improvement (1.4136), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.2542, mae: 1.4317, huber: 1.0688, swd: 1.6337, target_std: 6.5148
    Epoch [6/50], Val Losses: mse: 9.9410, mae: 1.7337, huber: 1.3704, swd: 2.7862, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 12.0059, mae: 1.8574, huber: 1.4822, swd: 2.6796, target_std: 4.7640
      Epoch 6 composite train-obj: 1.068762
    Epoch [6/50], Test Losses: mse: 11.9242, mae: 1.8610, huber: 1.4834, swd: 2.5071, target_std: 4.7640
    Best round's Test MSE: 11.9244, MAE: 1.8611, SWD: 2.5072
    Best round's Validation MSE: 8.4381, MAE: 1.5889
    Best round's Test verification MSE : 11.9242, MAE: 1.8610, SWD: 2.5071
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred336_20250429_1412)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.9909 ± 0.1482
      mae: 1.8568 ± 0.0191
      huber: 1.4794 ± 0.0174
      swd: 2.3757 ± 0.1031
      target_std: 4.7640 ± 0.0000
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 8.5524 ± 0.0812
      mae: 1.5919 ± 0.0028
      huber: 1.2256 ± 0.0028
      swd: 1.8613 ± 0.1214
      target_std: 4.2820 ± 0.0000
      count: 52.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm1_seq96_pred336_20250429_1412
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 96-720

#### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    global_std.shape: torch.Size([7])
    Global Std for ettm1: tensor([7.0829, 2.0413, 6.8291, 1.8072, 1.1741, 0.6004, 8.5648],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 375
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 10.8523, mae: 1.8842, huber: 1.4951, swd: 3.9099, ept: 185.2338
    Epoch [1/50], Val Losses: mse: 11.2874, mae: 1.8200, huber: 1.4495, swd: 2.4875, ept: 193.2639
    Epoch [1/50], Test Losses: mse: 13.0673, mae: 1.9883, huber: 1.6036, swd: 2.3771, ept: 170.1461
      Epoch 1 composite train-obj: 1.495083
            Val objective improved inf → 1.4495, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.6604, mae: 1.6355, huber: 1.2569, swd: 2.0191, ept: 218.8375
    Epoch [2/50], Val Losses: mse: 11.4562, mae: 1.8601, huber: 1.4903, swd: 3.0268, ept: 198.7782
    Epoch [2/50], Test Losses: mse: 12.6311, mae: 1.9743, huber: 1.5898, swd: 2.7621, ept: 178.0884
      Epoch 2 composite train-obj: 1.256916
            No improvement (1.4903), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.3838, mae: 1.6013, huber: 1.2256, swd: 1.8967, ept: 227.4294
    Epoch [3/50], Val Losses: mse: 12.5001, mae: 1.9771, huber: 1.6057, swd: 3.8486, ept: 196.1515
    Epoch [3/50], Test Losses: mse: 12.5322, mae: 1.9867, huber: 1.6031, swd: 2.9615, ept: 175.4436
      Epoch 3 composite train-obj: 1.225565
            No improvement (1.6057), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.1729, mae: 1.5753, huber: 1.2015, swd: 1.7889, ept: 232.0825
    Epoch [4/50], Val Losses: mse: 12.4043, mae: 1.9524, huber: 1.5833, swd: 3.5764, ept: 197.9825
    Epoch [4/50], Test Losses: mse: 12.8416, mae: 1.9928, huber: 1.6082, swd: 2.7648, ept: 173.8320
      Epoch 4 composite train-obj: 1.201468
            No improvement (1.5833), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.0202, mae: 1.5564, huber: 1.1839, swd: 1.7192, ept: 235.6836
    Epoch [5/50], Val Losses: mse: 13.0683, mae: 2.0594, huber: 1.6857, swd: 4.5137, ept: 188.5813
    Epoch [5/50], Test Losses: mse: 12.5164, mae: 2.0169, huber: 1.6316, swd: 3.4328, ept: 171.6012
      Epoch 5 composite train-obj: 1.183907
            No improvement (1.6857), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.9054, mae: 1.5426, huber: 1.1710, swd: 1.6773, ept: 238.6400
    Epoch [6/50], Val Losses: mse: 12.2640, mae: 1.9556, huber: 1.5869, swd: 3.7492, ept: 199.8837
    Epoch [6/50], Test Losses: mse: 12.6596, mae: 1.9977, huber: 1.6146, swd: 3.1112, ept: 175.4330
      Epoch 6 composite train-obj: 1.171036
    Epoch [6/50], Test Losses: mse: 13.0675, mae: 1.9883, huber: 1.6037, swd: 2.3773, ept: 170.1437
    Best round's Test MSE: 13.0673, MAE: 1.9883, SWD: 2.3771
    Best round's Validation MSE: 11.2874, MAE: 1.8200, SWD: 2.4875
    Best round's Test verification MSE : 13.0675, MAE: 1.9883, SWD: 2.3773
    Time taken: 71.77 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.9298, mae: 1.8933, huber: 1.5038, swd: 3.7777, ept: 182.8335
    Epoch [1/50], Val Losses: mse: 11.5354, mae: 1.8366, huber: 1.4646, swd: 2.2600, ept: 189.5404
    Epoch [1/50], Test Losses: mse: 13.4498, mae: 2.0240, huber: 1.6372, swd: 2.3616, ept: 163.2279
      Epoch 1 composite train-obj: 1.503840
            Val objective improved inf → 1.4646, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.6994, mae: 1.6432, huber: 1.2638, swd: 1.9426, ept: 215.0523
    Epoch [2/50], Val Losses: mse: 11.6137, mae: 1.8963, huber: 1.5238, swd: 3.0076, ept: 192.3557
    Epoch [2/50], Test Losses: mse: 12.4908, mae: 1.9727, huber: 1.5874, swd: 2.7039, ept: 175.6991
      Epoch 2 composite train-obj: 1.263773
            No improvement (1.5238), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.4217, mae: 1.6084, huber: 1.2318, swd: 1.8204, ept: 223.8188
    Epoch [3/50], Val Losses: mse: 12.1665, mae: 1.9484, huber: 1.5777, swd: 3.4059, ept: 193.8191
    Epoch [3/50], Test Losses: mse: 12.7262, mae: 1.9970, huber: 1.6119, swd: 2.9351, ept: 174.9794
      Epoch 3 composite train-obj: 1.231781
            No improvement (1.5777), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.2109, mae: 1.5819, huber: 1.2074, swd: 1.7195, ept: 228.7292
    Epoch [4/50], Val Losses: mse: 12.8326, mae: 2.0005, huber: 1.6316, swd: 3.8210, ept: 189.3634
    Epoch [4/50], Test Losses: mse: 12.9663, mae: 2.0073, huber: 1.6233, swd: 2.9118, ept: 170.3649
      Epoch 4 composite train-obj: 1.207397
            No improvement (1.6316), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.0479, mae: 1.5616, huber: 1.1884, swd: 1.6436, ept: 233.0773
    Epoch [5/50], Val Losses: mse: 12.8436, mae: 2.0152, huber: 1.6449, swd: 3.8350, ept: 193.2936
    Epoch [5/50], Test Losses: mse: 12.5475, mae: 1.9920, huber: 1.6083, swd: 2.9807, ept: 173.5307
      Epoch 5 composite train-obj: 1.188417
            No improvement (1.6449), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.9384, mae: 1.5479, huber: 1.1757, swd: 1.6064, ept: 235.9009
    Epoch [6/50], Val Losses: mse: 13.4075, mae: 2.0284, huber: 1.6578, swd: 3.8860, ept: 184.9117
    Epoch [6/50], Test Losses: mse: 13.3126, mae: 2.0303, huber: 1.6443, swd: 2.7147, ept: 165.4695
      Epoch 6 composite train-obj: 1.175740
    Epoch [6/50], Test Losses: mse: 13.4500, mae: 2.0241, huber: 1.6372, swd: 2.3616, ept: 163.2139
    Best round's Test MSE: 13.4498, MAE: 2.0240, SWD: 2.3616
    Best round's Validation MSE: 11.5354, MAE: 1.8366, SWD: 2.2600
    Best round's Test verification MSE : 13.4500, MAE: 2.0241, SWD: 2.3616
    Time taken: 70.26 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.8212, mae: 1.8864, huber: 1.4970, swd: 4.0050, ept: 184.4117
    Epoch [1/50], Val Losses: mse: 10.9081, mae: 1.7844, huber: 1.4135, swd: 2.4391, ept: 193.5254
    Epoch [1/50], Test Losses: mse: 12.9915, mae: 1.9855, huber: 1.6008, swd: 2.7005, ept: 173.6375
      Epoch 1 composite train-obj: 1.496999
            Val objective improved inf → 1.4135, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.6599, mae: 1.6376, huber: 1.2586, swd: 2.1053, ept: 217.5601
    Epoch [2/50], Val Losses: mse: 11.1586, mae: 1.8385, huber: 1.4677, swd: 2.9984, ept: 193.4141
    Epoch [2/50], Test Losses: mse: 12.2659, mae: 1.9450, huber: 1.5607, swd: 2.6851, ept: 179.2590
      Epoch 2 composite train-obj: 1.258638
            No improvement (1.4677), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.3716, mae: 1.6018, huber: 1.2257, swd: 1.9654, ept: 225.8244
    Epoch [3/50], Val Losses: mse: 12.2377, mae: 1.9697, huber: 1.5974, swd: 3.9930, ept: 191.9659
    Epoch [3/50], Test Losses: mse: 12.5549, mae: 1.9967, huber: 1.6120, swd: 3.1666, ept: 175.1832
      Epoch 3 composite train-obj: 1.225693
            No improvement (1.5974), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.1681, mae: 1.5767, huber: 1.2026, swd: 1.8615, ept: 230.5255
    Epoch [4/50], Val Losses: mse: 12.6224, mae: 2.0085, huber: 1.6355, swd: 4.2855, ept: 187.6351
    Epoch [4/50], Test Losses: mse: 12.4409, mae: 1.9980, huber: 1.6127, swd: 3.2412, ept: 174.0102
      Epoch 4 composite train-obj: 1.202583
            No improvement (1.6355), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.0382, mae: 1.5604, huber: 1.1876, swd: 1.8142, ept: 234.4849
    Epoch [5/50], Val Losses: mse: 13.1434, mae: 2.0070, huber: 1.6380, swd: 4.3260, ept: 191.6974
    Epoch [5/50], Test Losses: mse: 12.9966, mae: 2.0143, huber: 1.6295, swd: 3.1765, ept: 170.1283
      Epoch 5 composite train-obj: 1.187569
            No improvement (1.6380), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.9089, mae: 1.5439, huber: 1.1720, swd: 1.7543, ept: 237.7856
    Epoch [6/50], Val Losses: mse: 12.8525, mae: 2.0091, huber: 1.6391, swd: 4.3866, ept: 193.0349
    Epoch [6/50], Test Losses: mse: 12.5675, mae: 1.9947, huber: 1.6110, swd: 3.2441, ept: 174.0305
      Epoch 6 composite train-obj: 1.172033
    Epoch [6/50], Test Losses: mse: 12.9917, mae: 1.9855, huber: 1.6008, swd: 2.7003, ept: 173.6552
    Best round's Test MSE: 12.9915, MAE: 1.9855, SWD: 2.7005
    Best round's Validation MSE: 10.9081, MAE: 1.7844, SWD: 2.4391
    Best round's Test verification MSE : 12.9917, MAE: 1.9855, SWD: 2.7003
    Time taken: 69.03 seconds
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred720_20250512_0251)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 13.1695 ± 0.2006
      mae: 1.9993 ± 0.0175
      huber: 1.6139 ± 0.0165
      swd: 2.4797 ± 0.1562
      ept: 169.0039 ± 4.3258
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 11.2436 ± 0.2579
      mae: 1.8137 ± 0.0218
      huber: 1.4425 ± 0.0214
      swd: 2.3955 ± 0.0979
      ept: 192.1099 ± 1.8200
      count: 49.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 212.64 seconds
    
    Experiment complete: ACL_ettm1_seq96_pred720_20250512_0251
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

##### Ab: ori


```python
importlib.reload(monotonic)
importlib.reload(train_config)
# utils.reload_modules([modules_to_reload_list])

cfg_ACL_96_720 = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
)
cfg_ACL_96_720.x_to_z_delay.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_720.x_to_z_deri.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_720.z_to_x_main.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_720.z_to_y_main.use_magnitude_for_scale_and_translate = [False, True]
exp_ACL_96_720 = execute_model_evaluation('ettm1', cfg_ACL_96_720, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 11.5841, mae: 1.9423, huber: 1.5501, swd: 4.3017, target_std: 6.5153
    Epoch [1/50], Val Losses: mse: 11.4153, mae: 1.8731, huber: 1.4975, swd: 2.7953, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 13.2850, mae: 2.0389, huber: 1.6511, swd: 2.9098, target_std: 4.7646
      Epoch 1 composite train-obj: 1.550056
            Val objective improved inf → 1.4975, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.8123, mae: 1.6613, huber: 1.2800, swd: 2.0420, target_std: 6.5148
    Epoch [2/50], Val Losses: mse: 12.2462, mae: 1.9626, huber: 1.5899, swd: 3.4107, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 12.8896, mae: 2.0183, huber: 1.6309, swd: 2.6960, target_std: 4.7646
      Epoch 2 composite train-obj: 1.280042
            No improvement (1.5899), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.5426, mae: 1.6235, huber: 1.2461, swd: 1.9227, target_std: 6.5151
    Epoch [3/50], Val Losses: mse: 12.2391, mae: 1.9835, huber: 1.6109, swd: 3.8243, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 12.7176, mae: 2.0081, huber: 1.6217, swd: 2.9107, target_std: 4.7646
      Epoch 3 composite train-obj: 1.246087
            No improvement (1.6109), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.3100, mae: 1.5933, huber: 1.2184, swd: 1.8068, target_std: 6.5153
    Epoch [4/50], Val Losses: mse: 12.4312, mae: 1.9812, huber: 1.6097, swd: 3.8206, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 12.9052, mae: 2.0121, huber: 1.6265, swd: 2.9180, target_std: 4.7646
      Epoch 4 composite train-obj: 1.218372
            No improvement (1.6097), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.2012, mae: 1.5796, huber: 1.2057, swd: 1.7696, target_std: 6.5155
    Epoch [5/50], Val Losses: mse: 12.8237, mae: 2.0074, huber: 1.6360, swd: 4.1252, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 12.9451, mae: 2.0192, huber: 1.6343, swd: 3.0592, target_std: 4.7646
      Epoch 5 composite train-obj: 1.205722
            No improvement (1.6360), counter 4/5
    Epoch [6/50], Train Losses: mse: 7.0593, mae: 1.5614, huber: 1.1887, swd: 1.7127, target_std: 6.5152
    Epoch [6/50], Val Losses: mse: 12.2459, mae: 1.9500, huber: 1.5800, swd: 3.5279, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 12.8396, mae: 2.0015, huber: 1.6169, swd: 2.7401, target_std: 4.7646
      Epoch 6 composite train-obj: 1.188744
    Epoch [6/50], Test Losses: mse: 13.2857, mae: 2.0390, huber: 1.6512, swd: 2.9105, target_std: 4.7646
    Best round's Test MSE: 13.2850, MAE: 2.0389, SWD: 2.9098
    Best round's Validation MSE: 11.4153, MAE: 1.8731
    Best round's Test verification MSE : 13.2857, MAE: 2.0390, SWD: 2.9105
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.5119, mae: 1.9387, huber: 1.5467, swd: 4.1097, target_std: 6.5153
    Epoch [1/50], Val Losses: mse: 11.2009, mae: 1.8217, huber: 1.4487, swd: 2.1823, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 13.5706, mae: 2.0409, huber: 1.6532, swd: 2.6108, target_std: 4.7646
      Epoch 1 composite train-obj: 1.546652
            Val objective improved inf → 1.4487, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.8553, mae: 1.6687, huber: 1.2869, swd: 1.9958, target_std: 6.5152
    Epoch [2/50], Val Losses: mse: 11.2624, mae: 1.8541, huber: 1.4822, swd: 2.6305, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 12.7095, mae: 1.9853, huber: 1.5982, swd: 2.4799, target_std: 4.7646
      Epoch 2 composite train-obj: 1.286882
            No improvement (1.4822), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.4814, mae: 1.6185, huber: 1.2411, swd: 1.7906, target_std: 6.5150
    Epoch [3/50], Val Losses: mse: 12.5232, mae: 2.0010, huber: 1.6290, swd: 3.6539, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 13.0049, mae: 2.0270, huber: 1.6400, swd: 2.9070, target_std: 4.7646
      Epoch 3 composite train-obj: 1.241099
            No improvement (1.6290), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.3148, mae: 1.5950, huber: 1.2199, swd: 1.7244, target_std: 6.5153
    Epoch [4/50], Val Losses: mse: 12.4275, mae: 1.9864, huber: 1.6147, swd: 3.5159, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 12.7176, mae: 2.0055, huber: 1.6198, swd: 2.8880, target_std: 4.7646
      Epoch 4 composite train-obj: 1.219890
            No improvement (1.6147), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.1583, mae: 1.5737, huber: 1.2002, swd: 1.6586, target_std: 6.5151
    Epoch [5/50], Val Losses: mse: 12.9550, mae: 2.0170, huber: 1.6472, swd: 3.9379, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 12.9949, mae: 2.0231, huber: 1.6387, swd: 3.1412, target_std: 4.7646
      Epoch 5 composite train-obj: 1.200217
            No improvement (1.6472), counter 4/5
    Epoch [6/50], Train Losses: mse: 7.0476, mae: 1.5603, huber: 1.1878, swd: 1.6254, target_std: 6.5150
    Epoch [6/50], Val Losses: mse: 12.1667, mae: 1.9573, huber: 1.5867, swd: 3.4332, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 12.7629, mae: 2.0104, huber: 1.6259, swd: 3.1465, target_std: 4.7646
      Epoch 6 composite train-obj: 1.187819
    Epoch [6/50], Test Losses: mse: 13.5710, mae: 2.0409, huber: 1.6532, swd: 2.6108, target_std: 4.7646
    Best round's Test MSE: 13.5706, MAE: 2.0409, SWD: 2.6108
    Best round's Validation MSE: 11.2009, MAE: 1.8217
    Best round's Test verification MSE : 13.5710, MAE: 2.0409, SWD: 2.6108
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.3932, mae: 1.9292, huber: 1.5373, swd: 4.4181, target_std: 6.5148
    Epoch [1/50], Val Losses: mse: 10.8997, mae: 1.8213, huber: 1.4470, swd: 2.6836, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 12.9328, mae: 2.0086, huber: 1.6220, swd: 2.9762, target_std: 4.7646
      Epoch 1 composite train-obj: 1.537255
            Val objective improved inf → 1.4470, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.8361, mae: 1.6650, huber: 1.2840, swd: 2.2002, target_std: 6.5150
    Epoch [2/50], Val Losses: mse: 11.6326, mae: 1.8671, huber: 1.4973, swd: 2.9354, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 12.8334, mae: 1.9823, huber: 1.5957, swd: 2.4242, target_std: 4.7646
      Epoch 2 composite train-obj: 1.284026
            No improvement (1.4973), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.4909, mae: 1.6176, huber: 1.2404, swd: 1.9989, target_std: 6.5148
    Epoch [3/50], Val Losses: mse: 12.4492, mae: 1.9960, huber: 1.6227, swd: 4.1001, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 12.7879, mae: 2.0137, huber: 1.6269, swd: 3.1621, target_std: 4.7646
      Epoch 3 composite train-obj: 1.240402
            No improvement (1.6227), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.2685, mae: 1.5885, huber: 1.2136, swd: 1.8791, target_std: 6.5151
    Epoch [4/50], Val Losses: mse: 13.3765, mae: 2.0597, huber: 1.6883, swd: 4.7129, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 12.9557, mae: 2.0324, huber: 1.6464, swd: 3.3582, target_std: 4.7646
      Epoch 4 composite train-obj: 1.213568
            No improvement (1.6883), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.1308, mae: 1.5716, huber: 1.1979, swd: 1.8117, target_std: 6.5153
    Epoch [5/50], Val Losses: mse: 13.3772, mae: 2.0346, huber: 1.6651, swd: 4.6790, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 13.3615, mae: 2.0409, huber: 1.6545, swd: 3.1823, target_std: 4.7646
      Epoch 5 composite train-obj: 1.197931
            No improvement (1.6651), counter 4/5
    Epoch [6/50], Train Losses: mse: 7.0388, mae: 1.5601, huber: 1.1873, swd: 1.7780, target_std: 6.5151
    Epoch [6/50], Val Losses: mse: 13.5514, mae: 2.0544, huber: 1.6859, swd: 4.7836, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 12.8262, mae: 2.0092, huber: 1.6260, swd: 3.2161, target_std: 4.7646
      Epoch 6 composite train-obj: 1.187345
    Epoch [6/50], Test Losses: mse: 12.9328, mae: 2.0086, huber: 1.6220, swd: 2.9762, target_std: 4.7646
    Best round's Test MSE: 12.9328, MAE: 2.0086, SWD: 2.9762
    Best round's Validation MSE: 10.8997, MAE: 1.8213
    Best round's Test verification MSE : 12.9328, MAE: 2.0086, SWD: 2.9762
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred720_20250429_1517)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 13.2628 ± 0.2608
      mae: 2.0295 ± 0.0148
      huber: 1.6421 ± 0.0142
      swd: 2.8323 ± 0.1589
      target_std: 4.7646 ± 0.0000
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 11.1719 ± 0.2115
      mae: 1.8387 ± 0.0243
      huber: 1.4644 ± 0.0234
      swd: 2.5537 ± 0.2666
      target_std: 4.2630 ± 0.0000
      count: 49.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm1_seq96_pred720_20250429_1517
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

##### Ab: shift_inside_scale


```python
importlib.reload(monotonic)
importlib.reload(train_config)
# utils.reload_modules([modules_to_reload_list])

cfg_ACL_96_720 = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
)
cfg_ACL_96_720.x_to_z_delay.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_720.x_to_z_deri.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_720.z_to_x_main.use_magnitude_for_scale_and_translate = [False, True]
cfg_ACL_96_720.z_to_y_main.use_magnitude_for_scale_and_translate = [False, True]
exp_ACL_96_720 = execute_model_evaluation('ettm1', cfg_ACL_96_720, data_mgr, scale=False)
```

    [autoreload of data_generator failed: Traceback (most recent call last):
      File "c:\proj\DL_notebook\study\.venv\Lib\site-packages\IPython\extensions\autoreload.py", line 283, in check
        superreload(m, reload, self.old_objects)
      File "c:\proj\DL_notebook\study\.venv\Lib\site-packages\IPython\extensions\autoreload.py", line 508, in superreload
        update_generic(old_obj, new_obj)
      File "c:\proj\DL_notebook\study\.venv\Lib\site-packages\IPython\extensions\autoreload.py", line 405, in update_generic
        update(a, b)
      File "c:\proj\DL_notebook\study\.venv\Lib\site-packages\IPython\extensions\autoreload.py", line 357, in update_class
        if update_generic(old_obj, new_obj):
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "c:\proj\DL_notebook\study\.venv\Lib\site-packages\IPython\extensions\autoreload.py", line 405, in update_generic
        update(a, b)
      File "c:\proj\DL_notebook\study\.venv\Lib\site-packages\IPython\extensions\autoreload.py", line 317, in update_function
        setattr(old, name, getattr(new, name))
    ValueError: __init__() requires a code object with 0 free vars, not 2147483648001
    ]
    

    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 375
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 11.0593, mae: 1.9157, huber: 1.5225, swd: 4.0291, target_std: 6.5153
    Epoch [1/50], Val Losses: mse: 11.4271, mae: 1.8743, huber: 1.4970, swd: 2.7703, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 13.3376, mae: 2.0442, huber: 1.6552, swd: 2.7896, target_std: 4.7646
      Epoch 1 composite train-obj: 1.522544
            Val objective improved inf → 1.4970, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.7868, mae: 1.6581, huber: 1.2775, swd: 2.0060, target_std: 6.5148
    Epoch [2/50], Val Losses: mse: 12.0442, mae: 1.9359, huber: 1.5641, swd: 3.2345, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 12.6481, mae: 1.9984, huber: 1.6118, swd: 2.5540, target_std: 4.7646
      Epoch 2 composite train-obj: 1.277462
            No improvement (1.5641), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.5157, mae: 1.6203, huber: 1.2431, swd: 1.8897, target_std: 6.5151
    Epoch [3/50], Val Losses: mse: 13.5142, mae: 2.0776, huber: 1.7042, swd: 4.7923, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 12.9634, mae: 2.0369, huber: 1.6501, swd: 3.1871, target_std: 4.7646
      Epoch 3 composite train-obj: 1.243086
            No improvement (1.7042), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.2995, mae: 1.5932, huber: 1.2180, swd: 1.7876, target_std: 6.5153
    Epoch [4/50], Val Losses: mse: 12.9465, mae: 2.0212, huber: 1.6484, swd: 4.3730, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 12.7812, mae: 2.0146, huber: 1.6288, swd: 3.0100, target_std: 4.7646
      Epoch 4 composite train-obj: 1.218034
            No improvement (1.6484), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.2140, mae: 1.5822, huber: 1.2080, swd: 1.7697, target_std: 6.5155
    Epoch [5/50], Val Losses: mse: 13.1760, mae: 2.0252, huber: 1.6543, swd: 4.4476, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 12.8134, mae: 2.0102, huber: 1.6247, swd: 2.9160, target_std: 4.7646
      Epoch 5 composite train-obj: 1.207990
            No improvement (1.6543), counter 4/5
    Epoch [6/50], Train Losses: mse: 7.0925, mae: 1.5668, huber: 1.1937, swd: 1.7280, target_std: 6.5152
    Epoch [6/50], Val Losses: mse: 12.9646, mae: 2.0006, huber: 1.6316, swd: 4.0219, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 12.9514, mae: 2.0140, huber: 1.6290, swd: 2.7594, target_std: 4.7646
      Epoch 6 composite train-obj: 1.193724
    Epoch [6/50], Test Losses: mse: 13.3380, mae: 2.0442, huber: 1.6552, swd: 2.7896, target_std: 4.7646
    Best round's Test MSE: 13.3376, MAE: 2.0442, SWD: 2.7896
    Best round's Validation MSE: 11.4271, MAE: 1.8743
    Best round's Test verification MSE : 13.3380, MAE: 2.0442, SWD: 2.7896
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.2463, mae: 1.9276, huber: 1.5349, swd: 3.8921, target_std: 6.5153
    Epoch [1/50], Val Losses: mse: 11.7615, mae: 1.8639, huber: 1.4895, swd: 2.1974, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 13.9316, mae: 2.0736, huber: 1.6845, swd: 2.3423, target_std: 4.7646
      Epoch 1 composite train-obj: 1.534906
            Val objective improved inf → 1.4895, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.8428, mae: 1.6698, huber: 1.2877, swd: 1.9369, target_std: 6.5152
    Epoch [2/50], Val Losses: mse: 11.6530, mae: 1.8830, huber: 1.5114, swd: 2.7603, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 13.0036, mae: 2.0115, huber: 1.6235, swd: 2.4920, target_std: 4.7646
      Epoch 2 composite train-obj: 1.287699
            No improvement (1.5114), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.4882, mae: 1.6197, huber: 1.2424, swd: 1.7839, target_std: 6.5150
    Epoch [3/50], Val Losses: mse: 12.3074, mae: 1.9817, huber: 1.6086, swd: 3.5383, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 12.8240, mae: 2.0207, huber: 1.6344, swd: 3.0145, target_std: 4.7646
      Epoch 3 composite train-obj: 1.242408
            No improvement (1.6086), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.3158, mae: 1.5978, huber: 1.2222, swd: 1.7204, target_std: 6.5153
    Epoch [4/50], Val Losses: mse: 12.8506, mae: 2.0168, huber: 1.6449, swd: 3.7991, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 12.8566, mae: 2.0170, huber: 1.6315, swd: 2.9395, target_std: 4.7646
      Epoch 4 composite train-obj: 1.222224
            No improvement (1.6449), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.1466, mae: 1.5753, huber: 1.2013, swd: 1.6447, target_std: 6.5151
    Epoch [5/50], Val Losses: mse: 12.7594, mae: 1.9993, huber: 1.6295, swd: 3.7973, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 12.9059, mae: 2.0152, huber: 1.6303, swd: 3.0168, target_std: 4.7646
      Epoch 5 composite train-obj: 1.201258
            No improvement (1.6295), counter 4/5
    Epoch [6/50], Train Losses: mse: 7.0338, mae: 1.5624, huber: 1.1892, swd: 1.6175, target_std: 6.5150
    Epoch [6/50], Val Losses: mse: 12.8245, mae: 1.9987, huber: 1.6290, swd: 3.7254, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 12.8095, mae: 2.0107, huber: 1.6260, swd: 3.0401, target_std: 4.7646
      Epoch 6 composite train-obj: 1.189160
    Epoch [6/50], Test Losses: mse: 13.9317, mae: 2.0737, huber: 1.6845, swd: 2.3427, target_std: 4.7646
    Best round's Test MSE: 13.9316, MAE: 2.0736, SWD: 2.3423
    Best round's Validation MSE: 11.7615, MAE: 1.8639
    Best round's Test verification MSE : 13.9317, MAE: 2.0737, SWD: 2.3427
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.0668, mae: 1.9151, huber: 1.5225, swd: 4.1693, target_std: 6.5148
    Epoch [1/50], Val Losses: mse: 11.0765, mae: 1.8439, huber: 1.4680, swd: 2.8051, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 12.9648, mae: 2.0211, huber: 1.6334, swd: 2.9622, target_std: 4.7646
      Epoch 1 composite train-obj: 1.522487
            Val objective improved inf → 1.4680, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.8042, mae: 1.6621, huber: 1.2812, swd: 2.1425, target_std: 6.5150
    Epoch [2/50], Val Losses: mse: 11.2994, mae: 1.8309, huber: 1.4618, swd: 2.7024, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 12.7435, mae: 1.9737, huber: 1.5878, swd: 2.2910, target_std: 4.7646
      Epoch 2 composite train-obj: 1.281211
            Val objective improved 1.4680 → 1.4618, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.4940, mae: 1.6192, huber: 1.2418, swd: 1.9815, target_std: 6.5148
    Epoch [3/50], Val Losses: mse: 11.6157, mae: 1.8799, huber: 1.5093, swd: 3.0886, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 12.4415, mae: 1.9723, huber: 1.5865, swd: 2.6308, target_std: 4.7646
      Epoch 3 composite train-obj: 1.241788
            No improvement (1.5093), counter 1/5
    Epoch [4/50], Train Losses: mse: 7.2773, mae: 1.5919, huber: 1.2166, swd: 1.8813, target_std: 6.5151
    Epoch [4/50], Val Losses: mse: 12.1469, mae: 1.9379, huber: 1.5667, swd: 3.5611, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 12.5636, mae: 1.9936, huber: 1.6078, swd: 2.8945, target_std: 4.7646
      Epoch 4 composite train-obj: 1.216570
            No improvement (1.5667), counter 2/5
    Epoch [5/50], Train Losses: mse: 7.1490, mae: 1.5761, huber: 1.2019, swd: 1.8259, target_std: 6.5153
    Epoch [5/50], Val Losses: mse: 11.8695, mae: 1.9244, huber: 1.5537, swd: 3.5508, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 12.5660, mae: 1.9939, huber: 1.6081, swd: 2.8675, target_std: 4.7646
      Epoch 5 composite train-obj: 1.201933
            No improvement (1.5537), counter 3/5
    Epoch [6/50], Train Losses: mse: 7.0195, mae: 1.5606, huber: 1.1874, swd: 1.7662, target_std: 6.5151
    Epoch [6/50], Val Losses: mse: 13.2875, mae: 2.0357, huber: 1.6666, swd: 4.4234, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 12.7863, mae: 2.0115, huber: 1.6270, swd: 3.0966, target_std: 4.7646
      Epoch 6 composite train-obj: 1.187401
            No improvement (1.6666), counter 4/5
    Epoch [7/50], Train Losses: mse: 6.9270, mae: 1.5488, huber: 1.1765, swd: 1.7371, target_std: 6.5148
    Epoch [7/50], Val Losses: mse: 12.5906, mae: 2.0035, huber: 1.6320, swd: 4.3697, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 12.9641, mae: 2.0364, huber: 1.6512, swd: 3.5558, target_std: 4.7646
      Epoch 7 composite train-obj: 1.176501
    Epoch [7/50], Test Losses: mse: 12.7436, mae: 1.9737, huber: 1.5878, swd: 2.2908, target_std: 4.7646
    Best round's Test MSE: 12.7435, MAE: 1.9737, SWD: 2.2910
    Best round's Validation MSE: 11.2994, MAE: 1.8309
    Best round's Test verification MSE : 12.7436, MAE: 1.9737, SWD: 2.2908
    
    ==================================================
    Experiment Summary (ACL_ettm1_seq96_pred720_20250429_1510)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 13.3376 ± 0.4850
      mae: 2.0305 ± 0.0419
      huber: 1.6425 ± 0.0405
      swd: 2.4743 ± 0.2239
      target_std: 4.7646 ± 0.0000
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 11.4960 ± 0.1948
      mae: 1.8564 ± 0.0185
      huber: 1.4828 ± 0.0152
      swd: 2.5567 ± 0.2556
      target_std: 4.2630 ± 0.0000
      count: 49.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_ettm1_seq96_pred720_20250429_1510
    Model: ACL
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### Time Mixer

#### 96-96


```python
utils.reload_modules([utils])
cfg_time_mixer_96_96 = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm1']['channels'],
    enc_in=data_mgr.datasets['ettm1']['channels'],
    dec_in=data_mgr.datasets['ettm1']['channels'],
    c_out=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_time_mixer_96_96 = execute_model_evaluation('ettm1', cfg_time_mixer_96_96, data_mgr, scale=False)

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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 5.9096, mae: 1.3476, huber: 0.9974, swd: 1.7335, target_std: 6.5114
    Epoch [1/50], Val Losses: mse: 5.3788, mae: 1.1758, huber: 0.8374, swd: 0.9651, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.2804, mae: 1.4123, huber: 1.0664, swd: 1.4350, target_std: 4.7755
      Epoch 1 composite train-obj: 0.997440
            Val objective improved inf → 0.8374, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9707, mae: 1.2226, huber: 0.8806, swd: 1.5098, target_std: 6.5113
    Epoch [2/50], Val Losses: mse: 5.2607, mae: 1.1573, huber: 0.8202, swd: 0.9225, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.1718, mae: 1.3950, huber: 1.0500, swd: 1.4031, target_std: 4.7755
      Epoch 2 composite train-obj: 0.880559
            Val objective improved 0.8374 → 0.8202, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.8567, mae: 1.2045, huber: 0.8637, swd: 1.4786, target_std: 6.5115
    Epoch [3/50], Val Losses: mse: 5.2168, mae: 1.1537, huber: 0.8164, swd: 0.9231, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.0331, mae: 1.3937, huber: 1.0474, swd: 1.4404, target_std: 4.7755
      Epoch 3 composite train-obj: 0.863696
            Val objective improved 0.8202 → 0.8164, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.7914, mae: 1.1939, huber: 0.8537, swd: 1.4595, target_std: 6.5116
    Epoch [4/50], Val Losses: mse: 5.2154, mae: 1.1487, huber: 0.8115, swd: 0.8909, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.0121, mae: 1.3946, huber: 1.0475, swd: 1.4386, target_std: 4.7755
      Epoch 4 composite train-obj: 0.853655
            Val objective improved 0.8164 → 0.8115, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.7415, mae: 1.1848, huber: 0.8451, swd: 1.4396, target_std: 6.5113
    Epoch [5/50], Val Losses: mse: 5.2379, mae: 1.1505, huber: 0.8125, swd: 0.9181, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 7.9726, mae: 1.4040, huber: 1.0554, swd: 1.5288, target_std: 4.7755
      Epoch 5 composite train-obj: 0.845082
            No improvement (0.8125), counter 1/5
    Epoch [6/50], Train Losses: mse: 4.6892, mae: 1.1771, huber: 0.8377, swd: 1.4216, target_std: 6.5109
    Epoch [6/50], Val Losses: mse: 5.2455, mae: 1.1537, huber: 0.8158, swd: 0.8694, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 7.9992, mae: 1.3973, huber: 1.0498, swd: 1.4106, target_std: 4.7755
      Epoch 6 composite train-obj: 0.837679
            No improvement (0.8158), counter 2/5
    Epoch [7/50], Train Losses: mse: 4.6473, mae: 1.1700, huber: 0.8312, swd: 1.4068, target_std: 6.5109
    Epoch [7/50], Val Losses: mse: 5.2710, mae: 1.1531, huber: 0.8155, swd: 0.9567, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.1125, mae: 1.4174, huber: 1.0689, swd: 1.5852, target_std: 4.7755
      Epoch 7 composite train-obj: 0.831242
            No improvement (0.8155), counter 3/5
    Epoch [8/50], Train Losses: mse: 4.6075, mae: 1.1640, huber: 0.8258, swd: 1.3955, target_std: 6.5120
    Epoch [8/50], Val Losses: mse: 5.3308, mae: 1.1577, huber: 0.8201, swd: 0.9650, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.0342, mae: 1.4210, huber: 1.0716, swd: 1.6288, target_std: 4.7755
      Epoch 8 composite train-obj: 0.825754
            No improvement (0.8201), counter 4/5
    Epoch [9/50], Train Losses: mse: 4.5730, mae: 1.1589, huber: 0.8210, swd: 1.3832, target_std: 6.5110
    Epoch [9/50], Val Losses: mse: 5.2538, mae: 1.1535, huber: 0.8160, swd: 0.9379, target_std: 4.2867
    Epoch [9/50], Test Losses: mse: 8.0366, mae: 1.4240, huber: 1.0746, swd: 1.6418, target_std: 4.7755
      Epoch 9 composite train-obj: 0.821049
    Epoch [9/50], Test Losses: mse: 8.0121, mae: 1.3946, huber: 1.0475, swd: 1.4386, target_std: 4.7755
    Best round's Test MSE: 8.0121, MAE: 1.3946, SWD: 1.4386
    Best round's Validation MSE: 5.2154, MAE: 1.1487
    Best round's Test verification MSE : 8.0121, MAE: 1.3946, SWD: 1.4386
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.0512, mae: 1.3576, huber: 1.0064, swd: 1.7571, target_std: 6.5115
    Epoch [1/50], Val Losses: mse: 5.3760, mae: 1.1790, huber: 0.8401, swd: 0.9711, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.3316, mae: 1.4245, huber: 1.0772, swd: 1.5101, target_std: 4.7755
      Epoch 1 composite train-obj: 1.006410
            Val objective improved inf → 0.8401, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9933, mae: 1.2268, huber: 0.8842, swd: 1.4911, target_std: 6.5107
    Epoch [2/50], Val Losses: mse: 5.3240, mae: 1.1689, huber: 0.8297, swd: 0.9244, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.1699, mae: 1.4048, huber: 1.0570, swd: 1.4325, target_std: 4.7755
      Epoch 2 composite train-obj: 0.884179
            Val objective improved 0.8401 → 0.8297, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.8926, mae: 1.2112, huber: 0.8693, swd: 1.4672, target_std: 6.5111
    Epoch [3/50], Val Losses: mse: 5.2844, mae: 1.1727, huber: 0.8324, swd: 1.0634, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.2940, mae: 1.4428, huber: 1.0923, swd: 1.6884, target_std: 4.7755
      Epoch 3 composite train-obj: 0.869312
            No improvement (0.8324), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.8312, mae: 1.2015, huber: 0.8602, swd: 1.4508, target_std: 6.5112
    Epoch [4/50], Val Losses: mse: 5.2718, mae: 1.1564, huber: 0.8185, swd: 0.9238, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.2287, mae: 1.4021, huber: 1.0550, swd: 1.4521, target_std: 4.7755
      Epoch 4 composite train-obj: 0.860202
            Val objective improved 0.8297 → 0.8185, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.7917, mae: 1.1940, huber: 0.8534, swd: 1.4369, target_std: 6.5111
    Epoch [5/50], Val Losses: mse: 5.2265, mae: 1.1499, huber: 0.8120, swd: 0.9228, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.0995, mae: 1.4017, huber: 1.0535, swd: 1.5003, target_std: 4.7755
      Epoch 5 composite train-obj: 0.853370
            Val objective improved 0.8185 → 0.8120, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.7517, mae: 1.1886, huber: 0.8482, swd: 1.4265, target_std: 6.5111
    Epoch [6/50], Val Losses: mse: 5.2689, mae: 1.1543, huber: 0.8159, swd: 0.9486, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.1157, mae: 1.4069, huber: 1.0578, swd: 1.5303, target_std: 4.7755
      Epoch 6 composite train-obj: 0.848248
            No improvement (0.8159), counter 1/5
    Epoch [7/50], Train Losses: mse: 4.7161, mae: 1.1824, huber: 0.8425, swd: 1.4132, target_std: 6.5113
    Epoch [7/50], Val Losses: mse: 5.3329, mae: 1.1602, huber: 0.8213, swd: 0.9055, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.0755, mae: 1.3985, huber: 1.0501, swd: 1.4484, target_std: 4.7755
      Epoch 7 composite train-obj: 0.842522
            No improvement (0.8213), counter 2/5
    Epoch [8/50], Train Losses: mse: 4.6870, mae: 1.1779, huber: 0.8384, swd: 1.4041, target_std: 6.5108
    Epoch [8/50], Val Losses: mse: 5.2427, mae: 1.1504, huber: 0.8123, swd: 0.8890, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.0762, mae: 1.4011, huber: 1.0524, swd: 1.4532, target_std: 4.7755
      Epoch 8 composite train-obj: 0.838360
            No improvement (0.8123), counter 3/5
    Epoch [9/50], Train Losses: mse: 4.6591, mae: 1.1729, huber: 0.8338, swd: 1.3929, target_std: 6.5114
    Epoch [9/50], Val Losses: mse: 5.3282, mae: 1.1685, huber: 0.8285, swd: 1.0444, target_std: 4.2867
    Epoch [9/50], Test Losses: mse: 8.2177, mae: 1.4480, huber: 1.0965, swd: 1.7508, target_std: 4.7755
      Epoch 9 composite train-obj: 0.833766
            No improvement (0.8285), counter 4/5
    Epoch [10/50], Train Losses: mse: 4.6283, mae: 1.1679, huber: 0.8291, swd: 1.3840, target_std: 6.5109
    Epoch [10/50], Val Losses: mse: 5.2781, mae: 1.1593, huber: 0.8202, swd: 0.9516, target_std: 4.2867
    Epoch [10/50], Test Losses: mse: 8.1341, mae: 1.4214, huber: 1.0713, swd: 1.5992, target_std: 4.7755
      Epoch 10 composite train-obj: 0.829148
    Epoch [10/50], Test Losses: mse: 8.0995, mae: 1.4017, huber: 1.0535, swd: 1.5003, target_std: 4.7755
    Best round's Test MSE: 8.0995, MAE: 1.4017, SWD: 1.5003
    Best round's Validation MSE: 5.2265, MAE: 1.1499
    Best round's Test verification MSE : 8.0995, MAE: 1.4017, SWD: 1.5003
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.1533, mae: 1.3629, huber: 1.0125, swd: 1.6783, target_std: 6.5114
    Epoch [1/50], Val Losses: mse: 5.4342, mae: 1.1841, huber: 0.8452, swd: 0.8565, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.3853, mae: 1.4081, huber: 1.0638, swd: 1.2701, target_std: 4.7755
      Epoch 1 composite train-obj: 1.012474
            Val objective improved inf → 0.8452, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.9834, mae: 1.2247, huber: 0.8824, swd: 1.3888, target_std: 6.5111
    Epoch [2/50], Val Losses: mse: 5.2550, mae: 1.1593, huber: 0.8214, swd: 0.9104, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.2491, mae: 1.4161, huber: 1.0688, swd: 1.4323, target_std: 4.7755
      Epoch 2 composite train-obj: 0.882404
            Val objective improved 0.8452 → 0.8214, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.9136, mae: 1.2115, huber: 0.8701, swd: 1.3733, target_std: 6.5109
    Epoch [3/50], Val Losses: mse: 5.2344, mae: 1.1539, huber: 0.8162, swd: 0.8619, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.1828, mae: 1.3931, huber: 1.0474, swd: 1.3136, target_std: 4.7755
      Epoch 3 composite train-obj: 0.870078
            Val objective improved 0.8214 → 0.8162, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.8581, mae: 1.2021, huber: 0.8611, swd: 1.3590, target_std: 6.5110
    Epoch [4/50], Val Losses: mse: 5.2296, mae: 1.1581, huber: 0.8187, swd: 0.9480, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.1390, mae: 1.4179, huber: 1.0679, swd: 1.4853, target_std: 4.7755
      Epoch 4 composite train-obj: 0.861138
            No improvement (0.8187), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.8082, mae: 1.1942, huber: 0.8536, swd: 1.3480, target_std: 6.5110
    Epoch [5/50], Val Losses: mse: 5.2194, mae: 1.1499, huber: 0.8120, swd: 0.8453, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.1145, mae: 1.3967, huber: 1.0493, swd: 1.3367, target_std: 4.7755
      Epoch 5 composite train-obj: 0.853633
            Val objective improved 0.8162 → 0.8120, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.7689, mae: 1.1877, huber: 0.8477, swd: 1.3359, target_std: 6.5106
    Epoch [6/50], Val Losses: mse: 5.1733, mae: 1.1473, huber: 0.8091, swd: 0.8545, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.0311, mae: 1.4028, huber: 1.0541, swd: 1.3748, target_std: 4.7755
      Epoch 6 composite train-obj: 0.847707
            Val objective improved 0.8120 → 0.8091, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.7269, mae: 1.1815, huber: 0.8420, swd: 1.3233, target_std: 6.5105
    Epoch [7/50], Val Losses: mse: 5.2459, mae: 1.1555, huber: 0.8160, swd: 0.9178, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.1037, mae: 1.4343, huber: 1.0825, swd: 1.5522, target_std: 4.7755
      Epoch 7 composite train-obj: 0.842042
            No improvement (0.8160), counter 1/5
    Epoch [8/50], Train Losses: mse: 4.6929, mae: 1.1761, huber: 0.8371, swd: 1.3144, target_std: 6.5113
    Epoch [8/50], Val Losses: mse: 5.2114, mae: 1.1484, huber: 0.8110, swd: 0.8780, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.0438, mae: 1.4031, huber: 1.0554, swd: 1.4125, target_std: 4.7755
      Epoch 8 composite train-obj: 0.837070
            No improvement (0.8110), counter 2/5
    Epoch [9/50], Train Losses: mse: 4.6643, mae: 1.1718, huber: 0.8331, swd: 1.3067, target_std: 6.5110
    Epoch [9/50], Val Losses: mse: 5.2341, mae: 1.1519, huber: 0.8137, swd: 0.9520, target_std: 4.2867
    Epoch [9/50], Test Losses: mse: 8.1161, mae: 1.4287, huber: 1.0788, swd: 1.5815, target_std: 4.7755
      Epoch 9 composite train-obj: 0.833147
            No improvement (0.8137), counter 3/5
    Epoch [10/50], Train Losses: mse: 4.6289, mae: 1.1663, huber: 0.8281, swd: 1.2937, target_std: 6.5107
    Epoch [10/50], Val Losses: mse: 5.2459, mae: 1.1553, huber: 0.8170, swd: 0.8829, target_std: 4.2867
    Epoch [10/50], Test Losses: mse: 8.0959, mae: 1.4149, huber: 1.0668, swd: 1.4581, target_std: 4.7755
      Epoch 10 composite train-obj: 0.828138
            No improvement (0.8170), counter 4/5
    Epoch [11/50], Train Losses: mse: 4.6054, mae: 1.1628, huber: 0.8249, swd: 1.2864, target_std: 6.5110
    Epoch [11/50], Val Losses: mse: 5.1941, mae: 1.1492, huber: 0.8118, swd: 0.8620, target_std: 4.2867
    Epoch [11/50], Test Losses: mse: 8.0164, mae: 1.4041, huber: 1.0568, swd: 1.3904, target_std: 4.7755
      Epoch 11 composite train-obj: 0.824882
    Epoch [11/50], Test Losses: mse: 8.0311, mae: 1.4028, huber: 1.0541, swd: 1.3748, target_std: 4.7755
    Best round's Test MSE: 8.0311, MAE: 1.4028, SWD: 1.3748
    Best round's Validation MSE: 5.1733, MAE: 1.1473
    Best round's Test verification MSE : 8.0311, MAE: 1.4028, SWD: 1.3748
    
    ==================================================
    Experiment Summary (TimeMixer_ettm1_seq96_pred96_20250429_1155)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 8.0476 ± 0.0375
      mae: 1.3997 ± 0.0037
      huber: 1.0517 ± 0.0030
      swd: 1.4379 ± 0.0512
      target_std: 4.7755 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.2051 ± 0.0229
      mae: 1.1486 ± 0.0011
      huber: 0.8108 ± 0.0013
      swd: 0.8894 ± 0.0279
      target_std: 4.2867 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: TimeMixer_ettm1_seq96_pred96_20250429_1155
    Model: TimeMixer
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 96-196


```python
utils.reload_modules([utils])
cfg_time_mixer_96_196 = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm1']['channels'],
    enc_in=data_mgr.datasets['ettm1']['channels'],
    dec_in=data_mgr.datasets['ettm1']['channels'],
    c_out=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_time_mixer_96_196 = execute_model_evaluation('ettm1', cfg_time_mixer_96_196, data_mgr, scale=False)

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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 7.4133, mae: 1.5099, huber: 1.1512, swd: 2.1163, target_std: 6.5127
    Epoch [1/50], Val Losses: mse: 6.5643, mae: 1.3026, huber: 0.9571, swd: 0.9900, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.9314, mae: 1.6319, huber: 1.2774, swd: 1.7135, target_std: 4.7691
      Epoch 1 composite train-obj: 1.151239
            Val objective improved inf → 0.9571, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.3218, mae: 1.3875, huber: 1.0356, swd: 1.9058, target_std: 6.5130
    Epoch [2/50], Val Losses: mse: 6.4357, mae: 1.2867, huber: 0.9416, swd: 0.9393, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.7036, mae: 1.6068, huber: 1.2516, swd: 1.6092, target_std: 4.7691
      Epoch 2 composite train-obj: 1.035648
            Val objective improved 0.9571 → 0.9416, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.2185, mae: 1.3738, huber: 1.0225, swd: 1.8788, target_std: 6.5128
    Epoch [3/50], Val Losses: mse: 6.3111, mae: 1.2784, huber: 0.9329, swd: 1.0108, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.5863, mae: 1.6168, huber: 1.2596, swd: 1.7755, target_std: 4.7691
      Epoch 3 composite train-obj: 1.022534
            Val objective improved 0.9416 → 0.9329, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.1585, mae: 1.3663, huber: 1.0155, swd: 1.8641, target_std: 6.5132
    Epoch [4/50], Val Losses: mse: 6.3068, mae: 1.2741, huber: 0.9295, swd: 0.9675, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.4290, mae: 1.5967, huber: 1.2411, swd: 1.6738, target_std: 4.7691
      Epoch 4 composite train-obj: 1.015491
            Val objective improved 0.9329 → 0.9295, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.1008, mae: 1.3602, huber: 1.0097, swd: 1.8538, target_std: 6.5130
    Epoch [5/50], Val Losses: mse: 6.3062, mae: 1.2810, huber: 0.9352, swd: 1.0481, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.3854, mae: 1.6181, huber: 1.2595, swd: 1.8562, target_std: 4.7691
      Epoch 5 composite train-obj: 1.009684
            No improvement (0.9352), counter 1/5
    Epoch [6/50], Train Losses: mse: 6.0423, mae: 1.3534, huber: 1.0033, swd: 1.8370, target_std: 6.5130
    Epoch [6/50], Val Losses: mse: 6.3596, mae: 1.2870, huber: 0.9413, swd: 1.0613, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.4930, mae: 1.6286, huber: 1.2703, swd: 1.8810, target_std: 4.7691
      Epoch 6 composite train-obj: 1.003323
            No improvement (0.9413), counter 2/5
    Epoch [7/50], Train Losses: mse: 6.0006, mae: 1.3491, huber: 0.9992, swd: 1.8251, target_std: 6.5128
    Epoch [7/50], Val Losses: mse: 6.3351, mae: 1.2752, huber: 0.9308, swd: 0.9604, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.2980, mae: 1.5916, huber: 1.2356, swd: 1.6756, target_std: 4.7691
      Epoch 7 composite train-obj: 0.999241
            No improvement (0.9308), counter 3/5
    Epoch [8/50], Train Losses: mse: 5.9477, mae: 1.3432, huber: 0.9937, swd: 1.8086, target_std: 6.5128
    Epoch [8/50], Val Losses: mse: 6.3239, mae: 1.2814, huber: 0.9357, swd: 1.0046, target_std: 4.2926
    Epoch [8/50], Test Losses: mse: 10.2400, mae: 1.6055, huber: 1.2477, swd: 1.7853, target_std: 4.7691
      Epoch 8 composite train-obj: 0.993658
            No improvement (0.9357), counter 4/5
    Epoch [9/50], Train Losses: mse: 5.9017, mae: 1.3390, huber: 0.9897, swd: 1.7950, target_std: 6.5128
    Epoch [9/50], Val Losses: mse: 6.3145, mae: 1.2816, huber: 0.9360, swd: 0.9549, target_std: 4.2926
    Epoch [9/50], Test Losses: mse: 10.1686, mae: 1.5910, huber: 1.2347, swd: 1.6422, target_std: 4.7691
      Epoch 9 composite train-obj: 0.989716
    Epoch [9/50], Test Losses: mse: 10.4290, mae: 1.5967, huber: 1.2411, swd: 1.6738, target_std: 4.7691
    Best round's Test MSE: 10.4290, MAE: 1.5967, SWD: 1.6738
    Best round's Validation MSE: 6.3068, MAE: 1.2741
    Best round's Test verification MSE : 10.4290, MAE: 1.5967, SWD: 1.6738
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.2436, mae: 1.5024, huber: 1.1441, swd: 2.2410, target_std: 6.5127
    Epoch [1/50], Val Losses: mse: 6.3833, mae: 1.2935, huber: 0.9475, swd: 1.0638, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.7674, mae: 1.6491, huber: 1.2912, swd: 1.9369, target_std: 4.7691
      Epoch 1 composite train-obj: 1.144057
            Val objective improved inf → 0.9475, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.2842, mae: 1.3832, huber: 1.0318, swd: 1.9433, target_std: 6.5130
    Epoch [2/50], Val Losses: mse: 6.3168, mae: 1.2781, huber: 0.9333, swd: 0.9996, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.5647, mae: 1.6106, huber: 1.2553, swd: 1.7328, target_std: 4.7691
      Epoch 2 composite train-obj: 1.031842
            Val objective improved 0.9475 → 0.9333, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.1565, mae: 1.3663, huber: 1.0161, swd: 1.9022, target_std: 6.5128
    Epoch [3/50], Val Losses: mse: 6.2734, mae: 1.2736, huber: 0.9285, swd: 1.0374, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.4588, mae: 1.6247, huber: 1.2672, swd: 1.8772, target_std: 4.7691
      Epoch 3 composite train-obj: 1.016068
            Val objective improved 0.9333 → 0.9285, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.0743, mae: 1.3568, huber: 1.0071, swd: 1.8795, target_std: 6.5131
    Epoch [4/50], Val Losses: mse: 6.2420, mae: 1.2712, huber: 0.9268, swd: 1.0327, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.3167, mae: 1.6124, huber: 1.2561, swd: 1.8699, target_std: 4.7691
      Epoch 4 composite train-obj: 1.007071
            Val objective improved 0.9285 → 0.9268, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.0082, mae: 1.3496, huber: 1.0003, swd: 1.8615, target_std: 6.5128
    Epoch [5/50], Val Losses: mse: 6.2485, mae: 1.2720, huber: 0.9268, swd: 0.9907, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.2422, mae: 1.6028, huber: 1.2462, swd: 1.7666, target_std: 4.7691
      Epoch 5 composite train-obj: 1.000318
            Val objective improved 0.9268 → 0.9268, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.9542, mae: 1.3435, huber: 0.9944, swd: 1.8449, target_std: 6.5128
    Epoch [6/50], Val Losses: mse: 6.2811, mae: 1.2758, huber: 0.9309, swd: 1.0394, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.2342, mae: 1.6098, huber: 1.2528, swd: 1.8629, target_std: 4.7691
      Epoch 6 composite train-obj: 0.994429
            No improvement (0.9309), counter 1/5
    Epoch [7/50], Train Losses: mse: 5.8922, mae: 1.3371, huber: 0.9882, swd: 1.8224, target_std: 6.5130
    Epoch [7/50], Val Losses: mse: 6.2438, mae: 1.2720, huber: 0.9275, swd: 0.9902, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.1461, mae: 1.6032, huber: 1.2462, swd: 1.8043, target_std: 4.7691
      Epoch 7 composite train-obj: 0.988228
            No improvement (0.9275), counter 2/5
    Epoch [8/50], Train Losses: mse: 5.8461, mae: 1.3328, huber: 0.9840, swd: 1.8095, target_std: 6.5129
    Epoch [8/50], Val Losses: mse: 6.1856, mae: 1.2718, huber: 0.9268, swd: 1.0018, target_std: 4.2926
    Epoch [8/50], Test Losses: mse: 10.0914, mae: 1.6061, huber: 1.2488, swd: 1.8384, target_std: 4.7691
      Epoch 8 composite train-obj: 0.983971
            No improvement (0.9268), counter 3/5
    Epoch [9/50], Train Losses: mse: 5.7939, mae: 1.3275, huber: 0.9786, swd: 1.7917, target_std: 6.5130
    Epoch [9/50], Val Losses: mse: 6.2144, mae: 1.2803, huber: 0.9344, swd: 1.0521, target_std: 4.2926
    Epoch [9/50], Test Losses: mse: 10.0732, mae: 1.6067, huber: 1.2493, swd: 1.8705, target_std: 4.7691
      Epoch 9 composite train-obj: 0.978638
            No improvement (0.9344), counter 4/5
    Epoch [10/50], Train Losses: mse: 5.7523, mae: 1.3232, huber: 0.9743, swd: 1.7788, target_std: 6.5128
    Epoch [10/50], Val Losses: mse: 6.1663, mae: 1.2726, huber: 0.9264, swd: 1.0242, target_std: 4.2926
    Epoch [10/50], Test Losses: mse: 10.1548, mae: 1.6274, huber: 1.2675, swd: 1.9418, target_std: 4.7691
      Epoch 10 composite train-obj: 0.974335
            Val objective improved 0.9268 → 0.9264, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 5.7142, mae: 1.3194, huber: 0.9705, swd: 1.7660, target_std: 6.5131
    Epoch [11/50], Val Losses: mse: 6.2790, mae: 1.2845, huber: 0.9380, swd: 1.0756, target_std: 4.2926
    Epoch [11/50], Test Losses: mse: 10.1584, mae: 1.6190, huber: 1.2603, swd: 1.9401, target_std: 4.7691
      Epoch 11 composite train-obj: 0.970536
            No improvement (0.9380), counter 1/5
    Epoch [12/50], Train Losses: mse: 5.6750, mae: 1.3148, huber: 0.9660, swd: 1.7491, target_std: 6.5131
    Epoch [12/50], Val Losses: mse: 6.2393, mae: 1.2824, huber: 0.9361, swd: 1.0453, target_std: 4.2926
    Epoch [12/50], Test Losses: mse: 10.1127, mae: 1.6143, huber: 1.2559, swd: 1.8834, target_std: 4.7691
      Epoch 12 composite train-obj: 0.966042
            No improvement (0.9361), counter 2/5
    Epoch [13/50], Train Losses: mse: 5.6379, mae: 1.3110, huber: 0.9623, swd: 1.7371, target_std: 6.5128
    Epoch [13/50], Val Losses: mse: 6.2694, mae: 1.2861, huber: 0.9393, swd: 1.0045, target_std: 4.2926
    Epoch [13/50], Test Losses: mse: 10.0719, mae: 1.6121, huber: 1.2531, swd: 1.8136, target_std: 4.7691
      Epoch 13 composite train-obj: 0.962343
            No improvement (0.9393), counter 3/5
    Epoch [14/50], Train Losses: mse: 5.6074, mae: 1.3077, huber: 0.9590, swd: 1.7267, target_std: 6.5128
    Epoch [14/50], Val Losses: mse: 6.2749, mae: 1.2876, huber: 0.9407, swd: 1.0061, target_std: 4.2926
    Epoch [14/50], Test Losses: mse: 10.1484, mae: 1.6169, huber: 1.2577, swd: 1.8236, target_std: 4.7691
      Epoch 14 composite train-obj: 0.959022
            No improvement (0.9407), counter 4/5
    Epoch [15/50], Train Losses: mse: 5.5771, mae: 1.3038, huber: 0.9553, swd: 1.7118, target_std: 6.5129
    Epoch [15/50], Val Losses: mse: 6.2945, mae: 1.2888, huber: 0.9420, swd: 1.0180, target_std: 4.2926
    Epoch [15/50], Test Losses: mse: 10.1776, mae: 1.6166, huber: 1.2576, swd: 1.8347, target_std: 4.7691
      Epoch 15 composite train-obj: 0.955315
    Epoch [15/50], Test Losses: mse: 10.1548, mae: 1.6274, huber: 1.2675, swd: 1.9418, target_std: 4.7691
    Best round's Test MSE: 10.1548, MAE: 1.6274, SWD: 1.9418
    Best round's Validation MSE: 6.1663, MAE: 1.2726
    Best round's Test verification MSE : 10.1548, MAE: 1.6274, SWD: 1.9418
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.0311, mae: 1.4800, huber: 1.1230, swd: 1.9499, target_std: 6.5128
    Epoch [1/50], Val Losses: mse: 6.5104, mae: 1.2977, huber: 0.9520, swd: 0.9239, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.8295, mae: 1.6255, huber: 1.2702, swd: 1.6054, target_std: 4.7691
      Epoch 1 composite train-obj: 1.122960
            Val objective improved inf → 0.9520, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.2760, mae: 1.3827, huber: 1.0311, swd: 1.7084, target_std: 6.5131
    Epoch [2/50], Val Losses: mse: 6.3376, mae: 1.2790, huber: 0.9340, swd: 0.9012, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.5026, mae: 1.6051, huber: 1.2495, swd: 1.5984, target_std: 4.7691
      Epoch 2 composite train-obj: 1.031147
            Val objective improved 0.9520 → 0.9340, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.1473, mae: 1.3655, huber: 1.0152, swd: 1.6725, target_std: 6.5129
    Epoch [3/50], Val Losses: mse: 6.3130, mae: 1.2737, huber: 0.9293, swd: 0.9076, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.4224, mae: 1.6017, huber: 1.2461, swd: 1.6095, target_std: 4.7691
      Epoch 3 composite train-obj: 1.015205
            Val objective improved 0.9340 → 0.9293, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.0689, mae: 1.3560, huber: 1.0062, swd: 1.6510, target_std: 6.5130
    Epoch [4/50], Val Losses: mse: 6.3256, mae: 1.2773, huber: 0.9325, swd: 0.9640, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.4130, mae: 1.6137, huber: 1.2568, swd: 1.7370, target_std: 4.7691
      Epoch 4 composite train-obj: 1.006199
            No improvement (0.9325), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.0085, mae: 1.3486, huber: 0.9991, swd: 1.6350, target_std: 6.5130
    Epoch [5/50], Val Losses: mse: 6.3741, mae: 1.2799, huber: 0.9354, swd: 0.9443, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.3166, mae: 1.6019, huber: 1.2455, swd: 1.6707, target_std: 4.7691
      Epoch 5 composite train-obj: 0.999144
            No improvement (0.9354), counter 2/5
    Epoch [6/50], Train Losses: mse: 5.9474, mae: 1.3428, huber: 0.9933, swd: 1.6198, target_std: 6.5130
    Epoch [6/50], Val Losses: mse: 6.4239, mae: 1.2832, huber: 0.9385, swd: 0.9194, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.3620, mae: 1.5980, huber: 1.2420, swd: 1.5940, target_std: 4.7691
      Epoch 6 composite train-obj: 0.993307
            No improvement (0.9385), counter 3/5
    Epoch [7/50], Train Losses: mse: 5.8963, mae: 1.3365, huber: 0.9871, swd: 1.6038, target_std: 6.5130
    Epoch [7/50], Val Losses: mse: 6.3041, mae: 1.2807, huber: 0.9351, swd: 0.9886, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.2735, mae: 1.6153, huber: 1.2568, swd: 1.7620, target_std: 4.7691
      Epoch 7 composite train-obj: 0.987145
            No improvement (0.9351), counter 4/5
    Epoch [8/50], Train Losses: mse: 5.8372, mae: 1.3307, huber: 0.9815, swd: 1.5876, target_std: 6.5129
    Epoch [8/50], Val Losses: mse: 6.2853, mae: 1.2757, huber: 0.9305, swd: 0.9285, target_std: 4.2926
    Epoch [8/50], Test Losses: mse: 10.1956, mae: 1.6047, huber: 1.2469, swd: 1.6608, target_std: 4.7691
      Epoch 8 composite train-obj: 0.981473
    Epoch [8/50], Test Losses: mse: 10.4224, mae: 1.6017, huber: 1.2461, swd: 1.6095, target_std: 4.7691
    Best round's Test MSE: 10.4224, MAE: 1.6017, SWD: 1.6095
    Best round's Validation MSE: 6.3130, MAE: 1.2737
    Best round's Test verification MSE : 10.4224, MAE: 1.6017, SWD: 1.6095
    
    ==================================================
    Experiment Summary (TimeMixer_ettm1_seq96_pred196_20250429_1307)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.3354 ± 0.1277
      mae: 1.6086 ± 0.0134
      huber: 1.2516 ± 0.0115
      swd: 1.7417 ± 0.1439
      target_std: 4.7691 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 6.2620 ± 0.0677
      mae: 1.2734 ± 0.0006
      huber: 0.9284 ± 0.0014
      swd: 0.9664 ± 0.0476
      target_std: 4.2926 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: TimeMixer_ettm1_seq96_pred196_20250429_1307
    Model: TimeMixer
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 96-336


```python
utils.reload_modules([utils])
cfg_time_mixer_96_336 = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm1']['channels'],
    enc_in=data_mgr.datasets['ettm1']['channels'],
    dec_in=data_mgr.datasets['ettm1']['channels'],
    c_out=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_time_mixer_96_336 = execute_model_evaluation('ettm1', cfg_time_mixer_96_336, data_mgr, scale=False)

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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 8.5008, mae: 1.6434, huber: 1.2757, swd: 2.2703, target_std: 6.5149
    Epoch [1/50], Val Losses: mse: 7.7540, mae: 1.4212, huber: 1.0674, swd: 1.0264, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.9234, mae: 1.8157, huber: 1.4504, swd: 1.9322, target_std: 4.7640
      Epoch 1 composite train-obj: 1.275712
            Val objective improved inf → 1.0674, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.3574, mae: 1.5212, huber: 1.1601, swd: 2.1258, target_std: 6.5149
    Epoch [2/50], Val Losses: mse: 7.5840, mae: 1.4063, huber: 1.0527, swd: 0.9970, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 12.6496, mae: 1.7976, huber: 1.4317, swd: 1.9156, target_std: 4.7640
      Epoch 2 composite train-obj: 1.160134
            Val objective improved 1.0674 → 1.0527, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.2224, mae: 1.5064, huber: 1.1458, swd: 2.0885, target_std: 6.5147
    Epoch [3/50], Val Losses: mse: 7.5773, mae: 1.4025, huber: 1.0491, swd: 0.9830, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 12.5281, mae: 1.7890, huber: 1.4228, swd: 1.8870, target_std: 4.7640
      Epoch 3 composite train-obj: 1.145770
            Val objective improved 1.0527 → 1.0491, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.1293, mae: 1.4973, huber: 1.1370, swd: 2.0672, target_std: 6.5149
    Epoch [4/50], Val Losses: mse: 7.5021, mae: 1.3956, huber: 1.0425, swd: 1.0044, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 12.3367, mae: 1.7865, huber: 1.4197, swd: 1.9637, target_std: 4.7640
      Epoch 4 composite train-obj: 1.136990
            Val objective improved 1.0491 → 1.0425, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.0561, mae: 1.4897, huber: 1.1298, swd: 2.0457, target_std: 6.5149
    Epoch [5/50], Val Losses: mse: 7.4947, mae: 1.3971, huber: 1.0433, swd: 1.0115, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 12.2078, mae: 1.7890, huber: 1.4216, swd: 2.0141, target_std: 4.7640
      Epoch 5 composite train-obj: 1.129786
            No improvement (1.0433), counter 1/5
    Epoch [6/50], Train Losses: mse: 6.9968, mae: 1.4838, huber: 1.1241, swd: 2.0253, target_std: 6.5150
    Epoch [6/50], Val Losses: mse: 7.6462, mae: 1.4062, huber: 1.0529, swd: 0.9820, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 12.2045, mae: 1.7698, huber: 1.4045, swd: 1.8314, target_std: 4.7640
      Epoch 6 composite train-obj: 1.124066
            No improvement (1.0529), counter 2/5
    Epoch [7/50], Train Losses: mse: 6.9401, mae: 1.4783, huber: 1.1189, swd: 2.0056, target_std: 6.5149
    Epoch [7/50], Val Losses: mse: 7.5443, mae: 1.4026, huber: 1.0493, swd: 0.9952, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 12.0334, mae: 1.7692, huber: 1.4036, swd: 1.9118, target_std: 4.7640
      Epoch 7 composite train-obj: 1.118891
            No improvement (1.0493), counter 3/5
    Epoch [8/50], Train Losses: mse: 6.8870, mae: 1.4739, huber: 1.1145, swd: 1.9915, target_std: 6.5148
    Epoch [8/50], Val Losses: mse: 7.7292, mae: 1.4100, huber: 1.0570, swd: 0.9823, target_std: 4.2820
    Epoch [8/50], Test Losses: mse: 12.1608, mae: 1.7643, huber: 1.3992, swd: 1.8104, target_std: 4.7640
      Epoch 8 composite train-obj: 1.114543
            No improvement (1.0570), counter 4/5
    Epoch [9/50], Train Losses: mse: 6.8336, mae: 1.4690, huber: 1.1097, swd: 1.9738, target_std: 6.5149
    Epoch [9/50], Val Losses: mse: 7.5974, mae: 1.4051, huber: 1.0513, swd: 1.0131, target_std: 4.2820
    Epoch [9/50], Test Losses: mse: 11.9898, mae: 1.7688, huber: 1.4022, swd: 1.9037, target_std: 4.7640
      Epoch 9 composite train-obj: 1.109721
    Epoch [9/50], Test Losses: mse: 12.3367, mae: 1.7865, huber: 1.4197, swd: 1.9637, target_std: 4.7640
    Best round's Test MSE: 12.3367, MAE: 1.7865, SWD: 1.9637
    Best round's Validation MSE: 7.5021, MAE: 1.3956
    Best round's Test verification MSE : 12.3367, MAE: 1.7865, SWD: 1.9637
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.0405, mae: 1.5976, huber: 1.2320, swd: 2.3368, target_std: 6.5150
    Epoch [1/50], Val Losses: mse: 7.7624, mae: 1.4162, huber: 1.0628, swd: 1.0259, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.9020, mae: 1.8052, huber: 1.4406, swd: 1.9694, target_std: 4.7640
      Epoch 1 composite train-obj: 1.231958
            Val objective improved inf → 1.0628, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.2930, mae: 1.5125, huber: 1.1518, swd: 2.2026, target_std: 6.5148
    Epoch [2/50], Val Losses: mse: 7.5827, mae: 1.4023, huber: 1.0487, swd: 1.0234, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 12.5799, mae: 1.7925, huber: 1.4267, swd: 2.0003, target_std: 4.7640
      Epoch 2 composite train-obj: 1.151835
            Val objective improved 1.0628 → 1.0487, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.1611, mae: 1.4993, huber: 1.1391, swd: 2.1676, target_std: 6.5147
    Epoch [3/50], Val Losses: mse: 7.5587, mae: 1.3961, huber: 1.0433, swd: 1.0194, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 12.4268, mae: 1.7811, huber: 1.4157, swd: 1.9796, target_std: 4.7640
      Epoch 3 composite train-obj: 1.139120
            Val objective improved 1.0487 → 1.0433, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.0665, mae: 1.4901, huber: 1.1303, swd: 2.1409, target_std: 6.5150
    Epoch [4/50], Val Losses: mse: 7.6144, mae: 1.4015, huber: 1.0487, swd: 1.0095, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 12.3030, mae: 1.7733, huber: 1.4083, swd: 1.9213, target_std: 4.7640
      Epoch 4 composite train-obj: 1.130280
            No improvement (1.0487), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.9909, mae: 1.4827, huber: 1.1231, swd: 2.1147, target_std: 6.5148
    Epoch [5/50], Val Losses: mse: 7.6132, mae: 1.4004, huber: 1.0472, swd: 1.0612, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 12.2197, mae: 1.7823, huber: 1.4155, swd: 2.0577, target_std: 4.7640
      Epoch 5 composite train-obj: 1.123128
            No improvement (1.0472), counter 2/5
    Epoch [6/50], Train Losses: mse: 6.9278, mae: 1.4768, huber: 1.1175, swd: 2.0938, target_std: 6.5147
    Epoch [6/50], Val Losses: mse: 7.6861, mae: 1.4052, huber: 1.0522, swd: 1.0657, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 12.2586, mae: 1.7835, huber: 1.4170, swd: 2.0546, target_std: 4.7640
      Epoch 6 composite train-obj: 1.117456
            No improvement (1.0522), counter 3/5
    Epoch [7/50], Train Losses: mse: 6.8625, mae: 1.4710, huber: 1.1118, swd: 2.0739, target_std: 6.5152
    Epoch [7/50], Val Losses: mse: 7.6604, mae: 1.4041, huber: 1.0511, swd: 1.0883, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 12.1989, mae: 1.7908, huber: 1.4234, swd: 2.1107, target_std: 4.7640
      Epoch 7 composite train-obj: 1.111757
            No improvement (1.0511), counter 4/5
    Epoch [8/50], Train Losses: mse: 6.7950, mae: 1.4643, huber: 1.1054, swd: 2.0476, target_std: 6.5150
    Epoch [8/50], Val Losses: mse: 7.6298, mae: 1.4121, huber: 1.0575, swd: 1.0447, target_std: 4.2820
    Epoch [8/50], Test Losses: mse: 11.9826, mae: 1.7752, huber: 1.4069, swd: 1.9286, target_std: 4.7640
      Epoch 8 composite train-obj: 1.105363
    Epoch [8/50], Test Losses: mse: 12.4268, mae: 1.7811, huber: 1.4157, swd: 1.9796, target_std: 4.7640
    Best round's Test MSE: 12.4268, MAE: 1.7811, SWD: 1.9796
    Best round's Validation MSE: 7.5587, MAE: 1.3961
    Best round's Test verification MSE : 12.4268, MAE: 1.7811, SWD: 1.9796
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.9147, mae: 1.5920, huber: 1.2270, swd: 2.3221, target_std: 6.5152
    Epoch [1/50], Val Losses: mse: 7.7055, mae: 1.4156, huber: 1.0612, swd: 1.0139, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.7737, mae: 1.8025, huber: 1.4365, swd: 1.9992, target_std: 4.7640
      Epoch 1 composite train-obj: 1.227023
            Val objective improved inf → 1.0612, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.2290, mae: 1.5092, huber: 1.1484, swd: 2.0773, target_std: 6.5148
    Epoch [2/50], Val Losses: mse: 7.6708, mae: 1.4063, huber: 1.0536, swd: 1.0667, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 12.5555, mae: 1.7892, huber: 1.4243, swd: 2.0777, target_std: 4.7640
      Epoch 2 composite train-obj: 1.148385
            Val objective improved 1.0612 → 1.0536, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.1046, mae: 1.4959, huber: 1.1357, swd: 2.0448, target_std: 6.5147
    Epoch [3/50], Val Losses: mse: 7.5568, mae: 1.4005, huber: 1.0472, swd: 1.0791, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 12.2719, mae: 1.7859, huber: 1.4194, swd: 2.1438, target_std: 4.7640
      Epoch 3 composite train-obj: 1.135748
            Val objective improved 1.0536 → 1.0472, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.0147, mae: 1.4867, huber: 1.1270, swd: 2.0135, target_std: 6.5145
    Epoch [4/50], Val Losses: mse: 7.5692, mae: 1.4021, huber: 1.0486, swd: 1.0746, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 12.2099, mae: 1.7868, huber: 1.4201, swd: 2.1442, target_std: 4.7640
      Epoch 4 composite train-obj: 1.126982
            No improvement (1.0486), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.9290, mae: 1.4787, huber: 1.1192, swd: 1.9839, target_std: 6.5147
    Epoch [5/50], Val Losses: mse: 7.6303, mae: 1.4064, huber: 1.0528, swd: 1.0337, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 12.0886, mae: 1.7776, huber: 1.4106, swd: 2.0217, target_std: 4.7640
      Epoch 5 composite train-obj: 1.119172
            No improvement (1.0528), counter 2/5
    Epoch [6/50], Train Losses: mse: 6.8495, mae: 1.4714, huber: 1.1119, swd: 1.9568, target_std: 6.5148
    Epoch [6/50], Val Losses: mse: 7.6313, mae: 1.4097, huber: 1.0560, swd: 1.0626, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 12.0790, mae: 1.7790, huber: 1.4122, swd: 2.1119, target_std: 4.7640
      Epoch 6 composite train-obj: 1.111927
            No improvement (1.0560), counter 3/5
    Epoch [7/50], Train Losses: mse: 6.7749, mae: 1.4640, huber: 1.1047, swd: 1.9272, target_std: 6.5148
    Epoch [7/50], Val Losses: mse: 7.6011, mae: 1.4110, huber: 1.0567, swd: 1.0715, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 11.9685, mae: 1.7806, huber: 1.4128, swd: 2.1242, target_std: 4.7640
      Epoch 7 composite train-obj: 1.104713
            No improvement (1.0567), counter 4/5
    Epoch [8/50], Train Losses: mse: 6.7205, mae: 1.4584, huber: 1.0993, swd: 1.9061, target_std: 6.5147
    Epoch [8/50], Val Losses: mse: 7.5743, mae: 1.4120, huber: 1.0578, swd: 1.0994, target_std: 4.2820
    Epoch [8/50], Test Losses: mse: 11.8957, mae: 1.7810, huber: 1.4128, swd: 2.1514, target_std: 4.7640
      Epoch 8 composite train-obj: 1.099299
    Epoch [8/50], Test Losses: mse: 12.2719, mae: 1.7859, huber: 1.4194, swd: 2.1438, target_std: 4.7640
    Best round's Test MSE: 12.2719, MAE: 1.7859, SWD: 2.1438
    Best round's Validation MSE: 7.5568, MAE: 1.4005
    Best round's Test verification MSE : 12.2719, MAE: 1.7859, SWD: 2.1438
    
    ==================================================
    Experiment Summary (TimeMixer_ettm1_seq96_pred336_20250429_1342)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.3451 ± 0.0635
      mae: 1.7845 ± 0.0024
      huber: 1.4183 ± 0.0018
      swd: 2.0290 ± 0.0814
      target_std: 4.7640 ± 0.0000
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.5392 ± 0.0262
      mae: 1.3974 ± 0.0022
      huber: 1.0444 ± 0.0021
      swd: 1.0343 ± 0.0322
      target_std: 4.2820 ± 0.0000
      count: 52.0000 ± 0.0000
    ==================================================
    
    Experiment complete: TimeMixer_ettm1_seq96_pred336_20250429_1342
    Model: TimeMixer
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 96-720


```python
utils.reload_modules([utils])
cfg_time_mixer_96_720 = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm1']['channels'],
    enc_in=data_mgr.datasets['ettm1']['channels'],
    dec_in=data_mgr.datasets['ettm1']['channels'],
    c_out=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_time_mixer_96_720 = execute_model_evaluation('ettm1', cfg_time_mixer_96_720, data_mgr, scale=False)

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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 9.7241, mae: 1.8028, huber: 1.4255, swd: 2.5645, target_std: 6.5148
    Epoch [1/50], Val Losses: mse: 9.8174, mae: 1.5896, huber: 1.2277, swd: 1.1821, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 14.4340, mae: 2.0231, huber: 1.6448, swd: 2.4077, target_std: 4.7646
      Epoch 1 composite train-obj: 1.425532
            Val objective improved inf → 1.2277, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.4452, mae: 1.6791, huber: 1.3067, swd: 2.3289, target_std: 6.5151
    Epoch [2/50], Val Losses: mse: 9.7243, mae: 1.5792, huber: 1.2183, swd: 1.1178, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 14.0915, mae: 1.9864, huber: 1.6091, swd: 2.2143, target_std: 4.7646
      Epoch 2 composite train-obj: 1.306676
            Val objective improved 1.2277 → 1.2183, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.2829, mae: 1.6643, huber: 1.2923, swd: 2.2986, target_std: 6.5149
    Epoch [3/50], Val Losses: mse: 9.6601, mae: 1.5797, huber: 1.2183, swd: 1.1820, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 13.8018, mae: 1.9785, huber: 1.6006, swd: 2.2900, target_std: 4.7646
      Epoch 3 composite train-obj: 1.292312
            No improvement (1.2183), counter 1/5
    Epoch [4/50], Train Losses: mse: 8.1510, mae: 1.6533, huber: 1.2817, swd: 2.2675, target_std: 6.5155
    Epoch [4/50], Val Losses: mse: 9.8096, mae: 1.6083, huber: 1.2455, swd: 1.3956, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 13.9101, mae: 2.0152, huber: 1.6359, swd: 2.6297, target_std: 4.7646
      Epoch 4 composite train-obj: 1.281685
            No improvement (1.2455), counter 2/5
    Epoch [5/50], Train Losses: mse: 8.0559, mae: 1.6460, huber: 1.2746, swd: 2.2426, target_std: 6.5152
    Epoch [5/50], Val Losses: mse: 9.3637, mae: 1.5710, huber: 1.2087, swd: 1.2973, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 13.5053, mae: 1.9937, huber: 1.6136, swd: 2.6599, target_std: 4.7646
      Epoch 5 composite train-obj: 1.274555
            Val objective improved 1.2183 → 1.2087, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.9562, mae: 1.6377, huber: 1.2665, swd: 2.2116, target_std: 6.5152
    Epoch [6/50], Val Losses: mse: 9.4307, mae: 1.5741, huber: 1.2123, swd: 1.1517, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 13.1994, mae: 1.9464, huber: 1.5687, swd: 2.2080, target_std: 4.7646
      Epoch 6 composite train-obj: 1.266485
            No improvement (1.2123), counter 1/5
    Epoch [7/50], Train Losses: mse: 7.8752, mae: 1.6312, huber: 1.2602, swd: 2.1871, target_std: 6.5152
    Epoch [7/50], Val Losses: mse: 9.6108, mae: 1.5848, huber: 1.2233, swd: 1.1728, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 13.1605, mae: 1.9398, huber: 1.5629, swd: 2.1736, target_std: 4.7646
      Epoch 7 composite train-obj: 1.260153
            No improvement (1.2233), counter 2/5
    Epoch [8/50], Train Losses: mse: 7.8086, mae: 1.6260, huber: 1.2549, swd: 2.1630, target_std: 6.5152
    Epoch [8/50], Val Losses: mse: 9.5532, mae: 1.5821, huber: 1.2204, swd: 1.1418, target_std: 4.2630
    Epoch [8/50], Test Losses: mse: 13.1106, mae: 1.9414, huber: 1.5638, swd: 2.1659, target_std: 4.7646
      Epoch 8 composite train-obj: 1.254937
            No improvement (1.2204), counter 3/5
    Epoch [9/50], Train Losses: mse: 7.7543, mae: 1.6214, huber: 1.2503, swd: 2.1440, target_std: 6.5151
    Epoch [9/50], Val Losses: mse: 9.4134, mae: 1.5831, huber: 1.2205, swd: 1.2712, target_std: 4.2630
    Epoch [9/50], Test Losses: mse: 13.0531, mae: 1.9603, huber: 1.5810, swd: 2.4693, target_std: 4.7646
      Epoch 9 composite train-obj: 1.250345
            No improvement (1.2205), counter 4/5
    Epoch [10/50], Train Losses: mse: 7.6897, mae: 1.6157, huber: 1.2448, swd: 2.1172, target_std: 6.5152
    Epoch [10/50], Val Losses: mse: 9.3973, mae: 1.5817, huber: 1.2191, swd: 1.2368, target_std: 4.2630
    Epoch [10/50], Test Losses: mse: 12.9583, mae: 1.9565, huber: 1.5772, swd: 2.4328, target_std: 4.7646
      Epoch 10 composite train-obj: 1.244756
    Epoch [10/50], Test Losses: mse: 13.5053, mae: 1.9937, huber: 1.6136, swd: 2.6599, target_std: 4.7646
    Best round's Test MSE: 13.5053, MAE: 1.9937, SWD: 2.6599
    Best round's Validation MSE: 9.3637, MAE: 1.5710
    Best round's Test verification MSE : 13.5053, MAE: 1.9937, SWD: 2.6599
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.0898, mae: 1.7482, huber: 1.3721, swd: 2.3253, target_std: 6.5154
    Epoch [1/50], Val Losses: mse: 9.9153, mae: 1.5952, huber: 1.2334, swd: 1.0227, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 14.2596, mae: 1.9772, huber: 1.6023, swd: 1.9977, target_std: 4.7646
      Epoch 1 composite train-obj: 1.372118
            Val objective improved inf → 1.2334, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.3099, mae: 1.6681, huber: 1.2959, swd: 2.1990, target_std: 6.5156
    Epoch [2/50], Val Losses: mse: 9.5856, mae: 1.5782, huber: 1.2164, swd: 1.0427, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 13.8014, mae: 1.9719, huber: 1.5950, swd: 2.1275, target_std: 4.7646
      Epoch 2 composite train-obj: 1.295914
            Val objective improved 1.2334 → 1.2164, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.1596, mae: 1.6540, huber: 1.2822, swd: 2.1503, target_std: 6.5153
    Epoch [3/50], Val Losses: mse: 9.7346, mae: 1.5852, huber: 1.2239, swd: 1.0127, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 13.6499, mae: 1.9514, huber: 1.5755, swd: 1.9861, target_std: 4.7646
      Epoch 3 composite train-obj: 1.282186
            No improvement (1.2239), counter 1/5
    Epoch [4/50], Train Losses: mse: 8.0498, mae: 1.6444, huber: 1.2727, swd: 2.1149, target_std: 6.5153
    Epoch [4/50], Val Losses: mse: 9.5707, mae: 1.5785, huber: 1.2166, swd: 1.0323, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 13.3616, mae: 1.9513, huber: 1.5738, swd: 2.0682, target_std: 4.7646
      Epoch 4 composite train-obj: 1.272746
            No improvement (1.2166), counter 2/5
    Epoch [5/50], Train Losses: mse: 7.9605, mae: 1.6368, huber: 1.2653, swd: 2.0868, target_std: 6.5150
    Epoch [5/50], Val Losses: mse: 9.4216, mae: 1.5691, huber: 1.2076, swd: 1.0788, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 13.2053, mae: 1.9555, huber: 1.5770, swd: 2.2473, target_std: 4.7646
      Epoch 5 composite train-obj: 1.265275
            Val objective improved 1.2164 → 1.2076, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.8662, mae: 1.6286, huber: 1.2573, swd: 2.0571, target_std: 6.5153
    Epoch [6/50], Val Losses: mse: 9.5881, mae: 1.5814, huber: 1.2198, swd: 1.0531, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 13.1484, mae: 1.9431, huber: 1.5655, swd: 2.0725, target_std: 4.7646
      Epoch 6 composite train-obj: 1.257330
            No improvement (1.2198), counter 1/5
    Epoch [7/50], Train Losses: mse: 7.7927, mae: 1.6222, huber: 1.2511, swd: 2.0362, target_std: 6.5151
    Epoch [7/50], Val Losses: mse: 9.4839, mae: 1.5822, huber: 1.2202, swd: 1.0454, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 13.0072, mae: 1.9358, huber: 1.5583, swd: 2.0468, target_std: 4.7646
      Epoch 7 composite train-obj: 1.251148
            No improvement (1.2202), counter 2/5
    Epoch [8/50], Train Losses: mse: 7.7298, mae: 1.6164, huber: 1.2456, swd: 2.0153, target_std: 6.5151
    Epoch [8/50], Val Losses: mse: 9.3596, mae: 1.5726, huber: 1.2100, swd: 1.1670, target_std: 4.2630
    Epoch [8/50], Test Losses: mse: 13.1404, mae: 1.9806, huber: 1.6003, swd: 2.5606, target_std: 4.7646
      Epoch 8 composite train-obj: 1.245566
            No improvement (1.2100), counter 3/5
    Epoch [9/50], Train Losses: mse: 7.6679, mae: 1.6109, huber: 1.2403, swd: 1.9952, target_std: 6.5148
    Epoch [9/50], Val Losses: mse: 9.4943, mae: 1.5870, huber: 1.2243, swd: 1.0616, target_std: 4.2630
    Epoch [9/50], Test Losses: mse: 12.9820, mae: 1.9411, huber: 1.5626, swd: 2.0652, target_std: 4.7646
      Epoch 9 composite train-obj: 1.240267
            No improvement (1.2243), counter 4/5
    Epoch [10/50], Train Losses: mse: 7.6165, mae: 1.6059, huber: 1.2354, swd: 1.9737, target_std: 6.5153
    Epoch [10/50], Val Losses: mse: 9.4072, mae: 1.5832, huber: 1.2201, swd: 1.1286, target_std: 4.2630
    Epoch [10/50], Test Losses: mse: 12.9011, mae: 1.9561, huber: 1.5765, swd: 2.3132, target_std: 4.7646
      Epoch 10 composite train-obj: 1.235377
    Epoch [10/50], Test Losses: mse: 13.2053, mae: 1.9555, huber: 1.5770, swd: 2.2473, target_std: 4.7646
    Best round's Test MSE: 13.2053, MAE: 1.9555, SWD: 2.2473
    Best round's Validation MSE: 9.4216, MAE: 1.5691
    Best round's Test verification MSE : 13.2053, MAE: 1.9555, SWD: 2.2473
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.1369, mae: 1.7544, huber: 1.3786, swd: 2.6064, target_std: 6.5152
    Epoch [1/50], Val Losses: mse: 9.9343, mae: 1.5956, huber: 1.2337, swd: 1.2006, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 14.4218, mae: 2.0082, huber: 1.6304, swd: 2.3828, target_std: 4.7646
      Epoch 1 composite train-obj: 1.378633
            Val objective improved inf → 1.2337, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.3479, mae: 1.6715, huber: 1.2988, swd: 2.4081, target_std: 6.5151
    Epoch [2/50], Val Losses: mse: 9.6162, mae: 1.5838, huber: 1.2215, swd: 1.2321, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 13.8441, mae: 1.9812, huber: 1.6030, swd: 2.3979, target_std: 4.7646
      Epoch 2 composite train-obj: 1.298838
            Val objective improved 1.2337 → 1.2215, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.1909, mae: 1.6574, huber: 1.2853, swd: 2.3623, target_std: 6.5149
    Epoch [3/50], Val Losses: mse: 9.6703, mae: 1.5849, huber: 1.2232, swd: 1.2567, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 13.7171, mae: 1.9739, huber: 1.5959, swd: 2.4012, target_std: 4.7646
      Epoch 3 composite train-obj: 1.285328
            No improvement (1.2232), counter 1/5
    Epoch [4/50], Train Losses: mse: 8.0848, mae: 1.6479, huber: 1.2762, swd: 2.3264, target_std: 6.5152
    Epoch [4/50], Val Losses: mse: 9.6458, mae: 1.5807, huber: 1.2192, swd: 1.1742, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 13.5003, mae: 1.9520, huber: 1.5747, swd: 2.2167, target_std: 4.7646
      Epoch 4 composite train-obj: 1.276213
            Val objective improved 1.2215 → 1.2192, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.9858, mae: 1.6391, huber: 1.2678, swd: 2.2890, target_std: 6.5153
    Epoch [5/50], Val Losses: mse: 9.7730, mae: 1.5903, huber: 1.2290, swd: 1.2335, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 13.3946, mae: 1.9519, huber: 1.5743, swd: 2.2798, target_std: 4.7646
      Epoch 5 composite train-obj: 1.267845
            No improvement (1.2290), counter 1/5
    Epoch [6/50], Train Losses: mse: 7.8942, mae: 1.6311, huber: 1.2602, swd: 2.2617, target_std: 6.5151
    Epoch [6/50], Val Losses: mse: 9.4889, mae: 1.5780, huber: 1.2165, swd: 1.2400, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 13.1812, mae: 1.9569, huber: 1.5783, swd: 2.4336, target_std: 4.7646
      Epoch 6 composite train-obj: 1.260248
            Val objective improved 1.2192 → 1.2165, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.8130, mae: 1.6244, huber: 1.2537, swd: 2.2356, target_std: 6.5152
    Epoch [7/50], Val Losses: mse: 9.4484, mae: 1.5816, huber: 1.2195, swd: 1.2589, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 13.0379, mae: 1.9572, huber: 1.5776, swd: 2.4780, target_std: 4.7646
      Epoch 7 composite train-obj: 1.253691
            No improvement (1.2195), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.7370, mae: 1.6180, huber: 1.2475, swd: 2.2066, target_std: 6.5154
    Epoch [8/50], Val Losses: mse: 9.4183, mae: 1.5813, huber: 1.2186, swd: 1.2720, target_std: 4.2630
    Epoch [8/50], Test Losses: mse: 13.0274, mae: 1.9571, huber: 1.5770, swd: 2.4683, target_std: 4.7646
      Epoch 8 composite train-obj: 1.247520
            No improvement (1.2186), counter 2/5
    Epoch [9/50], Train Losses: mse: 7.6720, mae: 1.6123, huber: 1.2419, swd: 2.1821, target_std: 6.5149
    Epoch [9/50], Val Losses: mse: 9.4224, mae: 1.5791, huber: 1.2173, swd: 1.2244, target_std: 4.2630
    Epoch [9/50], Test Losses: mse: 12.9192, mae: 1.9479, huber: 1.5689, swd: 2.3945, target_std: 4.7646
      Epoch 9 composite train-obj: 1.241928
            No improvement (1.2173), counter 3/5
    Epoch [10/50], Train Losses: mse: 7.6134, mae: 1.6075, huber: 1.2372, swd: 2.1572, target_std: 6.5150
    Epoch [10/50], Val Losses: mse: 9.4361, mae: 1.5794, huber: 1.2179, swd: 1.2490, target_std: 4.2630
    Epoch [10/50], Test Losses: mse: 12.9211, mae: 1.9493, huber: 1.5707, swd: 2.4318, target_std: 4.7646
      Epoch 10 composite train-obj: 1.237234
            No improvement (1.2179), counter 4/5
    Epoch [11/50], Train Losses: mse: 7.5605, mae: 1.6030, huber: 1.2328, swd: 2.1384, target_std: 6.5155
    Epoch [11/50], Val Losses: mse: 9.3759, mae: 1.5789, huber: 1.2170, swd: 1.2313, target_std: 4.2630
    Epoch [11/50], Test Losses: mse: 12.8481, mae: 1.9449, huber: 1.5662, swd: 2.3726, target_std: 4.7646
      Epoch 11 composite train-obj: 1.232763
    Epoch [11/50], Test Losses: mse: 13.1812, mae: 1.9569, huber: 1.5783, swd: 2.4336, target_std: 4.7646
    Best round's Test MSE: 13.1812, MAE: 1.9569, SWD: 2.4336
    Best round's Validation MSE: 9.4889, MAE: 1.5780
    Best round's Test verification MSE : 13.1812, MAE: 1.9569, SWD: 2.4336
    
    ==================================================
    Experiment Summary (TimeMixer_ettm1_seq96_pred720_20250429_1346)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 13.2973 ± 0.1474
      mae: 1.9687 ± 0.0177
      huber: 1.5896 ± 0.0170
      swd: 2.4469 ± 0.1687
      target_std: 4.7646 ± 0.0000
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.4247 ± 0.0512
      mae: 1.5727 ± 0.0038
      huber: 1.2109 ± 0.0039
      swd: 1.2054 ± 0.0925
      target_std: 4.2630 ± 0.0000
      count: 49.0000 ± 0.0000
    ==================================================
    
    Experiment complete: TimeMixer_ettm1_seq96_pred720_20250429_1346
    Model: TimeMixer
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### PatchTST

#### 96-96


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm1']['channels'],
    enc_in=data_mgr.datasets['ettm1']['channels'],
    dec_in=data_mgr.datasets['ettm1']['channels'],
    c_out=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp_tst_96_96 = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 6.0015, mae: 1.3592, huber: 1.0081, swd: 1.6723, target_std: 6.5114
    Epoch [1/50], Val Losses: mse: 5.4293, mae: 1.1925, huber: 0.8541, swd: 1.1129, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.2745, mae: 1.4638, huber: 1.1141, swd: 1.8084, target_std: 4.7755
      Epoch 1 composite train-obj: 1.008074
            Val objective improved inf → 0.8541, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.6361, mae: 1.3055, huber: 0.9582, swd: 1.6121, target_std: 6.5117
    Epoch [2/50], Val Losses: mse: 5.3758, mae: 1.1759, huber: 0.8388, swd: 1.0599, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.2403, mae: 1.4478, huber: 1.0990, swd: 1.7420, target_std: 4.7755
      Epoch 2 composite train-obj: 0.958182
            Val objective improved 0.8541 → 0.8388, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.5534, mae: 1.2900, huber: 0.9440, swd: 1.5829, target_std: 6.5111
    Epoch [3/50], Val Losses: mse: 5.5840, mae: 1.1914, huber: 0.8538, swd: 1.2088, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.4281, mae: 1.4723, huber: 1.1230, swd: 2.0320, target_std: 4.7755
      Epoch 3 composite train-obj: 0.943953
            No improvement (0.8538), counter 1/5
    Epoch [4/50], Train Losses: mse: 5.4952, mae: 1.2793, huber: 0.9341, swd: 1.5620, target_std: 6.5115
    Epoch [4/50], Val Losses: mse: 5.3745, mae: 1.1716, huber: 0.8347, swd: 0.9905, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.1931, mae: 1.4446, huber: 1.0959, swd: 1.6772, target_std: 4.7755
      Epoch 4 composite train-obj: 0.934127
            Val objective improved 0.8388 → 0.8347, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.4478, mae: 1.2717, huber: 0.9272, swd: 1.5456, target_std: 6.5110
    Epoch [5/50], Val Losses: mse: 5.3424, mae: 1.1716, huber: 0.8342, swd: 1.1343, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.3029, mae: 1.4725, huber: 1.1218, swd: 1.9778, target_std: 4.7755
      Epoch 5 composite train-obj: 0.927152
            Val objective improved 0.8347 → 0.8342, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.4017, mae: 1.2651, huber: 0.9209, swd: 1.5286, target_std: 6.5114
    Epoch [6/50], Val Losses: mse: 5.2730, mae: 1.1618, huber: 0.8257, swd: 1.0122, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.0390, mae: 1.4170, huber: 1.0712, swd: 1.6334, target_std: 4.7755
      Epoch 6 composite train-obj: 0.920926
            Val objective improved 0.8342 → 0.8257, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.3517, mae: 1.2572, huber: 0.9136, swd: 1.5089, target_std: 6.5117
    Epoch [7/50], Val Losses: mse: 5.2823, mae: 1.1648, huber: 0.8283, swd: 1.0257, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.0805, mae: 1.4314, huber: 1.0835, swd: 1.7103, target_std: 4.7755
      Epoch 7 composite train-obj: 0.913561
            No improvement (0.8283), counter 1/5
    Epoch [8/50], Train Losses: mse: 5.3114, mae: 1.2519, huber: 0.9086, swd: 1.4979, target_std: 6.5115
    Epoch [8/50], Val Losses: mse: 5.2951, mae: 1.1618, huber: 0.8263, swd: 1.0075, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.0000, mae: 1.4286, huber: 1.0810, swd: 1.6777, target_std: 4.7755
      Epoch 8 composite train-obj: 0.908557
            No improvement (0.8263), counter 2/5
    Epoch [9/50], Train Losses: mse: 5.2570, mae: 1.2449, huber: 0.9020, swd: 1.4757, target_std: 6.5109
    Epoch [9/50], Val Losses: mse: 5.4246, mae: 1.1781, huber: 0.8421, swd: 1.0478, target_std: 4.2867
    Epoch [9/50], Test Losses: mse: 8.0602, mae: 1.4351, huber: 1.0876, swd: 1.6765, target_std: 4.7755
      Epoch 9 composite train-obj: 0.902010
            No improvement (0.8421), counter 3/5
    Epoch [10/50], Train Losses: mse: 5.2161, mae: 1.2397, huber: 0.8971, swd: 1.4592, target_std: 6.5109
    Epoch [10/50], Val Losses: mse: 5.4203, mae: 1.1751, huber: 0.8387, swd: 1.0378, target_std: 4.2867
    Epoch [10/50], Test Losses: mse: 7.9962, mae: 1.4381, huber: 1.0893, swd: 1.7320, target_std: 4.7755
      Epoch 10 composite train-obj: 0.897072
            No improvement (0.8387), counter 4/5
    Epoch [11/50], Train Losses: mse: 5.1725, mae: 1.2342, huber: 0.8918, swd: 1.4431, target_std: 6.5115
    Epoch [11/50], Val Losses: mse: 5.5526, mae: 1.1893, huber: 0.8506, swd: 1.1306, target_std: 4.2867
    Epoch [11/50], Test Losses: mse: 8.1934, mae: 1.4632, huber: 1.1127, swd: 1.8599, target_std: 4.7755
      Epoch 11 composite train-obj: 0.891784
    Epoch [11/50], Test Losses: mse: 8.0390, mae: 1.4170, huber: 1.0712, swd: 1.6334, target_std: 4.7755
    Best round's Test MSE: 8.0390, MAE: 1.4170, SWD: 1.6334
    Best round's Validation MSE: 5.2730, MAE: 1.1618
    Best round's Test verification MSE : 8.0390, MAE: 1.4170, SWD: 1.6334
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 5.9766, mae: 1.3570, huber: 1.0059, swd: 1.6448, target_std: 6.5112
    Epoch [1/50], Val Losses: mse: 5.6011, mae: 1.1995, huber: 0.8610, swd: 1.1037, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.5908, mae: 1.4914, huber: 1.1409, swd: 2.0004, target_std: 4.7755
      Epoch 1 composite train-obj: 1.005868
            Val objective improved inf → 0.8610, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.6260, mae: 1.3033, huber: 0.9561, swd: 1.5826, target_std: 6.5118
    Epoch [2/50], Val Losses: mse: 5.4104, mae: 1.1808, huber: 0.8436, swd: 1.0778, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.1883, mae: 1.4488, huber: 1.0999, swd: 1.8782, target_std: 4.7755
      Epoch 2 composite train-obj: 0.956097
            Val objective improved 0.8610 → 0.8436, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.5353, mae: 1.2864, huber: 0.9406, swd: 1.5561, target_std: 6.5109
    Epoch [3/50], Val Losses: mse: 5.3882, mae: 1.1685, huber: 0.8327, swd: 0.9933, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.1246, mae: 1.4353, huber: 1.0870, swd: 1.7111, target_std: 4.7755
      Epoch 3 composite train-obj: 0.940628
            Val objective improved 0.8436 → 0.8327, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.4760, mae: 1.2759, huber: 0.9310, swd: 1.5345, target_std: 6.5110
    Epoch [4/50], Val Losses: mse: 5.3415, mae: 1.1704, huber: 0.8333, swd: 1.0516, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.2326, mae: 1.4613, huber: 1.1108, swd: 1.9757, target_std: 4.7755
      Epoch 4 composite train-obj: 0.931034
            No improvement (0.8333), counter 1/5
    Epoch [5/50], Train Losses: mse: 5.4167, mae: 1.2673, huber: 0.9229, swd: 1.5163, target_std: 6.5112
    Epoch [5/50], Val Losses: mse: 5.4135, mae: 1.1736, huber: 0.8368, swd: 1.0173, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.0624, mae: 1.4180, huber: 1.0711, swd: 1.6254, target_std: 4.7755
      Epoch 5 composite train-obj: 0.922888
            No improvement (0.8368), counter 2/5
    Epoch [6/50], Train Losses: mse: 5.3701, mae: 1.2594, huber: 0.9156, swd: 1.4993, target_std: 6.5114
    Epoch [6/50], Val Losses: mse: 5.4936, mae: 1.1774, huber: 0.8420, swd: 1.0910, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.1162, mae: 1.4366, huber: 1.0901, swd: 1.8693, target_std: 4.7755
      Epoch 6 composite train-obj: 0.915599
            No improvement (0.8420), counter 3/5
    Epoch [7/50], Train Losses: mse: 5.3265, mae: 1.2533, huber: 0.9099, swd: 1.4784, target_std: 6.5115
    Epoch [7/50], Val Losses: mse: 5.2706, mae: 1.1551, huber: 0.8215, swd: 0.9592, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 7.9990, mae: 1.4051, huber: 1.0597, swd: 1.5811, target_std: 4.7755
      Epoch 7 composite train-obj: 0.909878
            Val objective improved 0.8327 → 0.8215, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.2793, mae: 1.2466, huber: 0.9037, swd: 1.4661, target_std: 6.5112
    Epoch [8/50], Val Losses: mse: 5.4146, mae: 1.1764, huber: 0.8388, swd: 1.0501, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.1369, mae: 1.4465, huber: 1.0970, swd: 1.8328, target_std: 4.7755
      Epoch 8 composite train-obj: 0.903666
            No improvement (0.8388), counter 1/5
    Epoch [9/50], Train Losses: mse: 5.2219, mae: 1.2400, huber: 0.8974, swd: 1.4479, target_std: 6.5112
    Epoch [9/50], Val Losses: mse: 5.3809, mae: 1.1722, huber: 0.8366, swd: 1.0511, target_std: 4.2867
    Epoch [9/50], Test Losses: mse: 8.1457, mae: 1.4468, huber: 1.0980, swd: 1.8340, target_std: 4.7755
      Epoch 9 composite train-obj: 0.897439
            No improvement (0.8366), counter 2/5
    Epoch [10/50], Train Losses: mse: 5.1649, mae: 1.2338, huber: 0.8914, swd: 1.4319, target_std: 6.5105
    Epoch [10/50], Val Losses: mse: 5.4028, mae: 1.1716, huber: 0.8367, swd: 0.9971, target_std: 4.2867
    Epoch [10/50], Test Losses: mse: 8.0975, mae: 1.4417, huber: 1.0939, swd: 1.7071, target_std: 4.7755
      Epoch 10 composite train-obj: 0.891380
            No improvement (0.8367), counter 3/5
    Epoch [11/50], Train Losses: mse: 5.1078, mae: 1.2277, huber: 0.8856, swd: 1.4190, target_std: 6.5106
    Epoch [11/50], Val Losses: mse: 5.5247, mae: 1.1856, huber: 0.8495, swd: 1.0622, target_std: 4.2867
    Epoch [11/50], Test Losses: mse: 8.0141, mae: 1.4247, huber: 1.0777, swd: 1.6757, target_std: 4.7755
      Epoch 11 composite train-obj: 0.885551
            No improvement (0.8495), counter 4/5
    Epoch [12/50], Train Losses: mse: 5.0473, mae: 1.2211, huber: 0.8791, swd: 1.3999, target_std: 6.5111
    Epoch [12/50], Val Losses: mse: 5.4872, mae: 1.1830, huber: 0.8452, swd: 1.0071, target_std: 4.2867
    Epoch [12/50], Test Losses: mse: 8.3083, mae: 1.4370, huber: 1.0885, swd: 1.6512, target_std: 4.7755
      Epoch 12 composite train-obj: 0.879141
    Epoch [12/50], Test Losses: mse: 7.9990, mae: 1.4051, huber: 1.0597, swd: 1.5811, target_std: 4.7755
    Best round's Test MSE: 7.9990, MAE: 1.4051, SWD: 1.5811
    Best round's Validation MSE: 5.2706, MAE: 1.1551
    Best round's Test verification MSE : 7.9990, MAE: 1.4051, SWD: 1.5811
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 5.9900, mae: 1.3587, huber: 1.0076, swd: 1.5353, target_std: 6.5114
    Epoch [1/50], Val Losses: mse: 5.5686, mae: 1.2026, huber: 0.8640, swd: 1.1447, target_std: 4.2867
    Epoch [1/50], Test Losses: mse: 8.4199, mae: 1.4804, huber: 1.1305, swd: 1.8658, target_std: 4.7755
      Epoch 1 composite train-obj: 1.007562
            Val objective improved inf → 0.8640, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.6377, mae: 1.3056, huber: 0.9583, swd: 1.4818, target_std: 6.5114
    Epoch [2/50], Val Losses: mse: 5.4894, mae: 1.1834, huber: 0.8463, swd: 1.0030, target_std: 4.2867
    Epoch [2/50], Test Losses: mse: 8.2264, mae: 1.4419, huber: 1.0929, swd: 1.6475, target_std: 4.7755
      Epoch 2 composite train-obj: 0.958272
            Val objective improved 0.8640 → 0.8463, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.5443, mae: 1.2893, huber: 0.9433, swd: 1.4550, target_std: 6.5112
    Epoch [3/50], Val Losses: mse: 5.3712, mae: 1.1734, huber: 0.8379, swd: 0.9713, target_std: 4.2867
    Epoch [3/50], Test Losses: mse: 8.1417, mae: 1.4345, huber: 1.0874, swd: 1.6307, target_std: 4.7755
      Epoch 3 composite train-obj: 0.943323
            Val objective improved 0.8463 → 0.8379, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.4821, mae: 1.2785, huber: 0.9334, swd: 1.4353, target_std: 6.5114
    Epoch [4/50], Val Losses: mse: 5.4027, mae: 1.1701, huber: 0.8331, swd: 1.0144, target_std: 4.2867
    Epoch [4/50], Test Losses: mse: 8.2491, mae: 1.4498, huber: 1.1000, swd: 1.6817, target_std: 4.7755
      Epoch 4 composite train-obj: 0.933422
            Val objective improved 0.8379 → 0.8331, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.4294, mae: 1.2689, huber: 0.9245, swd: 1.4133, target_std: 6.5113
    Epoch [5/50], Val Losses: mse: 5.4305, mae: 1.1863, huber: 0.8483, swd: 1.0728, target_std: 4.2867
    Epoch [5/50], Test Losses: mse: 8.1652, mae: 1.4515, huber: 1.1029, swd: 1.7962, target_std: 4.7755
      Epoch 5 composite train-obj: 0.924512
            No improvement (0.8483), counter 1/5
    Epoch [6/50], Train Losses: mse: 5.3688, mae: 1.2607, huber: 0.9169, swd: 1.3946, target_std: 6.5114
    Epoch [6/50], Val Losses: mse: 5.5042, mae: 1.1806, huber: 0.8431, swd: 1.0263, target_std: 4.2867
    Epoch [6/50], Test Losses: mse: 8.3020, mae: 1.4765, huber: 1.1251, swd: 1.8449, target_std: 4.7755
      Epoch 6 composite train-obj: 0.916876
            No improvement (0.8431), counter 2/5
    Epoch [7/50], Train Losses: mse: 5.3128, mae: 1.2530, huber: 0.9097, swd: 1.3791, target_std: 6.5109
    Epoch [7/50], Val Losses: mse: 5.3643, mae: 1.1854, huber: 0.8472, swd: 0.9675, target_std: 4.2867
    Epoch [7/50], Test Losses: mse: 8.1670, mae: 1.4547, huber: 1.1049, swd: 1.6044, target_std: 4.7755
      Epoch 7 composite train-obj: 0.909666
            No improvement (0.8472), counter 3/5
    Epoch [8/50], Train Losses: mse: 5.2489, mae: 1.2460, huber: 0.9030, swd: 1.3621, target_std: 6.5114
    Epoch [8/50], Val Losses: mse: 5.3678, mae: 1.1689, huber: 0.8334, swd: 0.9648, target_std: 4.2867
    Epoch [8/50], Test Losses: mse: 8.0817, mae: 1.4338, huber: 1.0864, swd: 1.6116, target_std: 4.7755
      Epoch 8 composite train-obj: 0.903028
            No improvement (0.8334), counter 4/5
    Epoch [9/50], Train Losses: mse: 5.1978, mae: 1.2389, huber: 0.8964, swd: 1.3441, target_std: 6.5112
    Epoch [9/50], Val Losses: mse: 5.5695, mae: 1.1834, huber: 0.8465, swd: 1.0196, target_std: 4.2867
    Epoch [9/50], Test Losses: mse: 8.1691, mae: 1.4481, huber: 1.0993, swd: 1.6846, target_std: 4.7755
      Epoch 9 composite train-obj: 0.896351
    Epoch [9/50], Test Losses: mse: 8.2491, mae: 1.4498, huber: 1.1000, swd: 1.6817, target_std: 4.7755
    Best round's Test MSE: 8.2491, MAE: 1.4498, SWD: 1.6817
    Best round's Validation MSE: 5.4027, MAE: 1.1701
    Best round's Test verification MSE : 8.2491, MAE: 1.4498, SWD: 1.6817
    
    ==================================================
    Experiment Summary (PatchTST_ettm1_seq96_pred96_20250430_0427)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 8.0957 ± 0.1097
      mae: 1.4240 ± 0.0189
      huber: 1.0770 ± 0.0169
      swd: 1.6321 ± 0.0411
      target_std: 4.7755 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.3154 ± 0.0617
      mae: 1.1623 ± 0.0061
      huber: 0.8267 ± 0.0048
      swd: 0.9953 ± 0.0255
      target_std: 4.2867 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: PatchTST_ettm1_seq96_pred96_20250430_0427
    Model: PatchTST
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 96-196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm1']['channels'],
    enc_in=data_mgr.datasets['ettm1']['channels'],
    dec_in=data_mgr.datasets['ettm1']['channels'],
    c_out=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp_tst_96_196 = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 7.1735, mae: 1.5002, huber: 1.1410, swd: 2.0335, target_std: 6.5129
    Epoch [1/50], Val Losses: mse: 6.4233, mae: 1.3087, huber: 0.9611, swd: 1.0838, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.7286, mae: 1.6542, huber: 1.2964, swd: 2.0045, target_std: 4.7691
      Epoch 1 composite train-obj: 1.140983
            Val objective improved inf → 0.9611, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.7939, mae: 1.4518, huber: 1.0954, swd: 1.9732, target_std: 6.5130
    Epoch [2/50], Val Losses: mse: 6.4222, mae: 1.3120, huber: 0.9638, swd: 1.0815, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.4745, mae: 1.6450, huber: 1.2855, swd: 1.9145, target_std: 4.7691
      Epoch 2 composite train-obj: 1.095406
            No improvement (0.9638), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.6770, mae: 1.4369, huber: 1.0814, swd: 1.9395, target_std: 6.5129
    Epoch [3/50], Val Losses: mse: 6.4408, mae: 1.3047, huber: 0.9580, swd: 1.0837, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.3662, mae: 1.6409, huber: 1.2824, swd: 1.9909, target_std: 4.7691
      Epoch 3 composite train-obj: 1.081439
            Val objective improved 0.9611 → 0.9580, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.5960, mae: 1.4270, huber: 1.0722, swd: 1.9125, target_std: 6.5131
    Epoch [4/50], Val Losses: mse: 6.4181, mae: 1.3119, huber: 0.9652, swd: 1.2677, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.4282, mae: 1.6645, huber: 1.3049, swd: 2.3166, target_std: 4.7691
      Epoch 4 composite train-obj: 1.072163
            No improvement (0.9652), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.5292, mae: 1.4191, huber: 1.0647, swd: 1.8897, target_std: 6.5126
    Epoch [5/50], Val Losses: mse: 6.4220, mae: 1.2993, huber: 0.9533, swd: 1.0839, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.2808, mae: 1.6384, huber: 1.2796, swd: 2.0406, target_std: 4.7691
      Epoch 5 composite train-obj: 1.064674
            Val objective improved 0.9580 → 0.9533, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.4686, mae: 1.4112, huber: 1.0572, swd: 1.8662, target_std: 6.5128
    Epoch [6/50], Val Losses: mse: 6.3002, mae: 1.2960, huber: 0.9486, swd: 1.1279, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.3878, mae: 1.6462, huber: 1.2867, swd: 2.1138, target_std: 4.7691
      Epoch 6 composite train-obj: 1.057157
            Val objective improved 0.9533 → 0.9486, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 6.4138, mae: 1.4046, huber: 1.0509, swd: 1.8493, target_std: 6.5131
    Epoch [7/50], Val Losses: mse: 6.2960, mae: 1.2961, huber: 0.9496, swd: 1.1203, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.3632, mae: 1.6384, huber: 1.2786, swd: 1.9511, target_std: 4.7691
      Epoch 7 composite train-obj: 1.050880
            No improvement (0.9496), counter 1/5
    Epoch [8/50], Train Losses: mse: 6.3610, mae: 1.3981, huber: 1.0446, swd: 1.8274, target_std: 6.5131
    Epoch [8/50], Val Losses: mse: 6.4176, mae: 1.3163, huber: 0.9681, swd: 1.1766, target_std: 4.2926
    Epoch [8/50], Test Losses: mse: 10.3596, mae: 1.6471, huber: 1.2867, swd: 2.0340, target_std: 4.7691
      Epoch 8 composite train-obj: 1.044552
            No improvement (0.9681), counter 2/5
    Epoch [9/50], Train Losses: mse: 6.3101, mae: 1.3924, huber: 1.0391, swd: 1.8149, target_std: 6.5130
    Epoch [9/50], Val Losses: mse: 6.4900, mae: 1.3078, huber: 0.9614, swd: 1.1141, target_std: 4.2926
    Epoch [9/50], Test Losses: mse: 10.3996, mae: 1.6338, huber: 1.2750, swd: 1.9309, target_std: 4.7691
      Epoch 9 composite train-obj: 1.039083
            No improvement (0.9614), counter 3/5
    Epoch [10/50], Train Losses: mse: 6.2570, mae: 1.3866, huber: 1.0335, swd: 1.7943, target_std: 6.5126
    Epoch [10/50], Val Losses: mse: 6.6056, mae: 1.3228, huber: 0.9746, swd: 1.2024, target_std: 4.2926
    Epoch [10/50], Test Losses: mse: 10.3979, mae: 1.6509, huber: 1.2897, swd: 2.1238, target_std: 4.7691
      Epoch 10 composite train-obj: 1.033470
            No improvement (0.9746), counter 4/5
    Epoch [11/50], Train Losses: mse: 6.2122, mae: 1.3814, huber: 1.0285, swd: 1.7813, target_std: 6.5129
    Epoch [11/50], Val Losses: mse: 6.8508, mae: 1.3493, huber: 1.0008, swd: 1.2289, target_std: 4.2926
    Epoch [11/50], Test Losses: mse: 10.6353, mae: 1.6557, huber: 1.2952, swd: 1.9599, target_std: 4.7691
      Epoch 11 composite train-obj: 1.028457
    Epoch [11/50], Test Losses: mse: 10.3878, mae: 1.6462, huber: 1.2867, swd: 2.1138, target_std: 4.7691
    Best round's Test MSE: 10.3878, MAE: 1.6462, SWD: 2.1138
    Best round's Validation MSE: 6.3002, MAE: 1.2960
    Best round's Test verification MSE : 10.3878, MAE: 1.6462, SWD: 2.1138
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.1970, mae: 1.5016, huber: 1.1425, swd: 2.0776, target_std: 6.5130
    Epoch [1/50], Val Losses: mse: 6.4807, mae: 1.3059, huber: 0.9588, swd: 0.9995, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.6687, mae: 1.6522, huber: 1.2923, swd: 1.8896, target_std: 4.7691
      Epoch 1 composite train-obj: 1.142498
            Val objective improved inf → 0.9588, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.8183, mae: 1.4514, huber: 1.0952, swd: 1.9995, target_std: 6.5129
    Epoch [2/50], Val Losses: mse: 6.3974, mae: 1.2970, huber: 0.9514, swd: 1.1656, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.5493, mae: 1.6558, huber: 1.2971, swd: 2.1554, target_std: 4.7691
      Epoch 2 composite train-obj: 1.095181
            Val objective improved 0.9588 → 0.9514, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.7033, mae: 1.4362, huber: 1.0809, swd: 1.9642, target_std: 6.5130
    Epoch [3/50], Val Losses: mse: 6.4100, mae: 1.3006, huber: 0.9538, swd: 1.1846, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.4306, mae: 1.6553, huber: 1.2959, swd: 2.2352, target_std: 4.7691
      Epoch 3 composite train-obj: 1.080919
            No improvement (0.9538), counter 1/5
    Epoch [4/50], Train Losses: mse: 6.6045, mae: 1.4250, huber: 1.0703, swd: 1.9373, target_std: 6.5130
    Epoch [4/50], Val Losses: mse: 6.2790, mae: 1.2901, huber: 0.9442, swd: 1.1133, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.2930, mae: 1.6371, huber: 1.2791, swd: 2.0467, target_std: 4.7691
      Epoch 4 composite train-obj: 1.070299
            Val objective improved 0.9514 → 0.9442, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.5202, mae: 1.4159, huber: 1.0616, swd: 1.9150, target_std: 6.5128
    Epoch [5/50], Val Losses: mse: 6.4188, mae: 1.3032, huber: 0.9561, swd: 1.0482, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.2931, mae: 1.6302, huber: 1.2713, swd: 1.9019, target_std: 4.7691
      Epoch 5 composite train-obj: 1.061566
            No improvement (0.9561), counter 1/5
    Epoch [6/50], Train Losses: mse: 6.4472, mae: 1.4078, huber: 1.0538, swd: 1.8918, target_std: 6.5129
    Epoch [6/50], Val Losses: mse: 6.3191, mae: 1.2917, huber: 0.9471, swd: 1.1343, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.2580, mae: 1.6367, huber: 1.2795, swd: 2.0598, target_std: 4.7691
      Epoch 6 composite train-obj: 1.053804
            No improvement (0.9471), counter 2/5
    Epoch [7/50], Train Losses: mse: 6.3949, mae: 1.4019, huber: 1.0482, swd: 1.8746, target_std: 6.5129
    Epoch [7/50], Val Losses: mse: 6.3302, mae: 1.2917, huber: 0.9463, swd: 1.0744, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.1370, mae: 1.6148, huber: 1.2570, swd: 1.8697, target_std: 4.7691
      Epoch 7 composite train-obj: 1.048199
            No improvement (0.9463), counter 3/5
    Epoch [8/50], Train Losses: mse: 6.3401, mae: 1.3953, huber: 1.0419, swd: 1.8553, target_std: 6.5127
    Epoch [8/50], Val Losses: mse: 6.5511, mae: 1.3106, huber: 0.9646, swd: 1.0703, target_std: 4.2926
    Epoch [8/50], Test Losses: mse: 10.2365, mae: 1.6303, huber: 1.2719, swd: 1.8954, target_std: 4.7691
      Epoch 8 composite train-obj: 1.041913
            No improvement (0.9646), counter 4/5
    Epoch [9/50], Train Losses: mse: 6.2884, mae: 1.3900, huber: 1.0367, swd: 1.8374, target_std: 6.5129
    Epoch [9/50], Val Losses: mse: 6.4658, mae: 1.3000, huber: 0.9545, swd: 1.1658, target_std: 4.2926
    Epoch [9/50], Test Losses: mse: 10.2692, mae: 1.6540, huber: 1.2937, swd: 2.1714, target_std: 4.7691
      Epoch 9 composite train-obj: 1.036665
    Epoch [9/50], Test Losses: mse: 10.2930, mae: 1.6371, huber: 1.2791, swd: 2.0467, target_std: 4.7691
    Best round's Test MSE: 10.2930, MAE: 1.6371, SWD: 2.0467
    Best round's Validation MSE: 6.2790, MAE: 1.2901
    Best round's Test verification MSE : 10.2930, MAE: 1.6371, SWD: 2.0467
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.1863, mae: 1.5019, huber: 1.1426, swd: 1.8283, target_std: 6.5130
    Epoch [1/50], Val Losses: mse: 6.5340, mae: 1.3166, huber: 0.9696, swd: 1.1302, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.5773, mae: 1.6601, huber: 1.3009, swd: 2.0250, target_std: 4.7691
      Epoch 1 composite train-obj: 1.142642
            Val objective improved inf → 0.9696, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.8199, mae: 1.4539, huber: 1.0975, swd: 1.7782, target_std: 6.5130
    Epoch [2/50], Val Losses: mse: 6.3980, mae: 1.3023, huber: 0.9557, swd: 1.0718, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.4997, mae: 1.6538, huber: 1.2956, swd: 2.0171, target_std: 4.7691
      Epoch 2 composite train-obj: 1.097474
            Val objective improved 0.9696 → 0.9557, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.7051, mae: 1.4380, huber: 1.0826, swd: 1.7448, target_std: 6.5131
    Epoch [3/50], Val Losses: mse: 6.4458, mae: 1.3056, huber: 0.9595, swd: 1.0579, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.3602, mae: 1.6366, huber: 1.2793, swd: 1.9353, target_std: 4.7691
      Epoch 3 composite train-obj: 1.082647
            No improvement (0.9595), counter 1/5
    Epoch [4/50], Train Losses: mse: 6.6049, mae: 1.4261, huber: 1.0715, swd: 1.7226, target_std: 6.5129
    Epoch [4/50], Val Losses: mse: 6.4154, mae: 1.3043, huber: 0.9568, swd: 1.1304, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.4541, mae: 1.6548, huber: 1.2946, swd: 2.0847, target_std: 4.7691
      Epoch 4 composite train-obj: 1.071492
            No improvement (0.9568), counter 2/5
    Epoch [5/50], Train Losses: mse: 6.5195, mae: 1.4170, huber: 1.0628, swd: 1.6973, target_std: 6.5128
    Epoch [5/50], Val Losses: mse: 6.3477, mae: 1.2951, huber: 0.9493, swd: 1.0121, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.2542, mae: 1.6208, huber: 1.2630, swd: 1.7510, target_std: 4.7691
      Epoch 5 composite train-obj: 1.062773
            Val objective improved 0.9557 → 0.9493, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.4559, mae: 1.4091, huber: 1.0552, swd: 1.6761, target_std: 6.5130
    Epoch [6/50], Val Losses: mse: 6.3518, mae: 1.2915, huber: 0.9462, swd: 1.0433, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.2881, mae: 1.6354, huber: 1.2776, swd: 1.9888, target_std: 4.7691
      Epoch 6 composite train-obj: 1.055177
            Val objective improved 0.9493 → 0.9462, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 6.4003, mae: 1.4020, huber: 1.0485, swd: 1.6607, target_std: 6.5127
    Epoch [7/50], Val Losses: mse: 6.2420, mae: 1.2910, huber: 0.9451, swd: 1.0486, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.2582, mae: 1.6456, huber: 1.2861, swd: 1.9233, target_std: 4.7691
      Epoch 7 composite train-obj: 1.048507
            Val objective improved 0.9462 → 0.9451, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 6.3429, mae: 1.3952, huber: 1.0419, swd: 1.6398, target_std: 6.5129
    Epoch [8/50], Val Losses: mse: 6.3325, mae: 1.2911, huber: 0.9455, swd: 1.0130, target_std: 4.2926
    Epoch [8/50], Test Losses: mse: 10.1948, mae: 1.6304, huber: 1.2717, swd: 1.8201, target_std: 4.7691
      Epoch 8 composite train-obj: 1.041867
            No improvement (0.9455), counter 1/5
    Epoch [9/50], Train Losses: mse: 6.2852, mae: 1.3887, huber: 1.0356, swd: 1.6224, target_std: 6.5131
    Epoch [9/50], Val Losses: mse: 6.3808, mae: 1.3052, huber: 0.9576, swd: 1.0521, target_std: 4.2926
    Epoch [9/50], Test Losses: mse: 10.1831, mae: 1.6377, huber: 1.2778, swd: 1.8885, target_std: 4.7691
      Epoch 9 composite train-obj: 1.035591
            No improvement (0.9576), counter 2/5
    Epoch [10/50], Train Losses: mse: 6.2364, mae: 1.3834, huber: 1.0305, swd: 1.6089, target_std: 6.5127
    Epoch [10/50], Val Losses: mse: 6.6554, mae: 1.3243, huber: 0.9760, swd: 0.9965, target_std: 4.2926
    Epoch [10/50], Test Losses: mse: 10.4147, mae: 1.6296, huber: 1.2709, swd: 1.6789, target_std: 4.7691
      Epoch 10 composite train-obj: 1.030505
            No improvement (0.9760), counter 3/5
    Epoch [11/50], Train Losses: mse: 6.1815, mae: 1.3772, huber: 1.0245, swd: 1.5915, target_std: 6.5128
    Epoch [11/50], Val Losses: mse: 6.5218, mae: 1.3049, huber: 0.9584, swd: 1.1034, target_std: 4.2926
    Epoch [11/50], Test Losses: mse: 10.2573, mae: 1.6360, huber: 1.2762, swd: 1.9316, target_std: 4.7691
      Epoch 11 composite train-obj: 1.024495
            No improvement (0.9584), counter 4/5
    Epoch [12/50], Train Losses: mse: 6.1283, mae: 1.3722, huber: 1.0195, swd: 1.5753, target_std: 6.5129
    Epoch [12/50], Val Losses: mse: 6.5589, mae: 1.3119, huber: 0.9653, swd: 1.0540, target_std: 4.2926
    Epoch [12/50], Test Losses: mse: 10.2586, mae: 1.6271, huber: 1.2688, swd: 1.8848, target_std: 4.7691
      Epoch 12 composite train-obj: 1.019516
    Epoch [12/50], Test Losses: mse: 10.2582, mae: 1.6456, huber: 1.2861, swd: 1.9233, target_std: 4.7691
    Best round's Test MSE: 10.2582, MAE: 1.6456, SWD: 1.9233
    Best round's Validation MSE: 6.2420, MAE: 1.2910
    Best round's Test verification MSE : 10.2582, MAE: 1.6456, SWD: 1.9233
    
    ==================================================
    Experiment Summary (PatchTST_ettm1_seq96_pred196_20250430_0430)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.3130 ± 0.0548
      mae: 1.6429 ± 0.0042
      huber: 1.2839 ± 0.0034
      swd: 2.0279 ± 0.0789
      target_std: 4.7691 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 6.2737 ± 0.0241
      mae: 1.2924 ± 0.0026
      huber: 0.9460 ± 0.0019
      swd: 1.0966 ± 0.0345
      target_std: 4.2926 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: PatchTST_ettm1_seq96_pred196_20250430_0430
    Model: PatchTST
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 96-336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm1']['channels'],
    enc_in=data_mgr.datasets['ettm1']['channels'],
    dec_in=data_mgr.datasets['ettm1']['channels'],
    c_out=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp_tst_96_336 = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 8.1043, mae: 1.6207, huber: 1.2536, swd: 2.2202, target_std: 6.5148
    Epoch [1/50], Val Losses: mse: 7.6023, mae: 1.4186, huber: 1.0647, swd: 1.0524, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.4518, mae: 1.8198, huber: 1.4538, swd: 2.1644, target_std: 4.7640
      Epoch 1 composite train-obj: 1.253602
            Val objective improved inf → 1.0647, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.7020, mae: 1.5742, huber: 1.2095, swd: 2.1593, target_std: 6.5151
    Epoch [2/50], Val Losses: mse: 7.6864, mae: 1.4221, huber: 1.0674, swd: 1.0643, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 12.2794, mae: 1.8070, huber: 1.4403, swd: 2.1123, target_std: 4.7640
      Epoch 2 composite train-obj: 1.209452
            No improvement (1.0674), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.5695, mae: 1.5594, huber: 1.1954, swd: 2.1235, target_std: 6.5151
    Epoch [3/50], Val Losses: mse: 7.6864, mae: 1.4291, huber: 1.0739, swd: 1.0595, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 12.0595, mae: 1.7951, huber: 1.4278, swd: 2.1036, target_std: 4.7640
      Epoch 3 composite train-obj: 1.195385
            No improvement (1.0739), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.4714, mae: 1.5488, huber: 1.1852, swd: 2.0932, target_std: 6.5148
    Epoch [4/50], Val Losses: mse: 7.6341, mae: 1.4211, huber: 1.0669, swd: 1.0893, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 11.9827, mae: 1.7912, huber: 1.4245, swd: 2.0688, target_std: 4.7640
      Epoch 4 composite train-obj: 1.185171
            No improvement (1.0669), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.3931, mae: 1.5401, huber: 1.1768, swd: 2.0661, target_std: 6.5150
    Epoch [5/50], Val Losses: mse: 7.7594, mae: 1.4330, huber: 1.0770, swd: 1.0710, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 12.0917, mae: 1.8073, huber: 1.4386, swd: 2.1648, target_std: 4.7640
      Epoch 5 composite train-obj: 1.176845
            No improvement (1.0770), counter 4/5
    Epoch [6/50], Train Losses: mse: 7.3202, mae: 1.5326, huber: 1.1696, swd: 2.0451, target_std: 6.5147
    Epoch [6/50], Val Losses: mse: 7.7659, mae: 1.4315, huber: 1.0769, swd: 1.1519, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 11.9211, mae: 1.8028, huber: 1.4354, swd: 2.2798, target_std: 4.7640
      Epoch 6 composite train-obj: 1.169635
    Epoch [6/50], Test Losses: mse: 12.4518, mae: 1.8198, huber: 1.4538, swd: 2.1644, target_std: 4.7640
    Best round's Test MSE: 12.4518, MAE: 1.8198, SWD: 2.1644
    Best round's Validation MSE: 7.6023, MAE: 1.4186
    Best round's Test verification MSE : 12.4518, MAE: 1.8198, SWD: 2.1644
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.1131, mae: 1.6209, huber: 1.2540, swd: 2.3342, target_std: 6.5150
    Epoch [1/50], Val Losses: mse: 7.7721, mae: 1.4307, huber: 1.0763, swd: 1.0228, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.4670, mae: 1.8031, huber: 1.4379, swd: 2.0045, target_std: 4.7640
      Epoch 1 composite train-obj: 1.253958
            Val objective improved inf → 1.0763, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.7292, mae: 1.5755, huber: 1.2108, swd: 2.2711, target_std: 6.5148
    Epoch [2/50], Val Losses: mse: 7.6871, mae: 1.4265, huber: 1.0731, swd: 1.2442, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 12.2936, mae: 1.8192, huber: 1.4533, swd: 2.4821, target_std: 4.7640
      Epoch 2 composite train-obj: 1.210778
            Val objective improved 1.0763 → 1.0731, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.5867, mae: 1.5597, huber: 1.1956, swd: 2.2240, target_std: 6.5150
    Epoch [3/50], Val Losses: mse: 7.6550, mae: 1.4198, huber: 1.0659, swd: 1.1561, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 12.1238, mae: 1.7971, huber: 1.4321, swd: 2.2301, target_std: 4.7640
      Epoch 3 composite train-obj: 1.195576
            Val objective improved 1.0731 → 1.0659, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.4731, mae: 1.5485, huber: 1.1848, swd: 2.1927, target_std: 6.5149
    Epoch [4/50], Val Losses: mse: 7.7244, mae: 1.4278, huber: 1.0731, swd: 1.1586, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 12.0820, mae: 1.8073, huber: 1.4401, swd: 2.3090, target_std: 4.7640
      Epoch 4 composite train-obj: 1.184835
            No improvement (1.0731), counter 1/5
    Epoch [5/50], Train Losses: mse: 7.3851, mae: 1.5389, huber: 1.1757, swd: 2.1595, target_std: 6.5149
    Epoch [5/50], Val Losses: mse: 7.6172, mae: 1.4212, huber: 1.0665, swd: 1.0614, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 11.9776, mae: 1.7944, huber: 1.4277, swd: 2.1642, target_std: 4.7640
      Epoch 5 composite train-obj: 1.175698
            No improvement (1.0665), counter 2/5
    Epoch [6/50], Train Losses: mse: 7.3115, mae: 1.5313, huber: 1.1684, swd: 2.1374, target_std: 6.5149
    Epoch [6/50], Val Losses: mse: 7.5253, mae: 1.4131, huber: 1.0594, swd: 1.1416, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 11.8724, mae: 1.7974, huber: 1.4302, swd: 2.3090, target_std: 4.7640
      Epoch 6 composite train-obj: 1.168367
            Val objective improved 1.0659 → 1.0594, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.2423, mae: 1.5241, huber: 1.1614, swd: 2.1118, target_std: 6.5147
    Epoch [7/50], Val Losses: mse: 7.7931, mae: 1.4385, huber: 1.0831, swd: 1.1164, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 11.9662, mae: 1.7977, huber: 1.4304, swd: 2.1366, target_std: 4.7640
      Epoch 7 composite train-obj: 1.161352
            No improvement (1.0831), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.1711, mae: 1.5170, huber: 1.1545, swd: 2.0890, target_std: 6.5151
    Epoch [8/50], Val Losses: mse: 7.8960, mae: 1.4475, huber: 1.0919, swd: 1.1496, target_std: 4.2820
    Epoch [8/50], Test Losses: mse: 11.8920, mae: 1.7974, huber: 1.4298, swd: 2.1715, target_std: 4.7640
      Epoch 8 composite train-obj: 1.154474
            No improvement (1.0919), counter 2/5
    Epoch [9/50], Train Losses: mse: 7.1275, mae: 1.5119, huber: 1.1495, swd: 2.0671, target_std: 6.5151
    Epoch [9/50], Val Losses: mse: 7.8270, mae: 1.4425, huber: 1.0876, swd: 1.1975, target_std: 4.2820
    Epoch [9/50], Test Losses: mse: 11.8165, mae: 1.7958, huber: 1.4285, swd: 2.2868, target_std: 4.7640
      Epoch 9 composite train-obj: 1.149467
            No improvement (1.0876), counter 3/5
    Epoch [10/50], Train Losses: mse: 7.0685, mae: 1.5062, huber: 1.1439, swd: 2.0490, target_std: 6.5147
    Epoch [10/50], Val Losses: mse: 7.8190, mae: 1.4363, huber: 1.0810, swd: 1.2712, target_std: 4.2820
    Epoch [10/50], Test Losses: mse: 11.8638, mae: 1.8083, huber: 1.4394, swd: 2.3953, target_std: 4.7640
      Epoch 10 composite train-obj: 1.143949
            No improvement (1.0810), counter 4/5
    Epoch [11/50], Train Losses: mse: 7.0192, mae: 1.5006, huber: 1.1385, swd: 2.0266, target_std: 6.5147
    Epoch [11/50], Val Losses: mse: 7.9569, mae: 1.4546, huber: 1.0992, swd: 1.1285, target_std: 4.2820
    Epoch [11/50], Test Losses: mse: 11.9069, mae: 1.7939, huber: 1.4266, swd: 2.0940, target_std: 4.7640
      Epoch 11 composite train-obj: 1.138457
    Epoch [11/50], Test Losses: mse: 11.8724, mae: 1.7974, huber: 1.4302, swd: 2.3090, target_std: 4.7640
    Best round's Test MSE: 11.8724, MAE: 1.7974, SWD: 2.3090
    Best round's Validation MSE: 7.5253, MAE: 1.4131
    Best round's Test verification MSE : 11.8724, MAE: 1.7974, SWD: 2.3090
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.1194, mae: 1.6223, huber: 1.2552, swd: 2.2155, target_std: 6.5148
    Epoch [1/50], Val Losses: mse: 7.6779, mae: 1.4331, huber: 1.0787, swd: 1.1094, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 12.3556, mae: 1.8112, huber: 1.4460, swd: 2.1960, target_std: 4.7640
      Epoch 1 composite train-obj: 1.255173
            Val objective improved inf → 1.0787, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.7250, mae: 1.5765, huber: 1.2118, swd: 2.1557, target_std: 6.5149
    Epoch [2/50], Val Losses: mse: 7.5966, mae: 1.4247, huber: 1.0698, swd: 1.1318, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 12.0953, mae: 1.7949, huber: 1.4295, swd: 2.2360, target_std: 4.7640
      Epoch 2 composite train-obj: 1.211796
            Val objective improved 1.0787 → 1.0698, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.5875, mae: 1.5611, huber: 1.1972, swd: 2.1181, target_std: 6.5149
    Epoch [3/50], Val Losses: mse: 7.5548, mae: 1.4171, huber: 1.0628, swd: 1.2467, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 12.2179, mae: 1.8382, huber: 1.4695, swd: 2.6502, target_std: 4.7640
      Epoch 3 composite train-obj: 1.197159
            Val objective improved 1.0698 → 1.0628, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.4661, mae: 1.5477, huber: 1.1843, swd: 2.0816, target_std: 6.5151
    Epoch [4/50], Val Losses: mse: 7.7448, mae: 1.4314, huber: 1.0765, swd: 1.0937, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 12.2004, mae: 1.8014, huber: 1.4341, swd: 2.1441, target_std: 4.7640
      Epoch 4 composite train-obj: 1.184339
            No improvement (1.0765), counter 1/5
    Epoch [5/50], Train Losses: mse: 7.3881, mae: 1.5393, huber: 1.1763, swd: 2.0570, target_std: 6.5148
    Epoch [5/50], Val Losses: mse: 7.5956, mae: 1.4183, huber: 1.0648, swd: 1.0891, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 12.0352, mae: 1.7956, huber: 1.4288, swd: 2.2142, target_std: 4.7640
      Epoch 5 composite train-obj: 1.176346
            No improvement (1.0648), counter 2/5
    Epoch [6/50], Train Losses: mse: 7.3086, mae: 1.5309, huber: 1.1682, swd: 2.0302, target_std: 6.5150
    Epoch [6/50], Val Losses: mse: 7.7991, mae: 1.4394, huber: 1.0845, swd: 1.2746, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 12.0733, mae: 1.8223, huber: 1.4538, swd: 2.4772, target_std: 4.7640
      Epoch 6 composite train-obj: 1.168154
            No improvement (1.0845), counter 3/5
    Epoch [7/50], Train Losses: mse: 7.2350, mae: 1.5231, huber: 1.1607, swd: 2.0066, target_std: 6.5146
    Epoch [7/50], Val Losses: mse: 7.7179, mae: 1.4310, huber: 1.0767, swd: 1.1387, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 12.0579, mae: 1.8081, huber: 1.4408, swd: 2.3527, target_std: 4.7640
      Epoch 7 composite train-obj: 1.160650
            No improvement (1.0767), counter 4/5
    Epoch [8/50], Train Losses: mse: 7.1648, mae: 1.5159, huber: 1.1537, swd: 1.9839, target_std: 6.5151
    Epoch [8/50], Val Losses: mse: 7.7538, mae: 1.4326, huber: 1.0780, swd: 1.1141, target_std: 4.2820
    Epoch [8/50], Test Losses: mse: 12.0542, mae: 1.8030, huber: 1.4360, swd: 2.2510, target_std: 4.7640
      Epoch 8 composite train-obj: 1.153657
    Epoch [8/50], Test Losses: mse: 12.2179, mae: 1.8382, huber: 1.4695, swd: 2.6502, target_std: 4.7640
    Best round's Test MSE: 12.2179, MAE: 1.8382, SWD: 2.6502
    Best round's Validation MSE: 7.5548, MAE: 1.4171
    Best round's Test verification MSE : 12.2179, MAE: 1.8382, SWD: 2.6502
    
    ==================================================
    Experiment Summary (PatchTST_ettm1_seq96_pred336_20250430_0433)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.1807 ± 0.2380
      mae: 1.8185 ± 0.0167
      huber: 1.4511 ± 0.0161
      swd: 2.3746 ± 0.2037
      target_std: 4.7640 ± 0.0000
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.5608 ± 0.0317
      mae: 1.4163 ± 0.0023
      huber: 1.0623 ± 0.0022
      swd: 1.1469 ± 0.0794
      target_std: 4.2820 ± 0.0000
      count: 52.0000 ± 0.0000
    ==================================================
    
    Experiment complete: PatchTST_ettm1_seq96_pred336_20250430_0433
    Model: PatchTST
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 96-720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm1']['channels'],
    enc_in=data_mgr.datasets['ettm1']['channels'],
    dec_in=data_mgr.datasets['ettm1']['channels'],
    c_out=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp_tst_96_720 = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 9.1548, mae: 1.7691, huber: 1.3916, swd: 2.4476, target_std: 6.5152
    Epoch [1/50], Val Losses: mse: 9.5554, mae: 1.5925, huber: 1.2296, swd: 1.2375, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 13.5447, mae: 1.9955, huber: 1.6172, swd: 2.6426, target_std: 4.7646
      Epoch 1 composite train-obj: 1.391634
            Val objective improved inf → 1.2296, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.7337, mae: 1.7268, huber: 1.3511, swd: 2.4012, target_std: 6.5149
    Epoch [2/50], Val Losses: mse: 9.2305, mae: 1.5636, huber: 1.2012, swd: 1.1910, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 13.3736, mae: 1.9809, huber: 1.6028, swd: 2.5645, target_std: 4.7646
      Epoch 2 composite train-obj: 1.351082
            Val objective improved 1.2296 → 1.2012, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.5915, mae: 1.7133, huber: 1.3381, swd: 2.3673, target_std: 6.5150
    Epoch [3/50], Val Losses: mse: 9.5136, mae: 1.5947, huber: 1.2306, swd: 1.2134, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 13.0619, mae: 1.9591, huber: 1.5812, swd: 2.3916, target_std: 4.7646
      Epoch 3 composite train-obj: 1.338087
            No improvement (1.2306), counter 1/5
    Epoch [4/50], Train Losses: mse: 8.4731, mae: 1.7023, huber: 1.3274, swd: 2.3279, target_std: 6.5152
    Epoch [4/50], Val Losses: mse: 9.5464, mae: 1.5912, huber: 1.2278, swd: 1.2495, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 13.1006, mae: 1.9672, huber: 1.5890, swd: 2.4820, target_std: 4.7646
      Epoch 4 composite train-obj: 1.327399
            No improvement (1.2278), counter 2/5
    Epoch [5/50], Train Losses: mse: 8.3775, mae: 1.6936, huber: 1.3189, swd: 2.2967, target_std: 6.5151
    Epoch [5/50], Val Losses: mse: 9.3094, mae: 1.5756, huber: 1.2123, swd: 1.2860, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 12.9795, mae: 1.9621, huber: 1.5830, swd: 2.5176, target_std: 4.7646
      Epoch 5 composite train-obj: 1.318873
            No improvement (1.2123), counter 3/5
    Epoch [6/50], Train Losses: mse: 8.2943, mae: 1.6854, huber: 1.3108, swd: 2.2704, target_std: 6.5148
    Epoch [6/50], Val Losses: mse: 9.4541, mae: 1.5855, huber: 1.2230, swd: 1.2118, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 12.8977, mae: 1.9499, huber: 1.5723, swd: 2.3328, target_std: 4.7646
      Epoch 6 composite train-obj: 1.310828
            No improvement (1.2230), counter 4/5
    Epoch [7/50], Train Losses: mse: 8.2241, mae: 1.6784, huber: 1.3039, swd: 2.2430, target_std: 6.5152
    Epoch [7/50], Val Losses: mse: 9.5948, mae: 1.5956, huber: 1.2314, swd: 1.2384, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 12.9250, mae: 1.9593, huber: 1.5793, swd: 2.3791, target_std: 4.7646
      Epoch 7 composite train-obj: 1.303896
    Epoch [7/50], Test Losses: mse: 13.3736, mae: 1.9809, huber: 1.6028, swd: 2.5645, target_std: 4.7646
    Best round's Test MSE: 13.3736, MAE: 1.9809, SWD: 2.5645
    Best round's Validation MSE: 9.2305, MAE: 1.5636
    Best round's Test verification MSE : 13.3736, MAE: 1.9809, SWD: 2.5645
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.1395, mae: 1.7682, huber: 1.3908, swd: 2.3280, target_std: 6.5156
    Epoch [1/50], Val Losses: mse: 9.4455, mae: 1.5779, huber: 1.2161, swd: 1.1908, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 13.7525, mae: 1.9943, huber: 1.6170, swd: 2.5452, target_std: 4.7646
      Epoch 1 composite train-obj: 1.390786
            Val objective improved inf → 1.2161, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.7484, mae: 1.7268, huber: 1.3512, swd: 2.2842, target_std: 6.5151
    Epoch [2/50], Val Losses: mse: 9.5556, mae: 1.5901, huber: 1.2270, swd: 1.1957, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 13.5115, mae: 1.9852, huber: 1.6065, swd: 2.5050, target_std: 4.7646
      Epoch 2 composite train-obj: 1.351186
            No improvement (1.2270), counter 1/5
    Epoch [3/50], Train Losses: mse: 8.5861, mae: 1.7120, huber: 1.3368, swd: 2.2398, target_std: 6.5152
    Epoch [3/50], Val Losses: mse: 9.1870, mae: 1.5553, huber: 1.1938, swd: 1.2945, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 13.4041, mae: 1.9895, huber: 1.6108, swd: 2.8811, target_std: 4.7646
      Epoch 3 composite train-obj: 1.336760
            Val objective improved 1.2161 → 1.1938, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.4599, mae: 1.7004, huber: 1.3254, swd: 2.2057, target_std: 6.5152
    Epoch [4/50], Val Losses: mse: 9.7193, mae: 1.5971, huber: 1.2345, swd: 1.0970, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 12.9200, mae: 1.9432, huber: 1.5658, swd: 2.2659, target_std: 4.7646
      Epoch 4 composite train-obj: 1.325421
            No improvement (1.2345), counter 1/5
    Epoch [5/50], Train Losses: mse: 8.3627, mae: 1.6917, huber: 1.3169, swd: 2.1746, target_std: 6.5151
    Epoch [5/50], Val Losses: mse: 9.5670, mae: 1.5884, huber: 1.2257, swd: 1.1126, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 12.9225, mae: 1.9444, huber: 1.5663, swd: 2.2109, target_std: 4.7646
      Epoch 5 composite train-obj: 1.316944
            No improvement (1.2257), counter 2/5
    Epoch [6/50], Train Losses: mse: 8.2732, mae: 1.6830, huber: 1.3085, swd: 2.1437, target_std: 6.5149
    Epoch [6/50], Val Losses: mse: 9.3418, mae: 1.5825, huber: 1.2197, swd: 1.1463, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 12.6562, mae: 1.9402, huber: 1.5618, swd: 2.3322, target_std: 4.7646
      Epoch 6 composite train-obj: 1.308463
            No improvement (1.2197), counter 3/5
    Epoch [7/50], Train Losses: mse: 8.1832, mae: 1.6742, huber: 1.2998, swd: 2.1125, target_std: 6.5149
    Epoch [7/50], Val Losses: mse: 9.2442, mae: 1.5660, huber: 1.2042, swd: 1.1713, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 12.9231, mae: 1.9557, huber: 1.5777, swd: 2.5218, target_std: 4.7646
      Epoch 7 composite train-obj: 1.299838
            No improvement (1.2042), counter 4/5
    Epoch [8/50], Train Losses: mse: 8.1320, mae: 1.6686, huber: 1.2945, swd: 2.0937, target_std: 6.5150
    Epoch [8/50], Val Losses: mse: 9.3425, mae: 1.5851, huber: 1.2211, swd: 1.2742, target_std: 4.2630
    Epoch [8/50], Test Losses: mse: 12.6507, mae: 1.9542, huber: 1.5746, swd: 2.5423, target_std: 4.7646
      Epoch 8 composite train-obj: 1.294478
    Epoch [8/50], Test Losses: mse: 13.4041, mae: 1.9895, huber: 1.6108, swd: 2.8811, target_std: 4.7646
    Best round's Test MSE: 13.4041, MAE: 1.9895, SWD: 2.8811
    Best round's Validation MSE: 9.1870, MAE: 1.5553
    Best round's Test verification MSE : 13.4041, MAE: 1.9895, SWD: 2.8811
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.1729, mae: 1.7697, huber: 1.3924, swd: 2.5544, target_std: 6.5153
    Epoch [1/50], Val Losses: mse: 9.6718, mae: 1.5897, huber: 1.2280, swd: 1.1480, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 13.6997, mae: 1.9701, huber: 1.5950, swd: 2.3107, target_std: 4.7646
      Epoch 1 composite train-obj: 1.392368
            Val objective improved inf → 1.2280, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.7500, mae: 1.7276, huber: 1.3519, swd: 2.4982, target_std: 6.5152
    Epoch [2/50], Val Losses: mse: 9.2870, mae: 1.5658, huber: 1.2042, swd: 1.1349, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 13.3781, mae: 1.9580, huber: 1.5820, swd: 2.3405, target_std: 4.7646
      Epoch 2 composite train-obj: 1.351920
            Val objective improved 1.2280 → 1.2042, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.5892, mae: 1.7124, huber: 1.3373, swd: 2.4570, target_std: 6.5150
    Epoch [3/50], Val Losses: mse: 9.5054, mae: 1.5856, huber: 1.2231, swd: 1.1973, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 13.0179, mae: 1.9431, huber: 1.5667, swd: 2.3584, target_std: 4.7646
      Epoch 3 composite train-obj: 1.337346
            No improvement (1.2231), counter 1/5
    Epoch [4/50], Train Losses: mse: 8.4544, mae: 1.7005, huber: 1.3258, swd: 2.4157, target_std: 6.5152
    Epoch [4/50], Val Losses: mse: 9.4641, mae: 1.5862, huber: 1.2229, swd: 1.2059, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 12.9148, mae: 1.9418, huber: 1.5646, swd: 2.4249, target_std: 4.7646
      Epoch 4 composite train-obj: 1.325811
            No improvement (1.2229), counter 2/5
    Epoch [5/50], Train Losses: mse: 8.3522, mae: 1.6906, huber: 1.3162, swd: 2.3817, target_std: 6.5150
    Epoch [5/50], Val Losses: mse: 9.5291, mae: 1.5981, huber: 1.2343, swd: 1.2467, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 12.8682, mae: 1.9487, huber: 1.5707, swd: 2.4490, target_std: 4.7646
      Epoch 5 composite train-obj: 1.316175
            No improvement (1.2343), counter 3/5
    Epoch [6/50], Train Losses: mse: 8.2794, mae: 1.6833, huber: 1.3091, swd: 2.3565, target_std: 6.5154
    Epoch [6/50], Val Losses: mse: 9.5751, mae: 1.5977, huber: 1.2348, swd: 1.2874, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 12.8753, mae: 1.9473, huber: 1.5693, swd: 2.3941, target_std: 4.7646
      Epoch 6 composite train-obj: 1.309097
            No improvement (1.2348), counter 4/5
    Epoch [7/50], Train Losses: mse: 8.1990, mae: 1.6751, huber: 1.3011, swd: 2.3266, target_std: 6.5150
    Epoch [7/50], Val Losses: mse: 9.7130, mae: 1.6162, huber: 1.2516, swd: 1.2520, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 12.9494, mae: 1.9607, huber: 1.5815, swd: 2.3265, target_std: 4.7646
      Epoch 7 composite train-obj: 1.301061
    Epoch [7/50], Test Losses: mse: 13.3781, mae: 1.9580, huber: 1.5820, swd: 2.3405, target_std: 4.7646
    Best round's Test MSE: 13.3781, MAE: 1.9580, SWD: 2.3405
    Best round's Validation MSE: 9.2870, MAE: 1.5658
    Best round's Test verification MSE : 13.3781, MAE: 1.9580, SWD: 2.3405
    
    ==================================================
    Experiment Summary (PatchTST_ettm1_seq96_pred720_20250430_0436)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 13.3853 ± 0.0135
      mae: 1.9762 ± 0.0133
      huber: 1.5986 ± 0.0121
      swd: 2.5954 ± 0.2217
      target_std: 4.7646 ± 0.0000
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.2348 ± 0.0409
      mae: 1.5616 ± 0.0045
      huber: 1.1997 ± 0.0044
      swd: 1.2068 ± 0.0661
      target_std: 4.2630 ± 0.0000
      count: 49.0000 ± 0.0000
    ==================================================
    
    Experiment complete: PatchTST_ettm1_seq96_pred720_20250430_0436
    Model: PatchTST
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### DLinear

#### 96-96


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_dlinear_96_96 = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([48776, 7])
    Shape of validation data: torch.Size([6968, 7])
    Shape of testing data: torch.Size([13936, 7])
    global_std.shape: torch.Size([7])
    Global Std for ettm1: tensor([7.0829, 2.0413, 6.8291, 1.8072, 1.1741, 0.6004, 8.5648],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([13936, 7]), torch.Size([13936, 7])
    Number of batches in train_loader: 380
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 7.3044, mae: 1.4391, huber: 1.0809, swd: 2.0658, ept: 62.1624
    Epoch [1/50], Val Losses: mse: 5.4640, mae: 1.1810, huber: 0.8383, swd: 0.9294, ept: 70.2403
    Epoch [1/50], Test Losses: mse: 8.5075, mae: 1.4299, huber: 1.0769, swd: 1.3370, ept: 59.7509
      Epoch 1 composite train-obj: 1.080913
            Val objective improved inf → 0.8383, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.0328, mae: 1.2281, huber: 0.8834, swd: 1.5132, ept: 70.1866
    Epoch [2/50], Val Losses: mse: 5.3717, mae: 1.1744, huber: 0.8334, swd: 0.9771, ept: 70.8050
    Epoch [2/50], Test Losses: mse: 8.2945, mae: 1.4057, huber: 1.0566, swd: 1.3523, ept: 60.3153
      Epoch 2 composite train-obj: 0.883352
            Val objective improved 0.8383 → 0.8334, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.9896, mae: 1.2216, huber: 0.8775, swd: 1.4907, ept: 70.4401
    Epoch [3/50], Val Losses: mse: 5.3171, mae: 1.1690, huber: 0.8285, swd: 0.9834, ept: 71.1348
    Epoch [3/50], Test Losses: mse: 8.2344, mae: 1.4043, huber: 1.0545, swd: 1.3638, ept: 60.5246
      Epoch 3 composite train-obj: 0.877519
            Val objective improved 0.8334 → 0.8285, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.9659, mae: 1.2188, huber: 0.8749, swd: 1.4789, ept: 70.5236
    Epoch [4/50], Val Losses: mse: 5.3432, mae: 1.1705, huber: 0.8299, swd: 0.9641, ept: 71.1036
    Epoch [4/50], Test Losses: mse: 8.2328, mae: 1.4008, huber: 1.0511, swd: 1.3226, ept: 60.5726
      Epoch 4 composite train-obj: 0.874905
            No improvement (0.8299), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.9606, mae: 1.2191, huber: 0.8750, swd: 1.4752, ept: 70.5496
    Epoch [5/50], Val Losses: mse: 5.3265, mae: 1.1688, huber: 0.8283, swd: 0.9614, ept: 71.5091
    Epoch [5/50], Test Losses: mse: 8.2278, mae: 1.4003, huber: 1.0503, swd: 1.3169, ept: 60.7399
      Epoch 5 composite train-obj: 0.874981
            Val objective improved 0.8285 → 0.8283, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.9510, mae: 1.2182, huber: 0.8741, swd: 1.4680, ept: 70.5491
    Epoch [6/50], Val Losses: mse: 5.3261, mae: 1.1692, huber: 0.8282, swd: 0.9554, ept: 71.1212
    Epoch [6/50], Test Losses: mse: 8.2660, mae: 1.4042, huber: 1.0549, swd: 1.3250, ept: 60.4728
      Epoch 6 composite train-obj: 0.874060
            Val objective improved 0.8283 → 0.8282, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.9440, mae: 1.2178, huber: 0.8737, swd: 1.4660, ept: 70.5800
    Epoch [7/50], Val Losses: mse: 5.3231, mae: 1.1697, huber: 0.8289, swd: 0.9588, ept: 71.1023
    Epoch [7/50], Test Losses: mse: 8.2497, mae: 1.4024, huber: 1.0535, swd: 1.3247, ept: 60.6215
      Epoch 7 composite train-obj: 0.873666
            No improvement (0.8289), counter 1/5
    Epoch [8/50], Train Losses: mse: 4.9486, mae: 1.2182, huber: 0.8741, swd: 1.4668, ept: 70.5494
    Epoch [8/50], Val Losses: mse: 5.3317, mae: 1.1761, huber: 0.8341, swd: 0.9870, ept: 70.8547
    Epoch [8/50], Test Losses: mse: 8.2321, mae: 1.4040, huber: 1.0556, swd: 1.3513, ept: 60.5068
      Epoch 8 composite train-obj: 0.874056
            No improvement (0.8341), counter 2/5
    Epoch [9/50], Train Losses: mse: 4.9423, mae: 1.2178, huber: 0.8737, swd: 1.4636, ept: 70.6001
    Epoch [9/50], Val Losses: mse: 5.3301, mae: 1.1686, huber: 0.8280, swd: 0.9331, ept: 71.2368
    Epoch [9/50], Test Losses: mse: 8.2442, mae: 1.4013, huber: 1.0521, swd: 1.2937, ept: 60.6585
      Epoch 9 composite train-obj: 0.873681
            Val objective improved 0.8282 → 0.8280, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 4.9406, mae: 1.2175, huber: 0.8734, swd: 1.4612, ept: 70.5746
    Epoch [10/50], Val Losses: mse: 5.3154, mae: 1.1737, huber: 0.8323, swd: 0.9846, ept: 70.7546
    Epoch [10/50], Test Losses: mse: 8.1909, mae: 1.4005, huber: 1.0518, swd: 1.3367, ept: 60.4329
      Epoch 10 composite train-obj: 0.873351
            No improvement (0.8323), counter 1/5
    Epoch [11/50], Train Losses: mse: 4.9426, mae: 1.2179, huber: 0.8737, swd: 1.4632, ept: 70.5832
    Epoch [11/50], Val Losses: mse: 5.3365, mae: 1.1776, huber: 0.8362, swd: 0.9935, ept: 70.7706
    Epoch [11/50], Test Losses: mse: 8.1920, mae: 1.4029, huber: 1.0529, swd: 1.3432, ept: 60.4274
      Epoch 11 composite train-obj: 0.873701
            No improvement (0.8362), counter 2/5
    Epoch [12/50], Train Losses: mse: 4.9375, mae: 1.2178, huber: 0.8735, swd: 1.4617, ept: 70.5710
    Epoch [12/50], Val Losses: mse: 5.3449, mae: 1.1747, huber: 0.8338, swd: 0.9636, ept: 71.2574
    Epoch [12/50], Test Losses: mse: 8.2617, mae: 1.4050, huber: 1.0558, swd: 1.3192, ept: 60.6164
      Epoch 12 composite train-obj: 0.873515
            No improvement (0.8338), counter 3/5
    Epoch [13/50], Train Losses: mse: 4.9390, mae: 1.2177, huber: 0.8735, swd: 1.4610, ept: 70.5655
    Epoch [13/50], Val Losses: mse: 5.3403, mae: 1.1706, huber: 0.8299, swd: 0.9364, ept: 71.3984
    Epoch [13/50], Test Losses: mse: 8.2780, mae: 1.4053, huber: 1.0553, swd: 1.2932, ept: 60.6045
      Epoch 13 composite train-obj: 0.873501
            No improvement (0.8299), counter 4/5
    Epoch [14/50], Train Losses: mse: 4.9421, mae: 1.2185, huber: 0.8742, swd: 1.4618, ept: 70.5679
    Epoch [14/50], Val Losses: mse: 5.3132, mae: 1.1675, huber: 0.8275, swd: 0.9472, ept: 71.4749
    Epoch [14/50], Test Losses: mse: 8.2152, mae: 1.3978, huber: 1.0491, swd: 1.2984, ept: 60.6666
      Epoch 14 composite train-obj: 0.874164
            Val objective improved 0.8280 → 0.8275, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 4.9384, mae: 1.2175, huber: 0.8734, swd: 1.4594, ept: 70.5682
    Epoch [15/50], Val Losses: mse: 5.3189, mae: 1.1693, huber: 0.8286, swd: 0.9567, ept: 71.5808
    Epoch [15/50], Test Losses: mse: 8.2406, mae: 1.4028, huber: 1.0529, swd: 1.3216, ept: 60.6612
      Epoch 15 composite train-obj: 0.873383
            No improvement (0.8286), counter 1/5
    Epoch [16/50], Train Losses: mse: 4.9384, mae: 1.2179, huber: 0.8736, swd: 1.4604, ept: 70.5588
    Epoch [16/50], Val Losses: mse: 5.3542, mae: 1.1773, huber: 0.8363, swd: 0.9741, ept: 71.5354
    Epoch [16/50], Test Losses: mse: 8.2684, mae: 1.4099, huber: 1.0598, swd: 1.3406, ept: 60.6219
      Epoch 16 composite train-obj: 0.873563
            No improvement (0.8363), counter 2/5
    Epoch [17/50], Train Losses: mse: 4.9419, mae: 1.2184, huber: 0.8741, swd: 1.4629, ept: 70.5940
    Epoch [17/50], Val Losses: mse: 5.3329, mae: 1.1718, huber: 0.8309, swd: 0.9501, ept: 71.6046
    Epoch [17/50], Test Losses: mse: 8.2447, mae: 1.4033, huber: 1.0533, swd: 1.3087, ept: 60.7097
      Epoch 17 composite train-obj: 0.874149
            No improvement (0.8309), counter 3/5
    Epoch [18/50], Train Losses: mse: 4.9362, mae: 1.2176, huber: 0.8734, swd: 1.4603, ept: 70.5770
    Epoch [18/50], Val Losses: mse: 5.3120, mae: 1.1687, huber: 0.8280, swd: 0.9549, ept: 71.0528
    Epoch [18/50], Test Losses: mse: 8.1948, mae: 1.3981, huber: 1.0494, swd: 1.2947, ept: 60.5334
      Epoch 18 composite train-obj: 0.873353
            No improvement (0.8280), counter 4/5
    Epoch [19/50], Train Losses: mse: 4.9393, mae: 1.2179, huber: 0.8736, swd: 1.4612, ept: 70.5503
    Epoch [19/50], Val Losses: mse: 5.3404, mae: 1.1729, huber: 0.8318, swd: 0.9515, ept: 71.4125
    Epoch [19/50], Test Losses: mse: 8.2101, mae: 1.3982, huber: 1.0487, swd: 1.2808, ept: 60.6003
      Epoch 19 composite train-obj: 0.873606
    Epoch [19/50], Test Losses: mse: 8.2152, mae: 1.3978, huber: 1.0491, swd: 1.2984, ept: 60.6666
    Best round's Test MSE: 8.2152, MAE: 1.3978, SWD: 1.2984
    Best round's Validation MSE: 5.3132, MAE: 1.1675, SWD: 0.9472
    Best round's Test verification MSE : 8.2152, MAE: 1.3978, SWD: 1.2984
    Time taken: 75.53 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.2359, mae: 1.4331, huber: 1.0755, swd: 2.0298, ept: 62.4547
    Epoch [1/50], Val Losses: mse: 5.4492, mae: 1.1822, huber: 0.8398, swd: 0.9529, ept: 70.3794
    Epoch [1/50], Test Losses: mse: 8.4639, mae: 1.4263, huber: 1.0740, swd: 1.3946, ept: 59.7593
      Epoch 1 composite train-obj: 1.075525
            Val objective improved inf → 0.8398, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.0349, mae: 1.2280, huber: 0.8833, swd: 1.4925, ept: 70.1993
    Epoch [2/50], Val Losses: mse: 5.3698, mae: 1.1656, huber: 0.8252, swd: 0.9135, ept: 70.9031
    Epoch [2/50], Test Losses: mse: 8.3813, mae: 1.4102, huber: 1.0601, swd: 1.3511, ept: 60.4490
      Epoch 2 composite train-obj: 0.883346
            Val objective improved 0.8398 → 0.8252, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.9885, mae: 1.2211, huber: 0.8772, swd: 1.4687, ept: 70.4616
    Epoch [3/50], Val Losses: mse: 5.3603, mae: 1.1745, huber: 0.8330, swd: 0.9775, ept: 70.0446
    Epoch [3/50], Test Losses: mse: 8.2825, mae: 1.4042, huber: 1.0557, swd: 1.3824, ept: 60.1333
      Epoch 3 composite train-obj: 0.877199
            No improvement (0.8330), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.9678, mae: 1.2190, huber: 0.8751, swd: 1.4571, ept: 70.5162
    Epoch [4/50], Val Losses: mse: 5.3319, mae: 1.1697, huber: 0.8291, swd: 0.9647, ept: 71.0849
    Epoch [4/50], Test Losses: mse: 8.2290, mae: 1.4012, huber: 1.0516, swd: 1.3674, ept: 60.5513
      Epoch 4 composite train-obj: 0.875084
            No improvement (0.8291), counter 2/5
    Epoch [5/50], Train Losses: mse: 4.9562, mae: 1.2182, huber: 0.8742, swd: 1.4514, ept: 70.5479
    Epoch [5/50], Val Losses: mse: 5.3318, mae: 1.1689, huber: 0.8286, swd: 0.9540, ept: 71.1607
    Epoch [5/50], Test Losses: mse: 8.2492, mae: 1.4019, huber: 1.0526, swd: 1.3616, ept: 60.4970
      Epoch 5 composite train-obj: 0.874230
            No improvement (0.8286), counter 3/5
    Epoch [6/50], Train Losses: mse: 4.9513, mae: 1.2184, huber: 0.8742, swd: 1.4477, ept: 70.5526
    Epoch [6/50], Val Losses: mse: 5.3147, mae: 1.1719, huber: 0.8306, swd: 0.9604, ept: 71.4284
    Epoch [6/50], Test Losses: mse: 8.2050, mae: 1.4021, huber: 1.0514, swd: 1.3518, ept: 60.5861
      Epoch 6 composite train-obj: 0.874230
            No improvement (0.8306), counter 4/5
    Epoch [7/50], Train Losses: mse: 4.9492, mae: 1.2182, huber: 0.8741, swd: 1.4459, ept: 70.5671
    Epoch [7/50], Val Losses: mse: 5.3191, mae: 1.1718, huber: 0.8311, swd: 0.9655, ept: 71.4425
    Epoch [7/50], Test Losses: mse: 8.2161, mae: 1.4023, huber: 1.0530, swd: 1.3543, ept: 60.7601
      Epoch 7 composite train-obj: 0.874066
    Epoch [7/50], Test Losses: mse: 8.3813, mae: 1.4102, huber: 1.0601, swd: 1.3511, ept: 60.4490
    Best round's Test MSE: 8.3813, MAE: 1.4102, SWD: 1.3511
    Best round's Validation MSE: 5.3698, MAE: 1.1656, SWD: 0.9135
    Best round's Test verification MSE : 8.3813, MAE: 1.4102, SWD: 1.3511
    Time taken: 27.80 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.1655, mae: 1.4331, huber: 1.0752, swd: 1.8373, ept: 62.1070
    Epoch [1/50], Val Losses: mse: 5.4312, mae: 1.1851, huber: 0.8423, swd: 0.9233, ept: 70.4421
    Epoch [1/50], Test Losses: mse: 8.4280, mae: 1.4297, huber: 1.0765, swd: 1.3268, ept: 59.7221
      Epoch 1 composite train-obj: 1.075181
            Val objective improved inf → 0.8423, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.0321, mae: 1.2276, huber: 0.8830, swd: 1.3897, ept: 70.1815
    Epoch [2/50], Val Losses: mse: 5.3553, mae: 1.1699, huber: 0.8290, swd: 0.9083, ept: 70.9209
    Epoch [2/50], Test Losses: mse: 8.3102, mae: 1.4084, huber: 1.0585, swd: 1.2855, ept: 60.2929
      Epoch 2 composite train-obj: 0.883014
            Val objective improved 0.8423 → 0.8290, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.9839, mae: 1.2204, huber: 0.8764, swd: 1.3661, ept: 70.4595
    Epoch [3/50], Val Losses: mse: 5.3582, mae: 1.1668, huber: 0.8265, swd: 0.8728, ept: 71.2344
    Epoch [3/50], Test Losses: mse: 8.3215, mae: 1.4049, huber: 1.0548, swd: 1.2335, ept: 60.5195
      Epoch 3 composite train-obj: 0.876434
            Val objective improved 0.8290 → 0.8265, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.9697, mae: 1.2195, huber: 0.8755, swd: 1.3596, ept: 70.5310
    Epoch [4/50], Val Losses: mse: 5.3242, mae: 1.1636, huber: 0.8235, swd: 0.8747, ept: 71.2646
    Epoch [4/50], Test Losses: mse: 8.2795, mae: 1.4033, huber: 1.0534, swd: 1.2349, ept: 60.5134
      Epoch 4 composite train-obj: 0.875539
            Val objective improved 0.8265 → 0.8235, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.9581, mae: 1.2188, huber: 0.8747, swd: 1.3522, ept: 70.5418
    Epoch [5/50], Val Losses: mse: 5.3197, mae: 1.1666, huber: 0.8263, swd: 0.8988, ept: 71.3281
    Epoch [5/50], Test Losses: mse: 8.2570, mae: 1.4011, huber: 1.0524, swd: 1.2582, ept: 60.6372
      Epoch 5 composite train-obj: 0.874688
            No improvement (0.8263), counter 1/5
    Epoch [6/50], Train Losses: mse: 4.9511, mae: 1.2181, huber: 0.8741, swd: 1.3487, ept: 70.5705
    Epoch [6/50], Val Losses: mse: 5.3507, mae: 1.1819, huber: 0.8392, swd: 0.9423, ept: 70.3592
    Epoch [6/50], Test Losses: mse: 8.2044, mae: 1.4005, huber: 1.0527, swd: 1.2822, ept: 60.2810
      Epoch 6 composite train-obj: 0.874091
            No improvement (0.8392), counter 2/5
    Epoch [7/50], Train Losses: mse: 4.9458, mae: 1.2178, huber: 0.8737, swd: 1.3464, ept: 70.5529
    Epoch [7/50], Val Losses: mse: 5.3334, mae: 1.1685, huber: 0.8274, swd: 0.8784, ept: 70.9890
    Epoch [7/50], Test Losses: mse: 8.2790, mae: 1.4031, huber: 1.0539, swd: 1.2298, ept: 60.5980
      Epoch 7 composite train-obj: 0.873661
            No improvement (0.8274), counter 3/5
    Epoch [8/50], Train Losses: mse: 4.9494, mae: 1.2188, huber: 0.8745, swd: 1.3477, ept: 70.5705
    Epoch [8/50], Val Losses: mse: 5.3297, mae: 1.1735, huber: 0.8324, swd: 0.9107, ept: 71.6054
    Epoch [8/50], Test Losses: mse: 8.2198, mae: 1.4019, huber: 1.0519, swd: 1.2456, ept: 60.7053
      Epoch 8 composite train-obj: 0.874482
            No improvement (0.8324), counter 4/5
    Epoch [9/50], Train Losses: mse: 4.9410, mae: 1.2176, huber: 0.8733, swd: 1.3425, ept: 70.5556
    Epoch [9/50], Val Losses: mse: 5.3365, mae: 1.1729, huber: 0.8318, swd: 0.9043, ept: 71.1800
    Epoch [9/50], Test Losses: mse: 8.2288, mae: 1.4021, huber: 1.0523, swd: 1.2473, ept: 60.5790
      Epoch 9 composite train-obj: 0.873347
    Epoch [9/50], Test Losses: mse: 8.2795, mae: 1.4033, huber: 1.0534, swd: 1.2349, ept: 60.5134
    Best round's Test MSE: 8.2795, MAE: 1.4033, SWD: 1.2349
    Best round's Validation MSE: 5.3242, MAE: 1.1636, SWD: 0.8747
    Best round's Test verification MSE : 8.2795, MAE: 1.4033, SWD: 1.2349
    Time taken: 35.75 seconds
    
    ==================================================
    Experiment Summary (DLinear_ettm1_seq96_pred96_20250512_1633)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 8.2920 ± 0.0684
      mae: 1.4037 ± 0.0051
      huber: 1.0542 ± 0.0045
      swd: 1.2948 ± 0.0475
      ept: 60.5430 ± 0.0913
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.3357 ± 0.0245
      mae: 1.1655 ± 0.0016
      huber: 0.8254 ± 0.0017
      swd: 0.9118 ± 0.0296
      ept: 71.2142 ± 0.2361
      count: 53.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 140.49 seconds
    
    Experiment complete: DLinear_ettm1_seq96_pred96_20250512_1633
    Model: DLinear
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 96-196


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_dlinear_96_196 = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 8.7339, mae: 1.5762, huber: 1.2123, swd: 2.3886, target_std: 6.5132
    Epoch [1/50], Val Losses: mse: 6.5798, mae: 1.3056, huber: 0.9577, swd: 1.0915, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.8814, mae: 1.6254, huber: 1.2667, swd: 1.7038, target_std: 4.7691
      Epoch 1 composite train-obj: 1.212276
            Val objective improved inf → 0.9577, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.3149, mae: 1.3804, huber: 1.0278, swd: 1.8912, target_std: 6.5131
    Epoch [2/50], Val Losses: mse: 6.4816, mae: 1.2902, huber: 0.9442, swd: 1.0562, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.7535, mae: 1.6071, huber: 1.2506, swd: 1.6507, target_std: 4.7691
      Epoch 2 composite train-obj: 1.027763
            Val objective improved 0.9577 → 0.9442, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.2550, mae: 1.3727, huber: 1.0205, swd: 1.8570, target_std: 6.5129
    Epoch [3/50], Val Losses: mse: 6.4421, mae: 1.2758, huber: 0.9312, swd: 0.9751, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.7672, mae: 1.6014, huber: 1.2458, swd: 1.5708, target_std: 4.7691
      Epoch 3 composite train-obj: 1.020538
            Val objective improved 0.9442 → 0.9312, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.2209, mae: 1.3699, huber: 1.0177, swd: 1.8395, target_std: 6.5130
    Epoch [4/50], Val Losses: mse: 6.4293, mae: 1.2814, huber: 0.9364, swd: 1.0324, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.7100, mae: 1.6040, huber: 1.2481, swd: 1.6291, target_std: 4.7691
      Epoch 4 composite train-obj: 1.017739
            No improvement (0.9364), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.2048, mae: 1.3691, huber: 1.0168, swd: 1.8286, target_std: 6.5128
    Epoch [5/50], Val Losses: mse: 6.3859, mae: 1.2739, huber: 0.9296, swd: 0.9952, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.6275, mae: 1.5936, huber: 1.2379, swd: 1.5632, target_std: 4.7691
      Epoch 5 composite train-obj: 1.016816
            Val objective improved 0.9312 → 0.9296, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.1954, mae: 1.3685, huber: 1.0163, swd: 1.8225, target_std: 6.5129
    Epoch [6/50], Val Losses: mse: 6.4333, mae: 1.2936, huber: 0.9469, swd: 1.1116, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.5590, mae: 1.5965, huber: 1.2407, swd: 1.6557, target_std: 4.7691
      Epoch 6 composite train-obj: 1.016262
            No improvement (0.9469), counter 1/5
    Epoch [7/50], Train Losses: mse: 6.1839, mae: 1.3673, huber: 1.0150, swd: 1.8142, target_std: 6.5128
    Epoch [7/50], Val Losses: mse: 6.4020, mae: 1.2868, huber: 0.9408, swd: 1.0741, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.5437, mae: 1.5930, huber: 1.2373, swd: 1.6120, target_std: 4.7691
      Epoch 7 composite train-obj: 1.015018
            No improvement (0.9408), counter 2/5
    Epoch [8/50], Train Losses: mse: 6.1833, mae: 1.3680, huber: 1.0157, swd: 1.8125, target_std: 6.5128
    Epoch [8/50], Val Losses: mse: 6.4318, mae: 1.2843, huber: 0.9393, swd: 1.0312, target_std: 4.2926
    Epoch [8/50], Test Losses: mse: 10.6100, mae: 1.5962, huber: 1.2394, swd: 1.5678, target_std: 4.7691
      Epoch 8 composite train-obj: 1.015663
            No improvement (0.9393), counter 3/5
    Epoch [9/50], Train Losses: mse: 6.1763, mae: 1.3675, huber: 1.0151, swd: 1.8095, target_std: 6.5129
    Epoch [9/50], Val Losses: mse: 6.4188, mae: 1.2833, huber: 0.9382, swd: 1.0270, target_std: 4.2926
    Epoch [9/50], Test Losses: mse: 10.6045, mae: 1.5940, huber: 1.2382, swd: 1.5713, target_std: 4.7691
      Epoch 9 composite train-obj: 1.015097
            No improvement (0.9382), counter 4/5
    Epoch [10/50], Train Losses: mse: 6.1779, mae: 1.3682, huber: 1.0157, swd: 1.8135, target_std: 6.5129
    Epoch [10/50], Val Losses: mse: 6.4595, mae: 1.2844, huber: 0.9396, swd: 1.0010, target_std: 4.2926
    Epoch [10/50], Test Losses: mse: 10.6978, mae: 1.5977, huber: 1.2425, swd: 1.5444, target_std: 4.7691
      Epoch 10 composite train-obj: 1.015703
    Epoch [10/50], Test Losses: mse: 10.6275, mae: 1.5936, huber: 1.2379, swd: 1.5632, target_std: 4.7691
    Best round's Test MSE: 10.6275, MAE: 1.5936, SWD: 1.5632
    Best round's Validation MSE: 6.3859, MAE: 1.2739
    Best round's Test verification MSE : 10.6275, MAE: 1.5936, SWD: 1.5632
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.5540, mae: 1.5724, huber: 1.2084, swd: 2.4786, target_std: 6.5130
    Epoch [1/50], Val Losses: mse: 6.5647, mae: 1.3012, huber: 0.9538, swd: 1.0632, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 10.9166, mae: 1.6267, huber: 1.2679, swd: 1.6897, target_std: 4.7691
      Epoch 1 composite train-obj: 1.208434
            Val objective improved inf → 0.9538, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.3133, mae: 1.3800, huber: 1.0274, swd: 1.9304, target_std: 6.5129
    Epoch [2/50], Val Losses: mse: 6.4571, mae: 1.2832, huber: 0.9374, swd: 1.0232, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.7655, mae: 1.6082, huber: 1.2517, swd: 1.6279, target_std: 4.7691
      Epoch 2 composite train-obj: 1.027391
            Val objective improved 0.9538 → 0.9374, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.2542, mae: 1.3727, huber: 1.0206, swd: 1.8959, target_std: 6.5129
    Epoch [3/50], Val Losses: mse: 6.4089, mae: 1.2754, huber: 0.9306, swd: 1.0014, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.7012, mae: 1.6019, huber: 1.2454, swd: 1.5970, target_std: 4.7691
      Epoch 3 composite train-obj: 1.020557
            Val objective improved 0.9374 → 0.9306, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.2222, mae: 1.3701, huber: 1.0179, swd: 1.8775, target_std: 6.5129
    Epoch [4/50], Val Losses: mse: 6.4320, mae: 1.2908, huber: 0.9445, swd: 1.0937, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.5731, mae: 1.5971, huber: 1.2410, swd: 1.6373, target_std: 4.7691
      Epoch 4 composite train-obj: 1.017885
            No improvement (0.9445), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.2038, mae: 1.3690, huber: 1.0167, swd: 1.8684, target_std: 6.5132
    Epoch [5/50], Val Losses: mse: 6.4066, mae: 1.2811, huber: 0.9363, swd: 1.0440, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.6195, mae: 1.5958, huber: 1.2399, swd: 1.6019, target_std: 4.7691
      Epoch 5 composite train-obj: 1.016701
            No improvement (0.9363), counter 2/5
    Epoch [6/50], Train Losses: mse: 6.1949, mae: 1.3685, huber: 1.0162, swd: 1.8608, target_std: 6.5129
    Epoch [6/50], Val Losses: mse: 6.4223, mae: 1.2847, huber: 0.9395, swd: 1.0535, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.6130, mae: 1.5978, huber: 1.2417, swd: 1.6101, target_std: 4.7691
      Epoch 6 composite train-obj: 1.016236
            No improvement (0.9395), counter 3/5
    Epoch [7/50], Train Losses: mse: 6.1861, mae: 1.3680, huber: 1.0156, swd: 1.8554, target_std: 6.5127
    Epoch [7/50], Val Losses: mse: 6.4512, mae: 1.2934, huber: 0.9473, swd: 1.0762, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.6095, mae: 1.5994, huber: 1.2430, swd: 1.6064, target_std: 4.7691
      Epoch 7 composite train-obj: 1.015624
            No improvement (0.9473), counter 4/5
    Epoch [8/50], Train Losses: mse: 6.1785, mae: 1.3675, huber: 1.0151, swd: 1.8491, target_std: 6.5131
    Epoch [8/50], Val Losses: mse: 6.4265, mae: 1.2853, huber: 0.9396, swd: 1.0270, target_std: 4.2926
    Epoch [8/50], Test Losses: mse: 10.6120, mae: 1.5962, huber: 1.2402, swd: 1.5608, target_std: 4.7691
      Epoch 8 composite train-obj: 1.015112
    Epoch [8/50], Test Losses: mse: 10.7012, mae: 1.6019, huber: 1.2454, swd: 1.5970, target_std: 4.7691
    Best round's Test MSE: 10.7012, MAE: 1.6019, SWD: 1.5970
    Best round's Validation MSE: 6.4089, MAE: 1.2754
    Best round's Test verification MSE : 10.7012, MAE: 1.6019, SWD: 1.5970
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.6623, mae: 1.5762, huber: 1.2121, swd: 2.1698, target_std: 6.5128
    Epoch [1/50], Val Losses: mse: 6.5750, mae: 1.2913, huber: 0.9446, swd: 0.9139, target_std: 4.2926
    Epoch [1/50], Test Losses: mse: 11.0389, mae: 1.6286, huber: 1.2695, swd: 1.5345, target_std: 4.7691
      Epoch 1 composite train-obj: 1.212105
            Val objective improved inf → 0.9446, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.3157, mae: 1.3805, huber: 1.0279, swd: 1.7033, target_std: 6.5130
    Epoch [2/50], Val Losses: mse: 6.4514, mae: 1.2820, huber: 0.9366, swd: 0.9617, target_std: 4.2926
    Epoch [2/50], Test Losses: mse: 10.7681, mae: 1.6081, huber: 1.2514, swd: 1.5516, target_std: 4.7691
      Epoch 2 composite train-obj: 1.027883
            Val objective improved 0.9446 → 0.9366, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.2538, mae: 1.3728, huber: 1.0207, swd: 1.6723, target_std: 6.5128
    Epoch [3/50], Val Losses: mse: 6.4178, mae: 1.2767, huber: 0.9312, swd: 0.9446, target_std: 4.2926
    Epoch [3/50], Test Losses: mse: 10.6965, mae: 1.5985, huber: 1.2427, swd: 1.5065, target_std: 4.7691
      Epoch 3 composite train-obj: 1.020658
            Val objective improved 0.9366 → 0.9312, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.2228, mae: 1.3703, huber: 1.0181, swd: 1.6576, target_std: 6.5130
    Epoch [4/50], Val Losses: mse: 6.4231, mae: 1.2862, huber: 0.9403, swd: 1.0132, target_std: 4.2926
    Epoch [4/50], Test Losses: mse: 10.6087, mae: 1.5983, huber: 1.2421, swd: 1.5580, target_std: 4.7691
      Epoch 4 composite train-obj: 1.018064
            No improvement (0.9403), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.2022, mae: 1.3686, huber: 1.0164, swd: 1.6457, target_std: 6.5130
    Epoch [5/50], Val Losses: mse: 6.4169, mae: 1.2815, huber: 0.9358, swd: 0.9564, target_std: 4.2926
    Epoch [5/50], Test Losses: mse: 10.6432, mae: 1.5982, huber: 1.2411, swd: 1.4980, target_std: 4.7691
      Epoch 5 composite train-obj: 1.016353
            No improvement (0.9358), counter 2/5
    Epoch [6/50], Train Losses: mse: 6.1929, mae: 1.3684, huber: 1.0161, swd: 1.6413, target_std: 6.5128
    Epoch [6/50], Val Losses: mse: 6.4125, mae: 1.2783, huber: 0.9338, swd: 0.9481, target_std: 4.2926
    Epoch [6/50], Test Losses: mse: 10.6467, mae: 1.5964, huber: 1.2401, swd: 1.4921, target_std: 4.7691
      Epoch 6 composite train-obj: 1.016061
            No improvement (0.9338), counter 3/5
    Epoch [7/50], Train Losses: mse: 6.1889, mae: 1.3685, huber: 1.0160, swd: 1.6374, target_std: 6.5129
    Epoch [7/50], Val Losses: mse: 6.4156, mae: 1.2862, huber: 0.9406, swd: 0.9996, target_std: 4.2926
    Epoch [7/50], Test Losses: mse: 10.5720, mae: 1.5938, huber: 1.2382, swd: 1.5314, target_std: 4.7691
      Epoch 7 composite train-obj: 1.016014
            No improvement (0.9406), counter 4/5
    Epoch [8/50], Train Losses: mse: 6.1787, mae: 1.3674, huber: 1.0151, swd: 1.6321, target_std: 6.5129
    Epoch [8/50], Val Losses: mse: 6.4236, mae: 1.2911, huber: 0.9453, swd: 1.0289, target_std: 4.2926
    Epoch [8/50], Test Losses: mse: 10.5139, mae: 1.5921, huber: 1.2362, swd: 1.5488, target_std: 4.7691
      Epoch 8 composite train-obj: 1.015079
    Epoch [8/50], Test Losses: mse: 10.6965, mae: 1.5985, huber: 1.2427, swd: 1.5065, target_std: 4.7691
    Best round's Test MSE: 10.6965, MAE: 1.5985, SWD: 1.5065
    Best round's Validation MSE: 6.4178, MAE: 1.2767
    Best round's Test verification MSE : 10.6965, MAE: 1.5985, SWD: 1.5065
    
    ==================================================
    Experiment Summary (DLinear_ettm1_seq96_pred196_20250430_0440)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.6750 ± 0.0337
      mae: 1.5980 ± 0.0034
      huber: 1.2420 ± 0.0031
      swd: 1.5555 ± 0.0373
      target_std: 4.7691 ± 0.0000
      count: 53.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 6.4042 ± 0.0134
      mae: 1.2753 ± 0.0011
      huber: 0.9305 ± 0.0007
      swd: 0.9804 ± 0.0255
      target_std: 4.2926 ± 0.0000
      count: 53.0000 ± 0.0000
    ==================================================
    
    Experiment complete: DLinear_ettm1_seq96_pred196_20250430_0440
    Model: DLinear
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 96-336


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_dlinear_96_336 = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 9.4612, mae: 1.6890, huber: 1.3176, swd: 2.5916, target_std: 6.5147
    Epoch [1/50], Val Losses: mse: 7.9481, mae: 1.4172, huber: 1.0634, swd: 1.0246, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 13.1106, mae: 1.8026, huber: 1.4358, swd: 1.8202, target_std: 4.7640
      Epoch 1 composite train-obj: 1.317575
            Val objective improved inf → 1.0634, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.3093, mae: 1.5084, huber: 1.1466, swd: 2.0943, target_std: 6.5150
    Epoch [2/50], Val Losses: mse: 7.8116, mae: 1.4101, huber: 1.0575, swd: 1.0842, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 12.7558, mae: 1.7799, huber: 1.4149, swd: 1.8299, target_std: 4.7640
      Epoch 2 composite train-obj: 1.146633
            Val objective improved 1.0634 → 1.0575, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.2338, mae: 1.5003, huber: 1.1389, swd: 2.0551, target_std: 6.5145
    Epoch [3/50], Val Losses: mse: 7.7545, mae: 1.3958, huber: 1.0444, swd: 1.0200, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 12.7640, mae: 1.7744, huber: 1.4099, swd: 1.7506, target_std: 4.7640
      Epoch 3 composite train-obj: 1.138891
            Val objective improved 1.0575 → 1.0444, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.1879, mae: 1.4964, huber: 1.1351, swd: 2.0307, target_std: 6.5147
    Epoch [4/50], Val Losses: mse: 7.7576, mae: 1.3993, huber: 1.0480, swd: 1.0342, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 12.7070, mae: 1.7709, huber: 1.4061, swd: 1.7378, target_std: 4.7640
      Epoch 4 composite train-obj: 1.135087
            No improvement (1.0480), counter 1/5
    Epoch [5/50], Train Losses: mse: 7.1595, mae: 1.4943, huber: 1.1329, swd: 2.0108, target_std: 6.5152
    Epoch [5/50], Val Losses: mse: 7.7836, mae: 1.4198, huber: 1.0662, swd: 1.1632, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 12.5711, mae: 1.7682, huber: 1.4044, swd: 1.8251, target_std: 4.7640
      Epoch 5 composite train-obj: 1.132942
            No improvement (1.0662), counter 2/5
    Epoch [6/50], Train Losses: mse: 7.1464, mae: 1.4939, huber: 1.1324, swd: 2.0007, target_std: 6.5151
    Epoch [6/50], Val Losses: mse: 7.7182, mae: 1.3994, huber: 1.0481, swd: 1.0511, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 12.6228, mae: 1.7661, huber: 1.4014, swd: 1.7247, target_std: 4.7640
      Epoch 6 composite train-obj: 1.132366
            No improvement (1.0481), counter 3/5
    Epoch [7/50], Train Losses: mse: 7.1327, mae: 1.4930, huber: 1.1315, swd: 1.9926, target_std: 6.5149
    Epoch [7/50], Val Losses: mse: 7.7541, mae: 1.4088, huber: 1.0572, swd: 1.1152, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 12.5829, mae: 1.7673, huber: 1.4023, swd: 1.7696, target_std: 4.7640
      Epoch 7 composite train-obj: 1.131482
            No improvement (1.0572), counter 4/5
    Epoch [8/50], Train Losses: mse: 7.1257, mae: 1.4926, huber: 1.1310, swd: 1.9890, target_std: 6.5147
    Epoch [8/50], Val Losses: mse: 7.7051, mae: 1.4086, huber: 1.0562, swd: 1.1269, target_std: 4.2820
    Epoch [8/50], Test Losses: mse: 12.4507, mae: 1.7571, huber: 1.3936, swd: 1.7473, target_std: 4.7640
      Epoch 8 composite train-obj: 1.131043
    Epoch [8/50], Test Losses: mse: 12.7640, mae: 1.7744, huber: 1.4099, swd: 1.7506, target_std: 4.7640
    Best round's Test MSE: 12.7640, MAE: 1.7744, SWD: 1.7506
    Best round's Validation MSE: 7.7545, MAE: 1.3958
    Best round's Test verification MSE : 12.7640, MAE: 1.7744, SWD: 1.7506
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.4401, mae: 1.6868, huber: 1.3155, swd: 2.7012, target_std: 6.5147
    Epoch [1/50], Val Losses: mse: 7.9930, mae: 1.4240, huber: 1.0705, swd: 1.0960, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 13.1336, mae: 1.8042, huber: 1.4373, swd: 1.9221, target_std: 4.7640
      Epoch 1 composite train-obj: 1.315492
            Val objective improved inf → 1.0705, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.3103, mae: 1.5086, huber: 1.1468, swd: 2.1980, target_std: 6.5151
    Epoch [2/50], Val Losses: mse: 7.9012, mae: 1.4123, huber: 1.0598, swd: 1.0938, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 12.9072, mae: 1.7814, huber: 1.4167, swd: 1.8584, target_std: 4.7640
      Epoch 2 composite train-obj: 1.146813
            Val objective improved 1.0705 → 1.0598, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.2315, mae: 1.5001, huber: 1.1387, swd: 2.1540, target_std: 6.5152
    Epoch [3/50], Val Losses: mse: 7.7880, mae: 1.4048, huber: 1.0525, swd: 1.0911, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 12.7093, mae: 1.7697, huber: 1.4055, swd: 1.8201, target_std: 4.7640
      Epoch 3 composite train-obj: 1.138667
            Val objective improved 1.0598 → 1.0525, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.1918, mae: 1.4969, huber: 1.1355, swd: 2.1277, target_std: 6.5148
    Epoch [4/50], Val Losses: mse: 7.7960, mae: 1.3999, huber: 1.0486, swd: 1.0544, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 12.8208, mae: 1.7768, huber: 1.4116, swd: 1.7978, target_std: 4.7640
      Epoch 4 composite train-obj: 1.135542
            Val objective improved 1.0525 → 1.0486, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.1685, mae: 1.4953, huber: 1.1339, swd: 2.1131, target_std: 6.5148
    Epoch [5/50], Val Losses: mse: 7.7443, mae: 1.4086, huber: 1.0562, swd: 1.1544, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 12.5365, mae: 1.7629, huber: 1.3985, swd: 1.8361, target_std: 4.7640
      Epoch 5 composite train-obj: 1.133856
            No improvement (1.0562), counter 1/5
    Epoch [6/50], Train Losses: mse: 7.1455, mae: 1.4938, huber: 1.1323, swd: 2.0986, target_std: 6.5149
    Epoch [6/50], Val Losses: mse: 7.7150, mae: 1.4089, huber: 1.0570, swd: 1.1705, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 12.4779, mae: 1.7613, huber: 1.3965, swd: 1.8364, target_std: 4.7640
      Epoch 6 composite train-obj: 1.132335
            No improvement (1.0570), counter 2/5
    Epoch [7/50], Train Losses: mse: 7.1351, mae: 1.4931, huber: 1.1316, swd: 2.0885, target_std: 6.5148
    Epoch [7/50], Val Losses: mse: 7.7008, mae: 1.4012, huber: 1.0499, swd: 1.1340, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 12.4977, mae: 1.7587, huber: 1.3941, swd: 1.7908, target_std: 4.7640
      Epoch 7 composite train-obj: 1.131591
            No improvement (1.0499), counter 3/5
    Epoch [8/50], Train Losses: mse: 7.1238, mae: 1.4924, huber: 1.1308, swd: 2.0822, target_std: 6.5151
    Epoch [8/50], Val Losses: mse: 7.7555, mae: 1.4125, huber: 1.0606, swd: 1.1840, target_std: 4.2820
    Epoch [8/50], Test Losses: mse: 12.5287, mae: 1.7618, huber: 1.3978, swd: 1.8226, target_std: 4.7640
      Epoch 8 composite train-obj: 1.130820
            No improvement (1.0606), counter 4/5
    Epoch [9/50], Train Losses: mse: 7.1192, mae: 1.4923, huber: 1.1307, swd: 2.0789, target_std: 6.5148
    Epoch [9/50], Val Losses: mse: 7.7954, mae: 1.4177, huber: 1.0661, swd: 1.1980, target_std: 4.2820
    Epoch [9/50], Test Losses: mse: 12.5728, mae: 1.7691, huber: 1.4038, swd: 1.8586, target_std: 4.7640
      Epoch 9 composite train-obj: 1.130697
    Epoch [9/50], Test Losses: mse: 12.8208, mae: 1.7768, huber: 1.4116, swd: 1.7978, target_std: 4.7640
    Best round's Test MSE: 12.8208, MAE: 1.7768, SWD: 1.7978
    Best round's Validation MSE: 7.7960, MAE: 1.3999
    Best round's Test verification MSE : 12.8208, MAE: 1.7768, SWD: 1.7978
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.4472, mae: 1.6879, huber: 1.3165, swd: 2.6185, target_std: 6.5150
    Epoch [1/50], Val Losses: mse: 7.9638, mae: 1.4251, huber: 1.0707, swd: 1.1198, target_std: 4.2820
    Epoch [1/50], Test Losses: mse: 13.0592, mae: 1.8025, huber: 1.4360, swd: 2.0219, target_std: 4.7640
      Epoch 1 composite train-obj: 1.316495
            Val objective improved inf → 1.0707, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.3134, mae: 1.5090, huber: 1.1472, swd: 2.0894, target_std: 6.5149
    Epoch [2/50], Val Losses: mse: 7.8865, mae: 1.4232, huber: 1.0699, swd: 1.1986, target_std: 4.2820
    Epoch [2/50], Test Losses: mse: 12.7774, mae: 1.7819, huber: 1.4169, swd: 2.0295, target_std: 4.7640
      Epoch 2 composite train-obj: 1.147206
            Val objective improved 1.0707 → 1.0699, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.2353, mae: 1.5006, huber: 1.1392, swd: 2.0487, target_std: 6.5153
    Epoch [3/50], Val Losses: mse: 7.7779, mae: 1.3917, huber: 1.0406, swd: 1.0147, target_std: 4.2820
    Epoch [3/50], Test Losses: mse: 12.8568, mae: 1.7760, huber: 1.4111, swd: 1.8598, target_std: 4.7640
      Epoch 3 composite train-obj: 1.139164
            Val objective improved 1.0699 → 1.0406, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.1897, mae: 1.4968, huber: 1.1353, swd: 2.0237, target_std: 6.5152
    Epoch [4/50], Val Losses: mse: 7.7695, mae: 1.4137, huber: 1.0614, swd: 1.1960, target_std: 4.2820
    Epoch [4/50], Test Losses: mse: 12.5912, mae: 1.7693, huber: 1.4047, swd: 1.9848, target_std: 4.7640
      Epoch 4 composite train-obj: 1.135320
            No improvement (1.0614), counter 1/5
    Epoch [5/50], Train Losses: mse: 7.1644, mae: 1.4950, huber: 1.1336, swd: 2.0069, target_std: 6.5147
    Epoch [5/50], Val Losses: mse: 7.7416, mae: 1.3991, huber: 1.0480, swd: 1.1001, target_std: 4.2820
    Epoch [5/50], Test Losses: mse: 12.6596, mae: 1.7673, huber: 1.4027, swd: 1.8913, target_std: 4.7640
      Epoch 5 composite train-obj: 1.133565
            No improvement (1.0480), counter 2/5
    Epoch [6/50], Train Losses: mse: 7.1442, mae: 1.4935, huber: 1.1320, swd: 1.9930, target_std: 6.5153
    Epoch [6/50], Val Losses: mse: 7.7684, mae: 1.4107, huber: 1.0591, swd: 1.1781, target_std: 4.2820
    Epoch [6/50], Test Losses: mse: 12.5536, mae: 1.7631, huber: 1.3987, swd: 1.9231, target_std: 4.7640
      Epoch 6 composite train-obj: 1.131995
            No improvement (1.0591), counter 3/5
    Epoch [7/50], Train Losses: mse: 7.1348, mae: 1.4927, huber: 1.1312, swd: 1.9879, target_std: 6.5145
    Epoch [7/50], Val Losses: mse: 7.7311, mae: 1.4055, huber: 1.0532, swd: 1.1388, target_std: 4.2820
    Epoch [7/50], Test Losses: mse: 12.5394, mae: 1.7610, huber: 1.3969, swd: 1.8857, target_std: 4.7640
      Epoch 7 composite train-obj: 1.131220
            No improvement (1.0532), counter 4/5
    Epoch [8/50], Train Losses: mse: 7.1249, mae: 1.4928, huber: 1.1311, swd: 1.9806, target_std: 6.5147
    Epoch [8/50], Val Losses: mse: 7.6989, mae: 1.4033, huber: 1.0516, swd: 1.1490, target_std: 4.2820
    Epoch [8/50], Test Losses: mse: 12.4971, mae: 1.7603, huber: 1.3959, swd: 1.8935, target_std: 4.7640
      Epoch 8 composite train-obj: 1.131104
    Epoch [8/50], Test Losses: mse: 12.8568, mae: 1.7760, huber: 1.4111, swd: 1.8598, target_std: 4.7640
    Best round's Test MSE: 12.8568, MAE: 1.7760, SWD: 1.8598
    Best round's Validation MSE: 7.7779, MAE: 1.3917
    Best round's Test verification MSE : 12.8568, MAE: 1.7760, SWD: 1.8598
    
    ==================================================
    Experiment Summary (DLinear_ettm1_seq96_pred336_20250430_0442)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.8139 ± 0.0382
      mae: 1.7758 ± 0.0010
      huber: 1.4109 ± 0.0007
      swd: 1.8027 ± 0.0447
      target_std: 4.7640 ± 0.0000
      count: 52.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.7762 ± 0.0170
      mae: 1.3958 ± 0.0033
      huber: 1.0446 ± 0.0033
      swd: 1.0297 ± 0.0176
      target_std: 4.2820 ± 0.0000
      count: 52.0000 ± 0.0000
    ==================================================
    
    Experiment complete: DLinear_ettm1_seq96_pred336_20250430_0442
    Model: DLinear
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 96-720


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['ettm1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_dlinear_96_720 = execute_model_evaluation('ettm1', cfg, data_mgr, scale=False)
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
    Data Preparation: ettm1
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
    
    Epoch [1/50], Train Losses: mse: 10.5921, mae: 1.8354, huber: 1.4541, swd: 2.7928, target_std: 6.5150
    Epoch [1/50], Val Losses: mse: 10.3498, mae: 1.6098, huber: 1.2481, swd: 1.2064, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 14.5064, mae: 1.9792, huber: 1.6014, swd: 2.0557, target_std: 4.7646
      Epoch 1 composite train-obj: 1.454073
            Val objective improved inf → 1.2481, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.3834, mae: 1.6611, huber: 1.2876, swd: 2.3193, target_std: 6.5150
    Epoch [2/50], Val Losses: mse: 10.1814, mae: 1.5932, huber: 1.2332, swd: 1.1706, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 14.2798, mae: 1.9590, huber: 1.5821, swd: 1.9774, target_std: 4.7646
      Epoch 2 composite train-obj: 1.287605
            Val objective improved 1.2481 → 1.2332, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.2857, mae: 1.6507, huber: 1.2776, swd: 2.2616, target_std: 6.5152
    Epoch [3/50], Val Losses: mse: 10.0897, mae: 1.6025, huber: 1.2409, swd: 1.2944, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 14.0191, mae: 1.9477, huber: 1.5714, swd: 2.0168, target_std: 4.7646
      Epoch 3 composite train-obj: 1.277566
            No improvement (1.2409), counter 1/5
    Epoch [4/50], Train Losses: mse: 8.2277, mae: 1.6454, huber: 1.2724, swd: 2.2272, target_std: 6.5153
    Epoch [4/50], Val Losses: mse: 10.0947, mae: 1.6070, huber: 1.2461, swd: 1.3521, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 13.9053, mae: 1.9401, huber: 1.5641, swd: 1.9974, target_std: 4.7646
      Epoch 4 composite train-obj: 1.272434
            No improvement (1.2461), counter 2/5
    Epoch [5/50], Train Losses: mse: 8.1874, mae: 1.6417, huber: 1.2687, swd: 2.1950, target_std: 6.5152
    Epoch [5/50], Val Losses: mse: 10.0024, mae: 1.6051, huber: 1.2439, swd: 1.3864, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 13.7651, mae: 1.9332, huber: 1.5573, swd: 1.9893, target_std: 4.7646
      Epoch 5 composite train-obj: 1.268732
            No improvement (1.2439), counter 3/5
    Epoch [6/50], Train Losses: mse: 8.1693, mae: 1.6403, huber: 1.2674, swd: 2.1824, target_std: 6.5151
    Epoch [6/50], Val Losses: mse: 10.0305, mae: 1.6172, huber: 1.2553, swd: 1.4447, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 13.6870, mae: 1.9301, huber: 1.5550, swd: 2.0005, target_std: 4.7646
      Epoch 6 composite train-obj: 1.267383
            No improvement (1.2553), counter 4/5
    Epoch [7/50], Train Losses: mse: 8.1465, mae: 1.6383, huber: 1.2653, swd: 2.1641, target_std: 6.5152
    Epoch [7/50], Val Losses: mse: 9.9625, mae: 1.6101, huber: 1.2477, swd: 1.4015, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 13.7135, mae: 1.9296, huber: 1.5551, swd: 1.9590, target_std: 4.7646
      Epoch 7 composite train-obj: 1.265348
    Epoch [7/50], Test Losses: mse: 14.2798, mae: 1.9590, huber: 1.5821, swd: 1.9774, target_std: 4.7646
    Best round's Test MSE: 14.2798, MAE: 1.9590, SWD: 1.9774
    Best round's Validation MSE: 10.1814, MAE: 1.5932
    Best round's Test verification MSE : 14.2798, MAE: 1.9590, SWD: 1.9774
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.6136, mae: 1.8347, huber: 1.4532, swd: 2.6318, target_std: 6.5150
    Epoch [1/50], Val Losses: mse: 10.3448, mae: 1.6029, huber: 1.2421, swd: 1.1300, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 14.5725, mae: 1.9809, huber: 1.6028, swd: 2.0252, target_std: 4.7646
      Epoch 1 composite train-obj: 1.453182
            Val objective improved inf → 1.2421, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.3788, mae: 1.6605, huber: 1.2870, swd: 2.1990, target_std: 6.5148
    Epoch [2/50], Val Losses: mse: 10.1585, mae: 1.6003, huber: 1.2395, swd: 1.2391, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 14.1274, mae: 1.9538, huber: 1.5773, swd: 2.0434, target_std: 4.7646
      Epoch 2 composite train-obj: 1.287034
            Val objective improved 1.2421 → 1.2395, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.2834, mae: 1.6506, huber: 1.2775, swd: 2.1500, target_std: 6.5154
    Epoch [3/50], Val Losses: mse: 10.0520, mae: 1.5904, huber: 1.2303, swd: 1.1982, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 14.0790, mae: 1.9463, huber: 1.5710, swd: 1.9880, target_std: 4.7646
      Epoch 3 composite train-obj: 1.277461
            Val objective improved 1.2395 → 1.2303, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.2213, mae: 1.6446, huber: 1.2717, swd: 2.1140, target_std: 6.5155
    Epoch [4/50], Val Losses: mse: 10.0738, mae: 1.6105, huber: 1.2485, swd: 1.3360, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 13.8976, mae: 1.9438, huber: 1.5680, swd: 2.0496, target_std: 4.7646
      Epoch 4 composite train-obj: 1.271731
            No improvement (1.2485), counter 1/5
    Epoch [5/50], Train Losses: mse: 8.1954, mae: 1.6426, huber: 1.2696, swd: 2.0941, target_std: 6.5154
    Epoch [5/50], Val Losses: mse: 9.9889, mae: 1.5805, huber: 1.2214, swd: 1.1243, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 14.0364, mae: 1.9423, huber: 1.5658, swd: 1.8652, target_std: 4.7646
      Epoch 5 composite train-obj: 1.269642
            Val objective improved 1.2303 → 1.2214, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 8.1680, mae: 1.6400, huber: 1.2671, swd: 2.0774, target_std: 6.5154
    Epoch [6/50], Val Losses: mse: 9.9589, mae: 1.6038, huber: 1.2430, swd: 1.3388, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 13.7589, mae: 1.9339, huber: 1.5582, swd: 1.9967, target_std: 4.7646
      Epoch 6 composite train-obj: 1.267082
            No improvement (1.2430), counter 1/5
    Epoch [7/50], Train Losses: mse: 8.1476, mae: 1.6383, huber: 1.2654, swd: 2.0574, target_std: 6.5148
    Epoch [7/50], Val Losses: mse: 9.8954, mae: 1.6006, huber: 1.2390, swd: 1.3357, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 13.6957, mae: 1.9303, huber: 1.5541, swd: 1.9778, target_std: 4.7646
      Epoch 7 composite train-obj: 1.265380
            No improvement (1.2390), counter 2/5
    Epoch [8/50], Train Losses: mse: 8.1384, mae: 1.6379, huber: 1.2649, swd: 2.0511, target_std: 6.5155
    Epoch [8/50], Val Losses: mse: 9.9654, mae: 1.5979, huber: 1.2380, swd: 1.2708, target_std: 4.2630
    Epoch [8/50], Test Losses: mse: 13.8696, mae: 1.9397, huber: 1.5634, swd: 1.9275, target_std: 4.7646
      Epoch 8 composite train-obj: 1.264894
            No improvement (1.2380), counter 3/5
    Epoch [9/50], Train Losses: mse: 8.1280, mae: 1.6371, huber: 1.2640, swd: 2.0427, target_std: 6.5151
    Epoch [9/50], Val Losses: mse: 9.9145, mae: 1.5906, huber: 1.2312, swd: 1.2468, target_std: 4.2630
    Epoch [9/50], Test Losses: mse: 13.7634, mae: 1.9281, huber: 1.5527, swd: 1.8726, target_std: 4.7646
      Epoch 9 composite train-obj: 1.264040
            No improvement (1.2312), counter 4/5
    Epoch [10/50], Train Losses: mse: 8.1210, mae: 1.6360, huber: 1.2631, swd: 2.0331, target_std: 6.5151
    Epoch [10/50], Val Losses: mse: 9.9226, mae: 1.5985, huber: 1.2381, swd: 1.2795, target_std: 4.2630
    Epoch [10/50], Test Losses: mse: 13.7383, mae: 1.9298, huber: 1.5546, swd: 1.9042, target_std: 4.7646
      Epoch 10 composite train-obj: 1.263067
    Epoch [10/50], Test Losses: mse: 14.0364, mae: 1.9423, huber: 1.5658, swd: 1.8652, target_std: 4.7646
    Best round's Test MSE: 14.0364, MAE: 1.9423, SWD: 1.8652
    Best round's Validation MSE: 9.9889, MAE: 1.5805
    Best round's Test verification MSE : 14.0364, MAE: 1.9423, SWD: 1.8652
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.5978, mae: 1.8353, huber: 1.4539, swd: 2.9226, target_std: 6.5150
    Epoch [1/50], Val Losses: mse: 10.3462, mae: 1.5996, huber: 1.2387, swd: 1.1784, target_std: 4.2630
    Epoch [1/50], Test Losses: mse: 14.6572, mae: 1.9855, huber: 1.6073, swd: 2.1587, target_std: 4.7646
      Epoch 1 composite train-obj: 1.453859
            Val objective improved inf → 1.2387, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.3828, mae: 1.6611, huber: 1.2875, swd: 2.4158, target_std: 6.5149
    Epoch [2/50], Val Losses: mse: 10.1625, mae: 1.5928, huber: 1.2329, swd: 1.2616, target_std: 4.2630
    Epoch [2/50], Test Losses: mse: 14.2280, mae: 1.9561, huber: 1.5796, swd: 2.1409, target_std: 4.7646
      Epoch 2 composite train-obj: 1.287542
            Val objective improved 1.2387 → 1.2329, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.2820, mae: 1.6503, huber: 1.2773, swd: 2.3573, target_std: 6.5153
    Epoch [3/50], Val Losses: mse: 10.0675, mae: 1.5833, huber: 1.2240, swd: 1.2299, target_std: 4.2630
    Epoch [3/50], Test Losses: mse: 14.1678, mae: 1.9503, huber: 1.5736, swd: 2.0709, target_std: 4.7646
      Epoch 3 composite train-obj: 1.277260
            Val objective improved 1.2329 → 1.2240, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.2272, mae: 1.6451, huber: 1.2721, swd: 2.3193, target_std: 6.5153
    Epoch [4/50], Val Losses: mse: 10.0465, mae: 1.5886, huber: 1.2284, swd: 1.2833, target_std: 4.2630
    Epoch [4/50], Test Losses: mse: 14.0506, mae: 1.9454, huber: 1.5693, swd: 2.0650, target_std: 4.7646
      Epoch 4 composite train-obj: 1.272139
            No improvement (1.2284), counter 1/5
    Epoch [5/50], Train Losses: mse: 8.1876, mae: 1.6418, huber: 1.2689, swd: 2.2916, target_std: 6.5155
    Epoch [5/50], Val Losses: mse: 9.9844, mae: 1.6032, huber: 1.2421, swd: 1.4443, target_std: 4.2630
    Epoch [5/50], Test Losses: mse: 13.7867, mae: 1.9349, huber: 1.5595, swd: 2.1512, target_std: 4.7646
      Epoch 5 composite train-obj: 1.268862
            No improvement (1.2421), counter 2/5
    Epoch [6/50], Train Losses: mse: 8.1681, mae: 1.6402, huber: 1.2673, swd: 2.2735, target_std: 6.5151
    Epoch [6/50], Val Losses: mse: 9.9409, mae: 1.6015, huber: 1.2400, swd: 1.4393, target_std: 4.2630
    Epoch [6/50], Test Losses: mse: 13.7415, mae: 1.9303, huber: 1.5555, swd: 2.1062, target_std: 4.7646
      Epoch 6 composite train-obj: 1.267257
            No improvement (1.2400), counter 3/5
    Epoch [7/50], Train Losses: mse: 8.1450, mae: 1.6384, huber: 1.2653, swd: 2.2540, target_std: 6.5150
    Epoch [7/50], Val Losses: mse: 9.9272, mae: 1.6122, huber: 1.2487, swd: 1.4998, target_std: 4.2630
    Epoch [7/50], Test Losses: mse: 13.6748, mae: 1.9309, huber: 1.5561, swd: 2.1351, target_std: 4.7646
      Epoch 7 composite train-obj: 1.265339
            No improvement (1.2487), counter 4/5
    Epoch [8/50], Train Losses: mse: 8.1367, mae: 1.6377, huber: 1.2647, swd: 2.2447, target_std: 6.5151
    Epoch [8/50], Val Losses: mse: 9.9826, mae: 1.6062, huber: 1.2458, swd: 1.4849, target_std: 4.2630
    Epoch [8/50], Test Losses: mse: 13.6810, mae: 1.9279, huber: 1.5515, swd: 2.0720, target_std: 4.7646
      Epoch 8 composite train-obj: 1.264695
    Epoch [8/50], Test Losses: mse: 14.1678, mae: 1.9503, huber: 1.5736, swd: 2.0709, target_std: 4.7646
    Best round's Test MSE: 14.1678, MAE: 1.9503, SWD: 2.0709
    Best round's Validation MSE: 10.0675, MAE: 1.5833
    Best round's Test verification MSE : 14.1678, MAE: 1.9503, SWD: 2.0709
    
    ==================================================
    Experiment Summary (DLinear_ettm1_seq96_pred720_20250430_0443)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 14.1613 ± 0.0995
      mae: 1.9505 ± 0.0068
      huber: 1.5739 ± 0.0066
      swd: 1.9712 ± 0.0841
      target_std: 4.7646 ± 0.0000
      count: 49.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 10.0793 ± 0.0790
      mae: 1.5857 ± 0.0055
      huber: 1.2262 ± 0.0051
      swd: 1.1750 ± 0.0432
      target_std: 4.2630 ± 0.0000
      count: 49.0000 ± 0.0000
    ==================================================
    
    Experiment complete: DLinear_ettm1_seq96_pred720_20250430_0443
    Model: DLinear
    Dataset: ettm1
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


