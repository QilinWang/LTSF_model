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
data_mgr = DatasetManager(device='cuda')

# Load a synthetic dataset
data_mgr.load_csv('etth1', './etth1.csv')
```

    
    ==================================================
    Dataset: etth1 (csv)
    ==================================================
    Shape: torch.Size([17420, 7])
    Channels: 7
    Length: 17420
    Source: ./etth1.csv
    
    Sample data (first 2 rows):
    tensor([[ 5.8270,  2.0090,  1.5990,  0.4620,  4.2030,  1.3400, 30.5310],
            [ 5.6930,  2.0760,  1.4920,  0.4260,  4.1420,  1.3710, 27.7870]])
    ==================================================
    




    <data_manager.DatasetManager at 0x1987b8dcfb0>



## Seq=196

### EigenACL

#### pred=96


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=196,
    pred_len=96,
    channels=data_mgr.datasets['etth1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 96
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 12
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5247, mae: 0.5103, huber: 0.2198, swd: 0.2838, ept: 39.3460
    Epoch [1/50], Val Losses: mse: 0.3965, mae: 0.4387, huber: 0.1715, swd: 0.1393, ept: 44.1465
    Epoch [1/50], Test Losses: mse: 0.4851, mae: 0.4906, huber: 0.2085, swd: 0.1727, ept: 33.8034
      Epoch 1 composite train-obj: 0.219839
            Val objective improved inf → 0.1715, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3672, mae: 0.4235, huber: 0.1609, swd: 0.1283, ept: 48.1447
    Epoch [2/50], Val Losses: mse: 0.3680, mae: 0.4143, huber: 0.1594, swd: 0.1224, ept: 46.4581
    Epoch [2/50], Test Losses: mse: 0.4525, mae: 0.4732, huber: 0.1966, swd: 0.1657, ept: 36.2254
      Epoch 2 composite train-obj: 0.160874
            Val objective improved 0.1715 → 0.1594, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3460, mae: 0.4087, huber: 0.1522, swd: 0.1163, ept: 50.1408
    Epoch [3/50], Val Losses: mse: 0.3706, mae: 0.4272, huber: 0.1621, swd: 0.1308, ept: 46.6843
    Epoch [3/50], Test Losses: mse: 0.4388, mae: 0.4659, huber: 0.1915, swd: 0.1674, ept: 36.9336
      Epoch 3 composite train-obj: 0.152164
            No improvement (0.1621), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3312, mae: 0.3997, huber: 0.1465, swd: 0.1094, ept: 51.5030
    Epoch [4/50], Val Losses: mse: 0.3659, mae: 0.4158, huber: 0.1590, swd: 0.1200, ept: 48.0568
    Epoch [4/50], Test Losses: mse: 0.4331, mae: 0.4609, huber: 0.1890, swd: 0.1585, ept: 37.8907
      Epoch 4 composite train-obj: 0.146488
            Val objective improved 0.1594 → 0.1590, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3181, mae: 0.3925, huber: 0.1418, swd: 0.1048, ept: 52.4379
    Epoch [5/50], Val Losses: mse: 0.3679, mae: 0.4196, huber: 0.1600, swd: 0.1170, ept: 48.3891
    Epoch [5/50], Test Losses: mse: 0.4279, mae: 0.4590, huber: 0.1873, swd: 0.1476, ept: 38.3732
      Epoch 5 composite train-obj: 0.141766
            No improvement (0.1600), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3043, mae: 0.3851, huber: 0.1369, swd: 0.0978, ept: 53.2351
    Epoch [6/50], Val Losses: mse: 0.3734, mae: 0.4188, huber: 0.1616, swd: 0.1101, ept: 48.9163
    Epoch [6/50], Test Losses: mse: 0.4294, mae: 0.4591, huber: 0.1876, swd: 0.1485, ept: 38.8044
      Epoch 6 composite train-obj: 0.136861
            No improvement (0.1616), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.2941, mae: 0.3795, huber: 0.1331, swd: 0.0917, ept: 53.6003
    Epoch [7/50], Val Losses: mse: 0.3773, mae: 0.4268, huber: 0.1638, swd: 0.1121, ept: 48.3180
    Epoch [7/50], Test Losses: mse: 0.4275, mae: 0.4584, huber: 0.1865, swd: 0.1437, ept: 39.3826
      Epoch 7 composite train-obj: 0.133087
            No improvement (0.1638), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.2874, mae: 0.3757, huber: 0.1305, swd: 0.0885, ept: 54.0135
    Epoch [8/50], Val Losses: mse: 0.3809, mae: 0.4327, huber: 0.1666, swd: 0.1358, ept: 48.6744
    Epoch [8/50], Test Losses: mse: 0.4508, mae: 0.4734, huber: 0.1963, swd: 0.1773, ept: 38.1944
      Epoch 8 composite train-obj: 0.130518
            No improvement (0.1666), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.2771, mae: 0.3695, huber: 0.1265, swd: 0.0825, ept: 54.5570
    Epoch [9/50], Val Losses: mse: 0.4063, mae: 0.4503, huber: 0.1767, swd: 0.1287, ept: 48.0287
    Epoch [9/50], Test Losses: mse: 0.4417, mae: 0.4679, huber: 0.1926, swd: 0.1531, ept: 39.0241
      Epoch 9 composite train-obj: 0.126532
    Epoch [9/50], Test Losses: mse: 0.4331, mae: 0.4609, huber: 0.1890, swd: 0.1585, ept: 37.8907
    Best round's Test MSE: 0.4331, MAE: 0.4609, SWD: 0.1585
    Best round's Validation MSE: 0.3659, MAE: 0.4158, SWD: 0.1200
    Best round's Test verification MSE : 0.4331, MAE: 0.4609, SWD: 0.1585
    Time taken: 22.87 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5323, mae: 0.5127, huber: 0.2221, swd: 0.2852, ept: 39.7343
    Epoch [1/50], Val Losses: mse: 0.3806, mae: 0.4245, huber: 0.1644, swd: 0.1232, ept: 45.4773
    Epoch [1/50], Test Losses: mse: 0.4850, mae: 0.4902, huber: 0.2078, swd: 0.1621, ept: 34.5632
      Epoch 1 composite train-obj: 0.222092
            Val objective improved inf → 0.1644, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3673, mae: 0.4239, huber: 0.1609, swd: 0.1260, ept: 48.4991
    Epoch [2/50], Val Losses: mse: 0.3751, mae: 0.4259, huber: 0.1632, swd: 0.1269, ept: 46.7145
    Epoch [2/50], Test Losses: mse: 0.4488, mae: 0.4704, huber: 0.1950, swd: 0.1592, ept: 36.2163
      Epoch 2 composite train-obj: 0.160907
            Val objective improved 0.1644 → 0.1632, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3450, mae: 0.4080, huber: 0.1518, swd: 0.1138, ept: 50.6163
    Epoch [3/50], Val Losses: mse: 0.3622, mae: 0.4132, huber: 0.1575, swd: 0.1163, ept: 47.3595
    Epoch [3/50], Test Losses: mse: 0.4368, mae: 0.4620, huber: 0.1903, swd: 0.1571, ept: 37.5749
      Epoch 3 composite train-obj: 0.151772
            Val objective improved 0.1632 → 0.1575, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3323, mae: 0.4007, huber: 0.1471, swd: 0.1086, ept: 51.4977
    Epoch [4/50], Val Losses: mse: 0.3779, mae: 0.4271, huber: 0.1646, swd: 0.1302, ept: 47.8620
    Epoch [4/50], Test Losses: mse: 0.4407, mae: 0.4661, huber: 0.1924, swd: 0.1626, ept: 37.7202
      Epoch 4 composite train-obj: 0.147082
            No improvement (0.1646), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3207, mae: 0.3937, huber: 0.1427, swd: 0.1037, ept: 52.5442
    Epoch [5/50], Val Losses: mse: 0.3755, mae: 0.4224, huber: 0.1626, swd: 0.1129, ept: 47.4812
    Epoch [5/50], Test Losses: mse: 0.4319, mae: 0.4606, huber: 0.1886, swd: 0.1455, ept: 38.6336
      Epoch 5 composite train-obj: 0.142696
            No improvement (0.1626), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3107, mae: 0.3883, huber: 0.1391, swd: 0.0996, ept: 53.1645
    Epoch [6/50], Val Losses: mse: 0.3704, mae: 0.4269, huber: 0.1621, swd: 0.1311, ept: 48.3971
    Epoch [6/50], Test Losses: mse: 0.4313, mae: 0.4604, huber: 0.1885, swd: 0.1632, ept: 38.3236
      Epoch 6 composite train-obj: 0.139106
            No improvement (0.1621), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.2988, mae: 0.3817, huber: 0.1347, swd: 0.0934, ept: 53.7817
    Epoch [7/50], Val Losses: mse: 0.3778, mae: 0.4290, huber: 0.1644, swd: 0.1168, ept: 48.7383
    Epoch [7/50], Test Losses: mse: 0.4236, mae: 0.4553, huber: 0.1851, swd: 0.1401, ept: 39.3255
      Epoch 7 composite train-obj: 0.134681
            No improvement (0.1644), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.2886, mae: 0.3760, huber: 0.1308, swd: 0.0881, ept: 54.1997
    Epoch [8/50], Val Losses: mse: 0.3885, mae: 0.4390, huber: 0.1694, swd: 0.1314, ept: 48.7948
    Epoch [8/50], Test Losses: mse: 0.4346, mae: 0.4635, huber: 0.1898, swd: 0.1601, ept: 38.7112
      Epoch 8 composite train-obj: 0.130828
    Epoch [8/50], Test Losses: mse: 0.4368, mae: 0.4620, huber: 0.1903, swd: 0.1571, ept: 37.5791
    Best round's Test MSE: 0.4368, MAE: 0.4620, SWD: 0.1571
    Best round's Validation MSE: 0.3622, MAE: 0.4132, SWD: 0.1163
    Best round's Test verification MSE : 0.4368, MAE: 0.4620, SWD: 0.1571
    Time taken: 20.49 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5254, mae: 0.5095, huber: 0.2197, swd: 0.2705, ept: 39.8580
    Epoch [1/50], Val Losses: mse: 0.3824, mae: 0.4304, huber: 0.1661, swd: 0.1334, ept: 45.3642
    Epoch [1/50], Test Losses: mse: 0.4753, mae: 0.4852, huber: 0.2046, swd: 0.1606, ept: 34.3718
      Epoch 1 composite train-obj: 0.219700
            Val objective improved inf → 0.1661, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3668, mae: 0.4222, huber: 0.1604, swd: 0.1217, ept: 48.7585
    Epoch [2/50], Val Losses: mse: 0.3638, mae: 0.4158, huber: 0.1581, swd: 0.1209, ept: 47.0179
    Epoch [2/50], Test Losses: mse: 0.4454, mae: 0.4679, huber: 0.1935, swd: 0.1585, ept: 36.5428
      Epoch 2 composite train-obj: 0.160367
            Val objective improved 0.1661 → 0.1581, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3442, mae: 0.4069, huber: 0.1513, swd: 0.1096, ept: 50.6246
    Epoch [3/50], Val Losses: mse: 0.3609, mae: 0.4139, huber: 0.1569, swd: 0.1175, ept: 47.2885
    Epoch [3/50], Test Losses: mse: 0.4373, mae: 0.4635, huber: 0.1907, swd: 0.1559, ept: 37.2225
      Epoch 3 composite train-obj: 0.151278
            Val objective improved 0.1581 → 0.1569, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3312, mae: 0.3996, huber: 0.1466, swd: 0.1041, ept: 51.6418
    Epoch [4/50], Val Losses: mse: 0.3532, mae: 0.4072, huber: 0.1536, swd: 0.1121, ept: 47.9004
    Epoch [4/50], Test Losses: mse: 0.4327, mae: 0.4609, huber: 0.1889, swd: 0.1514, ept: 37.6807
      Epoch 4 composite train-obj: 0.146559
            Val objective improved 0.1569 → 0.1536, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3205, mae: 0.3937, huber: 0.1427, swd: 0.1004, ept: 52.3311
    Epoch [5/50], Val Losses: mse: 0.3751, mae: 0.4148, huber: 0.1612, swd: 0.1073, ept: 47.9510
    Epoch [5/50], Test Losses: mse: 0.4327, mae: 0.4609, huber: 0.1886, swd: 0.1329, ept: 38.7194
      Epoch 5 composite train-obj: 0.142664
            No improvement (0.1612), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3104, mae: 0.3880, huber: 0.1390, swd: 0.0960, ept: 52.8921
    Epoch [6/50], Val Losses: mse: 0.3718, mae: 0.4210, huber: 0.1614, swd: 0.1150, ept: 47.9507
    Epoch [6/50], Test Losses: mse: 0.4268, mae: 0.4570, huber: 0.1866, swd: 0.1429, ept: 38.8833
      Epoch 6 composite train-obj: 0.138964
            No improvement (0.1614), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.3000, mae: 0.3828, huber: 0.1353, swd: 0.0910, ept: 53.4742
    Epoch [7/50], Val Losses: mse: 0.3717, mae: 0.4267, huber: 0.1625, swd: 0.1195, ept: 48.0916
    Epoch [7/50], Test Losses: mse: 0.4267, mae: 0.4575, huber: 0.1868, swd: 0.1475, ept: 38.5429
      Epoch 7 composite train-obj: 0.135296
            No improvement (0.1625), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.2899, mae: 0.3769, huber: 0.1314, swd: 0.0854, ept: 54.0250
    Epoch [8/50], Val Losses: mse: 0.3635, mae: 0.4193, huber: 0.1590, swd: 0.1241, ept: 48.8851
    Epoch [8/50], Test Losses: mse: 0.4382, mae: 0.4646, huber: 0.1912, swd: 0.1663, ept: 38.0936
      Epoch 8 composite train-obj: 0.131443
            No improvement (0.1590), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.2824, mae: 0.3726, huber: 0.1286, swd: 0.0816, ept: 54.3221
    Epoch [9/50], Val Losses: mse: 0.3841, mae: 0.4389, huber: 0.1687, swd: 0.1285, ept: 47.7785
    Epoch [9/50], Test Losses: mse: 0.4440, mae: 0.4671, huber: 0.1932, swd: 0.1560, ept: 38.3661
      Epoch 9 composite train-obj: 0.128583
    Epoch [9/50], Test Losses: mse: 0.4327, mae: 0.4609, huber: 0.1889, swd: 0.1513, ept: 37.6817
    Best round's Test MSE: 0.4327, MAE: 0.4609, SWD: 0.1514
    Best round's Validation MSE: 0.3532, MAE: 0.4072, SWD: 0.1121
    Best round's Test verification MSE : 0.4327, MAE: 0.4609, SWD: 0.1513
    Time taken: 22.95 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq196_pred96_20250510_1831)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4342 ± 0.0018
      mae: 0.4613 ± 0.0005
      huber: 0.1894 ± 0.0006
      swd: 0.1557 ± 0.0031
      ept: 37.7155 ± 0.1313
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3604 ± 0.0053
      mae: 0.4121 ± 0.0036
      huber: 0.1567 ± 0.0023
      swd: 0.1161 ± 0.0032
      ept: 47.7723 ± 0.2988
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 67.62 seconds
    
    Experiment complete: ACL_etth1_seq196_pred96_20250510_1831
    Model: ACL
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=196,
    pred_len=196,
    channels=data_mgr.datasets['etth1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 196
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 11
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6091, mae: 0.5493, huber: 0.2490, swd: 0.3455, ept: 56.2994
    Epoch [1/50], Val Losses: mse: 0.4528, mae: 0.4700, huber: 0.1924, swd: 0.1638, ept: 57.8765
    Epoch [1/50], Test Losses: mse: 0.5423, mae: 0.5252, huber: 0.2307, swd: 0.1858, ept: 51.0975
      Epoch 1 composite train-obj: 0.248997
            Val objective improved inf → 0.1924, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4459, mae: 0.4640, huber: 0.1896, swd: 0.1641, ept: 70.7808
    Epoch [2/50], Val Losses: mse: 0.4351, mae: 0.4520, huber: 0.1847, swd: 0.1206, ept: 59.5209
    Epoch [2/50], Test Losses: mse: 0.4802, mae: 0.4942, huber: 0.2090, swd: 0.1277, ept: 54.9181
      Epoch 2 composite train-obj: 0.189580
            Val objective improved 0.1924 → 0.1847, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4177, mae: 0.4486, huber: 0.1796, swd: 0.1480, ept: 74.2795
    Epoch [3/50], Val Losses: mse: 0.4179, mae: 0.4431, huber: 0.1795, swd: 0.1277, ept: 61.6434
    Epoch [3/50], Test Losses: mse: 0.4711, mae: 0.4926, huber: 0.2071, swd: 0.1477, ept: 54.7463
      Epoch 3 composite train-obj: 0.179639
            Val objective improved 0.1847 → 0.1795, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3958, mae: 0.4381, huber: 0.1722, swd: 0.1383, ept: 76.1009
    Epoch [4/50], Val Losses: mse: 0.4358, mae: 0.4658, huber: 0.1889, swd: 0.1470, ept: 61.4141
    Epoch [4/50], Test Losses: mse: 0.4698, mae: 0.4914, huber: 0.2067, swd: 0.1627, ept: 55.3944
      Epoch 4 composite train-obj: 0.172190
            No improvement (0.1889), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3782, mae: 0.4293, huber: 0.1660, swd: 0.1280, ept: 77.6855
    Epoch [5/50], Val Losses: mse: 0.4506, mae: 0.4719, huber: 0.1940, swd: 0.1490, ept: 62.4366
    Epoch [5/50], Test Losses: mse: 0.4617, mae: 0.4877, huber: 0.2038, swd: 0.1559, ept: 56.0946
      Epoch 5 composite train-obj: 0.166024
            No improvement (0.1940), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3632, mae: 0.4219, huber: 0.1608, swd: 0.1192, ept: 78.9027
    Epoch [6/50], Val Losses: mse: 0.4627, mae: 0.4830, huber: 0.1992, swd: 0.1469, ept: 63.2267
    Epoch [6/50], Test Losses: mse: 0.4638, mae: 0.4899, huber: 0.2048, swd: 0.1571, ept: 57.3386
      Epoch 6 composite train-obj: 0.160765
            No improvement (0.1992), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3491, mae: 0.4147, huber: 0.1556, swd: 0.1105, ept: 79.9993
    Epoch [7/50], Val Losses: mse: 0.4767, mae: 0.4916, huber: 0.2049, swd: 0.1451, ept: 62.9736
    Epoch [7/50], Test Losses: mse: 0.4717, mae: 0.4939, huber: 0.2076, swd: 0.1465, ept: 56.7099
      Epoch 7 composite train-obj: 0.155641
            No improvement (0.2049), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3382, mae: 0.4090, huber: 0.1516, swd: 0.1053, ept: 80.6036
    Epoch [8/50], Val Losses: mse: 0.4723, mae: 0.4986, huber: 0.2059, swd: 0.1654, ept: 63.6998
    Epoch [8/50], Test Losses: mse: 0.4818, mae: 0.4996, huber: 0.2113, swd: 0.1766, ept: 56.6241
      Epoch 8 composite train-obj: 0.151645
    Epoch [8/50], Test Losses: mse: 0.4710, mae: 0.4926, huber: 0.2071, swd: 0.1477, ept: 54.7445
    Best round's Test MSE: 0.4711, MAE: 0.4926, SWD: 0.1477
    Best round's Validation MSE: 0.4179, MAE: 0.4431, SWD: 0.1277
    Best round's Test verification MSE : 0.4710, MAE: 0.4926, SWD: 0.1477
    Time taken: 20.40 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6182, mae: 0.5537, huber: 0.2522, swd: 0.3497, ept: 55.1671
    Epoch [1/50], Val Losses: mse: 0.4701, mae: 0.4907, huber: 0.2017, swd: 0.1931, ept: 56.8116
    Epoch [1/50], Test Losses: mse: 0.5417, mae: 0.5287, huber: 0.2319, swd: 0.2022, ept: 48.6690
      Epoch 1 composite train-obj: 0.252158
            Val objective improved inf → 0.2017, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4486, mae: 0.4660, huber: 0.1908, swd: 0.1667, ept: 70.0178
    Epoch [2/50], Val Losses: mse: 0.4259, mae: 0.4485, huber: 0.1819, swd: 0.1211, ept: 60.1535
    Epoch [2/50], Test Losses: mse: 0.4798, mae: 0.4954, huber: 0.2096, swd: 0.1412, ept: 54.1645
      Epoch 2 composite train-obj: 0.190817
            Val objective improved 0.2017 → 0.1819, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4183, mae: 0.4489, huber: 0.1799, swd: 0.1501, ept: 73.7363
    Epoch [3/50], Val Losses: mse: 0.4499, mae: 0.4669, huber: 0.1918, swd: 0.1382, ept: 61.5581
    Epoch [3/50], Test Losses: mse: 0.4717, mae: 0.4918, huber: 0.2069, swd: 0.1355, ept: 54.2622
      Epoch 3 composite train-obj: 0.179906
            No improvement (0.1918), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3964, mae: 0.4382, huber: 0.1723, swd: 0.1377, ept: 75.9897
    Epoch [4/50], Val Losses: mse: 0.4493, mae: 0.4731, huber: 0.1939, swd: 0.1517, ept: 62.6725
    Epoch [4/50], Test Losses: mse: 0.4709, mae: 0.4958, huber: 0.2078, swd: 0.1708, ept: 55.6743
      Epoch 4 composite train-obj: 0.172259
            No improvement (0.1939), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3784, mae: 0.4294, huber: 0.1659, swd: 0.1284, ept: 77.5214
    Epoch [5/50], Val Losses: mse: 0.4725, mae: 0.4844, huber: 0.2015, swd: 0.1353, ept: 59.4770
    Epoch [5/50], Test Losses: mse: 0.4653, mae: 0.4891, huber: 0.2047, swd: 0.1287, ept: 56.4190
      Epoch 5 composite train-obj: 0.165923
            No improvement (0.2015), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3618, mae: 0.4214, huber: 0.1602, swd: 0.1170, ept: 78.4801
    Epoch [6/50], Val Losses: mse: 0.5093, mae: 0.5089, huber: 0.2164, swd: 0.1581, ept: 61.7910
    Epoch [6/50], Test Losses: mse: 0.4784, mae: 0.4981, huber: 0.2105, swd: 0.1479, ept: 56.5718
      Epoch 6 composite train-obj: 0.160233
            No improvement (0.2164), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3474, mae: 0.4142, huber: 0.1551, swd: 0.1074, ept: 79.4067
    Epoch [7/50], Val Losses: mse: 0.4789, mae: 0.4895, huber: 0.2042, swd: 0.1478, ept: 62.7810
    Epoch [7/50], Test Losses: mse: 0.4672, mae: 0.4914, huber: 0.2057, swd: 0.1373, ept: 57.6724
      Epoch 7 composite train-obj: 0.155085
    Epoch [7/50], Test Losses: mse: 0.4798, mae: 0.4954, huber: 0.2096, swd: 0.1412, ept: 54.1377
    Best round's Test MSE: 0.4798, MAE: 0.4954, SWD: 0.1412
    Best round's Validation MSE: 0.4259, MAE: 0.4485, SWD: 0.1211
    Best round's Test verification MSE : 0.4798, MAE: 0.4954, SWD: 0.1412
    Time taken: 17.55 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6103, mae: 0.5502, huber: 0.2497, swd: 0.3215, ept: 56.3987
    Epoch [1/50], Val Losses: mse: 0.4457, mae: 0.4684, huber: 0.1913, swd: 0.1557, ept: 59.4056
    Epoch [1/50], Test Losses: mse: 0.5220, mae: 0.5170, huber: 0.2246, swd: 0.1736, ept: 52.5085
      Epoch 1 composite train-obj: 0.249653
            Val objective improved inf → 0.1913, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4454, mae: 0.4643, huber: 0.1896, swd: 0.1520, ept: 71.2307
    Epoch [2/50], Val Losses: mse: 0.4293, mae: 0.4483, huber: 0.1833, swd: 0.1345, ept: 62.2045
    Epoch [2/50], Test Losses: mse: 0.4885, mae: 0.5022, huber: 0.2136, swd: 0.1548, ept: 54.7655
      Epoch 2 composite train-obj: 0.189572
            Val objective improved 0.1913 → 0.1833, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4184, mae: 0.4484, huber: 0.1796, swd: 0.1386, ept: 74.6277
    Epoch [3/50], Val Losses: mse: 0.4254, mae: 0.4480, huber: 0.1821, swd: 0.1200, ept: 62.1617
    Epoch [3/50], Test Losses: mse: 0.4659, mae: 0.4886, huber: 0.2047, swd: 0.1390, ept: 56.7403
      Epoch 3 composite train-obj: 0.179567
            Val objective improved 0.1833 → 0.1821, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3953, mae: 0.4372, huber: 0.1718, swd: 0.1298, ept: 76.8112
    Epoch [4/50], Val Losses: mse: 0.4237, mae: 0.4476, huber: 0.1816, swd: 0.1256, ept: 64.2858
    Epoch [4/50], Test Losses: mse: 0.4577, mae: 0.4831, huber: 0.2013, swd: 0.1392, ept: 57.4145
      Epoch 4 composite train-obj: 0.171768
            Val objective improved 0.1821 → 0.1816, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3796, mae: 0.4292, huber: 0.1661, swd: 0.1209, ept: 77.8298
    Epoch [5/50], Val Losses: mse: 0.4474, mae: 0.4808, huber: 0.1955, swd: 0.1572, ept: 63.3156
    Epoch [5/50], Test Losses: mse: 0.4720, mae: 0.4931, huber: 0.2073, swd: 0.1800, ept: 55.8423
      Epoch 5 composite train-obj: 0.166098
            No improvement (0.1955), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3633, mae: 0.4209, huber: 0.1603, swd: 0.1130, ept: 79.4274
    Epoch [6/50], Val Losses: mse: 0.4449, mae: 0.4752, huber: 0.1930, swd: 0.1385, ept: 64.9506
    Epoch [6/50], Test Losses: mse: 0.4621, mae: 0.4910, huber: 0.2041, swd: 0.1540, ept: 57.0824
      Epoch 6 composite train-obj: 0.160272
            No improvement (0.1930), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.3512, mae: 0.4151, huber: 0.1562, swd: 0.1059, ept: 80.0995
    Epoch [7/50], Val Losses: mse: 0.4808, mae: 0.5012, huber: 0.2085, swd: 0.1655, ept: 62.9172
    Epoch [7/50], Test Losses: mse: 0.4814, mae: 0.5001, huber: 0.2117, swd: 0.1745, ept: 56.2498
      Epoch 7 composite train-obj: 0.156195
            No improvement (0.2085), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.3375, mae: 0.4086, huber: 0.1514, swd: 0.0990, ept: 80.6091
    Epoch [8/50], Val Losses: mse: 0.4840, mae: 0.4969, huber: 0.2081, swd: 0.1618, ept: 64.3992
    Epoch [8/50], Test Losses: mse: 0.4784, mae: 0.4989, huber: 0.2102, swd: 0.1627, ept: 56.1275
      Epoch 8 composite train-obj: 0.151371
            No improvement (0.2081), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.3286, mae: 0.4034, huber: 0.1479, swd: 0.0943, ept: 81.1418
    Epoch [9/50], Val Losses: mse: 0.4798, mae: 0.5016, huber: 0.2089, swd: 0.1599, ept: 63.4242
    Epoch [9/50], Test Losses: mse: 0.4910, mae: 0.5050, huber: 0.2148, swd: 0.1709, ept: 55.7531
      Epoch 9 composite train-obj: 0.147888
    Epoch [9/50], Test Losses: mse: 0.4577, mae: 0.4831, huber: 0.2013, swd: 0.1392, ept: 57.3931
    Best round's Test MSE: 0.4577, MAE: 0.4831, SWD: 0.1392
    Best round's Validation MSE: 0.4237, MAE: 0.4476, SWD: 0.1256
    Best round's Test verification MSE : 0.4577, MAE: 0.4831, SWD: 0.1392
    Time taken: 22.65 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq196_pred196_20250510_1833)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4695 ± 0.0091
      mae: 0.4903 ± 0.0053
      huber: 0.2060 ± 0.0035
      swd: 0.1427 ± 0.0037
      ept: 55.4418 ± 1.4150
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4225 ± 0.0034
      mae: 0.4464 ± 0.0023
      huber: 0.1810 ± 0.0011
      swd: 0.1248 ± 0.0027
      ept: 62.0276 ± 1.7087
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 60.67 seconds
    
    Experiment complete: ACL_etth1_seq196_pred196_20250510_1833
    Model: ACL
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=196,
    pred_len=336,
    channels=data_mgr.datasets['etth1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 10
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6499, mae: 0.5701, huber: 0.2644, swd: 0.3514, ept: 71.0534
    Epoch [1/50], Val Losses: mse: 0.4732, mae: 0.4725, huber: 0.1995, swd: 0.1539, ept: 77.5287
    Epoch [1/50], Test Losses: mse: 0.5594, mae: 0.5438, huber: 0.2423, swd: 0.1858, ept: 70.5151
      Epoch 1 composite train-obj: 0.264358
            Val objective improved inf → 0.1995, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4996, mae: 0.4929, huber: 0.2101, swd: 0.1865, ept: 90.2993
    Epoch [2/50], Val Losses: mse: 0.4532, mae: 0.4517, huber: 0.1888, swd: 0.1152, ept: 79.2235
    Epoch [2/50], Test Losses: mse: 0.5153, mae: 0.5218, huber: 0.2267, swd: 0.1516, ept: 73.0732
      Epoch 2 composite train-obj: 0.210081
            Val objective improved 0.1995 → 0.1888, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4678, mae: 0.4764, huber: 0.1986, swd: 0.1679, ept: 94.1908
    Epoch [3/50], Val Losses: mse: 0.4706, mae: 0.4761, huber: 0.1990, swd: 0.1262, ept: 73.5714
    Epoch [3/50], Test Losses: mse: 0.5036, mae: 0.5183, huber: 0.2228, swd: 0.1507, ept: 72.3057
      Epoch 3 composite train-obj: 0.198558
            No improvement (0.1990), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4448, mae: 0.4649, huber: 0.1904, swd: 0.1535, ept: 97.5975
    Epoch [4/50], Val Losses: mse: 0.5038, mae: 0.5143, huber: 0.2165, swd: 0.1546, ept: 68.9073
    Epoch [4/50], Test Losses: mse: 0.5063, mae: 0.5236, huber: 0.2248, swd: 0.1810, ept: 69.2101
      Epoch 4 composite train-obj: 0.190423
            No improvement (0.2165), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4257, mae: 0.4552, huber: 0.1835, swd: 0.1421, ept: 99.9208
    Epoch [5/50], Val Losses: mse: 0.5390, mae: 0.5444, huber: 0.2331, swd: 0.1766, ept: 69.9161
    Epoch [5/50], Test Losses: mse: 0.5302, mae: 0.5423, huber: 0.2358, swd: 0.2104, ept: 69.6202
      Epoch 5 composite train-obj: 0.183471
            No improvement (0.2331), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4094, mae: 0.4472, huber: 0.1777, swd: 0.1318, ept: 101.0624
    Epoch [6/50], Val Losses: mse: 0.4913, mae: 0.5086, huber: 0.2124, swd: 0.1367, ept: 73.0004
    Epoch [6/50], Test Losses: mse: 0.5106, mae: 0.5261, huber: 0.2261, swd: 0.1631, ept: 72.8852
      Epoch 6 composite train-obj: 0.177706
            No improvement (0.2124), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3951, mae: 0.4400, huber: 0.1727, swd: 0.1220, ept: 102.5145
    Epoch [7/50], Val Losses: mse: 0.5121, mae: 0.5226, huber: 0.2212, swd: 0.1478, ept: 76.2581
    Epoch [7/50], Test Losses: mse: 0.5346, mae: 0.5436, huber: 0.2370, swd: 0.1938, ept: 73.7882
      Epoch 7 composite train-obj: 0.172714
    Epoch [7/50], Test Losses: mse: 0.5153, mae: 0.5218, huber: 0.2267, swd: 0.1516, ept: 73.0500
    Best round's Test MSE: 0.5153, MAE: 0.5218, SWD: 0.1516
    Best round's Validation MSE: 0.4532, MAE: 0.4517, SWD: 0.1152
    Best round's Test verification MSE : 0.5153, MAE: 0.5218, SWD: 0.1516
    Time taken: 17.61 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6628, mae: 0.5755, huber: 0.2685, swd: 0.3669, ept: 70.4239
    Epoch [1/50], Val Losses: mse: 0.4737, mae: 0.4778, huber: 0.2009, swd: 0.1662, ept: 75.6186
    Epoch [1/50], Test Losses: mse: 0.5658, mae: 0.5459, huber: 0.2440, swd: 0.2044, ept: 71.1603
      Epoch 1 composite train-obj: 0.268471
            Val objective improved inf → 0.2009, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4998, mae: 0.4934, huber: 0.2102, swd: 0.1903, ept: 90.3452
    Epoch [2/50], Val Losses: mse: 0.4710, mae: 0.4794, huber: 0.1999, swd: 0.1570, ept: 74.4460
    Epoch [2/50], Test Losses: mse: 0.5284, mae: 0.5302, huber: 0.2319, swd: 0.1956, ept: 70.7046
      Epoch 2 composite train-obj: 0.210174
            Val objective improved 0.2009 → 0.1999, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4683, mae: 0.4758, huber: 0.1985, swd: 0.1705, ept: 95.6691
    Epoch [3/50], Val Losses: mse: 0.4829, mae: 0.4767, huber: 0.2016, swd: 0.1343, ept: 76.9956
    Epoch [3/50], Test Losses: mse: 0.5108, mae: 0.5203, huber: 0.2254, swd: 0.1637, ept: 75.0500
      Epoch 3 composite train-obj: 0.198533
            No improvement (0.2016), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4435, mae: 0.4641, huber: 0.1900, swd: 0.1557, ept: 98.8217
    Epoch [4/50], Val Losses: mse: 0.5026, mae: 0.5082, huber: 0.2150, swd: 0.1711, ept: 70.6929
    Epoch [4/50], Test Losses: mse: 0.5108, mae: 0.5256, huber: 0.2266, swd: 0.1893, ept: 71.4765
      Epoch 4 composite train-obj: 0.189968
            No improvement (0.2150), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4261, mae: 0.4560, huber: 0.1840, swd: 0.1442, ept: 100.0788
    Epoch [5/50], Val Losses: mse: 0.5446, mae: 0.5364, huber: 0.2317, swd: 0.1766, ept: 71.1237
    Epoch [5/50], Test Losses: mse: 0.5183, mae: 0.5317, huber: 0.2299, swd: 0.1851, ept: 72.2768
      Epoch 5 composite train-obj: 0.183973
            No improvement (0.2317), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4102, mae: 0.4487, huber: 0.1787, swd: 0.1338, ept: 101.9726
    Epoch [6/50], Val Losses: mse: 0.5318, mae: 0.5322, huber: 0.2297, swd: 0.1810, ept: 68.0668
    Epoch [6/50], Test Losses: mse: 0.5206, mae: 0.5320, huber: 0.2304, swd: 0.2014, ept: 70.8927
      Epoch 6 composite train-obj: 0.178659
            No improvement (0.2297), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3908, mae: 0.4394, huber: 0.1718, swd: 0.1215, ept: 103.2633
    Epoch [7/50], Val Losses: mse: 0.5247, mae: 0.5252, huber: 0.2245, swd: 0.1561, ept: 78.3971
    Epoch [7/50], Test Losses: mse: 0.5190, mae: 0.5314, huber: 0.2295, swd: 0.1830, ept: 73.2069
      Epoch 7 composite train-obj: 0.171845
    Epoch [7/50], Test Losses: mse: 0.5284, mae: 0.5302, huber: 0.2319, swd: 0.1956, ept: 70.7469
    Best round's Test MSE: 0.5284, MAE: 0.5302, SWD: 0.1956
    Best round's Validation MSE: 0.4710, MAE: 0.4794, SWD: 0.1570
    Best round's Test verification MSE : 0.5284, MAE: 0.5302, SWD: 0.1956
    Time taken: 17.43 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6440, mae: 0.5684, huber: 0.2628, swd: 0.3493, ept: 71.6461
    Epoch [1/50], Val Losses: mse: 0.5287, mae: 0.5298, huber: 0.2277, swd: 0.2300, ept: 64.0375
    Epoch [1/50], Test Losses: mse: 0.5762, mae: 0.5577, huber: 0.2496, swd: 0.2436, ept: 60.4928
      Epoch 1 composite train-obj: 0.262779
            Val objective improved inf → 0.2277, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5005, mae: 0.4944, huber: 0.2107, swd: 0.1902, ept: 90.3601
    Epoch [2/50], Val Losses: mse: 0.4628, mae: 0.4663, huber: 0.1951, swd: 0.1483, ept: 76.6514
    Epoch [2/50], Test Losses: mse: 0.5260, mae: 0.5295, huber: 0.2314, swd: 0.1912, ept: 72.3442
      Epoch 2 composite train-obj: 0.210687
            Val objective improved 0.2277 → 0.1951, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4693, mae: 0.4775, huber: 0.1993, swd: 0.1706, ept: 94.4639
    Epoch [3/50], Val Losses: mse: 0.4860, mae: 0.4884, huber: 0.2057, swd: 0.1563, ept: 75.9891
    Epoch [3/50], Test Losses: mse: 0.5172, mae: 0.5254, huber: 0.2283, swd: 0.1870, ept: 74.9694
      Epoch 3 composite train-obj: 0.199254
            No improvement (0.2057), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4452, mae: 0.4649, huber: 0.1904, swd: 0.1560, ept: 98.0828
    Epoch [4/50], Val Losses: mse: 0.4905, mae: 0.4995, huber: 0.2091, swd: 0.1524, ept: 75.4952
    Epoch [4/50], Test Losses: mse: 0.5073, mae: 0.5251, huber: 0.2251, swd: 0.1907, ept: 71.5441
      Epoch 4 composite train-obj: 0.190357
            No improvement (0.2091), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4289, mae: 0.4565, huber: 0.1845, swd: 0.1459, ept: 99.7584
    Epoch [5/50], Val Losses: mse: 0.5070, mae: 0.5097, huber: 0.2154, swd: 0.1451, ept: 75.5093
    Epoch [5/50], Test Losses: mse: 0.5037, mae: 0.5254, huber: 0.2244, swd: 0.1825, ept: 72.9625
      Epoch 5 composite train-obj: 0.184473
            No improvement (0.2154), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4098, mae: 0.4478, huber: 0.1780, swd: 0.1349, ept: 101.1192
    Epoch [6/50], Val Losses: mse: 0.4908, mae: 0.5037, huber: 0.2109, swd: 0.1289, ept: 78.2795
    Epoch [6/50], Test Losses: mse: 0.5067, mae: 0.5267, huber: 0.2251, swd: 0.1857, ept: 73.6035
      Epoch 6 composite train-obj: 0.178038
            No improvement (0.2109), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3940, mae: 0.4400, huber: 0.1724, swd: 0.1249, ept: 102.5685
    Epoch [7/50], Val Losses: mse: 0.5300, mae: 0.5306, huber: 0.2302, swd: 0.1542, ept: 67.4394
    Epoch [7/50], Test Losses: mse: 0.5245, mae: 0.5334, huber: 0.2315, swd: 0.1817, ept: 70.4380
      Epoch 7 composite train-obj: 0.172405
    Epoch [7/50], Test Losses: mse: 0.5260, mae: 0.5295, huber: 0.2314, swd: 0.1912, ept: 72.3388
    Best round's Test MSE: 0.5260, MAE: 0.5295, SWD: 0.1912
    Best round's Validation MSE: 0.4628, MAE: 0.4663, SWD: 0.1483
    Best round's Test verification MSE : 0.5260, MAE: 0.5295, SWD: 0.1912
    Time taken: 17.44 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq196_pred336_20250510_1834)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5232 ± 0.0057
      mae: 0.5272 ± 0.0038
      huber: 0.2300 ± 0.0024
      swd: 0.1795 ± 0.0198
      ept: 72.0407 ± 0.9905
      count: 10.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4623 ± 0.0073
      mae: 0.4658 ± 0.0113
      huber: 0.1946 ± 0.0045
      swd: 0.1402 ± 0.0180
      ept: 76.7736 ± 1.9523
      count: 10.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 52.55 seconds
    
    Experiment complete: ACL_etth1_seq196_pred336_20250510_1834
    Model: ACL
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=196,
    pred_len=720,
    channels=data_mgr.datasets['etth1']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 7
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7163, mae: 0.6066, huber: 0.2903, swd: 0.3996, ept: 85.6115
    Epoch [1/50], Val Losses: mse: 0.4978, mae: 0.5144, huber: 0.2190, swd: 0.1779, ept: 92.3979
    Epoch [1/50], Test Losses: mse: 0.6757, mae: 0.6136, huber: 0.2904, swd: 0.2368, ept: 108.0541
      Epoch 1 composite train-obj: 0.290322
            Val objective improved inf → 0.2190, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5743, mae: 0.5382, huber: 0.2402, swd: 0.2382, ept: 112.6458
    Epoch [2/50], Val Losses: mse: 0.5365, mae: 0.5351, huber: 0.2318, swd: 0.1680, ept: 91.1419
    Epoch [2/50], Test Losses: mse: 0.6508, mae: 0.6049, huber: 0.2836, swd: 0.2172, ept: 107.5505
      Epoch 2 composite train-obj: 0.240170
            No improvement (0.2318), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5422, mae: 0.5197, huber: 0.2274, swd: 0.2163, ept: 118.6361
    Epoch [3/50], Val Losses: mse: 0.5376, mae: 0.5496, huber: 0.2372, swd: 0.1891, ept: 76.2221
    Epoch [3/50], Test Losses: mse: 0.6388, mae: 0.6000, huber: 0.2792, swd: 0.2622, ept: 98.6202
      Epoch 3 composite train-obj: 0.227430
            No improvement (0.2372), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.5156, mae: 0.5060, huber: 0.2173, swd: 0.2015, ept: 120.6778
    Epoch [4/50], Val Losses: mse: 0.5852, mae: 0.5736, huber: 0.2536, swd: 0.1488, ept: 78.8306
    Epoch [4/50], Test Losses: mse: 0.6366, mae: 0.5996, huber: 0.2781, swd: 0.1971, ept: 95.4463
      Epoch 4 composite train-obj: 0.217261
            No improvement (0.2536), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4939, mae: 0.4951, huber: 0.2091, swd: 0.1851, ept: 121.9748
    Epoch [5/50], Val Losses: mse: 0.5700, mae: 0.5794, huber: 0.2527, swd: 0.1532, ept: 74.1854
    Epoch [5/50], Test Losses: mse: 0.6330, mae: 0.5962, huber: 0.2764, swd: 0.2068, ept: 95.6768
      Epoch 5 composite train-obj: 0.209124
            No improvement (0.2527), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.4761, mae: 0.4861, huber: 0.2027, swd: 0.1717, ept: 124.4279
    Epoch [6/50], Val Losses: mse: 0.6713, mae: 0.6308, huber: 0.2922, swd: 0.1833, ept: 63.2621
    Epoch [6/50], Test Losses: mse: 0.6495, mae: 0.6090, huber: 0.2838, swd: 0.1819, ept: 90.0092
      Epoch 6 composite train-obj: 0.202743
    Epoch [6/50], Test Losses: mse: 0.6757, mae: 0.6136, huber: 0.2904, swd: 0.2368, ept: 108.0347
    Best round's Test MSE: 0.6757, MAE: 0.6136, SWD: 0.2368
    Best round's Validation MSE: 0.4978, MAE: 0.5144, SWD: 0.1779
    Best round's Test verification MSE : 0.6757, MAE: 0.6136, SWD: 0.2368
    Time taken: 14.71 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7068, mae: 0.6030, huber: 0.2875, swd: 0.3829, ept: 86.8714
    Epoch [1/50], Val Losses: mse: 0.5258, mae: 0.5385, huber: 0.2324, swd: 0.2058, ept: 76.5945
    Epoch [1/50], Test Losses: mse: 0.6654, mae: 0.6104, huber: 0.2873, swd: 0.2476, ept: 99.2759
      Epoch 1 composite train-obj: 0.287548
            Val objective improved inf → 0.2324, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5779, mae: 0.5389, huber: 0.2411, swd: 0.2359, ept: 112.3222
    Epoch [2/50], Val Losses: mse: 0.5259, mae: 0.5327, huber: 0.2298, swd: 0.1687, ept: 79.4246
    Epoch [2/50], Test Losses: mse: 0.6359, mae: 0.5983, huber: 0.2784, swd: 0.2228, ept: 100.0097
      Epoch 2 composite train-obj: 0.241058
            Val objective improved 0.2324 → 0.2298, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5440, mae: 0.5214, huber: 0.2283, swd: 0.2138, ept: 116.5484
    Epoch [3/50], Val Losses: mse: 0.5080, mae: 0.5200, huber: 0.2211, swd: 0.1490, ept: 84.0611
    Epoch [3/50], Test Losses: mse: 0.6243, mae: 0.5904, huber: 0.2729, swd: 0.2243, ept: 98.6295
      Epoch 3 composite train-obj: 0.228344
            Val objective improved 0.2298 → 0.2211, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5205, mae: 0.5084, huber: 0.2191, swd: 0.1994, ept: 118.6551
    Epoch [4/50], Val Losses: mse: 0.5529, mae: 0.5574, huber: 0.2416, swd: 0.1450, ept: 84.0772
    Epoch [4/50], Test Losses: mse: 0.6360, mae: 0.5997, huber: 0.2783, swd: 0.2164, ept: 102.5605
      Epoch 4 composite train-obj: 0.219054
            No improvement (0.2416), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4966, mae: 0.4963, huber: 0.2101, swd: 0.1836, ept: 120.5320
    Epoch [5/50], Val Losses: mse: 0.5844, mae: 0.5816, huber: 0.2574, swd: 0.1528, ept: 75.1358
    Epoch [5/50], Test Losses: mse: 0.6326, mae: 0.5969, huber: 0.2768, swd: 0.1964, ept: 101.5742
      Epoch 5 composite train-obj: 0.210115
            No improvement (0.2574), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4779, mae: 0.4869, huber: 0.2034, swd: 0.1705, ept: 122.5972
    Epoch [6/50], Val Losses: mse: 0.5688, mae: 0.5756, huber: 0.2523, swd: 0.1333, ept: 76.8292
    Epoch [6/50], Test Losses: mse: 0.6329, mae: 0.5965, huber: 0.2766, swd: 0.1857, ept: 101.4024
      Epoch 6 composite train-obj: 0.203387
            No improvement (0.2523), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.4582, mae: 0.4770, huber: 0.1966, swd: 0.1565, ept: 124.0606
    Epoch [7/50], Val Losses: mse: 0.5946, mae: 0.5900, huber: 0.2611, swd: 0.1296, ept: 79.6185
    Epoch [7/50], Test Losses: mse: 0.6219, mae: 0.5920, huber: 0.2725, swd: 0.1816, ept: 98.9570
      Epoch 7 composite train-obj: 0.196582
            No improvement (0.2611), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4435, mae: 0.4695, huber: 0.1915, swd: 0.1463, ept: 126.4817
    Epoch [8/50], Val Losses: mse: 0.6373, mae: 0.6093, huber: 0.2759, swd: 0.1429, ept: 73.7416
    Epoch [8/50], Test Losses: mse: 0.6549, mae: 0.6061, huber: 0.2846, swd: 0.1838, ept: 98.7530
      Epoch 8 composite train-obj: 0.191520
    Epoch [8/50], Test Losses: mse: 0.6243, mae: 0.5904, huber: 0.2729, swd: 0.2243, ept: 98.6343
    Best round's Test MSE: 0.6243, MAE: 0.5904, SWD: 0.2243
    Best round's Validation MSE: 0.5080, MAE: 0.5200, SWD: 0.1490
    Best round's Test verification MSE : 0.6243, MAE: 0.5904, SWD: 0.2243
    Time taken: 19.04 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7131, mae: 0.6050, huber: 0.2893, swd: 0.4130, ept: 85.6748
    Epoch [1/50], Val Losses: mse: 0.5042, mae: 0.5217, huber: 0.2221, swd: 0.1776, ept: 86.7762
    Epoch [1/50], Test Losses: mse: 0.6708, mae: 0.6110, huber: 0.2890, swd: 0.2159, ept: 100.1122
      Epoch 1 composite train-obj: 0.289281
            Val objective improved inf → 0.2221, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5725, mae: 0.5372, huber: 0.2395, swd: 0.2474, ept: 112.0116
    Epoch [2/50], Val Losses: mse: 0.5168, mae: 0.5127, huber: 0.2207, swd: 0.1402, ept: 91.1964
    Epoch [2/50], Test Losses: mse: 0.6359, mae: 0.5944, huber: 0.2769, swd: 0.1905, ept: 106.7383
      Epoch 2 composite train-obj: 0.239468
            Val objective improved 0.2221 → 0.2207, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5390, mae: 0.5187, huber: 0.2264, swd: 0.2223, ept: 118.1244
    Epoch [3/50], Val Losses: mse: 0.5936, mae: 0.5784, huber: 0.2593, swd: 0.2094, ept: 58.1124
    Epoch [3/50], Test Losses: mse: 0.6415, mae: 0.6024, huber: 0.2811, swd: 0.2370, ept: 89.7331
      Epoch 3 composite train-obj: 0.226439
            No improvement (0.2593), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5136, mae: 0.5051, huber: 0.2166, swd: 0.2040, ept: 118.8822
    Epoch [4/50], Val Losses: mse: 0.6096, mae: 0.5925, huber: 0.2659, swd: 0.1847, ept: 69.9239
    Epoch [4/50], Test Losses: mse: 0.6380, mae: 0.6002, huber: 0.2789, swd: 0.2029, ept: 91.4448
      Epoch 4 composite train-obj: 0.216575
            No improvement (0.2659), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4914, mae: 0.4937, huber: 0.2084, swd: 0.1871, ept: 121.0056
    Epoch [5/50], Val Losses: mse: 0.6139, mae: 0.6025, huber: 0.2700, swd: 0.1841, ept: 69.7624
    Epoch [5/50], Test Losses: mse: 0.6360, mae: 0.5994, huber: 0.2778, swd: 0.2037, ept: 93.0589
      Epoch 5 composite train-obj: 0.208388
            No improvement (0.2700), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4737, mae: 0.4847, huber: 0.2021, swd: 0.1744, ept: 124.0318
    Epoch [6/50], Val Losses: mse: 0.6707, mae: 0.6345, huber: 0.2943, swd: 0.2119, ept: 65.5745
    Epoch [6/50], Test Losses: mse: 0.6519, mae: 0.6102, huber: 0.2846, swd: 0.2049, ept: 91.1894
      Epoch 6 composite train-obj: 0.202100
            No improvement (0.2943), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4545, mae: 0.4753, huber: 0.1955, swd: 0.1610, ept: 127.4856
    Epoch [7/50], Val Losses: mse: 0.6627, mae: 0.6215, huber: 0.2870, swd: 0.1785, ept: 70.3666
    Epoch [7/50], Test Losses: mse: 0.6546, mae: 0.6065, huber: 0.2841, swd: 0.1751, ept: 92.3618
      Epoch 7 composite train-obj: 0.195533
    Epoch [7/50], Test Losses: mse: 0.6359, mae: 0.5944, huber: 0.2769, swd: 0.1905, ept: 106.7432
    Best round's Test MSE: 0.6359, MAE: 0.5944, SWD: 0.1905
    Best round's Validation MSE: 0.5168, MAE: 0.5127, SWD: 0.1402
    Best round's Test verification MSE : 0.6359, MAE: 0.5944, SWD: 0.1905
    Time taken: 17.07 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq196_pred720_20250510_1834)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6453 ± 0.0220
      mae: 0.5994 ± 0.0101
      huber: 0.2801 ± 0.0075
      swd: 0.2172 ± 0.0196
      ept: 104.4740 ± 4.1674
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.5075 ± 0.0078
      mae: 0.5157 ± 0.0031
      huber: 0.2202 ± 0.0009
      swd: 0.1557 ± 0.0161
      ept: 89.2185 ± 3.6797
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 50.88 seconds
    
    Experiment complete: ACL_etth1_seq196_pred720_20250510_1834
    Model: ACL
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### Timemixer

#### pred=96


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=196,
    pred_len=96,
    channels=data_mgr.datasets['etth1']['channels'],
    enc_in=data_mgr.datasets['etth1']['channels'],
    dec_in=data_mgr.datasets['etth1']['channels'],
    c_out=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 96
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 12
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4468, mae: 0.4636, huber: 0.1898, swd: 0.1310, ept: 43.7867
    Epoch [1/50], Val Losses: mse: 0.3532, mae: 0.3962, huber: 0.1506, swd: 0.1029, ept: 49.4191
    Epoch [1/50], Test Losses: mse: 0.4444, mae: 0.4561, huber: 0.1889, swd: 0.1202, ept: 39.8141
      Epoch 1 composite train-obj: 0.189833
            Val objective improved inf → 0.1506, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3702, mae: 0.4133, huber: 0.1588, swd: 0.1129, ept: 51.1106
    Epoch [2/50], Val Losses: mse: 0.3353, mae: 0.3813, huber: 0.1432, swd: 0.1029, ept: 51.0707
    Epoch [2/50], Test Losses: mse: 0.4312, mae: 0.4499, huber: 0.1851, swd: 0.1339, ept: 40.8043
      Epoch 2 composite train-obj: 0.158816
            Val objective improved 0.1506 → 0.1432, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3595, mae: 0.4056, huber: 0.1545, swd: 0.1102, ept: 52.3401
    Epoch [3/50], Val Losses: mse: 0.3337, mae: 0.3828, huber: 0.1431, swd: 0.0979, ept: 50.9327
    Epoch [3/50], Test Losses: mse: 0.4272, mae: 0.4472, huber: 0.1832, swd: 0.1226, ept: 41.3055
      Epoch 3 composite train-obj: 0.154526
            Val objective improved 0.1432 → 0.1431, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3511, mae: 0.4008, huber: 0.1515, swd: 0.1070, ept: 52.9245
    Epoch [4/50], Val Losses: mse: 0.3316, mae: 0.3809, huber: 0.1424, swd: 0.1041, ept: 51.3447
    Epoch [4/50], Test Losses: mse: 0.4284, mae: 0.4505, huber: 0.1846, swd: 0.1344, ept: 41.1394
      Epoch 4 composite train-obj: 0.151506
            Val objective improved 0.1431 → 0.1424, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3443, mae: 0.3970, huber: 0.1490, swd: 0.1049, ept: 53.4114
    Epoch [5/50], Val Losses: mse: 0.3322, mae: 0.3832, huber: 0.1433, swd: 0.1024, ept: 51.5693
    Epoch [5/50], Test Losses: mse: 0.4245, mae: 0.4476, huber: 0.1828, swd: 0.1280, ept: 41.6423
      Epoch 5 composite train-obj: 0.149021
            No improvement (0.1433), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3382, mae: 0.3942, huber: 0.1469, swd: 0.1029, ept: 53.7186
    Epoch [6/50], Val Losses: mse: 0.3400, mae: 0.3872, huber: 0.1460, swd: 0.0998, ept: 51.2742
    Epoch [6/50], Test Losses: mse: 0.4278, mae: 0.4479, huber: 0.1835, swd: 0.1181, ept: 41.7384
      Epoch 6 composite train-obj: 0.146877
            No improvement (0.1460), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.3312, mae: 0.3910, huber: 0.1445, swd: 0.0995, ept: 54.0415
    Epoch [7/50], Val Losses: mse: 0.3371, mae: 0.3863, huber: 0.1453, swd: 0.0981, ept: 51.4409
    Epoch [7/50], Test Losses: mse: 0.4291, mae: 0.4511, huber: 0.1848, swd: 0.1200, ept: 41.5089
      Epoch 7 composite train-obj: 0.144455
            No improvement (0.1453), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.3235, mae: 0.3878, huber: 0.1418, swd: 0.0975, ept: 54.2129
    Epoch [8/50], Val Losses: mse: 0.3380, mae: 0.3903, huber: 0.1465, swd: 0.1048, ept: 51.1825
    Epoch [8/50], Test Losses: mse: 0.4311, mae: 0.4547, huber: 0.1864, swd: 0.1290, ept: 41.3145
      Epoch 8 composite train-obj: 0.141825
            No improvement (0.1465), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.3149, mae: 0.3842, huber: 0.1389, swd: 0.0946, ept: 54.4856
    Epoch [9/50], Val Losses: mse: 0.3403, mae: 0.3917, huber: 0.1474, swd: 0.1034, ept: 50.8518
    Epoch [9/50], Test Losses: mse: 0.4332, mae: 0.4552, huber: 0.1870, swd: 0.1230, ept: 41.4626
      Epoch 9 composite train-obj: 0.138882
    Epoch [9/50], Test Losses: mse: 0.4284, mae: 0.4505, huber: 0.1846, swd: 0.1344, ept: 41.1394
    Best round's Test MSE: 0.4284, MAE: 0.4505, SWD: 0.1344
    Best round's Validation MSE: 0.3316, MAE: 0.3809, SWD: 0.1041
    Best round's Test verification MSE : 0.4284, MAE: 0.4505, SWD: 0.1344
    Time taken: 29.29 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4388, mae: 0.4586, huber: 0.1866, swd: 0.1297, ept: 44.8660
    Epoch [1/50], Val Losses: mse: 0.3453, mae: 0.3917, huber: 0.1476, swd: 0.0976, ept: 49.9260
    Epoch [1/50], Test Losses: mse: 0.4431, mae: 0.4565, huber: 0.1890, swd: 0.1222, ept: 40.3060
      Epoch 1 composite train-obj: 0.186574
            Val objective improved inf → 0.1476, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3729, mae: 0.4140, huber: 0.1597, swd: 0.1132, ept: 51.4230
    Epoch [2/50], Val Losses: mse: 0.3360, mae: 0.3828, huber: 0.1432, swd: 0.0926, ept: 50.9613
    Epoch [2/50], Test Losses: mse: 0.4330, mae: 0.4502, huber: 0.1850, swd: 0.1214, ept: 41.0386
      Epoch 2 composite train-obj: 0.159716
            Val objective improved 0.1476 → 0.1432, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3632, mae: 0.4076, huber: 0.1559, swd: 0.1097, ept: 52.2672
    Epoch [3/50], Val Losses: mse: 0.3345, mae: 0.3811, huber: 0.1427, swd: 0.0928, ept: 51.4855
    Epoch [3/50], Test Losses: mse: 0.4286, mae: 0.4467, huber: 0.1832, swd: 0.1199, ept: 41.4256
      Epoch 3 composite train-obj: 0.155921
            Val objective improved 0.1432 → 0.1427, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3557, mae: 0.4029, huber: 0.1531, swd: 0.1075, ept: 52.7432
    Epoch [4/50], Val Losses: mse: 0.3334, mae: 0.3822, huber: 0.1429, swd: 0.0927, ept: 51.4977
    Epoch [4/50], Test Losses: mse: 0.4222, mae: 0.4437, huber: 0.1812, swd: 0.1191, ept: 41.4992
      Epoch 4 composite train-obj: 0.153110
            No improvement (0.1429), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3490, mae: 0.3996, huber: 0.1508, swd: 0.1050, ept: 53.1181
    Epoch [5/50], Val Losses: mse: 0.3330, mae: 0.3821, huber: 0.1427, swd: 0.0924, ept: 51.5959
    Epoch [5/50], Test Losses: mse: 0.4225, mae: 0.4439, huber: 0.1815, swd: 0.1159, ept: 41.7881
      Epoch 5 composite train-obj: 0.150822
            Val objective improved 0.1427 → 0.1427, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3428, mae: 0.3967, huber: 0.1487, swd: 0.1030, ept: 53.4330
    Epoch [6/50], Val Losses: mse: 0.3361, mae: 0.3858, huber: 0.1446, swd: 0.1022, ept: 51.7512
    Epoch [6/50], Test Losses: mse: 0.4287, mae: 0.4529, huber: 0.1853, swd: 0.1295, ept: 41.2449
      Epoch 6 composite train-obj: 0.148670
            No improvement (0.1446), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3354, mae: 0.3932, huber: 0.1461, swd: 0.1006, ept: 53.7419
    Epoch [7/50], Val Losses: mse: 0.3322, mae: 0.3825, huber: 0.1427, swd: 0.0911, ept: 52.1294
    Epoch [7/50], Test Losses: mse: 0.4226, mae: 0.4458, huber: 0.1819, swd: 0.1173, ept: 41.7773
      Epoch 7 composite train-obj: 0.146087
            No improvement (0.1427), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3269, mae: 0.3895, huber: 0.1432, swd: 0.0977, ept: 53.9363
    Epoch [8/50], Val Losses: mse: 0.3335, mae: 0.3841, huber: 0.1436, swd: 0.0926, ept: 51.9748
    Epoch [8/50], Test Losses: mse: 0.4246, mae: 0.4485, huber: 0.1831, swd: 0.1187, ept: 41.7144
      Epoch 8 composite train-obj: 0.143222
            No improvement (0.1436), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3187, mae: 0.3855, huber: 0.1403, swd: 0.0949, ept: 54.2678
    Epoch [9/50], Val Losses: mse: 0.3399, mae: 0.3882, huber: 0.1463, swd: 0.0943, ept: 51.4676
    Epoch [9/50], Test Losses: mse: 0.4285, mae: 0.4496, huber: 0.1843, swd: 0.1186, ept: 41.8403
      Epoch 9 composite train-obj: 0.140346
            No improvement (0.1463), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.3113, mae: 0.3821, huber: 0.1378, swd: 0.0923, ept: 54.4327
    Epoch [10/50], Val Losses: mse: 0.3398, mae: 0.3889, huber: 0.1464, swd: 0.0993, ept: 51.5355
    Epoch [10/50], Test Losses: mse: 0.4325, mae: 0.4549, huber: 0.1869, swd: 0.1250, ept: 41.2917
      Epoch 10 composite train-obj: 0.137755
    Epoch [10/50], Test Losses: mse: 0.4225, mae: 0.4439, huber: 0.1815, swd: 0.1159, ept: 41.7881
    Best round's Test MSE: 0.4225, MAE: 0.4439, SWD: 0.1159
    Best round's Validation MSE: 0.3330, MAE: 0.3821, SWD: 0.0924
    Best round's Test verification MSE : 0.4225, MAE: 0.4439, SWD: 0.1159
    Time taken: 33.69 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4604, mae: 0.4700, huber: 0.1946, swd: 0.1218, ept: 42.8776
    Epoch [1/50], Val Losses: mse: 0.3468, mae: 0.3936, huber: 0.1486, swd: 0.1021, ept: 49.3724
    Epoch [1/50], Test Losses: mse: 0.4429, mae: 0.4580, huber: 0.1896, swd: 0.1232, ept: 39.4451
      Epoch 1 composite train-obj: 0.194557
            Val objective improved inf → 0.1486, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3768, mae: 0.4170, huber: 0.1614, swd: 0.1085, ept: 50.7652
    Epoch [2/50], Val Losses: mse: 0.3399, mae: 0.3846, huber: 0.1449, swd: 0.0984, ept: 50.6086
    Epoch [2/50], Test Losses: mse: 0.4337, mae: 0.4492, huber: 0.1850, swd: 0.1183, ept: 41.1873
      Epoch 2 composite train-obj: 0.161367
            Val objective improved 0.1486 → 0.1449, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3656, mae: 0.4087, huber: 0.1567, swd: 0.1056, ept: 51.9423
    Epoch [3/50], Val Losses: mse: 0.3418, mae: 0.3857, huber: 0.1457, swd: 0.0959, ept: 50.8074
    Epoch [3/50], Test Losses: mse: 0.4307, mae: 0.4465, huber: 0.1836, swd: 0.1127, ept: 41.2440
      Epoch 3 composite train-obj: 0.156735
            No improvement (0.1457), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3577, mae: 0.4041, huber: 0.1538, swd: 0.1032, ept: 52.4720
    Epoch [4/50], Val Losses: mse: 0.3363, mae: 0.3820, huber: 0.1439, swd: 0.1005, ept: 51.1935
    Epoch [4/50], Test Losses: mse: 0.4255, mae: 0.4455, huber: 0.1826, swd: 0.1232, ept: 41.4706
      Epoch 4 composite train-obj: 0.153848
            Val objective improved 0.1449 → 0.1439, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3510, mae: 0.4006, huber: 0.1515, swd: 0.1013, ept: 52.8476
    Epoch [5/50], Val Losses: mse: 0.3295, mae: 0.3798, huber: 0.1416, swd: 0.0986, ept: 51.4464
    Epoch [5/50], Test Losses: mse: 0.4226, mae: 0.4455, huber: 0.1821, swd: 0.1266, ept: 41.1392
      Epoch 5 composite train-obj: 0.151462
            Val objective improved 0.1439 → 0.1416, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3449, mae: 0.3977, huber: 0.1493, swd: 0.0988, ept: 53.1995
    Epoch [6/50], Val Losses: mse: 0.3285, mae: 0.3807, huber: 0.1415, swd: 0.0988, ept: 51.5927
    Epoch [6/50], Test Losses: mse: 0.4245, mae: 0.4490, huber: 0.1835, swd: 0.1301, ept: 40.9277
      Epoch 6 composite train-obj: 0.149344
            Val objective improved 0.1416 → 0.1415, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3374, mae: 0.3944, huber: 0.1468, swd: 0.0966, ept: 53.4859
    Epoch [7/50], Val Losses: mse: 0.3381, mae: 0.3863, huber: 0.1454, swd: 0.0962, ept: 51.0257
    Epoch [7/50], Test Losses: mse: 0.4248, mae: 0.4470, huber: 0.1829, swd: 0.1133, ept: 41.6082
      Epoch 7 composite train-obj: 0.146838
            No improvement (0.1454), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.3309, mae: 0.3917, huber: 0.1447, swd: 0.0945, ept: 53.6888
    Epoch [8/50], Val Losses: mse: 0.3335, mae: 0.3850, huber: 0.1439, swd: 0.0962, ept: 51.1666
    Epoch [8/50], Test Losses: mse: 0.4282, mae: 0.4509, huber: 0.1846, swd: 0.1211, ept: 41.1121
      Epoch 8 composite train-obj: 0.144678
            No improvement (0.1439), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.3230, mae: 0.3882, huber: 0.1419, swd: 0.0918, ept: 54.0165
    Epoch [9/50], Val Losses: mse: 0.3387, mae: 0.3868, huber: 0.1456, swd: 0.0934, ept: 50.9783
    Epoch [9/50], Test Losses: mse: 0.4305, mae: 0.4518, huber: 0.1855, swd: 0.1148, ept: 41.3314
      Epoch 9 composite train-obj: 0.141919
            No improvement (0.1456), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.3146, mae: 0.3849, huber: 0.1392, swd: 0.0898, ept: 54.1670
    Epoch [10/50], Val Losses: mse: 0.3482, mae: 0.3941, huber: 0.1496, swd: 0.1009, ept: 50.8884
    Epoch [10/50], Test Losses: mse: 0.4417, mae: 0.4587, huber: 0.1894, swd: 0.1161, ept: 41.2605
      Epoch 10 composite train-obj: 0.139188
            No improvement (0.1496), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.3068, mae: 0.3812, huber: 0.1364, swd: 0.0874, ept: 54.4628
    Epoch [11/50], Val Losses: mse: 0.3410, mae: 0.3922, huber: 0.1473, swd: 0.0999, ept: 51.0143
    Epoch [11/50], Test Losses: mse: 0.4354, mae: 0.4570, huber: 0.1881, swd: 0.1194, ept: 41.0581
      Epoch 11 composite train-obj: 0.136373
    Epoch [11/50], Test Losses: mse: 0.4245, mae: 0.4490, huber: 0.1835, swd: 0.1301, ept: 40.9277
    Best round's Test MSE: 0.4245, MAE: 0.4490, SWD: 0.1301
    Best round's Validation MSE: 0.3285, MAE: 0.3807, SWD: 0.0988
    Best round's Test verification MSE : 0.4245, MAE: 0.4490, SWD: 0.1301
    Time taken: 38.99 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq196_pred96_20250510_1835)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4251 ± 0.0024
      mae: 0.4478 ± 0.0028
      huber: 0.1832 ± 0.0013
      swd: 0.1268 ± 0.0079
      ept: 41.2851 ± 0.3661
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3311 ± 0.0019
      mae: 0.3812 ± 0.0006
      huber: 0.1422 ± 0.0005
      swd: 0.0984 ± 0.0048
      ept: 51.5111 ± 0.1177
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 102.04 seconds
    
    Experiment complete: TimeMixer_etth1_seq196_pred96_20250510_1835
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=196,
    pred_len=196,
    channels=data_mgr.datasets['etth1']['channels'],
    enc_in=data_mgr.datasets['etth1']['channels'],
    dec_in=data_mgr.datasets['etth1']['channels'],
    c_out=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 196
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 11
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5511, mae: 0.5115, huber: 0.2249, swd: 0.1654, ept: 64.3430
    Epoch [1/50], Val Losses: mse: 0.4054, mae: 0.4291, huber: 0.1708, swd: 0.1097, ept: 67.2531
    Epoch [1/50], Test Losses: mse: 0.4847, mae: 0.4887, huber: 0.2083, swd: 0.1168, ept: 58.7366
      Epoch 1 composite train-obj: 0.224915
            Val objective improved inf → 0.1708, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4645, mae: 0.4604, huber: 0.1924, swd: 0.1445, ept: 75.5055
    Epoch [2/50], Val Losses: mse: 0.3936, mae: 0.4179, huber: 0.1652, swd: 0.1052, ept: 69.7187
    Epoch [2/50], Test Losses: mse: 0.4721, mae: 0.4815, huber: 0.2036, swd: 0.1138, ept: 60.6019
      Epoch 2 composite train-obj: 0.192380
            Val objective improved 0.1708 → 0.1652, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4517, mae: 0.4529, huber: 0.1875, swd: 0.1415, ept: 77.6820
    Epoch [3/50], Val Losses: mse: 0.3966, mae: 0.4215, huber: 0.1669, swd: 0.1139, ept: 70.1230
    Epoch [3/50], Test Losses: mse: 0.4741, mae: 0.4838, huber: 0.2048, swd: 0.1244, ept: 60.8518
      Epoch 3 composite train-obj: 0.187479
            No improvement (0.1669), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4422, mae: 0.4488, huber: 0.1844, swd: 0.1390, ept: 78.6521
    Epoch [4/50], Val Losses: mse: 0.4003, mae: 0.4250, huber: 0.1686, swd: 0.1098, ept: 70.1635
    Epoch [4/50], Test Losses: mse: 0.4696, mae: 0.4817, huber: 0.2032, swd: 0.1154, ept: 61.6901
      Epoch 4 composite train-obj: 0.184359
            No improvement (0.1686), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4331, mae: 0.4456, huber: 0.1816, swd: 0.1364, ept: 79.3201
    Epoch [5/50], Val Losses: mse: 0.4137, mae: 0.4353, huber: 0.1744, swd: 0.1151, ept: 70.1098
    Epoch [5/50], Test Losses: mse: 0.4777, mae: 0.4894, huber: 0.2072, swd: 0.1193, ept: 60.7221
      Epoch 5 composite train-obj: 0.181629
            No improvement (0.1744), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4239, mae: 0.4422, huber: 0.1788, swd: 0.1321, ept: 79.8123
    Epoch [6/50], Val Losses: mse: 0.4183, mae: 0.4401, huber: 0.1770, swd: 0.1116, ept: 70.7300
    Epoch [6/50], Test Losses: mse: 0.4765, mae: 0.4896, huber: 0.2071, swd: 0.1187, ept: 60.6899
      Epoch 6 composite train-obj: 0.178811
            No improvement (0.1770), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4097, mae: 0.4371, huber: 0.1745, swd: 0.1274, ept: 79.8914
    Epoch [7/50], Val Losses: mse: 0.4245, mae: 0.4426, huber: 0.1789, swd: 0.1102, ept: 70.8497
    Epoch [7/50], Test Losses: mse: 0.4824, mae: 0.4945, huber: 0.2095, swd: 0.1123, ept: 60.2811
      Epoch 7 composite train-obj: 0.174541
    Epoch [7/50], Test Losses: mse: 0.4721, mae: 0.4815, huber: 0.2036, swd: 0.1138, ept: 60.6019
    Best round's Test MSE: 0.4721, MAE: 0.4815, SWD: 0.1138
    Best round's Validation MSE: 0.3936, MAE: 0.4179, SWD: 0.1052
    Best round's Test verification MSE : 0.4721, MAE: 0.4815, SWD: 0.1138
    Time taken: 25.41 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5333, mae: 0.5023, huber: 0.2188, swd: 0.1652, ept: 65.3891
    Epoch [1/50], Val Losses: mse: 0.3973, mae: 0.4264, huber: 0.1682, swd: 0.1077, ept: 67.8603
    Epoch [1/50], Test Losses: mse: 0.4845, mae: 0.4901, huber: 0.2088, swd: 0.1246, ept: 58.8260
      Epoch 1 composite train-obj: 0.218757
            Val objective improved inf → 0.1682, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4678, mae: 0.4619, huber: 0.1935, swd: 0.1451, ept: 75.4596
    Epoch [2/50], Val Losses: mse: 0.3924, mae: 0.4158, huber: 0.1641, swd: 0.1020, ept: 70.9364
    Epoch [2/50], Test Losses: mse: 0.4738, mae: 0.4799, huber: 0.2033, swd: 0.1105, ept: 61.5779
      Epoch 2 composite train-obj: 0.193511
            Val objective improved 0.1682 → 0.1641, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4580, mae: 0.4552, huber: 0.1896, swd: 0.1420, ept: 77.2173
    Epoch [3/50], Val Losses: mse: 0.3895, mae: 0.4136, huber: 0.1634, swd: 0.1099, ept: 71.3203
    Epoch [3/50], Test Losses: mse: 0.4706, mae: 0.4791, huber: 0.2028, swd: 0.1228, ept: 61.8355
      Epoch 3 composite train-obj: 0.189597
            Val objective improved 0.1641 → 0.1634, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4500, mae: 0.4512, huber: 0.1868, swd: 0.1404, ept: 77.9120
    Epoch [4/50], Val Losses: mse: 0.3852, mae: 0.4135, huber: 0.1626, swd: 0.1061, ept: 71.2171
    Epoch [4/50], Test Losses: mse: 0.4692, mae: 0.4807, huber: 0.2031, swd: 0.1233, ept: 60.8045
      Epoch 4 composite train-obj: 0.186814
            Val objective improved 0.1634 → 0.1626, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4406, mae: 0.4473, huber: 0.1838, swd: 0.1378, ept: 78.5659
    Epoch [5/50], Val Losses: mse: 0.3917, mae: 0.4175, huber: 0.1650, swd: 0.1048, ept: 70.7871
    Epoch [5/50], Test Losses: mse: 0.4693, mae: 0.4803, huber: 0.2029, swd: 0.1153, ept: 61.3847
      Epoch 5 composite train-obj: 0.183754
            No improvement (0.1650), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4292, mae: 0.4430, huber: 0.1802, swd: 0.1342, ept: 79.2478
    Epoch [6/50], Val Losses: mse: 0.3884, mae: 0.4207, huber: 0.1651, swd: 0.1019, ept: 70.8801
    Epoch [6/50], Test Losses: mse: 0.4701, mae: 0.4835, huber: 0.2042, swd: 0.1180, ept: 60.5436
      Epoch 6 composite train-obj: 0.180182
            No improvement (0.1651), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.4169, mae: 0.4393, huber: 0.1768, swd: 0.1302, ept: 79.6158
    Epoch [7/50], Val Losses: mse: 0.4211, mae: 0.4344, huber: 0.1756, swd: 0.1061, ept: 69.9925
    Epoch [7/50], Test Losses: mse: 0.4793, mae: 0.4875, huber: 0.2065, swd: 0.1063, ept: 60.8750
      Epoch 7 composite train-obj: 0.176780
            No improvement (0.1756), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.4041, mae: 0.4346, huber: 0.1728, swd: 0.1259, ept: 79.7049
    Epoch [8/50], Val Losses: mse: 0.4189, mae: 0.4382, huber: 0.1764, swd: 0.1039, ept: 70.0232
    Epoch [8/50], Test Losses: mse: 0.4767, mae: 0.4888, huber: 0.2067, swd: 0.1102, ept: 59.9960
      Epoch 8 composite train-obj: 0.172831
            No improvement (0.1764), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.3877, mae: 0.4282, huber: 0.1677, swd: 0.1208, ept: 80.0517
    Epoch [9/50], Val Losses: mse: 0.4076, mae: 0.4340, huber: 0.1730, swd: 0.1070, ept: 69.8167
    Epoch [9/50], Test Losses: mse: 0.4800, mae: 0.4929, huber: 0.2090, swd: 0.1192, ept: 59.8828
      Epoch 9 composite train-obj: 0.167685
    Epoch [9/50], Test Losses: mse: 0.4692, mae: 0.4807, huber: 0.2031, swd: 0.1233, ept: 60.8045
    Best round's Test MSE: 0.4692, MAE: 0.4807, SWD: 0.1233
    Best round's Validation MSE: 0.3852, MAE: 0.4135, SWD: 0.1061
    Best round's Test verification MSE : 0.4692, MAE: 0.4807, SWD: 0.1233
    Time taken: 29.71 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5682, mae: 0.5196, huber: 0.2311, swd: 0.1446, ept: 61.0290
    Epoch [1/50], Val Losses: mse: 0.4035, mae: 0.4288, huber: 0.1703, swd: 0.1076, ept: 67.9568
    Epoch [1/50], Test Losses: mse: 0.4847, mae: 0.4891, huber: 0.2086, swd: 0.1137, ept: 58.7513
      Epoch 1 composite train-obj: 0.231058
            Val objective improved inf → 0.1703, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4683, mae: 0.4637, huber: 0.1942, swd: 0.1325, ept: 74.7813
    Epoch [2/50], Val Losses: mse: 0.3912, mae: 0.4172, huber: 0.1646, swd: 0.1005, ept: 70.3479
    Epoch [2/50], Test Losses: mse: 0.4705, mae: 0.4792, huber: 0.2026, swd: 0.1072, ept: 60.9692
      Epoch 2 composite train-obj: 0.194174
            Val objective improved 0.1703 → 0.1646, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4561, mae: 0.4554, huber: 0.1893, swd: 0.1293, ept: 77.1251
    Epoch [3/50], Val Losses: mse: 0.3952, mae: 0.4167, huber: 0.1656, swd: 0.1066, ept: 71.2869
    Epoch [3/50], Test Losses: mse: 0.4635, mae: 0.4747, huber: 0.2000, swd: 0.1111, ept: 61.1545
      Epoch 3 composite train-obj: 0.189317
            No improvement (0.1656), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4472, mae: 0.4506, huber: 0.1861, swd: 0.1273, ept: 78.1976
    Epoch [4/50], Val Losses: mse: 0.4073, mae: 0.4250, huber: 0.1704, swd: 0.1083, ept: 70.3280
    Epoch [4/50], Test Losses: mse: 0.4674, mae: 0.4784, huber: 0.2017, swd: 0.1079, ept: 60.7797
      Epoch 4 composite train-obj: 0.186135
            No improvement (0.1704), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4382, mae: 0.4467, huber: 0.1831, swd: 0.1259, ept: 78.6085
    Epoch [5/50], Val Losses: mse: 0.4089, mae: 0.4270, huber: 0.1712, swd: 0.1081, ept: 70.4703
    Epoch [5/50], Test Losses: mse: 0.4689, mae: 0.4808, huber: 0.2028, swd: 0.1084, ept: 60.6171
      Epoch 5 composite train-obj: 0.183141
            No improvement (0.1712), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4268, mae: 0.4424, huber: 0.1796, swd: 0.1235, ept: 79.0422
    Epoch [6/50], Val Losses: mse: 0.4073, mae: 0.4276, huber: 0.1712, swd: 0.1047, ept: 70.3901
    Epoch [6/50], Test Losses: mse: 0.4753, mae: 0.4867, huber: 0.2058, swd: 0.1085, ept: 60.4600
      Epoch 6 composite train-obj: 0.179598
            No improvement (0.1712), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4144, mae: 0.4387, huber: 0.1762, swd: 0.1200, ept: 79.2403
    Epoch [7/50], Val Losses: mse: 0.3976, mae: 0.4239, huber: 0.1681, swd: 0.0981, ept: 71.0871
    Epoch [7/50], Test Losses: mse: 0.4748, mae: 0.4872, huber: 0.2062, swd: 0.1034, ept: 60.1279
      Epoch 7 composite train-obj: 0.176166
    Epoch [7/50], Test Losses: mse: 0.4705, mae: 0.4792, huber: 0.2026, swd: 0.1072, ept: 60.9692
    Best round's Test MSE: 0.4705, MAE: 0.4792, SWD: 0.1072
    Best round's Validation MSE: 0.3912, MAE: 0.4172, SWD: 0.1005
    Best round's Test verification MSE : 0.4705, MAE: 0.4792, SWD: 0.1072
    Time taken: 24.57 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq196_pred196_20250510_1837)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4706 ± 0.0012
      mae: 0.4804 ± 0.0010
      huber: 0.2031 ± 0.0004
      swd: 0.1148 ± 0.0066
      ept: 60.7919 ± 0.1502
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3900 ± 0.0035
      mae: 0.4162 ± 0.0019
      huber: 0.1641 ± 0.0011
      swd: 0.1039 ± 0.0024
      ept: 70.4279 ± 0.6144
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 79.75 seconds
    
    Experiment complete: TimeMixer_etth1_seq196_pred196_20250510_1837
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=196,
    pred_len=336,
    channels=data_mgr.datasets['etth1']['channels'],
    enc_in=data_mgr.datasets['etth1']['channels'],
    dec_in=data_mgr.datasets['etth1']['channels'],
    c_out=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 10
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6292, mae: 0.5491, huber: 0.2534, swd: 0.1690, ept: 75.9974
    Epoch [1/50], Val Losses: mse: 0.4287, mae: 0.4370, huber: 0.1780, swd: 0.1050, ept: 88.1314
    Epoch [1/50], Test Losses: mse: 0.5278, mae: 0.5177, huber: 0.2281, swd: 0.1198, ept: 77.8807
      Epoch 1 composite train-obj: 0.253437
            Val objective improved inf → 0.1780, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5492, mae: 0.5026, huber: 0.2229, swd: 0.1611, ept: 93.9562
    Epoch [2/50], Val Losses: mse: 0.4221, mae: 0.4266, huber: 0.1736, swd: 0.1075, ept: 90.6042
    Epoch [2/50], Test Losses: mse: 0.5211, mae: 0.5131, huber: 0.2254, swd: 0.1242, ept: 80.6699
      Epoch 2 composite train-obj: 0.222882
            Val objective improved 0.1780 → 0.1736, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5356, mae: 0.4944, huber: 0.2179, swd: 0.1588, ept: 97.3663
    Epoch [3/50], Val Losses: mse: 0.4337, mae: 0.4318, huber: 0.1771, swd: 0.0997, ept: 90.2754
    Epoch [3/50], Test Losses: mse: 0.5227, mae: 0.5130, huber: 0.2253, swd: 0.1071, ept: 81.9446
      Epoch 3 composite train-obj: 0.217876
            No improvement (0.1771), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5205, mae: 0.4885, huber: 0.2130, swd: 0.1558, ept: 98.7250
    Epoch [4/50], Val Losses: mse: 0.4405, mae: 0.4404, huber: 0.1813, swd: 0.1055, ept: 89.4847
    Epoch [4/50], Test Losses: mse: 0.5257, mae: 0.5179, huber: 0.2274, swd: 0.1193, ept: 80.8143
      Epoch 4 composite train-obj: 0.213013
            No improvement (0.1813), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5066, mae: 0.4837, huber: 0.2088, swd: 0.1532, ept: 99.7399
    Epoch [5/50], Val Losses: mse: 0.4383, mae: 0.4438, huber: 0.1820, swd: 0.1070, ept: 89.4693
    Epoch [5/50], Test Losses: mse: 0.5335, mae: 0.5255, huber: 0.2320, swd: 0.1340, ept: 79.5273
      Epoch 5 composite train-obj: 0.208802
            No improvement (0.1820), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4917, mae: 0.4787, huber: 0.2043, swd: 0.1498, ept: 100.5374
    Epoch [6/50], Val Losses: mse: 0.4404, mae: 0.4497, huber: 0.1847, swd: 0.1138, ept: 88.1310
    Epoch [6/50], Test Losses: mse: 0.5496, mae: 0.5364, huber: 0.2391, swd: 0.1460, ept: 78.4668
      Epoch 6 composite train-obj: 0.204297
            No improvement (0.1847), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4761, mae: 0.4732, huber: 0.1995, swd: 0.1458, ept: 101.1121
    Epoch [7/50], Val Losses: mse: 0.4709, mae: 0.4596, huber: 0.1932, swd: 0.1095, ept: 89.6319
    Epoch [7/50], Test Losses: mse: 0.5410, mae: 0.5292, huber: 0.2347, swd: 0.1177, ept: 79.5743
      Epoch 7 composite train-obj: 0.199509
    Epoch [7/50], Test Losses: mse: 0.5211, mae: 0.5131, huber: 0.2254, swd: 0.1242, ept: 80.6699
    Best round's Test MSE: 0.5211, MAE: 0.5131, SWD: 0.1242
    Best round's Validation MSE: 0.4221, MAE: 0.4266, SWD: 0.1075
    Best round's Test verification MSE : 0.5211, MAE: 0.5131, SWD: 0.1242
    Time taken: 27.69 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6399, mae: 0.5539, huber: 0.2562, swd: 0.1985, ept: 78.2314
    Epoch [1/50], Val Losses: mse: 0.4245, mae: 0.4381, huber: 0.1775, swd: 0.0971, ept: 86.9867
    Epoch [1/50], Test Losses: mse: 0.5285, mae: 0.5189, huber: 0.2285, swd: 0.1234, ept: 78.1879
      Epoch 1 composite train-obj: 0.256219
            Val objective improved inf → 0.1775, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5460, mae: 0.5018, huber: 0.2220, swd: 0.1664, ept: 94.1372
    Epoch [2/50], Val Losses: mse: 0.4052, mae: 0.4223, huber: 0.1691, swd: 0.0946, ept: 89.5029
    Epoch [2/50], Test Losses: mse: 0.5187, mae: 0.5101, huber: 0.2243, swd: 0.1301, ept: 81.2639
      Epoch 2 composite train-obj: 0.222009
            Val objective improved 0.1775 → 0.1691, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5316, mae: 0.4930, huber: 0.2165, swd: 0.1618, ept: 97.7715
    Epoch [3/50], Val Losses: mse: 0.4263, mae: 0.4278, huber: 0.1745, swd: 0.0984, ept: 90.3764
    Epoch [3/50], Test Losses: mse: 0.5189, mae: 0.5107, huber: 0.2242, swd: 0.1173, ept: 82.3374
      Epoch 3 composite train-obj: 0.216534
            No improvement (0.1745), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5197, mae: 0.4884, huber: 0.2128, swd: 0.1598, ept: 98.8339
    Epoch [4/50], Val Losses: mse: 0.4491, mae: 0.4398, huber: 0.1819, swd: 0.0978, ept: 91.0210
    Epoch [4/50], Test Losses: mse: 0.5232, mae: 0.5133, huber: 0.2257, swd: 0.1094, ept: 82.4086
      Epoch 4 composite train-obj: 0.212814
            No improvement (0.1819), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5092, mae: 0.4845, huber: 0.2095, swd: 0.1564, ept: 99.6654
    Epoch [5/50], Val Losses: mse: 0.4635, mae: 0.4504, huber: 0.1878, swd: 0.1027, ept: 90.2012
    Epoch [5/50], Test Losses: mse: 0.5405, mae: 0.5271, huber: 0.2336, swd: 0.1163, ept: 82.2153
      Epoch 5 composite train-obj: 0.209474
            No improvement (0.1878), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4957, mae: 0.4802, huber: 0.2055, swd: 0.1533, ept: 100.1105
    Epoch [6/50], Val Losses: mse: 0.4625, mae: 0.4579, huber: 0.1907, swd: 0.1089, ept: 89.8115
    Epoch [6/50], Test Losses: mse: 0.5404, mae: 0.5284, huber: 0.2342, swd: 0.1231, ept: 81.4191
      Epoch 6 composite train-obj: 0.205497
            No improvement (0.1907), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4811, mae: 0.4752, huber: 0.2010, swd: 0.1493, ept: 100.8508
    Epoch [7/50], Val Losses: mse: 0.4647, mae: 0.4629, huber: 0.1931, swd: 0.1105, ept: 90.2911
    Epoch [7/50], Test Losses: mse: 0.5381, mae: 0.5280, huber: 0.2336, swd: 0.1281, ept: 80.9720
      Epoch 7 composite train-obj: 0.200951
    Epoch [7/50], Test Losses: mse: 0.5187, mae: 0.5101, huber: 0.2243, swd: 0.1301, ept: 81.2639
    Best round's Test MSE: 0.5187, MAE: 0.5101, SWD: 0.1301
    Best round's Validation MSE: 0.4052, MAE: 0.4223, SWD: 0.0946
    Best round's Test verification MSE : 0.5187, MAE: 0.5101, SWD: 0.1301
    Time taken: 28.75 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7466, mae: 0.5984, huber: 0.2917, swd: 0.1818, ept: 65.3097
    Epoch [1/50], Val Losses: mse: 0.4401, mae: 0.4450, huber: 0.1827, swd: 0.1116, ept: 87.2732
    Epoch [1/50], Test Losses: mse: 0.5332, mae: 0.5240, huber: 0.2314, swd: 0.1202, ept: 77.3100
      Epoch 1 composite train-obj: 0.291712
            Val objective improved inf → 0.1827, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5527, mae: 0.5070, huber: 0.2250, swd: 0.1630, ept: 91.8069
    Epoch [2/50], Val Losses: mse: 0.4183, mae: 0.4282, huber: 0.1733, swd: 0.1028, ept: 90.4426
    Epoch [2/50], Test Losses: mse: 0.5189, mae: 0.5124, huber: 0.2247, swd: 0.1188, ept: 80.4217
      Epoch 2 composite train-obj: 0.224989
            Val objective improved 0.1827 → 0.1733, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5416, mae: 0.4988, huber: 0.2203, swd: 0.1615, ept: 95.8731
    Epoch [3/50], Val Losses: mse: 0.4195, mae: 0.4270, huber: 0.1731, swd: 0.1101, ept: 90.6637
    Epoch [3/50], Test Losses: mse: 0.5232, mae: 0.5165, huber: 0.2272, swd: 0.1300, ept: 81.1886
      Epoch 3 composite train-obj: 0.220310
            Val objective improved 0.1733 → 0.1731, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5328, mae: 0.4940, huber: 0.2172, swd: 0.1606, ept: 98.0200
    Epoch [4/50], Val Losses: mse: 0.4209, mae: 0.4266, huber: 0.1733, swd: 0.1040, ept: 91.7494
    Epoch [4/50], Test Losses: mse: 0.5171, mae: 0.5100, huber: 0.2236, swd: 0.1175, ept: 80.9316
      Epoch 4 composite train-obj: 0.217166
            No improvement (0.1733), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5229, mae: 0.4898, huber: 0.2139, swd: 0.1594, ept: 99.1356
    Epoch [5/50], Val Losses: mse: 0.4419, mae: 0.4389, huber: 0.1808, swd: 0.1048, ept: 89.8174
    Epoch [5/50], Test Losses: mse: 0.5290, mae: 0.5179, huber: 0.2279, swd: 0.1111, ept: 80.5253
      Epoch 5 composite train-obj: 0.213933
            No improvement (0.1808), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5164, mae: 0.4873, huber: 0.2118, swd: 0.1575, ept: 99.5810
    Epoch [6/50], Val Losses: mse: 0.4602, mae: 0.4522, huber: 0.1879, swd: 0.1098, ept: 88.7475
    Epoch [6/50], Test Losses: mse: 0.5350, mae: 0.5252, huber: 0.2315, swd: 0.1199, ept: 80.5607
      Epoch 6 composite train-obj: 0.211794
            No improvement (0.1879), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5042, mae: 0.4832, huber: 0.2081, swd: 0.1546, ept: 100.1614
    Epoch [7/50], Val Losses: mse: 0.4469, mae: 0.4464, huber: 0.1840, swd: 0.1066, ept: 89.5220
    Epoch [7/50], Test Losses: mse: 0.5301, mae: 0.5214, huber: 0.2298, swd: 0.1206, ept: 80.3994
      Epoch 7 composite train-obj: 0.208051
            No improvement (0.1840), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4949, mae: 0.4799, huber: 0.2052, swd: 0.1523, ept: 101.0449
    Epoch [8/50], Val Losses: mse: 0.4587, mae: 0.4515, huber: 0.1878, swd: 0.1074, ept: 89.2619
    Epoch [8/50], Test Losses: mse: 0.5326, mae: 0.5222, huber: 0.2306, swd: 0.1144, ept: 80.1354
      Epoch 8 composite train-obj: 0.205199
    Epoch [8/50], Test Losses: mse: 0.5232, mae: 0.5165, huber: 0.2272, swd: 0.1300, ept: 81.1886
    Best round's Test MSE: 0.5232, MAE: 0.5165, SWD: 0.1300
    Best round's Validation MSE: 0.4195, MAE: 0.4270, SWD: 0.1101
    Best round's Test verification MSE : 0.5232, MAE: 0.5165, SWD: 0.1300
    Time taken: 30.59 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq196_pred336_20250510_1838)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5210 ± 0.0019
      mae: 0.5132 ± 0.0026
      huber: 0.2256 ± 0.0012
      swd: 0.1281 ± 0.0027
      ept: 81.0408 ± 0.2641
      count: 10.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4156 ± 0.0074
      mae: 0.4253 ± 0.0021
      huber: 0.1719 ± 0.0020
      swd: 0.1041 ± 0.0068
      ept: 90.2569 ± 0.5338
      count: 10.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 87.10 seconds
    
    Experiment complete: TimeMixer_etth1_seq196_pred336_20250510_1838
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=196,
    pred_len=720,
    channels=data_mgr.datasets['etth1']['channels'],
    enc_in=data_mgr.datasets['etth1']['channels'],
    dec_in=data_mgr.datasets['etth1']['channels'],
    c_out=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 7
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7774, mae: 0.6196, huber: 0.3062, swd: 0.2118, ept: 89.1169
    Epoch [1/50], Val Losses: mse: 0.4076, mae: 0.4380, huber: 0.1763, swd: 0.0801, ept: 140.6803
    Epoch [1/50], Test Losses: mse: 0.6688, mae: 0.6010, huber: 0.2870, swd: 0.1447, ept: 112.0610
      Epoch 1 composite train-obj: 0.306185
            Val objective improved inf → 0.1763, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6937, mae: 0.5764, huber: 0.2758, swd: 0.2079, ept: 113.9726
    Epoch [2/50], Val Losses: mse: 0.4215, mae: 0.4417, huber: 0.1796, swd: 0.0836, ept: 143.1447
    Epoch [2/50], Test Losses: mse: 0.6690, mae: 0.6016, huber: 0.2874, swd: 0.1468, ept: 111.9508
      Epoch 2 composite train-obj: 0.275761
            No improvement (0.1796), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.6717, mae: 0.5676, huber: 0.2681, swd: 0.2069, ept: 117.9962
    Epoch [3/50], Val Losses: mse: 0.4969, mae: 0.4847, huber: 0.2069, swd: 0.0871, ept: 145.2194
    Epoch [3/50], Test Losses: mse: 0.7055, mae: 0.6232, huber: 0.3010, swd: 0.1477, ept: 112.0658
      Epoch 3 composite train-obj: 0.268112
            No improvement (0.2069), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.6528, mae: 0.5599, huber: 0.2616, swd: 0.2040, ept: 120.1751
    Epoch [4/50], Val Losses: mse: 0.4880, mae: 0.4791, huber: 0.2037, swd: 0.0776, ept: 145.4548
    Epoch [4/50], Test Losses: mse: 0.7018, mae: 0.6193, huber: 0.2993, swd: 0.1486, ept: 112.4468
      Epoch 4 composite train-obj: 0.261622
            No improvement (0.2037), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.6301, mae: 0.5506, huber: 0.2541, swd: 0.2007, ept: 121.1138
    Epoch [5/50], Val Losses: mse: 0.5272, mae: 0.4989, huber: 0.2182, swd: 0.0727, ept: 144.6206
    Epoch [5/50], Test Losses: mse: 0.7243, mae: 0.6325, huber: 0.3083, swd: 0.1537, ept: 112.2781
      Epoch 5 composite train-obj: 0.254122
            No improvement (0.2182), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.6076, mae: 0.5405, huber: 0.2463, swd: 0.1988, ept: 123.9773
    Epoch [6/50], Val Losses: mse: 0.5489, mae: 0.5143, huber: 0.2282, swd: 0.0720, ept: 143.9898
    Epoch [6/50], Test Losses: mse: 0.7289, mae: 0.6321, huber: 0.3099, swd: 0.1657, ept: 110.5757
      Epoch 6 composite train-obj: 0.246332
    Epoch [6/50], Test Losses: mse: 0.6688, mae: 0.6010, huber: 0.2870, swd: 0.1447, ept: 112.0610
    Best round's Test MSE: 0.6688, MAE: 0.6010, SWD: 0.1447
    Best round's Validation MSE: 0.4076, MAE: 0.4380, SWD: 0.0801
    Best round's Test verification MSE : 0.6688, MAE: 0.6010, SWD: 0.1447
    Time taken: 26.01 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7727, mae: 0.6164, huber: 0.3040, swd: 0.2158, ept: 94.1747
    Epoch [1/50], Val Losses: mse: 0.3953, mae: 0.4308, huber: 0.1716, swd: 0.0803, ept: 143.2200
    Epoch [1/50], Test Losses: mse: 0.6653, mae: 0.5983, huber: 0.2855, swd: 0.1487, ept: 113.9316
      Epoch 1 composite train-obj: 0.304017
            Val objective improved inf → 0.1716, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6944, mae: 0.5766, huber: 0.2761, swd: 0.2000, ept: 116.5849
    Epoch [2/50], Val Losses: mse: 0.3995, mae: 0.4271, huber: 0.1714, swd: 0.0808, ept: 146.3337
    Epoch [2/50], Test Losses: mse: 0.6597, mae: 0.5930, huber: 0.2829, swd: 0.1460, ept: 114.4966
      Epoch 2 composite train-obj: 0.276081
            Val objective improved 0.1716 → 0.1714, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6761, mae: 0.5678, huber: 0.2695, swd: 0.1994, ept: 120.4678
    Epoch [3/50], Val Losses: mse: 0.3988, mae: 0.4290, huber: 0.1715, swd: 0.0755, ept: 145.5738
    Epoch [3/50], Test Losses: mse: 0.7175, mae: 0.6244, huber: 0.3044, swd: 0.1503, ept: 114.9597
      Epoch 3 composite train-obj: 0.269485
            No improvement (0.1715), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.6481, mae: 0.5580, huber: 0.2606, swd: 0.2020, ept: 122.4770
    Epoch [4/50], Val Losses: mse: 0.4140, mae: 0.4366, huber: 0.1762, swd: 0.0682, ept: 145.2505
    Epoch [4/50], Test Losses: mse: 0.7686, mae: 0.6451, huber: 0.3215, swd: 0.1377, ept: 113.5320
      Epoch 4 composite train-obj: 0.260585
            No improvement (0.1762), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.6194, mae: 0.5476, huber: 0.2517, swd: 0.1962, ept: 124.8450
    Epoch [5/50], Val Losses: mse: 0.4302, mae: 0.4527, huber: 0.1841, swd: 0.0761, ept: 143.3622
    Epoch [5/50], Test Losses: mse: 0.7803, mae: 0.6551, huber: 0.3269, swd: 0.1426, ept: 114.9816
      Epoch 5 composite train-obj: 0.251744
            No improvement (0.1841), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.5960, mae: 0.5383, huber: 0.2440, swd: 0.1920, ept: 125.6528
    Epoch [6/50], Val Losses: mse: 0.4349, mae: 0.4541, huber: 0.1854, swd: 0.0719, ept: 145.2470
    Epoch [6/50], Test Losses: mse: 0.7578, mae: 0.6450, huber: 0.3188, swd: 0.1375, ept: 112.3420
      Epoch 6 composite train-obj: 0.244001
            No improvement (0.1854), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.5739, mae: 0.5287, huber: 0.2365, swd: 0.1847, ept: 127.5374
    Epoch [7/50], Val Losses: mse: 0.4626, mae: 0.4778, huber: 0.1984, swd: 0.0806, ept: 145.0648
    Epoch [7/50], Test Losses: mse: 0.7785, mae: 0.6555, huber: 0.3263, swd: 0.1487, ept: 112.2215
      Epoch 7 composite train-obj: 0.236510
    Epoch [7/50], Test Losses: mse: 0.6597, mae: 0.5930, huber: 0.2829, swd: 0.1460, ept: 114.4966
    Best round's Test MSE: 0.6597, MAE: 0.5930, SWD: 0.1460
    Best round's Validation MSE: 0.3995, MAE: 0.4271, SWD: 0.0808
    Best round's Test verification MSE : 0.6597, MAE: 0.5930, SWD: 0.1460
    Time taken: 30.37 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.8123, mae: 0.6357, huber: 0.3187, swd: 0.2329, ept: 85.1039
    Epoch [1/50], Val Losses: mse: 0.4184, mae: 0.4442, huber: 0.1805, swd: 0.0869, ept: 141.6264
    Epoch [1/50], Test Losses: mse: 0.6730, mae: 0.6046, huber: 0.2889, swd: 0.1420, ept: 109.3322
      Epoch 1 composite train-obj: 0.318730
            Val objective improved inf → 0.1805, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6992, mae: 0.5800, huber: 0.2780, swd: 0.2171, ept: 110.9321
    Epoch [2/50], Val Losses: mse: 0.4051, mae: 0.4351, huber: 0.1750, swd: 0.0908, ept: 144.8557
    Epoch [2/50], Test Losses: mse: 0.6699, mae: 0.6025, huber: 0.2880, swd: 0.1550, ept: 111.2041
      Epoch 2 composite train-obj: 0.277953
            Val objective improved 0.1805 → 0.1750, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6824, mae: 0.5718, huber: 0.2721, swd: 0.2150, ept: 115.6823
    Epoch [3/50], Val Losses: mse: 0.4409, mae: 0.4584, huber: 0.1885, swd: 0.0995, ept: 144.8695
    Epoch [3/50], Test Losses: mse: 0.7071, mae: 0.6263, huber: 0.3025, swd: 0.1599, ept: 108.2853
      Epoch 3 composite train-obj: 0.272087
            No improvement (0.1885), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.6596, mae: 0.5635, huber: 0.2644, swd: 0.2169, ept: 116.8910
    Epoch [4/50], Val Losses: mse: 0.5228, mae: 0.5034, huber: 0.2186, swd: 0.0947, ept: 145.3204
    Epoch [4/50], Test Losses: mse: 0.7362, mae: 0.6420, huber: 0.3128, swd: 0.1626, ept: 110.6829
      Epoch 4 composite train-obj: 0.264435
            No improvement (0.2186), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.6376, mae: 0.5545, huber: 0.2568, swd: 0.2158, ept: 119.0497
    Epoch [5/50], Val Losses: mse: 0.4740, mae: 0.4758, huber: 0.2002, swd: 0.0803, ept: 145.3291
    Epoch [5/50], Test Losses: mse: 0.7117, mae: 0.6219, huber: 0.3023, swd: 0.1590, ept: 111.2872
      Epoch 5 composite train-obj: 0.256831
            No improvement (0.2002), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.6216, mae: 0.5479, huber: 0.2515, swd: 0.2121, ept: 120.0984
    Epoch [6/50], Val Losses: mse: 0.5585, mae: 0.5196, huber: 0.2308, swd: 0.0803, ept: 142.6855
    Epoch [6/50], Test Losses: mse: 0.7438, mae: 0.6367, huber: 0.3133, swd: 0.1566, ept: 112.4773
      Epoch 6 composite train-obj: 0.251538
            No improvement (0.2308), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.6029, mae: 0.5395, huber: 0.2449, swd: 0.2075, ept: 120.7479
    Epoch [7/50], Val Losses: mse: 0.6942, mae: 0.5869, huber: 0.2813, swd: 0.1000, ept: 142.9942
    Epoch [7/50], Test Losses: mse: 0.8492, mae: 0.7009, huber: 0.3550, swd: 0.1651, ept: 107.0184
      Epoch 7 composite train-obj: 0.244938
    Epoch [7/50], Test Losses: mse: 0.6699, mae: 0.6025, huber: 0.2880, swd: 0.1550, ept: 111.2041
    Best round's Test MSE: 0.6699, MAE: 0.6025, SWD: 0.1550
    Best round's Validation MSE: 0.4051, MAE: 0.4351, SWD: 0.0908
    Best round's Test verification MSE : 0.6699, MAE: 0.6025, SWD: 0.1550
    Time taken: 30.37 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq196_pred720_20250510_1840)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6661 ± 0.0046
      mae: 0.5988 ± 0.0042
      huber: 0.2860 ± 0.0022
      swd: 0.1486 ± 0.0046
      ept: 112.5872 ± 1.3947
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4041 ± 0.0034
      mae: 0.4334 ± 0.0046
      huber: 0.1743 ± 0.0021
      swd: 0.0839 ± 0.0049
      ept: 143.9566 ± 2.3939
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 86.83 seconds
    
    Experiment complete: TimeMixer_etth1_seq196_pred720_20250510_1840
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### PatchTST

#### pred=96


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=196,
    pred_len=96,
    channels=data_mgr.datasets['etth1']['channels'],
    enc_in=data_mgr.datasets['etth1']['channels'],
    dec_in=data_mgr.datasets['etth1']['channels'],
    c_out=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 96
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 12
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4461, mae: 0.4614, huber: 0.1890, swd: 0.1123, ept: 42.3623
    Epoch [1/50], Val Losses: mse: 0.3541, mae: 0.3997, huber: 0.1516, swd: 0.1188, ept: 51.6733
    Epoch [1/50], Test Losses: mse: 0.4543, mae: 0.4684, huber: 0.1951, swd: 0.1389, ept: 40.1459
      Epoch 1 composite train-obj: 0.189018
            Val objective improved inf → 0.1516, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4009, mae: 0.4331, huber: 0.1713, swd: 0.1089, ept: 45.7940
    Epoch [2/50], Val Losses: mse: 0.3391, mae: 0.3888, huber: 0.1454, swd: 0.1085, ept: 52.3845
    Epoch [2/50], Test Losses: mse: 0.4395, mae: 0.4584, huber: 0.1893, swd: 0.1420, ept: 40.0331
      Epoch 2 composite train-obj: 0.171287
            Val objective improved 0.1516 → 0.1454, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3879, mae: 0.4262, huber: 0.1665, swd: 0.1050, ept: 46.2811
    Epoch [3/50], Val Losses: mse: 0.3355, mae: 0.3913, huber: 0.1459, swd: 0.1036, ept: 51.9131
    Epoch [3/50], Test Losses: mse: 0.4300, mae: 0.4549, huber: 0.1862, swd: 0.1298, ept: 40.2401
      Epoch 3 composite train-obj: 0.166460
            No improvement (0.1459), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3783, mae: 0.4218, huber: 0.1632, swd: 0.1025, ept: 46.5964
    Epoch [4/50], Val Losses: mse: 0.3326, mae: 0.3906, huber: 0.1454, swd: 0.1101, ept: 51.8364
    Epoch [4/50], Test Losses: mse: 0.4395, mae: 0.4621, huber: 0.1910, swd: 0.1321, ept: 39.9503
      Epoch 4 composite train-obj: 0.163165
            No improvement (0.1454), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3686, mae: 0.4171, huber: 0.1597, swd: 0.0989, ept: 46.9230
    Epoch [5/50], Val Losses: mse: 0.3424, mae: 0.4007, huber: 0.1506, swd: 0.1076, ept: 51.4991
    Epoch [5/50], Test Losses: mse: 0.4378, mae: 0.4621, huber: 0.1900, swd: 0.1246, ept: 40.2880
      Epoch 5 composite train-obj: 0.159724
            No improvement (0.1506), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3596, mae: 0.4132, huber: 0.1567, swd: 0.0964, ept: 47.0276
    Epoch [6/50], Val Losses: mse: 0.3336, mae: 0.3946, huber: 0.1472, swd: 0.1068, ept: 51.5059
    Epoch [6/50], Test Losses: mse: 0.4383, mae: 0.4634, huber: 0.1910, swd: 0.1288, ept: 39.7231
      Epoch 6 composite train-obj: 0.156702
            No improvement (0.1472), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3501, mae: 0.4085, huber: 0.1534, swd: 0.0940, ept: 47.3151
    Epoch [7/50], Val Losses: mse: 0.3453, mae: 0.4012, huber: 0.1517, swd: 0.1047, ept: 51.1556
    Epoch [7/50], Test Losses: mse: 0.4435, mae: 0.4654, huber: 0.1925, swd: 0.1186, ept: 40.2390
      Epoch 7 composite train-obj: 0.153379
    Epoch [7/50], Test Losses: mse: 0.4395, mae: 0.4584, huber: 0.1893, swd: 0.1420, ept: 40.0331
    Best round's Test MSE: 0.4395, MAE: 0.4584, SWD: 0.1420
    Best round's Validation MSE: 0.3391, MAE: 0.3888, SWD: 0.1085
    Best round's Test verification MSE : 0.4395, MAE: 0.4584, SWD: 0.1420
    Time taken: 16.80 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4448, mae: 0.4610, huber: 0.1887, swd: 0.1095, ept: 42.5654
    Epoch [1/50], Val Losses: mse: 0.3525, mae: 0.3974, huber: 0.1505, swd: 0.1013, ept: 51.5109
    Epoch [1/50], Test Losses: mse: 0.4440, mae: 0.4602, huber: 0.1903, swd: 0.1224, ept: 40.4473
      Epoch 1 composite train-obj: 0.188745
            Val objective improved inf → 0.1505, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3978, mae: 0.4326, huber: 0.1705, swd: 0.1045, ept: 45.8531
    Epoch [2/50], Val Losses: mse: 0.3316, mae: 0.3894, huber: 0.1443, swd: 0.0981, ept: 52.1272
    Epoch [2/50], Test Losses: mse: 0.4403, mae: 0.4605, huber: 0.1902, swd: 0.1238, ept: 40.3866
      Epoch 2 composite train-obj: 0.170451
            Val objective improved 0.1505 → 0.1443, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3834, mae: 0.4256, huber: 0.1654, swd: 0.1015, ept: 46.3137
    Epoch [3/50], Val Losses: mse: 0.3366, mae: 0.3941, huber: 0.1471, swd: 0.0932, ept: 51.7520
    Epoch [3/50], Test Losses: mse: 0.4298, mae: 0.4554, huber: 0.1866, swd: 0.1203, ept: 39.9196
      Epoch 3 composite train-obj: 0.165384
            No improvement (0.1471), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3745, mae: 0.4216, huber: 0.1623, swd: 0.0996, ept: 46.4863
    Epoch [4/50], Val Losses: mse: 0.3288, mae: 0.3918, huber: 0.1449, swd: 0.0929, ept: 51.6323
    Epoch [4/50], Test Losses: mse: 0.4311, mae: 0.4559, huber: 0.1871, swd: 0.1210, ept: 39.8865
      Epoch 4 composite train-obj: 0.162331
            No improvement (0.1449), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3635, mae: 0.4159, huber: 0.1584, swd: 0.0967, ept: 46.9414
    Epoch [5/50], Val Losses: mse: 0.3394, mae: 0.4000, huber: 0.1500, swd: 0.0916, ept: 51.2974
    Epoch [5/50], Test Losses: mse: 0.4301, mae: 0.4572, huber: 0.1873, swd: 0.1129, ept: 40.1410
      Epoch 5 composite train-obj: 0.158363
            No improvement (0.1500), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3554, mae: 0.4116, huber: 0.1554, swd: 0.0950, ept: 47.1360
    Epoch [6/50], Val Losses: mse: 0.3318, mae: 0.3971, huber: 0.1475, swd: 0.0929, ept: 50.7927
    Epoch [6/50], Test Losses: mse: 0.4301, mae: 0.4593, huber: 0.1882, swd: 0.1213, ept: 39.7325
      Epoch 6 composite train-obj: 0.155394
            No improvement (0.1475), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3476, mae: 0.4077, huber: 0.1527, swd: 0.0930, ept: 47.3662
    Epoch [7/50], Val Losses: mse: 0.3323, mae: 0.3971, huber: 0.1475, swd: 0.0923, ept: 50.9931
    Epoch [7/50], Test Losses: mse: 0.4285, mae: 0.4593, huber: 0.1879, swd: 0.1237, ept: 39.6206
      Epoch 7 composite train-obj: 0.152686
    Epoch [7/50], Test Losses: mse: 0.4403, mae: 0.4605, huber: 0.1902, swd: 0.1238, ept: 40.3866
    Best round's Test MSE: 0.4403, MAE: 0.4605, SWD: 0.1238
    Best round's Validation MSE: 0.3316, MAE: 0.3894, SWD: 0.0981
    Best round's Test verification MSE : 0.4403, MAE: 0.4605, SWD: 0.1238
    Time taken: 16.66 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4418, mae: 0.4597, huber: 0.1877, swd: 0.1039, ept: 42.5727
    Epoch [1/50], Val Losses: mse: 0.3433, mae: 0.3919, huber: 0.1469, swd: 0.1040, ept: 51.3962
    Epoch [1/50], Test Losses: mse: 0.4451, mae: 0.4618, huber: 0.1912, swd: 0.1374, ept: 39.5501
      Epoch 1 composite train-obj: 0.187726
            Val objective improved inf → 0.1469, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3982, mae: 0.4327, huber: 0.1706, swd: 0.1006, ept: 45.6905
    Epoch [2/50], Val Losses: mse: 0.3303, mae: 0.3874, huber: 0.1431, swd: 0.0986, ept: 51.3244
    Epoch [2/50], Test Losses: mse: 0.4420, mae: 0.4621, huber: 0.1907, swd: 0.1391, ept: 39.3284
      Epoch 2 composite train-obj: 0.170561
            Val objective improved 0.1469 → 0.1431, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3859, mae: 0.4263, huber: 0.1661, swd: 0.0975, ept: 46.2764
    Epoch [3/50], Val Losses: mse: 0.3290, mae: 0.3849, huber: 0.1427, swd: 0.1047, ept: 52.0028
    Epoch [3/50], Test Losses: mse: 0.4337, mae: 0.4552, huber: 0.1874, swd: 0.1389, ept: 40.2724
      Epoch 3 composite train-obj: 0.166115
            Val objective improved 0.1431 → 0.1427, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3748, mae: 0.4208, huber: 0.1623, swd: 0.0952, ept: 46.7406
    Epoch [4/50], Val Losses: mse: 0.3368, mae: 0.3962, huber: 0.1481, swd: 0.1061, ept: 51.6236
    Epoch [4/50], Test Losses: mse: 0.4356, mae: 0.4601, huber: 0.1891, swd: 0.1329, ept: 39.8264
      Epoch 4 composite train-obj: 0.162253
            No improvement (0.1481), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3660, mae: 0.4169, huber: 0.1592, swd: 0.0927, ept: 46.8820
    Epoch [5/50], Val Losses: mse: 0.3570, mae: 0.4068, huber: 0.1561, swd: 0.0987, ept: 51.1046
    Epoch [5/50], Test Losses: mse: 0.4305, mae: 0.4552, huber: 0.1867, swd: 0.1064, ept: 40.2875
      Epoch 5 composite train-obj: 0.159164
            No improvement (0.1561), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3553, mae: 0.4121, huber: 0.1556, swd: 0.0895, ept: 47.1534
    Epoch [6/50], Val Losses: mse: 0.3655, mae: 0.4127, huber: 0.1599, swd: 0.1098, ept: 50.6742
    Epoch [6/50], Test Losses: mse: 0.4540, mae: 0.4722, huber: 0.1971, swd: 0.1167, ept: 39.6989
      Epoch 6 composite train-obj: 0.155555
            No improvement (0.1599), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3453, mae: 0.4074, huber: 0.1520, swd: 0.0864, ept: 47.3567
    Epoch [7/50], Val Losses: mse: 0.3483, mae: 0.4027, huber: 0.1531, swd: 0.1034, ept: 50.8458
    Epoch [7/50], Test Losses: mse: 0.4410, mae: 0.4665, huber: 0.1925, swd: 0.1205, ept: 39.7585
      Epoch 7 composite train-obj: 0.151999
            No improvement (0.1531), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3365, mae: 0.4031, huber: 0.1489, swd: 0.0835, ept: 47.5624
    Epoch [8/50], Val Losses: mse: 0.3470, mae: 0.4025, huber: 0.1525, swd: 0.1033, ept: 50.4931
    Epoch [8/50], Test Losses: mse: 0.4427, mae: 0.4677, huber: 0.1934, swd: 0.1324, ept: 39.5199
      Epoch 8 composite train-obj: 0.148863
    Epoch [8/50], Test Losses: mse: 0.4337, mae: 0.4552, huber: 0.1874, swd: 0.1389, ept: 40.2724
    Best round's Test MSE: 0.4337, MAE: 0.4552, SWD: 0.1389
    Best round's Validation MSE: 0.3290, MAE: 0.3849, SWD: 0.1047
    Best round's Test verification MSE : 0.4337, MAE: 0.4552, SWD: 0.1389
    Time taken: 20.21 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq196_pred96_20250510_1841)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4378 ± 0.0030
      mae: 0.4580 ± 0.0022
      huber: 0.1890 ± 0.0012
      swd: 0.1349 ± 0.0079
      ept: 40.2307 ± 0.1473
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3333 ± 0.0043
      mae: 0.3877 ± 0.0020
      huber: 0.1441 ± 0.0011
      swd: 0.1038 ± 0.0043
      ept: 52.1715 ± 0.1589
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 53.72 seconds
    
    Experiment complete: PatchTST_etth1_seq196_pred96_20250510_1841
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=196,
    pred_len=196,
    channels=data_mgr.datasets['etth1']['channels'],
    enc_in=data_mgr.datasets['etth1']['channels'],
    dec_in=data_mgr.datasets['etth1']['channels'],
    c_out=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 196
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 11
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5332, mae: 0.5029, huber: 0.2194, swd: 0.1383, ept: 61.9301
    Epoch [1/50], Val Losses: mse: 0.4099, mae: 0.4328, huber: 0.1726, swd: 0.1075, ept: 70.5971
    Epoch [1/50], Test Losses: mse: 0.4890, mae: 0.4911, huber: 0.2100, swd: 0.1093, ept: 58.9610
      Epoch 1 composite train-obj: 0.219376
            Val objective improved inf → 0.1726, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4864, mae: 0.4777, huber: 0.2022, swd: 0.1340, ept: 66.8451
    Epoch [2/50], Val Losses: mse: 0.3992, mae: 0.4339, huber: 0.1718, swd: 0.1046, ept: 71.0699
    Epoch [2/50], Test Losses: mse: 0.4855, mae: 0.4937, huber: 0.2103, swd: 0.1161, ept: 59.4360
      Epoch 2 composite train-obj: 0.202154
            Val objective improved 0.1726 → 0.1718, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4642, mae: 0.4694, huber: 0.1954, swd: 0.1286, ept: 67.3974
    Epoch [3/50], Val Losses: mse: 0.4344, mae: 0.4491, huber: 0.1844, swd: 0.1132, ept: 71.0083
    Epoch [3/50], Test Losses: mse: 0.4821, mae: 0.4935, huber: 0.2100, swd: 0.1100, ept: 59.3387
      Epoch 3 composite train-obj: 0.195370
            No improvement (0.1844), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4450, mae: 0.4611, huber: 0.1891, swd: 0.1245, ept: 68.2560
    Epoch [4/50], Val Losses: mse: 0.4035, mae: 0.4381, huber: 0.1744, swd: 0.1086, ept: 70.9365
    Epoch [4/50], Test Losses: mse: 0.4844, mae: 0.4971, huber: 0.2116, swd: 0.1271, ept: 59.8902
      Epoch 4 composite train-obj: 0.189140
            No improvement (0.1744), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4340, mae: 0.4565, huber: 0.1854, swd: 0.1220, ept: 68.6544
    Epoch [5/50], Val Losses: mse: 0.3936, mae: 0.4352, huber: 0.1712, swd: 0.0988, ept: 70.2316
    Epoch [5/50], Test Losses: mse: 0.4776, mae: 0.4930, huber: 0.2089, swd: 0.1219, ept: 59.3905
      Epoch 5 composite train-obj: 0.185438
            Val objective improved 0.1718 → 0.1712, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4182, mae: 0.4491, huber: 0.1799, swd: 0.1170, ept: 69.2716
    Epoch [6/50], Val Losses: mse: 0.4057, mae: 0.4358, huber: 0.1744, swd: 0.1119, ept: 71.4240
    Epoch [6/50], Test Losses: mse: 0.4864, mae: 0.4985, huber: 0.2130, swd: 0.1236, ept: 59.6906
      Epoch 6 composite train-obj: 0.179888
            No improvement (0.1744), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4086, mae: 0.4443, huber: 0.1765, swd: 0.1128, ept: 69.7135
    Epoch [7/50], Val Losses: mse: 0.3923, mae: 0.4325, huber: 0.1707, swd: 0.1032, ept: 71.8486
    Epoch [7/50], Test Losses: mse: 0.4827, mae: 0.4962, huber: 0.2114, swd: 0.1268, ept: 58.8503
      Epoch 7 composite train-obj: 0.176454
            Val objective improved 0.1712 → 0.1707, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3993, mae: 0.4400, huber: 0.1731, swd: 0.1092, ept: 69.7781
    Epoch [8/50], Val Losses: mse: 0.4097, mae: 0.4403, huber: 0.1767, swd: 0.1045, ept: 71.3635
    Epoch [8/50], Test Losses: mse: 0.4915, mae: 0.5020, huber: 0.2152, swd: 0.1216, ept: 59.1990
      Epoch 8 composite train-obj: 0.173097
            No improvement (0.1767), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3884, mae: 0.4344, huber: 0.1690, swd: 0.1044, ept: 70.2837
    Epoch [9/50], Val Losses: mse: 0.4362, mae: 0.4462, huber: 0.1835, swd: 0.1123, ept: 71.7044
    Epoch [9/50], Test Losses: mse: 0.5046, mae: 0.5097, huber: 0.2201, swd: 0.1118, ept: 59.4003
      Epoch 9 composite train-obj: 0.169019
            No improvement (0.1835), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.3826, mae: 0.4311, huber: 0.1668, swd: 0.1016, ept: 70.5759
    Epoch [10/50], Val Losses: mse: 0.4228, mae: 0.4436, huber: 0.1799, swd: 0.1051, ept: 71.0004
    Epoch [10/50], Test Losses: mse: 0.4904, mae: 0.5011, huber: 0.2146, swd: 0.1100, ept: 59.3541
      Epoch 10 composite train-obj: 0.166795
            No improvement (0.1799), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.3740, mae: 0.4265, huber: 0.1635, swd: 0.0979, ept: 70.8973
    Epoch [11/50], Val Losses: mse: 0.4418, mae: 0.4461, huber: 0.1836, swd: 0.1059, ept: 70.9066
    Epoch [11/50], Test Losses: mse: 0.5055, mae: 0.5079, huber: 0.2202, swd: 0.1070, ept: 59.6645
      Epoch 11 composite train-obj: 0.163491
            No improvement (0.1836), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.3676, mae: 0.4230, huber: 0.1611, swd: 0.0951, ept: 71.2722
    Epoch [12/50], Val Losses: mse: 0.4286, mae: 0.4468, huber: 0.1814, swd: 0.0998, ept: 69.8016
    Epoch [12/50], Test Losses: mse: 0.5093, mae: 0.5104, huber: 0.2217, swd: 0.1260, ept: 58.9292
      Epoch 12 composite train-obj: 0.161051
    Epoch [12/50], Test Losses: mse: 0.4827, mae: 0.4962, huber: 0.2114, swd: 0.1268, ept: 58.8503
    Best round's Test MSE: 0.4827, MAE: 0.4962, SWD: 0.1268
    Best round's Validation MSE: 0.3923, MAE: 0.4325, SWD: 0.1032
    Best round's Test verification MSE : 0.4827, MAE: 0.4962, SWD: 0.1268
    Time taken: 29.29 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5342, mae: 0.5030, huber: 0.2196, swd: 0.1399, ept: 61.6221
    Epoch [1/50], Val Losses: mse: 0.4143, mae: 0.4416, huber: 0.1763, swd: 0.1329, ept: 72.3441
    Epoch [1/50], Test Losses: mse: 0.4983, mae: 0.5030, huber: 0.2161, swd: 0.1395, ept: 58.4782
      Epoch 1 composite train-obj: 0.219577
            Val objective improved inf → 0.1763, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4842, mae: 0.4775, huber: 0.2016, swd: 0.1348, ept: 66.7261
    Epoch [2/50], Val Losses: mse: 0.3818, mae: 0.4230, huber: 0.1648, swd: 0.1137, ept: 72.3354
    Epoch [2/50], Test Losses: mse: 0.4869, mae: 0.4951, huber: 0.2113, swd: 0.1422, ept: 58.6525
      Epoch 2 composite train-obj: 0.201637
            Val objective improved 0.1763 → 0.1648, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4579, mae: 0.4674, huber: 0.1935, swd: 0.1293, ept: 67.3151
    Epoch [3/50], Val Losses: mse: 0.3981, mae: 0.4333, huber: 0.1721, swd: 0.1089, ept: 70.6923
    Epoch [3/50], Test Losses: mse: 0.4906, mae: 0.5016, huber: 0.2143, swd: 0.1275, ept: 57.5344
      Epoch 3 composite train-obj: 0.193487
            No improvement (0.1721), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4414, mae: 0.4603, huber: 0.1880, swd: 0.1248, ept: 68.1074
    Epoch [4/50], Val Losses: mse: 0.3879, mae: 0.4278, huber: 0.1687, swd: 0.1055, ept: 70.9022
    Epoch [4/50], Test Losses: mse: 0.4908, mae: 0.4999, huber: 0.2141, swd: 0.1211, ept: 59.3436
      Epoch 4 composite train-obj: 0.188049
            No improvement (0.1687), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4289, mae: 0.4546, huber: 0.1838, swd: 0.1213, ept: 68.5352
    Epoch [5/50], Val Losses: mse: 0.4159, mae: 0.4412, huber: 0.1784, swd: 0.1036, ept: 71.5280
    Epoch [5/50], Test Losses: mse: 0.5052, mae: 0.5093, huber: 0.2205, swd: 0.1120, ept: 58.1569
      Epoch 5 composite train-obj: 0.183766
            No improvement (0.1784), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4146, mae: 0.4480, huber: 0.1788, swd: 0.1150, ept: 68.7181
    Epoch [6/50], Val Losses: mse: 0.4062, mae: 0.4379, huber: 0.1756, swd: 0.1115, ept: 71.4627
    Epoch [6/50], Test Losses: mse: 0.5076, mae: 0.5131, huber: 0.2222, swd: 0.1328, ept: 58.3526
      Epoch 6 composite train-obj: 0.178792
            No improvement (0.1756), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4038, mae: 0.4431, huber: 0.1751, swd: 0.1103, ept: 69.3704
    Epoch [7/50], Val Losses: mse: 0.4037, mae: 0.4357, huber: 0.1742, swd: 0.1068, ept: 70.6898
    Epoch [7/50], Test Losses: mse: 0.5133, mae: 0.5142, huber: 0.2239, swd: 0.1301, ept: 57.8087
      Epoch 7 composite train-obj: 0.175080
    Epoch [7/50], Test Losses: mse: 0.4869, mae: 0.4951, huber: 0.2113, swd: 0.1422, ept: 58.6525
    Best round's Test MSE: 0.4869, MAE: 0.4951, SWD: 0.1422
    Best round's Validation MSE: 0.3818, MAE: 0.4230, SWD: 0.1137
    Best round's Test verification MSE : 0.4869, MAE: 0.4951, SWD: 0.1422
    Time taken: 17.23 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5348, mae: 0.5037, huber: 0.2200, swd: 0.1257, ept: 61.5643
    Epoch [1/50], Val Losses: mse: 0.3991, mae: 0.4286, huber: 0.1690, swd: 0.1096, ept: 71.3638
    Epoch [1/50], Test Losses: mse: 0.4909, mae: 0.4956, huber: 0.2121, swd: 0.1222, ept: 58.7881
      Epoch 1 composite train-obj: 0.219968
            Val objective improved inf → 0.1690, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4892, mae: 0.4788, huber: 0.2030, swd: 0.1239, ept: 66.6747
    Epoch [2/50], Val Losses: mse: 0.3952, mae: 0.4297, huber: 0.1697, swd: 0.1110, ept: 72.1130
    Epoch [2/50], Test Losses: mse: 0.4814, mae: 0.4919, huber: 0.2091, swd: 0.1283, ept: 58.8955
      Epoch 2 composite train-obj: 0.203025
            No improvement (0.1697), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4681, mae: 0.4706, huber: 0.1963, swd: 0.1191, ept: 67.3002
    Epoch [3/50], Val Losses: mse: 0.4126, mae: 0.4404, huber: 0.1773, swd: 0.1045, ept: 71.4884
    Epoch [3/50], Test Losses: mse: 0.4914, mae: 0.4996, huber: 0.2140, swd: 0.1154, ept: 59.2083
      Epoch 3 composite train-obj: 0.196348
            No improvement (0.1773), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4486, mae: 0.4629, huber: 0.1901, swd: 0.1158, ept: 67.9074
    Epoch [4/50], Val Losses: mse: 0.4224, mae: 0.4511, huber: 0.1826, swd: 0.1063, ept: 70.5973
    Epoch [4/50], Test Losses: mse: 0.4950, mae: 0.5047, huber: 0.2165, swd: 0.1275, ept: 58.6314
      Epoch 4 composite train-obj: 0.190129
            No improvement (0.1826), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4329, mae: 0.4562, huber: 0.1850, swd: 0.1113, ept: 68.4079
    Epoch [5/50], Val Losses: mse: 0.4222, mae: 0.4434, huber: 0.1797, swd: 0.0982, ept: 70.6382
    Epoch [5/50], Test Losses: mse: 0.5001, mae: 0.5052, huber: 0.2175, swd: 0.0954, ept: 58.7490
      Epoch 5 composite train-obj: 0.185027
            No improvement (0.1797), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.4193, mae: 0.4503, huber: 0.1804, swd: 0.1063, ept: 68.9191
    Epoch [6/50], Val Losses: mse: 0.4190, mae: 0.4426, huber: 0.1791, swd: 0.1009, ept: 71.1154
    Epoch [6/50], Test Losses: mse: 0.4967, mae: 0.5022, huber: 0.2161, swd: 0.1100, ept: 59.6132
      Epoch 6 composite train-obj: 0.180428
    Epoch [6/50], Test Losses: mse: 0.4909, mae: 0.4956, huber: 0.2121, swd: 0.1222, ept: 58.7881
    Best round's Test MSE: 0.4909, MAE: 0.4956, SWD: 0.1222
    Best round's Validation MSE: 0.3991, MAE: 0.4286, SWD: 0.1096
    Best round's Test verification MSE : 0.4909, MAE: 0.4956, SWD: 0.1222
    Time taken: 15.12 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq196_pred196_20250510_1842)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4868 ± 0.0033
      mae: 0.4956 ± 0.0004
      huber: 0.2116 ± 0.0003
      swd: 0.1304 ± 0.0085
      ept: 58.7636 ± 0.0826
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3911 ± 0.0071
      mae: 0.4280 ± 0.0039
      huber: 0.1682 ± 0.0025
      swd: 0.1088 ± 0.0043
      ept: 71.8493 ± 0.3966
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 61.72 seconds
    
    Experiment complete: PatchTST_etth1_seq196_pred196_20250510_1842
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=196,
    pred_len=336,
    channels=data_mgr.datasets['etth1']['channels'],
    enc_in=data_mgr.datasets['etth1']['channels'],
    dec_in=data_mgr.datasets['etth1']['channels'],
    c_out=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 10
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6121, mae: 0.5402, huber: 0.2470, swd: 0.1554, ept: 76.3926
    Epoch [1/50], Val Losses: mse: 0.4377, mae: 0.4503, huber: 0.1835, swd: 0.1040, ept: 88.9396
    Epoch [1/50], Test Losses: mse: 0.5471, mae: 0.5305, huber: 0.2361, swd: 0.1302, ept: 78.3239
      Epoch 1 composite train-obj: 0.247048
            Val objective improved inf → 0.1835, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5638, mae: 0.5174, huber: 0.2301, swd: 0.1504, ept: 83.0718
    Epoch [2/50], Val Losses: mse: 0.4251, mae: 0.4453, huber: 0.1805, swd: 0.0966, ept: 89.0487
    Epoch [2/50], Test Losses: mse: 0.5336, mae: 0.5230, huber: 0.2311, swd: 0.1278, ept: 78.9227
      Epoch 2 composite train-obj: 0.230138
            Val objective improved 0.1835 → 0.1805, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5348, mae: 0.5068, huber: 0.2213, swd: 0.1440, ept: 83.7802
    Epoch [3/50], Val Losses: mse: 0.4833, mae: 0.4674, huber: 0.1982, swd: 0.1059, ept: 88.0273
    Epoch [3/50], Test Losses: mse: 0.5406, mae: 0.5301, huber: 0.2349, swd: 0.1151, ept: 78.1652
      Epoch 3 composite train-obj: 0.221305
            No improvement (0.1982), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5145, mae: 0.4968, huber: 0.2140, swd: 0.1411, ept: 85.1373
    Epoch [4/50], Val Losses: mse: 0.6383, mae: 0.5342, huber: 0.2482, swd: 0.1155, ept: 87.5802
    Epoch [4/50], Test Losses: mse: 0.5916, mae: 0.5588, huber: 0.2553, swd: 0.1141, ept: 78.8664
      Epoch 4 composite train-obj: 0.214026
            No improvement (0.2482), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4986, mae: 0.4898, huber: 0.2085, swd: 0.1362, ept: 85.7741
    Epoch [5/50], Val Losses: mse: 0.4635, mae: 0.4665, huber: 0.1950, swd: 0.1052, ept: 86.8840
    Epoch [5/50], Test Losses: mse: 0.5524, mae: 0.5359, huber: 0.2396, swd: 0.1326, ept: 79.3069
      Epoch 5 composite train-obj: 0.208463
            No improvement (0.1950), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4851, mae: 0.4837, huber: 0.2038, swd: 0.1321, ept: 86.4914
    Epoch [6/50], Val Losses: mse: 0.5691, mae: 0.5085, huber: 0.2270, swd: 0.1099, ept: 89.0628
    Epoch [6/50], Test Losses: mse: 0.5842, mae: 0.5533, huber: 0.2527, swd: 0.1191, ept: 80.1172
      Epoch 6 composite train-obj: 0.203832
            No improvement (0.2270), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4727, mae: 0.4779, huber: 0.1995, swd: 0.1274, ept: 87.0098
    Epoch [7/50], Val Losses: mse: 0.5207, mae: 0.4899, huber: 0.2127, swd: 0.1134, ept: 88.7973
    Epoch [7/50], Test Losses: mse: 0.5775, mae: 0.5518, huber: 0.2506, swd: 0.1252, ept: 79.3339
      Epoch 7 composite train-obj: 0.199504
    Epoch [7/50], Test Losses: mse: 0.5336, mae: 0.5230, huber: 0.2311, swd: 0.1278, ept: 78.9227
    Best round's Test MSE: 0.5336, MAE: 0.5230, SWD: 0.1278
    Best round's Validation MSE: 0.4251, MAE: 0.4453, SWD: 0.0966
    Best round's Test verification MSE : 0.5336, MAE: 0.5230, SWD: 0.1278
    Time taken: 17.55 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6103, mae: 0.5395, huber: 0.2464, swd: 0.1590, ept: 76.3714
    Epoch [1/50], Val Losses: mse: 0.4511, mae: 0.4613, huber: 0.1898, swd: 0.1185, ept: 89.5528
    Epoch [1/50], Test Losses: mse: 0.5531, mae: 0.5384, huber: 0.2398, swd: 0.1410, ept: 77.5236
      Epoch 1 composite train-obj: 0.246425
            Val objective improved inf → 0.1898, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5567, mae: 0.5149, huber: 0.2281, swd: 0.1547, ept: 83.2667
    Epoch [2/50], Val Losses: mse: 0.4567, mae: 0.4611, huber: 0.1911, swd: 0.1041, ept: 88.7136
    Epoch [2/50], Test Losses: mse: 0.5484, mae: 0.5315, huber: 0.2366, swd: 0.1129, ept: 78.2187
      Epoch 2 composite train-obj: 0.228055
            No improvement (0.1911), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5262, mae: 0.5034, huber: 0.2183, swd: 0.1484, ept: 84.1289
    Epoch [3/50], Val Losses: mse: 0.4200, mae: 0.4489, huber: 0.1809, swd: 0.0938, ept: 86.5496
    Epoch [3/50], Test Losses: mse: 0.5431, mae: 0.5267, huber: 0.2341, swd: 0.1117, ept: 79.0244
      Epoch 3 composite train-obj: 0.218337
            Val objective improved 0.1898 → 0.1809, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5068, mae: 0.4949, huber: 0.2118, swd: 0.1427, ept: 84.7635
    Epoch [4/50], Val Losses: mse: 0.4518, mae: 0.4600, huber: 0.1902, swd: 0.0958, ept: 87.7112
    Epoch [4/50], Test Losses: mse: 0.5452, mae: 0.5325, huber: 0.2368, swd: 0.1026, ept: 77.7989
      Epoch 4 composite train-obj: 0.211833
            No improvement (0.1902), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4931, mae: 0.4886, huber: 0.2070, swd: 0.1389, ept: 85.8617
    Epoch [5/50], Val Losses: mse: 0.4586, mae: 0.4618, huber: 0.1924, swd: 0.0999, ept: 88.2883
    Epoch [5/50], Test Losses: mse: 0.5684, mae: 0.5436, huber: 0.2456, swd: 0.1145, ept: 79.0495
      Epoch 5 composite train-obj: 0.207006
            No improvement (0.1924), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4772, mae: 0.4813, huber: 0.2015, swd: 0.1335, ept: 86.4906
    Epoch [6/50], Val Losses: mse: 0.5006, mae: 0.4842, huber: 0.2074, swd: 0.1104, ept: 88.4603
    Epoch [6/50], Test Losses: mse: 0.5833, mae: 0.5571, huber: 0.2526, swd: 0.1112, ept: 77.9984
      Epoch 6 composite train-obj: 0.201463
            No improvement (0.2074), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.4660, mae: 0.4761, huber: 0.1976, swd: 0.1297, ept: 87.0058
    Epoch [7/50], Val Losses: mse: 0.4525, mae: 0.4622, huber: 0.1916, swd: 0.1031, ept: 88.3616
    Epoch [7/50], Test Losses: mse: 0.5748, mae: 0.5474, huber: 0.2484, swd: 0.1225, ept: 79.8164
      Epoch 7 composite train-obj: 0.197565
            No improvement (0.1916), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4543, mae: 0.4707, huber: 0.1935, swd: 0.1243, ept: 87.8031
    Epoch [8/50], Val Losses: mse: 0.4820, mae: 0.4784, huber: 0.2020, swd: 0.1005, ept: 86.4642
    Epoch [8/50], Test Losses: mse: 0.5718, mae: 0.5448, huber: 0.2469, swd: 0.1107, ept: 79.2089
      Epoch 8 composite train-obj: 0.193491
    Epoch [8/50], Test Losses: mse: 0.5431, mae: 0.5267, huber: 0.2341, swd: 0.1117, ept: 79.0244
    Best round's Test MSE: 0.5431, MAE: 0.5267, SWD: 0.1117
    Best round's Validation MSE: 0.4200, MAE: 0.4489, SWD: 0.0938
    Best round's Test verification MSE : 0.5431, MAE: 0.5267, SWD: 0.1117
    Time taken: 20.45 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6089, mae: 0.5389, huber: 0.2460, swd: 0.1561, ept: 76.6973
    Epoch [1/50], Val Losses: mse: 0.4226, mae: 0.4411, huber: 0.1778, swd: 0.1085, ept: 89.4019
    Epoch [1/50], Test Losses: mse: 0.5435, mae: 0.5284, huber: 0.2347, swd: 0.1423, ept: 77.0504
      Epoch 1 composite train-obj: 0.246046
            Val objective improved inf → 0.1778, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5629, mae: 0.5164, huber: 0.2298, swd: 0.1536, ept: 83.2909
    Epoch [2/50], Val Losses: mse: 0.4646, mae: 0.4680, huber: 0.1947, swd: 0.1022, ept: 87.1024
    Epoch [2/50], Test Losses: mse: 0.5389, mae: 0.5272, huber: 0.2334, swd: 0.1108, ept: 80.2925
      Epoch 2 composite train-obj: 0.229750
            No improvement (0.1947), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5369, mae: 0.5069, huber: 0.2216, swd: 0.1471, ept: 84.1947
    Epoch [3/50], Val Losses: mse: 0.4828, mae: 0.4645, huber: 0.1965, swd: 0.1006, ept: 87.8277
    Epoch [3/50], Test Losses: mse: 0.5366, mae: 0.5262, huber: 0.2322, swd: 0.0978, ept: 79.2695
      Epoch 3 composite train-obj: 0.221606
            No improvement (0.1965), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.5141, mae: 0.4969, huber: 0.2138, swd: 0.1424, ept: 84.7907
    Epoch [4/50], Val Losses: mse: 0.5541, mae: 0.4996, huber: 0.2221, swd: 0.1107, ept: 86.7327
    Epoch [4/50], Test Losses: mse: 0.5746, mae: 0.5474, huber: 0.2476, swd: 0.1056, ept: 80.1041
      Epoch 4 composite train-obj: 0.213842
            No improvement (0.2221), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4978, mae: 0.4897, huber: 0.2082, swd: 0.1382, ept: 86.0837
    Epoch [5/50], Val Losses: mse: 0.4584, mae: 0.4626, huber: 0.1916, swd: 0.0965, ept: 84.4373
    Epoch [5/50], Test Losses: mse: 0.5386, mae: 0.5261, huber: 0.2338, swd: 0.1122, ept: 80.1474
      Epoch 5 composite train-obj: 0.208215
            No improvement (0.1916), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.4849, mae: 0.4831, huber: 0.2035, swd: 0.1354, ept: 86.8936
    Epoch [6/50], Val Losses: mse: 0.5295, mae: 0.4923, huber: 0.2139, swd: 0.1135, ept: 84.2311
    Epoch [6/50], Test Losses: mse: 0.5447, mae: 0.5319, huber: 0.2366, swd: 0.1082, ept: 78.1262
      Epoch 6 composite train-obj: 0.203535
    Epoch [6/50], Test Losses: mse: 0.5435, mae: 0.5284, huber: 0.2347, swd: 0.1423, ept: 77.0504
    Best round's Test MSE: 0.5435, MAE: 0.5284, SWD: 0.1423
    Best round's Validation MSE: 0.4226, MAE: 0.4411, SWD: 0.1085
    Best round's Test verification MSE : 0.5435, MAE: 0.5284, SWD: 0.1423
    Time taken: 15.33 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq196_pred336_20250510_1843)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5401 ± 0.0046
      mae: 0.5260 ± 0.0023
      huber: 0.2333 ± 0.0016
      swd: 0.1273 ± 0.0125
      ept: 78.3325 ± 0.9075
      count: 10.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4226 ± 0.0021
      mae: 0.4451 ± 0.0032
      huber: 0.1797 ± 0.0014
      swd: 0.0996 ± 0.0064
      ept: 88.3334 ± 1.2695
      count: 10.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 53.42 seconds
    
    Experiment complete: PatchTST_etth1_seq196_pred336_20250510_1843
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=196,
    pred_len=720,
    channels=data_mgr.datasets['etth1']['channels'],
    enc_in=data_mgr.datasets['etth1']['channels'],
    dec_in=data_mgr.datasets['etth1']['channels'],
    c_out=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 7
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7554, mae: 0.6112, huber: 0.2989, swd: 0.2008, ept: 89.4699
    Epoch [1/50], Val Losses: mse: 0.5031, mae: 0.4894, huber: 0.2109, swd: 0.0670, ept: 134.8567
    Epoch [1/50], Test Losses: mse: 0.7008, mae: 0.6208, huber: 0.2994, swd: 0.1210, ept: 111.4086
      Epoch 1 composite train-obj: 0.298941
            Val objective improved inf → 0.2109, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6843, mae: 0.5819, huber: 0.2753, swd: 0.1950, ept: 98.5307
    Epoch [2/50], Val Losses: mse: 0.4046, mae: 0.4465, huber: 0.1777, swd: 0.0618, ept: 138.0104
    Epoch [2/50], Test Losses: mse: 0.6702, mae: 0.6061, huber: 0.2885, swd: 0.1553, ept: 114.0524
      Epoch 2 composite train-obj: 0.275335
            Val objective improved 0.2109 → 0.1777, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6489, mae: 0.5671, huber: 0.2634, swd: 0.1875, ept: 101.0428
    Epoch [3/50], Val Losses: mse: 0.5064, mae: 0.4883, huber: 0.2106, swd: 0.0670, ept: 136.6189
    Epoch [3/50], Test Losses: mse: 0.6996, mae: 0.6238, huber: 0.3002, swd: 0.1346, ept: 109.0908
      Epoch 3 composite train-obj: 0.263393
            No improvement (0.2106), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.6229, mae: 0.5551, huber: 0.2543, swd: 0.1821, ept: 102.5294
    Epoch [4/50], Val Losses: mse: 0.5441, mae: 0.5018, huber: 0.2212, swd: 0.0656, ept: 135.4181
    Epoch [4/50], Test Losses: mse: 0.7023, mae: 0.6228, huber: 0.3006, swd: 0.1274, ept: 110.9074
      Epoch 4 composite train-obj: 0.254273
            No improvement (0.2212), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.6048, mae: 0.5469, huber: 0.2480, swd: 0.1750, ept: 103.5048
    Epoch [5/50], Val Losses: mse: 0.5900, mae: 0.5158, huber: 0.2337, swd: 0.0788, ept: 136.6187
    Epoch [5/50], Test Losses: mse: 0.7520, mae: 0.6469, huber: 0.3184, swd: 0.1268, ept: 110.2281
      Epoch 5 composite train-obj: 0.247973
            No improvement (0.2337), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.5858, mae: 0.5374, huber: 0.2411, swd: 0.1685, ept: 104.9835
    Epoch [6/50], Val Losses: mse: 0.5353, mae: 0.4983, huber: 0.2174, swd: 0.0623, ept: 130.6196
    Epoch [6/50], Test Losses: mse: 0.7127, mae: 0.6301, huber: 0.3053, swd: 0.1217, ept: 111.2405
      Epoch 6 composite train-obj: 0.241099
            No improvement (0.2174), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.5712, mae: 0.5311, huber: 0.2363, swd: 0.1630, ept: 105.8713
    Epoch [7/50], Val Losses: mse: 0.5334, mae: 0.4977, huber: 0.2171, swd: 0.0673, ept: 132.7479
    Epoch [7/50], Test Losses: mse: 0.7160, mae: 0.6323, huber: 0.3067, swd: 0.1165, ept: 108.9092
      Epoch 7 composite train-obj: 0.236334
    Epoch [7/50], Test Losses: mse: 0.6702, mae: 0.6061, huber: 0.2885, swd: 0.1553, ept: 114.0524
    Best round's Test MSE: 0.6702, MAE: 0.6061, SWD: 0.1553
    Best round's Validation MSE: 0.4046, MAE: 0.4465, SWD: 0.0618
    Best round's Test verification MSE : 0.6702, MAE: 0.6061, SWD: 0.1553
    Time taken: 18.07 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7530, mae: 0.6103, huber: 0.2982, swd: 0.1934, ept: 90.1363
    Epoch [1/50], Val Losses: mse: 0.5241, mae: 0.5106, huber: 0.2240, swd: 0.0828, ept: 143.3858
    Epoch [1/50], Test Losses: mse: 0.7442, mae: 0.6429, huber: 0.3158, swd: 0.1471, ept: 108.9995
      Epoch 1 composite train-obj: 0.298191
            Val objective improved inf → 0.2240, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6842, mae: 0.5820, huber: 0.2754, swd: 0.1884, ept: 98.8888
    Epoch [2/50], Val Losses: mse: 0.5351, mae: 0.5086, huber: 0.2234, swd: 0.0763, ept: 137.9178
    Epoch [2/50], Test Losses: mse: 0.7105, mae: 0.6269, huber: 0.3034, swd: 0.1503, ept: 113.9381
      Epoch 2 composite train-obj: 0.275370
            Val objective improved 0.2240 → 0.2234, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6462, mae: 0.5662, huber: 0.2626, swd: 0.1813, ept: 101.3722
    Epoch [3/50], Val Losses: mse: 0.4760, mae: 0.4843, huber: 0.2048, swd: 0.0662, ept: 138.0822
    Epoch [3/50], Test Losses: mse: 0.7138, mae: 0.6261, huber: 0.3044, swd: 0.1446, ept: 110.6114
      Epoch 3 composite train-obj: 0.262571
            Val objective improved 0.2234 → 0.2048, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.6218, mae: 0.5553, huber: 0.2541, swd: 0.1764, ept: 103.4017
    Epoch [4/50], Val Losses: mse: 0.5107, mae: 0.4956, huber: 0.2144, swd: 0.0638, ept: 136.4712
    Epoch [4/50], Test Losses: mse: 0.7003, mae: 0.6181, huber: 0.2988, swd: 0.1239, ept: 110.9666
      Epoch 4 composite train-obj: 0.254117
            No improvement (0.2144), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.6037, mae: 0.5465, huber: 0.2476, swd: 0.1706, ept: 103.6488
    Epoch [5/50], Val Losses: mse: 0.5159, mae: 0.4984, huber: 0.2166, swd: 0.0695, ept: 136.4008
    Epoch [5/50], Test Losses: mse: 0.7321, mae: 0.6358, huber: 0.3118, swd: 0.1333, ept: 110.3097
      Epoch 5 composite train-obj: 0.247641
            No improvement (0.2166), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5889, mae: 0.5399, huber: 0.2426, swd: 0.1667, ept: 105.2321
    Epoch [6/50], Val Losses: mse: 0.5168, mae: 0.5020, huber: 0.2179, swd: 0.0735, ept: 136.1205
    Epoch [6/50], Test Losses: mse: 0.7334, mae: 0.6368, huber: 0.3119, swd: 0.1375, ept: 110.0178
      Epoch 6 composite train-obj: 0.242609
            No improvement (0.2179), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5750, mae: 0.5334, huber: 0.2378, swd: 0.1610, ept: 105.4572
    Epoch [7/50], Val Losses: mse: 0.5784, mae: 0.5247, huber: 0.2361, swd: 0.0658, ept: 129.2654
    Epoch [7/50], Test Losses: mse: 0.7529, mae: 0.6416, huber: 0.3172, swd: 0.1092, ept: 112.5804
      Epoch 7 composite train-obj: 0.237799
            No improvement (0.2361), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.5627, mae: 0.5280, huber: 0.2336, swd: 0.1553, ept: 106.6257
    Epoch [8/50], Val Losses: mse: 0.4838, mae: 0.4854, huber: 0.2058, swd: 0.0681, ept: 137.2742
    Epoch [8/50], Test Losses: mse: 0.7324, mae: 0.6394, huber: 0.3126, swd: 0.1208, ept: 107.4059
      Epoch 8 composite train-obj: 0.233617
    Epoch [8/50], Test Losses: mse: 0.7138, mae: 0.6261, huber: 0.3044, swd: 0.1446, ept: 110.6114
    Best round's Test MSE: 0.7138, MAE: 0.6261, SWD: 0.1446
    Best round's Validation MSE: 0.4760, MAE: 0.4843, SWD: 0.0662
    Best round's Test verification MSE : 0.7138, MAE: 0.6261, SWD: 0.1446
    Time taken: 19.90 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7573, mae: 0.6120, huber: 0.2996, swd: 0.2111, ept: 89.4007
    Epoch [1/50], Val Losses: mse: 0.4263, mae: 0.4561, huber: 0.1852, swd: 0.0686, ept: 137.3997
    Epoch [1/50], Test Losses: mse: 0.6839, mae: 0.6108, huber: 0.2932, swd: 0.1275, ept: 111.1914
      Epoch 1 composite train-obj: 0.299613
            Val objective improved inf → 0.1852, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6921, mae: 0.5845, huber: 0.2776, swd: 0.2060, ept: 99.1255
    Epoch [2/50], Val Losses: mse: 0.4565, mae: 0.4696, huber: 0.1950, swd: 0.0665, ept: 137.6484
    Epoch [2/50], Test Losses: mse: 0.6784, mae: 0.6097, huber: 0.2913, swd: 0.1397, ept: 112.6917
      Epoch 2 composite train-obj: 0.277647
            No improvement (0.1950), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.6505, mae: 0.5674, huber: 0.2636, swd: 0.1979, ept: 100.2485
    Epoch [3/50], Val Losses: mse: 0.4670, mae: 0.4767, huber: 0.1992, swd: 0.0714, ept: 137.8136
    Epoch [3/50], Test Losses: mse: 0.6882, mae: 0.6146, huber: 0.2949, swd: 0.1396, ept: 114.6347
      Epoch 3 composite train-obj: 0.263597
            No improvement (0.1992), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.6224, mae: 0.5545, huber: 0.2537, swd: 0.1893, ept: 101.7330
    Epoch [4/50], Val Losses: mse: 0.4970, mae: 0.4879, huber: 0.2087, swd: 0.0712, ept: 134.8647
    Epoch [4/50], Test Losses: mse: 0.6994, mae: 0.6214, huber: 0.2994, swd: 0.1286, ept: 112.5526
      Epoch 4 composite train-obj: 0.253737
            No improvement (0.2087), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.6030, mae: 0.5449, huber: 0.2467, swd: 0.1839, ept: 103.3458
    Epoch [5/50], Val Losses: mse: 0.5381, mae: 0.5114, huber: 0.2242, swd: 0.0762, ept: 134.9533
    Epoch [5/50], Test Losses: mse: 0.7307, mae: 0.6345, huber: 0.3096, swd: 0.1428, ept: 110.9165
      Epoch 5 composite train-obj: 0.246703
            No improvement (0.2242), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.5890, mae: 0.5380, huber: 0.2416, swd: 0.1785, ept: 104.3797
    Epoch [6/50], Val Losses: mse: 0.5522, mae: 0.5165, huber: 0.2289, swd: 0.0799, ept: 136.6612
    Epoch [6/50], Test Losses: mse: 0.7466, mae: 0.6441, huber: 0.3160, swd: 0.1266, ept: 112.4459
      Epoch 6 composite train-obj: 0.241633
    Epoch [6/50], Test Losses: mse: 0.6839, mae: 0.6108, huber: 0.2932, swd: 0.1275, ept: 111.1914
    Best round's Test MSE: 0.6839, MAE: 0.6108, SWD: 0.1275
    Best round's Validation MSE: 0.4263, MAE: 0.4561, SWD: 0.0686
    Best round's Test verification MSE : 0.6839, MAE: 0.6108, SWD: 0.1275
    Time taken: 14.94 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq196_pred720_20250510_1844)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6893 ± 0.0182
      mae: 0.6143 ± 0.0085
      huber: 0.2953 ± 0.0067
      swd: 0.1424 ± 0.0115
      ept: 111.9517 ± 1.5041
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4357 ± 0.0299
      mae: 0.4623 ± 0.0161
      huber: 0.1892 ± 0.0114
      swd: 0.0655 ± 0.0028
      ept: 137.8308 ± 0.3062
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 53.00 seconds
    
    Experiment complete: PatchTST_etth1_seq196_pred720_20250510_1844
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### DLinear

#### pred=96


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=196,
    pred_len=96,
    channels=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 96
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 12
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4674, mae: 0.4772, huber: 0.1979, swd: 0.1371, ept: 39.8746
    Epoch [1/50], Val Losses: mse: 0.3721, mae: 0.4133, huber: 0.1598, swd: 0.1171, ept: 46.3547
    Epoch [1/50], Test Losses: mse: 0.4503, mae: 0.4667, huber: 0.1937, swd: 0.1330, ept: 38.1852
      Epoch 1 composite train-obj: 0.197904
            Val objective improved inf → 0.1598, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3650, mae: 0.4159, huber: 0.1583, swd: 0.1190, ept: 50.3327
    Epoch [2/50], Val Losses: mse: 0.3653, mae: 0.4049, huber: 0.1562, swd: 0.1195, ept: 48.8312
    Epoch [2/50], Test Losses: mse: 0.4325, mae: 0.4499, huber: 0.1854, swd: 0.1301, ept: 40.6005
      Epoch 2 composite train-obj: 0.158282
            Val objective improved 0.1598 → 0.1562, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3561, mae: 0.4082, huber: 0.1543, swd: 0.1166, ept: 52.0192
    Epoch [3/50], Val Losses: mse: 0.3582, mae: 0.3965, huber: 0.1525, swd: 0.1109, ept: 50.0757
    Epoch [3/50], Test Losses: mse: 0.4282, mae: 0.4454, huber: 0.1830, swd: 0.1251, ept: 41.6026
      Epoch 3 composite train-obj: 0.154289
            Val objective improved 0.1562 → 0.1525, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3523, mae: 0.4051, huber: 0.1527, swd: 0.1155, ept: 52.6876
    Epoch [4/50], Val Losses: mse: 0.3508, mae: 0.3912, huber: 0.1498, swd: 0.1093, ept: 50.6261
    Epoch [4/50], Test Losses: mse: 0.4234, mae: 0.4419, huber: 0.1811, swd: 0.1288, ept: 41.5952
      Epoch 4 composite train-obj: 0.152691
            Val objective improved 0.1525 → 0.1498, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3505, mae: 0.4031, huber: 0.1518, swd: 0.1151, ept: 53.0963
    Epoch [5/50], Val Losses: mse: 0.3493, mae: 0.3890, huber: 0.1489, swd: 0.1073, ept: 51.0312
    Epoch [5/50], Test Losses: mse: 0.4208, mae: 0.4398, huber: 0.1800, swd: 0.1273, ept: 42.1473
      Epoch 5 composite train-obj: 0.151778
            Val objective improved 0.1498 → 0.1489, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3495, mae: 0.4022, huber: 0.1514, swd: 0.1148, ept: 53.2968
    Epoch [6/50], Val Losses: mse: 0.3470, mae: 0.3874, huber: 0.1481, swd: 0.1077, ept: 51.3410
    Epoch [6/50], Test Losses: mse: 0.4221, mae: 0.4400, huber: 0.1803, swd: 0.1283, ept: 42.3237
      Epoch 6 composite train-obj: 0.151382
            Val objective improved 0.1489 → 0.1481, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3489, mae: 0.4018, huber: 0.1511, swd: 0.1148, ept: 53.4480
    Epoch [7/50], Val Losses: mse: 0.3467, mae: 0.3868, huber: 0.1477, swd: 0.1022, ept: 51.3185
    Epoch [7/50], Test Losses: mse: 0.4220, mae: 0.4389, huber: 0.1800, swd: 0.1229, ept: 42.4528
      Epoch 7 composite train-obj: 0.151117
            Val objective improved 0.1481 → 0.1477, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3484, mae: 0.4011, huber: 0.1509, swd: 0.1145, ept: 53.5562
    Epoch [8/50], Val Losses: mse: 0.3479, mae: 0.3871, huber: 0.1481, swd: 0.1043, ept: 51.3807
    Epoch [8/50], Test Losses: mse: 0.4216, mae: 0.4386, huber: 0.1797, swd: 0.1256, ept: 42.6218
      Epoch 8 composite train-obj: 0.150860
            No improvement (0.1481), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3477, mae: 0.4006, huber: 0.1506, swd: 0.1144, ept: 53.6534
    Epoch [9/50], Val Losses: mse: 0.3477, mae: 0.3870, huber: 0.1481, swd: 0.1096, ept: 51.5496
    Epoch [9/50], Test Losses: mse: 0.4198, mae: 0.4382, huber: 0.1793, swd: 0.1281, ept: 42.4491
      Epoch 9 composite train-obj: 0.150551
            No improvement (0.1481), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.3478, mae: 0.4005, huber: 0.1506, swd: 0.1150, ept: 53.6607
    Epoch [10/50], Val Losses: mse: 0.3476, mae: 0.3873, huber: 0.1483, swd: 0.1095, ept: 51.5567
    Epoch [10/50], Test Losses: mse: 0.4185, mae: 0.4368, huber: 0.1787, swd: 0.1252, ept: 42.7063
      Epoch 10 composite train-obj: 0.150591
            No improvement (0.1483), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.3475, mae: 0.4000, huber: 0.1504, swd: 0.1147, ept: 53.8398
    Epoch [11/50], Val Losses: mse: 0.3494, mae: 0.3866, huber: 0.1484, swd: 0.1068, ept: 51.6819
    Epoch [11/50], Test Losses: mse: 0.4215, mae: 0.4381, huber: 0.1795, swd: 0.1242, ept: 42.8680
      Epoch 11 composite train-obj: 0.150354
            No improvement (0.1484), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.3468, mae: 0.3995, huber: 0.1501, swd: 0.1143, ept: 53.8508
    Epoch [12/50], Val Losses: mse: 0.3468, mae: 0.3862, huber: 0.1478, swd: 0.1108, ept: 51.5854
    Epoch [12/50], Test Losses: mse: 0.4178, mae: 0.4364, huber: 0.1785, swd: 0.1302, ept: 42.7486
      Epoch 12 composite train-obj: 0.150133
    Epoch [12/50], Test Losses: mse: 0.4220, mae: 0.4389, huber: 0.1800, swd: 0.1229, ept: 42.4528
    Best round's Test MSE: 0.4220, MAE: 0.4389, SWD: 0.1229
    Best round's Validation MSE: 0.3467, MAE: 0.3868, SWD: 0.1022
    Best round's Test verification MSE : 0.4220, MAE: 0.4389, SWD: 0.1229
    Time taken: 11.16 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4633, mae: 0.4752, huber: 0.1964, swd: 0.1325, ept: 39.9927
    Epoch [1/50], Val Losses: mse: 0.3728, mae: 0.4154, huber: 0.1603, swd: 0.1113, ept: 46.4263
    Epoch [1/50], Test Losses: mse: 0.4499, mae: 0.4661, huber: 0.1934, swd: 0.1258, ept: 38.0328
      Epoch 1 composite train-obj: 0.196383
            Val objective improved inf → 0.1603, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3650, mae: 0.4161, huber: 0.1583, swd: 0.1167, ept: 50.2586
    Epoch [2/50], Val Losses: mse: 0.3559, mae: 0.3971, huber: 0.1521, swd: 0.1016, ept: 49.4416
    Epoch [2/50], Test Losses: mse: 0.4340, mae: 0.4511, huber: 0.1858, swd: 0.1195, ept: 40.8672
      Epoch 2 composite train-obj: 0.158292
            Val objective improved 0.1603 → 0.1521, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3558, mae: 0.4081, huber: 0.1542, swd: 0.1145, ept: 52.0101
    Epoch [3/50], Val Losses: mse: 0.3538, mae: 0.3923, huber: 0.1504, swd: 0.1011, ept: 50.3358
    Epoch [3/50], Test Losses: mse: 0.4281, mae: 0.4459, huber: 0.1830, swd: 0.1218, ept: 41.7391
      Epoch 3 composite train-obj: 0.154183
            Val objective improved 0.1521 → 0.1504, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3522, mae: 0.4047, huber: 0.1526, swd: 0.1130, ept: 52.6781
    Epoch [4/50], Val Losses: mse: 0.3511, mae: 0.3915, huber: 0.1497, swd: 0.1059, ept: 50.7759
    Epoch [4/50], Test Losses: mse: 0.4263, mae: 0.4444, huber: 0.1823, swd: 0.1254, ept: 42.0072
      Epoch 4 composite train-obj: 0.152563
            Val objective improved 0.1504 → 0.1497, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3510, mae: 0.4033, huber: 0.1519, swd: 0.1131, ept: 53.0736
    Epoch [5/50], Val Losses: mse: 0.3519, mae: 0.3904, huber: 0.1497, swd: 0.1066, ept: 51.1204
    Epoch [5/50], Test Losses: mse: 0.4230, mae: 0.4406, huber: 0.1806, swd: 0.1234, ept: 42.2760
      Epoch 5 composite train-obj: 0.151930
            No improvement (0.1497), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3491, mae: 0.4018, huber: 0.1511, swd: 0.1125, ept: 53.3092
    Epoch [6/50], Val Losses: mse: 0.3487, mae: 0.3887, huber: 0.1486, swd: 0.1032, ept: 51.2236
    Epoch [6/50], Test Losses: mse: 0.4223, mae: 0.4404, huber: 0.1803, swd: 0.1228, ept: 42.4277
      Epoch 6 composite train-obj: 0.151145
            Val objective improved 0.1497 → 0.1486, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3484, mae: 0.4013, huber: 0.1509, swd: 0.1128, ept: 53.4482
    Epoch [7/50], Val Losses: mse: 0.3466, mae: 0.3860, huber: 0.1476, swd: 0.0996, ept: 51.4231
    Epoch [7/50], Test Losses: mse: 0.4218, mae: 0.4390, huber: 0.1798, swd: 0.1202, ept: 42.6062
      Epoch 7 composite train-obj: 0.150879
            Val objective improved 0.1486 → 0.1476, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3484, mae: 0.4008, huber: 0.1508, swd: 0.1122, ept: 53.6493
    Epoch [8/50], Val Losses: mse: 0.3461, mae: 0.3875, huber: 0.1479, swd: 0.1053, ept: 51.4725
    Epoch [8/50], Test Losses: mse: 0.4189, mae: 0.4377, huber: 0.1791, swd: 0.1235, ept: 42.5612
      Epoch 8 composite train-obj: 0.150764
            No improvement (0.1479), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3476, mae: 0.4005, huber: 0.1505, swd: 0.1126, ept: 53.6737
    Epoch [9/50], Val Losses: mse: 0.3462, mae: 0.3849, huber: 0.1471, swd: 0.1011, ept: 51.6891
    Epoch [9/50], Test Losses: mse: 0.4217, mae: 0.4384, huber: 0.1797, swd: 0.1213, ept: 42.6857
      Epoch 9 composite train-obj: 0.150526
            Val objective improved 0.1476 → 0.1471, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.3476, mae: 0.4003, huber: 0.1505, swd: 0.1123, ept: 53.7397
    Epoch [10/50], Val Losses: mse: 0.3486, mae: 0.3874, huber: 0.1483, swd: 0.1064, ept: 51.5996
    Epoch [10/50], Test Losses: mse: 0.4223, mae: 0.4399, huber: 0.1803, swd: 0.1273, ept: 42.6071
      Epoch 10 composite train-obj: 0.150480
            No improvement (0.1483), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.3476, mae: 0.4001, huber: 0.1505, swd: 0.1125, ept: 53.8181
    Epoch [11/50], Val Losses: mse: 0.3491, mae: 0.3878, huber: 0.1487, swd: 0.1031, ept: 51.5330
    Epoch [11/50], Test Losses: mse: 0.4205, mae: 0.4374, huber: 0.1792, swd: 0.1207, ept: 42.7398
      Epoch 11 composite train-obj: 0.150472
            No improvement (0.1487), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.3473, mae: 0.3999, huber: 0.1503, swd: 0.1124, ept: 53.8737
    Epoch [12/50], Val Losses: mse: 0.3505, mae: 0.3888, huber: 0.1492, swd: 0.1081, ept: 51.3680
    Epoch [12/50], Test Losses: mse: 0.4216, mae: 0.4390, huber: 0.1801, swd: 0.1279, ept: 42.7144
      Epoch 12 composite train-obj: 0.150323
            No improvement (0.1492), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.3474, mae: 0.4000, huber: 0.1503, swd: 0.1126, ept: 53.8507
    Epoch [13/50], Val Losses: mse: 0.3442, mae: 0.3840, huber: 0.1466, swd: 0.1045, ept: 51.9617
    Epoch [13/50], Test Losses: mse: 0.4198, mae: 0.4377, huber: 0.1792, swd: 0.1262, ept: 42.7005
      Epoch 13 composite train-obj: 0.150339
            Val objective improved 0.1471 → 0.1466, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.3465, mae: 0.3994, huber: 0.1500, swd: 0.1122, ept: 53.9093
    Epoch [14/50], Val Losses: mse: 0.3505, mae: 0.3905, huber: 0.1495, swd: 0.1104, ept: 51.4996
    Epoch [14/50], Test Losses: mse: 0.4203, mae: 0.4374, huber: 0.1795, swd: 0.1280, ept: 42.4165
      Epoch 14 composite train-obj: 0.150024
            No improvement (0.1495), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.3467, mae: 0.3993, huber: 0.1500, swd: 0.1125, ept: 53.9385
    Epoch [15/50], Val Losses: mse: 0.3473, mae: 0.3858, huber: 0.1478, swd: 0.1039, ept: 51.9142
    Epoch [15/50], Test Losses: mse: 0.4189, mae: 0.4365, huber: 0.1787, swd: 0.1228, ept: 42.4873
      Epoch 15 composite train-obj: 0.150040
            No improvement (0.1478), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.3468, mae: 0.3995, huber: 0.1501, swd: 0.1124, ept: 53.9208
    Epoch [16/50], Val Losses: mse: 0.3470, mae: 0.3875, huber: 0.1482, swd: 0.1079, ept: 51.8860
    Epoch [16/50], Test Losses: mse: 0.4171, mae: 0.4356, huber: 0.1784, swd: 0.1264, ept: 42.5728
      Epoch 16 composite train-obj: 0.150093
            No improvement (0.1482), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.3469, mae: 0.3996, huber: 0.1502, swd: 0.1129, ept: 53.8506
    Epoch [17/50], Val Losses: mse: 0.3458, mae: 0.3835, huber: 0.1469, swd: 0.1005, ept: 52.1760
    Epoch [17/50], Test Losses: mse: 0.4199, mae: 0.4363, huber: 0.1788, swd: 0.1206, ept: 42.6755
      Epoch 17 composite train-obj: 0.150165
            No improvement (0.1469), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.3466, mae: 0.3991, huber: 0.1500, swd: 0.1125, ept: 53.9474
    Epoch [18/50], Val Losses: mse: 0.3470, mae: 0.3852, huber: 0.1476, swd: 0.1056, ept: 51.9253
    Epoch [18/50], Test Losses: mse: 0.4191, mae: 0.4369, huber: 0.1790, swd: 0.1239, ept: 42.7329
      Epoch 18 composite train-obj: 0.149975
    Epoch [18/50], Test Losses: mse: 0.4198, mae: 0.4377, huber: 0.1792, swd: 0.1262, ept: 42.7005
    Best round's Test MSE: 0.4198, MAE: 0.4377, SWD: 0.1262
    Best round's Validation MSE: 0.3442, MAE: 0.3840, SWD: 0.1045
    Best round's Test verification MSE : 0.4198, MAE: 0.4377, SWD: 0.1262
    Time taken: 17.58 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4672, mae: 0.4775, huber: 0.1980, swd: 0.1261, ept: 39.5966
    Epoch [1/50], Val Losses: mse: 0.3723, mae: 0.4129, huber: 0.1596, swd: 0.1103, ept: 46.5312
    Epoch [1/50], Test Losses: mse: 0.4500, mae: 0.4649, huber: 0.1931, swd: 0.1220, ept: 38.1595
      Epoch 1 composite train-obj: 0.197966
            Val objective improved inf → 0.1596, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3648, mae: 0.4159, huber: 0.1582, swd: 0.1117, ept: 50.3243
    Epoch [2/50], Val Losses: mse: 0.3589, mae: 0.3988, huber: 0.1532, swd: 0.1050, ept: 49.0664
    Epoch [2/50], Test Losses: mse: 0.4336, mae: 0.4504, huber: 0.1855, swd: 0.1198, ept: 40.5091
      Epoch 2 composite train-obj: 0.158231
            Val objective improved 0.1596 → 0.1532, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3560, mae: 0.4081, huber: 0.1543, swd: 0.1091, ept: 51.9490
    Epoch [3/50], Val Losses: mse: 0.3583, mae: 0.3965, huber: 0.1527, swd: 0.1089, ept: 50.1686
    Epoch [3/50], Test Losses: mse: 0.4287, mae: 0.4463, huber: 0.1832, swd: 0.1229, ept: 41.2955
      Epoch 3 composite train-obj: 0.154258
            Val objective improved 0.1532 → 0.1527, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3522, mae: 0.4050, huber: 0.1526, swd: 0.1089, ept: 52.6526
    Epoch [4/50], Val Losses: mse: 0.3506, mae: 0.3909, huber: 0.1497, swd: 0.1016, ept: 50.5037
    Epoch [4/50], Test Losses: mse: 0.4255, mae: 0.4430, huber: 0.1817, swd: 0.1189, ept: 41.8334
      Epoch 4 composite train-obj: 0.152603
            Val objective improved 0.1527 → 0.1497, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3508, mae: 0.4033, huber: 0.1519, swd: 0.1078, ept: 53.0123
    Epoch [5/50], Val Losses: mse: 0.3497, mae: 0.3886, huber: 0.1485, swd: 0.1027, ept: 51.2410
    Epoch [5/50], Test Losses: mse: 0.4250, mae: 0.4427, huber: 0.1813, swd: 0.1223, ept: 42.3312
      Epoch 5 composite train-obj: 0.151888
            Val objective improved 0.1497 → 0.1485, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3493, mae: 0.4022, huber: 0.1513, swd: 0.1077, ept: 53.2680
    Epoch [6/50], Val Losses: mse: 0.3498, mae: 0.3894, huber: 0.1491, swd: 0.1058, ept: 51.1955
    Epoch [6/50], Test Losses: mse: 0.4214, mae: 0.4394, huber: 0.1800, swd: 0.1221, ept: 42.4355
      Epoch 6 composite train-obj: 0.151303
            No improvement (0.1491), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3485, mae: 0.4013, huber: 0.1509, swd: 0.1075, ept: 53.5517
    Epoch [7/50], Val Losses: mse: 0.3487, mae: 0.3892, huber: 0.1486, swd: 0.1009, ept: 51.2842
    Epoch [7/50], Test Losses: mse: 0.4240, mae: 0.4413, huber: 0.1808, swd: 0.1181, ept: 42.5650
      Epoch 7 composite train-obj: 0.150921
            No improvement (0.1486), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3488, mae: 0.4011, huber: 0.1509, swd: 0.1077, ept: 53.6122
    Epoch [8/50], Val Losses: mse: 0.3470, mae: 0.3847, huber: 0.1472, swd: 0.1013, ept: 51.6998
    Epoch [8/50], Test Losses: mse: 0.4233, mae: 0.4400, huber: 0.1805, swd: 0.1227, ept: 42.4837
      Epoch 8 composite train-obj: 0.150869
            Val objective improved 0.1485 → 0.1472, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.3482, mae: 0.4010, huber: 0.1508, swd: 0.1077, ept: 53.6533
    Epoch [9/50], Val Losses: mse: 0.3475, mae: 0.3861, huber: 0.1479, swd: 0.1038, ept: 51.4770
    Epoch [9/50], Test Losses: mse: 0.4207, mae: 0.4375, huber: 0.1793, swd: 0.1205, ept: 42.6098
      Epoch 9 composite train-obj: 0.150769
            No improvement (0.1479), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.3472, mae: 0.4002, huber: 0.1504, swd: 0.1076, ept: 53.7022
    Epoch [10/50], Val Losses: mse: 0.3481, mae: 0.3867, huber: 0.1481, swd: 0.1033, ept: 51.6122
    Epoch [10/50], Test Losses: mse: 0.4193, mae: 0.4369, huber: 0.1788, swd: 0.1199, ept: 42.6642
      Epoch 10 composite train-obj: 0.150403
            No improvement (0.1481), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.3469, mae: 0.3998, huber: 0.1502, swd: 0.1074, ept: 53.8126
    Epoch [11/50], Val Losses: mse: 0.3485, mae: 0.3871, huber: 0.1483, swd: 0.1034, ept: 51.5681
    Epoch [11/50], Test Losses: mse: 0.4199, mae: 0.4373, huber: 0.1792, swd: 0.1180, ept: 42.5137
      Epoch 11 composite train-obj: 0.150200
            No improvement (0.1483), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.3472, mae: 0.3998, huber: 0.1503, swd: 0.1075, ept: 53.8151
    Epoch [12/50], Val Losses: mse: 0.3446, mae: 0.3849, huber: 0.1470, swd: 0.1028, ept: 51.4443
    Epoch [12/50], Test Losses: mse: 0.4193, mae: 0.4370, huber: 0.1790, swd: 0.1207, ept: 42.6055
      Epoch 12 composite train-obj: 0.150279
            Val objective improved 0.1472 → 0.1470, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.3471, mae: 0.3996, huber: 0.1502, swd: 0.1077, ept: 53.8959
    Epoch [13/50], Val Losses: mse: 0.3488, mae: 0.3854, huber: 0.1478, swd: 0.1036, ept: 52.0910
    Epoch [13/50], Test Losses: mse: 0.4208, mae: 0.4373, huber: 0.1792, swd: 0.1201, ept: 42.8228
      Epoch 13 composite train-obj: 0.150218
            No improvement (0.1478), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.3469, mae: 0.3994, huber: 0.1501, swd: 0.1075, ept: 53.9717
    Epoch [14/50], Val Losses: mse: 0.3462, mae: 0.3842, huber: 0.1471, swd: 0.1005, ept: 51.6346
    Epoch [14/50], Test Losses: mse: 0.4216, mae: 0.4381, huber: 0.1797, swd: 0.1192, ept: 42.8316
      Epoch 14 composite train-obj: 0.150097
            No improvement (0.1471), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.3469, mae: 0.3997, huber: 0.1502, swd: 0.1076, ept: 53.8704
    Epoch [15/50], Val Losses: mse: 0.3462, mae: 0.3855, huber: 0.1476, swd: 0.1022, ept: 51.7808
    Epoch [15/50], Test Losses: mse: 0.4175, mae: 0.4359, huber: 0.1782, swd: 0.1193, ept: 42.8536
      Epoch 15 composite train-obj: 0.150192
            No improvement (0.1476), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.3466, mae: 0.3993, huber: 0.1500, swd: 0.1076, ept: 53.9047
    Epoch [16/50], Val Losses: mse: 0.3446, mae: 0.3831, huber: 0.1463, swd: 0.1031, ept: 52.1117
    Epoch [16/50], Test Losses: mse: 0.4194, mae: 0.4375, huber: 0.1790, swd: 0.1242, ept: 42.8600
      Epoch 16 composite train-obj: 0.150036
            Val objective improved 0.1470 → 0.1463, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 0.3465, mae: 0.3990, huber: 0.1500, swd: 0.1076, ept: 53.9715
    Epoch [17/50], Val Losses: mse: 0.3489, mae: 0.3877, huber: 0.1488, swd: 0.1076, ept: 51.6166
    Epoch [17/50], Test Losses: mse: 0.4201, mae: 0.4368, huber: 0.1792, swd: 0.1200, ept: 42.9033
      Epoch 17 composite train-obj: 0.149952
            No improvement (0.1488), counter 1/5
    Epoch [18/50], Train Losses: mse: 0.3469, mae: 0.3994, huber: 0.1501, swd: 0.1078, ept: 53.9622
    Epoch [18/50], Val Losses: mse: 0.3440, mae: 0.3838, huber: 0.1469, swd: 0.1003, ept: 51.8185
    Epoch [18/50], Test Losses: mse: 0.4194, mae: 0.4368, huber: 0.1789, swd: 0.1172, ept: 42.7705
      Epoch 18 composite train-obj: 0.150093
            No improvement (0.1469), counter 2/5
    Epoch [19/50], Train Losses: mse: 0.3471, mae: 0.3993, huber: 0.1501, swd: 0.1078, ept: 54.0506
    Epoch [19/50], Val Losses: mse: 0.3452, mae: 0.3854, huber: 0.1470, swd: 0.1045, ept: 52.1288
    Epoch [19/50], Test Losses: mse: 0.4208, mae: 0.4386, huber: 0.1796, swd: 0.1226, ept: 42.8300
      Epoch 19 composite train-obj: 0.150111
            No improvement (0.1470), counter 3/5
    Epoch [20/50], Train Losses: mse: 0.3466, mae: 0.3991, huber: 0.1500, swd: 0.1074, ept: 54.0629
    Epoch [20/50], Val Losses: mse: 0.3493, mae: 0.3870, huber: 0.1485, swd: 0.1033, ept: 51.7517
    Epoch [20/50], Test Losses: mse: 0.4187, mae: 0.4354, huber: 0.1782, swd: 0.1152, ept: 42.7405
      Epoch 20 composite train-obj: 0.149971
            No improvement (0.1485), counter 4/5
    Epoch [21/50], Train Losses: mse: 0.3464, mae: 0.3992, huber: 0.1500, swd: 0.1075, ept: 54.0349
    Epoch [21/50], Val Losses: mse: 0.3456, mae: 0.3842, huber: 0.1470, swd: 0.1059, ept: 51.8553
    Epoch [21/50], Test Losses: mse: 0.4196, mae: 0.4372, huber: 0.1792, swd: 0.1235, ept: 42.7301
      Epoch 21 composite train-obj: 0.149980
    Epoch [21/50], Test Losses: mse: 0.4194, mae: 0.4375, huber: 0.1790, swd: 0.1242, ept: 42.8600
    Best round's Test MSE: 0.4194, MAE: 0.4375, SWD: 0.1242
    Best round's Validation MSE: 0.3446, MAE: 0.3831, SWD: 0.1031
    Best round's Test verification MSE : 0.4194, MAE: 0.4375, SWD: 0.1242
    Time taken: 26.93 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq196_pred96_20250510_1845)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4204 ± 0.0012
      mae: 0.4380 ± 0.0006
      huber: 0.1794 ± 0.0004
      swd: 0.1244 ± 0.0014
      ept: 42.6711 ± 0.1675
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3452 ± 0.0011
      mae: 0.3846 ± 0.0016
      huber: 0.1469 ± 0.0006
      swd: 0.1033 ± 0.0009
      ept: 51.7973 ± 0.3440
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 55.69 seconds
    
    Experiment complete: DLinear_etth1_seq196_pred96_20250510_1845
    Model: DLinear
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=196,
    pred_len=196,
    channels=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 196
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 11
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5353, mae: 0.5108, huber: 0.2222, swd: 0.1632, ept: 55.8905
    Epoch [1/50], Val Losses: mse: 0.4273, mae: 0.4443, huber: 0.1800, swd: 0.1266, ept: 63.4908
    Epoch [1/50], Test Losses: mse: 0.4869, mae: 0.4941, huber: 0.2105, swd: 0.1242, ept: 56.7938
      Epoch 1 composite train-obj: 0.222176
            Val objective improved inf → 0.1800, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4451, mae: 0.4576, huber: 0.1874, swd: 0.1516, ept: 74.2203
    Epoch [2/50], Val Losses: mse: 0.4225, mae: 0.4349, huber: 0.1769, swd: 0.1320, ept: 67.6913
    Epoch [2/50], Test Losses: mse: 0.4763, mae: 0.4834, huber: 0.2054, swd: 0.1299, ept: 60.4132
      Epoch 2 composite train-obj: 0.187416
            Val objective improved 0.1800 → 0.1769, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4373, mae: 0.4514, huber: 0.1841, swd: 0.1503, ept: 77.1165
    Epoch [3/50], Val Losses: mse: 0.4149, mae: 0.4300, huber: 0.1740, swd: 0.1324, ept: 68.0004
    Epoch [3/50], Test Losses: mse: 0.4678, mae: 0.4763, huber: 0.2015, swd: 0.1282, ept: 62.1338
      Epoch 3 composite train-obj: 0.184056
            Val objective improved 0.1769 → 0.1740, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4353, mae: 0.4491, huber: 0.1830, swd: 0.1507, ept: 78.2902
    Epoch [4/50], Val Losses: mse: 0.4071, mae: 0.4250, huber: 0.1710, swd: 0.1261, ept: 68.9905
    Epoch [4/50], Test Losses: mse: 0.4661, mae: 0.4754, huber: 0.2007, swd: 0.1253, ept: 62.7270
      Epoch 4 composite train-obj: 0.182997
            Val objective improved 0.1740 → 0.1710, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4337, mae: 0.4476, huber: 0.1823, swd: 0.1502, ept: 79.1118
    Epoch [5/50], Val Losses: mse: 0.4078, mae: 0.4244, huber: 0.1708, swd: 0.1252, ept: 69.8391
    Epoch [5/50], Test Losses: mse: 0.4625, mae: 0.4731, huber: 0.1991, swd: 0.1227, ept: 62.7272
      Epoch 5 composite train-obj: 0.182266
            Val objective improved 0.1710 → 0.1708, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4321, mae: 0.4463, huber: 0.1816, swd: 0.1493, ept: 79.2823
    Epoch [6/50], Val Losses: mse: 0.4096, mae: 0.4242, huber: 0.1714, swd: 0.1283, ept: 70.4118
    Epoch [6/50], Test Losses: mse: 0.4617, mae: 0.4720, huber: 0.1988, swd: 0.1277, ept: 63.4769
      Epoch 6 composite train-obj: 0.181614
            No improvement (0.1714), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4306, mae: 0.4456, huber: 0.1811, swd: 0.1494, ept: 79.6407
    Epoch [7/50], Val Losses: mse: 0.4123, mae: 0.4266, huber: 0.1728, swd: 0.1311, ept: 69.8823
    Epoch [7/50], Test Losses: mse: 0.4608, mae: 0.4719, huber: 0.1985, swd: 0.1262, ept: 63.6834
      Epoch 7 composite train-obj: 0.181126
            No improvement (0.1728), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4314, mae: 0.4458, huber: 0.1814, swd: 0.1502, ept: 79.6854
    Epoch [8/50], Val Losses: mse: 0.4053, mae: 0.4226, huber: 0.1701, swd: 0.1277, ept: 69.7870
    Epoch [8/50], Test Losses: mse: 0.4634, mae: 0.4726, huber: 0.1993, swd: 0.1283, ept: 63.9674
      Epoch 8 composite train-obj: 0.181360
            Val objective improved 0.1708 → 0.1701, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4318, mae: 0.4455, huber: 0.1814, swd: 0.1494, ept: 80.0824
    Epoch [9/50], Val Losses: mse: 0.4119, mae: 0.4253, huber: 0.1725, swd: 0.1315, ept: 70.5690
    Epoch [9/50], Test Losses: mse: 0.4648, mae: 0.4731, huber: 0.1999, swd: 0.1289, ept: 64.5011
      Epoch 9 composite train-obj: 0.181355
            No improvement (0.1725), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.4282, mae: 0.4436, huber: 0.1800, swd: 0.1492, ept: 80.2165
    Epoch [10/50], Val Losses: mse: 0.4063, mae: 0.4219, huber: 0.1702, swd: 0.1277, ept: 71.6024
    Epoch [10/50], Test Losses: mse: 0.4632, mae: 0.4720, huber: 0.1992, swd: 0.1258, ept: 63.7778
      Epoch 10 composite train-obj: 0.180049
            No improvement (0.1702), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.4301, mae: 0.4443, huber: 0.1806, swd: 0.1501, ept: 80.4424
    Epoch [11/50], Val Losses: mse: 0.4134, mae: 0.4278, huber: 0.1734, swd: 0.1341, ept: 70.3548
    Epoch [11/50], Test Losses: mse: 0.4625, mae: 0.4721, huber: 0.1991, swd: 0.1278, ept: 63.2753
      Epoch 11 composite train-obj: 0.180648
            No improvement (0.1734), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.4281, mae: 0.4438, huber: 0.1802, swd: 0.1491, ept: 80.2799
    Epoch [12/50], Val Losses: mse: 0.4090, mae: 0.4231, huber: 0.1712, swd: 0.1292, ept: 71.4704
    Epoch [12/50], Test Losses: mse: 0.4634, mae: 0.4723, huber: 0.1993, swd: 0.1260, ept: 64.1051
      Epoch 12 composite train-obj: 0.180183
            No improvement (0.1712), counter 4/5
    Epoch [13/50], Train Losses: mse: 0.4303, mae: 0.4445, huber: 0.1808, swd: 0.1495, ept: 80.3975
    Epoch [13/50], Val Losses: mse: 0.4068, mae: 0.4234, huber: 0.1710, swd: 0.1347, ept: 70.6564
    Epoch [13/50], Test Losses: mse: 0.4643, mae: 0.4723, huber: 0.1999, swd: 0.1338, ept: 64.0054
      Epoch 13 composite train-obj: 0.180809
    Epoch [13/50], Test Losses: mse: 0.4634, mae: 0.4726, huber: 0.1993, swd: 0.1283, ept: 63.9674
    Best round's Test MSE: 0.4634, MAE: 0.4726, SWD: 0.1283
    Best round's Validation MSE: 0.4053, MAE: 0.4226, SWD: 0.1277
    Best round's Test verification MSE : 0.4634, MAE: 0.4726, SWD: 0.1283
    Time taken: 18.07 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5341, mae: 0.5098, huber: 0.2217, swd: 0.1629, ept: 56.5564
    Epoch [1/50], Val Losses: mse: 0.4267, mae: 0.4452, huber: 0.1803, swd: 0.1279, ept: 61.9857
    Epoch [1/50], Test Losses: mse: 0.4875, mae: 0.4946, huber: 0.2108, swd: 0.1282, ept: 57.2224
      Epoch 1 composite train-obj: 0.221657
            Val objective improved inf → 0.1803, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4442, mae: 0.4574, huber: 0.1871, swd: 0.1530, ept: 74.2350
    Epoch [2/50], Val Losses: mse: 0.4150, mae: 0.4314, huber: 0.1741, swd: 0.1216, ept: 66.3136
    Epoch [2/50], Test Losses: mse: 0.4731, mae: 0.4812, huber: 0.2038, swd: 0.1199, ept: 60.6997
      Epoch 2 composite train-obj: 0.187150
            Val objective improved 0.1803 → 0.1741, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4376, mae: 0.4514, huber: 0.1842, swd: 0.1513, ept: 76.9863
    Epoch [3/50], Val Losses: mse: 0.4139, mae: 0.4307, huber: 0.1739, swd: 0.1241, ept: 66.3125
    Epoch [3/50], Test Losses: mse: 0.4692, mae: 0.4772, huber: 0.2019, swd: 0.1190, ept: 62.0164
      Epoch 3 composite train-obj: 0.184175
            Val objective improved 0.1741 → 0.1739, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4343, mae: 0.4488, huber: 0.1827, swd: 0.1508, ept: 78.2394
    Epoch [4/50], Val Losses: mse: 0.4059, mae: 0.4234, huber: 0.1702, swd: 0.1193, ept: 69.5153
    Epoch [4/50], Test Losses: mse: 0.4662, mae: 0.4754, huber: 0.2006, swd: 0.1231, ept: 62.5706
      Epoch 4 composite train-obj: 0.182742
            Val objective improved 0.1739 → 0.1702, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4315, mae: 0.4467, huber: 0.1816, swd: 0.1504, ept: 79.1845
    Epoch [5/50], Val Losses: mse: 0.4063, mae: 0.4232, huber: 0.1702, swd: 0.1212, ept: 69.3004
    Epoch [5/50], Test Losses: mse: 0.4634, mae: 0.4727, huber: 0.1994, swd: 0.1228, ept: 62.4338
      Epoch 5 composite train-obj: 0.181608
            No improvement (0.1702), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4326, mae: 0.4465, huber: 0.1818, swd: 0.1504, ept: 79.4081
    Epoch [6/50], Val Losses: mse: 0.4091, mae: 0.4273, huber: 0.1722, swd: 0.1275, ept: 68.2635
    Epoch [6/50], Test Losses: mse: 0.4608, mae: 0.4714, huber: 0.1986, swd: 0.1239, ept: 62.6143
      Epoch 6 composite train-obj: 0.181779
            No improvement (0.1722), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.4304, mae: 0.4454, huber: 0.1810, swd: 0.1508, ept: 79.6721
    Epoch [7/50], Val Losses: mse: 0.4090, mae: 0.4245, huber: 0.1713, swd: 0.1225, ept: 69.2229
    Epoch [7/50], Test Losses: mse: 0.4629, mae: 0.4720, huber: 0.1991, swd: 0.1207, ept: 63.3142
      Epoch 7 composite train-obj: 0.181009
            No improvement (0.1713), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.4302, mae: 0.4450, huber: 0.1809, swd: 0.1501, ept: 80.0173
    Epoch [8/50], Val Losses: mse: 0.4112, mae: 0.4244, huber: 0.1718, swd: 0.1270, ept: 69.6674
    Epoch [8/50], Test Losses: mse: 0.4648, mae: 0.4727, huber: 0.1998, swd: 0.1275, ept: 64.2481
      Epoch 8 composite train-obj: 0.180884
            No improvement (0.1718), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.4291, mae: 0.4444, huber: 0.1805, swd: 0.1502, ept: 80.1996
    Epoch [9/50], Val Losses: mse: 0.4049, mae: 0.4235, huber: 0.1706, swd: 0.1258, ept: 70.1892
    Epoch [9/50], Test Losses: mse: 0.4617, mae: 0.4709, huber: 0.1988, swd: 0.1245, ept: 63.6191
      Epoch 9 composite train-obj: 0.180453
    Epoch [9/50], Test Losses: mse: 0.4662, mae: 0.4754, huber: 0.2006, swd: 0.1231, ept: 62.5706
    Best round's Test MSE: 0.4662, MAE: 0.4754, SWD: 0.1231
    Best round's Validation MSE: 0.4059, MAE: 0.4234, SWD: 0.1193
    Best round's Test verification MSE : 0.4662, MAE: 0.4754, SWD: 0.1231
    Time taken: 12.09 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5333, mae: 0.5097, huber: 0.2214, swd: 0.1493, ept: 57.0080
    Epoch [1/50], Val Losses: mse: 0.4276, mae: 0.4445, huber: 0.1803, swd: 0.1215, ept: 62.5219
    Epoch [1/50], Test Losses: mse: 0.4881, mae: 0.4949, huber: 0.2109, swd: 0.1167, ept: 57.1449
      Epoch 1 composite train-obj: 0.221435
            Val objective improved inf → 0.1803, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4455, mae: 0.4579, huber: 0.1876, swd: 0.1401, ept: 74.3029
    Epoch [2/50], Val Losses: mse: 0.4177, mae: 0.4349, huber: 0.1759, swd: 0.1247, ept: 65.3538
    Epoch [2/50], Test Losses: mse: 0.4731, mae: 0.4820, huber: 0.2042, swd: 0.1205, ept: 60.6620
      Epoch 2 composite train-obj: 0.187580
            Val objective improved 0.1803 → 0.1759, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4381, mae: 0.4515, huber: 0.1843, swd: 0.1389, ept: 77.0846
    Epoch [3/50], Val Losses: mse: 0.4086, mae: 0.4274, huber: 0.1718, swd: 0.1193, ept: 67.3046
    Epoch [3/50], Test Losses: mse: 0.4665, mae: 0.4775, huber: 0.2012, swd: 0.1195, ept: 61.6523
      Epoch 3 composite train-obj: 0.184250
            Val objective improved 0.1759 → 0.1718, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4337, mae: 0.4487, huber: 0.1826, swd: 0.1379, ept: 78.2715
    Epoch [4/50], Val Losses: mse: 0.4124, mae: 0.4271, huber: 0.1727, swd: 0.1216, ept: 69.4266
    Epoch [4/50], Test Losses: mse: 0.4665, mae: 0.4758, huber: 0.2010, swd: 0.1198, ept: 62.9526
      Epoch 4 composite train-obj: 0.182642
            No improvement (0.1727), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4322, mae: 0.4469, huber: 0.1818, swd: 0.1377, ept: 78.9068
    Epoch [5/50], Val Losses: mse: 0.4071, mae: 0.4238, huber: 0.1707, swd: 0.1182, ept: 69.6420
    Epoch [5/50], Test Losses: mse: 0.4632, mae: 0.4735, huber: 0.1995, swd: 0.1164, ept: 62.8970
      Epoch 5 composite train-obj: 0.181763
            Val objective improved 0.1718 → 0.1707, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4315, mae: 0.4462, huber: 0.1815, swd: 0.1378, ept: 79.3857
    Epoch [6/50], Val Losses: mse: 0.4172, mae: 0.4301, huber: 0.1747, swd: 0.1283, ept: 68.7022
    Epoch [6/50], Test Losses: mse: 0.4652, mae: 0.4750, huber: 0.2005, swd: 0.1255, ept: 63.6660
      Epoch 6 composite train-obj: 0.181489
            No improvement (0.1747), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4311, mae: 0.4460, huber: 0.1813, swd: 0.1383, ept: 79.5294
    Epoch [7/50], Val Losses: mse: 0.4072, mae: 0.4224, huber: 0.1705, swd: 0.1230, ept: 70.7671
    Epoch [7/50], Test Losses: mse: 0.4647, mae: 0.4740, huber: 0.2002, swd: 0.1214, ept: 63.1436
      Epoch 7 composite train-obj: 0.181345
            Val objective improved 0.1707 → 0.1705, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4291, mae: 0.4445, huber: 0.1805, swd: 0.1373, ept: 79.9428
    Epoch [8/50], Val Losses: mse: 0.4062, mae: 0.4237, huber: 0.1707, swd: 0.1242, ept: 69.6326
    Epoch [8/50], Test Losses: mse: 0.4633, mae: 0.4741, huber: 0.1999, swd: 0.1276, ept: 63.5613
      Epoch 8 composite train-obj: 0.180462
            No improvement (0.1707), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4297, mae: 0.4447, huber: 0.1807, swd: 0.1379, ept: 80.0534
    Epoch [9/50], Val Losses: mse: 0.4050, mae: 0.4202, huber: 0.1693, swd: 0.1179, ept: 70.0436
    Epoch [9/50], Test Losses: mse: 0.4634, mae: 0.4731, huber: 0.1995, swd: 0.1184, ept: 63.4761
      Epoch 9 composite train-obj: 0.180744
            Val objective improved 0.1705 → 0.1693, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.4299, mae: 0.4445, huber: 0.1807, swd: 0.1379, ept: 80.3354
    Epoch [10/50], Val Losses: mse: 0.4058, mae: 0.4209, huber: 0.1697, swd: 0.1207, ept: 72.0288
    Epoch [10/50], Test Losses: mse: 0.4618, mae: 0.4720, huber: 0.1989, swd: 0.1207, ept: 63.6564
      Epoch 10 composite train-obj: 0.180677
            No improvement (0.1697), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.4296, mae: 0.4442, huber: 0.1805, swd: 0.1383, ept: 80.2983
    Epoch [11/50], Val Losses: mse: 0.4030, mae: 0.4196, huber: 0.1688, swd: 0.1157, ept: 71.4947
    Epoch [11/50], Test Losses: mse: 0.4598, mae: 0.4697, huber: 0.1977, swd: 0.1136, ept: 64.0930
      Epoch 11 composite train-obj: 0.180501
            Val objective improved 0.1693 → 0.1688, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.4306, mae: 0.4447, huber: 0.1809, swd: 0.1381, ept: 80.3444
    Epoch [12/50], Val Losses: mse: 0.4049, mae: 0.4205, huber: 0.1694, swd: 0.1186, ept: 72.8285
    Epoch [12/50], Test Losses: mse: 0.4640, mae: 0.4731, huber: 0.1997, swd: 0.1189, ept: 63.3430
      Epoch 12 composite train-obj: 0.180922
            No improvement (0.1694), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.4295, mae: 0.4442, huber: 0.1805, swd: 0.1384, ept: 80.5458
    Epoch [13/50], Val Losses: mse: 0.4030, mae: 0.4211, huber: 0.1693, swd: 0.1217, ept: 72.6683
    Epoch [13/50], Test Losses: mse: 0.4601, mae: 0.4705, huber: 0.1982, swd: 0.1228, ept: 63.6506
      Epoch 13 composite train-obj: 0.180550
            No improvement (0.1693), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.4285, mae: 0.4437, huber: 0.1802, swd: 0.1378, ept: 80.4485
    Epoch [14/50], Val Losses: mse: 0.4049, mae: 0.4204, huber: 0.1693, swd: 0.1196, ept: 72.3072
    Epoch [14/50], Test Losses: mse: 0.4586, mae: 0.4698, huber: 0.1975, swd: 0.1191, ept: 63.1467
      Epoch 14 composite train-obj: 0.180223
            No improvement (0.1693), counter 3/5
    Epoch [15/50], Train Losses: mse: 0.4292, mae: 0.4440, huber: 0.1804, swd: 0.1381, ept: 80.4233
    Epoch [15/50], Val Losses: mse: 0.4085, mae: 0.4249, huber: 0.1715, swd: 0.1243, ept: 69.6550
    Epoch [15/50], Test Losses: mse: 0.4607, mae: 0.4701, huber: 0.1983, swd: 0.1188, ept: 63.5487
      Epoch 15 composite train-obj: 0.180444
            No improvement (0.1715), counter 4/5
    Epoch [16/50], Train Losses: mse: 0.4278, mae: 0.4434, huber: 0.1800, swd: 0.1378, ept: 80.5282
    Epoch [16/50], Val Losses: mse: 0.4033, mae: 0.4212, huber: 0.1695, swd: 0.1246, ept: 72.5472
    Epoch [16/50], Test Losses: mse: 0.4600, mae: 0.4720, huber: 0.1986, swd: 0.1265, ept: 63.8500
      Epoch 16 composite train-obj: 0.179999
    Epoch [16/50], Test Losses: mse: 0.4598, mae: 0.4697, huber: 0.1977, swd: 0.1136, ept: 64.0930
    Best round's Test MSE: 0.4598, MAE: 0.4697, SWD: 0.1136
    Best round's Validation MSE: 0.4030, MAE: 0.4196, SWD: 0.1157
    Best round's Test verification MSE : 0.4598, MAE: 0.4697, SWD: 0.1136
    Time taken: 20.77 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq196_pred196_20250510_1846)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4631 ± 0.0026
      mae: 0.4726 ± 0.0023
      huber: 0.1992 ± 0.0012
      swd: 0.1217 ± 0.0061
      ept: 63.5437 ± 0.6900
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4047 ± 0.0013
      mae: 0.4219 ± 0.0016
      huber: 0.1697 ± 0.0007
      swd: 0.1209 ± 0.0050
      ept: 70.2657 ± 0.8761
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 50.98 seconds
    
    Experiment complete: DLinear_etth1_seq196_pred196_20250510_1846
    Model: DLinear
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=196,
    pred_len=336,
    channels=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 10
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5913, mae: 0.5400, huber: 0.2432, swd: 0.1806, ept: 68.9799
    Epoch [1/50], Val Losses: mse: 0.4629, mae: 0.4596, huber: 0.1917, swd: 0.1245, ept: 76.6326
    Epoch [1/50], Test Losses: mse: 0.5406, mae: 0.5302, huber: 0.2343, swd: 0.1354, ept: 76.7167
      Epoch 1 composite train-obj: 0.243220
            Val objective improved inf → 0.1917, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5064, mae: 0.4920, huber: 0.2112, swd: 0.1734, ept: 93.7002
    Epoch [2/50], Val Losses: mse: 0.4538, mae: 0.4484, huber: 0.1867, swd: 0.1230, ept: 81.9006
    Epoch [2/50], Test Losses: mse: 0.5270, mae: 0.5176, huber: 0.2280, swd: 0.1274, ept: 81.2735
      Epoch 2 composite train-obj: 0.211223
            Val objective improved 0.1917 → 0.1867, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5003, mae: 0.4862, huber: 0.2083, swd: 0.1711, ept: 98.2618
    Epoch [3/50], Val Losses: mse: 0.4505, mae: 0.4437, huber: 0.1845, swd: 0.1245, ept: 84.3665
    Epoch [3/50], Test Losses: mse: 0.5179, mae: 0.5127, huber: 0.2244, swd: 0.1313, ept: 82.3683
      Epoch 3 composite train-obj: 0.208285
            Val objective improved 0.1867 → 0.1845, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4963, mae: 0.4836, huber: 0.2067, swd: 0.1717, ept: 99.7080
    Epoch [4/50], Val Losses: mse: 0.4494, mae: 0.4454, huber: 0.1851, swd: 0.1287, ept: 82.8986
    Epoch [4/50], Test Losses: mse: 0.5176, mae: 0.5105, huber: 0.2240, swd: 0.1384, ept: 83.3282
      Epoch 4 composite train-obj: 0.206690
            No improvement (0.1851), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4950, mae: 0.4819, huber: 0.2059, swd: 0.1715, ept: 101.0491
    Epoch [5/50], Val Losses: mse: 0.4495, mae: 0.4430, huber: 0.1845, swd: 0.1261, ept: 83.7719
    Epoch [5/50], Test Losses: mse: 0.5187, mae: 0.5124, huber: 0.2246, swd: 0.1361, ept: 84.7264
      Epoch 5 composite train-obj: 0.205909
            No improvement (0.1845), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4937, mae: 0.4814, huber: 0.2056, swd: 0.1707, ept: 101.0002
    Epoch [6/50], Val Losses: mse: 0.4449, mae: 0.4395, huber: 0.1825, swd: 0.1246, ept: 85.6508
    Epoch [6/50], Test Losses: mse: 0.5171, mae: 0.5086, huber: 0.2234, swd: 0.1317, ept: 84.3375
      Epoch 6 composite train-obj: 0.205574
            Val objective improved 0.1845 → 0.1825, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4951, mae: 0.4818, huber: 0.2060, swd: 0.1729, ept: 101.5603
    Epoch [7/50], Val Losses: mse: 0.4423, mae: 0.4369, huber: 0.1812, swd: 0.1244, ept: 89.3392
    Epoch [7/50], Test Losses: mse: 0.5131, mae: 0.5088, huber: 0.2223, swd: 0.1374, ept: 84.4734
      Epoch 7 composite train-obj: 0.205991
            Val objective improved 0.1825 → 0.1812, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4929, mae: 0.4806, huber: 0.2052, swd: 0.1716, ept: 102.3703
    Epoch [8/50], Val Losses: mse: 0.4481, mae: 0.4426, huber: 0.1842, swd: 0.1303, ept: 83.7971
    Epoch [8/50], Test Losses: mse: 0.5129, mae: 0.5070, huber: 0.2220, swd: 0.1366, ept: 84.4447
      Epoch 8 composite train-obj: 0.205156
            No improvement (0.1842), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4932, mae: 0.4803, huber: 0.2052, swd: 0.1721, ept: 102.3877
    Epoch [9/50], Val Losses: mse: 0.4387, mae: 0.4347, huber: 0.1800, swd: 0.1213, ept: 88.4716
    Epoch [9/50], Test Losses: mse: 0.5133, mae: 0.5069, huber: 0.2220, swd: 0.1326, ept: 84.9014
      Epoch 9 composite train-obj: 0.205157
            Val objective improved 0.1812 → 0.1800, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.4934, mae: 0.4798, huber: 0.2050, swd: 0.1726, ept: 102.7458
    Epoch [10/50], Val Losses: mse: 0.4382, mae: 0.4351, huber: 0.1800, swd: 0.1250, ept: 86.8264
    Epoch [10/50], Test Losses: mse: 0.5145, mae: 0.5106, huber: 0.2230, swd: 0.1409, ept: 85.4925
      Epoch 10 composite train-obj: 0.204991
            Val objective improved 0.1800 → 0.1800, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.4918, mae: 0.4793, huber: 0.2046, swd: 0.1712, ept: 102.9140
    Epoch [11/50], Val Losses: mse: 0.4361, mae: 0.4333, huber: 0.1791, swd: 0.1220, ept: 89.4520
    Epoch [11/50], Test Losses: mse: 0.5128, mae: 0.5084, huber: 0.2221, swd: 0.1388, ept: 84.0746
      Epoch 11 composite train-obj: 0.204634
            Val objective improved 0.1800 → 0.1791, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.4921, mae: 0.4798, huber: 0.2048, swd: 0.1723, ept: 102.8976
    Epoch [12/50], Val Losses: mse: 0.4382, mae: 0.4353, huber: 0.1799, swd: 0.1260, ept: 90.3084
    Epoch [12/50], Test Losses: mse: 0.5135, mae: 0.5099, huber: 0.2228, swd: 0.1410, ept: 84.9952
      Epoch 12 composite train-obj: 0.204812
            No improvement (0.1799), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.4920, mae: 0.4795, huber: 0.2047, swd: 0.1721, ept: 103.0806
    Epoch [13/50], Val Losses: mse: 0.4408, mae: 0.4350, huber: 0.1805, swd: 0.1242, ept: 87.4837
    Epoch [13/50], Test Losses: mse: 0.5125, mae: 0.5076, huber: 0.2220, swd: 0.1356, ept: 85.2508
      Epoch 13 composite train-obj: 0.204736
            No improvement (0.1805), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.4918, mae: 0.4794, huber: 0.2047, swd: 0.1721, ept: 103.0295
    Epoch [14/50], Val Losses: mse: 0.4372, mae: 0.4357, huber: 0.1802, swd: 0.1257, ept: 84.4632
    Epoch [14/50], Test Losses: mse: 0.5106, mae: 0.5057, huber: 0.2211, swd: 0.1381, ept: 84.6139
      Epoch 14 composite train-obj: 0.204698
            No improvement (0.1802), counter 3/5
    Epoch [15/50], Train Losses: mse: 0.4917, mae: 0.4792, huber: 0.2046, swd: 0.1719, ept: 102.9678
    Epoch [15/50], Val Losses: mse: 0.4364, mae: 0.4326, huber: 0.1791, swd: 0.1207, ept: 92.1587
    Epoch [15/50], Test Losses: mse: 0.5176, mae: 0.5085, huber: 0.2236, swd: 0.1349, ept: 84.4944
      Epoch 15 composite train-obj: 0.204627
            No improvement (0.1791), counter 4/5
    Epoch [16/50], Train Losses: mse: 0.4918, mae: 0.4791, huber: 0.2046, swd: 0.1721, ept: 103.1819
    Epoch [16/50], Val Losses: mse: 0.4444, mae: 0.4380, huber: 0.1823, swd: 0.1281, ept: 85.8084
    Epoch [16/50], Test Losses: mse: 0.5160, mae: 0.5099, huber: 0.2234, swd: 0.1403, ept: 85.2300
      Epoch 16 composite train-obj: 0.204569
    Epoch [16/50], Test Losses: mse: 0.5128, mae: 0.5084, huber: 0.2221, swd: 0.1388, ept: 84.0746
    Best round's Test MSE: 0.5128, MAE: 0.5084, SWD: 0.1388
    Best round's Validation MSE: 0.4361, MAE: 0.4333, SWD: 0.1220
    Best round's Test verification MSE : 0.5128, MAE: 0.5084, SWD: 0.1388
    Time taken: 21.90 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5891, mae: 0.5390, huber: 0.2425, swd: 0.1844, ept: 68.9698
    Epoch [1/50], Val Losses: mse: 0.4602, mae: 0.4572, huber: 0.1904, swd: 0.1208, ept: 80.0700
    Epoch [1/50], Test Losses: mse: 0.5400, mae: 0.5289, huber: 0.2338, swd: 0.1340, ept: 76.7642
      Epoch 1 composite train-obj: 0.242503
            Val objective improved inf → 0.1904, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5074, mae: 0.4920, huber: 0.2114, swd: 0.1768, ept: 93.7759
    Epoch [2/50], Val Losses: mse: 0.4532, mae: 0.4486, huber: 0.1868, swd: 0.1267, ept: 81.9628
    Epoch [2/50], Test Losses: mse: 0.5322, mae: 0.5224, huber: 0.2304, swd: 0.1442, ept: 80.3909
      Epoch 2 composite train-obj: 0.211367
            Val objective improved 0.1904 → 0.1868, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5007, mae: 0.4862, huber: 0.2083, swd: 0.1760, ept: 98.0540
    Epoch [3/50], Val Losses: mse: 0.4437, mae: 0.4404, huber: 0.1820, swd: 0.1185, ept: 88.8801
    Epoch [3/50], Test Losses: mse: 0.5186, mae: 0.5133, huber: 0.2247, swd: 0.1355, ept: 81.6635
      Epoch 3 composite train-obj: 0.208331
            Val objective improved 0.1868 → 0.1820, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4976, mae: 0.4839, huber: 0.2070, swd: 0.1754, ept: 99.7568
    Epoch [4/50], Val Losses: mse: 0.4492, mae: 0.4440, huber: 0.1849, swd: 0.1251, ept: 85.5505
    Epoch [4/50], Test Losses: mse: 0.5211, mae: 0.5125, huber: 0.2253, swd: 0.1403, ept: 83.2777
      Epoch 4 composite train-obj: 0.207002
            No improvement (0.1849), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4949, mae: 0.4823, huber: 0.2060, swd: 0.1751, ept: 100.9387
    Epoch [5/50], Val Losses: mse: 0.4445, mae: 0.4395, huber: 0.1823, swd: 0.1211, ept: 85.3061
    Epoch [5/50], Test Losses: mse: 0.5151, mae: 0.5084, huber: 0.2228, swd: 0.1300, ept: 84.6709
      Epoch 5 composite train-obj: 0.206049
            No improvement (0.1823), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4940, mae: 0.4813, huber: 0.2056, swd: 0.1750, ept: 101.6715
    Epoch [6/50], Val Losses: mse: 0.4458, mae: 0.4390, huber: 0.1824, swd: 0.1235, ept: 87.8138
    Epoch [6/50], Test Losses: mse: 0.5155, mae: 0.5111, huber: 0.2233, swd: 0.1368, ept: 83.3013
      Epoch 6 composite train-obj: 0.205571
            No improvement (0.1824), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.4946, mae: 0.4813, huber: 0.2057, swd: 0.1755, ept: 102.0454
    Epoch [7/50], Val Losses: mse: 0.4438, mae: 0.4384, huber: 0.1821, swd: 0.1225, ept: 89.4254
    Epoch [7/50], Test Losses: mse: 0.5187, mae: 0.5115, huber: 0.2244, swd: 0.1405, ept: 83.3370
      Epoch 7 composite train-obj: 0.205747
            No improvement (0.1821), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4923, mae: 0.4802, huber: 0.2050, swd: 0.1755, ept: 101.9987
    Epoch [8/50], Val Losses: mse: 0.4447, mae: 0.4398, huber: 0.1829, swd: 0.1267, ept: 85.0075
    Epoch [8/50], Test Losses: mse: 0.5183, mae: 0.5120, huber: 0.2244, swd: 0.1423, ept: 85.0417
      Epoch 8 composite train-obj: 0.204963
    Epoch [8/50], Test Losses: mse: 0.5186, mae: 0.5133, huber: 0.2247, swd: 0.1355, ept: 81.6635
    Best round's Test MSE: 0.5186, MAE: 0.5133, SWD: 0.1355
    Best round's Validation MSE: 0.4437, MAE: 0.4404, SWD: 0.1185
    Best round's Test verification MSE : 0.5186, MAE: 0.5133, SWD: 0.1355
    Time taken: 11.12 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5886, mae: 0.5388, huber: 0.2424, swd: 0.1799, ept: 69.0782
    Epoch [1/50], Val Losses: mse: 0.4621, mae: 0.4590, huber: 0.1915, swd: 0.1284, ept: 76.4565
    Epoch [1/50], Test Losses: mse: 0.5407, mae: 0.5303, huber: 0.2345, swd: 0.1406, ept: 76.6358
      Epoch 1 composite train-obj: 0.242364
            Val objective improved inf → 0.1915, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5080, mae: 0.4923, huber: 0.2116, swd: 0.1747, ept: 93.8076
    Epoch [2/50], Val Losses: mse: 0.4575, mae: 0.4518, huber: 0.1885, swd: 0.1320, ept: 79.1093
    Epoch [2/50], Test Losses: mse: 0.5260, mae: 0.5187, huber: 0.2278, swd: 0.1414, ept: 80.9390
      Epoch 2 composite train-obj: 0.211646
            Val objective improved 0.1915 → 0.1885, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5008, mae: 0.4867, huber: 0.2085, swd: 0.1750, ept: 97.7635
    Epoch [3/50], Val Losses: mse: 0.4529, mae: 0.4481, huber: 0.1866, swd: 0.1303, ept: 81.9072
    Epoch [3/50], Test Losses: mse: 0.5172, mae: 0.5112, huber: 0.2239, swd: 0.1342, ept: 82.9356
      Epoch 3 composite train-obj: 0.208492
            Val objective improved 0.1885 → 0.1866, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4970, mae: 0.4836, huber: 0.2068, swd: 0.1740, ept: 100.0603
    Epoch [4/50], Val Losses: mse: 0.4467, mae: 0.4414, huber: 0.1833, swd: 0.1243, ept: 85.0367
    Epoch [4/50], Test Losses: mse: 0.5170, mae: 0.5109, huber: 0.2238, swd: 0.1321, ept: 84.1631
      Epoch 4 composite train-obj: 0.206821
            Val objective improved 0.1866 → 0.1833, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4943, mae: 0.4818, huber: 0.2058, swd: 0.1723, ept: 100.8242
    Epoch [5/50], Val Losses: mse: 0.4483, mae: 0.4437, huber: 0.1844, swd: 0.1281, ept: 83.2013
    Epoch [5/50], Test Losses: mse: 0.5159, mae: 0.5100, huber: 0.2233, swd: 0.1375, ept: 83.5132
      Epoch 5 composite train-obj: 0.205769
            No improvement (0.1844), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4962, mae: 0.4822, huber: 0.2062, swd: 0.1745, ept: 101.2940
    Epoch [6/50], Val Losses: mse: 0.4586, mae: 0.4530, huber: 0.1894, swd: 0.1393, ept: 82.0235
    Epoch [6/50], Test Losses: mse: 0.5138, mae: 0.5100, huber: 0.2227, swd: 0.1414, ept: 83.2322
      Epoch 6 composite train-obj: 0.206236
            No improvement (0.1894), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.4939, mae: 0.4812, huber: 0.2055, swd: 0.1730, ept: 101.6918
    Epoch [7/50], Val Losses: mse: 0.4504, mae: 0.4457, huber: 0.1856, swd: 0.1344, ept: 83.4243
    Epoch [7/50], Test Losses: mse: 0.5145, mae: 0.5102, huber: 0.2230, swd: 0.1419, ept: 83.5413
      Epoch 7 composite train-obj: 0.205494
            No improvement (0.1856), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.4934, mae: 0.4806, huber: 0.2053, swd: 0.1741, ept: 102.5147
    Epoch [8/50], Val Losses: mse: 0.4459, mae: 0.4410, huber: 0.1834, swd: 0.1323, ept: 85.2066
    Epoch [8/50], Test Losses: mse: 0.5191, mae: 0.5118, huber: 0.2247, swd: 0.1423, ept: 85.2324
      Epoch 8 composite train-obj: 0.205263
            No improvement (0.1834), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.4941, mae: 0.4806, huber: 0.2054, swd: 0.1740, ept: 102.5057
    Epoch [9/50], Val Losses: mse: 0.4395, mae: 0.4362, huber: 0.1806, swd: 0.1254, ept: 87.5343
    Epoch [9/50], Test Losses: mse: 0.5129, mae: 0.5062, huber: 0.2217, swd: 0.1357, ept: 84.2295
      Epoch 9 composite train-obj: 0.205368
            Val objective improved 0.1833 → 0.1806, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.4937, mae: 0.4802, huber: 0.2052, swd: 0.1736, ept: 102.6463
    Epoch [10/50], Val Losses: mse: 0.4378, mae: 0.4350, huber: 0.1800, swd: 0.1250, ept: 87.7920
    Epoch [10/50], Test Losses: mse: 0.5160, mae: 0.5090, huber: 0.2232, swd: 0.1404, ept: 85.6032
      Epoch 10 composite train-obj: 0.205216
            Val objective improved 0.1806 → 0.1800, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.4921, mae: 0.4800, huber: 0.2048, swd: 0.1751, ept: 102.6679
    Epoch [11/50], Val Losses: mse: 0.4483, mae: 0.4429, huber: 0.1845, swd: 0.1338, ept: 84.6667
    Epoch [11/50], Test Losses: mse: 0.5119, mae: 0.5071, huber: 0.2218, swd: 0.1379, ept: 84.5532
      Epoch 11 composite train-obj: 0.204843
            No improvement (0.1845), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.4921, mae: 0.4795, huber: 0.2047, swd: 0.1731, ept: 102.9731
    Epoch [12/50], Val Losses: mse: 0.4419, mae: 0.4362, huber: 0.1812, swd: 0.1268, ept: 90.0204
    Epoch [12/50], Test Losses: mse: 0.5137, mae: 0.5075, huber: 0.2223, swd: 0.1381, ept: 84.1673
      Epoch 12 composite train-obj: 0.204685
            No improvement (0.1812), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.4937, mae: 0.4799, huber: 0.2052, swd: 0.1750, ept: 103.1913
    Epoch [13/50], Val Losses: mse: 0.4402, mae: 0.4377, huber: 0.1814, swd: 0.1281, ept: 87.7951
    Epoch [13/50], Test Losses: mse: 0.5136, mae: 0.5078, huber: 0.2223, swd: 0.1410, ept: 85.5531
      Epoch 13 composite train-obj: 0.205153
            No improvement (0.1814), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.4924, mae: 0.4798, huber: 0.2049, swd: 0.1737, ept: 103.1763
    Epoch [14/50], Val Losses: mse: 0.4410, mae: 0.4382, huber: 0.1817, swd: 0.1331, ept: 87.1372
    Epoch [14/50], Test Losses: mse: 0.5108, mae: 0.5063, huber: 0.2213, swd: 0.1463, ept: 83.9325
      Epoch 14 composite train-obj: 0.204920
            No improvement (0.1817), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.4914, mae: 0.4790, huber: 0.2044, swd: 0.1745, ept: 103.2144
    Epoch [15/50], Val Losses: mse: 0.4366, mae: 0.4335, huber: 0.1792, swd: 0.1214, ept: 89.5953
    Epoch [15/50], Test Losses: mse: 0.5090, mae: 0.5046, huber: 0.2203, swd: 0.1332, ept: 85.2909
      Epoch 15 composite train-obj: 0.204397
            Val objective improved 0.1800 → 0.1792, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 0.4932, mae: 0.4792, huber: 0.2048, swd: 0.1741, ept: 103.3896
    Epoch [16/50], Val Losses: mse: 0.4378, mae: 0.4338, huber: 0.1796, swd: 0.1245, ept: 87.7643
    Epoch [16/50], Test Losses: mse: 0.5108, mae: 0.5068, huber: 0.2213, swd: 0.1398, ept: 84.2666
      Epoch 16 composite train-obj: 0.204775
            No improvement (0.1796), counter 1/5
    Epoch [17/50], Train Losses: mse: 0.4927, mae: 0.4794, huber: 0.2049, swd: 0.1752, ept: 103.2961
    Epoch [17/50], Val Losses: mse: 0.4354, mae: 0.4335, huber: 0.1789, swd: 0.1218, ept: 87.0396
    Epoch [17/50], Test Losses: mse: 0.5063, mae: 0.5044, huber: 0.2194, swd: 0.1372, ept: 83.8298
      Epoch 17 composite train-obj: 0.204852
            Val objective improved 0.1792 → 0.1789, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 0.4904, mae: 0.4786, huber: 0.2042, swd: 0.1737, ept: 103.2499
    Epoch [18/50], Val Losses: mse: 0.4453, mae: 0.4401, huber: 0.1833, swd: 0.1324, ept: 85.5609
    Epoch [18/50], Test Losses: mse: 0.5124, mae: 0.5064, huber: 0.2218, swd: 0.1393, ept: 85.3764
      Epoch 18 composite train-obj: 0.204174
            No improvement (0.1833), counter 1/5
    Epoch [19/50], Train Losses: mse: 0.4923, mae: 0.4791, huber: 0.2046, swd: 0.1742, ept: 103.3549
    Epoch [19/50], Val Losses: mse: 0.4336, mae: 0.4329, huber: 0.1785, swd: 0.1230, ept: 86.1729
    Epoch [19/50], Test Losses: mse: 0.5070, mae: 0.5025, huber: 0.2194, swd: 0.1358, ept: 85.1049
      Epoch 19 composite train-obj: 0.204631
            Val objective improved 0.1789 → 0.1785, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 0.4928, mae: 0.4794, huber: 0.2048, swd: 0.1744, ept: 103.3211
    Epoch [20/50], Val Losses: mse: 0.4436, mae: 0.4366, huber: 0.1816, swd: 0.1284, ept: 87.9294
    Epoch [20/50], Test Losses: mse: 0.5166, mae: 0.5116, huber: 0.2240, swd: 0.1416, ept: 84.6519
      Epoch 20 composite train-obj: 0.204815
            No improvement (0.1816), counter 1/5
    Epoch [21/50], Train Losses: mse: 0.4932, mae: 0.4795, huber: 0.2049, swd: 0.1748, ept: 103.5955
    Epoch [21/50], Val Losses: mse: 0.4370, mae: 0.4317, huber: 0.1789, swd: 0.1223, ept: 92.1394
    Epoch [21/50], Test Losses: mse: 0.5144, mae: 0.5067, huber: 0.2224, swd: 0.1325, ept: 84.6597
      Epoch 21 composite train-obj: 0.204913
            No improvement (0.1789), counter 2/5
    Epoch [22/50], Train Losses: mse: 0.4915, mae: 0.4789, huber: 0.2045, swd: 0.1734, ept: 103.2535
    Epoch [22/50], Val Losses: mse: 0.4368, mae: 0.4321, huber: 0.1788, swd: 0.1240, ept: 91.4311
    Epoch [22/50], Test Losses: mse: 0.5124, mae: 0.5069, huber: 0.2219, swd: 0.1373, ept: 84.2152
      Epoch 22 composite train-obj: 0.204487
            No improvement (0.1788), counter 3/5
    Epoch [23/50], Train Losses: mse: 0.4923, mae: 0.4791, huber: 0.2047, swd: 0.1743, ept: 103.4827
    Epoch [23/50], Val Losses: mse: 0.4435, mae: 0.4415, huber: 0.1831, swd: 0.1317, ept: 83.5444
    Epoch [23/50], Test Losses: mse: 0.5110, mae: 0.5067, huber: 0.2213, swd: 0.1437, ept: 84.4304
      Epoch 23 composite train-obj: 0.204690
            No improvement (0.1831), counter 4/5
    Epoch [24/50], Train Losses: mse: 0.4916, mae: 0.4790, huber: 0.2045, swd: 0.1745, ept: 103.5811
    Epoch [24/50], Val Losses: mse: 0.4393, mae: 0.4351, huber: 0.1805, swd: 0.1288, ept: 88.7677
    Epoch [24/50], Test Losses: mse: 0.5116, mae: 0.5071, huber: 0.2217, swd: 0.1428, ept: 85.2244
      Epoch 24 composite train-obj: 0.204495
    Epoch [24/50], Test Losses: mse: 0.5070, mae: 0.5025, huber: 0.2194, swd: 0.1358, ept: 85.1049
    Best round's Test MSE: 0.5070, MAE: 0.5025, SWD: 0.1358
    Best round's Validation MSE: 0.4336, MAE: 0.4329, SWD: 0.1230
    Best round's Test verification MSE : 0.5070, MAE: 0.5025, SWD: 0.1358
    Time taken: 30.30 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq196_pred336_20250510_1847)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5128 ± 0.0047
      mae: 0.5081 ± 0.0044
      huber: 0.2221 ± 0.0022
      swd: 0.1367 ± 0.0015
      ept: 83.6143 ± 1.4422
      count: 10.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4378 ± 0.0043
      mae: 0.4356 ± 0.0034
      huber: 0.1799 ± 0.0016
      swd: 0.1212 ± 0.0020
      ept: 88.1683 ± 1.4302
      count: 10.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 63.36 seconds
    
    Experiment complete: DLinear_etth1_seq196_pred336_20250510_1847
    Model: DLinear
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=196,
    pred_len=720,
    channels=data_mgr.datasets['etth1']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([1.1128, 0.9665, 1.1088, 0.9386, 0.9797, 0.9051, 1.0261],
           device='cuda:0')
    Train set sample shapes: torch.Size([196, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([196, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 196, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 196
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 7
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6654, mae: 0.5844, huber: 0.2734, swd: 0.2290, ept: 79.7444
    Epoch [1/50], Val Losses: mse: 0.4585, mae: 0.4798, huber: 0.1988, swd: 0.1224, ept: 111.0235
    Epoch [1/50], Test Losses: mse: 0.6603, mae: 0.6042, huber: 0.2852, swd: 0.1824, ept: 116.4101
      Epoch 1 composite train-obj: 0.273364
            Val objective improved inf → 0.1988, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5929, mae: 0.5432, huber: 0.2454, swd: 0.2268, ept: 114.2842
    Epoch [2/50], Val Losses: mse: 0.4430, mae: 0.4620, huber: 0.1906, swd: 0.1147, ept: 122.3257
    Epoch [2/50], Test Losses: mse: 0.6551, mae: 0.5975, huber: 0.2826, swd: 0.1762, ept: 122.2646
      Epoch 2 composite train-obj: 0.245444
            Val objective improved 0.1988 → 0.1906, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5871, mae: 0.5379, huber: 0.2426, swd: 0.2259, ept: 121.7273
    Epoch [3/50], Val Losses: mse: 0.4489, mae: 0.4663, huber: 0.1928, swd: 0.1219, ept: 107.7639
    Epoch [3/50], Test Losses: mse: 0.6475, mae: 0.5913, huber: 0.2792, swd: 0.1735, ept: 124.4496
      Epoch 3 composite train-obj: 0.242650
            No improvement (0.1928), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5844, mae: 0.5357, huber: 0.2414, swd: 0.2261, ept: 124.3222
    Epoch [4/50], Val Losses: mse: 0.4496, mae: 0.4709, huber: 0.1941, swd: 0.1323, ept: 103.3709
    Epoch [4/50], Test Losses: mse: 0.6405, mae: 0.5902, huber: 0.2770, swd: 0.1882, ept: 125.5385
      Epoch 4 composite train-obj: 0.241413
            No improvement (0.1941), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5825, mae: 0.5349, huber: 0.2409, swd: 0.2265, ept: 125.8152
    Epoch [5/50], Val Losses: mse: 0.4383, mae: 0.4580, huber: 0.1886, swd: 0.1203, ept: 120.0869
    Epoch [5/50], Test Losses: mse: 0.6473, mae: 0.5919, huber: 0.2795, swd: 0.1829, ept: 124.9129
      Epoch 5 composite train-obj: 0.240865
            Val objective improved 0.1906 → 0.1886, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5811, mae: 0.5338, huber: 0.2402, swd: 0.2262, ept: 127.0541
    Epoch [6/50], Val Losses: mse: 0.4431, mae: 0.4624, huber: 0.1907, swd: 0.1276, ept: 107.3206
    Epoch [6/50], Test Losses: mse: 0.6454, mae: 0.5908, huber: 0.2788, swd: 0.1834, ept: 126.0110
      Epoch 6 composite train-obj: 0.240227
            No improvement (0.1907), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.5818, mae: 0.5335, huber: 0.2402, swd: 0.2262, ept: 127.4826
    Epoch [7/50], Val Losses: mse: 0.4293, mae: 0.4487, huber: 0.1841, swd: 0.1127, ept: 128.7836
    Epoch [7/50], Test Losses: mse: 0.6508, mae: 0.5916, huber: 0.2805, swd: 0.1738, ept: 124.7807
      Epoch 7 composite train-obj: 0.240235
            Val objective improved 0.1886 → 0.1841, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.5810, mae: 0.5330, huber: 0.2400, swd: 0.2263, ept: 128.1562
    Epoch [8/50], Val Losses: mse: 0.4347, mae: 0.4555, huber: 0.1871, swd: 0.1237, ept: 115.5787
    Epoch [8/50], Test Losses: mse: 0.6436, mae: 0.5902, huber: 0.2782, swd: 0.1848, ept: 126.5352
      Epoch 8 composite train-obj: 0.239975
            No improvement (0.1871), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.5811, mae: 0.5333, huber: 0.2401, swd: 0.2281, ept: 128.4382
    Epoch [9/50], Val Losses: mse: 0.4355, mae: 0.4542, huber: 0.1868, swd: 0.1207, ept: 127.2597
    Epoch [9/50], Test Losses: mse: 0.6431, mae: 0.5914, huber: 0.2782, swd: 0.1851, ept: 123.0962
      Epoch 9 composite train-obj: 0.240080
            No improvement (0.1868), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.5799, mae: 0.5326, huber: 0.2397, swd: 0.2264, ept: 128.7854
    Epoch [10/50], Val Losses: mse: 0.4391, mae: 0.4588, huber: 0.1890, swd: 0.1292, ept: 115.6157
    Epoch [10/50], Test Losses: mse: 0.6377, mae: 0.5863, huber: 0.2757, swd: 0.1888, ept: 126.3810
      Epoch 10 composite train-obj: 0.239697
            No improvement (0.1890), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.5804, mae: 0.5325, huber: 0.2396, swd: 0.2275, ept: 129.2890
    Epoch [11/50], Val Losses: mse: 0.4423, mae: 0.4616, huber: 0.1902, swd: 0.1253, ept: 110.2860
    Epoch [11/50], Test Losses: mse: 0.6431, mae: 0.5902, huber: 0.2781, swd: 0.1797, ept: 127.3357
      Epoch 11 composite train-obj: 0.239646
            No improvement (0.1902), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.5793, mae: 0.5320, huber: 0.2394, swd: 0.2271, ept: 129.8057
    Epoch [12/50], Val Losses: mse: 0.4309, mae: 0.4513, huber: 0.1852, swd: 0.1208, ept: 125.1453
    Epoch [12/50], Test Losses: mse: 0.6429, mae: 0.5883, huber: 0.2777, swd: 0.1842, ept: 128.2434
      Epoch 12 composite train-obj: 0.239363
    Epoch [12/50], Test Losses: mse: 0.6508, mae: 0.5916, huber: 0.2805, swd: 0.1738, ept: 124.7807
    Best round's Test MSE: 0.6508, MAE: 0.5916, SWD: 0.1738
    Best round's Validation MSE: 0.4293, MAE: 0.4487, SWD: 0.1127
    Best round's Test verification MSE : 0.6508, MAE: 0.5916, SWD: 0.1738
    Time taken: 12.33 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6674, mae: 0.5853, huber: 0.2741, swd: 0.2223, ept: 79.5071
    Epoch [1/50], Val Losses: mse: 0.4565, mae: 0.4784, huber: 0.1980, swd: 0.1177, ept: 101.3557
    Epoch [1/50], Test Losses: mse: 0.6715, mae: 0.6086, huber: 0.2896, swd: 0.1729, ept: 118.1182
      Epoch 1 composite train-obj: 0.274054
            Val objective improved inf → 0.1980, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5927, mae: 0.5432, huber: 0.2454, swd: 0.2219, ept: 114.1740
    Epoch [2/50], Val Losses: mse: 0.4506, mae: 0.4713, huber: 0.1945, swd: 0.1202, ept: 109.1889
    Epoch [2/50], Test Losses: mse: 0.6569, mae: 0.5978, huber: 0.2832, swd: 0.1739, ept: 123.3613
      Epoch 2 composite train-obj: 0.245419
            Val objective improved 0.1980 → 0.1945, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5872, mae: 0.5380, huber: 0.2426, swd: 0.2207, ept: 121.4705
    Epoch [3/50], Val Losses: mse: 0.4310, mae: 0.4529, huber: 0.1856, swd: 0.1108, ept: 127.0877
    Epoch [3/50], Test Losses: mse: 0.6521, mae: 0.5951, huber: 0.2814, swd: 0.1762, ept: 121.7381
      Epoch 3 composite train-obj: 0.242617
            Val objective improved 0.1945 → 0.1856, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5847, mae: 0.5364, huber: 0.2418, swd: 0.2205, ept: 123.7384
    Epoch [4/50], Val Losses: mse: 0.4391, mae: 0.4590, huber: 0.1889, swd: 0.1166, ept: 124.1366
    Epoch [4/50], Test Losses: mse: 0.6376, mae: 0.5886, huber: 0.2759, swd: 0.1814, ept: 125.1260
      Epoch 4 composite train-obj: 0.241753
            No improvement (0.1889), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5831, mae: 0.5349, huber: 0.2409, swd: 0.2221, ept: 126.1177
    Epoch [5/50], Val Losses: mse: 0.4427, mae: 0.4621, huber: 0.1903, swd: 0.1182, ept: 109.6973
    Epoch [5/50], Test Losses: mse: 0.6400, mae: 0.5887, huber: 0.2766, swd: 0.1749, ept: 125.7223
      Epoch 5 composite train-obj: 0.240915
            No improvement (0.1903), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5836, mae: 0.5346, huber: 0.2409, swd: 0.2216, ept: 127.3854
    Epoch [6/50], Val Losses: mse: 0.4448, mae: 0.4633, huber: 0.1913, swd: 0.1226, ept: 111.9214
    Epoch [6/50], Test Losses: mse: 0.6431, mae: 0.5903, huber: 0.2779, swd: 0.1806, ept: 126.7690
      Epoch 6 composite train-obj: 0.240921
            No improvement (0.1913), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5800, mae: 0.5331, huber: 0.2398, swd: 0.2207, ept: 128.6620
    Epoch [7/50], Val Losses: mse: 0.4491, mae: 0.4704, huber: 0.1940, swd: 0.1308, ept: 104.0192
    Epoch [7/50], Test Losses: mse: 0.6448, mae: 0.5920, huber: 0.2788, swd: 0.1887, ept: 125.1955
      Epoch 7 composite train-obj: 0.239794
            No improvement (0.1940), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.5797, mae: 0.5326, huber: 0.2396, swd: 0.2214, ept: 128.9715
    Epoch [8/50], Val Losses: mse: 0.4384, mae: 0.4570, huber: 0.1883, swd: 0.1126, ept: 115.9613
    Epoch [8/50], Test Losses: mse: 0.6459, mae: 0.5886, huber: 0.2785, swd: 0.1664, ept: 126.2236
      Epoch 8 composite train-obj: 0.239578
    Epoch [8/50], Test Losses: mse: 0.6521, mae: 0.5951, huber: 0.2814, swd: 0.1762, ept: 121.7381
    Best round's Test MSE: 0.6521, MAE: 0.5951, SWD: 0.1762
    Best round's Validation MSE: 0.4310, MAE: 0.4529, SWD: 0.1108
    Best round's Test verification MSE : 0.6521, MAE: 0.5951, SWD: 0.1762
    Time taken: 7.89 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6642, mae: 0.5838, huber: 0.2730, swd: 0.2368, ept: 79.6770
    Epoch [1/50], Val Losses: mse: 0.4634, mae: 0.4851, huber: 0.2013, swd: 0.1339, ept: 101.3002
    Epoch [1/50], Test Losses: mse: 0.6603, mae: 0.6047, huber: 0.2853, swd: 0.1944, ept: 116.0136
      Epoch 1 composite train-obj: 0.272965
            Val objective improved inf → 0.2013, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5935, mae: 0.5431, huber: 0.2455, swd: 0.2362, ept: 114.1843
    Epoch [2/50], Val Losses: mse: 0.4496, mae: 0.4702, huber: 0.1942, swd: 0.1284, ept: 105.4448
    Epoch [2/50], Test Losses: mse: 0.6564, mae: 0.5982, huber: 0.2832, swd: 0.1828, ept: 122.9641
      Epoch 2 composite train-obj: 0.245503
            Val objective improved 0.2013 → 0.1942, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5856, mae: 0.5372, huber: 0.2421, swd: 0.2349, ept: 122.2259
    Epoch [3/50], Val Losses: mse: 0.4439, mae: 0.4611, huber: 0.1905, swd: 0.1222, ept: 121.8982
    Epoch [3/50], Test Losses: mse: 0.6494, mae: 0.5933, huber: 0.2802, swd: 0.1836, ept: 124.1674
      Epoch 3 composite train-obj: 0.242134
            Val objective improved 0.1942 → 0.1905, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5841, mae: 0.5357, huber: 0.2413, swd: 0.2358, ept: 124.6594
    Epoch [4/50], Val Losses: mse: 0.4481, mae: 0.4662, huber: 0.1927, swd: 0.1299, ept: 109.5522
    Epoch [4/50], Test Losses: mse: 0.6446, mae: 0.5900, huber: 0.2783, swd: 0.1855, ept: 125.7123
      Epoch 4 composite train-obj: 0.241344
            No improvement (0.1927), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5829, mae: 0.5348, huber: 0.2409, swd: 0.2361, ept: 126.8711
    Epoch [5/50], Val Losses: mse: 0.4389, mae: 0.4591, huber: 0.1887, swd: 0.1251, ept: 115.1641
    Epoch [5/50], Test Losses: mse: 0.6424, mae: 0.5903, huber: 0.2777, swd: 0.1880, ept: 126.7979
      Epoch 5 composite train-obj: 0.240870
            Val objective improved 0.1905 → 0.1887, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5815, mae: 0.5341, huber: 0.2404, swd: 0.2366, ept: 127.4104
    Epoch [6/50], Val Losses: mse: 0.4336, mae: 0.4507, huber: 0.1856, swd: 0.1197, ept: 134.5524
    Epoch [6/50], Test Losses: mse: 0.6398, mae: 0.5892, huber: 0.2767, swd: 0.1865, ept: 122.3414
      Epoch 6 composite train-obj: 0.240428
            Val objective improved 0.1887 → 0.1856, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.5802, mae: 0.5331, huber: 0.2399, swd: 0.2362, ept: 127.8177
    Epoch [7/50], Val Losses: mse: 0.4368, mae: 0.4572, huber: 0.1880, swd: 0.1295, ept: 120.0266
    Epoch [7/50], Test Losses: mse: 0.6410, mae: 0.5890, huber: 0.2772, swd: 0.1919, ept: 126.1723
      Epoch 7 composite train-obj: 0.239850
            No improvement (0.1880), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.5811, mae: 0.5331, huber: 0.2400, swd: 0.2367, ept: 128.2844
    Epoch [8/50], Val Losses: mse: 0.4417, mae: 0.4629, huber: 0.1905, swd: 0.1338, ept: 103.8718
    Epoch [8/50], Test Losses: mse: 0.6428, mae: 0.5900, huber: 0.2779, swd: 0.1927, ept: 126.9882
      Epoch 8 composite train-obj: 0.240005
            No improvement (0.1905), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.5795, mae: 0.5327, huber: 0.2396, swd: 0.2368, ept: 129.0241
    Epoch [9/50], Val Losses: mse: 0.4363, mae: 0.4553, huber: 0.1873, swd: 0.1245, ept: 117.4773
    Epoch [9/50], Test Losses: mse: 0.6486, mae: 0.5910, huber: 0.2797, swd: 0.1831, ept: 126.8578
      Epoch 9 composite train-obj: 0.239611
            No improvement (0.1873), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.5805, mae: 0.5328, huber: 0.2398, swd: 0.2376, ept: 129.2903
    Epoch [10/50], Val Losses: mse: 0.4362, mae: 0.4565, huber: 0.1878, swd: 0.1274, ept: 124.1421
    Epoch [10/50], Test Losses: mse: 0.6424, mae: 0.5892, huber: 0.2777, swd: 0.1912, ept: 125.2296
      Epoch 10 composite train-obj: 0.239787
            No improvement (0.1878), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.5792, mae: 0.5320, huber: 0.2393, swd: 0.2368, ept: 129.4777
    Epoch [11/50], Val Losses: mse: 0.4350, mae: 0.4544, huber: 0.1870, swd: 0.1268, ept: 125.8072
    Epoch [11/50], Test Losses: mse: 0.6430, mae: 0.5893, huber: 0.2778, swd: 0.1905, ept: 126.0288
      Epoch 11 composite train-obj: 0.239288
    Epoch [11/50], Test Losses: mse: 0.6398, mae: 0.5892, huber: 0.2767, swd: 0.1865, ept: 122.3414
    Best round's Test MSE: 0.6398, MAE: 0.5892, SWD: 0.1865
    Best round's Validation MSE: 0.4336, MAE: 0.4507, SWD: 0.1197
    Best round's Test verification MSE : 0.6398, MAE: 0.5892, SWD: 0.1865
    Time taken: 10.89 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq196_pred720_20250510_1848)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6476 ± 0.0055
      mae: 0.5920 ± 0.0024
      huber: 0.2796 ± 0.0020
      swd: 0.1788 ± 0.0055
      ept: 122.9534 ± 1.3154
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4313 ± 0.0017
      mae: 0.4508 ± 0.0017
      huber: 0.1851 ± 0.0007
      swd: 0.1144 ± 0.0038
      ept: 130.1412 ± 3.1951
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 31.17 seconds
    
    Experiment complete: DLinear_etth1_seq196_pred720_20250510_1848
    Model: DLinear
    Dataset: etth1
    Sequence Length: 196
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


```python

```






