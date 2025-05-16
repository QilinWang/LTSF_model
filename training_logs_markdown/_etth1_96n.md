# data

# ETTh1


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
    




    <data_manager.DatasetManager at 0x276f36a6750>



# Exp - 96

## Seq=96

### EigenACL

#### pred=96


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 94
    Validation Batches: 13
    Test Batches: 26
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5430, mae: 0.5188, huber: 0.2263, swd: 0.3071, ept: 38.7109
    Epoch [1/50], Val Losses: mse: 0.3945, mae: 0.4349, huber: 0.1685, swd: 0.1367, ept: 45.5324
    Epoch [1/50], Test Losses: mse: 0.4981, mae: 0.4950, huber: 0.2118, swd: 0.1926, ept: 34.6348
      Epoch 1 composite train-obj: 0.226291
            Val objective improved inf → 0.1685, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3728, mae: 0.4268, huber: 0.1631, swd: 0.1359, ept: 48.2055
    Epoch [2/50], Val Losses: mse: 0.3680, mae: 0.4135, huber: 0.1576, swd: 0.1118, ept: 48.1818
    Epoch [2/50], Test Losses: mse: 0.4448, mae: 0.4645, huber: 0.1920, swd: 0.1528, ept: 38.0060
      Epoch 2 composite train-obj: 0.163077
            Val objective improved 0.1685 → 0.1576, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3519, mae: 0.4106, huber: 0.1542, swd: 0.1205, ept: 50.6671
    Epoch [3/50], Val Losses: mse: 0.3615, mae: 0.4145, huber: 0.1558, swd: 0.1162, ept: 49.1656
    Epoch [3/50], Test Losses: mse: 0.4367, mae: 0.4604, huber: 0.1894, swd: 0.1599, ept: 38.6244
      Epoch 3 composite train-obj: 0.154192
            Val objective improved 0.1576 → 0.1558, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3426, mae: 0.4044, huber: 0.1505, swd: 0.1146, ept: 51.6874
    Epoch [4/50], Val Losses: mse: 0.3698, mae: 0.4172, huber: 0.1588, swd: 0.1232, ept: 49.5672
    Epoch [4/50], Test Losses: mse: 0.4330, mae: 0.4597, huber: 0.1885, swd: 0.1580, ept: 38.8135
      Epoch 4 composite train-obj: 0.150462
            No improvement (0.1588), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3366, mae: 0.4009, huber: 0.1483, swd: 0.1124, ept: 52.2053
    Epoch [5/50], Val Losses: mse: 0.3628, mae: 0.4126, huber: 0.1564, swd: 0.1286, ept: 50.0603
    Epoch [5/50], Test Losses: mse: 0.4361, mae: 0.4625, huber: 0.1904, swd: 0.1712, ept: 38.2624
      Epoch 5 composite train-obj: 0.148250
            No improvement (0.1564), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3311, mae: 0.3974, huber: 0.1461, swd: 0.1097, ept: 52.6954
    Epoch [6/50], Val Losses: mse: 0.3636, mae: 0.4098, huber: 0.1560, swd: 0.1172, ept: 50.2112
    Epoch [6/50], Test Losses: mse: 0.4297, mae: 0.4591, huber: 0.1880, swd: 0.1547, ept: 39.2740
      Epoch 6 composite train-obj: 0.146114
            No improvement (0.1560), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3257, mae: 0.3945, huber: 0.1442, swd: 0.1074, ept: 53.0312
    Epoch [7/50], Val Losses: mse: 0.3670, mae: 0.4166, huber: 0.1581, swd: 0.1220, ept: 50.4110
    Epoch [7/50], Test Losses: mse: 0.4278, mae: 0.4579, huber: 0.1871, swd: 0.1537, ept: 39.5580
      Epoch 7 composite train-obj: 0.144235
            No improvement (0.1581), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3221, mae: 0.3924, huber: 0.1429, swd: 0.1064, ept: 53.2103
    Epoch [8/50], Val Losses: mse: 0.3652, mae: 0.4195, huber: 0.1582, swd: 0.1292, ept: 50.4812
    Epoch [8/50], Test Losses: mse: 0.4277, mae: 0.4587, huber: 0.1871, swd: 0.1635, ept: 39.4663
      Epoch 8 composite train-obj: 0.142931
    Epoch [8/50], Test Losses: mse: 0.4367, mae: 0.4604, huber: 0.1894, swd: 0.1599, ept: 38.6267
    Best round's Test MSE: 0.4367, MAE: 0.4604, SWD: 0.1599
    Best round's Validation MSE: 0.3615, MAE: 0.4145, SWD: 0.1162
    Best round's Test verification MSE : 0.4367, MAE: 0.4604, SWD: 0.1599
    Time taken: 20.03 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5480, mae: 0.5208, huber: 0.2279, swd: 0.3007, ept: 38.5588
    Epoch [1/50], Val Losses: mse: 0.3943, mae: 0.4289, huber: 0.1679, swd: 0.1242, ept: 45.5209
    Epoch [1/50], Test Losses: mse: 0.5009, mae: 0.4935, huber: 0.2118, swd: 0.1734, ept: 34.5315
      Epoch 1 composite train-obj: 0.227904
            Val objective improved inf → 0.1679, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3738, mae: 0.4265, huber: 0.1633, swd: 0.1324, ept: 48.1679
    Epoch [2/50], Val Losses: mse: 0.3667, mae: 0.4088, huber: 0.1565, swd: 0.1106, ept: 48.3886
    Epoch [2/50], Test Losses: mse: 0.4527, mae: 0.4680, huber: 0.1947, swd: 0.1534, ept: 37.7543
      Epoch 2 composite train-obj: 0.163284
            Val objective improved 0.1679 → 0.1565, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3525, mae: 0.4108, huber: 0.1544, swd: 0.1189, ept: 50.4718
    Epoch [3/50], Val Losses: mse: 0.3610, mae: 0.4077, huber: 0.1547, swd: 0.1054, ept: 49.1528
    Epoch [3/50], Test Losses: mse: 0.4344, mae: 0.4580, huber: 0.1883, swd: 0.1445, ept: 38.6826
      Epoch 3 composite train-obj: 0.154354
            Val objective improved 0.1565 → 0.1547, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3435, mae: 0.4049, huber: 0.1508, swd: 0.1133, ept: 51.4250
    Epoch [4/50], Val Losses: mse: 0.3746, mae: 0.4242, huber: 0.1610, swd: 0.1152, ept: 49.3070
    Epoch [4/50], Test Losses: mse: 0.4297, mae: 0.4568, huber: 0.1867, swd: 0.1408, ept: 39.3452
      Epoch 4 composite train-obj: 0.150791
            No improvement (0.1610), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3368, mae: 0.4005, huber: 0.1481, swd: 0.1093, ept: 52.1599
    Epoch [5/50], Val Losses: mse: 0.3798, mae: 0.4264, huber: 0.1636, swd: 0.1288, ept: 49.1770
    Epoch [5/50], Test Losses: mse: 0.4389, mae: 0.4633, huber: 0.1912, swd: 0.1554, ept: 38.8015
      Epoch 5 composite train-obj: 0.148144
            No improvement (0.1636), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3309, mae: 0.3968, huber: 0.1459, swd: 0.1070, ept: 52.7123
    Epoch [6/50], Val Losses: mse: 0.3699, mae: 0.4233, huber: 0.1605, swd: 0.1327, ept: 49.5325
    Epoch [6/50], Test Losses: mse: 0.4340, mae: 0.4617, huber: 0.1897, swd: 0.1668, ept: 38.3081
      Epoch 6 composite train-obj: 0.145874
            No improvement (0.1605), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3265, mae: 0.3940, huber: 0.1441, swd: 0.1053, ept: 53.0533
    Epoch [7/50], Val Losses: mse: 0.3687, mae: 0.4149, huber: 0.1578, swd: 0.1077, ept: 50.7638
    Epoch [7/50], Test Losses: mse: 0.4232, mae: 0.4538, huber: 0.1843, swd: 0.1342, ept: 40.2887
      Epoch 7 composite train-obj: 0.144136
            No improvement (0.1578), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3213, mae: 0.3913, huber: 0.1423, swd: 0.1034, ept: 53.4510
    Epoch [8/50], Val Losses: mse: 0.3682, mae: 0.4209, huber: 0.1593, swd: 0.1223, ept: 50.9024
    Epoch [8/50], Test Losses: mse: 0.4304, mae: 0.4598, huber: 0.1877, swd: 0.1531, ept: 39.5182
      Epoch 8 composite train-obj: 0.142305
    Epoch [8/50], Test Losses: mse: 0.4344, mae: 0.4580, huber: 0.1883, swd: 0.1445, ept: 38.6810
    Best round's Test MSE: 0.4344, MAE: 0.4580, SWD: 0.1445
    Best round's Validation MSE: 0.3610, MAE: 0.4077, SWD: 0.1054
    Best round's Test verification MSE : 0.4344, MAE: 0.4580, SWD: 0.1445
    Time taken: 19.82 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5269, mae: 0.5105, huber: 0.2205, swd: 0.2713, ept: 39.6076
    Epoch [1/50], Val Losses: mse: 0.3915, mae: 0.4380, huber: 0.1687, swd: 0.1348, ept: 46.7015
    Epoch [1/50], Test Losses: mse: 0.4864, mae: 0.4884, huber: 0.2071, swd: 0.1712, ept: 35.5206
      Epoch 1 composite train-obj: 0.220516
            Val objective improved inf → 0.1687, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3700, mae: 0.4243, huber: 0.1617, swd: 0.1253, ept: 48.8879
    Epoch [2/50], Val Losses: mse: 0.3627, mae: 0.4129, huber: 0.1563, swd: 0.1183, ept: 49.0218
    Epoch [2/50], Test Losses: mse: 0.4469, mae: 0.4663, huber: 0.1931, swd: 0.1619, ept: 37.7003
      Epoch 2 composite train-obj: 0.161714
            Val objective improved 0.1687 → 0.1563, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3493, mae: 0.4086, huber: 0.1530, swd: 0.1123, ept: 51.0144
    Epoch [3/50], Val Losses: mse: 0.3573, mae: 0.4039, huber: 0.1535, swd: 0.1110, ept: 49.7826
    Epoch [3/50], Test Losses: mse: 0.4358, mae: 0.4605, huber: 0.1894, swd: 0.1587, ept: 39.0065
      Epoch 3 composite train-obj: 0.152970
            Val objective improved 0.1563 → 0.1535, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3402, mae: 0.4025, huber: 0.1493, swd: 0.1070, ept: 51.8503
    Epoch [4/50], Val Losses: mse: 0.3520, mae: 0.3997, huber: 0.1509, swd: 0.0979, ept: 50.0365
    Epoch [4/50], Test Losses: mse: 0.4231, mae: 0.4519, huber: 0.1841, swd: 0.1380, ept: 39.7342
      Epoch 4 composite train-obj: 0.149337
            Val objective improved 0.1535 → 0.1509, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3332, mae: 0.3979, huber: 0.1466, swd: 0.1029, ept: 52.5527
    Epoch [5/50], Val Losses: mse: 0.3506, mae: 0.3930, huber: 0.1493, swd: 0.0906, ept: 50.4840
    Epoch [5/50], Test Losses: mse: 0.4206, mae: 0.4509, huber: 0.1831, swd: 0.1300, ept: 40.4056
      Epoch 5 composite train-obj: 0.146589
            Val objective improved 0.1509 → 0.1493, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3293, mae: 0.3955, huber: 0.1451, swd: 0.1013, ept: 52.9800
    Epoch [6/50], Val Losses: mse: 0.3726, mae: 0.4244, huber: 0.1609, swd: 0.1272, ept: 50.0798
    Epoch [6/50], Test Losses: mse: 0.4320, mae: 0.4603, huber: 0.1886, swd: 0.1590, ept: 38.9416
      Epoch 6 composite train-obj: 0.145134
            No improvement (0.1609), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3228, mae: 0.3912, huber: 0.1425, swd: 0.0977, ept: 53.5376
    Epoch [7/50], Val Losses: mse: 0.3672, mae: 0.4229, huber: 0.1594, swd: 0.1251, ept: 50.2431
    Epoch [7/50], Test Losses: mse: 0.4290, mae: 0.4594, huber: 0.1876, swd: 0.1619, ept: 39.1941
      Epoch 7 composite train-obj: 0.142502
            No improvement (0.1594), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3178, mae: 0.3888, huber: 0.1408, swd: 0.0960, ept: 53.7227
    Epoch [8/50], Val Losses: mse: 0.3793, mae: 0.4257, huber: 0.1631, swd: 0.1222, ept: 50.2413
    Epoch [8/50], Test Losses: mse: 0.4305, mae: 0.4590, huber: 0.1876, swd: 0.1440, ept: 39.7801
      Epoch 8 composite train-obj: 0.140814
            No improvement (0.1631), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3122, mae: 0.3856, huber: 0.1388, swd: 0.0936, ept: 54.0665
    Epoch [9/50], Val Losses: mse: 0.3772, mae: 0.4184, huber: 0.1607, swd: 0.1156, ept: 50.7043
    Epoch [9/50], Test Losses: mse: 0.4331, mae: 0.4594, huber: 0.1882, swd: 0.1388, ept: 40.2435
      Epoch 9 composite train-obj: 0.138793
            No improvement (0.1607), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.3091, mae: 0.3839, huber: 0.1377, swd: 0.0931, ept: 54.1700
    Epoch [10/50], Val Losses: mse: 0.3805, mae: 0.4300, huber: 0.1641, swd: 0.1226, ept: 50.6825
    Epoch [10/50], Test Losses: mse: 0.4277, mae: 0.4590, huber: 0.1870, swd: 0.1419, ept: 39.8265
      Epoch 10 composite train-obj: 0.137655
    Epoch [10/50], Test Losses: mse: 0.4206, mae: 0.4509, huber: 0.1831, swd: 0.1300, ept: 40.3929
    Best round's Test MSE: 0.4206, MAE: 0.4509, SWD: 0.1300
    Best round's Validation MSE: 0.3506, MAE: 0.3930, SWD: 0.0906
    Best round's Test verification MSE : 0.4206, MAE: 0.4509, SWD: 0.1300
    Time taken: 25.58 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq96_pred96_20250510_1804)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4306 ± 0.0071
      mae: 0.4564 ± 0.0040
      huber: 0.1869 ± 0.0028
      swd: 0.1448 ± 0.0122
      ept: 39.2375 ± 0.8263
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3577 ± 0.0050
      mae: 0.4051 ± 0.0090
      huber: 0.1533 ± 0.0029
      swd: 0.1041 ± 0.0105
      ept: 49.6008 ± 0.6245
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 66.80 seconds
    
    Experiment complete: ACL_etth1_seq96_pred96_20250510_1804
    Model: ACL
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.6123, mae: 0.5512, huber: 0.2504, swd: 0.3503, ept: 56.0129
    Epoch [1/50], Val Losses: mse: 0.4477, mae: 0.4638, huber: 0.1894, swd: 0.1453, ept: 60.1436
    Epoch [1/50], Test Losses: mse: 0.5502, mae: 0.5277, huber: 0.2333, swd: 0.1798, ept: 52.7900
      Epoch 1 composite train-obj: 0.250379
            Val objective improved inf → 0.1894, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4556, mae: 0.4696, huber: 0.1933, swd: 0.1705, ept: 70.5279
    Epoch [2/50], Val Losses: mse: 0.4338, mae: 0.4598, huber: 0.1852, swd: 0.1422, ept: 61.1754
    Epoch [2/50], Test Losses: mse: 0.5009, mae: 0.5069, huber: 0.2172, swd: 0.1736, ept: 55.0849
      Epoch 2 composite train-obj: 0.193293
            Val objective improved 0.1894 → 0.1852, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4343, mae: 0.4563, huber: 0.1851, swd: 0.1563, ept: 73.6059
    Epoch [3/50], Val Losses: mse: 0.4253, mae: 0.4562, huber: 0.1833, swd: 0.1573, ept: 63.9526
    Epoch [3/50], Test Losses: mse: 0.4995, mae: 0.5074, huber: 0.2179, swd: 0.2022, ept: 55.6097
      Epoch 3 composite train-obj: 0.185126
            Val objective improved 0.1852 → 0.1833, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4222, mae: 0.4496, huber: 0.1808, swd: 0.1492, ept: 75.2873
    Epoch [4/50], Val Losses: mse: 0.4153, mae: 0.4557, huber: 0.1806, swd: 0.1510, ept: 63.3030
    Epoch [4/50], Test Losses: mse: 0.4902, mae: 0.5020, huber: 0.2136, swd: 0.2028, ept: 54.5844
      Epoch 4 composite train-obj: 0.180838
            Val objective improved 0.1833 → 0.1806, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4122, mae: 0.4445, huber: 0.1774, swd: 0.1458, ept: 76.5145
    Epoch [5/50], Val Losses: mse: 0.4376, mae: 0.4688, huber: 0.1894, swd: 0.1613, ept: 64.1583
    Epoch [5/50], Test Losses: mse: 0.4876, mae: 0.5039, huber: 0.2142, swd: 0.1953, ept: 54.5163
      Epoch 5 composite train-obj: 0.177421
            No improvement (0.1894), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4038, mae: 0.4407, huber: 0.1747, swd: 0.1421, ept: 77.1756
    Epoch [6/50], Val Losses: mse: 0.4415, mae: 0.4704, huber: 0.1909, swd: 0.1668, ept: 64.3425
    Epoch [6/50], Test Losses: mse: 0.4914, mae: 0.5058, huber: 0.2155, swd: 0.1938, ept: 55.3899
      Epoch 6 composite train-obj: 0.174685
            No improvement (0.1909), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.3964, mae: 0.4377, huber: 0.1724, swd: 0.1398, ept: 77.4539
    Epoch [7/50], Val Losses: mse: 0.4275, mae: 0.4615, huber: 0.1851, swd: 0.1503, ept: 64.8530
    Epoch [7/50], Test Losses: mse: 0.4820, mae: 0.4979, huber: 0.2107, swd: 0.1780, ept: 56.7232
      Epoch 7 composite train-obj: 0.172397
            No improvement (0.1851), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.3885, mae: 0.4333, huber: 0.1694, swd: 0.1334, ept: 78.2607
    Epoch [8/50], Val Losses: mse: 0.4323, mae: 0.4678, huber: 0.1881, swd: 0.1556, ept: 65.3498
    Epoch [8/50], Test Losses: mse: 0.4858, mae: 0.5025, huber: 0.2126, swd: 0.1842, ept: 56.0065
      Epoch 8 composite train-obj: 0.169428
            No improvement (0.1881), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.3818, mae: 0.4306, huber: 0.1672, swd: 0.1301, ept: 78.4112
    Epoch [9/50], Val Losses: mse: 0.4502, mae: 0.4750, huber: 0.1940, swd: 0.1569, ept: 65.6095
    Epoch [9/50], Test Losses: mse: 0.4849, mae: 0.5016, huber: 0.2127, swd: 0.1684, ept: 56.7469
      Epoch 9 composite train-obj: 0.167200
    Epoch [9/50], Test Losses: mse: 0.4902, mae: 0.5020, huber: 0.2136, swd: 0.2028, ept: 54.5844
    Best round's Test MSE: 0.4902, MAE: 0.5020, SWD: 0.2028
    Best round's Validation MSE: 0.4153, MAE: 0.4557, SWD: 0.1510
    Best round's Test verification MSE : 0.4902, MAE: 0.5020, SWD: 0.2028
    Time taken: 22.87 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6160, mae: 0.5527, huber: 0.2515, swd: 0.3461, ept: 55.5336
    Epoch [1/50], Val Losses: mse: 0.4595, mae: 0.4839, huber: 0.1968, swd: 0.1881, ept: 60.8736
    Epoch [1/50], Test Losses: mse: 0.5539, mae: 0.5317, huber: 0.2346, swd: 0.2202, ept: 50.2435
      Epoch 1 composite train-obj: 0.251522
            Val objective improved inf → 0.1968, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4526, mae: 0.4674, huber: 0.1921, swd: 0.1678, ept: 70.8564
    Epoch [2/50], Val Losses: mse: 0.4294, mae: 0.4472, huber: 0.1817, swd: 0.1250, ept: 63.3260
    Epoch [2/50], Test Losses: mse: 0.5030, mae: 0.5063, huber: 0.2175, swd: 0.1581, ept: 55.7510
      Epoch 2 composite train-obj: 0.192111
            Val objective improved 0.1968 → 0.1817, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4320, mae: 0.4552, huber: 0.1844, swd: 0.1550, ept: 73.6877
    Epoch [3/50], Val Losses: mse: 0.4251, mae: 0.4576, huber: 0.1834, swd: 0.1671, ept: 65.1372
    Epoch [3/50], Test Losses: mse: 0.5021, mae: 0.5092, huber: 0.2189, swd: 0.2113, ept: 54.4383
      Epoch 3 composite train-obj: 0.184423
            No improvement (0.1834), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4195, mae: 0.4482, huber: 0.1799, swd: 0.1480, ept: 75.4754
    Epoch [4/50], Val Losses: mse: 0.4310, mae: 0.4666, huber: 0.1871, swd: 0.1595, ept: 63.8980
    Epoch [4/50], Test Losses: mse: 0.4876, mae: 0.5020, huber: 0.2134, swd: 0.1911, ept: 53.9424
      Epoch 4 composite train-obj: 0.179902
            No improvement (0.1871), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4083, mae: 0.4426, huber: 0.1761, swd: 0.1438, ept: 76.7803
    Epoch [5/50], Val Losses: mse: 0.4224, mae: 0.4577, huber: 0.1828, swd: 0.1442, ept: 64.9197
    Epoch [5/50], Test Losses: mse: 0.4857, mae: 0.5000, huber: 0.2125, swd: 0.1848, ept: 56.3894
      Epoch 5 composite train-obj: 0.176119
            No improvement (0.1828), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4013, mae: 0.4393, huber: 0.1737, swd: 0.1402, ept: 77.2467
    Epoch [6/50], Val Losses: mse: 0.4181, mae: 0.4517, huber: 0.1802, swd: 0.1408, ept: 64.9117
    Epoch [6/50], Test Losses: mse: 0.4834, mae: 0.4977, huber: 0.2112, swd: 0.1662, ept: 56.6370
      Epoch 6 composite train-obj: 0.173747
            Val objective improved 0.1817 → 0.1802, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3928, mae: 0.4354, huber: 0.1708, swd: 0.1363, ept: 77.9588
    Epoch [7/50], Val Losses: mse: 0.4513, mae: 0.4795, huber: 0.1951, swd: 0.1643, ept: 63.7523
    Epoch [7/50], Test Losses: mse: 0.4882, mae: 0.5025, huber: 0.2141, swd: 0.1745, ept: 56.0006
      Epoch 7 composite train-obj: 0.170847
            No improvement (0.1951), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.3828, mae: 0.4304, huber: 0.1674, swd: 0.1300, ept: 78.6035
    Epoch [8/50], Val Losses: mse: 0.4423, mae: 0.4746, huber: 0.1920, swd: 0.1701, ept: 64.2345
    Epoch [8/50], Test Losses: mse: 0.4913, mae: 0.5042, huber: 0.2152, swd: 0.1866, ept: 54.9951
      Epoch 8 composite train-obj: 0.167401
            No improvement (0.1920), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.3751, mae: 0.4269, huber: 0.1648, swd: 0.1258, ept: 78.7650
    Epoch [9/50], Val Losses: mse: 0.4356, mae: 0.4670, huber: 0.1880, swd: 0.1509, ept: 65.0035
    Epoch [9/50], Test Losses: mse: 0.4911, mae: 0.5044, huber: 0.2151, swd: 0.1782, ept: 56.3595
      Epoch 9 composite train-obj: 0.164770
            No improvement (0.1880), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.3697, mae: 0.4245, huber: 0.1630, swd: 0.1217, ept: 78.8230
    Epoch [10/50], Val Losses: mse: 0.4486, mae: 0.4762, huber: 0.1934, swd: 0.1558, ept: 64.5395
    Epoch [10/50], Test Losses: mse: 0.4864, mae: 0.5021, huber: 0.2133, swd: 0.1619, ept: 55.9127
      Epoch 10 composite train-obj: 0.162964
            No improvement (0.1934), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.3736, mae: 0.4252, huber: 0.1637, swd: 0.1204, ept: 79.0673
    Epoch [11/50], Val Losses: mse: 0.4775, mae: 0.4999, huber: 0.2076, swd: 0.1805, ept: 61.5148
    Epoch [11/50], Test Losses: mse: 0.4981, mae: 0.5111, huber: 0.2188, swd: 0.1811, ept: 54.8810
      Epoch 11 composite train-obj: 0.163698
    Epoch [11/50], Test Losses: mse: 0.4834, mae: 0.4977, huber: 0.2112, swd: 0.1662, ept: 56.6310
    Best round's Test MSE: 0.4834, MAE: 0.4977, SWD: 0.1662
    Best round's Validation MSE: 0.4181, MAE: 0.4517, SWD: 0.1408
    Best round's Test verification MSE : 0.4834, MAE: 0.4977, SWD: 0.1662
    Time taken: 27.58 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5863, mae: 0.5385, huber: 0.2413, swd: 0.2954, ept: 58.1118
    Epoch [1/50], Val Losses: mse: 0.4414, mae: 0.4655, huber: 0.1881, swd: 0.1534, ept: 61.7125
    Epoch [1/50], Test Losses: mse: 0.5317, mae: 0.5198, huber: 0.2269, swd: 0.1775, ept: 53.7834
      Epoch 1 composite train-obj: 0.241315
            Val objective improved inf → 0.1881, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4481, mae: 0.4644, huber: 0.1903, swd: 0.1545, ept: 72.0754
    Epoch [2/50], Val Losses: mse: 0.4236, mae: 0.4465, huber: 0.1799, swd: 0.1330, ept: 64.3704
    Epoch [2/50], Test Losses: mse: 0.5030, mae: 0.5056, huber: 0.2174, swd: 0.1635, ept: 56.3538
      Epoch 2 composite train-obj: 0.190280
            Val objective improved 0.1881 → 0.1799, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4288, mae: 0.4516, huber: 0.1827, swd: 0.1413, ept: 75.2182
    Epoch [3/50], Val Losses: mse: 0.4154, mae: 0.4448, huber: 0.1776, swd: 0.1306, ept: 65.4688
    Epoch [3/50], Test Losses: mse: 0.4877, mae: 0.4982, huber: 0.2123, swd: 0.1678, ept: 57.5115
      Epoch 3 composite train-obj: 0.182682
            Val objective improved 0.1799 → 0.1776, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4161, mae: 0.4451, huber: 0.1783, swd: 0.1353, ept: 76.9317
    Epoch [4/50], Val Losses: mse: 0.4201, mae: 0.4526, huber: 0.1810, swd: 0.1413, ept: 66.2030
    Epoch [4/50], Test Losses: mse: 0.4824, mae: 0.4967, huber: 0.2110, swd: 0.1754, ept: 57.8451
      Epoch 4 composite train-obj: 0.178299
            No improvement (0.1810), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4053, mae: 0.4399, huber: 0.1747, swd: 0.1312, ept: 78.2392
    Epoch [5/50], Val Losses: mse: 0.4137, mae: 0.4508, huber: 0.1790, swd: 0.1343, ept: 66.5124
    Epoch [5/50], Test Losses: mse: 0.4809, mae: 0.4957, huber: 0.2099, swd: 0.1693, ept: 57.7565
      Epoch 5 composite train-obj: 0.174711
            No improvement (0.1790), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3954, mae: 0.4353, huber: 0.1715, swd: 0.1286, ept: 79.1257
    Epoch [6/50], Val Losses: mse: 0.4180, mae: 0.4515, huber: 0.1803, swd: 0.1353, ept: 67.1116
    Epoch [6/50], Test Losses: mse: 0.4847, mae: 0.4982, huber: 0.2120, swd: 0.1753, ept: 58.4574
      Epoch 6 composite train-obj: 0.171479
            No improvement (0.1803), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3871, mae: 0.4317, huber: 0.1687, swd: 0.1252, ept: 79.4534
    Epoch [7/50], Val Losses: mse: 0.4271, mae: 0.4539, huber: 0.1830, swd: 0.1269, ept: 67.4500
    Epoch [7/50], Test Losses: mse: 0.4779, mae: 0.4955, huber: 0.2093, swd: 0.1475, ept: 59.0343
      Epoch 7 composite train-obj: 0.168704
            No improvement (0.1830), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3783, mae: 0.4279, huber: 0.1658, swd: 0.1205, ept: 79.9250
    Epoch [8/50], Val Losses: mse: 0.4465, mae: 0.4759, huber: 0.1938, swd: 0.1515, ept: 65.3926
    Epoch [8/50], Test Losses: mse: 0.4845, mae: 0.5010, huber: 0.2125, swd: 0.1666, ept: 56.8574
      Epoch 8 composite train-obj: 0.165808
    Epoch [8/50], Test Losses: mse: 0.4877, mae: 0.4982, huber: 0.2123, swd: 0.1678, ept: 57.5208
    Best round's Test MSE: 0.4877, MAE: 0.4982, SWD: 0.1678
    Best round's Validation MSE: 0.4154, MAE: 0.4448, SWD: 0.1306
    Best round's Test verification MSE : 0.4877, MAE: 0.4982, SWD: 0.1678
    Time taken: 20.10 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq96_pred196_20250510_1811)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4871 ± 0.0028
      mae: 0.4993 ± 0.0019
      huber: 0.2123 ± 0.0010
      swd: 0.1789 ± 0.0169
      ept: 56.2443 ± 1.2268
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4163 ± 0.0013
      mae: 0.4507 ± 0.0045
      huber: 0.1795 ± 0.0013
      swd: 0.1408 ± 0.0083
      ept: 64.5612 ± 0.9183
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 70.63 seconds
    
    Experiment complete: ACL_etth1_seq96_pred196_20250510_1811
    Model: ACL
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 11
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6492, mae: 0.5702, huber: 0.2644, swd: 0.3483, ept: 71.0219
    Epoch [1/50], Val Losses: mse: 0.4945, mae: 0.4929, huber: 0.2089, swd: 0.1842, ept: 78.2190
    Epoch [1/50], Test Losses: mse: 0.5878, mae: 0.5574, huber: 0.2518, swd: 0.2253, ept: 72.3665
      Epoch 1 composite train-obj: 0.264436
            Val objective improved inf → 0.2089, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5136, mae: 0.5009, huber: 0.2154, swd: 0.1934, ept: 88.9537
    Epoch [2/50], Val Losses: mse: 0.4871, mae: 0.4865, huber: 0.2044, swd: 0.1669, ept: 78.8400
    Epoch [2/50], Test Losses: mse: 0.5528, mae: 0.5433, huber: 0.2407, swd: 0.2049, ept: 71.5068
      Epoch 2 composite train-obj: 0.215379
            Val objective improved 0.2089 → 0.2044, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4906, mae: 0.4872, huber: 0.2066, swd: 0.1769, ept: 93.4779
    Epoch [3/50], Val Losses: mse: 0.4648, mae: 0.4785, huber: 0.1980, swd: 0.1488, ept: 76.5700
    Epoch [3/50], Test Losses: mse: 0.5382, mae: 0.5346, huber: 0.2350, swd: 0.1986, ept: 73.4301
      Epoch 3 composite train-obj: 0.206627
            Val objective improved 0.2044 → 0.1980, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4758, mae: 0.4793, huber: 0.2012, swd: 0.1689, ept: 96.5566
    Epoch [4/50], Val Losses: mse: 0.4870, mae: 0.4935, huber: 0.2072, swd: 0.1540, ept: 76.8012
    Epoch [4/50], Test Losses: mse: 0.5404, mae: 0.5391, huber: 0.2369, swd: 0.1978, ept: 73.0895
      Epoch 4 composite train-obj: 0.201206
            No improvement (0.2072), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4644, mae: 0.4737, huber: 0.1973, swd: 0.1640, ept: 98.7605
    Epoch [5/50], Val Losses: mse: 0.4986, mae: 0.4983, huber: 0.2101, swd: 0.1446, ept: 79.2157
    Epoch [5/50], Test Losses: mse: 0.5302, mae: 0.5332, huber: 0.2327, swd: 0.1690, ept: 72.8873
      Epoch 5 composite train-obj: 0.197253
            No improvement (0.2101), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4572, mae: 0.4710, huber: 0.1950, swd: 0.1605, ept: 98.8312
    Epoch [6/50], Val Losses: mse: 0.4880, mae: 0.4967, huber: 0.2085, swd: 0.1679, ept: 78.7792
    Epoch [6/50], Test Losses: mse: 0.5321, mae: 0.5348, huber: 0.2338, swd: 0.1957, ept: 70.9593
      Epoch 6 composite train-obj: 0.195000
            No improvement (0.2085), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.4482, mae: 0.4664, huber: 0.1916, swd: 0.1547, ept: 99.5292
    Epoch [7/50], Val Losses: mse: 0.5006, mae: 0.5114, huber: 0.2161, swd: 0.1842, ept: 75.0875
    Epoch [7/50], Test Losses: mse: 0.5313, mae: 0.5364, huber: 0.2342, swd: 0.2030, ept: 69.1924
      Epoch 7 composite train-obj: 0.191642
            No improvement (0.2161), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4403, mae: 0.4617, huber: 0.1885, swd: 0.1502, ept: 100.8516
    Epoch [8/50], Val Losses: mse: 0.5076, mae: 0.5132, huber: 0.2172, swd: 0.1770, ept: 76.5583
    Epoch [8/50], Test Losses: mse: 0.5324, mae: 0.5385, huber: 0.2350, swd: 0.1936, ept: 69.4348
      Epoch 8 composite train-obj: 0.188461
    Epoch [8/50], Test Losses: mse: 0.5382, mae: 0.5346, huber: 0.2350, swd: 0.1986, ept: 73.4324
    Best round's Test MSE: 0.5382, MAE: 0.5346, SWD: 0.1986
    Best round's Validation MSE: 0.4648, MAE: 0.4785, SWD: 0.1488
    Best round's Test verification MSE : 0.5382, MAE: 0.5346, SWD: 0.1986
    Time taken: 20.30 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6567, mae: 0.5734, huber: 0.2668, swd: 0.3612, ept: 70.0692
    Epoch [1/50], Val Losses: mse: 0.4860, mae: 0.4857, huber: 0.2045, swd: 0.1637, ept: 79.2569
    Epoch [1/50], Test Losses: mse: 0.5852, mae: 0.5528, huber: 0.2496, swd: 0.2039, ept: 73.0882
      Epoch 1 composite train-obj: 0.266842
            Val objective improved inf → 0.2045, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5160, mae: 0.5018, huber: 0.2162, swd: 0.1970, ept: 88.9615
    Epoch [2/50], Val Losses: mse: 0.4558, mae: 0.4575, huber: 0.1906, swd: 0.1375, ept: 82.8491
    Epoch [2/50], Test Losses: mse: 0.5438, mae: 0.5357, huber: 0.2365, swd: 0.1921, ept: 73.8357
      Epoch 2 composite train-obj: 0.216211
            Val objective improved 0.2045 → 0.1906, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4927, mae: 0.4881, huber: 0.2075, swd: 0.1806, ept: 93.2962
    Epoch [3/50], Val Losses: mse: 0.4697, mae: 0.4693, huber: 0.1965, swd: 0.1252, ept: 81.3744
    Epoch [3/50], Test Losses: mse: 0.5265, mae: 0.5280, huber: 0.2306, swd: 0.1678, ept: 75.6298
      Epoch 3 composite train-obj: 0.207457
            No improvement (0.1965), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4787, mae: 0.4808, huber: 0.2023, swd: 0.1740, ept: 95.7153
    Epoch [4/50], Val Losses: mse: 0.4743, mae: 0.4803, huber: 0.2002, swd: 0.1412, ept: 82.1142
    Epoch [4/50], Test Losses: mse: 0.5235, mae: 0.5289, huber: 0.2303, swd: 0.1824, ept: 75.7068
      Epoch 4 composite train-obj: 0.202339
            No improvement (0.2002), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4672, mae: 0.4749, huber: 0.1981, swd: 0.1688, ept: 97.0862
    Epoch [5/50], Val Losses: mse: 0.4867, mae: 0.4901, huber: 0.2053, swd: 0.1463, ept: 79.8418
    Epoch [5/50], Test Losses: mse: 0.5176, mae: 0.5277, huber: 0.2282, swd: 0.1776, ept: 73.5736
      Epoch 5 composite train-obj: 0.198097
            No improvement (0.2053), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4566, mae: 0.4697, huber: 0.1943, swd: 0.1622, ept: 98.4216
    Epoch [6/50], Val Losses: mse: 0.4743, mae: 0.4943, huber: 0.2044, swd: 0.1679, ept: 76.6568
    Epoch [6/50], Test Losses: mse: 0.5306, mae: 0.5338, huber: 0.2326, swd: 0.2097, ept: 70.0157
      Epoch 6 composite train-obj: 0.194321
            No improvement (0.2044), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4466, mae: 0.4650, huber: 0.1908, swd: 0.1570, ept: 99.1659
    Epoch [7/50], Val Losses: mse: 0.4897, mae: 0.5019, huber: 0.2104, swd: 0.1791, ept: 78.9650
    Epoch [7/50], Test Losses: mse: 0.5320, mae: 0.5370, huber: 0.2342, swd: 0.2151, ept: 71.6315
      Epoch 7 composite train-obj: 0.190806
    Epoch [7/50], Test Losses: mse: 0.5438, mae: 0.5357, huber: 0.2365, swd: 0.1921, ept: 73.8779
    Best round's Test MSE: 0.5438, MAE: 0.5357, SWD: 0.1921
    Best round's Validation MSE: 0.4558, MAE: 0.4575, SWD: 0.1375
    Best round's Test verification MSE : 0.5438, MAE: 0.5357, SWD: 0.1921
    Time taken: 17.75 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6564, mae: 0.5727, huber: 0.2663, swd: 0.3575, ept: 70.0483
    Epoch [1/50], Val Losses: mse: 0.4879, mae: 0.4901, huber: 0.2059, swd: 0.1807, ept: 79.8210
    Epoch [1/50], Test Losses: mse: 0.5857, mae: 0.5534, huber: 0.2496, swd: 0.2173, ept: 71.6498
      Epoch 1 composite train-obj: 0.266338
            Val objective improved inf → 0.2059, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5131, mae: 0.5000, huber: 0.2150, swd: 0.1946, ept: 90.0970
    Epoch [2/50], Val Losses: mse: 0.4667, mae: 0.4765, huber: 0.1972, swd: 0.1582, ept: 79.6748
    Epoch [2/50], Test Losses: mse: 0.5439, mae: 0.5354, huber: 0.2362, swd: 0.2013, ept: 73.4577
      Epoch 2 composite train-obj: 0.215036
            Val objective improved 0.2059 → 0.1972, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4886, mae: 0.4852, huber: 0.2057, swd: 0.1789, ept: 94.6422
    Epoch [3/50], Val Losses: mse: 0.4808, mae: 0.4745, huber: 0.1999, swd: 0.1365, ept: 83.1610
    Epoch [3/50], Test Losses: mse: 0.5310, mae: 0.5300, huber: 0.2322, swd: 0.1736, ept: 77.2770
      Epoch 3 composite train-obj: 0.205701
            No improvement (0.1999), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4738, mae: 0.4775, huber: 0.2003, swd: 0.1707, ept: 97.3184
    Epoch [4/50], Val Losses: mse: 0.4982, mae: 0.4989, huber: 0.2105, swd: 0.1650, ept: 79.8097
    Epoch [4/50], Test Losses: mse: 0.5261, mae: 0.5324, huber: 0.2318, swd: 0.1907, ept: 73.7187
      Epoch 4 composite train-obj: 0.200312
            No improvement (0.2105), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4608, mae: 0.4714, huber: 0.1959, swd: 0.1640, ept: 99.2030
    Epoch [5/50], Val Losses: mse: 0.5213, mae: 0.5142, huber: 0.2196, swd: 0.1713, ept: 76.4057
    Epoch [5/50], Test Losses: mse: 0.5249, mae: 0.5331, huber: 0.2315, swd: 0.1826, ept: 70.5259
      Epoch 5 composite train-obj: 0.195874
            No improvement (0.2196), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4501, mae: 0.4663, huber: 0.1921, swd: 0.1583, ept: 100.5798
    Epoch [6/50], Val Losses: mse: 0.4827, mae: 0.4940, huber: 0.2063, swd: 0.1608, ept: 81.8750
    Epoch [6/50], Test Losses: mse: 0.5294, mae: 0.5329, huber: 0.2326, swd: 0.1992, ept: 74.1178
      Epoch 6 composite train-obj: 0.192091
            No improvement (0.2063), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4394, mae: 0.4610, huber: 0.1882, swd: 0.1523, ept: 101.6140
    Epoch [7/50], Val Losses: mse: 0.5068, mae: 0.5099, huber: 0.2158, swd: 0.1684, ept: 80.4360
    Epoch [7/50], Test Losses: mse: 0.5368, mae: 0.5400, huber: 0.2365, swd: 0.1920, ept: 73.4825
      Epoch 7 composite train-obj: 0.188174
    Epoch [7/50], Test Losses: mse: 0.5439, mae: 0.5354, huber: 0.2362, swd: 0.2013, ept: 73.5214
    Best round's Test MSE: 0.5439, MAE: 0.5354, SWD: 0.2013
    Best round's Validation MSE: 0.4667, MAE: 0.4765, SWD: 0.1582
    Best round's Test verification MSE : 0.5439, MAE: 0.5354, SWD: 0.2013
    Time taken: 17.51 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq96_pred336_20250510_1820)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5420 ± 0.0027
      mae: 0.5352 ± 0.0005
      huber: 0.2359 ± 0.0006
      swd: 0.1973 ± 0.0039
      ept: 73.5745 ± 0.1850
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4624 ± 0.0048
      mae: 0.4708 ± 0.0094
      huber: 0.1953 ± 0.0033
      swd: 0.1482 ± 0.0085
      ept: 79.6980 ± 2.5635
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 55.64 seconds
    
    Experiment complete: ACL_etth1_seq96_pred336_20250510_1820
    Model: ACL
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 8
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7263, mae: 0.6115, huber: 0.2942, swd: 0.4091, ept: 83.5211
    Epoch [1/50], Val Losses: mse: 0.5130, mae: 0.5238, huber: 0.2230, swd: 0.2168, ept: 86.1736
    Epoch [1/50], Test Losses: mse: 0.7016, mae: 0.6220, huber: 0.2979, swd: 0.2658, ept: 105.5157
      Epoch 1 composite train-obj: 0.294182
            Val objective improved inf → 0.2230, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5922, mae: 0.5459, huber: 0.2463, swd: 0.2495, ept: 108.9773
    Epoch [2/50], Val Losses: mse: 0.5083, mae: 0.5201, huber: 0.2212, swd: 0.1900, ept: 89.8777
    Epoch [2/50], Test Losses: mse: 0.6626, mae: 0.6093, huber: 0.2870, swd: 0.2546, ept: 105.4373
      Epoch 2 composite train-obj: 0.246256
            Val objective improved 0.2230 → 0.2212, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5656, mae: 0.5316, huber: 0.2363, swd: 0.2309, ept: 115.1264
    Epoch [3/50], Val Losses: mse: 0.5436, mae: 0.5458, huber: 0.2369, swd: 0.1957, ept: 69.3737
    Epoch [3/50], Test Losses: mse: 0.6434, mae: 0.6018, huber: 0.2807, swd: 0.2363, ept: 91.2877
      Epoch 3 composite train-obj: 0.236261
            No improvement (0.2369), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5473, mae: 0.5224, huber: 0.2293, swd: 0.2184, ept: 117.8731
    Epoch [4/50], Val Losses: mse: 0.5738, mae: 0.5760, huber: 0.2522, swd: 0.1996, ept: 72.6595
    Epoch [4/50], Test Losses: mse: 0.6744, mae: 0.6220, huber: 0.2939, swd: 0.2644, ept: 96.7414
      Epoch 4 composite train-obj: 0.229348
            No improvement (0.2522), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5343, mae: 0.5160, huber: 0.2244, swd: 0.2095, ept: 118.6174
    Epoch [5/50], Val Losses: mse: 0.5703, mae: 0.5658, huber: 0.2488, swd: 0.1880, ept: 74.5543
    Epoch [5/50], Test Losses: mse: 0.6554, mae: 0.6092, huber: 0.2854, swd: 0.2290, ept: 95.1940
      Epoch 5 composite train-obj: 0.224433
            No improvement (0.2488), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.5192, mae: 0.5080, huber: 0.2186, swd: 0.1996, ept: 119.5665
    Epoch [6/50], Val Losses: mse: 0.5605, mae: 0.5619, huber: 0.2440, swd: 0.1694, ept: 75.3906
    Epoch [6/50], Test Losses: mse: 0.6565, mae: 0.6107, huber: 0.2860, swd: 0.2212, ept: 91.1543
      Epoch 6 composite train-obj: 0.218618
            No improvement (0.2440), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.5087, mae: 0.5019, huber: 0.2145, swd: 0.1919, ept: 121.4744
    Epoch [7/50], Val Losses: mse: 0.5926, mae: 0.5834, huber: 0.2579, swd: 0.1814, ept: 73.9975
    Epoch [7/50], Test Losses: mse: 0.6737, mae: 0.6215, huber: 0.2934, swd: 0.2168, ept: 91.2175
      Epoch 7 composite train-obj: 0.214487
    Epoch [7/50], Test Losses: mse: 0.6626, mae: 0.6093, huber: 0.2870, swd: 0.2546, ept: 105.2781
    Best round's Test MSE: 0.6626, MAE: 0.6093, SWD: 0.2546
    Best round's Validation MSE: 0.5083, MAE: 0.5201, SWD: 0.1900
    Best round's Test verification MSE : 0.6626, MAE: 0.6093, SWD: 0.2546
    Time taken: 17.96 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7376, mae: 0.6156, huber: 0.2975, swd: 0.4152, ept: 82.3694
    Epoch [1/50], Val Losses: mse: 0.5239, mae: 0.5313, huber: 0.2283, swd: 0.2060, ept: 81.5454
    Epoch [1/50], Test Losses: mse: 0.7013, mae: 0.6221, huber: 0.2977, swd: 0.2516, ept: 99.7272
      Epoch 1 composite train-obj: 0.297474
            Val objective improved inf → 0.2283, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5979, mae: 0.5491, huber: 0.2485, swd: 0.2471, ept: 107.6749
    Epoch [2/50], Val Losses: mse: 0.5099, mae: 0.5241, huber: 0.2228, swd: 0.1909, ept: 79.7688
    Epoch [2/50], Test Losses: mse: 0.6707, mae: 0.6138, huber: 0.2900, swd: 0.2579, ept: 97.6709
      Epoch 2 composite train-obj: 0.248471
            Val objective improved 0.2283 → 0.2228, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5686, mae: 0.5334, huber: 0.2375, swd: 0.2286, ept: 114.5663
    Epoch [3/50], Val Losses: mse: 0.5307, mae: 0.5398, huber: 0.2324, swd: 0.1851, ept: 74.1804
    Epoch [3/50], Test Losses: mse: 0.6575, mae: 0.6089, huber: 0.2864, swd: 0.2411, ept: 92.3609
      Epoch 3 composite train-obj: 0.237535
            No improvement (0.2324), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5496, mae: 0.5236, huber: 0.2302, swd: 0.2158, ept: 118.0467
    Epoch [4/50], Val Losses: mse: 0.5240, mae: 0.5361, huber: 0.2285, swd: 0.1568, ept: 77.9241
    Epoch [4/50], Test Losses: mse: 0.6360, mae: 0.5986, huber: 0.2777, swd: 0.2202, ept: 94.3470
      Epoch 4 composite train-obj: 0.230229
            No improvement (0.2285), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5340, mae: 0.5156, huber: 0.2243, swd: 0.2042, ept: 119.7834
    Epoch [5/50], Val Losses: mse: 0.5743, mae: 0.5669, huber: 0.2505, swd: 0.1799, ept: 74.1591
    Epoch [5/50], Test Losses: mse: 0.6634, mae: 0.6152, huber: 0.2895, swd: 0.2202, ept: 93.3856
      Epoch 5 composite train-obj: 0.224315
            No improvement (0.2505), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.5189, mae: 0.5077, huber: 0.2185, swd: 0.1951, ept: 120.8315
    Epoch [6/50], Val Losses: mse: 0.5674, mae: 0.5587, huber: 0.2435, swd: 0.1513, ept: 81.5934
    Epoch [6/50], Test Losses: mse: 0.6560, mae: 0.6082, huber: 0.2852, swd: 0.2029, ept: 98.9323
      Epoch 6 composite train-obj: 0.218493
            No improvement (0.2435), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.5070, mae: 0.5012, huber: 0.2139, swd: 0.1870, ept: 123.1748
    Epoch [7/50], Val Losses: mse: 0.5580, mae: 0.5581, huber: 0.2418, swd: 0.1463, ept: 83.5172
    Epoch [7/50], Test Losses: mse: 0.6621, mae: 0.6106, huber: 0.2875, swd: 0.2018, ept: 99.0600
      Epoch 7 composite train-obj: 0.213949
    Epoch [7/50], Test Losses: mse: 0.6707, mae: 0.6138, huber: 0.2900, swd: 0.2579, ept: 97.7445
    Best round's Test MSE: 0.6707, MAE: 0.6138, SWD: 0.2579
    Best round's Validation MSE: 0.5099, MAE: 0.5241, SWD: 0.1909
    Best round's Test verification MSE : 0.6707, MAE: 0.6138, SWD: 0.2579
    Time taken: 17.97 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7175, mae: 0.6069, huber: 0.2908, swd: 0.4102, ept: 84.4367
    Epoch [1/50], Val Losses: mse: 0.5080, mae: 0.5182, huber: 0.2207, swd: 0.1983, ept: 90.2260
    Epoch [1/50], Test Losses: mse: 0.6897, mae: 0.6170, huber: 0.2938, swd: 0.2452, ept: 105.4640
      Epoch 1 composite train-obj: 0.290813
            Val objective improved inf → 0.2207, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5916, mae: 0.5456, huber: 0.2461, swd: 0.2580, ept: 110.1126
    Epoch [2/50], Val Losses: mse: 0.5147, mae: 0.5225, huber: 0.2229, swd: 0.1931, ept: 82.4515
    Epoch [2/50], Test Losses: mse: 0.6539, mae: 0.6047, huber: 0.2834, swd: 0.2457, ept: 103.6263
      Epoch 2 composite train-obj: 0.246063
            No improvement (0.2229), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5664, mae: 0.5317, huber: 0.2364, swd: 0.2403, ept: 115.2676
    Epoch [3/50], Val Losses: mse: 0.5232, mae: 0.5294, huber: 0.2265, swd: 0.1789, ept: 86.3752
    Epoch [3/50], Test Losses: mse: 0.6521, mae: 0.6040, huber: 0.2835, swd: 0.2321, ept: 102.6441
      Epoch 3 composite train-obj: 0.236404
            No improvement (0.2265), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.5480, mae: 0.5220, huber: 0.2293, swd: 0.2273, ept: 119.0974
    Epoch [4/50], Val Losses: mse: 0.5208, mae: 0.5406, huber: 0.2294, swd: 0.1789, ept: 75.0074
    Epoch [4/50], Test Losses: mse: 0.6457, mae: 0.6051, huber: 0.2819, swd: 0.2488, ept: 94.5010
      Epoch 4 composite train-obj: 0.229319
            No improvement (0.2294), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.5328, mae: 0.5148, huber: 0.2237, swd: 0.2157, ept: 120.1372
    Epoch [5/50], Val Losses: mse: 0.5539, mae: 0.5512, huber: 0.2396, swd: 0.1780, ept: 80.8449
    Epoch [5/50], Test Losses: mse: 0.6482, mae: 0.6044, huber: 0.2823, swd: 0.2193, ept: 98.0957
      Epoch 5 composite train-obj: 0.223696
            No improvement (0.2396), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.5186, mae: 0.5069, huber: 0.2182, swd: 0.2057, ept: 121.7941
    Epoch [6/50], Val Losses: mse: 0.5387, mae: 0.5549, huber: 0.2376, swd: 0.1785, ept: 79.0842
    Epoch [6/50], Test Losses: mse: 0.6568, mae: 0.6101, huber: 0.2861, swd: 0.2422, ept: 97.5851
      Epoch 6 composite train-obj: 0.218164
    Epoch [6/50], Test Losses: mse: 0.6897, mae: 0.6170, huber: 0.2938, swd: 0.2452, ept: 105.4700
    Best round's Test MSE: 0.6897, MAE: 0.6170, SWD: 0.2452
    Best round's Validation MSE: 0.5080, MAE: 0.5182, SWD: 0.1983
    Best round's Test verification MSE : 0.6897, MAE: 0.6170, SWD: 0.2452
    Time taken: 14.37 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq96_pred720_20250510_1819)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6743 ± 0.0114
      mae: 0.6133 ± 0.0032
      huber: 0.2903 ± 0.0028
      swd: 0.2526 ± 0.0054
      ept: 102.8574 ± 3.6675
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.5087 ± 0.0009
      mae: 0.5208 ± 0.0024
      huber: 0.2216 ± 0.0009
      swd: 0.1931 ± 0.0037
      ept: 86.6241 ± 4.8496
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 50.37 seconds
    
    Experiment complete: ACL_etth1_seq96_pred720_20250510_1819
    Model: ACL
    Dataset: etth1
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 94
    Validation Batches: 13
    Test Batches: 26
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5123, mae: 0.4984, huber: 0.2143, swd: 0.1343, ept: 39.1119
    Epoch [1/50], Val Losses: mse: 0.3577, mae: 0.4015, huber: 0.1518, swd: 0.1104, ept: 50.2412
    Epoch [1/50], Test Losses: mse: 0.4697, mae: 0.4739, huber: 0.1992, swd: 0.1551, ept: 38.8892
      Epoch 1 composite train-obj: 0.214290
            Val objective improved inf → 0.1518, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3923, mae: 0.4268, huber: 0.1673, swd: 0.1212, ept: 49.3536
    Epoch [2/50], Val Losses: mse: 0.3519, mae: 0.3926, huber: 0.1486, swd: 0.1109, ept: 51.5142
    Epoch [2/50], Test Losses: mse: 0.4592, mae: 0.4638, huber: 0.1944, swd: 0.1534, ept: 40.2190
      Epoch 2 composite train-obj: 0.167285
            Val objective improved 0.1518 → 0.1486, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3807, mae: 0.4170, huber: 0.1621, swd: 0.1191, ept: 50.7861
    Epoch [3/50], Val Losses: mse: 0.3519, mae: 0.3877, huber: 0.1472, swd: 0.1015, ept: 52.3540
    Epoch [3/50], Test Losses: mse: 0.4510, mae: 0.4527, huber: 0.1890, swd: 0.1344, ept: 41.6930
      Epoch 3 composite train-obj: 0.162132
            Val objective improved 0.1486 → 0.1472, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3743, mae: 0.4121, huber: 0.1594, swd: 0.1169, ept: 51.4911
    Epoch [4/50], Val Losses: mse: 0.3483, mae: 0.3854, huber: 0.1459, swd: 0.1000, ept: 52.5526
    Epoch [4/50], Test Losses: mse: 0.4463, mae: 0.4507, huber: 0.1876, swd: 0.1335, ept: 41.9788
      Epoch 4 composite train-obj: 0.159434
            Val objective improved 0.1472 → 0.1459, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3700, mae: 0.4094, huber: 0.1578, swd: 0.1156, ept: 51.7949
    Epoch [5/50], Val Losses: mse: 0.3409, mae: 0.3822, huber: 0.1439, swd: 0.1056, ept: 52.6842
    Epoch [5/50], Test Losses: mse: 0.4445, mae: 0.4541, huber: 0.1886, swd: 0.1477, ept: 41.7488
      Epoch 5 composite train-obj: 0.157780
            Val objective improved 0.1459 → 0.1439, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3660, mae: 0.4070, huber: 0.1563, swd: 0.1139, ept: 52.1378
    Epoch [6/50], Val Losses: mse: 0.3483, mae: 0.3863, huber: 0.1467, swd: 0.1081, ept: 52.7061
    Epoch [6/50], Test Losses: mse: 0.4481, mae: 0.4556, huber: 0.1897, swd: 0.1412, ept: 41.9440
      Epoch 6 composite train-obj: 0.156289
            No improvement (0.1467), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3625, mae: 0.4049, huber: 0.1550, swd: 0.1125, ept: 52.4602
    Epoch [7/50], Val Losses: mse: 0.3420, mae: 0.3847, huber: 0.1450, swd: 0.1089, ept: 52.6463
    Epoch [7/50], Test Losses: mse: 0.4456, mae: 0.4574, huber: 0.1899, swd: 0.1477, ept: 41.6792
      Epoch 7 composite train-obj: 0.154955
            No improvement (0.1450), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3596, mae: 0.4036, huber: 0.1540, swd: 0.1116, ept: 52.5881
    Epoch [8/50], Val Losses: mse: 0.3435, mae: 0.3852, huber: 0.1453, swd: 0.1000, ept: 52.7915
    Epoch [8/50], Test Losses: mse: 0.4371, mae: 0.4483, huber: 0.1854, swd: 0.1306, ept: 42.3564
      Epoch 8 composite train-obj: 0.153959
            No improvement (0.1453), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3563, mae: 0.4016, huber: 0.1527, swd: 0.1098, ept: 52.9579
    Epoch [9/50], Val Losses: mse: 0.3407, mae: 0.3830, huber: 0.1441, swd: 0.1003, ept: 52.9062
    Epoch [9/50], Test Losses: mse: 0.4344, mae: 0.4458, huber: 0.1842, swd: 0.1336, ept: 42.4178
      Epoch 9 composite train-obj: 0.152698
            No improvement (0.1441), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.3538, mae: 0.4003, huber: 0.1518, swd: 0.1090, ept: 53.1108
    Epoch [10/50], Val Losses: mse: 0.3424, mae: 0.3848, huber: 0.1449, swd: 0.0983, ept: 52.9259
    Epoch [10/50], Test Losses: mse: 0.4327, mae: 0.4449, huber: 0.1836, swd: 0.1279, ept: 42.5147
      Epoch 10 composite train-obj: 0.151762
    Epoch [10/50], Test Losses: mse: 0.4445, mae: 0.4541, huber: 0.1886, swd: 0.1477, ept: 41.7488
    Best round's Test MSE: 0.4445, MAE: 0.4541, SWD: 0.1477
    Best round's Validation MSE: 0.3409, MAE: 0.3822, SWD: 0.1056
    Best round's Test verification MSE : 0.4445, MAE: 0.4541, SWD: 0.1477
    Time taken: 23.28 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5344, mae: 0.5086, huber: 0.2217, swd: 0.1475, ept: 39.0332
    Epoch [1/50], Val Losses: mse: 0.3674, mae: 0.4089, huber: 0.1562, swd: 0.1085, ept: 48.6265
    Epoch [1/50], Test Losses: mse: 0.4762, mae: 0.4774, huber: 0.2016, swd: 0.1415, ept: 37.6098
      Epoch 1 composite train-obj: 0.221657
            Val objective improved inf → 0.1562, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3942, mae: 0.4293, huber: 0.1684, swd: 0.1203, ept: 49.1457
    Epoch [2/50], Val Losses: mse: 0.3562, mae: 0.3963, huber: 0.1502, swd: 0.1000, ept: 51.1011
    Epoch [2/50], Test Losses: mse: 0.4562, mae: 0.4609, huber: 0.1923, swd: 0.1344, ept: 40.2751
      Epoch 2 composite train-obj: 0.168360
            Val objective improved 0.1562 → 0.1502, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3818, mae: 0.4189, huber: 0.1628, swd: 0.1172, ept: 50.7248
    Epoch [3/50], Val Losses: mse: 0.3566, mae: 0.3922, huber: 0.1496, swd: 0.1006, ept: 51.9907
    Epoch [3/50], Test Losses: mse: 0.4555, mae: 0.4568, huber: 0.1910, swd: 0.1323, ept: 41.0081
      Epoch 3 composite train-obj: 0.162829
            Val objective improved 0.1502 → 0.1496, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3761, mae: 0.4141, huber: 0.1604, swd: 0.1155, ept: 51.4610
    Epoch [4/50], Val Losses: mse: 0.3462, mae: 0.3843, huber: 0.1453, swd: 0.0994, ept: 52.4954
    Epoch [4/50], Test Losses: mse: 0.4495, mae: 0.4545, huber: 0.1893, swd: 0.1372, ept: 41.5909
      Epoch 4 composite train-obj: 0.160354
            Val objective improved 0.1496 → 0.1453, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3717, mae: 0.4106, huber: 0.1585, swd: 0.1142, ept: 51.9434
    Epoch [5/50], Val Losses: mse: 0.3481, mae: 0.3837, huber: 0.1457, swd: 0.1012, ept: 52.7532
    Epoch [5/50], Test Losses: mse: 0.4488, mae: 0.4532, huber: 0.1888, swd: 0.1364, ept: 41.8949
      Epoch 5 composite train-obj: 0.158526
            No improvement (0.1457), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3687, mae: 0.4085, huber: 0.1573, swd: 0.1131, ept: 52.1477
    Epoch [6/50], Val Losses: mse: 0.3455, mae: 0.3828, huber: 0.1449, swd: 0.0997, ept: 52.8466
    Epoch [6/50], Test Losses: mse: 0.4447, mae: 0.4506, huber: 0.1873, swd: 0.1354, ept: 42.1233
      Epoch 6 composite train-obj: 0.157263
            Val objective improved 0.1453 → 0.1449, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3653, mae: 0.4066, huber: 0.1560, swd: 0.1119, ept: 52.3858
    Epoch [7/50], Val Losses: mse: 0.3423, mae: 0.3818, huber: 0.1441, swd: 0.0994, ept: 52.8348
    Epoch [7/50], Test Losses: mse: 0.4427, mae: 0.4508, huber: 0.1871, swd: 0.1382, ept: 41.9497
      Epoch 7 composite train-obj: 0.156041
            Val objective improved 0.1449 → 0.1441, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3625, mae: 0.4051, huber: 0.1550, swd: 0.1112, ept: 52.5343
    Epoch [8/50], Val Losses: mse: 0.3446, mae: 0.3831, huber: 0.1448, swd: 0.0980, ept: 52.8970
    Epoch [8/50], Test Losses: mse: 0.4420, mae: 0.4489, huber: 0.1863, swd: 0.1342, ept: 42.2101
      Epoch 8 composite train-obj: 0.155030
            No improvement (0.1448), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3602, mae: 0.4041, huber: 0.1543, swd: 0.1109, ept: 52.6097
    Epoch [9/50], Val Losses: mse: 0.3462, mae: 0.3848, huber: 0.1459, swd: 0.1039, ept: 52.8717
    Epoch [9/50], Test Losses: mse: 0.4453, mae: 0.4527, huber: 0.1882, swd: 0.1406, ept: 41.8408
      Epoch 9 composite train-obj: 0.154266
            No improvement (0.1459), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.3577, mae: 0.4026, huber: 0.1533, swd: 0.1099, ept: 52.8942
    Epoch [10/50], Val Losses: mse: 0.3428, mae: 0.3825, huber: 0.1445, swd: 0.0986, ept: 53.1939
    Epoch [10/50], Test Losses: mse: 0.4384, mae: 0.4464, huber: 0.1849, swd: 0.1330, ept: 42.5343
      Epoch 10 composite train-obj: 0.153273
            No improvement (0.1445), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.3553, mae: 0.4011, huber: 0.1523, swd: 0.1092, ept: 53.0636
    Epoch [11/50], Val Losses: mse: 0.3431, mae: 0.3863, huber: 0.1451, swd: 0.0940, ept: 52.8546
    Epoch [11/50], Test Losses: mse: 0.4375, mae: 0.4481, huber: 0.1849, swd: 0.1294, ept: 42.0702
      Epoch 11 composite train-obj: 0.152313
            No improvement (0.1451), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.3540, mae: 0.4007, huber: 0.1519, swd: 0.1084, ept: 53.2169
    Epoch [12/50], Val Losses: mse: 0.3402, mae: 0.3825, huber: 0.1440, swd: 0.0988, ept: 52.9868
    Epoch [12/50], Test Losses: mse: 0.4358, mae: 0.4473, huber: 0.1849, swd: 0.1370, ept: 42.2182
      Epoch 12 composite train-obj: 0.151901
            Val objective improved 0.1441 → 0.1440, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.3519, mae: 0.3994, huber: 0.1511, swd: 0.1078, ept: 53.3423
    Epoch [13/50], Val Losses: mse: 0.3434, mae: 0.3846, huber: 0.1451, swd: 0.0986, ept: 52.8615
    Epoch [13/50], Test Losses: mse: 0.4353, mae: 0.4462, huber: 0.1843, swd: 0.1322, ept: 42.3552
      Epoch 13 composite train-obj: 0.151080
            No improvement (0.1451), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.3497, mae: 0.3982, huber: 0.1502, swd: 0.1066, ept: 53.4960
    Epoch [14/50], Val Losses: mse: 0.3469, mae: 0.3860, huber: 0.1464, swd: 0.1001, ept: 52.7782
    Epoch [14/50], Test Losses: mse: 0.4386, mae: 0.4493, huber: 0.1858, swd: 0.1330, ept: 42.0437
      Epoch 14 composite train-obj: 0.150210
            No improvement (0.1464), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.3486, mae: 0.3980, huber: 0.1499, swd: 0.1065, ept: 53.6504
    Epoch [15/50], Val Losses: mse: 0.3415, mae: 0.3829, huber: 0.1445, swd: 0.1004, ept: 53.0468
    Epoch [15/50], Test Losses: mse: 0.4347, mae: 0.4476, huber: 0.1846, swd: 0.1352, ept: 42.2588
      Epoch 15 composite train-obj: 0.149902
            No improvement (0.1445), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.3470, mae: 0.3969, huber: 0.1493, swd: 0.1056, ept: 53.6940
    Epoch [16/50], Val Losses: mse: 0.3448, mae: 0.3886, huber: 0.1466, swd: 0.0945, ept: 52.4203
    Epoch [16/50], Test Losses: mse: 0.4341, mae: 0.4486, huber: 0.1847, swd: 0.1274, ept: 42.0414
      Epoch 16 composite train-obj: 0.149268
            No improvement (0.1466), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.3446, mae: 0.3957, huber: 0.1484, swd: 0.1047, ept: 53.8175
    Epoch [17/50], Val Losses: mse: 0.3427, mae: 0.3844, huber: 0.1452, swd: 0.1000, ept: 52.7129
    Epoch [17/50], Test Losses: mse: 0.4337, mae: 0.4470, huber: 0.1843, swd: 0.1307, ept: 42.2802
      Epoch 17 composite train-obj: 0.148384
    Epoch [17/50], Test Losses: mse: 0.4358, mae: 0.4473, huber: 0.1849, swd: 0.1370, ept: 42.2182
    Best round's Test MSE: 0.4358, MAE: 0.4473, SWD: 0.1370
    Best round's Validation MSE: 0.3402, MAE: 0.3825, SWD: 0.0988
    Best round's Test verification MSE : 0.4358, MAE: 0.4473, SWD: 0.1370
    Time taken: 39.30 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5818, mae: 0.5299, huber: 0.2390, swd: 0.1431, ept: 37.7318
    Epoch [1/50], Val Losses: mse: 0.3647, mae: 0.4058, huber: 0.1547, swd: 0.1072, ept: 49.3533
    Epoch [1/50], Test Losses: mse: 0.4716, mae: 0.4741, huber: 0.1997, swd: 0.1386, ept: 38.3979
      Epoch 1 composite train-obj: 0.238980
            Val objective improved inf → 0.1547, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3922, mae: 0.4274, huber: 0.1675, swd: 0.1138, ept: 49.4399
    Epoch [2/50], Val Losses: mse: 0.3530, mae: 0.3939, huber: 0.1489, swd: 0.0994, ept: 51.3717
    Epoch [2/50], Test Losses: mse: 0.4547, mae: 0.4610, huber: 0.1921, swd: 0.1317, ept: 40.4128
      Epoch 2 composite train-obj: 0.167470
            Val objective improved 0.1547 → 0.1489, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3802, mae: 0.4174, huber: 0.1621, swd: 0.1111, ept: 50.9437
    Epoch [3/50], Val Losses: mse: 0.3488, mae: 0.3870, huber: 0.1466, swd: 0.0998, ept: 52.2449
    Epoch [3/50], Test Losses: mse: 0.4489, mae: 0.4553, huber: 0.1894, swd: 0.1328, ept: 41.4358
      Epoch 3 composite train-obj: 0.162098
            Val objective improved 0.1489 → 0.1466, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3744, mae: 0.4128, huber: 0.1596, swd: 0.1097, ept: 51.6891
    Epoch [4/50], Val Losses: mse: 0.3486, mae: 0.3870, huber: 0.1462, swd: 0.0963, ept: 52.4020
    Epoch [4/50], Test Losses: mse: 0.4469, mae: 0.4533, huber: 0.1881, swd: 0.1271, ept: 41.6746
      Epoch 4 composite train-obj: 0.159625
            Val objective improved 0.1466 → 0.1462, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3709, mae: 0.4102, huber: 0.1582, swd: 0.1085, ept: 51.9719
    Epoch [5/50], Val Losses: mse: 0.3472, mae: 0.3841, huber: 0.1456, swd: 0.1028, ept: 52.9781
    Epoch [5/50], Test Losses: mse: 0.4465, mae: 0.4526, huber: 0.1882, swd: 0.1345, ept: 42.0433
      Epoch 5 composite train-obj: 0.158224
            Val objective improved 0.1462 → 0.1456, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3682, mae: 0.4081, huber: 0.1571, swd: 0.1077, ept: 52.2411
    Epoch [6/50], Val Losses: mse: 0.3442, mae: 0.3823, huber: 0.1445, swd: 0.1004, ept: 53.0611
    Epoch [6/50], Test Losses: mse: 0.4409, mae: 0.4487, huber: 0.1860, swd: 0.1315, ept: 42.0330
      Epoch 6 composite train-obj: 0.157094
            Val objective improved 0.1456 → 0.1445, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3656, mae: 0.4065, huber: 0.1561, swd: 0.1069, ept: 52.5026
    Epoch [7/50], Val Losses: mse: 0.3420, mae: 0.3830, huber: 0.1441, swd: 0.0950, ept: 53.1024
    Epoch [7/50], Test Losses: mse: 0.4379, mae: 0.4473, huber: 0.1848, swd: 0.1272, ept: 42.3018
      Epoch 7 composite train-obj: 0.156101
            Val objective improved 0.1445 → 0.1441, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3631, mae: 0.4051, huber: 0.1552, swd: 0.1062, ept: 52.7057
    Epoch [8/50], Val Losses: mse: 0.3434, mae: 0.3821, huber: 0.1444, swd: 0.1007, ept: 53.2685
    Epoch [8/50], Test Losses: mse: 0.4388, mae: 0.4471, huber: 0.1850, swd: 0.1306, ept: 42.3881
      Epoch 8 composite train-obj: 0.155205
            No improvement (0.1444), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3615, mae: 0.4040, huber: 0.1545, swd: 0.1055, ept: 52.9363
    Epoch [9/50], Val Losses: mse: 0.3462, mae: 0.3853, huber: 0.1459, swd: 0.0995, ept: 53.0936
    Epoch [9/50], Test Losses: mse: 0.4372, mae: 0.4458, huber: 0.1843, swd: 0.1262, ept: 42.4147
      Epoch 9 composite train-obj: 0.154508
            No improvement (0.1459), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.3597, mae: 0.4030, huber: 0.1539, swd: 0.1049, ept: 53.0713
    Epoch [10/50], Val Losses: mse: 0.3414, mae: 0.3841, huber: 0.1445, swd: 0.0971, ept: 52.9097
    Epoch [10/50], Test Losses: mse: 0.4341, mae: 0.4464, huber: 0.1839, swd: 0.1266, ept: 42.2786
      Epoch 10 composite train-obj: 0.153873
            No improvement (0.1445), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.3580, mae: 0.4023, huber: 0.1533, swd: 0.1043, ept: 53.1719
    Epoch [11/50], Val Losses: mse: 0.3496, mae: 0.3880, huber: 0.1474, swd: 0.1062, ept: 52.9837
    Epoch [11/50], Test Losses: mse: 0.4454, mae: 0.4543, huber: 0.1885, swd: 0.1316, ept: 42.0514
      Epoch 11 composite train-obj: 0.153300
            No improvement (0.1474), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.3562, mae: 0.4013, huber: 0.1526, swd: 0.1036, ept: 53.3678
    Epoch [12/50], Val Losses: mse: 0.3367, mae: 0.3806, huber: 0.1426, swd: 0.0984, ept: 53.2256
    Epoch [12/50], Test Losses: mse: 0.4305, mae: 0.4453, huber: 0.1830, swd: 0.1286, ept: 42.4315
      Epoch 12 composite train-obj: 0.152612
            Val objective improved 0.1441 → 0.1426, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.3550, mae: 0.4007, huber: 0.1522, swd: 0.1029, ept: 53.4804
    Epoch [13/50], Val Losses: mse: 0.3460, mae: 0.3855, huber: 0.1461, swd: 0.1014, ept: 53.0251
    Epoch [13/50], Test Losses: mse: 0.4394, mae: 0.4488, huber: 0.1857, swd: 0.1261, ept: 42.4001
      Epoch 13 composite train-obj: 0.152154
            No improvement (0.1461), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.3531, mae: 0.3999, huber: 0.1515, swd: 0.1024, ept: 53.5708
    Epoch [14/50], Val Losses: mse: 0.3446, mae: 0.3853, huber: 0.1458, swd: 0.1040, ept: 53.0129
    Epoch [14/50], Test Losses: mse: 0.4403, mae: 0.4518, huber: 0.1868, swd: 0.1292, ept: 42.2388
      Epoch 14 composite train-obj: 0.151524
            No improvement (0.1458), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.3515, mae: 0.3990, huber: 0.1509, swd: 0.1015, ept: 53.6434
    Epoch [15/50], Val Losses: mse: 0.3370, mae: 0.3819, huber: 0.1431, swd: 0.1005, ept: 52.9653
    Epoch [15/50], Test Losses: mse: 0.4323, mae: 0.4478, huber: 0.1842, swd: 0.1278, ept: 42.4556
      Epoch 15 composite train-obj: 0.150909
            No improvement (0.1431), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.3499, mae: 0.3983, huber: 0.1503, swd: 0.1008, ept: 53.7763
    Epoch [16/50], Val Losses: mse: 0.3417, mae: 0.3850, huber: 0.1451, swd: 0.0968, ept: 52.7648
    Epoch [16/50], Test Losses: mse: 0.4333, mae: 0.4462, huber: 0.1838, swd: 0.1221, ept: 42.6149
      Epoch 16 composite train-obj: 0.150318
            No improvement (0.1451), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.3487, mae: 0.3978, huber: 0.1499, swd: 0.1004, ept: 53.8050
    Epoch [17/50], Val Losses: mse: 0.3376, mae: 0.3841, huber: 0.1440, swd: 0.1007, ept: 52.9120
    Epoch [17/50], Test Losses: mse: 0.4306, mae: 0.4482, huber: 0.1842, swd: 0.1262, ept: 42.4087
      Epoch 17 composite train-obj: 0.149914
    Epoch [17/50], Test Losses: mse: 0.4305, mae: 0.4453, huber: 0.1830, swd: 0.1286, ept: 42.4315
    Best round's Test MSE: 0.4305, MAE: 0.4453, SWD: 0.1286
    Best round's Validation MSE: 0.3367, MAE: 0.3806, SWD: 0.0984
    Best round's Test verification MSE : 0.4305, MAE: 0.4453, SWD: 0.1286
    Time taken: 39.44 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq96_pred96_20250510_1805)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4369 ± 0.0058
      mae: 0.4489 ± 0.0038
      huber: 0.1855 ± 0.0023
      swd: 0.1378 ± 0.0078
      ept: 42.1328 ± 0.2852
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3393 ± 0.0019
      mae: 0.3817 ± 0.0008
      huber: 0.1435 ± 0.0006
      swd: 0.1010 ± 0.0033
      ept: 52.9655 ± 0.2215
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 102.07 seconds
    
    Experiment complete: TimeMixer_etth1_seq96_pred96_20250510_1805
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.6395, mae: 0.5525, huber: 0.2552, swd: 0.1713, ept: 54.3233
    Epoch [1/50], Val Losses: mse: 0.4246, mae: 0.4412, huber: 0.1776, swd: 0.1234, ept: 66.2616
    Epoch [1/50], Test Losses: mse: 0.5356, mae: 0.5141, huber: 0.2263, swd: 0.1477, ept: 56.4750
      Epoch 1 composite train-obj: 0.255195
            Val objective improved inf → 0.1776, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4997, mae: 0.4784, huber: 0.2043, swd: 0.1531, ept: 71.2127
    Epoch [2/50], Val Losses: mse: 0.4096, mae: 0.4275, huber: 0.1700, swd: 0.1150, ept: 69.6805
    Epoch [2/50], Test Losses: mse: 0.5161, mae: 0.4988, huber: 0.2173, swd: 0.1390, ept: 59.5819
      Epoch 2 composite train-obj: 0.204320
            Val objective improved 0.1776 → 0.1700, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4880, mae: 0.4690, huber: 0.1992, swd: 0.1516, ept: 74.1122
    Epoch [3/50], Val Losses: mse: 0.3960, mae: 0.4184, huber: 0.1647, swd: 0.1190, ept: 71.5691
    Epoch [3/50], Test Losses: mse: 0.5127, mae: 0.4989, huber: 0.2170, swd: 0.1568, ept: 60.3902
      Epoch 3 composite train-obj: 0.199234
            Val objective improved 0.1700 → 0.1647, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4818, mae: 0.4642, huber: 0.1966, swd: 0.1504, ept: 75.4291
    Epoch [4/50], Val Losses: mse: 0.3961, mae: 0.4155, huber: 0.1639, swd: 0.1139, ept: 72.4153
    Epoch [4/50], Test Losses: mse: 0.5077, mae: 0.4928, huber: 0.2141, swd: 0.1424, ept: 61.2209
      Epoch 4 composite train-obj: 0.196631
            Val objective improved 0.1647 → 0.1639, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4779, mae: 0.4612, huber: 0.1950, swd: 0.1493, ept: 76.0297
    Epoch [5/50], Val Losses: mse: 0.3962, mae: 0.4167, huber: 0.1638, swd: 0.1079, ept: 72.4458
    Epoch [5/50], Test Losses: mse: 0.5057, mae: 0.4917, huber: 0.2131, swd: 0.1329, ept: 61.9462
      Epoch 5 composite train-obj: 0.194961
            Val objective improved 0.1639 → 0.1638, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4740, mae: 0.4589, huber: 0.1936, swd: 0.1485, ept: 76.3804
    Epoch [6/50], Val Losses: mse: 0.3907, mae: 0.4136, huber: 0.1627, swd: 0.1193, ept: 73.0486
    Epoch [6/50], Test Losses: mse: 0.5062, mae: 0.4946, huber: 0.2146, swd: 0.1507, ept: 61.2153
      Epoch 6 composite train-obj: 0.193555
            Val objective improved 0.1638 → 0.1627, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4707, mae: 0.4575, huber: 0.1925, swd: 0.1477, ept: 76.8677
    Epoch [7/50], Val Losses: mse: 0.3975, mae: 0.4160, huber: 0.1646, swd: 0.1148, ept: 72.3353
    Epoch [7/50], Test Losses: mse: 0.5031, mae: 0.4887, huber: 0.2119, swd: 0.1352, ept: 62.0420
      Epoch 7 composite train-obj: 0.192460
            No improvement (0.1646), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4673, mae: 0.4556, huber: 0.1912, swd: 0.1466, ept: 77.3031
    Epoch [8/50], Val Losses: mse: 0.3868, mae: 0.4139, huber: 0.1618, swd: 0.1119, ept: 72.7981
    Epoch [8/50], Test Losses: mse: 0.4973, mae: 0.4893, huber: 0.2111, swd: 0.1388, ept: 62.2986
      Epoch 8 composite train-obj: 0.191215
            Val objective improved 0.1627 → 0.1618, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4641, mae: 0.4544, huber: 0.1902, swd: 0.1462, ept: 77.6186
    Epoch [9/50], Val Losses: mse: 0.3923, mae: 0.4157, huber: 0.1638, swd: 0.1136, ept: 72.7250
    Epoch [9/50], Test Losses: mse: 0.4994, mae: 0.4886, huber: 0.2113, swd: 0.1340, ept: 62.3814
      Epoch 9 composite train-obj: 0.190202
            No improvement (0.1638), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.4612, mae: 0.4533, huber: 0.1893, swd: 0.1449, ept: 77.9416
    Epoch [10/50], Val Losses: mse: 0.3850, mae: 0.4136, huber: 0.1618, swd: 0.1140, ept: 72.9900
    Epoch [10/50], Test Losses: mse: 0.4961, mae: 0.4906, huber: 0.2114, swd: 0.1433, ept: 62.5764
      Epoch 10 composite train-obj: 0.189273
            Val objective improved 0.1618 → 0.1618, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.4580, mae: 0.4518, huber: 0.1882, swd: 0.1439, ept: 78.1981
    Epoch [11/50], Val Losses: mse: 0.3836, mae: 0.4136, huber: 0.1613, swd: 0.1115, ept: 72.7593
    Epoch [11/50], Test Losses: mse: 0.4958, mae: 0.4892, huber: 0.2111, swd: 0.1377, ept: 62.1920
      Epoch 11 composite train-obj: 0.188205
            Val objective improved 0.1618 → 0.1613, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.4564, mae: 0.4512, huber: 0.1877, swd: 0.1435, ept: 78.3298
    Epoch [12/50], Val Losses: mse: 0.3804, mae: 0.4132, huber: 0.1606, swd: 0.1110, ept: 73.0565
    Epoch [12/50], Test Losses: mse: 0.4942, mae: 0.4896, huber: 0.2110, swd: 0.1375, ept: 62.2167
      Epoch 12 composite train-obj: 0.187656
            Val objective improved 0.1613 → 0.1606, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.4532, mae: 0.4501, huber: 0.1866, swd: 0.1422, ept: 78.7068
    Epoch [13/50], Val Losses: mse: 0.3819, mae: 0.4143, huber: 0.1614, swd: 0.1108, ept: 72.9972
    Epoch [13/50], Test Losses: mse: 0.4945, mae: 0.4905, huber: 0.2113, swd: 0.1364, ept: 62.3394
      Epoch 13 composite train-obj: 0.186628
            No improvement (0.1614), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.4513, mae: 0.4494, huber: 0.1860, swd: 0.1414, ept: 78.8646
    Epoch [14/50], Val Losses: mse: 0.3851, mae: 0.4157, huber: 0.1624, swd: 0.1119, ept: 73.0225
    Epoch [14/50], Test Losses: mse: 0.4974, mae: 0.4905, huber: 0.2120, swd: 0.1331, ept: 62.1243
      Epoch 14 composite train-obj: 0.185992
            No improvement (0.1624), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.4490, mae: 0.4484, huber: 0.1852, swd: 0.1405, ept: 79.1398
    Epoch [15/50], Val Losses: mse: 0.3819, mae: 0.4147, huber: 0.1617, swd: 0.1154, ept: 73.8399
    Epoch [15/50], Test Losses: mse: 0.4990, mae: 0.4954, huber: 0.2139, swd: 0.1427, ept: 61.7607
      Epoch 15 composite train-obj: 0.185183
            No improvement (0.1617), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.4463, mae: 0.4475, huber: 0.1844, swd: 0.1394, ept: 79.1526
    Epoch [16/50], Val Losses: mse: 0.3845, mae: 0.4175, huber: 0.1627, swd: 0.1101, ept: 72.7184
    Epoch [16/50], Test Losses: mse: 0.4953, mae: 0.4906, huber: 0.2117, swd: 0.1275, ept: 62.5257
      Epoch 16 composite train-obj: 0.184378
            No improvement (0.1627), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.4438, mae: 0.4464, huber: 0.1835, swd: 0.1382, ept: 79.3778
    Epoch [17/50], Val Losses: mse: 0.3889, mae: 0.4193, huber: 0.1643, swd: 0.1092, ept: 72.0387
    Epoch [17/50], Test Losses: mse: 0.4958, mae: 0.4899, huber: 0.2116, swd: 0.1241, ept: 62.4201
      Epoch 17 composite train-obj: 0.183486
    Epoch [17/50], Test Losses: mse: 0.4942, mae: 0.4896, huber: 0.2110, swd: 0.1375, ept: 62.2167
    Best round's Test MSE: 0.4942, MAE: 0.4896, SWD: 0.1375
    Best round's Validation MSE: 0.3804, MAE: 0.4132, SWD: 0.1110
    Best round's Test verification MSE : 0.4942, MAE: 0.4896, SWD: 0.1375
    Time taken: 41.19 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6138, mae: 0.5431, huber: 0.2485, swd: 0.1749, ept: 56.8309
    Epoch [1/50], Val Losses: mse: 0.4127, mae: 0.4353, huber: 0.1729, swd: 0.1167, ept: 67.3701
    Epoch [1/50], Test Losses: mse: 0.5291, mae: 0.5120, huber: 0.2242, swd: 0.1472, ept: 56.8600
      Epoch 1 composite train-obj: 0.248493
            Val objective improved inf → 0.1729, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4948, mae: 0.4749, huber: 0.2023, swd: 0.1562, ept: 72.3456
    Epoch [2/50], Val Losses: mse: 0.4045, mae: 0.4254, huber: 0.1683, swd: 0.1126, ept: 70.3200
    Epoch [2/50], Test Losses: mse: 0.5162, mae: 0.5006, huber: 0.2179, swd: 0.1422, ept: 59.7135
      Epoch 2 composite train-obj: 0.202336
            Val objective improved 0.1729 → 0.1683, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4844, mae: 0.4662, huber: 0.1977, swd: 0.1540, ept: 75.0974
    Epoch [3/50], Val Losses: mse: 0.4013, mae: 0.4192, huber: 0.1659, swd: 0.1103, ept: 71.6372
    Epoch [3/50], Test Losses: mse: 0.5085, mae: 0.4930, huber: 0.2141, swd: 0.1373, ept: 61.0196
      Epoch 3 composite train-obj: 0.197714
            Val objective improved 0.1683 → 0.1659, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4786, mae: 0.4616, huber: 0.1953, swd: 0.1519, ept: 76.3928
    Epoch [4/50], Val Losses: mse: 0.3968, mae: 0.4189, huber: 0.1646, swd: 0.1025, ept: 71.5750
    Epoch [4/50], Test Losses: mse: 0.5059, mae: 0.4923, huber: 0.2133, swd: 0.1283, ept: 61.9411
      Epoch 4 composite train-obj: 0.195255
            Val objective improved 0.1659 → 0.1646, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4748, mae: 0.4592, huber: 0.1938, swd: 0.1507, ept: 76.9140
    Epoch [5/50], Val Losses: mse: 0.3917, mae: 0.4155, huber: 0.1634, swd: 0.1147, ept: 72.9326
    Epoch [5/50], Test Losses: mse: 0.5036, mae: 0.4935, huber: 0.2136, swd: 0.1490, ept: 61.3113
      Epoch 5 composite train-obj: 0.193838
            Val objective improved 0.1646 → 0.1634, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4701, mae: 0.4567, huber: 0.1922, swd: 0.1494, ept: 77.5755
    Epoch [6/50], Val Losses: mse: 0.3935, mae: 0.4157, huber: 0.1638, swd: 0.1130, ept: 73.2001
    Epoch [6/50], Test Losses: mse: 0.5005, mae: 0.4900, huber: 0.2119, swd: 0.1392, ept: 61.6549
      Epoch 6 composite train-obj: 0.192155
            No improvement (0.1638), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4669, mae: 0.4555, huber: 0.1912, swd: 0.1486, ept: 77.8486
    Epoch [7/50], Val Losses: mse: 0.3844, mae: 0.4143, huber: 0.1616, swd: 0.1120, ept: 73.2693
    Epoch [7/50], Test Losses: mse: 0.4985, mae: 0.4924, huber: 0.2124, swd: 0.1467, ept: 61.0745
      Epoch 7 composite train-obj: 0.191152
            Val objective improved 0.1634 → 0.1616, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4633, mae: 0.4539, huber: 0.1899, swd: 0.1473, ept: 78.2335
    Epoch [8/50], Val Losses: mse: 0.3823, mae: 0.4148, huber: 0.1612, swd: 0.1030, ept: 72.8453
    Epoch [8/50], Test Losses: mse: 0.4931, mae: 0.4887, huber: 0.2103, swd: 0.1335, ept: 61.9245
      Epoch 8 composite train-obj: 0.189888
            Val objective improved 0.1616 → 0.1612, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4595, mae: 0.4527, huber: 0.1888, swd: 0.1462, ept: 78.4614
    Epoch [9/50], Val Losses: mse: 0.3882, mae: 0.4164, huber: 0.1632, swd: 0.1079, ept: 73.0480
    Epoch [9/50], Test Losses: mse: 0.4966, mae: 0.4892, huber: 0.2111, swd: 0.1311, ept: 61.7822
      Epoch 9 composite train-obj: 0.188788
            No improvement (0.1632), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.4567, mae: 0.4516, huber: 0.1878, swd: 0.1450, ept: 78.7770
    Epoch [10/50], Val Losses: mse: 0.3781, mae: 0.4134, huber: 0.1602, swd: 0.1050, ept: 73.9923
    Epoch [10/50], Test Losses: mse: 0.4916, mae: 0.4897, huber: 0.2105, swd: 0.1377, ept: 61.5280
      Epoch 10 composite train-obj: 0.187828
            Val objective improved 0.1612 → 0.1602, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.4535, mae: 0.4504, huber: 0.1868, swd: 0.1438, ept: 79.1232
    Epoch [11/50], Val Losses: mse: 0.3848, mae: 0.4178, huber: 0.1632, swd: 0.1085, ept: 73.3782
    Epoch [11/50], Test Losses: mse: 0.4928, mae: 0.4919, huber: 0.2113, swd: 0.1334, ept: 61.9273
      Epoch 11 composite train-obj: 0.186783
            No improvement (0.1632), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.4508, mae: 0.4495, huber: 0.1859, swd: 0.1425, ept: 79.1184
    Epoch [12/50], Val Losses: mse: 0.3841, mae: 0.4183, huber: 0.1630, swd: 0.1003, ept: 72.8252
    Epoch [12/50], Test Losses: mse: 0.4881, mae: 0.4884, huber: 0.2092, swd: 0.1229, ept: 62.0459
      Epoch 12 composite train-obj: 0.185934
            No improvement (0.1630), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.4475, mae: 0.4484, huber: 0.1849, swd: 0.1410, ept: 79.3938
    Epoch [13/50], Val Losses: mse: 0.3829, mae: 0.4182, huber: 0.1628, swd: 0.1039, ept: 73.4883
    Epoch [13/50], Test Losses: mse: 0.4877, mae: 0.4885, huber: 0.2094, swd: 0.1262, ept: 61.7194
      Epoch 13 composite train-obj: 0.184913
            No improvement (0.1628), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.4448, mae: 0.4475, huber: 0.1841, swd: 0.1397, ept: 79.6327
    Epoch [14/50], Val Losses: mse: 0.3877, mae: 0.4211, huber: 0.1648, swd: 0.1112, ept: 74.1063
    Epoch [14/50], Test Losses: mse: 0.4929, mae: 0.4946, huber: 0.2122, swd: 0.1343, ept: 61.4438
      Epoch 14 composite train-obj: 0.184064
            No improvement (0.1648), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.4415, mae: 0.4463, huber: 0.1830, swd: 0.1384, ept: 79.7067
    Epoch [15/50], Val Losses: mse: 0.3791, mae: 0.4178, huber: 0.1623, swd: 0.1126, ept: 73.5869
    Epoch [15/50], Test Losses: mse: 0.4901, mae: 0.4930, huber: 0.2116, swd: 0.1383, ept: 61.0412
      Epoch 15 composite train-obj: 0.183012
    Epoch [15/50], Test Losses: mse: 0.4916, mae: 0.4897, huber: 0.2105, swd: 0.1377, ept: 61.5280
    Best round's Test MSE: 0.4916, MAE: 0.4897, SWD: 0.1377
    Best round's Validation MSE: 0.3781, MAE: 0.4134, SWD: 0.1050
    Best round's Test verification MSE : 0.4916, MAE: 0.4897, SWD: 0.1377
    Time taken: 37.30 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5808, mae: 0.5262, huber: 0.2366, swd: 0.1561, ept: 61.6436
    Epoch [1/50], Val Losses: mse: 0.4162, mae: 0.4354, huber: 0.1735, swd: 0.1088, ept: 67.3701
    Epoch [1/50], Test Losses: mse: 0.5261, mae: 0.5068, huber: 0.2216, swd: 0.1302, ept: 58.0653
      Epoch 1 composite train-obj: 0.236621
            Val objective improved inf → 0.1735, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4913, mae: 0.4720, huber: 0.2008, swd: 0.1426, ept: 73.6561
    Epoch [2/50], Val Losses: mse: 0.4053, mae: 0.4253, huber: 0.1683, swd: 0.1086, ept: 70.1224
    Epoch [2/50], Test Losses: mse: 0.5166, mae: 0.4998, huber: 0.2176, swd: 0.1339, ept: 60.0192
      Epoch 2 composite train-obj: 0.200786
            Val objective improved 0.1735 → 0.1683, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4845, mae: 0.4663, huber: 0.1978, swd: 0.1410, ept: 75.3864
    Epoch [3/50], Val Losses: mse: 0.3935, mae: 0.4160, huber: 0.1633, swd: 0.1075, ept: 71.8966
    Epoch [3/50], Test Losses: mse: 0.5106, mae: 0.4972, huber: 0.2157, swd: 0.1414, ept: 61.0882
      Epoch 3 composite train-obj: 0.197763
            Val objective improved 0.1683 → 0.1633, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4803, mae: 0.4627, huber: 0.1959, swd: 0.1395, ept: 76.1550
    Epoch [4/50], Val Losses: mse: 0.3982, mae: 0.4156, huber: 0.1643, swd: 0.1079, ept: 72.4200
    Epoch [4/50], Test Losses: mse: 0.5085, mae: 0.4930, huber: 0.2141, swd: 0.1333, ept: 61.4565
      Epoch 4 composite train-obj: 0.195890
            No improvement (0.1643), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4773, mae: 0.4604, huber: 0.1946, swd: 0.1383, ept: 76.6021
    Epoch [5/50], Val Losses: mse: 0.3984, mae: 0.4158, huber: 0.1644, swd: 0.1074, ept: 72.8484
    Epoch [5/50], Test Losses: mse: 0.5058, mae: 0.4906, huber: 0.2128, swd: 0.1303, ept: 61.8581
      Epoch 5 composite train-obj: 0.194593
            No improvement (0.1644), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4740, mae: 0.4584, huber: 0.1933, swd: 0.1370, ept: 77.0586
    Epoch [6/50], Val Losses: mse: 0.3936, mae: 0.4151, huber: 0.1631, swd: 0.1044, ept: 73.1725
    Epoch [6/50], Test Losses: mse: 0.5041, mae: 0.4915, huber: 0.2126, swd: 0.1300, ept: 61.9822
      Epoch 6 composite train-obj: 0.193347
            Val objective improved 0.1633 → 0.1631, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4708, mae: 0.4566, huber: 0.1922, swd: 0.1365, ept: 77.5169
    Epoch [7/50], Val Losses: mse: 0.3949, mae: 0.4148, huber: 0.1637, swd: 0.1079, ept: 73.5325
    Epoch [7/50], Test Losses: mse: 0.5025, mae: 0.4902, huber: 0.2121, swd: 0.1308, ept: 61.5882
      Epoch 7 composite train-obj: 0.192179
            No improvement (0.1637), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4674, mae: 0.4550, huber: 0.1910, swd: 0.1353, ept: 77.8064
    Epoch [8/50], Val Losses: mse: 0.3898, mae: 0.4141, huber: 0.1627, swd: 0.1088, ept: 73.8280
    Epoch [8/50], Test Losses: mse: 0.4991, mae: 0.4909, huber: 0.2118, swd: 0.1353, ept: 61.6009
      Epoch 8 composite train-obj: 0.191010
            Val objective improved 0.1631 → 0.1627, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4643, mae: 0.4535, huber: 0.1899, swd: 0.1344, ept: 78.2347
    Epoch [9/50], Val Losses: mse: 0.3860, mae: 0.4130, huber: 0.1617, swd: 0.1088, ept: 74.0457
    Epoch [9/50], Test Losses: mse: 0.4979, mae: 0.4913, huber: 0.2118, swd: 0.1356, ept: 61.7142
      Epoch 9 composite train-obj: 0.189884
            Val objective improved 0.1627 → 0.1617, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.4616, mae: 0.4526, huber: 0.1891, swd: 0.1340, ept: 78.5330
    Epoch [10/50], Val Losses: mse: 0.3914, mae: 0.4158, huber: 0.1636, swd: 0.1073, ept: 73.3683
    Epoch [10/50], Test Losses: mse: 0.4974, mae: 0.4895, huber: 0.2111, swd: 0.1243, ept: 61.8723
      Epoch 10 composite train-obj: 0.189055
            No improvement (0.1636), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.4580, mae: 0.4512, huber: 0.1878, swd: 0.1327, ept: 78.9624
    Epoch [11/50], Val Losses: mse: 0.3882, mae: 0.4157, huber: 0.1628, swd: 0.1045, ept: 73.2701
    Epoch [11/50], Test Losses: mse: 0.4941, mae: 0.4884, huber: 0.2100, swd: 0.1229, ept: 62.0310
      Epoch 11 composite train-obj: 0.187820
            No improvement (0.1628), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.4556, mae: 0.4508, huber: 0.1872, swd: 0.1320, ept: 79.3412
    Epoch [12/50], Val Losses: mse: 0.3896, mae: 0.4175, huber: 0.1635, swd: 0.1012, ept: 73.1718
    Epoch [12/50], Test Losses: mse: 0.4922, mae: 0.4862, huber: 0.2089, swd: 0.1161, ept: 62.5628
      Epoch 12 composite train-obj: 0.187199
            No improvement (0.1635), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.4526, mae: 0.4492, huber: 0.1861, swd: 0.1301, ept: 79.5825
    Epoch [13/50], Val Losses: mse: 0.3824, mae: 0.4152, huber: 0.1617, swd: 0.1064, ept: 73.4052
    Epoch [13/50], Test Losses: mse: 0.4910, mae: 0.4897, huber: 0.2100, swd: 0.1258, ept: 61.8546
      Epoch 13 composite train-obj: 0.186051
            Val objective improved 0.1617 → 0.1617, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.4485, mae: 0.4479, huber: 0.1849, swd: 0.1290, ept: 79.7309
    Epoch [14/50], Val Losses: mse: 0.3825, mae: 0.4183, huber: 0.1621, swd: 0.0980, ept: 72.4838
    Epoch [14/50], Test Losses: mse: 0.4903, mae: 0.4888, huber: 0.2096, swd: 0.1158, ept: 62.3461
      Epoch 14 composite train-obj: 0.184862
            No improvement (0.1621), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.4452, mae: 0.4469, huber: 0.1839, swd: 0.1278, ept: 79.8069
    Epoch [15/50], Val Losses: mse: 0.3858, mae: 0.4167, huber: 0.1629, swd: 0.1038, ept: 73.2485
    Epoch [15/50], Test Losses: mse: 0.4909, mae: 0.4884, huber: 0.2097, swd: 0.1205, ept: 61.9024
      Epoch 15 composite train-obj: 0.183917
            No improvement (0.1629), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.4418, mae: 0.4456, huber: 0.1828, swd: 0.1265, ept: 79.9630
    Epoch [16/50], Val Losses: mse: 0.3851, mae: 0.4185, huber: 0.1635, swd: 0.1054, ept: 72.8436
    Epoch [16/50], Test Losses: mse: 0.4927, mae: 0.4930, huber: 0.2114, swd: 0.1243, ept: 61.3766
      Epoch 16 composite train-obj: 0.182813
            No improvement (0.1635), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.4387, mae: 0.4444, huber: 0.1818, swd: 0.1249, ept: 80.1155
    Epoch [17/50], Val Losses: mse: 0.3849, mae: 0.4190, huber: 0.1636, swd: 0.1028, ept: 72.5500
    Epoch [17/50], Test Losses: mse: 0.4907, mae: 0.4901, huber: 0.2101, swd: 0.1192, ept: 61.9061
      Epoch 17 composite train-obj: 0.181822
            No improvement (0.1636), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.4351, mae: 0.4433, huber: 0.1808, swd: 0.1237, ept: 80.2019
    Epoch [18/50], Val Losses: mse: 0.3828, mae: 0.4189, huber: 0.1631, swd: 0.1042, ept: 72.8420
    Epoch [18/50], Test Losses: mse: 0.4915, mae: 0.4925, huber: 0.2111, swd: 0.1261, ept: 61.6079
      Epoch 18 composite train-obj: 0.180785
    Epoch [18/50], Test Losses: mse: 0.4910, mae: 0.4897, huber: 0.2100, swd: 0.1258, ept: 61.8546
    Best round's Test MSE: 0.4910, MAE: 0.4897, SWD: 0.1258
    Best round's Validation MSE: 0.3824, MAE: 0.4152, SWD: 0.1064
    Best round's Test verification MSE : 0.4910, MAE: 0.4897, SWD: 0.1258
    Time taken: 45.35 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq96_pred196_20250510_1812)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4922 ± 0.0014
      mae: 0.4897 ± 0.0001
      huber: 0.2105 ± 0.0004
      swd: 0.1337 ± 0.0055
      ept: 61.8664 ± 0.2813
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3803 ± 0.0017
      mae: 0.4139 ± 0.0009
      huber: 0.1608 ± 0.0006
      swd: 0.1074 ± 0.0026
      ept: 73.4847 ± 0.3861
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 123.89 seconds
    
    Experiment complete: TimeMixer_etth1_seq96_pred196_20250510_1812
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 11
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7749, mae: 0.6076, huber: 0.2980, swd: 0.1791, ept: 63.3184
    Epoch [1/50], Val Losses: mse: 0.4464, mae: 0.4505, huber: 0.1849, swd: 0.1067, ept: 86.3871
    Epoch [1/50], Test Losses: mse: 0.5791, mae: 0.5407, huber: 0.2449, swd: 0.1400, ept: 74.9857
      Epoch 1 composite train-obj: 0.298026
            Val objective improved inf → 0.1849, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5892, mae: 0.5204, huber: 0.2354, swd: 0.1728, ept: 88.6536
    Epoch [2/50], Val Losses: mse: 0.4403, mae: 0.4401, huber: 0.1801, swd: 0.1051, ept: 90.6412
    Epoch [2/50], Test Losses: mse: 0.5612, mae: 0.5289, huber: 0.2374, swd: 0.1321, ept: 78.3438
      Epoch 2 composite train-obj: 0.235383
            Val objective improved 0.1849 → 0.1801, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5767, mae: 0.5111, huber: 0.2301, swd: 0.1732, ept: 93.1206
    Epoch [3/50], Val Losses: mse: 0.4307, mae: 0.4324, huber: 0.1758, swd: 0.1078, ept: 92.2754
    Epoch [3/50], Test Losses: mse: 0.5589, mae: 0.5274, huber: 0.2366, swd: 0.1432, ept: 79.5271
      Epoch 3 composite train-obj: 0.230114
            Val objective improved 0.1801 → 0.1758, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5717, mae: 0.5071, huber: 0.2279, swd: 0.1721, ept: 94.8586
    Epoch [4/50], Val Losses: mse: 0.4307, mae: 0.4295, huber: 0.1750, swd: 0.1084, ept: 94.2726
    Epoch [4/50], Test Losses: mse: 0.5562, mae: 0.5253, huber: 0.2355, swd: 0.1409, ept: 79.9786
      Epoch 4 composite train-obj: 0.227945
            Val objective improved 0.1758 → 0.1750, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5684, mae: 0.5046, huber: 0.2266, swd: 0.1706, ept: 95.9045
    Epoch [5/50], Val Losses: mse: 0.4381, mae: 0.4294, huber: 0.1763, swd: 0.1087, ept: 95.2585
    Epoch [5/50], Test Losses: mse: 0.5527, mae: 0.5218, huber: 0.2337, swd: 0.1325, ept: 80.7885
      Epoch 5 composite train-obj: 0.226570
            No improvement (0.1763), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.5655, mae: 0.5026, huber: 0.2254, swd: 0.1696, ept: 96.6607
    Epoch [6/50], Val Losses: mse: 0.4234, mae: 0.4261, huber: 0.1730, swd: 0.1114, ept: 95.4083
    Epoch [6/50], Test Losses: mse: 0.5520, mae: 0.5248, huber: 0.2348, swd: 0.1505, ept: 80.7751
      Epoch 6 composite train-obj: 0.225370
            Val objective improved 0.1750 → 0.1730, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.5624, mae: 0.5009, huber: 0.2243, swd: 0.1687, ept: 97.2795
    Epoch [7/50], Val Losses: mse: 0.4298, mae: 0.4283, huber: 0.1747, swd: 0.1046, ept: 94.9788
    Epoch [7/50], Test Losses: mse: 0.5459, mae: 0.5185, huber: 0.2316, swd: 0.1303, ept: 81.8291
      Epoch 7 composite train-obj: 0.224261
            No improvement (0.1747), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.5587, mae: 0.4992, huber: 0.2230, swd: 0.1676, ept: 98.0105
    Epoch [8/50], Val Losses: mse: 0.4327, mae: 0.4307, huber: 0.1761, swd: 0.1078, ept: 95.2583
    Epoch [8/50], Test Losses: mse: 0.5493, mae: 0.5225, huber: 0.2335, swd: 0.1339, ept: 81.8606
      Epoch 8 composite train-obj: 0.222981
            No improvement (0.1761), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.5546, mae: 0.4978, huber: 0.2217, swd: 0.1665, ept: 98.3891
    Epoch [9/50], Val Losses: mse: 0.4240, mae: 0.4282, huber: 0.1738, swd: 0.1084, ept: 95.0095
    Epoch [9/50], Test Losses: mse: 0.5483, mae: 0.5231, huber: 0.2339, swd: 0.1373, ept: 81.9935
      Epoch 9 composite train-obj: 0.221718
            No improvement (0.1738), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.5505, mae: 0.4962, huber: 0.2203, swd: 0.1658, ept: 98.7079
    Epoch [10/50], Val Losses: mse: 0.4309, mae: 0.4329, huber: 0.1764, swd: 0.1035, ept: 93.5487
    Epoch [10/50], Test Losses: mse: 0.5494, mae: 0.5232, huber: 0.2340, swd: 0.1253, ept: 82.4730
      Epoch 10 composite train-obj: 0.220323
            No improvement (0.1764), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.5471, mae: 0.4951, huber: 0.2193, swd: 0.1646, ept: 99.0987
    Epoch [11/50], Val Losses: mse: 0.4328, mae: 0.4342, huber: 0.1774, swd: 0.1091, ept: 94.5473
    Epoch [11/50], Test Losses: mse: 0.5512, mae: 0.5257, huber: 0.2353, swd: 0.1331, ept: 82.4516
      Epoch 11 composite train-obj: 0.219285
    Epoch [11/50], Test Losses: mse: 0.5520, mae: 0.5248, huber: 0.2348, swd: 0.1505, ept: 80.7751
    Best round's Test MSE: 0.5520, MAE: 0.5248, SWD: 0.1505
    Best round's Validation MSE: 0.4234, MAE: 0.4261, SWD: 0.1114
    Best round's Test verification MSE : 0.5520, MAE: 0.5248, SWD: 0.1505
    Time taken: 27.97 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6846, mae: 0.5697, huber: 0.2690, swd: 0.1983, ept: 75.4719
    Epoch [1/50], Val Losses: mse: 0.4385, mae: 0.4441, huber: 0.1809, swd: 0.1068, ept: 88.2345
    Epoch [1/50], Test Losses: mse: 0.5721, mae: 0.5381, huber: 0.2425, swd: 0.1467, ept: 76.4730
      Epoch 1 composite train-obj: 0.269020
            Val objective improved inf → 0.1809, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5788, mae: 0.5135, huber: 0.2312, swd: 0.1798, ept: 92.5597
    Epoch [2/50], Val Losses: mse: 0.4362, mae: 0.4356, huber: 0.1780, swd: 0.1085, ept: 92.9210
    Epoch [2/50], Test Losses: mse: 0.5588, mae: 0.5273, huber: 0.2367, swd: 0.1433, ept: 79.0936
      Epoch 2 composite train-obj: 0.231205
            Val objective improved 0.1809 → 0.1780, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5711, mae: 0.5074, huber: 0.2279, swd: 0.1779, ept: 94.8549
    Epoch [3/50], Val Losses: mse: 0.4358, mae: 0.4329, huber: 0.1767, swd: 0.1040, ept: 94.5174
    Epoch [3/50], Test Losses: mse: 0.5551, mae: 0.5245, huber: 0.2348, swd: 0.1348, ept: 80.4796
      Epoch 3 composite train-obj: 0.227871
            Val objective improved 0.1780 → 0.1767, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5663, mae: 0.5041, huber: 0.2260, swd: 0.1761, ept: 96.3919
    Epoch [4/50], Val Losses: mse: 0.4458, mae: 0.4342, huber: 0.1787, swd: 0.1054, ept: 95.2678
    Epoch [4/50], Test Losses: mse: 0.5542, mae: 0.5213, huber: 0.2336, swd: 0.1254, ept: 80.8398
      Epoch 4 composite train-obj: 0.225956
            No improvement (0.1787), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5620, mae: 0.5015, huber: 0.2244, swd: 0.1748, ept: 97.1437
    Epoch [5/50], Val Losses: mse: 0.4525, mae: 0.4362, huber: 0.1811, swd: 0.1138, ept: 96.8380
    Epoch [5/50], Test Losses: mse: 0.5559, mae: 0.5222, huber: 0.2344, swd: 0.1318, ept: 81.0882
      Epoch 5 composite train-obj: 0.224363
            No improvement (0.1811), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5569, mae: 0.4990, huber: 0.2226, swd: 0.1739, ept: 98.0007
    Epoch [6/50], Val Losses: mse: 0.4400, mae: 0.4342, huber: 0.1781, swd: 0.1079, ept: 95.7836
    Epoch [6/50], Test Losses: mse: 0.5493, mae: 0.5212, huber: 0.2330, swd: 0.1330, ept: 82.1416
      Epoch 6 composite train-obj: 0.222565
            No improvement (0.1781), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5519, mae: 0.4974, huber: 0.2211, swd: 0.1736, ept: 98.5911
    Epoch [7/50], Val Losses: mse: 0.4410, mae: 0.4361, huber: 0.1791, swd: 0.1087, ept: 95.9334
    Epoch [7/50], Test Losses: mse: 0.5500, mae: 0.5222, huber: 0.2336, swd: 0.1305, ept: 82.6188
      Epoch 7 composite train-obj: 0.221053
            No improvement (0.1791), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.5478, mae: 0.4961, huber: 0.2198, swd: 0.1721, ept: 99.0265
    Epoch [8/50], Val Losses: mse: 0.4513, mae: 0.4410, huber: 0.1825, swd: 0.1102, ept: 96.3292
    Epoch [8/50], Test Losses: mse: 0.5510, mae: 0.5238, huber: 0.2342, swd: 0.1284, ept: 82.6709
      Epoch 8 composite train-obj: 0.219767
    Epoch [8/50], Test Losses: mse: 0.5551, mae: 0.5245, huber: 0.2348, swd: 0.1348, ept: 80.4796
    Best round's Test MSE: 0.5551, MAE: 0.5245, SWD: 0.1348
    Best round's Validation MSE: 0.4358, MAE: 0.4329, SWD: 0.1040
    Best round's Test verification MSE : 0.5551, MAE: 0.5245, SWD: 0.1348
    Time taken: 20.29 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6561, mae: 0.5590, huber: 0.2613, swd: 0.2015, ept: 77.8362
    Epoch [1/50], Val Losses: mse: 0.4476, mae: 0.4459, huber: 0.1835, swd: 0.1145, ept: 90.2488
    Epoch [1/50], Test Losses: mse: 0.5663, mae: 0.5340, huber: 0.2403, swd: 0.1371, ept: 76.0961
      Epoch 1 composite train-obj: 0.261268
            Val objective improved inf → 0.1835, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5775, mae: 0.5124, huber: 0.2306, swd: 0.1768, ept: 92.6813
    Epoch [2/50], Val Losses: mse: 0.4406, mae: 0.4371, huber: 0.1794, swd: 0.1141, ept: 93.3969
    Epoch [2/50], Test Losses: mse: 0.5572, mae: 0.5266, huber: 0.2362, swd: 0.1375, ept: 78.6463
      Epoch 2 composite train-obj: 0.230619
            Val objective improved 0.1835 → 0.1794, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5702, mae: 0.5066, huber: 0.2275, swd: 0.1747, ept: 94.6868
    Epoch [3/50], Val Losses: mse: 0.4328, mae: 0.4308, huber: 0.1759, swd: 0.1115, ept: 94.2470
    Epoch [3/50], Test Losses: mse: 0.5515, mae: 0.5230, huber: 0.2341, swd: 0.1381, ept: 79.9880
      Epoch 3 composite train-obj: 0.227475
            Val objective improved 0.1794 → 0.1759, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5665, mae: 0.5037, huber: 0.2259, swd: 0.1735, ept: 95.8715
    Epoch [4/50], Val Losses: mse: 0.4329, mae: 0.4301, huber: 0.1759, swd: 0.1125, ept: 95.5528
    Epoch [4/50], Test Losses: mse: 0.5518, mae: 0.5227, huber: 0.2340, swd: 0.1385, ept: 80.7178
      Epoch 4 composite train-obj: 0.225862
            No improvement (0.1759), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5621, mae: 0.5011, huber: 0.2243, swd: 0.1724, ept: 96.9817
    Epoch [5/50], Val Losses: mse: 0.4232, mae: 0.4275, huber: 0.1734, swd: 0.1131, ept: 95.5218
    Epoch [5/50], Test Losses: mse: 0.5509, mae: 0.5245, huber: 0.2347, swd: 0.1426, ept: 80.6563
      Epoch 5 composite train-obj: 0.224261
            Val objective improved 0.1759 → 0.1734, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5582, mae: 0.4994, huber: 0.2230, swd: 0.1714, ept: 97.7249
    Epoch [6/50], Val Losses: mse: 0.4183, mae: 0.4264, huber: 0.1723, swd: 0.1155, ept: 96.3975
    Epoch [6/50], Test Losses: mse: 0.5516, mae: 0.5266, huber: 0.2356, swd: 0.1466, ept: 80.9051
      Epoch 6 composite train-obj: 0.222960
            Val objective improved 0.1734 → 0.1723, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.5544, mae: 0.4979, huber: 0.2218, swd: 0.1706, ept: 98.1010
    Epoch [7/50], Val Losses: mse: 0.4374, mae: 0.4332, huber: 0.1778, swd: 0.1128, ept: 96.7552
    Epoch [7/50], Test Losses: mse: 0.5537, mae: 0.5254, huber: 0.2351, swd: 0.1303, ept: 81.4028
      Epoch 7 composite train-obj: 0.221760
            No improvement (0.1778), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.5501, mae: 0.4963, huber: 0.2205, swd: 0.1693, ept: 98.6876
    Epoch [8/50], Val Losses: mse: 0.4285, mae: 0.4326, huber: 0.1759, swd: 0.1102, ept: 96.1647
    Epoch [8/50], Test Losses: mse: 0.5476, mae: 0.5235, huber: 0.2337, swd: 0.1334, ept: 81.3589
      Epoch 8 composite train-obj: 0.220450
            No improvement (0.1759), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.5462, mae: 0.4949, huber: 0.2192, swd: 0.1677, ept: 99.1808
    Epoch [9/50], Val Losses: mse: 0.4155, mae: 0.4292, huber: 0.1725, swd: 0.1065, ept: 94.2528
    Epoch [9/50], Test Losses: mse: 0.5434, mae: 0.5214, huber: 0.2327, swd: 0.1326, ept: 81.3783
      Epoch 9 composite train-obj: 0.219208
            No improvement (0.1725), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.5425, mae: 0.4937, huber: 0.2180, swd: 0.1663, ept: 99.2456
    Epoch [10/50], Val Losses: mse: 0.4450, mae: 0.4396, huber: 0.1812, swd: 0.1142, ept: 96.6949
    Epoch [10/50], Test Losses: mse: 0.5558, mae: 0.5294, huber: 0.2370, swd: 0.1297, ept: 81.2838
      Epoch 10 composite train-obj: 0.218044
            No improvement (0.1812), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.5382, mae: 0.4925, huber: 0.2168, swd: 0.1643, ept: 99.6311
    Epoch [11/50], Val Losses: mse: 0.4293, mae: 0.4350, huber: 0.1771, swd: 0.1146, ept: 94.9537
    Epoch [11/50], Test Losses: mse: 0.5482, mae: 0.5264, huber: 0.2353, swd: 0.1362, ept: 81.0336
      Epoch 11 composite train-obj: 0.216801
    Epoch [11/50], Test Losses: mse: 0.5516, mae: 0.5266, huber: 0.2356, swd: 0.1466, ept: 80.9051
    Best round's Test MSE: 0.5516, MAE: 0.5266, SWD: 0.1466
    Best round's Validation MSE: 0.4183, MAE: 0.4264, SWD: 0.1155
    Best round's Test verification MSE : 0.5516, MAE: 0.5266, SWD: 0.1466
    Time taken: 27.92 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq96_pred336_20250510_1817)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5529 ± 0.0015
      mae: 0.5253 ± 0.0009
      huber: 0.2351 ± 0.0004
      swd: 0.1440 ± 0.0067
      ept: 80.7199 ± 0.1780
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4258 ± 0.0074
      mae: 0.4285 ± 0.0031
      huber: 0.1740 ± 0.0019
      swd: 0.1103 ± 0.0048
      ept: 95.4411 ± 0.7679
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 76.23 seconds
    
    Experiment complete: TimeMixer_etth1_seq96_pred336_20250510_1817
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 8
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.9122, mae: 0.6767, huber: 0.3505, swd: 0.2392, ept: 71.0054
    Epoch [1/50], Val Losses: mse: 0.4396, mae: 0.4552, huber: 0.1873, swd: 0.1129, ept: 142.7128
    Epoch [1/50], Test Losses: mse: 0.7321, mae: 0.6310, huber: 0.3090, swd: 0.1731, ept: 106.3456
      Epoch 1 composite train-obj: 0.350466
            Val objective improved inf → 0.1873, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.7442, mae: 0.5965, huber: 0.2911, swd: 0.2254, ept: 105.1425
    Epoch [2/50], Val Losses: mse: 0.4247, mae: 0.4417, huber: 0.1793, swd: 0.1078, ept: 145.8304
    Epoch [2/50], Test Losses: mse: 0.7160, mae: 0.6222, huber: 0.3027, swd: 0.1711, ept: 110.0989
      Epoch 2 composite train-obj: 0.291080
            Val objective improved 0.1873 → 0.1793, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.7297, mae: 0.5871, huber: 0.2853, swd: 0.2250, ept: 111.6185
    Epoch [3/50], Val Losses: mse: 0.4129, mae: 0.4313, huber: 0.1733, swd: 0.0921, ept: 145.4321
    Epoch [3/50], Test Losses: mse: 0.7060, mae: 0.6141, huber: 0.2979, swd: 0.1543, ept: 111.3988
      Epoch 3 composite train-obj: 0.285291
            Val objective improved 0.1793 → 0.1733, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.7226, mae: 0.5830, huber: 0.2825, swd: 0.2242, ept: 113.7838
    Epoch [4/50], Val Losses: mse: 0.4447, mae: 0.4522, huber: 0.1860, swd: 0.1145, ept: 146.9564
    Epoch [4/50], Test Losses: mse: 0.7276, mae: 0.6287, huber: 0.3069, swd: 0.1727, ept: 110.0198
      Epoch 4 composite train-obj: 0.282536
            No improvement (0.1860), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.7151, mae: 0.5798, huber: 0.2799, swd: 0.2255, ept: 114.8611
    Epoch [5/50], Val Losses: mse: 0.4338, mae: 0.4463, huber: 0.1820, swd: 0.1066, ept: 147.5560
    Epoch [5/50], Test Losses: mse: 0.7225, mae: 0.6254, huber: 0.3051, swd: 0.1711, ept: 112.1679
      Epoch 5 composite train-obj: 0.279915
            No improvement (0.1820), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.7072, mae: 0.5769, huber: 0.2773, swd: 0.2248, ept: 116.8754
    Epoch [6/50], Val Losses: mse: 0.4872, mae: 0.4750, huber: 0.2010, swd: 0.1111, ept: 148.4662
    Epoch [6/50], Test Losses: mse: 0.7567, mae: 0.6436, huber: 0.3171, swd: 0.1631, ept: 113.1250
      Epoch 6 composite train-obj: 0.277260
            No improvement (0.2010), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.7012, mae: 0.5748, huber: 0.2753, swd: 0.2232, ept: 117.8096
    Epoch [7/50], Val Losses: mse: 0.4961, mae: 0.4838, huber: 0.2059, swd: 0.1127, ept: 148.2941
    Epoch [7/50], Test Losses: mse: 0.7577, mae: 0.6458, huber: 0.3182, swd: 0.1652, ept: 112.2791
      Epoch 7 composite train-obj: 0.275340
            No improvement (0.2059), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.6949, mae: 0.5721, huber: 0.2731, swd: 0.2221, ept: 119.3276
    Epoch [8/50], Val Losses: mse: 0.4679, mae: 0.4661, huber: 0.1948, swd: 0.1035, ept: 148.4222
    Epoch [8/50], Test Losses: mse: 0.7339, mae: 0.6303, huber: 0.3088, swd: 0.1600, ept: 113.4385
      Epoch 8 composite train-obj: 0.273149
    Epoch [8/50], Test Losses: mse: 0.7060, mae: 0.6141, huber: 0.2979, swd: 0.1543, ept: 111.3988
    Best round's Test MSE: 0.7060, MAE: 0.6141, SWD: 0.1543
    Best round's Validation MSE: 0.4129, MAE: 0.4313, SWD: 0.0921
    Best round's Test verification MSE : 0.7060, MAE: 0.6141, SWD: 0.1543
    Time taken: 25.20 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.8262, mae: 0.6358, huber: 0.3188, swd: 0.2496, ept: 92.6046
    Epoch [1/50], Val Losses: mse: 0.4195, mae: 0.4427, huber: 0.1784, swd: 0.0911, ept: 137.4629
    Epoch [1/50], Test Losses: mse: 0.7149, mae: 0.6192, huber: 0.3010, swd: 0.1538, ept: 108.8619
      Epoch 1 composite train-obj: 0.318795
            Val objective improved inf → 0.1784, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.7308, mae: 0.5882, huber: 0.2859, swd: 0.2197, ept: 110.3616
    Epoch [2/50], Val Losses: mse: 0.4179, mae: 0.4372, huber: 0.1764, swd: 0.0984, ept: 144.3326
    Epoch [2/50], Test Losses: mse: 0.7052, mae: 0.6151, huber: 0.2982, swd: 0.1632, ept: 112.2078
      Epoch 2 composite train-obj: 0.285864
            Val objective improved 0.1784 → 0.1764, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.7175, mae: 0.5813, huber: 0.2809, swd: 0.2185, ept: 113.4623
    Epoch [3/50], Val Losses: mse: 0.4320, mae: 0.4442, huber: 0.1811, swd: 0.1018, ept: 146.4784
    Epoch [3/50], Test Losses: mse: 0.7151, mae: 0.6214, huber: 0.3025, swd: 0.1644, ept: 112.2846
      Epoch 3 composite train-obj: 0.280903
            No improvement (0.1811), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.7088, mae: 0.5776, huber: 0.2778, swd: 0.2177, ept: 115.8793
    Epoch [4/50], Val Losses: mse: 0.4570, mae: 0.4565, huber: 0.1893, swd: 0.1005, ept: 147.7480
    Epoch [4/50], Test Losses: mse: 0.7267, mae: 0.6275, huber: 0.3069, swd: 0.1577, ept: 112.6808
      Epoch 4 composite train-obj: 0.277756
            No improvement (0.1893), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.7020, mae: 0.5751, huber: 0.2756, swd: 0.2154, ept: 117.5454
    Epoch [5/50], Val Losses: mse: 0.4411, mae: 0.4516, huber: 0.1849, swd: 0.0981, ept: 147.5725
    Epoch [5/50], Test Losses: mse: 0.7205, mae: 0.6248, huber: 0.3051, swd: 0.1656, ept: 113.1251
      Epoch 5 composite train-obj: 0.275552
            No improvement (0.1849), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.6978, mae: 0.5736, huber: 0.2742, swd: 0.2143, ept: 118.6829
    Epoch [6/50], Val Losses: mse: 0.4455, mae: 0.4526, huber: 0.1862, swd: 0.0903, ept: 147.0718
    Epoch [6/50], Test Losses: mse: 0.7138, mae: 0.6199, huber: 0.3024, swd: 0.1554, ept: 113.9031
      Epoch 6 composite train-obj: 0.274173
            No improvement (0.1862), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.6923, mae: 0.5715, huber: 0.2723, swd: 0.2125, ept: 119.9797
    Epoch [7/50], Val Losses: mse: 0.4571, mae: 0.4518, huber: 0.1879, swd: 0.0859, ept: 146.5609
    Epoch [7/50], Test Losses: mse: 0.7106, mae: 0.6167, huber: 0.3009, swd: 0.1417, ept: 114.6451
      Epoch 7 composite train-obj: 0.272341
    Epoch [7/50], Test Losses: mse: 0.7052, mae: 0.6151, huber: 0.2982, swd: 0.1632, ept: 112.2078
    Best round's Test MSE: 0.7052, MAE: 0.6151, SWD: 0.1632
    Best round's Validation MSE: 0.4179, MAE: 0.4372, SWD: 0.0984
    Best round's Test verification MSE : 0.7052, MAE: 0.6151, SWD: 0.1632
    Time taken: 21.97 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.8177, mae: 0.6343, huber: 0.3182, swd: 0.2483, ept: 87.6295
    Epoch [1/50], Val Losses: mse: 0.4355, mae: 0.4503, huber: 0.1844, swd: 0.1136, ept: 143.4214
    Epoch [1/50], Test Losses: mse: 0.7246, mae: 0.6274, huber: 0.3060, swd: 0.1728, ept: 108.2480
      Epoch 1 composite train-obj: 0.318172
            Val objective improved inf → 0.1844, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.7324, mae: 0.5887, huber: 0.2863, swd: 0.2372, ept: 110.9662
    Epoch [2/50], Val Losses: mse: 0.4201, mae: 0.4363, huber: 0.1766, swd: 0.1033, ept: 144.9954
    Epoch [2/50], Test Losses: mse: 0.7085, mae: 0.6158, huber: 0.2991, swd: 0.1655, ept: 110.5024
      Epoch 2 composite train-obj: 0.286333
            Val objective improved 0.1844 → 0.1766, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.7259, mae: 0.5840, huber: 0.2836, swd: 0.2351, ept: 114.3429
    Epoch [3/50], Val Losses: mse: 0.4136, mae: 0.4320, huber: 0.1740, swd: 0.1048, ept: 146.7652
    Epoch [3/50], Test Losses: mse: 0.7053, mae: 0.6148, huber: 0.2983, swd: 0.1680, ept: 110.3303
      Epoch 3 composite train-obj: 0.283615
            Val objective improved 0.1766 → 0.1740, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.7216, mae: 0.5813, huber: 0.2819, swd: 0.2340, ept: 115.5599
    Epoch [4/50], Val Losses: mse: 0.4110, mae: 0.4311, huber: 0.1732, swd: 0.1067, ept: 146.9209
    Epoch [4/50], Test Losses: mse: 0.7055, mae: 0.6164, huber: 0.2988, swd: 0.1746, ept: 111.0942
      Epoch 4 composite train-obj: 0.281905
            Val objective improved 0.1740 → 0.1732, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.7168, mae: 0.5790, huber: 0.2802, swd: 0.2329, ept: 116.4811
    Epoch [5/50], Val Losses: mse: 0.4132, mae: 0.4339, huber: 0.1744, swd: 0.1089, ept: 147.3243
    Epoch [5/50], Test Losses: mse: 0.7132, mae: 0.6216, huber: 0.3021, swd: 0.1758, ept: 111.1932
      Epoch 5 composite train-obj: 0.280215
            No improvement (0.1744), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.7104, mae: 0.5767, huber: 0.2781, swd: 0.2332, ept: 117.3598
    Epoch [6/50], Val Losses: mse: 0.4119, mae: 0.4305, huber: 0.1732, swd: 0.1066, ept: 148.4932
    Epoch [6/50], Test Losses: mse: 0.7203, mae: 0.6234, huber: 0.3044, swd: 0.1720, ept: 111.6026
      Epoch 6 composite train-obj: 0.278093
            No improvement (0.1732), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.7042, mae: 0.5749, huber: 0.2762, swd: 0.2329, ept: 118.5577
    Epoch [7/50], Val Losses: mse: 0.4474, mae: 0.4544, huber: 0.1870, swd: 0.1088, ept: 148.5405
    Epoch [7/50], Test Losses: mse: 0.7411, mae: 0.6377, huber: 0.3124, swd: 0.1689, ept: 112.7718
      Epoch 7 composite train-obj: 0.276202
            No improvement (0.1870), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.6974, mae: 0.5727, huber: 0.2740, swd: 0.2323, ept: 119.1663
    Epoch [8/50], Val Losses: mse: 0.4768, mae: 0.4680, huber: 0.1971, swd: 0.1068, ept: 147.6851
    Epoch [8/50], Test Losses: mse: 0.7492, mae: 0.6404, huber: 0.3150, swd: 0.1615, ept: 112.9755
      Epoch 8 composite train-obj: 0.274040
            No improvement (0.1971), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.6915, mae: 0.5707, huber: 0.2721, swd: 0.2305, ept: 119.9895
    Epoch [9/50], Val Losses: mse: 0.4470, mae: 0.4559, huber: 0.1876, swd: 0.0984, ept: 147.2317
    Epoch [9/50], Test Losses: mse: 0.7183, mae: 0.6242, huber: 0.3042, swd: 0.1694, ept: 113.8846
      Epoch 9 composite train-obj: 0.272053
    Epoch [9/50], Test Losses: mse: 0.7055, mae: 0.6164, huber: 0.2988, swd: 0.1746, ept: 111.0942
    Best round's Test MSE: 0.7055, MAE: 0.6164, SWD: 0.1746
    Best round's Validation MSE: 0.4110, MAE: 0.4311, SWD: 0.1067
    Best round's Test verification MSE : 0.7055, MAE: 0.6164, SWD: 0.1746
    Time taken: 28.52 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq96_pred720_20250510_1818)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.7056 ± 0.0003
      mae: 0.6152 ± 0.0010
      huber: 0.2983 ± 0.0004
      swd: 0.1640 ± 0.0083
      ept: 111.5670 ± 0.4699
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4139 ± 0.0029
      mae: 0.4332 ± 0.0028
      huber: 0.1743 ± 0.0015
      swd: 0.0991 ± 0.0060
      ept: 145.5619 ± 1.0606
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 75.74 seconds
    
    Experiment complete: TimeMixer_etth1_seq96_pred720_20250510_1818
    Model: TimeMixer
    Dataset: etth1
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 94
    Validation Batches: 13
    Test Batches: 26
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4340, mae: 0.4512, huber: 0.1833, swd: 0.1164, ept: 44.1973
    Epoch [1/50], Val Losses: mse: 0.3405, mae: 0.3834, huber: 0.1434, swd: 0.0998, ept: 53.0993
    Epoch [1/50], Test Losses: mse: 0.4505, mae: 0.4560, huber: 0.1896, swd: 0.1443, ept: 41.1847
      Epoch 1 composite train-obj: 0.183297
            Val objective improved inf → 0.1434, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4013, mae: 0.4304, huber: 0.1704, swd: 0.1150, ept: 46.8025
    Epoch [2/50], Val Losses: mse: 0.3399, mae: 0.3846, huber: 0.1443, swd: 0.1011, ept: 52.5691
    Epoch [2/50], Test Losses: mse: 0.4407, mae: 0.4542, huber: 0.1875, swd: 0.1384, ept: 40.9466
      Epoch 2 composite train-obj: 0.170405
            No improvement (0.1443), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3939, mae: 0.4260, huber: 0.1676, swd: 0.1132, ept: 47.1760
    Epoch [3/50], Val Losses: mse: 0.3313, mae: 0.3829, huber: 0.1411, swd: 0.0970, ept: 52.7949
    Epoch [3/50], Test Losses: mse: 0.4480, mae: 0.4601, huber: 0.1905, swd: 0.1578, ept: 40.5419
      Epoch 3 composite train-obj: 0.167582
            Val objective improved 0.1434 → 0.1411, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3900, mae: 0.4240, huber: 0.1662, swd: 0.1123, ept: 47.4675
    Epoch [4/50], Val Losses: mse: 0.3298, mae: 0.3809, huber: 0.1410, swd: 0.0958, ept: 52.6823
    Epoch [4/50], Test Losses: mse: 0.4308, mae: 0.4499, huber: 0.1845, swd: 0.1361, ept: 40.9901
      Epoch 4 composite train-obj: 0.166235
            Val objective improved 0.1411 → 0.1410, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3849, mae: 0.4214, huber: 0.1644, swd: 0.1100, ept: 47.6713
    Epoch [5/50], Val Losses: mse: 0.3349, mae: 0.3842, huber: 0.1436, swd: 0.1060, ept: 52.6811
    Epoch [5/50], Test Losses: mse: 0.4304, mae: 0.4528, huber: 0.1858, swd: 0.1388, ept: 40.7448
      Epoch 5 composite train-obj: 0.164418
            No improvement (0.1436), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3809, mae: 0.4195, huber: 0.1630, swd: 0.1088, ept: 47.8023
    Epoch [6/50], Val Losses: mse: 0.3295, mae: 0.3822, huber: 0.1416, swd: 0.0975, ept: 52.5426
    Epoch [6/50], Test Losses: mse: 0.4289, mae: 0.4497, huber: 0.1843, swd: 0.1371, ept: 40.9246
      Epoch 6 composite train-obj: 0.162995
            No improvement (0.1416), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.3771, mae: 0.4175, huber: 0.1616, swd: 0.1077, ept: 47.8470
    Epoch [7/50], Val Losses: mse: 0.3304, mae: 0.3837, huber: 0.1425, swd: 0.0977, ept: 52.3696
    Epoch [7/50], Test Losses: mse: 0.4236, mae: 0.4485, huber: 0.1833, swd: 0.1302, ept: 41.0631
      Epoch 7 composite train-obj: 0.161607
            No improvement (0.1425), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.3726, mae: 0.4150, huber: 0.1599, swd: 0.1060, ept: 47.9855
    Epoch [8/50], Val Losses: mse: 0.3416, mae: 0.3903, huber: 0.1467, swd: 0.0952, ept: 52.3244
    Epoch [8/50], Test Losses: mse: 0.4228, mae: 0.4467, huber: 0.1822, swd: 0.1164, ept: 41.6977
      Epoch 8 composite train-obj: 0.159912
            No improvement (0.1467), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.3671, mae: 0.4125, huber: 0.1580, swd: 0.1039, ept: 48.2114
    Epoch [9/50], Val Losses: mse: 0.3297, mae: 0.3842, huber: 0.1423, swd: 0.0923, ept: 52.5138
    Epoch [9/50], Test Losses: mse: 0.4225, mae: 0.4473, huber: 0.1828, swd: 0.1242, ept: 40.9964
      Epoch 9 composite train-obj: 0.157987
    Epoch [9/50], Test Losses: mse: 0.4308, mae: 0.4499, huber: 0.1845, swd: 0.1361, ept: 40.9901
    Best round's Test MSE: 0.4308, MAE: 0.4499, SWD: 0.1361
    Best round's Validation MSE: 0.3298, MAE: 0.3809, SWD: 0.0958
    Best round's Test verification MSE : 0.4308, MAE: 0.4499, SWD: 0.1361
    Time taken: 11.94 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4336, mae: 0.4510, huber: 0.1832, swd: 0.1140, ept: 44.3534
    Epoch [1/50], Val Losses: mse: 0.3500, mae: 0.3886, huber: 0.1470, swd: 0.0982, ept: 52.8886
    Epoch [1/50], Test Losses: mse: 0.4539, mae: 0.4597, huber: 0.1913, swd: 0.1379, ept: 40.9594
      Epoch 1 composite train-obj: 0.183180
            Val objective improved inf → 0.1470, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4025, mae: 0.4304, huber: 0.1707, swd: 0.1130, ept: 46.8866
    Epoch [2/50], Val Losses: mse: 0.3382, mae: 0.3824, huber: 0.1429, swd: 0.0968, ept: 53.1935
    Epoch [2/50], Test Losses: mse: 0.4419, mae: 0.4533, huber: 0.1873, swd: 0.1358, ept: 41.8170
      Epoch 2 composite train-obj: 0.170724
            Val objective improved 0.1470 → 0.1429, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3953, mae: 0.4262, huber: 0.1680, swd: 0.1111, ept: 47.3331
    Epoch [3/50], Val Losses: mse: 0.3436, mae: 0.3880, huber: 0.1461, swd: 0.1032, ept: 52.7972
    Epoch [3/50], Test Losses: mse: 0.4423, mae: 0.4564, huber: 0.1888, swd: 0.1342, ept: 41.3768
      Epoch 3 composite train-obj: 0.167985
            No improvement (0.1461), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3896, mae: 0.4233, huber: 0.1659, swd: 0.1096, ept: 47.5197
    Epoch [4/50], Val Losses: mse: 0.3382, mae: 0.3858, huber: 0.1443, swd: 0.0997, ept: 52.6909
    Epoch [4/50], Test Losses: mse: 0.4403, mae: 0.4560, huber: 0.1884, swd: 0.1303, ept: 41.5004
      Epoch 4 composite train-obj: 0.165904
            No improvement (0.1443), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3861, mae: 0.4218, huber: 0.1647, swd: 0.1082, ept: 47.5783
    Epoch [5/50], Val Losses: mse: 0.3302, mae: 0.3808, huber: 0.1415, swd: 0.0971, ept: 52.4495
    Epoch [5/50], Test Losses: mse: 0.4313, mae: 0.4526, huber: 0.1860, swd: 0.1314, ept: 40.8389
      Epoch 5 composite train-obj: 0.164714
            Val objective improved 0.1429 → 0.1415, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3823, mae: 0.4196, huber: 0.1633, swd: 0.1070, ept: 47.8366
    Epoch [6/50], Val Losses: mse: 0.3351, mae: 0.3872, huber: 0.1441, swd: 0.0893, ept: 52.2960
    Epoch [6/50], Test Losses: mse: 0.4267, mae: 0.4481, huber: 0.1834, swd: 0.1140, ept: 41.5697
      Epoch 6 composite train-obj: 0.163260
            No improvement (0.1441), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3783, mae: 0.4177, huber: 0.1619, swd: 0.1059, ept: 48.0955
    Epoch [7/50], Val Losses: mse: 0.3330, mae: 0.3848, huber: 0.1432, swd: 0.0907, ept: 52.4610
    Epoch [7/50], Test Losses: mse: 0.4274, mae: 0.4489, huber: 0.1837, swd: 0.1158, ept: 41.9024
      Epoch 7 composite train-obj: 0.161883
            No improvement (0.1432), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3739, mae: 0.4155, huber: 0.1603, swd: 0.1043, ept: 48.0972
    Epoch [8/50], Val Losses: mse: 0.3366, mae: 0.3876, huber: 0.1448, swd: 0.0918, ept: 52.3390
    Epoch [8/50], Test Losses: mse: 0.4247, mae: 0.4488, huber: 0.1836, swd: 0.1149, ept: 41.2334
      Epoch 8 composite train-obj: 0.160318
            No improvement (0.1448), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3705, mae: 0.4141, huber: 0.1592, swd: 0.1035, ept: 48.2421
    Epoch [9/50], Val Losses: mse: 0.3340, mae: 0.3868, huber: 0.1440, swd: 0.0879, ept: 52.5811
    Epoch [9/50], Test Losses: mse: 0.4232, mae: 0.4479, huber: 0.1827, swd: 0.1171, ept: 41.3598
      Epoch 9 composite train-obj: 0.159191
            No improvement (0.1440), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.3669, mae: 0.4123, huber: 0.1579, swd: 0.1018, ept: 48.3268
    Epoch [10/50], Val Losses: mse: 0.3353, mae: 0.3860, huber: 0.1443, swd: 0.0928, ept: 52.2603
    Epoch [10/50], Test Losses: mse: 0.4253, mae: 0.4513, huber: 0.1840, swd: 0.1269, ept: 41.4727
      Epoch 10 composite train-obj: 0.157894
    Epoch [10/50], Test Losses: mse: 0.4313, mae: 0.4526, huber: 0.1860, swd: 0.1314, ept: 40.8389
    Best round's Test MSE: 0.4313, MAE: 0.4526, SWD: 0.1314
    Best round's Validation MSE: 0.3302, MAE: 0.3808, SWD: 0.0971
    Best round's Test verification MSE : 0.4313, MAE: 0.4526, SWD: 0.1314
    Time taken: 13.12 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4335, mae: 0.4514, huber: 0.1833, swd: 0.1079, ept: 44.2435
    Epoch [1/50], Val Losses: mse: 0.3495, mae: 0.3867, huber: 0.1466, swd: 0.1011, ept: 53.1358
    Epoch [1/50], Test Losses: mse: 0.4484, mae: 0.4538, huber: 0.1889, swd: 0.1312, ept: 41.5850
      Epoch 1 composite train-obj: 0.183312
            Val objective improved inf → 0.1466, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4019, mae: 0.4300, huber: 0.1705, swd: 0.1078, ept: 46.9113
    Epoch [2/50], Val Losses: mse: 0.3531, mae: 0.3889, huber: 0.1481, swd: 0.0989, ept: 53.1071
    Epoch [2/50], Test Losses: mse: 0.4484, mae: 0.4530, huber: 0.1885, swd: 0.1237, ept: 41.5535
      Epoch 2 composite train-obj: 0.170489
            No improvement (0.1481), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3951, mae: 0.4264, huber: 0.1680, swd: 0.1064, ept: 47.2195
    Epoch [3/50], Val Losses: mse: 0.3361, mae: 0.3822, huber: 0.1428, swd: 0.0992, ept: 52.9797
    Epoch [3/50], Test Losses: mse: 0.4377, mae: 0.4533, huber: 0.1869, swd: 0.1321, ept: 41.2210
      Epoch 3 composite train-obj: 0.168010
            Val objective improved 0.1466 → 0.1428, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3905, mae: 0.4237, huber: 0.1663, swd: 0.1048, ept: 47.5510
    Epoch [4/50], Val Losses: mse: 0.3329, mae: 0.3836, huber: 0.1425, swd: 0.0962, ept: 52.4777
    Epoch [4/50], Test Losses: mse: 0.4329, mae: 0.4527, huber: 0.1857, swd: 0.1326, ept: 40.8205
      Epoch 4 composite train-obj: 0.166291
            Val objective improved 0.1428 → 0.1425, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3856, mae: 0.4216, huber: 0.1646, swd: 0.1034, ept: 47.5983
    Epoch [5/50], Val Losses: mse: 0.3292, mae: 0.3824, huber: 0.1413, swd: 0.0944, ept: 52.5309
    Epoch [5/50], Test Losses: mse: 0.4280, mae: 0.4521, huber: 0.1847, swd: 0.1304, ept: 40.7782
      Epoch 5 composite train-obj: 0.164639
            Val objective improved 0.1425 → 0.1413, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3830, mae: 0.4203, huber: 0.1637, swd: 0.1025, ept: 47.8185
    Epoch [6/50], Val Losses: mse: 0.3373, mae: 0.3874, huber: 0.1447, swd: 0.0896, ept: 52.2903
    Epoch [6/50], Test Losses: mse: 0.4240, mae: 0.4465, huber: 0.1821, swd: 0.1066, ept: 41.8656
      Epoch 6 composite train-obj: 0.163707
            No improvement (0.1447), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3777, mae: 0.4177, huber: 0.1618, swd: 0.1006, ept: 47.9214
    Epoch [7/50], Val Losses: mse: 0.3295, mae: 0.3831, huber: 0.1420, swd: 0.0987, ept: 52.5790
    Epoch [7/50], Test Losses: mse: 0.4253, mae: 0.4515, huber: 0.1844, swd: 0.1326, ept: 40.5951
      Epoch 7 composite train-obj: 0.161814
            No improvement (0.1420), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3744, mae: 0.4162, huber: 0.1607, swd: 0.0999, ept: 48.0810
    Epoch [8/50], Val Losses: mse: 0.3326, mae: 0.3869, huber: 0.1435, swd: 0.0885, ept: 52.3641
    Epoch [8/50], Test Losses: mse: 0.4272, mae: 0.4525, huber: 0.1848, swd: 0.1229, ept: 40.9568
      Epoch 8 composite train-obj: 0.160679
            No improvement (0.1435), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3698, mae: 0.4136, huber: 0.1589, swd: 0.0984, ept: 48.2203
    Epoch [9/50], Val Losses: mse: 0.3361, mae: 0.3884, huber: 0.1448, swd: 0.0857, ept: 51.7990
    Epoch [9/50], Test Losses: mse: 0.4223, mae: 0.4468, huber: 0.1819, swd: 0.1088, ept: 41.7027
      Epoch 9 composite train-obj: 0.158905
            No improvement (0.1448), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.3657, mae: 0.4117, huber: 0.1574, swd: 0.0964, ept: 48.2725
    Epoch [10/50], Val Losses: mse: 0.3429, mae: 0.3907, huber: 0.1473, swd: 0.0888, ept: 52.0612
    Epoch [10/50], Test Losses: mse: 0.4218, mae: 0.4470, huber: 0.1819, swd: 0.1053, ept: 41.6567
      Epoch 10 composite train-obj: 0.157423
    Epoch [10/50], Test Losses: mse: 0.4280, mae: 0.4521, huber: 0.1847, swd: 0.1304, ept: 40.7782
    Best round's Test MSE: 0.4280, MAE: 0.4521, SWD: 0.1304
    Best round's Validation MSE: 0.3292, MAE: 0.3824, SWD: 0.0944
    Best round's Test verification MSE : 0.4280, MAE: 0.4521, SWD: 0.1304
    Time taken: 13.38 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq96_pred96_20250510_1806)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4301 ± 0.0015
      mae: 0.4516 ± 0.0012
      huber: 0.1851 ± 0.0007
      swd: 0.1327 ± 0.0025
      ept: 40.8691 ± 0.0891
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3297 ± 0.0004
      mae: 0.3813 ± 0.0007
      huber: 0.1413 ± 0.0002
      swd: 0.0958 ± 0.0011
      ept: 52.5542 ± 0.0965
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 38.49 seconds
    
    Experiment complete: PatchTST_etth1_seq96_pred96_20250510_1806
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.5360, mae: 0.4983, huber: 0.2179, swd: 0.1473, ept: 64.4191
    Epoch [1/50], Val Losses: mse: 0.3985, mae: 0.4208, huber: 0.1658, swd: 0.1138, ept: 73.7729
    Epoch [1/50], Test Losses: mse: 0.5142, mae: 0.4987, huber: 0.2168, swd: 0.1449, ept: 60.5508
      Epoch 1 composite train-obj: 0.217934
            Val objective improved inf → 0.1658, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5035, mae: 0.4787, huber: 0.2054, swd: 0.1477, ept: 68.7825
    Epoch [2/50], Val Losses: mse: 0.3786, mae: 0.4142, huber: 0.1597, swd: 0.1028, ept: 73.6007
    Epoch [2/50], Test Losses: mse: 0.5067, mae: 0.4970, huber: 0.2151, swd: 0.1502, ept: 60.7274
      Epoch 2 composite train-obj: 0.205354
            Val objective improved 0.1658 → 0.1597, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4947, mae: 0.4748, huber: 0.2024, swd: 0.1460, ept: 69.3552
    Epoch [3/50], Val Losses: mse: 0.3776, mae: 0.4172, huber: 0.1607, swd: 0.0993, ept: 72.5037
    Epoch [3/50], Test Losses: mse: 0.4962, mae: 0.4916, huber: 0.2118, swd: 0.1289, ept: 60.8329
      Epoch 3 composite train-obj: 0.202412
            No improvement (0.1607), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4877, mae: 0.4719, huber: 0.2001, swd: 0.1438, ept: 69.7787
    Epoch [4/50], Val Losses: mse: 0.3828, mae: 0.4209, huber: 0.1634, swd: 0.0995, ept: 72.3268
    Epoch [4/50], Test Losses: mse: 0.4973, mae: 0.4944, huber: 0.2128, swd: 0.1248, ept: 61.2739
      Epoch 4 composite train-obj: 0.200098
            No improvement (0.1634), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4820, mae: 0.4698, huber: 0.1983, swd: 0.1420, ept: 70.0396
    Epoch [5/50], Val Losses: mse: 0.3791, mae: 0.4213, huber: 0.1630, swd: 0.0940, ept: 71.2043
    Epoch [5/50], Test Losses: mse: 0.4922, mae: 0.4929, huber: 0.2115, swd: 0.1205, ept: 60.7815
      Epoch 5 composite train-obj: 0.198321
            No improvement (0.1630), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4744, mae: 0.4671, huber: 0.1960, swd: 0.1395, ept: 70.2611
    Epoch [6/50], Val Losses: mse: 0.3760, mae: 0.4177, huber: 0.1618, swd: 0.1053, ept: 73.4891
    Epoch [6/50], Test Losses: mse: 0.4982, mae: 0.4975, huber: 0.2148, swd: 0.1348, ept: 60.6495
      Epoch 6 composite train-obj: 0.196016
            No improvement (0.1618), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4669, mae: 0.4645, huber: 0.1938, swd: 0.1375, ept: 70.5352
    Epoch [7/50], Val Losses: mse: 0.3757, mae: 0.4169, huber: 0.1613, swd: 0.0984, ept: 73.2256
    Epoch [7/50], Test Losses: mse: 0.4964, mae: 0.4959, huber: 0.2139, swd: 0.1301, ept: 60.4314
      Epoch 7 composite train-obj: 0.193808
    Epoch [7/50], Test Losses: mse: 0.5067, mae: 0.4970, huber: 0.2151, swd: 0.1502, ept: 60.7274
    Best round's Test MSE: 0.5067, MAE: 0.4970, SWD: 0.1502
    Best round's Validation MSE: 0.3786, MAE: 0.4142, SWD: 0.1028
    Best round's Test verification MSE : 0.5067, MAE: 0.4970, SWD: 0.1502
    Time taken: 9.52 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5361, mae: 0.4978, huber: 0.2178, swd: 0.1488, ept: 64.6350
    Epoch [1/50], Val Losses: mse: 0.3869, mae: 0.4169, huber: 0.1618, swd: 0.0981, ept: 73.3364
    Epoch [1/50], Test Losses: mse: 0.5174, mae: 0.5003, huber: 0.2178, swd: 0.1447, ept: 60.7938
      Epoch 1 composite train-obj: 0.217849
            Val objective improved inf → 0.1618, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5043, mae: 0.4790, huber: 0.2056, swd: 0.1488, ept: 68.9722
    Epoch [2/50], Val Losses: mse: 0.3844, mae: 0.4190, huber: 0.1627, swd: 0.0987, ept: 73.2495
    Epoch [2/50], Test Losses: mse: 0.5014, mae: 0.4928, huber: 0.2128, swd: 0.1292, ept: 61.1742
      Epoch 2 composite train-obj: 0.205618
            No improvement (0.1627), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4946, mae: 0.4750, huber: 0.2024, swd: 0.1461, ept: 69.2942
    Epoch [3/50], Val Losses: mse: 0.3795, mae: 0.4154, huber: 0.1616, swd: 0.1004, ept: 73.5299
    Epoch [3/50], Test Losses: mse: 0.4939, mae: 0.4895, huber: 0.2105, swd: 0.1245, ept: 61.1701
      Epoch 3 composite train-obj: 0.202418
            Val objective improved 0.1618 → 0.1616, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4865, mae: 0.4716, huber: 0.1998, swd: 0.1438, ept: 69.7441
    Epoch [4/50], Val Losses: mse: 0.3706, mae: 0.4147, huber: 0.1592, swd: 0.0971, ept: 72.5138
    Epoch [4/50], Test Losses: mse: 0.4942, mae: 0.4923, huber: 0.2116, swd: 0.1325, ept: 60.8532
      Epoch 4 composite train-obj: 0.199776
            Val objective improved 0.1616 → 0.1592, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4814, mae: 0.4696, huber: 0.1981, swd: 0.1417, ept: 69.9487
    Epoch [5/50], Val Losses: mse: 0.3797, mae: 0.4172, huber: 0.1625, swd: 0.1019, ept: 74.0093
    Epoch [5/50], Test Losses: mse: 0.4918, mae: 0.4915, huber: 0.2110, swd: 0.1233, ept: 61.2873
      Epoch 5 composite train-obj: 0.198103
            No improvement (0.1625), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4745, mae: 0.4668, huber: 0.1959, swd: 0.1405, ept: 70.1035
    Epoch [6/50], Val Losses: mse: 0.3830, mae: 0.4187, huber: 0.1638, swd: 0.1037, ept: 75.0088
    Epoch [6/50], Test Losses: mse: 0.5010, mae: 0.4992, huber: 0.2157, swd: 0.1289, ept: 60.3383
      Epoch 6 composite train-obj: 0.195896
            No improvement (0.1638), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.4689, mae: 0.4644, huber: 0.1940, swd: 0.1385, ept: 70.3869
    Epoch [7/50], Val Losses: mse: 0.3876, mae: 0.4220, huber: 0.1657, swd: 0.1056, ept: 74.8899
    Epoch [7/50], Test Losses: mse: 0.5070, mae: 0.5030, huber: 0.2177, swd: 0.1285, ept: 60.7328
      Epoch 7 composite train-obj: 0.194027
            No improvement (0.1657), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.4613, mae: 0.4615, huber: 0.1916, swd: 0.1365, ept: 70.6477
    Epoch [8/50], Val Losses: mse: 0.4233, mae: 0.4368, huber: 0.1773, swd: 0.0990, ept: 73.8398
    Epoch [8/50], Test Losses: mse: 0.5035, mae: 0.4975, huber: 0.2150, swd: 0.1060, ept: 60.9838
      Epoch 8 composite train-obj: 0.191641
            No improvement (0.1773), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.4558, mae: 0.4597, huber: 0.1901, swd: 0.1345, ept: 70.7039
    Epoch [9/50], Val Losses: mse: 0.3958, mae: 0.4251, huber: 0.1688, swd: 0.1044, ept: 74.8075
    Epoch [9/50], Test Losses: mse: 0.5040, mae: 0.5027, huber: 0.2177, swd: 0.1261, ept: 60.1993
      Epoch 9 composite train-obj: 0.190139
    Epoch [9/50], Test Losses: mse: 0.4942, mae: 0.4923, huber: 0.2116, swd: 0.1325, ept: 60.8532
    Best round's Test MSE: 0.4942, MAE: 0.4923, SWD: 0.1325
    Best round's Validation MSE: 0.3706, MAE: 0.4147, SWD: 0.0971
    Best round's Test verification MSE : 0.4942, MAE: 0.4923, SWD: 0.1325
    Time taken: 11.90 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5387, mae: 0.4994, huber: 0.2188, swd: 0.1339, ept: 64.2747
    Epoch [1/50], Val Losses: mse: 0.3932, mae: 0.4186, huber: 0.1643, swd: 0.1108, ept: 74.1954
    Epoch [1/50], Test Losses: mse: 0.5072, mae: 0.4954, huber: 0.2149, swd: 0.1383, ept: 60.5587
      Epoch 1 composite train-obj: 0.218825
            Val objective improved inf → 0.1643, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5032, mae: 0.4791, huber: 0.2054, swd: 0.1341, ept: 68.7111
    Epoch [2/50], Val Losses: mse: 0.3807, mae: 0.4165, huber: 0.1617, swd: 0.1070, ept: 73.5376
    Epoch [2/50], Test Losses: mse: 0.4988, mae: 0.4963, huber: 0.2138, swd: 0.1362, ept: 60.0448
      Epoch 2 composite train-obj: 0.205436
            Val objective improved 0.1643 → 0.1617, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4914, mae: 0.4742, huber: 0.2016, swd: 0.1315, ept: 69.2395
    Epoch [3/50], Val Losses: mse: 0.3765, mae: 0.4175, huber: 0.1612, swd: 0.0893, ept: 71.8372
    Epoch [3/50], Test Losses: mse: 0.4915, mae: 0.4914, huber: 0.2106, swd: 0.1144, ept: 60.8231
      Epoch 3 composite train-obj: 0.201594
            Val objective improved 0.1617 → 0.1612, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4843, mae: 0.4713, huber: 0.1993, swd: 0.1289, ept: 69.6760
    Epoch [4/50], Val Losses: mse: 0.3727, mae: 0.4140, huber: 0.1596, swd: 0.0981, ept: 73.9963
    Epoch [4/50], Test Losses: mse: 0.4910, mae: 0.4927, huber: 0.2113, swd: 0.1303, ept: 60.3813
      Epoch 4 composite train-obj: 0.199308
            Val objective improved 0.1612 → 0.1596, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4766, mae: 0.4681, huber: 0.1968, swd: 0.1273, ept: 69.8784
    Epoch [5/50], Val Losses: mse: 0.3695, mae: 0.4146, huber: 0.1589, swd: 0.0892, ept: 73.1340
    Epoch [5/50], Test Losses: mse: 0.4927, mae: 0.4934, huber: 0.2120, swd: 0.1252, ept: 59.7627
      Epoch 5 composite train-obj: 0.196807
            Val objective improved 0.1596 → 0.1589, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4710, mae: 0.4661, huber: 0.1951, swd: 0.1256, ept: 70.2101
    Epoch [6/50], Val Losses: mse: 0.3837, mae: 0.4186, huber: 0.1638, swd: 0.0951, ept: 73.0175
    Epoch [6/50], Test Losses: mse: 0.4895, mae: 0.4912, huber: 0.2108, swd: 0.1145, ept: 60.4449
      Epoch 6 composite train-obj: 0.195124
            No improvement (0.1638), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4644, mae: 0.4636, huber: 0.1931, swd: 0.1236, ept: 70.1744
    Epoch [7/50], Val Losses: mse: 0.3688, mae: 0.4113, huber: 0.1580, swd: 0.0993, ept: 74.8034
    Epoch [7/50], Test Losses: mse: 0.5026, mae: 0.4997, huber: 0.2161, swd: 0.1506, ept: 59.4153
      Epoch 7 composite train-obj: 0.193096
            Val objective improved 0.1589 → 0.1580, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4570, mae: 0.4607, huber: 0.1908, swd: 0.1219, ept: 70.5723
    Epoch [8/50], Val Losses: mse: 0.4028, mae: 0.4272, huber: 0.1702, swd: 0.0961, ept: 74.1492
    Epoch [8/50], Test Losses: mse: 0.4912, mae: 0.4950, huber: 0.2125, swd: 0.1136, ept: 60.7173
      Epoch 8 composite train-obj: 0.190826
            No improvement (0.1702), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4505, mae: 0.4583, huber: 0.1889, swd: 0.1203, ept: 70.6514
    Epoch [9/50], Val Losses: mse: 0.3842, mae: 0.4205, huber: 0.1644, swd: 0.0950, ept: 74.4872
    Epoch [9/50], Test Losses: mse: 0.4956, mae: 0.4987, huber: 0.2147, swd: 0.1299, ept: 60.2613
      Epoch 9 composite train-obj: 0.188909
            No improvement (0.1644), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.4418, mae: 0.4552, huber: 0.1863, swd: 0.1169, ept: 70.7252
    Epoch [10/50], Val Losses: mse: 0.3845, mae: 0.4232, huber: 0.1650, swd: 0.0851, ept: 73.1350
    Epoch [10/50], Test Losses: mse: 0.4955, mae: 0.4976, huber: 0.2142, swd: 0.1106, ept: 60.1862
      Epoch 10 composite train-obj: 0.186313
            No improvement (0.1650), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.4360, mae: 0.4534, huber: 0.1847, swd: 0.1146, ept: 70.9792
    Epoch [11/50], Val Losses: mse: 0.3905, mae: 0.4223, huber: 0.1663, swd: 0.0927, ept: 73.2714
    Epoch [11/50], Test Losses: mse: 0.4963, mae: 0.4973, huber: 0.2144, swd: 0.1256, ept: 60.2592
      Epoch 11 composite train-obj: 0.184741
            No improvement (0.1663), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.4289, mae: 0.4502, huber: 0.1823, swd: 0.1123, ept: 71.0980
    Epoch [12/50], Val Losses: mse: 0.3717, mae: 0.4133, huber: 0.1598, swd: 0.0931, ept: 73.4666
    Epoch [12/50], Test Losses: mse: 0.4995, mae: 0.4981, huber: 0.2156, swd: 0.1320, ept: 60.1262
      Epoch 12 composite train-obj: 0.182341
    Epoch [12/50], Test Losses: mse: 0.5026, mae: 0.4997, huber: 0.2161, swd: 0.1506, ept: 59.4153
    Best round's Test MSE: 0.5026, MAE: 0.4997, SWD: 0.1506
    Best round's Validation MSE: 0.3688, MAE: 0.4113, SWD: 0.0993
    Best round's Test verification MSE : 0.5026, MAE: 0.4997, SWD: 0.1506
    Time taken: 15.99 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq96_pred196_20250510_1827)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5012 ± 0.0052
      mae: 0.4963 ± 0.0031
      huber: 0.2143 ± 0.0019
      swd: 0.1444 ± 0.0085
      ept: 60.3320 ± 0.6502
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3727 ± 0.0043
      mae: 0.4134 ± 0.0015
      huber: 0.1590 ± 0.0007
      swd: 0.0997 ± 0.0024
      ept: 73.6393 ± 0.9351
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 37.47 seconds
    
    Experiment complete: PatchTST_etth1_seq96_pred196_20250510_1827
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 11
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6226, mae: 0.5382, huber: 0.2477, swd: 0.1669, ept: 80.3509
    Epoch [1/50], Val Losses: mse: 0.4277, mae: 0.4343, huber: 0.1762, swd: 0.1100, ept: 95.4248
    Epoch [1/50], Test Losses: mse: 0.5571, mae: 0.5268, huber: 0.2359, swd: 0.1459, ept: 79.8454
      Epoch 1 composite train-obj: 0.247678
            Val objective improved inf → 0.1762, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5866, mae: 0.5193, huber: 0.2346, swd: 0.1665, ept: 86.2812
    Epoch [2/50], Val Losses: mse: 0.4136, mae: 0.4333, huber: 0.1735, swd: 0.1002, ept: 93.8269
    Epoch [2/50], Test Losses: mse: 0.5474, mae: 0.5230, huber: 0.2333, swd: 0.1389, ept: 80.9340
      Epoch 2 composite train-obj: 0.234568
            Val objective improved 0.1762 → 0.1735, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5727, mae: 0.5143, huber: 0.2303, swd: 0.1634, ept: 87.5529
    Epoch [3/50], Val Losses: mse: 0.4238, mae: 0.4376, huber: 0.1773, swd: 0.1087, ept: 95.8709
    Epoch [3/50], Test Losses: mse: 0.5575, mae: 0.5331, huber: 0.2389, swd: 0.1410, ept: 79.6522
      Epoch 3 composite train-obj: 0.230337
            No improvement (0.1773), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5629, mae: 0.5109, huber: 0.2273, swd: 0.1608, ept: 87.5126
    Epoch [4/50], Val Losses: mse: 0.4179, mae: 0.4361, huber: 0.1761, swd: 0.1020, ept: 95.5489
    Epoch [4/50], Test Losses: mse: 0.5522, mae: 0.5281, huber: 0.2366, swd: 0.1380, ept: 80.8569
      Epoch 4 composite train-obj: 0.227333
            No improvement (0.1761), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5533, mae: 0.5076, huber: 0.2245, swd: 0.1582, ept: 88.1003
    Epoch [5/50], Val Losses: mse: 0.4353, mae: 0.4430, huber: 0.1812, swd: 0.1029, ept: 95.7144
    Epoch [5/50], Test Losses: mse: 0.5475, mae: 0.5284, huber: 0.2356, swd: 0.1289, ept: 80.8438
      Epoch 5 composite train-obj: 0.224498
            No improvement (0.1812), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.5426, mae: 0.5033, huber: 0.2212, swd: 0.1556, ept: 88.5103
    Epoch [6/50], Val Losses: mse: 0.4319, mae: 0.4419, huber: 0.1807, swd: 0.1028, ept: 95.9621
    Epoch [6/50], Test Losses: mse: 0.5514, mae: 0.5320, huber: 0.2380, swd: 0.1319, ept: 80.5347
      Epoch 6 composite train-obj: 0.221176
            No improvement (0.1807), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.5324, mae: 0.4994, huber: 0.2179, swd: 0.1536, ept: 88.9527
    Epoch [7/50], Val Losses: mse: 0.4268, mae: 0.4428, huber: 0.1798, swd: 0.0977, ept: 93.6799
    Epoch [7/50], Test Losses: mse: 0.5529, mae: 0.5306, huber: 0.2379, swd: 0.1248, ept: 80.9252
      Epoch 7 composite train-obj: 0.217914
    Epoch [7/50], Test Losses: mse: 0.5474, mae: 0.5230, huber: 0.2333, swd: 0.1389, ept: 80.9340
    Best round's Test MSE: 0.5474, MAE: 0.5230, SWD: 0.1389
    Best round's Validation MSE: 0.4136, MAE: 0.4333, SWD: 0.1002
    Best round's Test verification MSE : 0.5474, MAE: 0.5230, SWD: 0.1389
    Time taken: 9.43 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6229, mae: 0.5385, huber: 0.2478, swd: 0.1724, ept: 80.8014
    Epoch [1/50], Val Losses: mse: 0.4247, mae: 0.4334, huber: 0.1750, swd: 0.1131, ept: 96.5397
    Epoch [1/50], Test Losses: mse: 0.5592, mae: 0.5274, huber: 0.2368, swd: 0.1524, ept: 80.4266
      Epoch 1 composite train-obj: 0.247840
            Val objective improved inf → 0.1750, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5895, mae: 0.5204, huber: 0.2355, swd: 0.1728, ept: 86.5805
    Epoch [2/50], Val Losses: mse: 0.4376, mae: 0.4441, huber: 0.1819, swd: 0.1192, ept: 98.3309
    Epoch [2/50], Test Losses: mse: 0.5666, mae: 0.5390, huber: 0.2423, swd: 0.1396, ept: 79.6531
      Epoch 2 composite train-obj: 0.235495
            No improvement (0.1819), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5758, mae: 0.5157, huber: 0.2314, swd: 0.1682, ept: 87.2825
    Epoch [3/50], Val Losses: mse: 0.4394, mae: 0.4414, huber: 0.1814, swd: 0.1057, ept: 95.7855
    Epoch [3/50], Test Losses: mse: 0.5538, mae: 0.5279, huber: 0.2367, swd: 0.1247, ept: 81.0317
      Epoch 3 composite train-obj: 0.231395
            No improvement (0.1814), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.5648, mae: 0.5116, huber: 0.2280, swd: 0.1653, ept: 87.9304
    Epoch [4/50], Val Losses: mse: 0.4277, mae: 0.4401, huber: 0.1790, swd: 0.1029, ept: 95.8037
    Epoch [4/50], Test Losses: mse: 0.5463, mae: 0.5256, huber: 0.2343, swd: 0.1323, ept: 81.7540
      Epoch 4 composite train-obj: 0.228000
            No improvement (0.1790), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.5544, mae: 0.5077, huber: 0.2248, swd: 0.1634, ept: 88.2018
    Epoch [5/50], Val Losses: mse: 0.4494, mae: 0.4482, huber: 0.1860, swd: 0.1043, ept: 97.2418
    Epoch [5/50], Test Losses: mse: 0.5514, mae: 0.5298, huber: 0.2372, swd: 0.1231, ept: 81.2773
      Epoch 5 composite train-obj: 0.224767
            No improvement (0.1860), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.5458, mae: 0.5046, huber: 0.2222, swd: 0.1611, ept: 88.7092
    Epoch [6/50], Val Losses: mse: 0.4277, mae: 0.4393, huber: 0.1786, swd: 0.0987, ept: 95.1927
    Epoch [6/50], Test Losses: mse: 0.5454, mae: 0.5237, huber: 0.2336, swd: 0.1224, ept: 81.6334
      Epoch 6 composite train-obj: 0.222150
    Epoch [6/50], Test Losses: mse: 0.5592, mae: 0.5274, huber: 0.2368, swd: 0.1524, ept: 80.4266
    Best round's Test MSE: 0.5592, MAE: 0.5274, SWD: 0.1524
    Best round's Validation MSE: 0.4247, MAE: 0.4334, SWD: 0.1131
    Best round's Test verification MSE : 0.5592, MAE: 0.5274, SWD: 0.1524
    Time taken: 8.03 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6222, mae: 0.5380, huber: 0.2476, swd: 0.1689, ept: 80.9992
    Epoch [1/50], Val Losses: mse: 0.4281, mae: 0.4359, huber: 0.1768, swd: 0.1042, ept: 93.9806
    Epoch [1/50], Test Losses: mse: 0.5582, mae: 0.5260, huber: 0.2361, swd: 0.1349, ept: 80.3826
      Epoch 1 composite train-obj: 0.247561
            Val objective improved inf → 0.1768, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5873, mae: 0.5199, huber: 0.2349, swd: 0.1683, ept: 86.2085
    Epoch [2/50], Val Losses: mse: 0.4335, mae: 0.4396, huber: 0.1791, swd: 0.1018, ept: 94.0849
    Epoch [2/50], Test Losses: mse: 0.5453, mae: 0.5233, huber: 0.2331, swd: 0.1196, ept: 81.2187
      Epoch 2 composite train-obj: 0.234898
            No improvement (0.1791), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5731, mae: 0.5145, huber: 0.2304, swd: 0.1648, ept: 87.4272
    Epoch [3/50], Val Losses: mse: 0.4164, mae: 0.4337, huber: 0.1745, swd: 0.0929, ept: 92.9429
    Epoch [3/50], Test Losses: mse: 0.5409, mae: 0.5219, huber: 0.2318, swd: 0.1310, ept: 80.9843
      Epoch 3 composite train-obj: 0.230430
            Val objective improved 0.1768 → 0.1745, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5619, mae: 0.5106, huber: 0.2271, swd: 0.1607, ept: 87.6637
    Epoch [4/50], Val Losses: mse: 0.4254, mae: 0.4390, huber: 0.1781, swd: 0.1027, ept: 95.8738
    Epoch [4/50], Test Losses: mse: 0.5480, mae: 0.5274, huber: 0.2355, swd: 0.1431, ept: 80.6034
      Epoch 4 composite train-obj: 0.227101
            No improvement (0.1781), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5519, mae: 0.5069, huber: 0.2240, swd: 0.1593, ept: 88.0500
    Epoch [5/50], Val Losses: mse: 0.4225, mae: 0.4365, huber: 0.1773, swd: 0.1007, ept: 95.6771
    Epoch [5/50], Test Losses: mse: 0.5511, mae: 0.5302, huber: 0.2375, swd: 0.1307, ept: 80.5943
      Epoch 5 composite train-obj: 0.224003
            No improvement (0.1773), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5426, mae: 0.5036, huber: 0.2212, swd: 0.1568, ept: 88.2399
    Epoch [6/50], Val Losses: mse: 0.4350, mae: 0.4417, huber: 0.1809, swd: 0.0962, ept: 95.3182
    Epoch [6/50], Test Losses: mse: 0.5419, mae: 0.5241, huber: 0.2333, swd: 0.1188, ept: 80.6700
      Epoch 6 composite train-obj: 0.221247
            No improvement (0.1809), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5316, mae: 0.4991, huber: 0.2177, swd: 0.1535, ept: 88.6794
    Epoch [7/50], Val Losses: mse: 0.4118, mae: 0.4341, huber: 0.1741, swd: 0.0978, ept: 94.6782
    Epoch [7/50], Test Losses: mse: 0.5588, mae: 0.5331, huber: 0.2394, swd: 0.1496, ept: 80.5478
      Epoch 7 composite train-obj: 0.217717
            Val objective improved 0.1745 → 0.1741, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.5241, mae: 0.4965, huber: 0.2154, swd: 0.1522, ept: 89.0208
    Epoch [8/50], Val Losses: mse: 0.4344, mae: 0.4462, huber: 0.1825, swd: 0.0957, ept: 96.2149
    Epoch [8/50], Test Losses: mse: 0.5646, mae: 0.5376, huber: 0.2428, swd: 0.1340, ept: 79.7531
      Epoch 8 composite train-obj: 0.215390
            No improvement (0.1825), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.5162, mae: 0.4933, huber: 0.2129, swd: 0.1496, ept: 89.4421
    Epoch [9/50], Val Losses: mse: 0.4635, mae: 0.4541, huber: 0.1904, swd: 0.0984, ept: 94.3192
    Epoch [9/50], Test Losses: mse: 0.5706, mae: 0.5432, huber: 0.2456, swd: 0.1130, ept: 79.4918
      Epoch 9 composite train-obj: 0.212935
            No improvement (0.1904), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.5097, mae: 0.4908, huber: 0.2109, swd: 0.1471, ept: 89.7251
    Epoch [10/50], Val Losses: mse: 0.4413, mae: 0.4519, huber: 0.1857, swd: 0.0950, ept: 93.1008
    Epoch [10/50], Test Losses: mse: 0.5676, mae: 0.5389, huber: 0.2440, swd: 0.1196, ept: 79.2473
      Epoch 10 composite train-obj: 0.210863
            No improvement (0.1857), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.5007, mae: 0.4873, huber: 0.2081, swd: 0.1436, ept: 89.5522
    Epoch [11/50], Val Losses: mse: 0.4645, mae: 0.4540, huber: 0.1897, swd: 0.0989, ept: 94.7612
    Epoch [11/50], Test Losses: mse: 0.5654, mae: 0.5388, huber: 0.2435, swd: 0.1145, ept: 79.1621
      Epoch 11 composite train-obj: 0.208073
            No improvement (0.1897), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.4944, mae: 0.4848, huber: 0.2060, swd: 0.1413, ept: 90.1362
    Epoch [12/50], Val Losses: mse: 0.4432, mae: 0.4497, huber: 0.1851, swd: 0.0972, ept: 94.3737
    Epoch [12/50], Test Losses: mse: 0.5658, mae: 0.5385, huber: 0.2436, swd: 0.1308, ept: 79.1039
      Epoch 12 composite train-obj: 0.205991
    Epoch [12/50], Test Losses: mse: 0.5588, mae: 0.5331, huber: 0.2394, swd: 0.1496, ept: 80.5478
    Best round's Test MSE: 0.5588, MAE: 0.5331, SWD: 0.1496
    Best round's Validation MSE: 0.4118, MAE: 0.4341, SWD: 0.0978
    Best round's Test verification MSE : 0.5588, MAE: 0.5331, SWD: 0.1496
    Time taken: 15.82 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq96_pred336_20250510_1815)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5551 ± 0.0055
      mae: 0.5278 ± 0.0041
      huber: 0.2365 ± 0.0025
      swd: 0.1470 ± 0.0058
      ept: 80.6361 ± 0.2163
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4167 ± 0.0057
      mae: 0.4336 ± 0.0004
      huber: 0.1742 ± 0.0006
      swd: 0.1037 ± 0.0067
      ept: 95.0150 ± 1.1328
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 33.34 seconds
    
    Experiment complete: PatchTST_etth1_seq96_pred336_20250510_1815
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 8
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7752, mae: 0.6120, huber: 0.3022, swd: 0.2200, ept: 94.8739
    Epoch [1/50], Val Losses: mse: 0.4196, mae: 0.4424, huber: 0.1780, swd: 0.0913, ept: 140.4553
    Epoch [1/50], Test Losses: mse: 0.7304, mae: 0.6256, huber: 0.3065, swd: 0.1487, ept: 116.5706
      Epoch 1 composite train-obj: 0.302177
            Val objective improved inf → 0.1780, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.7331, mae: 0.5940, huber: 0.2879, swd: 0.2185, ept: 102.7480
    Epoch [2/50], Val Losses: mse: 0.4301, mae: 0.4483, huber: 0.1829, swd: 0.0908, ept: 142.4609
    Epoch [2/50], Test Losses: mse: 0.7113, mae: 0.6199, huber: 0.3012, swd: 0.1502, ept: 114.9849
      Epoch 2 composite train-obj: 0.287858
            No improvement (0.1829), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.7125, mae: 0.5861, huber: 0.2813, swd: 0.2154, ept: 103.7863
    Epoch [3/50], Val Losses: mse: 0.4701, mae: 0.4704, huber: 0.1985, swd: 0.0885, ept: 143.0766
    Epoch [3/50], Test Losses: mse: 0.7159, mae: 0.6253, huber: 0.3040, swd: 0.1500, ept: 115.0906
      Epoch 3 composite train-obj: 0.281253
            No improvement (0.1985), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.6962, mae: 0.5797, huber: 0.2760, swd: 0.2123, ept: 104.8079
    Epoch [4/50], Val Losses: mse: 0.4555, mae: 0.4599, huber: 0.1919, swd: 0.0965, ept: 145.0899
    Epoch [4/50], Test Losses: mse: 0.7378, mae: 0.6331, huber: 0.3109, swd: 0.1504, ept: 114.6804
      Epoch 4 composite train-obj: 0.276029
            No improvement (0.1919), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.6834, mae: 0.5742, huber: 0.2717, swd: 0.2091, ept: 105.9011
    Epoch [5/50], Val Losses: mse: 0.4681, mae: 0.4636, huber: 0.1954, swd: 0.0781, ept: 144.6181
    Epoch [5/50], Test Losses: mse: 0.7245, mae: 0.6267, huber: 0.3064, swd: 0.1358, ept: 112.0915
      Epoch 5 composite train-obj: 0.271697
            No improvement (0.1954), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.6719, mae: 0.5696, huber: 0.2679, swd: 0.2054, ept: 106.7709
    Epoch [6/50], Val Losses: mse: 0.4102, mae: 0.4409, huber: 0.1769, swd: 0.0847, ept: 140.1153
    Epoch [6/50], Test Losses: mse: 0.7152, mae: 0.6250, huber: 0.3038, swd: 0.1499, ept: 114.3443
      Epoch 6 composite train-obj: 0.267896
            Val objective improved 0.1780 → 0.1769, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.6609, mae: 0.5650, huber: 0.2643, swd: 0.2022, ept: 106.9865
    Epoch [7/50], Val Losses: mse: 0.4519, mae: 0.4571, huber: 0.1902, swd: 0.0838, ept: 143.0840
    Epoch [7/50], Test Losses: mse: 0.7304, mae: 0.6325, huber: 0.3094, swd: 0.1374, ept: 114.3123
      Epoch 7 composite train-obj: 0.264297
            No improvement (0.1902), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.6504, mae: 0.5608, huber: 0.2609, swd: 0.1997, ept: 107.3663
    Epoch [8/50], Val Losses: mse: 0.4804, mae: 0.4713, huber: 0.2003, swd: 0.0892, ept: 143.7338
    Epoch [8/50], Test Losses: mse: 0.7676, mae: 0.6455, huber: 0.3212, swd: 0.1386, ept: 114.0023
      Epoch 8 composite train-obj: 0.260907
            No improvement (0.2003), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.6413, mae: 0.5569, huber: 0.2578, swd: 0.1965, ept: 107.7570
    Epoch [9/50], Val Losses: mse: 0.4780, mae: 0.4727, huber: 0.1993, swd: 0.0780, ept: 140.2782
    Epoch [9/50], Test Losses: mse: 0.7190, mae: 0.6241, huber: 0.3045, swd: 0.1240, ept: 114.1534
      Epoch 9 composite train-obj: 0.257845
            No improvement (0.1993), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.6323, mae: 0.5532, huber: 0.2549, swd: 0.1938, ept: 108.5150
    Epoch [10/50], Val Losses: mse: 0.4445, mae: 0.4545, huber: 0.1874, swd: 0.0830, ept: 141.4551
    Epoch [10/50], Test Losses: mse: 0.7239, mae: 0.6302, huber: 0.3074, swd: 0.1361, ept: 113.2940
      Epoch 10 composite train-obj: 0.254881
            No improvement (0.1874), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.6274, mae: 0.5510, huber: 0.2533, swd: 0.1923, ept: 108.3234
    Epoch [11/50], Val Losses: mse: 0.4707, mae: 0.4712, huber: 0.1980, swd: 0.0798, ept: 139.3668
    Epoch [11/50], Test Losses: mse: 0.7439, mae: 0.6385, huber: 0.3143, swd: 0.1376, ept: 113.8769
      Epoch 11 composite train-obj: 0.253254
    Epoch [11/50], Test Losses: mse: 0.7152, mae: 0.6250, huber: 0.3038, swd: 0.1499, ept: 114.3443
    Best round's Test MSE: 0.7152, MAE: 0.6250, SWD: 0.1499
    Best round's Validation MSE: 0.4102, MAE: 0.4409, SWD: 0.0847
    Best round's Test verification MSE : 0.7152, MAE: 0.6250, SWD: 0.1499
    Time taken: 14.97 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7759, mae: 0.6124, huber: 0.3024, swd: 0.2108, ept: 94.9687
    Epoch [1/50], Val Losses: mse: 0.4487, mae: 0.4624, huber: 0.1904, swd: 0.1071, ept: 146.8874
    Epoch [1/50], Test Losses: mse: 0.7246, mae: 0.6286, huber: 0.3062, swd: 0.1740, ept: 113.3024
      Epoch 1 composite train-obj: 0.302437
            Val objective improved inf → 0.1904, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.7313, mae: 0.5935, huber: 0.2874, swd: 0.2097, ept: 102.0328
    Epoch [2/50], Val Losses: mse: 0.4538, mae: 0.4632, huber: 0.1921, swd: 0.0966, ept: 146.5323
    Epoch [2/50], Test Losses: mse: 0.7334, mae: 0.6336, huber: 0.3099, swd: 0.1531, ept: 114.7028
      Epoch 2 composite train-obj: 0.287416
            No improvement (0.1921), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.7122, mae: 0.5859, huber: 0.2811, swd: 0.2072, ept: 103.9779
    Epoch [3/50], Val Losses: mse: 0.4302, mae: 0.4495, huber: 0.1830, swd: 0.0854, ept: 138.0865
    Epoch [3/50], Test Losses: mse: 0.7144, mae: 0.6211, huber: 0.3023, swd: 0.1454, ept: 117.2468
      Epoch 3 composite train-obj: 0.281094
            Val objective improved 0.1904 → 0.1830, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.7000, mae: 0.5811, huber: 0.2771, swd: 0.2050, ept: 104.7931
    Epoch [4/50], Val Losses: mse: 0.4223, mae: 0.4434, huber: 0.1797, swd: 0.0805, ept: 143.6211
    Epoch [4/50], Test Losses: mse: 0.7066, mae: 0.6192, huber: 0.3001, swd: 0.1431, ept: 115.1988
      Epoch 4 composite train-obj: 0.277107
            Val objective improved 0.1830 → 0.1797, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.6849, mae: 0.5749, huber: 0.2723, swd: 0.2019, ept: 105.8806
    Epoch [5/50], Val Losses: mse: 0.4388, mae: 0.4509, huber: 0.1849, swd: 0.0823, ept: 142.6410
    Epoch [5/50], Test Losses: mse: 0.7000, mae: 0.6140, huber: 0.2971, swd: 0.1383, ept: 116.1515
      Epoch 5 composite train-obj: 0.272277
            No improvement (0.1849), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.6733, mae: 0.5702, huber: 0.2685, swd: 0.1986, ept: 106.7783
    Epoch [6/50], Val Losses: mse: 0.4379, mae: 0.4515, huber: 0.1859, swd: 0.0784, ept: 141.2211
    Epoch [6/50], Test Losses: mse: 0.7078, mae: 0.6195, huber: 0.3008, swd: 0.1446, ept: 116.6468
      Epoch 6 composite train-obj: 0.268534
            No improvement (0.1859), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.6603, mae: 0.5650, huber: 0.2644, swd: 0.1962, ept: 107.3229
    Epoch [7/50], Val Losses: mse: 0.4327, mae: 0.4506, huber: 0.1839, swd: 0.0806, ept: 140.8699
    Epoch [7/50], Test Losses: mse: 0.7093, mae: 0.6199, huber: 0.3011, swd: 0.1534, ept: 116.2170
      Epoch 7 composite train-obj: 0.264356
            No improvement (0.1839), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.6507, mae: 0.5611, huber: 0.2612, swd: 0.1929, ept: 107.3495
    Epoch [8/50], Val Losses: mse: 0.4342, mae: 0.4513, huber: 0.1846, swd: 0.0790, ept: 141.7843
    Epoch [8/50], Test Losses: mse: 0.7191, mae: 0.6242, huber: 0.3046, swd: 0.1567, ept: 115.9534
      Epoch 8 composite train-obj: 0.261193
            No improvement (0.1846), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.6403, mae: 0.5569, huber: 0.2579, swd: 0.1904, ept: 107.6328
    Epoch [9/50], Val Losses: mse: 0.4576, mae: 0.4584, huber: 0.1909, swd: 0.0758, ept: 140.2607
    Epoch [9/50], Test Losses: mse: 0.7202, mae: 0.6263, huber: 0.3059, swd: 0.1319, ept: 115.7046
      Epoch 9 composite train-obj: 0.257880
    Epoch [9/50], Test Losses: mse: 0.7066, mae: 0.6192, huber: 0.3001, swd: 0.1431, ept: 115.1988
    Best round's Test MSE: 0.7066, MAE: 0.6192, SWD: 0.1431
    Best round's Validation MSE: 0.4223, MAE: 0.4434, SWD: 0.0805
    Best round's Test verification MSE : 0.7066, MAE: 0.6192, SWD: 0.1431
    Time taken: 12.28 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7740, mae: 0.6117, huber: 0.3018, swd: 0.2291, ept: 94.5528
    Epoch [1/50], Val Losses: mse: 0.4510, mae: 0.4627, huber: 0.1910, swd: 0.1043, ept: 145.2082
    Epoch [1/50], Test Losses: mse: 0.7232, mae: 0.6276, huber: 0.3058, swd: 0.1666, ept: 113.1519
      Epoch 1 composite train-obj: 0.301818
            Val objective improved inf → 0.1910, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.7308, mae: 0.5930, huber: 0.2871, swd: 0.2274, ept: 102.6302
    Epoch [2/50], Val Losses: mse: 0.4491, mae: 0.4627, huber: 0.1919, swd: 0.1002, ept: 147.3085
    Epoch [2/50], Test Losses: mse: 0.7212, mae: 0.6282, huber: 0.3059, swd: 0.1736, ept: 112.8197
      Epoch 2 composite train-obj: 0.287134
            No improvement (0.1919), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.7111, mae: 0.5857, huber: 0.2809, swd: 0.2238, ept: 104.1528
    Epoch [3/50], Val Losses: mse: 0.4480, mae: 0.4603, huber: 0.1907, swd: 0.0922, ept: 145.0514
    Epoch [3/50], Test Losses: mse: 0.7177, mae: 0.6237, huber: 0.3039, swd: 0.1626, ept: 114.3579
      Epoch 3 composite train-obj: 0.280946
            Val objective improved 0.1910 → 0.1907, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.6960, mae: 0.5798, huber: 0.2761, swd: 0.2212, ept: 104.2994
    Epoch [4/50], Val Losses: mse: 0.4425, mae: 0.4575, huber: 0.1879, swd: 0.0824, ept: 140.8876
    Epoch [4/50], Test Losses: mse: 0.7105, mae: 0.6208, huber: 0.3012, swd: 0.1476, ept: 115.9016
      Epoch 4 composite train-obj: 0.276102
            Val objective improved 0.1907 → 0.1879, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.6839, mae: 0.5748, huber: 0.2721, swd: 0.2186, ept: 105.4497
    Epoch [5/50], Val Losses: mse: 0.4249, mae: 0.4471, huber: 0.1807, swd: 0.0788, ept: 142.1850
    Epoch [5/50], Test Losses: mse: 0.7092, mae: 0.6202, huber: 0.3009, swd: 0.1488, ept: 113.9312
      Epoch 5 composite train-obj: 0.272057
            Val objective improved 0.1879 → 0.1807, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.6737, mae: 0.5704, huber: 0.2686, swd: 0.2153, ept: 106.5748
    Epoch [6/50], Val Losses: mse: 0.4689, mae: 0.4673, huber: 0.1969, swd: 0.0871, ept: 144.4342
    Epoch [6/50], Test Losses: mse: 0.7378, mae: 0.6343, huber: 0.3115, swd: 0.1474, ept: 113.3057
      Epoch 6 composite train-obj: 0.268622
            No improvement (0.1969), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.6622, mae: 0.5655, huber: 0.2648, swd: 0.2123, ept: 106.9619
    Epoch [7/50], Val Losses: mse: 0.4592, mae: 0.4609, huber: 0.1919, swd: 0.0795, ept: 142.5845
    Epoch [7/50], Test Losses: mse: 0.7136, mae: 0.6226, huber: 0.3031, swd: 0.1360, ept: 114.4925
      Epoch 7 composite train-obj: 0.264794
            No improvement (0.1919), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.6530, mae: 0.5615, huber: 0.2617, swd: 0.2102, ept: 107.2956
    Epoch [8/50], Val Losses: mse: 0.4821, mae: 0.4736, huber: 0.2010, swd: 0.0777, ept: 141.7436
    Epoch [8/50], Test Losses: mse: 0.7280, mae: 0.6313, huber: 0.3088, swd: 0.1374, ept: 114.2422
      Epoch 8 composite train-obj: 0.261659
            No improvement (0.2010), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.6444, mae: 0.5580, huber: 0.2588, swd: 0.2062, ept: 107.6970
    Epoch [9/50], Val Losses: mse: 0.4280, mae: 0.4493, huber: 0.1830, swd: 0.0795, ept: 139.9671
    Epoch [9/50], Test Losses: mse: 0.7326, mae: 0.6336, huber: 0.3108, swd: 0.1465, ept: 113.7881
      Epoch 9 composite train-obj: 0.258841
            No improvement (0.1830), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.6361, mae: 0.5547, huber: 0.2561, swd: 0.2038, ept: 107.9715
    Epoch [10/50], Val Losses: mse: 0.4250, mae: 0.4488, huber: 0.1821, swd: 0.0743, ept: 140.0398
    Epoch [10/50], Test Losses: mse: 0.7141, mae: 0.6233, huber: 0.3039, swd: 0.1433, ept: 114.0196
      Epoch 10 composite train-obj: 0.256117
    Epoch [10/50], Test Losses: mse: 0.7092, mae: 0.6202, huber: 0.3009, swd: 0.1488, ept: 113.9312
    Best round's Test MSE: 0.7092, MAE: 0.6202, SWD: 0.1488
    Best round's Validation MSE: 0.4249, MAE: 0.4471, SWD: 0.0788
    Best round's Test verification MSE : 0.7092, MAE: 0.6202, SWD: 0.1488
    Time taken: 13.84 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq96_pred720_20250510_1827)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.7103 ± 0.0036
      mae: 0.6215 ± 0.0026
      huber: 0.3016 ± 0.0016
      swd: 0.1472 ± 0.0030
      ept: 114.4914 ± 0.5279
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4192 ± 0.0064
      mae: 0.4438 ± 0.0025
      huber: 0.1791 ± 0.0016
      swd: 0.0813 ± 0.0025
      ept: 141.9738 ± 1.4390
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 41.13 seconds
    
    Experiment complete: PatchTST_etth1_seq96_pred720_20250510_1827
    Model: PatchTST
    Dataset: etth1
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 94
    Validation Batches: 13
    Test Batches: 26
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5043, mae: 0.4951, huber: 0.2108, swd: 0.1495, ept: 37.4150
    Epoch [1/50], Val Losses: mse: 0.3783, mae: 0.4175, huber: 0.1611, swd: 0.1189, ept: 47.1381
    Epoch [1/50], Test Losses: mse: 0.4735, mae: 0.4756, huber: 0.2004, swd: 0.1487, ept: 37.7083
      Epoch 1 composite train-obj: 0.210751
            Val objective improved inf → 0.1611, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3747, mae: 0.4223, huber: 0.1621, swd: 0.1263, ept: 49.1688
    Epoch [2/50], Val Losses: mse: 0.3606, mae: 0.4002, huber: 0.1527, swd: 0.1154, ept: 50.5670
    Epoch [2/50], Test Losses: mse: 0.4503, mae: 0.4574, huber: 0.1903, swd: 0.1457, ept: 40.4944
      Epoch 2 composite train-obj: 0.162136
            Val objective improved 0.1611 → 0.1527, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3632, mae: 0.4126, huber: 0.1571, swd: 0.1229, ept: 51.1250
    Epoch [3/50], Val Losses: mse: 0.3544, mae: 0.3917, huber: 0.1493, swd: 0.1108, ept: 51.8764
    Epoch [3/50], Test Losses: mse: 0.4434, mae: 0.4503, huber: 0.1868, swd: 0.1420, ept: 41.4042
      Epoch 3 composite train-obj: 0.157060
            Val objective improved 0.1527 → 0.1493, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3594, mae: 0.4088, huber: 0.1552, swd: 0.1211, ept: 51.9241
    Epoch [4/50], Val Losses: mse: 0.3519, mae: 0.3896, huber: 0.1484, swd: 0.1127, ept: 52.0845
    Epoch [4/50], Test Losses: mse: 0.4381, mae: 0.4464, huber: 0.1847, swd: 0.1437, ept: 41.7774
      Epoch 4 composite train-obj: 0.155237
            Val objective improved 0.1493 → 0.1484, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3570, mae: 0.4064, huber: 0.1542, swd: 0.1206, ept: 52.3169
    Epoch [5/50], Val Losses: mse: 0.3536, mae: 0.3904, huber: 0.1491, swd: 0.1173, ept: 52.2783
    Epoch [5/50], Test Losses: mse: 0.4352, mae: 0.4435, huber: 0.1834, swd: 0.1445, ept: 42.1590
      Epoch 5 composite train-obj: 0.154150
            No improvement (0.1491), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3555, mae: 0.4053, huber: 0.1536, swd: 0.1209, ept: 52.6086
    Epoch [6/50], Val Losses: mse: 0.3498, mae: 0.3862, huber: 0.1472, swd: 0.1119, ept: 52.8321
    Epoch [6/50], Test Losses: mse: 0.4341, mae: 0.4422, huber: 0.1828, swd: 0.1425, ept: 42.3143
      Epoch 6 composite train-obj: 0.153566
            Val objective improved 0.1484 → 0.1472, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3549, mae: 0.4044, huber: 0.1532, swd: 0.1203, ept: 52.8211
    Epoch [7/50], Val Losses: mse: 0.3511, mae: 0.3880, huber: 0.1481, swd: 0.1158, ept: 52.9968
    Epoch [7/50], Test Losses: mse: 0.4325, mae: 0.4414, huber: 0.1823, swd: 0.1437, ept: 42.5551
      Epoch 7 composite train-obj: 0.153227
            No improvement (0.1481), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.3539, mae: 0.4038, huber: 0.1529, swd: 0.1205, ept: 52.9526
    Epoch [8/50], Val Losses: mse: 0.3486, mae: 0.3833, huber: 0.1462, swd: 0.1100, ept: 53.1659
    Epoch [8/50], Test Losses: mse: 0.4343, mae: 0.4406, huber: 0.1824, swd: 0.1416, ept: 42.7349
      Epoch 8 composite train-obj: 0.152863
            Val objective improved 0.1472 → 0.1462, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.3535, mae: 0.4032, huber: 0.1526, swd: 0.1201, ept: 53.0687
    Epoch [9/50], Val Losses: mse: 0.3467, mae: 0.3842, huber: 0.1462, swd: 0.1111, ept: 53.1625
    Epoch [9/50], Test Losses: mse: 0.4312, mae: 0.4392, huber: 0.1814, swd: 0.1408, ept: 42.6777
      Epoch 9 composite train-obj: 0.152628
            No improvement (0.1462), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.3533, mae: 0.4028, huber: 0.1525, swd: 0.1200, ept: 53.1610
    Epoch [10/50], Val Losses: mse: 0.3463, mae: 0.3837, huber: 0.1459, swd: 0.1122, ept: 53.3271
    Epoch [10/50], Test Losses: mse: 0.4298, mae: 0.4383, huber: 0.1809, swd: 0.1420, ept: 42.8510
      Epoch 10 composite train-obj: 0.152524
            Val objective improved 0.1462 → 0.1459, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.3530, mae: 0.4024, huber: 0.1523, swd: 0.1198, ept: 53.2478
    Epoch [11/50], Val Losses: mse: 0.3468, mae: 0.3846, huber: 0.1463, swd: 0.1141, ept: 53.1982
    Epoch [11/50], Test Losses: mse: 0.4311, mae: 0.4396, huber: 0.1815, swd: 0.1424, ept: 42.6887
      Epoch 11 composite train-obj: 0.152349
            No improvement (0.1463), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.3527, mae: 0.4023, huber: 0.1523, swd: 0.1201, ept: 53.2202
    Epoch [12/50], Val Losses: mse: 0.3496, mae: 0.3840, huber: 0.1467, swd: 0.1132, ept: 53.3462
    Epoch [12/50], Test Losses: mse: 0.4317, mae: 0.4387, huber: 0.1813, swd: 0.1422, ept: 42.8381
      Epoch 12 composite train-obj: 0.152286
            No improvement (0.1467), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.3527, mae: 0.4021, huber: 0.1522, swd: 0.1200, ept: 53.3289
    Epoch [13/50], Val Losses: mse: 0.3474, mae: 0.3840, huber: 0.1463, swd: 0.1147, ept: 53.3673
    Epoch [13/50], Test Losses: mse: 0.4307, mae: 0.4385, huber: 0.1812, swd: 0.1445, ept: 42.7288
      Epoch 13 composite train-obj: 0.152217
            No improvement (0.1463), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.3523, mae: 0.4019, huber: 0.1521, swd: 0.1204, ept: 53.3060
    Epoch [14/50], Val Losses: mse: 0.3461, mae: 0.3820, huber: 0.1453, swd: 0.1075, ept: 53.5045
    Epoch [14/50], Test Losses: mse: 0.4324, mae: 0.4386, huber: 0.1814, swd: 0.1374, ept: 42.8632
      Epoch 14 composite train-obj: 0.152097
            Val objective improved 0.1459 → 0.1453, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.3526, mae: 0.4019, huber: 0.1522, swd: 0.1196, ept: 53.3389
    Epoch [15/50], Val Losses: mse: 0.3465, mae: 0.3833, huber: 0.1458, swd: 0.1120, ept: 53.3731
    Epoch [15/50], Test Losses: mse: 0.4301, mae: 0.4381, huber: 0.1808, swd: 0.1411, ept: 42.7851
      Epoch 15 composite train-obj: 0.152161
            No improvement (0.1458), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.3519, mae: 0.4017, huber: 0.1520, swd: 0.1200, ept: 53.3816
    Epoch [16/50], Val Losses: mse: 0.3441, mae: 0.3807, huber: 0.1446, swd: 0.1088, ept: 53.5180
    Epoch [16/50], Test Losses: mse: 0.4313, mae: 0.4383, huber: 0.1811, swd: 0.1412, ept: 42.7861
      Epoch 16 composite train-obj: 0.151965
            Val objective improved 0.1453 → 0.1446, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 0.3522, mae: 0.4016, huber: 0.1520, swd: 0.1198, ept: 53.4327
    Epoch [17/50], Val Losses: mse: 0.3469, mae: 0.3817, huber: 0.1456, swd: 0.1099, ept: 53.4941
    Epoch [17/50], Test Losses: mse: 0.4322, mae: 0.4382, huber: 0.1813, swd: 0.1410, ept: 42.7002
      Epoch 17 composite train-obj: 0.152028
            No improvement (0.1456), counter 1/5
    Epoch [18/50], Train Losses: mse: 0.3522, mae: 0.4016, huber: 0.1520, swd: 0.1198, ept: 53.3916
    Epoch [18/50], Val Losses: mse: 0.3472, mae: 0.3838, huber: 0.1463, swd: 0.1157, ept: 53.2815
    Epoch [18/50], Test Losses: mse: 0.4291, mae: 0.4370, huber: 0.1805, swd: 0.1423, ept: 42.8167
      Epoch 18 composite train-obj: 0.152022
            No improvement (0.1463), counter 2/5
    Epoch [19/50], Train Losses: mse: 0.3526, mae: 0.4017, huber: 0.1521, swd: 0.1203, ept: 53.3849
    Epoch [19/50], Val Losses: mse: 0.3439, mae: 0.3818, huber: 0.1451, swd: 0.1109, ept: 53.3051
    Epoch [19/50], Test Losses: mse: 0.4292, mae: 0.4369, huber: 0.1805, swd: 0.1409, ept: 42.7534
      Epoch 19 composite train-obj: 0.152088
            No improvement (0.1451), counter 3/5
    Epoch [20/50], Train Losses: mse: 0.3522, mae: 0.4016, huber: 0.1520, swd: 0.1198, ept: 53.4655
    Epoch [20/50], Val Losses: mse: 0.3465, mae: 0.3838, huber: 0.1462, swd: 0.1145, ept: 53.3370
    Epoch [20/50], Test Losses: mse: 0.4279, mae: 0.4361, huber: 0.1800, swd: 0.1411, ept: 42.7281
      Epoch 20 composite train-obj: 0.152049
            No improvement (0.1462), counter 4/5
    Epoch [21/50], Train Losses: mse: 0.3521, mae: 0.4015, huber: 0.1520, swd: 0.1197, ept: 53.4061
    Epoch [21/50], Val Losses: mse: 0.3484, mae: 0.3847, huber: 0.1466, swd: 0.1162, ept: 53.5089
    Epoch [21/50], Test Losses: mse: 0.4300, mae: 0.4379, huber: 0.1808, swd: 0.1427, ept: 42.7703
      Epoch 21 composite train-obj: 0.151972
    Epoch [21/50], Test Losses: mse: 0.4313, mae: 0.4383, huber: 0.1811, swd: 0.1412, ept: 42.7861
    Best round's Test MSE: 0.4313, MAE: 0.4383, SWD: 0.1412
    Best round's Validation MSE: 0.3441, MAE: 0.3807, SWD: 0.1088
    Best round's Test verification MSE : 0.4313, MAE: 0.4383, SWD: 0.1412
    Time taken: 21.46 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4939, mae: 0.4913, huber: 0.2077, swd: 0.1454, ept: 37.9521
    Epoch [1/50], Val Losses: mse: 0.3755, mae: 0.4136, huber: 0.1592, swd: 0.1104, ept: 47.5747
    Epoch [1/50], Test Losses: mse: 0.4756, mae: 0.4760, huber: 0.2008, swd: 0.1438, ept: 37.9683
      Epoch 1 composite train-obj: 0.207683
            Val objective improved inf → 0.1592, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3746, mae: 0.4220, huber: 0.1620, swd: 0.1233, ept: 49.1673
    Epoch [2/50], Val Losses: mse: 0.3609, mae: 0.3993, huber: 0.1525, swd: 0.1101, ept: 50.8758
    Epoch [2/50], Test Losses: mse: 0.4515, mae: 0.4576, huber: 0.1906, swd: 0.1405, ept: 40.6319
      Epoch 2 composite train-obj: 0.162037
            Val objective improved 0.1592 → 0.1525, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3635, mae: 0.4126, huber: 0.1571, swd: 0.1204, ept: 51.0862
    Epoch [3/50], Val Losses: mse: 0.3555, mae: 0.3931, huber: 0.1499, swd: 0.1077, ept: 51.8493
    Epoch [3/50], Test Losses: mse: 0.4421, mae: 0.4496, huber: 0.1863, swd: 0.1365, ept: 41.4376
      Epoch 3 composite train-obj: 0.157099
            Val objective improved 0.1525 → 0.1499, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3589, mae: 0.4086, huber: 0.1551, swd: 0.1191, ept: 51.9187
    Epoch [4/50], Val Losses: mse: 0.3524, mae: 0.3901, huber: 0.1486, swd: 0.1083, ept: 52.5867
    Epoch [4/50], Test Losses: mse: 0.4375, mae: 0.4458, huber: 0.1843, swd: 0.1367, ept: 42.0550
      Epoch 4 composite train-obj: 0.155107
            Val objective improved 0.1499 → 0.1486, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3572, mae: 0.4067, huber: 0.1543, swd: 0.1185, ept: 52.3709
    Epoch [5/50], Val Losses: mse: 0.3530, mae: 0.3895, huber: 0.1487, swd: 0.1100, ept: 52.4406
    Epoch [5/50], Test Losses: mse: 0.4356, mae: 0.4434, huber: 0.1834, swd: 0.1357, ept: 42.3699
      Epoch 5 composite train-obj: 0.154251
            No improvement (0.1487), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3555, mae: 0.4051, huber: 0.1535, swd: 0.1182, ept: 52.6477
    Epoch [6/50], Val Losses: mse: 0.3511, mae: 0.3876, huber: 0.1479, swd: 0.1085, ept: 52.6866
    Epoch [6/50], Test Losses: mse: 0.4341, mae: 0.4419, huber: 0.1826, swd: 0.1368, ept: 42.4521
      Epoch 6 composite train-obj: 0.153521
            Val objective improved 0.1486 → 0.1479, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3547, mae: 0.4043, huber: 0.1532, swd: 0.1180, ept: 52.8233
    Epoch [7/50], Val Losses: mse: 0.3488, mae: 0.3863, huber: 0.1472, swd: 0.1099, ept: 52.7081
    Epoch [7/50], Test Losses: mse: 0.4316, mae: 0.4404, huber: 0.1818, swd: 0.1377, ept: 42.5686
      Epoch 7 composite train-obj: 0.153159
            Val objective improved 0.1479 → 0.1472, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3538, mae: 0.4036, huber: 0.1528, swd: 0.1183, ept: 52.9843
    Epoch [8/50], Val Losses: mse: 0.3492, mae: 0.3861, huber: 0.1470, swd: 0.1096, ept: 53.0556
    Epoch [8/50], Test Losses: mse: 0.4323, mae: 0.4405, huber: 0.1818, swd: 0.1359, ept: 42.6641
      Epoch 8 composite train-obj: 0.152801
            Val objective improved 0.1472 → 0.1470, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.3536, mae: 0.4032, huber: 0.1526, swd: 0.1180, ept: 53.0591
    Epoch [9/50], Val Losses: mse: 0.3475, mae: 0.3849, huber: 0.1465, swd: 0.1105, ept: 53.0360
    Epoch [9/50], Test Losses: mse: 0.4294, mae: 0.4379, huber: 0.1807, swd: 0.1364, ept: 42.8543
      Epoch 9 composite train-obj: 0.152638
            Val objective improved 0.1470 → 0.1465, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.3534, mae: 0.4028, huber: 0.1525, swd: 0.1177, ept: 53.1618
    Epoch [10/50], Val Losses: mse: 0.3493, mae: 0.3878, huber: 0.1479, swd: 0.1126, ept: 52.8840
    Epoch [10/50], Test Losses: mse: 0.4302, mae: 0.4388, huber: 0.1811, swd: 0.1367, ept: 42.7692
      Epoch 10 composite train-obj: 0.152520
            No improvement (0.1479), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.3530, mae: 0.4026, huber: 0.1524, swd: 0.1180, ept: 53.2126
    Epoch [11/50], Val Losses: mse: 0.3473, mae: 0.3847, huber: 0.1465, swd: 0.1089, ept: 53.0166
    Epoch [11/50], Test Losses: mse: 0.4297, mae: 0.4377, huber: 0.1808, swd: 0.1345, ept: 42.6134
      Epoch 11 composite train-obj: 0.152391
            Val objective improved 0.1465 → 0.1465, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.3525, mae: 0.4022, huber: 0.1522, swd: 0.1178, ept: 53.2683
    Epoch [12/50], Val Losses: mse: 0.3471, mae: 0.3834, huber: 0.1460, swd: 0.1081, ept: 53.2718
    Epoch [12/50], Test Losses: mse: 0.4320, mae: 0.4398, huber: 0.1817, swd: 0.1374, ept: 42.7942
      Epoch 12 composite train-obj: 0.152203
            Val objective improved 0.1465 → 0.1460, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.3526, mae: 0.4021, huber: 0.1522, swd: 0.1179, ept: 53.3601
    Epoch [13/50], Val Losses: mse: 0.3490, mae: 0.3858, huber: 0.1471, swd: 0.1109, ept: 53.1260
    Epoch [13/50], Test Losses: mse: 0.4302, mae: 0.4383, huber: 0.1809, swd: 0.1345, ept: 42.8954
      Epoch 13 composite train-obj: 0.152212
            No improvement (0.1471), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.3527, mae: 0.4019, huber: 0.1522, swd: 0.1180, ept: 53.3660
    Epoch [14/50], Val Losses: mse: 0.3477, mae: 0.3833, huber: 0.1462, swd: 0.1078, ept: 53.3112
    Epoch [14/50], Test Losses: mse: 0.4304, mae: 0.4378, huber: 0.1808, swd: 0.1353, ept: 42.9027
      Epoch 14 composite train-obj: 0.152191
            No improvement (0.1462), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.3523, mae: 0.4019, huber: 0.1521, swd: 0.1178, ept: 53.3839
    Epoch [15/50], Val Losses: mse: 0.3459, mae: 0.3831, huber: 0.1457, swd: 0.1063, ept: 53.2813
    Epoch [15/50], Test Losses: mse: 0.4302, mae: 0.4377, huber: 0.1809, swd: 0.1352, ept: 42.6736
      Epoch 15 composite train-obj: 0.152093
            Val objective improved 0.1460 → 0.1457, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 0.3524, mae: 0.4018, huber: 0.1521, swd: 0.1177, ept: 53.3789
    Epoch [16/50], Val Losses: mse: 0.3488, mae: 0.3837, huber: 0.1466, swd: 0.1090, ept: 52.9926
    Epoch [16/50], Test Losses: mse: 0.4309, mae: 0.4381, huber: 0.1811, swd: 0.1381, ept: 42.8188
      Epoch 16 composite train-obj: 0.152075
            No improvement (0.1466), counter 1/5
    Epoch [17/50], Train Losses: mse: 0.3522, mae: 0.4016, huber: 0.1520, swd: 0.1178, ept: 53.3937
    Epoch [17/50], Val Losses: mse: 0.3466, mae: 0.3829, huber: 0.1459, swd: 0.1080, ept: 53.3295
    Epoch [17/50], Test Losses: mse: 0.4297, mae: 0.4374, huber: 0.1806, swd: 0.1364, ept: 42.9059
      Epoch 17 composite train-obj: 0.152026
            No improvement (0.1459), counter 2/5
    Epoch [18/50], Train Losses: mse: 0.3520, mae: 0.4015, huber: 0.1519, swd: 0.1178, ept: 53.4031
    Epoch [18/50], Val Losses: mse: 0.3489, mae: 0.3835, huber: 0.1464, swd: 0.1074, ept: 53.4278
    Epoch [18/50], Test Losses: mse: 0.4312, mae: 0.4373, huber: 0.1809, swd: 0.1328, ept: 42.7937
      Epoch 18 composite train-obj: 0.151937
            No improvement (0.1464), counter 3/5
    Epoch [19/50], Train Losses: mse: 0.3522, mae: 0.4015, huber: 0.1520, swd: 0.1179, ept: 53.4300
    Epoch [19/50], Val Losses: mse: 0.3467, mae: 0.3814, huber: 0.1454, swd: 0.1063, ept: 53.4312
    Epoch [19/50], Test Losses: mse: 0.4319, mae: 0.4382, huber: 0.1813, swd: 0.1364, ept: 42.8225
      Epoch 19 composite train-obj: 0.151987
            Val objective improved 0.1457 → 0.1454, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 0.3520, mae: 0.4014, huber: 0.1519, swd: 0.1176, ept: 53.4476
    Epoch [20/50], Val Losses: mse: 0.3475, mae: 0.3830, huber: 0.1462, swd: 0.1062, ept: 53.3325
    Epoch [20/50], Test Losses: mse: 0.4300, mae: 0.4367, huber: 0.1804, swd: 0.1328, ept: 42.8702
      Epoch 20 composite train-obj: 0.151911
            No improvement (0.1462), counter 1/5
    Epoch [21/50], Train Losses: mse: 0.3523, mae: 0.4016, huber: 0.1521, swd: 0.1177, ept: 53.4448
    Epoch [21/50], Val Losses: mse: 0.3436, mae: 0.3813, huber: 0.1450, swd: 0.1063, ept: 53.2952
    Epoch [21/50], Test Losses: mse: 0.4300, mae: 0.4376, huber: 0.1808, swd: 0.1373, ept: 42.6177
      Epoch 21 composite train-obj: 0.152057
            Val objective improved 0.1454 → 0.1450, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 0.3518, mae: 0.4013, huber: 0.1519, swd: 0.1178, ept: 53.4959
    Epoch [22/50], Val Losses: mse: 0.3446, mae: 0.3809, huber: 0.1448, swd: 0.1049, ept: 53.6307
    Epoch [22/50], Test Losses: mse: 0.4303, mae: 0.4376, huber: 0.1807, swd: 0.1351, ept: 42.6977
      Epoch 22 composite train-obj: 0.151882
            Val objective improved 0.1450 → 0.1448, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 0.3520, mae: 0.4015, huber: 0.1519, swd: 0.1175, ept: 53.4334
    Epoch [23/50], Val Losses: mse: 0.3446, mae: 0.3820, huber: 0.1453, swd: 0.1085, ept: 53.5470
    Epoch [23/50], Test Losses: mse: 0.4286, mae: 0.4370, huber: 0.1803, swd: 0.1374, ept: 42.7723
      Epoch 23 composite train-obj: 0.151929
            No improvement (0.1453), counter 1/5
    Epoch [24/50], Train Losses: mse: 0.3522, mae: 0.4014, huber: 0.1520, swd: 0.1178, ept: 53.4568
    Epoch [24/50], Val Losses: mse: 0.3460, mae: 0.3818, huber: 0.1454, swd: 0.1069, ept: 53.5657
    Epoch [24/50], Test Losses: mse: 0.4309, mae: 0.4381, huber: 0.1811, swd: 0.1364, ept: 42.8063
      Epoch 24 composite train-obj: 0.151966
            No improvement (0.1454), counter 2/5
    Epoch [25/50], Train Losses: mse: 0.3522, mae: 0.4014, huber: 0.1520, swd: 0.1177, ept: 53.3887
    Epoch [25/50], Val Losses: mse: 0.3476, mae: 0.3849, huber: 0.1467, swd: 0.1120, ept: 53.1364
    Epoch [25/50], Test Losses: mse: 0.4282, mae: 0.4366, huber: 0.1803, swd: 0.1382, ept: 42.5651
      Epoch 25 composite train-obj: 0.151992
            No improvement (0.1467), counter 3/5
    Epoch [26/50], Train Losses: mse: 0.3521, mae: 0.4013, huber: 0.1519, swd: 0.1180, ept: 53.4560
    Epoch [26/50], Val Losses: mse: 0.3480, mae: 0.3843, huber: 0.1465, swd: 0.1104, ept: 53.1840
    Epoch [26/50], Test Losses: mse: 0.4281, mae: 0.4360, huber: 0.1800, swd: 0.1354, ept: 42.7570
      Epoch 26 composite train-obj: 0.151941
            No improvement (0.1465), counter 4/5
    Epoch [27/50], Train Losses: mse: 0.3519, mae: 0.4012, huber: 0.1518, swd: 0.1177, ept: 53.5406
    Epoch [27/50], Val Losses: mse: 0.3451, mae: 0.3818, huber: 0.1454, swd: 0.1083, ept: 53.4542
    Epoch [27/50], Test Losses: mse: 0.4294, mae: 0.4372, huber: 0.1806, swd: 0.1369, ept: 42.9307
      Epoch 27 composite train-obj: 0.151831
    Epoch [27/50], Test Losses: mse: 0.4303, mae: 0.4376, huber: 0.1807, swd: 0.1351, ept: 42.6977
    Best round's Test MSE: 0.4303, MAE: 0.4376, SWD: 0.1351
    Best round's Validation MSE: 0.3446, MAE: 0.3809, SWD: 0.1049
    Best round's Test verification MSE : 0.4303, MAE: 0.4376, SWD: 0.1351
    Time taken: 29.24 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4989, mae: 0.4935, huber: 0.2094, swd: 0.1381, ept: 37.2603
    Epoch [1/50], Val Losses: mse: 0.3782, mae: 0.4168, huber: 0.1608, swd: 0.1142, ept: 47.3318
    Epoch [1/50], Test Losses: mse: 0.4741, mae: 0.4757, huber: 0.2006, swd: 0.1426, ept: 37.6722
      Epoch 1 composite train-obj: 0.209367
            Val objective improved inf → 0.1608, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3750, mae: 0.4224, huber: 0.1622, swd: 0.1180, ept: 48.9824
    Epoch [2/50], Val Losses: mse: 0.3629, mae: 0.4011, huber: 0.1535, swd: 0.1135, ept: 50.3213
    Epoch [2/50], Test Losses: mse: 0.4511, mae: 0.4582, huber: 0.1907, swd: 0.1404, ept: 40.4574
      Epoch 2 composite train-obj: 0.162229
            Val objective improved 0.1608 → 0.1535, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3643, mae: 0.4132, huber: 0.1574, swd: 0.1159, ept: 51.0393
    Epoch [3/50], Val Losses: mse: 0.3543, mae: 0.3923, huber: 0.1494, swd: 0.1069, ept: 51.8409
    Epoch [3/50], Test Losses: mse: 0.4428, mae: 0.4504, huber: 0.1867, swd: 0.1333, ept: 41.4037
      Epoch 3 composite train-obj: 0.157445
            Val objective improved 0.1535 → 0.1494, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3590, mae: 0.4087, huber: 0.1552, swd: 0.1139, ept: 51.8417
    Epoch [4/50], Val Losses: mse: 0.3535, mae: 0.3909, huber: 0.1492, swd: 0.1110, ept: 52.3122
    Epoch [4/50], Test Losses: mse: 0.4369, mae: 0.4456, huber: 0.1843, swd: 0.1359, ept: 42.0416
      Epoch 4 composite train-obj: 0.155167
            Val objective improved 0.1494 → 0.1492, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3571, mae: 0.4067, huber: 0.1543, swd: 0.1136, ept: 52.2908
    Epoch [5/50], Val Losses: mse: 0.3506, mae: 0.3882, huber: 0.1477, swd: 0.1101, ept: 52.6461
    Epoch [5/50], Test Losses: mse: 0.4356, mae: 0.4443, huber: 0.1837, swd: 0.1362, ept: 42.2669
      Epoch 5 composite train-obj: 0.154256
            Val objective improved 0.1492 → 0.1477, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3554, mae: 0.4050, huber: 0.1535, swd: 0.1131, ept: 52.6329
    Epoch [6/50], Val Losses: mse: 0.3501, mae: 0.3876, huber: 0.1476, swd: 0.1111, ept: 52.7912
    Epoch [6/50], Test Losses: mse: 0.4327, mae: 0.4418, huber: 0.1823, swd: 0.1351, ept: 42.5358
      Epoch 6 composite train-obj: 0.153461
            Val objective improved 0.1477 → 0.1476, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3549, mae: 0.4046, huber: 0.1533, swd: 0.1133, ept: 52.7684
    Epoch [7/50], Val Losses: mse: 0.3497, mae: 0.3860, huber: 0.1472, swd: 0.1061, ept: 52.7778
    Epoch [7/50], Test Losses: mse: 0.4331, mae: 0.4409, huber: 0.1821, swd: 0.1327, ept: 42.4672
      Epoch 7 composite train-obj: 0.153265
            Val objective improved 0.1476 → 0.1472, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3543, mae: 0.4038, huber: 0.1530, swd: 0.1130, ept: 52.9601
    Epoch [8/50], Val Losses: mse: 0.3500, mae: 0.3856, huber: 0.1472, swd: 0.1076, ept: 53.1225
    Epoch [8/50], Test Losses: mse: 0.4331, mae: 0.4401, huber: 0.1819, swd: 0.1330, ept: 42.5528
      Epoch 8 composite train-obj: 0.152952
            No improvement (0.1472), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3536, mae: 0.4031, huber: 0.1527, swd: 0.1127, ept: 53.0668
    Epoch [9/50], Val Losses: mse: 0.3468, mae: 0.3856, huber: 0.1466, swd: 0.1118, ept: 52.9475
    Epoch [9/50], Test Losses: mse: 0.4293, mae: 0.4388, huber: 0.1810, swd: 0.1368, ept: 42.5461
      Epoch 9 composite train-obj: 0.152663
            Val objective improved 0.1472 → 0.1466, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.3533, mae: 0.4029, huber: 0.1525, swd: 0.1131, ept: 53.2001
    Epoch [10/50], Val Losses: mse: 0.3488, mae: 0.3845, huber: 0.1467, swd: 0.1084, ept: 53.1194
    Epoch [10/50], Test Losses: mse: 0.4318, mae: 0.4393, huber: 0.1815, swd: 0.1345, ept: 42.7698
      Epoch 10 composite train-obj: 0.152517
            No improvement (0.1467), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.3528, mae: 0.4024, huber: 0.1523, swd: 0.1125, ept: 53.2289
    Epoch [11/50], Val Losses: mse: 0.3473, mae: 0.3839, huber: 0.1463, swd: 0.1096, ept: 53.2032
    Epoch [11/50], Test Losses: mse: 0.4313, mae: 0.4390, huber: 0.1813, swd: 0.1355, ept: 42.7186
      Epoch 11 composite train-obj: 0.152306
            Val objective improved 0.1466 → 0.1463, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.3528, mae: 0.4023, huber: 0.1523, swd: 0.1130, ept: 53.2533
    Epoch [12/50], Val Losses: mse: 0.3470, mae: 0.3845, huber: 0.1464, swd: 0.1112, ept: 53.2641
    Epoch [12/50], Test Losses: mse: 0.4302, mae: 0.4387, huber: 0.1811, swd: 0.1358, ept: 42.7871
      Epoch 12 composite train-obj: 0.152315
            No improvement (0.1464), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.3526, mae: 0.4020, huber: 0.1522, swd: 0.1129, ept: 53.3575
    Epoch [13/50], Val Losses: mse: 0.3475, mae: 0.3846, huber: 0.1465, swd: 0.1091, ept: 53.3483
    Epoch [13/50], Test Losses: mse: 0.4314, mae: 0.4396, huber: 0.1815, swd: 0.1358, ept: 42.6343
      Epoch 13 composite train-obj: 0.152188
            No improvement (0.1465), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.3524, mae: 0.4021, huber: 0.1522, swd: 0.1128, ept: 53.3168
    Epoch [14/50], Val Losses: mse: 0.3458, mae: 0.3833, huber: 0.1458, swd: 0.1092, ept: 53.3764
    Epoch [14/50], Test Losses: mse: 0.4299, mae: 0.4377, huber: 0.1808, swd: 0.1345, ept: 42.6252
      Epoch 14 composite train-obj: 0.152166
            Val objective improved 0.1463 → 0.1458, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.3529, mae: 0.4020, huber: 0.1522, swd: 0.1130, ept: 53.3405
    Epoch [15/50], Val Losses: mse: 0.3476, mae: 0.3841, huber: 0.1466, swd: 0.1115, ept: 53.3013
    Epoch [15/50], Test Losses: mse: 0.4295, mae: 0.4380, huber: 0.1808, swd: 0.1368, ept: 42.6662
      Epoch 15 composite train-obj: 0.152241
            No improvement (0.1466), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.3524, mae: 0.4017, huber: 0.1521, swd: 0.1130, ept: 53.3910
    Epoch [16/50], Val Losses: mse: 0.3489, mae: 0.3841, huber: 0.1467, swd: 0.1114, ept: 53.4775
    Epoch [16/50], Test Losses: mse: 0.4304, mae: 0.4377, huber: 0.1809, swd: 0.1365, ept: 42.6745
      Epoch 16 composite train-obj: 0.152071
            No improvement (0.1467), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.3525, mae: 0.4018, huber: 0.1521, swd: 0.1128, ept: 53.4037
    Epoch [17/50], Val Losses: mse: 0.3470, mae: 0.3849, huber: 0.1466, swd: 0.1135, ept: 53.0731
    Epoch [17/50], Test Losses: mse: 0.4275, mae: 0.4372, huber: 0.1803, swd: 0.1370, ept: 42.6646
      Epoch 17 composite train-obj: 0.152143
            No improvement (0.1466), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.3525, mae: 0.4017, huber: 0.1521, swd: 0.1131, ept: 53.4145
    Epoch [18/50], Val Losses: mse: 0.3455, mae: 0.3836, huber: 0.1460, swd: 0.1110, ept: 53.1305
    Epoch [18/50], Test Losses: mse: 0.4281, mae: 0.4371, huber: 0.1802, swd: 0.1356, ept: 42.7023
      Epoch 18 composite train-obj: 0.152114
            No improvement (0.1460), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.3523, mae: 0.4017, huber: 0.1520, swd: 0.1127, ept: 53.4680
    Epoch [19/50], Val Losses: mse: 0.3455, mae: 0.3819, huber: 0.1455, swd: 0.1060, ept: 53.3318
    Epoch [19/50], Test Losses: mse: 0.4306, mae: 0.4381, huber: 0.1810, swd: 0.1344, ept: 42.7804
      Epoch 19 composite train-obj: 0.152027
            Val objective improved 0.1458 → 0.1455, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 0.3521, mae: 0.4015, huber: 0.1520, swd: 0.1127, ept: 53.3931
    Epoch [20/50], Val Losses: mse: 0.3458, mae: 0.3840, huber: 0.1460, swd: 0.1108, ept: 53.3708
    Epoch [20/50], Test Losses: mse: 0.4290, mae: 0.4375, huber: 0.1806, swd: 0.1349, ept: 42.6551
      Epoch 20 composite train-obj: 0.151973
            No improvement (0.1460), counter 1/5
    Epoch [21/50], Train Losses: mse: 0.3519, mae: 0.4015, huber: 0.1519, swd: 0.1131, ept: 53.4443
    Epoch [21/50], Val Losses: mse: 0.3458, mae: 0.3825, huber: 0.1456, swd: 0.1080, ept: 53.5613
    Epoch [21/50], Test Losses: mse: 0.4296, mae: 0.4376, huber: 0.1807, swd: 0.1351, ept: 42.8824
      Epoch 21 composite train-obj: 0.151923
            No improvement (0.1456), counter 2/5
    Epoch [22/50], Train Losses: mse: 0.3520, mae: 0.4014, huber: 0.1520, swd: 0.1125, ept: 53.4219
    Epoch [22/50], Val Losses: mse: 0.3451, mae: 0.3822, huber: 0.1454, swd: 0.1087, ept: 53.3492
    Epoch [22/50], Test Losses: mse: 0.4292, mae: 0.4372, huber: 0.1805, swd: 0.1345, ept: 42.8082
      Epoch 22 composite train-obj: 0.151963
            Val objective improved 0.1455 → 0.1454, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 0.3520, mae: 0.4013, huber: 0.1519, swd: 0.1128, ept: 53.5068
    Epoch [23/50], Val Losses: mse: 0.3462, mae: 0.3822, huber: 0.1457, swd: 0.1080, ept: 53.4612
    Epoch [23/50], Test Losses: mse: 0.4297, mae: 0.4367, huber: 0.1805, swd: 0.1334, ept: 42.7853
      Epoch 23 composite train-obj: 0.151899
            No improvement (0.1457), counter 1/5
    Epoch [24/50], Train Losses: mse: 0.3520, mae: 0.4013, huber: 0.1519, swd: 0.1129, ept: 53.4619
    Epoch [24/50], Val Losses: mse: 0.3438, mae: 0.3807, huber: 0.1448, swd: 0.1054, ept: 53.3823
    Epoch [24/50], Test Losses: mse: 0.4295, mae: 0.4368, huber: 0.1804, swd: 0.1332, ept: 42.8023
      Epoch 24 composite train-obj: 0.151904
            Val objective improved 0.1454 → 0.1448, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 0.3522, mae: 0.4013, huber: 0.1519, swd: 0.1126, ept: 53.4894
    Epoch [25/50], Val Losses: mse: 0.3463, mae: 0.3812, huber: 0.1451, swd: 0.1029, ept: 53.6787
    Epoch [25/50], Test Losses: mse: 0.4335, mae: 0.4391, huber: 0.1817, swd: 0.1308, ept: 42.8712
      Epoch 25 composite train-obj: 0.151937
            No improvement (0.1451), counter 1/5
    Epoch [26/50], Train Losses: mse: 0.3520, mae: 0.4013, huber: 0.1519, swd: 0.1126, ept: 53.4664
    Epoch [26/50], Val Losses: mse: 0.3446, mae: 0.3825, huber: 0.1454, swd: 0.1097, ept: 53.3989
    Epoch [26/50], Test Losses: mse: 0.4282, mae: 0.4365, huber: 0.1801, swd: 0.1341, ept: 42.8211
      Epoch 26 composite train-obj: 0.151936
            No improvement (0.1454), counter 2/5
    Epoch [27/50], Train Losses: mse: 0.3520, mae: 0.4014, huber: 0.1519, swd: 0.1129, ept: 53.5034
    Epoch [27/50], Val Losses: mse: 0.3473, mae: 0.3830, huber: 0.1462, swd: 0.1087, ept: 53.3699
    Epoch [27/50], Test Losses: mse: 0.4294, mae: 0.4364, huber: 0.1804, swd: 0.1333, ept: 42.7692
      Epoch 27 composite train-obj: 0.151927
            No improvement (0.1462), counter 3/5
    Epoch [28/50], Train Losses: mse: 0.3521, mae: 0.4012, huber: 0.1519, swd: 0.1123, ept: 53.4452
    Epoch [28/50], Val Losses: mse: 0.3458, mae: 0.3828, huber: 0.1458, swd: 0.1088, ept: 53.3157
    Epoch [28/50], Test Losses: mse: 0.4296, mae: 0.4374, huber: 0.1807, swd: 0.1359, ept: 42.8661
      Epoch 28 composite train-obj: 0.151898
            No improvement (0.1458), counter 4/5
    Epoch [29/50], Train Losses: mse: 0.3519, mae: 0.4012, huber: 0.1519, swd: 0.1131, ept: 53.4537
    Epoch [29/50], Val Losses: mse: 0.3459, mae: 0.3832, huber: 0.1458, swd: 0.1085, ept: 53.5497
    Epoch [29/50], Test Losses: mse: 0.4280, mae: 0.4367, huber: 0.1801, swd: 0.1336, ept: 42.7433
      Epoch 29 composite train-obj: 0.151868
    Epoch [29/50], Test Losses: mse: 0.4295, mae: 0.4368, huber: 0.1804, swd: 0.1332, ept: 42.8023
    Best round's Test MSE: 0.4295, MAE: 0.4368, SWD: 0.1332
    Best round's Validation MSE: 0.3438, MAE: 0.3807, SWD: 0.1054
    Best round's Test verification MSE : 0.4295, MAE: 0.4368, SWD: 0.1332
    Time taken: 29.00 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq96_pred96_20250510_1807)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4304 ± 0.0007
      mae: 0.4376 ± 0.0006
      huber: 0.1808 ± 0.0003
      swd: 0.1365 ± 0.0034
      ept: 42.7621 ± 0.0460
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3441 ± 0.0003
      mae: 0.3808 ± 0.0001
      huber: 0.1447 ± 0.0001
      swd: 0.1064 ± 0.0017
      ept: 53.5103 ± 0.1016
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 79.72 seconds
    
    Experiment complete: DLinear_etth1_seq96_pred96_20250510_1807
    Model: DLinear
    Dataset: etth1
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.5737, mae: 0.5293, huber: 0.2357, swd: 0.1819, ept: 52.6714
    Epoch [1/50], Val Losses: mse: 0.4357, mae: 0.4510, huber: 0.1829, swd: 0.1431, ept: 63.0058
    Epoch [1/50], Test Losses: mse: 0.5281, mae: 0.5108, huber: 0.2233, swd: 0.1637, ept: 56.6048
      Epoch 1 composite train-obj: 0.235701
            Val objective improved inf → 0.1829, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4592, mae: 0.4657, huber: 0.1927, swd: 0.1638, ept: 71.7227
    Epoch [2/50], Val Losses: mse: 0.4210, mae: 0.4383, huber: 0.1764, swd: 0.1431, ept: 67.4450
    Epoch [2/50], Test Losses: mse: 0.5049, mae: 0.4942, huber: 0.2136, swd: 0.1590, ept: 60.5476
      Epoch 2 composite train-obj: 0.192710
            Val objective improved 0.1829 → 0.1764, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4489, mae: 0.4576, huber: 0.1883, swd: 0.1614, ept: 75.2870
    Epoch [3/50], Val Losses: mse: 0.4157, mae: 0.4292, huber: 0.1726, swd: 0.1352, ept: 71.3566
    Epoch [3/50], Test Losses: mse: 0.5003, mae: 0.4890, huber: 0.2111, swd: 0.1534, ept: 61.9642
      Epoch 3 composite train-obj: 0.188330
            Val objective improved 0.1764 → 0.1726, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4451, mae: 0.4542, huber: 0.1866, swd: 0.1604, ept: 76.7386
    Epoch [4/50], Val Losses: mse: 0.4113, mae: 0.4275, huber: 0.1715, swd: 0.1369, ept: 70.9695
    Epoch [4/50], Test Losses: mse: 0.4935, mae: 0.4848, huber: 0.2084, swd: 0.1544, ept: 62.8339
      Epoch 4 composite train-obj: 0.186608
            Val objective improved 0.1726 → 0.1715, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4431, mae: 0.4523, huber: 0.1856, swd: 0.1599, ept: 77.5205
    Epoch [5/50], Val Losses: mse: 0.4132, mae: 0.4285, huber: 0.1723, swd: 0.1423, ept: 71.8589
    Epoch [5/50], Test Losses: mse: 0.4916, mae: 0.4833, huber: 0.2076, swd: 0.1557, ept: 63.3377
      Epoch 5 composite train-obj: 0.185643
            No improvement (0.1723), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4416, mae: 0.4512, huber: 0.1851, swd: 0.1599, ept: 78.0560
    Epoch [6/50], Val Losses: mse: 0.4095, mae: 0.4236, huber: 0.1701, swd: 0.1376, ept: 73.4114
    Epoch [6/50], Test Losses: mse: 0.4921, mae: 0.4819, huber: 0.2075, swd: 0.1557, ept: 63.5161
      Epoch 6 composite train-obj: 0.185129
            Val objective improved 0.1715 → 0.1701, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4410, mae: 0.4504, huber: 0.1848, swd: 0.1601, ept: 78.4119
    Epoch [7/50], Val Losses: mse: 0.4071, mae: 0.4204, huber: 0.1686, swd: 0.1334, ept: 74.5847
    Epoch [7/50], Test Losses: mse: 0.4908, mae: 0.4806, huber: 0.2067, swd: 0.1515, ept: 63.5850
      Epoch 7 composite train-obj: 0.184764
            Val objective improved 0.1701 → 0.1686, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4406, mae: 0.4499, huber: 0.1846, swd: 0.1597, ept: 78.7182
    Epoch [8/50], Val Losses: mse: 0.4117, mae: 0.4234, huber: 0.1706, swd: 0.1393, ept: 74.1377
    Epoch [8/50], Test Losses: mse: 0.4914, mae: 0.4811, huber: 0.2071, swd: 0.1533, ept: 63.6152
      Epoch 8 composite train-obj: 0.184584
            No improvement (0.1706), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4401, mae: 0.4495, huber: 0.1844, swd: 0.1598, ept: 78.7702
    Epoch [9/50], Val Losses: mse: 0.4119, mae: 0.4245, huber: 0.1710, swd: 0.1402, ept: 72.9569
    Epoch [9/50], Test Losses: mse: 0.4900, mae: 0.4801, huber: 0.2065, swd: 0.1561, ept: 63.9933
      Epoch 9 composite train-obj: 0.184376
            No improvement (0.1710), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.4396, mae: 0.4490, huber: 0.1841, swd: 0.1598, ept: 78.9640
    Epoch [10/50], Val Losses: mse: 0.4070, mae: 0.4226, huber: 0.1695, swd: 0.1409, ept: 73.3302
    Epoch [10/50], Test Losses: mse: 0.4871, mae: 0.4789, huber: 0.2056, swd: 0.1561, ept: 63.9902
      Epoch 10 composite train-obj: 0.184139
            No improvement (0.1695), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.4389, mae: 0.4486, huber: 0.1839, swd: 0.1596, ept: 79.1661
    Epoch [11/50], Val Losses: mse: 0.4056, mae: 0.4203, huber: 0.1686, swd: 0.1376, ept: 74.4987
    Epoch [11/50], Test Losses: mse: 0.4884, mae: 0.4791, huber: 0.2060, swd: 0.1556, ept: 64.0170
      Epoch 11 composite train-obj: 0.183903
            No improvement (0.1686), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.4390, mae: 0.4483, huber: 0.1838, swd: 0.1597, ept: 79.1624
    Epoch [12/50], Val Losses: mse: 0.4060, mae: 0.4196, huber: 0.1683, swd: 0.1347, ept: 73.3229
    Epoch [12/50], Test Losses: mse: 0.4905, mae: 0.4805, huber: 0.2067, swd: 0.1538, ept: 63.8317
      Epoch 12 composite train-obj: 0.183837
            Val objective improved 0.1686 → 0.1683, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.4388, mae: 0.4482, huber: 0.1838, swd: 0.1596, ept: 79.2763
    Epoch [13/50], Val Losses: mse: 0.4106, mae: 0.4232, huber: 0.1705, swd: 0.1432, ept: 73.1299
    Epoch [13/50], Test Losses: mse: 0.4899, mae: 0.4804, huber: 0.2067, swd: 0.1596, ept: 64.2144
      Epoch 13 composite train-obj: 0.183770
            No improvement (0.1705), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.4386, mae: 0.4482, huber: 0.1838, swd: 0.1594, ept: 79.2944
    Epoch [14/50], Val Losses: mse: 0.4088, mae: 0.4226, huber: 0.1701, swd: 0.1439, ept: 73.4565
    Epoch [14/50], Test Losses: mse: 0.4863, mae: 0.4778, huber: 0.2053, swd: 0.1574, ept: 63.9956
      Epoch 14 composite train-obj: 0.183779
            No improvement (0.1701), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.4388, mae: 0.4483, huber: 0.1838, swd: 0.1602, ept: 79.2950
    Epoch [15/50], Val Losses: mse: 0.4111, mae: 0.4241, huber: 0.1708, swd: 0.1407, ept: 73.1471
    Epoch [15/50], Test Losses: mse: 0.4899, mae: 0.4793, huber: 0.2062, swd: 0.1518, ept: 64.0179
      Epoch 15 composite train-obj: 0.183816
            No improvement (0.1708), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.4391, mae: 0.4482, huber: 0.1838, swd: 0.1596, ept: 79.3510
    Epoch [16/50], Val Losses: mse: 0.4124, mae: 0.4243, huber: 0.1712, swd: 0.1429, ept: 73.0152
    Epoch [16/50], Test Losses: mse: 0.4868, mae: 0.4774, huber: 0.2051, swd: 0.1551, ept: 64.1061
      Epoch 16 composite train-obj: 0.183848
            No improvement (0.1712), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.4383, mae: 0.4480, huber: 0.1836, swd: 0.1600, ept: 79.3473
    Epoch [17/50], Val Losses: mse: 0.4091, mae: 0.4220, huber: 0.1698, swd: 0.1384, ept: 73.5647
    Epoch [17/50], Test Losses: mse: 0.4872, mae: 0.4774, huber: 0.2052, swd: 0.1523, ept: 64.0293
      Epoch 17 composite train-obj: 0.183644
    Epoch [17/50], Test Losses: mse: 0.4905, mae: 0.4805, huber: 0.2067, swd: 0.1538, ept: 63.8317
    Best round's Test MSE: 0.4905, MAE: 0.4805, SWD: 0.1538
    Best round's Validation MSE: 0.4060, MAE: 0.4196, SWD: 0.1347
    Best round's Test verification MSE : 0.4905, MAE: 0.4805, SWD: 0.1538
    Time taken: 16.34 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5659, mae: 0.5263, huber: 0.2333, swd: 0.1803, ept: 52.9126
    Epoch [1/50], Val Losses: mse: 0.4390, mae: 0.4515, huber: 0.1837, swd: 0.1359, ept: 63.6810
    Epoch [1/50], Test Losses: mse: 0.5314, mae: 0.5122, huber: 0.2244, swd: 0.1561, ept: 56.5661
      Epoch 1 composite train-obj: 0.233315
            Val objective improved inf → 0.1837, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4589, mae: 0.4654, huber: 0.1925, swd: 0.1637, ept: 71.8192
    Epoch [2/50], Val Losses: mse: 0.4221, mae: 0.4380, huber: 0.1764, swd: 0.1382, ept: 68.2226
    Epoch [2/50], Test Losses: mse: 0.5064, mae: 0.4950, huber: 0.2142, swd: 0.1545, ept: 60.3838
      Epoch 2 composite train-obj: 0.192536
            Val objective improved 0.1837 → 0.1764, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4488, mae: 0.4576, huber: 0.1883, swd: 0.1621, ept: 75.2638
    Epoch [3/50], Val Losses: mse: 0.4132, mae: 0.4284, huber: 0.1719, swd: 0.1344, ept: 71.4570
    Epoch [3/50], Test Losses: mse: 0.5004, mae: 0.4894, huber: 0.2114, swd: 0.1548, ept: 61.8841
      Epoch 3 composite train-obj: 0.188274
            Val objective improved 0.1764 → 0.1719, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4452, mae: 0.4541, huber: 0.1866, swd: 0.1608, ept: 76.7212
    Epoch [4/50], Val Losses: mse: 0.4115, mae: 0.4270, huber: 0.1712, swd: 0.1344, ept: 71.8887
    Epoch [4/50], Test Losses: mse: 0.4942, mae: 0.4851, huber: 0.2087, swd: 0.1532, ept: 62.5855
      Epoch 4 composite train-obj: 0.186580
            Val objective improved 0.1719 → 0.1712, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4432, mae: 0.4525, huber: 0.1858, swd: 0.1608, ept: 77.4423
    Epoch [5/50], Val Losses: mse: 0.4089, mae: 0.4237, huber: 0.1698, swd: 0.1309, ept: 72.4794
    Epoch [5/50], Test Losses: mse: 0.4940, mae: 0.4835, huber: 0.2083, swd: 0.1519, ept: 63.3825
      Epoch 5 composite train-obj: 0.185774
            Val objective improved 0.1712 → 0.1698, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4419, mae: 0.4512, huber: 0.1852, swd: 0.1607, ept: 78.0166
    Epoch [6/50], Val Losses: mse: 0.4073, mae: 0.4212, huber: 0.1688, swd: 0.1279, ept: 74.7626
    Epoch [6/50], Test Losses: mse: 0.4916, mae: 0.4816, huber: 0.2072, swd: 0.1478, ept: 63.3649
      Epoch 6 composite train-obj: 0.185174
            Val objective improved 0.1698 → 0.1688, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4409, mae: 0.4504, huber: 0.1847, swd: 0.1597, ept: 78.3821
    Epoch [7/50], Val Losses: mse: 0.4053, mae: 0.4212, huber: 0.1686, swd: 0.1336, ept: 72.9364
    Epoch [7/50], Test Losses: mse: 0.4894, mae: 0.4809, huber: 0.2067, swd: 0.1530, ept: 63.5531
      Epoch 7 composite train-obj: 0.184734
            Val objective improved 0.1688 → 0.1686, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4402, mae: 0.4498, huber: 0.1845, swd: 0.1606, ept: 78.7279
    Epoch [8/50], Val Losses: mse: 0.4089, mae: 0.4224, huber: 0.1697, swd: 0.1328, ept: 73.3780
    Epoch [8/50], Test Losses: mse: 0.4906, mae: 0.4817, huber: 0.2071, swd: 0.1509, ept: 63.8023
      Epoch 8 composite train-obj: 0.184496
            No improvement (0.1697), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4401, mae: 0.4494, huber: 0.1843, swd: 0.1602, ept: 78.8466
    Epoch [9/50], Val Losses: mse: 0.4083, mae: 0.4214, huber: 0.1693, swd: 0.1343, ept: 74.7046
    Epoch [9/50], Test Losses: mse: 0.4882, mae: 0.4801, huber: 0.2061, swd: 0.1522, ept: 63.4356
      Epoch 9 composite train-obj: 0.184316
            No improvement (0.1693), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.4400, mae: 0.4493, huber: 0.1843, swd: 0.1607, ept: 78.8897
    Epoch [10/50], Val Losses: mse: 0.4070, mae: 0.4214, huber: 0.1691, swd: 0.1325, ept: 74.2790
    Epoch [10/50], Test Losses: mse: 0.4865, mae: 0.4780, huber: 0.2052, swd: 0.1504, ept: 64.0414
      Epoch 10 composite train-obj: 0.184302
            No improvement (0.1691), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.4396, mae: 0.4490, huber: 0.1842, swd: 0.1604, ept: 79.0034
    Epoch [11/50], Val Losses: mse: 0.4091, mae: 0.4239, huber: 0.1700, swd: 0.1340, ept: 72.2658
    Epoch [11/50], Test Losses: mse: 0.4882, mae: 0.4789, huber: 0.2056, swd: 0.1474, ept: 63.6051
      Epoch 11 composite train-obj: 0.184156
            No improvement (0.1700), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.4391, mae: 0.4486, huber: 0.1840, swd: 0.1598, ept: 79.1365
    Epoch [12/50], Val Losses: mse: 0.4100, mae: 0.4235, huber: 0.1704, swd: 0.1356, ept: 72.9897
    Epoch [12/50], Test Losses: mse: 0.4877, mae: 0.4784, huber: 0.2057, swd: 0.1515, ept: 63.8876
      Epoch 12 composite train-obj: 0.183962
    Epoch [12/50], Test Losses: mse: 0.4894, mae: 0.4809, huber: 0.2067, swd: 0.1530, ept: 63.5531
    Best round's Test MSE: 0.4894, MAE: 0.4809, SWD: 0.1530
    Best round's Validation MSE: 0.4053, MAE: 0.4212, SWD: 0.1336
    Best round's Test verification MSE : 0.4894, MAE: 0.4809, SWD: 0.1530
    Time taken: 11.43 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5719, mae: 0.5284, huber: 0.2351, swd: 0.1635, ept: 52.6839
    Epoch [1/50], Val Losses: mse: 0.4392, mae: 0.4529, huber: 0.1841, swd: 0.1333, ept: 62.4789
    Epoch [1/50], Test Losses: mse: 0.5284, mae: 0.5099, huber: 0.2232, swd: 0.1493, ept: 56.8433
      Epoch 1 composite train-obj: 0.235067
            Val objective improved inf → 0.1841, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4591, mae: 0.4655, huber: 0.1927, swd: 0.1510, ept: 71.9626
    Epoch [2/50], Val Losses: mse: 0.4230, mae: 0.4392, huber: 0.1769, swd: 0.1327, ept: 66.9986
    Epoch [2/50], Test Losses: mse: 0.5071, mae: 0.4958, huber: 0.2144, swd: 0.1484, ept: 60.6413
      Epoch 2 composite train-obj: 0.192652
            Val objective improved 0.1841 → 0.1769, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4491, mae: 0.4576, huber: 0.1883, swd: 0.1487, ept: 75.3064
    Epoch [3/50], Val Losses: mse: 0.4138, mae: 0.4294, huber: 0.1723, swd: 0.1300, ept: 71.0262
    Epoch [3/50], Test Losses: mse: 0.4991, mae: 0.4892, huber: 0.2108, swd: 0.1475, ept: 61.9947
      Epoch 3 composite train-obj: 0.188339
            Val objective improved 0.1769 → 0.1723, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4450, mae: 0.4543, huber: 0.1866, swd: 0.1485, ept: 76.6438
    Epoch [4/50], Val Losses: mse: 0.4099, mae: 0.4257, huber: 0.1705, swd: 0.1264, ept: 71.5984
    Epoch [4/50], Test Losses: mse: 0.4962, mae: 0.4857, huber: 0.2093, swd: 0.1422, ept: 62.5999
      Epoch 4 composite train-obj: 0.186629
            Val objective improved 0.1723 → 0.1705, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4426, mae: 0.4522, huber: 0.1856, swd: 0.1475, ept: 77.5242
    Epoch [5/50], Val Losses: mse: 0.4119, mae: 0.4245, huber: 0.1708, swd: 0.1300, ept: 73.4137
    Epoch [5/50], Test Losses: mse: 0.4942, mae: 0.4839, huber: 0.2085, swd: 0.1457, ept: 63.1372
      Epoch 5 composite train-obj: 0.185584
            No improvement (0.1708), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4418, mae: 0.4511, huber: 0.1851, swd: 0.1477, ept: 77.9730
    Epoch [6/50], Val Losses: mse: 0.4101, mae: 0.4246, huber: 0.1704, swd: 0.1285, ept: 72.3206
    Epoch [6/50], Test Losses: mse: 0.4915, mae: 0.4820, huber: 0.2072, swd: 0.1426, ept: 63.4726
      Epoch 6 composite train-obj: 0.185074
            Val objective improved 0.1705 → 0.1704, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4408, mae: 0.4503, huber: 0.1847, swd: 0.1474, ept: 78.2297
    Epoch [7/50], Val Losses: mse: 0.4112, mae: 0.4230, huber: 0.1704, swd: 0.1295, ept: 74.3507
    Epoch [7/50], Test Losses: mse: 0.4921, mae: 0.4819, huber: 0.2075, swd: 0.1422, ept: 63.5979
      Epoch 7 composite train-obj: 0.184718
            Val objective improved 0.1704 → 0.1704, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4405, mae: 0.4498, huber: 0.1845, swd: 0.1477, ept: 78.6568
    Epoch [8/50], Val Losses: mse: 0.4067, mae: 0.4220, huber: 0.1691, swd: 0.1293, ept: 74.0714
    Epoch [8/50], Test Losses: mse: 0.4887, mae: 0.4801, huber: 0.2064, swd: 0.1474, ept: 63.7643
      Epoch 8 composite train-obj: 0.184509
            Val objective improved 0.1704 → 0.1691, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4401, mae: 0.4493, huber: 0.1843, swd: 0.1475, ept: 78.9207
    Epoch [9/50], Val Losses: mse: 0.4066, mae: 0.4215, huber: 0.1689, swd: 0.1274, ept: 73.3305
    Epoch [9/50], Test Losses: mse: 0.4886, mae: 0.4792, huber: 0.2059, swd: 0.1421, ept: 63.8242
      Epoch 9 composite train-obj: 0.184295
            Val objective improved 0.1691 → 0.1689, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.4393, mae: 0.4489, huber: 0.1841, swd: 0.1473, ept: 79.0974
    Epoch [10/50], Val Losses: mse: 0.4074, mae: 0.4219, huber: 0.1693, swd: 0.1292, ept: 73.6396
    Epoch [10/50], Test Losses: mse: 0.4872, mae: 0.4785, huber: 0.2054, swd: 0.1430, ept: 63.9607
      Epoch 10 composite train-obj: 0.184052
            No improvement (0.1693), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.4391, mae: 0.4487, huber: 0.1839, swd: 0.1475, ept: 79.0399
    Epoch [11/50], Val Losses: mse: 0.4058, mae: 0.4191, huber: 0.1681, swd: 0.1275, ept: 74.9452
    Epoch [11/50], Test Losses: mse: 0.4894, mae: 0.4798, huber: 0.2063, swd: 0.1449, ept: 63.8711
      Epoch 11 composite train-obj: 0.183934
            Val objective improved 0.1689 → 0.1681, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.4394, mae: 0.4487, huber: 0.1840, swd: 0.1475, ept: 79.1485
    Epoch [12/50], Val Losses: mse: 0.4028, mae: 0.4190, huber: 0.1674, swd: 0.1280, ept: 74.3532
    Epoch [12/50], Test Losses: mse: 0.4858, mae: 0.4777, huber: 0.2050, swd: 0.1437, ept: 63.5761
      Epoch 12 composite train-obj: 0.184044
            Val objective improved 0.1681 → 0.1674, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.4389, mae: 0.4483, huber: 0.1838, swd: 0.1472, ept: 79.2387
    Epoch [13/50], Val Losses: mse: 0.4076, mae: 0.4212, huber: 0.1693, swd: 0.1325, ept: 73.6427
    Epoch [13/50], Test Losses: mse: 0.4879, mae: 0.4788, huber: 0.2060, swd: 0.1474, ept: 63.7266
      Epoch 13 composite train-obj: 0.183810
            No improvement (0.1693), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.4388, mae: 0.4483, huber: 0.1838, swd: 0.1478, ept: 79.2710
    Epoch [14/50], Val Losses: mse: 0.4043, mae: 0.4180, huber: 0.1675, swd: 0.1262, ept: 75.5232
    Epoch [14/50], Test Losses: mse: 0.4889, mae: 0.4792, huber: 0.2061, swd: 0.1454, ept: 63.7711
      Epoch 14 composite train-obj: 0.183815
            No improvement (0.1675), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.4386, mae: 0.4480, huber: 0.1837, swd: 0.1469, ept: 79.3358
    Epoch [15/50], Val Losses: mse: 0.4044, mae: 0.4201, huber: 0.1684, swd: 0.1315, ept: 73.6289
    Epoch [15/50], Test Losses: mse: 0.4859, mae: 0.4775, huber: 0.2051, swd: 0.1475, ept: 64.0282
      Epoch 15 composite train-obj: 0.183686
            No improvement (0.1684), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.4388, mae: 0.4481, huber: 0.1838, swd: 0.1479, ept: 79.2738
    Epoch [16/50], Val Losses: mse: 0.4046, mae: 0.4188, huber: 0.1679, swd: 0.1303, ept: 74.6935
    Epoch [16/50], Test Losses: mse: 0.4872, mae: 0.4788, huber: 0.2056, swd: 0.1448, ept: 63.6211
      Epoch 16 composite train-obj: 0.183750
            No improvement (0.1679), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.4387, mae: 0.4480, huber: 0.1837, swd: 0.1472, ept: 79.3654
    Epoch [17/50], Val Losses: mse: 0.4048, mae: 0.4201, huber: 0.1685, swd: 0.1331, ept: 73.8971
    Epoch [17/50], Test Losses: mse: 0.4880, mae: 0.4786, huber: 0.2059, swd: 0.1492, ept: 64.2001
      Epoch 17 composite train-obj: 0.183703
    Epoch [17/50], Test Losses: mse: 0.4858, mae: 0.4777, huber: 0.2050, swd: 0.1437, ept: 63.5761
    Best round's Test MSE: 0.4858, MAE: 0.4777, SWD: 0.1437
    Best round's Validation MSE: 0.4028, MAE: 0.4190, SWD: 0.1280
    Best round's Test verification MSE : 0.4858, MAE: 0.4777, SWD: 0.1437
    Time taken: 16.23 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq96_pred196_20250510_1814)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4886 ± 0.0020
      mae: 0.4797 ± 0.0014
      huber: 0.2061 ± 0.0008
      swd: 0.1502 ± 0.0046
      ept: 63.6536 ± 0.1262
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4047 ± 0.0014
      mae: 0.4199 ± 0.0010
      huber: 0.1681 ± 0.0005
      swd: 0.1321 ± 0.0029
      ept: 73.5375 ± 0.5980
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 44.02 seconds
    
    Experiment complete: DLinear_etth1_seq96_pred196_20250510_1814
    Model: DLinear
    Dataset: etth1
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 11
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6282, mae: 0.5576, huber: 0.2561, swd: 0.1990, ept: 63.9517
    Epoch [1/50], Val Losses: mse: 0.4771, mae: 0.4709, huber: 0.1972, swd: 0.1419, ept: 80.5796
    Epoch [1/50], Test Losses: mse: 0.5741, mae: 0.5435, huber: 0.2444, swd: 0.1654, ept: 76.1735
      Epoch 1 composite train-obj: 0.256107
            Val objective improved inf → 0.1972, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5237, mae: 0.5003, huber: 0.2170, swd: 0.1871, ept: 90.2072
    Epoch [2/50], Val Losses: mse: 0.4611, mae: 0.4550, huber: 0.1894, swd: 0.1423, ept: 88.1534
    Epoch [2/50], Test Losses: mse: 0.5538, mae: 0.5283, huber: 0.2358, swd: 0.1642, ept: 81.0300
      Epoch 2 composite train-obj: 0.216985
            Val objective improved 0.1972 → 0.1894, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5143, mae: 0.4931, huber: 0.2130, swd: 0.1861, ept: 95.2852
    Epoch [3/50], Val Losses: mse: 0.4540, mae: 0.4472, huber: 0.1857, swd: 0.1417, ept: 92.6243
    Epoch [3/50], Test Losses: mse: 0.5477, mae: 0.5239, huber: 0.2333, swd: 0.1643, ept: 82.1299
      Epoch 3 composite train-obj: 0.213041
            Val objective improved 0.1894 → 0.1857, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5108, mae: 0.4899, huber: 0.2114, swd: 0.1849, ept: 97.2008
    Epoch [4/50], Val Losses: mse: 0.4517, mae: 0.4438, huber: 0.1842, swd: 0.1397, ept: 93.0067
    Epoch [4/50], Test Losses: mse: 0.5423, mae: 0.5195, huber: 0.2308, swd: 0.1586, ept: 83.3991
      Epoch 4 composite train-obj: 0.211431
            Val objective improved 0.1857 → 0.1842, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5088, mae: 0.4882, huber: 0.2106, swd: 0.1847, ept: 98.5832
    Epoch [5/50], Val Losses: mse: 0.4476, mae: 0.4394, huber: 0.1821, swd: 0.1374, ept: 96.3777
    Epoch [5/50], Test Losses: mse: 0.5403, mae: 0.5180, huber: 0.2299, swd: 0.1610, ept: 83.4849
      Epoch 5 composite train-obj: 0.210577
            Val objective improved 0.1842 → 0.1821, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5076, mae: 0.4870, huber: 0.2100, swd: 0.1846, ept: 99.2921
    Epoch [6/50], Val Losses: mse: 0.4518, mae: 0.4426, huber: 0.1839, swd: 0.1405, ept: 93.3160
    Epoch [6/50], Test Losses: mse: 0.5369, mae: 0.5149, huber: 0.2283, swd: 0.1582, ept: 84.2156
      Epoch 6 composite train-obj: 0.209980
            No improvement (0.1839), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.5067, mae: 0.4863, huber: 0.2096, swd: 0.1844, ept: 99.8800
    Epoch [7/50], Val Losses: mse: 0.4490, mae: 0.4402, huber: 0.1827, swd: 0.1407, ept: 95.1023
    Epoch [7/50], Test Losses: mse: 0.5351, mae: 0.5143, huber: 0.2278, swd: 0.1599, ept: 84.4772
      Epoch 7 composite train-obj: 0.209642
            No improvement (0.1827), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.5063, mae: 0.4859, huber: 0.2095, swd: 0.1848, ept: 100.1377
    Epoch [8/50], Val Losses: mse: 0.4470, mae: 0.4402, huber: 0.1824, swd: 0.1410, ept: 94.9107
    Epoch [8/50], Test Losses: mse: 0.5347, mae: 0.5142, huber: 0.2276, swd: 0.1638, ept: 84.7858
      Epoch 8 composite train-obj: 0.209472
            No improvement (0.1824), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.5056, mae: 0.4853, huber: 0.2092, swd: 0.1848, ept: 100.3797
    Epoch [9/50], Val Losses: mse: 0.4455, mae: 0.4366, huber: 0.1811, swd: 0.1368, ept: 99.1958
    Epoch [9/50], Test Losses: mse: 0.5367, mae: 0.5140, huber: 0.2282, swd: 0.1599, ept: 83.9029
      Epoch 9 composite train-obj: 0.209165
            Val objective improved 0.1821 → 0.1811, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.5053, mae: 0.4852, huber: 0.2091, swd: 0.1845, ept: 100.5075
    Epoch [10/50], Val Losses: mse: 0.4438, mae: 0.4381, huber: 0.1812, swd: 0.1421, ept: 96.1919
    Epoch [10/50], Test Losses: mse: 0.5345, mae: 0.5139, huber: 0.2276, swd: 0.1676, ept: 84.5406
      Epoch 10 composite train-obj: 0.209064
            No improvement (0.1812), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.5049, mae: 0.4847, huber: 0.2089, swd: 0.1846, ept: 100.8452
    Epoch [11/50], Val Losses: mse: 0.4493, mae: 0.4394, huber: 0.1827, swd: 0.1425, ept: 96.3636
    Epoch [11/50], Test Losses: mse: 0.5376, mae: 0.5162, huber: 0.2288, swd: 0.1644, ept: 85.1300
      Epoch 11 composite train-obj: 0.208893
            No improvement (0.1827), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.5045, mae: 0.4846, huber: 0.2088, swd: 0.1848, ept: 100.9107
    Epoch [12/50], Val Losses: mse: 0.4510, mae: 0.4405, huber: 0.1834, swd: 0.1438, ept: 96.0144
    Epoch [12/50], Test Losses: mse: 0.5345, mae: 0.5131, huber: 0.2274, swd: 0.1634, ept: 85.1431
      Epoch 12 composite train-obj: 0.208755
            No improvement (0.1834), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.5044, mae: 0.4843, huber: 0.2087, swd: 0.1846, ept: 100.9366
    Epoch [13/50], Val Losses: mse: 0.4451, mae: 0.4353, huber: 0.1806, swd: 0.1368, ept: 98.7270
    Epoch [13/50], Test Losses: mse: 0.5336, mae: 0.5124, huber: 0.2269, swd: 0.1588, ept: 83.8779
      Epoch 13 composite train-obj: 0.208663
            Val objective improved 0.1811 → 0.1806, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.5042, mae: 0.4841, huber: 0.2086, swd: 0.1847, ept: 101.0851
    Epoch [14/50], Val Losses: mse: 0.4443, mae: 0.4350, huber: 0.1805, swd: 0.1392, ept: 99.1250
    Epoch [14/50], Test Losses: mse: 0.5373, mae: 0.5151, huber: 0.2286, swd: 0.1643, ept: 84.1107
      Epoch 14 composite train-obj: 0.208594
            Val objective improved 0.1806 → 0.1805, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.5039, mae: 0.4838, huber: 0.2084, swd: 0.1842, ept: 101.3092
    Epoch [15/50], Val Losses: mse: 0.4453, mae: 0.4354, huber: 0.1807, swd: 0.1391, ept: 99.3619
    Epoch [15/50], Test Losses: mse: 0.5385, mae: 0.5163, huber: 0.2291, swd: 0.1675, ept: 84.7859
      Epoch 15 composite train-obj: 0.208429
            No improvement (0.1807), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.5041, mae: 0.4842, huber: 0.2086, swd: 0.1851, ept: 101.2137
    Epoch [16/50], Val Losses: mse: 0.4529, mae: 0.4406, huber: 0.1837, swd: 0.1425, ept: 96.9097
    Epoch [16/50], Test Losses: mse: 0.5344, mae: 0.5127, huber: 0.2271, swd: 0.1591, ept: 85.4503
      Epoch 16 composite train-obj: 0.208616
            No improvement (0.1837), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.5042, mae: 0.4839, huber: 0.2085, swd: 0.1847, ept: 101.3113
    Epoch [17/50], Val Losses: mse: 0.4455, mae: 0.4372, huber: 0.1814, swd: 0.1392, ept: 98.3505
    Epoch [17/50], Test Losses: mse: 0.5350, mae: 0.5135, huber: 0.2276, swd: 0.1632, ept: 84.1016
      Epoch 17 composite train-obj: 0.208528
            No improvement (0.1814), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.5040, mae: 0.4838, huber: 0.2085, swd: 0.1842, ept: 101.2499
    Epoch [18/50], Val Losses: mse: 0.4495, mae: 0.4394, huber: 0.1829, swd: 0.1451, ept: 96.4058
    Epoch [18/50], Test Losses: mse: 0.5368, mae: 0.5152, huber: 0.2286, swd: 0.1666, ept: 84.9035
      Epoch 18 composite train-obj: 0.208491
            No improvement (0.1829), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.5041, mae: 0.4839, huber: 0.2085, swd: 0.1852, ept: 101.1869
    Epoch [19/50], Val Losses: mse: 0.4438, mae: 0.4346, huber: 0.1801, swd: 0.1358, ept: 98.6965
    Epoch [19/50], Test Losses: mse: 0.5305, mae: 0.5102, huber: 0.2255, swd: 0.1580, ept: 85.1088
      Epoch 19 composite train-obj: 0.208496
            Val objective improved 0.1805 → 0.1801, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 0.5044, mae: 0.4840, huber: 0.2086, swd: 0.1850, ept: 101.3661
    Epoch [20/50], Val Losses: mse: 0.4453, mae: 0.4360, huber: 0.1810, swd: 0.1409, ept: 97.2638
    Epoch [20/50], Test Losses: mse: 0.5352, mae: 0.5138, huber: 0.2278, swd: 0.1613, ept: 84.5993
      Epoch 20 composite train-obj: 0.208612
            No improvement (0.1810), counter 1/5
    Epoch [21/50], Train Losses: mse: 0.5039, mae: 0.4838, huber: 0.2084, swd: 0.1850, ept: 101.1462
    Epoch [21/50], Val Losses: mse: 0.4453, mae: 0.4382, huber: 0.1814, swd: 0.1410, ept: 95.3323
    Epoch [21/50], Test Losses: mse: 0.5336, mae: 0.5130, huber: 0.2271, swd: 0.1627, ept: 84.9175
      Epoch 21 composite train-obj: 0.208446
            No improvement (0.1814), counter 2/5
    Epoch [22/50], Train Losses: mse: 0.5037, mae: 0.4836, huber: 0.2084, swd: 0.1845, ept: 101.3471
    Epoch [22/50], Val Losses: mse: 0.4480, mae: 0.4388, huber: 0.1823, swd: 0.1418, ept: 95.3180
    Epoch [22/50], Test Losses: mse: 0.5335, mae: 0.5129, huber: 0.2270, swd: 0.1626, ept: 85.0312
      Epoch 22 composite train-obj: 0.208390
            No improvement (0.1823), counter 3/5
    Epoch [23/50], Train Losses: mse: 0.5036, mae: 0.4835, huber: 0.2083, swd: 0.1842, ept: 101.3472
    Epoch [23/50], Val Losses: mse: 0.4492, mae: 0.4396, huber: 0.1828, swd: 0.1428, ept: 95.8256
    Epoch [23/50], Test Losses: mse: 0.5317, mae: 0.5104, huber: 0.2260, swd: 0.1579, ept: 85.2096
      Epoch 23 composite train-obj: 0.208305
            No improvement (0.1828), counter 4/5
    Epoch [24/50], Train Losses: mse: 0.5036, mae: 0.4836, huber: 0.2083, swd: 0.1845, ept: 101.3738
    Epoch [24/50], Val Losses: mse: 0.4467, mae: 0.4362, huber: 0.1813, swd: 0.1418, ept: 99.1187
    Epoch [24/50], Test Losses: mse: 0.5389, mae: 0.5162, huber: 0.2293, swd: 0.1672, ept: 84.8825
      Epoch 24 composite train-obj: 0.208333
    Epoch [24/50], Test Losses: mse: 0.5305, mae: 0.5102, huber: 0.2255, swd: 0.1580, ept: 85.1088
    Best round's Test MSE: 0.5305, MAE: 0.5102, SWD: 0.1580
    Best round's Validation MSE: 0.4438, MAE: 0.4346, SWD: 0.1358
    Best round's Test verification MSE : 0.5305, MAE: 0.5102, SWD: 0.1580
    Time taken: 24.16 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6251, mae: 0.5559, huber: 0.2549, swd: 0.2031, ept: 64.8535
    Epoch [1/50], Val Losses: mse: 0.4776, mae: 0.4697, huber: 0.1971, swd: 0.1403, ept: 82.3445
    Epoch [1/50], Test Losses: mse: 0.5746, mae: 0.5419, huber: 0.2442, swd: 0.1610, ept: 76.2826
      Epoch 1 composite train-obj: 0.254949
            Val objective improved inf → 0.1971, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5231, mae: 0.4999, huber: 0.2168, swd: 0.1909, ept: 90.3445
    Epoch [2/50], Val Losses: mse: 0.4624, mae: 0.4539, huber: 0.1894, swd: 0.1416, ept: 89.8622
    Epoch [2/50], Test Losses: mse: 0.5599, mae: 0.5320, huber: 0.2382, swd: 0.1667, ept: 81.0588
      Epoch 2 composite train-obj: 0.216779
            Val objective improved 0.1971 → 0.1894, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5146, mae: 0.4930, huber: 0.2131, swd: 0.1891, ept: 95.2225
    Epoch [3/50], Val Losses: mse: 0.4536, mae: 0.4455, huber: 0.1851, swd: 0.1380, ept: 92.8043
    Epoch [3/50], Test Losses: mse: 0.5480, mae: 0.5237, huber: 0.2333, swd: 0.1614, ept: 81.6960
      Epoch 3 composite train-obj: 0.213066
            Val objective improved 0.1894 → 0.1851, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5111, mae: 0.4900, huber: 0.2115, swd: 0.1891, ept: 97.3496
    Epoch [4/50], Val Losses: mse: 0.4507, mae: 0.4434, huber: 0.1839, swd: 0.1396, ept: 92.9138
    Epoch [4/50], Test Losses: mse: 0.5426, mae: 0.5190, huber: 0.2308, swd: 0.1604, ept: 83.0543
      Epoch 4 composite train-obj: 0.211547
            Val objective improved 0.1851 → 0.1839, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5092, mae: 0.4884, huber: 0.2107, swd: 0.1886, ept: 98.5042
    Epoch [5/50], Val Losses: mse: 0.4454, mae: 0.4385, huber: 0.1814, swd: 0.1336, ept: 96.4912
    Epoch [5/50], Test Losses: mse: 0.5378, mae: 0.5154, huber: 0.2287, swd: 0.1556, ept: 83.2339
      Epoch 5 composite train-obj: 0.210709
            Val objective improved 0.1839 → 0.1814, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5075, mae: 0.4870, huber: 0.2100, swd: 0.1883, ept: 99.1893
    Epoch [6/50], Val Losses: mse: 0.4547, mae: 0.4440, huber: 0.1850, swd: 0.1422, ept: 94.2720
    Epoch [6/50], Test Losses: mse: 0.5422, mae: 0.5184, huber: 0.2305, swd: 0.1590, ept: 84.3601
      Epoch 6 composite train-obj: 0.209970
            No improvement (0.1850), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.5064, mae: 0.4862, huber: 0.2095, swd: 0.1881, ept: 99.7815
    Epoch [7/50], Val Losses: mse: 0.4516, mae: 0.4402, huber: 0.1832, swd: 0.1390, ept: 96.3615
    Epoch [7/50], Test Losses: mse: 0.5406, mae: 0.5173, huber: 0.2298, swd: 0.1619, ept: 84.2335
      Epoch 7 composite train-obj: 0.209525
            No improvement (0.1832), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.5058, mae: 0.4858, huber: 0.2093, swd: 0.1881, ept: 100.2710
    Epoch [8/50], Val Losses: mse: 0.4524, mae: 0.4447, huber: 0.1849, swd: 0.1469, ept: 93.4131
    Epoch [8/50], Test Losses: mse: 0.5370, mae: 0.5148, huber: 0.2285, swd: 0.1659, ept: 84.4386
      Epoch 8 composite train-obj: 0.209331
            No improvement (0.1849), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.5053, mae: 0.4852, huber: 0.2091, swd: 0.1882, ept: 100.4284
    Epoch [9/50], Val Losses: mse: 0.4462, mae: 0.4389, huber: 0.1818, swd: 0.1381, ept: 97.1970
    Epoch [9/50], Test Losses: mse: 0.5332, mae: 0.5129, huber: 0.2269, swd: 0.1612, ept: 84.4659
      Epoch 9 composite train-obj: 0.209092
            No improvement (0.1818), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.5054, mae: 0.4850, huber: 0.2091, swd: 0.1884, ept: 100.5574
    Epoch [10/50], Val Losses: mse: 0.4488, mae: 0.4407, huber: 0.1830, swd: 0.1413, ept: 94.9732
    Epoch [10/50], Test Losses: mse: 0.5359, mae: 0.5135, huber: 0.2279, swd: 0.1593, ept: 84.9430
      Epoch 10 composite train-obj: 0.209062
    Epoch [10/50], Test Losses: mse: 0.5378, mae: 0.5154, huber: 0.2287, swd: 0.1556, ept: 83.2339
    Best round's Test MSE: 0.5378, MAE: 0.5154, SWD: 0.1556
    Best round's Validation MSE: 0.4454, MAE: 0.4385, SWD: 0.1336
    Best round's Test verification MSE : 0.5378, MAE: 0.5154, SWD: 0.1556
    Time taken: 10.09 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6269, mae: 0.5563, huber: 0.2553, swd: 0.2004, ept: 64.8570
    Epoch [1/50], Val Losses: mse: 0.4753, mae: 0.4689, huber: 0.1964, swd: 0.1410, ept: 81.7727
    Epoch [1/50], Test Losses: mse: 0.5736, mae: 0.5427, huber: 0.2440, swd: 0.1654, ept: 76.3452
      Epoch 1 composite train-obj: 0.255296
            Val objective improved inf → 0.1964, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5232, mae: 0.5001, huber: 0.2169, swd: 0.1892, ept: 90.2397
    Epoch [2/50], Val Losses: mse: 0.4606, mae: 0.4543, huber: 0.1890, swd: 0.1426, ept: 88.5038
    Epoch [2/50], Test Losses: mse: 0.5545, mae: 0.5286, huber: 0.2360, swd: 0.1637, ept: 80.9089
      Epoch 2 composite train-obj: 0.216867
            Val objective improved 0.1964 → 0.1890, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5146, mae: 0.4929, huber: 0.2130, swd: 0.1876, ept: 95.2780
    Epoch [3/50], Val Losses: mse: 0.4559, mae: 0.4484, huber: 0.1864, swd: 0.1423, ept: 91.4043
    Epoch [3/50], Test Losses: mse: 0.5458, mae: 0.5226, huber: 0.2324, swd: 0.1641, ept: 82.6411
      Epoch 3 composite train-obj: 0.213009
            Val objective improved 0.1890 → 0.1864, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5109, mae: 0.4899, huber: 0.2115, swd: 0.1868, ept: 97.4879
    Epoch [4/50], Val Losses: mse: 0.4508, mae: 0.4436, huber: 0.1840, swd: 0.1427, ept: 93.9748
    Epoch [4/50], Test Losses: mse: 0.5456, mae: 0.5227, huber: 0.2325, swd: 0.1653, ept: 83.7299
      Epoch 4 composite train-obj: 0.211493
            Val objective improved 0.1864 → 0.1840, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5087, mae: 0.4883, huber: 0.2106, swd: 0.1871, ept: 98.5841
    Epoch [5/50], Val Losses: mse: 0.4475, mae: 0.4394, huber: 0.1821, swd: 0.1387, ept: 95.1373
    Epoch [5/50], Test Losses: mse: 0.5403, mae: 0.5171, huber: 0.2298, swd: 0.1607, ept: 83.5128
      Epoch 5 composite train-obj: 0.210592
            Val objective improved 0.1840 → 0.1821, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5075, mae: 0.4871, huber: 0.2100, swd: 0.1864, ept: 99.1381
    Epoch [6/50], Val Losses: mse: 0.4538, mae: 0.4441, huber: 0.1849, swd: 0.1439, ept: 94.0818
    Epoch [6/50], Test Losses: mse: 0.5423, mae: 0.5189, huber: 0.2307, swd: 0.1638, ept: 84.5358
      Epoch 6 composite train-obj: 0.209987
            No improvement (0.1849), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.5063, mae: 0.4863, huber: 0.2096, swd: 0.1866, ept: 99.7791
    Epoch [7/50], Val Losses: mse: 0.4461, mae: 0.4365, huber: 0.1811, swd: 0.1385, ept: 100.4084
    Epoch [7/50], Test Losses: mse: 0.5418, mae: 0.5180, huber: 0.2304, swd: 0.1618, ept: 83.0958
      Epoch 7 composite train-obj: 0.209562
            Val objective improved 0.1821 → 0.1811, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.5061, mae: 0.4857, huber: 0.2094, swd: 0.1867, ept: 100.0433
    Epoch [8/50], Val Losses: mse: 0.4509, mae: 0.4404, huber: 0.1832, swd: 0.1425, ept: 96.4740
    Epoch [8/50], Test Losses: mse: 0.5336, mae: 0.5134, huber: 0.2270, swd: 0.1612, ept: 84.8481
      Epoch 8 composite train-obj: 0.209351
            No improvement (0.1832), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.5055, mae: 0.4853, huber: 0.2091, swd: 0.1862, ept: 100.5622
    Epoch [9/50], Val Losses: mse: 0.4469, mae: 0.4395, huber: 0.1823, swd: 0.1444, ept: 96.1451
    Epoch [9/50], Test Losses: mse: 0.5359, mae: 0.5149, huber: 0.2282, swd: 0.1670, ept: 84.6368
      Epoch 9 composite train-obj: 0.209145
            No improvement (0.1823), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.5049, mae: 0.4849, huber: 0.2089, swd: 0.1870, ept: 100.6177
    Epoch [10/50], Val Losses: mse: 0.4483, mae: 0.4382, huber: 0.1822, swd: 0.1422, ept: 97.6601
    Epoch [10/50], Test Losses: mse: 0.5375, mae: 0.5151, huber: 0.2287, swd: 0.1607, ept: 85.1298
      Epoch 10 composite train-obj: 0.208913
            No improvement (0.1822), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.5051, mae: 0.4847, huber: 0.2089, swd: 0.1865, ept: 100.9321
    Epoch [11/50], Val Losses: mse: 0.4446, mae: 0.4349, huber: 0.1804, swd: 0.1373, ept: 98.9477
    Epoch [11/50], Test Losses: mse: 0.5376, mae: 0.5146, huber: 0.2285, swd: 0.1608, ept: 84.6068
      Epoch 11 composite train-obj: 0.208875
            Val objective improved 0.1811 → 0.1804, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.5049, mae: 0.4847, huber: 0.2089, swd: 0.1866, ept: 100.8609
    Epoch [12/50], Val Losses: mse: 0.4466, mae: 0.4394, huber: 0.1822, swd: 0.1461, ept: 96.9247
    Epoch [12/50], Test Losses: mse: 0.5329, mae: 0.5133, huber: 0.2271, swd: 0.1686, ept: 84.5798
      Epoch 12 composite train-obj: 0.208879
            No improvement (0.1822), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.5047, mae: 0.4844, huber: 0.2087, swd: 0.1868, ept: 101.0586
    Epoch [13/50], Val Losses: mse: 0.4457, mae: 0.4370, huber: 0.1812, swd: 0.1411, ept: 96.8356
    Epoch [13/50], Test Losses: mse: 0.5331, mae: 0.5120, huber: 0.2267, swd: 0.1620, ept: 85.1721
      Epoch 13 composite train-obj: 0.208744
            No improvement (0.1812), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.5045, mae: 0.4843, huber: 0.2087, swd: 0.1870, ept: 101.1377
    Epoch [14/50], Val Losses: mse: 0.4442, mae: 0.4341, huber: 0.1800, swd: 0.1384, ept: 100.2995
    Epoch [14/50], Test Losses: mse: 0.5348, mae: 0.5137, huber: 0.2275, swd: 0.1600, ept: 84.6872
      Epoch 14 composite train-obj: 0.208709
            Val objective improved 0.1804 → 0.1800, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.5041, mae: 0.4841, huber: 0.2086, swd: 0.1863, ept: 101.0529
    Epoch [15/50], Val Losses: mse: 0.4425, mae: 0.4342, huber: 0.1798, swd: 0.1386, ept: 99.6959
    Epoch [15/50], Test Losses: mse: 0.5356, mae: 0.5140, huber: 0.2279, swd: 0.1643, ept: 84.2665
      Epoch 15 composite train-obj: 0.208552
            Val objective improved 0.1800 → 0.1798, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 0.5047, mae: 0.4844, huber: 0.2088, swd: 0.1871, ept: 101.1862
    Epoch [16/50], Val Losses: mse: 0.4461, mae: 0.4361, huber: 0.1812, swd: 0.1403, ept: 97.1847
    Epoch [16/50], Test Losses: mse: 0.5386, mae: 0.5141, huber: 0.2287, swd: 0.1589, ept: 85.0978
      Epoch 16 composite train-obj: 0.208764
            No improvement (0.1812), counter 1/5
    Epoch [17/50], Train Losses: mse: 0.5043, mae: 0.4840, huber: 0.2086, swd: 0.1865, ept: 101.2113
    Epoch [17/50], Val Losses: mse: 0.4436, mae: 0.4357, huber: 0.1805, swd: 0.1432, ept: 98.2697
    Epoch [17/50], Test Losses: mse: 0.5369, mae: 0.5157, huber: 0.2287, swd: 0.1695, ept: 84.0749
      Epoch 17 composite train-obj: 0.208617
            No improvement (0.1805), counter 2/5
    Epoch [18/50], Train Losses: mse: 0.5045, mae: 0.4839, huber: 0.2086, swd: 0.1867, ept: 101.4971
    Epoch [18/50], Val Losses: mse: 0.4456, mae: 0.4364, huber: 0.1811, swd: 0.1432, ept: 96.8200
    Epoch [18/50], Test Losses: mse: 0.5359, mae: 0.5149, huber: 0.2281, swd: 0.1643, ept: 84.8521
      Epoch 18 composite train-obj: 0.208572
            No improvement (0.1811), counter 3/5
    Epoch [19/50], Train Losses: mse: 0.5040, mae: 0.4839, huber: 0.2085, swd: 0.1868, ept: 101.1718
    Epoch [19/50], Val Losses: mse: 0.4466, mae: 0.4369, huber: 0.1814, swd: 0.1421, ept: 99.5592
    Epoch [19/50], Test Losses: mse: 0.5346, mae: 0.5143, huber: 0.2277, swd: 0.1664, ept: 83.9296
      Epoch 19 composite train-obj: 0.208510
            No improvement (0.1814), counter 4/5
    Epoch [20/50], Train Losses: mse: 0.5038, mae: 0.4838, huber: 0.2084, swd: 0.1866, ept: 101.3895
    Epoch [20/50], Val Losses: mse: 0.4505, mae: 0.4415, huber: 0.1837, swd: 0.1475, ept: 94.5407
    Epoch [20/50], Test Losses: mse: 0.5360, mae: 0.5135, huber: 0.2280, swd: 0.1660, ept: 84.7799
      Epoch 20 composite train-obj: 0.208421
    Epoch [20/50], Test Losses: mse: 0.5356, mae: 0.5140, huber: 0.2279, swd: 0.1643, ept: 84.2665
    Best round's Test MSE: 0.5356, MAE: 0.5140, SWD: 0.1643
    Best round's Validation MSE: 0.4425, MAE: 0.4342, SWD: 0.1386
    Best round's Test verification MSE : 0.5356, MAE: 0.5140, SWD: 0.1643
    Time taken: 20.12 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq96_pred336_20250510_1815)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5347 ± 0.0031
      mae: 0.5132 ± 0.0022
      huber: 0.2274 ± 0.0013
      swd: 0.1593 ± 0.0037
      ept: 84.2031 ± 0.7667
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4439 ± 0.0012
      mae: 0.4358 ± 0.0019
      huber: 0.1804 ± 0.0007
      swd: 0.1360 ± 0.0020
      ept: 98.2946 ± 1.3388
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 54.39 seconds
    
    Experiment complete: DLinear_etth1_seq96_pred336_20250510_1815
    Model: DLinear
    Dataset: etth1
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
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 8
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7036, mae: 0.6009, huber: 0.2861, swd: 0.2498, ept: 73.0524
    Epoch [1/50], Val Losses: mse: 0.4775, mae: 0.4879, huber: 0.2035, swd: 0.1520, ept: 101.8724
    Epoch [1/50], Test Losses: mse: 0.6943, mae: 0.6161, huber: 0.2947, swd: 0.2076, ept: 114.2334
      Epoch 1 composite train-obj: 0.286101
            Val objective improved inf → 0.2035, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6103, mae: 0.5509, huber: 0.2512, swd: 0.2448, ept: 109.2011
    Epoch [2/50], Val Losses: mse: 0.4568, mae: 0.4683, huber: 0.1935, swd: 0.1440, ept: 123.8553
    Epoch [2/50], Test Losses: mse: 0.6763, mae: 0.6038, huber: 0.2875, swd: 0.2002, ept: 119.7069
      Epoch 2 composite train-obj: 0.251210
            Val objective improved 0.2035 → 0.1935, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6022, mae: 0.5448, huber: 0.2477, swd: 0.2436, ept: 116.7464
    Epoch [3/50], Val Losses: mse: 0.4587, mae: 0.4698, huber: 0.1941, swd: 0.1504, ept: 114.7313
    Epoch [3/50], Test Losses: mse: 0.6626, mae: 0.5960, huber: 0.2820, swd: 0.2045, ept: 125.9189
      Epoch 3 composite train-obj: 0.247736
            No improvement (0.1941), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5988, mae: 0.5421, huber: 0.2463, swd: 0.2428, ept: 120.3505
    Epoch [4/50], Val Losses: mse: 0.4599, mae: 0.4706, huber: 0.1944, swd: 0.1539, ept: 105.9106
    Epoch [4/50], Test Losses: mse: 0.6639, mae: 0.5953, huber: 0.2824, swd: 0.2000, ept: 127.8459
      Epoch 4 composite train-obj: 0.246264
            No improvement (0.1944), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5969, mae: 0.5405, huber: 0.2454, swd: 0.2426, ept: 122.2066
    Epoch [5/50], Val Losses: mse: 0.4564, mae: 0.4659, huber: 0.1927, swd: 0.1522, ept: 123.6973
    Epoch [5/50], Test Losses: mse: 0.6599, mae: 0.5941, huber: 0.2811, swd: 0.2080, ept: 126.5583
      Epoch 5 composite train-obj: 0.245415
            Val objective improved 0.1935 → 0.1927, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5959, mae: 0.5395, huber: 0.2450, swd: 0.2431, ept: 123.5939
    Epoch [6/50], Val Losses: mse: 0.4511, mae: 0.4624, huber: 0.1904, swd: 0.1479, ept: 115.2947
    Epoch [6/50], Test Losses: mse: 0.6598, mae: 0.5924, huber: 0.2809, swd: 0.1991, ept: 127.4805
      Epoch 6 composite train-obj: 0.244959
            Val objective improved 0.1927 → 0.1904, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.5948, mae: 0.5388, huber: 0.2445, swd: 0.2429, ept: 124.3905
    Epoch [7/50], Val Losses: mse: 0.4446, mae: 0.4563, huber: 0.1873, swd: 0.1438, ept: 124.4290
    Epoch [7/50], Test Losses: mse: 0.6605, mae: 0.5921, huber: 0.2810, swd: 0.1978, ept: 127.3062
      Epoch 7 composite train-obj: 0.244499
            Val objective improved 0.1904 → 0.1873, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.5940, mae: 0.5385, huber: 0.2443, swd: 0.2433, ept: 124.7609
    Epoch [8/50], Val Losses: mse: 0.4516, mae: 0.4647, huber: 0.1913, swd: 0.1526, ept: 115.3842
    Epoch [8/50], Test Losses: mse: 0.6622, mae: 0.5927, huber: 0.2818, swd: 0.2007, ept: 127.6464
      Epoch 8 composite train-obj: 0.244307
            No improvement (0.1913), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.5939, mae: 0.5380, huber: 0.2441, swd: 0.2431, ept: 125.5331
    Epoch [9/50], Val Losses: mse: 0.4507, mae: 0.4605, huber: 0.1898, swd: 0.1482, ept: 115.2735
    Epoch [9/50], Test Losses: mse: 0.6572, mae: 0.5916, huber: 0.2799, swd: 0.2028, ept: 128.5417
      Epoch 9 composite train-obj: 0.244138
            No improvement (0.1898), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.5939, mae: 0.5379, huber: 0.2441, swd: 0.2433, ept: 125.5584
    Epoch [10/50], Val Losses: mse: 0.4498, mae: 0.4631, huber: 0.1904, swd: 0.1537, ept: 114.1320
    Epoch [10/50], Test Losses: mse: 0.6545, mae: 0.5896, huber: 0.2789, swd: 0.2078, ept: 128.2303
      Epoch 10 composite train-obj: 0.244099
            No improvement (0.1904), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.5930, mae: 0.5375, huber: 0.2438, swd: 0.2436, ept: 126.0781
    Epoch [11/50], Val Losses: mse: 0.4464, mae: 0.4571, huber: 0.1882, swd: 0.1479, ept: 123.6703
    Epoch [11/50], Test Losses: mse: 0.6557, mae: 0.5896, huber: 0.2792, swd: 0.2021, ept: 128.2471
      Epoch 11 composite train-obj: 0.243793
            No improvement (0.1882), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.5928, mae: 0.5372, huber: 0.2437, swd: 0.2433, ept: 126.2553
    Epoch [12/50], Val Losses: mse: 0.4507, mae: 0.4624, huber: 0.1904, swd: 0.1532, ept: 113.2921
    Epoch [12/50], Test Losses: mse: 0.6564, mae: 0.5903, huber: 0.2796, swd: 0.2016, ept: 128.4869
      Epoch 12 composite train-obj: 0.243677
    Epoch [12/50], Test Losses: mse: 0.6605, mae: 0.5921, huber: 0.2810, swd: 0.1978, ept: 127.3062
    Best round's Test MSE: 0.6605, MAE: 0.5921, SWD: 0.1978
    Best round's Validation MSE: 0.4446, MAE: 0.4563, SWD: 0.1438
    Best round's Test verification MSE : 0.6605, MAE: 0.5921, SWD: 0.1978
    Time taken: 12.08 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7035, mae: 0.6008, huber: 0.2861, swd: 0.2436, ept: 72.3579
    Epoch [1/50], Val Losses: mse: 0.4773, mae: 0.4880, huber: 0.2033, swd: 0.1423, ept: 97.6288
    Epoch [1/50], Test Losses: mse: 0.7018, mae: 0.6192, huber: 0.2975, swd: 0.1949, ept: 114.7357
      Epoch 1 composite train-obj: 0.286055
            Val objective improved inf → 0.2033, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6102, mae: 0.5510, huber: 0.2512, swd: 0.2381, ept: 108.9012
    Epoch [2/50], Val Losses: mse: 0.4623, mae: 0.4738, huber: 0.1960, swd: 0.1437, ept: 110.4085
    Epoch [2/50], Test Losses: mse: 0.6773, mae: 0.6044, huber: 0.2879, swd: 0.1975, ept: 122.4030
      Epoch 2 composite train-obj: 0.251246
            Val objective improved 0.2033 → 0.1960, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6021, mae: 0.5447, huber: 0.2477, swd: 0.2381, ept: 116.6045
    Epoch [3/50], Val Losses: mse: 0.4512, mae: 0.4622, huber: 0.1906, swd: 0.1385, ept: 128.8270
    Epoch [3/50], Test Losses: mse: 0.6693, mae: 0.5991, huber: 0.2848, swd: 0.1973, ept: 121.7365
      Epoch 3 composite train-obj: 0.247718
            Val objective improved 0.1960 → 0.1906, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5989, mae: 0.5421, huber: 0.2463, swd: 0.2370, ept: 119.8973
    Epoch [4/50], Val Losses: mse: 0.4509, mae: 0.4613, huber: 0.1902, swd: 0.1405, ept: 119.4368
    Epoch [4/50], Test Losses: mse: 0.6666, mae: 0.5966, huber: 0.2836, swd: 0.1947, ept: 126.6677
      Epoch 4 composite train-obj: 0.246294
            Val objective improved 0.1906 → 0.1902, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5969, mae: 0.5405, huber: 0.2454, swd: 0.2374, ept: 122.0863
    Epoch [5/50], Val Losses: mse: 0.4521, mae: 0.4627, huber: 0.1909, swd: 0.1422, ept: 119.6670
    Epoch [5/50], Test Losses: mse: 0.6654, mae: 0.5953, huber: 0.2831, swd: 0.1943, ept: 126.6603
      Epoch 5 composite train-obj: 0.245438
            No improvement (0.1909), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.5956, mae: 0.5396, huber: 0.2449, swd: 0.2377, ept: 123.4568
    Epoch [6/50], Val Losses: mse: 0.4461, mae: 0.4587, huber: 0.1885, swd: 0.1412, ept: 123.1720
    Epoch [6/50], Test Losses: mse: 0.6616, mae: 0.5932, huber: 0.2815, swd: 0.1970, ept: 126.6309
      Epoch 6 composite train-obj: 0.244909
            Val objective improved 0.1902 → 0.1885, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.5948, mae: 0.5388, huber: 0.2445, swd: 0.2370, ept: 123.9825
    Epoch [7/50], Val Losses: mse: 0.4595, mae: 0.4702, huber: 0.1944, swd: 0.1499, ept: 107.9897
    Epoch [7/50], Test Losses: mse: 0.6596, mae: 0.5920, huber: 0.2806, swd: 0.1952, ept: 127.8628
      Epoch 7 composite train-obj: 0.244533
            No improvement (0.1944), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.5945, mae: 0.5386, huber: 0.2444, swd: 0.2383, ept: 124.8389
    Epoch [8/50], Val Losses: mse: 0.4463, mae: 0.4571, huber: 0.1881, swd: 0.1391, ept: 124.1235
    Epoch [8/50], Test Losses: mse: 0.6567, mae: 0.5906, huber: 0.2796, swd: 0.1936, ept: 127.3376
      Epoch 8 composite train-obj: 0.244395
            Val objective improved 0.1885 → 0.1881, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.5939, mae: 0.5381, huber: 0.2442, swd: 0.2378, ept: 125.1069
    Epoch [9/50], Val Losses: mse: 0.4397, mae: 0.4487, huber: 0.1845, swd: 0.1335, ept: 127.3017
    Epoch [9/50], Test Losses: mse: 0.6668, mae: 0.5951, huber: 0.2835, swd: 0.1902, ept: 125.2272
      Epoch 9 composite train-obj: 0.244181
            Val objective improved 0.1881 → 0.1845, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.5936, mae: 0.5377, huber: 0.2440, swd: 0.2376, ept: 125.4192
    Epoch [10/50], Val Losses: mse: 0.4481, mae: 0.4582, huber: 0.1888, swd: 0.1436, ept: 124.1241
    Epoch [10/50], Test Losses: mse: 0.6505, mae: 0.5888, huber: 0.2775, swd: 0.2035, ept: 125.6173
      Epoch 10 composite train-obj: 0.243960
            No improvement (0.1888), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.5931, mae: 0.5376, huber: 0.2439, swd: 0.2384, ept: 125.9323
    Epoch [11/50], Val Losses: mse: 0.4513, mae: 0.4633, huber: 0.1907, swd: 0.1466, ept: 110.9481
    Epoch [11/50], Test Losses: mse: 0.6592, mae: 0.5905, huber: 0.2804, swd: 0.1947, ept: 128.1408
      Epoch 11 composite train-obj: 0.243858
            No improvement (0.1907), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.5925, mae: 0.5371, huber: 0.2436, swd: 0.2378, ept: 126.2502
    Epoch [12/50], Val Losses: mse: 0.4505, mae: 0.4610, huber: 0.1901, swd: 0.1436, ept: 124.6044
    Epoch [12/50], Test Losses: mse: 0.6545, mae: 0.5893, huber: 0.2787, swd: 0.1978, ept: 128.7808
      Epoch 12 composite train-obj: 0.243597
            No improvement (0.1901), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.5927, mae: 0.5371, huber: 0.2437, swd: 0.2380, ept: 126.3288
    Epoch [13/50], Val Losses: mse: 0.4452, mae: 0.4569, huber: 0.1880, swd: 0.1438, ept: 128.4406
    Epoch [13/50], Test Losses: mse: 0.6554, mae: 0.5897, huber: 0.2792, swd: 0.2018, ept: 128.1590
      Epoch 13 composite train-obj: 0.243656
            No improvement (0.1880), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.5926, mae: 0.5369, huber: 0.2436, swd: 0.2381, ept: 126.5577
    Epoch [14/50], Val Losses: mse: 0.4557, mae: 0.4663, huber: 0.1925, swd: 0.1495, ept: 114.7306
    Epoch [14/50], Test Losses: mse: 0.6506, mae: 0.5875, huber: 0.2771, swd: 0.2013, ept: 128.7897
      Epoch 14 composite train-obj: 0.243556
    Epoch [14/50], Test Losses: mse: 0.6668, mae: 0.5951, huber: 0.2835, swd: 0.1902, ept: 125.2272
    Best round's Test MSE: 0.6668, MAE: 0.5951, SWD: 0.1902
    Best round's Validation MSE: 0.4397, MAE: 0.4487, SWD: 0.1335
    Best round's Test verification MSE : 0.6668, MAE: 0.5951, SWD: 0.1902
    Time taken: 14.47 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7055, mae: 0.6020, huber: 0.2869, swd: 0.2605, ept: 71.5277
    Epoch [1/50], Val Losses: mse: 0.4764, mae: 0.4880, huber: 0.2033, swd: 0.1549, ept: 94.5731
    Epoch [1/50], Test Losses: mse: 0.7011, mae: 0.6186, huber: 0.2973, swd: 0.2070, ept: 114.7929
      Epoch 1 composite train-obj: 0.286892
            Val objective improved inf → 0.2033, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6103, mae: 0.5512, huber: 0.2513, swd: 0.2545, ept: 108.9836
    Epoch [2/50], Val Losses: mse: 0.4639, mae: 0.4755, huber: 0.1969, swd: 0.1551, ept: 107.4976
    Epoch [2/50], Test Losses: mse: 0.6734, mae: 0.6029, huber: 0.2865, swd: 0.2092, ept: 123.5452
      Epoch 2 composite train-obj: 0.251311
            Val objective improved 0.2033 → 0.1969, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6026, mae: 0.5448, huber: 0.2478, swd: 0.2536, ept: 116.6623
    Epoch [3/50], Val Losses: mse: 0.4540, mae: 0.4663, huber: 0.1921, swd: 0.1516, ept: 114.6115
    Epoch [3/50], Test Losses: mse: 0.6701, mae: 0.5995, huber: 0.2851, swd: 0.2060, ept: 125.4212
      Epoch 3 composite train-obj: 0.247816
            Val objective improved 0.1969 → 0.1921, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5989, mae: 0.5421, huber: 0.2463, swd: 0.2535, ept: 120.1698
    Epoch [4/50], Val Losses: mse: 0.4539, mae: 0.4654, huber: 0.1920, swd: 0.1540, ept: 120.7404
    Epoch [4/50], Test Losses: mse: 0.6616, mae: 0.5942, huber: 0.2816, swd: 0.2094, ept: 126.1409
      Epoch 4 composite train-obj: 0.246301
            Val objective improved 0.1921 → 0.1920, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5972, mae: 0.5406, huber: 0.2455, swd: 0.2529, ept: 121.8519
    Epoch [5/50], Val Losses: mse: 0.4578, mae: 0.4690, huber: 0.1937, swd: 0.1591, ept: 112.2400
    Epoch [5/50], Test Losses: mse: 0.6585, mae: 0.5931, huber: 0.2805, swd: 0.2125, ept: 127.4625
      Epoch 5 composite train-obj: 0.245534
            No improvement (0.1937), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.5959, mae: 0.5398, huber: 0.2450, swd: 0.2540, ept: 123.2387
    Epoch [6/50], Val Losses: mse: 0.4530, mae: 0.4634, huber: 0.1910, swd: 0.1525, ept: 117.0976
    Epoch [6/50], Test Losses: mse: 0.6552, mae: 0.5900, huber: 0.2789, swd: 0.2082, ept: 126.7895
      Epoch 6 composite train-obj: 0.245033
            Val objective improved 0.1920 → 0.1910, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.5951, mae: 0.5389, huber: 0.2446, swd: 0.2533, ept: 124.2789
    Epoch [7/50], Val Losses: mse: 0.4514, mae: 0.4625, huber: 0.1906, swd: 0.1558, ept: 118.1071
    Epoch [7/50], Test Losses: mse: 0.6622, mae: 0.5945, huber: 0.2820, swd: 0.2096, ept: 127.5907
      Epoch 7 composite train-obj: 0.244638
            Val objective improved 0.1910 → 0.1906, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.5940, mae: 0.5383, huber: 0.2443, swd: 0.2539, ept: 124.9539
    Epoch [8/50], Val Losses: mse: 0.4454, mae: 0.4571, huber: 0.1878, swd: 0.1493, ept: 128.4252
    Epoch [8/50], Test Losses: mse: 0.6604, mae: 0.5921, huber: 0.2811, swd: 0.2069, ept: 127.0378
      Epoch 8 composite train-obj: 0.244264
            Val objective improved 0.1906 → 0.1878, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.5940, mae: 0.5381, huber: 0.2442, swd: 0.2536, ept: 125.0995
    Epoch [9/50], Val Losses: mse: 0.4518, mae: 0.4651, huber: 0.1912, swd: 0.1571, ept: 108.2290
    Epoch [9/50], Test Losses: mse: 0.6586, mae: 0.5913, huber: 0.2803, swd: 0.2048, ept: 128.1449
      Epoch 9 composite train-obj: 0.244174
            No improvement (0.1912), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.5931, mae: 0.5376, huber: 0.2439, swd: 0.2540, ept: 125.6087
    Epoch [10/50], Val Losses: mse: 0.4452, mae: 0.4567, huber: 0.1878, swd: 0.1518, ept: 126.6233
    Epoch [10/50], Test Losses: mse: 0.6587, mae: 0.5916, huber: 0.2805, swd: 0.2094, ept: 127.4397
      Epoch 10 composite train-obj: 0.243879
            Val objective improved 0.1878 → 0.1878, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.5931, mae: 0.5375, huber: 0.2438, swd: 0.2536, ept: 125.8978
    Epoch [11/50], Val Losses: mse: 0.4448, mae: 0.4570, huber: 0.1878, swd: 0.1553, ept: 122.4157
    Epoch [11/50], Test Losses: mse: 0.6557, mae: 0.5912, huber: 0.2796, swd: 0.2174, ept: 126.8632
      Epoch 11 composite train-obj: 0.243808
            No improvement (0.1878), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.5930, mae: 0.5375, huber: 0.2438, swd: 0.2550, ept: 126.2383
    Epoch [12/50], Val Losses: mse: 0.4403, mae: 0.4537, huber: 0.1858, swd: 0.1494, ept: 125.3344
    Epoch [12/50], Test Losses: mse: 0.6626, mae: 0.5932, huber: 0.2820, swd: 0.2066, ept: 128.4478
      Epoch 12 composite train-obj: 0.243796
            Val objective improved 0.1878 → 0.1858, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.5929, mae: 0.5372, huber: 0.2437, swd: 0.2538, ept: 126.4223
    Epoch [13/50], Val Losses: mse: 0.4425, mae: 0.4540, huber: 0.1865, swd: 0.1494, ept: 127.9782
    Epoch [13/50], Test Losses: mse: 0.6575, mae: 0.5901, huber: 0.2799, swd: 0.2075, ept: 127.2653
      Epoch 13 composite train-obj: 0.243724
            No improvement (0.1865), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.5925, mae: 0.5371, huber: 0.2436, swd: 0.2537, ept: 126.1772
    Epoch [14/50], Val Losses: mse: 0.4479, mae: 0.4593, huber: 0.1891, swd: 0.1548, ept: 130.6483
    Epoch [14/50], Test Losses: mse: 0.6489, mae: 0.5873, huber: 0.2766, swd: 0.2166, ept: 127.1688
      Epoch 14 composite train-obj: 0.243611
            No improvement (0.1891), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.5929, mae: 0.5370, huber: 0.2437, swd: 0.2543, ept: 126.3918
    Epoch [15/50], Val Losses: mse: 0.4443, mae: 0.4536, huber: 0.1869, swd: 0.1489, ept: 132.6156
    Epoch [15/50], Test Losses: mse: 0.6575, mae: 0.5896, huber: 0.2798, swd: 0.2079, ept: 126.8280
      Epoch 15 composite train-obj: 0.243675
            No improvement (0.1869), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.5924, mae: 0.5369, huber: 0.2435, swd: 0.2548, ept: 126.5765
    Epoch [16/50], Val Losses: mse: 0.4446, mae: 0.4558, huber: 0.1873, swd: 0.1508, ept: 122.4999
    Epoch [16/50], Test Losses: mse: 0.6567, mae: 0.5912, huber: 0.2797, swd: 0.2114, ept: 128.2446
      Epoch 16 composite train-obj: 0.243525
            No improvement (0.1873), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.5922, mae: 0.5368, huber: 0.2435, swd: 0.2539, ept: 126.4252
    Epoch [17/50], Val Losses: mse: 0.4589, mae: 0.4704, huber: 0.1944, swd: 0.1666, ept: 110.9038
    Epoch [17/50], Test Losses: mse: 0.6515, mae: 0.5895, huber: 0.2780, swd: 0.2197, ept: 128.2349
      Epoch 17 composite train-obj: 0.243478
    Epoch [17/50], Test Losses: mse: 0.6626, mae: 0.5932, huber: 0.2820, swd: 0.2066, ept: 128.4478
    Best round's Test MSE: 0.6626, MAE: 0.5932, SWD: 0.2066
    Best round's Validation MSE: 0.4403, MAE: 0.4537, SWD: 0.1494
    Best round's Test verification MSE : 0.6626, MAE: 0.5932, SWD: 0.2066
    Time taken: 17.00 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq96_pred720_20250510_1816)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6633 ± 0.0026
      mae: 0.5934 ± 0.0012
      huber: 0.2822 ± 0.0010
      swd: 0.1982 ± 0.0067
      ept: 126.9937 ± 1.3332
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4415 ± 0.0022
      mae: 0.4529 ± 0.0031
      huber: 0.1859 ± 0.0011
      swd: 0.1422 ± 0.0066
      ept: 125.6883 ± 1.1992
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 43.60 seconds
    
    Experiment complete: DLinear_etth1_seq96_pred720_20250510_1816
    Model: DLinear
    Dataset: etth1
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    






