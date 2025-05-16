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
    




    <data_manager.DatasetManager at 0x2359517acc0>



## Seq=720

### EigenACL

#### pred=96


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 96
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
    
    Epoch [1/50], Train Losses: mse: 0.5198, mae: 0.5086, huber: 0.2185, swd: 0.2835, ept: 39.4478
    Epoch [1/50], Val Losses: mse: 0.3667, mae: 0.4272, huber: 0.1632, swd: 0.1422, ept: 46.0043
    Epoch [1/50], Test Losses: mse: 0.4733, mae: 0.4958, huber: 0.2083, swd: 0.1834, ept: 30.5128
      Epoch 1 composite train-obj: 0.218508
            Val objective improved inf → 0.1632, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3331, mae: 0.4054, huber: 0.1487, swd: 0.1149, ept: 49.8949
    Epoch [2/50], Val Losses: mse: 0.3763, mae: 0.4301, huber: 0.1667, swd: 0.1245, ept: 45.9001
    Epoch [2/50], Test Losses: mse: 0.4495, mae: 0.4781, huber: 0.1974, swd: 0.1499, ept: 33.2291
      Epoch 2 composite train-obj: 0.148710
            No improvement (0.1667), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.2933, mae: 0.3813, huber: 0.1332, swd: 0.0921, ept: 52.2307
    Epoch [3/50], Val Losses: mse: 0.4331, mae: 0.4867, huber: 0.1946, swd: 0.1683, ept: 42.5568
    Epoch [3/50], Test Losses: mse: 0.4538, mae: 0.4832, huber: 0.2011, swd: 0.1552, ept: 33.1611
      Epoch 3 composite train-obj: 0.133200
            No improvement (0.1946), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2635, mae: 0.3623, huber: 0.1211, swd: 0.0739, ept: 53.9651
    Epoch [4/50], Val Losses: mse: 0.4196, mae: 0.4687, huber: 0.1859, swd: 0.1456, ept: 44.6636
    Epoch [4/50], Test Losses: mse: 0.4689, mae: 0.4874, huber: 0.2046, swd: 0.1509, ept: 34.4841
      Epoch 4 composite train-obj: 0.121122
            No improvement (0.1859), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2442, mae: 0.3483, huber: 0.1129, swd: 0.0633, ept: 55.3942
    Epoch [5/50], Val Losses: mse: 0.4340, mae: 0.4733, huber: 0.1907, swd: 0.1348, ept: 44.5083
    Epoch [5/50], Test Losses: mse: 0.4831, mae: 0.4967, huber: 0.2104, swd: 0.1565, ept: 34.6217
      Epoch 5 composite train-obj: 0.112905
            No improvement (0.1907), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2249, mae: 0.3336, huber: 0.1046, swd: 0.0531, ept: 56.8750
    Epoch [6/50], Val Losses: mse: 0.4863, mae: 0.5121, huber: 0.2129, swd: 0.1629, ept: 43.1690
    Epoch [6/50], Test Losses: mse: 0.5042, mae: 0.5080, huber: 0.2183, swd: 0.1607, ept: 33.8602
      Epoch 6 composite train-obj: 0.104567
    Epoch [6/50], Test Losses: mse: 0.4734, mae: 0.4959, huber: 0.2083, swd: 0.1835, ept: 30.5102
    Best round's Test MSE: 0.4733, MAE: 0.4958, SWD: 0.1834
    Best round's Validation MSE: 0.3667, MAE: 0.4272, SWD: 0.1422
    Best round's Test verification MSE : 0.4734, MAE: 0.4959, SWD: 0.1835
    Time taken: 15.15 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5203, mae: 0.5091, huber: 0.2189, swd: 0.2708, ept: 39.2655
    Epoch [1/50], Val Losses: mse: 0.3748, mae: 0.4347, huber: 0.1669, swd: 0.1350, ept: 45.8381
    Epoch [1/50], Test Losses: mse: 0.4739, mae: 0.4968, huber: 0.2087, swd: 0.1726, ept: 29.9491
      Epoch 1 composite train-obj: 0.218865
            Val objective improved inf → 0.1669, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3343, mae: 0.4073, huber: 0.1494, swd: 0.1138, ept: 49.2836
    Epoch [2/50], Val Losses: mse: 0.3873, mae: 0.4581, huber: 0.1744, swd: 0.1451, ept: 44.1159
    Epoch [2/50], Test Losses: mse: 0.4458, mae: 0.4781, huber: 0.1981, swd: 0.1642, ept: 31.5635
      Epoch 2 composite train-obj: 0.149394
            No improvement (0.1744), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.2895, mae: 0.3795, huber: 0.1318, swd: 0.0886, ept: 52.0404
    Epoch [3/50], Val Losses: mse: 0.3969, mae: 0.4512, huber: 0.1761, swd: 0.1302, ept: 44.7233
    Epoch [3/50], Test Losses: mse: 0.4478, mae: 0.4777, huber: 0.1977, swd: 0.1374, ept: 33.6777
      Epoch 3 composite train-obj: 0.131803
            No improvement (0.1761), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2583, mae: 0.3590, huber: 0.1189, swd: 0.0699, ept: 54.0989
    Epoch [4/50], Val Losses: mse: 0.4053, mae: 0.4554, huber: 0.1796, swd: 0.1378, ept: 44.6631
    Epoch [4/50], Test Losses: mse: 0.4664, mae: 0.4886, huber: 0.2053, swd: 0.1512, ept: 34.4444
      Epoch 4 composite train-obj: 0.118927
            No improvement (0.1796), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2359, mae: 0.3425, huber: 0.1094, swd: 0.0580, ept: 55.9379
    Epoch [5/50], Val Losses: mse: 0.4524, mae: 0.4945, huber: 0.2016, swd: 0.1671, ept: 42.3354
    Epoch [5/50], Test Losses: mse: 0.4775, mae: 0.4952, huber: 0.2097, swd: 0.1604, ept: 33.1653
      Epoch 5 composite train-obj: 0.109389
            No improvement (0.2016), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2210, mae: 0.3317, huber: 0.1031, swd: 0.0516, ept: 57.2005
    Epoch [6/50], Val Losses: mse: 0.4272, mae: 0.4671, huber: 0.1883, swd: 0.1291, ept: 44.1878
    Epoch [6/50], Test Losses: mse: 0.4943, mae: 0.4985, huber: 0.2134, swd: 0.1540, ept: 34.4699
      Epoch 6 composite train-obj: 0.103067
    Epoch [6/50], Test Losses: mse: 0.4739, mae: 0.4968, huber: 0.2088, swd: 0.1726, ept: 29.9269
    Best round's Test MSE: 0.4739, MAE: 0.4968, SWD: 0.1726
    Best round's Validation MSE: 0.3748, MAE: 0.4347, SWD: 0.1350
    Best round's Test verification MSE : 0.4739, MAE: 0.4968, SWD: 0.1726
    Time taken: 14.90 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5065, mae: 0.5026, huber: 0.2142, swd: 0.2567, ept: 40.1285
    Epoch [1/50], Val Losses: mse: 0.3911, mae: 0.4580, huber: 0.1755, swd: 0.1450, ept: 44.1901
    Epoch [1/50], Test Losses: mse: 0.4683, mae: 0.4917, huber: 0.2064, swd: 0.1701, ept: 29.4303
      Epoch 1 composite train-obj: 0.214167
            Val objective improved inf → 0.1755, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3375, mae: 0.4097, huber: 0.1507, swd: 0.1108, ept: 49.0295
    Epoch [2/50], Val Losses: mse: 0.3909, mae: 0.4489, huber: 0.1743, swd: 0.1357, ept: 45.4874
    Epoch [2/50], Test Losses: mse: 0.4530, mae: 0.4828, huber: 0.2002, swd: 0.1553, ept: 31.5455
      Epoch 2 composite train-obj: 0.150694
            Val objective improved 0.1755 → 0.1743, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2936, mae: 0.3828, huber: 0.1335, swd: 0.0877, ept: 51.4070
    Epoch [3/50], Val Losses: mse: 0.3896, mae: 0.4487, huber: 0.1742, swd: 0.1218, ept: 44.7092
    Epoch [3/50], Test Losses: mse: 0.4469, mae: 0.4776, huber: 0.1977, swd: 0.1430, ept: 32.4701
      Epoch 3 composite train-obj: 0.133522
            Val objective improved 0.1743 → 0.1742, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2621, mae: 0.3621, huber: 0.1206, swd: 0.0692, ept: 53.3810
    Epoch [4/50], Val Losses: mse: 0.4086, mae: 0.4635, huber: 0.1824, swd: 0.1214, ept: 44.0194
    Epoch [4/50], Test Losses: mse: 0.4453, mae: 0.4764, huber: 0.1963, swd: 0.1255, ept: 33.2159
      Epoch 4 composite train-obj: 0.120560
            No improvement (0.1824), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.2408, mae: 0.3466, huber: 0.1115, swd: 0.0588, ept: 55.0198
    Epoch [5/50], Val Losses: mse: 0.4222, mae: 0.4758, huber: 0.1893, swd: 0.1347, ept: 43.5948
    Epoch [5/50], Test Losses: mse: 0.4645, mae: 0.4887, huber: 0.2048, swd: 0.1514, ept: 32.8766
      Epoch 5 composite train-obj: 0.111548
            No improvement (0.1893), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.2236, mae: 0.3337, huber: 0.1042, swd: 0.0503, ept: 56.4183
    Epoch [6/50], Val Losses: mse: 0.4651, mae: 0.4999, huber: 0.2071, swd: 0.1573, ept: 41.9831
    Epoch [6/50], Test Losses: mse: 0.4778, mae: 0.4946, huber: 0.2093, swd: 0.1548, ept: 32.9190
      Epoch 6 composite train-obj: 0.104223
            No improvement (0.2071), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.2100, mae: 0.3240, huber: 0.0985, swd: 0.0440, ept: 57.5402
    Epoch [7/50], Val Losses: mse: 0.4758, mae: 0.5056, huber: 0.2104, swd: 0.1509, ept: 41.5530
    Epoch [7/50], Test Losses: mse: 0.4933, mae: 0.5056, huber: 0.2159, swd: 0.1668, ept: 32.2059
      Epoch 7 composite train-obj: 0.098500
            No improvement (0.2104), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.1997, mae: 0.3158, huber: 0.0940, swd: 0.0394, ept: 58.6359
    Epoch [8/50], Val Losses: mse: 0.4665, mae: 0.4963, huber: 0.2057, swd: 0.1358, ept: 41.7378
    Epoch [8/50], Test Losses: mse: 0.4924, mae: 0.5009, huber: 0.2146, swd: 0.1643, ept: 32.7903
      Epoch 8 composite train-obj: 0.093966
    Epoch [8/50], Test Losses: mse: 0.4469, mae: 0.4776, huber: 0.1977, swd: 0.1430, ept: 32.4635
    Best round's Test MSE: 0.4469, MAE: 0.4776, SWD: 0.1430
    Best round's Validation MSE: 0.3896, MAE: 0.4487, SWD: 0.1218
    Best round's Test verification MSE : 0.4469, MAE: 0.4776, SWD: 0.1430
    Time taken: 19.70 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq720_pred96_20250510_1948)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4647 ± 0.0126
      mae: 0.4900 ± 0.0088
      huber: 0.2049 ± 0.0051
      swd: 0.1663 ± 0.0171
      ept: 30.9773 ± 1.0803
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3770 ± 0.0095
      mae: 0.4368 ± 0.0089
      huber: 0.1681 ± 0.0046
      swd: 0.1330 ± 0.0085
      ept: 45.5172 ± 0.5754
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 50.50 seconds
    
    Experiment complete: ACL_etth1_seq720_pred96_20250510_1948
    Model: ACL
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.5764, mae: 0.5359, huber: 0.2385, swd: 0.3181, ept: 57.1193
    Epoch [1/50], Val Losses: mse: 0.4756, mae: 0.5085, huber: 0.2112, swd: 0.1926, ept: 55.3838
    Epoch [1/50], Test Losses: mse: 0.5029, mae: 0.5178, huber: 0.2223, swd: 0.1849, ept: 43.0231
      Epoch 1 composite train-obj: 0.238499
            Val objective improved inf → 0.2112, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3972, mae: 0.4424, huber: 0.1734, swd: 0.1439, ept: 72.7281
    Epoch [2/50], Val Losses: mse: 0.5182, mae: 0.5295, huber: 0.2269, swd: 0.1920, ept: 51.9401
    Epoch [2/50], Test Losses: mse: 0.4923, mae: 0.5122, huber: 0.2183, swd: 0.1657, ept: 45.4580
      Epoch 2 composite train-obj: 0.173366
            No improvement (0.2269), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3373, mae: 0.4111, huber: 0.1516, swd: 0.1075, ept: 76.3449
    Epoch [3/50], Val Losses: mse: 0.5872, mae: 0.5811, huber: 0.2574, swd: 0.2170, ept: 47.1387
    Epoch [3/50], Test Losses: mse: 0.5638, mae: 0.5581, huber: 0.2482, swd: 0.1951, ept: 44.4957
      Epoch 3 composite train-obj: 0.151570
            No improvement (0.2574), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2992, mae: 0.3884, huber: 0.1366, swd: 0.0822, ept: 79.3650
    Epoch [4/50], Val Losses: mse: 0.5953, mae: 0.5767, huber: 0.2627, swd: 0.2312, ept: 45.7248
    Epoch [4/50], Test Losses: mse: 0.5201, mae: 0.5242, huber: 0.2293, swd: 0.1831, ept: 44.7457
      Epoch 4 composite train-obj: 0.136562
            No improvement (0.2627), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2745, mae: 0.3734, huber: 0.1267, swd: 0.0661, ept: 81.8554
    Epoch [5/50], Val Losses: mse: 0.6591, mae: 0.6051, huber: 0.2876, swd: 0.2539, ept: 43.1693
    Epoch [5/50], Test Losses: mse: 0.5501, mae: 0.5377, huber: 0.2398, swd: 0.1854, ept: 44.1655
      Epoch 5 composite train-obj: 0.126652
            No improvement (0.2876), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2621, mae: 0.3649, huber: 0.1214, swd: 0.0590, ept: 83.7169
    Epoch [6/50], Val Losses: mse: 0.6530, mae: 0.6046, huber: 0.2865, swd: 0.2430, ept: 44.1727
    Epoch [6/50], Test Losses: mse: 0.5505, mae: 0.5397, huber: 0.2413, swd: 0.1941, ept: 44.8513
      Epoch 6 composite train-obj: 0.121411
    Epoch [6/50], Test Losses: mse: 0.5029, mae: 0.5178, huber: 0.2223, swd: 0.1849, ept: 43.0117
    Best round's Test MSE: 0.5029, MAE: 0.5178, SWD: 0.1849
    Best round's Validation MSE: 0.4756, MAE: 0.5085, SWD: 0.1926
    Best round's Test verification MSE : 0.5029, MAE: 0.5178, SWD: 0.1849
    Time taken: 14.94 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5865, mae: 0.5405, huber: 0.2420, swd: 0.3275, ept: 57.0768
    Epoch [1/50], Val Losses: mse: 0.4874, mae: 0.5086, huber: 0.2146, swd: 0.1675, ept: 53.3222
    Epoch [1/50], Test Losses: mse: 0.5082, mae: 0.5197, huber: 0.2239, swd: 0.1532, ept: 42.8908
      Epoch 1 composite train-obj: 0.242012
            Val objective improved inf → 0.2146, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3960, mae: 0.4435, huber: 0.1735, swd: 0.1435, ept: 71.4510
    Epoch [2/50], Val Losses: mse: 0.4859, mae: 0.4976, huber: 0.2120, swd: 0.1866, ept: 54.8630
    Epoch [2/50], Test Losses: mse: 0.5163, mae: 0.5317, huber: 0.2300, swd: 0.2056, ept: 42.6951
      Epoch 2 composite train-obj: 0.173475
            Val objective improved 0.2146 → 0.2120, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3326, mae: 0.4092, huber: 0.1500, swd: 0.1010, ept: 75.8939
    Epoch [3/50], Val Losses: mse: 0.5057, mae: 0.5120, huber: 0.2198, swd: 0.1413, ept: 52.5944
    Epoch [3/50], Test Losses: mse: 0.4964, mae: 0.5120, huber: 0.2186, swd: 0.1271, ept: 46.7571
      Epoch 3 composite train-obj: 0.149981
            No improvement (0.2198), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.2977, mae: 0.3876, huber: 0.1360, swd: 0.0779, ept: 79.3102
    Epoch [4/50], Val Losses: mse: 0.5531, mae: 0.5526, huber: 0.2441, swd: 0.1960, ept: 47.8710
    Epoch [4/50], Test Losses: mse: 0.5064, mae: 0.5171, huber: 0.2238, swd: 0.1550, ept: 46.9244
      Epoch 4 composite train-obj: 0.135982
            No improvement (0.2441), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.2762, mae: 0.3735, huber: 0.1271, swd: 0.0660, ept: 81.9197
    Epoch [5/50], Val Losses: mse: 0.6820, mae: 0.6211, huber: 0.2939, swd: 0.2172, ept: 43.0852
    Epoch [5/50], Test Losses: mse: 0.5547, mae: 0.5433, huber: 0.2416, swd: 0.1633, ept: 45.0747
      Epoch 5 composite train-obj: 0.127085
            No improvement (0.2939), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.2602, mae: 0.3628, huber: 0.1205, swd: 0.0571, ept: 83.7063
    Epoch [6/50], Val Losses: mse: 0.5795, mae: 0.5646, huber: 0.2557, swd: 0.2114, ept: 47.2035
    Epoch [6/50], Test Losses: mse: 0.5184, mae: 0.5230, huber: 0.2288, swd: 0.1611, ept: 45.9655
      Epoch 6 composite train-obj: 0.120456
            No improvement (0.2557), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.2475, mae: 0.3533, huber: 0.1149, swd: 0.0509, ept: 85.8948
    Epoch [7/50], Val Losses: mse: 0.6103, mae: 0.5793, huber: 0.2663, swd: 0.1959, ept: 47.8068
    Epoch [7/50], Test Losses: mse: 0.5343, mae: 0.5278, huber: 0.2330, swd: 0.1579, ept: 46.4289
      Epoch 7 composite train-obj: 0.114926
    Epoch [7/50], Test Losses: mse: 0.5163, mae: 0.5317, huber: 0.2300, swd: 0.2055, ept: 42.7228
    Best round's Test MSE: 0.5163, MAE: 0.5317, SWD: 0.2056
    Best round's Validation MSE: 0.4859, MAE: 0.4976, SWD: 0.1866
    Best round's Test verification MSE : 0.5163, MAE: 0.5317, SWD: 0.2055
    Time taken: 17.20 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5771, mae: 0.5361, huber: 0.2387, swd: 0.2940, ept: 57.6053
    Epoch [1/50], Val Losses: mse: 0.4745, mae: 0.4988, huber: 0.2089, swd: 0.1635, ept: 54.4711
    Epoch [1/50], Test Losses: mse: 0.5115, mae: 0.5221, huber: 0.2252, swd: 0.1669, ept: 44.0638
      Epoch 1 composite train-obj: 0.238726
            Val objective improved inf → 0.2089, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4037, mae: 0.4465, huber: 0.1757, swd: 0.1397, ept: 71.9897
    Epoch [2/50], Val Losses: mse: 0.5054, mae: 0.5245, huber: 0.2237, swd: 0.1743, ept: 49.9699
    Epoch [2/50], Test Losses: mse: 0.5012, mae: 0.5180, huber: 0.2227, swd: 0.1625, ept: 43.8923
      Epoch 2 composite train-obj: 0.175728
            No improvement (0.2237), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3437, mae: 0.4144, huber: 0.1538, swd: 0.1037, ept: 75.6771
    Epoch [3/50], Val Losses: mse: 0.5711, mae: 0.5633, huber: 0.2519, swd: 0.1917, ept: 45.7767
    Epoch [3/50], Test Losses: mse: 0.5057, mae: 0.5198, huber: 0.2243, swd: 0.1504, ept: 43.2607
      Epoch 3 composite train-obj: 0.153834
            No improvement (0.2519), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2997, mae: 0.3893, huber: 0.1369, swd: 0.0781, ept: 79.1960
    Epoch [4/50], Val Losses: mse: 0.6192, mae: 0.5894, huber: 0.2719, swd: 0.2061, ept: 43.6153
    Epoch [4/50], Test Losses: mse: 0.5306, mae: 0.5352, huber: 0.2350, swd: 0.1692, ept: 42.5546
      Epoch 4 composite train-obj: 0.136939
            No improvement (0.2719), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2745, mae: 0.3734, huber: 0.1267, swd: 0.0648, ept: 82.1469
    Epoch [5/50], Val Losses: mse: 0.6720, mae: 0.6127, huber: 0.2890, swd: 0.1815, ept: 47.4746
    Epoch [5/50], Test Losses: mse: 0.5713, mae: 0.5532, huber: 0.2483, swd: 0.1510, ept: 45.8492
      Epoch 5 composite train-obj: 0.126659
            No improvement (0.2890), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2553, mae: 0.3600, huber: 0.1185, swd: 0.0545, ept: 85.0686
    Epoch [6/50], Val Losses: mse: 0.6446, mae: 0.5970, huber: 0.2780, swd: 0.1838, ept: 47.7942
    Epoch [6/50], Test Losses: mse: 0.5491, mae: 0.5384, huber: 0.2389, swd: 0.1441, ept: 46.5143
      Epoch 6 composite train-obj: 0.118492
    Epoch [6/50], Test Losses: mse: 0.5115, mae: 0.5221, huber: 0.2252, swd: 0.1670, ept: 43.9677
    Best round's Test MSE: 0.5115, MAE: 0.5221, SWD: 0.1669
    Best round's Validation MSE: 0.4745, MAE: 0.4988, SWD: 0.1635
    Best round's Test verification MSE : 0.5115, MAE: 0.5221, SWD: 0.1670
    Time taken: 14.90 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq720_pred196_20250510_1949)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5102 ± 0.0056
      mae: 0.5239 ± 0.0058
      huber: 0.2259 ± 0.0032
      swd: 0.1858 ± 0.0158
      ept: 43.2607 ± 0.5835
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4787 ± 0.0051
      mae: 0.5016 ± 0.0049
      huber: 0.2107 ± 0.0013
      swd: 0.1809 ± 0.0126
      ept: 54.9060 ± 0.3738
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 47.15 seconds
    
    Experiment complete: ACL_etth1_seq720_pred196_20250510_1949
    Model: ACL
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=720,
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
cfg.x_to_z_delay.enable_magnitudes = [False, False]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, False]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, False]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, False]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU, nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, False]
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 88
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 88
    Validation Batches: 6
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7519, mae: 0.6169, huber: 0.2987, swd: 0.4648, ept: 61.4511
    Epoch [1/50], Val Losses: mse: 0.5564, mae: 0.5405, huber: 0.2268, swd: 0.3573, ept: 66.9494
    Epoch [1/50], Test Losses: mse: 1.0091, mae: 0.7158, huber: 0.3793, swd: 0.7265, ept: 54.2488
      Epoch 1 composite train-obj: 0.298702
            Val objective improved inf → 0.2268, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5733, mae: 0.5307, huber: 0.2355, swd: 0.2603, ept: 81.7176
    Epoch [2/50], Val Losses: mse: 0.5512, mae: 0.5369, huber: 0.2370, swd: 0.1704, ept: 58.3145
    Epoch [2/50], Test Losses: mse: 0.6254, mae: 0.5922, huber: 0.2729, swd: 0.1910, ept: 52.0677
      Epoch 2 composite train-obj: 0.235517
            No improvement (0.2370), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4493, mae: 0.4777, huber: 0.1953, swd: 0.1658, ept: 89.3781
    Epoch [3/50], Val Losses: mse: 0.5764, mae: 0.5766, huber: 0.2538, swd: 0.1913, ept: 48.1937
    Epoch [3/50], Test Losses: mse: 0.5918, mae: 0.5679, huber: 0.2589, swd: 0.1603, ept: 52.6006
      Epoch 3 composite train-obj: 0.195280
            No improvement (0.2538), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.3992, mae: 0.4522, huber: 0.1773, swd: 0.1295, ept: 91.5215
    Epoch [4/50], Val Losses: mse: 0.6372, mae: 0.5978, huber: 0.2770, swd: 0.2556, ept: 47.2808
    Epoch [4/50], Test Losses: mse: 0.6152, mae: 0.5852, huber: 0.2694, swd: 0.2288, ept: 46.8844
      Epoch 4 composite train-obj: 0.177327
            No improvement (0.2770), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3610, mae: 0.4293, huber: 0.1622, swd: 0.1027, ept: 95.2773
    Epoch [5/50], Val Losses: mse: 0.7117, mae: 0.6355, huber: 0.3068, swd: 0.2634, ept: 42.0494
    Epoch [5/50], Test Losses: mse: 0.6220, mae: 0.5872, huber: 0.2713, swd: 0.1864, ept: 48.7942
      Epoch 5 composite train-obj: 0.162162
            No improvement (0.3068), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3344, mae: 0.4124, huber: 0.1513, swd: 0.0842, ept: 99.6437
    Epoch [6/50], Val Losses: mse: 0.7960, mae: 0.6850, huber: 0.3350, swd: 0.1849, ept: 51.7886
    Epoch [6/50], Test Losses: mse: 0.7623, mae: 0.6614, huber: 0.3264, swd: 0.2139, ept: 55.4653
      Epoch 6 composite train-obj: 0.151335
    Epoch [6/50], Test Losses: mse: 1.0092, mae: 0.7158, huber: 0.3793, swd: 0.7266, ept: 54.2460
    Best round's Test MSE: 1.0091, MAE: 0.7158, SWD: 0.7265
    Best round's Validation MSE: 0.5564, MAE: 0.5405, SWD: 0.3573
    Best round's Test verification MSE : 1.0092, MAE: 0.7158, SWD: 0.7266
    Time taken: 12.74 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7659, mae: 0.6243, huber: 0.3042, swd: 0.4824, ept: 59.1392
    Epoch [1/50], Val Losses: mse: 0.5674, mae: 0.5495, huber: 0.2379, swd: 0.2783, ept: 40.5791
    Epoch [1/50], Test Losses: mse: 0.9171, mae: 0.7059, huber: 0.3638, swd: 0.4426, ept: 21.1936
      Epoch 1 composite train-obj: 0.304206
            Val objective improved inf → 0.2379, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5390, mae: 0.5164, huber: 0.2247, swd: 0.2130, ept: 83.7790
    Epoch [2/50], Val Losses: mse: 0.5702, mae: 0.5555, huber: 0.2466, swd: 0.2305, ept: 58.2532
    Epoch [2/50], Test Losses: mse: 0.5921, mae: 0.5762, huber: 0.2612, swd: 0.2104, ept: 53.2904
      Epoch 2 composite train-obj: 0.224721
            No improvement (0.2466), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4553, mae: 0.4779, huber: 0.1963, swd: 0.1690, ept: 89.4729
    Epoch [3/50], Val Losses: mse: 0.5827, mae: 0.5856, huber: 0.2605, swd: 0.2395, ept: 33.4793
    Epoch [3/50], Test Losses: mse: 0.5829, mae: 0.5706, huber: 0.2566, swd: 0.1890, ept: 41.4271
      Epoch 3 composite train-obj: 0.196344
            No improvement (0.2605), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4131, mae: 0.4591, huber: 0.1821, swd: 0.1413, ept: 90.0405
    Epoch [4/50], Val Losses: mse: 0.8622, mae: 0.7399, huber: 0.3700, swd: 0.2009, ept: 45.0457
    Epoch [4/50], Test Losses: mse: 0.7854, mae: 0.6725, huber: 0.3355, swd: 0.2321, ept: 55.6518
      Epoch 4 composite train-obj: 0.182127
            No improvement (0.3700), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3750, mae: 0.4379, huber: 0.1674, swd: 0.1132, ept: 94.8153
    Epoch [5/50], Val Losses: mse: 0.6633, mae: 0.6194, huber: 0.2870, swd: 0.1955, ept: 54.9465
    Epoch [5/50], Test Losses: mse: 0.6446, mae: 0.5918, huber: 0.2779, swd: 0.1831, ept: 55.2558
      Epoch 5 composite train-obj: 0.167398
            No improvement (0.2870), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3414, mae: 0.4166, huber: 0.1540, swd: 0.0897, ept: 99.3012
    Epoch [6/50], Val Losses: mse: 0.6601, mae: 0.6203, huber: 0.2870, swd: 0.2339, ept: 53.6277
    Epoch [6/50], Test Losses: mse: 0.6522, mae: 0.5966, huber: 0.2817, swd: 0.2475, ept: 55.4186
      Epoch 6 composite train-obj: 0.153965
    Epoch [6/50], Test Losses: mse: 0.9171, mae: 0.7058, huber: 0.3637, swd: 0.4427, ept: 21.1253
    Best round's Test MSE: 0.9171, MAE: 0.7059, SWD: 0.4426
    Best round's Validation MSE: 0.5674, MAE: 0.5495, SWD: 0.2783
    Best round's Test verification MSE : 0.9171, MAE: 0.7058, SWD: 0.4427
    Time taken: 12.77 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7499, mae: 0.6166, huber: 0.2983, swd: 0.4608, ept: 62.9361
    Epoch [1/50], Val Losses: mse: 0.5916, mae: 0.5648, huber: 0.2449, swd: 0.4467, ept: 69.7740
    Epoch [1/50], Test Losses: mse: 1.0624, mae: 0.7591, huber: 0.4094, swd: 0.7351, ept: 46.2655
      Epoch 1 composite train-obj: 0.298261
            Val objective improved inf → 0.2449, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5937, mae: 0.5392, huber: 0.2420, swd: 0.2909, ept: 79.9811
    Epoch [2/50], Val Losses: mse: 0.5767, mae: 0.5666, huber: 0.2514, swd: 0.2190, ept: 53.0416
    Epoch [2/50], Test Losses: mse: 0.5976, mae: 0.5718, huber: 0.2607, swd: 0.1983, ept: 48.8291
      Epoch 2 composite train-obj: 0.242008
            No improvement (0.2514), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4403, mae: 0.4739, huber: 0.1922, swd: 0.1602, ept: 89.1506
    Epoch [3/50], Val Losses: mse: 0.5662, mae: 0.5431, huber: 0.2416, swd: 0.1677, ept: 70.1356
    Epoch [3/50], Test Losses: mse: 0.7299, mae: 0.6571, huber: 0.3167, swd: 0.2300, ept: 36.6555
      Epoch 3 composite train-obj: 0.192201
            Val objective improved 0.2449 → 0.2416, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3929, mae: 0.4496, huber: 0.1752, swd: 0.1277, ept: 92.3120
    Epoch [4/50], Val Losses: mse: 0.7403, mae: 0.6344, huber: 0.3088, swd: 0.1452, ept: 69.2737
    Epoch [4/50], Test Losses: mse: 0.8241, mae: 0.7043, huber: 0.3516, swd: 0.2259, ept: 45.7102
      Epoch 4 composite train-obj: 0.175225
            No improvement (0.3088), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3727, mae: 0.4378, huber: 0.1672, swd: 0.1133, ept: 94.1134
    Epoch [5/50], Val Losses: mse: 0.8201, mae: 0.6998, huber: 0.3451, swd: 0.1843, ept: 53.2022
    Epoch [5/50], Test Losses: mse: 0.7941, mae: 0.6752, huber: 0.3376, swd: 0.2285, ept: 57.3395
      Epoch 5 composite train-obj: 0.167160
            No improvement (0.3451), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3423, mae: 0.4191, huber: 0.1549, swd: 0.0944, ept: 99.3644
    Epoch [6/50], Val Losses: mse: 0.6804, mae: 0.6201, huber: 0.2905, swd: 0.2072, ept: 53.6616
    Epoch [6/50], Test Losses: mse: 0.6308, mae: 0.5841, huber: 0.2725, swd: 0.2129, ept: 57.3125
      Epoch 6 composite train-obj: 0.154932
            No improvement (0.2905), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3205, mae: 0.4044, huber: 0.1459, swd: 0.0792, ept: 101.7681
    Epoch [7/50], Val Losses: mse: 0.7941, mae: 0.6748, huber: 0.3308, swd: 0.1584, ept: 55.0160
    Epoch [7/50], Test Losses: mse: 0.7265, mae: 0.6407, huber: 0.3103, swd: 0.1907, ept: 57.7900
      Epoch 7 composite train-obj: 0.145929
            No improvement (0.3308), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3134, mae: 0.4001, huber: 0.1430, swd: 0.0724, ept: 102.3997
    Epoch [8/50], Val Losses: mse: 0.7392, mae: 0.6492, huber: 0.3115, swd: 0.1885, ept: 46.2424
    Epoch [8/50], Test Losses: mse: 0.6695, mae: 0.6059, huber: 0.2883, swd: 0.2069, ept: 56.0089
      Epoch 8 composite train-obj: 0.142961
    Epoch [8/50], Test Losses: mse: 0.7300, mae: 0.6572, huber: 0.3167, swd: 0.2300, ept: 36.6192
    Best round's Test MSE: 0.7299, MAE: 0.6571, SWD: 0.2300
    Best round's Validation MSE: 0.5662, MAE: 0.5431, SWD: 0.1677
    Best round's Test verification MSE : 0.7300, MAE: 0.6572, SWD: 0.2300
    Time taken: 16.82 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq720_pred336_20250510_1950)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.8854 ± 0.1162
      mae: 0.6929 ± 0.0256
      huber: 0.3533 ± 0.0266
      swd: 0.4664 ± 0.2034
      ept: 37.3660 ± 13.5041
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.5633 ± 0.0049
      mae: 0.5444 ± 0.0038
      huber: 0.2354 ± 0.0063
      swd: 0.2678 ± 0.0778
      ept: 59.2214 ± 13.2461
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 42.41 seconds
    
    Experiment complete: ACL_etth1_seq720_pred336_20250510_1950
    Model: ACL
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 85
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 85
    Validation Batches: 3
    Test Batches: 16
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6880, mae: 0.5937, huber: 0.2801, swd: 0.3892, ept: 86.9062
    Epoch [1/50], Val Losses: mse: 0.6173, mae: 0.5934, huber: 0.2722, swd: 0.2328, ept: 44.8423
    Epoch [1/50], Test Losses: mse: 0.6750, mae: 0.6221, huber: 0.2951, swd: 0.2154, ept: 81.9713
      Epoch 1 composite train-obj: 0.280078
            Val objective improved inf → 0.2722, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5171, mae: 0.5084, huber: 0.2179, swd: 0.2108, ept: 113.9582
    Epoch [2/50], Val Losses: mse: 0.9306, mae: 0.7374, huber: 0.3884, swd: 0.3995, ept: 35.4894
    Epoch [2/50], Test Losses: mse: 0.7821, mae: 0.6852, huber: 0.3386, swd: 0.2636, ept: 43.9645
      Epoch 2 composite train-obj: 0.217884
            No improvement (0.3884), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4592, mae: 0.4750, huber: 0.1956, swd: 0.1715, ept: 118.1583
    Epoch [3/50], Val Losses: mse: 0.7202, mae: 0.6366, huber: 0.3056, swd: 0.2194, ept: 49.4060
    Epoch [3/50], Test Losses: mse: 0.7167, mae: 0.6374, huber: 0.3087, swd: 0.1846, ept: 78.0091
      Epoch 3 composite train-obj: 0.195607
            No improvement (0.3056), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.3947, mae: 0.4433, huber: 0.1730, swd: 0.1205, ept: 128.9638
    Epoch [4/50], Val Losses: mse: 0.8302, mae: 0.6909, huber: 0.3475, swd: 0.2066, ept: 47.6569
    Epoch [4/50], Test Losses: mse: 0.8490, mae: 0.6882, huber: 0.3518, swd: 0.1638, ept: 77.1168
      Epoch 4 composite train-obj: 0.172965
            No improvement (0.3475), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3587, mae: 0.4244, huber: 0.1601, swd: 0.0912, ept: 134.1813
    Epoch [5/50], Val Losses: mse: 0.7509, mae: 0.6593, huber: 0.3198, swd: 0.1475, ept: 45.6274
    Epoch [5/50], Test Losses: mse: 0.7104, mae: 0.6254, huber: 0.3027, swd: 0.1351, ept: 81.6660
      Epoch 5 composite train-obj: 0.160107
            No improvement (0.3198), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3419, mae: 0.4165, huber: 0.1543, swd: 0.0809, ept: 137.5407
    Epoch [6/50], Val Losses: mse: 0.8177, mae: 0.6845, huber: 0.3430, swd: 0.1998, ept: 47.7894
    Epoch [6/50], Test Losses: mse: 0.8250, mae: 0.6774, huber: 0.3430, swd: 0.1597, ept: 64.9444
      Epoch 6 composite train-obj: 0.154252
    Epoch [6/50], Test Losses: mse: 0.6751, mae: 0.6221, huber: 0.2951, swd: 0.2154, ept: 82.2886
    Best round's Test MSE: 0.6750, MAE: 0.6221, SWD: 0.2154
    Best round's Validation MSE: 0.6173, MAE: 0.5934, SWD: 0.2328
    Best round's Test verification MSE : 0.6751, MAE: 0.6221, SWD: 0.2154
    Time taken: 14.28 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6763, mae: 0.5900, huber: 0.2769, swd: 0.3717, ept: 87.9420
    Epoch [1/50], Val Losses: mse: 0.6031, mae: 0.5815, huber: 0.2642, swd: 0.1778, ept: 43.0391
    Epoch [1/50], Test Losses: mse: 0.6735, mae: 0.6202, huber: 0.2941, swd: 0.1898, ept: 79.6152
      Epoch 1 composite train-obj: 0.276858
            Val objective improved inf → 0.2642, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5250, mae: 0.5129, huber: 0.2211, swd: 0.2116, ept: 113.2392
    Epoch [2/50], Val Losses: mse: 0.7807, mae: 0.6797, huber: 0.3301, swd: 0.2259, ept: 51.6968
    Epoch [2/50], Test Losses: mse: 0.7653, mae: 0.6611, huber: 0.3285, swd: 0.2027, ept: 85.0821
      Epoch 2 composite train-obj: 0.221137
            No improvement (0.3301), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4613, mae: 0.4767, huber: 0.1965, swd: 0.1717, ept: 120.8890
    Epoch [3/50], Val Losses: mse: 0.9644, mae: 0.7794, huber: 0.4045, swd: 0.2443, ept: 48.9351
    Epoch [3/50], Test Losses: mse: 0.8382, mae: 0.6943, huber: 0.3538, swd: 0.1875, ept: 78.7312
      Epoch 3 composite train-obj: 0.196502
            No improvement (0.4045), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4091, mae: 0.4505, huber: 0.1783, swd: 0.1335, ept: 127.1595
    Epoch [4/50], Val Losses: mse: 0.8789, mae: 0.7254, huber: 0.3730, swd: 0.2163, ept: 47.8545
    Epoch [4/50], Test Losses: mse: 0.7985, mae: 0.6800, huber: 0.3401, swd: 0.1933, ept: 80.0381
      Epoch 4 composite train-obj: 0.178344
            No improvement (0.3730), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3677, mae: 0.4319, huber: 0.1646, swd: 0.0996, ept: 131.0168
    Epoch [5/50], Val Losses: mse: 0.8281, mae: 0.7213, huber: 0.3588, swd: 0.2339, ept: 45.0481
    Epoch [5/50], Test Losses: mse: 0.7497, mae: 0.6463, huber: 0.3185, swd: 0.1941, ept: 72.7477
      Epoch 5 composite train-obj: 0.164568
            No improvement (0.3588), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3354, mae: 0.4136, huber: 0.1520, swd: 0.0749, ept: 136.9879
    Epoch [6/50], Val Losses: mse: 0.8179, mae: 0.7090, huber: 0.3495, swd: 0.1887, ept: 48.4277
    Epoch [6/50], Test Losses: mse: 0.8353, mae: 0.6835, huber: 0.3496, swd: 0.1635, ept: 78.3000
      Epoch 6 composite train-obj: 0.151972
    Epoch [6/50], Test Losses: mse: 0.6734, mae: 0.6202, huber: 0.2941, swd: 0.1897, ept: 79.3242
    Best round's Test MSE: 0.6735, MAE: 0.6202, SWD: 0.1898
    Best round's Validation MSE: 0.6031, MAE: 0.5815, SWD: 0.1778
    Best round's Test verification MSE : 0.6734, MAE: 0.6202, SWD: 0.1897
    Time taken: 14.56 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6772, mae: 0.5893, huber: 0.2767, swd: 0.3845, ept: 90.1198
    Epoch [1/50], Val Losses: mse: 0.5980, mae: 0.5814, huber: 0.2617, swd: 0.2337, ept: 52.5891
    Epoch [1/50], Test Losses: mse: 0.6842, mae: 0.6278, huber: 0.2986, swd: 0.2547, ept: 86.9364
      Epoch 1 composite train-obj: 0.276687
            Val objective improved inf → 0.2617, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5134, mae: 0.5073, huber: 0.2169, swd: 0.2162, ept: 113.5218
    Epoch [2/50], Val Losses: mse: 0.7962, mae: 0.6682, huber: 0.3396, swd: 0.3675, ept: 33.9622
    Epoch [2/50], Test Losses: mse: 0.7370, mae: 0.6635, huber: 0.3224, swd: 0.2568, ept: 44.3855
      Epoch 2 composite train-obj: 0.216870
            No improvement (0.3396), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4471, mae: 0.4710, huber: 0.1920, swd: 0.1689, ept: 120.6724
    Epoch [3/50], Val Losses: mse: 0.7239, mae: 0.6422, huber: 0.3096, swd: 0.2326, ept: 50.4210
    Epoch [3/50], Test Losses: mse: 0.7203, mae: 0.6444, huber: 0.3121, swd: 0.2087, ept: 82.3199
      Epoch 3 composite train-obj: 0.191953
            No improvement (0.3096), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.3943, mae: 0.4420, huber: 0.1724, swd: 0.1230, ept: 126.8802
    Epoch [4/50], Val Losses: mse: 0.8282, mae: 0.6997, huber: 0.3508, swd: 0.2685, ept: 43.3334
    Epoch [4/50], Test Losses: mse: 0.8822, mae: 0.7172, huber: 0.3702, swd: 0.1900, ept: 65.5336
      Epoch 4 composite train-obj: 0.172450
            No improvement (0.3508), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3601, mae: 0.4250, huber: 0.1606, swd: 0.0949, ept: 131.8193
    Epoch [5/50], Val Losses: mse: 0.7077, mae: 0.6371, huber: 0.3008, swd: 0.1740, ept: 49.9768
    Epoch [5/50], Test Losses: mse: 0.8233, mae: 0.6833, huber: 0.3466, swd: 0.1832, ept: 87.8428
      Epoch 5 composite train-obj: 0.160626
            No improvement (0.3008), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3300, mae: 0.4084, huber: 0.1492, swd: 0.0738, ept: 139.2380
    Epoch [6/50], Val Losses: mse: 0.7115, mae: 0.6438, huber: 0.3053, swd: 0.1867, ept: 49.3439
    Epoch [6/50], Test Losses: mse: 0.7960, mae: 0.6673, huber: 0.3347, swd: 0.1984, ept: 78.4108
      Epoch 6 composite train-obj: 0.149153
    Epoch [6/50], Test Losses: mse: 0.6841, mae: 0.6278, huber: 0.2986, swd: 0.2547, ept: 87.2214
    Best round's Test MSE: 0.6842, MAE: 0.6278, SWD: 0.2547
    Best round's Validation MSE: 0.5980, MAE: 0.5814, SWD: 0.2337
    Best round's Test verification MSE : 0.6841, MAE: 0.6278, SWD: 0.2547
    Time taken: 17.44 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq720_pred720_20250510_1951)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6776 ± 0.0047
      mae: 0.6234 ± 0.0032
      huber: 0.2959 ± 0.0019
      swd: 0.2200 ± 0.0267
      ept: 82.8410 ± 3.0515
      count: 3.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.6061 ± 0.0082
      mae: 0.5854 ± 0.0057
      huber: 0.2661 ± 0.0045
      swd: 0.2148 ± 0.0261
      ept: 46.8235 ± 4.1428
      count: 3.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 46.40 seconds
    
    Experiment complete: ACL_etth1_seq720_pred720_20250510_1951
    Model: ACL
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### Timemixer

#### pred=96


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 96
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
    
    Epoch [1/50], Train Losses: mse: 0.4041, mae: 0.4453, huber: 0.1760, swd: 0.1208, ept: 44.8764
    Epoch [1/50], Val Losses: mse: 0.3362, mae: 0.4012, huber: 0.1490, swd: 0.0971, ept: 49.5800
    Epoch [1/50], Test Losses: mse: 0.4372, mae: 0.4659, huber: 0.1904, swd: 0.1292, ept: 36.1966
      Epoch 1 composite train-obj: 0.175954
            Val objective improved inf → 0.1490, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2963, mae: 0.3802, huber: 0.1340, swd: 0.0852, ept: 53.0168
    Epoch [2/50], Val Losses: mse: 0.3465, mae: 0.4124, huber: 0.1545, swd: 0.0966, ept: 49.6545
    Epoch [2/50], Test Losses: mse: 0.4397, mae: 0.4756, huber: 0.1936, swd: 0.1296, ept: 35.3376
      Epoch 2 composite train-obj: 0.133968
            No improvement (0.1545), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.2373, mae: 0.3423, huber: 0.1098, swd: 0.0592, ept: 56.4004
    Epoch [3/50], Val Losses: mse: 0.3725, mae: 0.4274, huber: 0.1645, swd: 0.0915, ept: 48.1821
    Epoch [3/50], Test Losses: mse: 0.4420, mae: 0.4788, huber: 0.1944, swd: 0.1204, ept: 35.1758
      Epoch 3 composite train-obj: 0.109833
            No improvement (0.1645), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.1957, mae: 0.3114, huber: 0.0919, swd: 0.0425, ept: 60.4184
    Epoch [4/50], Val Losses: mse: 0.3948, mae: 0.4406, huber: 0.1734, swd: 0.0986, ept: 47.9358
    Epoch [4/50], Test Losses: mse: 0.4506, mae: 0.4871, huber: 0.1990, swd: 0.1296, ept: 34.3311
      Epoch 4 composite train-obj: 0.091885
            No improvement (0.1734), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.1656, mae: 0.2891, huber: 0.0789, swd: 0.0319, ept: 64.1332
    Epoch [5/50], Val Losses: mse: 0.4223, mae: 0.4560, huber: 0.1839, swd: 0.0979, ept: 46.4947
    Epoch [5/50], Test Losses: mse: 0.4763, mae: 0.5023, huber: 0.2086, swd: 0.1292, ept: 33.4434
      Epoch 5 composite train-obj: 0.078926
            No improvement (0.1839), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.1392, mae: 0.2681, huber: 0.0672, swd: 0.0232, ept: 68.0731
    Epoch [6/50], Val Losses: mse: 0.4320, mae: 0.4594, huber: 0.1863, swd: 0.0981, ept: 45.2927
    Epoch [6/50], Test Losses: mse: 0.4909, mae: 0.5073, huber: 0.2128, swd: 0.1246, ept: 33.3641
      Epoch 6 composite train-obj: 0.067207
    Epoch [6/50], Test Losses: mse: 0.4372, mae: 0.4659, huber: 0.1904, swd: 0.1292, ept: 36.1966
    Best round's Test MSE: 0.4372, MAE: 0.4659, SWD: 0.1292
    Best round's Validation MSE: 0.3362, MAE: 0.4012, SWD: 0.0971
    Best round's Test verification MSE : 0.4372, MAE: 0.4659, SWD: 0.1292
    Time taken: 71.27 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4474, mae: 0.4682, huber: 0.1918, swd: 0.1036, ept: 41.2276
    Epoch [1/50], Val Losses: mse: 0.3415, mae: 0.4071, huber: 0.1518, swd: 0.0937, ept: 49.5312
    Epoch [1/50], Test Losses: mse: 0.4440, mae: 0.4734, huber: 0.1941, swd: 0.1241, ept: 35.5333
      Epoch 1 composite train-obj: 0.191803
            Val objective improved inf → 0.1518, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2956, mae: 0.3804, huber: 0.1337, swd: 0.0830, ept: 53.0025
    Epoch [2/50], Val Losses: mse: 0.3484, mae: 0.4179, huber: 0.1562, swd: 0.1038, ept: 49.2143
    Epoch [2/50], Test Losses: mse: 0.4452, mae: 0.4834, huber: 0.1970, swd: 0.1352, ept: 34.8390
      Epoch 2 composite train-obj: 0.133652
            No improvement (0.1562), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.2316, mae: 0.3384, huber: 0.1076, swd: 0.0574, ept: 57.2135
    Epoch [3/50], Val Losses: mse: 0.3764, mae: 0.4409, huber: 0.1684, swd: 0.0994, ept: 47.0544
    Epoch [3/50], Test Losses: mse: 0.4647, mae: 0.4935, huber: 0.2037, swd: 0.1127, ept: 35.3213
      Epoch 3 composite train-obj: 0.107596
            No improvement (0.1684), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.1927, mae: 0.3101, huber: 0.0910, swd: 0.0412, ept: 60.7952
    Epoch [4/50], Val Losses: mse: 0.4000, mae: 0.4547, huber: 0.1791, swd: 0.1053, ept: 45.8221
    Epoch [4/50], Test Losses: mse: 0.4853, mae: 0.5090, huber: 0.2133, swd: 0.1240, ept: 33.7172
      Epoch 4 composite train-obj: 0.091026
            No improvement (0.1791), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.1655, mae: 0.2902, huber: 0.0793, swd: 0.0303, ept: 64.1709
    Epoch [5/50], Val Losses: mse: 0.4182, mae: 0.4666, huber: 0.1861, swd: 0.0950, ept: 44.5198
    Epoch [5/50], Test Losses: mse: 0.4964, mae: 0.5155, huber: 0.2177, swd: 0.1163, ept: 32.8367
      Epoch 5 composite train-obj: 0.079270
            No improvement (0.1861), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.1407, mae: 0.2703, huber: 0.0681, swd: 0.0219, ept: 67.7522
    Epoch [6/50], Val Losses: mse: 0.4336, mae: 0.4747, huber: 0.1919, swd: 0.0970, ept: 43.7174
    Epoch [6/50], Test Losses: mse: 0.5233, mae: 0.5311, huber: 0.2284, swd: 0.1263, ept: 31.5639
      Epoch 6 composite train-obj: 0.068093
    Epoch [6/50], Test Losses: mse: 0.4440, mae: 0.4734, huber: 0.1941, swd: 0.1241, ept: 35.5333
    Best round's Test MSE: 0.4440, MAE: 0.4734, SWD: 0.1241
    Best round's Validation MSE: 0.3415, MAE: 0.4071, SWD: 0.0937
    Best round's Test verification MSE : 0.4440, MAE: 0.4734, SWD: 0.1241
    Time taken: 69.99 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4039, mae: 0.4446, huber: 0.1757, swd: 0.1188, ept: 46.0669
    Epoch [1/50], Val Losses: mse: 0.3277, mae: 0.3948, huber: 0.1458, swd: 0.0875, ept: 49.6626
    Epoch [1/50], Test Losses: mse: 0.4284, mae: 0.4617, huber: 0.1880, swd: 0.1240, ept: 36.3457
      Epoch 1 composite train-obj: 0.175714
            Val objective improved inf → 0.1458, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3083, mae: 0.3869, huber: 0.1385, swd: 0.0864, ept: 53.1575
    Epoch [2/50], Val Losses: mse: 0.3417, mae: 0.4080, huber: 0.1518, swd: 0.0878, ept: 49.7646
    Epoch [2/50], Test Losses: mse: 0.4363, mae: 0.4705, huber: 0.1916, swd: 0.1259, ept: 35.7677
      Epoch 2 composite train-obj: 0.138476
            No improvement (0.1518), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.2580, mae: 0.3569, huber: 0.1187, swd: 0.0672, ept: 55.7301
    Epoch [3/50], Val Losses: mse: 0.3594, mae: 0.4259, huber: 0.1609, swd: 0.0853, ept: 48.1084
    Epoch [3/50], Test Losses: mse: 0.4446, mae: 0.4832, huber: 0.1969, swd: 0.1222, ept: 35.3076
      Epoch 3 composite train-obj: 0.118700
            No improvement (0.1609), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2143, mae: 0.3260, huber: 0.1000, swd: 0.0484, ept: 58.9675
    Epoch [4/50], Val Losses: mse: 0.4198, mae: 0.4524, huber: 0.1817, swd: 0.0973, ept: 47.4775
    Epoch [4/50], Test Losses: mse: 0.4516, mae: 0.4866, huber: 0.1995, swd: 0.1203, ept: 34.3974
      Epoch 4 composite train-obj: 0.099989
            No improvement (0.1817), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.1769, mae: 0.2966, huber: 0.0835, swd: 0.0342, ept: 62.9890
    Epoch [5/50], Val Losses: mse: 0.4525, mae: 0.4741, huber: 0.1954, swd: 0.1081, ept: 45.5640
    Epoch [5/50], Test Losses: mse: 0.4675, mae: 0.5002, huber: 0.2066, swd: 0.1162, ept: 33.5438
      Epoch 5 composite train-obj: 0.083478
            No improvement (0.1954), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.1502, mae: 0.2753, huber: 0.0718, swd: 0.0259, ept: 66.6659
    Epoch [6/50], Val Losses: mse: 0.4727, mae: 0.4846, huber: 0.2022, swd: 0.1118, ept: 44.5552
    Epoch [6/50], Test Losses: mse: 0.4858, mae: 0.5084, huber: 0.2127, swd: 0.1158, ept: 32.9211
      Epoch 6 composite train-obj: 0.071753
    Epoch [6/50], Test Losses: mse: 0.4284, mae: 0.4617, huber: 0.1880, swd: 0.1240, ept: 36.3457
    Best round's Test MSE: 0.4284, MAE: 0.4617, SWD: 0.1240
    Best round's Validation MSE: 0.3277, MAE: 0.3948, SWD: 0.0875
    Best round's Test verification MSE : 0.4284, MAE: 0.4617, SWD: 0.1240
    Time taken: 70.46 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq720_pred96_20250510_1952)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4365 ± 0.0064
      mae: 0.4670 ± 0.0048
      huber: 0.1908 ± 0.0025
      swd: 0.1258 ± 0.0024
      ept: 36.0252 ± 0.3531
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3351 ± 0.0057
      mae: 0.4010 ± 0.0050
      huber: 0.1488 ± 0.0024
      swd: 0.0928 ± 0.0040
      ept: 49.5913 ± 0.0542
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 211.87 seconds
    
    Experiment complete: TimeMixer_etth1_seq720_pred96_20250510_1952
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.5240, mae: 0.5045, huber: 0.2185, swd: 0.1330, ept: 59.5097
    Epoch [1/50], Val Losses: mse: 0.3967, mae: 0.4393, huber: 0.1728, swd: 0.1032, ept: 66.5636
    Epoch [1/50], Test Losses: mse: 0.4865, mae: 0.5080, huber: 0.2147, swd: 0.1349, ept: 47.5793
      Epoch 1 composite train-obj: 0.218464
            Val objective improved inf → 0.1728, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3683, mae: 0.4223, huber: 0.1614, swd: 0.1054, ept: 76.8229
    Epoch [2/50], Val Losses: mse: 0.4388, mae: 0.4693, huber: 0.1905, swd: 0.1143, ept: 66.0262
    Epoch [2/50], Test Losses: mse: 0.5095, mae: 0.5266, huber: 0.2245, swd: 0.1339, ept: 50.1313
      Epoch 2 composite train-obj: 0.161422
            No improvement (0.1905), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.2926, mae: 0.3811, huber: 0.1332, swd: 0.0742, ept: 82.5782
    Epoch [3/50], Val Losses: mse: 0.4253, mae: 0.4644, huber: 0.1856, swd: 0.0984, ept: 64.9312
    Epoch [3/50], Test Losses: mse: 0.5043, mae: 0.5232, huber: 0.2220, swd: 0.1154, ept: 47.2762
      Epoch 3 composite train-obj: 0.133167
            No improvement (0.1856), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2437, mae: 0.3494, huber: 0.1132, swd: 0.0528, ept: 88.5689
    Epoch [4/50], Val Losses: mse: 0.4793, mae: 0.4957, huber: 0.2064, swd: 0.1026, ept: 60.8115
    Epoch [4/50], Test Losses: mse: 0.5372, mae: 0.5394, huber: 0.2331, swd: 0.1109, ept: 46.2143
      Epoch 4 composite train-obj: 0.113233
            No improvement (0.2064), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2127, mae: 0.3277, huber: 0.1001, swd: 0.0395, ept: 94.0872
    Epoch [5/50], Val Losses: mse: 0.4873, mae: 0.4956, huber: 0.2076, swd: 0.1039, ept: 63.1268
    Epoch [5/50], Test Losses: mse: 0.5449, mae: 0.5429, huber: 0.2358, swd: 0.1231, ept: 46.0497
      Epoch 5 composite train-obj: 0.100128
            No improvement (0.2076), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.1892, mae: 0.3096, huber: 0.0898, swd: 0.0306, ept: 99.1625
    Epoch [6/50], Val Losses: mse: 0.5081, mae: 0.5062, huber: 0.2149, swd: 0.1075, ept: 61.2566
    Epoch [6/50], Test Losses: mse: 0.5511, mae: 0.5459, huber: 0.2378, swd: 0.1242, ept: 46.4926
      Epoch 6 composite train-obj: 0.089790
    Epoch [6/50], Test Losses: mse: 0.4865, mae: 0.5080, huber: 0.2147, swd: 0.1349, ept: 47.5793
    Best round's Test MSE: 0.4865, MAE: 0.5080, SWD: 0.1349
    Best round's Validation MSE: 0.3967, MAE: 0.4393, SWD: 0.1032
    Best round's Test verification MSE : 0.4865, MAE: 0.5080, SWD: 0.1349
    Time taken: 72.20 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5473, mae: 0.5161, huber: 0.2265, swd: 0.1282, ept: 55.4399
    Epoch [1/50], Val Losses: mse: 0.4380, mae: 0.4625, huber: 0.1882, swd: 0.1075, ept: 66.7928
    Epoch [1/50], Test Losses: mse: 0.5221, mae: 0.5264, huber: 0.2276, swd: 0.1227, ept: 49.4610
      Epoch 1 composite train-obj: 0.226471
            Val objective improved inf → 0.1882, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3517, mae: 0.4135, huber: 0.1550, swd: 0.0980, ept: 77.2937
    Epoch [2/50], Val Losses: mse: 0.4132, mae: 0.4541, huber: 0.1802, swd: 0.0898, ept: 66.4959
    Epoch [2/50], Test Losses: mse: 0.4974, mae: 0.5215, huber: 0.2205, swd: 0.1227, ept: 46.8073
      Epoch 2 composite train-obj: 0.155008
            Val objective improved 0.1882 → 0.1802, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2677, mae: 0.3651, huber: 0.1227, swd: 0.0625, ept: 85.7379
    Epoch [3/50], Val Losses: mse: 0.4680, mae: 0.4853, huber: 0.2005, swd: 0.0974, ept: 64.3332
    Epoch [3/50], Test Losses: mse: 0.5205, mae: 0.5312, huber: 0.2276, swd: 0.1190, ept: 46.3870
      Epoch 3 composite train-obj: 0.122703
            No improvement (0.2005), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.2213, mae: 0.3332, huber: 0.1034, swd: 0.0438, ept: 93.2345
    Epoch [4/50], Val Losses: mse: 0.4847, mae: 0.4927, huber: 0.2068, swd: 0.0992, ept: 65.5077
    Epoch [4/50], Test Losses: mse: 0.5387, mae: 0.5424, huber: 0.2348, swd: 0.1223, ept: 44.8118
      Epoch 4 composite train-obj: 0.103440
            No improvement (0.2068), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.1921, mae: 0.3126, huber: 0.0911, swd: 0.0336, ept: 98.9145
    Epoch [5/50], Val Losses: mse: 0.4984, mae: 0.5020, huber: 0.2118, swd: 0.0970, ept: 65.6325
    Epoch [5/50], Test Losses: mse: 0.5403, mae: 0.5429, huber: 0.2351, swd: 0.1133, ept: 42.5154
      Epoch 5 composite train-obj: 0.091118
            No improvement (0.2118), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.1676, mae: 0.2943, huber: 0.0804, swd: 0.0259, ept: 104.8649
    Epoch [6/50], Val Losses: mse: 0.5241, mae: 0.5152, huber: 0.2207, swd: 0.0996, ept: 63.6034
    Epoch [6/50], Test Losses: mse: 0.5622, mae: 0.5530, huber: 0.2427, swd: 0.1149, ept: 41.2350
      Epoch 6 composite train-obj: 0.080408
            No improvement (0.2207), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.1500, mae: 0.2803, huber: 0.0725, swd: 0.0216, ept: 110.2942
    Epoch [7/50], Val Losses: mse: 0.5223, mae: 0.5134, huber: 0.2198, swd: 0.0964, ept: 63.5259
    Epoch [7/50], Test Losses: mse: 0.5597, mae: 0.5495, huber: 0.2408, swd: 0.1155, ept: 41.7648
      Epoch 7 composite train-obj: 0.072521
    Epoch [7/50], Test Losses: mse: 0.4974, mae: 0.5215, huber: 0.2205, swd: 0.1227, ept: 46.8073
    Best round's Test MSE: 0.4974, MAE: 0.5215, SWD: 0.1227
    Best round's Validation MSE: 0.4132, MAE: 0.4541, SWD: 0.0898
    Best round's Test verification MSE : 0.4974, MAE: 0.5215, SWD: 0.1227
    Time taken: 83.36 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4739, mae: 0.4753, huber: 0.1995, swd: 0.1339, ept: 70.0090
    Epoch [1/50], Val Losses: mse: 0.3954, mae: 0.4375, huber: 0.1723, swd: 0.0858, ept: 67.5771
    Epoch [1/50], Test Losses: mse: 0.4709, mae: 0.4927, huber: 0.2070, swd: 0.0996, ept: 50.6070
      Epoch 1 composite train-obj: 0.199496
            Val objective improved inf → 0.1723, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3787, mae: 0.4236, huber: 0.1640, swd: 0.1056, ept: 79.1716
    Epoch [2/50], Val Losses: mse: 0.4362, mae: 0.4604, huber: 0.1875, swd: 0.0894, ept: 66.8286
    Epoch [2/50], Test Losses: mse: 0.4829, mae: 0.5078, huber: 0.2137, swd: 0.1037, ept: 49.4987
      Epoch 2 composite train-obj: 0.164032
            No improvement (0.1875), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.2909, mae: 0.3790, huber: 0.1321, swd: 0.0710, ept: 83.9021
    Epoch [3/50], Val Losses: mse: 0.4942, mae: 0.4960, huber: 0.2105, swd: 0.1019, ept: 61.8706
    Epoch [3/50], Test Losses: mse: 0.5143, mae: 0.5304, huber: 0.2268, swd: 0.1170, ept: 47.2955
      Epoch 3 composite train-obj: 0.132144
            No improvement (0.2105), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2343, mae: 0.3417, huber: 0.1090, swd: 0.0478, ept: 90.9834
    Epoch [4/50], Val Losses: mse: 0.5465, mae: 0.5201, huber: 0.2276, swd: 0.1021, ept: 60.2246
    Epoch [4/50], Test Losses: mse: 0.5486, mae: 0.5453, huber: 0.2387, swd: 0.1054, ept: 45.0720
      Epoch 4 composite train-obj: 0.108959
            No improvement (0.2276), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.1958, mae: 0.3136, huber: 0.0924, swd: 0.0325, ept: 98.2743
    Epoch [5/50], Val Losses: mse: 0.5407, mae: 0.5166, huber: 0.2248, swd: 0.1027, ept: 58.1236
    Epoch [5/50], Test Losses: mse: 0.5588, mae: 0.5545, huber: 0.2439, swd: 0.1061, ept: 44.8641
      Epoch 5 composite train-obj: 0.092428
            No improvement (0.2248), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.1681, mae: 0.2917, huber: 0.0802, swd: 0.0253, ept: 105.9622
    Epoch [6/50], Val Losses: mse: 0.5752, mae: 0.5272, huber: 0.2339, swd: 0.1065, ept: 56.3975
    Epoch [6/50], Test Losses: mse: 0.5731, mae: 0.5608, huber: 0.2491, swd: 0.1056, ept: 45.4698
      Epoch 6 composite train-obj: 0.080190
    Epoch [6/50], Test Losses: mse: 0.4709, mae: 0.4927, huber: 0.2070, swd: 0.0996, ept: 50.6070
    Best round's Test MSE: 0.4709, MAE: 0.4927, SWD: 0.0996
    Best round's Validation MSE: 0.3954, MAE: 0.4375, SWD: 0.0858
    Best round's Test verification MSE : 0.4709, MAE: 0.4927, SWD: 0.0996
    Time taken: 71.62 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq720_pred196_20250510_1955)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4849 ± 0.0109
      mae: 0.5074 ± 0.0118
      huber: 0.2141 ± 0.0055
      swd: 0.1191 ± 0.0146
      ept: 48.3312 ± 1.6398
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4017 ± 0.0081
      mae: 0.4436 ± 0.0075
      huber: 0.1751 ± 0.0036
      swd: 0.0929 ± 0.0075
      ept: 66.8789 ± 0.4945
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 227.33 seconds
    
    Experiment complete: TimeMixer_etth1_seq720_pred196_20250510_1955
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 88
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 88
    Validation Batches: 6
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5660, mae: 0.5253, huber: 0.2340, swd: 0.1435, ept: 77.0954
    Epoch [1/50], Val Losses: mse: 0.4042, mae: 0.4415, huber: 0.1751, swd: 0.0878, ept: 87.9825
    Epoch [1/50], Test Losses: mse: 0.5402, mae: 0.5433, huber: 0.2385, swd: 0.1429, ept: 61.1783
      Epoch 1 composite train-obj: 0.233955
            Val objective improved inf → 0.1751, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4138, mae: 0.4471, huber: 0.1782, swd: 0.1148, ept: 100.5480
    Epoch [2/50], Val Losses: mse: 0.4360, mae: 0.4641, huber: 0.1878, swd: 0.0976, ept: 89.6399
    Epoch [2/50], Test Losses: mse: 0.5500, mae: 0.5487, huber: 0.2414, swd: 0.1249, ept: 64.1032
      Epoch 2 composite train-obj: 0.178210
            No improvement (0.1878), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3310, mae: 0.4052, huber: 0.1482, swd: 0.0834, ept: 108.8410
    Epoch [3/50], Val Losses: mse: 0.5244, mae: 0.5116, huber: 0.2198, swd: 0.1009, ept: 82.6132
    Epoch [3/50], Test Losses: mse: 0.5591, mae: 0.5569, huber: 0.2450, swd: 0.1109, ept: 58.6256
      Epoch 3 composite train-obj: 0.148243
            No improvement (0.2198), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2696, mae: 0.3676, huber: 0.1240, swd: 0.0570, ept: 119.5361
    Epoch [4/50], Val Losses: mse: 0.4947, mae: 0.4973, huber: 0.2090, swd: 0.0908, ept: 81.2704
    Epoch [4/50], Test Losses: mse: 0.5458, mae: 0.5491, huber: 0.2404, swd: 0.1124, ept: 58.6183
      Epoch 4 composite train-obj: 0.124029
            No improvement (0.2090), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2340, mae: 0.3437, huber: 0.1093, swd: 0.0418, ept: 129.9849
    Epoch [5/50], Val Losses: mse: 0.5384, mae: 0.5207, huber: 0.2258, swd: 0.0896, ept: 78.0824
    Epoch [5/50], Test Losses: mse: 0.6008, mae: 0.5806, huber: 0.2623, swd: 0.1228, ept: 53.9147
      Epoch 5 composite train-obj: 0.109276
            No improvement (0.2258), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2028, mae: 0.3215, huber: 0.0960, swd: 0.0310, ept: 138.2064
    Epoch [6/50], Val Losses: mse: 0.5522, mae: 0.5179, huber: 0.2256, swd: 0.0862, ept: 78.1751
    Epoch [6/50], Test Losses: mse: 0.6056, mae: 0.5779, huber: 0.2625, swd: 0.0990, ept: 53.5552
      Epoch 6 composite train-obj: 0.095986
    Epoch [6/50], Test Losses: mse: 0.5402, mae: 0.5433, huber: 0.2385, swd: 0.1429, ept: 61.1783
    Best round's Test MSE: 0.5402, MAE: 0.5433, SWD: 0.1429
    Best round's Validation MSE: 0.4042, MAE: 0.4415, SWD: 0.0878
    Best round's Test verification MSE : 0.5402, MAE: 0.5433, SWD: 0.1429
    Time taken: 72.26 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6442, mae: 0.5637, huber: 0.2613, swd: 0.1408, ept: 63.0129
    Epoch [1/50], Val Losses: mse: 0.4023, mae: 0.4403, huber: 0.1741, swd: 0.0751, ept: 86.8483
    Epoch [1/50], Test Losses: mse: 0.5248, mae: 0.5254, huber: 0.2306, swd: 0.1279, ept: 62.1382
      Epoch 1 composite train-obj: 0.261310
            Val objective improved inf → 0.1741, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4187, mae: 0.4513, huber: 0.1805, swd: 0.1123, ept: 96.1510
    Epoch [2/50], Val Losses: mse: 0.4711, mae: 0.4832, huber: 0.2004, swd: 0.0865, ept: 78.4565
    Epoch [2/50], Test Losses: mse: 0.5291, mae: 0.5371, huber: 0.2325, swd: 0.1115, ept: 60.7222
      Epoch 2 composite train-obj: 0.180481
            No improvement (0.2004), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3234, mae: 0.3990, huber: 0.1445, swd: 0.0764, ept: 107.6733
    Epoch [3/50], Val Losses: mse: 0.4845, mae: 0.5012, huber: 0.2089, swd: 0.0815, ept: 79.1399
    Epoch [3/50], Test Losses: mse: 0.5553, mae: 0.5537, huber: 0.2440, swd: 0.1249, ept: 56.9770
      Epoch 3 composite train-obj: 0.144496
            No improvement (0.2089), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2623, mae: 0.3623, huber: 0.1208, swd: 0.0528, ept: 118.7208
    Epoch [4/50], Val Losses: mse: 0.5784, mae: 0.5467, huber: 0.2420, swd: 0.0969, ept: 76.0086
    Epoch [4/50], Test Losses: mse: 0.6032, mae: 0.5727, huber: 0.2596, swd: 0.1139, ept: 56.2213
      Epoch 4 composite train-obj: 0.120781
            No improvement (0.2420), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2346, mae: 0.3446, huber: 0.1095, swd: 0.0427, ept: 126.4018
    Epoch [5/50], Val Losses: mse: 0.5579, mae: 0.5330, huber: 0.2327, swd: 0.0819, ept: 74.6794
    Epoch [5/50], Test Losses: mse: 0.5497, mae: 0.5472, huber: 0.2400, swd: 0.1019, ept: 51.5678
      Epoch 5 composite train-obj: 0.109470
            No improvement (0.2327), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2090, mae: 0.3268, huber: 0.0987, swd: 0.0334, ept: 133.0564
    Epoch [6/50], Val Losses: mse: 0.5692, mae: 0.5396, huber: 0.2371, swd: 0.0853, ept: 74.4571
    Epoch [6/50], Test Losses: mse: 0.5760, mae: 0.5631, huber: 0.2510, swd: 0.1075, ept: 51.4953
      Epoch 6 composite train-obj: 0.098656
    Epoch [6/50], Test Losses: mse: 0.5248, mae: 0.5254, huber: 0.2306, swd: 0.1279, ept: 62.1382
    Best round's Test MSE: 0.5248, MAE: 0.5254, SWD: 0.1279
    Best round's Validation MSE: 0.4023, MAE: 0.4403, SWD: 0.0751
    Best round's Test verification MSE : 0.5248, MAE: 0.5254, SWD: 0.1279
    Time taken: 72.44 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5499, mae: 0.5155, huber: 0.2276, swd: 0.1518, ept: 84.0185
    Epoch [1/50], Val Losses: mse: 0.4439, mae: 0.4606, huber: 0.1881, swd: 0.0864, ept: 85.4771
    Epoch [1/50], Test Losses: mse: 0.5151, mae: 0.5193, huber: 0.2255, swd: 0.0918, ept: 63.6078
      Epoch 1 composite train-obj: 0.227604
            Val objective improved inf → 0.1881, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4386, mae: 0.4588, huber: 0.1869, swd: 0.1273, ept: 98.9234
    Epoch [2/50], Val Losses: mse: 0.5713, mae: 0.5310, huber: 0.2341, swd: 0.1042, ept: 82.7154
    Epoch [2/50], Test Losses: mse: 0.5510, mae: 0.5498, huber: 0.2408, swd: 0.1110, ept: 63.8885
      Epoch 2 composite train-obj: 0.186874
            No improvement (0.2341), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3494, mae: 0.4146, huber: 0.1549, swd: 0.0937, ept: 106.0743
    Epoch [3/50], Val Losses: mse: 0.5577, mae: 0.5267, huber: 0.2305, swd: 0.0981, ept: 78.1497
    Epoch [3/50], Test Losses: mse: 0.5375, mae: 0.5402, huber: 0.2360, swd: 0.1060, ept: 60.5662
      Epoch 3 composite train-obj: 0.154912
            No improvement (0.2305), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2880, mae: 0.3777, huber: 0.1309, swd: 0.0641, ept: 114.9163
    Epoch [4/50], Val Losses: mse: 0.5964, mae: 0.5488, huber: 0.2450, swd: 0.1030, ept: 74.2601
    Epoch [4/50], Test Losses: mse: 0.5746, mae: 0.5587, huber: 0.2495, swd: 0.1058, ept: 55.5978
      Epoch 4 composite train-obj: 0.130908
            No improvement (0.2450), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2339, mae: 0.3433, huber: 0.1092, swd: 0.0411, ept: 126.1855
    Epoch [5/50], Val Losses: mse: 0.5398, mae: 0.5250, huber: 0.2259, swd: 0.0915, ept: 71.9794
    Epoch [5/50], Test Losses: mse: 0.5668, mae: 0.5587, huber: 0.2484, swd: 0.1090, ept: 53.3941
      Epoch 5 composite train-obj: 0.109228
            No improvement (0.2259), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2131, mae: 0.3261, huber: 0.0993, swd: 0.0351, ept: 134.0903
    Epoch [6/50], Val Losses: mse: 0.6120, mae: 0.5551, huber: 0.2484, swd: 0.1016, ept: 71.1477
    Epoch [6/50], Test Losses: mse: 0.6124, mae: 0.5829, huber: 0.2650, swd: 0.1025, ept: 51.1155
      Epoch 6 composite train-obj: 0.099277
    Epoch [6/50], Test Losses: mse: 0.5151, mae: 0.5193, huber: 0.2255, swd: 0.0918, ept: 63.6078
    Best round's Test MSE: 0.5151, MAE: 0.5193, SWD: 0.0918
    Best round's Validation MSE: 0.4439, MAE: 0.4606, SWD: 0.0864
    Best round's Test verification MSE : 0.5151, MAE: 0.5193, SWD: 0.0918
    Time taken: 72.13 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq720_pred336_20250510_1959)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5267 ± 0.0103
      mae: 0.5293 ± 0.0102
      huber: 0.2315 ± 0.0054
      swd: 0.1209 ± 0.0214
      ept: 62.3081 ± 0.9991
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4168 ± 0.0191
      mae: 0.4475 ± 0.0093
      huber: 0.1791 ± 0.0064
      swd: 0.0831 ± 0.0057
      ept: 86.7693 ± 1.0243
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 217.00 seconds
    
    Experiment complete: TimeMixer_etth1_seq720_pred336_20250510_1959
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 85
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 85
    Validation Batches: 3
    Test Batches: 16
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7526, mae: 0.6146, huber: 0.2997, swd: 0.1782, ept: 85.2404
    Epoch [1/50], Val Losses: mse: 0.4392, mae: 0.4609, huber: 0.1890, swd: 0.0744, ept: 132.8562
    Epoch [1/50], Test Losses: mse: 0.7131, mae: 0.6410, huber: 0.3096, swd: 0.1841, ept: 83.0753
      Epoch 1 composite train-obj: 0.299668
            Val objective improved inf → 0.1890, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5589, mae: 0.5242, huber: 0.2311, swd: 0.1685, ept: 115.8214
    Epoch [2/50], Val Losses: mse: 0.6523, mae: 0.5630, huber: 0.2608, swd: 0.0986, ept: 130.5526
    Epoch [2/50], Test Losses: mse: 0.7855, mae: 0.6809, huber: 0.3355, swd: 0.1612, ept: 86.3943
      Epoch 2 composite train-obj: 0.231119
            No improvement (0.2608), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4681, mae: 0.4798, huber: 0.1988, swd: 0.1474, ept: 129.0804
    Epoch [3/50], Val Losses: mse: 0.5203, mae: 0.5018, huber: 0.2158, swd: 0.0705, ept: 131.9404
    Epoch [3/50], Test Losses: mse: 0.7311, mae: 0.6472, huber: 0.3143, swd: 0.1255, ept: 90.4319
      Epoch 3 composite train-obj: 0.198785
            No improvement (0.2158), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.3856, mae: 0.4402, huber: 0.1700, swd: 0.1065, ept: 140.5939
    Epoch [4/50], Val Losses: mse: 0.7288, mae: 0.5962, huber: 0.2869, swd: 0.0910, ept: 134.6173
    Epoch [4/50], Test Losses: mse: 0.9155, mae: 0.7292, huber: 0.3786, swd: 0.1346, ept: 70.8758
      Epoch 4 composite train-obj: 0.170029
            No improvement (0.2869), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3285, mae: 0.4095, huber: 0.1489, swd: 0.0702, ept: 151.8268
    Epoch [5/50], Val Losses: mse: 0.5314, mae: 0.5142, huber: 0.2231, swd: 0.0609, ept: 129.6410
    Epoch [5/50], Test Losses: mse: 0.7567, mae: 0.6576, huber: 0.3223, swd: 0.1038, ept: 74.9148
      Epoch 5 composite train-obj: 0.148893
            No improvement (0.2231), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2920, mae: 0.3879, huber: 0.1347, swd: 0.0536, ept: 165.5982
    Epoch [6/50], Val Losses: mse: 0.6650, mae: 0.5778, huber: 0.2695, swd: 0.0796, ept: 132.1031
    Epoch [6/50], Test Losses: mse: 0.8133, mae: 0.6808, huber: 0.3422, swd: 0.1096, ept: 65.3899
      Epoch 6 composite train-obj: 0.134740
    Epoch [6/50], Test Losses: mse: 0.7131, mae: 0.6410, huber: 0.3096, swd: 0.1841, ept: 83.0753
    Best round's Test MSE: 0.7131, MAE: 0.6410, SWD: 0.1841
    Best round's Validation MSE: 0.4392, MAE: 0.4609, SWD: 0.0744
    Best round's Test verification MSE : 0.7131, MAE: 0.6410, SWD: 0.1841
    Time taken: 73.72 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7832, mae: 0.6267, huber: 0.3091, swd: 0.1694, ept: 75.1950
    Epoch [1/50], Val Losses: mse: 0.5180, mae: 0.4943, huber: 0.2143, swd: 0.0665, ept: 132.2605
    Epoch [1/50], Test Losses: mse: 0.6799, mae: 0.6195, huber: 0.2943, swd: 0.1341, ept: 87.1785
      Epoch 1 composite train-obj: 0.309087
            Val objective improved inf → 0.2143, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5433, mae: 0.5185, huber: 0.2264, swd: 0.1615, ept: 114.8802
    Epoch [2/50], Val Losses: mse: 0.7016, mae: 0.5958, huber: 0.2841, swd: 0.1143, ept: 133.0972
    Epoch [2/50], Test Losses: mse: 0.7850, mae: 0.6856, huber: 0.3379, swd: 0.1663, ept: 77.2190
      Epoch 2 composite train-obj: 0.226448
            No improvement (0.2841), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4530, mae: 0.4736, huber: 0.1938, swd: 0.1387, ept: 131.1713
    Epoch [3/50], Val Losses: mse: 0.8422, mae: 0.6562, huber: 0.3281, swd: 0.1250, ept: 139.1647
    Epoch [3/50], Test Losses: mse: 0.8432, mae: 0.6999, huber: 0.3541, swd: 0.1302, ept: 77.7766
      Epoch 3 composite train-obj: 0.193789
            No improvement (0.3281), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.3732, mae: 0.4346, huber: 0.1656, swd: 0.0993, ept: 143.6066
    Epoch [4/50], Val Losses: mse: 0.7469, mae: 0.6099, huber: 0.2945, swd: 0.0927, ept: 138.7182
    Epoch [4/50], Test Losses: mse: 0.7796, mae: 0.6660, huber: 0.3305, swd: 0.1070, ept: 77.9968
      Epoch 4 composite train-obj: 0.165567
            No improvement (0.2945), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3198, mae: 0.4063, huber: 0.1458, swd: 0.0676, ept: 154.9063
    Epoch [5/50], Val Losses: mse: 0.8905, mae: 0.6743, huber: 0.3443, swd: 0.1099, ept: 135.6259
    Epoch [5/50], Test Losses: mse: 0.8455, mae: 0.7002, huber: 0.3548, swd: 0.1188, ept: 71.6630
      Epoch 5 composite train-obj: 0.145821
            No improvement (0.3443), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2802, mae: 0.3819, huber: 0.1301, swd: 0.0502, ept: 165.4233
    Epoch [6/50], Val Losses: mse: 0.7589, mae: 0.6218, huber: 0.3020, swd: 0.0931, ept: 139.2708
    Epoch [6/50], Test Losses: mse: 0.8705, mae: 0.7043, huber: 0.3606, swd: 0.1167, ept: 68.5116
      Epoch 6 composite train-obj: 0.130062
    Epoch [6/50], Test Losses: mse: 0.6799, mae: 0.6195, huber: 0.2943, swd: 0.1341, ept: 87.1785
    Best round's Test MSE: 0.6799, MAE: 0.6195, SWD: 0.1341
    Best round's Validation MSE: 0.5180, MAE: 0.4943, SWD: 0.0665
    Best round's Test verification MSE : 0.6799, MAE: 0.6195, SWD: 0.1341
    Time taken: 73.84 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7666, mae: 0.6198, huber: 0.3040, swd: 0.1967, ept: 92.4976
    Epoch [1/50], Val Losses: mse: 0.4560, mae: 0.4675, huber: 0.1942, swd: 0.0654, ept: 134.9200
    Epoch [1/50], Test Losses: mse: 0.6961, mae: 0.6324, huber: 0.3021, swd: 0.1683, ept: 81.4377
      Epoch 1 composite train-obj: 0.303961
            Val objective improved inf → 0.1942, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5834, mae: 0.5360, huber: 0.2397, swd: 0.1774, ept: 118.2701
    Epoch [2/50], Val Losses: mse: 0.5719, mae: 0.5213, huber: 0.2320, swd: 0.0752, ept: 136.7717
    Epoch [2/50], Test Losses: mse: 0.7911, mae: 0.6756, huber: 0.3353, swd: 0.1658, ept: 95.6830
      Epoch 2 composite train-obj: 0.239726
            No improvement (0.2320), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4910, mae: 0.4930, huber: 0.2074, swd: 0.1591, ept: 128.1619
    Epoch [3/50], Val Losses: mse: 0.5254, mae: 0.5015, huber: 0.2168, swd: 0.0601, ept: 134.0534
    Epoch [3/50], Test Losses: mse: 0.7553, mae: 0.6568, huber: 0.3217, swd: 0.1432, ept: 86.1638
      Epoch 3 composite train-obj: 0.207421
            No improvement (0.2168), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4225, mae: 0.4586, huber: 0.1832, swd: 0.1289, ept: 136.6677
    Epoch [4/50], Val Losses: mse: 0.5060, mae: 0.4962, huber: 0.2119, swd: 0.0484, ept: 125.1889
    Epoch [4/50], Test Losses: mse: 0.7226, mae: 0.6397, huber: 0.3097, swd: 0.1179, ept: 85.2620
      Epoch 4 composite train-obj: 0.183189
            No improvement (0.2119), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3505, mae: 0.4232, huber: 0.1575, swd: 0.0846, ept: 143.9089
    Epoch [5/50], Val Losses: mse: 0.5759, mae: 0.5328, huber: 0.2369, swd: 0.0617, ept: 125.6671
    Epoch [5/50], Test Losses: mse: 0.7794, mae: 0.6628, huber: 0.3282, swd: 0.1214, ept: 69.8438
      Epoch 5 composite train-obj: 0.157501
            No improvement (0.2369), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3017, mae: 0.3940, huber: 0.1384, swd: 0.0581, ept: 154.9671
    Epoch [6/50], Val Losses: mse: 0.6246, mae: 0.5621, huber: 0.2561, swd: 0.0729, ept: 103.4309
    Epoch [6/50], Test Losses: mse: 0.8294, mae: 0.6825, huber: 0.3447, swd: 0.1110, ept: 67.9822
      Epoch 6 composite train-obj: 0.138433
    Epoch [6/50], Test Losses: mse: 0.6961, mae: 0.6324, huber: 0.3021, swd: 0.1683, ept: 81.4377
    Best round's Test MSE: 0.6961, MAE: 0.6324, SWD: 0.1683
    Best round's Validation MSE: 0.4560, MAE: 0.4675, SWD: 0.0654
    Best round's Test verification MSE : 0.6961, MAE: 0.6324, SWD: 0.1683
    Time taken: 73.91 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq720_pred720_20250510_2002)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6964 ± 0.0135
      mae: 0.6310 ± 0.0088
      huber: 0.3020 ± 0.0062
      swd: 0.1622 ± 0.0209
      ept: 83.8972 ± 2.4147
      count: 3.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4711 ± 0.0339
      mae: 0.4742 ± 0.0144
      huber: 0.1992 ± 0.0109
      swd: 0.0688 ± 0.0040
      ept: 133.3455 ± 1.1395
      count: 3.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 221.66 seconds
    
    Experiment complete: TimeMixer_etth1_seq720_pred720_20250510_2002
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### PatchTST

#### pred=96


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 96
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
    
    Epoch [1/50], Train Losses: mse: 0.7142, mae: 0.5897, huber: 0.2806, swd: 0.1625, ept: 27.9048
    Epoch [1/50], Val Losses: mse: 0.3462, mae: 0.4105, huber: 0.1531, swd: 0.0798, ept: 47.6753
    Epoch [1/50], Test Losses: mse: 0.4763, mae: 0.4902, huber: 0.2061, swd: 0.1130, ept: 32.9031
      Epoch 1 composite train-obj: 0.280636
            Val objective improved inf → 0.1531, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4345, mae: 0.4648, huber: 0.1882, swd: 0.1023, ept: 40.0496
    Epoch [2/50], Val Losses: mse: 0.3147, mae: 0.3900, huber: 0.1397, swd: 0.0870, ept: 52.9196
    Epoch [2/50], Test Losses: mse: 0.4673, mae: 0.4891, huber: 0.2043, swd: 0.1605, ept: 32.5255
      Epoch 2 composite train-obj: 0.188232
            Val objective improved 0.1531 → 0.1397, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4051, mae: 0.4495, huber: 0.1774, swd: 0.0974, ept: 41.4533
    Epoch [3/50], Val Losses: mse: 0.3441, mae: 0.4107, huber: 0.1526, swd: 0.0999, ept: 48.8803
    Epoch [3/50], Test Losses: mse: 0.4705, mae: 0.4950, huber: 0.2070, swd: 0.1551, ept: 33.4265
      Epoch 3 composite train-obj: 0.177369
            No improvement (0.1526), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3773, mae: 0.4352, huber: 0.1670, swd: 0.0890, ept: 41.9789
    Epoch [4/50], Val Losses: mse: 0.3490, mae: 0.4120, huber: 0.1547, swd: 0.1001, ept: 49.7220
    Epoch [4/50], Test Losses: mse: 0.4547, mae: 0.4863, huber: 0.2010, swd: 0.1459, ept: 33.5131
      Epoch 4 composite train-obj: 0.166995
            No improvement (0.1547), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3574, mae: 0.4246, huber: 0.1593, swd: 0.0816, ept: 42.6230
    Epoch [5/50], Val Losses: mse: 0.3654, mae: 0.4288, huber: 0.1623, swd: 0.1091, ept: 50.3746
    Epoch [5/50], Test Losses: mse: 0.4713, mae: 0.5016, huber: 0.2091, swd: 0.1651, ept: 32.6779
      Epoch 5 composite train-obj: 0.159312
            No improvement (0.1623), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3378, mae: 0.4124, huber: 0.1513, swd: 0.0756, ept: 43.2316
    Epoch [6/50], Val Losses: mse: 0.3661, mae: 0.4218, huber: 0.1612, swd: 0.1042, ept: 49.0043
    Epoch [6/50], Test Losses: mse: 0.4725, mae: 0.5024, huber: 0.2098, swd: 0.1554, ept: 32.7572
      Epoch 6 composite train-obj: 0.151287
            No improvement (0.1612), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3209, mae: 0.4013, huber: 0.1441, swd: 0.0705, ept: 44.1519
    Epoch [7/50], Val Losses: mse: 0.3669, mae: 0.4274, huber: 0.1632, swd: 0.0971, ept: 48.0184
    Epoch [7/50], Test Losses: mse: 0.4532, mae: 0.4880, huber: 0.2011, swd: 0.1522, ept: 33.1790
      Epoch 7 composite train-obj: 0.144087
    Epoch [7/50], Test Losses: mse: 0.4673, mae: 0.4891, huber: 0.2043, swd: 0.1605, ept: 32.5255
    Best round's Test MSE: 0.4673, MAE: 0.4891, SWD: 0.1605
    Best round's Validation MSE: 0.3147, MAE: 0.3900, SWD: 0.0870
    Best round's Test verification MSE : 0.4673, MAE: 0.4891, SWD: 0.1605
    Time taken: 59.92 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7038, mae: 0.5825, huber: 0.2762, swd: 0.1631, ept: 29.1716
    Epoch [1/50], Val Losses: mse: 0.3431, mae: 0.4121, huber: 0.1526, swd: 0.0800, ept: 45.0767
    Epoch [1/50], Test Losses: mse: 0.4751, mae: 0.4941, huber: 0.2071, swd: 0.1220, ept: 31.9489
      Epoch 1 composite train-obj: 0.276179
            Val objective improved inf → 0.1526, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4340, mae: 0.4643, huber: 0.1879, swd: 0.1016, ept: 39.9491
    Epoch [2/50], Val Losses: mse: 0.3259, mae: 0.3979, huber: 0.1445, swd: 0.0851, ept: 49.9539
    Epoch [2/50], Test Losses: mse: 0.4622, mae: 0.4860, huber: 0.2024, swd: 0.1407, ept: 33.6952
      Epoch 2 composite train-obj: 0.187883
            Val objective improved 0.1526 → 0.1445, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3979, mae: 0.4450, huber: 0.1744, swd: 0.0934, ept: 41.4754
    Epoch [3/50], Val Losses: mse: 0.3403, mae: 0.4122, huber: 0.1511, swd: 0.0958, ept: 48.9029
    Epoch [3/50], Test Losses: mse: 0.4573, mae: 0.4850, huber: 0.2007, swd: 0.1453, ept: 33.6696
      Epoch 3 composite train-obj: 0.174413
            No improvement (0.1511), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3693, mae: 0.4306, huber: 0.1639, swd: 0.0851, ept: 42.0380
    Epoch [4/50], Val Losses: mse: 0.3701, mae: 0.4297, huber: 0.1642, swd: 0.1051, ept: 46.5756
    Epoch [4/50], Test Losses: mse: 0.4488, mae: 0.4835, huber: 0.1987, swd: 0.1355, ept: 34.1293
      Epoch 4 composite train-obj: 0.163923
            No improvement (0.1642), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3521, mae: 0.4215, huber: 0.1573, swd: 0.0787, ept: 42.6223
    Epoch [5/50], Val Losses: mse: 0.3616, mae: 0.4211, huber: 0.1599, swd: 0.0964, ept: 47.7302
    Epoch [5/50], Test Losses: mse: 0.4505, mae: 0.4821, huber: 0.1987, swd: 0.1328, ept: 33.9030
      Epoch 5 composite train-obj: 0.157285
            No improvement (0.1599), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3345, mae: 0.4102, huber: 0.1499, swd: 0.0728, ept: 43.3797
    Epoch [6/50], Val Losses: mse: 0.3629, mae: 0.4205, huber: 0.1595, swd: 0.1047, ept: 47.5510
    Epoch [6/50], Test Losses: mse: 0.4618, mae: 0.4906, huber: 0.2040, swd: 0.1499, ept: 33.3217
      Epoch 6 composite train-obj: 0.149941
            No improvement (0.1595), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3226, mae: 0.4028, huber: 0.1450, swd: 0.0687, ept: 43.9233
    Epoch [7/50], Val Losses: mse: 0.3552, mae: 0.4153, huber: 0.1564, swd: 0.0967, ept: 48.6055
    Epoch [7/50], Test Losses: mse: 0.4601, mae: 0.4903, huber: 0.2041, swd: 0.1474, ept: 32.9103
      Epoch 7 composite train-obj: 0.144962
    Epoch [7/50], Test Losses: mse: 0.4622, mae: 0.4860, huber: 0.2024, swd: 0.1407, ept: 33.6952
    Best round's Test MSE: 0.4622, MAE: 0.4860, SWD: 0.1407
    Best round's Validation MSE: 0.3259, MAE: 0.3979, SWD: 0.0851
    Best round's Test verification MSE : 0.4622, MAE: 0.4860, SWD: 0.1407
    Time taken: 59.74 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7125, mae: 0.5850, huber: 0.2781, swd: 0.1670, ept: 29.1198
    Epoch [1/50], Val Losses: mse: 0.3558, mae: 0.4136, huber: 0.1568, swd: 0.0809, ept: 46.3542
    Epoch [1/50], Test Losses: mse: 0.4709, mae: 0.4912, huber: 0.2052, swd: 0.1054, ept: 32.6875
      Epoch 1 composite train-obj: 0.278125
            Val objective improved inf → 0.1568, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4332, mae: 0.4640, huber: 0.1877, swd: 0.0971, ept: 40.0828
    Epoch [2/50], Val Losses: mse: 0.3351, mae: 0.4098, huber: 0.1489, swd: 0.0857, ept: 48.7964
    Epoch [2/50], Test Losses: mse: 0.4681, mae: 0.4896, huber: 0.2045, swd: 0.1383, ept: 32.5426
      Epoch 2 composite train-obj: 0.187708
            Val objective improved 0.1568 → 0.1489, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4046, mae: 0.4497, huber: 0.1773, swd: 0.0901, ept: 41.2702
    Epoch [3/50], Val Losses: mse: 0.3577, mae: 0.4178, huber: 0.1574, swd: 0.0781, ept: 47.4722
    Epoch [3/50], Test Losses: mse: 0.4649, mae: 0.4919, huber: 0.2043, swd: 0.1174, ept: 32.4414
      Epoch 3 composite train-obj: 0.177294
            No improvement (0.1574), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3739, mae: 0.4329, huber: 0.1656, swd: 0.0822, ept: 42.0476
    Epoch [4/50], Val Losses: mse: 0.3443, mae: 0.4135, huber: 0.1530, swd: 0.0905, ept: 47.5659
    Epoch [4/50], Test Losses: mse: 0.4640, mae: 0.4927, huber: 0.2049, swd: 0.1531, ept: 32.2682
      Epoch 4 composite train-obj: 0.165639
            No improvement (0.1530), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3537, mae: 0.4216, huber: 0.1577, swd: 0.0766, ept: 42.7075
    Epoch [5/50], Val Losses: mse: 0.3653, mae: 0.4230, huber: 0.1610, swd: 0.0872, ept: 47.0701
    Epoch [5/50], Test Losses: mse: 0.4728, mae: 0.4954, huber: 0.2078, swd: 0.1310, ept: 32.6528
      Epoch 5 composite train-obj: 0.157747
            No improvement (0.1610), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3361, mae: 0.4105, huber: 0.1505, swd: 0.0715, ept: 43.4463
    Epoch [6/50], Val Losses: mse: 0.3671, mae: 0.4219, huber: 0.1609, swd: 0.0820, ept: 47.6080
    Epoch [6/50], Test Losses: mse: 0.4630, mae: 0.4905, huber: 0.2044, swd: 0.1294, ept: 32.7963
      Epoch 6 composite train-obj: 0.150467
            No improvement (0.1609), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3231, mae: 0.4022, huber: 0.1450, swd: 0.0676, ept: 43.9940
    Epoch [7/50], Val Losses: mse: 0.3889, mae: 0.4399, huber: 0.1712, swd: 0.0855, ept: 44.2109
    Epoch [7/50], Test Losses: mse: 0.4735, mae: 0.5008, huber: 0.2096, swd: 0.1281, ept: 32.0703
      Epoch 7 composite train-obj: 0.145017
    Epoch [7/50], Test Losses: mse: 0.4681, mae: 0.4896, huber: 0.2045, swd: 0.1383, ept: 32.5426
    Best round's Test MSE: 0.4681, MAE: 0.4896, SWD: 0.1383
    Best round's Validation MSE: 0.3351, MAE: 0.4098, SWD: 0.0857
    Best round's Test verification MSE : 0.4681, MAE: 0.4896, SWD: 0.1383
    Time taken: 59.97 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq720_pred96_20250510_2006)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4659 ± 0.0026
      mae: 0.4882 ± 0.0016
      huber: 0.2037 ± 0.0009
      swd: 0.1465 ± 0.0099
      ept: 32.9211 ± 0.5474
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3253 ± 0.0083
      mae: 0.3992 ± 0.0082
      huber: 0.1444 ± 0.0038
      swd: 0.0859 ± 0.0008
      ept: 50.5567 ± 1.7364
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 179.70 seconds
    
    Experiment complete: PatchTST_etth1_seq720_pred96_20250510_2006
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.7655, mae: 0.6131, huber: 0.2989, swd: 0.1726, ept: 39.3961
    Epoch [1/50], Val Losses: mse: 0.4659, mae: 0.4844, huber: 0.2009, swd: 0.1092, ept: 64.4296
    Epoch [1/50], Test Losses: mse: 0.5800, mae: 0.5651, huber: 0.2530, swd: 0.1238, ept: 41.5769
      Epoch 1 composite train-obj: 0.298950
            Val objective improved inf → 0.2009, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5067, mae: 0.5021, huber: 0.2148, swd: 0.1214, ept: 56.5699
    Epoch [2/50], Val Losses: mse: 0.3873, mae: 0.4378, huber: 0.1693, swd: 0.0979, ept: 66.3802
    Epoch [2/50], Test Losses: mse: 0.5321, mae: 0.5335, huber: 0.2329, swd: 0.1423, ept: 42.6793
      Epoch 2 composite train-obj: 0.214793
            Val objective improved 0.2009 → 0.1693, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4625, mae: 0.4816, huber: 0.1993, swd: 0.1089, ept: 58.7440
    Epoch [3/50], Val Losses: mse: 0.4100, mae: 0.4542, huber: 0.1797, swd: 0.1008, ept: 63.7967
    Epoch [3/50], Test Losses: mse: 0.5110, mae: 0.5219, huber: 0.2249, swd: 0.1303, ept: 47.0366
      Epoch 3 composite train-obj: 0.199329
            No improvement (0.1797), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4234, mae: 0.4624, huber: 0.1851, swd: 0.0980, ept: 60.5494
    Epoch [4/50], Val Losses: mse: 0.3894, mae: 0.4408, huber: 0.1703, swd: 0.0980, ept: 66.8138
    Epoch [4/50], Test Losses: mse: 0.5312, mae: 0.5356, huber: 0.2335, swd: 0.1731, ept: 45.5121
      Epoch 4 composite train-obj: 0.185073
            No improvement (0.1703), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3993, mae: 0.4489, huber: 0.1756, swd: 0.0898, ept: 61.6570
    Epoch [5/50], Val Losses: mse: 0.4075, mae: 0.4457, huber: 0.1769, swd: 0.0950, ept: 66.0132
    Epoch [5/50], Test Losses: mse: 0.5213, mae: 0.5322, huber: 0.2302, swd: 0.1403, ept: 46.0096
      Epoch 5 composite train-obj: 0.175585
            No improvement (0.1769), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3782, mae: 0.4363, huber: 0.1670, swd: 0.0836, ept: 63.4540
    Epoch [6/50], Val Losses: mse: 0.4462, mae: 0.4675, huber: 0.1918, swd: 0.0959, ept: 61.7711
    Epoch [6/50], Test Losses: mse: 0.5285, mae: 0.5340, huber: 0.2319, swd: 0.1201, ept: 45.5034
      Epoch 6 composite train-obj: 0.166953
            No improvement (0.1918), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3626, mae: 0.4268, huber: 0.1605, swd: 0.0785, ept: 64.3728
    Epoch [7/50], Val Losses: mse: 0.4193, mae: 0.4571, huber: 0.1825, swd: 0.0960, ept: 64.7629
    Epoch [7/50], Test Losses: mse: 0.5438, mae: 0.5465, huber: 0.2403, swd: 0.1632, ept: 46.2998
      Epoch 7 composite train-obj: 0.160533
    Epoch [7/50], Test Losses: mse: 0.5321, mae: 0.5335, huber: 0.2329, swd: 0.1423, ept: 42.6793
    Best round's Test MSE: 0.5321, MAE: 0.5335, SWD: 0.1423
    Best round's Validation MSE: 0.3873, MAE: 0.4378, SWD: 0.0979
    Best round's Test verification MSE : 0.5321, MAE: 0.5335, SWD: 0.1423
    Time taken: 59.65 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7535, mae: 0.6031, huber: 0.2927, swd: 0.1800, ept: 42.2204
    Epoch [1/50], Val Losses: mse: 0.3819, mae: 0.4348, huber: 0.1677, swd: 0.0953, ept: 62.6371
    Epoch [1/50], Test Losses: mse: 0.5631, mae: 0.5497, huber: 0.2452, swd: 0.1273, ept: 41.9258
      Epoch 1 composite train-obj: 0.292700
            Val objective improved inf → 0.1677, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5009, mae: 0.4992, huber: 0.2126, swd: 0.1238, ept: 56.4655
    Epoch [2/50], Val Losses: mse: 0.4207, mae: 0.4641, huber: 0.1850, swd: 0.1197, ept: 62.8927
    Epoch [2/50], Test Losses: mse: 0.5047, mae: 0.5202, huber: 0.2228, swd: 0.1532, ept: 42.9490
      Epoch 2 composite train-obj: 0.212648
            No improvement (0.1850), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4599, mae: 0.4810, huber: 0.1986, swd: 0.1108, ept: 57.8014
    Epoch [3/50], Val Losses: mse: 0.4209, mae: 0.4634, huber: 0.1838, swd: 0.1044, ept: 60.2727
    Epoch [3/50], Test Losses: mse: 0.5206, mae: 0.5273, huber: 0.2283, swd: 0.1355, ept: 42.7763
      Epoch 3 composite train-obj: 0.198565
            No improvement (0.1838), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4209, mae: 0.4606, huber: 0.1839, swd: 0.0990, ept: 59.6523
    Epoch [4/50], Val Losses: mse: 0.4550, mae: 0.4802, huber: 0.1978, swd: 0.1118, ept: 60.6497
    Epoch [4/50], Test Losses: mse: 0.5138, mae: 0.5278, huber: 0.2274, swd: 0.1325, ept: 44.2765
      Epoch 4 composite train-obj: 0.183887
            No improvement (0.1978), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3967, mae: 0.4479, huber: 0.1746, swd: 0.0896, ept: 60.8970
    Epoch [5/50], Val Losses: mse: 0.4631, mae: 0.4867, huber: 0.2021, swd: 0.1176, ept: 60.6267
    Epoch [5/50], Test Losses: mse: 0.5031, mae: 0.5197, huber: 0.2228, swd: 0.1355, ept: 45.4165
      Epoch 5 composite train-obj: 0.174580
            No improvement (0.2021), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3769, mae: 0.4359, huber: 0.1666, swd: 0.0834, ept: 62.2275
    Epoch [6/50], Val Losses: mse: 0.4502, mae: 0.4779, huber: 0.1958, swd: 0.1181, ept: 62.2373
    Epoch [6/50], Test Losses: mse: 0.5161, mae: 0.5283, huber: 0.2287, swd: 0.1594, ept: 45.7615
      Epoch 6 composite train-obj: 0.166602
    Epoch [6/50], Test Losses: mse: 0.5631, mae: 0.5497, huber: 0.2452, swd: 0.1273, ept: 41.9258
    Best round's Test MSE: 0.5631, MAE: 0.5497, SWD: 0.1273
    Best round's Validation MSE: 0.3819, MAE: 0.4348, SWD: 0.0953
    Best round's Test verification MSE : 0.5631, MAE: 0.5497, SWD: 0.1273
    Time taken: 51.20 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7841, mae: 0.6180, huber: 0.3032, swd: 0.1671, ept: 38.6309
    Epoch [1/50], Val Losses: mse: 0.4496, mae: 0.4744, huber: 0.1937, swd: 0.0935, ept: 60.8607
    Epoch [1/50], Test Losses: mse: 0.5553, mae: 0.5450, huber: 0.2411, swd: 0.0989, ept: 42.5324
      Epoch 1 composite train-obj: 0.303203
            Val objective improved inf → 0.1937, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5125, mae: 0.5044, huber: 0.2166, swd: 0.1133, ept: 56.0709
    Epoch [2/50], Val Losses: mse: 0.4013, mae: 0.4535, huber: 0.1764, swd: 0.0895, ept: 61.3314
    Epoch [2/50], Test Losses: mse: 0.5196, mae: 0.5235, huber: 0.2264, swd: 0.1094, ept: 46.3346
      Epoch 2 composite train-obj: 0.216571
            Val objective improved 0.1937 → 0.1764, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4629, mae: 0.4812, huber: 0.1991, swd: 0.1024, ept: 58.2982
    Epoch [3/50], Val Losses: mse: 0.4327, mae: 0.4773, huber: 0.1920, swd: 0.1006, ept: 61.0779
    Epoch [3/50], Test Losses: mse: 0.5009, mae: 0.5217, huber: 0.2221, swd: 0.1297, ept: 46.5863
      Epoch 3 composite train-obj: 0.199087
            No improvement (0.1920), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4218, mae: 0.4606, huber: 0.1841, swd: 0.0900, ept: 60.4024
    Epoch [4/50], Val Losses: mse: 0.4128, mae: 0.4596, huber: 0.1813, swd: 0.0973, ept: 62.5719
    Epoch [4/50], Test Losses: mse: 0.5067, mae: 0.5208, huber: 0.2232, swd: 0.1325, ept: 45.8708
      Epoch 4 composite train-obj: 0.184136
            No improvement (0.1813), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3999, mae: 0.4492, huber: 0.1758, swd: 0.0822, ept: 60.9642
    Epoch [5/50], Val Losses: mse: 0.4363, mae: 0.4680, huber: 0.1892, swd: 0.0978, ept: 61.4150
    Epoch [5/50], Test Losses: mse: 0.5729, mae: 0.5586, huber: 0.2503, swd: 0.1404, ept: 44.2664
      Epoch 5 composite train-obj: 0.175820
            No improvement (0.1892), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3767, mae: 0.4354, huber: 0.1664, swd: 0.0758, ept: 62.9367
    Epoch [6/50], Val Losses: mse: 0.4499, mae: 0.4791, huber: 0.1953, swd: 0.0985, ept: 61.5887
    Epoch [6/50], Test Losses: mse: 0.5239, mae: 0.5323, huber: 0.2308, swd: 0.1268, ept: 46.3714
      Epoch 6 composite train-obj: 0.166430
            No improvement (0.1953), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3595, mae: 0.4248, huber: 0.1594, swd: 0.0714, ept: 64.2328
    Epoch [7/50], Val Losses: mse: 0.4792, mae: 0.4953, huber: 0.2070, swd: 0.1014, ept: 60.4525
    Epoch [7/50], Test Losses: mse: 0.5495, mae: 0.5499, huber: 0.2418, swd: 0.1279, ept: 45.5181
      Epoch 7 composite train-obj: 0.159417
    Epoch [7/50], Test Losses: mse: 0.5196, mae: 0.5235, huber: 0.2264, swd: 0.1094, ept: 46.3346
    Best round's Test MSE: 0.5196, MAE: 0.5235, SWD: 0.1094
    Best round's Validation MSE: 0.4013, MAE: 0.4535, SWD: 0.0895
    Best round's Test verification MSE : 0.5196, MAE: 0.5235, SWD: 0.1094
    Time taken: 59.89 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq720_pred196_20250510_2009)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5383 ± 0.0183
      mae: 0.5356 ± 0.0108
      huber: 0.2348 ± 0.0078
      swd: 0.1263 ± 0.0134
      ept: 43.6465 ± 1.9255
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3902 ± 0.0082
      mae: 0.4420 ± 0.0082
      huber: 0.1711 ± 0.0038
      swd: 0.0942 ± 0.0035
      ept: 63.4496 ± 2.1397
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 170.85 seconds
    
    Experiment complete: PatchTST_etth1_seq720_pred196_20250510_2009
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 88
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 88
    Validation Batches: 6
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.8673, mae: 0.6545, huber: 0.3313, swd: 0.1884, ept: 43.8149
    Epoch [1/50], Val Losses: mse: 0.4598, mae: 0.4695, huber: 0.1939, swd: 0.0741, ept: 78.8344
    Epoch [1/50], Test Losses: mse: 0.6022, mae: 0.5694, huber: 0.2595, swd: 0.0881, ept: 46.0732
      Epoch 1 composite train-obj: 0.331283
            Val objective improved inf → 0.1939, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6068, mae: 0.5506, huber: 0.2504, swd: 0.1374, ept: 64.9834
    Epoch [2/50], Val Losses: mse: 0.3868, mae: 0.4410, huber: 0.1692, swd: 0.0733, ept: 81.7635
    Epoch [2/50], Test Losses: mse: 0.6288, mae: 0.5817, huber: 0.2692, swd: 0.1595, ept: 48.5315
      Epoch 2 composite train-obj: 0.250391
            Val objective improved 0.1939 → 0.1692, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5873, mae: 0.5426, huber: 0.2438, swd: 0.1321, ept: 67.8980
    Epoch [3/50], Val Losses: mse: 0.3989, mae: 0.4491, huber: 0.1750, swd: 0.0800, ept: 85.4572
    Epoch [3/50], Test Losses: mse: 0.5738, mae: 0.5575, huber: 0.2505, swd: 0.1425, ept: 55.2108
      Epoch 3 composite train-obj: 0.243765
            No improvement (0.1750), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5667, mae: 0.5362, huber: 0.2376, swd: 0.1271, ept: 65.3239
    Epoch [4/50], Val Losses: mse: 0.4708, mae: 0.5004, huber: 0.2080, swd: 0.1160, ept: 76.7554
    Epoch [4/50], Test Losses: mse: 0.5714, mae: 0.5618, huber: 0.2504, swd: 0.1384, ept: 54.6207
      Epoch 4 composite train-obj: 0.237606
            No improvement (0.2080), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5037, mae: 0.5040, huber: 0.2144, swd: 0.1181, ept: 71.3839
    Epoch [5/50], Val Losses: mse: 0.5300, mae: 0.5461, huber: 0.2355, swd: 0.1015, ept: 66.9838
    Epoch [5/50], Test Losses: mse: 0.6107, mae: 0.5824, huber: 0.2650, swd: 0.0854, ept: 50.8723
      Epoch 5 composite train-obj: 0.214372
            No improvement (0.2355), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4609, mae: 0.4843, huber: 0.1993, swd: 0.1038, ept: 74.1994
    Epoch [6/50], Val Losses: mse: 0.4754, mae: 0.4989, huber: 0.2071, swd: 0.1019, ept: 69.2746
    Epoch [6/50], Test Losses: mse: 0.6384, mae: 0.6021, huber: 0.2783, swd: 0.1451, ept: 50.2467
      Epoch 6 composite train-obj: 0.199318
            No improvement (0.2071), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4347, mae: 0.4705, huber: 0.1894, swd: 0.0949, ept: 75.9027
    Epoch [7/50], Val Losses: mse: 0.5685, mae: 0.5582, huber: 0.2488, swd: 0.1360, ept: 74.6162
    Epoch [7/50], Test Losses: mse: 0.7298, mae: 0.6281, huber: 0.3054, swd: 0.1248, ept: 52.9496
      Epoch 7 composite train-obj: 0.189381
    Epoch [7/50], Test Losses: mse: 0.6288, mae: 0.5817, huber: 0.2692, swd: 0.1595, ept: 48.5315
    Best round's Test MSE: 0.6288, MAE: 0.5817, SWD: 0.1595
    Best round's Validation MSE: 0.3868, MAE: 0.4410, SWD: 0.0733
    Best round's Test verification MSE : 0.6288, MAE: 0.5817, SWD: 0.1595
    Time taken: 59.02 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.8329, mae: 0.6373, huber: 0.3193, swd: 0.1917, ept: 49.7071
    Epoch [1/50], Val Losses: mse: 0.5245, mae: 0.5185, huber: 0.2225, swd: 0.0693, ept: 60.1047
    Epoch [1/50], Test Losses: mse: 0.7363, mae: 0.6438, huber: 0.3126, swd: 0.0934, ept: 39.2058
      Epoch 1 composite train-obj: 0.319255
            Val objective improved inf → 0.2225, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6076, mae: 0.5489, huber: 0.2497, swd: 0.1431, ept: 64.6161
    Epoch [2/50], Val Losses: mse: 0.4527, mae: 0.4772, huber: 0.1950, swd: 0.0870, ept: 76.4984
    Epoch [2/50], Test Losses: mse: 0.6159, mae: 0.5867, huber: 0.2692, swd: 0.1210, ept: 51.9763
      Epoch 2 composite train-obj: 0.249717
            Val objective improved 0.2225 → 0.1950, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5283, mae: 0.5140, huber: 0.2227, swd: 0.1258, ept: 70.3993
    Epoch [3/50], Val Losses: mse: 0.4596, mae: 0.4860, huber: 0.1995, swd: 0.0679, ept: 69.0326
    Epoch [3/50], Test Losses: mse: 0.6918, mae: 0.6130, huber: 0.2922, swd: 0.1150, ept: 42.7889
      Epoch 3 composite train-obj: 0.222659
            No improvement (0.1995), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4990, mae: 0.5016, huber: 0.2126, swd: 0.1171, ept: 70.4891
    Epoch [4/50], Val Losses: mse: 0.4527, mae: 0.4807, huber: 0.1960, swd: 0.0878, ept: 77.5963
    Epoch [4/50], Test Losses: mse: 0.6048, mae: 0.5793, huber: 0.2635, swd: 0.1205, ept: 53.4545
      Epoch 4 composite train-obj: 0.212588
            No improvement (0.1960), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4528, mae: 0.4770, huber: 0.1950, swd: 0.1055, ept: 75.2160
    Epoch [5/50], Val Losses: mse: 0.4997, mae: 0.5272, huber: 0.2214, swd: 0.0842, ept: 70.7605
    Epoch [5/50], Test Losses: mse: 0.6593, mae: 0.6078, huber: 0.2851, swd: 0.1111, ept: 50.8702
      Epoch 5 composite train-obj: 0.195036
            No improvement (0.2214), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4367, mae: 0.4711, huber: 0.1899, swd: 0.0970, ept: 75.1656
    Epoch [6/50], Val Losses: mse: 0.4888, mae: 0.5076, huber: 0.2125, swd: 0.0651, ept: 72.1886
    Epoch [6/50], Test Losses: mse: 0.6529, mae: 0.6049, huber: 0.2830, swd: 0.1036, ept: 51.6084
      Epoch 6 composite train-obj: 0.189885
            No improvement (0.2125), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4145, mae: 0.4600, huber: 0.1815, swd: 0.0857, ept: 76.7085
    Epoch [7/50], Val Losses: mse: 0.4619, mae: 0.4923, huber: 0.2026, swd: 0.0877, ept: 74.7063
    Epoch [7/50], Test Losses: mse: 0.6429, mae: 0.6031, huber: 0.2803, swd: 0.1479, ept: 53.4135
      Epoch 7 composite train-obj: 0.181511
    Epoch [7/50], Test Losses: mse: 0.6159, mae: 0.5867, huber: 0.2692, swd: 0.1210, ept: 51.9763
    Best round's Test MSE: 0.6159, MAE: 0.5867, SWD: 0.1210
    Best round's Validation MSE: 0.4527, MAE: 0.4772, SWD: 0.0870
    Best round's Test verification MSE : 0.6159, MAE: 0.5867, SWD: 0.1210
    Time taken: 58.99 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.8561, mae: 0.6472, huber: 0.3264, swd: 0.1969, ept: 47.8375
    Epoch [1/50], Val Losses: mse: 0.4819, mae: 0.4874, huber: 0.2048, swd: 0.0783, ept: 76.5801
    Epoch [1/50], Test Losses: mse: 0.7489, mae: 0.6477, huber: 0.3165, swd: 0.0940, ept: 43.3251
      Epoch 1 composite train-obj: 0.326420
            Val objective improved inf → 0.2048, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5985, mae: 0.5473, huber: 0.2476, swd: 0.1351, ept: 63.2381
    Epoch [2/50], Val Losses: mse: 0.4936, mae: 0.5189, huber: 0.2166, swd: 0.0655, ept: 49.9498
    Epoch [2/50], Test Losses: mse: 0.6767, mae: 0.6132, huber: 0.2892, swd: 0.0965, ept: 40.7847
      Epoch 2 composite train-obj: 0.247562
            No improvement (0.2166), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5507, mae: 0.5256, huber: 0.2308, swd: 0.1278, ept: 68.5792
    Epoch [3/50], Val Losses: mse: 0.4585, mae: 0.4835, huber: 0.1986, swd: 0.0731, ept: 69.3636
    Epoch [3/50], Test Losses: mse: 0.6595, mae: 0.6030, huber: 0.2829, swd: 0.0994, ept: 47.6328
      Epoch 3 composite train-obj: 0.230772
            Val objective improved 0.2048 → 0.1986, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5081, mae: 0.5069, huber: 0.2161, swd: 0.1182, ept: 69.7813
    Epoch [4/50], Val Losses: mse: 0.4741, mae: 0.4925, huber: 0.2042, swd: 0.0680, ept: 71.2704
    Epoch [4/50], Test Losses: mse: 0.6480, mae: 0.5960, huber: 0.2773, swd: 0.1175, ept: 49.9068
      Epoch 4 composite train-obj: 0.216096
            No improvement (0.2042), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4635, mae: 0.4838, huber: 0.1995, swd: 0.1052, ept: 72.4674
    Epoch [5/50], Val Losses: mse: 0.4305, mae: 0.4737, huber: 0.1891, swd: 0.0708, ept: 74.7817
    Epoch [5/50], Test Losses: mse: 0.6709, mae: 0.6077, huber: 0.2874, swd: 0.1457, ept: 48.9587
      Epoch 5 composite train-obj: 0.199452
            Val objective improved 0.1986 → 0.1891, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4475, mae: 0.4757, huber: 0.1937, swd: 0.1006, ept: 74.7922
    Epoch [6/50], Val Losses: mse: 0.4872, mae: 0.5061, huber: 0.2118, swd: 0.1001, ept: 61.2484
    Epoch [6/50], Test Losses: mse: 0.7912, mae: 0.6608, huber: 0.3280, swd: 0.1522, ept: 45.3767
      Epoch 6 composite train-obj: 0.193656
            No improvement (0.2118), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4558, mae: 0.4817, huber: 0.1971, swd: 0.0986, ept: 72.0915
    Epoch [7/50], Val Losses: mse: 0.4342, mae: 0.4729, huber: 0.1897, swd: 0.0983, ept: 78.5976
    Epoch [7/50], Test Losses: mse: 0.6837, mae: 0.6098, huber: 0.2896, swd: 0.1658, ept: 52.3651
      Epoch 7 composite train-obj: 0.197142
            No improvement (0.1897), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4059, mae: 0.4533, huber: 0.1777, swd: 0.0852, ept: 78.1339
    Epoch [8/50], Val Losses: mse: 0.5501, mae: 0.5314, huber: 0.2327, swd: 0.0938, ept: 77.3095
    Epoch [8/50], Test Losses: mse: 0.8330, mae: 0.6727, huber: 0.3390, swd: 0.1546, ept: 48.6449
      Epoch 8 composite train-obj: 0.177718
            No improvement (0.2327), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3998, mae: 0.4497, huber: 0.1752, swd: 0.0809, ept: 78.9539
    Epoch [9/50], Val Losses: mse: 0.4545, mae: 0.4830, huber: 0.1976, swd: 0.0945, ept: 78.1457
    Epoch [9/50], Test Losses: mse: 0.7337, mae: 0.6335, huber: 0.3078, swd: 0.1581, ept: 53.5466
      Epoch 9 composite train-obj: 0.175150
            No improvement (0.1976), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.3695, mae: 0.4319, huber: 0.1632, swd: 0.0718, ept: 82.7343
    Epoch [10/50], Val Losses: mse: 0.4479, mae: 0.4821, huber: 0.1956, swd: 0.0857, ept: 78.5363
    Epoch [10/50], Test Losses: mse: 0.6661, mae: 0.6087, huber: 0.2868, swd: 0.1474, ept: 55.0441
      Epoch 10 composite train-obj: 0.163231
    Epoch [10/50], Test Losses: mse: 0.6709, mae: 0.6077, huber: 0.2874, swd: 0.1457, ept: 48.9587
    Best round's Test MSE: 0.6709, MAE: 0.6077, SWD: 0.1457
    Best round's Validation MSE: 0.4305, MAE: 0.4737, SWD: 0.0708
    Best round's Test verification MSE : 0.6709, MAE: 0.6077, SWD: 0.1457
    Time taken: 83.98 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq720_pred336_20250510_2012)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6385 ± 0.0235
      mae: 0.5920 ± 0.0113
      huber: 0.2753 ± 0.0086
      swd: 0.1421 ± 0.0159
      ept: 49.8222 ± 1.5332
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4233 ± 0.0274
      mae: 0.4640 ± 0.0163
      huber: 0.1844 ± 0.0110
      swd: 0.0770 ± 0.0071
      ept: 77.6812 ± 2.9705
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 202.13 seconds
    
    Experiment complete: PatchTST_etth1_seq720_pred336_20250510_2012
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 85
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 85
    Validation Batches: 3
    Test Batches: 16
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 1.0000, mae: 0.7145, huber: 0.3780, swd: 0.2185, ept: 48.8386
    Epoch [1/50], Val Losses: mse: 0.7468, mae: 0.5996, huber: 0.2890, swd: 0.0905, ept: 56.3747
    Epoch [1/50], Test Losses: mse: 1.2133, mae: 0.8222, huber: 0.4595, swd: 0.2074, ept: 23.2640
      Epoch 1 composite train-obj: 0.377984
            Val objective improved inf → 0.2890, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.7647, mae: 0.6282, huber: 0.3078, swd: 0.1751, ept: 73.0148
    Epoch [2/50], Val Losses: mse: 0.5206, mae: 0.5080, huber: 0.2185, swd: 0.0902, ept: 135.1252
    Epoch [2/50], Test Losses: mse: 0.9200, mae: 0.7348, huber: 0.3816, swd: 0.1788, ept: 67.8868
      Epoch 2 composite train-obj: 0.307762
            Val objective improved 0.2890 → 0.2185, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6736, mae: 0.5896, huber: 0.2770, swd: 0.1604, ept: 81.2115
    Epoch [3/50], Val Losses: mse: 0.4341, mae: 0.4662, huber: 0.1885, swd: 0.0724, ept: 113.9047
    Epoch [3/50], Test Losses: mse: 0.7518, mae: 0.6543, huber: 0.3210, swd: 0.1567, ept: 76.9152
      Epoch 3 composite train-obj: 0.277031
            Val objective improved 0.2185 → 0.1885, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.6208, mae: 0.5667, huber: 0.2587, swd: 0.1483, ept: 82.1883
    Epoch [4/50], Val Losses: mse: 0.4999, mae: 0.4938, huber: 0.2103, swd: 0.0596, ept: 100.8624
    Epoch [4/50], Test Losses: mse: 0.7722, mae: 0.6615, huber: 0.3251, swd: 0.1263, ept: 80.4593
      Epoch 4 composite train-obj: 0.258718
            No improvement (0.2103), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5702, mae: 0.5434, huber: 0.2405, swd: 0.1384, ept: 85.5737
    Epoch [5/50], Val Losses: mse: 0.5438, mae: 0.5303, huber: 0.2297, swd: 0.0895, ept: 67.7358
    Epoch [5/50], Test Losses: mse: 0.9288, mae: 0.7238, huber: 0.3790, swd: 0.1293, ept: 74.7490
      Epoch 5 composite train-obj: 0.240517
            No improvement (0.2297), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5228, mae: 0.5215, huber: 0.2235, swd: 0.1194, ept: 86.7350
    Epoch [6/50], Val Losses: mse: 0.4372, mae: 0.4830, huber: 0.1953, swd: 0.0549, ept: 71.2209
    Epoch [6/50], Test Losses: mse: 0.7110, mae: 0.6368, huber: 0.3068, swd: 0.1246, ept: 66.9895
      Epoch 6 composite train-obj: 0.223530
            No improvement (0.1953), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.4891, mae: 0.5044, huber: 0.2109, swd: 0.1059, ept: 88.6469
    Epoch [7/50], Val Losses: mse: 0.4801, mae: 0.4921, huber: 0.2069, swd: 0.0423, ept: 98.2524
    Epoch [7/50], Test Losses: mse: 0.7296, mae: 0.6444, huber: 0.3134, swd: 0.1143, ept: 78.5385
      Epoch 7 composite train-obj: 0.210920
            No improvement (0.2069), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4546, mae: 0.4852, huber: 0.1975, swd: 0.0906, ept: 92.2600
    Epoch [8/50], Val Losses: mse: 0.5859, mae: 0.5370, huber: 0.2414, swd: 0.0886, ept: 88.9401
    Epoch [8/50], Test Losses: mse: 0.9781, mae: 0.7382, huber: 0.3922, swd: 0.1284, ept: 72.4121
      Epoch 8 composite train-obj: 0.197458
    Epoch [8/50], Test Losses: mse: 0.7518, mae: 0.6543, huber: 0.3210, swd: 0.1567, ept: 76.9152
    Best round's Test MSE: 0.7518, MAE: 0.6543, SWD: 0.1567
    Best round's Validation MSE: 0.4341, MAE: 0.4662, SWD: 0.0724
    Best round's Test verification MSE : 0.7518, MAE: 0.6543, SWD: 0.1567
    Time taken: 66.84 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.9977, mae: 0.7095, huber: 0.3751, swd: 0.2267, ept: 54.4003
    Epoch [1/50], Val Losses: mse: 0.4707, mae: 0.4974, huber: 0.2058, swd: 0.0377, ept: 83.7376
    Epoch [1/50], Test Losses: mse: 0.9283, mae: 0.7246, huber: 0.3795, swd: 0.1030, ept: 59.3356
      Epoch 1 composite train-obj: 0.375088
            Val objective improved inf → 0.2058, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.7180, mae: 0.6084, huber: 0.2923, swd: 0.1627, ept: 73.3601
    Epoch [2/50], Val Losses: mse: 0.4933, mae: 0.5023, huber: 0.2124, swd: 0.0611, ept: 84.6051
    Epoch [2/50], Test Losses: mse: 0.8328, mae: 0.6964, huber: 0.3508, swd: 0.1119, ept: 46.1448
      Epoch 2 composite train-obj: 0.292289
            No improvement (0.2124), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.6391, mae: 0.5756, huber: 0.2656, swd: 0.1496, ept: 74.8703
    Epoch [3/50], Val Losses: mse: 0.4426, mae: 0.4819, huber: 0.1953, swd: 0.0442, ept: 78.3103
    Epoch [3/50], Test Losses: mse: 0.7871, mae: 0.6695, huber: 0.3313, swd: 0.1244, ept: 52.3346
      Epoch 3 composite train-obj: 0.265639
            Val objective improved 0.2058 → 0.1953, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5887, mae: 0.5541, huber: 0.2482, swd: 0.1375, ept: 77.4094
    Epoch [4/50], Val Losses: mse: 0.4517, mae: 0.4914, huber: 0.1999, swd: 0.0665, ept: 97.7478
    Epoch [4/50], Test Losses: mse: 0.8749, mae: 0.7150, huber: 0.3646, swd: 0.1549, ept: 51.7419
      Epoch 4 composite train-obj: 0.248162
            No improvement (0.1999), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5454, mae: 0.5335, huber: 0.2325, swd: 0.1266, ept: 83.1909
    Epoch [5/50], Val Losses: mse: 0.4905, mae: 0.5049, huber: 0.2131, swd: 0.0648, ept: 64.5505
    Epoch [5/50], Test Losses: mse: 0.7043, mae: 0.6278, huber: 0.3001, swd: 0.1148, ept: 66.2711
      Epoch 5 composite train-obj: 0.232460
            No improvement (0.2131), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4921, mae: 0.5067, huber: 0.2126, swd: 0.1074, ept: 89.3181
    Epoch [6/50], Val Losses: mse: 0.5189, mae: 0.5202, huber: 0.2244, swd: 0.0305, ept: 76.8211
    Epoch [6/50], Test Losses: mse: 0.7414, mae: 0.6464, huber: 0.3131, swd: 0.0835, ept: 52.5647
      Epoch 6 composite train-obj: 0.212578
            No improvement (0.2244), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.4828, mae: 0.5029, huber: 0.2094, swd: 0.0965, ept: 88.2723
    Epoch [7/50], Val Losses: mse: 0.5993, mae: 0.5679, huber: 0.2560, swd: 0.1145, ept: 87.3546
    Epoch [7/50], Test Losses: mse: 0.7375, mae: 0.6499, huber: 0.3148, swd: 0.1314, ept: 76.0556
      Epoch 7 composite train-obj: 0.209420
            No improvement (0.2560), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4486, mae: 0.4836, huber: 0.1960, swd: 0.0854, ept: 95.0973
    Epoch [8/50], Val Losses: mse: 0.5180, mae: 0.5150, huber: 0.2226, swd: 0.0802, ept: 109.9104
    Epoch [8/50], Test Losses: mse: 0.7477, mae: 0.6463, huber: 0.3143, swd: 0.1249, ept: 69.8332
      Epoch 8 composite train-obj: 0.195976
    Epoch [8/50], Test Losses: mse: 0.7871, mae: 0.6695, huber: 0.3313, swd: 0.1244, ept: 52.3346
    Best round's Test MSE: 0.7871, MAE: 0.6695, SWD: 0.1244
    Best round's Validation MSE: 0.4426, MAE: 0.4819, SWD: 0.0442
    Best round's Test verification MSE : 0.7871, MAE: 0.6695, SWD: 0.1244
    Time taken: 68.11 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.9874, mae: 0.7071, huber: 0.3725, swd: 0.2274, ept: 52.7051
    Epoch [1/50], Val Losses: mse: 0.4944, mae: 0.5221, huber: 0.2183, swd: 0.0367, ept: 101.2721
    Epoch [1/50], Test Losses: mse: 0.8982, mae: 0.7108, huber: 0.3685, swd: 0.1053, ept: 53.6716
      Epoch 1 composite train-obj: 0.372524
            Val objective improved inf → 0.2183, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.7032, mae: 0.6018, huber: 0.2869, swd: 0.1695, ept: 74.6918
    Epoch [2/50], Val Losses: mse: 0.4525, mae: 0.4750, huber: 0.1955, swd: 0.0702, ept: 75.5600
    Epoch [2/50], Test Losses: mse: 0.7865, mae: 0.6671, huber: 0.3304, swd: 0.1317, ept: 56.0274
      Epoch 2 composite train-obj: 0.286949
            Val objective improved 0.2183 → 0.1955, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6315, mae: 0.5710, huber: 0.2622, swd: 0.1609, ept: 76.7495
    Epoch [3/50], Val Losses: mse: 0.4209, mae: 0.4688, huber: 0.1871, swd: 0.0454, ept: 97.4803
    Epoch [3/50], Test Losses: mse: 0.7140, mae: 0.6436, huber: 0.3083, swd: 0.1475, ept: 54.6987
      Epoch 3 composite train-obj: 0.262183
            Val objective improved 0.1955 → 0.1871, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5795, mae: 0.5473, huber: 0.2438, swd: 0.1502, ept: 81.4178
    Epoch [4/50], Val Losses: mse: 0.4718, mae: 0.4938, huber: 0.2049, swd: 0.0642, ept: 80.9813
    Epoch [4/50], Test Losses: mse: 0.7635, mae: 0.6673, huber: 0.3278, swd: 0.1404, ept: 55.6320
      Epoch 4 composite train-obj: 0.243818
            No improvement (0.2049), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5494, mae: 0.5341, huber: 0.2332, swd: 0.1360, ept: 82.6667
    Epoch [5/50], Val Losses: mse: 0.4532, mae: 0.4876, huber: 0.1998, swd: 0.0566, ept: 82.5685
    Epoch [5/50], Test Losses: mse: 0.7361, mae: 0.6450, huber: 0.3141, swd: 0.1337, ept: 61.7144
      Epoch 5 composite train-obj: 0.233209
            No improvement (0.1998), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5063, mae: 0.5121, huber: 0.2170, swd: 0.1195, ept: 87.4431
    Epoch [6/50], Val Losses: mse: 0.4945, mae: 0.5062, huber: 0.2142, swd: 0.0731, ept: 79.4364
    Epoch [6/50], Test Losses: mse: 0.7012, mae: 0.6313, huber: 0.3019, swd: 0.1296, ept: 55.5698
      Epoch 6 composite train-obj: 0.216967
            No improvement (0.2142), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5039, mae: 0.5106, huber: 0.2159, swd: 0.1152, ept: 88.4263
    Epoch [7/50], Val Losses: mse: 0.7119, mae: 0.6085, huber: 0.2886, swd: 0.1122, ept: 74.8910
    Epoch [7/50], Test Losses: mse: 0.8931, mae: 0.7162, huber: 0.3686, swd: 0.1257, ept: 60.0682
      Epoch 7 composite train-obj: 0.215909
            No improvement (0.2886), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4593, mae: 0.4872, huber: 0.1991, swd: 0.0990, ept: 94.1828
    Epoch [8/50], Val Losses: mse: 0.4967, mae: 0.5059, huber: 0.2143, swd: 0.0805, ept: 87.9675
    Epoch [8/50], Test Losses: mse: 0.7845, mae: 0.6650, huber: 0.3299, swd: 0.1608, ept: 62.3530
      Epoch 8 composite train-obj: 0.199139
    Epoch [8/50], Test Losses: mse: 0.7140, mae: 0.6436, huber: 0.3083, swd: 0.1475, ept: 54.6987
    Best round's Test MSE: 0.7140, MAE: 0.6436, SWD: 0.1475
    Best round's Validation MSE: 0.4209, MAE: 0.4688, SWD: 0.0454
    Best round's Test verification MSE : 0.7140, MAE: 0.6436, SWD: 0.1475
    Time taken: 67.54 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq720_pred720_20250510_2015)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.7509 ± 0.0299
      mae: 0.6558 ± 0.0106
      huber: 0.3202 ± 0.0094
      swd: 0.1429 ± 0.0136
      ept: 61.3162 ± 11.0723
      count: 3.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4325 ± 0.0089
      mae: 0.4723 ± 0.0068
      huber: 0.1903 ± 0.0036
      swd: 0.0540 ± 0.0130
      ept: 96.5651 ± 14.5457
      count: 3.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 202.78 seconds
    
    Experiment complete: PatchTST_etth1_seq720_pred720_20250510_2015
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### DLinear

#### pred=96


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 96
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
    
    Epoch [1/50], Train Losses: mse: 0.4430, mae: 0.4695, huber: 0.1908, swd: 0.1224, ept: 40.2244
    Epoch [1/50], Val Losses: mse: 0.3752, mae: 0.4278, huber: 0.1646, swd: 0.1039, ept: 44.7445
    Epoch [1/50], Test Losses: mse: 0.4579, mae: 0.4805, huber: 0.2000, swd: 0.1288, ept: 33.9964
      Epoch 1 composite train-obj: 0.190790
            Val objective improved inf → 0.1646, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3525, mae: 0.4127, huber: 0.1543, swd: 0.1058, ept: 50.9162
    Epoch [2/50], Val Losses: mse: 0.3649, mae: 0.4184, huber: 0.1597, swd: 0.0998, ept: 47.9936
    Epoch [2/50], Test Losses: mse: 0.4517, mae: 0.4734, huber: 0.1968, swd: 0.1301, ept: 35.7990
      Epoch 2 composite train-obj: 0.154300
            Val objective improved 0.1646 → 0.1597, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3456, mae: 0.4068, huber: 0.1510, swd: 0.1035, ept: 52.2420
    Epoch [3/50], Val Losses: mse: 0.3623, mae: 0.4143, huber: 0.1585, swd: 0.0922, ept: 47.4852
    Epoch [3/50], Test Losses: mse: 0.4511, mae: 0.4733, huber: 0.1965, swd: 0.1250, ept: 36.5381
      Epoch 3 composite train-obj: 0.151043
            Val objective improved 0.1597 → 0.1585, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3412, mae: 0.4035, huber: 0.1492, swd: 0.1024, ept: 53.0703
    Epoch [4/50], Val Losses: mse: 0.3657, mae: 0.4150, huber: 0.1593, swd: 0.0899, ept: 49.0833
    Epoch [4/50], Test Losses: mse: 0.4506, mae: 0.4692, huber: 0.1951, swd: 0.1194, ept: 36.8645
      Epoch 4 composite train-obj: 0.149223
            No improvement (0.1593), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3406, mae: 0.4025, huber: 0.1488, swd: 0.1020, ept: 53.2596
    Epoch [5/50], Val Losses: mse: 0.3657, mae: 0.4155, huber: 0.1592, swd: 0.0960, ept: 49.5398
    Epoch [5/50], Test Losses: mse: 0.4460, mae: 0.4671, huber: 0.1936, swd: 0.1246, ept: 36.6634
      Epoch 5 composite train-obj: 0.148760
            No improvement (0.1592), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3394, mae: 0.4014, huber: 0.1482, swd: 0.1022, ept: 53.6062
    Epoch [6/50], Val Losses: mse: 0.3906, mae: 0.4403, huber: 0.1713, swd: 0.1167, ept: 48.5002
    Epoch [6/50], Test Losses: mse: 0.4516, mae: 0.4714, huber: 0.1963, swd: 0.1298, ept: 36.2530
      Epoch 6 composite train-obj: 0.148232
            No improvement (0.1713), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3383, mae: 0.4003, huber: 0.1477, swd: 0.1023, ept: 53.8572
    Epoch [7/50], Val Losses: mse: 0.3656, mae: 0.4154, huber: 0.1592, swd: 0.0954, ept: 49.8854
    Epoch [7/50], Test Losses: mse: 0.4452, mae: 0.4634, huber: 0.1921, swd: 0.1218, ept: 37.0960
      Epoch 7 composite train-obj: 0.147693
            No improvement (0.1592), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3376, mae: 0.3999, huber: 0.1474, swd: 0.1017, ept: 53.8568
    Epoch [8/50], Val Losses: mse: 0.3621, mae: 0.4127, huber: 0.1575, swd: 0.0954, ept: 50.3759
    Epoch [8/50], Test Losses: mse: 0.4521, mae: 0.4708, huber: 0.1965, swd: 0.1285, ept: 36.8317
      Epoch 8 composite train-obj: 0.147407
            Val objective improved 0.1585 → 0.1575, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.3387, mae: 0.4007, huber: 0.1479, swd: 0.1019, ept: 53.7767
    Epoch [9/50], Val Losses: mse: 0.3634, mae: 0.4139, huber: 0.1579, swd: 0.0967, ept: 50.3603
    Epoch [9/50], Test Losses: mse: 0.4462, mae: 0.4665, huber: 0.1933, swd: 0.1297, ept: 36.8318
      Epoch 9 composite train-obj: 0.147906
            No improvement (0.1579), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.3375, mae: 0.3995, huber: 0.1472, swd: 0.1015, ept: 53.9819
    Epoch [10/50], Val Losses: mse: 0.3561, mae: 0.4072, huber: 0.1546, swd: 0.0974, ept: 50.7480
    Epoch [10/50], Test Losses: mse: 0.4459, mae: 0.4667, huber: 0.1937, swd: 0.1327, ept: 37.0850
      Epoch 10 composite train-obj: 0.147246
            Val objective improved 0.1575 → 0.1546, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.3369, mae: 0.3994, huber: 0.1471, swd: 0.1021, ept: 54.1092
    Epoch [11/50], Val Losses: mse: 0.3689, mae: 0.4187, huber: 0.1607, swd: 0.0974, ept: 49.8613
    Epoch [11/50], Test Losses: mse: 0.4519, mae: 0.4687, huber: 0.1952, swd: 0.1235, ept: 36.8458
      Epoch 11 composite train-obj: 0.147147
            No improvement (0.1607), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.3366, mae: 0.3990, huber: 0.1470, swd: 0.1016, ept: 54.1205
    Epoch [12/50], Val Losses: mse: 0.3684, mae: 0.4169, huber: 0.1600, swd: 0.0996, ept: 49.7404
    Epoch [12/50], Test Losses: mse: 0.4523, mae: 0.4692, huber: 0.1958, swd: 0.1285, ept: 37.0800
      Epoch 12 composite train-obj: 0.147010
            No improvement (0.1600), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.3368, mae: 0.3992, huber: 0.1470, swd: 0.1021, ept: 54.1410
    Epoch [13/50], Val Losses: mse: 0.3631, mae: 0.4128, huber: 0.1580, swd: 0.0847, ept: 50.2383
    Epoch [13/50], Test Losses: mse: 0.4431, mae: 0.4646, huber: 0.1916, swd: 0.1162, ept: 37.3402
      Epoch 13 composite train-obj: 0.147028
            No improvement (0.1580), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.3365, mae: 0.3989, huber: 0.1469, swd: 0.1012, ept: 54.2723
    Epoch [14/50], Val Losses: mse: 0.3557, mae: 0.4064, huber: 0.1548, swd: 0.0941, ept: 50.6865
    Epoch [14/50], Test Losses: mse: 0.4530, mae: 0.4713, huber: 0.1964, swd: 0.1319, ept: 36.6871
      Epoch 14 composite train-obj: 0.146934
            No improvement (0.1548), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.3359, mae: 0.3987, huber: 0.1467, swd: 0.1020, ept: 54.2842
    Epoch [15/50], Val Losses: mse: 0.3611, mae: 0.4111, huber: 0.1571, swd: 0.0953, ept: 50.7912
    Epoch [15/50], Test Losses: mse: 0.4425, mae: 0.4629, huber: 0.1916, swd: 0.1257, ept: 37.3025
      Epoch 15 composite train-obj: 0.146721
    Epoch [15/50], Test Losses: mse: 0.4459, mae: 0.4667, huber: 0.1937, swd: 0.1327, ept: 37.0850
    Best round's Test MSE: 0.4459, MAE: 0.4667, SWD: 0.1327
    Best round's Validation MSE: 0.3561, MAE: 0.4072, SWD: 0.0974
    Best round's Test verification MSE : 0.4459, MAE: 0.4667, SWD: 0.1327
    Time taken: 17.69 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4391, mae: 0.4665, huber: 0.1890, swd: 0.1184, ept: 40.7580
    Epoch [1/50], Val Losses: mse: 0.3728, mae: 0.4249, huber: 0.1634, swd: 0.0953, ept: 44.7453
    Epoch [1/50], Test Losses: mse: 0.4582, mae: 0.4812, huber: 0.2000, swd: 0.1216, ept: 33.7902
      Epoch 1 composite train-obj: 0.188968
            Val objective improved inf → 0.1634, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3529, mae: 0.4128, huber: 0.1543, swd: 0.1041, ept: 50.8679
    Epoch [2/50], Val Losses: mse: 0.3700, mae: 0.4209, huber: 0.1618, swd: 0.0954, ept: 47.2435
    Epoch [2/50], Test Losses: mse: 0.4525, mae: 0.4742, huber: 0.1968, swd: 0.1214, ept: 35.5640
      Epoch 2 composite train-obj: 0.154347
            Val objective improved 0.1634 → 0.1618, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3453, mae: 0.4067, huber: 0.1510, swd: 0.1021, ept: 52.2402
    Epoch [3/50], Val Losses: mse: 0.3658, mae: 0.4145, huber: 0.1593, swd: 0.0880, ept: 48.5022
    Epoch [3/50], Test Losses: mse: 0.4483, mae: 0.4691, huber: 0.1942, swd: 0.1144, ept: 36.6268
      Epoch 3 composite train-obj: 0.150988
            Val objective improved 0.1618 → 0.1593, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3422, mae: 0.4040, huber: 0.1496, swd: 0.1007, ept: 52.8462
    Epoch [4/50], Val Losses: mse: 0.3694, mae: 0.4197, huber: 0.1610, swd: 0.0993, ept: 49.4332
    Epoch [4/50], Test Losses: mse: 0.4489, mae: 0.4684, huber: 0.1949, swd: 0.1238, ept: 36.6848
      Epoch 4 composite train-obj: 0.149554
            No improvement (0.1610), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3400, mae: 0.4021, huber: 0.1485, swd: 0.1001, ept: 53.2946
    Epoch [5/50], Val Losses: mse: 0.3582, mae: 0.4105, huber: 0.1560, swd: 0.0941, ept: 50.1244
    Epoch [5/50], Test Losses: mse: 0.4486, mae: 0.4685, huber: 0.1948, swd: 0.1232, ept: 36.7088
      Epoch 5 composite train-obj: 0.148531
            Val objective improved 0.1593 → 0.1560, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3396, mae: 0.4015, huber: 0.1483, swd: 0.1008, ept: 53.5413
    Epoch [6/50], Val Losses: mse: 0.3694, mae: 0.4186, huber: 0.1611, swd: 0.0926, ept: 48.6342
    Epoch [6/50], Test Losses: mse: 0.4490, mae: 0.4679, huber: 0.1944, swd: 0.1185, ept: 36.7941
      Epoch 6 composite train-obj: 0.148295
            No improvement (0.1611), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3382, mae: 0.4006, huber: 0.1477, swd: 0.1002, ept: 53.7451
    Epoch [7/50], Val Losses: mse: 0.3657, mae: 0.4163, huber: 0.1591, swd: 0.0928, ept: 49.6178
    Epoch [7/50], Test Losses: mse: 0.4467, mae: 0.4659, huber: 0.1934, swd: 0.1172, ept: 37.0733
      Epoch 7 composite train-obj: 0.147748
            No improvement (0.1591), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3381, mae: 0.4001, huber: 0.1476, swd: 0.1003, ept: 53.8731
    Epoch [8/50], Val Losses: mse: 0.3661, mae: 0.4158, huber: 0.1590, swd: 0.0909, ept: 49.9499
    Epoch [8/50], Test Losses: mse: 0.4455, mae: 0.4642, huber: 0.1925, swd: 0.1142, ept: 36.6623
      Epoch 8 composite train-obj: 0.147559
            No improvement (0.1590), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3380, mae: 0.4003, huber: 0.1476, swd: 0.1001, ept: 53.9519
    Epoch [9/50], Val Losses: mse: 0.3640, mae: 0.4141, huber: 0.1582, swd: 0.0900, ept: 49.7384
    Epoch [9/50], Test Losses: mse: 0.4473, mae: 0.4677, huber: 0.1941, swd: 0.1207, ept: 36.9658
      Epoch 9 composite train-obj: 0.147611
            No improvement (0.1582), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.3384, mae: 0.4005, huber: 0.1477, swd: 0.1006, ept: 54.0443
    Epoch [10/50], Val Losses: mse: 0.3815, mae: 0.4314, huber: 0.1665, swd: 0.1050, ept: 49.0634
    Epoch [10/50], Test Losses: mse: 0.4504, mae: 0.4672, huber: 0.1945, swd: 0.1206, ept: 36.8423
      Epoch 10 composite train-obj: 0.147741
    Epoch [10/50], Test Losses: mse: 0.4486, mae: 0.4685, huber: 0.1948, swd: 0.1232, ept: 36.7088
    Best round's Test MSE: 0.4486, MAE: 0.4685, SWD: 0.1232
    Best round's Validation MSE: 0.3582, MAE: 0.4105, SWD: 0.0941
    Best round's Test verification MSE : 0.4486, MAE: 0.4685, SWD: 0.1232
    Time taken: 12.10 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4397, mae: 0.4673, huber: 0.1895, swd: 0.1130, ept: 40.5803
    Epoch [1/50], Val Losses: mse: 0.3797, mae: 0.4334, huber: 0.1672, swd: 0.0943, ept: 43.6143
    Epoch [1/50], Test Losses: mse: 0.4572, mae: 0.4799, huber: 0.1994, swd: 0.1145, ept: 34.0428
      Epoch 1 composite train-obj: 0.189503
            Val objective improved inf → 0.1672, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3532, mae: 0.4128, huber: 0.1545, swd: 0.0992, ept: 50.8910
    Epoch [2/50], Val Losses: mse: 0.3617, mae: 0.4165, huber: 0.1586, swd: 0.0889, ept: 47.3402
    Epoch [2/50], Test Losses: mse: 0.4513, mae: 0.4725, huber: 0.1963, swd: 0.1159, ept: 36.1285
      Epoch 2 composite train-obj: 0.154482
            Val objective improved 0.1672 → 0.1586, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3445, mae: 0.4060, huber: 0.1506, swd: 0.0968, ept: 52.4288
    Epoch [3/50], Val Losses: mse: 0.3690, mae: 0.4221, huber: 0.1616, swd: 0.0913, ept: 48.5338
    Epoch [3/50], Test Losses: mse: 0.4451, mae: 0.4667, huber: 0.1931, swd: 0.1123, ept: 36.4328
      Epoch 3 composite train-obj: 0.150595
            No improvement (0.1616), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3422, mae: 0.4037, huber: 0.1495, swd: 0.0956, ept: 52.8974
    Epoch [4/50], Val Losses: mse: 0.3596, mae: 0.4127, huber: 0.1567, swd: 0.0931, ept: 49.7479
    Epoch [4/50], Test Losses: mse: 0.4515, mae: 0.4708, huber: 0.1960, swd: 0.1229, ept: 36.4011
      Epoch 4 composite train-obj: 0.149483
            Val objective improved 0.1586 → 0.1567, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3403, mae: 0.4021, huber: 0.1486, swd: 0.0963, ept: 53.3404
    Epoch [5/50], Val Losses: mse: 0.3706, mae: 0.4205, huber: 0.1618, swd: 0.0924, ept: 49.0382
    Epoch [5/50], Test Losses: mse: 0.4436, mae: 0.4649, huber: 0.1923, swd: 0.1176, ept: 36.7481
      Epoch 5 composite train-obj: 0.148633
            No improvement (0.1618), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3397, mae: 0.4014, huber: 0.1483, swd: 0.0953, ept: 53.5480
    Epoch [6/50], Val Losses: mse: 0.3634, mae: 0.4119, huber: 0.1579, swd: 0.0896, ept: 49.9153
    Epoch [6/50], Test Losses: mse: 0.4430, mae: 0.4645, huber: 0.1920, swd: 0.1154, ept: 36.8701
      Epoch 6 composite train-obj: 0.148289
            No improvement (0.1579), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.3385, mae: 0.4006, huber: 0.1477, swd: 0.0955, ept: 53.7449
    Epoch [7/50], Val Losses: mse: 0.3630, mae: 0.4132, huber: 0.1579, swd: 0.0904, ept: 49.9695
    Epoch [7/50], Test Losses: mse: 0.4487, mae: 0.4675, huber: 0.1942, swd: 0.1162, ept: 36.8980
      Epoch 7 composite train-obj: 0.147744
            No improvement (0.1579), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.3379, mae: 0.3999, huber: 0.1475, swd: 0.0956, ept: 53.8817
    Epoch [8/50], Val Losses: mse: 0.3608, mae: 0.4113, huber: 0.1571, swd: 0.0860, ept: 50.2419
    Epoch [8/50], Test Losses: mse: 0.4411, mae: 0.4622, huber: 0.1907, swd: 0.1135, ept: 37.2601
      Epoch 8 composite train-obj: 0.147488
            No improvement (0.1571), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.3378, mae: 0.4002, huber: 0.1476, swd: 0.0954, ept: 53.9385
    Epoch [9/50], Val Losses: mse: 0.3778, mae: 0.4269, huber: 0.1647, swd: 0.1014, ept: 49.5740
    Epoch [9/50], Test Losses: mse: 0.4490, mae: 0.4670, huber: 0.1945, swd: 0.1200, ept: 36.8336
      Epoch 9 composite train-obj: 0.147568
    Epoch [9/50], Test Losses: mse: 0.4515, mae: 0.4708, huber: 0.1960, swd: 0.1229, ept: 36.4011
    Best round's Test MSE: 0.4515, MAE: 0.4708, SWD: 0.1229
    Best round's Validation MSE: 0.3596, MAE: 0.4127, SWD: 0.0931
    Best round's Test verification MSE : 0.4515, MAE: 0.4708, SWD: 0.1229
    Time taken: 10.71 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq720_pred96_20250510_2019)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4487 ± 0.0023
      mae: 0.4687 ± 0.0016
      huber: 0.1948 ± 0.0009
      swd: 0.1262 ± 0.0045
      ept: 36.7317 ± 0.2796
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3580 ± 0.0014
      mae: 0.4101 ± 0.0023
      huber: 0.1557 ± 0.0009
      swd: 0.0949 ± 0.0018
      ept: 50.2068 ± 0.4124
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 40.56 seconds
    
    Experiment complete: DLinear_etth1_seq720_pred96_20250510_2019
    Model: DLinear
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.5031, mae: 0.4993, huber: 0.2124, swd: 0.1416, ept: 58.1997
    Epoch [1/50], Val Losses: mse: 0.4534, mae: 0.4754, huber: 0.1960, swd: 0.1258, ept: 56.3176
    Epoch [1/50], Test Losses: mse: 0.4983, mae: 0.5101, huber: 0.2186, swd: 0.1226, ept: 48.3835
      Epoch 1 composite train-obj: 0.212377
            Val objective improved inf → 0.1960, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4262, mae: 0.4518, huber: 0.1813, swd: 0.1313, ept: 75.2348
    Epoch [2/50], Val Losses: mse: 0.4538, mae: 0.4712, huber: 0.1947, swd: 0.1232, ept: 59.3851
    Epoch [2/50], Test Losses: mse: 0.4903, mae: 0.5017, huber: 0.2143, swd: 0.1166, ept: 51.7798
      Epoch 2 composite train-obj: 0.181345
            Val objective improved 0.1960 → 0.1947, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4207, mae: 0.4469, huber: 0.1787, swd: 0.1302, ept: 77.8757
    Epoch [3/50], Val Losses: mse: 0.4560, mae: 0.4718, huber: 0.1956, swd: 0.1246, ept: 59.7601
    Epoch [3/50], Test Losses: mse: 0.5022, mae: 0.5094, huber: 0.2195, swd: 0.1232, ept: 51.6241
      Epoch 3 composite train-obj: 0.178671
            No improvement (0.1956), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4189, mae: 0.4449, huber: 0.1776, swd: 0.1295, ept: 78.6219
    Epoch [4/50], Val Losses: mse: 0.4377, mae: 0.4593, huber: 0.1880, swd: 0.1134, ept: 62.5779
    Epoch [4/50], Test Losses: mse: 0.4956, mae: 0.5054, huber: 0.2166, swd: 0.1227, ept: 53.2081
      Epoch 4 composite train-obj: 0.177592
            Val objective improved 0.1947 → 0.1880, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4158, mae: 0.4429, huber: 0.1764, swd: 0.1287, ept: 79.8683
    Epoch [5/50], Val Losses: mse: 0.4334, mae: 0.4561, huber: 0.1856, swd: 0.1148, ept: 64.7678
    Epoch [5/50], Test Losses: mse: 0.4912, mae: 0.4998, huber: 0.2142, swd: 0.1222, ept: 53.5256
      Epoch 5 composite train-obj: 0.176351
            Val objective improved 0.1880 → 0.1856, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4149, mae: 0.4420, huber: 0.1759, swd: 0.1294, ept: 79.9133
    Epoch [6/50], Val Losses: mse: 0.4326, mae: 0.4578, huber: 0.1857, swd: 0.1090, ept: 65.5140
    Epoch [6/50], Test Losses: mse: 0.4918, mae: 0.5008, huber: 0.2144, swd: 0.1214, ept: 53.2030
      Epoch 6 composite train-obj: 0.175914
            No improvement (0.1857), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4144, mae: 0.4419, huber: 0.1758, swd: 0.1288, ept: 80.1368
    Epoch [7/50], Val Losses: mse: 0.4495, mae: 0.4682, huber: 0.1924, swd: 0.1277, ept: 61.7338
    Epoch [7/50], Test Losses: mse: 0.4869, mae: 0.4972, huber: 0.2124, swd: 0.1296, ept: 50.9931
      Epoch 7 composite train-obj: 0.175780
            No improvement (0.1924), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4138, mae: 0.4416, huber: 0.1756, swd: 0.1291, ept: 80.3044
    Epoch [8/50], Val Losses: mse: 0.4473, mae: 0.4663, huber: 0.1919, swd: 0.1261, ept: 62.1537
    Epoch [8/50], Test Losses: mse: 0.4908, mae: 0.4983, huber: 0.2133, swd: 0.1208, ept: 52.9474
      Epoch 8 composite train-obj: 0.175580
            No improvement (0.1919), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.4118, mae: 0.4400, huber: 0.1747, swd: 0.1283, ept: 80.8648
    Epoch [9/50], Val Losses: mse: 0.4406, mae: 0.4608, huber: 0.1889, swd: 0.1157, ept: 63.6497
    Epoch [9/50], Test Losses: mse: 0.4883, mae: 0.4965, huber: 0.2123, swd: 0.1200, ept: 53.1524
      Epoch 9 composite train-obj: 0.174683
            No improvement (0.1889), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.4123, mae: 0.4403, huber: 0.1749, swd: 0.1288, ept: 80.8734
    Epoch [10/50], Val Losses: mse: 0.4446, mae: 0.4625, huber: 0.1899, swd: 0.1148, ept: 64.8602
    Epoch [10/50], Test Losses: mse: 0.4841, mae: 0.4941, huber: 0.2103, swd: 0.1151, ept: 53.5934
      Epoch 10 composite train-obj: 0.174910
    Epoch [10/50], Test Losses: mse: 0.4912, mae: 0.4998, huber: 0.2142, swd: 0.1222, ept: 53.5256
    Best round's Test MSE: 0.4912, MAE: 0.4998, SWD: 0.1222
    Best round's Validation MSE: 0.4334, MAE: 0.4561, SWD: 0.1148
    Best round's Test verification MSE : 0.4912, MAE: 0.4998, SWD: 0.1222
    Time taken: 12.10 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5033, mae: 0.4990, huber: 0.2123, swd: 0.1420, ept: 57.8714
    Epoch [1/50], Val Losses: mse: 0.4408, mae: 0.4639, huber: 0.1895, swd: 0.1118, ept: 57.4753
    Epoch [1/50], Test Losses: mse: 0.4996, mae: 0.5121, huber: 0.2194, swd: 0.1197, ept: 48.4071
      Epoch 1 composite train-obj: 0.212252
            Val objective improved inf → 0.1895, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4272, mae: 0.4521, huber: 0.1816, swd: 0.1333, ept: 75.3172
    Epoch [2/50], Val Losses: mse: 0.4388, mae: 0.4603, huber: 0.1880, swd: 0.1163, ept: 60.8473
    Epoch [2/50], Test Losses: mse: 0.4892, mae: 0.5013, huber: 0.2140, swd: 0.1186, ept: 52.1128
      Epoch 2 composite train-obj: 0.181625
            Val objective improved 0.1895 → 0.1880, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4214, mae: 0.4470, huber: 0.1788, swd: 0.1304, ept: 78.0745
    Epoch [3/50], Val Losses: mse: 0.4441, mae: 0.4627, huber: 0.1898, swd: 0.1155, ept: 62.2052
    Epoch [3/50], Test Losses: mse: 0.4988, mae: 0.5064, huber: 0.2176, swd: 0.1203, ept: 51.9245
      Epoch 3 composite train-obj: 0.178776
            No improvement (0.1898), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4192, mae: 0.4455, huber: 0.1780, swd: 0.1321, ept: 78.7534
    Epoch [4/50], Val Losses: mse: 0.4438, mae: 0.4632, huber: 0.1896, swd: 0.1129, ept: 62.8252
    Epoch [4/50], Test Losses: mse: 0.4947, mae: 0.5018, huber: 0.2154, swd: 0.1149, ept: 52.1141
      Epoch 4 composite train-obj: 0.177992
            No improvement (0.1896), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4147, mae: 0.4432, huber: 0.1763, swd: 0.1301, ept: 79.3892
    Epoch [5/50], Val Losses: mse: 0.4384, mae: 0.4581, huber: 0.1874, swd: 0.1151, ept: 62.2366
    Epoch [5/50], Test Losses: mse: 0.4926, mae: 0.5003, huber: 0.2145, swd: 0.1209, ept: 53.1628
      Epoch 5 composite train-obj: 0.176315
            Val objective improved 0.1880 → 0.1874, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4153, mae: 0.4423, huber: 0.1761, swd: 0.1301, ept: 79.7652
    Epoch [6/50], Val Losses: mse: 0.4282, mae: 0.4530, huber: 0.1835, swd: 0.1082, ept: 65.6077
    Epoch [6/50], Test Losses: mse: 0.4905, mae: 0.4996, huber: 0.2137, swd: 0.1231, ept: 52.8122
      Epoch 6 composite train-obj: 0.176054
            Val objective improved 0.1874 → 0.1835, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4131, mae: 0.4413, huber: 0.1754, swd: 0.1297, ept: 80.1734
    Epoch [7/50], Val Losses: mse: 0.4411, mae: 0.4627, huber: 0.1891, swd: 0.1212, ept: 65.1449
    Epoch [7/50], Test Losses: mse: 0.4961, mae: 0.5022, huber: 0.2159, swd: 0.1276, ept: 53.2856
      Epoch 7 composite train-obj: 0.175355
            No improvement (0.1891), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4162, mae: 0.4422, huber: 0.1762, swd: 0.1308, ept: 80.4584
    Epoch [8/50], Val Losses: mse: 0.4389, mae: 0.4581, huber: 0.1876, swd: 0.1119, ept: 62.3638
    Epoch [8/50], Test Losses: mse: 0.5027, mae: 0.5100, huber: 0.2194, swd: 0.1241, ept: 53.1590
      Epoch 8 composite train-obj: 0.176232
            No improvement (0.1876), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.4135, mae: 0.4417, huber: 0.1756, swd: 0.1298, ept: 80.5492
    Epoch [9/50], Val Losses: mse: 0.4445, mae: 0.4607, huber: 0.1895, swd: 0.1207, ept: 64.1713
    Epoch [9/50], Test Losses: mse: 0.4926, mae: 0.5000, huber: 0.2147, swd: 0.1239, ept: 53.2479
      Epoch 9 composite train-obj: 0.175603
            No improvement (0.1895), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.4144, mae: 0.4410, huber: 0.1756, swd: 0.1308, ept: 80.6402
    Epoch [10/50], Val Losses: mse: 0.4399, mae: 0.4622, huber: 0.1889, swd: 0.1241, ept: 63.7979
    Epoch [10/50], Test Losses: mse: 0.4865, mae: 0.4962, huber: 0.2120, swd: 0.1228, ept: 52.7911
      Epoch 10 composite train-obj: 0.175563
            No improvement (0.1889), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.4178, mae: 0.4425, huber: 0.1765, swd: 0.1304, ept: 80.7356
    Epoch [11/50], Val Losses: mse: 0.4215, mae: 0.4487, huber: 0.1804, swd: 0.1130, ept: 68.2554
    Epoch [11/50], Test Losses: mse: 0.4979, mae: 0.5065, huber: 0.2175, swd: 0.1380, ept: 53.0354
      Epoch 11 composite train-obj: 0.176510
            Val objective improved 0.1835 → 0.1804, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.4140, mae: 0.4415, huber: 0.1756, swd: 0.1315, ept: 80.6878
    Epoch [12/50], Val Losses: mse: 0.4393, mae: 0.4578, huber: 0.1875, swd: 0.1183, ept: 64.1787
    Epoch [12/50], Test Losses: mse: 0.4939, mae: 0.5025, huber: 0.2155, swd: 0.1235, ept: 52.4968
      Epoch 12 composite train-obj: 0.175628
            No improvement (0.1875), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.4117, mae: 0.4400, huber: 0.1747, swd: 0.1298, ept: 81.2080
    Epoch [13/50], Val Losses: mse: 0.4361, mae: 0.4587, huber: 0.1866, swd: 0.1131, ept: 66.0083
    Epoch [13/50], Test Losses: mse: 0.4904, mae: 0.4970, huber: 0.2132, swd: 0.1205, ept: 53.6148
      Epoch 13 composite train-obj: 0.174691
            No improvement (0.1866), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.4130, mae: 0.4402, huber: 0.1749, swd: 0.1310, ept: 81.1621
    Epoch [14/50], Val Losses: mse: 0.4387, mae: 0.4596, huber: 0.1879, swd: 0.1176, ept: 63.9621
    Epoch [14/50], Test Losses: mse: 0.4846, mae: 0.4946, huber: 0.2111, swd: 0.1188, ept: 53.5697
      Epoch 14 composite train-obj: 0.174935
            No improvement (0.1879), counter 3/5
    Epoch [15/50], Train Losses: mse: 0.4133, mae: 0.4402, huber: 0.1750, swd: 0.1314, ept: 81.2910
    Epoch [15/50], Val Losses: mse: 0.4412, mae: 0.4603, huber: 0.1888, swd: 0.1148, ept: 64.5096
    Epoch [15/50], Test Losses: mse: 0.4922, mae: 0.4994, huber: 0.2146, swd: 0.1188, ept: 53.0846
      Epoch 15 composite train-obj: 0.175045
            No improvement (0.1888), counter 4/5
    Epoch [16/50], Train Losses: mse: 0.4108, mae: 0.4393, huber: 0.1743, swd: 0.1302, ept: 81.4802
    Epoch [16/50], Val Losses: mse: 0.4460, mae: 0.4638, huber: 0.1909, swd: 0.1197, ept: 62.7161
    Epoch [16/50], Test Losses: mse: 0.4869, mae: 0.4953, huber: 0.2119, swd: 0.1130, ept: 53.4290
      Epoch 16 composite train-obj: 0.174307
    Epoch [16/50], Test Losses: mse: 0.4979, mae: 0.5065, huber: 0.2175, swd: 0.1380, ept: 53.0354
    Best round's Test MSE: 0.4979, MAE: 0.5065, SWD: 0.1380
    Best round's Validation MSE: 0.4215, MAE: 0.4487, SWD: 0.1130
    Best round's Test verification MSE : 0.4979, MAE: 0.5065, SWD: 0.1380
    Time taken: 18.88 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5041, mae: 0.4995, huber: 0.2125, swd: 0.1295, ept: 57.8188
    Epoch [1/50], Val Losses: mse: 0.4446, mae: 0.4677, huber: 0.1915, swd: 0.1095, ept: 56.7108
    Epoch [1/50], Test Losses: mse: 0.4960, mae: 0.5086, huber: 0.2175, swd: 0.1078, ept: 49.0356
      Epoch 1 composite train-obj: 0.212543
            Val objective improved inf → 0.1915, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4258, mae: 0.4512, huber: 0.1810, swd: 0.1209, ept: 75.4129
    Epoch [2/50], Val Losses: mse: 0.4384, mae: 0.4611, huber: 0.1881, swd: 0.1026, ept: 60.8648
    Epoch [2/50], Test Losses: mse: 0.4925, mae: 0.5014, huber: 0.2144, swd: 0.1022, ept: 51.7582
      Epoch 2 composite train-obj: 0.180990
            Val objective improved 0.1915 → 0.1881, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4193, mae: 0.4459, huber: 0.1780, swd: 0.1190, ept: 78.1191
    Epoch [3/50], Val Losses: mse: 0.4461, mae: 0.4643, huber: 0.1911, swd: 0.1072, ept: 63.6793
    Epoch [3/50], Test Losses: mse: 0.4951, mae: 0.5023, huber: 0.2155, swd: 0.1080, ept: 52.4966
      Epoch 3 composite train-obj: 0.178033
            No improvement (0.1911), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4176, mae: 0.4442, huber: 0.1771, swd: 0.1190, ept: 78.9447
    Epoch [4/50], Val Losses: mse: 0.4442, mae: 0.4606, huber: 0.1893, swd: 0.1069, ept: 62.6738
    Epoch [4/50], Test Losses: mse: 0.5015, mae: 0.5083, huber: 0.2185, swd: 0.1167, ept: 52.7979
      Epoch 4 composite train-obj: 0.177143
            No improvement (0.1893), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4154, mae: 0.4429, huber: 0.1763, swd: 0.1183, ept: 79.4926
    Epoch [5/50], Val Losses: mse: 0.4312, mae: 0.4543, huber: 0.1842, swd: 0.1042, ept: 65.4349
    Epoch [5/50], Test Losses: mse: 0.4897, mae: 0.4992, huber: 0.2134, swd: 0.1123, ept: 52.5841
      Epoch 5 composite train-obj: 0.176322
            Val objective improved 0.1881 → 0.1842, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4156, mae: 0.4426, huber: 0.1762, swd: 0.1198, ept: 79.8993
    Epoch [6/50], Val Losses: mse: 0.4355, mae: 0.4587, huber: 0.1869, swd: 0.1052, ept: 62.8374
    Epoch [6/50], Test Losses: mse: 0.4849, mae: 0.4946, huber: 0.2112, swd: 0.1060, ept: 53.0076
      Epoch 6 composite train-obj: 0.176194
            No improvement (0.1869), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4146, mae: 0.4421, huber: 0.1758, swd: 0.1175, ept: 79.9026
    Epoch [7/50], Val Losses: mse: 0.4474, mae: 0.4646, huber: 0.1916, swd: 0.1132, ept: 62.3078
    Epoch [7/50], Test Losses: mse: 0.4960, mae: 0.5031, huber: 0.2163, swd: 0.1159, ept: 52.8535
      Epoch 7 composite train-obj: 0.175831
            No improvement (0.1916), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4140, mae: 0.4412, huber: 0.1755, swd: 0.1180, ept: 80.2219
    Epoch [8/50], Val Losses: mse: 0.4493, mae: 0.4641, huber: 0.1916, swd: 0.1138, ept: 62.8489
    Epoch [8/50], Test Losses: mse: 0.4954, mae: 0.5042, huber: 0.2162, swd: 0.1209, ept: 53.6322
      Epoch 8 composite train-obj: 0.175529
            No improvement (0.1916), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.4144, mae: 0.4413, huber: 0.1756, swd: 0.1191, ept: 80.5816
    Epoch [9/50], Val Losses: mse: 0.4300, mae: 0.4532, huber: 0.1840, swd: 0.1004, ept: 64.8608
    Epoch [9/50], Test Losses: mse: 0.4907, mae: 0.4981, huber: 0.2135, swd: 0.1089, ept: 54.1618
      Epoch 9 composite train-obj: 0.175620
            Val objective improved 0.1842 → 0.1840, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.4127, mae: 0.4408, huber: 0.1751, swd: 0.1186, ept: 80.8397
    Epoch [10/50], Val Losses: mse: 0.4372, mae: 0.4597, huber: 0.1875, swd: 0.1057, ept: 63.1544
    Epoch [10/50], Test Losses: mse: 0.4856, mae: 0.4947, huber: 0.2115, swd: 0.1098, ept: 52.7226
      Epoch 10 composite train-obj: 0.175150
            No improvement (0.1875), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.4137, mae: 0.4409, huber: 0.1754, swd: 0.1184, ept: 80.7182
    Epoch [11/50], Val Losses: mse: 0.4461, mae: 0.4620, huber: 0.1905, swd: 0.1115, ept: 63.3082
    Epoch [11/50], Test Losses: mse: 0.4961, mae: 0.5038, huber: 0.2165, swd: 0.1161, ept: 52.9016
      Epoch 11 composite train-obj: 0.175371
            No improvement (0.1905), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.4127, mae: 0.4401, huber: 0.1749, swd: 0.1182, ept: 81.2863
    Epoch [12/50], Val Losses: mse: 0.4367, mae: 0.4573, huber: 0.1865, swd: 0.1062, ept: 64.7066
    Epoch [12/50], Test Losses: mse: 0.4983, mae: 0.5063, huber: 0.2175, swd: 0.1193, ept: 51.6637
      Epoch 12 composite train-obj: 0.174920
            No improvement (0.1865), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.4132, mae: 0.4405, huber: 0.1751, swd: 0.1190, ept: 80.9054
    Epoch [13/50], Val Losses: mse: 0.4405, mae: 0.4601, huber: 0.1888, swd: 0.1062, ept: 63.7567
    Epoch [13/50], Test Losses: mse: 0.4878, mae: 0.4967, huber: 0.2123, swd: 0.1099, ept: 52.9169
      Epoch 13 composite train-obj: 0.175100
            No improvement (0.1888), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.4104, mae: 0.4391, huber: 0.1742, swd: 0.1175, ept: 81.0231
    Epoch [14/50], Val Losses: mse: 0.4643, mae: 0.4746, huber: 0.1979, swd: 0.1174, ept: 62.0688
    Epoch [14/50], Test Losses: mse: 0.5029, mae: 0.5049, huber: 0.2179, swd: 0.1088, ept: 52.5350
      Epoch 14 composite train-obj: 0.174156
    Epoch [14/50], Test Losses: mse: 0.4907, mae: 0.4981, huber: 0.2135, swd: 0.1089, ept: 54.1618
    Best round's Test MSE: 0.4907, MAE: 0.4981, SWD: 0.1089
    Best round's Validation MSE: 0.4300, MAE: 0.4532, SWD: 0.1004
    Best round's Test verification MSE : 0.4907, MAE: 0.4981, SWD: 0.1089
    Time taken: 16.85 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq720_pred196_20250510_2019)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4933 ± 0.0033
      mae: 0.5015 ± 0.0036
      huber: 0.2150 ± 0.0017
      swd: 0.1230 ± 0.0119
      ept: 53.5743 ± 0.4612
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4283 ± 0.0050
      mae: 0.4527 ± 0.0030
      huber: 0.1833 ± 0.0022
      swd: 0.1094 ± 0.0064
      ept: 65.9613 ± 1.6226
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 47.90 seconds
    
    Experiment complete: DLinear_etth1_seq720_pred196_20250510_2019
    Model: DLinear
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 88
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 88
    Validation Batches: 6
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5492, mae: 0.5246, huber: 0.2302, swd: 0.1531, ept: 71.8455
    Epoch [1/50], Val Losses: mse: 0.4852, mae: 0.4891, huber: 0.2064, swd: 0.1094, ept: 66.7849
    Epoch [1/50], Test Losses: mse: 0.5753, mae: 0.5573, huber: 0.2512, swd: 0.1241, ept: 62.2349
      Epoch 1 composite train-obj: 0.230248
            Val objective improved inf → 0.2064, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4865, mae: 0.4852, huber: 0.2044, swd: 0.1465, ept: 94.8506
    Epoch [2/50], Val Losses: mse: 0.4777, mae: 0.4798, huber: 0.2015, swd: 0.1113, ept: 73.3957
    Epoch [2/50], Test Losses: mse: 0.5617, mae: 0.5493, huber: 0.2455, swd: 0.1326, ept: 65.4289
      Epoch 2 composite train-obj: 0.204421
            Val objective improved 0.2064 → 0.2015, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4785, mae: 0.4798, huber: 0.2010, swd: 0.1453, ept: 98.9025
    Epoch [3/50], Val Losses: mse: 0.4717, mae: 0.4753, huber: 0.1985, swd: 0.1091, ept: 75.7818
    Epoch [3/50], Test Losses: mse: 0.5588, mae: 0.5472, huber: 0.2441, swd: 0.1338, ept: 67.0333
      Epoch 3 composite train-obj: 0.201033
            Val objective improved 0.2015 → 0.1985, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4765, mae: 0.4787, huber: 0.2003, swd: 0.1462, ept: 98.7849
    Epoch [4/50], Val Losses: mse: 0.4928, mae: 0.4915, huber: 0.2078, swd: 0.1035, ept: 68.4138
    Epoch [4/50], Test Losses: mse: 0.5653, mae: 0.5492, huber: 0.2462, swd: 0.1189, ept: 62.7374
      Epoch 4 composite train-obj: 0.200305
            No improvement (0.2078), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4777, mae: 0.4785, huber: 0.2005, swd: 0.1445, ept: 99.5640
    Epoch [5/50], Val Losses: mse: 0.4670, mae: 0.4745, huber: 0.1978, swd: 0.1110, ept: 72.6613
    Epoch [5/50], Test Losses: mse: 0.5734, mae: 0.5544, huber: 0.2504, swd: 0.1330, ept: 64.9547
      Epoch 5 composite train-obj: 0.200476
            Val objective improved 0.1985 → 0.1978, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4733, mae: 0.4762, huber: 0.1989, swd: 0.1454, ept: 101.0942
    Epoch [6/50], Val Losses: mse: 0.4849, mae: 0.4809, huber: 0.2031, swd: 0.1057, ept: 71.8121
    Epoch [6/50], Test Losses: mse: 0.5638, mae: 0.5457, huber: 0.2453, swd: 0.1236, ept: 61.7923
      Epoch 6 composite train-obj: 0.198879
            No improvement (0.2031), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4757, mae: 0.4764, huber: 0.1993, swd: 0.1482, ept: 101.7674
    Epoch [7/50], Val Losses: mse: 0.4714, mae: 0.4768, huber: 0.1994, swd: 0.1127, ept: 73.7721
    Epoch [7/50], Test Losses: mse: 0.5529, mae: 0.5414, huber: 0.2414, swd: 0.1237, ept: 65.9168
      Epoch 7 composite train-obj: 0.199270
            No improvement (0.1994), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4745, mae: 0.4760, huber: 0.1991, swd: 0.1449, ept: 102.0637
    Epoch [8/50], Val Losses: mse: 0.4952, mae: 0.4915, huber: 0.2094, swd: 0.1132, ept: 69.3616
    Epoch [8/50], Test Losses: mse: 0.5578, mae: 0.5454, huber: 0.2437, swd: 0.1263, ept: 58.8344
      Epoch 8 composite train-obj: 0.199089
            No improvement (0.2094), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.4720, mae: 0.4747, huber: 0.1982, swd: 0.1451, ept: 102.4248
    Epoch [9/50], Val Losses: mse: 0.4960, mae: 0.4887, huber: 0.2082, swd: 0.1197, ept: 73.9876
    Epoch [9/50], Test Losses: mse: 0.5727, mae: 0.5529, huber: 0.2494, swd: 0.1331, ept: 67.0954
      Epoch 9 composite train-obj: 0.198177
            No improvement (0.2082), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.4748, mae: 0.4754, huber: 0.1989, swd: 0.1442, ept: 103.0496
    Epoch [10/50], Val Losses: mse: 0.4775, mae: 0.4801, huber: 0.2015, swd: 0.1044, ept: 76.2931
    Epoch [10/50], Test Losses: mse: 0.5646, mae: 0.5455, huber: 0.2453, swd: 0.1235, ept: 63.4137
      Epoch 10 composite train-obj: 0.198865
    Epoch [10/50], Test Losses: mse: 0.5734, mae: 0.5544, huber: 0.2504, swd: 0.1330, ept: 64.9547
    Best round's Test MSE: 0.5734, MAE: 0.5544, SWD: 0.1330
    Best round's Validation MSE: 0.4670, MAE: 0.4745, SWD: 0.1110
    Best round's Test verification MSE : 0.5734, MAE: 0.5544, SWD: 0.1330
    Time taken: 9.25 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5570, mae: 0.5279, huber: 0.2328, swd: 0.1609, ept: 72.1613
    Epoch [1/50], Val Losses: mse: 0.4794, mae: 0.4828, huber: 0.2022, swd: 0.1061, ept: 71.1224
    Epoch [1/50], Test Losses: mse: 0.5611, mae: 0.5516, huber: 0.2458, swd: 0.1282, ept: 62.3502
      Epoch 1 composite train-obj: 0.232821
            Val objective improved inf → 0.2022, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4853, mae: 0.4857, huber: 0.2044, swd: 0.1511, ept: 94.4887
    Epoch [2/50], Val Losses: mse: 0.4709, mae: 0.4803, huber: 0.2000, swd: 0.1092, ept: 69.2675
    Epoch [2/50], Test Losses: mse: 0.5526, mae: 0.5421, huber: 0.2414, swd: 0.1241, ept: 64.3440
      Epoch 2 composite train-obj: 0.204416
            Val objective improved 0.2022 → 0.2000, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4788, mae: 0.4800, huber: 0.2012, swd: 0.1497, ept: 98.5573
    Epoch [3/50], Val Losses: mse: 0.4677, mae: 0.4771, huber: 0.1988, swd: 0.1063, ept: 72.7488
    Epoch [3/50], Test Losses: mse: 0.5604, mae: 0.5444, huber: 0.2444, swd: 0.1200, ept: 64.3327
      Epoch 3 composite train-obj: 0.201203
            Val objective improved 0.2000 → 0.1988, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4806, mae: 0.4798, huber: 0.2016, swd: 0.1500, ept: 100.1627
    Epoch [4/50], Val Losses: mse: 0.4679, mae: 0.4731, huber: 0.1971, swd: 0.0952, ept: 69.9330
    Epoch [4/50], Test Losses: mse: 0.5585, mae: 0.5450, huber: 0.2433, swd: 0.1174, ept: 62.2328
      Epoch 4 composite train-obj: 0.201556
            Val objective improved 0.1988 → 0.1971, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4825, mae: 0.4810, huber: 0.2023, swd: 0.1494, ept: 99.5458
    Epoch [5/50], Val Losses: mse: 0.4770, mae: 0.4820, huber: 0.2017, swd: 0.1213, ept: 71.1136
    Epoch [5/50], Test Losses: mse: 0.5711, mae: 0.5528, huber: 0.2489, swd: 0.1403, ept: 63.7263
      Epoch 5 composite train-obj: 0.202316
            No improvement (0.2017), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4827, mae: 0.4800, huber: 0.2018, swd: 0.1520, ept: 101.2565
    Epoch [6/50], Val Losses: mse: 0.4691, mae: 0.4763, huber: 0.1984, swd: 0.1041, ept: 72.6678
    Epoch [6/50], Test Losses: mse: 0.5580, mae: 0.5449, huber: 0.2434, swd: 0.1266, ept: 63.2347
      Epoch 6 composite train-obj: 0.201831
            No improvement (0.1984), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.4743, mae: 0.4761, huber: 0.1991, swd: 0.1487, ept: 101.3195
    Epoch [7/50], Val Losses: mse: 0.5139, mae: 0.5099, huber: 0.2189, swd: 0.1497, ept: 54.3338
    Epoch [7/50], Test Losses: mse: 0.5724, mae: 0.5558, huber: 0.2498, swd: 0.1446, ept: 55.6126
      Epoch 7 composite train-obj: 0.199094
            No improvement (0.2189), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.4797, mae: 0.4780, huber: 0.2007, swd: 0.1516, ept: 101.1151
    Epoch [8/50], Val Losses: mse: 0.4737, mae: 0.4773, huber: 0.1996, swd: 0.0864, ept: 75.5952
    Epoch [8/50], Test Losses: mse: 0.5620, mae: 0.5492, huber: 0.2452, swd: 0.1169, ept: 63.5778
      Epoch 8 composite train-obj: 0.200670
            No improvement (0.1996), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.4783, mae: 0.4790, huber: 0.2009, swd: 0.1483, ept: 101.4841
    Epoch [9/50], Val Losses: mse: 0.4749, mae: 0.4784, huber: 0.2012, swd: 0.1131, ept: 71.5654
    Epoch [9/50], Test Losses: mse: 0.5665, mae: 0.5494, huber: 0.2474, swd: 0.1346, ept: 63.2958
      Epoch 9 composite train-obj: 0.200854
    Epoch [9/50], Test Losses: mse: 0.5585, mae: 0.5450, huber: 0.2433, swd: 0.1174, ept: 62.2328
    Best round's Test MSE: 0.5585, MAE: 0.5450, SWD: 0.1174
    Best round's Validation MSE: 0.4679, MAE: 0.4731, SWD: 0.0952
    Best round's Test verification MSE : 0.5585, MAE: 0.5450, SWD: 0.1174
    Time taken: 7.97 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5504, mae: 0.5246, huber: 0.2304, swd: 0.1542, ept: 71.5924
    Epoch [1/50], Val Losses: mse: 0.4886, mae: 0.4912, huber: 0.2071, swd: 0.1127, ept: 66.6300
    Epoch [1/50], Test Losses: mse: 0.5740, mae: 0.5605, huber: 0.2520, swd: 0.1362, ept: 58.0968
      Epoch 1 composite train-obj: 0.230419
            Val objective improved inf → 0.2071, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4839, mae: 0.4846, huber: 0.2038, swd: 0.1485, ept: 94.6519
    Epoch [2/50], Val Losses: mse: 0.4729, mae: 0.4807, huber: 0.2007, swd: 0.1072, ept: 69.7555
    Epoch [2/50], Test Losses: mse: 0.5633, mae: 0.5504, huber: 0.2467, swd: 0.1309, ept: 61.6918
      Epoch 2 composite train-obj: 0.203754
            Val objective improved 0.2071 → 0.2007, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4801, mae: 0.4805, huber: 0.2016, swd: 0.1480, ept: 98.0515
    Epoch [3/50], Val Losses: mse: 0.4897, mae: 0.4889, huber: 0.2068, swd: 0.1146, ept: 71.7379
    Epoch [3/50], Test Losses: mse: 0.5619, mae: 0.5485, huber: 0.2453, swd: 0.1294, ept: 65.1776
      Epoch 3 composite train-obj: 0.201623
            No improvement (0.2068), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4784, mae: 0.4792, huber: 0.2009, swd: 0.1468, ept: 99.2403
    Epoch [4/50], Val Losses: mse: 0.4873, mae: 0.4863, huber: 0.2058, swd: 0.1170, ept: 71.6730
    Epoch [4/50], Test Losses: mse: 0.5608, mae: 0.5484, huber: 0.2452, swd: 0.1337, ept: 66.4816
      Epoch 4 composite train-obj: 0.200863
            No improvement (0.2058), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4747, mae: 0.4770, huber: 0.1995, swd: 0.1464, ept: 100.4570
    Epoch [5/50], Val Losses: mse: 0.4810, mae: 0.4849, huber: 0.2036, swd: 0.1148, ept: 72.0371
    Epoch [5/50], Test Losses: mse: 0.5515, mae: 0.5392, huber: 0.2405, swd: 0.1313, ept: 64.9744
      Epoch 5 composite train-obj: 0.199452
            No improvement (0.2036), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4741, mae: 0.4762, huber: 0.1990, swd: 0.1460, ept: 101.8488
    Epoch [6/50], Val Losses: mse: 0.4806, mae: 0.4814, huber: 0.2029, swd: 0.1181, ept: 72.8813
    Epoch [6/50], Test Losses: mse: 0.5556, mae: 0.5425, huber: 0.2423, swd: 0.1332, ept: 66.4263
      Epoch 6 composite train-obj: 0.199044
            No improvement (0.2029), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4742, mae: 0.4763, huber: 0.1991, swd: 0.1479, ept: 101.7015
    Epoch [7/50], Val Losses: mse: 0.4706, mae: 0.4749, huber: 0.1989, swd: 0.1026, ept: 70.2080
    Epoch [7/50], Test Losses: mse: 0.5606, mae: 0.5472, huber: 0.2450, swd: 0.1258, ept: 63.1855
      Epoch 7 composite train-obj: 0.199097
            Val objective improved 0.2007 → 0.1989, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4761, mae: 0.4769, huber: 0.1997, swd: 0.1482, ept: 102.8348
    Epoch [8/50], Val Losses: mse: 0.5061, mae: 0.5003, huber: 0.2151, swd: 0.1029, ept: 52.5972
    Epoch [8/50], Test Losses: mse: 0.5653, mae: 0.5486, huber: 0.2465, swd: 0.1191, ept: 51.0888
      Epoch 8 composite train-obj: 0.199691
            No improvement (0.2151), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4780, mae: 0.4777, huber: 0.2003, swd: 0.1477, ept: 100.7954
    Epoch [9/50], Val Losses: mse: 0.4833, mae: 0.4831, huber: 0.2041, swd: 0.1174, ept: 73.7561
    Epoch [9/50], Test Losses: mse: 0.5870, mae: 0.5664, huber: 0.2569, swd: 0.1545, ept: 65.8752
      Epoch 9 composite train-obj: 0.200298
            No improvement (0.2041), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.4834, mae: 0.4790, huber: 0.2016, swd: 0.1500, ept: 101.7871
    Epoch [10/50], Val Losses: mse: 0.4661, mae: 0.4683, huber: 0.1953, swd: 0.0965, ept: 84.5072
    Epoch [10/50], Test Losses: mse: 0.5663, mae: 0.5486, huber: 0.2465, swd: 0.1243, ept: 65.2487
      Epoch 10 composite train-obj: 0.201578
            Val objective improved 0.1989 → 0.1953, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.4778, mae: 0.4767, huber: 0.1999, swd: 0.1496, ept: 102.9380
    Epoch [11/50], Val Losses: mse: 0.4851, mae: 0.4800, huber: 0.2035, swd: 0.1048, ept: 74.6031
    Epoch [11/50], Test Losses: mse: 0.5812, mae: 0.5604, huber: 0.2533, swd: 0.1332, ept: 65.0201
      Epoch 11 composite train-obj: 0.199917
            No improvement (0.2035), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.4750, mae: 0.4769, huber: 0.1995, swd: 0.1470, ept: 101.8864
    Epoch [12/50], Val Losses: mse: 0.4979, mae: 0.4941, huber: 0.2104, swd: 0.1170, ept: 73.2665
    Epoch [12/50], Test Losses: mse: 0.5686, mae: 0.5518, huber: 0.2476, swd: 0.1324, ept: 65.8428
      Epoch 12 composite train-obj: 0.199485
            No improvement (0.2104), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.4743, mae: 0.4762, huber: 0.1991, swd: 0.1485, ept: 103.4027
    Epoch [13/50], Val Losses: mse: 0.4840, mae: 0.4874, huber: 0.2042, swd: 0.1108, ept: 70.1107
    Epoch [13/50], Test Losses: mse: 0.5408, mae: 0.5336, huber: 0.2359, swd: 0.1219, ept: 63.5316
      Epoch 13 composite train-obj: 0.199141
            No improvement (0.2042), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.4735, mae: 0.4756, huber: 0.1987, swd: 0.1474, ept: 103.1721
    Epoch [14/50], Val Losses: mse: 0.4726, mae: 0.4722, huber: 0.1980, swd: 0.1051, ept: 81.5992
    Epoch [14/50], Test Losses: mse: 0.5584, mae: 0.5409, huber: 0.2424, swd: 0.1221, ept: 65.9379
      Epoch 14 composite train-obj: 0.198702
            No improvement (0.1980), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.4700, mae: 0.4729, huber: 0.1972, swd: 0.1466, ept: 103.9189
    Epoch [15/50], Val Losses: mse: 0.4910, mae: 0.4883, huber: 0.2073, swd: 0.1030, ept: 63.2823
    Epoch [15/50], Test Losses: mse: 0.5661, mae: 0.5505, huber: 0.2474, swd: 0.1235, ept: 56.9329
      Epoch 15 composite train-obj: 0.197210
    Epoch [15/50], Test Losses: mse: 0.5663, mae: 0.5486, huber: 0.2465, swd: 0.1243, ept: 65.2487
    Best round's Test MSE: 0.5663, MAE: 0.5486, SWD: 0.1243
    Best round's Validation MSE: 0.4661, MAE: 0.4683, SWD: 0.0965
    Best round's Test verification MSE : 0.5663, MAE: 0.5486, SWD: 0.1243
    Time taken: 13.19 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq720_pred336_20250510_1702)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5661 ± 0.0061
      mae: 0.5493 ± 0.0039
      huber: 0.2467 ± 0.0029
      swd: 0.1249 ± 0.0064
      ept: 64.1454 ± 1.3577
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4670 ± 0.0007
      mae: 0.4720 ± 0.0027
      huber: 0.1967 ± 0.0010
      swd: 0.1009 ± 0.0071
      ept: 75.7005 ± 6.3261
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 30.49 seconds
    
    Experiment complete: DLinear_etth1_seq720_pred336_20250510_1702
    Model: DLinear
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=720,
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
    Train set sample shapes: torch.Size([720, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([720, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 85
    Batch 0: Data shape torch.Size([128, 720, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 720
    Prediction Length: 720
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 85
    Validation Batches: 3
    Test Batches: 16
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6236, mae: 0.5682, huber: 0.2598, swd: 0.1961, ept: 85.0493
    Epoch [1/50], Val Losses: mse: 0.5480, mae: 0.5406, huber: 0.2363, swd: 0.1287, ept: 52.3946
    Epoch [1/50], Test Losses: mse: 0.6995, mae: 0.6364, huber: 0.3052, swd: 0.1608, ept: 87.8495
      Epoch 1 composite train-obj: 0.259783
            Val objective improved inf → 0.2363, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5663, mae: 0.5331, huber: 0.2363, swd: 0.1917, ept: 117.1208
    Epoch [2/50], Val Losses: mse: 0.5787, mae: 0.5627, huber: 0.2497, swd: 0.1421, ept: 54.2147
    Epoch [2/50], Test Losses: mse: 0.6893, mae: 0.6300, huber: 0.3000, swd: 0.1778, ept: 87.0702
      Epoch 2 composite train-obj: 0.236274
            No improvement (0.2497), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5618, mae: 0.5310, huber: 0.2345, swd: 0.1915, ept: 122.2249
    Epoch [3/50], Val Losses: mse: 0.5314, mae: 0.5135, huber: 0.2245, swd: 0.0952, ept: 61.2776
    Epoch [3/50], Test Losses: mse: 0.7158, mae: 0.6386, huber: 0.3105, swd: 0.1624, ept: 73.5270
      Epoch 3 composite train-obj: 0.234482
            Val objective improved 0.2363 → 0.2245, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5568, mae: 0.5277, huber: 0.2323, swd: 0.1903, ept: 123.8571
    Epoch [4/50], Val Losses: mse: 0.5418, mae: 0.5257, huber: 0.2304, swd: 0.1087, ept: 81.7754
    Epoch [4/50], Test Losses: mse: 0.7203, mae: 0.6404, huber: 0.3117, swd: 0.1592, ept: 90.2535
      Epoch 4 composite train-obj: 0.232321
            No improvement (0.2304), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5535, mae: 0.5251, huber: 0.2308, swd: 0.1880, ept: 128.0113
    Epoch [5/50], Val Losses: mse: 0.5320, mae: 0.5193, huber: 0.2266, swd: 0.1060, ept: 62.0262
    Epoch [5/50], Test Losses: mse: 0.7271, mae: 0.6475, huber: 0.3157, swd: 0.1677, ept: 70.0316
      Epoch 5 composite train-obj: 0.230790
            No improvement (0.2266), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5576, mae: 0.5266, huber: 0.2320, swd: 0.1927, ept: 126.3958
    Epoch [6/50], Val Losses: mse: 0.5836, mae: 0.5568, huber: 0.2485, swd: 0.1404, ept: 56.6152
    Epoch [6/50], Test Losses: mse: 0.7047, mae: 0.6373, huber: 0.3061, swd: 0.1620, ept: 85.5234
      Epoch 6 composite train-obj: 0.231996
            No improvement (0.2485), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5564, mae: 0.5261, huber: 0.2316, swd: 0.1879, ept: 127.5901
    Epoch [7/50], Val Losses: mse: 0.5854, mae: 0.5627, huber: 0.2508, swd: 0.1509, ept: 52.5869
    Epoch [7/50], Test Losses: mse: 0.7008, mae: 0.6345, huber: 0.3044, swd: 0.1769, ept: 75.6073
      Epoch 7 composite train-obj: 0.231644
            No improvement (0.2508), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.5626, mae: 0.5288, huber: 0.2338, swd: 0.1900, ept: 126.0147
    Epoch [8/50], Val Losses: mse: 0.5627, mae: 0.5518, huber: 0.2420, swd: 0.1358, ept: 56.9850
    Epoch [8/50], Test Losses: mse: 0.6697, mae: 0.6157, huber: 0.2913, swd: 0.1809, ept: 90.3848
      Epoch 8 composite train-obj: 0.233793
    Epoch [8/50], Test Losses: mse: 0.7158, mae: 0.6386, huber: 0.3105, swd: 0.1624, ept: 73.5270
    Best round's Test MSE: 0.7158, MAE: 0.6386, SWD: 0.1624
    Best round's Validation MSE: 0.5314, MAE: 0.5135, SWD: 0.0952
    Best round's Test verification MSE : 0.7158, MAE: 0.6386, SWD: 0.1624
    Time taken: 8.07 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6248, mae: 0.5691, huber: 0.2604, swd: 0.1903, ept: 83.6043
    Epoch [1/50], Val Losses: mse: 0.5411, mae: 0.5351, huber: 0.2325, swd: 0.1145, ept: 54.2133
    Epoch [1/50], Test Losses: mse: 0.6860, mae: 0.6319, huber: 0.3002, swd: 0.1674, ept: 87.6098
      Epoch 1 composite train-obj: 0.260378
            Val objective improved inf → 0.2325, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5635, mae: 0.5321, huber: 0.2353, swd: 0.1878, ept: 116.9297
    Epoch [2/50], Val Losses: mse: 0.5430, mae: 0.5285, huber: 0.2310, swd: 0.1046, ept: 62.6523
    Epoch [2/50], Test Losses: mse: 0.6915, mae: 0.6282, huber: 0.3010, swd: 0.1537, ept: 87.0206
      Epoch 2 composite train-obj: 0.235258
            Val objective improved 0.2325 → 0.2310, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5549, mae: 0.5264, huber: 0.2316, swd: 0.1852, ept: 123.6785
    Epoch [3/50], Val Losses: mse: 0.5548, mae: 0.5344, huber: 0.2351, swd: 0.0982, ept: 54.8381
    Epoch [3/50], Test Losses: mse: 0.7130, mae: 0.6387, huber: 0.3090, swd: 0.1546, ept: 70.5113
      Epoch 3 composite train-obj: 0.231639
            No improvement (0.2351), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5567, mae: 0.5276, huber: 0.2323, swd: 0.1854, ept: 123.0312
    Epoch [4/50], Val Losses: mse: 0.5327, mae: 0.5156, huber: 0.2252, swd: 0.0993, ept: 63.5736
    Epoch [4/50], Test Losses: mse: 0.6989, mae: 0.6289, huber: 0.3034, swd: 0.1530, ept: 82.6920
      Epoch 4 composite train-obj: 0.232301
            Val objective improved 0.2310 → 0.2252, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5534, mae: 0.5251, huber: 0.2308, swd: 0.1855, ept: 126.2473
    Epoch [5/50], Val Losses: mse: 0.5387, mae: 0.5292, huber: 0.2299, swd: 0.1101, ept: 65.1028
    Epoch [5/50], Test Losses: mse: 0.6936, mae: 0.6275, huber: 0.3008, swd: 0.1552, ept: 90.7786
      Epoch 5 composite train-obj: 0.230777
            No improvement (0.2299), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.5530, mae: 0.5246, huber: 0.2305, swd: 0.1851, ept: 128.0645
    Epoch [6/50], Val Losses: mse: 0.5440, mae: 0.5260, huber: 0.2307, swd: 0.1065, ept: 70.3613
    Epoch [6/50], Test Losses: mse: 0.7023, mae: 0.6303, huber: 0.3044, swd: 0.1519, ept: 86.6237
      Epoch 6 composite train-obj: 0.230512
            No improvement (0.2307), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.5573, mae: 0.5249, huber: 0.2312, swd: 0.1880, ept: 129.8624
    Epoch [7/50], Val Losses: mse: 0.5564, mae: 0.5354, huber: 0.2363, swd: 0.1035, ept: 87.2407
    Epoch [7/50], Test Losses: mse: 0.7018, mae: 0.6317, huber: 0.3040, swd: 0.1501, ept: 90.0114
      Epoch 7 composite train-obj: 0.231206
            No improvement (0.2363), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.5580, mae: 0.5266, huber: 0.2323, swd: 0.1867, ept: 128.7597
    Epoch [8/50], Val Losses: mse: 0.5650, mae: 0.5535, huber: 0.2431, swd: 0.1322, ept: 59.5608
    Epoch [8/50], Test Losses: mse: 0.7001, mae: 0.6340, huber: 0.3042, swd: 0.1687, ept: 91.9557
      Epoch 8 composite train-obj: 0.232260
            No improvement (0.2431), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.5544, mae: 0.5255, huber: 0.2311, swd: 0.1857, ept: 128.4218
    Epoch [9/50], Val Losses: mse: 0.5524, mae: 0.5419, huber: 0.2372, swd: 0.1219, ept: 52.1998
    Epoch [9/50], Test Losses: mse: 0.7012, mae: 0.6313, huber: 0.3041, swd: 0.1639, ept: 77.7112
      Epoch 9 composite train-obj: 0.231094
    Epoch [9/50], Test Losses: mse: 0.6989, mae: 0.6289, huber: 0.3034, swd: 0.1530, ept: 82.6920
    Best round's Test MSE: 0.6989, MAE: 0.6289, SWD: 0.1530
    Best round's Validation MSE: 0.5327, MAE: 0.5156, SWD: 0.0993
    Best round's Test verification MSE : 0.6989, MAE: 0.6289, SWD: 0.1530
    Time taken: 9.09 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6258, mae: 0.5689, huber: 0.2604, swd: 0.2061, ept: 84.6297
    Epoch [1/50], Val Losses: mse: 0.5541, mae: 0.5396, huber: 0.2366, swd: 0.1284, ept: 51.0296
    Epoch [1/50], Test Losses: mse: 0.6950, mae: 0.6334, huber: 0.3029, swd: 0.1693, ept: 88.9349
      Epoch 1 composite train-obj: 0.260395
            Val objective improved inf → 0.2366, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5621, mae: 0.5318, huber: 0.2349, swd: 0.1983, ept: 118.2601
    Epoch [2/50], Val Losses: mse: 0.5857, mae: 0.5654, huber: 0.2515, swd: 0.1487, ept: 53.4046
    Epoch [2/50], Test Losses: mse: 0.6871, mae: 0.6279, huber: 0.2994, swd: 0.1750, ept: 83.8449
      Epoch 2 composite train-obj: 0.234938
            No improvement (0.2515), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5563, mae: 0.5274, huber: 0.2321, swd: 0.1991, ept: 123.1870
    Epoch [3/50], Val Losses: mse: 0.5610, mae: 0.5442, huber: 0.2391, swd: 0.1303, ept: 77.1701
    Epoch [3/50], Test Losses: mse: 0.6941, mae: 0.6286, huber: 0.3013, swd: 0.1748, ept: 83.9644
      Epoch 3 composite train-obj: 0.232109
            No improvement (0.2391), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.5574, mae: 0.5273, huber: 0.2324, swd: 0.2002, ept: 126.1476
    Epoch [4/50], Val Losses: mse: 0.5613, mae: 0.5429, huber: 0.2394, swd: 0.1286, ept: 56.9193
    Epoch [4/50], Test Losses: mse: 0.6838, mae: 0.6238, huber: 0.2976, swd: 0.1717, ept: 80.2490
      Epoch 4 composite train-obj: 0.232376
            No improvement (0.2394), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.5524, mae: 0.5247, huber: 0.2305, swd: 0.1987, ept: 127.5181
    Epoch [5/50], Val Losses: mse: 0.5325, mae: 0.5228, huber: 0.2271, swd: 0.1189, ept: 67.2048
    Epoch [5/50], Test Losses: mse: 0.6951, mae: 0.6276, huber: 0.3016, swd: 0.1726, ept: 88.6591
      Epoch 5 composite train-obj: 0.230483
            Val objective improved 0.2366 → 0.2271, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5506, mae: 0.5235, huber: 0.2297, swd: 0.1980, ept: 128.5806
    Epoch [6/50], Val Losses: mse: 0.5575, mae: 0.5372, huber: 0.2370, swd: 0.1216, ept: 67.4033
    Epoch [6/50], Test Losses: mse: 0.6990, mae: 0.6317, huber: 0.3039, swd: 0.1657, ept: 83.7794
      Epoch 6 composite train-obj: 0.229746
            No improvement (0.2370), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.5518, mae: 0.5232, huber: 0.2299, swd: 0.1982, ept: 129.8068
    Epoch [7/50], Val Losses: mse: 0.5557, mae: 0.5478, huber: 0.2392, swd: 0.1546, ept: 56.0783
    Epoch [7/50], Test Losses: mse: 0.6823, mae: 0.6245, huber: 0.2973, swd: 0.1812, ept: 83.4553
      Epoch 7 composite train-obj: 0.229856
            No improvement (0.2392), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.5537, mae: 0.5241, huber: 0.2304, swd: 0.2007, ept: 129.2727
    Epoch [8/50], Val Losses: mse: 0.5398, mae: 0.5207, huber: 0.2282, swd: 0.1149, ept: 84.7780
    Epoch [8/50], Test Losses: mse: 0.7053, mae: 0.6333, huber: 0.3058, swd: 0.1649, ept: 88.6200
      Epoch 8 composite train-obj: 0.230402
            No improvement (0.2282), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.5545, mae: 0.5251, huber: 0.2309, swd: 0.1986, ept: 130.0333
    Epoch [9/50], Val Losses: mse: 0.5826, mae: 0.5645, huber: 0.2508, swd: 0.1537, ept: 54.4367
    Epoch [9/50], Test Losses: mse: 0.6997, mae: 0.6345, huber: 0.3046, swd: 0.1802, ept: 88.8861
      Epoch 9 composite train-obj: 0.230947
            No improvement (0.2508), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.5521, mae: 0.5234, huber: 0.2300, swd: 0.1960, ept: 130.4022
    Epoch [10/50], Val Losses: mse: 0.5708, mae: 0.5495, huber: 0.2436, swd: 0.1443, ept: 55.1348
    Epoch [10/50], Test Losses: mse: 0.6752, mae: 0.6197, huber: 0.2939, swd: 0.1660, ept: 83.0238
      Epoch 10 composite train-obj: 0.230015
    Epoch [10/50], Test Losses: mse: 0.6951, mae: 0.6276, huber: 0.3016, swd: 0.1726, ept: 88.6591
    Best round's Test MSE: 0.6951, MAE: 0.6276, SWD: 0.1726
    Best round's Validation MSE: 0.5325, MAE: 0.5228, SWD: 0.1189
    Best round's Test verification MSE : 0.6951, MAE: 0.6276, SWD: 0.1726
    Time taken: 9.95 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq720_pred720_20250510_1653)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.7033 ± 0.0090
      mae: 0.6317 ± 0.0049
      huber: 0.3052 ± 0.0038
      swd: 0.1627 ± 0.0080
      ept: 81.6260 ± 6.2235
      count: 3.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.5322 ± 0.0006
      mae: 0.5173 ± 0.0040
      huber: 0.2256 ± 0.0011
      swd: 0.1045 ± 0.0103
      ept: 64.0187 ± 2.4402
      count: 3.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 27.27 seconds
    
    Experiment complete: DLinear_etth1_seq720_pred720_20250510_1653
    Model: DLinear
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


