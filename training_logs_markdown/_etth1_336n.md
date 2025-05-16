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
    




    <data_manager.DatasetManager at 0x23142d2a300>



## Seq=336

### EigenACL

#### pred=96


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 96
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
    
    Epoch [1/50], Train Losses: mse: 0.5408, mae: 0.5184, huber: 0.2256, swd: 0.2976, ept: 38.6169
    Epoch [1/50], Val Losses: mse: 0.4009, mae: 0.4509, huber: 0.1755, swd: 0.1627, ept: 43.6403
    Epoch [1/50], Test Losses: mse: 0.4809, mae: 0.4927, huber: 0.2084, swd: 0.1881, ept: 32.1500
      Epoch 1 composite train-obj: 0.225646
            Val objective improved inf → 0.1755, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3576, mae: 0.4198, huber: 0.1580, swd: 0.1264, ept: 48.2834
    Epoch [2/50], Val Losses: mse: 0.3793, mae: 0.4356, huber: 0.1671, swd: 0.1310, ept: 45.3988
    Epoch [2/50], Test Losses: mse: 0.4349, mae: 0.4660, huber: 0.1911, swd: 0.1603, ept: 34.7919
      Epoch 2 composite train-obj: 0.158017
            Val objective improved 0.1755 → 0.1671, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3259, mae: 0.4003, huber: 0.1460, swd: 0.1119, ept: 50.8636
    Epoch [3/50], Val Losses: mse: 0.3719, mae: 0.4170, huber: 0.1624, swd: 0.1076, ept: 46.1254
    Epoch [3/50], Test Losses: mse: 0.4209, mae: 0.4536, huber: 0.1842, swd: 0.1412, ept: 37.6081
      Epoch 3 composite train-obj: 0.145984
            Val objective improved 0.1671 → 0.1624, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3053, mae: 0.3874, huber: 0.1378, swd: 0.0989, ept: 52.4528
    Epoch [4/50], Val Losses: mse: 0.3963, mae: 0.4318, huber: 0.1714, swd: 0.1263, ept: 46.8152
    Epoch [4/50], Test Losses: mse: 0.4311, mae: 0.4607, huber: 0.1881, swd: 0.1508, ept: 37.9071
      Epoch 4 composite train-obj: 0.137848
            No improvement (0.1714), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.2859, mae: 0.3753, huber: 0.1301, swd: 0.0876, ept: 53.6999
    Epoch [5/50], Val Losses: mse: 0.3883, mae: 0.4354, huber: 0.1697, swd: 0.1263, ept: 47.0539
    Epoch [5/50], Test Losses: mse: 0.4202, mae: 0.4538, huber: 0.1838, swd: 0.1458, ept: 38.3836
      Epoch 5 composite train-obj: 0.130117
            No improvement (0.1697), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.2708, mae: 0.3658, huber: 0.1240, swd: 0.0794, ept: 54.5524
    Epoch [6/50], Val Losses: mse: 0.3962, mae: 0.4345, huber: 0.1722, swd: 0.1178, ept: 46.9238
    Epoch [6/50], Test Losses: mse: 0.4222, mae: 0.4556, huber: 0.1846, swd: 0.1448, ept: 38.7393
      Epoch 6 composite train-obj: 0.124033
            No improvement (0.1722), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.2584, mae: 0.3574, huber: 0.1188, swd: 0.0728, ept: 55.3319
    Epoch [7/50], Val Losses: mse: 0.4185, mae: 0.4580, huber: 0.1829, swd: 0.1359, ept: 46.4441
    Epoch [7/50], Test Losses: mse: 0.4344, mae: 0.4636, huber: 0.1899, swd: 0.1679, ept: 37.8939
      Epoch 7 composite train-obj: 0.118798
            No improvement (0.1829), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.2479, mae: 0.3501, huber: 0.1144, swd: 0.0672, ept: 56.0942
    Epoch [8/50], Val Losses: mse: 0.4392, mae: 0.4714, huber: 0.1909, swd: 0.1300, ept: 45.6448
    Epoch [8/50], Test Losses: mse: 0.4310, mae: 0.4612, huber: 0.1878, swd: 0.1466, ept: 38.3291
      Epoch 8 composite train-obj: 0.114374
    Epoch [8/50], Test Losses: mse: 0.4209, mae: 0.4536, huber: 0.1842, swd: 0.1412, ept: 37.6060
    Best round's Test MSE: 0.4209, MAE: 0.4536, SWD: 0.1412
    Best round's Validation MSE: 0.3719, MAE: 0.4170, SWD: 0.1076
    Best round's Test verification MSE : 0.4209, MAE: 0.4536, SWD: 0.1412
    Time taken: 20.15 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5184, mae: 0.5075, huber: 0.2181, swd: 0.2685, ept: 39.5086
    Epoch [1/50], Val Losses: mse: 0.3896, mae: 0.4383, huber: 0.1703, swd: 0.1350, ept: 43.9169
    Epoch [1/50], Test Losses: mse: 0.4681, mae: 0.4841, huber: 0.2031, swd: 0.1593, ept: 33.4680
      Epoch 1 composite train-obj: 0.218077
            Val objective improved inf → 0.1703, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3539, mae: 0.4174, huber: 0.1566, swd: 0.1227, ept: 48.7088
    Epoch [2/50], Val Losses: mse: 0.3729, mae: 0.4168, huber: 0.1624, swd: 0.1141, ept: 46.0944
    Epoch [2/50], Test Losses: mse: 0.4324, mae: 0.4621, huber: 0.1894, swd: 0.1429, ept: 36.2807
      Epoch 2 composite train-obj: 0.156619
            Val objective improved 0.1703 → 0.1624, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3227, mae: 0.3988, huber: 0.1450, swd: 0.1080, ept: 51.0711
    Epoch [3/50], Val Losses: mse: 0.3845, mae: 0.4254, huber: 0.1672, swd: 0.1243, ept: 46.7577
    Epoch [3/50], Test Losses: mse: 0.4320, mae: 0.4617, huber: 0.1891, swd: 0.1518, ept: 36.8402
      Epoch 3 composite train-obj: 0.144974
            No improvement (0.1672), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3023, mae: 0.3870, huber: 0.1371, swd: 0.0968, ept: 52.1820
    Epoch [4/50], Val Losses: mse: 0.4034, mae: 0.4391, huber: 0.1746, swd: 0.1356, ept: 46.5983
    Epoch [4/50], Test Losses: mse: 0.4306, mae: 0.4630, huber: 0.1888, swd: 0.1567, ept: 36.9879
      Epoch 4 composite train-obj: 0.137091
            No improvement (0.1746), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.2833, mae: 0.3755, huber: 0.1295, swd: 0.0853, ept: 53.3392
    Epoch [5/50], Val Losses: mse: 0.4227, mae: 0.4665, huber: 0.1852, swd: 0.1535, ept: 46.1137
    Epoch [5/50], Test Losses: mse: 0.4256, mae: 0.4599, huber: 0.1868, swd: 0.1603, ept: 37.6016
      Epoch 5 composite train-obj: 0.129535
            No improvement (0.1852), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.2689, mae: 0.3662, huber: 0.1236, swd: 0.0772, ept: 54.1919
    Epoch [6/50], Val Losses: mse: 0.4086, mae: 0.4490, huber: 0.1778, swd: 0.1308, ept: 46.1998
    Epoch [6/50], Test Losses: mse: 0.4273, mae: 0.4606, huber: 0.1872, swd: 0.1547, ept: 37.7332
      Epoch 6 composite train-obj: 0.123579
            No improvement (0.1778), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.2574, mae: 0.3582, huber: 0.1187, swd: 0.0710, ept: 54.9045
    Epoch [7/50], Val Losses: mse: 0.4373, mae: 0.4628, huber: 0.1878, swd: 0.1345, ept: 45.6676
    Epoch [7/50], Test Losses: mse: 0.4386, mae: 0.4695, huber: 0.1919, swd: 0.1542, ept: 37.8185
      Epoch 7 composite train-obj: 0.118743
    Epoch [7/50], Test Losses: mse: 0.4324, mae: 0.4621, huber: 0.1894, swd: 0.1429, ept: 36.2893
    Best round's Test MSE: 0.4324, MAE: 0.4621, SWD: 0.1429
    Best round's Validation MSE: 0.3729, MAE: 0.4168, SWD: 0.1141
    Best round's Test verification MSE : 0.4324, MAE: 0.4621, SWD: 0.1429
    Time taken: 17.67 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5342, mae: 0.5164, huber: 0.2239, swd: 0.2798, ept: 38.7292
    Epoch [1/50], Val Losses: mse: 0.3882, mae: 0.4394, huber: 0.1703, swd: 0.1390, ept: 43.6615
    Epoch [1/50], Test Losses: mse: 0.4661, mae: 0.4846, huber: 0.2030, swd: 0.1618, ept: 33.3799
      Epoch 1 composite train-obj: 0.223891
            Val objective improved inf → 0.1703, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3570, mae: 0.4190, huber: 0.1578, swd: 0.1208, ept: 48.7909
    Epoch [2/50], Val Losses: mse: 0.3729, mae: 0.4214, huber: 0.1632, swd: 0.1231, ept: 45.9988
    Epoch [2/50], Test Losses: mse: 0.4358, mae: 0.4658, huber: 0.1913, swd: 0.1517, ept: 35.7887
      Epoch 2 composite train-obj: 0.157786
            Val objective improved 0.1703 → 0.1632, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3286, mae: 0.4021, huber: 0.1471, swd: 0.1062, ept: 50.7201
    Epoch [3/50], Val Losses: mse: 0.3902, mae: 0.4292, huber: 0.1693, swd: 0.1215, ept: 46.3782
    Epoch [3/50], Test Losses: mse: 0.4308, mae: 0.4610, huber: 0.1886, swd: 0.1421, ept: 37.0590
      Epoch 3 composite train-obj: 0.147109
            No improvement (0.1693), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3070, mae: 0.3898, huber: 0.1388, swd: 0.0953, ept: 51.9358
    Epoch [4/50], Val Losses: mse: 0.4060, mae: 0.4357, huber: 0.1747, swd: 0.1186, ept: 46.5414
    Epoch [4/50], Test Losses: mse: 0.4297, mae: 0.4609, huber: 0.1879, swd: 0.1352, ept: 38.1371
      Epoch 4 composite train-obj: 0.138806
            No improvement (0.1747), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.2888, mae: 0.3787, huber: 0.1315, swd: 0.0854, ept: 53.0072
    Epoch [5/50], Val Losses: mse: 0.4241, mae: 0.4629, huber: 0.1844, swd: 0.1406, ept: 46.5177
    Epoch [5/50], Test Losses: mse: 0.4290, mae: 0.4615, huber: 0.1882, swd: 0.1486, ept: 37.7856
      Epoch 5 composite train-obj: 0.131533
            No improvement (0.1844), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.2738, mae: 0.3691, huber: 0.1255, swd: 0.0777, ept: 53.9227
    Epoch [6/50], Val Losses: mse: 0.4258, mae: 0.4600, huber: 0.1850, swd: 0.1421, ept: 46.4305
    Epoch [6/50], Test Losses: mse: 0.4396, mae: 0.4689, huber: 0.1926, swd: 0.1628, ept: 36.7709
      Epoch 6 composite train-obj: 0.125485
            No improvement (0.1850), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.2594, mae: 0.3598, huber: 0.1196, swd: 0.0705, ept: 54.7639
    Epoch [7/50], Val Losses: mse: 0.4568, mae: 0.4815, huber: 0.1968, swd: 0.1507, ept: 46.5358
    Epoch [7/50], Test Losses: mse: 0.4471, mae: 0.4768, huber: 0.1965, swd: 0.1567, ept: 36.7886
      Epoch 7 composite train-obj: 0.119571
    Epoch [7/50], Test Losses: mse: 0.4358, mae: 0.4658, huber: 0.1913, swd: 0.1517, ept: 35.7904
    Best round's Test MSE: 0.4358, MAE: 0.4658, SWD: 0.1517
    Best round's Validation MSE: 0.3729, MAE: 0.4214, SWD: 0.1231
    Best round's Test verification MSE : 0.4358, MAE: 0.4658, SWD: 0.1517
    Time taken: 17.65 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq336_pred96_20250510_1914)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4297 ± 0.0064
      mae: 0.4605 ± 0.0051
      huber: 0.1883 ± 0.0030
      swd: 0.1453 ± 0.0046
      ept: 36.5592 ± 0.7684
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3726 ± 0.0005
      mae: 0.4184 ± 0.0021
      huber: 0.1627 ± 0.0004
      swd: 0.1149 ± 0.0064
      ept: 46.0729 ± 0.0539
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 56.84 seconds
    
    Experiment complete: ACL_etth1_seq336_pred96_20250510_1914
    Model: ACL
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.6117, mae: 0.5526, huber: 0.2508, swd: 0.3516, ept: 55.7291
    Epoch [1/50], Val Losses: mse: 0.4447, mae: 0.4715, huber: 0.1929, swd: 0.1524, ept: 57.5238
    Epoch [1/50], Test Losses: mse: 0.5265, mae: 0.5242, huber: 0.2281, swd: 0.1774, ept: 48.8603
      Epoch 1 composite train-obj: 0.250787
            Val objective improved inf → 0.1929, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4347, mae: 0.4641, huber: 0.1878, swd: 0.1652, ept: 70.2807
    Epoch [2/50], Val Losses: mse: 0.4376, mae: 0.4596, huber: 0.1893, swd: 0.1365, ept: 60.6396
    Epoch [2/50], Test Losses: mse: 0.4818, mae: 0.5023, huber: 0.2122, swd: 0.1631, ept: 51.3339
      Epoch 2 composite train-obj: 0.187808
            Val objective improved 0.1929 → 0.1893, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3907, mae: 0.4402, huber: 0.1719, swd: 0.1372, ept: 73.9238
    Epoch [3/50], Val Losses: mse: 0.4503, mae: 0.4682, huber: 0.1947, swd: 0.1377, ept: 61.3544
    Epoch [3/50], Test Losses: mse: 0.4657, mae: 0.4932, huber: 0.2060, swd: 0.1622, ept: 52.9833
      Epoch 3 composite train-obj: 0.171946
            No improvement (0.1947), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3591, mae: 0.4241, huber: 0.1605, swd: 0.1183, ept: 75.8730
    Epoch [4/50], Val Losses: mse: 0.4651, mae: 0.4799, huber: 0.2005, swd: 0.1323, ept: 59.6988
    Epoch [4/50], Test Losses: mse: 0.4556, mae: 0.4873, huber: 0.2019, swd: 0.1443, ept: 53.2398
      Epoch 4 composite train-obj: 0.160524
            No improvement (0.2005), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3335, mae: 0.4096, huber: 0.1505, swd: 0.1018, ept: 77.5960
    Epoch [5/50], Val Losses: mse: 0.5112, mae: 0.5096, huber: 0.2187, swd: 0.1502, ept: 59.6891
    Epoch [5/50], Test Losses: mse: 0.4668, mae: 0.4967, huber: 0.2071, swd: 0.1624, ept: 50.8994
      Epoch 5 composite train-obj: 0.150547
            No improvement (0.2187), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3169, mae: 0.3990, huber: 0.1437, swd: 0.0924, ept: 79.0389
    Epoch [6/50], Val Losses: mse: 0.4805, mae: 0.4992, huber: 0.2094, swd: 0.1550, ept: 60.8928
    Epoch [6/50], Test Losses: mse: 0.4811, mae: 0.5039, huber: 0.2124, swd: 0.1739, ept: 50.7380
      Epoch 6 composite train-obj: 0.143696
            No improvement (0.2094), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3035, mae: 0.3903, huber: 0.1381, swd: 0.0842, ept: 80.8125
    Epoch [7/50], Val Losses: mse: 0.5133, mae: 0.5144, huber: 0.2203, swd: 0.1424, ept: 60.7095
    Epoch [7/50], Test Losses: mse: 0.4626, mae: 0.4914, huber: 0.2045, swd: 0.1643, ept: 53.8004
      Epoch 7 composite train-obj: 0.138102
    Epoch [7/50], Test Losses: mse: 0.4818, mae: 0.5023, huber: 0.2122, swd: 0.1631, ept: 51.3439
    Best round's Test MSE: 0.4818, MAE: 0.5023, SWD: 0.1631
    Best round's Validation MSE: 0.4376, MAE: 0.4596, SWD: 0.1365
    Best round's Test verification MSE : 0.4818, MAE: 0.5023, SWD: 0.1631
    Time taken: 22.16 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6111, mae: 0.5524, huber: 0.2506, swd: 0.3475, ept: 55.8293
    Epoch [1/50], Val Losses: mse: 0.4627, mae: 0.4844, huber: 0.2014, swd: 0.1865, ept: 58.7105
    Epoch [1/50], Test Losses: mse: 0.5249, mae: 0.5244, huber: 0.2284, swd: 0.2009, ept: 48.6173
      Epoch 1 composite train-obj: 0.250637
            Val objective improved inf → 0.2014, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4278, mae: 0.4593, huber: 0.1849, swd: 0.1628, ept: 71.0461
    Epoch [2/50], Val Losses: mse: 0.4370, mae: 0.4620, huber: 0.1897, swd: 0.1090, ept: 57.2757
    Epoch [2/50], Test Losses: mse: 0.4737, mae: 0.4957, huber: 0.2085, swd: 0.1268, ept: 52.3234
      Epoch 2 composite train-obj: 0.184922
            Val objective improved 0.2014 → 0.1897, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3866, mae: 0.4372, huber: 0.1703, swd: 0.1350, ept: 74.5047
    Epoch [3/50], Val Losses: mse: 0.4420, mae: 0.4662, huber: 0.1917, swd: 0.1257, ept: 60.3170
    Epoch [3/50], Test Losses: mse: 0.4565, mae: 0.4877, huber: 0.2022, swd: 0.1337, ept: 53.5509
      Epoch 3 composite train-obj: 0.170304
            No improvement (0.1917), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3553, mae: 0.4208, huber: 0.1588, swd: 0.1153, ept: 77.0638
    Epoch [4/50], Val Losses: mse: 0.4703, mae: 0.4858, huber: 0.2034, swd: 0.1446, ept: 61.1215
    Epoch [4/50], Test Losses: mse: 0.4545, mae: 0.4868, huber: 0.2015, swd: 0.1537, ept: 54.0779
      Epoch 4 composite train-obj: 0.158831
            No improvement (0.2034), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3314, mae: 0.4082, huber: 0.1497, swd: 0.1001, ept: 78.7495
    Epoch [5/50], Val Losses: mse: 0.5013, mae: 0.5035, huber: 0.2151, swd: 0.1528, ept: 60.7384
    Epoch [5/50], Test Losses: mse: 0.4607, mae: 0.4911, huber: 0.2039, swd: 0.1544, ept: 53.8252
      Epoch 5 composite train-obj: 0.149708
            No improvement (0.2151), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3124, mae: 0.3962, huber: 0.1419, swd: 0.0878, ept: 80.5674
    Epoch [6/50], Val Losses: mse: 0.5368, mae: 0.5205, huber: 0.2278, swd: 0.1575, ept: 59.8659
    Epoch [6/50], Test Losses: mse: 0.4751, mae: 0.5016, huber: 0.2105, swd: 0.1741, ept: 52.8306
      Epoch 6 composite train-obj: 0.141852
            No improvement (0.2278), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.2981, mae: 0.3874, huber: 0.1362, swd: 0.0807, ept: 81.9943
    Epoch [7/50], Val Losses: mse: 0.5085, mae: 0.5032, huber: 0.2166, swd: 0.1460, ept: 61.4786
    Epoch [7/50], Test Losses: mse: 0.4620, mae: 0.4912, huber: 0.2044, swd: 0.1507, ept: 53.9510
      Epoch 7 composite train-obj: 0.136153
    Epoch [7/50], Test Losses: mse: 0.4737, mae: 0.4957, huber: 0.2085, swd: 0.1268, ept: 52.3260
    Best round's Test MSE: 0.4737, MAE: 0.4957, SWD: 0.1268
    Best round's Validation MSE: 0.4370, MAE: 0.4620, SWD: 0.1090
    Best round's Test verification MSE : 0.4737, MAE: 0.4957, SWD: 0.1268
    Time taken: 21.85 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6008, mae: 0.5474, huber: 0.2470, swd: 0.3158, ept: 56.1582
    Epoch [1/50], Val Losses: mse: 0.4375, mae: 0.4672, huber: 0.1898, swd: 0.1280, ept: 56.2392
    Epoch [1/50], Test Losses: mse: 0.5153, mae: 0.5178, huber: 0.2236, swd: 0.1520, ept: 47.3955
      Epoch 1 composite train-obj: 0.247043
            Val objective improved inf → 0.1898, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4295, mae: 0.4598, huber: 0.1855, swd: 0.1502, ept: 70.6610
    Epoch [2/50], Val Losses: mse: 0.4249, mae: 0.4508, huber: 0.1840, swd: 0.1051, ept: 59.5935
    Epoch [2/50], Test Losses: mse: 0.4754, mae: 0.4996, huber: 0.2102, swd: 0.1336, ept: 51.0756
      Epoch 2 composite train-obj: 0.185463
            Val objective improved 0.1898 → 0.1840, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3872, mae: 0.4373, huber: 0.1705, swd: 0.1276, ept: 74.2826
    Epoch [3/50], Val Losses: mse: 0.4683, mae: 0.4896, huber: 0.2041, swd: 0.1739, ept: 59.3981
    Epoch [3/50], Test Losses: mse: 0.4895, mae: 0.5097, huber: 0.2169, swd: 0.2109, ept: 49.6869
      Epoch 3 composite train-obj: 0.170491
            No improvement (0.2041), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3609, mae: 0.4234, huber: 0.1608, swd: 0.1141, ept: 76.2755
    Epoch [4/50], Val Losses: mse: 0.5126, mae: 0.5166, huber: 0.2215, swd: 0.1678, ept: 56.0562
    Epoch [4/50], Test Losses: mse: 0.4620, mae: 0.4952, huber: 0.2055, swd: 0.1472, ept: 50.0093
      Epoch 4 composite train-obj: 0.160792
            No improvement (0.2215), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3357, mae: 0.4104, huber: 0.1514, swd: 0.0971, ept: 77.5007
    Epoch [5/50], Val Losses: mse: 0.4874, mae: 0.4948, huber: 0.2093, swd: 0.1408, ept: 60.2981
    Epoch [5/50], Test Losses: mse: 0.4555, mae: 0.4879, huber: 0.2024, swd: 0.1411, ept: 52.5298
      Epoch 5 composite train-obj: 0.151360
            No improvement (0.2093), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3166, mae: 0.3990, huber: 0.1437, swd: 0.0874, ept: 79.3192
    Epoch [6/50], Val Losses: mse: 0.5273, mae: 0.5046, huber: 0.2201, swd: 0.1353, ept: 58.9970
    Epoch [6/50], Test Losses: mse: 0.4671, mae: 0.4967, huber: 0.2070, swd: 0.1438, ept: 52.8042
      Epoch 6 composite train-obj: 0.143688
            No improvement (0.2201), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3017, mae: 0.3890, huber: 0.1374, swd: 0.0795, ept: 81.2612
    Epoch [7/50], Val Losses: mse: 0.5284, mae: 0.5187, huber: 0.2242, swd: 0.1410, ept: 59.0273
    Epoch [7/50], Test Losses: mse: 0.4573, mae: 0.4905, huber: 0.2032, swd: 0.1439, ept: 53.6386
      Epoch 7 composite train-obj: 0.137407
    Epoch [7/50], Test Losses: mse: 0.4754, mae: 0.4996, huber: 0.2102, swd: 0.1336, ept: 51.0370
    Best round's Test MSE: 0.4754, MAE: 0.4996, SWD: 0.1336
    Best round's Validation MSE: 0.4249, MAE: 0.4508, SWD: 0.1051
    Best round's Test verification MSE : 0.4754, MAE: 0.4996, SWD: 0.1336
    Time taken: 21.05 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq336_pred196_20250510_1935)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4770 ± 0.0035
      mae: 0.4992 ± 0.0027
      huber: 0.2103 ± 0.0015
      swd: 0.1412 ± 0.0157
      ept: 51.5776 ± 0.5378
      count: 10.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4331 ± 0.0059
      mae: 0.4575 ± 0.0048
      huber: 0.1877 ± 0.0026
      swd: 0.1169 ± 0.0140
      ept: 59.1696 ± 1.4057
      count: 10.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 65.17 seconds
    
    Experiment complete: ACL_etth1_seq336_pred196_20250510_1935
    Model: ACL
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 91
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 91
    Validation Batches: 9
    Test Batches: 22
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6499, mae: 0.5721, huber: 0.2651, swd: 0.3545, ept: 70.3014
    Epoch [1/50], Val Losses: mse: 0.4650, mae: 0.4833, huber: 0.2021, swd: 0.1812, ept: 74.8638
    Epoch [1/50], Test Losses: mse: 0.5768, mae: 0.5563, huber: 0.2503, swd: 0.2274, ept: 66.9368
      Epoch 1 composite train-obj: 0.265122
            Val objective improved inf → 0.2021, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4869, mae: 0.4909, huber: 0.2070, swd: 0.1862, ept: 89.7887
    Epoch [2/50], Val Losses: mse: 0.4694, mae: 0.4879, huber: 0.2033, swd: 0.1463, ept: 71.8225
    Epoch [2/50], Test Losses: mse: 0.5268, mae: 0.5335, huber: 0.2330, swd: 0.1742, ept: 66.2509
      Epoch 2 composite train-obj: 0.207047
            No improvement (0.2033), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4487, mae: 0.4694, huber: 0.1929, swd: 0.1587, ept: 94.7318
    Epoch [3/50], Val Losses: mse: 0.4429, mae: 0.4568, huber: 0.1888, swd: 0.1053, ept: 79.9820
    Epoch [3/50], Test Losses: mse: 0.5131, mae: 0.5261, huber: 0.2266, swd: 0.1492, ept: 66.6436
      Epoch 3 composite train-obj: 0.192860
            Val objective improved 0.2021 → 0.1888, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4192, mae: 0.4546, huber: 0.1823, swd: 0.1398, ept: 96.7193
    Epoch [4/50], Val Losses: mse: 0.5479, mae: 0.5544, huber: 0.2418, swd: 0.1953, ept: 51.6273
    Epoch [4/50], Test Losses: mse: 0.5288, mae: 0.5406, huber: 0.2346, swd: 0.1696, ept: 59.4027
      Epoch 4 composite train-obj: 0.182281
            No improvement (0.2418), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3954, mae: 0.4432, huber: 0.1739, swd: 0.1228, ept: 98.5794
    Epoch [5/50], Val Losses: mse: 0.5426, mae: 0.5503, huber: 0.2378, swd: 0.1555, ept: 64.2510
    Epoch [5/50], Test Losses: mse: 0.5267, mae: 0.5408, huber: 0.2347, swd: 0.1497, ept: 62.2361
      Epoch 5 composite train-obj: 0.173905
            No improvement (0.2378), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3670, mae: 0.4278, huber: 0.1631, swd: 0.1060, ept: 101.3943
    Epoch [6/50], Val Losses: mse: 0.5827, mae: 0.5634, huber: 0.2501, swd: 0.1711, ept: 68.1581
    Epoch [6/50], Test Losses: mse: 0.5336, mae: 0.5456, huber: 0.2373, swd: 0.1679, ept: 64.1256
      Epoch 6 composite train-obj: 0.163082
            No improvement (0.2501), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3518, mae: 0.4193, huber: 0.1572, swd: 0.0972, ept: 102.3567
    Epoch [7/50], Val Losses: mse: 0.5619, mae: 0.5603, huber: 0.2444, swd: 0.1501, ept: 68.7065
    Epoch [7/50], Test Losses: mse: 0.5283, mae: 0.5420, huber: 0.2352, swd: 0.1800, ept: 65.8170
      Epoch 7 composite train-obj: 0.157218
            No improvement (0.2444), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3423, mae: 0.4140, huber: 0.1537, swd: 0.0913, ept: 103.2580
    Epoch [8/50], Val Losses: mse: 0.5804, mae: 0.5626, huber: 0.2483, swd: 0.1427, ept: 73.2139
    Epoch [8/50], Test Losses: mse: 0.5502, mae: 0.5502, huber: 0.2421, swd: 0.1605, ept: 65.2283
      Epoch 8 composite train-obj: 0.153671
    Epoch [8/50], Test Losses: mse: 0.5131, mae: 0.5261, huber: 0.2266, swd: 0.1492, ept: 66.5487
    Best round's Test MSE: 0.5131, MAE: 0.5261, SWD: 0.1492
    Best round's Validation MSE: 0.4429, MAE: 0.4568, SWD: 0.1053
    Best round's Test verification MSE : 0.5131, MAE: 0.5261, SWD: 0.1492
    Time taken: 24.10 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6468, mae: 0.5692, huber: 0.2637, swd: 0.3512, ept: 71.0857
    Epoch [1/50], Val Losses: mse: 0.5431, mae: 0.5467, huber: 0.2383, swd: 0.2504, ept: 49.9516
    Epoch [1/50], Test Losses: mse: 0.5763, mae: 0.5646, huber: 0.2526, swd: 0.2233, ept: 52.1638
      Epoch 1 composite train-obj: 0.263669
            Val objective improved inf → 0.2383, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4842, mae: 0.4896, huber: 0.2062, swd: 0.1860, ept: 91.2498
    Epoch [2/50], Val Losses: mse: 0.5112, mae: 0.5154, huber: 0.2209, swd: 0.1793, ept: 70.2583
    Epoch [2/50], Test Losses: mse: 0.5560, mae: 0.5537, huber: 0.2456, swd: 0.1966, ept: 64.0936
      Epoch 2 composite train-obj: 0.206246
            Val objective improved 0.2383 → 0.2209, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4431, mae: 0.4671, huber: 0.1911, swd: 0.1578, ept: 94.5159
    Epoch [3/50], Val Losses: mse: 0.4748, mae: 0.4897, huber: 0.2048, swd: 0.1206, ept: 71.1522
    Epoch [3/50], Test Losses: mse: 0.4994, mae: 0.5146, huber: 0.2207, swd: 0.1328, ept: 69.0639
      Epoch 3 composite train-obj: 0.191062
            Val objective improved 0.2209 → 0.2048, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4175, mae: 0.4543, huber: 0.1818, swd: 0.1407, ept: 96.0960
    Epoch [4/50], Val Losses: mse: 0.5521, mae: 0.5443, huber: 0.2373, swd: 0.1401, ept: 64.5972
    Epoch [4/50], Test Losses: mse: 0.5056, mae: 0.5205, huber: 0.2236, swd: 0.1184, ept: 66.9030
      Epoch 4 composite train-obj: 0.181811
            No improvement (0.2373), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3987, mae: 0.4457, huber: 0.1754, swd: 0.1252, ept: 96.9178
    Epoch [5/50], Val Losses: mse: 0.5422, mae: 0.5298, huber: 0.2307, swd: 0.1599, ept: 75.6443
    Epoch [5/50], Test Losses: mse: 0.5266, mae: 0.5406, huber: 0.2340, swd: 0.1800, ept: 64.5927
      Epoch 5 composite train-obj: 0.175382
            No improvement (0.2307), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3777, mae: 0.4335, huber: 0.1671, swd: 0.1104, ept: 99.1170
    Epoch [6/50], Val Losses: mse: 0.5696, mae: 0.5620, huber: 0.2468, swd: 0.1829, ept: 72.1293
    Epoch [6/50], Test Losses: mse: 0.5555, mae: 0.5572, huber: 0.2462, swd: 0.2159, ept: 63.9743
      Epoch 6 composite train-obj: 0.167076
            No improvement (0.2468), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3568, mae: 0.4215, huber: 0.1589, swd: 0.0994, ept: 102.2092
    Epoch [7/50], Val Losses: mse: 0.7295, mae: 0.6334, huber: 0.3094, swd: 0.2544, ept: 47.5979
    Epoch [7/50], Test Losses: mse: 0.5615, mae: 0.5608, huber: 0.2496, swd: 0.1857, ept: 60.4485
      Epoch 7 composite train-obj: 0.158860
            No improvement (0.3094), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3656, mae: 0.4282, huber: 0.1632, swd: 0.1024, ept: 99.8060
    Epoch [8/50], Val Losses: mse: 0.5934, mae: 0.5614, huber: 0.2495, swd: 0.1468, ept: 71.1632
    Epoch [8/50], Test Losses: mse: 0.5582, mae: 0.5508, huber: 0.2445, swd: 0.1220, ept: 64.1757
      Epoch 8 composite train-obj: 0.163200
    Epoch [8/50], Test Losses: mse: 0.4994, mae: 0.5146, huber: 0.2207, swd: 0.1328, ept: 69.1080
    Best round's Test MSE: 0.4994, MAE: 0.5146, SWD: 0.1328
    Best round's Validation MSE: 0.4748, MAE: 0.4897, SWD: 0.1206
    Best round's Test verification MSE : 0.4994, MAE: 0.5146, SWD: 0.1328
    Time taken: 24.36 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6600, mae: 0.5757, huber: 0.2683, swd: 0.3734, ept: 69.8960
    Epoch [1/50], Val Losses: mse: 0.4554, mae: 0.4762, huber: 0.1977, swd: 0.1326, ept: 71.4497
    Epoch [1/50], Test Losses: mse: 0.5745, mae: 0.5550, huber: 0.2492, swd: 0.1680, ept: 64.2585
      Epoch 1 composite train-obj: 0.268275
            Val objective improved inf → 0.1977, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4881, mae: 0.4918, huber: 0.2077, swd: 0.1867, ept: 88.6452
    Epoch [2/50], Val Losses: mse: 0.4713, mae: 0.4948, huber: 0.2059, swd: 0.1791, ept: 65.9007
    Epoch [2/50], Test Losses: mse: 0.5385, mae: 0.5417, huber: 0.2381, swd: 0.2113, ept: 59.9893
      Epoch 2 composite train-obj: 0.207695
            No improvement (0.2059), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4519, mae: 0.4715, huber: 0.1944, swd: 0.1626, ept: 92.9156
    Epoch [3/50], Val Losses: mse: 0.4981, mae: 0.5231, huber: 0.2204, swd: 0.2020, ept: 60.2918
    Epoch [3/50], Test Losses: mse: 0.5380, mae: 0.5429, huber: 0.2380, swd: 0.2339, ept: 59.0521
      Epoch 3 composite train-obj: 0.194382
            No improvement (0.2204), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4188, mae: 0.4559, huber: 0.1827, swd: 0.1403, ept: 93.5175
    Epoch [4/50], Val Losses: mse: 0.5198, mae: 0.5298, huber: 0.2262, swd: 0.1607, ept: 70.8504
    Epoch [4/50], Test Losses: mse: 0.5208, mae: 0.5368, huber: 0.2320, swd: 0.1911, ept: 65.4642
      Epoch 4 composite train-obj: 0.182674
            No improvement (0.2262), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3889, mae: 0.4405, huber: 0.1715, swd: 0.1214, ept: 96.0028
    Epoch [5/50], Val Losses: mse: 0.5603, mae: 0.5472, huber: 0.2401, swd: 0.1583, ept: 70.5514
    Epoch [5/50], Test Losses: mse: 0.5319, mae: 0.5445, huber: 0.2367, swd: 0.1923, ept: 65.1643
      Epoch 5 composite train-obj: 0.171548
            No improvement (0.2401), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3652, mae: 0.4262, huber: 0.1621, swd: 0.1054, ept: 98.8781
    Epoch [6/50], Val Losses: mse: 0.5322, mae: 0.5428, huber: 0.2328, swd: 0.1424, ept: 69.8968
    Epoch [6/50], Test Losses: mse: 0.5180, mae: 0.5330, huber: 0.2302, swd: 0.1797, ept: 65.8261
      Epoch 6 composite train-obj: 0.162092
    Epoch [6/50], Test Losses: mse: 0.5746, mae: 0.5551, huber: 0.2492, swd: 0.1680, ept: 64.2665
    Best round's Test MSE: 0.5745, MAE: 0.5550, SWD: 0.1680
    Best round's Validation MSE: 0.4554, MAE: 0.4762, SWD: 0.1326
    Best round's Test verification MSE : 0.5746, MAE: 0.5551, SWD: 0.1680
    Time taken: 18.51 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq336_pred336_20250510_1934)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5290 ± 0.0327
      mae: 0.5319 ± 0.0170
      huber: 0.2321 ± 0.0123
      swd: 0.1500 ± 0.0144
      ept: 66.6553 ± 1.9618
      count: 9.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4577 ± 0.0131
      mae: 0.4742 ± 0.0135
      huber: 0.1971 ± 0.0065
      swd: 0.1195 ± 0.0111
      ept: 74.1947 ± 4.0941
      count: 9.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 67.09 seconds
    
    Experiment complete: ACL_etth1_seq336_pred336_20250510_1934
    Model: ACL
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 88
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 720
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
    
    Epoch [1/50], Train Losses: mse: 0.7076, mae: 0.6038, huber: 0.2879, swd: 0.3994, ept: 86.2303
    Epoch [1/50], Val Losses: mse: 0.5018, mae: 0.5155, huber: 0.2213, swd: 0.1641, ept: 80.1928
    Epoch [1/50], Test Losses: mse: 0.6806, mae: 0.6187, huber: 0.2944, swd: 0.2239, ept: 83.0114
      Epoch 1 composite train-obj: 0.287872
            Val objective improved inf → 0.2213, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5631, mae: 0.5335, huber: 0.2364, swd: 0.2352, ept: 112.9617
    Epoch [2/50], Val Losses: mse: 0.5984, mae: 0.5672, huber: 0.2558, swd: 0.1776, ept: 82.0805
    Epoch [2/50], Test Losses: mse: 0.6792, mae: 0.6239, huber: 0.2962, swd: 0.2250, ept: 85.7691
      Epoch 2 composite train-obj: 0.236363
            No improvement (0.2558), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5221, mae: 0.5108, huber: 0.2206, swd: 0.2040, ept: 115.1952
    Epoch [3/50], Val Losses: mse: 0.6262, mae: 0.5953, huber: 0.2716, swd: 0.1908, ept: 60.7400
    Epoch [3/50], Test Losses: mse: 0.6265, mae: 0.5994, huber: 0.2768, swd: 0.2169, ept: 84.4446
      Epoch 3 composite train-obj: 0.220563
            No improvement (0.2716), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4830, mae: 0.4904, huber: 0.2057, swd: 0.1784, ept: 116.1078
    Epoch [4/50], Val Losses: mse: 0.8663, mae: 0.7425, huber: 0.3720, swd: 0.2194, ept: 69.7928
    Epoch [4/50], Test Losses: mse: 0.7093, mae: 0.6342, huber: 0.3046, swd: 0.1451, ept: 86.2536
      Epoch 4 composite train-obj: 0.205669
            No improvement (0.3720), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4589, mae: 0.4777, huber: 0.1970, swd: 0.1594, ept: 121.4428
    Epoch [5/50], Val Losses: mse: 0.7225, mae: 0.6597, huber: 0.3124, swd: 0.1988, ept: 58.3486
    Epoch [5/50], Test Losses: mse: 0.6573, mae: 0.6187, huber: 0.2891, swd: 0.1959, ept: 82.3689
      Epoch 5 composite train-obj: 0.196962
            No improvement (0.3124), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.4325, mae: 0.4628, huber: 0.1871, swd: 0.1408, ept: 126.0586
    Epoch [6/50], Val Losses: mse: 0.7025, mae: 0.6446, huber: 0.3027, swd: 0.1794, ept: 67.7996
    Epoch [6/50], Test Losses: mse: 0.6780, mae: 0.6311, huber: 0.2976, swd: 0.2107, ept: 90.8633
      Epoch 6 composite train-obj: 0.187136
    Epoch [6/50], Test Losses: mse: 0.6806, mae: 0.6187, huber: 0.2944, swd: 0.2239, ept: 82.9887
    Best round's Test MSE: 0.6806, MAE: 0.6187, SWD: 0.2239
    Best round's Validation MSE: 0.5018, MAE: 0.5155, SWD: 0.1641
    Best round's Test verification MSE : 0.6806, MAE: 0.6187, SWD: 0.2239
    Time taken: 18.02 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7030, mae: 0.6015, huber: 0.2861, swd: 0.3864, ept: 87.8585
    Epoch [1/50], Val Losses: mse: 0.6517, mae: 0.6116, huber: 0.2885, swd: 0.2952, ept: 43.3553
    Epoch [1/50], Test Losses: mse: 0.6981, mae: 0.6381, huber: 0.3039, swd: 0.2921, ept: 64.8520
      Epoch 1 composite train-obj: 0.286103
            Val objective improved inf → 0.2885, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5636, mae: 0.5347, huber: 0.2370, swd: 0.2323, ept: 110.3605
    Epoch [2/50], Val Losses: mse: 0.5387, mae: 0.5248, huber: 0.2308, swd: 0.1245, ept: 81.6615
    Epoch [2/50], Test Losses: mse: 0.6484, mae: 0.6046, huber: 0.2835, swd: 0.1792, ept: 93.7935
      Epoch 2 composite train-obj: 0.236951
            Val objective improved 0.2885 → 0.2308, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5301, mae: 0.5138, huber: 0.2232, swd: 0.2050, ept: 116.8853
    Epoch [3/50], Val Losses: mse: 0.6355, mae: 0.5937, huber: 0.2746, swd: 0.2028, ept: 64.4987
    Epoch [3/50], Test Losses: mse: 0.6303, mae: 0.5996, huber: 0.2777, swd: 0.2352, ept: 86.9640
      Epoch 3 composite train-obj: 0.223204
            No improvement (0.2746), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4917, mae: 0.4944, huber: 0.2088, swd: 0.1811, ept: 118.3604
    Epoch [4/50], Val Losses: mse: 0.7032, mae: 0.6406, huber: 0.3008, swd: 0.1747, ept: 61.9088
    Epoch [4/50], Test Losses: mse: 0.6401, mae: 0.6075, huber: 0.2818, swd: 0.1909, ept: 87.8011
      Epoch 4 composite train-obj: 0.208848
            No improvement (0.3008), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4589, mae: 0.4774, huber: 0.1967, swd: 0.1594, ept: 121.2318
    Epoch [5/50], Val Losses: mse: 0.7594, mae: 0.6772, huber: 0.3268, swd: 0.1639, ept: 63.2793
    Epoch [5/50], Test Losses: mse: 0.6570, mae: 0.6166, huber: 0.2883, swd: 0.1471, ept: 87.9423
      Epoch 5 composite train-obj: 0.196695
            No improvement (0.3268), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4415, mae: 0.4686, huber: 0.1908, swd: 0.1461, ept: 122.2560
    Epoch [6/50], Val Losses: mse: 0.7375, mae: 0.6731, huber: 0.3218, swd: 0.1765, ept: 58.8713
    Epoch [6/50], Test Losses: mse: 0.6530, mae: 0.6162, huber: 0.2873, swd: 0.1902, ept: 83.5490
      Epoch 6 composite train-obj: 0.190779
            No improvement (0.3218), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4205, mae: 0.4573, huber: 0.1832, swd: 0.1326, ept: 126.7713
    Epoch [7/50], Val Losses: mse: 0.6952, mae: 0.6477, huber: 0.3037, swd: 0.1617, ept: 60.0805
    Epoch [7/50], Test Losses: mse: 0.6613, mae: 0.6209, huber: 0.2907, swd: 0.1849, ept: 87.8135
      Epoch 7 composite train-obj: 0.183203
    Epoch [7/50], Test Losses: mse: 0.6484, mae: 0.6046, huber: 0.2835, swd: 0.1792, ept: 93.9960
    Best round's Test MSE: 0.6484, MAE: 0.6046, SWD: 0.1792
    Best round's Validation MSE: 0.5387, MAE: 0.5248, SWD: 0.1245
    Best round's Test verification MSE : 0.6484, MAE: 0.6046, SWD: 0.1792
    Time taken: 21.27 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7109, mae: 0.6049, huber: 0.2888, swd: 0.4151, ept: 87.5876
    Epoch [1/50], Val Losses: mse: 0.5175, mae: 0.5284, huber: 0.2287, swd: 0.1932, ept: 79.4941
    Epoch [1/50], Test Losses: mse: 0.6917, mae: 0.6247, huber: 0.2986, swd: 0.2540, ept: 95.8841
      Epoch 1 composite train-obj: 0.288767
            Val objective improved inf → 0.2287, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5630, mae: 0.5338, huber: 0.2365, swd: 0.2422, ept: 112.0538
    Epoch [2/50], Val Losses: mse: 0.5777, mae: 0.5596, huber: 0.2488, swd: 0.1867, ept: 79.5241
    Epoch [2/50], Test Losses: mse: 0.6811, mae: 0.6246, huber: 0.2968, swd: 0.2438, ept: 91.1985
      Epoch 2 composite train-obj: 0.236518
            No improvement (0.2488), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5198, mae: 0.5096, huber: 0.2197, swd: 0.2090, ept: 114.9095
    Epoch [3/50], Val Losses: mse: 0.7445, mae: 0.6616, huber: 0.3201, swd: 0.2782, ept: 49.7601
    Epoch [3/50], Test Losses: mse: 0.6854, mae: 0.6347, huber: 0.3013, swd: 0.2575, ept: 80.6946
      Epoch 3 composite train-obj: 0.219727
            No improvement (0.3201), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4829, mae: 0.4916, huber: 0.2062, swd: 0.1851, ept: 114.9119
    Epoch [4/50], Val Losses: mse: 0.7018, mae: 0.6385, huber: 0.3011, swd: 0.2155, ept: 50.2494
    Epoch [4/50], Test Losses: mse: 0.6396, mae: 0.6111, huber: 0.2818, swd: 0.1933, ept: 65.8992
      Epoch 4 composite train-obj: 0.206152
            No improvement (0.3011), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4548, mae: 0.4762, huber: 0.1957, swd: 0.1638, ept: 118.3688
    Epoch [5/50], Val Losses: mse: 0.6824, mae: 0.6319, huber: 0.2942, swd: 0.1682, ept: 79.6595
    Epoch [5/50], Test Losses: mse: 0.6813, mae: 0.6278, huber: 0.2971, swd: 0.2000, ept: 95.9426
      Epoch 5 composite train-obj: 0.195673
            No improvement (0.2942), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.4284, mae: 0.4617, huber: 0.1860, swd: 0.1468, ept: 124.4962
    Epoch [6/50], Val Losses: mse: 0.7341, mae: 0.6621, huber: 0.3156, swd: 0.1629, ept: 72.2576
    Epoch [6/50], Test Losses: mse: 0.6953, mae: 0.6349, huber: 0.3023, swd: 0.1731, ept: 93.5745
      Epoch 6 composite train-obj: 0.186031
    Epoch [6/50], Test Losses: mse: 0.6917, mae: 0.6247, huber: 0.2986, swd: 0.2539, ept: 96.1286
    Best round's Test MSE: 0.6917, MAE: 0.6247, SWD: 0.2540
    Best round's Validation MSE: 0.5175, MAE: 0.5284, SWD: 0.1932
    Best round's Test verification MSE : 0.6917, MAE: 0.6247, SWD: 0.2539
    Time taken: 17.62 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq336_pred720_20250510_1933)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6736 ± 0.0184
      mae: 0.6160 ± 0.0084
      huber: 0.2922 ± 0.0064
      swd: 0.2190 ± 0.0307
      ept: 90.8963 ± 5.6404
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.5193 ± 0.0151
      mae: 0.5229 ± 0.0054
      huber: 0.2269 ± 0.0041
      swd: 0.1606 ± 0.0281
      ept: 80.4495 ± 0.9032
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 57.03 seconds
    
    Experiment complete: ACL_etth1_seq336_pred720_20250510_1933
    Model: ACL
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### Timemixer

#### pred=96


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 96
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
    
    Epoch [1/50], Train Losses: mse: 0.4575, mae: 0.4682, huber: 0.1934, swd: 0.1260, ept: 43.9142
    Epoch [1/50], Val Losses: mse: 0.3379, mae: 0.3898, huber: 0.1467, swd: 0.0998, ept: 49.6338
    Epoch [1/50], Test Losses: mse: 0.4217, mae: 0.4474, huber: 0.1825, swd: 0.1248, ept: 40.4225
      Epoch 1 composite train-obj: 0.193415
            Val objective improved inf → 0.1467, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3556, mae: 0.4093, huber: 0.1553, swd: 0.1091, ept: 51.9527
    Epoch [2/50], Val Losses: mse: 0.3328, mae: 0.3843, huber: 0.1443, swd: 0.1026, ept: 51.1607
    Epoch [2/50], Test Losses: mse: 0.4170, mae: 0.4437, huber: 0.1803, swd: 0.1266, ept: 41.1484
      Epoch 2 composite train-obj: 0.155268
            Val objective improved 0.1467 → 0.1443, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3431, mae: 0.4020, huber: 0.1505, swd: 0.1065, ept: 53.0045
    Epoch [3/50], Val Losses: mse: 0.3394, mae: 0.3869, huber: 0.1466, swd: 0.0960, ept: 50.9418
    Epoch [3/50], Test Losses: mse: 0.4182, mae: 0.4401, huber: 0.1793, swd: 0.1130, ept: 42.2332
      Epoch 3 composite train-obj: 0.150512
            No improvement (0.1466), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3295, mae: 0.3953, huber: 0.1457, swd: 0.1031, ept: 53.5252
    Epoch [4/50], Val Losses: mse: 0.3340, mae: 0.3848, huber: 0.1450, swd: 0.1009, ept: 51.1794
    Epoch [4/50], Test Losses: mse: 0.4124, mae: 0.4409, huber: 0.1785, swd: 0.1225, ept: 41.7457
      Epoch 4 composite train-obj: 0.145740
            No improvement (0.1450), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3162, mae: 0.3885, huber: 0.1409, swd: 0.0983, ept: 54.0022
    Epoch [5/50], Val Losses: mse: 0.3476, mae: 0.3941, huber: 0.1505, swd: 0.1054, ept: 50.6694
    Epoch [5/50], Test Losses: mse: 0.4195, mae: 0.4466, huber: 0.1813, swd: 0.1207, ept: 41.9119
      Epoch 5 composite train-obj: 0.140861
            No improvement (0.1505), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3005, mae: 0.3818, huber: 0.1355, swd: 0.0924, ept: 54.2758
    Epoch [6/50], Val Losses: mse: 0.3529, mae: 0.3975, huber: 0.1525, swd: 0.1021, ept: 50.4268
    Epoch [6/50], Test Losses: mse: 0.4154, mae: 0.4460, huber: 0.1802, swd: 0.1139, ept: 41.4422
      Epoch 6 composite train-obj: 0.135464
            No improvement (0.1525), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.2833, mae: 0.3721, huber: 0.1287, swd: 0.0846, ept: 55.0066
    Epoch [7/50], Val Losses: mse: 0.3819, mae: 0.4163, huber: 0.1637, swd: 0.1099, ept: 49.7742
    Epoch [7/50], Test Losses: mse: 0.4343, mae: 0.4586, huber: 0.1876, swd: 0.1161, ept: 40.9761
      Epoch 7 composite train-obj: 0.128727
    Epoch [7/50], Test Losses: mse: 0.4170, mae: 0.4437, huber: 0.1803, swd: 0.1266, ept: 41.1484
    Best round's Test MSE: 0.4170, MAE: 0.4437, SWD: 0.1266
    Best round's Validation MSE: 0.3328, MAE: 0.3843, SWD: 0.1026
    Best round's Test verification MSE : 0.4170, MAE: 0.4437, SWD: 0.1266
    Time taken: 38.67 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4368, mae: 0.4611, huber: 0.1873, swd: 0.1245, ept: 43.6493
    Epoch [1/50], Val Losses: mse: 0.3359, mae: 0.3893, huber: 0.1458, swd: 0.0985, ept: 49.6348
    Epoch [1/50], Test Losses: mse: 0.4253, mae: 0.4519, huber: 0.1846, swd: 0.1268, ept: 39.5315
      Epoch 1 composite train-obj: 0.187303
            Val objective improved inf → 0.1458, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3541, mae: 0.4082, huber: 0.1546, swd: 0.1066, ept: 51.5841
    Epoch [2/50], Val Losses: mse: 0.3383, mae: 0.3901, huber: 0.1471, swd: 0.0992, ept: 50.2345
    Epoch [2/50], Test Losses: mse: 0.4184, mae: 0.4472, huber: 0.1815, swd: 0.1228, ept: 40.6723
      Epoch 2 composite train-obj: 0.154587
            No improvement (0.1471), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3269, mae: 0.3935, huber: 0.1446, swd: 0.0977, ept: 53.5083
    Epoch [3/50], Val Losses: mse: 0.3510, mae: 0.4021, huber: 0.1529, swd: 0.1083, ept: 50.3305
    Epoch [3/50], Test Losses: mse: 0.4224, mae: 0.4537, huber: 0.1842, swd: 0.1328, ept: 39.9110
      Epoch 3 composite train-obj: 0.144577
            No improvement (0.1529), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2992, mae: 0.3806, huber: 0.1348, swd: 0.0876, ept: 54.3777
    Epoch [4/50], Val Losses: mse: 0.3687, mae: 0.4076, huber: 0.1591, swd: 0.1019, ept: 49.2743
    Epoch [4/50], Test Losses: mse: 0.4276, mae: 0.4552, huber: 0.1850, swd: 0.1166, ept: 40.7318
      Epoch 4 composite train-obj: 0.134767
            No improvement (0.1591), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2746, mae: 0.3669, huber: 0.1252, swd: 0.0767, ept: 55.3831
    Epoch [5/50], Val Losses: mse: 0.3811, mae: 0.4143, huber: 0.1637, swd: 0.0989, ept: 48.7219
    Epoch [5/50], Test Losses: mse: 0.4482, mae: 0.4650, huber: 0.1922, swd: 0.1100, ept: 40.5829
      Epoch 5 composite train-obj: 0.125151
            No improvement (0.1637), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2519, mae: 0.3525, huber: 0.1157, swd: 0.0670, ept: 56.6566
    Epoch [6/50], Val Losses: mse: 0.3854, mae: 0.4211, huber: 0.1665, swd: 0.1030, ept: 48.1547
    Epoch [6/50], Test Losses: mse: 0.4425, mae: 0.4664, huber: 0.1919, swd: 0.1174, ept: 39.3118
      Epoch 6 composite train-obj: 0.115727
    Epoch [6/50], Test Losses: mse: 0.4253, mae: 0.4519, huber: 0.1846, swd: 0.1268, ept: 39.5315
    Best round's Test MSE: 0.4253, MAE: 0.4519, SWD: 0.1268
    Best round's Validation MSE: 0.3359, MAE: 0.3893, SWD: 0.0985
    Best round's Test verification MSE : 0.4253, MAE: 0.4519, SWD: 0.1268
    Time taken: 34.28 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4797, mae: 0.4793, huber: 0.2012, swd: 0.1102, ept: 41.2082
    Epoch [1/50], Val Losses: mse: 0.3402, mae: 0.3932, huber: 0.1479, swd: 0.0957, ept: 49.7809
    Epoch [1/50], Test Losses: mse: 0.4254, mae: 0.4516, huber: 0.1842, swd: 0.1133, ept: 39.9479
      Epoch 1 composite train-obj: 0.201151
            Val objective improved inf → 0.1479, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3614, mae: 0.4128, huber: 0.1574, swd: 0.1030, ept: 51.2560
    Epoch [2/50], Val Losses: mse: 0.3360, mae: 0.3872, huber: 0.1458, swd: 0.0950, ept: 50.3723
    Epoch [2/50], Test Losses: mse: 0.4127, mae: 0.4412, huber: 0.1789, swd: 0.1109, ept: 40.9831
      Epoch 2 composite train-obj: 0.157373
            Val objective improved 0.1479 → 0.1458, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3447, mae: 0.4022, huber: 0.1510, swd: 0.0994, ept: 52.6475
    Epoch [3/50], Val Losses: mse: 0.3350, mae: 0.3886, huber: 0.1459, swd: 0.1034, ept: 50.7586
    Epoch [3/50], Test Losses: mse: 0.4177, mae: 0.4469, huber: 0.1815, swd: 0.1235, ept: 40.8197
      Epoch 3 composite train-obj: 0.151002
            No improvement (0.1459), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3336, mae: 0.3970, huber: 0.1472, swd: 0.0970, ept: 53.1919
    Epoch [4/50], Val Losses: mse: 0.3401, mae: 0.3900, huber: 0.1476, swd: 0.0946, ept: 50.3844
    Epoch [4/50], Test Losses: mse: 0.4175, mae: 0.4455, huber: 0.1810, swd: 0.1121, ept: 41.0770
      Epoch 4 composite train-obj: 0.147151
            No improvement (0.1476), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3174, mae: 0.3892, huber: 0.1414, swd: 0.0914, ept: 53.7511
    Epoch [5/50], Val Losses: mse: 0.3425, mae: 0.3950, huber: 0.1493, swd: 0.0996, ept: 50.5736
    Epoch [5/50], Test Losses: mse: 0.4181, mae: 0.4494, huber: 0.1824, swd: 0.1224, ept: 40.1213
      Epoch 5 composite train-obj: 0.141412
            No improvement (0.1493), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.2974, mae: 0.3801, huber: 0.1343, swd: 0.0842, ept: 54.2081
    Epoch [6/50], Val Losses: mse: 0.3644, mae: 0.4052, huber: 0.1571, swd: 0.1009, ept: 49.6375
    Epoch [6/50], Test Losses: mse: 0.4279, mae: 0.4540, huber: 0.1850, swd: 0.1094, ept: 40.5783
      Epoch 6 composite train-obj: 0.134276
            No improvement (0.1571), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.2789, mae: 0.3698, huber: 0.1271, swd: 0.0767, ept: 54.8321
    Epoch [7/50], Val Losses: mse: 0.3763, mae: 0.4131, huber: 0.1618, swd: 0.1035, ept: 49.1345
    Epoch [7/50], Test Losses: mse: 0.4350, mae: 0.4603, huber: 0.1884, swd: 0.1100, ept: 40.4667
      Epoch 7 composite train-obj: 0.127076
    Epoch [7/50], Test Losses: mse: 0.4127, mae: 0.4412, huber: 0.1789, swd: 0.1109, ept: 40.9831
    Best round's Test MSE: 0.4127, MAE: 0.4412, SWD: 0.1109
    Best round's Validation MSE: 0.3360, MAE: 0.3872, SWD: 0.0950
    Best round's Test verification MSE : 0.4127, MAE: 0.4412, SWD: 0.1109
    Time taken: 39.75 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq336_pred96_20250510_1931)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4183 ± 0.0052
      mae: 0.4456 ± 0.0046
      huber: 0.1812 ± 0.0024
      swd: 0.1214 ± 0.0075
      ept: 40.5544 ± 0.7264
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3349 ± 0.0015
      mae: 0.3869 ± 0.0021
      huber: 0.1453 ± 0.0007
      swd: 0.0987 ± 0.0031
      ept: 50.3893 ± 0.6231
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 112.79 seconds
    
    Experiment complete: TimeMixer_etth1_seq336_pred96_20250510_1931
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.5182, mae: 0.5001, huber: 0.2161, swd: 0.1497, ept: 63.7551
    Epoch [1/50], Val Losses: mse: 0.3906, mae: 0.4254, huber: 0.1680, swd: 0.1014, ept: 68.3283
    Epoch [1/50], Test Losses: mse: 0.4726, mae: 0.4874, huber: 0.2058, swd: 0.1205, ept: 57.5851
      Epoch 1 composite train-obj: 0.216068
            Val objective improved inf → 0.1680, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4429, mae: 0.4547, huber: 0.1870, swd: 0.1380, ept: 76.2970
    Epoch [2/50], Val Losses: mse: 0.3861, mae: 0.4208, huber: 0.1658, swd: 0.1005, ept: 70.1081
    Epoch [2/50], Test Losses: mse: 0.4612, mae: 0.4808, huber: 0.2015, swd: 0.1206, ept: 58.0035
      Epoch 2 composite train-obj: 0.186955
            Val objective improved 0.1680 → 0.1658, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4143, mae: 0.4413, huber: 0.1772, swd: 0.1332, ept: 78.0240
    Epoch [3/50], Val Losses: mse: 0.4058, mae: 0.4341, huber: 0.1738, swd: 0.1029, ept: 69.7470
    Epoch [3/50], Test Losses: mse: 0.4623, mae: 0.4823, huber: 0.2019, swd: 0.1163, ept: 58.3318
      Epoch 3 composite train-obj: 0.177247
            No improvement (0.1738), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3851, mae: 0.4293, huber: 0.1676, swd: 0.1231, ept: 79.0008
    Epoch [4/50], Val Losses: mse: 0.4490, mae: 0.4576, huber: 0.1892, swd: 0.1149, ept: 68.3754
    Epoch [4/50], Test Losses: mse: 0.4755, mae: 0.4922, huber: 0.2073, swd: 0.1204, ept: 58.4950
      Epoch 4 composite train-obj: 0.167612
            No improvement (0.1892), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3577, mae: 0.4171, huber: 0.1581, swd: 0.1083, ept: 80.1254
    Epoch [5/50], Val Losses: mse: 0.4448, mae: 0.4623, huber: 0.1900, swd: 0.1114, ept: 68.1820
    Epoch [5/50], Test Losses: mse: 0.4791, mae: 0.4963, huber: 0.2100, swd: 0.1145, ept: 57.6534
      Epoch 5 composite train-obj: 0.158121
            No improvement (0.1900), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3309, mae: 0.4038, huber: 0.1481, swd: 0.0942, ept: 81.5074
    Epoch [6/50], Val Losses: mse: 0.5070, mae: 0.4899, huber: 0.2101, swd: 0.1209, ept: 65.6424
    Epoch [6/50], Test Losses: mse: 0.4980, mae: 0.5067, huber: 0.2167, swd: 0.1194, ept: 57.3371
      Epoch 6 composite train-obj: 0.148148
            No improvement (0.2101), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3080, mae: 0.3904, huber: 0.1390, swd: 0.0834, ept: 83.1687
    Epoch [7/50], Val Losses: mse: 0.4875, mae: 0.4845, huber: 0.2051, swd: 0.1203, ept: 65.0250
    Epoch [7/50], Test Losses: mse: 0.5051, mae: 0.5108, huber: 0.2198, swd: 0.1279, ept: 56.6794
      Epoch 7 composite train-obj: 0.139010
    Epoch [7/50], Test Losses: mse: 0.4612, mae: 0.4808, huber: 0.2015, swd: 0.1206, ept: 58.0035
    Best round's Test MSE: 0.4612, MAE: 0.4808, SWD: 0.1206
    Best round's Validation MSE: 0.3861, MAE: 0.4208, SWD: 0.1005
    Best round's Test verification MSE : 0.4612, MAE: 0.4808, SWD: 0.1206
    Time taken: 39.66 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5155, mae: 0.4989, huber: 0.2150, swd: 0.1535, ept: 64.3466
    Epoch [1/50], Val Losses: mse: 0.3913, mae: 0.4267, huber: 0.1688, swd: 0.0952, ept: 67.1749
    Epoch [1/50], Test Losses: mse: 0.4693, mae: 0.4851, huber: 0.2043, swd: 0.1145, ept: 57.2430
      Epoch 1 composite train-obj: 0.215035
            Val objective improved inf → 0.1688, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4384, mae: 0.4528, huber: 0.1854, swd: 0.1394, ept: 76.4561
    Epoch [2/50], Val Losses: mse: 0.3884, mae: 0.4266, huber: 0.1682, swd: 0.1063, ept: 69.0591
    Epoch [2/50], Test Losses: mse: 0.4703, mae: 0.4886, huber: 0.2059, swd: 0.1384, ept: 57.2356
      Epoch 2 composite train-obj: 0.185428
            Val objective improved 0.1688 → 0.1682, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4111, mae: 0.4402, huber: 0.1760, swd: 0.1325, ept: 78.3219
    Epoch [3/50], Val Losses: mse: 0.3999, mae: 0.4333, huber: 0.1727, swd: 0.0921, ept: 68.7596
    Epoch [3/50], Test Losses: mse: 0.4651, mae: 0.4858, huber: 0.2035, swd: 0.1080, ept: 57.6739
      Epoch 3 composite train-obj: 0.176000
            No improvement (0.1727), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3781, mae: 0.4264, huber: 0.1651, swd: 0.1176, ept: 80.0636
    Epoch [4/50], Val Losses: mse: 0.4346, mae: 0.4500, huber: 0.1844, swd: 0.0911, ept: 68.0457
    Epoch [4/50], Test Losses: mse: 0.4899, mae: 0.4994, huber: 0.2119, swd: 0.0999, ept: 57.7695
      Epoch 4 composite train-obj: 0.165133
            No improvement (0.1844), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3459, mae: 0.4124, huber: 0.1541, swd: 0.1012, ept: 81.1505
    Epoch [5/50], Val Losses: mse: 0.4474, mae: 0.4619, huber: 0.1905, swd: 0.1116, ept: 66.5084
    Epoch [5/50], Test Losses: mse: 0.5103, mae: 0.5141, huber: 0.2209, swd: 0.1230, ept: 57.7219
      Epoch 5 composite train-obj: 0.154097
            No improvement (0.1905), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3180, mae: 0.3974, huber: 0.1434, swd: 0.0875, ept: 82.8179
    Epoch [6/50], Val Losses: mse: 0.4409, mae: 0.4589, huber: 0.1887, swd: 0.0969, ept: 65.5982
    Epoch [6/50], Test Losses: mse: 0.4965, mae: 0.5066, huber: 0.2165, swd: 0.1107, ept: 57.3140
      Epoch 6 composite train-obj: 0.143437
            No improvement (0.1887), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.2958, mae: 0.3842, huber: 0.1345, swd: 0.0771, ept: 84.4510
    Epoch [7/50], Val Losses: mse: 0.4586, mae: 0.4671, huber: 0.1949, swd: 0.1075, ept: 65.0516
    Epoch [7/50], Test Losses: mse: 0.5196, mae: 0.5186, huber: 0.2249, swd: 0.1207, ept: 56.1325
      Epoch 7 composite train-obj: 0.134536
    Epoch [7/50], Test Losses: mse: 0.4703, mae: 0.4886, huber: 0.2059, swd: 0.1384, ept: 57.2356
    Best round's Test MSE: 0.4703, MAE: 0.4886, SWD: 0.1384
    Best round's Validation MSE: 0.3884, MAE: 0.4266, SWD: 0.1063
    Best round's Test verification MSE : 0.4703, MAE: 0.4886, SWD: 0.1384
    Time taken: 38.83 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5138, mae: 0.4981, huber: 0.2145, swd: 0.1431, ept: 64.4972
    Epoch [1/50], Val Losses: mse: 0.3902, mae: 0.4232, huber: 0.1675, swd: 0.0997, ept: 69.6320
    Epoch [1/50], Test Losses: mse: 0.4661, mae: 0.4821, huber: 0.2030, swd: 0.1122, ept: 57.2084
      Epoch 1 composite train-obj: 0.214505
            Val objective improved inf → 0.1675, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4393, mae: 0.4537, huber: 0.1859, swd: 0.1273, ept: 76.1975
    Epoch [2/50], Val Losses: mse: 0.3977, mae: 0.4266, huber: 0.1704, swd: 0.1017, ept: 70.0936
    Epoch [2/50], Test Losses: mse: 0.4663, mae: 0.4817, huber: 0.2026, swd: 0.1086, ept: 58.7704
      Epoch 2 composite train-obj: 0.185895
            No improvement (0.1704), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4122, mae: 0.4401, huber: 0.1762, swd: 0.1215, ept: 78.4246
    Epoch [3/50], Val Losses: mse: 0.4062, mae: 0.4336, huber: 0.1741, swd: 0.0934, ept: 70.0141
    Epoch [3/50], Test Losses: mse: 0.4608, mae: 0.4794, huber: 0.2007, swd: 0.0977, ept: 58.6395
      Epoch 3 composite train-obj: 0.176224
            No improvement (0.1741), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.3778, mae: 0.4264, huber: 0.1652, swd: 0.1091, ept: 79.5240
    Epoch [4/50], Val Losses: mse: 0.4158, mae: 0.4420, huber: 0.1785, swd: 0.1003, ept: 68.8850
    Epoch [4/50], Test Losses: mse: 0.4723, mae: 0.4910, huber: 0.2067, swd: 0.1124, ept: 58.0766
      Epoch 4 composite train-obj: 0.165188
            No improvement (0.1785), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3429, mae: 0.4105, huber: 0.1529, swd: 0.0927, ept: 80.8047
    Epoch [5/50], Val Losses: mse: 0.4459, mae: 0.4574, huber: 0.1892, swd: 0.1045, ept: 67.0439
    Epoch [5/50], Test Losses: mse: 0.4801, mae: 0.4965, huber: 0.2095, swd: 0.1043, ept: 58.8676
      Epoch 5 composite train-obj: 0.152879
            No improvement (0.1892), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3163, mae: 0.3960, huber: 0.1426, swd: 0.0807, ept: 82.6134
    Epoch [6/50], Val Losses: mse: 0.4724, mae: 0.4692, huber: 0.1980, swd: 0.1034, ept: 65.8032
    Epoch [6/50], Test Losses: mse: 0.5024, mae: 0.5084, huber: 0.2178, swd: 0.1016, ept: 57.6750
      Epoch 6 composite train-obj: 0.142564
    Epoch [6/50], Test Losses: mse: 0.4661, mae: 0.4821, huber: 0.2030, swd: 0.1122, ept: 57.2084
    Best round's Test MSE: 0.4661, MAE: 0.4821, SWD: 0.1122
    Best round's Validation MSE: 0.3902, MAE: 0.4232, SWD: 0.0997
    Best round's Test verification MSE : 0.4661, MAE: 0.4821, SWD: 0.1122
    Time taken: 33.34 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq336_pred196_20250510_1929)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4659 ± 0.0037
      mae: 0.4838 ± 0.0034
      huber: 0.2035 ± 0.0018
      swd: 0.1237 ± 0.0109
      ept: 57.4825 ± 0.3686
      count: 10.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3882 ± 0.0017
      mae: 0.4235 ± 0.0024
      huber: 0.1672 ± 0.0010
      swd: 0.1022 ± 0.0029
      ept: 69.5997 ± 0.4288
      count: 10.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 111.92 seconds
    
    Experiment complete: TimeMixer_etth1_seq336_pred196_20250510_1929
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 91
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 91
    Validation Batches: 9
    Test Batches: 22
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6018, mae: 0.5411, huber: 0.2465, swd: 0.1627, ept: 76.8199
    Epoch [1/50], Val Losses: mse: 0.3958, mae: 0.4269, huber: 0.1692, swd: 0.0810, ept: 87.3087
    Epoch [1/50], Test Losses: mse: 0.5246, mae: 0.5211, huber: 0.2290, swd: 0.1034, ept: 74.3615
      Epoch 1 composite train-obj: 0.246517
            Val objective improved inf → 0.1692, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5122, mae: 0.4921, huber: 0.2129, swd: 0.1506, ept: 96.0633
    Epoch [2/50], Val Losses: mse: 0.3822, mae: 0.4190, huber: 0.1642, swd: 0.0822, ept: 91.4237
    Epoch [2/50], Test Losses: mse: 0.5324, mae: 0.5282, huber: 0.2332, swd: 0.1225, ept: 72.6065
      Epoch 2 composite train-obj: 0.212869
            Val objective improved 0.1692 → 0.1642, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4845, mae: 0.4790, huber: 0.2032, swd: 0.1489, ept: 99.0209
    Epoch [3/50], Val Losses: mse: 0.4278, mae: 0.4410, huber: 0.1791, swd: 0.0873, ept: 88.8073
    Epoch [3/50], Test Losses: mse: 0.5452, mae: 0.5318, huber: 0.2348, swd: 0.1050, ept: 75.2103
      Epoch 3 composite train-obj: 0.203207
            No improvement (0.1791), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4524, mae: 0.4663, huber: 0.1928, swd: 0.1361, ept: 100.0401
    Epoch [4/50], Val Losses: mse: 0.3913, mae: 0.4378, huber: 0.1714, swd: 0.0712, ept: 90.3678
    Epoch [4/50], Test Losses: mse: 0.5454, mae: 0.5376, huber: 0.2382, swd: 0.1074, ept: 74.3173
      Epoch 4 composite train-obj: 0.192783
            No improvement (0.1714), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4270, mae: 0.4550, huber: 0.1838, swd: 0.1246, ept: 101.0845
    Epoch [5/50], Val Losses: mse: 0.4610, mae: 0.4678, huber: 0.1938, swd: 0.0893, ept: 88.0156
    Epoch [5/50], Test Losses: mse: 0.5572, mae: 0.5443, huber: 0.2414, swd: 0.1254, ept: 74.0716
      Epoch 5 composite train-obj: 0.183845
            No improvement (0.1938), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3969, mae: 0.4404, huber: 0.1729, swd: 0.1100, ept: 102.8351
    Epoch [6/50], Val Losses: mse: 0.4455, mae: 0.4719, huber: 0.1933, swd: 0.0993, ept: 86.1189
    Epoch [6/50], Test Losses: mse: 0.5750, mae: 0.5551, huber: 0.2500, swd: 0.1521, ept: 73.3450
      Epoch 6 composite train-obj: 0.172895
            No improvement (0.1933), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3836, mae: 0.4344, huber: 0.1681, swd: 0.1043, ept: 104.2212
    Epoch [7/50], Val Losses: mse: 0.5239, mae: 0.4981, huber: 0.2151, swd: 0.0941, ept: 85.4251
    Epoch [7/50], Test Losses: mse: 0.5889, mae: 0.5553, huber: 0.2500, swd: 0.1165, ept: 74.2945
      Epoch 7 composite train-obj: 0.168144
    Epoch [7/50], Test Losses: mse: 0.5324, mae: 0.5282, huber: 0.2332, swd: 0.1225, ept: 72.6065
    Best round's Test MSE: 0.5324, MAE: 0.5282, SWD: 0.1225
    Best round's Validation MSE: 0.3822, MAE: 0.4190, SWD: 0.0822
    Best round's Test verification MSE : 0.5324, MAE: 0.5282, SWD: 0.1225
    Time taken: 40.25 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6284, mae: 0.5535, huber: 0.2549, swd: 0.1612, ept: 73.0734
    Epoch [1/50], Val Losses: mse: 0.4008, mae: 0.4322, huber: 0.1720, swd: 0.0926, ept: 87.1087
    Epoch [1/50], Test Losses: mse: 0.5426, mae: 0.5332, huber: 0.2363, swd: 0.1252, ept: 70.5951
      Epoch 1 composite train-obj: 0.254881
            Val objective improved inf → 0.1720, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5167, mae: 0.4952, huber: 0.2146, swd: 0.1545, ept: 94.1160
    Epoch [2/50], Val Losses: mse: 0.3801, mae: 0.4182, huber: 0.1635, swd: 0.0829, ept: 90.4118
    Epoch [2/50], Test Losses: mse: 0.5237, mae: 0.5205, huber: 0.2288, swd: 0.1257, ept: 73.1565
      Epoch 2 composite train-obj: 0.214591
            Val objective improved 0.1720 → 0.1635, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4884, mae: 0.4815, huber: 0.2047, swd: 0.1522, ept: 98.6296
    Epoch [3/50], Val Losses: mse: 0.3821, mae: 0.4248, huber: 0.1656, swd: 0.0770, ept: 90.6894
    Epoch [3/50], Test Losses: mse: 0.5360, mae: 0.5315, huber: 0.2342, swd: 0.1262, ept: 72.1299
      Epoch 3 composite train-obj: 0.204665
            No improvement (0.1656), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4558, mae: 0.4675, huber: 0.1935, swd: 0.1432, ept: 99.7835
    Epoch [4/50], Val Losses: mse: 0.4212, mae: 0.4428, huber: 0.1783, swd: 0.0835, ept: 88.5315
    Epoch [4/50], Test Losses: mse: 0.5522, mae: 0.5451, huber: 0.2414, swd: 0.1242, ept: 73.5483
      Epoch 4 composite train-obj: 0.193486
            No improvement (0.1783), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4245, mae: 0.4542, huber: 0.1830, swd: 0.1283, ept: 101.3862
    Epoch [5/50], Val Losses: mse: 0.4986, mae: 0.4867, huber: 0.2064, swd: 0.0962, ept: 86.8548
    Epoch [5/50], Test Losses: mse: 0.6064, mae: 0.5782, huber: 0.2624, swd: 0.1312, ept: 71.9822
      Epoch 5 composite train-obj: 0.182959
            No improvement (0.2064), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3959, mae: 0.4407, huber: 0.1726, swd: 0.1149, ept: 103.3522
    Epoch [6/50], Val Losses: mse: 0.5129, mae: 0.4937, huber: 0.2110, swd: 0.0933, ept: 86.8169
    Epoch [6/50], Test Losses: mse: 0.5864, mae: 0.5631, huber: 0.2531, swd: 0.1154, ept: 72.6693
      Epoch 6 composite train-obj: 0.172639
            No improvement (0.2110), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3685, mae: 0.4271, huber: 0.1626, swd: 0.1000, ept: 104.5860
    Epoch [7/50], Val Losses: mse: 0.4739, mae: 0.4768, huber: 0.1997, swd: 0.0930, ept: 86.7269
    Epoch [7/50], Test Losses: mse: 0.5649, mae: 0.5485, huber: 0.2452, swd: 0.1202, ept: 74.6842
      Epoch 7 composite train-obj: 0.162591
    Epoch [7/50], Test Losses: mse: 0.5237, mae: 0.5205, huber: 0.2288, swd: 0.1257, ept: 73.1565
    Best round's Test MSE: 0.5237, MAE: 0.5205, SWD: 0.1257
    Best round's Validation MSE: 0.3801, MAE: 0.4182, SWD: 0.0829
    Best round's Test verification MSE : 0.5237, MAE: 0.5205, SWD: 0.1257
    Time taken: 39.58 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5982, mae: 0.5380, huber: 0.2441, swd: 0.1692, ept: 79.0881
    Epoch [1/50], Val Losses: mse: 0.4035, mae: 0.4294, huber: 0.1714, swd: 0.0901, ept: 88.8461
    Epoch [1/50], Test Losses: mse: 0.5388, mae: 0.5303, huber: 0.2344, swd: 0.1217, ept: 73.1962
      Epoch 1 composite train-obj: 0.244122
            Val objective improved inf → 0.1714, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5206, mae: 0.4946, huber: 0.2152, swd: 0.1571, ept: 95.2915
    Epoch [2/50], Val Losses: mse: 0.3936, mae: 0.4288, huber: 0.1689, swd: 0.0694, ept: 91.2883
    Epoch [2/50], Test Losses: mse: 0.5306, mae: 0.5254, huber: 0.2309, swd: 0.0939, ept: 73.1274
      Epoch 2 composite train-obj: 0.215179
            Val objective improved 0.1714 → 0.1689, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4855, mae: 0.4797, huber: 0.2036, swd: 0.1498, ept: 97.9807
    Epoch [3/50], Val Losses: mse: 0.4003, mae: 0.4328, huber: 0.1718, swd: 0.0687, ept: 90.1948
    Epoch [3/50], Test Losses: mse: 0.5288, mae: 0.5241, huber: 0.2302, swd: 0.0968, ept: 73.8228
      Epoch 3 composite train-obj: 0.203625
            No improvement (0.1718), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4571, mae: 0.4671, huber: 0.1937, swd: 0.1401, ept: 99.7098
    Epoch [4/50], Val Losses: mse: 0.3963, mae: 0.4358, huber: 0.1721, swd: 0.0841, ept: 90.9546
    Epoch [4/50], Test Losses: mse: 0.5389, mae: 0.5334, huber: 0.2356, swd: 0.1402, ept: 72.8934
      Epoch 4 composite train-obj: 0.193678
            No improvement (0.1721), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4229, mae: 0.4521, huber: 0.1818, swd: 0.1279, ept: 101.3096
    Epoch [5/50], Val Losses: mse: 0.4893, mae: 0.4804, huber: 0.2029, swd: 0.0921, ept: 87.9303
    Epoch [5/50], Test Losses: mse: 0.5636, mae: 0.5520, huber: 0.2452, swd: 0.1285, ept: 73.1549
      Epoch 5 composite train-obj: 0.181804
            No improvement (0.2029), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3988, mae: 0.4400, huber: 0.1729, swd: 0.1163, ept: 102.6672
    Epoch [6/50], Val Losses: mse: 0.4646, mae: 0.4630, huber: 0.1929, swd: 0.0807, ept: 88.4572
    Epoch [6/50], Test Losses: mse: 0.5578, mae: 0.5443, huber: 0.2422, swd: 0.1134, ept: 74.0591
      Epoch 6 composite train-obj: 0.172886
            No improvement (0.1929), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3699, mae: 0.4253, huber: 0.1623, swd: 0.1008, ept: 105.2755
    Epoch [7/50], Val Losses: mse: 0.4572, mae: 0.4648, huber: 0.1924, swd: 0.0840, ept: 86.3714
    Epoch [7/50], Test Losses: mse: 0.5949, mae: 0.5608, huber: 0.2564, swd: 0.1233, ept: 71.6082
      Epoch 7 composite train-obj: 0.162301
    Epoch [7/50], Test Losses: mse: 0.5306, mae: 0.5254, huber: 0.2309, swd: 0.0939, ept: 73.1274
    Best round's Test MSE: 0.5306, MAE: 0.5254, SWD: 0.0939
    Best round's Validation MSE: 0.3936, MAE: 0.4288, SWD: 0.0694
    Best round's Test verification MSE : 0.5306, MAE: 0.5254, SWD: 0.0939
    Time taken: 40.21 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq336_pred336_20250510_1927)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5289 ± 0.0037
      mae: 0.5247 ± 0.0032
      huber: 0.2309 ± 0.0018
      swd: 0.1140 ± 0.0143
      ept: 72.9634 ± 0.2527
      count: 9.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3853 ± 0.0059
      mae: 0.4220 ± 0.0048
      huber: 0.1655 ± 0.0024
      swd: 0.0782 ± 0.0062
      ept: 91.0413 ± 0.4485
      count: 9.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 120.11 seconds
    
    Experiment complete: TimeMixer_etth1_seq336_pred336_20250510_1927
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 88
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 720
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
    
    Epoch [1/50], Train Losses: mse: 0.7580, mae: 0.6150, huber: 0.3009, swd: 0.2119, ept: 94.7811
    Epoch [1/50], Val Losses: mse: 0.4226, mae: 0.4511, huber: 0.1840, swd: 0.0715, ept: 140.8250
    Epoch [1/50], Test Losses: mse: 0.6811, mae: 0.6143, huber: 0.2940, swd: 0.1476, ept: 99.7564
      Epoch 1 composite train-obj: 0.300866
            Val objective improved inf → 0.1840, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6515, mae: 0.5659, huber: 0.2642, swd: 0.1988, ept: 114.1120
    Epoch [2/50], Val Losses: mse: 0.4113, mae: 0.4469, huber: 0.1793, swd: 0.0773, ept: 138.3705
    Epoch [2/50], Test Losses: mse: 0.7105, mae: 0.6352, huber: 0.3070, swd: 0.1764, ept: 100.9987
      Epoch 2 composite train-obj: 0.264196
            Val objective improved 0.1840 → 0.1793, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6211, mae: 0.5529, huber: 0.2540, swd: 0.1996, ept: 117.1051
    Epoch [3/50], Val Losses: mse: 0.4067, mae: 0.4400, huber: 0.1768, swd: 0.0682, ept: 140.2358
    Epoch [3/50], Test Losses: mse: 0.6647, mae: 0.6104, huber: 0.2888, swd: 0.1633, ept: 109.9807
      Epoch 3 composite train-obj: 0.253991
            Val objective improved 0.1793 → 0.1768, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5849, mae: 0.5374, huber: 0.2414, swd: 0.1937, ept: 119.9459
    Epoch [4/50], Val Losses: mse: 0.4249, mae: 0.4532, huber: 0.1843, swd: 0.0604, ept: 141.8451
    Epoch [4/50], Test Losses: mse: 0.6478, mae: 0.6033, huber: 0.2821, swd: 0.1496, ept: 109.5935
      Epoch 4 composite train-obj: 0.241440
            No improvement (0.1843), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5576, mae: 0.5245, huber: 0.2315, swd: 0.1825, ept: 122.0497
    Epoch [5/50], Val Losses: mse: 0.5197, mae: 0.5064, huber: 0.2192, swd: 0.0649, ept: 141.6643
    Epoch [5/50], Test Losses: mse: 0.6972, mae: 0.6254, huber: 0.2988, swd: 0.1452, ept: 104.3238
      Epoch 5 composite train-obj: 0.231464
            No improvement (0.2192), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5276, mae: 0.5098, huber: 0.2203, swd: 0.1702, ept: 125.5514
    Epoch [6/50], Val Losses: mse: 0.5537, mae: 0.5314, huber: 0.2349, swd: 0.0823, ept: 141.9738
    Epoch [6/50], Test Losses: mse: 0.7796, mae: 0.6722, huber: 0.3321, swd: 0.1837, ept: 101.9432
      Epoch 6 composite train-obj: 0.220344
            No improvement (0.2349), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5136, mae: 0.5017, huber: 0.2150, swd: 0.1663, ept: 128.3438
    Epoch [7/50], Val Losses: mse: 0.5558, mae: 0.5227, huber: 0.2317, swd: 0.0702, ept: 140.8528
    Epoch [7/50], Test Losses: mse: 0.7092, mae: 0.6318, huber: 0.3037, swd: 0.1586, ept: 100.8648
      Epoch 7 composite train-obj: 0.214961
            No improvement (0.2317), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4764, mae: 0.4837, huber: 0.2018, swd: 0.1501, ept: 132.5778
    Epoch [8/50], Val Losses: mse: 0.5329, mae: 0.5180, huber: 0.2264, swd: 0.0687, ept: 140.4516
    Epoch [8/50], Test Losses: mse: 0.6993, mae: 0.6212, huber: 0.2985, swd: 0.1257, ept: 102.8658
      Epoch 8 composite train-obj: 0.201811
    Epoch [8/50], Test Losses: mse: 0.6647, mae: 0.6104, huber: 0.2888, swd: 0.1633, ept: 109.9807
    Best round's Test MSE: 0.6647, MAE: 0.6104, SWD: 0.1633
    Best round's Validation MSE: 0.4067, MAE: 0.4400, SWD: 0.0682
    Best round's Test verification MSE : 0.6647, MAE: 0.6104, SWD: 0.1633
    Time taken: 50.25 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7385, mae: 0.6069, huber: 0.2948, swd: 0.2038, ept: 98.4307
    Epoch [1/50], Val Losses: mse: 0.4239, mae: 0.4497, huber: 0.1839, swd: 0.0670, ept: 139.0926
    Epoch [1/50], Test Losses: mse: 0.6800, mae: 0.6104, huber: 0.2929, swd: 0.1347, ept: 104.4708
      Epoch 1 composite train-obj: 0.294829
            Val objective improved inf → 0.1839, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6505, mae: 0.5650, huber: 0.2639, swd: 0.1886, ept: 117.9385
    Epoch [2/50], Val Losses: mse: 0.4549, mae: 0.4626, huber: 0.1926, swd: 0.0725, ept: 144.8075
    Epoch [2/50], Test Losses: mse: 0.7282, mae: 0.6377, huber: 0.3112, swd: 0.1397, ept: 105.0616
      Epoch 2 composite train-obj: 0.263912
            No improvement (0.1926), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.6079, mae: 0.5460, huber: 0.2488, swd: 0.1873, ept: 119.6148
    Epoch [3/50], Val Losses: mse: 0.4607, mae: 0.4733, huber: 0.1972, swd: 0.0781, ept: 143.4318
    Epoch [3/50], Test Losses: mse: 0.7544, mae: 0.6574, huber: 0.3234, swd: 0.1685, ept: 99.5053
      Epoch 3 composite train-obj: 0.248794
            No improvement (0.1972), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.5692, mae: 0.5276, huber: 0.2347, swd: 0.1807, ept: 122.3299
    Epoch [4/50], Val Losses: mse: 0.4688, mae: 0.4738, huber: 0.1989, swd: 0.0710, ept: 141.6829
    Epoch [4/50], Test Losses: mse: 0.7158, mae: 0.6408, huber: 0.3096, swd: 0.1538, ept: 105.3351
      Epoch 4 composite train-obj: 0.234711
            No improvement (0.1989), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.5313, mae: 0.5090, huber: 0.2208, swd: 0.1678, ept: 127.0788
    Epoch [5/50], Val Losses: mse: 0.5008, mae: 0.4944, huber: 0.2126, swd: 0.0643, ept: 141.1623
    Epoch [5/50], Test Losses: mse: 0.7170, mae: 0.6372, huber: 0.3091, swd: 0.1455, ept: 104.1090
      Epoch 5 composite train-obj: 0.220831
            No improvement (0.2126), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.4964, mae: 0.4928, huber: 0.2087, swd: 0.1539, ept: 129.9854
    Epoch [6/50], Val Losses: mse: 0.5906, mae: 0.5424, huber: 0.2464, swd: 0.0675, ept: 141.3550
    Epoch [6/50], Test Losses: mse: 0.7810, mae: 0.6606, huber: 0.3288, swd: 0.1432, ept: 103.6165
      Epoch 6 composite train-obj: 0.208723
    Epoch [6/50], Test Losses: mse: 0.6800, mae: 0.6104, huber: 0.2929, swd: 0.1347, ept: 104.4708
    Best round's Test MSE: 0.6800, MAE: 0.6104, SWD: 0.1347
    Best round's Validation MSE: 0.4239, MAE: 0.4497, SWD: 0.0670
    Best round's Test verification MSE : 0.6800, MAE: 0.6104, SWD: 0.1347
    Time taken: 37.53 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7305, mae: 0.6037, huber: 0.2924, swd: 0.2234, ept: 96.1078
    Epoch [1/50], Val Losses: mse: 0.4294, mae: 0.4496, huber: 0.1840, swd: 0.0623, ept: 140.9697
    Epoch [1/50], Test Losses: mse: 0.6768, mae: 0.6086, huber: 0.2909, swd: 0.1266, ept: 104.1098
      Epoch 1 composite train-obj: 0.292408
            Val objective improved inf → 0.1840, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6344, mae: 0.5579, huber: 0.2581, swd: 0.2008, ept: 113.0982
    Epoch [2/50], Val Losses: mse: 0.5334, mae: 0.5055, huber: 0.2205, swd: 0.0853, ept: 142.7132
    Epoch [2/50], Test Losses: mse: 0.7473, mae: 0.6499, huber: 0.3185, swd: 0.1528, ept: 98.5784
      Epoch 2 composite train-obj: 0.258116
            No improvement (0.2205), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5867, mae: 0.5364, huber: 0.2410, swd: 0.1997, ept: 119.3729
    Epoch [3/50], Val Losses: mse: 0.5820, mae: 0.5303, huber: 0.2378, swd: 0.0818, ept: 142.4611
    Epoch [3/50], Test Losses: mse: 0.7336, mae: 0.6438, huber: 0.3137, swd: 0.1542, ept: 103.6752
      Epoch 3 composite train-obj: 0.240965
            No improvement (0.2378), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.5368, mae: 0.5125, huber: 0.2228, swd: 0.1807, ept: 123.4011
    Epoch [4/50], Val Losses: mse: 0.5288, mae: 0.5059, huber: 0.2200, swd: 0.0686, ept: 142.6604
    Epoch [4/50], Test Losses: mse: 0.7019, mae: 0.6240, huber: 0.3002, swd: 0.1312, ept: 102.9176
      Epoch 4 composite train-obj: 0.222823
            No improvement (0.2200), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4988, mae: 0.4943, huber: 0.2094, swd: 0.1635, ept: 127.3738
    Epoch [5/50], Val Losses: mse: 0.6841, mae: 0.5823, huber: 0.2737, swd: 0.0874, ept: 140.3692
    Epoch [5/50], Test Losses: mse: 0.7429, mae: 0.6413, huber: 0.3138, swd: 0.1325, ept: 101.1701
      Epoch 5 composite train-obj: 0.209393
            No improvement (0.2737), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.4624, mae: 0.4779, huber: 0.1971, swd: 0.1479, ept: 129.9913
    Epoch [6/50], Val Losses: mse: 0.6207, mae: 0.5581, huber: 0.2550, swd: 0.0865, ept: 138.0248
    Epoch [6/50], Test Losses: mse: 0.7788, mae: 0.6584, huber: 0.3270, swd: 0.1466, ept: 97.9209
      Epoch 6 composite train-obj: 0.197133
    Epoch [6/50], Test Losses: mse: 0.6768, mae: 0.6086, huber: 0.2909, swd: 0.1266, ept: 104.1098
    Best round's Test MSE: 0.6768, MAE: 0.6086, SWD: 0.1266
    Best round's Validation MSE: 0.4294, MAE: 0.4496, SWD: 0.0623
    Best round's Test verification MSE : 0.6768, MAE: 0.6086, SWD: 0.1266
    Time taken: 37.49 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq336_pred720_20250510_1925)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6739 ± 0.0066
      mae: 0.6098 ± 0.0009
      huber: 0.2909 ± 0.0017
      swd: 0.1415 ± 0.0158
      ept: 106.1871 ± 2.6865
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4200 ± 0.0097
      mae: 0.4464 ± 0.0045
      huber: 0.1816 ± 0.0034
      swd: 0.0658 ± 0.0025
      ept: 140.0994 ± 0.7724
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 125.34 seconds
    
    Experiment complete: TimeMixer_etth1_seq336_pred720_20250510_1925
    Model: TimeMixer
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### PatchTST

#### pred=96


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 96
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
    
    Epoch [1/50], Train Losses: mse: 0.4696, mae: 0.4788, huber: 0.1995, swd: 0.1114, ept: 39.7289
    Epoch [1/50], Val Losses: mse: 0.3682, mae: 0.4158, huber: 0.1596, swd: 0.1069, ept: 49.7842
    Epoch [1/50], Test Losses: mse: 0.4489, mae: 0.4666, huber: 0.1932, swd: 0.1180, ept: 38.2986
      Epoch 1 composite train-obj: 0.199483
            Val objective improved inf → 0.1596, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3992, mae: 0.4389, huber: 0.1727, swd: 0.1044, ept: 44.2468
    Epoch [2/50], Val Losses: mse: 0.3567, mae: 0.4086, huber: 0.1554, swd: 0.1046, ept: 49.9617
    Epoch [2/50], Test Losses: mse: 0.4397, mae: 0.4634, huber: 0.1908, swd: 0.1279, ept: 38.6541
      Epoch 2 composite train-obj: 0.172735
            Val objective improved 0.1596 → 0.1554, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3758, mae: 0.4279, huber: 0.1647, swd: 0.0984, ept: 44.7465
    Epoch [3/50], Val Losses: mse: 0.3695, mae: 0.4156, huber: 0.1607, swd: 0.1071, ept: 48.6403
    Epoch [3/50], Test Losses: mse: 0.4419, mae: 0.4686, huber: 0.1924, swd: 0.1280, ept: 38.0864
      Epoch 3 composite train-obj: 0.164744
            No improvement (0.1607), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3596, mae: 0.4199, huber: 0.1589, swd: 0.0939, ept: 45.2770
    Epoch [4/50], Val Losses: mse: 0.3894, mae: 0.4271, huber: 0.1686, swd: 0.1049, ept: 48.1181
    Epoch [4/50], Test Losses: mse: 0.4370, mae: 0.4642, huber: 0.1903, swd: 0.1120, ept: 38.6999
      Epoch 4 composite train-obj: 0.158907
            No improvement (0.1686), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3467, mae: 0.4130, huber: 0.1539, swd: 0.0890, ept: 45.6868
    Epoch [5/50], Val Losses: mse: 0.3676, mae: 0.4130, huber: 0.1597, swd: 0.1032, ept: 48.7566
    Epoch [5/50], Test Losses: mse: 0.4322, mae: 0.4629, huber: 0.1894, swd: 0.1355, ept: 38.3211
      Epoch 5 composite train-obj: 0.153947
            No improvement (0.1597), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3361, mae: 0.4070, huber: 0.1498, swd: 0.0859, ept: 46.1521
    Epoch [6/50], Val Losses: mse: 0.3627, mae: 0.4114, huber: 0.1583, swd: 0.1057, ept: 48.2968
    Epoch [6/50], Test Losses: mse: 0.4333, mae: 0.4636, huber: 0.1902, swd: 0.1434, ept: 38.4552
      Epoch 6 composite train-obj: 0.149784
            No improvement (0.1583), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3259, mae: 0.4007, huber: 0.1456, swd: 0.0823, ept: 46.4398
    Epoch [7/50], Val Losses: mse: 0.3593, mae: 0.4082, huber: 0.1567, swd: 0.0977, ept: 48.5569
    Epoch [7/50], Test Losses: mse: 0.4281, mae: 0.4592, huber: 0.1877, swd: 0.1271, ept: 38.2363
      Epoch 7 composite train-obj: 0.145611
    Epoch [7/50], Test Losses: mse: 0.4397, mae: 0.4634, huber: 0.1908, swd: 0.1279, ept: 38.6541
    Best round's Test MSE: 0.4397, MAE: 0.4634, SWD: 0.1279
    Best round's Validation MSE: 0.3567, MAE: 0.4086, SWD: 0.1046
    Best round's Test verification MSE : 0.4397, MAE: 0.4634, SWD: 0.1279
    Time taken: 26.52 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4693, mae: 0.4791, huber: 0.1996, swd: 0.1084, ept: 39.6261
    Epoch [1/50], Val Losses: mse: 0.3627, mae: 0.4138, huber: 0.1577, swd: 0.1077, ept: 49.8287
    Epoch [1/50], Test Losses: mse: 0.4461, mae: 0.4684, huber: 0.1939, swd: 0.1314, ept: 37.6598
      Epoch 1 composite train-obj: 0.199590
            Val objective improved inf → 0.1577, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3979, mae: 0.4386, huber: 0.1724, swd: 0.1006, ept: 44.0275
    Epoch [2/50], Val Losses: mse: 0.3588, mae: 0.4108, huber: 0.1571, swd: 0.1011, ept: 48.8408
    Epoch [2/50], Test Losses: mse: 0.4420, mae: 0.4666, huber: 0.1923, swd: 0.1233, ept: 38.2040
      Epoch 2 composite train-obj: 0.172405
            Val objective improved 0.1577 → 0.1571, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3787, mae: 0.4305, huber: 0.1661, swd: 0.0976, ept: 44.5279
    Epoch [3/50], Val Losses: mse: 0.3726, mae: 0.4165, huber: 0.1620, swd: 0.0999, ept: 48.0286
    Epoch [3/50], Test Losses: mse: 0.4325, mae: 0.4592, huber: 0.1879, swd: 0.1137, ept: 38.9585
      Epoch 3 composite train-obj: 0.166069
            No improvement (0.1620), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3600, mae: 0.4206, huber: 0.1592, swd: 0.0922, ept: 45.3374
    Epoch [4/50], Val Losses: mse: 0.3731, mae: 0.4232, huber: 0.1641, swd: 0.0989, ept: 48.2363
    Epoch [4/50], Test Losses: mse: 0.4415, mae: 0.4707, huber: 0.1939, swd: 0.1255, ept: 37.7735
      Epoch 4 composite train-obj: 0.159153
            No improvement (0.1641), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3455, mae: 0.4132, huber: 0.1537, swd: 0.0873, ept: 45.6020
    Epoch [5/50], Val Losses: mse: 0.3598, mae: 0.4111, huber: 0.1581, swd: 0.1061, ept: 48.8058
    Epoch [5/50], Test Losses: mse: 0.4454, mae: 0.4714, huber: 0.1950, swd: 0.1397, ept: 37.8768
      Epoch 5 composite train-obj: 0.153668
            No improvement (0.1581), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3310, mae: 0.4049, huber: 0.1480, swd: 0.0823, ept: 46.0925
    Epoch [6/50], Val Losses: mse: 0.3734, mae: 0.4193, huber: 0.1634, swd: 0.0978, ept: 48.0282
    Epoch [6/50], Test Losses: mse: 0.4493, mae: 0.4744, huber: 0.1966, swd: 0.1220, ept: 37.7327
      Epoch 6 composite train-obj: 0.147993
            No improvement (0.1634), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3210, mae: 0.3987, huber: 0.1439, swd: 0.0783, ept: 46.4926
    Epoch [7/50], Val Losses: mse: 0.3680, mae: 0.4189, huber: 0.1621, swd: 0.1017, ept: 47.9028
    Epoch [7/50], Test Losses: mse: 0.4624, mae: 0.4815, huber: 0.2015, swd: 0.1414, ept: 37.3435
      Epoch 7 composite train-obj: 0.143905
    Epoch [7/50], Test Losses: mse: 0.4420, mae: 0.4666, huber: 0.1923, swd: 0.1233, ept: 38.2040
    Best round's Test MSE: 0.4420, MAE: 0.4666, SWD: 0.1233
    Best round's Validation MSE: 0.3588, MAE: 0.4108, SWD: 0.1011
    Best round's Test verification MSE : 0.4420, MAE: 0.4666, SWD: 0.1233
    Time taken: 26.66 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4720, mae: 0.4807, huber: 0.2006, swd: 0.1038, ept: 39.6083
    Epoch [1/50], Val Losses: mse: 0.3559, mae: 0.4075, huber: 0.1541, swd: 0.0936, ept: 49.6454
    Epoch [1/50], Test Losses: mse: 0.4569, mae: 0.4717, huber: 0.1964, swd: 0.1174, ept: 38.0432
      Epoch 1 composite train-obj: 0.200630
            Val objective improved inf → 0.1541, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4012, mae: 0.4404, huber: 0.1736, swd: 0.0973, ept: 44.0067
    Epoch [2/50], Val Losses: mse: 0.3598, mae: 0.4097, huber: 0.1571, swd: 0.1002, ept: 49.1420
    Epoch [2/50], Test Losses: mse: 0.4324, mae: 0.4607, huber: 0.1886, swd: 0.1150, ept: 37.7700
      Epoch 2 composite train-obj: 0.173649
            No improvement (0.1571), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3794, mae: 0.4303, huber: 0.1662, swd: 0.0928, ept: 44.7642
    Epoch [3/50], Val Losses: mse: 0.3608, mae: 0.4136, huber: 0.1586, swd: 0.1162, ept: 48.7838
    Epoch [3/50], Test Losses: mse: 0.4406, mae: 0.4705, huber: 0.1938, swd: 0.1465, ept: 38.1525
      Epoch 3 composite train-obj: 0.166218
            No improvement (0.1586), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.3599, mae: 0.4210, huber: 0.1594, swd: 0.0882, ept: 45.3096
    Epoch [4/50], Val Losses: mse: 0.3770, mae: 0.4172, huber: 0.1636, swd: 0.1015, ept: 47.9241
    Epoch [4/50], Test Losses: mse: 0.4282, mae: 0.4572, huber: 0.1864, swd: 0.1032, ept: 38.5819
      Epoch 4 composite train-obj: 0.159413
            No improvement (0.1636), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3437, mae: 0.4124, huber: 0.1532, swd: 0.0826, ept: 45.7617
    Epoch [5/50], Val Losses: mse: 0.3677, mae: 0.4172, huber: 0.1614, swd: 0.1103, ept: 48.9557
    Epoch [5/50], Test Losses: mse: 0.4463, mae: 0.4725, huber: 0.1954, swd: 0.1411, ept: 37.5107
      Epoch 5 composite train-obj: 0.153236
            No improvement (0.1614), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3308, mae: 0.4050, huber: 0.1480, swd: 0.0779, ept: 46.2020
    Epoch [6/50], Val Losses: mse: 0.3592, mae: 0.4114, huber: 0.1581, swd: 0.1147, ept: 48.5622
    Epoch [6/50], Test Losses: mse: 0.4370, mae: 0.4675, huber: 0.1920, swd: 0.1418, ept: 38.3509
      Epoch 6 composite train-obj: 0.147996
    Epoch [6/50], Test Losses: mse: 0.4569, mae: 0.4717, huber: 0.1964, swd: 0.1174, ept: 38.0432
    Best round's Test MSE: 0.4569, MAE: 0.4717, SWD: 0.1174
    Best round's Validation MSE: 0.3559, MAE: 0.4075, SWD: 0.0936
    Best round's Test verification MSE : 0.4569, MAE: 0.4717, SWD: 0.1174
    Time taken: 23.57 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq336_pred96_20250510_1924)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4462 ± 0.0076
      mae: 0.4672 ± 0.0035
      huber: 0.1932 ± 0.0024
      swd: 0.1229 ± 0.0043
      ept: 38.3004 ± 0.2586
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3571 ± 0.0013
      mae: 0.4090 ± 0.0014
      huber: 0.1555 ± 0.0012
      swd: 0.0997 ± 0.0046
      ept: 49.4826 ± 0.4719
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 76.82 seconds
    
    Experiment complete: PatchTST_etth1_seq336_pred96_20250510_1924
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.5509, mae: 0.5179, huber: 0.2282, swd: 0.1358, ept: 57.0686
    Epoch [1/50], Val Losses: mse: 0.4153, mae: 0.4462, huber: 0.1787, swd: 0.1149, ept: 68.5913
    Epoch [1/50], Test Losses: mse: 0.5153, mae: 0.5160, huber: 0.2233, swd: 0.1354, ept: 51.7362
      Epoch 1 composite train-obj: 0.228227
            Val objective improved inf → 0.1787, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4785, mae: 0.4812, huber: 0.2021, swd: 0.1288, ept: 63.4942
    Epoch [2/50], Val Losses: mse: 0.4372, mae: 0.4605, huber: 0.1885, swd: 0.1210, ept: 66.8830
    Epoch [2/50], Test Losses: mse: 0.5088, mae: 0.5161, huber: 0.2232, swd: 0.1346, ept: 53.2251
      Epoch 2 composite train-obj: 0.202054
            No improvement (0.1885), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4451, mae: 0.4678, huber: 0.1913, swd: 0.1189, ept: 64.5800
    Epoch [3/50], Val Losses: mse: 0.4384, mae: 0.4619, huber: 0.1892, swd: 0.1131, ept: 65.5840
    Epoch [3/50], Test Losses: mse: 0.5067, mae: 0.5155, huber: 0.2222, swd: 0.1249, ept: 54.5467
      Epoch 3 composite train-obj: 0.191290
            No improvement (0.1892), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4169, mae: 0.4543, huber: 0.1813, swd: 0.1108, ept: 65.4366
    Epoch [4/50], Val Losses: mse: 0.4552, mae: 0.4738, huber: 0.1959, swd: 0.1221, ept: 64.9771
    Epoch [4/50], Test Losses: mse: 0.5082, mae: 0.5169, huber: 0.2227, swd: 0.1450, ept: 53.3508
      Epoch 4 composite train-obj: 0.181256
            No improvement (0.1959), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4021, mae: 0.4471, huber: 0.1759, swd: 0.1039, ept: 66.2077
    Epoch [5/50], Val Losses: mse: 0.4889, mae: 0.4868, huber: 0.2064, swd: 0.1181, ept: 65.0615
    Epoch [5/50], Test Losses: mse: 0.5119, mae: 0.5177, huber: 0.2243, swd: 0.1256, ept: 54.0748
      Epoch 5 composite train-obj: 0.175863
            No improvement (0.2064), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3874, mae: 0.4388, huber: 0.1701, swd: 0.0981, ept: 67.1087
    Epoch [6/50], Val Losses: mse: 0.4843, mae: 0.4791, huber: 0.2030, swd: 0.1164, ept: 65.3585
    Epoch [6/50], Test Losses: mse: 0.4953, mae: 0.5081, huber: 0.2178, swd: 0.1248, ept: 55.2518
      Epoch 6 composite train-obj: 0.170086
    Epoch [6/50], Test Losses: mse: 0.5153, mae: 0.5160, huber: 0.2233, swd: 0.1354, ept: 51.7362
    Best round's Test MSE: 0.5153, MAE: 0.5160, SWD: 0.1354
    Best round's Validation MSE: 0.4153, MAE: 0.4462, SWD: 0.1149
    Best round's Test verification MSE : 0.5153, MAE: 0.5160, SWD: 0.1354
    Time taken: 24.22 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5524, mae: 0.5186, huber: 0.2287, swd: 0.1368, ept: 56.6291
    Epoch [1/50], Val Losses: mse: 0.4463, mae: 0.4656, huber: 0.1911, swd: 0.1192, ept: 66.9099
    Epoch [1/50], Test Losses: mse: 0.5339, mae: 0.5294, huber: 0.2321, swd: 0.1332, ept: 53.8976
      Epoch 1 composite train-obj: 0.228674
            Val objective improved inf → 0.1911, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4772, mae: 0.4807, huber: 0.2015, swd: 0.1307, ept: 63.3069
    Epoch [2/50], Val Losses: mse: 0.4181, mae: 0.4513, huber: 0.1814, swd: 0.1036, ept: 65.7656
    Epoch [2/50], Test Losses: mse: 0.4875, mae: 0.5003, huber: 0.2131, swd: 0.1085, ept: 54.6492
      Epoch 2 composite train-obj: 0.201509
            Val objective improved 0.1911 → 0.1814, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4425, mae: 0.4660, huber: 0.1900, swd: 0.1196, ept: 64.6553
    Epoch [3/50], Val Losses: mse: 0.4163, mae: 0.4503, huber: 0.1808, swd: 0.1113, ept: 66.9369
    Epoch [3/50], Test Losses: mse: 0.4958, mae: 0.5072, huber: 0.2169, swd: 0.1245, ept: 54.4295
      Epoch 3 composite train-obj: 0.189998
            Val objective improved 0.1814 → 0.1808, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4184, mae: 0.4547, huber: 0.1816, swd: 0.1100, ept: 65.6716
    Epoch [4/50], Val Losses: mse: 0.4336, mae: 0.4611, huber: 0.1877, swd: 0.1164, ept: 66.5029
    Epoch [4/50], Test Losses: mse: 0.5097, mae: 0.5208, huber: 0.2244, swd: 0.1366, ept: 53.5317
      Epoch 4 composite train-obj: 0.181618
            No improvement (0.1877), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4000, mae: 0.4455, huber: 0.1748, swd: 0.1008, ept: 66.2758
    Epoch [5/50], Val Losses: mse: 0.4352, mae: 0.4585, huber: 0.1872, swd: 0.1059, ept: 65.7392
    Epoch [5/50], Test Losses: mse: 0.5111, mae: 0.5205, huber: 0.2244, swd: 0.1290, ept: 53.3468
      Epoch 5 composite train-obj: 0.174835
            No improvement (0.1872), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3820, mae: 0.4356, huber: 0.1679, swd: 0.0935, ept: 67.4695
    Epoch [6/50], Val Losses: mse: 0.4344, mae: 0.4581, huber: 0.1868, swd: 0.1032, ept: 64.8738
    Epoch [6/50], Test Losses: mse: 0.5184, mae: 0.5254, huber: 0.2280, swd: 0.1336, ept: 52.8869
      Epoch 6 composite train-obj: 0.167854
            No improvement (0.1868), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3696, mae: 0.4283, huber: 0.1629, swd: 0.0889, ept: 68.2458
    Epoch [7/50], Val Losses: mse: 0.4563, mae: 0.4610, huber: 0.1914, swd: 0.1011, ept: 66.4687
    Epoch [7/50], Test Losses: mse: 0.5100, mae: 0.5160, huber: 0.2230, swd: 0.1155, ept: 54.0386
      Epoch 7 composite train-obj: 0.162855
            No improvement (0.1914), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3572, mae: 0.4204, huber: 0.1577, swd: 0.0839, ept: 68.9826
    Epoch [8/50], Val Losses: mse: 0.4955, mae: 0.4855, huber: 0.2062, swd: 0.1091, ept: 64.3453
    Epoch [8/50], Test Losses: mse: 0.5381, mae: 0.5333, huber: 0.2345, swd: 0.1232, ept: 52.7967
      Epoch 8 composite train-obj: 0.157725
    Epoch [8/50], Test Losses: mse: 0.4958, mae: 0.5072, huber: 0.2169, swd: 0.1245, ept: 54.4295
    Best round's Test MSE: 0.4958, MAE: 0.5072, SWD: 0.1245
    Best round's Validation MSE: 0.4163, MAE: 0.4503, SWD: 0.1113
    Best round's Test verification MSE : 0.4958, MAE: 0.5072, SWD: 0.1245
    Time taken: 30.67 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5466, mae: 0.5153, huber: 0.2266, swd: 0.1232, ept: 56.9968
    Epoch [1/50], Val Losses: mse: 0.4034, mae: 0.4417, huber: 0.1747, swd: 0.0995, ept: 68.2061
    Epoch [1/50], Test Losses: mse: 0.4952, mae: 0.5017, huber: 0.2151, swd: 0.1131, ept: 54.4788
      Epoch 1 composite train-obj: 0.226647
            Val objective improved inf → 0.1747, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4791, mae: 0.4820, huber: 0.2027, swd: 0.1174, ept: 63.4055
    Epoch [2/50], Val Losses: mse: 0.4410, mae: 0.4588, huber: 0.1886, swd: 0.1098, ept: 67.5480
    Epoch [2/50], Test Losses: mse: 0.5144, mae: 0.5181, huber: 0.2240, swd: 0.1164, ept: 55.1639
      Epoch 2 composite train-obj: 0.202656
            No improvement (0.1886), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4513, mae: 0.4694, huber: 0.1931, swd: 0.1112, ept: 64.6413
    Epoch [3/50], Val Losses: mse: 0.4250, mae: 0.4606, huber: 0.1858, swd: 0.1168, ept: 66.1382
    Epoch [3/50], Test Losses: mse: 0.4936, mae: 0.5091, huber: 0.2171, swd: 0.1339, ept: 55.1304
      Epoch 3 composite train-obj: 0.193072
            No improvement (0.1858), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4268, mae: 0.4588, huber: 0.1848, swd: 0.1031, ept: 65.5472
    Epoch [4/50], Val Losses: mse: 0.4482, mae: 0.4636, huber: 0.1917, swd: 0.1101, ept: 65.2044
    Epoch [4/50], Test Losses: mse: 0.4940, mae: 0.5077, huber: 0.2163, swd: 0.1151, ept: 53.8478
      Epoch 4 composite train-obj: 0.184808
            No improvement (0.1917), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4043, mae: 0.4475, huber: 0.1765, swd: 0.0951, ept: 66.5581
    Epoch [5/50], Val Losses: mse: 0.4373, mae: 0.4568, huber: 0.1870, swd: 0.1095, ept: 65.3896
    Epoch [5/50], Test Losses: mse: 0.4888, mae: 0.5057, huber: 0.2150, swd: 0.1216, ept: 54.5221
      Epoch 5 composite train-obj: 0.176501
            No improvement (0.1870), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3870, mae: 0.4381, huber: 0.1699, swd: 0.0895, ept: 67.7756
    Epoch [6/50], Val Losses: mse: 0.4390, mae: 0.4588, huber: 0.1880, swd: 0.1192, ept: 67.7672
    Epoch [6/50], Test Losses: mse: 0.5027, mae: 0.5170, huber: 0.2215, swd: 0.1417, ept: 54.5689
      Epoch 6 composite train-obj: 0.169910
    Epoch [6/50], Test Losses: mse: 0.4952, mae: 0.5017, huber: 0.2151, swd: 0.1131, ept: 54.4788
    Best round's Test MSE: 0.4952, MAE: 0.5017, SWD: 0.1131
    Best round's Validation MSE: 0.4034, MAE: 0.4417, SWD: 0.0995
    Best round's Test verification MSE : 0.4952, MAE: 0.5017, SWD: 0.1131
    Time taken: 22.94 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq336_pred196_20250510_1922)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5021 ± 0.0093
      mae: 0.5083 ± 0.0059
      huber: 0.2184 ± 0.0035
      swd: 0.1243 ± 0.0091
      ept: 53.5482 ± 1.2814
      count: 10.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4117 ± 0.0059
      mae: 0.4461 ± 0.0035
      huber: 0.1781 ± 0.0025
      swd: 0.1086 ± 0.0066
      ept: 67.9114 ± 0.7068
      count: 10.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 77.89 seconds
    
    Experiment complete: PatchTST_etth1_seq336_pred196_20250510_1922
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 91
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 91
    Validation Batches: 9
    Test Batches: 22
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6347, mae: 0.5567, huber: 0.2572, swd: 0.1510, ept: 69.2659
    Epoch [1/50], Val Losses: mse: 0.5315, mae: 0.5138, huber: 0.2240, swd: 0.0758, ept: 78.7245
    Epoch [1/50], Test Losses: mse: 0.6996, mae: 0.6195, huber: 0.2972, swd: 0.0972, ept: 64.6731
      Epoch 1 composite train-obj: 0.257203
            Val objective improved inf → 0.2240, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5692, mae: 0.5277, huber: 0.2348, swd: 0.1465, ept: 77.1505
    Epoch [2/50], Val Losses: mse: 0.4873, mae: 0.4978, huber: 0.2093, swd: 0.0863, ept: 77.9514
    Epoch [2/50], Test Losses: mse: 0.5948, mae: 0.5653, huber: 0.2577, swd: 0.1211, ept: 68.5841
      Epoch 2 composite train-obj: 0.234836
            Val objective improved 0.2240 → 0.2093, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5129, mae: 0.5030, huber: 0.2157, swd: 0.1354, ept: 80.2905
    Epoch [3/50], Val Losses: mse: 0.5148, mae: 0.5146, huber: 0.2208, swd: 0.0899, ept: 76.0888
    Epoch [3/50], Test Losses: mse: 0.6291, mae: 0.5875, huber: 0.2725, swd: 0.1176, ept: 67.1818
      Epoch 3 composite train-obj: 0.215684
            No improvement (0.2208), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4877, mae: 0.4914, huber: 0.2066, swd: 0.1266, ept: 81.0983
    Epoch [4/50], Val Losses: mse: 0.5643, mae: 0.5195, huber: 0.2293, swd: 0.0980, ept: 79.4510
    Epoch [4/50], Test Losses: mse: 0.5984, mae: 0.5663, huber: 0.2577, swd: 0.0871, ept: 68.6339
      Epoch 4 composite train-obj: 0.206633
            No improvement (0.2293), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4599, mae: 0.4778, huber: 0.1969, swd: 0.1159, ept: 82.6349
    Epoch [5/50], Val Losses: mse: 0.4775, mae: 0.4874, huber: 0.2040, swd: 0.0884, ept: 81.9258
    Epoch [5/50], Test Losses: mse: 0.5971, mae: 0.5654, huber: 0.2586, swd: 0.1139, ept: 70.2713
      Epoch 5 composite train-obj: 0.196891
            Val objective improved 0.2093 → 0.2040, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4471, mae: 0.4710, huber: 0.1921, swd: 0.1110, ept: 83.4760
    Epoch [6/50], Val Losses: mse: 0.5026, mae: 0.4929, huber: 0.2106, swd: 0.0926, ept: 81.2759
    Epoch [6/50], Test Losses: mse: 0.5792, mae: 0.5598, huber: 0.2527, swd: 0.1098, ept: 68.6764
      Epoch 6 composite train-obj: 0.192078
            No improvement (0.2106), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4315, mae: 0.4622, huber: 0.1860, swd: 0.1041, ept: 84.8724
    Epoch [7/50], Val Losses: mse: 0.5318, mae: 0.5019, huber: 0.2181, swd: 0.0869, ept: 81.8148
    Epoch [7/50], Test Losses: mse: 0.5920, mae: 0.5631, huber: 0.2564, swd: 0.1061, ept: 70.7275
      Epoch 7 composite train-obj: 0.186041
            No improvement (0.2181), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4212, mae: 0.4571, huber: 0.1823, swd: 0.0986, ept: 84.5468
    Epoch [8/50], Val Losses: mse: 0.4837, mae: 0.4824, huber: 0.2035, swd: 0.0922, ept: 83.0561
    Epoch [8/50], Test Losses: mse: 0.5751, mae: 0.5568, huber: 0.2513, swd: 0.1355, ept: 70.7677
      Epoch 8 composite train-obj: 0.182272
            Val objective improved 0.2040 → 0.2035, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4001, mae: 0.4452, huber: 0.1742, swd: 0.0923, ept: 86.9366
    Epoch [9/50], Val Losses: mse: 0.4988, mae: 0.4854, huber: 0.2069, swd: 0.0920, ept: 83.9550
    Epoch [9/50], Test Losses: mse: 0.5983, mae: 0.5696, huber: 0.2605, swd: 0.1241, ept: 69.9759
      Epoch 9 composite train-obj: 0.174173
            No improvement (0.2069), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.3900, mae: 0.4394, huber: 0.1702, swd: 0.0866, ept: 87.9433
    Epoch [10/50], Val Losses: mse: 0.5213, mae: 0.5114, huber: 0.2206, swd: 0.1052, ept: 81.7449
    Epoch [10/50], Test Losses: mse: 0.6456, mae: 0.5970, huber: 0.2793, swd: 0.1585, ept: 69.0424
      Epoch 10 composite train-obj: 0.170170
            No improvement (0.2206), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.3779, mae: 0.4326, huber: 0.1655, swd: 0.0830, ept: 89.0257
    Epoch [11/50], Val Losses: mse: 0.5169, mae: 0.5006, huber: 0.2155, swd: 0.1008, ept: 81.8626
    Epoch [11/50], Test Losses: mse: 0.5984, mae: 0.5693, huber: 0.2598, swd: 0.1208, ept: 71.0357
      Epoch 11 composite train-obj: 0.165520
            No improvement (0.2155), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.3684, mae: 0.4277, huber: 0.1620, swd: 0.0785, ept: 89.2540
    Epoch [12/50], Val Losses: mse: 0.5554, mae: 0.5127, huber: 0.2255, swd: 0.0942, ept: 81.7048
    Epoch [12/50], Test Losses: mse: 0.5852, mae: 0.5630, huber: 0.2544, swd: 0.1137, ept: 71.2418
      Epoch 12 composite train-obj: 0.162022
            No improvement (0.2255), counter 4/5
    Epoch [13/50], Train Losses: mse: 0.3603, mae: 0.4232, huber: 0.1590, swd: 0.0751, ept: 89.9662
    Epoch [13/50], Val Losses: mse: 0.5355, mae: 0.4995, huber: 0.2178, swd: 0.0989, ept: 83.3337
    Epoch [13/50], Test Losses: mse: 0.6035, mae: 0.5734, huber: 0.2621, swd: 0.1185, ept: 69.0597
      Epoch 13 composite train-obj: 0.158950
    Epoch [13/50], Test Losses: mse: 0.5751, mae: 0.5568, huber: 0.2513, swd: 0.1355, ept: 70.7677
    Best round's Test MSE: 0.5751, MAE: 0.5568, SWD: 0.1355
    Best round's Validation MSE: 0.4837, MAE: 0.4824, SWD: 0.0922
    Best round's Test verification MSE : 0.5751, MAE: 0.5568, SWD: 0.1355
    Time taken: 51.42 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6242, mae: 0.5526, huber: 0.2540, swd: 0.1548, ept: 70.0083
    Epoch [1/50], Val Losses: mse: 0.4717, mae: 0.4911, huber: 0.2044, swd: 0.0824, ept: 77.9613
    Epoch [1/50], Test Losses: mse: 0.6044, mae: 0.5688, huber: 0.2603, swd: 0.1136, ept: 66.3020
      Epoch 1 composite train-obj: 0.254032
            Val objective improved inf → 0.2044, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5491, mae: 0.5186, huber: 0.2281, swd: 0.1474, ept: 77.4974
    Epoch [2/50], Val Losses: mse: 0.4793, mae: 0.4980, huber: 0.2082, swd: 0.0947, ept: 78.6864
    Epoch [2/50], Test Losses: mse: 0.5979, mae: 0.5676, huber: 0.2586, swd: 0.1193, ept: 67.9079
      Epoch 2 composite train-obj: 0.228092
            No improvement (0.2082), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5079, mae: 0.5013, huber: 0.2143, swd: 0.1339, ept: 79.7274
    Epoch [3/50], Val Losses: mse: 0.4712, mae: 0.4853, huber: 0.2026, swd: 0.0976, ept: 83.1375
    Epoch [3/50], Test Losses: mse: 0.6575, mae: 0.6061, huber: 0.2844, swd: 0.1272, ept: 67.6947
      Epoch 3 composite train-obj: 0.214325
            Val objective improved 0.2044 → 0.2026, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4787, mae: 0.4878, huber: 0.2039, swd: 0.1246, ept: 81.2404
    Epoch [4/50], Val Losses: mse: 0.4873, mae: 0.4930, huber: 0.2079, swd: 0.1025, ept: 82.8130
    Epoch [4/50], Test Losses: mse: 0.6383, mae: 0.5984, huber: 0.2779, swd: 0.1343, ept: 68.8972
      Epoch 4 composite train-obj: 0.203936
            No improvement (0.2079), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4605, mae: 0.4788, huber: 0.1973, swd: 0.1179, ept: 82.0974
    Epoch [5/50], Val Losses: mse: 0.5382, mae: 0.5156, huber: 0.2257, swd: 0.1057, ept: 80.4352
    Epoch [5/50], Test Losses: mse: 0.6167, mae: 0.5746, huber: 0.2649, swd: 0.1113, ept: 70.8546
      Epoch 5 composite train-obj: 0.197285
            No improvement (0.2257), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4373, mae: 0.4661, huber: 0.1885, swd: 0.1095, ept: 84.0357
    Epoch [6/50], Val Losses: mse: 0.4406, mae: 0.4665, huber: 0.1911, swd: 0.0912, ept: 83.2291
    Epoch [6/50], Test Losses: mse: 0.5726, mae: 0.5508, huber: 0.2485, swd: 0.1212, ept: 72.2297
      Epoch 6 composite train-obj: 0.188486
            Val objective improved 0.2026 → 0.1911, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4214, mae: 0.4578, huber: 0.1826, swd: 0.1029, ept: 85.0397
    Epoch [7/50], Val Losses: mse: 0.4674, mae: 0.4749, huber: 0.1984, swd: 0.1064, ept: 83.7017
    Epoch [7/50], Test Losses: mse: 0.6042, mae: 0.5691, huber: 0.2616, swd: 0.1277, ept: 69.3615
      Epoch 7 composite train-obj: 0.182595
            No improvement (0.1984), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4102, mae: 0.4517, huber: 0.1783, swd: 0.0970, ept: 85.3594
    Epoch [8/50], Val Losses: mse: 0.4433, mae: 0.4654, huber: 0.1911, swd: 0.1052, ept: 86.1343
    Epoch [8/50], Test Losses: mse: 0.6377, mae: 0.5932, huber: 0.2764, swd: 0.1452, ept: 68.8612
      Epoch 8 composite train-obj: 0.178269
            No improvement (0.1911), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.4000, mae: 0.4461, huber: 0.1744, swd: 0.0922, ept: 86.1981
    Epoch [9/50], Val Losses: mse: 0.5463, mae: 0.4992, huber: 0.2193, swd: 0.1057, ept: 82.4154
    Epoch [9/50], Test Losses: mse: 0.6078, mae: 0.5660, huber: 0.2599, swd: 0.0945, ept: 71.7877
      Epoch 9 composite train-obj: 0.174369
            No improvement (0.2193), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.3855, mae: 0.4378, huber: 0.1688, swd: 0.0859, ept: 87.9683
    Epoch [10/50], Val Losses: mse: 0.4620, mae: 0.4690, huber: 0.1949, swd: 0.0914, ept: 84.2333
    Epoch [10/50], Test Losses: mse: 0.5886, mae: 0.5625, huber: 0.2555, swd: 0.1267, ept: 70.1818
      Epoch 10 composite train-obj: 0.168778
            No improvement (0.1949), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.3748, mae: 0.4313, huber: 0.1645, swd: 0.0824, ept: 89.3080
    Epoch [11/50], Val Losses: mse: 0.4950, mae: 0.4843, huber: 0.2059, swd: 0.0992, ept: 84.0599
    Epoch [11/50], Test Losses: mse: 0.6047, mae: 0.5710, huber: 0.2617, swd: 0.1251, ept: 71.6254
      Epoch 11 composite train-obj: 0.164508
    Epoch [11/50], Test Losses: mse: 0.5726, mae: 0.5508, huber: 0.2485, swd: 0.1212, ept: 72.2297
    Best round's Test MSE: 0.5726, MAE: 0.5508, SWD: 0.1212
    Best round's Validation MSE: 0.4406, MAE: 0.4665, SWD: 0.0912
    Best round's Test verification MSE : 0.5726, MAE: 0.5508, SWD: 0.1212
    Time taken: 43.69 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6312, mae: 0.5560, huber: 0.2568, swd: 0.1505, ept: 69.5623
    Epoch [1/50], Val Losses: mse: 0.4437, mae: 0.4692, huber: 0.1914, swd: 0.0894, ept: 82.5348
    Epoch [1/50], Test Losses: mse: 0.7703, mae: 0.6495, huber: 0.3217, swd: 0.1313, ept: 60.2409
      Epoch 1 composite train-obj: 0.256759
            Val objective improved inf → 0.1914, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5976, mae: 0.5381, huber: 0.2434, swd: 0.1534, ept: 74.3884
    Epoch [2/50], Val Losses: mse: 0.4526, mae: 0.4737, huber: 0.1945, swd: 0.0875, ept: 78.2543
    Epoch [2/50], Test Losses: mse: 0.6296, mae: 0.5833, huber: 0.2703, swd: 0.1114, ept: 61.1915
      Epoch 2 composite train-obj: 0.243448
            No improvement (0.1945), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5236, mae: 0.5066, huber: 0.2193, swd: 0.1369, ept: 79.2711
    Epoch [3/50], Val Losses: mse: 0.4524, mae: 0.4880, huber: 0.1988, swd: 0.0827, ept: 73.7915
    Epoch [3/50], Test Losses: mse: 0.6151, mae: 0.5763, huber: 0.2644, swd: 0.1088, ept: 60.5467
      Epoch 3 composite train-obj: 0.219307
            No improvement (0.1988), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4988, mae: 0.4955, huber: 0.2107, swd: 0.1297, ept: 80.3003
    Epoch [4/50], Val Losses: mse: 0.4576, mae: 0.4820, huber: 0.1985, swd: 0.0995, ept: 76.6202
    Epoch [4/50], Test Losses: mse: 0.6023, mae: 0.5693, huber: 0.2601, swd: 0.1180, ept: 67.3014
      Epoch 4 composite train-obj: 0.210661
            No improvement (0.1985), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4781, mae: 0.4849, huber: 0.2028, swd: 0.1234, ept: 81.1608
    Epoch [5/50], Val Losses: mse: 0.4330, mae: 0.4590, huber: 0.1863, swd: 0.0874, ept: 81.1681
    Epoch [5/50], Test Losses: mse: 0.5865, mae: 0.5578, huber: 0.2536, swd: 0.1144, ept: 70.6203
      Epoch 5 composite train-obj: 0.202849
            Val objective improved 0.1914 → 0.1863, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4538, mae: 0.4741, huber: 0.1945, swd: 0.1148, ept: 83.4327
    Epoch [6/50], Val Losses: mse: 0.4549, mae: 0.4671, huber: 0.1932, swd: 0.0975, ept: 83.0701
    Epoch [6/50], Test Losses: mse: 0.6048, mae: 0.5753, huber: 0.2631, swd: 0.1370, ept: 68.9034
      Epoch 6 composite train-obj: 0.194456
            No improvement (0.1932), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4434, mae: 0.4681, huber: 0.1904, swd: 0.1111, ept: 84.0636
    Epoch [7/50], Val Losses: mse: 0.4604, mae: 0.4709, huber: 0.1954, swd: 0.0912, ept: 80.5958
    Epoch [7/50], Test Losses: mse: 0.5846, mae: 0.5607, huber: 0.2542, swd: 0.1237, ept: 67.8072
      Epoch 7 composite train-obj: 0.190379
            No improvement (0.1954), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4281, mae: 0.4605, huber: 0.1848, swd: 0.1026, ept: 84.1427
    Epoch [8/50], Val Losses: mse: 0.4906, mae: 0.4826, huber: 0.2045, swd: 0.0944, ept: 79.9262
    Epoch [8/50], Test Losses: mse: 0.5907, mae: 0.5637, huber: 0.2564, swd: 0.1244, ept: 70.2510
      Epoch 8 composite train-obj: 0.184826
            No improvement (0.2045), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.4159, mae: 0.4540, huber: 0.1803, swd: 0.0977, ept: 85.1681
    Epoch [9/50], Val Losses: mse: 0.4957, mae: 0.4893, huber: 0.2083, swd: 0.0943, ept: 76.2512
    Epoch [9/50], Test Losses: mse: 0.6059, mae: 0.5721, huber: 0.2626, swd: 0.1249, ept: 70.3486
      Epoch 9 composite train-obj: 0.180309
            No improvement (0.2083), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.4225, mae: 0.4576, huber: 0.1827, swd: 0.0991, ept: 85.0671
    Epoch [10/50], Val Losses: mse: 0.4496, mae: 0.4672, huber: 0.1923, swd: 0.0846, ept: 80.9465
    Epoch [10/50], Test Losses: mse: 0.5895, mae: 0.5595, huber: 0.2552, swd: 0.1212, ept: 70.8833
      Epoch 10 composite train-obj: 0.182689
    Epoch [10/50], Test Losses: mse: 0.5865, mae: 0.5578, huber: 0.2536, swd: 0.1144, ept: 70.6203
    Best round's Test MSE: 0.5865, MAE: 0.5578, SWD: 0.1144
    Best round's Validation MSE: 0.4330, MAE: 0.4590, SWD: 0.0874
    Best round's Test verification MSE : 0.5865, MAE: 0.5578, SWD: 0.1144
    Time taken: 39.64 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq336_pred336_20250510_1920)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5781 ± 0.0061
      mae: 0.5551 ± 0.0031
      huber: 0.2511 ± 0.0021
      swd: 0.1237 ± 0.0088
      ept: 71.2059 ± 0.7265
      count: 9.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4524 ± 0.0223
      mae: 0.4693 ± 0.0098
      huber: 0.1936 ± 0.0072
      swd: 0.0902 ± 0.0020
      ept: 82.4844 ± 0.9334
      count: 9.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 134.84 seconds
    
    Experiment complete: PatchTST_etth1_seq336_pred336_20250510_1920
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 88
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 720
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
    
    Epoch [1/50], Train Losses: mse: 0.7673, mae: 0.6233, huber: 0.3058, swd: 0.1881, ept: 78.9399
    Epoch [1/50], Val Losses: mse: 0.7454, mae: 0.6066, huber: 0.2983, swd: 0.0777, ept: 123.6847
    Epoch [1/50], Test Losses: mse: 0.8658, mae: 0.7110, huber: 0.3641, swd: 0.1330, ept: 84.6209
      Epoch 1 composite train-obj: 0.305753
            Val objective improved inf → 0.2983, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6692, mae: 0.5826, huber: 0.2730, swd: 0.1794, ept: 90.8022
    Epoch [2/50], Val Losses: mse: 0.9654, mae: 0.6788, huber: 0.3597, swd: 0.1174, ept: 123.7338
    Epoch [2/50], Test Losses: mse: 0.9493, mae: 0.7354, huber: 0.3874, swd: 0.1295, ept: 84.1132
      Epoch 2 composite train-obj: 0.273008
            No improvement (0.3597), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.6259, mae: 0.5631, huber: 0.2578, swd: 0.1698, ept: 93.8891
    Epoch [3/50], Val Losses: mse: 0.5602, mae: 0.5286, huber: 0.2346, swd: 0.0624, ept: 124.2199
    Epoch [3/50], Test Losses: mse: 0.7813, mae: 0.6666, huber: 0.3307, swd: 0.1517, ept: 91.8817
      Epoch 3 composite train-obj: 0.257804
            Val objective improved 0.2983 → 0.2346, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5931, mae: 0.5481, huber: 0.2463, swd: 0.1618, ept: 96.7009
    Epoch [4/50], Val Losses: mse: 0.5936, mae: 0.5457, huber: 0.2461, swd: 0.0615, ept: 113.3735
    Epoch [4/50], Test Losses: mse: 0.7628, mae: 0.6589, huber: 0.3249, swd: 0.1335, ept: 95.4850
      Epoch 4 composite train-obj: 0.246328
            No improvement (0.2461), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5655, mae: 0.5344, huber: 0.2361, swd: 0.1519, ept: 98.6365
    Epoch [5/50], Val Losses: mse: 0.6401, mae: 0.5539, huber: 0.2560, swd: 0.0678, ept: 114.5400
    Epoch [5/50], Test Losses: mse: 0.7931, mae: 0.6702, huber: 0.3347, swd: 0.1104, ept: 93.5216
      Epoch 5 composite train-obj: 0.236107
            No improvement (0.2560), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5450, mae: 0.5247, huber: 0.2288, swd: 0.1425, ept: 100.7885
    Epoch [6/50], Val Losses: mse: 0.7429, mae: 0.5928, huber: 0.2879, swd: 0.0982, ept: 123.9475
    Epoch [6/50], Test Losses: mse: 0.8076, mae: 0.6794, huber: 0.3401, swd: 0.1190, ept: 94.5243
      Epoch 6 composite train-obj: 0.228839
            No improvement (0.2879), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5244, mae: 0.5135, huber: 0.2210, swd: 0.1340, ept: 103.7591
    Epoch [7/50], Val Losses: mse: 0.5802, mae: 0.5261, huber: 0.2368, swd: 0.0701, ept: 133.5627
    Epoch [7/50], Test Losses: mse: 0.8556, mae: 0.7052, huber: 0.3596, swd: 0.1412, ept: 81.3050
      Epoch 7 composite train-obj: 0.221004
            No improvement (0.2368), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.5286, mae: 0.5167, huber: 0.2230, swd: 0.1307, ept: 102.5158
    Epoch [8/50], Val Losses: mse: 0.6430, mae: 0.5449, huber: 0.2526, swd: 0.0770, ept: 113.3521
    Epoch [8/50], Test Losses: mse: 0.7771, mae: 0.6641, huber: 0.3294, swd: 0.1089, ept: 101.8706
      Epoch 8 composite train-obj: 0.223022
    Epoch [8/50], Test Losses: mse: 0.7813, mae: 0.6666, huber: 0.3307, swd: 0.1517, ept: 91.8817
    Best round's Test MSE: 0.7813, MAE: 0.6666, SWD: 0.1517
    Best round's Validation MSE: 0.5602, MAE: 0.5286, SWD: 0.0624
    Best round's Test verification MSE : 0.7813, MAE: 0.6666, SWD: 0.1517
    Time taken: 31.78 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7671, mae: 0.6234, huber: 0.3059, swd: 0.1835, ept: 80.7451
    Epoch [1/50], Val Losses: mse: 0.5617, mae: 0.5333, huber: 0.2378, swd: 0.0558, ept: 128.2231
    Epoch [1/50], Test Losses: mse: 0.7702, mae: 0.6635, huber: 0.3278, swd: 0.1185, ept: 88.1786
      Epoch 1 composite train-obj: 0.305861
            Val objective improved inf → 0.2378, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6715, mae: 0.5830, huber: 0.2735, swd: 0.1760, ept: 91.6975
    Epoch [2/50], Val Losses: mse: 0.5631, mae: 0.5316, huber: 0.2360, swd: 0.0666, ept: 126.8754
    Epoch [2/50], Test Losses: mse: 0.7508, mae: 0.6523, huber: 0.3194, swd: 0.1223, ept: 94.4876
      Epoch 2 composite train-obj: 0.273502
            Val objective improved 0.2378 → 0.2360, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6087, mae: 0.5553, huber: 0.2519, swd: 0.1650, ept: 96.4176
    Epoch [3/50], Val Losses: mse: 0.6700, mae: 0.5824, huber: 0.2755, swd: 0.0795, ept: 123.9967
    Epoch [3/50], Test Losses: mse: 0.8308, mae: 0.6922, huber: 0.3496, swd: 0.1407, ept: 97.2138
      Epoch 3 composite train-obj: 0.251882
            No improvement (0.2755), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5833, mae: 0.5429, huber: 0.2427, swd: 0.1553, ept: 98.3711
    Epoch [4/50], Val Losses: mse: 0.6822, mae: 0.5755, huber: 0.2721, swd: 0.0761, ept: 124.2070
    Epoch [4/50], Test Losses: mse: 0.8751, mae: 0.6946, huber: 0.3580, swd: 0.0970, ept: 92.0722
      Epoch 4 composite train-obj: 0.242663
            No improvement (0.2721), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5569, mae: 0.5305, huber: 0.2333, swd: 0.1435, ept: 100.9357
    Epoch [5/50], Val Losses: mse: 0.5442, mae: 0.5110, huber: 0.2247, swd: 0.0643, ept: 128.0314
    Epoch [5/50], Test Losses: mse: 0.7730, mae: 0.6705, huber: 0.3309, swd: 0.1197, ept: 94.6522
      Epoch 5 composite train-obj: 0.233284
            Val objective improved 0.2360 → 0.2247, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5337, mae: 0.5184, huber: 0.2246, swd: 0.1340, ept: 102.6459
    Epoch [6/50], Val Losses: mse: 0.5960, mae: 0.5313, huber: 0.2403, swd: 0.0648, ept: 121.9805
    Epoch [6/50], Test Losses: mse: 0.8135, mae: 0.6779, huber: 0.3418, swd: 0.1163, ept: 96.4764
      Epoch 6 composite train-obj: 0.224600
            No improvement (0.2403), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.5183, mae: 0.5115, huber: 0.2193, swd: 0.1243, ept: 103.6087
    Epoch [7/50], Val Losses: mse: 0.5939, mae: 0.5306, huber: 0.2399, swd: 0.0646, ept: 124.3807
    Epoch [7/50], Test Losses: mse: 0.8200, mae: 0.6775, huber: 0.3401, swd: 0.1152, ept: 94.9765
      Epoch 7 composite train-obj: 0.219301
            No improvement (0.2399), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4988, mae: 0.5011, huber: 0.2121, swd: 0.1169, ept: 105.3738
    Epoch [8/50], Val Losses: mse: 0.5997, mae: 0.5338, huber: 0.2430, swd: 0.0695, ept: 128.3046
    Epoch [8/50], Test Losses: mse: 0.8651, mae: 0.6929, huber: 0.3541, swd: 0.1316, ept: 92.8192
      Epoch 8 composite train-obj: 0.212068
            No improvement (0.2430), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.4864, mae: 0.4955, huber: 0.2078, swd: 0.1097, ept: 106.1960
    Epoch [9/50], Val Losses: mse: 0.6416, mae: 0.5383, huber: 0.2491, swd: 0.0795, ept: 118.3737
    Epoch [9/50], Test Losses: mse: 0.8469, mae: 0.6819, huber: 0.3457, swd: 0.1147, ept: 97.3281
      Epoch 9 composite train-obj: 0.207771
            No improvement (0.2491), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.4973, mae: 0.5016, huber: 0.2121, swd: 0.1147, ept: 104.4026
    Epoch [10/50], Val Losses: mse: 0.5403, mae: 0.5037, huber: 0.2214, swd: 0.0676, ept: 127.0910
    Epoch [10/50], Test Losses: mse: 0.9049, mae: 0.7005, huber: 0.3625, swd: 0.1295, ept: 92.0618
      Epoch 10 composite train-obj: 0.212143
            Val objective improved 0.2247 → 0.2214, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.4669, mae: 0.4853, huber: 0.2007, swd: 0.1038, ept: 107.6468
    Epoch [11/50], Val Losses: mse: 0.6009, mae: 0.5187, huber: 0.2360, swd: 0.0729, ept: 125.4368
    Epoch [11/50], Test Losses: mse: 0.8626, mae: 0.6843, huber: 0.3480, swd: 0.1209, ept: 95.7995
      Epoch 11 composite train-obj: 0.200725
            No improvement (0.2360), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.4555, mae: 0.4796, huber: 0.1965, swd: 0.0968, ept: 108.7306
    Epoch [12/50], Val Losses: mse: 0.6815, mae: 0.5502, huber: 0.2586, swd: 0.0922, ept: 115.2081
    Epoch [12/50], Test Losses: mse: 0.9179, mae: 0.7033, huber: 0.3650, swd: 0.1105, ept: 96.1663
      Epoch 12 composite train-obj: 0.196548
            No improvement (0.2586), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.4508, mae: 0.4760, huber: 0.1943, swd: 0.0969, ept: 108.8108
    Epoch [13/50], Val Losses: mse: 0.5810, mae: 0.5166, huber: 0.2324, swd: 0.0777, ept: 118.4741
    Epoch [13/50], Test Losses: mse: 0.8571, mae: 0.6816, huber: 0.3475, swd: 0.1240, ept: 99.0588
      Epoch 13 composite train-obj: 0.194261
            No improvement (0.2324), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.4305, mae: 0.4660, huber: 0.1871, swd: 0.0875, ept: 109.9157
    Epoch [14/50], Val Losses: mse: 0.5557, mae: 0.5114, huber: 0.2260, swd: 0.0702, ept: 111.5033
    Epoch [14/50], Test Losses: mse: 0.7911, mae: 0.6600, huber: 0.3278, swd: 0.1246, ept: 97.3618
      Epoch 14 composite train-obj: 0.187112
            No improvement (0.2260), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.4205, mae: 0.4608, huber: 0.1834, swd: 0.0833, ept: 110.6067
    Epoch [15/50], Val Losses: mse: 0.5466, mae: 0.5053, huber: 0.2234, swd: 0.0755, ept: 126.0719
    Epoch [15/50], Test Losses: mse: 0.9065, mae: 0.6989, huber: 0.3591, swd: 0.1589, ept: 93.4722
      Epoch 15 composite train-obj: 0.183395
    Epoch [15/50], Test Losses: mse: 0.9049, mae: 0.7005, huber: 0.3625, swd: 0.1295, ept: 92.0618
    Best round's Test MSE: 0.9049, MAE: 0.7005, SWD: 0.1295
    Best round's Validation MSE: 0.5403, MAE: 0.5037, SWD: 0.0676
    Best round's Test verification MSE : 0.9049, MAE: 0.7005, SWD: 0.1295
    Time taken: 58.59 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7737, mae: 0.6247, huber: 0.3073, swd: 0.2025, ept: 80.3087
    Epoch [1/50], Val Losses: mse: 0.6168, mae: 0.5620, huber: 0.2593, swd: 0.0709, ept: 130.3359
    Epoch [1/50], Test Losses: mse: 0.8041, mae: 0.6824, huber: 0.3418, swd: 0.1305, ept: 81.4249
      Epoch 1 composite train-obj: 0.307270
            Val objective improved inf → 0.2593, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6743, mae: 0.5835, huber: 0.2741, swd: 0.1923, ept: 90.9584
    Epoch [2/50], Val Losses: mse: 0.6557, mae: 0.5665, huber: 0.2658, swd: 0.0726, ept: 124.6584
    Epoch [2/50], Test Losses: mse: 0.7986, mae: 0.6755, huber: 0.3372, swd: 0.1184, ept: 80.1840
      Epoch 2 composite train-obj: 0.274064
            No improvement (0.2658), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.6249, mae: 0.5633, huber: 0.2577, swd: 0.1776, ept: 94.1006
    Epoch [3/50], Val Losses: mse: 0.7166, mae: 0.5868, huber: 0.2817, swd: 0.0790, ept: 122.1334
    Epoch [3/50], Test Losses: mse: 0.7992, mae: 0.6755, huber: 0.3370, swd: 0.1241, ept: 90.2472
      Epoch 3 composite train-obj: 0.257683
            No improvement (0.2817), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.5896, mae: 0.5459, huber: 0.2446, swd: 0.1695, ept: 98.4324
    Epoch [4/50], Val Losses: mse: 0.8644, mae: 0.6434, huber: 0.3266, swd: 0.1211, ept: 124.0395
    Epoch [4/50], Test Losses: mse: 0.8782, mae: 0.7024, huber: 0.3619, swd: 0.1072, ept: 89.6476
      Epoch 4 composite train-obj: 0.244636
            No improvement (0.3266), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.5674, mae: 0.5348, huber: 0.2364, swd: 0.1589, ept: 100.0924
    Epoch [5/50], Val Losses: mse: 0.7007, mae: 0.5670, huber: 0.2698, swd: 0.0907, ept: 133.9439
    Epoch [5/50], Test Losses: mse: 0.8079, mae: 0.6769, huber: 0.3393, swd: 0.1182, ept: 94.5730
      Epoch 5 composite train-obj: 0.236417
            No improvement (0.2698), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.5404, mae: 0.5226, huber: 0.2272, swd: 0.1458, ept: 101.2302
    Epoch [6/50], Val Losses: mse: 0.6739, mae: 0.5604, huber: 0.2630, swd: 0.0810, ept: 133.4266
    Epoch [6/50], Test Losses: mse: 0.8074, mae: 0.6791, huber: 0.3405, swd: 0.1238, ept: 84.4568
      Epoch 6 composite train-obj: 0.227172
    Epoch [6/50], Test Losses: mse: 0.8041, mae: 0.6824, huber: 0.3418, swd: 0.1305, ept: 81.4249
    Best round's Test MSE: 0.8041, MAE: 0.6824, SWD: 0.1305
    Best round's Validation MSE: 0.6168, MAE: 0.5620, SWD: 0.0709
    Best round's Test verification MSE : 0.8041, MAE: 0.6824, SWD: 0.1305
    Time taken: 23.56 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq336_pred720_20250510_1918)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.8301 ± 0.0537
      mae: 0.6832 ± 0.0138
      huber: 0.3450 ± 0.0132
      swd: 0.1372 ± 0.0102
      ept: 88.4561 ± 4.9724
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.5724 ± 0.0324
      mae: 0.5314 ± 0.0239
      huber: 0.2384 ± 0.0157
      swd: 0.0670 ± 0.0035
      ept: 127.2156 ± 2.4984
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 114.10 seconds
    
    Experiment complete: PatchTST_etth1_seq336_pred720_20250510_1918
    Model: PatchTST
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### DLinear

#### pred=96


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 96
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
    
    Epoch [1/50], Train Losses: mse: 0.4533, mae: 0.4713, huber: 0.1934, swd: 0.1299, ept: 40.5430
    Epoch [1/50], Val Losses: mse: 0.3731, mae: 0.4163, huber: 0.1617, swd: 0.1065, ept: 45.7823
    Epoch [1/50], Test Losses: mse: 0.4396, mae: 0.4633, huber: 0.1906, swd: 0.1174, ept: 37.6097
      Epoch 1 composite train-obj: 0.193437
            Val objective improved inf → 0.1617, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3573, mae: 0.4126, huber: 0.1558, swd: 0.1134, ept: 51.0291
    Epoch [2/50], Val Losses: mse: 0.3602, mae: 0.4037, huber: 0.1555, swd: 0.1015, ept: 48.5072
    Epoch [2/50], Test Losses: mse: 0.4290, mae: 0.4529, huber: 0.1853, swd: 0.1182, ept: 40.1091
      Epoch 2 composite train-obj: 0.155762
            Val objective improved 0.1617 → 0.1555, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3499, mae: 0.4060, huber: 0.1523, swd: 0.1108, ept: 52.5525
    Epoch [3/50], Val Losses: mse: 0.3568, mae: 0.4000, huber: 0.1541, swd: 0.1027, ept: 48.8167
    Epoch [3/50], Test Losses: mse: 0.4260, mae: 0.4489, huber: 0.1838, swd: 0.1224, ept: 40.6410
      Epoch 3 composite train-obj: 0.152288
            Val objective improved 0.1555 → 0.1541, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3462, mae: 0.4031, huber: 0.1508, swd: 0.1099, ept: 53.1887
    Epoch [4/50], Val Losses: mse: 0.3576, mae: 0.3986, huber: 0.1539, swd: 0.0991, ept: 50.0665
    Epoch [4/50], Test Losses: mse: 0.4257, mae: 0.4474, huber: 0.1832, swd: 0.1163, ept: 41.4715
      Epoch 4 composite train-obj: 0.150775
            Val objective improved 0.1541 → 0.1539, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3452, mae: 0.4017, huber: 0.1502, swd: 0.1094, ept: 53.5192
    Epoch [5/50], Val Losses: mse: 0.3564, mae: 0.3977, huber: 0.1536, swd: 0.0961, ept: 49.7991
    Epoch [5/50], Test Losses: mse: 0.4229, mae: 0.4447, huber: 0.1816, swd: 0.1143, ept: 41.2291
      Epoch 5 composite train-obj: 0.150164
            Val objective improved 0.1539 → 0.1536, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3441, mae: 0.4008, huber: 0.1496, swd: 0.1094, ept: 53.7682
    Epoch [6/50], Val Losses: mse: 0.3578, mae: 0.4011, huber: 0.1545, swd: 0.1033, ept: 50.3017
    Epoch [6/50], Test Losses: mse: 0.4234, mae: 0.4443, huber: 0.1820, swd: 0.1191, ept: 41.3654
      Epoch 6 composite train-obj: 0.149628
            No improvement (0.1545), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3433, mae: 0.4002, huber: 0.1494, swd: 0.1092, ept: 53.8285
    Epoch [7/50], Val Losses: mse: 0.3642, mae: 0.4044, huber: 0.1571, swd: 0.1080, ept: 50.1803
    Epoch [7/50], Test Losses: mse: 0.4221, mae: 0.4440, huber: 0.1816, swd: 0.1193, ept: 41.5624
      Epoch 7 composite train-obj: 0.149402
            No improvement (0.1571), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3429, mae: 0.3997, huber: 0.1491, swd: 0.1092, ept: 54.1001
    Epoch [8/50], Val Losses: mse: 0.3573, mae: 0.3997, huber: 0.1543, swd: 0.1058, ept: 50.4506
    Epoch [8/50], Test Losses: mse: 0.4182, mae: 0.4408, huber: 0.1800, swd: 0.1186, ept: 41.6218
      Epoch 8 composite train-obj: 0.149142
            No improvement (0.1543), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3427, mae: 0.3993, huber: 0.1490, swd: 0.1092, ept: 54.1253
    Epoch [9/50], Val Losses: mse: 0.3526, mae: 0.3946, huber: 0.1519, swd: 0.1033, ept: 50.6711
    Epoch [9/50], Test Losses: mse: 0.4211, mae: 0.4435, huber: 0.1811, swd: 0.1225, ept: 41.3785
      Epoch 9 composite train-obj: 0.148977
            Val objective improved 0.1536 → 0.1519, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.3417, mae: 0.3986, huber: 0.1486, swd: 0.1091, ept: 54.2160
    Epoch [10/50], Val Losses: mse: 0.3551, mae: 0.3957, huber: 0.1528, swd: 0.1037, ept: 50.6863
    Epoch [10/50], Test Losses: mse: 0.4207, mae: 0.4425, huber: 0.1809, swd: 0.1211, ept: 41.6606
      Epoch 10 composite train-obj: 0.148640
            No improvement (0.1528), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.3417, mae: 0.3984, huber: 0.1486, swd: 0.1091, ept: 54.2531
    Epoch [11/50], Val Losses: mse: 0.3527, mae: 0.3941, huber: 0.1519, swd: 0.0990, ept: 50.6660
    Epoch [11/50], Test Losses: mse: 0.4206, mae: 0.4431, huber: 0.1808, swd: 0.1180, ept: 41.6734
      Epoch 11 composite train-obj: 0.148551
            Val objective improved 0.1519 → 0.1519, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.3422, mae: 0.3989, huber: 0.1488, swd: 0.1092, ept: 54.2944
    Epoch [12/50], Val Losses: mse: 0.3526, mae: 0.3947, huber: 0.1518, swd: 0.1015, ept: 51.0173
    Epoch [12/50], Test Losses: mse: 0.4233, mae: 0.4431, huber: 0.1817, swd: 0.1192, ept: 41.6427
      Epoch 12 composite train-obj: 0.148791
            Val objective improved 0.1519 → 0.1518, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.3410, mae: 0.3983, huber: 0.1484, swd: 0.1091, ept: 54.3649
    Epoch [13/50], Val Losses: mse: 0.3567, mae: 0.3968, huber: 0.1533, swd: 0.0985, ept: 50.9394
    Epoch [13/50], Test Losses: mse: 0.4209, mae: 0.4427, huber: 0.1807, swd: 0.1160, ept: 41.7693
      Epoch 13 composite train-obj: 0.148393
            No improvement (0.1533), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.3419, mae: 0.3983, huber: 0.1485, swd: 0.1089, ept: 54.3985
    Epoch [14/50], Val Losses: mse: 0.3537, mae: 0.3964, huber: 0.1527, swd: 0.1000, ept: 50.5077
    Epoch [14/50], Test Losses: mse: 0.4217, mae: 0.4439, huber: 0.1815, swd: 0.1205, ept: 41.8358
      Epoch 14 composite train-obj: 0.148545
            No improvement (0.1527), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.3412, mae: 0.3980, huber: 0.1484, swd: 0.1090, ept: 54.3432
    Epoch [15/50], Val Losses: mse: 0.3542, mae: 0.3983, huber: 0.1533, swd: 0.1072, ept: 50.4472
    Epoch [15/50], Test Losses: mse: 0.4208, mae: 0.4420, huber: 0.1809, swd: 0.1245, ept: 41.5753
      Epoch 15 composite train-obj: 0.148364
            No improvement (0.1533), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.3414, mae: 0.3982, huber: 0.1485, swd: 0.1089, ept: 54.3382
    Epoch [16/50], Val Losses: mse: 0.3549, mae: 0.3952, huber: 0.1524, swd: 0.1062, ept: 50.8522
    Epoch [16/50], Test Losses: mse: 0.4247, mae: 0.4446, huber: 0.1823, swd: 0.1264, ept: 41.6819
      Epoch 16 composite train-obj: 0.148479
            No improvement (0.1524), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.3409, mae: 0.3979, huber: 0.1483, swd: 0.1091, ept: 54.4192
    Epoch [17/50], Val Losses: mse: 0.3522, mae: 0.3947, huber: 0.1519, swd: 0.0994, ept: 50.6098
    Epoch [17/50], Test Losses: mse: 0.4187, mae: 0.4414, huber: 0.1799, swd: 0.1191, ept: 41.7654
      Epoch 17 composite train-obj: 0.148258
    Epoch [17/50], Test Losses: mse: 0.4233, mae: 0.4431, huber: 0.1817, swd: 0.1192, ept: 41.6427
    Best round's Test MSE: 0.4233, MAE: 0.4431, SWD: 0.1192
    Best round's Validation MSE: 0.3526, MAE: 0.3947, SWD: 0.1015
    Best round's Test verification MSE : 0.4233, MAE: 0.4431, SWD: 0.1192
    Time taken: 18.44 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4501, mae: 0.4697, huber: 0.1923, swd: 0.1280, ept: 40.6666
    Epoch [1/50], Val Losses: mse: 0.3764, mae: 0.4192, huber: 0.1632, swd: 0.1042, ept: 45.6414
    Epoch [1/50], Test Losses: mse: 0.4389, mae: 0.4631, huber: 0.1905, swd: 0.1132, ept: 37.6640
      Epoch 1 composite train-obj: 0.192259
            Val objective improved inf → 0.1632, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3579, mae: 0.4130, huber: 0.1559, swd: 0.1117, ept: 51.1030
    Epoch [2/50], Val Losses: mse: 0.3612, mae: 0.4032, huber: 0.1558, swd: 0.0936, ept: 48.0856
    Epoch [2/50], Test Losses: mse: 0.4291, mae: 0.4522, huber: 0.1849, swd: 0.1117, ept: 39.9980
      Epoch 2 composite train-obj: 0.155918
            Val objective improved 0.1632 → 0.1558, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3493, mae: 0.4058, huber: 0.1522, swd: 0.1082, ept: 52.6538
    Epoch [3/50], Val Losses: mse: 0.3608, mae: 0.4030, huber: 0.1557, swd: 0.0993, ept: 49.0387
    Epoch [3/50], Test Losses: mse: 0.4251, mae: 0.4479, huber: 0.1834, swd: 0.1141, ept: 40.7252
      Epoch 3 composite train-obj: 0.152163
            Val objective improved 0.1558 → 0.1557, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3463, mae: 0.4031, huber: 0.1508, swd: 0.1080, ept: 53.2090
    Epoch [4/50], Val Losses: mse: 0.3565, mae: 0.3976, huber: 0.1533, swd: 0.0957, ept: 49.8115
    Epoch [4/50], Test Losses: mse: 0.4252, mae: 0.4470, huber: 0.1828, swd: 0.1131, ept: 41.1799
      Epoch 4 composite train-obj: 0.150776
            Val objective improved 0.1557 → 0.1533, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3446, mae: 0.4013, huber: 0.1499, swd: 0.1073, ept: 53.6383
    Epoch [5/50], Val Losses: mse: 0.3526, mae: 0.3958, huber: 0.1521, swd: 0.0945, ept: 50.4969
    Epoch [5/50], Test Losses: mse: 0.4213, mae: 0.4437, huber: 0.1812, swd: 0.1138, ept: 41.1027
      Epoch 5 composite train-obj: 0.149939
            Val objective improved 0.1533 → 0.1521, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3439, mae: 0.4010, huber: 0.1497, swd: 0.1076, ept: 53.7317
    Epoch [6/50], Val Losses: mse: 0.3559, mae: 0.3969, huber: 0.1531, swd: 0.0983, ept: 50.4144
    Epoch [6/50], Test Losses: mse: 0.4249, mae: 0.4460, huber: 0.1827, swd: 0.1164, ept: 41.4450
      Epoch 6 composite train-obj: 0.149697
            No improvement (0.1531), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3433, mae: 0.3999, huber: 0.1492, swd: 0.1072, ept: 53.9630
    Epoch [7/50], Val Losses: mse: 0.3539, mae: 0.3963, huber: 0.1524, swd: 0.0991, ept: 50.3929
    Epoch [7/50], Test Losses: mse: 0.4216, mae: 0.4458, huber: 0.1818, swd: 0.1198, ept: 41.6135
      Epoch 7 composite train-obj: 0.149250
            No improvement (0.1524), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3427, mae: 0.3997, huber: 0.1491, swd: 0.1074, ept: 54.0313
    Epoch [8/50], Val Losses: mse: 0.3578, mae: 0.4006, huber: 0.1545, swd: 0.0970, ept: 49.9809
    Epoch [8/50], Test Losses: mse: 0.4201, mae: 0.4424, huber: 0.1806, swd: 0.1118, ept: 41.5827
      Epoch 8 composite train-obj: 0.149136
            No improvement (0.1545), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3426, mae: 0.3992, huber: 0.1490, swd: 0.1075, ept: 54.2098
    Epoch [9/50], Val Losses: mse: 0.3529, mae: 0.3943, huber: 0.1521, swd: 0.0962, ept: 50.6882
    Epoch [9/50], Test Losses: mse: 0.4244, mae: 0.4458, huber: 0.1823, swd: 0.1171, ept: 41.4492
      Epoch 9 composite train-obj: 0.148952
            Val objective improved 0.1521 → 0.1521, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.3421, mae: 0.3990, huber: 0.1488, swd: 0.1071, ept: 54.2271
    Epoch [10/50], Val Losses: mse: 0.3532, mae: 0.3944, huber: 0.1521, swd: 0.0965, ept: 50.7312
    Epoch [10/50], Test Losses: mse: 0.4209, mae: 0.4426, huber: 0.1809, swd: 0.1144, ept: 41.6290
      Epoch 10 composite train-obj: 0.148767
            Val objective improved 0.1521 → 0.1521, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.3424, mae: 0.3990, huber: 0.1489, swd: 0.1077, ept: 54.2235
    Epoch [11/50], Val Losses: mse: 0.3525, mae: 0.3936, huber: 0.1518, swd: 0.0952, ept: 50.6075
    Epoch [11/50], Test Losses: mse: 0.4204, mae: 0.4426, huber: 0.1806, swd: 0.1148, ept: 41.4548
      Epoch 11 composite train-obj: 0.148867
            Val objective improved 0.1521 → 0.1518, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.3418, mae: 0.3986, huber: 0.1486, swd: 0.1071, ept: 54.2912
    Epoch [12/50], Val Losses: mse: 0.3515, mae: 0.3931, huber: 0.1511, swd: 0.0927, ept: 50.9598
    Epoch [12/50], Test Losses: mse: 0.4232, mae: 0.4451, huber: 0.1819, swd: 0.1144, ept: 41.6340
      Epoch 12 composite train-obj: 0.148648
            Val objective improved 0.1518 → 0.1511, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.3414, mae: 0.3984, huber: 0.1485, swd: 0.1076, ept: 54.2857
    Epoch [13/50], Val Losses: mse: 0.3539, mae: 0.3935, huber: 0.1521, swd: 0.0908, ept: 50.6649
    Epoch [13/50], Test Losses: mse: 0.4215, mae: 0.4424, huber: 0.1806, swd: 0.1091, ept: 42.0907
      Epoch 13 composite train-obj: 0.148504
            No improvement (0.1521), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.3416, mae: 0.3983, huber: 0.1485, swd: 0.1071, ept: 54.3650
    Epoch [14/50], Val Losses: mse: 0.3565, mae: 0.3952, huber: 0.1531, swd: 0.1022, ept: 50.9081
    Epoch [14/50], Test Losses: mse: 0.4208, mae: 0.4423, huber: 0.1809, swd: 0.1181, ept: 41.6566
      Epoch 14 composite train-obj: 0.148517
            No improvement (0.1531), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.3414, mae: 0.3982, huber: 0.1484, swd: 0.1073, ept: 54.3445
    Epoch [15/50], Val Losses: mse: 0.3516, mae: 0.3947, huber: 0.1514, swd: 0.0943, ept: 50.8198
    Epoch [15/50], Test Losses: mse: 0.4213, mae: 0.4435, huber: 0.1809, swd: 0.1153, ept: 41.4637
      Epoch 15 composite train-obj: 0.148425
            No improvement (0.1514), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.3414, mae: 0.3981, huber: 0.1484, swd: 0.1074, ept: 54.3630
    Epoch [16/50], Val Losses: mse: 0.3518, mae: 0.3943, huber: 0.1515, swd: 0.0967, ept: 51.0161
    Epoch [16/50], Test Losses: mse: 0.4221, mae: 0.4443, huber: 0.1814, swd: 0.1163, ept: 41.6556
      Epoch 16 composite train-obj: 0.148422
            No improvement (0.1515), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.3408, mae: 0.3976, huber: 0.1482, swd: 0.1074, ept: 54.5298
    Epoch [17/50], Val Losses: mse: 0.3556, mae: 0.3965, huber: 0.1534, swd: 0.0974, ept: 50.3656
    Epoch [17/50], Test Losses: mse: 0.4195, mae: 0.4407, huber: 0.1802, swd: 0.1142, ept: 41.6268
      Epoch 17 composite train-obj: 0.148183
    Epoch [17/50], Test Losses: mse: 0.4232, mae: 0.4451, huber: 0.1819, swd: 0.1144, ept: 41.6340
    Best round's Test MSE: 0.4232, MAE: 0.4451, SWD: 0.1144
    Best round's Validation MSE: 0.3515, MAE: 0.3931, SWD: 0.0927
    Best round's Test verification MSE : 0.4232, MAE: 0.4451, SWD: 0.1144
    Time taken: 22.71 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4472, mae: 0.4685, huber: 0.1912, swd: 0.1189, ept: 40.8153
    Epoch [1/50], Val Losses: mse: 0.3758, mae: 0.4196, huber: 0.1630, swd: 0.1049, ept: 45.4209
    Epoch [1/50], Test Losses: mse: 0.4396, mae: 0.4641, huber: 0.1908, swd: 0.1122, ept: 37.4509
      Epoch 1 composite train-obj: 0.191203
            Val objective improved inf → 0.1630, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3580, mae: 0.4129, huber: 0.1559, swd: 0.1062, ept: 51.0903
    Epoch [2/50], Val Losses: mse: 0.3640, mae: 0.4053, huber: 0.1569, swd: 0.1003, ept: 48.5976
    Epoch [2/50], Test Losses: mse: 0.4298, mae: 0.4535, huber: 0.1857, swd: 0.1133, ept: 40.0471
      Epoch 2 composite train-obj: 0.155888
            Val objective improved 0.1630 → 0.1569, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3492, mae: 0.4060, huber: 0.1521, swd: 0.1035, ept: 52.5380
    Epoch [3/50], Val Losses: mse: 0.3579, mae: 0.3989, huber: 0.1542, swd: 0.0940, ept: 49.4006
    Epoch [3/50], Test Losses: mse: 0.4262, mae: 0.4476, huber: 0.1833, swd: 0.1085, ept: 41.1300
      Epoch 3 composite train-obj: 0.152138
            Val objective improved 0.1569 → 0.1542, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3469, mae: 0.4033, huber: 0.1509, swd: 0.1029, ept: 53.2498
    Epoch [4/50], Val Losses: mse: 0.3564, mae: 0.3998, huber: 0.1541, swd: 0.1012, ept: 49.7261
    Epoch [4/50], Test Losses: mse: 0.4240, mae: 0.4471, huber: 0.1827, swd: 0.1189, ept: 41.0385
      Epoch 4 composite train-obj: 0.150931
            Val objective improved 0.1542 → 0.1541, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3451, mae: 0.4019, huber: 0.1502, swd: 0.1026, ept: 53.4538
    Epoch [5/50], Val Losses: mse: 0.3564, mae: 0.3990, huber: 0.1537, swd: 0.0987, ept: 50.4050
    Epoch [5/50], Test Losses: mse: 0.4231, mae: 0.4449, huber: 0.1819, swd: 0.1144, ept: 41.2880
      Epoch 5 composite train-obj: 0.150172
            Val objective improved 0.1541 → 0.1537, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3444, mae: 0.4009, huber: 0.1498, swd: 0.1026, ept: 53.7245
    Epoch [6/50], Val Losses: mse: 0.3541, mae: 0.3962, huber: 0.1525, swd: 0.0978, ept: 50.3717
    Epoch [6/50], Test Losses: mse: 0.4232, mae: 0.4457, huber: 0.1820, swd: 0.1138, ept: 41.3153
      Epoch 6 composite train-obj: 0.149784
            Val objective improved 0.1537 → 0.1525, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3431, mae: 0.4001, huber: 0.1493, swd: 0.1023, ept: 53.8547
    Epoch [7/50], Val Losses: mse: 0.3538, mae: 0.3963, huber: 0.1527, swd: 0.0980, ept: 50.3645
    Epoch [7/50], Test Losses: mse: 0.4219, mae: 0.4432, huber: 0.1814, swd: 0.1129, ept: 41.5841
      Epoch 7 composite train-obj: 0.149280
            No improvement (0.1527), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.3426, mae: 0.3996, huber: 0.1491, swd: 0.1025, ept: 53.9652
    Epoch [8/50], Val Losses: mse: 0.3542, mae: 0.3955, huber: 0.1526, swd: 0.1007, ept: 50.7506
    Epoch [8/50], Test Losses: mse: 0.4224, mae: 0.4434, huber: 0.1815, swd: 0.1171, ept: 41.5610
      Epoch 8 composite train-obj: 0.149097
            No improvement (0.1526), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.3427, mae: 0.3996, huber: 0.1491, swd: 0.1019, ept: 54.1046
    Epoch [9/50], Val Losses: mse: 0.3574, mae: 0.3984, huber: 0.1538, swd: 0.1017, ept: 50.4630
    Epoch [9/50], Test Losses: mse: 0.4225, mae: 0.4444, huber: 0.1817, swd: 0.1174, ept: 41.4942
      Epoch 9 composite train-obj: 0.149103
            No improvement (0.1538), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.3427, mae: 0.3993, huber: 0.1490, swd: 0.1022, ept: 54.1864
    Epoch [10/50], Val Losses: mse: 0.3573, mae: 0.4003, huber: 0.1541, swd: 0.1046, ept: 50.1980
    Epoch [10/50], Test Losses: mse: 0.4256, mae: 0.4462, huber: 0.1830, swd: 0.1227, ept: 41.6097
      Epoch 10 composite train-obj: 0.148982
            No improvement (0.1541), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.3422, mae: 0.3988, huber: 0.1488, swd: 0.1028, ept: 54.2735
    Epoch [11/50], Val Losses: mse: 0.3549, mae: 0.3972, huber: 0.1532, swd: 0.0998, ept: 50.4545
    Epoch [11/50], Test Losses: mse: 0.4243, mae: 0.4447, huber: 0.1824, swd: 0.1153, ept: 41.5335
      Epoch 11 composite train-obj: 0.148771
    Epoch [11/50], Test Losses: mse: 0.4232, mae: 0.4457, huber: 0.1820, swd: 0.1138, ept: 41.3153
    Best round's Test MSE: 0.4232, MAE: 0.4457, SWD: 0.1138
    Best round's Validation MSE: 0.3541, MAE: 0.3962, SWD: 0.0978
    Best round's Test verification MSE : 0.4232, MAE: 0.4457, SWD: 0.1138
    Time taken: 15.39 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq336_pred96_20250510_1917)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4232 ± 0.0000
      mae: 0.4446 ± 0.0011
      huber: 0.1819 ± 0.0001
      swd: 0.1158 ± 0.0024
      ept: 41.5307 ± 0.1523
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3527 ± 0.0011
      mae: 0.3947 ± 0.0013
      huber: 0.1518 ± 0.0006
      swd: 0.0973 ± 0.0036
      ept: 50.7829 ± 0.2917
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 56.58 seconds
    
    Experiment complete: DLinear_etth1_seq336_pred96_20250510_1917
    Model: DLinear
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 196
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
    
    Epoch [1/50], Train Losses: mse: 0.5163, mae: 0.5030, huber: 0.2162, swd: 0.1535, ept: 58.0769
    Epoch [1/50], Val Losses: mse: 0.4262, mae: 0.4487, huber: 0.1823, swd: 0.1152, ept: 60.7815
    Epoch [1/50], Test Losses: mse: 0.4837, mae: 0.4953, huber: 0.2107, swd: 0.1190, ept: 55.0975
      Epoch 1 composite train-obj: 0.216238
            Val objective improved inf → 0.1823, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4383, mae: 0.4556, huber: 0.1855, swd: 0.1455, ept: 75.6029
    Epoch [2/50], Val Losses: mse: 0.4137, mae: 0.4375, huber: 0.1767, swd: 0.1135, ept: 65.6907
    Epoch [2/50], Test Losses: mse: 0.4738, mae: 0.4870, huber: 0.2060, swd: 0.1236, ept: 57.9934
      Epoch 2 composite train-obj: 0.185550
            Val objective improved 0.1823 → 0.1767, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4296, mae: 0.4491, huber: 0.1819, swd: 0.1434, ept: 78.2019
    Epoch [3/50], Val Losses: mse: 0.4078, mae: 0.4336, huber: 0.1743, swd: 0.1091, ept: 67.8069
    Epoch [3/50], Test Losses: mse: 0.4709, mae: 0.4844, huber: 0.2046, swd: 0.1222, ept: 59.7022
      Epoch 3 composite train-obj: 0.181889
            Val objective improved 0.1767 → 0.1743, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4272, mae: 0.4472, huber: 0.1808, swd: 0.1429, ept: 79.0523
    Epoch [4/50], Val Losses: mse: 0.4163, mae: 0.4391, huber: 0.1781, swd: 0.1180, ept: 65.1096
    Epoch [4/50], Test Losses: mse: 0.4709, mae: 0.4830, huber: 0.2042, swd: 0.1266, ept: 60.3168
      Epoch 4 composite train-obj: 0.180848
            No improvement (0.1781), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4269, mae: 0.4462, huber: 0.1805, swd: 0.1429, ept: 79.7410
    Epoch [5/50], Val Losses: mse: 0.4200, mae: 0.4412, huber: 0.1796, swd: 0.1221, ept: 65.4381
    Epoch [5/50], Test Losses: mse: 0.4716, mae: 0.4835, huber: 0.2046, swd: 0.1274, ept: 59.8722
      Epoch 5 composite train-obj: 0.180495
            No improvement (0.1796), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4240, mae: 0.4448, huber: 0.1795, swd: 0.1425, ept: 80.1279
    Epoch [6/50], Val Losses: mse: 0.4067, mae: 0.4329, huber: 0.1741, swd: 0.1079, ept: 68.4073
    Epoch [6/50], Test Losses: mse: 0.4645, mae: 0.4784, huber: 0.2015, swd: 0.1206, ept: 61.0848
      Epoch 6 composite train-obj: 0.179490
            Val objective improved 0.1743 → 0.1741, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4251, mae: 0.4451, huber: 0.1799, swd: 0.1431, ept: 80.2368
    Epoch [7/50], Val Losses: mse: 0.4095, mae: 0.4348, huber: 0.1750, swd: 0.1165, ept: 69.3659
    Epoch [7/50], Test Losses: mse: 0.4687, mae: 0.4818, huber: 0.2034, swd: 0.1294, ept: 60.5064
      Epoch 7 composite train-obj: 0.179863
            No improvement (0.1750), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4240, mae: 0.4440, huber: 0.1793, swd: 0.1424, ept: 80.8003
    Epoch [8/50], Val Losses: mse: 0.4138, mae: 0.4345, huber: 0.1767, swd: 0.1141, ept: 67.1568
    Epoch [8/50], Test Losses: mse: 0.4708, mae: 0.4820, huber: 0.2040, swd: 0.1229, ept: 61.3670
      Epoch 8 composite train-obj: 0.179296
            No improvement (0.1767), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.4217, mae: 0.4430, huber: 0.1785, swd: 0.1423, ept: 80.7864
    Epoch [9/50], Val Losses: mse: 0.4168, mae: 0.4369, huber: 0.1779, swd: 0.1151, ept: 66.2989
    Epoch [9/50], Test Losses: mse: 0.4691, mae: 0.4791, huber: 0.2029, swd: 0.1198, ept: 61.3155
      Epoch 9 composite train-obj: 0.178533
            No improvement (0.1779), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.4242, mae: 0.4435, huber: 0.1792, swd: 0.1436, ept: 81.1306
    Epoch [10/50], Val Losses: mse: 0.4057, mae: 0.4301, huber: 0.1734, swd: 0.1093, ept: 69.9242
    Epoch [10/50], Test Losses: mse: 0.4672, mae: 0.4782, huber: 0.2023, swd: 0.1173, ept: 61.1205
      Epoch 10 composite train-obj: 0.179157
            Val objective improved 0.1741 → 0.1734, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.4223, mae: 0.4430, huber: 0.1787, swd: 0.1429, ept: 81.1011
    Epoch [11/50], Val Losses: mse: 0.4069, mae: 0.4310, huber: 0.1740, swd: 0.1110, ept: 67.6037
    Epoch [11/50], Test Losses: mse: 0.4631, mae: 0.4758, huber: 0.2004, swd: 0.1180, ept: 61.5330
      Epoch 11 composite train-obj: 0.178665
            No improvement (0.1740), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.4245, mae: 0.4436, huber: 0.1793, swd: 0.1434, ept: 81.1745
    Epoch [12/50], Val Losses: mse: 0.4091, mae: 0.4338, huber: 0.1754, swd: 0.1094, ept: 65.6940
    Epoch [12/50], Test Losses: mse: 0.4669, mae: 0.4803, huber: 0.2024, swd: 0.1207, ept: 61.8310
      Epoch 12 composite train-obj: 0.179311
            No improvement (0.1754), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.4215, mae: 0.4428, huber: 0.1784, swd: 0.1423, ept: 81.1176
    Epoch [13/50], Val Losses: mse: 0.4135, mae: 0.4363, huber: 0.1769, swd: 0.1140, ept: 66.0766
    Epoch [13/50], Test Losses: mse: 0.4636, mae: 0.4761, huber: 0.2009, swd: 0.1171, ept: 60.1011
      Epoch 13 composite train-obj: 0.178445
            No improvement (0.1769), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.4228, mae: 0.4427, huber: 0.1786, swd: 0.1424, ept: 81.3104
    Epoch [14/50], Val Losses: mse: 0.4103, mae: 0.4318, huber: 0.1747, swd: 0.1088, ept: 69.7708
    Epoch [14/50], Test Losses: mse: 0.4647, mae: 0.4781, huber: 0.2014, swd: 0.1175, ept: 61.5349
      Epoch 14 composite train-obj: 0.178595
            No improvement (0.1747), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.4221, mae: 0.4426, huber: 0.1784, swd: 0.1424, ept: 81.3754
    Epoch [15/50], Val Losses: mse: 0.4175, mae: 0.4354, huber: 0.1777, swd: 0.1146, ept: 67.3961
    Epoch [15/50], Test Losses: mse: 0.4724, mae: 0.4836, huber: 0.2047, swd: 0.1228, ept: 61.5412
      Epoch 15 composite train-obj: 0.178441
    Epoch [15/50], Test Losses: mse: 0.4672, mae: 0.4782, huber: 0.2023, swd: 0.1173, ept: 61.1205
    Best round's Test MSE: 0.4672, MAE: 0.4782, SWD: 0.1173
    Best round's Validation MSE: 0.4057, MAE: 0.4301, SWD: 0.1093
    Best round's Test verification MSE : 0.4672, MAE: 0.4782, SWD: 0.1173
    Time taken: 15.00 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5196, mae: 0.5041, huber: 0.2172, swd: 0.1553, ept: 58.0077
    Epoch [1/50], Val Losses: mse: 0.4271, mae: 0.4491, huber: 0.1827, swd: 0.1153, ept: 60.0641
    Epoch [1/50], Test Losses: mse: 0.4833, mae: 0.4952, huber: 0.2105, swd: 0.1194, ept: 55.1762
      Epoch 1 composite train-obj: 0.217189
            Val objective improved inf → 0.1827, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4368, mae: 0.4548, huber: 0.1850, swd: 0.1467, ept: 75.4163
    Epoch [2/50], Val Losses: mse: 0.4195, mae: 0.4401, huber: 0.1788, swd: 0.1124, ept: 64.2254
    Epoch [2/50], Test Losses: mse: 0.4736, mae: 0.4861, huber: 0.2058, swd: 0.1178, ept: 58.1713
      Epoch 2 composite train-obj: 0.184961
            Val objective improved 0.1827 → 0.1788, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4300, mae: 0.4496, huber: 0.1822, swd: 0.1446, ept: 77.9092
    Epoch [3/50], Val Losses: mse: 0.4232, mae: 0.4425, huber: 0.1806, swd: 0.1160, ept: 64.0062
    Epoch [3/50], Test Losses: mse: 0.4696, mae: 0.4824, huber: 0.2037, swd: 0.1188, ept: 59.9164
      Epoch 3 composite train-obj: 0.182158
            No improvement (0.1806), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4280, mae: 0.4478, huber: 0.1812, swd: 0.1447, ept: 78.9197
    Epoch [4/50], Val Losses: mse: 0.4095, mae: 0.4336, huber: 0.1748, swd: 0.1065, ept: 68.1102
    Epoch [4/50], Test Losses: mse: 0.4708, mae: 0.4845, huber: 0.2044, swd: 0.1201, ept: 59.7143
      Epoch 4 composite train-obj: 0.181152
            Val objective improved 0.1788 → 0.1748, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4267, mae: 0.4465, huber: 0.1805, swd: 0.1444, ept: 79.4862
    Epoch [5/50], Val Losses: mse: 0.4119, mae: 0.4336, huber: 0.1755, swd: 0.1091, ept: 67.4716
    Epoch [5/50], Test Losses: mse: 0.4711, mae: 0.4837, huber: 0.2044, swd: 0.1209, ept: 59.7355
      Epoch 5 composite train-obj: 0.180540
            No improvement (0.1755), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4250, mae: 0.4449, huber: 0.1797, swd: 0.1444, ept: 80.2832
    Epoch [6/50], Val Losses: mse: 0.4129, mae: 0.4347, huber: 0.1764, swd: 0.1100, ept: 66.5621
    Epoch [6/50], Test Losses: mse: 0.4701, mae: 0.4820, huber: 0.2036, swd: 0.1215, ept: 60.4488
      Epoch 6 composite train-obj: 0.179745
            No improvement (0.1764), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.4229, mae: 0.4440, huber: 0.1791, swd: 0.1438, ept: 80.5009
    Epoch [7/50], Val Losses: mse: 0.4130, mae: 0.4351, huber: 0.1766, swd: 0.1089, ept: 65.5181
    Epoch [7/50], Test Losses: mse: 0.4660, mae: 0.4787, huber: 0.2020, swd: 0.1147, ept: 60.6324
      Epoch 7 composite train-obj: 0.179105
            No improvement (0.1766), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.4250, mae: 0.4447, huber: 0.1796, swd: 0.1440, ept: 80.3884
    Epoch [8/50], Val Losses: mse: 0.4096, mae: 0.4348, huber: 0.1757, swd: 0.1127, ept: 65.7329
    Epoch [8/50], Test Losses: mse: 0.4677, mae: 0.4797, huber: 0.2027, swd: 0.1208, ept: 61.7423
      Epoch 8 composite train-obj: 0.179650
            No improvement (0.1757), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.4229, mae: 0.4436, huber: 0.1790, swd: 0.1437, ept: 81.0853
    Epoch [9/50], Val Losses: mse: 0.4104, mae: 0.4330, huber: 0.1751, swd: 0.1094, ept: 69.0835
    Epoch [9/50], Test Losses: mse: 0.4629, mae: 0.4772, huber: 0.2007, swd: 0.1182, ept: 60.8009
      Epoch 9 composite train-obj: 0.178967
    Epoch [9/50], Test Losses: mse: 0.4708, mae: 0.4845, huber: 0.2044, swd: 0.1201, ept: 59.7143
    Best round's Test MSE: 0.4708, MAE: 0.4845, SWD: 0.1201
    Best round's Validation MSE: 0.4095, MAE: 0.4336, SWD: 0.1065
    Best round's Test verification MSE : 0.4708, MAE: 0.4845, SWD: 0.1201
    Time taken: 8.97 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5161, mae: 0.5029, huber: 0.2161, swd: 0.1403, ept: 58.2078
    Epoch [1/50], Val Losses: mse: 0.4216, mae: 0.4477, huber: 0.1811, swd: 0.1006, ept: 61.7133
    Epoch [1/50], Test Losses: mse: 0.4774, mae: 0.4916, huber: 0.2080, swd: 0.1017, ept: 55.2884
      Epoch 1 composite train-obj: 0.216099
            Val objective improved inf → 0.1811, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4376, mae: 0.4554, huber: 0.1854, swd: 0.1327, ept: 75.4849
    Epoch [2/50], Val Losses: mse: 0.4235, mae: 0.4447, huber: 0.1811, swd: 0.1136, ept: 63.7505
    Epoch [2/50], Test Losses: mse: 0.4719, mae: 0.4849, huber: 0.2051, swd: 0.1158, ept: 58.9017
      Epoch 2 composite train-obj: 0.185378
            No improvement (0.1811), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4297, mae: 0.4495, huber: 0.1820, swd: 0.1324, ept: 78.0957
    Epoch [3/50], Val Losses: mse: 0.4172, mae: 0.4397, huber: 0.1784, swd: 0.1094, ept: 65.5914
    Epoch [3/50], Test Losses: mse: 0.4674, mae: 0.4813, huber: 0.2029, swd: 0.1167, ept: 59.8719
      Epoch 3 composite train-obj: 0.182033
            Val objective improved 0.1811 → 0.1784, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4271, mae: 0.4471, huber: 0.1808, swd: 0.1316, ept: 79.2051
    Epoch [4/50], Val Losses: mse: 0.4207, mae: 0.4408, huber: 0.1798, swd: 0.1094, ept: 65.6517
    Epoch [4/50], Test Losses: mse: 0.4665, mae: 0.4791, huber: 0.2023, swd: 0.1090, ept: 59.7175
      Epoch 4 composite train-obj: 0.180801
            No improvement (0.1798), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4254, mae: 0.4454, huber: 0.1799, swd: 0.1314, ept: 79.7654
    Epoch [5/50], Val Losses: mse: 0.4155, mae: 0.4359, huber: 0.1772, swd: 0.1054, ept: 66.7178
    Epoch [5/50], Test Losses: mse: 0.4719, mae: 0.4824, huber: 0.2044, swd: 0.1089, ept: 60.4502
      Epoch 5 composite train-obj: 0.179897
            Val objective improved 0.1784 → 0.1772, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4241, mae: 0.4446, huber: 0.1794, swd: 0.1310, ept: 80.1987
    Epoch [6/50], Val Losses: mse: 0.4071, mae: 0.4305, huber: 0.1736, swd: 0.1011, ept: 68.7822
    Epoch [6/50], Test Losses: mse: 0.4681, mae: 0.4796, huber: 0.2026, swd: 0.1094, ept: 61.1661
      Epoch 6 composite train-obj: 0.179446
            Val objective improved 0.1772 → 0.1736, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4241, mae: 0.4441, huber: 0.1794, swd: 0.1314, ept: 80.5298
    Epoch [7/50], Val Losses: mse: 0.4057, mae: 0.4302, huber: 0.1732, swd: 0.1001, ept: 67.2934
    Epoch [7/50], Test Losses: mse: 0.4653, mae: 0.4801, huber: 0.2020, swd: 0.1118, ept: 60.4952
      Epoch 7 composite train-obj: 0.179386
            Val objective improved 0.1736 → 0.1732, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4240, mae: 0.4442, huber: 0.1793, swd: 0.1313, ept: 80.4839
    Epoch [8/50], Val Losses: mse: 0.4227, mae: 0.4419, huber: 0.1807, swd: 0.1130, ept: 65.4084
    Epoch [8/50], Test Losses: mse: 0.4664, mae: 0.4794, huber: 0.2021, swd: 0.1120, ept: 60.8357
      Epoch 8 composite train-obj: 0.179327
            No improvement (0.1807), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4223, mae: 0.4434, huber: 0.1788, swd: 0.1311, ept: 80.8311
    Epoch [9/50], Val Losses: mse: 0.4102, mae: 0.4319, huber: 0.1748, swd: 0.1008, ept: 69.9685
    Epoch [9/50], Test Losses: mse: 0.4638, mae: 0.4779, huber: 0.2012, swd: 0.1075, ept: 60.1020
      Epoch 9 composite train-obj: 0.178799
            No improvement (0.1748), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.4236, mae: 0.4439, huber: 0.1792, swd: 0.1314, ept: 80.7116
    Epoch [10/50], Val Losses: mse: 0.4187, mae: 0.4407, huber: 0.1794, swd: 0.1182, ept: 65.8776
    Epoch [10/50], Test Losses: mse: 0.4684, mae: 0.4799, huber: 0.2028, swd: 0.1183, ept: 61.2315
      Epoch 10 composite train-obj: 0.179162
            No improvement (0.1794), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.4217, mae: 0.4426, huber: 0.1784, swd: 0.1309, ept: 81.0269
    Epoch [11/50], Val Losses: mse: 0.4089, mae: 0.4314, huber: 0.1746, swd: 0.1040, ept: 69.1992
    Epoch [11/50], Test Losses: mse: 0.4630, mae: 0.4758, huber: 0.2005, swd: 0.1111, ept: 61.5803
      Epoch 11 composite train-obj: 0.178402
            No improvement (0.1746), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.4223, mae: 0.4429, huber: 0.1786, swd: 0.1318, ept: 81.2197
    Epoch [12/50], Val Losses: mse: 0.4098, mae: 0.4328, huber: 0.1753, swd: 0.1018, ept: 67.4241
    Epoch [12/50], Test Losses: mse: 0.4634, mae: 0.4764, huber: 0.2008, swd: 0.1085, ept: 61.6174
      Epoch 12 composite train-obj: 0.178620
    Epoch [12/50], Test Losses: mse: 0.4653, mae: 0.4801, huber: 0.2020, swd: 0.1118, ept: 60.4952
    Best round's Test MSE: 0.4653, MAE: 0.4801, SWD: 0.1118
    Best round's Validation MSE: 0.4057, MAE: 0.4302, SWD: 0.1001
    Best round's Test verification MSE : 0.4653, MAE: 0.4801, SWD: 0.1118
    Time taken: 11.80 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq336_pred196_20250510_1917)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.4678 ± 0.0023
      mae: 0.4809 ± 0.0026
      huber: 0.2029 ± 0.0011
      swd: 0.1164 ± 0.0035
      ept: 60.4433 ± 0.5752
      count: 10.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4070 ± 0.0018
      mae: 0.4313 ± 0.0016
      huber: 0.1738 ± 0.0007
      swd: 0.1053 ± 0.0039
      ept: 68.4426 ± 1.0994
      count: 10.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 35.81 seconds
    
    Experiment complete: DLinear_etth1_seq336_pred196_20250510_1917
    Model: DLinear
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 91
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 336
    Batch Size: 128
    Scaling: Yes
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 91
    Validation Batches: 9
    Test Batches: 22
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5673, mae: 0.5306, huber: 0.2358, swd: 0.1688, ept: 71.7328
    Epoch [1/50], Val Losses: mse: 0.4303, mae: 0.4537, huber: 0.1848, swd: 0.1110, ept: 74.2050
    Epoch [1/50], Test Losses: mse: 0.5490, mae: 0.5393, huber: 0.2395, swd: 0.1345, ept: 72.2974
      Epoch 1 composite train-obj: 0.235847
            Val objective improved inf → 0.1848, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4977, mae: 0.4895, huber: 0.2088, swd: 0.1631, ept: 94.9993
    Epoch [2/50], Val Losses: mse: 0.4209, mae: 0.4436, huber: 0.1799, swd: 0.1063, ept: 79.6593
    Epoch [2/50], Test Losses: mse: 0.5381, mae: 0.5289, huber: 0.2343, swd: 0.1297, ept: 75.6842
      Epoch 2 composite train-obj: 0.208752
            Val objective improved 0.1848 → 0.1799, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4925, mae: 0.4849, huber: 0.2063, swd: 0.1621, ept: 98.8707
    Epoch [3/50], Val Losses: mse: 0.4271, mae: 0.4443, huber: 0.1815, swd: 0.1040, ept: 82.4757
    Epoch [3/50], Test Losses: mse: 0.5395, mae: 0.5289, huber: 0.2343, swd: 0.1273, ept: 76.6047
      Epoch 3 composite train-obj: 0.206258
            No improvement (0.1815), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4918, mae: 0.4838, huber: 0.2058, swd: 0.1618, ept: 100.0603
    Epoch [4/50], Val Losses: mse: 0.4308, mae: 0.4458, huber: 0.1829, swd: 0.1123, ept: 83.4971
    Epoch [4/50], Test Losses: mse: 0.5487, mae: 0.5333, huber: 0.2379, swd: 0.1303, ept: 76.8268
      Epoch 4 composite train-obj: 0.205770
            No improvement (0.1829), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4964, mae: 0.4848, huber: 0.2068, swd: 0.1637, ept: 100.8469
    Epoch [5/50], Val Losses: mse: 0.4199, mae: 0.4406, huber: 0.1791, swd: 0.1109, ept: 86.7413
    Epoch [5/50], Test Losses: mse: 0.5297, mae: 0.5230, huber: 0.2304, swd: 0.1371, ept: 77.2286
      Epoch 5 composite train-obj: 0.206837
            Val objective improved 0.1799 → 0.1791, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4871, mae: 0.4803, huber: 0.2038, swd: 0.1614, ept: 102.2466
    Epoch [6/50], Val Losses: mse: 0.4161, mae: 0.4368, huber: 0.1769, swd: 0.1040, ept: 88.2484
    Epoch [6/50], Test Losses: mse: 0.5259, mae: 0.5204, huber: 0.2287, swd: 0.1294, ept: 77.4636
      Epoch 6 composite train-obj: 0.203764
            Val objective improved 0.1791 → 0.1769, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4857, mae: 0.4792, huber: 0.2031, swd: 0.1620, ept: 102.6485
    Epoch [7/50], Val Losses: mse: 0.4243, mae: 0.4425, huber: 0.1809, swd: 0.1061, ept: 83.0930
    Epoch [7/50], Test Losses: mse: 0.5376, mae: 0.5255, huber: 0.2333, swd: 0.1239, ept: 76.3228
      Epoch 7 composite train-obj: 0.203103
            No improvement (0.1809), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4862, mae: 0.4795, huber: 0.2034, swd: 0.1614, ept: 102.7891
    Epoch [8/50], Val Losses: mse: 0.4152, mae: 0.4352, huber: 0.1768, swd: 0.1019, ept: 86.5113
    Epoch [8/50], Test Losses: mse: 0.5362, mae: 0.5239, huber: 0.2328, swd: 0.1235, ept: 78.4274
      Epoch 8 composite train-obj: 0.203397
            Val objective improved 0.1769 → 0.1768, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4850, mae: 0.4791, huber: 0.2031, swd: 0.1617, ept: 103.1489
    Epoch [9/50], Val Losses: mse: 0.4186, mae: 0.4387, huber: 0.1777, swd: 0.1006, ept: 85.3802
    Epoch [9/50], Test Losses: mse: 0.5343, mae: 0.5250, huber: 0.2321, swd: 0.1277, ept: 74.3091
      Epoch 9 composite train-obj: 0.203055
            No improvement (0.1777), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.4845, mae: 0.4782, huber: 0.2026, swd: 0.1613, ept: 103.7015
    Epoch [10/50], Val Losses: mse: 0.4135, mae: 0.4359, huber: 0.1768, swd: 0.1068, ept: 83.2098
    Epoch [10/50], Test Losses: mse: 0.5266, mae: 0.5185, huber: 0.2287, swd: 0.1278, ept: 78.5351
      Epoch 10 composite train-obj: 0.202603
            Val objective improved 0.1768 → 0.1768, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.4895, mae: 0.4796, huber: 0.2039, swd: 0.1645, ept: 103.3094
    Epoch [11/50], Val Losses: mse: 0.4232, mae: 0.4405, huber: 0.1800, swd: 0.1121, ept: 86.9003
    Epoch [11/50], Test Losses: mse: 0.5419, mae: 0.5320, huber: 0.2360, swd: 0.1446, ept: 77.6886
      Epoch 11 composite train-obj: 0.203914
            No improvement (0.1800), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.4916, mae: 0.4818, huber: 0.2053, swd: 0.1661, ept: 103.6382
    Epoch [12/50], Val Losses: mse: 0.4091, mae: 0.4314, huber: 0.1743, swd: 0.1048, ept: 85.3854
    Epoch [12/50], Test Losses: mse: 0.5326, mae: 0.5248, huber: 0.2318, swd: 0.1310, ept: 78.0429
      Epoch 12 composite train-obj: 0.205299
            Val objective improved 0.1768 → 0.1743, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.4893, mae: 0.4801, huber: 0.2041, swd: 0.1641, ept: 103.2416
    Epoch [13/50], Val Losses: mse: 0.4118, mae: 0.4336, huber: 0.1758, swd: 0.1070, ept: 85.6206
    Epoch [13/50], Test Losses: mse: 0.5265, mae: 0.5179, huber: 0.2286, swd: 0.1298, ept: 78.8371
      Epoch 13 composite train-obj: 0.204083
            No improvement (0.1758), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.4888, mae: 0.4795, huber: 0.2037, swd: 0.1650, ept: 103.7689
    Epoch [14/50], Val Losses: mse: 0.4117, mae: 0.4323, huber: 0.1747, swd: 0.0954, ept: 88.3466
    Epoch [14/50], Test Losses: mse: 0.5321, mae: 0.5235, huber: 0.2311, swd: 0.1178, ept: 76.4605
      Epoch 14 composite train-obj: 0.203680
            No improvement (0.1747), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.4837, mae: 0.4780, huber: 0.2024, swd: 0.1606, ept: 103.9777
    Epoch [15/50], Val Losses: mse: 0.4254, mae: 0.4425, huber: 0.1814, swd: 0.1116, ept: 84.4931
    Epoch [15/50], Test Losses: mse: 0.5415, mae: 0.5266, huber: 0.2347, swd: 0.1312, ept: 77.0764
      Epoch 15 composite train-obj: 0.202443
            No improvement (0.1814), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.4837, mae: 0.4777, huber: 0.2023, swd: 0.1619, ept: 103.7118
    Epoch [16/50], Val Losses: mse: 0.4188, mae: 0.4385, huber: 0.1783, swd: 0.1076, ept: 86.5427
    Epoch [16/50], Test Losses: mse: 0.5249, mae: 0.5176, huber: 0.2282, swd: 0.1287, ept: 77.8678
      Epoch 16 composite train-obj: 0.202346
            No improvement (0.1783), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.4858, mae: 0.4784, huber: 0.2029, swd: 0.1627, ept: 103.7980
    Epoch [17/50], Val Losses: mse: 0.4208, mae: 0.4395, huber: 0.1795, swd: 0.1101, ept: 82.9350
    Epoch [17/50], Test Losses: mse: 0.5269, mae: 0.5184, huber: 0.2289, swd: 0.1286, ept: 78.6269
      Epoch 17 composite train-obj: 0.202890
    Epoch [17/50], Test Losses: mse: 0.5326, mae: 0.5248, huber: 0.2318, swd: 0.1310, ept: 78.0429
    Best round's Test MSE: 0.5326, MAE: 0.5248, SWD: 0.1310
    Best round's Validation MSE: 0.4091, MAE: 0.4314, SWD: 0.1048
    Best round's Test verification MSE : 0.5326, MAE: 0.5248, SWD: 0.1310
    Time taken: 16.61 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5744, mae: 0.5337, huber: 0.2382, swd: 0.1724, ept: 71.2744
    Epoch [1/50], Val Losses: mse: 0.4356, mae: 0.4554, huber: 0.1862, swd: 0.1096, ept: 74.5942
    Epoch [1/50], Test Losses: mse: 0.5529, mae: 0.5411, huber: 0.2408, swd: 0.1341, ept: 71.8396
      Epoch 1 composite train-obj: 0.238184
            Val objective improved inf → 0.1862, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5000, mae: 0.4907, huber: 0.2096, swd: 0.1685, ept: 94.8317
    Epoch [2/50], Val Losses: mse: 0.4211, mae: 0.4435, huber: 0.1796, swd: 0.1046, ept: 78.9846
    Epoch [2/50], Test Losses: mse: 0.5351, mae: 0.5273, huber: 0.2329, swd: 0.1281, ept: 74.5876
      Epoch 2 composite train-obj: 0.209629
            Val objective improved 0.1862 → 0.1796, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4938, mae: 0.4855, huber: 0.2067, swd: 0.1654, ept: 98.9873
    Epoch [3/50], Val Losses: mse: 0.4195, mae: 0.4414, huber: 0.1792, swd: 0.1093, ept: 80.8542
    Epoch [3/50], Test Losses: mse: 0.5378, mae: 0.5299, huber: 0.2343, swd: 0.1410, ept: 77.2907
      Epoch 3 composite train-obj: 0.206737
            Val objective improved 0.1796 → 0.1792, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4882, mae: 0.4821, huber: 0.2045, swd: 0.1660, ept: 100.5410
    Epoch [4/50], Val Losses: mse: 0.4233, mae: 0.4453, huber: 0.1811, swd: 0.1009, ept: 81.3511
    Epoch [4/50], Test Losses: mse: 0.5336, mae: 0.5252, huber: 0.2321, swd: 0.1231, ept: 76.8205
      Epoch 4 composite train-obj: 0.204510
            No improvement (0.1811), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4906, mae: 0.4824, huber: 0.2051, swd: 0.1650, ept: 100.4712
    Epoch [5/50], Val Losses: mse: 0.4180, mae: 0.4419, huber: 0.1782, swd: 0.1013, ept: 85.6745
    Epoch [5/50], Test Losses: mse: 0.5400, mae: 0.5296, huber: 0.2348, swd: 0.1311, ept: 73.9552
      Epoch 5 composite train-obj: 0.205115
            Val objective improved 0.1792 → 0.1782, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4917, mae: 0.4816, huber: 0.2049, swd: 0.1697, ept: 102.2543
    Epoch [6/50], Val Losses: mse: 0.4225, mae: 0.4430, huber: 0.1805, swd: 0.1099, ept: 79.8570
    Epoch [6/50], Test Losses: mse: 0.5253, mae: 0.5205, huber: 0.2286, swd: 0.1301, ept: 78.2797
      Epoch 6 composite train-obj: 0.204922
            No improvement (0.1805), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4880, mae: 0.4806, huber: 0.2041, swd: 0.1653, ept: 102.4671
    Epoch [7/50], Val Losses: mse: 0.4170, mae: 0.4391, huber: 0.1782, swd: 0.1101, ept: 82.4405
    Epoch [7/50], Test Losses: mse: 0.5320, mae: 0.5225, huber: 0.2311, swd: 0.1334, ept: 77.6815
      Epoch 7 composite train-obj: 0.204085
            No improvement (0.1782), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4852, mae: 0.4789, huber: 0.2030, swd: 0.1654, ept: 102.8910
    Epoch [8/50], Val Losses: mse: 0.4089, mae: 0.4327, huber: 0.1745, swd: 0.0985, ept: 85.2943
    Epoch [8/50], Test Losses: mse: 0.5249, mae: 0.5181, huber: 0.2281, swd: 0.1207, ept: 78.3568
      Epoch 8 composite train-obj: 0.202962
            Val objective improved 0.1782 → 0.1745, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4846, mae: 0.4782, huber: 0.2026, swd: 0.1658, ept: 103.4752
    Epoch [9/50], Val Losses: mse: 0.4111, mae: 0.4330, huber: 0.1750, swd: 0.0997, ept: 87.0403
    Epoch [9/50], Test Losses: mse: 0.5431, mae: 0.5306, huber: 0.2360, swd: 0.1275, ept: 77.9097
      Epoch 9 composite train-obj: 0.202629
            No improvement (0.1750), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.4921, mae: 0.4815, huber: 0.2053, swd: 0.1673, ept: 103.3438
    Epoch [10/50], Val Losses: mse: 0.4151, mae: 0.4373, huber: 0.1772, swd: 0.1079, ept: 83.7636
    Epoch [10/50], Test Losses: mse: 0.5253, mae: 0.5200, huber: 0.2287, swd: 0.1302, ept: 77.6491
      Epoch 10 composite train-obj: 0.205325
            No improvement (0.1772), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.4863, mae: 0.4795, huber: 0.2035, swd: 0.1670, ept: 103.5896
    Epoch [11/50], Val Losses: mse: 0.4253, mae: 0.4467, huber: 0.1821, swd: 0.1178, ept: 80.4126
    Epoch [11/50], Test Losses: mse: 0.5303, mae: 0.5223, huber: 0.2307, swd: 0.1272, ept: 72.3196
      Epoch 11 composite train-obj: 0.203480
            No improvement (0.1821), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.4864, mae: 0.4795, huber: 0.2035, swd: 0.1662, ept: 103.0959
    Epoch [12/50], Val Losses: mse: 0.4176, mae: 0.4367, huber: 0.1777, swd: 0.1063, ept: 82.4763
    Epoch [12/50], Test Losses: mse: 0.5262, mae: 0.5193, huber: 0.2287, swd: 0.1261, ept: 78.4717
      Epoch 12 composite train-obj: 0.203464
            No improvement (0.1777), counter 4/5
    Epoch [13/50], Train Losses: mse: 0.4869, mae: 0.4792, huber: 0.2035, swd: 0.1667, ept: 103.7770
    Epoch [13/50], Val Losses: mse: 0.4196, mae: 0.4362, huber: 0.1777, swd: 0.1078, ept: 88.3141
    Epoch [13/50], Test Losses: mse: 0.5329, mae: 0.5267, huber: 0.2323, swd: 0.1338, ept: 77.2200
      Epoch 13 composite train-obj: 0.203459
    Epoch [13/50], Test Losses: mse: 0.5249, mae: 0.5181, huber: 0.2281, swd: 0.1207, ept: 78.3568
    Best round's Test MSE: 0.5249, MAE: 0.5181, SWD: 0.1207
    Best round's Validation MSE: 0.4089, MAE: 0.4327, SWD: 0.0985
    Best round's Test verification MSE : 0.5249, MAE: 0.5181, SWD: 0.1207
    Time taken: 12.93 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5691, mae: 0.5317, huber: 0.2365, swd: 0.1698, ept: 71.6955
    Epoch [1/50], Val Losses: mse: 0.4278, mae: 0.4492, huber: 0.1827, swd: 0.1006, ept: 80.3616
    Epoch [1/50], Test Losses: mse: 0.5457, mae: 0.5340, huber: 0.2374, swd: 0.1227, ept: 72.8068
      Epoch 1 composite train-obj: 0.236478
            Val objective improved inf → 0.1827, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4978, mae: 0.4886, huber: 0.2085, swd: 0.1650, ept: 95.7841
    Epoch [2/50], Val Losses: mse: 0.4201, mae: 0.4436, huber: 0.1797, swd: 0.1023, ept: 78.4977
    Epoch [2/50], Test Losses: mse: 0.5367, mae: 0.5261, huber: 0.2331, swd: 0.1250, ept: 75.5214
      Epoch 2 composite train-obj: 0.208460
            Val objective improved 0.1827 → 0.1797, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4918, mae: 0.4842, huber: 0.2060, swd: 0.1631, ept: 98.5346
    Epoch [3/50], Val Losses: mse: 0.4330, mae: 0.4464, huber: 0.1834, swd: 0.1082, ept: 83.7722
    Epoch [3/50], Test Losses: mse: 0.5458, mae: 0.5324, huber: 0.2370, swd: 0.1327, ept: 76.5681
      Epoch 3 composite train-obj: 0.205960
            No improvement (0.1834), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4940, mae: 0.4847, huber: 0.2065, swd: 0.1646, ept: 99.8878
    Epoch [4/50], Val Losses: mse: 0.4284, mae: 0.4479, huber: 0.1829, swd: 0.1094, ept: 80.1287
    Epoch [4/50], Test Losses: mse: 0.5439, mae: 0.5327, huber: 0.2365, swd: 0.1372, ept: 76.3469
      Epoch 4 composite train-obj: 0.206525
            No improvement (0.1829), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4863, mae: 0.4801, huber: 0.2035, swd: 0.1620, ept: 101.8344
    Epoch [5/50], Val Losses: mse: 0.4151, mae: 0.4357, huber: 0.1767, swd: 0.1035, ept: 83.1328
    Epoch [5/50], Test Losses: mse: 0.5307, mae: 0.5226, huber: 0.2308, swd: 0.1295, ept: 77.8581
      Epoch 5 composite train-obj: 0.203477
            Val objective improved 0.1797 → 0.1767, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4874, mae: 0.4798, huber: 0.2037, swd: 0.1639, ept: 102.4999
    Epoch [6/50], Val Losses: mse: 0.4131, mae: 0.4346, huber: 0.1759, swd: 0.1075, ept: 88.3722
    Epoch [6/50], Test Losses: mse: 0.5371, mae: 0.5288, huber: 0.2338, swd: 0.1420, ept: 78.2965
      Epoch 6 composite train-obj: 0.203657
            Val objective improved 0.1767 → 0.1759, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4881, mae: 0.4812, huber: 0.2043, swd: 0.1642, ept: 101.9238
    Epoch [7/50], Val Losses: mse: 0.4376, mae: 0.4563, huber: 0.1878, swd: 0.1177, ept: 81.3711
    Epoch [7/50], Test Losses: mse: 0.5443, mae: 0.5329, huber: 0.2364, swd: 0.1344, ept: 75.1205
      Epoch 7 composite train-obj: 0.204303
            No improvement (0.1878), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4943, mae: 0.4833, huber: 0.2060, swd: 0.1644, ept: 102.2411
    Epoch [8/50], Val Losses: mse: 0.4073, mae: 0.4306, huber: 0.1736, swd: 0.0982, ept: 83.7556
    Epoch [8/50], Test Losses: mse: 0.5299, mae: 0.5206, huber: 0.2302, swd: 0.1251, ept: 78.1276
      Epoch 8 composite train-obj: 0.205982
            Val objective improved 0.1759 → 0.1736, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4883, mae: 0.4801, huber: 0.2040, swd: 0.1644, ept: 103.5989
    Epoch [9/50], Val Losses: mse: 0.4107, mae: 0.4364, huber: 0.1755, swd: 0.1054, ept: 84.2370
    Epoch [9/50], Test Losses: mse: 0.5261, mae: 0.5202, huber: 0.2288, swd: 0.1343, ept: 76.7298
      Epoch 9 composite train-obj: 0.203994
            No improvement (0.1755), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.4863, mae: 0.4791, huber: 0.2031, swd: 0.1614, ept: 103.5539
    Epoch [10/50], Val Losses: mse: 0.4211, mae: 0.4417, huber: 0.1800, swd: 0.1082, ept: 82.4366
    Epoch [10/50], Test Losses: mse: 0.5315, mae: 0.5228, huber: 0.2311, swd: 0.1301, ept: 76.7478
      Epoch 10 composite train-obj: 0.203118
            No improvement (0.1800), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.4844, mae: 0.4785, huber: 0.2028, swd: 0.1637, ept: 103.2509
    Epoch [11/50], Val Losses: mse: 0.4201, mae: 0.4394, huber: 0.1788, swd: 0.1048, ept: 85.4933
    Epoch [11/50], Test Losses: mse: 0.5320, mae: 0.5225, huber: 0.2309, swd: 0.1292, ept: 77.4688
      Epoch 11 composite train-obj: 0.202823
            No improvement (0.1788), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.4842, mae: 0.4782, huber: 0.2026, swd: 0.1632, ept: 103.3815
    Epoch [12/50], Val Losses: mse: 0.4137, mae: 0.4341, huber: 0.1760, swd: 0.1061, ept: 86.7606
    Epoch [12/50], Test Losses: mse: 0.5276, mae: 0.5187, huber: 0.2292, swd: 0.1301, ept: 78.0685
      Epoch 12 composite train-obj: 0.202640
            No improvement (0.1760), counter 4/5
    Epoch [13/50], Train Losses: mse: 0.4884, mae: 0.4793, huber: 0.2038, swd: 0.1647, ept: 103.9624
    Epoch [13/50], Val Losses: mse: 0.4203, mae: 0.4422, huber: 0.1797, swd: 0.1172, ept: 83.8829
    Epoch [13/50], Test Losses: mse: 0.5292, mae: 0.5230, huber: 0.2304, swd: 0.1457, ept: 76.4941
      Epoch 13 composite train-obj: 0.203794
    Epoch [13/50], Test Losses: mse: 0.5299, mae: 0.5206, huber: 0.2302, swd: 0.1251, ept: 78.1276
    Best round's Test MSE: 0.5299, MAE: 0.5206, SWD: 0.1251
    Best round's Validation MSE: 0.4073, MAE: 0.4306, SWD: 0.0982
    Best round's Test verification MSE : 0.5299, MAE: 0.5206, SWD: 0.1251
    Time taken: 12.70 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq336_pred336_20250510_1916)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.5291 ± 0.0032
      mae: 0.5212 ± 0.0028
      huber: 0.2300 ± 0.0015
      swd: 0.1256 ± 0.0042
      ept: 78.1758 ± 0.1326
      count: 9.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4084 ± 0.0008
      mae: 0.4316 ± 0.0009
      huber: 0.1741 ± 0.0004
      swd: 0.1005 ± 0.0031
      ept: 84.8118 ± 0.7478
      count: 9.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 42.28 seconds
    
    Experiment complete: DLinear_etth1_seq336_pred336_20250510_1916
    Model: DLinear
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([336, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 88
    Batch 0: Data shape torch.Size([128, 336, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth1
    ==================================================
    Sequence Length: 336
    Prediction Length: 720
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
    
    Epoch [1/50], Train Losses: mse: 0.6467, mae: 0.5776, huber: 0.2677, swd: 0.2154, ept: 82.0393
    Epoch [1/50], Val Losses: mse: 0.4685, mae: 0.4901, huber: 0.2044, swd: 0.1161, ept: 91.3559
    Epoch [1/50], Test Losses: mse: 0.6745, mae: 0.6163, huber: 0.2929, swd: 0.1753, ept: 108.4257
      Epoch 1 composite train-obj: 0.267708
            Val objective improved inf → 0.2044, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5828, mae: 0.5396, huber: 0.2423, swd: 0.2141, ept: 116.3185
    Epoch [2/50], Val Losses: mse: 0.4778, mae: 0.4919, huber: 0.2070, swd: 0.1205, ept: 94.3279
    Epoch [2/50], Test Losses: mse: 0.6690, mae: 0.6098, huber: 0.2900, swd: 0.1739, ept: 114.1262
      Epoch 2 composite train-obj: 0.242261
            No improvement (0.2070), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5770, mae: 0.5352, huber: 0.2397, swd: 0.2133, ept: 122.7080
    Epoch [3/50], Val Losses: mse: 0.4687, mae: 0.4834, huber: 0.2026, swd: 0.1111, ept: 98.6079
    Epoch [3/50], Test Losses: mse: 0.6737, mae: 0.6095, huber: 0.2914, swd: 0.1669, ept: 110.8994
      Epoch 3 composite train-obj: 0.239706
            Val objective improved 0.2044 → 0.2026, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5808, mae: 0.5355, huber: 0.2404, swd: 0.2131, ept: 124.2539
    Epoch [4/50], Val Losses: mse: 0.4861, mae: 0.4956, huber: 0.2099, swd: 0.1269, ept: 88.7811
    Epoch [4/50], Test Losses: mse: 0.6604, mae: 0.6052, huber: 0.2863, swd: 0.1772, ept: 111.5885
      Epoch 4 composite train-obj: 0.240426
            No improvement (0.2099), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5744, mae: 0.5338, huber: 0.2388, swd: 0.2131, ept: 125.7540
    Epoch [5/50], Val Losses: mse: 0.4889, mae: 0.4952, huber: 0.2106, swd: 0.1211, ept: 88.8864
    Epoch [5/50], Test Losses: mse: 0.6650, mae: 0.6052, huber: 0.2877, swd: 0.1680, ept: 110.8699
      Epoch 5 composite train-obj: 0.238769
            No improvement (0.2106), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5758, mae: 0.5335, huber: 0.2390, swd: 0.2129, ept: 127.0679
    Epoch [6/50], Val Losses: mse: 0.4706, mae: 0.4834, huber: 0.2031, swd: 0.1158, ept: 93.0045
    Epoch [6/50], Test Losses: mse: 0.6679, mae: 0.6080, huber: 0.2895, swd: 0.1739, ept: 114.0960
      Epoch 6 composite train-obj: 0.238967
            No improvement (0.2031), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5712, mae: 0.5308, huber: 0.2371, swd: 0.2130, ept: 128.2799
    Epoch [7/50], Val Losses: mse: 0.4671, mae: 0.4782, huber: 0.2011, swd: 0.1082, ept: 98.8325
    Epoch [7/50], Test Losses: mse: 0.6741, mae: 0.6085, huber: 0.2914, swd: 0.1620, ept: 116.0637
      Epoch 7 composite train-obj: 0.237100
            Val objective improved 0.2026 → 0.2011, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.5724, mae: 0.5310, huber: 0.2375, swd: 0.2131, ept: 129.7379
    Epoch [8/50], Val Losses: mse: 0.4652, mae: 0.4772, huber: 0.2002, swd: 0.1160, ept: 110.9151
    Epoch [8/50], Test Losses: mse: 0.6635, mae: 0.6077, huber: 0.2883, swd: 0.1813, ept: 113.1815
      Epoch 8 composite train-obj: 0.237464
            Val objective improved 0.2011 → 0.2002, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.5708, mae: 0.5305, huber: 0.2370, swd: 0.2138, ept: 129.8262
    Epoch [9/50], Val Losses: mse: 0.4687, mae: 0.4819, huber: 0.2026, swd: 0.1185, ept: 93.9710
    Epoch [9/50], Test Losses: mse: 0.6743, mae: 0.6100, huber: 0.2920, swd: 0.1755, ept: 111.9158
      Epoch 9 composite train-obj: 0.237027
            No improvement (0.2026), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.5728, mae: 0.5310, huber: 0.2375, swd: 0.2151, ept: 129.9683
    Epoch [10/50], Val Losses: mse: 0.4701, mae: 0.4839, huber: 0.2030, swd: 0.1153, ept: 99.3915
    Epoch [10/50], Test Losses: mse: 0.6501, mae: 0.5996, huber: 0.2823, swd: 0.1702, ept: 114.1149
      Epoch 10 composite train-obj: 0.237494
            No improvement (0.2030), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.5711, mae: 0.5306, huber: 0.2371, swd: 0.2131, ept: 129.9889
    Epoch [11/50], Val Losses: mse: 0.4621, mae: 0.4754, huber: 0.1990, swd: 0.1107, ept: 117.9670
    Epoch [11/50], Test Losses: mse: 0.6614, mae: 0.6049, huber: 0.2868, swd: 0.1776, ept: 111.6246
      Epoch 11 composite train-obj: 0.237084
            Val objective improved 0.2002 → 0.1990, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.5741, mae: 0.5314, huber: 0.2378, swd: 0.2187, ept: 130.8658
    Epoch [12/50], Val Losses: mse: 0.4558, mae: 0.4704, huber: 0.1964, swd: 0.1033, ept: 110.4983
    Epoch [12/50], Test Losses: mse: 0.6706, mae: 0.6102, huber: 0.2910, swd: 0.1701, ept: 103.8922
      Epoch 12 composite train-obj: 0.237838
            Val objective improved 0.1990 → 0.1964, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.5708, mae: 0.5304, huber: 0.2370, swd: 0.2131, ept: 129.0322
    Epoch [13/50], Val Losses: mse: 0.4719, mae: 0.4843, huber: 0.2034, swd: 0.1222, ept: 92.4560
    Epoch [13/50], Test Losses: mse: 0.6502, mae: 0.5986, huber: 0.2825, swd: 0.1763, ept: 114.4007
      Epoch 13 composite train-obj: 0.237017
            No improvement (0.2034), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.5734, mae: 0.5311, huber: 0.2377, swd: 0.2160, ept: 130.4356
    Epoch [14/50], Val Losses: mse: 0.4690, mae: 0.4855, huber: 0.2029, swd: 0.1260, ept: 90.6221
    Epoch [14/50], Test Losses: mse: 0.6571, mae: 0.6020, huber: 0.2851, swd: 0.1761, ept: 113.0845
      Epoch 14 composite train-obj: 0.237679
            No improvement (0.2029), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.5706, mae: 0.5298, huber: 0.2367, swd: 0.2145, ept: 131.2410
    Epoch [15/50], Val Losses: mse: 0.4593, mae: 0.4735, huber: 0.1979, swd: 0.1097, ept: 112.9018
    Epoch [15/50], Test Losses: mse: 0.6677, mae: 0.6087, huber: 0.2897, swd: 0.1725, ept: 112.2464
      Epoch 15 composite train-obj: 0.236681
            No improvement (0.1979), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.5669, mae: 0.5283, huber: 0.2356, swd: 0.2147, ept: 132.6682
    Epoch [16/50], Val Losses: mse: 0.4578, mae: 0.4672, huber: 0.1960, swd: 0.1064, ept: 107.7266
    Epoch [16/50], Test Losses: mse: 0.6552, mae: 0.5988, huber: 0.2840, swd: 0.1641, ept: 117.0712
      Epoch 16 composite train-obj: 0.235589
            Val objective improved 0.1964 → 0.1960, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 0.5716, mae: 0.5301, huber: 0.2371, swd: 0.2149, ept: 131.2084
    Epoch [17/50], Val Losses: mse: 0.4668, mae: 0.4788, huber: 0.2008, swd: 0.1156, ept: 98.6253
    Epoch [17/50], Test Losses: mse: 0.6549, mae: 0.6007, huber: 0.2842, swd: 0.1736, ept: 116.1708
      Epoch 17 composite train-obj: 0.237095
            No improvement (0.2008), counter 1/5
    Epoch [18/50], Train Losses: mse: 0.5696, mae: 0.5296, huber: 0.2366, swd: 0.2138, ept: 130.5981
    Epoch [18/50], Val Losses: mse: 0.4828, mae: 0.4908, huber: 0.2078, swd: 0.1252, ept: 93.0596
    Epoch [18/50], Test Losses: mse: 0.6531, mae: 0.6015, huber: 0.2837, swd: 0.1770, ept: 110.9924
      Epoch 18 composite train-obj: 0.236564
            No improvement (0.2078), counter 2/5
    Epoch [19/50], Train Losses: mse: 0.5715, mae: 0.5309, huber: 0.2373, swd: 0.2159, ept: 130.1134
    Epoch [19/50], Val Losses: mse: 0.4703, mae: 0.4772, huber: 0.2017, swd: 0.1147, ept: 105.0166
    Epoch [19/50], Test Losses: mse: 0.6748, mae: 0.6084, huber: 0.2916, swd: 0.1679, ept: 117.7058
      Epoch 19 composite train-obj: 0.237341
            No improvement (0.2017), counter 3/5
    Epoch [20/50], Train Losses: mse: 0.5774, mae: 0.5325, huber: 0.2389, swd: 0.2152, ept: 131.0820
    Epoch [20/50], Val Losses: mse: 0.4731, mae: 0.4887, huber: 0.2048, swd: 0.1302, ept: 91.2338
    Epoch [20/50], Test Losses: mse: 0.6462, mae: 0.5969, huber: 0.2808, swd: 0.1827, ept: 117.1572
      Epoch 20 composite train-obj: 0.238861
            No improvement (0.2048), counter 4/5
    Epoch [21/50], Train Losses: mse: 0.5682, mae: 0.5291, huber: 0.2361, swd: 0.2143, ept: 130.8501
    Epoch [21/50], Val Losses: mse: 0.4608, mae: 0.4712, huber: 0.1977, swd: 0.1069, ept: 120.7754
    Epoch [21/50], Test Losses: mse: 0.6490, mae: 0.5978, huber: 0.2818, swd: 0.1751, ept: 111.0659
      Epoch 21 composite train-obj: 0.236114
    Epoch [21/50], Test Losses: mse: 0.6552, mae: 0.5988, huber: 0.2840, swd: 0.1641, ept: 117.0712
    Best round's Test MSE: 0.6552, MAE: 0.5988, SWD: 0.1641
    Best round's Validation MSE: 0.4578, MAE: 0.4672, SWD: 0.1064
    Best round's Test verification MSE : 0.6552, MAE: 0.5988, SWD: 0.1641
    Time taken: 20.49 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6504, mae: 0.5792, huber: 0.2690, swd: 0.2092, ept: 82.3530
    Epoch [1/50], Val Losses: mse: 0.4794, mae: 0.4936, huber: 0.2080, swd: 0.1074, ept: 94.5872
    Epoch [1/50], Test Losses: mse: 0.6827, mae: 0.6207, huber: 0.2963, swd: 0.1677, ept: 108.0848
      Epoch 1 composite train-obj: 0.269031
            Val objective improved inf → 0.2080, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5863, mae: 0.5415, huber: 0.2436, swd: 0.2108, ept: 115.8355
    Epoch [2/50], Val Losses: mse: 0.4789, mae: 0.4904, huber: 0.2071, swd: 0.1080, ept: 91.0846
    Epoch [2/50], Test Losses: mse: 0.6814, mae: 0.6148, huber: 0.2946, swd: 0.1620, ept: 110.7755
      Epoch 2 composite train-obj: 0.243587
            Val objective improved 0.2080 → 0.2071, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5822, mae: 0.5380, huber: 0.2418, swd: 0.2095, ept: 121.9040
    Epoch [3/50], Val Losses: mse: 0.4794, mae: 0.4864, huber: 0.2059, swd: 0.1052, ept: 100.2777
    Epoch [3/50], Test Losses: mse: 0.6805, mae: 0.6147, huber: 0.2943, swd: 0.1632, ept: 110.6204
      Epoch 3 composite train-obj: 0.241803
            Val objective improved 0.2071 → 0.2059, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5796, mae: 0.5355, huber: 0.2403, swd: 0.2088, ept: 123.8468
    Epoch [4/50], Val Losses: mse: 0.4714, mae: 0.4852, huber: 0.2036, swd: 0.1124, ept: 94.4953
    Epoch [4/50], Test Losses: mse: 0.6550, mae: 0.6009, huber: 0.2842, swd: 0.1705, ept: 114.7397
      Epoch 4 composite train-obj: 0.240293
            Val objective improved 0.2059 → 0.2036, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5745, mae: 0.5331, huber: 0.2386, swd: 0.2106, ept: 127.2238
    Epoch [5/50], Val Losses: mse: 0.4444, mae: 0.4577, huber: 0.1901, swd: 0.0888, ept: 137.7490
    Epoch [5/50], Test Losses: mse: 0.6842, mae: 0.6174, huber: 0.2965, swd: 0.1651, ept: 90.0413
      Epoch 5 composite train-obj: 0.238617
            Val objective improved 0.2036 → 0.1901, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5762, mae: 0.5335, huber: 0.2388, swd: 0.2102, ept: 127.0706
    Epoch [6/50], Val Losses: mse: 0.4667, mae: 0.4766, huber: 0.2003, swd: 0.1009, ept: 106.5079
    Epoch [6/50], Test Losses: mse: 0.6721, mae: 0.6109, huber: 0.2915, swd: 0.1636, ept: 113.5013
      Epoch 6 composite train-obj: 0.238836
            No improvement (0.2003), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.5709, mae: 0.5311, huber: 0.2372, swd: 0.2073, ept: 128.1692
    Epoch [7/50], Val Losses: mse: 0.4698, mae: 0.4829, huber: 0.2026, swd: 0.1069, ept: 96.5143
    Epoch [7/50], Test Losses: mse: 0.6616, mae: 0.6027, huber: 0.2865, swd: 0.1547, ept: 111.4905
      Epoch 7 composite train-obj: 0.237212
            No improvement (0.2026), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.5690, mae: 0.5297, huber: 0.2364, swd: 0.2078, ept: 128.1741
    Epoch [8/50], Val Losses: mse: 0.4702, mae: 0.4785, huber: 0.2020, swd: 0.1062, ept: 92.2335
    Epoch [8/50], Test Losses: mse: 0.6696, mae: 0.6065, huber: 0.2896, swd: 0.1613, ept: 113.0927
      Epoch 8 composite train-obj: 0.236350
            No improvement (0.2020), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.5708, mae: 0.5305, huber: 0.2370, swd: 0.2093, ept: 129.8534
    Epoch [9/50], Val Losses: mse: 0.4548, mae: 0.4684, huber: 0.1956, swd: 0.0997, ept: 118.6174
    Epoch [9/50], Test Losses: mse: 0.6542, mae: 0.6011, huber: 0.2840, swd: 0.1673, ept: 114.7945
      Epoch 9 composite train-obj: 0.236991
            No improvement (0.1956), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.5707, mae: 0.5302, huber: 0.2369, swd: 0.2087, ept: 129.5868
    Epoch [10/50], Val Losses: mse: 0.4588, mae: 0.4748, huber: 0.1980, swd: 0.0993, ept: 106.3135
    Epoch [10/50], Test Losses: mse: 0.6693, mae: 0.6073, huber: 0.2899, swd: 0.1630, ept: 110.1024
      Epoch 10 composite train-obj: 0.236860
    Epoch [10/50], Test Losses: mse: 0.6842, mae: 0.6174, huber: 0.2965, swd: 0.1651, ept: 90.0413
    Best round's Test MSE: 0.6842, MAE: 0.6174, SWD: 0.1651
    Best round's Validation MSE: 0.4444, MAE: 0.4577, SWD: 0.0888
    Best round's Test verification MSE : 0.6842, MAE: 0.6174, SWD: 0.1651
    Time taken: 10.07 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6483, mae: 0.5784, huber: 0.2684, swd: 0.2251, ept: 82.5386
    Epoch [1/50], Val Losses: mse: 0.4758, mae: 0.4915, huber: 0.2065, swd: 0.1157, ept: 95.6730
    Epoch [1/50], Test Losses: mse: 0.6852, mae: 0.6218, huber: 0.2973, swd: 0.1738, ept: 102.3565
      Epoch 1 composite train-obj: 0.268402
            Val objective improved inf → 0.2065, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5818, mae: 0.5396, huber: 0.2421, swd: 0.2222, ept: 115.8726
    Epoch [2/50], Val Losses: mse: 0.4659, mae: 0.4797, huber: 0.2012, swd: 0.1096, ept: 106.0124
    Epoch [2/50], Test Losses: mse: 0.6810, mae: 0.6150, huber: 0.2947, swd: 0.1703, ept: 107.5653
      Epoch 2 composite train-obj: 0.242133
            Val objective improved 0.2065 → 0.2012, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5785, mae: 0.5358, huber: 0.2403, swd: 0.2230, ept: 122.6412
    Epoch [3/50], Val Losses: mse: 0.4748, mae: 0.4858, huber: 0.2046, swd: 0.1215, ept: 97.1059
    Epoch [3/50], Test Losses: mse: 0.6629, mae: 0.6075, huber: 0.2877, swd: 0.1873, ept: 113.2541
      Epoch 3 composite train-obj: 0.240269
            No improvement (0.2046), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5748, mae: 0.5337, huber: 0.2387, swd: 0.2222, ept: 125.7443
    Epoch [4/50], Val Losses: mse: 0.4703, mae: 0.4815, huber: 0.2024, swd: 0.1164, ept: 91.0132
    Epoch [4/50], Test Losses: mse: 0.6571, mae: 0.6022, huber: 0.2849, swd: 0.1786, ept: 113.2494
      Epoch 4 composite train-obj: 0.238733
            No improvement (0.2024), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5735, mae: 0.5320, huber: 0.2379, swd: 0.2220, ept: 127.5079
    Epoch [5/50], Val Losses: mse: 0.4635, mae: 0.4784, huber: 0.2004, swd: 0.1152, ept: 99.8346
    Epoch [5/50], Test Losses: mse: 0.6670, mae: 0.6074, huber: 0.2889, swd: 0.1805, ept: 109.7222
      Epoch 5 composite train-obj: 0.237934
            Val objective improved 0.2012 → 0.2004, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5760, mae: 0.5330, huber: 0.2389, swd: 0.2225, ept: 127.3967
    Epoch [6/50], Val Losses: mse: 0.4650, mae: 0.4797, huber: 0.2008, swd: 0.1191, ept: 99.3246
    Epoch [6/50], Test Losses: mse: 0.6644, mae: 0.6060, huber: 0.2879, swd: 0.1860, ept: 117.4106
      Epoch 6 composite train-obj: 0.238886
            No improvement (0.2008), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.5753, mae: 0.5331, huber: 0.2387, swd: 0.2255, ept: 128.0393
    Epoch [7/50], Val Losses: mse: 0.4703, mae: 0.4842, huber: 0.2035, swd: 0.1242, ept: 104.5974
    Epoch [7/50], Test Losses: mse: 0.6644, mae: 0.6056, huber: 0.2879, swd: 0.1809, ept: 112.9095
      Epoch 7 composite train-obj: 0.238712
            No improvement (0.2035), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.5742, mae: 0.5323, huber: 0.2382, swd: 0.2228, ept: 128.4243
    Epoch [8/50], Val Losses: mse: 0.4808, mae: 0.4937, huber: 0.2083, swd: 0.1354, ept: 89.3116
    Epoch [8/50], Test Losses: mse: 0.6766, mae: 0.6126, huber: 0.2930, swd: 0.1825, ept: 108.2031
      Epoch 8 composite train-obj: 0.238195
            No improvement (0.2083), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.5689, mae: 0.5298, huber: 0.2364, swd: 0.2226, ept: 129.6091
    Epoch [9/50], Val Losses: mse: 0.4559, mae: 0.4688, huber: 0.1960, swd: 0.1089, ept: 114.2039
    Epoch [9/50], Test Losses: mse: 0.6627, mae: 0.6029, huber: 0.2871, swd: 0.1754, ept: 109.7085
      Epoch 9 composite train-obj: 0.236361
            Val objective improved 0.2004 → 0.1960, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.5691, mae: 0.5296, huber: 0.2364, swd: 0.2225, ept: 128.9249
    Epoch [10/50], Val Losses: mse: 0.4681, mae: 0.4805, huber: 0.2015, swd: 0.1179, ept: 107.9067
    Epoch [10/50], Test Losses: mse: 0.6454, mae: 0.5968, huber: 0.2804, swd: 0.1799, ept: 116.1112
      Epoch 10 composite train-obj: 0.236386
            No improvement (0.2015), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.5690, mae: 0.5293, huber: 0.2362, swd: 0.2233, ept: 130.3617
    Epoch [11/50], Val Losses: mse: 0.4623, mae: 0.4754, huber: 0.1993, swd: 0.1163, ept: 99.3042
    Epoch [11/50], Test Losses: mse: 0.6675, mae: 0.6061, huber: 0.2891, swd: 0.1752, ept: 116.6553
      Epoch 11 composite train-obj: 0.236224
            No improvement (0.1993), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.5695, mae: 0.5296, huber: 0.2365, swd: 0.2233, ept: 130.0117
    Epoch [12/50], Val Losses: mse: 0.4791, mae: 0.4909, huber: 0.2074, swd: 0.1285, ept: 86.8177
    Epoch [12/50], Test Losses: mse: 0.6635, mae: 0.6039, huber: 0.2875, swd: 0.1741, ept: 107.8832
      Epoch 12 composite train-obj: 0.236513
            No improvement (0.2074), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.5708, mae: 0.5301, huber: 0.2368, swd: 0.2219, ept: 129.9578
    Epoch [13/50], Val Losses: mse: 0.4686, mae: 0.4795, huber: 0.2017, swd: 0.1163, ept: 91.2327
    Epoch [13/50], Test Losses: mse: 0.6685, mae: 0.6063, huber: 0.2893, swd: 0.1764, ept: 114.1422
      Epoch 13 composite train-obj: 0.236814
            No improvement (0.2017), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.5736, mae: 0.5314, huber: 0.2378, swd: 0.2237, ept: 129.5796
    Epoch [14/50], Val Losses: mse: 0.4541, mae: 0.4694, huber: 0.1955, swd: 0.1117, ept: 98.0929
    Epoch [14/50], Test Losses: mse: 0.6633, mae: 0.6050, huber: 0.2876, swd: 0.1800, ept: 110.9464
      Epoch 14 composite train-obj: 0.237810
            Val objective improved 0.1960 → 0.1955, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.5686, mae: 0.5289, huber: 0.2360, swd: 0.2228, ept: 131.0210
    Epoch [15/50], Val Losses: mse: 0.4631, mae: 0.4777, huber: 0.2001, swd: 0.1212, ept: 98.1697
    Epoch [15/50], Test Losses: mse: 0.6650, mae: 0.6055, huber: 0.2883, swd: 0.1816, ept: 116.4467
      Epoch 15 composite train-obj: 0.236025
            No improvement (0.2001), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.5689, mae: 0.5292, huber: 0.2363, swd: 0.2234, ept: 131.5619
    Epoch [16/50], Val Losses: mse: 0.4735, mae: 0.4858, huber: 0.2047, swd: 0.1229, ept: 100.7997
    Epoch [16/50], Test Losses: mse: 0.6600, mae: 0.6049, huber: 0.2864, swd: 0.1862, ept: 116.6774
      Epoch 16 composite train-obj: 0.236291
            No improvement (0.2047), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.5727, mae: 0.5306, huber: 0.2374, swd: 0.2245, ept: 130.3551
    Epoch [17/50], Val Losses: mse: 0.4545, mae: 0.4694, huber: 0.1955, swd: 0.1089, ept: 118.5237
    Epoch [17/50], Test Losses: mse: 0.6556, mae: 0.6010, huber: 0.2845, swd: 0.1785, ept: 111.6425
      Epoch 17 composite train-obj: 0.237402
            No improvement (0.1955), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.5681, mae: 0.5287, huber: 0.2359, swd: 0.2225, ept: 130.9090
    Epoch [18/50], Val Losses: mse: 0.4726, mae: 0.4859, huber: 0.2041, swd: 0.1230, ept: 91.0232
    Epoch [18/50], Test Losses: mse: 0.6621, mae: 0.6037, huber: 0.2867, swd: 0.1800, ept: 111.4651
      Epoch 18 composite train-obj: 0.235896
            No improvement (0.2041), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.5740, mae: 0.5318, huber: 0.2382, swd: 0.2250, ept: 130.2762
    Epoch [19/50], Val Losses: mse: 0.4548, mae: 0.4713, huber: 0.1963, swd: 0.1063, ept: 103.9572
    Epoch [19/50], Test Losses: mse: 0.6612, mae: 0.6041, huber: 0.2866, swd: 0.1731, ept: 104.8986
      Epoch 19 composite train-obj: 0.238170
    Epoch [19/50], Test Losses: mse: 0.6633, mae: 0.6050, huber: 0.2876, swd: 0.1800, ept: 110.9464
    Best round's Test MSE: 0.6633, MAE: 0.6050, SWD: 0.1800
    Best round's Validation MSE: 0.4541, MAE: 0.4694, SWD: 0.1117
    Best round's Test verification MSE : 0.6633, MAE: 0.6050, SWD: 0.1800
    Time taken: 18.52 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq336_pred720_20250510_1915)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.6676 ± 0.0122
      mae: 0.6071 ± 0.0077
      huber: 0.2894 ± 0.0052
      swd: 0.1697 ± 0.0072
      ept: 106.0196 ± 11.5718
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.4521 ± 0.0057
      mae: 0.4648 ± 0.0051
      huber: 0.1939 ± 0.0026
      swd: 0.1023 ± 0.0098
      ept: 114.5228 ± 16.8877
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 49.16 seconds
    
    Experiment complete: DLinear_etth1_seq336_pred720_20250510_1915
    Model: DLinear
    Dataset: etth1
    Sequence Length: 336
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    




