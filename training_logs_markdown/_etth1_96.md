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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([7.0675, 2.0423, 6.8268, 1.8092, 1.1645, 0.5995, 8.5667],
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 8
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 15.1362, mae: 2.2047, huber: 1.8004, swd: 6.8679, ept: 31.0176
    Epoch [1/50], Val Losses: mse: 8.4542, mae: 1.6360, huber: 1.2622, swd: 1.4982, ept: 43.5000
    Epoch [1/50], Test Losses: mse: 14.8318, mae: 2.0643, huber: 1.6768, swd: 3.7588, ept: 27.4919
      Epoch 1 composite train-obj: 1.800378
            Val objective improved inf → 1.2622, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.1113, mae: 1.5846, huber: 1.2050, swd: 1.9172, ept: 43.3500
    Epoch [2/50], Val Losses: mse: 8.5335, mae: 1.6652, huber: 1.2954, swd: 1.9070, ept: 43.6823
    Epoch [2/50], Test Losses: mse: 10.4888, mae: 1.8565, huber: 1.4729, swd: 2.9593, ept: 28.9886
      Epoch 2 composite train-obj: 1.205024
            No improvement (1.2954), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.2616, mae: 1.5080, huber: 1.1331, swd: 1.6197, ept: 45.5215
    Epoch [3/50], Val Losses: mse: 8.2580, mae: 1.6394, huber: 1.2705, swd: 1.8845, ept: 44.1172
    Epoch [3/50], Test Losses: mse: 9.8318, mae: 1.8086, huber: 1.4269, swd: 2.5440, ept: 29.9402
      Epoch 3 composite train-obj: 1.133116
            No improvement (1.2705), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.0060, mae: 1.4711, huber: 1.0993, swd: 1.5240, ept: 47.0563
    Epoch [4/50], Val Losses: mse: 8.9161, mae: 1.7132, huber: 1.3423, swd: 2.5651, ept: 42.9585
    Epoch [4/50], Test Losses: mse: 9.6772, mae: 1.8227, huber: 1.4375, swd: 2.9002, ept: 29.7106
      Epoch 4 composite train-obj: 1.099335
            No improvement (1.3423), counter 3/5
    Epoch [5/50], Train Losses: mse: 5.7235, mae: 1.4263, huber: 1.0577, swd: 1.3639, ept: 48.4523
    Epoch [5/50], Val Losses: mse: 8.7118, mae: 1.6860, huber: 1.3167, swd: 2.3413, ept: 42.8814
    Epoch [5/50], Test Losses: mse: 9.5852, mae: 1.7871, huber: 1.4076, swd: 2.5475, ept: 31.5061
      Epoch 5 composite train-obj: 1.057653
            No improvement (1.3167), counter 4/5
    Epoch [6/50], Train Losses: mse: 5.5401, mae: 1.3986, huber: 1.0317, swd: 1.2940, ept: 49.3442
    Epoch [6/50], Val Losses: mse: 8.9954, mae: 1.6981, huber: 1.3276, swd: 2.1668, ept: 43.7310
    Epoch [6/50], Test Losses: mse: 9.4894, mae: 1.7609, huber: 1.3838, swd: 2.2171, ept: 32.8530
      Epoch 6 composite train-obj: 1.031716
    Epoch [6/50], Test Losses: mse: 14.8319, mae: 2.0643, huber: 1.6769, swd: 3.7593, ept: 27.4851
    Best round's Test MSE: 14.8318, MAE: 2.0643, SWD: 3.7588
    Best round's Validation MSE: 8.4542, MAE: 1.6360, SWD: 1.4982
    Best round's Test verification MSE : 14.8319, MAE: 2.0643, SWD: 3.7593
    Time taken: 24.36 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 15.2804, mae: 2.2143, huber: 1.8092, swd: 6.6667, ept: 31.0314
    Epoch [1/50], Val Losses: mse: 8.7277, mae: 1.6777, huber: 1.3052, swd: 1.9267, ept: 44.9421
    Epoch [1/50], Test Losses: mse: 14.3534, mae: 2.0644, huber: 1.6761, swd: 4.0815, ept: 27.6282
      Epoch 1 composite train-obj: 1.809187
            Val objective improved inf → 1.3052, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.9793, mae: 1.5779, huber: 1.1973, swd: 1.8268, ept: 43.1971
    Epoch [2/50], Val Losses: mse: 8.2634, mae: 1.6449, huber: 1.2745, swd: 1.4359, ept: 42.8135
    Epoch [2/50], Test Losses: mse: 10.3240, mae: 1.8341, huber: 1.4499, swd: 2.6318, ept: 28.5784
      Epoch 2 composite train-obj: 1.197267
            Val objective improved 1.3052 → 1.2745, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.2647, mae: 1.5108, huber: 1.1351, swd: 1.6054, ept: 45.2467
    Epoch [3/50], Val Losses: mse: 8.8058, mae: 1.6784, huber: 1.3077, swd: 1.4751, ept: 42.0202
    Epoch [3/50], Test Losses: mse: 9.5196, mae: 1.7714, huber: 1.3903, swd: 2.0251, ept: 29.8045
      Epoch 3 composite train-obj: 1.135055
            No improvement (1.3077), counter 1/5
    Epoch [4/50], Train Losses: mse: 5.8987, mae: 1.4571, huber: 1.0858, swd: 1.4253, ept: 47.3513
    Epoch [4/50], Val Losses: mse: 8.4609, mae: 1.6509, huber: 1.2832, swd: 1.9501, ept: 44.1869
    Epoch [4/50], Test Losses: mse: 9.5082, mae: 1.7794, huber: 1.3997, swd: 2.4797, ept: 30.9039
      Epoch 4 composite train-obj: 1.085762
            No improvement (1.2832), counter 2/5
    Epoch [5/50], Train Losses: mse: 5.6653, mae: 1.4184, huber: 1.0502, swd: 1.3003, ept: 48.6788
    Epoch [5/50], Val Losses: mse: 8.8110, mae: 1.6801, huber: 1.3117, swd: 2.1346, ept: 44.1826
    Epoch [5/50], Test Losses: mse: 9.3901, mae: 1.7650, huber: 1.3861, swd: 2.3961, ept: 31.8885
      Epoch 5 composite train-obj: 1.050176
            No improvement (1.3117), counter 3/5
    Epoch [6/50], Train Losses: mse: 5.5700, mae: 1.4045, huber: 1.0373, swd: 1.2825, ept: 49.3840
    Epoch [6/50], Val Losses: mse: 8.8031, mae: 1.6590, huber: 1.2909, swd: 1.6922, ept: 44.5752
    Epoch [6/50], Test Losses: mse: 9.4485, mae: 1.7572, huber: 1.3799, swd: 2.1220, ept: 32.9028
      Epoch 6 composite train-obj: 1.037287
            No improvement (1.2909), counter 4/5
    Epoch [7/50], Train Losses: mse: 5.3733, mae: 1.3695, huber: 1.0050, swd: 1.1691, ept: 50.4378
    Epoch [7/50], Val Losses: mse: 9.4811, mae: 1.7647, huber: 1.3941, swd: 3.0508, ept: 45.4291
    Epoch [7/50], Test Losses: mse: 9.5873, mae: 1.8113, huber: 1.4290, swd: 2.8915, ept: 32.1171
      Epoch 7 composite train-obj: 1.005021
    Epoch [7/50], Test Losses: mse: 10.3242, mae: 1.8341, huber: 1.4499, swd: 2.6320, ept: 28.5771
    Best round's Test MSE: 10.3240, MAE: 1.8341, SWD: 2.6318
    Best round's Validation MSE: 8.2634, MAE: 1.6449, SWD: 1.4359
    Best round's Test verification MSE : 10.3242, MAE: 1.8341, SWD: 2.6320
    Time taken: 20.70 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 15.7163, mae: 2.2450, huber: 1.8393, swd: 6.6315, ept: 30.2990
    Epoch [1/50], Val Losses: mse: 8.7690, mae: 1.6808, huber: 1.3036, swd: 1.4618, ept: 43.4916
    Epoch [1/50], Test Losses: mse: 14.7842, mae: 2.0645, huber: 1.6746, swd: 3.7770, ept: 27.4801
      Epoch 1 composite train-obj: 1.839344
            Val objective improved inf → 1.3036, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.1011, mae: 1.5825, huber: 1.2019, swd: 1.8438, ept: 43.2627
    Epoch [2/50], Val Losses: mse: 8.8239, mae: 1.7029, huber: 1.3321, swd: 1.9381, ept: 44.6329
    Epoch [2/50], Test Losses: mse: 10.4369, mae: 1.8627, huber: 1.4777, swd: 3.4272, ept: 28.8322
      Epoch 2 composite train-obj: 1.201920
            No improvement (1.3321), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.2180, mae: 1.5045, huber: 1.1287, swd: 1.5448, ept: 45.3841
    Epoch [3/50], Val Losses: mse: 8.4108, mae: 1.6485, huber: 1.2803, swd: 1.7299, ept: 45.4948
    Epoch [3/50], Test Losses: mse: 9.5541, mae: 1.7779, huber: 1.3969, swd: 2.4674, ept: 30.1310
      Epoch 3 composite train-obj: 1.128734
            Val objective improved 1.3036 → 1.2803, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.9133, mae: 1.4635, huber: 1.0908, swd: 1.4233, ept: 47.2267
    Epoch [4/50], Val Losses: mse: 8.9674, mae: 1.6923, huber: 1.3236, swd: 1.8880, ept: 43.9768
    Epoch [4/50], Test Losses: mse: 9.2710, mae: 1.7579, huber: 1.3775, swd: 2.2542, ept: 31.1309
      Epoch 4 composite train-obj: 1.090848
            No improvement (1.3236), counter 1/5
    Epoch [5/50], Train Losses: mse: 5.6645, mae: 1.4202, huber: 1.0514, swd: 1.2717, ept: 48.6481
    Epoch [5/50], Val Losses: mse: 8.7945, mae: 1.6902, huber: 1.3215, swd: 2.1577, ept: 45.3792
    Epoch [5/50], Test Losses: mse: 9.3402, mae: 1.7637, huber: 1.3842, swd: 2.3869, ept: 31.7772
      Epoch 5 composite train-obj: 1.051367
            No improvement (1.3215), counter 2/5
    Epoch [6/50], Train Losses: mse: 5.5353, mae: 1.3979, huber: 1.0309, swd: 1.2105, ept: 49.4351
    Epoch [6/50], Val Losses: mse: 9.3286, mae: 1.7487, huber: 1.3781, swd: 2.5920, ept: 45.2528
    Epoch [6/50], Test Losses: mse: 9.3577, mae: 1.7769, huber: 1.3963, swd: 2.5387, ept: 32.1095
      Epoch 6 composite train-obj: 1.030919
            No improvement (1.3781), counter 3/5
    Epoch [7/50], Train Losses: mse: 5.3450, mae: 1.3660, huber: 1.0014, swd: 1.1060, ept: 50.3745
    Epoch [7/50], Val Losses: mse: 9.1678, mae: 1.7261, huber: 1.3567, swd: 2.2668, ept: 44.7870
    Epoch [7/50], Test Losses: mse: 9.4009, mae: 1.7670, huber: 1.3873, swd: 2.3285, ept: 32.5396
      Epoch 7 composite train-obj: 1.001412
            No improvement (1.3567), counter 4/5
    Epoch [8/50], Train Losses: mse: 5.1950, mae: 1.3419, huber: 0.9792, swd: 1.0518, ept: 50.9979
    Epoch [8/50], Val Losses: mse: 9.5199, mae: 1.7515, huber: 1.3817, swd: 2.3447, ept: 44.6676
    Epoch [8/50], Test Losses: mse: 9.6729, mae: 1.7738, huber: 1.3950, swd: 2.3419, ept: 32.5951
      Epoch 8 composite train-obj: 0.979189
    Epoch [8/50], Test Losses: mse: 9.5538, mae: 1.7779, huber: 1.3969, swd: 2.4678, ept: 30.1291
    Best round's Test MSE: 9.5541, MAE: 1.7779, SWD: 2.4674
    Best round's Validation MSE: 8.4108, MAE: 1.6485, SWD: 1.7299
    Best round's Test verification MSE : 9.5538, MAE: 1.7779, SWD: 2.4678
    Time taken: 20.16 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq720_pred96_20250510_1546)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.5700 ± 2.3278
      mae: 1.8921 ± 0.1239
      huber: 1.5079 ± 0.1214
      swd: 2.9527 ± 0.5739
      ept: 28.7338 ± 1.0830
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 8.3761 ± 0.0816
      mae: 1.6431 ± 0.0052
      huber: 1.2723 ± 0.0076
      swd: 1.5547 ± 0.1265
      ept: 43.9361 ± 1.1372
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 66.48 seconds
    
    Experiment complete: ACL_etth1_seq720_pred96_20250510_1546
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([7.0675, 2.0423, 6.8268, 1.8092, 1.1645, 0.5995, 8.5667],
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 7
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 16.3271, mae: 2.3153, huber: 1.9064, swd: 7.2948, ept: 43.0308
    Epoch [1/50], Val Losses: mse: 10.0431, mae: 1.8030, huber: 1.4225, swd: 1.9128, ept: 59.1293
    Epoch [1/50], Test Losses: mse: 16.2636, mae: 2.1779, huber: 1.7840, swd: 4.1549, ept: 40.2202
      Epoch 1 composite train-obj: 1.906419
            Val objective improved inf → 1.4225, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.7693, mae: 1.6714, huber: 1.2847, swd: 1.9261, ept: 63.5555
    Epoch [2/50], Val Losses: mse: 10.9317, mae: 1.9131, huber: 1.5336, swd: 2.4318, ept: 55.9716
    Epoch [2/50], Test Losses: mse: 11.1062, mae: 1.9702, huber: 1.5743, swd: 3.7546, ept: 40.8179
      Epoch 2 composite train-obj: 1.284744
            No improvement (1.5336), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.8699, mae: 1.6076, huber: 1.2240, swd: 1.6707, ept: 65.7374
    Epoch [3/50], Val Losses: mse: 11.6739, mae: 1.9981, huber: 1.6152, swd: 3.1922, ept: 55.4785
    Epoch [3/50], Test Losses: mse: 10.5949, mae: 1.9835, huber: 1.5849, swd: 3.9185, ept: 40.8651
      Epoch 3 composite train-obj: 1.223974
            No improvement (1.6152), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.4922, mae: 1.5579, huber: 1.1775, swd: 1.4545, ept: 67.8880
    Epoch [4/50], Val Losses: mse: 11.4279, mae: 1.9211, huber: 1.5434, swd: 2.0405, ept: 54.6969
    Epoch [4/50], Test Losses: mse: 9.9352, mae: 1.8562, huber: 1.4697, swd: 2.4673, ept: 42.8436
      Epoch 4 composite train-obj: 1.177511
            No improvement (1.5434), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.2575, mae: 1.5220, huber: 1.1445, swd: 1.3246, ept: 69.5115
    Epoch [5/50], Val Losses: mse: 12.6911, mae: 2.0309, huber: 1.6515, swd: 3.0689, ept: 58.0885
    Epoch [5/50], Test Losses: mse: 9.9957, mae: 1.9116, huber: 1.5192, swd: 3.1410, ept: 42.9891
      Epoch 5 composite train-obj: 1.144491
            No improvement (1.6515), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.0915, mae: 1.4933, huber: 1.1184, swd: 1.2363, ept: 70.7195
    Epoch [6/50], Val Losses: mse: 12.7147, mae: 2.0130, huber: 1.6343, swd: 2.6776, ept: 55.6785
    Epoch [6/50], Test Losses: mse: 9.6120, mae: 1.8344, huber: 1.4488, swd: 2.3726, ept: 44.0411
      Epoch 6 composite train-obj: 1.118401
    Epoch [6/50], Test Losses: mse: 16.2640, mae: 2.1779, huber: 1.7841, swd: 4.1557, ept: 40.2298
    Best round's Test MSE: 16.2636, MAE: 2.1779, SWD: 4.1549
    Best round's Validation MSE: 10.0431, MAE: 1.8030, SWD: 1.9128
    Best round's Test verification MSE : 16.2640, MAE: 2.1779, SWD: 4.1557
    Time taken: 18.85 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 16.7012, mae: 2.3322, huber: 1.9232, swd: 7.4646, ept: 42.7000
    Epoch [1/50], Val Losses: mse: 9.9578, mae: 1.8011, huber: 1.4234, swd: 2.1226, ept: 64.7222
    Epoch [1/50], Test Losses: mse: 15.8956, mae: 2.1681, huber: 1.7758, swd: 4.0712, ept: 40.4311
      Epoch 1 composite train-obj: 1.923164
            Val objective improved inf → 1.4234, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.6656, mae: 1.6658, huber: 1.2793, swd: 1.8602, ept: 63.9693
    Epoch [2/50], Val Losses: mse: 11.4264, mae: 1.9347, huber: 1.5590, swd: 2.9002, ept: 60.2182
    Epoch [2/50], Test Losses: mse: 10.9589, mae: 1.9518, huber: 1.5608, swd: 3.5633, ept: 42.6565
      Epoch 2 composite train-obj: 1.279272
            No improvement (1.5590), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.7719, mae: 1.5951, huber: 1.2124, swd: 1.6289, ept: 67.0023
    Epoch [3/50], Val Losses: mse: 11.9008, mae: 1.9634, huber: 1.5839, swd: 1.8202, ept: 53.2468
    Epoch [3/50], Test Losses: mse: 10.0377, mae: 1.8670, huber: 1.4782, swd: 2.0727, ept: 41.5934
      Epoch 3 composite train-obj: 1.212363
            No improvement (1.5839), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.4042, mae: 1.5449, huber: 1.1659, swd: 1.4133, ept: 69.6352
    Epoch [4/50], Val Losses: mse: 12.4646, mae: 2.0181, huber: 1.6310, swd: 2.4032, ept: 47.4258
    Epoch [4/50], Test Losses: mse: 9.7562, mae: 1.8485, huber: 1.4576, swd: 2.1521, ept: 42.5901
      Epoch 4 composite train-obj: 1.165906
            No improvement (1.6310), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.2266, mae: 1.5216, huber: 1.1443, swd: 1.3337, ept: 70.3422
    Epoch [5/50], Val Losses: mse: 12.5510, mae: 1.9853, huber: 1.6092, swd: 2.5295, ept: 54.3149
    Epoch [5/50], Test Losses: mse: 9.5880, mae: 1.8282, huber: 1.4418, swd: 2.2788, ept: 44.4073
      Epoch 5 composite train-obj: 1.144320
            No improvement (1.6092), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.0341, mae: 1.4874, huber: 1.1130, swd: 1.2401, ept: 71.8743
    Epoch [6/50], Val Losses: mse: 13.4033, mae: 2.0715, huber: 1.6921, swd: 3.5057, ept: 56.1139
    Epoch [6/50], Test Losses: mse: 9.7545, mae: 1.8760, huber: 1.4869, swd: 2.8219, ept: 45.1386
      Epoch 6 composite train-obj: 1.112996
    Epoch [6/50], Test Losses: mse: 15.8956, mae: 2.1681, huber: 1.7758, swd: 4.0714, ept: 40.3946
    Best round's Test MSE: 15.8956, MAE: 2.1681, SWD: 4.0712
    Best round's Validation MSE: 9.9578, MAE: 1.8011, SWD: 2.1226
    Best round's Test verification MSE : 15.8956, MAE: 2.1681, SWD: 4.0714
    Time taken: 16.64 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 16.5566, mae: 2.3334, huber: 1.9237, swd: 6.6873, ept: 43.2307
    Epoch [1/50], Val Losses: mse: 10.0457, mae: 1.7972, huber: 1.4171, swd: 1.6254, ept: 58.8926
    Epoch [1/50], Test Losses: mse: 16.6372, mae: 2.1999, huber: 1.8053, swd: 3.8306, ept: 38.8756
      Epoch 1 composite train-obj: 1.923716
            Val objective improved inf → 1.4171, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.7396, mae: 1.6670, huber: 1.2800, swd: 1.7485, ept: 63.3633
    Epoch [2/50], Val Losses: mse: 10.6331, mae: 1.8696, huber: 1.4931, swd: 1.7640, ept: 57.1716
    Epoch [2/50], Test Losses: mse: 10.9801, mae: 1.9190, huber: 1.5293, swd: 2.8926, ept: 41.4176
      Epoch 2 composite train-obj: 1.280039
            No improvement (1.4931), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.7503, mae: 1.5878, huber: 1.2050, swd: 1.4184, ept: 66.6482
    Epoch [3/50], Val Losses: mse: 11.5250, mae: 1.9337, huber: 1.5554, swd: 1.6855, ept: 53.6666
    Epoch [3/50], Test Losses: mse: 10.2839, mae: 1.8815, huber: 1.4925, swd: 2.3847, ept: 41.2837
      Epoch 3 composite train-obj: 1.204976
            No improvement (1.5554), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.4374, mae: 1.5478, huber: 1.1678, swd: 1.2934, ept: 68.5235
    Epoch [4/50], Val Losses: mse: 11.9462, mae: 1.9970, huber: 1.6177, swd: 2.9993, ept: 59.4177
    Epoch [4/50], Test Losses: mse: 10.3126, mae: 1.9319, huber: 1.5398, swd: 3.4103, ept: 43.6122
      Epoch 4 composite train-obj: 1.167778
            No improvement (1.6177), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.1851, mae: 1.5126, huber: 1.1353, swd: 1.1961, ept: 70.4332
    Epoch [5/50], Val Losses: mse: 12.2776, mae: 1.9936, huber: 1.6157, swd: 2.4691, ept: 55.1202
    Epoch [5/50], Test Losses: mse: 9.7573, mae: 1.8505, huber: 1.4632, swd: 2.4739, ept: 44.8789
      Epoch 5 composite train-obj: 1.135296
            No improvement (1.6157), counter 4/5
    Epoch [6/50], Train Losses: mse: 5.9392, mae: 1.4736, huber: 1.0992, swd: 1.0686, ept: 72.1287
    Epoch [6/50], Val Losses: mse: 12.5709, mae: 2.0336, huber: 1.6526, swd: 2.9085, ept: 56.6621
    Epoch [6/50], Test Losses: mse: 9.8508, mae: 1.8977, huber: 1.5043, swd: 2.8258, ept: 44.0923
      Epoch 6 composite train-obj: 1.099184
    Epoch [6/50], Test Losses: mse: 16.6379, mae: 2.2000, huber: 1.8054, swd: 3.8310, ept: 38.8499
    Best round's Test MSE: 16.6372, MAE: 2.1999, SWD: 3.8306
    Best round's Validation MSE: 10.0457, MAE: 1.7972, SWD: 1.6254
    Best round's Test verification MSE : 16.6379, MAE: 2.2000, SWD: 3.8310
    Time taken: 13.69 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq720_pred196_20250510_1547)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 16.2655 ± 0.3028
      mae: 2.1820 ± 0.0133
      huber: 1.7884 ± 0.0125
      swd: 4.0189 ± 0.1374
      ept: 39.8423 ± 0.6890
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 10.0155 ± 0.0409
      mae: 1.8004 ± 0.0024
      huber: 1.4210 ± 0.0028
      swd: 1.8869 ± 0.2038
      ept: 60.9147 ± 2.6940
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 49.26 seconds
    
    Experiment complete: ACL_etth1_seq720_pred196_20250510_1547
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([7.0675, 2.0423, 6.8268, 1.8092, 1.1645, 0.5995, 8.5667],
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 88
    Validation Batches: 6
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.4668, mae: 2.4070, huber: 1.9946, swd: 7.4849, ept: 54.3405
    Epoch [1/50], Val Losses: mse: 10.5172, mae: 1.8358, huber: 1.4526, swd: 2.1336, ept: 77.2107
    Epoch [1/50], Test Losses: mse: 18.2402, mae: 2.3391, huber: 1.9386, swd: 4.1791, ept: 55.2337
      Epoch 1 composite train-obj: 1.994589
            Val objective improved inf → 1.4526, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.4423, mae: 1.7567, huber: 1.3638, swd: 1.9380, ept: 80.5541
    Epoch [2/50], Val Losses: mse: 11.3285, mae: 1.9151, huber: 1.5372, swd: 1.5705, ept: 72.1292
    Epoch [2/50], Test Losses: mse: 11.4509, mae: 1.9866, huber: 1.5940, swd: 2.6052, ept: 57.3447
      Epoch 2 composite train-obj: 1.363785
            No improvement (1.5372), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.1332, mae: 1.6607, huber: 1.2714, swd: 1.5725, ept: 85.3087
    Epoch [3/50], Val Losses: mse: 12.7309, mae: 2.0763, huber: 1.6855, swd: 1.3363, ept: 55.3825
    Epoch [3/50], Test Losses: mse: 11.3195, mae: 2.0487, huber: 1.6500, swd: 2.4015, ept: 45.9612
      Epoch 3 composite train-obj: 1.271399
            No improvement (1.6855), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.9394, mae: 1.6383, huber: 1.2508, swd: 1.5103, ept: 86.5369
    Epoch [4/50], Val Losses: mse: 12.7370, mae: 2.0657, huber: 1.6698, swd: 1.6297, ept: 46.9439
    Epoch [4/50], Test Losses: mse: 10.6084, mae: 1.9696, huber: 1.5716, swd: 2.0533, ept: 50.3801
      Epoch 4 composite train-obj: 1.250828
            No improvement (1.6698), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.7511, mae: 1.6123, huber: 1.2266, swd: 1.4063, ept: 87.5723
    Epoch [5/50], Val Losses: mse: 14.2704, mae: 2.2523, huber: 1.8433, swd: 1.7846, ept: 41.5885
    Epoch [5/50], Test Losses: mse: 11.8400, mae: 2.1268, huber: 1.7219, swd: 2.1268, ept: 38.7582
      Epoch 5 composite train-obj: 1.226582
            No improvement (1.8433), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.7251, mae: 1.6038, huber: 1.2197, swd: 1.4189, ept: 88.6303
    Epoch [6/50], Val Losses: mse: 13.7033, mae: 2.0791, huber: 1.7005, swd: 2.5571, ept: 64.3934
    Epoch [6/50], Test Losses: mse: 10.1378, mae: 1.9478, huber: 1.5520, swd: 2.7584, ept: 59.2358
      Epoch 6 composite train-obj: 1.219746
    Epoch [6/50], Test Losses: mse: 18.2402, mae: 2.3391, huber: 1.9386, swd: 4.1786, ept: 55.2367
    Best round's Test MSE: 18.2402, MAE: 2.3391, SWD: 4.1791
    Best round's Validation MSE: 10.5172, MAE: 1.8358, SWD: 2.1336
    Best round's Test verification MSE : 18.2402, MAE: 2.3391, SWD: 4.1786
    Time taken: 13.90 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.6109, mae: 2.4358, huber: 2.0226, swd: 7.4893, ept: 51.2899
    Epoch [1/50], Val Losses: mse: 10.6665, mae: 1.8389, huber: 1.4556, swd: 2.1721, ept: 75.7190
    Epoch [1/50], Test Losses: mse: 18.8359, mae: 2.3725, huber: 1.9725, swd: 3.6914, ept: 54.9743
      Epoch 1 composite train-obj: 2.022647
            Val objective improved inf → 1.4556, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.4896, mae: 1.7601, huber: 1.3667, swd: 1.8873, ept: 79.9103
    Epoch [2/50], Val Losses: mse: 11.7592, mae: 1.9715, huber: 1.5903, swd: 1.4270, ept: 61.9797
    Epoch [2/50], Test Losses: mse: 11.8276, mae: 2.0203, huber: 1.6274, swd: 2.4268, ept: 53.2098
      Epoch 2 composite train-obj: 1.366690
            No improvement (1.5903), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.3479, mae: 1.6850, huber: 1.2949, swd: 1.6988, ept: 82.8624
    Epoch [3/50], Val Losses: mse: 12.3447, mae: 1.9780, huber: 1.5993, swd: 1.6835, ept: 65.6509
    Epoch [3/50], Test Losses: mse: 10.4826, mae: 1.9361, huber: 1.5443, swd: 2.2735, ept: 57.4117
      Epoch 3 composite train-obj: 1.294857
            No improvement (1.5993), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.0233, mae: 1.6494, huber: 1.2615, swd: 1.5539, ept: 84.8612
    Epoch [4/50], Val Losses: mse: 13.0032, mae: 2.1223, huber: 1.7264, swd: 3.4092, ept: 67.1659
    Epoch [4/50], Test Losses: mse: 10.8007, mae: 2.0917, huber: 1.6746, swd: 3.8316, ept: 51.8317
      Epoch 4 composite train-obj: 1.261476
            No improvement (1.7264), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.7776, mae: 1.6190, huber: 1.2333, swd: 1.4445, ept: 86.9700
    Epoch [5/50], Val Losses: mse: 13.0219, mae: 2.0186, huber: 1.6400, swd: 1.8121, ept: 62.8222
    Epoch [5/50], Test Losses: mse: 10.1065, mae: 1.9241, huber: 1.5303, swd: 2.1304, ept: 55.9872
      Epoch 5 composite train-obj: 1.233315
            No improvement (1.6400), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.5363, mae: 1.5795, huber: 1.1973, swd: 1.2828, ept: 89.1944
    Epoch [6/50], Val Losses: mse: 12.8018, mae: 2.0204, huber: 1.6410, swd: 2.6100, ept: 67.6403
    Epoch [6/50], Test Losses: mse: 10.1709, mae: 1.9465, huber: 1.5531, swd: 2.7789, ept: 58.9676
      Epoch 6 composite train-obj: 1.197258
    Epoch [6/50], Test Losses: mse: 18.8351, mae: 2.3724, huber: 1.9724, swd: 3.6910, ept: 55.0291
    Best round's Test MSE: 18.8359, MAE: 2.3725, SWD: 3.6914
    Best round's Validation MSE: 10.6665, MAE: 1.8389, SWD: 2.1721
    Best round's Test verification MSE : 18.8351, MAE: 2.3724, SWD: 3.6910
    Time taken: 13.89 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 16.6933, mae: 2.3666, huber: 1.9550, swd: 6.8489, ept: 54.5354
    Epoch [1/50], Val Losses: mse: 10.2559, mae: 1.8196, huber: 1.4390, swd: 1.5317, ept: 77.6530
    Epoch [1/50], Test Losses: mse: 16.4240, mae: 2.2375, huber: 1.8406, swd: 3.9448, ept: 56.1271
      Epoch 1 composite train-obj: 1.955016
            Val objective improved inf → 1.4390, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.0340, mae: 1.7308, huber: 1.3390, swd: 1.8617, ept: 82.4713
    Epoch [2/50], Val Losses: mse: 11.6250, mae: 1.9521, huber: 1.5716, swd: 1.3149, ept: 64.8977
    Epoch [2/50], Test Losses: mse: 11.3780, mae: 2.0188, huber: 1.6228, swd: 2.7329, ept: 52.7055
      Epoch 2 composite train-obj: 1.339003
            No improvement (1.5716), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.1335, mae: 1.6643, huber: 1.2759, swd: 1.5384, ept: 85.1410
    Epoch [3/50], Val Losses: mse: 11.9407, mae: 1.9631, huber: 1.5845, swd: 2.0839, ept: 69.7378
    Epoch [3/50], Test Losses: mse: 10.5717, mae: 1.9795, huber: 1.5846, swd: 3.1114, ept: 58.3225
      Epoch 3 composite train-obj: 1.275854
            No improvement (1.5845), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.8379, mae: 1.6264, huber: 1.2399, swd: 1.3861, ept: 87.7718
    Epoch [4/50], Val Losses: mse: 12.4846, mae: 2.0125, huber: 1.6288, swd: 2.2631, ept: 71.9451
    Epoch [4/50], Test Losses: mse: 10.3859, mae: 1.9938, huber: 1.5926, swd: 3.1521, ept: 56.2017
      Epoch 4 composite train-obj: 1.239940
            No improvement (1.6288), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.6306, mae: 1.5949, huber: 1.2109, swd: 1.2821, ept: 89.3708
    Epoch [5/50], Val Losses: mse: 14.1688, mae: 2.3103, huber: 1.8973, swd: 4.2663, ept: 62.8913
    Epoch [5/50], Test Losses: mse: 11.6296, mae: 2.2593, huber: 1.8305, swd: 5.1428, ept: 47.1794
      Epoch 5 composite train-obj: 1.210887
            No improvement (1.8973), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.7275, mae: 1.6068, huber: 1.2228, swd: 1.4374, ept: 88.5891
    Epoch [6/50], Val Losses: mse: 12.9482, mae: 2.0141, huber: 1.6362, swd: 2.1953, ept: 71.9671
    Epoch [6/50], Test Losses: mse: 9.9845, mae: 1.9316, huber: 1.5384, swd: 2.5786, ept: 59.9550
      Epoch 6 composite train-obj: 1.222831
    Epoch [6/50], Test Losses: mse: 16.4240, mae: 2.2375, huber: 1.8406, swd: 3.9445, ept: 56.1375
    Best round's Test MSE: 16.4240, MAE: 2.2375, SWD: 3.9448
    Best round's Validation MSE: 10.2559, MAE: 1.8196, SWD: 1.5317
    Best round's Test verification MSE : 16.4240, MAE: 2.2375, SWD: 3.9445
    Time taken: 15.22 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq720_pred336_20250510_1548)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 17.8334 ± 1.0258
      mae: 2.3164 ± 0.0574
      huber: 1.9173 ± 0.0559
      swd: 3.9384 ± 0.1992
      ept: 55.4450 ± 0.4938
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 10.4798 ± 0.1697
      mae: 1.8314 ± 0.0084
      huber: 1.4491 ± 0.0072
      swd: 1.9458 ± 0.2933
      ept: 76.8609 ± 0.8274
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 43.12 seconds
    
    Experiment complete: ACL_etth1_seq720_pred336_20250510_1548
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth1: tensor([7.0675, 2.0423, 6.8268, 1.8092, 1.1645, 0.5995, 8.5667],
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 85
    Validation Batches: 3
    Test Batches: 16
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.0448, mae: 2.4375, huber: 2.0207, swd: 6.8412, ept: 67.5002
    Epoch [1/50], Val Losses: mse: 11.2459, mae: 1.9300, huber: 1.5464, swd: 2.4511, ept: 68.8065
    Epoch [1/50], Test Losses: mse: 18.5049, mae: 2.4417, huber: 2.0325, swd: 3.9690, ept: 76.8100
      Epoch 1 composite train-obj: 2.020667
            Val objective improved inf → 1.5464, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.1601, mae: 1.8709, huber: 1.4691, swd: 2.1204, ept: 97.6913
    Epoch [2/50], Val Losses: mse: 12.6441, mae: 2.1335, huber: 1.7452, swd: 4.0008, ept: 116.5338
    Epoch [2/50], Test Losses: mse: 12.6842, mae: 2.2259, huber: 1.8136, swd: 4.8030, ept: 89.0111
      Epoch 2 composite train-obj: 1.469062
            No improvement (1.7452), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.8336, mae: 1.7716, huber: 1.3727, swd: 1.7622, ept: 103.2141
    Epoch [3/50], Val Losses: mse: 13.8675, mae: 2.1740, huber: 1.7913, swd: 2.4885, ept: 41.6309
    Epoch [3/50], Test Losses: mse: 11.8553, mae: 2.1632, huber: 1.7573, swd: 2.6587, ept: 67.1836
      Epoch 3 composite train-obj: 1.372699
            No improvement (1.7913), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.5870, mae: 1.7417, huber: 1.3446, swd: 1.6494, ept: 104.3969
    Epoch [4/50], Val Losses: mse: 15.4560, mae: 2.3287, huber: 1.9325, swd: 3.8364, ept: 34.4655
    Epoch [4/50], Test Losses: mse: 10.7033, mae: 2.0364, huber: 1.6339, swd: 2.3980, ept: 90.1176
      Epoch 4 composite train-obj: 1.344582
            No improvement (1.9325), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.2089, mae: 1.6936, huber: 1.2988, swd: 1.3915, ept: 109.4696
    Epoch [5/50], Val Losses: mse: 16.1030, mae: 2.3912, huber: 1.9976, swd: 4.2300, ept: 44.8565
    Epoch [5/50], Test Losses: mse: 11.8096, mae: 2.2278, huber: 1.8125, swd: 4.0181, ept: 73.2916
      Epoch 5 composite train-obj: 1.298820
            No improvement (1.9976), counter 4/5
    Epoch [6/50], Train Losses: mse: 7.0791, mae: 1.6694, huber: 1.2762, swd: 1.3565, ept: 111.8332
    Epoch [6/50], Val Losses: mse: 18.0085, mae: 2.5386, huber: 2.1481, swd: 5.2595, ept: 40.7877
    Epoch [6/50], Test Losses: mse: 11.9287, mae: 2.2135, huber: 1.8021, swd: 3.6687, ept: 69.5451
      Epoch 6 composite train-obj: 1.276237
    Epoch [6/50], Test Losses: mse: 18.5038, mae: 2.4416, huber: 2.0324, swd: 3.9692, ept: 76.6996
    Best round's Test MSE: 18.5049, MAE: 2.4417, SWD: 3.9690
    Best round's Validation MSE: 11.2459, MAE: 1.9300, SWD: 2.4511
    Best round's Test verification MSE : 18.5038, MAE: 2.4416, SWD: 3.9692
    Time taken: 13.76 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.8854, mae: 2.4968, huber: 2.0791, swd: 7.0675, ept: 66.0813
    Epoch [1/50], Val Losses: mse: 11.7761, mae: 1.9402, huber: 1.5532, swd: 2.0599, ept: 59.8766
    Epoch [1/50], Test Losses: mse: 20.9356, mae: 2.5778, huber: 2.1650, swd: 4.0382, ept: 71.9667
      Epoch 1 composite train-obj: 2.079127
            Val objective improved inf → 1.5532, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.3584, mae: 1.8817, huber: 1.4800, swd: 2.0740, ept: 98.6702
    Epoch [2/50], Val Losses: mse: 12.4530, mae: 2.1182, huber: 1.7303, swd: 3.4056, ept: 77.0388
    Epoch [2/50], Test Losses: mse: 12.6175, mae: 2.2394, huber: 1.8248, swd: 4.6537, ept: 86.2563
      Epoch 2 composite train-obj: 1.480016
            No improvement (1.7303), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.7317, mae: 1.7591, huber: 1.3613, swd: 1.5987, ept: 106.3944
    Epoch [3/50], Val Losses: mse: 14.7101, mae: 2.3564, huber: 1.9478, swd: 3.8127, ept: 45.6985
    Epoch [3/50], Test Losses: mse: 11.7448, mae: 2.2377, huber: 1.8114, swd: 4.1055, ept: 80.4950
      Epoch 3 composite train-obj: 1.361311
            No improvement (1.9478), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.3239, mae: 1.7130, huber: 1.3170, swd: 1.3991, ept: 109.0643
    Epoch [4/50], Val Losses: mse: 15.2887, mae: 2.4146, huber: 2.0135, swd: 4.8837, ept: 66.6990
    Epoch [4/50], Test Losses: mse: 11.9137, mae: 2.2661, huber: 1.8439, swd: 4.9456, ept: 89.5278
      Epoch 4 composite train-obj: 1.317006
            No improvement (2.0135), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.2536, mae: 1.6979, huber: 1.3036, swd: 1.4120, ept: 109.0199
    Epoch [5/50], Val Losses: mse: 17.3402, mae: 2.5166, huber: 2.1270, swd: 4.8597, ept: 42.7343
    Epoch [5/50], Test Losses: mse: 11.2834, mae: 2.1511, huber: 1.7448, swd: 3.5281, ept: 88.9395
      Epoch 5 composite train-obj: 1.303599
            No improvement (2.1270), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.9109, mae: 1.6466, huber: 1.2552, swd: 1.2084, ept: 114.1820
    Epoch [6/50], Val Losses: mse: 18.5432, mae: 2.5528, huber: 2.1652, swd: 5.1052, ept: 41.4938
    Epoch [6/50], Test Losses: mse: 11.4031, mae: 2.1161, huber: 1.7085, swd: 2.3745, ept: 72.9094
      Epoch 6 composite train-obj: 1.255225
    Epoch [6/50], Test Losses: mse: 20.9374, mae: 2.5780, huber: 2.1652, swd: 4.0383, ept: 72.0197
    Best round's Test MSE: 20.9356, MAE: 2.5778, SWD: 4.0382
    Best round's Validation MSE: 11.7761, MAE: 1.9402, SWD: 2.0599
    Best round's Test verification MSE : 20.9374, MAE: 2.5780, SWD: 4.0383
    Time taken: 13.22 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.7388, mae: 2.4772, huber: 2.0598, swd: 7.4506, ept: 68.1274
    Epoch [1/50], Val Losses: mse: 11.5127, mae: 1.9384, huber: 1.5527, swd: 2.9375, ept: 118.0317
    Epoch [1/50], Test Losses: mse: 21.0155, mae: 2.5628, huber: 2.1530, swd: 4.6487, ept: 84.6127
      Epoch 1 composite train-obj: 2.059824
            Val objective improved inf → 1.5527, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.3742, mae: 1.8773, huber: 1.4755, swd: 2.1625, ept: 99.4147
    Epoch [2/50], Val Losses: mse: 12.9633, mae: 2.1677, huber: 1.7817, swd: 2.3846, ept: 41.5527
    Epoch [2/50], Test Losses: mse: 13.7685, mae: 2.3126, huber: 1.9046, swd: 3.5451, ept: 52.5869
      Epoch 2 composite train-obj: 1.475505
            No improvement (1.7817), counter 1/5
    Epoch [3/50], Train Losses: mse: 8.0009, mae: 1.7874, huber: 1.3887, swd: 1.9222, ept: 102.4532
    Epoch [3/50], Val Losses: mse: 14.0138, mae: 2.2220, huber: 1.8348, swd: 4.1321, ept: 63.7232
    Epoch [3/50], Test Losses: mse: 11.6237, mae: 2.1491, huber: 1.7407, swd: 4.1022, ept: 89.1547
      Epoch 3 composite train-obj: 1.388673
            No improvement (1.8348), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.4580, mae: 1.7260, huber: 1.3300, swd: 1.6191, ept: 108.6746
    Epoch [4/50], Val Losses: mse: 14.8626, mae: 2.2398, huber: 1.8549, swd: 2.9820, ept: 43.9651
    Epoch [4/50], Test Losses: mse: 10.8972, mae: 2.0603, huber: 1.6563, swd: 2.3262, ept: 81.1808
      Epoch 4 composite train-obj: 1.329982
            No improvement (1.8549), counter 3/5
    Epoch [5/50], Train Losses: mse: 7.2912, mae: 1.7029, huber: 1.3081, swd: 1.5451, ept: 110.1605
    Epoch [5/50], Val Losses: mse: 15.7822, mae: 2.3585, huber: 1.9688, swd: 4.3859, ept: 51.9806
    Epoch [5/50], Test Losses: mse: 11.1209, mae: 2.1225, huber: 1.7166, swd: 3.8171, ept: 97.1954
      Epoch 5 composite train-obj: 1.308058
            No improvement (1.9688), counter 4/5
    Epoch [6/50], Train Losses: mse: 6.9676, mae: 1.6515, huber: 1.2599, swd: 1.3520, ept: 114.5618
    Epoch [6/50], Val Losses: mse: 17.1518, mae: 2.5416, huber: 2.1331, swd: 4.8726, ept: 46.5616
    Epoch [6/50], Test Losses: mse: 11.4470, mae: 2.2008, huber: 1.7841, swd: 3.8265, ept: 85.4921
      Epoch 6 composite train-obj: 1.259926
    Epoch [6/50], Test Losses: mse: 21.0149, mae: 2.5628, huber: 2.1530, swd: 4.6488, ept: 84.6572
    Best round's Test MSE: 21.0155, MAE: 2.5628, SWD: 4.6487
    Best round's Validation MSE: 11.5127, MAE: 1.9384, SWD: 2.9375
    Best round's Test verification MSE : 21.0149, MAE: 2.5628, SWD: 4.6488
    Time taken: 12.94 seconds
    
    ==================================================
    Experiment Summary (ACL_etth1_seq720_pred720_20250510_1551)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 20.1520 ± 1.1651
      mae: 2.5275 ± 0.0609
      huber: 2.1168 ± 0.0598
      swd: 4.2186 ± 0.3054
      ept: 77.7965 ± 5.2096
      count: 3.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 11.5116 ± 0.2165
      mae: 1.9362 ± 0.0044
      huber: 1.5508 ± 0.0031
      swd: 2.4828 ± 0.3590
      ept: 82.2383 ± 25.5710
      count: 3.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 40.04 seconds
    
    Experiment complete: ACL_etth1_seq720_pred720_20250510_1551
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 8
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.7340, mae: 1.5581, huber: 1.1816, swd: 1.5080, target_std: 6.1755
    Epoch [1/50], Val Losses: mse: 7.7975, mae: 1.5350, huber: 1.1731, swd: 1.5668, target_std: 4.4805
    Epoch [1/50], Test Losses: mse: 9.9740, mae: 1.8322, huber: 1.4527, swd: 2.5394, target_std: 4.8844
      Epoch 1 composite train-obj: 1.181639
            Val objective improved inf → 1.1731, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.6088, mae: 1.2654, huber: 0.9047, swd: 0.8020, target_std: 6.1754
    Epoch [2/50], Val Losses: mse: 8.6695, mae: 1.6569, huber: 1.2895, swd: 1.6843, target_std: 4.4805
    Epoch [2/50], Test Losses: mse: 10.9482, mae: 1.9364, huber: 1.5567, swd: 2.9987, target_std: 4.8844
      Epoch 2 composite train-obj: 0.904700
            No improvement (1.2895), counter 1/5
    Epoch [3/50], Train Losses: mse: 3.5118, mae: 1.1044, huber: 0.7532, swd: 0.5315, target_std: 6.1756
    Epoch [3/50], Val Losses: mse: 9.0230, mae: 1.6889, huber: 1.3189, swd: 1.3280, target_std: 4.4805
    Epoch [3/50], Test Losses: mse: 11.4614, mae: 1.9522, huber: 1.5727, swd: 2.7124, target_std: 4.8844
      Epoch 3 composite train-obj: 0.753182
            No improvement (1.3189), counter 2/5
    Epoch [4/50], Train Losses: mse: 2.8546, mae: 0.9967, huber: 0.6531, swd: 0.3912, target_std: 6.1755
    Epoch [4/50], Val Losses: mse: 9.1401, mae: 1.6851, huber: 1.3159, swd: 1.3798, target_std: 4.4805
    Epoch [4/50], Test Losses: mse: 11.7255, mae: 1.9893, huber: 1.6075, swd: 3.0306, target_std: 4.8844
      Epoch 4 composite train-obj: 0.653073
            No improvement (1.3159), counter 3/5
    Epoch [5/50], Train Losses: mse: 2.4065, mae: 0.9207, huber: 0.5832, swd: 0.3094, target_std: 6.1757
    Epoch [5/50], Val Losses: mse: 9.5747, mae: 1.7095, huber: 1.3401, swd: 1.2694, target_std: 4.4805
    Epoch [5/50], Test Losses: mse: 11.7654, mae: 1.9721, huber: 1.5914, swd: 2.6399, target_std: 4.8844
      Epoch 5 composite train-obj: 0.583233
            No improvement (1.3401), counter 4/5
    Epoch [6/50], Train Losses: mse: 2.0417, mae: 0.8568, huber: 0.5249, swd: 0.2495, target_std: 6.1759
    Epoch [6/50], Val Losses: mse: 10.0215, mae: 1.7375, huber: 1.3667, swd: 1.3884, target_std: 4.4805
    Epoch [6/50], Test Losses: mse: 11.7734, mae: 1.9766, huber: 1.5955, swd: 2.7109, target_std: 4.8844
      Epoch 6 composite train-obj: 0.524885
    Epoch [6/50], Test Losses: mse: 9.9740, mae: 1.8322, huber: 1.4527, swd: 2.5394, target_std: 4.8844
    Best round's Test MSE: 9.9740, MAE: 1.8322, SWD: 2.5394
    Best round's Validation MSE: 7.7975, MAE: 1.5350
    Best round's Test verification MSE : 9.9740, MAE: 1.8322, SWD: 2.5394
    Time taken: 71.36 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.6531, mae: 1.6519, huber: 1.2710, swd: 1.3903, target_std: 6.1759
    Epoch [1/50], Val Losses: mse: 7.5327, mae: 1.5280, huber: 1.1673, swd: 1.5901, target_std: 4.4805
    Epoch [1/50], Test Losses: mse: 10.1214, mae: 1.8324, huber: 1.4522, swd: 2.1477, target_std: 4.8844
      Epoch 1 composite train-obj: 1.271006
            Val objective improved inf → 1.1673, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.8770, mae: 1.2925, huber: 0.9310, swd: 0.8375, target_std: 6.1754
    Epoch [2/50], Val Losses: mse: 7.7967, mae: 1.5957, huber: 1.2257, swd: 1.6001, target_std: 4.4805
    Epoch [2/50], Test Losses: mse: 10.7168, mae: 1.8973, huber: 1.5146, swd: 2.4061, target_std: 4.8844
      Epoch 2 composite train-obj: 0.930989
            No improvement (1.2257), counter 1/5
    Epoch [3/50], Train Losses: mse: 3.8135, mae: 1.1237, huber: 0.7737, swd: 0.5527, target_std: 6.1755
    Epoch [3/50], Val Losses: mse: 8.2628, mae: 1.6344, huber: 1.2647, swd: 1.6076, target_std: 4.4805
    Epoch [3/50], Test Losses: mse: 10.7177, mae: 1.8959, huber: 1.5144, swd: 2.2998, target_std: 4.8844
      Epoch 3 composite train-obj: 0.773720
            No improvement (1.2647), counter 2/5
    Epoch [4/50], Train Losses: mse: 3.0660, mae: 1.0137, huber: 0.6717, swd: 0.3908, target_std: 6.1758
    Epoch [4/50], Val Losses: mse: 8.7498, mae: 1.6675, huber: 1.2963, swd: 1.5556, target_std: 4.4805
    Epoch [4/50], Test Losses: mse: 11.3647, mae: 1.9587, huber: 1.5750, swd: 2.6722, target_std: 4.8844
      Epoch 4 composite train-obj: 0.671661
            No improvement (1.2963), counter 3/5
    Epoch [5/50], Train Losses: mse: 2.4604, mae: 0.9249, huber: 0.5886, swd: 0.2892, target_std: 6.1758
    Epoch [5/50], Val Losses: mse: 8.9320, mae: 1.6717, huber: 1.3002, swd: 1.6373, target_std: 4.4805
    Epoch [5/50], Test Losses: mse: 11.9531, mae: 1.9910, huber: 1.6060, swd: 2.8502, target_std: 4.8844
      Epoch 5 composite train-obj: 0.588552
            No improvement (1.3002), counter 4/5
    Epoch [6/50], Train Losses: mse: 2.0334, mae: 0.8533, huber: 0.5225, swd: 0.2275, target_std: 6.1757
    Epoch [6/50], Val Losses: mse: 9.1726, mae: 1.6895, huber: 1.3175, swd: 1.5551, target_std: 4.4805
    Epoch [6/50], Test Losses: mse: 12.3732, mae: 2.0227, huber: 1.6371, swd: 2.8771, target_std: 4.8844
      Epoch 6 composite train-obj: 0.522527
    Epoch [6/50], Test Losses: mse: 10.1214, mae: 1.8324, huber: 1.4522, swd: 2.1477, target_std: 4.8844
    Best round's Test MSE: 10.1214, MAE: 1.8324, SWD: 2.1477
    Best round's Validation MSE: 7.5327, MAE: 1.5280
    Best round's Test verification MSE : 10.1214, MAE: 1.8324, SWD: 2.1477
    Time taken: 73.68 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.7625, mae: 1.5615, huber: 1.1854, swd: 1.5554, target_std: 6.1754
    Epoch [1/50], Val Losses: mse: 7.4813, mae: 1.4875, huber: 1.1279, swd: 1.1965, target_std: 4.4805
    Epoch [1/50], Test Losses: mse: 9.9617, mae: 1.8211, huber: 1.4419, swd: 2.1584, target_std: 4.8844
      Epoch 1 composite train-obj: 1.185356
            Val objective improved inf → 1.1279, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.1105, mae: 1.3266, huber: 0.9638, swd: 0.9363, target_std: 6.1757
    Epoch [2/50], Val Losses: mse: 7.6536, mae: 1.5282, huber: 1.1673, swd: 1.3849, target_std: 4.4805
    Epoch [2/50], Test Losses: mse: 10.2339, mae: 1.8258, huber: 1.4477, swd: 2.2033, target_std: 4.8844
      Epoch 2 composite train-obj: 0.963808
            No improvement (1.1673), counter 1/5
    Epoch [3/50], Train Losses: mse: 4.1114, mae: 1.1747, huber: 0.8205, swd: 0.6349, target_std: 6.1755
    Epoch [3/50], Val Losses: mse: 7.9110, mae: 1.5875, huber: 1.2209, swd: 1.3853, target_std: 4.4805
    Epoch [3/50], Test Losses: mse: 10.8029, mae: 1.8926, huber: 1.5119, swd: 2.5956, target_std: 4.8844
      Epoch 3 composite train-obj: 0.820535
            No improvement (1.2209), counter 2/5
    Epoch [4/50], Train Losses: mse: 3.2812, mae: 1.0519, huber: 0.7056, swd: 0.4489, target_std: 6.1759
    Epoch [4/50], Val Losses: mse: 9.3947, mae: 1.6779, huber: 1.3106, swd: 1.3711, target_std: 4.4805
    Epoch [4/50], Test Losses: mse: 10.8589, mae: 1.9030, huber: 1.5233, swd: 2.4910, target_std: 4.8844
      Epoch 4 composite train-obj: 0.705565
            No improvement (1.3106), counter 3/5
    Epoch [5/50], Train Losses: mse: 2.6670, mae: 0.9552, huber: 0.6160, swd: 0.3325, target_std: 6.1758
    Epoch [5/50], Val Losses: mse: 10.3732, mae: 1.7344, huber: 1.3637, swd: 1.2516, target_std: 4.4805
    Epoch [5/50], Test Losses: mse: 11.4840, mae: 1.9389, huber: 1.5575, swd: 2.3579, target_std: 4.8844
      Epoch 5 composite train-obj: 0.615990
            No improvement (1.3637), counter 4/5
    Epoch [6/50], Train Losses: mse: 2.2218, mae: 0.8798, huber: 0.5470, swd: 0.2539, target_std: 6.1756
    Epoch [6/50], Val Losses: mse: 10.7905, mae: 1.7471, huber: 1.3764, swd: 1.1086, target_std: 4.4805
    Epoch [6/50], Test Losses: mse: 11.7002, mae: 1.9497, huber: 1.5689, swd: 2.3155, target_std: 4.8844
      Epoch 6 composite train-obj: 0.547018
    Epoch [6/50], Test Losses: mse: 9.9617, mae: 1.8211, huber: 1.4419, swd: 2.1584, target_std: 4.8844
    Best round's Test MSE: 9.9617, MAE: 1.8211, SWD: 2.1584
    Best round's Validation MSE: 7.4813, MAE: 1.4875
    Best round's Test verification MSE : 9.9617, MAE: 1.8211, SWD: 2.1584
    Time taken: 72.04 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq720_pred96_20250503_2023)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.0190 ± 0.0725
      mae: 1.8286 ± 0.0053
      huber: 1.4489 ± 0.0050
      swd: 2.2818 ± 0.1822
      target_std: 4.8844 ± 0.0000
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.6038 ± 0.1385
      mae: 1.5168 ± 0.0209
      huber: 1.1561 ± 0.0201
      swd: 1.4511 ± 0.1803
      target_std: 4.4805 ± 0.0000
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 217.23 seconds
    
    Experiment complete: TimeMixer_etth1_seq720_pred96_20250503_2023
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 7
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.0361, mae: 1.7304, huber: 1.3438, swd: 1.4576, target_std: 6.1605
    Epoch [1/50], Val Losses: mse: 9.1588, mae: 1.6898, huber: 1.3195, swd: 1.5487, target_std: 4.4590
    Epoch [1/50], Test Losses: mse: 10.6041, mae: 1.9238, huber: 1.5357, swd: 2.2528, target_std: 4.8938
      Epoch 1 composite train-obj: 1.343791
            Val objective improved inf → 1.3195, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.5599, mae: 1.4147, huber: 1.0433, swd: 0.9034, target_std: 6.1610
    Epoch [2/50], Val Losses: mse: 10.2599, mae: 1.8056, huber: 1.4315, swd: 1.8434, target_std: 4.4590
    Epoch [2/50], Test Losses: mse: 11.1052, mae: 2.0044, huber: 1.6150, swd: 2.8328, target_std: 4.8938
      Epoch 2 composite train-obj: 1.043322
            No improvement (1.4315), counter 1/5
    Epoch [3/50], Train Losses: mse: 4.4457, mae: 1.2545, huber: 0.8912, swd: 0.6311, target_std: 6.1574
    Epoch [3/50], Val Losses: mse: 10.8567, mae: 1.8782, huber: 1.4981, swd: 1.5068, target_std: 4.4590
    Epoch [3/50], Test Losses: mse: 11.5539, mae: 2.0289, huber: 1.6394, swd: 2.5601, target_std: 4.8938
      Epoch 3 composite train-obj: 0.891187
            No improvement (1.4981), counter 2/5
    Epoch [4/50], Train Losses: mse: 3.7286, mae: 1.1423, huber: 0.7861, swd: 0.4622, target_std: 6.1606
    Epoch [4/50], Val Losses: mse: 12.1433, mae: 1.9519, huber: 1.5714, swd: 1.5306, target_std: 4.4590
    Epoch [4/50], Test Losses: mse: 12.1657, mae: 2.0613, huber: 1.6716, swd: 2.6304, target_std: 4.8938
      Epoch 4 composite train-obj: 0.786089
            No improvement (1.5714), counter 3/5
    Epoch [5/50], Train Losses: mse: 3.2869, mae: 1.0661, huber: 0.7158, swd: 0.3649, target_std: 6.1647
    Epoch [5/50], Val Losses: mse: 11.5008, mae: 1.9025, huber: 1.5226, swd: 1.3413, target_std: 4.4590
    Epoch [5/50], Test Losses: mse: 12.0530, mae: 2.0594, huber: 1.6684, swd: 2.7071, target_std: 4.8938
      Epoch 5 composite train-obj: 0.715774
            No improvement (1.5226), counter 4/5
    Epoch [6/50], Train Losses: mse: 2.9230, mae: 1.0044, huber: 0.6592, swd: 0.3008, target_std: 6.1622
    Epoch [6/50], Val Losses: mse: 12.4780, mae: 1.9636, huber: 1.5819, swd: 1.5011, target_std: 4.4590
    Epoch [6/50], Test Losses: mse: 12.3764, mae: 2.0780, huber: 1.6866, swd: 2.6146, target_std: 4.8938
      Epoch 6 composite train-obj: 0.659232
    Epoch [6/50], Test Losses: mse: 10.6041, mae: 1.9238, huber: 1.5357, swd: 2.2528, target_std: 4.8938
    Best round's Test MSE: 10.6041, MAE: 1.9238, SWD: 2.2528
    Best round's Validation MSE: 9.1588, MAE: 1.6898
    Best round's Test verification MSE : 10.6041, MAE: 1.9238, SWD: 2.2528
    Time taken: 72.34 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.5337, mae: 1.7753, huber: 1.3863, swd: 1.3993, target_std: 6.1606
    Epoch [1/50], Val Losses: mse: 10.0318, mae: 1.7779, huber: 1.4048, swd: 1.4497, target_std: 4.4590
    Epoch [1/50], Test Losses: mse: 10.3972, mae: 1.9055, huber: 1.5194, swd: 1.8806, target_std: 4.8938
      Epoch 1 composite train-obj: 1.386342
            Val objective improved inf → 1.4048, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.1087, mae: 1.3621, huber: 0.9921, swd: 0.7255, target_std: 6.1640
    Epoch [2/50], Val Losses: mse: 10.8213, mae: 1.8355, huber: 1.4591, swd: 1.3379, target_std: 4.4590
    Epoch [2/50], Test Losses: mse: 11.7259, mae: 2.0460, huber: 1.6579, swd: 2.5134, target_std: 4.8938
      Epoch 2 composite train-obj: 0.992057
            No improvement (1.4591), counter 1/5
    Epoch [3/50], Train Losses: mse: 3.8386, mae: 1.1721, huber: 0.8127, swd: 0.4623, target_std: 6.1728
    Epoch [3/50], Val Losses: mse: 11.1693, mae: 1.8590, huber: 1.4837, swd: 1.7087, target_std: 4.4590
    Epoch [3/50], Test Losses: mse: 12.0867, mae: 2.0809, huber: 1.6909, swd: 2.9452, target_std: 4.8938
      Epoch 3 composite train-obj: 0.812747
            No improvement (1.4837), counter 2/5
    Epoch [4/50], Train Losses: mse: 3.2048, mae: 1.0619, huber: 0.7113, swd: 0.3449, target_std: 6.1643
    Epoch [4/50], Val Losses: mse: 12.0435, mae: 1.9002, huber: 1.5242, swd: 1.2315, target_std: 4.4590
    Epoch [4/50], Test Losses: mse: 12.5502, mae: 2.0868, huber: 1.6963, swd: 2.6165, target_std: 4.8938
      Epoch 4 composite train-obj: 0.711320
            No improvement (1.5242), counter 3/5
    Epoch [5/50], Train Losses: mse: 2.7379, mae: 0.9887, huber: 0.6436, swd: 0.2835, target_std: 6.1617
    Epoch [5/50], Val Losses: mse: 11.9997, mae: 1.9015, huber: 1.5242, swd: 1.2932, target_std: 4.4590
    Epoch [5/50], Test Losses: mse: 12.5591, mae: 2.0862, huber: 1.6960, swd: 2.4881, target_std: 4.8938
      Epoch 5 composite train-obj: 0.643625
            No improvement (1.5242), counter 4/5
    Epoch [6/50], Train Losses: mse: 2.3828, mae: 0.9321, huber: 0.5911, swd: 0.2375, target_std: 6.1578
    Epoch [6/50], Val Losses: mse: 11.8714, mae: 1.8907, huber: 1.5129, swd: 1.2202, target_std: 4.4590
    Epoch [6/50], Test Losses: mse: 13.0033, mae: 2.1295, huber: 1.7361, swd: 2.6037, target_std: 4.8938
      Epoch 6 composite train-obj: 0.591073
    Epoch [6/50], Test Losses: mse: 10.3972, mae: 1.9055, huber: 1.5194, swd: 1.8806, target_std: 4.8938
    Best round's Test MSE: 10.3972, MAE: 1.9055, SWD: 1.8806
    Best round's Validation MSE: 10.0318, MAE: 1.7779
    Best round's Test verification MSE : 10.3972, MAE: 1.9055, SWD: 1.8806
    Time taken: 71.93 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.0516, mae: 1.6214, huber: 1.2404, swd: 1.4489, target_std: 6.1650
    Epoch [1/50], Val Losses: mse: 9.8696, mae: 1.7286, huber: 1.3592, swd: 1.2115, target_std: 4.4590
    Epoch [1/50], Test Losses: mse: 10.2812, mae: 1.8841, huber: 1.4987, swd: 1.7082, target_std: 4.8938
      Epoch 1 composite train-obj: 1.240410
            Val objective improved inf → 1.3592, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.4362, mae: 1.3937, huber: 1.0244, swd: 0.8285, target_std: 6.1670
    Epoch [2/50], Val Losses: mse: 10.9375, mae: 1.8320, huber: 1.4595, swd: 1.3370, target_std: 4.4590
    Epoch [2/50], Test Losses: mse: 10.8892, mae: 1.9539, huber: 1.5690, swd: 1.8912, target_std: 4.8938
      Epoch 2 composite train-obj: 1.024411
            No improvement (1.4595), counter 1/5
    Epoch [3/50], Train Losses: mse: 4.2712, mae: 1.2248, huber: 0.8637, swd: 0.5381, target_std: 6.1686
    Epoch [3/50], Val Losses: mse: 11.1970, mae: 1.8897, huber: 1.5108, swd: 1.5863, target_std: 4.4590
    Epoch [3/50], Test Losses: mse: 11.8041, mae: 2.0722, huber: 1.6824, swd: 2.5977, target_std: 4.8938
      Epoch 3 composite train-obj: 0.863705
            No improvement (1.5108), counter 2/5
    Epoch [4/50], Train Losses: mse: 3.4897, mae: 1.1035, huber: 0.7496, swd: 0.3990, target_std: 6.1666
    Epoch [4/50], Val Losses: mse: 12.9165, mae: 1.9883, huber: 1.6068, swd: 1.4351, target_std: 4.4590
    Epoch [4/50], Test Losses: mse: 12.1156, mae: 2.0756, huber: 1.6866, swd: 2.3979, target_std: 4.8938
      Epoch 4 composite train-obj: 0.749600
            No improvement (1.6068), counter 3/5
    Epoch [5/50], Train Losses: mse: 2.9643, mae: 1.0151, huber: 0.6679, swd: 0.3090, target_std: 6.1632
    Epoch [5/50], Val Losses: mse: 14.1489, mae: 2.0552, huber: 1.6719, swd: 1.5343, target_std: 4.4590
    Epoch [5/50], Test Losses: mse: 13.0871, mae: 2.1444, huber: 1.7533, swd: 2.5071, target_std: 4.8938
      Epoch 5 composite train-obj: 0.667858
            No improvement (1.6719), counter 4/5
    Epoch [6/50], Train Losses: mse: 2.5896, mae: 0.9515, huber: 0.6096, swd: 0.2562, target_std: 6.1595
    Epoch [6/50], Val Losses: mse: 14.1577, mae: 2.0521, huber: 1.6685, swd: 1.4306, target_std: 4.4590
    Epoch [6/50], Test Losses: mse: 12.9148, mae: 2.1165, huber: 1.7259, swd: 2.3517, target_std: 4.8938
      Epoch 6 composite train-obj: 0.609595
    Epoch [6/50], Test Losses: mse: 10.2812, mae: 1.8841, huber: 1.4987, swd: 1.7082, target_std: 4.8938
    Best round's Test MSE: 10.2812, MAE: 1.8841, SWD: 1.7082
    Best round's Validation MSE: 9.8696, MAE: 1.7286
    Best round's Test verification MSE : 10.2812, MAE: 1.8841, SWD: 1.7082
    Time taken: 74.01 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq720_pred196_20250503_2027)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.4275 ± 0.1335
      mae: 1.9044 ± 0.0162
      huber: 1.5179 ± 0.0151
      swd: 1.9472 ± 0.2273
      target_std: 4.8938 ± 0.0000
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.6867 ± 0.3791
      mae: 1.7321 ± 0.0361
      huber: 1.3612 ± 0.0349
      swd: 1.4033 ± 0.1415
      target_std: 4.4590 ± 0.0000
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 218.40 seconds
    
    Experiment complete: TimeMixer_etth1_seq720_pred196_20250503_2027
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 88
    Validation Batches: 6
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.0919, mae: 1.7630, huber: 1.3723, swd: 1.4132, target_std: 6.1333
    Epoch [1/50], Val Losses: mse: 10.5122, mae: 1.7783, huber: 1.4079, swd: 1.3369, target_std: 4.4388
    Epoch [1/50], Test Losses: mse: 10.8540, mae: 2.0012, huber: 1.6077, swd: 2.0757, target_std: 4.9300
      Epoch 1 composite train-obj: 1.372317
            Val objective improved inf → 1.4079, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.8735, mae: 1.4695, huber: 1.0932, swd: 0.8426, target_std: 6.1476
    Epoch [2/50], Val Losses: mse: 9.8290, mae: 1.7600, huber: 1.3858, swd: 1.4792, target_std: 4.4388
    Epoch [2/50], Test Losses: mse: 11.9186, mae: 2.1060, huber: 1.7099, swd: 3.0201, target_std: 4.9300
      Epoch 2 composite train-obj: 1.093155
            Val objective improved 1.4079 → 1.3858, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7942, mae: 1.3128, huber: 0.9446, swd: 0.6332, target_std: 6.1350
    Epoch [3/50], Val Losses: mse: 12.6632, mae: 1.9641, huber: 1.5817, swd: 1.0514, target_std: 4.4388
    Epoch [3/50], Test Losses: mse: 12.0725, mae: 2.0724, huber: 1.6794, swd: 1.9215, target_std: 4.9300
      Epoch 3 composite train-obj: 0.944644
            No improvement (1.5817), counter 1/5
    Epoch [4/50], Train Losses: mse: 4.0707, mae: 1.1990, huber: 0.8375, swd: 0.4784, target_std: 6.1285
    Epoch [4/50], Val Losses: mse: 13.7500, mae: 1.9905, huber: 1.6084, swd: 1.0731, target_std: 4.4388
    Epoch [4/50], Test Losses: mse: 12.0702, mae: 2.0606, huber: 1.6672, swd: 1.8259, target_std: 4.9300
      Epoch 4 composite train-obj: 0.837507
            No improvement (1.6084), counter 2/5
    Epoch [5/50], Train Losses: mse: 3.5012, mae: 1.1008, huber: 0.7470, swd: 0.3439, target_std: 6.1442
    Epoch [5/50], Val Losses: mse: 12.5834, mae: 1.9364, huber: 1.5547, swd: 1.0573, target_std: 4.4388
    Epoch [5/50], Test Losses: mse: 12.5364, mae: 2.1073, huber: 1.7131, swd: 2.0322, target_std: 4.9300
      Epoch 5 composite train-obj: 0.746996
            No improvement (1.5547), counter 3/5
    Epoch [6/50], Train Losses: mse: 3.1758, mae: 1.0519, huber: 0.7015, swd: 0.3115, target_std: 6.1565
    Epoch [6/50], Val Losses: mse: 14.4920, mae: 2.0390, huber: 1.6550, swd: 1.0526, target_std: 4.4388
    Epoch [6/50], Test Losses: mse: 12.7034, mae: 2.1120, huber: 1.7163, swd: 1.8337, target_std: 4.9300
      Epoch 6 composite train-obj: 0.701467
            No improvement (1.6550), counter 4/5
    Epoch [7/50], Train Losses: mse: 2.8214, mae: 0.9932, huber: 0.6477, swd: 0.2589, target_std: 6.1526
    Epoch [7/50], Val Losses: mse: 14.4087, mae: 2.0457, huber: 1.6604, swd: 0.9372, target_std: 4.4388
    Epoch [7/50], Test Losses: mse: 13.3576, mae: 2.1571, huber: 1.7617, swd: 1.8090, target_std: 4.9300
      Epoch 7 composite train-obj: 0.647744
    Epoch [7/50], Test Losses: mse: 11.9186, mae: 2.1060, huber: 1.7099, swd: 3.0201, target_std: 4.9300
    Best round's Test MSE: 11.9186, MAE: 2.1060, SWD: 3.0201
    Best round's Validation MSE: 9.8290, MAE: 1.7600
    Best round's Test verification MSE : 11.9186, MAE: 2.1060, SWD: 3.0201
    Time taken: 85.15 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.5129, mae: 1.9110, huber: 1.5137, swd: 1.4995, target_std: 6.1482
    Epoch [1/50], Val Losses: mse: 9.3158, mae: 1.7013, huber: 1.3301, swd: 1.2532, target_std: 4.4388
    Epoch [1/50], Test Losses: mse: 10.8359, mae: 1.9820, huber: 1.5916, swd: 2.2326, target_std: 4.9300
      Epoch 1 composite train-obj: 1.513722
            Val objective improved inf → 1.3301, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.9058, mae: 1.4914, huber: 1.1126, swd: 0.8435, target_std: 6.1361
    Epoch [2/50], Val Losses: mse: 11.4364, mae: 1.8675, huber: 1.4936, swd: 1.1082, target_std: 4.4388
    Epoch [2/50], Test Losses: mse: 11.6989, mae: 2.0456, huber: 1.6578, swd: 1.9733, target_std: 4.9300
      Epoch 2 composite train-obj: 1.112603
            No improvement (1.4936), counter 1/5
    Epoch [3/50], Train Losses: mse: 4.6333, mae: 1.3047, huber: 0.9357, swd: 0.5604, target_std: 6.1418
    Epoch [3/50], Val Losses: mse: 11.5340, mae: 1.8886, huber: 1.5127, swd: 1.0849, target_std: 4.4388
    Epoch [3/50], Test Losses: mse: 12.5818, mae: 2.1248, huber: 1.7360, swd: 2.2060, target_std: 4.9300
      Epoch 3 composite train-obj: 0.935717
            No improvement (1.5127), counter 2/5
    Epoch [4/50], Train Losses: mse: 3.8973, mae: 1.1860, huber: 0.8247, swd: 0.4136, target_std: 6.1284
    Epoch [4/50], Val Losses: mse: 11.7141, mae: 1.8874, huber: 1.5100, swd: 0.9408, target_std: 4.4388
    Epoch [4/50], Test Losses: mse: 12.5685, mae: 2.1092, huber: 1.7195, swd: 2.0219, target_std: 4.9300
      Epoch 4 composite train-obj: 0.824653
            No improvement (1.5100), counter 3/5
    Epoch [5/50], Train Losses: mse: 3.3363, mae: 1.0897, huber: 0.7353, swd: 0.3184, target_std: 6.1370
    Epoch [5/50], Val Losses: mse: 12.9847, mae: 1.9597, huber: 1.5794, swd: 0.9014, target_std: 4.4388
    Epoch [5/50], Test Losses: mse: 13.1985, mae: 2.1438, huber: 1.7531, swd: 1.8723, target_std: 4.9300
      Epoch 5 composite train-obj: 0.735339
            No improvement (1.5794), counter 4/5
    Epoch [6/50], Train Losses: mse: 3.0141, mae: 1.0370, huber: 0.6868, swd: 0.2781, target_std: 6.1529
    Epoch [6/50], Val Losses: mse: 12.7631, mae: 1.9501, huber: 1.5678, swd: 1.2027, target_std: 4.4388
    Epoch [6/50], Test Losses: mse: 13.6672, mae: 2.2090, huber: 1.8141, swd: 2.4079, target_std: 4.9300
      Epoch 6 composite train-obj: 0.686835
    Epoch [6/50], Test Losses: mse: 10.8359, mae: 1.9820, huber: 1.5916, swd: 2.2326, target_std: 4.9300
    Best round's Test MSE: 10.8359, MAE: 1.9820, SWD: 2.2326
    Best round's Validation MSE: 9.3158, MAE: 1.7013
    Best round's Test verification MSE : 10.8359, MAE: 1.9820, SWD: 2.2326
    Time taken: 72.96 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.6754, mae: 1.7264, huber: 1.3379, swd: 1.5617, target_std: 6.1448
    Epoch [1/50], Val Losses: mse: 10.7716, mae: 1.8162, huber: 1.4444, swd: 1.3335, target_std: 4.4388
    Epoch [1/50], Test Losses: mse: 10.3851, mae: 1.9258, huber: 1.5376, swd: 1.6756, target_std: 4.9300
      Epoch 1 composite train-obj: 1.337863
            Val objective improved inf → 1.4444, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.0835, mae: 1.5062, huber: 1.1279, swd: 0.9341, target_std: 6.1455
    Epoch [2/50], Val Losses: mse: 12.3946, mae: 1.9067, huber: 1.5338, swd: 1.4530, target_std: 4.4388
    Epoch [2/50], Test Losses: mse: 11.0935, mae: 2.0062, huber: 1.6142, swd: 2.0063, target_std: 4.9300
      Epoch 2 composite train-obj: 1.127885
            No improvement (1.5338), counter 1/5
    Epoch [3/50], Train Losses: mse: 5.0487, mae: 1.3574, huber: 0.9858, swd: 0.6567, target_std: 6.1419
    Epoch [3/50], Val Losses: mse: 12.8015, mae: 1.9469, huber: 1.5687, swd: 1.1724, target_std: 4.4388
    Epoch [3/50], Test Losses: mse: 12.0766, mae: 2.0853, huber: 1.6930, swd: 1.8811, target_std: 4.9300
      Epoch 3 composite train-obj: 0.985758
            No improvement (1.5687), counter 2/5
    Epoch [4/50], Train Losses: mse: 4.2698, mae: 1.2382, huber: 0.8732, swd: 0.5104, target_std: 6.1431
    Epoch [4/50], Val Losses: mse: 14.5561, mae: 2.0583, huber: 1.6757, swd: 1.3098, target_std: 4.4388
    Epoch [4/50], Test Losses: mse: 13.0646, mae: 2.1535, huber: 1.7593, swd: 1.9900, target_std: 4.9300
      Epoch 4 composite train-obj: 0.873179
            No improvement (1.6757), counter 3/5
    Epoch [5/50], Train Losses: mse: 3.7130, mae: 1.1435, huber: 0.7847, swd: 0.3890, target_std: 6.1452
    Epoch [5/50], Val Losses: mse: 13.8955, mae: 2.0372, huber: 1.6517, swd: 1.2144, target_std: 4.4388
    Epoch [5/50], Test Losses: mse: 13.2241, mae: 2.1687, huber: 1.7726, swd: 2.0627, target_std: 4.9300
      Epoch 5 composite train-obj: 0.784718
            No improvement (1.6517), counter 4/5
    Epoch [6/50], Train Losses: mse: 3.3337, mae: 1.0797, huber: 0.7259, swd: 0.3306, target_std: 6.1348
    Epoch [6/50], Val Losses: mse: 13.6090, mae: 2.0126, huber: 1.6277, swd: 0.9837, target_std: 4.4388
    Epoch [6/50], Test Losses: mse: 13.5850, mae: 2.2009, huber: 1.8045, swd: 1.8852, target_std: 4.9300
      Epoch 6 composite train-obj: 0.725885
    Epoch [6/50], Test Losses: mse: 10.3851, mae: 1.9258, huber: 1.5376, swd: 1.6756, target_std: 4.9300
    Best round's Test MSE: 10.3851, MAE: 1.9258, SWD: 1.6756
    Best round's Validation MSE: 10.7716, MAE: 1.8162
    Best round's Test verification MSE : 10.3851, MAE: 1.9258, SWD: 1.6756
    Time taken: 73.39 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq720_pred336_20250503_2030)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.0465 ± 0.6435
      mae: 2.0046 ± 0.0753
      huber: 1.6130 ± 0.0720
      swd: 2.3094 ± 0.5515
      target_std: 4.9300 ± 0.0000
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.9721 ± 0.6029
      mae: 1.7592 ± 0.0469
      huber: 1.3868 ± 0.0467
      swd: 1.3553 ± 0.0936
      target_std: 4.4388 ± 0.0000
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 231.63 seconds
    
    Experiment complete: TimeMixer_etth1_seq720_pred336_20250503_2030
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 85
    Validation Batches: 3
    Test Batches: 16
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.2403, mae: 2.0227, huber: 1.6164, swd: 2.0474, target_std: 6.0882
    Epoch [1/50], Val Losses: mse: 10.0872, mae: 1.7476, huber: 1.3767, swd: 0.9057, target_std: 4.4502
    Epoch [1/50], Test Losses: mse: 11.7136, mae: 2.1635, huber: 1.7592, swd: 2.2457, target_std: 4.9747
      Epoch 1 composite train-obj: 1.616417
            Val objective improved inf → 1.3767, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.3403, mae: 1.7031, huber: 1.3094, swd: 1.2470, target_std: 6.0888
    Epoch [2/50], Val Losses: mse: 12.3379, mae: 1.9157, huber: 1.5389, swd: 1.2833, target_std: 4.4502
    Epoch [2/50], Test Losses: mse: 14.3103, mae: 2.4060, huber: 1.9976, swd: 3.4224, target_std: 4.9747
      Epoch 2 composite train-obj: 1.309363
            No improvement (1.5389), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.4069, mae: 1.5726, huber: 1.1853, swd: 0.9469, target_std: 6.0946
    Epoch [3/50], Val Losses: mse: 13.4733, mae: 2.0226, huber: 1.6402, swd: 1.1924, target_std: 4.4502
    Epoch [3/50], Test Losses: mse: 13.4787, mae: 2.2883, huber: 1.8853, swd: 2.5847, target_std: 4.9747
      Epoch 3 composite train-obj: 1.185299
            No improvement (1.6402), counter 2/5
    Epoch [4/50], Train Losses: mse: 5.6046, mae: 1.4582, huber: 1.0769, swd: 0.7474, target_std: 6.0746
    Epoch [4/50], Val Losses: mse: 14.6809, mae: 2.0784, huber: 1.6948, swd: 1.2658, target_std: 4.4502
    Epoch [4/50], Test Losses: mse: 13.3494, mae: 2.2447, huber: 1.8408, swd: 2.3386, target_std: 4.9747
      Epoch 4 composite train-obj: 1.076874
            No improvement (1.6948), counter 3/5
    Epoch [5/50], Train Losses: mse: 4.9303, mae: 1.3528, huber: 0.9776, swd: 0.5994, target_std: 6.0889
    Epoch [5/50], Val Losses: mse: 16.6592, mae: 2.1425, huber: 1.7593, swd: 1.0108, target_std: 4.4502
    Epoch [5/50], Test Losses: mse: 14.3995, mae: 2.3206, huber: 1.9134, swd: 2.4327, target_std: 4.9747
      Epoch 5 composite train-obj: 0.977581
            No improvement (1.7593), counter 4/5
    Epoch [6/50], Train Losses: mse: 4.3999, mae: 1.2645, huber: 0.8948, swd: 0.4689, target_std: 6.0942
    Epoch [6/50], Val Losses: mse: 16.2472, mae: 2.1116, huber: 1.7306, swd: 0.8512, target_std: 4.4502
    Epoch [6/50], Test Losses: mse: 14.5267, mae: 2.3474, huber: 1.9392, swd: 2.3726, target_std: 4.9747
      Epoch 6 composite train-obj: 0.894802
    Epoch [6/50], Test Losses: mse: 11.7136, mae: 2.1635, huber: 1.7592, swd: 2.2457, target_std: 4.9747
    Best round's Test MSE: 11.7136, MAE: 2.1635, SWD: 2.2457
    Best round's Validation MSE: 10.0872, MAE: 1.7476
    Best round's Test verification MSE : 11.7136, MAE: 2.1635, SWD: 2.2457
    Time taken: 74.80 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.9760, mae: 2.0784, huber: 1.6717, swd: 1.8976, target_std: 6.0982
    Epoch [1/50], Val Losses: mse: 12.0880, mae: 1.8815, huber: 1.5091, swd: 0.9461, target_std: 4.4502
    Epoch [1/50], Test Losses: mse: 11.9870, mae: 2.1482, huber: 1.7449, swd: 2.0759, target_std: 4.9747
      Epoch 1 composite train-obj: 1.671744
            Val objective improved inf → 1.5091, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.2326, mae: 1.6905, huber: 1.2974, swd: 1.1558, target_std: 6.0871
    Epoch [2/50], Val Losses: mse: 13.4703, mae: 2.0080, huber: 1.6331, swd: 1.4097, target_std: 4.4502
    Epoch [2/50], Test Losses: mse: 12.5046, mae: 2.2055, huber: 1.7989, swd: 2.5183, target_std: 4.9747
      Epoch 2 composite train-obj: 1.297387
            No improvement (1.6331), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.1131, mae: 1.5334, huber: 1.1477, swd: 0.8252, target_std: 6.0906
    Epoch [3/50], Val Losses: mse: 17.8370, mae: 2.3199, huber: 1.9309, swd: 1.8875, target_std: 4.4502
    Epoch [3/50], Test Losses: mse: 14.6929, mae: 2.4134, huber: 2.0015, swd: 2.9907, target_std: 4.9747
      Epoch 3 composite train-obj: 1.147664
            No improvement (1.9309), counter 2/5
    Epoch [4/50], Train Losses: mse: 5.2569, mae: 1.4074, huber: 1.0278, swd: 0.6493, target_std: 6.0778
    Epoch [4/50], Val Losses: mse: 16.2236, mae: 2.1989, huber: 1.8143, swd: 1.4279, target_std: 4.4502
    Epoch [4/50], Test Losses: mse: 13.1063, mae: 2.2535, huber: 1.8441, swd: 2.3054, target_std: 4.9747
      Epoch 4 composite train-obj: 1.027814
            No improvement (1.8143), counter 3/5
    Epoch [5/50], Train Losses: mse: 4.6128, mae: 1.3039, huber: 0.9306, swd: 0.5050, target_std: 6.0863
    Epoch [5/50], Val Losses: mse: 17.3478, mae: 2.2183, huber: 1.8333, swd: 1.0438, target_std: 4.4502
    Epoch [5/50], Test Losses: mse: 13.8799, mae: 2.2771, huber: 1.8678, swd: 2.0833, target_std: 4.9747
      Epoch 5 composite train-obj: 0.930602
            No improvement (1.8333), counter 4/5
    Epoch [6/50], Train Losses: mse: 4.2745, mae: 1.2526, huber: 0.8821, swd: 0.4551, target_std: 6.0951
    Epoch [6/50], Val Losses: mse: 16.7300, mae: 2.1888, huber: 1.8037, swd: 1.1772, target_std: 4.4502
    Epoch [6/50], Test Losses: mse: 13.8370, mae: 2.3005, huber: 1.8876, swd: 2.0820, target_std: 4.9747
      Epoch 6 composite train-obj: 0.882139
    Epoch [6/50], Test Losses: mse: 11.9870, mae: 2.1482, huber: 1.7449, swd: 2.0759, target_std: 4.9747
    Best round's Test MSE: 11.9870, MAE: 2.1482, SWD: 2.0759
    Best round's Validation MSE: 12.0880, MAE: 1.8815
    Best round's Test verification MSE : 11.9870, MAE: 2.1482, SWD: 2.0759
    Time taken: 74.36 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.6502, mae: 2.0547, huber: 1.6479, swd: 2.5796, target_std: 6.0977
    Epoch [1/50], Val Losses: mse: 11.4759, mae: 1.8449, huber: 1.4724, swd: 1.2534, target_std: 4.4502
    Epoch [1/50], Test Losses: mse: 12.3344, mae: 2.2204, huber: 1.8160, swd: 2.8838, target_std: 4.9747
      Epoch 1 composite train-obj: 1.647865
            Val objective improved inf → 1.4724, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.7661, mae: 1.7592, huber: 1.3634, swd: 1.5951, target_std: 6.0955
    Epoch [2/50], Val Losses: mse: 14.6311, mae: 2.0639, huber: 1.6838, swd: 1.8248, target_std: 4.4502
    Epoch [2/50], Test Losses: mse: 15.8085, mae: 2.5838, huber: 2.1689, swd: 3.8836, target_std: 4.9747
      Epoch 2 composite train-obj: 1.363422
            No improvement (1.6838), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.9877, mae: 1.6527, huber: 1.2618, swd: 1.2460, target_std: 6.0931
    Epoch [3/50], Val Losses: mse: 12.4276, mae: 1.9272, huber: 1.5528, swd: 1.0979, target_std: 4.4502
    Epoch [3/50], Test Losses: mse: 13.4137, mae: 2.3041, huber: 1.8972, swd: 2.1138, target_std: 4.9747
      Epoch 3 composite train-obj: 1.261763
            No improvement (1.5528), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.2444, mae: 1.5458, huber: 1.1598, swd: 0.9428, target_std: 6.0971
    Epoch [4/50], Val Losses: mse: 12.9473, mae: 1.9573, huber: 1.5785, swd: 0.9934, target_std: 4.4502
    Epoch [4/50], Test Losses: mse: 12.8841, mae: 2.2416, huber: 1.8333, swd: 1.9133, target_std: 4.9747
      Epoch 4 composite train-obj: 1.159797
            No improvement (1.5785), counter 3/5
    Epoch [5/50], Train Losses: mse: 5.5477, mae: 1.4376, huber: 1.0569, swd: 0.7500, target_std: 6.1020
    Epoch [5/50], Val Losses: mse: 14.7707, mae: 2.0922, huber: 1.7054, swd: 1.0465, target_std: 4.4502
    Epoch [5/50], Test Losses: mse: 13.9633, mae: 2.3438, huber: 1.9289, swd: 2.3465, target_std: 4.9747
      Epoch 5 composite train-obj: 1.056875
            No improvement (1.7054), counter 4/5
    Epoch [6/50], Train Losses: mse: 5.0010, mae: 1.3533, huber: 0.9773, swd: 0.6373, target_std: 6.0968
    Epoch [6/50], Val Losses: mse: 17.4685, mae: 2.2091, huber: 1.8250, swd: 1.1549, target_std: 4.4502
    Epoch [6/50], Test Losses: mse: 14.6296, mae: 2.3836, huber: 1.9674, swd: 2.1299, target_std: 4.9747
      Epoch 6 composite train-obj: 0.977342
    Epoch [6/50], Test Losses: mse: 12.3344, mae: 2.2204, huber: 1.8160, swd: 2.8838, target_std: 4.9747
    Best round's Test MSE: 12.3344, MAE: 2.2204, SWD: 2.8838
    Best round's Validation MSE: 11.4759, MAE: 1.8449
    Best round's Test verification MSE : 12.3344, MAE: 2.2204, SWD: 2.8838
    Time taken: 74.00 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth1_seq720_pred720_20250503_2034)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.0117 ± 0.2540
      mae: 2.1774 ± 0.0311
      huber: 1.7734 ± 0.0307
      swd: 2.4018 ± 0.3478
      target_std: 4.9747 ± 0.0000
      count: 3.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 11.2170 ± 0.8371
      mae: 1.8247 ± 0.0565
      huber: 1.4527 ± 0.0558
      swd: 1.0351 ± 0.1552
      target_std: 4.4502 ± 0.0000
      count: 3.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 223.33 seconds
    
    Experiment complete: TimeMixer_etth1_seq720_pred720_20250503_2034
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 8
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.6555, mae: 2.0888, huber: 1.6905, swd: 2.7639, target_std: 6.1756
    Epoch [1/50], Val Losses: mse: 7.4829, mae: 1.5341, huber: 1.1663, swd: 1.0491, target_std: 4.4805
    Epoch [1/50], Test Losses: mse: 10.3880, mae: 1.9025, huber: 1.5201, swd: 2.0551, target_std: 4.8844
      Epoch 1 composite train-obj: 1.690484
            Val objective improved inf → 1.1663, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.2451, mae: 1.6354, huber: 1.2520, swd: 1.6223, target_std: 6.1757
    Epoch [2/50], Val Losses: mse: 7.0827, mae: 1.4835, huber: 1.1169, swd: 1.0883, target_std: 4.4805
    Epoch [2/50], Test Losses: mse: 10.4828, mae: 1.9008, huber: 1.5172, swd: 2.6811, target_std: 4.8844
      Epoch 2 composite train-obj: 1.251965
            Val objective improved 1.1663 → 1.1169, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.6779, mae: 1.5573, huber: 1.1766, swd: 1.4240, target_std: 6.1756
    Epoch [3/50], Val Losses: mse: 7.7095, mae: 1.5466, huber: 1.1792, swd: 1.2091, target_std: 4.4805
    Epoch [3/50], Test Losses: mse: 10.3143, mae: 1.8654, huber: 1.4848, swd: 2.2545, target_std: 4.8844
      Epoch 3 composite train-obj: 1.176644
            No improvement (1.1792), counter 1/5
    Epoch [4/50], Train Losses: mse: 6.3007, mae: 1.5069, huber: 1.1278, swd: 1.3085, target_std: 6.1757
    Epoch [4/50], Val Losses: mse: 7.6216, mae: 1.5420, huber: 1.1744, swd: 1.3572, target_std: 4.4805
    Epoch [4/50], Test Losses: mse: 10.2097, mae: 1.8885, huber: 1.5034, swd: 2.5516, target_std: 4.8844
      Epoch 4 composite train-obj: 1.127814
            No improvement (1.1744), counter 2/5
    Epoch [5/50], Train Losses: mse: 5.9631, mae: 1.4622, huber: 1.0846, swd: 1.1785, target_std: 6.1756
    Epoch [5/50], Val Losses: mse: 7.3044, mae: 1.5087, huber: 1.1398, swd: 1.2272, target_std: 4.4805
    Epoch [5/50], Test Losses: mse: 10.2498, mae: 1.8660, huber: 1.4837, swd: 2.7272, target_std: 4.8844
      Epoch 5 composite train-obj: 1.084645
            No improvement (1.1398), counter 3/5
    Epoch [6/50], Train Losses: mse: 5.6840, mae: 1.4253, huber: 1.0491, swd: 1.0878, target_std: 6.1758
    Epoch [6/50], Val Losses: mse: 7.6614, mae: 1.5351, huber: 1.1671, swd: 1.1641, target_std: 4.4805
    Epoch [6/50], Test Losses: mse: 10.0706, mae: 1.8449, huber: 1.4630, swd: 2.3293, target_std: 4.8844
      Epoch 6 composite train-obj: 1.049081
            No improvement (1.1671), counter 4/5
    Epoch [7/50], Train Losses: mse: 5.3535, mae: 1.3766, huber: 1.0028, swd: 0.9523, target_std: 6.1759
    Epoch [7/50], Val Losses: mse: 8.4735, mae: 1.6104, huber: 1.2408, swd: 1.4717, target_std: 4.4805
    Epoch [7/50], Test Losses: mse: 10.2180, mae: 1.8670, huber: 1.4829, swd: 2.3329, target_std: 4.8844
      Epoch 7 composite train-obj: 1.002756
    Epoch [7/50], Test Losses: mse: 10.4828, mae: 1.9008, huber: 1.5172, swd: 2.6811, target_std: 4.8844
    Best round's Test MSE: 10.4828, MAE: 1.9008, SWD: 2.6811
    Best round's Validation MSE: 7.0827, MAE: 1.4835
    Best round's Test verification MSE : 10.4828, MAE: 1.9008, SWD: 2.6811
    Time taken: 60.54 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.6661, mae: 2.0493, huber: 1.6536, swd: 2.9982, target_std: 6.1755
    Epoch [1/50], Val Losses: mse: 7.5436, mae: 1.5319, huber: 1.1674, swd: 1.1458, target_std: 4.4805
    Epoch [1/50], Test Losses: mse: 10.2697, mae: 1.8886, huber: 1.5080, swd: 2.2277, target_std: 4.8844
      Epoch 1 composite train-obj: 1.653570
            Val objective improved inf → 1.1674, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.1878, mae: 1.6210, huber: 1.2391, swd: 1.5229, target_std: 6.1758
    Epoch [2/50], Val Losses: mse: 6.6913, mae: 1.4256, huber: 1.0615, swd: 0.9448, target_std: 4.4805
    Epoch [2/50], Test Losses: mse: 10.4097, mae: 1.8617, huber: 1.4801, swd: 2.4372, target_std: 4.8844
      Epoch 2 composite train-obj: 1.239056
            Val objective improved 1.1674 → 1.0615, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.6863, mae: 1.5511, huber: 1.1718, swd: 1.3653, target_std: 6.1755
    Epoch [3/50], Val Losses: mse: 7.5282, mae: 1.5408, huber: 1.1692, swd: 1.0276, target_std: 4.4805
    Epoch [3/50], Test Losses: mse: 10.0724, mae: 1.8313, huber: 1.4494, swd: 1.8495, target_std: 4.8844
      Epoch 3 composite train-obj: 1.171789
            No improvement (1.1692), counter 1/5
    Epoch [4/50], Train Losses: mse: 6.2792, mae: 1.5005, huber: 1.1219, swd: 1.2250, target_std: 6.1757
    Epoch [4/50], Val Losses: mse: 7.0791, mae: 1.4809, huber: 1.1168, swd: 1.4298, target_std: 4.4805
    Epoch [4/50], Test Losses: mse: 10.2360, mae: 1.8857, huber: 1.4994, swd: 2.7906, target_std: 4.8844
      Epoch 4 composite train-obj: 1.121872
            No improvement (1.1168), counter 2/5
    Epoch [5/50], Train Losses: mse: 5.9433, mae: 1.4527, huber: 1.0763, swd: 1.1171, target_std: 6.1756
    Epoch [5/50], Val Losses: mse: 7.4761, mae: 1.5118, huber: 1.1448, swd: 1.2446, target_std: 4.4805
    Epoch [5/50], Test Losses: mse: 9.9036, mae: 1.8261, huber: 1.4439, swd: 2.2140, target_std: 4.8844
      Epoch 5 composite train-obj: 1.076339
            No improvement (1.1448), counter 3/5
    Epoch [6/50], Train Losses: mse: 5.6549, mae: 1.4136, huber: 1.0387, swd: 1.0095, target_std: 6.1755
    Epoch [6/50], Val Losses: mse: 7.7130, mae: 1.5242, huber: 1.1555, swd: 1.0825, target_std: 4.4805
    Epoch [6/50], Test Losses: mse: 10.2559, mae: 1.8641, huber: 1.4793, swd: 2.1135, target_std: 4.8844
      Epoch 6 composite train-obj: 1.038682
            No improvement (1.1555), counter 4/5
    Epoch [7/50], Train Losses: mse: 5.3916, mae: 1.3750, huber: 1.0020, swd: 0.9130, target_std: 6.1758
    Epoch [7/50], Val Losses: mse: 8.0496, mae: 1.5541, huber: 1.1832, swd: 1.1350, target_std: 4.4805
    Epoch [7/50], Test Losses: mse: 10.0745, mae: 1.8495, huber: 1.4654, swd: 2.2422, target_std: 4.8844
      Epoch 7 composite train-obj: 1.001959
    Epoch [7/50], Test Losses: mse: 10.4097, mae: 1.8617, huber: 1.4801, swd: 2.4372, target_std: 4.8844
    Best round's Test MSE: 10.4097, MAE: 1.8617, SWD: 2.4372
    Best round's Validation MSE: 6.6913, MAE: 1.4256
    Best round's Test verification MSE : 10.4097, MAE: 1.8617, SWD: 2.4372
    Time taken: 60.46 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.7256, mae: 2.0524, huber: 1.6559, swd: 3.1266, target_std: 6.1755
    Epoch [1/50], Val Losses: mse: 7.5489, mae: 1.5163, huber: 1.1532, swd: 0.9265, target_std: 4.4805
    Epoch [1/50], Test Losses: mse: 10.3519, mae: 1.9121, huber: 1.5264, swd: 1.9828, target_std: 4.8844
      Epoch 1 composite train-obj: 1.655923
            Val objective improved inf → 1.1532, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.2130, mae: 1.6297, huber: 1.2470, swd: 1.4761, target_std: 6.1756
    Epoch [2/50], Val Losses: mse: 7.0977, mae: 1.5111, huber: 1.1432, swd: 1.2874, target_std: 4.4805
    Epoch [2/50], Test Losses: mse: 10.4771, mae: 1.8874, huber: 1.5042, swd: 2.4425, target_std: 4.8844
      Epoch 2 composite train-obj: 1.246993
            Val objective improved 1.1532 → 1.1432, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.7629, mae: 1.5671, huber: 1.1865, swd: 1.3542, target_std: 6.1755
    Epoch [3/50], Val Losses: mse: 6.9022, mae: 1.4738, huber: 1.1084, swd: 1.1672, target_std: 4.4805
    Epoch [3/50], Test Losses: mse: 10.3224, mae: 1.8647, huber: 1.4814, swd: 2.0965, target_std: 4.8844
      Epoch 3 composite train-obj: 1.186483
            Val objective improved 1.1432 → 1.1084, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.3974, mae: 1.5139, huber: 1.1354, swd: 1.2317, target_std: 6.1753
    Epoch [4/50], Val Losses: mse: 6.9692, mae: 1.4923, huber: 1.1259, swd: 1.3184, target_std: 4.4805
    Epoch [4/50], Test Losses: mse: 10.1961, mae: 1.8567, huber: 1.4734, swd: 2.3510, target_std: 4.8844
      Epoch 4 composite train-obj: 1.135376
            No improvement (1.1259), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.0733, mae: 1.4699, huber: 1.0927, swd: 1.1500, target_std: 6.1757
    Epoch [5/50], Val Losses: mse: 7.2047, mae: 1.5043, huber: 1.1405, swd: 1.4091, target_std: 4.4805
    Epoch [5/50], Test Losses: mse: 9.9491, mae: 1.8389, huber: 1.4569, swd: 2.2975, target_std: 4.8844
      Epoch 5 composite train-obj: 1.092745
            No improvement (1.1405), counter 2/5
    Epoch [6/50], Train Losses: mse: 5.7268, mae: 1.4204, huber: 1.0454, swd: 1.0241, target_std: 6.1754
    Epoch [6/50], Val Losses: mse: 8.1288, mae: 1.5888, huber: 1.2201, swd: 1.6340, target_std: 4.4805
    Epoch [6/50], Test Losses: mse: 10.3520, mae: 1.8976, huber: 1.5141, swd: 2.6442, target_std: 4.8844
      Epoch 6 composite train-obj: 1.045382
            No improvement (1.2201), counter 3/5
    Epoch [7/50], Train Losses: mse: 5.4486, mae: 1.3822, huber: 1.0086, swd: 0.9291, target_std: 6.1758
    Epoch [7/50], Val Losses: mse: 8.0841, mae: 1.5722, huber: 1.2024, swd: 1.4610, target_std: 4.4805
    Epoch [7/50], Test Losses: mse: 10.1018, mae: 1.8601, huber: 1.4767, swd: 2.4719, target_std: 4.8844
      Epoch 7 composite train-obj: 1.008629
            No improvement (1.2024), counter 4/5
    Epoch [8/50], Train Losses: mse: 5.2044, mae: 1.3485, huber: 0.9765, swd: 0.8464, target_std: 6.1755
    Epoch [8/50], Val Losses: mse: 9.0305, mae: 1.6134, huber: 1.2451, swd: 1.3087, target_std: 4.4805
    Epoch [8/50], Test Losses: mse: 10.1932, mae: 1.8728, huber: 1.4887, swd: 2.2069, target_std: 4.8844
      Epoch 8 composite train-obj: 0.976496
    Epoch [8/50], Test Losses: mse: 10.3224, mae: 1.8647, huber: 1.4814, swd: 2.0965, target_std: 4.8844
    Best round's Test MSE: 10.3224, MAE: 1.8647, SWD: 2.0965
    Best round's Validation MSE: 6.9022, MAE: 1.4738
    Best round's Test verification MSE : 10.3224, MAE: 1.8647, SWD: 2.0965
    Time taken: 68.97 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq720_pred96_20250503_2043)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.4050 ± 0.0656
      mae: 1.8757 ± 0.0178
      huber: 1.4929 ± 0.0172
      swd: 2.4049 ± 0.2397
      target_std: 4.8844 ± 0.0000
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 6.8921 ± 0.1600
      mae: 1.4610 ± 0.0253
      huber: 1.0956 ± 0.0244
      swd: 1.0668 ± 0.0921
      target_std: 4.4805 ± 0.0000
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 190.04 seconds
    
    Experiment complete: PatchTST_etth1_seq720_pred96_20250503_2043
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 7
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.0429, mae: 2.1454, huber: 1.7431, swd: 2.7950, target_std: 6.1613
    Epoch [1/50], Val Losses: mse: 9.0616, mae: 1.7053, huber: 1.3301, swd: 1.0911, target_std: 4.4590
    Epoch [1/50], Test Losses: mse: 11.4959, mae: 2.0626, huber: 1.6717, swd: 2.1290, target_std: 4.8938
      Epoch 1 composite train-obj: 1.743140
            Val objective improved inf → 1.3301, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.8196, mae: 1.7260, huber: 1.3362, swd: 1.5680, target_std: 6.1679
    Epoch [2/50], Val Losses: mse: 9.3728, mae: 1.7279, huber: 1.3523, swd: 1.3865, target_std: 4.4590
    Epoch [2/50], Test Losses: mse: 10.9011, mae: 1.9963, huber: 1.6060, swd: 2.6621, target_std: 4.8938
      Epoch 2 composite train-obj: 1.336240
            No improvement (1.3523), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.1801, mae: 1.6432, huber: 1.2552, swd: 1.3497, target_std: 6.1609
    Epoch [3/50], Val Losses: mse: 9.8449, mae: 1.8091, huber: 1.4293, swd: 2.2107, target_std: 4.4590
    Epoch [3/50], Test Losses: mse: 11.1186, mae: 2.0653, huber: 1.6688, swd: 3.2135, target_std: 4.8938
      Epoch 3 composite train-obj: 1.255248
            No improvement (1.4293), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.7269, mae: 1.5841, huber: 1.1980, swd: 1.2226, target_std: 6.1565
    Epoch [4/50], Val Losses: mse: 9.1319, mae: 1.7025, huber: 1.3268, swd: 1.2342, target_std: 4.4590
    Epoch [4/50], Test Losses: mse: 10.3330, mae: 1.9246, huber: 1.5345, swd: 2.1054, target_std: 4.8938
      Epoch 4 composite train-obj: 1.197979
            Val objective improved 1.3301 → 1.3268, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.3479, mae: 1.5327, huber: 1.1486, swd: 1.0819, target_std: 6.1691
    Epoch [5/50], Val Losses: mse: 9.4578, mae: 1.7478, huber: 1.3676, swd: 1.2978, target_std: 4.4590
    Epoch [5/50], Test Losses: mse: 10.5192, mae: 1.9709, huber: 1.5785, swd: 2.2311, target_std: 4.8938
      Epoch 5 composite train-obj: 1.148579
            No improvement (1.3676), counter 1/5
    Epoch [6/50], Train Losses: mse: 6.0179, mae: 1.4866, huber: 1.1042, swd: 0.9675, target_std: 6.1607
    Epoch [6/50], Val Losses: mse: 10.5304, mae: 1.8003, huber: 1.4212, swd: 1.3011, target_std: 4.4590
    Epoch [6/50], Test Losses: mse: 10.5610, mae: 1.9661, huber: 1.5728, swd: 2.0735, target_std: 4.8938
      Epoch 6 composite train-obj: 1.104211
            No improvement (1.4212), counter 2/5
    Epoch [7/50], Train Losses: mse: 5.7412, mae: 1.4476, huber: 1.0667, swd: 0.8650, target_std: 6.1641
    Epoch [7/50], Val Losses: mse: 9.6982, mae: 1.7518, huber: 1.3716, swd: 1.3859, target_std: 4.4590
    Epoch [7/50], Test Losses: mse: 10.6897, mae: 1.9916, huber: 1.5971, swd: 2.5884, target_std: 4.8938
      Epoch 7 composite train-obj: 1.066734
            No improvement (1.3716), counter 3/5
    Epoch [8/50], Train Losses: mse: 5.5007, mae: 1.4125, huber: 1.0333, swd: 0.8093, target_std: 6.1604
    Epoch [8/50], Val Losses: mse: 10.6158, mae: 1.7998, huber: 1.4203, swd: 1.1868, target_std: 4.4590
    Epoch [8/50], Test Losses: mse: 10.4870, mae: 1.9664, huber: 1.5731, swd: 2.1267, target_std: 4.8938
      Epoch 8 composite train-obj: 1.033296
            No improvement (1.4203), counter 4/5
    Epoch [9/50], Train Losses: mse: 5.2562, mae: 1.3759, huber: 0.9986, swd: 0.7304, target_std: 6.1619
    Epoch [9/50], Val Losses: mse: 12.1616, mae: 1.8727, huber: 1.4963, swd: 1.2812, target_std: 4.4590
    Epoch [9/50], Test Losses: mse: 10.9965, mae: 2.0370, huber: 1.6380, swd: 2.3438, target_std: 4.8938
      Epoch 9 composite train-obj: 0.998612
    Epoch [9/50], Test Losses: mse: 10.3330, mae: 1.9246, huber: 1.5345, swd: 2.1054, target_std: 4.8938
    Best round's Test MSE: 10.3330, MAE: 1.9246, SWD: 2.1054
    Best round's Validation MSE: 9.1319, MAE: 1.7025
    Best round's Test verification MSE : 10.3330, MAE: 1.9246, SWD: 2.1054
    Time taken: 77.48 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.9220, mae: 2.1144, huber: 1.7141, swd: 2.8133, target_std: 6.1620
    Epoch [1/50], Val Losses: mse: 8.7296, mae: 1.6688, huber: 1.2982, swd: 1.3223, target_std: 4.4590
    Epoch [1/50], Test Losses: mse: 11.2715, mae: 2.0313, huber: 1.6417, swd: 2.5112, target_std: 4.8938
      Epoch 1 composite train-obj: 1.714102
            Val objective improved inf → 1.2982, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.7788, mae: 1.7157, huber: 1.3271, swd: 1.5531, target_std: 6.1726
    Epoch [2/50], Val Losses: mse: 9.7366, mae: 1.7598, huber: 1.3850, swd: 1.7207, target_std: 4.4590
    Epoch [2/50], Test Losses: mse: 10.9253, mae: 2.0353, huber: 1.6417, swd: 2.8652, target_std: 4.8938
      Epoch 2 composite train-obj: 1.327135
            No improvement (1.3850), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.2551, mae: 1.6504, huber: 1.2631, swd: 1.3997, target_std: 6.1619
    Epoch [3/50], Val Losses: mse: 9.6678, mae: 1.7672, huber: 1.3877, swd: 1.6038, target_std: 4.4590
    Epoch [3/50], Test Losses: mse: 10.6684, mae: 1.9724, huber: 1.5795, swd: 2.6146, target_std: 4.8938
      Epoch 3 composite train-obj: 1.263128
            No improvement (1.3877), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.7033, mae: 1.5805, huber: 1.1952, swd: 1.2296, target_std: 6.1622
    Epoch [4/50], Val Losses: mse: 9.4223, mae: 1.7108, huber: 1.3369, swd: 1.2863, target_std: 4.4590
    Epoch [4/50], Test Losses: mse: 10.4988, mae: 1.9426, huber: 1.5509, swd: 2.2148, target_std: 4.8938
      Epoch 4 composite train-obj: 1.195161
            No improvement (1.3369), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.2860, mae: 1.5239, huber: 1.1403, swd: 1.0919, target_std: 6.1635
    Epoch [5/50], Val Losses: mse: 10.8166, mae: 1.8028, huber: 1.4269, swd: 1.1131, target_std: 4.4590
    Epoch [5/50], Test Losses: mse: 10.2115, mae: 1.8928, huber: 1.5032, swd: 1.6875, target_std: 4.8938
      Epoch 5 composite train-obj: 1.140337
            No improvement (1.4269), counter 4/5
    Epoch [6/50], Train Losses: mse: 5.9456, mae: 1.4770, huber: 1.0952, swd: 0.9643, target_std: 6.1607
    Epoch [6/50], Val Losses: mse: 10.1710, mae: 1.7894, huber: 1.4095, swd: 1.5336, target_std: 4.4590
    Epoch [6/50], Test Losses: mse: 10.4881, mae: 1.9619, huber: 1.5663, swd: 2.4232, target_std: 4.8938
      Epoch 6 composite train-obj: 1.095233
    Epoch [6/50], Test Losses: mse: 11.2715, mae: 2.0313, huber: 1.6417, swd: 2.5112, target_std: 4.8938
    Best round's Test MSE: 11.2715, MAE: 2.0313, SWD: 2.5112
    Best round's Validation MSE: 8.7296, MAE: 1.6688
    Best round's Test verification MSE : 11.2715, MAE: 2.0313, SWD: 2.5112
    Time taken: 51.73 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.3644, mae: 2.1565, huber: 1.7544, swd: 2.6690, target_std: 6.1555
    Epoch [1/50], Val Losses: mse: 9.2714, mae: 1.7234, huber: 1.3497, swd: 1.2863, target_std: 4.4590
    Epoch [1/50], Test Losses: mse: 11.5655, mae: 2.0759, huber: 1.6829, swd: 2.3521, target_std: 4.8938
      Epoch 1 composite train-obj: 1.754402
            Val objective improved inf → 1.3497, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.7987, mae: 1.7242, huber: 1.3346, swd: 1.3962, target_std: 6.1672
    Epoch [2/50], Val Losses: mse: 9.1050, mae: 1.7349, huber: 1.3566, swd: 1.3300, target_std: 4.4590
    Epoch [2/50], Test Losses: mse: 11.0754, mae: 2.0147, huber: 1.6225, swd: 2.0512, target_std: 4.8938
      Epoch 2 composite train-obj: 1.334623
            No improvement (1.3566), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.2170, mae: 1.6507, huber: 1.2631, swd: 1.2463, target_std: 6.1656
    Epoch [3/50], Val Losses: mse: 9.1465, mae: 1.7453, huber: 1.3638, swd: 1.3880, target_std: 4.4590
    Epoch [3/50], Test Losses: mse: 10.6717, mae: 1.9708, huber: 1.5789, swd: 2.0370, target_std: 4.8938
      Epoch 3 composite train-obj: 1.263070
            No improvement (1.3638), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.7394, mae: 1.5859, huber: 1.2003, swd: 1.1132, target_std: 6.1654
    Epoch [4/50], Val Losses: mse: 9.6460, mae: 1.7716, huber: 1.3915, swd: 1.0637, target_std: 4.4590
    Epoch [4/50], Test Losses: mse: 10.4229, mae: 1.9185, huber: 1.5285, swd: 1.5358, target_std: 4.8938
      Epoch 4 composite train-obj: 1.200350
            No improvement (1.3915), counter 3/5
    Epoch [5/50], Train Losses: mse: 6.3194, mae: 1.5285, huber: 1.1448, swd: 0.9877, target_std: 6.1608
    Epoch [5/50], Val Losses: mse: 9.5843, mae: 1.7825, huber: 1.4046, swd: 1.8784, target_std: 4.4590
    Epoch [5/50], Test Losses: mse: 10.8312, mae: 2.0245, huber: 1.6293, swd: 2.8910, target_std: 4.8938
      Epoch 5 composite train-obj: 1.144849
            No improvement (1.4046), counter 4/5
    Epoch [6/50], Train Losses: mse: 5.9884, mae: 1.4809, huber: 1.0993, swd: 0.8908, target_std: 6.1609
    Epoch [6/50], Val Losses: mse: 10.1377, mae: 1.7863, huber: 1.4099, swd: 1.4635, target_std: 4.4590
    Epoch [6/50], Test Losses: mse: 10.6312, mae: 1.9771, huber: 1.5844, swd: 2.2740, target_std: 4.8938
      Epoch 6 composite train-obj: 1.099335
    Epoch [6/50], Test Losses: mse: 11.5655, mae: 2.0759, huber: 1.6829, swd: 2.3521, target_std: 4.8938
    Best round's Test MSE: 11.5655, MAE: 2.0759, SWD: 2.3521
    Best round's Validation MSE: 9.2714, MAE: 1.7234
    Best round's Test verification MSE : 11.5655, MAE: 2.0759, SWD: 2.3521
    Time taken: 51.85 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq720_pred196_20250503_2046)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.0567 ± 0.5256
      mae: 2.0106 ± 0.0635
      huber: 1.6197 ± 0.0626
      swd: 2.3229 ± 0.1670
      target_std: 4.8938 ± 0.0000
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.0443 ± 0.2297
      mae: 1.6982 ± 0.0225
      huber: 1.3249 ± 0.0211
      swd: 1.2809 ± 0.0361
      target_std: 4.4590 ± 0.0000
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 181.16 seconds
    
    Experiment complete: PatchTST_etth1_seq720_pred196_20250503_2046
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 88
    Validation Batches: 6
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.9236, mae: 2.2348, huber: 1.8283, swd: 2.9142, target_std: 6.1419
    Epoch [1/50], Val Losses: mse: 13.1638, mae: 2.0978, huber: 1.7119, swd: 2.0643, target_std: 4.4388
    Epoch [1/50], Test Losses: mse: 15.8612, mae: 2.5648, huber: 2.1546, swd: 2.7618, target_std: 4.9300
      Epoch 1 composite train-obj: 1.828270
            Val objective improved inf → 1.7119, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.7805, mae: 1.8633, huber: 1.4652, swd: 1.6246, target_std: 6.1467
    Epoch [2/50], Val Losses: mse: 9.9935, mae: 1.8045, huber: 1.4199, swd: 1.2754, target_std: 4.4388
    Epoch [2/50], Test Losses: mse: 12.5288, mae: 2.2471, huber: 1.8419, swd: 2.6220, target_std: 4.9300
      Epoch 2 composite train-obj: 1.465197
            Val objective improved 1.7119 → 1.4199, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.9938, mae: 1.7707, huber: 1.3749, swd: 1.5243, target_std: 6.1549
    Epoch [3/50], Val Losses: mse: 10.1419, mae: 1.7944, huber: 1.4147, swd: 1.1769, target_std: 4.4388
    Epoch [3/50], Test Losses: mse: 11.3369, mae: 2.1215, huber: 1.7199, swd: 2.5903, target_std: 4.9300
      Epoch 3 composite train-obj: 1.374938
            Val objective improved 1.4199 → 1.4147, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.4827, mae: 1.7017, huber: 1.3088, swd: 1.3495, target_std: 6.1548
    Epoch [4/50], Val Losses: mse: 12.2435, mae: 1.9821, huber: 1.5939, swd: 1.2412, target_std: 4.4388
    Epoch [4/50], Test Losses: mse: 11.8830, mae: 2.1022, huber: 1.7048, swd: 1.8130, target_std: 4.9300
      Epoch 4 composite train-obj: 1.308840
            No improvement (1.5939), counter 1/5
    Epoch [5/50], Train Losses: mse: 7.0533, mae: 1.6395, huber: 1.2492, swd: 1.1958, target_std: 6.1439
    Epoch [5/50], Val Losses: mse: 10.9912, mae: 1.8726, huber: 1.4918, swd: 0.9978, target_std: 4.4388
    Epoch [5/50], Test Losses: mse: 10.9720, mae: 2.0118, huber: 1.6187, swd: 1.6649, target_std: 4.9300
      Epoch 5 composite train-obj: 1.249231
            No improvement (1.4918), counter 2/5
    Epoch [6/50], Train Losses: mse: 6.7146, mae: 1.5955, huber: 1.2067, swd: 1.0625, target_std: 6.1633
    Epoch [6/50], Val Losses: mse: 9.6531, mae: 1.8000, huber: 1.4152, swd: 1.0549, target_std: 4.4388
    Epoch [6/50], Test Losses: mse: 11.8294, mae: 2.1054, huber: 1.7066, swd: 1.7839, target_std: 4.9300
      Epoch 6 composite train-obj: 1.206744
            No improvement (1.4152), counter 3/5
    Epoch [7/50], Train Losses: mse: 6.6841, mae: 1.5928, huber: 1.2036, swd: 1.0463, target_std: 6.1503
    Epoch [7/50], Val Losses: mse: 10.6874, mae: 1.9368, huber: 1.5430, swd: 1.7273, target_std: 4.4388
    Epoch [7/50], Test Losses: mse: 12.3942, mae: 2.3123, huber: 1.8977, swd: 3.4858, target_std: 4.9300
      Epoch 7 composite train-obj: 1.203560
            No improvement (1.5430), counter 4/5
    Epoch [8/50], Train Losses: mse: 6.6329, mae: 1.5849, huber: 1.1956, swd: 1.0920, target_std: 6.1516
    Epoch [8/50], Val Losses: mse: 10.6012, mae: 1.8607, huber: 1.4773, swd: 1.1630, target_std: 4.4388
    Epoch [8/50], Test Losses: mse: 10.9576, mae: 2.0545, huber: 1.6558, swd: 1.9209, target_std: 4.9300
      Epoch 8 composite train-obj: 1.195644
    Epoch [8/50], Test Losses: mse: 11.3369, mae: 2.1215, huber: 1.7199, swd: 2.5903, target_std: 4.9300
    Best round's Test MSE: 11.3369, MAE: 2.1215, SWD: 2.5903
    Best round's Validation MSE: 10.1419, MAE: 1.7944
    Best round's Test verification MSE : 11.3369, MAE: 2.1215, SWD: 2.5903
    Time taken: 68.39 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.6222, mae: 2.2004, huber: 1.7951, swd: 2.8985, target_std: 6.1448
    Epoch [1/50], Val Losses: mse: 13.1553, mae: 2.0851, huber: 1.7013, swd: 2.2109, target_std: 4.4388
    Epoch [1/50], Test Losses: mse: 14.9189, mae: 2.5001, huber: 2.0899, swd: 2.6199, target_std: 4.9300
      Epoch 1 composite train-obj: 1.795146
            Val objective improved inf → 1.7013, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.7022, mae: 1.8533, huber: 1.4557, swd: 1.7227, target_std: 6.1502
    Epoch [2/50], Val Losses: mse: 10.1501, mae: 1.7900, huber: 1.4114, swd: 1.0282, target_std: 4.4388
    Epoch [2/50], Test Losses: mse: 11.2203, mae: 2.0528, huber: 1.6569, swd: 1.9029, target_std: 4.9300
      Epoch 2 composite train-obj: 1.455688
            Val objective improved 1.7013 → 1.4114, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.8225, mae: 1.7454, huber: 1.3510, swd: 1.5132, target_std: 6.1529
    Epoch [3/50], Val Losses: mse: 10.5729, mae: 1.8820, huber: 1.4974, swd: 1.7867, target_std: 4.4388
    Epoch [3/50], Test Losses: mse: 11.7809, mae: 2.1717, huber: 1.7683, swd: 2.7401, target_std: 4.9300
      Epoch 3 composite train-obj: 1.350951
            No improvement (1.4974), counter 1/5
    Epoch [4/50], Train Losses: mse: 7.5503, mae: 1.7053, huber: 1.3128, swd: 1.4355, target_std: 6.1493
    Epoch [4/50], Val Losses: mse: 10.6282, mae: 1.8477, huber: 1.4652, swd: 0.8425, target_std: 4.4388
    Epoch [4/50], Test Losses: mse: 11.1414, mae: 2.0092, huber: 1.6147, swd: 1.2085, target_std: 4.9300
      Epoch 4 composite train-obj: 1.312779
            No improvement (1.4652), counter 2/5
    Epoch [5/50], Train Losses: mse: 7.0623, mae: 1.6399, huber: 1.2495, swd: 1.2326, target_std: 6.1642
    Epoch [5/50], Val Losses: mse: 9.5983, mae: 1.8021, huber: 1.4168, swd: 1.0190, target_std: 4.4388
    Epoch [5/50], Test Losses: mse: 10.8991, mae: 2.0096, huber: 1.6152, swd: 1.5736, target_std: 4.9300
      Epoch 5 composite train-obj: 1.249532
            No improvement (1.4168), counter 3/5
    Epoch [6/50], Train Losses: mse: 7.0906, mae: 1.6441, huber: 1.2534, swd: 1.2477, target_std: 6.1609
    Epoch [6/50], Val Losses: mse: 9.2203, mae: 1.7277, huber: 1.3496, swd: 0.9586, target_std: 4.4388
    Epoch [6/50], Test Losses: mse: 11.1724, mae: 2.0428, huber: 1.6475, swd: 1.8257, target_std: 4.9300
      Epoch 6 composite train-obj: 1.253418
            Val objective improved 1.4114 → 1.3496, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 6.6894, mae: 1.5858, huber: 1.1979, swd: 1.0660, target_std: 6.1267
    Epoch [7/50], Val Losses: mse: 9.4850, mae: 1.7843, huber: 1.4009, swd: 1.1327, target_std: 4.4388
    Epoch [7/50], Test Losses: mse: 11.0276, mae: 2.0423, huber: 1.6434, swd: 1.8642, target_std: 4.9300
      Epoch 7 composite train-obj: 1.197899
            No improvement (1.4009), counter 1/5
    Epoch [8/50], Train Losses: mse: 6.4087, mae: 1.5473, huber: 1.1609, swd: 0.9685, target_std: 6.1599
    Epoch [8/50], Val Losses: mse: 9.6781, mae: 1.8112, huber: 1.4271, swd: 1.0219, target_std: 4.4388
    Epoch [8/50], Test Losses: mse: 11.2780, mae: 2.0584, huber: 1.6603, swd: 1.7461, target_std: 4.9300
      Epoch 8 composite train-obj: 1.160912
            No improvement (1.4271), counter 2/5
    Epoch [9/50], Train Losses: mse: 6.2139, mae: 1.5218, huber: 1.1363, swd: 0.8971, target_std: 6.1387
    Epoch [9/50], Val Losses: mse: 9.0730, mae: 1.7208, huber: 1.3435, swd: 0.8730, target_std: 4.4388
    Epoch [9/50], Test Losses: mse: 10.8335, mae: 2.0053, huber: 1.6091, swd: 1.6939, target_std: 4.9300
      Epoch 9 composite train-obj: 1.136263
            Val objective improved 1.3496 → 1.3435, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 5.8990, mae: 1.4757, huber: 1.0923, swd: 0.7995, target_std: 6.1438
    Epoch [10/50], Val Losses: mse: 9.0796, mae: 1.7421, huber: 1.3626, swd: 1.5449, target_std: 4.4388
    Epoch [10/50], Test Losses: mse: 11.2723, mae: 2.1141, huber: 1.7106, swd: 2.8640, target_std: 4.9300
      Epoch 10 composite train-obj: 1.092303
            No improvement (1.3626), counter 1/5
    Epoch [11/50], Train Losses: mse: 5.7623, mae: 1.4587, huber: 1.0756, swd: 0.7721, target_std: 6.1457
    Epoch [11/50], Val Losses: mse: 9.6445, mae: 1.8321, huber: 1.4415, swd: 1.3486, target_std: 4.4388
    Epoch [11/50], Test Losses: mse: 12.1083, mae: 2.2194, huber: 1.8078, swd: 2.6193, target_std: 4.9300
      Epoch 11 composite train-obj: 1.075619
            No improvement (1.4415), counter 2/5
    Epoch [12/50], Train Losses: mse: 5.7953, mae: 1.4713, huber: 1.0865, swd: 0.7796, target_std: 6.1495
    Epoch [12/50], Val Losses: mse: 9.4089, mae: 1.7661, huber: 1.3822, swd: 1.2029, target_std: 4.4388
    Epoch [12/50], Test Losses: mse: 11.9371, mae: 2.1912, huber: 1.7788, swd: 2.6236, target_std: 4.9300
      Epoch 12 composite train-obj: 1.086487
            No improvement (1.3822), counter 3/5
    Epoch [13/50], Train Losses: mse: 5.4981, mae: 1.4242, huber: 1.0419, swd: 0.7088, target_std: 6.1414
    Epoch [13/50], Val Losses: mse: 9.7896, mae: 1.7975, huber: 1.4159, swd: 1.3702, target_std: 4.4388
    Epoch [13/50], Test Losses: mse: 11.8733, mae: 2.1583, huber: 1.7534, swd: 2.2909, target_std: 4.9300
      Epoch 13 composite train-obj: 1.041872
            No improvement (1.4159), counter 4/5
    Epoch [14/50], Train Losses: mse: 5.3128, mae: 1.3964, huber: 1.0154, swd: 0.6395, target_std: 6.1555
    Epoch [14/50], Val Losses: mse: 10.3011, mae: 1.8333, huber: 1.4494, swd: 1.1275, target_std: 4.4388
    Epoch [14/50], Test Losses: mse: 12.0981, mae: 2.1361, huber: 1.7322, swd: 1.7973, target_std: 4.9300
      Epoch 14 composite train-obj: 1.015435
    Epoch [14/50], Test Losses: mse: 10.8335, mae: 2.0053, huber: 1.6091, swd: 1.6939, target_std: 4.9300
    Best round's Test MSE: 10.8335, MAE: 2.0053, SWD: 1.6939
    Best round's Validation MSE: 9.0730, MAE: 1.7208
    Best round's Test verification MSE : 10.8335, MAE: 2.0053, SWD: 1.6939
    Time taken: 117.95 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.5952, mae: 2.2001, huber: 1.7945, swd: 2.9224, target_std: 6.1470
    Epoch [1/50], Val Losses: mse: 10.0983, mae: 1.8515, huber: 1.4670, swd: 0.7555, target_std: 4.4388
    Epoch [1/50], Test Losses: mse: 16.5039, mae: 2.5637, huber: 2.1555, swd: 1.6291, target_std: 4.9300
      Epoch 1 composite train-obj: 1.794518
            Val objective improved inf → 1.4670, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.5861, mae: 1.8446, huber: 1.4473, swd: 1.5642, target_std: 6.1379
    Epoch [2/50], Val Losses: mse: 9.6384, mae: 1.7516, huber: 1.3745, swd: 0.9295, target_std: 4.4388
    Epoch [2/50], Test Losses: mse: 12.0234, mae: 2.1840, huber: 1.7810, swd: 1.8784, target_std: 4.9300
      Epoch 2 composite train-obj: 1.447260
            Val objective improved 1.4670 → 1.3745, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.9878, mae: 1.7710, huber: 1.3755, swd: 1.4206, target_std: 6.1447
    Epoch [3/50], Val Losses: mse: 10.3088, mae: 1.8583, huber: 1.4772, swd: 0.8910, target_std: 4.4388
    Epoch [3/50], Test Losses: mse: 12.2202, mae: 2.1789, huber: 1.7811, swd: 1.2041, target_std: 4.9300
      Epoch 3 composite train-obj: 1.375494
            No improvement (1.4772), counter 1/5
    Epoch [4/50], Train Losses: mse: 7.3953, mae: 1.6926, huber: 1.3000, swd: 1.2775, target_std: 6.1476
    Epoch [4/50], Val Losses: mse: 10.0190, mae: 1.7797, huber: 1.3980, swd: 0.9863, target_std: 4.4388
    Epoch [4/50], Test Losses: mse: 11.5193, mae: 2.1106, huber: 1.7096, swd: 1.9165, target_std: 4.9300
      Epoch 4 composite train-obj: 1.299951
            No improvement (1.3980), counter 2/5
    Epoch [5/50], Train Losses: mse: 6.9736, mae: 1.6348, huber: 1.2441, swd: 1.1331, target_std: 6.1361
    Epoch [5/50], Val Losses: mse: 9.9861, mae: 1.7935, huber: 1.4105, swd: 0.9462, target_std: 4.4388
    Epoch [5/50], Test Losses: mse: 11.0510, mae: 2.0264, huber: 1.6297, swd: 1.6912, target_std: 4.9300
      Epoch 5 composite train-obj: 1.244054
            No improvement (1.4105), counter 3/5
    Epoch [6/50], Train Losses: mse: 6.6583, mae: 1.5925, huber: 1.2030, swd: 1.0285, target_std: 6.1395
    Epoch [6/50], Val Losses: mse: 10.4166, mae: 1.8301, huber: 1.4450, swd: 1.0674, target_std: 4.4388
    Epoch [6/50], Test Losses: mse: 12.0165, mae: 2.1406, huber: 1.7353, swd: 1.8981, target_std: 4.9300
      Epoch 6 composite train-obj: 1.203040
            No improvement (1.4450), counter 4/5
    Epoch [7/50], Train Losses: mse: 6.4653, mae: 1.5645, huber: 1.1760, swd: 0.9612, target_std: 6.1409
    Epoch [7/50], Val Losses: mse: 9.2192, mae: 1.7173, huber: 1.3381, swd: 1.1213, target_std: 4.4388
    Epoch [7/50], Test Losses: mse: 12.1601, mae: 2.2647, huber: 1.8472, swd: 2.8450, target_std: 4.9300
      Epoch 7 composite train-obj: 1.176014
            Val objective improved 1.3745 → 1.3381, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 6.2308, mae: 1.5279, huber: 1.1409, swd: 0.8821, target_std: 6.1498
    Epoch [8/50], Val Losses: mse: 11.4367, mae: 1.8780, huber: 1.4954, swd: 0.7658, target_std: 4.4388
    Epoch [8/50], Test Losses: mse: 12.0335, mae: 2.1137, huber: 1.7076, swd: 1.5235, target_std: 4.9300
      Epoch 8 composite train-obj: 1.140941
            No improvement (1.4954), counter 1/5
    Epoch [9/50], Train Losses: mse: 5.9466, mae: 1.4852, huber: 1.1007, swd: 0.7977, target_std: 6.1301
    Epoch [9/50], Val Losses: mse: 9.8622, mae: 1.7790, huber: 1.3969, swd: 0.9453, target_std: 4.4388
    Epoch [9/50], Test Losses: mse: 11.3422, mae: 2.1003, huber: 1.6968, swd: 2.0804, target_std: 4.9300
      Epoch 9 composite train-obj: 1.100669
            No improvement (1.3969), counter 2/5
    Epoch [10/50], Train Losses: mse: 5.7447, mae: 1.4601, huber: 1.0759, swd: 0.7476, target_std: 6.1301
    Epoch [10/50], Val Losses: mse: 10.7149, mae: 1.8224, huber: 1.4410, swd: 0.8206, target_std: 4.4388
    Epoch [10/50], Test Losses: mse: 11.1722, mae: 2.0433, huber: 1.6430, swd: 1.7000, target_std: 4.9300
      Epoch 10 composite train-obj: 1.075931
            No improvement (1.4410), counter 3/5
    Epoch [11/50], Train Losses: mse: 5.5771, mae: 1.4336, huber: 1.0509, swd: 0.7083, target_std: 6.1422
    Epoch [11/50], Val Losses: mse: 11.1792, mae: 1.8436, huber: 1.4628, swd: 0.9878, target_std: 4.4388
    Epoch [11/50], Test Losses: mse: 11.7881, mae: 2.1183, huber: 1.7144, swd: 1.8634, target_std: 4.9300
      Epoch 11 composite train-obj: 1.050901
            No improvement (1.4628), counter 4/5
    Epoch [12/50], Train Losses: mse: 5.4544, mae: 1.4160, huber: 1.0342, swd: 0.6842, target_std: 6.1531
    Epoch [12/50], Val Losses: mse: 14.7475, mae: 2.1057, huber: 1.7121, swd: 1.4408, target_std: 4.4388
    Epoch [12/50], Test Losses: mse: 15.6418, mae: 2.4295, huber: 2.0154, swd: 2.0832, target_std: 4.9300
      Epoch 12 composite train-obj: 1.034227
    Epoch [12/50], Test Losses: mse: 12.1601, mae: 2.2647, huber: 1.8472, swd: 2.8450, target_std: 4.9300
    Best round's Test MSE: 12.1601, MAE: 2.2647, SWD: 2.8450
    Best round's Validation MSE: 9.2192, MAE: 1.7173
    Best round's Test verification MSE : 12.1601, MAE: 2.2647, SWD: 2.8450
    Time taken: 101.12 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq720_pred336_20250503_2049)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.4435 ± 0.5468
      mae: 2.1305 ± 0.1061
      huber: 1.7254 ± 0.0973
      swd: 2.3764 ± 0.4936
      target_std: 4.9300 ± 0.0000
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.4780 ± 0.4732
      mae: 1.7442 ± 0.0356
      huber: 1.3654 ± 0.0349
      swd: 1.0571 ± 0.1321
      target_std: 4.4388 ± 0.0000
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 287.61 seconds
    
    Experiment complete: PatchTST_etth1_seq720_pred336_20250503_2049
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 85
    Validation Batches: 3
    Test Batches: 16
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 15.1454, mae: 2.4038, huber: 1.9883, swd: 3.1421, target_std: 6.0909
    Epoch [1/50], Val Losses: mse: 14.7002, mae: 2.1951, huber: 1.8121, swd: 2.3488, target_std: 4.4502
    Epoch [1/50], Test Losses: mse: 17.7030, mae: 2.7669, huber: 2.3507, swd: 2.9474, target_std: 4.9747
      Epoch 1 composite train-obj: 1.988283
            Val objective improved inf → 1.8121, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.2129, mae: 2.0450, huber: 1.6359, swd: 2.0952, target_std: 6.1003
    Epoch [2/50], Val Losses: mse: 11.6333, mae: 1.9483, huber: 1.5674, swd: 2.1147, target_std: 4.4502
    Epoch [2/50], Test Losses: mse: 13.3331, mae: 2.4142, huber: 2.0035, swd: 3.4613, target_std: 4.9747
      Epoch 2 composite train-obj: 1.635860
            Val objective improved 1.8121 → 1.5674, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 9.2741, mae: 1.9483, huber: 1.5412, swd: 1.9674, target_std: 6.0772
    Epoch [3/50], Val Losses: mse: 11.6297, mae: 1.9014, huber: 1.5268, swd: 1.3567, target_std: 4.4502
    Epoch [3/50], Test Losses: mse: 12.0887, mae: 2.2357, huber: 1.8292, swd: 2.5595, target_std: 4.9747
      Epoch 3 composite train-obj: 1.541202
            Val objective improved 1.5674 → 1.5268, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.3885, mae: 1.8234, huber: 1.4212, swd: 1.6254, target_std: 6.0930
    Epoch [4/50], Val Losses: mse: 11.0364, mae: 1.8395, huber: 1.4638, swd: 1.0530, target_std: 4.4502
    Epoch [4/50], Test Losses: mse: 12.0481, mae: 2.2227, huber: 1.8156, swd: 2.3014, target_std: 4.9747
      Epoch 4 composite train-obj: 1.421250
            Val objective improved 1.5268 → 1.4638, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.8931, mae: 1.7590, huber: 1.3588, swd: 1.4426, target_std: 6.1052
    Epoch [5/50], Val Losses: mse: 10.6111, mae: 1.8695, huber: 1.4908, swd: 0.8796, target_std: 4.4502
    Epoch [5/50], Test Losses: mse: 12.7489, mae: 2.2973, huber: 1.8893, swd: 2.0761, target_std: 4.9747
      Epoch 5 composite train-obj: 1.358825
            No improvement (1.4908), counter 1/5
    Epoch [6/50], Train Losses: mse: 7.5569, mae: 1.7175, huber: 1.3184, swd: 1.2665, target_std: 6.0776
    Epoch [6/50], Val Losses: mse: 9.9300, mae: 1.7925, huber: 1.4174, swd: 0.9208, target_std: 4.4502
    Epoch [6/50], Test Losses: mse: 11.6436, mae: 2.1606, huber: 1.7530, swd: 2.0726, target_std: 4.9747
      Epoch 6 composite train-obj: 1.318357
            Val objective improved 1.4638 → 1.4174, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.2418, mae: 1.6725, huber: 1.2752, swd: 1.1433, target_std: 6.0847
    Epoch [7/50], Val Losses: mse: 11.7735, mae: 2.1399, huber: 1.7343, swd: 1.2477, target_std: 4.4502
    Epoch [7/50], Test Losses: mse: 14.2334, mae: 2.4683, huber: 2.0452, swd: 2.0135, target_std: 4.9747
      Epoch 7 composite train-obj: 1.275238
            No improvement (1.7343), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.8367, mae: 1.7632, huber: 1.3612, swd: 1.3549, target_std: 6.0798
    Epoch [8/50], Val Losses: mse: 11.8659, mae: 1.9979, huber: 1.6074, swd: 1.5465, target_std: 4.4502
    Epoch [8/50], Test Losses: mse: 12.3147, mae: 2.2783, huber: 1.8626, swd: 2.4118, target_std: 4.9747
      Epoch 8 composite train-obj: 1.361214
            No improvement (1.6074), counter 2/5
    Epoch [9/50], Train Losses: mse: 7.1362, mae: 1.6655, huber: 1.2681, swd: 1.1013, target_std: 6.0780
    Epoch [9/50], Val Losses: mse: 11.4160, mae: 1.9426, huber: 1.5577, swd: 1.4232, target_std: 4.4502
    Epoch [9/50], Test Losses: mse: 11.9104, mae: 2.2112, huber: 1.7986, swd: 2.2276, target_std: 4.9747
      Epoch 9 composite train-obj: 1.268086
            No improvement (1.5577), counter 3/5
    Epoch [10/50], Train Losses: mse: 6.8609, mae: 1.6214, huber: 1.2261, swd: 1.0054, target_std: 6.0845
    Epoch [10/50], Val Losses: mse: 11.1661, mae: 2.0666, huber: 1.6677, swd: 1.5325, target_std: 4.4502
    Epoch [10/50], Test Losses: mse: 13.7364, mae: 2.4503, huber: 2.0267, swd: 2.6327, target_std: 4.9747
      Epoch 10 composite train-obj: 1.226105
            No improvement (1.6677), counter 4/5
    Epoch [11/50], Train Losses: mse: 7.1586, mae: 1.6689, huber: 1.2708, swd: 1.1126, target_std: 6.0826
    Epoch [11/50], Val Losses: mse: 12.1559, mae: 1.9604, huber: 1.5821, swd: 1.5916, target_std: 4.4502
    Epoch [11/50], Test Losses: mse: 11.5825, mae: 2.1781, huber: 1.7657, swd: 2.1372, target_std: 4.9747
      Epoch 11 composite train-obj: 1.270781
    Epoch [11/50], Test Losses: mse: 11.6436, mae: 2.1606, huber: 1.7530, swd: 2.0726, target_std: 4.9747
    Best round's Test MSE: 11.6436, MAE: 2.1606, SWD: 2.0726
    Best round's Validation MSE: 9.9300, MAE: 1.7925
    Best round's Test verification MSE : 11.6436, MAE: 2.1606, SWD: 2.0726
    Time taken: 92.29 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 15.1354, mae: 2.3732, huber: 1.9593, swd: 3.3281, target_std: 6.0798
    Epoch [1/50], Val Losses: mse: 14.3066, mae: 2.1689, huber: 1.7889, swd: 2.4624, target_std: 4.4502
    Epoch [1/50], Test Losses: mse: 15.8261, mae: 2.6387, huber: 2.2245, swd: 3.2270, target_std: 4.9747
      Epoch 1 composite train-obj: 1.959263
            Val objective improved inf → 1.7889, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.9296, mae: 2.0155, huber: 1.6075, swd: 1.9525, target_std: 6.0958
    Epoch [2/50], Val Losses: mse: 13.0733, mae: 2.0091, huber: 1.6278, swd: 1.2336, target_std: 4.4502
    Epoch [2/50], Test Losses: mse: 13.6346, mae: 2.4271, huber: 2.0138, swd: 2.9578, target_std: 4.9747
      Epoch 2 composite train-obj: 1.607520
            Val objective improved 1.7889 → 1.6278, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.9669, mae: 1.9082, huber: 1.5022, swd: 1.7046, target_std: 6.0989
    Epoch [3/50], Val Losses: mse: 12.0409, mae: 1.9509, huber: 1.5715, swd: 1.4319, target_std: 4.4502
    Epoch [3/50], Test Losses: mse: 12.6150, mae: 2.3393, huber: 1.9238, swd: 3.0278, target_std: 4.9747
      Epoch 3 composite train-obj: 1.502215
            Val objective improved 1.6278 → 1.5715, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.3301, mae: 1.8273, huber: 1.4245, swd: 1.4729, target_std: 6.0895
    Epoch [4/50], Val Losses: mse: 10.6497, mae: 1.8332, huber: 1.4564, swd: 0.8682, target_std: 4.4502
    Epoch [4/50], Test Losses: mse: 12.2374, mae: 2.2718, huber: 1.8608, swd: 2.2102, target_std: 4.9747
      Epoch 4 composite train-obj: 1.424506
            Val objective improved 1.5715 → 1.4564, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.7842, mae: 1.7565, huber: 1.3560, swd: 1.2874, target_std: 6.0907
    Epoch [5/50], Val Losses: mse: 13.5243, mae: 2.0407, huber: 1.6590, swd: 1.5245, target_std: 4.4502
    Epoch [5/50], Test Losses: mse: 12.5199, mae: 2.2799, huber: 1.8693, swd: 2.6170, target_std: 4.9747
      Epoch 5 composite train-obj: 1.356011
            No improvement (1.6590), counter 1/5
    Epoch [6/50], Train Losses: mse: 7.4392, mae: 1.7086, huber: 1.3099, swd: 1.1620, target_std: 6.0950
    Epoch [6/50], Val Losses: mse: 11.9528, mae: 2.0181, huber: 1.6316, swd: 2.2901, target_std: 4.4502
    Epoch [6/50], Test Losses: mse: 12.8399, mae: 2.3642, huber: 1.9470, swd: 3.9535, target_std: 4.9747
      Epoch 6 composite train-obj: 1.309909
            No improvement (1.6316), counter 2/5
    Epoch [7/50], Train Losses: mse: 7.3353, mae: 1.6974, huber: 1.2987, swd: 1.1550, target_std: 6.0878
    Epoch [7/50], Val Losses: mse: 12.3192, mae: 1.9739, huber: 1.5913, swd: 1.4474, target_std: 4.4502
    Epoch [7/50], Test Losses: mse: 11.9247, mae: 2.2197, huber: 1.8081, swd: 2.3973, target_std: 4.9747
      Epoch 7 composite train-obj: 1.298728
            No improvement (1.5913), counter 3/5
    Epoch [8/50], Train Losses: mse: 6.9743, mae: 1.6382, huber: 1.2428, swd: 0.9999, target_std: 6.0989
    Epoch [8/50], Val Losses: mse: 15.4134, mae: 2.0858, huber: 1.7063, swd: 0.9698, target_std: 4.4502
    Epoch [8/50], Test Losses: mse: 12.4911, mae: 2.2419, huber: 1.8324, swd: 1.7294, target_std: 4.9747
      Epoch 8 composite train-obj: 1.242813
            No improvement (1.7063), counter 4/5
    Epoch [9/50], Train Losses: mse: 6.6850, mae: 1.5986, huber: 1.2048, swd: 0.8987, target_std: 6.0731
    Epoch [9/50], Val Losses: mse: 13.0846, mae: 1.9775, huber: 1.5986, swd: 1.0866, target_std: 4.4502
    Epoch [9/50], Test Losses: mse: 11.9061, mae: 2.1888, huber: 1.7783, swd: 1.9259, target_std: 4.9747
      Epoch 9 composite train-obj: 1.204780
    Epoch [9/50], Test Losses: mse: 12.2374, mae: 2.2718, huber: 1.8608, swd: 2.2102, target_std: 4.9747
    Best round's Test MSE: 12.2374, MAE: 2.2718, SWD: 2.2102
    Best round's Validation MSE: 10.6497, MAE: 1.8332
    Best round's Test verification MSE : 12.2374, MAE: 2.2718, SWD: 2.2102
    Time taken: 74.98 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 15.1357, mae: 2.3828, huber: 1.9681, swd: 3.3063, target_std: 6.0994
    Epoch [1/50], Val Losses: mse: 10.4522, mae: 1.8500, huber: 1.4724, swd: 0.6801, target_std: 4.4502
    Epoch [1/50], Test Losses: mse: 16.1462, mae: 2.6389, huber: 2.2243, swd: 2.3562, target_std: 4.9747
      Epoch 1 composite train-obj: 1.968075
            Val objective improved inf → 1.4724, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.9167, mae: 2.0120, huber: 1.6048, swd: 2.1145, target_std: 6.1113
    Epoch [2/50], Val Losses: mse: 14.8107, mae: 2.2527, huber: 1.8645, swd: 3.5255, target_std: 4.4502
    Epoch [2/50], Test Losses: mse: 16.3178, mae: 2.7399, huber: 2.3198, swd: 4.5801, target_std: 4.9747
      Epoch 2 composite train-obj: 1.604796
            No improvement (1.8645), counter 1/5
    Epoch [3/50], Train Losses: mse: 9.4035, mae: 1.9599, huber: 1.5532, swd: 1.9950, target_std: 6.0750
    Epoch [3/50], Val Losses: mse: 11.3961, mae: 1.8722, huber: 1.4993, swd: 1.1692, target_std: 4.4502
    Epoch [3/50], Test Losses: mse: 11.9968, mae: 2.2184, huber: 1.8115, swd: 2.0760, target_std: 4.9747
      Epoch 3 composite train-obj: 1.553167
            No improvement (1.4993), counter 2/5
    Epoch [4/50], Train Losses: mse: 8.4120, mae: 1.8356, huber: 1.4328, swd: 1.6332, target_std: 6.0997
    Epoch [4/50], Val Losses: mse: 12.0131, mae: 1.9446, huber: 1.5647, swd: 1.3067, target_std: 4.4502
    Epoch [4/50], Test Losses: mse: 12.6780, mae: 2.3154, huber: 1.9044, swd: 2.4683, target_std: 4.9747
      Epoch 4 composite train-obj: 1.432810
            No improvement (1.5647), counter 3/5
    Epoch [5/50], Train Losses: mse: 8.1465, mae: 1.8032, huber: 1.4008, swd: 1.5077, target_std: 6.0838
    Epoch [5/50], Val Losses: mse: 13.5682, mae: 2.0474, huber: 1.6658, swd: 1.5420, target_std: 4.4502
    Epoch [5/50], Test Losses: mse: 12.1079, mae: 2.2659, huber: 1.8517, swd: 2.3569, target_std: 4.9747
      Epoch 5 composite train-obj: 1.400786
            No improvement (1.6658), counter 4/5
    Epoch [6/50], Train Losses: mse: 7.7103, mae: 1.7435, huber: 1.3434, swd: 1.3194, target_std: 6.0800
    Epoch [6/50], Val Losses: mse: 11.9561, mae: 1.9757, huber: 1.5934, swd: 1.9143, target_std: 4.4502
    Epoch [6/50], Test Losses: mse: 12.3295, mae: 2.3025, huber: 1.8881, swd: 3.3291, target_std: 4.9747
      Epoch 6 composite train-obj: 1.343394
    Epoch [6/50], Test Losses: mse: 16.1462, mae: 2.6389, huber: 2.2243, swd: 2.3562, target_std: 4.9747
    Best round's Test MSE: 16.1462, MAE: 2.6389, SWD: 2.3562
    Best round's Validation MSE: 10.4522, MAE: 1.8500
    Best round's Test verification MSE : 16.1462, MAE: 2.6389, SWD: 2.3562
    Time taken: 50.00 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth1_seq720_pred720_20250503_2054)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 13.3424 ± 1.9974
      mae: 2.3571 ± 0.2044
      huber: 1.9460 ± 0.2016
      swd: 2.2130 ± 0.1158
      target_std: 4.9747 ± 0.0000
      count: 3.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 10.3440 ± 0.3036
      mae: 1.8252 ± 0.0241
      huber: 1.4487 ± 0.0231
      swd: 0.8231 ± 0.1033
      target_std: 4.4502 ± 0.0000
      count: 3.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 217.49 seconds
    
    Experiment complete: PatchTST_etth1_seq720_pred720_20250503_2054
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 8
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.1130, mae: 1.8851, huber: 1.4959, swd: 2.3989, target_std: 6.1753
    Epoch [1/50], Val Losses: mse: 8.2379, mae: 1.6068, huber: 1.2381, swd: 1.0991, target_std: 4.4805
    Epoch [1/50], Test Losses: mse: 10.7805, mae: 1.8881, huber: 1.5065, swd: 1.8905, target_std: 4.8844
      Epoch 1 composite train-obj: 1.495875
            Val objective improved inf → 1.2381, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.2297, mae: 1.4972, huber: 1.1274, swd: 1.5223, target_std: 6.1755
    Epoch [2/50], Val Losses: mse: 7.4671, mae: 1.5155, huber: 1.1506, swd: 1.0536, target_std: 4.4805
    Epoch [2/50], Test Losses: mse: 10.4892, mae: 1.8362, huber: 1.4577, swd: 2.0109, target_std: 4.8844
      Epoch 2 composite train-obj: 1.127424
            Val objective improved 1.2381 → 1.1506, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.0390, mae: 1.4616, huber: 1.0948, swd: 1.4980, target_std: 6.1755
    Epoch [3/50], Val Losses: mse: 7.4217, mae: 1.5066, huber: 1.1418, swd: 1.0169, target_std: 4.4805
    Epoch [3/50], Test Losses: mse: 10.2591, mae: 1.7951, huber: 1.4182, swd: 1.7802, target_std: 4.8844
      Epoch 3 composite train-obj: 1.094791
            Val objective improved 1.1506 → 1.1418, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.8971, mae: 1.4375, huber: 1.0723, swd: 1.4550, target_std: 6.1757
    Epoch [4/50], Val Losses: mse: 7.3733, mae: 1.4995, huber: 1.1353, swd: 0.9745, target_std: 4.4805
    Epoch [4/50], Test Losses: mse: 10.2912, mae: 1.7864, huber: 1.4106, swd: 1.6814, target_std: 4.8844
      Epoch 4 composite train-obj: 1.072286
            Val objective improved 1.1418 → 1.1353, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.8907, mae: 1.4318, huber: 1.0675, swd: 1.4598, target_std: 6.1752
    Epoch [5/50], Val Losses: mse: 7.4721, mae: 1.5066, huber: 1.1428, swd: 1.1929, target_std: 4.4805
    Epoch [5/50], Test Losses: mse: 10.2690, mae: 1.7870, huber: 1.4099, swd: 1.8069, target_std: 4.8844
      Epoch 5 composite train-obj: 1.067462
            No improvement (1.1428), counter 1/5
    Epoch [6/50], Train Losses: mse: 5.8556, mae: 1.4274, huber: 1.0632, swd: 1.4489, target_std: 6.1758
    Epoch [6/50], Val Losses: mse: 7.7881, mae: 1.5397, huber: 1.1752, swd: 1.4052, target_std: 4.4805
    Epoch [6/50], Test Losses: mse: 10.2021, mae: 1.7743, huber: 1.3966, swd: 1.8089, target_std: 4.8844
      Epoch 6 composite train-obj: 1.063180
            No improvement (1.1752), counter 2/5
    Epoch [7/50], Train Losses: mse: 5.8567, mae: 1.4254, huber: 1.0613, swd: 1.4475, target_std: 6.1753
    Epoch [7/50], Val Losses: mse: 7.2649, mae: 1.4833, huber: 1.1189, swd: 1.0382, target_std: 4.4805
    Epoch [7/50], Test Losses: mse: 10.2715, mae: 1.7816, huber: 1.4054, swd: 1.7336, target_std: 4.8844
      Epoch 7 composite train-obj: 1.061316
            Val objective improved 1.1353 → 1.1189, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.7913, mae: 1.4139, huber: 1.0506, swd: 1.4270, target_std: 6.1755
    Epoch [8/50], Val Losses: mse: 6.9807, mae: 1.4532, huber: 1.0902, swd: 0.9199, target_std: 4.4805
    Epoch [8/50], Test Losses: mse: 10.1400, mae: 1.7674, huber: 1.3905, swd: 1.7605, target_std: 4.8844
      Epoch 8 composite train-obj: 1.050588
            Val objective improved 1.1189 → 1.0902, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 5.8139, mae: 1.4172, huber: 1.0534, swd: 1.4419, target_std: 6.1754
    Epoch [9/50], Val Losses: mse: 7.1224, mae: 1.4620, huber: 1.1001, swd: 1.0321, target_std: 4.4805
    Epoch [9/50], Test Losses: mse: 10.1902, mae: 1.7687, huber: 1.3932, swd: 1.8427, target_std: 4.8844
      Epoch 9 composite train-obj: 1.053442
            No improvement (1.1001), counter 1/5
    Epoch [10/50], Train Losses: mse: 5.7936, mae: 1.4138, huber: 1.0506, swd: 1.4249, target_std: 6.1754
    Epoch [10/50], Val Losses: mse: 7.0609, mae: 1.4626, huber: 1.0996, swd: 1.0681, target_std: 4.4805
    Epoch [10/50], Test Losses: mse: 10.0592, mae: 1.7530, huber: 1.3791, swd: 1.8343, target_std: 4.8844
      Epoch 10 composite train-obj: 1.050642
            No improvement (1.0996), counter 2/5
    Epoch [11/50], Train Losses: mse: 5.7713, mae: 1.4108, huber: 1.0473, swd: 1.4263, target_std: 6.1759
    Epoch [11/50], Val Losses: mse: 7.3681, mae: 1.4925, huber: 1.1289, swd: 1.0526, target_std: 4.4805
    Epoch [11/50], Test Losses: mse: 10.2502, mae: 1.7729, huber: 1.3966, swd: 1.6385, target_std: 4.8844
      Epoch 11 composite train-obj: 1.047278
            No improvement (1.1289), counter 3/5
    Epoch [12/50], Train Losses: mse: 5.8151, mae: 1.4147, huber: 1.0512, swd: 1.4328, target_std: 6.1753
    Epoch [12/50], Val Losses: mse: 7.1046, mae: 1.4699, huber: 1.1060, swd: 1.0435, target_std: 4.4805
    Epoch [12/50], Test Losses: mse: 10.1792, mae: 1.7676, huber: 1.3922, swd: 1.7867, target_std: 4.8844
      Epoch 12 composite train-obj: 1.051190
            No improvement (1.1060), counter 4/5
    Epoch [13/50], Train Losses: mse: 5.7844, mae: 1.4099, huber: 1.0469, swd: 1.4228, target_std: 6.1755
    Epoch [13/50], Val Losses: mse: 7.6650, mae: 1.5250, huber: 1.1612, swd: 1.3765, target_std: 4.4805
    Epoch [13/50], Test Losses: mse: 10.1972, mae: 1.7652, huber: 1.3908, swd: 1.7633, target_std: 4.8844
      Epoch 13 composite train-obj: 1.046884
    Epoch [13/50], Test Losses: mse: 10.1400, mae: 1.7674, huber: 1.3905, swd: 1.7605, target_std: 4.8844
    Best round's Test MSE: 10.1400, MAE: 1.7674, SWD: 1.7605
    Best round's Validation MSE: 6.9807, MAE: 1.4532
    Best round's Test verification MSE : 10.1400, MAE: 1.7674, SWD: 1.7605
    Time taken: 15.84 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.9781, mae: 1.8708, huber: 1.4817, swd: 2.1347, target_std: 6.1754
    Epoch [1/50], Val Losses: mse: 8.3206, mae: 1.6133, huber: 1.2432, swd: 1.0271, target_std: 4.4805
    Epoch [1/50], Test Losses: mse: 10.8097, mae: 1.8901, huber: 1.5073, swd: 1.6935, target_std: 4.8844
      Epoch 1 composite train-obj: 1.481700
            Val objective improved inf → 1.2432, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.2254, mae: 1.4960, huber: 1.1264, swd: 1.4592, target_std: 6.1754
    Epoch [2/50], Val Losses: mse: 7.6123, mae: 1.5238, huber: 1.1591, swd: 0.9604, target_std: 4.4805
    Epoch [2/50], Test Losses: mse: 10.3524, mae: 1.8099, huber: 1.4313, swd: 1.6494, target_std: 4.8844
      Epoch 2 composite train-obj: 1.126403
            Val objective improved 1.2432 → 1.1591, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.0215, mae: 1.4594, huber: 1.0925, swd: 1.4168, target_std: 6.1758
    Epoch [3/50], Val Losses: mse: 7.7610, mae: 1.5302, huber: 1.1663, swd: 0.9949, target_std: 4.4805
    Epoch [3/50], Test Losses: mse: 10.4113, mae: 1.8017, huber: 1.4256, swd: 1.5960, target_std: 4.8844
      Epoch 3 composite train-obj: 1.092470
            No improvement (1.1663), counter 1/5
    Epoch [4/50], Train Losses: mse: 5.9355, mae: 1.4436, huber: 1.0782, swd: 1.3984, target_std: 6.1759
    Epoch [4/50], Val Losses: mse: 7.2408, mae: 1.4808, huber: 1.1173, swd: 0.9643, target_std: 4.4805
    Epoch [4/50], Test Losses: mse: 10.2624, mae: 1.7880, huber: 1.4119, swd: 1.6785, target_std: 4.8844
      Epoch 4 composite train-obj: 1.078182
            Val objective improved 1.1591 → 1.1173, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.8878, mae: 1.4339, huber: 1.0688, swd: 1.3820, target_std: 6.1754
    Epoch [5/50], Val Losses: mse: 7.1841, mae: 1.4739, huber: 1.1117, swd: 1.0275, target_std: 4.4805
    Epoch [5/50], Test Losses: mse: 10.1922, mae: 1.7794, huber: 1.4027, swd: 1.7445, target_std: 4.8844
      Epoch 5 composite train-obj: 1.068824
            Val objective improved 1.1173 → 1.1117, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.8610, mae: 1.4263, huber: 1.0622, swd: 1.3804, target_std: 6.1760
    Epoch [6/50], Val Losses: mse: 7.4477, mae: 1.5024, huber: 1.1391, swd: 1.0863, target_std: 4.4805
    Epoch [6/50], Test Losses: mse: 10.2557, mae: 1.7784, huber: 1.4032, swd: 1.6662, target_std: 4.8844
      Epoch 6 composite train-obj: 1.062164
            No improvement (1.1391), counter 1/5
    Epoch [7/50], Train Losses: mse: 5.8345, mae: 1.4223, huber: 1.0588, swd: 1.3685, target_std: 6.1754
    Epoch [7/50], Val Losses: mse: 7.3863, mae: 1.4911, huber: 1.1285, swd: 1.1457, target_std: 4.4805
    Epoch [7/50], Test Losses: mse: 10.1812, mae: 1.7665, huber: 1.3916, swd: 1.6826, target_std: 4.8844
      Epoch 7 composite train-obj: 1.058777
            No improvement (1.1285), counter 2/5
    Epoch [8/50], Train Losses: mse: 5.8092, mae: 1.4159, huber: 1.0528, swd: 1.3661, target_std: 6.1758
    Epoch [8/50], Val Losses: mse: 7.3740, mae: 1.4906, huber: 1.1253, swd: 1.0382, target_std: 4.4805
    Epoch [8/50], Test Losses: mse: 10.1504, mae: 1.7680, huber: 1.3914, swd: 1.6452, target_std: 4.8844
      Epoch 8 composite train-obj: 1.052764
            No improvement (1.1253), counter 3/5
    Epoch [9/50], Train Losses: mse: 5.8120, mae: 1.4176, huber: 1.0538, swd: 1.3686, target_std: 6.1756
    Epoch [9/50], Val Losses: mse: 7.4329, mae: 1.5022, huber: 1.1387, swd: 1.1940, target_std: 4.4805
    Epoch [9/50], Test Losses: mse: 10.1655, mae: 1.7731, huber: 1.3953, swd: 1.7414, target_std: 4.8844
      Epoch 9 composite train-obj: 1.053834
            No improvement (1.1387), counter 4/5
    Epoch [10/50], Train Losses: mse: 5.7917, mae: 1.4141, huber: 1.0506, swd: 1.3718, target_std: 6.1755
    Epoch [10/50], Val Losses: mse: 7.5113, mae: 1.5106, huber: 1.1460, swd: 1.1941, target_std: 4.4805
    Epoch [10/50], Test Losses: mse: 10.1621, mae: 1.7618, huber: 1.3872, swd: 1.6384, target_std: 4.8844
      Epoch 10 composite train-obj: 1.050580
    Epoch [10/50], Test Losses: mse: 10.1922, mae: 1.7794, huber: 1.4027, swd: 1.7445, target_std: 4.8844
    Best round's Test MSE: 10.1922, MAE: 1.7794, SWD: 1.7445
    Best round's Validation MSE: 7.1841, MAE: 1.4739
    Best round's Test verification MSE : 10.1922, MAE: 1.7794, SWD: 1.7445
    Time taken: 11.69 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.9657, mae: 1.8656, huber: 1.4768, swd: 1.9764, target_std: 6.1760
    Epoch [1/50], Val Losses: mse: 8.6750, mae: 1.6469, huber: 1.2773, swd: 1.2433, target_std: 4.4805
    Epoch [1/50], Test Losses: mse: 10.7297, mae: 1.8769, huber: 1.4959, swd: 1.7625, target_std: 4.8844
      Epoch 1 composite train-obj: 1.476829
            Val objective improved inf → 1.2773, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.2430, mae: 1.4992, huber: 1.1294, swd: 1.4027, target_std: 6.1756
    Epoch [2/50], Val Losses: mse: 7.8143, mae: 1.5499, huber: 1.1840, swd: 1.0846, target_std: 4.4805
    Epoch [2/50], Test Losses: mse: 10.4287, mae: 1.8200, huber: 1.4408, swd: 1.7029, target_std: 4.8844
      Epoch 2 composite train-obj: 1.129407
            Val objective improved 1.2773 → 1.1840, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.9911, mae: 1.4562, huber: 1.0897, swd: 1.3559, target_std: 6.1757
    Epoch [3/50], Val Losses: mse: 7.5513, mae: 1.5169, huber: 1.1531, swd: 0.9642, target_std: 4.4805
    Epoch [3/50], Test Losses: mse: 10.3230, mae: 1.8006, huber: 1.4225, swd: 1.6273, target_std: 4.8844
      Epoch 3 composite train-obj: 1.089687
            Val objective improved 1.1840 → 1.1531, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.9204, mae: 1.4419, huber: 1.0762, swd: 1.3422, target_std: 6.1754
    Epoch [4/50], Val Losses: mse: 7.4545, mae: 1.5121, huber: 1.1476, swd: 1.2256, target_std: 4.4805
    Epoch [4/50], Test Losses: mse: 10.2745, mae: 1.7950, huber: 1.4162, swd: 1.8225, target_std: 4.8844
      Epoch 4 composite train-obj: 1.076201
            Val objective improved 1.1531 → 1.1476, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.8855, mae: 1.4323, huber: 1.0675, swd: 1.3337, target_std: 6.1757
    Epoch [5/50], Val Losses: mse: 7.2294, mae: 1.4794, huber: 1.1165, swd: 0.9296, target_std: 4.4805
    Epoch [5/50], Test Losses: mse: 10.2156, mae: 1.7761, huber: 1.3998, swd: 1.6247, target_std: 4.8844
      Epoch 5 composite train-obj: 1.067499
            Val objective improved 1.1476 → 1.1165, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.8464, mae: 1.4251, huber: 1.0609, swd: 1.3195, target_std: 6.1754
    Epoch [6/50], Val Losses: mse: 7.3232, mae: 1.4831, huber: 1.1209, swd: 0.9765, target_std: 4.4805
    Epoch [6/50], Test Losses: mse: 10.1781, mae: 1.7651, huber: 1.3907, swd: 1.6015, target_std: 4.8844
      Epoch 6 composite train-obj: 1.060945
            No improvement (1.1209), counter 1/5
    Epoch [7/50], Train Losses: mse: 5.8388, mae: 1.4221, huber: 1.0584, swd: 1.3201, target_std: 6.1756
    Epoch [7/50], Val Losses: mse: 7.3073, mae: 1.4909, huber: 1.1261, swd: 0.9551, target_std: 4.4805
    Epoch [7/50], Test Losses: mse: 10.2117, mae: 1.7748, huber: 1.3983, swd: 1.6009, target_std: 4.8844
      Epoch 7 composite train-obj: 1.058353
            No improvement (1.1261), counter 2/5
    Epoch [8/50], Train Losses: mse: 5.8350, mae: 1.4218, huber: 1.0578, swd: 1.3272, target_std: 6.1757
    Epoch [8/50], Val Losses: mse: 7.3248, mae: 1.4883, huber: 1.1249, swd: 0.8967, target_std: 4.4805
    Epoch [8/50], Test Losses: mse: 10.2863, mae: 1.7793, huber: 1.4028, swd: 1.5380, target_std: 4.8844
      Epoch 8 composite train-obj: 1.057799
            No improvement (1.1249), counter 3/5
    Epoch [9/50], Train Losses: mse: 5.7676, mae: 1.4102, huber: 1.0473, swd: 1.2959, target_std: 6.1756
    Epoch [9/50], Val Losses: mse: 7.5408, mae: 1.5178, huber: 1.1538, swd: 1.1219, target_std: 4.4805
    Epoch [9/50], Test Losses: mse: 10.2068, mae: 1.7722, huber: 1.3960, swd: 1.6544, target_std: 4.8844
      Epoch 9 composite train-obj: 1.047253
            No improvement (1.1538), counter 4/5
    Epoch [10/50], Train Losses: mse: 5.8055, mae: 1.4163, huber: 1.0527, swd: 1.3094, target_std: 6.1757
    Epoch [10/50], Val Losses: mse: 7.2748, mae: 1.4780, huber: 1.1155, swd: 0.9064, target_std: 4.4805
    Epoch [10/50], Test Losses: mse: 10.1203, mae: 1.7549, huber: 1.3802, swd: 1.5286, target_std: 4.8844
      Epoch 10 composite train-obj: 1.052717
            Val objective improved 1.1165 → 1.1155, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 5.8182, mae: 1.4162, huber: 1.0530, swd: 1.3171, target_std: 6.1756
    Epoch [11/50], Val Losses: mse: 7.1363, mae: 1.4682, huber: 1.1054, swd: 0.9329, target_std: 4.4805
    Epoch [11/50], Test Losses: mse: 10.1334, mae: 1.7613, huber: 1.3861, swd: 1.6718, target_std: 4.8844
      Epoch 11 composite train-obj: 1.053006
            Val objective improved 1.1155 → 1.1054, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 5.7711, mae: 1.4095, huber: 1.0461, swd: 1.3034, target_std: 6.1757
    Epoch [12/50], Val Losses: mse: 7.4422, mae: 1.5009, huber: 1.1375, swd: 1.0917, target_std: 4.4805
    Epoch [12/50], Test Losses: mse: 10.1695, mae: 1.7602, huber: 1.3841, swd: 1.6247, target_std: 4.8844
      Epoch 12 composite train-obj: 1.046055
            No improvement (1.1375), counter 1/5
    Epoch [13/50], Train Losses: mse: 5.7890, mae: 1.4118, huber: 1.0485, swd: 1.3229, target_std: 6.1757
    Epoch [13/50], Val Losses: mse: 7.1256, mae: 1.4710, huber: 1.1071, swd: 0.9783, target_std: 4.4805
    Epoch [13/50], Test Losses: mse: 10.0949, mae: 1.7560, huber: 1.3814, swd: 1.6265, target_std: 4.8844
      Epoch 13 composite train-obj: 1.048478
            No improvement (1.1071), counter 2/5
    Epoch [14/50], Train Losses: mse: 5.7621, mae: 1.4074, huber: 1.0446, swd: 1.2947, target_std: 6.1756
    Epoch [14/50], Val Losses: mse: 7.1667, mae: 1.4713, huber: 1.1091, swd: 1.0730, target_std: 4.4805
    Epoch [14/50], Test Losses: mse: 10.2172, mae: 1.7724, huber: 1.3956, swd: 1.7530, target_std: 4.8844
      Epoch 14 composite train-obj: 1.044590
            No improvement (1.1091), counter 3/5
    Epoch [15/50], Train Losses: mse: 5.7488, mae: 1.4043, huber: 1.0413, swd: 1.2950, target_std: 6.1756
    Epoch [15/50], Val Losses: mse: 7.3691, mae: 1.4930, huber: 1.1299, swd: 1.1949, target_std: 4.4805
    Epoch [15/50], Test Losses: mse: 10.1371, mae: 1.7570, huber: 1.3825, swd: 1.7053, target_std: 4.8844
      Epoch 15 composite train-obj: 1.041321
            No improvement (1.1299), counter 4/5
    Epoch [16/50], Train Losses: mse: 5.8038, mae: 1.4113, huber: 1.0486, swd: 1.3173, target_std: 6.1756
    Epoch [16/50], Val Losses: mse: 7.0809, mae: 1.4722, huber: 1.1071, swd: 0.9929, target_std: 4.4805
    Epoch [16/50], Test Losses: mse: 10.1555, mae: 1.7649, huber: 1.3895, swd: 1.6736, target_std: 4.8844
      Epoch 16 composite train-obj: 1.048644
    Epoch [16/50], Test Losses: mse: 10.1334, mae: 1.7613, huber: 1.3861, swd: 1.6718, target_std: 4.8844
    Best round's Test MSE: 10.1334, MAE: 1.7613, SWD: 1.6718
    Best round's Validation MSE: 7.1363, MAE: 1.4682
    Best round's Test verification MSE : 10.1334, MAE: 1.7613, SWD: 1.6718
    Time taken: 19.15 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq720_pred96_20250503_1951)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.1552 ± 0.0263
      mae: 1.7694 ± 0.0075
      huber: 1.3931 ± 0.0070
      swd: 1.7256 ± 0.0386
      target_std: 4.8844 ± 0.0000
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.1003 ± 0.0868
      mae: 1.4651 ± 0.0087
      huber: 1.1024 ± 0.0091
      swd: 0.9601 ± 0.0480
      target_std: 4.4805 ± 0.0000
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 46.74 seconds
    
    Experiment complete: DLinear_etth1_seq720_pred96_20250503_1951
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 89
    Validation Batches: 7
    Test Batches: 21
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.2116, mae: 1.9336, huber: 1.5390, swd: 2.1330, target_std: 6.1713
    Epoch [1/50], Val Losses: mse: 10.8437, mae: 1.8570, huber: 1.4790, swd: 1.8069, target_std: 4.4590
    Epoch [1/50], Test Losses: mse: 11.1322, mae: 1.9853, huber: 1.5943, swd: 2.0714, target_std: 4.8938
      Epoch 1 composite train-obj: 1.539042
            Val objective improved inf → 1.4790, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.8238, mae: 1.5994, huber: 1.2208, swd: 1.4803, target_std: 6.1611
    Epoch [2/50], Val Losses: mse: 10.1295, mae: 1.7799, huber: 1.4023, swd: 1.2080, target_std: 4.4590
    Epoch [2/50], Test Losses: mse: 10.8449, mae: 1.9221, huber: 1.5350, swd: 1.5886, target_std: 4.8938
      Epoch 2 composite train-obj: 1.220757
            Val objective improved 1.4790 → 1.4023, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.6246, mae: 1.5649, huber: 1.1888, swd: 1.4587, target_std: 6.1567
    Epoch [3/50], Val Losses: mse: 9.8140, mae: 1.7332, huber: 1.3597, swd: 1.3159, target_std: 4.4590
    Epoch [3/50], Test Losses: mse: 10.6773, mae: 1.8835, huber: 1.4988, swd: 1.6260, target_std: 4.8938
      Epoch 3 composite train-obj: 1.188813
            Val objective improved 1.4023 → 1.3597, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.5672, mae: 1.5516, huber: 1.1766, swd: 1.4485, target_std: 6.1700
    Epoch [4/50], Val Losses: mse: 9.8931, mae: 1.7465, huber: 1.3713, swd: 1.3071, target_std: 4.4590
    Epoch [4/50], Test Losses: mse: 10.7169, mae: 1.8909, huber: 1.5061, swd: 1.6523, target_std: 4.8938
      Epoch 4 composite train-obj: 1.176563
            No improvement (1.3713), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.5002, mae: 1.5406, huber: 1.1664, swd: 1.4372, target_std: 6.1644
    Epoch [5/50], Val Losses: mse: 9.4425, mae: 1.6990, huber: 1.3262, swd: 1.1833, target_std: 4.4590
    Epoch [5/50], Test Losses: mse: 10.6525, mae: 1.8854, huber: 1.4950, swd: 1.6315, target_std: 4.8938
      Epoch 5 composite train-obj: 1.166429
            Val objective improved 1.3597 → 1.3262, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.5200, mae: 1.5405, huber: 1.1665, swd: 1.4560, target_std: 6.1693
    Epoch [6/50], Val Losses: mse: 9.4647, mae: 1.7020, huber: 1.3292, swd: 1.2467, target_std: 4.4590
    Epoch [6/50], Test Losses: mse: 10.6554, mae: 1.8797, huber: 1.4939, swd: 1.6764, target_std: 4.8938
      Epoch 6 composite train-obj: 1.166503
            No improvement (1.3292), counter 1/5
    Epoch [7/50], Train Losses: mse: 6.4856, mae: 1.5345, huber: 1.1607, swd: 1.4282, target_std: 6.1579
    Epoch [7/50], Val Losses: mse: 9.5461, mae: 1.7122, huber: 1.3372, swd: 1.2530, target_std: 4.4590
    Epoch [7/50], Test Losses: mse: 10.6567, mae: 1.8771, huber: 1.4918, swd: 1.7176, target_std: 4.8938
      Epoch 7 composite train-obj: 1.160735
            No improvement (1.3372), counter 2/5
    Epoch [8/50], Train Losses: mse: 6.4263, mae: 1.5263, huber: 1.1529, swd: 1.4104, target_std: 6.1599
    Epoch [8/50], Val Losses: mse: 9.9972, mae: 1.7429, huber: 1.3697, swd: 1.4114, target_std: 4.4590
    Epoch [8/50], Test Losses: mse: 10.6502, mae: 1.8714, huber: 1.4860, swd: 1.6027, target_std: 4.8938
      Epoch 8 composite train-obj: 1.152936
            No improvement (1.3697), counter 3/5
    Epoch [9/50], Train Losses: mse: 6.4615, mae: 1.5289, huber: 1.1557, swd: 1.4321, target_std: 6.1612
    Epoch [9/50], Val Losses: mse: 9.9635, mae: 1.7382, huber: 1.3653, swd: 1.4509, target_std: 4.4590
    Epoch [9/50], Test Losses: mse: 10.6626, mae: 1.8670, huber: 1.4809, swd: 1.6905, target_std: 4.8938
      Epoch 9 composite train-obj: 1.155671
            No improvement (1.3653), counter 4/5
    Epoch [10/50], Train Losses: mse: 6.4295, mae: 1.5249, huber: 1.1518, swd: 1.4148, target_std: 6.1673
    Epoch [10/50], Val Losses: mse: 10.2501, mae: 1.7630, huber: 1.3884, swd: 1.5719, target_std: 4.4590
    Epoch [10/50], Test Losses: mse: 10.6874, mae: 1.8720, huber: 1.4860, swd: 1.6955, target_std: 4.8938
      Epoch 10 composite train-obj: 1.151800
    Epoch [10/50], Test Losses: mse: 10.6525, mae: 1.8854, huber: 1.4950, swd: 1.6315, target_std: 4.8938
    Best round's Test MSE: 10.6525, MAE: 1.8854, SWD: 1.6315
    Best round's Validation MSE: 9.4425, MAE: 1.6990
    Best round's Test verification MSE : 10.6525, MAE: 1.8854, SWD: 1.6315
    Time taken: 11.94 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.5413, mae: 1.9438, huber: 1.5492, swd: 2.2138, target_std: 6.1668
    Epoch [1/50], Val Losses: mse: 10.2503, mae: 1.7760, huber: 1.4005, swd: 1.1292, target_std: 4.4590
    Epoch [1/50], Test Losses: mse: 11.0389, mae: 1.9584, huber: 1.5680, swd: 1.5695, target_std: 4.8938
      Epoch 1 composite train-obj: 1.549197
            Val objective improved inf → 1.4005, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.8022, mae: 1.5949, huber: 1.2166, swd: 1.4664, target_std: 6.1703
    Epoch [2/50], Val Losses: mse: 10.1768, mae: 1.7633, huber: 1.3879, swd: 1.3070, target_std: 4.4590
    Epoch [2/50], Test Losses: mse: 10.7133, mae: 1.8975, huber: 1.5116, swd: 1.4633, target_std: 4.8938
      Epoch 2 composite train-obj: 1.216612
            Val objective improved 1.4005 → 1.3879, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.6505, mae: 1.5665, huber: 1.1905, swd: 1.4440, target_std: 6.1598
    Epoch [3/50], Val Losses: mse: 9.8212, mae: 1.7398, huber: 1.3655, swd: 1.4477, target_std: 4.4590
    Epoch [3/50], Test Losses: mse: 10.8767, mae: 1.9197, huber: 1.5339, swd: 1.8779, target_std: 4.8938
      Epoch 3 composite train-obj: 1.190477
            Val objective improved 1.3879 → 1.3655, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.5823, mae: 1.5531, huber: 1.1783, swd: 1.4363, target_std: 6.1620
    Epoch [4/50], Val Losses: mse: 10.1792, mae: 1.7699, huber: 1.3957, swd: 1.7348, target_std: 4.4590
    Epoch [4/50], Test Losses: mse: 10.7458, mae: 1.8918, huber: 1.5082, swd: 1.8052, target_std: 4.8938
      Epoch 4 composite train-obj: 1.178328
            No improvement (1.3957), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.5213, mae: 1.5435, huber: 1.1692, swd: 1.4216, target_std: 6.1599
    Epoch [5/50], Val Losses: mse: 9.7455, mae: 1.7310, huber: 1.3572, swd: 1.5154, target_std: 4.4590
    Epoch [5/50], Test Losses: mse: 10.6463, mae: 1.8802, huber: 1.4962, swd: 1.7118, target_std: 4.8938
      Epoch 5 composite train-obj: 1.169217
            Val objective improved 1.3655 → 1.3572, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.5269, mae: 1.5399, huber: 1.1662, swd: 1.4316, target_std: 6.1648
    Epoch [6/50], Val Losses: mse: 9.9867, mae: 1.7444, huber: 1.3711, swd: 1.5526, target_std: 4.4590
    Epoch [6/50], Test Losses: mse: 10.7664, mae: 1.8882, huber: 1.5042, swd: 1.7843, target_std: 4.8938
      Epoch 6 composite train-obj: 1.166198
            No improvement (1.3711), counter 1/5
    Epoch [7/50], Train Losses: mse: 6.4864, mae: 1.5349, huber: 1.1614, swd: 1.4298, target_std: 6.1620
    Epoch [7/50], Val Losses: mse: 9.9257, mae: 1.7479, huber: 1.3737, swd: 1.5755, target_std: 4.4590
    Epoch [7/50], Test Losses: mse: 10.6888, mae: 1.8760, huber: 1.4914, swd: 1.6568, target_std: 4.8938
      Epoch 7 composite train-obj: 1.161379
            No improvement (1.3737), counter 2/5
    Epoch [8/50], Train Losses: mse: 6.4618, mae: 1.5328, huber: 1.1589, swd: 1.4273, target_std: 6.1618
    Epoch [8/50], Val Losses: mse: 9.5116, mae: 1.7022, huber: 1.3305, swd: 1.1659, target_std: 4.4590
    Epoch [8/50], Test Losses: mse: 10.6430, mae: 1.8689, huber: 1.4823, swd: 1.4734, target_std: 4.8938
      Epoch 8 composite train-obj: 1.158888
            Val objective improved 1.3572 → 1.3305, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 6.4204, mae: 1.5251, huber: 1.1518, swd: 1.3955, target_std: 6.1588
    Epoch [9/50], Val Losses: mse: 9.4044, mae: 1.6948, huber: 1.3225, swd: 1.2194, target_std: 4.4590
    Epoch [9/50], Test Losses: mse: 10.5721, mae: 1.8642, huber: 1.4801, swd: 1.6094, target_std: 4.8938
      Epoch 9 composite train-obj: 1.151774
            Val objective improved 1.3305 → 1.3225, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 6.4447, mae: 1.5273, huber: 1.1538, swd: 1.3987, target_std: 6.1708
    Epoch [10/50], Val Losses: mse: 10.0603, mae: 1.7670, huber: 1.3896, swd: 1.7177, target_std: 4.4590
    Epoch [10/50], Test Losses: mse: 10.6407, mae: 1.8866, huber: 1.4987, swd: 1.7720, target_std: 4.8938
      Epoch 10 composite train-obj: 1.153791
            No improvement (1.3896), counter 1/5
    Epoch [11/50], Train Losses: mse: 6.4242, mae: 1.5257, huber: 1.1518, swd: 1.3914, target_std: 6.1578
    Epoch [11/50], Val Losses: mse: 9.1176, mae: 1.6871, huber: 1.3103, swd: 1.1641, target_std: 4.4590
    Epoch [11/50], Test Losses: mse: 10.7910, mae: 1.8874, huber: 1.5023, swd: 1.7114, target_std: 4.8938
      Epoch 11 composite train-obj: 1.151775
            Val objective improved 1.3225 → 1.3103, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 6.4317, mae: 1.5250, huber: 1.1518, swd: 1.4059, target_std: 6.1690
    Epoch [12/50], Val Losses: mse: 9.7379, mae: 1.7187, huber: 1.3465, swd: 1.3287, target_std: 4.4590
    Epoch [12/50], Test Losses: mse: 10.6085, mae: 1.8675, huber: 1.4816, swd: 1.5175, target_std: 4.8938
      Epoch 12 composite train-obj: 1.151754
            No improvement (1.3465), counter 1/5
    Epoch [13/50], Train Losses: mse: 6.4232, mae: 1.5217, huber: 1.1490, swd: 1.3896, target_std: 6.1595
    Epoch [13/50], Val Losses: mse: 9.9348, mae: 1.7472, huber: 1.3726, swd: 1.6415, target_std: 4.4590
    Epoch [13/50], Test Losses: mse: 10.6359, mae: 1.8685, huber: 1.4836, swd: 1.7242, target_std: 4.8938
      Epoch 13 composite train-obj: 1.148954
            No improvement (1.3726), counter 2/5
    Epoch [14/50], Train Losses: mse: 6.3854, mae: 1.5176, huber: 1.1450, swd: 1.3848, target_std: 6.1700
    Epoch [14/50], Val Losses: mse: 10.1256, mae: 1.7570, huber: 1.3840, swd: 1.7622, target_std: 4.4590
    Epoch [14/50], Test Losses: mse: 10.6038, mae: 1.8584, huber: 1.4761, swd: 1.7087, target_std: 4.8938
      Epoch 14 composite train-obj: 1.144999
            No improvement (1.3840), counter 3/5
    Epoch [15/50], Train Losses: mse: 6.3861, mae: 1.5165, huber: 1.1439, swd: 1.3685, target_std: 6.1654
    Epoch [15/50], Val Losses: mse: 10.4103, mae: 1.7892, huber: 1.4144, swd: 1.8743, target_std: 4.4590
    Epoch [15/50], Test Losses: mse: 10.6364, mae: 1.8743, huber: 1.4887, swd: 1.7246, target_std: 4.8938
      Epoch 15 composite train-obj: 1.143901
            No improvement (1.4144), counter 4/5
    Epoch [16/50], Train Losses: mse: 6.3882, mae: 1.5183, huber: 1.1454, swd: 1.3834, target_std: 6.1611
    Epoch [16/50], Val Losses: mse: 9.9389, mae: 1.7399, huber: 1.3664, swd: 1.3006, target_std: 4.4590
    Epoch [16/50], Test Losses: mse: 10.7855, mae: 1.8957, huber: 1.5092, swd: 1.5058, target_std: 4.8938
      Epoch 16 composite train-obj: 1.145395
    Epoch [16/50], Test Losses: mse: 10.7910, mae: 1.8874, huber: 1.5023, swd: 1.7114, target_std: 4.8938
    Best round's Test MSE: 10.7910, MAE: 1.8874, SWD: 1.7114
    Best round's Validation MSE: 9.1176, MAE: 1.6871
    Best round's Test verification MSE : 10.7910, MAE: 1.8874, SWD: 1.7114
    Time taken: 18.93 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.6099, mae: 1.9523, huber: 1.5579, swd: 1.9791, target_std: 6.1675
    Epoch [1/50], Val Losses: mse: 10.8254, mae: 1.8309, huber: 1.4549, swd: 1.3678, target_std: 4.4590
    Epoch [1/50], Test Losses: mse: 10.9754, mae: 1.9517, huber: 1.5632, swd: 1.5826, target_std: 4.8938
      Epoch 1 composite train-obj: 1.557900
            Val objective improved inf → 1.4549, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.8334, mae: 1.5988, huber: 1.2205, swd: 1.3303, target_std: 6.1540
    Epoch [2/50], Val Losses: mse: 10.4404, mae: 1.7874, huber: 1.4128, swd: 1.1980, target_std: 4.4590
    Epoch [2/50], Test Losses: mse: 10.8096, mae: 1.9011, huber: 1.5150, swd: 1.3729, target_std: 4.8938
      Epoch 2 composite train-obj: 1.220493
            Val objective improved 1.4549 → 1.4128, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.6066, mae: 1.5609, huber: 1.1852, swd: 1.3019, target_std: 6.1620
    Epoch [3/50], Val Losses: mse: 10.1039, mae: 1.7599, huber: 1.3865, swd: 1.2246, target_std: 4.4590
    Epoch [3/50], Test Losses: mse: 10.7517, mae: 1.8923, huber: 1.5077, swd: 1.4965, target_std: 4.8938
      Epoch 3 composite train-obj: 1.185241
            Val objective improved 1.4128 → 1.3865, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.5596, mae: 1.5498, huber: 1.1753, swd: 1.2944, target_std: 6.1656
    Epoch [4/50], Val Losses: mse: 9.7068, mae: 1.7251, huber: 1.3514, swd: 1.0678, target_std: 4.4590
    Epoch [4/50], Test Losses: mse: 10.6976, mae: 1.8919, huber: 1.5022, swd: 1.4127, target_std: 4.8938
      Epoch 4 composite train-obj: 1.175309
            Val objective improved 1.3865 → 1.3514, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.5141, mae: 1.5421, huber: 1.1676, swd: 1.2920, target_std: 6.1578
    Epoch [5/50], Val Losses: mse: 9.3782, mae: 1.6915, huber: 1.3192, swd: 1.1186, target_std: 4.4590
    Epoch [5/50], Test Losses: mse: 10.7127, mae: 1.8846, huber: 1.5004, swd: 1.6786, target_std: 4.8938
      Epoch 5 composite train-obj: 1.167624
            Val objective improved 1.3514 → 1.3192, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.4982, mae: 1.5378, huber: 1.1642, swd: 1.2997, target_std: 6.1716
    Epoch [6/50], Val Losses: mse: 9.9818, mae: 1.7432, huber: 1.3695, swd: 1.2650, target_std: 4.4590
    Epoch [6/50], Test Losses: mse: 10.6425, mae: 1.8726, huber: 1.4874, swd: 1.4619, target_std: 4.8938
      Epoch 6 composite train-obj: 1.164218
            No improvement (1.3695), counter 1/5
    Epoch [7/50], Train Losses: mse: 6.5015, mae: 1.5368, huber: 1.1631, swd: 1.2843, target_std: 6.1632
    Epoch [7/50], Val Losses: mse: 9.6298, mae: 1.7153, huber: 1.3416, swd: 1.1794, target_std: 4.4590
    Epoch [7/50], Test Losses: mse: 10.6324, mae: 1.8765, huber: 1.4922, swd: 1.6133, target_std: 4.8938
      Epoch 7 composite train-obj: 1.163120
            No improvement (1.3416), counter 2/5
    Epoch [8/50], Train Losses: mse: 6.4698, mae: 1.5316, huber: 1.1580, swd: 1.2861, target_std: 6.1639
    Epoch [8/50], Val Losses: mse: 10.0408, mae: 1.7372, huber: 1.3646, swd: 1.2874, target_std: 4.4590
    Epoch [8/50], Test Losses: mse: 10.6293, mae: 1.8714, huber: 1.4829, swd: 1.4913, target_std: 4.8938
      Epoch 8 composite train-obj: 1.158026
            No improvement (1.3646), counter 3/5
    Epoch [9/50], Train Losses: mse: 6.4380, mae: 1.5261, huber: 1.1528, swd: 1.2648, target_std: 6.1666
    Epoch [9/50], Val Losses: mse: 9.7980, mae: 1.7242, huber: 1.3524, swd: 1.2233, target_std: 4.4590
    Epoch [9/50], Test Losses: mse: 10.6152, mae: 1.8654, huber: 1.4795, swd: 1.4296, target_std: 4.8938
      Epoch 9 composite train-obj: 1.152787
            No improvement (1.3524), counter 4/5
    Epoch [10/50], Train Losses: mse: 6.4060, mae: 1.5217, huber: 1.1487, swd: 1.2639, target_std: 6.1642
    Epoch [10/50], Val Losses: mse: 9.7413, mae: 1.7397, huber: 1.3619, swd: 1.1944, target_std: 4.4590
    Epoch [10/50], Test Losses: mse: 10.6992, mae: 1.8743, huber: 1.4909, swd: 1.4950, target_std: 4.8938
      Epoch 10 composite train-obj: 1.148715
    Epoch [10/50], Test Losses: mse: 10.7127, mae: 1.8846, huber: 1.5004, swd: 1.6786, target_std: 4.8938
    Best round's Test MSE: 10.7127, MAE: 1.8846, SWD: 1.6786
    Best round's Validation MSE: 9.3782, MAE: 1.6915
    Best round's Test verification MSE : 10.7127, MAE: 1.8846, SWD: 1.6786
    Time taken: 12.54 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq720_pred196_20250503_1952)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 10.7188 ± 0.0567
      mae: 1.8858 ± 0.0012
      huber: 1.4992 ± 0.0031
      swd: 1.6738 ± 0.0328
      target_std: 4.8938 ± 0.0000
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.3128 ± 0.1404
      mae: 1.6925 ± 0.0049
      huber: 1.3186 ± 0.0065
      swd: 1.1553 ± 0.0271
      target_std: 4.4590 ± 0.0000
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 43.51 seconds
    
    Experiment complete: DLinear_etth1_seq720_pred196_20250503_1952
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 88
    Validation Batches: 6
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.8549, mae: 2.0096, huber: 1.6101, swd: 2.1639, target_std: 6.1435
    Epoch [1/50], Val Losses: mse: 11.7888, mae: 1.9278, huber: 1.5430, swd: 1.5345, target_std: 4.4388
    Epoch [1/50], Test Losses: mse: 12.0481, mae: 2.1208, huber: 1.7218, swd: 1.8711, target_std: 4.9300
      Epoch 1 composite train-obj: 1.610120
            Val objective improved inf → 1.5430, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.4004, mae: 1.6925, huber: 1.3066, swd: 1.5199, target_std: 6.1430
    Epoch [2/50], Val Losses: mse: 10.9919, mae: 1.8274, huber: 1.4488, swd: 1.2059, target_std: 4.4388
    Epoch [2/50], Test Losses: mse: 11.4702, mae: 2.0287, huber: 1.6364, swd: 1.5930, target_std: 4.9300
      Epoch 2 composite train-obj: 1.306640
            Val objective improved 1.5430 → 1.4488, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.2172, mae: 1.6642, huber: 1.2807, swd: 1.5488, target_std: 6.1378
    Epoch [3/50], Val Losses: mse: 10.4863, mae: 1.7954, huber: 1.4134, swd: 0.9656, target_std: 4.4388
    Epoch [3/50], Test Losses: mse: 11.5607, mae: 2.0396, huber: 1.6450, swd: 1.6859, target_std: 4.9300
      Epoch 3 composite train-obj: 1.280674
            Val objective improved 1.4488 → 1.4134, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.0954, mae: 1.6481, huber: 1.2648, swd: 1.5021, target_std: 6.1465
    Epoch [4/50], Val Losses: mse: 11.1837, mae: 1.8625, huber: 1.4791, swd: 1.1559, target_std: 4.4388
    Epoch [4/50], Test Losses: mse: 11.9037, mae: 2.0845, huber: 1.6873, swd: 1.6963, target_std: 4.9300
      Epoch 4 composite train-obj: 1.264795
            No improvement (1.4791), counter 1/5
    Epoch [5/50], Train Losses: mse: 7.1348, mae: 1.6467, huber: 1.2642, swd: 1.4664, target_std: 6.1371
    Epoch [5/50], Val Losses: mse: 10.6769, mae: 1.7984, huber: 1.4198, swd: 1.1701, target_std: 4.4388
    Epoch [5/50], Test Losses: mse: 11.5013, mae: 2.0344, huber: 1.6335, swd: 1.7660, target_std: 4.9300
      Epoch 5 composite train-obj: 1.264166
            No improvement (1.4198), counter 2/5
    Epoch [6/50], Train Losses: mse: 7.0357, mae: 1.6339, huber: 1.2520, swd: 1.4871, target_std: 6.1385
    Epoch [6/50], Val Losses: mse: 11.2520, mae: 1.8428, huber: 1.4629, swd: 1.1241, target_std: 4.4388
    Epoch [6/50], Test Losses: mse: 11.3651, mae: 2.0071, huber: 1.6118, swd: 1.5258, target_std: 4.9300
      Epoch 6 composite train-obj: 1.252048
            No improvement (1.4629), counter 3/5
    Epoch [7/50], Train Losses: mse: 6.9980, mae: 1.6278, huber: 1.2459, swd: 1.4659, target_std: 6.1425
    Epoch [7/50], Val Losses: mse: 10.7391, mae: 1.8057, huber: 1.4253, swd: 0.7963, target_std: 4.4388
    Epoch [7/50], Test Losses: mse: 11.5675, mae: 2.0370, huber: 1.6398, swd: 1.4676, target_std: 4.9300
      Epoch 7 composite train-obj: 1.245937
            No improvement (1.4253), counter 4/5
    Epoch [8/50], Train Losses: mse: 7.0539, mae: 1.6323, huber: 1.2505, swd: 1.4683, target_std: 6.1427
    Epoch [8/50], Val Losses: mse: 10.7830, mae: 1.8104, huber: 1.4302, swd: 1.1284, target_std: 4.4388
    Epoch [8/50], Test Losses: mse: 11.3733, mae: 2.0038, huber: 1.6114, swd: 1.6631, target_std: 4.9300
      Epoch 8 composite train-obj: 1.250456
    Epoch [8/50], Test Losses: mse: 11.5607, mae: 2.0396, huber: 1.6450, swd: 1.6859, target_std: 4.9300
    Best round's Test MSE: 11.5607, MAE: 2.0396, SWD: 1.6859
    Best round's Validation MSE: 10.4863, MAE: 1.7954
    Best round's Test verification MSE : 11.5607, MAE: 2.0396, SWD: 1.6859
    Time taken: 10.27 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.8639, mae: 2.0130, huber: 1.6134, swd: 2.3197, target_std: 6.1666
    Epoch [1/50], Val Losses: mse: 11.2068, mae: 1.8545, huber: 1.4725, swd: 1.0752, target_std: 4.4388
    Epoch [1/50], Test Losses: mse: 11.8164, mae: 2.0876, huber: 1.6862, swd: 1.5630, target_std: 4.9300
      Epoch 1 composite train-obj: 1.613368
            Val objective improved inf → 1.4725, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.4350, mae: 1.6969, huber: 1.3109, swd: 1.6159, target_std: 6.1470
    Epoch [2/50], Val Losses: mse: 10.6513, mae: 1.8123, huber: 1.4293, swd: 0.8309, target_std: 4.4388
    Epoch [2/50], Test Losses: mse: 11.6061, mae: 2.0439, huber: 1.6498, swd: 1.4300, target_std: 4.9300
      Epoch 2 composite train-obj: 1.310881
            Val objective improved 1.4725 → 1.4293, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.2469, mae: 1.6666, huber: 1.2827, swd: 1.5485, target_std: 6.1418
    Epoch [3/50], Val Losses: mse: 10.9072, mae: 1.8246, huber: 1.4454, swd: 1.2970, target_std: 4.4388
    Epoch [3/50], Test Losses: mse: 11.4216, mae: 2.0225, huber: 1.6287, swd: 1.6777, target_std: 4.9300
      Epoch 3 composite train-obj: 1.282721
            No improvement (1.4454), counter 1/5
    Epoch [4/50], Train Losses: mse: 7.0937, mae: 1.6470, huber: 1.2642, swd: 1.5320, target_std: 6.1584
    Epoch [4/50], Val Losses: mse: 10.9343, mae: 1.8223, huber: 1.4419, swd: 0.9954, target_std: 4.4388
    Epoch [4/50], Test Losses: mse: 12.0381, mae: 2.0911, huber: 1.6967, swd: 1.8348, target_std: 4.9300
      Epoch 4 composite train-obj: 1.264162
            No improvement (1.4419), counter 2/5
    Epoch [5/50], Train Losses: mse: 7.2689, mae: 1.6617, huber: 1.2785, swd: 1.5342, target_std: 6.1386
    Epoch [5/50], Val Losses: mse: 10.9776, mae: 1.8811, huber: 1.4931, swd: 1.7274, target_std: 4.4388
    Epoch [5/50], Test Losses: mse: 11.9281, mae: 2.1207, huber: 1.7126, swd: 2.3088, target_std: 4.9300
      Epoch 5 composite train-obj: 1.278535
            No improvement (1.4931), counter 3/5
    Epoch [6/50], Train Losses: mse: 7.0485, mae: 1.6383, huber: 1.2551, swd: 1.5108, target_std: 6.1502
    Epoch [6/50], Val Losses: mse: 10.8206, mae: 1.8039, huber: 1.4233, swd: 0.9995, target_std: 4.4388
    Epoch [6/50], Test Losses: mse: 11.5987, mae: 2.0360, huber: 1.6375, swd: 1.5170, target_std: 4.9300
      Epoch 6 composite train-obj: 1.255142
            Val objective improved 1.4293 → 1.4233, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.0321, mae: 1.6319, huber: 1.2498, swd: 1.4822, target_std: 6.1496
    Epoch [7/50], Val Losses: mse: 11.4518, mae: 1.8890, huber: 1.5009, swd: 1.5230, target_std: 4.4388
    Epoch [7/50], Test Losses: mse: 11.8454, mae: 2.0875, huber: 1.6803, swd: 1.9702, target_std: 4.9300
      Epoch 7 composite train-obj: 1.249779
            No improvement (1.5009), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.0180, mae: 1.6299, huber: 1.2477, swd: 1.4874, target_std: 6.1587
    Epoch [8/50], Val Losses: mse: 11.5230, mae: 1.8917, huber: 1.5031, swd: 1.2069, target_std: 4.4388
    Epoch [8/50], Test Losses: mse: 11.7819, mae: 2.0736, huber: 1.6726, swd: 1.4675, target_std: 4.9300
      Epoch 8 composite train-obj: 1.247747
            No improvement (1.5031), counter 2/5
    Epoch [9/50], Train Losses: mse: 7.0862, mae: 1.6415, huber: 1.2577, swd: 1.4752, target_std: 6.1390
    Epoch [9/50], Val Losses: mse: 10.8296, mae: 1.8116, huber: 1.4313, swd: 1.2269, target_std: 4.4388
    Epoch [9/50], Test Losses: mse: 11.2614, mae: 1.9984, huber: 1.6022, swd: 1.5334, target_std: 4.9300
      Epoch 9 composite train-obj: 1.257655
            No improvement (1.4313), counter 3/5
    Epoch [10/50], Train Losses: mse: 6.9435, mae: 1.6171, huber: 1.2358, swd: 1.4708, target_std: 6.1405
    Epoch [10/50], Val Losses: mse: 10.0449, mae: 1.7707, huber: 1.3875, swd: 0.9977, target_std: 4.4388
    Epoch [10/50], Test Losses: mse: 11.8768, mae: 2.0863, huber: 1.6836, swd: 2.0081, target_std: 4.9300
      Epoch 10 composite train-obj: 1.235830
            Val objective improved 1.4233 → 1.3875, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 6.9991, mae: 1.6281, huber: 1.2458, swd: 1.5031, target_std: 6.1408
    Epoch [11/50], Val Losses: mse: 10.3289, mae: 1.7926, huber: 1.4071, swd: 0.9762, target_std: 4.4388
    Epoch [11/50], Test Losses: mse: 11.3971, mae: 2.0129, huber: 1.6177, swd: 1.6288, target_std: 4.9300
      Epoch 11 composite train-obj: 1.245782
            No improvement (1.4071), counter 1/5
    Epoch [12/50], Train Losses: mse: 6.9285, mae: 1.6148, huber: 1.2335, swd: 1.4564, target_std: 6.1343
    Epoch [12/50], Val Losses: mse: 11.0888, mae: 1.8440, huber: 1.4627, swd: 1.4218, target_std: 4.4388
    Epoch [12/50], Test Losses: mse: 11.6910, mae: 2.0476, huber: 1.6481, swd: 1.8097, target_std: 4.9300
      Epoch 12 composite train-obj: 1.233477
            No improvement (1.4627), counter 2/5
    Epoch [13/50], Train Losses: mse: 6.9529, mae: 1.6192, huber: 1.2377, swd: 1.4587, target_std: 6.1387
    Epoch [13/50], Val Losses: mse: 11.0999, mae: 1.8676, huber: 1.4813, swd: 1.1957, target_std: 4.4388
    Epoch [13/50], Test Losses: mse: 11.7651, mae: 2.0642, huber: 1.6685, swd: 1.8962, target_std: 4.9300
      Epoch 13 composite train-obj: 1.237747
            No improvement (1.4813), counter 3/5
    Epoch [14/50], Train Losses: mse: 6.9764, mae: 1.6242, huber: 1.2419, swd: 1.4635, target_std: 6.1481
    Epoch [14/50], Val Losses: mse: 10.3796, mae: 1.8060, huber: 1.4200, swd: 0.6944, target_std: 4.4388
    Epoch [14/50], Test Losses: mse: 11.6518, mae: 2.0537, huber: 1.6558, swd: 1.5222, target_std: 4.9300
      Epoch 14 composite train-obj: 1.241947
            No improvement (1.4200), counter 4/5
    Epoch [15/50], Train Losses: mse: 7.0056, mae: 1.6275, huber: 1.2456, swd: 1.4740, target_std: 6.1386
    Epoch [15/50], Val Losses: mse: 10.6779, mae: 1.7959, huber: 1.4192, swd: 1.1842, target_std: 4.4388
    Epoch [15/50], Test Losses: mse: 11.2717, mae: 1.9928, huber: 1.5988, swd: 1.5784, target_std: 4.9300
      Epoch 15 composite train-obj: 1.245570
    Epoch [15/50], Test Losses: mse: 11.8768, mae: 2.0863, huber: 1.6836, swd: 2.0081, target_std: 4.9300
    Best round's Test MSE: 11.8768, MAE: 2.0863, SWD: 2.0081
    Best round's Validation MSE: 10.0449, MAE: 1.7707
    Best round's Test verification MSE : 11.8768, MAE: 2.0863, SWD: 2.0081
    Time taken: 17.39 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.8633, mae: 2.0108, huber: 1.6113, swd: 2.1559, target_std: 6.1401
    Epoch [1/50], Val Losses: mse: 11.1160, mae: 1.8604, huber: 1.4776, swd: 1.0597, target_std: 4.4388
    Epoch [1/50], Test Losses: mse: 11.8211, mae: 2.0907, huber: 1.6935, swd: 1.7453, target_std: 4.9300
      Epoch 1 composite train-obj: 1.611272
            Val objective improved inf → 1.4776, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.3518, mae: 1.6886, huber: 1.3029, swd: 1.5009, target_std: 6.1395
    Epoch [2/50], Val Losses: mse: 10.9338, mae: 1.8356, huber: 1.4546, swd: 1.1556, target_std: 4.4388
    Epoch [2/50], Test Losses: mse: 11.4136, mae: 2.0369, huber: 1.6373, swd: 1.6654, target_std: 4.9300
      Epoch 2 composite train-obj: 1.302891
            Val objective improved 1.4776 → 1.4546, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.1708, mae: 1.6582, huber: 1.2747, swd: 1.4735, target_std: 6.1411
    Epoch [3/50], Val Losses: mse: 10.5732, mae: 1.8223, huber: 1.4345, swd: 0.8134, target_std: 4.4388
    Epoch [3/50], Test Losses: mse: 11.6440, mae: 2.0709, huber: 1.6613, swd: 1.4264, target_std: 4.9300
      Epoch 3 composite train-obj: 1.274672
            Val objective improved 1.4546 → 1.4345, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.1363, mae: 1.6511, huber: 1.2678, swd: 1.4695, target_std: 6.1392
    Epoch [4/50], Val Losses: mse: 10.5308, mae: 1.7957, huber: 1.4147, swd: 0.9096, target_std: 4.4388
    Epoch [4/50], Test Losses: mse: 11.3553, mae: 2.0159, huber: 1.6208, swd: 1.5285, target_std: 4.9300
      Epoch 4 composite train-obj: 1.267791
            Val objective improved 1.4345 → 1.4147, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.0437, mae: 1.6366, huber: 1.2545, swd: 1.4462, target_std: 6.1407
    Epoch [5/50], Val Losses: mse: 10.3614, mae: 1.7869, huber: 1.4050, swd: 0.9511, target_std: 4.4388
    Epoch [5/50], Test Losses: mse: 11.3761, mae: 2.0200, huber: 1.6237, swd: 1.6038, target_std: 4.9300
      Epoch 5 composite train-obj: 1.254512
            Val objective improved 1.4147 → 1.4050, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.0475, mae: 1.6364, huber: 1.2537, swd: 1.4481, target_std: 6.1432
    Epoch [6/50], Val Losses: mse: 10.5730, mae: 1.7844, huber: 1.4038, swd: 0.9411, target_std: 4.4388
    Epoch [6/50], Test Losses: mse: 11.3939, mae: 2.0199, huber: 1.6198, swd: 1.5699, target_std: 4.9300
      Epoch 6 composite train-obj: 1.253683
            Val objective improved 1.4050 → 1.4038, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.0040, mae: 1.6293, huber: 1.2472, swd: 1.4520, target_std: 6.1618
    Epoch [7/50], Val Losses: mse: 10.5071, mae: 1.7872, huber: 1.4080, swd: 0.8165, target_std: 4.4388
    Epoch [7/50], Test Losses: mse: 11.5476, mae: 2.0339, huber: 1.6382, swd: 1.5705, target_std: 4.9300
      Epoch 7 composite train-obj: 1.247217
            No improvement (1.4080), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.0662, mae: 1.6379, huber: 1.2553, swd: 1.4585, target_std: 6.1701
    Epoch [8/50], Val Losses: mse: 11.0015, mae: 1.8482, huber: 1.4650, swd: 1.0824, target_std: 4.4388
    Epoch [8/50], Test Losses: mse: 12.2987, mae: 2.1348, huber: 1.7374, swd: 1.9986, target_std: 4.9300
      Epoch 8 composite train-obj: 1.255349
            No improvement (1.4650), counter 2/5
    Epoch [9/50], Train Losses: mse: 7.3161, mae: 1.6561, huber: 1.2738, swd: 1.4834, target_std: 6.1413
    Epoch [9/50], Val Losses: mse: 10.2939, mae: 1.8216, huber: 1.4310, swd: 1.2461, target_std: 4.4388
    Epoch [9/50], Test Losses: mse: 11.9385, mae: 2.1317, huber: 1.7185, swd: 2.2207, target_std: 4.9300
      Epoch 9 composite train-obj: 1.273753
            No improvement (1.4310), counter 3/5
    Epoch [10/50], Train Losses: mse: 7.0518, mae: 1.6335, huber: 1.2504, swd: 1.4628, target_std: 6.1424
    Epoch [10/50], Val Losses: mse: 10.2798, mae: 1.7551, huber: 1.3784, swd: 0.6674, target_std: 4.4388
    Epoch [10/50], Test Losses: mse: 11.5524, mae: 2.0210, huber: 1.6219, swd: 1.3117, target_std: 4.9300
      Epoch 10 composite train-obj: 1.250403
            Val objective improved 1.4038 → 1.3784, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 6.9839, mae: 1.6229, huber: 1.2416, swd: 1.4377, target_std: 6.1622
    Epoch [11/50], Val Losses: mse: 10.6852, mae: 1.8105, huber: 1.4260, swd: 0.8559, target_std: 4.4388
    Epoch [11/50], Test Losses: mse: 11.5547, mae: 2.0450, huber: 1.6435, swd: 1.3795, target_std: 4.9300
      Epoch 11 composite train-obj: 1.241565
            No improvement (1.4260), counter 1/5
    Epoch [12/50], Train Losses: mse: 7.1343, mae: 1.6396, huber: 1.2576, swd: 1.4620, target_std: 6.1509
    Epoch [12/50], Val Losses: mse: 11.2584, mae: 1.8407, huber: 1.4609, swd: 1.3936, target_std: 4.4388
    Epoch [12/50], Test Losses: mse: 11.4748, mae: 2.0124, huber: 1.6191, swd: 1.7104, target_std: 4.9300
      Epoch 12 composite train-obj: 1.257631
            No improvement (1.4609), counter 2/5
    Epoch [13/50], Train Losses: mse: 6.9400, mae: 1.6174, huber: 1.2359, swd: 1.4027, target_std: 6.1520
    Epoch [13/50], Val Losses: mse: 10.1618, mae: 1.7794, huber: 1.3947, swd: 0.7434, target_std: 4.4388
    Epoch [13/50], Test Losses: mse: 11.8704, mae: 2.0686, huber: 1.6716, swd: 1.6729, target_std: 4.9300
      Epoch 13 composite train-obj: 1.235899
            No improvement (1.3947), counter 3/5
    Epoch [14/50], Train Losses: mse: 7.0800, mae: 1.6318, huber: 1.2497, swd: 1.4565, target_std: 6.1481
    Epoch [14/50], Val Losses: mse: 10.6116, mae: 1.7791, huber: 1.4017, swd: 0.9239, target_std: 4.4388
    Epoch [14/50], Test Losses: mse: 11.3211, mae: 1.9949, huber: 1.6003, swd: 1.3308, target_std: 4.9300
      Epoch 14 composite train-obj: 1.249697
            No improvement (1.4017), counter 4/5
    Epoch [15/50], Train Losses: mse: 7.0063, mae: 1.6207, huber: 1.2401, swd: 1.4489, target_std: 6.1569
    Epoch [15/50], Val Losses: mse: 12.6535, mae: 2.0217, huber: 1.6282, swd: 1.8970, target_std: 4.4388
    Epoch [15/50], Test Losses: mse: 11.9703, mae: 2.1159, huber: 1.7137, swd: 2.0443, target_std: 4.9300
      Epoch 15 composite train-obj: 1.240053
    Epoch [15/50], Test Losses: mse: 11.5524, mae: 2.0210, huber: 1.6219, swd: 1.3117, target_std: 4.9300
    Best round's Test MSE: 11.5524, MAE: 2.0210, SWD: 1.3117
    Best round's Validation MSE: 10.2798, MAE: 1.7551
    Best round's Test verification MSE : 11.5524, MAE: 2.0210, SWD: 1.3117
    Time taken: 17.74 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq720_pred336_20250503_1953)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.6633 ± 0.1510
      mae: 2.0490 ± 0.0275
      huber: 1.6502 ± 0.0254
      swd: 1.6686 ± 0.2845
      target_std: 4.9300 ± 0.0000
      count: 6.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 10.2703 ± 0.1803
      mae: 1.7737 ± 0.0166
      huber: 1.3931 ± 0.0148
      swd: 0.8769 ± 0.1487
      target_std: 4.4388 ± 0.0000
      count: 6.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 45.49 seconds
    
    Experiment complete: DLinear_etth1_seq720_pred336_20250503_1953
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
exp = execute_model_evaluation('etth1', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
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
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 85
    Validation Batches: 3
    Test Batches: 16
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.0762, mae: 2.1616, huber: 1.7506, swd: 2.7087, target_std: 6.0802
    Epoch [1/50], Val Losses: mse: 11.5355, mae: 1.9320, huber: 1.5441, swd: 1.2135, target_std: 4.4502
    Epoch [1/50], Test Losses: mse: 12.8952, mae: 2.2903, huber: 1.8704, swd: 2.2351, target_std: 4.9747
      Epoch 1 composite train-obj: 1.750621
            Val objective improved inf → 1.5441, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.5711, mae: 1.8646, huber: 1.4646, swd: 2.0018, target_std: 6.0939
    Epoch [2/50], Val Losses: mse: 12.5987, mae: 2.1170, huber: 1.7195, swd: 2.3012, target_std: 4.4502
    Epoch [2/50], Test Losses: mse: 13.5535, mae: 2.3581, huber: 1.9431, swd: 3.0259, target_std: 4.9747
      Epoch 2 composite train-obj: 1.464579
            No improvement (1.7195), counter 1/5
    Epoch [3/50], Train Losses: mse: 8.4436, mae: 1.8405, huber: 1.4424, swd: 2.0074, target_std: 6.1008
    Epoch [3/50], Val Losses: mse: 12.7901, mae: 2.0261, huber: 1.6381, swd: 1.5535, target_std: 4.4502
    Epoch [3/50], Test Losses: mse: 12.8885, mae: 2.2650, huber: 1.8454, swd: 2.1528, target_std: 4.9747
      Epoch 3 composite train-obj: 1.442350
            No improvement (1.6381), counter 2/5
    Epoch [4/50], Train Losses: mse: 8.3599, mae: 1.8249, huber: 1.4281, swd: 1.9774, target_std: 6.0827
    Epoch [4/50], Val Losses: mse: 11.2300, mae: 1.9014, huber: 1.5192, swd: 1.1118, target_std: 4.4502
    Epoch [4/50], Test Losses: mse: 12.9658, mae: 2.2665, huber: 1.8550, swd: 2.0682, target_std: 4.9747
      Epoch 4 composite train-obj: 1.428108
            Val objective improved 1.5441 → 1.5192, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 8.2370, mae: 1.8077, huber: 1.4119, swd: 1.9234, target_std: 6.0864
    Epoch [5/50], Val Losses: mse: 12.1922, mae: 1.9903, huber: 1.6038, swd: 1.8559, target_std: 4.4502
    Epoch [5/50], Test Losses: mse: 12.8724, mae: 2.2621, huber: 1.8469, swd: 2.4758, target_std: 4.9747
      Epoch 5 composite train-obj: 1.411887
            No improvement (1.6038), counter 1/5
    Epoch [6/50], Train Losses: mse: 8.2553, mae: 1.8067, huber: 1.4114, swd: 1.9732, target_std: 6.0987
    Epoch [6/50], Val Losses: mse: 11.8567, mae: 1.9517, huber: 1.5623, swd: 1.2434, target_std: 4.4502
    Epoch [6/50], Test Losses: mse: 12.6774, mae: 2.2191, huber: 1.8032, swd: 1.7207, target_std: 4.9747
      Epoch 6 composite train-obj: 1.411435
            No improvement (1.5623), counter 2/5
    Epoch [7/50], Train Losses: mse: 8.2331, mae: 1.8033, huber: 1.4080, swd: 1.9347, target_std: 6.1023
    Epoch [7/50], Val Losses: mse: 13.5959, mae: 2.2008, huber: 1.7999, swd: 2.7523, target_std: 4.4502
    Epoch [7/50], Test Losses: mse: 13.7979, mae: 2.3812, huber: 1.9657, swd: 3.3013, target_std: 4.9747
      Epoch 7 composite train-obj: 1.407991
            No improvement (1.7999), counter 3/5
    Epoch [8/50], Train Losses: mse: 8.3660, mae: 1.8228, huber: 1.4263, swd: 1.9793, target_std: 6.0970
    Epoch [8/50], Val Losses: mse: 11.8033, mae: 2.1279, huber: 1.7188, swd: 2.1686, target_std: 4.4502
    Epoch [8/50], Test Losses: mse: 13.0816, mae: 2.3103, huber: 1.8921, swd: 2.6842, target_std: 4.9747
      Epoch 8 composite train-obj: 1.426303
            No improvement (1.7188), counter 4/5
    Epoch [9/50], Train Losses: mse: 8.1766, mae: 1.7998, huber: 1.4042, swd: 1.9172, target_std: 6.0902
    Epoch [9/50], Val Losses: mse: 11.2242, mae: 1.8864, huber: 1.5014, swd: 1.2341, target_std: 4.4502
    Epoch [9/50], Test Losses: mse: 12.7147, mae: 2.2274, huber: 1.8118, swd: 1.8323, target_std: 4.9747
      Epoch 9 composite train-obj: 1.404166
            Val objective improved 1.5192 → 1.5014, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 8.1156, mae: 1.7857, huber: 1.3914, swd: 1.8717, target_std: 6.1006
    Epoch [10/50], Val Losses: mse: 13.1152, mae: 2.0879, huber: 1.6992, swd: 2.3195, target_std: 4.4502
    Epoch [10/50], Test Losses: mse: 13.0438, mae: 2.2718, huber: 1.8614, swd: 2.6022, target_std: 4.9747
      Epoch 10 composite train-obj: 1.391432
            No improvement (1.6992), counter 1/5
    Epoch [11/50], Train Losses: mse: 8.2067, mae: 1.7944, huber: 1.4004, swd: 1.8952, target_std: 6.0816
    Epoch [11/50], Val Losses: mse: 12.5253, mae: 2.0273, huber: 1.6413, swd: 2.1612, target_std: 4.4502
    Epoch [11/50], Test Losses: mse: 12.5535, mae: 2.2178, huber: 1.8050, swd: 2.2239, target_std: 4.9747
      Epoch 11 composite train-obj: 1.400416
            No improvement (1.6413), counter 2/5
    Epoch [12/50], Train Losses: mse: 8.0517, mae: 1.7753, huber: 1.3819, swd: 1.8562, target_std: 6.0726
    Epoch [12/50], Val Losses: mse: 11.1213, mae: 1.8908, huber: 1.5031, swd: 1.2962, target_std: 4.4502
    Epoch [12/50], Test Losses: mse: 12.8647, mae: 2.2511, huber: 1.8344, swd: 2.3470, target_std: 4.9747
      Epoch 12 composite train-obj: 1.381950
            No improvement (1.5031), counter 3/5
    Epoch [13/50], Train Losses: mse: 8.0190, mae: 1.7749, huber: 1.3810, swd: 1.8531, target_std: 6.0908
    Epoch [13/50], Val Losses: mse: 10.8786, mae: 1.9017, huber: 1.5088, swd: 1.1933, target_std: 4.4502
    Epoch [13/50], Test Losses: mse: 12.4708, mae: 2.2094, huber: 1.7897, swd: 1.8825, target_std: 4.9747
      Epoch 13 composite train-obj: 1.381036
            No improvement (1.5088), counter 4/5
    Epoch [14/50], Train Losses: mse: 8.0131, mae: 1.7708, huber: 1.3772, swd: 1.8413, target_std: 6.0902
    Epoch [14/50], Val Losses: mse: 12.4409, mae: 2.0121, huber: 1.6251, swd: 1.7955, target_std: 4.4502
    Epoch [14/50], Test Losses: mse: 12.7169, mae: 2.2259, huber: 1.8155, swd: 2.0717, target_std: 4.9747
      Epoch 14 composite train-obj: 1.377211
    Epoch [14/50], Test Losses: mse: 12.7147, mae: 2.2274, huber: 1.8118, swd: 1.8323, target_std: 4.9747
    Best round's Test MSE: 12.7147, MAE: 2.2274, SWD: 1.8323
    Best round's Validation MSE: 11.2242, MAE: 1.8864
    Best round's Test verification MSE : 12.7147, MAE: 2.2274, SWD: 1.8323
    Time taken: 17.97 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.1732, mae: 2.1708, huber: 1.7597, swd: 2.5376, target_std: 6.0913
    Epoch [1/50], Val Losses: mse: 11.8705, mae: 1.9726, huber: 1.5837, swd: 1.1615, target_std: 4.4502
    Epoch [1/50], Test Losses: mse: 12.9469, mae: 2.2859, huber: 1.8707, swd: 2.1138, target_std: 4.9747
      Epoch 1 composite train-obj: 1.759739
            Val objective improved inf → 1.5837, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.5301, mae: 1.8573, huber: 1.4574, swd: 1.8872, target_std: 6.0799
    Epoch [2/50], Val Losses: mse: 10.9515, mae: 1.8628, huber: 1.4792, swd: 0.7017, target_std: 4.4502
    Epoch [2/50], Test Losses: mse: 12.6639, mae: 2.2446, huber: 1.8252, swd: 1.7569, target_std: 4.9747
      Epoch 2 composite train-obj: 1.457380
            Val objective improved 1.5837 → 1.4792, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.3428, mae: 1.8278, huber: 1.4303, swd: 1.8545, target_std: 6.1042
    Epoch [3/50], Val Losses: mse: 13.6425, mae: 2.1467, huber: 1.7502, swd: 1.7436, target_std: 4.4502
    Epoch [3/50], Test Losses: mse: 13.5349, mae: 2.3576, huber: 1.9379, swd: 2.4320, target_std: 4.9747
      Epoch 3 composite train-obj: 1.430343
            No improvement (1.7502), counter 1/5
    Epoch [4/50], Train Losses: mse: 8.5060, mae: 1.8409, huber: 1.4430, swd: 1.8583, target_std: 6.0934
    Epoch [4/50], Val Losses: mse: 11.9377, mae: 1.9604, huber: 1.5723, swd: 1.1467, target_std: 4.4502
    Epoch [4/50], Test Losses: mse: 12.5748, mae: 2.2263, huber: 1.8081, swd: 1.7911, target_std: 4.9747
      Epoch 4 composite train-obj: 1.443025
            No improvement (1.5723), counter 2/5
    Epoch [5/50], Train Losses: mse: 8.2505, mae: 1.8093, huber: 1.4134, swd: 1.8567, target_std: 6.0885
    Epoch [5/50], Val Losses: mse: 12.0817, mae: 2.0129, huber: 1.6206, swd: 1.4943, target_std: 4.4502
    Epoch [5/50], Test Losses: mse: 13.1059, mae: 2.2755, huber: 1.8687, swd: 2.2699, target_std: 4.9747
      Epoch 5 composite train-obj: 1.413396
            No improvement (1.6206), counter 3/5
    Epoch [6/50], Train Losses: mse: 8.2205, mae: 1.8024, huber: 1.4071, swd: 1.7962, target_std: 6.0801
    Epoch [6/50], Val Losses: mse: 11.8586, mae: 1.9778, huber: 1.5872, swd: 1.3750, target_std: 4.4502
    Epoch [6/50], Test Losses: mse: 12.7120, mae: 2.2326, huber: 1.8198, swd: 2.0012, target_std: 4.9747
      Epoch 6 composite train-obj: 1.407064
            No improvement (1.5872), counter 4/5
    Epoch [7/50], Train Losses: mse: 8.1545, mae: 1.7951, huber: 1.3999, swd: 1.8108, target_std: 6.0976
    Epoch [7/50], Val Losses: mse: 11.3938, mae: 1.8930, huber: 1.5084, swd: 0.9398, target_std: 4.4502
    Epoch [7/50], Test Losses: mse: 12.7675, mae: 2.2317, huber: 1.8156, swd: 1.7854, target_std: 4.9747
      Epoch 7 composite train-obj: 1.399907
    Epoch [7/50], Test Losses: mse: 12.6639, mae: 2.2446, huber: 1.8252, swd: 1.7569, target_std: 4.9747
    Best round's Test MSE: 12.6639, MAE: 2.2446, SWD: 1.7569
    Best round's Validation MSE: 10.9515, MAE: 1.8628
    Best round's Test verification MSE : 12.6639, MAE: 2.2446, SWD: 1.7569
    Time taken: 8.57 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.1879, mae: 2.1704, huber: 1.7594, swd: 2.9012, target_std: 6.1006
    Epoch [1/50], Val Losses: mse: 11.9141, mae: 1.9644, huber: 1.5682, swd: 1.1117, target_std: 4.4502
    Epoch [1/50], Test Losses: mse: 12.9352, mae: 2.3015, huber: 1.8727, swd: 2.1558, target_std: 4.9747
      Epoch 1 composite train-obj: 1.759389
            Val objective improved inf → 1.5682, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.5434, mae: 1.8596, huber: 1.4595, swd: 2.0853, target_std: 6.0906
    Epoch [2/50], Val Losses: mse: 12.0098, mae: 1.9991, huber: 1.6058, swd: 1.4749, target_std: 4.4502
    Epoch [2/50], Test Losses: mse: 12.6600, mae: 2.2456, huber: 1.8292, swd: 2.1918, target_std: 4.9747
      Epoch 2 composite train-obj: 1.459485
            No improvement (1.6058), counter 1/5
    Epoch [3/50], Train Losses: mse: 8.3877, mae: 1.8309, huber: 1.4335, swd: 2.0950, target_std: 6.0884
    Epoch [3/50], Val Losses: mse: 11.5788, mae: 1.9705, huber: 1.5781, swd: 1.2937, target_std: 4.4502
    Epoch [3/50], Test Losses: mse: 12.8255, mae: 2.2556, huber: 1.8435, swd: 2.1838, target_std: 4.9747
      Epoch 3 composite train-obj: 1.433510
            No improvement (1.5781), counter 2/5
    Epoch [4/50], Train Losses: mse: 8.2646, mae: 1.8151, huber: 1.4183, swd: 2.0301, target_std: 6.0769
    Epoch [4/50], Val Losses: mse: 12.0385, mae: 1.9874, huber: 1.5937, swd: 1.4358, target_std: 4.4502
    Epoch [4/50], Test Losses: mse: 12.9872, mae: 2.2737, huber: 1.8545, swd: 2.4174, target_std: 4.9747
      Epoch 4 composite train-obj: 1.418332
            No improvement (1.5937), counter 3/5
    Epoch [5/50], Train Losses: mse: 8.2230, mae: 1.8074, huber: 1.4110, swd: 2.0225, target_std: 6.0921
    Epoch [5/50], Val Losses: mse: 11.0672, mae: 1.8909, huber: 1.5051, swd: 1.1937, target_std: 4.4502
    Epoch [5/50], Test Losses: mse: 12.5756, mae: 2.2270, huber: 1.8092, swd: 1.9846, target_std: 4.9747
      Epoch 5 composite train-obj: 1.411039
            Val objective improved 1.5682 → 1.5051, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 8.2069, mae: 1.7998, huber: 1.4045, swd: 2.0269, target_std: 6.0839
    Epoch [6/50], Val Losses: mse: 11.6668, mae: 1.9075, huber: 1.5217, swd: 1.0605, target_std: 4.4502
    Epoch [6/50], Test Losses: mse: 12.8052, mae: 2.2412, huber: 1.8237, swd: 1.9055, target_std: 4.9747
      Epoch 6 composite train-obj: 1.404474
            No improvement (1.5217), counter 1/5
    Epoch [7/50], Train Losses: mse: 8.2184, mae: 1.8003, huber: 1.4050, swd: 2.0191, target_std: 6.0883
    Epoch [7/50], Val Losses: mse: 11.3098, mae: 1.9371, huber: 1.5447, swd: 1.5090, target_std: 4.4502
    Epoch [7/50], Test Losses: mse: 12.7411, mae: 2.2483, huber: 1.8299, swd: 2.3469, target_std: 4.9747
      Epoch 7 composite train-obj: 1.404976
            No improvement (1.5447), counter 2/5
    Epoch [8/50], Train Losses: mse: 8.1362, mae: 1.7927, huber: 1.3977, swd: 1.9949, target_std: 6.0980
    Epoch [8/50], Val Losses: mse: 11.5834, mae: 1.9396, huber: 1.5435, swd: 1.2304, target_std: 4.4502
    Epoch [8/50], Test Losses: mse: 12.5983, mae: 2.2474, huber: 1.8171, swd: 1.8703, target_std: 4.9747
      Epoch 8 composite train-obj: 1.397692
            No improvement (1.5435), counter 3/5
    Epoch [9/50], Train Losses: mse: 8.1405, mae: 1.7928, huber: 1.3970, swd: 1.9992, target_std: 6.0868
    Epoch [9/50], Val Losses: mse: 10.3514, mae: 1.8908, huber: 1.4937, swd: 0.9624, target_std: 4.4502
    Epoch [9/50], Test Losses: mse: 13.0446, mae: 2.2761, huber: 1.8587, swd: 2.2412, target_std: 4.9747
      Epoch 9 composite train-obj: 1.396967
            Val objective improved 1.5051 → 1.4937, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 8.2461, mae: 1.8014, huber: 1.4065, swd: 2.0075, target_std: 6.0850
    Epoch [10/50], Val Losses: mse: 12.0691, mae: 1.9906, huber: 1.6017, swd: 1.7776, target_std: 4.4502
    Epoch [10/50], Test Losses: mse: 12.6587, mae: 2.2128, huber: 1.8057, swd: 2.1872, target_std: 4.9747
      Epoch 10 composite train-obj: 1.406540
            No improvement (1.6017), counter 1/5
    Epoch [11/50], Train Losses: mse: 8.1795, mae: 1.7907, huber: 1.3966, swd: 2.0125, target_std: 6.1069
    Epoch [11/50], Val Losses: mse: 12.3801, mae: 2.0331, huber: 1.6328, swd: 1.7431, target_std: 4.4502
    Epoch [11/50], Test Losses: mse: 12.8857, mae: 2.2733, huber: 1.8487, swd: 2.4501, target_std: 4.9747
      Epoch 11 composite train-obj: 1.396614
            No improvement (1.6328), counter 2/5
    Epoch [12/50], Train Losses: mse: 8.1081, mae: 1.7870, huber: 1.3921, swd: 1.9718, target_std: 6.1047
    Epoch [12/50], Val Losses: mse: 12.4318, mae: 1.9562, huber: 1.5688, swd: 1.3469, target_std: 4.4502
    Epoch [12/50], Test Losses: mse: 12.9994, mae: 2.2582, huber: 1.8392, swd: 2.2249, target_std: 4.9747
      Epoch 12 composite train-obj: 1.392137
            No improvement (1.5688), counter 3/5
    Epoch [13/50], Train Losses: mse: 8.1575, mae: 1.7891, huber: 1.3949, swd: 1.9805, target_std: 6.1040
    Epoch [13/50], Val Losses: mse: 11.5723, mae: 1.8909, huber: 1.5086, swd: 1.2969, target_std: 4.4502
    Epoch [13/50], Test Losses: mse: 12.9593, mae: 2.2413, huber: 1.8338, swd: 2.4047, target_std: 4.9747
      Epoch 13 composite train-obj: 1.394864
            No improvement (1.5086), counter 4/5
    Epoch [14/50], Train Losses: mse: 8.0944, mae: 1.7813, huber: 1.3875, swd: 1.9618, target_std: 6.0951
    Epoch [14/50], Val Losses: mse: 11.0916, mae: 1.9171, huber: 1.5273, swd: 1.2563, target_std: 4.4502
    Epoch [14/50], Test Losses: mse: 12.7281, mae: 2.2326, huber: 1.8195, swd: 2.2797, target_std: 4.9747
      Epoch 14 composite train-obj: 1.387463
    Epoch [14/50], Test Losses: mse: 13.0446, mae: 2.2761, huber: 1.8587, swd: 2.2412, target_std: 4.9747
    Best round's Test MSE: 13.0446, MAE: 2.2761, SWD: 2.2412
    Best round's Validation MSE: 10.3514, MAE: 1.8908
    Best round's Test verification MSE : 13.0446, MAE: 2.2761, SWD: 2.2412
    Time taken: 20.39 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth1_seq720_pred720_20250503_1954)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.8077 ± 0.1687
      mae: 2.2494 ± 0.0202
      huber: 1.8319 ± 0.0197
      swd: 1.9435 ± 0.2128
      target_std: 4.9747 ± 0.0000
      count: 3.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 10.8424 ± 0.3646
      mae: 1.8800 ± 0.0123
      huber: 1.4914 ± 0.0092
      swd: 0.9660 ± 0.2173
      target_std: 4.4502 ± 0.0000
      count: 3.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 47.10 seconds
    
    Experiment complete: DLinear_etth1_seq720_pred720_20250503_1954
    Model: DLinear
    Dataset: etth1
    Sequence Length: 720
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


```python

```


