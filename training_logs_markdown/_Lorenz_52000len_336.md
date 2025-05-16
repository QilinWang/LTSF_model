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
data_mgr.load_trajectory('lorenz', steps=51999, dt=1e-2) # 51999 36999
# SCALE = False
```

    LorenzSystem initialized with method: rk4 on device: cuda
    
    ==================================================
    Dataset: lorenz (synthetic)
    ==================================================
    Shape: torch.Size([52000, 3])
    Channels: 3
    Length: 52000
    Parameters: {'steps': 51999, 'dt': 0.01}
    
    Sample data (first 2 rows):
    tensor([[1.0000, 0.9800, 1.1000],
            [1.0106, 1.2389, 1.0820]], device='cuda:0')
    ==================================================
    




    <data_manager.DatasetManager at 0x2a05bed9790>



## seq=336

### EigenACL

#### 336-96
##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=96,
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
    # single_magnitude_for_shift=True,
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False,
    

)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 282
    Validation Batches: 38
    Test Batches: 78
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 34.7820, mae: 3.6312, huber: 3.1884, swd: 16.8735, ept: 60.0233
    Epoch [1/50], Val Losses: mse: 21.7887, mae: 2.8574, huber: 2.4168, swd: 2.6863, ept: 75.5482
    Epoch [1/50], Test Losses: mse: 20.1222, mae: 2.7785, huber: 2.3379, swd: 2.2687, ept: 76.3366
      Epoch 1 composite train-obj: 3.188408
            Val objective improved inf → 2.4168, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.0218, mae: 1.4461, huber: 1.0620, swd: 1.4743, ept: 88.1114
    Epoch [2/50], Val Losses: mse: 16.5572, mae: 2.3158, huber: 1.8917, swd: 2.8962, ept: 83.2372
    Epoch [2/50], Test Losses: mse: 16.2355, mae: 2.3329, huber: 1.9062, swd: 2.7243, ept: 83.2223
      Epoch 2 composite train-obj: 1.061982
            Val objective improved 2.4168 → 1.8917, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.1502, mae: 1.2093, huber: 0.8441, swd: 1.0280, ept: 90.6465
    Epoch [3/50], Val Losses: mse: 7.3081, mae: 1.5214, huber: 1.1230, swd: 1.2532, ept: 87.3442
    Epoch [3/50], Test Losses: mse: 7.1584, mae: 1.5104, huber: 1.1141, swd: 1.2410, ept: 87.5103
      Epoch 3 composite train-obj: 0.844143
            Val objective improved 1.8917 → 1.1230, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.7329, mae: 1.0112, huber: 0.6666, swd: 0.7836, ept: 92.2107
    Epoch [4/50], Val Losses: mse: 10.5917, mae: 1.7705, huber: 1.3764, swd: 1.1722, ept: 87.1124
    Epoch [4/50], Test Losses: mse: 10.0884, mae: 1.7363, huber: 1.3417, swd: 1.1240, ept: 87.4896
      Epoch 4 composite train-obj: 0.666589
            No improvement (1.3764), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.2643, mae: 0.9359, huber: 0.6023, swd: 0.6987, ept: 93.0441
    Epoch [5/50], Val Losses: mse: 13.2834, mae: 1.8653, huber: 1.4647, swd: 1.3545, ept: 87.8302
    Epoch [5/50], Test Losses: mse: 12.9421, mae: 1.8548, huber: 1.4552, swd: 1.3305, ept: 87.4294
      Epoch 5 composite train-obj: 0.602284
            No improvement (1.4647), counter 2/5
    Epoch [6/50], Train Losses: mse: 4.6669, mae: 0.9643, huber: 0.6310, swd: 0.7659, ept: 92.6572
    Epoch [6/50], Val Losses: mse: 8.0737, mae: 1.3763, huber: 1.0056, swd: 1.3440, ept: 90.0021
    Epoch [6/50], Test Losses: mse: 7.6452, mae: 1.3563, huber: 0.9879, swd: 1.2575, ept: 90.2513
      Epoch 6 composite train-obj: 0.630989
            Val objective improved 1.1230 → 1.0056, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.9116, mae: 0.8675, huber: 0.5456, swd: 0.6751, ept: 93.2126
    Epoch [7/50], Val Losses: mse: 4.6209, mae: 1.1038, huber: 0.7433, swd: 1.1159, ept: 92.8788
    Epoch [7/50], Test Losses: mse: 4.6827, mae: 1.1016, huber: 0.7420, swd: 1.0255, ept: 93.2425
      Epoch 7 composite train-obj: 0.545584
            Val objective improved 1.0056 → 0.7433, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.5381, mae: 0.8218, huber: 0.5050, swd: 0.6015, ept: 93.3396
    Epoch [8/50], Val Losses: mse: 18.4412, mae: 2.2151, huber: 1.8125, swd: 2.2117, ept: 84.9949
    Epoch [8/50], Test Losses: mse: 16.4207, mae: 2.1181, huber: 1.7188, swd: 2.0754, ept: 85.5529
      Epoch 8 composite train-obj: 0.505011
            No improvement (1.8125), counter 1/5
    Epoch [9/50], Train Losses: mse: 5.9239, mae: 1.0739, huber: 0.7364, swd: 0.9303, ept: 91.8264
    Epoch [9/50], Val Losses: mse: 6.5637, mae: 1.2844, huber: 0.9162, swd: 0.9106, ept: 91.5188
    Epoch [9/50], Test Losses: mse: 5.6993, mae: 1.2438, huber: 0.8782, swd: 0.8001, ept: 91.6838
      Epoch 9 composite train-obj: 0.736440
            No improvement (0.9162), counter 2/5
    Epoch [10/50], Train Losses: mse: 3.9179, mae: 0.8380, huber: 0.5223, swd: 0.5976, ept: 93.4968
    Epoch [10/50], Val Losses: mse: 3.8962, mae: 1.0079, huber: 0.6627, swd: 1.1449, ept: 90.6162
    Epoch [10/50], Test Losses: mse: 3.4318, mae: 0.9980, huber: 0.6494, swd: 1.0160, ept: 91.3069
      Epoch 10 composite train-obj: 0.522291
            Val objective improved 0.7433 → 0.6627, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 3.4106, mae: 0.7769, huber: 0.4707, swd: 0.5612, ept: 93.5519
    Epoch [11/50], Val Losses: mse: 5.2157, mae: 1.0944, huber: 0.7583, swd: 0.7254, ept: 90.8687
    Epoch [11/50], Test Losses: mse: 4.8904, mae: 1.0697, huber: 0.7323, swd: 0.7361, ept: 91.3777
      Epoch 11 composite train-obj: 0.470711
            No improvement (0.7583), counter 1/5
    Epoch [12/50], Train Losses: mse: 3.1722, mae: 0.7410, huber: 0.4408, swd: 0.5310, ept: 93.9123
    Epoch [12/50], Val Losses: mse: 22.6348, mae: 2.4038, huber: 1.9924, swd: 1.2824, ept: 82.4350
    Epoch [12/50], Test Losses: mse: 19.7187, mae: 2.2487, huber: 1.8419, swd: 1.2353, ept: 83.6913
      Epoch 12 composite train-obj: 0.440812
            No improvement (1.9924), counter 2/5
    Epoch [13/50], Train Losses: mse: 9.9071, mae: 1.4932, huber: 1.1262, swd: 2.0296, ept: 87.9305
    Epoch [13/50], Val Losses: mse: 14.1600, mae: 1.7571, huber: 1.3838, swd: 1.5140, ept: 86.2942
    Epoch [13/50], Test Losses: mse: 13.1980, mae: 1.7313, huber: 1.3575, swd: 1.4445, ept: 86.2658
      Epoch 13 composite train-obj: 1.126207
            No improvement (1.3838), counter 3/5
    Epoch [14/50], Train Losses: mse: 5.0717, mae: 0.9367, huber: 0.6185, swd: 0.7006, ept: 92.2504
    Epoch [14/50], Val Losses: mse: 8.9177, mae: 1.4638, huber: 1.0911, swd: 0.8511, ept: 88.1297
    Epoch [14/50], Test Losses: mse: 8.1014, mae: 1.4214, huber: 1.0473, swd: 0.7125, ept: 88.6434
      Epoch 14 composite train-obj: 0.618484
            No improvement (1.0911), counter 4/5
    Epoch [15/50], Train Losses: mse: 3.4486, mae: 0.7478, huber: 0.4508, swd: 0.5052, ept: 93.4231
    Epoch [15/50], Val Losses: mse: 3.6647, mae: 0.9718, huber: 0.6305, swd: 0.7921, ept: 93.0907
    Epoch [15/50], Test Losses: mse: 3.6108, mae: 0.9647, huber: 0.6257, swd: 0.7255, ept: 93.2447
      Epoch 15 composite train-obj: 0.450820
            Val objective improved 0.6627 → 0.6305, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 3.3312, mae: 0.7368, huber: 0.4402, swd: 0.5079, ept: 93.5032
    Epoch [16/50], Val Losses: mse: 29.4285, mae: 2.4785, huber: 2.0917, swd: 2.2794, ept: 82.4329
    Epoch [16/50], Test Losses: mse: 29.7929, mae: 2.4819, huber: 2.0948, swd: 2.0935, ept: 82.7960
      Epoch 16 composite train-obj: 0.440194
            No improvement (2.0917), counter 1/5
    Epoch [17/50], Train Losses: mse: 6.0131, mae: 1.0278, huber: 0.7041, swd: 0.8617, ept: 91.6204
    Epoch [17/50], Val Losses: mse: 8.7182, mae: 1.3652, huber: 1.0090, swd: 1.2038, ept: 91.0533
    Epoch [17/50], Test Losses: mse: 7.8821, mae: 1.2866, huber: 0.9356, swd: 1.0562, ept: 91.7776
      Epoch 17 composite train-obj: 0.704082
            No improvement (1.0090), counter 2/5
    Epoch [18/50], Train Losses: mse: 4.0738, mae: 0.7942, huber: 0.4956, swd: 0.5771, ept: 93.4657
    Epoch [18/50], Val Losses: mse: 3.1386, mae: 0.9247, huber: 0.5825, swd: 0.9277, ept: 93.0757
    Epoch [18/50], Test Losses: mse: 3.3541, mae: 0.9186, huber: 0.5786, swd: 0.7982, ept: 93.1651
      Epoch 18 composite train-obj: 0.495612
            Val objective improved 0.6305 → 0.5825, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 2.7753, mae: 0.6518, huber: 0.3719, swd: 0.4073, ept: 94.1835
    Epoch [19/50], Val Losses: mse: 3.8556, mae: 0.8799, huber: 0.5616, swd: 0.7419, ept: 92.5207
    Epoch [19/50], Test Losses: mse: 3.1832, mae: 0.8314, huber: 0.5170, swd: 0.4542, ept: 93.0449
      Epoch 19 composite train-obj: 0.371920
            Val objective improved 0.5825 → 0.5616, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 2.7125, mae: 0.6579, huber: 0.3752, swd: 0.4252, ept: 94.1512
    Epoch [20/50], Val Losses: mse: 7.5855, mae: 1.2409, huber: 0.8931, swd: 0.8404, ept: 90.2843
    Epoch [20/50], Test Losses: mse: 7.3916, mae: 1.2570, huber: 0.9062, swd: 0.7236, ept: 89.8984
      Epoch 20 composite train-obj: 0.375196
            No improvement (0.8931), counter 1/5
    Epoch [21/50], Train Losses: mse: 3.3542, mae: 0.6953, huber: 0.4150, swd: 0.4838, ept: 93.8777
    Epoch [21/50], Val Losses: mse: 4.4255, mae: 0.8742, huber: 0.5639, swd: 0.5895, ept: 92.6633
    Epoch [21/50], Test Losses: mse: 3.7523, mae: 0.8432, huber: 0.5328, swd: 0.5151, ept: 92.9806
      Epoch 21 composite train-obj: 0.414997
            No improvement (0.5639), counter 2/5
    Epoch [22/50], Train Losses: mse: 2.3546, mae: 0.6016, huber: 0.3309, swd: 0.3606, ept: 94.4657
    Epoch [22/50], Val Losses: mse: 3.1065, mae: 0.8766, huber: 0.5463, swd: 0.5758, ept: 93.8620
    Epoch [22/50], Test Losses: mse: 2.7531, mae: 0.8192, huber: 0.4966, swd: 0.4857, ept: 94.3615
      Epoch 22 composite train-obj: 0.330941
            Val objective improved 0.5616 → 0.5463, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 2.4448, mae: 0.6193, huber: 0.3442, swd: 0.3759, ept: 94.4597
    Epoch [23/50], Val Losses: mse: 4.8762, mae: 1.0021, huber: 0.6747, swd: 0.6540, ept: 92.4550
    Epoch [23/50], Test Losses: mse: 4.1621, mae: 0.9505, huber: 0.6287, swd: 0.5581, ept: 92.4407
      Epoch 23 composite train-obj: 0.344153
            No improvement (0.6747), counter 1/5
    Epoch [24/50], Train Losses: mse: 2.5041, mae: 0.6284, huber: 0.3531, swd: 0.4081, ept: 94.4215
    Epoch [24/50], Val Losses: mse: 5.2967, mae: 1.1517, huber: 0.8050, swd: 0.8245, ept: 90.4589
    Epoch [24/50], Test Losses: mse: 4.7323, mae: 1.1285, huber: 0.7783, swd: 0.6768, ept: 90.7500
      Epoch 24 composite train-obj: 0.353103
            No improvement (0.8050), counter 2/5
    Epoch [25/50], Train Losses: mse: 2.6019, mae: 0.6343, huber: 0.3581, swd: 0.3909, ept: 94.4721
    Epoch [25/50], Val Losses: mse: 4.0862, mae: 0.8587, huber: 0.5460, swd: 0.8573, ept: 93.0474
    Epoch [25/50], Test Losses: mse: 3.4968, mae: 0.8139, huber: 0.5029, swd: 0.6980, ept: 93.4239
      Epoch 25 composite train-obj: 0.358074
            Val objective improved 0.5463 → 0.5460, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 2.5935, mae: 0.6295, huber: 0.3551, swd: 0.4300, ept: 94.3789
    Epoch [26/50], Val Losses: mse: 2.4182, mae: 0.6847, huber: 0.3987, swd: 0.6860, ept: 94.3358
    Epoch [26/50], Test Losses: mse: 2.0948, mae: 0.6566, huber: 0.3715, swd: 0.5687, ept: 94.4933
      Epoch 26 composite train-obj: 0.355149
            Val objective improved 0.5460 → 0.3987, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 1.9917, mae: 0.5432, huber: 0.2863, swd: 0.3308, ept: 94.7104
    Epoch [27/50], Val Losses: mse: 9.0625, mae: 1.2460, huber: 0.8897, swd: 0.6132, ept: 91.3491
    Epoch [27/50], Test Losses: mse: 8.3444, mae: 1.2216, huber: 0.8677, swd: 0.5857, ept: 91.5390
      Epoch 27 composite train-obj: 0.286330
            No improvement (0.8897), counter 1/5
    Epoch [28/50], Train Losses: mse: 2.3973, mae: 0.5927, huber: 0.3272, swd: 0.4170, ept: 94.5428
    Epoch [28/50], Val Losses: mse: 2.3883, mae: 0.6645, huber: 0.3824, swd: 0.5074, ept: 94.2201
    Epoch [28/50], Test Losses: mse: 2.2905, mae: 0.6493, huber: 0.3683, swd: 0.4418, ept: 94.5288
      Epoch 28 composite train-obj: 0.327192
            Val objective improved 0.3987 → 0.3824, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 1.9206, mae: 0.5351, huber: 0.2801, swd: 0.3250, ept: 94.8070
    Epoch [29/50], Val Losses: mse: 3.4929, mae: 0.8082, huber: 0.4937, swd: 0.7485, ept: 93.8939
    Epoch [29/50], Test Losses: mse: 2.7497, mae: 0.7771, huber: 0.4633, swd: 0.5153, ept: 94.2989
      Epoch 29 composite train-obj: 0.280122
            No improvement (0.4937), counter 1/5
    Epoch [30/50], Train Losses: mse: 2.1426, mae: 0.5656, huber: 0.3030, swd: 0.3657, ept: 94.7053
    Epoch [30/50], Val Losses: mse: 2.2646, mae: 0.7086, huber: 0.4094, swd: 0.4439, ept: 94.6987
    Epoch [30/50], Test Losses: mse: 1.8087, mae: 0.6738, huber: 0.3788, swd: 0.3922, ept: 94.9652
      Epoch 30 composite train-obj: 0.303011
            No improvement (0.4094), counter 2/5
    Epoch [31/50], Train Losses: mse: 2.7536, mae: 0.6559, huber: 0.3761, swd: 0.5656, ept: 94.1557
    Epoch [31/50], Val Losses: mse: 3.5366, mae: 0.7833, huber: 0.4844, swd: 0.6832, ept: 93.3762
    Epoch [31/50], Test Losses: mse: 3.2910, mae: 0.7628, huber: 0.4653, swd: 0.5569, ept: 93.6691
      Epoch 31 composite train-obj: 0.376118
            No improvement (0.4844), counter 3/5
    Epoch [32/50], Train Losses: mse: 2.3821, mae: 0.6114, huber: 0.3385, swd: 0.4348, ept: 94.4488
    Epoch [32/50], Val Losses: mse: 5.8606, mae: 1.0807, huber: 0.7441, swd: 0.8574, ept: 91.3081
    Epoch [32/50], Test Losses: mse: 5.4242, mae: 1.0660, huber: 0.7270, swd: 0.7174, ept: 91.3406
      Epoch 32 composite train-obj: 0.338510
            No improvement (0.7441), counter 4/5
    Epoch [33/50], Train Losses: mse: 2.2613, mae: 0.6013, huber: 0.3294, swd: 0.3900, ept: 94.5556
    Epoch [33/50], Val Losses: mse: 2.0484, mae: 0.6493, huber: 0.3577, swd: 0.6918, ept: 94.5597
    Epoch [33/50], Test Losses: mse: 1.6730, mae: 0.6219, huber: 0.3317, swd: 0.5168, ept: 94.8696
      Epoch 33 composite train-obj: 0.329404
            Val objective improved 0.3824 → 0.3577, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 2.1231, mae: 0.5751, huber: 0.3094, swd: 0.3879, ept: 94.6106
    Epoch [34/50], Val Losses: mse: 2.9534, mae: 0.8198, huber: 0.4958, swd: 0.6591, ept: 93.9189
    Epoch [34/50], Test Losses: mse: 2.4399, mae: 0.7726, huber: 0.4493, swd: 0.4688, ept: 94.5797
      Epoch 34 composite train-obj: 0.309411
            No improvement (0.4958), counter 1/5
    Epoch [35/50], Train Losses: mse: 2.1367, mae: 0.5841, huber: 0.3204, swd: 0.4078, ept: 94.2403
    Epoch [35/50], Val Losses: mse: 2.8210, mae: 0.7586, huber: 0.4529, swd: 0.7799, ept: 93.9102
    Epoch [35/50], Test Losses: mse: 2.1465, mae: 0.7055, huber: 0.4038, swd: 0.5771, ept: 94.2813
      Epoch 35 composite train-obj: 0.320372
            No improvement (0.4529), counter 2/5
    Epoch [36/50], Train Losses: mse: 2.1227, mae: 0.5793, huber: 0.3105, swd: 0.3846, ept: 94.6840
    Epoch [36/50], Val Losses: mse: 2.8254, mae: 0.7037, huber: 0.4053, swd: 0.7722, ept: 94.1774
    Epoch [36/50], Test Losses: mse: 2.5412, mae: 0.6726, huber: 0.3798, swd: 0.5748, ept: 94.5054
      Epoch 36 composite train-obj: 0.310500
            No improvement (0.4053), counter 3/5
    Epoch [37/50], Train Losses: mse: 2.0365, mae: 0.5689, huber: 0.3042, swd: 0.3635, ept: 94.6314
    Epoch [37/50], Val Losses: mse: 6.7398, mae: 1.0682, huber: 0.7426, swd: 0.9663, ept: 90.9822
    Epoch [37/50], Test Losses: mse: 6.9592, mae: 1.0756, huber: 0.7471, swd: 0.8781, ept: 91.1098
      Epoch 37 composite train-obj: 0.304241
            No improvement (0.7426), counter 4/5
    Epoch [38/50], Train Losses: mse: 2.8703, mae: 0.6820, huber: 0.4026, swd: 0.4143, ept: 93.8200
    Epoch [38/50], Val Losses: mse: 27.8816, mae: 2.1805, huber: 1.8046, swd: 7.9705, ept: 84.2131
    Epoch [38/50], Test Losses: mse: 28.5865, mae: 2.2155, huber: 1.8377, swd: 8.2540, ept: 84.1910
      Epoch 38 composite train-obj: 0.402596
    Epoch [38/50], Test Losses: mse: 1.6725, mae: 0.6218, huber: 0.3317, swd: 0.5169, ept: 94.8719
    Best round's Test MSE: 1.6730, MAE: 0.6219, SWD: 0.5168
    Best round's Validation MSE: 2.0484, MAE: 0.6493, SWD: 0.6918
    Best round's Test verification MSE : 1.6725, MAE: 0.6218, SWD: 0.5169
    Time taken: 303.53 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.8665, mae: 3.4835, huber: 3.0428, swd: 15.6344, ept: 64.5561
    Epoch [1/50], Val Losses: mse: 18.6553, mae: 2.6240, huber: 2.1909, swd: 2.5824, ept: 79.2317
    Epoch [1/50], Test Losses: mse: 17.9230, mae: 2.5748, huber: 2.1432, swd: 2.4801, ept: 79.4923
      Epoch 1 composite train-obj: 3.042754
            Val objective improved inf → 2.1909, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.6805, mae: 1.5106, huber: 1.1226, swd: 1.5858, ept: 89.7293
    Epoch [2/50], Val Losses: mse: 11.7384, mae: 2.0364, huber: 1.6166, swd: 1.8561, ept: 84.9255
    Epoch [2/50], Test Losses: mse: 11.0903, mae: 1.9849, huber: 1.5663, swd: 1.8240, ept: 85.2354
      Epoch 2 composite train-obj: 1.122555
            Val objective improved 2.1909 → 1.6166, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.8408, mae: 1.1459, huber: 0.7871, swd: 1.0817, ept: 92.1895
    Epoch [3/50], Val Losses: mse: 24.0517, mae: 2.7325, huber: 2.3066, swd: 1.9546, ept: 80.6778
    Epoch [3/50], Test Losses: mse: 21.9937, mae: 2.5981, huber: 2.1762, swd: 1.7638, ept: 81.7534
      Epoch 3 composite train-obj: 0.787052
            No improvement (2.3066), counter 1/5
    Epoch [4/50], Train Losses: mse: 7.6775, mae: 1.3364, huber: 0.9688, swd: 1.3082, ept: 90.5061
    Epoch [4/50], Val Losses: mse: 13.1579, mae: 1.8425, huber: 1.4485, swd: 1.1606, ept: 86.6514
    Epoch [4/50], Test Losses: mse: 12.3311, mae: 1.8140, huber: 1.4206, swd: 0.9578, ept: 86.5939
      Epoch 4 composite train-obj: 0.968803
            Val objective improved 1.6166 → 1.4485, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.4997, mae: 1.0697, huber: 0.7235, swd: 0.8511, ept: 92.2886
    Epoch [5/50], Val Losses: mse: 9.1546, mae: 1.5139, huber: 1.1319, swd: 1.9626, ept: 90.6092
    Epoch [5/50], Test Losses: mse: 9.0885, mae: 1.4987, huber: 1.1175, swd: 1.9479, ept: 90.9202
      Epoch 5 composite train-obj: 0.723457
            Val objective improved 1.4485 → 1.1319, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.8176, mae: 0.9869, huber: 0.6483, swd: 0.7936, ept: 93.0294
    Epoch [6/50], Val Losses: mse: 6.0278, mae: 1.2508, huber: 0.8979, swd: 1.1157, ept: 90.9298
    Epoch [6/50], Test Losses: mse: 5.9368, mae: 1.2522, huber: 0.8984, swd: 1.1423, ept: 90.8668
      Epoch 6 composite train-obj: 0.648347
            Val objective improved 1.1319 → 0.8979, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.5199, mae: 0.8205, huber: 0.5043, swd: 0.6132, ept: 93.8214
    Epoch [7/50], Val Losses: mse: 6.8543, mae: 1.2619, huber: 0.9058, swd: 0.6680, ept: 91.1919
    Epoch [7/50], Test Losses: mse: 6.5100, mae: 1.2440, huber: 0.8891, swd: 0.6277, ept: 91.1501
      Epoch 7 composite train-obj: 0.504333
            No improvement (0.9058), counter 1/5
    Epoch [8/50], Train Losses: mse: 3.2495, mae: 0.7737, huber: 0.4651, swd: 0.5569, ept: 94.1008
    Epoch [8/50], Val Losses: mse: 4.9618, mae: 1.0590, huber: 0.7110, swd: 0.7781, ept: 93.2358
    Epoch [8/50], Test Losses: mse: 4.0404, mae: 1.0310, huber: 0.6834, swd: 0.6534, ept: 93.4384
      Epoch 8 composite train-obj: 0.465065
            Val objective improved 0.8979 → 0.7110, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.0414, mae: 0.7380, huber: 0.4363, swd: 0.5123, ept: 94.2360
    Epoch [9/50], Val Losses: mse: 30.2757, mae: 2.5672, huber: 2.1678, swd: 1.7897, ept: 82.1084
    Epoch [9/50], Test Losses: mse: 29.7451, mae: 2.5352, huber: 2.1356, swd: 1.5863, ept: 82.6574
      Epoch 9 composite train-obj: 0.436304
            No improvement (2.1678), counter 1/5
    Epoch [10/50], Train Losses: mse: 6.1211, mae: 1.0715, huber: 0.7383, swd: 0.8450, ept: 92.0212
    Epoch [10/50], Val Losses: mse: 10.2962, mae: 1.6309, huber: 1.2487, swd: 1.1931, ept: 89.3373
    Epoch [10/50], Test Losses: mse: 9.5296, mae: 1.5518, huber: 1.1743, swd: 1.1128, ept: 89.8297
      Epoch 10 composite train-obj: 0.738269
            No improvement (1.2487), counter 2/5
    Epoch [11/50], Train Losses: mse: 4.0015, mae: 0.8252, huber: 0.5184, swd: 0.6155, ept: 93.6634
    Epoch [11/50], Val Losses: mse: 4.7275, mae: 1.0225, huber: 0.6858, swd: 0.9704, ept: 92.4676
    Epoch [11/50], Test Losses: mse: 4.5208, mae: 0.9972, huber: 0.6637, swd: 0.7856, ept: 92.7398
      Epoch 11 composite train-obj: 0.518398
            Val objective improved 0.7110 → 0.6858, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 3.1094, mae: 0.7270, huber: 0.4305, swd: 0.4816, ept: 94.1039
    Epoch [12/50], Val Losses: mse: 4.3037, mae: 1.0405, huber: 0.6976, swd: 0.7739, ept: 92.0077
    Epoch [12/50], Test Losses: mse: 4.0326, mae: 1.0176, huber: 0.6740, swd: 0.5714, ept: 92.3301
      Epoch 12 composite train-obj: 0.430464
            No improvement (0.6976), counter 1/5
    Epoch [13/50], Train Losses: mse: 3.0168, mae: 0.7046, huber: 0.4124, swd: 0.4723, ept: 94.2825
    Epoch [13/50], Val Losses: mse: 12.0239, mae: 1.6625, huber: 1.2844, swd: 1.5422, ept: 88.1128
    Epoch [13/50], Test Losses: mse: 12.6707, mae: 1.7181, huber: 1.3386, swd: 1.4408, ept: 87.9870
      Epoch 13 composite train-obj: 0.412420
            No improvement (1.2844), counter 2/5
    Epoch [14/50], Train Losses: mse: 5.0025, mae: 0.9319, huber: 0.6136, swd: 0.8933, ept: 92.8862
    Epoch [14/50], Val Losses: mse: 15.1764, mae: 1.6918, huber: 1.3089, swd: 2.1812, ept: 89.4151
    Epoch [14/50], Test Losses: mse: 13.2973, mae: 1.6155, huber: 1.2329, swd: 1.8948, ept: 90.2217
      Epoch 14 composite train-obj: 0.613560
            No improvement (1.3089), counter 3/5
    Epoch [15/50], Train Losses: mse: 5.3753, mae: 0.9871, huber: 0.6601, swd: 0.9302, ept: 92.5614
    Epoch [15/50], Val Losses: mse: 3.5241, mae: 0.9523, huber: 0.6130, swd: 0.6332, ept: 92.8557
    Epoch [15/50], Test Losses: mse: 3.1931, mae: 0.9337, huber: 0.5933, swd: 0.4929, ept: 93.1966
      Epoch 15 composite train-obj: 0.660088
            Val objective improved 0.6858 → 0.6130, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 3.2642, mae: 0.7536, huber: 0.4531, swd: 0.6323, ept: 93.9075
    Epoch [16/50], Val Losses: mse: 3.3578, mae: 0.8133, huber: 0.4957, swd: 0.6294, ept: 93.5935
    Epoch [16/50], Test Losses: mse: 2.9521, mae: 0.7673, huber: 0.4556, swd: 0.4295, ept: 93.9366
      Epoch 16 composite train-obj: 0.453131
            Val objective improved 0.6130 → 0.4957, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 2.6796, mae: 0.6625, huber: 0.3772, swd: 0.4707, ept: 94.4220
    Epoch [17/50], Val Losses: mse: 4.2150, mae: 0.9584, huber: 0.6329, swd: 0.7117, ept: 92.5596
    Epoch [17/50], Test Losses: mse: 3.6234, mae: 0.9420, huber: 0.6137, swd: 0.6053, ept: 92.9487
      Epoch 17 composite train-obj: 0.377223
            No improvement (0.6329), counter 1/5
    Epoch [18/50], Train Losses: mse: 2.7078, mae: 0.6671, huber: 0.3814, swd: 0.4603, ept: 94.3744
    Epoch [18/50], Val Losses: mse: 11.7510, mae: 1.6471, huber: 1.2825, swd: 1.4334, ept: 88.2351
    Epoch [18/50], Test Losses: mse: 11.5275, mae: 1.6157, huber: 1.2551, swd: 1.2196, ept: 88.2961
      Epoch 18 composite train-obj: 0.381431
            No improvement (1.2825), counter 2/5
    Epoch [19/50], Train Losses: mse: 4.9787, mae: 0.8829, huber: 0.5735, swd: 0.8971, ept: 93.4530
    Epoch [19/50], Val Losses: mse: 17.9785, mae: 1.9561, huber: 1.5784, swd: 1.1985, ept: 85.6215
    Epoch [19/50], Test Losses: mse: 16.4245, mae: 1.8585, huber: 1.4855, swd: 1.2386, ept: 86.4235
      Epoch 19 composite train-obj: 0.573453
            No improvement (1.5784), counter 3/5
    Epoch [20/50], Train Losses: mse: 5.9829, mae: 1.0274, huber: 0.7011, swd: 1.1329, ept: 92.2905
    Epoch [20/50], Val Losses: mse: 28.8313, mae: 2.6784, huber: 2.2640, swd: 5.0603, ept: 83.5185
    Epoch [20/50], Test Losses: mse: 24.6636, mae: 2.3933, huber: 1.9856, swd: 5.0838, ept: 85.0476
      Epoch 20 composite train-obj: 0.701108
            No improvement (2.2640), counter 4/5
    Epoch [21/50], Train Losses: mse: 7.4299, mae: 1.2233, huber: 0.8832, swd: 1.6709, ept: 90.0600
    Epoch [21/50], Val Losses: mse: 11.4682, mae: 1.5791, huber: 1.2031, swd: 1.3444, ept: 89.4894
    Epoch [21/50], Test Losses: mse: 10.1868, mae: 1.4933, huber: 1.1219, swd: 1.0250, ept: 90.2282
      Epoch 21 composite train-obj: 0.883236
    Epoch [21/50], Test Losses: mse: 2.9520, mae: 0.7673, huber: 0.4556, swd: 0.4294, ept: 93.9362
    Best round's Test MSE: 2.9521, MAE: 0.7673, SWD: 0.4295
    Best round's Validation MSE: 3.3578, MAE: 0.8133, SWD: 0.6294
    Best round's Test verification MSE : 2.9520, MAE: 0.7673, SWD: 0.4294
    Time taken: 172.27 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 34.7427, mae: 3.6030, huber: 3.1593, swd: 16.1295, ept: 62.4697
    Epoch [1/50], Val Losses: mse: 17.5038, mae: 2.4897, huber: 2.0576, swd: 3.7484, ept: 77.1374
    Epoch [1/50], Test Losses: mse: 15.5476, mae: 2.3905, huber: 1.9604, swd: 3.6417, ept: 77.8234
      Epoch 1 composite train-obj: 3.159347
            Val objective improved inf → 2.0576, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.1091, mae: 1.5418, huber: 1.1524, swd: 1.6614, ept: 88.1304
    Epoch [2/50], Val Losses: mse: 15.6116, mae: 2.3823, huber: 1.9551, swd: 1.8420, ept: 80.8610
    Epoch [2/50], Test Losses: mse: 14.9848, mae: 2.3617, huber: 1.9344, swd: 1.6748, ept: 80.8081
      Epoch 2 composite train-obj: 1.152375
            Val objective improved 2.0576 → 1.9551, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.2466, mae: 1.2029, huber: 0.8396, swd: 1.0527, ept: 91.3961
    Epoch [3/50], Val Losses: mse: 13.2976, mae: 1.7928, huber: 1.4041, swd: 1.1885, ept: 86.9256
    Epoch [3/50], Test Losses: mse: 13.1020, mae: 1.7656, huber: 1.3790, swd: 1.1750, ept: 87.2473
      Epoch 3 composite train-obj: 0.839649
            Val objective improved 1.9551 → 1.4041, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.4365, mae: 1.0736, huber: 0.7246, swd: 0.8112, ept: 92.4467
    Epoch [4/50], Val Losses: mse: 19.9258, mae: 2.1535, huber: 1.7609, swd: 1.0870, ept: 84.0973
    Epoch [4/50], Test Losses: mse: 18.4913, mae: 2.0560, huber: 1.6683, swd: 1.0353, ept: 84.9989
      Epoch 4 composite train-obj: 0.724563
            No improvement (1.7609), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.6147, mae: 1.1870, huber: 0.8327, swd: 0.9978, ept: 91.4609
    Epoch [5/50], Val Losses: mse: 7.3726, mae: 1.2821, huber: 0.9201, swd: 1.2282, ept: 91.4082
    Epoch [5/50], Test Losses: mse: 7.2624, mae: 1.2853, huber: 0.9215, swd: 1.2308, ept: 91.5622
      Epoch 5 composite train-obj: 0.832721
            Val objective improved 1.4041 → 0.9201, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.7710, mae: 0.8463, huber: 0.5259, swd: 0.6089, ept: 93.6904
    Epoch [6/50], Val Losses: mse: 4.9976, mae: 1.1289, huber: 0.7740, swd: 0.7428, ept: 91.5586
    Epoch [6/50], Test Losses: mse: 4.5723, mae: 1.1037, huber: 0.7492, swd: 0.5382, ept: 91.7372
      Epoch 6 composite train-obj: 0.525879
            Val objective improved 0.9201 → 0.7740, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.3908, mae: 0.7875, huber: 0.4763, swd: 0.5452, ept: 94.0106
    Epoch [7/50], Val Losses: mse: 6.8703, mae: 1.2964, huber: 0.9197, swd: 0.8936, ept: 91.5532
    Epoch [7/50], Test Losses: mse: 6.3956, mae: 1.2705, huber: 0.8954, swd: 0.8382, ept: 91.7282
      Epoch 7 composite train-obj: 0.476313
            No improvement (0.9197), counter 1/5
    Epoch [8/50], Train Losses: mse: 3.6028, mae: 0.8228, huber: 0.5063, swd: 0.5380, ept: 93.8403
    Epoch [8/50], Val Losses: mse: 3.2397, mae: 0.8826, huber: 0.5546, swd: 0.7904, ept: 93.6747
    Epoch [8/50], Test Losses: mse: 2.4710, mae: 0.8534, huber: 0.5235, swd: 0.5581, ept: 93.7937
      Epoch 8 composite train-obj: 0.506317
            Val objective improved 0.7740 → 0.5546, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 2.6788, mae: 0.6897, huber: 0.3956, swd: 0.4269, ept: 94.3255
    Epoch [9/50], Val Losses: mse: 12.1921, mae: 1.5533, huber: 1.1801, swd: 1.0862, ept: 90.7047
    Epoch [9/50], Test Losses: mse: 11.4619, mae: 1.5179, huber: 1.1463, swd: 0.9418, ept: 90.7297
      Epoch 9 composite train-obj: 0.395558
            No improvement (1.1801), counter 1/5
    Epoch [10/50], Train Losses: mse: 4.4364, mae: 0.9099, huber: 0.5858, swd: 0.6614, ept: 93.4408
    Epoch [10/50], Val Losses: mse: 7.4688, mae: 1.2379, huber: 0.8780, swd: 1.0944, ept: 92.6739
    Epoch [10/50], Test Losses: mse: 8.2523, mae: 1.2921, huber: 0.9295, swd: 1.1371, ept: 92.6257
      Epoch 10 composite train-obj: 0.585777
            No improvement (0.8780), counter 2/5
    Epoch [11/50], Train Losses: mse: 4.3383, mae: 0.8620, huber: 0.5455, swd: 0.6222, ept: 93.7399
    Epoch [11/50], Val Losses: mse: 4.3251, mae: 1.0239, huber: 0.6776, swd: 0.7867, ept: 92.4537
    Epoch [11/50], Test Losses: mse: 3.8438, mae: 0.9974, huber: 0.6549, swd: 0.6753, ept: 92.6205
      Epoch 11 composite train-obj: 0.545468
            No improvement (0.6776), counter 3/5
    Epoch [12/50], Train Losses: mse: 3.6542, mae: 0.8094, huber: 0.4996, swd: 0.6680, ept: 93.7047
    Epoch [12/50], Val Losses: mse: 4.8100, mae: 1.0404, huber: 0.7099, swd: 0.6715, ept: 92.1114
    Epoch [12/50], Test Losses: mse: 4.5379, mae: 1.0263, huber: 0.6959, swd: 0.5510, ept: 92.3088
      Epoch 12 composite train-obj: 0.499590
            No improvement (0.7099), counter 4/5
    Epoch [13/50], Train Losses: mse: 3.6845, mae: 0.8125, huber: 0.5022, swd: 0.6939, ept: 93.5648
    Epoch [13/50], Val Losses: mse: 5.2905, mae: 1.2147, huber: 0.8433, swd: 1.0440, ept: 91.8681
    Epoch [13/50], Test Losses: mse: 4.9917, mae: 1.2156, huber: 0.8428, swd: 1.1222, ept: 91.8877
      Epoch 13 composite train-obj: 0.502216
    Epoch [13/50], Test Losses: mse: 2.4704, mae: 0.8533, huber: 0.5234, swd: 0.5582, ept: 93.7952
    Best round's Test MSE: 2.4710, MAE: 0.8534, SWD: 0.5581
    Best round's Validation MSE: 3.2397, MAE: 0.8826, SWD: 0.7904
    Best round's Test verification MSE : 2.4704, MAE: 0.8533, SWD: 0.5582
    Time taken: 109.19 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred96_20250510_0857)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 2.3654 ± 0.5275
      mae: 0.7475 ± 0.0955
      huber: 0.4369 ± 0.0794
      swd: 0.5015 ± 0.0536
      ept: 94.2000 ± 0.4771
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 2.8819 ± 0.5914
      mae: 0.7817 ± 0.0978
      huber: 0.4693 ± 0.0825
      swd: 0.7039 ± 0.0663
      ept: 93.9426 ± 0.4376
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 585.09 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred96_20250510_0857
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    


```python
importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  
    seq_len=336,
    pred_len=96,
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
    ablate_shift_inside_scale=True,
    householder_reflects_latent = 4,
    householder_reflects_data = 8,
    mixing_strategy='delay_only',
    # single_magnitude_for_shift=True,
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False # changed

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
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    Train set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 282
    Validation Batches: 38
    Test Batches: 78
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 39.5042, mae: 3.9858, huber: 3.5350, swd: 20.9763, target_std: 13.6425
    Epoch [1/50], Val Losses: mse: 20.2738, mae: 2.7349, huber: 2.2963, swd: 3.5567, target_std: 13.7293
    Epoch [1/50], Test Losses: mse: 19.2256, mae: 2.6982, huber: 2.2593, swd: 3.5408, target_std: 13.1684
      Epoch 1 composite train-obj: 3.535003
            Val objective improved inf → 2.2963, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.2693, mae: 1.5545, huber: 1.1634, swd: 1.5851, target_std: 13.6706
    Epoch [2/50], Val Losses: mse: 8.8239, mae: 1.5756, huber: 1.1777, swd: 1.2257, target_std: 13.7293
    Epoch [2/50], Test Losses: mse: 7.5734, mae: 1.4844, huber: 1.0908, swd: 1.1161, target_std: 13.1684
      Epoch 2 composite train-obj: 1.163384
            Val objective improved 2.2963 → 1.1777, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.0416, mae: 1.1526, huber: 0.7919, swd: 0.9723, target_std: 13.6427
    Epoch [3/50], Val Losses: mse: 8.2757, mae: 1.5356, huber: 1.1423, swd: 1.6489, target_std: 13.7293
    Epoch [3/50], Test Losses: mse: 7.4225, mae: 1.4812, huber: 1.0870, swd: 1.5505, target_std: 13.1684
      Epoch 3 composite train-obj: 0.791857
            Val objective improved 1.1777 → 1.1423, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.8756, mae: 0.9845, huber: 0.6445, swd: 0.7774, target_std: 13.6677
    Epoch [4/50], Val Losses: mse: 12.8268, mae: 1.8172, huber: 1.4332, swd: 1.2529, target_std: 13.7293
    Epoch [4/50], Test Losses: mse: 11.3399, mae: 1.6913, huber: 1.3135, swd: 1.2318, target_std: 13.1684
      Epoch 4 composite train-obj: 0.644462
            No improvement (1.4332), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.6912, mae: 0.9524, huber: 0.6184, swd: 0.7061, target_std: 13.6678
    Epoch [5/50], Val Losses: mse: 6.8122, mae: 1.2441, huber: 0.8913, swd: 0.7281, target_std: 13.7293
    Epoch [5/50], Test Losses: mse: 6.1606, mae: 1.2196, huber: 0.8655, swd: 0.5924, target_std: 13.1684
      Epoch 5 composite train-obj: 0.618426
            Val objective improved 1.1423 → 0.8913, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.8556, mae: 0.8146, huber: 0.5024, swd: 0.5916, target_std: 13.6423
    Epoch [6/50], Val Losses: mse: 14.3393, mae: 1.6907, huber: 1.3298, swd: 1.7062, target_std: 13.7293
    Epoch [6/50], Test Losses: mse: 12.9718, mae: 1.6300, huber: 1.2723, swd: 1.5536, target_std: 13.1684
      Epoch 6 composite train-obj: 0.502417
            No improvement (1.3298), counter 1/5
    Epoch [7/50], Train Losses: mse: 4.6179, mae: 0.8902, huber: 0.5714, swd: 0.6931, target_std: 13.6662
    Epoch [7/50], Val Losses: mse: 6.8526, mae: 1.2761, huber: 0.9251, swd: 1.1853, target_std: 13.7293
    Epoch [7/50], Test Losses: mse: 6.5358, mae: 1.2687, huber: 0.9153, swd: 1.1475, target_std: 13.1684
      Epoch 7 composite train-obj: 0.571419
            No improvement (0.9251), counter 2/5
    Epoch [8/50], Train Losses: mse: 3.9477, mae: 0.8112, huber: 0.5022, swd: 0.6178, target_std: 13.6766
    Epoch [8/50], Val Losses: mse: 3.8098, mae: 1.0175, huber: 0.6746, swd: 0.8879, target_std: 13.7293
    Epoch [8/50], Test Losses: mse: 3.5475, mae: 0.9691, huber: 0.6310, swd: 0.7692, target_std: 13.1684
      Epoch 8 composite train-obj: 0.502184
            Val objective improved 0.8913 → 0.6746, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.1753, mae: 0.7166, huber: 0.4211, swd: 0.4950, target_std: 13.6479
    Epoch [9/50], Val Losses: mse: 4.0527, mae: 0.9190, huber: 0.5957, swd: 0.7458, target_std: 13.7293
    Epoch [9/50], Test Losses: mse: 3.8097, mae: 0.9092, huber: 0.5839, swd: 0.5306, target_std: 13.1684
      Epoch 9 composite train-obj: 0.421096
            Val objective improved 0.6746 → 0.5957, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 2.9116, mae: 0.6750, huber: 0.3868, swd: 0.4411, target_std: 13.6735
    Epoch [10/50], Val Losses: mse: 3.1324, mae: 0.8828, huber: 0.5545, swd: 0.7424, target_std: 13.7293
    Epoch [10/50], Test Losses: mse: 2.7819, mae: 0.8572, huber: 0.5316, swd: 0.6031, target_std: 13.1684
      Epoch 10 composite train-obj: 0.386836
            Val objective improved 0.5957 → 0.5545, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 2.6658, mae: 0.6479, huber: 0.3644, swd: 0.4031, target_std: 13.6613
    Epoch [11/50], Val Losses: mse: 4.2418, mae: 0.9912, huber: 0.6628, swd: 0.7831, target_std: 13.7293
    Epoch [11/50], Test Losses: mse: 4.1169, mae: 0.9615, huber: 0.6368, swd: 0.6555, target_std: 13.1684
      Epoch 11 composite train-obj: 0.364394
            No improvement (0.6628), counter 1/5
    Epoch [12/50], Train Losses: mse: 2.6762, mae: 0.6314, huber: 0.3539, swd: 0.3879, target_std: 13.6404
    Epoch [12/50], Val Losses: mse: 12.7519, mae: 1.5286, huber: 1.1730, swd: 3.1605, target_std: 13.7293
    Epoch [12/50], Test Losses: mse: 12.2735, mae: 1.4931, huber: 1.1374, swd: 2.8639, target_std: 13.1684
      Epoch 12 composite train-obj: 0.353865
            No improvement (1.1730), counter 2/5
    Epoch [13/50], Train Losses: mse: 3.9606, mae: 0.8155, huber: 0.5087, swd: 0.6477, target_std: 13.6497
    Epoch [13/50], Val Losses: mse: 6.0266, mae: 1.0925, huber: 0.7581, swd: 0.8515, target_std: 13.7293
    Epoch [13/50], Test Losses: mse: 5.8724, mae: 1.1028, huber: 0.7665, swd: 0.7167, target_std: 13.1684
      Epoch 13 composite train-obj: 0.508702
            No improvement (0.7581), counter 3/5
    Epoch [14/50], Train Losses: mse: 2.7154, mae: 0.6598, huber: 0.3761, swd: 0.4177, target_std: 13.6469
    Epoch [14/50], Val Losses: mse: 4.3130, mae: 1.0961, huber: 0.7409, swd: 0.7284, target_std: 13.7293
    Epoch [14/50], Test Losses: mse: 4.0635, mae: 1.0863, huber: 0.7311, swd: 0.5800, target_std: 13.1684
      Epoch 14 composite train-obj: 0.376080
            No improvement (0.7409), counter 4/5
    Epoch [15/50], Train Losses: mse: 2.7466, mae: 0.6829, huber: 0.3943, swd: 0.4151, target_std: 13.6467
    Epoch [15/50], Val Losses: mse: 2.3570, mae: 0.7244, huber: 0.4197, swd: 0.5764, target_std: 13.7293
    Epoch [15/50], Test Losses: mse: 2.2941, mae: 0.7068, huber: 0.4057, swd: 0.4914, target_std: 13.1684
      Epoch 15 composite train-obj: 0.394253
            Val objective improved 0.5545 → 0.4197, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 2.1757, mae: 0.5784, huber: 0.3104, swd: 0.3207, target_std: 13.6489
    Epoch [16/50], Val Losses: mse: 3.4967, mae: 0.8543, huber: 0.5392, swd: 0.6196, target_std: 13.7293
    Epoch [16/50], Test Losses: mse: 3.6612, mae: 0.8680, huber: 0.5514, swd: 0.5071, target_std: 13.1684
      Epoch 16 composite train-obj: 0.310369
            No improvement (0.5392), counter 1/5
    Epoch [17/50], Train Losses: mse: 2.2442, mae: 0.5945, huber: 0.3226, swd: 0.3209, target_std: 13.6816
    Epoch [17/50], Val Losses: mse: 2.5382, mae: 0.7681, huber: 0.4564, swd: 0.5157, target_std: 13.7293
    Epoch [17/50], Test Losses: mse: 2.2015, mae: 0.7009, huber: 0.4023, swd: 0.4086, target_std: 13.1684
      Epoch 17 composite train-obj: 0.322603
            No improvement (0.4564), counter 2/5
    Epoch [18/50], Train Losses: mse: 1.9714, mae: 0.5507, huber: 0.2888, swd: 0.2846, target_std: 13.6582
    Epoch [18/50], Val Losses: mse: 9.7070, mae: 1.4326, huber: 1.0846, swd: 0.6465, target_std: 13.7293
    Epoch [18/50], Test Losses: mse: 9.9266, mae: 1.4391, huber: 1.0935, swd: 0.6284, target_std: 13.1684
      Epoch 18 composite train-obj: 0.288849
            No improvement (1.0846), counter 3/5
    Epoch [19/50], Train Losses: mse: 4.6914, mae: 0.8424, huber: 0.5427, swd: 0.6452, target_std: 13.6613
    Epoch [19/50], Val Losses: mse: 4.2521, mae: 0.9719, huber: 0.6431, swd: 0.5548, target_std: 13.7293
    Epoch [19/50], Test Losses: mse: 4.2467, mae: 0.9714, huber: 0.6446, swd: 0.4527, target_std: 13.1684
      Epoch 19 composite train-obj: 0.542694
            No improvement (0.6431), counter 4/5
    Epoch [20/50], Train Losses: mse: 2.6347, mae: 0.6031, huber: 0.3344, swd: 0.3426, target_std: 13.6492
    Epoch [20/50], Val Losses: mse: 3.1289, mae: 0.8663, huber: 0.5469, swd: 0.4511, target_std: 13.7293
    Epoch [20/50], Test Losses: mse: 3.0187, mae: 0.8630, huber: 0.5421, swd: 0.3348, target_std: 13.1684
      Epoch 20 composite train-obj: 0.334395
    Epoch [20/50], Test Losses: mse: 2.2971, mae: 0.7066, huber: 0.4056, swd: 0.4931, target_std: 13.1684
    Best round's Test MSE: 2.2941, MAE: 0.7068, SWD: 0.4914
    Best round's Validation MSE: 2.3570, MAE: 0.7244, SWD: 0.5764
    Best round's Test verification MSE : 2.2971, MAE: 0.7066, SWD: 0.4931
    Time taken: 211.18 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 39.1391, mae: 3.9663, huber: 3.5150, swd: 20.4804, target_std: 13.6437
    Epoch [1/50], Val Losses: mse: 20.1340, mae: 2.6741, huber: 2.2381, swd: 2.8731, target_std: 13.7293
    Epoch [1/50], Test Losses: mse: 19.4630, mae: 2.6278, huber: 2.1931, swd: 2.9399, target_std: 13.1684
      Epoch 1 composite train-obj: 3.514984
            Val objective improved inf → 2.2381, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.8523, mae: 1.6242, huber: 1.2276, swd: 1.6400, target_std: 13.6708
    Epoch [2/50], Val Losses: mse: 10.5635, mae: 1.6717, huber: 1.2739, swd: 1.6061, target_std: 13.7293
    Epoch [2/50], Test Losses: mse: 8.7400, mae: 1.5934, huber: 1.1972, swd: 1.4237, target_std: 13.1684
      Epoch 2 composite train-obj: 1.227557
            Val objective improved 2.2381 → 1.2739, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.5731, mae: 1.2096, huber: 0.8427, swd: 1.0551, target_std: 13.6460
    Epoch [3/50], Val Losses: mse: 8.4346, mae: 1.5855, huber: 1.1972, swd: 1.1931, target_std: 13.7293
    Epoch [3/50], Test Losses: mse: 7.6223, mae: 1.5684, huber: 1.1769, swd: 0.8980, target_std: 13.1684
      Epoch 3 composite train-obj: 0.842698
            Val objective improved 1.2739 → 1.1972, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.1015, mae: 1.0178, huber: 0.6729, swd: 0.8241, target_std: 13.6454
    Epoch [4/50], Val Losses: mse: 7.2649, mae: 1.2722, huber: 0.9072, swd: 0.7498, target_std: 13.7293
    Epoch [4/50], Test Losses: mse: 6.0678, mae: 1.2022, huber: 0.8390, swd: 0.6671, target_std: 13.1684
      Epoch 4 composite train-obj: 0.672925
            Val objective improved 1.1972 → 0.9072, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.5567, mae: 0.9328, huber: 0.5990, swd: 0.7115, target_std: 13.6758
    Epoch [5/50], Val Losses: mse: 3.5671, mae: 0.9081, huber: 0.5746, swd: 0.6793, target_std: 13.7293
    Epoch [5/50], Test Losses: mse: 3.4867, mae: 0.8866, huber: 0.5563, swd: 0.5403, target_std: 13.1684
      Epoch 5 composite train-obj: 0.599018
            Val objective improved 0.9072 → 0.5746, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.4944, mae: 0.7908, huber: 0.4768, swd: 0.5566, target_std: 13.6773
    Epoch [6/50], Val Losses: mse: 5.0983, mae: 1.1524, huber: 0.7901, swd: 1.0662, target_std: 13.7293
    Epoch [6/50], Test Losses: mse: 4.8932, mae: 1.1170, huber: 0.7604, swd: 0.8548, target_std: 13.1684
      Epoch 6 composite train-obj: 0.476773
            No improvement (0.7901), counter 1/5
    Epoch [7/50], Train Losses: mse: 3.2306, mae: 0.7672, huber: 0.4568, swd: 0.5109, target_std: 13.6427
    Epoch [7/50], Val Losses: mse: 5.6718, mae: 1.2028, huber: 0.8525, swd: 0.9280, target_std: 13.7293
    Epoch [7/50], Test Losses: mse: 5.3683, mae: 1.1880, huber: 0.8382, swd: 0.7707, target_std: 13.1684
      Epoch 7 composite train-obj: 0.456826
            No improvement (0.8525), counter 2/5
    Epoch [8/50], Train Losses: mse: 3.1209, mae: 0.7415, huber: 0.4376, swd: 0.4811, target_std: 13.6663
    Epoch [8/50], Val Losses: mse: 3.1763, mae: 0.9794, huber: 0.6295, swd: 0.6099, target_std: 13.7293
    Epoch [8/50], Test Losses: mse: 3.1890, mae: 0.9254, huber: 0.5838, swd: 0.5129, target_std: 13.1684
      Epoch 8 composite train-obj: 0.437586
            No improvement (0.6295), counter 3/5
    Epoch [9/50], Train Losses: mse: 2.7320, mae: 0.6927, huber: 0.3964, swd: 0.4142, target_std: 13.6422
    Epoch [9/50], Val Losses: mse: 4.0404, mae: 0.9448, huber: 0.6174, swd: 0.6697, target_std: 13.7293
    Epoch [9/50], Test Losses: mse: 4.5993, mae: 0.9802, huber: 0.6509, swd: 0.6340, target_std: 13.1684
      Epoch 9 composite train-obj: 0.396388
            No improvement (0.6174), counter 4/5
    Epoch [10/50], Train Losses: mse: 2.7893, mae: 0.6990, huber: 0.4026, swd: 0.4314, target_std: 13.6454
    Epoch [10/50], Val Losses: mse: 6.4453, mae: 1.1901, huber: 0.8549, swd: 1.0149, target_std: 13.7293
    Epoch [10/50], Test Losses: mse: 6.1640, mae: 1.1824, huber: 0.8445, swd: 0.9282, target_std: 13.1684
      Epoch 10 composite train-obj: 0.402573
    Epoch [10/50], Test Losses: mse: 3.4880, mae: 0.8869, huber: 0.5566, swd: 0.5350, target_std: 13.1684
    Best round's Test MSE: 3.4867, MAE: 0.8866, SWD: 0.5403
    Best round's Validation MSE: 3.5671, MAE: 0.9081, SWD: 0.6793
    Best round's Test verification MSE : 3.4880, MAE: 0.8869, SWD: 0.5350
    Time taken: 103.19 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 40.3800, mae: 4.0509, huber: 3.5991, swd: 20.5085, target_std: 13.6714
    Epoch [1/50], Val Losses: mse: 18.8691, mae: 2.5185, huber: 2.0865, swd: 2.7167, target_std: 13.7293
    Epoch [1/50], Test Losses: mse: 16.7342, mae: 2.4024, huber: 1.9723, swd: 2.5512, target_std: 13.1684
      Epoch 1 composite train-obj: 3.599116
            Val objective improved inf → 2.0865, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.0602, mae: 1.5394, huber: 1.1474, swd: 1.4877, target_std: 13.6547
    Epoch [2/50], Val Losses: mse: 11.5793, mae: 1.8482, huber: 1.4464, swd: 1.3626, target_std: 13.7293
    Epoch [2/50], Test Losses: mse: 11.0888, mae: 1.8286, huber: 1.4257, swd: 1.2970, target_std: 13.1684
      Epoch 2 composite train-obj: 1.147361
            Val objective improved 2.0865 → 1.4464, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.3466, mae: 1.1957, huber: 0.8313, swd: 0.9955, target_std: 13.6721
    Epoch [3/50], Val Losses: mse: 5.9432, mae: 1.2809, huber: 0.9081, swd: 0.9870, target_std: 13.7293
    Epoch [3/50], Test Losses: mse: 5.7095, mae: 1.2558, huber: 0.8847, swd: 0.9073, target_std: 13.1684
      Epoch 3 composite train-obj: 0.831301
            Val objective improved 1.4464 → 0.9081, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6358, mae: 0.9623, huber: 0.6226, swd: 0.7071, target_std: 13.6709
    Epoch [4/50], Val Losses: mse: 4.1110, mae: 1.0583, huber: 0.7010, swd: 0.8049, target_std: 13.7293
    Epoch [4/50], Test Losses: mse: 3.9616, mae: 1.0108, huber: 0.6587, swd: 0.7257, target_std: 13.1684
      Epoch 4 composite train-obj: 0.622558
            Val objective improved 0.9081 → 0.7010, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 3.8968, mae: 0.8594, huber: 0.5331, swd: 0.6002, target_std: 13.6482
    Epoch [5/50], Val Losses: mse: 6.3154, mae: 1.1470, huber: 0.7953, swd: 0.7745, target_std: 13.7293
    Epoch [5/50], Test Losses: mse: 5.1421, mae: 1.0850, huber: 0.7333, swd: 0.7598, target_std: 13.1684
      Epoch 5 composite train-obj: 0.533102
            No improvement (0.7953), counter 1/5
    Epoch [6/50], Train Losses: mse: 3.6363, mae: 0.8263, huber: 0.5068, swd: 0.5502, target_std: 13.6654
    Epoch [6/50], Val Losses: mse: 4.2017, mae: 0.9582, huber: 0.6144, swd: 0.5960, target_std: 13.7293
    Epoch [6/50], Test Losses: mse: 3.5450, mae: 0.9139, huber: 0.5742, swd: 0.5204, target_std: 13.1684
      Epoch 6 composite train-obj: 0.506822
            Val objective improved 0.7010 → 0.6144, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.1564, mae: 0.7549, huber: 0.4469, swd: 0.4767, target_std: 13.6490
    Epoch [7/50], Val Losses: mse: 15.9482, mae: 1.7682, huber: 1.4079, swd: 1.5536, target_std: 13.7293
    Epoch [7/50], Test Losses: mse: 14.7670, mae: 1.7073, huber: 1.3497, swd: 1.4914, target_std: 13.1684
      Epoch 7 composite train-obj: 0.446913
            No improvement (1.4079), counter 1/5
    Epoch [8/50], Train Losses: mse: 5.9567, mae: 1.0433, huber: 0.7080, swd: 0.7952, target_std: 13.6494
    Epoch [8/50], Val Losses: mse: 9.9260, mae: 1.4571, huber: 1.1000, swd: 1.1547, target_std: 13.7293
    Epoch [8/50], Test Losses: mse: 9.2427, mae: 1.4237, huber: 1.0668, swd: 1.0712, target_std: 13.1684
      Epoch 8 composite train-obj: 0.707976
            No improvement (1.1000), counter 2/5
    Epoch [9/50], Train Losses: mse: 4.8662, mae: 0.9159, huber: 0.5944, swd: 0.6554, target_std: 13.6493
    Epoch [9/50], Val Losses: mse: 6.1782, mae: 1.2543, huber: 0.8957, swd: 0.9240, target_std: 13.7293
    Epoch [9/50], Test Losses: mse: 5.8328, mae: 1.2141, huber: 0.8544, swd: 0.7810, target_std: 13.1684
      Epoch 9 composite train-obj: 0.594352
            No improvement (0.8957), counter 3/5
    Epoch [10/50], Train Losses: mse: 3.5803, mae: 0.7613, huber: 0.4589, swd: 0.5116, target_std: 13.6439
    Epoch [10/50], Val Losses: mse: 3.8678, mae: 0.9389, huber: 0.6084, swd: 0.6491, target_std: 13.7293
    Epoch [10/50], Test Losses: mse: 4.4082, mae: 0.9628, huber: 0.6306, swd: 0.6166, target_std: 13.1684
      Epoch 10 composite train-obj: 0.458904
            Val objective improved 0.6144 → 0.6084, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 3.2676, mae: 0.7259, huber: 0.4283, swd: 0.4620, target_std: 13.6624
    Epoch [11/50], Val Losses: mse: 6.8992, mae: 1.2894, huber: 0.9302, swd: 0.8620, target_std: 13.7293
    Epoch [11/50], Test Losses: mse: 6.7388, mae: 1.2730, huber: 0.9145, swd: 0.7101, target_std: 13.1684
      Epoch 11 composite train-obj: 0.428282
            No improvement (0.9302), counter 1/5
    Epoch [12/50], Train Losses: mse: 3.1352, mae: 0.7167, huber: 0.4211, swd: 0.4282, target_std: 13.6756
    Epoch [12/50], Val Losses: mse: 2.8490, mae: 0.8224, huber: 0.5048, swd: 0.5522, target_std: 13.7293
    Epoch [12/50], Test Losses: mse: 2.7307, mae: 0.7701, huber: 0.4613, swd: 0.3758, target_std: 13.1684
      Epoch 12 composite train-obj: 0.421110
            Val objective improved 0.6084 → 0.5048, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 2.4087, mae: 0.6050, huber: 0.3300, swd: 0.3473, target_std: 13.6731
    Epoch [13/50], Val Losses: mse: 3.0630, mae: 0.7713, huber: 0.4708, swd: 0.4188, target_std: 13.7293
    Epoch [13/50], Test Losses: mse: 3.0712, mae: 0.7610, huber: 0.4625, swd: 0.3662, target_std: 13.1684
      Epoch 13 composite train-obj: 0.329989
            Val objective improved 0.5048 → 0.4708, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 2.3335, mae: 0.6106, huber: 0.3332, swd: 0.3345, target_std: 13.6458
    Epoch [14/50], Val Losses: mse: 5.1200, mae: 1.0052, huber: 0.6885, swd: 0.5847, target_std: 13.7293
    Epoch [14/50], Test Losses: mse: 5.1911, mae: 0.9945, huber: 0.6794, swd: 0.4956, target_std: 13.1684
      Epoch 14 composite train-obj: 0.333217
            No improvement (0.6885), counter 1/5
    Epoch [15/50], Train Losses: mse: 2.6480, mae: 0.6515, huber: 0.3694, swd: 0.3533, target_std: 13.6472
    Epoch [15/50], Val Losses: mse: 2.5709, mae: 0.8034, huber: 0.4820, swd: 0.5228, target_std: 13.7293
    Epoch [15/50], Test Losses: mse: 2.5637, mae: 0.7710, huber: 0.4554, swd: 0.4654, target_std: 13.1684
      Epoch 15 composite train-obj: 0.369426
            No improvement (0.4820), counter 2/5
    Epoch [16/50], Train Losses: mse: 2.0840, mae: 0.5672, huber: 0.3002, swd: 0.2882, target_std: 13.6607
    Epoch [16/50], Val Losses: mse: 4.1206, mae: 0.9290, huber: 0.6053, swd: 0.4960, target_std: 13.7293
    Epoch [16/50], Test Losses: mse: 3.9139, mae: 0.9125, huber: 0.5893, swd: 0.4441, target_std: 13.1684
      Epoch 16 composite train-obj: 0.300247
            No improvement (0.6053), counter 3/5
    Epoch [17/50], Train Losses: mse: 2.5320, mae: 0.6157, huber: 0.3425, swd: 0.3254, target_std: 13.6511
    Epoch [17/50], Val Losses: mse: 4.8434, mae: 0.9536, huber: 0.6480, swd: 0.5658, target_std: 13.7293
    Epoch [17/50], Test Losses: mse: 4.9507, mae: 0.9578, huber: 0.6530, swd: 0.4747, target_std: 13.1684
      Epoch 17 composite train-obj: 0.342482
            No improvement (0.6480), counter 4/5
    Epoch [18/50], Train Losses: mse: 2.5016, mae: 0.6166, huber: 0.3429, swd: 0.3457, target_std: 13.6483
    Epoch [18/50], Val Losses: mse: 12.0755, mae: 1.3940, huber: 1.0447, swd: 1.9774, target_std: 13.7293
    Epoch [18/50], Test Losses: mse: 12.6530, mae: 1.3859, huber: 1.0426, swd: 2.1250, target_std: 13.1684
      Epoch 18 composite train-obj: 0.342947
    Epoch [18/50], Test Losses: mse: 3.0690, mae: 0.7610, huber: 0.4625, swd: 0.3669, target_std: 13.1684
    Best round's Test MSE: 3.0712, MAE: 0.7610, SWD: 0.3662
    Best round's Validation MSE: 3.0630, MAE: 0.7713, SWD: 0.4188
    Best round's Test verification MSE : 3.0690, MAE: 0.7610, SWD: 0.3669
    Time taken: 188.70 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred96_20250506_1331)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 2.9507 ± 0.4943
      mae: 0.7848 ± 0.0753
      huber: 0.4749 ± 0.0621
      swd: 0.4660 ± 0.0733
      target_std: 13.1684 ± 0.0000
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 2.9957 ± 0.4963
      mae: 0.8013 ± 0.0779
      huber: 0.4884 ± 0.0645
      swd: 0.5582 ± 0.1071
      target_std: 13.7293 ± 0.0000
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 503.17 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred96_20250506_1331
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 336-196
##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=196,
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
    # single_magnitude_for_shift=True,
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False,
    

)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 281
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 281
    Validation Batches: 37
    Test Batches: 78
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 51.0567, mae: 4.8908, huber: 4.4279, swd: 24.4218, ept: 72.8115
    Epoch [1/50], Val Losses: mse: 29.7042, mae: 3.3643, huber: 2.9213, swd: 4.7738, ept: 119.1863
    Epoch [1/50], Test Losses: mse: 29.3578, mae: 3.3650, huber: 2.9209, swd: 4.7725, ept: 116.9549
      Epoch 1 composite train-obj: 4.427924
            Val objective improved inf → 2.9213, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 21.8115, mae: 2.6859, huber: 2.2578, swd: 3.0562, ept: 141.0873
    Epoch [2/50], Val Losses: mse: 21.8252, mae: 2.6426, huber: 2.2169, swd: 2.5135, ept: 145.5370
    Epoch [2/50], Test Losses: mse: 18.6155, mae: 2.4726, huber: 2.0490, swd: 2.3747, ept: 145.6300
      Epoch 2 composite train-obj: 2.257756
            Val objective improved 2.9213 → 2.2169, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 15.4004, mae: 2.1136, huber: 1.7038, swd: 1.7031, ept: 156.2440
    Epoch [3/50], Val Losses: mse: 16.4081, mae: 2.2110, huber: 1.7949, swd: 1.5163, ept: 157.0250
    Epoch [3/50], Test Losses: mse: 13.9700, mae: 2.0575, huber: 1.6446, swd: 1.2856, ept: 159.9621
      Epoch 3 composite train-obj: 1.703836
            Val objective improved 2.2169 → 1.7949, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.3834, mae: 1.8084, huber: 1.4129, swd: 1.2828, ept: 164.6337
    Epoch [4/50], Val Losses: mse: 14.2493, mae: 1.9810, huber: 1.5794, swd: 1.3407, ept: 160.1617
    Epoch [4/50], Test Losses: mse: 11.4062, mae: 1.8174, huber: 1.4174, swd: 1.1059, ept: 163.1530
      Epoch 4 composite train-obj: 1.412866
            Val objective improved 1.7949 → 1.5794, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.8188, mae: 1.6468, huber: 1.2601, swd: 1.1032, ept: 168.7522
    Epoch [5/50], Val Losses: mse: 11.7886, mae: 1.7737, huber: 1.3798, swd: 1.3336, ept: 164.6730
    Epoch [5/50], Test Losses: mse: 9.6124, mae: 1.6619, huber: 1.2689, swd: 1.1316, ept: 167.4244
      Epoch 5 composite train-obj: 1.260064
            Val objective improved 1.5794 → 1.3798, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 9.5066, mae: 1.5016, huber: 1.1240, swd: 0.9677, ept: 172.1225
    Epoch [6/50], Val Losses: mse: 10.7617, mae: 1.6257, huber: 1.2406, swd: 1.1867, ept: 168.9442
    Epoch [6/50], Test Losses: mse: 8.0414, mae: 1.4642, huber: 1.0817, swd: 0.8796, ept: 172.8877
      Epoch 6 composite train-obj: 1.124035
            Val objective improved 1.3798 → 1.2406, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 8.2897, mae: 1.3618, huber: 0.9953, swd: 0.8635, ept: 175.5552
    Epoch [7/50], Val Losses: mse: 8.7259, mae: 1.4161, huber: 1.0479, swd: 1.1975, ept: 172.8829
    Epoch [7/50], Test Losses: mse: 7.2097, mae: 1.3187, huber: 0.9530, swd: 0.9679, ept: 176.5979
      Epoch 7 composite train-obj: 0.995316
            Val objective improved 1.2406 → 1.0479, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 7.5795, mae: 1.2865, huber: 0.9265, swd: 0.7898, ept: 177.5322
    Epoch [8/50], Val Losses: mse: 8.3937, mae: 1.3984, huber: 1.0300, swd: 1.0247, ept: 174.3595
    Epoch [8/50], Test Losses: mse: 6.6056, mae: 1.3044, huber: 0.9373, swd: 0.8059, ept: 177.3152
      Epoch 8 composite train-obj: 0.926459
            Val objective improved 1.0479 → 1.0300, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.1631, mae: 1.2330, huber: 0.8777, swd: 0.7429, ept: 178.4621
    Epoch [9/50], Val Losses: mse: 9.7480, mae: 1.4539, huber: 1.0905, swd: 0.9738, ept: 172.4773
    Epoch [9/50], Test Losses: mse: 7.4771, mae: 1.3176, huber: 0.9574, swd: 0.7706, ept: 175.3855
      Epoch 9 composite train-obj: 0.877729
            No improvement (1.0905), counter 1/5
    Epoch [10/50], Train Losses: mse: 6.9060, mae: 1.2081, huber: 0.8550, swd: 0.7134, ept: 179.6012
    Epoch [10/50], Val Losses: mse: 6.8511, mae: 1.2333, huber: 0.8805, swd: 0.7840, ept: 177.2139
    Epoch [10/50], Test Losses: mse: 5.6173, mae: 1.1663, huber: 0.8141, swd: 0.6458, ept: 180.5625
      Epoch 10 composite train-obj: 0.854980
            Val objective improved 1.0300 → 0.8805, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 6.3676, mae: 1.1502, huber: 0.8027, swd: 0.6673, ept: 180.6757
    Epoch [11/50], Val Losses: mse: 5.9567, mae: 1.1417, huber: 0.7968, swd: 0.7816, ept: 178.9564
    Epoch [11/50], Test Losses: mse: 5.2036, mae: 1.0882, huber: 0.7462, swd: 0.6925, ept: 181.4457
      Epoch 11 composite train-obj: 0.802691
            Val objective improved 0.8805 → 0.7968, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 6.0474, mae: 1.1039, huber: 0.7618, swd: 0.6210, ept: 181.4810
    Epoch [12/50], Val Losses: mse: 6.8191, mae: 1.2177, huber: 0.8662, swd: 0.7726, ept: 175.5302
    Epoch [12/50], Test Losses: mse: 5.5718, mae: 1.1407, huber: 0.7907, swd: 0.6305, ept: 179.1845
      Epoch 12 composite train-obj: 0.761838
            No improvement (0.8662), counter 1/5
    Epoch [13/50], Train Losses: mse: 6.2705, mae: 1.1261, huber: 0.7824, swd: 0.6385, ept: 181.0749
    Epoch [13/50], Val Losses: mse: 6.4174, mae: 1.1462, huber: 0.8059, swd: 0.7946, ept: 179.6643
    Epoch [13/50], Test Losses: mse: 4.7009, mae: 1.0397, huber: 0.7009, swd: 0.5854, ept: 182.8424
      Epoch 13 composite train-obj: 0.782406
            No improvement (0.8059), counter 2/5
    Epoch [14/50], Train Losses: mse: 5.6483, mae: 1.0478, huber: 0.7126, swd: 0.5813, ept: 182.3314
    Epoch [14/50], Val Losses: mse: 8.9113, mae: 1.3193, huber: 0.9682, swd: 0.8482, ept: 177.4505
    Epoch [14/50], Test Losses: mse: 6.2583, mae: 1.1765, huber: 0.8281, swd: 0.7051, ept: 180.6183
      Epoch 14 composite train-obj: 0.712617
            No improvement (0.9682), counter 3/5
    Epoch [15/50], Train Losses: mse: 6.0073, mae: 1.0856, huber: 0.7473, swd: 0.6081, ept: 181.6067
    Epoch [15/50], Val Losses: mse: 6.9721, mae: 1.2276, huber: 0.8779, swd: 0.8400, ept: 176.5844
    Epoch [15/50], Test Losses: mse: 5.3302, mae: 1.1007, huber: 0.7574, swd: 0.5941, ept: 180.7659
      Epoch 15 composite train-obj: 0.747340
            No improvement (0.8779), counter 4/5
    Epoch [16/50], Train Losses: mse: 5.4930, mae: 1.0240, huber: 0.6922, swd: 0.5589, ept: 182.7944
    Epoch [16/50], Val Losses: mse: 8.6660, mae: 1.3462, huber: 0.9887, swd: 0.7439, ept: 175.1623
    Epoch [16/50], Test Losses: mse: 7.3682, mae: 1.2905, huber: 0.9329, swd: 0.6604, ept: 176.7589
      Epoch 16 composite train-obj: 0.692173
    Epoch [16/50], Test Losses: mse: 5.2021, mae: 1.0880, huber: 0.7461, swd: 0.6919, ept: 181.4427
    Best round's Test MSE: 5.2036, MAE: 1.0882, SWD: 0.6925
    Best round's Validation MSE: 5.9567, MAE: 1.1417, SWD: 0.7816
    Best round's Test verification MSE : 5.2021, MAE: 1.0880, SWD: 0.6919
    Time taken: 129.69 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 48.9632, mae: 4.7419, huber: 4.2805, swd: 22.3479, ept: 76.9523
    Epoch [1/50], Val Losses: mse: 30.5866, mae: 3.3514, huber: 2.9097, swd: 5.2320, ept: 123.5291
    Epoch [1/50], Test Losses: mse: 29.1721, mae: 3.2874, huber: 2.8457, swd: 5.0044, ept: 122.2235
      Epoch 1 composite train-obj: 4.280475
            Val objective improved inf → 2.9097, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 20.6645, mae: 2.6047, huber: 2.1779, swd: 2.8289, ept: 143.1711
    Epoch [2/50], Val Losses: mse: 18.8416, mae: 2.4554, huber: 2.0325, swd: 2.2878, ept: 148.1848
    Epoch [2/50], Test Losses: mse: 16.1070, mae: 2.2971, huber: 1.8763, swd: 2.0248, ept: 150.8838
      Epoch 2 composite train-obj: 2.177932
            Val objective improved 2.9097 → 2.0325, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 14.4732, mae: 2.0231, huber: 1.6168, swd: 1.5990, ept: 157.4315
    Epoch [3/50], Val Losses: mse: 15.3238, mae: 2.0914, huber: 1.6820, swd: 2.2517, ept: 156.6756
    Epoch [3/50], Test Losses: mse: 12.8554, mae: 1.9399, huber: 1.5334, swd: 1.9898, ept: 160.6189
      Epoch 3 composite train-obj: 1.616759
            Val objective improved 2.0325 → 1.6820, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.1771, mae: 1.7868, huber: 1.3916, swd: 1.2688, ept: 163.9486
    Epoch [4/50], Val Losses: mse: 13.9329, mae: 1.8936, huber: 1.5000, swd: 1.6749, ept: 161.0218
    Epoch [4/50], Test Losses: mse: 11.3491, mae: 1.7405, huber: 1.3505, swd: 1.3775, ept: 163.9030
      Epoch 4 composite train-obj: 1.391600
            Val objective improved 1.6820 → 1.5000, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.3563, mae: 1.5930, huber: 1.2096, swd: 1.0622, ept: 168.9530
    Epoch [5/50], Val Losses: mse: 12.6090, mae: 1.7859, huber: 1.3950, swd: 1.6962, ept: 163.6360
    Epoch [5/50], Test Losses: mse: 9.9578, mae: 1.6269, huber: 1.2388, swd: 1.2921, ept: 167.7319
      Epoch 5 composite train-obj: 1.209598
            Val objective improved 1.5000 → 1.3950, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 9.1851, mae: 1.4601, huber: 1.0862, swd: 0.9421, ept: 171.4975
    Epoch [6/50], Val Losses: mse: 10.0716, mae: 1.5695, huber: 1.1921, swd: 1.1376, ept: 168.2574
    Epoch [6/50], Test Losses: mse: 8.1096, mae: 1.4362, huber: 1.0622, swd: 1.0039, ept: 172.7317
      Epoch 6 composite train-obj: 1.086165
            Val objective improved 1.3950 → 1.1921, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 8.8130, mae: 1.4137, huber: 1.0439, swd: 0.8995, ept: 173.4278
    Epoch [7/50], Val Losses: mse: 10.3111, mae: 1.5909, huber: 1.2136, swd: 1.0665, ept: 168.4176
    Epoch [7/50], Test Losses: mse: 8.9637, mae: 1.5047, huber: 1.1297, swd: 0.9680, ept: 171.7166
      Epoch 7 composite train-obj: 1.043855
            No improvement (1.2136), counter 1/5
    Epoch [8/50], Train Losses: mse: 8.1303, mae: 1.3442, huber: 0.9801, swd: 0.8181, ept: 175.5559
    Epoch [8/50], Val Losses: mse: 7.3729, mae: 1.3081, huber: 0.9479, swd: 0.9869, ept: 171.1932
    Epoch [8/50], Test Losses: mse: 6.1110, mae: 1.2307, huber: 0.8721, swd: 0.7869, ept: 174.8264
      Epoch 8 composite train-obj: 0.980122
            Val objective improved 1.1921 → 0.9479, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.3323, mae: 1.2543, huber: 0.8978, swd: 0.7391, ept: 176.5134
    Epoch [9/50], Val Losses: mse: 8.1815, mae: 1.4009, huber: 1.0264, swd: 0.8598, ept: 172.6985
    Epoch [9/50], Test Losses: mse: 6.6360, mae: 1.2951, huber: 0.9236, swd: 0.6875, ept: 176.6468
      Epoch 9 composite train-obj: 0.897789
            No improvement (1.0264), counter 1/5
    Epoch [10/50], Train Losses: mse: 6.8130, mae: 1.1924, huber: 0.8410, swd: 0.6922, ept: 178.2416
    Epoch [10/50], Val Losses: mse: 8.4601, mae: 1.3855, huber: 1.0180, swd: 0.9249, ept: 172.8521
    Epoch [10/50], Test Losses: mse: 7.0087, mae: 1.2954, huber: 0.9290, swd: 0.7755, ept: 176.3898
      Epoch 10 composite train-obj: 0.841003
            No improvement (1.0180), counter 2/5
    Epoch [11/50], Train Losses: mse: 6.3606, mae: 1.1440, huber: 0.7971, swd: 0.6384, ept: 179.0857
    Epoch [11/50], Val Losses: mse: 7.6710, mae: 1.2791, huber: 0.9264, swd: 0.7576, ept: 174.9967
    Epoch [11/50], Test Losses: mse: 6.0943, mae: 1.1790, huber: 0.8275, swd: 0.5773, ept: 178.9868
      Epoch 11 composite train-obj: 0.797072
            Val objective improved 0.9479 → 0.9264, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 6.1147, mae: 1.1104, huber: 0.7672, swd: 0.6147, ept: 179.1822
    Epoch [12/50], Val Losses: mse: 7.3660, mae: 1.3034, huber: 0.9457, swd: 0.7948, ept: 171.6751
    Epoch [12/50], Test Losses: mse: 6.2520, mae: 1.2237, huber: 0.8685, swd: 0.6634, ept: 175.3565
      Epoch 12 composite train-obj: 0.767220
            No improvement (0.9457), counter 1/5
    Epoch [13/50], Train Losses: mse: 6.4203, mae: 1.1384, huber: 0.7940, swd: 0.6445, ept: 178.2080
    Epoch [13/50], Val Losses: mse: 6.8372, mae: 1.2565, huber: 0.8976, swd: 0.7482, ept: 174.4574
    Epoch [13/50], Test Losses: mse: 5.5270, mae: 1.1547, huber: 0.8006, swd: 0.5628, ept: 178.6068
      Epoch 13 composite train-obj: 0.794012
            Val objective improved 0.9264 → 0.8976, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 5.8807, mae: 1.0838, huber: 0.7444, swd: 0.5868, ept: 180.6990
    Epoch [14/50], Val Losses: mse: 4.9976, mae: 1.0344, huber: 0.7024, swd: 0.6018, ept: 178.7358
    Epoch [14/50], Test Losses: mse: 4.2461, mae: 0.9709, huber: 0.6422, swd: 0.5048, ept: 182.0396
      Epoch 14 composite train-obj: 0.744355
            Val objective improved 0.8976 → 0.7024, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 5.7499, mae: 1.0665, huber: 0.7294, swd: 0.5708, ept: 182.0810
    Epoch [15/50], Val Losses: mse: 5.3387, mae: 1.0582, huber: 0.7182, swd: 0.6612, ept: 180.7201
    Epoch [15/50], Test Losses: mse: 4.5469, mae: 0.9926, huber: 0.6562, swd: 0.5361, ept: 184.3319
      Epoch 15 composite train-obj: 0.729386
            No improvement (0.7182), counter 1/5
    Epoch [16/50], Train Losses: mse: 5.1199, mae: 0.9833, huber: 0.6564, swd: 0.5181, ept: 183.3854
    Epoch [16/50], Val Losses: mse: 5.3396, mae: 1.0639, huber: 0.7258, swd: 0.6933, ept: 180.1371
    Epoch [16/50], Test Losses: mse: 4.1750, mae: 0.9750, huber: 0.6401, swd: 0.5456, ept: 183.9384
      Epoch 16 composite train-obj: 0.656430
            No improvement (0.7258), counter 2/5
    Epoch [17/50], Train Losses: mse: 5.0820, mae: 0.9845, huber: 0.6563, swd: 0.5099, ept: 183.4090
    Epoch [17/50], Val Losses: mse: 7.2672, mae: 1.2029, huber: 0.8637, swd: 0.8334, ept: 175.0677
    Epoch [17/50], Test Losses: mse: 5.5037, mae: 1.0963, huber: 0.7592, swd: 0.6250, ept: 179.0649
      Epoch 17 composite train-obj: 0.656340
            No improvement (0.8637), counter 3/5
    Epoch [18/50], Train Losses: mse: 5.0582, mae: 0.9748, huber: 0.6486, swd: 0.4991, ept: 183.4754
    Epoch [18/50], Val Losses: mse: 5.0712, mae: 1.0477, huber: 0.7104, swd: 0.5842, ept: 180.2094
    Epoch [18/50], Test Losses: mse: 4.5625, mae: 0.9891, huber: 0.6575, swd: 0.4858, ept: 183.3592
      Epoch 18 composite train-obj: 0.648562
            No improvement (0.7104), counter 4/5
    Epoch [19/50], Train Losses: mse: 5.1887, mae: 0.9849, huber: 0.6588, swd: 0.5110, ept: 183.7027
    Epoch [19/50], Val Losses: mse: 6.9259, mae: 1.1987, huber: 0.8555, swd: 0.8555, ept: 176.3387
    Epoch [19/50], Test Losses: mse: 5.6966, mae: 1.0921, huber: 0.7537, swd: 0.6393, ept: 180.4133
      Epoch 19 composite train-obj: 0.658765
    Epoch [19/50], Test Losses: mse: 4.2464, mae: 0.9710, huber: 0.6422, swd: 0.5050, ept: 182.0417
    Best round's Test MSE: 4.2461, MAE: 0.9709, SWD: 0.5048
    Best round's Validation MSE: 4.9976, MAE: 1.0344, SWD: 0.6018
    Best round's Test verification MSE : 4.2464, MAE: 0.9710, SWD: 0.5050
    Time taken: 153.85 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 53.1152, mae: 5.0308, huber: 4.5660, swd: 25.0881, ept: 69.2541
    Epoch [1/50], Val Losses: mse: 31.2778, mae: 3.4951, huber: 3.0487, swd: 5.0531, ept: 118.7153
    Epoch [1/50], Test Losses: mse: 29.7780, mae: 3.4375, huber: 2.9907, swd: 5.2372, ept: 116.3347
      Epoch 1 composite train-obj: 4.566033
            Val objective improved inf → 3.0487, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 22.4382, mae: 2.7421, huber: 2.3120, swd: 2.9305, ept: 139.4205
    Epoch [2/50], Val Losses: mse: 21.7011, mae: 2.6200, huber: 2.1949, swd: 2.4993, ept: 144.1135
    Epoch [2/50], Test Losses: mse: 18.8057, mae: 2.4669, huber: 2.0435, swd: 2.2190, ept: 143.7603
      Epoch 2 composite train-obj: 2.311981
            Val objective improved 3.0487 → 2.1949, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 15.5745, mae: 2.1264, huber: 1.7156, swd: 1.5930, ept: 156.0137
    Epoch [3/50], Val Losses: mse: 15.4890, mae: 2.1225, huber: 1.7120, swd: 1.4987, ept: 154.5753
    Epoch [3/50], Test Losses: mse: 14.4447, mae: 2.0798, huber: 1.6693, swd: 1.4830, ept: 154.8012
      Epoch 3 composite train-obj: 1.715626
            Val objective improved 2.1949 → 1.7120, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.6638, mae: 1.8309, huber: 1.4340, swd: 1.2345, ept: 164.5137
    Epoch [4/50], Val Losses: mse: 12.6075, mae: 1.8329, huber: 1.4371, swd: 1.4950, ept: 163.8141
    Epoch [4/50], Test Losses: mse: 10.4714, mae: 1.6914, huber: 1.2980, swd: 1.1506, ept: 167.9980
      Epoch 4 composite train-obj: 1.433972
            Val objective improved 1.7120 → 1.4371, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.0784, mae: 1.6516, huber: 1.2657, swd: 1.0750, ept: 169.1589
    Epoch [5/50], Val Losses: mse: 10.9670, mae: 1.7204, huber: 1.3302, swd: 1.0350, ept: 165.9142
    Epoch [5/50], Test Losses: mse: 9.6274, mae: 1.6346, huber: 1.2453, swd: 0.9263, ept: 169.2852
      Epoch 5 composite train-obj: 1.265677
            Val objective improved 1.4371 → 1.3302, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.2038, mae: 1.5507, huber: 1.1720, swd: 1.0122, ept: 171.7715
    Epoch [6/50], Val Losses: mse: 13.6711, mae: 1.8814, huber: 1.4870, swd: 1.2779, ept: 164.2595
    Epoch [6/50], Test Losses: mse: 11.5287, mae: 1.7704, huber: 1.3776, swd: 1.2063, ept: 167.3611
      Epoch 6 composite train-obj: 1.171972
            No improvement (1.4870), counter 1/5
    Epoch [7/50], Train Losses: mse: 8.9873, mae: 1.4151, huber: 1.0467, swd: 0.8841, ept: 174.6436
    Epoch [7/50], Val Losses: mse: 12.5722, mae: 1.7520, huber: 1.3691, swd: 1.1823, ept: 164.3835
    Epoch [7/50], Test Losses: mse: 10.6215, mae: 1.6525, huber: 1.2704, swd: 1.0714, ept: 167.1844
      Epoch 7 composite train-obj: 1.046667
            No improvement (1.3691), counter 2/5
    Epoch [8/50], Train Losses: mse: 8.8248, mae: 1.3972, huber: 1.0307, swd: 0.8786, ept: 175.4915
    Epoch [8/50], Val Losses: mse: 14.0466, mae: 1.8709, huber: 1.4802, swd: 1.3365, ept: 164.4702
    Epoch [8/50], Test Losses: mse: 11.8022, mae: 1.7526, huber: 1.3639, swd: 1.2362, ept: 165.9686
      Epoch 8 composite train-obj: 1.030717
            No improvement (1.4802), counter 3/5
    Epoch [9/50], Train Losses: mse: 8.3770, mae: 1.3398, huber: 0.9784, swd: 0.8206, ept: 176.8698
    Epoch [9/50], Val Losses: mse: 9.8437, mae: 1.4690, huber: 1.1026, swd: 0.7737, ept: 173.0723
    Epoch [9/50], Test Losses: mse: 8.4259, mae: 1.4087, huber: 1.0413, swd: 0.7644, ept: 175.9980
      Epoch 9 composite train-obj: 0.978409
            Val objective improved 1.3302 → 1.1026, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.5095, mae: 1.2426, huber: 0.8904, swd: 0.7476, ept: 178.5883
    Epoch [10/50], Val Losses: mse: 10.0732, mae: 1.5055, huber: 1.1384, swd: 0.9760, ept: 171.7262
    Epoch [10/50], Test Losses: mse: 8.6043, mae: 1.4281, huber: 1.0624, swd: 0.8966, ept: 174.9893
      Epoch 10 composite train-obj: 0.890385
            No improvement (1.1384), counter 1/5
    Epoch [11/50], Train Losses: mse: 7.1710, mae: 1.2032, huber: 0.8545, swd: 0.7077, ept: 179.4890
    Epoch [11/50], Val Losses: mse: 7.6722, mae: 1.2637, huber: 0.9132, swd: 0.8263, ept: 175.9240
    Epoch [11/50], Test Losses: mse: 6.7452, mae: 1.1984, huber: 0.8514, swd: 0.7502, ept: 178.8520
      Epoch 11 composite train-obj: 0.854467
            Val objective improved 1.1026 → 0.9132, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 7.0289, mae: 1.1819, huber: 0.8358, swd: 0.7035, ept: 180.1188
    Epoch [12/50], Val Losses: mse: 6.8475, mae: 1.2627, huber: 0.9077, swd: 0.9076, ept: 175.8616
    Epoch [12/50], Test Losses: mse: 5.2979, mae: 1.1528, huber: 0.8017, swd: 0.6905, ept: 179.8781
      Epoch 12 composite train-obj: 0.835812
            Val objective improved 0.9132 → 0.9077, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 6.4691, mae: 1.1276, huber: 0.7862, swd: 0.6352, ept: 181.0397
    Epoch [13/50], Val Losses: mse: 6.7686, mae: 1.1809, huber: 0.8388, swd: 0.7177, ept: 176.8587
    Epoch [13/50], Test Losses: mse: 5.6150, mae: 1.0808, huber: 0.7444, swd: 0.6080, ept: 181.6374
      Epoch 13 composite train-obj: 0.786156
            Val objective improved 0.9077 → 0.8388, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 6.4556, mae: 1.1212, huber: 0.7807, swd: 0.6175, ept: 180.9415
    Epoch [14/50], Val Losses: mse: 7.5153, mae: 1.2785, huber: 0.9250, swd: 0.8606, ept: 174.9686
    Epoch [14/50], Test Losses: mse: 5.5629, mae: 1.1617, huber: 0.8104, swd: 0.6121, ept: 178.9665
      Epoch 14 composite train-obj: 0.780708
            No improvement (0.9250), counter 1/5
    Epoch [15/50], Train Losses: mse: 6.1168, mae: 1.0785, huber: 0.7430, swd: 0.5997, ept: 181.7157
    Epoch [15/50], Val Losses: mse: 7.7719, mae: 1.2368, huber: 0.8908, swd: 0.9226, ept: 176.5586
    Epoch [15/50], Test Losses: mse: 6.3208, mae: 1.1446, huber: 0.8005, swd: 0.8109, ept: 180.5707
      Epoch 15 composite train-obj: 0.742965
            No improvement (0.8908), counter 2/5
    Epoch [16/50], Train Losses: mse: 6.1790, mae: 1.0910, huber: 0.7549, swd: 0.5948, ept: 181.7644
    Epoch [16/50], Val Losses: mse: 5.6269, mae: 1.0360, huber: 0.7119, swd: 0.6810, ept: 179.4674
    Epoch [16/50], Test Losses: mse: 4.4066, mae: 0.9403, huber: 0.6201, swd: 0.5219, ept: 183.6989
      Epoch 16 composite train-obj: 0.754945
            Val objective improved 0.8388 → 0.7119, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 5.7394, mae: 1.0299, huber: 0.7016, swd: 0.5898, ept: 182.5277
    Epoch [17/50], Val Losses: mse: 5.8736, mae: 1.0366, huber: 0.7123, swd: 0.6295, ept: 180.3788
    Epoch [17/50], Test Losses: mse: 4.9497, mae: 0.9570, huber: 0.6364, swd: 0.4994, ept: 184.5046
      Epoch 17 composite train-obj: 0.701558
            No improvement (0.7123), counter 1/5
    Epoch [18/50], Train Losses: mse: 5.7992, mae: 1.0280, huber: 0.7005, swd: 0.5619, ept: 182.9595
    Epoch [18/50], Val Losses: mse: 7.3913, mae: 1.2314, huber: 0.8882, swd: 0.9033, ept: 176.7047
    Epoch [18/50], Test Losses: mse: 5.8226, mae: 1.1094, huber: 0.7724, swd: 0.6964, ept: 180.2597
      Epoch 18 composite train-obj: 0.700524
            No improvement (0.8882), counter 2/5
    Epoch [19/50], Train Losses: mse: 5.4595, mae: 1.0053, huber: 0.6792, swd: 0.5385, ept: 183.4764
    Epoch [19/50], Val Losses: mse: 5.3885, mae: 1.0069, huber: 0.6811, swd: 0.5557, ept: 181.6935
    Epoch [19/50], Test Losses: mse: 3.9803, mae: 0.9032, huber: 0.5836, swd: 0.4843, ept: 184.9247
      Epoch 19 composite train-obj: 0.679156
            Val objective improved 0.7119 → 0.6811, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 5.0488, mae: 0.9543, huber: 0.6336, swd: 0.4911, ept: 184.3452
    Epoch [20/50], Val Losses: mse: 5.0767, mae: 1.0019, huber: 0.6756, swd: 0.6093, ept: 182.1751
    Epoch [20/50], Test Losses: mse: 3.5681, mae: 0.8833, huber: 0.5616, swd: 0.4618, ept: 185.8241
      Epoch 20 composite train-obj: 0.633648
            Val objective improved 0.6811 → 0.6756, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 4.7901, mae: 0.9197, huber: 0.6040, swd: 0.4702, ept: 185.0318
    Epoch [21/50], Val Losses: mse: 5.8377, mae: 1.0854, huber: 0.7511, swd: 0.7503, ept: 179.8413
    Epoch [21/50], Test Losses: mse: 4.4517, mae: 0.9762, huber: 0.6473, swd: 0.5736, ept: 182.9368
      Epoch 21 composite train-obj: 0.603980
            No improvement (0.7511), counter 1/5
    Epoch [22/50], Train Losses: mse: 5.0855, mae: 0.9577, huber: 0.6366, swd: 0.4863, ept: 184.6364
    Epoch [22/50], Val Losses: mse: 6.1808, mae: 1.1236, huber: 0.7883, swd: 0.7048, ept: 178.8865
    Epoch [22/50], Test Losses: mse: 4.4451, mae: 0.9962, huber: 0.6653, swd: 0.5612, ept: 183.0254
      Epoch 22 composite train-obj: 0.636617
            No improvement (0.7883), counter 2/5
    Epoch [23/50], Train Losses: mse: 4.9612, mae: 0.9410, huber: 0.6221, swd: 0.4836, ept: 184.7428
    Epoch [23/50], Val Losses: mse: 5.9236, mae: 1.0763, huber: 0.7429, swd: 0.7150, ept: 181.5295
    Epoch [23/50], Test Losses: mse: 4.6997, mae: 0.9837, huber: 0.6554, swd: 0.5753, ept: 184.1953
      Epoch 23 composite train-obj: 0.622089
            No improvement (0.7429), counter 3/5
    Epoch [24/50], Train Losses: mse: 4.9192, mae: 0.9319, huber: 0.6153, swd: 0.4869, ept: 184.6361
    Epoch [24/50], Val Losses: mse: 6.6233, mae: 1.1584, huber: 0.8208, swd: 0.6637, ept: 177.4494
    Epoch [24/50], Test Losses: mse: 5.6587, mae: 1.0969, huber: 0.7596, swd: 0.5483, ept: 181.0309
      Epoch 24 composite train-obj: 0.615276
            No improvement (0.8208), counter 4/5
    Epoch [25/50], Train Losses: mse: 4.7897, mae: 0.9184, huber: 0.6038, swd: 0.4837, ept: 185.0515
    Epoch [25/50], Val Losses: mse: 6.2329, mae: 1.0940, huber: 0.7635, swd: 0.6649, ept: 179.0646
    Epoch [25/50], Test Losses: mse: 4.5581, mae: 1.0030, huber: 0.6714, swd: 0.5636, ept: 183.1315
      Epoch 25 composite train-obj: 0.603764
    Epoch [25/50], Test Losses: mse: 3.5679, mae: 0.8833, huber: 0.5616, swd: 0.4619, ept: 185.8177
    Best round's Test MSE: 3.5681, MAE: 0.8833, SWD: 0.4618
    Best round's Validation MSE: 5.0767, MAE: 1.0019, SWD: 0.6093
    Best round's Test verification MSE : 3.5679, MAE: 0.8833, SWD: 0.4619
    Time taken: 203.42 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred196_20250510_0907)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 4.3393 ± 0.6710
      mae: 0.9808 ± 0.0839
      huber: 0.6500 ± 0.0756
      swd: 0.5530 ± 0.1002
      ept: 183.1031 ± 1.9392
      count: 37.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.3437 ± 0.4347
      mae: 1.0593 ± 0.0597
      huber: 0.7249 ± 0.0520
      swd: 0.6642 ± 0.0831
      ept: 179.9557 ± 1.5719
      count: 37.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 487.04 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred196_20250510_0907
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 336-336
##### huber


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
    # single_magnitude_for_shift=True,
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False,
    

)
cfg.x_to_z_delay.enable_magnitudes = [False, False]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, False]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, False]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, False]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, False]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 280
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
    Training Batches: 280
    Validation Batches: 36
    Test Batches: 77
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.6123, mae: 6.5200, huber: 6.0398, swd: 41.4769, ept: 31.7121
    Epoch [1/50], Val Losses: mse: 55.1509, mae: 5.4593, huber: 4.9862, swd: 25.2937, ept: 57.5766
    Epoch [1/50], Test Losses: mse: 53.8095, mae: 5.3572, huber: 4.8846, swd: 26.2665, ept: 57.3996
      Epoch 1 composite train-obj: 6.039762
            Val objective improved inf → 4.9862, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 46.5734, mae: 4.7640, huber: 4.2993, swd: 16.2663, ept: 96.6749
    Epoch [2/50], Val Losses: mse: 46.9755, mae: 4.6948, huber: 4.2334, swd: 11.9818, ept: 104.7933
    Epoch [2/50], Test Losses: mse: 45.7250, mae: 4.5925, huber: 4.1330, swd: 12.7715, ept: 102.6665
      Epoch 2 composite train-obj: 4.299296
            Val objective improved 4.9862 → 4.2334, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.4463, mae: 4.1636, huber: 3.7110, swd: 9.4912, ept: 138.1554
    Epoch [3/50], Val Losses: mse: 42.6181, mae: 4.2581, huber: 3.8073, swd: 8.7084, ept: 140.5412
    Epoch [3/50], Test Losses: mse: 41.1569, mae: 4.1496, huber: 3.6994, swd: 8.8459, ept: 139.5077
      Epoch 3 composite train-obj: 3.710951
            Val objective improved 4.2334 → 3.8073, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.1445, mae: 3.8383, huber: 3.3932, swd: 7.1901, ept: 161.2817
    Epoch [4/50], Val Losses: mse: 40.7833, mae: 4.1148, huber: 3.6672, swd: 7.2515, ept: 153.2758
    Epoch [4/50], Test Losses: mse: 38.2643, mae: 3.9305, huber: 3.4855, swd: 7.0423, ept: 155.5466
      Epoch 4 composite train-obj: 3.393233
            Val objective improved 3.8073 → 3.6672, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 34.2541, mae: 3.5932, huber: 3.1537, swd: 5.6295, ept: 173.2820
    Epoch [5/50], Val Losses: mse: 40.3202, mae: 3.9986, huber: 3.5547, swd: 6.3105, ept: 163.0908
    Epoch [5/50], Test Losses: mse: 36.3290, mae: 3.7521, huber: 3.3104, swd: 5.9624, ept: 167.5208
      Epoch 5 composite train-obj: 3.153681
            Val objective improved 3.6672 → 3.5547, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 32.2111, mae: 3.4210, huber: 2.9859, swd: 4.7795, ept: 182.8845
    Epoch [6/50], Val Losses: mse: 38.6311, mae: 3.8938, huber: 3.4496, swd: 5.3292, ept: 160.1741
    Epoch [6/50], Test Losses: mse: 35.2977, mae: 3.6823, huber: 3.2401, swd: 5.0966, ept: 165.4074
      Epoch 6 composite train-obj: 2.985892
            Val objective improved 3.5547 → 3.4496, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 30.4275, mae: 3.2622, huber: 2.8319, swd: 4.1474, ept: 190.2448
    Epoch [7/50], Val Losses: mse: 33.8553, mae: 3.5160, huber: 3.0814, swd: 4.5988, ept: 182.3411
    Epoch [7/50], Test Losses: mse: 30.2962, mae: 3.2786, huber: 2.8471, swd: 4.2863, ept: 189.0078
      Epoch 7 composite train-obj: 2.831862
            Val objective improved 3.4496 → 3.0814, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 28.8986, mae: 3.1296, huber: 2.7033, swd: 3.7165, ept: 197.5170
    Epoch [8/50], Val Losses: mse: 34.0249, mae: 3.5106, huber: 3.0768, swd: 4.1250, ept: 181.7222
    Epoch [8/50], Test Losses: mse: 30.2380, mae: 3.2722, huber: 2.8409, swd: 3.7136, ept: 186.6695
      Epoch 8 composite train-obj: 2.703287
            Val objective improved 3.0814 → 3.0768, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 27.4520, mae: 3.0042, huber: 2.5822, swd: 3.3349, ept: 203.9630
    Epoch [9/50], Val Losses: mse: 38.6071, mae: 3.7220, huber: 3.2860, swd: 3.8711, ept: 181.5672
    Epoch [9/50], Test Losses: mse: 31.5598, mae: 3.3351, huber: 2.9031, swd: 3.6981, ept: 191.1753
      Epoch 9 composite train-obj: 2.582167
            No improvement (3.2860), counter 1/5
    Epoch [10/50], Train Losses: mse: 26.6442, mae: 2.9376, huber: 2.5169, swd: 3.0848, ept: 208.5888
    Epoch [10/50], Val Losses: mse: 32.4656, mae: 3.3562, huber: 2.9271, swd: 3.6486, ept: 185.8681
    Epoch [10/50], Test Losses: mse: 28.7159, mae: 3.1321, huber: 2.7056, swd: 3.3641, ept: 192.9233
      Epoch 10 composite train-obj: 2.516925
            Val objective improved 3.0768 → 2.9271, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 25.8359, mae: 2.8671, huber: 2.4485, swd: 2.9175, ept: 211.9512
    Epoch [11/50], Val Losses: mse: 33.9579, mae: 3.4059, huber: 2.9772, swd: 3.3235, ept: 193.0325
    Epoch [11/50], Test Losses: mse: 30.9106, mae: 3.2081, huber: 2.7819, swd: 3.0388, ept: 202.0256
      Epoch 11 composite train-obj: 2.448529
            No improvement (2.9772), counter 1/5
    Epoch [12/50], Train Losses: mse: 24.2835, mae: 2.7319, huber: 2.3187, swd: 2.6368, ept: 218.4419
    Epoch [12/50], Val Losses: mse: 30.4805, mae: 3.1934, huber: 2.7657, swd: 3.3220, ept: 199.5157
    Epoch [12/50], Test Losses: mse: 26.5364, mae: 2.9493, huber: 2.5249, swd: 2.9404, ept: 208.7979
      Epoch 12 composite train-obj: 2.318689
            Val objective improved 2.9271 → 2.7657, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 24.1162, mae: 2.7171, huber: 2.3033, swd: 2.5151, ept: 219.8861
    Epoch [13/50], Val Losses: mse: 30.8264, mae: 3.1740, huber: 2.7521, swd: 2.9788, ept: 198.1907
    Epoch [13/50], Test Losses: mse: 26.8605, mae: 2.9385, huber: 2.5195, swd: 2.7421, ept: 206.1088
      Epoch 13 composite train-obj: 2.303320
            Val objective improved 2.7657 → 2.7521, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 23.2072, mae: 2.6337, huber: 2.2236, swd: 2.3461, ept: 223.3219
    Epoch [14/50], Val Losses: mse: 28.9416, mae: 3.0478, huber: 2.6271, swd: 3.0932, ept: 206.5601
    Epoch [14/50], Test Losses: mse: 25.3149, mae: 2.8087, huber: 2.3918, swd: 2.6384, ept: 217.6203
      Epoch 14 composite train-obj: 2.223598
            Val objective improved 2.7521 → 2.6271, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 22.5487, mae: 2.5755, huber: 2.1673, swd: 2.2396, ept: 227.0763
    Epoch [15/50], Val Losses: mse: 27.0351, mae: 2.9541, huber: 2.5335, swd: 3.0401, ept: 207.7557
    Epoch [15/50], Test Losses: mse: 23.3483, mae: 2.7269, huber: 2.3099, swd: 2.7536, ept: 215.3412
      Epoch 15 composite train-obj: 2.167276
            Val objective improved 2.6271 → 2.5335, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 22.1862, mae: 2.5427, huber: 2.1358, swd: 2.1543, ept: 229.5376
    Epoch [16/50], Val Losses: mse: 28.8659, mae: 2.9935, huber: 2.5766, swd: 2.4153, ept: 212.2562
    Epoch [16/50], Test Losses: mse: 22.8292, mae: 2.6542, huber: 2.2410, swd: 2.2585, ept: 221.4912
      Epoch 16 composite train-obj: 2.135788
            No improvement (2.5766), counter 1/5
    Epoch [17/50], Train Losses: mse: 21.2143, mae: 2.4639, huber: 2.0599, swd: 2.0291, ept: 232.5191
    Epoch [17/50], Val Losses: mse: 27.6546, mae: 2.9185, huber: 2.5028, swd: 2.4078, ept: 213.6025
    Epoch [17/50], Test Losses: mse: 24.1837, mae: 2.7084, huber: 2.2961, swd: 2.1843, ept: 220.2419
      Epoch 17 composite train-obj: 2.059906
            Val objective improved 2.5335 → 2.5028, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 21.3477, mae: 2.4653, huber: 2.0612, swd: 1.9812, ept: 234.0058
    Epoch [18/50], Val Losses: mse: 26.3270, mae: 2.8174, huber: 2.4042, swd: 2.3819, ept: 218.9101
    Epoch [18/50], Test Losses: mse: 21.8267, mae: 2.5441, huber: 2.1355, swd: 2.1106, ept: 229.9237
      Epoch 18 composite train-obj: 2.061175
            Val objective improved 2.5028 → 2.4042, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 20.3379, mae: 2.3782, huber: 1.9782, swd: 1.8689, ept: 238.1088
    Epoch [19/50], Val Losses: mse: 28.3204, mae: 2.9206, huber: 2.5074, swd: 2.2471, ept: 216.2416
    Epoch [19/50], Test Losses: mse: 24.6782, mae: 2.7028, huber: 2.2934, swd: 1.9756, ept: 225.0376
      Epoch 19 composite train-obj: 1.978242
            No improvement (2.5074), counter 1/5
    Epoch [20/50], Train Losses: mse: 21.4771, mae: 2.4667, huber: 2.0627, swd: 1.9838, ept: 235.0957
    Epoch [20/50], Val Losses: mse: 27.2065, mae: 2.8932, huber: 2.4758, swd: 2.6508, ept: 219.8874
    Epoch [20/50], Test Losses: mse: 20.7458, mae: 2.5053, huber: 2.0948, swd: 2.2721, ept: 230.1988
      Epoch 20 composite train-obj: 2.062723
            No improvement (2.4758), counter 2/5
    Epoch [21/50], Train Losses: mse: 20.3768, mae: 2.3797, huber: 1.9790, swd: 1.8135, ept: 238.9345
    Epoch [21/50], Val Losses: mse: 37.4275, mae: 3.3408, huber: 2.9218, swd: 2.3258, ept: 210.5306
    Epoch [21/50], Test Losses: mse: 31.1711, mae: 3.0083, huber: 2.5953, swd: 2.2291, ept: 220.6046
      Epoch 21 composite train-obj: 1.979047
            No improvement (2.9218), counter 3/5
    Epoch [22/50], Train Losses: mse: 19.5027, mae: 2.2982, huber: 1.9022, swd: 1.7497, ept: 242.7517
    Epoch [22/50], Val Losses: mse: 28.4505, mae: 2.9507, huber: 2.5315, swd: 2.0615, ept: 216.5808
    Epoch [22/50], Test Losses: mse: 22.5764, mae: 2.6046, huber: 2.1912, swd: 1.8274, ept: 227.6547
      Epoch 22 composite train-obj: 1.902218
            No improvement (2.5315), counter 4/5
    Epoch [23/50], Train Losses: mse: 19.3922, mae: 2.2949, huber: 1.8976, swd: 1.7142, ept: 243.6409
    Epoch [23/50], Val Losses: mse: 23.0367, mae: 2.5023, huber: 2.1060, swd: 2.0609, ept: 234.2191
    Epoch [23/50], Test Losses: mse: 18.6753, mae: 2.2418, huber: 1.8508, swd: 1.8302, ept: 244.3466
      Epoch 23 composite train-obj: 1.897576
            Val objective improved 2.4042 → 2.1060, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 17.8529, mae: 2.1680, huber: 1.7767, swd: 1.5410, ept: 247.6470
    Epoch [24/50], Val Losses: mse: 26.3434, mae: 2.7331, huber: 2.3275, swd: 2.0100, ept: 226.6226
    Epoch [24/50], Test Losses: mse: 20.6967, mae: 2.4141, huber: 2.0126, swd: 1.7322, ept: 238.6911
      Epoch 24 composite train-obj: 1.776712
            No improvement (2.3275), counter 1/5
    Epoch [25/50], Train Losses: mse: 18.3566, mae: 2.2076, huber: 1.8139, swd: 1.5593, ept: 247.3431
    Epoch [25/50], Val Losses: mse: 25.4608, mae: 2.6199, huber: 2.2182, swd: 1.8515, ept: 233.1771
    Epoch [25/50], Test Losses: mse: 19.9203, mae: 2.3118, huber: 1.9162, swd: 1.5955, ept: 245.5463
      Epoch 25 composite train-obj: 1.813874
            No improvement (2.2182), counter 2/5
    Epoch [26/50], Train Losses: mse: 17.4145, mae: 2.1241, huber: 1.7350, swd: 1.4763, ept: 250.8104
    Epoch [26/50], Val Losses: mse: 27.8448, mae: 2.8088, huber: 2.4023, swd: 1.8125, ept: 217.5592
    Epoch [26/50], Test Losses: mse: 22.5023, mae: 2.5242, huber: 2.1229, swd: 1.6058, ept: 228.5378
      Epoch 26 composite train-obj: 1.734976
            No improvement (2.4023), counter 3/5
    Epoch [27/50], Train Losses: mse: 17.5535, mae: 2.1425, huber: 1.7512, swd: 1.4653, ept: 249.9883
    Epoch [27/50], Val Losses: mse: 22.6503, mae: 2.4780, huber: 2.0793, swd: 1.8238, ept: 234.7822
    Epoch [27/50], Test Losses: mse: 20.0318, mae: 2.3235, huber: 1.9269, swd: 1.5848, ept: 244.4896
      Epoch 27 composite train-obj: 1.751168
            Val objective improved 2.1060 → 2.0793, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 17.2652, mae: 2.1105, huber: 1.7213, swd: 1.4144, ept: 251.4712
    Epoch [28/50], Val Losses: mse: 23.0172, mae: 2.5335, huber: 2.1290, swd: 2.0548, ept: 232.8815
    Epoch [28/50], Test Losses: mse: 18.3563, mae: 2.2539, huber: 1.8537, swd: 1.6869, ept: 245.0113
      Epoch 28 composite train-obj: 1.721339
            No improvement (2.1290), counter 1/5
    Epoch [29/50], Train Losses: mse: 16.9559, mae: 2.0810, huber: 1.6934, swd: 1.3703, ept: 253.0758
    Epoch [29/50], Val Losses: mse: 23.0576, mae: 2.5001, huber: 2.1030, swd: 1.8890, ept: 232.9581
    Epoch [29/50], Test Losses: mse: 19.3263, mae: 2.2795, huber: 1.8874, swd: 1.6509, ept: 242.6436
      Epoch 29 composite train-obj: 1.693410
            No improvement (2.1030), counter 2/5
    Epoch [30/50], Train Losses: mse: 16.8559, mae: 2.0765, huber: 1.6885, swd: 1.3544, ept: 253.2943
    Epoch [30/50], Val Losses: mse: 21.1996, mae: 2.3926, huber: 1.9984, swd: 1.6559, ept: 236.1667
    Epoch [30/50], Test Losses: mse: 17.1560, mae: 2.1471, huber: 1.7579, swd: 1.3781, ept: 247.7507
      Epoch 30 composite train-obj: 1.688518
            Val objective improved 2.0793 → 1.9984, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 15.9099, mae: 1.9961, huber: 1.6124, swd: 1.2758, ept: 256.5906
    Epoch [31/50], Val Losses: mse: 21.6746, mae: 2.4257, huber: 2.0283, swd: 1.7622, ept: 239.3444
    Epoch [31/50], Test Losses: mse: 16.7417, mae: 2.1220, huber: 1.7306, swd: 1.4853, ept: 250.8833
      Epoch 31 composite train-obj: 1.612389
            No improvement (2.0283), counter 1/5
    Epoch [32/50], Train Losses: mse: 16.2323, mae: 2.0146, huber: 1.6304, swd: 1.2769, ept: 256.5792
    Epoch [32/50], Val Losses: mse: 23.3528, mae: 2.4860, huber: 2.0894, swd: 1.7844, ept: 237.4768
    Epoch [32/50], Test Losses: mse: 18.0432, mae: 2.1787, huber: 1.7862, swd: 1.3865, ept: 250.3701
      Epoch 32 composite train-obj: 1.630368
            No improvement (2.0894), counter 2/5
    Epoch [33/50], Train Losses: mse: 15.8806, mae: 1.9867, huber: 1.6033, swd: 1.2474, ept: 257.5155
    Epoch [33/50], Val Losses: mse: 23.0626, mae: 2.4556, huber: 2.0583, swd: 1.5175, ept: 237.2806
    Epoch [33/50], Test Losses: mse: 20.9931, mae: 2.3256, huber: 1.9316, swd: 1.3379, ept: 246.8238
      Epoch 33 composite train-obj: 1.603309
            No improvement (2.0583), counter 3/5
    Epoch [34/50], Train Losses: mse: 15.8888, mae: 1.9912, huber: 1.6075, swd: 1.2517, ept: 257.5136
    Epoch [34/50], Val Losses: mse: 19.4180, mae: 2.2477, huber: 1.8567, swd: 1.3889, ept: 242.0818
    Epoch [34/50], Test Losses: mse: 14.7391, mae: 1.9774, huber: 1.5926, swd: 1.1949, ept: 253.1003
      Epoch 34 composite train-obj: 1.607470
            Val objective improved 1.9984 → 1.8567, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 15.0307, mae: 1.9157, huber: 1.5360, swd: 1.1581, ept: 260.2948
    Epoch [35/50], Val Losses: mse: 18.5086, mae: 2.1808, huber: 1.7929, swd: 1.5553, ept: 246.4750
    Epoch [35/50], Test Losses: mse: 14.7206, mae: 1.9420, huber: 1.5603, swd: 1.3467, ept: 258.5661
      Epoch 35 composite train-obj: 1.535995
            Val objective improved 1.8567 → 1.7929, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 14.5808, mae: 1.8783, huber: 1.5005, swd: 1.1209, ept: 262.1391
    Epoch [36/50], Val Losses: mse: 24.8138, mae: 2.5107, huber: 2.1141, swd: 1.5771, ept: 239.2104
    Epoch [36/50], Test Losses: mse: 19.6387, mae: 2.2296, huber: 1.8386, swd: 1.5041, ept: 251.4793
      Epoch 36 composite train-obj: 1.500535
            No improvement (2.1141), counter 1/5
    Epoch [37/50], Train Losses: mse: 14.6664, mae: 1.8853, huber: 1.5066, swd: 1.1314, ept: 262.2591
    Epoch [37/50], Val Losses: mse: 19.6863, mae: 2.3021, huber: 1.9072, swd: 1.6481, ept: 238.5416
    Epoch [37/50], Test Losses: mse: 15.5296, mae: 2.0339, huber: 1.6453, swd: 1.2697, ept: 251.3129
      Epoch 37 composite train-obj: 1.506587
            No improvement (1.9072), counter 2/5
    Epoch [38/50], Train Losses: mse: 14.4515, mae: 1.8607, huber: 1.4845, swd: 1.0930, ept: 263.4538
    Epoch [38/50], Val Losses: mse: 23.6502, mae: 2.5094, huber: 2.1128, swd: 1.4753, ept: 235.3811
    Epoch [38/50], Test Losses: mse: 17.0876, mae: 2.1348, huber: 1.7460, swd: 1.2282, ept: 248.4264
      Epoch 38 composite train-obj: 1.484488
            No improvement (2.1128), counter 3/5
    Epoch [39/50], Train Losses: mse: 15.1029, mae: 1.9136, huber: 1.5343, swd: 1.1504, ept: 261.6340
    Epoch [39/50], Val Losses: mse: 20.9421, mae: 2.3051, huber: 1.9156, swd: 1.7425, ept: 244.5088
    Epoch [39/50], Test Losses: mse: 17.6485, mae: 2.1001, huber: 1.7166, swd: 1.4178, ept: 255.2068
      Epoch 39 composite train-obj: 1.534278
            No improvement (1.9156), counter 4/5
    Epoch [40/50], Train Losses: mse: 13.8184, mae: 1.8141, huber: 1.4395, swd: 1.0458, ept: 265.1520
    Epoch [40/50], Val Losses: mse: 21.0603, mae: 2.3299, huber: 1.9390, swd: 1.5786, ept: 242.8780
    Epoch [40/50], Test Losses: mse: 16.4705, mae: 2.0428, huber: 1.6567, swd: 1.1992, ept: 255.9252
      Epoch 40 composite train-obj: 1.439455
    Epoch [40/50], Test Losses: mse: 14.7219, mae: 1.9420, huber: 1.5603, swd: 1.3466, ept: 258.5661
    Best round's Test MSE: 14.7206, MAE: 1.9420, SWD: 1.3467
    Best round's Validation MSE: 18.5086, MAE: 2.1808, SWD: 1.5553
    Best round's Test verification MSE : 14.7219, MAE: 1.9420, SWD: 1.3466
    Time taken: 267.63 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 73.1604, mae: 6.5613, huber: 6.0810, swd: 40.5859, ept: 28.3190
    Epoch [1/50], Val Losses: mse: 55.9656, mae: 5.5100, huber: 5.0359, swd: 19.9853, ept: 58.1023
    Epoch [1/50], Test Losses: mse: 54.0782, mae: 5.3844, huber: 4.9105, swd: 19.6524, ept: 59.2774
      Epoch 1 composite train-obj: 6.081031
            Val objective improved inf → 5.0359, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 44.3004, mae: 4.5803, huber: 4.1185, swd: 12.7933, ept: 99.2762
    Epoch [2/50], Val Losses: mse: 44.4512, mae: 4.5379, huber: 4.0790, swd: 11.1847, ept: 110.6415
    Epoch [2/50], Test Losses: mse: 41.6859, mae: 4.3406, huber: 3.8842, swd: 11.1201, ept: 108.8790
      Epoch 2 composite train-obj: 4.118542
            Val objective improved 5.0359 → 4.0790, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.5895, mae: 4.0273, huber: 3.5767, swd: 8.3125, ept: 134.8347
    Epoch [3/50], Val Losses: mse: 40.0390, mae: 4.1277, huber: 3.6763, swd: 7.8485, ept: 131.0548
    Epoch [3/50], Test Losses: mse: 38.1522, mae: 3.9772, huber: 3.5282, swd: 7.4938, ept: 131.4189
      Epoch 3 composite train-obj: 3.576726
            Val objective improved 4.0790 → 3.6763, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.3306, mae: 3.7150, huber: 3.2721, swd: 6.4481, ept: 157.6700
    Epoch [4/50], Val Losses: mse: 38.0584, mae: 3.8992, huber: 3.4540, swd: 6.6782, ept: 155.8246
    Epoch [4/50], Test Losses: mse: 34.7882, mae: 3.6920, huber: 3.2490, swd: 6.4629, ept: 154.8730
      Epoch 4 composite train-obj: 3.272080
            Val objective improved 3.6763 → 3.4540, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.7087, mae: 3.4832, huber: 3.0462, swd: 5.2811, ept: 174.0415
    Epoch [5/50], Val Losses: mse: 36.1338, mae: 3.7124, huber: 3.2736, swd: 5.8105, ept: 167.7786
    Epoch [5/50], Test Losses: mse: 32.8140, mae: 3.5109, huber: 3.0741, swd: 5.8823, ept: 167.4698
      Epoch 5 composite train-obj: 3.046245
            Val objective improved 3.4540 → 3.2736, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.3072, mae: 3.3547, huber: 2.9208, swd: 4.6144, ept: 184.5835
    Epoch [6/50], Val Losses: mse: 34.2066, mae: 3.5726, huber: 3.1362, swd: 4.5455, ept: 178.8212
    Epoch [6/50], Test Losses: mse: 30.3143, mae: 3.3131, huber: 2.8802, swd: 4.2606, ept: 183.1801
      Epoch 6 composite train-obj: 2.920849
            Val objective improved 3.2736 → 3.1362, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 29.6036, mae: 3.1995, huber: 2.7707, swd: 4.0308, ept: 193.7534
    Epoch [7/50], Val Losses: mse: 34.6151, mae: 3.5250, huber: 3.0934, swd: 4.5742, ept: 185.3434
    Epoch [7/50], Test Losses: mse: 29.6496, mae: 3.2127, huber: 2.7847, swd: 4.0549, ept: 190.8180
      Epoch 7 composite train-obj: 2.770749
            Val objective improved 3.1362 → 3.0934, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 28.1249, mae: 3.0759, huber: 2.6507, swd: 3.5894, ept: 199.4985
    Epoch [8/50], Val Losses: mse: 32.8806, mae: 3.4109, huber: 2.9810, swd: 3.8461, ept: 182.8179
    Epoch [8/50], Test Losses: mse: 28.5947, mae: 3.1436, huber: 2.7169, swd: 3.5676, ept: 190.2310
      Epoch 8 composite train-obj: 2.650713
            Val objective improved 3.0934 → 2.9810, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 26.4267, mae: 2.9231, huber: 2.5037, swd: 3.1917, ept: 206.4482
    Epoch [9/50], Val Losses: mse: 30.5601, mae: 3.2453, huber: 2.8188, swd: 3.8693, ept: 187.6286
    Epoch [9/50], Test Losses: mse: 26.4342, mae: 3.0002, huber: 2.5775, swd: 3.5177, ept: 191.2560
      Epoch 9 composite train-obj: 2.503701
            Val objective improved 2.9810 → 2.8188, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 25.5884, mae: 2.8553, huber: 2.4371, swd: 2.9795, ept: 209.8566
    Epoch [10/50], Val Losses: mse: 29.8845, mae: 3.1585, huber: 2.7362, swd: 3.3426, ept: 196.1509
    Epoch [10/50], Test Losses: mse: 25.6355, mae: 2.9024, huber: 2.4831, swd: 3.0743, ept: 201.2006
      Epoch 10 composite train-obj: 2.437079
            Val objective improved 2.8188 → 2.7362, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 24.4778, mae: 2.7628, huber: 2.3478, swd: 2.7408, ept: 214.0158
    Epoch [11/50], Val Losses: mse: 31.4990, mae: 3.2507, huber: 2.8237, swd: 3.6451, ept: 199.6835
    Epoch [11/50], Test Losses: mse: 25.9960, mae: 2.9201, huber: 2.4977, swd: 3.3549, ept: 206.7344
      Epoch 11 composite train-obj: 2.347835
            No improvement (2.8237), counter 1/5
    Epoch [12/50], Train Losses: mse: 24.4558, mae: 2.7650, huber: 2.3489, swd: 2.6645, ept: 215.1462
    Epoch [12/50], Val Losses: mse: 28.1104, mae: 2.9747, huber: 2.5601, swd: 2.9168, ept: 206.7159
    Epoch [12/50], Test Losses: mse: 23.7141, mae: 2.7197, huber: 2.3082, swd: 2.7252, ept: 214.3111
      Epoch 12 composite train-obj: 2.348876
            Val objective improved 2.7362 → 2.5601, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 23.2934, mae: 2.6485, huber: 2.2379, swd: 2.4469, ept: 221.6579
    Epoch [13/50], Val Losses: mse: 28.9277, mae: 3.0585, huber: 2.6377, swd: 2.6643, ept: 203.3673
    Epoch [13/50], Test Losses: mse: 24.4045, mae: 2.8015, huber: 2.3833, swd: 2.4398, ept: 209.6091
      Epoch 13 composite train-obj: 2.237949
            No improvement (2.6377), counter 1/5
    Epoch [14/50], Train Losses: mse: 21.8999, mae: 2.5384, huber: 2.1314, swd: 2.2202, ept: 226.2589
    Epoch [14/50], Val Losses: mse: 27.6633, mae: 2.9757, huber: 2.5558, swd: 2.8871, ept: 208.9993
    Epoch [14/50], Test Losses: mse: 24.3918, mae: 2.7706, huber: 2.3539, swd: 2.5882, ept: 218.2803
      Epoch 14 composite train-obj: 2.131407
            Val objective improved 2.5601 → 2.5558, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 21.1261, mae: 2.4660, huber: 2.0622, swd: 2.0793, ept: 230.7483
    Epoch [15/50], Val Losses: mse: 26.8389, mae: 2.8159, huber: 2.4064, swd: 2.2694, ept: 217.1233
    Epoch [15/50], Test Losses: mse: 22.1727, mae: 2.5548, huber: 2.1485, swd: 2.1159, ept: 226.2355
      Epoch 15 composite train-obj: 2.062172
            Val objective improved 2.5558 → 2.4064, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 20.4701, mae: 2.4086, huber: 2.0070, swd: 1.9569, ept: 234.1982
    Epoch [16/50], Val Losses: mse: 24.4805, mae: 2.7739, huber: 2.3620, swd: 2.2240, ept: 212.4743
    Epoch [16/50], Test Losses: mse: 20.4955, mae: 2.5300, huber: 2.1212, swd: 1.9841, ept: 221.2924
      Epoch 16 composite train-obj: 2.006989
            Val objective improved 2.4064 → 2.3620, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 20.3060, mae: 2.3946, huber: 1.9931, swd: 1.9059, ept: 235.4136
    Epoch [17/50], Val Losses: mse: 25.5477, mae: 2.6953, huber: 2.2936, swd: 2.1494, ept: 224.6555
    Epoch [17/50], Test Losses: mse: 20.4786, mae: 2.4110, huber: 2.0121, swd: 1.9334, ept: 234.0291
      Epoch 17 composite train-obj: 1.993139
            Val objective improved 2.3620 → 2.2936, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 19.1990, mae: 2.3022, huber: 1.9047, swd: 1.7605, ept: 239.5341
    Epoch [18/50], Val Losses: mse: 22.3600, mae: 2.5193, huber: 2.1181, swd: 1.9997, ept: 228.0685
    Epoch [18/50], Test Losses: mse: 18.2979, mae: 2.2770, huber: 1.8797, swd: 1.7850, ept: 237.6885
      Epoch 18 composite train-obj: 1.904674
            Val objective improved 2.2936 → 2.1181, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 18.8065, mae: 2.2645, huber: 1.8686, swd: 1.6912, ept: 241.6022
    Epoch [19/50], Val Losses: mse: 25.3141, mae: 2.7436, huber: 2.3340, swd: 2.2393, ept: 221.5907
    Epoch [19/50], Test Losses: mse: 20.7922, mae: 2.4797, huber: 2.0735, swd: 2.0278, ept: 231.8671
      Epoch 19 composite train-obj: 1.868644
            No improvement (2.3340), counter 1/5
    Epoch [20/50], Train Losses: mse: 19.0412, mae: 2.2829, huber: 1.8853, swd: 1.6954, ept: 242.0695
    Epoch [20/50], Val Losses: mse: 30.9943, mae: 3.0172, huber: 2.6042, swd: 2.0398, ept: 219.1721
    Epoch [20/50], Test Losses: mse: 24.5162, mae: 2.6920, huber: 2.2805, swd: 1.9513, ept: 226.9969
      Epoch 20 composite train-obj: 1.885299
            No improvement (2.6042), counter 2/5
    Epoch [21/50], Train Losses: mse: 19.7173, mae: 2.3368, huber: 1.9370, swd: 1.8359, ept: 240.3110
    Epoch [21/50], Val Losses: mse: 23.9708, mae: 2.6088, huber: 2.2039, swd: 1.9905, ept: 228.1870
    Epoch [21/50], Test Losses: mse: 19.0129, mae: 2.3286, huber: 1.9268, swd: 1.7718, ept: 238.0099
      Epoch 21 composite train-obj: 1.936986
            No improvement (2.2039), counter 3/5
    Epoch [22/50], Train Losses: mse: 18.2504, mae: 2.2110, huber: 1.8166, swd: 1.5793, ept: 246.2078
    Epoch [22/50], Val Losses: mse: 25.2009, mae: 2.6394, huber: 2.2385, swd: 1.8002, ept: 226.9477
    Epoch [22/50], Test Losses: mse: 19.8206, mae: 2.3272, huber: 1.9317, swd: 1.5781, ept: 241.3993
      Epoch 22 composite train-obj: 1.816609
            No improvement (2.2385), counter 4/5
    Epoch [23/50], Train Losses: mse: 17.4665, mae: 2.1388, huber: 1.7486, swd: 1.4985, ept: 249.5100
    Epoch [23/50], Val Losses: mse: 21.6132, mae: 2.4637, huber: 2.0647, swd: 1.8320, ept: 229.9883
    Epoch [23/50], Test Losses: mse: 17.7384, mae: 2.2376, huber: 1.8414, swd: 1.6189, ept: 241.1815
      Epoch 23 composite train-obj: 1.748577
            Val objective improved 2.1181 → 2.0647, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 17.3913, mae: 2.1311, huber: 1.7408, swd: 1.4681, ept: 250.3200
    Epoch [24/50], Val Losses: mse: 24.8423, mae: 2.5625, huber: 2.1673, swd: 1.8079, ept: 234.7797
    Epoch [24/50], Test Losses: mse: 19.7763, mae: 2.2881, huber: 1.8959, swd: 1.6224, ept: 246.3899
      Epoch 24 composite train-obj: 1.740849
            No improvement (2.1673), counter 1/5
    Epoch [25/50], Train Losses: mse: 16.8454, mae: 2.0800, huber: 1.6928, swd: 1.4015, ept: 253.4903
    Epoch [25/50], Val Losses: mse: 23.2114, mae: 2.4852, huber: 2.0893, swd: 1.9079, ept: 238.4184
    Epoch [25/50], Test Losses: mse: 17.6144, mae: 2.1761, huber: 1.7846, swd: 1.6660, ept: 249.9563
      Epoch 25 composite train-obj: 1.692811
            No improvement (2.0893), counter 2/5
    Epoch [26/50], Train Losses: mse: 16.3277, mae: 2.0340, huber: 1.6493, swd: 1.3700, ept: 255.4436
    Epoch [26/50], Val Losses: mse: 19.9819, mae: 2.2567, huber: 1.8664, swd: 1.6170, ept: 243.4854
    Epoch [26/50], Test Losses: mse: 16.4672, mae: 2.0583, huber: 1.6727, swd: 1.4875, ept: 255.6985
      Epoch 26 composite train-obj: 1.649296
            Val objective improved 2.0647 → 1.8664, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 15.7116, mae: 1.9771, huber: 1.5955, swd: 1.2896, ept: 258.0782
    Epoch [27/50], Val Losses: mse: 21.6074, mae: 2.3820, huber: 1.9881, swd: 1.5580, ept: 241.8409
    Epoch [27/50], Test Losses: mse: 16.7198, mae: 2.0939, huber: 1.7045, swd: 1.3321, ept: 255.1349
      Epoch 27 composite train-obj: 1.595490
            No improvement (1.9881), counter 1/5
    Epoch [28/50], Train Losses: mse: 15.4264, mae: 1.9600, huber: 1.5784, swd: 1.2728, ept: 258.7995
    Epoch [28/50], Val Losses: mse: 18.2481, mae: 2.1464, huber: 1.7622, swd: 1.6389, ept: 249.9392
    Epoch [28/50], Test Losses: mse: 14.6103, mae: 1.9258, huber: 1.5475, swd: 1.3856, ept: 260.7548
      Epoch 28 composite train-obj: 1.578439
            Val objective improved 1.8664 → 1.7622, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 15.0378, mae: 1.9221, huber: 1.5428, swd: 1.2289, ept: 260.8071
    Epoch [29/50], Val Losses: mse: 19.4647, mae: 2.2633, huber: 1.8722, swd: 1.5854, ept: 243.4020
    Epoch [29/50], Test Losses: mse: 14.8516, mae: 1.9966, huber: 1.6097, swd: 1.3128, ept: 255.0617
      Epoch 29 composite train-obj: 1.542765
            No improvement (1.8722), counter 1/5
    Epoch [30/50], Train Losses: mse: 14.8502, mae: 1.9063, huber: 1.5277, swd: 1.1991, ept: 261.6249
    Epoch [30/50], Val Losses: mse: 22.5116, mae: 2.4120, huber: 2.0182, swd: 1.6141, ept: 237.7254
    Epoch [30/50], Test Losses: mse: 20.0329, mae: 2.2405, huber: 1.8510, swd: 1.4211, ept: 250.2566
      Epoch 30 composite train-obj: 1.527668
            No improvement (2.0182), counter 2/5
    Epoch [31/50], Train Losses: mse: 15.4113, mae: 1.9558, huber: 1.5740, swd: 1.2292, ept: 260.0691
    Epoch [31/50], Val Losses: mse: 20.9449, mae: 2.3100, huber: 1.9189, swd: 1.5342, ept: 241.5980
    Epoch [31/50], Test Losses: mse: 15.1906, mae: 1.9769, huber: 1.5918, swd: 1.2115, ept: 256.5787
      Epoch 31 composite train-obj: 1.574020
            No improvement (1.9189), counter 3/5
    Epoch [32/50], Train Losses: mse: 14.8414, mae: 1.9022, huber: 1.5237, swd: 1.1772, ept: 262.5953
    Epoch [32/50], Val Losses: mse: 18.3164, mae: 2.1092, huber: 1.7297, swd: 1.5930, ept: 251.2773
    Epoch [32/50], Test Losses: mse: 13.9821, mae: 1.8391, huber: 1.4675, swd: 1.3398, ept: 265.0072
      Epoch 32 composite train-obj: 1.523740
            Val objective improved 1.7622 → 1.7297, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 13.5405, mae: 1.7800, huber: 1.4098, swd: 1.0848, ept: 267.1176
    Epoch [33/50], Val Losses: mse: 18.8231, mae: 2.1920, huber: 1.8058, swd: 1.6023, ept: 245.9970
    Epoch [33/50], Test Losses: mse: 14.0448, mae: 1.9104, huber: 1.5307, swd: 1.3004, ept: 258.9137
      Epoch 33 composite train-obj: 1.409816
            No improvement (1.8058), counter 1/5
    Epoch [34/50], Train Losses: mse: 13.9494, mae: 1.8213, huber: 1.4474, swd: 1.0960, ept: 266.2560
    Epoch [34/50], Val Losses: mse: 19.8464, mae: 2.2374, huber: 1.8476, swd: 1.5908, ept: 246.8933
    Epoch [34/50], Test Losses: mse: 15.0273, mae: 1.9492, huber: 1.5658, swd: 1.3550, ept: 261.8896
      Epoch 34 composite train-obj: 1.447378
            No improvement (1.8476), counter 2/5
    Epoch [35/50], Train Losses: mse: 13.9960, mae: 1.8222, huber: 1.4485, swd: 1.0986, ept: 266.5808
    Epoch [35/50], Val Losses: mse: 15.8256, mae: 1.9793, huber: 1.6005, swd: 1.3932, ept: 254.0015
    Epoch [35/50], Test Losses: mse: 13.2019, mae: 1.8135, huber: 1.4390, swd: 1.2565, ept: 264.6856
      Epoch 35 composite train-obj: 1.448518
            Val objective improved 1.7297 → 1.6005, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 13.9254, mae: 1.8191, huber: 1.4446, swd: 1.0778, ept: 266.6415
    Epoch [36/50], Val Losses: mse: 17.1096, mae: 2.0352, huber: 1.6568, swd: 1.2851, ept: 252.8016
    Epoch [36/50], Test Losses: mse: 12.5866, mae: 1.7778, huber: 1.4035, swd: 1.1100, ept: 266.0199
      Epoch 36 composite train-obj: 1.444582
            No improvement (1.6568), counter 1/5
    Epoch [37/50], Train Losses: mse: 13.2620, mae: 1.7581, huber: 1.3883, swd: 1.0263, ept: 268.9605
    Epoch [37/50], Val Losses: mse: 17.7111, mae: 2.1198, huber: 1.7375, swd: 1.4795, ept: 247.7728
    Epoch [37/50], Test Losses: mse: 14.6691, mae: 1.9304, huber: 1.5511, swd: 1.2449, ept: 260.5269
      Epoch 37 composite train-obj: 1.388275
            No improvement (1.7375), counter 2/5
    Epoch [38/50], Train Losses: mse: 12.8071, mae: 1.7116, huber: 1.3447, swd: 0.9903, ept: 271.5255
    Epoch [38/50], Val Losses: mse: 21.4901, mae: 2.3369, huber: 1.9455, swd: 1.3861, ept: 243.1187
    Epoch [38/50], Test Losses: mse: 16.3272, mae: 2.0372, huber: 1.6515, swd: 1.2032, ept: 258.5824
      Epoch 38 composite train-obj: 1.344725
            No improvement (1.9455), counter 3/5
    Epoch [39/50], Train Losses: mse: 12.9958, mae: 1.7352, huber: 1.3664, swd: 0.9882, ept: 270.8742
    Epoch [39/50], Val Losses: mse: 19.1052, mae: 2.1023, huber: 1.7272, swd: 1.3850, ept: 255.0503
    Epoch [39/50], Test Losses: mse: 13.7960, mae: 1.7999, huber: 1.4317, swd: 1.1780, ept: 269.8528
      Epoch 39 composite train-obj: 1.366429
            No improvement (1.7272), counter 4/5
    Epoch [40/50], Train Losses: mse: 12.6367, mae: 1.6976, huber: 1.3312, swd: 0.9636, ept: 272.9378
    Epoch [40/50], Val Losses: mse: 17.3748, mae: 2.0263, huber: 1.6519, swd: 1.4196, ept: 254.8404
    Epoch [40/50], Test Losses: mse: 13.8524, mae: 1.8082, huber: 1.4395, swd: 1.2003, ept: 268.3127
      Epoch 40 composite train-obj: 1.331173
    Epoch [40/50], Test Losses: mse: 13.2025, mae: 1.8135, huber: 1.4390, swd: 1.2564, ept: 264.6993
    Best round's Test MSE: 13.2019, MAE: 1.8135, SWD: 1.2565
    Best round's Validation MSE: 15.8256, MAE: 1.9793, SWD: 1.3932
    Best round's Test verification MSE : 13.2025, MAE: 1.8135, SWD: 1.2564
    Time taken: 265.92 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 73.9763, mae: 6.6056, huber: 6.1248, swd: 42.5556, ept: 28.2858
    Epoch [1/50], Val Losses: mse: 56.5951, mae: 5.5563, huber: 5.0824, swd: 25.1836, ept: 58.8755
    Epoch [1/50], Test Losses: mse: 54.7712, mae: 5.4261, huber: 4.9530, swd: 25.9952, ept: 57.9677
      Epoch 1 composite train-obj: 6.124786
            Val objective improved inf → 5.0824, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 45.3755, mae: 4.6585, huber: 4.1954, swd: 14.9984, ept: 98.7785
    Epoch [2/50], Val Losses: mse: 44.8399, mae: 4.5741, huber: 4.1131, swd: 11.7914, ept: 110.1778
    Epoch [2/50], Test Losses: mse: 42.9053, mae: 4.4316, huber: 3.9717, swd: 11.8628, ept: 107.1005
      Epoch 2 composite train-obj: 4.195404
            Val objective improved 5.0824 → 4.1131, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.6453, mae: 4.1035, huber: 3.6518, swd: 9.7248, ept: 138.2155
    Epoch [3/50], Val Losses: mse: 42.1550, mae: 4.2808, huber: 3.8279, swd: 9.8318, ept: 142.1678
    Epoch [3/50], Test Losses: mse: 39.7962, mae: 4.1172, huber: 3.6661, swd: 9.6129, ept: 141.3821
      Epoch 3 composite train-obj: 3.651797
            Val objective improved 4.1131 → 3.8279, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.6751, mae: 3.8090, huber: 3.3644, swd: 7.4682, ept: 162.2396
    Epoch [4/50], Val Losses: mse: 38.7798, mae: 3.9816, huber: 3.5367, swd: 7.6361, ept: 155.1227
    Epoch [4/50], Test Losses: mse: 36.6832, mae: 3.8377, huber: 3.3936, swd: 7.6453, ept: 156.8347
      Epoch 4 composite train-obj: 3.364390
            Val objective improved 3.8279 → 3.5367, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.9457, mae: 3.5732, huber: 3.1345, swd: 6.1404, ept: 175.5714
    Epoch [5/50], Val Losses: mse: 36.9552, mae: 3.7987, huber: 3.3578, swd: 6.1184, ept: 168.5960
    Epoch [5/50], Test Losses: mse: 33.9919, mae: 3.6062, huber: 3.1675, swd: 5.8874, ept: 168.7044
      Epoch 5 composite train-obj: 3.134532
            Val objective improved 3.5367 → 3.3578, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.7595, mae: 3.3903, huber: 2.9562, swd: 5.0854, ept: 183.7705
    Epoch [6/50], Val Losses: mse: 36.8109, mae: 3.7505, huber: 3.3094, swd: 6.2167, ept: 168.7410
    Epoch [6/50], Test Losses: mse: 33.3744, mae: 3.5321, huber: 3.0937, swd: 5.8281, ept: 171.4744
      Epoch 6 composite train-obj: 2.956211
            Val objective improved 3.3578 → 3.3094, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 30.4757, mae: 3.2679, huber: 2.8379, swd: 4.6368, ept: 189.4490
    Epoch [7/50], Val Losses: mse: 35.2067, mae: 3.6026, huber: 3.1686, swd: 4.8468, ept: 175.8412
    Epoch [7/50], Test Losses: mse: 31.7888, mae: 3.3716, huber: 2.9407, swd: 4.5072, ept: 177.9701
      Epoch 7 composite train-obj: 2.837857
            Val objective improved 3.3094 → 3.1686, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 29.0721, mae: 3.1344, huber: 2.7087, swd: 4.1778, ept: 196.5351
    Epoch [8/50], Val Losses: mse: 34.2134, mae: 3.5083, huber: 3.0754, swd: 4.6631, ept: 180.4934
    Epoch [8/50], Test Losses: mse: 29.9560, mae: 3.2504, huber: 2.8202, swd: 4.1344, ept: 183.1656
      Epoch 8 composite train-obj: 2.708743
            Val objective improved 3.1686 → 3.0754, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 27.8648, mae: 3.0314, huber: 2.6088, swd: 3.8892, ept: 201.6103
    Epoch [9/50], Val Losses: mse: 32.7164, mae: 3.3939, huber: 2.9627, swd: 4.2676, ept: 182.3519
    Epoch [9/50], Test Losses: mse: 29.7950, mae: 3.2099, huber: 2.7810, swd: 3.9837, ept: 184.4337
      Epoch 9 composite train-obj: 2.608764
            Val objective improved 3.0754 → 2.9627, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 27.0761, mae: 2.9681, huber: 2.5475, swd: 3.6687, ept: 203.7718
    Epoch [10/50], Val Losses: mse: 37.0587, mae: 3.7105, huber: 3.2718, swd: 4.4689, ept: 177.2048
    Epoch [10/50], Test Losses: mse: 31.5743, mae: 3.3664, huber: 2.9320, swd: 4.2346, ept: 185.4092
      Epoch 10 composite train-obj: 2.547471
            No improvement (3.2718), counter 1/5
    Epoch [11/50], Train Losses: mse: 26.0288, mae: 2.8828, huber: 2.4649, swd: 3.4785, ept: 207.2106
    Epoch [11/50], Val Losses: mse: 31.4981, mae: 3.2315, huber: 2.8083, swd: 3.8653, ept: 194.5657
    Epoch [11/50], Test Losses: mse: 27.1685, mae: 2.9813, huber: 2.5604, swd: 3.5557, ept: 199.5534
      Epoch 11 composite train-obj: 2.464894
            Val objective improved 2.9627 → 2.8083, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 24.4920, mae: 2.7544, huber: 2.3407, swd: 3.1521, ept: 212.1913
    Epoch [12/50], Val Losses: mse: 30.4953, mae: 3.1756, huber: 2.7522, swd: 3.7589, ept: 190.6425
    Epoch [12/50], Test Losses: mse: 27.4270, mae: 2.9838, huber: 2.5632, swd: 3.3357, ept: 197.0777
      Epoch 12 composite train-obj: 2.340663
            Val objective improved 2.8083 → 2.7522, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 24.1451, mae: 2.7255, huber: 2.3124, swd: 2.9814, ept: 213.3727
    Epoch [13/50], Val Losses: mse: 29.4735, mae: 3.1278, huber: 2.7048, swd: 3.6889, ept: 195.2101
    Epoch [13/50], Test Losses: mse: 26.5290, mae: 2.9300, huber: 2.5102, swd: 3.4241, ept: 201.4001
      Epoch 13 composite train-obj: 2.312431
            Val objective improved 2.7522 → 2.7048, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 22.7931, mae: 2.6059, huber: 2.1978, swd: 2.7505, ept: 218.0393
    Epoch [14/50], Val Losses: mse: 25.6687, mae: 2.8421, huber: 2.4286, swd: 3.2027, ept: 207.2969
    Epoch [14/50], Test Losses: mse: 22.7545, mae: 2.6611, huber: 2.2490, swd: 2.9075, ept: 212.9638
      Epoch 14 composite train-obj: 2.197819
            Val objective improved 2.7048 → 2.4286, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 22.8932, mae: 2.6117, huber: 2.2025, swd: 2.6431, ept: 218.4769
    Epoch [15/50], Val Losses: mse: 28.0442, mae: 3.0227, huber: 2.6021, swd: 3.1904, ept: 197.9601
    Epoch [15/50], Test Losses: mse: 24.2195, mae: 2.7835, huber: 2.3671, swd: 2.8468, ept: 203.6820
      Epoch 15 composite train-obj: 2.202498
            No improvement (2.6021), counter 1/5
    Epoch [16/50], Train Losses: mse: 22.2324, mae: 2.5545, huber: 2.1477, swd: 2.5232, ept: 220.8222
    Epoch [16/50], Val Losses: mse: 25.6976, mae: 2.7848, huber: 2.3761, swd: 3.2685, ept: 207.4706
    Epoch [16/50], Test Losses: mse: 21.4856, mae: 2.5401, huber: 2.1341, swd: 2.8301, ept: 214.2361
      Epoch 16 composite train-obj: 2.147702
            Val objective improved 2.4286 → 2.3761, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 21.2619, mae: 2.4635, huber: 2.0607, swd: 2.3611, ept: 224.6442
    Epoch [17/50], Val Losses: mse: 32.3269, mae: 3.2271, huber: 2.8050, swd: 3.2626, ept: 198.7146
    Epoch [17/50], Test Losses: mse: 28.5937, mae: 3.0118, huber: 2.5917, swd: 3.1077, ept: 201.4948
      Epoch 17 composite train-obj: 2.060699
            No improvement (2.8050), counter 1/5
    Epoch [18/50], Train Losses: mse: 21.8843, mae: 2.5201, huber: 2.1143, swd: 2.4303, ept: 223.1465
    Epoch [18/50], Val Losses: mse: 28.2794, mae: 2.9494, huber: 2.5326, swd: 2.7566, ept: 208.5046
    Epoch [18/50], Test Losses: mse: 22.9592, mae: 2.6563, huber: 2.2420, swd: 2.2994, ept: 212.3847
      Epoch 18 composite train-obj: 2.114258
            No improvement (2.5326), counter 2/5
    Epoch [19/50], Train Losses: mse: 20.0233, mae: 2.3596, huber: 1.9613, swd: 2.1121, ept: 228.9144
    Epoch [19/50], Val Losses: mse: 27.3310, mae: 2.8185, huber: 2.4121, swd: 2.7430, ept: 210.7474
    Epoch [19/50], Test Losses: mse: 23.2936, mae: 2.5923, huber: 2.1884, swd: 2.4925, ept: 216.6464
      Epoch 19 composite train-obj: 1.961348
            No improvement (2.4121), counter 3/5
    Epoch [20/50], Train Losses: mse: 19.7809, mae: 2.3426, huber: 1.9441, swd: 2.0338, ept: 229.9125
    Epoch [20/50], Val Losses: mse: 27.7078, mae: 2.8557, huber: 2.4459, swd: 2.7989, ept: 212.1741
    Epoch [20/50], Test Losses: mse: 23.4895, mae: 2.6015, huber: 2.1965, swd: 2.6089, ept: 220.9661
      Epoch 20 composite train-obj: 1.944125
            No improvement (2.4459), counter 4/5
    Epoch [21/50], Train Losses: mse: 19.7313, mae: 2.3266, huber: 1.9293, swd: 2.0206, ept: 232.2897
    Epoch [21/50], Val Losses: mse: 25.6474, mae: 2.7241, huber: 2.3173, swd: 2.5105, ept: 216.5309
    Epoch [21/50], Test Losses: mse: 20.3385, mae: 2.4119, huber: 2.0088, swd: 2.0869, ept: 226.2982
      Epoch 21 composite train-obj: 1.929295
            Val objective improved 2.3761 → 2.3173, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 18.9759, mae: 2.2689, huber: 1.8734, swd: 1.9015, ept: 234.5813
    Epoch [22/50], Val Losses: mse: 24.0973, mae: 2.6510, huber: 2.2459, swd: 2.4325, ept: 215.8669
    Epoch [22/50], Test Losses: mse: 20.8447, mae: 2.4590, huber: 2.0560, swd: 2.2514, ept: 224.5854
      Epoch 22 composite train-obj: 1.873409
            Val objective improved 2.3173 → 2.2459, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 18.8178, mae: 2.2495, huber: 1.8551, swd: 1.8599, ept: 238.2414
    Epoch [23/50], Val Losses: mse: 25.4047, mae: 2.6989, huber: 2.2921, swd: 2.3063, ept: 220.9626
    Epoch [23/50], Test Losses: mse: 19.5584, mae: 2.3666, huber: 1.9645, swd: 1.9512, ept: 233.6087
      Epoch 23 composite train-obj: 1.855119
            No improvement (2.2921), counter 1/5
    Epoch [24/50], Train Losses: mse: 18.1988, mae: 2.1981, huber: 1.8058, swd: 1.7593, ept: 241.0076
    Epoch [24/50], Val Losses: mse: 22.9422, mae: 2.5393, huber: 2.1370, swd: 2.1776, ept: 224.1020
    Epoch [24/50], Test Losses: mse: 19.1083, mae: 2.3198, huber: 1.9215, swd: 1.9335, ept: 232.6250
      Epoch 24 composite train-obj: 1.805820
            Val objective improved 2.2459 → 2.1370, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 18.2159, mae: 2.1984, huber: 1.8059, swd: 1.7346, ept: 241.9049
    Epoch [25/50], Val Losses: mse: 22.7529, mae: 2.5687, huber: 2.1621, swd: 2.2079, ept: 222.6872
    Epoch [25/50], Test Losses: mse: 18.7821, mae: 2.3352, huber: 1.9319, swd: 1.8494, ept: 232.4771
      Epoch 25 composite train-obj: 1.805907
            No improvement (2.1621), counter 1/5
    Epoch [26/50], Train Losses: mse: 17.5311, mae: 2.1428, huber: 1.7527, swd: 1.6646, ept: 244.1820
    Epoch [26/50], Val Losses: mse: 24.3334, mae: 2.6458, huber: 2.2381, swd: 2.3731, ept: 218.9897
    Epoch [26/50], Test Losses: mse: 20.6496, mae: 2.4453, huber: 2.0397, swd: 2.0783, ept: 228.9504
      Epoch 26 composite train-obj: 1.752708
            No improvement (2.2381), counter 2/5
    Epoch [27/50], Train Losses: mse: 17.3863, mae: 2.1302, huber: 1.7401, swd: 1.6196, ept: 245.7188
    Epoch [27/50], Val Losses: mse: 24.8524, mae: 2.6298, huber: 2.2270, swd: 2.0721, ept: 224.7196
    Epoch [27/50], Test Losses: mse: 19.6543, mae: 2.3409, huber: 1.9419, swd: 1.7916, ept: 234.8265
      Epoch 27 composite train-obj: 1.740114
            No improvement (2.2270), counter 3/5
    Epoch [28/50], Train Losses: mse: 17.1462, mae: 2.1062, huber: 1.7175, swd: 1.6137, ept: 247.2475
    Epoch [28/50], Val Losses: mse: 23.1365, mae: 2.5053, huber: 2.1081, swd: 1.9670, ept: 229.3521
    Epoch [28/50], Test Losses: mse: 18.3755, mae: 2.2310, huber: 1.8385, swd: 1.6658, ept: 240.7635
      Epoch 28 composite train-obj: 1.717511
            Val objective improved 2.1370 → 2.1081, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 16.7344, mae: 2.0773, huber: 1.6893, swd: 1.5090, ept: 248.6008
    Epoch [29/50], Val Losses: mse: 23.4493, mae: 2.4991, huber: 2.1013, swd: 1.9308, ept: 231.2088
    Epoch [29/50], Test Losses: mse: 18.7712, mae: 2.2186, huber: 1.8283, swd: 1.7085, ept: 243.8337
      Epoch 29 composite train-obj: 1.689279
            Val objective improved 2.1081 → 2.1013, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 16.0301, mae: 2.0131, huber: 1.6288, swd: 1.4647, ept: 251.6110
    Epoch [30/50], Val Losses: mse: 25.3610, mae: 2.6271, huber: 2.2237, swd: 1.9220, ept: 231.5517
    Epoch [30/50], Test Losses: mse: 19.4955, mae: 2.3115, huber: 1.9125, swd: 1.6229, ept: 242.3810
      Epoch 30 composite train-obj: 1.628795
            No improvement (2.2237), counter 1/5
    Epoch [31/50], Train Losses: mse: 15.7767, mae: 1.9879, huber: 1.6049, swd: 1.4153, ept: 253.0240
    Epoch [31/50], Val Losses: mse: 23.1974, mae: 2.4995, huber: 2.0996, swd: 2.0986, ept: 231.5098
    Epoch [31/50], Test Losses: mse: 18.2338, mae: 2.2066, huber: 1.8132, swd: 1.6669, ept: 243.8048
      Epoch 31 composite train-obj: 1.604876
            Val objective improved 2.1013 → 2.0996, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 15.3478, mae: 1.9526, huber: 1.5715, swd: 1.3661, ept: 254.2140
    Epoch [32/50], Val Losses: mse: 22.3756, mae: 2.4591, huber: 2.0597, swd: 1.8552, ept: 231.3358
    Epoch [32/50], Test Losses: mse: 16.8919, mae: 2.1625, huber: 1.7660, swd: 1.5311, ept: 241.7288
      Epoch 32 composite train-obj: 1.571500
            Val objective improved 2.0996 → 2.0597, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 15.9599, mae: 2.0069, huber: 1.6220, swd: 1.4011, ept: 253.3584
    Epoch [33/50], Val Losses: mse: 22.0334, mae: 2.4221, huber: 2.0267, swd: 1.8413, ept: 231.1236
    Epoch [33/50], Test Losses: mse: 18.2166, mae: 2.2059, huber: 1.8140, swd: 1.6691, ept: 243.1236
      Epoch 33 composite train-obj: 1.622037
            Val objective improved 2.0597 → 2.0267, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 14.6829, mae: 1.8945, huber: 1.5160, swd: 1.2839, ept: 257.2402
    Epoch [34/50], Val Losses: mse: 21.4677, mae: 2.3907, huber: 1.9941, swd: 1.7531, ept: 234.1742
    Epoch [34/50], Test Losses: mse: 16.7921, mae: 2.1155, huber: 1.7226, swd: 1.3654, ept: 246.6923
      Epoch 34 composite train-obj: 1.516038
            Val objective improved 2.0267 → 1.9941, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 15.0609, mae: 1.9254, huber: 1.5454, swd: 1.2869, ept: 256.2342
    Epoch [35/50], Val Losses: mse: 21.8396, mae: 2.4063, huber: 2.0090, swd: 1.8909, ept: 238.9337
    Epoch [35/50], Test Losses: mse: 16.2570, mae: 2.0950, huber: 1.7033, swd: 1.5793, ept: 250.8154
      Epoch 35 composite train-obj: 1.545385
            No improvement (2.0090), counter 1/5
    Epoch [36/50], Train Losses: mse: 17.7917, mae: 2.1654, huber: 1.7721, swd: 1.5093, ept: 248.5707
    Epoch [36/50], Val Losses: mse: 20.3108, mae: 2.3296, huber: 1.9338, swd: 1.8566, ept: 241.3803
    Epoch [36/50], Test Losses: mse: 16.9175, mae: 2.1151, huber: 1.7241, swd: 1.5288, ept: 251.2629
      Epoch 36 composite train-obj: 1.772096
            Val objective improved 1.9941 → 1.9338, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 15.7496, mae: 1.9914, huber: 1.6068, swd: 1.3350, ept: 254.7565
    Epoch [37/50], Val Losses: mse: 23.4580, mae: 2.5150, huber: 2.1146, swd: 1.6483, ept: 232.9722
    Epoch [37/50], Test Losses: mse: 19.4028, mae: 2.2914, huber: 1.8948, swd: 1.4463, ept: 243.5835
      Epoch 37 composite train-obj: 1.606788
            No improvement (2.1146), counter 1/5
    Epoch [38/50], Train Losses: mse: 15.0981, mae: 1.9264, huber: 1.5462, swd: 1.2731, ept: 258.1344
    Epoch [38/50], Val Losses: mse: 20.7398, mae: 2.3027, huber: 1.9113, swd: 1.6369, ept: 241.9559
    Epoch [38/50], Test Losses: mse: 17.2338, mae: 2.1104, huber: 1.7235, swd: 1.3873, ept: 251.2993
      Epoch 38 composite train-obj: 1.546151
            Val objective improved 1.9338 → 1.9113, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 14.3107, mae: 1.8637, huber: 1.4863, swd: 1.1987, ept: 260.4105
    Epoch [39/50], Val Losses: mse: 23.1903, mae: 2.4310, huber: 2.0340, swd: 1.7280, ept: 240.3456
    Epoch [39/50], Test Losses: mse: 18.1589, mae: 2.1709, huber: 1.7782, swd: 1.4739, ept: 250.6333
      Epoch 39 composite train-obj: 1.486265
            No improvement (2.0340), counter 1/5
    Epoch [40/50], Train Losses: mse: 14.0551, mae: 1.8344, huber: 1.4593, swd: 1.1824, ept: 262.3601
    Epoch [40/50], Val Losses: mse: 18.2704, mae: 2.1650, huber: 1.7770, swd: 1.6624, ept: 243.0135
    Epoch [40/50], Test Losses: mse: 14.5541, mae: 1.9444, huber: 1.5601, swd: 1.3844, ept: 256.0301
      Epoch 40 composite train-obj: 1.459257
            Val objective improved 1.9113 → 1.7770, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 14.7338, mae: 1.8809, huber: 1.5034, swd: 1.2287, ept: 261.1017
    Epoch [41/50], Val Losses: mse: 20.7858, mae: 2.3369, huber: 1.9424, swd: 1.8587, ept: 237.5618
    Epoch [41/50], Test Losses: mse: 16.3487, mae: 2.0831, huber: 1.6923, swd: 1.5218, ept: 250.0141
      Epoch 41 composite train-obj: 1.503354
            No improvement (1.9424), counter 1/5
    Epoch [42/50], Train Losses: mse: 13.7039, mae: 1.8050, huber: 1.4311, swd: 1.1249, ept: 263.8504
    Epoch [42/50], Val Losses: mse: 19.3278, mae: 2.2642, huber: 1.8753, swd: 1.6336, ept: 236.2590
    Epoch [42/50], Test Losses: mse: 15.3841, mae: 2.0334, huber: 1.6484, swd: 1.3343, ept: 248.4951
      Epoch 42 composite train-obj: 1.431085
            No improvement (1.8753), counter 2/5
    Epoch [43/50], Train Losses: mse: 12.9986, mae: 1.7453, huber: 1.3744, swd: 1.0797, ept: 266.3374
    Epoch [43/50], Val Losses: mse: 19.9356, mae: 2.2524, huber: 1.8642, swd: 1.6346, ept: 241.3774
    Epoch [43/50], Test Losses: mse: 15.6439, mae: 1.9932, huber: 1.6106, swd: 1.3294, ept: 253.6954
      Epoch 43 composite train-obj: 1.374379
            No improvement (1.8642), counter 3/5
    Epoch [44/50], Train Losses: mse: 13.0665, mae: 1.7480, huber: 1.3773, swd: 1.0651, ept: 266.4405
    Epoch [44/50], Val Losses: mse: 21.4339, mae: 2.3311, huber: 1.9405, swd: 1.7679, ept: 241.8083
    Epoch [44/50], Test Losses: mse: 15.9728, mae: 2.0293, huber: 1.6440, swd: 1.4896, ept: 252.6872
      Epoch 44 composite train-obj: 1.377256
            No improvement (1.9405), counter 4/5
    Epoch [45/50], Train Losses: mse: 12.3120, mae: 1.6791, huber: 1.3125, swd: 1.0120, ept: 269.1016
    Epoch [45/50], Val Losses: mse: 17.4883, mae: 2.0741, huber: 1.6935, swd: 1.5838, ept: 248.0914
    Epoch [45/50], Test Losses: mse: 14.8332, mae: 1.9170, huber: 1.5396, swd: 1.3457, ept: 259.4318
      Epoch 45 composite train-obj: 1.312504
            Val objective improved 1.7770 → 1.6935, saving checkpoint.
    Epoch [46/50], Train Losses: mse: 13.3153, mae: 1.7690, huber: 1.3962, swd: 1.0709, ept: 266.6062
    Epoch [46/50], Val Losses: mse: 20.3770, mae: 2.2186, huber: 1.8344, swd: 1.5314, ept: 246.2421
    Epoch [46/50], Test Losses: mse: 16.3415, mae: 1.9712, huber: 1.5934, swd: 1.2633, ept: 260.7153
      Epoch 46 composite train-obj: 1.396219
            No improvement (1.8344), counter 1/5
    Epoch [47/50], Train Losses: mse: 12.9473, mae: 1.7327, huber: 1.3629, swd: 1.0397, ept: 268.3737
    Epoch [47/50], Val Losses: mse: 20.5961, mae: 2.1936, huber: 1.8116, swd: 1.3729, ept: 251.0385
    Epoch [47/50], Test Losses: mse: 16.4363, mae: 1.9676, huber: 1.5916, swd: 1.2713, ept: 262.6610
      Epoch 47 composite train-obj: 1.362904
            No improvement (1.8116), counter 2/5
    Epoch [48/50], Train Losses: mse: 12.1861, mae: 1.6650, huber: 1.2991, swd: 0.9861, ept: 270.8241
    Epoch [48/50], Val Losses: mse: 18.6667, mae: 2.1374, huber: 1.7574, swd: 1.5843, ept: 247.9395
    Epoch [48/50], Test Losses: mse: 13.6111, mae: 1.8474, huber: 1.4717, swd: 1.2440, ept: 262.1698
      Epoch 48 composite train-obj: 1.299052
            No improvement (1.7574), counter 3/5
    Epoch [49/50], Train Losses: mse: 12.5316, mae: 1.6966, huber: 1.3285, swd: 1.0029, ept: 269.9264
    Epoch [49/50], Val Losses: mse: 17.3212, mae: 2.0327, huber: 1.6546, swd: 1.4791, ept: 255.8113
    Epoch [49/50], Test Losses: mse: 11.6897, mae: 1.6978, huber: 1.3258, swd: 1.1107, ept: 269.6877
      Epoch 49 composite train-obj: 1.328515
            Val objective improved 1.6935 → 1.6546, saving checkpoint.
    Epoch [50/50], Train Losses: mse: 12.1478, mae: 1.6662, huber: 1.3001, swd: 0.9698, ept: 271.1022
    Epoch [50/50], Val Losses: mse: 18.8153, mae: 2.1633, huber: 1.7810, swd: 1.5238, ept: 245.6544
    Epoch [50/50], Test Losses: mse: 13.7820, mae: 1.8568, huber: 1.4826, swd: 1.1933, ept: 261.6693
      Epoch 50 composite train-obj: 1.300071
            No improvement (1.7810), counter 1/5
    Epoch [50/50], Test Losses: mse: 11.6887, mae: 1.6977, huber: 1.3257, swd: 1.1103, ept: 269.6948
    Best round's Test MSE: 11.6897, MAE: 1.6978, SWD: 1.1107
    Best round's Validation MSE: 17.3212, MAE: 2.0327, SWD: 1.4791
    Best round's Test verification MSE : 11.6887, MAE: 1.6977, SWD: 1.1103
    Time taken: 328.45 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250510_1723)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 13.2041 ± 1.2374
      mae: 1.8178 ± 0.0997
      huber: 1.4417 ± 0.0957
      swd: 1.2380 ± 0.0972
      ept: 264.3132 ± 4.5480
      count: 36.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 17.2185 ± 1.0977
      mae: 2.0642 ± 0.0852
      huber: 1.6827 ± 0.0810
      swd: 1.4759 ± 0.0662
      ept: 252.0959 ± 4.0427
      count: 36.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 862.07 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250510_1723
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    


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
    # single_magnitude_for_shift=True,
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False,
    

)
cfg.x_to_z_delay.enable_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_delay.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_delay.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_delay.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_delay.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.x_to_z_deri.enable_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_scale_shift = [True, False]
cfg.x_to_z_deri.spectral_flags_magnitudes = [False, True]
cfg.x_to_z_deri.spectral_flags_hidden_layers = [False, False]
cfg.x_to_z_deri.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.x_to_z_deri.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_to_x_main.enable_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_x_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_x_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_x_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_x_main.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_push_to_z.enable_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_scale_shift = [True, False]
cfg.z_push_to_z.spectral_flags_magnitudes = [False, True]
cfg.z_push_to_z.spectral_flags_hidden_layers = [False, False]
cfg.z_push_to_z.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_push_to_z.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]

cfg.z_to_y_main.enable_magnitudes = [False, False]
cfg.z_to_y_main.spectral_flags_scale_shift = [True, False]
cfg.z_to_y_main.spectral_flags_magnitudes = [False, True]
cfg.z_to_y_main.spectral_flags_hidden_layers = [False, False]
cfg.z_to_y_main.activations_scale_shift = ["relu6", "dynamic_tanh"]
cfg.z_to_y_main.activations_hidden_layers = [nn.ELU,
                                              nn.LogSigmoid]
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 280
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
    Training Batches: 280
    Validation Batches: 36
    Test Batches: 77
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 61.8879, mae: 5.7381, huber: 5.2648, swd: 30.2303, ept: 57.8880
    Epoch [1/50], Val Losses: mse: 49.1995, mae: 4.8764, huber: 4.4128, swd: 12.6877, ept: 98.4557
    Epoch [1/50], Test Losses: mse: 46.7910, mae: 4.7140, huber: 4.2520, swd: 12.7656, ept: 95.8533
      Epoch 1 composite train-obj: 5.264789
            Val objective improved inf → 4.4128, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 38.2853, mae: 4.0104, huber: 3.5594, swd: 7.7271, ept: 147.3143
    Epoch [2/50], Val Losses: mse: 40.2135, mae: 4.1026, huber: 3.6521, swd: 6.4403, ept: 154.3859
    Epoch [2/50], Test Losses: mse: 37.2356, mae: 3.9268, huber: 3.4774, swd: 6.3785, ept: 157.0667
      Epoch 2 composite train-obj: 3.559364
            Val objective improved 4.4128 → 3.6521, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 32.3933, mae: 3.4779, huber: 3.0390, swd: 4.4324, ept: 179.1805
    Epoch [3/50], Val Losses: mse: 37.4352, mae: 3.8257, huber: 3.3829, swd: 5.2524, ept: 164.0011
    Epoch [3/50], Test Losses: mse: 32.9147, mae: 3.5391, huber: 3.0995, swd: 4.8787, ept: 168.5209
      Epoch 3 composite train-obj: 3.038955
            Val objective improved 3.6521 → 3.3829, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 29.2320, mae: 3.1874, huber: 2.7569, swd: 3.4698, ept: 193.3707
    Epoch [4/50], Val Losses: mse: 35.1542, mae: 3.5542, huber: 3.1186, swd: 3.9259, ept: 184.1266
    Epoch [4/50], Test Losses: mse: 30.0878, mae: 3.2653, huber: 2.8324, swd: 3.7216, ept: 190.7933
      Epoch 4 composite train-obj: 2.756949
            Val objective improved 3.3829 → 3.1186, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 26.8077, mae: 2.9523, huber: 2.5308, swd: 2.8451, ept: 206.9352
    Epoch [5/50], Val Losses: mse: 35.7612, mae: 3.5200, huber: 3.0886, swd: 3.5202, ept: 185.5062
    Epoch [5/50], Test Losses: mse: 29.6181, mae: 3.2061, huber: 2.7773, swd: 3.5185, ept: 192.9431
      Epoch 5 composite train-obj: 2.530792
            Val objective improved 3.1186 → 3.0886, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 25.2889, mae: 2.8199, huber: 2.4032, swd: 2.5376, ept: 214.4022
    Epoch [6/50], Val Losses: mse: 33.5694, mae: 3.3588, huber: 2.9325, swd: 3.0658, ept: 191.3684
    Epoch [6/50], Test Losses: mse: 28.2494, mae: 3.0773, huber: 2.6529, swd: 3.0052, ept: 198.7031
      Epoch 6 composite train-obj: 2.403242
            Val objective improved 3.0886 → 2.9325, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 23.6177, mae: 2.6848, huber: 2.2722, swd: 2.2752, ept: 220.0971
    Epoch [7/50], Val Losses: mse: 30.8494, mae: 3.1014, huber: 2.6849, swd: 2.6571, ept: 207.1922
    Epoch [7/50], Test Losses: mse: 25.5247, mae: 2.8146, huber: 2.4011, swd: 2.4629, ept: 216.1701
      Epoch 7 composite train-obj: 2.272175
            Val objective improved 2.9325 → 2.6849, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 22.0118, mae: 2.5473, huber: 2.1398, swd: 2.0182, ept: 225.9558
    Epoch [8/50], Val Losses: mse: 29.9553, mae: 3.0615, huber: 2.6412, swd: 2.5484, ept: 208.5941
    Epoch [8/50], Test Losses: mse: 23.9051, mae: 2.7317, huber: 2.3148, swd: 2.3039, ept: 217.9255
      Epoch 8 composite train-obj: 2.139810
            Val objective improved 2.6849 → 2.6412, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 20.6978, mae: 2.4524, huber: 2.0469, swd: 1.8298, ept: 229.8360
    Epoch [9/50], Val Losses: mse: 30.0030, mae: 3.0490, huber: 2.6328, swd: 2.5310, ept: 207.3910
    Epoch [9/50], Test Losses: mse: 24.8936, mae: 2.7775, huber: 2.3637, swd: 2.3817, ept: 214.4271
      Epoch 9 composite train-obj: 2.046903
            Val objective improved 2.6412 → 2.6328, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 19.2959, mae: 2.3383, huber: 1.9370, swd: 1.6617, ept: 234.4720
    Epoch [10/50], Val Losses: mse: 28.8565, mae: 2.8932, huber: 2.4821, swd: 1.8324, ept: 219.1241
    Epoch [10/50], Test Losses: mse: 22.2573, mae: 2.5595, huber: 2.1519, swd: 1.7647, ept: 227.3090
      Epoch 10 composite train-obj: 1.937013
            Val objective improved 2.6328 → 2.4821, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 17.8739, mae: 2.2119, huber: 1.8163, swd: 1.4884, ept: 240.0318
    Epoch [11/50], Val Losses: mse: 27.5443, mae: 2.8668, huber: 2.4545, swd: 2.0231, ept: 217.4376
    Epoch [11/50], Test Losses: mse: 23.7847, mae: 2.6733, huber: 2.2633, swd: 1.9718, ept: 223.1376
      Epoch 11 composite train-obj: 1.816343
            Val objective improved 2.4821 → 2.4545, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 17.5722, mae: 2.1939, huber: 1.7977, swd: 1.4582, ept: 241.6353
    Epoch [12/50], Val Losses: mse: 26.0631, mae: 2.6926, huber: 2.2890, swd: 1.9639, ept: 225.6370
    Epoch [12/50], Test Losses: mse: 21.4784, mae: 2.4400, huber: 2.0400, swd: 1.8723, ept: 235.1305
      Epoch 12 composite train-obj: 1.797717
            Val objective improved 2.4545 → 2.2890, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 16.3377, mae: 2.0874, huber: 1.6962, swd: 1.3070, ept: 246.8691
    Epoch [13/50], Val Losses: mse: 25.0701, mae: 2.7249, huber: 2.3145, swd: 1.9297, ept: 219.3925
    Epoch [13/50], Test Losses: mse: 20.0983, mae: 2.4376, huber: 2.0320, swd: 1.6289, ept: 227.8319
      Epoch 13 composite train-obj: 1.696215
            No improvement (2.3145), counter 1/5
    Epoch [14/50], Train Losses: mse: 16.4135, mae: 2.0878, huber: 1.6967, swd: 1.3112, ept: 247.9013
    Epoch [14/50], Val Losses: mse: 25.1090, mae: 2.6960, huber: 2.2897, swd: 1.9228, ept: 224.2419
    Epoch [14/50], Test Losses: mse: 20.4811, mae: 2.4287, huber: 2.0270, swd: 1.7695, ept: 231.5183
      Epoch 14 composite train-obj: 1.696677
            No improvement (2.2897), counter 2/5
    Epoch [15/50], Train Losses: mse: 15.4489, mae: 2.0070, huber: 1.6195, swd: 1.2161, ept: 251.4801
    Epoch [15/50], Val Losses: mse: 22.6591, mae: 2.4924, huber: 2.0959, swd: 2.0253, ept: 230.0425
    Epoch [15/50], Test Losses: mse: 17.4475, mae: 2.1953, huber: 1.8029, swd: 1.8059, ept: 238.8411
      Epoch 15 composite train-obj: 1.619506
            Val objective improved 2.2890 → 2.0959, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 14.5657, mae: 1.9292, huber: 1.5456, swd: 1.1264, ept: 254.9037
    Epoch [16/50], Val Losses: mse: 21.2689, mae: 2.3911, huber: 1.9953, swd: 1.6137, ept: 234.7799
    Epoch [16/50], Test Losses: mse: 17.9631, mae: 2.1986, huber: 1.8064, swd: 1.4911, ept: 243.9381
      Epoch 16 composite train-obj: 1.545578
            Val objective improved 2.0959 → 1.9953, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 13.8558, mae: 1.8626, huber: 1.4827, swd: 1.0629, ept: 257.7453
    Epoch [17/50], Val Losses: mse: 21.5426, mae: 2.3679, huber: 1.9760, swd: 1.6667, ept: 239.0353
    Epoch [17/50], Test Losses: mse: 17.6002, mae: 2.1571, huber: 1.7682, swd: 1.5645, ept: 248.0111
      Epoch 17 composite train-obj: 1.482717
            Val objective improved 1.9953 → 1.9760, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 14.0876, mae: 1.8862, huber: 1.5039, swd: 1.0883, ept: 257.5297
    Epoch [18/50], Val Losses: mse: 22.5464, mae: 2.4559, huber: 2.0583, swd: 1.5868, ept: 238.4744
    Epoch [18/50], Test Losses: mse: 17.6179, mae: 2.1723, huber: 1.7798, swd: 1.3587, ept: 245.6119
      Epoch 18 composite train-obj: 1.503946
            No improvement (2.0583), counter 1/5
    Epoch [19/50], Train Losses: mse: 13.8069, mae: 1.8632, huber: 1.4821, swd: 1.0585, ept: 258.5824
    Epoch [19/50], Val Losses: mse: 19.6867, mae: 2.2782, huber: 1.8844, swd: 1.5383, ept: 242.6770
    Epoch [19/50], Test Losses: mse: 15.2337, mae: 2.0117, huber: 1.6229, swd: 1.3729, ept: 252.4704
      Epoch 19 composite train-obj: 1.482119
            Val objective improved 1.9760 → 1.8844, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 12.8623, mae: 1.7625, huber: 1.3889, swd: 0.9613, ept: 263.1342
    Epoch [20/50], Val Losses: mse: 20.7240, mae: 2.3101, huber: 1.9194, swd: 1.4760, ept: 241.9226
    Epoch [20/50], Test Losses: mse: 16.3215, mae: 2.0581, huber: 1.6725, swd: 1.3458, ept: 252.3208
      Epoch 20 composite train-obj: 1.388921
            No improvement (1.9194), counter 1/5
    Epoch [21/50], Train Losses: mse: 12.6650, mae: 1.7470, huber: 1.3738, swd: 0.9436, ept: 264.0306
    Epoch [21/50], Val Losses: mse: 21.4508, mae: 2.3177, huber: 1.9317, swd: 1.5639, ept: 243.6623
    Epoch [21/50], Test Losses: mse: 16.1084, mae: 2.0101, huber: 1.6293, swd: 1.2620, ept: 253.1320
      Epoch 21 composite train-obj: 1.373821
            No improvement (1.9317), counter 2/5
    Epoch [22/50], Train Losses: mse: 12.1791, mae: 1.7130, huber: 1.3400, swd: 0.9192, ept: 265.8611
    Epoch [22/50], Val Losses: mse: 18.0572, mae: 2.1040, huber: 1.7218, swd: 1.4308, ept: 249.7052
    Epoch [22/50], Test Losses: mse: 14.6415, mae: 1.9101, huber: 1.5305, swd: 1.3176, ept: 258.8754
      Epoch 22 composite train-obj: 1.339985
            Val objective improved 1.8844 → 1.7218, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 12.0257, mae: 1.6921, huber: 1.3212, swd: 0.8946, ept: 267.0963
    Epoch [23/50], Val Losses: mse: 19.7399, mae: 2.1762, huber: 1.7960, swd: 1.3152, ept: 251.0128
    Epoch [23/50], Test Losses: mse: 15.3527, mae: 1.9157, huber: 1.5420, swd: 1.1839, ept: 261.4732
      Epoch 23 composite train-obj: 1.321201
            No improvement (1.7960), counter 1/5
    Epoch [24/50], Train Losses: mse: 11.5483, mae: 1.6495, huber: 1.2812, swd: 0.8588, ept: 268.9641
    Epoch [24/50], Val Losses: mse: 20.1920, mae: 2.2543, huber: 1.8650, swd: 1.3613, ept: 247.1745
    Epoch [24/50], Test Losses: mse: 15.0371, mae: 1.9496, huber: 1.5670, swd: 1.1595, ept: 258.6608
      Epoch 24 composite train-obj: 1.281200
            No improvement (1.8650), counter 2/5
    Epoch [25/50], Train Losses: mse: 11.7321, mae: 1.6608, huber: 1.2924, swd: 0.8587, ept: 268.8492
    Epoch [25/50], Val Losses: mse: 20.2779, mae: 2.2456, huber: 1.8555, swd: 1.3831, ept: 246.1325
    Epoch [25/50], Test Losses: mse: 15.5192, mae: 1.9747, huber: 1.5888, swd: 1.1821, ept: 255.6514
      Epoch 25 composite train-obj: 1.292359
            No improvement (1.8555), counter 3/5
    Epoch [26/50], Train Losses: mse: 11.2036, mae: 1.6131, huber: 1.2474, swd: 0.8333, ept: 271.0985
    Epoch [26/50], Val Losses: mse: 19.9868, mae: 2.2347, huber: 1.8425, swd: 1.3292, ept: 246.3471
    Epoch [26/50], Test Losses: mse: 16.0151, mae: 2.0166, huber: 1.6292, swd: 1.1863, ept: 255.5232
      Epoch 26 composite train-obj: 1.247432
            No improvement (1.8425), counter 4/5
    Epoch [27/50], Train Losses: mse: 10.6392, mae: 1.5619, huber: 1.1998, swd: 0.7831, ept: 272.9193
    Epoch [27/50], Val Losses: mse: 20.7610, mae: 2.2754, huber: 1.8857, swd: 1.1568, ept: 248.7156
    Epoch [27/50], Test Losses: mse: 16.9308, mae: 2.0460, huber: 1.6619, swd: 1.0505, ept: 259.8319
      Epoch 27 composite train-obj: 1.199798
    Epoch [27/50], Test Losses: mse: 14.6418, mae: 1.9101, huber: 1.5305, swd: 1.3176, ept: 258.8556
    Best round's Test MSE: 14.6415, MAE: 1.9101, SWD: 1.3176
    Best round's Validation MSE: 18.0572, MAE: 2.1040, SWD: 1.4308
    Best round's Test verification MSE : 14.6418, MAE: 1.9101, SWD: 1.3176
    Time taken: 210.61 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 60.2090, mae: 5.6070, huber: 5.1353, swd: 27.8739, ept: 65.5126
    Epoch [1/50], Val Losses: mse: 48.8835, mae: 4.8696, huber: 4.4037, swd: 11.3410, ept: 99.9058
    Epoch [1/50], Test Losses: mse: 46.2907, mae: 4.7155, huber: 4.2495, swd: 11.8130, ept: 98.9493
      Epoch 1 composite train-obj: 5.135275
            Val objective improved inf → 4.4037, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 38.0045, mae: 3.9937, huber: 3.5429, swd: 7.4630, ept: 146.1582
    Epoch [2/50], Val Losses: mse: 38.7245, mae: 3.9739, huber: 3.5275, swd: 6.1550, ept: 153.4018
    Epoch [2/50], Test Losses: mse: 35.6932, mae: 3.7807, huber: 3.3356, swd: 5.9500, ept: 154.1758
      Epoch 2 composite train-obj: 3.542949
            Val objective improved 4.4037 → 3.5275, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 32.6153, mae: 3.4983, huber: 3.0594, swd: 4.7657, ept: 174.1611
    Epoch [3/50], Val Losses: mse: 36.0728, mae: 3.7105, huber: 3.2719, swd: 4.8100, ept: 170.0829
    Epoch [3/50], Test Losses: mse: 32.3881, mae: 3.4965, huber: 3.0589, swd: 4.6592, ept: 171.4570
      Epoch 3 composite train-obj: 3.059407
            Val objective improved 3.5275 → 3.2719, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 29.4897, mae: 3.2019, huber: 2.7722, swd: 3.7083, ept: 188.7166
    Epoch [4/50], Val Losses: mse: 38.4743, mae: 3.7819, huber: 3.3435, swd: 4.3833, ept: 168.5352
    Epoch [4/50], Test Losses: mse: 33.2686, mae: 3.5002, huber: 3.0636, swd: 4.3566, ept: 171.0785
      Epoch 4 composite train-obj: 2.772212
            No improvement (3.3435), counter 1/5
    Epoch [5/50], Train Losses: mse: 27.5848, mae: 3.0279, huber: 2.6038, swd: 3.2061, ept: 198.2469
    Epoch [5/50], Val Losses: mse: 36.4453, mae: 3.6936, huber: 3.2531, swd: 4.7282, ept: 172.0186
    Epoch [5/50], Test Losses: mse: 34.2731, mae: 3.5404, huber: 3.1013, swd: 4.6089, ept: 177.5549
      Epoch 5 composite train-obj: 2.603780
            Val objective improved 3.2719 → 3.2531, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 25.8419, mae: 2.8791, huber: 2.4596, swd: 2.9195, ept: 206.9721
    Epoch [6/50], Val Losses: mse: 32.5893, mae: 3.3728, huber: 2.9433, swd: 3.3999, ept: 187.2348
    Epoch [6/50], Test Losses: mse: 27.6741, mae: 3.0747, huber: 2.6482, swd: 3.1550, ept: 193.3957
      Epoch 6 composite train-obj: 2.459628
            Val objective improved 3.2531 → 2.9433, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 23.9866, mae: 2.7322, huber: 2.3169, swd: 2.5300, ept: 214.1451
    Epoch [7/50], Val Losses: mse: 31.3162, mae: 3.2046, huber: 2.7826, swd: 2.8845, ept: 200.2113
    Epoch [7/50], Test Losses: mse: 26.5093, mae: 2.9406, huber: 2.5201, swd: 2.6812, ept: 205.4590
      Epoch 7 composite train-obj: 2.316880
            Val objective improved 2.9433 → 2.7826, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 22.2759, mae: 2.5917, huber: 2.1806, swd: 2.2549, ept: 221.1266
    Epoch [8/50], Val Losses: mse: 30.0906, mae: 3.1639, huber: 2.7406, swd: 2.6970, ept: 194.5330
    Epoch [8/50], Test Losses: mse: 24.7152, mae: 2.8656, huber: 2.4451, swd: 2.4525, ept: 199.7707
      Epoch 8 composite train-obj: 2.180591
            Val objective improved 2.7826 → 2.7406, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 20.9861, mae: 2.4920, huber: 2.0836, swd: 2.0373, ept: 225.5422
    Epoch [9/50], Val Losses: mse: 25.7340, mae: 2.8224, huber: 2.4089, swd: 2.5676, ept: 208.9866
    Epoch [9/50], Test Losses: mse: 21.0938, mae: 2.5666, huber: 2.1548, swd: 2.3025, ept: 217.4835
      Epoch 9 composite train-obj: 2.083591
            Val objective improved 2.7406 → 2.4089, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 19.7736, mae: 2.3877, huber: 1.9831, swd: 1.8473, ept: 230.8895
    Epoch [10/50], Val Losses: mse: 28.4909, mae: 2.9690, huber: 2.5523, swd: 2.5118, ept: 208.6501
    Epoch [10/50], Test Losses: mse: 23.1215, mae: 2.6836, huber: 2.2692, swd: 2.3880, ept: 216.5461
      Epoch 10 composite train-obj: 1.983114
            No improvement (2.5523), counter 1/5
    Epoch [11/50], Train Losses: mse: 18.7635, mae: 2.2946, huber: 1.8941, swd: 1.7012, ept: 235.9622
    Epoch [11/50], Val Losses: mse: 27.4198, mae: 2.8324, huber: 2.4250, swd: 2.3838, ept: 218.4604
    Epoch [11/50], Test Losses: mse: 22.1393, mae: 2.5423, huber: 2.1383, swd: 2.2065, ept: 227.0148
      Epoch 11 composite train-obj: 1.894134
            No improvement (2.4250), counter 2/5
    Epoch [12/50], Train Losses: mse: 17.5417, mae: 2.1882, huber: 1.7924, swd: 1.5400, ept: 240.7805
    Epoch [12/50], Val Losses: mse: 25.8785, mae: 2.7275, huber: 2.3188, swd: 1.8367, ept: 220.7545
    Epoch [12/50], Test Losses: mse: 20.5917, mae: 2.4434, huber: 2.0375, swd: 1.6853, ept: 230.4001
      Epoch 12 composite train-obj: 1.792352
            Val objective improved 2.4089 → 2.3188, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 17.1569, mae: 2.1597, huber: 1.7641, swd: 1.4891, ept: 242.6656
    Epoch [13/50], Val Losses: mse: 25.3913, mae: 2.6974, huber: 2.2891, swd: 1.9045, ept: 224.5254
    Epoch [13/50], Test Losses: mse: 21.7813, mae: 2.4903, huber: 2.0850, swd: 1.7457, ept: 232.9560
      Epoch 13 composite train-obj: 1.764124
            Val objective improved 2.3188 → 2.2891, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 16.5088, mae: 2.0941, huber: 1.7021, swd: 1.4283, ept: 246.5060
    Epoch [14/50], Val Losses: mse: 24.0676, mae: 2.6423, huber: 2.2320, swd: 1.7622, ept: 223.6794
    Epoch [14/50], Test Losses: mse: 19.7030, mae: 2.3916, huber: 1.9857, swd: 1.5758, ept: 232.2639
      Epoch 14 composite train-obj: 1.702092
            Val objective improved 2.2891 → 2.2320, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 15.5096, mae: 2.0075, huber: 1.6195, swd: 1.2985, ept: 251.0559
    Epoch [15/50], Val Losses: mse: 23.7919, mae: 2.5981, huber: 2.1903, swd: 2.0326, ept: 226.8132
    Epoch [15/50], Test Losses: mse: 19.6572, mae: 2.3612, huber: 1.9566, swd: 1.8083, ept: 237.9916
      Epoch 15 composite train-obj: 1.619489
            Val objective improved 2.2320 → 2.1903, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 15.3022, mae: 1.9924, huber: 1.6049, swd: 1.2670, ept: 251.9344
    Epoch [16/50], Val Losses: mse: 23.1393, mae: 2.5371, huber: 2.1337, swd: 1.6581, ept: 228.6600
    Epoch [16/50], Test Losses: mse: 19.7438, mae: 2.3305, huber: 1.9304, swd: 1.4201, ept: 237.2614
      Epoch 16 composite train-obj: 1.604939
            Val objective improved 2.1903 → 2.1337, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 14.3283, mae: 1.9013, huber: 1.5189, swd: 1.1721, ept: 256.4003
    Epoch [17/50], Val Losses: mse: 23.0223, mae: 2.5039, huber: 2.1027, swd: 1.7876, ept: 229.4982
    Epoch [17/50], Test Losses: mse: 17.6951, mae: 2.2043, huber: 1.8083, swd: 1.6061, ept: 239.3221
      Epoch 17 composite train-obj: 1.518881
            Val objective improved 2.1337 → 2.1027, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 13.8928, mae: 1.8677, huber: 1.4862, swd: 1.1238, ept: 258.1568
    Epoch [18/50], Val Losses: mse: 19.9397, mae: 2.3114, huber: 1.9180, swd: 1.6434, ept: 238.0413
    Epoch [18/50], Test Losses: mse: 16.4380, mae: 2.0967, huber: 1.7077, swd: 1.4243, ept: 246.4262
      Epoch 18 composite train-obj: 1.486204
            Val objective improved 2.1027 → 1.9180, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 13.7579, mae: 1.8552, huber: 1.4743, swd: 1.1114, ept: 259.6634
    Epoch [19/50], Val Losses: mse: 23.2207, mae: 2.4458, huber: 2.0526, swd: 1.5348, ept: 239.5464
    Epoch [19/50], Test Losses: mse: 17.1800, mae: 2.0964, huber: 1.7097, swd: 1.3646, ept: 253.4233
      Epoch 19 composite train-obj: 1.474318
            No improvement (2.0526), counter 1/5
    Epoch [20/50], Train Losses: mse: 12.8701, mae: 1.7723, huber: 1.3965, swd: 1.0316, ept: 263.9672
    Epoch [20/50], Val Losses: mse: 19.2921, mae: 2.2176, huber: 1.8282, swd: 1.4637, ept: 248.4618
    Epoch [20/50], Test Losses: mse: 15.4727, mae: 1.9838, huber: 1.6007, swd: 1.3217, ept: 257.9898
      Epoch 20 composite train-obj: 1.396507
            Val objective improved 1.9180 → 1.8282, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 12.8351, mae: 1.7751, huber: 1.3980, swd: 1.0311, ept: 264.5763
    Epoch [21/50], Val Losses: mse: 20.4181, mae: 2.3532, huber: 1.9562, swd: 1.5177, ept: 233.0367
    Epoch [21/50], Test Losses: mse: 16.1563, mae: 2.1058, huber: 1.7133, swd: 1.3066, ept: 245.0057
      Epoch 21 composite train-obj: 1.397965
            No improvement (1.9562), counter 1/5
    Epoch [22/50], Train Losses: mse: 12.2487, mae: 1.7142, huber: 1.3416, swd: 0.9746, ept: 267.3076
    Epoch [22/50], Val Losses: mse: 18.5947, mae: 2.2020, huber: 1.8132, swd: 1.5831, ept: 243.5481
    Epoch [22/50], Test Losses: mse: 14.9696, mae: 1.9750, huber: 1.5906, swd: 1.3751, ept: 253.3902
      Epoch 22 composite train-obj: 1.341642
            Val objective improved 1.8282 → 1.8132, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 12.1193, mae: 1.7062, huber: 1.3337, swd: 0.9529, ept: 268.0528
    Epoch [23/50], Val Losses: mse: 21.4098, mae: 2.3358, huber: 1.9459, swd: 1.6089, ept: 241.9187
    Epoch [23/50], Test Losses: mse: 17.3397, mae: 2.0994, huber: 1.7136, swd: 1.4330, ept: 252.8115
      Epoch 23 composite train-obj: 1.333715
            No improvement (1.9459), counter 1/5
    Epoch [24/50], Train Losses: mse: 11.9896, mae: 1.7009, huber: 1.3279, swd: 0.9355, ept: 268.1837
    Epoch [24/50], Val Losses: mse: 20.1705, mae: 2.2923, huber: 1.8982, swd: 1.3767, ept: 244.2841
    Epoch [24/50], Test Losses: mse: 15.9646, mae: 2.0407, huber: 1.6531, swd: 1.2848, ept: 253.2379
      Epoch 24 composite train-obj: 1.327907
            No improvement (1.8982), counter 2/5
    Epoch [25/50], Train Losses: mse: 11.6803, mae: 1.6622, huber: 1.2923, swd: 0.9193, ept: 270.4793
    Epoch [25/50], Val Losses: mse: 17.7817, mae: 2.1091, huber: 1.7252, swd: 1.3400, ept: 247.9182
    Epoch [25/50], Test Losses: mse: 14.0639, mae: 1.8850, huber: 1.5060, swd: 1.1853, ept: 258.3566
      Epoch 25 composite train-obj: 1.292294
            Val objective improved 1.8132 → 1.7252, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 11.4540, mae: 1.6417, huber: 1.2732, swd: 0.8909, ept: 271.7095
    Epoch [26/50], Val Losses: mse: 17.3060, mae: 2.0683, huber: 1.6869, swd: 1.2290, ept: 250.5973
    Epoch [26/50], Test Losses: mse: 13.0978, mae: 1.8161, huber: 1.4391, swd: 1.0131, ept: 261.1043
      Epoch 26 composite train-obj: 1.273204
            Val objective improved 1.7252 → 1.6869, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 10.6917, mae: 1.5672, huber: 1.2036, swd: 0.8327, ept: 275.1988
    Epoch [27/50], Val Losses: mse: 17.5387, mae: 2.0471, huber: 1.6667, swd: 1.2564, ept: 255.4590
    Epoch [27/50], Test Losses: mse: 13.3251, mae: 1.7928, huber: 1.4176, swd: 1.1121, ept: 269.0046
      Epoch 27 composite train-obj: 1.203616
            Val objective improved 1.6869 → 1.6667, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 10.7932, mae: 1.5747, huber: 1.2106, swd: 0.8361, ept: 275.3711
    Epoch [28/50], Val Losses: mse: 16.7906, mae: 1.9878, huber: 1.6117, swd: 1.2509, ept: 257.0760
    Epoch [28/50], Test Losses: mse: 14.3940, mae: 1.8318, huber: 1.4597, swd: 1.0730, ept: 267.8207
      Epoch 28 composite train-obj: 1.210590
            Val objective improved 1.6667 → 1.6117, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 10.2923, mae: 1.5281, huber: 1.1669, swd: 0.7959, ept: 277.7318
    Epoch [29/50], Val Losses: mse: 18.0756, mae: 2.0895, huber: 1.7067, swd: 1.3648, ept: 253.8880
    Epoch [29/50], Test Losses: mse: 13.6692, mae: 1.8414, huber: 1.4621, swd: 1.1705, ept: 266.5767
      Epoch 29 composite train-obj: 1.166859
            No improvement (1.7067), counter 1/5
    Epoch [30/50], Train Losses: mse: 10.3363, mae: 1.5382, huber: 1.1753, swd: 0.8000, ept: 277.6530
    Epoch [30/50], Val Losses: mse: 19.5575, mae: 2.1994, huber: 1.8121, swd: 1.1502, ept: 246.2229
    Epoch [30/50], Test Losses: mse: 15.9690, mae: 1.9930, huber: 1.6106, swd: 1.0040, ept: 257.7186
      Epoch 30 composite train-obj: 1.175283
            No improvement (1.8121), counter 2/5
    Epoch [31/50], Train Losses: mse: 9.9992, mae: 1.5076, huber: 1.1476, swd: 0.7630, ept: 279.5079
    Epoch [31/50], Val Losses: mse: 17.5435, mae: 2.0599, huber: 1.6758, swd: 1.0588, ept: 256.7237
    Epoch [31/50], Test Losses: mse: 13.2034, mae: 1.8050, huber: 1.4260, swd: 0.9178, ept: 267.4046
      Epoch 31 composite train-obj: 1.147567
            No improvement (1.6758), counter 3/5
    Epoch [32/50], Train Losses: mse: 9.7199, mae: 1.4716, huber: 1.1145, swd: 0.7470, ept: 281.1840
    Epoch [32/50], Val Losses: mse: 16.0196, mae: 1.9413, huber: 1.5671, swd: 1.1597, ept: 261.2395
    Epoch [32/50], Test Losses: mse: 12.1045, mae: 1.7019, huber: 1.3325, swd: 0.9600, ept: 272.6665
      Epoch 32 composite train-obj: 1.114542
            Val objective improved 1.6117 → 1.5671, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 9.7889, mae: 1.4893, huber: 1.1298, swd: 0.7429, ept: 280.3370
    Epoch [33/50], Val Losses: mse: 16.9808, mae: 2.0090, huber: 1.6302, swd: 1.2178, ept: 259.3751
    Epoch [33/50], Test Losses: mse: 12.9449, mae: 1.7601, huber: 1.3874, swd: 1.0364, ept: 269.9899
      Epoch 33 composite train-obj: 1.129817
            No improvement (1.6302), counter 1/5
    Epoch [34/50], Train Losses: mse: 9.5632, mae: 1.4593, huber: 1.1029, swd: 0.7204, ept: 281.8557
    Epoch [34/50], Val Losses: mse: 15.1638, mae: 1.8822, huber: 1.5055, swd: 1.1258, ept: 262.5065
    Epoch [34/50], Test Losses: mse: 12.5743, mae: 1.7137, huber: 1.3424, swd: 1.0124, ept: 274.7979
      Epoch 34 composite train-obj: 1.102916
            Val objective improved 1.5671 → 1.5055, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 9.9501, mae: 1.4939, huber: 1.1344, swd: 0.7688, ept: 281.1286
    Epoch [35/50], Val Losses: mse: 17.1233, mae: 2.0173, huber: 1.6387, swd: 1.3279, ept: 258.4074
    Epoch [35/50], Test Losses: mse: 12.8674, mae: 1.7648, huber: 1.3916, swd: 1.0948, ept: 269.1092
      Epoch 35 composite train-obj: 1.134369
            No improvement (1.6387), counter 1/5
    Epoch [36/50], Train Losses: mse: 8.9756, mae: 1.3964, huber: 1.0451, swd: 0.6778, ept: 284.6859
    Epoch [36/50], Val Losses: mse: 17.1974, mae: 2.0085, huber: 1.6301, swd: 1.1667, ept: 258.5158
    Epoch [36/50], Test Losses: mse: 12.9544, mae: 1.7413, huber: 1.3699, swd: 0.9826, ept: 271.5913
      Epoch 36 composite train-obj: 1.045101
            No improvement (1.6301), counter 2/5
    Epoch [37/50], Train Losses: mse: 9.0719, mae: 1.4104, huber: 1.0574, swd: 0.6836, ept: 284.4164
    Epoch [37/50], Val Losses: mse: 13.8734, mae: 1.7651, huber: 1.4017, swd: 1.2024, ept: 266.4088
    Epoch [37/50], Test Losses: mse: 10.3283, mae: 1.5310, huber: 1.1748, swd: 0.9738, ept: 279.2317
      Epoch 37 composite train-obj: 1.057411
            Val objective improved 1.5055 → 1.4017, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 8.5238, mae: 1.3501, huber: 1.0025, swd: 0.6434, ept: 286.9229
    Epoch [38/50], Val Losses: mse: 15.5421, mae: 1.9145, huber: 1.5411, swd: 1.1565, ept: 258.8217
    Epoch [38/50], Test Losses: mse: 11.5122, mae: 1.6496, huber: 1.2837, swd: 0.9192, ept: 273.1034
      Epoch 38 composite train-obj: 1.002481
            No improvement (1.5411), counter 1/5
    Epoch [39/50], Train Losses: mse: 9.2031, mae: 1.4239, huber: 1.0696, swd: 0.6876, ept: 284.3898
    Epoch [39/50], Val Losses: mse: 15.9047, mae: 1.9046, huber: 1.5325, swd: 1.1827, ept: 261.5685
    Epoch [39/50], Test Losses: mse: 11.1618, mae: 1.6269, huber: 1.2604, swd: 1.0093, ept: 273.9082
      Epoch 39 composite train-obj: 1.069593
            No improvement (1.5325), counter 2/5
    Epoch [40/50], Train Losses: mse: 8.7578, mae: 1.3687, huber: 1.0197, swd: 0.6619, ept: 286.6243
    Epoch [40/50], Val Losses: mse: 15.8150, mae: 1.9122, huber: 1.5373, swd: 1.1138, ept: 261.2009
    Epoch [40/50], Test Losses: mse: 13.5386, mae: 1.7645, huber: 1.3943, swd: 0.9911, ept: 271.5691
      Epoch 40 composite train-obj: 1.019661
            No improvement (1.5373), counter 3/5
    Epoch [41/50], Train Losses: mse: 8.8472, mae: 1.3819, huber: 1.0312, swd: 0.6682, ept: 286.5118
    Epoch [41/50], Val Losses: mse: 13.5030, mae: 1.7746, huber: 1.4036, swd: 1.1070, ept: 266.1520
    Epoch [41/50], Test Losses: mse: 10.7423, mae: 1.6033, huber: 1.2363, swd: 0.9734, ept: 275.4412
      Epoch 41 composite train-obj: 1.031194
            No improvement (1.4036), counter 4/5
    Epoch [42/50], Train Losses: mse: 8.4953, mae: 1.3539, huber: 1.0055, swd: 0.6489, ept: 287.1402
    Epoch [42/50], Val Losses: mse: 17.8485, mae: 2.0086, huber: 1.6330, swd: 1.1490, ept: 261.4276
    Epoch [42/50], Test Losses: mse: 13.9069, mae: 1.7791, huber: 1.4084, swd: 0.9972, ept: 272.6208
      Epoch 42 composite train-obj: 1.005498
    Epoch [42/50], Test Losses: mse: 10.3291, mae: 1.5310, huber: 1.1749, swd: 0.9738, ept: 279.2254
    Best round's Test MSE: 10.3283, MAE: 1.5310, SWD: 0.9738
    Best round's Validation MSE: 13.8734, MAE: 1.7651, SWD: 1.2024
    Best round's Test verification MSE : 10.3291, MAE: 1.5310, SWD: 0.9738
    Time taken: 311.29 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 61.0089, mae: 5.6392, huber: 5.1673, swd: 29.3974, ept: 66.5119
    Epoch [1/50], Val Losses: mse: 46.5562, mae: 4.6957, huber: 4.2328, swd: 11.2929, ept: 115.0932
    Epoch [1/50], Test Losses: mse: 43.9286, mae: 4.5259, huber: 4.0638, swd: 11.6327, ept: 115.7508
      Epoch 1 composite train-obj: 5.167290
            Val objective improved inf → 4.2328, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 37.0977, mae: 3.9183, huber: 3.4686, swd: 7.5614, ept: 152.7405
    Epoch [2/50], Val Losses: mse: 40.0370, mae: 4.0995, huber: 3.6482, swd: 7.3090, ept: 143.3157
    Epoch [2/50], Test Losses: mse: 36.8497, mae: 3.9044, huber: 3.4546, swd: 7.2129, ept: 145.9793
      Epoch 2 composite train-obj: 3.468556
            Val objective improved 4.2328 → 3.6482, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 31.8674, mae: 3.4325, huber: 2.9946, swd: 4.8455, ept: 177.7345
    Epoch [3/50], Val Losses: mse: 35.7427, mae: 3.7093, huber: 3.2686, swd: 5.0763, ept: 165.9362
    Epoch [3/50], Test Losses: mse: 32.3665, mae: 3.5082, huber: 3.0685, swd: 4.8918, ept: 170.3135
      Epoch 3 composite train-obj: 2.994637
            Val objective improved 3.6482 → 3.2686, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 28.6506, mae: 3.1370, huber: 2.7087, swd: 3.8706, ept: 194.3138
    Epoch [4/50], Val Losses: mse: 33.3681, mae: 3.4792, huber: 3.0453, swd: 4.2687, ept: 183.9864
    Epoch [4/50], Test Losses: mse: 29.4022, mae: 3.2385, huber: 2.8072, swd: 4.1662, ept: 188.6860
      Epoch 4 composite train-obj: 2.708665
            Val objective improved 3.2686 → 3.0453, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 26.4116, mae: 2.9471, huber: 2.5246, swd: 3.3215, ept: 205.1793
    Epoch [5/50], Val Losses: mse: 32.7280, mae: 3.3907, huber: 2.9598, swd: 3.5871, ept: 192.5085
    Epoch [5/50], Test Losses: mse: 27.3572, mae: 3.0892, huber: 2.6602, swd: 3.2679, ept: 199.4465
      Epoch 5 composite train-obj: 2.524621
            Val objective improved 3.0453 → 2.9598, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 24.4472, mae: 2.7837, huber: 2.3664, swd: 2.9088, ept: 213.4961
    Epoch [6/50], Val Losses: mse: 31.3493, mae: 3.2623, huber: 2.8350, swd: 3.0139, ept: 192.5569
    Epoch [6/50], Test Losses: mse: 26.0091, mae: 2.9562, huber: 2.5326, swd: 2.8100, ept: 200.7954
      Epoch 6 composite train-obj: 2.366446
            Val objective improved 2.9598 → 2.8350, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 22.9466, mae: 2.6582, huber: 2.2449, swd: 2.5768, ept: 219.0342
    Epoch [7/50], Val Losses: mse: 29.1939, mae: 3.1067, huber: 2.6847, swd: 3.1259, ept: 202.0010
    Epoch [7/50], Test Losses: mse: 25.4754, mae: 2.8611, huber: 2.4429, swd: 2.9167, ept: 209.0754
      Epoch 7 composite train-obj: 2.244949
            Val objective improved 2.8350 → 2.6847, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 21.5413, mae: 2.5335, huber: 2.1248, swd: 2.3033, ept: 224.8846
    Epoch [8/50], Val Losses: mse: 27.0241, mae: 2.8924, huber: 2.4797, swd: 2.8398, ept: 208.9002
    Epoch [8/50], Test Losses: mse: 23.1045, mae: 2.6744, huber: 2.2638, swd: 2.6326, ept: 215.2106
      Epoch 8 composite train-obj: 2.124812
            Val objective improved 2.6847 → 2.4797, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 20.4113, mae: 2.4394, huber: 2.0340, swd: 2.1029, ept: 229.0611
    Epoch [9/50], Val Losses: mse: 24.8183, mae: 2.7850, huber: 2.3705, swd: 2.9726, ept: 210.1539
    Epoch [9/50], Test Losses: mse: 21.2651, mae: 2.5713, huber: 2.1595, swd: 2.6476, ept: 216.9425
      Epoch 9 composite train-obj: 2.033971
            Val objective improved 2.4797 → 2.3705, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 19.4598, mae: 2.3532, huber: 1.9511, swd: 1.9402, ept: 232.8508
    Epoch [10/50], Val Losses: mse: 25.7707, mae: 2.8024, huber: 2.3878, swd: 2.3837, ept: 215.5409
    Epoch [10/50], Test Losses: mse: 21.7851, mae: 2.5747, huber: 2.1616, swd: 2.0978, ept: 224.1798
      Epoch 10 composite train-obj: 1.951060
            No improvement (2.3878), counter 1/5
    Epoch [11/50], Train Losses: mse: 18.7129, mae: 2.2810, huber: 1.8823, swd: 1.8387, ept: 236.2664
    Epoch [11/50], Val Losses: mse: 24.3960, mae: 2.6673, huber: 2.2602, swd: 2.1834, ept: 223.3837
    Epoch [11/50], Test Losses: mse: 20.7242, mae: 2.4636, huber: 2.0584, swd: 1.9182, ept: 230.1013
      Epoch 11 composite train-obj: 1.882274
            Val objective improved 2.3705 → 2.2602, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 17.7267, mae: 2.1959, huber: 1.8011, swd: 1.7276, ept: 240.2424
    Epoch [12/50], Val Losses: mse: 23.7522, mae: 2.6231, huber: 2.2167, swd: 2.2742, ept: 221.7454
    Epoch [12/50], Test Losses: mse: 20.1998, mae: 2.4288, huber: 2.0246, swd: 2.1329, ept: 227.2181
      Epoch 12 composite train-obj: 1.801096
            Val objective improved 2.2602 → 2.2167, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 17.6816, mae: 2.1961, huber: 1.8003, swd: 1.6710, ept: 240.6289
    Epoch [13/50], Val Losses: mse: 22.3486, mae: 2.5366, huber: 2.1338, swd: 2.2631, ept: 227.5685
    Epoch [13/50], Test Losses: mse: 18.5056, mae: 2.3180, huber: 1.9176, swd: 1.9964, ept: 234.8577
      Epoch 13 composite train-obj: 1.800295
            Val objective improved 2.2167 → 2.1338, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 16.9895, mae: 2.1291, huber: 1.7368, swd: 1.5762, ept: 243.8145
    Epoch [14/50], Val Losses: mse: 23.2913, mae: 2.6097, huber: 2.1995, swd: 2.3006, ept: 224.0561
    Epoch [14/50], Test Losses: mse: 19.3983, mae: 2.3881, huber: 1.9799, swd: 2.0496, ept: 232.2140
      Epoch 14 composite train-obj: 1.736809
            No improvement (2.1995), counter 1/5
    Epoch [15/50], Train Losses: mse: 15.7211, mae: 2.0162, huber: 1.6300, swd: 1.4260, ept: 248.2218
    Epoch [15/50], Val Losses: mse: 22.9783, mae: 2.5832, huber: 2.1772, swd: 1.7019, ept: 221.8595
    Epoch [15/50], Test Losses: mse: 18.6297, mae: 2.3293, huber: 1.9274, swd: 1.4273, ept: 229.4907
      Epoch 15 composite train-obj: 1.629994
            No improvement (2.1772), counter 2/5
    Epoch [16/50], Train Losses: mse: 16.1212, mae: 2.0516, huber: 1.6624, swd: 1.4696, ept: 247.8894
    Epoch [16/50], Val Losses: mse: 24.6837, mae: 2.6566, huber: 2.2497, swd: 2.2241, ept: 223.1050
    Epoch [16/50], Test Losses: mse: 20.3566, mae: 2.4137, huber: 2.0103, swd: 2.0691, ept: 230.8528
      Epoch 16 composite train-obj: 1.662370
            No improvement (2.2497), counter 3/5
    Epoch [17/50], Train Losses: mse: 15.5002, mae: 1.9960, huber: 1.6104, swd: 1.3965, ept: 249.8403
    Epoch [17/50], Val Losses: mse: 18.5890, mae: 2.2303, huber: 1.8387, swd: 1.7775, ept: 237.7125
    Epoch [17/50], Test Losses: mse: 15.5004, mae: 2.0456, huber: 1.6571, swd: 1.4956, ept: 245.6915
      Epoch 17 composite train-obj: 1.610446
            Val objective improved 2.1338 → 1.8387, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 13.9972, mae: 1.8659, huber: 1.4867, swd: 1.2205, ept: 254.9492
    Epoch [18/50], Val Losses: mse: 19.6973, mae: 2.3089, huber: 1.9126, swd: 1.7521, ept: 236.9610
    Epoch [18/50], Test Losses: mse: 16.0324, mae: 2.0938, huber: 1.7012, swd: 1.4959, ept: 244.5044
      Epoch 18 composite train-obj: 1.486706
            No improvement (1.9126), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.0275, mae: 1.8690, huber: 1.4892, swd: 1.2150, ept: 255.1673
    Epoch [19/50], Val Losses: mse: 21.4809, mae: 2.4165, huber: 2.0159, swd: 1.6933, ept: 235.3371
    Epoch [19/50], Test Losses: mse: 17.9080, mae: 2.2161, huber: 1.8185, swd: 1.5162, ept: 240.6563
      Epoch 19 composite train-obj: 1.489151
            No improvement (2.0159), counter 2/5
    Epoch [20/50], Train Losses: mse: 14.1088, mae: 1.8759, huber: 1.4954, swd: 1.2226, ept: 255.5585
    Epoch [20/50], Val Losses: mse: 19.7335, mae: 2.2787, huber: 1.8878, swd: 1.7348, ept: 237.3011
    Epoch [20/50], Test Losses: mse: 15.1922, mae: 2.0148, huber: 1.6284, swd: 1.4636, ept: 247.2719
      Epoch 20 composite train-obj: 1.495395
            No improvement (1.8878), counter 3/5
    Epoch [21/50], Train Losses: mse: 13.0843, mae: 1.7926, huber: 1.4161, swd: 1.1128, ept: 258.5430
    Epoch [21/50], Val Losses: mse: 18.8100, mae: 2.2219, huber: 1.8307, swd: 1.7249, ept: 238.1086
    Epoch [21/50], Test Losses: mse: 15.5291, mae: 2.0335, huber: 1.6452, swd: 1.5654, ept: 246.8033
      Epoch 21 composite train-obj: 1.416144
            Val objective improved 1.8387 → 1.8307, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.1165, mae: 1.7879, huber: 1.4123, swd: 1.1154, ept: 259.5619
    Epoch [22/50], Val Losses: mse: 18.0035, mae: 2.1171, huber: 1.7316, swd: 1.3992, ept: 247.3125
    Epoch [22/50], Test Losses: mse: 14.5131, mae: 1.9236, huber: 1.5418, swd: 1.2501, ept: 255.5516
      Epoch 22 composite train-obj: 1.412294
            Val objective improved 1.8307 → 1.7316, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 12.3892, mae: 1.7193, huber: 1.3479, swd: 1.0448, ept: 262.9577
    Epoch [23/50], Val Losses: mse: 20.3521, mae: 2.2795, huber: 1.8897, swd: 1.5013, ept: 240.7485
    Epoch [23/50], Test Losses: mse: 15.4978, mae: 2.0081, huber: 1.6238, swd: 1.3662, ept: 248.2613
      Epoch 23 composite train-obj: 1.347923
            No improvement (1.8897), counter 1/5
    Epoch [24/50], Train Losses: mse: 12.4773, mae: 1.7231, huber: 1.3516, swd: 1.0456, ept: 263.1708
    Epoch [24/50], Val Losses: mse: 20.5691, mae: 2.3148, huber: 1.9230, swd: 1.7040, ept: 237.2340
    Epoch [24/50], Test Losses: mse: 16.5955, mae: 2.0966, huber: 1.7071, swd: 1.5327, ept: 245.8107
      Epoch 24 composite train-obj: 1.351637
            No improvement (1.9230), counter 2/5
    Epoch [25/50], Train Losses: mse: 12.3153, mae: 1.7198, huber: 1.3473, swd: 1.0286, ept: 263.8605
    Epoch [25/50], Val Losses: mse: 18.2165, mae: 2.1052, huber: 1.7236, swd: 1.3611, ept: 249.1603
    Epoch [25/50], Test Losses: mse: 13.0504, mae: 1.8119, huber: 1.4353, swd: 1.0843, ept: 259.9062
      Epoch 25 composite train-obj: 1.347305
            Val objective improved 1.7316 → 1.7236, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 11.3613, mae: 1.6214, huber: 1.2565, swd: 0.9332, ept: 267.9415
    Epoch [26/50], Val Losses: mse: 15.0836, mae: 1.9372, huber: 1.5561, swd: 1.3331, ept: 253.6858
    Epoch [26/50], Test Losses: mse: 14.1916, mae: 1.8592, huber: 1.4816, swd: 1.2120, ept: 264.4175
      Epoch 26 composite train-obj: 1.256492
            Val objective improved 1.7236 → 1.5561, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 11.7176, mae: 1.6571, huber: 1.2888, swd: 0.9699, ept: 267.0415
    Epoch [27/50], Val Losses: mse: 16.6161, mae: 2.0426, huber: 1.6587, swd: 1.4303, ept: 249.2721
    Epoch [27/50], Test Losses: mse: 14.1589, mae: 1.8627, huber: 1.4856, swd: 1.1901, ept: 260.5765
      Epoch 27 composite train-obj: 1.288846
            No improvement (1.6587), counter 1/5
    Epoch [28/50], Train Losses: mse: 11.5588, mae: 1.6468, huber: 1.2792, swd: 0.9580, ept: 268.4766
    Epoch [28/50], Val Losses: mse: 17.5931, mae: 2.1140, huber: 1.7286, swd: 1.6439, ept: 246.6542
    Epoch [28/50], Test Losses: mse: 14.2212, mae: 1.9134, huber: 1.5313, swd: 1.5102, ept: 256.0974
      Epoch 28 composite train-obj: 1.279167
            No improvement (1.7286), counter 2/5
    Epoch [29/50], Train Losses: mse: 10.6726, mae: 1.5565, huber: 1.1954, swd: 0.8829, ept: 272.9869
    Epoch [29/50], Val Losses: mse: 15.0622, mae: 1.9256, huber: 1.5470, swd: 1.4598, ept: 256.4012
    Epoch [29/50], Test Losses: mse: 12.0207, mae: 1.7222, huber: 1.3498, swd: 1.2063, ept: 266.5337
      Epoch 29 composite train-obj: 1.195388
            Val objective improved 1.5561 → 1.5470, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 10.3226, mae: 1.5222, huber: 1.1634, swd: 0.8501, ept: 274.5627
    Epoch [30/50], Val Losses: mse: 15.6283, mae: 1.8990, huber: 1.5280, swd: 1.2894, ept: 261.1777
    Epoch [30/50], Test Losses: mse: 12.2218, mae: 1.6897, huber: 1.3225, swd: 1.0081, ept: 271.4512
      Epoch 30 composite train-obj: 1.163426
            Val objective improved 1.5470 → 1.5280, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 10.6527, mae: 1.5597, huber: 1.1974, swd: 0.8724, ept: 273.4693
    Epoch [31/50], Val Losses: mse: 17.6323, mae: 2.0635, huber: 1.6824, swd: 1.3269, ept: 250.2947
    Epoch [31/50], Test Losses: mse: 13.0818, mae: 1.8016, huber: 1.4248, swd: 1.0341, ept: 262.9261
      Epoch 31 composite train-obj: 1.197396
            No improvement (1.6824), counter 1/5
    Epoch [32/50], Train Losses: mse: 10.7180, mae: 1.5653, huber: 1.2025, swd: 0.8957, ept: 273.7039
    Epoch [32/50], Val Losses: mse: 17.4724, mae: 2.0401, huber: 1.6611, swd: 1.2207, ept: 253.8017
    Epoch [32/50], Test Losses: mse: 12.7814, mae: 1.7722, huber: 1.3996, swd: 0.9651, ept: 262.7337
      Epoch 32 composite train-obj: 1.202467
            No improvement (1.6611), counter 2/5
    Epoch [33/50], Train Losses: mse: 10.4593, mae: 1.5380, huber: 1.1776, swd: 0.8564, ept: 274.8736
    Epoch [33/50], Val Losses: mse: 16.9588, mae: 2.0092, huber: 1.6296, swd: 1.3439, ept: 255.3133
    Epoch [33/50], Test Losses: mse: 13.1316, mae: 1.7799, huber: 1.4063, swd: 1.2113, ept: 266.1448
      Epoch 33 composite train-obj: 1.177585
            No improvement (1.6296), counter 3/5
    Epoch [34/50], Train Losses: mse: 10.1051, mae: 1.5020, huber: 1.1440, swd: 0.8228, ept: 276.3952
    Epoch [34/50], Val Losses: mse: 16.4293, mae: 1.9580, huber: 1.5833, swd: 1.1896, ept: 258.6142
    Epoch [34/50], Test Losses: mse: 11.3758, mae: 1.6534, huber: 1.2861, swd: 0.9260, ept: 269.7738
      Epoch 34 composite train-obj: 1.144010
            No improvement (1.5833), counter 4/5
    Epoch [35/50], Train Losses: mse: 10.0405, mae: 1.4975, huber: 1.1399, swd: 0.8065, ept: 276.9750
    Epoch [35/50], Val Losses: mse: 16.2264, mae: 1.9947, huber: 1.6126, swd: 1.4399, ept: 257.1152
    Epoch [35/50], Test Losses: mse: 14.1472, mae: 1.8645, huber: 1.4851, swd: 1.1804, ept: 266.1921
      Epoch 35 composite train-obj: 1.139914
    Epoch [35/50], Test Losses: mse: 12.2227, mae: 1.6898, huber: 1.3226, swd: 1.0082, ept: 271.3884
    Best round's Test MSE: 12.2218, MAE: 1.6897, SWD: 1.0081
    Best round's Validation MSE: 15.6283, MAE: 1.8990, SWD: 1.2894
    Best round's Test verification MSE : 12.2227, MAE: 1.6898, SWD: 1.0082
    Time taken: 263.82 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250510_1710)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.3972 ± 1.7652
      mae: 1.7102 ± 0.1554
      huber: 1.3426 ± 0.1459
      swd: 1.0998 ± 0.1546
      ept: 269.8528 ± 8.3869
      count: 36.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 15.8530 ± 1.7154
      mae: 1.9227 ± 0.1394
      huber: 1.5505 ± 0.1317
      swd: 1.3075 ± 0.0941
      ept: 259.0972 ± 6.9761
      count: 36.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 786.91 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250510_1710
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 336-720
##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
    pred_len=720,
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
    # single_magnitude_for_shift=True,
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

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 277
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 277
    Validation Batches: 33
    Test Batches: 74
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 68.8849, mae: 6.2787, huber: 5.8003, swd: 35.8427, ept: 50.3252
    Epoch [1/50], Val Losses: mse: 60.3264, mae: 5.8117, huber: 5.3361, swd: 22.6376, ept: 81.9153
    Epoch [1/50], Test Losses: mse: 59.3872, mae: 5.7159, huber: 5.2412, swd: 23.4818, ept: 78.5328
      Epoch 1 composite train-obj: 5.800293
            Val objective improved inf → 5.3361, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 55.1192, mae: 5.3225, huber: 4.8538, swd: 16.7452, ept: 115.2398
    Epoch [2/50], Val Losses: mse: 58.3986, mae: 5.4876, huber: 5.0179, swd: 14.6031, ept: 120.6809
    Epoch [2/50], Test Losses: mse: 56.9469, mae: 5.3699, huber: 4.9015, swd: 14.6229, ept: 116.5364
      Epoch 2 composite train-obj: 4.853793
            Val objective improved 5.3361 → 5.0179, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.2640, mae: 4.9031, huber: 4.4405, swd: 11.2943, ept: 154.5708
    Epoch [3/50], Val Losses: mse: 58.3388, mae: 5.3418, huber: 4.8758, swd: 10.9773, ept: 150.5248
    Epoch [3/50], Test Losses: mse: 56.8717, mae: 5.2262, huber: 4.7615, swd: 11.0089, ept: 146.2830
      Epoch 3 composite train-obj: 4.440474
            Val objective improved 5.0179 → 4.8758, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 46.5425, mae: 4.6057, huber: 4.1472, swd: 8.4735, ept: 178.0198
    Epoch [4/50], Val Losses: mse: 59.9673, mae: 5.3348, huber: 4.8695, swd: 8.2052, ept: 159.6353
    Epoch [4/50], Test Losses: mse: 57.9835, mae: 5.2197, huber: 4.7558, swd: 8.3353, ept: 159.5502
      Epoch 4 composite train-obj: 4.147209
            Val objective improved 4.8758 → 4.8695, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 43.3025, mae: 4.3439, huber: 3.8900, swd: 6.7513, ept: 198.5147
    Epoch [5/50], Val Losses: mse: 62.0760, mae: 5.3351, huber: 4.8727, swd: 6.6353, ept: 177.3116
    Epoch [5/50], Test Losses: mse: 58.0281, mae: 5.1188, huber: 4.6583, swd: 6.7492, ept: 176.3851
      Epoch 5 composite train-obj: 3.889979
            No improvement (4.8727), counter 1/5
    Epoch [6/50], Train Losses: mse: 40.8658, mae: 4.1516, huber: 3.7011, swd: 5.7127, ept: 212.6640
    Epoch [6/50], Val Losses: mse: 61.9893, mae: 5.2584, huber: 4.7984, swd: 6.2094, ept: 190.4039
    Epoch [6/50], Test Losses: mse: 58.9966, mae: 5.0831, huber: 4.6250, swd: 6.0752, ept: 192.5344
      Epoch 6 composite train-obj: 3.701140
            Val objective improved 4.8695 → 4.7984, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 39.0335, mae: 4.0061, huber: 3.5586, swd: 5.0782, ept: 225.6272
    Epoch [7/50], Val Losses: mse: 62.9055, mae: 5.2510, huber: 4.7934, swd: 5.1875, ept: 195.0506
    Epoch [7/50], Test Losses: mse: 59.0806, mae: 5.0478, huber: 4.5922, swd: 5.1920, ept: 201.1532
      Epoch 7 composite train-obj: 3.558605
            Val objective improved 4.7984 → 4.7934, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 37.5395, mae: 3.8878, huber: 3.4428, swd: 4.6067, ept: 234.3727
    Epoch [8/50], Val Losses: mse: 63.5887, mae: 5.2814, huber: 4.8226, swd: 4.5973, ept: 195.9632
    Epoch [8/50], Test Losses: mse: 58.9002, mae: 5.0343, huber: 4.5777, swd: 4.5223, ept: 202.9197
      Epoch 8 composite train-obj: 3.442792
            No improvement (4.8226), counter 1/5
    Epoch [9/50], Train Losses: mse: 35.9412, mae: 3.7609, huber: 3.3188, swd: 4.1782, ept: 244.1369
    Epoch [9/50], Val Losses: mse: 64.0532, mae: 5.2299, huber: 4.7752, swd: 4.3358, ept: 207.0033
    Epoch [9/50], Test Losses: mse: 58.7413, mae: 4.9640, huber: 4.5118, swd: 4.5784, ept: 213.2316
      Epoch 9 composite train-obj: 3.318847
            Val objective improved 4.7934 → 4.7752, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 35.1415, mae: 3.7010, huber: 3.2599, swd: 3.9760, ept: 248.7161
    Epoch [10/50], Val Losses: mse: 64.2693, mae: 5.2716, huber: 4.8138, swd: 4.2091, ept: 198.9953
    Epoch [10/50], Test Losses: mse: 60.3611, mae: 5.0685, huber: 4.6129, swd: 4.3962, ept: 204.1627
      Epoch 10 composite train-obj: 3.259920
            No improvement (4.8138), counter 1/5
    Epoch [11/50], Train Losses: mse: 33.8139, mae: 3.5958, huber: 3.1573, swd: 3.6788, ept: 256.7479
    Epoch [11/50], Val Losses: mse: 62.6340, mae: 5.1288, huber: 4.6758, swd: 4.3493, ept: 216.9331
    Epoch [11/50], Test Losses: mse: 58.8979, mae: 4.9310, huber: 4.4805, swd: 4.5531, ept: 224.8736
      Epoch 11 composite train-obj: 3.157268
            Val objective improved 4.7752 → 4.6758, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 32.6277, mae: 3.5064, huber: 3.0699, swd: 3.4551, ept: 262.9464
    Epoch [12/50], Val Losses: mse: 63.7609, mae: 5.2253, huber: 4.7702, swd: 4.0059, ept: 206.3931
    Epoch [12/50], Test Losses: mse: 60.9353, mae: 5.0679, huber: 4.6149, swd: 4.1505, ept: 211.6497
      Epoch 12 composite train-obj: 3.069862
            No improvement (4.7702), counter 1/5
    Epoch [13/50], Train Losses: mse: 32.1896, mae: 3.4737, huber: 3.0377, swd: 3.3507, ept: 266.5566
    Epoch [13/50], Val Losses: mse: 63.1478, mae: 5.2030, huber: 4.7454, swd: 4.9803, ept: 205.2320
    Epoch [13/50], Test Losses: mse: 60.1058, mae: 5.0372, huber: 4.5814, swd: 4.9595, ept: 210.4431
      Epoch 13 composite train-obj: 3.037675
            No improvement (4.7454), counter 2/5
    Epoch [14/50], Train Losses: mse: 31.2253, mae: 3.4001, huber: 2.9657, swd: 3.1844, ept: 273.1268
    Epoch [14/50], Val Losses: mse: 64.7608, mae: 5.2228, huber: 4.7674, swd: 3.5011, ept: 215.2968
    Epoch [14/50], Test Losses: mse: 60.3293, mae: 5.0016, huber: 4.5485, swd: 3.8947, ept: 221.8447
      Epoch 14 composite train-obj: 2.965732
            No improvement (4.7674), counter 3/5
    Epoch [15/50], Train Losses: mse: 30.2674, mae: 3.3179, huber: 2.8861, swd: 2.9962, ept: 281.6241
    Epoch [15/50], Val Losses: mse: 64.3517, mae: 5.1807, huber: 4.7286, swd: 3.8099, ept: 221.9281
    Epoch [15/50], Test Losses: mse: 59.1644, mae: 4.9091, huber: 4.4604, swd: 4.0073, ept: 228.9679
      Epoch 15 composite train-obj: 2.886062
            No improvement (4.7286), counter 4/5
    Epoch [16/50], Train Losses: mse: 29.8108, mae: 3.2837, huber: 2.8526, swd: 2.9151, ept: 285.0154
    Epoch [16/50], Val Losses: mse: 65.8895, mae: 5.2555, huber: 4.8005, swd: 3.2673, ept: 219.9165
    Epoch [16/50], Test Losses: mse: 60.7069, mae: 4.9821, huber: 4.5298, swd: 3.2826, ept: 231.0855
      Epoch 16 composite train-obj: 2.852615
    Epoch [16/50], Test Losses: mse: 58.8960, mae: 4.9310, huber: 4.4804, swd: 4.5537, ept: 224.8292
    Best round's Test MSE: 58.8979, MAE: 4.9310, SWD: 4.5531
    Best round's Validation MSE: 62.6340, MAE: 5.1288, SWD: 4.3493
    Best round's Test verification MSE : 58.8960, MAE: 4.9310, SWD: 4.5537
    Time taken: 128.43 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 68.7530, mae: 6.2589, huber: 5.7807, swd: 35.2006, ept: 49.4111
    Epoch [1/50], Val Losses: mse: 62.2274, mae: 5.9118, huber: 5.4357, swd: 19.1282, ept: 83.7951
    Epoch [1/50], Test Losses: mse: 60.8241, mae: 5.7869, huber: 5.3118, swd: 19.9397, ept: 78.6028
      Epoch 1 composite train-obj: 5.780710
            Val objective improved inf → 5.4357, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 54.6132, mae: 5.2834, huber: 4.8152, swd: 15.7835, ept: 117.1145
    Epoch [2/50], Val Losses: mse: 57.9291, mae: 5.4246, huber: 4.9558, swd: 13.6651, ept: 130.5375
    Epoch [2/50], Test Losses: mse: 55.9755, mae: 5.2782, huber: 4.8110, swd: 13.5362, ept: 126.3558
      Epoch 2 composite train-obj: 4.815243
            Val objective improved 5.4357 → 4.9558, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 49.7906, mae: 4.8658, huber: 4.4038, swd: 10.7319, ept: 165.4396
    Epoch [3/50], Val Losses: mse: 58.1635, mae: 5.3483, huber: 4.8817, swd: 10.2377, ept: 150.6282
    Epoch [3/50], Test Losses: mse: 56.6561, mae: 5.2326, huber: 4.7675, swd: 10.0900, ept: 153.8250
      Epoch 3 composite train-obj: 4.403773
            Val objective improved 4.9558 → 4.8817, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 46.2575, mae: 4.5760, huber: 4.1185, swd: 8.1855, ept: 189.3555
    Epoch [4/50], Val Losses: mse: 61.5719, mae: 5.4314, huber: 4.9661, swd: 8.0808, ept: 158.9487
    Epoch [4/50], Test Losses: mse: 58.4639, mae: 5.2562, huber: 4.7925, swd: 8.1151, ept: 159.6989
      Epoch 4 composite train-obj: 4.118476
            No improvement (4.9661), counter 1/5
    Epoch [5/50], Train Losses: mse: 43.4669, mae: 4.3491, huber: 3.8956, swd: 6.7031, ept: 204.0125
    Epoch [5/50], Val Losses: mse: 61.4143, mae: 5.3020, huber: 4.8403, swd: 6.5229, ept: 180.3531
    Epoch [5/50], Test Losses: mse: 58.1549, mae: 5.1235, huber: 4.6631, swd: 6.5765, ept: 183.7619
      Epoch 5 composite train-obj: 3.895605
            Val objective improved 4.8817 → 4.8403, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 41.5781, mae: 4.2011, huber: 3.7502, swd: 5.8707, ept: 213.0614
    Epoch [6/50], Val Losses: mse: 63.0992, mae: 5.3556, huber: 4.8941, swd: 6.6241, ept: 176.9684
    Epoch [6/50], Test Losses: mse: 59.8298, mae: 5.1892, huber: 4.7291, swd: 6.8142, ept: 179.6495
      Epoch 6 composite train-obj: 3.750194
            No improvement (4.8941), counter 1/5
    Epoch [7/50], Train Losses: mse: 39.4161, mae: 4.0369, huber: 3.5893, swd: 5.1103, ept: 223.4469
    Epoch [7/50], Val Losses: mse: 62.9573, mae: 5.3002, huber: 4.8408, swd: 5.3344, ept: 186.3522
    Epoch [7/50], Test Losses: mse: 59.2091, mae: 5.1002, huber: 4.6425, swd: 5.3929, ept: 190.2878
      Epoch 7 composite train-obj: 3.589252
            No improvement (4.8408), counter 2/5
    Epoch [8/50], Train Losses: mse: 37.9712, mae: 3.9241, huber: 3.4784, swd: 4.6457, ept: 231.7614
    Epoch [8/50], Val Losses: mse: 63.4324, mae: 5.2264, huber: 4.7701, swd: 4.6102, ept: 197.7990
    Epoch [8/50], Test Losses: mse: 60.5385, mae: 5.0784, huber: 4.6239, swd: 4.7386, ept: 201.7946
      Epoch 8 composite train-obj: 3.478448
            Val objective improved 4.8403 → 4.7701, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 36.4629, mae: 3.8092, huber: 3.3659, swd: 4.2324, ept: 239.6218
    Epoch [9/50], Val Losses: mse: 63.7291, mae: 5.2175, huber: 4.7623, swd: 4.6074, ept: 206.7372
    Epoch [9/50], Test Losses: mse: 60.1182, mae: 5.0262, huber: 4.5725, swd: 4.5456, ept: 214.2667
      Epoch 9 composite train-obj: 3.365947
            Val objective improved 4.7701 → 4.7623, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 34.8642, mae: 3.6842, huber: 3.2437, swd: 3.8462, ept: 248.3660
    Epoch [10/50], Val Losses: mse: 64.3490, mae: 5.2337, huber: 4.7792, swd: 4.5645, ept: 208.5364
    Epoch [10/50], Test Losses: mse: 59.4250, mae: 5.0133, huber: 4.5603, swd: 4.6247, ept: 218.2927
      Epoch 10 composite train-obj: 3.243724
            No improvement (4.7792), counter 1/5
    Epoch [11/50], Train Losses: mse: 33.6904, mae: 3.5910, huber: 3.1527, swd: 3.5923, ept: 256.2889
    Epoch [11/50], Val Losses: mse: 64.5105, mae: 5.2423, huber: 4.7868, swd: 3.9447, ept: 203.6564
    Epoch [11/50], Test Losses: mse: 61.0490, mae: 5.0697, huber: 4.6160, swd: 4.0494, ept: 208.8172
      Epoch 11 composite train-obj: 3.152666
            No improvement (4.7868), counter 2/5
    Epoch [12/50], Train Losses: mse: 32.7729, mae: 3.5236, huber: 3.0865, swd: 3.4022, ept: 260.8944
    Epoch [12/50], Val Losses: mse: 64.8834, mae: 5.2596, huber: 4.8038, swd: 4.0363, ept: 204.2739
    Epoch [12/50], Test Losses: mse: 59.9325, mae: 5.0091, huber: 4.5558, swd: 4.1346, ept: 212.4861
      Epoch 12 composite train-obj: 3.086502
            No improvement (4.8038), counter 3/5
    Epoch [13/50], Train Losses: mse: 32.1100, mae: 3.4744, huber: 3.0382, swd: 3.2360, ept: 263.9785
    Epoch [13/50], Val Losses: mse: 65.9904, mae: 5.2417, huber: 4.7871, swd: 3.3293, ept: 220.2670
    Epoch [13/50], Test Losses: mse: 61.3415, mae: 4.9957, huber: 4.5441, swd: 3.4103, ept: 230.2855
      Epoch 13 composite train-obj: 3.038197
            No improvement (4.7871), counter 4/5
    Epoch [14/50], Train Losses: mse: 30.8275, mae: 3.3714, huber: 2.9381, swd: 3.0370, ept: 273.4388
    Epoch [14/50], Val Losses: mse: 65.6680, mae: 5.2029, huber: 4.7519, swd: 3.4599, ept: 216.2958
    Epoch [14/50], Test Losses: mse: 61.3140, mae: 4.9987, huber: 4.5497, swd: 3.5374, ept: 224.7729
      Epoch 14 composite train-obj: 2.938056
            Val objective improved 4.7623 → 4.7519, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 30.2516, mae: 3.3285, huber: 2.8958, swd: 2.9215, ept: 276.4003
    Epoch [15/50], Val Losses: mse: 62.7988, mae: 5.1111, huber: 4.6602, swd: 3.6351, ept: 220.0205
    Epoch [15/50], Test Losses: mse: 59.8016, mae: 4.9280, huber: 4.4793, swd: 3.6479, ept: 228.3538
      Epoch 15 composite train-obj: 2.895805
            Val objective improved 4.7519 → 4.6602, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 29.6129, mae: 3.2729, huber: 2.8420, swd: 2.8057, ept: 281.9535
    Epoch [16/50], Val Losses: mse: 66.3524, mae: 5.2325, huber: 4.7815, swd: 3.0387, ept: 217.5146
    Epoch [16/50], Test Losses: mse: 63.3001, mae: 5.0740, huber: 4.6253, swd: 3.0566, ept: 222.9541
      Epoch 16 composite train-obj: 2.841980
            No improvement (4.7815), counter 1/5
    Epoch [17/50], Train Losses: mse: 28.9799, mae: 3.2235, huber: 2.7938, swd: 2.7012, ept: 287.5169
    Epoch [17/50], Val Losses: mse: 66.6986, mae: 5.2484, huber: 4.7975, swd: 3.0325, ept: 220.7770
    Epoch [17/50], Test Losses: mse: 61.3995, mae: 4.9745, huber: 4.5259, swd: 3.1576, ept: 229.7688
      Epoch 17 composite train-obj: 2.793793
            No improvement (4.7975), counter 2/5
    Epoch [18/50], Train Losses: mse: 28.4774, mae: 3.1864, huber: 2.7574, swd: 2.5946, ept: 290.9454
    Epoch [18/50], Val Losses: mse: 66.0406, mae: 5.2031, huber: 4.7524, swd: 3.1193, ept: 223.9341
    Epoch [18/50], Test Losses: mse: 60.8621, mae: 4.9424, huber: 4.4940, swd: 3.2305, ept: 233.9359
      Epoch 18 composite train-obj: 2.757423
            No improvement (4.7524), counter 3/5
    Epoch [19/50], Train Losses: mse: 27.9520, mae: 3.1418, huber: 2.7139, swd: 2.5265, ept: 296.6073
    Epoch [19/50], Val Losses: mse: 65.0909, mae: 5.1959, huber: 4.7444, swd: 2.8991, ept: 220.0278
    Epoch [19/50], Test Losses: mse: 61.3549, mae: 4.9855, huber: 4.5362, swd: 2.9506, ept: 226.9646
      Epoch 19 composite train-obj: 2.713912
            No improvement (4.7444), counter 4/5
    Epoch [20/50], Train Losses: mse: 27.3488, mae: 3.0885, huber: 2.6624, swd: 2.4296, ept: 303.0607
    Epoch [20/50], Val Losses: mse: 67.5104, mae: 5.2520, huber: 4.8004, swd: 3.0079, ept: 225.4020
    Epoch [20/50], Test Losses: mse: 62.6067, mae: 5.0341, huber: 4.5849, swd: 3.2912, ept: 234.3842
      Epoch 20 composite train-obj: 2.662408
    Epoch [20/50], Test Losses: mse: 59.8011, mae: 4.9280, huber: 4.4793, swd: 3.6481, ept: 228.3467
    Best round's Test MSE: 59.8016, MAE: 4.9280, SWD: 3.6479
    Best round's Validation MSE: 62.7988, MAE: 5.1111, SWD: 3.6351
    Best round's Test verification MSE : 59.8011, MAE: 4.9280, SWD: 3.6481
    Time taken: 167.60 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 69.0664, mae: 6.2763, huber: 5.7981, swd: 36.3469, ept: 48.9153
    Epoch [1/50], Val Losses: mse: 59.8980, mae: 5.7116, huber: 5.2381, swd: 21.9846, ept: 79.4885
    Epoch [1/50], Test Losses: mse: 58.7185, mae: 5.6062, huber: 5.1335, swd: 21.4608, ept: 78.5364
      Epoch 1 composite train-obj: 5.798090
            Val objective improved inf → 5.2381, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 54.4167, mae: 5.2641, huber: 4.7959, swd: 16.7369, ept: 120.2138
    Epoch [2/50], Val Losses: mse: 59.9682, mae: 5.5326, huber: 5.0631, swd: 14.2109, ept: 128.3381
    Epoch [2/50], Test Losses: mse: 57.1889, mae: 5.3509, huber: 4.8832, swd: 14.0063, ept: 125.8294
      Epoch 2 composite train-obj: 4.795940
            Val objective improved 5.2381 → 5.0631, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 49.5548, mae: 4.8514, huber: 4.3889, swd: 11.1875, ept: 160.1248
    Epoch [3/50], Val Losses: mse: 59.8441, mae: 5.4215, huber: 4.9545, swd: 10.8646, ept: 147.6363
    Epoch [3/50], Test Losses: mse: 57.0383, mae: 5.2294, huber: 4.7645, swd: 10.1356, ept: 148.0099
      Epoch 3 composite train-obj: 4.388892
            Val objective improved 5.0631 → 4.9545, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 45.8980, mae: 4.5513, huber: 4.0938, swd: 8.5408, ept: 187.3569
    Epoch [4/50], Val Losses: mse: 59.8009, mae: 5.3193, huber: 4.8553, swd: 9.5619, ept: 168.5191
    Epoch [4/50], Test Losses: mse: 57.5516, mae: 5.1777, huber: 4.7156, swd: 9.2152, ept: 171.3680
      Epoch 4 composite train-obj: 4.093764
            Val objective improved 4.9545 → 4.8553, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 42.9631, mae: 4.3163, huber: 3.8627, swd: 6.9618, ept: 205.8045
    Epoch [5/50], Val Losses: mse: 62.0168, mae: 5.3280, huber: 4.8656, swd: 7.1773, ept: 180.8428
    Epoch [5/50], Test Losses: mse: 58.5691, mae: 5.1246, huber: 4.6645, swd: 6.9375, ept: 183.6866
      Epoch 5 composite train-obj: 3.862721
            No improvement (4.8656), counter 1/5
    Epoch [6/50], Train Losses: mse: 40.7152, mae: 4.1430, huber: 3.6924, swd: 5.9314, ept: 216.0133
    Epoch [6/50], Val Losses: mse: 61.7973, mae: 5.2464, huber: 4.7868, swd: 6.2113, ept: 190.6391
    Epoch [6/50], Test Losses: mse: 58.6266, mae: 5.0691, huber: 4.6112, swd: 5.8154, ept: 195.1553
      Epoch 6 composite train-obj: 3.692416
            Val objective improved 4.8553 → 4.7868, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 38.7127, mae: 3.9871, huber: 3.5395, swd: 5.1832, ept: 226.1932
    Epoch [7/50], Val Losses: mse: 64.1647, mae: 5.3436, huber: 4.8832, swd: 5.5729, ept: 183.4656
    Epoch [7/50], Test Losses: mse: 60.7292, mae: 5.1651, huber: 4.7070, swd: 5.4915, ept: 192.7495
      Epoch 7 composite train-obj: 3.539516
            No improvement (4.8832), counter 1/5
    Epoch [8/50], Train Losses: mse: 37.0387, mae: 3.8580, huber: 3.4128, swd: 4.6560, ept: 234.9091
    Epoch [8/50], Val Losses: mse: 64.3747, mae: 5.2997, huber: 4.8403, swd: 5.0039, ept: 197.7465
    Epoch [8/50], Test Losses: mse: 60.2461, mae: 5.0980, huber: 4.6403, swd: 4.8529, ept: 201.8586
      Epoch 8 composite train-obj: 3.412750
            No improvement (4.8403), counter 2/5
    Epoch [9/50], Train Losses: mse: 35.6492, mae: 3.7494, huber: 3.3067, swd: 4.2768, ept: 242.1942
    Epoch [9/50], Val Losses: mse: 64.1835, mae: 5.2101, huber: 4.7554, swd: 4.6026, ept: 206.4217
    Epoch [9/50], Test Losses: mse: 60.3383, mae: 5.0155, huber: 4.5633, swd: 4.6016, ept: 214.1705
      Epoch 9 composite train-obj: 3.306709
            Val objective improved 4.7868 → 4.7554, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 34.4153, mae: 3.6532, huber: 3.2126, swd: 3.9622, ept: 248.5911
    Epoch [10/50], Val Losses: mse: 64.2444, mae: 5.2317, huber: 4.7752, swd: 4.2155, ept: 208.9138
    Epoch [10/50], Test Losses: mse: 59.9466, mae: 5.0079, huber: 4.5541, swd: 4.2844, ept: 217.4881
      Epoch 10 composite train-obj: 3.212553
            No improvement (4.7752), counter 1/5
    Epoch [11/50], Train Losses: mse: 33.1190, mae: 3.5548, huber: 3.1162, swd: 3.6621, ept: 254.9008
    Epoch [11/50], Val Losses: mse: 65.7033, mae: 5.2650, huber: 4.8106, swd: 3.8239, ept: 209.3752
    Epoch [11/50], Test Losses: mse: 61.5636, mae: 5.0649, huber: 4.6125, swd: 4.0413, ept: 218.2432
      Epoch 11 composite train-obj: 3.116228
            No improvement (4.8106), counter 2/5
    Epoch [12/50], Train Losses: mse: 32.1635, mae: 3.4839, huber: 3.0469, swd: 3.4747, ept: 260.6746
    Epoch [12/50], Val Losses: mse: 65.3749, mae: 5.2228, huber: 4.7688, swd: 3.7720, ept: 213.0440
    Epoch [12/50], Test Losses: mse: 60.2860, mae: 4.9678, huber: 4.5168, swd: 3.6781, ept: 224.2896
      Epoch 12 composite train-obj: 3.046874
            No improvement (4.7688), counter 3/5
    Epoch [13/50], Train Losses: mse: 31.1181, mae: 3.3990, huber: 2.9642, swd: 3.2592, ept: 267.5285
    Epoch [13/50], Val Losses: mse: 63.8484, mae: 5.1852, huber: 4.7289, swd: 4.7669, ept: 210.4004
    Epoch [13/50], Test Losses: mse: 59.8034, mae: 4.9844, huber: 4.5309, swd: 4.6613, ept: 220.4545
      Epoch 13 composite train-obj: 2.964215
            Val objective improved 4.7554 → 4.7289, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 30.6986, mae: 3.3660, huber: 2.9318, swd: 3.1885, ept: 272.4895
    Epoch [14/50], Val Losses: mse: 64.1668, mae: 5.1598, huber: 4.7063, swd: 3.6697, ept: 216.1626
    Epoch [14/50], Test Losses: mse: 60.8749, mae: 4.9900, huber: 4.5389, swd: 3.5631, ept: 223.8928
      Epoch 14 composite train-obj: 2.931754
            Val objective improved 4.7289 → 4.7063, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 29.4622, mae: 3.2692, huber: 2.8376, swd: 2.9638, ept: 279.5730
    Epoch [15/50], Val Losses: mse: 66.2933, mae: 5.2578, huber: 4.8055, swd: 3.5981, ept: 214.1887
    Epoch [15/50], Test Losses: mse: 61.5273, mae: 5.0155, huber: 4.5653, swd: 3.5262, ept: 222.8219
      Epoch 15 composite train-obj: 2.837564
            No improvement (4.8055), counter 1/5
    Epoch [16/50], Train Losses: mse: 28.9063, mae: 3.2233, huber: 2.7928, swd: 2.8739, ept: 284.9364
    Epoch [16/50], Val Losses: mse: 66.6551, mae: 5.2497, huber: 4.7968, swd: 3.1064, ept: 215.9638
    Epoch [16/50], Test Losses: mse: 62.9496, mae: 5.0315, huber: 4.5813, swd: 2.9630, ept: 228.4534
      Epoch 16 composite train-obj: 2.792783
            No improvement (4.7968), counter 2/5
    Epoch [17/50], Train Losses: mse: 28.5400, mae: 3.1940, huber: 2.7639, swd: 2.7943, ept: 287.6699
    Epoch [17/50], Val Losses: mse: 66.2699, mae: 5.2571, huber: 4.8037, swd: 3.7673, ept: 216.0698
    Epoch [17/50], Test Losses: mse: 60.2436, mae: 4.9571, huber: 4.5072, swd: 3.6490, ept: 232.1093
      Epoch 17 composite train-obj: 2.763934
            No improvement (4.8037), counter 3/5
    Epoch [18/50], Train Losses: mse: 27.8634, mae: 3.1372, huber: 2.7088, swd: 2.6868, ept: 293.2518
    Epoch [18/50], Val Losses: mse: 66.0304, mae: 5.2244, huber: 4.7731, swd: 3.2173, ept: 223.9722
    Epoch [18/50], Test Losses: mse: 61.1634, mae: 4.9438, huber: 4.4958, swd: 2.9738, ept: 238.3491
      Epoch 18 composite train-obj: 2.708846
            No improvement (4.7731), counter 4/5
    Epoch [19/50], Train Losses: mse: 26.5914, mae: 3.0342, huber: 2.6090, swd: 2.4998, ept: 301.0370
    Epoch [19/50], Val Losses: mse: 66.5338, mae: 5.1983, huber: 4.7498, swd: 3.2279, ept: 228.7162
    Epoch [19/50], Test Losses: mse: 60.0361, mae: 4.8629, huber: 4.4178, swd: 3.0618, ept: 243.7873
      Epoch 19 composite train-obj: 2.609024
    Epoch [19/50], Test Losses: mse: 60.8759, mae: 4.9901, huber: 4.5389, swd: 3.5629, ept: 223.8808
    Best round's Test MSE: 60.8749, MAE: 4.9900, SWD: 3.5631
    Best round's Validation MSE: 64.1668, MAE: 5.1598, SWD: 3.6697
    Best round's Test verification MSE : 60.8759, MAE: 4.9901, SWD: 3.5629
    Time taken: 154.57 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred720_20250510_0915)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 59.8581 ± 0.8081
      mae: 4.9497 ± 0.0285
      huber: 4.4995 ± 0.0278
      swd: 3.9214 ± 0.4480
      ept: 225.7067 ± 1.9141
      count: 33.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 63.1999 ± 0.6870
      mae: 5.1332 ± 0.0201
      huber: 4.6808 ± 0.0191
      swd: 3.8847 ± 0.3288
      ept: 217.7054 ± 1.6670
      count: 33.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 450.70 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred720_20250510_0915
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### TimeMixer

#### 336-96
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
    pred_len=96,
    channels=data_mgr.datasets['lorenz']['channels'],
    enc_in=data_mgr.datasets['lorenz']['channels'],
    dec_in=data_mgr.datasets['lorenz']['channels'],
    c_out=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0]
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 282
    Validation Batches: 38
    Test Batches: 78
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 45.7229, mae: 4.5079, huber: 4.0457, swd: 11.0788, ept: 55.8223
    Epoch [1/50], Val Losses: mse: 29.3050, mae: 3.4506, huber: 3.0014, swd: 9.0607, ept: 70.9767
    Epoch [1/50], Test Losses: mse: 27.6317, mae: 3.3571, huber: 2.9077, swd: 8.9950, ept: 72.0272
      Epoch 1 composite train-obj: 4.045677
            Val objective improved inf → 3.0014, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 19.1868, mae: 2.5216, huber: 2.0921, swd: 6.0959, ept: 80.0678
    Epoch [2/50], Val Losses: mse: 18.6010, mae: 2.4746, huber: 2.0489, swd: 5.7784, ept: 81.0637
    Epoch [2/50], Test Losses: mse: 16.6945, mae: 2.3537, huber: 1.9305, swd: 5.8536, ept: 82.2268
      Epoch 2 composite train-obj: 2.092061
            Val objective improved 3.0014 → 2.0489, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.4969, mae: 1.9035, huber: 1.4955, swd: 4.1291, ept: 85.8976
    Epoch [3/50], Val Losses: mse: 11.6419, mae: 1.8031, huber: 1.4016, swd: 3.9173, ept: 86.9350
    Epoch [3/50], Test Losses: mse: 9.9622, mae: 1.7104, huber: 1.3101, swd: 3.6449, ept: 87.8623
      Epoch 3 composite train-obj: 1.495525
            Val objective improved 2.0489 → 1.4016, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.2477, mae: 1.4693, huber: 1.0829, swd: 2.5271, ept: 89.7754
    Epoch [4/50], Val Losses: mse: 8.3719, mae: 1.4929, huber: 1.1017, swd: 2.9090, ept: 90.2018
    Epoch [4/50], Test Losses: mse: 6.8355, mae: 1.4319, huber: 1.0399, swd: 2.2212, ept: 91.1534
      Epoch 4 composite train-obj: 1.082857
            Val objective improved 1.4016 → 1.1017, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.0519, mae: 1.2218, huber: 0.8517, swd: 1.6902, ept: 91.8317
    Epoch [5/50], Val Losses: mse: 6.7087, mae: 1.2320, huber: 0.8670, swd: 2.1606, ept: 91.3680
    Epoch [5/50], Test Losses: mse: 5.2134, mae: 1.1508, huber: 0.7860, swd: 1.6463, ept: 92.0950
      Epoch 5 composite train-obj: 0.851709
            Val objective improved 1.1017 → 0.8670, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.0066, mae: 1.0774, huber: 0.7216, swd: 1.3082, ept: 92.6620
    Epoch [6/50], Val Losses: mse: 5.5510, mae: 1.0999, huber: 0.7519, swd: 1.7485, ept: 92.0909
    Epoch [6/50], Test Losses: mse: 4.5947, mae: 1.0477, huber: 0.7010, swd: 1.2746, ept: 92.7010
      Epoch 6 composite train-obj: 0.721569
            Val objective improved 0.8670 → 0.7519, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.3287, mae: 0.9800, huber: 0.6354, swd: 1.0833, ept: 93.1717
    Epoch [7/50], Val Losses: mse: 5.3296, mae: 1.1588, huber: 0.8011, swd: 1.8273, ept: 91.8266
    Epoch [7/50], Test Losses: mse: 5.2347, mae: 1.1444, huber: 0.7851, swd: 1.5007, ept: 92.4095
      Epoch 7 composite train-obj: 0.635447
            No improvement (0.8011), counter 1/5
    Epoch [8/50], Train Losses: mse: 4.1962, mae: 0.9486, huber: 0.6098, swd: 1.0405, ept: 93.3127
    Epoch [8/50], Val Losses: mse: 7.0671, mae: 1.3279, huber: 0.9415, swd: 1.9862, ept: 91.4863
    Epoch [8/50], Test Losses: mse: 5.7265, mae: 1.2995, huber: 0.9101, swd: 1.7258, ept: 92.2356
      Epoch 8 composite train-obj: 0.609788
            No improvement (0.9415), counter 2/5
    Epoch [9/50], Train Losses: mse: 4.0169, mae: 0.9210, huber: 0.5873, swd: 1.0294, ept: 93.3524
    Epoch [9/50], Val Losses: mse: 4.1895, mae: 0.9707, huber: 0.6246, swd: 1.3232, ept: 93.1197
    Epoch [9/50], Test Losses: mse: 3.6616, mae: 0.9216, huber: 0.5793, swd: 0.9992, ept: 93.4409
      Epoch 9 composite train-obj: 0.587278
            Val objective improved 0.7519 → 0.6246, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 3.4411, mae: 0.8411, huber: 0.5177, swd: 0.8242, ept: 93.8084
    Epoch [10/50], Val Losses: mse: 9.5444, mae: 1.4041, huber: 1.0358, swd: 2.0428, ept: 90.7302
    Epoch [10/50], Test Losses: mse: 7.7067, mae: 1.3018, huber: 0.9385, swd: 1.6293, ept: 91.5275
      Epoch 10 composite train-obj: 0.517669
            No improvement (1.0358), counter 1/5
    Epoch [11/50], Train Losses: mse: 4.3779, mae: 0.9358, huber: 0.6040, swd: 1.0003, ept: 93.4137
    Epoch [11/50], Val Losses: mse: 4.2699, mae: 0.9097, huber: 0.5864, swd: 1.2721, ept: 92.9459
    Epoch [11/50], Test Losses: mse: 3.6428, mae: 0.8517, huber: 0.5321, swd: 0.8640, ept: 93.7045
      Epoch 11 composite train-obj: 0.604049
            Val objective improved 0.6246 → 0.5864, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 3.0694, mae: 0.7806, huber: 0.4672, swd: 0.7153, ept: 94.0851
    Epoch [12/50], Val Losses: mse: 3.8025, mae: 0.8667, huber: 0.5470, swd: 1.2104, ept: 93.5232
    Epoch [12/50], Test Losses: mse: 3.4536, mae: 0.8379, huber: 0.5188, swd: 0.8472, ept: 94.0049
      Epoch 12 composite train-obj: 0.467229
            Val objective improved 0.5864 → 0.5470, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 2.9100, mae: 0.7524, huber: 0.4446, swd: 0.6538, ept: 94.2016
    Epoch [13/50], Val Losses: mse: 3.5002, mae: 0.7683, huber: 0.4724, swd: 1.0671, ept: 93.3276
    Epoch [13/50], Test Losses: mse: 2.9644, mae: 0.7265, huber: 0.4308, swd: 0.8515, ept: 93.8921
      Epoch 13 composite train-obj: 0.444575
            Val objective improved 0.5470 → 0.4724, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 2.7467, mae: 0.7263, huber: 0.4232, swd: 0.6234, ept: 94.3184
    Epoch [14/50], Val Losses: mse: 4.0488, mae: 0.8311, huber: 0.5268, swd: 0.9672, ept: 93.3775
    Epoch [14/50], Test Losses: mse: 3.6332, mae: 0.7908, huber: 0.4893, swd: 0.8141, ept: 93.9697
      Epoch 14 composite train-obj: 0.423247
            No improvement (0.5268), counter 1/5
    Epoch [15/50], Train Losses: mse: 2.6268, mae: 0.7087, huber: 0.4094, swd: 0.5860, ept: 94.3712
    Epoch [15/50], Val Losses: mse: 8.3800, mae: 1.1493, huber: 0.8197, swd: 1.2190, ept: 91.3966
    Epoch [15/50], Test Losses: mse: 7.4152, mae: 1.1070, huber: 0.7806, swd: 1.0005, ept: 91.5936
      Epoch 15 composite train-obj: 0.409407
            No improvement (0.8197), counter 2/5
    Epoch [16/50], Train Losses: mse: 3.9266, mae: 0.8540, huber: 0.5364, swd: 0.9593, ept: 93.5971
    Epoch [16/50], Val Losses: mse: 4.3379, mae: 1.0799, huber: 0.7031, swd: 1.5130, ept: 93.3349
    Epoch [16/50], Test Losses: mse: 3.9800, mae: 1.0535, huber: 0.6746, swd: 1.1352, ept: 93.8239
      Epoch 16 composite train-obj: 0.536400
            No improvement (0.7031), counter 3/5
    Epoch [17/50], Train Losses: mse: 2.7959, mae: 0.7198, huber: 0.4190, swd: 0.6322, ept: 94.3831
    Epoch [17/50], Val Losses: mse: 4.1992, mae: 0.8790, huber: 0.5601, swd: 0.9588, ept: 93.4979
    Epoch [17/50], Test Losses: mse: 3.3724, mae: 0.8153, huber: 0.4996, swd: 0.9260, ept: 93.9803
      Epoch 17 composite train-obj: 0.419016
            No improvement (0.5601), counter 4/5
    Epoch [18/50], Train Losses: mse: 2.6018, mae: 0.6896, huber: 0.3949, swd: 0.5637, ept: 94.5117
    Epoch [18/50], Val Losses: mse: 4.8046, mae: 0.8974, huber: 0.5797, swd: 1.1669, ept: 93.1464
    Epoch [18/50], Test Losses: mse: 4.1170, mae: 0.8599, huber: 0.5439, swd: 1.0560, ept: 93.4404
      Epoch 18 composite train-obj: 0.394908
    Epoch [18/50], Test Losses: mse: 2.9644, mae: 0.7265, huber: 0.4308, swd: 0.8515, ept: 93.8921
    Best round's Test MSE: 2.9644, MAE: 0.7265, SWD: 0.8515
    Best round's Validation MSE: 3.5002, MAE: 0.7683, SWD: 1.0671
    Best round's Test verification MSE : 2.9644, MAE: 0.7265, SWD: 0.8515
    Time taken: 160.17 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 38.9384, mae: 4.0217, huber: 3.5675, swd: 9.9864, ept: 60.5328
    Epoch [1/50], Val Losses: mse: 23.4746, mae: 3.0610, huber: 2.6183, swd: 8.7911, ept: 75.8403
    Epoch [1/50], Test Losses: mse: 21.5083, mae: 2.9459, huber: 2.5041, swd: 8.7933, ept: 77.0732
      Epoch 1 composite train-obj: 3.567465
            Val objective improved inf → 2.6183, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 13.9864, mae: 2.0580, huber: 1.6454, swd: 4.9582, ept: 84.3213
    Epoch [2/50], Val Losses: mse: 12.9931, mae: 1.9479, huber: 1.5378, swd: 4.4264, ept: 86.1767
    Epoch [2/50], Test Losses: mse: 11.2365, mae: 1.8602, huber: 1.4505, swd: 4.1576, ept: 86.8186
      Epoch 2 composite train-obj: 1.645443
            Val objective improved 2.6183 → 1.5378, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.8183, mae: 1.5142, huber: 1.1278, swd: 3.0487, ept: 89.1475
    Epoch [3/50], Val Losses: mse: 10.1478, mae: 1.6983, huber: 1.2993, swd: 3.5092, ept: 88.5367
    Epoch [3/50], Test Losses: mse: 8.9166, mae: 1.6595, huber: 1.2587, swd: 3.4946, ept: 89.3281
      Epoch 3 composite train-obj: 1.127767
            Val objective improved 1.5378 → 1.2993, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.8839, mae: 1.2789, huber: 0.9097, swd: 2.2928, ept: 90.9670
    Epoch [4/50], Val Losses: mse: 7.5341, mae: 1.3475, huber: 0.9820, swd: 2.6900, ept: 90.7237
    Epoch [4/50], Test Losses: mse: 6.4101, mae: 1.2861, huber: 0.9217, swd: 2.4307, ept: 91.0484
      Epoch 4 composite train-obj: 0.909744
            Val objective improved 1.2993 → 0.9820, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.4111, mae: 1.0977, huber: 0.7441, swd: 1.6573, ept: 92.2096
    Epoch [5/50], Val Losses: mse: 5.7530, mae: 1.0818, huber: 0.7393, swd: 1.8947, ept: 91.6785
    Epoch [5/50], Test Losses: mse: 4.5225, mae: 1.0102, huber: 0.6677, swd: 1.5798, ept: 92.7587
      Epoch 5 composite train-obj: 0.744051
            Val objective improved 0.9820 → 0.7393, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.2670, mae: 0.9479, huber: 0.6095, swd: 1.2253, ept: 93.1393
    Epoch [6/50], Val Losses: mse: 5.0817, mae: 1.0870, huber: 0.7333, swd: 1.8979, ept: 92.5252
    Epoch [6/50], Test Losses: mse: 4.0318, mae: 1.0237, huber: 0.6698, swd: 1.4075, ept: 93.3857
      Epoch 6 composite train-obj: 0.609491
            Val objective improved 0.7393 → 0.7333, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.9540, mae: 0.9084, huber: 0.5762, swd: 1.1100, ept: 93.3576
    Epoch [7/50], Val Losses: mse: 5.2061, mae: 1.0648, huber: 0.7206, swd: 1.5174, ept: 92.2781
    Epoch [7/50], Test Losses: mse: 4.4209, mae: 1.0096, huber: 0.6690, swd: 1.2614, ept: 92.9891
      Epoch 7 composite train-obj: 0.576238
            Val objective improved 0.7333 → 0.7206, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.6713, mae: 0.8628, huber: 0.5373, swd: 0.9932, ept: 93.5869
    Epoch [8/50], Val Losses: mse: 4.2258, mae: 0.8466, huber: 0.5340, swd: 1.3920, ept: 92.9642
    Epoch [8/50], Test Losses: mse: 3.4267, mae: 0.7866, huber: 0.4769, swd: 1.0546, ept: 93.6785
      Epoch 8 composite train-obj: 0.537268
            Val objective improved 0.7206 → 0.5340, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.1157, mae: 0.7708, huber: 0.4597, swd: 0.8259, ept: 94.0260
    Epoch [9/50], Val Losses: mse: 5.6202, mae: 1.1261, huber: 0.7760, swd: 1.7434, ept: 92.3141
    Epoch [9/50], Test Losses: mse: 4.4137, mae: 1.0779, huber: 0.7261, swd: 1.5322, ept: 92.8145
      Epoch 9 composite train-obj: 0.459749
            No improvement (0.7760), counter 1/5
    Epoch [10/50], Train Losses: mse: 3.1176, mae: 0.7776, huber: 0.4664, swd: 0.8195, ept: 93.9987
    Epoch [10/50], Val Losses: mse: 3.9383, mae: 0.8379, huber: 0.5283, swd: 1.4564, ept: 93.2995
    Epoch [10/50], Test Losses: mse: 3.4199, mae: 0.7810, huber: 0.4723, swd: 1.0028, ept: 94.0224
      Epoch 10 composite train-obj: 0.466409
            Val objective improved 0.5340 → 0.5283, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 2.7606, mae: 0.7192, huber: 0.4178, swd: 0.7108, ept: 94.2520
    Epoch [11/50], Val Losses: mse: 4.3730, mae: 0.9987, huber: 0.6590, swd: 1.5765, ept: 92.7438
    Epoch [11/50], Test Losses: mse: 3.8363, mae: 0.9555, huber: 0.6172, swd: 1.2905, ept: 93.4810
      Epoch 11 composite train-obj: 0.417808
            No improvement (0.6590), counter 1/5
    Epoch [12/50], Train Losses: mse: 2.7210, mae: 0.7238, huber: 0.4223, swd: 0.7282, ept: 94.2615
    Epoch [12/50], Val Losses: mse: 4.3642, mae: 0.9032, huber: 0.5840, swd: 1.4434, ept: 93.2249
    Epoch [12/50], Test Losses: mse: 3.6801, mae: 0.8555, huber: 0.5366, swd: 0.9940, ept: 93.6523
      Epoch 12 composite train-obj: 0.422305
            No improvement (0.5840), counter 2/5
    Epoch [13/50], Train Losses: mse: 2.5166, mae: 0.6963, huber: 0.3993, swd: 0.6350, ept: 94.4099
    Epoch [13/50], Val Losses: mse: 4.8171, mae: 0.8886, huber: 0.5818, swd: 1.3245, ept: 93.0802
    Epoch [13/50], Test Losses: mse: 4.3065, mae: 0.8538, huber: 0.5473, swd: 1.0008, ept: 93.3522
      Epoch 13 composite train-obj: 0.399319
            No improvement (0.5818), counter 3/5
    Epoch [14/50], Train Losses: mse: 2.8675, mae: 0.7182, huber: 0.4207, swd: 0.6944, ept: 94.1627
    Epoch [14/50], Val Losses: mse: 3.5957, mae: 0.8614, huber: 0.5449, swd: 1.0712, ept: 93.8604
    Epoch [14/50], Test Losses: mse: 3.2004, mae: 0.8081, huber: 0.4973, swd: 0.8956, ept: 94.0530
      Epoch 14 composite train-obj: 0.420672
            No improvement (0.5449), counter 4/5
    Epoch [15/50], Train Losses: mse: 2.4007, mae: 0.6675, huber: 0.3773, swd: 0.5795, ept: 94.5301
    Epoch [15/50], Val Losses: mse: 7.0307, mae: 1.2237, huber: 0.8697, swd: 2.0529, ept: 92.1392
    Epoch [15/50], Test Losses: mse: 5.9326, mae: 1.1756, huber: 0.8222, swd: 2.0809, ept: 92.7234
      Epoch 15 composite train-obj: 0.377252
    Epoch [15/50], Test Losses: mse: 3.4199, mae: 0.7810, huber: 0.4723, swd: 1.0028, ept: 94.0224
    Best round's Test MSE: 3.4199, MAE: 0.7810, SWD: 1.0028
    Best round's Validation MSE: 3.9383, MAE: 0.8379, SWD: 1.4564
    Best round's Test verification MSE : 3.4199, MAE: 0.7810, SWD: 1.0028
    Time taken: 137.51 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 51.8757, mae: 4.8485, huber: 4.3835, swd: 10.4915, ept: 50.4415
    Epoch [1/50], Val Losses: mse: 31.2130, mae: 3.5460, huber: 3.0954, swd: 8.1265, ept: 68.9936
    Epoch [1/50], Test Losses: mse: 29.8708, mae: 3.4755, huber: 3.0252, swd: 8.5545, ept: 68.9499
      Epoch 1 composite train-obj: 4.383528
            Val objective improved inf → 3.0954, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 18.5855, mae: 2.5030, huber: 2.0748, swd: 5.5899, ept: 79.5752
    Epoch [2/50], Val Losses: mse: 15.5636, mae: 2.1499, huber: 1.7354, swd: 4.5548, ept: 82.8853
    Epoch [2/50], Test Losses: mse: 13.5650, mae: 2.0581, huber: 1.6445, swd: 4.3677, ept: 83.7105
      Epoch 2 composite train-obj: 2.074840
            Val objective improved 3.0954 → 1.7354, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 10.4531, mae: 1.7209, huber: 1.3213, swd: 3.2179, ept: 87.2518
    Epoch [3/50], Val Losses: mse: 9.6165, mae: 1.6917, huber: 1.2947, swd: 3.2792, ept: 87.8379
    Epoch [3/50], Test Losses: mse: 8.2171, mae: 1.6305, huber: 1.2324, swd: 2.7154, ept: 89.1762
      Epoch 3 composite train-obj: 1.321291
            Val objective improved 1.7354 → 1.2947, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.1351, mae: 1.3456, huber: 0.9684, swd: 2.1057, ept: 90.4886
    Epoch [4/50], Val Losses: mse: 8.1573, mae: 1.4109, huber: 1.0333, swd: 2.3242, ept: 89.7878
    Epoch [4/50], Test Losses: mse: 6.1576, mae: 1.3040, huber: 0.9287, swd: 1.9591, ept: 91.4637
      Epoch 4 composite train-obj: 0.968384
            Val objective improved 1.2947 → 1.0333, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.4853, mae: 1.1278, huber: 0.7691, swd: 1.5274, ept: 92.0460
    Epoch [5/50], Val Losses: mse: 6.1049, mae: 1.2424, huber: 0.8749, swd: 1.7088, ept: 91.0956
    Epoch [5/50], Test Losses: mse: 5.2600, mae: 1.2010, huber: 0.8317, swd: 1.5073, ept: 92.0171
      Epoch 5 composite train-obj: 0.769078
            Val objective improved 1.0333 → 0.8749, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.7077, mae: 1.0208, huber: 0.6743, swd: 1.2836, ept: 92.7407
    Epoch [6/50], Val Losses: mse: 6.4890, mae: 1.1671, huber: 0.8114, swd: 1.5756, ept: 91.3766
    Epoch [6/50], Test Losses: mse: 4.8420, mae: 1.0974, huber: 0.7416, swd: 1.3017, ept: 92.1801
      Epoch 6 composite train-obj: 0.674280
            Val objective improved 0.8749 → 0.8114, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.1662, mae: 0.9316, huber: 0.5971, swd: 1.0861, ept: 93.0691
    Epoch [7/50], Val Losses: mse: 4.4873, mae: 0.9706, huber: 0.6321, swd: 1.3057, ept: 92.5871
    Epoch [7/50], Test Losses: mse: 3.5911, mae: 0.9190, huber: 0.5809, swd: 1.1028, ept: 93.5150
      Epoch 7 composite train-obj: 0.597058
            Val objective improved 0.8114 → 0.6321, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.5584, mae: 0.8463, huber: 0.5233, swd: 0.9221, ept: 93.6446
    Epoch [8/50], Val Losses: mse: 6.0932, mae: 1.1509, huber: 0.8039, swd: 1.6556, ept: 90.7379
    Epoch [8/50], Test Losses: mse: 5.9850, mae: 1.1463, huber: 0.7995, swd: 1.3823, ept: 91.2778
      Epoch 8 composite train-obj: 0.523313
            No improvement (0.8039), counter 1/5
    Epoch [9/50], Train Losses: mse: 4.1511, mae: 0.8925, huber: 0.5680, swd: 1.0663, ept: 93.1474
    Epoch [9/50], Val Losses: mse: 6.0514, mae: 1.0380, huber: 0.7091, swd: 1.5185, ept: 91.8146
    Epoch [9/50], Test Losses: mse: 5.3174, mae: 0.9969, huber: 0.6682, swd: 1.2695, ept: 92.4831
      Epoch 9 composite train-obj: 0.567956
            No improvement (0.7091), counter 2/5
    Epoch [10/50], Train Losses: mse: 3.5116, mae: 0.8160, huber: 0.5017, swd: 0.9059, ept: 93.5662
    Epoch [10/50], Val Losses: mse: 3.6666, mae: 0.8553, huber: 0.5383, swd: 1.3426, ept: 93.0184
    Epoch [10/50], Test Losses: mse: 3.0627, mae: 0.8023, huber: 0.4862, swd: 1.1681, ept: 93.5162
      Epoch 10 composite train-obj: 0.501725
            Val objective improved 0.6321 → 0.5383, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 2.8638, mae: 0.7414, huber: 0.4362, swd: 0.7099, ept: 94.1233
    Epoch [11/50], Val Losses: mse: 3.7785, mae: 0.8732, huber: 0.5536, swd: 1.0260, ept: 93.5949
    Epoch [11/50], Test Losses: mse: 2.9043, mae: 0.8107, huber: 0.4934, swd: 0.7956, ept: 94.0062
      Epoch 11 composite train-obj: 0.436246
            No improvement (0.5536), counter 1/5
    Epoch [12/50], Train Losses: mse: 2.7053, mae: 0.7209, huber: 0.4203, swd: 0.6516, ept: 94.2591
    Epoch [12/50], Val Losses: mse: 3.4802, mae: 0.8350, huber: 0.5186, swd: 0.9573, ept: 93.7654
    Epoch [12/50], Test Losses: mse: 3.1042, mae: 0.8062, huber: 0.4904, swd: 0.8652, ept: 93.9075
      Epoch 12 composite train-obj: 0.420286
            Val objective improved 0.5383 → 0.5186, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 2.5485, mae: 0.6894, huber: 0.3950, swd: 0.6004, ept: 94.3607
    Epoch [13/50], Val Losses: mse: 4.1105, mae: 0.8705, huber: 0.5600, swd: 1.0187, ept: 92.8194
    Epoch [13/50], Test Losses: mse: 3.4093, mae: 0.8185, huber: 0.5099, swd: 0.8447, ept: 93.3742
      Epoch 13 composite train-obj: 0.394984
            No improvement (0.5600), counter 1/5
    Epoch [14/50], Train Losses: mse: 2.4935, mae: 0.6796, huber: 0.3874, swd: 0.5759, ept: 94.4040
    Epoch [14/50], Val Losses: mse: 3.1338, mae: 0.7939, huber: 0.4846, swd: 0.9113, ept: 93.6361
    Epoch [14/50], Test Losses: mse: 2.9271, mae: 0.7585, huber: 0.4526, swd: 0.6327, ept: 94.1421
      Epoch 14 composite train-obj: 0.387367
            Val objective improved 0.5186 → 0.4846, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 2.3755, mae: 0.6591, huber: 0.3704, swd: 0.5391, ept: 94.5212
    Epoch [15/50], Val Losses: mse: 2.5933, mae: 0.6886, huber: 0.4045, swd: 0.7574, ept: 94.0012
    Epoch [15/50], Test Losses: mse: 2.5196, mae: 0.6631, huber: 0.3803, swd: 0.6236, ept: 94.3111
      Epoch 15 composite train-obj: 0.370411
            Val objective improved 0.4846 → 0.4045, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 2.1425, mae: 0.6256, huber: 0.3442, swd: 0.4940, ept: 94.6560
    Epoch [16/50], Val Losses: mse: 5.0518, mae: 0.9585, huber: 0.6399, swd: 1.0243, ept: 92.9661
    Epoch [16/50], Test Losses: mse: 4.5306, mae: 0.9262, huber: 0.6071, swd: 0.9162, ept: 93.2831
      Epoch 16 composite train-obj: 0.344184
            No improvement (0.6399), counter 1/5
    Epoch [17/50], Train Losses: mse: 2.8222, mae: 0.7014, huber: 0.4093, swd: 0.6922, ept: 94.3120
    Epoch [17/50], Val Losses: mse: 3.0878, mae: 0.7956, huber: 0.4854, swd: 1.0355, ept: 93.9824
    Epoch [17/50], Test Losses: mse: 2.8839, mae: 0.7912, huber: 0.4778, swd: 0.8435, ept: 94.0636
      Epoch 17 composite train-obj: 0.409330
            No improvement (0.4854), counter 2/5
    Epoch [18/50], Train Losses: mse: 2.1824, mae: 0.6341, huber: 0.3515, swd: 0.5124, ept: 94.6242
    Epoch [18/50], Val Losses: mse: 3.0279, mae: 0.7309, huber: 0.4401, swd: 0.9182, ept: 93.9442
    Epoch [18/50], Test Losses: mse: 2.8077, mae: 0.6902, huber: 0.4033, swd: 0.6739, ept: 94.2069
      Epoch 18 composite train-obj: 0.351485
            No improvement (0.4401), counter 3/5
    Epoch [19/50], Train Losses: mse: 2.0104, mae: 0.6026, huber: 0.3274, swd: 0.4454, ept: 94.6708
    Epoch [19/50], Val Losses: mse: 6.7419, mae: 1.0340, huber: 0.7165, swd: 1.3801, ept: 91.5327
    Epoch [19/50], Test Losses: mse: 6.3764, mae: 0.9972, huber: 0.6798, swd: 0.7616, ept: 92.1741
      Epoch 19 composite train-obj: 0.327438
            No improvement (0.7165), counter 4/5
    Epoch [20/50], Train Losses: mse: 3.5549, mae: 0.7808, huber: 0.4810, swd: 0.8051, ept: 93.3388
    Epoch [20/50], Val Losses: mse: 3.0247, mae: 0.8013, huber: 0.4895, swd: 0.9120, ept: 93.6659
    Epoch [20/50], Test Losses: mse: 2.6404, mae: 0.7531, huber: 0.4449, swd: 0.8831, ept: 94.0620
      Epoch 20 composite train-obj: 0.480969
    Epoch [20/50], Test Losses: mse: 2.5196, mae: 0.6631, huber: 0.3803, swd: 0.6236, ept: 94.3111
    Best round's Test MSE: 2.5196, MAE: 0.6631, SWD: 0.6236
    Best round's Validation MSE: 2.5933, MAE: 0.6886, SWD: 0.7574
    Best round's Test verification MSE : 2.5196, MAE: 0.6631, SWD: 0.6236
    Time taken: 182.32 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq336_pred96_20250513_2252)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 2.9680 ± 0.3676
      mae: 0.7235 ± 0.0482
      huber: 0.4278 ± 0.0376
      swd: 0.8260 ± 0.1559
      ept: 94.0752 ± 0.1751
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 3.3439 ± 0.5601
      mae: 0.7649 ± 0.0610
      huber: 0.4684 ± 0.0506
      swd: 1.0936 ± 0.2860
      ept: 93.5428 ± 0.3244
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 480.07 seconds
    
    Experiment complete: TimeMixer_lorenz_seq336_pred96_20250513_2252
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 336-196
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
    pred_len=196,
    channels=data_mgr.datasets['lorenz']['channels'],
    enc_in=data_mgr.datasets['lorenz']['channels'],
    dec_in=data_mgr.datasets['lorenz']['channels'],
    c_out=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0]
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 281
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 281
    Validation Batches: 37
    Test Batches: 78
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 54.1921, mae: 5.1094, huber: 4.6413, swd: 12.8125, ept: 72.8911
    Epoch [1/50], Val Losses: mse: 41.6002, mae: 4.2256, huber: 3.7706, swd: 10.7168, ept: 105.7076
    Epoch [1/50], Test Losses: mse: 40.2723, mae: 4.1421, huber: 3.6880, swd: 11.2038, ept: 104.0207
      Epoch 1 composite train-obj: 4.641261
            Val objective improved inf → 3.7706, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 32.3574, mae: 3.4990, huber: 3.0546, swd: 7.7088, ept: 124.0870
    Epoch [2/50], Val Losses: mse: 35.7523, mae: 3.6245, huber: 3.1842, swd: 6.9998, ept: 126.8613
    Epoch [2/50], Test Losses: mse: 32.8248, mae: 3.4664, huber: 3.0272, swd: 7.2945, ept: 126.1666
      Epoch 2 composite train-obj: 3.054625
            Val objective improved 3.7706 → 3.1842, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 24.5578, mae: 2.8735, huber: 2.4430, swd: 5.1036, ept: 141.2295
    Epoch [3/50], Val Losses: mse: 27.8713, mae: 3.0695, huber: 2.6390, swd: 5.2985, ept: 142.2531
    Epoch [3/50], Test Losses: mse: 24.2683, mae: 2.8731, huber: 2.4447, swd: 4.8879, ept: 143.1783
      Epoch 3 composite train-obj: 2.443005
            Val objective improved 3.1842 → 2.6390, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 17.6669, mae: 2.3543, huber: 1.9354, swd: 3.3412, ept: 155.8063
    Epoch [4/50], Val Losses: mse: 21.7523, mae: 2.5813, huber: 2.1641, swd: 4.1369, ept: 153.5168
    Epoch [4/50], Test Losses: mse: 17.5246, mae: 2.3447, huber: 1.9301, swd: 3.3295, ept: 157.4402
      Epoch 4 composite train-obj: 1.935408
            Val objective improved 2.6390 → 2.1641, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 13.5121, mae: 1.9807, huber: 1.5742, swd: 2.3560, ept: 165.9545
    Epoch [5/50], Val Losses: mse: 17.8420, mae: 2.2062, huber: 1.8023, swd: 3.3415, ept: 162.0497
    Epoch [5/50], Test Losses: mse: 12.9748, mae: 1.9541, huber: 1.5523, swd: 2.5235, ept: 167.3607
      Epoch 5 composite train-obj: 1.574240
            Val objective improved 2.1641 → 1.8023, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.1645, mae: 1.7443, huber: 1.3486, swd: 1.8569, ept: 171.6407
    Epoch [6/50], Val Losses: mse: 14.9083, mae: 1.9577, huber: 1.5651, swd: 2.5518, ept: 166.1743
    Epoch [6/50], Test Losses: mse: 10.9856, mae: 1.7484, huber: 1.3577, swd: 1.9126, ept: 171.6954
      Epoch 6 composite train-obj: 1.348594
            Val objective improved 1.8023 → 1.5651, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 9.7836, mae: 1.5915, huber: 1.2044, swd: 1.5406, ept: 174.8404
    Epoch [7/50], Val Losses: mse: 15.0269, mae: 1.9479, huber: 1.5552, swd: 2.3794, ept: 166.6544
    Epoch [7/50], Test Losses: mse: 11.2431, mae: 1.7398, huber: 1.3491, swd: 1.7188, ept: 171.6754
      Epoch 7 composite train-obj: 1.204360
            Val objective improved 1.5651 → 1.5552, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 8.8184, mae: 1.4806, huber: 1.1007, swd: 1.3488, ept: 177.1494
    Epoch [8/50], Val Losses: mse: 13.0407, mae: 1.7580, huber: 1.3759, swd: 2.3096, ept: 169.9032
    Epoch [8/50], Test Losses: mse: 9.8873, mae: 1.5821, huber: 1.2016, swd: 1.6310, ept: 174.9087
      Epoch 8 composite train-obj: 1.100704
            Val objective improved 1.5552 → 1.3759, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 8.0543, mae: 1.3950, huber: 1.0212, swd: 1.2025, ept: 178.7691
    Epoch [9/50], Val Losses: mse: 11.4996, mae: 1.6354, huber: 1.2606, swd: 1.9331, ept: 170.9852
    Epoch [9/50], Test Losses: mse: 8.3378, mae: 1.4567, huber: 1.0852, swd: 1.5481, ept: 176.2709
      Epoch 9 composite train-obj: 1.021246
            Val objective improved 1.3759 → 1.2606, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.4081, mae: 1.3168, huber: 0.9494, swd: 1.0899, ept: 180.1747
    Epoch [10/50], Val Losses: mse: 12.5916, mae: 1.6512, huber: 1.2813, swd: 2.0299, ept: 172.3046
    Epoch [10/50], Test Losses: mse: 8.4362, mae: 1.4208, huber: 1.0546, swd: 1.5457, ept: 177.4447
      Epoch 10 composite train-obj: 0.949446
            No improvement (1.2813), counter 1/5
    Epoch [11/50], Train Losses: mse: 6.9719, mae: 1.2621, huber: 0.8996, swd: 0.9927, ept: 181.0635
    Epoch [11/50], Val Losses: mse: 10.0073, mae: 1.4802, huber: 1.1152, swd: 1.6887, ept: 173.9522
    Epoch [11/50], Test Losses: mse: 7.4962, mae: 1.3208, huber: 0.9610, swd: 1.1605, ept: 178.9588
      Epoch 11 composite train-obj: 0.899598
            Val objective improved 1.2606 → 1.1152, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 6.4546, mae: 1.2041, huber: 0.8464, swd: 0.9185, ept: 182.1239
    Epoch [12/50], Val Losses: mse: 10.3745, mae: 1.4944, huber: 1.1332, swd: 1.7650, ept: 173.0823
    Epoch [12/50], Test Losses: mse: 7.2739, mae: 1.3010, huber: 0.9444, swd: 1.3108, ept: 178.6923
      Epoch 12 composite train-obj: 0.846412
            No improvement (1.1332), counter 1/5
    Epoch [13/50], Train Losses: mse: 6.1716, mae: 1.1727, huber: 0.8177, swd: 0.8516, ept: 182.8005
    Epoch [13/50], Val Losses: mse: 9.4697, mae: 1.4201, huber: 1.0623, swd: 1.7256, ept: 174.3155
    Epoch [13/50], Test Losses: mse: 7.2055, mae: 1.2667, huber: 0.9129, swd: 1.2120, ept: 179.4702
      Epoch 13 composite train-obj: 0.817708
            Val objective improved 1.1152 → 1.0623, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 5.9892, mae: 1.1444, huber: 0.7929, swd: 0.8382, ept: 183.2805
    Epoch [14/50], Val Losses: mse: 10.1780, mae: 1.4169, huber: 1.0627, swd: 1.5926, ept: 175.7278
    Epoch [14/50], Test Losses: mse: 6.7856, mae: 1.2281, huber: 0.8783, swd: 1.1542, ept: 180.8161
      Epoch 14 composite train-obj: 0.792860
            No improvement (1.0627), counter 1/5
    Epoch [15/50], Train Losses: mse: 5.7397, mae: 1.1183, huber: 0.7690, swd: 0.7930, ept: 183.5965
    Epoch [15/50], Val Losses: mse: 9.0716, mae: 1.3815, huber: 1.0252, swd: 1.6191, ept: 176.2819
    Epoch [15/50], Test Losses: mse: 7.3735, mae: 1.2553, huber: 0.9039, swd: 1.0963, ept: 180.8024
      Epoch 15 composite train-obj: 0.769043
            Val objective improved 1.0623 → 1.0252, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 5.3903, mae: 1.0760, huber: 0.7313, swd: 0.7401, ept: 184.1943
    Epoch [16/50], Val Losses: mse: 10.2723, mae: 1.4081, huber: 1.0561, swd: 1.6133, ept: 175.5719
    Epoch [16/50], Test Losses: mse: 6.9710, mae: 1.2116, huber: 0.8639, swd: 0.9695, ept: 181.1201
      Epoch 16 composite train-obj: 0.731327
            No improvement (1.0561), counter 1/5
    Epoch [17/50], Train Losses: mse: 5.2178, mae: 1.0564, huber: 0.7138, swd: 0.7054, ept: 184.6441
    Epoch [17/50], Val Losses: mse: 9.0319, mae: 1.3289, huber: 0.9825, swd: 1.4017, ept: 176.2415
    Epoch [17/50], Test Losses: mse: 6.4946, mae: 1.1666, huber: 0.8238, swd: 0.8959, ept: 181.6663
      Epoch 17 composite train-obj: 0.713814
            Val objective improved 1.0252 → 0.9825, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 5.1165, mae: 1.0434, huber: 0.7022, swd: 0.6954, ept: 184.7508
    Epoch [18/50], Val Losses: mse: 8.6934, mae: 1.3390, huber: 0.9865, swd: 1.3813, ept: 176.2965
    Epoch [18/50], Test Losses: mse: 6.8581, mae: 1.2048, huber: 0.8559, swd: 0.8944, ept: 181.2963
      Epoch 18 composite train-obj: 0.702217
            No improvement (0.9865), counter 1/5
    Epoch [19/50], Train Losses: mse: 4.9903, mae: 1.0247, huber: 0.6862, swd: 0.6697, ept: 185.1583
    Epoch [19/50], Val Losses: mse: 9.3723, mae: 1.3343, huber: 0.9842, swd: 1.2201, ept: 177.6399
    Epoch [19/50], Test Losses: mse: 6.5019, mae: 1.1723, huber: 0.8269, swd: 0.9658, ept: 181.9236
      Epoch 19 composite train-obj: 0.686157
            No improvement (0.9842), counter 2/5
    Epoch [20/50], Train Losses: mse: 4.8797, mae: 1.0120, huber: 0.6748, swd: 0.6615, ept: 185.3522
    Epoch [20/50], Val Losses: mse: 8.9901, mae: 1.2698, huber: 0.9330, swd: 1.2898, ept: 177.1928
    Epoch [20/50], Test Losses: mse: 6.4366, mae: 1.1071, huber: 0.7754, swd: 0.8463, ept: 182.8627
      Epoch 20 composite train-obj: 0.674833
            Val objective improved 0.9825 → 0.9330, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 4.6409, mae: 0.9816, huber: 0.6479, swd: 0.6235, ept: 185.8021
    Epoch [21/50], Val Losses: mse: 7.6514, mae: 1.2317, huber: 0.8903, swd: 1.3083, ept: 177.6749
    Epoch [21/50], Test Losses: mse: 5.8994, mae: 1.1000, huber: 0.7625, swd: 0.9202, ept: 182.5810
      Epoch 21 composite train-obj: 0.647877
            Val objective improved 0.9330 → 0.8903, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 4.5026, mae: 0.9642, huber: 0.6328, swd: 0.6069, ept: 186.0457
    Epoch [22/50], Val Losses: mse: 8.8511, mae: 1.2749, huber: 0.9341, swd: 1.3012, ept: 177.2903
    Epoch [22/50], Test Losses: mse: 6.1604, mae: 1.1039, huber: 0.7685, swd: 0.8437, ept: 182.6469
      Epoch 22 composite train-obj: 0.632816
            No improvement (0.9341), counter 1/5
    Epoch [23/50], Train Losses: mse: 4.4339, mae: 0.9586, huber: 0.6280, swd: 0.5885, ept: 186.1149
    Epoch [23/50], Val Losses: mse: 7.5439, mae: 1.1742, huber: 0.8414, swd: 1.1864, ept: 178.8868
    Epoch [23/50], Test Losses: mse: 5.7337, mae: 1.0439, huber: 0.7167, swd: 0.8578, ept: 183.7559
      Epoch 23 composite train-obj: 0.627963
            Val objective improved 0.8903 → 0.8414, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 4.2887, mae: 0.9405, huber: 0.6121, swd: 0.5742, ept: 186.3772
    Epoch [24/50], Val Losses: mse: 8.2015, mae: 1.2262, huber: 0.8917, swd: 1.4504, ept: 177.7891
    Epoch [24/50], Test Losses: mse: 6.1638, mae: 1.0723, huber: 0.7425, swd: 0.8042, ept: 182.6287
      Epoch 24 composite train-obj: 0.612128
            No improvement (0.8917), counter 1/5
    Epoch [25/50], Train Losses: mse: 4.1259, mae: 0.9166, huber: 0.5911, swd: 0.5557, ept: 186.7304
    Epoch [25/50], Val Losses: mse: 8.3782, mae: 1.2172, huber: 0.8850, swd: 1.2109, ept: 178.4987
    Epoch [25/50], Test Losses: mse: 5.9838, mae: 1.0620, huber: 0.7353, swd: 0.8166, ept: 183.4807
      Epoch 25 composite train-obj: 0.591126
            No improvement (0.8850), counter 2/5
    Epoch [26/50], Train Losses: mse: 4.1768, mae: 0.9232, huber: 0.5973, swd: 0.5570, ept: 186.6215
    Epoch [26/50], Val Losses: mse: 7.4742, mae: 1.1609, huber: 0.8312, swd: 1.1669, ept: 178.7435
    Epoch [26/50], Test Losses: mse: 5.7278, mae: 1.0414, huber: 0.7153, swd: 0.8241, ept: 183.2405
      Epoch 26 composite train-obj: 0.597347
            Val objective improved 0.8414 → 0.8312, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 4.0458, mae: 0.9066, huber: 0.5831, swd: 0.5386, ept: 186.9281
    Epoch [27/50], Val Losses: mse: 7.8701, mae: 1.1954, huber: 0.8624, swd: 1.1513, ept: 179.1653
    Epoch [27/50], Test Losses: mse: 6.0486, mae: 1.0561, huber: 0.7288, swd: 0.7879, ept: 183.4165
      Epoch 27 composite train-obj: 0.583082
            No improvement (0.8624), counter 1/5
    Epoch [28/50], Train Losses: mse: 3.9619, mae: 0.8949, huber: 0.5728, swd: 0.5210, ept: 187.0906
    Epoch [28/50], Val Losses: mse: 8.1056, mae: 1.2126, huber: 0.8808, swd: 1.2836, ept: 179.2293
    Epoch [28/50], Test Losses: mse: 5.8067, mae: 1.0437, huber: 0.7194, swd: 0.8104, ept: 184.0085
      Epoch 28 composite train-obj: 0.572771
            No improvement (0.8808), counter 2/5
    Epoch [29/50], Train Losses: mse: 3.8960, mae: 0.8902, huber: 0.5682, swd: 0.5178, ept: 187.1936
    Epoch [29/50], Val Losses: mse: 6.9389, mae: 1.1367, huber: 0.8057, swd: 1.3185, ept: 180.3231
    Epoch [29/50], Test Losses: mse: 5.3997, mae: 1.0008, huber: 0.6755, swd: 0.7240, ept: 185.2788
      Epoch 29 composite train-obj: 0.568248
            Val objective improved 0.8312 → 0.8057, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 3.9815, mae: 0.8932, huber: 0.5719, swd: 0.5157, ept: 187.1978
    Epoch [30/50], Val Losses: mse: 8.8625, mae: 1.2558, huber: 0.9207, swd: 1.2981, ept: 178.2725
    Epoch [30/50], Test Losses: mse: 6.3382, mae: 1.0970, huber: 0.7678, swd: 0.8794, ept: 183.2654
      Epoch 30 composite train-obj: 0.571937
            No improvement (0.9207), counter 1/5
    Epoch [31/50], Train Losses: mse: 3.6965, mae: 0.8635, huber: 0.5454, swd: 0.4876, ept: 187.6174
    Epoch [31/50], Val Losses: mse: 8.0759, mae: 1.1990, huber: 0.8619, swd: 1.2151, ept: 179.5015
    Epoch [31/50], Test Losses: mse: 5.7060, mae: 1.0509, huber: 0.7189, swd: 0.8477, ept: 184.3329
      Epoch 31 composite train-obj: 0.545387
            No improvement (0.8619), counter 2/5
    Epoch [32/50], Train Losses: mse: 3.6891, mae: 0.8603, huber: 0.5430, swd: 0.4904, ept: 187.6515
    Epoch [32/50], Val Losses: mse: 7.4211, mae: 1.1417, huber: 0.8154, swd: 1.2549, ept: 179.7193
    Epoch [32/50], Test Losses: mse: 5.3744, mae: 0.9857, huber: 0.6651, swd: 0.6717, ept: 184.9927
      Epoch 32 composite train-obj: 0.543033
            No improvement (0.8154), counter 3/5
    Epoch [33/50], Train Losses: mse: 3.6476, mae: 0.8530, huber: 0.5368, swd: 0.4718, ept: 187.6453
    Epoch [33/50], Val Losses: mse: 7.1290, mae: 1.1226, huber: 0.7953, swd: 1.2047, ept: 180.2823
    Epoch [33/50], Test Losses: mse: 5.4027, mae: 0.9951, huber: 0.6721, swd: 0.7655, ept: 184.7471
      Epoch 33 composite train-obj: 0.536824
            Val objective improved 0.8057 → 0.7953, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 3.5961, mae: 0.8475, huber: 0.5324, swd: 0.4679, ept: 187.8421
    Epoch [34/50], Val Losses: mse: 7.0163, mae: 1.1161, huber: 0.7907, swd: 1.0397, ept: 179.7969
    Epoch [34/50], Test Losses: mse: 5.3373, mae: 0.9931, huber: 0.6724, swd: 0.6783, ept: 184.4868
      Epoch 34 composite train-obj: 0.532379
            Val objective improved 0.7953 → 0.7907, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 3.4822, mae: 0.8328, huber: 0.5196, swd: 0.4529, ept: 188.0088
    Epoch [35/50], Val Losses: mse: 7.3816, mae: 1.1541, huber: 0.8239, swd: 1.1836, ept: 179.6289
    Epoch [35/50], Test Losses: mse: 5.4493, mae: 1.0256, huber: 0.6999, swd: 0.8500, ept: 184.1857
      Epoch 35 composite train-obj: 0.519588
            No improvement (0.8239), counter 1/5
    Epoch [36/50], Train Losses: mse: 3.5559, mae: 0.8430, huber: 0.5286, swd: 0.4664, ept: 187.9601
    Epoch [36/50], Val Losses: mse: 6.7390, mae: 1.0873, huber: 0.7658, swd: 1.0161, ept: 181.1255
    Epoch [36/50], Test Losses: mse: 5.0360, mae: 0.9610, huber: 0.6446, swd: 0.6742, ept: 184.8925
      Epoch 36 composite train-obj: 0.528618
            Val objective improved 0.7907 → 0.7658, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 3.4006, mae: 0.8224, huber: 0.5110, swd: 0.4431, ept: 188.1732
    Epoch [37/50], Val Losses: mse: 7.7680, mae: 1.1637, huber: 0.8399, swd: 1.1541, ept: 179.2471
    Epoch [37/50], Test Losses: mse: 6.2847, mae: 1.0463, huber: 0.7260, swd: 0.6746, ept: 183.9488
      Epoch 37 composite train-obj: 0.510955
            No improvement (0.8399), counter 1/5
    Epoch [38/50], Train Losses: mse: 3.4569, mae: 0.8304, huber: 0.5177, swd: 0.4510, ept: 188.1576
    Epoch [38/50], Val Losses: mse: 7.1668, mae: 1.1265, huber: 0.8011, swd: 1.0428, ept: 180.0895
    Epoch [38/50], Test Losses: mse: 5.6880, mae: 1.0198, huber: 0.6994, swd: 0.7179, ept: 184.0909
      Epoch 38 composite train-obj: 0.517721
            No improvement (0.8011), counter 2/5
    Epoch [39/50], Train Losses: mse: 3.3028, mae: 0.8125, huber: 0.5020, swd: 0.4347, ept: 188.4537
    Epoch [39/50], Val Losses: mse: 7.6951, mae: 1.1476, huber: 0.8221, swd: 1.1315, ept: 179.1948
    Epoch [39/50], Test Losses: mse: 5.6765, mae: 0.9924, huber: 0.6736, swd: 0.6728, ept: 184.4803
      Epoch 39 composite train-obj: 0.502012
            No improvement (0.8221), counter 3/5
    Epoch [40/50], Train Losses: mse: 3.2776, mae: 0.8058, huber: 0.4969, swd: 0.4314, ept: 188.4362
    Epoch [40/50], Val Losses: mse: 7.6790, mae: 1.1750, huber: 0.8412, swd: 0.9929, ept: 179.1653
    Epoch [40/50], Test Losses: mse: 6.3609, mae: 1.0831, huber: 0.7532, swd: 0.7933, ept: 183.0923
      Epoch 40 composite train-obj: 0.496855
            No improvement (0.8412), counter 4/5
    Epoch [41/50], Train Losses: mse: 3.2487, mae: 0.8043, huber: 0.4955, swd: 0.4176, ept: 188.4970
    Epoch [41/50], Val Losses: mse: 6.8896, mae: 1.0774, huber: 0.7583, swd: 0.9474, ept: 182.0189
    Epoch [41/50], Test Losses: mse: 5.9585, mae: 0.9996, huber: 0.6858, swd: 0.6524, ept: 185.0462
      Epoch 41 composite train-obj: 0.495547
            Val objective improved 0.7658 → 0.7583, saving checkpoint.
    Epoch [42/50], Train Losses: mse: 3.2428, mae: 0.8005, huber: 0.4924, swd: 0.4217, ept: 188.5719
    Epoch [42/50], Val Losses: mse: 6.9999, mae: 1.0883, huber: 0.7702, swd: 0.9217, ept: 181.2102
    Epoch [42/50], Test Losses: mse: 5.4496, mae: 0.9737, huber: 0.6587, swd: 0.6556, ept: 185.5324
      Epoch 42 composite train-obj: 0.492353
            No improvement (0.7702), counter 1/5
    Epoch [43/50], Train Losses: mse: 3.2678, mae: 0.7999, huber: 0.4922, swd: 0.4197, ept: 188.6157
    Epoch [43/50], Val Losses: mse: 7.3178, mae: 1.1084, huber: 0.7909, swd: 1.0367, ept: 181.3805
    Epoch [43/50], Test Losses: mse: 5.3492, mae: 0.9784, huber: 0.6641, swd: 0.6610, ept: 184.1045
      Epoch 43 composite train-obj: 0.492173
            No improvement (0.7909), counter 2/5
    Epoch [44/50], Train Losses: mse: 3.0227, mae: 0.7749, huber: 0.4699, swd: 0.3940, ept: 188.9766
    Epoch [44/50], Val Losses: mse: 7.2193, mae: 1.0988, huber: 0.7786, swd: 0.9196, ept: 181.3676
    Epoch [44/50], Test Losses: mse: 5.7883, mae: 0.9964, huber: 0.6808, swd: 0.6929, ept: 184.7691
      Epoch 44 composite train-obj: 0.469889
            No improvement (0.7786), counter 3/5
    Epoch [45/50], Train Losses: mse: 3.0465, mae: 0.7747, huber: 0.4703, swd: 0.3938, ept: 189.0644
    Epoch [45/50], Val Losses: mse: 6.3613, mae: 1.0539, huber: 0.7333, swd: 0.9285, ept: 181.7373
    Epoch [45/50], Test Losses: mse: 6.0839, mae: 1.0009, huber: 0.6830, swd: 0.6440, ept: 185.0379
      Epoch 45 composite train-obj: 0.470263
            Val objective improved 0.7583 → 0.7333, saving checkpoint.
    Epoch [46/50], Train Losses: mse: 3.0372, mae: 0.7756, huber: 0.4709, swd: 0.3908, ept: 189.0575
    Epoch [46/50], Val Losses: mse: 6.4275, mae: 1.0291, huber: 0.7143, swd: 0.9128, ept: 182.7278
    Epoch [46/50], Test Losses: mse: 5.0161, mae: 0.9261, huber: 0.6158, swd: 0.6070, ept: 185.8568
      Epoch 46 composite train-obj: 0.470883
            Val objective improved 0.7333 → 0.7143, saving checkpoint.
    Epoch [47/50], Train Losses: mse: 3.0560, mae: 0.7775, huber: 0.4729, swd: 0.3955, ept: 189.0040
    Epoch [47/50], Val Losses: mse: 6.2544, mae: 1.0960, huber: 0.7618, swd: 1.0256, ept: 181.0176
    Epoch [47/50], Test Losses: mse: 5.3756, mae: 1.0117, huber: 0.6821, swd: 0.8017, ept: 184.8377
      Epoch 47 composite train-obj: 0.472949
            No improvement (0.7618), counter 1/5
    Epoch [48/50], Train Losses: mse: 2.9113, mae: 0.7569, huber: 0.4553, swd: 0.3765, ept: 189.2555
    Epoch [48/50], Val Losses: mse: 5.8692, mae: 1.0505, huber: 0.7170, swd: 1.1087, ept: 183.1606
    Epoch [48/50], Test Losses: mse: 5.2807, mae: 0.9756, huber: 0.6451, swd: 0.6829, ept: 186.4291
      Epoch 48 composite train-obj: 0.455275
            No improvement (0.7170), counter 2/5
    Epoch [49/50], Train Losses: mse: 2.9381, mae: 0.7624, huber: 0.4597, swd: 0.3771, ept: 189.2103
    Epoch [49/50], Val Losses: mse: 6.7701, mae: 1.0782, huber: 0.7589, swd: 1.0082, ept: 181.8428
    Epoch [49/50], Test Losses: mse: 5.9490, mae: 0.9943, huber: 0.6795, swd: 0.6314, ept: 185.2591
      Epoch 49 composite train-obj: 0.459673
            No improvement (0.7589), counter 3/5
    Epoch [50/50], Train Losses: mse: 2.8897, mae: 0.7571, huber: 0.4553, swd: 0.3722, ept: 189.2795
    Epoch [50/50], Val Losses: mse: 6.5662, mae: 1.0588, huber: 0.7390, swd: 1.0704, ept: 181.6751
    Epoch [50/50], Test Losses: mse: 5.2749, mae: 0.9490, huber: 0.6336, swd: 0.6864, ept: 185.8738
      Epoch 50 composite train-obj: 0.455326
            No improvement (0.7390), counter 4/5
    Epoch [50/50], Test Losses: mse: 5.0161, mae: 0.9261, huber: 0.6158, swd: 0.6070, ept: 185.8568
    Best round's Test MSE: 5.0161, MAE: 0.9261, SWD: 0.6070
    Best round's Validation MSE: 6.4275, MAE: 1.0291, SWD: 0.9128
    Best round's Test verification MSE : 5.0161, MAE: 0.9261, SWD: 0.6070
    Time taken: 437.94 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 52.4796, mae: 4.9929, huber: 4.5262, swd: 12.8627, ept: 77.5025
    Epoch [1/50], Val Losses: mse: 43.3615, mae: 4.3184, huber: 3.8611, swd: 10.5581, ept: 105.7250
    Epoch [1/50], Test Losses: mse: 41.5039, mae: 4.1655, huber: 3.7101, swd: 10.8529, ept: 106.0015
      Epoch 1 composite train-obj: 4.526186
            Val objective improved inf → 3.8611, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.8896, mae: 3.4565, huber: 3.0127, swd: 7.4124, ept: 125.1501
    Epoch [2/50], Val Losses: mse: 34.5961, mae: 3.5472, huber: 3.1086, swd: 6.4287, ept: 127.6968
    Epoch [2/50], Test Losses: mse: 30.7467, mae: 3.3430, huber: 2.9056, swd: 6.0609, ept: 128.3105
      Epoch 2 composite train-obj: 3.012664
            Val objective improved 3.8611 → 3.1086, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.8933, mae: 2.7664, huber: 2.3368, swd: 4.4558, ept: 144.8900
    Epoch [3/50], Val Losses: mse: 27.3433, mae: 2.9517, huber: 2.5244, swd: 4.0598, ept: 146.0658
    Epoch [3/50], Test Losses: mse: 22.8199, mae: 2.7370, huber: 2.3103, swd: 3.7099, ept: 149.7276
      Epoch 3 composite train-obj: 2.336813
            Val objective improved 3.1086 → 2.5244, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 16.0797, mae: 2.2238, huber: 1.8086, swd: 2.8911, ept: 160.0434
    Epoch [4/50], Val Losses: mse: 20.3834, mae: 2.4482, huber: 2.0324, swd: 3.3330, ept: 156.0208
    Epoch [4/50], Test Losses: mse: 15.5694, mae: 2.1868, huber: 1.7735, swd: 2.6979, ept: 163.2257
      Epoch 4 composite train-obj: 1.808625
            Val objective improved 2.5244 → 2.0324, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 12.5434, mae: 1.8930, huber: 1.4902, swd: 2.1793, ept: 168.5790
    Epoch [5/50], Val Losses: mse: 17.1175, mae: 2.1917, huber: 1.7855, swd: 3.0553, ept: 161.5758
    Epoch [5/50], Test Losses: mse: 13.5164, mae: 1.9750, huber: 1.5712, swd: 2.1875, ept: 168.3847
      Epoch 5 composite train-obj: 1.490155
            Val objective improved 2.0324 → 1.7855, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.4882, mae: 1.6724, huber: 1.2809, swd: 1.7263, ept: 173.2189
    Epoch [6/50], Val Losses: mse: 15.0316, mae: 1.9520, huber: 1.5577, swd: 2.3882, ept: 167.0096
    Epoch [6/50], Test Losses: mse: 11.0099, mae: 1.7313, huber: 1.3390, swd: 1.7696, ept: 173.1027
      Epoch 6 composite train-obj: 1.280919
            Val objective improved 1.7855 → 1.5577, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 9.3332, mae: 1.5373, huber: 1.1544, swd: 1.4641, ept: 176.0459
    Epoch [7/50], Val Losses: mse: 14.4368, mae: 1.8327, huber: 1.4472, swd: 1.9470, ept: 169.8497
    Epoch [7/50], Test Losses: mse: 10.0419, mae: 1.6148, huber: 1.2309, swd: 1.5126, ept: 174.9264
      Epoch 7 composite train-obj: 1.154376
            Val objective improved 1.5577 → 1.4472, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 8.4947, mae: 1.4368, huber: 1.0610, swd: 1.2811, ept: 178.0597
    Epoch [8/50], Val Losses: mse: 12.8212, mae: 1.7236, huber: 1.3467, swd: 2.2794, ept: 170.1392
    Epoch [8/50], Test Losses: mse: 9.0683, mae: 1.4885, huber: 1.1161, swd: 1.3268, ept: 176.2350
      Epoch 8 composite train-obj: 1.060965
            Val objective improved 1.4472 → 1.3467, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.7591, mae: 1.3539, huber: 0.9844, swd: 1.1476, ept: 179.4652
    Epoch [9/50], Val Losses: mse: 12.3080, mae: 1.6233, huber: 1.2548, swd: 1.8606, ept: 172.2696
    Epoch [9/50], Test Losses: mse: 8.7738, mae: 1.4148, huber: 1.0514, swd: 1.2441, ept: 177.3227
      Epoch 9 composite train-obj: 0.984372
            Val objective improved 1.3467 → 1.2548, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.3886, mae: 1.3009, huber: 0.9363, swd: 1.0516, ept: 180.4212
    Epoch [10/50], Val Losses: mse: 10.6501, mae: 1.4948, huber: 1.1310, swd: 1.4950, ept: 174.5441
    Epoch [10/50], Test Losses: mse: 7.7515, mae: 1.3224, huber: 0.9621, swd: 1.0405, ept: 180.1136
      Epoch 10 composite train-obj: 0.936289
            Val objective improved 1.2548 → 1.1310, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 6.7614, mae: 1.2273, huber: 0.8690, swd: 0.9349, ept: 181.6222
    Epoch [11/50], Val Losses: mse: 11.1937, mae: 1.5351, huber: 1.1721, swd: 1.7483, ept: 173.4444
    Epoch [11/50], Test Losses: mse: 7.6087, mae: 1.3143, huber: 0.9556, swd: 1.0464, ept: 179.1612
      Epoch 11 composite train-obj: 0.869039
            No improvement (1.1721), counter 1/5
    Epoch [12/50], Train Losses: mse: 6.4957, mae: 1.1940, huber: 0.8386, swd: 0.8804, ept: 182.3695
    Epoch [12/50], Val Losses: mse: 8.8506, mae: 1.3752, huber: 1.0183, swd: 1.4291, ept: 175.8826
    Epoch [12/50], Test Losses: mse: 6.9464, mae: 1.2316, huber: 0.8799, swd: 0.9636, ept: 181.0359
      Epoch 12 composite train-obj: 0.838633
            Val objective improved 1.1310 → 1.0183, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 6.1623, mae: 1.1554, huber: 0.8040, swd: 0.8378, ept: 183.0570
    Epoch [13/50], Val Losses: mse: 9.7487, mae: 1.4204, huber: 1.0613, swd: 1.4489, ept: 176.2204
    Epoch [13/50], Test Losses: mse: 7.1774, mae: 1.2575, huber: 0.9033, swd: 1.0111, ept: 180.6801
      Epoch 13 composite train-obj: 0.803971
            No improvement (1.0613), counter 1/5
    Epoch [14/50], Train Losses: mse: 5.9700, mae: 1.1302, huber: 0.7813, swd: 0.7827, ept: 183.5101
    Epoch [14/50], Val Losses: mse: 9.7401, mae: 1.4214, huber: 1.0635, swd: 1.3631, ept: 176.3409
    Epoch [14/50], Test Losses: mse: 7.1818, mae: 1.2627, huber: 0.9086, swd: 0.9341, ept: 180.5349
      Epoch 14 composite train-obj: 0.781340
            No improvement (1.0635), counter 2/5
    Epoch [15/50], Train Losses: mse: 5.6015, mae: 1.0886, huber: 0.7442, swd: 0.7414, ept: 183.9910
    Epoch [15/50], Val Losses: mse: 10.1323, mae: 1.3963, huber: 1.0451, swd: 1.2753, ept: 176.5017
    Epoch [15/50], Test Losses: mse: 6.7562, mae: 1.1916, huber: 0.8456, swd: 0.8402, ept: 180.9361
      Epoch 15 composite train-obj: 0.744229
            No improvement (1.0451), counter 3/5
    Epoch [16/50], Train Losses: mse: 5.4668, mae: 1.0693, huber: 0.7269, swd: 0.6993, ept: 184.6306
    Epoch [16/50], Val Losses: mse: 9.6012, mae: 1.3607, huber: 1.0134, swd: 1.4190, ept: 176.4193
    Epoch [16/50], Test Losses: mse: 6.6364, mae: 1.1800, huber: 0.8361, swd: 0.9106, ept: 181.5048
      Epoch 16 composite train-obj: 0.726861
            Val objective improved 1.0183 → 1.0134, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 5.2024, mae: 1.0406, huber: 0.7011, swd: 0.6608, ept: 184.9417
    Epoch [17/50], Val Losses: mse: 9.8605, mae: 1.3668, huber: 1.0183, swd: 1.4479, ept: 177.6467
    Epoch [17/50], Test Losses: mse: 6.7847, mae: 1.1670, huber: 0.8231, swd: 0.7799, ept: 182.9998
      Epoch 17 composite train-obj: 0.701136
            No improvement (1.0183), counter 1/5
    Epoch [18/50], Train Losses: mse: 5.2084, mae: 1.0366, huber: 0.6983, swd: 0.6621, ept: 185.0772
    Epoch [18/50], Val Losses: mse: 9.3144, mae: 1.3019, huber: 0.9572, swd: 1.2655, ept: 177.9329
    Epoch [18/50], Test Losses: mse: 6.5781, mae: 1.1462, huber: 0.8060, swd: 0.8417, ept: 182.2705
      Epoch 18 composite train-obj: 0.698268
            Val objective improved 1.0134 → 0.9572, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 5.0081, mae: 1.0141, huber: 0.6782, swd: 0.6366, ept: 185.3658
    Epoch [19/50], Val Losses: mse: 10.2273, mae: 1.3920, huber: 1.0367, swd: 1.0936, ept: 178.0111
    Epoch [19/50], Test Losses: mse: 6.6356, mae: 1.1924, huber: 0.8428, swd: 0.8514, ept: 182.2501
      Epoch 19 composite train-obj: 0.678204
            No improvement (1.0367), counter 1/5
    Epoch [20/50], Train Losses: mse: 4.8725, mae: 0.9975, huber: 0.6636, swd: 0.6126, ept: 185.6432
    Epoch [20/50], Val Losses: mse: 9.9112, mae: 1.3586, huber: 1.0086, swd: 1.1639, ept: 177.8668
    Epoch [20/50], Test Losses: mse: 7.2225, mae: 1.1941, huber: 0.8482, swd: 0.7085, ept: 182.2592
      Epoch 20 composite train-obj: 0.663567
            No improvement (1.0086), counter 2/5
    Epoch [21/50], Train Losses: mse: 4.7419, mae: 0.9790, huber: 0.6472, swd: 0.5878, ept: 186.0195
    Epoch [21/50], Val Losses: mse: 8.3606, mae: 1.2512, huber: 0.9096, swd: 1.0440, ept: 179.7078
    Epoch [21/50], Test Losses: mse: 6.0678, mae: 1.1053, huber: 0.7687, swd: 0.6261, ept: 183.3794
      Epoch 21 composite train-obj: 0.647193
            Val objective improved 0.9572 → 0.9096, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 4.5844, mae: 0.9660, huber: 0.6353, swd: 0.5631, ept: 186.1717
    Epoch [22/50], Val Losses: mse: 8.2209, mae: 1.2303, huber: 0.8943, swd: 1.1562, ept: 178.2718
    Epoch [22/50], Test Losses: mse: 5.7573, mae: 1.0630, huber: 0.7320, swd: 0.7061, ept: 183.3862
      Epoch 22 composite train-obj: 0.635342
            Val objective improved 0.9096 → 0.8943, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 4.4462, mae: 0.9465, huber: 0.6186, swd: 0.5417, ept: 186.4966
    Epoch [23/50], Val Losses: mse: 8.2999, mae: 1.2114, huber: 0.8786, swd: 1.0864, ept: 180.1593
    Epoch [23/50], Test Losses: mse: 5.8941, mae: 1.0551, huber: 0.7270, swd: 0.6890, ept: 183.5485
      Epoch 23 composite train-obj: 0.618593
            Val objective improved 0.8943 → 0.8786, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 4.3401, mae: 0.9329, huber: 0.6067, swd: 0.5290, ept: 186.8301
    Epoch [24/50], Val Losses: mse: 8.8139, mae: 1.2132, huber: 0.8812, swd: 0.9572, ept: 179.9968
    Epoch [24/50], Test Losses: mse: 5.8248, mae: 1.0345, huber: 0.7087, swd: 0.6580, ept: 183.9668
      Epoch 24 composite train-obj: 0.606745
            No improvement (0.8812), counter 1/5
    Epoch [25/50], Train Losses: mse: 4.2431, mae: 0.9170, huber: 0.5935, swd: 0.5177, ept: 186.8677
    Epoch [25/50], Val Losses: mse: 9.1153, mae: 1.2553, huber: 0.9177, swd: 1.0089, ept: 180.4375
    Epoch [25/50], Test Losses: mse: 6.1679, mae: 1.0804, huber: 0.7489, swd: 0.6618, ept: 183.1909
      Epoch 25 composite train-obj: 0.593473
            No improvement (0.9177), counter 2/5
    Epoch [26/50], Train Losses: mse: 4.1575, mae: 0.9087, huber: 0.5862, swd: 0.5059, ept: 187.0009
    Epoch [26/50], Val Losses: mse: 9.6723, mae: 1.2698, huber: 0.9335, swd: 0.9458, ept: 180.2394
    Epoch [26/50], Test Losses: mse: 7.0734, mae: 1.1258, huber: 0.7929, swd: 0.7361, ept: 183.3491
      Epoch 26 composite train-obj: 0.586176
            No improvement (0.9335), counter 3/5
    Epoch [27/50], Train Losses: mse: 4.1023, mae: 0.9026, huber: 0.5805, swd: 0.4929, ept: 187.2117
    Epoch [27/50], Val Losses: mse: 7.3472, mae: 1.1561, huber: 0.8220, swd: 0.9286, ept: 180.7537
    Epoch [27/50], Test Losses: mse: 5.5806, mae: 1.0329, huber: 0.7032, swd: 0.6540, ept: 184.2624
      Epoch 27 composite train-obj: 0.580498
            Val objective improved 0.8786 → 0.8220, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 4.0093, mae: 0.8899, huber: 0.5698, swd: 0.4808, ept: 187.3458
    Epoch [28/50], Val Losses: mse: 8.8912, mae: 1.1842, huber: 0.8586, swd: 0.9115, ept: 181.2444
    Epoch [28/50], Test Losses: mse: 5.5810, mae: 0.9976, huber: 0.6768, swd: 0.5989, ept: 184.6007
      Epoch 28 composite train-obj: 0.569778
            No improvement (0.8586), counter 1/5
    Epoch [29/50], Train Losses: mse: 3.9145, mae: 0.8811, huber: 0.5617, swd: 0.4698, ept: 187.5302
    Epoch [29/50], Val Losses: mse: 9.3905, mae: 1.2496, huber: 0.9220, swd: 0.9656, ept: 179.5387
    Epoch [29/50], Test Losses: mse: 6.8485, mae: 1.0954, huber: 0.7716, swd: 0.6546, ept: 182.9061
      Epoch 29 composite train-obj: 0.561659
            No improvement (0.9220), counter 2/5
    Epoch [30/50], Train Losses: mse: 3.9123, mae: 0.8779, huber: 0.5592, swd: 0.4629, ept: 187.6888
    Epoch [30/50], Val Losses: mse: 9.2618, mae: 1.2527, huber: 0.9183, swd: 0.9928, ept: 179.3866
    Epoch [30/50], Test Losses: mse: 6.3987, mae: 1.0812, huber: 0.7519, swd: 0.6040, ept: 183.4860
      Epoch 30 composite train-obj: 0.559160
            No improvement (0.9183), counter 3/5
    Epoch [31/50], Train Losses: mse: 3.9357, mae: 0.8828, huber: 0.5639, swd: 0.4744, ept: 187.5743
    Epoch [31/50], Val Losses: mse: 9.1585, mae: 1.2299, huber: 0.8973, swd: 0.9820, ept: 180.1571
    Epoch [31/50], Test Losses: mse: 6.3604, mae: 1.0448, huber: 0.7180, swd: 0.5919, ept: 184.1124
      Epoch 31 composite train-obj: 0.563882
            No improvement (0.8973), counter 4/5
    Epoch [32/50], Train Losses: mse: 3.7369, mae: 0.8587, huber: 0.5422, swd: 0.4365, ept: 187.9639
    Epoch [32/50], Val Losses: mse: 7.7696, mae: 1.1557, huber: 0.8261, swd: 0.8325, ept: 180.9838
    Epoch [32/50], Test Losses: mse: 6.0319, mae: 1.0366, huber: 0.7122, swd: 0.5737, ept: 184.6220
      Epoch 32 composite train-obj: 0.542227
    Epoch [32/50], Test Losses: mse: 5.5806, mae: 1.0329, huber: 0.7032, swd: 0.6540, ept: 184.2624
    Best round's Test MSE: 5.5806, MAE: 1.0329, SWD: 0.6540
    Best round's Validation MSE: 7.3472, MAE: 1.1561, SWD: 0.9286
    Best round's Test verification MSE : 5.5806, MAE: 1.0329, SWD: 0.6540
    Time taken: 292.88 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 53.5273, mae: 5.0665, huber: 4.5988, swd: 12.7341, ept: 76.9422
    Epoch [1/50], Val Losses: mse: 42.6956, mae: 4.2774, huber: 3.8228, swd: 10.5430, ept: 105.5433
    Epoch [1/50], Test Losses: mse: 41.1384, mae: 4.1563, huber: 3.7032, swd: 10.9157, ept: 104.6401
      Epoch 1 composite train-obj: 4.598805
            Val objective improved inf → 3.8228, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 32.2548, mae: 3.4748, huber: 3.0308, swd: 7.6523, ept: 124.9410
    Epoch [2/50], Val Losses: mse: 35.6086, mae: 3.6042, huber: 3.1648, swd: 6.9107, ept: 127.6048
    Epoch [2/50], Test Losses: mse: 32.8063, mae: 3.4437, huber: 3.0057, swd: 6.9275, ept: 126.9387
      Epoch 2 composite train-obj: 3.030836
            Val objective improved 3.8228 → 3.1648, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 24.1622, mae: 2.8298, huber: 2.4006, swd: 4.8435, ept: 143.3158
    Epoch [3/50], Val Losses: mse: 26.8722, mae: 2.9497, huber: 2.5246, swd: 4.8572, ept: 145.2698
    Epoch [3/50], Test Losses: mse: 23.7398, mae: 2.7792, huber: 2.3560, swd: 4.0050, ept: 146.4046
      Epoch 3 composite train-obj: 2.400575
            Val objective improved 3.1648 → 2.5246, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 17.3727, mae: 2.2954, huber: 1.8801, swd: 3.1918, ept: 158.4014
    Epoch [4/50], Val Losses: mse: 21.2570, mae: 2.5032, huber: 2.0895, swd: 3.7358, ept: 156.9098
    Epoch [4/50], Test Losses: mse: 17.2826, mae: 2.3099, huber: 1.8966, swd: 3.5235, ept: 160.2198
      Epoch 4 composite train-obj: 1.880086
            Val objective improved 2.5246 → 2.0895, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 12.9696, mae: 1.9136, huber: 1.5116, swd: 2.3983, ept: 168.0741
    Epoch [5/50], Val Losses: mse: 17.5507, mae: 2.1899, huber: 1.7888, swd: 3.3810, ept: 161.8029
    Epoch [5/50], Test Losses: mse: 13.5477, mae: 1.9652, huber: 1.5668, swd: 2.7215, ept: 168.2752
      Epoch 5 composite train-obj: 1.511620
            Val objective improved 2.0895 → 1.7888, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.6042, mae: 1.6653, huber: 1.2760, swd: 1.9190, ept: 173.3332
    Epoch [6/50], Val Losses: mse: 14.4709, mae: 1.9183, huber: 1.5295, swd: 2.8612, ept: 166.2369
    Epoch [6/50], Test Losses: mse: 10.4311, mae: 1.6836, huber: 1.2976, swd: 2.1842, ept: 172.4088
      Epoch 6 composite train-obj: 1.275965
            Val objective improved 1.7888 → 1.5295, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 9.3469, mae: 1.5182, huber: 1.1384, swd: 1.6218, ept: 175.9879
    Epoch [7/50], Val Losses: mse: 12.0152, mae: 1.6804, huber: 1.3076, swd: 2.4326, ept: 169.4777
    Epoch [7/50], Test Losses: mse: 9.0874, mae: 1.5077, huber: 1.1379, swd: 1.8638, ept: 175.5958
      Epoch 7 composite train-obj: 1.138448
            Val objective improved 1.5295 → 1.3076, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 8.5914, mae: 1.4288, huber: 1.0552, swd: 1.4388, ept: 177.6298
    Epoch [8/50], Val Losses: mse: 13.7204, mae: 1.7763, huber: 1.4010, swd: 2.4055, ept: 170.2504
    Epoch [8/50], Test Losses: mse: 9.3574, mae: 1.5302, huber: 1.1574, swd: 1.8606, ept: 176.2533
      Epoch 8 composite train-obj: 1.055213
            No improvement (1.4010), counter 1/5
    Epoch [9/50], Train Losses: mse: 7.9650, mae: 1.3539, huber: 0.9863, swd: 1.2861, ept: 178.9134
    Epoch [9/50], Val Losses: mse: 10.6516, mae: 1.5472, huber: 1.1820, swd: 2.2476, ept: 172.4474
    Epoch [9/50], Test Losses: mse: 7.9922, mae: 1.3770, huber: 1.0155, swd: 1.4928, ept: 177.9444
      Epoch 9 composite train-obj: 0.986310
            Val objective improved 1.3076 → 1.1820, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.4319, mae: 1.2901, huber: 0.9275, swd: 1.1537, ept: 180.1316
    Epoch [10/50], Val Losses: mse: 10.9838, mae: 1.5564, huber: 1.1918, swd: 1.8771, ept: 171.6651
    Epoch [10/50], Test Losses: mse: 8.3584, mae: 1.4018, huber: 1.0398, swd: 1.5292, ept: 177.6604
      Epoch 10 composite train-obj: 0.927547
            No improvement (1.1918), counter 1/5
    Epoch [11/50], Train Losses: mse: 7.0392, mae: 1.2406, huber: 0.8829, swd: 1.0658, ept: 180.9250
    Epoch [11/50], Val Losses: mse: 10.3868, mae: 1.4699, huber: 1.1117, swd: 1.6025, ept: 173.7417
    Epoch [11/50], Test Losses: mse: 7.4760, mae: 1.2955, huber: 0.9398, swd: 1.3130, ept: 180.0724
      Epoch 11 composite train-obj: 0.882899
            Val objective improved 1.1820 → 1.1117, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 6.6760, mae: 1.1988, huber: 0.8444, swd: 0.9816, ept: 181.7248
    Epoch [12/50], Val Losses: mse: 11.1363, mae: 1.4998, huber: 1.1427, swd: 1.9677, ept: 173.6521
    Epoch [12/50], Test Losses: mse: 7.7428, mae: 1.2912, huber: 0.9376, swd: 1.3345, ept: 179.8352
      Epoch 12 composite train-obj: 0.844445
            No improvement (1.1427), counter 1/5
    Epoch [13/50], Train Losses: mse: 6.3224, mae: 1.1600, huber: 0.8092, swd: 0.9055, ept: 182.4074
    Epoch [13/50], Val Losses: mse: 9.2212, mae: 1.3635, huber: 1.0133, swd: 1.6774, ept: 174.7841
    Epoch [13/50], Test Losses: mse: 6.4240, mae: 1.1959, huber: 0.8474, swd: 1.1031, ept: 180.9173
      Epoch 13 composite train-obj: 0.809245
            Val objective improved 1.1117 → 1.0133, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 6.1744, mae: 1.1418, huber: 0.7926, swd: 0.8583, ept: 182.8458
    Epoch [14/50], Val Losses: mse: 9.9914, mae: 1.4074, huber: 1.0547, swd: 1.6067, ept: 175.1241
    Epoch [14/50], Test Losses: mse: 6.7691, mae: 1.2148, huber: 0.8652, swd: 1.1248, ept: 180.4616
      Epoch 14 composite train-obj: 0.792553
            No improvement (1.0547), counter 1/5
    Epoch [15/50], Train Losses: mse: 5.9287, mae: 1.1141, huber: 0.7676, swd: 0.8140, ept: 183.2251
    Epoch [15/50], Val Losses: mse: 9.0472, mae: 1.3645, huber: 1.0148, swd: 1.5380, ept: 176.2418
    Epoch [15/50], Test Losses: mse: 6.9284, mae: 1.2152, huber: 0.8703, swd: 1.0292, ept: 180.9352
      Epoch 15 composite train-obj: 0.767621
            No improvement (1.0148), counter 2/5
    Epoch [16/50], Train Losses: mse: 5.5863, mae: 1.0738, huber: 0.7318, swd: 0.7552, ept: 183.8225
    Epoch [16/50], Val Losses: mse: 7.9466, mae: 1.2736, huber: 0.9258, swd: 1.3081, ept: 176.8988
    Epoch [16/50], Test Losses: mse: 6.1666, mae: 1.1495, huber: 0.8044, swd: 0.9195, ept: 182.0508
      Epoch 16 composite train-obj: 0.731768
            Val objective improved 1.0133 → 0.9258, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 5.5118, mae: 1.0607, huber: 0.7203, swd: 0.7228, ept: 184.1836
    Epoch [17/50], Val Losses: mse: 7.7777, mae: 1.2324, huber: 0.8935, swd: 1.2269, ept: 177.2217
    Epoch [17/50], Test Losses: mse: 6.2043, mae: 1.1199, huber: 0.7842, swd: 0.9597, ept: 181.7691
      Epoch 17 composite train-obj: 0.720345
            Val objective improved 0.9258 → 0.8935, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 5.3166, mae: 1.0393, huber: 0.7013, swd: 0.7049, ept: 184.5841
    Epoch [18/50], Val Losses: mse: 7.3361, mae: 1.2063, huber: 0.8668, swd: 1.1853, ept: 177.3825
    Epoch [18/50], Test Losses: mse: 6.1105, mae: 1.1061, huber: 0.7701, swd: 0.9205, ept: 182.2769
      Epoch 18 composite train-obj: 0.701260
            Val objective improved 0.8935 → 0.8668, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 5.1657, mae: 1.0165, huber: 0.6812, swd: 0.6741, ept: 184.8956
    Epoch [19/50], Val Losses: mse: 8.4366, mae: 1.2942, huber: 0.9492, swd: 1.3029, ept: 175.8994
    Epoch [19/50], Test Losses: mse: 6.4312, mae: 1.1594, huber: 0.8176, swd: 0.9076, ept: 181.3941
      Epoch 19 composite train-obj: 0.681165
            No improvement (0.9492), counter 1/5
    Epoch [20/50], Train Losses: mse: 5.0397, mae: 1.0058, huber: 0.6716, swd: 0.6437, ept: 185.0382
    Epoch [20/50], Val Losses: mse: 7.8475, mae: 1.2214, huber: 0.8840, swd: 1.1430, ept: 178.1520
    Epoch [20/50], Test Losses: mse: 5.7135, mae: 1.0930, huber: 0.7577, swd: 0.7712, ept: 182.2389
      Epoch 20 composite train-obj: 0.671648
            No improvement (0.8840), counter 2/5
    Epoch [21/50], Train Losses: mse: 4.8397, mae: 0.9811, huber: 0.6498, swd: 0.6139, ept: 185.4084
    Epoch [21/50], Val Losses: mse: 9.6740, mae: 1.3217, huber: 0.9783, swd: 1.1644, ept: 177.7547
    Epoch [21/50], Test Losses: mse: 7.2727, mae: 1.1827, huber: 0.8430, swd: 0.8871, ept: 181.9503
      Epoch 21 composite train-obj: 0.649784
            No improvement (0.9783), counter 3/5
    Epoch [22/50], Train Losses: mse: 4.8046, mae: 0.9777, huber: 0.6467, swd: 0.6168, ept: 185.4911
    Epoch [22/50], Val Losses: mse: 8.5845, mae: 1.2604, huber: 0.9248, swd: 1.3857, ept: 177.6121
    Epoch [22/50], Test Losses: mse: 5.7867, mae: 1.0734, huber: 0.7427, swd: 0.7685, ept: 182.5563
      Epoch 22 composite train-obj: 0.646739
            No improvement (0.9248), counter 4/5
    Epoch [23/50], Train Losses: mse: 4.5414, mae: 0.9461, huber: 0.6189, swd: 0.5737, ept: 185.9269
    Epoch [23/50], Val Losses: mse: 7.9251, mae: 1.2012, huber: 0.8657, swd: 1.1334, ept: 178.2825
    Epoch [23/50], Test Losses: mse: 5.9561, mae: 1.0800, huber: 0.7464, swd: 0.8288, ept: 183.3469
      Epoch 23 composite train-obj: 0.618931
            Val objective improved 0.8668 → 0.8657, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 4.6194, mae: 0.9524, huber: 0.6250, swd: 0.5786, ept: 185.8722
    Epoch [24/50], Val Losses: mse: 8.7494, mae: 1.2543, huber: 0.9171, swd: 1.1579, ept: 178.7285
    Epoch [24/50], Test Losses: mse: 5.9088, mae: 1.0770, huber: 0.7457, swd: 0.7862, ept: 183.0654
      Epoch 24 composite train-obj: 0.625038
            No improvement (0.9171), counter 1/5
    Epoch [25/50], Train Losses: mse: 4.5160, mae: 0.9422, huber: 0.6162, swd: 0.5672, ept: 185.9915
    Epoch [25/50], Val Losses: mse: 7.6456, mae: 1.2054, huber: 0.8676, swd: 1.1219, ept: 178.8175
    Epoch [25/50], Test Losses: mse: 5.9651, mae: 1.0961, huber: 0.7623, swd: 0.8566, ept: 182.7559
      Epoch 25 composite train-obj: 0.616240
            No improvement (0.8676), counter 2/5
    Epoch [26/50], Train Losses: mse: 4.3277, mae: 0.9202, huber: 0.5970, swd: 0.5416, ept: 186.2793
    Epoch [26/50], Val Losses: mse: 7.1431, mae: 1.1556, huber: 0.8226, swd: 1.0711, ept: 179.4882
    Epoch [26/50], Test Losses: mse: 5.1674, mae: 1.0182, huber: 0.6905, swd: 0.7756, ept: 184.0985
      Epoch 26 composite train-obj: 0.596980
            Val objective improved 0.8657 → 0.8226, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 4.2787, mae: 0.9126, huber: 0.5900, swd: 0.5354, ept: 186.4861
    Epoch [27/50], Val Losses: mse: 7.8132, mae: 1.1796, huber: 0.8462, swd: 1.1977, ept: 178.6444
    Epoch [27/50], Test Losses: mse: 5.6420, mae: 1.0427, huber: 0.7134, swd: 0.7925, ept: 183.7633
      Epoch 27 composite train-obj: 0.589980
            No improvement (0.8462), counter 1/5
    Epoch [28/50], Train Losses: mse: 4.1675, mae: 0.8945, huber: 0.5751, swd: 0.5173, ept: 186.7749
    Epoch [28/50], Val Losses: mse: 7.1446, mae: 1.1351, huber: 0.8083, swd: 1.0628, ept: 179.5227
    Epoch [28/50], Test Losses: mse: 4.9857, mae: 0.9897, huber: 0.6679, swd: 0.7363, ept: 183.8206
      Epoch 28 composite train-obj: 0.575137
            Val objective improved 0.8226 → 0.8083, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 4.0734, mae: 0.8896, huber: 0.5702, swd: 0.4997, ept: 186.8712
    Epoch [29/50], Val Losses: mse: 7.5705, mae: 1.1444, huber: 0.8180, swd: 0.9845, ept: 179.5914
    Epoch [29/50], Test Losses: mse: 5.4567, mae: 1.0072, huber: 0.6868, swd: 0.6576, ept: 183.9624
      Epoch 29 composite train-obj: 0.570234
            No improvement (0.8180), counter 1/5
    Epoch [30/50], Train Losses: mse: 4.0162, mae: 0.8753, huber: 0.5585, swd: 0.4946, ept: 187.0745
    Epoch [30/50], Val Losses: mse: 8.5559, mae: 1.2057, huber: 0.8787, swd: 1.0110, ept: 179.5011
    Epoch [30/50], Test Losses: mse: 6.0616, mae: 1.0633, huber: 0.7390, swd: 0.7440, ept: 182.9361
      Epoch 30 composite train-obj: 0.558463
            No improvement (0.8787), counter 2/5
    Epoch [31/50], Train Losses: mse: 3.8880, mae: 0.8675, huber: 0.5511, swd: 0.4774, ept: 187.2396
    Epoch [31/50], Val Losses: mse: 7.1216, mae: 1.1580, huber: 0.8282, swd: 1.1617, ept: 178.8150
    Epoch [31/50], Test Losses: mse: 5.0556, mae: 1.0116, huber: 0.6866, swd: 0.7431, ept: 183.5746
      Epoch 31 composite train-obj: 0.551079
            No improvement (0.8282), counter 3/5
    Epoch [32/50], Train Losses: mse: 3.8015, mae: 0.8522, huber: 0.5381, swd: 0.4673, ept: 187.4632
    Epoch [32/50], Val Losses: mse: 8.2074, mae: 1.1902, huber: 0.8624, swd: 1.0336, ept: 179.9208
    Epoch [32/50], Test Losses: mse: 5.6595, mae: 1.0329, huber: 0.7094, swd: 0.7237, ept: 184.3625
      Epoch 32 composite train-obj: 0.538078
            No improvement (0.8624), counter 4/5
    Epoch [33/50], Train Losses: mse: 3.8034, mae: 0.8572, huber: 0.5420, swd: 0.4592, ept: 187.5117
    Epoch [33/50], Val Losses: mse: 6.4559, mae: 1.0818, huber: 0.7590, swd: 0.9857, ept: 179.8032
    Epoch [33/50], Test Losses: mse: 4.8304, mae: 0.9641, huber: 0.6453, swd: 0.6956, ept: 184.1183
      Epoch 33 composite train-obj: 0.541957
            Val objective improved 0.8083 → 0.7590, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 3.6166, mae: 0.8334, huber: 0.5217, swd: 0.4387, ept: 187.7376
    Epoch [34/50], Val Losses: mse: 6.6832, mae: 1.0770, huber: 0.7561, swd: 0.8446, ept: 182.1091
    Epoch [34/50], Test Losses: mse: 4.9405, mae: 0.9585, huber: 0.6433, swd: 0.6392, ept: 184.8958
      Epoch 34 composite train-obj: 0.521665
            Val objective improved 0.7590 → 0.7561, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 3.5634, mae: 0.8241, huber: 0.5138, swd: 0.4304, ept: 188.0245
    Epoch [35/50], Val Losses: mse: 7.2927, mae: 1.1213, huber: 0.8019, swd: 1.1185, ept: 180.4647
    Epoch [35/50], Test Losses: mse: 5.3187, mae: 0.9926, huber: 0.6770, swd: 0.6687, ept: 184.3572
      Epoch 35 composite train-obj: 0.513784
            No improvement (0.8019), counter 1/5
    Epoch [36/50], Train Losses: mse: 3.5621, mae: 0.8297, huber: 0.5181, swd: 0.4314, ept: 188.0037
    Epoch [36/50], Val Losses: mse: 6.7762, mae: 1.0716, huber: 0.7540, swd: 0.9675, ept: 180.7262
    Epoch [36/50], Test Losses: mse: 5.1645, mae: 0.9550, huber: 0.6430, swd: 0.7865, ept: 184.6489
      Epoch 36 composite train-obj: 0.518144
            Val objective improved 0.7561 → 0.7540, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 3.5287, mae: 0.8203, huber: 0.5109, swd: 0.4262, ept: 188.1665
    Epoch [37/50], Val Losses: mse: 6.5985, mae: 1.0756, huber: 0.7551, swd: 0.9135, ept: 180.4761
    Epoch [37/50], Test Losses: mse: 5.0412, mae: 0.9628, huber: 0.6465, swd: 0.6681, ept: 184.5393
      Epoch 37 composite train-obj: 0.510885
            No improvement (0.7551), counter 1/5
    Epoch [38/50], Train Losses: mse: 3.4449, mae: 0.8121, huber: 0.5034, swd: 0.4152, ept: 188.2274
    Epoch [38/50], Val Losses: mse: 7.2441, mae: 1.0869, huber: 0.7687, swd: 0.8717, ept: 182.1304
    Epoch [38/50], Test Losses: mse: 4.9115, mae: 0.9404, huber: 0.6277, swd: 0.6369, ept: 185.0760
      Epoch 38 composite train-obj: 0.503417
            No improvement (0.7687), counter 2/5
    Epoch [39/50], Train Losses: mse: 3.3850, mae: 0.8037, huber: 0.4967, swd: 0.4094, ept: 188.3824
    Epoch [39/50], Val Losses: mse: 7.0541, mae: 1.0840, huber: 0.7636, swd: 0.8322, ept: 181.8815
    Epoch [39/50], Test Losses: mse: 5.1366, mae: 0.9711, huber: 0.6533, swd: 0.6126, ept: 185.3228
      Epoch 39 composite train-obj: 0.496705
            No improvement (0.7636), counter 3/5
    Epoch [40/50], Train Losses: mse: 3.3173, mae: 0.7927, huber: 0.4873, swd: 0.4045, ept: 188.5652
    Epoch [40/50], Val Losses: mse: 7.4960, mae: 1.1397, huber: 0.8181, swd: 1.0309, ept: 179.1797
    Epoch [40/50], Test Losses: mse: 5.7872, mae: 1.0232, huber: 0.7044, swd: 0.7597, ept: 183.5743
      Epoch 40 composite train-obj: 0.487323
            No improvement (0.8181), counter 4/5
    Epoch [41/50], Train Losses: mse: 3.3255, mae: 0.7947, huber: 0.4891, swd: 0.3965, ept: 188.5430
    Epoch [41/50], Val Losses: mse: 6.7752, mae: 1.0573, huber: 0.7439, swd: 0.9660, ept: 181.5868
    Epoch [41/50], Test Losses: mse: 5.0768, mae: 0.9422, huber: 0.6328, swd: 0.6022, ept: 184.9682
      Epoch 41 composite train-obj: 0.489080
            Val objective improved 0.7540 → 0.7439, saving checkpoint.
    Epoch [42/50], Train Losses: mse: 3.2269, mae: 0.7841, huber: 0.4795, swd: 0.3873, ept: 188.7532
    Epoch [42/50], Val Losses: mse: 7.6850, mae: 1.1237, huber: 0.8036, swd: 0.9445, ept: 180.6633
    Epoch [42/50], Test Losses: mse: 5.5249, mae: 0.9879, huber: 0.6726, swd: 0.6979, ept: 184.5617
      Epoch 42 composite train-obj: 0.479546
            No improvement (0.8036), counter 1/5
    Epoch [43/50], Train Losses: mse: 3.3086, mae: 0.7906, huber: 0.4859, swd: 0.3957, ept: 188.6465
    Epoch [43/50], Val Losses: mse: 6.8222, mae: 1.0485, huber: 0.7359, swd: 0.7993, ept: 181.6483
    Epoch [43/50], Test Losses: mse: 4.9634, mae: 0.9360, huber: 0.6263, swd: 0.6169, ept: 184.9380
      Epoch 43 composite train-obj: 0.485892
            Val objective improved 0.7439 → 0.7359, saving checkpoint.
    Epoch [44/50], Train Losses: mse: 3.2125, mae: 0.7768, huber: 0.4743, swd: 0.3886, ept: 188.8530
    Epoch [44/50], Val Losses: mse: 6.1999, mae: 1.0171, huber: 0.7078, swd: 0.9499, ept: 182.1377
    Epoch [44/50], Test Losses: mse: 4.7806, mae: 0.9084, huber: 0.6043, swd: 0.5818, ept: 185.5313
      Epoch 44 composite train-obj: 0.474266
            Val objective improved 0.7359 → 0.7078, saving checkpoint.
    Epoch [45/50], Train Losses: mse: 3.1530, mae: 0.7681, huber: 0.4671, swd: 0.3736, ept: 188.8953
    Epoch [45/50], Val Losses: mse: 6.8818, mae: 1.0810, huber: 0.7640, swd: 0.8445, ept: 181.9281
    Epoch [45/50], Test Losses: mse: 5.6429, mae: 0.9893, huber: 0.6763, swd: 0.6182, ept: 184.7985
      Epoch 45 composite train-obj: 0.467106
            No improvement (0.7640), counter 1/5
    Epoch [46/50], Train Losses: mse: 3.1619, mae: 0.7718, huber: 0.4695, swd: 0.3779, ept: 189.0514
    Epoch [46/50], Val Losses: mse: 6.9798, mae: 1.0439, huber: 0.7348, swd: 0.8849, ept: 181.7520
    Epoch [46/50], Test Losses: mse: 5.5005, mae: 0.9479, huber: 0.6420, swd: 0.6103, ept: 184.9702
      Epoch 46 composite train-obj: 0.469521
            No improvement (0.7348), counter 2/5
    Epoch [47/50], Train Losses: mse: 2.9961, mae: 0.7508, huber: 0.4520, swd: 0.3566, ept: 189.2784
    Epoch [47/50], Val Losses: mse: 6.6423, mae: 1.0444, huber: 0.7319, swd: 0.7911, ept: 181.8499
    Epoch [47/50], Test Losses: mse: 5.3757, mae: 0.9481, huber: 0.6399, swd: 0.5653, ept: 185.5376
      Epoch 47 composite train-obj: 0.452014
            No improvement (0.7319), counter 3/5
    Epoch [48/50], Train Losses: mse: 3.0035, mae: 0.7530, huber: 0.4537, swd: 0.3579, ept: 189.2690
    Epoch [48/50], Val Losses: mse: 7.3587, mae: 1.1096, huber: 0.7878, swd: 0.8036, ept: 181.1732
    Epoch [48/50], Test Losses: mse: 5.8154, mae: 1.0004, huber: 0.6834, swd: 0.5762, ept: 184.9592
      Epoch 48 composite train-obj: 0.453734
            No improvement (0.7878), counter 4/5
    Epoch [49/50], Train Losses: mse: 3.0226, mae: 0.7528, huber: 0.4538, swd: 0.3613, ept: 189.2819
    Epoch [49/50], Val Losses: mse: 6.5590, mae: 1.0237, huber: 0.7163, swd: 0.7423, ept: 181.5143
    Epoch [49/50], Test Losses: mse: 5.2662, mae: 0.9408, huber: 0.6365, swd: 0.6345, ept: 184.9995
      Epoch 49 composite train-obj: 0.453785
    Epoch [49/50], Test Losses: mse: 4.7806, mae: 0.9084, huber: 0.6043, swd: 0.5818, ept: 185.5313
    Best round's Test MSE: 4.7806, MAE: 0.9084, SWD: 0.5818
    Best round's Validation MSE: 6.1999, MAE: 1.0171, SWD: 0.9499
    Best round's Test verification MSE : 4.7806, MAE: 0.9084, SWD: 0.5818
    Time taken: 468.51 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq336_pred196_20250513_2232)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 5.1257 ± 0.3356
      mae: 0.9558 ± 0.0550
      huber: 0.6411 ± 0.0442
      swd: 0.6143 ± 0.0299
      ept: 185.2168 ± 0.6879
      count: 37.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 6.6582 ± 0.4960
      mae: 1.0674 ± 0.0629
      huber: 0.7481 ± 0.0524
      swd: 0.9304 ± 0.0152
      ept: 181.8731 ± 0.8274
      count: 37.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 1201.16 seconds
    
    Experiment complete: TimeMixer_lorenz_seq336_pred196_20250513_2232
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 336-336
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],
    enc_in=data_mgr.datasets['lorenz']['channels'],
    dec_in=data_mgr.datasets['lorenz']['channels'],
    c_out=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0]
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 280
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
    Training Batches: 280
    Validation Batches: 36
    Test Batches: 77
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 62.1250, mae: 5.6608, huber: 5.1873, swd: 15.3276, target_std: 13.6390
    Epoch [1/50], Val Losses: mse: 55.4265, mae: 5.2585, huber: 4.7898, swd: 14.9364, target_std: 13.8286
    Epoch [1/50], Test Losses: mse: 53.7948, mae: 5.1324, huber: 4.6650, swd: 15.4885, target_std: 13.1386
      Epoch 1 composite train-obj: 5.187303
            Val objective improved inf → 4.7898, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 46.0226, mae: 4.5231, huber: 4.0630, swd: 11.4842, target_std: 13.6405
    Epoch [2/50], Val Losses: mse: 53.7238, mae: 4.9578, huber: 4.4973, swd: 11.3780, target_std: 13.8286
    Epoch [2/50], Test Losses: mse: 49.5373, mae: 4.7073, huber: 4.2488, swd: 11.8995, target_std: 13.1386
      Epoch 2 composite train-obj: 4.063036
            Val objective improved 4.7898 → 4.4973, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.9498, mae: 4.0968, huber: 3.6449, swd: 8.9706, target_std: 13.6411
    Epoch [3/50], Val Losses: mse: 52.0518, mae: 4.7523, huber: 4.2969, swd: 9.3256, target_std: 13.8286
    Epoch [3/50], Test Losses: mse: 48.0393, mae: 4.5115, huber: 4.0576, swd: 9.7003, target_std: 13.1386
      Epoch 3 composite train-obj: 3.644899
            Val objective improved 4.4973 → 4.2969, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.2047, mae: 3.7931, huber: 3.3471, swd: 7.2691, target_std: 13.6394
    Epoch [4/50], Val Losses: mse: 51.8708, mae: 4.6316, huber: 4.1795, swd: 7.1832, target_std: 13.8286
    Epoch [4/50], Test Losses: mse: 45.7921, mae: 4.2875, huber: 3.8387, swd: 7.6218, target_std: 13.1386
      Epoch 4 composite train-obj: 3.347102
            Val objective improved 4.2969 → 4.1795, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.9589, mae: 3.5472, huber: 3.1057, swd: 5.9845, target_std: 13.6362
    Epoch [5/50], Val Losses: mse: 50.5213, mae: 4.4875, huber: 4.0387, swd: 6.4022, target_std: 13.8286
    Epoch [5/50], Test Losses: mse: 44.0317, mae: 4.1393, huber: 3.6933, swd: 6.1955, target_std: 13.1386
      Epoch 5 composite train-obj: 3.105689
            Val objective improved 4.1795 → 4.0387, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.4420, mae: 3.3539, huber: 2.9160, swd: 5.0483, target_std: 13.6391
    Epoch [6/50], Val Losses: mse: 48.9184, mae: 4.3440, huber: 3.8980, swd: 5.5438, target_std: 13.8286
    Epoch [6/50], Test Losses: mse: 42.1145, mae: 4.0005, huber: 3.5567, swd: 5.6099, target_std: 13.1386
      Epoch 6 composite train-obj: 2.915985
            Val objective improved 4.0387 → 3.8980, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 29.2167, mae: 3.1858, huber: 2.7514, swd: 4.4100, target_std: 13.6415
    Epoch [7/50], Val Losses: mse: 48.4143, mae: 4.2523, huber: 3.8097, swd: 4.9543, target_std: 13.8286
    Epoch [7/50], Test Losses: mse: 41.1724, mae: 3.8986, huber: 3.4582, swd: 5.0619, target_std: 13.1386
      Epoch 7 composite train-obj: 2.751386
            Val objective improved 3.8980 → 3.8097, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 27.3996, mae: 3.0472, huber: 2.6157, swd: 3.9283, target_std: 13.6375
    Epoch [8/50], Val Losses: mse: 47.8460, mae: 4.1915, huber: 3.7488, swd: 4.4789, target_std: 13.8286
    Epoch [8/50], Test Losses: mse: 40.3417, mae: 3.8372, huber: 3.3959, swd: 4.5683, target_std: 13.1386
      Epoch 8 composite train-obj: 2.615735
            Val objective improved 3.8097 → 3.7488, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 25.8655, mae: 2.9328, huber: 2.5038, swd: 3.5634, target_std: 13.6399
    Epoch [9/50], Val Losses: mse: 46.6187, mae: 4.0907, huber: 3.6515, swd: 4.1590, target_std: 13.8286
    Epoch [9/50], Test Losses: mse: 39.3459, mae: 3.7512, huber: 3.3141, swd: 4.1362, target_std: 13.1386
      Epoch 9 composite train-obj: 2.503818
            Val objective improved 3.7488 → 3.6515, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 24.5982, mae: 2.8342, huber: 2.4072, swd: 3.2619, target_std: 13.6381
    Epoch [10/50], Val Losses: mse: 46.4033, mae: 4.0459, huber: 3.6080, swd: 3.9237, target_std: 13.8286
    Epoch [10/50], Test Losses: mse: 38.1706, mae: 3.6499, huber: 3.2150, swd: 3.9012, target_std: 13.1386
      Epoch 10 composite train-obj: 2.407221
            Val objective improved 3.6515 → 3.6080, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 23.5064, mae: 2.7490, huber: 2.3241, swd: 3.0198, target_std: 13.6417
    Epoch [11/50], Val Losses: mse: 45.7667, mae: 3.9961, huber: 3.5603, swd: 4.0211, target_std: 13.8286
    Epoch [11/50], Test Losses: mse: 37.2990, mae: 3.5891, huber: 3.1564, swd: 3.7637, target_std: 13.1386
      Epoch 11 composite train-obj: 2.324132
            Val objective improved 3.6080 → 3.5603, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 22.5508, mae: 2.6789, huber: 2.2556, swd: 2.8366, target_std: 13.6395
    Epoch [12/50], Val Losses: mse: 45.3427, mae: 3.9664, huber: 3.5311, swd: 3.8329, target_std: 13.8286
    Epoch [12/50], Test Losses: mse: 37.5416, mae: 3.5906, huber: 3.1574, swd: 3.5516, target_std: 13.1386
      Epoch 12 composite train-obj: 2.255579
            Val objective improved 3.5603 → 3.5311, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 21.7490, mae: 2.6118, huber: 2.1905, swd: 2.6722, target_std: 13.6411
    Epoch [13/50], Val Losses: mse: 44.6879, mae: 3.9108, huber: 3.4753, swd: 3.5440, target_std: 13.8286
    Epoch [13/50], Test Losses: mse: 36.7512, mae: 3.5405, huber: 3.1069, swd: 3.2168, target_std: 13.1386
      Epoch 13 composite train-obj: 2.190501
            Val objective improved 3.5311 → 3.4753, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 21.0401, mae: 2.5569, huber: 2.1370, swd: 2.5482, target_std: 13.6401
    Epoch [14/50], Val Losses: mse: 43.7965, mae: 3.8470, huber: 3.4148, swd: 3.6075, target_std: 13.8286
    Epoch [14/50], Test Losses: mse: 35.9841, mae: 3.4674, huber: 3.0374, swd: 3.4251, target_std: 13.1386
      Epoch 14 composite train-obj: 2.136981
            Val objective improved 3.4753 → 3.4148, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 20.4167, mae: 2.5047, huber: 2.0866, swd: 2.4203, target_std: 13.6377
    Epoch [15/50], Val Losses: mse: 44.4244, mae: 3.8628, huber: 3.4305, swd: 3.4273, target_std: 13.8286
    Epoch [15/50], Test Losses: mse: 35.7998, mae: 3.4627, huber: 3.0328, swd: 3.1709, target_std: 13.1386
      Epoch 15 composite train-obj: 2.086553
            No improvement (3.4305), counter 1/5
    Epoch [16/50], Train Losses: mse: 19.7713, mae: 2.4532, huber: 2.0367, swd: 2.3083, target_std: 13.6424
    Epoch [16/50], Val Losses: mse: 43.1312, mae: 3.7833, huber: 3.3546, swd: 3.3105, target_std: 13.8286
    Epoch [16/50], Test Losses: mse: 36.0185, mae: 3.4343, huber: 3.0080, swd: 3.0183, target_std: 13.1386
      Epoch 16 composite train-obj: 2.036651
            Val objective improved 3.4148 → 3.3546, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 19.2370, mae: 2.4125, huber: 1.9969, swd: 2.2263, target_std: 13.6437
    Epoch [17/50], Val Losses: mse: 43.8547, mae: 3.8052, huber: 3.3752, swd: 3.2065, target_std: 13.8286
    Epoch [17/50], Test Losses: mse: 34.9311, mae: 3.3846, huber: 2.9568, swd: 2.9237, target_std: 13.1386
      Epoch 17 composite train-obj: 1.996940
            No improvement (3.3752), counter 1/5
    Epoch [18/50], Train Losses: mse: 18.7343, mae: 2.3702, huber: 1.9561, swd: 2.1350, target_std: 13.6424
    Epoch [18/50], Val Losses: mse: 42.6904, mae: 3.7548, huber: 3.3264, swd: 3.2030, target_std: 13.8286
    Epoch [18/50], Test Losses: mse: 35.0507, mae: 3.3526, huber: 2.9270, swd: 2.7345, target_std: 13.1386
      Epoch 18 composite train-obj: 1.956067
            Val objective improved 3.3546 → 3.3264, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 18.2656, mae: 2.3321, huber: 1.9192, swd: 2.0687, target_std: 13.6400
    Epoch [19/50], Val Losses: mse: 42.0974, mae: 3.6781, huber: 3.2539, swd: 3.0961, target_std: 13.8286
    Epoch [19/50], Test Losses: mse: 34.3532, mae: 3.2887, huber: 2.8670, swd: 2.6832, target_std: 13.1386
      Epoch 19 composite train-obj: 1.919200
            Val objective improved 3.3264 → 3.2539, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 17.7957, mae: 2.2929, huber: 1.8814, swd: 1.9882, target_std: 13.6379
    Epoch [20/50], Val Losses: mse: 41.3798, mae: 3.6808, huber: 3.2544, swd: 3.3282, target_std: 13.8286
    Epoch [20/50], Test Losses: mse: 33.9644, mae: 3.2796, huber: 2.8560, swd: 2.7077, target_std: 13.1386
      Epoch 20 composite train-obj: 1.881427
            No improvement (3.2544), counter 1/5
    Epoch [21/50], Train Losses: mse: 17.4751, mae: 2.2685, huber: 1.8577, swd: 1.9516, target_std: 13.6399
    Epoch [21/50], Val Losses: mse: 41.3929, mae: 3.6414, huber: 3.2163, swd: 3.1343, target_std: 13.8286
    Epoch [21/50], Test Losses: mse: 33.8150, mae: 3.2591, huber: 2.8364, swd: 2.6311, target_std: 13.1386
      Epoch 21 composite train-obj: 1.857650
            Val objective improved 3.2539 → 3.2163, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 17.0659, mae: 2.2329, huber: 1.8233, swd: 1.8841, target_std: 13.6411
    Epoch [22/50], Val Losses: mse: 41.3593, mae: 3.6362, huber: 3.2120, swd: 2.9788, target_std: 13.8286
    Epoch [22/50], Test Losses: mse: 33.7934, mae: 3.2439, huber: 2.8230, swd: 2.4183, target_std: 13.1386
      Epoch 22 composite train-obj: 1.823297
            Val objective improved 3.2163 → 3.2120, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 16.7088, mae: 2.2050, huber: 1.7963, swd: 1.8431, target_std: 13.6395
    Epoch [23/50], Val Losses: mse: 41.6158, mae: 3.6401, huber: 3.2150, swd: 2.6892, target_std: 13.8286
    Epoch [23/50], Test Losses: mse: 34.2333, mae: 3.2755, huber: 2.8523, swd: 2.4050, target_std: 13.1386
      Epoch 23 composite train-obj: 1.796332
            No improvement (3.2150), counter 1/5
    Epoch [24/50], Train Losses: mse: 16.4086, mae: 2.1819, huber: 1.7739, swd: 1.7913, target_std: 13.6428
    Epoch [24/50], Val Losses: mse: 41.4261, mae: 3.6156, huber: 3.1930, swd: 2.9168, target_std: 13.8286
    Epoch [24/50], Test Losses: mse: 33.5503, mae: 3.2146, huber: 2.7950, swd: 2.4322, target_std: 13.1386
      Epoch 24 composite train-obj: 1.773863
            Val objective improved 3.2120 → 3.1930, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 15.9677, mae: 2.1436, huber: 1.7372, swd: 1.7434, target_std: 13.6379
    Epoch [25/50], Val Losses: mse: 41.5763, mae: 3.6140, huber: 3.1896, swd: 2.6390, target_std: 13.8286
    Epoch [25/50], Test Losses: mse: 33.6750, mae: 3.2163, huber: 2.7957, swd: 2.2475, target_std: 13.1386
      Epoch 25 composite train-obj: 1.737199
            Val objective improved 3.1930 → 3.1896, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 15.7788, mae: 2.1274, huber: 1.7213, swd: 1.7052, target_std: 13.6393
    Epoch [26/50], Val Losses: mse: 41.1791, mae: 3.5927, huber: 3.1702, swd: 2.7765, target_std: 13.8286
    Epoch [26/50], Test Losses: mse: 33.5240, mae: 3.2029, huber: 2.7830, swd: 2.4154, target_std: 13.1386
      Epoch 26 composite train-obj: 1.721283
            Val objective improved 3.1896 → 3.1702, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 15.6712, mae: 2.1204, huber: 1.7147, swd: 1.7034, target_std: 13.6404
    Epoch [27/50], Val Losses: mse: 39.5216, mae: 3.5184, huber: 3.0953, swd: 2.9157, target_std: 13.8286
    Epoch [27/50], Test Losses: mse: 32.3422, mae: 3.1604, huber: 2.7403, swd: 2.6867, target_std: 13.1386
      Epoch 27 composite train-obj: 1.714683
            Val objective improved 3.1702 → 3.0953, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 15.2801, mae: 2.0862, huber: 1.6817, swd: 1.6348, target_std: 13.6414
    Epoch [28/50], Val Losses: mse: 40.3344, mae: 3.5391, huber: 3.1180, swd: 2.7354, target_std: 13.8286
    Epoch [28/50], Test Losses: mse: 33.0695, mae: 3.1657, huber: 2.7472, swd: 2.2165, target_std: 13.1386
      Epoch 28 composite train-obj: 1.681731
            No improvement (3.1180), counter 1/5
    Epoch [29/50], Train Losses: mse: 14.9409, mae: 2.0590, huber: 1.6556, swd: 1.6107, target_std: 13.6417
    Epoch [29/50], Val Losses: mse: 40.4588, mae: 3.5458, huber: 3.1229, swd: 2.6139, target_std: 13.8286
    Epoch [29/50], Test Losses: mse: 32.9281, mae: 3.1705, huber: 2.7508, swd: 2.1694, target_std: 13.1386
      Epoch 29 composite train-obj: 1.655580
            No improvement (3.1229), counter 2/5
    Epoch [30/50], Train Losses: mse: 14.7318, mae: 2.0444, huber: 1.6414, swd: 1.5779, target_std: 13.6421
    Epoch [30/50], Val Losses: mse: 40.1650, mae: 3.5196, huber: 3.0986, swd: 2.6486, target_std: 13.8286
    Epoch [30/50], Test Losses: mse: 32.9933, mae: 3.1632, huber: 2.7442, swd: 2.1342, target_std: 13.1386
      Epoch 30 composite train-obj: 1.641437
            No improvement (3.0986), counter 3/5
    Epoch [31/50], Train Losses: mse: 14.4685, mae: 2.0203, huber: 1.6181, swd: 1.5404, target_std: 13.6429
    Epoch [31/50], Val Losses: mse: 39.1582, mae: 3.4604, huber: 3.0407, swd: 2.5745, target_std: 13.8286
    Epoch [31/50], Test Losses: mse: 32.1187, mae: 3.1048, huber: 2.6881, swd: 2.1883, target_std: 13.1386
      Epoch 31 composite train-obj: 1.618050
            Val objective improved 3.0953 → 3.0407, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 14.3271, mae: 2.0099, huber: 1.6083, swd: 1.5219, target_std: 13.6416
    Epoch [32/50], Val Losses: mse: 40.0754, mae: 3.5124, huber: 3.0921, swd: 2.8151, target_std: 13.8286
    Epoch [32/50], Test Losses: mse: 32.2832, mae: 3.1175, huber: 2.7011, swd: 2.2924, target_std: 13.1386
      Epoch 32 composite train-obj: 1.608336
            No improvement (3.0921), counter 1/5
    Epoch [33/50], Train Losses: mse: 14.1051, mae: 1.9861, huber: 1.5855, swd: 1.4945, target_std: 13.6411
    Epoch [33/50], Val Losses: mse: 39.6565, mae: 3.4877, huber: 3.0694, swd: 2.7493, target_std: 13.8286
    Epoch [33/50], Test Losses: mse: 31.9655, mae: 3.1007, huber: 2.6855, swd: 2.3784, target_std: 13.1386
      Epoch 33 composite train-obj: 1.585482
            No improvement (3.0694), counter 2/5
    Epoch [34/50], Train Losses: mse: 13.8994, mae: 1.9740, huber: 1.5736, swd: 1.4813, target_std: 13.6403
    Epoch [34/50], Val Losses: mse: 40.7566, mae: 3.5343, huber: 3.1141, swd: 2.6173, target_std: 13.8286
    Epoch [34/50], Test Losses: mse: 32.7638, mae: 3.1241, huber: 2.7076, swd: 2.1108, target_std: 13.1386
      Epoch 34 composite train-obj: 1.573640
            No improvement (3.1141), counter 3/5
    Epoch [35/50], Train Losses: mse: 13.7183, mae: 1.9564, huber: 1.5569, swd: 1.4404, target_std: 13.6412
    Epoch [35/50], Val Losses: mse: 39.4473, mae: 3.4861, huber: 3.0644, swd: 2.7627, target_std: 13.8286
    Epoch [35/50], Test Losses: mse: 32.7857, mae: 3.1338, huber: 2.7158, swd: 2.0935, target_std: 13.1386
      Epoch 35 composite train-obj: 1.556886
            No improvement (3.0644), counter 4/5
    Epoch [36/50], Train Losses: mse: 13.5801, mae: 1.9468, huber: 1.5475, swd: 1.4391, target_std: 13.6419
    Epoch [36/50], Val Losses: mse: 38.7217, mae: 3.4171, huber: 3.0002, swd: 2.5201, target_std: 13.8286
    Epoch [36/50], Test Losses: mse: 31.6577, mae: 3.0540, huber: 2.6398, swd: 2.1343, target_std: 13.1386
      Epoch 36 composite train-obj: 1.547479
            Val objective improved 3.0407 → 3.0002, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 13.3226, mae: 1.9222, huber: 1.5239, swd: 1.4021, target_std: 13.6395
    Epoch [37/50], Val Losses: mse: 38.1420, mae: 3.3942, huber: 2.9781, swd: 2.5906, target_std: 13.8286
    Epoch [37/50], Test Losses: mse: 32.1906, mae: 3.0866, huber: 2.6728, swd: 2.1934, target_std: 13.1386
      Epoch 37 composite train-obj: 1.523871
            Val objective improved 3.0002 → 2.9781, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 13.2720, mae: 1.9212, huber: 1.5229, swd: 1.4062, target_std: 13.6415
    Epoch [38/50], Val Losses: mse: 41.6103, mae: 3.5590, huber: 3.1388, swd: 2.5851, target_std: 13.8286
    Epoch [38/50], Test Losses: mse: 32.2824, mae: 3.1144, huber: 2.6968, swd: 2.1741, target_std: 13.1386
      Epoch 38 composite train-obj: 1.522881
            No improvement (3.1388), counter 1/5
    Epoch [39/50], Train Losses: mse: 13.2131, mae: 1.9143, huber: 1.5164, swd: 1.3891, target_std: 13.6382
    Epoch [39/50], Val Losses: mse: 39.1317, mae: 3.4403, huber: 3.0235, swd: 2.7403, target_std: 13.8286
    Epoch [39/50], Test Losses: mse: 31.8257, mae: 3.0687, huber: 2.6543, swd: 2.1084, target_std: 13.1386
      Epoch 39 composite train-obj: 1.516417
            No improvement (3.0235), counter 2/5
    Epoch [40/50], Train Losses: mse: 12.8600, mae: 1.8832, huber: 1.4865, swd: 1.3419, target_std: 13.6372
    Epoch [40/50], Val Losses: mse: 38.8043, mae: 3.4182, huber: 3.0024, swd: 2.7537, target_std: 13.8286
    Epoch [40/50], Test Losses: mse: 30.7193, mae: 3.0030, huber: 2.5902, swd: 2.2481, target_std: 13.1386
      Epoch 40 composite train-obj: 1.486532
            No improvement (3.0024), counter 3/5
    Epoch [41/50], Train Losses: mse: 12.7511, mae: 1.8754, huber: 1.4790, swd: 1.3432, target_std: 13.6382
    Epoch [41/50], Val Losses: mse: 39.6811, mae: 3.4530, huber: 3.0379, swd: 2.7550, target_std: 13.8286
    Epoch [41/50], Test Losses: mse: 31.9174, mae: 3.0479, huber: 2.6355, swd: 2.1035, target_std: 13.1386
      Epoch 41 composite train-obj: 1.479017
            No improvement (3.0379), counter 4/5
    Epoch [42/50], Train Losses: mse: 12.6727, mae: 1.8701, huber: 1.4738, swd: 1.3375, target_std: 13.6395
    Epoch [42/50], Val Losses: mse: 38.4553, mae: 3.4094, huber: 2.9917, swd: 2.4940, target_std: 13.8286
    Epoch [42/50], Test Losses: mse: 32.1002, mae: 3.0735, huber: 2.6596, swd: 2.0227, target_std: 13.1386
      Epoch 42 composite train-obj: 1.473769
    Epoch [42/50], Test Losses: mse: 32.1906, mae: 3.0866, huber: 2.6728, swd: 2.1934, target_std: 13.1386
    Best round's Test MSE: 32.1906, MAE: 3.0866, SWD: 2.1934
    Best round's Validation MSE: 38.1420, MAE: 3.3942, SWD: 2.5906
    Best round's Test verification MSE : 32.1906, MAE: 3.0866, SWD: 2.1934
    Time taken: 437.26 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 62.9104, mae: 5.7109, huber: 5.2369, swd: 15.0763, target_std: 13.6396
    Epoch [1/50], Val Losses: mse: 57.7535, mae: 5.4039, huber: 4.9338, swd: 14.5393, target_std: 13.8286
    Epoch [1/50], Test Losses: mse: 54.7661, mae: 5.1752, huber: 4.7064, swd: 14.9296, target_std: 13.1386
      Epoch 1 composite train-obj: 5.236852
            Val objective improved inf → 4.9338, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 45.8259, mae: 4.5419, huber: 4.0807, swd: 11.8673, target_std: 13.6389
    Epoch [2/50], Val Losses: mse: 54.3247, mae: 4.9907, huber: 4.5294, swd: 11.3128, target_std: 13.8286
    Epoch [2/50], Test Losses: mse: 50.1805, mae: 4.7364, huber: 4.2764, swd: 11.3795, target_std: 13.1386
      Epoch 2 composite train-obj: 4.080689
            Val objective improved 4.9338 → 4.5294, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.5136, mae: 4.0883, huber: 3.6355, swd: 8.9790, target_std: 13.6382
    Epoch [3/50], Val Losses: mse: 52.8163, mae: 4.7770, huber: 4.3211, swd: 9.0725, target_std: 13.8286
    Epoch [3/50], Test Losses: mse: 47.1159, mae: 4.4561, huber: 4.0022, swd: 9.1027, target_std: 13.1386
      Epoch 3 composite train-obj: 3.635469
            Val objective improved 4.5294 → 4.3211, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.8418, mae: 3.7855, huber: 3.3387, swd: 7.2081, target_std: 13.6408
    Epoch [4/50], Val Losses: mse: 52.3960, mae: 4.6478, huber: 4.1936, swd: 6.8909, target_std: 13.8286
    Epoch [4/50], Test Losses: mse: 46.3576, mae: 4.3341, huber: 3.8817, swd: 7.0159, target_std: 13.1386
      Epoch 4 composite train-obj: 3.338661
            Val objective improved 4.3211 → 4.1936, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.7739, mae: 3.5474, huber: 3.1049, swd: 5.9504, target_std: 13.6370
    Epoch [5/50], Val Losses: mse: 51.7055, mae: 4.5254, huber: 4.0747, swd: 5.7254, target_std: 13.8286
    Epoch [5/50], Test Losses: mse: 44.4888, mae: 4.1777, huber: 3.7289, swd: 5.7722, target_std: 13.1386
      Epoch 5 composite train-obj: 3.104884
            Val objective improved 4.1936 → 4.0747, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.9872, mae: 3.3413, huber: 2.9026, swd: 5.0305, target_std: 13.6389
    Epoch [6/50], Val Losses: mse: 50.6991, mae: 4.4187, huber: 3.9690, swd: 4.9757, target_std: 13.8286
    Epoch [6/50], Test Losses: mse: 43.7054, mae: 4.1073, huber: 3.6595, swd: 5.2731, target_std: 13.1386
      Epoch 6 composite train-obj: 2.902603
            Val objective improved 4.0747 → 3.9690, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.4990, mae: 3.1582, huber: 2.7225, swd: 4.2654, target_std: 13.6387
    Epoch [7/50], Val Losses: mse: 48.4713, mae: 4.2332, huber: 3.7885, swd: 4.2424, target_std: 13.8286
    Epoch [7/50], Test Losses: mse: 40.8369, mae: 3.8866, huber: 3.4440, swd: 4.3516, target_std: 13.1386
      Epoch 7 composite train-obj: 2.722532
            Val objective improved 3.9690 → 3.7885, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 26.2581, mae: 2.9895, huber: 2.5574, swd: 3.7073, target_std: 13.6388
    Epoch [8/50], Val Losses: mse: 47.4703, mae: 4.1220, huber: 3.6804, swd: 4.0503, target_std: 13.8286
    Epoch [8/50], Test Losses: mse: 39.1565, mae: 3.7599, huber: 3.3194, swd: 4.1079, target_std: 13.1386
      Epoch 8 composite train-obj: 2.557381
            Val objective improved 3.7885 → 3.6804, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.4230, mae: 2.8460, huber: 2.4171, swd: 3.2671, target_std: 13.6395
    Epoch [9/50], Val Losses: mse: 46.3133, mae: 4.0287, huber: 3.5897, swd: 3.7392, target_std: 13.8286
    Epoch [9/50], Test Losses: mse: 37.8170, mae: 3.6531, huber: 3.2161, swd: 3.7630, target_std: 13.1386
      Epoch 9 composite train-obj: 2.417084
            Val objective improved 3.6804 → 3.5897, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 22.8441, mae: 2.7236, huber: 2.2978, swd: 2.9539, target_std: 13.6399
    Epoch [10/50], Val Losses: mse: 44.4573, mae: 3.9025, huber: 3.4672, swd: 3.5045, target_std: 13.8286
    Epoch [10/50], Test Losses: mse: 35.6326, mae: 3.4881, huber: 3.0555, swd: 3.3388, target_std: 13.1386
      Epoch 10 composite train-obj: 2.297791
            Val objective improved 3.5897 → 3.4672, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.3416, mae: 2.6133, huber: 2.1899, swd: 2.6978, target_std: 13.6390
    Epoch [11/50], Val Losses: mse: 42.9166, mae: 3.8119, huber: 3.3791, swd: 3.6101, target_std: 13.8286
    Epoch [11/50], Test Losses: mse: 34.8039, mae: 3.4209, huber: 2.9904, swd: 3.2693, target_std: 13.1386
      Epoch 11 composite train-obj: 2.189945
            Val objective improved 3.4672 → 3.3791, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.0680, mae: 2.5217, huber: 2.1005, swd: 2.5172, target_std: 13.6392
    Epoch [12/50], Val Losses: mse: 40.7222, mae: 3.7022, huber: 3.2702, swd: 3.7163, target_std: 13.8286
    Epoch [12/50], Test Losses: mse: 32.8314, mae: 3.3164, huber: 2.8867, swd: 3.2419, target_std: 13.1386
      Epoch 12 composite train-obj: 2.100535
            Val objective improved 3.3791 → 3.2702, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.8566, mae: 2.4353, huber: 2.0162, swd: 2.3520, target_std: 13.6410
    Epoch [13/50], Val Losses: mse: 39.5584, mae: 3.6181, huber: 3.1886, swd: 3.3831, target_std: 13.8286
    Epoch [13/50], Test Losses: mse: 31.3300, mae: 3.2197, huber: 2.7931, swd: 3.0398, target_std: 13.1386
      Epoch 13 composite train-obj: 2.016184
            Val objective improved 3.2702 → 3.1886, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 17.8055, mae: 2.3525, huber: 1.9358, swd: 2.1985, target_std: 13.6410
    Epoch [14/50], Val Losses: mse: 38.8471, mae: 3.5984, huber: 3.1669, swd: 3.6559, target_std: 13.8286
    Epoch [14/50], Test Losses: mse: 30.9267, mae: 3.1870, huber: 2.7591, swd: 2.9622, target_std: 13.1386
      Epoch 14 composite train-obj: 1.935833
            Val objective improved 3.1886 → 3.1669, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 16.9941, mae: 2.2888, huber: 1.8740, swd: 2.0719, target_std: 13.6398
    Epoch [15/50], Val Losses: mse: 37.6501, mae: 3.4922, huber: 3.0647, swd: 3.2986, target_std: 13.8286
    Epoch [15/50], Test Losses: mse: 28.9877, mae: 3.0523, huber: 2.6281, swd: 2.7048, target_std: 13.1386
      Epoch 15 composite train-obj: 1.874008
            Val objective improved 3.1669 → 3.0647, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.1255, mae: 2.2141, huber: 1.8018, swd: 1.9356, target_std: 13.6397
    Epoch [16/50], Val Losses: mse: 36.3263, mae: 3.4262, huber: 2.9990, swd: 2.9335, target_std: 13.8286
    Epoch [16/50], Test Losses: mse: 28.8731, mae: 3.0605, huber: 2.6361, swd: 2.6992, target_std: 13.1386
      Epoch 16 composite train-obj: 1.801804
            Val objective improved 3.0647 → 2.9990, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.5771, mae: 2.1707, huber: 1.7599, swd: 1.8692, target_std: 13.6386
    Epoch [17/50], Val Losses: mse: 35.3443, mae: 3.3590, huber: 2.9342, swd: 3.4003, target_std: 13.8286
    Epoch [17/50], Test Losses: mse: 27.5777, mae: 2.9541, huber: 2.5326, swd: 2.6656, target_std: 13.1386
      Epoch 17 composite train-obj: 1.759867
            Val objective improved 2.9990 → 2.9342, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 14.9368, mae: 2.1158, huber: 1.7069, swd: 1.7778, target_std: 13.6389
    Epoch [18/50], Val Losses: mse: 33.7084, mae: 3.2585, huber: 2.8363, swd: 3.0485, target_std: 13.8286
    Epoch [18/50], Test Losses: mse: 26.1388, mae: 2.8520, huber: 2.4335, swd: 2.4703, target_std: 13.1386
      Epoch 18 composite train-obj: 1.706876
            Val objective improved 2.9342 → 2.8363, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 14.4252, mae: 2.0692, huber: 1.6620, swd: 1.6969, target_std: 13.6380
    Epoch [19/50], Val Losses: mse: 34.0560, mae: 3.2538, huber: 2.8317, swd: 3.0029, target_std: 13.8286
    Epoch [19/50], Test Losses: mse: 26.4877, mae: 2.8742, huber: 2.4548, swd: 2.5515, target_std: 13.1386
      Epoch 19 composite train-obj: 1.661951
            Val objective improved 2.8363 → 2.8317, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 14.0735, mae: 2.0378, huber: 1.6317, swd: 1.6534, target_std: 13.6400
    Epoch [20/50], Val Losses: mse: 33.3515, mae: 3.2293, huber: 2.8069, swd: 2.8928, target_std: 13.8286
    Epoch [20/50], Test Losses: mse: 25.5162, mae: 2.8317, huber: 2.4128, swd: 2.5022, target_std: 13.1386
      Epoch 20 composite train-obj: 1.631686
            Val objective improved 2.8317 → 2.8069, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 13.5105, mae: 1.9865, huber: 1.5824, swd: 1.5642, target_std: 13.6411
    Epoch [21/50], Val Losses: mse: 32.8350, mae: 3.1917, huber: 2.7716, swd: 2.8756, target_std: 13.8286
    Epoch [21/50], Test Losses: mse: 26.0749, mae: 2.8218, huber: 2.4049, swd: 2.0544, target_std: 13.1386
      Epoch 21 composite train-obj: 1.582383
            Val objective improved 2.8069 → 2.7716, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.1458, mae: 1.9566, huber: 1.5536, swd: 1.5210, target_std: 13.6405
    Epoch [22/50], Val Losses: mse: 32.4405, mae: 3.1464, huber: 2.7291, swd: 3.0723, target_std: 13.8286
    Epoch [22/50], Test Losses: mse: 24.6789, mae: 2.7344, huber: 2.3210, swd: 2.3338, target_std: 13.1386
      Epoch 22 composite train-obj: 1.553638
            Val objective improved 2.7716 → 2.7291, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 12.8055, mae: 1.9224, huber: 1.5208, swd: 1.4836, target_std: 13.6379
    Epoch [23/50], Val Losses: mse: 33.2010, mae: 3.1861, huber: 2.7673, swd: 2.9026, target_std: 13.8286
    Epoch [23/50], Test Losses: mse: 25.0150, mae: 2.7531, huber: 2.3386, swd: 2.1948, target_std: 13.1386
      Epoch 23 composite train-obj: 1.520820
            No improvement (2.7673), counter 1/5
    Epoch [24/50], Train Losses: mse: 12.5339, mae: 1.8978, huber: 1.4971, swd: 1.4439, target_std: 13.6368
    Epoch [24/50], Val Losses: mse: 30.4624, mae: 3.0554, huber: 2.6367, swd: 2.8401, target_std: 13.8286
    Epoch [24/50], Test Losses: mse: 23.8510, mae: 2.7002, huber: 2.2852, swd: 2.1970, target_std: 13.1386
      Epoch 24 composite train-obj: 1.497126
            Val objective improved 2.7291 → 2.6367, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 12.1991, mae: 1.8677, huber: 1.4684, swd: 1.4026, target_std: 13.6376
    Epoch [25/50], Val Losses: mse: 31.4618, mae: 3.0862, huber: 2.6685, swd: 2.8688, target_std: 13.8286
    Epoch [25/50], Test Losses: mse: 24.6726, mae: 2.7312, huber: 2.3170, swd: 2.1416, target_std: 13.1386
      Epoch 25 composite train-obj: 1.468363
            No improvement (2.6685), counter 1/5
    Epoch [26/50], Train Losses: mse: 11.9321, mae: 1.8448, huber: 1.4462, swd: 1.3654, target_std: 13.6390
    Epoch [26/50], Val Losses: mse: 31.3498, mae: 3.0579, huber: 2.6418, swd: 2.9809, target_std: 13.8286
    Epoch [26/50], Test Losses: mse: 23.7258, mae: 2.6764, huber: 2.2641, swd: 2.3714, target_std: 13.1386
      Epoch 26 composite train-obj: 1.446209
            No improvement (2.6418), counter 2/5
    Epoch [27/50], Train Losses: mse: 11.6936, mae: 1.8219, huber: 1.4244, swd: 1.3228, target_std: 13.6407
    Epoch [27/50], Val Losses: mse: 29.5224, mae: 2.9689, huber: 2.5546, swd: 2.8861, target_std: 13.8286
    Epoch [27/50], Test Losses: mse: 22.5682, mae: 2.6004, huber: 2.1902, swd: 2.2373, target_std: 13.1386
      Epoch 27 composite train-obj: 1.424359
            Val objective improved 2.6367 → 2.5546, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 11.3064, mae: 1.7860, huber: 1.3900, swd: 1.2732, target_std: 13.6399
    Epoch [28/50], Val Losses: mse: 30.3069, mae: 3.0148, huber: 2.5997, swd: 2.9434, target_std: 13.8286
    Epoch [28/50], Test Losses: mse: 23.7688, mae: 2.6578, huber: 2.2465, swd: 2.0894, target_std: 13.1386
      Epoch 28 composite train-obj: 1.389969
            No improvement (2.5997), counter 1/5
    Epoch [29/50], Train Losses: mse: 11.1608, mae: 1.7724, huber: 1.3769, swd: 1.2565, target_std: 13.6369
    Epoch [29/50], Val Losses: mse: 30.6544, mae: 3.0169, huber: 2.6035, swd: 3.1296, target_std: 13.8286
    Epoch [29/50], Test Losses: mse: 22.2527, mae: 2.5774, huber: 2.1683, swd: 2.3156, target_std: 13.1386
      Epoch 29 composite train-obj: 1.376915
            No improvement (2.6035), counter 2/5
    Epoch [30/50], Train Losses: mse: 10.9166, mae: 1.7517, huber: 1.3571, swd: 1.2326, target_std: 13.6393
    Epoch [30/50], Val Losses: mse: 28.6948, mae: 2.9357, huber: 2.5211, swd: 2.8593, target_std: 13.8286
    Epoch [30/50], Test Losses: mse: 22.2446, mae: 2.5770, huber: 2.1661, swd: 2.0925, target_std: 13.1386
      Epoch 30 composite train-obj: 1.357082
            Val objective improved 2.5546 → 2.5211, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 10.6628, mae: 1.7258, huber: 1.3326, swd: 1.2000, target_std: 13.6385
    Epoch [31/50], Val Losses: mse: 30.2367, mae: 2.9991, huber: 2.5842, swd: 2.8222, target_std: 13.8286
    Epoch [31/50], Test Losses: mse: 22.5867, mae: 2.6026, huber: 2.1919, swd: 2.1751, target_std: 13.1386
      Epoch 31 composite train-obj: 1.332610
            No improvement (2.5842), counter 1/5
    Epoch [32/50], Train Losses: mse: 10.4912, mae: 1.7098, huber: 1.3172, swd: 1.1737, target_std: 13.6435
    Epoch [32/50], Val Losses: mse: 28.9028, mae: 2.9182, huber: 2.5060, swd: 2.6338, target_std: 13.8286
    Epoch [32/50], Test Losses: mse: 21.3511, mae: 2.5154, huber: 2.1076, swd: 2.0508, target_std: 13.1386
      Epoch 32 composite train-obj: 1.317249
            Val objective improved 2.5211 → 2.5060, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 10.4156, mae: 1.7013, huber: 1.3092, swd: 1.1686, target_std: 13.6395
    Epoch [33/50], Val Losses: mse: 28.2056, mae: 2.8686, huber: 2.4590, swd: 2.8517, target_std: 13.8286
    Epoch [33/50], Test Losses: mse: 21.1587, mae: 2.4835, huber: 2.0782, swd: 2.1313, target_std: 13.1386
      Epoch 33 composite train-obj: 1.309179
            Val objective improved 2.5060 → 2.4590, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 10.0541, mae: 1.6654, huber: 1.2752, swd: 1.1073, target_std: 13.6408
    Epoch [34/50], Val Losses: mse: 28.2833, mae: 2.8876, huber: 2.4745, swd: 2.5595, target_std: 13.8286
    Epoch [34/50], Test Losses: mse: 21.5430, mae: 2.5067, huber: 2.0975, swd: 1.9374, target_std: 13.1386
      Epoch 34 composite train-obj: 1.275151
            No improvement (2.4745), counter 1/5
    Epoch [35/50], Train Losses: mse: 9.9035, mae: 1.6513, huber: 1.2615, swd: 1.0964, target_std: 13.6386
    Epoch [35/50], Val Losses: mse: 28.5421, mae: 2.8796, huber: 2.4700, swd: 2.9909, target_std: 13.8286
    Epoch [35/50], Test Losses: mse: 20.8854, mae: 2.4738, huber: 2.0681, swd: 2.1648, target_std: 13.1386
      Epoch 35 composite train-obj: 1.261541
            No improvement (2.4700), counter 2/5
    Epoch [36/50], Train Losses: mse: 9.8018, mae: 1.6392, huber: 1.2503, swd: 1.0836, target_std: 13.6388
    Epoch [36/50], Val Losses: mse: 28.4161, mae: 2.8751, huber: 2.4648, swd: 2.8887, target_std: 13.8286
    Epoch [36/50], Test Losses: mse: 21.1596, mae: 2.4917, huber: 2.0859, swd: 2.1135, target_std: 13.1386
      Epoch 36 composite train-obj: 1.250309
            No improvement (2.4648), counter 3/5
    Epoch [37/50], Train Losses: mse: 9.6200, mae: 1.6210, huber: 1.2329, swd: 1.0491, target_std: 13.6405
    Epoch [37/50], Val Losses: mse: 27.5618, mae: 2.8471, huber: 2.4369, swd: 2.8394, target_std: 13.8286
    Epoch [37/50], Test Losses: mse: 20.9406, mae: 2.4838, huber: 2.0775, swd: 2.1379, target_std: 13.1386
      Epoch 37 composite train-obj: 1.232860
            Val objective improved 2.4590 → 2.4369, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 9.5326, mae: 1.6128, huber: 1.2250, swd: 1.0502, target_std: 13.6422
    Epoch [38/50], Val Losses: mse: 26.8467, mae: 2.7907, huber: 2.3830, swd: 3.0288, target_std: 13.8286
    Epoch [38/50], Test Losses: mse: 20.3730, mae: 2.4245, huber: 2.0214, swd: 2.1678, target_std: 13.1386
      Epoch 38 composite train-obj: 1.225041
            Val objective improved 2.4369 → 2.3830, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 9.3057, mae: 1.5869, huber: 1.2007, swd: 1.0144, target_std: 13.6401
    Epoch [39/50], Val Losses: mse: 26.1924, mae: 2.7525, huber: 2.3461, swd: 2.8545, target_std: 13.8286
    Epoch [39/50], Test Losses: mse: 19.9960, mae: 2.4188, huber: 2.0164, swd: 2.1286, target_std: 13.1386
      Epoch 39 composite train-obj: 1.200705
            Val objective improved 2.3830 → 2.3461, saving checkpoint.
    Epoch [40/50], Train Losses: mse: 9.1084, mae: 1.5668, huber: 1.1818, swd: 0.9988, target_std: 13.6390
    Epoch [40/50], Val Losses: mse: 25.5330, mae: 2.7213, huber: 2.3157, swd: 2.7791, target_std: 13.8286
    Epoch [40/50], Test Losses: mse: 19.2819, mae: 2.3635, huber: 1.9628, swd: 2.0169, target_std: 13.1386
      Epoch 40 composite train-obj: 1.181831
            Val objective improved 2.3461 → 2.3157, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 9.0298, mae: 1.5574, huber: 1.1729, swd: 0.9744, target_std: 13.6414
    Epoch [41/50], Val Losses: mse: 26.1081, mae: 2.7618, huber: 2.3524, swd: 2.7215, target_std: 13.8286
    Epoch [41/50], Test Losses: mse: 19.6699, mae: 2.3981, huber: 1.9934, swd: 1.8728, target_std: 13.1386
      Epoch 41 composite train-obj: 1.172948
            No improvement (2.3524), counter 1/5
    Epoch [42/50], Train Losses: mse: 8.9847, mae: 1.5551, huber: 1.1705, swd: 0.9759, target_std: 13.6381
    Epoch [42/50], Val Losses: mse: 25.5493, mae: 2.7179, huber: 2.3107, swd: 2.6513, target_std: 13.8286
    Epoch [42/50], Test Losses: mse: 19.7484, mae: 2.3929, huber: 1.9893, swd: 1.9701, target_std: 13.1386
      Epoch 42 composite train-obj: 1.170451
            Val objective improved 2.3157 → 2.3107, saving checkpoint.
    Epoch [43/50], Train Losses: mse: 8.7977, mae: 1.5336, huber: 1.1502, swd: 0.9424, target_std: 13.6424
    Epoch [43/50], Val Losses: mse: 25.5333, mae: 2.7298, huber: 2.3225, swd: 2.8518, target_std: 13.8286
    Epoch [43/50], Test Losses: mse: 19.7346, mae: 2.3854, huber: 1.9836, swd: 2.0042, target_std: 13.1386
      Epoch 43 composite train-obj: 1.150236
            No improvement (2.3225), counter 1/5
    Epoch [44/50], Train Losses: mse: 8.7549, mae: 1.5260, huber: 1.1432, swd: 0.9412, target_std: 13.6385
    Epoch [44/50], Val Losses: mse: 24.9140, mae: 2.6664, huber: 2.2606, swd: 2.5840, target_std: 13.8286
    Epoch [44/50], Test Losses: mse: 18.7928, mae: 2.3178, huber: 1.9165, swd: 1.8835, target_std: 13.1386
      Epoch 44 composite train-obj: 1.143248
            Val objective improved 2.3107 → 2.2606, saving checkpoint.
    Epoch [45/50], Train Losses: mse: 8.5196, mae: 1.5027, huber: 1.1213, swd: 0.9042, target_std: 13.6401
    Epoch [45/50], Val Losses: mse: 24.8842, mae: 2.6937, huber: 2.2859, swd: 2.6451, target_std: 13.8286
    Epoch [45/50], Test Losses: mse: 18.8687, mae: 2.3515, huber: 1.9484, swd: 1.9614, target_std: 13.1386
      Epoch 45 composite train-obj: 1.121280
            No improvement (2.2859), counter 1/5
    Epoch [46/50], Train Losses: mse: 8.5512, mae: 1.5038, huber: 1.1224, swd: 0.9146, target_std: 13.6415
    Epoch [46/50], Val Losses: mse: 25.4724, mae: 2.7240, huber: 2.3165, swd: 2.6667, target_std: 13.8286
    Epoch [46/50], Test Losses: mse: 19.5491, mae: 2.3950, huber: 1.9918, swd: 1.8250, target_std: 13.1386
      Epoch 46 composite train-obj: 1.122406
            No improvement (2.3165), counter 2/5
    Epoch [47/50], Train Losses: mse: 8.3987, mae: 1.4906, huber: 1.1098, swd: 0.8895, target_std: 13.6393
    Epoch [47/50], Val Losses: mse: 25.0751, mae: 2.6862, huber: 2.2796, swd: 2.7781, target_std: 13.8286
    Epoch [47/50], Test Losses: mse: 19.3952, mae: 2.3569, huber: 1.9546, swd: 2.0557, target_std: 13.1386
      Epoch 47 composite train-obj: 1.109774
            No improvement (2.2796), counter 3/5
    Epoch [48/50], Train Losses: mse: 8.2459, mae: 1.4715, huber: 1.0918, swd: 0.8748, target_std: 13.6394
    Epoch [48/50], Val Losses: mse: 24.6951, mae: 2.6731, huber: 2.2678, swd: 2.6805, target_std: 13.8286
    Epoch [48/50], Test Losses: mse: 18.6353, mae: 2.3153, huber: 1.9148, swd: 1.7905, target_std: 13.1386
      Epoch 48 composite train-obj: 1.091844
            No improvement (2.2678), counter 4/5
    Epoch [49/50], Train Losses: mse: 8.1833, mae: 1.4669, huber: 1.0875, swd: 0.8677, target_std: 13.6404
    Epoch [49/50], Val Losses: mse: 24.1433, mae: 2.6417, huber: 2.2377, swd: 2.7707, target_std: 13.8286
    Epoch [49/50], Test Losses: mse: 18.5796, mae: 2.3076, huber: 1.9092, swd: 1.8703, target_std: 13.1386
      Epoch 49 composite train-obj: 1.087539
            Val objective improved 2.2606 → 2.2377, saving checkpoint.
    Epoch [50/50], Train Losses: mse: 8.1169, mae: 1.4559, huber: 1.0774, swd: 0.8498, target_std: 13.6375
    Epoch [50/50], Val Losses: mse: 25.4465, mae: 2.6794, huber: 2.2750, swd: 2.4046, target_std: 13.8286
    Epoch [50/50], Test Losses: mse: 19.1917, mae: 2.3327, huber: 1.9324, swd: 1.8038, target_std: 13.1386
      Epoch 50 composite train-obj: 1.077353
            No improvement (2.2750), counter 1/5
    Epoch [50/50], Test Losses: mse: 18.5796, mae: 2.3076, huber: 1.9092, swd: 1.8703, target_std: 13.1386
    Best round's Test MSE: 18.5796, MAE: 2.3076, SWD: 1.8703
    Best round's Validation MSE: 24.1433, MAE: 2.6417, SWD: 2.7707
    Best round's Test verification MSE : 18.5796, MAE: 2.3076, SWD: 1.8703
    Time taken: 491.01 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 58.6105, mae: 5.4502, huber: 4.9788, swd: 15.1259, target_std: 13.6421
    Epoch [1/50], Val Losses: mse: 56.2373, mae: 5.2190, huber: 4.7531, swd: 13.9365, target_std: 13.8286
    Epoch [1/50], Test Losses: mse: 52.8435, mae: 4.9905, huber: 4.5268, swd: 14.2139, target_std: 13.1386
      Epoch 1 composite train-obj: 4.978803
            Val objective improved inf → 4.7531, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 44.6247, mae: 4.3922, huber: 3.9351, swd: 11.1362, target_std: 13.6403
    Epoch [2/50], Val Losses: mse: 54.4012, mae: 4.9199, huber: 4.4616, swd: 10.5312, target_std: 13.8286
    Epoch [2/50], Test Losses: mse: 50.0637, mae: 4.6701, huber: 4.2137, swd: 10.5475, target_std: 13.1386
      Epoch 2 composite train-obj: 3.935150
            Val objective improved 4.7531 → 4.4616, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.9864, mae: 4.0091, huber: 3.5591, swd: 8.6690, target_std: 13.6396
    Epoch [3/50], Val Losses: mse: 51.8914, mae: 4.6909, huber: 4.2381, swd: 9.1170, target_std: 13.8286
    Epoch [3/50], Test Losses: mse: 46.1177, mae: 4.3638, huber: 3.9134, swd: 9.1619, target_std: 13.1386
      Epoch 3 composite train-obj: 3.559079
            Val objective improved 4.4616 → 4.2381, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.3600, mae: 3.7360, huber: 3.2909, swd: 7.1308, target_std: 13.6415
    Epoch [4/50], Val Losses: mse: 51.0153, mae: 4.5456, huber: 4.0954, swd: 7.3244, target_std: 13.8286
    Epoch [4/50], Test Losses: mse: 45.3280, mae: 4.2599, huber: 3.8109, swd: 7.2854, target_std: 13.1386
      Epoch 4 composite train-obj: 3.290895
            Val objective improved 4.2381 → 4.0954, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.6063, mae: 3.5224, huber: 3.0813, swd: 5.9966, target_std: 13.6415
    Epoch [5/50], Val Losses: mse: 49.6374, mae: 4.4186, huber: 3.9720, swd: 6.4257, target_std: 13.8286
    Epoch [5/50], Test Losses: mse: 43.2502, mae: 4.0902, huber: 3.6451, swd: 6.2548, target_std: 13.1386
      Epoch 5 composite train-obj: 3.081297
            Val objective improved 4.0954 → 3.9720, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.2184, mae: 3.3433, huber: 2.9056, swd: 5.1793, target_std: 13.6408
    Epoch [6/50], Val Losses: mse: 49.4410, mae: 4.3732, huber: 3.9279, swd: 6.0384, target_std: 13.8286
    Epoch [6/50], Test Losses: mse: 42.2225, mae: 4.0272, huber: 3.5839, swd: 5.9968, target_std: 13.1386
      Epoch 6 composite train-obj: 2.905641
            Val objective improved 3.9720 → 3.9279, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 29.2327, mae: 3.2001, huber: 2.7649, swd: 4.5797, target_std: 13.6399
    Epoch [7/50], Val Losses: mse: 49.1747, mae: 4.2966, huber: 3.8523, swd: 5.1840, target_std: 13.8286
    Epoch [7/50], Test Losses: mse: 41.2689, mae: 3.9408, huber: 3.4980, swd: 5.0468, target_std: 13.1386
      Epoch 7 composite train-obj: 2.764870
            Val objective improved 3.9279 → 3.8523, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 27.4854, mae: 3.0688, huber: 2.6363, swd: 4.1203, target_std: 13.6380
    Epoch [8/50], Val Losses: mse: 46.9875, mae: 4.1514, huber: 3.7105, swd: 5.1939, target_std: 13.8286
    Epoch [8/50], Test Losses: mse: 39.5408, mae: 3.7971, huber: 3.3581, swd: 4.9871, target_std: 13.1386
      Epoch 8 composite train-obj: 2.636295
            Val objective improved 3.8523 → 3.7105, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 25.8883, mae: 2.9478, huber: 2.5180, swd: 3.7151, target_std: 13.6408
    Epoch [9/50], Val Losses: mse: 45.7779, mae: 4.0549, huber: 3.6175, swd: 4.8973, target_std: 13.8286
    Epoch [9/50], Test Losses: mse: 38.3373, mae: 3.7019, huber: 3.2669, swd: 4.5761, target_std: 13.1386
      Epoch 9 composite train-obj: 2.518019
            Val objective improved 3.7105 → 3.6175, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 24.5232, mae: 2.8434, huber: 2.4159, swd: 3.4178, target_std: 13.6412
    Epoch [10/50], Val Losses: mse: 46.8723, mae: 4.0586, huber: 3.6214, swd: 4.1731, target_std: 13.8286
    Epoch [10/50], Test Losses: mse: 38.5831, mae: 3.6818, huber: 3.2463, swd: 3.8790, target_std: 13.1386
      Epoch 10 composite train-obj: 2.415924
            No improvement (3.6214), counter 1/5
    Epoch [11/50], Train Losses: mse: 23.3872, mae: 2.7536, huber: 2.3282, swd: 3.1518, target_std: 13.6430
    Epoch [11/50], Val Losses: mse: 44.8574, mae: 3.9662, huber: 3.5285, swd: 4.6957, target_std: 13.8286
    Epoch [11/50], Test Losses: mse: 37.1325, mae: 3.6126, huber: 3.1768, swd: 4.3431, target_std: 13.1386
      Epoch 11 composite train-obj: 2.328165
            Val objective improved 3.6175 → 3.5285, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 22.4061, mae: 2.6752, huber: 2.2518, swd: 2.9516, target_std: 13.6367
    Epoch [12/50], Val Losses: mse: 44.9556, mae: 3.9413, huber: 3.5065, swd: 3.9996, target_std: 13.8286
    Epoch [12/50], Test Losses: mse: 37.2562, mae: 3.5793, huber: 3.1469, swd: 3.5784, target_std: 13.1386
      Epoch 12 composite train-obj: 2.251812
            Val objective improved 3.5285 → 3.5065, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 21.5367, mae: 2.6022, huber: 2.1810, swd: 2.7569, target_std: 13.6405
    Epoch [13/50], Val Losses: mse: 43.9560, mae: 3.8550, huber: 3.4222, swd: 3.6872, target_std: 13.8286
    Epoch [13/50], Test Losses: mse: 36.2081, mae: 3.4837, huber: 3.0535, swd: 3.3934, target_std: 13.1386
      Epoch 13 composite train-obj: 2.180956
            Val objective improved 3.5065 → 3.4222, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 20.8170, mae: 2.5453, huber: 2.1258, swd: 2.6383, target_std: 13.6368
    Epoch [14/50], Val Losses: mse: 42.9811, mae: 3.8021, huber: 3.3691, swd: 3.5976, target_std: 13.8286
    Epoch [14/50], Test Losses: mse: 35.7958, mae: 3.4795, huber: 3.0480, swd: 3.3495, target_std: 13.1386
      Epoch 14 composite train-obj: 2.125814
            Val objective improved 3.4222 → 3.3691, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 20.0386, mae: 2.4830, huber: 2.0651, swd: 2.4921, target_std: 13.6403
    Epoch [15/50], Val Losses: mse: 43.3047, mae: 3.8033, huber: 3.3722, swd: 3.6538, target_std: 13.8286
    Epoch [15/50], Test Losses: mse: 35.5568, mae: 3.4214, huber: 2.9928, swd: 3.1889, target_std: 13.1386
      Epoch 15 composite train-obj: 2.065123
            No improvement (3.3722), counter 1/5
    Epoch [16/50], Train Losses: mse: 19.4197, mae: 2.4330, huber: 2.0168, swd: 2.3884, target_std: 13.6412
    Epoch [16/50], Val Losses: mse: 43.5614, mae: 3.8033, huber: 3.3705, swd: 3.3962, target_std: 13.8286
    Epoch [16/50], Test Losses: mse: 35.5423, mae: 3.4183, huber: 2.9882, swd: 3.0225, target_std: 13.1386
      Epoch 16 composite train-obj: 2.016821
            No improvement (3.3705), counter 2/5
    Epoch [17/50], Train Losses: mse: 18.8579, mae: 2.3867, huber: 1.9717, swd: 2.2910, target_std: 13.6388
    Epoch [17/50], Val Losses: mse: 42.1091, mae: 3.7053, huber: 3.2751, swd: 3.4451, target_std: 13.8286
    Epoch [17/50], Test Losses: mse: 34.3560, mae: 3.3591, huber: 2.9311, swd: 3.0914, target_std: 13.1386
      Epoch 17 composite train-obj: 1.971676
            Val objective improved 3.3691 → 3.2751, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 18.2695, mae: 2.3401, huber: 1.9268, swd: 2.2144, target_std: 13.6390
    Epoch [18/50], Val Losses: mse: 42.6311, mae: 3.7155, huber: 3.2871, swd: 3.0245, target_std: 13.8286
    Epoch [18/50], Test Losses: mse: 34.5153, mae: 3.3310, huber: 2.9044, swd: 2.6354, target_std: 13.1386
      Epoch 18 composite train-obj: 1.926772
            No improvement (3.2871), counter 1/5
    Epoch [19/50], Train Losses: mse: 17.7916, mae: 2.2966, huber: 1.8847, swd: 2.1270, target_std: 13.6387
    Epoch [19/50], Val Losses: mse: 41.8237, mae: 3.6697, huber: 3.2437, swd: 3.3901, target_std: 13.8286
    Epoch [19/50], Test Losses: mse: 33.6701, mae: 3.2833, huber: 2.8602, swd: 2.9702, target_std: 13.1386
      Epoch 19 composite train-obj: 1.884695
            Val objective improved 3.2751 → 3.2437, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 17.2885, mae: 2.2600, huber: 1.8492, swd: 2.0636, target_std: 13.6399
    Epoch [20/50], Val Losses: mse: 43.3319, mae: 3.7128, huber: 3.2863, swd: 3.2537, target_std: 13.8286
    Epoch [20/50], Test Losses: mse: 34.4559, mae: 3.3002, huber: 2.8769, swd: 2.9835, target_std: 13.1386
      Epoch 20 composite train-obj: 1.849190
            No improvement (3.2863), counter 1/5
    Epoch [21/50], Train Losses: mse: 17.0660, mae: 2.2411, huber: 1.8310, swd: 2.0321, target_std: 13.6413
    Epoch [21/50], Val Losses: mse: 41.7264, mae: 3.6280, huber: 3.2021, swd: 2.9602, target_std: 13.8286
    Epoch [21/50], Test Losses: mse: 33.7310, mae: 3.2526, huber: 2.8289, swd: 2.5682, target_std: 13.1386
      Epoch 21 composite train-obj: 1.831024
            Val objective improved 3.2437 → 3.2021, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 16.5294, mae: 2.2000, huber: 1.7911, swd: 1.9500, target_std: 13.6378
    Epoch [22/50], Val Losses: mse: 40.1835, mae: 3.5462, huber: 3.1232, swd: 3.3195, target_std: 13.8286
    Epoch [22/50], Test Losses: mse: 32.9685, mae: 3.1961, huber: 2.7760, swd: 2.7487, target_std: 13.1386
      Epoch 22 composite train-obj: 1.791066
            Val objective improved 3.2021 → 3.1232, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 16.1696, mae: 2.1680, huber: 1.7605, swd: 1.9091, target_std: 13.6411
    Epoch [23/50], Val Losses: mse: 42.2576, mae: 3.6319, huber: 3.2068, swd: 2.9204, target_std: 13.8286
    Epoch [23/50], Test Losses: mse: 33.3573, mae: 3.2141, huber: 2.7916, swd: 2.5009, target_std: 13.1386
      Epoch 23 composite train-obj: 1.760485
            No improvement (3.2068), counter 1/5
    Epoch [24/50], Train Losses: mse: 15.7300, mae: 2.1291, huber: 1.7231, swd: 1.8375, target_std: 13.6391
    Epoch [24/50], Val Losses: mse: 39.0187, mae: 3.4848, huber: 3.0644, swd: 3.0709, target_std: 13.8286
    Epoch [24/50], Test Losses: mse: 32.6334, mae: 3.1656, huber: 2.7476, swd: 2.6115, target_std: 13.1386
      Epoch 24 composite train-obj: 1.723104
            Val objective improved 3.1232 → 3.0644, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 15.4433, mae: 2.1080, huber: 1.7026, swd: 1.8066, target_std: 13.6420
    Epoch [25/50], Val Losses: mse: 39.1238, mae: 3.4835, huber: 3.0631, swd: 2.9897, target_std: 13.8286
    Epoch [25/50], Test Losses: mse: 32.3944, mae: 3.1460, huber: 2.7281, swd: 2.5685, target_std: 13.1386
      Epoch 25 composite train-obj: 1.702590
            Val objective improved 3.0644 → 3.0631, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 15.0218, mae: 2.0729, huber: 1.6688, swd: 1.7493, target_std: 13.6410
    Epoch [26/50], Val Losses: mse: 39.5145, mae: 3.5041, huber: 3.0815, swd: 3.0347, target_std: 13.8286
    Epoch [26/50], Test Losses: mse: 32.4580, mae: 3.1509, huber: 2.7308, swd: 2.4735, target_std: 13.1386
      Epoch 26 composite train-obj: 1.668781
            No improvement (3.0815), counter 1/5
    Epoch [27/50], Train Losses: mse: 14.7771, mae: 2.0521, huber: 1.6487, swd: 1.7077, target_std: 13.6403
    Epoch [27/50], Val Losses: mse: 40.3532, mae: 3.5149, huber: 3.0954, swd: 2.9669, target_std: 13.8286
    Epoch [27/50], Test Losses: mse: 31.7253, mae: 3.0948, huber: 2.6784, swd: 2.4956, target_std: 13.1386
      Epoch 27 composite train-obj: 1.648689
            No improvement (3.0954), counter 2/5
    Epoch [28/50], Train Losses: mse: 14.6514, mae: 2.0418, huber: 1.6388, swd: 1.6898, target_std: 13.6403
    Epoch [28/50], Val Losses: mse: 38.6860, mae: 3.4669, huber: 3.0448, swd: 3.1121, target_std: 13.8286
    Epoch [28/50], Test Losses: mse: 31.9312, mae: 3.1159, huber: 2.6963, swd: 2.4250, target_std: 13.1386
      Epoch 28 composite train-obj: 1.638823
            Val objective improved 3.0631 → 3.0448, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 14.3137, mae: 2.0114, huber: 1.6097, swd: 1.6428, target_std: 13.6406
    Epoch [29/50], Val Losses: mse: 38.7776, mae: 3.4453, huber: 3.0256, swd: 2.7325, target_std: 13.8286
    Epoch [29/50], Test Losses: mse: 32.1205, mae: 3.1060, huber: 2.6894, swd: 2.2904, target_std: 13.1386
      Epoch 29 composite train-obj: 1.609711
            Val objective improved 3.0448 → 3.0256, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 14.0725, mae: 1.9941, huber: 1.5930, swd: 1.6194, target_std: 13.6393
    Epoch [30/50], Val Losses: mse: 38.7984, mae: 3.4426, huber: 3.0245, swd: 3.1749, target_std: 13.8286
    Epoch [30/50], Test Losses: mse: 31.7121, mae: 3.0795, huber: 2.6642, swd: 2.5512, target_std: 13.1386
      Epoch 30 composite train-obj: 1.592951
            Val objective improved 3.0256 → 3.0245, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 13.7995, mae: 1.9702, huber: 1.5701, swd: 1.5823, target_std: 13.6377
    Epoch [31/50], Val Losses: mse: 37.0259, mae: 3.3477, huber: 2.9278, swd: 2.7926, target_std: 13.8286
    Epoch [31/50], Test Losses: mse: 31.2333, mae: 3.0455, huber: 2.6288, swd: 2.2210, target_std: 13.1386
      Epoch 31 composite train-obj: 1.570146
            Val objective improved 3.0245 → 2.9278, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 13.5361, mae: 1.9475, huber: 1.5484, swd: 1.5414, target_std: 13.6444
    Epoch [32/50], Val Losses: mse: 38.3859, mae: 3.3828, huber: 2.9676, swd: 2.8625, target_std: 13.8286
    Epoch [32/50], Test Losses: mse: 30.7862, mae: 3.0082, huber: 2.5960, swd: 2.3215, target_std: 13.1386
      Epoch 32 composite train-obj: 1.548434
            No improvement (2.9676), counter 1/5
    Epoch [33/50], Train Losses: mse: 13.3588, mae: 1.9307, huber: 1.5322, swd: 1.5201, target_std: 13.6408
    Epoch [33/50], Val Losses: mse: 37.8933, mae: 3.3637, huber: 2.9487, swd: 2.8964, target_std: 13.8286
    Epoch [33/50], Test Losses: mse: 31.6734, mae: 3.0456, huber: 2.6326, swd: 2.2764, target_std: 13.1386
      Epoch 33 composite train-obj: 1.532187
            No improvement (2.9487), counter 2/5
    Epoch [34/50], Train Losses: mse: 13.2035, mae: 1.9209, huber: 1.5227, swd: 1.4973, target_std: 13.6396
    Epoch [34/50], Val Losses: mse: 38.0531, mae: 3.4012, huber: 2.9849, swd: 3.0735, target_std: 13.8286
    Epoch [34/50], Test Losses: mse: 31.2140, mae: 3.0452, huber: 2.6320, swd: 2.3659, target_std: 13.1386
      Epoch 34 composite train-obj: 1.522703
            No improvement (2.9849), counter 3/5
    Epoch [35/50], Train Losses: mse: 12.9116, mae: 1.8937, huber: 1.4967, swd: 1.4641, target_std: 13.6387
    Epoch [35/50], Val Losses: mse: 37.8823, mae: 3.3684, huber: 2.9533, swd: 2.9304, target_std: 13.8286
    Epoch [35/50], Test Losses: mse: 30.9059, mae: 3.0070, huber: 2.5949, swd: 2.2384, target_std: 13.1386
      Epoch 35 composite train-obj: 1.496691
            No improvement (2.9533), counter 4/5
    Epoch [36/50], Train Losses: mse: 12.7692, mae: 1.8807, huber: 1.4841, swd: 1.4489, target_std: 13.6380
    Epoch [36/50], Val Losses: mse: 40.3292, mae: 3.4749, huber: 3.0570, swd: 2.7039, target_std: 13.8286
    Epoch [36/50], Test Losses: mse: 31.6835, mae: 3.0574, huber: 2.6419, swd: 2.0379, target_std: 13.1386
      Epoch 36 composite train-obj: 1.484131
    Epoch [36/50], Test Losses: mse: 31.2333, mae: 3.0455, huber: 2.6288, swd: 2.2210, target_std: 13.1386
    Best round's Test MSE: 31.2333, MAE: 3.0455, SWD: 2.2210
    Best round's Validation MSE: 37.0259, MAE: 3.3477, SWD: 2.7926
    Best round's Test verification MSE : 31.2333, MAE: 3.0455, SWD: 2.2210
    Time taken: 335.83 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq336_pred336_20250507_1312)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 27.3345 ± 6.2030
      mae: 2.8132 ± 0.3579
      huber: 2.4036 ± 0.3501
      swd: 2.0949 ± 0.1593
      target_std: 13.1386 ± 0.0000
      count: 36.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 33.1037 ± 6.3523
      mae: 3.1279 ± 0.3443
      huber: 2.7145 ± 0.3378
      swd: 2.7180 ± 0.0905
      target_std: 13.8286 ± 0.0000
      count: 36.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 1264.47 seconds
    
    Experiment complete: TimeMixer_lorenz_seq336_pred336_20250507_1312
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 336-720
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
    pred_len=720,
    channels=data_mgr.datasets['lorenz']['channels'],
    enc_in=data_mgr.datasets['lorenz']['channels'],
    dec_in=data_mgr.datasets['lorenz']['channels'],
    c_out=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0]
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 277
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 277
    Validation Batches: 33
    Test Batches: 74
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 68.6446, mae: 6.1935, huber: 5.7156, swd: 19.1728, ept: 71.1376
    Epoch [1/50], Val Losses: mse: 63.0775, mae: 5.8672, huber: 5.3922, swd: 20.6474, ept: 93.2239
    Epoch [1/50], Test Losses: mse: 62.4516, mae: 5.7918, huber: 5.3180, swd: 20.7476, ept: 89.0802
      Epoch 1 composite train-obj: 5.715581
            Val objective improved inf → 5.3922, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 56.9107, mae: 5.4247, huber: 4.9536, swd: 16.5753, ept: 127.3354
    Epoch [2/50], Val Losses: mse: 63.8703, mae: 5.8037, huber: 5.3301, swd: 14.8790, ept: 122.1551
    Epoch [2/50], Test Losses: mse: 62.2693, mae: 5.6655, huber: 5.1935, swd: 14.7669, ept: 119.5490
      Epoch 2 composite train-obj: 4.953595
            Val objective improved 5.3922 → 5.3301, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 52.9951, mae: 5.1210, huber: 4.6538, swd: 13.6162, ept: 151.8715
    Epoch [3/50], Val Losses: mse: 63.8004, mae: 5.7213, huber: 5.2504, swd: 12.4642, ept: 141.7457
    Epoch [3/50], Test Losses: mse: 62.2304, mae: 5.6036, huber: 5.1339, swd: 12.6793, ept: 135.8685
      Epoch 3 composite train-obj: 4.653770
            Val objective improved 5.3301 → 5.2504, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 50.0676, mae: 4.8912, huber: 4.4272, swd: 11.5666, ept: 169.9703
    Epoch [4/50], Val Losses: mse: 64.8931, mae: 5.7222, huber: 5.2530, swd: 10.8689, ept: 145.8786
    Epoch [4/50], Test Losses: mse: 62.5832, mae: 5.5698, huber: 5.1018, swd: 10.8657, ept: 146.0425
      Epoch 4 composite train-obj: 4.427195
            No improvement (5.2530), counter 1/5
    Epoch [5/50], Train Losses: mse: 47.7216, mae: 4.7078, huber: 4.2462, swd: 9.9635, ept: 184.9326
    Epoch [5/50], Val Losses: mse: 66.1512, mae: 5.7166, huber: 5.2493, swd: 9.2952, ept: 159.8321
    Epoch [5/50], Test Losses: mse: 62.9047, mae: 5.5276, huber: 5.0616, swd: 9.6538, ept: 158.7799
      Epoch 5 composite train-obj: 4.246242
            Val objective improved 5.2504 → 5.2493, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 45.5971, mae: 4.5427, huber: 4.0833, swd: 8.7106, ept: 198.5630
    Epoch [6/50], Val Losses: mse: 66.5520, mae: 5.7050, huber: 5.2370, swd: 8.1389, ept: 168.5779
    Epoch [6/50], Test Losses: mse: 62.9540, mae: 5.5090, huber: 5.0426, swd: 8.4251, ept: 169.1697
      Epoch 6 composite train-obj: 4.083332
            Val objective improved 5.2493 → 5.2370, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 43.7184, mae: 4.3997, huber: 3.9422, swd: 7.6833, ept: 210.7370
    Epoch [7/50], Val Losses: mse: 67.2689, mae: 5.6981, huber: 5.2321, swd: 7.1993, ept: 175.6248
    Epoch [7/50], Test Losses: mse: 63.1448, mae: 5.4865, huber: 5.0217, swd: 7.4389, ept: 178.7126
      Epoch 7 composite train-obj: 3.942183
            Val objective improved 5.2370 → 5.2321, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 42.0526, mae: 4.2751, huber: 3.8193, swd: 6.9341, ept: 222.0537
    Epoch [8/50], Val Losses: mse: 68.3180, mae: 5.7285, huber: 5.2623, swd: 6.3055, ept: 182.3249
    Epoch [8/50], Test Losses: mse: 64.5655, mae: 5.5393, huber: 5.0744, swd: 6.6502, ept: 185.1827
      Epoch 8 composite train-obj: 3.819311
            No improvement (5.2623), counter 1/5
    Epoch [9/50], Train Losses: mse: 40.6531, mae: 4.1718, huber: 3.7174, swd: 6.3468, ept: 232.6155
    Epoch [9/50], Val Losses: mse: 68.2333, mae: 5.6899, huber: 5.2250, swd: 6.1879, ept: 186.7826
    Epoch [9/50], Test Losses: mse: 63.9665, mae: 5.4753, huber: 5.0120, swd: 6.4937, ept: 190.7238
      Epoch 9 composite train-obj: 3.717414
            Val objective improved 5.2321 → 5.2250, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 39.2871, mae: 4.0673, huber: 3.6144, swd: 5.8445, ept: 243.2712
    Epoch [10/50], Val Losses: mse: 69.4870, mae: 5.6932, huber: 5.2295, swd: 5.3627, ept: 193.4380
    Epoch [10/50], Test Losses: mse: 64.5856, mae: 5.4693, huber: 5.0071, swd: 5.8744, ept: 197.6596
      Epoch 10 composite train-obj: 3.614431
            No improvement (5.2295), counter 1/5
    Epoch [11/50], Train Losses: mse: 38.0985, mae: 3.9791, huber: 3.5273, swd: 5.4627, ept: 252.7673
    Epoch [11/50], Val Losses: mse: 68.6649, mae: 5.6500, huber: 5.1865, swd: 5.2522, ept: 199.4960
    Epoch [11/50], Test Losses: mse: 64.0780, mae: 5.4449, huber: 4.9826, swd: 5.6664, ept: 202.5507
      Epoch 11 composite train-obj: 3.527332
            Val objective improved 5.2250 → 5.1865, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 37.1719, mae: 3.9107, huber: 3.4600, swd: 5.1676, ept: 259.6457
    Epoch [12/50], Val Losses: mse: 70.1967, mae: 5.6963, huber: 5.2331, swd: 4.8547, ept: 201.7499
    Epoch [12/50], Test Losses: mse: 65.4837, mae: 5.4745, huber: 5.0129, swd: 5.1990, ept: 205.0436
      Epoch 12 composite train-obj: 3.459954
            No improvement (5.2331), counter 1/5
    Epoch [13/50], Train Losses: mse: 36.2779, mae: 3.8433, huber: 3.3937, swd: 4.9105, ept: 267.2467
    Epoch [13/50], Val Losses: mse: 69.5914, mae: 5.6711, huber: 5.2080, swd: 4.8860, ept: 203.4373
    Epoch [13/50], Test Losses: mse: 64.4314, mae: 5.4287, huber: 4.9670, swd: 5.2670, ept: 208.6606
      Epoch 13 composite train-obj: 3.393674
            No improvement (5.2080), counter 2/5
    Epoch [14/50], Train Losses: mse: 35.4277, mae: 3.7810, huber: 3.3322, swd: 4.6744, ept: 273.9160
    Epoch [14/50], Val Losses: mse: 71.3920, mae: 5.7293, huber: 5.2661, swd: 4.2927, ept: 205.3542
    Epoch [14/50], Test Losses: mse: 65.9329, mae: 5.4724, huber: 5.0112, swd: 4.6678, ept: 210.4525
      Epoch 14 composite train-obj: 3.332242
            No improvement (5.2661), counter 3/5
    Epoch [15/50], Train Losses: mse: 34.7351, mae: 3.7292, huber: 3.2813, swd: 4.4869, ept: 279.2661
    Epoch [15/50], Val Losses: mse: 71.2961, mae: 5.7191, huber: 5.2572, swd: 4.2669, ept: 206.9692
    Epoch [15/50], Test Losses: mse: 66.0787, mae: 5.4711, huber: 5.0108, swd: 4.4956, ept: 213.8714
      Epoch 15 composite train-obj: 3.281304
            No improvement (5.2572), counter 4/5
    Epoch [16/50], Train Losses: mse: 34.0049, mae: 3.6780, huber: 3.2310, swd: 4.3482, ept: 284.7386
    Epoch [16/50], Val Losses: mse: 71.8776, mae: 5.7228, huber: 5.2609, swd: 4.1197, ept: 210.5899
    Epoch [16/50], Test Losses: mse: 66.6490, mae: 5.4786, huber: 5.0182, swd: 4.3088, ept: 218.2586
      Epoch 16 composite train-obj: 3.230969
    Epoch [16/50], Test Losses: mse: 64.0780, mae: 5.4449, huber: 4.9826, swd: 5.6664, ept: 202.5507
    Best round's Test MSE: 64.0780, MAE: 5.4449, SWD: 5.6664
    Best round's Validation MSE: 68.6649, MAE: 5.6500, SWD: 5.2522
    Best round's Test verification MSE : 64.0780, MAE: 5.4449, SWD: 5.6664
    Time taken: 179.43 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 67.0765, mae: 6.1001, huber: 5.6228, swd: 18.8556, ept: 75.8914
    Epoch [1/50], Val Losses: mse: 62.8255, mae: 5.8607, huber: 5.3856, swd: 19.3718, ept: 100.2553
    Epoch [1/50], Test Losses: mse: 62.2505, mae: 5.7894, huber: 5.3152, swd: 19.7179, ept: 97.4682
      Epoch 1 composite train-obj: 5.622777
            Val objective improved inf → 5.3856, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 55.3099, mae: 5.3090, huber: 4.8392, swd: 15.4777, ept: 136.0310
    Epoch [2/50], Val Losses: mse: 63.2091, mae: 5.7227, huber: 5.2511, swd: 13.8794, ept: 132.4365
    Epoch [2/50], Test Losses: mse: 61.7674, mae: 5.6200, huber: 5.1491, swd: 13.9564, ept: 128.5689
      Epoch 2 composite train-obj: 4.839177
            Val objective improved 5.3856 → 5.2511, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 51.0217, mae: 4.9654, huber: 4.5003, swd: 12.0479, ept: 165.1340
    Epoch [3/50], Val Losses: mse: 64.0445, mae: 5.6886, huber: 5.2195, swd: 11.4043, ept: 147.3530
    Epoch [3/50], Test Losses: mse: 61.6406, mae: 5.5396, huber: 5.0718, swd: 11.4645, ept: 145.2747
      Epoch 3 composite train-obj: 4.500273
            Val objective improved 5.2511 → 5.2195, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 48.0078, mae: 4.7304, huber: 4.2686, swd: 10.0483, ept: 183.7173
    Epoch [4/50], Val Losses: mse: 65.8785, mae: 5.7409, huber: 5.2726, swd: 9.6728, ept: 155.9699
    Epoch [4/50], Test Losses: mse: 62.6175, mae: 5.5491, huber: 5.0818, swd: 9.7810, ept: 157.1864
      Epoch 4 composite train-obj: 4.268642
            No improvement (5.2726), counter 1/5
    Epoch [5/50], Train Losses: mse: 45.5577, mae: 4.5414, huber: 4.0824, swd: 8.6103, ept: 198.9781
    Epoch [5/50], Val Losses: mse: 65.5559, mae: 5.6768, huber: 5.2097, swd: 8.2514, ept: 168.6111
    Epoch [5/50], Test Losses: mse: 62.2971, mae: 5.4859, huber: 5.0202, swd: 8.3837, ept: 170.7184
      Epoch 5 composite train-obj: 4.082395
            Val objective improved 5.2195 → 5.2097, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 43.4553, mae: 4.3829, huber: 3.9261, swd: 7.5600, ept: 212.8942
    Epoch [6/50], Val Losses: mse: 67.1842, mae: 5.7011, huber: 5.2346, swd: 6.7077, ept: 177.6612
    Epoch [6/50], Test Losses: mse: 63.4748, mae: 5.5069, huber: 5.0415, swd: 6.9741, ept: 182.3158
      Epoch 6 composite train-obj: 3.926058
            No improvement (5.2346), counter 1/5
    Epoch [7/50], Train Losses: mse: 41.5177, mae: 4.2383, huber: 3.7835, swd: 6.7125, ept: 225.8738
    Epoch [7/50], Val Losses: mse: 68.2082, mae: 5.7027, huber: 5.2372, swd: 5.9406, ept: 183.4089
    Epoch [7/50], Test Losses: mse: 64.2653, mae: 5.5205, huber: 5.0557, swd: 6.4532, ept: 187.0164
      Epoch 7 composite train-obj: 3.783462
            No improvement (5.2372), counter 2/5
    Epoch [8/50], Train Losses: mse: 39.9183, mae: 4.1197, huber: 3.6664, swd: 6.0530, ept: 238.1394
    Epoch [8/50], Val Losses: mse: 68.0529, mae: 5.6593, huber: 5.1948, swd: 5.4709, ept: 193.6439
    Epoch [8/50], Test Losses: mse: 63.9183, mae: 5.4579, huber: 4.9946, swd: 5.8985, ept: 196.7906
      Epoch 8 composite train-obj: 3.666406
            Val objective improved 5.2097 → 5.1948, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 38.4235, mae: 4.0088, huber: 3.5571, swd: 5.5300, ept: 249.5020
    Epoch [9/50], Val Losses: mse: 68.5189, mae: 5.6693, huber: 5.2052, swd: 5.2239, ept: 194.8404
    Epoch [9/50], Test Losses: mse: 63.9394, mae: 5.4628, huber: 4.9997, swd: 5.6233, ept: 200.7234
      Epoch 9 composite train-obj: 3.557066
            No improvement (5.2052), counter 1/5
    Epoch [10/50], Train Losses: mse: 37.2014, mae: 3.9138, huber: 3.4637, swd: 5.1095, ept: 259.9033
    Epoch [10/50], Val Losses: mse: 68.7650, mae: 5.6646, huber: 5.2009, swd: 5.0671, ept: 199.5413
    Epoch [10/50], Test Losses: mse: 64.0804, mae: 5.4524, huber: 4.9895, swd: 5.5156, ept: 204.3350
      Epoch 10 composite train-obj: 3.463668
            No improvement (5.2009), counter 2/5
    Epoch [11/50], Train Losses: mse: 36.1178, mae: 3.8349, huber: 3.3858, swd: 4.7973, ept: 268.9592
    Epoch [11/50], Val Losses: mse: 69.5927, mae: 5.6650, huber: 5.2015, swd: 4.2005, ept: 205.2355
    Epoch [11/50], Test Losses: mse: 65.0474, mae: 5.4639, huber: 5.0015, swd: 4.7101, ept: 212.1822
      Epoch 11 composite train-obj: 3.385767
            No improvement (5.2015), counter 3/5
    Epoch [12/50], Train Losses: mse: 35.1792, mae: 3.7612, huber: 3.3133, swd: 4.5121, ept: 276.9255
    Epoch [12/50], Val Losses: mse: 69.6740, mae: 5.6470, huber: 5.1851, swd: 4.2789, ept: 209.3617
    Epoch [12/50], Test Losses: mse: 64.4838, mae: 5.4073, huber: 4.9471, swd: 4.6848, ept: 215.8387
      Epoch 12 composite train-obj: 3.313290
            Val objective improved 5.1948 → 5.1851, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 34.2886, mae: 3.6974, huber: 3.2505, swd: 4.2960, ept: 284.0989
    Epoch [13/50], Val Losses: mse: 70.6752, mae: 5.6639, huber: 5.2023, swd: 3.8378, ept: 212.1483
    Epoch [13/50], Test Losses: mse: 65.5936, mae: 5.4455, huber: 4.9851, swd: 4.2224, ept: 219.5117
      Epoch 13 composite train-obj: 3.250483
            No improvement (5.2023), counter 1/5
    Epoch [14/50], Train Losses: mse: 33.5802, mae: 3.6441, huber: 3.1981, swd: 4.1297, ept: 290.2423
    Epoch [14/50], Val Losses: mse: 71.0473, mae: 5.6925, huber: 5.2304, swd: 3.7860, ept: 213.5348
    Epoch [14/50], Test Losses: mse: 66.1155, mae: 5.4750, huber: 5.0142, swd: 4.2437, ept: 218.9076
      Epoch 14 composite train-obj: 3.198057
            No improvement (5.2304), counter 2/5
    Epoch [15/50], Train Losses: mse: 32.8947, mae: 3.5903, huber: 3.1453, swd: 3.9698, ept: 296.0807
    Epoch [15/50], Val Losses: mse: 70.7148, mae: 5.6713, huber: 5.2098, swd: 3.7513, ept: 218.1780
    Epoch [15/50], Test Losses: mse: 65.7794, mae: 5.4404, huber: 4.9806, swd: 4.1334, ept: 222.9495
      Epoch 15 composite train-obj: 3.145252
            No improvement (5.2098), counter 3/5
    Epoch [16/50], Train Losses: mse: 32.2614, mae: 3.5460, huber: 3.1017, swd: 3.8380, ept: 300.7770
    Epoch [16/50], Val Losses: mse: 72.3140, mae: 5.6922, huber: 5.2311, swd: 3.3911, ept: 217.7390
    Epoch [16/50], Test Losses: mse: 66.5550, mae: 5.4373, huber: 4.9784, swd: 3.8313, ept: 224.8517
      Epoch 16 composite train-obj: 3.101651
            No improvement (5.2311), counter 4/5
    Epoch [17/50], Train Losses: mse: 31.6222, mae: 3.4977, huber: 3.0541, swd: 3.7005, ept: 306.3821
    Epoch [17/50], Val Losses: mse: 72.1051, mae: 5.6923, huber: 5.2319, swd: 3.5154, ept: 219.1911
    Epoch [17/50], Test Losses: mse: 66.1015, mae: 5.4167, huber: 4.9586, swd: 3.8420, ept: 227.2637
      Epoch 17 composite train-obj: 3.054106
    Epoch [17/50], Test Losses: mse: 64.4838, mae: 5.4073, huber: 4.9471, swd: 4.6848, ept: 215.8387
    Best round's Test MSE: 64.4838, MAE: 5.4073, SWD: 4.6848
    Best round's Validation MSE: 69.6740, MAE: 5.6470, SWD: 4.2789
    Best round's Test verification MSE : 64.4838, MAE: 5.4073, SWD: 4.6848
    Time taken: 185.44 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 65.6430, mae: 6.0050, huber: 5.5287, swd: 18.5021, ept: 83.1946
    Epoch [1/50], Val Losses: mse: 64.1809, mae: 5.8665, huber: 5.3923, swd: 17.7335, ept: 108.9176
    Epoch [1/50], Test Losses: mse: 62.5456, mae: 5.7338, huber: 5.2611, swd: 17.4540, ept: 105.5278
      Epoch 1 composite train-obj: 5.528660
            Val objective improved inf → 5.3923, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 54.8191, mae: 5.2465, huber: 4.7782, swd: 14.7582, ept: 141.0142
    Epoch [2/50], Val Losses: mse: 63.8076, mae: 5.7822, huber: 5.3105, swd: 14.5655, ept: 126.3028
    Epoch [2/50], Test Losses: mse: 62.1862, mae: 5.6700, huber: 5.1990, swd: 14.5992, ept: 124.3660
      Epoch 2 composite train-obj: 4.778193
            Val objective improved 5.3923 → 5.3105, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 51.3646, mae: 4.9758, huber: 4.5113, swd: 12.4376, ept: 163.6102
    Epoch [3/50], Val Losses: mse: 64.8742, mae: 5.7693, huber: 5.2986, swd: 12.2530, ept: 139.4935
    Epoch [3/50], Test Losses: mse: 62.6269, mae: 5.6229, huber: 5.1531, swd: 12.2017, ept: 138.2602
      Epoch 3 composite train-obj: 4.511285
            Val objective improved 5.3105 → 5.2986, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 48.8891, mae: 4.7855, huber: 4.3237, swd: 10.8635, ept: 179.9013
    Epoch [4/50], Val Losses: mse: 65.0838, mae: 5.6827, huber: 5.2149, swd: 10.8528, ept: 151.9368
    Epoch [4/50], Test Losses: mse: 62.3109, mae: 5.5282, huber: 5.0618, swd: 10.8034, ept: 149.1055
      Epoch 4 composite train-obj: 4.323666
            Val objective improved 5.2986 → 5.2149, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 46.9586, mae: 4.6390, huber: 4.1791, swd: 9.7321, ept: 192.2979
    Epoch [5/50], Val Losses: mse: 65.0857, mae: 5.6999, huber: 5.2315, swd: 10.2754, ept: 160.6067
    Epoch [5/50], Test Losses: mse: 61.7552, mae: 5.4985, huber: 5.0317, swd: 10.1572, ept: 160.6409
      Epoch 5 composite train-obj: 4.179092
            No improvement (5.2315), counter 1/5
    Epoch [6/50], Train Losses: mse: 45.2675, mae: 4.5158, huber: 4.0573, swd: 8.7996, ept: 202.8396
    Epoch [6/50], Val Losses: mse: 65.9531, mae: 5.6833, huber: 5.2162, swd: 8.8714, ept: 168.0884
    Epoch [6/50], Test Losses: mse: 62.7040, mae: 5.5057, huber: 5.0402, swd: 8.8064, ept: 168.5633
      Epoch 6 composite train-obj: 4.057310
            No improvement (5.2162), counter 2/5
    Epoch [7/50], Train Losses: mse: 43.7895, mae: 4.4062, huber: 3.9491, swd: 8.0312, ept: 212.6189
    Epoch [7/50], Val Losses: mse: 68.4382, mae: 5.7573, huber: 5.2903, swd: 7.2934, ept: 173.2717
    Epoch [7/50], Test Losses: mse: 64.1348, mae: 5.5294, huber: 5.0640, swd: 7.4443, ept: 177.2116
      Epoch 7 composite train-obj: 3.949103
            No improvement (5.2903), counter 3/5
    Epoch [8/50], Train Losses: mse: 42.5364, mae: 4.3108, huber: 3.8550, swd: 7.4273, ept: 220.6254
    Epoch [8/50], Val Losses: mse: 68.2176, mae: 5.7356, huber: 5.2686, swd: 7.1886, ept: 178.8120
    Epoch [8/50], Test Losses: mse: 63.9663, mae: 5.5267, huber: 5.0611, swd: 7.3095, ept: 180.4473
      Epoch 8 composite train-obj: 3.854952
            No improvement (5.2686), counter 4/5
    Epoch [9/50], Train Losses: mse: 41.3045, mae: 4.2211, huber: 3.7664, swd: 6.9167, ept: 229.0343
    Epoch [9/50], Val Losses: mse: 67.8350, mae: 5.6867, huber: 5.2214, swd: 7.0528, ept: 181.2080
    Epoch [9/50], Test Losses: mse: 63.3708, mae: 5.4722, huber: 5.0086, swd: 7.3114, ept: 186.6019
      Epoch 9 composite train-obj: 3.766366
    Epoch [9/50], Test Losses: mse: 62.3109, mae: 5.5282, huber: 5.0618, swd: 10.8034, ept: 149.1055
    Best round's Test MSE: 62.3109, MAE: 5.5282, SWD: 10.8034
    Best round's Validation MSE: 65.0838, MAE: 5.6827, SWD: 10.8528
    Best round's Test verification MSE : 62.3109, MAE: 5.5282, SWD: 10.8034
    Time taken: 97.31 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq336_pred720_20250513_2300)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 63.6242 ± 0.9433
      mae: 5.4601 ± 0.0505
      huber: 4.9972 ± 0.0480
      swd: 7.0515 ± 2.6831
      ept: 189.1650 ± 28.8411
      count: 33.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 67.8076 ± 1.9695
      mae: 5.6599 ± 0.0162
      huber: 5.1955 ± 0.0137
      swd: 6.7946 ± 2.8970
      ept: 186.9315 ± 25.0706
      count: 33.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 462.26 seconds
    
    Experiment complete: TimeMixer_lorenz_seq336_pred720_20250513_2300
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### PatchTST

#### 336-96
##### huber


```python
utils.reload_modules([utils])
cfg_patch_tst = train_config.FlatPatchTSTConfig(
    seq_len=336,
    pred_len=96,
    channels=data_mgr.datasets['lorenz']['channels'],
    enc_in=data_mgr.datasets['lorenz']['channels'],
    dec_in=data_mgr.datasets['lorenz']['channels'],
    c_out=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0]
)
exp_patch_tst = execute_model_evaluation('lorenz', cfg_patch_tst, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 282
    Validation Batches: 38
    Test Batches: 78
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 44.2488, mae: 4.6542, huber: 4.1852, swd: 7.9709, ept: 30.1192
    Epoch [1/50], Val Losses: mse: 50.9703, mae: 5.1308, huber: 4.6572, swd: 8.5287, ept: 46.1557
    Epoch [1/50], Test Losses: mse: 49.4988, mae: 5.0377, huber: 4.5654, swd: 8.3166, ept: 47.0484
      Epoch 1 composite train-obj: 4.185248
            Val objective improved inf → 4.6572, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 28.3161, mae: 3.4744, huber: 3.0194, swd: 5.2538, ept: 35.1235
    Epoch [2/50], Val Losses: mse: 29.6114, mae: 3.8806, huber: 3.4145, swd: 6.1323, ept: 67.9112
    Epoch [2/50], Test Losses: mse: 27.6661, mae: 3.7783, huber: 3.3125, swd: 5.7380, ept: 69.1151
      Epoch 2 composite train-obj: 3.019401
            Val objective improved 4.6572 → 3.4145, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.7018, mae: 2.9738, huber: 2.5285, swd: 4.2315, ept: 36.3479
    Epoch [3/50], Val Losses: mse: 17.2588, mae: 2.6675, huber: 2.2198, swd: 4.0867, ept: 84.3479
    Epoch [3/50], Test Losses: mse: 15.0625, mae: 2.5480, huber: 2.1010, swd: 3.7945, ept: 85.4405
      Epoch 3 composite train-obj: 2.528518
            Val objective improved 3.4145 → 2.2198, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 19.9484, mae: 2.6963, huber: 2.2591, swd: 3.5489, ept: 36.5684
    Epoch [4/50], Val Losses: mse: 15.0062, mae: 2.5481, huber: 2.1005, swd: 4.1063, ept: 86.7011
    Epoch [4/50], Test Losses: mse: 13.1651, mae: 2.4381, huber: 1.9925, swd: 3.5514, ept: 87.5394
      Epoch 4 composite train-obj: 2.259062
            Val objective improved 2.2198 → 2.1005, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 18.4116, mae: 2.5231, huber: 2.0919, swd: 3.1766, ept: 36.7840
    Epoch [5/50], Val Losses: mse: 23.8780, mae: 3.0454, huber: 2.5919, swd: 3.0871, ept: 83.2314
    Epoch [5/50], Test Losses: mse: 22.7321, mae: 3.0602, huber: 2.6058, swd: 3.3559, ept: 82.7952
      Epoch 5 composite train-obj: 2.091911
            No improvement (2.5919), counter 1/5
    Epoch [6/50], Train Losses: mse: 18.0751, mae: 2.4686, huber: 2.0406, swd: 3.1338, ept: 37.0114
    Epoch [6/50], Val Losses: mse: 23.1995, mae: 2.7904, huber: 2.3474, swd: 3.1949, ept: 84.4669
    Epoch [6/50], Test Losses: mse: 21.1748, mae: 2.7423, huber: 2.2985, swd: 3.6290, ept: 84.8665
      Epoch 6 composite train-obj: 2.040640
            No improvement (2.3474), counter 2/5
    Epoch [7/50], Train Losses: mse: 16.8493, mae: 2.3317, huber: 1.9091, swd: 2.7278, ept: 37.1227
    Epoch [7/50], Val Losses: mse: 11.3407, mae: 1.9465, huber: 1.5208, swd: 2.7548, ept: 90.2353
    Epoch [7/50], Test Losses: mse: 8.9934, mae: 1.8416, huber: 1.4166, swd: 2.5030, ept: 91.1984
      Epoch 7 composite train-obj: 1.909086
            Val objective improved 2.1005 → 1.5208, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 15.8423, mae: 2.2131, huber: 1.7959, swd: 2.5928, ept: 37.3802
    Epoch [8/50], Val Losses: mse: 8.7051, mae: 1.6535, huber: 1.2346, swd: 2.9970, ept: 90.7381
    Epoch [8/50], Test Losses: mse: 6.7560, mae: 1.5494, huber: 1.1320, swd: 2.4605, ept: 91.9553
      Epoch 8 composite train-obj: 1.795895
            Val objective improved 1.5208 → 1.2346, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 14.9369, mae: 2.0959, huber: 1.6857, swd: 2.3511, ept: 37.1802
    Epoch [9/50], Val Losses: mse: 9.7057, mae: 1.8966, huber: 1.4618, swd: 2.6256, ept: 90.2566
    Epoch [9/50], Test Losses: mse: 7.9659, mae: 1.8261, huber: 1.3918, swd: 2.5545, ept: 91.4104
      Epoch 9 composite train-obj: 1.685661
            No improvement (1.4618), counter 1/5
    Epoch [10/50], Train Losses: mse: 14.7453, mae: 2.0741, huber: 1.6653, swd: 2.2841, ept: 37.3303
    Epoch [10/50], Val Losses: mse: 10.0444, mae: 2.0524, huber: 1.6109, swd: 2.2601, ept: 90.3851
    Epoch [10/50], Test Losses: mse: 8.2546, mae: 1.9770, huber: 1.5351, swd: 1.8423, ept: 91.2808
      Epoch 10 composite train-obj: 1.665286
            No improvement (1.6109), counter 2/5
    Epoch [11/50], Train Losses: mse: 14.5479, mae: 2.0459, huber: 1.6395, swd: 2.2294, ept: 37.1893
    Epoch [11/50], Val Losses: mse: 10.3791, mae: 1.8558, huber: 1.4354, swd: 2.4749, ept: 89.2094
    Epoch [11/50], Test Losses: mse: 8.7101, mae: 1.7568, huber: 1.3398, swd: 1.9934, ept: 89.8245
      Epoch 11 composite train-obj: 1.639450
            No improvement (1.4354), counter 3/5
    Epoch [12/50], Train Losses: mse: 14.4576, mae: 2.0225, huber: 1.6186, swd: 2.2455, ept: 37.4598
    Epoch [12/50], Val Losses: mse: 22.1544, mae: 2.5970, huber: 2.1544, swd: 2.3790, ept: 84.5110
    Epoch [12/50], Test Losses: mse: 23.1636, mae: 2.6923, huber: 2.2488, swd: 2.6004, ept: 83.7719
      Epoch 12 composite train-obj: 1.618620
            No improvement (2.1544), counter 4/5
    Epoch [13/50], Train Losses: mse: 15.2847, mae: 2.0987, huber: 1.6917, swd: 2.3408, ept: 37.2535
    Epoch [13/50], Val Losses: mse: 12.2453, mae: 2.1164, huber: 1.6754, swd: 2.4219, ept: 90.4893
    Epoch [13/50], Test Losses: mse: 9.6675, mae: 2.0108, huber: 1.5695, swd: 2.2445, ept: 91.5885
      Epoch 13 composite train-obj: 1.691689
    Epoch [13/50], Test Losses: mse: 6.7560, mae: 1.5494, huber: 1.1320, swd: 2.4605, ept: 91.9553
    Best round's Test MSE: 6.7560, MAE: 1.5494, SWD: 2.4605
    Best round's Validation MSE: 8.7051, MAE: 1.6535, SWD: 2.9970
    Best round's Test verification MSE : 6.7560, MAE: 1.5494, SWD: 2.4605
    Time taken: 70.77 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 45.5751, mae: 4.7349, huber: 4.2654, swd: 8.4750, ept: 29.8742
    Epoch [1/50], Val Losses: mse: 52.7440, mae: 5.4398, huber: 4.9615, swd: 13.3217, ept: 44.8936
    Epoch [1/50], Test Losses: mse: 49.7407, mae: 5.2312, huber: 4.7542, swd: 12.9381, ept: 47.7225
      Epoch 1 composite train-obj: 4.265392
            Val objective improved inf → 4.9615, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 28.7428, mae: 3.5245, huber: 3.0685, swd: 5.8339, ept: 34.7883
    Epoch [2/50], Val Losses: mse: 36.7226, mae: 4.0320, huber: 3.5658, swd: 6.8587, ept: 68.3201
    Epoch [2/50], Test Losses: mse: 33.2786, mae: 3.9164, huber: 3.4506, swd: 6.7604, ept: 68.1708
      Epoch 2 composite train-obj: 3.068520
            Val objective improved 4.9615 → 3.5658, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 23.0915, mae: 2.9997, huber: 2.5543, swd: 4.3392, ept: 36.0755
    Epoch [3/50], Val Losses: mse: 19.3205, mae: 3.0374, huber: 2.5802, swd: 6.2578, ept: 81.1404
    Epoch [3/50], Test Losses: mse: 17.7452, mae: 2.9179, huber: 2.4629, swd: 5.5352, ept: 81.9062
      Epoch 3 composite train-obj: 2.554343
            Val objective improved 3.5658 → 2.5802, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 20.0886, mae: 2.7173, huber: 2.2794, swd: 3.8083, ept: 36.8171
    Epoch [4/50], Val Losses: mse: 15.6887, mae: 2.6718, huber: 2.2210, swd: 4.8178, ept: 86.1815
    Epoch [4/50], Test Losses: mse: 13.8549, mae: 2.5600, huber: 2.1101, swd: 4.1551, ept: 87.5166
      Epoch 4 composite train-obj: 2.279360
            Val objective improved 2.5802 → 2.2210, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 18.4454, mae: 2.5316, huber: 2.1002, swd: 3.3846, ept: 36.8874
    Epoch [5/50], Val Losses: mse: 17.6066, mae: 2.5608, huber: 2.1207, swd: 3.9703, ept: 84.1155
    Epoch [5/50], Test Losses: mse: 16.2516, mae: 2.5037, huber: 2.0650, swd: 4.1080, ept: 84.0451
      Epoch 5 composite train-obj: 2.100177
            Val objective improved 2.2210 → 2.1207, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 17.4629, mae: 2.4002, huber: 1.9745, swd: 3.0202, ept: 36.9643
    Epoch [6/50], Val Losses: mse: 19.2494, mae: 2.6996, huber: 2.2578, swd: 3.9118, ept: 81.7829
    Epoch [6/50], Test Losses: mse: 18.4362, mae: 2.6616, huber: 2.2215, swd: 3.8098, ept: 81.9903
      Epoch 6 composite train-obj: 1.974537
            No improvement (2.2578), counter 1/5
    Epoch [7/50], Train Losses: mse: 17.1601, mae: 2.3563, huber: 1.9334, swd: 2.8103, ept: 37.4958
    Epoch [7/50], Val Losses: mse: 10.0745, mae: 1.7892, huber: 1.3705, swd: 3.5315, ept: 89.5400
    Epoch [7/50], Test Losses: mse: 7.8175, mae: 1.6531, huber: 1.2367, swd: 2.9501, ept: 90.9159
      Epoch 7 composite train-obj: 1.933393
            Val objective improved 2.1207 → 1.3705, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 15.4188, mae: 2.1525, huber: 1.7391, swd: 2.5410, ept: 37.5305
    Epoch [8/50], Val Losses: mse: 11.3862, mae: 1.7582, huber: 1.3523, swd: 3.0337, ept: 89.5281
    Epoch [8/50], Test Losses: mse: 8.7459, mae: 1.6538, huber: 1.2478, swd: 3.1061, ept: 90.5787
      Epoch 8 composite train-obj: 1.739057
            Val objective improved 1.3705 → 1.3523, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 14.9489, mae: 2.0910, huber: 1.6814, swd: 2.3956, ept: 37.2770
    Epoch [9/50], Val Losses: mse: 7.9327, mae: 1.6643, huber: 1.2435, swd: 2.5685, ept: 91.2880
    Epoch [9/50], Test Losses: mse: 6.2635, mae: 1.5539, huber: 1.1340, swd: 1.6824, ept: 92.6426
      Epoch 9 composite train-obj: 1.681404
            Val objective improved 1.3523 → 1.2435, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 14.7946, mae: 2.0767, huber: 1.6681, swd: 2.3708, ept: 37.2707
    Epoch [10/50], Val Losses: mse: 19.1736, mae: 2.2546, huber: 1.8474, swd: 2.2710, ept: 84.6654
    Epoch [10/50], Test Losses: mse: 17.4323, mae: 2.1668, huber: 1.7613, swd: 2.2662, ept: 85.6395
      Epoch 10 composite train-obj: 1.668131
            No improvement (1.8474), counter 1/5
    Epoch [11/50], Train Losses: mse: 15.3649, mae: 2.1238, huber: 1.7145, swd: 2.3735, ept: 37.2172
    Epoch [11/50], Val Losses: mse: 8.9059, mae: 1.6957, huber: 1.2811, swd: 2.7296, ept: 90.1208
    Epoch [11/50], Test Losses: mse: 7.3758, mae: 1.6045, huber: 1.1912, swd: 2.0724, ept: 91.3839
      Epoch 11 composite train-obj: 1.714503
            No improvement (1.2811), counter 2/5
    Epoch [12/50], Train Losses: mse: 14.2603, mae: 1.9978, huber: 1.5953, swd: 2.1968, ept: 37.4110
    Epoch [12/50], Val Losses: mse: 6.4671, mae: 1.4334, huber: 1.0295, swd: 2.2080, ept: 91.8001
    Epoch [12/50], Test Losses: mse: 4.9894, mae: 1.3334, huber: 0.9317, swd: 1.5539, ept: 93.2270
      Epoch 12 composite train-obj: 1.595252
            Val objective improved 1.2435 → 1.0295, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 13.8106, mae: 1.9499, huber: 1.5504, swd: 2.1032, ept: 37.4694
    Epoch [13/50], Val Losses: mse: 7.1710, mae: 1.6501, huber: 1.2312, swd: 2.3336, ept: 91.7148
    Epoch [13/50], Test Losses: mse: 6.0074, mae: 1.5740, huber: 1.1541, swd: 1.9562, ept: 92.9611
      Epoch 13 composite train-obj: 1.550391
            No improvement (1.2312), counter 1/5
    Epoch [14/50], Train Losses: mse: 13.7381, mae: 1.9377, huber: 1.5389, swd: 2.0686, ept: 37.4744
    Epoch [14/50], Val Losses: mse: 8.3489, mae: 1.6164, huber: 1.2068, swd: 2.3315, ept: 91.1617
    Epoch [14/50], Test Losses: mse: 6.8406, mae: 1.5360, huber: 1.1269, swd: 1.8646, ept: 92.3070
      Epoch 14 composite train-obj: 1.538872
            No improvement (1.2068), counter 2/5
    Epoch [15/50], Train Losses: mse: 13.4776, mae: 1.9073, huber: 1.5111, swd: 1.9607, ept: 37.4090
    Epoch [15/50], Val Losses: mse: 9.1384, mae: 1.5924, huber: 1.1896, swd: 2.1609, ept: 90.4882
    Epoch [15/50], Test Losses: mse: 7.1012, mae: 1.4799, huber: 1.0788, swd: 1.9718, ept: 91.8986
      Epoch 15 composite train-obj: 1.511089
            No improvement (1.1896), counter 3/5
    Epoch [16/50], Train Losses: mse: 13.4386, mae: 1.9021, huber: 1.5066, swd: 1.9597, ept: 37.5358
    Epoch [16/50], Val Losses: mse: 13.3755, mae: 1.8302, huber: 1.4182, swd: 2.2290, ept: 89.5998
    Epoch [16/50], Test Losses: mse: 11.7932, mae: 1.8015, huber: 1.3907, swd: 2.1339, ept: 90.0502
      Epoch 16 composite train-obj: 1.506567
            No improvement (1.4182), counter 4/5
    Epoch [17/50], Train Losses: mse: 13.6624, mae: 1.8986, huber: 1.5055, swd: 1.9927, ept: 37.4828
    Epoch [17/50], Val Losses: mse: 6.3304, mae: 1.2817, huber: 0.8954, swd: 2.0088, ept: 92.2723
    Epoch [17/50], Test Losses: mse: 4.5351, mae: 1.1815, huber: 0.7983, swd: 1.6885, ept: 93.5130
      Epoch 17 composite train-obj: 1.505531
            Val objective improved 1.0295 → 0.8954, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 12.9856, mae: 1.8335, huber: 1.4436, swd: 1.8197, ept: 37.4014
    Epoch [18/50], Val Losses: mse: 7.7197, mae: 1.6875, huber: 1.2583, swd: 1.9177, ept: 91.7174
    Epoch [18/50], Test Losses: mse: 6.3118, mae: 1.6055, huber: 1.1756, swd: 1.5026, ept: 93.0322
      Epoch 18 composite train-obj: 1.443613
            No improvement (1.2583), counter 1/5
    Epoch [19/50], Train Losses: mse: 13.2648, mae: 1.8501, huber: 1.4600, swd: 1.8402, ept: 37.6363
    Epoch [19/50], Val Losses: mse: 8.0521, mae: 1.4839, huber: 1.0949, swd: 1.7500, ept: 90.7627
    Epoch [19/50], Test Losses: mse: 6.4430, mae: 1.3880, huber: 1.0022, swd: 1.2110, ept: 91.6014
      Epoch 19 composite train-obj: 1.460025
            No improvement (1.0949), counter 2/5
    Epoch [20/50], Train Losses: mse: 13.1615, mae: 1.8471, huber: 1.4571, swd: 1.8399, ept: 37.5057
    Epoch [20/50], Val Losses: mse: 7.2921, mae: 1.6421, huber: 1.2192, swd: 1.9545, ept: 91.7460
    Epoch [20/50], Test Losses: mse: 6.5049, mae: 1.6081, huber: 1.1847, swd: 1.6817, ept: 92.9483
      Epoch 20 composite train-obj: 1.457140
            No improvement (1.2192), counter 3/5
    Epoch [21/50], Train Losses: mse: 12.9220, mae: 1.8206, huber: 1.4324, swd: 1.7847, ept: 37.5063
    Epoch [21/50], Val Losses: mse: 7.2548, mae: 1.5104, huber: 1.0990, swd: 1.9105, ept: 91.6866
    Epoch [21/50], Test Losses: mse: 6.1002, mae: 1.4248, huber: 1.0175, swd: 1.4685, ept: 92.7807
      Epoch 21 composite train-obj: 1.432389
            No improvement (1.0990), counter 4/5
    Epoch [22/50], Train Losses: mse: 12.9266, mae: 1.8055, huber: 1.4193, swd: 1.7609, ept: 37.5878
    Epoch [22/50], Val Losses: mse: 6.1938, mae: 1.3971, huber: 0.9964, swd: 2.2609, ept: 91.9065
    Epoch [22/50], Test Losses: mse: 4.7506, mae: 1.2901, huber: 0.8909, swd: 1.5413, ept: 93.3528
      Epoch 22 composite train-obj: 1.419310
    Epoch [22/50], Test Losses: mse: 4.5351, mae: 1.1815, huber: 0.7983, swd: 1.6885, ept: 93.5130
    Best round's Test MSE: 4.5351, MAE: 1.1815, SWD: 1.6885
    Best round's Validation MSE: 6.3304, MAE: 1.2817, SWD: 2.0088
    Best round's Test verification MSE : 4.5351, MAE: 1.1815, SWD: 1.6885
    Time taken: 120.79 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 45.3993, mae: 4.7354, huber: 4.2660, swd: 7.8640, ept: 29.7937
    Epoch [1/50], Val Losses: mse: 47.5446, mae: 5.1151, huber: 4.6393, swd: 13.4792, ept: 46.0284
    Epoch [1/50], Test Losses: mse: 45.6780, mae: 5.0149, huber: 4.5397, swd: 14.2681, ept: 47.3783
      Epoch 1 composite train-obj: 4.265963
            Val objective improved inf → 4.6393, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 29.0192, mae: 3.5374, huber: 3.0817, swd: 5.7222, ept: 34.7729
    Epoch [2/50], Val Losses: mse: 25.3697, mae: 3.4284, huber: 2.9685, swd: 7.1293, ept: 76.7573
    Epoch [2/50], Test Losses: mse: 23.5380, mae: 3.3246, huber: 2.8653, swd: 6.8875, ept: 76.9404
      Epoch 2 composite train-obj: 3.081712
            Val objective improved 4.6393 → 2.9685, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 23.6209, mae: 3.0599, huber: 2.6135, swd: 4.5167, ept: 36.0401
    Epoch [3/50], Val Losses: mse: 18.0108, mae: 2.8556, huber: 2.4012, swd: 6.3918, ept: 82.4315
    Epoch [3/50], Test Losses: mse: 16.5970, mae: 2.7797, huber: 2.3265, swd: 6.1452, ept: 83.0358
      Epoch 3 composite train-obj: 2.613527
            Val objective improved 2.9685 → 2.4012, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 20.2652, mae: 2.7215, huber: 2.2837, swd: 3.7242, ept: 36.6690
    Epoch [4/50], Val Losses: mse: 15.6086, mae: 2.6224, huber: 2.1739, swd: 4.0110, ept: 84.9867
    Epoch [4/50], Test Losses: mse: 14.4857, mae: 2.5518, huber: 2.1051, swd: 3.8172, ept: 85.0275
      Epoch 4 composite train-obj: 2.283743
            Val objective improved 2.4012 → 2.1739, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 18.7331, mae: 2.5536, huber: 2.1219, swd: 3.2335, ept: 37.0591
    Epoch [5/50], Val Losses: mse: 18.4561, mae: 2.6655, huber: 2.2227, swd: 4.0332, ept: 84.0932
    Epoch [5/50], Test Losses: mse: 16.6086, mae: 2.5530, huber: 2.1120, swd: 3.9067, ept: 85.0032
      Epoch 5 composite train-obj: 2.121937
            No improvement (2.2227), counter 1/5
    Epoch [6/50], Train Losses: mse: 18.1594, mae: 2.4746, huber: 2.0467, swd: 3.1289, ept: 37.0065
    Epoch [6/50], Val Losses: mse: 11.7661, mae: 2.0137, huber: 1.5865, swd: 2.8102, ept: 89.3651
    Epoch [6/50], Test Losses: mse: 9.9652, mae: 1.9468, huber: 1.5195, swd: 2.5181, ept: 89.7094
      Epoch 6 composite train-obj: 2.046710
            Val objective improved 2.1739 → 1.5865, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 16.6933, mae: 2.3212, huber: 1.8994, swd: 2.7419, ept: 37.1694
    Epoch [7/50], Val Losses: mse: 9.1991, mae: 1.7631, huber: 1.3478, swd: 2.7691, ept: 90.5747
    Epoch [7/50], Test Losses: mse: 7.4948, mae: 1.6618, huber: 1.2482, swd: 2.1293, ept: 91.1230
      Epoch 7 composite train-obj: 1.899417
            Val objective improved 1.5865 → 1.3478, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 15.9474, mae: 2.2290, huber: 1.8121, swd: 2.6357, ept: 37.2716
    Epoch [8/50], Val Losses: mse: 32.7387, mae: 3.5488, huber: 3.0962, swd: 4.4660, ept: 76.7266
    Epoch [8/50], Test Losses: mse: 32.4359, mae: 3.5290, huber: 3.0774, swd: 4.6049, ept: 76.5100
      Epoch 8 composite train-obj: 1.812092
            No improvement (3.0962), counter 1/5
    Epoch [9/50], Train Losses: mse: 17.7694, mae: 2.3843, huber: 1.9639, swd: 3.0344, ept: 37.0241
    Epoch [9/50], Val Losses: mse: 11.2275, mae: 2.0317, huber: 1.6037, swd: 3.4582, ept: 89.1630
    Epoch [9/50], Test Losses: mse: 9.8880, mae: 1.9143, huber: 1.4921, swd: 3.3224, ept: 89.0143
      Epoch 9 composite train-obj: 1.963920
            No improvement (1.6037), counter 2/5
    Epoch [10/50], Train Losses: mse: 15.4694, mae: 2.1564, huber: 1.7440, swd: 2.4628, ept: 37.3775
    Epoch [10/50], Val Losses: mse: 10.4287, mae: 1.8298, huber: 1.4088, swd: 2.7613, ept: 90.0316
    Epoch [10/50], Test Losses: mse: 8.7302, mae: 1.7499, huber: 1.3304, swd: 2.3441, ept: 90.4881
      Epoch 10 composite train-obj: 1.744025
            No improvement (1.4088), counter 3/5
    Epoch [11/50], Train Losses: mse: 15.2658, mae: 2.1139, huber: 1.7053, swd: 2.3620, ept: 37.2512
    Epoch [11/50], Val Losses: mse: 8.9314, mae: 1.6915, huber: 1.2757, swd: 2.9047, ept: 90.6859
    Epoch [11/50], Test Losses: mse: 7.2150, mae: 1.6047, huber: 1.1919, swd: 2.7176, ept: 91.3946
      Epoch 11 composite train-obj: 1.705261
            Val objective improved 1.3478 → 1.2757, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 14.4610, mae: 2.0308, huber: 1.6260, swd: 2.2089, ept: 37.3279
    Epoch [12/50], Val Losses: mse: 8.2361, mae: 1.5864, huber: 1.1793, swd: 2.2570, ept: 91.4213
    Epoch [12/50], Test Losses: mse: 6.7190, mae: 1.5261, huber: 1.1188, swd: 2.0113, ept: 92.3461
      Epoch 12 composite train-obj: 1.626044
            Val objective improved 1.2757 → 1.1793, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 14.7397, mae: 2.0238, huber: 1.6208, swd: 2.2471, ept: 37.4060
    Epoch [13/50], Val Losses: mse: 29.9727, mae: 3.1046, huber: 2.6594, swd: 3.6738, ept: 80.0243
    Epoch [13/50], Test Losses: mse: 33.1652, mae: 3.2467, huber: 2.8024, swd: 3.7456, ept: 78.3572
      Epoch 13 composite train-obj: 1.620831
            No improvement (2.6594), counter 1/5
    Epoch [14/50], Train Losses: mse: 16.3470, mae: 2.2119, huber: 1.8016, swd: 2.9603, ept: 37.0287
    Epoch [14/50], Val Losses: mse: 6.1970, mae: 1.2718, huber: 0.8871, swd: 1.7840, ept: 92.1965
    Epoch [14/50], Test Losses: mse: 5.0573, mae: 1.2311, huber: 0.8471, swd: 1.3196, ept: 92.8286
      Epoch 14 composite train-obj: 1.801593
            Val objective improved 1.1793 → 0.8871, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 13.9644, mae: 1.9552, huber: 1.5572, swd: 2.1116, ept: 37.4020
    Epoch [15/50], Val Losses: mse: 7.4023, mae: 1.6311, huber: 1.2130, swd: 1.9583, ept: 91.3761
    Epoch [15/50], Test Losses: mse: 6.5722, mae: 1.5996, huber: 1.1826, swd: 1.4254, ept: 92.3100
      Epoch 15 composite train-obj: 1.557195
            No improvement (1.2130), counter 1/5
    Epoch [16/50], Train Losses: mse: 13.7138, mae: 1.9255, huber: 1.5294, swd: 1.9895, ept: 37.3313
    Epoch [16/50], Val Losses: mse: 5.8940, mae: 1.3875, huber: 0.9839, swd: 1.5947, ept: 92.1676
    Epoch [16/50], Test Losses: mse: 4.9923, mae: 1.3406, huber: 0.9364, swd: 1.4484, ept: 92.9528
      Epoch 16 composite train-obj: 1.529425
            No improvement (0.9839), counter 2/5
    Epoch [17/50], Train Losses: mse: 13.5782, mae: 1.9158, huber: 1.5198, swd: 1.9756, ept: 37.2539
    Epoch [17/50], Val Losses: mse: 10.3140, mae: 1.6987, huber: 1.2833, swd: 1.8592, ept: 90.5087
    Epoch [17/50], Test Losses: mse: 10.8230, mae: 1.7546, huber: 1.3409, swd: 2.0587, ept: 90.2575
      Epoch 17 composite train-obj: 1.519796
            No improvement (1.2833), counter 3/5
    Epoch [18/50], Train Losses: mse: 14.2672, mae: 1.9651, huber: 1.5684, swd: 2.0690, ept: 37.4095
    Epoch [18/50], Val Losses: mse: 8.3016, mae: 1.6150, huber: 1.2105, swd: 2.3977, ept: 91.3667
    Epoch [18/50], Test Losses: mse: 7.1149, mae: 1.5689, huber: 1.1624, swd: 2.2478, ept: 92.0826
      Epoch 18 composite train-obj: 1.568355
            No improvement (1.2105), counter 4/5
    Epoch [19/50], Train Losses: mse: 13.5688, mae: 1.9011, huber: 1.5075, swd: 1.9009, ept: 37.4555
    Epoch [19/50], Val Losses: mse: 8.2674, mae: 1.5217, huber: 1.1225, swd: 1.6335, ept: 91.1118
    Epoch [19/50], Test Losses: mse: 6.5055, mae: 1.4042, huber: 1.0093, swd: 1.3520, ept: 92.2131
      Epoch 19 composite train-obj: 1.507472
    Epoch [19/50], Test Losses: mse: 5.0573, mae: 1.2311, huber: 0.8471, swd: 1.3196, ept: 92.8286
    Best round's Test MSE: 5.0573, MAE: 1.2311, SWD: 1.3196
    Best round's Validation MSE: 6.1970, MAE: 1.2718, SWD: 1.7840
    Best round's Test verification MSE : 5.0573, MAE: 1.2311, SWD: 1.3196
    Time taken: 107.57 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq336_pred96_20250513_2308)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 5.4495 ± 0.9481
      mae: 1.3207 ± 0.1630
      huber: 0.9258 ± 0.1472
      swd: 1.8229 ± 0.4754
      ept: 92.7656 ± 0.6375
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.0775 ± 1.1522
      mae: 1.4023 ± 0.1776
      huber: 1.0057 ± 0.1619
      swd: 2.2633 ± 0.5269
      ept: 91.7356 ± 0.7060
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 299.18 seconds
    
    Experiment complete: PatchTST_lorenz_seq336_pred96_20250513_2308
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 336-196
##### huber


```python
utils.reload_modules([utils])
cfg_patch_tst = train_config.FlatPatchTSTConfig(
    seq_len=336,
    pred_len=196,
    channels=data_mgr.datasets['lorenz']['channels'],
    enc_in=data_mgr.datasets['lorenz']['channels'],
    dec_in=data_mgr.datasets['lorenz']['channels'],
    c_out=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0]
)
exp_patch_tst = execute_model_evaluation('lorenz', cfg_patch_tst, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 281
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 281
    Validation Batches: 37
    Test Batches: 78
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 57.0663, mae: 5.4904, huber: 5.0155, swd: 9.9352, ept: 33.1516
    Epoch [1/50], Val Losses: mse: 41.5534, mae: 4.4186, huber: 3.9537, swd: 8.9887, ept: 100.6638
    Epoch [1/50], Test Losses: mse: 41.5885, mae: 4.3881, huber: 3.9240, swd: 9.1982, ept: 97.4903
      Epoch 1 composite train-obj: 5.015475
            Val objective improved inf → 3.9537, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.9557, mae: 4.3218, huber: 3.8577, swd: 7.2497, ept: 41.1777
    Epoch [2/50], Val Losses: mse: 32.3436, mae: 3.6704, huber: 3.2162, swd: 7.3709, ept: 126.0259
    Epoch [2/50], Test Losses: mse: 30.1200, mae: 3.5452, huber: 3.0909, swd: 6.6937, ept: 126.8380
      Epoch 2 composite train-obj: 3.857744
            Val objective improved 3.9537 → 3.2162, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 34.4041, mae: 3.8448, huber: 3.3882, swd: 5.7576, ept: 42.7194
    Epoch [3/50], Val Losses: mse: 31.7161, mae: 3.5533, huber: 3.1033, swd: 6.7130, ept: 133.7217
    Epoch [3/50], Test Losses: mse: 27.9394, mae: 3.3289, huber: 2.8816, swd: 6.0946, ept: 133.9221
      Epoch 3 composite train-obj: 3.388204
            Val objective improved 3.2162 → 3.1033, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 31.4825, mae: 3.5705, huber: 3.1199, swd: 5.1539, ept: 43.2061
    Epoch [4/50], Val Losses: mse: 28.5501, mae: 3.2348, huber: 2.7942, swd: 5.9483, ept: 140.8988
    Epoch [4/50], Test Losses: mse: 25.6656, mae: 3.0631, huber: 2.6230, swd: 5.1750, ept: 144.8265
      Epoch 4 composite train-obj: 3.119854
            Val objective improved 3.1033 → 2.7942, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 29.3025, mae: 3.3778, huber: 2.9316, swd: 4.6714, ept: 43.5827
    Epoch [5/50], Val Losses: mse: 27.9121, mae: 3.1443, huber: 2.7005, swd: 5.0124, ept: 146.7766
    Epoch [5/50], Test Losses: mse: 22.8310, mae: 2.8607, huber: 2.4199, swd: 4.0717, ept: 151.0162
      Epoch 5 composite train-obj: 2.931574
            Val objective improved 2.7942 → 2.7005, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 27.4775, mae: 3.2147, huber: 2.7726, swd: 4.2481, ept: 43.6824
    Epoch [6/50], Val Losses: mse: 23.3371, mae: 2.7013, huber: 2.2752, swd: 4.7374, ept: 153.2237
    Epoch [6/50], Test Losses: mse: 18.1341, mae: 2.3944, huber: 1.9717, swd: 3.5870, ept: 158.4318
      Epoch 6 composite train-obj: 2.772595
            Val objective improved 2.7005 → 2.2752, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 26.0691, mae: 3.0844, huber: 2.6460, swd: 3.9209, ept: 43.7210
    Epoch [7/50], Val Losses: mse: 27.7501, mae: 2.9566, huber: 2.5211, swd: 4.1470, ept: 151.8277
    Epoch [7/50], Test Losses: mse: 22.2004, mae: 2.7099, huber: 2.2763, swd: 3.7475, ept: 155.0472
      Epoch 7 composite train-obj: 2.645972
            No improvement (2.5211), counter 1/5
    Epoch [8/50], Train Losses: mse: 25.1965, mae: 3.0034, huber: 2.5673, swd: 3.7373, ept: 44.1365
    Epoch [8/50], Val Losses: mse: 20.4366, mae: 2.6194, huber: 2.1858, swd: 4.2899, ept: 154.2498
    Epoch [8/50], Test Losses: mse: 16.9467, mae: 2.3636, huber: 1.9339, swd: 3.1354, ept: 158.6876
      Epoch 8 composite train-obj: 2.567293
            Val objective improved 2.2752 → 2.1858, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.3729, mae: 2.9210, huber: 2.4879, swd: 3.5508, ept: 44.0929
    Epoch [9/50], Val Losses: mse: 23.0257, mae: 2.9540, huber: 2.5069, swd: 3.7068, ept: 156.5772
    Epoch [9/50], Test Losses: mse: 18.8713, mae: 2.7188, huber: 2.2731, swd: 2.8135, ept: 161.5680
      Epoch 9 composite train-obj: 2.487872
            No improvement (2.5069), counter 1/5
    Epoch [10/50], Train Losses: mse: 23.5973, mae: 2.8507, huber: 2.4197, swd: 3.3889, ept: 44.1335
    Epoch [10/50], Val Losses: mse: 25.3936, mae: 2.7054, huber: 2.2793, swd: 3.3005, ept: 157.3983
    Epoch [10/50], Test Losses: mse: 19.0714, mae: 2.4187, huber: 1.9954, swd: 3.2461, ept: 163.2165
      Epoch 10 composite train-obj: 2.419681
            No improvement (2.2793), counter 2/5
    Epoch [11/50], Train Losses: mse: 23.2854, mae: 2.8155, huber: 2.3861, swd: 3.2927, ept: 44.1543
    Epoch [11/50], Val Losses: mse: 19.1906, mae: 2.5261, huber: 2.0937, swd: 4.2709, ept: 159.6730
    Epoch [11/50], Test Losses: mse: 14.5146, mae: 2.2325, huber: 1.8051, swd: 3.3638, ept: 166.0108
      Epoch 11 composite train-obj: 2.386054
            Val objective improved 2.1858 → 2.0937, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 22.7476, mae: 2.7691, huber: 2.3410, swd: 3.1813, ept: 44.1726
    Epoch [12/50], Val Losses: mse: 18.2007, mae: 2.3528, huber: 1.9382, swd: 2.8961, ept: 158.9839
    Epoch [12/50], Test Losses: mse: 15.0443, mae: 2.1488, huber: 1.7364, swd: 2.3684, ept: 163.7908
      Epoch 12 composite train-obj: 2.341008
            Val objective improved 2.0937 → 1.9382, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 22.4558, mae: 2.7398, huber: 2.3129, swd: 3.1183, ept: 44.2825
    Epoch [13/50], Val Losses: mse: 23.0349, mae: 2.6625, huber: 2.2344, swd: 3.4179, ept: 156.5521
    Epoch [13/50], Test Losses: mse: 20.0290, mae: 2.5335, huber: 2.1059, swd: 3.1798, ept: 160.3550
      Epoch 13 composite train-obj: 2.312851
            No improvement (2.2344), counter 1/5
    Epoch [14/50], Train Losses: mse: 22.0735, mae: 2.7068, huber: 2.2810, swd: 3.0237, ept: 44.4263
    Epoch [14/50], Val Losses: mse: 18.7467, mae: 2.2947, huber: 1.8826, swd: 3.0555, ept: 161.3368
    Epoch [14/50], Test Losses: mse: 13.7841, mae: 2.0077, huber: 1.5985, swd: 2.5683, ept: 167.3411
      Epoch 14 composite train-obj: 2.281006
            Val objective improved 1.9382 → 1.8826, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 21.6204, mae: 2.6631, huber: 2.2387, swd: 2.9331, ept: 44.3694
    Epoch [15/50], Val Losses: mse: 18.5727, mae: 2.4093, huber: 1.9871, swd: 2.7053, ept: 162.3048
    Epoch [15/50], Test Losses: mse: 14.6762, mae: 2.1715, huber: 1.7522, swd: 2.1177, ept: 167.9093
      Epoch 15 composite train-obj: 2.238654
            No improvement (1.9871), counter 1/5
    Epoch [16/50], Train Losses: mse: 21.3901, mae: 2.6416, huber: 2.2182, swd: 2.8915, ept: 44.3513
    Epoch [16/50], Val Losses: mse: 22.0516, mae: 2.5351, huber: 2.1112, swd: 3.3867, ept: 160.9709
    Epoch [16/50], Test Losses: mse: 16.6292, mae: 2.2661, huber: 1.8448, swd: 2.9153, ept: 167.0950
      Epoch 16 composite train-obj: 2.218211
            No improvement (2.1112), counter 2/5
    Epoch [17/50], Train Losses: mse: 21.1955, mae: 2.6212, huber: 2.1988, swd: 2.8325, ept: 44.4426
    Epoch [17/50], Val Losses: mse: 17.7591, mae: 2.3110, huber: 1.8860, swd: 2.8123, ept: 163.7102
    Epoch [17/50], Test Losses: mse: 12.9173, mae: 2.0348, huber: 1.6115, swd: 2.1795, ept: 170.7305
      Epoch 17 composite train-obj: 2.198752
            No improvement (1.8860), counter 3/5
    Epoch [18/50], Train Losses: mse: 20.7776, mae: 2.5892, huber: 2.1676, swd: 2.7625, ept: 44.4381
    Epoch [18/50], Val Losses: mse: 19.2326, mae: 2.4581, huber: 2.0324, swd: 3.6179, ept: 163.1344
    Epoch [18/50], Test Losses: mse: 14.5267, mae: 2.1843, huber: 1.7627, swd: 2.9050, ept: 168.9257
      Epoch 18 composite train-obj: 2.167573
            No improvement (2.0324), counter 4/5
    Epoch [19/50], Train Losses: mse: 20.6806, mae: 2.5711, huber: 2.1510, swd: 2.7248, ept: 44.5418
    Epoch [19/50], Val Losses: mse: 17.7823, mae: 2.2573, huber: 1.8363, swd: 3.2995, ept: 164.3438
    Epoch [19/50], Test Losses: mse: 13.6862, mae: 2.0256, huber: 1.6083, swd: 2.6810, ept: 169.5884
      Epoch 19 composite train-obj: 2.150982
            Val objective improved 1.8826 → 1.8363, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 20.3699, mae: 2.5472, huber: 2.1275, swd: 2.6773, ept: 44.4225
    Epoch [20/50], Val Losses: mse: 16.1146, mae: 2.1157, huber: 1.7057, swd: 2.8438, ept: 163.9803
    Epoch [20/50], Test Losses: mse: 11.7250, mae: 1.8466, huber: 1.4396, swd: 2.0572, ept: 171.0461
      Epoch 20 composite train-obj: 2.127513
            Val objective improved 1.8363 → 1.7057, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 20.4134, mae: 2.5497, huber: 2.1300, swd: 2.6679, ept: 44.4668
    Epoch [21/50], Val Losses: mse: 20.3382, mae: 2.4000, huber: 1.9792, swd: 2.5862, ept: 165.4923
    Epoch [21/50], Test Losses: mse: 14.1861, mae: 2.0969, huber: 1.6799, swd: 2.4097, ept: 171.1376
      Epoch 21 composite train-obj: 2.129992
            No improvement (1.9792), counter 1/5
    Epoch [22/50], Train Losses: mse: 20.2171, mae: 2.5271, huber: 2.1085, swd: 2.6002, ept: 44.5401
    Epoch [22/50], Val Losses: mse: 17.9128, mae: 2.4833, huber: 2.0473, swd: 3.0175, ept: 164.4246
    Epoch [22/50], Test Losses: mse: 13.4330, mae: 2.2188, huber: 1.7858, swd: 2.3301, ept: 169.1712
      Epoch 22 composite train-obj: 2.108494
            No improvement (2.0473), counter 2/5
    Epoch [23/50], Train Losses: mse: 20.0064, mae: 2.5110, huber: 2.0927, swd: 2.5901, ept: 44.4984
    Epoch [23/50], Val Losses: mse: 19.0230, mae: 2.2704, huber: 1.8521, swd: 2.7376, ept: 166.7931
    Epoch [23/50], Test Losses: mse: 14.0782, mae: 2.0168, huber: 1.6017, swd: 2.3298, ept: 172.7749
      Epoch 23 composite train-obj: 2.092686
            No improvement (1.8521), counter 3/5
    Epoch [24/50], Train Losses: mse: 19.9835, mae: 2.5020, huber: 2.0847, swd: 2.5537, ept: 44.5109
    Epoch [24/50], Val Losses: mse: 18.4416, mae: 2.3077, huber: 1.8846, swd: 3.4969, ept: 166.4819
    Epoch [24/50], Test Losses: mse: 13.4373, mae: 2.0192, huber: 1.6001, swd: 2.8391, ept: 172.4046
      Epoch 24 composite train-obj: 2.084653
            No improvement (1.8846), counter 4/5
    Epoch [25/50], Train Losses: mse: 19.8142, mae: 2.4896, huber: 2.0723, swd: 2.5233, ept: 44.7452
    Epoch [25/50], Val Losses: mse: 18.0687, mae: 2.3066, huber: 1.8788, swd: 2.4135, ept: 166.0607
    Epoch [25/50], Test Losses: mse: 13.7572, mae: 2.0836, huber: 1.6592, swd: 2.0002, ept: 171.9928
      Epoch 25 composite train-obj: 2.072334
    Epoch [25/50], Test Losses: mse: 11.7250, mae: 1.8466, huber: 1.4396, swd: 2.0572, ept: 171.0461
    Best round's Test MSE: 11.7250, MAE: 1.8466, SWD: 2.0572
    Best round's Validation MSE: 16.1146, MAE: 2.1157, SWD: 2.8438
    Best round's Test verification MSE : 11.7250, MAE: 1.8466, SWD: 2.0572
    Time taken: 147.97 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 58.0266, mae: 5.5441, huber: 5.0688, swd: 9.8862, ept: 32.9308
    Epoch [1/50], Val Losses: mse: 41.8566, mae: 4.4717, huber: 4.0064, swd: 9.6928, ept: 96.9295
    Epoch [1/50], Test Losses: mse: 41.8210, mae: 4.4597, huber: 3.9943, swd: 9.3962, ept: 93.8188
      Epoch 1 composite train-obj: 5.068803
            Val objective improved inf → 4.0064, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 41.4098, mae: 4.4294, huber: 3.9642, swd: 7.3313, ept: 40.8684
    Epoch [2/50], Val Losses: mse: 35.5533, mae: 3.8151, huber: 3.3629, swd: 7.4531, ept: 119.9175
    Epoch [2/50], Test Losses: mse: 34.4012, mae: 3.7541, huber: 3.3018, swd: 7.1827, ept: 119.3672
      Epoch 2 composite train-obj: 3.964208
            Val objective improved 4.0064 → 3.3629, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 36.2906, mae: 3.9702, huber: 3.5122, swd: 6.1796, ept: 42.3643
    Epoch [3/50], Val Losses: mse: 30.9483, mae: 3.4204, huber: 2.9752, swd: 5.6354, ept: 135.3912
    Epoch [3/50], Test Losses: mse: 29.8171, mae: 3.3430, huber: 2.8988, swd: 5.2980, ept: 136.1302
      Epoch 3 composite train-obj: 3.512227
            Val objective improved 3.3629 → 2.9752, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 32.6702, mae: 3.6477, huber: 3.1962, swd: 5.3186, ept: 43.1830
    Epoch [4/50], Val Losses: mse: 28.7009, mae: 3.1679, huber: 2.7321, swd: 5.1561, ept: 139.6796
    Epoch [4/50], Test Losses: mse: 25.1664, mae: 2.9879, huber: 2.5529, swd: 4.8698, ept: 140.4084
      Epoch 4 composite train-obj: 3.196195
            Val objective improved 2.9752 → 2.7321, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 29.8195, mae: 3.4120, huber: 2.9656, swd: 4.6592, ept: 43.4257
    Epoch [5/50], Val Losses: mse: 26.2059, mae: 3.0077, huber: 2.5706, swd: 4.4072, ept: 145.8858
    Epoch [5/50], Test Losses: mse: 22.3465, mae: 2.8012, huber: 2.3668, swd: 4.3137, ept: 146.5883
      Epoch 5 composite train-obj: 2.965607
            Val objective improved 2.7321 → 2.5706, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 28.0902, mae: 3.2595, huber: 2.8167, swd: 4.1918, ept: 43.4667
    Epoch [6/50], Val Losses: mse: 24.3163, mae: 2.9035, huber: 2.4644, swd: 4.3529, ept: 148.9441
    Epoch [6/50], Test Losses: mse: 21.6413, mae: 2.7519, huber: 2.3130, swd: 3.9892, ept: 152.3869
      Epoch 6 composite train-obj: 2.816746
            Val objective improved 2.5706 → 2.4644, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 26.3734, mae: 3.1076, huber: 2.6690, swd: 3.8054, ept: 43.8606
    Epoch [7/50], Val Losses: mse: 26.1539, mae: 3.0202, huber: 2.5758, swd: 3.3892, ept: 148.4270
    Epoch [7/50], Test Losses: mse: 21.6017, mae: 2.7661, huber: 2.3247, swd: 3.2398, ept: 152.8170
      Epoch 7 composite train-obj: 2.669022
            No improvement (2.5758), counter 1/5
    Epoch [8/50], Train Losses: mse: 25.5645, mae: 3.0302, huber: 2.5940, swd: 3.6248, ept: 44.0292
    Epoch [8/50], Val Losses: mse: 22.1825, mae: 2.8366, huber: 2.3955, swd: 3.9942, ept: 154.0126
    Epoch [8/50], Test Losses: mse: 19.0535, mae: 2.6636, huber: 2.2234, swd: 3.8125, ept: 157.4063
      Epoch 8 composite train-obj: 2.593987
            Val objective improved 2.4644 → 2.3955, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.4618, mae: 2.9258, huber: 2.4930, swd: 3.4134, ept: 44.0283
    Epoch [9/50], Val Losses: mse: 22.8198, mae: 2.6870, huber: 2.2598, swd: 3.0872, ept: 154.9705
    Epoch [9/50], Test Losses: mse: 18.4419, mae: 2.4392, huber: 2.0149, swd: 2.7076, ept: 158.3450
      Epoch 9 composite train-obj: 2.492987
            Val objective improved 2.3955 → 2.2598, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 24.1414, mae: 2.8903, huber: 2.4589, swd: 3.3446, ept: 44.1077
    Epoch [10/50], Val Losses: mse: 20.5602, mae: 2.6092, huber: 2.1744, swd: 3.1070, ept: 156.5410
    Epoch [10/50], Test Losses: mse: 17.5747, mae: 2.4350, huber: 2.0027, swd: 3.0670, ept: 159.3241
      Epoch 10 composite train-obj: 2.458901
            Val objective improved 2.2598 → 2.1744, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 23.4722, mae: 2.8278, huber: 2.3984, swd: 3.2203, ept: 44.0550
    Epoch [11/50], Val Losses: mse: 19.0804, mae: 2.3913, huber: 1.9722, swd: 3.1595, ept: 159.5128
    Epoch [11/50], Test Losses: mse: 15.5860, mae: 2.1843, huber: 1.7674, swd: 2.7099, ept: 163.2223
      Epoch 11 composite train-obj: 2.398388
            Val objective improved 2.1744 → 1.9722, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 22.8893, mae: 2.7753, huber: 2.3476, swd: 3.1059, ept: 44.1754
    Epoch [12/50], Val Losses: mse: 20.2946, mae: 2.5354, huber: 2.0995, swd: 3.0270, ept: 160.9214
    Epoch [12/50], Test Losses: mse: 16.2883, mae: 2.3292, huber: 1.8951, swd: 2.7296, ept: 165.7826
      Epoch 12 composite train-obj: 2.347628
            No improvement (2.0995), counter 1/5
    Epoch [13/50], Train Losses: mse: 22.5324, mae: 2.7400, huber: 2.3136, swd: 3.0133, ept: 44.1599
    Epoch [13/50], Val Losses: mse: 20.2224, mae: 2.5905, huber: 2.1568, swd: 4.0665, ept: 159.4296
    Epoch [13/50], Test Losses: mse: 16.8563, mae: 2.4359, huber: 2.0024, swd: 3.9455, ept: 162.6610
      Epoch 13 composite train-obj: 2.313584
            No improvement (2.1568), counter 2/5
    Epoch [14/50], Train Losses: mse: 22.0613, mae: 2.6992, huber: 2.2742, swd: 2.9431, ept: 44.4045
    Epoch [14/50], Val Losses: mse: 19.2459, mae: 2.5261, huber: 2.0919, swd: 2.7841, ept: 161.1732
    Epoch [14/50], Test Losses: mse: 16.4283, mae: 2.3601, huber: 1.9270, swd: 2.2536, ept: 164.1551
      Epoch 14 composite train-obj: 2.274180
            No improvement (2.0919), counter 3/5
    Epoch [15/50], Train Losses: mse: 21.7075, mae: 2.6627, huber: 2.2392, swd: 2.8610, ept: 44.5052
    Epoch [15/50], Val Losses: mse: 18.6916, mae: 2.4226, huber: 1.9909, swd: 2.9146, ept: 162.0583
    Epoch [15/50], Test Losses: mse: 15.2400, mae: 2.2527, huber: 1.8215, swd: 2.5185, ept: 166.5546
      Epoch 15 composite train-obj: 2.239188
            No improvement (1.9909), counter 4/5
    Epoch [16/50], Train Losses: mse: 21.3878, mae: 2.6334, huber: 2.2108, swd: 2.7992, ept: 44.4133
    Epoch [16/50], Val Losses: mse: 18.9753, mae: 2.5508, huber: 2.1105, swd: 3.6996, ept: 162.8467
    Epoch [16/50], Test Losses: mse: 15.5112, mae: 2.3507, huber: 1.9125, swd: 2.9071, ept: 167.2146
      Epoch 16 composite train-obj: 2.210837
    Epoch [16/50], Test Losses: mse: 15.5860, mae: 2.1843, huber: 1.7674, swd: 2.7099, ept: 163.2223
    Best round's Test MSE: 15.5860, MAE: 2.1843, SWD: 2.7099
    Best round's Validation MSE: 19.0804, MAE: 2.3913, SWD: 3.1595
    Best round's Test verification MSE : 15.5860, MAE: 2.1843, SWD: 2.7099
    Time taken: 93.91 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 56.5988, mae: 5.4643, huber: 4.9896, swd: 9.7053, ept: 33.3914
    Epoch [1/50], Val Losses: mse: 41.5947, mae: 4.4278, huber: 3.9636, swd: 9.7066, ept: 94.2398
    Epoch [1/50], Test Losses: mse: 42.2766, mae: 4.4399, huber: 3.9759, swd: 9.3234, ept: 94.1749
      Epoch 1 composite train-obj: 4.989563
            Val objective improved inf → 3.9636, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 41.1601, mae: 4.3941, huber: 3.9292, swd: 7.4534, ept: 40.8251
    Epoch [2/50], Val Losses: mse: 36.6913, mae: 3.8511, huber: 3.3980, swd: 7.2819, ept: 124.1252
    Epoch [2/50], Test Losses: mse: 34.3346, mae: 3.7276, huber: 3.2753, swd: 6.9554, ept: 122.4318
      Epoch 2 composite train-obj: 3.929201
            Val objective improved 3.9636 → 3.3980, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 36.1593, mae: 3.9541, huber: 3.4963, swd: 6.1074, ept: 42.6156
    Epoch [3/50], Val Losses: mse: 30.4059, mae: 3.3710, huber: 2.9261, swd: 5.9391, ept: 135.9663
    Epoch [3/50], Test Losses: mse: 28.3624, mae: 3.2620, huber: 2.8169, swd: 5.7322, ept: 136.3770
      Epoch 3 composite train-obj: 3.496339
            Val objective improved 3.3980 → 2.9261, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 32.3239, mae: 3.6204, huber: 3.1692, swd: 5.1483, ept: 43.2070
    Epoch [4/50], Val Losses: mse: 28.6693, mae: 3.1881, huber: 2.7474, swd: 5.2138, ept: 141.1056
    Epoch [4/50], Test Losses: mse: 26.0114, mae: 3.0467, huber: 2.6071, swd: 4.9615, ept: 142.5849
      Epoch 4 composite train-obj: 3.169244
            Val objective improved 2.9261 → 2.7474, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 29.4461, mae: 3.3902, huber: 2.9435, swd: 4.3926, ept: 43.6891
    Epoch [5/50], Val Losses: mse: 25.6961, mae: 2.9617, huber: 2.5266, swd: 4.3702, ept: 146.5469
    Epoch [5/50], Test Losses: mse: 21.9834, mae: 2.7532, huber: 2.3183, swd: 4.0899, ept: 148.8236
      Epoch 5 composite train-obj: 2.943526
            Val objective improved 2.7474 → 2.5266, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 27.5096, mae: 3.2139, huber: 2.7717, swd: 3.9358, ept: 43.9340
    Epoch [6/50], Val Losses: mse: 24.7668, mae: 2.9230, huber: 2.4837, swd: 4.5253, ept: 150.9751
    Epoch [6/50], Test Losses: mse: 20.4694, mae: 2.6915, huber: 2.2540, swd: 4.0821, ept: 153.2389
      Epoch 6 composite train-obj: 2.771725
            Val objective improved 2.5266 → 2.4837, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 25.9960, mae: 3.0809, huber: 2.6424, swd: 3.6376, ept: 44.1304
    Epoch [7/50], Val Losses: mse: 24.0047, mae: 2.9679, huber: 2.5214, swd: 4.5098, ept: 152.1062
    Epoch [7/50], Test Losses: mse: 19.0042, mae: 2.7007, huber: 2.2559, swd: 4.0670, ept: 156.0589
      Epoch 7 composite train-obj: 2.642357
            No improvement (2.5214), counter 1/5
    Epoch [8/50], Train Losses: mse: 25.0494, mae: 2.9911, huber: 2.5552, swd: 3.4517, ept: 44.1180
    Epoch [8/50], Val Losses: mse: 23.5186, mae: 2.7731, huber: 2.3409, swd: 4.0505, ept: 153.8208
    Epoch [8/50], Test Losses: mse: 18.6327, mae: 2.5138, huber: 2.0821, swd: 3.6601, ept: 158.5211
      Epoch 8 composite train-obj: 2.555234
            Val objective improved 2.4837 → 2.3409, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.4249, mae: 2.9300, huber: 2.4960, swd: 3.3216, ept: 44.1248
    Epoch [9/50], Val Losses: mse: 20.8250, mae: 2.6348, huber: 2.2058, swd: 3.2392, ept: 156.1198
    Epoch [9/50], Test Losses: mse: 17.0102, mae: 2.4134, huber: 1.9865, swd: 2.8031, ept: 159.7222
      Epoch 9 composite train-obj: 2.496041
            Val objective improved 2.3409 → 2.2058, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.4956, mae: 2.8458, huber: 2.4149, swd: 3.1805, ept: 44.3999
    Epoch [10/50], Val Losses: mse: 21.5195, mae: 2.7229, huber: 2.2796, swd: 3.6166, ept: 156.5685
    Epoch [10/50], Test Losses: mse: 17.6842, mae: 2.5286, huber: 2.0871, swd: 3.0291, ept: 160.4695
      Epoch 10 composite train-obj: 2.414917
            No improvement (2.2796), counter 1/5
    Epoch [11/50], Train Losses: mse: 23.0749, mae: 2.8060, huber: 2.3766, swd: 3.0905, ept: 44.2907
    Epoch [11/50], Val Losses: mse: 23.8901, mae: 2.7591, huber: 2.3273, swd: 2.8039, ept: 155.1825
    Epoch [11/50], Test Losses: mse: 19.3132, mae: 2.5579, huber: 2.1261, swd: 2.6804, ept: 159.5933
      Epoch 11 composite train-obj: 2.376555
            No improvement (2.3273), counter 2/5
    Epoch [12/50], Train Losses: mse: 22.5493, mae: 2.7600, huber: 2.3320, swd: 2.9891, ept: 44.3121
    Epoch [12/50], Val Losses: mse: 19.4360, mae: 2.5577, huber: 2.1188, swd: 3.4191, ept: 161.7096
    Epoch [12/50], Test Losses: mse: 14.7586, mae: 2.3039, huber: 1.8664, swd: 2.6899, ept: 166.5777
      Epoch 12 composite train-obj: 2.331965
            Val objective improved 2.2058 → 2.1188, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 22.0782, mae: 2.7187, huber: 2.2919, swd: 2.9212, ept: 44.5580
    Epoch [13/50], Val Losses: mse: 21.1641, mae: 2.7033, huber: 2.2565, swd: 3.2373, ept: 158.2004
    Epoch [13/50], Test Losses: mse: 16.1106, mae: 2.4507, huber: 2.0052, swd: 2.3933, ept: 162.2632
      Epoch 13 composite train-obj: 2.291876
            No improvement (2.2565), counter 1/5
    Epoch [14/50], Train Losses: mse: 21.8562, mae: 2.6920, huber: 2.2666, swd: 2.8496, ept: 44.4529
    Epoch [14/50], Val Losses: mse: 22.8906, mae: 2.6474, huber: 2.2128, swd: 3.1875, ept: 156.8523
    Epoch [14/50], Test Losses: mse: 17.3610, mae: 2.3868, huber: 1.9544, swd: 3.1935, ept: 161.5825
      Epoch 14 composite train-obj: 2.266603
            No improvement (2.2128), counter 2/5
    Epoch [15/50], Train Losses: mse: 21.5539, mae: 2.6626, huber: 2.2382, swd: 2.8076, ept: 44.4837
    Epoch [15/50], Val Losses: mse: 22.0210, mae: 2.5542, huber: 2.1305, swd: 2.6283, ept: 161.3111
    Epoch [15/50], Test Losses: mse: 15.6540, mae: 2.2401, huber: 1.8188, swd: 2.4798, ept: 167.1522
      Epoch 15 composite train-obj: 2.238242
            No improvement (2.1305), counter 3/5
    Epoch [16/50], Train Losses: mse: 21.1561, mae: 2.6233, huber: 2.2005, swd: 2.7417, ept: 44.5132
    Epoch [16/50], Val Losses: mse: 20.5807, mae: 2.4549, huber: 2.0323, swd: 3.1803, ept: 162.5907
    Epoch [16/50], Test Losses: mse: 15.5935, mae: 2.2064, huber: 1.7856, swd: 2.8739, ept: 167.8179
      Epoch 16 composite train-obj: 2.200541
            Val objective improved 2.1188 → 2.0323, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 21.0676, mae: 2.6093, huber: 2.1873, swd: 2.7085, ept: 44.5559
    Epoch [17/50], Val Losses: mse: 17.3413, mae: 2.3790, huber: 1.9465, swd: 2.7523, ept: 162.9856
    Epoch [17/50], Test Losses: mse: 13.6443, mae: 2.1797, huber: 1.7485, swd: 2.2048, ept: 166.9995
      Epoch 17 composite train-obj: 2.187275
            Val objective improved 2.0323 → 1.9465, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 20.6724, mae: 2.5799, huber: 2.1588, swd: 2.6524, ept: 44.5776
    Epoch [18/50], Val Losses: mse: 19.3207, mae: 2.3973, huber: 1.9707, swd: 3.0527, ept: 162.6308
    Epoch [18/50], Test Losses: mse: 14.2164, mae: 2.1215, huber: 1.6975, swd: 2.5867, ept: 168.1148
      Epoch 18 composite train-obj: 2.158757
            No improvement (1.9707), counter 1/5
    Epoch [19/50], Train Losses: mse: 20.6229, mae: 2.5699, huber: 2.1493, swd: 2.6208, ept: 44.6930
    Epoch [19/50], Val Losses: mse: 21.3010, mae: 2.6069, huber: 2.1680, swd: 2.9135, ept: 163.5626
    Epoch [19/50], Test Losses: mse: 16.9660, mae: 2.4099, huber: 1.9724, swd: 2.6792, ept: 167.3915
      Epoch 19 composite train-obj: 2.149250
            No improvement (2.1680), counter 2/5
    Epoch [20/50], Train Losses: mse: 20.1870, mae: 2.5327, huber: 2.1135, swd: 2.5884, ept: 44.7429
    Epoch [20/50], Val Losses: mse: 18.5975, mae: 2.2956, huber: 1.8833, swd: 2.6027, ept: 162.9432
    Epoch [20/50], Test Losses: mse: 14.7049, mae: 2.1243, huber: 1.7127, swd: 2.4359, ept: 166.9791
      Epoch 20 composite train-obj: 2.113504
            Val objective improved 1.9465 → 1.8833, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 19.9928, mae: 2.5158, huber: 2.0972, swd: 2.5289, ept: 44.6159
    Epoch [21/50], Val Losses: mse: 17.0281, mae: 2.4140, huber: 1.9768, swd: 2.9435, ept: 162.4852
    Epoch [21/50], Test Losses: mse: 13.9155, mae: 2.2519, huber: 1.8141, swd: 2.2946, ept: 166.4572
      Epoch 21 composite train-obj: 2.097207
            No improvement (1.9768), counter 1/5
    Epoch [22/50], Train Losses: mse: 20.1069, mae: 2.5205, huber: 2.1018, swd: 2.5287, ept: 44.7278
    Epoch [22/50], Val Losses: mse: 16.8961, mae: 2.2680, huber: 1.8450, swd: 2.4915, ept: 164.8722
    Epoch [22/50], Test Losses: mse: 13.3120, mae: 2.0696, huber: 1.6479, swd: 1.9147, ept: 170.5415
      Epoch 22 composite train-obj: 2.101753
            Val objective improved 1.8833 → 1.8450, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 19.8346, mae: 2.4926, huber: 2.0754, swd: 2.4931, ept: 44.8714
    Epoch [23/50], Val Losses: mse: 15.6515, mae: 2.2517, huber: 1.8168, swd: 2.8893, ept: 165.7005
    Epoch [23/50], Test Losses: mse: 12.4410, mae: 2.0586, huber: 1.6268, swd: 2.2509, ept: 171.1160
      Epoch 23 composite train-obj: 2.075402
            Val objective improved 1.8450 → 1.8168, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 19.6766, mae: 2.4829, huber: 2.0656, swd: 2.4627, ept: 44.7665
    Epoch [24/50], Val Losses: mse: 19.7969, mae: 2.4301, huber: 2.0102, swd: 2.6026, ept: 166.0104
    Epoch [24/50], Test Losses: mse: 15.4869, mae: 2.2404, huber: 1.8196, swd: 2.2614, ept: 170.4360
      Epoch 24 composite train-obj: 2.065591
            No improvement (2.0102), counter 1/5
    Epoch [25/50], Train Losses: mse: 19.4154, mae: 2.4582, huber: 2.0420, swd: 2.4351, ept: 44.6621
    Epoch [25/50], Val Losses: mse: 22.4685, mae: 2.7994, huber: 2.3625, swd: 2.9408, ept: 165.4547
    Epoch [25/50], Test Losses: mse: 17.6075, mae: 2.5750, huber: 2.1394, swd: 2.7094, ept: 167.7224
      Epoch 25 composite train-obj: 2.041976
            No improvement (2.3625), counter 2/5
    Epoch [26/50], Train Losses: mse: 19.3430, mae: 2.4488, huber: 2.0332, swd: 2.3912, ept: 44.6879
    Epoch [26/50], Val Losses: mse: 16.7437, mae: 2.1305, huber: 1.7143, swd: 2.4530, ept: 166.0520
    Epoch [26/50], Test Losses: mse: 12.2708, mae: 1.9091, huber: 1.4943, swd: 2.0182, ept: 172.9503
      Epoch 26 composite train-obj: 2.033197
            Val objective improved 1.8168 → 1.7143, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 19.1333, mae: 2.4272, huber: 2.0126, swd: 2.3666, ept: 44.7363
    Epoch [27/50], Val Losses: mse: 15.6736, mae: 2.1886, huber: 1.7727, swd: 2.6452, ept: 164.1547
    Epoch [27/50], Test Losses: mse: 12.1394, mae: 1.9890, huber: 1.5732, swd: 2.0285, ept: 168.9553
      Epoch 27 composite train-obj: 2.012580
            No improvement (1.7727), counter 1/5
    Epoch [28/50], Train Losses: mse: 19.1041, mae: 2.4248, huber: 2.0104, swd: 2.3624, ept: 44.6938
    Epoch [28/50], Val Losses: mse: 16.2567, mae: 2.1694, huber: 1.7454, swd: 2.5585, ept: 166.9601
    Epoch [28/50], Test Losses: mse: 11.9590, mae: 1.9482, huber: 1.5250, swd: 1.9566, ept: 171.6834
      Epoch 28 composite train-obj: 2.010369
            No improvement (1.7454), counter 2/5
    Epoch [29/50], Train Losses: mse: 19.1045, mae: 2.4224, huber: 2.0082, swd: 2.3505, ept: 44.6930
    Epoch [29/50], Val Losses: mse: 16.0592, mae: 2.1302, huber: 1.7155, swd: 2.5230, ept: 166.5907
    Epoch [29/50], Test Losses: mse: 11.2358, mae: 1.8887, huber: 1.4739, swd: 2.0436, ept: 171.0153
      Epoch 29 composite train-obj: 2.008211
            No improvement (1.7155), counter 3/5
    Epoch [30/50], Train Losses: mse: 18.8075, mae: 2.3978, huber: 1.9843, swd: 2.3010, ept: 44.8017
    Epoch [30/50], Val Losses: mse: 16.8677, mae: 2.3309, huber: 1.8945, swd: 2.5760, ept: 167.2423
    Epoch [30/50], Test Losses: mse: 12.3973, mae: 2.1018, huber: 1.6658, swd: 1.9738, ept: 173.1510
      Epoch 30 composite train-obj: 1.984259
            No improvement (1.8945), counter 4/5
    Epoch [31/50], Train Losses: mse: 18.8588, mae: 2.4030, huber: 1.9892, swd: 2.3208, ept: 44.7942
    Epoch [31/50], Val Losses: mse: 15.5427, mae: 2.0953, huber: 1.6884, swd: 2.8917, ept: 168.4910
    Epoch [31/50], Test Losses: mse: 11.1333, mae: 1.8474, huber: 1.4422, swd: 2.3409, ept: 174.3397
      Epoch 31 composite train-obj: 1.989182
            Val objective improved 1.7143 → 1.6884, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 18.8163, mae: 2.3958, huber: 1.9825, swd: 2.2895, ept: 44.7440
    Epoch [32/50], Val Losses: mse: 15.9660, mae: 2.2180, huber: 1.7977, swd: 3.1414, ept: 165.9035
    Epoch [32/50], Test Losses: mse: 11.6081, mae: 1.9660, huber: 1.5477, swd: 2.5874, ept: 169.9879
      Epoch 32 composite train-obj: 1.982523
            No improvement (1.7977), counter 1/5
    Epoch [33/50], Train Losses: mse: 18.5041, mae: 2.3642, huber: 1.9524, swd: 2.2386, ept: 44.7850
    Epoch [33/50], Val Losses: mse: 15.8584, mae: 2.0706, huber: 1.6559, swd: 2.2966, ept: 169.6986
    Epoch [33/50], Test Losses: mse: 10.8696, mae: 1.8051, huber: 1.3917, swd: 1.9827, ept: 174.5247
      Epoch 33 composite train-obj: 1.952410
            Val objective improved 1.6884 → 1.6559, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 18.7192, mae: 2.3839, huber: 1.9712, swd: 2.2609, ept: 44.8378
    Epoch [34/50], Val Losses: mse: 15.0732, mae: 2.0159, huber: 1.6045, swd: 2.5750, ept: 168.6536
    Epoch [34/50], Test Losses: mse: 10.2918, mae: 1.7495, huber: 1.3421, swd: 1.9862, ept: 173.4763
      Epoch 34 composite train-obj: 1.971247
            Val objective improved 1.6559 → 1.6045, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 18.3810, mae: 2.3541, huber: 1.9427, swd: 2.2196, ept: 44.9243
    Epoch [35/50], Val Losses: mse: 15.2160, mae: 2.0424, huber: 1.6369, swd: 2.3757, ept: 166.7547
    Epoch [35/50], Test Losses: mse: 11.5571, mae: 1.8138, huber: 1.4113, swd: 1.6825, ept: 173.0193
      Epoch 35 composite train-obj: 1.942657
            No improvement (1.6369), counter 1/5
    Epoch [36/50], Train Losses: mse: 18.4631, mae: 2.3610, huber: 1.9494, swd: 2.2075, ept: 44.8617
    Epoch [36/50], Val Losses: mse: 17.4732, mae: 2.0389, huber: 1.6450, swd: 2.3123, ept: 168.2361
    Epoch [36/50], Test Losses: mse: 12.1569, mae: 1.7783, huber: 1.3877, swd: 2.2181, ept: 172.7528
      Epoch 36 composite train-obj: 1.949378
            No improvement (1.6450), counter 2/5
    Epoch [37/50], Train Losses: mse: 18.2351, mae: 2.3378, huber: 1.9274, swd: 2.1628, ept: 44.7664
    Epoch [37/50], Val Losses: mse: 16.6539, mae: 2.0793, huber: 1.6705, swd: 2.5212, ept: 169.6259
    Epoch [37/50], Test Losses: mse: 12.4945, mae: 1.8779, huber: 1.4696, swd: 2.0687, ept: 174.6850
      Epoch 37 composite train-obj: 1.927396
            No improvement (1.6705), counter 3/5
    Epoch [38/50], Train Losses: mse: 18.1308, mae: 2.3300, huber: 1.9196, swd: 2.1610, ept: 44.8594
    Epoch [38/50], Val Losses: mse: 15.6383, mae: 2.1073, huber: 1.6949, swd: 2.9525, ept: 169.0581
    Epoch [38/50], Test Losses: mse: 11.4748, mae: 1.8729, huber: 1.4637, swd: 2.6312, ept: 173.7813
      Epoch 38 composite train-obj: 1.919569
            No improvement (1.6949), counter 4/5
    Epoch [39/50], Train Losses: mse: 18.1158, mae: 2.3236, huber: 1.9140, swd: 2.1399, ept: 44.8306
    Epoch [39/50], Val Losses: mse: 16.6042, mae: 2.0967, huber: 1.6918, swd: 2.1375, ept: 170.2804
    Epoch [39/50], Test Losses: mse: 11.5747, mae: 1.8571, huber: 1.4531, swd: 1.9501, ept: 175.5520
      Epoch 39 composite train-obj: 1.913999
    Epoch [39/50], Test Losses: mse: 10.2918, mae: 1.7495, huber: 1.3421, swd: 1.9862, ept: 173.4763
    Best round's Test MSE: 10.2918, MAE: 1.7495, SWD: 1.9862
    Best round's Validation MSE: 15.0732, MAE: 2.0159, SWD: 2.5750
    Best round's Test verification MSE : 10.2918, MAE: 1.7495, SWD: 1.9862
    Time taken: 235.35 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq336_pred196_20250513_2313)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.5343 ± 2.2358
      mae: 1.9268 ± 0.1863
      huber: 1.5164 ± 0.1819
      swd: 2.2511 ± 0.3257
      ept: 169.2482 ± 4.3749
      count: 37.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 16.7561 ± 1.6976
      mae: 2.1743 ± 0.1588
      huber: 1.7608 ± 0.1551
      swd: 2.8594 ± 0.2389
      ept: 164.0489 ± 3.7320
      count: 37.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 477.30 seconds
    
    Experiment complete: PatchTST_lorenz_seq336_pred196_20250513_2313
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 336-336
##### huber


```python
utils.reload_modules([utils])
cfg_patch_tst = train_config.FlatPatchTSTConfig(
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],
    enc_in=data_mgr.datasets['lorenz']['channels'],
    dec_in=data_mgr.datasets['lorenz']['channels'],
    c_out=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0]
)
exp_patch_tst = execute_model_evaluation('lorenz', cfg_patch_tst, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 280
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
    Training Batches: 280
    Validation Batches: 36
    Test Batches: 77
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 65.3053, mae: 6.0007, huber: 5.5227, swd: 11.1108, target_std: 13.6398
    Epoch [1/50], Val Losses: mse: 55.2165, mae: 5.3721, huber: 4.8986, swd: 13.1801, target_std: 13.8286
    Epoch [1/50], Test Losses: mse: 55.0193, mae: 5.3277, huber: 4.8549, swd: 12.8882, target_std: 13.1386
      Epoch 1 composite train-obj: 5.522692
            Val objective improved inf → 4.8986, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 51.5022, mae: 5.1341, huber: 4.6628, swd: 9.8333, target_std: 13.6396
    Epoch [2/50], Val Losses: mse: 50.8270, mae: 4.9500, huber: 4.4830, swd: 10.5435, target_std: 13.8286
    Epoch [2/50], Test Losses: mse: 48.1456, mae: 4.8057, huber: 4.3396, swd: 11.1222, target_std: 13.1386
      Epoch 2 composite train-obj: 4.662762
            Val objective improved 4.8986 → 4.4830, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 46.8786, mae: 4.7423, huber: 4.2762, swd: 8.5763, target_std: 13.6415
    Epoch [3/50], Val Losses: mse: 50.0297, mae: 4.7524, huber: 4.2900, swd: 9.6594, target_std: 13.8286
    Epoch [3/50], Test Losses: mse: 46.1596, mae: 4.5362, huber: 4.0754, swd: 9.7776, target_std: 13.1386
      Epoch 3 composite train-obj: 4.276238
            Val objective improved 4.4830 → 4.2900, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 44.0734, mae: 4.4827, huber: 4.0210, swd: 7.7535, target_std: 13.6383
    Epoch [4/50], Val Losses: mse: 48.7624, mae: 4.5991, huber: 4.1416, swd: 8.8898, target_std: 13.8286
    Epoch [4/50], Test Losses: mse: 43.9936, mae: 4.3194, huber: 3.8639, swd: 8.7646, target_std: 13.1386
      Epoch 4 composite train-obj: 4.020999
            Val objective improved 4.2900 → 4.1416, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 41.9480, mae: 4.2952, huber: 3.8372, swd: 7.1954, target_std: 13.6401
    Epoch [5/50], Val Losses: mse: 47.8576, mae: 4.5431, huber: 4.0825, swd: 8.1408, target_std: 13.8286
    Epoch [5/50], Test Losses: mse: 42.6415, mae: 4.2437, huber: 3.7856, swd: 7.6614, target_std: 13.1386
      Epoch 5 composite train-obj: 3.837166
            Val objective improved 4.1416 → 4.0825, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 40.5846, mae: 4.1748, huber: 3.7192, swd: 6.7874, target_std: 13.6423
    Epoch [6/50], Val Losses: mse: 47.8032, mae: 4.4498, huber: 3.9955, swd: 7.3865, target_std: 13.8286
    Epoch [6/50], Test Losses: mse: 42.7030, mae: 4.1775, huber: 3.7240, swd: 7.0740, target_std: 13.1386
      Epoch 6 composite train-obj: 3.719207
            Val objective improved 4.0825 → 3.9955, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 39.3557, mae: 4.0687, huber: 3.6154, swd: 6.4060, target_std: 13.6393
    Epoch [7/50], Val Losses: mse: 46.9373, mae: 4.4459, huber: 3.9891, swd: 7.7203, target_std: 13.8286
    Epoch [7/50], Test Losses: mse: 40.6452, mae: 4.0883, huber: 3.6351, swd: 7.4862, target_std: 13.1386
      Epoch 7 composite train-obj: 3.615418
            Val objective improved 3.9955 → 3.9891, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 38.3200, mae: 3.9882, huber: 3.5364, swd: 6.0610, target_std: 13.6381
    Epoch [8/50], Val Losses: mse: 46.2020, mae: 4.3098, huber: 3.8546, swd: 6.6685, target_std: 13.8286
    Epoch [8/50], Test Losses: mse: 40.6718, mae: 4.0064, huber: 3.5537, swd: 6.5529, target_std: 13.1386
      Epoch 8 composite train-obj: 3.536424
            Val objective improved 3.9891 → 3.8546, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 37.4590, mae: 3.9146, huber: 3.4644, swd: 5.7693, target_std: 13.6396
    Epoch [9/50], Val Losses: mse: 45.6578, mae: 4.2294, huber: 3.7789, swd: 6.1210, target_std: 13.8286
    Epoch [9/50], Test Losses: mse: 38.8767, mae: 3.8673, huber: 3.4201, swd: 5.9770, target_std: 13.1386
      Epoch 9 composite train-obj: 3.464411
            Val objective improved 3.8546 → 3.7789, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 36.5193, mae: 3.8446, huber: 3.3956, swd: 5.4251, target_std: 13.6389
    Epoch [10/50], Val Losses: mse: 47.5213, mae: 4.3192, huber: 3.8641, swd: 5.6296, target_std: 13.8286
    Epoch [10/50], Test Losses: mse: 39.8226, mae: 3.9190, huber: 3.4670, swd: 5.3523, target_std: 13.1386
      Epoch 10 composite train-obj: 3.395648
            No improvement (3.8641), counter 1/5
    Epoch [11/50], Train Losses: mse: 35.9548, mae: 3.7949, huber: 3.3470, swd: 5.1961, target_std: 13.6402
    Epoch [11/50], Val Losses: mse: 44.9723, mae: 4.2155, huber: 3.7597, swd: 5.1338, target_std: 13.8286
    Epoch [11/50], Test Losses: mse: 38.1749, mae: 3.8639, huber: 3.4095, swd: 5.1500, target_std: 13.1386
      Epoch 11 composite train-obj: 3.347039
            Val objective improved 3.7789 → 3.7597, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 35.2974, mae: 3.7453, huber: 3.2982, swd: 5.0179, target_std: 13.6384
    Epoch [12/50], Val Losses: mse: 48.9972, mae: 4.4738, huber: 4.0127, swd: 6.2134, target_std: 13.8286
    Epoch [12/50], Test Losses: mse: 40.9764, mae: 4.0849, huber: 3.6265, swd: 6.0772, target_std: 13.1386
      Epoch 12 composite train-obj: 3.298207
            No improvement (4.0127), counter 1/5
    Epoch [13/50], Train Losses: mse: 34.8759, mae: 3.7128, huber: 3.2663, swd: 4.8907, target_std: 13.6419
    Epoch [13/50], Val Losses: mse: 46.0466, mae: 4.1889, huber: 3.7381, swd: 5.4118, target_std: 13.8286
    Epoch [13/50], Test Losses: mse: 37.9322, mae: 3.8014, huber: 3.3530, swd: 5.7119, target_std: 13.1386
      Epoch 13 composite train-obj: 3.266330
            Val objective improved 3.7597 → 3.7381, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 34.4166, mae: 3.6758, huber: 3.2301, swd: 4.7413, target_std: 13.6417
    Epoch [14/50], Val Losses: mse: 45.6184, mae: 4.1136, huber: 3.6653, swd: 4.7061, target_std: 13.8286
    Epoch [14/50], Test Losses: mse: 37.3535, mae: 3.7333, huber: 3.2871, swd: 4.8323, target_std: 13.1386
      Epoch 14 composite train-obj: 3.230066
            Val objective improved 3.7381 → 3.6653, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 34.1761, mae: 3.6568, huber: 3.2115, swd: 4.6731, target_std: 13.6408
    Epoch [15/50], Val Losses: mse: 44.7867, mae: 4.0766, huber: 3.6282, swd: 4.7992, target_std: 13.8286
    Epoch [15/50], Test Losses: mse: 37.3885, mae: 3.6785, huber: 3.2333, swd: 4.3328, target_std: 13.1386
      Epoch 15 composite train-obj: 3.211536
            Val objective improved 3.6653 → 3.6282, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 33.6686, mae: 3.6172, huber: 3.1727, swd: 4.5261, target_std: 13.6403
    Epoch [16/50], Val Losses: mse: 44.0792, mae: 4.0097, huber: 3.5647, swd: 4.8156, target_std: 13.8286
    Epoch [16/50], Test Losses: mse: 37.0055, mae: 3.6684, huber: 3.2257, swd: 4.6991, target_std: 13.1386
      Epoch 16 composite train-obj: 3.172738
            Val objective improved 3.6282 → 3.5647, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 33.1446, mae: 3.5776, huber: 3.1340, swd: 4.4193, target_std: 13.6397
    Epoch [17/50], Val Losses: mse: 47.7741, mae: 4.2477, huber: 3.7998, swd: 3.6287, target_std: 13.8286
    Epoch [17/50], Test Losses: mse: 40.7319, mae: 3.9101, huber: 3.4634, swd: 3.5375, target_std: 13.1386
      Epoch 17 composite train-obj: 3.134005
            No improvement (3.7998), counter 1/5
    Epoch [18/50], Train Losses: mse: 33.0512, mae: 3.5690, huber: 3.1256, swd: 4.3740, target_std: 13.6392
    Epoch [18/50], Val Losses: mse: 45.4215, mae: 4.1615, huber: 3.7079, swd: 4.7206, target_std: 13.8286
    Epoch [18/50], Test Losses: mse: 36.9891, mae: 3.7544, huber: 3.3032, swd: 4.9854, target_std: 13.1386
      Epoch 18 composite train-obj: 3.125593
            No improvement (3.7079), counter 2/5
    Epoch [19/50], Train Losses: mse: 32.6965, mae: 3.5432, huber: 3.1003, swd: 4.2823, target_std: 13.6414
    Epoch [19/50], Val Losses: mse: 42.3955, mae: 3.9841, huber: 3.5363, swd: 4.7768, target_std: 13.8286
    Epoch [19/50], Test Losses: mse: 35.4659, mae: 3.5865, huber: 3.1429, swd: 4.4543, target_std: 13.1386
      Epoch 19 composite train-obj: 3.100281
            Val objective improved 3.5647 → 3.5363, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 32.4086, mae: 3.5222, huber: 3.0796, swd: 4.2240, target_std: 13.6377
    Epoch [20/50], Val Losses: mse: 43.0188, mae: 4.0273, huber: 3.5755, swd: 4.5097, target_std: 13.8286
    Epoch [20/50], Test Losses: mse: 35.9468, mae: 3.6666, huber: 3.2172, swd: 4.1866, target_std: 13.1386
      Epoch 20 composite train-obj: 3.079559
            No improvement (3.5755), counter 1/5
    Epoch [21/50], Train Losses: mse: 32.0166, mae: 3.4916, huber: 3.0497, swd: 4.1308, target_std: 13.6399
    Epoch [21/50], Val Losses: mse: 43.7532, mae: 4.0781, huber: 3.6250, swd: 5.0083, target_std: 13.8286
    Epoch [21/50], Test Losses: mse: 36.1220, mae: 3.6950, huber: 3.2439, swd: 4.6770, target_std: 13.1386
      Epoch 21 composite train-obj: 3.049728
            No improvement (3.6250), counter 2/5
    Epoch [22/50], Train Losses: mse: 31.8667, mae: 3.4816, huber: 3.0397, swd: 4.0942, target_std: 13.6401
    Epoch [22/50], Val Losses: mse: 43.8885, mae: 3.9983, huber: 3.5467, swd: 4.2714, target_std: 13.8286
    Epoch [22/50], Test Losses: mse: 36.2616, mae: 3.6306, huber: 3.1820, swd: 4.1075, target_std: 13.1386
      Epoch 22 composite train-obj: 3.039735
            No improvement (3.5467), counter 3/5
    Epoch [23/50], Train Losses: mse: 31.5503, mae: 3.4551, huber: 3.0140, swd: 4.0080, target_std: 13.6405
    Epoch [23/50], Val Losses: mse: 42.1448, mae: 3.8688, huber: 3.4286, swd: 4.4869, target_std: 13.8286
    Epoch [23/50], Test Losses: mse: 34.5258, mae: 3.4840, huber: 3.0452, swd: 4.3581, target_std: 13.1386
      Epoch 23 composite train-obj: 3.013960
            Val objective improved 3.5363 → 3.4286, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 31.4294, mae: 3.4466, huber: 3.0056, swd: 3.9686, target_std: 13.6404
    Epoch [24/50], Val Losses: mse: 48.6588, mae: 4.1617, huber: 3.7189, swd: 4.2084, target_std: 13.8286
    Epoch [24/50], Test Losses: mse: 40.6347, mae: 3.8081, huber: 3.3668, swd: 4.5718, target_std: 13.1386
      Epoch 24 composite train-obj: 3.005623
            No improvement (3.7189), counter 1/5
    Epoch [25/50], Train Losses: mse: 31.3478, mae: 3.4400, huber: 2.9991, swd: 3.9583, target_std: 13.6396
    Epoch [25/50], Val Losses: mse: 43.5625, mae: 3.9044, huber: 3.4628, swd: 4.2572, target_std: 13.8286
    Epoch [25/50], Test Losses: mse: 34.8243, mae: 3.4672, huber: 3.0287, swd: 4.0173, target_std: 13.1386
      Epoch 25 composite train-obj: 2.999072
            No improvement (3.4628), counter 2/5
    Epoch [26/50], Train Losses: mse: 31.1188, mae: 3.4241, huber: 2.9836, swd: 3.8903, target_std: 13.6401
    Epoch [26/50], Val Losses: mse: 44.7621, mae: 3.9962, huber: 3.5515, swd: 4.7066, target_std: 13.8286
    Epoch [26/50], Test Losses: mse: 36.0552, mae: 3.5701, huber: 3.1288, swd: 4.8028, target_std: 13.1386
      Epoch 26 composite train-obj: 2.983559
            No improvement (3.5515), counter 3/5
    Epoch [27/50], Train Losses: mse: 30.8123, mae: 3.3988, huber: 2.9588, swd: 3.8360, target_std: 13.6408
    Epoch [27/50], Val Losses: mse: 42.1866, mae: 3.9440, huber: 3.4895, swd: 4.2785, target_std: 13.8286
    Epoch [27/50], Test Losses: mse: 33.9767, mae: 3.4953, huber: 3.0435, swd: 3.5693, target_std: 13.1386
      Epoch 27 composite train-obj: 2.958800
            No improvement (3.4895), counter 4/5
    Epoch [28/50], Train Losses: mse: 30.6283, mae: 3.3834, huber: 2.9438, swd: 3.7862, target_std: 13.6432
    Epoch [28/50], Val Losses: mse: 41.5889, mae: 3.8378, huber: 3.3952, swd: 4.3254, target_std: 13.8286
    Epoch [28/50], Test Losses: mse: 33.5314, mae: 3.4197, huber: 2.9803, swd: 3.8389, target_std: 13.1386
      Epoch 28 composite train-obj: 2.943784
            Val objective improved 3.4286 → 3.3952, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 30.5068, mae: 3.3724, huber: 2.9331, swd: 3.7474, target_std: 13.6397
    Epoch [29/50], Val Losses: mse: 42.7999, mae: 3.9356, huber: 3.4915, swd: 4.6002, target_std: 13.8286
    Epoch [29/50], Test Losses: mse: 34.7175, mae: 3.5452, huber: 3.1031, swd: 4.4152, target_std: 13.1386
      Epoch 29 composite train-obj: 2.933053
            No improvement (3.4915), counter 1/5
    Epoch [30/50], Train Losses: mse: 30.5063, mae: 3.3705, huber: 2.9313, swd: 3.7538, target_std: 13.6404
    Epoch [30/50], Val Losses: mse: 42.6570, mae: 3.9204, huber: 3.4764, swd: 4.9811, target_std: 13.8286
    Epoch [30/50], Test Losses: mse: 33.9160, mae: 3.4798, huber: 3.0386, swd: 4.5899, target_std: 13.1386
      Epoch 30 composite train-obj: 2.931277
            No improvement (3.4764), counter 2/5
    Epoch [31/50], Train Losses: mse: 30.2181, mae: 3.3544, huber: 2.9151, swd: 3.6939, target_std: 13.6398
    Epoch [31/50], Val Losses: mse: 41.2655, mae: 3.8547, huber: 3.4085, swd: 4.7456, target_std: 13.8286
    Epoch [31/50], Test Losses: mse: 33.5736, mae: 3.4516, huber: 3.0094, swd: 4.3801, target_std: 13.1386
      Epoch 31 composite train-obj: 2.915135
            No improvement (3.4085), counter 3/5
    Epoch [32/50], Train Losses: mse: 29.9324, mae: 3.3333, huber: 2.8946, swd: 3.6558, target_std: 13.6423
    Epoch [32/50], Val Losses: mse: 42.7115, mae: 4.0383, huber: 3.5850, swd: 4.7960, target_std: 13.8286
    Epoch [32/50], Test Losses: mse: 34.3907, mae: 3.5966, huber: 3.1460, swd: 4.4806, target_std: 13.1386
      Epoch 32 composite train-obj: 2.894589
            No improvement (3.5850), counter 4/5
    Epoch [33/50], Train Losses: mse: 29.9098, mae: 3.3291, huber: 2.8906, swd: 3.6168, target_std: 13.6374
    Epoch [33/50], Val Losses: mse: 43.1387, mae: 3.9315, huber: 3.4906, swd: 4.6781, target_std: 13.8286
    Epoch [33/50], Test Losses: mse: 33.6161, mae: 3.4425, huber: 3.0052, swd: 4.4920, target_std: 13.1386
      Epoch 33 composite train-obj: 2.890607
    Epoch [33/50], Test Losses: mse: 33.5314, mae: 3.4197, huber: 2.9803, swd: 3.8389, target_std: 13.1386
    Best round's Test MSE: 33.5314, MAE: 3.4197, SWD: 3.8389
    Best round's Validation MSE: 41.5889, MAE: 3.8378, SWD: 4.3254
    Best round's Test verification MSE : 33.5314, MAE: 3.4197, SWD: 3.8389
    Time taken: 188.64 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 65.0795, mae: 5.9882, huber: 5.5103, swd: 11.2080, target_std: 13.6410
    Epoch [1/50], Val Losses: mse: 56.5024, mae: 5.5368, huber: 5.0619, swd: 12.5434, target_std: 13.8286
    Epoch [1/50], Test Losses: mse: 55.5110, mae: 5.4417, huber: 4.9674, swd: 12.6693, target_std: 13.1386
      Epoch 1 composite train-obj: 5.510294
            Val objective improved inf → 5.0619, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 50.7930, mae: 5.0711, huber: 4.6005, swd: 9.6427, target_std: 13.6401
    Epoch [2/50], Val Losses: mse: 49.6157, mae: 4.8955, huber: 4.4293, swd: 10.9899, target_std: 13.8286
    Epoch [2/50], Test Losses: mse: 48.1255, mae: 4.7755, huber: 4.3099, swd: 10.7290, target_std: 13.1386
      Epoch 2 composite train-obj: 4.600486
            Val objective improved 5.0619 → 4.4293, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 45.9797, mae: 4.6710, huber: 4.2057, swd: 8.4183, target_std: 13.6432
    Epoch [3/50], Val Losses: mse: 49.0130, mae: 4.6290, huber: 4.1716, swd: 8.6182, target_std: 13.8286
    Epoch [3/50], Test Losses: mse: 45.2860, mae: 4.3714, huber: 3.9163, swd: 8.5666, target_std: 13.1386
      Epoch 3 composite train-obj: 4.205715
            Val objective improved 4.4293 → 4.1716, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 43.0260, mae: 4.4084, huber: 3.9477, swd: 7.4647, target_std: 13.6374
    Epoch [4/50], Val Losses: mse: 47.4791, mae: 4.5099, huber: 4.0537, swd: 8.2180, target_std: 13.8286
    Epoch [4/50], Test Losses: mse: 42.2704, mae: 4.2168, huber: 3.7627, swd: 8.2903, target_std: 13.1386
      Epoch 4 composite train-obj: 3.947736
            Val objective improved 4.1716 → 4.0537, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 41.3623, mae: 4.2559, huber: 3.7983, swd: 7.0007, target_std: 13.6393
    Epoch [5/50], Val Losses: mse: 47.6433, mae: 4.4448, huber: 3.9892, swd: 7.5611, target_std: 13.8286
    Epoch [5/50], Test Losses: mse: 42.1312, mae: 4.1627, huber: 3.7078, swd: 7.6203, target_std: 13.1386
      Epoch 5 composite train-obj: 3.798336
            Val objective improved 4.0537 → 3.9892, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 39.6963, mae: 4.1091, huber: 3.6546, swd: 6.4326, target_std: 13.6416
    Epoch [6/50], Val Losses: mse: 46.9377, mae: 4.3447, huber: 3.8931, swd: 6.8412, target_std: 13.8286
    Epoch [6/50], Test Losses: mse: 39.4988, mae: 3.9674, huber: 3.5172, swd: 6.9292, target_std: 13.1386
      Epoch 6 composite train-obj: 3.654572
            Val objective improved 3.9892 → 3.8931, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 38.5172, mae: 4.0082, huber: 3.5559, swd: 6.0583, target_std: 13.6404
    Epoch [7/50], Val Losses: mse: 45.7414, mae: 4.3214, huber: 3.8650, swd: 7.3023, target_std: 13.8286
    Epoch [7/50], Test Losses: mse: 39.8018, mae: 4.0226, huber: 3.5678, swd: 6.9331, target_std: 13.1386
      Epoch 7 composite train-obj: 3.555935
            Val objective improved 3.8931 → 3.8650, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 37.7552, mae: 3.9405, huber: 3.4898, swd: 5.7620, target_std: 13.6429
    Epoch [8/50], Val Losses: mse: 48.0333, mae: 4.4595, huber: 3.9976, swd: 6.0341, target_std: 13.8286
    Epoch [8/50], Test Losses: mse: 40.6642, mae: 4.0735, huber: 3.6132, swd: 5.2657, target_std: 13.1386
      Epoch 8 composite train-obj: 3.489776
            No improvement (3.9976), counter 1/5
    Epoch [9/50], Train Losses: mse: 36.9962, mae: 3.8754, huber: 3.4260, swd: 5.5290, target_std: 13.6399
    Epoch [9/50], Val Losses: mse: 47.2828, mae: 4.3266, huber: 3.8744, swd: 5.2630, target_std: 13.8286
    Epoch [9/50], Test Losses: mse: 40.9777, mae: 3.9962, huber: 3.5459, swd: 4.9474, target_std: 13.1386
      Epoch 9 composite train-obj: 3.426034
            No improvement (3.8744), counter 2/5
    Epoch [10/50], Train Losses: mse: 36.4898, mae: 3.8315, huber: 3.3831, swd: 5.3325, target_std: 13.6405
    Epoch [10/50], Val Losses: mse: 45.7687, mae: 4.2243, huber: 3.7722, swd: 5.8744, target_std: 13.8286
    Epoch [10/50], Test Losses: mse: 38.2619, mae: 3.8359, huber: 3.3875, swd: 5.6876, target_std: 13.1386
      Epoch 10 composite train-obj: 3.383070
            Val objective improved 3.8650 → 3.7722, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 35.7286, mae: 3.7717, huber: 3.3245, swd: 5.0925, target_std: 13.6411
    Epoch [11/50], Val Losses: mse: 49.0689, mae: 4.3679, huber: 3.9125, swd: 4.9814, target_std: 13.8286
    Epoch [11/50], Test Losses: mse: 40.4563, mae: 3.9513, huber: 3.4977, swd: 4.7875, target_std: 13.1386
      Epoch 11 composite train-obj: 3.324506
            No improvement (3.9125), counter 1/5
    Epoch [12/50], Train Losses: mse: 35.2429, mae: 3.7292, huber: 3.2831, swd: 4.9468, target_std: 13.6393
    Epoch [12/50], Val Losses: mse: 46.7492, mae: 4.2950, huber: 3.8424, swd: 5.0677, target_std: 13.8286
    Epoch [12/50], Test Losses: mse: 39.7791, mae: 3.9162, huber: 3.4669, swd: 5.0678, target_std: 13.1386
      Epoch 12 composite train-obj: 3.283106
            No improvement (3.8424), counter 2/5
    Epoch [13/50], Train Losses: mse: 34.9331, mae: 3.7045, huber: 3.2589, swd: 4.8660, target_std: 13.6408
    Epoch [13/50], Val Losses: mse: 46.8371, mae: 4.2857, huber: 3.8304, swd: 4.3353, target_std: 13.8286
    Epoch [13/50], Test Losses: mse: 39.9645, mae: 3.9298, huber: 3.4765, swd: 4.5490, target_std: 13.1386
      Epoch 13 composite train-obj: 3.258915
            No improvement (3.8304), counter 3/5
    Epoch [14/50], Train Losses: mse: 34.4105, mae: 3.6608, huber: 3.2161, swd: 4.7134, target_std: 13.6409
    Epoch [14/50], Val Losses: mse: 47.1709, mae: 4.1983, huber: 3.7513, swd: 4.6493, target_std: 13.8286
    Epoch [14/50], Test Losses: mse: 39.6852, mae: 3.8150, huber: 3.3697, swd: 4.4933, target_std: 13.1386
      Epoch 14 composite train-obj: 3.216085
            Val objective improved 3.7722 → 3.7513, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 34.1906, mae: 3.6445, huber: 3.2002, swd: 4.6221, target_std: 13.6362
    Epoch [15/50], Val Losses: mse: 46.2161, mae: 4.1443, huber: 3.6968, swd: 4.7090, target_std: 13.8286
    Epoch [15/50], Test Losses: mse: 37.6194, mae: 3.7172, huber: 3.2721, swd: 4.8616, target_std: 13.1386
      Epoch 15 composite train-obj: 3.200175
            Val objective improved 3.7513 → 3.6968, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 33.6371, mae: 3.6042, huber: 3.1606, swd: 4.5175, target_std: 13.6402
    Epoch [16/50], Val Losses: mse: 47.1442, mae: 4.2658, huber: 3.8121, swd: 5.3455, target_std: 13.8286
    Epoch [16/50], Test Losses: mse: 38.4361, mae: 3.8363, huber: 3.3849, swd: 4.9210, target_std: 13.1386
      Epoch 16 composite train-obj: 3.160630
            No improvement (3.8121), counter 1/5
    Epoch [17/50], Train Losses: mse: 33.4690, mae: 3.5894, huber: 3.1461, swd: 4.4462, target_std: 13.6360
    Epoch [17/50], Val Losses: mse: 47.4136, mae: 4.1148, huber: 3.6723, swd: 4.4968, target_std: 13.8286
    Epoch [17/50], Test Losses: mse: 37.1953, mae: 3.6612, huber: 3.2200, swd: 4.7602, target_std: 13.1386
      Epoch 17 composite train-obj: 3.146144
            Val objective improved 3.6968 → 3.6723, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 33.0784, mae: 3.5595, huber: 3.1170, swd: 4.3876, target_std: 13.6389
    Epoch [18/50], Val Losses: mse: 45.6472, mae: 4.0580, huber: 3.6196, swd: 4.6627, target_std: 13.8286
    Epoch [18/50], Test Losses: mse: 35.4457, mae: 3.5466, huber: 3.1113, swd: 4.4825, target_std: 13.1386
      Epoch 18 composite train-obj: 3.117043
            Val objective improved 3.6723 → 3.6196, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 32.7836, mae: 3.5356, huber: 3.0936, swd: 4.2828, target_std: 13.6390
    Epoch [19/50], Val Losses: mse: 46.6836, mae: 4.1012, huber: 3.6507, swd: 3.8577, target_std: 13.8286
    Epoch [19/50], Test Losses: mse: 37.9242, mae: 3.6854, huber: 3.2381, swd: 3.8859, target_std: 13.1386
      Epoch 19 composite train-obj: 3.093607
            No improvement (3.6507), counter 1/5
    Epoch [20/50], Train Losses: mse: 32.6434, mae: 3.5279, huber: 3.0858, swd: 4.2627, target_std: 13.6430
    Epoch [20/50], Val Losses: mse: 46.9733, mae: 4.1846, huber: 3.7364, swd: 3.4500, target_std: 13.8286
    Epoch [20/50], Test Losses: mse: 37.5513, mae: 3.7330, huber: 3.2882, swd: 3.7665, target_std: 13.1386
      Epoch 20 composite train-obj: 3.085766
            No improvement (3.7364), counter 2/5
    Epoch [21/50], Train Losses: mse: 32.3233, mae: 3.5023, huber: 3.0608, swd: 4.1790, target_std: 13.6418
    Epoch [21/50], Val Losses: mse: 43.3657, mae: 3.9692, huber: 3.5254, swd: 4.3876, target_std: 13.8286
    Epoch [21/50], Test Losses: mse: 35.4383, mae: 3.5471, huber: 3.1071, swd: 4.1348, target_std: 13.1386
      Epoch 21 composite train-obj: 3.060797
            Val objective improved 3.6196 → 3.5254, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 32.2346, mae: 3.4937, huber: 3.0526, swd: 4.1796, target_std: 13.6434
    Epoch [22/50], Val Losses: mse: 43.7693, mae: 3.9614, huber: 3.5168, swd: 4.4496, target_std: 13.8286
    Epoch [22/50], Test Losses: mse: 35.6928, mae: 3.5810, huber: 3.1385, swd: 4.6061, target_std: 13.1386
      Epoch 22 composite train-obj: 3.052649
            Val objective improved 3.5254 → 3.5168, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 31.8428, mae: 3.4612, huber: 3.0210, swd: 4.0889, target_std: 13.6406
    Epoch [23/50], Val Losses: mse: 46.3792, mae: 4.0902, huber: 3.6446, swd: 3.8753, target_std: 13.8286
    Epoch [23/50], Test Losses: mse: 36.8523, mae: 3.6488, huber: 3.2048, swd: 3.9556, target_std: 13.1386
      Epoch 23 composite train-obj: 3.021018
            No improvement (3.6446), counter 1/5
    Epoch [24/50], Train Losses: mse: 31.7130, mae: 3.4532, huber: 3.0130, swd: 4.0540, target_std: 13.6407
    Epoch [24/50], Val Losses: mse: 45.8568, mae: 4.0680, huber: 3.6221, swd: 4.2926, target_std: 13.8286
    Epoch [24/50], Test Losses: mse: 35.6921, mae: 3.5500, huber: 3.1086, swd: 3.9404, target_std: 13.1386
      Epoch 24 composite train-obj: 3.013032
            No improvement (3.6221), counter 2/5
    Epoch [25/50], Train Losses: mse: 31.5491, mae: 3.4420, huber: 3.0020, swd: 4.0126, target_std: 13.6384
    Epoch [25/50], Val Losses: mse: 44.3652, mae: 3.9175, huber: 3.4821, swd: 3.7756, target_std: 13.8286
    Epoch [25/50], Test Losses: mse: 34.0484, mae: 3.3922, huber: 2.9610, swd: 3.5982, target_std: 13.1386
      Epoch 25 composite train-obj: 3.002026
            Val objective improved 3.5168 → 3.4821, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 31.3025, mae: 3.4190, huber: 2.9798, swd: 3.9751, target_std: 13.6432
    Epoch [26/50], Val Losses: mse: 41.6330, mae: 3.8378, huber: 3.3926, swd: 3.9992, target_std: 13.8286
    Epoch [26/50], Test Losses: mse: 33.8221, mae: 3.4166, huber: 2.9748, swd: 3.7264, target_std: 13.1386
      Epoch 26 composite train-obj: 2.979752
            Val objective improved 3.4821 → 3.3926, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 31.2244, mae: 3.4136, huber: 2.9745, swd: 3.9434, target_std: 13.6397
    Epoch [27/50], Val Losses: mse: 44.1407, mae: 3.9372, huber: 3.4976, swd: 4.3329, target_std: 13.8286
    Epoch [27/50], Test Losses: mse: 34.9556, mae: 3.4961, huber: 3.0585, swd: 4.3891, target_std: 13.1386
      Epoch 27 composite train-obj: 2.974511
            No improvement (3.4976), counter 1/5
    Epoch [28/50], Train Losses: mse: 30.9952, mae: 3.3931, huber: 2.9545, swd: 3.9066, target_std: 13.6404
    Epoch [28/50], Val Losses: mse: 46.1774, mae: 4.0042, huber: 3.5565, swd: 4.0679, target_std: 13.8286
    Epoch [28/50], Test Losses: mse: 36.6566, mae: 3.6204, huber: 3.1737, swd: 4.7154, target_std: 13.1386
      Epoch 28 composite train-obj: 2.954455
            No improvement (3.5565), counter 2/5
    Epoch [29/50], Train Losses: mse: 30.7868, mae: 3.3845, huber: 2.9459, swd: 3.8938, target_std: 13.6383
    Epoch [29/50], Val Losses: mse: 46.7074, mae: 4.0875, huber: 3.6405, swd: 4.2902, target_std: 13.8286
    Epoch [29/50], Test Losses: mse: 38.2215, mae: 3.6970, huber: 3.2522, swd: 4.3996, target_std: 13.1386
      Epoch 29 composite train-obj: 2.945916
            No improvement (3.6405), counter 3/5
    Epoch [30/50], Train Losses: mse: 30.6678, mae: 3.3702, huber: 2.9321, swd: 3.8582, target_std: 13.6432
    Epoch [30/50], Val Losses: mse: 48.8823, mae: 4.1310, huber: 3.6779, swd: 3.2823, target_std: 13.8286
    Epoch [30/50], Test Losses: mse: 38.6673, mae: 3.7080, huber: 3.2577, swd: 3.5816, target_std: 13.1386
      Epoch 30 composite train-obj: 2.932145
            No improvement (3.6779), counter 4/5
    Epoch [31/50], Train Losses: mse: 30.5859, mae: 3.3672, huber: 2.9289, swd: 3.8085, target_std: 13.6412
    Epoch [31/50], Val Losses: mse: 43.0070, mae: 3.8328, huber: 3.3892, swd: 3.6511, target_std: 13.8286
    Epoch [31/50], Test Losses: mse: 33.7605, mae: 3.3871, huber: 2.9454, swd: 3.6367, target_std: 13.1386
      Epoch 31 composite train-obj: 2.928894
            Val objective improved 3.3926 → 3.3892, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 30.2172, mae: 3.3378, huber: 2.9002, swd: 3.7427, target_std: 13.6416
    Epoch [32/50], Val Losses: mse: 44.9710, mae: 3.9012, huber: 3.4635, swd: 4.2817, target_std: 13.8286
    Epoch [32/50], Test Losses: mse: 33.9500, mae: 3.3869, huber: 2.9530, swd: 4.2234, target_std: 13.1386
      Epoch 32 composite train-obj: 2.900180
            No improvement (3.4635), counter 1/5
    Epoch [33/50], Train Losses: mse: 30.2163, mae: 3.3358, huber: 2.8985, swd: 3.7540, target_std: 13.6396
    Epoch [33/50], Val Losses: mse: 43.3042, mae: 3.9343, huber: 3.4936, swd: 4.0974, target_std: 13.8286
    Epoch [33/50], Test Losses: mse: 34.4725, mae: 3.4529, huber: 3.0164, swd: 3.8352, target_std: 13.1386
      Epoch 33 composite train-obj: 2.898470
            No improvement (3.4936), counter 2/5
    Epoch [34/50], Train Losses: mse: 30.1345, mae: 3.3267, huber: 2.8898, swd: 3.7173, target_std: 13.6390
    Epoch [34/50], Val Losses: mse: 44.9558, mae: 3.8623, huber: 3.4280, swd: 3.6693, target_std: 13.8286
    Epoch [34/50], Test Losses: mse: 34.4676, mae: 3.3699, huber: 2.9382, swd: 3.6503, target_std: 13.1386
      Epoch 34 composite train-obj: 2.889769
            No improvement (3.4280), counter 3/5
    Epoch [35/50], Train Losses: mse: 30.0579, mae: 3.3234, huber: 2.8862, swd: 3.7235, target_std: 13.6389
    Epoch [35/50], Val Losses: mse: 44.0029, mae: 3.8917, huber: 3.4544, swd: 3.0854, target_std: 13.8286
    Epoch [35/50], Test Losses: mse: 33.9026, mae: 3.3646, huber: 2.9301, swd: 3.0454, target_std: 13.1386
      Epoch 35 composite train-obj: 2.886218
            No improvement (3.4544), counter 4/5
    Epoch [36/50], Train Losses: mse: 30.0342, mae: 3.3219, huber: 2.8848, swd: 3.7041, target_std: 13.6382
    Epoch [36/50], Val Losses: mse: 43.6669, mae: 3.8695, huber: 3.4276, swd: 4.5938, target_std: 13.8286
    Epoch [36/50], Test Losses: mse: 34.1687, mae: 3.4135, huber: 2.9745, swd: 4.9637, target_std: 13.1386
      Epoch 36 composite train-obj: 2.884834
    Epoch [36/50], Test Losses: mse: 33.7605, mae: 3.3871, huber: 2.9454, swd: 3.6367, target_std: 13.1386
    Best round's Test MSE: 33.7605, MAE: 3.3871, SWD: 3.6367
    Best round's Validation MSE: 43.0070, MAE: 3.8328, SWD: 3.6511
    Best round's Test verification MSE : 33.7605, MAE: 3.3871, SWD: 3.6367
    Time taken: 200.14 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 64.0714, mae: 5.9446, huber: 5.4668, swd: 11.6849, target_std: 13.6437
    Epoch [1/50], Val Losses: mse: 53.2250, mae: 5.2724, huber: 4.7994, swd: 13.7254, target_std: 13.8286
    Epoch [1/50], Test Losses: mse: 53.4889, mae: 5.2518, huber: 4.7789, swd: 13.2543, target_std: 13.1386
      Epoch 1 composite train-obj: 5.466807
            Val objective improved inf → 4.7994, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 51.0383, mae: 5.0915, huber: 4.6207, swd: 10.3832, target_std: 13.6396
    Epoch [2/50], Val Losses: mse: 49.8401, mae: 4.9527, huber: 4.4846, swd: 11.2851, target_std: 13.8286
    Epoch [2/50], Test Losses: mse: 48.3538, mae: 4.8570, huber: 4.3890, swd: 11.3403, target_std: 13.1386
      Epoch 2 composite train-obj: 4.620650
            Val objective improved 4.7994 → 4.4846, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 46.4672, mae: 4.6976, huber: 4.2323, swd: 9.2857, target_std: 13.6416
    Epoch [3/50], Val Losses: mse: 51.8166, mae: 4.8677, huber: 4.4033, swd: 9.9477, target_std: 13.8286
    Epoch [3/50], Test Losses: mse: 48.9770, mae: 4.7054, huber: 4.2419, swd: 10.1987, target_std: 13.1386
      Epoch 3 composite train-obj: 4.232286
            Val objective improved 4.4846 → 4.4033, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 43.6405, mae: 4.4456, huber: 3.9846, swd: 8.3536, target_std: 13.6406
    Epoch [4/50], Val Losses: mse: 50.0688, mae: 4.7204, huber: 4.2605, swd: 8.7114, target_std: 13.8286
    Epoch [4/50], Test Losses: mse: 47.1285, mae: 4.5177, huber: 4.0591, swd: 8.8608, target_std: 13.1386
      Epoch 4 composite train-obj: 3.984626
            Val objective improved 4.4033 → 4.2605, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 41.5978, mae: 4.2694, huber: 3.8118, swd: 7.5831, target_std: 13.6412
    Epoch [5/50], Val Losses: mse: 47.5776, mae: 4.4755, huber: 4.0170, swd: 7.4506, target_std: 13.8286
    Epoch [5/50], Test Losses: mse: 41.6561, mae: 4.1665, huber: 3.7094, swd: 7.2672, target_std: 13.1386
      Epoch 5 composite train-obj: 3.811802
            Val objective improved 4.2605 → 4.0170, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 39.9897, mae: 4.1382, huber: 3.6829, swd: 6.9562, target_std: 13.6393
    Epoch [6/50], Val Losses: mse: 44.8357, mae: 4.3110, huber: 3.8562, swd: 7.8050, target_std: 13.8286
    Epoch [6/50], Test Losses: mse: 39.7748, mae: 4.0171, huber: 3.5639, swd: 7.6199, target_std: 13.1386
      Epoch 6 composite train-obj: 3.682873
            Val objective improved 4.0170 → 3.8562, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 38.9049, mae: 4.0429, huber: 3.5898, swd: 6.5587, target_std: 13.6414
    Epoch [7/50], Val Losses: mse: 44.8130, mae: 4.2862, huber: 3.8311, swd: 7.5557, target_std: 13.8286
    Epoch [7/50], Test Losses: mse: 39.6079, mae: 4.0154, huber: 3.5612, swd: 7.6027, target_std: 13.1386
      Epoch 7 composite train-obj: 3.589780
            Val objective improved 3.8562 → 3.8311, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 37.8547, mae: 3.9541, huber: 3.5029, swd: 6.2017, target_std: 13.6394
    Epoch [8/50], Val Losses: mse: 46.4100, mae: 4.4247, huber: 3.9662, swd: 8.8675, target_std: 13.8286
    Epoch [8/50], Test Losses: mse: 39.7505, mae: 4.1074, huber: 3.6497, swd: 9.0610, target_std: 13.1386
      Epoch 8 composite train-obj: 3.502851
            No improvement (3.9662), counter 1/5
    Epoch [9/50], Train Losses: mse: 37.0782, mae: 3.8931, huber: 3.4431, swd: 5.9820, target_std: 13.6405
    Epoch [9/50], Val Losses: mse: 46.8082, mae: 4.2824, huber: 3.8301, swd: 6.1892, target_std: 13.8286
    Epoch [9/50], Test Losses: mse: 39.9436, mae: 3.9324, huber: 3.4826, swd: 5.8779, target_std: 13.1386
      Epoch 9 composite train-obj: 3.443104
            Val objective improved 3.8311 → 3.8301, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 36.4089, mae: 3.8382, huber: 3.3893, swd: 5.7757, target_std: 13.6382
    Epoch [10/50], Val Losses: mse: 47.6520, mae: 4.2866, huber: 3.8346, swd: 5.9524, target_std: 13.8286
    Epoch [10/50], Test Losses: mse: 39.1612, mae: 3.8867, huber: 3.4370, swd: 5.8796, target_std: 13.1386
      Epoch 10 composite train-obj: 3.389259
            No improvement (3.8346), counter 1/5
    Epoch [11/50], Train Losses: mse: 35.6812, mae: 3.7793, huber: 3.3316, swd: 5.5693, target_std: 13.6414
    Epoch [11/50], Val Losses: mse: 48.5027, mae: 4.3461, huber: 3.8910, swd: 5.7677, target_std: 13.8286
    Epoch [11/50], Test Losses: mse: 40.5397, mae: 3.9976, huber: 3.5427, swd: 5.7025, target_std: 13.1386
      Epoch 11 composite train-obj: 3.331649
            No improvement (3.8910), counter 2/5
    Epoch [12/50], Train Losses: mse: 35.2567, mae: 3.7435, huber: 3.2967, swd: 5.4147, target_std: 13.6389
    Epoch [12/50], Val Losses: mse: 45.2388, mae: 4.1524, huber: 3.7044, swd: 6.1394, target_std: 13.8286
    Epoch [12/50], Test Losses: mse: 37.2192, mae: 3.7572, huber: 3.3116, swd: 5.9913, target_std: 13.1386
      Epoch 12 composite train-obj: 3.296688
            Val objective improved 3.8301 → 3.7044, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 34.8238, mae: 3.7118, huber: 3.2655, swd: 5.2721, target_std: 13.6388
    Epoch [13/50], Val Losses: mse: 44.6288, mae: 4.1041, huber: 3.6548, swd: 5.1530, target_std: 13.8286
    Epoch [13/50], Test Losses: mse: 38.5295, mae: 3.7662, huber: 3.3196, swd: 4.7267, target_std: 13.1386
      Epoch 13 composite train-obj: 3.265512
            Val objective improved 3.7044 → 3.6548, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 34.3617, mae: 3.6749, huber: 3.2295, swd: 5.1655, target_std: 13.6440
    Epoch [14/50], Val Losses: mse: 43.6626, mae: 4.1226, huber: 3.6688, swd: 5.7815, target_std: 13.8286
    Epoch [14/50], Test Losses: mse: 37.2866, mae: 3.7820, huber: 3.3310, swd: 5.3314, target_std: 13.1386
      Epoch 14 composite train-obj: 3.229487
            No improvement (3.6688), counter 1/5
    Epoch [15/50], Train Losses: mse: 33.9434, mae: 3.6411, huber: 3.1963, swd: 5.0316, target_std: 13.6393
    Epoch [15/50], Val Losses: mse: 44.3024, mae: 4.0238, huber: 3.5757, swd: 5.1515, target_std: 13.8286
    Epoch [15/50], Test Losses: mse: 37.2573, mae: 3.6625, huber: 3.2170, swd: 4.9638, target_std: 13.1386
      Epoch 15 composite train-obj: 3.196316
            Val objective improved 3.6548 → 3.5757, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 33.5405, mae: 3.6088, huber: 3.1648, swd: 4.9134, target_std: 13.6433
    Epoch [16/50], Val Losses: mse: 45.3178, mae: 4.0957, huber: 3.6503, swd: 5.0381, target_std: 13.8286
    Epoch [16/50], Test Losses: mse: 37.5084, mae: 3.7320, huber: 3.2887, swd: 5.1373, target_std: 13.1386
      Epoch 16 composite train-obj: 3.164793
            No improvement (3.6503), counter 1/5
    Epoch [17/50], Train Losses: mse: 33.2111, mae: 3.5836, huber: 3.1401, swd: 4.8188, target_std: 13.6413
    Epoch [17/50], Val Losses: mse: 51.7469, mae: 4.4075, huber: 3.9588, swd: 4.7354, target_std: 13.8286
    Epoch [17/50], Test Losses: mse: 43.2052, mae: 4.0515, huber: 3.6048, swd: 4.7633, target_std: 13.1386
      Epoch 17 composite train-obj: 3.140074
            No improvement (3.9588), counter 2/5
    Epoch [18/50], Train Losses: mse: 32.9819, mae: 3.5642, huber: 3.1211, swd: 4.7243, target_std: 13.6427
    Epoch [18/50], Val Losses: mse: 45.0725, mae: 3.9907, huber: 3.5500, swd: 5.1339, target_std: 13.8286
    Epoch [18/50], Test Losses: mse: 37.3996, mae: 3.6565, huber: 3.2162, swd: 5.3257, target_std: 13.1386
      Epoch 18 composite train-obj: 3.121062
            Val objective improved 3.5757 → 3.5500, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 32.5958, mae: 3.5348, huber: 3.0924, swd: 4.6285, target_std: 13.6387
    Epoch [19/50], Val Losses: mse: 46.9045, mae: 4.2227, huber: 3.7801, swd: 5.6073, target_std: 13.8286
    Epoch [19/50], Test Losses: mse: 37.5344, mae: 3.7585, huber: 3.3192, swd: 5.5342, target_std: 13.1386
      Epoch 19 composite train-obj: 3.092352
            No improvement (3.7801), counter 1/5
    Epoch [20/50], Train Losses: mse: 32.4325, mae: 3.5207, huber: 3.0786, swd: 4.5907, target_std: 13.6424
    Epoch [20/50], Val Losses: mse: 44.5603, mae: 4.0839, huber: 3.6340, swd: 4.4531, target_std: 13.8286
    Epoch [20/50], Test Losses: mse: 38.2755, mae: 3.7682, huber: 3.3203, swd: 3.9421, target_std: 13.1386
      Epoch 20 composite train-obj: 3.078563
            No improvement (3.6340), counter 2/5
    Epoch [21/50], Train Losses: mse: 32.0782, mae: 3.4916, huber: 3.0502, swd: 4.4828, target_std: 13.6401
    Epoch [21/50], Val Losses: mse: 43.8297, mae: 4.0107, huber: 3.5668, swd: 5.1137, target_std: 13.8286
    Epoch [21/50], Test Losses: mse: 38.7673, mae: 3.7271, huber: 3.2854, swd: 5.0023, target_std: 13.1386
      Epoch 21 composite train-obj: 3.050242
            No improvement (3.5668), counter 3/5
    Epoch [22/50], Train Losses: mse: 31.8919, mae: 3.4819, huber: 3.0405, swd: 4.4358, target_std: 13.6401
    Epoch [22/50], Val Losses: mse: 43.7322, mae: 3.9482, huber: 3.5057, swd: 4.9508, target_std: 13.8286
    Epoch [22/50], Test Losses: mse: 35.8830, mae: 3.5248, huber: 3.0852, swd: 4.7295, target_std: 13.1386
      Epoch 22 composite train-obj: 3.040527
            Val objective improved 3.5500 → 3.5057, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 31.6033, mae: 3.4587, huber: 3.0178, swd: 4.3561, target_std: 13.6400
    Epoch [23/50], Val Losses: mse: 44.2439, mae: 3.9774, huber: 3.5297, swd: 4.9737, target_std: 13.8286
    Epoch [23/50], Test Losses: mse: 34.9990, mae: 3.5280, huber: 3.0833, swd: 4.6124, target_std: 13.1386
      Epoch 23 composite train-obj: 3.017793
            No improvement (3.5297), counter 1/5
    Epoch [24/50], Train Losses: mse: 31.4068, mae: 3.4449, huber: 3.0042, swd: 4.3104, target_std: 13.6395
    Epoch [24/50], Val Losses: mse: 44.4584, mae: 3.8859, huber: 3.4489, swd: 3.9601, target_std: 13.8286
    Epoch [24/50], Test Losses: mse: 34.6353, mae: 3.4312, huber: 2.9956, swd: 3.6449, target_std: 13.1386
      Epoch 24 composite train-obj: 3.004184
            Val objective improved 3.5057 → 3.4489, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 31.2898, mae: 3.4348, huber: 2.9945, swd: 4.2758, target_std: 13.6401
    Epoch [25/50], Val Losses: mse: 43.4430, mae: 3.9688, huber: 3.5236, swd: 4.6020, target_std: 13.8286
    Epoch [25/50], Test Losses: mse: 36.3153, mae: 3.5678, huber: 3.1264, swd: 4.3743, target_std: 13.1386
      Epoch 25 composite train-obj: 2.994498
            No improvement (3.5236), counter 1/5
    Epoch [26/50], Train Losses: mse: 31.0328, mae: 3.4138, huber: 2.9741, swd: 4.2060, target_std: 13.6399
    Epoch [26/50], Val Losses: mse: 43.9906, mae: 4.0760, huber: 3.6237, swd: 4.7088, target_std: 13.8286
    Epoch [26/50], Test Losses: mse: 35.6004, mae: 3.6474, huber: 3.1963, swd: 4.5430, target_std: 13.1386
      Epoch 26 composite train-obj: 2.974062
            No improvement (3.6237), counter 2/5
    Epoch [27/50], Train Losses: mse: 30.7329, mae: 3.3907, huber: 2.9514, swd: 4.1333, target_std: 13.6389
    Epoch [27/50], Val Losses: mse: 45.8878, mae: 4.0355, huber: 3.5927, swd: 4.0776, target_std: 13.8286
    Epoch [27/50], Test Losses: mse: 36.2025, mae: 3.6229, huber: 3.1822, swd: 3.8581, target_std: 13.1386
      Epoch 27 composite train-obj: 2.951448
            No improvement (3.5927), counter 3/5
    Epoch [28/50], Train Losses: mse: 30.8284, mae: 3.3995, huber: 2.9601, swd: 4.1517, target_std: 13.6413
    Epoch [28/50], Val Losses: mse: 45.1945, mae: 3.9952, huber: 3.5508, swd: 4.3252, target_std: 13.8286
    Epoch [28/50], Test Losses: mse: 35.6527, mae: 3.5424, huber: 3.1000, swd: 4.0485, target_std: 13.1386
      Epoch 28 composite train-obj: 2.960064
            No improvement (3.5508), counter 4/5
    Epoch [29/50], Train Losses: mse: 30.4583, mae: 3.3704, huber: 2.9314, swd: 4.0750, target_std: 13.6387
    Epoch [29/50], Val Losses: mse: 41.6434, mae: 3.8582, huber: 3.4152, swd: 4.1456, target_std: 13.8286
    Epoch [29/50], Test Losses: mse: 34.8333, mae: 3.5284, huber: 3.0866, swd: 3.9134, target_std: 13.1386
      Epoch 29 composite train-obj: 2.931433
            Val objective improved 3.4489 → 3.4152, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 30.2932, mae: 3.3603, huber: 2.9213, swd: 4.0293, target_std: 13.6381
    Epoch [30/50], Val Losses: mse: 42.9498, mae: 3.8990, huber: 3.4506, swd: 4.5689, target_std: 13.8286
    Epoch [30/50], Test Losses: mse: 33.6055, mae: 3.4445, huber: 2.9994, swd: 4.1720, target_std: 13.1386
      Epoch 30 composite train-obj: 2.921342
            No improvement (3.4506), counter 1/5
    Epoch [31/50], Train Losses: mse: 30.0878, mae: 3.3443, huber: 2.9058, swd: 3.9778, target_std: 13.6368
    Epoch [31/50], Val Losses: mse: 41.4388, mae: 3.8032, huber: 3.3662, swd: 4.1925, target_std: 13.8286
    Epoch [31/50], Test Losses: mse: 34.6318, mae: 3.4291, huber: 2.9959, swd: 3.8122, target_std: 13.1386
      Epoch 31 composite train-obj: 2.905758
            Val objective improved 3.4152 → 3.3662, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 30.0313, mae: 3.3399, huber: 2.9016, swd: 3.9720, target_std: 13.6390
    Epoch [32/50], Val Losses: mse: 41.2109, mae: 3.7722, huber: 3.3340, swd: 4.3362, target_std: 13.8286
    Epoch [32/50], Test Losses: mse: 32.4288, mae: 3.2801, huber: 2.8471, swd: 3.6910, target_std: 13.1386
      Epoch 32 composite train-obj: 2.901573
            Val objective improved 3.3662 → 3.3340, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 29.9190, mae: 3.3290, huber: 2.8911, swd: 3.9346, target_std: 13.6403
    Epoch [33/50], Val Losses: mse: 43.0271, mae: 3.8865, huber: 3.4494, swd: 4.8383, target_std: 13.8286
    Epoch [33/50], Test Losses: mse: 33.4494, mae: 3.3915, huber: 2.9582, swd: 4.4336, target_std: 13.1386
      Epoch 33 composite train-obj: 2.891097
            No improvement (3.4494), counter 1/5
    Epoch [34/50], Train Losses: mse: 29.6981, mae: 3.3115, huber: 2.8740, swd: 3.8889, target_std: 13.6417
    Epoch [34/50], Val Losses: mse: 42.8450, mae: 3.8104, huber: 3.3750, swd: 3.6624, target_std: 13.8286
    Epoch [34/50], Test Losses: mse: 34.4643, mae: 3.3949, huber: 2.9616, swd: 3.4105, target_std: 13.1386
      Epoch 34 composite train-obj: 2.874028
            No improvement (3.3750), counter 2/5
    Epoch [35/50], Train Losses: mse: 29.4567, mae: 3.2940, huber: 2.8567, swd: 3.8207, target_std: 13.6413
    Epoch [35/50], Val Losses: mse: 43.4721, mae: 3.9220, huber: 3.4859, swd: 4.3253, target_std: 13.8286
    Epoch [35/50], Test Losses: mse: 33.6078, mae: 3.4066, huber: 2.9742, swd: 4.1174, target_std: 13.1386
      Epoch 35 composite train-obj: 2.856704
            No improvement (3.4859), counter 3/5
    Epoch [36/50], Train Losses: mse: 29.5295, mae: 3.3010, huber: 2.8635, swd: 3.8227, target_std: 13.6430
    Epoch [36/50], Val Losses: mse: 39.7463, mae: 3.6014, huber: 3.1782, swd: 3.5412, target_std: 13.8286
    Epoch [36/50], Test Losses: mse: 31.3170, mae: 3.1595, huber: 2.7397, swd: 3.1085, target_std: 13.1386
      Epoch 36 composite train-obj: 2.863512
            Val objective improved 3.3340 → 3.1782, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 29.3270, mae: 3.2809, huber: 2.8441, swd: 3.7855, target_std: 13.6396
    Epoch [37/50], Val Losses: mse: 40.6015, mae: 3.7537, huber: 3.3118, swd: 3.9422, target_std: 13.8286
    Epoch [37/50], Test Losses: mse: 32.2133, mae: 3.3304, huber: 2.8915, swd: 3.5628, target_std: 13.1386
      Epoch 37 composite train-obj: 2.844125
            No improvement (3.3118), counter 1/5
    Epoch [38/50], Train Losses: mse: 29.0646, mae: 3.2628, huber: 2.8263, swd: 3.7334, target_std: 13.6391
    Epoch [38/50], Val Losses: mse: 41.3910, mae: 3.7737, huber: 3.3368, swd: 4.0109, target_std: 13.8286
    Epoch [38/50], Test Losses: mse: 32.7782, mae: 3.3456, huber: 2.9106, swd: 3.5070, target_std: 13.1386
      Epoch 38 composite train-obj: 2.826309
            No improvement (3.3368), counter 2/5
    Epoch [39/50], Train Losses: mse: 29.1263, mae: 3.2649, huber: 2.8285, swd: 3.7228, target_std: 13.6403
    Epoch [39/50], Val Losses: mse: 44.3379, mae: 3.8934, huber: 3.4544, swd: 3.9829, target_std: 13.8286
    Epoch [39/50], Test Losses: mse: 34.8733, mae: 3.4735, huber: 3.0357, swd: 3.8119, target_std: 13.1386
      Epoch 39 composite train-obj: 2.828483
            No improvement (3.4544), counter 3/5
    Epoch [40/50], Train Losses: mse: 29.0799, mae: 3.2617, huber: 2.8253, swd: 3.7151, target_std: 13.6406
    Epoch [40/50], Val Losses: mse: 40.1483, mae: 3.6703, huber: 3.2369, swd: 4.5322, target_std: 13.8286
    Epoch [40/50], Test Losses: mse: 31.5423, mae: 3.2235, huber: 2.7942, swd: 3.8887, target_std: 13.1386
      Epoch 40 composite train-obj: 2.825319
            No improvement (3.2369), counter 4/5
    Epoch [41/50], Train Losses: mse: 28.7380, mae: 3.2394, huber: 2.8032, swd: 3.6595, target_std: 13.6425
    Epoch [41/50], Val Losses: mse: 40.8531, mae: 3.7589, huber: 3.3212, swd: 4.1334, target_std: 13.8286
    Epoch [41/50], Test Losses: mse: 33.6058, mae: 3.3591, huber: 2.9243, swd: 3.9484, target_std: 13.1386
      Epoch 41 composite train-obj: 2.803211
    Epoch [41/50], Test Losses: mse: 31.3170, mae: 3.1595, huber: 2.7397, swd: 3.1085, target_std: 13.1386
    Best round's Test MSE: 31.3170, MAE: 3.1595, SWD: 3.1085
    Best round's Validation MSE: 39.7463, MAE: 3.6014, SWD: 3.5412
    Best round's Test verification MSE : 31.3170, MAE: 3.1595, SWD: 3.1085
    Time taken: 227.79 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq336_pred336_20250507_1335)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 32.8696 ± 1.1018
      mae: 3.3221 ± 0.1157
      huber: 2.8885 ± 0.1061
      swd: 3.5280 ± 0.3079
      target_std: 13.1386 ± 0.0000
      count: 36.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 41.4474 ± 1.3349
      mae: 3.7573 ± 0.1102
      huber: 3.3209 ± 0.1009
      swd: 3.8392 ± 0.3467
      target_std: 13.8286 ± 0.0000
      count: 36.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 616.69 seconds
    
    Experiment complete: PatchTST_lorenz_seq336_pred336_20250507_1335
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 336-720
##### huber


```python
utils.reload_modules([utils])
cfg_patch_tst = train_config.FlatPatchTSTConfig(
    seq_len=336,
    pred_len=720,
    channels=data_mgr.datasets['lorenz']['channels'],
    enc_in=data_mgr.datasets['lorenz']['channels'],
    dec_in=data_mgr.datasets['lorenz']['channels'],
    c_out=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0]
)
exp_patch_tst = execute_model_evaluation('lorenz', cfg_patch_tst, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 277
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 277
    Validation Batches: 33
    Test Batches: 74
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 73.6958, mae: 6.5666, huber: 6.0854, swd: 13.5246, ept: 36.6875
    Epoch [1/50], Val Losses: mse: 66.1515, mae: 6.1499, huber: 5.6714, swd: 17.0219, ept: 85.4137
    Epoch [1/50], Test Losses: mse: 66.7936, mae: 6.1385, huber: 5.6605, swd: 17.0811, ept: 83.9394
      Epoch 1 composite train-obj: 6.085431
            Val objective improved inf → 5.6714, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.3360, mae: 5.9070, huber: 5.4298, swd: 14.1719, ept: 44.0175
    Epoch [2/50], Val Losses: mse: 61.6440, mae: 5.8518, huber: 5.3755, swd: 16.5593, ept: 115.6501
    Epoch [2/50], Test Losses: mse: 60.8746, mae: 5.7758, huber: 5.3002, swd: 15.9721, ept: 116.6718
      Epoch 2 composite train-obj: 5.429848
            Val objective improved 5.6714 → 5.3755, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 58.0495, mae: 5.5869, huber: 5.1129, swd: 13.3240, ept: 46.5325
    Epoch [3/50], Val Losses: mse: 60.5188, mae: 5.6838, huber: 5.2102, swd: 15.5328, ept: 134.1506
    Epoch [3/50], Test Losses: mse: 59.2812, mae: 5.5914, huber: 5.1185, swd: 16.0289, ept: 129.9636
      Epoch 3 composite train-obj: 5.112933
            Val objective improved 5.3755 → 5.2102, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 55.5236, mae: 5.3821, huber: 4.9107, swd: 12.4849, ept: 47.8808
    Epoch [4/50], Val Losses: mse: 64.4938, mae: 5.8957, huber: 5.4206, swd: 13.5742, ept: 131.2783
    Epoch [4/50], Test Losses: mse: 63.5694, mae: 5.8743, huber: 5.3991, swd: 14.7521, ept: 126.6944
      Epoch 4 composite train-obj: 4.910716
            No improvement (5.4206), counter 1/5
    Epoch [5/50], Train Losses: mse: 53.9282, mae: 5.2484, huber: 4.7788, swd: 11.7517, ept: 48.7855
    Epoch [5/50], Val Losses: mse: 64.4750, mae: 5.8134, huber: 5.3377, swd: 11.4282, ept: 150.1134
    Epoch [5/50], Test Losses: mse: 63.0442, mae: 5.7357, huber: 5.2608, swd: 12.2216, ept: 147.5896
      Epoch 5 composite train-obj: 4.778781
            No improvement (5.3377), counter 2/5
    Epoch [6/50], Train Losses: mse: 52.7685, mae: 5.1544, huber: 4.6860, swd: 11.0936, ept: 49.6496
    Epoch [6/50], Val Losses: mse: 61.1620, mae: 5.5445, huber: 5.0753, swd: 11.8174, ept: 164.8449
    Epoch [6/50], Test Losses: mse: 59.2665, mae: 5.4388, huber: 4.9706, swd: 12.5318, ept: 158.9669
      Epoch 6 composite train-obj: 4.686013
            Val objective improved 5.2102 → 5.0753, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 51.7934, mae: 5.0753, huber: 4.6080, swd: 10.5261, ept: 49.9195
    Epoch [7/50], Val Losses: mse: 63.5372, mae: 5.6402, huber: 5.1692, swd: 9.4162, ept: 173.9617
    Epoch [7/50], Test Losses: mse: 60.8437, mae: 5.5180, huber: 5.0476, swd: 10.4562, ept: 172.4065
      Epoch 7 composite train-obj: 4.608038
            No improvement (5.1692), counter 1/5
    Epoch [8/50], Train Losses: mse: 51.0257, mae: 5.0151, huber: 4.5485, swd: 10.0223, ept: 50.4620
    Epoch [8/50], Val Losses: mse: 61.7921, mae: 5.5907, huber: 5.1181, swd: 10.5704, ept: 169.5765
    Epoch [8/50], Test Losses: mse: 60.8937, mae: 5.5545, huber: 5.0819, swd: 10.9316, ept: 166.2691
      Epoch 8 composite train-obj: 4.548496
            No improvement (5.1181), counter 2/5
    Epoch [9/50], Train Losses: mse: 50.2649, mae: 4.9586, huber: 4.4928, swd: 9.6420, ept: 50.8062
    Epoch [9/50], Val Losses: mse: 63.8488, mae: 5.6086, huber: 5.1391, swd: 8.9729, ept: 177.2926
    Epoch [9/50], Test Losses: mse: 62.1756, mae: 5.5243, huber: 5.0554, swd: 9.2640, ept: 174.8474
      Epoch 9 composite train-obj: 4.492840
            No improvement (5.1391), counter 3/5
    Epoch [10/50], Train Losses: mse: 49.7815, mae: 4.9213, huber: 4.4559, swd: 9.3145, ept: 51.0697
    Epoch [10/50], Val Losses: mse: 63.9510, mae: 5.6461, huber: 5.1745, swd: 8.8460, ept: 178.8355
    Epoch [10/50], Test Losses: mse: 61.2276, mae: 5.5063, huber: 5.0357, swd: 9.3125, ept: 178.3358
      Epoch 10 composite train-obj: 4.455916
            No improvement (5.1745), counter 4/5
    Epoch [11/50], Train Losses: mse: 49.1035, mae: 4.8728, huber: 4.4080, swd: 9.0214, ept: 51.2511
    Epoch [11/50], Val Losses: mse: 65.6066, mae: 5.6661, huber: 5.1956, swd: 7.4093, ept: 184.2946
    Epoch [11/50], Test Losses: mse: 61.6778, mae: 5.4562, huber: 4.9874, swd: 7.7768, ept: 184.0581
      Epoch 11 composite train-obj: 4.407994
    Epoch [11/50], Test Losses: mse: 59.2665, mae: 5.4388, huber: 4.9706, swd: 12.5318, ept: 158.9669
    Best round's Test MSE: 59.2665, MAE: 5.4388, SWD: 12.5318
    Best round's Validation MSE: 61.1620, MAE: 5.5445, SWD: 11.8174
    Best round's Test verification MSE : 59.2665, MAE: 5.4388, SWD: 12.5318
    Time taken: 72.52 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 73.6477, mae: 6.5527, huber: 6.0716, swd: 13.0912, ept: 37.4071
    Epoch [1/50], Val Losses: mse: 64.6902, mae: 6.0907, huber: 5.6125, swd: 17.1423, ept: 89.1632
    Epoch [1/50], Test Losses: mse: 65.9525, mae: 6.1105, huber: 5.6327, swd: 16.5568, ept: 88.0444
      Epoch 1 composite train-obj: 6.071644
            Val objective improved inf → 5.6125, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.1435, mae: 5.8919, huber: 5.4149, swd: 13.6530, ept: 45.0577
    Epoch [2/50], Val Losses: mse: 61.3967, mae: 5.7847, huber: 5.3094, swd: 15.2976, ept: 125.3065
    Epoch [2/50], Test Losses: mse: 61.3487, mae: 5.7260, huber: 5.2517, swd: 15.2066, ept: 120.3726
      Epoch 2 composite train-obj: 5.414909
            Val objective improved 5.6125 → 5.3094, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 57.9532, mae: 5.5782, huber: 5.1044, swd: 13.1051, ept: 46.9662
    Epoch [3/50], Val Losses: mse: 60.5324, mae: 5.6464, huber: 5.1736, swd: 14.2401, ept: 143.1013
    Epoch [3/50], Test Losses: mse: 59.8587, mae: 5.5737, huber: 5.1018, swd: 14.4107, ept: 136.9742
      Epoch 3 composite train-obj: 5.104361
            Val objective improved 5.3094 → 5.1736, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 55.6296, mae: 5.3864, huber: 4.9149, swd: 12.3743, ept: 48.1638
    Epoch [4/50], Val Losses: mse: 60.7340, mae: 5.6691, huber: 5.1956, swd: 13.1372, ept: 146.1759
    Epoch [4/50], Test Losses: mse: 59.8801, mae: 5.5693, huber: 5.0971, swd: 12.8111, ept: 144.3231
      Epoch 4 composite train-obj: 4.914918
            No improvement (5.1956), counter 1/5
    Epoch [5/50], Train Losses: mse: 54.0419, mae: 5.2568, huber: 4.7872, swd: 11.8368, ept: 48.9515
    Epoch [5/50], Val Losses: mse: 62.1025, mae: 5.7179, huber: 5.2426, swd: 12.7926, ept: 151.5026
    Epoch [5/50], Test Losses: mse: 60.7720, mae: 5.6218, huber: 5.1481, swd: 13.7518, ept: 144.2980
      Epoch 5 composite train-obj: 4.787193
            No improvement (5.2426), counter 2/5
    Epoch [6/50], Train Losses: mse: 52.9291, mae: 5.1650, huber: 4.6966, swd: 11.2358, ept: 49.6786
    Epoch [6/50], Val Losses: mse: 61.3798, mae: 5.5549, huber: 5.0866, swd: 11.1758, ept: 167.7033
    Epoch [6/50], Test Losses: mse: 60.0928, mae: 5.4776, huber: 5.0101, swd: 11.9303, ept: 164.8472
      Epoch 6 composite train-obj: 4.696626
            Val objective improved 5.1736 → 5.0866, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 52.0819, mae: 5.0935, huber: 4.6261, swd: 10.6829, ept: 49.9408
    Epoch [7/50], Val Losses: mse: 61.8946, mae: 5.5867, huber: 5.1171, swd: 11.5298, ept: 162.7782
    Epoch [7/50], Test Losses: mse: 59.5715, mae: 5.4611, huber: 4.9922, swd: 11.8846, ept: 159.8745
      Epoch 7 composite train-obj: 4.626119
            No improvement (5.1171), counter 1/5
    Epoch [8/50], Train Losses: mse: 51.1928, mae: 5.0232, huber: 4.5567, swd: 10.2024, ept: 50.4472
    Epoch [8/50], Val Losses: mse: 62.9180, mae: 5.5705, huber: 5.1012, swd: 9.8157, ept: 171.7372
    Epoch [8/50], Test Losses: mse: 61.0110, mae: 5.4739, huber: 5.0055, swd: 10.5390, ept: 169.2380
      Epoch 8 composite train-obj: 4.556725
            No improvement (5.1012), counter 2/5
    Epoch [9/50], Train Losses: mse: 50.4732, mae: 4.9707, huber: 4.5049, swd: 9.7216, ept: 50.7903
    Epoch [9/50], Val Losses: mse: 62.6825, mae: 5.5770, huber: 5.1071, swd: 10.1499, ept: 171.8713
    Epoch [9/50], Test Losses: mse: 59.9511, mae: 5.4009, huber: 4.9326, swd: 9.7109, ept: 171.4039
      Epoch 9 composite train-obj: 4.504873
            No improvement (5.1071), counter 3/5
    Epoch [10/50], Train Losses: mse: 49.8332, mae: 4.9215, huber: 4.4562, swd: 9.3778, ept: 51.2144
    Epoch [10/50], Val Losses: mse: 62.6632, mae: 5.5401, huber: 5.0713, swd: 8.5330, ept: 180.5435
    Epoch [10/50], Test Losses: mse: 60.5417, mae: 5.4079, huber: 4.9405, swd: 8.8165, ept: 181.3233
      Epoch 10 composite train-obj: 4.456213
            Val objective improved 5.0866 → 5.0713, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 49.2926, mae: 4.8813, huber: 4.4163, swd: 9.0014, ept: 51.4568
    Epoch [11/50], Val Losses: mse: 64.9485, mae: 5.6932, huber: 5.2211, swd: 8.4593, ept: 178.9185
    Epoch [11/50], Test Losses: mse: 62.0853, mae: 5.5380, huber: 5.0673, swd: 8.9088, ept: 178.4292
      Epoch 11 composite train-obj: 4.416348
            No improvement (5.2211), counter 1/5
    Epoch [12/50], Train Losses: mse: 48.7622, mae: 4.8430, huber: 4.3786, swd: 8.7617, ept: 51.5235
    Epoch [12/50], Val Losses: mse: 63.8808, mae: 5.6299, huber: 5.1569, swd: 8.1159, ept: 181.3390
    Epoch [12/50], Test Losses: mse: 61.0643, mae: 5.4645, huber: 4.9929, swd: 8.3037, ept: 183.7802
      Epoch 12 composite train-obj: 4.378635
            No improvement (5.1569), counter 2/5
    Epoch [13/50], Train Losses: mse: 48.2528, mae: 4.8049, huber: 4.3409, swd: 8.4511, ept: 51.8902
    Epoch [13/50], Val Losses: mse: 64.3285, mae: 5.6099, huber: 5.1390, swd: 7.3450, ept: 189.3997
    Epoch [13/50], Test Losses: mse: 61.6115, mae: 5.4822, huber: 5.0119, swd: 7.9501, ept: 189.3965
      Epoch 13 composite train-obj: 4.340861
            No improvement (5.1390), counter 3/5
    Epoch [14/50], Train Losses: mse: 47.7972, mae: 4.7727, huber: 4.3091, swd: 8.2477, ept: 51.6758
    Epoch [14/50], Val Losses: mse: 64.7861, mae: 5.6934, huber: 5.2200, swd: 8.6174, ept: 178.1605
    Epoch [14/50], Test Losses: mse: 61.6530, mae: 5.5074, huber: 5.0357, swd: 8.5804, ept: 179.8101
      Epoch 14 composite train-obj: 4.309084
            No improvement (5.2200), counter 4/5
    Epoch [15/50], Train Losses: mse: 47.4166, mae: 4.7453, huber: 4.2820, swd: 8.0698, ept: 51.8684
    Epoch [15/50], Val Losses: mse: 64.4387, mae: 5.5761, huber: 5.1069, swd: 7.0487, ept: 191.7658
    Epoch [15/50], Test Losses: mse: 61.4428, mae: 5.4179, huber: 4.9500, swd: 7.5376, ept: 194.1886
      Epoch 15 composite train-obj: 4.281990
    Epoch [15/50], Test Losses: mse: 60.5417, mae: 5.4079, huber: 4.9405, swd: 8.8165, ept: 181.3233
    Best round's Test MSE: 60.5417, MAE: 5.4079, SWD: 8.8165
    Best round's Validation MSE: 62.6632, MAE: 5.5401, SWD: 8.5330
    Best round's Test verification MSE : 60.5417, MAE: 5.4079, SWD: 8.8165
    Time taken: 98.67 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.6085, mae: 6.5156, huber: 6.0346, swd: 14.4099, ept: 37.0366
    Epoch [1/50], Val Losses: mse: 63.5224, mae: 6.0534, huber: 5.5746, swd: 18.6561, ept: 89.5160
    Epoch [1/50], Test Losses: mse: 63.9443, mae: 6.0420, huber: 5.5636, swd: 18.2869, ept: 85.7649
      Epoch 1 composite train-obj: 6.034569
            Val objective improved inf → 5.5746, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 61.0209, mae: 5.8196, huber: 5.3431, swd: 14.9537, ept: 44.6877
    Epoch [2/50], Val Losses: mse: 61.1197, mae: 5.7736, huber: 5.2986, swd: 17.2309, ept: 125.2607
    Epoch [2/50], Test Losses: mse: 60.0811, mae: 5.6833, huber: 5.2091, swd: 17.4055, ept: 121.9489
      Epoch 2 composite train-obj: 5.343145
            Val objective improved 5.5746 → 5.2986, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 57.6720, mae: 5.5550, huber: 5.0813, swd: 13.9771, ept: 46.6593
    Epoch [3/50], Val Losses: mse: 62.0490, mae: 5.7003, huber: 5.2280, swd: 14.9530, ept: 142.2559
    Epoch [3/50], Test Losses: mse: 61.3087, mae: 5.6407, huber: 5.1691, swd: 15.2872, ept: 136.6384
      Epoch 3 composite train-obj: 5.081323
            Val objective improved 5.2986 → 5.2280, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 55.5471, mae: 5.3806, huber: 4.9091, swd: 13.1263, ept: 48.2682
    Epoch [4/50], Val Losses: mse: 64.1263, mae: 5.7812, huber: 5.3076, swd: 12.5357, ept: 147.9009
    Epoch [4/50], Test Losses: mse: 61.2038, mae: 5.5979, huber: 5.1264, swd: 12.8546, ept: 142.2138
      Epoch 4 composite train-obj: 4.909149
            No improvement (5.3076), counter 1/5
    Epoch [5/50], Train Losses: mse: 53.9664, mae: 5.2513, huber: 4.7816, swd: 12.3166, ept: 49.1710
    Epoch [5/50], Val Losses: mse: 63.9483, mae: 5.7258, huber: 5.2533, swd: 10.7989, ept: 155.4108
    Epoch [5/50], Test Losses: mse: 62.4493, mae: 5.6327, huber: 5.1605, swd: 10.1429, ept: 150.4992
      Epoch 5 composite train-obj: 4.781554
            No improvement (5.2533), counter 2/5
    Epoch [6/50], Train Losses: mse: 52.7642, mae: 5.1552, huber: 4.6868, swd: 11.6259, ept: 49.5957
    Epoch [6/50], Val Losses: mse: 64.3094, mae: 5.7090, huber: 5.2382, swd: 11.3507, ept: 158.7396
    Epoch [6/50], Test Losses: mse: 62.1488, mae: 5.6160, huber: 5.1453, swd: 12.2634, ept: 156.9482
      Epoch 6 composite train-obj: 4.686759
            No improvement (5.2382), counter 3/5
    Epoch [7/50], Train Losses: mse: 51.7732, mae: 5.0759, huber: 4.6085, swd: 10.9937, ept: 50.1005
    Epoch [7/50], Val Losses: mse: 65.3675, mae: 5.8929, huber: 5.4140, swd: 9.4990, ept: 159.1323
    Epoch [7/50], Test Losses: mse: 63.3761, mae: 5.8027, huber: 5.3243, swd: 9.3368, ept: 156.1828
      Epoch 7 composite train-obj: 4.608515
            No improvement (5.4140), counter 4/5
    Epoch [8/50], Train Losses: mse: 50.8630, mae: 5.0096, huber: 4.5430, swd: 10.5463, ept: 50.0195
    Epoch [8/50], Val Losses: mse: 61.4755, mae: 5.5666, huber: 5.0950, swd: 11.9025, ept: 170.0746
    Epoch [8/50], Test Losses: mse: 59.1610, mae: 5.4343, huber: 4.9638, swd: 11.8777, ept: 166.7404
      Epoch 8 composite train-obj: 4.543002
            Val objective improved 5.2280 → 5.0950, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 50.1349, mae: 4.9530, huber: 4.4871, swd: 10.0463, ept: 50.4629
    Epoch [9/50], Val Losses: mse: 64.9352, mae: 5.6633, huber: 5.1950, swd: 9.2344, ept: 175.3102
    Epoch [9/50], Test Losses: mse: 61.4446, mae: 5.4947, huber: 5.0273, swd: 9.8356, ept: 180.1668
      Epoch 9 composite train-obj: 4.487113
            No improvement (5.1950), counter 1/5
    Epoch [10/50], Train Losses: mse: 49.5559, mae: 4.9096, huber: 4.4442, swd: 9.6805, ept: 50.8666
    Epoch [10/50], Val Losses: mse: 63.1786, mae: 5.5479, huber: 5.0784, swd: 9.3681, ept: 178.2401
    Epoch [10/50], Test Losses: mse: 60.1526, mae: 5.4014, huber: 4.9330, swd: 9.6478, ept: 176.1115
      Epoch 10 composite train-obj: 4.444170
            Val objective improved 5.0950 → 5.0784, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 48.9913, mae: 4.8677, huber: 4.4027, swd: 9.3318, ept: 50.8760
    Epoch [11/50], Val Losses: mse: 64.7294, mae: 5.7576, huber: 5.2829, swd: 8.9272, ept: 174.4516
    Epoch [11/50], Test Losses: mse: 61.3448, mae: 5.5895, huber: 5.1157, swd: 9.3060, ept: 171.3536
      Epoch 11 composite train-obj: 4.402741
            No improvement (5.2829), counter 1/5
    Epoch [12/50], Train Losses: mse: 48.6114, mae: 4.8385, huber: 4.3739, swd: 9.1015, ept: 51.0022
    Epoch [12/50], Val Losses: mse: 64.7290, mae: 5.6025, huber: 5.1336, swd: 7.7844, ept: 186.7389
    Epoch [12/50], Test Losses: mse: 60.8280, mae: 5.3915, huber: 4.9244, swd: 7.9457, ept: 188.9772
      Epoch 12 composite train-obj: 4.373856
            No improvement (5.1336), counter 2/5
    Epoch [13/50], Train Losses: mse: 48.0864, mae: 4.7992, huber: 4.3351, swd: 8.8427, ept: 51.2064
    Epoch [13/50], Val Losses: mse: 63.9212, mae: 5.5850, huber: 5.1159, swd: 8.9573, ept: 185.1754
    Epoch [13/50], Test Losses: mse: 60.8114, mae: 5.4582, huber: 4.9898, swd: 9.4545, ept: 183.6859
      Epoch 13 composite train-obj: 4.335087
            No improvement (5.1159), counter 3/5
    Epoch [14/50], Train Losses: mse: 47.6320, mae: 4.7657, huber: 4.3019, swd: 8.6112, ept: 51.4265
    Epoch [14/50], Val Losses: mse: 63.0241, mae: 5.5329, huber: 5.0650, swd: 8.6903, ept: 188.9162
    Epoch [14/50], Test Losses: mse: 59.5254, mae: 5.3537, huber: 4.8871, swd: 8.8118, ept: 190.3549
      Epoch 14 composite train-obj: 4.301947
            Val objective improved 5.0784 → 5.0650, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 47.3295, mae: 4.7434, huber: 4.2799, swd: 8.4289, ept: 51.3003
    Epoch [15/50], Val Losses: mse: 66.1630, mae: 5.6906, huber: 5.2206, swd: 7.7200, ept: 180.7564
    Epoch [15/50], Test Losses: mse: 62.0488, mae: 5.4711, huber: 5.0020, swd: 8.2659, ept: 184.5579
      Epoch 15 composite train-obj: 4.279918
            No improvement (5.2206), counter 1/5
    Epoch [16/50], Train Losses: mse: 47.0673, mae: 4.7245, huber: 4.2612, swd: 8.3161, ept: 51.3751
    Epoch [16/50], Val Losses: mse: 64.9665, mae: 5.6945, huber: 5.2189, swd: 7.8197, ept: 185.0243
    Epoch [16/50], Test Losses: mse: 62.2022, mae: 5.5615, huber: 5.0869, swd: 7.8420, ept: 186.3106
      Epoch 16 composite train-obj: 4.261230
            No improvement (5.2189), counter 2/5
    Epoch [17/50], Train Losses: mse: 46.6017, mae: 4.6906, huber: 4.2278, swd: 8.1044, ept: 51.3951
    Epoch [17/50], Val Losses: mse: 64.7585, mae: 5.6010, huber: 5.1331, swd: 7.8993, ept: 194.2132
    Epoch [17/50], Test Losses: mse: 60.2038, mae: 5.3874, huber: 4.9204, swd: 8.4168, ept: 195.7409
      Epoch 17 composite train-obj: 4.227830
            No improvement (5.1331), counter 3/5
    Epoch [18/50], Train Losses: mse: 46.3376, mae: 4.6713, huber: 4.2087, swd: 7.9705, ept: 51.8032
    Epoch [18/50], Val Losses: mse: 65.8849, mae: 5.6013, huber: 5.1330, swd: 6.9323, ept: 197.8764
    Epoch [18/50], Test Losses: mse: 61.1649, mae: 5.3805, huber: 4.9137, swd: 7.4197, ept: 199.4110
      Epoch 18 composite train-obj: 4.208665
            No improvement (5.1330), counter 4/5
    Epoch [19/50], Train Losses: mse: 46.0751, mae: 4.6498, huber: 4.1875, swd: 7.8204, ept: 51.7957
    Epoch [19/50], Val Losses: mse: 65.2529, mae: 5.6655, huber: 5.1930, swd: 7.5699, ept: 191.4511
    Epoch [19/50], Test Losses: mse: 61.3061, mae: 5.4854, huber: 5.0140, swd: 8.1175, ept: 193.1247
      Epoch 19 composite train-obj: 4.187539
    Epoch [19/50], Test Losses: mse: 59.5254, mae: 5.3537, huber: 4.8871, swd: 8.8118, ept: 190.3549
    Best round's Test MSE: 59.5254, MAE: 5.3537, SWD: 8.8118
    Best round's Validation MSE: 63.0241, MAE: 5.5329, SWD: 8.6903
    Best round's Test verification MSE : 59.5254, MAE: 5.3537, SWD: 8.8118
    Time taken: 124.69 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq336_pred720_20250513_2321)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 59.7778 ± 0.5504
      mae: 5.4001 ± 0.0352
      huber: 4.9327 ± 0.0345
      swd: 10.0533 ± 1.7525
      ept: 176.8817 ± 13.1934
      count: 33.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 62.2831 ± 0.8063
      mae: 5.5392 ± 0.0048
      huber: 5.0705 ± 0.0043
      swd: 9.6802 ± 1.5125
      ept: 178.1015 ± 9.9776
      count: 33.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 295.99 seconds
    
    Experiment complete: PatchTST_lorenz_seq336_pred720_20250513_2321
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### Dlinear

#### 336-96
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=336,
    pred_len=96,
    channels=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 282
    Validation Batches: 38
    Test Batches: 78
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 59.9519, mae: 5.5218, huber: 5.0525, swd: 21.0345, ept: 40.2217
    Epoch [1/50], Val Losses: mse: 62.2805, mae: 5.7107, huber: 5.2364, swd: 17.0906, ept: 43.9335
    Epoch [1/50], Test Losses: mse: 63.5833, mae: 5.7472, huber: 5.2731, swd: 17.3513, ept: 43.6345
      Epoch 1 composite train-obj: 5.052539
            Val objective improved inf → 5.2364, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 54.2740, mae: 5.1173, huber: 4.6559, swd: 17.6696, ept: 49.2608
    Epoch [2/50], Val Losses: mse: 61.2972, mae: 5.6428, huber: 5.1743, swd: 18.8743, ept: 45.5413
    Epoch [2/50], Test Losses: mse: 62.2762, mae: 5.6389, huber: 5.1703, swd: 18.7326, ept: 45.4572
      Epoch 2 composite train-obj: 4.655905
            Val objective improved 5.2364 → 5.1743, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 53.9818, mae: 5.0631, huber: 4.6047, swd: 17.3607, ept: 50.5326
    Epoch [3/50], Val Losses: mse: 58.9292, mae: 5.4865, huber: 5.0185, swd: 18.1409, ept: 45.7515
    Epoch [3/50], Test Losses: mse: 60.5543, mae: 5.5233, huber: 5.0560, swd: 18.0110, ept: 46.0206
      Epoch 3 composite train-obj: 4.604686
            Val objective improved 5.1743 → 5.0185, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 53.3680, mae: 5.0185, huber: 4.5617, swd: 16.9089, ept: 50.8239
    Epoch [4/50], Val Losses: mse: 57.3162, mae: 5.3169, huber: 4.8519, swd: 17.7116, ept: 49.0196
    Epoch [4/50], Test Losses: mse: 58.3868, mae: 5.3332, huber: 4.8694, swd: 17.8885, ept: 48.4199
      Epoch 4 composite train-obj: 4.561747
            Val objective improved 5.0185 → 4.8519, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 53.3472, mae: 4.9954, huber: 4.5400, swd: 16.6351, ept: 51.3230
    Epoch [5/50], Val Losses: mse: 57.7832, mae: 5.3892, huber: 4.9242, swd: 17.9864, ept: 47.2313
    Epoch [5/50], Test Losses: mse: 58.7134, mae: 5.4066, huber: 4.9410, swd: 17.8166, ept: 47.5923
      Epoch 5 composite train-obj: 4.539968
            No improvement (4.9242), counter 1/5
    Epoch [6/50], Train Losses: mse: 53.1038, mae: 4.9813, huber: 4.5266, swd: 16.5845, ept: 51.3750
    Epoch [6/50], Val Losses: mse: 58.7967, mae: 5.4341, huber: 4.9714, swd: 17.4612, ept: 47.0618
    Epoch [6/50], Test Losses: mse: 60.0975, mae: 5.4642, huber: 5.0013, swd: 17.4199, ept: 47.1794
      Epoch 6 composite train-obj: 4.526607
            No improvement (4.9714), counter 2/5
    Epoch [7/50], Train Losses: mse: 53.3456, mae: 4.9826, huber: 4.5288, swd: 16.3974, ept: 51.6003
    Epoch [7/50], Val Losses: mse: 59.0661, mae: 5.4910, huber: 5.0281, swd: 18.0038, ept: 44.6326
    Epoch [7/50], Test Losses: mse: 60.1766, mae: 5.5219, huber: 5.0591, swd: 17.8306, ept: 44.5277
      Epoch 7 composite train-obj: 4.528767
            No improvement (5.0281), counter 3/5
    Epoch [8/50], Train Losses: mse: 52.8605, mae: 4.9643, huber: 4.5110, swd: 16.5977, ept: 51.4178
    Epoch [8/50], Val Losses: mse: 56.9834, mae: 5.3129, huber: 4.8521, swd: 16.5377, ept: 46.8766
    Epoch [8/50], Test Losses: mse: 58.5642, mae: 5.3482, huber: 4.8868, swd: 16.4400, ept: 46.7609
      Epoch 8 composite train-obj: 4.511000
            No improvement (4.8521), counter 4/5
    Epoch [9/50], Train Losses: mse: 52.9725, mae: 4.9551, huber: 4.5022, swd: 16.1119, ept: 51.9720
    Epoch [9/50], Val Losses: mse: 58.6556, mae: 5.4051, huber: 4.9419, swd: 16.2158, ept: 48.0923
    Epoch [9/50], Test Losses: mse: 60.7008, mae: 5.4852, huber: 5.0209, swd: 16.4160, ept: 47.9620
      Epoch 9 composite train-obj: 4.502192
    Epoch [9/50], Test Losses: mse: 58.3868, mae: 5.3332, huber: 4.8694, swd: 17.8885, ept: 48.4199
    Best round's Test MSE: 58.3868, MAE: 5.3332, SWD: 17.8885
    Best round's Validation MSE: 57.3162, MAE: 5.3169, SWD: 17.7116
    Best round's Test verification MSE : 58.3868, MAE: 5.3332, SWD: 17.8885
    Time taken: 19.97 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 60.4512, mae: 5.5352, huber: 5.0659, swd: 21.2489, ept: 40.3461
    Epoch [1/50], Val Losses: mse: 58.8454, mae: 5.5867, huber: 5.1146, swd: 20.3439, ept: 41.2696
    Epoch [1/50], Test Losses: mse: 59.3454, mae: 5.5749, huber: 5.1032, swd: 20.1640, ept: 41.9923
      Epoch 1 composite train-obj: 5.065947
            Val objective improved inf → 5.1146, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 54.2042, mae: 5.1171, huber: 4.6558, swd: 18.0682, ept: 49.1957
    Epoch [2/50], Val Losses: mse: 58.2523, mae: 5.4288, huber: 4.9612, swd: 18.1301, ept: 46.4426
    Epoch [2/50], Test Losses: mse: 59.5052, mae: 5.4589, huber: 4.9916, swd: 17.9676, ept: 46.5382
      Epoch 2 composite train-obj: 4.655805
            Val objective improved 5.1146 → 4.9612, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 53.8571, mae: 5.0506, huber: 4.5923, swd: 17.2123, ept: 50.4357
    Epoch [3/50], Val Losses: mse: 58.1673, mae: 5.3909, huber: 4.9275, swd: 17.2875, ept: 46.5237
    Epoch [3/50], Test Losses: mse: 59.3039, mae: 5.4184, huber: 4.9549, swd: 17.6099, ept: 45.8579
      Epoch 3 composite train-obj: 4.592259
            Val objective improved 4.9612 → 4.9275, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 53.5264, mae: 5.0173, huber: 4.5610, swd: 16.8803, ept: 50.9382
    Epoch [4/50], Val Losses: mse: 57.1085, mae: 5.3360, huber: 4.8703, swd: 17.1660, ept: 46.2521
    Epoch [4/50], Test Losses: mse: 58.1227, mae: 5.3397, huber: 4.8739, swd: 17.1579, ept: 45.6906
      Epoch 4 composite train-obj: 4.560976
            Val objective improved 4.9275 → 4.8703, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 53.2156, mae: 4.9931, huber: 4.5377, swd: 16.6403, ept: 51.1931
    Epoch [5/50], Val Losses: mse: 58.3973, mae: 5.3708, huber: 4.9038, swd: 16.6485, ept: 49.1458
    Epoch [5/50], Test Losses: mse: 58.7849, mae: 5.3473, huber: 4.8805, swd: 16.7411, ept: 48.7446
      Epoch 5 composite train-obj: 4.537672
            No improvement (4.9038), counter 1/5
    Epoch [6/50], Train Losses: mse: 53.4483, mae: 5.0031, huber: 4.5480, swd: 16.6370, ept: 51.3594
    Epoch [6/50], Val Losses: mse: 56.9839, mae: 5.3177, huber: 4.8500, swd: 17.6611, ept: 49.9648
    Epoch [6/50], Test Losses: mse: 58.2905, mae: 5.3375, huber: 4.8705, swd: 17.7854, ept: 49.2839
      Epoch 6 composite train-obj: 4.548008
            Val objective improved 4.8703 → 4.8500, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 53.0399, mae: 4.9661, huber: 4.5124, swd: 16.5108, ept: 51.8807
    Epoch [7/50], Val Losses: mse: 60.0620, mae: 5.4851, huber: 5.0223, swd: 16.2243, ept: 48.6420
    Epoch [7/50], Test Losses: mse: 61.0623, mae: 5.4837, huber: 5.0224, swd: 16.2570, ept: 48.6775
      Epoch 7 composite train-obj: 4.512360
            No improvement (5.0223), counter 1/5
    Epoch [8/50], Train Losses: mse: 52.9790, mae: 4.9612, huber: 4.5084, swd: 16.4048, ept: 51.8146
    Epoch [8/50], Val Losses: mse: 60.3014, mae: 5.4809, huber: 5.0173, swd: 17.3562, ept: 48.4836
    Epoch [8/50], Test Losses: mse: 61.8486, mae: 5.5171, huber: 5.0539, swd: 17.3689, ept: 47.7540
      Epoch 8 composite train-obj: 4.508359
            No improvement (5.0173), counter 2/5
    Epoch [9/50], Train Losses: mse: 53.1268, mae: 4.9570, huber: 4.5043, swd: 16.4170, ept: 52.0147
    Epoch [9/50], Val Losses: mse: 57.6441, mae: 5.3576, huber: 4.8969, swd: 16.2666, ept: 47.4157
    Epoch [9/50], Test Losses: mse: 59.0808, mae: 5.4023, huber: 4.9403, swd: 16.2695, ept: 46.8850
      Epoch 9 composite train-obj: 4.504325
            No improvement (4.8969), counter 3/5
    Epoch [10/50], Train Losses: mse: 52.7864, mae: 4.9410, huber: 4.4890, swd: 16.1685, ept: 52.0084
    Epoch [10/50], Val Losses: mse: 57.7351, mae: 5.3452, huber: 4.8817, swd: 16.0601, ept: 47.7925
    Epoch [10/50], Test Losses: mse: 59.3399, mae: 5.3950, huber: 4.9309, swd: 16.1293, ept: 47.6385
      Epoch 10 composite train-obj: 4.489007
            No improvement (4.8817), counter 4/5
    Epoch [11/50], Train Losses: mse: 52.9282, mae: 4.9340, huber: 4.4824, swd: 16.2346, ept: 52.0521
    Epoch [11/50], Val Losses: mse: 57.4653, mae: 5.3218, huber: 4.8629, swd: 17.9630, ept: 45.9659
    Epoch [11/50], Test Losses: mse: 58.3482, mae: 5.3426, huber: 4.8832, swd: 17.9873, ept: 46.7049
      Epoch 11 composite train-obj: 4.482442
    Epoch [11/50], Test Losses: mse: 58.2905, mae: 5.3375, huber: 4.8705, swd: 17.7854, ept: 49.2839
    Best round's Test MSE: 58.2905, MAE: 5.3375, SWD: 17.7854
    Best round's Validation MSE: 56.9839, MAE: 5.3177, SWD: 17.6611
    Best round's Test verification MSE : 58.2905, MAE: 5.3375, SWD: 17.7854
    Time taken: 25.69 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 60.6269, mae: 5.5422, huber: 5.0733, swd: 19.7413, ept: 39.7817
    Epoch [1/50], Val Losses: mse: 59.8630, mae: 5.7262, huber: 5.2519, swd: 19.2196, ept: 39.1812
    Epoch [1/50], Test Losses: mse: 60.3255, mae: 5.6931, huber: 5.2208, swd: 19.0570, ept: 38.8027
      Epoch 1 composite train-obj: 5.073293
            Val objective improved inf → 5.2519, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 54.1033, mae: 5.1165, huber: 4.6554, swd: 16.8221, ept: 49.0336
    Epoch [2/50], Val Losses: mse: 58.2131, mae: 5.3989, huber: 4.9358, swd: 17.2510, ept: 44.5493
    Epoch [2/50], Test Losses: mse: 59.1188, mae: 5.4138, huber: 4.9504, swd: 17.1352, ept: 44.6240
      Epoch 2 composite train-obj: 4.655385
            Val objective improved 5.2519 → 4.9358, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 53.7748, mae: 5.0429, huber: 4.5850, swd: 15.7491, ept: 50.7156
    Epoch [3/50], Val Losses: mse: 57.1448, mae: 5.3524, huber: 4.8861, swd: 15.7453, ept: 47.0380
    Epoch [3/50], Test Losses: mse: 58.9930, mae: 5.4059, huber: 4.9394, swd: 15.5679, ept: 46.6535
      Epoch 3 composite train-obj: 4.585033
            Val objective improved 4.9358 → 4.8861, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 53.5247, mae: 5.0222, huber: 4.5657, swd: 15.6985, ept: 50.8184
    Epoch [4/50], Val Losses: mse: 58.5374, mae: 5.3626, huber: 4.8981, swd: 15.3484, ept: 49.7333
    Epoch [4/50], Test Losses: mse: 59.4292, mae: 5.3708, huber: 4.9067, swd: 15.5547, ept: 49.1053
      Epoch 4 composite train-obj: 4.565718
            No improvement (4.8981), counter 1/5
    Epoch [5/50], Train Losses: mse: 53.2686, mae: 5.0030, huber: 4.5476, swd: 15.5096, ept: 51.0521
    Epoch [5/50], Val Losses: mse: 58.6369, mae: 5.4649, huber: 4.9992, swd: 16.0106, ept: 45.5090
    Epoch [5/50], Test Losses: mse: 59.5629, mae: 5.4696, huber: 5.0049, swd: 15.9194, ept: 44.8235
      Epoch 5 composite train-obj: 4.547586
            No improvement (4.9992), counter 2/5
    Epoch [6/50], Train Losses: mse: 53.3937, mae: 4.9935, huber: 4.5389, swd: 15.1791, ept: 51.5751
    Epoch [6/50], Val Losses: mse: 58.8928, mae: 5.4643, huber: 4.9988, swd: 15.9862, ept: 46.7058
    Epoch [6/50], Test Losses: mse: 59.7748, mae: 5.4654, huber: 5.0012, swd: 15.8749, ept: 46.1024
      Epoch 6 composite train-obj: 4.538917
            No improvement (4.9988), counter 3/5
    Epoch [7/50], Train Losses: mse: 53.0505, mae: 4.9734, huber: 4.5193, swd: 15.1203, ept: 51.6518
    Epoch [7/50], Val Losses: mse: 58.8517, mae: 5.3784, huber: 4.9204, swd: 15.4028, ept: 48.3764
    Epoch [7/50], Test Losses: mse: 60.3924, mae: 5.4352, huber: 4.9752, swd: 15.4841, ept: 48.0177
      Epoch 7 composite train-obj: 4.519259
            No improvement (4.9204), counter 4/5
    Epoch [8/50], Train Losses: mse: 53.0782, mae: 4.9620, huber: 4.5087, swd: 14.9272, ept: 51.9711
    Epoch [8/50], Val Losses: mse: 58.6670, mae: 5.3937, huber: 4.9318, swd: 16.1757, ept: 49.2265
    Epoch [8/50], Test Losses: mse: 59.3925, mae: 5.3956, huber: 4.9328, swd: 16.2775, ept: 49.3548
      Epoch 8 composite train-obj: 4.508733
    Epoch [8/50], Test Losses: mse: 58.9930, mae: 5.4059, huber: 4.9394, swd: 15.5679, ept: 46.6535
    Best round's Test MSE: 58.9930, MAE: 5.4059, SWD: 15.5679
    Best round's Validation MSE: 57.1448, MAE: 5.3524, SWD: 15.7453
    Best round's Test verification MSE : 58.9930, MAE: 5.4059, SWD: 15.5679
    Time taken: 19.53 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq336_pred96_20250514_0138)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 58.5568 ± 0.3109
      mae: 5.3589 ± 0.0333
      huber: 4.8931 ± 0.0327
      swd: 17.0806 ± 1.0705
      ept: 48.1191 ± 1.0947
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 57.1483 ± 0.1357
      mae: 5.3290 ± 0.0165
      huber: 4.8627 ± 0.0166
      swd: 17.0393 ± 0.9153
      ept: 48.6741 ± 1.2196
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 65.22 seconds
    
    Experiment complete: DLinear_lorenz_seq336_pred96_20250514_0138
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 336-196
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=336,
    pred_len=196,
    channels=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 281
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 281
    Validation Batches: 37
    Test Batches: 78
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 64.8719, mae: 5.8817, huber: 5.4086, swd: 23.1786, ept: 45.4832
    Epoch [1/50], Val Losses: mse: 64.1199, mae: 5.8673, huber: 5.3957, swd: 22.1888, ept: 49.9234
    Epoch [1/50], Test Losses: mse: 62.9839, mae: 5.7375, huber: 5.2674, swd: 22.1445, ept: 51.7206
      Epoch 1 composite train-obj: 5.408615
            Val objective improved inf → 5.3957, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 59.6481, mae: 5.5646, huber: 5.0963, swd: 20.6816, ept: 57.9730
    Epoch [2/50], Val Losses: mse: 63.0139, mae: 5.7787, huber: 5.3110, swd: 22.6590, ept: 52.3963
    Epoch [2/50], Test Losses: mse: 61.9857, mae: 5.6476, huber: 5.1811, swd: 22.4494, ept: 54.6313
      Epoch 2 composite train-obj: 5.096336
            Val objective improved 5.3957 → 5.3110, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.3624, mae: 5.5299, huber: 5.0631, swd: 20.1609, ept: 59.4238
    Epoch [3/50], Val Losses: mse: 63.9296, mae: 5.8334, huber: 5.3631, swd: 20.5012, ept: 53.4340
    Epoch [3/50], Test Losses: mse: 63.4922, mae: 5.7238, huber: 5.2554, swd: 20.2789, ept: 55.6988
      Epoch 3 composite train-obj: 5.063108
            No improvement (5.3631), counter 1/5
    Epoch [4/50], Train Losses: mse: 59.2376, mae: 5.5190, huber: 5.0529, swd: 19.9856, ept: 60.0154
    Epoch [4/50], Val Losses: mse: 63.4808, mae: 5.7692, huber: 5.3022, swd: 20.8013, ept: 55.6718
    Epoch [4/50], Test Losses: mse: 62.8154, mae: 5.6644, huber: 5.1977, swd: 20.7202, ept: 56.9427
      Epoch 4 composite train-obj: 5.052939
            Val objective improved 5.3110 → 5.3022, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 59.1901, mae: 5.5097, huber: 5.0442, swd: 19.8466, ept: 60.3272
    Epoch [5/50], Val Losses: mse: 63.0152, mae: 5.7470, huber: 5.2811, swd: 20.3121, ept: 53.8312
    Epoch [5/50], Test Losses: mse: 62.3960, mae: 5.6365, huber: 5.1719, swd: 19.9158, ept: 55.9491
      Epoch 5 composite train-obj: 5.044182
            Val objective improved 5.3022 → 5.2811, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 59.1320, mae: 5.5041, huber: 5.0388, swd: 19.6496, ept: 60.6517
    Epoch [6/50], Val Losses: mse: 63.9615, mae: 5.7854, huber: 5.3190, swd: 19.6171, ept: 55.8123
    Epoch [6/50], Test Losses: mse: 63.5297, mae: 5.7007, huber: 5.2340, swd: 19.5044, ept: 57.0525
      Epoch 6 composite train-obj: 5.038817
            No improvement (5.3190), counter 1/5
    Epoch [7/50], Train Losses: mse: 59.1202, mae: 5.4949, huber: 5.0302, swd: 19.5816, ept: 61.0467
    Epoch [7/50], Val Losses: mse: 64.0016, mae: 5.8121, huber: 5.3440, swd: 19.7523, ept: 56.4916
    Epoch [7/50], Test Losses: mse: 62.8784, mae: 5.6783, huber: 5.2113, swd: 19.7107, ept: 58.0530
      Epoch 7 composite train-obj: 5.030164
            No improvement (5.3440), counter 2/5
    Epoch [8/50], Train Losses: mse: 59.1046, mae: 5.4886, huber: 5.0243, swd: 19.4269, ept: 61.2761
    Epoch [8/50], Val Losses: mse: 63.5959, mae: 5.7718, huber: 5.3065, swd: 20.1146, ept: 55.9697
    Epoch [8/50], Test Losses: mse: 62.9967, mae: 5.6627, huber: 5.1985, swd: 19.9033, ept: 57.3690
      Epoch 8 composite train-obj: 5.024281
            No improvement (5.3065), counter 3/5
    Epoch [9/50], Train Losses: mse: 58.9422, mae: 5.4846, huber: 5.0202, swd: 19.5649, ept: 61.1719
    Epoch [9/50], Val Losses: mse: 63.2516, mae: 5.7562, huber: 5.2904, swd: 20.3615, ept: 55.7080
    Epoch [9/50], Test Losses: mse: 62.9297, mae: 5.6691, huber: 5.2046, swd: 20.1323, ept: 56.8558
      Epoch 9 composite train-obj: 5.020162
            No improvement (5.2904), counter 4/5
    Epoch [10/50], Train Losses: mse: 58.9556, mae: 5.4820, huber: 5.0180, swd: 19.4451, ept: 61.2068
    Epoch [10/50], Val Losses: mse: 63.0536, mae: 5.7474, huber: 5.2817, swd: 20.4875, ept: 54.4751
    Epoch [10/50], Test Losses: mse: 61.5694, mae: 5.5933, huber: 5.1286, swd: 20.2974, ept: 56.6720
      Epoch 10 composite train-obj: 5.018037
    Epoch [10/50], Test Losses: mse: 62.3960, mae: 5.6365, huber: 5.1719, swd: 19.9158, ept: 55.9491
    Best round's Test MSE: 62.3960, MAE: 5.6365, SWD: 19.9158
    Best round's Validation MSE: 63.0152, MAE: 5.7470, SWD: 20.3121
    Best round's Test verification MSE : 62.3960, MAE: 5.6365, SWD: 19.9158
    Time taken: 25.43 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 64.3085, mae: 5.8639, huber: 5.3908, swd: 22.9884, ept: 45.5222
    Epoch [1/50], Val Losses: mse: 64.0737, mae: 5.8648, huber: 5.3935, swd: 22.0861, ept: 49.6588
    Epoch [1/50], Test Losses: mse: 62.9603, mae: 5.7376, huber: 5.2672, swd: 21.9852, ept: 51.6436
      Epoch 1 composite train-obj: 5.390815
            Val objective improved inf → 5.3935, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 59.6177, mae: 5.5608, huber: 5.0925, swd: 20.5546, ept: 58.0856
    Epoch [2/50], Val Losses: mse: 63.3633, mae: 5.8031, huber: 5.3348, swd: 22.2368, ept: 51.8744
    Epoch [2/50], Test Losses: mse: 62.0350, mae: 5.6610, huber: 5.1940, swd: 22.0203, ept: 54.0338
      Epoch 2 composite train-obj: 5.092532
            Val objective improved 5.3935 → 5.3348, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.2912, mae: 5.5305, huber: 5.0636, swd: 20.2505, ept: 59.3887
    Epoch [3/50], Val Losses: mse: 64.4067, mae: 5.8309, huber: 5.3625, swd: 19.9536, ept: 54.1944
    Epoch [3/50], Test Losses: mse: 63.8064, mae: 5.7259, huber: 5.2582, swd: 19.6646, ept: 55.8079
      Epoch 3 composite train-obj: 5.063587
            No improvement (5.3625), counter 1/5
    Epoch [4/50], Train Losses: mse: 59.2619, mae: 5.5177, huber: 5.0515, swd: 19.8608, ept: 59.9116
    Epoch [4/50], Val Losses: mse: 63.8580, mae: 5.8086, huber: 5.3400, swd: 20.5223, ept: 54.6863
    Epoch [4/50], Test Losses: mse: 63.2210, mae: 5.7020, huber: 5.2349, swd: 20.3312, ept: 56.3156
      Epoch 4 composite train-obj: 5.051525
            No improvement (5.3400), counter 2/5
    Epoch [5/50], Train Losses: mse: 59.2208, mae: 5.5060, huber: 5.0407, swd: 19.6806, ept: 60.5656
    Epoch [5/50], Val Losses: mse: 63.3920, mae: 5.7777, huber: 5.3117, swd: 20.7958, ept: 53.7159
    Epoch [5/50], Test Losses: mse: 62.6383, mae: 5.6635, huber: 5.1984, swd: 20.4099, ept: 55.8139
      Epoch 5 composite train-obj: 5.040650
            Val objective improved 5.3348 → 5.3117, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 59.0938, mae: 5.4983, huber: 5.0334, swd: 19.6536, ept: 60.7085
    Epoch [6/50], Val Losses: mse: 63.0483, mae: 5.7626, huber: 5.2951, swd: 20.6158, ept: 53.4381
    Epoch [6/50], Test Losses: mse: 62.0623, mae: 5.6286, huber: 5.1629, swd: 20.4649, ept: 55.5807
      Epoch 6 composite train-obj: 5.033412
            Val objective improved 5.3117 → 5.2951, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 59.1306, mae: 5.4944, huber: 5.0297, swd: 19.4875, ept: 61.1165
    Epoch [7/50], Val Losses: mse: 63.2059, mae: 5.7677, huber: 5.3025, swd: 20.5903, ept: 54.0455
    Epoch [7/50], Test Losses: mse: 62.2165, mae: 5.6467, huber: 5.1817, swd: 20.2281, ept: 56.0811
      Epoch 7 composite train-obj: 5.029668
            No improvement (5.3025), counter 1/5
    Epoch [8/50], Train Losses: mse: 58.9664, mae: 5.4889, huber: 5.0246, swd: 19.5630, ept: 60.9317
    Epoch [8/50], Val Losses: mse: 63.3372, mae: 5.7561, huber: 5.2901, swd: 20.1103, ept: 55.6229
    Epoch [8/50], Test Losses: mse: 62.6012, mae: 5.6361, huber: 5.1718, swd: 19.8485, ept: 57.0135
      Epoch 8 composite train-obj: 5.024600
            Val objective improved 5.2951 → 5.2901, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 58.9904, mae: 5.4818, huber: 5.0177, swd: 19.3297, ept: 61.1279
    Epoch [9/50], Val Losses: mse: 63.4316, mae: 5.7592, huber: 5.2925, swd: 20.7809, ept: 57.2258
    Epoch [9/50], Test Losses: mse: 62.1544, mae: 5.6197, huber: 5.1541, swd: 20.4971, ept: 58.2546
      Epoch 9 composite train-obj: 5.017749
            No improvement (5.2925), counter 1/5
    Epoch [10/50], Train Losses: mse: 59.0492, mae: 5.4840, huber: 5.0200, swd: 19.2835, ept: 61.3284
    Epoch [10/50], Val Losses: mse: 62.8916, mae: 5.7381, huber: 5.2722, swd: 21.0841, ept: 56.0004
    Epoch [10/50], Test Losses: mse: 62.1032, mae: 5.6216, huber: 5.1565, swd: 20.6977, ept: 57.6643
      Epoch 10 composite train-obj: 5.020000
            Val objective improved 5.2901 → 5.2722, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 58.9202, mae: 5.4793, huber: 5.0153, swd: 19.5016, ept: 61.4977
    Epoch [11/50], Val Losses: mse: 63.0861, mae: 5.7606, huber: 5.2945, swd: 20.3981, ept: 54.4628
    Epoch [11/50], Test Losses: mse: 62.9865, mae: 5.6847, huber: 5.2190, swd: 20.1263, ept: 56.4362
      Epoch 11 composite train-obj: 5.015339
            No improvement (5.2945), counter 1/5
    Epoch [12/50], Train Losses: mse: 58.8906, mae: 5.4791, huber: 5.0153, swd: 19.3304, ept: 61.4445
    Epoch [12/50], Val Losses: mse: 63.1922, mae: 5.7319, huber: 5.2662, swd: 19.4624, ept: 57.3070
    Epoch [12/50], Test Losses: mse: 62.7828, mae: 5.6340, huber: 5.1696, swd: 19.1479, ept: 58.2103
      Epoch 12 composite train-obj: 5.015284
            Val objective improved 5.2722 → 5.2662, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 58.9914, mae: 5.4732, huber: 5.0099, swd: 19.1146, ept: 61.6127
    Epoch [13/50], Val Losses: mse: 63.3087, mae: 5.7345, huber: 5.2700, swd: 19.8468, ept: 56.0506
    Epoch [13/50], Test Losses: mse: 63.0029, mae: 5.6471, huber: 5.1830, swd: 19.4778, ept: 57.5135
      Epoch 13 composite train-obj: 5.009920
            No improvement (5.2700), counter 1/5
    Epoch [14/50], Train Losses: mse: 58.9370, mae: 5.4742, huber: 5.0109, swd: 19.3154, ept: 61.8841
    Epoch [14/50], Val Losses: mse: 63.9659, mae: 5.7756, huber: 5.3098, swd: 18.8115, ept: 57.6105
    Epoch [14/50], Test Losses: mse: 63.4229, mae: 5.6772, huber: 5.2120, swd: 18.8420, ept: 59.0797
      Epoch 14 composite train-obj: 5.010908
            No improvement (5.3098), counter 2/5
    Epoch [15/50], Train Losses: mse: 58.8085, mae: 5.4665, huber: 5.0032, swd: 19.3011, ept: 61.6884
    Epoch [15/50], Val Losses: mse: 63.0815, mae: 5.7526, huber: 5.2868, swd: 20.3617, ept: 54.7018
    Epoch [15/50], Test Losses: mse: 61.3961, mae: 5.5868, huber: 5.1234, swd: 20.0887, ept: 56.3348
      Epoch 15 composite train-obj: 5.003197
            No improvement (5.2868), counter 3/5
    Epoch [16/50], Train Losses: mse: 58.8756, mae: 5.4726, huber: 5.0095, swd: 19.2895, ept: 61.6108
    Epoch [16/50], Val Losses: mse: 63.0724, mae: 5.7160, huber: 5.2506, swd: 19.5138, ept: 55.6305
    Epoch [16/50], Test Losses: mse: 62.6451, mae: 5.6174, huber: 5.1534, swd: 19.4023, ept: 57.7359
      Epoch 16 composite train-obj: 5.009458
            Val objective improved 5.2662 → 5.2506, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 58.8461, mae: 5.4645, huber: 5.0016, swd: 19.1902, ept: 62.0825
    Epoch [17/50], Val Losses: mse: 63.1667, mae: 5.7257, huber: 5.2616, swd: 19.8619, ept: 55.5708
    Epoch [17/50], Test Losses: mse: 62.3776, mae: 5.6131, huber: 5.1497, swd: 19.5907, ept: 57.3464
      Epoch 17 composite train-obj: 5.001588
            No improvement (5.2616), counter 1/5
    Epoch [18/50], Train Losses: mse: 58.8410, mae: 5.4663, huber: 5.0033, swd: 19.1732, ept: 61.8160
    Epoch [18/50], Val Losses: mse: 62.9651, mae: 5.7364, huber: 5.2711, swd: 19.8091, ept: 54.9519
    Epoch [18/50], Test Losses: mse: 62.5198, mae: 5.6301, huber: 5.1672, swd: 19.6330, ept: 57.0407
      Epoch 18 composite train-obj: 5.003323
            No improvement (5.2711), counter 2/5
    Epoch [19/50], Train Losses: mse: 58.8232, mae: 5.4643, huber: 5.0015, swd: 19.1914, ept: 61.7960
    Epoch [19/50], Val Losses: mse: 62.7142, mae: 5.7287, huber: 5.2640, swd: 20.5654, ept: 55.9416
    Epoch [19/50], Test Losses: mse: 61.7436, mae: 5.6009, huber: 5.1375, swd: 20.3102, ept: 57.7036
      Epoch 19 composite train-obj: 5.001549
            No improvement (5.2640), counter 3/5
    Epoch [20/50], Train Losses: mse: 58.7248, mae: 5.4590, huber: 4.9962, swd: 19.1973, ept: 61.9758
    Epoch [20/50], Val Losses: mse: 62.9468, mae: 5.7253, huber: 5.2607, swd: 20.8606, ept: 54.5835
    Epoch [20/50], Test Losses: mse: 62.0033, mae: 5.6034, huber: 5.1407, swd: 20.4655, ept: 57.0342
      Epoch 20 composite train-obj: 4.996226
            No improvement (5.2607), counter 4/5
    Epoch [21/50], Train Losses: mse: 58.7593, mae: 5.4610, huber: 4.9983, swd: 19.1870, ept: 62.0750
    Epoch [21/50], Val Losses: mse: 63.4161, mae: 5.7270, huber: 5.2630, swd: 19.3126, ept: 55.5273
    Epoch [21/50], Test Losses: mse: 62.6496, mae: 5.6119, huber: 5.1489, swd: 19.1187, ept: 57.7757
      Epoch 21 composite train-obj: 4.998258
    Epoch [21/50], Test Losses: mse: 62.6451, mae: 5.6174, huber: 5.1534, swd: 19.4023, ept: 57.7359
    Best round's Test MSE: 62.6451, MAE: 5.6174, SWD: 19.4023
    Best round's Validation MSE: 63.0724, MAE: 5.7160, SWD: 19.5138
    Best round's Test verification MSE : 62.6451, MAE: 5.6174, SWD: 19.4023
    Time taken: 51.26 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 64.5250, mae: 5.8695, huber: 5.3964, swd: 22.0132, ept: 45.3303
    Epoch [1/50], Val Losses: mse: 63.8503, mae: 5.8449, huber: 5.3741, swd: 21.8395, ept: 50.0533
    Epoch [1/50], Test Losses: mse: 63.0046, mae: 5.7212, huber: 5.2527, swd: 21.7815, ept: 51.9259
      Epoch 1 composite train-obj: 5.396406
            Val objective improved inf → 5.3741, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 59.5468, mae: 5.5612, huber: 5.0929, swd: 19.8463, ept: 57.7488
    Epoch [2/50], Val Losses: mse: 63.5160, mae: 5.7982, huber: 5.3288, swd: 20.1506, ept: 53.6195
    Epoch [2/50], Test Losses: mse: 63.0981, mae: 5.7098, huber: 5.2414, swd: 20.2012, ept: 54.6700
      Epoch 2 composite train-obj: 5.092858
            Val objective improved 5.3741 → 5.3288, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.3133, mae: 5.5302, huber: 5.0632, swd: 19.3157, ept: 59.3542
    Epoch [3/50], Val Losses: mse: 64.1001, mae: 5.8174, huber: 5.3489, swd: 19.6231, ept: 54.2814
    Epoch [3/50], Test Losses: mse: 62.8236, mae: 5.6747, huber: 5.2077, swd: 19.6583, ept: 56.3163
      Epoch 3 composite train-obj: 5.063167
            No improvement (5.3489), counter 1/5
    Epoch [4/50], Train Losses: mse: 59.3083, mae: 5.5160, huber: 5.0500, swd: 18.9157, ept: 60.2296
    Epoch [4/50], Val Losses: mse: 63.7587, mae: 5.8054, huber: 5.3379, swd: 19.9368, ept: 53.3340
    Epoch [4/50], Test Losses: mse: 62.6797, mae: 5.6656, huber: 5.1999, swd: 19.8226, ept: 55.6406
      Epoch 4 composite train-obj: 5.050042
            No improvement (5.3379), counter 2/5
    Epoch [5/50], Train Losses: mse: 59.1923, mae: 5.5104, huber: 5.0447, swd: 18.8932, ept: 60.5759
    Epoch [5/50], Val Losses: mse: 63.2905, mae: 5.7650, huber: 5.2979, swd: 19.4324, ept: 55.2391
    Epoch [5/50], Test Losses: mse: 62.7760, mae: 5.6687, huber: 5.2025, swd: 19.3731, ept: 56.4281
      Epoch 5 composite train-obj: 5.044736
            Val objective improved 5.3288 → 5.2979, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 59.1041, mae: 5.4961, huber: 5.0311, swd: 18.6884, ept: 60.7368
    Epoch [6/50], Val Losses: mse: 63.9143, mae: 5.8092, huber: 5.3406, swd: 19.4068, ept: 54.3805
    Epoch [6/50], Test Losses: mse: 63.3617, mae: 5.7106, huber: 5.2425, swd: 19.2501, ept: 56.2794
      Epoch 6 composite train-obj: 5.031069
            No improvement (5.3406), counter 1/5
    Epoch [7/50], Train Losses: mse: 59.0449, mae: 5.4938, huber: 5.0290, swd: 18.7630, ept: 60.9321
    Epoch [7/50], Val Losses: mse: 63.1517, mae: 5.7527, huber: 5.2871, swd: 18.9670, ept: 54.4860
    Epoch [7/50], Test Losses: mse: 62.6636, mae: 5.6559, huber: 5.1912, swd: 19.0605, ept: 56.3372
      Epoch 7 composite train-obj: 5.029047
            Val objective improved 5.2979 → 5.2871, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 59.1130, mae: 5.4922, huber: 5.0277, swd: 18.5931, ept: 61.1606
    Epoch [8/50], Val Losses: mse: 63.3919, mae: 5.7758, huber: 5.3091, swd: 19.1684, ept: 55.6922
    Epoch [8/50], Test Losses: mse: 62.8480, mae: 5.6731, huber: 5.2078, swd: 19.1946, ept: 57.2471
      Epoch 8 composite train-obj: 5.027678
            No improvement (5.3091), counter 1/5
    Epoch [9/50], Train Losses: mse: 58.9608, mae: 5.4835, huber: 5.0193, swd: 18.5500, ept: 61.2344
    Epoch [9/50], Val Losses: mse: 63.0876, mae: 5.7285, huber: 5.2640, swd: 19.8999, ept: 55.6865
    Epoch [9/50], Test Losses: mse: 62.5259, mae: 5.6357, huber: 5.1714, swd: 19.8474, ept: 57.4023
      Epoch 9 composite train-obj: 5.019344
            Val objective improved 5.2871 → 5.2640, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 58.9105, mae: 5.4778, huber: 5.0138, swd: 18.5891, ept: 61.4272
    Epoch [10/50], Val Losses: mse: 63.5287, mae: 5.7574, huber: 5.2924, swd: 18.5165, ept: 55.4029
    Epoch [10/50], Test Losses: mse: 63.0572, mae: 5.6555, huber: 5.1919, swd: 18.7149, ept: 57.9180
      Epoch 10 composite train-obj: 5.013787
            No improvement (5.2924), counter 1/5
    Epoch [11/50], Train Losses: mse: 59.0082, mae: 5.4802, huber: 5.0164, swd: 18.4149, ept: 61.6708
    Epoch [11/50], Val Losses: mse: 62.8168, mae: 5.7122, huber: 5.2481, swd: 19.4224, ept: 56.2850
    Epoch [11/50], Test Losses: mse: 62.2116, mae: 5.6131, huber: 5.1499, swd: 19.2469, ept: 57.3463
      Epoch 11 composite train-obj: 5.016426
            Val objective improved 5.2640 → 5.2481, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 58.9221, mae: 5.4776, huber: 5.0139, swd: 18.4738, ept: 61.4285
    Epoch [12/50], Val Losses: mse: 63.2859, mae: 5.7404, huber: 5.2750, swd: 18.4692, ept: 56.4986
    Epoch [12/50], Test Losses: mse: 62.7977, mae: 5.6467, huber: 5.1821, swd: 18.3215, ept: 58.2970
      Epoch 12 composite train-obj: 5.013889
            No improvement (5.2750), counter 1/5
    Epoch [13/50], Train Losses: mse: 58.9621, mae: 5.4748, huber: 5.0114, swd: 18.3594, ept: 61.5388
    Epoch [13/50], Val Losses: mse: 62.4842, mae: 5.7135, huber: 5.2481, swd: 20.0063, ept: 55.0816
    Epoch [13/50], Test Losses: mse: 61.7686, mae: 5.6087, huber: 5.1442, swd: 20.0469, ept: 57.2609
      Epoch 13 composite train-obj: 5.011428
            Val objective improved 5.2481 → 5.2481, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 58.8760, mae: 5.4703, huber: 5.0070, swd: 18.3919, ept: 61.4378
    Epoch [14/50], Val Losses: mse: 62.9607, mae: 5.7295, huber: 5.2638, swd: 18.8292, ept: 55.7494
    Epoch [14/50], Test Losses: mse: 62.8160, mae: 5.6494, huber: 5.1846, swd: 18.7129, ept: 57.3162
      Epoch 14 composite train-obj: 5.007016
            No improvement (5.2638), counter 1/5
    Epoch [15/50], Train Losses: mse: 58.8596, mae: 5.4696, huber: 5.0065, swd: 18.3645, ept: 61.6561
    Epoch [15/50], Val Losses: mse: 63.6119, mae: 5.7540, huber: 5.2893, swd: 18.2786, ept: 57.0612
    Epoch [15/50], Test Losses: mse: 63.3776, mae: 5.6709, huber: 5.2068, swd: 18.3767, ept: 58.3077
      Epoch 15 composite train-obj: 5.006517
            No improvement (5.2893), counter 2/5
    Epoch [16/50], Train Losses: mse: 58.8890, mae: 5.4680, huber: 5.0050, swd: 18.3406, ept: 61.7882
    Epoch [16/50], Val Losses: mse: 63.4790, mae: 5.7545, huber: 5.2893, swd: 18.4394, ept: 55.4301
    Epoch [16/50], Test Losses: mse: 62.9434, mae: 5.6489, huber: 5.1851, swd: 18.4577, ept: 56.8904
      Epoch 16 composite train-obj: 5.004960
            No improvement (5.2893), counter 3/5
    Epoch [17/50], Train Losses: mse: 58.8144, mae: 5.4668, huber: 5.0039, swd: 18.3086, ept: 61.7576
    Epoch [17/50], Val Losses: mse: 63.0570, mae: 5.7220, huber: 5.2567, swd: 19.0155, ept: 56.8407
    Epoch [17/50], Test Losses: mse: 62.6676, mae: 5.6240, huber: 5.1606, swd: 18.9160, ept: 57.8569
      Epoch 17 composite train-obj: 5.003902
            No improvement (5.2567), counter 4/5
    Epoch [18/50], Train Losses: mse: 58.8065, mae: 5.4627, huber: 4.9998, swd: 18.3145, ept: 61.9181
    Epoch [18/50], Val Losses: mse: 63.1656, mae: 5.7339, huber: 5.2680, swd: 18.5916, ept: 56.5320
    Epoch [18/50], Test Losses: mse: 62.9591, mae: 5.6428, huber: 5.1786, swd: 18.5436, ept: 58.7896
      Epoch 18 composite train-obj: 4.999835
    Epoch [18/50], Test Losses: mse: 61.7686, mae: 5.6087, huber: 5.1442, swd: 20.0469, ept: 57.2609
    Best round's Test MSE: 61.7686, MAE: 5.6087, SWD: 20.0469
    Best round's Validation MSE: 62.4842, MAE: 5.7135, SWD: 20.0063
    Best round's Test verification MSE : 61.7686, MAE: 5.6087, SWD: 20.0469
    Time taken: 47.82 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq336_pred196_20250514_0139)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 62.2699 ± 0.3688
      mae: 5.6209 ± 0.0116
      huber: 5.1565 ± 0.0115
      swd: 19.7883 ± 0.2781
      ept: 56.9820 ± 0.7557
      count: 37.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 62.8573 ± 0.2648
      mae: 5.7255 ± 0.0153
      huber: 5.2599 ± 0.0150
      swd: 19.9441 ± 0.3289
      ept: 54.8478 ± 0.7529
      count: 37.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 124.54 seconds
    
    Experiment complete: DLinear_lorenz_seq336_pred196_20250514_0139
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 336-336
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=336,
    pred_len=336,
    channels=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 280
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
    Training Batches: 280
    Validation Batches: 36
    Test Batches: 77
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 66.9666, mae: 6.0744, huber: 5.5987, swd: 24.4359, ept: 47.1646
    Epoch [1/50], Val Losses: mse: 66.1178, mae: 6.0648, huber: 5.5905, swd: 22.9704, ept: 51.4301
    Epoch [1/50], Test Losses: mse: 65.9576, mae: 5.9710, huber: 5.4980, swd: 23.1392, ept: 53.6332
      Epoch 1 composite train-obj: 5.598748
            Val objective improved inf → 5.5905, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.1860, mae: 5.8127, huber: 5.3403, swd: 22.5348, ept: 60.8169
    Epoch [2/50], Val Losses: mse: 65.9491, mae: 6.0516, huber: 5.5778, swd: 23.0427, ept: 55.3013
    Epoch [2/50], Test Losses: mse: 65.6855, mae: 5.9456, huber: 5.4730, swd: 22.9614, ept: 57.2346
      Epoch 2 composite train-obj: 5.340302
            Val objective improved 5.5905 → 5.5778, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 62.1187, mae: 5.7984, huber: 5.3267, swd: 22.0837, ept: 62.5167
    Epoch [3/50], Val Losses: mse: 66.6516, mae: 6.0952, huber: 5.6212, swd: 21.9706, ept: 55.9928
    Epoch [3/50], Test Losses: mse: 66.1856, mae: 5.9753, huber: 5.5028, swd: 21.8637, ept: 57.5957
      Epoch 3 composite train-obj: 5.326740
            No improvement (5.6212), counter 1/5
    Epoch [4/50], Train Losses: mse: 61.9993, mae: 5.7893, huber: 5.3181, swd: 21.8876, ept: 63.4721
    Epoch [4/50], Val Losses: mse: 67.1116, mae: 6.0871, huber: 5.6141, swd: 21.1031, ept: 57.0631
    Epoch [4/50], Test Losses: mse: 66.5531, mae: 5.9692, huber: 5.4978, swd: 20.9461, ept: 58.6070
      Epoch 4 composite train-obj: 5.318098
            No improvement (5.6141), counter 2/5
    Epoch [5/50], Train Losses: mse: 62.0044, mae: 5.7816, huber: 5.3109, swd: 21.7527, ept: 64.0420
    Epoch [5/50], Val Losses: mse: 67.6525, mae: 6.1290, huber: 5.6559, swd: 21.1391, ept: 58.0794
    Epoch [5/50], Test Losses: mse: 66.6580, mae: 5.9905, huber: 5.5189, swd: 21.0835, ept: 58.7838
      Epoch 5 composite train-obj: 5.310863
            No improvement (5.6559), counter 3/5
    Epoch [6/50], Train Losses: mse: 62.0151, mae: 5.7811, huber: 5.3104, swd: 21.5448, ept: 64.1589
    Epoch [6/50], Val Losses: mse: 66.4304, mae: 6.0531, huber: 5.5806, swd: 21.9218, ept: 58.4097
    Epoch [6/50], Test Losses: mse: 65.7314, mae: 5.9263, huber: 5.4552, swd: 21.7386, ept: 59.8208
      Epoch 6 composite train-obj: 5.310418
            No improvement (5.5806), counter 4/5
    Epoch [7/50], Train Losses: mse: 61.8979, mae: 5.7753, huber: 5.3048, swd: 21.6296, ept: 64.4779
    Epoch [7/50], Val Losses: mse: 65.5716, mae: 6.0203, huber: 5.5477, swd: 22.0209, ept: 58.3983
    Epoch [7/50], Test Losses: mse: 65.6574, mae: 5.9339, huber: 5.4625, swd: 22.1952, ept: 59.6151
      Epoch 7 composite train-obj: 5.304795
            Val objective improved 5.5778 → 5.5477, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 61.9522, mae: 5.7728, huber: 5.3026, swd: 21.4892, ept: 64.4426
    Epoch [8/50], Val Losses: mse: 66.3146, mae: 6.0584, huber: 5.5854, swd: 21.8772, ept: 58.0599
    Epoch [8/50], Test Losses: mse: 65.9378, mae: 5.9476, huber: 5.4758, swd: 21.9305, ept: 59.3930
      Epoch 8 composite train-obj: 5.302628
            No improvement (5.5854), counter 1/5
    Epoch [9/50], Train Losses: mse: 61.8914, mae: 5.7712, huber: 5.3010, swd: 21.5536, ept: 64.8827
    Epoch [9/50], Val Losses: mse: 65.7880, mae: 6.0145, huber: 5.5427, swd: 22.1042, ept: 57.6680
    Epoch [9/50], Test Losses: mse: 65.5792, mae: 5.9173, huber: 5.4470, swd: 21.9940, ept: 59.3255
      Epoch 9 composite train-obj: 5.301012
            Val objective improved 5.5477 → 5.5427, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 61.8528, mae: 5.7673, huber: 5.2974, swd: 21.5022, ept: 64.8164
    Epoch [10/50], Val Losses: mse: 65.6282, mae: 6.0110, huber: 5.5395, swd: 21.9323, ept: 58.9878
    Epoch [10/50], Test Losses: mse: 65.2748, mae: 5.9058, huber: 5.4351, swd: 22.0438, ept: 60.5347
      Epoch 10 composite train-obj: 5.297365
            Val objective improved 5.5427 → 5.5395, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 61.7661, mae: 5.7604, huber: 5.2905, swd: 21.5225, ept: 64.9991
    Epoch [11/50], Val Losses: mse: 66.1502, mae: 6.0335, huber: 5.5618, swd: 21.4302, ept: 56.9081
    Epoch [11/50], Test Losses: mse: 65.6299, mae: 5.9198, huber: 5.4490, swd: 21.5931, ept: 58.8253
      Epoch 11 composite train-obj: 5.290534
            No improvement (5.5618), counter 1/5
    Epoch [12/50], Train Losses: mse: 61.8577, mae: 5.7666, huber: 5.2968, swd: 21.4714, ept: 64.8389
    Epoch [12/50], Val Losses: mse: 66.0677, mae: 6.0430, huber: 5.5709, swd: 21.2784, ept: 59.2557
    Epoch [12/50], Test Losses: mse: 65.9421, mae: 5.9425, huber: 5.4718, swd: 21.2589, ept: 60.1771
      Epoch 12 composite train-obj: 5.296827
            No improvement (5.5709), counter 2/5
    Epoch [13/50], Train Losses: mse: 61.7869, mae: 5.7608, huber: 5.2911, swd: 21.4750, ept: 65.2635
    Epoch [13/50], Val Losses: mse: 66.1583, mae: 6.0427, huber: 5.5709, swd: 21.6528, ept: 57.5509
    Epoch [13/50], Test Losses: mse: 65.7191, mae: 5.9327, huber: 5.4617, swd: 21.7063, ept: 59.3697
      Epoch 13 composite train-obj: 5.291093
            No improvement (5.5709), counter 3/5
    Epoch [14/50], Train Losses: mse: 61.7614, mae: 5.7573, huber: 5.2878, swd: 21.4397, ept: 65.2103
    Epoch [14/50], Val Losses: mse: 66.5366, mae: 6.0465, huber: 5.5750, swd: 20.7055, ept: 57.4312
    Epoch [14/50], Test Losses: mse: 66.0426, mae: 5.9269, huber: 5.4570, swd: 20.6670, ept: 59.8430
      Epoch 14 composite train-obj: 5.287805
            No improvement (5.5750), counter 4/5
    Epoch [15/50], Train Losses: mse: 61.8113, mae: 5.7586, huber: 5.2891, swd: 21.3404, ept: 65.2719
    Epoch [15/50], Val Losses: mse: 66.0038, mae: 6.0385, huber: 5.5669, swd: 22.2421, ept: 55.8471
    Epoch [15/50], Test Losses: mse: 65.1194, mae: 5.9067, huber: 5.4360, swd: 22.2573, ept: 57.3206
      Epoch 15 composite train-obj: 5.289124
    Epoch [15/50], Test Losses: mse: 65.2748, mae: 5.9058, huber: 5.4351, swd: 22.0438, ept: 60.5347
    Best round's Test MSE: 65.2748, MAE: 5.9058, SWD: 22.0438
    Best round's Validation MSE: 65.6282, MAE: 6.0110, SWD: 21.9323
    Best round's Test verification MSE : 65.2748, MAE: 5.9058, SWD: 22.0438
    Time taken: 47.89 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 67.2003, mae: 6.0776, huber: 5.6021, swd: 24.7984, ept: 47.5272
    Epoch [1/50], Val Losses: mse: 66.3099, mae: 6.0820, huber: 5.6072, swd: 23.1155, ept: 52.6208
    Epoch [1/50], Test Losses: mse: 65.9672, mae: 5.9850, huber: 5.5110, swd: 23.5232, ept: 54.5069
      Epoch 1 composite train-obj: 5.602069
            Val objective improved inf → 5.6072, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.2558, mae: 5.8174, huber: 5.3448, swd: 22.7441, ept: 60.8994
    Epoch [2/50], Val Losses: mse: 67.2706, mae: 6.1010, huber: 5.6275, swd: 21.6479, ept: 55.0417
    Epoch [2/50], Test Losses: mse: 66.5114, mae: 5.9732, huber: 5.5009, swd: 21.6037, ept: 57.1255
      Epoch 2 composite train-obj: 5.344848
            No improvement (5.6275), counter 1/5
    Epoch [3/50], Train Losses: mse: 62.1216, mae: 5.7965, huber: 5.3248, swd: 22.2485, ept: 62.9025
    Epoch [3/50], Val Losses: mse: 66.0535, mae: 6.0597, huber: 5.5861, swd: 23.5574, ept: 54.0608
    Epoch [3/50], Test Losses: mse: 64.8203, mae: 5.9144, huber: 5.4420, swd: 23.5538, ept: 57.9005
      Epoch 3 composite train-obj: 5.324815
            Val objective improved 5.6072 → 5.5861, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 61.9935, mae: 5.7903, huber: 5.3191, swd: 22.2799, ept: 63.4457
    Epoch [4/50], Val Losses: mse: 66.3686, mae: 6.0584, huber: 5.5849, swd: 22.0843, ept: 56.3735
    Epoch [4/50], Test Losses: mse: 66.0459, mae: 5.9488, huber: 5.4768, swd: 22.1730, ept: 58.4989
      Epoch 4 composite train-obj: 5.319062
            Val objective improved 5.5861 → 5.5849, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 62.0215, mae: 5.7840, huber: 5.3130, swd: 22.0187, ept: 64.2012
    Epoch [5/50], Val Losses: mse: 66.5020, mae: 6.0578, huber: 5.5851, swd: 21.9555, ept: 58.5075
    Epoch [5/50], Test Losses: mse: 65.4928, mae: 5.9222, huber: 5.4502, swd: 21.9741, ept: 59.6249
      Epoch 5 composite train-obj: 5.312995
            No improvement (5.5851), counter 1/5
    Epoch [6/50], Train Losses: mse: 61.9803, mae: 5.7785, huber: 5.3079, swd: 21.9118, ept: 64.4329
    Epoch [6/50], Val Losses: mse: 65.9824, mae: 6.0258, huber: 5.5535, swd: 21.6265, ept: 57.2296
    Epoch [6/50], Test Losses: mse: 65.6219, mae: 5.9191, huber: 5.4480, swd: 21.7837, ept: 58.6445
      Epoch 6 composite train-obj: 5.307881
            Val objective improved 5.5849 → 5.5535, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 61.9213, mae: 5.7743, huber: 5.3040, swd: 21.9057, ept: 64.3302
    Epoch [7/50], Val Losses: mse: 66.6616, mae: 6.0749, huber: 5.6021, swd: 21.6812, ept: 57.6518
    Epoch [7/50], Test Losses: mse: 65.9807, mae: 5.9559, huber: 5.4846, swd: 21.7584, ept: 58.5370
      Epoch 7 composite train-obj: 5.303955
            No improvement (5.6021), counter 1/5
    Epoch [8/50], Train Losses: mse: 61.9156, mae: 5.7734, huber: 5.3031, swd: 21.8165, ept: 64.5336
    Epoch [8/50], Val Losses: mse: 66.2303, mae: 6.0385, huber: 5.5663, swd: 22.3298, ept: 57.2964
    Epoch [8/50], Test Losses: mse: 65.6157, mae: 5.9188, huber: 5.4485, swd: 22.3127, ept: 59.4161
      Epoch 8 composite train-obj: 5.303121
            No improvement (5.5663), counter 2/5
    Epoch [9/50], Train Losses: mse: 61.8696, mae: 5.7678, huber: 5.2977, swd: 21.7040, ept: 64.7778
    Epoch [9/50], Val Losses: mse: 66.2228, mae: 6.0335, huber: 5.5624, swd: 22.4312, ept: 57.1597
    Epoch [9/50], Test Losses: mse: 65.6290, mae: 5.9246, huber: 5.4540, swd: 22.4562, ept: 58.8842
      Epoch 9 composite train-obj: 5.297658
            No improvement (5.5624), counter 3/5
    Epoch [10/50], Train Losses: mse: 61.9692, mae: 5.7728, huber: 5.3028, swd: 21.7377, ept: 64.8847
    Epoch [10/50], Val Losses: mse: 66.5966, mae: 6.0603, huber: 5.5879, swd: 21.9902, ept: 56.3395
    Epoch [10/50], Test Losses: mse: 65.5616, mae: 5.9220, huber: 5.4510, swd: 21.8375, ept: 59.2206
      Epoch 10 composite train-obj: 5.302829
            No improvement (5.5879), counter 4/5
    Epoch [11/50], Train Losses: mse: 61.8932, mae: 5.7670, huber: 5.2971, swd: 21.7051, ept: 65.2458
    Epoch [11/50], Val Losses: mse: 65.7193, mae: 6.0242, huber: 5.5529, swd: 22.1946, ept: 57.0387
    Epoch [11/50], Test Losses: mse: 65.6895, mae: 5.9295, huber: 5.4594, swd: 22.1604, ept: 58.9081
      Epoch 11 composite train-obj: 5.297137
            Val objective improved 5.5535 → 5.5529, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 61.8396, mae: 5.7669, huber: 5.2971, swd: 21.7162, ept: 64.6969
    Epoch [12/50], Val Losses: mse: 65.8921, mae: 6.0337, huber: 5.5610, swd: 22.1234, ept: 56.7768
    Epoch [12/50], Test Losses: mse: 65.8212, mae: 5.9370, huber: 5.4659, swd: 22.2468, ept: 58.8611
      Epoch 12 composite train-obj: 5.297106
            No improvement (5.5610), counter 1/5
    Epoch [13/50], Train Losses: mse: 61.7188, mae: 5.7587, huber: 5.2890, swd: 21.7536, ept: 64.9924
    Epoch [13/50], Val Losses: mse: 66.1081, mae: 6.0342, huber: 5.5625, swd: 21.6670, ept: 57.9130
    Epoch [13/50], Test Losses: mse: 66.0126, mae: 5.9463, huber: 5.4757, swd: 21.9250, ept: 59.2989
      Epoch 13 composite train-obj: 5.289014
            No improvement (5.5625), counter 2/5
    Epoch [14/50], Train Losses: mse: 61.9001, mae: 5.7667, huber: 5.2970, swd: 21.6131, ept: 65.2305
    Epoch [14/50], Val Losses: mse: 65.5756, mae: 6.0172, huber: 5.5449, swd: 22.5145, ept: 58.3477
    Epoch [14/50], Test Losses: mse: 65.2706, mae: 5.9083, huber: 5.4374, swd: 22.4935, ept: 59.7053
      Epoch 14 composite train-obj: 5.297049
            Val objective improved 5.5529 → 5.5449, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 61.7775, mae: 5.7603, huber: 5.2908, swd: 21.6757, ept: 65.1030
    Epoch [15/50], Val Losses: mse: 65.6543, mae: 6.0131, huber: 5.5421, swd: 22.3218, ept: 57.6947
    Epoch [15/50], Test Losses: mse: 64.8667, mae: 5.8896, huber: 5.4199, swd: 22.2235, ept: 59.6877
      Epoch 15 composite train-obj: 5.290840
            Val objective improved 5.5449 → 5.5421, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 61.8198, mae: 5.7664, huber: 5.2968, swd: 21.7331, ept: 64.9167
    Epoch [16/50], Val Losses: mse: 65.8123, mae: 6.0084, huber: 5.5373, swd: 21.8147, ept: 58.5518
    Epoch [16/50], Test Losses: mse: 65.6098, mae: 5.9098, huber: 5.4398, swd: 21.8958, ept: 60.4124
      Epoch 16 composite train-obj: 5.296756
            Val objective improved 5.5421 → 5.5373, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 61.8147, mae: 5.7598, huber: 5.2905, swd: 21.5860, ept: 65.2632
    Epoch [17/50], Val Losses: mse: 66.7676, mae: 6.0684, huber: 5.5969, swd: 21.0760, ept: 58.2182
    Epoch [17/50], Test Losses: mse: 66.6224, mae: 5.9755, huber: 5.5048, swd: 21.0525, ept: 59.5689
      Epoch 17 composite train-obj: 5.290470
            No improvement (5.5969), counter 1/5
    Epoch [18/50], Train Losses: mse: 61.7469, mae: 5.7520, huber: 5.2828, swd: 21.5024, ept: 65.5763
    Epoch [18/50], Val Losses: mse: 66.1429, mae: 6.0387, huber: 5.5673, swd: 22.1990, ept: 55.9461
    Epoch [18/50], Test Losses: mse: 65.5205, mae: 5.9198, huber: 5.4495, swd: 22.1429, ept: 58.8807
      Epoch 18 composite train-obj: 5.282777
            No improvement (5.5673), counter 2/5
    Epoch [19/50], Train Losses: mse: 61.7547, mae: 5.7567, huber: 5.2874, swd: 21.5672, ept: 65.3212
    Epoch [19/50], Val Losses: mse: 65.6806, mae: 6.0140, huber: 5.5428, swd: 22.4824, ept: 59.1010
    Epoch [19/50], Test Losses: mse: 65.1219, mae: 5.8960, huber: 5.4259, swd: 22.4877, ept: 60.1698
      Epoch 19 composite train-obj: 5.287433
            No improvement (5.5428), counter 3/5
    Epoch [20/50], Train Losses: mse: 61.7341, mae: 5.7521, huber: 5.2829, swd: 21.5496, ept: 65.4547
    Epoch [20/50], Val Losses: mse: 66.1913, mae: 6.0293, huber: 5.5583, swd: 21.8525, ept: 58.4061
    Epoch [20/50], Test Losses: mse: 65.1042, mae: 5.8928, huber: 5.4228, swd: 22.0075, ept: 60.4261
      Epoch 20 composite train-obj: 5.282948
            No improvement (5.5583), counter 4/5
    Epoch [21/50], Train Losses: mse: 61.7635, mae: 5.7585, huber: 5.2892, swd: 21.6474, ept: 65.4310
    Epoch [21/50], Val Losses: mse: 66.0953, mae: 6.0255, huber: 5.5546, swd: 21.6929, ept: 57.9596
    Epoch [21/50], Test Losses: mse: 65.4421, mae: 5.8996, huber: 5.4299, swd: 21.8377, ept: 59.7030
      Epoch 21 composite train-obj: 5.289249
    Epoch [21/50], Test Losses: mse: 65.6098, mae: 5.9098, huber: 5.4398, swd: 21.8958, ept: 60.4124
    Best round's Test MSE: 65.6098, MAE: 5.9098, SWD: 21.8958
    Best round's Validation MSE: 65.8123, MAE: 6.0084, SWD: 21.8147
    Best round's Test verification MSE : 65.6098, MAE: 5.9098, SWD: 21.8958
    Time taken: 57.61 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 66.9982, mae: 6.0780, huber: 5.6023, swd: 24.6092, ept: 47.3409
    Epoch [1/50], Val Losses: mse: 66.9711, mae: 6.1275, huber: 5.6513, swd: 22.9875, ept: 51.1234
    Epoch [1/50], Test Losses: mse: 66.7096, mae: 6.0137, huber: 5.5395, swd: 22.9268, ept: 53.5093
      Epoch 1 composite train-obj: 5.602269
            Val objective improved inf → 5.6513, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.2554, mae: 5.8163, huber: 5.3438, swd: 22.5376, ept: 61.1204
    Epoch [2/50], Val Losses: mse: 65.9167, mae: 6.0555, huber: 5.5814, swd: 23.3345, ept: 55.7319
    Epoch [2/50], Test Losses: mse: 65.3292, mae: 5.9461, huber: 5.4725, swd: 23.3064, ept: 57.2926
      Epoch 2 composite train-obj: 5.343795
            Val objective improved 5.6513 → 5.5814, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 62.0874, mae: 5.7971, huber: 5.3254, swd: 22.1948, ept: 62.6226
    Epoch [3/50], Val Losses: mse: 65.5163, mae: 6.0222, huber: 5.5495, swd: 23.4327, ept: 56.6094
    Epoch [3/50], Test Losses: mse: 65.2789, mae: 5.9252, huber: 5.4539, swd: 23.4140, ept: 57.4818
      Epoch 3 composite train-obj: 5.325404
            Val objective improved 5.5814 → 5.5495, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 62.0231, mae: 5.7896, huber: 5.3184, swd: 22.0629, ept: 63.7443
    Epoch [4/50], Val Losses: mse: 65.8874, mae: 6.0378, huber: 5.5647, swd: 23.1349, ept: 55.9252
    Epoch [4/50], Test Losses: mse: 65.3495, mae: 5.9239, huber: 5.4520, swd: 23.1793, ept: 57.9313
      Epoch 4 composite train-obj: 5.318395
            No improvement (5.5647), counter 1/5
    Epoch [5/50], Train Losses: mse: 62.0266, mae: 5.7847, huber: 5.3138, swd: 21.9065, ept: 63.9051
    Epoch [5/50], Val Losses: mse: 67.4017, mae: 6.1127, huber: 5.6388, swd: 21.5189, ept: 56.4132
    Epoch [5/50], Test Losses: mse: 66.4554, mae: 5.9803, huber: 5.5081, swd: 21.3790, ept: 58.4377
      Epoch 5 composite train-obj: 5.313850
            No improvement (5.6388), counter 2/5
    Epoch [6/50], Train Losses: mse: 61.9811, mae: 5.7805, huber: 5.3099, swd: 21.7999, ept: 64.1165
    Epoch [6/50], Val Losses: mse: 66.0583, mae: 6.0470, huber: 5.5747, swd: 22.1905, ept: 55.7009
    Epoch [6/50], Test Losses: mse: 65.5566, mae: 5.9340, huber: 5.4631, swd: 22.0640, ept: 57.7083
      Epoch 6 composite train-obj: 5.309878
            No improvement (5.5747), counter 3/5
    Epoch [7/50], Train Losses: mse: 61.9358, mae: 5.7764, huber: 5.3059, swd: 21.7709, ept: 64.0640
    Epoch [7/50], Val Losses: mse: 66.3456, mae: 6.0568, huber: 5.5842, swd: 21.6016, ept: 59.1484
    Epoch [7/50], Test Losses: mse: 66.1308, mae: 5.9542, huber: 5.4827, swd: 21.5332, ept: 59.7124
      Epoch 7 composite train-obj: 5.305931
            No improvement (5.5842), counter 4/5
    Epoch [8/50], Train Losses: mse: 61.9135, mae: 5.7740, huber: 5.3036, swd: 21.6557, ept: 64.7385
    Epoch [8/50], Val Losses: mse: 65.8054, mae: 6.0135, huber: 5.5424, swd: 22.6551, ept: 58.3499
    Epoch [8/50], Test Losses: mse: 65.2640, mae: 5.8979, huber: 5.4278, swd: 22.6244, ept: 59.6505
      Epoch 8 composite train-obj: 5.303607
            Val objective improved 5.5495 → 5.5424, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 61.8358, mae: 5.7680, huber: 5.2979, swd: 21.6843, ept: 64.8272
    Epoch [9/50], Val Losses: mse: 65.5312, mae: 6.0099, huber: 5.5383, swd: 22.9564, ept: 56.8302
    Epoch [9/50], Test Losses: mse: 64.7336, mae: 5.8712, huber: 5.4008, swd: 22.7864, ept: 59.1798
      Epoch 9 composite train-obj: 5.297920
            Val objective improved 5.5424 → 5.5383, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 61.9194, mae: 5.7691, huber: 5.2991, swd: 21.5142, ept: 64.9122
    Epoch [10/50], Val Losses: mse: 65.8621, mae: 6.0332, huber: 5.5613, swd: 22.4680, ept: 55.4813
    Epoch [10/50], Test Losses: mse: 64.9962, mae: 5.8999, huber: 5.4292, swd: 22.4086, ept: 58.0622
      Epoch 10 composite train-obj: 5.299097
            No improvement (5.5613), counter 1/5
    Epoch [11/50], Train Losses: mse: 61.8697, mae: 5.7654, huber: 5.2956, swd: 21.5361, ept: 64.9221
    Epoch [11/50], Val Losses: mse: 65.3643, mae: 6.0127, huber: 5.5407, swd: 22.8215, ept: 57.6386
    Epoch [11/50], Test Losses: mse: 64.7889, mae: 5.8926, huber: 5.4220, swd: 22.7367, ept: 58.6929
      Epoch 11 composite train-obj: 5.295574
            No improvement (5.5407), counter 2/5
    Epoch [12/50], Train Losses: mse: 61.7750, mae: 5.7640, huber: 5.2942, swd: 21.6032, ept: 64.6456
    Epoch [12/50], Val Losses: mse: 66.5729, mae: 6.0521, huber: 5.5804, swd: 21.2146, ept: 59.1046
    Epoch [12/50], Test Losses: mse: 66.6254, mae: 5.9662, huber: 5.4957, swd: 21.0196, ept: 60.1288
      Epoch 12 composite train-obj: 5.294201
            No improvement (5.5804), counter 3/5
    Epoch [13/50], Train Losses: mse: 61.8658, mae: 5.7659, huber: 5.2960, swd: 21.4737, ept: 65.1355
    Epoch [13/50], Val Losses: mse: 65.6935, mae: 6.0020, huber: 5.5303, swd: 21.9342, ept: 57.9290
    Epoch [13/50], Test Losses: mse: 65.3541, mae: 5.8983, huber: 5.4279, swd: 21.8023, ept: 59.2538
      Epoch 13 composite train-obj: 5.296036
            Val objective improved 5.5383 → 5.5303, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 61.8165, mae: 5.7588, huber: 5.2893, swd: 21.4275, ept: 65.3471
    Epoch [14/50], Val Losses: mse: 65.6457, mae: 6.0003, huber: 5.5289, swd: 22.7571, ept: 58.2465
    Epoch [14/50], Test Losses: mse: 65.2896, mae: 5.8931, huber: 5.4229, swd: 22.6239, ept: 59.4212
      Epoch 14 composite train-obj: 5.289329
            Val objective improved 5.5303 → 5.5289, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 61.6909, mae: 5.7553, huber: 5.2858, swd: 21.5777, ept: 65.0682
    Epoch [15/50], Val Losses: mse: 66.0812, mae: 6.0417, huber: 5.5697, swd: 21.8135, ept: 58.9554
    Epoch [15/50], Test Losses: mse: 65.3262, mae: 5.9167, huber: 5.4461, swd: 21.8415, ept: 60.2690
      Epoch 15 composite train-obj: 5.285755
            No improvement (5.5697), counter 1/5
    Epoch [16/50], Train Losses: mse: 61.7860, mae: 5.7575, huber: 5.2881, swd: 21.4421, ept: 65.5273
    Epoch [16/50], Val Losses: mse: 66.6572, mae: 6.0688, huber: 5.5968, swd: 21.9111, ept: 58.0520
    Epoch [16/50], Test Losses: mse: 65.9208, mae: 5.9431, huber: 5.4726, swd: 21.7707, ept: 59.5947
      Epoch 16 composite train-obj: 5.288100
            No improvement (5.5968), counter 2/5
    Epoch [17/50], Train Losses: mse: 61.7229, mae: 5.7549, huber: 5.2856, swd: 21.4628, ept: 65.1126
    Epoch [17/50], Val Losses: mse: 66.6222, mae: 6.0645, huber: 5.5925, swd: 21.4085, ept: 57.9229
    Epoch [17/50], Test Losses: mse: 66.5736, mae: 5.9781, huber: 5.5074, swd: 21.3317, ept: 59.7701
      Epoch 17 composite train-obj: 5.285606
            No improvement (5.5925), counter 3/5
    Epoch [18/50], Train Losses: mse: 61.7881, mae: 5.7593, huber: 5.2899, swd: 21.4082, ept: 65.6327
    Epoch [18/50], Val Losses: mse: 65.7724, mae: 6.0154, huber: 5.5448, swd: 21.9584, ept: 57.6456
    Epoch [18/50], Test Losses: mse: 65.1188, mae: 5.8899, huber: 5.4204, swd: 21.7667, ept: 59.5029
      Epoch 18 composite train-obj: 5.289913
            No improvement (5.5448), counter 4/5
    Epoch [19/50], Train Losses: mse: 61.7215, mae: 5.7557, huber: 5.2864, swd: 21.4410, ept: 65.2947
    Epoch [19/50], Val Losses: mse: 65.7342, mae: 6.0090, huber: 5.5382, swd: 21.9857, ept: 58.7711
    Epoch [19/50], Test Losses: mse: 65.2916, mae: 5.9025, huber: 5.4330, swd: 21.9340, ept: 59.5260
      Epoch 19 composite train-obj: 5.286371
    Epoch [19/50], Test Losses: mse: 65.2896, mae: 5.8931, huber: 5.4229, swd: 22.6239, ept: 59.4212
    Best round's Test MSE: 65.2896, MAE: 5.8931, SWD: 22.6239
    Best round's Validation MSE: 65.6457, MAE: 6.0003, SWD: 22.7571
    Best round's Test verification MSE : 65.2896, MAE: 5.8931, SWD: 22.6239
    Time taken: 49.64 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq336_pred336_20250514_0141)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 65.3914 ± 0.1545
      mae: 5.9029 ± 0.0071
      huber: 5.4326 ± 0.0071
      swd: 22.1878 ± 0.3142
      ept: 60.1228 ± 0.4986
      count: 36.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 65.6954 ± 0.0830
      mae: 6.0066 ± 0.0046
      huber: 5.5352 ± 0.0046
      swd: 22.1680 ± 0.4193
      ept: 58.5954 ± 0.3042
      count: 36.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 155.18 seconds
    
    Experiment complete: DLinear_lorenz_seq336_pred336_20250514_0141
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 336-720
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=336,
    pred_len=720,
    channels=data_mgr.datasets['lorenz']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 277
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 336
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 277
    Validation Batches: 33
    Test Batches: 74
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.6884, mae: 6.3784, huber: 5.8998, swd: 27.5726, ept: 48.4960
    Epoch [1/50], Val Losses: mse: 68.4111, mae: 6.3065, huber: 5.8290, swd: 27.3986, ept: 51.6508
    Epoch [1/50], Test Losses: mse: 67.8979, mae: 6.2256, huber: 5.7485, swd: 27.4591, ept: 53.6173
      Epoch 1 composite train-obj: 5.899832
            Val objective improved inf → 5.8290, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 66.2772, mae: 6.1733, huber: 5.6964, swd: 26.2203, ept: 62.5782
    Epoch [2/50], Val Losses: mse: 68.4663, mae: 6.3331, huber: 5.8547, swd: 26.9425, ept: 54.5364
    Epoch [2/50], Test Losses: mse: 67.7816, mae: 6.2321, huber: 5.7548, swd: 26.8232, ept: 55.8660
      Epoch 2 composite train-obj: 5.696439
            No improvement (5.8547), counter 1/5
    Epoch [3/50], Train Losses: mse: 66.2385, mae: 6.1653, huber: 5.6889, swd: 25.9307, ept: 64.1283
    Epoch [3/50], Val Losses: mse: 68.1922, mae: 6.2958, huber: 5.8183, swd: 27.2027, ept: 54.5663
    Epoch [3/50], Test Losses: mse: 67.5001, mae: 6.2034, huber: 5.7264, swd: 27.1032, ept: 56.3852
      Epoch 3 composite train-obj: 5.688905
            Val objective improved 5.8290 → 5.8183, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 66.1353, mae: 6.1573, huber: 5.6810, swd: 25.8703, ept: 64.7900
    Epoch [4/50], Val Losses: mse: 68.8585, mae: 6.3260, huber: 5.8492, swd: 25.9345, ept: 56.2284
    Epoch [4/50], Test Losses: mse: 67.9917, mae: 6.2120, huber: 5.7357, swd: 25.8409, ept: 57.5889
      Epoch 4 composite train-obj: 5.681046
            No improvement (5.8492), counter 1/5
    Epoch [5/50], Train Losses: mse: 66.1205, mae: 6.1545, huber: 5.6784, swd: 25.7528, ept: 65.2889
    Epoch [5/50], Val Losses: mse: 68.2317, mae: 6.2974, huber: 5.8201, swd: 26.8784, ept: 54.4561
    Epoch [5/50], Test Losses: mse: 67.3276, mae: 6.1880, huber: 5.7118, swd: 26.6330, ept: 56.4114
      Epoch 5 composite train-obj: 5.678407
            No improvement (5.8201), counter 2/5
    Epoch [6/50], Train Losses: mse: 66.1324, mae: 6.1537, huber: 5.6777, swd: 25.6898, ept: 65.5928
    Epoch [6/50], Val Losses: mse: 69.0451, mae: 6.3337, huber: 5.8566, swd: 26.1287, ept: 56.6252
    Epoch [6/50], Test Losses: mse: 68.5089, mae: 6.2519, huber: 5.7755, swd: 26.0669, ept: 57.2879
      Epoch 6 composite train-obj: 5.677722
            No improvement (5.8566), counter 3/5
    Epoch [7/50], Train Losses: mse: 66.1179, mae: 6.1515, huber: 5.6757, swd: 25.6297, ept: 65.9454
    Epoch [7/50], Val Losses: mse: 69.1685, mae: 6.3290, huber: 5.8524, swd: 26.7227, ept: 55.8805
    Epoch [7/50], Test Losses: mse: 68.4125, mae: 6.2344, huber: 5.7584, swd: 26.2847, ept: 57.7103
      Epoch 7 composite train-obj: 5.675662
            No improvement (5.8524), counter 4/5
    Epoch [8/50], Train Losses: mse: 66.1208, mae: 6.1513, huber: 5.6755, swd: 25.5505, ept: 65.9561
    Epoch [8/50], Val Losses: mse: 68.1810, mae: 6.2957, huber: 5.8194, swd: 26.1670, ept: 55.8880
    Epoch [8/50], Test Losses: mse: 67.1541, mae: 6.1741, huber: 5.6988, swd: 26.1051, ept: 57.4772
      Epoch 8 composite train-obj: 5.675544
    Epoch [8/50], Test Losses: mse: 67.5001, mae: 6.2034, huber: 5.7264, swd: 27.1032, ept: 56.3852
    Best round's Test MSE: 67.5001, MAE: 6.2034, SWD: 27.1032
    Best round's Validation MSE: 68.1922, MAE: 6.2958, SWD: 27.2027
    Best round's Test verification MSE : 67.5001, MAE: 6.2034, SWD: 27.1032
    Time taken: 21.81 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.5260, mae: 6.3746, huber: 5.8961, swd: 27.3616, ept: 47.6722
    Epoch [1/50], Val Losses: mse: 67.8956, mae: 6.2937, huber: 5.8156, swd: 26.4750, ept: 49.2795
    Epoch [1/50], Test Losses: mse: 67.8041, mae: 6.2277, huber: 5.7507, swd: 26.8187, ept: 52.1892
      Epoch 1 composite train-obj: 5.896136
            Val objective improved inf → 5.8156, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 66.2128, mae: 6.1704, huber: 5.6935, swd: 26.1912, ept: 62.4403
    Epoch [2/50], Val Losses: mse: 68.1176, mae: 6.2953, huber: 5.8182, swd: 26.4827, ept: 53.6271
    Epoch [2/50], Test Losses: mse: 67.2558, mae: 6.1845, huber: 5.7080, swd: 26.7329, ept: 54.9684
      Epoch 2 composite train-obj: 5.693521
            No improvement (5.8182), counter 1/5
    Epoch [3/50], Train Losses: mse: 66.2521, mae: 6.1670, huber: 5.6906, swd: 25.8898, ept: 64.0407
    Epoch [3/50], Val Losses: mse: 68.7579, mae: 6.3375, huber: 5.8601, swd: 25.5775, ept: 56.1804
    Epoch [3/50], Test Losses: mse: 67.9731, mae: 6.2218, huber: 5.7453, swd: 25.8279, ept: 57.3129
      Epoch 3 composite train-obj: 5.690560
            No improvement (5.8601), counter 2/5
    Epoch [4/50], Train Losses: mse: 66.1800, mae: 6.1597, huber: 5.6835, swd: 25.6899, ept: 64.7711
    Epoch [4/50], Val Losses: mse: 68.2635, mae: 6.3043, huber: 5.8272, swd: 25.7650, ept: 55.8609
    Epoch [4/50], Test Losses: mse: 67.4964, mae: 6.1981, huber: 5.7220, swd: 25.9373, ept: 57.0816
      Epoch 4 composite train-obj: 5.683513
            No improvement (5.8272), counter 3/5
    Epoch [5/50], Train Losses: mse: 66.1746, mae: 6.1582, huber: 5.6822, swd: 25.6095, ept: 65.3247
    Epoch [5/50], Val Losses: mse: 68.5888, mae: 6.3156, huber: 5.8384, swd: 25.4951, ept: 56.6745
    Epoch [5/50], Test Losses: mse: 67.7449, mae: 6.2046, huber: 5.7285, swd: 25.8155, ept: 58.3463
      Epoch 5 composite train-obj: 5.682186
            No improvement (5.8384), counter 4/5
    Epoch [6/50], Train Losses: mse: 66.1030, mae: 6.1525, huber: 5.6766, swd: 25.6486, ept: 65.8402
    Epoch [6/50], Val Losses: mse: 68.2945, mae: 6.2821, huber: 5.8053, swd: 25.4247, ept: 59.0819
    Epoch [6/50], Test Losses: mse: 67.7083, mae: 6.1879, huber: 5.7121, swd: 25.6527, ept: 59.2545
      Epoch 6 composite train-obj: 5.676589
            Val objective improved 5.8156 → 5.8053, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 66.1309, mae: 6.1516, huber: 5.6758, swd: 25.4866, ept: 65.9877
    Epoch [7/50], Val Losses: mse: 68.3303, mae: 6.3045, huber: 5.8270, swd: 26.3369, ept: 56.5411
    Epoch [7/50], Test Losses: mse: 67.6822, mae: 6.2004, huber: 5.7238, swd: 26.3908, ept: 57.7045
      Epoch 7 composite train-obj: 5.675773
            No improvement (5.8270), counter 1/5
    Epoch [8/50], Train Losses: mse: 66.1426, mae: 6.1531, huber: 5.6773, swd: 25.4764, ept: 65.9606
    Epoch [8/50], Val Losses: mse: 68.1564, mae: 6.2798, huber: 5.8033, swd: 26.1574, ept: 57.9040
    Epoch [8/50], Test Losses: mse: 67.5209, mae: 6.1882, huber: 5.7123, swd: 26.3257, ept: 58.9177
      Epoch 8 composite train-obj: 5.677255
            Val objective improved 5.8053 → 5.8033, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 66.0939, mae: 6.1494, huber: 5.6737, swd: 25.4764, ept: 65.8942
    Epoch [9/50], Val Losses: mse: 69.0569, mae: 6.3268, huber: 5.8501, swd: 26.1333, ept: 56.6441
    Epoch [9/50], Test Losses: mse: 68.3143, mae: 6.2350, huber: 5.7585, swd: 26.4085, ept: 58.8968
      Epoch 9 composite train-obj: 5.673676
            No improvement (5.8501), counter 1/5
    Epoch [10/50], Train Losses: mse: 66.1036, mae: 6.1485, huber: 5.6729, swd: 25.3918, ept: 66.2335
    Epoch [10/50], Val Losses: mse: 68.2634, mae: 6.2947, huber: 5.8182, swd: 25.5174, ept: 57.4235
    Epoch [10/50], Test Losses: mse: 67.4440, mae: 6.1819, huber: 5.7063, swd: 25.6571, ept: 58.3074
      Epoch 10 composite train-obj: 5.672946
            No improvement (5.8182), counter 2/5
    Epoch [11/50], Train Losses: mse: 66.0738, mae: 6.1476, huber: 5.6720, swd: 25.4219, ept: 66.1398
    Epoch [11/50], Val Losses: mse: 67.9270, mae: 6.2657, huber: 5.7899, swd: 25.9443, ept: 58.0466
    Epoch [11/50], Test Losses: mse: 67.8063, mae: 6.1973, huber: 5.7218, swd: 25.9860, ept: 59.3769
      Epoch 11 composite train-obj: 5.672046
            Val objective improved 5.8033 → 5.7899, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 66.1312, mae: 6.1490, huber: 5.6735, swd: 25.4064, ept: 66.6804
    Epoch [12/50], Val Losses: mse: 68.1551, mae: 6.2909, huber: 5.8141, swd: 26.0737, ept: 57.6315
    Epoch [12/50], Test Losses: mse: 67.7703, mae: 6.2025, huber: 5.7266, swd: 26.0402, ept: 58.2550
      Epoch 12 composite train-obj: 5.673520
            No improvement (5.8141), counter 1/5
    Epoch [13/50], Train Losses: mse: 66.0156, mae: 6.1446, huber: 5.6691, swd: 25.4177, ept: 66.1697
    Epoch [13/50], Val Losses: mse: 68.5859, mae: 6.3072, huber: 5.8305, swd: 25.3165, ept: 59.0781
    Epoch [13/50], Test Losses: mse: 67.9194, mae: 6.2130, huber: 5.7372, swd: 25.4804, ept: 59.4710
      Epoch 13 composite train-obj: 5.669088
            No improvement (5.8305), counter 2/5
    Epoch [14/50], Train Losses: mse: 66.0986, mae: 6.1466, huber: 5.6711, swd: 25.3224, ept: 66.7619
    Epoch [14/50], Val Losses: mse: 68.3821, mae: 6.2964, huber: 5.8198, swd: 25.4513, ept: 56.5906
    Epoch [14/50], Test Losses: mse: 67.9411, mae: 6.2091, huber: 5.7333, swd: 25.6669, ept: 58.0336
      Epoch 14 composite train-obj: 5.671109
            No improvement (5.8198), counter 3/5
    Epoch [15/50], Train Losses: mse: 66.0697, mae: 6.1466, huber: 5.6711, swd: 25.3582, ept: 66.6643
    Epoch [15/50], Val Losses: mse: 68.7516, mae: 6.3089, huber: 5.8328, swd: 26.5911, ept: 56.4459
    Epoch [15/50], Test Losses: mse: 67.5144, mae: 6.1848, huber: 5.7091, swd: 26.6617, ept: 58.1759
      Epoch 15 composite train-obj: 5.671104
            No improvement (5.8328), counter 4/5
    Epoch [16/50], Train Losses: mse: 66.0694, mae: 6.1461, huber: 5.6707, swd: 25.3753, ept: 66.4813
    Epoch [16/50], Val Losses: mse: 67.4955, mae: 6.2550, huber: 5.7793, swd: 26.2596, ept: 55.4208
    Epoch [16/50], Test Losses: mse: 66.7110, mae: 6.1507, huber: 5.6755, swd: 26.3332, ept: 57.0977
      Epoch 16 composite train-obj: 5.670681
            Val objective improved 5.7899 → 5.7793, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 66.0894, mae: 6.1474, huber: 5.6720, swd: 25.3320, ept: 66.4313
    Epoch [17/50], Val Losses: mse: 68.1293, mae: 6.2639, huber: 5.7881, swd: 25.2347, ept: 60.0052
    Epoch [17/50], Test Losses: mse: 67.7510, mae: 6.1867, huber: 5.7115, swd: 25.3600, ept: 60.5554
      Epoch 17 composite train-obj: 5.671997
            No improvement (5.7881), counter 1/5
    Epoch [18/50], Train Losses: mse: 66.0081, mae: 6.1408, huber: 5.6656, swd: 25.3565, ept: 66.9390
    Epoch [18/50], Val Losses: mse: 68.2790, mae: 6.2842, huber: 5.8078, swd: 25.0689, ept: 57.7441
    Epoch [18/50], Test Losses: mse: 67.8729, mae: 6.2006, huber: 5.7249, swd: 25.2098, ept: 59.4961
      Epoch 18 composite train-obj: 5.665583
            No improvement (5.8078), counter 2/5
    Epoch [19/50], Train Losses: mse: 66.0206, mae: 6.1397, huber: 5.6644, swd: 25.2745, ept: 66.9795
    Epoch [19/50], Val Losses: mse: 68.3436, mae: 6.3000, huber: 5.8236, swd: 25.0221, ept: 57.2081
    Epoch [19/50], Test Losses: mse: 67.4662, mae: 6.1915, huber: 5.7156, swd: 25.1917, ept: 59.3083
      Epoch 19 composite train-obj: 5.664443
            No improvement (5.8236), counter 3/5
    Epoch [20/50], Train Losses: mse: 65.9865, mae: 6.1392, huber: 5.6640, swd: 25.3183, ept: 66.9949
    Epoch [20/50], Val Losses: mse: 68.4652, mae: 6.3068, huber: 5.8305, swd: 25.6441, ept: 55.8123
    Epoch [20/50], Test Losses: mse: 67.7990, mae: 6.2104, huber: 5.7348, swd: 25.7690, ept: 58.1909
      Epoch 20 composite train-obj: 5.664002
            No improvement (5.8305), counter 4/5
    Epoch [21/50], Train Losses: mse: 66.0267, mae: 6.1443, huber: 5.6689, swd: 25.3279, ept: 66.6181
    Epoch [21/50], Val Losses: mse: 68.3948, mae: 6.2998, huber: 5.8237, swd: 25.8125, ept: 57.1056
    Epoch [21/50], Test Losses: mse: 67.6351, mae: 6.1888, huber: 5.7134, swd: 25.9447, ept: 58.7623
      Epoch 21 composite train-obj: 5.668949
    Epoch [21/50], Test Losses: mse: 66.7110, mae: 6.1507, huber: 5.6755, swd: 26.3332, ept: 57.0977
    Best round's Test MSE: 66.7110, MAE: 6.1507, SWD: 26.3332
    Best round's Validation MSE: 67.4955, MAE: 6.2550, SWD: 26.2596
    Best round's Test verification MSE : 66.7110, MAE: 6.1507, SWD: 26.3332
    Time taken: 54.11 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.5305, mae: 6.3764, huber: 5.8979, swd: 28.3538, ept: 47.8001
    Epoch [1/50], Val Losses: mse: 68.1214, mae: 6.2883, huber: 5.8106, swd: 28.0574, ept: 51.3614
    Epoch [1/50], Test Losses: mse: 67.5713, mae: 6.2004, huber: 5.7238, swd: 27.8242, ept: 54.0974
      Epoch 1 composite train-obj: 5.897854
            Val objective improved inf → 5.8106, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 66.2730, mae: 6.1743, huber: 5.6975, swd: 26.9650, ept: 62.4767
    Epoch [2/50], Val Losses: mse: 68.0820, mae: 6.2940, huber: 5.8169, swd: 26.4811, ept: 52.9565
    Epoch [2/50], Test Losses: mse: 67.8400, mae: 6.2210, huber: 5.7444, swd: 26.2394, ept: 55.1970
      Epoch 2 composite train-obj: 5.697486
            No improvement (5.8169), counter 1/5
    Epoch [3/50], Train Losses: mse: 66.1688, mae: 6.1631, huber: 5.6867, swd: 26.7231, ept: 64.0924
    Epoch [3/50], Val Losses: mse: 68.0166, mae: 6.2698, huber: 5.7931, swd: 26.3879, ept: 57.7396
    Epoch [3/50], Test Losses: mse: 67.9112, mae: 6.2043, huber: 5.7282, swd: 25.9548, ept: 58.4190
      Epoch 3 composite train-obj: 5.686676
            Val objective improved 5.8106 → 5.7931, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 66.1782, mae: 6.1586, huber: 5.6824, swd: 26.3658, ept: 64.9971
    Epoch [4/50], Val Losses: mse: 68.8611, mae: 6.3326, huber: 5.8550, swd: 25.7235, ept: 57.2972
    Epoch [4/50], Test Losses: mse: 68.1496, mae: 6.2235, huber: 5.7472, swd: 25.4046, ept: 58.4520
      Epoch 4 composite train-obj: 5.682411
            No improvement (5.8550), counter 1/5
    Epoch [5/50], Train Losses: mse: 66.1695, mae: 6.1586, huber: 5.6825, swd: 26.3527, ept: 65.4766
    Epoch [5/50], Val Losses: mse: 69.3040, mae: 6.3500, huber: 5.8729, swd: 27.0361, ept: 57.0790
    Epoch [5/50], Test Losses: mse: 68.3392, mae: 6.2294, huber: 5.7533, swd: 26.7038, ept: 58.0797
      Epoch 5 composite train-obj: 5.682542
            No improvement (5.8729), counter 2/5
    Epoch [6/50], Train Losses: mse: 66.1076, mae: 6.1536, huber: 5.6776, swd: 26.2816, ept: 65.2639
    Epoch [6/50], Val Losses: mse: 68.0115, mae: 6.2793, huber: 5.8024, swd: 26.7194, ept: 56.6941
    Epoch [6/50], Test Losses: mse: 67.4089, mae: 6.1824, huber: 5.7066, swd: 26.3828, ept: 57.8694
      Epoch 6 composite train-obj: 5.677643
            No improvement (5.8024), counter 3/5
    Epoch [7/50], Train Losses: mse: 66.1634, mae: 6.1559, huber: 5.6799, swd: 26.2259, ept: 65.6264
    Epoch [7/50], Val Losses: mse: 68.1936, mae: 6.2861, huber: 5.8097, swd: 27.2656, ept: 54.7528
    Epoch [7/50], Test Losses: mse: 67.7291, mae: 6.2053, huber: 5.7292, swd: 26.8076, ept: 56.7607
      Epoch 7 composite train-obj: 5.679935
            No improvement (5.8097), counter 4/5
    Epoch [8/50], Train Losses: mse: 66.1361, mae: 6.1532, huber: 5.6774, swd: 26.1099, ept: 65.6670
    Epoch [8/50], Val Losses: mse: 68.1074, mae: 6.2836, huber: 5.8071, swd: 27.0559, ept: 56.7350
    Epoch [8/50], Test Losses: mse: 67.6208, mae: 6.1887, huber: 5.7130, swd: 26.6209, ept: 58.0855
      Epoch 8 composite train-obj: 5.677377
    Epoch [8/50], Test Losses: mse: 67.9112, mae: 6.2043, huber: 5.7282, swd: 25.9548, ept: 58.4190
    Best round's Test MSE: 67.9112, MAE: 6.2043, SWD: 25.9548
    Best round's Validation MSE: 68.0166, MAE: 6.2698, SWD: 26.3879
    Best round's Test verification MSE : 67.9112, MAE: 6.2043, SWD: 25.9548
    Time taken: 21.49 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq336_pred720_20250514_0144)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 67.3741 ± 0.4980
      mae: 6.1861 ± 0.0251
      huber: 5.7101 ± 0.0244
      swd: 26.4637 ± 0.4778
      ept: 57.3006 ± 0.8426
      count: 33.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 67.9014 ± 0.2959
      mae: 6.2735 ± 0.0168
      huber: 5.7969 ± 0.0161
      swd: 26.6167 ± 0.4176
      ept: 55.9089 ± 1.3407
      count: 33.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 97.45 seconds
    
    Experiment complete: DLinear_lorenz_seq336_pred720_20250514_0144
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    
