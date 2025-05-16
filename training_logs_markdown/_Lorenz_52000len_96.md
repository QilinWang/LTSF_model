# Data



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
    




    <data_manager.DatasetManager at 0x21731346420>



# Exp



## seq=96

### EigenACL

#### 96-96
##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 283
    Validation Batches: 40
    Test Batches: 80
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 30.1868, mae: 3.2345, huber: 2.7987, swd: 14.4878, ept: 68.0599
    Epoch [1/50], Val Losses: mse: 8.0001, mae: 1.4701, huber: 1.0827, swd: 1.3315, ept: 90.1583
    Epoch [1/50], Test Losses: mse: 7.0033, mae: 1.4066, huber: 1.0203, swd: 1.1316, ept: 90.8744
      Epoch 1 composite train-obj: 2.798732
            Val objective improved inf → 1.0827, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.6810, mae: 1.2665, huber: 0.8936, swd: 1.2441, ept: 91.5248
    Epoch [2/50], Val Losses: mse: 4.3790, mae: 1.0806, huber: 0.7199, swd: 0.9187, ept: 92.7460
    Epoch [2/50], Test Losses: mse: 4.0221, mae: 1.0529, huber: 0.6930, swd: 0.7443, ept: 93.3204
      Epoch 2 composite train-obj: 0.893613
            Val objective improved 1.0827 → 0.7199, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7798, mae: 0.9926, huber: 0.6479, swd: 0.8529, ept: 93.2198
    Epoch [3/50], Val Losses: mse: 3.2918, mae: 0.8833, huber: 0.5445, swd: 0.5441, ept: 93.9899
    Epoch [3/50], Test Losses: mse: 2.6910, mae: 0.8503, huber: 0.5114, swd: 0.5076, ept: 94.3895
      Epoch 3 composite train-obj: 0.647927
            Val objective improved 0.7199 → 0.5445, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.0079, mae: 0.8759, huber: 0.5471, swd: 0.6991, ept: 93.7433
    Epoch [4/50], Val Losses: mse: 4.3533, mae: 0.9750, huber: 0.6371, swd: 0.8149, ept: 92.9275
    Epoch [4/50], Test Losses: mse: 4.0213, mae: 0.9339, huber: 0.6012, swd: 0.7325, ept: 93.1211
      Epoch 4 composite train-obj: 0.547064
            No improvement (0.6371), counter 1/5
    Epoch [5/50], Train Losses: mse: 3.4818, mae: 0.7953, huber: 0.4795, swd: 0.6007, ept: 94.0843
    Epoch [5/50], Val Losses: mse: 2.5640, mae: 0.7148, huber: 0.4154, swd: 0.4048, ept: 93.7176
    Epoch [5/50], Test Losses: mse: 2.1970, mae: 0.7045, huber: 0.4035, swd: 0.3257, ept: 93.8469
      Epoch 5 composite train-obj: 0.479534
            Val objective improved 0.5445 → 0.4154, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.4804, mae: 0.7857, huber: 0.4731, swd: 0.5831, ept: 94.1142
    Epoch [6/50], Val Losses: mse: 1.8805, mae: 0.6785, huber: 0.3741, swd: 0.4129, ept: 94.5081
    Epoch [6/50], Test Losses: mse: 1.7317, mae: 0.6639, huber: 0.3604, swd: 0.3432, ept: 94.7688
      Epoch 6 composite train-obj: 0.473143
            Val objective improved 0.4154 → 0.3741, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.0979, mae: 0.7374, huber: 0.4332, swd: 0.5335, ept: 94.3138
    Epoch [7/50], Val Losses: mse: 1.7041, mae: 0.6297, huber: 0.3392, swd: 0.4629, ept: 94.5788
    Epoch [7/50], Test Losses: mse: 1.5833, mae: 0.6178, huber: 0.3296, swd: 0.3840, ept: 94.5750
      Epoch 7 composite train-obj: 0.433215
            Val objective improved 0.3741 → 0.3392, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.0467, mae: 0.7234, huber: 0.4229, swd: 0.4949, ept: 94.3217
    Epoch [8/50], Val Losses: mse: 2.6368, mae: 0.7933, huber: 0.4774, swd: 0.4873, ept: 94.1277
    Epoch [8/50], Test Losses: mse: 2.0332, mae: 0.7579, huber: 0.4419, swd: 0.4114, ept: 94.3097
      Epoch 8 composite train-obj: 0.422854
            No improvement (0.4774), counter 1/5
    Epoch [9/50], Train Losses: mse: 2.6972, mae: 0.6758, huber: 0.3830, swd: 0.4566, ept: 94.5212
    Epoch [9/50], Val Losses: mse: 3.2216, mae: 0.7878, huber: 0.4723, swd: 0.5248, ept: 94.1503
    Epoch [9/50], Test Losses: mse: 3.0456, mae: 0.7555, huber: 0.4448, swd: 0.5346, ept: 94.3830
      Epoch 9 composite train-obj: 0.383023
            No improvement (0.4723), counter 2/5
    Epoch [10/50], Train Losses: mse: 2.7296, mae: 0.6641, huber: 0.3757, swd: 0.4403, ept: 94.5444
    Epoch [10/50], Val Losses: mse: 1.7218, mae: 0.5719, huber: 0.3022, swd: 0.3176, ept: 94.6714
    Epoch [10/50], Test Losses: mse: 1.3576, mae: 0.5543, huber: 0.2857, swd: 0.2708, ept: 94.7841
      Epoch 10 composite train-obj: 0.375713
            Val objective improved 0.3392 → 0.3022, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 2.4532, mae: 0.6179, huber: 0.3396, swd: 0.3896, ept: 94.6599
    Epoch [11/50], Val Losses: mse: 1.8896, mae: 0.6290, huber: 0.3465, swd: 0.3312, ept: 94.5104
    Epoch [11/50], Test Losses: mse: 1.8278, mae: 0.6198, huber: 0.3374, swd: 0.2762, ept: 94.6020
      Epoch 11 composite train-obj: 0.339641
            No improvement (0.3465), counter 1/5
    Epoch [12/50], Train Losses: mse: 2.3070, mae: 0.6035, huber: 0.3274, swd: 0.3733, ept: 94.7365
    Epoch [12/50], Val Losses: mse: 1.8156, mae: 0.6150, huber: 0.3350, swd: 0.4000, ept: 94.8178
    Epoch [12/50], Test Losses: mse: 1.5223, mae: 0.5813, huber: 0.3070, swd: 0.3243, ept: 95.0186
      Epoch 12 composite train-obj: 0.327374
            No improvement (0.3350), counter 2/5
    Epoch [13/50], Train Losses: mse: 2.3755, mae: 0.6095, huber: 0.3331, swd: 0.3876, ept: 94.7096
    Epoch [13/50], Val Losses: mse: 2.7737, mae: 0.7907, huber: 0.4683, swd: 0.5614, ept: 94.2262
    Epoch [13/50], Test Losses: mse: 2.4664, mae: 0.7635, huber: 0.4446, swd: 0.4663, ept: 94.4787
      Epoch 13 composite train-obj: 0.333106
            No improvement (0.4683), counter 3/5
    Epoch [14/50], Train Losses: mse: 2.3603, mae: 0.6048, huber: 0.3295, swd: 0.3684, ept: 94.7306
    Epoch [14/50], Val Losses: mse: 1.8148, mae: 0.5788, huber: 0.3071, swd: 0.4109, ept: 94.5105
    Epoch [14/50], Test Losses: mse: 1.4287, mae: 0.5416, huber: 0.2734, swd: 0.2700, ept: 94.8825
      Epoch 14 composite train-obj: 0.329533
            No improvement (0.3071), counter 4/5
    Epoch [15/50], Train Losses: mse: 2.2196, mae: 0.5840, huber: 0.3138, swd: 0.3650, ept: 94.7781
    Epoch [15/50], Val Losses: mse: 1.5900, mae: 0.4961, huber: 0.2461, swd: 0.3673, ept: 94.9870
    Epoch [15/50], Test Losses: mse: 1.3133, mae: 0.4857, huber: 0.2357, swd: 0.2622, ept: 95.2185
      Epoch 15 composite train-obj: 0.313814
            Val objective improved 0.3022 → 0.2461, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 2.2074, mae: 0.5791, huber: 0.3113, swd: 0.3562, ept: 94.7657
    Epoch [16/50], Val Losses: mse: 1.7180, mae: 0.5583, huber: 0.2928, swd: 0.3531, ept: 94.8272
    Epoch [16/50], Test Losses: mse: 1.3007, mae: 0.5305, huber: 0.2665, swd: 0.2802, ept: 95.0404
      Epoch 16 composite train-obj: 0.311274
            No improvement (0.2928), counter 1/5
    Epoch [17/50], Train Losses: mse: 2.1660, mae: 0.5678, huber: 0.3027, swd: 0.3428, ept: 94.8230
    Epoch [17/50], Val Losses: mse: 2.0238, mae: 0.5679, huber: 0.3034, swd: 0.4024, ept: 94.3734
    Epoch [17/50], Test Losses: mse: 1.4248, mae: 0.5232, huber: 0.2629, swd: 0.2863, ept: 94.7577
      Epoch 17 composite train-obj: 0.302686
            No improvement (0.3034), counter 2/5
    Epoch [18/50], Train Losses: mse: 2.1624, mae: 0.5700, huber: 0.3037, swd: 0.3405, ept: 94.7945
    Epoch [18/50], Val Losses: mse: 1.8767, mae: 0.5818, huber: 0.3095, swd: 0.2792, ept: 94.8620
    Epoch [18/50], Test Losses: mse: 1.4861, mae: 0.5483, huber: 0.2807, swd: 0.2755, ept: 95.0971
      Epoch 18 composite train-obj: 0.303739
            No improvement (0.3095), counter 3/5
    Epoch [19/50], Train Losses: mse: 2.1272, mae: 0.5561, huber: 0.2943, swd: 0.3360, ept: 94.8612
    Epoch [19/50], Val Losses: mse: 1.9854, mae: 0.6079, huber: 0.3290, swd: 0.4387, ept: 94.6782
    Epoch [19/50], Test Losses: mse: 1.8389, mae: 0.5905, huber: 0.3161, swd: 0.3319, ept: 94.7609
      Epoch 19 composite train-obj: 0.294252
            No improvement (0.3290), counter 4/5
    Epoch [20/50], Train Losses: mse: 2.1307, mae: 0.5538, huber: 0.2932, swd: 0.3300, ept: 94.8611
    Epoch [20/50], Val Losses: mse: 1.7765, mae: 0.6090, huber: 0.3292, swd: 0.3368, ept: 94.8598
    Epoch [20/50], Test Losses: mse: 1.5181, mae: 0.6032, huber: 0.3204, swd: 0.2521, ept: 94.9827
      Epoch 20 composite train-obj: 0.293194
    Epoch [20/50], Test Losses: mse: 1.3133, mae: 0.4857, huber: 0.2357, swd: 0.2622, ept: 95.2185
    Best round's Test MSE: 1.3133, MAE: 0.4857, SWD: 0.2622
    Best round's Validation MSE: 1.5900, MAE: 0.4961, SWD: 0.3673
    Best round's Test verification MSE : 1.3133, MAE: 0.4857, SWD: 0.2622
    Time taken: 162.61 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 30.2911, mae: 3.2303, huber: 2.7964, swd: 15.3338, ept: 66.0736
    Epoch [1/50], Val Losses: mse: 8.5180, mae: 1.5464, huber: 1.1535, swd: 1.8750, ept: 86.7798
    Epoch [1/50], Test Losses: mse: 7.5647, mae: 1.4722, huber: 1.0821, swd: 1.5926, ept: 87.4122
      Epoch 1 composite train-obj: 2.796407
            Val objective improved inf → 1.1535, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.5088, mae: 1.2773, huber: 0.9014, swd: 1.4513, ept: 90.9648
    Epoch [2/50], Val Losses: mse: 5.4463, mae: 1.1746, huber: 0.8105, swd: 1.0726, ept: 91.4923
    Epoch [2/50], Test Losses: mse: 4.9234, mae: 1.1632, huber: 0.7967, swd: 1.0858, ept: 91.8297
      Epoch 2 composite train-obj: 0.901440
            Val objective improved 1.1535 → 0.8105, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.7444, mae: 1.0064, huber: 0.6603, swd: 0.9755, ept: 92.9109
    Epoch [3/50], Val Losses: mse: 3.1597, mae: 0.8805, huber: 0.5435, swd: 0.7534, ept: 93.3858
    Epoch [3/50], Test Losses: mse: 3.0439, mae: 0.8589, huber: 0.5230, swd: 0.6722, ept: 93.9736
      Epoch 3 composite train-obj: 0.660276
            Val objective improved 0.8105 → 0.5435, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.4865, mae: 0.9641, huber: 0.6234, swd: 0.9130, ept: 93.2174
    Epoch [4/50], Val Losses: mse: 3.7590, mae: 1.0399, huber: 0.6795, swd: 0.9103, ept: 93.3825
    Epoch [4/50], Test Losses: mse: 3.5250, mae: 1.0313, huber: 0.6699, swd: 0.8663, ept: 93.5728
      Epoch 4 composite train-obj: 0.623398
            No improvement (0.6795), counter 1/5
    Epoch [5/50], Train Losses: mse: 4.0901, mae: 0.9027, huber: 0.5713, swd: 0.8247, ept: 93.6137
    Epoch [5/50], Val Losses: mse: 2.3666, mae: 0.7765, huber: 0.4564, swd: 0.6287, ept: 94.1725
    Epoch [5/50], Test Losses: mse: 2.3404, mae: 0.7606, huber: 0.4426, swd: 0.4843, ept: 94.4388
      Epoch 5 composite train-obj: 0.571293
            Val objective improved 0.5435 → 0.4564, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.5443, mae: 0.8251, huber: 0.5049, swd: 0.7053, ept: 93.8866
    Epoch [6/50], Val Losses: mse: 2.3187, mae: 0.7508, huber: 0.4373, swd: 0.5215, ept: 94.1279
    Epoch [6/50], Test Losses: mse: 2.0264, mae: 0.7323, huber: 0.4186, swd: 0.4277, ept: 94.4323
      Epoch 6 composite train-obj: 0.504940
            Val objective improved 0.4564 → 0.4373, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.5711, mae: 0.8066, huber: 0.4911, swd: 0.7511, ept: 93.8495
    Epoch [7/50], Val Losses: mse: 2.3414, mae: 0.7714, huber: 0.4527, swd: 0.7856, ept: 94.0564
    Epoch [7/50], Test Losses: mse: 2.5355, mae: 0.7720, huber: 0.4562, swd: 0.8034, ept: 94.1808
      Epoch 7 composite train-obj: 0.491121
            No improvement (0.4527), counter 1/5
    Epoch [8/50], Train Losses: mse: 4.4266, mae: 0.8978, huber: 0.5740, swd: 0.9656, ept: 93.2004
    Epoch [8/50], Val Losses: mse: 5.1059, mae: 1.0364, huber: 0.6944, swd: 0.7945, ept: 92.4887
    Epoch [8/50], Test Losses: mse: 4.1911, mae: 1.0116, huber: 0.6691, swd: 0.8195, ept: 92.8147
      Epoch 8 composite train-obj: 0.574005
            No improvement (0.6944), counter 2/5
    Epoch [9/50], Train Losses: mse: 3.8097, mae: 0.8275, huber: 0.5121, swd: 0.7080, ept: 93.7095
    Epoch [9/50], Val Losses: mse: 2.2270, mae: 0.7120, huber: 0.4053, swd: 0.4947, ept: 94.1774
    Epoch [9/50], Test Losses: mse: 2.0376, mae: 0.6976, huber: 0.3895, swd: 0.3800, ept: 94.6240
      Epoch 9 composite train-obj: 0.512106
            Val objective improved 0.4373 → 0.4053, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 3.2740, mae: 0.7368, huber: 0.4367, swd: 0.6274, ept: 94.1146
    Epoch [10/50], Val Losses: mse: 2.5735, mae: 0.6941, huber: 0.3964, swd: 0.6177, ept: 94.4247
    Epoch [10/50], Test Losses: mse: 2.0761, mae: 0.6604, huber: 0.3641, swd: 0.4749, ept: 94.7582
      Epoch 10 composite train-obj: 0.436655
            Val objective improved 0.4053 → 0.3964, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 3.0971, mae: 0.7110, huber: 0.4164, swd: 0.5727, ept: 94.2537
    Epoch [11/50], Val Losses: mse: 2.5605, mae: 0.7600, huber: 0.4461, swd: 0.6295, ept: 93.9757
    Epoch [11/50], Test Losses: mse: 2.5585, mae: 0.7537, huber: 0.4424, swd: 0.6728, ept: 94.1572
      Epoch 11 composite train-obj: 0.416382
            No improvement (0.4461), counter 1/5
    Epoch [12/50], Train Losses: mse: 3.1150, mae: 0.7116, huber: 0.4168, swd: 0.5807, ept: 94.2493
    Epoch [12/50], Val Losses: mse: 2.9221, mae: 0.7493, huber: 0.4512, swd: 0.4234, ept: 93.9129
    Epoch [12/50], Test Losses: mse: 3.0121, mae: 0.7552, huber: 0.4557, swd: 0.4487, ept: 94.0897
      Epoch 12 composite train-obj: 0.416759
            No improvement (0.4512), counter 2/5
    Epoch [13/50], Train Losses: mse: 3.3730, mae: 0.7405, huber: 0.4438, swd: 0.6280, ept: 94.0588
    Epoch [13/50], Val Losses: mse: 2.5271, mae: 0.7844, huber: 0.4647, swd: 0.4924, ept: 94.0730
    Epoch [13/50], Test Losses: mse: 2.5226, mae: 0.7736, huber: 0.4532, swd: 0.5652, ept: 94.2919
      Epoch 13 composite train-obj: 0.443793
            No improvement (0.4647), counter 3/5
    Epoch [14/50], Train Losses: mse: 3.1047, mae: 0.7120, huber: 0.4179, swd: 0.6034, ept: 94.2077
    Epoch [14/50], Val Losses: mse: 2.4323, mae: 0.6689, huber: 0.3846, swd: 0.4936, ept: 94.2128
    Epoch [14/50], Test Losses: mse: 2.1136, mae: 0.6485, huber: 0.3648, swd: 0.4029, ept: 94.5441
      Epoch 14 composite train-obj: 0.417853
            Val objective improved 0.3964 → 0.3846, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 3.3274, mae: 0.7405, huber: 0.4398, swd: 0.6034, ept: 94.1744
    Epoch [15/50], Val Losses: mse: 1.9804, mae: 0.6937, huber: 0.3894, swd: 0.5081, ept: 94.7003
    Epoch [15/50], Test Losses: mse: 1.7527, mae: 0.6580, huber: 0.3593, swd: 0.3895, ept: 94.9648
      Epoch 15 composite train-obj: 0.439756
            No improvement (0.3894), counter 1/5
    Epoch [16/50], Train Losses: mse: 3.1534, mae: 0.7147, huber: 0.4217, swd: 0.6861, ept: 94.0876
    Epoch [16/50], Val Losses: mse: 1.6723, mae: 0.6080, huber: 0.3215, swd: 0.3297, ept: 94.6876
    Epoch [16/50], Test Losses: mse: 1.5196, mae: 0.5895, huber: 0.3060, swd: 0.3373, ept: 94.9254
      Epoch 16 composite train-obj: 0.421676
            Val objective improved 0.3846 → 0.3215, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 2.7326, mae: 0.6509, huber: 0.3680, swd: 0.5427, ept: 94.4480
    Epoch [17/50], Val Losses: mse: 1.1998, mae: 0.4840, huber: 0.2292, swd: 0.4193, ept: 95.2206
    Epoch [17/50], Test Losses: mse: 0.9302, mae: 0.4597, huber: 0.2061, swd: 0.2483, ept: 95.4065
      Epoch 17 composite train-obj: 0.367995
            Val objective improved 0.3215 → 0.2292, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 2.4945, mae: 0.6185, huber: 0.3426, swd: 0.5123, ept: 94.5856
    Epoch [18/50], Val Losses: mse: 1.5272, mae: 0.5866, huber: 0.3061, swd: 0.4703, ept: 94.8181
    Epoch [18/50], Test Losses: mse: 1.4770, mae: 0.5716, huber: 0.2933, swd: 0.3817, ept: 95.0119
      Epoch 18 composite train-obj: 0.342610
            No improvement (0.3061), counter 1/5
    Epoch [19/50], Train Losses: mse: 3.2022, mae: 0.7072, huber: 0.4173, swd: 0.7713, ept: 94.1574
    Epoch [19/50], Val Losses: mse: 2.0472, mae: 0.6314, huber: 0.3428, swd: 0.6206, ept: 94.6163
    Epoch [19/50], Test Losses: mse: 1.7199, mae: 0.6085, huber: 0.3201, swd: 0.5518, ept: 94.8186
      Epoch 19 composite train-obj: 0.417274
            No improvement (0.3428), counter 2/5
    Epoch [20/50], Train Losses: mse: 3.0627, mae: 0.6848, huber: 0.3988, swd: 0.7217, ept: 94.3049
    Epoch [20/50], Val Losses: mse: 2.0856, mae: 0.6197, huber: 0.3429, swd: 0.4672, ept: 94.5299
    Epoch [20/50], Test Losses: mse: 1.6823, mae: 0.5886, huber: 0.3139, swd: 0.4067, ept: 94.8380
      Epoch 20 composite train-obj: 0.398761
            No improvement (0.3429), counter 3/5
    Epoch [21/50], Train Losses: mse: 3.8713, mae: 0.7778, huber: 0.4828, swd: 0.9098, ept: 93.6834
    Epoch [21/50], Val Losses: mse: 2.7827, mae: 0.7218, huber: 0.4233, swd: 0.3835, ept: 94.0489
    Epoch [21/50], Test Losses: mse: 2.5753, mae: 0.7149, huber: 0.4157, swd: 0.3571, ept: 94.0859
      Epoch 21 composite train-obj: 0.482812
            No improvement (0.4233), counter 4/5
    Epoch [22/50], Train Losses: mse: 3.6755, mae: 0.7406, huber: 0.4504, swd: 0.6701, ept: 93.8543
    Epoch [22/50], Val Losses: mse: 3.4360, mae: 0.8702, huber: 0.5471, swd: 0.7440, ept: 93.2700
    Epoch [22/50], Test Losses: mse: 3.3383, mae: 0.8713, huber: 0.5458, swd: 0.7498, ept: 93.5848
      Epoch 22 composite train-obj: 0.450386
    Epoch [22/50], Test Losses: mse: 0.9302, mae: 0.4597, huber: 0.2061, swd: 0.2483, ept: 95.4065
    Best round's Test MSE: 0.9302, MAE: 0.4597, SWD: 0.2483
    Best round's Validation MSE: 1.1998, MAE: 0.4840, SWD: 0.4193
    Best round's Test verification MSE : 0.9302, MAE: 0.4597, SWD: 0.2483
    Time taken: 180.79 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 31.1459, mae: 3.2698, huber: 2.8354, swd: 14.5832, ept: 65.1906
    Epoch [1/50], Val Losses: mse: 8.0476, mae: 1.5404, huber: 1.1466, swd: 1.5872, ept: 85.9749
    Epoch [1/50], Test Losses: mse: 7.4094, mae: 1.4935, huber: 1.1010, swd: 1.5210, ept: 86.9582
      Epoch 1 composite train-obj: 2.835351
            Val objective improved inf → 1.1466, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.1327, mae: 1.3012, huber: 0.9272, swd: 1.2562, ept: 89.3923
    Epoch [2/50], Val Losses: mse: 6.3116, mae: 1.2385, huber: 0.8699, swd: 1.1311, ept: 90.1573
    Epoch [2/50], Test Losses: mse: 5.6790, mae: 1.2208, huber: 0.8483, swd: 1.0405, ept: 90.3785
      Epoch 2 composite train-obj: 0.927233
            Val objective improved 1.1466 → 0.8699, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.3580, mae: 1.0473, huber: 0.7002, swd: 0.8776, ept: 91.6847
    Epoch [3/50], Val Losses: mse: 3.9297, mae: 0.8875, huber: 0.5585, swd: 0.8051, ept: 92.5613
    Epoch [3/50], Test Losses: mse: 3.3596, mae: 0.8488, huber: 0.5218, swd: 0.5957, ept: 92.9351
      Epoch 3 composite train-obj: 0.700231
            Val objective improved 0.8699 → 0.5585, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.4246, mae: 1.0334, huber: 0.6903, swd: 0.8486, ept: 92.4084
    Epoch [4/50], Val Losses: mse: 2.7956, mae: 0.7903, huber: 0.4705, swd: 0.5113, ept: 93.8902
    Epoch [4/50], Test Losses: mse: 2.5875, mae: 0.7706, huber: 0.4529, swd: 0.4671, ept: 94.2869
      Epoch 4 composite train-obj: 0.690286
            Val objective improved 0.5585 → 0.4705, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.2656, mae: 0.8838, huber: 0.5612, swd: 0.7125, ept: 93.0594
    Epoch [5/50], Val Losses: mse: 3.6233, mae: 0.8918, huber: 0.5641, swd: 0.6231, ept: 92.7609
    Epoch [5/50], Test Losses: mse: 3.2608, mae: 0.8661, huber: 0.5392, swd: 0.5245, ept: 92.9200
      Epoch 5 composite train-obj: 0.561229
            No improvement (0.5641), counter 1/5
    Epoch [6/50], Train Losses: mse: 3.8518, mae: 0.8371, huber: 0.5202, swd: 0.6149, ept: 93.4321
    Epoch [6/50], Val Losses: mse: 5.5212, mae: 1.0928, huber: 0.7467, swd: 1.0159, ept: 91.7410
    Epoch [6/50], Test Losses: mse: 4.6054, mae: 1.0224, huber: 0.6807, swd: 0.8116, ept: 92.3390
      Epoch 6 composite train-obj: 0.520211
            No improvement (0.7467), counter 2/5
    Epoch [7/50], Train Losses: mse: 4.8260, mae: 0.9343, huber: 0.6090, swd: 0.8677, ept: 92.7229
    Epoch [7/50], Val Losses: mse: 4.2095, mae: 0.9152, huber: 0.5968, swd: 1.0890, ept: 92.2458
    Epoch [7/50], Test Losses: mse: 3.8144, mae: 0.8853, huber: 0.5693, swd: 1.0391, ept: 92.2972
      Epoch 7 composite train-obj: 0.609045
            No improvement (0.5968), counter 3/5
    Epoch [8/50], Train Losses: mse: 3.3982, mae: 0.7796, huber: 0.4724, swd: 0.5863, ept: 93.8245
    Epoch [8/50], Val Losses: mse: 2.7225, mae: 0.7361, huber: 0.4409, swd: 0.4599, ept: 93.6871
    Epoch [8/50], Test Losses: mse: 2.1711, mae: 0.7042, huber: 0.4118, swd: 0.3863, ept: 93.9019
      Epoch 8 composite train-obj: 0.472382
            Val objective improved 0.4705 → 0.4409, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.1024, mae: 0.7231, huber: 0.4256, swd: 0.4943, ept: 94.1967
    Epoch [9/50], Val Losses: mse: 1.7558, mae: 0.6281, huber: 0.3393, swd: 0.3484, ept: 94.6292
    Epoch [9/50], Test Losses: mse: 1.7510, mae: 0.6147, huber: 0.3282, swd: 0.3513, ept: 94.8123
      Epoch 9 composite train-obj: 0.425636
            Val objective improved 0.4409 → 0.3393, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 3.0530, mae: 0.7100, huber: 0.4159, swd: 0.5038, ept: 94.3227
    Epoch [10/50], Val Losses: mse: 2.7704, mae: 0.7219, huber: 0.4370, swd: 0.4751, ept: 93.6382
    Epoch [10/50], Test Losses: mse: 2.8216, mae: 0.7229, huber: 0.4374, swd: 0.4319, ept: 93.5706
      Epoch 10 composite train-obj: 0.415851
            No improvement (0.4370), counter 1/5
    Epoch [11/50], Train Losses: mse: 3.3079, mae: 0.7316, huber: 0.4362, swd: 0.5227, ept: 94.1455
    Epoch [11/50], Val Losses: mse: 1.8271, mae: 0.6236, huber: 0.3391, swd: 0.4185, ept: 94.5626
    Epoch [11/50], Test Losses: mse: 1.7809, mae: 0.6127, huber: 0.3295, swd: 0.3913, ept: 94.7245
      Epoch 11 composite train-obj: 0.436178
            Val objective improved 0.3393 → 0.3391, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 3.2659, mae: 0.7349, huber: 0.4371, swd: 0.5623, ept: 94.1781
    Epoch [12/50], Val Losses: mse: 2.0472, mae: 0.6884, huber: 0.3853, swd: 0.4218, ept: 94.5383
    Epoch [12/50], Test Losses: mse: 1.7209, mae: 0.6668, huber: 0.3640, swd: 0.3452, ept: 94.7832
      Epoch 12 composite train-obj: 0.437051
            No improvement (0.3853), counter 1/5
    Epoch [13/50], Train Losses: mse: 3.2292, mae: 0.7178, huber: 0.4253, swd: 0.5272, ept: 94.2021
    Epoch [13/50], Val Losses: mse: 2.4217, mae: 0.6860, huber: 0.3896, swd: 0.6546, ept: 94.1994
    Epoch [13/50], Test Losses: mse: 2.1549, mae: 0.6665, huber: 0.3697, swd: 0.5295, ept: 94.5181
      Epoch 13 composite train-obj: 0.425278
            No improvement (0.3896), counter 2/5
    Epoch [14/50], Train Losses: mse: 2.6736, mae: 0.6390, huber: 0.3600, swd: 0.4612, ept: 94.5334
    Epoch [14/50], Val Losses: mse: 3.2239, mae: 0.6807, huber: 0.3964, swd: 0.4872, ept: 94.2669
    Epoch [14/50], Test Losses: mse: 2.5930, mae: 0.6494, huber: 0.3660, swd: 0.4070, ept: 94.5125
      Epoch 14 composite train-obj: 0.360023
            No improvement (0.3964), counter 3/5
    Epoch [15/50], Train Losses: mse: 2.7700, mae: 0.6461, huber: 0.3668, swd: 0.4477, ept: 94.4827
    Epoch [15/50], Val Losses: mse: 3.3543, mae: 0.8286, huber: 0.5122, swd: 0.8144, ept: 93.1721
    Epoch [15/50], Test Losses: mse: 3.0875, mae: 0.8053, huber: 0.4909, swd: 0.7272, ept: 93.6415
      Epoch 15 composite train-obj: 0.366770
            No improvement (0.5122), counter 4/5
    Epoch [16/50], Train Losses: mse: 2.5760, mae: 0.6401, huber: 0.3588, swd: 0.4892, ept: 94.4843
    Epoch [16/50], Val Losses: mse: 1.7866, mae: 0.5796, huber: 0.3110, swd: 0.5157, ept: 94.5219
    Epoch [16/50], Test Losses: mse: 1.6961, mae: 0.5519, huber: 0.2867, swd: 0.4922, ept: 94.7404
      Epoch 16 composite train-obj: 0.358839
            Val objective improved 0.3391 → 0.3110, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 2.7956, mae: 0.6684, huber: 0.3830, swd: 0.4634, ept: 94.4830
    Epoch [17/50], Val Losses: mse: 2.1717, mae: 0.6158, huber: 0.3409, swd: 0.4758, ept: 94.4554
    Epoch [17/50], Test Losses: mse: 1.7421, mae: 0.5931, huber: 0.3187, swd: 0.3010, ept: 94.6301
      Epoch 17 composite train-obj: 0.382956
            No improvement (0.3409), counter 1/5
    Epoch [18/50], Train Losses: mse: 2.4338, mae: 0.6258, huber: 0.3471, swd: 0.4104, ept: 94.6485
    Epoch [18/50], Val Losses: mse: 2.0126, mae: 0.6183, huber: 0.3424, swd: 0.4596, ept: 94.3173
    Epoch [18/50], Test Losses: mse: 1.9568, mae: 0.6133, huber: 0.3371, swd: 0.4256, ept: 94.5133
      Epoch 18 composite train-obj: 0.347089
            No improvement (0.3424), counter 2/5
    Epoch [19/50], Train Losses: mse: 2.0293, mae: 0.5521, huber: 0.2900, swd: 0.3349, ept: 94.8999
    Epoch [19/50], Val Losses: mse: 1.5955, mae: 0.5322, huber: 0.2662, swd: 0.3713, ept: 94.9874
    Epoch [19/50], Test Losses: mse: 1.4001, mae: 0.5180, huber: 0.2516, swd: 0.3560, ept: 95.1292
      Epoch 19 composite train-obj: 0.289993
            Val objective improved 0.3110 → 0.2662, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 2.2147, mae: 0.5619, huber: 0.3002, swd: 0.3391, ept: 94.8140
    Epoch [20/50], Val Losses: mse: 1.6683, mae: 0.5192, huber: 0.2675, swd: 0.4739, ept: 94.9267
    Epoch [20/50], Test Losses: mse: 1.5352, mae: 0.5009, huber: 0.2496, swd: 0.3954, ept: 95.1434
      Epoch 20 composite train-obj: 0.300198
            No improvement (0.2675), counter 1/5
    Epoch [21/50], Train Losses: mse: 2.1747, mae: 0.5803, huber: 0.3125, swd: 0.3721, ept: 94.7858
    Epoch [21/50], Val Losses: mse: 1.0267, mae: 0.4748, huber: 0.2173, swd: 0.2530, ept: 95.4924
    Epoch [21/50], Test Losses: mse: 0.7902, mae: 0.4492, huber: 0.1974, swd: 0.2157, ept: 95.5268
      Epoch 21 composite train-obj: 0.312517
            Val objective improved 0.2662 → 0.2173, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 2.4413, mae: 0.6070, huber: 0.3351, swd: 0.4110, ept: 94.6651
    Epoch [22/50], Val Losses: mse: 1.8675, mae: 0.6215, huber: 0.3356, swd: 0.4702, ept: 94.8240
    Epoch [22/50], Test Losses: mse: 1.7047, mae: 0.5947, huber: 0.3133, swd: 0.4257, ept: 94.9396
      Epoch 22 composite train-obj: 0.335135
            No improvement (0.3356), counter 1/5
    Epoch [23/50], Train Losses: mse: 3.1561, mae: 0.7120, huber: 0.4215, swd: 0.5632, ept: 94.2214
    Epoch [23/50], Val Losses: mse: 2.5968, mae: 0.6963, huber: 0.4040, swd: 0.5035, ept: 94.0870
    Epoch [23/50], Test Losses: mse: 2.1981, mae: 0.6634, huber: 0.3737, swd: 0.4248, ept: 94.4398
      Epoch 23 composite train-obj: 0.421502
            No improvement (0.4040), counter 2/5
    Epoch [24/50], Train Losses: mse: 2.1609, mae: 0.5700, huber: 0.3054, swd: 0.3838, ept: 94.7668
    Epoch [24/50], Val Losses: mse: 1.3580, mae: 0.5344, huber: 0.2678, swd: 0.3734, ept: 95.1139
    Epoch [24/50], Test Losses: mse: 1.2136, mae: 0.5284, huber: 0.2594, swd: 0.2843, ept: 95.2729
      Epoch 24 composite train-obj: 0.305386
            No improvement (0.2678), counter 3/5
    Epoch [25/50], Train Losses: mse: 2.6029, mae: 0.6303, huber: 0.3567, swd: 0.5785, ept: 94.3845
    Epoch [25/50], Val Losses: mse: 2.5999, mae: 0.6501, huber: 0.3753, swd: 0.6099, ept: 93.8068
    Epoch [25/50], Test Losses: mse: 2.2555, mae: 0.6147, huber: 0.3443, swd: 0.4129, ept: 94.2738
      Epoch 25 composite train-obj: 0.356667
            No improvement (0.3753), counter 4/5
    Epoch [26/50], Train Losses: mse: 3.0943, mae: 0.7044, huber: 0.4142, swd: 0.5214, ept: 94.3155
    Epoch [26/50], Val Losses: mse: 1.2765, mae: 0.5385, huber: 0.2726, swd: 0.3242, ept: 94.9494
    Epoch [26/50], Test Losses: mse: 1.3301, mae: 0.5393, huber: 0.2737, swd: 0.2521, ept: 95.0886
      Epoch 26 composite train-obj: 0.414219
    Epoch [26/50], Test Losses: mse: 0.7902, mae: 0.4492, huber: 0.1974, swd: 0.2157, ept: 95.5267
    Best round's Test MSE: 0.7902, MAE: 0.4492, SWD: 0.2157
    Best round's Validation MSE: 1.0267, MAE: 0.4748, SWD: 0.2530
    Best round's Test verification MSE : 0.7902, MAE: 0.4492, SWD: 0.2157
    Time taken: 211.88 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq96_pred96_20250513_0028)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 1.0112 ± 0.2211
      mae: 0.4649 ± 0.0153
      huber: 0.2131 ± 0.0164
      swd: 0.2421 ± 0.0195
      ept: 95.3839 ± 0.1269
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 1.2721 ± 0.2356
      mae: 0.4850 ± 0.0087
      huber: 0.2309 ± 0.0118
      swd: 0.3465 ± 0.0695
      ept: 95.2334 ± 0.2065
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 556.70 seconds
    
    Experiment complete: ACL_lorenz_seq96_pred96_20250513_0028
    Model: ACL
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 96-196
##### huber



```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 283
    Validation Batches: 39
    Test Batches: 79
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 46.2066, mae: 4.5224, huber: 4.0641, swd: 21.7703, ept: 82.0895
    Epoch [1/50], Val Losses: mse: 27.1849, mae: 3.2047, huber: 2.7609, swd: 3.8873, ept: 124.4401
    Epoch [1/50], Test Losses: mse: 25.8670, mae: 3.1196, huber: 2.6779, swd: 3.6018, ept: 124.8849
      Epoch 1 composite train-obj: 4.064131
            Val objective improved inf → 2.7609, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 18.5820, mae: 2.4271, huber: 2.0047, swd: 2.5801, ept: 146.9428
    Epoch [2/50], Val Losses: mse: 17.1969, mae: 2.3390, huber: 1.9184, swd: 2.2356, ept: 148.9188
    Epoch [2/50], Test Losses: mse: 16.0855, mae: 2.2879, huber: 1.8675, swd: 2.1740, ept: 150.0895
      Epoch 2 composite train-obj: 2.004717
            Val objective improved 2.7609 → 1.9184, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 13.3761, mae: 1.9111, huber: 1.5088, swd: 1.6684, ept: 161.4323
    Epoch [3/50], Val Losses: mse: 14.5925, mae: 2.0681, huber: 1.6590, swd: 2.1786, ept: 157.2965
    Epoch [3/50], Test Losses: mse: 13.5856, mae: 1.9870, huber: 1.5811, swd: 1.9345, ept: 160.0580
      Epoch 3 composite train-obj: 1.508781
            Val objective improved 1.9184 → 1.6590, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.4099, mae: 1.7005, huber: 1.3099, swd: 1.3670, ept: 167.7473
    Epoch [4/50], Val Losses: mse: 14.2637, mae: 1.9642, huber: 1.5626, swd: 1.2128, ept: 161.8816
    Epoch [4/50], Test Losses: mse: 12.5134, mae: 1.8915, huber: 1.4890, swd: 1.1489, ept: 163.3129
      Epoch 4 composite train-obj: 1.309948
            Val objective improved 1.6590 → 1.5626, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.0117, mae: 1.5407, huber: 1.1615, swd: 1.1900, ept: 171.4622
    Epoch [5/50], Val Losses: mse: 10.7477, mae: 1.7043, huber: 1.3132, swd: 1.5125, ept: 166.6269
    Epoch [5/50], Test Losses: mse: 9.7391, mae: 1.6548, huber: 1.2636, swd: 1.3406, ept: 168.2179
      Epoch 5 composite train-obj: 1.161479
            Val objective improved 1.5626 → 1.3132, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 9.6298, mae: 1.4909, huber: 1.1154, swd: 1.1072, ept: 173.3242
    Epoch [6/50], Val Losses: mse: 10.6175, mae: 1.6765, huber: 1.2935, swd: 1.3377, ept: 163.9643
    Epoch [6/50], Test Losses: mse: 10.3845, mae: 1.6768, huber: 1.2921, swd: 1.2533, ept: 164.6508
      Epoch 6 composite train-obj: 1.115367
            Val objective improved 1.3132 → 1.2935, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 8.7647, mae: 1.3937, huber: 1.0264, swd: 1.0153, ept: 175.2618
    Epoch [7/50], Val Losses: mse: 11.6488, mae: 1.7063, huber: 1.3203, swd: 1.4295, ept: 167.0394
    Epoch [7/50], Test Losses: mse: 10.1627, mae: 1.6257, huber: 1.2416, swd: 1.2379, ept: 169.5890
      Epoch 7 composite train-obj: 1.026363
            No improvement (1.3203), counter 1/5
    Epoch [8/50], Train Losses: mse: 8.8844, mae: 1.3929, huber: 1.0272, swd: 1.0396, ept: 175.4046
    Epoch [8/50], Val Losses: mse: 9.6648, mae: 1.5376, huber: 1.1614, swd: 0.9405, ept: 171.4421
    Epoch [8/50], Test Losses: mse: 8.4949, mae: 1.4851, huber: 1.1096, swd: 0.8581, ept: 172.9549
      Epoch 8 composite train-obj: 1.027203
            Val objective improved 1.2935 → 1.1614, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.7545, mae: 1.2803, huber: 0.9230, swd: 0.8679, ept: 177.9423
    Epoch [9/50], Val Losses: mse: 7.6398, mae: 1.3470, huber: 0.9803, swd: 0.8771, ept: 175.3263
    Epoch [9/50], Test Losses: mse: 6.9370, mae: 1.3172, huber: 0.9497, swd: 0.7637, ept: 176.9087
      Epoch 9 composite train-obj: 0.922999
            Val objective improved 1.1614 → 0.9803, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.7515, mae: 1.2672, huber: 0.9125, swd: 0.8778, ept: 177.9574
    Epoch [10/50], Val Losses: mse: 8.4889, mae: 1.4418, huber: 1.0664, swd: 0.9898, ept: 173.8661
    Epoch [10/50], Test Losses: mse: 7.6985, mae: 1.4088, huber: 1.0345, swd: 0.9333, ept: 175.1947
      Epoch 10 composite train-obj: 0.912509
            No improvement (1.0664), counter 1/5
    Epoch [11/50], Train Losses: mse: 7.1340, mae: 1.1908, huber: 0.8434, swd: 0.7917, ept: 179.9667
    Epoch [11/50], Val Losses: mse: 6.9236, mae: 1.2466, huber: 0.8926, swd: 1.0435, ept: 177.5973
    Epoch [11/50], Test Losses: mse: 5.8757, mae: 1.1926, huber: 0.8392, swd: 0.9650, ept: 179.6028
      Epoch 11 composite train-obj: 0.843419
            Val objective improved 0.9803 → 0.8926, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 7.0041, mae: 1.1687, huber: 0.8250, swd: 0.7776, ept: 180.2791
    Epoch [12/50], Val Losses: mse: 9.4784, mae: 1.4944, huber: 1.1152, swd: 1.0313, ept: 172.7880
    Epoch [12/50], Test Losses: mse: 8.1232, mae: 1.4405, huber: 1.0624, swd: 0.9087, ept: 174.1438
      Epoch 12 composite train-obj: 0.825012
            No improvement (1.1152), counter 1/5
    Epoch [13/50], Train Losses: mse: 7.0926, mae: 1.1749, huber: 0.8305, swd: 0.7639, ept: 180.4115
    Epoch [13/50], Val Losses: mse: 7.6434, mae: 1.2926, huber: 0.9378, swd: 0.9361, ept: 176.1193
    Epoch [13/50], Test Losses: mse: 6.6040, mae: 1.2411, huber: 0.8866, swd: 0.8362, ept: 178.4222
      Epoch 13 composite train-obj: 0.830482
            No improvement (0.9378), counter 2/5
    Epoch [14/50], Train Losses: mse: 6.5768, mae: 1.1259, huber: 0.7852, swd: 0.7216, ept: 181.2162
    Epoch [14/50], Val Losses: mse: 5.6253, mae: 1.0759, huber: 0.7365, swd: 0.6830, ept: 181.0831
    Epoch [14/50], Test Losses: mse: 4.9147, mae: 1.0370, huber: 0.6983, swd: 0.6259, ept: 183.6065
      Epoch 14 composite train-obj: 0.785198
            Val objective improved 0.8926 → 0.7365, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 6.4048, mae: 1.0972, huber: 0.7611, swd: 0.6658, ept: 181.7168
    Epoch [15/50], Val Losses: mse: 6.7445, mae: 1.2063, huber: 0.8594, swd: 0.6694, ept: 177.7381
    Epoch [15/50], Test Losses: mse: 5.4557, mae: 1.1322, huber: 0.7855, swd: 0.5966, ept: 180.0394
      Epoch 15 composite train-obj: 0.761058
            No improvement (0.8594), counter 1/5
    Epoch [16/50], Train Losses: mse: 6.5876, mae: 1.1148, huber: 0.7779, swd: 0.7430, ept: 181.5863
    Epoch [16/50], Val Losses: mse: 6.9487, mae: 1.2452, huber: 0.8911, swd: 0.8511, ept: 178.0400
    Epoch [16/50], Test Losses: mse: 6.0828, mae: 1.1680, huber: 0.8187, swd: 0.7106, ept: 180.5611
      Epoch 16 composite train-obj: 0.777899
            No improvement (0.8911), counter 2/5
    Epoch [17/50], Train Losses: mse: 5.8595, mae: 1.0321, huber: 0.7035, swd: 0.6560, ept: 183.1469
    Epoch [17/50], Val Losses: mse: 7.3855, mae: 1.2764, huber: 0.9227, swd: 0.7941, ept: 177.6225
    Epoch [17/50], Test Losses: mse: 6.7269, mae: 1.2277, huber: 0.8774, swd: 0.6784, ept: 179.5644
      Epoch 17 composite train-obj: 0.703549
            No improvement (0.9227), counter 3/5
    Epoch [18/50], Train Losses: mse: 5.8263, mae: 1.0310, huber: 0.7024, swd: 0.6221, ept: 183.2445
    Epoch [18/50], Val Losses: mse: 6.5097, mae: 1.1452, huber: 0.8041, swd: 0.8665, ept: 179.1682
    Epoch [18/50], Test Losses: mse: 5.0233, mae: 1.0673, huber: 0.7264, swd: 0.6749, ept: 181.5485
      Epoch 18 composite train-obj: 0.702366
            No improvement (0.8041), counter 4/5
    Epoch [19/50], Train Losses: mse: 5.8960, mae: 1.0370, huber: 0.7084, swd: 0.6319, ept: 183.0978
    Epoch [19/50], Val Losses: mse: 9.4346, mae: 1.3019, huber: 0.9569, swd: 0.8869, ept: 178.7970
    Epoch [19/50], Test Losses: mse: 8.1329, mae: 1.2173, huber: 0.8769, swd: 0.7230, ept: 181.4511
      Epoch 19 composite train-obj: 0.708428
    Epoch [19/50], Test Losses: mse: 4.9175, mae: 1.0372, huber: 0.6985, swd: 0.6269, ept: 183.5996
    Best round's Test MSE: 4.9147, MAE: 1.0370, SWD: 0.6259
    Best round's Validation MSE: 5.6253, MAE: 1.0759, SWD: 0.6830
    Best round's Test verification MSE : 4.9175, MAE: 1.0372, SWD: 0.6269
    Time taken: 165.98 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 46.3315, mae: 4.5197, huber: 4.0612, swd: 21.5065, ept: 82.8346
    Epoch [1/50], Val Losses: mse: 26.5908, mae: 3.1527, huber: 2.7112, swd: 4.8509, ept: 124.4225
    Epoch [1/50], Test Losses: mse: 25.4645, mae: 3.0797, huber: 2.6392, swd: 4.5542, ept: 125.2889
      Epoch 1 composite train-obj: 4.061192
            Val objective improved inf → 2.7112, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 20.0619, mae: 2.5398, huber: 2.1146, swd: 2.6014, ept: 144.9317
    Epoch [2/50], Val Losses: mse: 22.1230, mae: 2.6455, huber: 2.2165, swd: 1.7410, ept: 148.3270
    Epoch [2/50], Test Losses: mse: 20.2050, mae: 2.5501, huber: 2.1218, swd: 1.5922, ept: 149.2395
      Epoch 2 composite train-obj: 2.114602
            Val objective improved 2.7112 → 2.2165, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 14.9614, mae: 2.0430, huber: 1.6363, swd: 1.6881, ept: 160.0456
    Epoch [3/50], Val Losses: mse: 17.0818, mae: 2.2484, huber: 1.8348, swd: 1.8099, ept: 154.4406
    Epoch [3/50], Test Losses: mse: 15.5679, mae: 2.1852, huber: 1.7710, swd: 1.7208, ept: 155.5105
      Epoch 3 composite train-obj: 1.636309
            Val objective improved 2.2165 → 1.8348, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 13.0292, mae: 1.8414, huber: 1.4446, swd: 1.4514, ept: 166.3116
    Epoch [4/50], Val Losses: mse: 14.4526, mae: 1.9861, huber: 1.5797, swd: 1.5432, ept: 163.1461
    Epoch [4/50], Test Losses: mse: 12.2717, mae: 1.8695, huber: 1.4636, swd: 1.3381, ept: 166.5163
      Epoch 4 composite train-obj: 1.444553
            Val objective improved 1.8348 → 1.5797, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.4656, mae: 1.6729, huber: 1.2858, swd: 1.2451, ept: 170.2350
    Epoch [5/50], Val Losses: mse: 12.8355, mae: 1.8594, huber: 1.4619, swd: 1.2857, ept: 164.3229
    Epoch [5/50], Test Losses: mse: 11.4351, mae: 1.7950, huber: 1.3968, swd: 1.1624, ept: 166.0333
      Epoch 5 composite train-obj: 1.285782
            Val objective improved 1.5797 → 1.4619, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.5451, mae: 1.5627, huber: 1.1835, swd: 1.1192, ept: 172.7407
    Epoch [6/50], Val Losses: mse: 10.7537, mae: 1.6729, huber: 1.2843, swd: 1.3869, ept: 169.1660
    Epoch [6/50], Test Losses: mse: 10.0629, mae: 1.6327, huber: 1.2439, swd: 1.2743, ept: 171.7103
      Epoch 6 composite train-obj: 1.183458
            Val objective improved 1.4619 → 1.2843, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 10.0049, mae: 1.5071, huber: 1.1322, swd: 1.0568, ept: 173.8353
    Epoch [7/50], Val Losses: mse: 11.1633, mae: 1.6737, huber: 1.2920, swd: 1.2055, ept: 166.1062
    Epoch [7/50], Test Losses: mse: 10.0274, mae: 1.6279, huber: 1.2446, swd: 1.1000, ept: 167.1983
      Epoch 7 composite train-obj: 1.132241
            No improvement (1.2920), counter 1/5
    Epoch [8/50], Train Losses: mse: 9.8994, mae: 1.4728, huber: 1.1020, swd: 1.0962, ept: 174.6899
    Epoch [8/50], Val Losses: mse: 10.5166, mae: 1.6314, huber: 1.2471, swd: 1.1855, ept: 167.9352
    Epoch [8/50], Test Losses: mse: 9.0121, mae: 1.5464, huber: 1.1640, swd: 1.0413, ept: 169.9109
      Epoch 8 composite train-obj: 1.102004
            Val objective improved 1.2843 → 1.2471, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 8.6515, mae: 1.3544, huber: 0.9919, swd: 0.9346, ept: 176.9458
    Epoch [9/50], Val Losses: mse: 9.0406, mae: 1.5055, huber: 1.1310, swd: 1.0442, ept: 168.8418
    Epoch [9/50], Test Losses: mse: 7.9722, mae: 1.4509, huber: 1.0765, swd: 0.9139, ept: 171.1696
      Epoch 9 composite train-obj: 0.991940
            Val objective improved 1.2471 → 1.1310, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 8.3229, mae: 1.3138, huber: 0.9553, swd: 0.8656, ept: 177.7838
    Epoch [10/50], Val Losses: mse: 7.9863, mae: 1.3540, huber: 0.9877, swd: 1.0146, ept: 175.5255
    Epoch [10/50], Test Losses: mse: 7.6071, mae: 1.3240, huber: 0.9598, swd: 0.9433, ept: 177.9626
      Epoch 10 composite train-obj: 0.955277
            Val objective improved 1.1310 → 0.9877, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 7.8707, mae: 1.2501, huber: 0.8986, swd: 0.8510, ept: 178.8858
    Epoch [11/50], Val Losses: mse: 7.4058, mae: 1.3018, huber: 0.9416, swd: 0.8985, ept: 175.6499
    Epoch [11/50], Test Losses: mse: 6.5639, mae: 1.2585, huber: 0.8982, swd: 0.7062, ept: 177.6783
      Epoch 11 composite train-obj: 0.898602
            Val objective improved 0.9877 → 0.9416, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 9.1207, mae: 1.3703, huber: 1.0103, swd: 0.9879, ept: 176.6433
    Epoch [12/50], Val Losses: mse: 11.8111, mae: 1.6945, huber: 1.3159, swd: 1.6873, ept: 165.0114
    Epoch [12/50], Test Losses: mse: 10.2518, mae: 1.6095, huber: 1.2322, swd: 1.4957, ept: 166.8754
      Epoch 12 composite train-obj: 1.010283
            No improvement (1.3159), counter 1/5
    Epoch [13/50], Train Losses: mse: 8.3864, mae: 1.3165, huber: 0.9602, swd: 0.9823, ept: 177.3263
    Epoch [13/50], Val Losses: mse: 11.2112, mae: 1.6519, huber: 1.2735, swd: 1.1999, ept: 169.0773
    Epoch [13/50], Test Losses: mse: 9.4157, mae: 1.5446, huber: 1.1691, swd: 1.0855, ept: 170.9862
      Epoch 13 composite train-obj: 0.960162
            No improvement (1.2735), counter 2/5
    Epoch [14/50], Train Losses: mse: 7.4086, mae: 1.2061, huber: 0.8593, swd: 0.8114, ept: 179.7671
    Epoch [14/50], Val Losses: mse: 7.0753, mae: 1.2155, huber: 0.8688, swd: 0.8092, ept: 178.1518
    Epoch [14/50], Test Losses: mse: 6.1562, mae: 1.1640, huber: 0.8182, swd: 0.7009, ept: 179.9971
      Epoch 14 composite train-obj: 0.859284
            Val objective improved 0.9416 → 0.8688, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 7.3021, mae: 1.1826, huber: 0.8392, swd: 0.8035, ept: 180.3154
    Epoch [15/50], Val Losses: mse: 6.4150, mae: 1.1940, huber: 0.8460, swd: 0.7681, ept: 177.8680
    Epoch [15/50], Test Losses: mse: 5.4779, mae: 1.1366, huber: 0.7906, swd: 0.6974, ept: 179.5764
      Epoch 15 composite train-obj: 0.839217
            Val objective improved 0.8688 → 0.8460, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 6.7228, mae: 1.1138, huber: 0.7785, swd: 0.7727, ept: 181.3010
    Epoch [16/50], Val Losses: mse: 9.5428, mae: 1.4713, huber: 1.1031, swd: 0.8929, ept: 172.7534
    Epoch [16/50], Test Losses: mse: 9.3501, mae: 1.4467, huber: 1.0804, swd: 0.8224, ept: 174.4482
      Epoch 16 composite train-obj: 0.778486
            No improvement (1.1031), counter 1/5
    Epoch [17/50], Train Losses: mse: 7.5870, mae: 1.2092, huber: 0.8635, swd: 0.8144, ept: 180.0157
    Epoch [17/50], Val Losses: mse: 9.9838, mae: 1.5003, huber: 1.1307, swd: 0.9927, ept: 173.3288
    Epoch [17/50], Test Losses: mse: 9.0068, mae: 1.4427, huber: 1.0752, swd: 0.9868, ept: 174.9233
      Epoch 17 composite train-obj: 0.863469
            No improvement (1.1307), counter 2/5
    Epoch [18/50], Train Losses: mse: 6.8172, mae: 1.1323, huber: 0.7945, swd: 0.7354, ept: 181.0952
    Epoch [18/50], Val Losses: mse: 8.9386, mae: 1.3626, huber: 1.0123, swd: 0.8523, ept: 173.0872
    Epoch [18/50], Test Losses: mse: 8.4337, mae: 1.3237, huber: 0.9747, swd: 0.7555, ept: 175.6372
      Epoch 18 composite train-obj: 0.794508
            No improvement (1.0123), counter 3/5
    Epoch [19/50], Train Losses: mse: 6.7412, mae: 1.1200, huber: 0.7837, swd: 0.7141, ept: 181.5017
    Epoch [19/50], Val Losses: mse: 6.4812, mae: 1.1407, huber: 0.8002, swd: 0.8683, ept: 180.3137
    Epoch [19/50], Test Losses: mse: 5.3476, mae: 1.0589, huber: 0.7219, swd: 0.7061, ept: 183.2050
      Epoch 19 composite train-obj: 0.783741
            Val objective improved 0.8460 → 0.8002, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 6.5474, mae: 1.0984, huber: 0.7657, swd: 0.7572, ept: 181.7190
    Epoch [20/50], Val Losses: mse: 6.6964, mae: 1.2032, huber: 0.8588, swd: 1.0263, ept: 177.2174
    Epoch [20/50], Test Losses: mse: 5.8930, mae: 1.1399, huber: 0.7990, swd: 0.8526, ept: 179.2994
      Epoch 20 composite train-obj: 0.765677
            No improvement (0.8588), counter 1/5
    Epoch [21/50], Train Losses: mse: 6.6636, mae: 1.1090, huber: 0.7748, swd: 0.7072, ept: 181.6029
    Epoch [21/50], Val Losses: mse: 6.9153, mae: 1.2380, huber: 0.8866, swd: 0.9046, ept: 176.3782
    Epoch [21/50], Test Losses: mse: 6.1470, mae: 1.1762, huber: 0.8285, swd: 0.7583, ept: 179.2129
      Epoch 21 composite train-obj: 0.774804
            No improvement (0.8866), counter 2/5
    Epoch [22/50], Train Losses: mse: 6.1917, mae: 1.0696, huber: 0.7387, swd: 0.7209, ept: 182.2453
    Epoch [22/50], Val Losses: mse: 7.4083, mae: 1.3236, huber: 0.9646, swd: 0.8982, ept: 174.6844
    Epoch [22/50], Test Losses: mse: 7.1073, mae: 1.3014, huber: 0.9416, swd: 0.8786, ept: 178.0327
      Epoch 22 composite train-obj: 0.738690
            No improvement (0.9646), counter 3/5
    Epoch [23/50], Train Losses: mse: 7.2930, mae: 1.1854, huber: 0.8452, swd: 0.9561, ept: 179.5727
    Epoch [23/50], Val Losses: mse: 6.2882, mae: 1.1313, huber: 0.7957, swd: 0.7702, ept: 178.4985
    Epoch [23/50], Test Losses: mse: 4.5825, mae: 1.0243, huber: 0.6916, swd: 0.5934, ept: 181.5612
      Epoch 23 composite train-obj: 0.845191
            Val objective improved 0.8002 → 0.7957, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 7.2808, mae: 1.1941, huber: 0.8535, swd: 0.9613, ept: 179.2086
    Epoch [24/50], Val Losses: mse: 7.8579, mae: 1.2898, huber: 0.9428, swd: 0.9798, ept: 175.1082
    Epoch [24/50], Test Losses: mse: 6.4696, mae: 1.2148, huber: 0.8673, swd: 0.7652, ept: 177.8415
      Epoch 24 composite train-obj: 0.853480
            No improvement (0.9428), counter 1/5
    Epoch [25/50], Train Losses: mse: 6.6778, mae: 1.1203, huber: 0.7874, swd: 0.8631, ept: 180.5972
    Epoch [25/50], Val Losses: mse: 5.8325, mae: 1.1479, huber: 0.8046, swd: 0.8636, ept: 177.6478
    Epoch [25/50], Test Losses: mse: 4.9153, mae: 1.0826, huber: 0.7413, swd: 0.7316, ept: 180.6258
      Epoch 25 composite train-obj: 0.787436
            No improvement (0.8046), counter 2/5
    Epoch [26/50], Train Losses: mse: 6.0950, mae: 1.0496, huber: 0.7242, swd: 0.8108, ept: 181.7734
    Epoch [26/50], Val Losses: mse: 6.9832, mae: 1.2016, huber: 0.8530, swd: 1.3387, ept: 177.8206
    Epoch [26/50], Test Losses: mse: 5.8096, mae: 1.1558, huber: 0.8057, swd: 1.3379, ept: 178.9345
      Epoch 26 composite train-obj: 0.724181
            No improvement (0.8530), counter 3/5
    Epoch [27/50], Train Losses: mse: 6.3501, mae: 1.0866, huber: 0.7573, swd: 0.9205, ept: 181.0697
    Epoch [27/50], Val Losses: mse: 7.0016, mae: 1.1978, huber: 0.8529, swd: 0.9564, ept: 178.9810
    Epoch [27/50], Test Losses: mse: 5.3433, mae: 1.1280, huber: 0.7828, swd: 0.9853, ept: 180.3621
      Epoch 27 composite train-obj: 0.757318
            No improvement (0.8529), counter 4/5
    Epoch [28/50], Train Losses: mse: 6.0060, mae: 1.0499, huber: 0.7228, swd: 0.7888, ept: 182.1226
    Epoch [28/50], Val Losses: mse: 7.7260, mae: 1.2712, huber: 0.9227, swd: 0.8302, ept: 177.3847
    Epoch [28/50], Test Losses: mse: 6.5020, mae: 1.1902, huber: 0.8457, swd: 0.7267, ept: 179.9261
      Epoch 28 composite train-obj: 0.722847
    Epoch [28/50], Test Losses: mse: 4.5779, mae: 1.0241, huber: 0.6913, swd: 0.5934, ept: 181.5545
    Best round's Test MSE: 4.5825, MAE: 1.0243, SWD: 0.5934
    Best round's Validation MSE: 6.2882, MAE: 1.1313, SWD: 0.7702
    Best round's Test verification MSE : 4.5779, MAE: 1.0241, SWD: 0.5934
    Time taken: 247.87 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 47.1955, mae: 4.5750, huber: 4.1164, swd: 21.0783, ept: 75.8652
    Epoch [1/50], Val Losses: mse: 26.4279, mae: 3.1106, huber: 2.6700, swd: 3.1223, ept: 121.3640
    Epoch [1/50], Test Losses: mse: 25.2691, mae: 3.0578, huber: 2.6176, swd: 3.1359, ept: 121.7298
      Epoch 1 composite train-obj: 4.116426
            Val objective improved inf → 2.6700, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 19.5174, mae: 2.4829, huber: 2.0606, swd: 2.5447, ept: 145.5694
    Epoch [2/50], Val Losses: mse: 17.1808, mae: 2.3123, huber: 1.8932, swd: 1.7370, ept: 151.2634
    Epoch [2/50], Test Losses: mse: 15.0588, mae: 2.1837, huber: 1.7676, swd: 1.5180, ept: 153.7874
      Epoch 2 composite train-obj: 2.060556
            Val objective improved 2.6700 → 1.8932, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 13.7904, mae: 1.9305, huber: 1.5286, swd: 1.5793, ept: 163.0857
    Epoch [3/50], Val Losses: mse: 13.9940, mae: 1.9717, huber: 1.5691, swd: 1.6617, ept: 161.7025
    Epoch [3/50], Test Losses: mse: 12.5753, mae: 1.8938, huber: 1.4917, swd: 1.4727, ept: 164.7407
      Epoch 3 composite train-obj: 1.528559
            Val objective improved 1.8932 → 1.5691, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.6235, mae: 1.6964, huber: 1.3077, swd: 1.2563, ept: 169.3258
    Epoch [4/50], Val Losses: mse: 11.8324, mae: 1.7699, huber: 1.3766, swd: 1.1650, ept: 167.0576
    Epoch [4/50], Test Losses: mse: 10.2974, mae: 1.6941, huber: 1.3014, swd: 1.0243, ept: 169.4033
      Epoch 4 composite train-obj: 1.307748
            Val objective improved 1.5691 → 1.3766, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 9.9950, mae: 1.5117, huber: 1.1362, swd: 1.0747, ept: 173.8829
    Epoch [5/50], Val Losses: mse: 11.7221, mae: 1.7020, huber: 1.3184, swd: 1.2141, ept: 168.0105
    Epoch [5/50], Test Losses: mse: 10.3078, mae: 1.6160, huber: 1.2340, swd: 1.0141, ept: 170.6658
      Epoch 5 composite train-obj: 1.136198
            Val objective improved 1.3766 → 1.3184, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 9.0993, mae: 1.4120, huber: 1.0439, swd: 0.9621, ept: 175.9844
    Epoch [6/50], Val Losses: mse: 11.0989, mae: 1.6501, huber: 1.2659, swd: 1.3313, ept: 171.0128
    Epoch [6/50], Test Losses: mse: 9.6733, mae: 1.5635, huber: 1.1821, swd: 1.1044, ept: 172.7698
      Epoch 6 composite train-obj: 1.043931
            Val objective improved 1.3184 → 1.2659, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 8.2398, mae: 1.3093, huber: 0.9505, swd: 0.8700, ept: 178.1266
    Epoch [7/50], Val Losses: mse: 11.6331, mae: 1.6480, huber: 1.2701, swd: 1.1090, ept: 171.7237
    Epoch [7/50], Test Losses: mse: 9.2206, mae: 1.4903, huber: 1.1177, swd: 0.9416, ept: 174.2450
      Epoch 7 composite train-obj: 0.950541
            No improvement (1.2701), counter 1/5
    Epoch [8/50], Train Losses: mse: 8.0915, mae: 1.2861, huber: 0.9300, swd: 0.8544, ept: 178.9018
    Epoch [8/50], Val Losses: mse: 6.8385, mae: 1.2163, huber: 0.8627, swd: 0.8275, ept: 178.9282
    Epoch [8/50], Test Losses: mse: 5.5861, mae: 1.1290, huber: 0.7808, swd: 0.7148, ept: 181.0485
      Epoch 8 composite train-obj: 0.930010
            Val objective improved 1.2659 → 0.8627, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.3315, mae: 1.2007, huber: 0.8528, swd: 0.7559, ept: 180.3501
    Epoch [9/50], Val Losses: mse: 8.8211, mae: 1.4528, huber: 1.0840, swd: 0.8594, ept: 171.6778
    Epoch [9/50], Test Losses: mse: 8.1790, mae: 1.4143, huber: 1.0473, swd: 0.8500, ept: 173.2005
      Epoch 9 composite train-obj: 0.852779
            No improvement (1.0840), counter 1/5
    Epoch [10/50], Train Losses: mse: 7.1319, mae: 1.1736, huber: 0.8289, swd: 0.7215, ept: 181.0492
    Epoch [10/50], Val Losses: mse: 7.6040, mae: 1.3142, huber: 0.9560, swd: 0.9105, ept: 174.7564
    Epoch [10/50], Test Losses: mse: 6.4432, mae: 1.2463, huber: 0.8904, swd: 0.7409, ept: 176.5759
      Epoch 10 composite train-obj: 0.828905
            No improvement (0.9560), counter 2/5
    Epoch [11/50], Train Losses: mse: 6.8590, mae: 1.1380, huber: 0.7974, swd: 0.6960, ept: 181.8680
    Epoch [11/50], Val Losses: mse: 6.8711, mae: 1.2153, huber: 0.8665, swd: 0.7266, ept: 178.7651
    Epoch [11/50], Test Losses: mse: 5.9249, mae: 1.1499, huber: 0.8050, swd: 0.6838, ept: 180.6811
      Epoch 11 composite train-obj: 0.797407
            No improvement (0.8665), counter 3/5
    Epoch [12/50], Train Losses: mse: 6.4599, mae: 1.0956, huber: 0.7593, swd: 0.6644, ept: 182.4608
    Epoch [12/50], Val Losses: mse: 8.1837, mae: 1.2988, huber: 0.9531, swd: 0.7071, ept: 175.3576
    Epoch [12/50], Test Losses: mse: 7.6954, mae: 1.2877, huber: 0.9422, swd: 0.7199, ept: 175.2520
      Epoch 12 composite train-obj: 0.759274
            No improvement (0.9531), counter 4/5
    Epoch [13/50], Train Losses: mse: 6.4662, mae: 1.0887, huber: 0.7544, swd: 0.6573, ept: 182.4173
    Epoch [13/50], Val Losses: mse: 6.4253, mae: 1.1771, huber: 0.8315, swd: 0.8555, ept: 178.7674
    Epoch [13/50], Test Losses: mse: 6.2603, mae: 1.1596, huber: 0.8160, swd: 0.7915, ept: 180.7577
      Epoch 13 composite train-obj: 0.754438
            Val objective improved 0.8627 → 0.8315, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 5.9562, mae: 1.0318, huber: 0.7034, swd: 0.6130, ept: 183.6409
    Epoch [14/50], Val Losses: mse: 7.2975, mae: 1.2732, huber: 0.9162, swd: 0.8520, ept: 179.0531
    Epoch [14/50], Test Losses: mse: 6.0799, mae: 1.2105, huber: 0.8536, swd: 0.7849, ept: 180.5342
      Epoch 14 composite train-obj: 0.703350
            No improvement (0.9162), counter 1/5
    Epoch [15/50], Train Losses: mse: 6.3129, mae: 1.0668, huber: 0.7354, swd: 0.6245, ept: 183.0331
    Epoch [15/50], Val Losses: mse: 7.0668, mae: 1.2349, huber: 0.8867, swd: 0.8172, ept: 177.3140
    Epoch [15/50], Test Losses: mse: 6.7501, mae: 1.2020, huber: 0.8542, swd: 0.6936, ept: 179.7199
      Epoch 15 composite train-obj: 0.735364
            No improvement (0.8867), counter 2/5
    Epoch [16/50], Train Losses: mse: 5.8537, mae: 1.0167, huber: 0.6913, swd: 0.5809, ept: 183.6999
    Epoch [16/50], Val Losses: mse: 6.3187, mae: 1.1331, huber: 0.7919, swd: 0.7160, ept: 179.9029
    Epoch [16/50], Test Losses: mse: 5.0836, mae: 1.0600, huber: 0.7205, swd: 0.5437, ept: 181.9631
      Epoch 16 composite train-obj: 0.691281
            Val objective improved 0.8315 → 0.7919, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 5.6787, mae: 0.9969, huber: 0.6734, swd: 0.5681, ept: 184.2417
    Epoch [17/50], Val Losses: mse: 6.1748, mae: 1.1070, huber: 0.7729, swd: 0.6783, ept: 180.6435
    Epoch [17/50], Test Losses: mse: 5.0657, mae: 1.0366, huber: 0.7042, swd: 0.5110, ept: 183.0135
      Epoch 17 composite train-obj: 0.673433
            Val objective improved 0.7919 → 0.7729, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 5.6791, mae: 0.9879, huber: 0.6672, swd: 0.5745, ept: 184.2244
    Epoch [18/50], Val Losses: mse: 5.3257, mae: 0.9820, huber: 0.6613, swd: 0.6281, ept: 183.8225
    Epoch [18/50], Test Losses: mse: 4.4103, mae: 0.9298, huber: 0.6097, swd: 0.5445, ept: 185.5329
      Epoch 18 composite train-obj: 0.667157
            Val objective improved 0.7729 → 0.6613, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 5.5233, mae: 0.9597, huber: 0.6440, swd: 0.5470, ept: 184.6864
    Epoch [19/50], Val Losses: mse: 5.6357, mae: 1.0844, huber: 0.7495, swd: 0.7246, ept: 181.0604
    Epoch [19/50], Test Losses: mse: 4.7190, mae: 1.0274, huber: 0.6926, swd: 0.5441, ept: 183.2770
      Epoch 19 composite train-obj: 0.643972
            No improvement (0.7495), counter 1/5
    Epoch [20/50], Train Losses: mse: 5.2042, mae: 0.9269, huber: 0.6141, swd: 0.5217, ept: 185.4481
    Epoch [20/50], Val Losses: mse: 5.1424, mae: 0.9811, huber: 0.6579, swd: 0.5745, ept: 183.6185
    Epoch [20/50], Test Losses: mse: 4.0343, mae: 0.9032, huber: 0.5842, swd: 0.4579, ept: 185.2816
      Epoch 20 composite train-obj: 0.614095
            Val objective improved 0.6613 → 0.6579, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 5.2520, mae: 0.9389, huber: 0.6242, swd: 0.5177, ept: 185.1314
    Epoch [21/50], Val Losses: mse: 5.8180, mae: 1.0827, huber: 0.7541, swd: 0.6828, ept: 178.8978
    Epoch [21/50], Test Losses: mse: 5.3342, mae: 1.0266, huber: 0.7023, swd: 0.6251, ept: 181.2012
      Epoch 21 composite train-obj: 0.624159
            No improvement (0.7541), counter 1/5
    Epoch [22/50], Train Losses: mse: 5.0120, mae: 0.9074, huber: 0.5973, swd: 0.5006, ept: 185.6264
    Epoch [22/50], Val Losses: mse: 5.2466, mae: 1.0413, huber: 0.7098, swd: 0.6277, ept: 182.5393
    Epoch [22/50], Test Losses: mse: 4.4742, mae: 0.9836, huber: 0.6557, swd: 0.5632, ept: 184.1034
      Epoch 22 composite train-obj: 0.597263
            No improvement (0.7098), counter 2/5
    Epoch [23/50], Train Losses: mse: 5.0223, mae: 0.9000, huber: 0.5922, swd: 0.4913, ept: 185.7918
    Epoch [23/50], Val Losses: mse: 4.6929, mae: 0.9550, huber: 0.6397, swd: 0.5537, ept: 182.9098
    Epoch [23/50], Test Losses: mse: 4.0698, mae: 0.9013, huber: 0.5897, swd: 0.4242, ept: 184.7731
      Epoch 23 composite train-obj: 0.592167
            Val objective improved 0.6579 → 0.6397, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 5.0585, mae: 0.9081, huber: 0.5992, swd: 0.4840, ept: 185.6468
    Epoch [24/50], Val Losses: mse: 5.0925, mae: 0.9627, huber: 0.6455, swd: 0.4965, ept: 182.4607
    Epoch [24/50], Test Losses: mse: 3.7420, mae: 0.8790, huber: 0.5664, swd: 0.4582, ept: 184.1434
      Epoch 24 composite train-obj: 0.599213
            No improvement (0.6455), counter 1/5
    Epoch [25/50], Train Losses: mse: 4.8443, mae: 0.8808, huber: 0.5761, swd: 0.4683, ept: 186.0247
    Epoch [25/50], Val Losses: mse: 5.9656, mae: 1.0210, huber: 0.6974, swd: 0.5570, ept: 183.0123
    Epoch [25/50], Test Losses: mse: 4.1234, mae: 0.9265, huber: 0.6054, swd: 0.4490, ept: 185.0829
      Epoch 25 composite train-obj: 0.576126
            No improvement (0.6974), counter 2/5
    Epoch [26/50], Train Losses: mse: 4.8388, mae: 0.8722, huber: 0.5693, swd: 0.4703, ept: 186.2396
    Epoch [26/50], Val Losses: mse: 6.1351, mae: 1.0899, huber: 0.7628, swd: 0.5952, ept: 180.5731
    Epoch [26/50], Test Losses: mse: 5.6879, mae: 1.0583, huber: 0.7327, swd: 0.5506, ept: 181.6518
      Epoch 26 composite train-obj: 0.569285
            No improvement (0.7628), counter 3/5
    Epoch [27/50], Train Losses: mse: 4.7273, mae: 0.8683, huber: 0.5649, swd: 0.4634, ept: 186.2061
    Epoch [27/50], Val Losses: mse: 4.0428, mae: 0.8370, huber: 0.5370, swd: 0.4133, ept: 185.4355
    Epoch [27/50], Test Losses: mse: 3.2517, mae: 0.7662, huber: 0.4714, swd: 0.3242, ept: 187.6341
      Epoch 27 composite train-obj: 0.564859
            Val objective improved 0.6397 → 0.5370, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 4.6593, mae: 0.8527, huber: 0.5520, swd: 0.4491, ept: 186.5519
    Epoch [28/50], Val Losses: mse: 6.3842, mae: 1.1269, huber: 0.7943, swd: 0.6540, ept: 180.5628
    Epoch [28/50], Test Losses: mse: 5.1715, mae: 1.0286, huber: 0.7023, swd: 0.5862, ept: 182.6983
      Epoch 28 composite train-obj: 0.551993
            No improvement (0.7943), counter 1/5
    Epoch [29/50], Train Losses: mse: 4.7680, mae: 0.8648, huber: 0.5634, swd: 0.4501, ept: 186.3478
    Epoch [29/50], Val Losses: mse: 5.4981, mae: 1.0603, huber: 0.7304, swd: 0.5541, ept: 181.7682
    Epoch [29/50], Test Losses: mse: 4.5460, mae: 1.0110, huber: 0.6804, swd: 0.4881, ept: 183.2526
      Epoch 29 composite train-obj: 0.563401
            No improvement (0.7304), counter 2/5
    Epoch [30/50], Train Losses: mse: 4.5989, mae: 0.8344, huber: 0.5388, swd: 0.4419, ept: 186.8206
    Epoch [30/50], Val Losses: mse: 4.5186, mae: 0.8833, huber: 0.5804, swd: 0.4525, ept: 184.9078
    Epoch [30/50], Test Losses: mse: 3.5457, mae: 0.8180, huber: 0.5180, swd: 0.4069, ept: 186.9241
      Epoch 30 composite train-obj: 0.538849
            No improvement (0.5804), counter 3/5
    Epoch [31/50], Train Losses: mse: 4.6035, mae: 0.8437, huber: 0.5455, swd: 0.4423, ept: 186.6814
    Epoch [31/50], Val Losses: mse: 7.4674, mae: 1.1487, huber: 0.8224, swd: 0.6713, ept: 179.6494
    Epoch [31/50], Test Losses: mse: 5.9851, mae: 1.0576, huber: 0.7351, swd: 0.6060, ept: 181.5118
      Epoch 31 composite train-obj: 0.545515
            No improvement (0.8224), counter 4/5
    Epoch [32/50], Train Losses: mse: 4.6225, mae: 0.8519, huber: 0.5517, swd: 0.4450, ept: 186.6786
    Epoch [32/50], Val Losses: mse: 4.9716, mae: 0.8712, huber: 0.5783, swd: 0.4511, ept: 184.9527
    Epoch [32/50], Test Losses: mse: 3.1657, mae: 0.7519, huber: 0.4654, swd: 0.3308, ept: 187.4522
      Epoch 32 composite train-obj: 0.551695
    Epoch [32/50], Test Losses: mse: 3.2517, mae: 0.7662, huber: 0.4714, swd: 0.3242, ept: 187.6341
    Best round's Test MSE: 3.2517, MAE: 0.7662, SWD: 0.3242
    Best round's Validation MSE: 4.0428, MAE: 0.8370, SWD: 0.4133
    Best round's Test verification MSE : 3.2517, MAE: 0.7662, SWD: 0.3242
    Time taken: 285.88 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq96_pred196_20250513_0055)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 4.2496 ± 0.7186
      mae: 0.9425 ± 0.1248
      huber: 0.6204 ± 0.1054
      swd: 0.5145 ± 0.1352
      ept: 184.2673 ± 2.5229
      count: 39.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.3188 ± 0.9420
      mae: 1.0148 ± 0.1277
      huber: 0.6897 ± 0.1106
      swd: 0.6222 ± 0.1519
      ept: 181.6723 ± 2.8625
      count: 39.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 699.80 seconds
    
    Experiment complete: ACL_lorenz_seq96_pred196_20250513_0055
    Model: ACL
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    




```python

```

#### 96-336
##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
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
    
    Epoch [1/50], Train Losses: mse: 70.5825, mae: 6.3643, huber: 5.8854, swd: 38.7052, ept: 37.1224
    Epoch [1/50], Val Losses: mse: 65.0353, mae: 6.0210, huber: 5.5434, swd: 17.2712, ept: 54.8217
    Epoch [1/50], Test Losses: mse: 64.5208, mae: 5.9819, huber: 5.5035, swd: 15.9468, ept: 57.4464
      Epoch 1 composite train-obj: 5.885445
            Val objective improved inf → 5.5434, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 44.3248, mae: 4.5579, huber: 4.0972, swd: 12.0604, ept: 105.9287
    Epoch [2/50], Val Losses: mse: 67.9599, mae: 5.9605, huber: 5.4895, swd: 12.4369, ept: 70.9309
    Epoch [2/50], Test Losses: mse: 67.5692, mae: 5.9545, huber: 5.4829, swd: 12.8087, ept: 68.5680
      Epoch 2 composite train-obj: 4.097243
            Val objective improved 5.5434 → 5.4895, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.1889, mae: 4.1647, huber: 3.7122, swd: 8.2576, ept: 133.2166
    Epoch [3/50], Val Losses: mse: 41.3419, mae: 4.2130, huber: 3.7596, swd: 7.2884, ept: 133.9316
    Epoch [3/50], Test Losses: mse: 39.9744, mae: 4.1028, huber: 3.6510, swd: 7.4229, ept: 134.3947
      Epoch 3 composite train-obj: 3.712216
            Val objective improved 5.4895 → 3.7596, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.7547, mae: 3.7399, huber: 3.2969, swd: 6.2306, ept: 159.4553
    Epoch [4/50], Val Losses: mse: 39.2410, mae: 4.0902, huber: 3.6384, swd: 7.2948, ept: 133.2895
    Epoch [4/50], Test Losses: mse: 38.0556, mae: 3.9947, huber: 3.5440, swd: 7.2250, ept: 134.4190
      Epoch 4 composite train-obj: 3.296870
            Val objective improved 3.7596 → 3.6384, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.4891, mae: 3.5343, huber: 3.0968, swd: 5.2859, ept: 170.8574
    Epoch [5/50], Val Losses: mse: 43.3978, mae: 4.2532, huber: 3.8015, swd: 5.1544, ept: 145.7399
    Epoch [5/50], Test Losses: mse: 42.3224, mae: 4.1698, huber: 3.7192, swd: 5.1423, ept: 144.3436
      Epoch 5 composite train-obj: 3.096784
            No improvement (3.8015), counter 1/5
    Epoch [6/50], Train Losses: mse: 34.2582, mae: 3.6071, huber: 3.1678, swd: 6.0248, ept: 167.5628
    Epoch [6/50], Val Losses: mse: 48.3213, mae: 4.5549, huber: 4.0975, swd: 5.4558, ept: 128.8429
    Epoch [6/50], Test Losses: mse: 43.8050, mae: 4.3063, huber: 3.8512, swd: 5.7805, ept: 128.2831
      Epoch 6 composite train-obj: 3.167809
            No improvement (4.0975), counter 2/5
    Epoch [7/50], Train Losses: mse: 34.3424, mae: 3.6008, huber: 3.1622, swd: 6.1913, ept: 169.3157
    Epoch [7/50], Val Losses: mse: 37.1924, mae: 3.9377, huber: 3.4863, swd: 5.7757, ept: 157.7791
    Epoch [7/50], Test Losses: mse: 34.4729, mae: 3.7300, huber: 3.2829, swd: 5.5406, ept: 157.9645
      Epoch 7 composite train-obj: 3.162194
            Val objective improved 3.6384 → 3.4863, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 30.4120, mae: 3.2794, huber: 2.8488, swd: 4.5892, ept: 184.2569
    Epoch [8/50], Val Losses: mse: 35.5616, mae: 3.7367, huber: 3.2921, swd: 4.6007, ept: 160.8081
    Epoch [8/50], Test Losses: mse: 33.4902, mae: 3.5944, huber: 3.1512, swd: 4.2741, ept: 161.1665
      Epoch 8 composite train-obj: 2.848758
            Val objective improved 3.4863 → 3.2921, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 28.8534, mae: 3.1313, huber: 2.7050, swd: 3.8790, ept: 191.9266
    Epoch [9/50], Val Losses: mse: 40.4969, mae: 3.9764, huber: 3.5301, swd: 3.4561, ept: 159.8675
    Epoch [9/50], Test Losses: mse: 34.4910, mae: 3.6181, huber: 3.1777, swd: 3.5769, ept: 165.3670
      Epoch 9 composite train-obj: 2.704989
            No improvement (3.5301), counter 1/5
    Epoch [10/50], Train Losses: mse: 27.7350, mae: 3.0486, huber: 2.6242, swd: 3.6018, ept: 196.6168
    Epoch [10/50], Val Losses: mse: 51.4678, mae: 4.7432, huber: 4.2879, swd: 4.8666, ept: 137.1521
    Epoch [10/50], Test Losses: mse: 50.2635, mae: 4.6791, huber: 4.2240, swd: 4.6625, ept: 137.8261
      Epoch 10 composite train-obj: 2.624192
            No improvement (4.2879), counter 2/5
    Epoch [11/50], Train Losses: mse: 31.4356, mae: 3.4163, huber: 2.9802, swd: 3.8602, ept: 177.9574
    Epoch [11/50], Val Losses: mse: 36.0790, mae: 3.8406, huber: 3.3952, swd: 4.6545, ept: 152.0061
    Epoch [11/50], Test Losses: mse: 32.7021, mae: 3.6065, huber: 3.1651, swd: 4.1012, ept: 156.6076
      Epoch 11 composite train-obj: 2.980192
            No improvement (3.3952), counter 3/5
    Epoch [12/50], Train Losses: mse: 26.7599, mae: 2.9853, huber: 2.5618, swd: 3.0680, ept: 198.8722
    Epoch [12/50], Val Losses: mse: 35.8547, mae: 3.6921, huber: 3.2525, swd: 3.0261, ept: 169.6975
    Epoch [12/50], Test Losses: mse: 31.8264, mae: 3.4323, huber: 2.9968, swd: 2.9130, ept: 173.9515
      Epoch 12 composite train-obj: 2.561846
            Val objective improved 3.2921 → 3.2525, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 25.7206, mae: 2.8722, huber: 2.4532, swd: 2.8143, ept: 205.3324
    Epoch [13/50], Val Losses: mse: 36.3203, mae: 3.6865, huber: 3.2451, swd: 3.1107, ept: 174.1570
    Epoch [13/50], Test Losses: mse: 32.3958, mae: 3.4553, huber: 3.0166, swd: 3.0066, ept: 178.4090
      Epoch 13 composite train-obj: 2.453216
            Val objective improved 3.2525 → 3.2451, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 25.3268, mae: 2.8294, huber: 2.4120, swd: 2.8202, ept: 210.1411
    Epoch [14/50], Val Losses: mse: 28.0146, mae: 3.0443, huber: 2.6199, swd: 3.0906, ept: 202.2893
    Epoch [14/50], Test Losses: mse: 24.3874, mae: 2.8125, huber: 2.3915, swd: 2.7190, ept: 208.2629
      Epoch 14 composite train-obj: 2.411963
            Val objective improved 3.2451 → 2.6199, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 23.3598, mae: 2.6495, huber: 2.2390, swd: 2.4495, ept: 218.6860
    Epoch [15/50], Val Losses: mse: 30.6990, mae: 3.2596, huber: 2.8330, swd: 2.8904, ept: 188.4203
    Epoch [15/50], Test Losses: mse: 26.1717, mae: 2.9978, huber: 2.5740, swd: 2.6524, ept: 191.5919
      Epoch 15 composite train-obj: 2.238975
            No improvement (2.8330), counter 1/5
    Epoch [16/50], Train Losses: mse: 22.6384, mae: 2.5733, huber: 2.1670, swd: 2.3025, ept: 223.7829
    Epoch [16/50], Val Losses: mse: 43.0319, mae: 3.9917, huber: 3.5538, swd: 2.7141, ept: 166.9854
    Epoch [16/50], Test Losses: mse: 42.0692, mae: 3.9485, huber: 3.5095, swd: 2.5635, ept: 168.5974
      Epoch 16 composite train-obj: 2.166972
            No improvement (3.5538), counter 2/5
    Epoch [17/50], Train Losses: mse: 26.4471, mae: 2.9295, huber: 2.5097, swd: 2.5052, ept: 204.4245
    Epoch [17/50], Val Losses: mse: 29.9157, mae: 3.2039, huber: 2.7763, swd: 2.6053, ept: 192.4681
    Epoch [17/50], Test Losses: mse: 26.6239, mae: 3.0141, huber: 2.5871, swd: 2.4255, ept: 197.4186
      Epoch 17 composite train-obj: 2.509694
            No improvement (2.7763), counter 3/5
    Epoch [18/50], Train Losses: mse: 23.3963, mae: 2.6488, huber: 2.2393, swd: 2.1866, ept: 218.8081
    Epoch [18/50], Val Losses: mse: 29.2599, mae: 3.1570, huber: 2.7325, swd: 2.8453, ept: 190.1638
    Epoch [18/50], Test Losses: mse: 25.8296, mae: 2.9638, huber: 2.5402, swd: 2.6981, ept: 195.8758
      Epoch 18 composite train-obj: 2.239325
            No improvement (2.7325), counter 4/5
    Epoch [19/50], Train Losses: mse: 23.2120, mae: 2.6427, huber: 2.2328, swd: 2.1506, ept: 220.4830
    Epoch [19/50], Val Losses: mse: 32.4142, mae: 3.3382, huber: 2.9084, swd: 2.4105, ept: 193.8170
    Epoch [19/50], Test Losses: mse: 29.5191, mae: 3.1544, huber: 2.7265, swd: 2.2543, ept: 198.4790
      Epoch 19 composite train-obj: 2.232790
    Epoch [19/50], Test Losses: mse: 24.3875, mae: 2.8125, huber: 2.3915, swd: 2.7191, ept: 208.2543
    Best round's Test MSE: 24.3874, MAE: 2.8125, SWD: 2.7190
    Best round's Validation MSE: 28.0146, MAE: 3.0443, SWD: 3.0906
    Best round's Test verification MSE : 24.3875, MAE: 2.8125, SWD: 2.7191
    Time taken: 151.32 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.8673, mae: 6.3717, huber: 5.8929, swd: 40.2487, ept: 35.0408
    Epoch [1/50], Val Losses: mse: 59.7111, mae: 5.7094, huber: 5.2342, swd: 16.7574, ept: 59.1376
    Epoch [1/50], Test Losses: mse: 56.6975, mae: 5.5128, huber: 5.0391, swd: 16.8427, ept: 64.5865
      Epoch 1 composite train-obj: 5.892890
            Val objective improved inf → 5.2342, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 44.7133, mae: 4.5879, huber: 4.1265, swd: 12.8093, ept: 108.7441
    Epoch [2/50], Val Losses: mse: 56.4683, mae: 5.3739, huber: 4.9049, swd: 11.2384, ept: 89.7162
    Epoch [2/50], Test Losses: mse: 54.5563, mae: 5.2728, huber: 4.8044, swd: 11.6935, ept: 87.6007
      Epoch 2 composite train-obj: 4.126540
            Val objective improved 5.2342 → 4.9049, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.5276, mae: 4.0970, huber: 3.6451, swd: 8.5164, ept: 145.2233
    Epoch [3/50], Val Losses: mse: 52.4166, mae: 4.9801, huber: 4.5163, swd: 8.7032, ept: 103.0191
    Epoch [3/50], Test Losses: mse: 50.3069, mae: 4.8247, huber: 4.3625, swd: 7.5380, ept: 109.3370
      Epoch 3 composite train-obj: 3.645116
            Val objective improved 4.9049 → 4.5163, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.6783, mae: 3.8333, huber: 3.3871, swd: 6.6371, ept: 160.3233
    Epoch [4/50], Val Losses: mse: 48.4196, mae: 4.7065, huber: 4.2465, swd: 7.0267, ept: 126.7518
    Epoch [4/50], Test Losses: mse: 45.6290, mae: 4.5068, huber: 4.0496, swd: 7.0387, ept: 130.3075
      Epoch 4 composite train-obj: 3.387113
            Val objective improved 4.5163 → 4.2465, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 34.5518, mae: 3.6527, huber: 3.2107, swd: 5.5765, ept: 169.5042
    Epoch [5/50], Val Losses: mse: 39.0788, mae: 4.0921, huber: 3.6393, swd: 7.1492, ept: 151.1718
    Epoch [5/50], Test Losses: mse: 37.0592, mae: 3.9545, huber: 3.5034, swd: 6.8201, ept: 153.6504
      Epoch 5 composite train-obj: 3.210653
            Val objective improved 4.2465 → 3.6393, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 32.0287, mae: 3.4141, huber: 2.9784, swd: 4.7201, ept: 183.1153
    Epoch [6/50], Val Losses: mse: 39.7283, mae: 3.9908, huber: 3.5421, swd: 4.0880, ept: 155.3957
    Epoch [6/50], Test Losses: mse: 35.8338, mae: 3.7406, huber: 3.2953, swd: 3.9865, ept: 162.7213
      Epoch 6 composite train-obj: 2.978427
            Val objective improved 3.6393 → 3.5421, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 30.0468, mae: 3.2208, huber: 2.7919, swd: 4.0770, ept: 194.6405
    Epoch [7/50], Val Losses: mse: 35.1574, mae: 3.6855, huber: 3.2430, swd: 4.1917, ept: 169.6479
    Epoch [7/50], Test Losses: mse: 32.5293, mae: 3.4954, huber: 3.0558, swd: 3.8767, ept: 175.3336
      Epoch 7 composite train-obj: 2.791893
            Val objective improved 3.5421 → 3.2430, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 29.1315, mae: 3.1381, huber: 2.7117, swd: 3.8049, ept: 200.1510
    Epoch [8/50], Val Losses: mse: 33.7835, mae: 3.5669, huber: 3.1261, swd: 3.8501, ept: 179.3638
    Epoch [8/50], Test Losses: mse: 30.5148, mae: 3.3517, huber: 2.9139, swd: 3.6068, ept: 183.8647
      Epoch 8 composite train-obj: 2.711734
            Val objective improved 3.2430 → 3.1261, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 27.8628, mae: 3.0276, huber: 2.6045, swd: 3.4496, ept: 207.2701
    Epoch [9/50], Val Losses: mse: 38.1245, mae: 3.8759, huber: 3.4277, swd: 5.3001, ept: 167.7631
    Epoch [9/50], Test Losses: mse: 34.8152, mae: 3.6758, huber: 3.2298, swd: 5.0630, ept: 173.3824
      Epoch 9 composite train-obj: 2.604545
            No improvement (3.4277), counter 1/5
    Epoch [10/50], Train Losses: mse: 27.3715, mae: 2.9789, huber: 2.5577, swd: 3.3245, ept: 210.2076
    Epoch [10/50], Val Losses: mse: 40.0762, mae: 3.9124, huber: 3.4695, swd: 3.4751, ept: 169.3023
    Epoch [10/50], Test Losses: mse: 37.2000, mae: 3.7370, huber: 3.2956, swd: 3.0514, ept: 175.2214
      Epoch 10 composite train-obj: 2.557722
            No improvement (3.4695), counter 2/5
    Epoch [11/50], Train Losses: mse: 27.5313, mae: 2.9919, huber: 2.5705, swd: 3.1824, ept: 209.7316
    Epoch [11/50], Val Losses: mse: 37.6166, mae: 3.8291, huber: 3.3854, swd: 4.1861, ept: 163.1251
    Epoch [11/50], Test Losses: mse: 34.7430, mae: 3.6536, huber: 3.2120, swd: 3.9657, ept: 169.1534
      Epoch 11 composite train-obj: 2.570468
            No improvement (3.3854), counter 3/5
    Epoch [12/50], Train Losses: mse: 26.7453, mae: 2.9261, huber: 2.5067, swd: 3.0511, ept: 213.2975
    Epoch [12/50], Val Losses: mse: 33.7341, mae: 3.4712, huber: 3.0360, swd: 3.4188, ept: 191.4344
    Epoch [12/50], Test Losses: mse: 30.9224, mae: 3.2602, huber: 2.8297, swd: 3.0940, ept: 200.5545
      Epoch 12 composite train-obj: 2.506709
            Val objective improved 3.1261 → 3.0360, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 25.5519, mae: 2.8224, huber: 2.4066, swd: 3.0859, ept: 219.2432
    Epoch [13/50], Val Losses: mse: 32.1351, mae: 3.4277, huber: 2.9903, swd: 4.0986, ept: 189.2477
    Epoch [13/50], Test Losses: mse: 30.0616, mae: 3.3106, huber: 2.8743, swd: 3.9223, ept: 192.1267
      Epoch 13 composite train-obj: 2.406614
            Val objective improved 3.0360 → 2.9903, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 23.9202, mae: 2.6863, huber: 2.2751, swd: 2.6429, ept: 225.2246
    Epoch [14/50], Val Losses: mse: 42.3518, mae: 3.9865, huber: 3.5453, swd: 3.9645, ept: 177.7258
    Epoch [14/50], Test Losses: mse: 40.2634, mae: 3.8367, huber: 3.3986, swd: 3.3242, ept: 183.7757
      Epoch 14 composite train-obj: 2.275102
            No improvement (3.5453), counter 1/5
    Epoch [15/50], Train Losses: mse: 26.3376, mae: 2.9198, huber: 2.5000, swd: 2.7560, ept: 213.1323
    Epoch [15/50], Val Losses: mse: 34.8862, mae: 3.5769, huber: 3.1400, swd: 3.3232, ept: 180.6071
    Epoch [15/50], Test Losses: mse: 30.6244, mae: 3.3391, huber: 2.9032, swd: 3.0789, ept: 187.2450
      Epoch 15 composite train-obj: 2.500011
            No improvement (3.1400), counter 2/5
    Epoch [16/50], Train Losses: mse: 23.5610, mae: 2.6679, huber: 2.2564, swd: 2.4670, ept: 225.5247
    Epoch [16/50], Val Losses: mse: 44.4086, mae: 4.0477, huber: 3.6047, swd: 3.3099, ept: 173.9467
    Epoch [16/50], Test Losses: mse: 41.8338, mae: 3.9203, huber: 3.4775, swd: 3.0077, ept: 176.7553
      Epoch 16 composite train-obj: 2.256363
            No improvement (3.6047), counter 3/5
    Epoch [17/50], Train Losses: mse: 25.9013, mae: 2.8637, huber: 2.4460, swd: 2.7018, ept: 214.1079
    Epoch [17/50], Val Losses: mse: 31.3987, mae: 3.3651, huber: 2.9303, swd: 3.0702, ept: 197.8990
    Epoch [17/50], Test Losses: mse: 26.5842, mae: 3.0708, huber: 2.6407, swd: 2.9100, ept: 205.6072
      Epoch 17 composite train-obj: 2.446006
            Val objective improved 2.9903 → 2.9303, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 22.4739, mae: 2.5548, huber: 2.1477, swd: 2.2418, ept: 231.0651
    Epoch [18/50], Val Losses: mse: 30.8242, mae: 3.2000, huber: 2.7701, swd: 2.4164, ept: 202.8510
    Epoch [18/50], Test Losses: mse: 27.3845, mae: 3.0082, huber: 2.5805, swd: 2.2328, ept: 210.2119
      Epoch 18 composite train-obj: 2.147748
            Val objective improved 2.9303 → 2.7701, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 21.9145, mae: 2.4951, huber: 2.0913, swd: 2.1489, ept: 234.7501
    Epoch [19/50], Val Losses: mse: 32.4263, mae: 3.3417, huber: 2.9111, swd: 3.4845, ept: 194.3119
    Epoch [19/50], Test Losses: mse: 29.9452, mae: 3.2098, huber: 2.7813, swd: 3.3932, ept: 199.4034
      Epoch 19 composite train-obj: 2.091332
            No improvement (2.9111), counter 1/5
    Epoch [20/50], Train Losses: mse: 21.5059, mae: 2.4615, huber: 2.0589, swd: 2.1251, ept: 235.7981
    Epoch [20/50], Val Losses: mse: 31.6977, mae: 3.2185, huber: 2.7935, swd: 2.1471, ept: 201.1242
    Epoch [20/50], Test Losses: mse: 26.4752, mae: 2.9125, huber: 2.4923, swd: 1.8228, ept: 207.8052
      Epoch 20 composite train-obj: 2.058943
            No improvement (2.7935), counter 2/5
    Epoch [21/50], Train Losses: mse: 21.3009, mae: 2.4359, huber: 2.0345, swd: 2.0506, ept: 238.4467
    Epoch [21/50], Val Losses: mse: 24.1814, mae: 2.7466, huber: 2.3270, swd: 2.5618, ept: 220.9796
    Epoch [21/50], Test Losses: mse: 21.4731, mae: 2.5542, huber: 2.1391, swd: 2.2590, ept: 229.9175
      Epoch 21 composite train-obj: 2.034512
            Val objective improved 2.7701 → 2.3270, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 20.1429, mae: 2.3558, huber: 1.9569, swd: 1.9068, ept: 240.2887
    Epoch [22/50], Val Losses: mse: 33.3728, mae: 3.3087, huber: 2.8795, swd: 2.4964, ept: 197.8021
    Epoch [22/50], Test Losses: mse: 28.8254, mae: 3.0626, huber: 2.6370, swd: 2.2119, ept: 206.1118
      Epoch 22 composite train-obj: 1.956934
            No improvement (2.8795), counter 1/5
    Epoch [23/50], Train Losses: mse: 20.4219, mae: 2.3609, huber: 1.9631, swd: 1.8845, ept: 241.0077
    Epoch [23/50], Val Losses: mse: 26.5612, mae: 2.8606, huber: 2.4436, swd: 2.0133, ept: 216.3051
    Epoch [23/50], Test Losses: mse: 23.5057, mae: 2.6662, huber: 2.2531, swd: 1.7861, ept: 223.9622
      Epoch 23 composite train-obj: 1.963076
            No improvement (2.4436), counter 2/5
    Epoch [24/50], Train Losses: mse: 19.8717, mae: 2.3111, huber: 1.9150, swd: 1.8117, ept: 243.9070
    Epoch [24/50], Val Losses: mse: 24.3110, mae: 2.6921, huber: 2.2778, swd: 1.7613, ept: 225.8530
    Epoch [24/50], Test Losses: mse: 19.7824, mae: 2.4329, huber: 2.0229, swd: 1.6360, ept: 233.6541
      Epoch 24 composite train-obj: 1.915049
            Val objective improved 2.3270 → 2.2778, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 18.8039, mae: 2.2148, huber: 1.8239, swd: 1.7117, ept: 248.0176
    Epoch [25/50], Val Losses: mse: 27.3712, mae: 2.9395, huber: 2.5175, swd: 2.1139, ept: 212.3770
    Epoch [25/50], Test Losses: mse: 23.9413, mae: 2.7630, huber: 2.3420, swd: 1.9921, ept: 216.0080
      Epoch 25 composite train-obj: 1.823919
            No improvement (2.5175), counter 1/5
    Epoch [26/50], Train Losses: mse: 18.8034, mae: 2.2255, huber: 1.8329, swd: 1.7013, ept: 247.2462
    Epoch [26/50], Val Losses: mse: 27.6160, mae: 2.9430, huber: 2.5263, swd: 2.2667, ept: 211.7135
    Epoch [26/50], Test Losses: mse: 25.8876, mae: 2.8510, huber: 2.4349, swd: 2.0796, ept: 217.5430
      Epoch 26 composite train-obj: 1.832943
            No improvement (2.5263), counter 2/5
    Epoch [27/50], Train Losses: mse: 18.6076, mae: 2.1993, huber: 1.8092, swd: 1.6786, ept: 248.7571
    Epoch [27/50], Val Losses: mse: 27.6955, mae: 2.9609, huber: 2.5389, swd: 2.4726, ept: 215.0052
    Epoch [27/50], Test Losses: mse: 24.4407, mae: 2.7831, huber: 2.3625, swd: 2.4327, ept: 219.4055
      Epoch 27 composite train-obj: 1.809247
            No improvement (2.5389), counter 3/5
    Epoch [28/50], Train Losses: mse: 18.4746, mae: 2.1969, huber: 1.8055, swd: 1.6433, ept: 249.0451
    Epoch [28/50], Val Losses: mse: 29.1738, mae: 2.9848, huber: 2.5681, swd: 2.1034, ept: 213.3020
    Epoch [28/50], Test Losses: mse: 25.2643, mae: 2.7595, huber: 2.3468, swd: 1.8414, ept: 221.2018
      Epoch 28 composite train-obj: 1.805542
            No improvement (2.5681), counter 4/5
    Epoch [29/50], Train Losses: mse: 18.8470, mae: 2.2203, huber: 1.8277, swd: 1.6833, ept: 249.1140
    Epoch [29/50], Val Losses: mse: 37.0509, mae: 3.4609, huber: 3.0336, swd: 2.3488, ept: 200.2805
    Epoch [29/50], Test Losses: mse: 32.2272, mae: 3.1757, huber: 2.7542, swd: 2.0857, ept: 210.8928
      Epoch 29 composite train-obj: 1.827749
    Epoch [29/50], Test Losses: mse: 19.7824, mae: 2.4329, huber: 2.0229, swd: 1.6360, ept: 233.6496
    Best round's Test MSE: 19.7824, MAE: 2.4329, SWD: 1.6360
    Best round's Validation MSE: 24.3110, MAE: 2.6921, SWD: 1.7613
    Best round's Test verification MSE : 19.7824, MAE: 2.4329, SWD: 1.6360
    Time taken: 223.40 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.7319, mae: 6.3486, huber: 5.8699, swd: 40.1044, ept: 37.8268
    Epoch [1/50], Val Losses: mse: 63.0059, mae: 5.9308, huber: 5.4537, swd: 19.6623, ept: 56.9736
    Epoch [1/50], Test Losses: mse: 61.3093, mae: 5.8109, huber: 5.3350, swd: 20.2947, ept: 60.3964
      Epoch 1 composite train-obj: 5.869910
            Val objective improved inf → 5.4537, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 46.2646, mae: 4.7152, huber: 4.2522, swd: 15.9102, ept: 103.7630
    Epoch [2/50], Val Losses: mse: 46.9174, mae: 4.7775, huber: 4.3138, swd: 10.7667, ept: 95.0146
    Epoch [2/50], Test Losses: mse: 44.7756, mae: 4.6454, huber: 4.1825, swd: 10.5803, ept: 94.1490
      Epoch 2 composite train-obj: 4.252180
            Val objective improved 5.4537 → 4.3138, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.2777, mae: 4.1016, huber: 3.6490, swd: 8.5858, ept: 137.3814
    Epoch [3/50], Val Losses: mse: 43.0647, mae: 4.3381, huber: 3.8816, swd: 7.7507, ept: 123.7341
    Epoch [3/50], Test Losses: mse: 40.4135, mae: 4.1452, huber: 3.6917, swd: 7.3428, ept: 127.2946
      Epoch 3 composite train-obj: 3.649019
            Val objective improved 4.3138 → 3.8816, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.7170, mae: 3.7570, huber: 3.3128, swd: 6.7052, ept: 164.9195
    Epoch [4/50], Val Losses: mse: 62.2283, mae: 5.4832, huber: 5.0166, swd: 7.8675, ept: 106.0650
    Epoch [4/50], Test Losses: mse: 60.5491, mae: 5.4043, huber: 4.9376, swd: 7.5301, ept: 106.3053
      Epoch 4 composite train-obj: 3.312806
            No improvement (5.0166), counter 1/5
    Epoch [5/50], Train Losses: mse: 36.3689, mae: 3.8307, huber: 3.3848, swd: 6.1201, ept: 161.5661
    Epoch [5/50], Val Losses: mse: 42.8311, mae: 4.1921, huber: 3.7402, swd: 4.8200, ept: 151.6071
    Epoch [5/50], Test Losses: mse: 41.9800, mae: 4.1370, huber: 3.6850, swd: 4.7924, ept: 150.9840
      Epoch 5 composite train-obj: 3.384803
            Val objective improved 3.8816 → 3.7402, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 33.7773, mae: 3.5714, huber: 3.1319, swd: 5.5613, ept: 177.0627
    Epoch [6/50], Val Losses: mse: 53.3282, mae: 4.9169, huber: 4.4579, swd: 5.9954, ept: 122.7944
    Epoch [6/50], Test Losses: mse: 51.1595, mae: 4.8263, huber: 4.3667, swd: 5.7927, ept: 120.9980
      Epoch 6 composite train-obj: 3.131928
            No improvement (4.4579), counter 1/5
    Epoch [7/50], Train Losses: mse: 33.8786, mae: 3.6056, huber: 3.1663, swd: 5.0949, ept: 173.4912
    Epoch [7/50], Val Losses: mse: 36.8098, mae: 3.8466, huber: 3.4019, swd: 5.3002, ept: 161.5354
    Epoch [7/50], Test Losses: mse: 34.7008, mae: 3.7010, huber: 3.2573, swd: 4.9644, ept: 164.4825
      Epoch 7 composite train-obj: 3.166254
            Val objective improved 3.7402 → 3.4019, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 29.9995, mae: 3.2455, huber: 2.8148, swd: 4.1568, ept: 192.9838
    Epoch [8/50], Val Losses: mse: 42.6753, mae: 4.2395, huber: 3.7870, swd: 4.0600, ept: 157.5415
    Epoch [8/50], Test Losses: mse: 39.4316, mae: 4.0252, huber: 3.5756, swd: 3.8867, ept: 162.2115
      Epoch 8 composite train-obj: 2.814792
            No improvement (3.7870), counter 1/5
    Epoch [9/50], Train Losses: mse: 29.5356, mae: 3.1869, huber: 2.7589, swd: 3.8813, ept: 196.5075
    Epoch [9/50], Val Losses: mse: 34.6382, mae: 3.7334, huber: 3.2861, swd: 6.6248, ept: 168.3970
    Epoch [9/50], Test Losses: mse: 32.4660, mae: 3.6304, huber: 3.1816, swd: 6.8070, ept: 169.0550
      Epoch 9 composite train-obj: 2.758916
            Val objective improved 3.4019 → 3.2861, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 27.4455, mae: 3.0010, huber: 2.5792, swd: 3.5404, ept: 205.8893
    Epoch [10/50], Val Losses: mse: 34.9520, mae: 3.5672, huber: 3.1305, swd: 3.7929, ept: 180.4343
    Epoch [10/50], Test Losses: mse: 31.7281, mae: 3.3666, huber: 2.9335, swd: 3.5320, ept: 185.6473
      Epoch 10 composite train-obj: 2.579180
            Val objective improved 3.2861 → 3.1305, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 26.7891, mae: 2.9347, huber: 2.5151, swd: 3.3188, ept: 210.9027
    Epoch [11/50], Val Losses: mse: 37.8787, mae: 3.7623, huber: 3.3214, swd: 3.8775, ept: 176.7151
    Epoch [11/50], Test Losses: mse: 38.5922, mae: 3.8214, huber: 3.3768, swd: 3.7632, ept: 175.0910
      Epoch 11 composite train-obj: 2.515082
            No improvement (3.3214), counter 1/5
    Epoch [12/50], Train Losses: mse: 27.6637, mae: 3.0128, huber: 2.5903, swd: 3.5340, ept: 207.8932
    Epoch [12/50], Val Losses: mse: 31.4014, mae: 3.3677, huber: 2.9360, swd: 4.0733, ept: 191.8627
    Epoch [12/50], Test Losses: mse: 27.3376, mae: 3.1175, huber: 2.6903, swd: 3.7748, ept: 194.3976
      Epoch 12 composite train-obj: 2.590313
            Val objective improved 3.1305 → 2.9360, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 25.4300, mae: 2.8133, huber: 2.3979, swd: 2.9884, ept: 216.1084
    Epoch [13/50], Val Losses: mse: 39.9582, mae: 3.9244, huber: 3.4821, swd: 3.4153, ept: 172.4238
    Epoch [13/50], Test Losses: mse: 33.8129, mae: 3.5517, huber: 3.1148, swd: 3.1808, ept: 180.7745
      Epoch 13 composite train-obj: 2.397906
            No improvement (3.4821), counter 1/5
    Epoch [14/50], Train Losses: mse: 25.6541, mae: 2.8455, huber: 2.4285, swd: 3.0715, ept: 215.5472
    Epoch [14/50], Val Losses: mse: 29.8999, mae: 3.2294, huber: 2.8001, swd: 3.4047, ept: 197.5227
    Epoch [14/50], Test Losses: mse: 26.9353, mae: 3.0594, huber: 2.6323, swd: 3.3403, ept: 199.5742
      Epoch 14 composite train-obj: 2.428470
            Val objective improved 2.9360 → 2.8001, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 23.8969, mae: 2.6787, huber: 2.2683, swd: 2.6652, ept: 223.8358
    Epoch [15/50], Val Losses: mse: 33.0693, mae: 3.4132, huber: 2.9805, swd: 4.1634, ept: 193.5869
    Epoch [15/50], Test Losses: mse: 28.4985, mae: 3.1549, huber: 2.7259, swd: 3.7821, ept: 198.0958
      Epoch 15 composite train-obj: 2.268309
            No improvement (2.9805), counter 1/5
    Epoch [16/50], Train Losses: mse: 23.7087, mae: 2.6689, huber: 2.2584, swd: 2.6266, ept: 224.7069
    Epoch [16/50], Val Losses: mse: 32.8115, mae: 3.3307, huber: 2.9026, swd: 3.1919, ept: 196.0659
    Epoch [16/50], Test Losses: mse: 29.8443, mae: 3.1706, huber: 2.7433, swd: 2.9109, ept: 200.7679
      Epoch 16 composite train-obj: 2.258406
            No improvement (2.9026), counter 2/5
    Epoch [17/50], Train Losses: mse: 23.4295, mae: 2.6419, huber: 2.2326, swd: 2.5367, ept: 227.1636
    Epoch [17/50], Val Losses: mse: 32.4341, mae: 3.4082, huber: 2.9743, swd: 3.2164, ept: 188.5815
    Epoch [17/50], Test Losses: mse: 29.1581, mae: 3.2330, huber: 2.7996, swd: 2.8992, ept: 191.1531
      Epoch 17 composite train-obj: 2.232551
            No improvement (2.9743), counter 3/5
    Epoch [18/50], Train Losses: mse: 22.4526, mae: 2.5606, huber: 2.1542, swd: 2.3948, ept: 230.4251
    Epoch [18/50], Val Losses: mse: 49.3747, mae: 4.2624, huber: 3.8199, swd: 3.2341, ept: 166.3547
    Epoch [18/50], Test Losses: mse: 42.5495, mae: 3.8745, huber: 3.4375, swd: 2.9523, ept: 175.3947
      Epoch 18 composite train-obj: 2.154247
            No improvement (3.8199), counter 4/5
    Epoch [19/50], Train Losses: mse: 34.7839, mae: 3.5788, huber: 3.1452, swd: 8.5861, ept: 181.4007
    Epoch [19/50], Val Losses: mse: 41.1278, mae: 3.9468, huber: 3.5041, swd: 7.6534, ept: 173.4179
    Epoch [19/50], Test Losses: mse: 39.0386, mae: 3.7908, huber: 3.3503, swd: 6.5400, ept: 178.6119
      Epoch 19 composite train-obj: 3.145181
    Epoch [19/50], Test Losses: mse: 26.9356, mae: 3.0595, huber: 2.6324, swd: 3.3404, ept: 199.5698
    Best round's Test MSE: 26.9353, MAE: 3.0594, SWD: 3.3403
    Best round's Validation MSE: 29.8999, MAE: 3.2294, SWD: 3.4047
    Best round's Test verification MSE : 26.9356, MAE: 3.0595, SWD: 3.3404
    Time taken: 149.93 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq96_pred336_20250513_2356)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 23.7017 ± 2.9601
      mae: 2.7683 ± 0.2577
      huber: 2.3489 ± 0.2506
      swd: 2.5651 ± 0.7043
      ept: 213.8304 ± 14.4593
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 27.4085 ± 2.3216
      mae: 2.9886 ± 0.2229
      huber: 2.5659 ± 0.2166
      swd: 2.7522 ± 0.7123
      ept: 208.5550 ± 12.3854
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 524.71 seconds
    
    Experiment complete: ACL_lorenz_seq96_pred336_20250513_2356
    Model: ACL
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 96-720
##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 279
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 279
    Validation Batches: 35
    Test Batches: 75
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 67.9549, mae: 6.1787, huber: 5.7018, swd: 35.8915, ept: 60.4502
    Epoch [1/50], Val Losses: mse: 64.7488, mae: 6.0276, huber: 5.5529, swd: 21.7811, ept: 72.2356
    Epoch [1/50], Test Losses: mse: 63.6005, mae: 5.9342, huber: 5.4596, swd: 21.3126, ept: 73.6010
      Epoch 1 composite train-obj: 5.701755
            Val objective improved inf → 5.5529, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 54.2896, mae: 5.2837, huber: 4.8167, swd: 17.1008, ept: 128.8205
    Epoch [2/50], Val Losses: mse: 62.6237, mae: 5.8265, huber: 5.3527, swd: 14.9894, ept: 107.3529
    Epoch [2/50], Test Losses: mse: 61.5659, mae: 5.7276, huber: 5.2541, swd: 14.1172, ept: 106.8442
      Epoch 2 composite train-obj: 4.816739
            Val objective improved 5.5529 → 5.3527, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 51.0038, mae: 4.9759, huber: 4.5147, swd: 12.5872, ept: 166.0906
    Epoch [3/50], Val Losses: mse: 56.4943, mae: 5.3534, huber: 4.8862, swd: 11.0808, ept: 140.3021
    Epoch [3/50], Test Losses: mse: 55.3038, mae: 5.2374, huber: 4.7721, swd: 10.8732, ept: 142.6502
      Epoch 3 composite train-obj: 4.514736
            Val objective improved 5.3527 → 4.8862, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 49.2878, mae: 4.8117, huber: 4.3544, swd: 10.7571, ept: 184.0045
    Epoch [4/50], Val Losses: mse: 54.6335, mae: 5.2034, huber: 4.7419, swd: 11.1320, ept: 159.2930
    Epoch [4/50], Test Losses: mse: 53.6053, mae: 5.1025, huber: 4.6417, swd: 10.8241, ept: 159.5955
      Epoch 4 composite train-obj: 4.354386
            Val objective improved 4.8862 → 4.7419, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 47.5500, mae: 4.6521, huber: 4.1988, swd: 9.6808, ept: 201.2397
    Epoch [5/50], Val Losses: mse: 53.2795, mae: 5.0416, huber: 4.5825, swd: 9.6568, ept: 176.2294
    Epoch [5/50], Test Losses: mse: 52.2665, mae: 4.9324, huber: 4.4751, swd: 9.4204, ept: 177.2023
      Epoch 5 composite train-obj: 4.198834
            Val objective improved 4.7419 → 4.5825, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 46.5467, mae: 4.5620, huber: 4.1108, swd: 8.8381, ept: 210.6825
    Epoch [6/50], Val Losses: mse: 52.1329, mae: 4.9645, huber: 4.5063, swd: 8.8014, ept: 186.8851
    Epoch [6/50], Test Losses: mse: 51.6824, mae: 4.8885, huber: 4.4313, swd: 8.5394, ept: 185.2549
      Epoch 6 composite train-obj: 4.110833
            Val objective improved 4.5825 → 4.5063, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 45.3781, mae: 4.4449, huber: 3.9974, swd: 8.2329, ept: 224.8224
    Epoch [7/50], Val Losses: mse: 55.0606, mae: 5.0945, huber: 4.6351, swd: 8.9022, ept: 184.7289
    Epoch [7/50], Test Losses: mse: 54.3040, mae: 5.0193, huber: 4.5609, swd: 8.7716, ept: 181.9038
      Epoch 7 composite train-obj: 3.997418
            No improvement (4.6351), counter 1/5
    Epoch [8/50], Train Losses: mse: 45.1156, mae: 4.4217, huber: 3.9748, swd: 8.1715, ept: 227.4147
    Epoch [8/50], Val Losses: mse: 53.2532, mae: 5.0088, huber: 4.5520, swd: 8.2993, ept: 191.9737
    Epoch [8/50], Test Losses: mse: 51.4343, mae: 4.8608, huber: 4.4053, swd: 7.8417, ept: 193.6602
      Epoch 8 composite train-obj: 3.974769
            No improvement (4.5520), counter 2/5
    Epoch [9/50], Train Losses: mse: 44.2728, mae: 4.3453, huber: 3.9007, swd: 7.5865, ept: 235.9761
    Epoch [9/50], Val Losses: mse: 54.4322, mae: 5.0506, huber: 4.5931, swd: 8.5658, ept: 187.0651
    Epoch [9/50], Test Losses: mse: 52.7171, mae: 4.9143, huber: 4.4591, swd: 8.5314, ept: 189.9032
      Epoch 9 composite train-obj: 3.900676
            No improvement (4.5931), counter 3/5
    Epoch [10/50], Train Losses: mse: 43.9859, mae: 4.3114, huber: 3.8674, swd: 7.2611, ept: 238.8045
    Epoch [10/50], Val Losses: mse: 51.6555, mae: 4.8366, huber: 4.3824, swd: 7.6340, ept: 206.8451
    Epoch [10/50], Test Losses: mse: 50.3525, mae: 4.7171, huber: 4.2647, swd: 7.4769, ept: 208.4044
      Epoch 10 composite train-obj: 3.867429
            Val objective improved 4.5063 → 4.3824, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 43.2525, mae: 4.2450, huber: 3.8029, swd: 6.8804, ept: 245.3738
    Epoch [11/50], Val Losses: mse: 52.3876, mae: 4.9103, huber: 4.4552, swd: 7.7802, ept: 204.3396
    Epoch [11/50], Test Losses: mse: 50.2963, mae: 4.7438, huber: 4.2910, swd: 7.6174, ept: 210.9130
      Epoch 11 composite train-obj: 3.802948
            No improvement (4.4552), counter 1/5
    Epoch [12/50], Train Losses: mse: 43.1339, mae: 4.2387, huber: 3.7963, swd: 7.0302, ept: 247.8394
    Epoch [12/50], Val Losses: mse: 51.0131, mae: 4.7428, huber: 4.2914, swd: 7.0552, ept: 215.7338
    Epoch [12/50], Test Losses: mse: 50.4563, mae: 4.6559, huber: 4.2062, swd: 6.6849, ept: 218.5527
      Epoch 12 composite train-obj: 3.796299
            Val objective improved 4.3824 → 4.2914, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 42.5233, mae: 4.1708, huber: 3.7309, swd: 6.6893, ept: 254.5959
    Epoch [13/50], Val Losses: mse: 54.9610, mae: 4.9988, huber: 4.5428, swd: 6.5032, ept: 198.7569
    Epoch [13/50], Test Losses: mse: 53.3262, mae: 4.8792, huber: 4.4243, swd: 6.3814, ept: 199.3002
      Epoch 13 composite train-obj: 3.730941
            No improvement (4.5428), counter 1/5
    Epoch [14/50], Train Losses: mse: 41.9967, mae: 4.1306, huber: 3.6918, swd: 6.4022, ept: 258.9392
    Epoch [14/50], Val Losses: mse: 50.2157, mae: 4.6789, huber: 4.2300, swd: 6.7337, ept: 223.8139
    Epoch [14/50], Test Losses: mse: 48.6787, mae: 4.5568, huber: 4.1095, swd: 6.6003, ept: 226.7193
      Epoch 14 composite train-obj: 3.691773
            Val objective improved 4.2914 → 4.2300, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 41.3857, mae: 4.0827, huber: 3.6450, swd: 5.9831, ept: 261.2702
    Epoch [15/50], Val Losses: mse: 51.2561, mae: 4.6964, huber: 4.2482, swd: 6.3743, ept: 225.3988
    Epoch [15/50], Test Losses: mse: 50.4928, mae: 4.6015, huber: 4.1549, swd: 6.2555, ept: 232.3097
      Epoch 15 composite train-obj: 3.645046
            No improvement (4.2482), counter 1/5
    Epoch [16/50], Train Losses: mse: 41.6418, mae: 4.0875, huber: 3.6500, swd: 6.2937, ept: 261.8643
    Epoch [16/50], Val Losses: mse: 50.7306, mae: 4.6626, huber: 4.2150, swd: 6.2449, ept: 230.9418
    Epoch [16/50], Test Losses: mse: 48.8458, mae: 4.5211, huber: 4.0750, swd: 5.9577, ept: 236.0638
      Epoch 16 composite train-obj: 3.650047
            Val objective improved 4.2300 → 4.2150, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 39.8580, mae: 3.9378, huber: 3.5052, swd: 5.5107, ept: 275.0329
    Epoch [17/50], Val Losses: mse: 53.1671, mae: 4.7924, huber: 4.3425, swd: 6.0228, ept: 220.7246
    Epoch [17/50], Test Losses: mse: 50.9385, mae: 4.6427, huber: 4.1940, swd: 5.9238, ept: 224.4791
      Epoch 17 composite train-obj: 3.505166
            No improvement (4.3425), counter 1/5
    Epoch [18/50], Train Losses: mse: 40.4624, mae: 3.9851, huber: 3.5506, swd: 5.4215, ept: 268.6242
    Epoch [18/50], Val Losses: mse: 50.1653, mae: 4.5826, huber: 4.1387, swd: 5.5667, ept: 237.2594
    Epoch [18/50], Test Losses: mse: 49.0673, mae: 4.4817, huber: 4.0399, swd: 5.5819, ept: 239.6783
      Epoch 18 composite train-obj: 3.550604
            Val objective improved 4.2150 → 4.1387, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 39.8568, mae: 3.9246, huber: 3.4923, swd: 5.2579, ept: 275.6412
    Epoch [19/50], Val Losses: mse: 57.0264, mae: 4.9950, huber: 4.5458, swd: 6.1778, ept: 221.8939
    Epoch [19/50], Test Losses: mse: 54.9360, mae: 4.8572, huber: 4.4099, swd: 6.1169, ept: 224.6790
      Epoch 19 composite train-obj: 3.492256
            No improvement (4.5458), counter 1/5
    Epoch [20/50], Train Losses: mse: 40.3594, mae: 3.9832, huber: 3.5487, swd: 5.2558, ept: 271.7055
    Epoch [20/50], Val Losses: mse: 52.9663, mae: 4.7680, huber: 4.3199, swd: 5.2361, ept: 232.0807
    Epoch [20/50], Test Losses: mse: 49.5007, mae: 4.5394, huber: 4.0948, swd: 5.1916, ept: 239.9194
      Epoch 20 composite train-obj: 3.548708
            No improvement (4.3199), counter 2/5
    Epoch [21/50], Train Losses: mse: 40.1710, mae: 3.9767, huber: 3.5413, swd: 5.2333, ept: 270.4443
    Epoch [21/50], Val Losses: mse: 50.1657, mae: 4.5291, huber: 4.0883, swd: 5.4435, ept: 249.9612
    Epoch [21/50], Test Losses: mse: 47.2561, mae: 4.3452, huber: 3.9065, swd: 5.3549, ept: 254.3482
      Epoch 21 composite train-obj: 3.541338
            Val objective improved 4.1387 → 4.0883, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 38.9498, mae: 3.8497, huber: 3.4193, swd: 4.9278, ept: 282.8820
    Epoch [22/50], Val Losses: mse: 52.6855, mae: 4.6864, huber: 4.2414, swd: 5.1930, ept: 238.9697
    Epoch [22/50], Test Losses: mse: 50.2512, mae: 4.5318, huber: 4.0890, swd: 5.1381, ept: 242.4687
      Epoch 22 composite train-obj: 3.419286
            No improvement (4.2414), counter 1/5
    Epoch [23/50], Train Losses: mse: 38.5878, mae: 3.8142, huber: 3.3849, swd: 4.8073, ept: 286.4001
    Epoch [23/50], Val Losses: mse: 53.1811, mae: 4.7109, huber: 4.2667, swd: 4.6243, ept: 238.9051
    Epoch [23/50], Test Losses: mse: 49.7930, mae: 4.4843, huber: 4.0433, swd: 4.5729, ept: 244.0787
      Epoch 23 composite train-obj: 3.384875
            No improvement (4.2667), counter 2/5
    Epoch [24/50], Train Losses: mse: 38.4066, mae: 3.7955, huber: 3.3671, swd: 4.7011, ept: 288.7311
    Epoch [24/50], Val Losses: mse: 54.2787, mae: 4.8518, huber: 4.4030, swd: 5.1737, ept: 226.2231
    Epoch [24/50], Test Losses: mse: 51.2997, mae: 4.6725, huber: 4.2250, swd: 5.0794, ept: 229.1603
      Epoch 24 composite train-obj: 3.367092
            No improvement (4.4030), counter 3/5
    Epoch [25/50], Train Losses: mse: 40.1157, mae: 3.9544, huber: 3.5212, swd: 4.8650, ept: 271.9104
    Epoch [25/50], Val Losses: mse: 52.4379, mae: 4.6736, huber: 4.2301, swd: 5.2696, ept: 236.5417
    Epoch [25/50], Test Losses: mse: 50.1844, mae: 4.5220, huber: 4.0804, swd: 5.1400, ept: 240.4346
      Epoch 25 composite train-obj: 3.521195
            No improvement (4.2301), counter 4/5
    Epoch [26/50], Train Losses: mse: 38.5868, mae: 3.8177, huber: 3.3880, swd: 4.7571, ept: 284.5898
    Epoch [26/50], Val Losses: mse: 51.4359, mae: 4.5990, huber: 4.1564, swd: 5.0262, ept: 248.5067
    Epoch [26/50], Test Losses: mse: 48.6655, mae: 4.4257, huber: 3.9852, swd: 5.0019, ept: 248.9711
      Epoch 26 composite train-obj: 3.388044
    Epoch [26/50], Test Losses: mse: 47.2565, mae: 4.3452, huber: 3.9065, swd: 5.3552, ept: 254.3741
    Best round's Test MSE: 47.2561, MAE: 4.3452, SWD: 5.3549
    Best round's Validation MSE: 50.1657, MAE: 4.5291, SWD: 5.4435
    Best round's Test verification MSE : 47.2565, MAE: 4.3452, SWD: 5.3552
    Time taken: 225.92 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 67.6523, mae: 6.1602, huber: 5.6834, swd: 35.0530, ept: 63.2190
    Epoch [1/50], Val Losses: mse: 64.8908, mae: 6.0499, huber: 5.5743, swd: 20.7736, ept: 74.1890
    Epoch [1/50], Test Losses: mse: 63.6148, mae: 5.9446, huber: 5.4696, swd: 20.5484, ept: 74.0466
      Epoch 1 composite train-obj: 5.683449
            Val objective improved inf → 5.5743, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 53.4773, mae: 5.2206, huber: 4.7547, swd: 15.5866, ept: 140.7247
    Epoch [2/50], Val Losses: mse: 55.6920, mae: 5.3463, huber: 4.8795, swd: 13.3274, ept: 141.3336
    Epoch [2/50], Test Losses: mse: 54.4269, mae: 5.2380, huber: 4.7721, swd: 13.3059, ept: 141.0670
      Epoch 2 composite train-obj: 4.754682
            Val objective improved 5.5743 → 4.8795, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.1970, mae: 4.9152, huber: 4.4554, swd: 11.8150, ept: 180.9755
    Epoch [3/50], Val Losses: mse: 61.2659, mae: 5.6474, huber: 5.1778, swd: 13.3913, ept: 140.3213
    Epoch [3/50], Test Losses: mse: 60.5254, mae: 5.5699, huber: 5.1003, swd: 12.7028, ept: 139.5257
      Epoch 3 composite train-obj: 4.455384
            No improvement (5.1778), counter 1/5
    Epoch [4/50], Train Losses: mse: 48.4679, mae: 4.7474, huber: 4.2918, swd: 10.2664, ept: 197.9486
    Epoch [4/50], Val Losses: mse: 55.8028, mae: 5.1887, huber: 4.7263, swd: 10.2907, ept: 173.4154
    Epoch [4/50], Test Losses: mse: 54.1094, mae: 5.0392, huber: 4.5785, swd: 9.8674, ept: 175.3636
      Epoch 4 composite train-obj: 4.291838
            Val objective improved 4.8795 → 4.7263, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 47.3470, mae: 4.6320, huber: 4.1792, swd: 9.5093, ept: 209.4166
    Epoch [5/50], Val Losses: mse: 52.9222, mae: 5.0968, huber: 4.6348, swd: 12.1198, ept: 178.7300
    Epoch [5/50], Test Losses: mse: 51.8539, mae: 4.9880, huber: 4.5278, swd: 11.9696, ept: 180.6139
      Epoch 5 composite train-obj: 4.179189
            Val objective improved 4.7263 → 4.6348, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 46.1916, mae: 4.5283, huber: 4.0785, swd: 8.7087, ept: 222.2090
    Epoch [6/50], Val Losses: mse: 53.3022, mae: 5.0324, huber: 4.5730, swd: 9.3305, ept: 193.3454
    Epoch [6/50], Test Losses: mse: 51.2072, mae: 4.8698, huber: 4.4120, swd: 9.1062, ept: 194.0451
      Epoch 6 composite train-obj: 4.078524
            Val objective improved 4.6348 → 4.5730, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 45.0052, mae: 4.4234, huber: 3.9765, swd: 7.9996, ept: 234.0704
    Epoch [7/50], Val Losses: mse: 62.5791, mae: 5.5651, huber: 5.1032, swd: 10.4493, ept: 174.9526
    Epoch [7/50], Test Losses: mse: 61.6002, mae: 5.4730, huber: 5.0126, swd: 10.4504, ept: 176.9560
      Epoch 7 composite train-obj: 3.976453
            No improvement (5.1032), counter 1/5
    Epoch [8/50], Train Losses: mse: 47.0589, mae: 4.6065, huber: 4.1554, swd: 8.1348, ept: 217.3198
    Epoch [8/50], Val Losses: mse: 53.9348, mae: 5.0398, huber: 4.5831, swd: 8.0104, ept: 187.6471
    Epoch [8/50], Test Losses: mse: 52.3746, mae: 4.9142, huber: 4.4589, swd: 8.0030, ept: 189.3101
      Epoch 8 composite train-obj: 4.155384
            No improvement (4.5831), counter 2/5
    Epoch [9/50], Train Losses: mse: 46.1017, mae: 4.5017, huber: 4.0533, swd: 7.5312, ept: 228.0676
    Epoch [9/50], Val Losses: mse: 55.4776, mae: 5.1185, huber: 4.6597, swd: 9.3008, ept: 183.8458
    Epoch [9/50], Test Losses: mse: 53.6499, mae: 4.9846, huber: 4.5277, swd: 9.3527, ept: 188.0730
      Epoch 9 composite train-obj: 4.053300
            No improvement (4.6597), counter 3/5
    Epoch [10/50], Train Losses: mse: 44.9878, mae: 4.4028, huber: 3.9566, swd: 7.7577, ept: 237.1203
    Epoch [10/50], Val Losses: mse: 53.6753, mae: 4.9253, huber: 4.4710, swd: 7.5190, ept: 211.6672
    Epoch [10/50], Test Losses: mse: 50.6704, mae: 4.7158, huber: 4.2648, swd: 7.3325, ept: 217.6728
      Epoch 10 composite train-obj: 3.956593
            Val objective improved 4.5730 → 4.4710, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 43.5785, mae: 4.2758, huber: 3.8331, swd: 6.9380, ept: 250.0075
    Epoch [11/50], Val Losses: mse: 56.9317, mae: 5.1165, huber: 4.6591, swd: 7.7793, ept: 197.0132
    Epoch [11/50], Test Losses: mse: 57.7657, mae: 5.1009, huber: 4.6439, swd: 7.4559, ept: 194.7156
      Epoch 11 composite train-obj: 3.833145
            No improvement (4.6591), counter 1/5
    Epoch [12/50], Train Losses: mse: 45.4075, mae: 4.4334, huber: 3.9869, swd: 8.9087, ept: 233.6557
    Epoch [12/50], Val Losses: mse: 52.2504, mae: 4.8613, huber: 4.4084, swd: 7.7725, ept: 209.6307
    Epoch [12/50], Test Losses: mse: 50.9138, mae: 4.7608, huber: 4.3088, swd: 7.9697, ept: 211.9507
      Epoch 12 composite train-obj: 3.986949
            Val objective improved 4.4710 → 4.4084, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 42.8491, mae: 4.2119, huber: 3.7713, swd: 6.9870, ept: 254.5749
    Epoch [13/50], Val Losses: mse: 51.4139, mae: 4.7297, huber: 4.2801, swd: 6.6347, ept: 224.6765
    Epoch [13/50], Test Losses: mse: 49.7255, mae: 4.6114, huber: 4.1628, swd: 6.7581, ept: 225.6176
      Epoch 13 composite train-obj: 3.771347
            Val objective improved 4.4084 → 4.2801, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 42.6183, mae: 4.1787, huber: 3.7390, swd: 6.7511, ept: 258.1318
    Epoch [14/50], Val Losses: mse: 53.1996, mae: 4.8698, huber: 4.4177, swd: 6.3854, ept: 218.7145
    Epoch [14/50], Test Losses: mse: 50.8485, mae: 4.7014, huber: 4.2511, swd: 6.3667, ept: 223.1089
      Epoch 14 composite train-obj: 3.738988
            No improvement (4.4177), counter 1/5
    Epoch [15/50], Train Losses: mse: 42.7012, mae: 4.1921, huber: 3.7520, swd: 6.4040, ept: 257.0012
    Epoch [15/50], Val Losses: mse: 52.3209, mae: 4.8516, huber: 4.3991, swd: 8.4729, ept: 217.6833
    Epoch [15/50], Test Losses: mse: 50.0806, mae: 4.6859, huber: 4.2358, swd: 8.4964, ept: 224.2412
      Epoch 15 composite train-obj: 3.751979
            No improvement (4.3991), counter 2/5
    Epoch [16/50], Train Losses: mse: 41.8708, mae: 4.1338, huber: 3.6944, swd: 6.6498, ept: 263.9481
    Epoch [16/50], Val Losses: mse: 54.2269, mae: 4.9479, huber: 4.4938, swd: 6.9003, ept: 212.7128
    Epoch [16/50], Test Losses: mse: 52.4180, mae: 4.8070, huber: 4.3548, swd: 6.7515, ept: 214.7260
      Epoch 16 composite train-obj: 3.694423
            No improvement (4.4938), counter 3/5
    Epoch [17/50], Train Losses: mse: 41.6888, mae: 4.1018, huber: 3.6639, swd: 5.9943, ept: 265.4983
    Epoch [17/50], Val Losses: mse: 53.4714, mae: 4.7799, huber: 4.3333, swd: 5.9470, ept: 227.0743
    Epoch [17/50], Test Losses: mse: 51.6175, mae: 4.6497, huber: 4.2044, swd: 5.7088, ept: 226.2488
      Epoch 17 composite train-obj: 3.663891
            No improvement (4.3333), counter 4/5
    Epoch [18/50], Train Losses: mse: 41.5068, mae: 4.0740, huber: 3.6375, swd: 6.2521, ept: 267.7294
    Epoch [18/50], Val Losses: mse: 51.1780, mae: 4.6750, huber: 4.2263, swd: 6.9157, ept: 236.2694
    Epoch [18/50], Test Losses: mse: 49.0048, mae: 4.5304, huber: 4.0833, swd: 6.8849, ept: 238.4230
      Epoch 18 composite train-obj: 3.637532
            Val objective improved 4.2801 → 4.2263, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 40.3604, mae: 3.9735, huber: 3.5398, swd: 5.6714, ept: 278.0770
    Epoch [19/50], Val Losses: mse: 59.9801, mae: 5.2107, huber: 4.7540, swd: 6.0765, ept: 201.4276
    Epoch [19/50], Test Losses: mse: 56.1479, mae: 4.9782, huber: 4.5241, swd: 6.0201, ept: 208.3782
      Epoch 19 composite train-obj: 3.539846
            No improvement (4.7540), counter 1/5
    Epoch [20/50], Train Losses: mse: 42.3428, mae: 4.1623, huber: 3.7227, swd: 6.0715, ept: 258.5186
    Epoch [20/50], Val Losses: mse: 50.6872, mae: 4.6134, huber: 4.1700, swd: 5.6664, ept: 239.8783
    Epoch [20/50], Test Losses: mse: 47.3153, mae: 4.3984, huber: 3.9568, swd: 5.3543, ept: 245.0551
      Epoch 20 composite train-obj: 3.722684
            Val objective improved 4.2263 → 4.1700, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 39.5232, mae: 3.8949, huber: 3.4644, swd: 5.1244, ept: 285.3774
    Epoch [21/50], Val Losses: mse: 49.8291, mae: 4.5644, huber: 4.1217, swd: 6.0263, ept: 238.6189
    Epoch [21/50], Test Losses: mse: 47.0349, mae: 4.3802, huber: 3.9398, swd: 6.0368, ept: 244.5228
      Epoch 21 composite train-obj: 3.464351
            Val objective improved 4.1700 → 4.1217, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 38.8346, mae: 3.8301, huber: 3.4016, swd: 4.9057, ept: 290.8736
    Epoch [22/50], Val Losses: mse: 49.3876, mae: 4.4960, huber: 4.0552, swd: 5.2838, ept: 254.6185
    Epoch [22/50], Test Losses: mse: 46.0731, mae: 4.2849, huber: 3.8479, swd: 5.2844, ept: 254.8580
      Epoch 22 composite train-obj: 3.401589
            Val objective improved 4.1217 → 4.0552, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 38.6387, mae: 3.8148, huber: 3.3864, swd: 4.8050, ept: 292.1984
    Epoch [23/50], Val Losses: mse: 53.6121, mae: 4.7282, huber: 4.2846, swd: 5.1299, ept: 236.3111
    Epoch [23/50], Test Losses: mse: 50.4070, mae: 4.5458, huber: 4.1034, swd: 5.1839, ept: 236.4343
      Epoch 23 composite train-obj: 3.386371
            No improvement (4.2846), counter 1/5
    Epoch [24/50], Train Losses: mse: 38.3031, mae: 3.7835, huber: 3.3561, swd: 4.6840, ept: 292.9629
    Epoch [24/50], Val Losses: mse: 51.4493, mae: 4.6022, huber: 4.1603, swd: 4.7881, ept: 248.2746
    Epoch [24/50], Test Losses: mse: 48.5561, mae: 4.4332, huber: 3.9932, swd: 4.9187, ept: 253.3215
      Epoch 24 composite train-obj: 3.356120
            No improvement (4.1603), counter 2/5
    Epoch [25/50], Train Losses: mse: 37.9851, mae: 3.7571, huber: 3.3308, swd: 4.5344, ept: 295.5837
    Epoch [25/50], Val Losses: mse: 52.5835, mae: 4.6345, huber: 4.1907, swd: 4.3692, ept: 251.3722
    Epoch [25/50], Test Losses: mse: 49.9706, mae: 4.4650, huber: 4.0225, swd: 4.3959, ept: 249.3175
      Epoch 25 composite train-obj: 3.330799
            No improvement (4.1907), counter 3/5
    Epoch [26/50], Train Losses: mse: 38.9564, mae: 3.8309, huber: 3.4019, swd: 4.7603, ept: 288.9130
    Epoch [26/50], Val Losses: mse: 52.4069, mae: 4.6660, huber: 4.2225, swd: 5.2474, ept: 241.6368
    Epoch [26/50], Test Losses: mse: 49.0753, mae: 4.4622, huber: 4.0200, swd: 5.3446, ept: 241.4277
      Epoch 26 composite train-obj: 3.401940
            No improvement (4.2225), counter 4/5
    Epoch [27/50], Train Losses: mse: 37.5211, mae: 3.7134, huber: 3.2884, swd: 4.4157, ept: 298.7021
    Epoch [27/50], Val Losses: mse: 53.6482, mae: 4.7320, huber: 4.2880, swd: 4.6046, ept: 237.5380
    Epoch [27/50], Test Losses: mse: 50.9852, mae: 4.5714, huber: 4.1284, swd: 4.6482, ept: 238.3410
      Epoch 27 composite train-obj: 3.288370
    Epoch [27/50], Test Losses: mse: 46.0735, mae: 4.2850, huber: 3.8479, swd: 5.2843, ept: 254.8651
    Best round's Test MSE: 46.0731, MAE: 4.2849, SWD: 5.2844
    Best round's Validation MSE: 49.3876, MAE: 4.4960, SWD: 5.2838
    Best round's Test verification MSE : 46.0735, MAE: 4.2850, SWD: 5.2843
    Time taken: 233.00 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 67.2307, mae: 6.1528, huber: 5.6759, swd: 36.5080, ept: 61.7013
    Epoch [1/50], Val Losses: mse: 63.0070, mae: 5.9583, huber: 5.4833, swd: 26.6607, ept: 68.7780
    Epoch [1/50], Test Losses: mse: 62.5641, mae: 5.9160, huber: 5.4409, swd: 26.7479, ept: 64.8548
      Epoch 1 composite train-obj: 5.675941
            Val objective improved inf → 5.4833, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 53.4328, mae: 5.2197, huber: 4.7534, swd: 17.1796, ept: 135.7044
    Epoch [2/50], Val Losses: mse: 58.0785, mae: 5.5416, huber: 5.0711, swd: 16.3594, ept: 128.7512
    Epoch [2/50], Test Losses: mse: 56.1663, mae: 5.3877, huber: 4.9185, swd: 15.5549, ept: 132.4310
      Epoch 2 composite train-obj: 4.753402
            Val objective improved 5.4833 → 5.0711, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.7902, mae: 4.9735, huber: 4.5118, swd: 13.7264, ept: 174.7321
    Epoch [3/50], Val Losses: mse: 62.7351, mae: 5.7385, huber: 5.2698, swd: 13.3824, ept: 136.0321
    Epoch [3/50], Test Losses: mse: 61.4412, mae: 5.6395, huber: 5.1709, swd: 12.8576, ept: 133.9215
      Epoch 3 composite train-obj: 4.511838
            No improvement (5.2698), counter 1/5
    Epoch [4/50], Train Losses: mse: 50.0352, mae: 4.8994, huber: 4.4401, swd: 11.6420, ept: 186.1136
    Epoch [4/50], Val Losses: mse: 52.8476, mae: 5.0388, huber: 4.5775, swd: 11.1128, ept: 180.8430
    Epoch [4/50], Test Losses: mse: 52.2760, mae: 4.9518, huber: 4.4915, swd: 10.3956, ept: 183.2813
      Epoch 4 composite train-obj: 4.440067
            Val objective improved 5.0711 → 4.5775, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 47.6881, mae: 4.6754, huber: 4.2213, swd: 10.3795, ept: 209.2898
    Epoch [5/50], Val Losses: mse: 56.4602, mae: 5.2908, huber: 4.8268, swd: 12.2349, ept: 173.0489
    Epoch [5/50], Test Losses: mse: 53.6786, mae: 5.0806, huber: 4.6192, swd: 11.7546, ept: 179.3616
      Epoch 5 composite train-obj: 4.221285
            No improvement (4.8268), counter 1/5
    Epoch [6/50], Train Losses: mse: 47.1145, mae: 4.6261, huber: 4.1731, swd: 10.5242, ept: 214.3524
    Epoch [6/50], Val Losses: mse: 53.8276, mae: 5.0421, huber: 4.5822, swd: 8.6369, ept: 187.6879
    Epoch [6/50], Test Losses: mse: 52.3298, mae: 4.8961, huber: 4.4378, swd: 8.0643, ept: 192.2651
      Epoch 6 composite train-obj: 4.173142
            No improvement (4.5822), counter 2/5
    Epoch [7/50], Train Losses: mse: 46.4061, mae: 4.5477, huber: 4.0972, swd: 9.8391, ept: 222.7737
    Epoch [7/50], Val Losses: mse: 54.9872, mae: 5.0881, huber: 4.6302, swd: 9.4008, ept: 191.6641
    Epoch [7/50], Test Losses: mse: 52.5726, mae: 4.9047, huber: 4.4487, swd: 8.9105, ept: 196.4461
      Epoch 7 composite train-obj: 4.097207
            No improvement (4.6302), counter 3/5
    Epoch [8/50], Train Losses: mse: 45.4643, mae: 4.4520, huber: 4.0043, swd: 8.7437, ept: 231.4606
    Epoch [8/50], Val Losses: mse: 53.3746, mae: 4.9565, huber: 4.5019, swd: 8.1902, ept: 200.1836
    Epoch [8/50], Test Losses: mse: 50.9600, mae: 4.7826, huber: 4.3297, swd: 7.8309, ept: 202.6494
      Epoch 8 composite train-obj: 4.004268
            Val objective improved 4.5775 → 4.5019, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 43.8820, mae: 4.3110, huber: 3.8676, swd: 7.9188, ept: 245.5930
    Epoch [9/50], Val Losses: mse: 54.4836, mae: 5.0722, huber: 4.6119, swd: 8.3743, ept: 186.5177
    Epoch [9/50], Test Losses: mse: 52.6036, mae: 4.9241, huber: 4.4656, swd: 7.9921, ept: 189.9844
      Epoch 9 composite train-obj: 3.867597
            No improvement (4.6119), counter 1/5
    Epoch [10/50], Train Losses: mse: 44.0319, mae: 4.3231, huber: 3.8791, swd: 7.9524, ept: 245.7589
    Epoch [10/50], Val Losses: mse: 49.8679, mae: 4.7360, huber: 4.2841, swd: 8.1499, ept: 223.1219
    Epoch [10/50], Test Losses: mse: 48.6343, mae: 4.6267, huber: 4.1753, swd: 7.9285, ept: 222.9652
      Epoch 10 composite train-obj: 3.879059
            Val objective improved 4.5019 → 4.2841, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 42.5218, mae: 4.1902, huber: 3.7501, swd: 7.2126, ept: 257.0009
    Epoch [11/50], Val Losses: mse: 51.2523, mae: 4.7304, huber: 4.2802, swd: 7.0460, ept: 232.2163
    Epoch [11/50], Test Losses: mse: 48.3867, mae: 4.5437, huber: 4.0949, swd: 6.5922, ept: 233.3111
      Epoch 11 composite train-obj: 3.750114
            Val objective improved 4.2841 → 4.2802, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 42.0167, mae: 4.1408, huber: 3.7017, swd: 6.8097, ept: 262.0414
    Epoch [12/50], Val Losses: mse: 51.2705, mae: 4.7779, huber: 4.3257, swd: 8.1173, ept: 217.8641
    Epoch [12/50], Test Losses: mse: 49.3434, mae: 4.6504, huber: 4.1990, swd: 7.9246, ept: 216.4255
      Epoch 12 composite train-obj: 3.701700
            No improvement (4.3257), counter 1/5
    Epoch [13/50], Train Losses: mse: 42.4284, mae: 4.1791, huber: 3.7386, swd: 7.3632, ept: 259.0956
    Epoch [13/50], Val Losses: mse: 63.0031, mae: 5.4449, huber: 4.9815, swd: 8.5113, ept: 173.0865
    Epoch [13/50], Test Losses: mse: 59.1484, mae: 5.2010, huber: 4.7406, swd: 8.4874, ept: 183.5233
      Epoch 13 composite train-obj: 3.738557
            No improvement (4.9815), counter 2/5
    Epoch [14/50], Train Losses: mse: 46.6652, mae: 4.5674, huber: 4.1161, swd: 10.5692, ept: 214.7701
    Epoch [14/50], Val Losses: mse: 51.4205, mae: 4.8008, huber: 4.3493, swd: 9.7841, ept: 210.0253
    Epoch [14/50], Test Losses: mse: 47.8192, mae: 4.5389, huber: 4.0911, swd: 8.0917, ept: 222.0918
      Epoch 14 composite train-obj: 4.116059
            No improvement (4.3493), counter 3/5
    Epoch [15/50], Train Losses: mse: 42.5598, mae: 4.2025, huber: 3.7611, swd: 7.9039, ept: 249.0505
    Epoch [15/50], Val Losses: mse: 53.7363, mae: 4.9250, huber: 4.4705, swd: 8.9561, ept: 213.5045
    Epoch [15/50], Test Losses: mse: 50.2666, mae: 4.6800, huber: 4.2292, swd: 7.8051, ept: 224.4996
      Epoch 15 composite train-obj: 3.761062
            No improvement (4.4705), counter 4/5
    Epoch [16/50], Train Losses: mse: 42.6722, mae: 4.2051, huber: 3.7637, swd: 7.6221, ept: 250.8926
    Epoch [16/50], Val Losses: mse: 52.0843, mae: 4.7659, huber: 4.3173, swd: 7.3988, ept: 222.3188
    Epoch [16/50], Test Losses: mse: 49.3427, mae: 4.5558, huber: 4.1105, swd: 6.4997, ept: 229.7104
      Epoch 16 composite train-obj: 3.763674
    Epoch [16/50], Test Losses: mse: 48.3862, mae: 4.5436, huber: 4.0949, swd: 6.5920, ept: 233.3242
    Best round's Test MSE: 48.3867, MAE: 4.5437, SWD: 6.5922
    Best round's Validation MSE: 51.2523, MAE: 4.7304, SWD: 7.0460
    Best round's Test verification MSE : 48.3862, MAE: 4.5436, SWD: 6.5920
    Time taken: 141.03 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq96_pred720_20250513_2346)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 47.2386 ± 0.9446
      mae: 4.3913 ± 0.1105
      huber: 3.9497 ± 0.1054
      swd: 5.7438 ± 0.6005
      ept: 247.5058 ± 10.0393
      count: 35.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 50.2686 ± 0.7647
      mae: 4.5852 ± 0.1036
      huber: 4.1412 ± 0.0992
      swd: 5.9244 ± 0.7958
      ept: 245.5987 ± 9.6519
      count: 35.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 600.05 seconds
    
    Experiment complete: ACL_lorenz_seq96_pred720_20250513_2346
    Model: ACL
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### TimeMixer

#### 96-96
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 283
    Validation Batches: 40
    Test Batches: 80
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 54.8017, mae: 5.0106, huber: 4.5446, swd: 10.9305, ept: 48.6383
    Epoch [1/50], Val Losses: mse: 28.3732, mae: 3.3598, huber: 2.9166, swd: 10.0071, ept: 69.0356
    Epoch [1/50], Test Losses: mse: 28.2302, mae: 3.3645, huber: 2.9213, swd: 10.2391, ept: 68.1460
      Epoch 1 composite train-obj: 4.544600
            Val objective improved inf → 2.9166, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 24.7471, mae: 3.0765, huber: 2.6374, swd: 8.9119, ept: 72.3807
    Epoch [2/50], Val Losses: mse: 19.5039, mae: 2.5606, huber: 2.1411, swd: 7.6414, ept: 77.2831
    Epoch [2/50], Test Losses: mse: 18.7505, mae: 2.5208, huber: 2.1011, swd: 7.4520, ept: 77.3390
      Epoch 2 composite train-obj: 2.637439
            Val objective improved 2.9166 → 2.1411, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 18.5797, mae: 2.5198, huber: 2.0953, swd: 6.8737, ept: 79.1165
    Epoch [3/50], Val Losses: mse: 14.6313, mae: 2.1137, huber: 1.7086, swd: 5.7709, ept: 82.2796
    Epoch [3/50], Test Losses: mse: 14.0539, mae: 2.0743, huber: 1.6697, swd: 5.5690, ept: 82.5089
      Epoch 3 composite train-obj: 2.095276
            Val objective improved 2.1411 → 1.7086, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 15.0561, mae: 2.1844, huber: 1.7704, swd: 5.5835, ept: 83.0836
    Epoch [4/50], Val Losses: mse: 12.5460, mae: 1.8920, huber: 1.4956, swd: 4.9781, ept: 84.7964
    Epoch [4/50], Test Losses: mse: 11.7709, mae: 1.8326, huber: 1.4370, swd: 4.6195, ept: 85.1338
      Epoch 4 composite train-obj: 1.770376
            Val objective improved 1.7086 → 1.4956, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 13.1152, mae: 1.9766, huber: 1.5707, swd: 4.8550, ept: 85.2338
    Epoch [5/50], Val Losses: mse: 10.5446, mae: 1.6829, huber: 1.2959, swd: 4.1181, ept: 86.3961
    Epoch [5/50], Test Losses: mse: 9.8959, mae: 1.6384, huber: 1.2522, swd: 3.8478, ept: 87.1356
      Epoch 5 composite train-obj: 1.570699
            Val objective improved 1.4956 → 1.2959, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.7393, mae: 1.8285, huber: 1.4291, swd: 4.2831, ept: 86.6070
    Epoch [6/50], Val Losses: mse: 9.6146, mae: 1.5809, huber: 1.2025, swd: 3.6797, ept: 87.4608
    Epoch [6/50], Test Losses: mse: 8.9244, mae: 1.5290, huber: 1.1517, swd: 3.4453, ept: 88.3236
      Epoch 6 composite train-obj: 1.429088
            Val objective improved 1.2959 → 1.2025, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 10.7845, mae: 1.7176, huber: 1.3238, swd: 3.8638, ept: 87.5707
    Epoch [7/50], Val Losses: mse: 8.9836, mae: 1.4933, huber: 1.1209, swd: 3.2233, ept: 88.1661
    Epoch [7/50], Test Losses: mse: 8.3235, mae: 1.4489, huber: 1.0766, swd: 2.9578, ept: 88.9779
      Epoch 7 composite train-obj: 1.323841
            Val objective improved 1.2025 → 1.1209, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.0011, mae: 1.6311, huber: 1.2415, swd: 3.5323, ept: 88.3233
    Epoch [8/50], Val Losses: mse: 8.3687, mae: 1.4216, huber: 1.0520, swd: 3.1282, ept: 88.8267
    Epoch [8/50], Test Losses: mse: 7.6605, mae: 1.3784, huber: 1.0084, swd: 2.8235, ept: 89.5940
      Epoch 8 composite train-obj: 1.241526
            Val objective improved 1.1209 → 1.0520, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 9.4122, mae: 1.5634, huber: 1.1776, swd: 3.2577, ept: 88.9169
    Epoch [9/50], Val Losses: mse: 7.9615, mae: 1.3856, huber: 1.0142, swd: 2.9054, ept: 89.4033
    Epoch [9/50], Test Losses: mse: 7.3253, mae: 1.3490, huber: 0.9769, swd: 2.5953, ept: 90.0846
      Epoch 9 composite train-obj: 1.177579
            Val objective improved 1.0520 → 1.0142, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 8.8480, mae: 1.4943, huber: 1.1128, swd: 3.0209, ept: 89.4288
    Epoch [10/50], Val Losses: mse: 7.1430, mae: 1.2426, huber: 0.8907, swd: 2.7267, ept: 89.7519
    Epoch [10/50], Test Losses: mse: 6.4072, mae: 1.1961, huber: 0.8444, swd: 2.3162, ept: 90.4616
      Epoch 10 composite train-obj: 1.112761
            Val objective improved 1.0142 → 0.8907, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 8.4062, mae: 1.4435, huber: 1.0655, swd: 2.8382, ept: 89.8278
    Epoch [11/50], Val Losses: mse: 6.8626, mae: 1.2314, huber: 0.8778, swd: 2.5199, ept: 90.0564
    Epoch [11/50], Test Losses: mse: 6.1882, mae: 1.1897, huber: 0.8361, swd: 2.1508, ept: 90.8140
      Epoch 11 composite train-obj: 1.065543
            Val objective improved 0.8907 → 0.8778, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 7.9920, mae: 1.3938, huber: 1.0191, swd: 2.6670, ept: 90.1925
    Epoch [12/50], Val Losses: mse: 6.2867, mae: 1.1885, huber: 0.8344, swd: 2.3553, ept: 90.6051
    Epoch [12/50], Test Losses: mse: 5.6634, mae: 1.1575, huber: 0.8024, swd: 2.0208, ept: 91.2498
      Epoch 12 composite train-obj: 1.019147
            Val objective improved 0.8778 → 0.8344, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 7.6480, mae: 1.3533, huber: 0.9814, swd: 2.5125, ept: 90.4626
    Epoch [13/50], Val Losses: mse: 6.1767, mae: 1.1697, huber: 0.8215, swd: 2.1687, ept: 90.6159
    Epoch [13/50], Test Losses: mse: 5.5418, mae: 1.1319, huber: 0.7830, swd: 1.8287, ept: 91.3893
      Epoch 13 composite train-obj: 0.981430
            Val objective improved 0.8344 → 0.8215, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 7.3294, mae: 1.3149, huber: 0.9459, swd: 2.3965, ept: 90.7399
    Epoch [14/50], Val Losses: mse: 6.6006, mae: 1.2137, huber: 0.8620, swd: 2.1350, ept: 90.5251
    Epoch [14/50], Test Losses: mse: 5.9769, mae: 1.1794, huber: 0.8277, swd: 1.8223, ept: 91.3534
      Epoch 14 composite train-obj: 0.945888
            No improvement (0.8620), counter 1/5
    Epoch [15/50], Train Losses: mse: 7.0498, mae: 1.2756, huber: 0.9102, swd: 2.2615, ept: 91.0134
    Epoch [15/50], Val Losses: mse: 5.6317, mae: 1.0961, huber: 0.7559, swd: 1.9142, ept: 91.0738
    Epoch [15/50], Test Losses: mse: 5.0984, mae: 1.0632, huber: 0.7233, swd: 1.6671, ept: 91.8459
      Epoch 15 composite train-obj: 0.910174
            Val objective improved 0.8215 → 0.7559, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 6.8094, mae: 1.2490, huber: 0.8858, swd: 2.1788, ept: 91.2205
    Epoch [16/50], Val Losses: mse: 5.3654, mae: 1.0867, huber: 0.7447, swd: 1.9269, ept: 91.4317
    Epoch [16/50], Test Losses: mse: 4.8336, mae: 1.0604, huber: 0.7174, swd: 1.6223, ept: 92.1054
      Epoch 16 composite train-obj: 0.885780
            Val objective improved 0.7559 → 0.7447, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 6.5656, mae: 1.2152, huber: 0.8551, swd: 2.0794, ept: 91.4454
    Epoch [17/50], Val Losses: mse: 5.1921, mae: 1.0716, huber: 0.7285, swd: 1.7828, ept: 91.4937
    Epoch [17/50], Test Losses: mse: 4.6982, mae: 1.0444, huber: 0.7004, swd: 1.5649, ept: 92.2098
      Epoch 17 composite train-obj: 0.855113
            Val objective improved 0.7447 → 0.7285, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 6.3883, mae: 1.1961, huber: 0.8373, swd: 2.0180, ept: 91.5752
    Epoch [18/50], Val Losses: mse: 5.1455, mae: 1.0374, huber: 0.7027, swd: 1.7484, ept: 91.5673
    Epoch [18/50], Test Losses: mse: 4.6013, mae: 1.0105, huber: 0.6753, swd: 1.4413, ept: 92.3089
      Epoch 18 composite train-obj: 0.837263
            Val objective improved 0.7285 → 0.7027, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 6.1799, mae: 1.1658, huber: 0.8101, swd: 1.9254, ept: 91.7503
    Epoch [19/50], Val Losses: mse: 5.2685, mae: 1.0665, huber: 0.7271, swd: 1.6509, ept: 91.6561
    Epoch [19/50], Test Losses: mse: 4.7383, mae: 1.0356, huber: 0.6961, swd: 1.4161, ept: 92.3329
      Epoch 19 composite train-obj: 0.810092
            No improvement (0.7271), counter 1/5
    Epoch [20/50], Train Losses: mse: 6.0376, mae: 1.1446, huber: 0.7907, swd: 1.8734, ept: 91.8722
    Epoch [20/50], Val Losses: mse: 5.0674, mae: 1.0554, huber: 0.7172, swd: 1.5730, ept: 91.7896
    Epoch [20/50], Test Losses: mse: 4.5869, mae: 1.0300, huber: 0.6911, swd: 1.3488, ept: 92.5134
      Epoch 20 composite train-obj: 0.790664
            No improvement (0.7172), counter 2/5
    Epoch [21/50], Train Losses: mse: 5.8703, mae: 1.1225, huber: 0.7710, swd: 1.8235, ept: 92.0143
    Epoch [21/50], Val Losses: mse: 5.1008, mae: 1.0168, huber: 0.6854, swd: 1.5995, ept: 91.8476
    Epoch [21/50], Test Losses: mse: 4.5604, mae: 0.9875, huber: 0.6555, swd: 1.3243, ept: 92.5358
      Epoch 21 composite train-obj: 0.771022
            Val objective improved 0.7027 → 0.6854, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 5.7059, mae: 1.0999, huber: 0.7507, swd: 1.7628, ept: 92.1434
    Epoch [22/50], Val Losses: mse: 4.5956, mae: 0.9648, huber: 0.6376, swd: 1.5535, ept: 92.1845
    Epoch [22/50], Test Losses: mse: 4.0414, mae: 0.9328, huber: 0.6056, swd: 1.2516, ept: 92.8121
      Epoch 22 composite train-obj: 0.750710
            Val objective improved 0.6854 → 0.6376, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 5.6060, mae: 1.0872, huber: 0.7396, swd: 1.7015, ept: 92.2163
    Epoch [23/50], Val Losses: mse: 5.1336, mae: 1.0133, huber: 0.6862, swd: 1.5404, ept: 91.8856
    Epoch [23/50], Test Losses: mse: 4.6408, mae: 0.9888, huber: 0.6600, swd: 1.2697, ept: 92.5625
      Epoch 23 composite train-obj: 0.739555
            No improvement (0.6862), counter 1/5
    Epoch [24/50], Train Losses: mse: 5.4855, mae: 1.0660, huber: 0.7206, swd: 1.6706, ept: 92.3257
    Epoch [24/50], Val Losses: mse: 4.9212, mae: 1.0166, huber: 0.6847, swd: 1.4638, ept: 92.1952
    Epoch [24/50], Test Losses: mse: 4.4022, mae: 0.9861, huber: 0.6531, swd: 1.2192, ept: 92.7676
      Epoch 24 composite train-obj: 0.720638
            No improvement (0.6847), counter 2/5
    Epoch [25/50], Train Losses: mse: 5.3869, mae: 1.0544, huber: 0.7107, swd: 1.6207, ept: 92.3685
    Epoch [25/50], Val Losses: mse: 5.1618, mae: 1.0184, huber: 0.6914, swd: 1.4439, ept: 92.1273
    Epoch [25/50], Test Losses: mse: 4.6162, mae: 0.9859, huber: 0.6590, swd: 1.1847, ept: 92.5934
      Epoch 25 composite train-obj: 0.710721
            No improvement (0.6914), counter 3/5
    Epoch [26/50], Train Losses: mse: 5.2831, mae: 1.0411, huber: 0.6988, swd: 1.5980, ept: 92.4429
    Epoch [26/50], Val Losses: mse: 5.4790, mae: 1.1219, huber: 0.7777, swd: 1.3713, ept: 91.8600
    Epoch [26/50], Test Losses: mse: 5.0574, mae: 1.1036, huber: 0.7578, swd: 1.2077, ept: 92.4135
      Epoch 26 composite train-obj: 0.698820
            No improvement (0.7777), counter 4/5
    Epoch [27/50], Train Losses: mse: 5.1415, mae: 1.0218, huber: 0.6817, swd: 1.5470, ept: 92.5480
    Epoch [27/50], Val Losses: mse: 4.4468, mae: 0.9542, huber: 0.6304, swd: 1.4008, ept: 92.5837
    Epoch [27/50], Test Losses: mse: 3.9445, mae: 0.9257, huber: 0.6008, swd: 1.1525, ept: 93.0581
      Epoch 27 composite train-obj: 0.681693
            Val objective improved 0.6376 → 0.6304, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 5.0589, mae: 1.0096, huber: 0.6705, swd: 1.5361, ept: 92.6052
    Epoch [28/50], Val Losses: mse: 4.9474, mae: 1.0229, huber: 0.6940, swd: 1.3697, ept: 92.1040
    Epoch [28/50], Test Losses: mse: 4.4395, mae: 0.9967, huber: 0.6670, swd: 1.1518, ept: 92.7087
      Epoch 28 composite train-obj: 0.670535
            No improvement (0.6940), counter 1/5
    Epoch [29/50], Train Losses: mse: 4.9535, mae: 0.9939, huber: 0.6571, swd: 1.4793, ept: 92.6958
    Epoch [29/50], Val Losses: mse: 4.5316, mae: 0.9640, huber: 0.6368, swd: 1.3621, ept: 92.4980
    Epoch [29/50], Test Losses: mse: 4.0182, mae: 0.9352, huber: 0.6074, swd: 1.0973, ept: 93.0436
      Epoch 29 composite train-obj: 0.657146
            No improvement (0.6368), counter 2/5
    Epoch [30/50], Train Losses: mse: 4.9084, mae: 0.9878, huber: 0.6518, swd: 1.4782, ept: 92.7477
    Epoch [30/50], Val Losses: mse: 4.8197, mae: 0.9957, huber: 0.6706, swd: 1.2981, ept: 92.0852
    Epoch [30/50], Test Losses: mse: 4.2888, mae: 0.9653, huber: 0.6400, swd: 1.0794, ept: 92.8121
      Epoch 30 composite train-obj: 0.651801
            No improvement (0.6706), counter 3/5
    Epoch [31/50], Train Losses: mse: 4.7964, mae: 0.9712, huber: 0.6375, swd: 1.4253, ept: 92.8213
    Epoch [31/50], Val Losses: mse: 5.2938, mae: 1.0378, huber: 0.7128, swd: 1.3802, ept: 91.9917
    Epoch [31/50], Test Losses: mse: 4.8894, mae: 1.0182, huber: 0.6925, swd: 1.2045, ept: 92.5188
      Epoch 31 composite train-obj: 0.637546
            No improvement (0.7128), counter 4/5
    Epoch [32/50], Train Losses: mse: 4.7159, mae: 0.9618, huber: 0.6295, swd: 1.4117, ept: 92.8398
    Epoch [32/50], Val Losses: mse: 5.1406, mae: 1.0334, huber: 0.7029, swd: 1.3186, ept: 91.8960
    Epoch [32/50], Test Losses: mse: 4.6835, mae: 1.0067, huber: 0.6763, swd: 1.1477, ept: 92.5182
      Epoch 32 composite train-obj: 0.629507
    Epoch [32/50], Test Losses: mse: 3.9445, mae: 0.9257, huber: 0.6008, swd: 1.1525, ept: 93.0581
    Best round's Test MSE: 3.9445, MAE: 0.9257, SWD: 1.1525
    Best round's Validation MSE: 4.4468, MAE: 0.9542, SWD: 1.4008
    Best round's Test verification MSE : 3.9445, MAE: 0.9257, SWD: 1.1525
    Time taken: 260.38 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 56.8242, mae: 5.1899, huber: 4.7214, swd: 11.8560, ept: 46.6807
    Epoch [1/50], Val Losses: mse: 30.2945, mae: 3.5535, huber: 3.1062, swd: 11.0980, ept: 65.7148
    Epoch [1/50], Test Losses: mse: 30.3862, mae: 3.5733, huber: 3.1260, swd: 11.5114, ept: 64.9554
      Epoch 1 composite train-obj: 4.721449
            Val objective improved inf → 3.1062, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 24.9249, mae: 3.0907, huber: 2.6514, swd: 9.2692, ept: 72.0398
    Epoch [2/50], Val Losses: mse: 19.1886, mae: 2.5228, huber: 2.1068, swd: 7.5487, ept: 77.7638
    Epoch [2/50], Test Losses: mse: 18.4711, mae: 2.4801, huber: 2.0642, swd: 7.4261, ept: 77.6854
      Epoch 2 composite train-obj: 2.651426
            Val objective improved 3.1062 → 2.1068, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 17.7017, mae: 2.4246, huber: 2.0028, swd: 6.4595, ept: 80.3543
    Epoch [3/50], Val Losses: mse: 14.1271, mae: 2.0221, huber: 1.6210, swd: 5.3567, ept: 83.4472
    Epoch [3/50], Test Losses: mse: 13.3277, mae: 1.9737, huber: 1.5722, swd: 5.1071, ept: 83.7731
      Epoch 3 composite train-obj: 2.002759
            Val objective improved 2.1068 → 1.6210, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 14.1698, mae: 2.0642, huber: 1.6555, swd: 5.1086, ept: 84.3429
    Epoch [4/50], Val Losses: mse: 11.5165, mae: 1.7429, huber: 1.3587, swd: 4.5103, ept: 85.9923
    Epoch [4/50], Test Losses: mse: 10.7489, mae: 1.6928, huber: 1.3096, swd: 4.2859, ept: 86.4304
      Epoch 4 composite train-obj: 1.655527
            Val objective improved 1.6210 → 1.3587, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 12.3838, mae: 1.8658, huber: 1.4666, swd: 4.4758, ept: 86.2604
    Epoch [5/50], Val Losses: mse: 10.1923, mae: 1.6171, huber: 1.2356, swd: 3.8609, ept: 87.3468
    Epoch [5/50], Test Losses: mse: 9.5969, mae: 1.5823, huber: 1.2020, swd: 3.6667, ept: 88.0555
      Epoch 5 composite train-obj: 1.466564
            Val objective improved 1.3587 → 1.2356, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.1011, mae: 1.7198, huber: 1.3282, swd: 3.9818, ept: 87.5991
    Epoch [6/50], Val Losses: mse: 9.1104, mae: 1.4779, huber: 1.1054, swd: 3.5256, ept: 88.2986
    Epoch [6/50], Test Losses: mse: 8.3541, mae: 1.4342, huber: 1.0615, swd: 3.2224, ept: 89.0110
      Epoch 6 composite train-obj: 1.328222
            Val objective improved 1.2356 → 1.1054, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 10.1266, mae: 1.6066, huber: 1.2217, swd: 3.5940, ept: 88.5216
    Epoch [7/50], Val Losses: mse: 8.2173, mae: 1.3502, huber: 0.9891, swd: 3.1534, ept: 88.8202
    Epoch [7/50], Test Losses: mse: 7.4556, mae: 1.3047, huber: 0.9443, swd: 2.8552, ept: 89.8657
      Epoch 7 composite train-obj: 1.221692
            Val objective improved 1.1054 → 0.9891, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 9.3759, mae: 1.5161, huber: 1.1372, swd: 3.2709, ept: 89.2276
    Epoch [8/50], Val Losses: mse: 7.5773, mae: 1.2639, huber: 0.9125, swd: 2.8536, ept: 89.4109
    Epoch [8/50], Test Losses: mse: 6.8959, mae: 1.2267, huber: 0.8740, swd: 2.5501, ept: 90.4518
      Epoch 8 composite train-obj: 1.137203
            Val objective improved 0.9891 → 0.9125, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 8.7613, mae: 1.4459, huber: 1.0720, swd: 3.0354, ept: 89.8057
    Epoch [9/50], Val Losses: mse: 7.0783, mae: 1.2074, huber: 0.8589, swd: 2.7694, ept: 89.9445
    Epoch [9/50], Test Losses: mse: 6.3878, mae: 1.1638, huber: 0.8157, swd: 2.4251, ept: 90.6781
      Epoch 9 composite train-obj: 1.072009
            Val objective improved 0.9125 → 0.8589, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 8.2606, mae: 1.3841, huber: 1.0149, swd: 2.8255, ept: 90.2163
    Epoch [10/50], Val Losses: mse: 6.4765, mae: 1.1523, huber: 0.8074, swd: 2.4747, ept: 90.3234
    Epoch [10/50], Test Losses: mse: 5.8050, mae: 1.1158, huber: 0.7695, swd: 2.1623, ept: 91.2118
      Epoch 10 composite train-obj: 1.014949
            Val objective improved 0.8589 → 0.8074, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 7.8340, mae: 1.3297, huber: 0.9650, swd: 2.6478, ept: 90.5508
    Epoch [11/50], Val Losses: mse: 6.1442, mae: 1.1407, huber: 0.7940, swd: 2.2619, ept: 90.8260
    Epoch [11/50], Test Losses: mse: 5.5376, mae: 1.1050, huber: 0.7575, swd: 2.0022, ept: 91.5796
      Epoch 11 composite train-obj: 0.965013
            Val objective improved 0.8074 → 0.7940, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 7.4869, mae: 1.2909, huber: 0.9292, swd: 2.5027, ept: 90.9015
    Epoch [12/50], Val Losses: mse: 5.7279, mae: 1.0596, huber: 0.7268, swd: 2.1240, ept: 91.0382
    Epoch [12/50], Test Losses: mse: 5.1361, mae: 1.0218, huber: 0.6897, swd: 1.8636, ept: 91.8110
      Epoch 12 composite train-obj: 0.929247
            Val objective improved 0.7940 → 0.7268, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 7.1804, mae: 1.2548, huber: 0.8959, swd: 2.3955, ept: 91.1115
    Epoch [13/50], Val Losses: mse: 5.5012, mae: 1.0401, huber: 0.7073, swd: 2.1323, ept: 91.2387
    Epoch [13/50], Test Losses: mse: 4.8775, mae: 1.0018, huber: 0.6694, swd: 1.8242, ept: 91.9623
      Epoch 13 composite train-obj: 0.895925
            Val objective improved 0.7268 → 0.7073, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 6.9290, mae: 1.2192, huber: 0.8642, swd: 2.2961, ept: 91.3279
    Epoch [14/50], Val Losses: mse: 5.0775, mae: 0.9795, huber: 0.6543, swd: 1.9698, ept: 91.6247
    Epoch [14/50], Test Losses: mse: 4.4822, mae: 0.9436, huber: 0.6179, swd: 1.6720, ept: 92.2800
      Epoch 14 composite train-obj: 0.864189
            Val objective improved 0.7073 → 0.6543, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 6.6450, mae: 1.1840, huber: 0.8320, swd: 2.1950, ept: 91.5383
    Epoch [15/50], Val Losses: mse: 5.0685, mae: 0.9756, huber: 0.6497, swd: 1.8755, ept: 91.6969
    Epoch [15/50], Test Losses: mse: 4.4152, mae: 0.9389, huber: 0.6118, swd: 1.5794, ept: 92.3674
      Epoch 15 composite train-obj: 0.832046
            Val objective improved 0.6543 → 0.6497, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 6.4590, mae: 1.1595, huber: 0.8098, swd: 2.1073, ept: 91.7060
    Epoch [16/50], Val Losses: mse: 4.9050, mae: 0.9539, huber: 0.6275, swd: 1.9189, ept: 91.8926
    Epoch [16/50], Test Losses: mse: 4.2809, mae: 0.9177, huber: 0.5904, swd: 1.5580, ept: 92.4463
      Epoch 16 composite train-obj: 0.809771
            Val objective improved 0.6497 → 0.6275, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 6.2778, mae: 1.1344, huber: 0.7875, swd: 2.0525, ept: 91.8454
    Epoch [17/50], Val Losses: mse: 4.5465, mae: 0.8959, huber: 0.5841, swd: 1.8127, ept: 92.1201
    Epoch [17/50], Test Losses: mse: 3.9565, mae: 0.8600, huber: 0.5472, swd: 1.4896, ept: 92.7087
      Epoch 17 composite train-obj: 0.787456
            Val objective improved 0.6275 → 0.5841, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 6.0971, mae: 1.1091, huber: 0.7652, swd: 1.9840, ept: 91.9733
    Epoch [18/50], Val Losses: mse: 4.4012, mae: 0.8827, huber: 0.5723, swd: 1.5778, ept: 92.2281
    Epoch [18/50], Test Losses: mse: 3.8283, mae: 0.8475, huber: 0.5380, swd: 1.3687, ept: 92.8863
      Epoch 18 composite train-obj: 0.765173
            Val objective improved 0.5841 → 0.5723, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 5.9218, mae: 1.0912, huber: 0.7487, swd: 1.9013, ept: 92.0869
    Epoch [19/50], Val Losses: mse: 4.5013, mae: 0.8804, huber: 0.5699, swd: 1.7348, ept: 92.3824
    Epoch [19/50], Test Losses: mse: 3.8433, mae: 0.8418, huber: 0.5305, swd: 1.3620, ept: 92.9271
      Epoch 19 composite train-obj: 0.748720
            Val objective improved 0.5723 → 0.5699, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 5.7625, mae: 1.0659, huber: 0.7263, swd: 1.8309, ept: 92.2279
    Epoch [20/50], Val Losses: mse: 4.2619, mae: 0.8860, huber: 0.5716, swd: 1.6468, ept: 92.5193
    Epoch [20/50], Test Losses: mse: 3.7061, mae: 0.8532, huber: 0.5384, swd: 1.3576, ept: 93.0486
      Epoch 20 composite train-obj: 0.726324
            No improvement (0.5716), counter 1/5
    Epoch [21/50], Train Losses: mse: 5.6689, mae: 1.0537, huber: 0.7156, swd: 1.8139, ept: 92.3337
    Epoch [21/50], Val Losses: mse: 3.9690, mae: 0.8249, huber: 0.5199, swd: 1.4358, ept: 92.7378
    Epoch [21/50], Test Losses: mse: 3.3803, mae: 0.7891, huber: 0.4843, swd: 1.1935, ept: 93.2458
      Epoch 21 composite train-obj: 0.715643
            Val objective improved 0.5699 → 0.5199, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 5.5357, mae: 1.0342, huber: 0.6983, swd: 1.7570, ept: 92.4277
    Epoch [22/50], Val Losses: mse: 3.7439, mae: 0.8231, huber: 0.5169, swd: 1.3523, ept: 92.7980
    Epoch [22/50], Test Losses: mse: 3.2863, mae: 0.7992, huber: 0.4911, swd: 1.1948, ept: 93.2775
      Epoch 22 composite train-obj: 0.698286
            Val objective improved 0.5199 → 0.5169, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 5.4116, mae: 1.0190, huber: 0.6847, swd: 1.7141, ept: 92.5127
    Epoch [23/50], Val Losses: mse: 3.9615, mae: 0.8544, huber: 0.5391, swd: 1.4181, ept: 92.8049
    Epoch [23/50], Test Losses: mse: 3.4408, mae: 0.8233, huber: 0.5081, swd: 1.1995, ept: 93.2620
      Epoch 23 composite train-obj: 0.684737
            No improvement (0.5391), counter 1/5
    Epoch [24/50], Train Losses: mse: 5.3100, mae: 1.0048, huber: 0.6724, swd: 1.6695, ept: 92.5887
    Epoch [24/50], Val Losses: mse: 3.5451, mae: 0.7947, huber: 0.4894, swd: 1.2933, ept: 93.0607
    Epoch [24/50], Test Losses: mse: 3.0825, mae: 0.7679, huber: 0.4623, swd: 1.1248, ept: 93.5048
      Epoch 24 composite train-obj: 0.672435
            Val objective improved 0.5169 → 0.4894, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 5.2382, mae: 0.9954, huber: 0.6643, swd: 1.6564, ept: 92.6424
    Epoch [25/50], Val Losses: mse: 3.4798, mae: 0.7908, huber: 0.4870, swd: 1.2979, ept: 93.1172
    Epoch [25/50], Test Losses: mse: 3.0602, mae: 0.7680, huber: 0.4630, swd: 1.1182, ept: 93.5625
      Epoch 25 composite train-obj: 0.664302
            Val objective improved 0.4894 → 0.4870, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 5.1355, mae: 0.9815, huber: 0.6523, swd: 1.6124, ept: 92.7160
    Epoch [26/50], Val Losses: mse: 3.4808, mae: 0.7848, huber: 0.4834, swd: 1.3276, ept: 93.1542
    Epoch [26/50], Test Losses: mse: 2.9758, mae: 0.7531, huber: 0.4510, swd: 1.0965, ept: 93.5836
      Epoch 26 composite train-obj: 0.652300
            Val objective improved 0.4870 → 0.4834, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 5.0219, mae: 0.9664, huber: 0.6389, swd: 1.5752, ept: 92.7643
    Epoch [27/50], Val Losses: mse: 3.2477, mae: 0.7323, huber: 0.4431, swd: 1.2387, ept: 93.4076
    Epoch [27/50], Test Losses: mse: 2.8129, mae: 0.6989, huber: 0.4109, swd: 1.0264, ept: 93.7660
      Epoch 27 composite train-obj: 0.638920
            Val objective improved 0.4834 → 0.4431, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 4.9847, mae: 0.9598, huber: 0.6333, swd: 1.5476, ept: 92.8387
    Epoch [28/50], Val Losses: mse: 3.1763, mae: 0.7424, huber: 0.4475, swd: 1.2532, ept: 93.2539
    Epoch [28/50], Test Losses: mse: 2.7485, mae: 0.7140, huber: 0.4188, swd: 1.0364, ept: 93.6720
      Epoch 28 composite train-obj: 0.633319
            No improvement (0.4475), counter 1/5
    Epoch [29/50], Train Losses: mse: 4.8807, mae: 0.9474, huber: 0.6227, swd: 1.5236, ept: 92.9138
    Epoch [29/50], Val Losses: mse: 3.1544, mae: 0.7348, huber: 0.4451, swd: 1.2103, ept: 93.4652
    Epoch [29/50], Test Losses: mse: 2.7091, mae: 0.7031, huber: 0.4142, swd: 0.9805, ept: 93.8152
      Epoch 29 composite train-obj: 0.622654
            No improvement (0.4451), counter 2/5
    Epoch [30/50], Train Losses: mse: 4.7826, mae: 0.9328, huber: 0.6099, swd: 1.4928, ept: 92.9742
    Epoch [30/50], Val Losses: mse: 3.1105, mae: 0.7526, huber: 0.4543, swd: 1.1136, ept: 93.4523
    Epoch [30/50], Test Losses: mse: 2.7023, mae: 0.7262, huber: 0.4274, swd: 0.9357, ept: 93.7724
      Epoch 30 composite train-obj: 0.609922
            No improvement (0.4543), counter 3/5
    Epoch [31/50], Train Losses: mse: 4.7172, mae: 0.9205, huber: 0.5999, swd: 1.4538, ept: 92.9866
    Epoch [31/50], Val Losses: mse: 2.9199, mae: 0.7072, huber: 0.4206, swd: 1.0727, ept: 93.6687
    Epoch [31/50], Test Losses: mse: 2.5457, mae: 0.6811, huber: 0.3950, swd: 0.9424, ept: 93.9699
      Epoch 31 composite train-obj: 0.599910
            Val objective improved 0.4431 → 0.4206, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 4.6910, mae: 0.9142, huber: 0.5946, swd: 1.4542, ept: 93.0510
    Epoch [32/50], Val Losses: mse: 2.9830, mae: 0.7068, huber: 0.4204, swd: 1.0998, ept: 93.5329
    Epoch [32/50], Test Losses: mse: 2.5468, mae: 0.6805, huber: 0.3938, swd: 0.9254, ept: 93.9138
      Epoch 32 composite train-obj: 0.594569
            Val objective improved 0.4206 → 0.4204, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 4.6326, mae: 0.9101, huber: 0.5907, swd: 1.4304, ept: 93.0582
    Epoch [33/50], Val Losses: mse: 2.8931, mae: 0.6964, huber: 0.4126, swd: 1.1093, ept: 93.4967
    Epoch [33/50], Test Losses: mse: 2.4562, mae: 0.6651, huber: 0.3818, swd: 0.9049, ept: 93.8691
      Epoch 33 composite train-obj: 0.590684
            Val objective improved 0.4204 → 0.4126, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 4.5472, mae: 0.8988, huber: 0.5812, swd: 1.3945, ept: 93.0836
    Epoch [34/50], Val Losses: mse: 2.8280, mae: 0.6871, huber: 0.4023, swd: 1.0361, ept: 93.6769
    Epoch [34/50], Test Losses: mse: 2.3971, mae: 0.6586, huber: 0.3731, swd: 0.8497, ept: 94.0590
      Epoch 34 composite train-obj: 0.581196
            Val objective improved 0.4126 → 0.4023, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 4.4795, mae: 0.8878, huber: 0.5723, swd: 1.3772, ept: 93.1194
    Epoch [35/50], Val Losses: mse: 2.9673, mae: 0.7209, huber: 0.4309, swd: 1.0577, ept: 93.5766
    Epoch [35/50], Test Losses: mse: 2.6093, mae: 0.6992, huber: 0.4083, swd: 0.9154, ept: 93.9480
      Epoch 35 composite train-obj: 0.572285
            No improvement (0.4309), counter 1/5
    Epoch [36/50], Train Losses: mse: 4.4570, mae: 0.8857, huber: 0.5700, swd: 1.3663, ept: 93.1921
    Epoch [36/50], Val Losses: mse: 2.7700, mae: 0.6934, huber: 0.4105, swd: 0.9982, ept: 93.8593
    Epoch [36/50], Test Losses: mse: 2.4118, mae: 0.6654, huber: 0.3836, swd: 0.8440, ept: 94.1882
      Epoch 36 composite train-obj: 0.570013
            No improvement (0.4105), counter 2/5
    Epoch [37/50], Train Losses: mse: 4.3778, mae: 0.8748, huber: 0.5608, swd: 1.3432, ept: 93.2234
    Epoch [37/50], Val Losses: mse: 2.9162, mae: 0.7419, huber: 0.4437, swd: 0.9653, ept: 93.7254
    Epoch [37/50], Test Losses: mse: 2.5809, mae: 0.7235, huber: 0.4240, swd: 0.8807, ept: 94.1273
      Epoch 37 composite train-obj: 0.560820
            No improvement (0.4437), counter 3/5
    Epoch [38/50], Train Losses: mse: 4.3048, mae: 0.8669, huber: 0.5536, swd: 1.3193, ept: 93.2883
    Epoch [38/50], Val Losses: mse: 2.6035, mae: 0.6800, huber: 0.3971, swd: 0.9004, ept: 93.9566
    Epoch [38/50], Test Losses: mse: 2.2784, mae: 0.6590, huber: 0.3752, swd: 0.7901, ept: 94.2306
      Epoch 38 composite train-obj: 0.553648
            Val objective improved 0.4023 → 0.3971, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 4.2939, mae: 0.8632, huber: 0.5509, swd: 1.3044, ept: 93.2919
    Epoch [39/50], Val Losses: mse: 2.6223, mae: 0.7095, huber: 0.4100, swd: 0.9584, ept: 93.9334
    Epoch [39/50], Test Losses: mse: 2.3034, mae: 0.6886, huber: 0.3882, swd: 0.8600, ept: 94.3343
      Epoch 39 composite train-obj: 0.550877
            No improvement (0.4100), counter 1/5
    Epoch [40/50], Train Losses: mse: 4.2206, mae: 0.8507, huber: 0.5403, swd: 1.2754, ept: 93.3400
    Epoch [40/50], Val Losses: mse: 2.6937, mae: 0.6876, huber: 0.4034, swd: 0.9783, ept: 93.7926
    Epoch [40/50], Test Losses: mse: 2.3265, mae: 0.6642, huber: 0.3796, swd: 0.8285, ept: 94.1400
      Epoch 40 composite train-obj: 0.540291
            No improvement (0.4034), counter 2/5
    Epoch [41/50], Train Losses: mse: 4.2151, mae: 0.8519, huber: 0.5412, swd: 1.2607, ept: 93.3495
    Epoch [41/50], Val Losses: mse: 2.4098, mae: 0.6289, huber: 0.3548, swd: 0.9485, ept: 94.0893
    Epoch [41/50], Test Losses: mse: 2.0606, mae: 0.6031, huber: 0.3289, swd: 0.7939, ept: 94.4281
      Epoch 41 composite train-obj: 0.541159
            Val objective improved 0.3971 → 0.3548, saving checkpoint.
    Epoch [42/50], Train Losses: mse: 4.1598, mae: 0.8446, huber: 0.5352, swd: 1.2464, ept: 93.3900
    Epoch [42/50], Val Losses: mse: 2.4362, mae: 0.6292, huber: 0.3582, swd: 0.9658, ept: 94.1007
    Epoch [42/50], Test Losses: mse: 2.0548, mae: 0.6114, huber: 0.3377, swd: 0.7691, ept: 94.4180
      Epoch 42 composite train-obj: 0.535221
            No improvement (0.3582), counter 1/5
    Epoch [43/50], Train Losses: mse: 4.0887, mae: 0.8341, huber: 0.5261, swd: 1.2366, ept: 93.4402
    Epoch [43/50], Val Losses: mse: 2.4615, mae: 0.6636, huber: 0.3797, swd: 0.9474, ept: 94.0847
    Epoch [43/50], Test Losses: mse: 2.1390, mae: 0.6365, huber: 0.3537, swd: 0.7797, ept: 94.3243
      Epoch 43 composite train-obj: 0.526134
            No improvement (0.3797), counter 2/5
    Epoch [44/50], Train Losses: mse: 4.0444, mae: 0.8292, huber: 0.5219, swd: 1.2200, ept: 93.4603
    Epoch [44/50], Val Losses: mse: 2.4425, mae: 0.6386, huber: 0.3651, swd: 0.9192, ept: 94.0772
    Epoch [44/50], Test Losses: mse: 2.1746, mae: 0.6146, huber: 0.3429, swd: 0.7948, ept: 94.3471
      Epoch 44 composite train-obj: 0.521948
            No improvement (0.3651), counter 3/5
    Epoch [45/50], Train Losses: mse: 4.0081, mae: 0.8226, huber: 0.5169, swd: 1.2085, ept: 93.4390
    Epoch [45/50], Val Losses: mse: 2.3427, mae: 0.6411, huber: 0.3621, swd: 0.7870, ept: 94.1415
    Epoch [45/50], Test Losses: mse: 2.0913, mae: 0.6195, huber: 0.3414, swd: 0.6937, ept: 94.3816
      Epoch 45 composite train-obj: 0.516874
            No improvement (0.3621), counter 4/5
    Epoch [46/50], Train Losses: mse: 3.9820, mae: 0.8170, huber: 0.5119, swd: 1.1854, ept: 93.4820
    Epoch [46/50], Val Losses: mse: 2.5231, mae: 0.6338, huber: 0.3649, swd: 0.9004, ept: 94.0270
    Epoch [46/50], Test Losses: mse: 2.1683, mae: 0.6136, huber: 0.3427, swd: 0.7299, ept: 94.3092
      Epoch 46 composite train-obj: 0.511854
    Epoch [46/50], Test Losses: mse: 2.0606, mae: 0.6031, huber: 0.3289, swd: 0.7939, ept: 94.4281
    Best round's Test MSE: 2.0606, MAE: 0.6031, SWD: 0.7939
    Best round's Validation MSE: 2.4098, MAE: 0.6289, SWD: 0.9485
    Best round's Test verification MSE : 2.0606, MAE: 0.6031, SWD: 0.7939
    Time taken: 376.86 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 59.5386, mae: 5.2670, huber: 4.7989, swd: 11.6213, ept: 47.1687
    Epoch [1/50], Val Losses: mse: 30.8634, mae: 3.5566, huber: 3.1116, swd: 9.7964, ept: 65.2191
    Epoch [1/50], Test Losses: mse: 31.1120, mae: 3.5900, huber: 3.1446, swd: 10.0918, ept: 64.3115
      Epoch 1 composite train-obj: 4.798868
            Val objective improved inf → 3.1116, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 26.8404, mae: 3.2357, huber: 2.7942, swd: 8.8693, ept: 70.0463
    Epoch [2/50], Val Losses: mse: 21.8801, mae: 2.7662, huber: 2.3441, swd: 7.2201, ept: 75.9117
    Epoch [2/50], Test Losses: mse: 21.1967, mae: 2.7221, huber: 2.3000, swd: 7.0145, ept: 76.2295
      Epoch 2 composite train-obj: 2.794210
            Val objective improved 3.1116 → 2.3441, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 19.1080, mae: 2.5552, huber: 2.1303, swd: 6.3142, ept: 78.5933
    Epoch [3/50], Val Losses: mse: 15.2818, mae: 2.1982, huber: 1.7896, swd: 5.3238, ept: 81.7769
    Epoch [3/50], Test Losses: mse: 14.4583, mae: 2.1379, huber: 1.7297, swd: 4.9411, ept: 82.5460
      Epoch 3 composite train-obj: 2.130284
            Val objective improved 2.3441 → 1.7896, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 15.1199, mae: 2.1823, huber: 1.7689, swd: 4.9133, ept: 83.1478
    Epoch [4/50], Val Losses: mse: 12.5865, mae: 1.8864, huber: 1.4920, swd: 4.4036, ept: 85.1404
    Epoch [4/50], Test Losses: mse: 11.8414, mae: 1.8361, huber: 1.4422, swd: 4.0578, ept: 85.6439
      Epoch 4 composite train-obj: 1.768856
            Val objective improved 1.7896 → 1.4920, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 13.0694, mae: 1.9667, huber: 1.5622, swd: 4.2049, ept: 85.4547
    Epoch [5/50], Val Losses: mse: 10.9780, mae: 1.7564, huber: 1.3701, swd: 3.7826, ept: 86.4545
    Epoch [5/50], Test Losses: mse: 10.3675, mae: 1.7125, huber: 1.3266, swd: 3.5696, ept: 87.2420
      Epoch 5 composite train-obj: 1.562188
            Val objective improved 1.4920 → 1.3701, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.8953, mae: 1.8390, huber: 1.4401, swd: 3.8088, ept: 86.6994
    Epoch [6/50], Val Losses: mse: 10.3800, mae: 1.6891, huber: 1.3046, swd: 3.5045, ept: 87.3947
    Epoch [6/50], Test Losses: mse: 9.6995, mae: 1.6423, huber: 1.2586, swd: 3.2343, ept: 88.1230
      Epoch 6 composite train-obj: 1.440114
            Val objective improved 1.3701 → 1.3046, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 11.0120, mae: 1.7385, huber: 1.3449, swd: 3.4881, ept: 87.6749
    Epoch [7/50], Val Losses: mse: 9.3440, mae: 1.5790, huber: 1.1986, swd: 3.1541, ept: 88.0932
    Epoch [7/50], Test Losses: mse: 8.8433, mae: 1.5359, huber: 1.1575, swd: 2.8101, ept: 88.7173
      Epoch 7 composite train-obj: 1.344874
            Val objective improved 1.3046 → 1.1986, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.2949, mae: 1.6554, huber: 1.2663, swd: 3.2471, ept: 88.3545
    Epoch [8/50], Val Losses: mse: 8.5950, mae: 1.4824, huber: 1.1094, swd: 2.8744, ept: 88.5831
    Epoch [8/50], Test Losses: mse: 8.0683, mae: 1.4472, huber: 1.0750, swd: 2.7081, ept: 89.2975
      Epoch 8 composite train-obj: 1.266269
            Val objective improved 1.1986 → 1.1094, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 9.7655, mae: 1.5952, huber: 1.2095, swd: 3.0569, ept: 88.8934
    Epoch [9/50], Val Losses: mse: 8.0685, mae: 1.4406, huber: 1.0681, swd: 2.7745, ept: 89.3190
    Epoch [9/50], Test Losses: mse: 7.6212, mae: 1.4136, huber: 1.0416, swd: 2.5588, ept: 89.8587
      Epoch 9 composite train-obj: 1.209491
            Val objective improved 1.1094 → 1.0681, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 9.2597, mae: 1.5385, huber: 1.1563, swd: 2.8803, ept: 89.3108
    Epoch [10/50], Val Losses: mse: 7.6618, mae: 1.3811, huber: 1.0138, swd: 2.5538, ept: 89.3875
    Epoch [10/50], Test Losses: mse: 7.0885, mae: 1.3417, huber: 0.9758, swd: 2.3423, ept: 90.2819
      Epoch 10 composite train-obj: 1.156342
            Val objective improved 1.0681 → 1.0138, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 8.8494, mae: 1.4888, huber: 1.1099, swd: 2.7331, ept: 89.6925
    Epoch [11/50], Val Losses: mse: 8.0038, mae: 1.4116, huber: 1.0445, swd: 2.5635, ept: 89.2912
    Epoch [11/50], Test Losses: mse: 7.4566, mae: 1.3756, huber: 1.0096, swd: 2.3771, ept: 90.1078
      Epoch 11 composite train-obj: 1.109922
            No improvement (1.0445), counter 1/5
    Epoch [12/50], Train Losses: mse: 8.5776, mae: 1.4575, huber: 1.0805, swd: 2.6391, ept: 89.9894
    Epoch [12/50], Val Losses: mse: 7.5024, mae: 1.3893, huber: 1.0201, swd: 2.4304, ept: 89.6661
    Epoch [12/50], Test Losses: mse: 6.9513, mae: 1.3542, huber: 0.9862, swd: 2.1760, ept: 90.4268
      Epoch 12 composite train-obj: 1.080457
            No improvement (1.0201), counter 2/5
    Epoch [13/50], Train Losses: mse: 8.3060, mae: 1.4211, huber: 1.0470, swd: 2.5403, ept: 90.2024
    Epoch [13/50], Val Losses: mse: 6.7691, mae: 1.3041, huber: 0.9403, swd: 2.2535, ept: 90.2065
    Epoch [13/50], Test Losses: mse: 6.3206, mae: 1.2769, huber: 0.9135, swd: 2.0522, ept: 90.9958
      Epoch 13 composite train-obj: 1.047003
            Val objective improved 1.0138 → 0.9403, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 8.0529, mae: 1.3970, huber: 1.0243, swd: 2.4716, ept: 90.4030
    Epoch [14/50], Val Losses: mse: 6.6918, mae: 1.2709, huber: 0.9149, swd: 2.1955, ept: 90.1633
    Epoch [14/50], Test Losses: mse: 6.1522, mae: 1.2334, huber: 0.8787, swd: 1.9607, ept: 90.9256
      Epoch 14 composite train-obj: 1.024265
            Val objective improved 0.9403 → 0.9149, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 7.7671, mae: 1.3615, huber: 0.9917, swd: 2.3985, ept: 90.6460
    Epoch [15/50], Val Losses: mse: 6.9644, mae: 1.2994, huber: 0.9418, swd: 2.1869, ept: 90.3005
    Epoch [15/50], Test Losses: mse: 6.4690, mae: 1.2657, huber: 0.9094, swd: 2.0282, ept: 91.0734
      Epoch 15 composite train-obj: 0.991652
            No improvement (0.9418), counter 1/5
    Epoch [16/50], Train Losses: mse: 7.5795, mae: 1.3377, huber: 0.9698, swd: 2.3184, ept: 90.7748
    Epoch [16/50], Val Losses: mse: 6.6968, mae: 1.2483, huber: 0.8987, swd: 2.0991, ept: 90.3796
    Epoch [16/50], Test Losses: mse: 6.1521, mae: 1.2099, huber: 0.8619, swd: 1.7585, ept: 91.1370
      Epoch 16 composite train-obj: 0.969793
            Val objective improved 0.9149 → 0.8987, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 7.3667, mae: 1.3099, huber: 0.9442, swd: 2.2393, ept: 90.9285
    Epoch [17/50], Val Losses: mse: 6.3146, mae: 1.2141, huber: 0.8599, swd: 1.9238, ept: 90.7749
    Epoch [17/50], Test Losses: mse: 5.8696, mae: 1.1887, huber: 0.8357, swd: 1.7872, ept: 91.4704
      Epoch 17 composite train-obj: 0.944202
            Val objective improved 0.8987 → 0.8599, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 7.1871, mae: 1.2897, huber: 0.9256, swd: 2.1934, ept: 91.1102
    Epoch [18/50], Val Losses: mse: 6.3442, mae: 1.2282, huber: 0.8749, swd: 1.9255, ept: 90.7065
    Epoch [18/50], Test Losses: mse: 5.8621, mae: 1.2037, huber: 0.8509, swd: 1.7412, ept: 91.4126
      Epoch 18 composite train-obj: 0.925587
            No improvement (0.8749), counter 1/5
    Epoch [19/50], Train Losses: mse: 7.0275, mae: 1.2701, huber: 0.9075, swd: 2.1371, ept: 91.2078
    Epoch [19/50], Val Losses: mse: 6.6106, mae: 1.2610, huber: 0.9069, swd: 1.9312, ept: 90.5529
    Epoch [19/50], Test Losses: mse: 6.1582, mae: 1.2339, huber: 0.8805, swd: 1.7423, ept: 91.2129
      Epoch 19 composite train-obj: 0.907463
            No improvement (0.9069), counter 2/5
    Epoch [20/50], Train Losses: mse: 6.9124, mae: 1.2552, huber: 0.8939, swd: 2.0968, ept: 91.2955
    Epoch [20/50], Val Losses: mse: 6.2787, mae: 1.2340, huber: 0.8808, swd: 1.9343, ept: 91.0589
    Epoch [20/50], Test Losses: mse: 5.7882, mae: 1.1963, huber: 0.8447, swd: 1.6685, ept: 91.6660
      Epoch 20 composite train-obj: 0.893910
            No improvement (0.8808), counter 3/5
    Epoch [21/50], Train Losses: mse: 6.7690, mae: 1.2355, huber: 0.8759, swd: 2.0435, ept: 91.4239
    Epoch [21/50], Val Losses: mse: 6.0269, mae: 1.1882, huber: 0.8407, swd: 1.8110, ept: 90.8834
    Epoch [21/50], Test Losses: mse: 5.5229, mae: 1.1499, huber: 0.8035, swd: 1.5326, ept: 91.6547
      Epoch 21 composite train-obj: 0.875868
            Val objective improved 0.8599 → 0.8407, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 6.5621, mae: 1.2111, huber: 0.8535, swd: 1.9911, ept: 91.5580
    Epoch [22/50], Val Losses: mse: 5.8350, mae: 1.1869, huber: 0.8365, swd: 1.8418, ept: 91.2568
    Epoch [22/50], Test Losses: mse: 5.3687, mae: 1.1604, huber: 0.8103, swd: 1.5973, ept: 91.8716
      Epoch 22 composite train-obj: 0.853499
            Val objective improved 0.8407 → 0.8365, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 6.4834, mae: 1.2013, huber: 0.8446, swd: 1.9528, ept: 91.6321
    Epoch [23/50], Val Losses: mse: 6.2074, mae: 1.2177, huber: 0.8721, swd: 1.8115, ept: 91.0012
    Epoch [23/50], Test Losses: mse: 5.7793, mae: 1.1896, huber: 0.8452, swd: 1.5753, ept: 91.6475
      Epoch 23 composite train-obj: 0.844568
            No improvement (0.8721), counter 1/5
    Epoch [24/50], Train Losses: mse: 6.3376, mae: 1.1852, huber: 0.8298, swd: 1.9090, ept: 91.7513
    Epoch [24/50], Val Losses: mse: 6.1592, mae: 1.2388, huber: 0.8843, swd: 1.7510, ept: 91.0422
    Epoch [24/50], Test Losses: mse: 5.7924, mae: 1.2212, huber: 0.8665, swd: 1.5837, ept: 91.6311
      Epoch 24 composite train-obj: 0.829793
            No improvement (0.8843), counter 2/5
    Epoch [25/50], Train Losses: mse: 6.2633, mae: 1.1729, huber: 0.8191, swd: 1.8917, ept: 91.8154
    Epoch [25/50], Val Losses: mse: 6.2400, mae: 1.2227, huber: 0.8745, swd: 1.7148, ept: 91.0036
    Epoch [25/50], Test Losses: mse: 5.8029, mae: 1.1895, huber: 0.8425, swd: 1.4575, ept: 91.5954
      Epoch 25 composite train-obj: 0.819112
            No improvement (0.8745), counter 3/5
    Epoch [26/50], Train Losses: mse: 6.1112, mae: 1.1535, huber: 0.8014, swd: 1.8417, ept: 91.9199
    Epoch [26/50], Val Losses: mse: 5.4393, mae: 1.1075, huber: 0.7683, swd: 1.6350, ept: 91.6006
    Epoch [26/50], Test Losses: mse: 5.0408, mae: 1.0820, huber: 0.7437, swd: 1.4082, ept: 91.9691
      Epoch 26 composite train-obj: 0.801415
            Val objective improved 0.8365 → 0.7683, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 6.0088, mae: 1.1396, huber: 0.7889, swd: 1.8073, ept: 91.9630
    Epoch [27/50], Val Losses: mse: 5.7986, mae: 1.1793, huber: 0.8313, swd: 1.6138, ept: 91.3066
    Epoch [27/50], Test Losses: mse: 5.4891, mae: 1.1609, huber: 0.8129, swd: 1.4749, ept: 91.8128
      Epoch 27 composite train-obj: 0.788867
            No improvement (0.8313), counter 1/5
    Epoch [28/50], Train Losses: mse: 5.9353, mae: 1.1312, huber: 0.7814, swd: 1.7736, ept: 92.0195
    Epoch [28/50], Val Losses: mse: 5.7778, mae: 1.1661, huber: 0.8206, swd: 1.6128, ept: 91.3301
    Epoch [28/50], Test Losses: mse: 5.3961, mae: 1.1381, huber: 0.7940, swd: 1.3701, ept: 91.7586
      Epoch 28 composite train-obj: 0.781400
            No improvement (0.8206), counter 2/5
    Epoch [29/50], Train Losses: mse: 5.8259, mae: 1.1181, huber: 0.7691, swd: 1.7584, ept: 92.1140
    Epoch [29/50], Val Losses: mse: 5.8051, mae: 1.1842, huber: 0.8350, swd: 1.6053, ept: 91.3849
    Epoch [29/50], Test Losses: mse: 5.5847, mae: 1.1626, huber: 0.8153, swd: 1.4478, ept: 91.9487
      Epoch 29 composite train-obj: 0.769101
            No improvement (0.8350), counter 3/5
    Epoch [30/50], Train Losses: mse: 5.7247, mae: 1.1067, huber: 0.7590, swd: 1.7141, ept: 92.1673
    Epoch [30/50], Val Losses: mse: 6.3703, mae: 1.2352, huber: 0.8888, swd: 1.6729, ept: 91.3278
    Epoch [30/50], Test Losses: mse: 6.1210, mae: 1.2163, huber: 0.8708, swd: 1.3997, ept: 91.6711
      Epoch 30 composite train-obj: 0.758967
            No improvement (0.8888), counter 4/5
    Epoch [31/50], Train Losses: mse: 5.6683, mae: 1.0953, huber: 0.7491, swd: 1.6906, ept: 92.2197
    Epoch [31/50], Val Losses: mse: 5.8547, mae: 1.1778, huber: 0.8349, swd: 1.5861, ept: 91.3667
    Epoch [31/50], Test Losses: mse: 5.6291, mae: 1.1644, huber: 0.8219, swd: 1.3631, ept: 91.7568
      Epoch 31 composite train-obj: 0.749051
    Epoch [31/50], Test Losses: mse: 5.0408, mae: 1.0820, huber: 0.7437, swd: 1.4082, ept: 91.9691
    Best round's Test MSE: 5.0408, MAE: 1.0820, SWD: 1.4082
    Best round's Validation MSE: 5.4393, MAE: 1.1075, SWD: 1.6350
    Best round's Test verification MSE : 5.0408, MAE: 1.0820, SWD: 1.4082
    Time taken: 254.56 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq96_pred96_20250513_0437)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 3.6820 ± 1.2307
      mae: 0.8703 ± 0.1994
      huber: 0.5578 ± 0.1721
      swd: 1.1182 ± 0.2519
      ept: 93.1517 ± 1.0061
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 4.0986 ± 1.2611
      mae: 0.8969 ± 0.1995
      huber: 0.5845 ± 0.1719
      swd: 1.3281 ± 0.2849
      ept: 92.7579 ± 1.0234
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 891.85 seconds
    
    Experiment complete: TimeMixer_lorenz_seq96_pred96_20250513_0437
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 96-196
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 283
    Validation Batches: 39
    Test Batches: 79
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 71.4151, mae: 6.0145, huber: 5.5393, swd: 10.3479, ept: 56.1964
    Epoch [1/50], Val Losses: mse: 50.2358, mae: 4.9164, huber: 4.4506, swd: 13.2909, ept: 73.9650
    Epoch [1/50], Test Losses: mse: 49.5633, mae: 4.8689, huber: 4.4039, swd: 13.6570, ept: 73.1798
      Epoch 1 composite train-obj: 5.539264
            Val objective improved inf → 4.4506, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 43.8035, mae: 4.4940, huber: 4.0335, swd: 13.2385, ept: 89.3414
    Epoch [2/50], Val Losses: mse: 40.7234, mae: 4.2198, huber: 3.7674, swd: 12.8957, ept: 99.1819
    Epoch [2/50], Test Losses: mse: 40.3341, mae: 4.1865, huber: 3.7347, swd: 13.2570, ept: 97.3317
      Epoch 2 composite train-obj: 4.033496
            Val objective improved 4.4506 → 3.7674, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.6072, mae: 4.0544, huber: 3.6025, swd: 11.9183, ept: 105.0443
    Epoch [3/50], Val Losses: mse: 37.5816, mae: 3.9311, huber: 3.4836, swd: 10.9911, ept: 108.9538
    Epoch [3/50], Test Losses: mse: 37.0424, mae: 3.8834, huber: 3.4367, swd: 11.0959, ept: 108.4081
      Epoch 3 composite train-obj: 3.602543
            Val objective improved 3.7674 → 3.4836, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.3885, mae: 3.7706, huber: 3.3252, swd: 10.3486, ept: 114.6935
    Epoch [4/50], Val Losses: mse: 34.3704, mae: 3.6929, huber: 3.2510, swd: 9.5344, ept: 116.1977
    Epoch [4/50], Test Losses: mse: 33.7420, mae: 3.6396, huber: 3.1988, swd: 9.5402, ept: 116.7886
      Epoch 4 composite train-obj: 3.325205
            Val objective improved 3.4836 → 3.2510, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.4995, mae: 3.5321, huber: 3.0920, swd: 8.8266, ept: 122.2030
    Epoch [5/50], Val Losses: mse: 31.6560, mae: 3.4667, huber: 3.0282, swd: 8.0800, ept: 126.1160
    Epoch [5/50], Test Losses: mse: 30.9645, mae: 3.4075, huber: 2.9701, swd: 7.9805, ept: 125.9793
      Epoch 5 composite train-obj: 3.092009
            Val objective improved 3.2510 → 3.0282, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.1017, mae: 3.3436, huber: 2.9069, swd: 7.6563, ept: 128.6890
    Epoch [6/50], Val Losses: mse: 28.7193, mae: 3.2689, huber: 2.8331, swd: 6.9337, ept: 131.6480
    Epoch [6/50], Test Losses: mse: 28.0678, mae: 3.2183, huber: 2.7840, swd: 6.7615, ept: 131.7231
      Epoch 6 composite train-obj: 2.906935
            Val objective improved 3.0282 → 2.8331, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.9878, mae: 3.1856, huber: 2.7519, swd: 6.7628, ept: 133.6536
    Epoch [7/50], Val Losses: mse: 26.3480, mae: 3.0842, huber: 2.6548, swd: 6.1255, ept: 135.5003
    Epoch [7/50], Test Losses: mse: 25.6364, mae: 3.0352, huber: 2.6075, swd: 6.0693, ept: 136.7996
      Epoch 7 composite train-obj: 2.751934
            Val objective improved 2.8331 → 2.6548, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 26.0784, mae: 3.0381, huber: 2.6073, swd: 5.9295, ept: 138.2700
    Epoch [8/50], Val Losses: mse: 25.2266, mae: 2.9550, huber: 2.5276, swd: 5.2758, ept: 141.1640
    Epoch [8/50], Test Losses: mse: 24.2026, mae: 2.8883, huber: 2.4626, swd: 5.2039, ept: 142.2264
      Epoch 8 composite train-obj: 2.607299
            Val objective improved 2.6548 → 2.5276, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.4141, mae: 2.9045, huber: 2.4761, swd: 5.2264, ept: 143.0613
    Epoch [9/50], Val Losses: mse: 23.3386, mae: 2.8325, huber: 2.4061, swd: 4.7631, ept: 144.2217
    Epoch [9/50], Test Losses: mse: 22.3243, mae: 2.7651, huber: 2.3404, swd: 4.6707, ept: 146.1139
      Epoch 9 composite train-obj: 2.476147
            Val objective improved 2.5276 → 2.4061, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.0123, mae: 2.7869, huber: 2.3613, swd: 4.7707, ept: 146.8156
    Epoch [10/50], Val Losses: mse: 22.3126, mae: 2.7655, huber: 2.3396, swd: 4.4311, ept: 149.0763
    Epoch [10/50], Test Losses: mse: 21.1858, mae: 2.6870, huber: 2.2630, swd: 4.1755, ept: 151.0791
      Epoch 10 composite train-obj: 2.361274
            Val objective improved 2.4061 → 2.3396, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.7891, mae: 2.6848, huber: 2.2616, swd: 4.4012, ept: 150.1426
    Epoch [11/50], Val Losses: mse: 20.7577, mae: 2.6186, huber: 2.1972, swd: 3.8770, ept: 149.6640
    Epoch [11/50], Test Losses: mse: 19.7356, mae: 2.5433, huber: 2.1235, swd: 3.7929, ept: 152.1404
      Epoch 11 composite train-obj: 2.261632
            Val objective improved 2.3396 → 2.1972, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.8083, mae: 2.6044, huber: 2.1831, swd: 4.1471, ept: 152.4735
    Epoch [12/50], Val Losses: mse: 19.8761, mae: 2.4993, huber: 2.0846, swd: 3.6401, ept: 152.8357
    Epoch [12/50], Test Losses: mse: 18.6825, mae: 2.4156, huber: 2.0031, swd: 3.4550, ept: 155.3986
      Epoch 12 composite train-obj: 2.183102
            Val objective improved 2.1972 → 2.0846, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 19.9422, mae: 2.5251, huber: 2.1064, swd: 3.9247, ept: 154.5054
    Epoch [13/50], Val Losses: mse: 20.6746, mae: 2.5542, huber: 2.1361, swd: 3.2588, ept: 152.9033
    Epoch [13/50], Test Losses: mse: 19.7301, mae: 2.4815, huber: 2.0650, swd: 3.0298, ept: 156.3998
      Epoch 13 composite train-obj: 2.106432
            No improvement (2.1361), counter 1/5
    Epoch [14/50], Train Losses: mse: 19.2169, mae: 2.4632, huber: 2.0461, swd: 3.7105, ept: 156.2255
    Epoch [14/50], Val Losses: mse: 19.1045, mae: 2.4390, huber: 2.0243, swd: 3.2620, ept: 155.9134
    Epoch [14/50], Test Losses: mse: 17.8724, mae: 2.3572, huber: 1.9435, swd: 3.0238, ept: 158.5832
      Epoch 14 composite train-obj: 2.046063
            Val objective improved 2.0846 → 2.0243, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 18.5071, mae: 2.4009, huber: 1.9855, swd: 3.5551, ept: 157.8498
    Epoch [15/50], Val Losses: mse: 19.1753, mae: 2.4332, huber: 2.0202, swd: 3.0730, ept: 155.9498
    Epoch [15/50], Test Losses: mse: 18.1127, mae: 2.3535, huber: 1.9428, swd: 2.8894, ept: 158.3721
      Epoch 15 composite train-obj: 1.985512
            Val objective improved 2.0243 → 2.0202, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 17.7977, mae: 2.3362, huber: 1.9230, swd: 3.3774, ept: 159.2693
    Epoch [16/50], Val Losses: mse: 18.2691, mae: 2.3291, huber: 1.9207, swd: 2.7864, ept: 158.3437
    Epoch [16/50], Test Losses: mse: 17.1023, mae: 2.2543, huber: 1.8475, swd: 2.7186, ept: 161.0350
      Epoch 16 composite train-obj: 1.922970
            Val objective improved 2.0202 → 1.9207, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 17.2890, mae: 2.2894, huber: 1.8776, swd: 3.2613, ept: 160.3832
    Epoch [17/50], Val Losses: mse: 19.5748, mae: 2.4690, huber: 2.0534, swd: 2.6586, ept: 155.5963
    Epoch [17/50], Test Losses: mse: 18.1910, mae: 2.3711, huber: 1.9570, swd: 2.4196, ept: 159.2149
      Epoch 17 composite train-obj: 1.877552
            No improvement (2.0534), counter 1/5
    Epoch [18/50], Train Losses: mse: 16.7120, mae: 2.2373, huber: 1.8271, swd: 3.1026, ept: 161.6996
    Epoch [18/50], Val Losses: mse: 17.1681, mae: 2.2802, huber: 1.8686, swd: 2.7461, ept: 158.9568
    Epoch [18/50], Test Losses: mse: 15.5322, mae: 2.1682, huber: 1.7584, swd: 2.4928, ept: 163.0237
      Epoch 18 composite train-obj: 1.827139
            Val objective improved 1.9207 → 1.8686, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 16.2243, mae: 2.1908, huber: 1.7824, swd: 3.0183, ept: 162.7220
    Epoch [19/50], Val Losses: mse: 18.6734, mae: 2.3707, huber: 1.9544, swd: 2.3665, ept: 158.5478
    Epoch [19/50], Test Losses: mse: 17.1927, mae: 2.2840, huber: 1.8684, swd: 2.2626, ept: 162.4636
      Epoch 19 composite train-obj: 1.782384
            No improvement (1.9544), counter 1/5
    Epoch [20/50], Train Losses: mse: 15.8738, mae: 2.1614, huber: 1.7533, swd: 2.8959, ept: 163.4306
    Epoch [20/50], Val Losses: mse: 18.5870, mae: 2.3224, huber: 1.9131, swd: 2.3825, ept: 159.1186
    Epoch [20/50], Test Losses: mse: 17.0803, mae: 2.2327, huber: 1.8245, swd: 2.3204, ept: 162.5807
      Epoch 20 composite train-obj: 1.753334
            No improvement (1.9131), counter 2/5
    Epoch [21/50], Train Losses: mse: 15.4727, mae: 2.1185, huber: 1.7124, swd: 2.8209, ept: 164.4773
    Epoch [21/50], Val Losses: mse: 19.2999, mae: 2.3432, huber: 1.9349, swd: 2.2787, ept: 158.2155
    Epoch [21/50], Test Losses: mse: 17.8417, mae: 2.2524, huber: 1.8455, swd: 2.1553, ept: 162.1520
      Epoch 21 composite train-obj: 1.712435
            No improvement (1.9349), counter 3/5
    Epoch [22/50], Train Losses: mse: 15.1778, mae: 2.0891, huber: 1.6839, swd: 2.7482, ept: 164.9905
    Epoch [22/50], Val Losses: mse: 18.2383, mae: 2.2942, huber: 1.8810, swd: 2.3000, ept: 160.2530
    Epoch [22/50], Test Losses: mse: 16.4503, mae: 2.1877, huber: 1.7766, swd: 2.2258, ept: 164.1610
      Epoch 22 composite train-obj: 1.683920
            No improvement (1.8810), counter 4/5
    Epoch [23/50], Train Losses: mse: 14.8280, mae: 2.0577, huber: 1.6538, swd: 2.6445, ept: 165.6683
    Epoch [23/50], Val Losses: mse: 17.9606, mae: 2.2648, huber: 1.8582, swd: 2.4268, ept: 159.9700
    Epoch [23/50], Test Losses: mse: 16.1555, mae: 2.1454, huber: 1.7405, swd: 2.1628, ept: 164.4184
      Epoch 23 composite train-obj: 1.653784
            Val objective improved 1.8686 → 1.8582, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 14.6076, mae: 2.0319, huber: 1.6289, swd: 2.6005, ept: 166.2912
    Epoch [24/50], Val Losses: mse: 15.5013, mae: 2.0962, huber: 1.6948, swd: 2.7906, ept: 162.8484
    Epoch [24/50], Test Losses: mse: 13.6967, mae: 1.9821, huber: 1.5817, swd: 2.4259, ept: 166.7767
      Epoch 24 composite train-obj: 1.628916
            Val objective improved 1.8582 → 1.6948, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 14.4220, mae: 2.0156, huber: 1.6132, swd: 2.5702, ept: 166.8109
    Epoch [25/50], Val Losses: mse: 18.2081, mae: 2.2520, huber: 1.8454, swd: 2.1767, ept: 161.9454
    Epoch [25/50], Test Losses: mse: 16.5236, mae: 2.1530, huber: 1.7475, swd: 2.0828, ept: 166.5175
      Epoch 25 composite train-obj: 1.613155
            No improvement (1.8454), counter 1/5
    Epoch [26/50], Train Losses: mse: 14.1311, mae: 1.9854, huber: 1.5845, swd: 2.4995, ept: 167.2391
    Epoch [26/50], Val Losses: mse: 17.5780, mae: 2.1941, huber: 1.7898, swd: 2.3008, ept: 162.3802
    Epoch [26/50], Test Losses: mse: 16.0851, mae: 2.1027, huber: 1.7000, swd: 2.0898, ept: 166.2805
      Epoch 26 composite train-obj: 1.584486
            No improvement (1.7898), counter 2/5
    Epoch [27/50], Train Losses: mse: 13.8582, mae: 1.9557, huber: 1.5562, swd: 2.4591, ept: 167.7054
    Epoch [27/50], Val Losses: mse: 17.7745, mae: 2.1992, huber: 1.7976, swd: 2.2104, ept: 162.0524
    Epoch [27/50], Test Losses: mse: 15.7797, mae: 2.0732, huber: 1.6731, swd: 1.9828, ept: 166.5759
      Epoch 27 composite train-obj: 1.556246
            No improvement (1.7976), counter 3/5
    Epoch [28/50], Train Losses: mse: 13.6112, mae: 1.9301, huber: 1.5318, swd: 2.3815, ept: 168.4494
    Epoch [28/50], Val Losses: mse: 17.0962, mae: 2.1714, huber: 1.7681, swd: 2.2501, ept: 163.2303
    Epoch [28/50], Test Losses: mse: 15.2929, mae: 2.0533, huber: 1.6517, swd: 2.0610, ept: 167.5380
      Epoch 28 composite train-obj: 1.531824
            No improvement (1.7681), counter 4/5
    Epoch [29/50], Train Losses: mse: 13.3885, mae: 1.9120, huber: 1.5143, swd: 2.3691, ept: 168.8616
    Epoch [29/50], Val Losses: mse: 17.2001, mae: 2.1318, huber: 1.7317, swd: 2.1215, ept: 163.4444
    Epoch [29/50], Test Losses: mse: 15.0863, mae: 2.0000, huber: 1.6023, swd: 1.9240, ept: 167.5891
      Epoch 29 composite train-obj: 1.514317
    Epoch [29/50], Test Losses: mse: 13.6967, mae: 1.9821, huber: 1.5817, swd: 2.4259, ept: 166.7767
    Best round's Test MSE: 13.6967, MAE: 1.9821, SWD: 2.4259
    Best round's Validation MSE: 15.5013, MAE: 2.0962, SWD: 2.7906
    Best round's Test verification MSE : 13.6967, MAE: 1.9821, SWD: 2.4259
    Time taken: 235.83 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 65.0621, mae: 5.7388, huber: 5.2656, swd: 12.4362, ept: 59.9804
    Epoch [1/50], Val Losses: mse: 46.5450, mae: 4.6896, huber: 4.2268, swd: 13.0885, ept: 84.6374
    Epoch [1/50], Test Losses: mse: 45.9745, mae: 4.6454, huber: 4.1834, swd: 13.3706, ept: 83.8986
      Epoch 1 composite train-obj: 5.265572
            Val objective improved inf → 4.2268, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 41.3046, mae: 4.2888, huber: 3.8326, swd: 12.7235, ept: 96.9803
    Epoch [2/50], Val Losses: mse: 40.0467, mae: 4.1361, huber: 3.6844, swd: 11.5544, ept: 105.2911
    Epoch [2/50], Test Losses: mse: 39.4046, mae: 4.0909, huber: 3.6400, swd: 11.6590, ept: 104.0775
      Epoch 2 composite train-obj: 3.832637
            Val objective improved 4.2268 → 3.6844, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 36.5351, mae: 3.8566, huber: 3.4097, swd: 10.6612, ept: 111.2951
    Epoch [3/50], Val Losses: mse: 36.2784, mae: 3.7873, huber: 3.3442, swd: 9.4241, ept: 114.6866
    Epoch [3/50], Test Losses: mse: 35.4973, mae: 3.7329, huber: 3.2907, swd: 9.3755, ept: 114.0215
      Epoch 3 composite train-obj: 3.409747
            Val objective improved 3.6844 → 3.3442, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 32.9307, mae: 3.5555, huber: 3.1150, swd: 8.8715, ept: 121.1463
    Epoch [4/50], Val Losses: mse: 33.4295, mae: 3.5276, huber: 3.0892, swd: 7.7275, ept: 123.2511
    Epoch [4/50], Test Losses: mse: 32.5886, mae: 3.4763, huber: 3.0383, swd: 7.6588, ept: 122.9638
      Epoch 4 composite train-obj: 3.115003
            Val objective improved 3.3442 → 3.0892, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 29.9170, mae: 3.3133, huber: 2.8775, swd: 7.4825, ept: 129.1667
    Epoch [5/50], Val Losses: mse: 28.9507, mae: 3.2070, huber: 2.7769, swd: 6.9046, ept: 131.9065
    Epoch [5/50], Test Losses: mse: 28.0599, mae: 3.1568, huber: 2.7266, swd: 6.7259, ept: 132.0791
      Epoch 5 composite train-obj: 2.877491
            Val objective improved 3.0892 → 2.7769, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 27.1809, mae: 3.1098, huber: 2.6778, swd: 6.5201, ept: 135.4773
    Epoch [6/50], Val Losses: mse: 26.5954, mae: 3.0197, huber: 2.5938, swd: 6.0645, ept: 138.3568
    Epoch [6/50], Test Losses: mse: 25.6777, mae: 2.9664, huber: 2.5411, swd: 5.8272, ept: 138.2494
      Epoch 6 composite train-obj: 2.677779
            Val objective improved 2.7769 → 2.5938, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 24.9493, mae: 2.9375, huber: 2.5090, swd: 5.7281, ept: 140.7795
    Epoch [7/50], Val Losses: mse: 24.1966, mae: 2.8729, huber: 2.4496, swd: 5.4756, ept: 141.7121
    Epoch [7/50], Test Losses: mse: 23.2099, mae: 2.7982, huber: 2.3761, swd: 5.1886, ept: 142.8629
      Epoch 7 composite train-obj: 2.509011
            Val objective improved 2.5938 → 2.4496, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 23.1384, mae: 2.7945, huber: 2.3692, swd: 5.1189, ept: 145.2780
    Epoch [8/50], Val Losses: mse: 22.1700, mae: 2.7704, huber: 2.3428, swd: 4.8930, ept: 145.1277
    Epoch [8/50], Test Losses: mse: 21.3329, mae: 2.7046, huber: 2.2787, swd: 4.8002, ept: 147.4277
      Epoch 8 composite train-obj: 2.369182
            Val objective improved 2.4496 → 2.3428, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 21.7081, mae: 2.6756, huber: 2.2529, swd: 4.6613, ept: 149.1646
    Epoch [9/50], Val Losses: mse: 20.6483, mae: 2.5816, huber: 2.1656, swd: 4.5566, ept: 150.4878
    Epoch [9/50], Test Losses: mse: 19.6228, mae: 2.5100, huber: 2.0950, swd: 4.2653, ept: 152.8990
      Epoch 9 composite train-obj: 2.252948
            Val objective improved 2.3428 → 2.1656, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 20.5580, mae: 2.5785, huber: 2.1586, swd: 4.3277, ept: 152.2557
    Epoch [10/50], Val Losses: mse: 20.3602, mae: 2.5623, huber: 2.1437, swd: 3.6810, ept: 151.3892
    Epoch [10/50], Test Losses: mse: 19.2347, mae: 2.4778, huber: 2.0604, swd: 3.4449, ept: 154.3197
      Epoch 10 composite train-obj: 2.158574
            Val objective improved 2.1656 → 2.1437, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 19.5755, mae: 2.4891, huber: 2.0716, swd: 4.0153, ept: 155.0008
    Epoch [11/50], Val Losses: mse: 18.8863, mae: 2.4185, huber: 2.0039, swd: 3.7257, ept: 153.8399
    Epoch [11/50], Test Losses: mse: 17.5966, mae: 2.3244, huber: 1.9116, swd: 3.4447, ept: 157.2896
      Epoch 11 composite train-obj: 2.071607
            Val objective improved 2.1437 → 2.0039, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 18.7864, mae: 2.4170, huber: 2.0018, swd: 3.8185, ept: 156.8268
    Epoch [12/50], Val Losses: mse: 18.2435, mae: 2.3651, huber: 1.9534, swd: 3.5021, ept: 155.8261
    Epoch [12/50], Test Losses: mse: 16.9749, mae: 2.2764, huber: 1.8660, swd: 3.1857, ept: 158.9476
      Epoch 12 composite train-obj: 2.001781
            Val objective improved 2.0039 → 1.9534, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.1395, mae: 2.3561, huber: 1.9430, swd: 3.6640, ept: 158.5330
    Epoch [13/50], Val Losses: mse: 17.0635, mae: 2.2680, huber: 1.8589, swd: 3.4726, ept: 157.6333
    Epoch [13/50], Test Losses: mse: 15.8757, mae: 2.1805, huber: 1.7731, swd: 3.2860, ept: 160.8219
      Epoch 13 composite train-obj: 1.942998
            Val objective improved 1.9534 → 1.8589, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 17.4524, mae: 2.2945, huber: 1.8832, swd: 3.4753, ept: 160.0585
    Epoch [14/50], Val Losses: mse: 16.4972, mae: 2.2207, huber: 1.8135, swd: 3.5087, ept: 159.3251
    Epoch [14/50], Test Losses: mse: 15.2673, mae: 2.1352, huber: 1.7295, swd: 3.2218, ept: 162.0872
      Epoch 14 composite train-obj: 1.883182
            Val objective improved 1.8589 → 1.8135, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 16.8090, mae: 2.2419, huber: 1.8322, swd: 3.3453, ept: 161.3844
    Epoch [15/50], Val Losses: mse: 16.7649, mae: 2.2299, huber: 1.8208, swd: 3.1733, ept: 159.7458
    Epoch [15/50], Test Losses: mse: 15.2226, mae: 2.1266, huber: 1.7188, swd: 2.8785, ept: 163.5371
      Epoch 15 composite train-obj: 1.832195
            No improvement (1.8208), counter 1/5
    Epoch [16/50], Train Losses: mse: 16.3673, mae: 2.2011, huber: 1.7925, swd: 3.2078, ept: 162.3402
    Epoch [16/50], Val Losses: mse: 16.0816, mae: 2.1669, huber: 1.7585, swd: 3.2263, ept: 161.0032
    Epoch [16/50], Test Losses: mse: 14.4809, mae: 2.0531, huber: 1.6463, swd: 2.8471, ept: 164.6030
      Epoch 16 composite train-obj: 1.792513
            Val objective improved 1.8135 → 1.7585, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.8604, mae: 2.1510, huber: 1.7447, swd: 3.0889, ept: 163.5236
    Epoch [17/50], Val Losses: mse: 15.0252, mae: 2.0755, huber: 1.6736, swd: 2.9585, ept: 162.4522
    Epoch [17/50], Test Losses: mse: 13.5328, mae: 1.9656, huber: 1.5659, swd: 2.6084, ept: 165.8159
      Epoch 17 composite train-obj: 1.744699
            Val objective improved 1.7585 → 1.6736, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.5054, mae: 2.1227, huber: 1.7169, swd: 3.0025, ept: 164.3674
    Epoch [18/50], Val Losses: mse: 15.2596, mae: 2.0833, huber: 1.6816, swd: 2.8202, ept: 162.2655
    Epoch [18/50], Test Losses: mse: 13.7128, mae: 1.9730, huber: 1.5729, swd: 2.5139, ept: 166.2051
      Epoch 18 composite train-obj: 1.716911
            No improvement (1.6816), counter 1/5
    Epoch [19/50], Train Losses: mse: 15.0642, mae: 2.0775, huber: 1.6740, swd: 2.8943, ept: 165.3387
    Epoch [19/50], Val Losses: mse: 15.3840, mae: 2.0679, huber: 1.6705, swd: 2.9905, ept: 163.4146
    Epoch [19/50], Test Losses: mse: 13.7929, mae: 1.9624, huber: 1.5670, swd: 2.6188, ept: 166.6147
      Epoch 19 composite train-obj: 1.673979
            Val objective improved 1.6736 → 1.6705, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 14.7896, mae: 2.0480, huber: 1.6456, swd: 2.8172, ept: 166.0222
    Epoch [20/50], Val Losses: mse: 14.1150, mae: 1.9848, huber: 1.5880, swd: 2.7046, ept: 164.8398
    Epoch [20/50], Test Losses: mse: 12.7519, mae: 1.8899, huber: 1.4941, swd: 2.3839, ept: 168.4480
      Epoch 20 composite train-obj: 1.645568
            Val objective improved 1.6705 → 1.5880, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 14.4060, mae: 2.0161, huber: 1.6148, swd: 2.7360, ept: 166.7035
    Epoch [21/50], Val Losses: mse: 13.5974, mae: 1.9347, huber: 1.5386, swd: 2.6337, ept: 166.0570
    Epoch [21/50], Test Losses: mse: 12.1659, mae: 1.8367, huber: 1.4416, swd: 2.2923, ept: 169.2615
      Epoch 21 composite train-obj: 1.614839
            Val objective improved 1.5880 → 1.5386, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 14.1908, mae: 1.9915, huber: 1.5913, swd: 2.6706, ept: 167.4801
    Epoch [22/50], Val Losses: mse: 15.0156, mae: 2.0463, huber: 1.6426, swd: 2.6614, ept: 165.5074
    Epoch [22/50], Test Losses: mse: 13.3757, mae: 1.9448, huber: 1.5415, swd: 2.3398, ept: 168.5816
      Epoch 22 composite train-obj: 1.591266
            No improvement (1.6426), counter 1/5
    Epoch [23/50], Train Losses: mse: 13.8862, mae: 1.9600, huber: 1.5611, swd: 2.5982, ept: 168.0327
    Epoch [23/50], Val Losses: mse: 14.6094, mae: 2.0052, huber: 1.6071, swd: 2.3938, ept: 165.1368
    Epoch [23/50], Test Losses: mse: 13.1657, mae: 1.9031, huber: 1.5069, swd: 2.1489, ept: 169.0851
      Epoch 23 composite train-obj: 1.561083
            No improvement (1.6071), counter 2/5
    Epoch [24/50], Train Losses: mse: 13.6126, mae: 1.9359, huber: 1.5381, swd: 2.5375, ept: 168.5689
    Epoch [24/50], Val Losses: mse: 14.2900, mae: 1.9848, huber: 1.5883, swd: 2.3662, ept: 165.7344
    Epoch [24/50], Test Losses: mse: 12.8506, mae: 1.8911, huber: 1.4958, swd: 2.1459, ept: 169.5985
      Epoch 24 composite train-obj: 1.538103
            No improvement (1.5883), counter 3/5
    Epoch [25/50], Train Losses: mse: 13.5135, mae: 1.9244, huber: 1.5272, swd: 2.5338, ept: 168.9226
    Epoch [25/50], Val Losses: mse: 13.2244, mae: 1.9142, huber: 1.5172, swd: 2.2178, ept: 166.6128
    Epoch [25/50], Test Losses: mse: 11.7506, mae: 1.8075, huber: 1.4126, swd: 1.9419, ept: 170.9376
      Epoch 25 composite train-obj: 1.527191
            Val objective improved 1.5386 → 1.5172, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 13.2583, mae: 1.8984, huber: 1.5023, swd: 2.4366, ept: 169.5367
    Epoch [26/50], Val Losses: mse: 13.4457, mae: 1.8830, huber: 1.4918, swd: 2.5406, ept: 167.6637
    Epoch [26/50], Test Losses: mse: 11.8823, mae: 1.7743, huber: 1.3847, swd: 2.2074, ept: 170.9902
      Epoch 26 composite train-obj: 1.502335
            Val objective improved 1.5172 → 1.4918, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 12.9862, mae: 1.8715, huber: 1.4768, swd: 2.3841, ept: 170.0500
    Epoch [27/50], Val Losses: mse: 12.9769, mae: 1.9057, huber: 1.5078, swd: 2.1689, ept: 167.5691
    Epoch [27/50], Test Losses: mse: 11.5493, mae: 1.8071, huber: 1.4106, swd: 1.9892, ept: 171.3663
      Epoch 27 composite train-obj: 1.476795
            No improvement (1.5078), counter 1/5
    Epoch [28/50], Train Losses: mse: 12.8189, mae: 1.8550, huber: 1.4610, swd: 2.3590, ept: 170.5209
    Epoch [28/50], Val Losses: mse: 12.3297, mae: 1.7933, huber: 1.4044, swd: 2.2743, ept: 168.1925
    Epoch [28/50], Test Losses: mse: 10.9179, mae: 1.6929, huber: 1.3061, swd: 1.9911, ept: 172.0800
      Epoch 28 composite train-obj: 1.460961
            Val objective improved 1.4918 → 1.4044, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 12.6824, mae: 1.8351, huber: 1.4422, swd: 2.2996, ept: 170.7897
    Epoch [29/50], Val Losses: mse: 13.2114, mae: 1.9025, huber: 1.5070, swd: 2.2268, ept: 168.7815
    Epoch [29/50], Test Losses: mse: 11.7044, mae: 1.7980, huber: 1.4041, swd: 1.9441, ept: 172.5058
      Epoch 29 composite train-obj: 1.442186
            No improvement (1.5070), counter 1/5
    Epoch [30/50], Train Losses: mse: 12.4898, mae: 1.8199, huber: 1.4275, swd: 2.2649, ept: 171.2166
    Epoch [30/50], Val Losses: mse: 12.2772, mae: 1.7990, huber: 1.4102, swd: 2.1629, ept: 169.8742
    Epoch [30/50], Test Losses: mse: 10.8679, mae: 1.7031, huber: 1.3155, swd: 1.8375, ept: 173.5977
      Epoch 30 composite train-obj: 1.427456
            No improvement (1.4102), counter 2/5
    Epoch [31/50], Train Losses: mse: 12.2648, mae: 1.7988, huber: 1.4074, swd: 2.2305, ept: 171.5778
    Epoch [31/50], Val Losses: mse: 13.6739, mae: 1.8816, huber: 1.4918, swd: 2.0885, ept: 168.2544
    Epoch [31/50], Test Losses: mse: 12.1651, mae: 1.7825, huber: 1.3937, swd: 1.8490, ept: 172.3186
      Epoch 31 composite train-obj: 1.407426
            No improvement (1.4918), counter 3/5
    Epoch [32/50], Train Losses: mse: 12.1148, mae: 1.7802, huber: 1.3900, swd: 2.1942, ept: 171.9743
    Epoch [32/50], Val Losses: mse: 13.6559, mae: 1.8846, huber: 1.4943, swd: 2.1256, ept: 168.5800
    Epoch [32/50], Test Losses: mse: 12.1172, mae: 1.7837, huber: 1.3950, swd: 1.9279, ept: 172.5066
      Epoch 32 composite train-obj: 1.389984
            No improvement (1.4943), counter 4/5
    Epoch [33/50], Train Losses: mse: 11.9904, mae: 1.7679, huber: 1.3782, swd: 2.1406, ept: 172.2229
    Epoch [33/50], Val Losses: mse: 12.1269, mae: 1.7955, huber: 1.4062, swd: 2.0680, ept: 169.6670
    Epoch [33/50], Test Losses: mse: 10.7331, mae: 1.7032, huber: 1.3151, swd: 1.8321, ept: 173.4311
      Epoch 33 composite train-obj: 1.378185
    Epoch [33/50], Test Losses: mse: 10.9179, mae: 1.6929, huber: 1.3061, swd: 1.9911, ept: 172.0800
    Best round's Test MSE: 10.9179, MAE: 1.6929, SWD: 1.9911
    Best round's Validation MSE: 12.3297, MAE: 1.7933, SWD: 2.2743
    Best round's Test verification MSE : 10.9179, MAE: 1.6929, SWD: 1.9911
    Time taken: 268.40 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 66.9388, mae: 5.8437, huber: 5.3696, swd: 12.4468, ept: 61.6011
    Epoch [1/50], Val Losses: mse: 48.0874, mae: 4.8023, huber: 4.3395, swd: 14.0668, ept: 83.2959
    Epoch [1/50], Test Losses: mse: 47.4099, mae: 4.7478, huber: 4.2859, swd: 14.4963, ept: 83.2195
      Epoch 1 composite train-obj: 5.369562
            Val objective improved inf → 4.3395, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 42.3492, mae: 4.3887, huber: 3.9304, swd: 13.1300, ept: 96.8759
    Epoch [2/50], Val Losses: mse: 40.1207, mae: 4.1813, huber: 3.7283, swd: 12.2087, ept: 104.3340
    Epoch [2/50], Test Losses: mse: 39.6042, mae: 4.1307, huber: 3.6792, swd: 12.3128, ept: 104.0961
      Epoch 2 composite train-obj: 3.930435
            Val objective improved 4.3395 → 3.7283, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 37.4327, mae: 3.9547, huber: 3.5050, swd: 11.0095, ept: 110.3705
    Epoch [3/50], Val Losses: mse: 37.1089, mae: 3.9036, huber: 3.4570, swd: 9.6754, ept: 111.5125
    Epoch [3/50], Test Losses: mse: 36.1432, mae: 3.8308, huber: 3.3850, swd: 9.4939, ept: 112.6862
      Epoch 3 composite train-obj: 3.504977
            Val objective improved 3.7283 → 3.4570, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 33.2994, mae: 3.6150, huber: 3.1714, swd: 8.7010, ept: 120.4100
    Epoch [4/50], Val Losses: mse: 33.4046, mae: 3.5551, huber: 3.1175, swd: 8.1234, ept: 122.3755
    Epoch [4/50], Test Losses: mse: 32.5344, mae: 3.4950, huber: 3.0584, swd: 7.9642, ept: 122.6777
      Epoch 4 composite train-obj: 3.171361
            Val objective improved 3.4570 → 3.1175, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 30.0470, mae: 3.3479, huber: 2.9098, swd: 7.1931, ept: 128.6771
    Epoch [5/50], Val Losses: mse: 30.1439, mae: 3.3096, huber: 2.8732, swd: 6.3281, ept: 129.4017
    Epoch [5/50], Test Losses: mse: 29.2235, mae: 3.2443, huber: 2.8093, swd: 6.0709, ept: 131.0668
      Epoch 5 composite train-obj: 2.909843
            Val objective improved 3.1175 → 2.8732, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 27.5557, mae: 3.1416, huber: 2.7081, swd: 6.1734, ept: 135.1051
    Epoch [6/50], Val Losses: mse: 27.3719, mae: 3.1054, huber: 2.6755, swd: 5.4774, ept: 135.7760
    Epoch [6/50], Test Losses: mse: 26.4005, mae: 3.0404, huber: 2.6112, swd: 5.2296, ept: 138.1280
      Epoch 6 composite train-obj: 2.708148
            Val objective improved 2.8732 → 2.6755, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 25.4586, mae: 2.9763, huber: 2.5466, swd: 5.5026, ept: 140.3129
    Epoch [7/50], Val Losses: mse: 25.1396, mae: 2.9382, huber: 2.5138, swd: 5.1874, ept: 139.7981
    Epoch [7/50], Test Losses: mse: 23.9731, mae: 2.8529, huber: 2.4302, swd: 4.9418, ept: 143.2153
      Epoch 7 composite train-obj: 2.546575
            Val objective improved 2.6755 → 2.5138, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 23.6706, mae: 2.8278, huber: 2.4020, swd: 4.9534, ept: 144.7949
    Epoch [8/50], Val Losses: mse: 23.2263, mae: 2.8071, huber: 2.3853, swd: 4.7935, ept: 143.7424
    Epoch [8/50], Test Losses: mse: 21.7187, mae: 2.7078, huber: 2.2873, swd: 4.4954, ept: 147.6773
      Epoch 8 composite train-obj: 2.401997
            Val objective improved 2.5138 → 2.3853, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 22.1298, mae: 2.7040, huber: 2.2811, swd: 4.4961, ept: 148.3855
    Epoch [9/50], Val Losses: mse: 21.8595, mae: 2.6854, huber: 2.2645, swd: 4.0776, ept: 147.4694
    Epoch [9/50], Test Losses: mse: 20.5559, mae: 2.5857, huber: 2.1678, swd: 3.7984, ept: 151.6126
      Epoch 9 composite train-obj: 2.281144
            Val objective improved 2.3853 → 2.2645, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 20.8836, mae: 2.6018, huber: 2.1814, swd: 4.1508, ept: 151.5535
    Epoch [10/50], Val Losses: mse: 20.9387, mae: 2.6269, huber: 2.2080, swd: 3.8761, ept: 149.7679
    Epoch [10/50], Test Losses: mse: 19.5641, mae: 2.5219, huber: 2.1058, swd: 3.7495, ept: 153.0060
      Epoch 10 composite train-obj: 2.181442
            Val objective improved 2.2645 → 2.2080, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 19.9049, mae: 2.5139, huber: 2.0961, swd: 3.8756, ept: 154.0740
    Epoch [11/50], Val Losses: mse: 21.0758, mae: 2.6268, huber: 2.2061, swd: 3.2492, ept: 151.8979
    Epoch [11/50], Test Losses: mse: 19.8573, mae: 2.5454, huber: 2.1261, swd: 2.9675, ept: 155.2245
      Epoch 11 composite train-obj: 2.096090
            Val objective improved 2.2080 → 2.2061, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 19.0375, mae: 2.4410, huber: 2.0252, swd: 3.6538, ept: 156.3159
    Epoch [12/50], Val Losses: mse: 20.1671, mae: 2.5675, huber: 2.1485, swd: 3.5363, ept: 152.5658
    Epoch [12/50], Test Losses: mse: 18.8040, mae: 2.4628, huber: 2.0461, swd: 3.3622, ept: 155.9906
      Epoch 12 composite train-obj: 2.025184
            Val objective improved 2.2061 → 2.1485, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.3742, mae: 2.3823, huber: 1.9684, swd: 3.4929, ept: 157.6362
    Epoch [13/50], Val Losses: mse: 18.7017, mae: 2.4031, huber: 1.9937, swd: 3.2313, ept: 155.9172
    Epoch [13/50], Test Losses: mse: 17.4628, mae: 2.3044, huber: 1.8972, swd: 2.8858, ept: 158.9106
      Epoch 13 composite train-obj: 1.968355
            Val objective improved 2.1485 → 1.9937, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 17.7635, mae: 2.3235, huber: 1.9118, swd: 3.3341, ept: 159.1832
    Epoch [14/50], Val Losses: mse: 19.8323, mae: 2.4600, huber: 2.0473, swd: 2.7222, ept: 155.6912
    Epoch [14/50], Test Losses: mse: 18.4934, mae: 2.3621, huber: 1.9511, swd: 2.6313, ept: 159.2722
      Epoch 14 composite train-obj: 1.911799
            No improvement (2.0473), counter 1/5
    Epoch [15/50], Train Losses: mse: 17.2711, mae: 2.2816, huber: 1.8711, swd: 3.2188, ept: 160.1638
    Epoch [15/50], Val Losses: mse: 18.3235, mae: 2.3570, huber: 1.9481, swd: 2.8484, ept: 157.2099
    Epoch [15/50], Test Losses: mse: 16.6001, mae: 2.2364, huber: 1.8302, swd: 2.6662, ept: 160.7322
      Epoch 15 composite train-obj: 1.871126
            Val objective improved 1.9937 → 1.9481, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.7202, mae: 2.2356, huber: 1.8265, swd: 3.1260, ept: 161.1651
    Epoch [16/50], Val Losses: mse: 19.3128, mae: 2.4333, huber: 2.0200, swd: 2.7972, ept: 156.7054
    Epoch [16/50], Test Losses: mse: 18.2736, mae: 2.3471, huber: 1.9362, swd: 2.6054, ept: 159.9751
      Epoch 16 composite train-obj: 1.826493
            No improvement (2.0200), counter 1/5
    Epoch [17/50], Train Losses: mse: 16.3604, mae: 2.2017, huber: 1.7937, swd: 3.0287, ept: 161.9115
    Epoch [17/50], Val Losses: mse: 18.8274, mae: 2.3949, huber: 1.9826, swd: 2.6401, ept: 157.6371
    Epoch [17/50], Test Losses: mse: 17.1892, mae: 2.2815, huber: 1.8711, swd: 2.4015, ept: 160.9879
      Epoch 17 composite train-obj: 1.793668
            No improvement (1.9826), counter 2/5
    Epoch [18/50], Train Losses: mse: 16.0633, mae: 2.1792, huber: 1.7716, swd: 2.9581, ept: 162.6448
    Epoch [18/50], Val Losses: mse: 18.5377, mae: 2.3735, huber: 1.9619, swd: 2.7930, ept: 157.2767
    Epoch [18/50], Test Losses: mse: 17.1984, mae: 2.2740, huber: 1.8640, swd: 2.5913, ept: 160.7010
      Epoch 18 composite train-obj: 1.771644
            No improvement (1.9619), counter 3/5
    Epoch [19/50], Train Losses: mse: 15.6300, mae: 2.1380, huber: 1.7322, swd: 2.9074, ept: 163.4629
    Epoch [19/50], Val Losses: mse: 19.5391, mae: 2.4025, huber: 1.9919, swd: 2.4182, ept: 156.7521
    Epoch [19/50], Test Losses: mse: 18.0090, mae: 2.2991, huber: 1.8904, swd: 2.2497, ept: 160.0005
      Epoch 19 composite train-obj: 1.732216
            No improvement (1.9919), counter 4/5
    Epoch [20/50], Train Losses: mse: 15.3074, mae: 2.1058, huber: 1.7012, swd: 2.8076, ept: 164.1450
    Epoch [20/50], Val Losses: mse: 17.1248, mae: 2.2156, huber: 1.8106, swd: 2.5871, ept: 160.4036
    Epoch [20/50], Test Losses: mse: 15.8160, mae: 2.1253, huber: 1.7216, swd: 2.3503, ept: 163.7722
      Epoch 20 composite train-obj: 1.701151
            Val objective improved 1.9481 → 1.8106, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 15.1089, mae: 2.0877, huber: 1.6837, swd: 2.7684, ept: 164.5141
    Epoch [21/50], Val Losses: mse: 17.7946, mae: 2.2960, huber: 1.8880, swd: 2.6441, ept: 158.1123
    Epoch [21/50], Test Losses: mse: 15.7445, mae: 2.1635, huber: 1.7577, swd: 2.4278, ept: 161.7009
      Epoch 21 composite train-obj: 1.683688
            No improvement (1.8880), counter 1/5
    Epoch [22/50], Train Losses: mse: 14.8526, mae: 2.0651, huber: 1.6616, swd: 2.6798, ept: 165.1201
    Epoch [22/50], Val Losses: mse: 18.7174, mae: 2.3268, huber: 1.9207, swd: 2.4157, ept: 156.7571
    Epoch [22/50], Test Losses: mse: 16.9546, mae: 2.2164, huber: 1.8119, swd: 2.3232, ept: 160.4935
      Epoch 22 composite train-obj: 1.661565
            No improvement (1.9207), counter 2/5
    Epoch [23/50], Train Losses: mse: 14.5384, mae: 2.0364, huber: 1.6341, swd: 2.6351, ept: 165.7337
    Epoch [23/50], Val Losses: mse: 18.9299, mae: 2.3137, huber: 1.9062, swd: 2.4583, ept: 159.2689
    Epoch [23/50], Test Losses: mse: 17.3456, mae: 2.2070, huber: 1.8016, swd: 2.2219, ept: 162.5101
      Epoch 23 composite train-obj: 1.634132
            No improvement (1.9062), counter 3/5
    Epoch [24/50], Train Losses: mse: 14.3824, mae: 2.0200, huber: 1.6184, swd: 2.5741, ept: 166.2526
    Epoch [24/50], Val Losses: mse: 18.1596, mae: 2.2961, huber: 1.8872, swd: 2.3676, ept: 158.8745
    Epoch [24/50], Test Losses: mse: 16.7421, mae: 2.2038, huber: 1.7968, swd: 2.2667, ept: 162.1739
      Epoch 24 composite train-obj: 1.618408
            No improvement (1.8872), counter 4/5
    Epoch [25/50], Train Losses: mse: 14.1919, mae: 1.9972, huber: 1.5967, swd: 2.5373, ept: 166.7045
    Epoch [25/50], Val Losses: mse: 20.1339, mae: 2.4249, huber: 2.0166, swd: 2.3717, ept: 155.0919
    Epoch [25/50], Test Losses: mse: 18.3833, mae: 2.3038, huber: 1.8979, swd: 2.0722, ept: 158.6671
      Epoch 25 composite train-obj: 1.596702
    Epoch [25/50], Test Losses: mse: 15.8160, mae: 2.1253, huber: 1.7216, swd: 2.3503, ept: 163.7722
    Best round's Test MSE: 15.8160, MAE: 2.1253, SWD: 2.3503
    Best round's Validation MSE: 17.1248, MAE: 2.2156, SWD: 2.5871
    Best round's Test verification MSE : 15.8160, MAE: 2.1253, SWD: 2.3503
    Time taken: 203.91 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq96_pred196_20250513_0452)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 13.4769 ± 2.0056
      mae: 1.9334 ± 0.1798
      huber: 1.5365 ± 0.1726
      swd: 2.2558 ± 0.1897
      ept: 167.5430 ± 3.4347
      count: 39.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 14.9852 ± 1.9913
      mae: 2.0350 ± 0.1778
      huber: 1.6366 ± 0.1708
      swd: 2.5507 ± 0.2124
      ept: 163.8148 ± 3.2524
      count: 39.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 708.19 seconds
    
    Experiment complete: TimeMixer_lorenz_seq96_pred196_20250513_0452
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 96-336
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
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
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
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
    
    Epoch [1/50], Train Losses: mse: 78.9275, mae: 6.4674, huber: 5.9892, swd: 11.0179, ept: 57.0926
    Epoch [1/50], Val Losses: mse: 61.0218, mae: 5.6534, huber: 5.1806, swd: 16.9809, ept: 74.9749
    Epoch [1/50], Test Losses: mse: 60.0933, mae: 5.5651, huber: 5.0929, swd: 16.8152, ept: 73.8181
      Epoch 1 composite train-obj: 5.989218
            Val objective improved inf → 5.1806, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 53.6948, mae: 5.2359, huber: 4.7665, swd: 16.6330, ept: 91.6832
    Epoch [2/50], Val Losses: mse: 53.9773, mae: 5.1687, huber: 4.7037, swd: 16.1772, ept: 94.9596
    Epoch [2/50], Test Losses: mse: 52.3670, mae: 5.0540, huber: 4.5902, swd: 16.3385, ept: 95.4444
      Epoch 2 composite train-obj: 4.766455
            Val objective improved 5.1806 → 4.7037, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 49.1312, mae: 4.8954, huber: 4.4312, swd: 15.4803, ept: 111.0304
    Epoch [3/50], Val Losses: mse: 54.9199, mae: 5.2075, huber: 4.7409, swd: 13.3315, ept: 102.1804
    Epoch [3/50], Test Losses: mse: 53.2979, mae: 5.0922, huber: 4.6266, swd: 13.2923, ept: 103.1540
      Epoch 3 composite train-obj: 4.431226
            No improvement (4.7409), counter 1/5
    Epoch [4/50], Train Losses: mse: 46.6834, mae: 4.6720, huber: 4.2118, swd: 13.8466, ept: 125.5643
    Epoch [4/50], Val Losses: mse: 48.3860, mae: 4.7408, huber: 4.2834, swd: 13.8004, ept: 123.0648
    Epoch [4/50], Test Losses: mse: 46.8546, mae: 4.6112, huber: 4.1556, swd: 13.6185, ept: 124.6556
      Epoch 4 composite train-obj: 4.211797
            Val objective improved 4.7037 → 4.2834, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 44.3128, mae: 4.4621, huber: 4.0058, swd: 12.3267, ept: 138.5108
    Epoch [5/50], Val Losses: mse: 48.6959, mae: 4.6886, huber: 4.2319, swd: 11.1963, ept: 132.7614
    Epoch [5/50], Test Losses: mse: 46.8910, mae: 4.5577, huber: 4.1024, swd: 11.1813, ept: 135.0261
      Epoch 5 composite train-obj: 4.005811
            Val objective improved 4.2834 → 4.2319, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 42.7386, mae: 4.3231, huber: 3.8695, swd: 11.1862, ept: 147.8580
    Epoch [6/50], Val Losses: mse: 45.8917, mae: 4.4680, huber: 4.0158, swd: 10.6340, ept: 141.8782
    Epoch [6/50], Test Losses: mse: 44.2161, mae: 4.3412, huber: 3.8906, swd: 10.6060, ept: 145.5244
      Epoch 6 composite train-obj: 3.869529
            Val objective improved 4.2319 → 4.0158, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 41.3417, mae: 4.1999, huber: 3.7487, swd: 10.2558, ept: 155.6331
    Epoch [7/50], Val Losses: mse: 44.8480, mae: 4.4150, huber: 3.9630, swd: 10.0742, ept: 147.2558
    Epoch [7/50], Test Losses: mse: 42.7638, mae: 4.2674, huber: 3.8166, swd: 9.9825, ept: 150.1767
      Epoch 7 composite train-obj: 3.748693
            Val objective improved 4.0158 → 3.9630, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 40.1221, mae: 4.0999, huber: 3.6505, swd: 9.4927, ept: 161.6925
    Epoch [8/50], Val Losses: mse: 43.9514, mae: 4.3433, huber: 3.8919, swd: 9.2853, ept: 153.8596
    Epoch [8/50], Test Losses: mse: 41.9731, mae: 4.2000, huber: 3.7496, swd: 9.0624, ept: 156.3539
      Epoch 8 composite train-obj: 3.650509
            Val objective improved 3.9630 → 3.8919, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 39.1725, mae: 4.0213, huber: 3.5736, swd: 8.9595, ept: 166.6675
    Epoch [9/50], Val Losses: mse: 42.2480, mae: 4.2573, huber: 3.8059, swd: 8.8479, ept: 156.9011
    Epoch [9/50], Test Losses: mse: 40.3318, mae: 4.1206, huber: 3.6702, swd: 8.5503, ept: 161.0190
      Epoch 9 composite train-obj: 3.573568
            Val objective improved 3.8919 → 3.8059, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 38.5064, mae: 3.9608, huber: 3.5144, swd: 8.4673, ept: 170.5651
    Epoch [10/50], Val Losses: mse: 42.6974, mae: 4.2510, huber: 3.7992, swd: 8.6156, ept: 157.5696
    Epoch [10/50], Test Losses: mse: 40.7778, mae: 4.1101, huber: 3.6599, swd: 8.5194, ept: 161.7825
      Epoch 10 composite train-obj: 3.514398
            Val objective improved 3.8059 → 3.7992, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 37.4902, mae: 3.8827, huber: 3.4380, swd: 7.9844, ept: 175.0179
    Epoch [11/50], Val Losses: mse: 42.3691, mae: 4.2185, huber: 3.7708, swd: 7.0067, ept: 164.6043
    Epoch [11/50], Test Losses: mse: 39.7988, mae: 4.0533, huber: 3.6069, swd: 6.8693, ept: 168.7497
      Epoch 11 composite train-obj: 3.437986
            Val objective improved 3.7992 → 3.7708, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 36.7348, mae: 3.8252, huber: 3.3816, swd: 7.6167, ept: 178.5371
    Epoch [12/50], Val Losses: mse: 39.7583, mae: 4.0581, huber: 3.6112, swd: 7.4963, ept: 168.9243
    Epoch [12/50], Test Losses: mse: 37.5086, mae: 3.9068, huber: 3.4611, swd: 7.2040, ept: 174.4033
      Epoch 12 composite train-obj: 3.381621
            Val objective improved 3.7708 → 3.6112, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 35.7695, mae: 3.7457, huber: 3.3037, swd: 7.1072, ept: 183.3035
    Epoch [13/50], Val Losses: mse: 39.8679, mae: 3.9750, huber: 3.5328, swd: 6.7133, ept: 173.2181
    Epoch [13/50], Test Losses: mse: 37.4121, mae: 3.8004, huber: 3.3602, swd: 6.4927, ept: 180.1874
      Epoch 13 composite train-obj: 3.303652
            Val objective improved 3.6112 → 3.5328, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 34.9310, mae: 3.6766, huber: 3.2361, swd: 6.7390, ept: 187.2786
    Epoch [14/50], Val Losses: mse: 38.6667, mae: 3.9278, huber: 3.4848, swd: 6.6025, ept: 175.2798
    Epoch [14/50], Test Losses: mse: 36.4147, mae: 3.7662, huber: 3.3246, swd: 6.1447, ept: 182.3570
      Epoch 14 composite train-obj: 3.236137
            Val objective improved 3.5328 → 3.4848, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 34.5010, mae: 3.6381, huber: 3.1985, swd: 6.4486, ept: 189.4064
    Epoch [15/50], Val Losses: mse: 39.9914, mae: 3.9839, huber: 3.5415, swd: 6.0526, ept: 174.5033
    Epoch [15/50], Test Losses: mse: 37.4056, mae: 3.8164, huber: 3.3748, swd: 5.6612, ept: 182.1427
      Epoch 15 composite train-obj: 3.198535
            No improvement (3.5415), counter 1/5
    Epoch [16/50], Train Losses: mse: 34.2799, mae: 3.6127, huber: 3.1740, swd: 6.2644, ept: 191.2699
    Epoch [16/50], Val Losses: mse: 39.3263, mae: 4.0479, huber: 3.6004, swd: 6.2179, ept: 175.8783
    Epoch [16/50], Test Losses: mse: 36.9281, mae: 3.9055, huber: 3.4592, swd: 6.0042, ept: 181.2421
      Epoch 16 composite train-obj: 3.174009
            No improvement (3.6004), counter 2/5
    Epoch [17/50], Train Losses: mse: 34.1146, mae: 3.6100, huber: 3.1710, swd: 6.3095, ept: 191.8220
    Epoch [17/50], Val Losses: mse: 38.3407, mae: 3.8767, huber: 3.4350, swd: 5.9986, ept: 180.5662
    Epoch [17/50], Test Losses: mse: 35.3251, mae: 3.6843, huber: 3.2439, swd: 5.4340, ept: 188.5789
      Epoch 17 composite train-obj: 3.171013
            Val objective improved 3.4848 → 3.4350, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 33.2201, mae: 3.5245, huber: 3.0881, swd: 5.9729, ept: 195.9402
    Epoch [18/50], Val Losses: mse: 37.4304, mae: 3.8379, huber: 3.3957, swd: 5.9886, ept: 179.9076
    Epoch [18/50], Test Losses: mse: 35.0038, mae: 3.6683, huber: 3.2280, swd: 5.5315, ept: 187.8808
      Epoch 18 composite train-obj: 3.088143
            Val objective improved 3.4350 → 3.3957, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 32.9125, mae: 3.5001, huber: 3.0641, swd: 5.8104, ept: 196.9986
    Epoch [19/50], Val Losses: mse: 38.5803, mae: 3.9108, huber: 3.4688, swd: 7.0028, ept: 176.6263
    Epoch [19/50], Test Losses: mse: 35.8257, mae: 3.7429, huber: 3.3027, swd: 6.8825, ept: 183.5271
      Epoch 19 composite train-obj: 3.064133
            No improvement (3.4688), counter 1/5
    Epoch [20/50], Train Losses: mse: 32.6295, mae: 3.4809, huber: 3.0451, swd: 5.7440, ept: 198.1190
    Epoch [20/50], Val Losses: mse: 38.3902, mae: 3.8252, huber: 3.3865, swd: 5.5855, ept: 182.2246
    Epoch [20/50], Test Losses: mse: 35.6428, mae: 3.6456, huber: 3.2087, swd: 5.2988, ept: 190.1804
      Epoch 20 composite train-obj: 3.045101
            Val objective improved 3.3957 → 3.3865, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 32.0155, mae: 3.4325, huber: 2.9979, swd: 5.5413, ept: 200.5020
    Epoch [21/50], Val Losses: mse: 38.2367, mae: 3.8129, huber: 3.3738, swd: 5.5951, ept: 182.0307
    Epoch [21/50], Test Losses: mse: 35.1360, mae: 3.6359, huber: 3.1980, swd: 5.4797, ept: 189.5592
      Epoch 21 composite train-obj: 2.997899
            Val objective improved 3.3865 → 3.3738, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 31.5195, mae: 3.4018, huber: 2.9678, swd: 5.4381, ept: 201.8649
    Epoch [22/50], Val Losses: mse: 37.9602, mae: 3.7819, huber: 3.3436, swd: 5.2897, ept: 184.3671
    Epoch [22/50], Test Losses: mse: 34.6167, mae: 3.5752, huber: 3.1385, swd: 4.8585, ept: 192.8688
      Epoch 22 composite train-obj: 2.967766
            Val objective improved 3.3738 → 3.3436, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 31.1807, mae: 3.3699, huber: 2.9367, swd: 5.2647, ept: 203.2498
    Epoch [23/50], Val Losses: mse: 36.0024, mae: 3.7177, huber: 3.2795, swd: 5.5300, ept: 184.0463
    Epoch [23/50], Test Losses: mse: 33.2874, mae: 3.5358, huber: 3.0994, swd: 5.1845, ept: 191.9071
      Epoch 23 composite train-obj: 2.936676
            Val objective improved 3.3436 → 3.2795, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 30.6986, mae: 3.3371, huber: 2.9046, swd: 5.1644, ept: 205.1431
    Epoch [24/50], Val Losses: mse: 37.1981, mae: 3.7097, huber: 3.2754, swd: 5.5176, ept: 185.4192
    Epoch [24/50], Test Losses: mse: 33.9336, mae: 3.5149, huber: 3.0821, swd: 5.2324, ept: 193.4628
      Epoch 24 composite train-obj: 2.904570
            Val objective improved 3.2795 → 3.2754, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 30.3942, mae: 3.3171, huber: 2.8847, swd: 5.0477, ept: 205.8457
    Epoch [25/50], Val Losses: mse: 37.2829, mae: 3.7561, huber: 3.3173, swd: 5.1076, ept: 185.2859
    Epoch [25/50], Test Losses: mse: 33.9366, mae: 3.5635, huber: 3.1257, swd: 5.0046, ept: 193.8736
      Epoch 25 composite train-obj: 2.884726
            No improvement (3.3173), counter 1/5
    Epoch [26/50], Train Losses: mse: 29.9292, mae: 3.2838, huber: 2.8522, swd: 4.9453, ept: 207.6007
    Epoch [26/50], Val Losses: mse: 35.8831, mae: 3.6903, huber: 3.2526, swd: 4.5311, ept: 190.7910
    Epoch [26/50], Test Losses: mse: 32.8080, mae: 3.5155, huber: 3.0783, swd: 3.9632, ept: 199.8877
      Epoch 26 composite train-obj: 2.852163
            Val objective improved 3.2754 → 3.2526, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 29.7660, mae: 3.2769, huber: 2.8450, swd: 4.8337, ept: 207.7415
    Epoch [27/50], Val Losses: mse: 37.9573, mae: 3.8205, huber: 3.3770, swd: 4.8423, ept: 189.3418
    Epoch [27/50], Test Losses: mse: 34.7427, mae: 3.6412, huber: 3.1987, swd: 4.6689, ept: 196.0427
      Epoch 27 composite train-obj: 2.845019
            No improvement (3.3770), counter 1/5
    Epoch [28/50], Train Losses: mse: 29.5011, mae: 3.2593, huber: 2.8277, swd: 4.8035, ept: 208.7379
    Epoch [28/50], Val Losses: mse: 36.5031, mae: 3.7245, huber: 3.2836, swd: 5.3625, ept: 188.9516
    Epoch [28/50], Test Losses: mse: 33.7816, mae: 3.5678, huber: 3.1281, swd: 5.2833, ept: 196.0528
      Epoch 28 composite train-obj: 2.827708
            No improvement (3.2836), counter 2/5
    Epoch [29/50], Train Losses: mse: 29.9586, mae: 3.2923, huber: 2.8597, swd: 4.8442, ept: 208.2682
    Epoch [29/50], Val Losses: mse: 36.1430, mae: 3.6820, huber: 3.2433, swd: 4.8575, ept: 191.3257
    Epoch [29/50], Test Losses: mse: 32.8309, mae: 3.4917, huber: 3.0541, swd: 4.5913, ept: 199.0884
      Epoch 29 composite train-obj: 2.859723
            Val objective improved 3.2526 → 3.2433, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 28.9952, mae: 3.2147, huber: 2.7841, swd: 4.5967, ept: 211.6995
    Epoch [30/50], Val Losses: mse: 38.4900, mae: 3.7402, huber: 3.3037, swd: 3.8842, ept: 193.7672
    Epoch [30/50], Test Losses: mse: 34.7145, mae: 3.5329, huber: 3.0976, swd: 3.5497, ept: 202.9562
      Epoch 30 composite train-obj: 2.784141
            No improvement (3.3037), counter 1/5
    Epoch [31/50], Train Losses: mse: 28.8200, mae: 3.1955, huber: 2.7656, swd: 4.5319, ept: 213.0332
    Epoch [31/50], Val Losses: mse: 35.9887, mae: 3.5939, huber: 3.1594, swd: 4.3419, ept: 195.1497
    Epoch [31/50], Test Losses: mse: 32.2967, mae: 3.3839, huber: 2.9511, swd: 4.0326, ept: 204.9348
      Epoch 31 composite train-obj: 2.765568
            Val objective improved 3.2433 → 3.1594, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 28.0211, mae: 3.1380, huber: 2.7091, swd: 4.3649, ept: 215.5349
    Epoch [32/50], Val Losses: mse: 34.2819, mae: 3.5226, huber: 3.0892, swd: 4.6858, ept: 199.0778
    Epoch [32/50], Test Losses: mse: 30.9749, mae: 3.3437, huber: 2.9114, swd: 4.5457, ept: 207.3518
      Epoch 32 composite train-obj: 2.709117
            Val objective improved 3.1594 → 3.0892, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 28.0675, mae: 3.1484, huber: 2.7189, swd: 4.3476, ept: 215.5466
    Epoch [33/50], Val Losses: mse: 36.9236, mae: 3.6586, huber: 3.2227, swd: 4.0725, ept: 197.4846
    Epoch [33/50], Test Losses: mse: 33.2168, mae: 3.4558, huber: 3.0206, swd: 3.8276, ept: 204.8732
      Epoch 33 composite train-obj: 2.718894
            No improvement (3.2227), counter 1/5
    Epoch [34/50], Train Losses: mse: 28.0318, mae: 3.1325, huber: 2.7038, swd: 4.3051, ept: 216.6450
    Epoch [34/50], Val Losses: mse: 32.6234, mae: 3.5343, huber: 3.0950, swd: 4.8849, ept: 195.4526
    Epoch [34/50], Test Losses: mse: 30.3570, mae: 3.3994, huber: 2.9612, swd: 4.5463, ept: 204.4607
      Epoch 34 composite train-obj: 2.703751
            No improvement (3.0950), counter 2/5
    Epoch [35/50], Train Losses: mse: 28.5195, mae: 3.1823, huber: 2.7521, swd: 4.5930, ept: 213.9389
    Epoch [35/50], Val Losses: mse: 36.8401, mae: 3.6786, huber: 3.2416, swd: 3.7526, ept: 196.3319
    Epoch [35/50], Test Losses: mse: 33.3347, mae: 3.4907, huber: 3.0551, swd: 3.6283, ept: 202.9223
      Epoch 35 composite train-obj: 2.752124
            No improvement (3.2416), counter 3/5
    Epoch [36/50], Train Losses: mse: 28.0926, mae: 3.1445, huber: 2.7150, swd: 4.3384, ept: 216.2512
    Epoch [36/50], Val Losses: mse: 36.5801, mae: 3.6779, huber: 3.2400, swd: 4.9533, ept: 192.4307
    Epoch [36/50], Test Losses: mse: 33.6486, mae: 3.5010, huber: 3.0644, swd: 4.7867, ept: 200.6582
      Epoch 36 composite train-obj: 2.715050
            No improvement (3.2400), counter 4/5
    Epoch [37/50], Train Losses: mse: 27.4142, mae: 3.0861, huber: 2.6583, swd: 4.1697, ept: 219.0835
    Epoch [37/50], Val Losses: mse: 34.3176, mae: 3.4942, huber: 3.0592, swd: 3.5250, ept: 206.0580
    Epoch [37/50], Test Losses: mse: 30.0662, mae: 3.2697, huber: 2.8357, swd: 3.3198, ept: 215.7519
      Epoch 37 composite train-obj: 2.658319
            Val objective improved 3.0892 → 3.0592, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 27.1318, mae: 3.0634, huber: 2.6362, swd: 4.0639, ept: 220.3187
    Epoch [38/50], Val Losses: mse: 37.3584, mae: 3.6644, huber: 3.2297, swd: 3.8370, ept: 195.7000
    Epoch [38/50], Test Losses: mse: 33.7084, mae: 3.4629, huber: 3.0295, swd: 3.6845, ept: 204.9808
      Epoch 38 composite train-obj: 2.636161
            No improvement (3.2297), counter 1/5
    Epoch [39/50], Train Losses: mse: 27.2099, mae: 3.0683, huber: 2.6410, swd: 4.1347, ept: 219.4359
    Epoch [39/50], Val Losses: mse: 35.4281, mae: 3.6009, huber: 3.1615, swd: 3.8012, ept: 203.9625
    Epoch [39/50], Test Losses: mse: 31.4336, mae: 3.3977, huber: 2.9590, swd: 3.5324, ept: 212.9947
      Epoch 39 composite train-obj: 2.641000
            No improvement (3.1615), counter 2/5
    Epoch [40/50], Train Losses: mse: 26.6368, mae: 3.0216, huber: 2.5954, swd: 3.9516, ept: 222.8146
    Epoch [40/50], Val Losses: mse: 38.0017, mae: 3.6701, huber: 3.2356, swd: 3.6531, ept: 199.0956
    Epoch [40/50], Test Losses: mse: 33.6485, mae: 3.4437, huber: 3.0108, swd: 3.4515, ept: 207.6343
      Epoch 40 composite train-obj: 2.595405
            No improvement (3.2356), counter 3/5
    Epoch [41/50], Train Losses: mse: 26.5150, mae: 3.0121, huber: 2.5860, swd: 3.8720, ept: 222.8031
    Epoch [41/50], Val Losses: mse: 36.5802, mae: 3.5846, huber: 3.1517, swd: 3.8477, ept: 200.3332
    Epoch [41/50], Test Losses: mse: 32.6605, mae: 3.3779, huber: 2.9465, swd: 3.7638, ept: 207.8821
      Epoch 41 composite train-obj: 2.585983
            No improvement (3.1517), counter 4/5
    Epoch [42/50], Train Losses: mse: 26.5575, mae: 3.0073, huber: 2.5816, swd: 3.8764, ept: 223.3912
    Epoch [42/50], Val Losses: mse: 34.7481, mae: 3.5182, huber: 3.0832, swd: 3.9198, ept: 203.1157
    Epoch [42/50], Test Losses: mse: 30.9421, mae: 3.3242, huber: 2.8900, swd: 3.8655, ept: 211.5873
      Epoch 42 composite train-obj: 2.581556
    Epoch [42/50], Test Losses: mse: 30.0662, mae: 3.2697, huber: 2.8357, swd: 3.3198, ept: 215.7519
    Best round's Test MSE: 30.0662, MAE: 3.2697, SWD: 3.3198
    Best round's Validation MSE: 34.3176, MAE: 3.4942, SWD: 3.5250
    Best round's Test verification MSE : 30.0662, MAE: 3.2697, SWD: 3.3198
    Time taken: 343.05 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 71.2673, mae: 6.1416, huber: 5.6649, swd: 13.0810, ept: 63.3223
    Epoch [1/50], Val Losses: mse: 59.8036, mae: 5.5865, huber: 5.1139, swd: 16.7777, ept: 79.1943
    Epoch [1/50], Test Losses: mse: 58.4578, mae: 5.4970, huber: 5.0246, swd: 17.4492, ept: 75.2642
      Epoch 1 composite train-obj: 5.664913
            Val objective improved inf → 5.1139, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 52.0065, mae: 5.1130, huber: 4.6454, swd: 16.4441, ept: 99.3333
    Epoch [2/50], Val Losses: mse: 56.4744, mae: 5.4484, huber: 4.9773, swd: 16.3898, ept: 96.0627
    Epoch [2/50], Test Losses: mse: 55.0823, mae: 5.3755, huber: 4.9044, swd: 17.1641, ept: 94.0229
      Epoch 2 composite train-obj: 4.645414
            Val objective improved 5.1139 → 4.9773, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 48.7311, mae: 4.8475, huber: 4.3849, swd: 15.4182, ept: 115.4374
    Epoch [3/50], Val Losses: mse: 51.8487, mae: 5.0394, huber: 4.5764, swd: 14.9395, ept: 109.7299
    Epoch [3/50], Test Losses: mse: 50.2885, mae: 4.8972, huber: 4.4356, swd: 14.2963, ept: 112.2683
      Epoch 3 composite train-obj: 4.384922
            Val objective improved 4.9773 → 4.5764, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 46.3992, mae: 4.6227, huber: 4.1646, swd: 13.7518, ept: 128.6254
    Epoch [4/50], Val Losses: mse: 48.4943, mae: 4.7840, huber: 4.3244, swd: 13.8202, ept: 119.5934
    Epoch [4/50], Test Losses: mse: 47.2444, mae: 4.6893, huber: 4.2308, swd: 13.9245, ept: 120.9926
      Epoch 4 composite train-obj: 4.164577
            Val objective improved 4.5764 → 4.3244, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 44.3788, mae: 4.4327, huber: 3.9782, swd: 12.2090, ept: 140.7207
    Epoch [5/50], Val Losses: mse: 46.6668, mae: 4.5568, huber: 4.1044, swd: 12.5679, ept: 135.2639
    Epoch [5/50], Test Losses: mse: 45.0439, mae: 4.4238, huber: 3.9726, swd: 12.2293, ept: 138.1721
      Epoch 5 composite train-obj: 3.978178
            Val objective improved 4.3244 → 4.1044, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 42.6041, mae: 4.2849, huber: 3.8331, swd: 11.3020, ept: 150.2195
    Epoch [6/50], Val Losses: mse: 46.1787, mae: 4.6256, huber: 4.1653, swd: 11.0215, ept: 140.3345
    Epoch [6/50], Test Losses: mse: 44.7279, mae: 4.4990, huber: 4.0393, swd: 10.5082, ept: 143.6045
      Epoch 6 composite train-obj: 3.833089
            No improvement (4.1653), counter 1/5
    Epoch [7/50], Train Losses: mse: 41.2504, mae: 4.1778, huber: 3.7282, swd: 10.5928, ept: 156.7184
    Epoch [7/50], Val Losses: mse: 45.1739, mae: 4.4544, huber: 4.0016, swd: 10.2344, ept: 144.7551
    Epoch [7/50], Test Losses: mse: 43.8700, mae: 4.3444, huber: 3.8924, swd: 10.1058, ept: 147.7157
      Epoch 7 composite train-obj: 3.728188
            Val objective improved 4.1044 → 4.0016, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 39.9431, mae: 4.0706, huber: 3.6233, swd: 9.9218, ept: 162.7090
    Epoch [8/50], Val Losses: mse: 42.7776, mae: 4.2599, huber: 3.8099, swd: 9.8749, ept: 153.9271
    Epoch [8/50], Test Losses: mse: 40.9322, mae: 4.1217, huber: 3.6731, swd: 9.7100, ept: 157.6375
      Epoch 8 composite train-obj: 3.623292
            Val objective improved 4.0016 → 3.8099, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 38.6556, mae: 3.9724, huber: 3.5268, swd: 9.2943, ept: 168.3117
    Epoch [9/50], Val Losses: mse: 41.6704, mae: 4.1592, huber: 3.7134, swd: 9.2449, ept: 159.2121
    Epoch [9/50], Test Losses: mse: 39.7177, mae: 4.0179, huber: 3.5734, swd: 8.9956, ept: 163.0100
      Epoch 9 composite train-obj: 3.526764
            Val objective improved 3.8099 → 3.7134, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 37.5074, mae: 3.8856, huber: 3.4416, swd: 8.6978, ept: 173.6130
    Epoch [10/50], Val Losses: mse: 41.5706, mae: 4.1898, huber: 3.7423, swd: 9.2992, ept: 161.2490
    Epoch [10/50], Test Losses: mse: 39.3974, mae: 4.0426, huber: 3.5965, swd: 9.0225, ept: 164.3690
      Epoch 10 composite train-obj: 3.441553
            No improvement (3.7423), counter 1/5
    Epoch [11/50], Train Losses: mse: 36.6609, mae: 3.8192, huber: 3.3764, swd: 8.2013, ept: 177.7719
    Epoch [11/50], Val Losses: mse: 40.8029, mae: 4.1001, huber: 3.6557, swd: 7.8408, ept: 168.8134
    Epoch [11/50], Test Losses: mse: 38.9540, mae: 3.9631, huber: 3.5195, swd: 7.5395, ept: 172.8894
      Epoch 11 composite train-obj: 3.376368
            Val objective improved 3.7134 → 3.6557, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 36.2209, mae: 3.7864, huber: 3.3441, swd: 7.8457, ept: 179.9022
    Epoch [12/50], Val Losses: mse: 39.6101, mae: 4.0385, huber: 3.5930, swd: 7.7253, ept: 171.6215
    Epoch [12/50], Test Losses: mse: 37.5428, mae: 3.8938, huber: 3.4495, swd: 7.4803, ept: 176.1425
      Epoch 12 composite train-obj: 3.344126
            Val objective improved 3.6557 → 3.5930, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 35.3599, mae: 3.7112, huber: 3.2707, swd: 7.4306, ept: 184.6591
    Epoch [13/50], Val Losses: mse: 39.2594, mae: 4.0445, huber: 3.5977, swd: 7.7083, ept: 171.5027
    Epoch [13/50], Test Losses: mse: 37.2279, mae: 3.9079, huber: 3.4616, swd: 7.3427, ept: 175.3566
      Epoch 13 composite train-obj: 3.270749
            No improvement (3.5977), counter 1/5
    Epoch [14/50], Train Losses: mse: 34.7775, mae: 3.6651, huber: 3.2255, swd: 7.1019, ept: 187.2213
    Epoch [14/50], Val Losses: mse: 37.4289, mae: 3.8715, huber: 3.4288, swd: 7.7448, ept: 176.8586
    Epoch [14/50], Test Losses: mse: 35.1164, mae: 3.7071, huber: 3.2654, swd: 7.2723, ept: 182.6002
      Epoch 14 composite train-obj: 3.225492
            Val objective improved 3.5930 → 3.4288, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 34.1750, mae: 3.6156, huber: 3.1771, swd: 6.8421, ept: 190.3600
    Epoch [15/50], Val Losses: mse: 40.5329, mae: 4.1373, huber: 3.6851, swd: 6.5492, ept: 170.9189
    Epoch [15/50], Test Losses: mse: 37.6850, mae: 3.9454, huber: 3.4949, swd: 6.2439, ept: 177.2487
      Epoch 15 composite train-obj: 3.177064
            No improvement (3.6851), counter 1/5
    Epoch [16/50], Train Losses: mse: 34.0997, mae: 3.6150, huber: 3.1762, swd: 6.7432, ept: 190.7917
    Epoch [16/50], Val Losses: mse: 38.1357, mae: 3.9386, huber: 3.4901, swd: 6.8295, ept: 178.2327
    Epoch [16/50], Test Losses: mse: 35.8408, mae: 3.7743, huber: 3.3274, swd: 6.3725, ept: 186.1539
      Epoch 16 composite train-obj: 3.176214
            No improvement (3.4901), counter 2/5
    Epoch [17/50], Train Losses: mse: 33.1036, mae: 3.5376, huber: 3.1006, swd: 6.4703, ept: 194.2005
    Epoch [17/50], Val Losses: mse: 41.7298, mae: 4.1293, huber: 3.6802, swd: 7.1107, ept: 175.6394
    Epoch [17/50], Test Losses: mse: 39.3920, mae: 3.9822, huber: 3.5346, swd: 6.8731, ept: 180.6856
      Epoch 17 composite train-obj: 3.100609
            No improvement (3.6802), counter 3/5
    Epoch [18/50], Train Losses: mse: 34.0405, mae: 3.6015, huber: 3.1631, swd: 6.6736, ept: 192.2774
    Epoch [18/50], Val Losses: mse: 36.6894, mae: 3.8561, huber: 3.4102, swd: 7.3081, ept: 176.5318
    Epoch [18/50], Test Losses: mse: 34.3429, mae: 3.7021, huber: 3.2578, swd: 7.1109, ept: 182.9221
      Epoch 18 composite train-obj: 3.163115
            Val objective improved 3.4288 → 3.4102, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 32.6547, mae: 3.5017, huber: 3.0654, swd: 6.1807, ept: 196.7438
    Epoch [19/50], Val Losses: mse: 36.5745, mae: 3.7650, huber: 3.3266, swd: 6.1207, ept: 184.0071
    Epoch [19/50], Test Losses: mse: 33.9565, mae: 3.5834, huber: 3.1469, swd: 5.6855, ept: 190.4424
      Epoch 19 composite train-obj: 3.065362
            Val objective improved 3.4102 → 3.3266, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 31.9071, mae: 3.4415, huber: 3.0068, swd: 6.0096, ept: 199.6546
    Epoch [20/50], Val Losses: mse: 35.0766, mae: 3.6890, huber: 3.2515, swd: 6.4749, ept: 186.0272
    Epoch [20/50], Test Losses: mse: 32.5366, mae: 3.5187, huber: 3.0830, swd: 6.1022, ept: 192.5320
      Epoch 20 composite train-obj: 3.006781
            Val objective improved 3.3266 → 3.2515, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 31.6588, mae: 3.4233, huber: 2.9889, swd: 5.8819, ept: 200.6442
    Epoch [21/50], Val Losses: mse: 37.4964, mae: 3.8270, huber: 3.3875, swd: 6.1734, ept: 183.3535
    Epoch [21/50], Test Losses: mse: 34.5432, mae: 3.6325, huber: 3.1949, swd: 5.6992, ept: 189.6023
      Epoch 21 composite train-obj: 2.988922
            No improvement (3.3875), counter 1/5
    Epoch [22/50], Train Losses: mse: 31.3424, mae: 3.4006, huber: 2.9664, swd: 5.7329, ept: 202.1522
    Epoch [22/50], Val Losses: mse: 34.3412, mae: 3.6198, huber: 3.1813, swd: 5.9395, ept: 190.6708
    Epoch [22/50], Test Losses: mse: 31.9661, mae: 3.4618, huber: 3.0248, swd: 5.6798, ept: 196.3019
      Epoch 22 composite train-obj: 2.966364
            Val objective improved 3.2515 → 3.1813, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 30.7149, mae: 3.3534, huber: 2.9203, swd: 5.6026, ept: 204.1087
    Epoch [23/50], Val Losses: mse: 37.3590, mae: 3.8601, huber: 3.4182, swd: 5.6412, ept: 180.0061
    Epoch [23/50], Test Losses: mse: 34.9288, mae: 3.7057, huber: 3.2652, swd: 5.4612, ept: 187.8812
      Epoch 23 composite train-obj: 2.920286
            No improvement (3.4182), counter 1/5
    Epoch [24/50], Train Losses: mse: 31.0040, mae: 3.3769, huber: 2.9434, swd: 5.6003, ept: 203.5728
    Epoch [24/50], Val Losses: mse: 34.3731, mae: 3.5818, huber: 3.1483, swd: 5.8434, ept: 193.2992
    Epoch [24/50], Test Losses: mse: 31.6013, mae: 3.3963, huber: 2.9647, swd: 5.4943, ept: 201.1553
      Epoch 24 composite train-obj: 2.943350
            Val objective improved 3.1813 → 3.1483, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 30.2102, mae: 3.3047, huber: 2.8732, swd: 5.3711, ept: 206.4154
    Epoch [25/50], Val Losses: mse: 35.2232, mae: 3.7314, huber: 3.2890, swd: 5.2867, ept: 189.7118
    Epoch [25/50], Test Losses: mse: 33.0914, mae: 3.5821, huber: 3.1411, swd: 4.9582, ept: 196.6301
      Epoch 25 composite train-obj: 2.873187
            No improvement (3.2890), counter 1/5
    Epoch [26/50], Train Losses: mse: 30.3734, mae: 3.3266, huber: 2.8942, swd: 5.3856, ept: 206.0614
    Epoch [26/50], Val Losses: mse: 34.6403, mae: 3.6294, huber: 3.1892, swd: 5.0576, ept: 196.5086
    Epoch [26/50], Test Losses: mse: 31.6714, mae: 3.4317, huber: 2.9930, swd: 4.5508, ept: 205.5624
      Epoch 26 composite train-obj: 2.894242
            No improvement (3.1892), counter 2/5
    Epoch [27/50], Train Losses: mse: 29.7231, mae: 3.2733, huber: 2.8423, swd: 5.2340, ept: 208.1622
    Epoch [27/50], Val Losses: mse: 39.5241, mae: 4.0098, huber: 3.5588, swd: 4.6359, ept: 185.1858
    Epoch [27/50], Test Losses: mse: 36.3572, mae: 3.8360, huber: 3.3857, swd: 4.5265, ept: 190.5498
      Epoch 27 composite train-obj: 2.842310
            No improvement (3.5588), counter 3/5
    Epoch [28/50], Train Losses: mse: 31.0661, mae: 3.3929, huber: 2.9583, swd: 5.6927, ept: 202.4926
    Epoch [28/50], Val Losses: mse: 33.5659, mae: 3.5455, huber: 3.1110, swd: 5.4914, ept: 194.9031
    Epoch [28/50], Test Losses: mse: 31.0968, mae: 3.3810, huber: 2.9488, swd: 5.2196, ept: 202.0664
      Epoch 28 composite train-obj: 2.958285
            Val objective improved 3.1483 → 3.1110, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 29.3015, mae: 3.2383, huber: 2.8081, swd: 5.0726, ept: 210.0545
    Epoch [29/50], Val Losses: mse: 39.6047, mae: 3.9382, huber: 3.4916, swd: 4.6383, ept: 190.3774
    Epoch [29/50], Test Losses: mse: 35.2074, mae: 3.6998, huber: 3.2547, swd: 4.4231, ept: 197.5027
      Epoch 29 composite train-obj: 2.808078
            No improvement (3.4916), counter 1/5
    Epoch [30/50], Train Losses: mse: 30.1160, mae: 3.3078, huber: 2.8754, swd: 5.2210, ept: 207.5082
    Epoch [30/50], Val Losses: mse: 32.8850, mae: 3.4750, huber: 3.0401, swd: 5.1264, ept: 199.4009
    Epoch [30/50], Test Losses: mse: 30.2765, mae: 3.3158, huber: 2.8822, swd: 4.9317, ept: 206.9790
      Epoch 30 composite train-obj: 2.875431
            Val objective improved 3.1110 → 3.0401, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 28.9271, mae: 3.2085, huber: 2.7789, swd: 4.9065, ept: 212.0462
    Epoch [31/50], Val Losses: mse: 33.1578, mae: 3.4871, huber: 3.0545, swd: 5.3559, ept: 196.1702
    Epoch [31/50], Test Losses: mse: 30.2942, mae: 3.3128, huber: 2.8820, swd: 5.1184, ept: 204.3774
      Epoch 31 composite train-obj: 2.778869
            No improvement (3.0545), counter 1/5
    Epoch [32/50], Train Losses: mse: 28.8090, mae: 3.1950, huber: 2.7659, swd: 4.8441, ept: 212.8564
    Epoch [32/50], Val Losses: mse: 32.5835, mae: 3.5384, huber: 3.0986, swd: 5.2070, ept: 196.9276
    Epoch [32/50], Test Losses: mse: 29.9572, mae: 3.3774, huber: 2.9386, swd: 5.0102, ept: 203.4310
      Epoch 32 composite train-obj: 2.765855
            No improvement (3.0986), counter 2/5
    Epoch [33/50], Train Losses: mse: 28.6155, mae: 3.1866, huber: 2.7575, swd: 4.8000, ept: 213.6569
    Epoch [33/50], Val Losses: mse: 32.6198, mae: 3.4836, huber: 3.0489, swd: 4.6086, ept: 201.0717
    Epoch [33/50], Test Losses: mse: 29.8329, mae: 3.3055, huber: 2.8720, swd: 4.3669, ept: 208.2171
      Epoch 33 composite train-obj: 2.757455
            No improvement (3.0489), counter 3/5
    Epoch [34/50], Train Losses: mse: 28.3912, mae: 3.1677, huber: 2.7390, swd: 4.7145, ept: 214.1023
    Epoch [34/50], Val Losses: mse: 33.8284, mae: 3.5563, huber: 3.1208, swd: 5.0015, ept: 197.7586
    Epoch [34/50], Test Losses: mse: 30.5270, mae: 3.3509, huber: 2.9171, swd: 4.5470, ept: 205.6428
      Epoch 34 composite train-obj: 2.738968
            No improvement (3.1208), counter 4/5
    Epoch [35/50], Train Losses: mse: 27.8646, mae: 3.1274, huber: 2.6996, swd: 4.6003, ept: 216.2953
    Epoch [35/50], Val Losses: mse: 34.1413, mae: 3.5069, huber: 3.0742, swd: 4.4767, ept: 202.3246
    Epoch [35/50], Test Losses: mse: 30.6808, mae: 3.2931, huber: 2.8624, swd: 4.0980, ept: 211.5956
      Epoch 35 composite train-obj: 2.699555
    Epoch [35/50], Test Losses: mse: 30.2765, mae: 3.3158, huber: 2.8822, swd: 4.9317, ept: 206.9790
    Best round's Test MSE: 30.2765, MAE: 3.3158, SWD: 4.9317
    Best round's Validation MSE: 32.8850, MAE: 3.4750, SWD: 5.1264
    Best round's Test verification MSE : 30.2765, MAE: 3.3158, SWD: 4.9317
    Time taken: 287.21 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 69.8291, mae: 6.0950, huber: 5.6186, swd: 14.3664, ept: 66.7780
    Epoch [1/50], Val Losses: mse: 57.4587, mae: 5.4476, huber: 4.9774, swd: 18.0533, ept: 81.1095
    Epoch [1/50], Test Losses: mse: 55.7739, mae: 5.3193, huber: 4.8501, swd: 18.1041, ept: 80.9960
      Epoch 1 composite train-obj: 5.618567
            Val objective improved inf → 4.9774, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 50.9728, mae: 5.0418, huber: 4.5753, swd: 17.0590, ept: 101.9258
    Epoch [2/50], Val Losses: mse: 51.9009, mae: 5.0744, huber: 4.6099, swd: 17.8876, ept: 100.0613
    Epoch [2/50], Test Losses: mse: 50.2080, mae: 4.9557, huber: 4.4924, swd: 17.9695, ept: 100.1705
      Epoch 2 composite train-obj: 4.575273
            Val objective improved 4.9774 → 4.6099, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 46.9331, mae: 4.7068, huber: 4.2459, swd: 15.3373, ept: 121.8141
    Epoch [3/50], Val Losses: mse: 49.4108, mae: 4.8601, huber: 4.4004, swd: 15.7133, ept: 112.4460
    Epoch [3/50], Test Losses: mse: 47.6909, mae: 4.7325, huber: 4.2738, swd: 15.2951, ept: 114.9049
      Epoch 3 composite train-obj: 4.245910
            Val objective improved 4.6099 → 4.4004, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 44.9490, mae: 4.5107, huber: 4.0538, swd: 13.6768, ept: 135.9795
    Epoch [4/50], Val Losses: mse: 50.3887, mae: 5.0337, huber: 4.5666, swd: 15.0598, ept: 116.0796
    Epoch [4/50], Test Losses: mse: 49.2544, mae: 4.9678, huber: 4.5016, swd: 15.3571, ept: 116.8900
      Epoch 4 composite train-obj: 4.053795
            No improvement (4.5666), counter 1/5
    Epoch [5/50], Train Losses: mse: 43.6338, mae: 4.3883, huber: 3.9339, swd: 12.7224, ept: 145.2852
    Epoch [5/50], Val Losses: mse: 46.0367, mae: 4.5296, huber: 4.0765, swd: 12.9075, ept: 137.7278
    Epoch [5/50], Test Losses: mse: 43.8711, mae: 4.3630, huber: 3.9114, swd: 12.2622, ept: 142.5077
      Epoch 5 composite train-obj: 3.933909
            Val objective improved 4.4004 → 4.0765, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 41.7618, mae: 4.2228, huber: 3.7718, swd: 11.7073, ept: 155.4501
    Epoch [6/50], Val Losses: mse: 45.7485, mae: 4.5489, huber: 4.0929, swd: 12.0192, ept: 143.9611
    Epoch [6/50], Test Losses: mse: 43.7487, mae: 4.4142, huber: 3.9605, swd: 11.9790, ept: 146.3615
      Epoch 6 composite train-obj: 3.771814
            No improvement (4.0929), counter 1/5
    Epoch [7/50], Train Losses: mse: 40.5833, mae: 4.1310, huber: 3.6816, swd: 10.9650, ept: 160.5812
    Epoch [7/50], Val Losses: mse: 44.1974, mae: 4.3691, huber: 3.9195, swd: 11.3587, ept: 150.6254
    Epoch [7/50], Test Losses: mse: 42.3459, mae: 4.2209, huber: 3.7734, swd: 11.1760, ept: 155.2221
      Epoch 7 composite train-obj: 3.681551
            Val objective improved 4.0765 → 3.9195, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 40.0191, mae: 4.0791, huber: 3.6305, swd: 10.3306, ept: 165.2885
    Epoch [8/50], Val Losses: mse: 44.2212, mae: 4.3887, huber: 3.9341, swd: 10.6679, ept: 155.2638
    Epoch [8/50], Test Losses: mse: 42.0432, mae: 4.2161, huber: 3.7629, swd: 9.8328, ept: 160.7012
      Epoch 8 composite train-obj: 3.630513
            No improvement (3.9341), counter 1/5
    Epoch [9/50], Train Losses: mse: 38.9770, mae: 3.9865, huber: 3.5402, swd: 9.7184, ept: 170.5029
    Epoch [9/50], Val Losses: mse: 42.7556, mae: 4.2735, huber: 3.8248, swd: 10.4768, ept: 157.3328
    Epoch [9/50], Test Losses: mse: 40.6161, mae: 4.1282, huber: 3.6809, swd: 10.3085, ept: 162.5665
      Epoch 9 composite train-obj: 3.540178
            Val objective improved 3.9195 → 3.8248, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 38.3957, mae: 3.9315, huber: 3.4865, swd: 9.2950, ept: 173.9179
    Epoch [10/50], Val Losses: mse: 43.1032, mae: 4.2415, huber: 3.7924, swd: 9.0390, ept: 161.3886
    Epoch [10/50], Test Losses: mse: 40.7552, mae: 4.0655, huber: 3.6186, swd: 8.4206, ept: 166.4731
      Epoch 10 composite train-obj: 3.486469
            Val objective improved 3.8248 → 3.7924, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 37.6384, mae: 3.8699, huber: 3.4262, swd: 8.8611, ept: 177.5443
    Epoch [11/50], Val Losses: mse: 41.4667, mae: 4.1965, huber: 3.7479, swd: 9.3751, ept: 163.8864
    Epoch [11/50], Test Losses: mse: 39.3325, mae: 4.0539, huber: 3.6070, swd: 9.3536, ept: 169.4632
      Epoch 11 composite train-obj: 3.426226
            Val objective improved 3.7924 → 3.7479, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 36.9401, mae: 3.8180, huber: 3.3752, swd: 8.5164, ept: 180.3908
    Epoch [12/50], Val Losses: mse: 41.3855, mae: 4.1453, huber: 3.6981, swd: 8.8821, ept: 165.8036
    Epoch [12/50], Test Losses: mse: 39.0231, mae: 3.9932, huber: 3.5477, swd: 8.7878, ept: 171.3118
      Epoch 12 composite train-obj: 3.375167
            Val objective improved 3.7479 → 3.6981, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 36.4968, mae: 3.7780, huber: 3.3361, swd: 8.2091, ept: 182.7924
    Epoch [13/50], Val Losses: mse: 41.5235, mae: 4.1210, huber: 3.6734, swd: 7.7179, ept: 169.0372
    Epoch [13/50], Test Losses: mse: 38.3086, mae: 3.9014, huber: 3.4564, swd: 7.2012, ept: 175.5959
      Epoch 13 composite train-obj: 3.336129
            Val objective improved 3.6981 → 3.6734, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 35.9542, mae: 3.7385, huber: 3.2973, swd: 7.9412, ept: 184.6860
    Epoch [14/50], Val Losses: mse: 41.8986, mae: 4.1160, huber: 3.6690, swd: 7.7618, ept: 171.0159
    Epoch [14/50], Test Losses: mse: 39.3080, mae: 3.9371, huber: 3.4920, swd: 7.5199, ept: 177.5897
      Epoch 14 composite train-obj: 3.297276
            Val objective improved 3.6734 → 3.6690, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 35.9740, mae: 3.7369, huber: 3.2957, swd: 8.0806, ept: 185.2021
    Epoch [15/50], Val Losses: mse: 39.7399, mae: 4.0158, huber: 3.5718, swd: 8.1456, ept: 172.7576
    Epoch [15/50], Test Losses: mse: 37.0477, mae: 3.8379, huber: 3.3964, swd: 7.8457, ept: 178.5428
      Epoch 15 composite train-obj: 3.295726
            Val objective improved 3.6690 → 3.5718, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 34.9845, mae: 3.6576, huber: 3.2180, swd: 7.4964, ept: 189.4656
    Epoch [16/50], Val Losses: mse: 40.5124, mae: 4.0598, huber: 3.6112, swd: 7.1154, ept: 178.6150
    Epoch [16/50], Test Losses: mse: 37.9324, mae: 3.8795, huber: 3.4333, swd: 6.6363, ept: 185.3265
      Epoch 16 composite train-obj: 3.218015
            No improvement (3.6112), counter 1/5
    Epoch [17/50], Train Losses: mse: 34.7690, mae: 3.6341, huber: 3.1954, swd: 7.3227, ept: 191.0666
    Epoch [17/50], Val Losses: mse: 39.1655, mae: 3.9153, huber: 3.4758, swd: 7.4866, ept: 178.3459
    Epoch [17/50], Test Losses: mse: 36.1051, mae: 3.7189, huber: 3.2822, swd: 7.1408, ept: 185.1716
      Epoch 17 composite train-obj: 3.195418
            Val objective improved 3.5718 → 3.4758, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 34.1097, mae: 3.5795, huber: 3.1423, swd: 7.0846, ept: 193.8930
    Epoch [18/50], Val Losses: mse: 40.4735, mae: 4.0320, huber: 3.5838, swd: 6.5221, ept: 180.8585
    Epoch [18/50], Test Losses: mse: 36.9125, mae: 3.8195, huber: 3.3730, swd: 6.1956, ept: 187.7222
      Epoch 18 composite train-obj: 3.142254
            No improvement (3.5838), counter 1/5
    Epoch [19/50], Train Losses: mse: 33.9628, mae: 3.5705, huber: 3.1333, swd: 7.0016, ept: 194.3490
    Epoch [19/50], Val Losses: mse: 40.2125, mae: 3.9529, huber: 3.5118, swd: 6.6939, ept: 182.9160
    Epoch [19/50], Test Losses: mse: 36.3718, mae: 3.7253, huber: 3.2861, swd: 6.2741, ept: 189.0001
      Epoch 19 composite train-obj: 3.133340
            No improvement (3.5118), counter 2/5
    Epoch [20/50], Train Losses: mse: 33.7301, mae: 3.5421, huber: 3.1059, swd: 6.8484, ept: 196.1207
    Epoch [20/50], Val Losses: mse: 42.0490, mae: 4.0508, huber: 3.6093, swd: 6.4555, ept: 175.7029
    Epoch [20/50], Test Losses: mse: 38.0062, mae: 3.8214, huber: 3.3820, swd: 6.2723, ept: 183.2037
      Epoch 20 composite train-obj: 3.105910
            No improvement (3.6093), counter 3/5
    Epoch [21/50], Train Losses: mse: 34.1050, mae: 3.5775, huber: 3.1403, swd: 7.0583, ept: 194.3367
    Epoch [21/50], Val Losses: mse: 39.8245, mae: 3.9288, huber: 3.4877, swd: 6.6473, ept: 183.1108
    Epoch [21/50], Test Losses: mse: 36.2318, mae: 3.6992, huber: 3.2602, swd: 5.9809, ept: 189.9130
      Epoch 21 composite train-obj: 3.140326
            No improvement (3.4877), counter 4/5
    Epoch [22/50], Train Losses: mse: 33.2729, mae: 3.5013, huber: 3.0664, swd: 6.6556, ept: 198.5946
    Epoch [22/50], Val Losses: mse: 37.7307, mae: 3.7738, huber: 3.3399, swd: 7.2096, ept: 187.4297
    Epoch [22/50], Test Losses: mse: 34.4782, mae: 3.5657, huber: 3.1341, swd: 6.7812, ept: 192.9816
      Epoch 22 composite train-obj: 3.066375
            Val objective improved 3.4758 → 3.3399, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 32.6578, mae: 3.4554, huber: 3.0213, swd: 6.4613, ept: 200.4782
    Epoch [23/50], Val Losses: mse: 37.5327, mae: 3.7918, huber: 3.3538, swd: 6.5779, ept: 187.6196
    Epoch [23/50], Test Losses: mse: 34.4210, mae: 3.5938, huber: 3.1576, swd: 6.1106, ept: 194.8727
      Epoch 23 composite train-obj: 3.021347
            No improvement (3.3538), counter 1/5
    Epoch [24/50], Train Losses: mse: 32.3609, mae: 3.4343, huber: 3.0010, swd: 6.3434, ept: 201.6720
    Epoch [24/50], Val Losses: mse: 37.7375, mae: 3.7817, huber: 3.3458, swd: 7.0898, ept: 188.1247
    Epoch [24/50], Test Losses: mse: 34.4369, mae: 3.5880, huber: 3.1540, swd: 6.9696, ept: 192.5843
      Epoch 24 composite train-obj: 3.000982
            No improvement (3.3458), counter 2/5
    Epoch [25/50], Train Losses: mse: 32.1005, mae: 3.4109, huber: 2.9782, swd: 6.2415, ept: 203.0635
    Epoch [25/50], Val Losses: mse: 40.6093, mae: 3.9479, huber: 3.5034, swd: 5.7402, ept: 187.7167
    Epoch [25/50], Test Losses: mse: 36.0550, mae: 3.6869, huber: 3.2445, swd: 5.2463, ept: 194.8471
      Epoch 25 composite train-obj: 2.978219
            No improvement (3.5034), counter 3/5
    Epoch [26/50], Train Losses: mse: 31.7547, mae: 3.3833, huber: 2.9514, swd: 6.1092, ept: 204.4497
    Epoch [26/50], Val Losses: mse: 39.2544, mae: 3.8350, huber: 3.3980, swd: 6.2526, ept: 188.7074
    Epoch [26/50], Test Losses: mse: 34.8847, mae: 3.5922, huber: 3.1575, swd: 6.0668, ept: 195.2763
      Epoch 26 composite train-obj: 2.951403
            No improvement (3.3980), counter 4/5
    Epoch [27/50], Train Losses: mse: 31.5687, mae: 3.3658, huber: 2.9343, swd: 6.0222, ept: 205.5411
    Epoch [27/50], Val Losses: mse: 37.2955, mae: 3.7123, huber: 3.2784, swd: 6.4414, ept: 189.8907
    Epoch [27/50], Test Losses: mse: 33.5535, mae: 3.4929, huber: 3.0614, swd: 6.2287, ept: 197.9602
      Epoch 27 composite train-obj: 2.934263
            Val objective improved 3.3399 → 3.2784, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 31.3368, mae: 3.3474, huber: 2.9167, swd: 5.9487, ept: 206.4437
    Epoch [28/50], Val Losses: mse: 36.2590, mae: 3.6783, huber: 3.2436, swd: 6.1633, ept: 193.3844
    Epoch [28/50], Test Losses: mse: 32.9476, mae: 3.4838, huber: 3.0509, swd: 5.7372, ept: 200.3290
      Epoch 28 composite train-obj: 2.916653
            Val objective improved 3.2784 → 3.2436, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 31.0237, mae: 3.3323, huber: 2.9016, swd: 5.8806, ept: 207.1939
    Epoch [29/50], Val Losses: mse: 37.4878, mae: 3.7778, huber: 3.3401, swd: 6.3024, ept: 189.8345
    Epoch [29/50], Test Losses: mse: 33.6821, mae: 3.5641, huber: 3.1287, swd: 6.0346, ept: 194.9317
      Epoch 29 composite train-obj: 2.901569
            No improvement (3.3401), counter 1/5
    Epoch [30/50], Train Losses: mse: 30.8787, mae: 3.3236, huber: 2.8928, swd: 5.8302, ept: 207.9028
    Epoch [30/50], Val Losses: mse: 38.5142, mae: 3.7631, huber: 3.3285, swd: 5.4157, ept: 193.6597
    Epoch [30/50], Test Losses: mse: 34.6477, mae: 3.5415, huber: 3.1096, swd: 5.2370, ept: 199.9158
      Epoch 30 composite train-obj: 2.892763
            No improvement (3.3285), counter 2/5
    Epoch [31/50], Train Losses: mse: 30.5034, mae: 3.2870, huber: 2.8574, swd: 5.7032, ept: 209.3988
    Epoch [31/50], Val Losses: mse: 34.9209, mae: 3.6072, huber: 3.1750, swd: 6.6423, ept: 194.1174
    Epoch [31/50], Test Losses: mse: 31.9644, mae: 3.4225, huber: 2.9921, swd: 6.3319, ept: 201.0566
      Epoch 31 composite train-obj: 2.857367
            Val objective improved 3.2436 → 3.1750, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 30.5795, mae: 3.2966, huber: 2.8667, swd: 5.7544, ept: 209.0500
    Epoch [32/50], Val Losses: mse: 37.7828, mae: 3.7485, huber: 3.3082, swd: 5.5970, ept: 194.7494
    Epoch [32/50], Test Losses: mse: 34.0943, mae: 3.5319, huber: 3.0936, swd: 5.4113, ept: 201.6844
      Epoch 32 composite train-obj: 2.866702
            No improvement (3.3082), counter 1/5
    Epoch [33/50], Train Losses: mse: 30.5311, mae: 3.2864, huber: 2.8565, swd: 5.7558, ept: 209.7329
    Epoch [33/50], Val Losses: mse: 37.6683, mae: 3.7876, huber: 3.3488, swd: 6.4378, ept: 188.5066
    Epoch [33/50], Test Losses: mse: 33.8059, mae: 3.5754, huber: 3.1394, swd: 6.3950, ept: 194.7716
      Epoch 33 composite train-obj: 2.856530
            No improvement (3.3488), counter 2/5
    Epoch [34/50], Train Losses: mse: 30.0752, mae: 3.2569, huber: 2.8281, swd: 5.5865, ept: 210.6746
    Epoch [34/50], Val Losses: mse: 37.3730, mae: 3.7247, huber: 3.2885, swd: 5.8571, ept: 192.6668
    Epoch [34/50], Test Losses: mse: 33.3722, mae: 3.5078, huber: 3.0730, swd: 5.6631, ept: 198.0743
      Epoch 34 composite train-obj: 2.828069
            No improvement (3.2885), counter 3/5
    Epoch [35/50], Train Losses: mse: 30.0899, mae: 3.2570, huber: 2.8284, swd: 5.5387, ept: 210.9697
    Epoch [35/50], Val Losses: mse: 37.1431, mae: 3.8121, huber: 3.3692, swd: 6.8996, ept: 186.2882
    Epoch [35/50], Test Losses: mse: 33.3903, mae: 3.6103, huber: 3.1696, swd: 6.8535, ept: 194.0193
      Epoch 35 composite train-obj: 2.828362
            No improvement (3.3692), counter 4/5
    Epoch [36/50], Train Losses: mse: 31.0824, mae: 3.3554, huber: 2.9238, swd: 6.1545, ept: 206.0388
    Epoch [36/50], Val Losses: mse: 37.8413, mae: 3.7611, huber: 3.3233, swd: 5.3932, ept: 191.8308
    Epoch [36/50], Test Losses: mse: 33.4101, mae: 3.5300, huber: 3.0942, swd: 5.3666, ept: 198.4634
      Epoch 36 composite train-obj: 2.923773
    Epoch [36/50], Test Losses: mse: 31.9644, mae: 3.4225, huber: 2.9921, swd: 6.3319, ept: 201.0566
    Best round's Test MSE: 31.9644, MAE: 3.4225, SWD: 6.3319
    Best round's Validation MSE: 34.9209, MAE: 3.6072, SWD: 6.6423
    Best round's Test verification MSE : 31.9644, MAE: 3.4225, SWD: 6.3319
    Time taken: 296.00 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq96_pred336_20250513_0504)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 30.7690 ± 0.8496
      mae: 3.3360 ± 0.0640
      huber: 2.9033 ± 0.0656
      swd: 4.8611 ± 1.2307
      ept: 207.9292 ± 6.0368
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 34.0412 ± 0.8538
      mae: 3.5255 ± 0.0583
      huber: 3.0914 ± 0.0596
      swd: 5.0979 ± 1.2728
      ept: 199.8588 ± 4.8855
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 926.32 seconds
    
    Experiment complete: TimeMixer_lorenz_seq96_pred336_20250513_0504
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 96-720
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 279
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 279
    Validation Batches: 35
    Test Batches: 75
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 83.0910, mae: 6.9145, huber: 6.4328, swd: 15.5470, ept: 60.3700
    Epoch [1/50], Val Losses: mse: 67.6639, mae: 6.2780, huber: 5.7994, swd: 24.7365, ept: 50.2644
    Epoch [1/50], Test Losses: mse: 66.8622, mae: 6.2185, huber: 5.7405, swd: 25.5570, ept: 49.1743
      Epoch 1 composite train-obj: 6.432806
            Val objective improved inf → 5.7994, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 63.8393, mae: 6.0070, huber: 5.5302, swd: 22.9816, ept: 83.8531
    Epoch [2/50], Val Losses: mse: 63.9156, mae: 6.0684, huber: 5.5909, swd: 26.5498, ept: 69.5444
    Epoch [2/50], Test Losses: mse: 64.2788, mae: 6.0902, huber: 5.6128, swd: 27.0762, ept: 67.7266
      Epoch 2 composite train-obj: 5.530229
            Val objective improved 5.7994 → 5.5909, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 60.6135, mae: 5.8002, huber: 5.3257, swd: 22.9499, ept: 100.7089
    Epoch [3/50], Val Losses: mse: 63.5018, mae: 5.9732, huber: 5.4980, swd: 21.8149, ept: 96.7075
    Epoch [3/50], Test Losses: mse: 61.8740, mae: 5.8243, huber: 5.3503, swd: 21.1676, ept: 95.1829
      Epoch 3 composite train-obj: 5.325660
            Val objective improved 5.5909 → 5.4980, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 59.4320, mae: 5.7041, huber: 5.2310, swd: 22.3432, ept: 111.6998
    Epoch [4/50], Val Losses: mse: 60.8324, mae: 5.7706, huber: 5.2982, swd: 23.0928, ept: 106.3813
    Epoch [4/50], Test Losses: mse: 59.6795, mae: 5.6529, huber: 5.1813, swd: 22.5732, ept: 105.6547
      Epoch 4 composite train-obj: 5.230968
            Val objective improved 5.4980 → 5.2982, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 58.4766, mae: 5.6159, huber: 5.1443, swd: 21.5960, ept: 120.8760
    Epoch [5/50], Val Losses: mse: 59.5192, mae: 5.6898, huber: 5.2189, swd: 22.4155, ept: 115.2193
    Epoch [5/50], Test Losses: mse: 58.4657, mae: 5.5831, huber: 5.1129, swd: 21.9853, ept: 113.9754
      Epoch 5 composite train-obj: 5.144252
            Val objective improved 5.2982 → 5.2189, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 57.7490, mae: 5.5443, huber: 5.0739, swd: 20.8153, ept: 128.3467
    Epoch [6/50], Val Losses: mse: 59.9814, mae: 5.7052, huber: 5.2335, swd: 22.3872, ept: 115.5153
    Epoch [6/50], Test Losses: mse: 59.5952, mae: 5.6616, huber: 5.1907, swd: 22.6665, ept: 113.3979
      Epoch 6 composite train-obj: 5.073862
            No improvement (5.2335), counter 1/5
    Epoch [7/50], Train Losses: mse: 57.2368, mae: 5.4960, huber: 5.0264, swd: 20.1629, ept: 133.9361
    Epoch [7/50], Val Losses: mse: 59.5956, mae: 5.6853, huber: 5.2131, swd: 21.6432, ept: 118.1055
    Epoch [7/50], Test Losses: mse: 59.2400, mae: 5.6530, huber: 5.1813, swd: 21.9599, ept: 116.1282
      Epoch 7 composite train-obj: 5.026394
            Val objective improved 5.2189 → 5.2131, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 56.7521, mae: 5.4497, huber: 4.9808, swd: 19.6920, ept: 139.4850
    Epoch [8/50], Val Losses: mse: 61.4336, mae: 5.8853, huber: 5.4086, swd: 22.8494, ept: 107.1443
    Epoch [8/50], Test Losses: mse: 61.6094, mae: 5.9045, huber: 5.4282, swd: 23.7908, ept: 103.5025
      Epoch 8 composite train-obj: 4.980827
            No improvement (5.4086), counter 1/5
    Epoch [9/50], Train Losses: mse: 56.2913, mae: 5.4086, huber: 4.9404, swd: 19.0793, ept: 143.5191
    Epoch [9/50], Val Losses: mse: 57.8386, mae: 5.5161, huber: 5.0479, swd: 20.5535, ept: 132.8179
    Epoch [9/50], Test Losses: mse: 56.9854, mae: 5.4357, huber: 4.9685, swd: 20.7097, ept: 133.0426
      Epoch 9 composite train-obj: 4.940423
            Val objective improved 5.2131 → 5.0479, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 55.3801, mae: 5.3374, huber: 4.8704, swd: 18.5911, ept: 150.8322
    Epoch [10/50], Val Losses: mse: 59.7760, mae: 5.7165, huber: 5.2433, swd: 20.1301, ept: 124.0721
    Epoch [10/50], Test Losses: mse: 59.5957, mae: 5.6896, huber: 5.2169, swd: 20.4585, ept: 123.2276
      Epoch 10 composite train-obj: 4.870352
            No improvement (5.2433), counter 1/5
    Epoch [11/50], Train Losses: mse: 55.1331, mae: 5.3157, huber: 4.8489, swd: 18.1442, ept: 154.2701
    Epoch [11/50], Val Losses: mse: 57.6652, mae: 5.4678, huber: 5.0005, swd: 18.1434, ept: 141.8183
    Epoch [11/50], Test Losses: mse: 56.6259, mae: 5.3598, huber: 4.8937, swd: 17.8625, ept: 141.6226
      Epoch 11 composite train-obj: 4.848920
            Val objective improved 5.0479 → 5.0005, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 54.6116, mae: 5.2672, huber: 4.8012, swd: 17.5380, ept: 159.9941
    Epoch [12/50], Val Losses: mse: 56.5302, mae: 5.4074, huber: 4.9410, swd: 18.2218, ept: 144.5442
    Epoch [12/50], Test Losses: mse: 55.3701, mae: 5.2893, huber: 4.8240, swd: 17.8207, ept: 145.4915
      Epoch 12 composite train-obj: 4.801245
            Val objective improved 5.0005 → 4.9410, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 53.9736, mae: 5.2187, huber: 4.7536, swd: 17.1888, ept: 165.0163
    Epoch [13/50], Val Losses: mse: 56.6593, mae: 5.4017, huber: 4.9346, swd: 17.3372, ept: 150.6407
    Epoch [13/50], Test Losses: mse: 55.5117, mae: 5.2864, huber: 4.8204, swd: 16.9565, ept: 150.6924
      Epoch 13 composite train-obj: 4.753569
            Val objective improved 4.9410 → 4.9346, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 53.6366, mae: 5.1887, huber: 4.7241, swd: 16.7626, ept: 168.2374
    Epoch [14/50], Val Losses: mse: 56.4368, mae: 5.3663, huber: 4.9003, swd: 17.0191, ept: 153.6896
    Epoch [14/50], Test Losses: mse: 55.5318, mae: 5.2717, huber: 4.8070, swd: 17.0410, ept: 154.5185
      Epoch 14 composite train-obj: 4.724118
            Val objective improved 4.9346 → 4.9003, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 53.2088, mae: 5.1512, huber: 4.6873, swd: 16.3857, ept: 173.2285
    Epoch [15/50], Val Losses: mse: 56.9580, mae: 5.4233, huber: 4.9550, swd: 16.5510, ept: 154.7959
    Epoch [15/50], Test Losses: mse: 56.3318, mae: 5.3586, huber: 4.8911, swd: 16.6613, ept: 155.9869
      Epoch 15 composite train-obj: 4.687252
            No improvement (4.9550), counter 1/5
    Epoch [16/50], Train Losses: mse: 52.9311, mae: 5.1279, huber: 4.6643, swd: 15.9495, ept: 176.1336
    Epoch [16/50], Val Losses: mse: 55.9951, mae: 5.3366, huber: 4.8707, swd: 15.8400, ept: 162.5040
    Epoch [16/50], Test Losses: mse: 54.6689, mae: 5.2036, huber: 4.7390, swd: 15.3640, ept: 162.3778
      Epoch 16 composite train-obj: 4.664254
            Val objective improved 4.9003 → 4.8707, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 52.5957, mae: 5.0984, huber: 4.6355, swd: 15.5972, ept: 180.1173
    Epoch [17/50], Val Losses: mse: 55.8929, mae: 5.3598, huber: 4.8926, swd: 15.9574, ept: 161.5459
    Epoch [17/50], Test Losses: mse: 55.2943, mae: 5.3017, huber: 4.8350, swd: 16.2952, ept: 162.4280
      Epoch 17 composite train-obj: 4.635458
            No improvement (4.8926), counter 1/5
    Epoch [18/50], Train Losses: mse: 52.6279, mae: 5.0982, huber: 4.6353, swd: 15.3521, ept: 181.2202
    Epoch [18/50], Val Losses: mse: 56.2158, mae: 5.4307, huber: 4.9617, swd: 17.5879, ept: 155.6407
    Epoch [18/50], Test Losses: mse: 55.7533, mae: 5.3866, huber: 4.9180, swd: 18.0244, ept: 155.2559
      Epoch 18 composite train-obj: 4.635270
            No improvement (4.9617), counter 2/5
    Epoch [19/50], Train Losses: mse: 52.3111, mae: 5.0712, huber: 4.6087, swd: 15.0945, ept: 184.0401
    Epoch [19/50], Val Losses: mse: 54.8530, mae: 5.2466, huber: 4.7827, swd: 15.0444, ept: 170.1011
    Epoch [19/50], Test Losses: mse: 53.6415, mae: 5.1279, huber: 4.6654, swd: 14.8741, ept: 170.8543
      Epoch 19 composite train-obj: 4.608665
            Val objective improved 4.8707 → 4.7827, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 51.9362, mae: 5.0381, huber: 4.5763, swd: 14.7852, ept: 188.4423
    Epoch [20/50], Val Losses: mse: 54.2244, mae: 5.2205, huber: 4.7577, swd: 16.3141, ept: 168.6105
    Epoch [20/50], Test Losses: mse: 53.4938, mae: 5.1460, huber: 4.6844, swd: 16.5161, ept: 170.5992
      Epoch 20 composite train-obj: 4.576271
            Val objective improved 4.7827 → 4.7577, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 51.6818, mae: 5.0165, huber: 4.5551, swd: 14.5548, ept: 190.6579
    Epoch [21/50], Val Losses: mse: 54.4732, mae: 5.2225, huber: 4.7594, swd: 15.3338, ept: 176.4758
    Epoch [21/50], Test Losses: mse: 53.5791, mae: 5.1342, huber: 4.6725, swd: 15.3385, ept: 176.1055
      Epoch 21 composite train-obj: 4.555130
            No improvement (4.7594), counter 1/5
    Epoch [22/50], Train Losses: mse: 51.4066, mae: 4.9963, huber: 4.5352, swd: 14.3681, ept: 192.4340
    Epoch [22/50], Val Losses: mse: 55.2474, mae: 5.2480, huber: 4.7828, swd: 13.8828, ept: 176.5176
    Epoch [22/50], Test Losses: mse: 53.8724, mae: 5.1173, huber: 4.6532, swd: 13.6493, ept: 178.4172
      Epoch 22 composite train-obj: 4.535227
            No improvement (4.7828), counter 2/5
    Epoch [23/50], Train Losses: mse: 51.2675, mae: 4.9788, huber: 4.5181, swd: 14.0648, ept: 195.3216
    Epoch [23/50], Val Losses: mse: 55.1072, mae: 5.2333, huber: 4.7690, swd: 13.8407, ept: 179.0123
    Epoch [23/50], Test Losses: mse: 53.7353, mae: 5.0985, huber: 4.6354, swd: 13.3253, ept: 180.1416
      Epoch 23 composite train-obj: 4.518092
            No improvement (4.7690), counter 3/5
    Epoch [24/50], Train Losses: mse: 51.0824, mae: 4.9629, huber: 4.5026, swd: 13.8863, ept: 196.8143
    Epoch [24/50], Val Losses: mse: 54.7905, mae: 5.2368, huber: 4.7716, swd: 13.6954, ept: 179.4871
    Epoch [24/50], Test Losses: mse: 53.3639, mae: 5.1028, huber: 4.6384, swd: 13.1427, ept: 180.9871
      Epoch 24 composite train-obj: 4.502589
            No improvement (4.7716), counter 4/5
    Epoch [25/50], Train Losses: mse: 51.0075, mae: 4.9587, huber: 4.4984, swd: 13.8171, ept: 197.7460
    Epoch [25/50], Val Losses: mse: 54.0851, mae: 5.1581, huber: 4.6963, swd: 13.9708, ept: 182.0224
    Epoch [25/50], Test Losses: mse: 53.1058, mae: 5.0591, huber: 4.5987, swd: 13.8004, ept: 183.4675
      Epoch 25 composite train-obj: 4.498364
            Val objective improved 4.7577 → 4.6963, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 50.6997, mae: 4.9298, huber: 4.4702, swd: 13.5327, ept: 200.0076
    Epoch [26/50], Val Losses: mse: 54.0322, mae: 5.1538, huber: 4.6915, swd: 13.7084, ept: 180.8430
    Epoch [26/50], Test Losses: mse: 52.9926, mae: 5.0510, huber: 4.5900, swd: 13.6084, ept: 185.6315
      Epoch 26 composite train-obj: 4.470166
            Val objective improved 4.6963 → 4.6915, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 50.6817, mae: 4.9258, huber: 4.4662, swd: 13.4327, ept: 202.3833
    Epoch [27/50], Val Losses: mse: 54.6744, mae: 5.2666, huber: 4.7996, swd: 13.3279, ept: 178.3242
    Epoch [27/50], Test Losses: mse: 53.5462, mae: 5.1534, huber: 4.6870, swd: 12.6777, ept: 181.0968
      Epoch 27 composite train-obj: 4.466231
            No improvement (4.7996), counter 1/5
    Epoch [28/50], Train Losses: mse: 50.4977, mae: 4.9144, huber: 4.4549, swd: 13.3163, ept: 202.9680
    Epoch [28/50], Val Losses: mse: 53.1966, mae: 5.0991, huber: 4.6389, swd: 13.8185, ept: 186.3064
    Epoch [28/50], Test Losses: mse: 52.2822, mae: 5.0000, huber: 4.5415, swd: 13.6942, ept: 189.9881
      Epoch 28 composite train-obj: 4.454901
            Val objective improved 4.6915 → 4.6389, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 50.1958, mae: 4.8859, huber: 4.4272, swd: 13.1647, ept: 206.2940
    Epoch [29/50], Val Losses: mse: 53.4031, mae: 5.1292, huber: 4.6672, swd: 14.1814, ept: 183.7935
    Epoch [29/50], Test Losses: mse: 52.4624, mae: 5.0383, huber: 4.5773, swd: 14.0691, ept: 187.2204
      Epoch 29 composite train-obj: 4.427163
            No improvement (4.6672), counter 1/5
    Epoch [30/50], Train Losses: mse: 50.1190, mae: 4.8784, huber: 4.4197, swd: 13.0285, ept: 207.8015
    Epoch [30/50], Val Losses: mse: 53.1296, mae: 5.0928, huber: 4.6321, swd: 13.4224, ept: 186.5316
    Epoch [30/50], Test Losses: mse: 52.2424, mae: 4.9975, huber: 4.5383, swd: 13.2552, ept: 190.2267
      Epoch 30 composite train-obj: 4.419719
            Val objective improved 4.6389 → 4.6321, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 49.8868, mae: 4.8597, huber: 4.4014, swd: 12.9409, ept: 208.9762
    Epoch [31/50], Val Losses: mse: 53.7772, mae: 5.1314, huber: 4.6703, swd: 13.0068, ept: 185.0710
    Epoch [31/50], Test Losses: mse: 52.4351, mae: 5.0091, huber: 4.5492, swd: 12.7699, ept: 190.7886
      Epoch 31 composite train-obj: 4.401414
            No improvement (4.6703), counter 1/5
    Epoch [32/50], Train Losses: mse: 49.8374, mae: 4.8570, huber: 4.3987, swd: 12.8522, ept: 209.7997
    Epoch [32/50], Val Losses: mse: 53.8730, mae: 5.1493, huber: 4.6875, swd: 13.1572, ept: 183.6020
    Epoch [32/50], Test Losses: mse: 52.8641, mae: 5.0468, huber: 4.5861, swd: 13.1414, ept: 188.1476
      Epoch 32 composite train-obj: 4.398727
            No improvement (4.6875), counter 2/5
    Epoch [33/50], Train Losses: mse: 50.0436, mae: 4.8763, huber: 4.4175, swd: 12.8669, ept: 208.3613
    Epoch [33/50], Val Losses: mse: 54.4906, mae: 5.1978, huber: 4.7322, swd: 11.7883, ept: 189.4655
    Epoch [33/50], Test Losses: mse: 53.0290, mae: 5.0570, huber: 4.5929, swd: 11.2023, ept: 192.5213
      Epoch 33 composite train-obj: 4.417539
            No improvement (4.7322), counter 3/5
    Epoch [34/50], Train Losses: mse: 49.6227, mae: 4.8370, huber: 4.3791, swd: 12.6076, ept: 212.2159
    Epoch [34/50], Val Losses: mse: 53.0601, mae: 5.0970, huber: 4.6356, swd: 13.5578, ept: 183.9720
    Epoch [34/50], Test Losses: mse: 52.0628, mae: 5.0048, huber: 4.5443, swd: 13.4925, ept: 188.5864
      Epoch 34 composite train-obj: 4.379120
            No improvement (4.6356), counter 4/5
    Epoch [35/50], Train Losses: mse: 49.5071, mae: 4.8278, huber: 4.3703, swd: 12.6507, ept: 212.6271
    Epoch [35/50], Val Losses: mse: 53.1043, mae: 5.0789, huber: 4.6181, swd: 12.7805, ept: 191.2296
    Epoch [35/50], Test Losses: mse: 51.9572, mae: 4.9602, huber: 4.5005, swd: 12.3506, ept: 194.7062
      Epoch 35 composite train-obj: 4.370287
            Val objective improved 4.6321 → 4.6181, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 49.3338, mae: 4.8119, huber: 4.3546, swd: 12.4628, ept: 214.4413
    Epoch [36/50], Val Losses: mse: 52.8765, mae: 5.0605, huber: 4.6009, swd: 12.8402, ept: 191.8491
    Epoch [36/50], Test Losses: mse: 51.8912, mae: 4.9611, huber: 4.5028, swd: 12.6675, ept: 195.2722
      Epoch 36 composite train-obj: 4.354601
            Val objective improved 4.6181 → 4.6009, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 49.3840, mae: 4.8144, huber: 4.3572, swd: 12.4223, ept: 214.7847
    Epoch [37/50], Val Losses: mse: 52.8942, mae: 5.1102, huber: 4.6474, swd: 13.3206, ept: 188.2033
    Epoch [37/50], Test Losses: mse: 51.5207, mae: 4.9803, huber: 4.5184, swd: 12.8119, ept: 192.8052
      Epoch 37 composite train-obj: 4.357175
            No improvement (4.6474), counter 1/5
    Epoch [38/50], Train Losses: mse: 49.1471, mae: 4.7982, huber: 4.3412, swd: 12.3025, ept: 215.9855
    Epoch [38/50], Val Losses: mse: 53.0559, mae: 5.0917, huber: 4.6294, swd: 12.9807, ept: 189.2476
    Epoch [38/50], Test Losses: mse: 52.3270, mae: 5.0115, huber: 4.5502, swd: 12.8310, ept: 193.9982
      Epoch 38 composite train-obj: 4.341155
            No improvement (4.6294), counter 2/5
    Epoch [39/50], Train Losses: mse: 48.9531, mae: 4.7851, huber: 4.3283, swd: 12.2510, ept: 217.0846
    Epoch [39/50], Val Losses: mse: 52.8668, mae: 5.0644, huber: 4.6038, swd: 11.8296, ept: 190.7716
    Epoch [39/50], Test Losses: mse: 51.8882, mae: 4.9610, huber: 4.5013, swd: 11.3967, ept: 196.0187
      Epoch 39 composite train-obj: 4.328318
            No improvement (4.6038), counter 3/5
    Epoch [40/50], Train Losses: mse: 49.0307, mae: 4.7890, huber: 4.3322, swd: 12.2088, ept: 217.6879
    Epoch [40/50], Val Losses: mse: 52.9713, mae: 5.0613, huber: 4.6005, swd: 11.5960, ept: 195.1197
    Epoch [40/50], Test Losses: mse: 51.8520, mae: 4.9444, huber: 4.4849, swd: 11.1152, ept: 199.9399
      Epoch 40 composite train-obj: 4.332157
            Val objective improved 4.6009 → 4.6005, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 48.6299, mae: 4.7559, huber: 4.2997, swd: 12.0051, ept: 220.3519
    Epoch [41/50], Val Losses: mse: 53.0391, mae: 5.0502, huber: 4.5900, swd: 11.4477, ept: 196.5451
    Epoch [41/50], Test Losses: mse: 51.7593, mae: 4.9290, huber: 4.4701, swd: 10.8868, ept: 200.6063
      Epoch 41 composite train-obj: 4.299717
            Val objective improved 4.6005 → 4.5900, saving checkpoint.
    Epoch [42/50], Train Losses: mse: 48.5478, mae: 4.7488, huber: 4.2928, swd: 11.9116, ept: 221.4636
    Epoch [42/50], Val Losses: mse: 52.4804, mae: 5.0256, huber: 4.5659, swd: 12.1217, ept: 195.9569
    Epoch [42/50], Test Losses: mse: 51.1674, mae: 4.9020, huber: 4.4432, swd: 11.6256, ept: 199.8153
      Epoch 42 composite train-obj: 4.292782
            Val objective improved 4.5900 → 4.5659, saving checkpoint.
    Epoch [43/50], Train Losses: mse: 48.4650, mae: 4.7455, huber: 4.2896, swd: 11.9432, ept: 221.8468
    Epoch [43/50], Val Losses: mse: 52.4053, mae: 5.0284, huber: 4.5687, swd: 12.1077, ept: 199.0633
    Epoch [43/50], Test Losses: mse: 51.3889, mae: 4.9318, huber: 4.4733, swd: 11.8527, ept: 203.2400
      Epoch 43 composite train-obj: 4.289628
            No improvement (4.5687), counter 1/5
    Epoch [44/50], Train Losses: mse: 48.2199, mae: 4.7268, huber: 4.2712, swd: 11.8129, ept: 223.6523
    Epoch [44/50], Val Losses: mse: 53.4137, mae: 5.0865, huber: 4.6258, swd: 11.9550, ept: 192.7096
    Epoch [44/50], Test Losses: mse: 52.2943, mae: 4.9795, huber: 4.5199, swd: 11.6228, ept: 194.9793
      Epoch 44 composite train-obj: 4.271189
            No improvement (4.6258), counter 2/5
    Epoch [45/50], Train Losses: mse: 48.1386, mae: 4.7169, huber: 4.2616, swd: 11.6743, ept: 224.2502
    Epoch [45/50], Val Losses: mse: 52.4563, mae: 5.0261, huber: 4.5666, swd: 11.7747, ept: 197.5658
    Epoch [45/50], Test Losses: mse: 51.2948, mae: 4.9105, huber: 4.4526, swd: 11.5163, ept: 202.7553
      Epoch 45 composite train-obj: 4.261599
            No improvement (4.5666), counter 3/5
    Epoch [46/50], Train Losses: mse: 48.0251, mae: 4.7100, huber: 4.2547, swd: 11.6363, ept: 225.4335
    Epoch [46/50], Val Losses: mse: 51.8752, mae: 4.9763, huber: 4.5178, swd: 12.6433, ept: 199.6219
    Epoch [46/50], Test Losses: mse: 50.7512, mae: 4.8642, huber: 4.4073, swd: 12.2669, ept: 204.2126
      Epoch 46 composite train-obj: 4.254712
            Val objective improved 4.5659 → 4.5178, saving checkpoint.
    Epoch [47/50], Train Losses: mse: 47.8333, mae: 4.6960, huber: 4.2411, swd: 11.5756, ept: 226.2969
    Epoch [47/50], Val Losses: mse: 52.7978, mae: 5.0376, huber: 4.5773, swd: 11.8591, ept: 196.9383
    Epoch [47/50], Test Losses: mse: 52.0318, mae: 4.9609, huber: 4.5016, swd: 11.7521, ept: 202.4172
      Epoch 47 composite train-obj: 4.241115
            No improvement (4.5773), counter 1/5
    Epoch [48/50], Train Losses: mse: 47.8746, mae: 4.6993, huber: 4.2443, swd: 11.5220, ept: 226.1258
    Epoch [48/50], Val Losses: mse: 51.8935, mae: 4.9611, huber: 4.5042, swd: 12.0905, ept: 200.7663
    Epoch [48/50], Test Losses: mse: 50.8542, mae: 4.8521, huber: 4.3963, swd: 11.6560, ept: 206.2062
      Epoch 48 composite train-obj: 4.244303
            Val objective improved 4.5178 → 4.5042, saving checkpoint.
    Epoch [49/50], Train Losses: mse: 47.7451, mae: 4.6876, huber: 4.2328, swd: 11.4100, ept: 227.7492
    Epoch [49/50], Val Losses: mse: 51.9458, mae: 5.0294, huber: 4.5672, swd: 12.6162, ept: 195.7866
    Epoch [49/50], Test Losses: mse: 50.9799, mae: 4.9304, huber: 4.4695, swd: 12.2268, ept: 201.0271
      Epoch 49 composite train-obj: 4.232840
            No improvement (4.5672), counter 1/5
    Epoch [50/50], Train Losses: mse: 47.7242, mae: 4.6861, huber: 4.2312, swd: 11.3359, ept: 227.9462
    Epoch [50/50], Val Losses: mse: 53.0210, mae: 5.0704, huber: 4.6089, swd: 11.7039, ept: 195.2043
    Epoch [50/50], Test Losses: mse: 51.6166, mae: 4.9487, huber: 4.4880, swd: 11.1767, ept: 199.1638
      Epoch 50 composite train-obj: 4.231178
            No improvement (4.6089), counter 2/5
    Epoch [50/50], Test Losses: mse: 50.8542, mae: 4.8521, huber: 4.3963, swd: 11.6560, ept: 206.2062
    Best round's Test MSE: 50.8542, MAE: 4.8521, SWD: 11.6560
    Best round's Validation MSE: 51.8935, MAE: 4.9611, SWD: 12.0905
    Best round's Test verification MSE : 50.8542, MAE: 4.8521, SWD: 11.6560
    Time taken: 430.15 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.2270, mae: 6.5519, huber: 6.0719, swd: 16.7175, ept: 68.5862
    Epoch [1/50], Val Losses: mse: 65.7544, mae: 6.0890, huber: 5.6119, swd: 22.1492, ept: 71.7387
    Epoch [1/50], Test Losses: mse: 64.6118, mae: 5.9698, huber: 5.4937, swd: 21.8025, ept: 72.9832
      Epoch 1 composite train-obj: 6.071936
            Val objective improved inf → 5.6119, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.0278, mae: 5.8734, huber: 5.3982, swd: 22.4271, ept: 96.8738
    Epoch [2/50], Val Losses: mse: 62.5077, mae: 5.9593, huber: 5.4828, swd: 24.7441, ept: 83.9654
    Epoch [2/50], Test Losses: mse: 62.9255, mae: 5.9550, huber: 5.4789, swd: 24.9889, ept: 84.1348
      Epoch 2 composite train-obj: 5.398207
            Val objective improved 5.6119 → 5.4828, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.8556, mae: 5.7229, huber: 5.2497, swd: 21.9474, ept: 112.8499
    Epoch [3/50], Val Losses: mse: 60.6290, mae: 5.8294, huber: 5.3547, swd: 24.0386, ept: 92.3129
    Epoch [3/50], Test Losses: mse: 60.6711, mae: 5.7991, huber: 5.3251, swd: 24.4991, ept: 91.7786
      Epoch 3 composite train-obj: 5.249735
            Val objective improved 5.4828 → 5.3547, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 58.5978, mae: 5.6159, huber: 5.1444, swd: 21.0594, ept: 123.8751
    Epoch [4/50], Val Losses: mse: 59.7023, mae: 5.7061, huber: 5.2345, swd: 22.1021, ept: 114.8150
    Epoch [4/50], Test Losses: mse: 59.3858, mae: 5.6421, huber: 5.1713, swd: 22.4248, ept: 113.8405
      Epoch 4 composite train-obj: 5.144389
            Val objective improved 5.3547 → 5.2345, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 57.6832, mae: 5.5297, huber: 5.0598, swd: 20.1365, ept: 132.8944
    Epoch [5/50], Val Losses: mse: 59.2543, mae: 5.6373, huber: 5.1676, swd: 20.0373, ept: 122.6799
    Epoch [5/50], Test Losses: mse: 58.5873, mae: 5.5394, huber: 5.0708, swd: 19.8238, ept: 122.4979
      Epoch 5 composite train-obj: 5.059752
            Val objective improved 5.2345 → 5.1676, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 56.9920, mae: 5.4587, huber: 4.9899, swd: 19.0744, ept: 141.2718
    Epoch [6/50], Val Losses: mse: 58.8542, mae: 5.5890, huber: 5.1193, swd: 19.6566, ept: 131.7652
    Epoch [6/50], Test Losses: mse: 58.2600, mae: 5.5096, huber: 5.0405, swd: 19.7936, ept: 130.6936
      Epoch 6 composite train-obj: 4.989935
            Val objective improved 5.1676 → 5.1193, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 56.3354, mae: 5.3996, huber: 4.9319, swd: 18.3199, ept: 147.8762
    Epoch [7/50], Val Losses: mse: 58.1784, mae: 5.5634, huber: 5.0932, swd: 19.3876, ept: 134.1293
    Epoch [7/50], Test Losses: mse: 57.9495, mae: 5.5117, huber: 5.0425, swd: 19.7122, ept: 134.5519
      Epoch 7 composite train-obj: 4.931880
            Val objective improved 5.1193 → 5.0932, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 55.5391, mae: 5.3330, huber: 4.8662, swd: 17.5925, ept: 155.2385
    Epoch [8/50], Val Losses: mse: 57.8907, mae: 5.4711, huber: 5.0033, swd: 17.4069, ept: 144.8795
    Epoch [8/50], Test Losses: mse: 56.9180, mae: 5.3492, huber: 4.8827, swd: 17.0006, ept: 143.2154
      Epoch 8 composite train-obj: 4.866250
            Val objective improved 5.0932 → 5.0033, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 54.9014, mae: 5.2733, huber: 4.8077, swd: 16.9376, ept: 161.6867
    Epoch [9/50], Val Losses: mse: 58.9641, mae: 5.5255, huber: 5.0571, swd: 15.8406, ept: 146.4422
    Epoch [9/50], Test Losses: mse: 58.3752, mae: 5.4379, huber: 4.9706, swd: 15.4610, ept: 146.6886
      Epoch 9 composite train-obj: 4.807703
            No improvement (5.0571), counter 1/5
    Epoch [10/50], Train Losses: mse: 54.6973, mae: 5.2523, huber: 4.7870, swd: 16.5732, ept: 164.4279
    Epoch [10/50], Val Losses: mse: 57.4661, mae: 5.4348, huber: 4.9674, swd: 16.3866, ept: 149.4172
    Epoch [10/50], Test Losses: mse: 56.5269, mae: 5.3203, huber: 4.8538, swd: 16.1974, ept: 148.6679
      Epoch 10 composite train-obj: 4.786976
            Val objective improved 5.0033 → 4.9674, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 54.0200, mae: 5.1982, huber: 4.7336, swd: 16.0848, ept: 170.2211
    Epoch [11/50], Val Losses: mse: 56.6708, mae: 5.3745, huber: 4.9083, swd: 16.6051, ept: 156.8800
    Epoch [11/50], Test Losses: mse: 55.9622, mae: 5.2900, huber: 4.8250, swd: 16.7516, ept: 158.1444
      Epoch 11 composite train-obj: 4.733604
            Val objective improved 4.9674 → 4.9083, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 53.6733, mae: 5.1660, huber: 4.7020, swd: 15.6494, ept: 175.3476
    Epoch [12/50], Val Losses: mse: 57.3049, mae: 5.4600, huber: 4.9907, swd: 15.1904, ept: 154.0320
    Epoch [12/50], Test Losses: mse: 56.1095, mae: 5.3267, huber: 4.8584, swd: 14.4212, ept: 154.0716
      Epoch 12 composite train-obj: 4.702009
            No improvement (4.9907), counter 1/5
    Epoch [13/50], Train Losses: mse: 53.3423, mae: 5.1368, huber: 4.6732, swd: 15.3173, ept: 178.3448
    Epoch [13/50], Val Losses: mse: 55.6834, mae: 5.3200, huber: 4.8543, swd: 15.5331, ept: 161.1440
    Epoch [13/50], Test Losses: mse: 54.6882, mae: 5.2092, huber: 4.7446, swd: 15.4009, ept: 161.1539
      Epoch 13 composite train-obj: 4.673223
            Val objective improved 4.9083 → 4.8543, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 52.6754, mae: 5.0811, huber: 4.6185, swd: 14.8391, ept: 184.1810
    Epoch [14/50], Val Losses: mse: 55.2361, mae: 5.2407, huber: 4.7771, swd: 14.8380, ept: 167.7483
    Epoch [14/50], Test Losses: mse: 54.3287, mae: 5.1384, huber: 4.6758, swd: 14.6216, ept: 168.3630
      Epoch 14 composite train-obj: 4.618504
            Val objective improved 4.8543 → 4.7771, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 52.1932, mae: 5.0428, huber: 4.5809, swd: 14.4963, ept: 189.4207
    Epoch [15/50], Val Losses: mse: 54.9403, mae: 5.2574, huber: 4.7944, swd: 15.3597, ept: 169.4701
    Epoch [15/50], Test Losses: mse: 54.3228, mae: 5.1769, huber: 4.7152, swd: 15.5213, ept: 171.7360
      Epoch 15 composite train-obj: 4.580860
            No improvement (4.7944), counter 1/5
    Epoch [16/50], Train Losses: mse: 52.1419, mae: 5.0348, huber: 4.5731, swd: 14.3374, ept: 190.0382
    Epoch [16/50], Val Losses: mse: 54.6130, mae: 5.2235, huber: 4.7597, swd: 14.5050, ept: 173.9807
    Epoch [16/50], Test Losses: mse: 53.8986, mae: 5.1362, huber: 4.6734, swd: 14.3470, ept: 175.7327
      Epoch 16 composite train-obj: 4.573100
            Val objective improved 4.7771 → 4.7597, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 51.8548, mae: 5.0092, huber: 4.5480, swd: 14.0110, ept: 193.0918
    Epoch [17/50], Val Losses: mse: 55.1117, mae: 5.2168, huber: 4.7544, swd: 14.7810, ept: 174.0026
    Epoch [17/50], Test Losses: mse: 54.3005, mae: 5.1323, huber: 4.6708, swd: 14.9707, ept: 176.6158
      Epoch 17 composite train-obj: 4.547979
            Val objective improved 4.7597 → 4.7544, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 51.6644, mae: 4.9922, huber: 4.5315, swd: 13.8251, ept: 196.0109
    Epoch [18/50], Val Losses: mse: 53.8572, mae: 5.2038, huber: 4.7396, swd: 14.4485, ept: 176.9469
    Epoch [18/50], Test Losses: mse: 53.3769, mae: 5.1372, huber: 4.6738, swd: 14.4442, ept: 179.6436
      Epoch 18 composite train-obj: 4.531483
            Val objective improved 4.7544 → 4.7396, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 51.1111, mae: 4.9571, huber: 4.4968, swd: 13.6829, ept: 199.3845
    Epoch [19/50], Val Losses: mse: 53.9723, mae: 5.1460, huber: 4.6843, swd: 13.4706, ept: 181.8612
    Epoch [19/50], Test Losses: mse: 53.1360, mae: 5.0469, huber: 4.5862, swd: 13.2642, ept: 185.2103
      Epoch 19 composite train-obj: 4.496812
            Val objective improved 4.7396 → 4.6843, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 51.0188, mae: 4.9432, huber: 4.4832, swd: 13.3809, ept: 200.8797
    Epoch [20/50], Val Losses: mse: 55.0270, mae: 5.2182, huber: 4.7539, swd: 13.1149, ept: 179.8201
    Epoch [20/50], Test Losses: mse: 54.2825, mae: 5.1272, huber: 4.6639, swd: 13.0027, ept: 182.1505
      Epoch 20 composite train-obj: 4.483248
            No improvement (4.7539), counter 1/5
    Epoch [21/50], Train Losses: mse: 50.8896, mae: 4.9331, huber: 4.4733, swd: 13.2431, ept: 202.4058
    Epoch [21/50], Val Losses: mse: 54.5710, mae: 5.1551, huber: 4.6937, swd: 13.4464, ept: 182.9484
    Epoch [21/50], Test Losses: mse: 53.3998, mae: 5.0374, huber: 4.5769, swd: 13.1278, ept: 185.9064
      Epoch 21 composite train-obj: 4.473280
            No improvement (4.6937), counter 2/5
    Epoch [22/50], Train Losses: mse: 50.5465, mae: 4.9036, huber: 4.4443, swd: 13.0160, ept: 205.2197
    Epoch [22/50], Val Losses: mse: 53.0590, mae: 5.0738, huber: 4.6140, swd: 13.6670, ept: 186.6086
    Epoch [22/50], Test Losses: mse: 51.9876, mae: 4.9683, huber: 4.5095, swd: 13.4876, ept: 189.4996
      Epoch 22 composite train-obj: 4.444331
            Val objective improved 4.6843 → 4.6140, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 50.2901, mae: 4.8855, huber: 4.4267, swd: 12.9133, ept: 207.7984
    Epoch [23/50], Val Losses: mse: 53.2803, mae: 5.0949, huber: 4.6343, swd: 13.1679, ept: 190.3803
    Epoch [23/50], Test Losses: mse: 52.4093, mae: 4.9986, huber: 4.5394, swd: 13.0148, ept: 193.1947
      Epoch 23 composite train-obj: 4.426675
            No improvement (4.6343), counter 1/5
    Epoch [24/50], Train Losses: mse: 50.0479, mae: 4.8633, huber: 4.4050, swd: 12.6761, ept: 210.4318
    Epoch [24/50], Val Losses: mse: 53.0852, mae: 5.1408, huber: 4.6769, swd: 13.7924, ept: 179.7589
    Epoch [24/50], Test Losses: mse: 52.3407, mae: 5.0598, huber: 4.5967, swd: 13.7563, ept: 181.9344
      Epoch 24 composite train-obj: 4.405012
            No improvement (4.6769), counter 2/5
    Epoch [25/50], Train Losses: mse: 50.4064, mae: 4.9134, huber: 4.4536, swd: 12.8765, ept: 205.4048
    Epoch [25/50], Val Losses: mse: 53.3599, mae: 5.1071, huber: 4.6461, swd: 13.2801, ept: 186.5998
    Epoch [25/50], Test Losses: mse: 52.5569, mae: 5.0234, huber: 4.5630, swd: 13.4688, ept: 188.5935
      Epoch 25 composite train-obj: 4.453599
            No improvement (4.6461), counter 3/5
    Epoch [26/50], Train Losses: mse: 50.1375, mae: 4.8766, huber: 4.4179, swd: 12.5964, ept: 209.0424
    Epoch [26/50], Val Losses: mse: 52.6793, mae: 5.0700, huber: 4.6083, swd: 12.8047, ept: 190.3357
    Epoch [26/50], Test Losses: mse: 52.1571, mae: 4.9996, huber: 4.5388, swd: 12.8715, ept: 193.4966
      Epoch 26 composite train-obj: 4.417942
            Val objective improved 4.6140 → 4.6083, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 49.6657, mae: 4.8381, huber: 4.3800, swd: 12.3587, ept: 212.7154
    Epoch [27/50], Val Losses: mse: 53.9198, mae: 5.1364, huber: 4.6743, swd: 11.7178, ept: 189.3128
    Epoch [27/50], Test Losses: mse: 53.2525, mae: 5.0534, huber: 4.5921, swd: 11.2094, ept: 190.7396
      Epoch 27 composite train-obj: 4.380011
            No improvement (4.6743), counter 1/5
    Epoch [28/50], Train Losses: mse: 49.8601, mae: 4.8483, huber: 4.3901, swd: 12.3284, ept: 211.8255
    Epoch [28/50], Val Losses: mse: 52.7606, mae: 5.0521, huber: 4.5920, swd: 13.5545, ept: 188.8581
    Epoch [28/50], Test Losses: mse: 51.8382, mae: 4.9571, huber: 4.4984, swd: 13.7323, ept: 193.8709
      Epoch 28 composite train-obj: 4.390116
            Val objective improved 4.6083 → 4.5920, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 49.5226, mae: 4.8206, huber: 4.3630, swd: 12.1541, ept: 215.3190
    Epoch [29/50], Val Losses: mse: 52.2224, mae: 5.0169, huber: 4.5576, swd: 12.4330, ept: 196.4813
    Epoch [29/50], Test Losses: mse: 51.2583, mae: 4.9155, huber: 4.4570, swd: 12.1550, ept: 197.0327
      Epoch 29 composite train-obj: 4.362967
            Val objective improved 4.5920 → 4.5576, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 49.0446, mae: 4.7841, huber: 4.3273, swd: 11.9443, ept: 218.0055
    Epoch [30/50], Val Losses: mse: 52.2103, mae: 4.9941, huber: 4.5354, swd: 12.4184, ept: 197.9946
    Epoch [30/50], Test Losses: mse: 51.2657, mae: 4.8909, huber: 4.4336, swd: 12.2376, ept: 201.2108
      Epoch 30 composite train-obj: 4.327312
            Val objective improved 4.5576 → 4.5354, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 48.8505, mae: 4.7669, huber: 4.3105, swd: 11.7883, ept: 220.0669
    Epoch [31/50], Val Losses: mse: 52.2201, mae: 5.0360, huber: 4.5761, swd: 12.6294, ept: 197.6945
    Epoch [31/50], Test Losses: mse: 51.7269, mae: 4.9665, huber: 4.5074, swd: 12.6229, ept: 200.4640
      Epoch 31 composite train-obj: 4.310455
            No improvement (4.5761), counter 1/5
    Epoch [32/50], Train Losses: mse: 48.9104, mae: 4.7777, huber: 4.3207, swd: 11.7735, ept: 219.2950
    Epoch [32/50], Val Losses: mse: 51.8208, mae: 4.9674, huber: 4.5102, swd: 12.2417, ept: 201.5783
    Epoch [32/50], Test Losses: mse: 50.9049, mae: 4.8676, huber: 4.4117, swd: 12.0945, ept: 205.6862
      Epoch 32 composite train-obj: 4.320734
            Val objective improved 4.5354 → 4.5102, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 48.6140, mae: 4.7507, huber: 4.2944, swd: 11.6018, ept: 221.9778
    Epoch [33/50], Val Losses: mse: 51.7941, mae: 4.9668, huber: 4.5088, swd: 11.9408, ept: 201.0416
    Epoch [33/50], Test Losses: mse: 50.8666, mae: 4.8708, huber: 4.4138, swd: 11.8559, ept: 205.6090
      Epoch 33 composite train-obj: 4.294353
            Val objective improved 4.5102 → 4.5088, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 48.4545, mae: 4.7402, huber: 4.2840, swd: 11.4854, ept: 222.9034
    Epoch [34/50], Val Losses: mse: 51.7779, mae: 4.9589, huber: 4.5009, swd: 11.7038, ept: 199.7351
    Epoch [34/50], Test Losses: mse: 50.8347, mae: 4.8590, huber: 4.4022, swd: 11.2821, ept: 203.9209
      Epoch 34 composite train-obj: 4.284029
            Val objective improved 4.5088 → 4.5009, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 48.5000, mae: 4.7460, huber: 4.2897, swd: 11.5358, ept: 222.7719
    Epoch [35/50], Val Losses: mse: 53.3578, mae: 5.1009, huber: 4.6387, swd: 11.7187, ept: 191.5840
    Epoch [35/50], Test Losses: mse: 52.1636, mae: 4.9788, huber: 4.5181, swd: 11.6924, ept: 197.1779
      Epoch 35 composite train-obj: 4.289672
            No improvement (4.6387), counter 1/5
    Epoch [36/50], Train Losses: mse: 48.5248, mae: 4.7478, huber: 4.2913, swd: 11.4067, ept: 222.3622
    Epoch [36/50], Val Losses: mse: 51.0667, mae: 4.9204, huber: 4.4641, swd: 12.4844, ept: 204.1397
    Epoch [36/50], Test Losses: mse: 50.2483, mae: 4.8315, huber: 4.3764, swd: 12.4051, ept: 207.7846
      Epoch 36 composite train-obj: 4.291293
            Val objective improved 4.5009 → 4.4641, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 48.1507, mae: 4.7141, huber: 4.2586, swd: 11.2341, ept: 226.7512
    Epoch [37/50], Val Losses: mse: 51.3431, mae: 4.9665, huber: 4.5079, swd: 12.6977, ept: 200.4537
    Epoch [37/50], Test Losses: mse: 50.4307, mae: 4.8712, huber: 4.4134, swd: 12.5910, ept: 204.3397
      Epoch 37 composite train-obj: 4.258637
            No improvement (4.5079), counter 1/5
    Epoch [38/50], Train Losses: mse: 48.1406, mae: 4.7186, huber: 4.2627, swd: 11.2053, ept: 226.0755
    Epoch [38/50], Val Losses: mse: 52.0805, mae: 5.0192, huber: 4.5579, swd: 11.0943, ept: 201.5026
    Epoch [38/50], Test Losses: mse: 51.1019, mae: 4.9141, huber: 4.4536, swd: 10.7517, ept: 206.9267
      Epoch 38 composite train-obj: 4.262735
            No improvement (4.5579), counter 2/5
    Epoch [39/50], Train Losses: mse: 48.1776, mae: 4.7223, huber: 4.2663, swd: 11.1871, ept: 225.6572
    Epoch [39/50], Val Losses: mse: 53.2438, mae: 5.0631, huber: 4.6015, swd: 11.4360, ept: 202.1781
    Epoch [39/50], Test Losses: mse: 51.8586, mae: 4.9419, huber: 4.4814, swd: 11.3000, ept: 205.6703
      Epoch 39 composite train-obj: 4.266284
            No improvement (4.6015), counter 3/5
    Epoch [40/50], Train Losses: mse: 48.2287, mae: 4.7320, huber: 4.2755, swd: 11.1734, ept: 225.1381
    Epoch [40/50], Val Losses: mse: 50.6051, mae: 4.8842, huber: 4.4280, swd: 12.0100, ept: 206.5485
    Epoch [40/50], Test Losses: mse: 49.7219, mae: 4.7929, huber: 4.3375, swd: 11.8961, ept: 211.0238
      Epoch 40 composite train-obj: 4.275547
            Val objective improved 4.4641 → 4.4280, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 47.5787, mae: 4.6695, huber: 4.2149, swd: 10.8990, ept: 231.0060
    Epoch [41/50], Val Losses: mse: 51.3172, mae: 4.9331, huber: 4.4753, swd: 11.2185, ept: 207.1525
    Epoch [41/50], Test Losses: mse: 50.0194, mae: 4.8176, huber: 4.3609, swd: 11.0140, ept: 211.0418
      Epoch 41 composite train-obj: 4.214913
            No improvement (4.4753), counter 1/5
    Epoch [42/50], Train Losses: mse: 47.5963, mae: 4.6704, huber: 4.2158, swd: 10.8043, ept: 231.3294
    Epoch [42/50], Val Losses: mse: 51.4426, mae: 4.9711, huber: 4.5092, swd: 10.8110, ept: 210.2125
    Epoch [42/50], Test Losses: mse: 50.9290, mae: 4.9139, huber: 4.4528, swd: 10.7565, ept: 212.4028
      Epoch 42 composite train-obj: 4.215793
            No improvement (4.5092), counter 2/5
    Epoch [43/50], Train Losses: mse: 47.6158, mae: 4.6768, huber: 4.2218, swd: 10.7999, ept: 230.2512
    Epoch [43/50], Val Losses: mse: 50.9517, mae: 4.9032, huber: 4.4457, swd: 11.2426, ept: 211.8221
    Epoch [43/50], Test Losses: mse: 49.6522, mae: 4.7857, huber: 4.3297, swd: 10.9628, ept: 214.8837
      Epoch 43 composite train-obj: 4.221757
            No improvement (4.4457), counter 3/5
    Epoch [44/50], Train Losses: mse: 47.3464, mae: 4.6528, huber: 4.1983, swd: 10.6718, ept: 233.3004
    Epoch [44/50], Val Losses: mse: 50.4846, mae: 4.9062, huber: 4.4482, swd: 12.5986, ept: 203.6857
    Epoch [44/50], Test Losses: mse: 49.2486, mae: 4.7893, huber: 4.3321, swd: 12.2523, ept: 209.8324
      Epoch 44 composite train-obj: 4.198332
            No improvement (4.4482), counter 4/5
    Epoch [45/50], Train Losses: mse: 47.2074, mae: 4.6413, huber: 4.1870, swd: 10.5294, ept: 234.2918
    Epoch [45/50], Val Losses: mse: 50.5775, mae: 4.8620, huber: 4.4060, swd: 11.1568, ept: 212.8212
    Epoch [45/50], Test Losses: mse: 49.5797, mae: 4.7582, huber: 4.3031, swd: 10.8306, ept: 218.8927
      Epoch 45 composite train-obj: 4.187041
            Val objective improved 4.4280 → 4.4060, saving checkpoint.
    Epoch [46/50], Train Losses: mse: 47.0424, mae: 4.6319, huber: 4.1777, swd: 10.4982, ept: 234.4902
    Epoch [46/50], Val Losses: mse: 50.9162, mae: 4.8618, huber: 4.4062, swd: 10.5856, ept: 216.9503
    Epoch [46/50], Test Losses: mse: 49.6594, mae: 4.7459, huber: 4.2918, swd: 10.3417, ept: 222.1689
      Epoch 46 composite train-obj: 4.177712
            No improvement (4.4062), counter 1/5
    Epoch [47/50], Train Losses: mse: 46.8333, mae: 4.6089, huber: 4.1553, swd: 10.3422, ept: 237.8773
    Epoch [47/50], Val Losses: mse: 51.3187, mae: 4.9101, huber: 4.4545, swd: 11.0508, ept: 208.9075
    Epoch [47/50], Test Losses: mse: 50.2450, mae: 4.8035, huber: 4.3488, swd: 10.7205, ept: 215.2268
      Epoch 47 composite train-obj: 4.155318
            No improvement (4.4545), counter 2/5
    Epoch [48/50], Train Losses: mse: 46.8896, mae: 4.6197, huber: 4.1657, swd: 10.3678, ept: 236.9405
    Epoch [48/50], Val Losses: mse: 50.5761, mae: 4.8559, huber: 4.3998, swd: 10.1977, ept: 217.3812
    Epoch [48/50], Test Losses: mse: 49.4262, mae: 4.7457, huber: 4.2906, swd: 9.9255, ept: 222.2900
      Epoch 48 composite train-obj: 4.165673
            Val objective improved 4.4060 → 4.3998, saving checkpoint.
    Epoch [49/50], Train Losses: mse: 46.7384, mae: 4.6019, huber: 4.1483, swd: 10.2318, ept: 238.2249
    Epoch [49/50], Val Losses: mse: 51.0999, mae: 4.8805, huber: 4.4245, swd: 10.1197, ept: 215.6538
    Epoch [49/50], Test Losses: mse: 49.8921, mae: 4.7716, huber: 4.3167, swd: 9.7103, ept: 220.6512
      Epoch 49 composite train-obj: 4.148335
            No improvement (4.4245), counter 1/5
    Epoch [50/50], Train Losses: mse: 46.7353, mae: 4.6043, huber: 4.1508, swd: 10.2368, ept: 238.3472
    Epoch [50/50], Val Losses: mse: 50.5583, mae: 4.8542, huber: 4.3983, swd: 10.6351, ept: 215.6580
    Epoch [50/50], Test Losses: mse: 49.0659, mae: 4.7291, huber: 4.2747, swd: 10.4980, ept: 221.2194
      Epoch 50 composite train-obj: 4.150805
            Val objective improved 4.3998 → 4.3983, saving checkpoint.
    Epoch [50/50], Test Losses: mse: 49.0659, mae: 4.7291, huber: 4.2747, swd: 10.4980, ept: 221.2194
    Best round's Test MSE: 49.0659, MAE: 4.7291, SWD: 10.4980
    Best round's Validation MSE: 50.5583, MAE: 4.8542, SWD: 10.6351
    Best round's Test verification MSE : 49.0659, MAE: 4.7291, SWD: 10.4980
    Time taken: 428.97 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 80.0244, mae: 6.7259, huber: 6.2452, swd: 14.4108, ept: 69.8525
    Epoch [1/50], Val Losses: mse: 68.1602, mae: 6.2599, huber: 5.7813, swd: 23.3909, ept: 62.7866
    Epoch [1/50], Test Losses: mse: 66.8774, mae: 6.1739, huber: 5.6961, swd: 24.2884, ept: 59.5562
      Epoch 1 composite train-obj: 6.245248
            Val objective improved inf → 5.7813, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.6935, mae: 5.9365, huber: 5.4603, swd: 23.3454, ept: 91.6214
    Epoch [2/50], Val Losses: mse: 61.6704, mae: 5.8606, huber: 5.3863, swd: 25.7228, ept: 89.3161
    Epoch [2/50], Test Losses: mse: 60.9275, mae: 5.7730, huber: 5.2995, swd: 25.2719, ept: 89.4339
      Epoch 2 composite train-obj: 5.460262
            Val objective improved 5.7813 → 5.3863, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 60.0705, mae: 5.7538, huber: 5.2798, swd: 23.2327, ept: 109.4456
    Epoch [3/50], Val Losses: mse: 60.1191, mae: 5.7694, huber: 5.2960, swd: 23.7266, ept: 102.7893
    Epoch [3/50], Test Losses: mse: 59.1970, mae: 5.6606, huber: 5.1881, swd: 22.7959, ept: 103.6212
      Epoch 3 composite train-obj: 5.279840
            Val objective improved 5.3863 → 5.2960, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 58.7457, mae: 5.6439, huber: 5.1715, swd: 22.4836, ept: 120.5124
    Epoch [4/50], Val Losses: mse: 59.6363, mae: 5.7345, huber: 5.2618, swd: 24.2560, ept: 109.2384
    Epoch [4/50], Test Losses: mse: 59.3011, mae: 5.6825, huber: 5.2106, swd: 24.2370, ept: 109.7502
      Epoch 4 composite train-obj: 5.171518
            Val objective improved 5.2960 → 5.2618, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 58.1821, mae: 5.5843, huber: 5.1130, swd: 21.5905, ept: 128.5619
    Epoch [5/50], Val Losses: mse: 59.8784, mae: 5.7655, huber: 5.2917, swd: 21.3985, ept: 112.6404
    Epoch [5/50], Test Losses: mse: 58.7359, mae: 5.6456, huber: 5.1723, swd: 20.1700, ept: 115.2755
      Epoch 5 composite train-obj: 5.112978
            No improvement (5.2917), counter 1/5
    Epoch [6/50], Train Losses: mse: 57.4305, mae: 5.5197, huber: 5.0493, swd: 21.0028, ept: 134.9949
    Epoch [6/50], Val Losses: mse: 58.4756, mae: 5.5962, huber: 5.1260, swd: 21.0624, ept: 124.3667
    Epoch [6/50], Test Losses: mse: 57.1835, mae: 5.4656, huber: 4.9966, swd: 20.1486, ept: 124.5619
      Epoch 6 composite train-obj: 5.049349
            Val objective improved 5.2618 → 5.1260, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 56.4629, mae: 5.4324, huber: 4.9634, swd: 20.1343, ept: 143.4241
    Epoch [7/50], Val Losses: mse: 58.7911, mae: 5.5842, huber: 5.1143, swd: 20.0367, ept: 133.1388
    Epoch [7/50], Test Losses: mse: 57.6829, mae: 5.4638, huber: 4.9953, swd: 18.9860, ept: 134.1018
      Epoch 7 composite train-obj: 4.963377
            Val objective improved 5.1260 → 5.1143, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 55.9287, mae: 5.3821, huber: 4.9139, swd: 19.5378, ept: 149.0224
    Epoch [8/50], Val Losses: mse: 57.8268, mae: 5.5246, huber: 5.0551, swd: 20.2714, ept: 136.3097
    Epoch [8/50], Test Losses: mse: 57.1510, mae: 5.4413, huber: 4.9733, swd: 19.8590, ept: 136.5352
      Epoch 8 composite train-obj: 4.913921
            Val objective improved 5.1143 → 5.0551, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 55.2789, mae: 5.3284, huber: 4.8610, swd: 18.9664, ept: 154.9295
    Epoch [9/50], Val Losses: mse: 57.2700, mae: 5.4598, huber: 4.9921, swd: 19.1005, ept: 144.0388
    Epoch [9/50], Test Losses: mse: 55.9589, mae: 5.3269, huber: 4.8607, swd: 18.2566, ept: 144.6520
      Epoch 9 composite train-obj: 4.861016
            Val objective improved 5.0551 → 4.9921, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 54.7544, mae: 5.2785, huber: 4.8121, swd: 18.3275, ept: 161.0179
    Epoch [10/50], Val Losses: mse: 57.0733, mae: 5.4678, huber: 5.0000, swd: 19.2151, ept: 145.3008
    Epoch [10/50], Test Losses: mse: 55.7075, mae: 5.3337, huber: 4.8674, swd: 18.5846, ept: 148.1738
      Epoch 10 composite train-obj: 4.812071
            No improvement (5.0000), counter 1/5
    Epoch [11/50], Train Losses: mse: 54.3107, mae: 5.2418, huber: 4.7759, swd: 17.9102, ept: 165.3557
    Epoch [11/50], Val Losses: mse: 56.6912, mae: 5.4130, huber: 4.9469, swd: 18.2376, ept: 148.7265
    Epoch [11/50], Test Losses: mse: 55.5318, mae: 5.2930, huber: 4.8280, swd: 17.6450, ept: 151.3930
      Epoch 11 composite train-obj: 4.775938
            Val objective improved 4.9921 → 4.9469, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 53.8882, mae: 5.2042, huber: 4.7390, swd: 17.4413, ept: 169.8943
    Epoch [12/50], Val Losses: mse: 56.0434, mae: 5.3677, huber: 4.9013, swd: 18.1134, ept: 154.3524
    Epoch [12/50], Test Losses: mse: 55.0242, mae: 5.2490, huber: 4.7844, swd: 17.4570, ept: 156.6457
      Epoch 12 composite train-obj: 4.739019
            Val objective improved 4.9469 → 4.9013, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 53.5191, mae: 5.1747, huber: 4.7099, swd: 17.1167, ept: 172.9928
    Epoch [13/50], Val Losses: mse: 57.0071, mae: 5.3957, huber: 4.9293, swd: 16.5282, ept: 155.8071
    Epoch [13/50], Test Losses: mse: 55.7402, mae: 5.2598, huber: 4.7950, swd: 15.3777, ept: 158.1403
      Epoch 13 composite train-obj: 4.709894
            No improvement (4.9293), counter 1/5
    Epoch [14/50], Train Losses: mse: 53.3040, mae: 5.1557, huber: 4.6912, swd: 16.7840, ept: 175.3194
    Epoch [14/50], Val Losses: mse: 57.6011, mae: 5.4314, huber: 4.9646, swd: 16.1753, ept: 157.5810
    Epoch [14/50], Test Losses: mse: 56.1326, mae: 5.2919, huber: 4.8265, swd: 15.2903, ept: 159.8931
      Epoch 14 composite train-obj: 4.691215
            No improvement (4.9646), counter 2/5
    Epoch [15/50], Train Losses: mse: 53.0822, mae: 5.1354, huber: 4.6713, swd: 16.4665, ept: 178.9530
    Epoch [15/50], Val Losses: mse: 57.6033, mae: 5.4032, huber: 4.9370, swd: 15.1164, ept: 160.5215
    Epoch [15/50], Test Losses: mse: 55.7356, mae: 5.2373, huber: 4.7726, swd: 14.1387, ept: 162.2359
      Epoch 15 composite train-obj: 4.671287
            No improvement (4.9370), counter 3/5
    Epoch [16/50], Train Losses: mse: 52.9570, mae: 5.1220, huber: 4.6581, swd: 16.2416, ept: 180.5022
    Epoch [16/50], Val Losses: mse: 56.3711, mae: 5.3299, huber: 4.8640, swd: 15.4634, ept: 163.8181
    Epoch [16/50], Test Losses: mse: 55.0610, mae: 5.1996, huber: 4.7354, swd: 14.7157, ept: 166.5343
      Epoch 16 composite train-obj: 4.658079
            Val objective improved 4.9013 → 4.8640, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 52.3799, mae: 5.0709, huber: 4.6080, swd: 15.8008, ept: 185.1041
    Epoch [17/50], Val Losses: mse: 56.1731, mae: 5.3157, huber: 4.8504, swd: 15.5165, ept: 165.7681
    Epoch [17/50], Test Losses: mse: 54.6012, mae: 5.1678, huber: 4.7043, swd: 14.6670, ept: 169.5831
      Epoch 17 composite train-obj: 4.607997
            Val objective improved 4.8640 → 4.8504, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 52.2084, mae: 5.0563, huber: 4.5935, swd: 15.5303, ept: 186.9748
    Epoch [18/50], Val Losses: mse: 55.4846, mae: 5.2638, huber: 4.8001, swd: 15.2830, ept: 170.1855
    Epoch [18/50], Test Losses: mse: 54.1913, mae: 5.1492, huber: 4.6868, swd: 14.7600, ept: 173.8861
      Epoch 18 composite train-obj: 4.593538
            Val objective improved 4.8504 → 4.8001, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 52.0973, mae: 5.0450, huber: 4.5826, swd: 15.3234, ept: 188.9999
    Epoch [19/50], Val Losses: mse: 55.3142, mae: 5.2450, huber: 4.7822, swd: 15.5583, ept: 170.6852
    Epoch [19/50], Test Losses: mse: 53.7681, mae: 5.1071, huber: 4.6459, swd: 14.8180, ept: 173.5965
      Epoch 19 composite train-obj: 4.582598
            Val objective improved 4.8001 → 4.7822, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 51.7783, mae: 5.0162, huber: 4.5544, swd: 15.0207, ept: 191.3068
    Epoch [20/50], Val Losses: mse: 54.6483, mae: 5.2146, huber: 4.7522, swd: 15.9395, ept: 172.5801
    Epoch [20/50], Test Losses: mse: 53.5117, mae: 5.1005, huber: 4.6398, swd: 15.3700, ept: 177.2725
      Epoch 20 composite train-obj: 4.554372
            Val objective improved 4.7822 → 4.7522, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 51.5380, mae: 4.9984, huber: 4.5368, swd: 14.8795, ept: 193.6838
    Epoch [21/50], Val Losses: mse: 54.5838, mae: 5.2239, huber: 4.7600, swd: 15.7607, ept: 171.3301
    Epoch [21/50], Test Losses: mse: 53.6024, mae: 5.1254, huber: 4.6632, swd: 15.5661, ept: 177.4001
      Epoch 21 composite train-obj: 4.536758
            No improvement (4.7600), counter 1/5
    Epoch [22/50], Train Losses: mse: 51.6398, mae: 5.0005, huber: 4.5388, swd: 14.6022, ept: 194.6296
    Epoch [22/50], Val Losses: mse: 55.8658, mae: 5.3133, huber: 4.8467, swd: 14.0126, ept: 176.6254
    Epoch [22/50], Test Losses: mse: 53.9162, mae: 5.1586, huber: 4.6930, swd: 13.0956, ept: 181.6435
      Epoch 22 composite train-obj: 4.538752
            No improvement (4.8467), counter 2/5
    Epoch [23/50], Train Losses: mse: 51.4622, mae: 4.9933, huber: 4.5315, swd: 14.6327, ept: 195.2680
    Epoch [23/50], Val Losses: mse: 54.7781, mae: 5.2392, huber: 4.7738, swd: 14.9855, ept: 175.6562
    Epoch [23/50], Test Losses: mse: 53.4238, mae: 5.1123, huber: 4.6485, swd: 14.3538, ept: 180.3935
      Epoch 23 composite train-obj: 4.531545
            No improvement (4.7738), counter 3/5
    Epoch [24/50], Train Losses: mse: 51.0833, mae: 4.9564, huber: 4.4954, swd: 14.3128, ept: 198.5899
    Epoch [24/50], Val Losses: mse: 55.3371, mae: 5.2366, huber: 4.7721, swd: 14.2216, ept: 175.5945
    Epoch [24/50], Test Losses: mse: 53.7615, mae: 5.0915, huber: 4.6288, swd: 13.5156, ept: 181.3628
      Epoch 24 composite train-obj: 4.495403
            No improvement (4.7721), counter 4/5
    Epoch [25/50], Train Losses: mse: 50.8474, mae: 4.9353, huber: 4.4747, swd: 14.1663, ept: 200.4845
    Epoch [25/50], Val Losses: mse: 54.6254, mae: 5.1887, huber: 4.7262, swd: 14.2529, ept: 179.1247
    Epoch [25/50], Test Losses: mse: 53.4497, mae: 5.0687, huber: 4.6078, swd: 13.7677, ept: 185.6225
      Epoch 25 composite train-obj: 4.474699
            Val objective improved 4.7522 → 4.7262, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 50.7421, mae: 4.9250, huber: 4.4647, swd: 14.0551, ept: 201.4949
    Epoch [26/50], Val Losses: mse: 54.5917, mae: 5.1924, huber: 4.7294, swd: 13.9236, ept: 181.4271
    Epoch [26/50], Test Losses: mse: 53.1964, mae: 5.0633, huber: 4.6017, swd: 13.2159, ept: 186.5627
      Epoch 26 composite train-obj: 4.464687
            No improvement (4.7294), counter 1/5
    Epoch [27/50], Train Losses: mse: 50.5971, mae: 4.9138, huber: 4.4536, swd: 13.9009, ept: 203.1217
    Epoch [27/50], Val Losses: mse: 56.1070, mae: 5.2917, huber: 4.8244, swd: 12.2629, ept: 180.9833
    Epoch [27/50], Test Losses: mse: 54.2641, mae: 5.1391, huber: 4.6736, swd: 11.8501, ept: 186.4213
      Epoch 27 composite train-obj: 4.453570
            No improvement (4.8244), counter 2/5
    Epoch [28/50], Train Losses: mse: 50.5820, mae: 4.9155, huber: 4.4551, swd: 13.8696, ept: 203.6540
    Epoch [28/50], Val Losses: mse: 55.3692, mae: 5.2366, huber: 4.7727, swd: 13.1371, ept: 173.9140
    Epoch [28/50], Test Losses: mse: 54.4000, mae: 5.1301, huber: 4.6675, swd: 12.5847, ept: 181.0114
      Epoch 28 composite train-obj: 4.455109
            No improvement (4.7727), counter 3/5
    Epoch [29/50], Train Losses: mse: 50.6572, mae: 4.9166, huber: 4.4564, swd: 13.8248, ept: 203.0413
    Epoch [29/50], Val Losses: mse: 56.6182, mae: 5.3240, huber: 4.8565, swd: 12.1944, ept: 180.6476
    Epoch [29/50], Test Losses: mse: 54.5627, mae: 5.1477, huber: 4.6821, swd: 11.3110, ept: 187.1389
      Epoch 29 composite train-obj: 4.456426
            No improvement (4.8565), counter 4/5
    Epoch [30/50], Train Losses: mse: 50.6806, mae: 4.9265, huber: 4.4657, swd: 13.8687, ept: 201.7615
    Epoch [30/50], Val Losses: mse: 55.5763, mae: 5.2260, huber: 4.7635, swd: 13.4881, ept: 178.3107
    Epoch [30/50], Test Losses: mse: 54.1113, mae: 5.0960, huber: 4.6349, swd: 12.9232, ept: 183.6653
      Epoch 30 composite train-obj: 4.465655
    Epoch [30/50], Test Losses: mse: 53.4497, mae: 5.0687, huber: 4.6078, swd: 13.7677, ept: 185.6225
    Best round's Test MSE: 53.4497, MAE: 5.0687, SWD: 13.7677
    Best round's Validation MSE: 54.6254, MAE: 5.1887, SWD: 14.2529
    Best round's Test verification MSE : 53.4497, MAE: 5.0687, SWD: 13.7677
    Time taken: 256.78 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq96_pred720_20250513_0520)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 51.1233 ± 1.7998
      mae: 4.8833 ± 0.1404
      huber: 4.4262 ± 0.1376
      swd: 11.9739 ± 1.3537
      ept: 204.3494 ± 14.5916
      count: 35.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 52.3591 ± 1.6927
      mae: 5.0013 ± 0.1395
      huber: 4.5429 ± 0.1366
      swd: 12.3262 ± 1.4863
      ept: 198.5163 ± 14.9993
      count: 35.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 1115.96 seconds
    
    Experiment complete: TimeMixer_lorenz_seq96_pred720_20250513_0520
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### PatchTST

#### 96-96
##### huber


```python
utils.reload_modules([utils])
cfg_patch_tst = train_config.FlatPatchTSTConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 283
    Validation Batches: 40
    Test Batches: 80
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 41.0689, mae: 4.3413, huber: 3.8805, swd: 8.9036, ept: 35.8416
    Epoch [1/50], Val Losses: mse: 21.5450, mae: 2.8839, huber: 2.4451, swd: 7.4485, ept: 77.6121
    Epoch [1/50], Test Losses: mse: 21.1082, mae: 2.8504, huber: 2.4123, swd: 7.4670, ept: 77.5510
      Epoch 1 composite train-obj: 3.880490
            Val objective improved inf → 2.4451, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 26.0922, mae: 3.1898, huber: 2.7472, swd: 5.6555, ept: 39.4531
    Epoch [2/50], Val Losses: mse: 13.0601, mae: 2.0261, huber: 1.6116, swd: 4.8593, ept: 85.1368
    Epoch [2/50], Test Losses: mse: 12.3536, mae: 1.9562, huber: 1.5444, swd: 4.4822, ept: 86.0870
      Epoch 2 composite train-obj: 2.747235
            Val objective improved 2.4451 → 1.6116, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.4877, mae: 2.8346, huber: 2.4018, swd: 4.5974, ept: 40.0876
    Epoch [3/50], Val Losses: mse: 10.7972, mae: 1.8047, huber: 1.3989, swd: 3.5905, ept: 87.7327
    Epoch [3/50], Test Losses: mse: 9.9814, mae: 1.7519, huber: 1.3481, swd: 3.3868, ept: 88.4285
      Epoch 3 composite train-obj: 2.401838
            Val objective improved 1.6116 → 1.3989, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 20.7472, mae: 2.6553, huber: 2.2288, swd: 4.0389, ept: 40.2725
    Epoch [4/50], Val Losses: mse: 9.1468, mae: 1.6699, huber: 1.2660, swd: 2.9813, ept: 89.4173
    Epoch [4/50], Test Losses: mse: 8.0347, mae: 1.5805, huber: 1.1786, swd: 2.6417, ept: 90.2635
      Epoch 4 composite train-obj: 2.228849
            Val objective improved 1.3989 → 1.2660, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 19.4729, mae: 2.5229, huber: 2.1014, swd: 3.6633, ept: 40.5945
    Epoch [5/50], Val Losses: mse: 7.7998, mae: 1.4801, huber: 1.0906, swd: 2.3665, ept: 90.1718
    Epoch [5/50], Test Losses: mse: 7.2070, mae: 1.4330, huber: 1.0461, swd: 2.3465, ept: 90.8212
      Epoch 5 composite train-obj: 2.101426
            Val objective improved 1.2660 → 1.0906, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 18.4058, mae: 2.4099, huber: 1.9933, swd: 3.2971, ept: 40.5737
    Epoch [6/50], Val Losses: mse: 7.9366, mae: 1.4854, huber: 1.0911, swd: 2.5799, ept: 89.9443
    Epoch [6/50], Test Losses: mse: 7.1539, mae: 1.4274, huber: 1.0359, swd: 2.3563, ept: 90.9842
      Epoch 6 composite train-obj: 1.993283
            No improvement (1.0911), counter 1/5
    Epoch [7/50], Train Losses: mse: 17.8273, mae: 2.3435, huber: 1.9302, swd: 3.0918, ept: 40.6487
    Epoch [7/50], Val Losses: mse: 6.3711, mae: 1.3209, huber: 0.9391, swd: 1.8742, ept: 91.4210
    Epoch [7/50], Test Losses: mse: 5.8609, mae: 1.2678, huber: 0.8902, swd: 1.6740, ept: 91.8706
      Epoch 7 composite train-obj: 1.930152
            Val objective improved 1.0906 → 0.9391, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 17.2658, mae: 2.2834, huber: 1.8724, swd: 2.8995, ept: 40.8218
    Epoch [8/50], Val Losses: mse: 5.8852, mae: 1.2738, huber: 0.8962, swd: 1.9818, ept: 91.9091
    Epoch [8/50], Test Losses: mse: 5.3323, mae: 1.2349, huber: 0.8594, swd: 1.8832, ept: 92.3166
      Epoch 8 composite train-obj: 1.872385
            Val objective improved 0.9391 → 0.8962, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 16.7992, mae: 2.2337, huber: 1.8255, swd: 2.7659, ept: 40.8322
    Epoch [9/50], Val Losses: mse: 5.9770, mae: 1.2479, huber: 0.8725, swd: 1.7356, ept: 91.9532
    Epoch [9/50], Test Losses: mse: 5.2091, mae: 1.1876, huber: 0.8152, swd: 1.4715, ept: 92.5711
      Epoch 9 composite train-obj: 1.825467
            Val objective improved 0.8962 → 0.8725, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 16.5350, mae: 2.1996, huber: 1.7934, swd: 2.6425, ept: 40.8522
    Epoch [10/50], Val Losses: mse: 5.1781, mae: 1.1245, huber: 0.7643, swd: 1.4113, ept: 92.3114
    Epoch [10/50], Test Losses: mse: 4.5651, mae: 1.0886, huber: 0.7299, swd: 1.3022, ept: 92.9278
      Epoch 10 composite train-obj: 1.793392
            Val objective improved 0.8725 → 0.7643, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 16.3330, mae: 2.1675, huber: 1.7635, swd: 2.5925, ept: 40.9071
    Epoch [11/50], Val Losses: mse: 6.1901, mae: 1.2352, huber: 0.8652, swd: 1.8613, ept: 91.7580
    Epoch [11/50], Test Losses: mse: 5.4687, mae: 1.1830, huber: 0.8152, swd: 1.6051, ept: 92.5205
      Epoch 11 composite train-obj: 1.763476
            No improvement (0.8652), counter 1/5
    Epoch [12/50], Train Losses: mse: 16.1031, mae: 2.1397, huber: 1.7374, swd: 2.4989, ept: 40.8158
    Epoch [12/50], Val Losses: mse: 4.8996, mae: 1.1166, huber: 0.7563, swd: 1.2983, ept: 92.7193
    Epoch [12/50], Test Losses: mse: 4.3591, mae: 1.0792, huber: 0.7204, swd: 1.2426, ept: 93.1856
      Epoch 12 composite train-obj: 1.737403
            Val objective improved 0.7643 → 0.7563, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 15.8365, mae: 2.1123, huber: 1.7114, swd: 2.4202, ept: 40.9251
    Epoch [13/50], Val Losses: mse: 5.4928, mae: 1.2503, huber: 0.8664, swd: 1.5382, ept: 92.3893
    Epoch [13/50], Test Losses: mse: 4.9457, mae: 1.2018, huber: 0.8222, swd: 1.3566, ept: 92.7591
      Epoch 13 composite train-obj: 1.711398
            No improvement (0.8664), counter 1/5
    Epoch [14/50], Train Losses: mse: 15.6569, mae: 2.0900, huber: 1.6904, swd: 2.3806, ept: 40.9599
    Epoch [14/50], Val Losses: mse: 4.6913, mae: 1.1068, huber: 0.7377, swd: 1.3608, ept: 92.9865
    Epoch [14/50], Test Losses: mse: 4.1016, mae: 1.0717, huber: 0.7034, swd: 1.1449, ept: 93.4762
      Epoch 14 composite train-obj: 1.690419
            Val objective improved 0.7563 → 0.7377, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 15.4181, mae: 2.0590, huber: 1.6618, swd: 2.2958, ept: 40.9779
    Epoch [15/50], Val Losses: mse: 5.0049, mae: 1.1310, huber: 0.7632, swd: 1.6058, ept: 92.6786
    Epoch [15/50], Test Losses: mse: 4.3030, mae: 1.0794, huber: 0.7147, swd: 1.2012, ept: 93.2584
      Epoch 15 composite train-obj: 1.661826
            No improvement (0.7632), counter 1/5
    Epoch [16/50], Train Losses: mse: 15.3770, mae: 2.0553, huber: 1.6584, swd: 2.2791, ept: 41.0626
    Epoch [16/50], Val Losses: mse: 4.4491, mae: 1.0743, huber: 0.7096, swd: 1.2065, ept: 92.9880
    Epoch [16/50], Test Losses: mse: 3.8027, mae: 1.0274, huber: 0.6657, swd: 1.0044, ept: 93.6062
      Epoch 16 composite train-obj: 1.658354
            Val objective improved 0.7377 → 0.7096, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.1734, mae: 2.0301, huber: 1.6348, swd: 2.2091, ept: 41.0679
    Epoch [17/50], Val Losses: mse: 4.4967, mae: 1.0275, huber: 0.6746, swd: 1.5221, ept: 93.0297
    Epoch [17/50], Test Losses: mse: 3.9500, mae: 0.9718, huber: 0.6246, swd: 1.2163, ept: 93.4981
      Epoch 17 composite train-obj: 1.634803
            Val objective improved 0.7096 → 0.6746, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.1675, mae: 2.0218, huber: 1.6276, swd: 2.2034, ept: 40.9624
    Epoch [18/50], Val Losses: mse: 4.1914, mae: 1.0193, huber: 0.6645, swd: 1.2855, ept: 92.9727
    Epoch [18/50], Test Losses: mse: 3.6633, mae: 0.9857, huber: 0.6328, swd: 1.0411, ept: 93.6483
      Epoch 18 composite train-obj: 1.627620
            Val objective improved 0.6746 → 0.6645, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 15.0309, mae: 2.0098, huber: 1.6161, swd: 2.1785, ept: 41.0202
    Epoch [19/50], Val Losses: mse: 4.6984, mae: 1.1693, huber: 0.7888, swd: 1.4086, ept: 93.0270
    Epoch [19/50], Test Losses: mse: 4.1967, mae: 1.1404, huber: 0.7599, swd: 1.2877, ept: 93.4419
      Epoch 19 composite train-obj: 1.616096
            No improvement (0.7888), counter 1/5
    Epoch [20/50], Train Losses: mse: 14.9366, mae: 1.9933, huber: 1.6012, swd: 2.1276, ept: 40.9764
    Epoch [20/50], Val Losses: mse: 4.4192, mae: 1.0722, huber: 0.7136, swd: 1.0927, ept: 92.9089
    Epoch [20/50], Test Losses: mse: 3.9952, mae: 1.0445, huber: 0.6872, swd: 1.0007, ept: 93.3777
      Epoch 20 composite train-obj: 1.601239
            No improvement (0.7136), counter 2/5
    Epoch [21/50], Train Losses: mse: 14.7920, mae: 1.9834, huber: 1.5917, swd: 2.0853, ept: 41.1057
    Epoch [21/50], Val Losses: mse: 4.3538, mae: 1.0882, huber: 0.7182, swd: 1.0317, ept: 93.0941
    Epoch [21/50], Test Losses: mse: 3.7315, mae: 1.0427, huber: 0.6744, swd: 0.8417, ept: 93.5591
      Epoch 21 composite train-obj: 1.591653
            No improvement (0.7182), counter 3/5
    Epoch [22/50], Train Losses: mse: 14.7177, mae: 1.9690, huber: 1.5785, swd: 2.0699, ept: 41.1431
    Epoch [22/50], Val Losses: mse: 4.4775, mae: 1.0614, huber: 0.6980, swd: 1.3426, ept: 93.4592
    Epoch [22/50], Test Losses: mse: 3.8634, mae: 0.9967, huber: 0.6374, swd: 1.1181, ept: 93.8590
      Epoch 22 composite train-obj: 1.578472
            No improvement (0.6980), counter 4/5
    Epoch [23/50], Train Losses: mse: 14.5620, mae: 1.9457, huber: 1.5575, swd: 2.0209, ept: 41.1067
    Epoch [23/50], Val Losses: mse: 3.9049, mae: 0.9907, huber: 0.6359, swd: 0.9448, ept: 93.4801
    Epoch [23/50], Test Losses: mse: 3.3095, mae: 0.9377, huber: 0.5866, swd: 0.8745, ept: 93.9296
      Epoch 23 composite train-obj: 1.557540
            Val objective improved 0.6645 → 0.6359, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 14.5882, mae: 1.9449, huber: 1.5570, swd: 2.0030, ept: 40.9879
    Epoch [24/50], Val Losses: mse: 3.9539, mae: 0.9484, huber: 0.6099, swd: 1.3127, ept: 93.1060
    Epoch [24/50], Test Losses: mse: 3.3360, mae: 0.9016, huber: 0.5650, swd: 1.0768, ept: 93.6549
      Epoch 24 composite train-obj: 1.556961
            Val objective improved 0.6359 → 0.6099, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 14.4830, mae: 1.9328, huber: 1.5457, swd: 1.9925, ept: 41.0659
    Epoch [25/50], Val Losses: mse: 3.9565, mae: 0.9948, huber: 0.6394, swd: 1.1446, ept: 93.4982
    Epoch [25/50], Test Losses: mse: 3.3350, mae: 0.9595, huber: 0.6050, swd: 0.8852, ept: 93.9634
      Epoch 25 composite train-obj: 1.545675
            No improvement (0.6394), counter 1/5
    Epoch [26/50], Train Losses: mse: 14.4692, mae: 1.9303, huber: 1.5435, swd: 1.9644, ept: 41.0753
    Epoch [26/50], Val Losses: mse: 4.1980, mae: 1.1353, huber: 0.7510, swd: 1.1401, ept: 93.5470
    Epoch [26/50], Test Losses: mse: 3.6643, mae: 1.1064, huber: 0.7212, swd: 1.1084, ept: 94.0057
      Epoch 26 composite train-obj: 1.543456
            No improvement (0.7510), counter 2/5
    Epoch [27/50], Train Losses: mse: 14.3364, mae: 1.9145, huber: 1.5287, swd: 1.9545, ept: 41.0502
    Epoch [27/50], Val Losses: mse: 3.7470, mae: 1.0102, huber: 0.6500, swd: 1.2063, ept: 93.6578
    Epoch [27/50], Test Losses: mse: 3.2551, mae: 0.9773, huber: 0.6177, swd: 1.0459, ept: 94.1382
      Epoch 27 composite train-obj: 1.528732
            No improvement (0.6500), counter 3/5
    Epoch [28/50], Train Losses: mse: 14.3928, mae: 1.9177, huber: 1.5324, swd: 1.9567, ept: 41.0404
    Epoch [28/50], Val Losses: mse: 3.4486, mae: 0.8611, huber: 0.5329, swd: 0.9836, ept: 93.7267
    Epoch [28/50], Test Losses: mse: 2.8725, mae: 0.8167, huber: 0.4900, swd: 0.7277, ept: 94.1203
      Epoch 28 composite train-obj: 1.532416
            Val objective improved 0.6099 → 0.5329, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 14.2037, mae: 1.8980, huber: 1.5139, swd: 1.9144, ept: 41.1091
    Epoch [29/50], Val Losses: mse: 3.5234, mae: 0.9014, huber: 0.5631, swd: 0.9615, ept: 93.6753
    Epoch [29/50], Test Losses: mse: 2.9923, mae: 0.8551, huber: 0.5216, swd: 0.7716, ept: 94.0377
      Epoch 29 composite train-obj: 1.513881
            No improvement (0.5631), counter 1/5
    Epoch [30/50], Train Losses: mse: 14.1867, mae: 1.8902, huber: 1.5068, swd: 1.8976, ept: 41.1341
    Epoch [30/50], Val Losses: mse: 3.2929, mae: 0.8968, huber: 0.5538, swd: 0.9522, ept: 93.7018
    Epoch [30/50], Test Losses: mse: 2.7198, mae: 0.8529, huber: 0.5112, swd: 0.7230, ept: 94.2786
      Epoch 30 composite train-obj: 1.506844
            No improvement (0.5538), counter 2/5
    Epoch [31/50], Train Losses: mse: 14.1600, mae: 1.8882, huber: 1.5049, swd: 1.8881, ept: 40.9817
    Epoch [31/50], Val Losses: mse: 3.5542, mae: 0.9506, huber: 0.5952, swd: 1.2570, ept: 93.9744
    Epoch [31/50], Test Losses: mse: 3.0672, mae: 0.9062, huber: 0.5516, swd: 1.1007, ept: 94.2858
      Epoch 31 composite train-obj: 1.504945
            No improvement (0.5952), counter 3/5
    Epoch [32/50], Train Losses: mse: 14.0828, mae: 1.8780, huber: 1.4957, swd: 1.8807, ept: 41.1049
    Epoch [32/50], Val Losses: mse: 3.8493, mae: 0.9865, huber: 0.6335, swd: 1.1065, ept: 93.6554
    Epoch [32/50], Test Losses: mse: 3.5364, mae: 0.9543, huber: 0.6030, swd: 1.0134, ept: 94.0890
      Epoch 32 composite train-obj: 1.495688
            No improvement (0.6335), counter 4/5
    Epoch [33/50], Train Losses: mse: 14.0558, mae: 1.8729, huber: 1.4914, swd: 1.8611, ept: 41.1032
    Epoch [33/50], Val Losses: mse: 3.4092, mae: 0.9658, huber: 0.6062, swd: 1.0728, ept: 93.9627
    Epoch [33/50], Test Losses: mse: 2.9723, mae: 0.9274, huber: 0.5701, swd: 0.9579, ept: 94.3728
      Epoch 33 composite train-obj: 1.491442
    Epoch [33/50], Test Losses: mse: 2.8725, mae: 0.8167, huber: 0.4900, swd: 0.7277, ept: 94.1203
    Best round's Test MSE: 2.8725, MAE: 0.8167, SWD: 0.7277
    Best round's Validation MSE: 3.4486, MAE: 0.8611, SWD: 0.9836
    Best round's Test verification MSE : 2.8725, MAE: 0.8167, SWD: 0.7277
    Time taken: 112.16 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 41.1127, mae: 4.3759, huber: 3.9140, swd: 8.6988, ept: 35.7739
    Epoch [1/50], Val Losses: mse: 20.0970, mae: 2.7209, huber: 2.2848, swd: 6.3181, ept: 78.9025
    Epoch [1/50], Test Losses: mse: 19.5375, mae: 2.6911, huber: 2.2558, swd: 6.2029, ept: 79.4522
      Epoch 1 composite train-obj: 3.913967
            Val objective improved inf → 2.2848, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 26.3701, mae: 3.2312, huber: 2.7873, swd: 5.3086, ept: 39.3809
    Epoch [2/50], Val Losses: mse: 14.7810, mae: 2.2339, huber: 1.8118, swd: 4.6073, ept: 84.5434
    Epoch [2/50], Test Losses: mse: 13.9656, mae: 2.1667, huber: 1.7465, swd: 4.5281, ept: 85.1271
      Epoch 2 composite train-obj: 2.787278
            Val objective improved 2.2848 → 1.8118, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.7385, mae: 2.8637, huber: 2.4298, swd: 4.3834, ept: 40.0709
    Epoch [3/50], Val Losses: mse: 10.8497, mae: 1.8329, huber: 1.4213, swd: 3.4776, ept: 88.1531
    Epoch [3/50], Test Losses: mse: 10.0647, mae: 1.7689, huber: 1.3598, swd: 3.2781, ept: 89.0167
      Epoch 3 composite train-obj: 2.429760
            Val objective improved 1.8118 → 1.4213, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 20.6913, mae: 2.6534, huber: 2.2265, swd: 3.8427, ept: 40.5501
    Epoch [4/50], Val Losses: mse: 9.2341, mae: 1.6201, huber: 1.2231, swd: 2.8763, ept: 89.5331
    Epoch [4/50], Test Losses: mse: 8.5020, mae: 1.5518, huber: 1.1587, swd: 2.6801, ept: 90.2417
      Epoch 4 composite train-obj: 2.226469
            Val objective improved 1.4213 → 1.2231, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 19.5009, mae: 2.5190, huber: 2.0981, swd: 3.5070, ept: 40.5806
    Epoch [5/50], Val Losses: mse: 7.9814, mae: 1.5180, huber: 1.1214, swd: 2.2577, ept: 90.3686
    Epoch [5/50], Test Losses: mse: 7.3506, mae: 1.4654, huber: 1.0708, swd: 2.1605, ept: 90.9682
      Epoch 5 composite train-obj: 2.098066
            Val objective improved 1.2231 → 1.1214, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 18.7465, mae: 2.4356, huber: 2.0182, swd: 3.2627, ept: 40.6228
    Epoch [6/50], Val Losses: mse: 7.6538, mae: 1.4545, huber: 1.0681, swd: 2.3846, ept: 90.5721
    Epoch [6/50], Test Losses: mse: 6.9611, mae: 1.3984, huber: 1.0148, swd: 2.1394, ept: 91.1851
      Epoch 6 composite train-obj: 2.018167
            Val objective improved 1.1214 → 1.0681, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 18.1274, mae: 2.3662, huber: 1.9520, swd: 3.1006, ept: 40.7836
    Epoch [7/50], Val Losses: mse: 7.0017, mae: 1.3778, huber: 0.9958, swd: 1.8832, ept: 91.3400
    Epoch [7/50], Test Losses: mse: 6.3455, mae: 1.3260, huber: 0.9455, swd: 1.7570, ept: 91.9707
      Epoch 7 composite train-obj: 1.951986
            Val objective improved 1.0681 → 0.9958, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 17.6260, mae: 2.3103, huber: 1.8991, swd: 2.9347, ept: 40.7453
    Epoch [8/50], Val Losses: mse: 6.1270, mae: 1.2530, huber: 0.8770, swd: 1.7387, ept: 91.7813
    Epoch [8/50], Test Losses: mse: 5.5639, mae: 1.2170, huber: 0.8410, swd: 1.6771, ept: 92.3630
      Epoch 8 composite train-obj: 1.899107
            Val objective improved 0.9958 → 0.8770, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 17.1859, mae: 2.2567, huber: 1.8483, swd: 2.8209, ept: 40.8170
    Epoch [9/50], Val Losses: mse: 5.7191, mae: 1.2042, huber: 0.8367, swd: 1.7882, ept: 91.9173
    Epoch [9/50], Test Losses: mse: 5.0749, mae: 1.1558, huber: 0.7906, swd: 1.6151, ept: 92.5025
      Epoch 9 composite train-obj: 1.848346
            Val objective improved 0.8770 → 0.8367, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 16.9182, mae: 2.2299, huber: 1.8226, swd: 2.7327, ept: 40.8833
    Epoch [10/50], Val Losses: mse: 6.3750, mae: 1.2837, huber: 0.9050, swd: 1.7256, ept: 91.9720
    Epoch [10/50], Test Losses: mse: 5.9106, mae: 1.2595, huber: 0.8798, swd: 1.7284, ept: 92.4513
      Epoch 10 composite train-obj: 1.822609
            No improvement (0.9050), counter 1/5
    Epoch [11/50], Train Losses: mse: 16.6835, mae: 2.1972, huber: 1.7921, swd: 2.6560, ept: 40.9080
    Epoch [11/50], Val Losses: mse: 6.3011, mae: 1.2681, huber: 0.8828, swd: 1.5650, ept: 92.1270
    Epoch [11/50], Test Losses: mse: 5.6800, mae: 1.2113, huber: 0.8289, swd: 1.5113, ept: 92.7631
      Epoch 11 composite train-obj: 1.792052
            No improvement (0.8828), counter 2/5
    Epoch [12/50], Train Losses: mse: 16.4134, mae: 2.1664, huber: 1.7631, swd: 2.5633, ept: 40.9755
    Epoch [12/50], Val Losses: mse: 5.4615, mae: 1.1624, huber: 0.7939, swd: 1.5803, ept: 92.4107
    Epoch [12/50], Test Losses: mse: 4.8624, mae: 1.1245, huber: 0.7567, swd: 1.4123, ept: 92.9144
      Epoch 12 composite train-obj: 1.763099
            Val objective improved 0.8367 → 0.7939, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 15.9902, mae: 2.1201, huber: 1.7197, swd: 2.4619, ept: 40.8968
    Epoch [13/50], Val Losses: mse: 5.6413, mae: 1.1708, huber: 0.8069, swd: 1.6684, ept: 92.2063
    Epoch [13/50], Test Losses: mse: 5.0657, mae: 1.1513, huber: 0.7858, swd: 1.5094, ept: 92.7972
      Epoch 13 composite train-obj: 1.719730
            No improvement (0.8069), counter 1/5
    Epoch [14/50], Train Losses: mse: 15.9321, mae: 2.1127, huber: 1.7122, swd: 2.4313, ept: 40.9367
    Epoch [14/50], Val Losses: mse: 5.1884, mae: 1.1401, huber: 0.7722, swd: 1.4530, ept: 92.8605
    Epoch [14/50], Test Losses: mse: 4.6060, mae: 1.0986, huber: 0.7310, swd: 1.3195, ept: 93.2116
      Epoch 14 composite train-obj: 1.712158
            Val objective improved 0.7939 → 0.7722, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 15.8090, mae: 2.0944, huber: 1.6955, swd: 2.3856, ept: 40.9724
    Epoch [15/50], Val Losses: mse: 6.6078, mae: 1.2625, huber: 0.8831, swd: 1.4720, ept: 92.4431
    Epoch [15/50], Test Losses: mse: 5.7623, mae: 1.2214, huber: 0.8404, swd: 1.4440, ept: 92.9262
      Epoch 15 composite train-obj: 1.695486
            No improvement (0.8831), counter 1/5
    Epoch [16/50], Train Losses: mse: 15.6216, mae: 2.0681, huber: 1.6710, swd: 2.3133, ept: 40.9144
    Epoch [16/50], Val Losses: mse: 4.6070, mae: 1.1000, huber: 0.7387, swd: 1.2723, ept: 92.9782
    Epoch [16/50], Test Losses: mse: 4.1453, mae: 1.0608, huber: 0.7013, swd: 1.1805, ept: 93.4442
      Epoch 16 composite train-obj: 1.671010
            Val objective improved 0.7722 → 0.7387, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.5116, mae: 2.0549, huber: 1.6588, swd: 2.2728, ept: 41.0995
    Epoch [17/50], Val Losses: mse: 4.4995, mae: 1.0435, huber: 0.6903, swd: 1.2957, ept: 93.1690
    Epoch [17/50], Test Losses: mse: 3.9982, mae: 0.9900, huber: 0.6413, swd: 1.0896, ept: 93.6443
      Epoch 17 composite train-obj: 1.658830
            Val objective improved 0.7387 → 0.6903, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.4034, mae: 2.0366, huber: 1.6421, swd: 2.2591, ept: 40.9756
    Epoch [18/50], Val Losses: mse: 4.1334, mae: 1.0090, huber: 0.6598, swd: 1.1804, ept: 93.3414
    Epoch [18/50], Test Losses: mse: 3.6177, mae: 0.9785, huber: 0.6305, swd: 0.9585, ept: 93.7870
      Epoch 18 composite train-obj: 1.642124
            Val objective improved 0.6903 → 0.6598, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 15.3068, mae: 2.0291, huber: 1.6351, swd: 2.2278, ept: 41.0559
    Epoch [19/50], Val Losses: mse: 4.8213, mae: 1.0435, huber: 0.6912, swd: 1.4499, ept: 93.1536
    Epoch [19/50], Test Losses: mse: 4.1303, mae: 1.0024, huber: 0.6501, swd: 1.2922, ept: 93.6295
      Epoch 19 composite train-obj: 1.635096
            No improvement (0.6912), counter 1/5
    Epoch [20/50], Train Losses: mse: 15.1702, mae: 2.0068, huber: 1.6146, swd: 2.1974, ept: 41.0867
    Epoch [20/50], Val Losses: mse: 4.6245, mae: 1.0517, huber: 0.6942, swd: 1.2608, ept: 93.2589
    Epoch [20/50], Test Losses: mse: 4.0366, mae: 1.0230, huber: 0.6641, swd: 1.0914, ept: 93.5997
      Epoch 20 composite train-obj: 1.614586
            No improvement (0.6942), counter 2/5
    Epoch [21/50], Train Losses: mse: 15.0645, mae: 1.9992, huber: 1.6070, swd: 2.1790, ept: 41.0111
    Epoch [21/50], Val Losses: mse: 4.4412, mae: 1.0886, huber: 0.7193, swd: 1.4535, ept: 93.0917
    Epoch [21/50], Test Losses: mse: 3.9208, mae: 1.0653, huber: 0.6938, swd: 1.2836, ept: 93.5404
      Epoch 21 composite train-obj: 1.607005
            No improvement (0.7193), counter 3/5
    Epoch [22/50], Train Losses: mse: 14.9450, mae: 1.9841, huber: 1.5935, swd: 2.1215, ept: 41.0369
    Epoch [22/50], Val Losses: mse: 4.3464, mae: 1.1411, huber: 0.7618, swd: 1.3134, ept: 93.1702
    Epoch [22/50], Test Losses: mse: 3.9066, mae: 1.1153, huber: 0.7354, swd: 1.2008, ept: 93.7061
      Epoch 22 composite train-obj: 1.593460
            No improvement (0.7618), counter 4/5
    Epoch [23/50], Train Losses: mse: 14.8147, mae: 1.9660, huber: 1.5764, swd: 2.0793, ept: 41.0664
    Epoch [23/50], Val Losses: mse: 4.0075, mae: 0.9875, huber: 0.6362, swd: 1.1190, ept: 93.3529
    Epoch [23/50], Test Losses: mse: 3.6697, mae: 0.9661, huber: 0.6155, swd: 1.0003, ept: 93.7851
      Epoch 23 composite train-obj: 1.576380
            Val objective improved 0.6598 → 0.6362, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 14.7899, mae: 1.9621, huber: 1.5732, swd: 2.0711, ept: 40.9577
    Epoch [24/50], Val Losses: mse: 4.0997, mae: 1.0027, huber: 0.6528, swd: 1.2890, ept: 93.2591
    Epoch [24/50], Test Losses: mse: 3.4917, mae: 0.9600, huber: 0.6093, swd: 1.0625, ept: 93.7780
      Epoch 24 composite train-obj: 1.573153
            No improvement (0.6528), counter 1/5
    Epoch [25/50], Train Losses: mse: 14.7180, mae: 1.9517, huber: 1.5633, swd: 2.0481, ept: 41.0600
    Epoch [25/50], Val Losses: mse: 3.8173, mae: 0.9472, huber: 0.6010, swd: 0.9328, ept: 93.4684
    Epoch [25/50], Test Losses: mse: 3.3352, mae: 0.9082, huber: 0.5643, swd: 0.8054, ept: 93.8303
      Epoch 25 composite train-obj: 1.563256
            Val objective improved 0.6362 → 0.6010, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 14.6276, mae: 1.9395, huber: 1.5528, swd: 2.0191, ept: 41.0095
    Epoch [26/50], Val Losses: mse: 3.8616, mae: 0.9482, huber: 0.5986, swd: 1.0230, ept: 93.7331
    Epoch [26/50], Test Losses: mse: 3.2281, mae: 0.9076, huber: 0.5605, swd: 0.9501, ept: 94.0961
      Epoch 26 composite train-obj: 1.552815
            Val objective improved 0.6010 → 0.5986, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 14.6134, mae: 1.9383, huber: 1.5513, swd: 2.0197, ept: 41.0518
    Epoch [27/50], Val Losses: mse: 4.1591, mae: 1.0030, huber: 0.6517, swd: 1.0603, ept: 93.5725
    Epoch [27/50], Test Losses: mse: 3.3367, mae: 0.9596, huber: 0.6069, swd: 0.8105, ept: 93.9580
      Epoch 27 composite train-obj: 1.551323
            No improvement (0.6517), counter 1/5
    Epoch [28/50], Train Losses: mse: 14.4388, mae: 1.9209, huber: 1.5350, swd: 1.9978, ept: 41.0769
    Epoch [28/50], Val Losses: mse: 4.0649, mae: 0.9914, huber: 0.6369, swd: 0.9757, ept: 93.6678
    Epoch [28/50], Test Losses: mse: 3.3039, mae: 0.9414, huber: 0.5892, swd: 0.6984, ept: 94.0154
      Epoch 28 composite train-obj: 1.535035
            No improvement (0.6369), counter 2/5
    Epoch [29/50], Train Losses: mse: 14.4679, mae: 1.9161, huber: 1.5309, swd: 1.9744, ept: 41.0516
    Epoch [29/50], Val Losses: mse: 4.5139, mae: 1.0571, huber: 0.7064, swd: 1.2066, ept: 93.1324
    Epoch [29/50], Test Losses: mse: 3.7353, mae: 1.0086, huber: 0.6600, swd: 1.0558, ept: 93.6526
      Epoch 29 composite train-obj: 1.530930
            No improvement (0.7064), counter 3/5
    Epoch [30/50], Train Losses: mse: 14.3217, mae: 1.8991, huber: 1.5156, swd: 1.9357, ept: 40.9836
    Epoch [30/50], Val Losses: mse: 3.7178, mae: 0.9171, huber: 0.5846, swd: 1.0333, ept: 93.5196
    Epoch [30/50], Test Losses: mse: 3.3143, mae: 0.8795, huber: 0.5487, swd: 0.7980, ept: 93.9227
      Epoch 30 composite train-obj: 1.515574
            Val objective improved 0.5986 → 0.5846, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 14.2627, mae: 1.8916, huber: 1.5085, swd: 1.9260, ept: 41.0808
    Epoch [31/50], Val Losses: mse: 3.4125, mae: 0.8842, huber: 0.5465, swd: 0.9738, ept: 93.9388
    Epoch [31/50], Test Losses: mse: 2.8685, mae: 0.8510, huber: 0.5140, swd: 0.8383, ept: 94.2111
      Epoch 31 composite train-obj: 1.508466
            Val objective improved 0.5846 → 0.5465, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 14.2689, mae: 1.8874, huber: 1.5050, swd: 1.9203, ept: 41.0793
    Epoch [32/50], Val Losses: mse: 3.7129, mae: 0.9969, huber: 0.6433, swd: 0.9886, ept: 93.8116
    Epoch [32/50], Test Losses: mse: 3.2693, mae: 0.9761, huber: 0.6218, swd: 0.8735, ept: 94.1531
      Epoch 32 composite train-obj: 1.505047
            No improvement (0.6433), counter 1/5
    Epoch [33/50], Train Losses: mse: 14.2183, mae: 1.8847, huber: 1.5023, swd: 1.8939, ept: 41.1287
    Epoch [33/50], Val Losses: mse: 3.6570, mae: 0.9571, huber: 0.6065, swd: 0.9368, ept: 93.6620
    Epoch [33/50], Test Losses: mse: 3.5239, mae: 0.9559, huber: 0.6042, swd: 0.8512, ept: 93.9775
      Epoch 33 composite train-obj: 1.502287
            No improvement (0.6065), counter 2/5
    Epoch [34/50], Train Losses: mse: 14.1758, mae: 1.8817, huber: 1.4996, swd: 1.8936, ept: 41.0329
    Epoch [34/50], Val Losses: mse: 4.0961, mae: 1.0421, huber: 0.6863, swd: 0.9801, ept: 93.3439
    Epoch [34/50], Test Losses: mse: 3.6473, mae: 1.0346, huber: 0.6770, swd: 0.9567, ept: 93.5313
      Epoch 34 composite train-obj: 1.499626
            No improvement (0.6863), counter 3/5
    Epoch [35/50], Train Losses: mse: 14.1351, mae: 1.8731, huber: 1.4921, swd: 1.8893, ept: 41.0122
    Epoch [35/50], Val Losses: mse: 3.1743, mae: 0.8861, huber: 0.5393, swd: 0.8720, ept: 94.0678
    Epoch [35/50], Test Losses: mse: 2.6826, mae: 0.8546, huber: 0.5075, swd: 0.7545, ept: 94.3757
      Epoch 35 composite train-obj: 1.492107
            Val objective improved 0.5465 → 0.5393, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 14.0957, mae: 1.8679, huber: 1.4873, swd: 1.8606, ept: 41.1041
    Epoch [36/50], Val Losses: mse: 3.7618, mae: 0.9057, huber: 0.5720, swd: 1.1373, ept: 93.8546
    Epoch [36/50], Test Losses: mse: 3.2871, mae: 0.8818, huber: 0.5498, swd: 0.9315, ept: 94.1532
      Epoch 36 composite train-obj: 1.487339
            No improvement (0.5720), counter 1/5
    Epoch [37/50], Train Losses: mse: 14.0486, mae: 1.8610, huber: 1.4809, swd: 1.8635, ept: 41.0699
    Epoch [37/50], Val Losses: mse: 3.7951, mae: 0.9712, huber: 0.6159, swd: 1.1149, ept: 93.6408
    Epoch [37/50], Test Losses: mse: 3.3666, mae: 0.9458, huber: 0.5876, swd: 0.9238, ept: 94.0128
      Epoch 37 composite train-obj: 1.480947
            No improvement (0.6159), counter 2/5
    Epoch [38/50], Train Losses: mse: 13.9895, mae: 1.8581, huber: 1.4776, swd: 1.8554, ept: 41.1156
    Epoch [38/50], Val Losses: mse: 3.4631, mae: 0.9379, huber: 0.5848, swd: 0.9585, ept: 93.8926
    Epoch [38/50], Test Losses: mse: 3.1586, mae: 0.9226, huber: 0.5690, swd: 0.8315, ept: 94.1674
      Epoch 38 composite train-obj: 1.477632
            No improvement (0.5848), counter 3/5
    Epoch [39/50], Train Losses: mse: 13.9950, mae: 1.8533, huber: 1.4740, swd: 1.8327, ept: 41.1409
    Epoch [39/50], Val Losses: mse: 4.1774, mae: 1.0092, huber: 0.6628, swd: 0.9798, ept: 93.3929
    Epoch [39/50], Test Losses: mse: 3.7276, mae: 0.9851, huber: 0.6377, swd: 0.7864, ept: 93.7451
      Epoch 39 composite train-obj: 1.473974
            No improvement (0.6628), counter 4/5
    Epoch [40/50], Train Losses: mse: 13.8802, mae: 1.8396, huber: 1.4616, swd: 1.8169, ept: 41.1400
    Epoch [40/50], Val Losses: mse: 3.4312, mae: 0.8999, huber: 0.5697, swd: 1.0637, ept: 93.8018
    Epoch [40/50], Test Losses: mse: 3.0378, mae: 0.8772, huber: 0.5473, swd: 0.9379, ept: 94.1307
      Epoch 40 composite train-obj: 1.461553
    Epoch [40/50], Test Losses: mse: 2.6826, mae: 0.8546, huber: 0.5075, swd: 0.7545, ept: 94.3757
    Best round's Test MSE: 2.6826, MAE: 0.8546, SWD: 0.7545
    Best round's Validation MSE: 3.1743, MAE: 0.8861, SWD: 0.8720
    Best round's Test verification MSE : 2.6826, MAE: 0.8546, SWD: 0.7545
    Time taken: 132.98 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 41.0052, mae: 4.3663, huber: 3.9043, swd: 8.0816, ept: 35.7274
    Epoch [1/50], Val Losses: mse: 19.8054, mae: 2.7205, huber: 2.2844, swd: 6.4042, ept: 79.3037
    Epoch [1/50], Test Losses: mse: 19.1163, mae: 2.6635, huber: 2.2287, swd: 6.1727, ept: 79.9642
      Epoch 1 composite train-obj: 3.904315
            Val objective improved inf → 2.2844, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 26.6088, mae: 3.2353, huber: 2.7914, swd: 5.2374, ept: 39.3099
    Epoch [2/50], Val Losses: mse: 13.1452, mae: 2.0408, huber: 1.6254, swd: 3.8073, ept: 85.8415
    Epoch [2/50], Test Losses: mse: 12.5539, mae: 1.9974, huber: 1.5829, swd: 3.7929, ept: 86.0033
      Epoch 2 composite train-obj: 2.791377
            Val objective improved 2.2844 → 1.6254, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.7263, mae: 2.8521, huber: 2.4188, swd: 4.2222, ept: 40.0397
    Epoch [3/50], Val Losses: mse: 12.0535, mae: 2.0400, huber: 1.6205, swd: 3.5163, ept: 84.7638
    Epoch [3/50], Test Losses: mse: 11.5804, mae: 1.9936, huber: 1.5737, swd: 3.4941, ept: 85.8567
      Epoch 3 composite train-obj: 2.418800
            Val objective improved 1.6254 → 1.6205, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 20.6152, mae: 2.6285, huber: 2.2032, swd: 3.6642, ept: 40.4424
    Epoch [4/50], Val Losses: mse: 10.0782, mae: 1.6873, huber: 1.2838, swd: 2.4796, ept: 89.1995
    Epoch [4/50], Test Losses: mse: 9.2030, mae: 1.6298, huber: 1.2290, swd: 2.3476, ept: 89.7881
      Epoch 4 composite train-obj: 2.203163
            Val objective improved 1.6205 → 1.2838, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 19.4798, mae: 2.5151, huber: 2.0937, swd: 3.2891, ept: 40.5364
    Epoch [5/50], Val Losses: mse: 7.9731, mae: 1.5302, huber: 1.1308, swd: 1.9570, ept: 90.0883
    Epoch [5/50], Test Losses: mse: 7.4575, mae: 1.4844, huber: 1.0877, swd: 1.9286, ept: 90.5193
      Epoch 5 composite train-obj: 2.093747
            Val objective improved 1.2838 → 1.1308, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 18.4878, mae: 2.4105, huber: 1.9937, swd: 3.0180, ept: 40.6634
    Epoch [6/50], Val Losses: mse: 7.6054, mae: 1.3744, huber: 0.9964, swd: 1.9681, ept: 90.8261
    Epoch [6/50], Test Losses: mse: 6.7320, mae: 1.3092, huber: 0.9337, swd: 1.6926, ept: 91.5709
      Epoch 6 composite train-obj: 1.993716
            Val objective improved 1.1308 → 0.9964, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 17.8369, mae: 2.3329, huber: 1.9199, swd: 2.8297, ept: 40.7138
    Epoch [7/50], Val Losses: mse: 6.2431, mae: 1.2719, huber: 0.8976, swd: 1.6308, ept: 91.4531
    Epoch [7/50], Test Losses: mse: 5.5420, mae: 1.2130, huber: 0.8408, swd: 1.3876, ept: 92.1534
      Epoch 7 composite train-obj: 1.919905
            Val objective improved 0.9964 → 0.8976, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 17.2992, mae: 2.2747, huber: 1.8647, swd: 2.6758, ept: 40.8273
    Epoch [8/50], Val Losses: mse: 6.3241, mae: 1.2638, huber: 0.8954, swd: 1.5900, ept: 91.4355
    Epoch [8/50], Test Losses: mse: 5.4665, mae: 1.2089, huber: 0.8428, swd: 1.3830, ept: 92.2181
      Epoch 8 composite train-obj: 1.864697
            Val objective improved 0.8976 → 0.8954, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 16.8878, mae: 2.2242, huber: 1.8171, swd: 2.5350, ept: 40.9118
    Epoch [9/50], Val Losses: mse: 5.3960, mae: 1.1924, huber: 0.8178, swd: 1.3098, ept: 92.0861
    Epoch [9/50], Test Losses: mse: 4.7129, mae: 1.1384, huber: 0.7640, swd: 1.1217, ept: 92.7346
      Epoch 9 composite train-obj: 1.817091
            Val objective improved 0.8954 → 0.8178, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 16.5602, mae: 2.1903, huber: 1.7846, swd: 2.4520, ept: 40.7309
    Epoch [10/50], Val Losses: mse: 5.7860, mae: 1.2231, huber: 0.8472, swd: 1.4886, ept: 91.9598
    Epoch [10/50], Test Losses: mse: 5.1413, mae: 1.1834, huber: 0.8066, swd: 1.1871, ept: 92.7816
      Epoch 10 composite train-obj: 1.784557
            No improvement (0.8472), counter 1/5
    Epoch [11/50], Train Losses: mse: 16.2444, mae: 2.1526, huber: 1.7492, swd: 2.3578, ept: 40.8668
    Epoch [11/50], Val Losses: mse: 5.3656, mae: 1.1565, huber: 0.7861, swd: 1.4392, ept: 92.1434
    Epoch [11/50], Test Losses: mse: 4.6332, mae: 1.1105, huber: 0.7407, swd: 1.1425, ept: 92.6954
      Epoch 11 composite train-obj: 1.749240
            Val objective improved 0.8178 → 0.7861, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 16.0410, mae: 2.1266, huber: 1.7249, swd: 2.3070, ept: 40.8244
    Epoch [12/50], Val Losses: mse: 5.2401, mae: 1.1071, huber: 0.7505, swd: 1.2612, ept: 92.5424
    Epoch [12/50], Test Losses: mse: 4.5408, mae: 1.0616, huber: 0.7073, swd: 1.1802, ept: 93.0115
      Epoch 12 composite train-obj: 1.724899
            Val objective improved 0.7861 → 0.7505, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 15.8569, mae: 2.1030, huber: 1.7029, swd: 2.2434, ept: 40.8446
    Epoch [13/50], Val Losses: mse: 4.6720, mae: 1.0764, huber: 0.7135, swd: 1.3565, ept: 92.9686
    Epoch [13/50], Test Losses: mse: 4.1291, mae: 1.0381, huber: 0.6763, swd: 1.1136, ept: 93.3713
      Epoch 13 composite train-obj: 1.702859
            Val objective improved 0.7505 → 0.7135, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 15.6646, mae: 2.0790, huber: 1.6806, swd: 2.1829, ept: 40.9060
    Epoch [14/50], Val Losses: mse: 5.5023, mae: 1.1598, huber: 0.7843, swd: 1.3660, ept: 92.6740
    Epoch [14/50], Test Losses: mse: 4.7394, mae: 1.1183, huber: 0.7419, swd: 1.2975, ept: 93.1780
      Epoch 14 composite train-obj: 1.680562
            No improvement (0.7843), counter 1/5
    Epoch [15/50], Train Losses: mse: 15.5158, mae: 2.0569, huber: 1.6601, swd: 2.1485, ept: 40.9719
    Epoch [15/50], Val Losses: mse: 4.3271, mae: 1.0209, huber: 0.6643, swd: 1.1965, ept: 93.3563
    Epoch [15/50], Test Losses: mse: 3.8613, mae: 1.0056, huber: 0.6463, swd: 1.1730, ept: 93.6449
      Epoch 15 composite train-obj: 1.660148
            Val objective improved 0.7135 → 0.6643, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 15.3801, mae: 2.0440, huber: 1.6479, swd: 2.1220, ept: 40.9166
    Epoch [16/50], Val Losses: mse: 5.3380, mae: 1.2013, huber: 0.8207, swd: 1.2347, ept: 92.7363
    Epoch [16/50], Test Losses: mse: 4.8987, mae: 1.1772, huber: 0.7938, swd: 1.1706, ept: 93.2090
      Epoch 16 composite train-obj: 1.647911
            No improvement (0.8207), counter 1/5
    Epoch [17/50], Train Losses: mse: 15.1393, mae: 2.0160, huber: 1.6218, swd: 2.0693, ept: 40.9969
    Epoch [17/50], Val Losses: mse: 4.7127, mae: 1.0938, huber: 0.7278, swd: 1.3084, ept: 92.9471
    Epoch [17/50], Test Losses: mse: 4.0837, mae: 1.0540, huber: 0.6885, swd: 1.1881, ept: 93.4383
      Epoch 17 composite train-obj: 1.621828
            No improvement (0.7278), counter 2/5
    Epoch [18/50], Train Losses: mse: 15.0294, mae: 1.9999, huber: 1.6071, swd: 2.0412, ept: 40.9941
    Epoch [18/50], Val Losses: mse: 4.7138, mae: 1.0663, huber: 0.7066, swd: 1.3596, ept: 93.0886
    Epoch [18/50], Test Losses: mse: 4.3658, mae: 1.0436, huber: 0.6844, swd: 1.1852, ept: 93.4883
      Epoch 18 composite train-obj: 1.607069
            No improvement (0.7066), counter 3/5
    Epoch [19/50], Train Losses: mse: 14.9961, mae: 1.9938, huber: 1.6015, swd: 1.9952, ept: 41.0394
    Epoch [19/50], Val Losses: mse: 4.7399, mae: 1.0554, huber: 0.7021, swd: 0.9962, ept: 92.9182
    Epoch [19/50], Test Losses: mse: 4.1079, mae: 1.0143, huber: 0.6615, swd: 0.8934, ept: 93.4243
      Epoch 19 composite train-obj: 1.601531
            No improvement (0.7021), counter 4/5
    Epoch [20/50], Train Losses: mse: 14.9113, mae: 1.9830, huber: 1.5915, swd: 1.9737, ept: 40.9086
    Epoch [20/50], Val Losses: mse: 4.2431, mae: 1.0421, huber: 0.6834, swd: 1.2670, ept: 93.1829
    Epoch [20/50], Test Losses: mse: 3.6207, mae: 1.0073, huber: 0.6486, swd: 1.0452, ept: 93.6122
      Epoch 20 composite train-obj: 1.591479
    Epoch [20/50], Test Losses: mse: 3.8613, mae: 1.0056, huber: 0.6463, swd: 1.1730, ept: 93.6449
    Best round's Test MSE: 3.8613, MAE: 1.0056, SWD: 1.1730
    Best round's Validation MSE: 4.3271, MAE: 1.0209, SWD: 1.1965
    Best round's Test verification MSE : 3.8613, MAE: 1.0056, SWD: 1.1730
    Time taken: 65.66 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq96_pred96_20250514_0111)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 3.1388 ± 0.5167
      mae: 0.8923 ± 0.0816
      huber: 0.5479 ± 0.0699
      swd: 0.8851 ± 0.2039
      ept: 94.0470 ± 0.3028
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 3.6500 ± 0.4917
      mae: 0.9227 ± 0.0702
      huber: 0.5789 ± 0.0605
      swd: 1.0174 ± 0.1346
      ept: 93.7169 ± 0.2906
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 310.93 seconds
    
    Experiment complete: PatchTST_lorenz_seq96_pred96_20250514_0111
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 96-196
##### huber


```python
utils.reload_modules([utils])
cfg_patch_tst = train_config.FlatPatchTSTConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 283
    Validation Batches: 39
    Test Batches: 79
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 53.7633, mae: 5.2326, huber: 4.7624, swd: 11.3607, ept: 44.6535
    Epoch [1/50], Val Losses: mse: 39.6164, mae: 4.2242, huber: 3.7686, swd: 11.7221, ept: 100.9637
    Epoch [1/50], Test Losses: mse: 38.4232, mae: 4.1248, huber: 3.6704, swd: 11.5880, ept: 102.2013
      Epoch 1 composite train-obj: 4.762423
            Val objective improved inf → 3.7686, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 42.3524, mae: 4.3854, huber: 3.9259, swd: 8.6077, ept: 50.6748
    Epoch [2/50], Val Losses: mse: 34.8044, mae: 3.7452, huber: 3.2965, swd: 7.9825, ept: 122.2544
    Epoch [2/50], Test Losses: mse: 34.3883, mae: 3.7119, huber: 3.2631, swd: 7.9170, ept: 122.6207
      Epoch 2 composite train-obj: 3.925895
            Val objective improved 3.7686 → 3.2965, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.3422, mae: 4.0355, huber: 3.5826, swd: 7.0158, ept: 52.1937
    Epoch [3/50], Val Losses: mse: 31.2081, mae: 3.4388, huber: 2.9981, swd: 6.3910, ept: 131.2177
    Epoch [3/50], Test Losses: mse: 29.9916, mae: 3.3607, huber: 2.9216, swd: 6.0884, ept: 133.6740
      Epoch 3 composite train-obj: 3.582637
            Val objective improved 3.2965 → 2.9981, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.7570, mae: 3.8340, huber: 3.3850, swd: 6.2312, ept: 52.8669
    Epoch [4/50], Val Losses: mse: 26.6915, mae: 3.1637, huber: 2.7261, swd: 5.9822, ept: 138.1368
    Epoch [4/50], Test Losses: mse: 25.3348, mae: 3.0668, huber: 2.6301, swd: 6.0090, ept: 140.7092
      Epoch 4 composite train-obj: 3.384959
            Val objective improved 2.9981 → 2.7261, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.6982, mae: 3.6743, huber: 3.2282, swd: 5.5835, ept: 53.1503
    Epoch [5/50], Val Losses: mse: 23.9384, mae: 2.9098, huber: 2.4781, swd: 5.0225, ept: 145.1480
    Epoch [5/50], Test Losses: mse: 22.7769, mae: 2.8219, huber: 2.3913, swd: 5.1464, ept: 148.2558
      Epoch 5 composite train-obj: 3.228176
            Val objective improved 2.7261 → 2.4781, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 32.0729, mae: 3.5462, huber: 3.1030, swd: 5.1429, ept: 53.4009
    Epoch [6/50], Val Losses: mse: 22.6672, mae: 2.7453, huber: 2.3253, swd: 4.3637, ept: 147.1121
    Epoch [6/50], Test Losses: mse: 21.2659, mae: 2.6317, huber: 2.2145, swd: 4.3375, ept: 150.7162
      Epoch 6 composite train-obj: 3.102955
            Val objective improved 2.4781 → 2.3253, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 30.7784, mae: 3.4449, huber: 3.0036, swd: 4.8013, ept: 53.5240
    Epoch [7/50], Val Losses: mse: 21.3319, mae: 2.6590, huber: 2.2331, swd: 4.3299, ept: 152.3470
    Epoch [7/50], Test Losses: mse: 19.8019, mae: 2.5443, huber: 2.1202, swd: 4.0843, ept: 155.3062
      Epoch 7 composite train-obj: 3.003577
            Val objective improved 2.3253 → 2.2331, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 29.8655, mae: 3.3644, huber: 2.9250, swd: 4.5268, ept: 53.6072
    Epoch [8/50], Val Losses: mse: 21.0898, mae: 2.6642, huber: 2.2360, swd: 3.9761, ept: 152.2530
    Epoch [8/50], Test Losses: mse: 20.1507, mae: 2.5917, huber: 2.1648, swd: 3.7926, ept: 154.9162
      Epoch 8 composite train-obj: 2.924978
            No improvement (2.2360), counter 1/5
    Epoch [9/50], Train Losses: mse: 29.1478, mae: 3.3009, huber: 2.8629, swd: 4.2950, ept: 53.8281
    Epoch [9/50], Val Losses: mse: 18.1898, mae: 2.4003, huber: 1.9822, swd: 3.4828, ept: 157.4176
    Epoch [9/50], Test Losses: mse: 16.6908, mae: 2.2957, huber: 1.8806, swd: 3.5124, ept: 160.0133
      Epoch 9 composite train-obj: 2.862912
            Val objective improved 2.2331 → 1.9822, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 28.5430, mae: 3.2459, huber: 2.8094, swd: 4.1662, ept: 53.7782
    Epoch [10/50], Val Losses: mse: 25.4234, mae: 2.9542, huber: 2.5210, swd: 4.3343, ept: 147.2429
    Epoch [10/50], Test Losses: mse: 24.7260, mae: 2.9029, huber: 2.4707, swd: 4.5536, ept: 148.1248
      Epoch 10 composite train-obj: 2.809364
            No improvement (2.5210), counter 1/5
    Epoch [11/50], Train Losses: mse: 28.2841, mae: 3.2191, huber: 2.7836, swd: 4.0957, ept: 53.9110
    Epoch [11/50], Val Losses: mse: 16.6762, mae: 2.3053, huber: 1.8872, swd: 3.1056, ept: 158.4778
    Epoch [11/50], Test Losses: mse: 15.1228, mae: 2.2120, huber: 1.7942, swd: 2.8203, ept: 162.5052
      Epoch 11 composite train-obj: 2.783606
            Val objective improved 1.9822 → 1.8872, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 27.4880, mae: 3.1497, huber: 2.7161, swd: 3.9316, ept: 54.0306
    Epoch [12/50], Val Losses: mse: 20.5730, mae: 2.4965, huber: 2.0814, swd: 3.2774, ept: 156.9230
    Epoch [12/50], Test Losses: mse: 19.1622, mae: 2.4016, huber: 1.9890, swd: 3.2553, ept: 159.3812
      Epoch 12 composite train-obj: 2.716082
            No improvement (2.0814), counter 1/5
    Epoch [13/50], Train Losses: mse: 27.1408, mae: 3.1146, huber: 2.6820, swd: 3.8515, ept: 54.0270
    Epoch [13/50], Val Losses: mse: 17.4445, mae: 2.3916, huber: 1.9728, swd: 3.4376, ept: 155.3787
    Epoch [13/50], Test Losses: mse: 16.2698, mae: 2.3210, huber: 1.9015, swd: 3.1745, ept: 156.8308
      Epoch 13 composite train-obj: 2.681998
            No improvement (1.9728), counter 2/5
    Epoch [14/50], Train Losses: mse: 26.8394, mae: 3.0858, huber: 2.6542, swd: 3.7768, ept: 54.1298
    Epoch [14/50], Val Losses: mse: 20.0728, mae: 2.4741, huber: 2.0589, swd: 2.8648, ept: 154.8255
    Epoch [14/50], Test Losses: mse: 18.3336, mae: 2.3787, huber: 1.9625, swd: 2.7263, ept: 157.5452
      Epoch 14 composite train-obj: 2.654178
            No improvement (2.0589), counter 3/5
    Epoch [15/50], Train Losses: mse: 26.6171, mae: 3.0652, huber: 2.6344, swd: 3.7057, ept: 54.2513
    Epoch [15/50], Val Losses: mse: 15.3174, mae: 2.0818, huber: 1.6779, swd: 2.7617, ept: 163.9567
    Epoch [15/50], Test Losses: mse: 13.5273, mae: 1.9593, huber: 1.5579, swd: 2.5902, ept: 167.6433
      Epoch 15 composite train-obj: 2.634425
            Val objective improved 1.8872 → 1.6779, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 26.2378, mae: 3.0318, huber: 2.6018, swd: 3.6365, ept: 54.1970
    Epoch [16/50], Val Losses: mse: 15.8877, mae: 2.1438, huber: 1.7389, swd: 2.7629, ept: 162.1111
    Epoch [16/50], Test Losses: mse: 14.0856, mae: 2.0257, huber: 1.6230, swd: 2.5919, ept: 165.6194
      Epoch 16 composite train-obj: 2.601814
            No improvement (1.7389), counter 1/5
    Epoch [17/50], Train Losses: mse: 25.8763, mae: 3.0039, huber: 2.5747, swd: 3.5768, ept: 54.2056
    Epoch [17/50], Val Losses: mse: 16.5276, mae: 2.1585, huber: 1.7579, swd: 2.7468, ept: 164.0230
    Epoch [17/50], Test Losses: mse: 14.6964, mae: 2.0351, huber: 1.6365, swd: 2.4466, ept: 167.2423
      Epoch 17 composite train-obj: 2.574687
            No improvement (1.7579), counter 2/5
    Epoch [18/50], Train Losses: mse: 25.4907, mae: 2.9697, huber: 2.5416, swd: 3.5160, ept: 54.2790
    Epoch [18/50], Val Losses: mse: 15.9637, mae: 2.1379, huber: 1.7351, swd: 2.6122, ept: 161.4209
    Epoch [18/50], Test Losses: mse: 13.9014, mae: 2.0171, huber: 1.6159, swd: 2.4442, ept: 163.5815
      Epoch 18 composite train-obj: 2.541589
            No improvement (1.7351), counter 3/5
    Epoch [19/50], Train Losses: mse: 25.2285, mae: 2.9425, huber: 2.5154, swd: 3.4546, ept: 54.1960
    Epoch [19/50], Val Losses: mse: 13.3730, mae: 1.9454, huber: 1.5471, swd: 2.4058, ept: 165.7541
    Epoch [19/50], Test Losses: mse: 11.5963, mae: 1.8327, huber: 1.4338, swd: 2.1205, ept: 169.7294
      Epoch 19 composite train-obj: 2.515428
            Val objective improved 1.6779 → 1.5471, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 25.0470, mae: 2.9257, huber: 2.4991, swd: 3.3896, ept: 54.2635
    Epoch [20/50], Val Losses: mse: 15.4810, mae: 2.1525, huber: 1.7425, swd: 2.9876, ept: 162.6634
    Epoch [20/50], Test Losses: mse: 14.1094, mae: 2.0621, huber: 1.6529, swd: 2.6540, ept: 166.3120
      Epoch 20 composite train-obj: 2.499100
            No improvement (1.7425), counter 1/5
    Epoch [21/50], Train Losses: mse: 25.0665, mae: 2.9218, huber: 2.4956, swd: 3.3904, ept: 54.2590
    Epoch [21/50], Val Losses: mse: 13.7910, mae: 2.0306, huber: 1.6275, swd: 2.6075, ept: 163.5079
    Epoch [21/50], Test Losses: mse: 12.3139, mae: 1.9352, huber: 1.5341, swd: 2.3803, ept: 165.3022
      Epoch 21 composite train-obj: 2.495605
            No improvement (1.6275), counter 2/5
    Epoch [22/50], Train Losses: mse: 24.7748, mae: 2.8976, huber: 2.4725, swd: 3.3635, ept: 54.2867
    Epoch [22/50], Val Losses: mse: 16.0678, mae: 2.1427, huber: 1.7375, swd: 2.8355, ept: 164.6055
    Epoch [22/50], Test Losses: mse: 13.9076, mae: 1.9836, huber: 1.5820, swd: 2.6198, ept: 168.8829
      Epoch 22 composite train-obj: 2.472475
            No improvement (1.7375), counter 3/5
    Epoch [23/50], Train Losses: mse: 24.8138, mae: 2.8923, huber: 2.4675, swd: 3.3456, ept: 54.4613
    Epoch [23/50], Val Losses: mse: 14.7718, mae: 2.0702, huber: 1.6678, swd: 2.4600, ept: 164.8623
    Epoch [23/50], Test Losses: mse: 12.8922, mae: 1.9372, huber: 1.5369, swd: 2.1506, ept: 168.9200
      Epoch 23 composite train-obj: 2.467507
            No improvement (1.6678), counter 4/5
    Epoch [24/50], Train Losses: mse: 24.3804, mae: 2.8608, huber: 2.4365, swd: 3.2678, ept: 54.4183
    Epoch [24/50], Val Losses: mse: 15.6348, mae: 2.0322, huber: 1.6367, swd: 2.6238, ept: 167.3908
    Epoch [24/50], Test Losses: mse: 13.9979, mae: 1.9270, huber: 1.5349, swd: 2.4858, ept: 170.4837
      Epoch 24 composite train-obj: 2.436475
    Epoch [24/50], Test Losses: mse: 11.5963, mae: 1.8327, huber: 1.4338, swd: 2.1205, ept: 169.7294
    Best round's Test MSE: 11.5963, MAE: 1.8327, SWD: 2.1205
    Best round's Validation MSE: 13.3730, MAE: 1.9454, SWD: 2.4058
    Best round's Test verification MSE : 11.5963, MAE: 1.8327, SWD: 2.1205
    Time taken: 79.52 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 53.9134, mae: 5.2433, huber: 4.7730, swd: 10.9991, ept: 45.1546
    Epoch [1/50], Val Losses: mse: 39.8939, mae: 4.2644, huber: 3.8068, swd: 12.3766, ept: 99.1889
    Epoch [1/50], Test Losses: mse: 38.8438, mae: 4.1877, huber: 3.7307, swd: 11.9390, ept: 99.7244
      Epoch 1 composite train-obj: 4.773014
            Val objective improved inf → 3.8068, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 41.9514, mae: 4.3746, huber: 3.9153, swd: 8.3525, ept: 50.9087
    Epoch [2/50], Val Losses: mse: 33.1025, mae: 3.6600, huber: 3.2151, swd: 8.0310, ept: 121.8751
    Epoch [2/50], Test Losses: mse: 32.1546, mae: 3.5862, huber: 3.1423, swd: 7.5479, ept: 124.1434
      Epoch 2 composite train-obj: 3.915331
            Val objective improved 3.8068 → 3.2151, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.0060, mae: 4.0342, huber: 3.5810, swd: 6.6394, ept: 52.2332
    Epoch [3/50], Val Losses: mse: 29.4509, mae: 3.4088, huber: 2.9635, swd: 6.5879, ept: 130.2960
    Epoch [3/50], Test Losses: mse: 28.3732, mae: 3.3256, huber: 2.8812, swd: 6.2847, ept: 133.9336
      Epoch 3 composite train-obj: 3.581019
            Val objective improved 3.2151 → 2.9635, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.3178, mae: 3.8063, huber: 3.3576, swd: 5.7087, ept: 52.8976
    Epoch [4/50], Val Losses: mse: 26.9359, mae: 3.1723, huber: 2.7346, swd: 5.5431, ept: 137.5864
    Epoch [4/50], Test Losses: mse: 25.8064, mae: 3.0916, huber: 2.6548, swd: 5.1972, ept: 139.4699
      Epoch 4 composite train-obj: 3.357574
            Val objective improved 2.9635 → 2.7346, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.5170, mae: 3.6544, huber: 3.2088, swd: 5.2218, ept: 53.1782
    Epoch [5/50], Val Losses: mse: 24.6661, mae: 2.9108, huber: 2.4846, swd: 4.8961, ept: 145.1554
    Epoch [5/50], Test Losses: mse: 23.2783, mae: 2.8109, huber: 2.3869, swd: 4.5270, ept: 148.5427
      Epoch 5 composite train-obj: 3.208792
            Val objective improved 2.7346 → 2.4846, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 32.0972, mae: 3.5380, huber: 3.0949, swd: 4.8683, ept: 53.4376
    Epoch [6/50], Val Losses: mse: 22.6959, mae: 2.7922, huber: 2.3628, swd: 4.0857, ept: 146.4829
    Epoch [6/50], Test Losses: mse: 20.9784, mae: 2.6749, huber: 2.2484, swd: 3.9850, ept: 149.8947
      Epoch 6 composite train-obj: 3.094937
            Val objective improved 2.4846 → 2.3628, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 30.9893, mae: 3.4487, huber: 3.0075, swd: 4.6038, ept: 53.5220
    Epoch [7/50], Val Losses: mse: 21.6714, mae: 2.6515, huber: 2.2322, swd: 4.5926, ept: 150.7497
    Epoch [7/50], Test Losses: mse: 20.3976, mae: 2.5715, huber: 2.1536, swd: 4.6317, ept: 152.7882
      Epoch 7 composite train-obj: 3.007467
            Val objective improved 2.3628 → 2.2322, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 30.0411, mae: 3.3655, huber: 2.9265, swd: 4.4218, ept: 53.8073
    Epoch [8/50], Val Losses: mse: 20.5086, mae: 2.5906, huber: 2.1705, swd: 3.8469, ept: 154.0250
    Epoch [8/50], Test Losses: mse: 19.0479, mae: 2.4859, huber: 2.0677, swd: 3.8289, ept: 156.2807
      Epoch 8 composite train-obj: 2.926462
            Val objective improved 2.2322 → 2.1705, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 29.3461, mae: 3.3110, huber: 2.8731, swd: 4.2333, ept: 53.8087
    Epoch [9/50], Val Losses: mse: 19.3839, mae: 2.5226, huber: 2.1016, swd: 3.5649, ept: 152.5320
    Epoch [9/50], Test Losses: mse: 17.8545, mae: 2.4197, huber: 1.9993, swd: 3.4213, ept: 156.0123
      Epoch 9 composite train-obj: 2.873137
            Val objective improved 2.1705 → 2.1016, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 28.7470, mae: 3.2620, huber: 2.8252, swd: 4.1040, ept: 53.9064
    Epoch [10/50], Val Losses: mse: 19.2542, mae: 2.4831, huber: 2.0661, swd: 3.3812, ept: 154.3941
    Epoch [10/50], Test Losses: mse: 16.6479, mae: 2.3392, huber: 1.9236, swd: 3.4033, ept: 157.1368
      Epoch 10 composite train-obj: 2.825207
            Val objective improved 2.1016 → 2.0661, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 27.8884, mae: 3.1936, huber: 2.7584, swd: 3.9104, ept: 54.0359
    Epoch [11/50], Val Losses: mse: 17.7359, mae: 2.3561, huber: 1.9411, swd: 3.5113, ept: 158.9127
    Epoch [11/50], Test Losses: mse: 16.5643, mae: 2.2856, huber: 1.8709, swd: 3.4448, ept: 161.3394
      Epoch 11 composite train-obj: 2.758435
            Val objective improved 2.0661 → 1.9411, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 27.5423, mae: 3.1615, huber: 2.7271, swd: 3.8206, ept: 53.8707
    Epoch [12/50], Val Losses: mse: 17.9180, mae: 2.4003, huber: 1.9846, swd: 2.8595, ept: 156.2467
    Epoch [12/50], Test Losses: mse: 16.3463, mae: 2.2929, huber: 1.8794, swd: 2.7697, ept: 159.7311
      Epoch 12 composite train-obj: 2.727108
            No improvement (1.9846), counter 1/5
    Epoch [13/50], Train Losses: mse: 27.1666, mae: 3.1265, huber: 2.6930, swd: 3.7480, ept: 54.1321
    Epoch [13/50], Val Losses: mse: 17.5939, mae: 2.2947, huber: 1.8835, swd: 3.1226, ept: 160.2990
    Epoch [13/50], Test Losses: mse: 15.9288, mae: 2.1858, huber: 1.7776, swd: 3.0519, ept: 163.2514
      Epoch 13 composite train-obj: 2.692954
            Val objective improved 1.9411 → 1.8835, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 26.6665, mae: 3.0821, huber: 2.6498, swd: 3.5821, ept: 53.8776
    Epoch [14/50], Val Losses: mse: 16.9583, mae: 2.2465, huber: 1.8375, swd: 2.9362, ept: 160.9649
    Epoch [14/50], Test Losses: mse: 15.5460, mae: 2.1332, huber: 1.7267, swd: 2.7606, ept: 163.1024
      Epoch 14 composite train-obj: 2.649838
            Val objective improved 1.8835 → 1.8375, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 26.4945, mae: 3.0634, huber: 2.6317, swd: 3.5589, ept: 54.1909
    Epoch [15/50], Val Losses: mse: 14.8320, mae: 2.1190, huber: 1.7083, swd: 2.7318, ept: 163.1792
    Epoch [15/50], Test Losses: mse: 13.6630, mae: 2.0464, huber: 1.6373, swd: 2.5907, ept: 165.7177
      Epoch 15 composite train-obj: 2.631695
            Val objective improved 1.8375 → 1.7083, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 26.1052, mae: 3.0292, huber: 2.5987, swd: 3.4808, ept: 54.0696
    Epoch [16/50], Val Losses: mse: 20.3044, mae: 2.4100, huber: 1.9999, swd: 2.4435, ept: 160.4548
    Epoch [16/50], Test Losses: mse: 16.6760, mae: 2.1644, huber: 1.7592, swd: 1.9623, ept: 165.4181
      Epoch 16 composite train-obj: 2.598707
            No improvement (1.9999), counter 1/5
    Epoch [17/50], Train Losses: mse: 25.8514, mae: 3.0068, huber: 2.5769, swd: 3.4019, ept: 54.1548
    Epoch [17/50], Val Losses: mse: 15.7812, mae: 2.1006, huber: 1.6995, swd: 2.6765, ept: 164.2806
    Epoch [17/50], Test Losses: mse: 14.4235, mae: 2.0193, huber: 1.6181, swd: 2.6285, ept: 166.8632
      Epoch 17 composite train-obj: 2.576902
            Val objective improved 1.7083 → 1.6995, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 25.6273, mae: 2.9835, huber: 2.5543, swd: 3.3477, ept: 54.2665
    Epoch [18/50], Val Losses: mse: 16.5682, mae: 2.2475, huber: 1.8328, swd: 2.5356, ept: 162.6598
    Epoch [18/50], Test Losses: mse: 15.3345, mae: 2.1660, huber: 1.7529, swd: 2.4775, ept: 164.2716
      Epoch 18 composite train-obj: 2.554319
            No improvement (1.8328), counter 1/5
    Epoch [19/50], Train Losses: mse: 25.3795, mae: 2.9589, huber: 2.5305, swd: 3.2912, ept: 54.3454
    Epoch [19/50], Val Losses: mse: 14.3612, mae: 2.0232, huber: 1.6227, swd: 2.4715, ept: 165.3007
    Epoch [19/50], Test Losses: mse: 12.7337, mae: 1.9225, huber: 1.5245, swd: 2.3464, ept: 168.0242
      Epoch 19 composite train-obj: 2.530493
            Val objective improved 1.6995 → 1.6227, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 25.3038, mae: 2.9512, huber: 2.5232, swd: 3.2564, ept: 54.1484
    Epoch [20/50], Val Losses: mse: 14.2197, mae: 2.0358, huber: 1.6319, swd: 2.6837, ept: 163.9460
    Epoch [20/50], Test Losses: mse: 13.0289, mae: 1.9592, huber: 1.5583, swd: 2.6120, ept: 166.1659
      Epoch 20 composite train-obj: 2.523244
            No improvement (1.6319), counter 1/5
    Epoch [21/50], Train Losses: mse: 25.0340, mae: 2.9231, huber: 2.4963, swd: 3.2215, ept: 54.3386
    Epoch [21/50], Val Losses: mse: 14.8112, mae: 2.0862, huber: 1.6760, swd: 2.6908, ept: 164.5554
    Epoch [21/50], Test Losses: mse: 12.6996, mae: 1.9487, huber: 1.5421, swd: 2.4832, ept: 168.2063
      Epoch 21 composite train-obj: 2.496286
            No improvement (1.6760), counter 2/5
    Epoch [22/50], Train Losses: mse: 24.7978, mae: 2.9044, huber: 2.4780, swd: 3.1719, ept: 54.1898
    Epoch [22/50], Val Losses: mse: 14.7785, mae: 2.0308, huber: 1.6313, swd: 2.4685, ept: 165.4153
    Epoch [22/50], Test Losses: mse: 12.4486, mae: 1.8724, huber: 1.4763, swd: 2.2927, ept: 169.0445
      Epoch 22 composite train-obj: 2.477957
            No improvement (1.6313), counter 3/5
    Epoch [23/50], Train Losses: mse: 24.8281, mae: 2.8988, huber: 2.4729, swd: 3.1565, ept: 54.4471
    Epoch [23/50], Val Losses: mse: 15.9044, mae: 2.0774, huber: 1.6746, swd: 2.5577, ept: 164.8238
    Epoch [23/50], Test Losses: mse: 14.2089, mae: 1.9609, huber: 1.5604, swd: 2.3581, ept: 168.1058
      Epoch 23 composite train-obj: 2.472948
            No improvement (1.6746), counter 4/5
    Epoch [24/50], Train Losses: mse: 24.6125, mae: 2.8841, huber: 2.4585, swd: 3.1299, ept: 54.3486
    Epoch [24/50], Val Losses: mse: 14.8330, mae: 2.1028, huber: 1.6988, swd: 2.1995, ept: 164.4865
    Epoch [24/50], Test Losses: mse: 12.9583, mae: 1.9659, huber: 1.5662, swd: 2.0893, ept: 167.7747
      Epoch 24 composite train-obj: 2.458538
    Epoch [24/50], Test Losses: mse: 12.7337, mae: 1.9225, huber: 1.5245, swd: 2.3464, ept: 168.0242
    Best round's Test MSE: 12.7337, MAE: 1.9225, SWD: 2.3464
    Best round's Validation MSE: 14.3612, MAE: 2.0232, SWD: 2.4715
    Best round's Test verification MSE : 12.7337, MAE: 1.9225, SWD: 2.3464
    Time taken: 86.37 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 53.2620, mae: 5.2039, huber: 4.7340, swd: 10.7794, ept: 45.0094
    Epoch [1/50], Val Losses: mse: 40.7455, mae: 4.3756, huber: 3.9143, swd: 10.8264, ept: 96.9316
    Epoch [1/50], Test Losses: mse: 39.6773, mae: 4.2973, huber: 3.8362, swd: 10.9144, ept: 97.8510
      Epoch 1 composite train-obj: 4.733954
            Val objective improved inf → 3.9143, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 41.8867, mae: 4.3482, huber: 3.8893, swd: 8.1782, ept: 50.8524
    Epoch [2/50], Val Losses: mse: 33.5019, mae: 3.6316, huber: 3.1870, swd: 7.3517, ept: 127.6406
    Epoch [2/50], Test Losses: mse: 32.5340, mae: 3.5518, huber: 3.1080, swd: 6.8891, ept: 128.5570
      Epoch 2 composite train-obj: 3.889324
            Val objective improved 3.9143 → 3.1870, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.1493, mae: 4.0216, huber: 3.5692, swd: 6.7543, ept: 52.3344
    Epoch [3/50], Val Losses: mse: 29.3557, mae: 3.3747, huber: 2.9341, swd: 6.6516, ept: 126.5261
    Epoch [3/50], Test Losses: mse: 28.8777, mae: 3.3104, huber: 2.8712, swd: 6.5328, ept: 128.3492
      Epoch 3 composite train-obj: 3.569233
            Val objective improved 3.1870 → 2.9341, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.4641, mae: 3.8129, huber: 3.3642, swd: 5.8634, ept: 52.8468
    Epoch [4/50], Val Losses: mse: 26.1374, mae: 3.1311, huber: 2.6937, swd: 5.4844, ept: 141.1665
    Epoch [4/50], Test Losses: mse: 25.0016, mae: 3.0574, huber: 2.6196, swd: 5.3603, ept: 143.0896
      Epoch 4 composite train-obj: 3.364178
            Val objective improved 2.9341 → 2.6937, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.4179, mae: 3.6529, huber: 3.2073, swd: 5.2507, ept: 53.2723
    Epoch [5/50], Val Losses: mse: 25.1326, mae: 3.0251, huber: 2.5904, swd: 4.5326, ept: 143.2760
    Epoch [5/50], Test Losses: mse: 23.6547, mae: 2.9268, huber: 2.4921, swd: 4.2778, ept: 147.2245
      Epoch 5 composite train-obj: 3.207341
            Val objective improved 2.6937 → 2.5904, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.6970, mae: 3.5123, huber: 3.0699, swd: 4.7157, ept: 53.5056
    Epoch [6/50], Val Losses: mse: 22.8173, mae: 2.7848, huber: 2.3551, swd: 3.3565, ept: 152.8061
    Epoch [6/50], Test Losses: mse: 20.7507, mae: 2.6469, huber: 2.2203, swd: 3.3589, ept: 156.0404
      Epoch 6 composite train-obj: 3.069923
            Val objective improved 2.5904 → 2.3551, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 30.2767, mae: 3.4026, huber: 2.9621, swd: 4.3350, ept: 53.5759
    Epoch [7/50], Val Losses: mse: 20.3732, mae: 2.6418, huber: 2.2174, swd: 3.6515, ept: 151.4976
    Epoch [7/50], Test Losses: mse: 18.8014, mae: 2.5303, huber: 2.1080, swd: 3.3986, ept: 155.1869
      Epoch 7 composite train-obj: 2.962134
            Val objective improved 2.3551 → 2.2174, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 29.4525, mae: 3.3306, huber: 2.8919, swd: 4.1248, ept: 53.7314
    Epoch [8/50], Val Losses: mse: 21.1602, mae: 2.5700, huber: 2.1488, swd: 3.3456, ept: 154.9006
    Epoch [8/50], Test Losses: mse: 18.6269, mae: 2.4071, huber: 1.9893, swd: 3.1528, ept: 158.8386
      Epoch 8 composite train-obj: 2.891919
            Val objective improved 2.2174 → 2.1488, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 28.6921, mae: 3.2602, huber: 2.8233, swd: 3.9140, ept: 53.9071
    Epoch [9/50], Val Losses: mse: 18.2491, mae: 2.4847, huber: 2.0611, swd: 3.5456, ept: 157.6518
    Epoch [9/50], Test Losses: mse: 17.3966, mae: 2.4256, huber: 2.0035, swd: 3.5481, ept: 159.9867
      Epoch 9 composite train-obj: 2.823275
            Val objective improved 2.1488 → 2.0611, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 28.1426, mae: 3.2115, huber: 2.7760, swd: 3.8038, ept: 53.8304
    Epoch [10/50], Val Losses: mse: 18.5232, mae: 2.3805, huber: 1.9662, swd: 3.0282, ept: 158.5136
    Epoch [10/50], Test Losses: mse: 16.7212, mae: 2.2667, huber: 1.8534, swd: 2.8740, ept: 162.1413
      Epoch 10 composite train-obj: 2.776037
            Val objective improved 2.0611 → 1.9662, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 27.5865, mae: 3.1608, huber: 2.7266, swd: 3.6588, ept: 53.9441
    Epoch [11/50], Val Losses: mse: 20.8098, mae: 2.6040, huber: 2.1801, swd: 3.2020, ept: 156.0409
    Epoch [11/50], Test Losses: mse: 18.8904, mae: 2.4841, huber: 2.0602, swd: 2.9614, ept: 159.2479
      Epoch 11 composite train-obj: 2.726612
            No improvement (2.1801), counter 1/5
    Epoch [12/50], Train Losses: mse: 27.1741, mae: 3.1178, huber: 2.6853, swd: 3.5668, ept: 53.9441
    Epoch [12/50], Val Losses: mse: 16.7698, mae: 2.2457, huber: 1.8329, swd: 2.7219, ept: 162.0117
    Epoch [12/50], Test Losses: mse: 15.1726, mae: 2.1563, huber: 1.7441, swd: 2.6647, ept: 164.1152
      Epoch 12 composite train-obj: 2.685285
            Val objective improved 1.9662 → 1.8329, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 26.7640, mae: 3.0825, huber: 2.6511, swd: 3.4967, ept: 54.1731
    Epoch [13/50], Val Losses: mse: 18.9123, mae: 2.3706, huber: 1.9636, swd: 3.0791, ept: 158.5408
    Epoch [13/50], Test Losses: mse: 16.6850, mae: 2.2278, huber: 1.8219, swd: 2.9159, ept: 162.5302
      Epoch 13 composite train-obj: 2.651076
            No improvement (1.9636), counter 1/5
    Epoch [14/50], Train Losses: mse: 26.4264, mae: 3.0532, huber: 2.6224, swd: 3.4429, ept: 54.0683
    Epoch [14/50], Val Losses: mse: 15.4468, mae: 2.1584, huber: 1.7500, swd: 2.8895, ept: 162.4103
    Epoch [14/50], Test Losses: mse: 14.0861, mae: 2.0666, huber: 1.6585, swd: 2.8071, ept: 165.8910
      Epoch 14 composite train-obj: 2.622443
            Val objective improved 1.8329 → 1.7500, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 26.0890, mae: 3.0176, huber: 2.5881, swd: 3.3587, ept: 54.0924
    Epoch [15/50], Val Losses: mse: 15.6694, mae: 2.1580, huber: 1.7510, swd: 2.9373, ept: 164.6737
    Epoch [15/50], Test Losses: mse: 13.9088, mae: 2.0334, huber: 1.6288, swd: 2.5977, ept: 167.4933
      Epoch 15 composite train-obj: 2.588063
            No improvement (1.7510), counter 1/5
    Epoch [16/50], Train Losses: mse: 25.7511, mae: 2.9896, huber: 2.5610, swd: 3.3300, ept: 54.0742
    Epoch [16/50], Val Losses: mse: 16.7266, mae: 2.2370, huber: 1.8293, swd: 2.4942, ept: 160.7147
    Epoch [16/50], Test Losses: mse: 15.1395, mae: 2.1278, huber: 1.7226, swd: 2.4338, ept: 163.1558
      Epoch 16 composite train-obj: 2.561005
            No improvement (1.8293), counter 2/5
    Epoch [17/50], Train Losses: mse: 25.5258, mae: 2.9697, huber: 2.5417, swd: 3.2759, ept: 54.0958
    Epoch [17/50], Val Losses: mse: 18.2437, mae: 2.4458, huber: 2.0288, swd: 2.6533, ept: 155.5449
    Epoch [17/50], Test Losses: mse: 17.2807, mae: 2.3951, huber: 1.9793, swd: 2.7896, ept: 156.8134
      Epoch 17 composite train-obj: 2.541735
            No improvement (2.0288), counter 3/5
    Epoch [18/50], Train Losses: mse: 25.4930, mae: 2.9615, huber: 2.5339, swd: 3.2484, ept: 54.3186
    Epoch [18/50], Val Losses: mse: 16.9205, mae: 2.2164, huber: 1.8072, swd: 3.1019, ept: 163.7082
    Epoch [18/50], Test Losses: mse: 14.8083, mae: 2.0787, huber: 1.6734, swd: 2.9673, ept: 166.7014
      Epoch 18 composite train-obj: 2.533934
            No improvement (1.8072), counter 4/5
    Epoch [19/50], Train Losses: mse: 25.1100, mae: 2.9271, huber: 2.5007, swd: 3.2029, ept: 54.1807
    Epoch [19/50], Val Losses: mse: 17.0799, mae: 2.2364, huber: 1.8260, swd: 2.4669, ept: 163.0825
    Epoch [19/50], Test Losses: mse: 15.3713, mae: 2.1204, huber: 1.7136, swd: 2.3058, ept: 166.6437
      Epoch 19 composite train-obj: 2.500680
    Epoch [19/50], Test Losses: mse: 14.0861, mae: 2.0666, huber: 1.6585, swd: 2.8071, ept: 165.8910
    Best round's Test MSE: 14.0861, MAE: 2.0666, SWD: 2.8071
    Best round's Validation MSE: 15.4468, MAE: 2.1584, SWD: 2.8895
    Best round's Test verification MSE : 14.0861, MAE: 2.0666, SWD: 2.8071
    Time taken: 69.97 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq96_pred196_20250514_0116)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.8054 ± 1.0178
      mae: 1.9406 ± 0.0964
      huber: 1.5389 ± 0.0923
      swd: 2.4246 ± 0.2857
      ept: 167.8815 ± 1.5703
      count: 39.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 14.3937 ± 0.8469
      mae: 2.0423 ± 0.0880
      huber: 1.6399 ± 0.0837
      swd: 2.5889 ± 0.2142
      ept: 164.4884 ± 1.4810
      count: 39.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 235.92 seconds
    
    Experiment complete: PatchTST_lorenz_seq96_pred196_20250514_0116
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 96-336
##### huber


```python
utils.reload_modules([utils])
cfg_patch_tst = train_config.FlatPatchTSTConfig(
    seq_len=96,
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
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9175, 9.0168, 8.6295], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
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
    
    Epoch [1/50], Train Losses: mse: 61.8450, mae: 5.7746, huber: 5.2997, swd: 13.1610, ept: 49.5004
    Epoch [1/50], Val Losses: mse: 58.1334, mae: 5.6239, huber: 5.1503, swd: 15.6692, ept: 81.3605
    Epoch [1/50], Test Losses: mse: 55.7329, mae: 5.4541, huber: 4.9813, swd: 14.6290, ept: 84.4542
      Epoch 1 composite train-obj: 5.299658
            Val objective improved inf → 5.1503, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 53.6633, mae: 5.2224, huber: 4.7534, swd: 12.1719, ept: 56.4150
    Epoch [2/50], Val Losses: mse: 55.1477, mae: 5.4028, huber: 4.9305, swd: 15.0048, ept: 88.7420
    Epoch [2/50], Test Losses: mse: 53.4479, mae: 5.2716, huber: 4.7999, swd: 14.4623, ept: 89.6501
      Epoch 2 composite train-obj: 4.753398
            Val objective improved 5.1503 → 4.9305, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.5303, mae: 4.9691, huber: 4.5037, swd: 10.9089, ept: 59.2407
    Epoch [3/50], Val Losses: mse: 65.5293, mae: 5.7687, huber: 5.2980, swd: 12.8964, ept: 103.0536
    Epoch [3/50], Test Losses: mse: 64.0860, mae: 5.6934, huber: 5.2235, swd: 13.1159, ept: 104.4482
      Epoch 3 composite train-obj: 4.503681
            No improvement (5.2980), counter 1/5
    Epoch [4/50], Train Losses: mse: 49.1743, mae: 4.8407, huber: 4.3776, swd: 10.0937, ept: 59.6907
    Epoch [4/50], Val Losses: mse: 46.0705, mae: 4.5536, huber: 4.0985, swd: 12.1439, ept: 136.0467
    Epoch [4/50], Test Losses: mse: 44.4484, mae: 4.4524, huber: 3.9975, swd: 11.7079, ept: 142.1341
      Epoch 4 composite train-obj: 4.377644
            Val objective improved 4.9305 → 4.0985, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 47.0054, mae: 4.6576, huber: 4.1978, swd: 9.2748, ept: 60.9034
    Epoch [5/50], Val Losses: mse: 45.9797, mae: 4.5708, huber: 4.1110, swd: 9.1029, ept: 143.2611
    Epoch [5/50], Test Losses: mse: 43.6819, mae: 4.4026, huber: 3.9442, swd: 8.4535, ept: 151.1211
      Epoch 5 composite train-obj: 4.197847
            No improvement (4.1110), counter 1/5
    Epoch [6/50], Train Losses: mse: 46.5506, mae: 4.6248, huber: 4.1655, swd: 8.9057, ept: 60.8357
    Epoch [6/50], Val Losses: mse: 46.8511, mae: 4.6766, huber: 4.2142, swd: 9.5778, ept: 133.8854
    Epoch [6/50], Test Losses: mse: 44.0884, mae: 4.4795, huber: 4.0201, swd: 9.8284, ept: 138.6241
      Epoch 6 composite train-obj: 4.165498
            No improvement (4.2142), counter 2/5
    Epoch [7/50], Train Losses: mse: 45.7942, mae: 4.5607, huber: 4.1025, swd: 8.6283, ept: 61.0471
    Epoch [7/50], Val Losses: mse: 42.9196, mae: 4.3560, huber: 3.8993, swd: 8.9642, ept: 148.1161
    Epoch [7/50], Test Losses: mse: 39.8861, mae: 4.1661, huber: 3.7097, swd: 8.8582, ept: 154.5646
      Epoch 7 composite train-obj: 4.102527
            Val objective improved 4.0985 → 3.8993, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 44.5456, mae: 4.4620, huber: 4.0056, swd: 8.0954, ept: 61.5960
    Epoch [8/50], Val Losses: mse: 47.2880, mae: 4.6235, huber: 4.1652, swd: 8.2153, ept: 136.9560
    Epoch [8/50], Test Losses: mse: 44.7503, mae: 4.4397, huber: 3.9824, swd: 7.7768, ept: 142.9161
      Epoch 8 composite train-obj: 4.005572
            No improvement (4.1652), counter 1/5
    Epoch [9/50], Train Losses: mse: 44.1486, mae: 4.4119, huber: 3.9567, swd: 7.6909, ept: 61.8415
    Epoch [9/50], Val Losses: mse: 43.9263, mae: 4.4575, huber: 4.0010, swd: 7.9829, ept: 148.0746
    Epoch [9/50], Test Losses: mse: 41.2327, mae: 4.2863, huber: 3.8310, swd: 7.9175, ept: 153.3761
      Epoch 9 composite train-obj: 3.956747
            No improvement (4.0010), counter 2/5
    Epoch [10/50], Train Losses: mse: 44.1009, mae: 4.4134, huber: 3.9582, swd: 7.6772, ept: 61.9095
    Epoch [10/50], Val Losses: mse: 42.3957, mae: 4.2123, huber: 3.7622, swd: 7.0145, ept: 164.4041
    Epoch [10/50], Test Losses: mse: 39.5053, mae: 4.0172, huber: 3.5702, swd: 7.0479, ept: 166.8404
      Epoch 10 composite train-obj: 3.958210
            Val objective improved 3.8993 → 3.7622, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 43.1565, mae: 4.3365, huber: 3.8829, swd: 7.3703, ept: 62.3260
    Epoch [11/50], Val Losses: mse: 41.8487, mae: 4.2436, huber: 3.7931, swd: 7.8485, ept: 155.4169
    Epoch [11/50], Test Losses: mse: 39.6105, mae: 4.0909, huber: 3.6417, swd: 7.8941, ept: 161.6898
      Epoch 11 composite train-obj: 3.882875
            No improvement (3.7931), counter 1/5
    Epoch [12/50], Train Losses: mse: 42.6673, mae: 4.2973, huber: 3.8443, swd: 7.1920, ept: 62.4147
    Epoch [12/50], Val Losses: mse: 40.0343, mae: 4.0632, huber: 3.6161, swd: 7.7319, ept: 165.3603
    Epoch [12/50], Test Losses: mse: 36.8738, mae: 3.8475, huber: 3.4035, swd: 7.4722, ept: 175.1883
      Epoch 12 composite train-obj: 3.844309
            Val objective improved 3.7622 → 3.6161, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 42.1172, mae: 4.2503, huber: 3.7983, swd: 7.0198, ept: 62.5758
    Epoch [13/50], Val Losses: mse: 42.0472, mae: 4.2271, huber: 3.7740, swd: 8.2498, ept: 164.2656
    Epoch [13/50], Test Losses: mse: 39.4554, mae: 4.0564, huber: 3.6044, swd: 7.5786, ept: 170.4368
      Epoch 13 composite train-obj: 3.798290
            No improvement (3.7740), counter 1/5
    Epoch [14/50], Train Losses: mse: 41.7328, mae: 4.2200, huber: 3.7685, swd: 6.8521, ept: 63.6792
    Epoch [14/50], Val Losses: mse: 37.3800, mae: 3.8352, huber: 3.3941, swd: 6.9072, ept: 182.8348
    Epoch [14/50], Test Losses: mse: 34.6212, mae: 3.6443, huber: 3.2051, swd: 6.6025, ept: 191.9613
      Epoch 14 composite train-obj: 3.768524
            Val objective improved 3.6161 → 3.3941, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 41.4292, mae: 4.1866, huber: 3.7363, swd: 6.8532, ept: 62.6504
    Epoch [15/50], Val Losses: mse: 39.2043, mae: 4.0093, huber: 3.5629, swd: 7.2606, ept: 175.1361
    Epoch [15/50], Test Losses: mse: 36.2114, mae: 3.8239, huber: 3.3797, swd: 7.3577, ept: 181.2804
      Epoch 15 composite train-obj: 3.736281
            No improvement (3.5629), counter 1/5
    Epoch [16/50], Train Losses: mse: 41.1768, mae: 4.1694, huber: 3.7192, swd: 6.6225, ept: 62.9069
    Epoch [16/50], Val Losses: mse: 43.2722, mae: 4.2714, huber: 3.8182, swd: 6.7072, ept: 162.4223
    Epoch [16/50], Test Losses: mse: 40.8030, mae: 4.1126, huber: 3.6615, swd: 6.5558, ept: 165.3726
      Epoch 16 composite train-obj: 3.719152
            No improvement (3.8182), counter 2/5
    Epoch [17/50], Train Losses: mse: 41.2452, mae: 4.1742, huber: 3.7238, swd: 6.5241, ept: 62.9930
    Epoch [17/50], Val Losses: mse: 39.1718, mae: 3.8924, huber: 3.4548, swd: 6.6657, ept: 180.3871
    Epoch [17/50], Test Losses: mse: 36.3035, mae: 3.7015, huber: 3.2649, swd: 6.4920, ept: 187.8218
      Epoch 17 composite train-obj: 3.723804
            No improvement (3.4548), counter 3/5
    Epoch [18/50], Train Losses: mse: 40.7909, mae: 4.1340, huber: 3.6846, swd: 6.4137, ept: 63.1382
    Epoch [18/50], Val Losses: mse: 40.4819, mae: 4.0011, huber: 3.5606, swd: 7.3615, ept: 175.0126
    Epoch [18/50], Test Losses: mse: 37.2389, mae: 3.7952, huber: 3.3573, swd: 6.9718, ept: 182.5969
      Epoch 18 composite train-obj: 3.684649
            No improvement (3.5606), counter 4/5
    Epoch [19/50], Train Losses: mse: 40.9037, mae: 4.1451, huber: 3.6955, swd: 6.5521, ept: 63.0246
    Epoch [19/50], Val Losses: mse: 36.6702, mae: 3.7899, huber: 3.3491, swd: 6.6015, ept: 181.0669
    Epoch [19/50], Test Losses: mse: 33.4966, mae: 3.5839, huber: 3.1449, swd: 6.3918, ept: 189.5400
      Epoch 19 composite train-obj: 3.695524
            Val objective improved 3.3941 → 3.3491, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 40.1050, mae: 4.0825, huber: 3.6341, swd: 6.1777, ept: 63.2463
    Epoch [20/50], Val Losses: mse: 39.3014, mae: 3.9649, huber: 3.5212, swd: 6.6638, ept: 181.4633
    Epoch [20/50], Test Losses: mse: 35.1923, mae: 3.7008, huber: 3.2600, swd: 6.3937, ept: 189.5993
      Epoch 20 composite train-obj: 3.634143
            No improvement (3.5212), counter 1/5
    Epoch [21/50], Train Losses: mse: 40.6748, mae: 4.1280, huber: 3.6786, swd: 6.2553, ept: 63.0864
    Epoch [21/50], Val Losses: mse: 41.7609, mae: 3.9657, huber: 3.5272, swd: 4.8537, ept: 181.8143
    Epoch [21/50], Test Losses: mse: 36.5108, mae: 3.6583, huber: 3.2226, swd: 4.4259, ept: 193.6798
      Epoch 21 composite train-obj: 3.678630
            No improvement (3.5272), counter 2/5
    Epoch [22/50], Train Losses: mse: 40.4178, mae: 4.1042, huber: 3.6553, swd: 6.3315, ept: 63.2027
    Epoch [22/50], Val Losses: mse: 38.6795, mae: 3.8224, huber: 3.3841, swd: 6.4009, ept: 182.4286
    Epoch [22/50], Test Losses: mse: 35.1495, mae: 3.6169, huber: 3.1802, swd: 6.3021, ept: 189.9673
      Epoch 22 composite train-obj: 3.655292
            No improvement (3.3841), counter 3/5
    Epoch [23/50], Train Losses: mse: 40.0021, mae: 4.0648, huber: 3.6171, swd: 6.0835, ept: 63.6200
    Epoch [23/50], Val Losses: mse: 36.8165, mae: 3.7222, huber: 3.2875, swd: 6.5933, ept: 188.8186
    Epoch [23/50], Test Losses: mse: 33.3183, mae: 3.5221, huber: 3.0876, swd: 6.3984, ept: 196.5827
      Epoch 23 composite train-obj: 3.617088
            Val objective improved 3.3491 → 3.2875, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 39.4190, mae: 4.0163, huber: 3.5696, swd: 5.8978, ept: 63.5140
    Epoch [24/50], Val Losses: mse: 39.3215, mae: 3.8531, huber: 3.4149, swd: 6.1836, ept: 180.4443
    Epoch [24/50], Test Losses: mse: 33.6149, mae: 3.5217, huber: 3.0872, swd: 5.6101, ept: 191.4279
      Epoch 24 composite train-obj: 3.569646
            No improvement (3.4149), counter 1/5
    Epoch [25/50], Train Losses: mse: 39.6386, mae: 4.0274, huber: 3.5805, swd: 5.8708, ept: 63.5039
    Epoch [25/50], Val Losses: mse: 35.8642, mae: 3.6698, huber: 3.2311, swd: 6.0280, ept: 191.7256
    Epoch [25/50], Test Losses: mse: 32.2632, mae: 3.4570, huber: 3.0195, swd: 5.8313, ept: 199.4207
      Epoch 25 composite train-obj: 3.580514
            Val objective improved 3.2875 → 3.2311, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 39.3384, mae: 4.0185, huber: 3.5716, swd: 5.9223, ept: 63.3526
    Epoch [26/50], Val Losses: mse: 36.4292, mae: 3.6473, huber: 3.2155, swd: 5.7297, ept: 191.9180
    Epoch [26/50], Test Losses: mse: 32.6328, mae: 3.4357, huber: 3.0048, swd: 5.4578, ept: 200.6837
      Epoch 26 composite train-obj: 3.571584
            Val objective improved 3.2311 → 3.2155, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 38.7297, mae: 3.9591, huber: 3.5136, swd: 5.5909, ept: 64.3296
    Epoch [27/50], Val Losses: mse: 40.4225, mae: 3.9143, huber: 3.4793, swd: 5.6116, ept: 179.8058
    Epoch [27/50], Test Losses: mse: 37.3244, mae: 3.7702, huber: 3.3337, swd: 5.3045, ept: 185.2582
      Epoch 27 composite train-obj: 3.513615
            No improvement (3.4793), counter 1/5
    Epoch [28/50], Train Losses: mse: 38.7819, mae: 3.9650, huber: 3.5193, swd: 5.6076, ept: 63.5562
    Epoch [28/50], Val Losses: mse: 36.7656, mae: 3.7214, huber: 3.2823, swd: 5.1428, ept: 187.6662
    Epoch [28/50], Test Losses: mse: 33.3106, mae: 3.4970, huber: 3.0611, swd: 4.6502, ept: 193.8702
      Epoch 28 composite train-obj: 3.519296
            No improvement (3.2823), counter 2/5
    Epoch [29/50], Train Losses: mse: 38.6602, mae: 3.9566, huber: 3.5111, swd: 5.6212, ept: 63.7527
    Epoch [29/50], Val Losses: mse: 39.3902, mae: 3.8505, huber: 3.4114, swd: 5.7338, ept: 175.1358
    Epoch [29/50], Test Losses: mse: 37.8760, mae: 3.7746, huber: 3.3325, swd: 5.5719, ept: 177.2167
      Epoch 29 composite train-obj: 3.511109
            No improvement (3.4114), counter 3/5
    Epoch [30/50], Train Losses: mse: 39.3211, mae: 4.0092, huber: 3.5624, swd: 5.7675, ept: 63.0693
    Epoch [30/50], Val Losses: mse: 36.8574, mae: 3.7473, huber: 3.3103, swd: 6.6012, ept: 190.1502
    Epoch [30/50], Test Losses: mse: 33.8205, mae: 3.5701, huber: 3.1342, swd: 6.2980, ept: 199.6540
      Epoch 30 composite train-obj: 3.562432
            No improvement (3.3103), counter 4/5
    Epoch [31/50], Train Losses: mse: 38.7202, mae: 3.9676, huber: 3.5215, swd: 5.6684, ept: 63.6795
    Epoch [31/50], Val Losses: mse: 37.0992, mae: 3.6586, huber: 3.2255, swd: 5.2488, ept: 194.7556
    Epoch [31/50], Test Losses: mse: 33.9642, mae: 3.4838, huber: 3.0533, swd: 5.3245, ept: 202.1205
      Epoch 31 composite train-obj: 3.521451
    Epoch [31/50], Test Losses: mse: 32.6328, mae: 3.4357, huber: 3.0048, swd: 5.4578, ept: 200.6837
    Best round's Test MSE: 32.6328, MAE: 3.4357, SWD: 5.4578
    Best round's Validation MSE: 36.4292, MAE: 3.6473, SWD: 5.7297
    Best round's Test verification MSE : 32.6328, MAE: 3.4357, SWD: 5.4578
    Time taken: 116.07 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 62.1934, mae: 5.7982, huber: 5.3229, swd: 13.1008, ept: 48.7211
    Epoch [1/50], Val Losses: mse: 58.5362, mae: 5.5796, huber: 5.1043, swd: 14.7960, ept: 88.2722
    Epoch [1/50], Test Losses: mse: 56.8301, mae: 5.4470, huber: 4.9727, swd: 14.6278, ept: 93.0689
      Epoch 1 composite train-obj: 5.322910
            Val objective improved inf → 5.1043, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 52.9068, mae: 5.1814, huber: 4.7129, swd: 11.7967, ept: 56.5590
    Epoch [2/50], Val Losses: mse: 54.0314, mae: 5.3341, huber: 4.8635, swd: 13.0363, ept: 98.6712
    Epoch [2/50], Test Losses: mse: 52.1472, mae: 5.2011, huber: 4.7309, swd: 12.2910, ept: 101.9908
      Epoch 2 composite train-obj: 4.712882
            Val objective improved 5.1043 → 4.8635, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.9814, mae: 5.0046, huber: 4.5388, swd: 10.6059, ept: 58.1312
    Epoch [3/50], Val Losses: mse: 47.9509, mae: 4.7861, huber: 4.3226, swd: 11.8994, ept: 130.6572
    Epoch [3/50], Test Losses: mse: 46.4771, mae: 4.6421, huber: 4.1798, swd: 10.8691, ept: 134.4046
      Epoch 3 composite train-obj: 4.538780
            Val objective improved 4.8635 → 4.3226, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 48.4638, mae: 4.7766, huber: 4.3147, swd: 9.7300, ept: 60.2556
    Epoch [4/50], Val Losses: mse: 47.3960, mae: 4.6978, huber: 4.2353, swd: 11.7233, ept: 137.1257
    Epoch [4/50], Test Losses: mse: 45.5057, mae: 4.5660, huber: 4.1043, swd: 10.6477, ept: 140.9779
      Epoch 4 composite train-obj: 4.314668
            Val objective improved 4.3226 → 4.2353, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 47.1022, mae: 4.6614, huber: 4.2015, swd: 9.1356, ept: 61.1007
    Epoch [5/50], Val Losses: mse: 46.0098, mae: 4.6554, huber: 4.1948, swd: 10.6452, ept: 134.8967
    Epoch [5/50], Test Losses: mse: 44.1627, mae: 4.5198, huber: 4.0596, swd: 10.5668, ept: 137.9732
      Epoch 5 composite train-obj: 4.201544
            Val objective improved 4.2353 → 4.1948, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 46.2314, mae: 4.5789, huber: 4.1206, swd: 8.6682, ept: 61.1255
    Epoch [6/50], Val Losses: mse: 45.6910, mae: 4.5800, huber: 4.1215, swd: 10.3405, ept: 142.1505
    Epoch [6/50], Test Losses: mse: 43.3142, mae: 4.4198, huber: 3.9626, swd: 9.4507, ept: 150.6398
      Epoch 6 composite train-obj: 4.120637
            Val objective improved 4.1948 → 4.1215, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 45.9329, mae: 4.5725, huber: 4.1142, swd: 8.7559, ept: 61.5534
    Epoch [7/50], Val Losses: mse: 43.3435, mae: 4.3834, huber: 3.9281, swd: 9.5154, ept: 152.5790
    Epoch [7/50], Test Losses: mse: 42.4497, mae: 4.3032, huber: 3.8487, swd: 9.3496, ept: 155.3333
      Epoch 7 composite train-obj: 4.114182
            Val objective improved 4.1215 → 3.9281, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 44.6770, mae: 4.4714, huber: 4.0148, swd: 8.2383, ept: 61.8329
    Epoch [8/50], Val Losses: mse: 42.0145, mae: 4.3387, huber: 3.8807, swd: 8.1210, ept: 152.9035
    Epoch [8/50], Test Losses: mse: 39.6376, mae: 4.1770, huber: 3.7202, swd: 8.1716, ept: 158.0154
      Epoch 8 composite train-obj: 4.014761
            Val objective improved 3.9281 → 3.8807, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 44.2955, mae: 4.4287, huber: 3.9731, swd: 7.8442, ept: 62.1629
    Epoch [9/50], Val Losses: mse: 51.5597, mae: 4.8850, huber: 4.4250, swd: 9.5869, ept: 131.5299
    Epoch [9/50], Test Losses: mse: 49.6829, mae: 4.7663, huber: 4.3056, swd: 9.3189, ept: 132.7097
      Epoch 9 composite train-obj: 3.973133
            No improvement (4.4250), counter 1/5
    Epoch [10/50], Train Losses: mse: 45.7230, mae: 4.5478, huber: 4.0902, swd: 8.4375, ept: 61.8120
    Epoch [10/50], Val Losses: mse: 49.2255, mae: 4.6687, huber: 4.2148, swd: 8.2586, ept: 144.9874
    Epoch [10/50], Test Losses: mse: 45.7571, mae: 4.4583, huber: 4.0050, swd: 7.8244, ept: 149.8833
      Epoch 10 composite train-obj: 4.090197
            No improvement (4.2148), counter 2/5
    Epoch [11/50], Train Losses: mse: 43.4134, mae: 4.3565, huber: 3.9023, swd: 7.5561, ept: 62.1742
    Epoch [11/50], Val Losses: mse: 44.4508, mae: 4.2755, huber: 3.8299, swd: 9.4195, ept: 161.4464
    Epoch [11/50], Test Losses: mse: 41.5536, mae: 4.0818, huber: 3.6381, swd: 8.7723, ept: 169.1846
      Epoch 11 composite train-obj: 3.902284
            Val objective improved 3.8807 → 3.8299, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 43.3411, mae: 4.3546, huber: 3.9003, swd: 7.6214, ept: 62.2717
    Epoch [12/50], Val Losses: mse: 41.8502, mae: 4.1479, huber: 3.6995, swd: 7.6124, ept: 167.2698
    Epoch [12/50], Test Losses: mse: 39.5729, mae: 3.9985, huber: 3.5512, swd: 7.2809, ept: 170.9403
      Epoch 12 composite train-obj: 3.900256
            Val objective improved 3.8299 → 3.6995, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 42.4285, mae: 4.2692, huber: 3.8168, swd: 7.0865, ept: 62.5970
    Epoch [13/50], Val Losses: mse: 39.7253, mae: 3.9919, huber: 3.5491, swd: 7.4294, ept: 173.1804
    Epoch [13/50], Test Losses: mse: 36.9357, mae: 3.8226, huber: 3.3810, swd: 7.3518, ept: 181.0371
      Epoch 13 composite train-obj: 3.816764
            Val objective improved 3.6995 → 3.5491, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 41.8239, mae: 4.2228, huber: 3.7710, swd: 6.8826, ept: 63.6280
    Epoch [14/50], Val Losses: mse: 42.1784, mae: 4.1079, huber: 3.6688, swd: 6.7866, ept: 171.5797
    Epoch [14/50], Test Losses: mse: 38.6870, mae: 3.9097, huber: 3.4719, swd: 6.5608, ept: 177.9596
      Epoch 14 composite train-obj: 3.770968
            No improvement (3.6688), counter 1/5
    Epoch [15/50], Train Losses: mse: 41.5300, mae: 4.1957, huber: 3.7447, swd: 6.7376, ept: 62.8623
    Epoch [15/50], Val Losses: mse: 41.4244, mae: 4.1374, huber: 3.6891, swd: 7.6753, ept: 166.4689
    Epoch [15/50], Test Losses: mse: 37.5645, mae: 3.8853, huber: 3.4401, swd: 7.2055, ept: 175.6260
      Epoch 15 composite train-obj: 3.744692
            No improvement (3.6891), counter 2/5
    Epoch [16/50], Train Losses: mse: 41.4798, mae: 4.1893, huber: 3.7386, swd: 6.6565, ept: 63.0518
    Epoch [16/50], Val Losses: mse: 40.3327, mae: 3.9760, huber: 3.5325, swd: 6.9231, ept: 175.9493
    Epoch [16/50], Test Losses: mse: 36.8821, mae: 3.7732, huber: 3.3310, swd: 6.5093, ept: 182.9089
      Epoch 16 composite train-obj: 3.738571
            Val objective improved 3.5491 → 3.5325, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 41.0030, mae: 4.1509, huber: 3.7006, swd: 6.4615, ept: 63.0324
    Epoch [17/50], Val Losses: mse: 40.9589, mae: 3.9951, huber: 3.5524, swd: 6.8082, ept: 177.1915
    Epoch [17/50], Test Losses: mse: 37.8570, mae: 3.7952, huber: 3.3547, swd: 6.3584, ept: 181.6336
      Epoch 17 composite train-obj: 3.700641
            No improvement (3.5524), counter 1/5
    Epoch [18/50], Train Losses: mse: 41.1636, mae: 4.1675, huber: 3.7168, swd: 6.6302, ept: 63.2604
    Epoch [18/50], Val Losses: mse: 38.1317, mae: 3.8197, huber: 3.3817, swd: 6.8004, ept: 181.1215
    Epoch [18/50], Test Losses: mse: 36.1452, mae: 3.6867, huber: 3.2486, swd: 6.5527, ept: 189.0789
      Epoch 18 composite train-obj: 3.716831
            Val objective improved 3.5325 → 3.3817, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 40.7076, mae: 4.1233, huber: 3.6739, swd: 6.3039, ept: 63.2329
    Epoch [19/50], Val Losses: mse: 39.7128, mae: 3.9178, huber: 3.4801, swd: 6.3751, ept: 182.6562
    Epoch [19/50], Test Losses: mse: 34.4780, mae: 3.5808, huber: 3.1475, swd: 6.0183, ept: 195.3741
      Epoch 19 composite train-obj: 3.673912
            No improvement (3.4801), counter 1/5
    Epoch [20/50], Train Losses: mse: 40.3449, mae: 4.0934, huber: 3.6445, swd: 6.1580, ept: 63.2731
    Epoch [20/50], Val Losses: mse: 38.9996, mae: 3.8500, huber: 3.4105, swd: 6.1122, ept: 188.0539
    Epoch [20/50], Test Losses: mse: 34.9421, mae: 3.6301, huber: 3.1923, swd: 5.9050, ept: 194.9445
      Epoch 20 composite train-obj: 3.644466
            No improvement (3.4105), counter 2/5
    Epoch [21/50], Train Losses: mse: 40.2674, mae: 4.0994, huber: 3.6501, swd: 6.2370, ept: 63.4524
    Epoch [21/50], Val Losses: mse: 36.9552, mae: 3.8078, huber: 3.3660, swd: 5.7770, ept: 187.8117
    Epoch [21/50], Test Losses: mse: 33.4484, mae: 3.5890, huber: 3.1497, swd: 5.5626, ept: 195.2680
      Epoch 21 composite train-obj: 3.650115
            Val objective improved 3.3817 → 3.3660, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 39.9688, mae: 4.0758, huber: 3.6269, swd: 6.0791, ept: 63.9684
    Epoch [22/50], Val Losses: mse: 41.5613, mae: 4.1018, huber: 3.6579, swd: 6.3031, ept: 173.0407
    Epoch [22/50], Test Losses: mse: 38.8951, mae: 3.9518, huber: 3.5079, swd: 5.9260, ept: 179.8145
      Epoch 22 composite train-obj: 3.626892
            No improvement (3.6579), counter 1/5
    Epoch [23/50], Train Losses: mse: 39.7629, mae: 4.0497, huber: 3.6017, swd: 5.9531, ept: 63.4248
    Epoch [23/50], Val Losses: mse: 36.3769, mae: 3.6803, huber: 3.2436, swd: 5.7413, ept: 191.5344
    Epoch [23/50], Test Losses: mse: 33.0540, mae: 3.4780, huber: 3.0431, swd: 5.3272, ept: 198.8881
      Epoch 23 composite train-obj: 3.601681
            Val objective improved 3.3660 → 3.2436, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 39.4231, mae: 4.0181, huber: 3.5707, swd: 5.8610, ept: 63.6653
    Epoch [24/50], Val Losses: mse: 36.4314, mae: 3.7210, huber: 3.2831, swd: 6.1827, ept: 189.9760
    Epoch [24/50], Test Losses: mse: 32.5756, mae: 3.4906, huber: 3.0560, swd: 5.9357, ept: 201.0036
      Epoch 24 composite train-obj: 3.570679
            No improvement (3.2831), counter 1/5
    Epoch [25/50], Train Losses: mse: 39.3838, mae: 4.0163, huber: 3.5689, swd: 5.8679, ept: 63.5065
    Epoch [25/50], Val Losses: mse: 36.7819, mae: 3.6886, huber: 3.2508, swd: 5.7497, ept: 190.9400
    Epoch [25/50], Test Losses: mse: 33.9411, mae: 3.5180, huber: 3.0821, swd: 5.4858, ept: 198.7929
      Epoch 25 composite train-obj: 3.568948
            No improvement (3.2508), counter 2/5
    Epoch [26/50], Train Losses: mse: 39.4461, mae: 4.0172, huber: 3.5701, swd: 5.8356, ept: 63.5998
    Epoch [26/50], Val Losses: mse: 41.9979, mae: 4.1010, huber: 3.6535, swd: 5.5902, ept: 173.7966
    Epoch [26/50], Test Losses: mse: 38.7048, mae: 3.8971, huber: 3.4522, swd: 5.5587, ept: 183.6333
      Epoch 26 composite train-obj: 3.570095
            No improvement (3.6535), counter 3/5
    Epoch [27/50], Train Losses: mse: 39.4415, mae: 4.0202, huber: 3.5726, swd: 5.8174, ept: 63.5173
    Epoch [27/50], Val Losses: mse: 34.5507, mae: 3.5578, huber: 3.1250, swd: 5.9733, ept: 196.3840
    Epoch [27/50], Test Losses: mse: 31.3367, mae: 3.3602, huber: 2.9293, swd: 5.5453, ept: 203.1581
      Epoch 27 composite train-obj: 3.572616
            Val objective improved 3.2436 → 3.1250, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 38.5849, mae: 3.9467, huber: 3.5010, swd: 5.5992, ept: 63.8306
    Epoch [28/50], Val Losses: mse: 38.9031, mae: 3.7850, huber: 3.3484, swd: 5.6314, ept: 193.9786
    Epoch [28/50], Test Losses: mse: 35.2922, mae: 3.5659, huber: 3.1329, swd: 5.5424, ept: 202.7978
      Epoch 28 composite train-obj: 3.501011
            No improvement (3.3484), counter 1/5
    Epoch [29/50], Train Losses: mse: 38.8361, mae: 3.9625, huber: 3.5166, swd: 5.6184, ept: 64.0472
    Epoch [29/50], Val Losses: mse: 37.0806, mae: 3.7176, huber: 3.2834, swd: 5.6808, ept: 189.4806
    Epoch [29/50], Test Losses: mse: 33.8856, mae: 3.5370, huber: 3.1040, swd: 5.1472, ept: 196.3210
      Epoch 29 composite train-obj: 3.516582
            No improvement (3.2834), counter 2/5
    Epoch [30/50], Train Losses: mse: 38.7044, mae: 3.9519, huber: 3.5061, swd: 5.5116, ept: 63.7174
    Epoch [30/50], Val Losses: mse: 36.0555, mae: 3.5958, huber: 3.1654, swd: 5.4072, ept: 199.2362
    Epoch [30/50], Test Losses: mse: 32.4916, mae: 3.4008, huber: 2.9720, swd: 5.0958, ept: 202.8730
      Epoch 30 composite train-obj: 3.506075
            No improvement (3.1654), counter 3/5
    Epoch [31/50], Train Losses: mse: 38.7083, mae: 3.9548, huber: 3.5091, swd: 5.5757, ept: 63.5992
    Epoch [31/50], Val Losses: mse: 34.3871, mae: 3.5676, huber: 3.1289, swd: 4.8848, ept: 201.8628
    Epoch [31/50], Test Losses: mse: 30.8640, mae: 3.3630, huber: 2.9257, swd: 4.5429, ept: 211.2470
      Epoch 31 composite train-obj: 3.509069
            No improvement (3.1289), counter 4/5
    Epoch [32/50], Train Losses: mse: 38.2543, mae: 3.9240, huber: 3.4786, swd: 5.4819, ept: 63.9601
    Epoch [32/50], Val Losses: mse: 34.8580, mae: 3.5765, huber: 3.1453, swd: 5.4885, ept: 196.0432
    Epoch [32/50], Test Losses: mse: 31.8166, mae: 3.4001, huber: 2.9711, swd: 5.2446, ept: 201.9682
      Epoch 32 composite train-obj: 3.478630
    Epoch [32/50], Test Losses: mse: 31.3367, mae: 3.3602, huber: 2.9293, swd: 5.5453, ept: 203.1581
    Best round's Test MSE: 31.3367, MAE: 3.3602, SWD: 5.5453
    Best round's Validation MSE: 34.5507, MAE: 3.5578, SWD: 5.9733
    Best round's Test verification MSE : 31.3367, MAE: 3.3602, SWD: 5.5453
    Time taken: 121.82 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 61.7467, mae: 5.7884, huber: 5.3131, swd: 13.3679, ept: 48.8789
    Epoch [1/50], Val Losses: mse: 61.0789, mae: 5.7485, huber: 5.2730, swd: 15.4620, ept: 80.0675
    Epoch [1/50], Test Losses: mse: 59.1350, mae: 5.6174, huber: 5.1427, swd: 15.9119, ept: 76.9997
      Epoch 1 composite train-obj: 5.313063
            Val objective improved inf → 5.2730, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 53.3081, mae: 5.2119, huber: 4.7430, swd: 12.6232, ept: 55.7230
    Epoch [2/50], Val Losses: mse: 53.5817, mae: 5.2857, huber: 4.8144, swd: 14.9975, ept: 111.3049
    Epoch [2/50], Test Losses: mse: 50.7324, mae: 5.0935, huber: 4.6233, swd: 14.1734, ept: 114.6049
      Epoch 2 composite train-obj: 4.742968
            Val objective improved 5.2730 → 4.8144, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.1312, mae: 4.9397, huber: 4.4747, swd: 11.2622, ept: 58.5809
    Epoch [3/50], Val Losses: mse: 50.1771, mae: 5.0505, huber: 4.5832, swd: 14.1737, ept: 108.9367
    Epoch [3/50], Test Losses: mse: 47.5955, mae: 4.8741, huber: 4.4085, swd: 13.3928, ept: 112.8476
      Epoch 3 composite train-obj: 4.474724
            Val objective improved 4.8144 → 4.5832, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 48.7632, mae: 4.8255, huber: 4.3624, swd: 10.7300, ept: 60.0318
    Epoch [4/50], Val Losses: mse: 50.9358, mae: 4.8931, huber: 4.4315, swd: 11.8772, ept: 133.3271
    Epoch [4/50], Test Losses: mse: 48.2413, mae: 4.7245, huber: 4.2636, swd: 10.9945, ept: 140.5720
      Epoch 4 composite train-obj: 4.362390
            Val objective improved 4.5832 → 4.4315, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 47.1113, mae: 4.6774, huber: 4.2170, swd: 9.7348, ept: 60.3336
    Epoch [5/50], Val Losses: mse: 47.0904, mae: 4.6939, huber: 4.2330, swd: 11.7045, ept: 138.2985
    Epoch [5/50], Test Losses: mse: 45.1531, mae: 4.5557, huber: 4.0971, swd: 11.1635, ept: 143.3303
      Epoch 5 composite train-obj: 4.216979
            Val objective improved 4.4315 → 4.2330, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 46.2841, mae: 4.6203, huber: 4.1605, swd: 9.2409, ept: 60.5961
    Epoch [6/50], Val Losses: mse: 43.2124, mae: 4.3840, huber: 3.9273, swd: 10.3837, ept: 155.8959
    Epoch [6/50], Test Losses: mse: 40.5274, mae: 4.2008, huber: 3.7461, swd: 9.7347, ept: 161.1113
      Epoch 6 composite train-obj: 4.160484
            Val objective improved 4.2330 → 3.9273, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 45.1937, mae: 4.5249, huber: 4.0668, swd: 8.5883, ept: 60.9586
    Epoch [7/50], Val Losses: mse: 41.2994, mae: 4.2371, huber: 3.7838, swd: 10.7213, ept: 152.2891
    Epoch [7/50], Test Losses: mse: 38.9581, mae: 4.0724, huber: 3.6218, swd: 10.6190, ept: 158.6454
      Epoch 7 composite train-obj: 4.066834
            Val objective improved 3.9273 → 3.7838, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 44.5881, mae: 4.4631, huber: 4.0064, swd: 8.2401, ept: 61.6231
    Epoch [8/50], Val Losses: mse: 42.5110, mae: 4.2551, huber: 3.8035, swd: 9.4739, ept: 157.3812
    Epoch [8/50], Test Losses: mse: 40.7260, mae: 4.1225, huber: 3.6717, swd: 9.1140, ept: 163.3913
      Epoch 8 composite train-obj: 4.006370
            No improvement (3.8035), counter 1/5
    Epoch [9/50], Train Losses: mse: 43.6470, mae: 4.3805, huber: 3.9251, swd: 7.8467, ept: 61.6107
    Epoch [9/50], Val Losses: mse: 40.2165, mae: 4.0788, huber: 3.6306, swd: 8.2628, ept: 164.6049
    Epoch [9/50], Test Losses: mse: 37.7250, mae: 3.9195, huber: 3.4730, swd: 7.9383, ept: 170.4706
      Epoch 9 composite train-obj: 3.925078
            Val objective improved 3.7838 → 3.6306, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 42.8595, mae: 4.3214, huber: 3.8670, swd: 7.4917, ept: 62.7427
    Epoch [10/50], Val Losses: mse: 40.5501, mae: 4.0650, huber: 3.6209, swd: 8.1224, ept: 172.1993
    Epoch [10/50], Test Losses: mse: 37.1342, mae: 3.8274, huber: 3.3872, swd: 7.9257, ept: 178.9422
      Epoch 10 composite train-obj: 3.866972
            Val objective improved 3.6306 → 3.6209, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 42.4015, mae: 4.2733, huber: 3.8203, swd: 7.2797, ept: 62.1180
    Epoch [11/50], Val Losses: mse: 39.6357, mae: 4.0257, huber: 3.5773, swd: 7.7019, ept: 174.9903
    Epoch [11/50], Test Losses: mse: 36.7531, mae: 3.8278, huber: 3.3817, swd: 7.4073, ept: 180.4452
      Epoch 11 composite train-obj: 3.820270
            Val objective improved 3.6209 → 3.5773, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 42.4865, mae: 4.2918, huber: 3.8380, swd: 7.3241, ept: 62.8269
    Epoch [12/50], Val Losses: mse: 39.5571, mae: 3.9195, huber: 3.4811, swd: 7.4073, ept: 177.5465
    Epoch [12/50], Test Losses: mse: 36.1275, mae: 3.7058, huber: 3.2692, swd: 7.1468, ept: 184.4978
      Epoch 12 composite train-obj: 3.837969
            Val objective improved 3.5773 → 3.4811, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 41.8906, mae: 4.2225, huber: 3.7707, swd: 7.0079, ept: 62.3039
    Epoch [13/50], Val Losses: mse: 43.3874, mae: 4.1734, huber: 3.7278, swd: 8.0537, ept: 168.9902
    Epoch [13/50], Test Losses: mse: 40.1371, mae: 3.9757, huber: 3.5331, swd: 7.4984, ept: 180.1906
      Epoch 13 composite train-obj: 3.770688
            No improvement (3.7278), counter 1/5
    Epoch [14/50], Train Losses: mse: 42.0824, mae: 4.2429, huber: 3.7907, swd: 7.2027, ept: 62.0473
    Epoch [14/50], Val Losses: mse: 38.1284, mae: 3.8855, huber: 3.4449, swd: 8.2161, ept: 178.8557
    Epoch [14/50], Test Losses: mse: 35.2754, mae: 3.7066, huber: 3.2685, swd: 7.8941, ept: 186.2558
      Epoch 14 composite train-obj: 3.790748
            Val objective improved 3.4811 → 3.4449, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 41.2953, mae: 4.1859, huber: 3.7343, swd: 6.8506, ept: 63.6230
    Epoch [15/50], Val Losses: mse: 36.6107, mae: 3.7344, huber: 3.2993, swd: 7.2888, ept: 184.1614
    Epoch [15/50], Test Losses: mse: 33.8001, mae: 3.5576, huber: 3.1243, swd: 7.0460, ept: 191.7223
      Epoch 15 composite train-obj: 3.734315
            Val objective improved 3.4449 → 3.2993, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 40.6504, mae: 4.1203, huber: 3.6706, swd: 6.6310, ept: 62.7280
    Epoch [16/50], Val Losses: mse: 37.4149, mae: 3.7512, huber: 3.3134, swd: 6.2090, ept: 186.7659
    Epoch [16/50], Test Losses: mse: 34.2878, mae: 3.5702, huber: 3.1335, swd: 5.7401, ept: 193.9579
      Epoch 16 composite train-obj: 3.670610
            No improvement (3.3134), counter 1/5
    Epoch [17/50], Train Losses: mse: 40.5933, mae: 4.1150, huber: 3.6654, swd: 6.5433, ept: 62.7376
    Epoch [17/50], Val Losses: mse: 43.0857, mae: 4.1966, huber: 3.7503, swd: 6.8018, ept: 164.2767
    Epoch [17/50], Test Losses: mse: 39.6380, mae: 3.9877, huber: 3.5424, swd: 6.5428, ept: 168.8514
      Epoch 17 composite train-obj: 3.665413
            No improvement (3.7503), counter 2/5
    Epoch [18/50], Train Losses: mse: 41.1391, mae: 4.1635, huber: 3.7130, swd: 6.8602, ept: 63.0149
    Epoch [18/50], Val Losses: mse: 39.4617, mae: 3.8901, huber: 3.4531, swd: 5.9588, ept: 181.8226
    Epoch [18/50], Test Losses: mse: 36.0941, mae: 3.6904, huber: 3.2552, swd: 5.4474, ept: 188.4689
      Epoch 18 composite train-obj: 3.712997
            No improvement (3.4531), counter 3/5
    Epoch [19/50], Train Losses: mse: 40.4215, mae: 4.1021, huber: 3.6529, swd: 6.4853, ept: 62.7479
    Epoch [19/50], Val Losses: mse: 34.7332, mae: 3.6103, huber: 3.1762, swd: 6.4754, ept: 194.1205
    Epoch [19/50], Test Losses: mse: 32.5173, mae: 3.4864, huber: 3.0527, swd: 6.2705, ept: 200.7325
      Epoch 19 composite train-obj: 3.652872
            Val objective improved 3.2993 → 3.1762, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 39.8977, mae: 4.0622, huber: 3.6135, swd: 6.3688, ept: 63.2412
    Epoch [20/50], Val Losses: mse: 38.7544, mae: 3.8180, huber: 3.3814, swd: 6.5491, ept: 185.4075
    Epoch [20/50], Test Losses: mse: 34.9693, mae: 3.5829, huber: 3.1498, swd: 6.2223, ept: 194.6773
      Epoch 20 composite train-obj: 3.613533
            No improvement (3.3814), counter 1/5
    Epoch [21/50], Train Losses: mse: 39.5085, mae: 4.0260, huber: 3.5783, swd: 6.1815, ept: 64.1308
    Epoch [21/50], Val Losses: mse: 45.1084, mae: 4.2648, huber: 3.8213, swd: 6.7515, ept: 170.1147
    Epoch [21/50], Test Losses: mse: 41.9924, mae: 4.1143, huber: 3.6700, swd: 6.2051, ept: 174.5787
      Epoch 21 composite train-obj: 3.578270
            No improvement (3.8213), counter 2/5
    Epoch [22/50], Train Losses: mse: 39.9202, mae: 4.0575, huber: 3.6092, swd: 6.2704, ept: 63.1356
    Epoch [22/50], Val Losses: mse: 39.0516, mae: 3.8335, huber: 3.3940, swd: 5.6351, ept: 188.5406
    Epoch [22/50], Test Losses: mse: 35.0895, mae: 3.6074, huber: 3.1690, swd: 5.4238, ept: 195.6937
      Epoch 22 composite train-obj: 3.609228
            No improvement (3.3940), counter 3/5
    Epoch [23/50], Train Losses: mse: 39.5751, mae: 4.0307, huber: 3.5828, swd: 6.2005, ept: 63.2661
    Epoch [23/50], Val Losses: mse: 40.3526, mae: 4.0260, huber: 3.5804, swd: 7.2543, ept: 171.4713
    Epoch [23/50], Test Losses: mse: 37.6101, mae: 3.8758, huber: 3.4307, swd: 7.3292, ept: 178.9487
      Epoch 23 composite train-obj: 3.582815
            No improvement (3.5804), counter 4/5
    Epoch [24/50], Train Losses: mse: 39.3422, mae: 4.0123, huber: 3.5649, swd: 6.0844, ept: 63.1336
    Epoch [24/50], Val Losses: mse: 35.1688, mae: 3.5787, huber: 3.1468, swd: 5.8392, ept: 198.1865
    Epoch [24/50], Test Losses: mse: 31.9200, mae: 3.3896, huber: 2.9599, swd: 5.6118, ept: 204.0494
      Epoch 24 composite train-obj: 3.564876
            Val objective improved 3.1762 → 3.1468, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 39.1964, mae: 4.0002, huber: 3.5530, swd: 6.0973, ept: 63.5085
    Epoch [25/50], Val Losses: mse: 37.9057, mae: 3.7682, huber: 3.3304, swd: 5.8122, ept: 189.4610
    Epoch [25/50], Test Losses: mse: 37.0118, mae: 3.7334, huber: 3.2934, swd: 5.5751, ept: 193.3514
      Epoch 25 composite train-obj: 3.553035
            No improvement (3.3304), counter 1/5
    Epoch [26/50], Train Losses: mse: 38.8363, mae: 3.9722, huber: 3.5255, swd: 5.9778, ept: 64.2934
    Epoch [26/50], Val Losses: mse: 34.0785, mae: 3.4986, huber: 3.0679, swd: 6.2298, ept: 199.2607
    Epoch [26/50], Test Losses: mse: 30.8283, mae: 3.3180, huber: 2.8876, swd: 6.0556, ept: 206.2601
      Epoch 26 composite train-obj: 3.525528
            Val objective improved 3.1468 → 3.0679, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 38.3869, mae: 3.9329, huber: 3.4875, swd: 5.8704, ept: 63.4843
    Epoch [27/50], Val Losses: mse: 35.9254, mae: 3.6426, huber: 3.2058, swd: 6.5005, ept: 192.3564
    Epoch [27/50], Test Losses: mse: 32.1760, mae: 3.4289, huber: 2.9928, swd: 6.2223, ept: 198.4404
      Epoch 27 composite train-obj: 3.487527
            No improvement (3.2058), counter 1/5
    Epoch [28/50], Train Losses: mse: 38.7339, mae: 3.9574, huber: 3.5112, swd: 5.8747, ept: 63.6179
    Epoch [28/50], Val Losses: mse: 43.5975, mae: 4.1141, huber: 3.6694, swd: 6.4719, ept: 180.7685
    Epoch [28/50], Test Losses: mse: 36.2178, mae: 3.6898, huber: 3.2500, swd: 6.5060, ept: 193.8229
      Epoch 28 composite train-obj: 3.511193
            No improvement (3.6694), counter 2/5
    Epoch [29/50], Train Losses: mse: 40.2207, mae: 4.0852, huber: 3.6363, swd: 6.6406, ept: 62.5363
    Epoch [29/50], Val Losses: mse: 35.0400, mae: 3.5771, huber: 3.1422, swd: 5.6619, ept: 196.9010
    Epoch [29/50], Test Losses: mse: 31.4396, mae: 3.3556, huber: 2.9237, swd: 5.3938, ept: 205.3739
      Epoch 29 composite train-obj: 3.636343
            No improvement (3.1422), counter 3/5
    Epoch [30/50], Train Losses: mse: 38.5756, mae: 3.9512, huber: 3.5050, swd: 5.9531, ept: 63.5706
    Epoch [30/50], Val Losses: mse: 38.7175, mae: 3.7806, huber: 3.3425, swd: 6.2160, ept: 194.6313
    Epoch [30/50], Test Losses: mse: 34.4418, mae: 3.5640, huber: 3.1279, swd: 6.0634, ept: 201.8058
      Epoch 30 composite train-obj: 3.504999
            No improvement (3.3425), counter 4/5
    Epoch [31/50], Train Losses: mse: 38.1735, mae: 3.9169, huber: 3.4714, swd: 5.7776, ept: 63.6889
    Epoch [31/50], Val Losses: mse: 47.3862, mae: 4.4257, huber: 3.9780, swd: 6.6865, ept: 155.4758
    Epoch [31/50], Test Losses: mse: 45.8315, mae: 4.3503, huber: 3.9015, swd: 6.5293, ept: 156.8076
      Epoch 31 composite train-obj: 3.471401
    Epoch [31/50], Test Losses: mse: 30.8283, mae: 3.3180, huber: 2.8876, swd: 6.0556, ept: 206.2601
    Best round's Test MSE: 30.8283, MAE: 3.3180, SWD: 6.0556
    Best round's Validation MSE: 34.0785, MAE: 3.4986, SWD: 6.2298
    Best round's Test verification MSE : 30.8283, MAE: 3.3180, SWD: 6.0556
    Time taken: 123.45 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq96_pred336_20250514_0120)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 31.5992 ± 0.7597
      mae: 3.3713 ± 0.0487
      huber: 2.9406 ± 0.0485
      swd: 5.6862 ± 0.2636
      ept: 203.3673 ± 2.2813
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 35.0195 ± 1.0153
      mae: 3.5679 ± 0.0611
      huber: 3.1361 ± 0.0608
      swd: 5.9776 ± 0.2042
      ept: 195.8542 ± 3.0210
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 361.40 seconds
    
    Experiment complete: PatchTST_lorenz_seq96_pred336_20250514_0120
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred 720
##### huber


```python
utils.reload_modules([utils])
cfg_patch_tst = train_config.FlatPatchTSTConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 279
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 279
    Validation Batches: 35
    Test Batches: 75
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.2201, mae: 6.3624, huber: 5.8832, swd: 16.4291, ept: 53.7134
    Epoch [1/50], Val Losses: mse: 66.8061, mae: 6.2283, huber: 5.7500, swd: 22.5169, ept: 79.0522
    Epoch [1/50], Test Losses: mse: 65.7787, mae: 6.1462, huber: 5.6685, swd: 22.1572, ept: 79.3999
      Epoch 1 composite train-obj: 5.883173
            Val objective improved inf → 5.7500, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 63.7850, mae: 5.9783, huber: 5.5023, swd: 17.2182, ept: 59.4658
    Epoch [2/50], Val Losses: mse: 63.6598, mae: 6.0033, huber: 5.5262, swd: 19.3474, ept: 104.3577
    Epoch [2/50], Test Losses: mse: 62.4457, mae: 5.8987, huber: 5.4226, swd: 19.9942, ept: 105.7995
      Epoch 2 composite train-obj: 5.502263
            Val objective improved 5.7500 → 5.5262, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 61.9044, mae: 5.8270, huber: 5.3527, swd: 16.2394, ept: 62.4926
    Epoch [3/50], Val Losses: mse: 61.9142, mae: 5.7886, huber: 5.3154, swd: 19.0265, ept: 113.1371
    Epoch [3/50], Test Losses: mse: 60.5246, mae: 5.6443, huber: 5.1722, swd: 17.8844, ept: 113.4788
      Epoch 3 composite train-obj: 5.352682
            Val objective improved 5.5262 → 5.3154, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 61.0656, mae: 5.7500, huber: 5.2769, swd: 15.9227, ept: 64.0231
    Epoch [4/50], Val Losses: mse: 60.2120, mae: 5.7184, huber: 5.2459, swd: 18.2132, ept: 119.8027
    Epoch [4/50], Test Losses: mse: 58.3582, mae: 5.5494, huber: 5.0775, swd: 17.5073, ept: 118.8039
      Epoch 4 composite train-obj: 5.276854
            Val objective improved 5.3154 → 5.2459, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 59.8505, mae: 5.6641, huber: 5.1921, swd: 15.6382, ept: 65.1161
    Epoch [5/50], Val Losses: mse: 58.7825, mae: 5.6611, huber: 5.1878, swd: 18.4591, ept: 125.9397
    Epoch [5/50], Test Losses: mse: 57.1486, mae: 5.4959, huber: 5.0243, swd: 18.3031, ept: 128.8246
      Epoch 5 composite train-obj: 5.192061
            Val objective improved 5.2459 → 5.1878, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 59.4665, mae: 5.6313, huber: 5.1598, swd: 15.4035, ept: 65.7727
    Epoch [6/50], Val Losses: mse: 59.4400, mae: 5.6440, huber: 5.1727, swd: 17.8666, ept: 138.1249
    Epoch [6/50], Test Losses: mse: 57.8337, mae: 5.5143, huber: 5.0438, swd: 17.8223, ept: 137.5603
      Epoch 6 composite train-obj: 5.159792
            Val objective improved 5.1878 → 5.1727, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 58.7045, mae: 5.5688, huber: 5.0982, swd: 15.0887, ept: 66.7455
    Epoch [7/50], Val Losses: mse: 57.7284, mae: 5.5101, huber: 5.0393, swd: 17.8061, ept: 147.5843
    Epoch [7/50], Test Losses: mse: 56.8586, mae: 5.4215, huber: 4.9517, swd: 17.9676, ept: 147.4536
      Epoch 7 composite train-obj: 5.098206
            Val objective improved 5.1727 → 5.0393, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 58.1784, mae: 5.5274, huber: 5.0574, swd: 14.6765, ept: 68.5820
    Epoch [8/50], Val Losses: mse: 56.1588, mae: 5.3733, huber: 4.9077, swd: 17.1033, ept: 159.4400
    Epoch [8/50], Test Losses: mse: 54.9555, mae: 5.2591, huber: 4.7943, swd: 16.6553, ept: 160.8900
      Epoch 8 composite train-obj: 5.057365
            Val objective improved 5.0393 → 4.9077, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 57.6361, mae: 5.4763, huber: 5.0072, swd: 14.3348, ept: 67.6552
    Epoch [9/50], Val Losses: mse: 55.3605, mae: 5.3693, huber: 4.9017, swd: 16.9577, ept: 160.9941
    Epoch [9/50], Test Losses: mse: 54.2678, mae: 5.2492, huber: 4.7829, swd: 17.0089, ept: 163.9645
      Epoch 9 composite train-obj: 5.007200
            Val objective improved 4.9077 → 4.9017, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 57.3500, mae: 5.4587, huber: 4.9897, swd: 14.1539, ept: 67.9097
    Epoch [10/50], Val Losses: mse: 57.3150, mae: 5.4991, huber: 5.0300, swd: 15.8757, ept: 146.6003
    Epoch [10/50], Test Losses: mse: 56.2994, mae: 5.3882, huber: 4.9207, swd: 15.6693, ept: 149.1855
      Epoch 10 composite train-obj: 4.989697
            No improvement (5.0300), counter 1/5
    Epoch [11/50], Train Losses: mse: 57.1199, mae: 5.4342, huber: 4.9656, swd: 13.7397, ept: 67.9536
    Epoch [11/50], Val Losses: mse: 55.6813, mae: 5.3383, huber: 4.8725, swd: 15.3063, ept: 166.3076
    Epoch [11/50], Test Losses: mse: 54.7693, mae: 5.2471, huber: 4.7824, swd: 15.2452, ept: 170.5223
      Epoch 11 composite train-obj: 4.965645
            Val objective improved 4.9017 → 4.8725, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 57.1123, mae: 5.4409, huber: 4.9721, swd: 13.7316, ept: 68.1198
    Epoch [12/50], Val Losses: mse: 55.2536, mae: 5.2997, huber: 4.8331, swd: 14.6456, ept: 171.3503
    Epoch [12/50], Test Losses: mse: 54.0069, mae: 5.1660, huber: 4.7013, swd: 13.9559, ept: 175.6274
      Epoch 12 composite train-obj: 4.972086
            Val objective improved 4.8725 → 4.8331, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 56.4206, mae: 5.3803, huber: 4.9126, swd: 13.2921, ept: 68.6407
    Epoch [13/50], Val Losses: mse: 55.3341, mae: 5.2336, huber: 4.7702, swd: 14.7000, ept: 173.9553
    Epoch [13/50], Test Losses: mse: 54.0706, mae: 5.1130, huber: 4.6498, swd: 14.4369, ept: 177.9584
      Epoch 13 composite train-obj: 4.912578
            Val objective improved 4.8331 → 4.7702, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 56.2698, mae: 5.3593, huber: 4.8920, swd: 13.0040, ept: 68.5652
    Epoch [14/50], Val Losses: mse: 54.5394, mae: 5.2451, huber: 4.7789, swd: 13.8130, ept: 174.1228
    Epoch [14/50], Test Losses: mse: 54.1631, mae: 5.1800, huber: 4.7146, swd: 13.3858, ept: 177.1147
      Epoch 14 composite train-obj: 4.892016
            No improvement (4.7789), counter 1/5
    Epoch [15/50], Train Losses: mse: 55.9194, mae: 5.3355, huber: 4.8685, swd: 12.8630, ept: 68.9998
    Epoch [15/50], Val Losses: mse: 55.5532, mae: 5.3228, huber: 4.8567, swd: 14.1287, ept: 169.5742
    Epoch [15/50], Test Losses: mse: 53.9692, mae: 5.1851, huber: 4.7199, swd: 14.2217, ept: 173.0812
      Epoch 15 composite train-obj: 4.868451
            No improvement (4.8567), counter 2/5
    Epoch [16/50], Train Losses: mse: 55.9127, mae: 5.3337, huber: 4.8668, swd: 12.7779, ept: 69.1362
    Epoch [16/50], Val Losses: mse: 55.8390, mae: 5.3410, huber: 4.8752, swd: 15.0979, ept: 166.2778
    Epoch [16/50], Test Losses: mse: 54.5558, mae: 5.2104, huber: 4.7467, swd: 15.1196, ept: 172.6474
      Epoch 16 composite train-obj: 4.866756
            No improvement (4.8752), counter 3/5
    Epoch [17/50], Train Losses: mse: 55.7404, mae: 5.3136, huber: 4.8470, swd: 12.5637, ept: 68.6400
    Epoch [17/50], Val Losses: mse: 54.7698, mae: 5.2175, huber: 4.7544, swd: 13.9065, ept: 177.1548
    Epoch [17/50], Test Losses: mse: 54.2693, mae: 5.1516, huber: 4.6886, swd: 13.4520, ept: 177.9439
      Epoch 17 composite train-obj: 4.847000
            Val objective improved 4.7702 → 4.7544, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 55.3750, mae: 5.2868, huber: 4.8206, swd: 12.3967, ept: 70.8239
    Epoch [18/50], Val Losses: mse: 55.1927, mae: 5.2425, huber: 4.7792, swd: 14.2734, ept: 184.9553
    Epoch [18/50], Test Losses: mse: 53.9158, mae: 5.1308, huber: 4.6678, swd: 13.8509, ept: 184.5502
      Epoch 18 composite train-obj: 4.820594
            No improvement (4.7792), counter 1/5
    Epoch [19/50], Train Losses: mse: 55.1095, mae: 5.2636, huber: 4.7979, swd: 12.2058, ept: 69.4061
    Epoch [19/50], Val Losses: mse: 54.4288, mae: 5.2236, huber: 4.7597, swd: 13.6108, ept: 175.8342
    Epoch [19/50], Test Losses: mse: 52.8241, mae: 5.0703, huber: 4.6082, swd: 13.3809, ept: 181.2997
      Epoch 19 composite train-obj: 4.797893
            No improvement (4.7597), counter 2/5
    Epoch [20/50], Train Losses: mse: 55.1767, mae: 5.2674, huber: 4.8016, swd: 12.1296, ept: 69.6364
    Epoch [20/50], Val Losses: mse: 55.3377, mae: 5.2543, huber: 4.7905, swd: 14.4399, ept: 175.8146
    Epoch [20/50], Test Losses: mse: 54.0756, mae: 5.1351, huber: 4.6738, swd: 14.5480, ept: 185.6841
      Epoch 20 composite train-obj: 4.801583
            No improvement (4.7905), counter 3/5
    Epoch [21/50], Train Losses: mse: 55.2551, mae: 5.2668, huber: 4.8011, swd: 12.1319, ept: 69.0914
    Epoch [21/50], Val Losses: mse: 54.1669, mae: 5.1597, huber: 4.6966, swd: 14.0622, ept: 188.4252
    Epoch [21/50], Test Losses: mse: 52.5717, mae: 5.0171, huber: 4.5555, swd: 13.7822, ept: 190.0259
      Epoch 21 composite train-obj: 4.801069
            Val objective improved 4.7544 → 4.6966, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 54.8069, mae: 5.2329, huber: 4.7677, swd: 11.9034, ept: 69.7431
    Epoch [22/50], Val Losses: mse: 55.6712, mae: 5.2506, huber: 4.7864, swd: 12.8610, ept: 181.2927
    Epoch [22/50], Test Losses: mse: 53.3755, mae: 5.0724, huber: 4.6093, swd: 12.6035, ept: 185.8858
      Epoch 22 composite train-obj: 4.767668
            No improvement (4.7864), counter 1/5
    Epoch [23/50], Train Losses: mse: 54.7039, mae: 5.2274, huber: 4.7623, swd: 11.8214, ept: 69.9641
    Epoch [23/50], Val Losses: mse: 53.6277, mae: 5.1338, huber: 4.6713, swd: 13.4446, ept: 189.5284
    Epoch [23/50], Test Losses: mse: 52.6168, mae: 5.0318, huber: 4.5710, swd: 13.5621, ept: 194.3216
      Epoch 23 composite train-obj: 4.762315
            Val objective improved 4.6966 → 4.6713, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 54.6410, mae: 5.2140, huber: 4.7492, swd: 11.6284, ept: 69.9798
    Epoch [24/50], Val Losses: mse: 54.7755, mae: 5.2004, huber: 4.7368, swd: 13.5969, ept: 178.4598
    Epoch [24/50], Test Losses: mse: 53.6912, mae: 5.0995, huber: 4.6356, swd: 13.1905, ept: 180.0649
      Epoch 24 composite train-obj: 4.749152
            No improvement (4.7368), counter 1/5
    Epoch [25/50], Train Losses: mse: 54.5566, mae: 5.2123, huber: 4.7475, swd: 11.6180, ept: 70.8148
    Epoch [25/50], Val Losses: mse: 55.4873, mae: 5.2080, huber: 4.7465, swd: 13.0506, ept: 192.4437
    Epoch [25/50], Test Losses: mse: 54.0397, mae: 5.0748, huber: 4.6141, swd: 12.1927, ept: 195.3070
      Epoch 25 composite train-obj: 4.747489
            No improvement (4.7465), counter 2/5
    Epoch [26/50], Train Losses: mse: 54.5460, mae: 5.2062, huber: 4.7415, swd: 11.5008, ept: 71.1392
    Epoch [26/50], Val Losses: mse: 54.1870, mae: 5.1394, huber: 4.6799, swd: 13.9346, ept: 190.2776
    Epoch [26/50], Test Losses: mse: 53.4216, mae: 5.0595, huber: 4.6008, swd: 13.4904, ept: 193.2627
      Epoch 26 composite train-obj: 4.741512
            No improvement (4.6799), counter 3/5
    Epoch [27/50], Train Losses: mse: 54.1792, mae: 5.1780, huber: 4.7137, swd: 11.3042, ept: 70.0581
    Epoch [27/50], Val Losses: mse: 55.0829, mae: 5.2472, huber: 4.7822, swd: 13.0485, ept: 173.8809
    Epoch [27/50], Test Losses: mse: 54.4407, mae: 5.1680, huber: 4.7034, swd: 13.0511, ept: 175.3440
      Epoch 27 composite train-obj: 4.713729
            No improvement (4.7822), counter 4/5
    Epoch [28/50], Train Losses: mse: 54.5245, mae: 5.2082, huber: 4.7434, swd: 11.4755, ept: 70.0561
    Epoch [28/50], Val Losses: mse: 55.0088, mae: 5.2033, huber: 4.7400, swd: 13.1901, ept: 184.6626
    Epoch [28/50], Test Losses: mse: 53.4652, mae: 5.0677, huber: 4.6056, swd: 13.1655, ept: 189.4813
      Epoch 28 composite train-obj: 4.743414
    Epoch [28/50], Test Losses: mse: 52.6168, mae: 5.0318, huber: 4.5710, swd: 13.5621, ept: 194.3216
    Best round's Test MSE: 52.6168, MAE: 5.0318, SWD: 13.5621
    Best round's Validation MSE: 53.6277, MAE: 5.1338, SWD: 13.4446
    Best round's Test verification MSE : 52.6168, MAE: 5.0318, SWD: 13.5621
    Time taken: 104.48 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.5312, mae: 6.3842, huber: 5.9047, swd: 16.0523, ept: 53.1464
    Epoch [1/50], Val Losses: mse: 68.0926, mae: 6.2620, huber: 5.7831, swd: 17.9755, ept: 79.7386
    Epoch [1/50], Test Losses: mse: 66.0520, mae: 6.0945, huber: 5.6161, swd: 16.8849, ept: 82.4304
      Epoch 1 composite train-obj: 5.904740
            Val objective improved inf → 5.7831, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 63.6861, mae: 5.9709, huber: 5.4949, swd: 16.8632, ept: 60.1340
    Epoch [2/50], Val Losses: mse: 60.6212, mae: 5.8481, huber: 5.3717, swd: 19.4588, ept: 108.4191
    Epoch [2/50], Test Losses: mse: 59.5123, mae: 5.7242, huber: 5.2490, swd: 18.9610, ept: 110.1290
      Epoch 2 composite train-obj: 5.494888
            Val objective improved 5.7831 → 5.3717, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 61.9643, mae: 5.8313, huber: 5.3570, swd: 15.9410, ept: 62.4677
    Epoch [3/50], Val Losses: mse: 62.0690, mae: 5.8500, huber: 5.3749, swd: 17.2681, ept: 119.1584
    Epoch [3/50], Test Losses: mse: 60.6370, mae: 5.7104, huber: 5.2363, swd: 17.0219, ept: 118.9858
      Epoch 3 composite train-obj: 5.356951
            No improvement (5.3749), counter 1/5
    Epoch [4/50], Train Losses: mse: 60.5326, mae: 5.7198, huber: 5.2469, swd: 15.3676, ept: 64.5079
    Epoch [4/50], Val Losses: mse: 57.8116, mae: 5.5423, huber: 5.0717, swd: 17.9993, ept: 141.3833
    Epoch [4/50], Test Losses: mse: 56.6246, mae: 5.4202, huber: 4.9512, swd: 17.7715, ept: 141.9003
      Epoch 4 composite train-obj: 5.246931
            Val objective improved 5.3717 → 5.0717, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 59.5763, mae: 5.6404, huber: 5.1686, swd: 14.9730, ept: 65.6826
    Epoch [5/50], Val Losses: mse: 59.1971, mae: 5.6854, huber: 5.2134, swd: 18.1266, ept: 127.3060
    Epoch [5/50], Test Losses: mse: 58.6895, mae: 5.5957, huber: 5.1250, swd: 17.9362, ept: 130.5862
      Epoch 5 composite train-obj: 5.168561
            No improvement (5.2134), counter 1/5
    Epoch [6/50], Train Losses: mse: 59.1342, mae: 5.5991, huber: 5.1281, swd: 14.6431, ept: 66.0404
    Epoch [6/50], Val Losses: mse: 58.5179, mae: 5.6250, huber: 5.1526, swd: 16.8221, ept: 131.3130
    Epoch [6/50], Test Losses: mse: 57.7211, mae: 5.5228, huber: 5.0516, swd: 16.6461, ept: 133.8026
      Epoch 6 composite train-obj: 5.128089
            No improvement (5.1526), counter 2/5
    Epoch [7/50], Train Losses: mse: 58.5537, mae: 5.5494, huber: 5.0792, swd: 14.3583, ept: 66.7622
    Epoch [7/50], Val Losses: mse: 57.6677, mae: 5.5975, huber: 5.1248, swd: 16.6741, ept: 136.5430
    Epoch [7/50], Test Losses: mse: 56.8384, mae: 5.4930, huber: 5.0214, swd: 16.3020, ept: 138.2103
      Epoch 7 composite train-obj: 5.079165
            No improvement (5.1248), counter 3/5
    Epoch [8/50], Train Losses: mse: 58.0660, mae: 5.5197, huber: 5.0497, swd: 14.2866, ept: 67.1285
    Epoch [8/50], Val Losses: mse: 56.4643, mae: 5.3802, huber: 4.9125, swd: 14.9800, ept: 161.3494
    Epoch [8/50], Test Losses: mse: 55.7949, mae: 5.3074, huber: 4.8404, swd: 15.2223, ept: 159.9925
      Epoch 8 composite train-obj: 5.049713
            Val objective improved 5.0717 → 4.9125, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 57.7704, mae: 5.4859, huber: 5.0166, swd: 13.9651, ept: 67.5214
    Epoch [9/50], Val Losses: mse: 55.7723, mae: 5.2935, huber: 4.8292, swd: 15.0484, ept: 166.3389
    Epoch [9/50], Test Losses: mse: 54.4732, mae: 5.1627, huber: 4.6995, swd: 14.9354, ept: 168.2269
      Epoch 9 composite train-obj: 5.016558
            Val objective improved 4.9125 → 4.8292, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 57.1586, mae: 5.4398, huber: 4.9711, swd: 13.7047, ept: 68.0941
    Epoch [10/50], Val Losses: mse: 54.8786, mae: 5.2940, huber: 4.8269, swd: 15.8389, ept: 168.4397
    Epoch [10/50], Test Losses: mse: 54.2901, mae: 5.2170, huber: 4.7513, swd: 15.8276, ept: 172.4552
      Epoch 10 composite train-obj: 4.971133
            Val objective improved 4.8292 → 4.8269, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 56.8240, mae: 5.4080, huber: 4.9400, swd: 13.3784, ept: 68.1043
    Epoch [11/50], Val Losses: mse: 55.2956, mae: 5.3132, huber: 4.8464, swd: 15.5765, ept: 168.2136
    Epoch [11/50], Test Losses: mse: 54.1966, mae: 5.2155, huber: 4.7491, swd: 15.3221, ept: 167.6471
      Epoch 11 composite train-obj: 4.939984
            No improvement (4.8464), counter 1/5
    Epoch [12/50], Train Losses: mse: 56.7668, mae: 5.4070, huber: 4.9388, swd: 13.3200, ept: 68.0196
    Epoch [12/50], Val Losses: mse: 54.7653, mae: 5.2995, huber: 4.8325, swd: 15.6185, ept: 159.1144
    Epoch [12/50], Test Losses: mse: 53.8636, mae: 5.2016, huber: 4.7359, swd: 15.9343, ept: 161.4672
      Epoch 12 composite train-obj: 4.938768
            No improvement (4.8325), counter 2/5
    Epoch [13/50], Train Losses: mse: 56.3938, mae: 5.3712, huber: 4.9037, swd: 12.9766, ept: 68.5818
    Epoch [13/50], Val Losses: mse: 56.1743, mae: 5.3276, huber: 4.8634, swd: 14.3977, ept: 159.4492
    Epoch [13/50], Test Losses: mse: 54.8538, mae: 5.2054, huber: 4.7418, swd: 14.2662, ept: 161.8833
      Epoch 13 composite train-obj: 4.903690
            No improvement (4.8634), counter 3/5
    Epoch [14/50], Train Losses: mse: 56.3306, mae: 5.3725, huber: 4.9050, swd: 13.1479, ept: 68.6379
    Epoch [14/50], Val Losses: mse: 54.3658, mae: 5.1864, huber: 4.7239, swd: 13.7619, ept: 181.5753
    Epoch [14/50], Test Losses: mse: 52.8701, mae: 5.0466, huber: 4.5863, swd: 13.8586, ept: 182.4551
      Epoch 14 composite train-obj: 4.905019
            Val objective improved 4.8269 → 4.7239, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 55.9064, mae: 5.3358, huber: 4.8687, swd: 12.7257, ept: 68.8597
    Epoch [15/50], Val Losses: mse: 54.7726, mae: 5.2165, huber: 4.7528, swd: 14.4733, ept: 176.6711
    Epoch [15/50], Test Losses: mse: 53.4914, mae: 5.1016, huber: 4.6385, swd: 14.6662, ept: 177.2841
      Epoch 15 composite train-obj: 4.868746
            No improvement (4.7528), counter 1/5
    Epoch [16/50], Train Losses: mse: 55.6433, mae: 5.3200, huber: 4.8532, swd: 12.5906, ept: 69.2919
    Epoch [16/50], Val Losses: mse: 55.3911, mae: 5.2697, huber: 4.8051, swd: 14.2525, ept: 171.1678
    Epoch [16/50], Test Losses: mse: 54.3548, mae: 5.1652, huber: 4.7009, swd: 14.1010, ept: 170.8192
      Epoch 16 composite train-obj: 4.853181
            No improvement (4.8051), counter 2/5
    Epoch [17/50], Train Losses: mse: 55.9737, mae: 5.3460, huber: 4.8787, swd: 12.7404, ept: 68.6385
    Epoch [17/50], Val Losses: mse: 53.7163, mae: 5.1463, huber: 4.6836, swd: 13.8214, ept: 185.6138
    Epoch [17/50], Test Losses: mse: 52.9562, mae: 5.0569, huber: 4.5961, swd: 13.9210, ept: 188.7389
      Epoch 17 composite train-obj: 4.878651
            Val objective improved 4.7239 → 4.6836, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 55.4281, mae: 5.2938, huber: 4.8275, swd: 12.2772, ept: 69.3164
    Epoch [18/50], Val Losses: mse: 57.7928, mae: 5.4373, huber: 4.9694, swd: 13.2791, ept: 166.2632
    Epoch [18/50], Test Losses: mse: 54.6014, mae: 5.1926, huber: 4.7275, swd: 13.4906, ept: 172.1737
      Epoch 18 composite train-obj: 4.827521
            No improvement (4.9694), counter 1/5
    Epoch [19/50], Train Losses: mse: 55.7652, mae: 5.3241, huber: 4.8572, swd: 12.6051, ept: 68.9167
    Epoch [19/50], Val Losses: mse: 54.0516, mae: 5.1702, huber: 4.7060, swd: 13.1019, ept: 186.2882
    Epoch [19/50], Test Losses: mse: 52.5708, mae: 5.0305, huber: 4.5677, swd: 12.9511, ept: 191.3284
      Epoch 19 composite train-obj: 4.857207
            No improvement (4.7060), counter 2/5
    Epoch [20/50], Train Losses: mse: 55.0456, mae: 5.2659, huber: 4.8000, swd: 12.1432, ept: 69.5542
    Epoch [20/50], Val Losses: mse: 56.1172, mae: 5.3487, huber: 4.8830, swd: 13.3046, ept: 161.7532
    Epoch [20/50], Test Losses: mse: 55.2157, mae: 5.2436, huber: 4.7795, swd: 13.4851, ept: 164.4831
      Epoch 20 composite train-obj: 4.799967
            No improvement (4.8830), counter 3/5
    Epoch [21/50], Train Losses: mse: 55.3066, mae: 5.2820, huber: 4.8159, swd: 11.9758, ept: 69.0188
    Epoch [21/50], Val Losses: mse: 55.5270, mae: 5.2304, huber: 4.7685, swd: 13.6557, ept: 178.0149
    Epoch [21/50], Test Losses: mse: 54.5963, mae: 5.1369, huber: 4.6762, swd: 13.4016, ept: 183.3145
      Epoch 21 composite train-obj: 4.815862
            No improvement (4.7685), counter 4/5
    Epoch [22/50], Train Losses: mse: 55.2966, mae: 5.2844, huber: 4.8183, swd: 12.3380, ept: 69.5029
    Epoch [22/50], Val Losses: mse: 56.7773, mae: 5.3512, huber: 4.8843, swd: 14.2229, ept: 171.4023
    Epoch [22/50], Test Losses: mse: 54.7360, mae: 5.1926, huber: 4.7279, swd: 14.3106, ept: 176.5428
      Epoch 22 composite train-obj: 4.818313
    Epoch [22/50], Test Losses: mse: 52.9562, mae: 5.0569, huber: 4.5961, swd: 13.9210, ept: 188.7389
    Best round's Test MSE: 52.9562, MAE: 5.0569, SWD: 13.9210
    Best round's Validation MSE: 53.7163, MAE: 5.1463, SWD: 13.8214
    Best round's Test verification MSE : 52.9562, MAE: 5.0569, SWD: 13.9210
    Time taken: 81.37 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.2914, mae: 6.3751, huber: 5.8957, swd: 16.7319, ept: 53.0588
    Epoch [1/50], Val Losses: mse: 67.7458, mae: 6.2822, huber: 5.8025, swd: 19.7155, ept: 77.1640
    Epoch [1/50], Test Losses: mse: 65.8142, mae: 6.1227, huber: 5.6443, swd: 19.1051, ept: 76.6526
      Epoch 1 composite train-obj: 5.895710
            Val objective improved inf → 5.8025, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 63.9297, mae: 5.9916, huber: 5.5155, swd: 17.7529, ept: 59.0604
    Epoch [2/50], Val Losses: mse: 63.5602, mae: 5.9759, huber: 5.4996, swd: 20.5177, ept: 99.5468
    Epoch [2/50], Test Losses: mse: 62.5721, mae: 5.8657, huber: 5.3903, swd: 19.8297, ept: 98.9478
      Epoch 2 composite train-obj: 5.515466
            Val objective improved 5.8025 → 5.4996, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 62.2369, mae: 5.8536, huber: 5.3790, swd: 17.1519, ept: 62.4129
    Epoch [3/50], Val Losses: mse: 62.0048, mae: 5.8578, huber: 5.3837, swd: 22.0711, ept: 112.6990
    Epoch [3/50], Test Losses: mse: 61.2061, mae: 5.7815, huber: 5.3088, swd: 21.8949, ept: 112.6014
      Epoch 3 composite train-obj: 5.378954
            Val objective improved 5.4996 → 5.3837, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 60.4553, mae: 5.7126, huber: 5.2398, swd: 16.3310, ept: 64.2346
    Epoch [4/50], Val Losses: mse: 60.8518, mae: 5.7461, huber: 5.2729, swd: 18.8222, ept: 121.9290
    Epoch [4/50], Test Losses: mse: 59.3618, mae: 5.6062, huber: 5.1334, swd: 17.8899, ept: 118.7973
      Epoch 4 composite train-obj: 5.239797
            Val objective improved 5.3837 → 5.2729, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 60.0423, mae: 5.6797, huber: 5.2074, swd: 16.4329, ept: 64.9396
    Epoch [5/50], Val Losses: mse: 58.2633, mae: 5.6035, huber: 5.1314, swd: 18.1822, ept: 132.4973
    Epoch [5/50], Test Losses: mse: 57.5176, mae: 5.5132, huber: 5.0425, swd: 18.1919, ept: 136.1537
      Epoch 5 composite train-obj: 5.207396
            Val objective improved 5.2729 → 5.1314, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 58.9431, mae: 5.5880, huber: 5.1169, swd: 15.6358, ept: 65.8632
    Epoch [6/50], Val Losses: mse: 57.8489, mae: 5.5119, huber: 5.0428, swd: 18.0445, ept: 146.9457
    Epoch [6/50], Test Losses: mse: 56.3322, mae: 5.3749, huber: 4.9070, swd: 17.4493, ept: 146.5841
      Epoch 6 composite train-obj: 5.116902
            Val objective improved 5.1314 → 5.0428, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 58.3295, mae: 5.5410, huber: 5.0706, swd: 15.3844, ept: 67.6799
    Epoch [7/50], Val Losses: mse: 57.8116, mae: 5.5027, huber: 5.0351, swd: 16.8140, ept: 150.5304
    Epoch [7/50], Test Losses: mse: 56.2796, mae: 5.3600, huber: 4.8933, swd: 15.7389, ept: 154.6491
      Epoch 7 composite train-obj: 5.070643
            Val objective improved 5.0428 → 5.0351, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 57.7585, mae: 5.4857, huber: 5.0163, swd: 14.9579, ept: 67.0608
    Epoch [8/50], Val Losses: mse: 58.1556, mae: 5.4703, huber: 5.0018, swd: 15.4317, ept: 158.8887
    Epoch [8/50], Test Losses: mse: 56.4317, mae: 5.3238, huber: 4.8562, swd: 14.4976, ept: 158.4178
      Epoch 8 composite train-obj: 5.016284
            Val objective improved 5.0351 → 5.0018, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 57.6775, mae: 5.4815, huber: 5.0121, swd: 14.7705, ept: 67.5841
    Epoch [9/50], Val Losses: mse: 56.3650, mae: 5.3860, huber: 4.9174, swd: 16.6598, ept: 159.4707
    Epoch [9/50], Test Losses: mse: 55.5005, mae: 5.2956, huber: 4.8286, swd: 15.9381, ept: 164.8879
      Epoch 9 composite train-obj: 5.012139
            Val objective improved 5.0018 → 4.9174, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 57.2416, mae: 5.4447, huber: 4.9758, swd: 14.3908, ept: 67.8100
    Epoch [10/50], Val Losses: mse: 54.8273, mae: 5.2946, huber: 4.8271, swd: 16.6690, ept: 166.5394
    Epoch [10/50], Test Losses: mse: 53.9412, mae: 5.1907, huber: 4.7246, swd: 15.9986, ept: 171.5957
      Epoch 10 composite train-obj: 4.975834
            Val objective improved 4.9174 → 4.8271, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 56.8888, mae: 5.4196, huber: 4.9511, swd: 14.2688, ept: 67.7523
    Epoch [11/50], Val Losses: mse: 55.4251, mae: 5.3407, huber: 4.8723, swd: 17.0065, ept: 163.0444
    Epoch [11/50], Test Losses: mse: 54.2728, mae: 5.2252, huber: 4.7576, swd: 16.4254, ept: 164.0783
      Epoch 11 composite train-obj: 4.951149
            No improvement (4.8723), counter 1/5
    Epoch [12/50], Train Losses: mse: 56.8781, mae: 5.4207, huber: 4.9522, swd: 14.2056, ept: 67.9294
    Epoch [12/50], Val Losses: mse: 59.3528, mae: 5.5637, huber: 5.0949, swd: 15.8291, ept: 147.4948
    Epoch [12/50], Test Losses: mse: 57.8218, mae: 5.4330, huber: 4.9649, swd: 15.6846, ept: 146.6610
      Epoch 12 composite train-obj: 4.952215
            No improvement (5.0949), counter 2/5
    Epoch [13/50], Train Losses: mse: 57.2364, mae: 5.4496, huber: 4.9807, swd: 14.2757, ept: 69.2117
    Epoch [13/50], Val Losses: mse: 54.2486, mae: 5.2117, huber: 4.7478, swd: 16.4451, ept: 176.8657
    Epoch [13/50], Test Losses: mse: 53.3162, mae: 5.1002, huber: 4.6385, swd: 16.0428, ept: 179.4152
      Epoch 13 composite train-obj: 4.980732
            Val objective improved 4.8271 → 4.7478, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 56.1666, mae: 5.3538, huber: 4.8867, swd: 13.5965, ept: 68.5590
    Epoch [14/50], Val Losses: mse: 53.6428, mae: 5.1737, huber: 4.7106, swd: 15.8662, ept: 181.1564
    Epoch [14/50], Test Losses: mse: 52.5766, mae: 5.0478, huber: 4.5862, swd: 15.3943, ept: 184.0437
      Epoch 14 composite train-obj: 4.886653
            Val objective improved 4.7478 → 4.7106, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 55.9289, mae: 5.3358, huber: 4.8688, swd: 13.4181, ept: 68.8759
    Epoch [15/50], Val Losses: mse: 55.4795, mae: 5.2581, huber: 4.7943, swd: 14.1237, ept: 174.2809
    Epoch [15/50], Test Losses: mse: 53.9026, mae: 5.1038, huber: 4.6415, swd: 13.6730, ept: 175.4264
      Epoch 15 composite train-obj: 4.868805
            No improvement (4.7943), counter 1/5
    Epoch [16/50], Train Losses: mse: 55.6752, mae: 5.3149, huber: 4.8482, swd: 13.2316, ept: 68.7000
    Epoch [16/50], Val Losses: mse: 56.5061, mae: 5.3486, huber: 4.8817, swd: 14.4917, ept: 170.4993
    Epoch [16/50], Test Losses: mse: 55.3907, mae: 5.2088, huber: 4.7438, swd: 13.8853, ept: 171.1056
      Epoch 16 composite train-obj: 4.848213
            No improvement (4.8817), counter 2/5
    Epoch [17/50], Train Losses: mse: 55.6789, mae: 5.3143, huber: 4.8476, swd: 13.1921, ept: 69.1718
    Epoch [17/50], Val Losses: mse: 53.2708, mae: 5.1429, huber: 4.6805, swd: 15.2771, ept: 184.1050
    Epoch [17/50], Test Losses: mse: 53.1021, mae: 5.0947, huber: 4.6328, swd: 14.9338, ept: 185.9149
      Epoch 17 composite train-obj: 4.847639
            Val objective improved 4.7106 → 4.6805, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 55.2587, mae: 5.2758, huber: 4.8099, swd: 12.8348, ept: 69.0780
    Epoch [18/50], Val Losses: mse: 54.3020, mae: 5.1870, huber: 4.7234, swd: 14.0156, ept: 185.9175
    Epoch [18/50], Test Losses: mse: 52.8049, mae: 5.0432, huber: 4.5810, swd: 13.3502, ept: 191.5747
      Epoch 18 composite train-obj: 4.809908
            No improvement (4.7234), counter 1/5
    Epoch [19/50], Train Losses: mse: 55.0912, mae: 5.2679, huber: 4.8020, swd: 12.8417, ept: 69.2960
    Epoch [19/50], Val Losses: mse: 53.4146, mae: 5.1371, huber: 4.6750, swd: 14.2432, ept: 185.6161
    Epoch [19/50], Test Losses: mse: 52.1783, mae: 5.0094, huber: 4.5488, swd: 13.5619, ept: 192.9853
      Epoch 19 composite train-obj: 4.802030
            Val objective improved 4.6805 → 4.6750, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 55.1880, mae: 5.2750, huber: 4.8089, swd: 12.8362, ept: 69.0260
    Epoch [20/50], Val Losses: mse: 54.4661, mae: 5.2145, huber: 4.7507, swd: 14.7983, ept: 183.4445
    Epoch [20/50], Test Losses: mse: 53.5356, mae: 5.1182, huber: 4.6553, swd: 14.6023, ept: 182.6129
      Epoch 20 composite train-obj: 4.808909
            No improvement (4.7507), counter 1/5
    Epoch [21/50], Train Losses: mse: 55.0502, mae: 5.2669, huber: 4.8010, swd: 12.7851, ept: 69.3155
    Epoch [21/50], Val Losses: mse: 53.9018, mae: 5.1746, huber: 4.7113, swd: 14.0629, ept: 188.8590
    Epoch [21/50], Test Losses: mse: 52.6068, mae: 5.0559, huber: 4.5941, swd: 14.0339, ept: 190.1650
      Epoch 21 composite train-obj: 4.800957
            No improvement (4.7113), counter 2/5
    Epoch [22/50], Train Losses: mse: 55.2630, mae: 5.2709, huber: 4.8051, swd: 12.5811, ept: 69.1285
    Epoch [22/50], Val Losses: mse: 57.2419, mae: 5.4248, huber: 4.9557, swd: 14.0500, ept: 157.0551
    Epoch [22/50], Test Losses: mse: 58.1466, mae: 5.4180, huber: 4.9497, swd: 13.8134, ept: 155.6081
      Epoch 22 composite train-obj: 4.805141
            No improvement (4.9557), counter 3/5
    Epoch [23/50], Train Losses: mse: 55.7573, mae: 5.3236, huber: 4.8569, swd: 13.2687, ept: 69.1015
    Epoch [23/50], Val Losses: mse: 56.2406, mae: 5.2343, huber: 4.7741, swd: 12.8759, ept: 186.5699
    Epoch [23/50], Test Losses: mse: 55.0203, mae: 5.1133, huber: 4.6537, swd: 11.9379, ept: 188.3929
      Epoch 23 composite train-obj: 4.856933
            No improvement (4.7741), counter 4/5
    Epoch [24/50], Train Losses: mse: 54.7953, mae: 5.2342, huber: 4.7690, swd: 12.4436, ept: 69.2802
    Epoch [24/50], Val Losses: mse: 53.6544, mae: 5.1078, huber: 4.6469, swd: 13.3763, ept: 192.4265
    Epoch [24/50], Test Losses: mse: 52.2051, mae: 4.9694, huber: 4.5109, swd: 12.8168, ept: 198.2961
      Epoch 24 composite train-obj: 4.769003
            Val objective improved 4.6750 → 4.6469, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 54.6036, mae: 5.2194, huber: 4.7544, swd: 12.2925, ept: 69.6466
    Epoch [25/50], Val Losses: mse: 55.4336, mae: 5.1639, huber: 4.7048, swd: 12.8024, ept: 194.8245
    Epoch [25/50], Test Losses: mse: 53.7102, mae: 4.9987, huber: 4.5415, swd: 12.0703, ept: 196.0759
      Epoch 25 composite train-obj: 4.754381
            No improvement (4.7048), counter 1/5
    Epoch [26/50], Train Losses: mse: 54.5220, mae: 5.2060, huber: 4.7413, swd: 12.1194, ept: 69.8402
    Epoch [26/50], Val Losses: mse: 53.9732, mae: 5.1133, huber: 4.6534, swd: 13.5161, ept: 196.1181
    Epoch [26/50], Test Losses: mse: 52.8833, mae: 5.0020, huber: 4.5428, swd: 12.5431, ept: 198.3699
      Epoch 26 composite train-obj: 4.741336
            No improvement (4.6534), counter 2/5
    Epoch [27/50], Train Losses: mse: 54.2151, mae: 5.1839, huber: 4.7197, swd: 12.0642, ept: 69.8238
    Epoch [27/50], Val Losses: mse: 53.4282, mae: 5.0829, huber: 4.6225, swd: 12.9692, ept: 193.5315
    Epoch [27/50], Test Losses: mse: 52.1405, mae: 4.9667, huber: 4.5074, swd: 12.5521, ept: 197.3664
      Epoch 27 composite train-obj: 4.719697
            Val objective improved 4.6469 → 4.6225, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 54.3320, mae: 5.1912, huber: 4.7267, swd: 11.9493, ept: 69.8159
    Epoch [28/50], Val Losses: mse: 53.1364, mae: 5.0723, huber: 4.6132, swd: 13.6428, ept: 196.1045
    Epoch [28/50], Test Losses: mse: 51.6106, mae: 4.9248, huber: 4.4676, swd: 13.0020, ept: 206.2427
      Epoch 28 composite train-obj: 4.726661
            Val objective improved 4.6225 → 4.6132, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 54.0906, mae: 5.1722, huber: 4.7080, swd: 11.8172, ept: 69.7754
    Epoch [29/50], Val Losses: mse: 53.9580, mae: 5.1711, huber: 4.7081, swd: 13.6723, ept: 187.5363
    Epoch [29/50], Test Losses: mse: 53.0058, mae: 5.0632, huber: 4.6014, swd: 13.4309, ept: 192.4537
      Epoch 29 composite train-obj: 4.708026
            No improvement (4.7081), counter 1/5
    Epoch [30/50], Train Losses: mse: 54.1866, mae: 5.1818, huber: 4.7174, swd: 11.8659, ept: 70.0822
    Epoch [30/50], Val Losses: mse: 56.9778, mae: 5.3548, huber: 4.8885, swd: 12.9114, ept: 176.7110
    Epoch [30/50], Test Losses: mse: 56.0740, mae: 5.2509, huber: 4.7850, swd: 12.4565, ept: 175.7361
      Epoch 30 composite train-obj: 4.717415
            No improvement (4.8885), counter 2/5
    Epoch [31/50], Train Losses: mse: 54.2748, mae: 5.1858, huber: 4.7214, swd: 11.8026, ept: 69.2796
    Epoch [31/50], Val Losses: mse: 53.4709, mae: 5.0620, huber: 4.6034, swd: 12.7125, ept: 199.9736
    Epoch [31/50], Test Losses: mse: 51.8651, mae: 4.9194, huber: 4.4617, swd: 12.3087, ept: 200.4249
      Epoch 31 composite train-obj: 4.721419
            Val objective improved 4.6132 → 4.6034, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 53.8024, mae: 5.1458, huber: 4.6821, swd: 11.6176, ept: 70.1727
    Epoch [32/50], Val Losses: mse: 52.5875, mae: 5.0344, huber: 4.5724, swd: 13.5177, ept: 199.9279
    Epoch [32/50], Test Losses: mse: 51.2676, mae: 4.9105, huber: 4.4503, swd: 13.1544, ept: 206.0707
      Epoch 32 composite train-obj: 4.682128
            Val objective improved 4.6034 → 4.5724, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 53.6902, mae: 5.1405, huber: 4.6768, swd: 11.5230, ept: 70.1714
    Epoch [33/50], Val Losses: mse: 52.7205, mae: 5.0471, huber: 4.5858, swd: 12.5577, ept: 193.0696
    Epoch [33/50], Test Losses: mse: 52.3581, mae: 4.9968, huber: 4.5365, swd: 12.1999, ept: 200.5648
      Epoch 33 composite train-obj: 4.676786
            No improvement (4.5858), counter 1/5
    Epoch [34/50], Train Losses: mse: 53.8037, mae: 5.1451, huber: 4.6815, swd: 11.5215, ept: 70.2603
    Epoch [34/50], Val Losses: mse: 52.9695, mae: 5.0062, huber: 4.5487, swd: 12.8104, ept: 204.7367
    Epoch [34/50], Test Losses: mse: 51.6029, mae: 4.8883, huber: 4.4319, swd: 12.5359, ept: 206.1645
      Epoch 34 composite train-obj: 4.681481
            Val objective improved 4.5724 → 4.5487, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 53.6103, mae: 5.1285, huber: 4.6651, swd: 11.4030, ept: 70.5686
    Epoch [35/50], Val Losses: mse: 52.9950, mae: 4.9809, huber: 4.5260, swd: 11.7278, ept: 204.7275
    Epoch [35/50], Test Losses: mse: 51.7005, mae: 4.8620, huber: 4.4085, swd: 11.3015, ept: 209.9382
      Epoch 35 composite train-obj: 4.665078
            Val objective improved 4.5487 → 4.5260, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 53.3914, mae: 5.1129, huber: 4.6498, swd: 11.3025, ept: 70.3036
    Epoch [36/50], Val Losses: mse: 53.0048, mae: 5.0251, huber: 4.5666, swd: 12.9731, ept: 203.3249
    Epoch [36/50], Test Losses: mse: 51.9132, mae: 4.9199, huber: 4.4629, swd: 12.5358, ept: 210.1800
      Epoch 36 composite train-obj: 4.649813
            No improvement (4.5666), counter 1/5
    Epoch [37/50], Train Losses: mse: 53.6606, mae: 5.1309, huber: 4.6676, swd: 11.4122, ept: 70.3972
    Epoch [37/50], Val Losses: mse: 60.9333, mae: 5.5606, huber: 5.0929, swd: 13.5797, ept: 159.0962
    Epoch [37/50], Test Losses: mse: 59.7588, mae: 5.4761, huber: 5.0092, swd: 13.2856, ept: 157.4728
      Epoch 37 composite train-obj: 4.667561
            No improvement (5.0929), counter 2/5
    Epoch [38/50], Train Losses: mse: 55.4058, mae: 5.2867, huber: 4.8206, swd: 12.5996, ept: 68.8632
    Epoch [38/50], Val Losses: mse: 53.3871, mae: 5.0362, huber: 4.5778, swd: 12.5800, ept: 200.4325
    Epoch [38/50], Test Losses: mse: 52.5501, mae: 4.9480, huber: 4.4914, swd: 12.2915, ept: 201.8584
      Epoch 38 composite train-obj: 4.820625
            No improvement (4.5778), counter 3/5
    Epoch [39/50], Train Losses: mse: 53.6413, mae: 5.1298, huber: 4.6664, swd: 11.4041, ept: 70.6732
    Epoch [39/50], Val Losses: mse: 53.2839, mae: 5.0323, huber: 4.5750, swd: 12.2286, ept: 205.4082
    Epoch [39/50], Test Losses: mse: 51.7661, mae: 4.8853, huber: 4.4298, swd: 11.7921, ept: 209.5994
      Epoch 39 composite train-obj: 4.666448
            No improvement (4.5750), counter 4/5
    Epoch [40/50], Train Losses: mse: 53.3523, mae: 5.1054, huber: 4.6425, swd: 11.2455, ept: 70.4483
    Epoch [40/50], Val Losses: mse: 51.8825, mae: 4.9499, huber: 4.4927, swd: 12.1986, ept: 210.2005
    Epoch [40/50], Test Losses: mse: 50.8153, mae: 4.8393, huber: 4.3829, swd: 11.5065, ept: 216.9680
      Epoch 40 composite train-obj: 4.642475
            Val objective improved 4.5260 → 4.4927, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 53.2837, mae: 5.1027, huber: 4.6398, swd: 11.1984, ept: 70.4989
    Epoch [41/50], Val Losses: mse: 54.1346, mae: 5.0811, huber: 4.6223, swd: 12.2257, ept: 198.6162
    Epoch [41/50], Test Losses: mse: 52.5762, mae: 4.9315, huber: 4.4748, swd: 11.3509, ept: 203.3783
      Epoch 41 composite train-obj: 4.639816
            No improvement (4.6223), counter 1/5
    Epoch [42/50], Train Losses: mse: 53.5133, mae: 5.1160, huber: 4.6530, swd: 11.2139, ept: 70.3924
    Epoch [42/50], Val Losses: mse: 53.3886, mae: 5.0326, huber: 4.5741, swd: 11.3868, ept: 203.2344
    Epoch [42/50], Test Losses: mse: 52.9107, mae: 4.9643, huber: 4.5066, swd: 10.7402, ept: 212.9341
      Epoch 42 composite train-obj: 4.652962
            No improvement (4.5741), counter 2/5
    Epoch [43/50], Train Losses: mse: 53.4554, mae: 5.1150, huber: 4.6518, swd: 11.3332, ept: 70.3228
    Epoch [43/50], Val Losses: mse: 54.2476, mae: 5.1440, huber: 4.6824, swd: 12.9175, ept: 185.8269
    Epoch [43/50], Test Losses: mse: 53.4344, mae: 5.0513, huber: 4.5902, swd: 12.3351, ept: 185.2356
      Epoch 43 composite train-obj: 4.651835
            No improvement (4.6824), counter 3/5
    Epoch [44/50], Train Losses: mse: 53.8693, mae: 5.1546, huber: 4.6907, swd: 11.7702, ept: 70.2402
    Epoch [44/50], Val Losses: mse: 52.0002, mae: 4.9892, huber: 4.5306, swd: 12.6009, ept: 203.5077
    Epoch [44/50], Test Losses: mse: 52.2195, mae: 4.9541, huber: 4.4968, swd: 12.3830, ept: 205.0549
      Epoch 44 composite train-obj: 4.690747
            No improvement (4.5306), counter 4/5
    Epoch [45/50], Train Losses: mse: 53.1710, mae: 5.0937, huber: 4.6308, swd: 11.2280, ept: 70.7437
    Epoch [45/50], Val Losses: mse: 52.8992, mae: 4.9839, huber: 4.5280, swd: 12.5278, ept: 207.7425
    Epoch [45/50], Test Losses: mse: 51.0127, mae: 4.8299, huber: 4.3755, swd: 12.0140, ept: 208.7886
      Epoch 45 composite train-obj: 4.630834
    Epoch [45/50], Test Losses: mse: 50.8153, mae: 4.8393, huber: 4.3829, swd: 11.5065, ept: 216.9680
    Best round's Test MSE: 50.8153, MAE: 4.8393, SWD: 11.5065
    Best round's Validation MSE: 51.8825, MAE: 4.9499, SWD: 12.1986
    Best round's Test verification MSE : 50.8153, MAE: 4.8393, SWD: 11.5065
    Time taken: 170.08 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq96_pred720_20250513_2340)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 52.1294 ± 0.9395
      mae: 4.9760 ± 0.0972
      huber: 4.5167 ± 0.0951
      swd: 12.9965 ± 1.0637
      ept: 200.0095 ± 12.2062
      count: 35.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 53.0755 ± 0.8443
      mae: 5.0767 ± 0.0898
      huber: 4.6158 ± 0.0872
      swd: 13.1549 ± 0.6934
      ept: 195.1142 ± 10.7866
      count: 35.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 356.00 seconds
    
    Experiment complete: PatchTST_lorenz_seq96_pred720_20250513_2340
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    

### Dlinear

#### 96-96
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 283
    Validation Batches: 40
    Test Batches: 80
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 67.5802, mae: 5.8322, huber: 5.3625, swd: 22.7905, ept: 37.8385
    Epoch [1/50], Val Losses: mse: 58.7565, mae: 5.3466, huber: 4.8853, swd: 18.5681, ept: 46.9791
    Epoch [1/50], Test Losses: mse: 59.6648, mae: 5.3766, huber: 4.9156, swd: 18.7761, ept: 45.4589
      Epoch 1 composite train-obj: 5.362518
            Val objective improved inf → 4.8853, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 57.1977, mae: 5.2232, huber: 4.7638, swd: 18.5460, ept: 49.2667
    Epoch [2/50], Val Losses: mse: 57.7461, mae: 5.2383, huber: 4.7819, swd: 18.3912, ept: 48.5593
    Epoch [2/50], Test Losses: mse: 58.6748, mae: 5.2658, huber: 4.8104, swd: 18.5813, ept: 46.9841
      Epoch 2 composite train-obj: 4.763802
            Val objective improved 4.8853 → 4.7819, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 56.7286, mae: 5.1564, huber: 4.7005, swd: 17.9162, ept: 50.4837
    Epoch [3/50], Val Losses: mse: 56.7919, mae: 5.1927, huber: 4.7362, swd: 18.9322, ept: 48.5576
    Epoch [3/50], Test Losses: mse: 57.5400, mae: 5.2063, huber: 4.7511, swd: 19.0928, ept: 47.1482
      Epoch 3 composite train-obj: 4.700480
            Val objective improved 4.7819 → 4.7362, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 56.4875, mae: 5.1237, huber: 4.6697, swd: 17.5498, ept: 51.1184
    Epoch [4/50], Val Losses: mse: 56.7030, mae: 5.1659, huber: 4.7113, swd: 18.4876, ept: 49.2305
    Epoch [4/50], Test Losses: mse: 57.4379, mae: 5.1796, huber: 4.7262, swd: 18.6359, ept: 47.8238
      Epoch 4 composite train-obj: 4.669661
            Val objective improved 4.7362 → 4.7113, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 56.2448, mae: 5.1018, huber: 4.6492, swd: 17.4134, ept: 51.4435
    Epoch [5/50], Val Losses: mse: 56.9091, mae: 5.1462, huber: 4.6946, swd: 17.9562, ept: 49.8347
    Epoch [5/50], Test Losses: mse: 57.7978, mae: 5.1735, huber: 4.7222, swd: 18.1549, ept: 48.3341
      Epoch 5 composite train-obj: 4.649244
            Val objective improved 4.7113 → 4.6946, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 56.1173, mae: 5.0858, huber: 4.6341, swd: 17.2284, ept: 51.8014
    Epoch [6/50], Val Losses: mse: 56.7041, mae: 5.1361, huber: 4.6833, swd: 17.5049, ept: 49.8951
    Epoch [6/50], Test Losses: mse: 57.5175, mae: 5.1491, huber: 4.6978, swd: 17.6037, ept: 48.4701
      Epoch 6 composite train-obj: 4.634075
            Val objective improved 4.6946 → 4.6833, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 56.0768, mae: 5.0723, huber: 4.6214, swd: 17.0030, ept: 52.0144
    Epoch [7/50], Val Losses: mse: 56.6152, mae: 5.1269, huber: 4.6745, swd: 17.1778, ept: 50.2749
    Epoch [7/50], Test Losses: mse: 57.5458, mae: 5.1479, huber: 4.6965, swd: 17.3003, ept: 48.8865
      Epoch 7 composite train-obj: 4.621443
            Val objective improved 4.6833 → 4.6745, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 55.8818, mae: 5.0619, huber: 4.6118, swd: 17.0442, ept: 52.0065
    Epoch [8/50], Val Losses: mse: 56.9081, mae: 5.1294, huber: 4.6776, swd: 16.5165, ept: 51.2133
    Epoch [8/50], Test Losses: mse: 57.9085, mae: 5.1590, huber: 4.7076, swd: 16.6975, ept: 49.7990
      Epoch 8 composite train-obj: 4.611844
            No improvement (4.6776), counter 1/5
    Epoch [9/50], Train Losses: mse: 55.7774, mae: 5.0515, huber: 4.6019, swd: 16.9241, ept: 52.1777
    Epoch [9/50], Val Losses: mse: 56.4735, mae: 5.1146, huber: 4.6634, swd: 17.2711, ept: 50.3997
    Epoch [9/50], Test Losses: mse: 57.2517, mae: 5.1250, huber: 4.6751, swd: 17.3382, ept: 48.8599
      Epoch 9 composite train-obj: 4.601854
            Val objective improved 4.6745 → 4.6634, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 55.7672, mae: 5.0431, huber: 4.5942, swd: 16.7974, ept: 52.3842
    Epoch [10/50], Val Losses: mse: 56.7088, mae: 5.0935, huber: 4.6449, swd: 16.6289, ept: 51.5352
    Epoch [10/50], Test Losses: mse: 57.7547, mae: 5.1283, huber: 4.6806, swd: 16.8237, ept: 50.0236
      Epoch 10 composite train-obj: 4.594180
            Val objective improved 4.6634 → 4.6449, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 55.5687, mae: 5.0363, huber: 4.5879, swd: 16.8656, ept: 52.2820
    Epoch [11/50], Val Losses: mse: 57.2015, mae: 5.1036, huber: 4.6551, swd: 15.9340, ept: 51.6471
    Epoch [11/50], Test Losses: mse: 58.2467, mae: 5.1391, huber: 4.6909, swd: 16.0644, ept: 50.2424
      Epoch 11 composite train-obj: 4.587853
            No improvement (4.6551), counter 1/5
    Epoch [12/50], Train Losses: mse: 55.6857, mae: 5.0339, huber: 4.5857, swd: 16.6606, ept: 52.4661
    Epoch [12/50], Val Losses: mse: 56.7563, mae: 5.1010, huber: 4.6518, swd: 16.4880, ept: 51.2626
    Epoch [12/50], Test Losses: mse: 57.6579, mae: 5.1258, huber: 4.6766, swd: 16.6042, ept: 49.9603
      Epoch 12 composite train-obj: 4.585663
            No improvement (4.6518), counter 2/5
    Epoch [13/50], Train Losses: mse: 55.5650, mae: 5.0233, huber: 4.5758, swd: 16.6228, ept: 52.5769
    Epoch [13/50], Val Losses: mse: 56.6296, mae: 5.0842, huber: 4.6359, swd: 16.5920, ept: 51.3997
    Epoch [13/50], Test Losses: mse: 57.6272, mae: 5.1167, huber: 4.6690, swd: 16.7610, ept: 49.9649
      Epoch 13 composite train-obj: 4.575809
            Val objective improved 4.6449 → 4.6359, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 55.4820, mae: 5.0191, huber: 4.5718, swd: 16.6365, ept: 52.5834
    Epoch [14/50], Val Losses: mse: 56.0759, mae: 5.0724, huber: 4.6256, swd: 17.2893, ept: 51.2714
    Epoch [14/50], Test Losses: mse: 56.9699, mae: 5.0967, huber: 4.6514, swd: 17.4503, ept: 49.7023
      Epoch 14 composite train-obj: 4.571761
            Val objective improved 4.6359 → 4.6256, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 55.4785, mae: 5.0156, huber: 4.5686, swd: 16.5383, ept: 52.6699
    Epoch [15/50], Val Losses: mse: 55.5484, mae: 5.0805, huber: 4.6301, swd: 17.7833, ept: 50.0824
    Epoch [15/50], Test Losses: mse: 56.1601, mae: 5.0783, huber: 4.6291, swd: 17.7867, ept: 48.8346
      Epoch 15 composite train-obj: 4.568597
            No improvement (4.6301), counter 1/5
    Epoch [16/50], Train Losses: mse: 55.3006, mae: 5.0105, huber: 4.5637, swd: 16.6637, ept: 52.5001
    Epoch [16/50], Val Losses: mse: 57.3492, mae: 5.0797, huber: 4.6330, swd: 15.2951, ept: 52.1833
    Epoch [16/50], Test Losses: mse: 58.4904, mae: 5.1260, huber: 4.6792, swd: 15.4721, ept: 50.6253
      Epoch 16 composite train-obj: 4.563727
            No improvement (4.6330), counter 2/5
    Epoch [17/50], Train Losses: mse: 55.3004, mae: 5.0060, huber: 4.5595, swd: 16.6032, ept: 52.6773
    Epoch [17/50], Val Losses: mse: 57.1458, mae: 5.0741, huber: 4.6260, swd: 15.7232, ept: 52.5870
    Epoch [17/50], Test Losses: mse: 58.2448, mae: 5.1161, huber: 4.6678, swd: 15.9104, ept: 51.0452
      Epoch 17 composite train-obj: 4.559475
            No improvement (4.6260), counter 3/5
    Epoch [18/50], Train Losses: mse: 55.3719, mae: 5.0048, huber: 4.5585, swd: 16.4987, ept: 52.7776
    Epoch [18/50], Val Losses: mse: 56.4871, mae: 5.0656, huber: 4.6187, swd: 15.9854, ept: 52.0369
    Epoch [18/50], Test Losses: mse: 57.5685, mae: 5.1033, huber: 4.6562, swd: 16.1456, ept: 50.5764
      Epoch 18 composite train-obj: 4.558493
            Val objective improved 4.6256 → 4.6187, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 55.2765, mae: 4.9982, huber: 4.5522, swd: 16.4602, ept: 52.8364
    Epoch [19/50], Val Losses: mse: 55.9596, mae: 5.0572, huber: 4.6116, swd: 16.9437, ept: 51.4113
    Epoch [19/50], Test Losses: mse: 56.8768, mae: 5.0843, huber: 4.6393, swd: 17.0863, ept: 49.8696
      Epoch 19 composite train-obj: 4.552234
            Val objective improved 4.6187 → 4.6116, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 55.2703, mae: 4.9961, huber: 4.5504, swd: 16.3820, ept: 52.9007
    Epoch [20/50], Val Losses: mse: 55.6381, mae: 5.0531, huber: 4.6072, swd: 17.2172, ept: 51.4090
    Epoch [20/50], Test Losses: mse: 56.4760, mae: 5.0750, huber: 4.6304, swd: 17.3367, ept: 49.9492
      Epoch 20 composite train-obj: 4.550428
            Val objective improved 4.6116 → 4.6072, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 55.1102, mae: 4.9938, huber: 4.5480, swd: 16.5896, ept: 52.7422
    Epoch [21/50], Val Losses: mse: 56.1893, mae: 5.0736, huber: 4.6260, swd: 16.5237, ept: 50.9289
    Epoch [21/50], Test Losses: mse: 57.0042, mae: 5.0834, huber: 4.6371, swd: 16.6033, ept: 49.5381
      Epoch 21 composite train-obj: 4.548010
            No improvement (4.6260), counter 1/5
    Epoch [22/50], Train Losses: mse: 55.1276, mae: 4.9893, huber: 4.5439, swd: 16.4426, ept: 52.7640
    Epoch [22/50], Val Losses: mse: 56.2956, mae: 5.0500, huber: 4.6050, swd: 16.4119, ept: 51.8516
    Epoch [22/50], Test Losses: mse: 57.2387, mae: 5.0770, huber: 4.6330, swd: 16.5177, ept: 50.2443
      Epoch 22 composite train-obj: 4.543913
            Val objective improved 4.6072 → 4.6050, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 55.1937, mae: 4.9863, huber: 4.5413, swd: 16.3367, ept: 53.0136
    Epoch [23/50], Val Losses: mse: 56.7262, mae: 5.0779, huber: 4.6307, swd: 15.9020, ept: 51.9089
    Epoch [23/50], Test Losses: mse: 57.8964, mae: 5.1226, huber: 4.6748, swd: 16.1228, ept: 50.6504
      Epoch 23 composite train-obj: 4.541256
            No improvement (4.6307), counter 1/5
    Epoch [24/50], Train Losses: mse: 55.0331, mae: 4.9827, huber: 4.5378, swd: 16.4253, ept: 52.8409
    Epoch [24/50], Val Losses: mse: 56.0728, mae: 5.0451, huber: 4.6008, swd: 16.5804, ept: 51.6678
    Epoch [24/50], Test Losses: mse: 57.0869, mae: 5.0794, huber: 4.6356, swd: 16.7687, ept: 50.2254
      Epoch 24 composite train-obj: 4.537802
            Val objective improved 4.6050 → 4.6008, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 55.1224, mae: 4.9803, huber: 4.5356, swd: 16.3230, ept: 52.9941
    Epoch [25/50], Val Losses: mse: 56.2065, mae: 5.0738, huber: 4.6268, swd: 16.2903, ept: 51.6670
    Epoch [25/50], Test Losses: mse: 57.2448, mae: 5.1063, huber: 4.6599, swd: 16.4383, ept: 50.1668
      Epoch 25 composite train-obj: 4.535554
            No improvement (4.6268), counter 1/5
    Epoch [26/50], Train Losses: mse: 54.9513, mae: 4.9805, huber: 4.5356, swd: 16.4565, ept: 52.8243
    Epoch [26/50], Val Losses: mse: 56.1454, mae: 5.0401, huber: 4.5960, swd: 16.3768, ept: 52.2885
    Epoch [26/50], Test Losses: mse: 57.1526, mae: 5.0707, huber: 4.6277, swd: 16.5407, ept: 50.8482
      Epoch 26 composite train-obj: 4.535556
            Val objective improved 4.6008 → 4.5960, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 55.0268, mae: 4.9774, huber: 4.5328, swd: 16.3376, ept: 53.0243
    Epoch [27/50], Val Losses: mse: 55.5485, mae: 5.0332, huber: 4.5884, swd: 16.7785, ept: 51.5266
    Epoch [27/50], Test Losses: mse: 56.4361, mae: 5.0539, huber: 4.6097, swd: 16.9461, ept: 50.1199
      Epoch 27 composite train-obj: 4.532767
            Val objective improved 4.5960 → 4.5884, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 55.0302, mae: 4.9731, huber: 4.5288, swd: 16.2732, ept: 53.1409
    Epoch [28/50], Val Losses: mse: 55.5126, mae: 5.0358, huber: 4.5913, swd: 16.7448, ept: 51.5354
    Epoch [28/50], Test Losses: mse: 56.4388, mae: 5.0584, huber: 4.6142, swd: 16.9180, ept: 50.1389
      Epoch 28 composite train-obj: 4.528787
            No improvement (4.5913), counter 1/5
    Epoch [29/50], Train Losses: mse: 54.8536, mae: 4.9710, huber: 4.5268, swd: 16.3941, ept: 52.8943
    Epoch [29/50], Val Losses: mse: 56.1899, mae: 5.0290, huber: 4.5859, swd: 15.8853, ept: 52.1905
    Epoch [29/50], Test Losses: mse: 57.2161, mae: 5.0629, huber: 4.6196, swd: 16.0501, ept: 51.0072
      Epoch 29 composite train-obj: 4.526806
            Val objective improved 4.5884 → 4.5859, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 54.9375, mae: 4.9683, huber: 4.5241, swd: 16.3178, ept: 53.2329
    Epoch [30/50], Val Losses: mse: 56.1682, mae: 5.0374, huber: 4.5942, swd: 16.1949, ept: 52.1975
    Epoch [30/50], Test Losses: mse: 57.1965, mae: 5.0713, huber: 4.6288, swd: 16.3782, ept: 50.6768
      Epoch 30 composite train-obj: 4.524145
            No improvement (4.5942), counter 1/5
    Epoch [31/50], Train Losses: mse: 54.9479, mae: 4.9683, huber: 4.5243, swd: 16.2196, ept: 53.0861
    Epoch [31/50], Val Losses: mse: 55.7675, mae: 5.0345, huber: 4.5904, swd: 16.6226, ept: 51.4012
    Epoch [31/50], Test Losses: mse: 56.6076, mae: 5.0555, huber: 4.6119, swd: 16.7248, ept: 50.1342
      Epoch 31 composite train-obj: 4.524331
            No improvement (4.5904), counter 2/5
    Epoch [32/50], Train Losses: mse: 54.8105, mae: 4.9650, huber: 4.5211, swd: 16.4035, ept: 53.1076
    Epoch [32/50], Val Losses: mse: 56.0566, mae: 5.0212, huber: 4.5770, swd: 15.8976, ept: 52.5458
    Epoch [32/50], Test Losses: mse: 57.0464, mae: 5.0481, huber: 4.6049, swd: 16.0420, ept: 51.1851
      Epoch 32 composite train-obj: 4.521073
            Val objective improved 4.5859 → 4.5770, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 54.9150, mae: 4.9642, huber: 4.5205, swd: 16.1915, ept: 53.2191
    Epoch [33/50], Val Losses: mse: 55.5844, mae: 5.0274, huber: 4.5830, swd: 16.4851, ept: 51.5863
    Epoch [33/50], Test Losses: mse: 56.3898, mae: 5.0448, huber: 4.6012, swd: 16.5659, ept: 50.1749
      Epoch 33 composite train-obj: 4.520516
            No improvement (4.5830), counter 1/5
    Epoch [34/50], Train Losses: mse: 54.8415, mae: 4.9621, huber: 4.5184, swd: 16.2597, ept: 53.2530
    Epoch [34/50], Val Losses: mse: 55.5273, mae: 5.0196, huber: 4.5749, swd: 16.4604, ept: 51.7981
    Epoch [34/50], Test Losses: mse: 56.3694, mae: 5.0326, huber: 4.5893, swd: 16.5515, ept: 50.3264
      Epoch 34 composite train-obj: 4.518353
            Val objective improved 4.5770 → 4.5749, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 54.7878, mae: 4.9594, huber: 4.5158, swd: 16.2402, ept: 53.1499
    Epoch [35/50], Val Losses: mse: 56.2914, mae: 5.0411, huber: 4.5961, swd: 15.8891, ept: 52.2760
    Epoch [35/50], Test Losses: mse: 57.3021, mae: 5.0724, huber: 4.6275, swd: 16.0608, ept: 51.0671
      Epoch 35 composite train-obj: 4.515811
            No improvement (4.5961), counter 1/5
    Epoch [36/50], Train Losses: mse: 54.8365, mae: 4.9588, huber: 4.5155, swd: 16.1659, ept: 53.2434
    Epoch [36/50], Val Losses: mse: 55.8884, mae: 5.0240, huber: 4.5798, swd: 16.2205, ept: 52.0731
    Epoch [36/50], Test Losses: mse: 56.8987, mae: 5.0542, huber: 4.6102, swd: 16.3955, ept: 50.6649
      Epoch 36 composite train-obj: 4.515459
            No improvement (4.5798), counter 2/5
    Epoch [37/50], Train Losses: mse: 54.7320, mae: 4.9562, huber: 4.5128, swd: 16.2305, ept: 53.1988
    Epoch [37/50], Val Losses: mse: 56.0826, mae: 5.0120, huber: 4.5694, swd: 15.8866, ept: 52.5179
    Epoch [37/50], Test Losses: mse: 57.1051, mae: 5.0451, huber: 4.6031, swd: 16.0597, ept: 51.1655
      Epoch 37 composite train-obj: 4.512784
            Val objective improved 4.5749 → 4.5694, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 54.8039, mae: 4.9542, huber: 4.5110, swd: 16.1929, ept: 53.3011
    Epoch [38/50], Val Losses: mse: 55.5995, mae: 5.0165, huber: 4.5731, swd: 16.3332, ept: 51.8886
    Epoch [38/50], Test Losses: mse: 56.6020, mae: 5.0457, huber: 4.6028, swd: 16.5034, ept: 50.4814
      Epoch 38 composite train-obj: 4.511012
            No improvement (4.5731), counter 1/5
    Epoch [39/50], Train Losses: mse: 54.7504, mae: 4.9530, huber: 4.5100, swd: 16.1619, ept: 53.2525
    Epoch [39/50], Val Losses: mse: 55.8507, mae: 5.0196, huber: 4.5763, swd: 16.0147, ept: 52.0767
    Epoch [39/50], Test Losses: mse: 56.8421, mae: 5.0484, huber: 4.6056, swd: 16.1584, ept: 50.5195
      Epoch 39 composite train-obj: 4.509951
            No improvement (4.5763), counter 2/5
    Epoch [40/50], Train Losses: mse: 54.6184, mae: 4.9514, huber: 4.5082, swd: 16.2182, ept: 53.1674
    Epoch [40/50], Val Losses: mse: 56.4464, mae: 5.0330, huber: 4.5899, swd: 15.4462, ept: 52.7067
    Epoch [40/50], Test Losses: mse: 57.6018, mae: 5.0788, huber: 4.6352, swd: 15.6820, ept: 51.1891
      Epoch 40 composite train-obj: 4.508164
            No improvement (4.5899), counter 3/5
    Epoch [41/50], Train Losses: mse: 54.7253, mae: 4.9515, huber: 4.5084, swd: 16.1247, ept: 53.3496
    Epoch [41/50], Val Losses: mse: 55.8330, mae: 5.0133, huber: 4.5694, swd: 15.9857, ept: 51.9155
    Epoch [41/50], Test Losses: mse: 56.8148, mae: 5.0399, huber: 4.5969, swd: 16.1109, ept: 50.5292
      Epoch 41 composite train-obj: 4.508355
            No improvement (4.5694), counter 4/5
    Epoch [42/50], Train Losses: mse: 54.6967, mae: 4.9479, huber: 4.5050, swd: 16.1348, ept: 53.2955
    Epoch [42/50], Val Losses: mse: 55.5918, mae: 5.0072, huber: 4.5652, swd: 16.1725, ept: 52.1414
    Epoch [42/50], Test Losses: mse: 56.6343, mae: 5.0388, huber: 4.5973, swd: 16.3459, ept: 50.7611
      Epoch 42 composite train-obj: 4.505012
            Val objective improved 4.5694 → 4.5652, saving checkpoint.
    Epoch [43/50], Train Losses: mse: 54.6209, mae: 4.9458, huber: 4.5029, swd: 16.2138, ept: 53.2986
    Epoch [43/50], Val Losses: mse: 55.5601, mae: 5.0197, huber: 4.5756, swd: 16.2469, ept: 51.9758
    Epoch [43/50], Test Losses: mse: 56.5186, mae: 5.0392, huber: 4.5965, swd: 16.3971, ept: 50.6741
      Epoch 43 composite train-obj: 4.502922
            No improvement (4.5756), counter 1/5
    Epoch [44/50], Train Losses: mse: 54.6980, mae: 4.9477, huber: 4.5048, swd: 16.0686, ept: 53.3572
    Epoch [44/50], Val Losses: mse: 55.3397, mae: 5.0172, huber: 4.5729, swd: 16.6973, ept: 52.0302
    Epoch [44/50], Test Losses: mse: 56.3028, mae: 5.0433, huber: 4.5999, swd: 16.9075, ept: 50.6693
      Epoch 44 composite train-obj: 4.504820
            No improvement (4.5729), counter 2/5
    Epoch [45/50], Train Losses: mse: 54.5805, mae: 4.9447, huber: 4.5019, swd: 16.1164, ept: 53.2663
    Epoch [45/50], Val Losses: mse: 55.5142, mae: 5.0045, huber: 4.5602, swd: 16.4334, ept: 52.3553
    Epoch [45/50], Test Losses: mse: 56.5384, mae: 5.0359, huber: 4.5919, swd: 16.6539, ept: 50.9686
      Epoch 45 composite train-obj: 4.501862
            Val objective improved 4.5652 → 4.5602, saving checkpoint.
    Epoch [46/50], Train Losses: mse: 54.5809, mae: 4.9398, huber: 4.4974, swd: 16.0952, ept: 53.3799
    Epoch [46/50], Val Losses: mse: 55.4058, mae: 4.9983, huber: 4.5543, swd: 16.1052, ept: 52.4381
    Epoch [46/50], Test Losses: mse: 56.4668, mae: 5.0330, huber: 4.5891, swd: 16.2910, ept: 51.0817
      Epoch 46 composite train-obj: 4.497393
            Val objective improved 4.5602 → 4.5543, saving checkpoint.
    Epoch [47/50], Train Losses: mse: 54.5571, mae: 4.9398, huber: 4.4973, swd: 16.1870, ept: 53.3937
    Epoch [47/50], Val Losses: mse: 55.4289, mae: 4.9992, huber: 4.5570, swd: 16.4038, ept: 51.8889
    Epoch [47/50], Test Losses: mse: 56.3497, mae: 5.0252, huber: 4.5835, swd: 16.5556, ept: 50.4318
      Epoch 47 composite train-obj: 4.497305
            No improvement (4.5570), counter 1/5
    Epoch [48/50], Train Losses: mse: 54.5953, mae: 4.9403, huber: 4.4979, swd: 16.0971, ept: 53.4615
    Epoch [48/50], Val Losses: mse: 55.4730, mae: 4.9989, huber: 4.5569, swd: 15.9577, ept: 52.5681
    Epoch [48/50], Test Losses: mse: 56.4526, mae: 5.0268, huber: 4.5851, swd: 16.1316, ept: 51.1397
      Epoch 48 composite train-obj: 4.497897
            No improvement (4.5569), counter 2/5
    Epoch [49/50], Train Losses: mse: 54.5941, mae: 4.9386, huber: 4.4963, swd: 16.0639, ept: 53.4810
    Epoch [49/50], Val Losses: mse: 55.2332, mae: 4.9999, huber: 4.5567, swd: 16.4828, ept: 51.9043
    Epoch [49/50], Test Losses: mse: 56.1579, mae: 5.0181, huber: 4.5762, swd: 16.6341, ept: 50.4268
      Epoch 49 composite train-obj: 4.496335
            No improvement (4.5567), counter 3/5
    Epoch [50/50], Train Losses: mse: 54.4519, mae: 4.9349, huber: 4.4927, swd: 16.1605, ept: 53.3038
    Epoch [50/50], Val Losses: mse: 55.9458, mae: 5.0060, huber: 4.5644, swd: 15.9555, ept: 52.5988
    Epoch [50/50], Test Losses: mse: 57.0858, mae: 5.0523, huber: 4.6101, swd: 16.2050, ept: 51.2165
      Epoch 50 composite train-obj: 4.492724
            No improvement (4.5644), counter 4/5
    Epoch [50/50], Test Losses: mse: 56.4668, mae: 5.0330, huber: 4.5891, swd: 16.2910, ept: 51.0817
    Best round's Test MSE: 56.4668, MAE: 5.0330, SWD: 16.2910
    Best round's Validation MSE: 55.4058, MAE: 4.9983, SWD: 16.1052
    Best round's Test verification MSE : 56.4668, MAE: 5.0330, SWD: 16.2910
    Time taken: 125.25 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 68.5781, mae: 5.8515, huber: 5.3820, swd: 22.9973, ept: 37.6019
    Epoch [1/50], Val Losses: mse: 58.9441, mae: 5.3460, huber: 4.8828, swd: 18.7577, ept: 47.0872
    Epoch [1/50], Test Losses: mse: 59.9368, mae: 5.3780, huber: 4.9156, swd: 19.0193, ept: 45.5574
      Epoch 1 composite train-obj: 5.381986
            Val objective improved inf → 4.8828, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 57.2875, mae: 5.2251, huber: 4.7657, swd: 18.6840, ept: 49.1685
    Epoch [2/50], Val Losses: mse: 58.2775, mae: 5.2409, huber: 4.7830, swd: 18.0261, ept: 49.2003
    Epoch [2/50], Test Losses: mse: 59.2515, mae: 5.2761, huber: 4.8185, swd: 18.2668, ept: 47.7252
      Epoch 2 composite train-obj: 4.765747
            Val objective improved 4.8828 → 4.7830, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 56.8660, mae: 5.1561, huber: 4.7005, swd: 17.9524, ept: 50.5730
    Epoch [3/50], Val Losses: mse: 57.2036, mae: 5.2037, huber: 4.7484, swd: 17.9784, ept: 49.3419
    Epoch [3/50], Test Losses: mse: 58.1115, mae: 5.2266, huber: 4.7720, swd: 18.1492, ept: 47.8927
      Epoch 3 composite train-obj: 4.700487
            Val objective improved 4.7830 → 4.7484, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 56.5173, mae: 5.1220, huber: 4.6681, swd: 17.5236, ept: 51.1408
    Epoch [4/50], Val Losses: mse: 56.8368, mae: 5.1734, huber: 4.7186, swd: 18.3761, ept: 49.6779
    Epoch [4/50], Test Losses: mse: 57.7005, mae: 5.1936, huber: 4.7397, swd: 18.6018, ept: 48.2929
      Epoch 4 composite train-obj: 4.668137
            Val objective improved 4.7484 → 4.7186, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 56.2703, mae: 5.1012, huber: 4.6486, swd: 17.4428, ept: 51.5085
    Epoch [5/50], Val Losses: mse: 56.9532, mae: 5.1654, huber: 4.7119, swd: 18.0532, ept: 49.5378
    Epoch [5/50], Test Losses: mse: 57.7597, mae: 5.1878, huber: 4.7352, swd: 18.2296, ept: 48.0194
      Epoch 5 composite train-obj: 4.648589
            Val objective improved 4.7186 → 4.7119, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 56.0942, mae: 5.0859, huber: 4.6342, swd: 17.3625, ept: 51.6081
    Epoch [6/50], Val Losses: mse: 57.0380, mae: 5.1360, huber: 4.6851, swd: 17.1403, ept: 50.7807
    Epoch [6/50], Test Losses: mse: 58.0231, mae: 5.1649, huber: 4.7144, swd: 17.3366, ept: 49.3270
      Epoch 6 composite train-obj: 4.634190
            Val objective improved 4.7119 → 4.6851, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 55.9982, mae: 5.0709, huber: 4.6202, swd: 17.1522, ept: 51.9909
    Epoch [7/50], Val Losses: mse: 56.8926, mae: 5.1170, huber: 4.6671, swd: 17.0017, ept: 51.1855
    Epoch [7/50], Test Losses: mse: 57.8058, mae: 5.1425, huber: 4.6933, swd: 17.1544, ept: 49.7311
      Epoch 7 composite train-obj: 4.620235
            Val objective improved 4.6851 → 4.6671, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 55.9059, mae: 5.0602, huber: 4.6103, swd: 17.0952, ept: 52.0542
    Epoch [8/50], Val Losses: mse: 57.3091, mae: 5.1301, huber: 4.6801, swd: 16.5992, ept: 51.0225
    Epoch [8/50], Test Losses: mse: 58.2637, mae: 5.1641, huber: 4.7145, swd: 16.7560, ept: 49.5452
      Epoch 8 composite train-obj: 4.610285
            No improvement (4.6801), counter 1/5
    Epoch [9/50], Train Losses: mse: 55.8686, mae: 5.0527, huber: 4.6034, swd: 16.9066, ept: 52.2441
    Epoch [9/50], Val Losses: mse: 56.7262, mae: 5.1074, huber: 4.6588, swd: 17.3539, ept: 51.0004
    Epoch [9/50], Test Losses: mse: 57.6329, mae: 5.1339, huber: 4.6864, swd: 17.5417, ept: 49.5771
      Epoch 9 composite train-obj: 4.603421
            Val objective improved 4.6671 → 4.6588, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 55.7656, mae: 5.0443, huber: 4.5952, swd: 16.8527, ept: 52.4110
    Epoch [10/50], Val Losses: mse: 56.1830, mae: 5.0976, huber: 4.6480, swd: 17.3873, ept: 50.6088
    Epoch [10/50], Test Losses: mse: 56.9779, mae: 5.1103, huber: 4.6622, swd: 17.5077, ept: 49.1549
      Epoch 10 composite train-obj: 4.595228
            Val objective improved 4.6588 → 4.6480, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 55.6807, mae: 5.0362, huber: 4.5877, swd: 16.8030, ept: 52.4536
    Epoch [11/50], Val Losses: mse: 55.9583, mae: 5.0915, huber: 4.6424, swd: 17.6348, ept: 50.4434
    Epoch [11/50], Test Losses: mse: 56.7587, mae: 5.1059, huber: 4.6580, swd: 17.7929, ept: 48.9634
      Epoch 11 composite train-obj: 4.587716
            Val objective improved 4.6480 → 4.6424, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 55.5222, mae: 5.0321, huber: 4.5841, swd: 16.8776, ept: 52.3344
    Epoch [12/50], Val Losses: mse: 56.4212, mae: 5.0966, huber: 4.6482, swd: 17.2008, ept: 50.7944
    Epoch [12/50], Test Losses: mse: 57.2959, mae: 5.1200, huber: 4.6721, swd: 17.3603, ept: 49.4546
      Epoch 12 composite train-obj: 4.584068
            No improvement (4.6482), counter 1/5
    Epoch [13/50], Train Losses: mse: 55.6274, mae: 5.0243, huber: 4.5767, swd: 16.5953, ept: 52.6393
    Epoch [13/50], Val Losses: mse: 55.9585, mae: 5.1066, huber: 4.6555, swd: 17.8861, ept: 50.5870
    Epoch [13/50], Test Losses: mse: 56.7726, mae: 5.1265, huber: 4.6765, swd: 18.0670, ept: 49.0701
      Epoch 13 composite train-obj: 4.576700
            No improvement (4.6555), counter 2/5
    Epoch [14/50], Train Losses: mse: 55.3764, mae: 5.0203, huber: 4.5728, swd: 16.8771, ept: 52.3617
    Epoch [14/50], Val Losses: mse: 56.4060, mae: 5.0848, huber: 4.6363, swd: 16.9673, ept: 51.2977
    Epoch [14/50], Test Losses: mse: 57.3271, mae: 5.1101, huber: 4.6621, swd: 17.1518, ept: 49.9862
      Epoch 14 composite train-obj: 4.572847
            Val objective improved 4.6424 → 4.6363, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 55.4976, mae: 5.0151, huber: 4.5682, swd: 16.6580, ept: 52.7142
    Epoch [15/50], Val Losses: mse: 55.9722, mae: 5.0741, huber: 4.6259, swd: 17.1856, ept: 51.0214
    Epoch [15/50], Test Losses: mse: 56.8639, mae: 5.0964, huber: 4.6492, swd: 17.3377, ept: 49.4982
      Epoch 15 composite train-obj: 4.568200
            Val objective improved 4.6363 → 4.6259, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 55.4279, mae: 5.0086, huber: 4.5620, swd: 16.6137, ept: 52.7691
    Epoch [16/50], Val Losses: mse: 55.9401, mae: 5.0731, huber: 4.6251, swd: 17.0574, ept: 50.9026
    Epoch [16/50], Test Losses: mse: 56.8476, mae: 5.0944, huber: 4.6471, swd: 17.2387, ept: 49.5700
      Epoch 16 composite train-obj: 4.562023
            Val objective improved 4.6259 → 4.6251, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 55.3194, mae: 5.0077, huber: 4.5611, swd: 16.6377, ept: 52.6871
    Epoch [17/50], Val Losses: mse: 56.1011, mae: 5.0610, huber: 4.6151, swd: 16.7821, ept: 51.3685
    Epoch [17/50], Test Losses: mse: 57.0730, mae: 5.0898, huber: 4.6441, swd: 16.9636, ept: 49.9050
      Epoch 17 composite train-obj: 4.561087
            Val objective improved 4.6251 → 4.6151, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 55.2717, mae: 5.0029, huber: 4.5566, swd: 16.5782, ept: 52.7133
    Epoch [18/50], Val Losses: mse: 56.0573, mae: 5.0636, huber: 4.6183, swd: 16.8962, ept: 51.5566
    Epoch [18/50], Test Losses: mse: 56.9403, mae: 5.0883, huber: 4.6437, swd: 17.0896, ept: 50.1864
      Epoch 18 composite train-obj: 4.556587
            No improvement (4.6183), counter 1/5
    Epoch [19/50], Train Losses: mse: 55.2111, mae: 4.9969, huber: 4.5510, swd: 16.6243, ept: 52.7809
    Epoch [19/50], Val Losses: mse: 56.4024, mae: 5.0495, huber: 4.6038, swd: 16.3129, ept: 52.1875
    Epoch [19/50], Test Losses: mse: 57.4087, mae: 5.0831, huber: 4.6376, swd: 16.5012, ept: 50.7056
      Epoch 19 composite train-obj: 4.550996
            Val objective improved 4.6151 → 4.6038, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 55.2269, mae: 4.9961, huber: 4.5502, swd: 16.5306, ept: 52.7782
    Epoch [20/50], Val Losses: mse: 56.7483, mae: 5.0632, huber: 4.6152, swd: 15.7999, ept: 52.3736
    Epoch [20/50], Test Losses: mse: 57.8255, mae: 5.1023, huber: 4.6545, swd: 15.9853, ept: 50.8386
      Epoch 20 composite train-obj: 4.550242
            No improvement (4.6152), counter 1/5
    Epoch [21/50], Train Losses: mse: 55.2312, mae: 4.9918, huber: 4.5461, swd: 16.4832, ept: 53.0212
    Epoch [21/50], Val Losses: mse: 56.1705, mae: 5.0575, huber: 4.6106, swd: 16.5158, ept: 51.7242
    Epoch [21/50], Test Losses: mse: 57.1470, mae: 5.0833, huber: 4.6372, swd: 16.7356, ept: 50.3267
      Epoch 21 composite train-obj: 4.546120
            No improvement (4.6106), counter 2/5
    Epoch [22/50], Train Losses: mse: 55.1124, mae: 4.9893, huber: 4.5439, swd: 16.5835, ept: 52.8956
    Epoch [22/50], Val Losses: mse: 56.5643, mae: 5.0587, huber: 4.6118, swd: 15.7697, ept: 51.8803
    Epoch [22/50], Test Losses: mse: 57.6017, mae: 5.0870, huber: 4.6408, swd: 15.9420, ept: 50.5833
      Epoch 22 composite train-obj: 4.543905
            No improvement (4.6118), counter 3/5
    Epoch [23/50], Train Losses: mse: 55.1873, mae: 4.9834, huber: 4.5385, swd: 16.3355, ept: 53.0701
    Epoch [23/50], Val Losses: mse: 55.8519, mae: 5.0509, huber: 4.6064, swd: 16.5892, ept: 51.1362
    Epoch [23/50], Test Losses: mse: 56.8189, mae: 5.0759, huber: 4.6322, swd: 16.7476, ept: 49.8243
      Epoch 23 composite train-obj: 4.538521
            No improvement (4.6064), counter 4/5
    Epoch [24/50], Train Losses: mse: 55.0426, mae: 4.9830, huber: 4.5379, swd: 16.5673, ept: 52.9168
    Epoch [24/50], Val Losses: mse: 56.2883, mae: 5.0575, huber: 4.6111, swd: 16.0244, ept: 52.0949
    Epoch [24/50], Test Losses: mse: 57.3663, mae: 5.0896, huber: 4.6434, swd: 16.2119, ept: 50.6799
      Epoch 24 composite train-obj: 4.537895
    Epoch [24/50], Test Losses: mse: 57.4087, mae: 5.0831, huber: 4.6376, swd: 16.5012, ept: 50.7056
    Best round's Test MSE: 57.4087, MAE: 5.0831, SWD: 16.5012
    Best round's Validation MSE: 56.4024, MAE: 5.0495, SWD: 16.3129
    Best round's Test verification MSE : 57.4087, MAE: 5.0831, SWD: 16.5012
    Time taken: 56.96 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 66.4006, mae: 5.8154, huber: 5.3455, swd: 21.6298, ept: 37.4852
    Epoch [1/50], Val Losses: mse: 58.6315, mae: 5.3571, huber: 4.8951, swd: 17.6252, ept: 46.4280
    Epoch [1/50], Test Losses: mse: 59.5439, mae: 5.3846, huber: 4.9234, swd: 17.8175, ept: 44.9992
      Epoch 1 composite train-obj: 5.345492
            Val objective improved inf → 4.8951, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 57.2100, mae: 5.2240, huber: 4.7649, swd: 17.2247, ept: 48.9578
    Epoch [2/50], Val Losses: mse: 57.8471, mae: 5.2429, huber: 4.7866, swd: 16.7812, ept: 48.1988
    Epoch [2/50], Test Losses: mse: 58.7303, mae: 5.2648, huber: 4.8096, swd: 16.9100, ept: 46.6092
      Epoch 2 composite train-obj: 4.764948
            Val objective improved 4.8951 → 4.7866, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 56.7172, mae: 5.1571, huber: 4.7014, swd: 16.5525, ept: 50.3361
    Epoch [3/50], Val Losses: mse: 58.1768, mae: 5.2055, huber: 4.7502, swd: 15.7887, ept: 50.0979
    Epoch [3/50], Test Losses: mse: 59.2931, mae: 5.2526, huber: 4.7977, swd: 15.9824, ept: 48.4728
      Epoch 3 composite train-obj: 4.701424
            Val objective improved 4.7866 → 4.7502, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 56.5647, mae: 5.1267, huber: 4.6726, swd: 16.1713, ept: 51.1005
    Epoch [4/50], Val Losses: mse: 57.0518, mae: 5.1720, huber: 4.7194, swd: 16.0651, ept: 49.3687
    Epoch [4/50], Test Losses: mse: 57.9769, mae: 5.1940, huber: 4.7422, swd: 16.1532, ept: 47.8996
      Epoch 4 composite train-obj: 4.672647
            Val objective improved 4.7502 → 4.7194, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 56.1984, mae: 5.1023, huber: 4.6496, swd: 16.0503, ept: 51.2810
    Epoch [5/50], Val Losses: mse: 57.3141, mae: 5.1555, huber: 4.7033, swd: 15.7725, ept: 50.4436
    Epoch [5/50], Test Losses: mse: 58.2583, mae: 5.1827, huber: 4.7316, swd: 15.8761, ept: 48.9617
      Epoch 5 composite train-obj: 4.649590
            Val objective improved 4.7194 → 4.7033, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 56.1768, mae: 5.0841, huber: 4.6327, swd: 15.6616, ept: 51.7322
    Epoch [6/50], Val Losses: mse: 57.1666, mae: 5.1321, huber: 4.6800, swd: 16.0126, ept: 50.8189
    Epoch [6/50], Test Losses: mse: 58.1629, mae: 5.1697, huber: 4.7179, swd: 16.2027, ept: 49.3792
      Epoch 6 composite train-obj: 4.632662
            Val objective improved 4.7033 → 4.6800, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 55.9895, mae: 5.0726, huber: 4.6220, swd: 15.7369, ept: 51.9012
    Epoch [7/50], Val Losses: mse: 56.9606, mae: 5.1315, huber: 4.6813, swd: 15.9125, ept: 51.1693
    Epoch [7/50], Test Losses: mse: 57.8906, mae: 5.1624, huber: 4.7129, swd: 16.0313, ept: 49.4879
      Epoch 7 composite train-obj: 4.622020
            No improvement (4.6813), counter 1/5
    Epoch [8/50], Train Losses: mse: 55.9398, mae: 5.0614, huber: 4.6113, swd: 15.5558, ept: 52.1800
    Epoch [8/50], Val Losses: mse: 57.0767, mae: 5.1204, huber: 4.6702, swd: 15.1866, ept: 50.9684
    Epoch [8/50], Test Losses: mse: 58.1110, mae: 5.1528, huber: 4.7030, swd: 15.3285, ept: 49.6276
      Epoch 8 composite train-obj: 4.611309
            Val objective improved 4.6800 → 4.6702, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 55.8331, mae: 5.0500, huber: 4.6006, swd: 15.4089, ept: 52.2746
    Epoch [9/50], Val Losses: mse: 56.3229, mae: 5.1049, huber: 4.6537, swd: 15.9436, ept: 50.8225
    Epoch [9/50], Test Losses: mse: 57.1763, mae: 5.1218, huber: 4.6719, swd: 16.0702, ept: 49.4089
      Epoch 9 composite train-obj: 4.600620
            Val objective improved 4.6702 → 4.6537, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 55.7061, mae: 5.0449, huber: 4.5960, swd: 15.5098, ept: 52.1978
    Epoch [10/50], Val Losses: mse: 56.6403, mae: 5.1008, huber: 4.6526, swd: 15.2049, ept: 51.0971
    Epoch [10/50], Test Losses: mse: 57.6421, mae: 5.1288, huber: 4.6810, swd: 15.3058, ept: 49.6397
      Epoch 10 composite train-obj: 4.596030
            Val objective improved 4.6537 → 4.6526, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 55.7000, mae: 5.0382, huber: 4.5896, swd: 15.3229, ept: 52.3468
    Epoch [11/50], Val Losses: mse: 56.8196, mae: 5.1045, huber: 4.6560, swd: 14.8524, ept: 51.2468
    Epoch [11/50], Test Losses: mse: 57.9149, mae: 5.1422, huber: 4.6943, swd: 14.9697, ept: 49.7618
      Epoch 11 composite train-obj: 4.589614
            No improvement (4.6560), counter 1/5
    Epoch [12/50], Train Losses: mse: 55.6033, mae: 5.0307, huber: 4.5828, swd: 15.3223, ept: 52.4987
    Epoch [12/50], Val Losses: mse: 56.0308, mae: 5.0989, huber: 4.6494, swd: 15.6624, ept: 51.0736
    Epoch [12/50], Test Losses: mse: 56.9804, mae: 5.1267, huber: 4.6773, swd: 15.7940, ept: 49.6159
      Epoch 12 composite train-obj: 4.582806
            Val objective improved 4.6526 → 4.6494, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 55.5269, mae: 5.0243, huber: 4.5767, swd: 15.2917, ept: 52.5136
    Epoch [13/50], Val Losses: mse: 55.8450, mae: 5.0803, huber: 4.6323, swd: 16.3798, ept: 50.5037
    Epoch [13/50], Test Losses: mse: 56.6669, mae: 5.0996, huber: 4.6528, swd: 16.5019, ept: 49.0146
      Epoch 13 composite train-obj: 4.576729
            Val objective improved 4.6494 → 4.6323, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 55.4700, mae: 5.0207, huber: 4.5732, swd: 15.2831, ept: 52.5117
    Epoch [14/50], Val Losses: mse: 56.3656, mae: 5.0756, huber: 4.6284, swd: 15.0704, ept: 51.5818
    Epoch [14/50], Test Losses: mse: 57.3395, mae: 5.1020, huber: 4.6551, swd: 15.1736, ept: 50.1749
      Epoch 14 composite train-obj: 4.573226
            Val objective improved 4.6323 → 4.6284, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 55.4465, mae: 5.0191, huber: 4.5719, swd: 15.2432, ept: 52.6312
    Epoch [15/50], Val Losses: mse: 56.4486, mae: 5.0856, huber: 4.6382, swd: 15.1854, ept: 51.3335
    Epoch [15/50], Test Losses: mse: 57.4385, mae: 5.1189, huber: 4.6713, swd: 15.3285, ept: 49.9282
      Epoch 15 composite train-obj: 4.571877
            No improvement (4.6382), counter 1/5
    Epoch [16/50], Train Losses: mse: 55.3185, mae: 5.0091, huber: 4.5625, swd: 15.2662, ept: 52.6090
    Epoch [16/50], Val Losses: mse: 56.6833, mae: 5.0786, huber: 4.6301, swd: 14.4169, ept: 52.0607
    Epoch [16/50], Test Losses: mse: 57.7149, mae: 5.1092, huber: 4.6614, swd: 14.5338, ept: 50.5669
      Epoch 16 composite train-obj: 4.562453
            No improvement (4.6301), counter 2/5
    Epoch [17/50], Train Losses: mse: 55.3588, mae: 5.0063, huber: 4.5598, swd: 15.0964, ept: 52.7309
    Epoch [17/50], Val Losses: mse: 56.3901, mae: 5.0743, huber: 4.6283, swd: 15.1040, ept: 51.6710
    Epoch [17/50], Test Losses: mse: 57.3621, mae: 5.1021, huber: 4.6570, swd: 15.2498, ept: 50.3044
      Epoch 17 composite train-obj: 4.559787
            Val objective improved 4.6284 → 4.6283, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 55.2510, mae: 5.0015, huber: 4.5552, swd: 15.1680, ept: 52.6789
    Epoch [18/50], Val Losses: mse: 56.2303, mae: 5.0670, huber: 4.6204, swd: 15.0753, ept: 51.3212
    Epoch [18/50], Test Losses: mse: 57.1847, mae: 5.0921, huber: 4.6466, swd: 15.1900, ept: 49.9035
      Epoch 18 composite train-obj: 4.555165
            Val objective improved 4.6283 → 4.6204, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 55.2543, mae: 4.9969, huber: 4.5511, swd: 15.0951, ept: 52.7670
    Epoch [19/50], Val Losses: mse: 55.9868, mae: 5.0537, huber: 4.6079, swd: 15.3102, ept: 51.8759
    Epoch [19/50], Test Losses: mse: 56.8896, mae: 5.0799, huber: 4.6346, swd: 15.4422, ept: 50.3880
      Epoch 19 composite train-obj: 4.551115
            Val objective improved 4.6204 → 4.6079, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 55.1530, mae: 4.9952, huber: 4.5494, swd: 15.0681, ept: 52.8726
    Epoch [20/50], Val Losses: mse: 56.2476, mae: 5.0704, huber: 4.6220, swd: 15.0206, ept: 52.2167
    Epoch [20/50], Test Losses: mse: 57.2624, mae: 5.1084, huber: 4.6597, swd: 15.1458, ept: 50.6467
      Epoch 20 composite train-obj: 4.549410
            No improvement (4.6220), counter 1/5
    Epoch [21/50], Train Losses: mse: 55.1508, mae: 4.9939, huber: 4.5481, swd: 15.0788, ept: 52.7702
    Epoch [21/50], Val Losses: mse: 56.2080, mae: 5.0439, huber: 4.5975, swd: 15.1765, ept: 51.8135
    Epoch [21/50], Test Losses: mse: 57.1321, mae: 5.0693, huber: 4.6240, swd: 15.3141, ept: 50.4362
      Epoch 21 composite train-obj: 4.548147
            Val objective improved 4.6079 → 4.5975, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 55.1391, mae: 4.9882, huber: 4.5428, swd: 15.0456, ept: 52.9233
    Epoch [22/50], Val Losses: mse: 55.3988, mae: 5.0418, huber: 4.5947, swd: 15.6461, ept: 51.0794
    Epoch [22/50], Test Losses: mse: 56.2266, mae: 5.0581, huber: 4.6122, swd: 15.7259, ept: 49.6675
      Epoch 22 composite train-obj: 4.542811
            Val objective improved 4.5975 → 4.5947, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 55.0320, mae: 4.9845, huber: 4.5393, swd: 15.0899, ept: 52.8680
    Epoch [23/50], Val Losses: mse: 56.3786, mae: 5.0394, huber: 4.5939, swd: 14.8213, ept: 51.8931
    Epoch [23/50], Test Losses: mse: 57.3739, mae: 5.0656, huber: 4.6213, swd: 14.9533, ept: 50.5182
      Epoch 23 composite train-obj: 4.539316
            Val objective improved 4.5947 → 4.5939, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 55.1390, mae: 4.9853, huber: 4.5402, swd: 14.9496, ept: 53.0165
    Epoch [24/50], Val Losses: mse: 56.2422, mae: 5.0480, huber: 4.6021, swd: 14.7506, ept: 51.9425
    Epoch [24/50], Test Losses: mse: 57.3191, mae: 5.0856, huber: 4.6395, swd: 14.8859, ept: 50.6047
      Epoch 24 composite train-obj: 4.540158
            No improvement (4.6021), counter 1/5
    Epoch [25/50], Train Losses: mse: 55.0721, mae: 4.9807, huber: 4.5358, swd: 14.9438, ept: 53.0810
    Epoch [25/50], Val Losses: mse: 55.5789, mae: 5.0284, huber: 4.5839, swd: 15.5945, ept: 51.1928
    Epoch [25/50], Test Losses: mse: 56.4975, mae: 5.0516, huber: 4.6079, swd: 15.7292, ept: 49.7808
      Epoch 25 composite train-obj: 4.535806
            Val objective improved 4.5939 → 4.5839, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 55.0258, mae: 4.9781, huber: 4.5333, swd: 14.9965, ept: 53.0786
    Epoch [26/50], Val Losses: mse: 55.5576, mae: 5.0510, huber: 4.6038, swd: 15.5934, ept: 51.4845
    Epoch [26/50], Test Losses: mse: 56.4031, mae: 5.0691, huber: 4.6233, swd: 15.6739, ept: 49.9937
      Epoch 26 composite train-obj: 4.533336
            No improvement (4.6038), counter 1/5
    Epoch [27/50], Train Losses: mse: 54.9151, mae: 4.9777, huber: 4.5329, swd: 15.0277, ept: 52.9060
    Epoch [27/50], Val Losses: mse: 56.2896, mae: 5.0541, huber: 4.6095, swd: 14.6500, ept: 52.1815
    Epoch [27/50], Test Losses: mse: 57.3350, mae: 5.0877, huber: 4.6428, swd: 14.8106, ept: 50.8199
      Epoch 27 composite train-obj: 4.532923
            No improvement (4.6095), counter 2/5
    Epoch [28/50], Train Losses: mse: 55.0135, mae: 4.9737, huber: 4.5293, swd: 14.8870, ept: 53.1530
    Epoch [28/50], Val Losses: mse: 55.3138, mae: 5.0330, huber: 4.5870, swd: 15.6419, ept: 51.1458
    Epoch [28/50], Test Losses: mse: 56.1365, mae: 5.0449, huber: 4.6005, swd: 15.7307, ept: 49.7166
      Epoch 28 composite train-obj: 4.529309
            No improvement (4.5870), counter 3/5
    Epoch [29/50], Train Losses: mse: 54.9017, mae: 4.9712, huber: 4.5268, swd: 14.9910, ept: 52.9748
    Epoch [29/50], Val Losses: mse: 55.8254, mae: 5.0316, huber: 4.5860, swd: 14.9947, ept: 51.5977
    Epoch [29/50], Test Losses: mse: 56.7555, mae: 5.0526, huber: 4.6078, swd: 15.0850, ept: 50.2558
      Epoch 29 composite train-obj: 4.526804
            No improvement (4.5860), counter 4/5
    Epoch [30/50], Train Losses: mse: 54.9236, mae: 4.9696, huber: 4.5255, swd: 14.8296, ept: 53.0895
    Epoch [30/50], Val Losses: mse: 56.3836, mae: 5.0457, huber: 4.6015, swd: 14.6784, ept: 52.4024
    Epoch [30/50], Test Losses: mse: 57.4169, mae: 5.0814, huber: 4.6375, swd: 14.8454, ept: 51.0257
      Epoch 30 composite train-obj: 4.525509
    Epoch [30/50], Test Losses: mse: 56.4975, mae: 5.0516, huber: 4.6079, swd: 15.7292, ept: 49.7808
    Best round's Test MSE: 56.4975, MAE: 5.0516, SWD: 15.7292
    Best round's Validation MSE: 55.5789, MAE: 5.0284, SWD: 15.5945
    Best round's Test verification MSE : 56.4975, MAE: 5.0516, SWD: 15.7292
    Time taken: 68.95 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq96_pred96_20250514_0126)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 56.7910 ± 0.4369
      mae: 5.0559 ± 0.0207
      huber: 4.6116 ± 0.0200
      swd: 16.1738 ± 0.3259
      ept: 50.5227 ± 0.5466
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 55.7957 ± 0.4348
      mae: 5.0254 ± 0.0210
      huber: 4.5806 ± 0.0203
      swd: 16.0042 ± 0.3019
      ept: 51.9395 ± 0.5378
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 251.17 seconds
    
    Experiment complete: DLinear_lorenz_seq96_pred96_20250514_0126
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### 96-196
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 283
    Validation Batches: 39
    Test Batches: 79
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 71.7901, mae: 6.1617, huber: 5.6883, swd: 25.5064, ept: 42.3695
    Epoch [1/50], Val Losses: mse: 66.9177, mae: 5.9393, huber: 5.4701, swd: 23.1895, ept: 50.3603
    Epoch [1/50], Test Losses: mse: 64.5512, mae: 5.7780, huber: 5.3104, swd: 22.8263, ept: 50.5235
      Epoch 1 composite train-obj: 5.688348
            Val objective improved inf → 5.4701, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 63.0743, mae: 5.6991, huber: 5.2320, swd: 21.7502, ept: 56.4856
    Epoch [2/50], Val Losses: mse: 66.3686, mae: 5.8819, huber: 5.4148, swd: 22.6922, ept: 52.7655
    Epoch [2/50], Test Losses: mse: 63.8123, mae: 5.7087, huber: 5.2441, swd: 22.2828, ept: 52.6844
      Epoch 2 composite train-obj: 5.232022
            Val objective improved 5.4701 → 5.4148, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 62.8256, mae: 5.6614, huber: 5.1960, swd: 21.1479, ept: 58.4177
    Epoch [3/50], Val Losses: mse: 66.4638, mae: 5.8747, huber: 5.4087, swd: 21.9680, ept: 55.6984
    Epoch [3/50], Test Losses: mse: 64.0755, mae: 5.7125, huber: 5.2482, swd: 21.5526, ept: 54.9501
      Epoch 3 composite train-obj: 5.195969
            Val objective improved 5.4148 → 5.4087, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 62.7186, mae: 5.6480, huber: 5.1833, swd: 20.9133, ept: 59.0620
    Epoch [4/50], Val Losses: mse: 66.4246, mae: 5.8472, huber: 5.3823, swd: 21.4533, ept: 55.2061
    Epoch [4/50], Test Losses: mse: 63.9994, mae: 5.6805, huber: 5.2174, swd: 21.0401, ept: 54.8865
      Epoch 4 composite train-obj: 5.183338
            Val objective improved 5.4087 → 5.3823, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 62.5993, mae: 5.6371, huber: 5.1730, swd: 20.7434, ept: 59.4204
    Epoch [5/50], Val Losses: mse: 66.6156, mae: 5.8567, huber: 5.3914, swd: 20.8989, ept: 56.1251
    Epoch [5/50], Test Losses: mse: 64.1303, mae: 5.6856, huber: 5.2224, swd: 20.4506, ept: 55.5982
      Epoch 5 composite train-obj: 5.173020
            No improvement (5.3914), counter 1/5
    Epoch [6/50], Train Losses: mse: 62.5460, mae: 5.6278, huber: 5.1642, swd: 20.6552, ept: 59.9545
    Epoch [6/50], Val Losses: mse: 66.4195, mae: 5.8473, huber: 5.3820, swd: 21.0409, ept: 56.5001
    Epoch [6/50], Test Losses: mse: 63.9395, mae: 5.6750, huber: 5.2120, swd: 20.6281, ept: 55.7301
      Epoch 6 composite train-obj: 5.164204
            Val objective improved 5.3823 → 5.3820, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 62.4407, mae: 5.6192, huber: 5.1561, swd: 20.5468, ept: 60.1317
    Epoch [7/50], Val Losses: mse: 66.0380, mae: 5.8175, huber: 5.3538, swd: 21.4068, ept: 56.3036
    Epoch [7/50], Test Losses: mse: 63.6262, mae: 5.6510, huber: 5.1891, swd: 20.9992, ept: 55.7273
      Epoch 7 composite train-obj: 5.156130
            Val objective improved 5.3820 → 5.3538, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 62.3929, mae: 5.6133, huber: 5.1506, swd: 20.4968, ept: 60.5003
    Epoch [8/50], Val Losses: mse: 65.8307, mae: 5.8113, huber: 5.3480, swd: 21.5014, ept: 56.1448
    Epoch [8/50], Test Losses: mse: 63.4811, mae: 5.6456, huber: 5.1837, swd: 21.0869, ept: 55.8532
      Epoch 8 composite train-obj: 5.150572
            Val objective improved 5.3538 → 5.3480, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 62.3611, mae: 5.6099, huber: 5.1474, swd: 20.4688, ept: 60.3835
    Epoch [9/50], Val Losses: mse: 66.4046, mae: 5.8301, huber: 5.3657, swd: 20.6416, ept: 56.1434
    Epoch [9/50], Test Losses: mse: 63.9082, mae: 5.6598, huber: 5.1971, swd: 20.2463, ept: 55.9741
      Epoch 9 composite train-obj: 5.147438
            No improvement (5.3657), counter 1/5
    Epoch [10/50], Train Losses: mse: 62.3549, mae: 5.6094, huber: 5.1470, swd: 20.4389, ept: 60.5321
    Epoch [10/50], Val Losses: mse: 66.5240, mae: 5.8352, huber: 5.3709, swd: 20.5438, ept: 57.9844
    Epoch [10/50], Test Losses: mse: 64.1725, mae: 5.6774, huber: 5.2144, swd: 20.1573, ept: 57.4722
      Epoch 10 composite train-obj: 5.147039
            No improvement (5.3709), counter 2/5
    Epoch [11/50], Train Losses: mse: 62.2545, mae: 5.6018, huber: 5.1399, swd: 20.4002, ept: 60.8190
    Epoch [11/50], Val Losses: mse: 65.9011, mae: 5.8185, huber: 5.3544, swd: 21.3214, ept: 56.2096
    Epoch [11/50], Test Losses: mse: 63.4407, mae: 5.6478, huber: 5.1860, swd: 20.9198, ept: 55.8614
      Epoch 11 composite train-obj: 5.139863
            No improvement (5.3544), counter 3/5
    Epoch [12/50], Train Losses: mse: 62.2073, mae: 5.5966, huber: 5.1348, swd: 20.2960, ept: 60.9618
    Epoch [12/50], Val Losses: mse: 66.2658, mae: 5.8279, huber: 5.3647, swd: 20.4331, ept: 56.4422
    Epoch [12/50], Test Losses: mse: 63.8465, mae: 5.6581, huber: 5.1969, swd: 20.0157, ept: 56.1614
      Epoch 12 composite train-obj: 5.134752
            No improvement (5.3647), counter 4/5
    Epoch [13/50], Train Losses: mse: 62.1994, mae: 5.5952, huber: 5.1336, swd: 20.2981, ept: 60.9773
    Epoch [13/50], Val Losses: mse: 66.3308, mae: 5.8159, huber: 5.3529, swd: 20.2481, ept: 56.5387
    Epoch [13/50], Test Losses: mse: 64.0504, mae: 5.6590, huber: 5.1974, swd: 19.8780, ept: 56.6186
      Epoch 13 composite train-obj: 5.133630
    Epoch [13/50], Test Losses: mse: 63.4811, mae: 5.6456, huber: 5.1837, swd: 21.0869, ept: 55.8532
    Best round's Test MSE: 63.4811, MAE: 5.6456, SWD: 21.0869
    Best round's Validation MSE: 65.8307, MAE: 5.8113, SWD: 21.5014
    Best round's Test verification MSE : 63.4811, MAE: 5.6456, SWD: 21.0869
    Time taken: 33.15 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.9258, mae: 6.1355, huber: 5.6622, swd: 25.1552, ept: 41.8107
    Epoch [1/50], Val Losses: mse: 66.8742, mae: 5.9545, huber: 5.4849, swd: 23.5039, ept: 49.6673
    Epoch [1/50], Test Losses: mse: 64.3848, mae: 5.7883, huber: 5.3203, swd: 23.0954, ept: 49.6783
      Epoch 1 composite train-obj: 5.662207
            Val objective improved inf → 5.4849, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 63.0348, mae: 5.6997, huber: 5.2323, swd: 21.6985, ept: 56.3774
    Epoch [2/50], Val Losses: mse: 67.2146, mae: 5.9128, huber: 5.4454, swd: 21.0931, ept: 53.5498
    Epoch [2/50], Test Losses: mse: 64.6788, mae: 5.7423, huber: 5.2767, swd: 20.6567, ept: 53.4298
      Epoch 2 composite train-obj: 5.232288
            Val objective improved 5.4849 → 5.4454, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 62.7415, mae: 5.6627, huber: 5.1971, swd: 21.1873, ept: 58.1750
    Epoch [3/50], Val Losses: mse: 67.5650, mae: 5.8981, huber: 5.4325, swd: 20.1259, ept: 55.2547
    Epoch [3/50], Test Losses: mse: 65.1749, mae: 5.7376, huber: 5.2737, swd: 19.6703, ept: 55.0416
      Epoch 3 composite train-obj: 5.197057
            Val objective improved 5.4454 → 5.4325, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 62.8217, mae: 5.6494, huber: 5.1847, swd: 20.7440, ept: 59.1926
    Epoch [4/50], Val Losses: mse: 65.8499, mae: 5.8392, huber: 5.3743, swd: 22.3753, ept: 54.4877
    Epoch [4/50], Test Losses: mse: 63.4600, mae: 5.6763, huber: 5.2129, swd: 21.8998, ept: 54.1776
      Epoch 4 composite train-obj: 5.184682
            Val objective improved 5.4325 → 5.3743, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 62.5070, mae: 5.6338, huber: 5.1697, swd: 20.7467, ept: 59.5644
    Epoch [5/50], Val Losses: mse: 67.4826, mae: 5.8750, huber: 5.4103, swd: 19.9916, ept: 57.6605
    Epoch [5/50], Test Losses: mse: 64.9443, mae: 5.7056, huber: 5.2426, swd: 19.5491, ept: 56.9954
      Epoch 5 composite train-obj: 5.169716
            No improvement (5.4103), counter 1/5
    Epoch [6/50], Train Losses: mse: 62.5931, mae: 5.6263, huber: 5.1628, swd: 20.4679, ept: 60.2630
    Epoch [6/50], Val Losses: mse: 66.8365, mae: 5.8544, huber: 5.3899, swd: 21.0203, ept: 54.7654
    Epoch [6/50], Test Losses: mse: 64.4495, mae: 5.6962, huber: 5.2336, swd: 20.5772, ept: 54.8200
      Epoch 6 composite train-obj: 5.162765
            No improvement (5.3899), counter 2/5
    Epoch [7/50], Train Losses: mse: 62.4911, mae: 5.6197, huber: 5.1566, swd: 20.4835, ept: 60.2655
    Epoch [7/50], Val Losses: mse: 66.1614, mae: 5.8285, huber: 5.3652, swd: 21.1310, ept: 56.4223
    Epoch [7/50], Test Losses: mse: 63.7562, mae: 5.6638, huber: 5.2020, swd: 20.6828, ept: 56.2789
      Epoch 7 composite train-obj: 5.156639
            Val objective improved 5.3743 → 5.3652, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 62.3892, mae: 5.6146, huber: 5.1518, swd: 20.4440, ept: 60.3308
    Epoch [8/50], Val Losses: mse: 66.4890, mae: 5.8420, huber: 5.3778, swd: 20.6997, ept: 55.8986
    Epoch [8/50], Test Losses: mse: 63.9613, mae: 5.6704, huber: 5.2079, swd: 20.2632, ept: 55.7548
      Epoch 8 composite train-obj: 5.151814
            No improvement (5.3778), counter 1/5
    Epoch [9/50], Train Losses: mse: 62.4732, mae: 5.6109, huber: 5.1484, swd: 20.2255, ept: 60.8463
    Epoch [9/50], Val Losses: mse: 65.6213, mae: 5.8106, huber: 5.3481, swd: 21.7526, ept: 56.5449
    Epoch [9/50], Test Losses: mse: 63.3058, mae: 5.6510, huber: 5.1897, swd: 21.2928, ept: 56.0924
      Epoch 9 composite train-obj: 5.148409
            Val objective improved 5.3652 → 5.3481, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 62.2611, mae: 5.6060, huber: 5.1438, swd: 20.4029, ept: 60.5735
    Epoch [10/50], Val Losses: mse: 66.0922, mae: 5.8335, huber: 5.3691, swd: 20.9879, ept: 56.7491
    Epoch [10/50], Test Losses: mse: 63.8034, mae: 5.6737, huber: 5.2110, swd: 20.5271, ept: 56.6880
      Epoch 10 composite train-obj: 5.143762
            No improvement (5.3691), counter 1/5
    Epoch [11/50], Train Losses: mse: 62.2783, mae: 5.6032, huber: 5.1411, swd: 20.3022, ept: 60.7885
    Epoch [11/50], Val Losses: mse: 66.5869, mae: 5.8360, huber: 5.3723, swd: 20.3376, ept: 57.7188
    Epoch [11/50], Test Losses: mse: 64.1528, mae: 5.6706, huber: 5.2089, swd: 19.9086, ept: 57.2928
      Epoch 11 composite train-obj: 5.141072
            No improvement (5.3723), counter 2/5
    Epoch [12/50], Train Losses: mse: 62.2414, mae: 5.5978, huber: 5.1361, swd: 20.2099, ept: 60.9663
    Epoch [12/50], Val Losses: mse: 66.1259, mae: 5.8250, huber: 5.3617, swd: 20.4088, ept: 55.5274
    Epoch [12/50], Test Losses: mse: 63.7598, mae: 5.6578, huber: 5.1964, swd: 19.9433, ept: 55.3482
      Epoch 12 composite train-obj: 5.136086
            No improvement (5.3617), counter 3/5
    Epoch [13/50], Train Losses: mse: 62.2469, mae: 5.5965, huber: 5.1347, swd: 20.1315, ept: 60.9314
    Epoch [13/50], Val Losses: mse: 65.4911, mae: 5.7975, huber: 5.3345, swd: 21.0708, ept: 56.5806
    Epoch [13/50], Test Losses: mse: 63.3216, mae: 5.6397, huber: 5.1788, swd: 20.6477, ept: 56.1928
      Epoch 13 composite train-obj: 5.134745
            Val objective improved 5.3481 → 5.3345, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 62.1012, mae: 5.5905, huber: 5.1290, swd: 20.1924, ept: 60.9265
    Epoch [14/50], Val Losses: mse: 65.9038, mae: 5.8072, huber: 5.3447, swd: 20.8415, ept: 56.9202
    Epoch [14/50], Test Losses: mse: 63.3804, mae: 5.6358, huber: 5.1753, swd: 20.4165, ept: 56.3967
      Epoch 14 composite train-obj: 5.129007
            No improvement (5.3447), counter 1/5
    Epoch [15/50], Train Losses: mse: 62.1309, mae: 5.5902, huber: 5.1289, swd: 20.1709, ept: 60.9086
    Epoch [15/50], Val Losses: mse: 66.2898, mae: 5.8267, huber: 5.3634, swd: 20.2474, ept: 58.0682
    Epoch [15/50], Test Losses: mse: 63.8351, mae: 5.6591, huber: 5.1977, swd: 19.8136, ept: 57.8108
      Epoch 15 composite train-obj: 5.128929
            No improvement (5.3634), counter 2/5
    Epoch [16/50], Train Losses: mse: 62.1647, mae: 5.5883, huber: 5.1272, swd: 20.0565, ept: 61.2279
    Epoch [16/50], Val Losses: mse: 65.9743, mae: 5.8093, huber: 5.3465, swd: 20.7371, ept: 57.8947
    Epoch [16/50], Test Losses: mse: 63.7117, mae: 5.6499, huber: 5.1887, swd: 20.3084, ept: 57.4021
      Epoch 16 composite train-obj: 5.127192
            No improvement (5.3465), counter 3/5
    Epoch [17/50], Train Losses: mse: 62.0959, mae: 5.5847, huber: 5.1238, swd: 20.1058, ept: 61.0860
    Epoch [17/50], Val Losses: mse: 65.7972, mae: 5.8138, huber: 5.3517, swd: 21.1282, ept: 57.2527
    Epoch [17/50], Test Losses: mse: 63.5116, mae: 5.6551, huber: 5.1944, swd: 20.6490, ept: 56.4588
      Epoch 17 composite train-obj: 5.123782
            No improvement (5.3517), counter 4/5
    Epoch [18/50], Train Losses: mse: 62.0112, mae: 5.5824, huber: 5.1215, swd: 20.1419, ept: 61.1620
    Epoch [18/50], Val Losses: mse: 65.5015, mae: 5.7971, huber: 5.3353, swd: 20.9350, ept: 57.4304
    Epoch [18/50], Test Losses: mse: 63.2235, mae: 5.6391, huber: 5.1787, swd: 20.5044, ept: 57.0257
      Epoch 18 composite train-obj: 5.121492
    Epoch [18/50], Test Losses: mse: 63.3216, mae: 5.6397, huber: 5.1788, swd: 20.6477, ept: 56.1928
    Best round's Test MSE: 63.3216, MAE: 5.6397, SWD: 20.6477
    Best round's Validation MSE: 65.4911, MAE: 5.7975, SWD: 21.0708
    Best round's Test verification MSE : 63.3216, MAE: 5.6397, SWD: 20.6477
    Time taken: 45.79 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 71.8526, mae: 6.1624, huber: 5.6891, swd: 23.7018, ept: 41.6873
    Epoch [1/50], Val Losses: mse: 66.8937, mae: 5.9376, huber: 5.4680, swd: 21.6699, ept: 50.4482
    Epoch [1/50], Test Losses: mse: 64.5798, mae: 5.7784, huber: 5.3106, swd: 21.3838, ept: 50.5747
      Epoch 1 composite train-obj: 5.689093
            Val objective improved inf → 5.4680, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 63.1144, mae: 5.7007, huber: 5.2335, swd: 20.4756, ept: 56.5025
    Epoch [2/50], Val Losses: mse: 66.5615, mae: 5.8927, huber: 5.4258, swd: 21.2418, ept: 53.0821
    Epoch [2/50], Test Losses: mse: 64.0995, mae: 5.7269, huber: 5.2620, swd: 20.9460, ept: 53.0118
      Epoch 2 composite train-obj: 5.233491
            Val objective improved 5.4680 → 5.4258, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 62.7096, mae: 5.6634, huber: 5.1978, swd: 20.1884, ept: 57.9610
    Epoch [3/50], Val Losses: mse: 66.6658, mae: 5.8574, huber: 5.3913, swd: 20.5524, ept: 56.1575
    Epoch [3/50], Test Losses: mse: 64.1689, mae: 5.6872, huber: 5.2230, swd: 20.2582, ept: 55.6096
      Epoch 3 composite train-obj: 5.197776
            Val objective improved 5.4258 → 5.3913, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 62.7619, mae: 5.6488, huber: 5.1841, swd: 19.7534, ept: 59.2435
    Epoch [4/50], Val Losses: mse: 66.8459, mae: 5.8758, huber: 5.4109, swd: 20.5607, ept: 54.5645
    Epoch [4/50], Test Losses: mse: 64.2935, mae: 5.7025, huber: 5.2396, swd: 20.2354, ept: 54.5590
      Epoch 4 composite train-obj: 5.184128
            No improvement (5.4109), counter 1/5
    Epoch [5/50], Train Losses: mse: 62.6406, mae: 5.6374, huber: 5.1733, swd: 19.6731, ept: 59.5083
    Epoch [5/50], Val Losses: mse: 66.7308, mae: 5.8527, huber: 5.3890, swd: 19.8190, ept: 56.0995
    Epoch [5/50], Test Losses: mse: 64.3386, mae: 5.6869, huber: 5.2248, swd: 19.5152, ept: 55.9444
      Epoch 5 composite train-obj: 5.173321
            Val objective improved 5.3913 → 5.3890, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 62.4821, mae: 5.6261, huber: 5.1626, swd: 19.5231, ept: 60.0087
    Epoch [6/50], Val Losses: mse: 67.1493, mae: 5.8684, huber: 5.4051, swd: 19.3716, ept: 57.3554
    Epoch [6/50], Test Losses: mse: 64.7409, mae: 5.7066, huber: 5.2448, swd: 19.0591, ept: 56.8249
      Epoch 6 composite train-obj: 5.162552
            No improvement (5.4051), counter 1/5
    Epoch [7/50], Train Losses: mse: 62.5041, mae: 5.6197, huber: 5.1566, swd: 19.3479, ept: 60.3643
    Epoch [7/50], Val Losses: mse: 66.7312, mae: 5.8513, huber: 5.3869, swd: 19.5289, ept: 56.1701
    Epoch [7/50], Test Losses: mse: 64.2605, mae: 5.6842, huber: 5.2213, swd: 19.2242, ept: 56.2657
      Epoch 7 composite train-obj: 5.156581
            Val objective improved 5.3890 → 5.3869, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 62.4058, mae: 5.6153, huber: 5.1524, swd: 19.3861, ept: 60.2316
    Epoch [8/50], Val Losses: mse: 66.4683, mae: 5.8327, huber: 5.3695, swd: 19.7622, ept: 56.7625
    Epoch [8/50], Test Losses: mse: 64.0544, mae: 5.6655, huber: 5.2044, swd: 19.4805, ept: 56.8343
      Epoch 8 composite train-obj: 5.152424
            Val objective improved 5.3869 → 5.3695, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 62.3528, mae: 5.6101, huber: 5.1476, swd: 19.3260, ept: 60.4269
    Epoch [9/50], Val Losses: mse: 66.7317, mae: 5.8355, huber: 5.3721, swd: 19.1628, ept: 56.2249
    Epoch [9/50], Test Losses: mse: 64.1718, mae: 5.6635, huber: 5.2024, swd: 18.8687, ept: 56.0441
      Epoch 9 composite train-obj: 5.147575
            No improvement (5.3721), counter 1/5
    Epoch [10/50], Train Losses: mse: 62.3693, mae: 5.6051, huber: 5.1430, swd: 19.1918, ept: 60.7827
    Epoch [10/50], Val Losses: mse: 65.5266, mae: 5.7960, huber: 5.3325, swd: 20.3462, ept: 56.6328
    Epoch [10/50], Test Losses: mse: 63.2381, mae: 5.6349, huber: 5.1732, swd: 20.0817, ept: 56.1373
      Epoch 10 composite train-obj: 5.142972
            Val objective improved 5.3695 → 5.3325, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 62.3055, mae: 5.6045, huber: 5.1425, swd: 19.2465, ept: 60.6285
    Epoch [11/50], Val Losses: mse: 65.7745, mae: 5.8059, huber: 5.3423, swd: 19.7604, ept: 57.3791
    Epoch [11/50], Test Losses: mse: 63.4997, mae: 5.6464, huber: 5.1848, swd: 19.4724, ept: 56.7691
      Epoch 11 composite train-obj: 5.142465
            No improvement (5.3423), counter 1/5
    Epoch [12/50], Train Losses: mse: 62.2526, mae: 5.6004, huber: 5.1385, swd: 19.2173, ept: 60.9249
    Epoch [12/50], Val Losses: mse: 66.6872, mae: 5.8383, huber: 5.3760, swd: 18.9760, ept: 57.4894
    Epoch [12/50], Test Losses: mse: 64.2861, mae: 5.6731, huber: 5.2125, swd: 18.6637, ept: 56.9156
      Epoch 12 composite train-obj: 5.138482
            No improvement (5.3760), counter 2/5
    Epoch [13/50], Train Losses: mse: 62.1706, mae: 5.5943, huber: 5.1328, swd: 19.1616, ept: 60.8779
    Epoch [13/50], Val Losses: mse: 66.4699, mae: 5.8153, huber: 5.3526, swd: 19.3592, ept: 57.5768
    Epoch [13/50], Test Losses: mse: 64.0545, mae: 5.6490, huber: 5.1881, swd: 19.0510, ept: 57.2016
      Epoch 13 composite train-obj: 5.132759
            No improvement (5.3526), counter 3/5
    Epoch [14/50], Train Losses: mse: 62.2374, mae: 5.5933, huber: 5.1317, swd: 19.0561, ept: 61.1017
    Epoch [14/50], Val Losses: mse: 66.4121, mae: 5.8301, huber: 5.3665, swd: 19.6450, ept: 56.5727
    Epoch [14/50], Test Losses: mse: 63.9189, mae: 5.6631, huber: 5.2016, swd: 19.3756, ept: 56.6524
      Epoch 14 composite train-obj: 5.131746
            No improvement (5.3665), counter 4/5
    Epoch [15/50], Train Losses: mse: 62.0649, mae: 5.5897, huber: 5.1283, swd: 19.2266, ept: 60.9515
    Epoch [15/50], Val Losses: mse: 66.4897, mae: 5.8251, huber: 5.3633, swd: 18.5936, ept: 57.0310
    Epoch [15/50], Test Losses: mse: 64.1567, mae: 5.6597, huber: 5.2002, swd: 18.2707, ept: 56.8167
      Epoch 15 composite train-obj: 5.128347
    Epoch [15/50], Test Losses: mse: 63.2381, mae: 5.6349, huber: 5.1732, swd: 20.0817, ept: 56.1373
    Best round's Test MSE: 63.2381, MAE: 5.6349, SWD: 20.0817
    Best round's Validation MSE: 65.5266, MAE: 5.7960, SWD: 20.3462
    Best round's Test verification MSE : 63.2381, MAE: 5.6349, SWD: 20.0817
    Time taken: 37.31 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq96_pred196_20250514_0131)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 63.3469 ± 0.1008
      mae: 5.6401 ± 0.0044
      huber: 5.1786 ± 0.0043
      swd: 20.6054 ± 0.4115
      ept: 56.0611 ± 0.1487
      count: 39.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 65.6161 ± 0.1524
      mae: 5.8016 ± 0.0069
      huber: 5.3383 ± 0.0069
      swd: 20.9728 ± 0.4767
      ept: 56.4527 ± 0.2188
      count: 39.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 116.28 seconds
    
    Experiment complete: DLinear_lorenz_seq96_pred196_20250514_0131
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### 96-336
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
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
    
    Epoch [1/50], Train Losses: mse: 72.6601, mae: 6.3184, huber: 5.8423, swd: 27.4076, ept: 43.5096
    Epoch [1/50], Val Losses: mse: 68.3196, mae: 6.1428, huber: 5.6682, swd: 24.6572, ept: 51.6464
    Epoch [1/50], Test Losses: mse: 67.2442, mae: 6.0414, huber: 5.5688, swd: 24.6108, ept: 49.7518
      Epoch 1 composite train-obj: 5.842326
            Val objective improved inf → 5.6682, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 64.9749, mae: 5.9425, huber: 5.4703, swd: 24.6164, ept: 58.9034
    Epoch [2/50], Val Losses: mse: 68.0734, mae: 6.1196, huber: 5.6459, swd: 25.3336, ept: 53.2613
    Epoch [2/50], Test Losses: mse: 66.8611, mae: 6.0179, huber: 5.5459, swd: 25.3069, ept: 51.6672
      Epoch 2 composite train-obj: 5.470295
            Val objective improved 5.6682 → 5.6459, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 64.9400, mae: 5.9251, huber: 5.4538, swd: 24.2051, ept: 60.9889
    Epoch [3/50], Val Losses: mse: 68.0850, mae: 6.0939, huber: 5.6210, swd: 24.0110, ept: 58.0366
    Epoch [3/50], Test Losses: mse: 66.8843, mae: 5.9903, huber: 5.5191, swd: 23.9662, ept: 55.7410
      Epoch 3 composite train-obj: 5.453789
            Val objective improved 5.6459 → 5.6210, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 64.8364, mae: 5.9110, huber: 5.4405, swd: 24.0376, ept: 62.2125
    Epoch [4/50], Val Losses: mse: 68.1057, mae: 6.0859, huber: 5.6142, swd: 22.9892, ept: 56.7778
    Epoch [4/50], Test Losses: mse: 67.1800, mae: 5.9894, huber: 5.5192, swd: 22.9192, ept: 55.3509
      Epoch 4 composite train-obj: 5.440478
            Val objective improved 5.6210 → 5.6142, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 64.7409, mae: 5.9052, huber: 5.4349, swd: 23.8168, ept: 62.5070
    Epoch [5/50], Val Losses: mse: 68.1152, mae: 6.0987, huber: 5.6272, swd: 24.7092, ept: 57.0460
    Epoch [5/50], Test Losses: mse: 66.9586, mae: 5.9943, huber: 5.5241, swd: 24.6537, ept: 55.4508
      Epoch 5 composite train-obj: 5.434940
            No improvement (5.6272), counter 1/5
    Epoch [6/50], Train Losses: mse: 64.7023, mae: 5.8997, huber: 5.4298, swd: 23.8179, ept: 63.4535
    Epoch [6/50], Val Losses: mse: 68.3232, mae: 6.0977, huber: 5.6264, swd: 23.1309, ept: 58.7394
    Epoch [6/50], Test Losses: mse: 67.4932, mae: 6.0102, huber: 5.5399, swd: 23.0748, ept: 57.4653
      Epoch 6 composite train-obj: 5.429797
            No improvement (5.6264), counter 2/5
    Epoch [7/50], Train Losses: mse: 64.9277, mae: 5.9063, huber: 5.4365, swd: 23.5561, ept: 63.3880
    Epoch [7/50], Val Losses: mse: 67.5024, mae: 6.0929, huber: 5.6194, swd: 24.6640, ept: 56.0036
    Epoch [7/50], Test Losses: mse: 66.1977, mae: 5.9759, huber: 5.5042, swd: 24.5691, ept: 54.5513
      Epoch 7 composite train-obj: 5.436507
            No improvement (5.6194), counter 3/5
    Epoch [8/50], Train Losses: mse: 64.5202, mae: 5.8938, huber: 5.4241, swd: 23.9054, ept: 62.8034
    Epoch [8/50], Val Losses: mse: 67.9346, mae: 6.0885, huber: 5.6156, swd: 23.4999, ept: 58.8487
    Epoch [8/50], Test Losses: mse: 66.6433, mae: 5.9753, huber: 5.5039, swd: 23.4337, ept: 57.4437
      Epoch 8 composite train-obj: 5.424103
            No improvement (5.6156), counter 4/5
    Epoch [9/50], Train Losses: mse: 64.4757, mae: 5.8864, huber: 5.4171, swd: 23.7068, ept: 63.3152
    Epoch [9/50], Val Losses: mse: 68.0317, mae: 6.0860, huber: 5.6136, swd: 22.9043, ept: 60.6975
    Epoch [9/50], Test Losses: mse: 67.1628, mae: 5.9961, huber: 5.5247, swd: 22.8657, ept: 58.6500
      Epoch 9 composite train-obj: 5.417142
            Val objective improved 5.6142 → 5.6136, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 64.5799, mae: 5.8873, huber: 5.4181, swd: 23.4818, ept: 63.4324
    Epoch [10/50], Val Losses: mse: 67.7936, mae: 6.0820, huber: 5.6107, swd: 24.1437, ept: 58.2990
    Epoch [10/50], Test Losses: mse: 66.6221, mae: 5.9810, huber: 5.5109, swd: 24.1142, ept: 56.8135
      Epoch 10 composite train-obj: 5.418067
            Val objective improved 5.6136 → 5.6107, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 64.4365, mae: 5.8818, huber: 5.4127, swd: 23.5329, ept: 63.7717
    Epoch [11/50], Val Losses: mse: 69.0005, mae: 6.1074, huber: 5.6371, swd: 21.8267, ept: 59.6844
    Epoch [11/50], Test Losses: mse: 67.9363, mae: 6.0079, huber: 5.5388, swd: 21.7335, ept: 58.1634
      Epoch 11 composite train-obj: 5.412688
            No improvement (5.6371), counter 1/5
    Epoch [12/50], Train Losses: mse: 64.6082, mae: 5.8830, huber: 5.4141, swd: 23.3087, ept: 64.0014
    Epoch [12/50], Val Losses: mse: 68.2434, mae: 6.0982, huber: 5.6274, swd: 23.1502, ept: 59.1054
    Epoch [12/50], Test Losses: mse: 67.2916, mae: 6.0070, huber: 5.5370, swd: 23.0633, ept: 57.2724
      Epoch 12 composite train-obj: 5.414085
            No improvement (5.6274), counter 2/5
    Epoch [13/50], Train Losses: mse: 64.4879, mae: 5.8787, huber: 5.4098, swd: 23.3403, ept: 64.1603
    Epoch [13/50], Val Losses: mse: 67.7453, mae: 6.0591, huber: 5.5882, swd: 24.4879, ept: 60.1007
    Epoch [13/50], Test Losses: mse: 66.5660, mae: 5.9614, huber: 5.4911, swd: 24.4646, ept: 57.8985
      Epoch 13 composite train-obj: 5.409800
            Val objective improved 5.6107 → 5.5882, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 64.5465, mae: 5.8811, huber: 5.4123, swd: 23.5033, ept: 63.9577
    Epoch [14/50], Val Losses: mse: 67.3758, mae: 6.0516, huber: 5.5804, swd: 23.4426, ept: 59.7964
    Epoch [14/50], Test Losses: mse: 66.2432, mae: 5.9377, huber: 5.4688, swd: 23.3709, ept: 58.1071
      Epoch 14 composite train-obj: 5.412330
            Val objective improved 5.5882 → 5.5804, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 64.4481, mae: 5.8762, huber: 5.4076, swd: 23.4795, ept: 63.8169
    Epoch [15/50], Val Losses: mse: 67.1768, mae: 6.0572, huber: 5.5855, swd: 23.9148, ept: 58.6444
    Epoch [15/50], Test Losses: mse: 66.0442, mae: 5.9540, huber: 5.4839, swd: 23.8591, ept: 57.1696
      Epoch 15 composite train-obj: 5.407579
            No improvement (5.5855), counter 1/5
    Epoch [16/50], Train Losses: mse: 64.4337, mae: 5.8810, huber: 5.4123, swd: 23.5447, ept: 63.5292
    Epoch [16/50], Val Losses: mse: 68.0860, mae: 6.0843, huber: 5.6139, swd: 23.1062, ept: 57.9130
    Epoch [16/50], Test Losses: mse: 67.0352, mae: 5.9804, huber: 5.5115, swd: 22.9740, ept: 55.6433
      Epoch 16 composite train-obj: 5.412292
            No improvement (5.6139), counter 2/5
    Epoch [17/50], Train Losses: mse: 64.3952, mae: 5.8711, huber: 5.4025, swd: 23.2706, ept: 64.2134
    Epoch [17/50], Val Losses: mse: 67.3731, mae: 6.0579, huber: 5.5870, swd: 23.4891, ept: 58.9483
    Epoch [17/50], Test Losses: mse: 66.1695, mae: 5.9445, huber: 5.4752, swd: 23.3962, ept: 56.7823
      Epoch 17 composite train-obj: 5.402532
            No improvement (5.5870), counter 3/5
    Epoch [18/50], Train Losses: mse: 64.5370, mae: 5.8801, huber: 5.4116, swd: 23.2865, ept: 63.9021
    Epoch [18/50], Val Losses: mse: 67.5395, mae: 6.0741, huber: 5.6026, swd: 23.7459, ept: 57.9226
    Epoch [18/50], Test Losses: mse: 66.6742, mae: 5.9763, huber: 5.5068, swd: 23.6683, ept: 56.2495
      Epoch 18 composite train-obj: 5.411593
            No improvement (5.6026), counter 4/5
    Epoch [19/50], Train Losses: mse: 64.5312, mae: 5.8813, huber: 5.4128, swd: 23.5012, ept: 63.9023
    Epoch [19/50], Val Losses: mse: 66.9915, mae: 6.0464, huber: 5.5757, swd: 24.6044, ept: 59.4197
    Epoch [19/50], Test Losses: mse: 65.7784, mae: 5.9388, huber: 5.4691, swd: 24.5448, ept: 57.6144
      Epoch 19 composite train-obj: 5.412803
            Val objective improved 5.5804 → 5.5757, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 64.4438, mae: 5.8797, huber: 5.4113, swd: 23.6811, ept: 63.9704
    Epoch [20/50], Val Losses: mse: 66.9341, mae: 6.0608, huber: 5.5883, swd: 24.5341, ept: 58.6378
    Epoch [20/50], Test Losses: mse: 65.9130, mae: 5.9563, huber: 5.4852, swd: 24.4601, ept: 56.9368
      Epoch 20 composite train-obj: 5.411277
            No improvement (5.5883), counter 1/5
    Epoch [21/50], Train Losses: mse: 64.1765, mae: 5.8697, huber: 5.4013, swd: 23.6091, ept: 63.7047
    Epoch [21/50], Val Losses: mse: 67.7015, mae: 6.0505, huber: 5.5797, swd: 22.6029, ept: 60.8645
    Epoch [21/50], Test Losses: mse: 66.7463, mae: 5.9587, huber: 5.4893, swd: 22.5978, ept: 58.6740
      Epoch 21 composite train-obj: 5.401291
            No improvement (5.5797), counter 2/5
    Epoch [22/50], Train Losses: mse: 64.6231, mae: 5.8819, huber: 5.4136, swd: 23.3621, ept: 64.3486
    Epoch [22/50], Val Losses: mse: 67.0213, mae: 6.0394, huber: 5.5671, swd: 23.5658, ept: 60.3211
    Epoch [22/50], Test Losses: mse: 65.8902, mae: 5.9356, huber: 5.4647, swd: 23.5854, ept: 58.7718
      Epoch 22 composite train-obj: 5.413645
            Val objective improved 5.5757 → 5.5671, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 64.3982, mae: 5.8815, huber: 5.4132, swd: 23.5354, ept: 63.9806
    Epoch [23/50], Val Losses: mse: 66.8255, mae: 6.0226, huber: 5.5514, swd: 23.9942, ept: 59.2974
    Epoch [23/50], Test Losses: mse: 65.8583, mae: 5.9249, huber: 5.4549, swd: 23.9648, ept: 57.9447
      Epoch 23 composite train-obj: 5.413154
            Val objective improved 5.5671 → 5.5514, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 64.1314, mae: 5.8611, huber: 5.3929, swd: 23.4165, ept: 64.1764
    Epoch [24/50], Val Losses: mse: 68.2523, mae: 6.0711, huber: 5.6014, swd: 22.0794, ept: 59.6161
    Epoch [24/50], Test Losses: mse: 67.5098, mae: 5.9867, huber: 5.5184, swd: 21.9901, ept: 58.4117
      Epoch 24 composite train-obj: 5.392936
            No improvement (5.6014), counter 1/5
    Epoch [25/50], Train Losses: mse: 64.2644, mae: 5.8635, huber: 5.3955, swd: 23.2340, ept: 64.4206
    Epoch [25/50], Val Losses: mse: 67.7569, mae: 6.0636, huber: 5.5931, swd: 22.8440, ept: 61.3108
    Epoch [25/50], Test Losses: mse: 66.6255, mae: 5.9600, huber: 5.4905, swd: 22.8047, ept: 59.5474
      Epoch 25 composite train-obj: 5.395483
            No improvement (5.5931), counter 2/5
    Epoch [26/50], Train Losses: mse: 64.2813, mae: 5.8609, huber: 5.3929, swd: 23.2472, ept: 64.6979
    Epoch [26/50], Val Losses: mse: 67.2422, mae: 6.0303, huber: 5.5604, swd: 23.4730, ept: 60.4648
    Epoch [26/50], Test Losses: mse: 66.1828, mae: 5.9295, huber: 5.4608, swd: 23.3935, ept: 57.9760
      Epoch 26 composite train-obj: 5.392926
            No improvement (5.5604), counter 3/5
    Epoch [27/50], Train Losses: mse: 64.2614, mae: 5.8612, huber: 5.3932, swd: 23.2632, ept: 64.5878
    Epoch [27/50], Val Losses: mse: 67.4305, mae: 6.0663, huber: 5.5947, swd: 23.3067, ept: 58.8599
    Epoch [27/50], Test Losses: mse: 66.6325, mae: 5.9815, huber: 5.5108, swd: 23.2598, ept: 57.2173
      Epoch 27 composite train-obj: 5.393202
            No improvement (5.5947), counter 4/5
    Epoch [28/50], Train Losses: mse: 64.1968, mae: 5.8635, huber: 5.3954, swd: 23.3094, ept: 64.1349
    Epoch [28/50], Val Losses: mse: 67.4265, mae: 6.0459, huber: 5.5764, swd: 23.5868, ept: 58.0756
    Epoch [28/50], Test Losses: mse: 66.2574, mae: 5.9380, huber: 5.4700, swd: 23.5092, ept: 56.2207
      Epoch 28 composite train-obj: 5.395384
    Epoch [28/50], Test Losses: mse: 65.8583, mae: 5.9249, huber: 5.4549, swd: 23.9648, ept: 57.9447
    Best round's Test MSE: 65.8583, MAE: 5.9249, SWD: 23.9648
    Best round's Validation MSE: 66.8255, MAE: 6.0226, SWD: 23.9942
    Best round's Test verification MSE : 65.8583, MAE: 5.9249, SWD: 23.9648
    Time taken: 72.83 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.9100, mae: 6.3221, huber: 5.8460, swd: 27.6319, ept: 43.8870
    Epoch [1/50], Val Losses: mse: 68.6698, mae: 6.1613, huber: 5.6861, swd: 25.4718, ept: 52.2760
    Epoch [1/50], Test Losses: mse: 67.5212, mae: 6.0623, huber: 5.5882, swd: 25.4269, ept: 50.3783
      Epoch 1 composite train-obj: 5.846033
            Val objective improved inf → 5.6861, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 65.1671, mae: 5.9521, huber: 5.4798, swd: 24.8649, ept: 59.4193
    Epoch [2/50], Val Losses: mse: 67.8137, mae: 6.0981, huber: 5.6251, swd: 25.3098, ept: 55.8652
    Epoch [2/50], Test Losses: mse: 66.8493, mae: 6.0012, huber: 5.5299, swd: 25.2272, ept: 53.5437
      Epoch 2 composite train-obj: 5.479810
            Val objective improved 5.6861 → 5.6251, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 64.9062, mae: 5.9244, huber: 5.4531, swd: 24.3999, ept: 61.1263
    Epoch [3/50], Val Losses: mse: 68.0253, mae: 6.1043, huber: 5.6315, swd: 24.7222, ept: 56.0275
    Epoch [3/50], Test Losses: mse: 66.8190, mae: 5.9972, huber: 5.5257, swd: 24.6271, ept: 53.7625
      Epoch 3 composite train-obj: 5.453115
            No improvement (5.6315), counter 1/5
    Epoch [4/50], Train Losses: mse: 64.8098, mae: 5.9140, huber: 5.4433, swd: 24.2527, ept: 62.0099
    Epoch [4/50], Val Losses: mse: 68.5327, mae: 6.1048, huber: 5.6321, swd: 23.5011, ept: 57.9921
    Epoch [4/50], Test Losses: mse: 67.5960, mae: 6.0140, huber: 5.5426, swd: 23.4168, ept: 56.3486
      Epoch 4 composite train-obj: 5.443294
            No improvement (5.6321), counter 2/5
    Epoch [5/50], Train Losses: mse: 64.7896, mae: 5.9060, huber: 5.4357, swd: 24.1132, ept: 62.4388
    Epoch [5/50], Val Losses: mse: 68.0229, mae: 6.0816, huber: 5.6106, swd: 24.2510, ept: 58.6703
    Epoch [5/50], Test Losses: mse: 66.9538, mae: 5.9805, huber: 5.5105, swd: 24.1521, ept: 56.9922
      Epoch 5 composite train-obj: 5.435746
            Val objective improved 5.6251 → 5.6106, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 64.7661, mae: 5.9040, huber: 5.4340, swd: 24.0864, ept: 63.1540
    Epoch [6/50], Val Losses: mse: 68.0440, mae: 6.1035, huber: 5.6309, swd: 24.1704, ept: 58.2024
    Epoch [6/50], Test Losses: mse: 66.9301, mae: 5.9926, huber: 5.5218, swd: 24.0931, ept: 56.3789
      Epoch 6 composite train-obj: 5.434030
            No improvement (5.6309), counter 1/5
    Epoch [7/50], Train Losses: mse: 64.6263, mae: 5.8942, huber: 5.4245, swd: 23.9627, ept: 63.2712
    Epoch [7/50], Val Losses: mse: 68.2097, mae: 6.0894, huber: 5.6184, swd: 23.4355, ept: 58.3299
    Epoch [7/50], Test Losses: mse: 67.3140, mae: 6.0014, huber: 5.5319, swd: 23.3420, ept: 56.6121
      Epoch 7 composite train-obj: 5.424525
            No improvement (5.6184), counter 2/5
    Epoch [8/50], Train Losses: mse: 64.6050, mae: 5.8922, huber: 5.4227, swd: 23.8918, ept: 63.6149
    Epoch [8/50], Val Losses: mse: 68.7400, mae: 6.1256, huber: 5.6542, swd: 22.9540, ept: 58.5429
    Epoch [8/50], Test Losses: mse: 67.8928, mae: 6.0410, huber: 5.5707, swd: 22.8528, ept: 56.9652
      Epoch 8 composite train-obj: 5.422735
            No improvement (5.6542), counter 3/5
    Epoch [9/50], Train Losses: mse: 64.6467, mae: 5.8896, huber: 5.4203, swd: 23.7381, ept: 63.9838
    Epoch [9/50], Val Losses: mse: 67.9310, mae: 6.0786, huber: 5.6069, swd: 23.5364, ept: 58.2736
    Epoch [9/50], Test Losses: mse: 66.5092, mae: 5.9567, huber: 5.4866, swd: 23.4634, ept: 56.7238
      Epoch 9 composite train-obj: 5.420277
            Val objective improved 5.6106 → 5.6069, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 64.6004, mae: 5.8881, huber: 5.4189, swd: 23.6501, ept: 63.5108
    Epoch [10/50], Val Losses: mse: 67.7633, mae: 6.0718, huber: 5.5993, swd: 24.0986, ept: 58.1790
    Epoch [10/50], Test Losses: mse: 66.5859, mae: 5.9633, huber: 5.4923, swd: 23.9928, ept: 57.0472
      Epoch 10 composite train-obj: 5.418856
            Val objective improved 5.6069 → 5.5993, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 64.6432, mae: 5.8892, huber: 5.4200, swd: 23.6762, ept: 63.7735
    Epoch [11/50], Val Losses: mse: 68.1220, mae: 6.0892, huber: 5.6174, swd: 23.0542, ept: 58.7637
    Epoch [11/50], Test Losses: mse: 67.2392, mae: 5.9911, huber: 5.5208, swd: 22.9239, ept: 57.1291
      Epoch 11 composite train-obj: 5.420006
            No improvement (5.6174), counter 1/5
    Epoch [12/50], Train Losses: mse: 64.5236, mae: 5.8823, huber: 5.4133, swd: 23.6947, ept: 63.7943
    Epoch [12/50], Val Losses: mse: 68.1351, mae: 6.0788, huber: 5.6086, swd: 22.5556, ept: 58.7030
    Epoch [12/50], Test Losses: mse: 67.4371, mae: 5.9985, huber: 5.5294, swd: 22.4029, ept: 56.4232
      Epoch 12 composite train-obj: 5.413298
            No improvement (5.6086), counter 2/5
    Epoch [13/50], Train Losses: mse: 64.5197, mae: 5.8799, huber: 5.4111, swd: 23.4874, ept: 63.4753
    Epoch [13/50], Val Losses: mse: 68.2432, mae: 6.0792, huber: 5.6092, swd: 23.6505, ept: 58.5623
    Epoch [13/50], Test Losses: mse: 67.0877, mae: 5.9780, huber: 5.5092, swd: 23.5379, ept: 56.9760
      Epoch 13 composite train-obj: 5.411073
            No improvement (5.6092), counter 3/5
    Epoch [14/50], Train Losses: mse: 64.6062, mae: 5.8856, huber: 5.4168, swd: 23.6215, ept: 63.8559
    Epoch [14/50], Val Losses: mse: 67.7058, mae: 6.0715, huber: 5.5999, swd: 23.8754, ept: 60.1550
    Epoch [14/50], Test Losses: mse: 66.5428, mae: 5.9594, huber: 5.4896, swd: 23.8031, ept: 58.2686
      Epoch 14 composite train-obj: 5.416784
            No improvement (5.5999), counter 4/5
    Epoch [15/50], Train Losses: mse: 64.4968, mae: 5.8811, huber: 5.4125, swd: 23.6941, ept: 63.6801
    Epoch [15/50], Val Losses: mse: 67.6508, mae: 6.0663, huber: 5.5950, swd: 23.7595, ept: 59.6604
    Epoch [15/50], Test Losses: mse: 66.5326, mae: 5.9596, huber: 5.4899, swd: 23.6773, ept: 57.3358
      Epoch 15 composite train-obj: 5.412491
            Val objective improved 5.5993 → 5.5950, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 64.3921, mae: 5.8745, huber: 5.4057, swd: 23.6126, ept: 63.9587
    Epoch [16/50], Val Losses: mse: 67.8367, mae: 6.0760, huber: 5.6040, swd: 23.9199, ept: 59.0864
    Epoch [16/50], Test Losses: mse: 66.8577, mae: 5.9830, huber: 5.5121, swd: 23.8354, ept: 56.8550
      Epoch 16 composite train-obj: 5.405748
            No improvement (5.6040), counter 1/5
    Epoch [17/50], Train Losses: mse: 64.3657, mae: 5.8731, huber: 5.4045, swd: 23.6655, ept: 63.8399
    Epoch [17/50], Val Losses: mse: 68.0191, mae: 6.0810, huber: 5.6109, swd: 24.6860, ept: 58.8138
    Epoch [17/50], Test Losses: mse: 66.9659, mae: 5.9869, huber: 5.5182, swd: 24.6195, ept: 56.8190
      Epoch 17 composite train-obj: 5.404478
            No improvement (5.6109), counter 2/5
    Epoch [18/50], Train Losses: mse: 64.5222, mae: 5.8775, huber: 5.4091, swd: 23.4871, ept: 64.3840
    Epoch [18/50], Val Losses: mse: 67.1296, mae: 6.0511, huber: 5.5792, swd: 25.0098, ept: 57.4045
    Epoch [18/50], Test Losses: mse: 65.6369, mae: 5.9238, huber: 5.4540, swd: 24.9381, ept: 56.7555
      Epoch 18 composite train-obj: 5.409098
            Val objective improved 5.5950 → 5.5792, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 64.2439, mae: 5.8672, huber: 5.3988, swd: 23.6671, ept: 64.3151
    Epoch [19/50], Val Losses: mse: 68.7155, mae: 6.1093, huber: 5.6385, swd: 22.4511, ept: 60.4249
    Epoch [19/50], Test Losses: mse: 67.7904, mae: 6.0176, huber: 5.5476, swd: 22.3838, ept: 59.0331
      Epoch 19 composite train-obj: 5.398789
            No improvement (5.6385), counter 1/5
    Epoch [20/50], Train Losses: mse: 64.3898, mae: 5.8700, huber: 5.4016, swd: 23.4118, ept: 64.4935
    Epoch [20/50], Val Losses: mse: 67.3988, mae: 6.0430, huber: 5.5722, swd: 24.2580, ept: 60.4640
    Epoch [20/50], Test Losses: mse: 66.2677, mae: 5.9413, huber: 5.4722, swd: 24.2009, ept: 58.2108
      Epoch 20 composite train-obj: 5.401611
            Val objective improved 5.5792 → 5.5722, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 64.3547, mae: 5.8674, huber: 5.3991, swd: 23.4550, ept: 64.5124
    Epoch [21/50], Val Losses: mse: 67.8187, mae: 6.0776, huber: 5.6059, swd: 23.4540, ept: 60.4840
    Epoch [21/50], Test Losses: mse: 66.6499, mae: 5.9738, huber: 5.5034, swd: 23.4006, ept: 58.2951
      Epoch 21 composite train-obj: 5.399091
            No improvement (5.6059), counter 1/5
    Epoch [22/50], Train Losses: mse: 64.3363, mae: 5.8682, huber: 5.3999, swd: 23.4500, ept: 64.6401
    Epoch [22/50], Val Losses: mse: 68.4113, mae: 6.0897, huber: 5.6184, swd: 23.1190, ept: 59.6452
    Epoch [22/50], Test Losses: mse: 67.3383, mae: 5.9997, huber: 5.5290, swd: 23.0654, ept: 58.0442
      Epoch 22 composite train-obj: 5.399915
            No improvement (5.6184), counter 2/5
    Epoch [23/50], Train Losses: mse: 64.4055, mae: 5.8732, huber: 5.4049, swd: 23.4967, ept: 64.4101
    Epoch [23/50], Val Losses: mse: 68.5685, mae: 6.1162, huber: 5.6447, swd: 22.5452, ept: 60.7039
    Epoch [23/50], Test Losses: mse: 67.6444, mae: 6.0316, huber: 5.5609, swd: 22.4521, ept: 58.6186
      Epoch 23 composite train-obj: 5.404938
            No improvement (5.6447), counter 3/5
    Epoch [24/50], Train Losses: mse: 64.5151, mae: 5.8714, huber: 5.4033, swd: 23.3118, ept: 64.3139
    Epoch [24/50], Val Losses: mse: 67.7014, mae: 6.0598, huber: 5.5890, swd: 23.2168, ept: 60.0528
    Epoch [24/50], Test Losses: mse: 66.4376, mae: 5.9505, huber: 5.4812, swd: 23.1124, ept: 58.4559
      Epoch 24 composite train-obj: 5.403317
            No improvement (5.5890), counter 4/5
    Epoch [25/50], Train Losses: mse: 64.2309, mae: 5.8638, huber: 5.3958, swd: 23.6068, ept: 64.0200
    Epoch [25/50], Val Losses: mse: 67.3344, mae: 6.0535, huber: 5.5818, swd: 23.7481, ept: 59.8211
    Epoch [25/50], Test Losses: mse: 66.2703, mae: 5.9494, huber: 5.4796, swd: 23.6915, ept: 58.6959
      Epoch 25 composite train-obj: 5.395756
    Epoch [25/50], Test Losses: mse: 66.2677, mae: 5.9413, huber: 5.4722, swd: 24.2009, ept: 58.2108
    Best round's Test MSE: 66.2677, MAE: 5.9413, SWD: 24.2009
    Best round's Validation MSE: 67.3988, MAE: 6.0430, SWD: 24.2580
    Best round's Test verification MSE : 66.2677, MAE: 5.9413, SWD: 24.2009
    Time taken: 65.82 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.8505, mae: 6.3192, huber: 5.8431, swd: 27.2625, ept: 43.7065
    Epoch [1/50], Val Losses: mse: 67.9620, mae: 6.1207, huber: 5.6465, swd: 25.3181, ept: 53.2893
    Epoch [1/50], Test Losses: mse: 67.0212, mae: 6.0353, huber: 5.5624, swd: 25.2146, ept: 51.7713
      Epoch 1 composite train-obj: 5.843145
            Val objective improved inf → 5.6465, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 65.0008, mae: 5.9426, huber: 5.4705, swd: 24.7563, ept: 58.9670
    Epoch [2/50], Val Losses: mse: 68.1974, mae: 6.1177, huber: 5.6444, swd: 23.9529, ept: 56.6800
    Epoch [2/50], Test Losses: mse: 67.2670, mae: 6.0254, huber: 5.5535, swd: 23.8179, ept: 54.5880
      Epoch 2 composite train-obj: 5.470462
            Val objective improved 5.6465 → 5.6444, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 64.9467, mae: 5.9257, huber: 5.4545, swd: 24.1936, ept: 60.8222
    Epoch [3/50], Val Losses: mse: 68.5331, mae: 6.1222, huber: 5.6493, swd: 23.8033, ept: 55.1455
    Epoch [3/50], Test Losses: mse: 67.3286, mae: 6.0203, huber: 5.5486, swd: 23.6497, ept: 53.9311
      Epoch 3 composite train-obj: 5.454529
            No improvement (5.6493), counter 1/5
    Epoch [4/50], Train Losses: mse: 64.7965, mae: 5.9106, huber: 5.4401, swd: 23.8853, ept: 62.1149
    Epoch [4/50], Val Losses: mse: 67.5956, mae: 6.0790, huber: 5.6065, swd: 23.8930, ept: 58.5665
    Epoch [4/50], Test Losses: mse: 66.9093, mae: 5.9991, huber: 5.5282, swd: 23.7539, ept: 56.3325
      Epoch 4 composite train-obj: 5.440082
            Val objective improved 5.6444 → 5.6065, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 64.6658, mae: 5.9005, huber: 5.4303, swd: 23.6774, ept: 62.9196
    Epoch [5/50], Val Losses: mse: 69.1808, mae: 6.1507, huber: 5.6783, swd: 22.7938, ept: 58.1385
    Epoch [5/50], Test Losses: mse: 68.2229, mae: 6.0555, huber: 5.5841, swd: 22.6137, ept: 56.5378
      Epoch 5 composite train-obj: 5.430313
            No improvement (5.6783), counter 1/5
    Epoch [6/50], Train Losses: mse: 64.7901, mae: 5.9003, huber: 5.4304, swd: 23.5329, ept: 63.2791
    Epoch [6/50], Val Losses: mse: 67.5210, mae: 6.0842, huber: 5.6109, swd: 24.3360, ept: 57.3562
    Epoch [6/50], Test Losses: mse: 66.5746, mae: 5.9893, huber: 5.5177, swd: 24.1663, ept: 55.0679
      Epoch 6 composite train-obj: 5.430383
            No improvement (5.6109), counter 2/5
    Epoch [7/50], Train Losses: mse: 64.6413, mae: 5.8964, huber: 5.4266, swd: 23.7116, ept: 62.8195
    Epoch [7/50], Val Losses: mse: 67.6364, mae: 6.0751, huber: 5.6032, swd: 23.8913, ept: 59.3989
    Epoch [7/50], Test Losses: mse: 66.6414, mae: 5.9735, huber: 5.5029, swd: 23.7117, ept: 57.4648
      Epoch 7 composite train-obj: 5.426551
            Val objective improved 5.6065 → 5.6032, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 64.5624, mae: 5.8918, huber: 5.4222, swd: 23.7360, ept: 63.4910
    Epoch [8/50], Val Losses: mse: 67.8925, mae: 6.0724, huber: 5.6017, swd: 23.3256, ept: 56.5709
    Epoch [8/50], Test Losses: mse: 67.0928, mae: 5.9862, huber: 5.5167, swd: 23.1264, ept: 55.0046
      Epoch 8 composite train-obj: 5.422232
            Val objective improved 5.6032 → 5.6017, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 64.6224, mae: 5.8913, huber: 5.4220, swd: 23.4816, ept: 62.9460
    Epoch [9/50], Val Losses: mse: 67.9863, mae: 6.0833, huber: 5.6114, swd: 23.1696, ept: 58.9160
    Epoch [9/50], Test Losses: mse: 67.1519, mae: 5.9942, huber: 5.5238, swd: 23.0100, ept: 57.5911
      Epoch 9 composite train-obj: 5.422017
            No improvement (5.6114), counter 1/5
    Epoch [10/50], Train Losses: mse: 64.5522, mae: 5.8850, huber: 5.4158, swd: 23.4226, ept: 63.6936
    Epoch [10/50], Val Losses: mse: 67.4881, mae: 6.0693, huber: 5.5976, swd: 24.1188, ept: 58.4347
    Epoch [10/50], Test Losses: mse: 66.3978, mae: 5.9660, huber: 5.4956, swd: 23.9837, ept: 56.0221
      Epoch 10 composite train-obj: 5.415845
            Val objective improved 5.6017 → 5.5976, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 64.5256, mae: 5.8861, huber: 5.4170, swd: 23.5218, ept: 63.3561
    Epoch [11/50], Val Losses: mse: 68.0655, mae: 6.0848, huber: 5.6137, swd: 23.1346, ept: 59.0136
    Epoch [11/50], Test Losses: mse: 67.2155, mae: 5.9930, huber: 5.5231, swd: 22.9741, ept: 57.5633
      Epoch 11 composite train-obj: 5.417045
            No improvement (5.6137), counter 1/5
    Epoch [12/50], Train Losses: mse: 64.5591, mae: 5.8834, huber: 5.4144, swd: 23.1429, ept: 63.8025
    Epoch [12/50], Val Losses: mse: 67.1132, mae: 6.0580, huber: 5.5860, swd: 24.7192, ept: 59.5587
    Epoch [12/50], Test Losses: mse: 65.9927, mae: 5.9458, huber: 5.4755, swd: 24.5939, ept: 57.0950
      Epoch 12 composite train-obj: 5.414380
            Val objective improved 5.5976 → 5.5860, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 64.4971, mae: 5.8817, huber: 5.4127, swd: 23.4633, ept: 63.8017
    Epoch [13/50], Val Losses: mse: 67.1625, mae: 6.0503, huber: 5.5794, swd: 25.1446, ept: 59.4112
    Epoch [13/50], Test Losses: mse: 65.9064, mae: 5.9417, huber: 5.4719, swd: 25.0506, ept: 57.2420
      Epoch 13 composite train-obj: 5.412746
            Val objective improved 5.5860 → 5.5794, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 64.4568, mae: 5.8812, huber: 5.4123, swd: 23.4835, ept: 64.0123
    Epoch [14/50], Val Losses: mse: 68.0355, mae: 6.0827, huber: 5.6113, swd: 23.3843, ept: 59.6065
    Epoch [14/50], Test Losses: mse: 67.1612, mae: 5.9991, huber: 5.5287, swd: 23.2061, ept: 57.7441
      Epoch 14 composite train-obj: 5.412296
            No improvement (5.6113), counter 1/5
    Epoch [15/50], Train Losses: mse: 64.4328, mae: 5.8758, huber: 5.4072, swd: 23.3845, ept: 64.0174
    Epoch [15/50], Val Losses: mse: 68.0330, mae: 6.0843, huber: 5.6136, swd: 23.4020, ept: 55.8213
    Epoch [15/50], Test Losses: mse: 66.9228, mae: 5.9855, huber: 5.5161, swd: 23.2072, ept: 54.6095
      Epoch 15 composite train-obj: 5.407171
            No improvement (5.6136), counter 2/5
    Epoch [16/50], Train Losses: mse: 64.4211, mae: 5.8765, huber: 5.4080, swd: 23.2857, ept: 63.8554
    Epoch [16/50], Val Losses: mse: 68.6992, mae: 6.1220, huber: 5.6501, swd: 22.5074, ept: 60.1521
    Epoch [16/50], Test Losses: mse: 67.7604, mae: 6.0310, huber: 5.5602, swd: 22.3704, ept: 57.7240
      Epoch 16 composite train-obj: 5.408017
            No improvement (5.6501), counter 3/5
    Epoch [17/50], Train Losses: mse: 64.4263, mae: 5.8728, huber: 5.4043, swd: 23.2695, ept: 64.3860
    Epoch [17/50], Val Losses: mse: 68.4839, mae: 6.0904, huber: 5.6199, swd: 22.2457, ept: 59.6854
    Epoch [17/50], Test Losses: mse: 67.3565, mae: 5.9869, huber: 5.5179, swd: 22.0960, ept: 57.8730
      Epoch 17 composite train-obj: 5.404293
            No improvement (5.6199), counter 4/5
    Epoch [18/50], Train Losses: mse: 64.4028, mae: 5.8746, huber: 5.4061, swd: 23.2003, ept: 64.0038
    Epoch [18/50], Val Losses: mse: 67.8289, mae: 6.0611, huber: 5.5910, swd: 23.0377, ept: 61.3489
    Epoch [18/50], Test Losses: mse: 66.7993, mae: 5.9624, huber: 5.4939, swd: 22.8840, ept: 59.1409
      Epoch 18 composite train-obj: 5.406112
    Epoch [18/50], Test Losses: mse: 65.9064, mae: 5.9417, huber: 5.4719, swd: 25.0506, ept: 57.2420
    Best round's Test MSE: 65.9064, MAE: 5.9417, SWD: 25.0506
    Best round's Validation MSE: 67.1625, MAE: 6.0503, SWD: 25.1446
    Best round's Test verification MSE : 65.9064, MAE: 5.9417, SWD: 25.0506
    Time taken: 49.14 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq96_pred336_20250514_0133)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 66.0108 ± 0.1827
      mae: 5.9360 ± 0.0078
      huber: 5.4663 ± 0.0081
      swd: 24.4054 ± 0.4662
      ept: 57.7992 ± 0.4087
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 67.1289 ± 0.2353
      mae: 6.0386 ± 0.0117
      huber: 5.5677 ± 0.0119
      swd: 24.4656 ± 0.4921
      ept: 59.7242 ± 0.5252
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 187.81 seconds
    
    Experiment complete: DLinear_lorenz_seq96_pred336_20250514_0133
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### 96-720
##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
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
    Train set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 279
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: lorenz
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 279
    Validation Batches: 35
    Test Batches: 75
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 76.1161, mae: 6.5923, huber: 6.1132, swd: 31.5578, ept: 43.5613
    Epoch [1/50], Val Losses: mse: 70.9527, mae: 6.4233, huber: 5.9451, swd: 28.6667, ept: 51.7996
    Epoch [1/50], Test Losses: mse: 69.8785, mae: 6.3256, huber: 5.8480, swd: 28.7265, ept: 50.8513
      Epoch 1 composite train-obj: 6.113193
            Val objective improved inf → 5.9451, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 68.1398, mae: 6.2675, huber: 5.7906, swd: 29.4243, ept: 60.3379
    Epoch [2/50], Val Losses: mse: 70.1817, mae: 6.3782, huber: 5.9011, swd: 29.2920, ept: 55.3491
    Epoch [2/50], Test Losses: mse: 68.8340, mae: 6.2631, huber: 5.7868, swd: 29.3121, ept: 53.9663
      Epoch 2 composite train-obj: 5.790583
            Val objective improved 5.9451 → 5.9011, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 67.9643, mae: 6.2529, huber: 5.7765, swd: 29.2218, ept: 62.8079
    Epoch [3/50], Val Losses: mse: 70.4135, mae: 6.3827, huber: 5.9057, swd: 27.9781, ept: 58.2149
    Epoch [3/50], Test Losses: mse: 69.3027, mae: 6.2809, huber: 5.8046, swd: 27.9079, ept: 56.1901
      Epoch 3 composite train-obj: 5.776476
            No improvement (5.9057), counter 1/5
    Epoch [4/50], Train Losses: mse: 68.0446, mae: 6.2546, huber: 5.7785, swd: 28.9896, ept: 63.3593
    Epoch [4/50], Val Losses: mse: 69.4937, mae: 6.3371, huber: 5.8598, swd: 29.5709, ept: 57.4593
    Epoch [4/50], Test Losses: mse: 68.4995, mae: 6.2405, huber: 5.7639, swd: 29.3806, ept: 55.6921
      Epoch 4 composite train-obj: 5.778512
            Val objective improved 5.9011 → 5.8598, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 67.8535, mae: 6.2397, huber: 5.7637, swd: 28.9862, ept: 64.5380
    Epoch [5/50], Val Losses: mse: 70.8541, mae: 6.3981, huber: 5.9216, swd: 27.9924, ept: 57.8065
    Epoch [5/50], Test Losses: mse: 69.5681, mae: 6.2869, huber: 5.8109, swd: 27.8648, ept: 56.0465
      Epoch 5 composite train-obj: 5.763698
            No improvement (5.9216), counter 1/5
    Epoch [6/50], Train Losses: mse: 67.9264, mae: 6.2444, huber: 5.7686, swd: 28.6777, ept: 64.2212
    Epoch [6/50], Val Losses: mse: 70.8276, mae: 6.3959, huber: 5.9194, swd: 27.9569, ept: 58.2280
    Epoch [6/50], Test Losses: mse: 69.4384, mae: 6.2790, huber: 5.8034, swd: 27.8633, ept: 56.3335
      Epoch 6 composite train-obj: 5.768604
            No improvement (5.9194), counter 2/5
    Epoch [7/50], Train Losses: mse: 67.9583, mae: 6.2434, huber: 5.7677, swd: 28.5945, ept: 65.1823
    Epoch [7/50], Val Losses: mse: 69.8743, mae: 6.3640, huber: 5.8871, swd: 29.3972, ept: 58.1336
    Epoch [7/50], Test Losses: mse: 68.5921, mae: 6.2467, huber: 5.7706, swd: 29.2753, ept: 56.1988
      Epoch 7 composite train-obj: 5.767678
            No improvement (5.8871), counter 3/5
    Epoch [8/50], Train Losses: mse: 67.8400, mae: 6.2404, huber: 5.7647, swd: 28.8261, ept: 64.6233
    Epoch [8/50], Val Losses: mse: 70.2521, mae: 6.3710, huber: 5.8937, swd: 28.5930, ept: 58.8466
    Epoch [8/50], Test Losses: mse: 69.1109, mae: 6.2694, huber: 5.7928, swd: 28.4407, ept: 57.2832
      Epoch 8 composite train-obj: 5.764711
            No improvement (5.8937), counter 4/5
    Epoch [9/50], Train Losses: mse: 67.7700, mae: 6.2377, huber: 5.7620, swd: 28.6564, ept: 64.8777
    Epoch [9/50], Val Losses: mse: 70.4465, mae: 6.3732, huber: 5.8969, swd: 29.1772, ept: 59.6918
    Epoch [9/50], Test Losses: mse: 69.2245, mae: 6.2712, huber: 5.7952, swd: 29.1257, ept: 57.4732
      Epoch 9 composite train-obj: 5.762014
    Epoch [9/50], Test Losses: mse: 68.4995, mae: 6.2405, huber: 5.7639, swd: 29.3806, ept: 55.6921
    Best round's Test MSE: 68.4995, MAE: 6.2405, SWD: 29.3806
    Best round's Validation MSE: 69.4937, MAE: 6.3371, SWD: 29.5709
    Best round's Test verification MSE : 68.4995, MAE: 6.2405, SWD: 29.3806
    Time taken: 24.98 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.4627, mae: 6.5713, huber: 6.0923, swd: 30.9430, ept: 43.9274
    Epoch [1/50], Val Losses: mse: 70.3336, mae: 6.3908, huber: 5.9126, swd: 30.1516, ept: 51.7283
    Epoch [1/50], Test Losses: mse: 69.1380, mae: 6.2848, huber: 5.8075, swd: 30.5047, ept: 50.7454
      Epoch 1 composite train-obj: 6.092290
            Val objective improved inf → 5.9126, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 68.1628, mae: 6.2711, huber: 5.7940, swd: 29.2023, ept: 61.0920
    Epoch [2/50], Val Losses: mse: 70.4310, mae: 6.3939, huber: 5.9166, swd: 29.1857, ept: 53.0068
    Epoch [2/50], Test Losses: mse: 69.2906, mae: 6.2916, huber: 5.8150, swd: 29.3879, ept: 52.0376
      Epoch 2 composite train-obj: 5.793995
            No improvement (5.9166), counter 1/5
    Epoch [3/50], Train Losses: mse: 67.9623, mae: 6.2524, huber: 5.7759, swd: 28.8213, ept: 62.6121
    Epoch [3/50], Val Losses: mse: 70.4161, mae: 6.3822, huber: 5.9051, swd: 28.6602, ept: 56.8391
    Epoch [3/50], Test Losses: mse: 69.3218, mae: 6.2788, huber: 5.8024, swd: 28.8409, ept: 54.9907
      Epoch 3 composite train-obj: 5.775944
            Val objective improved 5.9126 → 5.9051, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 67.9534, mae: 6.2490, huber: 5.7728, swd: 28.7772, ept: 63.5679
    Epoch [4/50], Val Losses: mse: 70.7470, mae: 6.3985, huber: 5.9214, swd: 27.8976, ept: 58.6573
    Epoch [4/50], Test Losses: mse: 69.5734, mae: 6.2954, huber: 5.8188, swd: 28.1202, ept: 56.8660
      Epoch 4 composite train-obj: 5.772836
            No improvement (5.9214), counter 1/5
    Epoch [5/50], Train Losses: mse: 67.9609, mae: 6.2461, huber: 5.7701, swd: 28.5188, ept: 64.3641
    Epoch [5/50], Val Losses: mse: 69.9988, mae: 6.3651, huber: 5.8878, swd: 29.2656, ept: 57.6527
    Epoch [5/50], Test Losses: mse: 68.7068, mae: 6.2513, huber: 5.7754, swd: 29.4243, ept: 56.0256
      Epoch 5 composite train-obj: 5.770059
            Val objective improved 5.9051 → 5.8878, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 67.8555, mae: 6.2401, huber: 5.7642, swd: 28.4834, ept: 64.6451
    Epoch [6/50], Val Losses: mse: 70.9647, mae: 6.4072, huber: 5.9302, swd: 27.9730, ept: 58.5913
    Epoch [6/50], Test Losses: mse: 69.5185, mae: 6.2865, huber: 5.8103, swd: 28.2031, ept: 56.8164
      Epoch 6 composite train-obj: 5.764198
            No improvement (5.9302), counter 1/5
    Epoch [7/50], Train Losses: mse: 67.9028, mae: 6.2414, huber: 5.7656, swd: 28.3183, ept: 64.8495
    Epoch [7/50], Val Losses: mse: 70.7884, mae: 6.4016, huber: 5.9244, swd: 28.3639, ept: 60.0051
    Epoch [7/50], Test Losses: mse: 69.3540, mae: 6.2876, huber: 5.8108, swd: 28.6157, ept: 58.4345
      Epoch 7 composite train-obj: 5.765586
            No improvement (5.9244), counter 2/5
    Epoch [8/50], Train Losses: mse: 67.8761, mae: 6.2379, huber: 5.7622, swd: 28.3226, ept: 65.3648
    Epoch [8/50], Val Losses: mse: 70.2203, mae: 6.3696, huber: 5.8923, swd: 29.3043, ept: 56.3015
    Epoch [8/50], Test Losses: mse: 68.9527, mae: 6.2599, huber: 5.7833, swd: 29.3879, ept: 55.0489
      Epoch 8 composite train-obj: 5.762211
            No improvement (5.8923), counter 3/5
    Epoch [9/50], Train Losses: mse: 67.9122, mae: 6.2430, huber: 5.7673, swd: 28.3479, ept: 65.0042
    Epoch [9/50], Val Losses: mse: 70.3027, mae: 6.3635, huber: 5.8871, swd: 28.0714, ept: 60.1394
    Epoch [9/50], Test Losses: mse: 69.0150, mae: 6.2576, huber: 5.7816, swd: 28.2993, ept: 57.9451
      Epoch 9 composite train-obj: 5.767256
            Val objective improved 5.8878 → 5.8871, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 67.8048, mae: 6.2329, huber: 5.7575, swd: 28.2011, ept: 65.9102
    Epoch [10/50], Val Losses: mse: 69.8258, mae: 6.3613, huber: 5.8839, swd: 29.0754, ept: 57.6754
    Epoch [10/50], Test Losses: mse: 68.7704, mae: 6.2619, huber: 5.7852, swd: 29.1649, ept: 56.7124
      Epoch 10 composite train-obj: 5.757492
            Val objective improved 5.8871 → 5.8839, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 67.7935, mae: 6.2342, huber: 5.7588, swd: 28.3580, ept: 65.0730
    Epoch [11/50], Val Losses: mse: 69.7779, mae: 6.3418, huber: 5.8653, swd: 29.0981, ept: 60.9642
    Epoch [11/50], Test Losses: mse: 68.6042, mae: 6.2390, huber: 5.7635, swd: 29.2261, ept: 58.9369
      Epoch 11 composite train-obj: 5.758795
            Val objective improved 5.8839 → 5.8653, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 67.7575, mae: 6.2320, huber: 5.7567, swd: 28.2391, ept: 65.5349
    Epoch [12/50], Val Losses: mse: 70.9175, mae: 6.4126, huber: 5.9357, swd: 28.1539, ept: 57.6946
    Epoch [12/50], Test Losses: mse: 69.5115, mae: 6.2994, huber: 5.8226, swd: 28.4250, ept: 56.5226
      Epoch 12 composite train-obj: 5.756696
            No improvement (5.9357), counter 1/5
    Epoch [13/50], Train Losses: mse: 67.7568, mae: 6.2315, huber: 5.7561, swd: 28.1727, ept: 65.7784
    Epoch [13/50], Val Losses: mse: 70.1623, mae: 6.3582, huber: 5.8816, swd: 28.4981, ept: 57.7902
    Epoch [13/50], Test Losses: mse: 68.8167, mae: 6.2435, huber: 5.7673, swd: 28.6233, ept: 56.6087
      Epoch 13 composite train-obj: 5.756114
            No improvement (5.8816), counter 2/5
    Epoch [14/50], Train Losses: mse: 67.7104, mae: 6.2289, huber: 5.7535, swd: 28.1919, ept: 65.6961
    Epoch [14/50], Val Losses: mse: 69.8911, mae: 6.3508, huber: 5.8745, swd: 28.7292, ept: 58.4134
    Epoch [14/50], Test Losses: mse: 68.7114, mae: 6.2464, huber: 5.7706, swd: 28.8570, ept: 56.5971
      Epoch 14 composite train-obj: 5.753534
            No improvement (5.8745), counter 3/5
    Epoch [15/50], Train Losses: mse: 67.6799, mae: 6.2270, huber: 5.7517, swd: 28.1922, ept: 65.8526
    Epoch [15/50], Val Losses: mse: 70.0447, mae: 6.3581, huber: 5.8816, swd: 28.6740, ept: 60.8615
    Epoch [15/50], Test Losses: mse: 68.7875, mae: 6.2488, huber: 5.7729, swd: 28.8587, ept: 58.4326
      Epoch 15 composite train-obj: 5.751730
            No improvement (5.8816), counter 4/5
    Epoch [16/50], Train Losses: mse: 67.7116, mae: 6.2257, huber: 5.7505, swd: 28.0877, ept: 66.1081
    Epoch [16/50], Val Losses: mse: 70.0415, mae: 6.3578, huber: 5.8821, swd: 28.5820, ept: 57.9080
    Epoch [16/50], Test Losses: mse: 68.8018, mae: 6.2503, huber: 5.7749, swd: 28.7396, ept: 56.1392
      Epoch 16 composite train-obj: 5.750534
    Epoch [16/50], Test Losses: mse: 68.6042, mae: 6.2390, huber: 5.7635, swd: 29.2261, ept: 58.9369
    Best round's Test MSE: 68.6042, MAE: 6.2390, SWD: 29.2261
    Best round's Validation MSE: 69.7779, MAE: 6.3418, SWD: 29.0981
    Best round's Test verification MSE : 68.6042, MAE: 6.2390, SWD: 29.2261
    Time taken: 46.30 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.9022, mae: 6.5832, huber: 6.1042, swd: 32.0715, ept: 44.7500
    Epoch [1/50], Val Losses: mse: 71.4105, mae: 6.4485, huber: 5.9699, swd: 29.1306, ept: 51.7896
    Epoch [1/50], Test Losses: mse: 70.1923, mae: 6.3464, huber: 5.8686, swd: 29.1220, ept: 50.3607
      Epoch 1 composite train-obj: 6.104158
            Val objective improved inf → 5.9699, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 68.1281, mae: 6.2669, huber: 5.7900, swd: 29.9358, ept: 60.7760
    Epoch [2/50], Val Losses: mse: 70.7680, mae: 6.4070, huber: 5.9292, swd: 28.7640, ept: 58.0105
    Epoch [2/50], Test Losses: mse: 69.4862, mae: 6.2944, huber: 5.8176, swd: 28.6414, ept: 55.8887
      Epoch 2 composite train-obj: 5.790024
            Val objective improved 5.9699 → 5.9292, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 68.0460, mae: 6.2536, huber: 5.7773, swd: 29.5290, ept: 63.0290
    Epoch [3/50], Val Losses: mse: 70.5200, mae: 6.4051, huber: 5.9274, swd: 29.8349, ept: 55.6426
    Epoch [3/50], Test Losses: mse: 69.1657, mae: 6.2929, huber: 5.8158, swd: 29.6348, ept: 54.1644
      Epoch 3 composite train-obj: 5.777318
            Val objective improved 5.9292 → 5.9274, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 67.9195, mae: 6.2490, huber: 5.7728, swd: 29.6306, ept: 63.1695
    Epoch [4/50], Val Losses: mse: 70.6122, mae: 6.3981, huber: 5.9211, swd: 29.1834, ept: 55.4859
    Epoch [4/50], Test Losses: mse: 69.4414, mae: 6.2918, huber: 5.8154, swd: 28.9026, ept: 54.2307
      Epoch 4 composite train-obj: 5.772837
            Val objective improved 5.9274 → 5.9211, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 67.9336, mae: 6.2445, huber: 5.7686, swd: 29.3086, ept: 64.1659
    Epoch [5/50], Val Losses: mse: 70.6191, mae: 6.3863, huber: 5.9095, swd: 28.1880, ept: 60.2571
    Epoch [5/50], Test Losses: mse: 69.5929, mae: 6.2884, huber: 5.8123, swd: 27.9553, ept: 58.2402
      Epoch 5 composite train-obj: 5.768582
            Val objective improved 5.9211 → 5.9095, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 67.8916, mae: 6.2409, huber: 5.7650, swd: 29.1851, ept: 65.0517
    Epoch [6/50], Val Losses: mse: 70.5537, mae: 6.3711, huber: 5.8952, swd: 28.1276, ept: 59.0150
    Epoch [6/50], Test Losses: mse: 69.4952, mae: 6.2736, huber: 5.7981, swd: 27.7737, ept: 56.8271
      Epoch 6 composite train-obj: 5.765006
            Val objective improved 5.9095 → 5.8952, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 67.9191, mae: 6.2410, huber: 5.7653, swd: 28.9522, ept: 64.7906
    Epoch [7/50], Val Losses: mse: 70.0673, mae: 6.3538, huber: 5.8771, swd: 29.2666, ept: 57.5492
    Epoch [7/50], Test Losses: mse: 68.8877, mae: 6.2521, huber: 5.7760, swd: 28.9259, ept: 55.9753
      Epoch 7 composite train-obj: 5.765291
            Val objective improved 5.8952 → 5.8771, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 67.8544, mae: 6.2386, huber: 5.7630, swd: 29.1568, ept: 65.0118
    Epoch [8/50], Val Losses: mse: 70.8751, mae: 6.3875, huber: 5.9112, swd: 28.8393, ept: 58.0045
    Epoch [8/50], Test Losses: mse: 69.7626, mae: 6.2892, huber: 5.8135, swd: 28.5545, ept: 56.4605
      Epoch 8 composite train-obj: 5.762962
            No improvement (5.9112), counter 1/5
    Epoch [9/50], Train Losses: mse: 67.8137, mae: 6.2338, huber: 5.7583, swd: 29.1071, ept: 65.5807
    Epoch [9/50], Val Losses: mse: 70.8404, mae: 6.3959, huber: 5.9191, swd: 28.3032, ept: 58.3473
    Epoch [9/50], Test Losses: mse: 69.5709, mae: 6.2860, huber: 5.8101, swd: 27.9909, ept: 55.8446
      Epoch 9 composite train-obj: 5.758275
            No improvement (5.9191), counter 2/5
    Epoch [10/50], Train Losses: mse: 67.8193, mae: 6.2353, huber: 5.7597, swd: 29.0038, ept: 65.7422
    Epoch [10/50], Val Losses: mse: 70.7202, mae: 6.3825, huber: 5.9062, swd: 28.8532, ept: 60.5194
    Epoch [10/50], Test Losses: mse: 69.4095, mae: 6.2742, huber: 5.7983, swd: 28.6291, ept: 58.1647
      Epoch 10 composite train-obj: 5.759740
            No improvement (5.9062), counter 3/5
    Epoch [11/50], Train Losses: mse: 67.7664, mae: 6.2320, huber: 5.7566, swd: 28.9699, ept: 65.7630
    Epoch [11/50], Val Losses: mse: 70.6664, mae: 6.3825, huber: 5.9059, swd: 29.3854, ept: 60.0684
    Epoch [11/50], Test Losses: mse: 69.1364, mae: 6.2612, huber: 5.7851, swd: 29.1354, ept: 58.0228
      Epoch 11 composite train-obj: 5.756619
            No improvement (5.9059), counter 4/5
    Epoch [12/50], Train Losses: mse: 67.8544, mae: 6.2367, huber: 5.7613, swd: 29.0298, ept: 65.6941
    Epoch [12/50], Val Losses: mse: 69.7841, mae: 6.3550, huber: 5.8788, swd: 29.6721, ept: 59.1289
    Epoch [12/50], Test Losses: mse: 68.5725, mae: 6.2497, huber: 5.7737, swd: 29.4077, ept: 57.3933
      Epoch 12 composite train-obj: 5.761264
    Epoch [12/50], Test Losses: mse: 68.8877, mae: 6.2521, huber: 5.7760, swd: 28.9259, ept: 55.9753
    Best round's Test MSE: 68.8877, MAE: 6.2521, SWD: 28.9259
    Best round's Validation MSE: 70.0673, MAE: 6.3538, SWD: 29.2666
    Best round's Test verification MSE : 68.8877, MAE: 6.2521, SWD: 28.9259
    Time taken: 36.90 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq96_pred720_20250514_0136)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 68.6638 ± 0.1640
      mae: 6.2439 ± 0.0058
      huber: 5.7678 ± 0.0058
      swd: 29.1775 ± 0.1888
      ept: 56.8681 ± 1.4674
      count: 35.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 69.7796 ± 0.2341
      mae: 6.3442 ± 0.0071
      huber: 5.8674 ± 0.0072
      swd: 29.3119 ± 0.1956
      ept: 58.6575 ± 1.6315
      count: 35.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 108.21 seconds
    
    Experiment complete: DLinear_lorenz_seq96_pred720_20250514_0136
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


