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
data_mgr.load_trajectory('rossler', steps=51999, dt=1e-2) # 50399
# SCALE = False
# trajectory = utils.generate_trajectory('lorenz',steps=52200, dt=1e-2) 
# trajectory = utils.generate_hyperchaotic_rossler(steps=12000, dt=1e-3)
# trajectory_2 = utils.generate_henon(steps=52000) 
```

    RosslerSystem initialized with method: rk4 on device: cuda
    
    ==================================================
    Dataset: rossler (synthetic)
    ==================================================
    Shape: torch.Size([52000, 3])
    Channels: 3
    Length: 52000
    Parameters: {'steps': 51999, 'dt': 0.01}
    
    Sample data (first 2 rows):
    tensor([[1.0000, 1.0000, 1.0000],
            [0.9802, 1.0119, 0.9559]], device='cuda:0')
    ==================================================
    




    <data_manager.DatasetManager at 0x1bc1fc32150>





# Rossler


```python
data_mgr.datasets
```

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
    channels=data_mgr.datasets['rossler']['channels'],# data_mgr.channels,              # ← number of features in your data
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

# cfg.x_to_z_delay.scale_zeroing_threshold = 1e-4
# cfg.x_to_z_deri.scale_zeroing_threshold = 1e-4
# cfg.z_to_x_main.scale_zeroing_threshold = 1e-4
# cfg.z_push_to_z.scale_zeroing_threshold = 1e-4
# cfg.z_to_y_main.scale_zeroing_threshold = 1e-4
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 2.3434, mae: 0.5442, huber: 0.3462, swd: 1.9665, ept: 91.2373
    Epoch [1/50], Val Losses: mse: 0.8194, mae: 0.2719, huber: 0.1106, swd: 0.6852, ept: 94.6999
    Epoch [1/50], Test Losses: mse: 0.7460, mae: 0.2832, huber: 0.1139, swd: 0.6106, ept: 94.6955
      Epoch 1 composite train-obj: 0.346218
            Val objective improved inf → 0.1106, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3488, mae: 0.1803, huber: 0.0575, swd: 0.2699, ept: 95.2773
    Epoch [2/50], Val Losses: mse: 0.1049, mae: 0.1374, huber: 0.0318, swd: 0.0625, ept: 95.4742
    Epoch [2/50], Test Losses: mse: 0.1137, mae: 0.1480, huber: 0.0362, swd: 0.0716, ept: 95.3813
      Epoch 2 composite train-obj: 0.057456
            Val objective improved 0.1106 → 0.0318, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0570, mae: 0.1198, huber: 0.0214, swd: 0.0351, ept: 95.7890
    Epoch [3/50], Val Losses: mse: 0.0317, mae: 0.1078, huber: 0.0144, swd: 0.0176, ept: 95.9624
    Epoch [3/50], Test Losses: mse: 0.0330, mae: 0.1106, huber: 0.0151, swd: 0.0195, ept: 95.9501
      Epoch 3 composite train-obj: 0.021408
            Val objective improved 0.0318 → 0.0144, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0221, mae: 0.0839, huber: 0.0103, swd: 0.0136, ept: 95.9724
    Epoch [4/50], Val Losses: mse: 0.0210, mae: 0.0724, huber: 0.0097, swd: 0.0144, ept: 95.9592
    Epoch [4/50], Test Losses: mse: 0.0214, mae: 0.0770, huber: 0.0101, swd: 0.0142, ept: 95.9669
      Epoch 4 composite train-obj: 0.010262
            Val objective improved 0.0144 → 0.0097, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0191, mae: 0.0802, huber: 0.0092, swd: 0.0124, ept: 95.9884
    Epoch [5/50], Val Losses: mse: 0.0077, mae: 0.0601, huber: 0.0038, swd: 0.0046, ept: 96.0000
    Epoch [5/50], Test Losses: mse: 0.0081, mae: 0.0619, huber: 0.0040, swd: 0.0048, ept: 96.0000
      Epoch 5 composite train-obj: 0.009208
            Val objective improved 0.0097 → 0.0038, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0194, mae: 0.0774, huber: 0.0091, swd: 0.0131, ept: 95.9755
    Epoch [6/50], Val Losses: mse: 0.0571, mae: 0.0827, huber: 0.0182, swd: 0.0469, ept: 95.6463
    Epoch [6/50], Test Losses: mse: 0.0554, mae: 0.0888, huber: 0.0193, swd: 0.0446, ept: 95.7058
      Epoch 6 composite train-obj: 0.009060
            No improvement (0.0182), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0128, mae: 0.0670, huber: 0.0063, swd: 0.0083, ept: 95.9984
    Epoch [7/50], Val Losses: mse: 0.0115, mae: 0.0644, huber: 0.0056, swd: 0.0088, ept: 96.0000
    Epoch [7/50], Test Losses: mse: 0.0123, mae: 0.0673, huber: 0.0060, swd: 0.0093, ept: 95.9997
      Epoch 7 composite train-obj: 0.006260
            No improvement (0.0056), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.0138, mae: 0.0689, huber: 0.0068, swd: 0.0087, ept: 95.9983
    Epoch [8/50], Val Losses: mse: 0.0152, mae: 0.0809, huber: 0.0076, swd: 0.0119, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0171, mae: 0.0865, huber: 0.0086, swd: 0.0133, ept: 96.0000
      Epoch 8 composite train-obj: 0.006783
            No improvement (0.0076), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.0130, mae: 0.0666, huber: 0.0063, swd: 0.0089, ept: 95.9959
    Epoch [9/50], Val Losses: mse: 0.0092, mae: 0.0548, huber: 0.0046, swd: 0.0071, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0100, mae: 0.0604, huber: 0.0049, swd: 0.0075, ept: 96.0000
      Epoch 9 composite train-obj: 0.006334
            No improvement (0.0046), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.0121, mae: 0.0634, huber: 0.0059, swd: 0.0080, ept: 95.9943
    Epoch [10/50], Val Losses: mse: 0.0116, mae: 0.0647, huber: 0.0057, swd: 0.0095, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0128, mae: 0.0694, huber: 0.0062, swd: 0.0104, ept: 96.0000
      Epoch 10 composite train-obj: 0.005856
    Epoch [10/50], Test Losses: mse: 0.0081, mae: 0.0619, huber: 0.0040, swd: 0.0048, ept: 96.0000
    Best round's Test MSE: 0.0081, MAE: 0.0619, SWD: 0.0048
    Best round's Validation MSE: 0.0077, MAE: 0.0601, SWD: 0.0046
    Best round's Test verification MSE : 0.0081, MAE: 0.0619, SWD: 0.0048
    Time taken: 86.77 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.1772, mae: 0.5308, huber: 0.3328, swd: 1.7790, ept: 91.5046
    Epoch [1/50], Val Losses: mse: 0.5634, mae: 0.2049, huber: 0.0763, swd: 0.5030, ept: 94.9108
    Epoch [1/50], Test Losses: mse: 0.5071, mae: 0.2129, huber: 0.0779, swd: 0.4375, ept: 94.9198
      Epoch 1 composite train-obj: 0.332757
            Val objective improved inf → 0.0763, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.1929, mae: 0.1643, huber: 0.0425, swd: 0.1460, ept: 95.5595
    Epoch [2/50], Val Losses: mse: 0.0519, mae: 0.1127, huber: 0.0196, swd: 0.0326, ept: 95.8137
    Epoch [2/50], Test Losses: mse: 0.0538, mae: 0.1181, huber: 0.0211, swd: 0.0345, ept: 95.8163
      Epoch 2 composite train-obj: 0.042502
            Val objective improved 0.0763 → 0.0196, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0394, mae: 0.1110, huber: 0.0172, swd: 0.0231, ept: 95.9290
    Epoch [3/50], Val Losses: mse: 0.0254, mae: 0.1018, huber: 0.0124, swd: 0.0131, ept: 95.9952
    Epoch [3/50], Test Losses: mse: 0.0273, mae: 0.1079, huber: 0.0133, swd: 0.0145, ept: 95.9955
      Epoch 3 composite train-obj: 0.017199
            Val objective improved 0.0196 → 0.0124, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0236, mae: 0.0889, huber: 0.0109, swd: 0.0145, ept: 95.9738
    Epoch [4/50], Val Losses: mse: 0.0242, mae: 0.1066, huber: 0.0121, swd: 0.0124, ept: 95.9994
    Epoch [4/50], Test Losses: mse: 0.0263, mae: 0.1126, huber: 0.0131, swd: 0.0136, ept: 95.9995
      Epoch 4 composite train-obj: 0.010919
            Val objective improved 0.0124 → 0.0121, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0158, mae: 0.0761, huber: 0.0077, swd: 0.0095, ept: 95.9909
    Epoch [5/50], Val Losses: mse: 0.0127, mae: 0.0726, huber: 0.0063, swd: 0.0071, ept: 96.0000
    Epoch [5/50], Test Losses: mse: 0.0137, mae: 0.0766, huber: 0.0068, swd: 0.0078, ept: 96.0000
      Epoch 5 composite train-obj: 0.007727
            Val objective improved 0.0121 → 0.0063, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0156, mae: 0.0759, huber: 0.0077, swd: 0.0097, ept: 95.9960
    Epoch [6/50], Val Losses: mse: 0.0067, mae: 0.0481, huber: 0.0034, swd: 0.0043, ept: 96.0000
    Epoch [6/50], Test Losses: mse: 0.0068, mae: 0.0506, huber: 0.0034, swd: 0.0043, ept: 96.0000
      Epoch 6 composite train-obj: 0.007659
            Val objective improved 0.0063 → 0.0034, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0141, mae: 0.0700, huber: 0.0069, swd: 0.0090, ept: 95.9959
    Epoch [7/50], Val Losses: mse: 0.0118, mae: 0.0686, huber: 0.0059, swd: 0.0062, ept: 96.0000
    Epoch [7/50], Test Losses: mse: 0.0124, mae: 0.0714, huber: 0.0062, swd: 0.0068, ept: 96.0000
      Epoch 7 composite train-obj: 0.006880
            No improvement (0.0059), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0152, mae: 0.0744, huber: 0.0075, swd: 0.0096, ept: 95.9970
    Epoch [8/50], Val Losses: mse: 0.0172, mae: 0.0862, huber: 0.0086, swd: 0.0129, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0186, mae: 0.0908, huber: 0.0092, swd: 0.0142, ept: 96.0000
      Epoch 8 composite train-obj: 0.007496
            No improvement (0.0086), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.0125, mae: 0.0665, huber: 0.0061, swd: 0.0084, ept: 95.9963
    Epoch [9/50], Val Losses: mse: 0.0075, mae: 0.0534, huber: 0.0038, swd: 0.0047, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0077, mae: 0.0561, huber: 0.0038, swd: 0.0047, ept: 96.0000
      Epoch 9 composite train-obj: 0.006118
            No improvement (0.0038), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.0099, mae: 0.0612, huber: 0.0049, swd: 0.0066, ept: 95.9992
    Epoch [10/50], Val Losses: mse: 0.0164, mae: 0.0669, huber: 0.0079, swd: 0.0121, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0180, mae: 0.0723, huber: 0.0087, swd: 0.0134, ept: 96.0000
      Epoch 10 composite train-obj: 0.004919
            No improvement (0.0079), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.0132, mae: 0.0663, huber: 0.0064, swd: 0.0087, ept: 95.9955
    Epoch [11/50], Val Losses: mse: 0.0127, mae: 0.0701, huber: 0.0064, swd: 0.0081, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0132, mae: 0.0746, huber: 0.0066, swd: 0.0086, ept: 96.0000
      Epoch 11 composite train-obj: 0.006438
    Epoch [11/50], Test Losses: mse: 0.0068, mae: 0.0506, huber: 0.0034, swd: 0.0043, ept: 96.0000
    Best round's Test MSE: 0.0068, MAE: 0.0506, SWD: 0.0043
    Best round's Validation MSE: 0.0067, MAE: 0.0481, SWD: 0.0043
    Best round's Test verification MSE : 0.0068, MAE: 0.0506, SWD: 0.0043
    Time taken: 87.32 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.3818, mae: 0.5533, huber: 0.3524, swd: 1.8439, ept: 91.2183
    Epoch [1/50], Val Losses: mse: 0.7466, mae: 0.2268, huber: 0.0959, swd: 0.6177, ept: 94.9351
    Epoch [1/50], Test Losses: mse: 0.6804, mae: 0.2363, huber: 0.0991, swd: 0.5523, ept: 94.8323
      Epoch 1 composite train-obj: 0.352396
            Val objective improved inf → 0.0959, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3074, mae: 0.1763, huber: 0.0538, swd: 0.2330, ept: 95.3488
    Epoch [2/50], Val Losses: mse: 0.0984, mae: 0.1199, huber: 0.0258, swd: 0.0687, ept: 95.6901
    Epoch [2/50], Test Losses: mse: 0.0928, mae: 0.1250, huber: 0.0261, swd: 0.0632, ept: 95.7066
      Epoch 2 composite train-obj: 0.053785
            Val objective improved 0.0959 → 0.0258, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0629, mae: 0.1257, huber: 0.0234, swd: 0.0388, ept: 95.8242
    Epoch [3/50], Val Losses: mse: 0.0262, mae: 0.0827, huber: 0.0112, swd: 0.0134, ept: 95.9748
    Epoch [3/50], Test Losses: mse: 0.0276, mae: 0.0879, huber: 0.0120, swd: 0.0143, ept: 95.9766
      Epoch 3 composite train-obj: 0.023380
            Val objective improved 0.0258 → 0.0112, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0271, mae: 0.0924, huber: 0.0122, swd: 0.0162, ept: 95.9633
    Epoch [4/50], Val Losses: mse: 0.0368, mae: 0.1157, huber: 0.0182, swd: 0.0226, ept: 95.9980
    Epoch [4/50], Test Losses: mse: 0.0421, mae: 0.1259, huber: 0.0208, swd: 0.0259, ept: 95.9967
      Epoch 4 composite train-obj: 0.012238
            No improvement (0.0182), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0254, mae: 0.0917, huber: 0.0120, swd: 0.0152, ept: 95.9786
    Epoch [5/50], Val Losses: mse: 0.0134, mae: 0.0701, huber: 0.0065, swd: 0.0079, ept: 95.9986
    Epoch [5/50], Test Losses: mse: 0.0154, mae: 0.0745, huber: 0.0075, swd: 0.0092, ept: 95.9985
      Epoch 5 composite train-obj: 0.012021
            Val objective improved 0.0112 → 0.0065, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0189, mae: 0.0779, huber: 0.0089, swd: 0.0118, ept: 95.9827
    Epoch [6/50], Val Losses: mse: 0.0114, mae: 0.0649, huber: 0.0057, swd: 0.0064, ept: 96.0000
    Epoch [6/50], Test Losses: mse: 0.0125, mae: 0.0703, huber: 0.0063, swd: 0.0071, ept: 96.0000
      Epoch 6 composite train-obj: 0.008906
            Val objective improved 0.0065 → 0.0057, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0180, mae: 0.0733, huber: 0.0085, swd: 0.0113, ept: 95.9863
    Epoch [7/50], Val Losses: mse: 0.0100, mae: 0.0601, huber: 0.0050, swd: 0.0065, ept: 96.0000
    Epoch [7/50], Test Losses: mse: 0.0106, mae: 0.0643, huber: 0.0053, swd: 0.0069, ept: 96.0000
      Epoch 7 composite train-obj: 0.008512
            Val objective improved 0.0057 → 0.0050, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0172, mae: 0.0750, huber: 0.0083, swd: 0.0105, ept: 95.9947
    Epoch [8/50], Val Losses: mse: 0.0213, mae: 0.0999, huber: 0.0107, swd: 0.0131, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0241, mae: 0.1071, huber: 0.0121, swd: 0.0151, ept: 96.0000
      Epoch 8 composite train-obj: 0.008346
            No improvement (0.0107), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.0102, mae: 0.0600, huber: 0.0050, swd: 0.0064, ept: 95.9989
    Epoch [9/50], Val Losses: mse: 0.0102, mae: 0.0572, huber: 0.0051, swd: 0.0066, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0105, mae: 0.0613, huber: 0.0053, swd: 0.0066, ept: 96.0000
      Epoch 9 composite train-obj: 0.005022
            No improvement (0.0051), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.0151, mae: 0.0701, huber: 0.0074, swd: 0.0096, ept: 95.9942
    Epoch [10/50], Val Losses: mse: 0.0087, mae: 0.0516, huber: 0.0043, swd: 0.0058, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0100, mae: 0.0562, huber: 0.0049, swd: 0.0069, ept: 96.0000
      Epoch 10 composite train-obj: 0.007416
            Val objective improved 0.0050 → 0.0043, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0115, mae: 0.0620, huber: 0.0057, swd: 0.0071, ept: 95.9992
    Epoch [11/50], Val Losses: mse: 0.0148, mae: 0.0660, huber: 0.0073, swd: 0.0101, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0136, mae: 0.0687, huber: 0.0067, swd: 0.0088, ept: 96.0000
      Epoch 11 composite train-obj: 0.005674
            No improvement (0.0073), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.0077, mae: 0.0528, huber: 0.0038, swd: 0.0048, ept: 95.9996
    Epoch [12/50], Val Losses: mse: 0.0194, mae: 0.0921, huber: 0.0097, swd: 0.0142, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0221, mae: 0.1006, huber: 0.0110, swd: 0.0164, ept: 96.0000
      Epoch 12 composite train-obj: 0.003837
            No improvement (0.0097), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.0158, mae: 0.0720, huber: 0.0076, swd: 0.0097, ept: 95.9909
    Epoch [13/50], Val Losses: mse: 0.0083, mae: 0.0572, huber: 0.0041, swd: 0.0057, ept: 96.0000
    Epoch [13/50], Test Losses: mse: 0.0094, mae: 0.0615, huber: 0.0047, swd: 0.0064, ept: 96.0000
      Epoch 13 composite train-obj: 0.007573
            Val objective improved 0.0043 → 0.0041, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0098, mae: 0.0592, huber: 0.0049, swd: 0.0061, ept: 96.0000
    Epoch [14/50], Val Losses: mse: 0.0125, mae: 0.0584, huber: 0.0062, swd: 0.0069, ept: 96.0000
    Epoch [14/50], Test Losses: mse: 0.0134, mae: 0.0626, huber: 0.0067, swd: 0.0075, ept: 96.0000
      Epoch 14 composite train-obj: 0.004869
            No improvement (0.0062), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.0099, mae: 0.0562, huber: 0.0049, swd: 0.0061, ept: 95.9982
    Epoch [15/50], Val Losses: mse: 0.0060, mae: 0.0479, huber: 0.0030, swd: 0.0038, ept: 96.0000
    Epoch [15/50], Test Losses: mse: 0.0060, mae: 0.0500, huber: 0.0030, swd: 0.0037, ept: 96.0000
      Epoch 15 composite train-obj: 0.004874
            Val objective improved 0.0041 → 0.0030, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 0.0083, mae: 0.0531, huber: 0.0041, swd: 0.0053, ept: 95.9991
    Epoch [16/50], Val Losses: mse: 0.0093, mae: 0.0554, huber: 0.0047, swd: 0.0073, ept: 96.0000
    Epoch [16/50], Test Losses: mse: 0.0089, mae: 0.0580, huber: 0.0044, swd: 0.0068, ept: 96.0000
      Epoch 16 composite train-obj: 0.004115
            No improvement (0.0047), counter 1/5
    Epoch [17/50], Train Losses: mse: 0.0151, mae: 0.0691, huber: 0.0074, swd: 0.0090, ept: 95.9952
    Epoch [17/50], Val Losses: mse: 0.0039, mae: 0.0391, huber: 0.0020, swd: 0.0025, ept: 96.0000
    Epoch [17/50], Test Losses: mse: 0.0040, mae: 0.0407, huber: 0.0020, swd: 0.0026, ept: 96.0000
      Epoch 17 composite train-obj: 0.007371
            Val objective improved 0.0030 → 0.0020, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 0.0072, mae: 0.0499, huber: 0.0036, swd: 0.0045, ept: 96.0000
    Epoch [18/50], Val Losses: mse: 0.0089, mae: 0.0556, huber: 0.0044, swd: 0.0062, ept: 96.0000
    Epoch [18/50], Test Losses: mse: 0.0097, mae: 0.0603, huber: 0.0048, swd: 0.0066, ept: 96.0000
      Epoch 18 composite train-obj: 0.003563
            No improvement (0.0044), counter 1/5
    Epoch [19/50], Train Losses: mse: 0.0104, mae: 0.0581, huber: 0.0051, swd: 0.0065, ept: 95.9996
    Epoch [19/50], Val Losses: mse: 0.0218, mae: 0.0771, huber: 0.0105, swd: 0.0119, ept: 96.0000
    Epoch [19/50], Test Losses: mse: 0.0248, mae: 0.0837, huber: 0.0120, swd: 0.0139, ept: 96.0000
      Epoch 19 composite train-obj: 0.005133
            No improvement (0.0105), counter 2/5
    Epoch [20/50], Train Losses: mse: 0.0063, mae: 0.0455, huber: 0.0031, swd: 0.0039, ept: 96.0000
    Epoch [20/50], Val Losses: mse: 0.0128, mae: 0.0580, huber: 0.0062, swd: 0.0089, ept: 96.0000
    Epoch [20/50], Test Losses: mse: 0.0130, mae: 0.0613, huber: 0.0063, swd: 0.0087, ept: 96.0000
      Epoch 20 composite train-obj: 0.003115
            No improvement (0.0062), counter 3/5
    Epoch [21/50], Train Losses: mse: 0.0108, mae: 0.0585, huber: 0.0052, swd: 0.0071, ept: 95.9968
    Epoch [21/50], Val Losses: mse: 0.0163, mae: 0.0755, huber: 0.0081, swd: 0.0118, ept: 96.0000
    Epoch [21/50], Test Losses: mse: 0.0194, mae: 0.0823, huber: 0.0096, swd: 0.0144, ept: 96.0000
      Epoch 21 composite train-obj: 0.005242
            No improvement (0.0081), counter 4/5
    Epoch [22/50], Train Losses: mse: 0.0124, mae: 0.0643, huber: 0.0061, swd: 0.0074, ept: 95.9998
    Epoch [22/50], Val Losses: mse: 0.0064, mae: 0.0498, huber: 0.0032, swd: 0.0044, ept: 96.0000
    Epoch [22/50], Test Losses: mse: 0.0065, mae: 0.0522, huber: 0.0032, swd: 0.0043, ept: 96.0000
      Epoch 22 composite train-obj: 0.006128
    Epoch [22/50], Test Losses: mse: 0.0040, mae: 0.0407, huber: 0.0020, swd: 0.0026, ept: 96.0000
    Best round's Test MSE: 0.0040, MAE: 0.0407, SWD: 0.0026
    Best round's Validation MSE: 0.0039, MAE: 0.0391, SWD: 0.0025
    Best round's Test verification MSE : 0.0040, MAE: 0.0407, SWD: 0.0026
    Time taken: 173.27 seconds
    
    ==================================================
    Experiment Summary (ACL_rossler_seq96_pred96_20250511_2122)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0063 ± 0.0017
      mae: 0.0510 ± 0.0086
      huber: 0.0031 ± 0.0008
      swd: 0.0039 ± 0.0010
      ept: 96.0000 ± 0.0000
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0061 ± 0.0016
      mae: 0.0491 ± 0.0086
      huber: 0.0030 ± 0.0008
      swd: 0.0038 ± 0.0009
      ept: 96.0000 ± 0.0000
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 347.43 seconds
    
    Experiment complete: ACL_rossler_seq96_pred96_20250511_2122
    Model: ACL
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

##### do not use ablations: rotations (4,2)


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['rossler']['channels'],# data_mgr.channels,              # ← number of features in your data
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
cfg.x_to_z_delay.enable_magnitudes_scale_shift = [False, True]
cfg.x_to_z_deri.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_x_main.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_y_main.enable_magnitudes_scale_shift = [False, True]
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 2.3266, mae: 0.5618, huber: 0.3521, swd: 1.9839, target_std: 3.9251
    Epoch [1/50], Val Losses: mse: 0.5240, mae: 0.2358, huber: 0.0785, swd: 0.4133, target_std: 3.9370
    Epoch [1/50], Test Losses: mse: 0.4725, mae: 0.2462, huber: 0.0808, swd: 0.3642, target_std: 4.0964
      Epoch 1 composite train-obj: 0.352114
            Val objective improved inf → 0.0785, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.1832, mae: 0.1728, huber: 0.0434, swd: 0.1289, target_std: 3.9253
    Epoch [2/50], Val Losses: mse: 0.0572, mae: 0.1476, huber: 0.0260, swd: 0.0318, target_std: 3.9370
    Epoch [2/50], Test Losses: mse: 0.0675, mae: 0.1611, huber: 0.0307, swd: 0.0391, target_std: 4.0964
      Epoch 2 composite train-obj: 0.043358
            Val objective improved 0.0785 → 0.0260, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0434, mae: 0.1212, huber: 0.0196, swd: 0.0266, target_std: 3.9252
    Epoch [3/50], Val Losses: mse: 0.0532, mae: 0.1327, huber: 0.0251, swd: 0.0346, target_std: 3.9370
    Epoch [3/50], Test Losses: mse: 0.0662, mae: 0.1471, huber: 0.0310, swd: 0.0437, target_std: 4.0964
      Epoch 3 composite train-obj: 0.019579
            Val objective improved 0.0260 → 0.0251, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0312, mae: 0.1033, huber: 0.0146, swd: 0.0199, target_std: 3.9251
    Epoch [4/50], Val Losses: mse: 0.0459, mae: 0.1105, huber: 0.0202, swd: 0.0375, target_std: 3.9370
    Epoch [4/50], Test Losses: mse: 0.0504, mae: 0.1202, huber: 0.0225, swd: 0.0411, target_std: 4.0964
      Epoch 4 composite train-obj: 0.014600
            Val objective improved 0.0251 → 0.0202, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0333, mae: 0.1046, huber: 0.0153, swd: 0.0221, target_std: 3.9252
    Epoch [5/50], Val Losses: mse: 0.0115, mae: 0.0641, huber: 0.0057, swd: 0.0065, target_std: 3.9370
    Epoch [5/50], Test Losses: mse: 0.0133, mae: 0.0681, huber: 0.0065, swd: 0.0078, target_std: 4.0964
      Epoch 5 composite train-obj: 0.015347
            Val objective improved 0.0202 → 0.0057, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0161, mae: 0.0738, huber: 0.0078, swd: 0.0102, target_std: 3.9252
    Epoch [6/50], Val Losses: mse: 0.0180, mae: 0.0762, huber: 0.0088, swd: 0.0112, target_std: 3.9370
    Epoch [6/50], Test Losses: mse: 0.0195, mae: 0.0825, huber: 0.0096, swd: 0.0117, target_std: 4.0964
      Epoch 6 composite train-obj: 0.007760
            No improvement (0.0088), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0190, mae: 0.0803, huber: 0.0091, swd: 0.0123, target_std: 3.9252
    Epoch [7/50], Val Losses: mse: 0.0140, mae: 0.0654, huber: 0.0070, swd: 0.0075, target_std: 3.9370
    Epoch [7/50], Test Losses: mse: 0.0147, mae: 0.0695, huber: 0.0073, swd: 0.0077, target_std: 4.0964
      Epoch 7 composite train-obj: 0.009087
            No improvement (0.0070), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.0203, mae: 0.0814, huber: 0.0096, swd: 0.0137, target_std: 3.9251
    Epoch [8/50], Val Losses: mse: 0.0189, mae: 0.0925, huber: 0.0093, swd: 0.0146, target_std: 3.9370
    Epoch [8/50], Test Losses: mse: 0.0203, mae: 0.0971, huber: 0.0100, swd: 0.0159, target_std: 4.0964
      Epoch 8 composite train-obj: 0.009579
            No improvement (0.0093), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.0151, mae: 0.0737, huber: 0.0074, swd: 0.0103, target_std: 3.9252
    Epoch [9/50], Val Losses: mse: 0.0130, mae: 0.0784, huber: 0.0065, swd: 0.0086, target_std: 3.9370
    Epoch [9/50], Test Losses: mse: 0.0143, mae: 0.0835, huber: 0.0072, swd: 0.0096, target_std: 4.0964
      Epoch 9 composite train-obj: 0.007409
            No improvement (0.0065), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.0168, mae: 0.0759, huber: 0.0082, swd: 0.0112, target_std: 3.9252
    Epoch [10/50], Val Losses: mse: 0.0219, mae: 0.0722, huber: 0.0101, swd: 0.0167, target_std: 3.9370
    Epoch [10/50], Test Losses: mse: 0.0227, mae: 0.0768, huber: 0.0105, swd: 0.0172, target_std: 4.0964
      Epoch 10 composite train-obj: 0.008205
    Epoch [10/50], Test Losses: mse: 0.0133, mae: 0.0681, huber: 0.0065, swd: 0.0078, target_std: 4.0964
    Best round's Test MSE: 0.0133, MAE: 0.0681, SWD: 0.0078
    Best round's Validation MSE: 0.0115, MAE: 0.0641
    Best round's Test verification MSE : 0.0133, MAE: 0.0681, SWD: 0.0078
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.2006, mae: 0.5392, huber: 0.3362, swd: 1.8786, target_std: 3.9251
    Epoch [1/50], Val Losses: mse: 0.5826, mae: 0.2266, huber: 0.0837, swd: 0.5365, target_std: 3.9370
    Epoch [1/50], Test Losses: mse: 0.5263, mae: 0.2389, huber: 0.0867, swd: 0.4734, target_std: 4.0964
      Epoch 1 composite train-obj: 0.336209
            Val objective improved inf → 0.0837, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2120, mae: 0.1702, huber: 0.0452, swd: 0.1740, target_std: 3.9252
    Epoch [2/50], Val Losses: mse: 0.0531, mae: 0.1243, huber: 0.0215, swd: 0.0275, target_std: 3.9370
    Epoch [2/50], Test Losses: mse: 0.0600, mae: 0.1337, huber: 0.0247, swd: 0.0349, target_std: 4.0964
      Epoch 2 composite train-obj: 0.045211
            Val objective improved 0.0837 → 0.0215, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0451, mae: 0.1163, huber: 0.0192, swd: 0.0283, target_std: 3.9252
    Epoch [3/50], Val Losses: mse: 0.0257, mae: 0.0898, huber: 0.0120, swd: 0.0144, target_std: 3.9370
    Epoch [3/50], Test Losses: mse: 0.0275, mae: 0.0967, huber: 0.0129, swd: 0.0156, target_std: 4.0964
      Epoch 3 composite train-obj: 0.019222
            Val objective improved 0.0215 → 0.0120, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0277, mae: 0.0985, huber: 0.0131, swd: 0.0178, target_std: 3.9252
    Epoch [4/50], Val Losses: mse: 0.0186, mae: 0.0768, huber: 0.0093, swd: 0.0124, target_std: 3.9370
    Epoch [4/50], Test Losses: mse: 0.0194, mae: 0.0803, huber: 0.0096, swd: 0.0129, target_std: 4.0964
      Epoch 4 composite train-obj: 0.013097
            Val objective improved 0.0120 → 0.0093, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0219, mae: 0.0857, huber: 0.0105, swd: 0.0146, target_std: 3.9251
    Epoch [5/50], Val Losses: mse: 0.0112, mae: 0.0629, huber: 0.0056, swd: 0.0056, target_std: 3.9370
    Epoch [5/50], Test Losses: mse: 0.0127, mae: 0.0675, huber: 0.0063, swd: 0.0063, target_std: 4.0964
      Epoch 5 composite train-obj: 0.010480
            Val objective improved 0.0093 → 0.0056, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0197, mae: 0.0831, huber: 0.0096, swd: 0.0128, target_std: 3.9252
    Epoch [6/50], Val Losses: mse: 0.0091, mae: 0.0573, huber: 0.0045, swd: 0.0057, target_std: 3.9370
    Epoch [6/50], Test Losses: mse: 0.0095, mae: 0.0602, huber: 0.0047, swd: 0.0059, target_std: 4.0964
      Epoch 6 composite train-obj: 0.009567
            Val objective improved 0.0056 → 0.0045, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0183, mae: 0.0778, huber: 0.0088, swd: 0.0127, target_std: 3.9251
    Epoch [7/50], Val Losses: mse: 0.0143, mae: 0.0584, huber: 0.0068, swd: 0.0103, target_std: 3.9370
    Epoch [7/50], Test Losses: mse: 0.0132, mae: 0.0606, huber: 0.0063, swd: 0.0092, target_std: 4.0964
      Epoch 7 composite train-obj: 0.008792
            No improvement (0.0068), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0181, mae: 0.0782, huber: 0.0087, swd: 0.0123, target_std: 3.9251
    Epoch [8/50], Val Losses: mse: 0.0287, mae: 0.0866, huber: 0.0140, swd: 0.0249, target_std: 3.9370
    Epoch [8/50], Test Losses: mse: 0.0298, mae: 0.0908, huber: 0.0146, swd: 0.0257, target_std: 4.0964
      Epoch 8 composite train-obj: 0.008746
            No improvement (0.0140), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.0168, mae: 0.0736, huber: 0.0082, swd: 0.0118, target_std: 3.9252
    Epoch [9/50], Val Losses: mse: 0.0093, mae: 0.0536, huber: 0.0045, swd: 0.0068, target_std: 3.9370
    Epoch [9/50], Test Losses: mse: 0.0094, mae: 0.0563, huber: 0.0046, swd: 0.0067, target_std: 4.0964
      Epoch 9 composite train-obj: 0.008236
            Val objective improved 0.0045 → 0.0045, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.0181, mae: 0.0740, huber: 0.0085, swd: 0.0129, target_std: 3.9251
    Epoch [10/50], Val Losses: mse: 0.0102, mae: 0.0612, huber: 0.0051, swd: 0.0067, target_std: 3.9370
    Epoch [10/50], Test Losses: mse: 0.0107, mae: 0.0646, huber: 0.0053, swd: 0.0072, target_std: 4.0964
      Epoch 10 composite train-obj: 0.008515
            No improvement (0.0051), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.0183, mae: 0.0765, huber: 0.0087, swd: 0.0128, target_std: 3.9252
    Epoch [11/50], Val Losses: mse: 0.0086, mae: 0.0613, huber: 0.0043, swd: 0.0055, target_std: 3.9370
    Epoch [11/50], Test Losses: mse: 0.0093, mae: 0.0647, huber: 0.0047, swd: 0.0061, target_std: 4.0964
      Epoch 11 composite train-obj: 0.008731
            Val objective improved 0.0045 → 0.0043, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0146, mae: 0.0685, huber: 0.0070, swd: 0.0102, target_std: 3.9252
    Epoch [12/50], Val Losses: mse: 0.0130, mae: 0.0722, huber: 0.0065, swd: 0.0086, target_std: 3.9370
    Epoch [12/50], Test Losses: mse: 0.0142, mae: 0.0772, huber: 0.0071, swd: 0.0094, target_std: 4.0964
      Epoch 12 composite train-obj: 0.007046
            No improvement (0.0065), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0173, mae: 0.0740, huber: 0.0083, swd: 0.0124, target_std: 3.9252
    Epoch [13/50], Val Losses: mse: 0.0179, mae: 0.0719, huber: 0.0087, swd: 0.0114, target_std: 3.9370
    Epoch [13/50], Test Losses: mse: 0.0218, mae: 0.0800, huber: 0.0107, swd: 0.0147, target_std: 4.0964
      Epoch 13 composite train-obj: 0.008341
            No improvement (0.0087), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.0159, mae: 0.0720, huber: 0.0076, swd: 0.0112, target_std: 3.9252
    Epoch [14/50], Val Losses: mse: 0.0255, mae: 0.0799, huber: 0.0119, swd: 0.0168, target_std: 3.9370
    Epoch [14/50], Test Losses: mse: 0.0262, mae: 0.0857, huber: 0.0124, swd: 0.0167, target_std: 4.0964
      Epoch 14 composite train-obj: 0.007647
            No improvement (0.0119), counter 3/5
    Epoch [15/50], Train Losses: mse: 0.0134, mae: 0.0667, huber: 0.0065, swd: 0.0087, target_std: 3.9252
    Epoch [15/50], Val Losses: mse: 0.0140, mae: 0.0601, huber: 0.0066, swd: 0.0121, target_std: 3.9370
    Epoch [15/50], Test Losses: mse: 0.0136, mae: 0.0629, huber: 0.0066, swd: 0.0114, target_std: 4.0964
      Epoch 15 composite train-obj: 0.006534
            No improvement (0.0066), counter 4/5
    Epoch [16/50], Train Losses: mse: 0.0125, mae: 0.0631, huber: 0.0060, swd: 0.0090, target_std: 3.9252
    Epoch [16/50], Val Losses: mse: 0.0124, mae: 0.0709, huber: 0.0062, swd: 0.0091, target_std: 3.9370
    Epoch [16/50], Test Losses: mse: 0.0139, mae: 0.0769, huber: 0.0069, swd: 0.0103, target_std: 4.0964
      Epoch 16 composite train-obj: 0.006017
    Epoch [16/50], Test Losses: mse: 0.0093, mae: 0.0647, huber: 0.0047, swd: 0.0061, target_std: 4.0964
    Best round's Test MSE: 0.0093, MAE: 0.0647, SWD: 0.0061
    Best round's Validation MSE: 0.0086, MAE: 0.0613
    Best round's Test verification MSE : 0.0093, MAE: 0.0647, SWD: 0.0061
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.2387, mae: 0.5293, huber: 0.3305, swd: 1.7657, target_std: 3.9251
    Epoch [1/50], Val Losses: mse: 0.7692, mae: 0.2667, huber: 0.1068, swd: 0.6322, target_std: 3.9370
    Epoch [1/50], Test Losses: mse: 0.6902, mae: 0.2738, huber: 0.1073, swd: 0.5587, target_std: 4.0964
      Epoch 1 composite train-obj: 0.330508
            Val objective improved inf → 0.1068, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3408, mae: 0.1873, huber: 0.0589, swd: 0.2638, target_std: 3.9251
    Epoch [2/50], Val Losses: mse: 0.1291, mae: 0.1335, huber: 0.0321, swd: 0.0922, target_std: 3.9370
    Epoch [2/50], Test Losses: mse: 0.1193, mae: 0.1392, huber: 0.0322, swd: 0.0831, target_std: 4.0964
      Epoch 2 composite train-obj: 0.058929
            Val objective improved 0.1068 → 0.0321, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0659, mae: 0.1239, huber: 0.0234, swd: 0.0425, target_std: 3.9251
    Epoch [3/50], Val Losses: mse: 0.0350, mae: 0.0949, huber: 0.0154, swd: 0.0228, target_std: 3.9370
    Epoch [3/50], Test Losses: mse: 0.0430, mae: 0.1051, huber: 0.0190, swd: 0.0298, target_std: 4.0964
      Epoch 3 composite train-obj: 0.023393
            Val objective improved 0.0321 → 0.0154, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0361, mae: 0.1003, huber: 0.0156, swd: 0.0228, target_std: 3.9251
    Epoch [4/50], Val Losses: mse: 0.0344, mae: 0.0994, huber: 0.0165, swd: 0.0160, target_std: 3.9370
    Epoch [4/50], Test Losses: mse: 0.0394, mae: 0.1080, huber: 0.0189, swd: 0.0186, target_std: 4.0964
      Epoch 4 composite train-obj: 0.015577
            No improvement (0.0165), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0230, mae: 0.0812, huber: 0.0103, swd: 0.0151, target_std: 3.9252
    Epoch [5/50], Val Losses: mse: 0.0245, mae: 0.0828, huber: 0.0114, swd: 0.0157, target_std: 3.9370
    Epoch [5/50], Test Losses: mse: 0.0262, mae: 0.0885, huber: 0.0123, swd: 0.0167, target_std: 4.0964
      Epoch 5 composite train-obj: 0.010305
            Val objective improved 0.0154 → 0.0114, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0239, mae: 0.0850, huber: 0.0110, swd: 0.0154, target_std: 3.9251
    Epoch [6/50], Val Losses: mse: 0.0151, mae: 0.0817, huber: 0.0075, swd: 0.0088, target_std: 3.9370
    Epoch [6/50], Test Losses: mse: 0.0167, mae: 0.0878, huber: 0.0083, swd: 0.0100, target_std: 4.0964
      Epoch 6 composite train-obj: 0.011018
            Val objective improved 0.0114 → 0.0075, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0157, mae: 0.0714, huber: 0.0075, swd: 0.0097, target_std: 3.9252
    Epoch [7/50], Val Losses: mse: 0.0098, mae: 0.0654, huber: 0.0049, swd: 0.0055, target_std: 3.9370
    Epoch [7/50], Test Losses: mse: 0.0110, mae: 0.0692, huber: 0.0055, swd: 0.0064, target_std: 4.0964
      Epoch 7 composite train-obj: 0.007477
            Val objective improved 0.0075 → 0.0049, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0173, mae: 0.0751, huber: 0.0082, swd: 0.0115, target_std: 3.9253
    Epoch [8/50], Val Losses: mse: 0.0072, mae: 0.0533, huber: 0.0036, swd: 0.0050, target_std: 3.9370
    Epoch [8/50], Test Losses: mse: 0.0074, mae: 0.0552, huber: 0.0037, swd: 0.0050, target_std: 4.0964
      Epoch 8 composite train-obj: 0.008245
            Val objective improved 0.0049 → 0.0036, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0178, mae: 0.0770, huber: 0.0086, swd: 0.0114, target_std: 3.9252
    Epoch [9/50], Val Losses: mse: 0.0107, mae: 0.0546, huber: 0.0053, swd: 0.0065, target_std: 3.9370
    Epoch [9/50], Test Losses: mse: 0.0121, mae: 0.0600, huber: 0.0060, swd: 0.0074, target_std: 4.0964
      Epoch 9 composite train-obj: 0.008553
            No improvement (0.0053), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.0130, mae: 0.0658, huber: 0.0063, swd: 0.0084, target_std: 3.9251
    Epoch [10/50], Val Losses: mse: 0.0052, mae: 0.0456, huber: 0.0026, swd: 0.0030, target_std: 3.9370
    Epoch [10/50], Test Losses: mse: 0.0053, mae: 0.0475, huber: 0.0027, swd: 0.0031, target_std: 4.0964
      Epoch 10 composite train-obj: 0.006337
            Val objective improved 0.0036 → 0.0026, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0153, mae: 0.0678, huber: 0.0073, swd: 0.0100, target_std: 3.9252
    Epoch [11/50], Val Losses: mse: 0.0214, mae: 0.0828, huber: 0.0105, swd: 0.0115, target_std: 3.9370
    Epoch [11/50], Test Losses: mse: 0.0217, mae: 0.0855, huber: 0.0107, swd: 0.0117, target_std: 4.0964
      Epoch 11 composite train-obj: 0.007311
            No improvement (0.0105), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.0178, mae: 0.0752, huber: 0.0084, swd: 0.0113, target_std: 3.9252
    Epoch [12/50], Val Losses: mse: 0.0133, mae: 0.0803, huber: 0.0067, swd: 0.0091, target_std: 3.9370
    Epoch [12/50], Test Losses: mse: 0.0148, mae: 0.0856, huber: 0.0074, swd: 0.0103, target_std: 4.0964
      Epoch 12 composite train-obj: 0.008377
            No improvement (0.0067), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.0134, mae: 0.0648, huber: 0.0065, swd: 0.0086, target_std: 3.9252
    Epoch [13/50], Val Losses: mse: 0.0294, mae: 0.0940, huber: 0.0141, swd: 0.0154, target_std: 3.9370
    Epoch [13/50], Test Losses: mse: 0.0332, mae: 0.1039, huber: 0.0160, swd: 0.0175, target_std: 4.0964
      Epoch 13 composite train-obj: 0.006534
            No improvement (0.0141), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.0120, mae: 0.0617, huber: 0.0058, swd: 0.0078, target_std: 3.9251
    Epoch [14/50], Val Losses: mse: 0.0069, mae: 0.0494, huber: 0.0035, swd: 0.0035, target_std: 3.9370
    Epoch [14/50], Test Losses: mse: 0.0072, mae: 0.0524, huber: 0.0036, swd: 0.0037, target_std: 4.0964
      Epoch 14 composite train-obj: 0.005789
            No improvement (0.0035), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.0139, mae: 0.0659, huber: 0.0067, swd: 0.0087, target_std: 3.9253
    Epoch [15/50], Val Losses: mse: 0.0405, mae: 0.0917, huber: 0.0158, swd: 0.0263, target_std: 3.9370
    Epoch [15/50], Test Losses: mse: 0.0416, mae: 0.0966, huber: 0.0168, swd: 0.0270, target_std: 4.0964
      Epoch 15 composite train-obj: 0.006711
    Epoch [15/50], Test Losses: mse: 0.0053, mae: 0.0475, huber: 0.0027, swd: 0.0031, target_std: 4.0964
    Best round's Test MSE: 0.0053, MAE: 0.0475, SWD: 0.0031
    Best round's Validation MSE: 0.0052, MAE: 0.0456
    Best round's Test verification MSE : 0.0053, MAE: 0.0475, SWD: 0.0031
    
    ==================================================
    Experiment Summary (ACL_rossler_seq96_pred96_20250430_0456)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0093 ± 0.0033
      mae: 0.0601 ± 0.0090
      huber: 0.0046 ± 0.0016
      swd: 0.0056 ± 0.0019
      target_std: 4.0964 ± 0.0000
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0084 ± 0.0026
      mae: 0.0570 ± 0.0082
      huber: 0.0042 ± 0.0013
      swd: 0.0050 ± 0.0015
      target_std: 3.9370 ± 0.0000
      count: 40.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_rossler_seq96_pred96_20250430_0456
    Model: ACL
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

##### do not use ablations: Threshold: all 


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['rossler']['channels'],# data_mgr.channels,              # ← number of features in your data
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

cfg.x_to_z_delay.scale_zeroing_threshold = 1e-4
cfg.x_to_z_deri.scale_zeroing_threshold = 1e-4
cfg.z_to_x_main.scale_zeroing_threshold = 1e-4
cfg.z_push_to_z.scale_zeroing_threshold = 1e-4
cfg.z_to_y_main.scale_zeroing_threshold = 1e-4
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 2.3404, mae: 0.5379, huber: 0.3439, swd: 1.9649, ept: 91.2539
    Epoch [1/50], Val Losses: mse: 0.7724, mae: 0.2537, huber: 0.0997, swd: 0.6481, ept: 94.8781
    Epoch [1/50], Test Losses: mse: 0.7001, mae: 0.2644, huber: 0.1020, swd: 0.5737, ept: 94.8042
      Epoch 1 composite train-obj: 0.343894
            Val objective improved inf → 0.0997, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3365, mae: 0.1802, huber: 0.0564, swd: 0.2595, ept: 95.3029
    Epoch [2/50], Val Losses: mse: 0.1233, mae: 0.1580, huber: 0.0364, swd: 0.0767, ept: 95.6699
    Epoch [2/50], Test Losses: mse: 0.1205, mae: 0.1659, huber: 0.0385, swd: 0.0749, ept: 95.7093
      Epoch 2 composite train-obj: 0.056449
            Val objective improved 0.0997 → 0.0364, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0536, mae: 0.1203, huber: 0.0211, swd: 0.0317, ept: 95.7962
    Epoch [3/50], Val Losses: mse: 0.0439, mae: 0.1275, huber: 0.0198, swd: 0.0251, ept: 95.8734
    Epoch [3/50], Test Losses: mse: 0.0474, mae: 0.1341, huber: 0.0216, swd: 0.0279, ept: 95.8808
      Epoch 3 composite train-obj: 0.021148
            Val objective improved 0.0364 → 0.0198, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0234, mae: 0.0870, huber: 0.0110, swd: 0.0144, ept: 95.9722
    Epoch [4/50], Val Losses: mse: 0.0381, mae: 0.1252, huber: 0.0184, swd: 0.0284, ept: 95.9840
    Epoch [4/50], Test Losses: mse: 0.0390, mae: 0.1299, huber: 0.0190, swd: 0.0294, ept: 95.9863
      Epoch 4 composite train-obj: 0.010967
            Val objective improved 0.0198 → 0.0184, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0170, mae: 0.0777, huber: 0.0083, swd: 0.0107, ept: 95.9882
    Epoch [5/50], Val Losses: mse: 0.0076, mae: 0.0558, huber: 0.0038, swd: 0.0049, ept: 96.0000
    Epoch [5/50], Test Losses: mse: 0.0087, mae: 0.0600, huber: 0.0044, swd: 0.0057, ept: 96.0000
      Epoch 5 composite train-obj: 0.008304
            Val objective improved 0.0184 → 0.0038, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0204, mae: 0.0811, huber: 0.0096, swd: 0.0140, ept: 95.9754
    Epoch [6/50], Val Losses: mse: 0.0450, mae: 0.0743, huber: 0.0153, swd: 0.0368, ept: 95.7615
    Epoch [6/50], Test Losses: mse: 0.0441, mae: 0.0803, huber: 0.0163, swd: 0.0355, ept: 95.7865
      Epoch 6 composite train-obj: 0.009581
            No improvement (0.0153), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0172, mae: 0.0774, huber: 0.0084, swd: 0.0119, ept: 95.9939
    Epoch [7/50], Val Losses: mse: 0.0145, mae: 0.0645, huber: 0.0070, swd: 0.0101, ept: 95.9938
    Epoch [7/50], Test Losses: mse: 0.0154, mae: 0.0684, huber: 0.0075, swd: 0.0107, ept: 95.9949
      Epoch 7 composite train-obj: 0.008361
            No improvement (0.0070), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.0085, mae: 0.0567, huber: 0.0042, swd: 0.0057, ept: 96.0000
    Epoch [8/50], Val Losses: mse: 0.0145, mae: 0.0834, huber: 0.0072, swd: 0.0105, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0155, mae: 0.0870, huber: 0.0078, swd: 0.0112, ept: 96.0000
      Epoch 8 composite train-obj: 0.004240
            No improvement (0.0072), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.0111, mae: 0.0629, huber: 0.0055, swd: 0.0074, ept: 95.9973
    Epoch [9/50], Val Losses: mse: 0.0054, mae: 0.0474, huber: 0.0027, swd: 0.0034, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0059, mae: 0.0507, huber: 0.0029, swd: 0.0038, ept: 96.0000
      Epoch 9 composite train-obj: 0.005466
            Val objective improved 0.0038 → 0.0027, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.0131, mae: 0.0672, huber: 0.0064, swd: 0.0089, ept: 95.9968
    Epoch [10/50], Val Losses: mse: 0.0104, mae: 0.0568, huber: 0.0052, swd: 0.0057, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0114, mae: 0.0615, huber: 0.0057, swd: 0.0064, ept: 96.0000
      Epoch 10 composite train-obj: 0.006419
            No improvement (0.0052), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.0111, mae: 0.0609, huber: 0.0054, swd: 0.0075, ept: 95.9985
    Epoch [11/50], Val Losses: mse: 0.0241, mae: 0.0980, huber: 0.0120, swd: 0.0184, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0280, mae: 0.1074, huber: 0.0140, swd: 0.0216, ept: 96.0000
      Epoch 11 composite train-obj: 0.005441
            No improvement (0.0120), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.0091, mae: 0.0561, huber: 0.0045, swd: 0.0060, ept: 95.9997
    Epoch [12/50], Val Losses: mse: 0.0080, mae: 0.0509, huber: 0.0040, swd: 0.0056, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0088, mae: 0.0559, huber: 0.0044, swd: 0.0058, ept: 96.0000
      Epoch 12 composite train-obj: 0.004523
            No improvement (0.0040), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.0078, mae: 0.0530, huber: 0.0039, swd: 0.0054, ept: 96.0000
    Epoch [13/50], Val Losses: mse: 0.0144, mae: 0.0666, huber: 0.0068, swd: 0.0120, ept: 96.0000
    Epoch [13/50], Test Losses: mse: 0.0138, mae: 0.0697, huber: 0.0066, swd: 0.0113, ept: 96.0000
      Epoch 13 composite train-obj: 0.003878
            No improvement (0.0068), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.0165, mae: 0.0728, huber: 0.0080, swd: 0.0113, ept: 95.9924
    Epoch [14/50], Val Losses: mse: 0.0181, mae: 0.0759, huber: 0.0088, swd: 0.0106, ept: 96.0000
    Epoch [14/50], Test Losses: mse: 0.0190, mae: 0.0803, huber: 0.0093, swd: 0.0115, ept: 96.0000
      Epoch 14 composite train-obj: 0.008012
    Epoch [14/50], Test Losses: mse: 0.0059, mae: 0.0507, huber: 0.0030, swd: 0.0038, ept: 96.0000
    Best round's Test MSE: 0.0059, MAE: 0.0507, SWD: 0.0038
    Best round's Validation MSE: 0.0054, MAE: 0.0474, SWD: 0.0034
    Best round's Test verification MSE : 0.0059, MAE: 0.0507, SWD: 0.0038
    Time taken: 117.33 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.2022, mae: 0.5348, huber: 0.3366, swd: 1.7993, ept: 91.4697
    Epoch [1/50], Val Losses: mse: 0.6469, mae: 0.2403, huber: 0.0932, swd: 0.5759, ept: 94.9296
    Epoch [1/50], Test Losses: mse: 0.5895, mae: 0.2520, huber: 0.0969, swd: 0.5066, ept: 94.9512
      Epoch 1 composite train-obj: 0.336595
            Val objective improved inf → 0.0932, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2190, mae: 0.1656, huber: 0.0441, swd: 0.1725, ept: 95.5001
    Epoch [2/50], Val Losses: mse: 0.0551, mae: 0.1301, huber: 0.0215, swd: 0.0293, ept: 95.9033
    Epoch [2/50], Test Losses: mse: 0.0551, mae: 0.1351, huber: 0.0221, swd: 0.0292, ept: 95.9108
      Epoch 2 composite train-obj: 0.044116
            Val objective improved 0.0932 → 0.0215, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0372, mae: 0.1090, huber: 0.0162, swd: 0.0208, ept: 95.9334
    Epoch [3/50], Val Losses: mse: 0.0165, mae: 0.0778, huber: 0.0079, swd: 0.0072, ept: 95.9954
    Epoch [3/50], Test Losses: mse: 0.0176, mae: 0.0824, huber: 0.0085, swd: 0.0081, ept: 95.9957
      Epoch 3 composite train-obj: 0.016154
            Val objective improved 0.0215 → 0.0079, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0236, mae: 0.0918, huber: 0.0112, swd: 0.0145, ept: 95.9800
    Epoch [4/50], Val Losses: mse: 0.0261, mae: 0.0848, huber: 0.0125, swd: 0.0174, ept: 95.9577
    Epoch [4/50], Test Losses: mse: 0.0274, mae: 0.0910, huber: 0.0132, swd: 0.0183, ept: 95.9585
      Epoch 4 composite train-obj: 0.011160
            No improvement (0.0125), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0221, mae: 0.0884, huber: 0.0107, swd: 0.0135, ept: 95.9884
    Epoch [5/50], Val Losses: mse: 0.0354, mae: 0.1246, huber: 0.0174, swd: 0.0261, ept: 95.9990
    Epoch [5/50], Test Losses: mse: 0.0392, mae: 0.1336, huber: 0.0193, swd: 0.0285, ept: 95.9991
      Epoch 5 composite train-obj: 0.010719
            No improvement (0.0174), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.0162, mae: 0.0752, huber: 0.0078, swd: 0.0102, ept: 95.9910
    Epoch [6/50], Val Losses: mse: 0.0101, mae: 0.0645, huber: 0.0050, swd: 0.0061, ept: 96.0000
    Epoch [6/50], Test Losses: mse: 0.0105, mae: 0.0682, huber: 0.0053, swd: 0.0062, ept: 96.0000
      Epoch 6 composite train-obj: 0.007830
            Val objective improved 0.0079 → 0.0050, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0159, mae: 0.0748, huber: 0.0078, swd: 0.0098, ept: 95.9978
    Epoch [7/50], Val Losses: mse: 0.0172, mae: 0.0664, huber: 0.0080, swd: 0.0128, ept: 95.9807
    Epoch [7/50], Test Losses: mse: 0.0190, mae: 0.0723, huber: 0.0089, swd: 0.0138, ept: 95.9812
      Epoch 7 composite train-obj: 0.007790
            No improvement (0.0080), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0134, mae: 0.0688, huber: 0.0065, swd: 0.0086, ept: 95.9971
    Epoch [8/50], Val Losses: mse: 0.0197, mae: 0.0885, huber: 0.0097, swd: 0.0154, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0209, mae: 0.0933, huber: 0.0103, swd: 0.0165, ept: 96.0000
      Epoch 8 composite train-obj: 0.006511
            No improvement (0.0097), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.0143, mae: 0.0676, huber: 0.0068, swd: 0.0097, ept: 95.9902
    Epoch [9/50], Val Losses: mse: 0.0089, mae: 0.0594, huber: 0.0044, swd: 0.0058, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0100, mae: 0.0638, huber: 0.0050, swd: 0.0066, ept: 96.0000
      Epoch 9 composite train-obj: 0.006820
            Val objective improved 0.0050 → 0.0044, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.0101, mae: 0.0613, huber: 0.0050, swd: 0.0067, ept: 95.9995
    Epoch [10/50], Val Losses: mse: 0.0078, mae: 0.0551, huber: 0.0039, swd: 0.0045, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0083, mae: 0.0585, huber: 0.0042, swd: 0.0047, ept: 96.0000
      Epoch 10 composite train-obj: 0.005021
            Val objective improved 0.0044 → 0.0039, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0140, mae: 0.0689, huber: 0.0069, swd: 0.0096, ept: 95.9975
    Epoch [11/50], Val Losses: mse: 0.0142, mae: 0.0726, huber: 0.0070, swd: 0.0084, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0146, mae: 0.0773, huber: 0.0073, swd: 0.0089, ept: 96.0000
      Epoch 11 composite train-obj: 0.006884
            No improvement (0.0070), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.0109, mae: 0.0612, huber: 0.0053, swd: 0.0070, ept: 95.9988
    Epoch [12/50], Val Losses: mse: 0.0298, mae: 0.1090, huber: 0.0149, swd: 0.0242, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0320, mae: 0.1158, huber: 0.0160, swd: 0.0261, ept: 96.0000
      Epoch 12 composite train-obj: 0.005341
            No improvement (0.0149), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.0080, mae: 0.0533, huber: 0.0040, swd: 0.0054, ept: 95.9959
    Epoch [13/50], Val Losses: mse: 0.0043, mae: 0.0407, huber: 0.0021, swd: 0.0029, ept: 96.0000
    Epoch [13/50], Test Losses: mse: 0.0047, mae: 0.0431, huber: 0.0023, swd: 0.0032, ept: 96.0000
      Epoch 13 composite train-obj: 0.003961
            Val objective improved 0.0039 → 0.0021, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0084, mae: 0.0550, huber: 0.0042, swd: 0.0058, ept: 95.9996
    Epoch [14/50], Val Losses: mse: 0.0076, mae: 0.0464, huber: 0.0038, swd: 0.0058, ept: 96.0000
    Epoch [14/50], Test Losses: mse: 0.0075, mae: 0.0494, huber: 0.0037, swd: 0.0057, ept: 96.0000
      Epoch 14 composite train-obj: 0.004193
            No improvement (0.0038), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.0086, mae: 0.0558, huber: 0.0043, swd: 0.0058, ept: 95.9992
    Epoch [15/50], Val Losses: mse: 0.0229, mae: 0.0876, huber: 0.0109, swd: 0.0149, ept: 95.9974
    Epoch [15/50], Test Losses: mse: 0.0227, mae: 0.0909, huber: 0.0109, swd: 0.0149, ept: 95.9971
      Epoch 15 composite train-obj: 0.004266
            No improvement (0.0109), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.0137, mae: 0.0667, huber: 0.0067, swd: 0.0090, ept: 95.9972
    Epoch [16/50], Val Losses: mse: 0.0168, mae: 0.0851, huber: 0.0083, swd: 0.0148, ept: 96.0000
    Epoch [16/50], Test Losses: mse: 0.0182, mae: 0.0913, huber: 0.0090, swd: 0.0159, ept: 96.0000
      Epoch 16 composite train-obj: 0.006730
            No improvement (0.0083), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.0124, mae: 0.0648, huber: 0.0061, swd: 0.0081, ept: 95.9996
    Epoch [17/50], Val Losses: mse: 0.0167, mae: 0.0829, huber: 0.0083, swd: 0.0109, ept: 96.0000
    Epoch [17/50], Test Losses: mse: 0.0179, mae: 0.0883, huber: 0.0089, swd: 0.0117, ept: 96.0000
      Epoch 17 composite train-obj: 0.006142
            No improvement (0.0083), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.0061, mae: 0.0462, huber: 0.0030, swd: 0.0042, ept: 96.0000
    Epoch [18/50], Val Losses: mse: 0.0105, mae: 0.0608, huber: 0.0052, swd: 0.0074, ept: 96.0000
    Epoch [18/50], Test Losses: mse: 0.0114, mae: 0.0651, huber: 0.0057, swd: 0.0081, ept: 96.0000
      Epoch 18 composite train-obj: 0.003050
    Epoch [18/50], Test Losses: mse: 0.0047, mae: 0.0431, huber: 0.0023, swd: 0.0032, ept: 96.0000
    Best round's Test MSE: 0.0047, MAE: 0.0431, SWD: 0.0032
    Best round's Validation MSE: 0.0043, MAE: 0.0407, SWD: 0.0029
    Best round's Test verification MSE : 0.0047, MAE: 0.0431, SWD: 0.0032
    Time taken: 149.49 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.3694, mae: 0.5460, huber: 0.3483, swd: 1.8347, ept: 91.2856
    Epoch [1/50], Val Losses: mse: 0.7729, mae: 0.2162, huber: 0.0939, swd: 0.6434, ept: 94.8268
    Epoch [1/50], Test Losses: mse: 0.7079, mae: 0.2267, huber: 0.0981, swd: 0.5797, ept: 94.6260
      Epoch 1 composite train-obj: 0.348326
            Val objective improved inf → 0.0939, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3113, mae: 0.1814, huber: 0.0554, swd: 0.2343, ept: 95.3632
    Epoch [2/50], Val Losses: mse: 0.0721, mae: 0.1200, huber: 0.0224, swd: 0.0397, ept: 95.8417
    Epoch [2/50], Test Losses: mse: 0.0748, mae: 0.1276, huber: 0.0248, swd: 0.0422, ept: 95.8522
      Epoch 2 composite train-obj: 0.055389
            Val objective improved 0.0939 → 0.0224, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0607, mae: 0.1228, huber: 0.0226, swd: 0.0381, ept: 95.8033
    Epoch [3/50], Val Losses: mse: 0.0297, mae: 0.0931, huber: 0.0128, swd: 0.0168, ept: 95.9395
    Epoch [3/50], Test Losses: mse: 0.0306, mae: 0.0986, huber: 0.0134, swd: 0.0169, ept: 95.9511
      Epoch 3 composite train-obj: 0.022576
            Val objective improved 0.0224 → 0.0128, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0238, mae: 0.0882, huber: 0.0110, swd: 0.0137, ept: 95.9700
    Epoch [4/50], Val Losses: mse: 0.0664, mae: 0.1522, huber: 0.0324, swd: 0.0366, ept: 95.9980
    Epoch [4/50], Test Losses: mse: 0.0737, mae: 0.1630, huber: 0.0361, swd: 0.0403, ept: 95.9922
      Epoch 4 composite train-obj: 0.010971
            No improvement (0.0324), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0244, mae: 0.0920, huber: 0.0117, swd: 0.0151, ept: 95.9857
    Epoch [5/50], Val Losses: mse: 0.0103, mae: 0.0597, huber: 0.0051, swd: 0.0057, ept: 96.0000
    Epoch [5/50], Test Losses: mse: 0.0112, mae: 0.0635, huber: 0.0055, swd: 0.0061, ept: 96.0000
      Epoch 5 composite train-obj: 0.011716
            Val objective improved 0.0128 → 0.0051, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0173, mae: 0.0763, huber: 0.0083, swd: 0.0107, ept: 95.9892
    Epoch [6/50], Val Losses: mse: 0.0359, mae: 0.1022, huber: 0.0173, swd: 0.0264, ept: 95.9948
    Epoch [6/50], Test Losses: mse: 0.0397, mae: 0.1129, huber: 0.0193, swd: 0.0291, ept: 95.9974
      Epoch 6 composite train-obj: 0.008284
            No improvement (0.0173), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0181, mae: 0.0762, huber: 0.0087, swd: 0.0110, ept: 95.9928
    Epoch [7/50], Val Losses: mse: 0.0165, mae: 0.0754, huber: 0.0082, swd: 0.0094, ept: 96.0000
    Epoch [7/50], Test Losses: mse: 0.0176, mae: 0.0810, huber: 0.0088, swd: 0.0097, ept: 96.0000
      Epoch 7 composite train-obj: 0.008744
            No improvement (0.0082), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.0125, mae: 0.0671, huber: 0.0062, swd: 0.0078, ept: 95.9988
    Epoch [8/50], Val Losses: mse: 0.0074, mae: 0.0552, huber: 0.0037, swd: 0.0041, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0076, mae: 0.0572, huber: 0.0038, swd: 0.0041, ept: 96.0000
      Epoch 8 composite train-obj: 0.006186
            Val objective improved 0.0051 → 0.0037, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0103, mae: 0.0606, huber: 0.0051, swd: 0.0065, ept: 95.9994
    Epoch [9/50], Val Losses: mse: 0.0150, mae: 0.0670, huber: 0.0073, swd: 0.0100, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0158, mae: 0.0724, huber: 0.0078, swd: 0.0103, ept: 96.0000
      Epoch 9 composite train-obj: 0.005103
            No improvement (0.0073), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.0131, mae: 0.0679, huber: 0.0065, swd: 0.0081, ept: 95.9971
    Epoch [10/50], Val Losses: mse: 0.0114, mae: 0.0631, huber: 0.0057, swd: 0.0060, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0127, mae: 0.0680, huber: 0.0063, swd: 0.0065, ept: 96.0000
      Epoch 10 composite train-obj: 0.006454
            No improvement (0.0057), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.0113, mae: 0.0636, huber: 0.0056, swd: 0.0068, ept: 95.9976
    Epoch [11/50], Val Losses: mse: 0.0082, mae: 0.0516, huber: 0.0041, swd: 0.0048, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0082, mae: 0.0543, huber: 0.0041, swd: 0.0046, ept: 96.0000
      Epoch 11 composite train-obj: 0.005622
            No improvement (0.0041), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.0092, mae: 0.0575, huber: 0.0046, swd: 0.0055, ept: 95.9997
    Epoch [12/50], Val Losses: mse: 0.0237, mae: 0.1035, huber: 0.0119, swd: 0.0148, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0271, mae: 0.1113, huber: 0.0136, swd: 0.0170, ept: 96.0000
      Epoch 12 composite train-obj: 0.004566
            No improvement (0.0119), counter 4/5
    Epoch [13/50], Train Losses: mse: 0.0162, mae: 0.0727, huber: 0.0078, swd: 0.0098, ept: 95.9908
    Epoch [13/50], Val Losses: mse: 0.0046, mae: 0.0432, huber: 0.0023, swd: 0.0030, ept: 96.0000
    Epoch [13/50], Test Losses: mse: 0.0054, mae: 0.0470, huber: 0.0027, swd: 0.0035, ept: 96.0000
      Epoch 13 composite train-obj: 0.007816
            Val objective improved 0.0037 → 0.0023, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0088, mae: 0.0551, huber: 0.0044, swd: 0.0057, ept: 95.9998
    Epoch [14/50], Val Losses: mse: 0.0364, mae: 0.1121, huber: 0.0171, swd: 0.0224, ept: 95.9923
    Epoch [14/50], Test Losses: mse: 0.0374, mae: 0.1197, huber: 0.0180, swd: 0.0230, ept: 95.9978
      Epoch 14 composite train-obj: 0.004359
            No improvement (0.0171), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.0125, mae: 0.0652, huber: 0.0062, swd: 0.0074, ept: 95.9987
    Epoch [15/50], Val Losses: mse: 0.0064, mae: 0.0492, huber: 0.0032, swd: 0.0039, ept: 96.0000
    Epoch [15/50], Test Losses: mse: 0.0061, mae: 0.0507, huber: 0.0031, swd: 0.0036, ept: 96.0000
      Epoch 15 composite train-obj: 0.006191
            No improvement (0.0032), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.0070, mae: 0.0487, huber: 0.0035, swd: 0.0045, ept: 95.9991
    Epoch [16/50], Val Losses: mse: 0.0095, mae: 0.0497, huber: 0.0048, swd: 0.0041, ept: 96.0000
    Epoch [16/50], Test Losses: mse: 0.0107, mae: 0.0546, huber: 0.0053, swd: 0.0043, ept: 96.0000
      Epoch 16 composite train-obj: 0.003478
            No improvement (0.0048), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.0094, mae: 0.0571, huber: 0.0047, swd: 0.0057, ept: 95.9998
    Epoch [17/50], Val Losses: mse: 0.0064, mae: 0.0534, huber: 0.0032, swd: 0.0046, ept: 96.0000
    Epoch [17/50], Test Losses: mse: 0.0069, mae: 0.0563, huber: 0.0035, swd: 0.0051, ept: 96.0000
      Epoch 17 composite train-obj: 0.004674
            No improvement (0.0032), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.0094, mae: 0.0561, huber: 0.0047, swd: 0.0058, ept: 95.9997
    Epoch [18/50], Val Losses: mse: 0.0078, mae: 0.0560, huber: 0.0039, swd: 0.0046, ept: 96.0000
    Epoch [18/50], Test Losses: mse: 0.0084, mae: 0.0596, huber: 0.0042, swd: 0.0049, ept: 96.0000
      Epoch 18 composite train-obj: 0.004652
    Epoch [18/50], Test Losses: mse: 0.0054, mae: 0.0470, huber: 0.0027, swd: 0.0035, ept: 96.0000
    Best round's Test MSE: 0.0054, MAE: 0.0470, SWD: 0.0035
    Best round's Validation MSE: 0.0046, MAE: 0.0432, SWD: 0.0030
    Best round's Test verification MSE : 0.0054, MAE: 0.0470, SWD: 0.0035
    Time taken: 149.02 seconds
    
    ==================================================
    Experiment Summary (ACL_rossler_seq96_pred96_20250511_2128)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0053 ± 0.0005
      mae: 0.0469 ± 0.0031
      huber: 0.0027 ± 0.0002
      swd: 0.0035 ± 0.0002
      ept: 96.0000 ± 0.0000
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0047 ± 0.0005
      mae: 0.0438 ± 0.0028
      huber: 0.0024 ± 0.0002
      swd: 0.0031 ± 0.0002
      ept: 96.0000 ± 0.0000
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 415.90 seconds
    
    Experiment complete: ACL_rossler_seq96_pred96_20250511_2128
    Model: ACL
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

##### do not use ablations: only threshold z_to_z and z_to_y


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['rossler']['channels'],# data_mgr.channels,              # ← number of features in your data
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

cfg.x_to_z_delay.scale_zeroing_threshold = 0e-4
cfg.x_to_z_deri.scale_zeroing_threshold = 0e-4
cfg.z_to_x_main.scale_zeroing_threshold = 0e-4
cfg.z_push_to_z.scale_zeroing_threshold = 1e-4
cfg.z_to_y_main.scale_zeroing_threshold = 1e-4
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 2.3531, mae: 0.5433, huber: 0.3464, swd: 1.9766, ept: 91.2247
    Epoch [1/50], Val Losses: mse: 0.9892, mae: 0.4044, huber: 0.1871, swd: 0.8377, ept: 94.3325
    Epoch [1/50], Test Losses: mse: 0.9384, mae: 0.4251, huber: 0.2001, swd: 0.7859, ept: 94.0398
      Epoch 1 composite train-obj: 0.346446
            Val objective improved inf → 0.1871, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3651, mae: 0.1898, huber: 0.0619, swd: 0.2834, ept: 95.2600
    Epoch [2/50], Val Losses: mse: 0.1557, mae: 0.1787, huber: 0.0441, swd: 0.0944, ept: 95.3811
    Epoch [2/50], Test Losses: mse: 0.1491, mae: 0.1896, huber: 0.0460, swd: 0.0869, ept: 95.4934
      Epoch 2 composite train-obj: 0.061867
            Val objective improved 0.1871 → 0.0441, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0517, mae: 0.1103, huber: 0.0186, swd: 0.0301, ept: 95.8268
    Epoch [3/50], Val Losses: mse: 0.0351, mae: 0.0961, huber: 0.0153, swd: 0.0217, ept: 95.8753
    Epoch [3/50], Test Losses: mse: 0.0355, mae: 0.1020, huber: 0.0159, swd: 0.0221, ept: 95.9005
      Epoch 3 composite train-obj: 0.018586
            Val objective improved 0.0441 → 0.0153, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0231, mae: 0.0870, huber: 0.0109, swd: 0.0140, ept: 95.9732
    Epoch [4/50], Val Losses: mse: 0.0179, mae: 0.0770, huber: 0.0087, swd: 0.0127, ept: 95.9991
    Epoch [4/50], Test Losses: mse: 0.0179, mae: 0.0810, huber: 0.0087, swd: 0.0125, ept: 95.9992
      Epoch 4 composite train-obj: 0.010869
            Val objective improved 0.0153 → 0.0087, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0170, mae: 0.0762, huber: 0.0082, swd: 0.0110, ept: 95.9893
    Epoch [5/50], Val Losses: mse: 0.0096, mae: 0.0587, huber: 0.0048, swd: 0.0061, ept: 96.0000
    Epoch [5/50], Test Losses: mse: 0.0112, mae: 0.0632, huber: 0.0056, swd: 0.0074, ept: 96.0000
      Epoch 5 composite train-obj: 0.008214
            Val objective improved 0.0087 → 0.0048, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0192, mae: 0.0759, huber: 0.0090, swd: 0.0131, ept: 95.9720
    Epoch [6/50], Val Losses: mse: 0.0950, mae: 0.1045, huber: 0.0261, swd: 0.0772, ept: 95.5781
    Epoch [6/50], Test Losses: mse: 0.0860, mae: 0.1104, huber: 0.0267, swd: 0.0681, ept: 95.6431
      Epoch 6 composite train-obj: 0.009003
            No improvement (0.0261), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0135, mae: 0.0676, huber: 0.0065, swd: 0.0089, ept: 95.9957
    Epoch [7/50], Val Losses: mse: 0.0075, mae: 0.0534, huber: 0.0038, swd: 0.0045, ept: 96.0000
    Epoch [7/50], Test Losses: mse: 0.0080, mae: 0.0564, huber: 0.0040, swd: 0.0048, ept: 96.0000
      Epoch 7 composite train-obj: 0.006548
            Val objective improved 0.0048 → 0.0038, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0093, mae: 0.0573, huber: 0.0046, swd: 0.0060, ept: 95.9991
    Epoch [8/50], Val Losses: mse: 0.0083, mae: 0.0575, huber: 0.0042, swd: 0.0059, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0087, mae: 0.0607, huber: 0.0044, swd: 0.0061, ept: 96.0000
      Epoch 8 composite train-obj: 0.004613
            No improvement (0.0042), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.0099, mae: 0.0606, huber: 0.0049, swd: 0.0067, ept: 96.0000
    Epoch [9/50], Val Losses: mse: 0.0256, mae: 0.1013, huber: 0.0127, swd: 0.0178, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0291, mae: 0.1111, huber: 0.0145, swd: 0.0201, ept: 96.0000
      Epoch 9 composite train-obj: 0.004946
            No improvement (0.0127), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.0102, mae: 0.0582, huber: 0.0051, swd: 0.0069, ept: 95.9994
    Epoch [10/50], Val Losses: mse: 0.0137, mae: 0.0780, huber: 0.0069, swd: 0.0095, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0154, mae: 0.0841, huber: 0.0077, swd: 0.0107, ept: 96.0000
      Epoch 10 composite train-obj: 0.005052
            No improvement (0.0069), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.0151, mae: 0.0719, huber: 0.0074, swd: 0.0102, ept: 95.9970
    Epoch [11/50], Val Losses: mse: 0.0164, mae: 0.0776, huber: 0.0082, swd: 0.0127, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0174, mae: 0.0829, huber: 0.0087, swd: 0.0134, ept: 96.0000
      Epoch 11 composite train-obj: 0.007395
            No improvement (0.0082), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.0054, mae: 0.0444, huber: 0.0027, swd: 0.0036, ept: 96.0000
    Epoch [12/50], Val Losses: mse: 0.0023, mae: 0.0299, huber: 0.0011, swd: 0.0015, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0024, mae: 0.0311, huber: 0.0012, swd: 0.0016, ept: 96.0000
      Epoch 12 composite train-obj: 0.002698
            Val objective improved 0.0038 → 0.0011, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.0107, mae: 0.0586, huber: 0.0053, swd: 0.0074, ept: 95.9982
    Epoch [13/50], Val Losses: mse: 0.0100, mae: 0.0561, huber: 0.0049, swd: 0.0066, ept: 96.0000
    Epoch [13/50], Test Losses: mse: 0.0104, mae: 0.0608, huber: 0.0051, swd: 0.0069, ept: 96.0000
      Epoch 13 composite train-obj: 0.005280
            No improvement (0.0049), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.0132, mae: 0.0626, huber: 0.0064, swd: 0.0083, ept: 95.9982
    Epoch [14/50], Val Losses: mse: 0.0260, mae: 0.0919, huber: 0.0129, swd: 0.0136, ept: 96.0000
    Epoch [14/50], Test Losses: mse: 0.0277, mae: 0.0974, huber: 0.0137, swd: 0.0145, ept: 96.0000
      Epoch 14 composite train-obj: 0.006414
            No improvement (0.0129), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.0132, mae: 0.0681, huber: 0.0065, swd: 0.0092, ept: 95.9977
    Epoch [15/50], Val Losses: mse: 0.0141, mae: 0.0720, huber: 0.0070, swd: 0.0091, ept: 96.0000
    Epoch [15/50], Test Losses: mse: 0.0143, mae: 0.0757, huber: 0.0071, swd: 0.0090, ept: 96.0000
      Epoch 15 composite train-obj: 0.006482
            No improvement (0.0070), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.0067, mae: 0.0465, huber: 0.0033, swd: 0.0048, ept: 95.9994
    Epoch [16/50], Val Losses: mse: 0.0073, mae: 0.0462, huber: 0.0036, swd: 0.0059, ept: 96.0000
    Epoch [16/50], Test Losses: mse: 0.0074, mae: 0.0493, huber: 0.0037, swd: 0.0058, ept: 96.0000
      Epoch 16 composite train-obj: 0.003324
            No improvement (0.0036), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.0081, mae: 0.0519, huber: 0.0040, swd: 0.0059, ept: 95.9998
    Epoch [17/50], Val Losses: mse: 0.0053, mae: 0.0440, huber: 0.0026, swd: 0.0035, ept: 96.0000
    Epoch [17/50], Test Losses: mse: 0.0060, mae: 0.0481, huber: 0.0030, swd: 0.0038, ept: 96.0000
      Epoch 17 composite train-obj: 0.004027
    Epoch [17/50], Test Losses: mse: 0.0024, mae: 0.0311, huber: 0.0012, swd: 0.0016, ept: 96.0000
    Best round's Test MSE: 0.0024, MAE: 0.0311, SWD: 0.0016
    Best round's Validation MSE: 0.0023, MAE: 0.0299, SWD: 0.0015
    Best round's Test verification MSE : 0.0024, MAE: 0.0311, SWD: 0.0016
    Time taken: 140.86 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.1900, mae: 0.5306, huber: 0.3340, swd: 1.7912, ept: 91.4486
    Epoch [1/50], Val Losses: mse: 0.5668, mae: 0.2309, huber: 0.0833, swd: 0.4966, ept: 94.9838
    Epoch [1/50], Test Losses: mse: 0.5161, mae: 0.2398, huber: 0.0858, swd: 0.4366, ept: 94.8577
      Epoch 1 composite train-obj: 0.333985
            Val objective improved inf → 0.0833, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2068, mae: 0.1655, huber: 0.0438, swd: 0.1593, ept: 95.5293
    Epoch [2/50], Val Losses: mse: 0.0677, mae: 0.1324, huber: 0.0255, swd: 0.0399, ept: 95.7957
    Epoch [2/50], Test Losses: mse: 0.0680, mae: 0.1370, huber: 0.0266, swd: 0.0402, ept: 95.7988
      Epoch 2 composite train-obj: 0.043830
            Val objective improved 0.0833 → 0.0255, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0447, mae: 0.1212, huber: 0.0198, swd: 0.0261, ept: 95.8968
    Epoch [3/50], Val Losses: mse: 0.0429, mae: 0.1454, huber: 0.0209, swd: 0.0268, ept: 95.9836
    Epoch [3/50], Test Losses: mse: 0.0467, mae: 0.1528, huber: 0.0228, swd: 0.0301, ept: 95.9837
      Epoch 3 composite train-obj: 0.019765
            Val objective improved 0.0255 → 0.0209, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0222, mae: 0.0895, huber: 0.0107, swd: 0.0129, ept: 95.9857
    Epoch [4/50], Val Losses: mse: 0.0253, mae: 0.0923, huber: 0.0126, swd: 0.0124, ept: 96.0000
    Epoch [4/50], Test Losses: mse: 0.0271, mae: 0.0978, huber: 0.0135, swd: 0.0131, ept: 96.0000
      Epoch 4 composite train-obj: 0.010657
            Val objective improved 0.0209 → 0.0126, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0165, mae: 0.0787, huber: 0.0081, swd: 0.0098, ept: 95.9972
    Epoch [5/50], Val Losses: mse: 0.0235, mae: 0.0660, huber: 0.0105, swd: 0.0184, ept: 95.9593
    Epoch [5/50], Test Losses: mse: 0.0223, mae: 0.0697, huber: 0.0102, swd: 0.0170, ept: 95.9632
      Epoch 5 composite train-obj: 0.008088
            Val objective improved 0.0126 → 0.0105, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0169, mae: 0.0763, huber: 0.0080, swd: 0.0109, ept: 95.9890
    Epoch [6/50], Val Losses: mse: 0.0121, mae: 0.0678, huber: 0.0061, swd: 0.0069, ept: 96.0000
    Epoch [6/50], Test Losses: mse: 0.0128, mae: 0.0717, huber: 0.0064, swd: 0.0073, ept: 96.0000
      Epoch 6 composite train-obj: 0.008017
            Val objective improved 0.0105 → 0.0061, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0155, mae: 0.0728, huber: 0.0075, swd: 0.0099, ept: 95.9945
    Epoch [7/50], Val Losses: mse: 0.0113, mae: 0.0625, huber: 0.0056, swd: 0.0068, ept: 96.0000
    Epoch [7/50], Test Losses: mse: 0.0123, mae: 0.0667, huber: 0.0061, swd: 0.0076, ept: 96.0000
      Epoch 7 composite train-obj: 0.007490
            Val objective improved 0.0061 → 0.0056, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0142, mae: 0.0726, huber: 0.0070, swd: 0.0093, ept: 95.9989
    Epoch [8/50], Val Losses: mse: 0.0140, mae: 0.0722, huber: 0.0069, swd: 0.0085, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0151, mae: 0.0764, huber: 0.0075, swd: 0.0094, ept: 96.0000
      Epoch 8 composite train-obj: 0.007006
            No improvement (0.0069), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.0174, mae: 0.0754, huber: 0.0084, swd: 0.0114, ept: 95.9929
    Epoch [9/50], Val Losses: mse: 0.0116, mae: 0.0618, huber: 0.0058, swd: 0.0080, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0130, mae: 0.0673, huber: 0.0065, swd: 0.0093, ept: 96.0000
      Epoch 9 composite train-obj: 0.008381
            No improvement (0.0058), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.0090, mae: 0.0563, huber: 0.0044, swd: 0.0061, ept: 95.9978
    Epoch [10/50], Val Losses: mse: 0.0208, mae: 0.0828, huber: 0.0102, swd: 0.0149, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0223, mae: 0.0892, huber: 0.0110, swd: 0.0161, ept: 96.0000
      Epoch 10 composite train-obj: 0.004435
            No improvement (0.0102), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.0116, mae: 0.0637, huber: 0.0057, swd: 0.0079, ept: 95.9976
    Epoch [11/50], Val Losses: mse: 0.0191, mae: 0.0778, huber: 0.0091, swd: 0.0139, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0188, mae: 0.0826, huber: 0.0091, swd: 0.0135, ept: 96.0000
      Epoch 11 composite train-obj: 0.005676
            No improvement (0.0091), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.0122, mae: 0.0657, huber: 0.0060, swd: 0.0083, ept: 95.9994
    Epoch [12/50], Val Losses: mse: 0.0287, mae: 0.0722, huber: 0.0116, swd: 0.0253, ept: 95.9253
    Epoch [12/50], Test Losses: mse: 0.0278, mae: 0.0744, huber: 0.0115, swd: 0.0243, ept: 95.9285
      Epoch 12 composite train-obj: 0.005999
    Epoch [12/50], Test Losses: mse: 0.0123, mae: 0.0667, huber: 0.0061, swd: 0.0076, ept: 96.0000
    Best round's Test MSE: 0.0123, MAE: 0.0667, SWD: 0.0076
    Best round's Validation MSE: 0.0113, MAE: 0.0625, SWD: 0.0068
    Best round's Test verification MSE : 0.0123, MAE: 0.0667, SWD: 0.0076
    Time taken: 101.98 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.3705, mae: 0.5488, huber: 0.3499, swd: 1.8361, ept: 91.2403
    Epoch [1/50], Val Losses: mse: 0.6930, mae: 0.2005, huber: 0.0807, swd: 0.5737, ept: 94.9880
    Epoch [1/50], Test Losses: mse: 0.6268, mae: 0.2077, huber: 0.0827, swd: 0.5089, ept: 94.8549
      Epoch 1 composite train-obj: 0.349899
            Val objective improved inf → 0.0807, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2986, mae: 0.1791, huber: 0.0545, swd: 0.2256, ept: 95.3188
    Epoch [2/50], Val Losses: mse: 0.0938, mae: 0.1366, huber: 0.0277, swd: 0.0576, ept: 95.7429
    Epoch [2/50], Test Losses: mse: 0.1015, mae: 0.1479, huber: 0.0324, swd: 0.0642, ept: 95.6852
      Epoch 2 composite train-obj: 0.054477
            Val objective improved 0.0807 → 0.0277, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0542, mae: 0.1158, huber: 0.0201, swd: 0.0331, ept: 95.8418
    Epoch [3/50], Val Losses: mse: 0.0193, mae: 0.0803, huber: 0.0087, swd: 0.0093, ept: 95.9870
    Epoch [3/50], Test Losses: mse: 0.0217, mae: 0.0865, huber: 0.0099, swd: 0.0111, ept: 95.9850
      Epoch 3 composite train-obj: 0.020091
            Val objective improved 0.0277 → 0.0087, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0252, mae: 0.0907, huber: 0.0116, swd: 0.0149, ept: 95.9679
    Epoch [4/50], Val Losses: mse: 0.0149, mae: 0.0760, huber: 0.0073, swd: 0.0067, ept: 96.0000
    Epoch [4/50], Test Losses: mse: 0.0165, mae: 0.0817, huber: 0.0081, swd: 0.0075, ept: 95.9995
      Epoch 4 composite train-obj: 0.011610
            Val objective improved 0.0087 → 0.0073, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0239, mae: 0.0892, huber: 0.0113, swd: 0.0152, ept: 95.9804
    Epoch [5/50], Val Losses: mse: 0.0125, mae: 0.0744, huber: 0.0062, swd: 0.0079, ept: 96.0000
    Epoch [5/50], Test Losses: mse: 0.0138, mae: 0.0788, huber: 0.0068, swd: 0.0087, ept: 96.0000
      Epoch 5 composite train-obj: 0.011255
            Val objective improved 0.0073 → 0.0062, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0189, mae: 0.0776, huber: 0.0090, swd: 0.0120, ept: 95.9831
    Epoch [6/50], Val Losses: mse: 0.0186, mae: 0.0825, huber: 0.0093, swd: 0.0114, ept: 96.0000
    Epoch [6/50], Test Losses: mse: 0.0205, mae: 0.0896, huber: 0.0102, swd: 0.0123, ept: 96.0000
      Epoch 6 composite train-obj: 0.008961
            No improvement (0.0093), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0162, mae: 0.0739, huber: 0.0079, swd: 0.0096, ept: 95.9946
    Epoch [7/50], Val Losses: mse: 0.0061, mae: 0.0487, huber: 0.0031, swd: 0.0035, ept: 96.0000
    Epoch [7/50], Test Losses: mse: 0.0063, mae: 0.0506, huber: 0.0032, swd: 0.0036, ept: 96.0000
      Epoch 7 composite train-obj: 0.007898
            Val objective improved 0.0062 → 0.0031, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0127, mae: 0.0657, huber: 0.0062, swd: 0.0078, ept: 95.9971
    Epoch [8/50], Val Losses: mse: 0.0109, mae: 0.0652, huber: 0.0054, swd: 0.0058, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0126, mae: 0.0704, huber: 0.0063, swd: 0.0067, ept: 96.0000
      Epoch 8 composite train-obj: 0.006210
            No improvement (0.0054), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.0131, mae: 0.0666, huber: 0.0065, swd: 0.0084, ept: 95.9988
    Epoch [9/50], Val Losses: mse: 0.0233, mae: 0.0906, huber: 0.0116, swd: 0.0143, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0264, mae: 0.0989, huber: 0.0132, swd: 0.0158, ept: 96.0000
      Epoch 9 composite train-obj: 0.006461
            No improvement (0.0116), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.0145, mae: 0.0687, huber: 0.0071, swd: 0.0087, ept: 95.9987
    Epoch [10/50], Val Losses: mse: 0.0151, mae: 0.0716, huber: 0.0075, swd: 0.0087, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0159, mae: 0.0769, huber: 0.0079, swd: 0.0094, ept: 96.0000
      Epoch 10 composite train-obj: 0.007141
            No improvement (0.0075), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.0133, mae: 0.0657, huber: 0.0065, swd: 0.0084, ept: 95.9985
    Epoch [11/50], Val Losses: mse: 0.0146, mae: 0.0680, huber: 0.0071, swd: 0.0102, ept: 95.9984
    Epoch [11/50], Test Losses: mse: 0.0134, mae: 0.0695, huber: 0.0066, swd: 0.0090, ept: 96.0000
      Epoch 11 composite train-obj: 0.006517
            No improvement (0.0071), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.0103, mae: 0.0597, huber: 0.0051, swd: 0.0060, ept: 96.0000
    Epoch [12/50], Val Losses: mse: 0.0102, mae: 0.0663, huber: 0.0051, swd: 0.0064, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0107, mae: 0.0692, huber: 0.0053, swd: 0.0066, ept: 96.0000
      Epoch 12 composite train-obj: 0.005113
    Epoch [12/50], Test Losses: mse: 0.0063, mae: 0.0506, huber: 0.0032, swd: 0.0036, ept: 96.0000
    Best round's Test MSE: 0.0063, MAE: 0.0506, SWD: 0.0036
    Best round's Validation MSE: 0.0061, MAE: 0.0487, SWD: 0.0035
    Best round's Test verification MSE : 0.0063, MAE: 0.0506, SWD: 0.0036
    Time taken: 96.51 seconds
    
    ==================================================
    Experiment Summary (ACL_rossler_seq96_pred96_20250511_2157)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0070 ± 0.0041
      mae: 0.0495 ± 0.0146
      huber: 0.0035 ± 0.0020
      swd: 0.0042 ± 0.0025
      ept: 96.0000 ± 0.0000
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0066 ± 0.0037
      mae: 0.0470 ± 0.0134
      huber: 0.0033 ± 0.0018
      swd: 0.0039 ± 0.0022
      ept: 96.0000 ± 0.0000
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 339.44 seconds
    
    Experiment complete: ACL_rossler_seq96_pred96_20250511_2157
    Model: ACL
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

##### huber +0.1SWD


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['rossler']['channels'],# data_mgr.channels,              # ← number of features in your data
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
    loss_backward_weights = [0.0, 0.0, 1.0, 0.1, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.1, 0.0],
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
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 2.3605, mae: 0.5891, huber: 0.3722, swd: 1.5033, ept: 87.3599
    Epoch [1/50], Val Losses: mse: 0.2558, mae: 0.2056, huber: 0.0628, swd: 0.1375, ept: 95.0676
    Epoch [1/50], Test Losses: mse: 0.2460, mae: 0.2188, huber: 0.0667, swd: 0.1309, ept: 95.0725
      Epoch 1 composite train-obj: 0.522535
            Val objective improved inf → 0.0765, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.1073, mae: 0.1551, huber: 0.0355, swd: 0.0561, ept: 95.4705
    Epoch [2/50], Val Losses: mse: 0.0587, mae: 0.1355, huber: 0.0253, swd: 0.0313, ept: 95.7279
    Epoch [2/50], Test Losses: mse: 0.0624, mae: 0.1453, huber: 0.0276, swd: 0.0345, ept: 95.7933
      Epoch 2 composite train-obj: 0.041159
            Val objective improved 0.0765 → 0.0285, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0384, mae: 0.1077, huber: 0.0168, swd: 0.0209, ept: 95.8776
    Epoch [3/50], Val Losses: mse: 0.0191, mae: 0.0756, huber: 0.0089, swd: 0.0098, ept: 95.9767
    Epoch [3/50], Test Losses: mse: 0.0208, mae: 0.0810, huber: 0.0097, swd: 0.0104, ept: 95.9734
      Epoch 3 composite train-obj: 0.018933
            Val objective improved 0.0285 → 0.0099, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0241, mae: 0.0875, huber: 0.0112, swd: 0.0141, ept: 95.9588
    Epoch [4/50], Val Losses: mse: 0.0139, mae: 0.0753, huber: 0.0069, swd: 0.0074, ept: 96.0000
    Epoch [4/50], Test Losses: mse: 0.0145, mae: 0.0793, huber: 0.0072, swd: 0.0073, ept: 95.9977
      Epoch 4 composite train-obj: 0.012641
            Val objective improved 0.0099 → 0.0076, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0178, mae: 0.0761, huber: 0.0084, swd: 0.0112, ept: 95.9778
    Epoch [5/50], Val Losses: mse: 0.0228, mae: 0.0833, huber: 0.0109, swd: 0.0111, ept: 95.9941
    Epoch [5/50], Test Losses: mse: 0.0239, mae: 0.0891, huber: 0.0115, swd: 0.0118, ept: 95.9914
      Epoch 5 composite train-obj: 0.009528
            No improvement (0.0120), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.0233, mae: 0.0831, huber: 0.0107, swd: 0.0153, ept: 95.9652
    Epoch [6/50], Val Losses: mse: 0.0829, mae: 0.1441, huber: 0.0335, swd: 0.0643, ept: 95.5483
    Epoch [6/50], Test Losses: mse: 0.0922, mae: 0.1577, huber: 0.0389, swd: 0.0690, ept: 95.6486
      Epoch 6 composite train-obj: 0.012205
            No improvement (0.0399), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.0181, mae: 0.0764, huber: 0.0086, swd: 0.0116, ept: 95.9849
    Epoch [7/50], Val Losses: mse: 0.0174, mae: 0.0769, huber: 0.0083, swd: 0.0108, ept: 95.9937
    Epoch [7/50], Test Losses: mse: 0.0182, mae: 0.0820, huber: 0.0087, swd: 0.0110, ept: 95.9941
      Epoch 7 composite train-obj: 0.009804
            No improvement (0.0093), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.0157, mae: 0.0714, huber: 0.0075, swd: 0.0103, ept: 95.9896
    Epoch [8/50], Val Losses: mse: 0.0252, mae: 0.0769, huber: 0.0112, swd: 0.0205, ept: 95.9768
    Epoch [8/50], Test Losses: mse: 0.0248, mae: 0.0807, huber: 0.0115, swd: 0.0197, ept: 95.9877
      Epoch 8 composite train-obj: 0.008543
            No improvement (0.0132), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.0139, mae: 0.0663, huber: 0.0066, swd: 0.0093, ept: 95.9901
    Epoch [9/50], Val Losses: mse: 0.0071, mae: 0.0547, huber: 0.0036, swd: 0.0041, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0082, mae: 0.0588, huber: 0.0041, swd: 0.0044, ept: 96.0000
      Epoch 9 composite train-obj: 0.007552
            Val objective improved 0.0076 → 0.0040, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.0145, mae: 0.0700, huber: 0.0071, swd: 0.0099, ept: 95.9947
    Epoch [10/50], Val Losses: mse: 0.0104, mae: 0.0549, huber: 0.0051, swd: 0.0082, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0110, mae: 0.0586, huber: 0.0054, swd: 0.0086, ept: 95.9984
      Epoch 10 composite train-obj: 0.008059
            No improvement (0.0059), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.0095, mae: 0.0559, huber: 0.0047, swd: 0.0063, ept: 95.9978
    Epoch [11/50], Val Losses: mse: 0.0070, mae: 0.0467, huber: 0.0035, swd: 0.0041, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0070, mae: 0.0482, huber: 0.0035, swd: 0.0043, ept: 96.0000
      Epoch 11 composite train-obj: 0.005297
            Val objective improved 0.0040 → 0.0039, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0114, mae: 0.0607, huber: 0.0056, swd: 0.0075, ept: 95.9980
    Epoch [12/50], Val Losses: mse: 0.0125, mae: 0.0688, huber: 0.0063, swd: 0.0079, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0140, mae: 0.0746, huber: 0.0070, swd: 0.0085, ept: 96.0000
      Epoch 12 composite train-obj: 0.006385
            No improvement (0.0071), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0113, mae: 0.0616, huber: 0.0056, swd: 0.0072, ept: 95.9994
    Epoch [13/50], Val Losses: mse: 0.0052, mae: 0.0456, huber: 0.0026, swd: 0.0032, ept: 96.0000
    Epoch [13/50], Test Losses: mse: 0.0052, mae: 0.0467, huber: 0.0026, swd: 0.0033, ept: 96.0000
      Epoch 13 composite train-obj: 0.006299
            Val objective improved 0.0039 → 0.0029, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0078, mae: 0.0491, huber: 0.0038, swd: 0.0051, ept: 95.9985
    Epoch [14/50], Val Losses: mse: 0.0241, mae: 0.0984, huber: 0.0119, swd: 0.0166, ept: 96.0000
    Epoch [14/50], Test Losses: mse: 0.0243, mae: 0.1021, huber: 0.0121, swd: 0.0165, ept: 96.0000
      Epoch 14 composite train-obj: 0.004329
            No improvement (0.0136), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.0161, mae: 0.0723, huber: 0.0078, swd: 0.0109, ept: 95.9932
    Epoch [15/50], Val Losses: mse: 0.0212, mae: 0.0922, huber: 0.0105, swd: 0.0143, ept: 96.0000
    Epoch [15/50], Test Losses: mse: 0.0217, mae: 0.0980, huber: 0.0108, swd: 0.0147, ept: 96.0000
      Epoch 15 composite train-obj: 0.008896
            No improvement (0.0119), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.0091, mae: 0.0530, huber: 0.0044, swd: 0.0061, ept: 95.9968
    Epoch [16/50], Val Losses: mse: 0.0060, mae: 0.0376, huber: 0.0029, swd: 0.0046, ept: 96.0000
    Epoch [16/50], Test Losses: mse: 0.0061, mae: 0.0394, huber: 0.0030, swd: 0.0046, ept: 96.0000
      Epoch 16 composite train-obj: 0.005026
            No improvement (0.0034), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.0076, mae: 0.0514, huber: 0.0038, swd: 0.0052, ept: 95.9995
    Epoch [17/50], Val Losses: mse: 0.0043, mae: 0.0408, huber: 0.0022, swd: 0.0028, ept: 96.0000
    Epoch [17/50], Test Losses: mse: 0.0043, mae: 0.0429, huber: 0.0021, swd: 0.0029, ept: 96.0000
      Epoch 17 composite train-obj: 0.004302
            Val objective improved 0.0029 → 0.0024, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 0.0104, mae: 0.0539, huber: 0.0049, swd: 0.0073, ept: 95.9885
    Epoch [18/50], Val Losses: mse: 0.0045, mae: 0.0393, huber: 0.0023, swd: 0.0028, ept: 96.0000
    Epoch [18/50], Test Losses: mse: 0.0045, mae: 0.0408, huber: 0.0023, swd: 0.0028, ept: 96.0000
      Epoch 18 composite train-obj: 0.005627
            No improvement (0.0025), counter 1/5
    Epoch [19/50], Train Losses: mse: 0.0068, mae: 0.0482, huber: 0.0034, swd: 0.0045, ept: 95.9992
    Epoch [19/50], Val Losses: mse: 0.0024, mae: 0.0317, huber: 0.0012, swd: 0.0015, ept: 96.0000
    Epoch [19/50], Test Losses: mse: 0.0027, mae: 0.0341, huber: 0.0013, swd: 0.0017, ept: 96.0000
      Epoch 19 composite train-obj: 0.003841
            Val objective improved 0.0024 → 0.0013, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 0.0145, mae: 0.0661, huber: 0.0070, swd: 0.0098, ept: 95.9905
    Epoch [20/50], Val Losses: mse: 0.0063, mae: 0.0460, huber: 0.0032, swd: 0.0041, ept: 96.0000
    Epoch [20/50], Test Losses: mse: 0.0062, mae: 0.0480, huber: 0.0031, swd: 0.0041, ept: 96.0000
      Epoch 20 composite train-obj: 0.007928
            No improvement (0.0036), counter 1/5
    Epoch [21/50], Train Losses: mse: 0.0115, mae: 0.0540, huber: 0.0054, swd: 0.0071, ept: 95.9888
    Epoch [21/50], Val Losses: mse: 0.0029, mae: 0.0331, huber: 0.0014, swd: 0.0019, ept: 96.0000
    Epoch [21/50], Test Losses: mse: 0.0030, mae: 0.0348, huber: 0.0015, swd: 0.0019, ept: 96.0000
      Epoch 21 composite train-obj: 0.006094
            No improvement (0.0016), counter 2/5
    Epoch [22/50], Train Losses: mse: 0.0067, mae: 0.0455, huber: 0.0033, swd: 0.0046, ept: 95.9988
    Epoch [22/50], Val Losses: mse: 0.0089, mae: 0.0549, huber: 0.0044, swd: 0.0058, ept: 96.0000
    Epoch [22/50], Test Losses: mse: 0.0092, mae: 0.0582, huber: 0.0046, swd: 0.0059, ept: 96.0000
      Epoch 22 composite train-obj: 0.003765
            No improvement (0.0050), counter 3/5
    Epoch [23/50], Train Losses: mse: 0.0090, mae: 0.0548, huber: 0.0045, swd: 0.0064, ept: 95.9990
    Epoch [23/50], Val Losses: mse: 0.0094, mae: 0.0590, huber: 0.0047, swd: 0.0061, ept: 96.0000
    Epoch [23/50], Test Losses: mse: 0.0099, mae: 0.0620, huber: 0.0050, swd: 0.0065, ept: 96.0000
      Epoch 23 composite train-obj: 0.005104
            No improvement (0.0053), counter 4/5
    Epoch [24/50], Train Losses: mse: 0.0047, mae: 0.0399, huber: 0.0023, swd: 0.0031, ept: 96.0000
    Epoch [24/50], Val Losses: mse: 0.0014, mae: 0.0229, huber: 0.0007, swd: 0.0009, ept: 96.0000
    Epoch [24/50], Test Losses: mse: 0.0014, mae: 0.0238, huber: 0.0007, swd: 0.0009, ept: 96.0000
      Epoch 24 composite train-obj: 0.002651
            Val objective improved 0.0013 → 0.0008, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 0.0068, mae: 0.0438, huber: 0.0033, swd: 0.0045, ept: 95.9984
    Epoch [25/50], Val Losses: mse: 0.0077, mae: 0.0559, huber: 0.0039, swd: 0.0058, ept: 96.0000
    Epoch [25/50], Test Losses: mse: 0.0083, mae: 0.0602, huber: 0.0042, swd: 0.0064, ept: 96.0000
      Epoch 25 composite train-obj: 0.003775
            No improvement (0.0044), counter 1/5
    Epoch [26/50], Train Losses: mse: 0.0082, mae: 0.0516, huber: 0.0041, swd: 0.0053, ept: 95.9994
    Epoch [26/50], Val Losses: mse: 0.0037, mae: 0.0364, huber: 0.0018, swd: 0.0024, ept: 96.0000
    Epoch [26/50], Test Losses: mse: 0.0041, mae: 0.0391, huber: 0.0021, swd: 0.0027, ept: 96.0000
      Epoch 26 composite train-obj: 0.004588
            No improvement (0.0021), counter 2/5
    Epoch [27/50], Train Losses: mse: 0.0056, mae: 0.0428, huber: 0.0028, swd: 0.0039, ept: 95.9998
    Epoch [27/50], Val Losses: mse: 0.0046, mae: 0.0368, huber: 0.0023, swd: 0.0031, ept: 96.0000
    Epoch [27/50], Test Losses: mse: 0.0048, mae: 0.0401, huber: 0.0024, swd: 0.0032, ept: 96.0000
      Epoch 27 composite train-obj: 0.003155
            No improvement (0.0026), counter 3/5
    Epoch [28/50], Train Losses: mse: 0.0045, mae: 0.0370, huber: 0.0022, swd: 0.0032, ept: 95.9996
    Epoch [28/50], Val Losses: mse: 0.0150, mae: 0.0736, huber: 0.0074, swd: 0.0113, ept: 96.0000
    Epoch [28/50], Test Losses: mse: 0.0155, mae: 0.0776, huber: 0.0077, swd: 0.0117, ept: 96.0000
      Epoch 28 composite train-obj: 0.002544
            No improvement (0.0086), counter 4/5
    Epoch [29/50], Train Losses: mse: 0.0142, mae: 0.0628, huber: 0.0068, swd: 0.0097, ept: 95.9901
    Epoch [29/50], Val Losses: mse: 0.0063, mae: 0.0412, huber: 0.0031, swd: 0.0048, ept: 96.0000
    Epoch [29/50], Test Losses: mse: 0.0061, mae: 0.0435, huber: 0.0031, swd: 0.0046, ept: 96.0000
      Epoch 29 composite train-obj: 0.007746
    Epoch [29/50], Test Losses: mse: 0.0014, mae: 0.0238, huber: 0.0007, swd: 0.0009, ept: 96.0000
    Best round's Test MSE: 0.0014, MAE: 0.0238, SWD: 0.0009
    Best round's Validation MSE: 0.0014, MAE: 0.0229, SWD: 0.0009
    Best round's Test verification MSE : 0.0014, MAE: 0.0238, SWD: 0.0009
    Time taken: 243.92 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.1888, mae: 0.5876, huber: 0.3663, swd: 1.4129, ept: 89.9325
    Epoch [1/50], Val Losses: mse: 0.2105, mae: 0.1990, huber: 0.0541, swd: 0.1317, ept: 95.0143
    Epoch [1/50], Test Losses: mse: 0.2006, mae: 0.2088, huber: 0.0563, swd: 0.1192, ept: 95.1730
      Epoch 1 composite train-obj: 0.507633
            Val objective improved inf → 0.0673, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.0772, mae: 0.1413, huber: 0.0279, swd: 0.0439, ept: 95.6847
    Epoch [2/50], Val Losses: mse: 0.0305, mae: 0.1005, huber: 0.0134, swd: 0.0172, ept: 95.9003
    Epoch [2/50], Test Losses: mse: 0.0328, mae: 0.1058, huber: 0.0146, swd: 0.0189, ept: 95.9153
      Epoch 2 composite train-obj: 0.032295
            Val objective improved 0.0673 → 0.0151, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0328, mae: 0.1053, huber: 0.0154, swd: 0.0195, ept: 95.9590
    Epoch [3/50], Val Losses: mse: 0.0230, mae: 0.0866, huber: 0.0108, swd: 0.0121, ept: 95.9734
    Epoch [3/50], Test Losses: mse: 0.0256, mae: 0.0929, huber: 0.0121, swd: 0.0138, ept: 95.9719
      Epoch 3 composite train-obj: 0.017306
            Val objective improved 0.0151 → 0.0120, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0181, mae: 0.0807, huber: 0.0087, swd: 0.0106, ept: 95.9895
    Epoch [4/50], Val Losses: mse: 0.0126, mae: 0.0682, huber: 0.0063, swd: 0.0079, ept: 96.0000
    Epoch [4/50], Test Losses: mse: 0.0140, mae: 0.0745, huber: 0.0070, swd: 0.0089, ept: 96.0000
      Epoch 4 composite train-obj: 0.009775
            Val objective improved 0.0120 → 0.0070, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0179, mae: 0.0815, huber: 0.0087, swd: 0.0110, ept: 95.9941
    Epoch [5/50], Val Losses: mse: 0.0120, mae: 0.0670, huber: 0.0060, swd: 0.0078, ept: 96.0000
    Epoch [5/50], Test Losses: mse: 0.0121, mae: 0.0698, huber: 0.0060, swd: 0.0078, ept: 96.0000
      Epoch 5 composite train-obj: 0.009847
            Val objective improved 0.0070 → 0.0068, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0173, mae: 0.0746, huber: 0.0082, swd: 0.0108, ept: 95.9848
    Epoch [6/50], Val Losses: mse: 0.0058, mae: 0.0463, huber: 0.0029, swd: 0.0027, ept: 96.0000
    Epoch [6/50], Test Losses: mse: 0.0062, mae: 0.0488, huber: 0.0031, swd: 0.0031, ept: 96.0000
      Epoch 6 composite train-obj: 0.009332
            Val objective improved 0.0068 → 0.0031, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0180, mae: 0.0786, huber: 0.0087, swd: 0.0118, ept: 95.9887
    Epoch [7/50], Val Losses: mse: 0.0163, mae: 0.0751, huber: 0.0081, swd: 0.0099, ept: 96.0000
    Epoch [7/50], Test Losses: mse: 0.0184, mae: 0.0814, huber: 0.0091, swd: 0.0114, ept: 96.0000
      Epoch 7 composite train-obj: 0.009889
            No improvement (0.0090), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0169, mae: 0.0733, huber: 0.0080, swd: 0.0110, ept: 95.9873
    Epoch [8/50], Val Losses: mse: 0.0213, mae: 0.0906, huber: 0.0105, swd: 0.0168, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0230, mae: 0.0967, huber: 0.0114, swd: 0.0183, ept: 96.0000
      Epoch 8 composite train-obj: 0.009138
            No improvement (0.0122), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.0172, mae: 0.0792, huber: 0.0085, swd: 0.0111, ept: 95.9979
    Epoch [9/50], Val Losses: mse: 0.0072, mae: 0.0536, huber: 0.0036, swd: 0.0048, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0077, mae: 0.0564, huber: 0.0039, swd: 0.0051, ept: 96.0000
      Epoch 9 composite train-obj: 0.009583
            No improvement (0.0041), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.0141, mae: 0.0674, huber: 0.0068, swd: 0.0093, ept: 95.9906
    Epoch [10/50], Val Losses: mse: 0.0115, mae: 0.0594, huber: 0.0056, swd: 0.0064, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0123, mae: 0.0640, huber: 0.0060, swd: 0.0069, ept: 95.9984
      Epoch 10 composite train-obj: 0.007683
            No improvement (0.0063), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.0109, mae: 0.0605, huber: 0.0054, swd: 0.0071, ept: 95.9955
    Epoch [11/50], Val Losses: mse: 0.0074, mae: 0.0574, huber: 0.0037, swd: 0.0044, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0081, mae: 0.0610, huber: 0.0040, swd: 0.0047, ept: 96.0000
      Epoch 11 composite train-obj: 0.006073
    Epoch [11/50], Test Losses: mse: 0.0062, mae: 0.0488, huber: 0.0031, swd: 0.0031, ept: 96.0000
    Best round's Test MSE: 0.0062, MAE: 0.0488, SWD: 0.0031
    Best round's Validation MSE: 0.0058, MAE: 0.0463, SWD: 0.0027
    Best round's Test verification MSE : 0.0062, MAE: 0.0488, SWD: 0.0031
    Time taken: 98.26 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.3837, mae: 0.6064, huber: 0.3821, swd: 1.4031, ept: 89.1125
    Epoch [1/50], Val Losses: mse: 0.2501, mae: 0.2420, huber: 0.0707, swd: 0.1222, ept: 94.5866
    Epoch [1/50], Test Losses: mse: 0.2582, mae: 0.2573, huber: 0.0777, swd: 0.1309, ept: 94.5158
      Epoch 1 composite train-obj: 0.522434
            Val objective improved inf → 0.0830, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.1006, mae: 0.1544, huber: 0.0343, swd: 0.0506, ept: 95.3415
    Epoch [2/50], Val Losses: mse: 0.0374, mae: 0.1125, huber: 0.0175, swd: 0.0132, ept: 95.7756
    Epoch [2/50], Test Losses: mse: 0.0415, mae: 0.1190, huber: 0.0195, swd: 0.0158, ept: 95.8058
      Epoch 2 composite train-obj: 0.039392
            Val objective improved 0.0830 → 0.0188, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0428, mae: 0.1154, huber: 0.0187, swd: 0.0234, ept: 95.8503
    Epoch [3/50], Val Losses: mse: 0.0180, mae: 0.0800, huber: 0.0086, swd: 0.0091, ept: 95.9940
    Epoch [3/50], Test Losses: mse: 0.0196, mae: 0.0845, huber: 0.0094, swd: 0.0097, ept: 95.9819
      Epoch 3 composite train-obj: 0.020996
            Val objective improved 0.0188 → 0.0095, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0260, mae: 0.0965, huber: 0.0124, swd: 0.0149, ept: 95.9772
    Epoch [4/50], Val Losses: mse: 0.0325, mae: 0.1170, huber: 0.0160, swd: 0.0200, ept: 96.0000
    Epoch [4/50], Test Losses: mse: 0.0359, mae: 0.1257, huber: 0.0177, swd: 0.0217, ept: 95.9986
      Epoch 4 composite train-obj: 0.013889
            No improvement (0.0180), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0240, mae: 0.0919, huber: 0.0115, swd: 0.0141, ept: 95.9885
    Epoch [5/50], Val Losses: mse: 0.0173, mae: 0.0770, huber: 0.0085, swd: 0.0091, ept: 96.0000
    Epoch [5/50], Test Losses: mse: 0.0187, mae: 0.0814, huber: 0.0091, swd: 0.0097, ept: 95.9987
      Epoch 5 composite train-obj: 0.012937
            Val objective improved 0.0095 → 0.0094, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0215, mae: 0.0823, huber: 0.0101, swd: 0.0122, ept: 95.9805
    Epoch [6/50], Val Losses: mse: 0.0116, mae: 0.0715, huber: 0.0058, swd: 0.0067, ept: 96.0000
    Epoch [6/50], Test Losses: mse: 0.0123, mae: 0.0760, huber: 0.0061, swd: 0.0069, ept: 96.0000
      Epoch 6 composite train-obj: 0.011319
            Val objective improved 0.0094 → 0.0064, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0161, mae: 0.0757, huber: 0.0079, swd: 0.0093, ept: 95.9950
    Epoch [7/50], Val Losses: mse: 0.0241, mae: 0.0879, huber: 0.0117, swd: 0.0156, ept: 95.9993
    Epoch [7/50], Test Losses: mse: 0.0278, mae: 0.0980, huber: 0.0136, swd: 0.0176, ept: 95.9996
      Epoch 7 composite train-obj: 0.008795
            No improvement (0.0133), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0148, mae: 0.0744, huber: 0.0073, swd: 0.0090, ept: 95.9988
    Epoch [8/50], Val Losses: mse: 0.0179, mae: 0.0880, huber: 0.0089, swd: 0.0121, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0192, mae: 0.0929, huber: 0.0096, swd: 0.0130, ept: 96.0000
      Epoch 8 composite train-obj: 0.008233
            No improvement (0.0101), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.0128, mae: 0.0680, huber: 0.0063, swd: 0.0079, ept: 95.9973
    Epoch [9/50], Val Losses: mse: 0.0155, mae: 0.0721, huber: 0.0077, swd: 0.0088, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0179, mae: 0.0790, huber: 0.0089, swd: 0.0102, ept: 96.0000
      Epoch 9 composite train-obj: 0.007057
            No improvement (0.0086), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.0169, mae: 0.0780, huber: 0.0083, swd: 0.0103, ept: 95.9991
    Epoch [10/50], Val Losses: mse: 0.0087, mae: 0.0509, huber: 0.0043, swd: 0.0049, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0095, mae: 0.0550, huber: 0.0047, swd: 0.0053, ept: 96.0000
      Epoch 10 composite train-obj: 0.009343
            Val objective improved 0.0064 → 0.0048, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0138, mae: 0.0670, huber: 0.0067, swd: 0.0078, ept: 95.9961
    Epoch [11/50], Val Losses: mse: 0.0051, mae: 0.0442, huber: 0.0026, swd: 0.0030, ept: 96.0000
    Epoch [11/50], Test Losses: mse: 0.0053, mae: 0.0472, huber: 0.0027, swd: 0.0030, ept: 96.0000
      Epoch 11 composite train-obj: 0.007511
            Val objective improved 0.0048 → 0.0029, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0116, mae: 0.0605, huber: 0.0055, swd: 0.0077, ept: 95.9891
    Epoch [12/50], Val Losses: mse: 0.0085, mae: 0.0573, huber: 0.0043, swd: 0.0045, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0092, mae: 0.0618, huber: 0.0046, swd: 0.0050, ept: 96.0000
      Epoch 12 composite train-obj: 0.006249
            No improvement (0.0047), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0109, mae: 0.0587, huber: 0.0052, swd: 0.0066, ept: 95.9939
    Epoch [13/50], Val Losses: mse: 0.0045, mae: 0.0411, huber: 0.0022, swd: 0.0029, ept: 96.0000
    Epoch [13/50], Test Losses: mse: 0.0050, mae: 0.0441, huber: 0.0025, swd: 0.0032, ept: 96.0000
      Epoch 13 composite train-obj: 0.005900
            Val objective improved 0.0029 → 0.0025, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0091, mae: 0.0558, huber: 0.0045, swd: 0.0057, ept: 95.9987
    Epoch [14/50], Val Losses: mse: 0.0155, mae: 0.0777, huber: 0.0077, swd: 0.0105, ept: 96.0000
    Epoch [14/50], Test Losses: mse: 0.0155, mae: 0.0808, huber: 0.0077, swd: 0.0103, ept: 96.0000
      Epoch 14 composite train-obj: 0.005042
            No improvement (0.0087), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.0144, mae: 0.0704, huber: 0.0070, swd: 0.0092, ept: 95.9950
    Epoch [15/50], Val Losses: mse: 0.0140, mae: 0.0701, huber: 0.0069, swd: 0.0087, ept: 96.0000
    Epoch [15/50], Test Losses: mse: 0.0156, mae: 0.0757, huber: 0.0077, swd: 0.0098, ept: 96.0000
      Epoch 15 composite train-obj: 0.007961
            No improvement (0.0078), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.0080, mae: 0.0514, huber: 0.0039, swd: 0.0050, ept: 95.9992
    Epoch [16/50], Val Losses: mse: 0.0081, mae: 0.0487, huber: 0.0041, swd: 0.0047, ept: 96.0000
    Epoch [16/50], Test Losses: mse: 0.0100, mae: 0.0546, huber: 0.0050, swd: 0.0059, ept: 96.0000
      Epoch 16 composite train-obj: 0.004434
            No improvement (0.0045), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.0124, mae: 0.0636, huber: 0.0061, swd: 0.0075, ept: 95.9963
    Epoch [17/50], Val Losses: mse: 0.0053, mae: 0.0451, huber: 0.0026, swd: 0.0033, ept: 96.0000
    Epoch [17/50], Test Losses: mse: 0.0054, mae: 0.0479, huber: 0.0027, swd: 0.0032, ept: 96.0000
      Epoch 17 composite train-obj: 0.006820
            No improvement (0.0030), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.0078, mae: 0.0521, huber: 0.0039, swd: 0.0048, ept: 95.9999
    Epoch [18/50], Val Losses: mse: 0.0080, mae: 0.0594, huber: 0.0040, swd: 0.0054, ept: 96.0000
    Epoch [18/50], Test Losses: mse: 0.0084, mae: 0.0622, huber: 0.0042, swd: 0.0058, ept: 96.0000
      Epoch 18 composite train-obj: 0.004356
    Epoch [18/50], Test Losses: mse: 0.0050, mae: 0.0441, huber: 0.0025, swd: 0.0032, ept: 96.0000
    Best round's Test MSE: 0.0050, MAE: 0.0441, SWD: 0.0032
    Best round's Validation MSE: 0.0045, MAE: 0.0411, SWD: 0.0029
    Best round's Test verification MSE : 0.0050, MAE: 0.0441, SWD: 0.0032
    Time taken: 143.53 seconds
    
    ==================================================
    Experiment Summary (ACL_rossler_seq96_pred96_20250511_0147)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0042 ± 0.0021
      mae: 0.0389 ± 0.0109
      huber: 0.0021 ± 0.0010
      swd: 0.0024 ± 0.0011
      ept: 96.0000 ± 0.0000
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0039 ± 0.0018
      mae: 0.0368 ± 0.0100
      huber: 0.0019 ± 0.0009
      swd: 0.0022 ± 0.0009
      ept: 96.0000 ± 0.0000
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 485.79 seconds
    
    Experiment complete: ACL_rossler_seq96_pred96_20250511_0147
    Model: ACL
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['rossler']['channels'],# data_mgr.channels,              # ← number of features in your data
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
cfg.x_to_z_delay.enable_magnitudes_scale_shift = [False, True]
cfg.x_to_z_deri.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_x_main.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_y_main.enable_magnitudes_scale_shift = [False, True]
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    Train set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 3.5811, mae: 0.7285, huber: 0.4918, swd: 3.1932, target_std: 3.9312
    Epoch [1/50], Val Losses: mse: 2.0045, mae: 0.3579, huber: 0.1889, swd: 1.9314, target_std: 3.9491
    Epoch [1/50], Test Losses: mse: 1.8512, mae: 0.3788, huber: 0.2005, swd: 1.7675, target_std: 4.1121
      Epoch 1 composite train-obj: 0.491792
            Val objective improved inf → 0.1889, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 1.1605, mae: 0.3013, huber: 0.1372, swd: 1.0664, target_std: 3.9315
    Epoch [2/50], Val Losses: mse: 0.8013, mae: 0.3000, huber: 0.1237, swd: 0.7392, target_std: 3.9491
    Epoch [2/50], Test Losses: mse: 0.7636, mae: 0.3269, huber: 0.1390, swd: 0.6761, target_std: 4.1121
      Epoch 2 composite train-obj: 0.137221
            Val objective improved 0.1889 → 0.1237, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3334, mae: 0.2162, huber: 0.0719, swd: 0.2792, target_std: 3.9311
    Epoch [3/50], Val Losses: mse: 0.2152, mae: 0.2286, huber: 0.0645, swd: 0.1628, target_std: 3.9491
    Epoch [3/50], Test Losses: mse: 0.2084, mae: 0.2407, huber: 0.0668, swd: 0.1513, target_std: 4.1121
      Epoch 3 composite train-obj: 0.071866
            Val objective improved 0.1237 → 0.0645, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.1726, mae: 0.1683, huber: 0.0460, swd: 0.1411, target_std: 3.9324
    Epoch [4/50], Val Losses: mse: 0.3857, mae: 0.2501, huber: 0.0867, swd: 0.3489, target_std: 3.9491
    Epoch [4/50], Test Losses: mse: 0.3800, mae: 0.2684, huber: 0.0949, swd: 0.3367, target_std: 4.1121
      Epoch 4 composite train-obj: 0.045995
            No improvement (0.0867), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0909, mae: 0.1371, huber: 0.0309, swd: 0.0702, target_std: 3.9321
    Epoch [5/50], Val Losses: mse: 0.1177, mae: 0.1602, huber: 0.0375, swd: 0.1024, target_std: 3.9491
    Epoch [5/50], Test Losses: mse: 0.1198, mae: 0.1696, huber: 0.0401, swd: 0.1032, target_std: 4.1121
      Epoch 5 composite train-obj: 0.030895
            Val objective improved 0.0645 → 0.0375, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0711, mae: 0.1218, huber: 0.0250, swd: 0.0546, target_std: 3.9308
    Epoch [6/50], Val Losses: mse: 0.0623, mae: 0.1305, huber: 0.0277, swd: 0.0420, target_std: 3.9491
    Epoch [6/50], Test Losses: mse: 0.0652, mae: 0.1416, huber: 0.0299, swd: 0.0426, target_std: 4.1121
      Epoch 6 composite train-obj: 0.025040
            Val objective improved 0.0375 → 0.0277, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0807, mae: 0.1238, huber: 0.0264, swd: 0.0642, target_std: 3.9305
    Epoch [7/50], Val Losses: mse: 0.1413, mae: 0.1582, huber: 0.0427, swd: 0.1236, target_std: 3.9491
    Epoch [7/50], Test Losses: mse: 0.1381, mae: 0.1694, huber: 0.0452, swd: 0.1193, target_std: 4.1121
      Epoch 7 composite train-obj: 0.026357
            No improvement (0.0427), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0548, mae: 0.1084, huber: 0.0201, swd: 0.0424, target_std: 3.9308
    Epoch [8/50], Val Losses: mse: 0.0604, mae: 0.1280, huber: 0.0249, swd: 0.0325, target_std: 3.9491
    Epoch [8/50], Test Losses: mse: 0.0793, mae: 0.1438, huber: 0.0317, swd: 0.0484, target_std: 4.1121
      Epoch 8 composite train-obj: 0.020112
            Val objective improved 0.0277 → 0.0249, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0430, mae: 0.1004, huber: 0.0171, swd: 0.0322, target_std: 3.9312
    Epoch [9/50], Val Losses: mse: 0.0476, mae: 0.1364, huber: 0.0225, swd: 0.0286, target_std: 3.9491
    Epoch [9/50], Test Losses: mse: 0.0497, mae: 0.1444, huber: 0.0237, swd: 0.0290, target_std: 4.1121
      Epoch 9 composite train-obj: 0.017080
            Val objective improved 0.0249 → 0.0225, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.0386, mae: 0.0973, huber: 0.0157, swd: 0.0285, target_std: 3.9312
    Epoch [10/50], Val Losses: mse: 0.2075, mae: 0.1690, huber: 0.0592, swd: 0.1666, target_std: 3.9491
    Epoch [10/50], Test Losses: mse: 0.2877, mae: 0.1973, huber: 0.0757, swd: 0.2477, target_std: 4.1121
      Epoch 10 composite train-obj: 0.015676
            No improvement (0.0592), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.0400, mae: 0.0987, huber: 0.0163, swd: 0.0287, target_std: 3.9338
    Epoch [11/50], Val Losses: mse: 0.0364, mae: 0.1075, huber: 0.0170, swd: 0.0238, target_std: 3.9491
    Epoch [11/50], Test Losses: mse: 0.0393, mae: 0.1154, huber: 0.0184, swd: 0.0266, target_std: 4.1121
      Epoch 11 composite train-obj: 0.016294
            Val objective improved 0.0225 → 0.0170, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0287, mae: 0.0871, huber: 0.0125, swd: 0.0205, target_std: 3.9319
    Epoch [12/50], Val Losses: mse: 0.0659, mae: 0.1072, huber: 0.0262, swd: 0.0445, target_std: 3.9491
    Epoch [12/50], Test Losses: mse: 0.0759, mae: 0.1199, huber: 0.0313, swd: 0.0496, target_std: 4.1121
      Epoch 12 composite train-obj: 0.012496
            No improvement (0.0262), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0352, mae: 0.0950, huber: 0.0150, swd: 0.0262, target_std: 3.9314
    Epoch [13/50], Val Losses: mse: 0.0358, mae: 0.0895, huber: 0.0154, swd: 0.0254, target_std: 3.9491
    Epoch [13/50], Test Losses: mse: 0.0412, mae: 0.0973, huber: 0.0177, swd: 0.0316, target_std: 4.1121
      Epoch 13 composite train-obj: 0.014984
            Val objective improved 0.0170 → 0.0154, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0344, mae: 0.0971, huber: 0.0148, swd: 0.0246, target_std: 3.9301
    Epoch [14/50], Val Losses: mse: 0.0177, mae: 0.0823, huber: 0.0088, swd: 0.0098, target_std: 3.9491
    Epoch [14/50], Test Losses: mse: 0.0190, mae: 0.0886, huber: 0.0095, swd: 0.0108, target_std: 4.1121
      Epoch 14 composite train-obj: 0.014829
            Val objective improved 0.0154 → 0.0088, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.0217, mae: 0.0725, huber: 0.0094, swd: 0.0160, target_std: 3.9313
    Epoch [15/50], Val Losses: mse: 0.0695, mae: 0.0786, huber: 0.0201, swd: 0.0605, target_std: 3.9491
    Epoch [15/50], Test Losses: mse: 0.0725, mae: 0.0847, huber: 0.0223, swd: 0.0670, target_std: 4.1121
      Epoch 15 composite train-obj: 0.009390
            No improvement (0.0201), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.0258, mae: 0.0828, huber: 0.0112, swd: 0.0189, target_std: 3.9319
    Epoch [16/50], Val Losses: mse: 0.0172, mae: 0.0665, huber: 0.0082, swd: 0.0129, target_std: 3.9491
    Epoch [16/50], Test Losses: mse: 0.0171, mae: 0.0717, huber: 0.0083, swd: 0.0125, target_std: 4.1121
      Epoch 16 composite train-obj: 0.011229
            Val objective improved 0.0088 → 0.0082, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 0.0202, mae: 0.0727, huber: 0.0089, swd: 0.0149, target_std: 3.9335
    Epoch [17/50], Val Losses: mse: 0.0134, mae: 0.0641, huber: 0.0067, swd: 0.0090, target_std: 3.9491
    Epoch [17/50], Test Losses: mse: 0.0131, mae: 0.0675, huber: 0.0065, swd: 0.0087, target_std: 4.1121
      Epoch 17 composite train-obj: 0.008870
            Val objective improved 0.0082 → 0.0067, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 0.0217, mae: 0.0738, huber: 0.0095, swd: 0.0160, target_std: 3.9311
    Epoch [18/50], Val Losses: mse: 0.0792, mae: 0.1038, huber: 0.0268, swd: 0.0658, target_std: 3.9491
    Epoch [18/50], Test Losses: mse: 0.0835, mae: 0.1145, huber: 0.0301, swd: 0.0715, target_std: 4.1121
      Epoch 18 composite train-obj: 0.009465
            No improvement (0.0268), counter 1/5
    Epoch [19/50], Train Losses: mse: 0.0281, mae: 0.0905, huber: 0.0130, swd: 0.0195, target_std: 3.9314
    Epoch [19/50], Val Losses: mse: 0.0206, mae: 0.0731, huber: 0.0096, swd: 0.0148, target_std: 3.9491
    Epoch [19/50], Test Losses: mse: 0.0211, mae: 0.0806, huber: 0.0101, swd: 0.0148, target_std: 4.1121
      Epoch 19 composite train-obj: 0.012965
            No improvement (0.0096), counter 2/5
    Epoch [20/50], Train Losses: mse: 0.0195, mae: 0.0703, huber: 0.0086, swd: 0.0141, target_std: 3.9321
    Epoch [20/50], Val Losses: mse: 0.0729, mae: 0.0849, huber: 0.0206, swd: 0.0656, target_std: 3.9491
    Epoch [20/50], Test Losses: mse: 0.0747, mae: 0.0928, huber: 0.0228, swd: 0.0693, target_std: 4.1121
      Epoch 20 composite train-obj: 0.008559
            No improvement (0.0206), counter 3/5
    Epoch [21/50], Train Losses: mse: 0.0223, mae: 0.0719, huber: 0.0091, swd: 0.0169, target_std: 3.9330
    Epoch [21/50], Val Losses: mse: 0.1449, mae: 0.1523, huber: 0.0423, swd: 0.1153, target_std: 3.9491
    Epoch [21/50], Test Losses: mse: 0.1462, mae: 0.1642, huber: 0.0456, swd: 0.1200, target_std: 4.1121
      Epoch 21 composite train-obj: 0.009135
            No improvement (0.0423), counter 4/5
    Epoch [22/50], Train Losses: mse: 0.0275, mae: 0.0846, huber: 0.0119, swd: 0.0196, target_std: 3.9312
    Epoch [22/50], Val Losses: mse: 0.0136, mae: 0.0671, huber: 0.0067, swd: 0.0078, target_std: 3.9491
    Epoch [22/50], Test Losses: mse: 0.0152, mae: 0.0734, huber: 0.0075, swd: 0.0096, target_std: 4.1121
      Epoch 22 composite train-obj: 0.011879
    Epoch [22/50], Test Losses: mse: 0.0131, mae: 0.0675, huber: 0.0065, swd: 0.0087, target_std: 4.1121
    Best round's Test MSE: 0.0131, MAE: 0.0675, SWD: 0.0087
    Best round's Validation MSE: 0.0134, MAE: 0.0641
    Best round's Test verification MSE : 0.0131, MAE: 0.0675, SWD: 0.0087
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 3.5251, mae: 0.7211, huber: 0.4852, swd: 3.1396, target_std: 3.9320
    Epoch [1/50], Val Losses: mse: 2.1780, mae: 0.4858, huber: 0.2584, swd: 2.0795, target_std: 3.9491
    Epoch [1/50], Test Losses: mse: 2.0368, mae: 0.5125, huber: 0.2753, swd: 1.9249, target_std: 4.1121
      Epoch 1 composite train-obj: 0.485238
            Val objective improved inf → 0.2584, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 1.2633, mae: 0.3037, huber: 0.1407, swd: 1.1774, target_std: 3.9331
    Epoch [2/50], Val Losses: mse: 0.7786, mae: 0.2492, huber: 0.1059, swd: 0.7047, target_std: 3.9491
    Epoch [2/50], Test Losses: mse: 0.6994, mae: 0.2592, huber: 0.1086, swd: 0.6188, target_std: 4.1121
      Epoch 2 composite train-obj: 0.140721
            Val objective improved 0.2584 → 0.1059, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3611, mae: 0.2187, huber: 0.0708, swd: 0.2884, target_std: 3.9325
    Epoch [3/50], Val Losses: mse: 0.1288, mae: 0.1607, huber: 0.0436, swd: 0.0708, target_std: 3.9491
    Epoch [3/50], Test Losses: mse: 0.1404, mae: 0.1755, huber: 0.0501, swd: 0.0859, target_std: 4.1121
      Epoch 3 composite train-obj: 0.070802
            Val objective improved 0.1059 → 0.0436, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.1160, mae: 0.1536, huber: 0.0365, swd: 0.0776, target_std: 3.9301
    Epoch [4/50], Val Losses: mse: 0.1076, mae: 0.1358, huber: 0.0315, swd: 0.0803, target_std: 3.9491
    Epoch [4/50], Test Losses: mse: 0.1043, mae: 0.1451, huber: 0.0329, swd: 0.0745, target_std: 4.1121
      Epoch 4 composite train-obj: 0.036531
            Val objective improved 0.0436 → 0.0315, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0692, mae: 0.1281, huber: 0.0262, swd: 0.0439, target_std: 3.9310
    Epoch [5/50], Val Losses: mse: 0.0495, mae: 0.1208, huber: 0.0218, swd: 0.0236, target_std: 3.9491
    Epoch [5/50], Test Losses: mse: 0.0543, mae: 0.1308, huber: 0.0243, swd: 0.0268, target_std: 4.1121
      Epoch 5 composite train-obj: 0.026187
            Val objective improved 0.0315 → 0.0218, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0469, mae: 0.1065, huber: 0.0187, swd: 0.0305, target_std: 3.9308
    Epoch [6/50], Val Losses: mse: 0.0661, mae: 0.1237, huber: 0.0259, swd: 0.0471, target_std: 3.9491
    Epoch [6/50], Test Losses: mse: 0.0677, mae: 0.1346, huber: 0.0278, swd: 0.0460, target_std: 4.1121
      Epoch 6 composite train-obj: 0.018711
            No improvement (0.0259), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0446, mae: 0.1076, huber: 0.0187, swd: 0.0293, target_std: 3.9312
    Epoch [7/50], Val Losses: mse: 0.0390, mae: 0.1118, huber: 0.0189, swd: 0.0217, target_std: 3.9491
    Epoch [7/50], Test Losses: mse: 0.0435, mae: 0.1230, huber: 0.0212, swd: 0.0252, target_std: 4.1121
      Epoch 7 composite train-obj: 0.018726
            Val objective improved 0.0218 → 0.0189, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0367, mae: 0.1014, huber: 0.0161, swd: 0.0236, target_std: 3.9322
    Epoch [8/50], Val Losses: mse: 0.0346, mae: 0.0851, huber: 0.0141, swd: 0.0222, target_std: 3.9491
    Epoch [8/50], Test Losses: mse: 0.0370, mae: 0.0918, huber: 0.0160, swd: 0.0248, target_std: 4.1121
      Epoch 8 composite train-obj: 0.016117
            Val objective improved 0.0189 → 0.0141, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0363, mae: 0.0974, huber: 0.0153, swd: 0.0233, target_std: 3.9326
    Epoch [9/50], Val Losses: mse: 0.1990, mae: 0.1636, huber: 0.0491, swd: 0.1692, target_std: 3.9491
    Epoch [9/50], Test Losses: mse: 0.1931, mae: 0.1763, huber: 0.0520, swd: 0.1639, target_std: 4.1121
      Epoch 9 composite train-obj: 0.015343
            No improvement (0.0491), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.0454, mae: 0.1022, huber: 0.0173, swd: 0.0321, target_std: 3.9289
    Epoch [10/50], Val Losses: mse: 0.0172, mae: 0.0813, huber: 0.0083, swd: 0.0090, target_std: 3.9491
    Epoch [10/50], Test Losses: mse: 0.0177, mae: 0.0834, huber: 0.0086, swd: 0.0093, target_std: 4.1121
      Epoch 10 composite train-obj: 0.017310
            Val objective improved 0.0141 → 0.0083, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0305, mae: 0.0861, huber: 0.0122, swd: 0.0208, target_std: 3.9278
    Epoch [11/50], Val Losses: mse: 0.1302, mae: 0.1447, huber: 0.0365, swd: 0.0986, target_std: 3.9491
    Epoch [11/50], Test Losses: mse: 0.1278, mae: 0.1546, huber: 0.0386, swd: 0.0972, target_std: 4.1121
      Epoch 11 composite train-obj: 0.012159
            No improvement (0.0365), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.0258, mae: 0.0802, huber: 0.0112, swd: 0.0172, target_std: 3.9314
    Epoch [12/50], Val Losses: mse: 0.0648, mae: 0.1129, huber: 0.0251, swd: 0.0374, target_std: 3.9491
    Epoch [12/50], Test Losses: mse: 0.0679, mae: 0.1257, huber: 0.0279, swd: 0.0429, target_std: 4.1121
      Epoch 12 composite train-obj: 0.011158
            No improvement (0.0251), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.0267, mae: 0.0849, huber: 0.0117, swd: 0.0179, target_std: 3.9317
    Epoch [13/50], Val Losses: mse: 0.0394, mae: 0.1097, huber: 0.0181, swd: 0.0278, target_std: 3.9491
    Epoch [13/50], Test Losses: mse: 0.0402, mae: 0.1175, huber: 0.0188, swd: 0.0275, target_std: 4.1121
      Epoch 13 composite train-obj: 0.011743
            No improvement (0.0181), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.0252, mae: 0.0871, huber: 0.0114, swd: 0.0169, target_std: 3.9320
    Epoch [14/50], Val Losses: mse: 0.0542, mae: 0.0978, huber: 0.0222, swd: 0.0397, target_std: 3.9491
    Epoch [14/50], Test Losses: mse: 0.0590, mae: 0.1093, huber: 0.0250, swd: 0.0436, target_std: 4.1121
      Epoch 14 composite train-obj: 0.011401
            No improvement (0.0222), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.0234, mae: 0.0808, huber: 0.0104, swd: 0.0159, target_std: 3.9306
    Epoch [15/50], Val Losses: mse: 0.0288, mae: 0.0930, huber: 0.0135, swd: 0.0193, target_std: 3.9491
    Epoch [15/50], Test Losses: mse: 0.0307, mae: 0.0998, huber: 0.0145, swd: 0.0199, target_std: 4.1121
      Epoch 15 composite train-obj: 0.010351
    Epoch [15/50], Test Losses: mse: 0.0177, mae: 0.0834, huber: 0.0086, swd: 0.0093, target_std: 4.1121
    Best round's Test MSE: 0.0177, MAE: 0.0834, SWD: 0.0093
    Best round's Validation MSE: 0.0172, MAE: 0.0813
    Best round's Test verification MSE : 0.0177, MAE: 0.0834, SWD: 0.0093
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 3.3862, mae: 0.6770, huber: 0.4510, swd: 2.7974, target_std: 3.9326
    Epoch [1/50], Val Losses: mse: 2.1448, mae: 0.4028, huber: 0.2135, swd: 1.9464, target_std: 3.9491
    Epoch [1/50], Test Losses: mse: 1.9973, mae: 0.4242, huber: 0.2277, swd: 1.7962, target_std: 4.1121
      Epoch 1 composite train-obj: 0.451046
            Val objective improved inf → 0.2135, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 1.3699, mae: 0.3082, huber: 0.1490, swd: 1.2032, target_std: 3.9327
    Epoch [2/50], Val Losses: mse: 1.0071, mae: 0.3484, huber: 0.1526, swd: 0.8322, target_std: 3.9491
    Epoch [2/50], Test Losses: mse: 0.9299, mae: 0.3665, huber: 0.1606, swd: 0.7482, target_std: 4.1121
      Epoch 2 composite train-obj: 0.148954
            Val objective improved 0.2135 → 0.1526, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4031, mae: 0.2154, huber: 0.0730, swd: 0.2956, target_std: 3.9304
    Epoch [3/50], Val Losses: mse: 0.1520, mae: 0.1710, huber: 0.0445, swd: 0.0754, target_std: 3.9491
    Epoch [3/50], Test Losses: mse: 0.1839, mae: 0.1884, huber: 0.0535, swd: 0.1037, target_std: 4.1121
      Epoch 3 composite train-obj: 0.073025
            Val objective improved 0.1526 → 0.0445, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.1780, mae: 0.1697, huber: 0.0459, swd: 0.1270, target_std: 3.9315
    Epoch [4/50], Val Losses: mse: 0.3508, mae: 0.2332, huber: 0.0917, swd: 0.2495, target_std: 3.9491
    Epoch [4/50], Test Losses: mse: 0.5272, mae: 0.2740, huber: 0.1217, swd: 0.4064, target_std: 4.1121
      Epoch 4 composite train-obj: 0.045870
            No improvement (0.0917), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.1024, mae: 0.1390, huber: 0.0318, swd: 0.0704, target_std: 3.9334
    Epoch [5/50], Val Losses: mse: 0.4190, mae: 0.2203, huber: 0.0893, swd: 0.3399, target_std: 3.9491
    Epoch [5/50], Test Losses: mse: 0.6045, mae: 0.2482, huber: 0.1096, swd: 0.5153, target_std: 4.1121
      Epoch 5 composite train-obj: 0.031823
            No improvement (0.0893), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.0847, mae: 0.1283, huber: 0.0273, swd: 0.0588, target_std: 3.9292
    Epoch [6/50], Val Losses: mse: 0.0873, mae: 0.1554, huber: 0.0333, swd: 0.0615, target_std: 3.9491
    Epoch [6/50], Test Losses: mse: 0.0889, mae: 0.1634, huber: 0.0349, swd: 0.0613, target_std: 4.1121
      Epoch 6 composite train-obj: 0.027274
            Val objective improved 0.0445 → 0.0333, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0463, mae: 0.1056, huber: 0.0183, swd: 0.0296, target_std: 3.9305
    Epoch [7/50], Val Losses: mse: 0.0346, mae: 0.1060, huber: 0.0159, swd: 0.0177, target_std: 3.9491
    Epoch [7/50], Test Losses: mse: 0.0373, mae: 0.1141, huber: 0.0174, swd: 0.0193, target_std: 4.1121
      Epoch 7 composite train-obj: 0.018267
            Val objective improved 0.0333 → 0.0159, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0504, mae: 0.1064, huber: 0.0189, swd: 0.0335, target_std: 3.9315
    Epoch [8/50], Val Losses: mse: 0.0742, mae: 0.1144, huber: 0.0255, swd: 0.0505, target_std: 3.9491
    Epoch [8/50], Test Losses: mse: 0.1019, mae: 0.1311, huber: 0.0330, swd: 0.0767, target_std: 4.1121
      Epoch 8 composite train-obj: 0.018943
            No improvement (0.0255), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.0376, mae: 0.0969, huber: 0.0157, swd: 0.0239, target_std: 3.9331
    Epoch [9/50], Val Losses: mse: 0.1046, mae: 0.1320, huber: 0.0366, swd: 0.0610, target_std: 3.9491
    Epoch [9/50], Test Losses: mse: 0.1067, mae: 0.1414, huber: 0.0389, swd: 0.0657, target_std: 4.1121
      Epoch 9 composite train-obj: 0.015697
            No improvement (0.0366), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.0401, mae: 0.0989, huber: 0.0161, swd: 0.0262, target_std: 3.9324
    Epoch [10/50], Val Losses: mse: 0.0362, mae: 0.1005, huber: 0.0170, swd: 0.0220, target_std: 3.9491
    Epoch [10/50], Test Losses: mse: 0.0402, mae: 0.1099, huber: 0.0191, swd: 0.0251, target_std: 4.1121
      Epoch 10 composite train-obj: 0.016114
            No improvement (0.0170), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.0335, mae: 0.0901, huber: 0.0136, swd: 0.0216, target_std: 3.9319
    Epoch [11/50], Val Losses: mse: 0.0221, mae: 0.0725, huber: 0.0098, swd: 0.0158, target_std: 3.9491
    Epoch [11/50], Test Losses: mse: 0.0234, mae: 0.0798, huber: 0.0105, swd: 0.0162, target_std: 4.1121
      Epoch 11 composite train-obj: 0.013615
            Val objective improved 0.0159 → 0.0098, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0302, mae: 0.0887, huber: 0.0128, swd: 0.0194, target_std: 3.9297
    Epoch [12/50], Val Losses: mse: 0.0734, mae: 0.1132, huber: 0.0273, swd: 0.0464, target_std: 3.9491
    Epoch [12/50], Test Losses: mse: 0.0893, mae: 0.1312, huber: 0.0341, swd: 0.0590, target_std: 4.1121
      Epoch 12 composite train-obj: 0.012769
            No improvement (0.0273), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0280, mae: 0.0822, huber: 0.0119, swd: 0.0187, target_std: 3.9291
    Epoch [13/50], Val Losses: mse: 0.0285, mae: 0.0880, huber: 0.0129, swd: 0.0158, target_std: 3.9491
    Epoch [13/50], Test Losses: mse: 0.0298, mae: 0.0938, huber: 0.0136, swd: 0.0171, target_std: 4.1121
      Epoch 13 composite train-obj: 0.011879
            No improvement (0.0129), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.0289, mae: 0.0867, huber: 0.0127, swd: 0.0179, target_std: 3.9336
    Epoch [14/50], Val Losses: mse: 0.0807, mae: 0.1272, huber: 0.0292, swd: 0.0592, target_std: 3.9491
    Epoch [14/50], Test Losses: mse: 0.0851, mae: 0.1375, huber: 0.0321, swd: 0.0629, target_std: 4.1121
      Epoch 14 composite train-obj: 0.012749
            No improvement (0.0292), counter 3/5
    Epoch [15/50], Train Losses: mse: 0.0294, mae: 0.0864, huber: 0.0126, swd: 0.0191, target_std: 3.9293
    Epoch [15/50], Val Losses: mse: 0.1064, mae: 0.1355, huber: 0.0376, swd: 0.0595, target_std: 3.9491
    Epoch [15/50], Test Losses: mse: 0.1430, mae: 0.1564, huber: 0.0482, swd: 0.0898, target_std: 4.1121
      Epoch 15 composite train-obj: 0.012585
            No improvement (0.0376), counter 4/5
    Epoch [16/50], Train Losses: mse: 0.0310, mae: 0.0894, huber: 0.0133, swd: 0.0198, target_std: 3.9322
    Epoch [16/50], Val Losses: mse: 0.0318, mae: 0.0786, huber: 0.0129, swd: 0.0231, target_std: 3.9491
    Epoch [16/50], Test Losses: mse: 0.0318, mae: 0.0849, huber: 0.0133, swd: 0.0230, target_std: 4.1121
      Epoch 16 composite train-obj: 0.013323
    Epoch [16/50], Test Losses: mse: 0.0234, mae: 0.0798, huber: 0.0105, swd: 0.0162, target_std: 4.1121
    Best round's Test MSE: 0.0234, MAE: 0.0798, SWD: 0.0162
    Best round's Validation MSE: 0.0221, MAE: 0.0725
    Best round's Test verification MSE : 0.0234, MAE: 0.0798, SWD: 0.0162
    
    ==================================================
    Experiment Summary (ACL_rossler_seq96_pred196_20250430_1315)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0181 ± 0.0042
      mae: 0.0769 ± 0.0068
      huber: 0.0085 ± 0.0016
      swd: 0.0114 ± 0.0034
      target_std: 4.1121 ± 0.0000
      count: 39.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0176 ± 0.0036
      mae: 0.0726 ± 0.0070
      huber: 0.0083 ± 0.0013
      swd: 0.0113 ± 0.0032
      target_std: 3.9491 ± 0.0000
      count: 39.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_rossler_seq96_pred196_20250430_1315
    Model: ACL
    Dataset: rossler
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
    channels=data_mgr.datasets['rossler']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 3.8776, mae: 0.7598, huber: 0.5186, swd: 3.0869, ept: 295.3539
    Epoch [1/50], Val Losses: mse: 3.5122, mae: 0.7177, huber: 0.4618, swd: 2.8246, ept: 306.8404
    Epoch [1/50], Test Losses: mse: 3.4079, mae: 0.7807, huber: 0.5122, swd: 2.6877, ept: 302.7580
      Epoch 1 composite train-obj: 0.518581
            Val objective improved inf → 0.4618, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.1544, mae: 0.3832, huber: 0.2143, swd: 1.9093, ept: 317.2276
    Epoch [2/50], Val Losses: mse: 2.0938, mae: 0.3672, huber: 0.1992, swd: 1.7819, ept: 320.4488
    Epoch [2/50], Test Losses: mse: 1.9077, mae: 0.3922, huber: 0.2115, swd: 1.5919, ept: 317.4056
      Epoch 2 composite train-obj: 0.214327
            Val objective improved 0.4618 → 0.1992, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.0364, mae: 0.2730, huber: 0.1278, swd: 0.7180, ept: 322.2916
    Epoch [3/50], Val Losses: mse: 3.5493, mae: 0.7852, huber: 0.5077, swd: 0.4943, ept: 295.6060
    Epoch [3/50], Test Losses: mse: 3.8181, mae: 0.8714, huber: 0.5795, swd: 0.5963, ept: 290.5742
      Epoch 3 composite train-obj: 0.127817
            No improvement (0.5077), counter 1/5
    Epoch [4/50], Train Losses: mse: 1.3127, mae: 0.2867, huber: 0.1460, swd: 1.0303, ept: 320.9031
    Epoch [4/50], Val Losses: mse: 0.7056, mae: 0.3995, huber: 0.1725, swd: 0.4329, ept: 322.9530
    Epoch [4/50], Test Losses: mse: 0.6773, mae: 0.4233, huber: 0.1838, swd: 0.3997, ept: 321.6268
      Epoch 4 composite train-obj: 0.146041
            Val objective improved 0.1992 → 0.1725, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.1787, mae: 0.1524, huber: 0.0436, swd: 0.0811, ept: 327.0270
    Epoch [5/50], Val Losses: mse: 0.0875, mae: 0.1272, huber: 0.0262, swd: 0.0399, ept: 328.8217
    Epoch [5/50], Test Losses: mse: 0.0826, mae: 0.1349, huber: 0.0268, swd: 0.0350, ept: 330.3614
      Epoch 5 composite train-obj: 0.043646
            Val objective improved 0.1725 → 0.0262, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0790, mae: 0.1176, huber: 0.0251, swd: 0.0337, ept: 330.6471
    Epoch [6/50], Val Losses: mse: 0.1292, mae: 0.1374, huber: 0.0376, swd: 0.0464, ept: 329.4953
    Epoch [6/50], Test Losses: mse: 0.1286, mae: 0.1484, huber: 0.0405, swd: 0.0469, ept: 329.5145
      Epoch 6 composite train-obj: 0.025115
            No improvement (0.0376), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0483, mae: 0.1046, huber: 0.0185, swd: 0.0196, ept: 332.6186
    Epoch [7/50], Val Losses: mse: 1.0810, mae: 0.4971, huber: 0.2713, swd: 0.5574, ept: 319.3124
    Epoch [7/50], Test Losses: mse: 1.2212, mae: 0.5571, huber: 0.3176, swd: 0.6588, ept: 318.7361
      Epoch 7 composite train-obj: 0.018507
            No improvement (0.2713), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.1050, mae: 0.1459, huber: 0.0373, swd: 0.0440, ept: 333.0827
    Epoch [8/50], Val Losses: mse: 0.0653, mae: 0.1202, huber: 0.0272, swd: 0.0394, ept: 333.9805
    Epoch [8/50], Test Losses: mse: 0.0775, mae: 0.1339, huber: 0.0321, swd: 0.0489, ept: 333.9591
      Epoch 8 composite train-obj: 0.037254
            No improvement (0.0272), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.0318, mae: 0.0905, huber: 0.0137, swd: 0.0126, ept: 334.5181
    Epoch [9/50], Val Losses: mse: 0.1001, mae: 0.1414, huber: 0.0375, swd: 0.0318, ept: 331.3568
    Epoch [9/50], Test Losses: mse: 0.0993, mae: 0.1528, huber: 0.0392, swd: 0.0330, ept: 331.9101
      Epoch 9 composite train-obj: 0.013684
            No improvement (0.0375), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.0265, mae: 0.0862, huber: 0.0120, swd: 0.0107, ept: 335.0109
    Epoch [10/50], Val Losses: mse: 0.0328, mae: 0.1056, huber: 0.0152, swd: 0.0189, ept: 335.4705
    Epoch [10/50], Test Losses: mse: 0.0371, mae: 0.1134, huber: 0.0169, swd: 0.0219, ept: 335.3957
      Epoch 10 composite train-obj: 0.011988
            Val objective improved 0.0262 → 0.0152, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0240, mae: 0.0823, huber: 0.0107, swd: 0.0095, ept: 335.1690
    Epoch [11/50], Val Losses: mse: 0.0180, mae: 0.0758, huber: 0.0085, swd: 0.0059, ept: 335.5493
    Epoch [11/50], Test Losses: mse: 0.0184, mae: 0.0788, huber: 0.0087, swd: 0.0064, ept: 335.5398
      Epoch 11 composite train-obj: 0.010724
            Val objective improved 0.0152 → 0.0085, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0185, mae: 0.0734, huber: 0.0085, swd: 0.0082, ept: 335.4362
    Epoch [12/50], Val Losses: mse: 0.0641, mae: 0.1355, huber: 0.0263, swd: 0.0339, ept: 334.0321
    Epoch [12/50], Test Losses: mse: 0.0690, mae: 0.1448, huber: 0.0292, swd: 0.0381, ept: 334.2930
      Epoch 12 composite train-obj: 0.008539
            No improvement (0.0263), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0257, mae: 0.0826, huber: 0.0112, swd: 0.0111, ept: 335.2227
    Epoch [13/50], Val Losses: mse: 0.0276, mae: 0.0845, huber: 0.0123, swd: 0.0156, ept: 335.5156
    Epoch [13/50], Test Losses: mse: 0.0309, mae: 0.0917, huber: 0.0138, swd: 0.0179, ept: 335.5351
      Epoch 13 composite train-obj: 0.011235
            No improvement (0.0123), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.0166, mae: 0.0712, huber: 0.0079, swd: 0.0073, ept: 335.7121
    Epoch [14/50], Val Losses: mse: 0.0803, mae: 0.1380, huber: 0.0317, swd: 0.0435, ept: 329.7547
    Epoch [14/50], Test Losses: mse: 0.0829, mae: 0.1464, huber: 0.0341, swd: 0.0482, ept: 330.8201
      Epoch 14 composite train-obj: 0.007923
            No improvement (0.0317), counter 3/5
    Epoch [15/50], Train Losses: mse: 0.0173, mae: 0.0750, huber: 0.0084, swd: 0.0079, ept: 335.7996
    Epoch [15/50], Val Losses: mse: 0.4566, mae: 0.3252, huber: 0.1572, swd: 0.1655, ept: 328.6504
    Epoch [15/50], Test Losses: mse: 0.6314, mae: 0.3778, huber: 0.1960, swd: 0.2322, ept: 327.3637
      Epoch 15 composite train-obj: 0.008397
            No improvement (0.1572), counter 4/5
    Epoch [16/50], Train Losses: mse: 0.0426, mae: 0.0946, huber: 0.0180, swd: 0.0168, ept: 335.2733
    Epoch [16/50], Val Losses: mse: 0.0463, mae: 0.1266, huber: 0.0221, swd: 0.0156, ept: 335.8145
    Epoch [16/50], Test Losses: mse: 0.0496, mae: 0.1370, huber: 0.0239, swd: 0.0181, ept: 335.7647
      Epoch 16 composite train-obj: 0.018039
    Epoch [16/50], Test Losses: mse: 0.0184, mae: 0.0788, huber: 0.0087, swd: 0.0064, ept: 335.5398
    Best round's Test MSE: 0.0184, MAE: 0.0788, SWD: 0.0064
    Best round's Validation MSE: 0.0180, MAE: 0.0758, SWD: 0.0059
    Best round's Test verification MSE : 0.0184, MAE: 0.0788, SWD: 0.0064
    Time taken: 126.72 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 3.9715, mae: 0.7764, huber: 0.5332, swd: 3.1848, ept: 292.5705
    Epoch [1/50], Val Losses: mse: 2.7252, mae: 0.4554, huber: 0.2534, swd: 2.5218, ept: 315.2048
    Epoch [1/50], Test Losses: mse: 2.5267, mae: 0.4843, huber: 0.2720, swd: 2.3120, ept: 309.5515
      Epoch 1 composite train-obj: 0.533226
            Val objective improved inf → 0.2534, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.2007, mae: 0.3663, huber: 0.2068, swd: 2.0395, ept: 315.3855
    Epoch [2/50], Val Losses: mse: 3.4045, mae: 0.6898, huber: 0.4571, swd: 2.6908, ept: 304.7373
    Epoch [2/50], Test Losses: mse: 3.4664, mae: 0.7645, huber: 0.5169, swd: 2.6105, ept: 297.0964
      Epoch 2 composite train-obj: 0.206775
            No improvement (0.4571), counter 1/5
    Epoch [3/50], Train Losses: mse: 1.3663, mae: 0.3164, huber: 0.1604, swd: 1.0859, ept: 318.2242
    Epoch [3/50], Val Losses: mse: 0.5872, mae: 0.2527, huber: 0.1007, swd: 0.3560, ept: 319.6615
    Epoch [3/50], Test Losses: mse: 0.5380, mae: 0.2659, huber: 0.1037, swd: 0.3180, ept: 317.2963
      Epoch 3 composite train-obj: 0.160420
            Val objective improved 0.2534 → 0.1007, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2750, mae: 0.1760, huber: 0.0554, swd: 0.1680, ept: 325.4135
    Epoch [4/50], Val Losses: mse: 0.2599, mae: 0.2384, huber: 0.0754, swd: 0.1306, ept: 325.8383
    Epoch [4/50], Test Losses: mse: 0.2587, mae: 0.2608, huber: 0.0838, swd: 0.1251, ept: 326.5206
      Epoch 4 composite train-obj: 0.055425
            Val objective improved 0.1007 → 0.0754, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.1085, mae: 0.1381, huber: 0.0316, swd: 0.0585, ept: 329.8570
    Epoch [5/50], Val Losses: mse: 0.0708, mae: 0.1151, huber: 0.0217, swd: 0.0407, ept: 331.1944
    Epoch [5/50], Test Losses: mse: 0.0676, mae: 0.1233, huber: 0.0225, swd: 0.0365, ept: 332.1436
      Epoch 5 composite train-obj: 0.031610
            Val objective improved 0.0754 → 0.0217, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0629, mae: 0.1163, huber: 0.0223, swd: 0.0297, ept: 332.2660
    Epoch [6/50], Val Losses: mse: 0.9399, mae: 0.4533, huber: 0.2462, swd: 0.2574, ept: 319.9680
    Epoch [6/50], Test Losses: mse: 1.2086, mae: 0.5295, huber: 0.3056, swd: 0.2891, ept: 317.6269
      Epoch 6 composite train-obj: 0.022285
            No improvement (0.2462), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.1060, mae: 0.1324, huber: 0.0318, swd: 0.0430, ept: 331.9908
    Epoch [7/50], Val Losses: mse: 0.0713, mae: 0.1213, huber: 0.0277, swd: 0.0205, ept: 329.0493
    Epoch [7/50], Test Losses: mse: 0.0721, mae: 0.1324, huber: 0.0297, swd: 0.0234, ept: 330.2168
      Epoch 7 composite train-obj: 0.031763
            No improvement (0.0277), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.0263, mae: 0.0812, huber: 0.0109, swd: 0.0111, ept: 334.9440
    Epoch [8/50], Val Losses: mse: 0.0910, mae: 0.1479, huber: 0.0404, swd: 0.0262, ept: 334.1512
    Epoch [8/50], Test Losses: mse: 0.0993, mae: 0.1630, huber: 0.0452, swd: 0.0327, ept: 334.6209
      Epoch 8 composite train-obj: 0.010888
            No improvement (0.0404), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.0306, mae: 0.0921, huber: 0.0134, swd: 0.0129, ept: 335.3328
    Epoch [9/50], Val Losses: mse: 0.5044, mae: 0.2236, huber: 0.0859, swd: 0.4501, ept: 326.7503
    Epoch [9/50], Test Losses: mse: 0.5898, mae: 0.2442, huber: 0.0980, swd: 0.5411, ept: 327.4242
      Epoch 9 composite train-obj: 0.013396
            No improvement (0.0859), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.0514, mae: 0.0948, huber: 0.0164, swd: 0.0294, ept: 334.8242
    Epoch [10/50], Val Losses: mse: 0.0721, mae: 0.1447, huber: 0.0307, swd: 0.0372, ept: 330.9736
    Epoch [10/50], Test Losses: mse: 0.0755, mae: 0.1571, huber: 0.0331, swd: 0.0408, ept: 332.0574
      Epoch 10 composite train-obj: 0.016370
    Epoch [10/50], Test Losses: mse: 0.0676, mae: 0.1233, huber: 0.0225, swd: 0.0365, ept: 332.1436
    Best round's Test MSE: 0.0676, MAE: 0.1233, SWD: 0.0365
    Best round's Validation MSE: 0.0708, MAE: 0.1151, SWD: 0.0407
    Best round's Test verification MSE : 0.0676, MAE: 0.1233, SWD: 0.0365
    Time taken: 79.61 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.0374, mae: 0.7941, huber: 0.5476, swd: 3.2043, ept: 293.3004
    Epoch [1/50], Val Losses: mse: 3.8210, mae: 0.7616, huber: 0.5040, swd: 2.7885, ept: 302.3010
    Epoch [1/50], Test Losses: mse: 3.8836, mae: 0.8485, huber: 0.5794, swd: 2.6214, ept: 296.2852
      Epoch 1 composite train-obj: 0.547574
            Val objective improved inf → 0.5040, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 1.9050, mae: 0.3798, huber: 0.2063, swd: 1.6164, ept: 316.3405
    Epoch [2/50], Val Losses: mse: 1.3300, mae: 0.4557, huber: 0.2305, swd: 0.7615, ept: 320.9628
    Epoch [2/50], Test Losses: mse: 1.2892, mae: 0.4984, huber: 0.2588, swd: 0.6823, ept: 319.3437
      Epoch 2 composite train-obj: 0.206346
            Val objective improved 0.5040 → 0.2305, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3985, mae: 0.2164, huber: 0.0770, swd: 0.2103, ept: 326.1145
    Epoch [3/50], Val Losses: mse: 0.3927, mae: 0.2121, huber: 0.0808, swd: 0.0975, ept: 325.6403
    Epoch [3/50], Test Losses: mse: 0.3845, mae: 0.2235, huber: 0.0857, swd: 0.0982, ept: 324.5827
      Epoch 3 composite train-obj: 0.077044
            Val objective improved 0.2305 → 0.0808, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0889, mae: 0.1354, huber: 0.0294, swd: 0.0335, ept: 331.4028
    Epoch [4/50], Val Losses: mse: 1.3337, mae: 0.3547, huber: 0.1759, swd: 0.2256, ept: 322.9030
    Epoch [4/50], Test Losses: mse: 1.2974, mae: 0.3808, huber: 0.1903, swd: 0.2396, ept: 320.6471
      Epoch 4 composite train-obj: 0.029393
            No improvement (0.1759), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4186, mae: 0.1790, huber: 0.0643, swd: 0.2628, ept: 328.0997
    Epoch [5/50], Val Losses: mse: 0.2437, mae: 0.2214, huber: 0.0775, swd: 0.0632, ept: 324.0131
    Epoch [5/50], Test Losses: mse: 0.2389, mae: 0.2390, huber: 0.0819, swd: 0.0634, ept: 326.4790
      Epoch 5 composite train-obj: 0.064293
            Val objective improved 0.0808 → 0.0775, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0696, mae: 0.1176, huber: 0.0236, swd: 0.0200, ept: 332.4895
    Epoch [6/50], Val Losses: mse: 0.1153, mae: 0.1334, huber: 0.0358, swd: 0.0343, ept: 330.0280
    Epoch [6/50], Test Losses: mse: 0.1160, mae: 0.1428, huber: 0.0384, swd: 0.0374, ept: 329.3594
      Epoch 6 composite train-obj: 0.023557
            Val objective improved 0.0775 → 0.0358, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0503, mae: 0.1111, huber: 0.0199, swd: 0.0156, ept: 333.7026
    Epoch [7/50], Val Losses: mse: 2.2834, mae: 0.3088, huber: 0.1700, swd: 2.0433, ept: 323.7386
    Epoch [7/50], Test Losses: mse: 2.5298, mae: 0.3424, huber: 0.1919, swd: 2.3194, ept: 322.4473
      Epoch 7 composite train-obj: 0.019901
            No improvement (0.1700), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.2650, mae: 0.1248, huber: 0.0361, swd: 0.2028, ept: 332.5797
    Epoch [8/50], Val Losses: mse: 0.1454, mae: 0.1283, huber: 0.0350, swd: 0.1001, ept: 333.1145
    Epoch [8/50], Test Losses: mse: 0.1410, mae: 0.1365, huber: 0.0368, swd: 0.0968, ept: 333.3728
      Epoch 8 composite train-obj: 0.036097
            Val objective improved 0.0358 → 0.0350, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0479, mae: 0.0992, huber: 0.0170, swd: 0.0212, ept: 334.7644
    Epoch [9/50], Val Losses: mse: 0.8320, mae: 0.4471, huber: 0.2303, swd: 0.1583, ept: 323.5285
    Epoch [9/50], Test Losses: mse: 0.9363, mae: 0.5041, huber: 0.2707, swd: 0.2055, ept: 323.1931
      Epoch 9 composite train-obj: 0.016976
            No improvement (0.2303), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.0791, mae: 0.1052, huber: 0.0232, swd: 0.0241, ept: 333.5260
    Epoch [10/50], Val Losses: mse: 0.0922, mae: 0.1497, huber: 0.0408, swd: 0.0401, ept: 335.4111
    Epoch [10/50], Test Losses: mse: 0.1069, mae: 0.1693, huber: 0.0482, swd: 0.0525, ept: 335.5603
      Epoch 10 composite train-obj: 0.023182
            No improvement (0.0408), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.0302, mae: 0.0911, huber: 0.0133, swd: 0.0112, ept: 335.3144
    Epoch [11/50], Val Losses: mse: 0.0611, mae: 0.1156, huber: 0.0239, swd: 0.0312, ept: 330.9278
    Epoch [11/50], Test Losses: mse: 0.0635, mae: 0.1257, huber: 0.0261, swd: 0.0346, ept: 332.2709
      Epoch 11 composite train-obj: 0.013311
            Val objective improved 0.0350 → 0.0239, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0197, mae: 0.0703, huber: 0.0085, swd: 0.0082, ept: 335.4933
    Epoch [12/50], Val Losses: mse: 0.0184, mae: 0.0790, huber: 0.0084, swd: 0.0081, ept: 335.9688
    Epoch [12/50], Test Losses: mse: 0.0189, mae: 0.0837, huber: 0.0088, swd: 0.0083, ept: 335.9712
      Epoch 12 composite train-obj: 0.008503
            Val objective improved 0.0239 → 0.0084, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.0168, mae: 0.0724, huber: 0.0077, swd: 0.0067, ept: 335.8858
    Epoch [13/50], Val Losses: mse: 0.0408, mae: 0.1144, huber: 0.0196, swd: 0.0162, ept: 335.9816
    Epoch [13/50], Test Losses: mse: 0.0460, mae: 0.1239, huber: 0.0222, swd: 0.0207, ept: 335.9691
      Epoch 13 composite train-obj: 0.007739
            No improvement (0.0196), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.0235, mae: 0.0793, huber: 0.0103, swd: 0.0093, ept: 335.5076
    Epoch [14/50], Val Losses: mse: 0.8241, mae: 0.2673, huber: 0.1266, swd: 0.1120, ept: 323.6329
    Epoch [14/50], Test Losses: mse: 0.7515, mae: 0.2798, huber: 0.1299, swd: 0.1100, ept: 321.5496
      Epoch 14 composite train-obj: 0.010281
            No improvement (0.1266), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.1741, mae: 0.1163, huber: 0.0315, swd: 0.0836, ept: 331.5504
    Epoch [15/50], Val Losses: mse: 0.0673, mae: 0.1504, huber: 0.0283, swd: 0.0284, ept: 334.2438
    Epoch [15/50], Test Losses: mse: 0.0669, mae: 0.1578, huber: 0.0291, swd: 0.0284, ept: 334.6019
      Epoch 15 composite train-obj: 0.031548
            No improvement (0.0283), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.0183, mae: 0.0763, huber: 0.0086, swd: 0.0079, ept: 335.8353
    Epoch [16/50], Val Losses: mse: 0.0419, mae: 0.1217, huber: 0.0201, swd: 0.0179, ept: 335.9527
    Epoch [16/50], Test Losses: mse: 0.0441, mae: 0.1303, huber: 0.0214, swd: 0.0195, ept: 335.9470
      Epoch 16 composite train-obj: 0.008577
            No improvement (0.0201), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.0306, mae: 0.0931, huber: 0.0139, swd: 0.0134, ept: 335.3703
    Epoch [17/50], Val Losses: mse: 0.0332, mae: 0.1100, huber: 0.0160, swd: 0.0136, ept: 335.8773
    Epoch [17/50], Test Losses: mse: 0.0358, mae: 0.1185, huber: 0.0174, swd: 0.0144, ept: 335.8780
      Epoch 17 composite train-obj: 0.013938
    Epoch [17/50], Test Losses: mse: 0.0189, mae: 0.0837, huber: 0.0088, swd: 0.0083, ept: 335.9712
    Best round's Test MSE: 0.0189, MAE: 0.0837, SWD: 0.0083
    Best round's Validation MSE: 0.0184, MAE: 0.0790, SWD: 0.0081
    Best round's Test verification MSE : 0.0189, MAE: 0.0837, SWD: 0.0083
    Time taken: 135.32 seconds
    
    ==================================================
    Experiment Summary (ACL_rossler_seq96_pred336_20250511_0358)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0349 ± 0.0231
      mae: 0.0953 ± 0.0199
      huber: 0.0134 ± 0.0065
      swd: 0.0170 ± 0.0138
      ept: 334.5515 ± 1.7118
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0357 ± 0.0248
      mae: 0.0900 ± 0.0178
      huber: 0.0129 ± 0.0063
      swd: 0.0182 ± 0.0159
      ept: 334.2375 ± 2.1586
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 341.72 seconds
    
    Experiment complete: ACL_rossler_seq96_pred336_20250511_0358
    Model: ACL
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    


```python

```


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['rossler']['channels'],# data_mgr.channels,              # ← number of features in your data
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
cfg.x_to_z_delay.enable_magnitudes_scale_shift = [False, True]
cfg.x_to_z_deri.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_x_main.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_y_main.enable_magnitudes_scale_shift = [False, True]
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 4.0875, mae: 0.7933, huber: 0.5457, swd: 3.3423, target_std: 3.9412
    Epoch [1/50], Val Losses: mse: 4.5950, mae: 0.8581, huber: 0.5837, swd: 2.9817, target_std: 3.9670
    Epoch [1/50], Test Losses: mse: 4.3068, mae: 0.9045, huber: 0.6180, swd: 2.7458, target_std: 4.1272
      Epoch 1 composite train-obj: 0.545742
            Val objective improved inf → 0.5837, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.2038, mae: 0.4441, huber: 0.2459, swd: 1.8664, target_std: 3.9307
    Epoch [2/50], Val Losses: mse: 2.0780, mae: 0.3914, huber: 0.2084, swd: 1.8742, target_std: 3.9670
    Epoch [2/50], Test Losses: mse: 1.9248, mae: 0.4195, huber: 0.2254, swd: 1.7122, target_std: 4.1272
      Epoch 2 composite train-obj: 0.245925
            Val objective improved 0.5837 → 0.2084, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.1449, mae: 0.2944, huber: 0.1382, swd: 0.9448, target_std: 3.9418
    Epoch [3/50], Val Losses: mse: 0.7866, mae: 0.3090, huber: 0.1322, swd: 0.5462, target_std: 3.9670
    Epoch [3/50], Test Losses: mse: 0.7053, mae: 0.3205, huber: 0.1339, swd: 0.4778, target_std: 4.1272
      Epoch 3 composite train-obj: 0.138199
            Val objective improved 0.2084 → 0.1322, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5460, mae: 0.2348, huber: 0.0919, swd: 0.3853, target_std: 3.9412
    Epoch [4/50], Val Losses: mse: 0.3881, mae: 0.3167, huber: 0.1172, swd: 0.2148, target_std: 3.9670
    Epoch [4/50], Test Losses: mse: 0.4008, mae: 0.3456, huber: 0.1317, swd: 0.2238, target_std: 4.1272
      Epoch 4 composite train-obj: 0.091905
            Val objective improved 0.1322 → 0.1172, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.2111, mae: 0.1885, huber: 0.0581, swd: 0.1119, target_std: 3.9400
    Epoch [5/50], Val Losses: mse: 0.0908, mae: 0.1560, huber: 0.0354, swd: 0.0354, target_std: 3.9670
    Epoch [5/50], Test Losses: mse: 0.0918, mae: 0.1664, huber: 0.0374, swd: 0.0380, target_std: 4.1272
      Epoch 5 composite train-obj: 0.058105
            Val objective improved 0.1172 → 0.0354, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.1151, mae: 0.1516, huber: 0.0383, swd: 0.0572, target_std: 3.9349
    Epoch [6/50], Val Losses: mse: 0.8628, mae: 0.3562, huber: 0.1879, swd: 0.4745, target_std: 3.9670
    Epoch [6/50], Test Losses: mse: 0.8521, mae: 0.3847, huber: 0.2033, swd: 0.4929, target_std: 4.1272
      Epoch 6 composite train-obj: 0.038321
            No improvement (0.1879), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.1212, mae: 0.1540, huber: 0.0398, swd: 0.0584, target_std: 3.9435
    Epoch [7/50], Val Losses: mse: 2.1135, mae: 0.4702, huber: 0.2688, swd: 1.6239, target_std: 3.9670
    Epoch [7/50], Test Losses: mse: 2.1131, mae: 0.5297, huber: 0.3138, swd: 1.5734, target_std: 4.1272
      Epoch 7 composite train-obj: 0.039757
            No improvement (0.2688), counter 2/5
    Epoch [8/50], Train Losses: mse: 1.0846, mae: 0.2649, huber: 0.1277, swd: 0.8778, target_std: 3.9451
    Epoch [8/50], Val Losses: mse: 2.1175, mae: 0.6117, huber: 0.3719, swd: 0.7158, target_std: 3.9670
    Epoch [8/50], Test Losses: mse: 2.5910, mae: 0.6916, huber: 0.4357, swd: 1.0175, target_std: 4.1272
      Epoch 8 composite train-obj: 0.127669
            No improvement (0.3719), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3832, mae: 0.1918, huber: 0.0701, swd: 0.2098, target_std: 3.9414
    Epoch [9/50], Val Losses: mse: 1.4629, mae: 0.4644, huber: 0.2348, swd: 0.9836, target_std: 3.9670
    Epoch [9/50], Test Losses: mse: 1.4343, mae: 0.5028, huber: 0.2610, swd: 0.9708, target_std: 4.1272
      Epoch 9 composite train-obj: 0.070100
            No improvement (0.2348), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.2872, mae: 0.1636, huber: 0.0513, swd: 0.2092, target_std: 3.9469
    Epoch [10/50], Val Losses: mse: 0.1962, mae: 0.1589, huber: 0.0547, swd: 0.1109, target_std: 3.9670
    Epoch [10/50], Test Losses: mse: 0.1990, mae: 0.1736, huber: 0.0594, swd: 0.1180, target_std: 4.1272
      Epoch 10 composite train-obj: 0.051338
    Epoch [10/50], Test Losses: mse: 0.0918, mae: 0.1663, huber: 0.0374, swd: 0.0380, target_std: 4.1272
    Best round's Test MSE: 0.0918, MAE: 0.1664, SWD: 0.0380
    Best round's Validation MSE: 0.0908, MAE: 0.1560
    Best round's Test verification MSE : 0.0918, MAE: 0.1663, SWD: 0.0380
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.1528, mae: 0.8168, huber: 0.5656, swd: 3.4177, target_std: 3.9393
    Epoch [1/50], Val Losses: mse: 2.8909, mae: 0.5561, huber: 0.3265, swd: 2.5341, target_std: 3.9670
    Epoch [1/50], Test Losses: mse: 2.7099, mae: 0.5882, huber: 0.3504, swd: 2.3245, target_std: 4.1272
      Epoch 1 composite train-obj: 0.565629
            Val objective improved inf → 0.3265, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.0279, mae: 0.3807, huber: 0.2036, swd: 1.8345, target_std: 3.9339
    Epoch [2/50], Val Losses: mse: 2.1424, mae: 0.4482, huber: 0.2373, swd: 1.9053, target_std: 3.9670
    Epoch [2/50], Test Losses: mse: 1.9968, mae: 0.4842, huber: 0.2589, swd: 1.7397, target_std: 4.1272
      Epoch 2 composite train-obj: 0.203619
            Val objective improved 0.3265 → 0.2373, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.1867, mae: 0.3050, huber: 0.1434, swd: 0.9933, target_std: 3.9357
    Epoch [3/50], Val Losses: mse: 2.0891, mae: 0.4397, huber: 0.2366, swd: 1.9319, target_std: 3.9670
    Epoch [3/50], Test Losses: mse: 1.9799, mae: 0.4773, huber: 0.2605, swd: 1.8104, target_std: 4.1272
      Epoch 3 composite train-obj: 0.143382
            Val objective improved 0.2373 → 0.2366, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5575, mae: 0.2416, huber: 0.0942, swd: 0.4087, target_std: 3.9439
    Epoch [4/50], Val Losses: mse: 0.9716, mae: 0.3336, huber: 0.1508, swd: 0.5818, target_std: 3.9670
    Epoch [4/50], Test Losses: mse: 0.9255, mae: 0.3534, huber: 0.1604, swd: 0.5723, target_std: 4.1272
      Epoch 4 composite train-obj: 0.094212
            Val objective improved 0.2366 → 0.1508, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.1897, mae: 0.1840, huber: 0.0547, swd: 0.0946, target_std: 3.9437
    Epoch [5/50], Val Losses: mse: 4.9266, mae: 0.6964, huber: 0.4686, swd: 3.1611, target_std: 3.9670
    Epoch [5/50], Test Losses: mse: 5.9642, mae: 0.7766, huber: 0.5374, swd: 3.9407, target_std: 4.1272
      Epoch 5 composite train-obj: 0.054707
            No improvement (0.4686), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.6509, mae: 0.2198, huber: 0.0890, swd: 0.4915, target_std: 3.9420
    Epoch [6/50], Val Losses: mse: 0.0930, mae: 0.1699, huber: 0.0381, swd: 0.0347, target_std: 3.9670
    Epoch [6/50], Test Losses: mse: 0.0946, mae: 0.1824, huber: 0.0400, swd: 0.0370, target_std: 4.1272
      Epoch 6 composite train-obj: 0.089008
            Val objective improved 0.1508 → 0.0381, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0816, mae: 0.1420, huber: 0.0311, swd: 0.0348, target_std: 3.9393
    Epoch [7/50], Val Losses: mse: 0.1001, mae: 0.1433, huber: 0.0343, swd: 0.0521, target_std: 3.9670
    Epoch [7/50], Test Losses: mse: 0.0938, mae: 0.1499, huber: 0.0343, swd: 0.0501, target_std: 4.1272
      Epoch 7 composite train-obj: 0.031139
            Val objective improved 0.0381 → 0.0343, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0575, mae: 0.1204, huber: 0.0231, swd: 0.0259, target_std: 3.9355
    Epoch [8/50], Val Losses: mse: 0.1671, mae: 0.1668, huber: 0.0507, swd: 0.1031, target_std: 3.9670
    Epoch [8/50], Test Losses: mse: 0.1967, mae: 0.1883, huber: 0.0610, swd: 0.1308, target_std: 4.1272
      Epoch 8 composite train-obj: 0.023070
            No improvement (0.0507), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.0651, mae: 0.1275, huber: 0.0259, swd: 0.0296, target_std: 3.9414
    Epoch [9/50], Val Losses: mse: 0.2159, mae: 0.2400, huber: 0.0862, swd: 0.0668, target_std: 3.9670
    Epoch [9/50], Test Losses: mse: 0.2403, mae: 0.2591, huber: 0.0966, swd: 0.0905, target_std: 4.1272
      Epoch 9 composite train-obj: 0.025851
            No improvement (0.0862), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.0526, mae: 0.1176, huber: 0.0219, swd: 0.0235, target_std: 3.9361
    Epoch [10/50], Val Losses: mse: 0.2626, mae: 0.2327, huber: 0.0792, swd: 0.1590, target_std: 3.9670
    Epoch [10/50], Test Losses: mse: 0.2659, mae: 0.2511, huber: 0.0866, swd: 0.1654, target_std: 4.1272
      Epoch 10 composite train-obj: 0.021862
            No improvement (0.0792), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.0816, mae: 0.1231, huber: 0.0258, swd: 0.0425, target_std: 3.9423
    Epoch [11/50], Val Losses: mse: 0.0618, mae: 0.1464, huber: 0.0277, swd: 0.0332, target_std: 3.9670
    Epoch [11/50], Test Losses: mse: 0.0694, mae: 0.1586, huber: 0.0316, swd: 0.0385, target_std: 4.1272
      Epoch 11 composite train-obj: 0.025810
            Val objective improved 0.0343 → 0.0277, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0393, mae: 0.1008, huber: 0.0167, swd: 0.0184, target_std: 3.9368
    Epoch [12/50], Val Losses: mse: 0.2751, mae: 0.2366, huber: 0.0890, swd: 0.0991, target_std: 3.9670
    Epoch [12/50], Test Losses: mse: 0.3483, mae: 0.2694, huber: 0.1100, swd: 0.1527, target_std: 4.1272
      Epoch 12 composite train-obj: 0.016749
            No improvement (0.0890), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0393, mae: 0.1002, huber: 0.0167, swd: 0.0188, target_std: 3.9307
    Epoch [13/50], Val Losses: mse: 0.0318, mae: 0.0887, huber: 0.0138, swd: 0.0162, target_std: 3.9670
    Epoch [13/50], Test Losses: mse: 0.0338, mae: 0.0945, huber: 0.0149, swd: 0.0195, target_std: 4.1272
      Epoch 13 composite train-obj: 0.016652
            Val objective improved 0.0277 → 0.0138, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0340, mae: 0.0936, huber: 0.0149, swd: 0.0161, target_std: 3.9364
    Epoch [14/50], Val Losses: mse: 0.1134, mae: 0.1682, huber: 0.0428, swd: 0.0805, target_std: 3.9670
    Epoch [14/50], Test Losses: mse: 0.1249, mae: 0.1855, huber: 0.0494, swd: 0.0919, target_std: 4.1272
      Epoch 14 composite train-obj: 0.014851
            No improvement (0.0428), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.0328, mae: 0.0901, huber: 0.0141, swd: 0.0159, target_std: 3.9462
    Epoch [15/50], Val Losses: mse: 1.9254, mae: 0.5674, huber: 0.3355, swd: 0.9006, target_std: 3.9670
    Epoch [15/50], Test Losses: mse: 1.9800, mae: 0.6292, huber: 0.3838, swd: 0.9314, target_std: 4.1272
      Epoch 15 composite train-obj: 0.014118
            No improvement (0.3355), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.7245, mae: 0.2307, huber: 0.1017, swd: 0.5240, target_std: 3.9412
    Epoch [16/50], Val Losses: mse: 0.7770, mae: 0.2011, huber: 0.0936, swd: 0.7202, target_std: 3.9670
    Epoch [16/50], Test Losses: mse: 0.7494, mae: 0.2166, huber: 0.1029, swd: 0.6892, target_std: 4.1272
      Epoch 16 composite train-obj: 0.101690
            No improvement (0.0936), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.2630, mae: 0.1452, huber: 0.0455, swd: 0.1905, target_std: 3.9439
    Epoch [17/50], Val Losses: mse: 0.0951, mae: 0.1896, huber: 0.0428, swd: 0.0563, target_std: 3.9670
    Epoch [17/50], Test Losses: mse: 0.0973, mae: 0.1988, huber: 0.0448, swd: 0.0563, target_std: 4.1272
      Epoch 17 composite train-obj: 0.045453
            No improvement (0.0428), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.0485, mae: 0.1103, huber: 0.0202, swd: 0.0214, target_std: 3.9357
    Epoch [18/50], Val Losses: mse: 0.0451, mae: 0.1241, huber: 0.0215, swd: 0.0240, target_std: 3.9670
    Epoch [18/50], Test Losses: mse: 0.0504, mae: 0.1337, huber: 0.0239, swd: 0.0276, target_std: 4.1272
      Epoch 18 composite train-obj: 0.020184
    Epoch [18/50], Test Losses: mse: 0.0338, mae: 0.0945, huber: 0.0149, swd: 0.0195, target_std: 4.1272
    Best round's Test MSE: 0.0338, MAE: 0.0945, SWD: 0.0195
    Best round's Validation MSE: 0.0318, MAE: 0.0887
    Best round's Test verification MSE : 0.0338, MAE: 0.0945, SWD: 0.0195
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.0858, mae: 0.8057, huber: 0.5561, swd: 3.3311, target_std: 3.9395
    Epoch [1/50], Val Losses: mse: 3.4638, mae: 0.7331, huber: 0.4674, swd: 2.6669, target_std: 3.9670
    Epoch [1/50], Test Losses: mse: 3.3695, mae: 0.7993, huber: 0.5212, swd: 2.4920, target_std: 4.1272
      Epoch 1 composite train-obj: 0.556093
            Val objective improved inf → 0.4674, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.2211, mae: 0.3887, huber: 0.2173, swd: 2.0748, target_std: 3.9371
    Epoch [2/50], Val Losses: mse: 2.3336, mae: 0.4180, huber: 0.2209, swd: 2.1811, target_std: 3.9670
    Epoch [2/50], Test Losses: mse: 2.1461, mae: 0.4417, huber: 0.2358, swd: 1.9770, target_std: 4.1272
      Epoch 2 composite train-obj: 0.217328
            Val objective improved 0.4674 → 0.2209, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.4037, mae: 0.3220, huber: 0.1589, swd: 1.1587, target_std: 3.9422
    Epoch [3/50], Val Losses: mse: 2.8883, mae: 0.8540, huber: 0.5476, swd: 0.9321, target_std: 3.9670
    Epoch [3/50], Test Losses: mse: 2.7962, mae: 0.8964, huber: 0.5794, swd: 0.9516, target_std: 4.1272
      Epoch 3 composite train-obj: 0.158882
            No improvement (0.5476), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5449, mae: 0.2817, huber: 0.1175, swd: 0.2728, target_std: 3.9482
    Epoch [4/50], Val Losses: mse: 2.5783, mae: 0.5789, huber: 0.3515, swd: 0.4407, target_std: 3.9670
    Epoch [4/50], Test Losses: mse: 2.5581, mae: 0.6310, huber: 0.3905, swd: 0.4669, target_std: 4.1272
      Epoch 4 composite train-obj: 0.117480
            No improvement (0.3515), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.7983, mae: 0.2454, huber: 0.1072, swd: 0.6264, target_std: 3.9344
    Epoch [5/50], Val Losses: mse: 0.2167, mae: 0.1854, huber: 0.0574, swd: 0.1289, target_std: 3.9670
    Epoch [5/50], Test Losses: mse: 0.2283, mae: 0.2007, huber: 0.0641, swd: 0.1443, target_std: 4.1272
      Epoch 5 composite train-obj: 0.107209
            Val objective improved 0.2209 → 0.0574, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.1016, mae: 0.1330, huber: 0.0322, swd: 0.0516, target_std: 3.9320
    Epoch [6/50], Val Losses: mse: 0.1073, mae: 0.1420, huber: 0.0352, swd: 0.0721, target_std: 3.9670
    Epoch [6/50], Test Losses: mse: 0.1245, mae: 0.1583, huber: 0.0426, swd: 0.0875, target_std: 4.1272
      Epoch 6 composite train-obj: 0.032235
            Val objective improved 0.0574 → 0.0352, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0713, mae: 0.1243, huber: 0.0265, swd: 0.0348, target_std: 3.9387
    Epoch [7/50], Val Losses: mse: 0.1005, mae: 0.1690, huber: 0.0437, swd: 0.0414, target_std: 3.9670
    Epoch [7/50], Test Losses: mse: 0.1195, mae: 0.1893, huber: 0.0522, swd: 0.0560, target_std: 4.1272
      Epoch 7 composite train-obj: 0.026502
            No improvement (0.0437), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0437, mae: 0.1036, huber: 0.0182, swd: 0.0209, target_std: 3.9417
    Epoch [8/50], Val Losses: mse: 1.1275, mae: 0.2635, huber: 0.1376, swd: 0.8913, target_std: 3.9670
    Epoch [8/50], Test Losses: mse: 1.1579, mae: 0.2957, huber: 0.1594, swd: 0.9562, target_std: 4.1272
      Epoch 8 composite train-obj: 0.018228
            No improvement (0.1376), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.4933, mae: 0.1785, huber: 0.0677, swd: 0.3811, target_std: 3.9425
    Epoch [9/50], Val Losses: mse: 0.1362, mae: 0.1970, huber: 0.0535, swd: 0.0546, target_std: 3.9670
    Epoch [9/50], Test Losses: mse: 0.1409, mae: 0.2118, huber: 0.0579, swd: 0.0533, target_std: 4.1272
      Epoch 9 composite train-obj: 0.067668
            No improvement (0.0535), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.0421, mae: 0.1002, huber: 0.0170, swd: 0.0204, target_std: 3.9420
    Epoch [10/50], Val Losses: mse: 0.0721, mae: 0.1122, huber: 0.0255, swd: 0.0334, target_std: 3.9670
    Epoch [10/50], Test Losses: mse: 0.0803, mae: 0.1204, huber: 0.0281, swd: 0.0406, target_std: 4.1272
      Epoch 10 composite train-obj: 0.017010
            Val objective improved 0.0352 → 0.0255, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0384, mae: 0.0957, huber: 0.0157, swd: 0.0182, target_std: 3.9338
    Epoch [11/50], Val Losses: mse: 0.0296, mae: 0.0956, huber: 0.0144, swd: 0.0102, target_std: 3.9670
    Epoch [11/50], Test Losses: mse: 0.0315, mae: 0.1030, huber: 0.0153, swd: 0.0115, target_std: 4.1272
      Epoch 11 composite train-obj: 0.015714
            Val objective improved 0.0255 → 0.0144, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0236, mae: 0.0763, huber: 0.0103, swd: 0.0114, target_std: 3.9327
    Epoch [12/50], Val Losses: mse: 0.0253, mae: 0.0972, huber: 0.0123, swd: 0.0108, target_std: 3.9670
    Epoch [12/50], Test Losses: mse: 0.0272, mae: 0.1044, huber: 0.0133, swd: 0.0119, target_std: 4.1272
      Epoch 12 composite train-obj: 0.010304
            Val objective improved 0.0144 → 0.0123, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.0249, mae: 0.0825, huber: 0.0112, swd: 0.0122, target_std: 3.9333
    Epoch [13/50], Val Losses: mse: 0.1520, mae: 0.1336, huber: 0.0418, swd: 0.0370, target_std: 3.9670
    Epoch [13/50], Test Losses: mse: 0.1470, mae: 0.1408, huber: 0.0431, swd: 0.0421, target_std: 4.1272
      Epoch 13 composite train-obj: 0.011177
            No improvement (0.0418), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.0309, mae: 0.0903, huber: 0.0134, swd: 0.0141, target_std: 3.9424
    Epoch [14/50], Val Losses: mse: 0.0359, mae: 0.0993, huber: 0.0166, swd: 0.0118, target_std: 3.9670
    Epoch [14/50], Test Losses: mse: 0.0379, mae: 0.1054, huber: 0.0177, swd: 0.0133, target_std: 4.1272
      Epoch 14 composite train-obj: 0.013414
            No improvement (0.0166), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.0258, mae: 0.0861, huber: 0.0118, swd: 0.0123, target_std: 3.9445
    Epoch [15/50], Val Losses: mse: 0.2110, mae: 0.2255, huber: 0.0666, swd: 0.1162, target_std: 3.9670
    Epoch [15/50], Test Losses: mse: 0.1987, mae: 0.2364, huber: 0.0686, swd: 0.1161, target_std: 4.1272
      Epoch 15 composite train-obj: 0.011824
            No improvement (0.0666), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.0399, mae: 0.0906, huber: 0.0147, swd: 0.0220, target_std: 3.9404
    Epoch [16/50], Val Losses: mse: 0.0158, mae: 0.0783, huber: 0.0079, swd: 0.0063, target_std: 3.9670
    Epoch [16/50], Test Losses: mse: 0.0166, mae: 0.0833, huber: 0.0082, swd: 0.0068, target_std: 4.1272
      Epoch 16 composite train-obj: 0.014740
            Val objective improved 0.0123 → 0.0079, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 0.0255, mae: 0.0801, huber: 0.0112, swd: 0.0127, target_std: 3.9361
    Epoch [17/50], Val Losses: mse: 0.0646, mae: 0.1331, huber: 0.0288, swd: 0.0264, target_std: 3.9670
    Epoch [17/50], Test Losses: mse: 0.0662, mae: 0.1434, huber: 0.0308, swd: 0.0295, target_std: 4.1272
      Epoch 17 composite train-obj: 0.011204
            No improvement (0.0288), counter 1/5
    Epoch [18/50], Train Losses: mse: 0.0203, mae: 0.0760, huber: 0.0095, swd: 0.0100, target_std: 3.9339
    Epoch [18/50], Val Losses: mse: 0.0946, mae: 0.1834, huber: 0.0458, swd: 0.0285, target_std: 3.9670
    Epoch [18/50], Test Losses: mse: 0.1022, mae: 0.1975, huber: 0.0498, swd: 0.0320, target_std: 4.1272
      Epoch 18 composite train-obj: 0.009503
            No improvement (0.0458), counter 2/5
    Epoch [19/50], Train Losses: mse: 0.0231, mae: 0.0792, huber: 0.0110, swd: 0.0108, target_std: 3.9389
    Epoch [19/50], Val Losses: mse: 0.0617, mae: 0.1333, huber: 0.0275, swd: 0.0417, target_std: 3.9670
    Epoch [19/50], Test Losses: mse: 0.0691, mae: 0.1457, huber: 0.0313, swd: 0.0485, target_std: 4.1272
      Epoch 19 composite train-obj: 0.010951
            No improvement (0.0275), counter 3/5
    Epoch [20/50], Train Losses: mse: 0.0246, mae: 0.0791, huber: 0.0110, swd: 0.0127, target_std: 3.9429
    Epoch [20/50], Val Losses: mse: 0.0946, mae: 0.1998, huber: 0.0466, swd: 0.0408, target_std: 3.9670
    Epoch [20/50], Test Losses: mse: 0.1014, mae: 0.2141, huber: 0.0504, swd: 0.0436, target_std: 4.1272
      Epoch 20 composite train-obj: 0.010981
            No improvement (0.0466), counter 4/5
    Epoch [21/50], Train Losses: mse: 0.0269, mae: 0.0858, huber: 0.0125, swd: 0.0127, target_std: 3.9336
    Epoch [21/50], Val Losses: mse: 0.0616, mae: 0.0968, huber: 0.0221, swd: 0.0186, target_std: 3.9670
    Epoch [21/50], Test Losses: mse: 0.0597, mae: 0.1041, huber: 0.0230, swd: 0.0198, target_std: 4.1272
      Epoch 21 composite train-obj: 0.012494
    Epoch [21/50], Test Losses: mse: 0.0166, mae: 0.0833, huber: 0.0082, swd: 0.0068, target_std: 4.1272
    Best round's Test MSE: 0.0166, MAE: 0.0833, SWD: 0.0068
    Best round's Validation MSE: 0.0158, MAE: 0.0783
    Best round's Test verification MSE : 0.0166, MAE: 0.0833, SWD: 0.0068
    
    ==================================================
    Experiment Summary (ACL_rossler_seq96_pred336_20250430_1329)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0474 ± 0.0322
      mae: 0.1147 ± 0.0368
      huber: 0.0202 ± 0.0125
      swd: 0.0214 ± 0.0128
      target_std: 4.1272 ± 0.0000
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0461 ± 0.0322
      mae: 0.1077 ± 0.0344
      huber: 0.0190 ± 0.0118
      swd: 0.0193 ± 0.0121
      target_std: 3.9670 ± 0.0000
      count: 38.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_rossler_seq96_pred336_20250430_1329
    Model: ACL
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['rossler']['channels'],# data_mgr.channels,              # ← number of features in your data
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
cfg.x_to_z_delay.enable_magnitudes_scale_shift = [False, True]
cfg.x_to_z_deri.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_x_main.enable_magnitudes_scale_shift = [False, True]
cfg.z_to_y_main.enable_magnitudes_scale_shift = [False, True]
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    Train set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 279
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 4.9768, mae: 1.0178, huber: 0.7347, swd: 3.5616, target_std: 3.9644
    Epoch [1/50], Val Losses: mse: 5.9631, mae: 1.1195, huber: 0.8222, swd: 3.3931, target_std: 4.0274
    Epoch [1/50], Test Losses: mse: 5.2926, mae: 1.1280, huber: 0.8229, swd: 2.9222, target_std: 4.1166
      Epoch 1 composite train-obj: 0.734731
            Val objective improved inf → 0.8222, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.6754, mae: 0.5430, huber: 0.3178, swd: 2.0920, target_std: 3.9577
    Epoch [2/50], Val Losses: mse: 2.7539, mae: 0.4922, huber: 0.2795, swd: 2.2392, target_std: 4.0274
    Epoch [2/50], Test Losses: mse: 2.2783, mae: 0.5009, huber: 0.2761, swd: 1.7950, target_std: 4.1166
      Epoch 2 composite train-obj: 0.317754
            Val objective improved 0.8222 → 0.2795, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.9133, mae: 0.3982, huber: 0.2072, swd: 1.5269, target_std: 3.9638
    Epoch [3/50], Val Losses: mse: 3.8928, mae: 0.9490, huber: 0.6464, swd: 1.7884, target_std: 4.0274
    Epoch [3/50], Test Losses: mse: 3.4403, mae: 0.9647, huber: 0.6512, swd: 1.4628, target_std: 4.1166
      Epoch 3 composite train-obj: 0.207198
            No improvement (0.6464), counter 1/5
    Epoch [4/50], Train Losses: mse: 1.3880, mae: 0.3908, huber: 0.1925, swd: 0.9698, target_std: 3.9646
    Epoch [4/50], Val Losses: mse: 1.2872, mae: 0.4902, huber: 0.2380, swd: 0.7587, target_std: 4.0274
    Epoch [4/50], Test Losses: mse: 1.0727, mae: 0.4983, huber: 0.2378, swd: 0.5984, target_std: 4.1166
      Epoch 4 composite train-obj: 0.192544
            Val objective improved 0.2795 → 0.2380, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5212, mae: 0.2569, huber: 0.0973, swd: 0.3033, target_std: 3.9610
    Epoch [5/50], Val Losses: mse: 2.0578, mae: 0.7301, huber: 0.4563, swd: 0.7760, target_std: 4.0274
    Epoch [5/50], Test Losses: mse: 2.2231, mae: 0.8094, huber: 0.5212, swd: 0.8321, target_std: 4.1166
      Epoch 5 composite train-obj: 0.097288
            No improvement (0.4563), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4300, mae: 0.3000, huber: 0.1172, swd: 0.1787, target_std: 3.9656
    Epoch [6/50], Val Losses: mse: 0.6050, mae: 0.4148, huber: 0.1838, swd: 0.1530, target_std: 4.0274
    Epoch [6/50], Test Losses: mse: 0.5603, mae: 0.4179, huber: 0.1790, swd: 0.1336, target_std: 4.1166
      Epoch 6 composite train-obj: 0.117228
            Val objective improved 0.2380 → 0.1838, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.1789, mae: 0.1895, huber: 0.0539, swd: 0.0718, target_std: 3.9655
    Epoch [7/50], Val Losses: mse: 0.2252, mae: 0.2465, huber: 0.0766, swd: 0.0937, target_std: 4.0274
    Epoch [7/50], Test Losses: mse: 0.2745, mae: 0.2677, huber: 0.0890, swd: 0.1184, target_std: 4.1166
      Epoch 7 composite train-obj: 0.053879
            Val objective improved 0.1838 → 0.0766, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.1116, mae: 0.1619, huber: 0.0386, swd: 0.0460, target_std: 3.9658
    Epoch [8/50], Val Losses: mse: 0.3954, mae: 0.2627, huber: 0.0990, swd: 0.2155, target_std: 4.0274
    Epoch [8/50], Test Losses: mse: 0.3619, mae: 0.2750, huber: 0.1017, swd: 0.1917, target_std: 4.1166
      Epoch 8 composite train-obj: 0.038641
            No improvement (0.0990), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.1074, mae: 0.1553, huber: 0.0358, swd: 0.0467, target_std: 3.9640
    Epoch [9/50], Val Losses: mse: 0.1821, mae: 0.2047, huber: 0.0616, swd: 0.0667, target_std: 4.0274
    Epoch [9/50], Test Losses: mse: 0.1758, mae: 0.2196, huber: 0.0647, swd: 0.0709, target_std: 4.1166
      Epoch 9 composite train-obj: 0.035803
            Val objective improved 0.0766 → 0.0616, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.0666, mae: 0.1332, huber: 0.0259, swd: 0.0274, target_std: 3.9636
    Epoch [10/50], Val Losses: mse: 0.1001, mae: 0.1918, huber: 0.0449, swd: 0.0379, target_std: 4.0274
    Epoch [10/50], Test Losses: mse: 0.1117, mae: 0.2064, huber: 0.0501, swd: 0.0505, target_std: 4.1166
      Epoch 10 composite train-obj: 0.025883
            Val objective improved 0.0616 → 0.0449, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0635, mae: 0.1288, huber: 0.0246, swd: 0.0278, target_std: 3.9607
    Epoch [11/50], Val Losses: mse: 0.1162, mae: 0.1899, huber: 0.0514, swd: 0.0364, target_std: 4.0274
    Epoch [11/50], Test Losses: mse: 0.1394, mae: 0.2133, huber: 0.0615, swd: 0.0517, target_std: 4.1166
      Epoch 11 composite train-obj: 0.024635
            No improvement (0.0514), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.0596, mae: 0.1273, huber: 0.0240, swd: 0.0249, target_std: 3.9651
    Epoch [12/50], Val Losses: mse: 0.4659, mae: 0.2916, huber: 0.1214, swd: 0.1073, target_std: 4.0274
    Epoch [12/50], Test Losses: mse: 0.5022, mae: 0.3272, huber: 0.1433, swd: 0.1240, target_std: 4.1166
      Epoch 12 composite train-obj: 0.024047
            No improvement (0.1214), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.1055, mae: 0.1395, huber: 0.0309, swd: 0.0372, target_std: 3.9569
    Epoch [13/50], Val Losses: mse: 0.1241, mae: 0.1871, huber: 0.0466, swd: 0.0587, target_std: 4.0274
    Epoch [13/50], Test Losses: mse: 0.1483, mae: 0.2032, huber: 0.0544, swd: 0.0630, target_std: 4.1166
      Epoch 13 composite train-obj: 0.030942
            No improvement (0.0466), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.0498, mae: 0.1155, huber: 0.0201, swd: 0.0221, target_std: 3.9579
    Epoch [14/50], Val Losses: mse: 0.1312, mae: 0.1713, huber: 0.0468, swd: 0.0589, target_std: 4.0274
    Epoch [14/50], Test Losses: mse: 0.1637, mae: 0.1884, huber: 0.0537, swd: 0.0840, target_std: 4.1166
      Epoch 14 composite train-obj: 0.020143
            No improvement (0.0468), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.0501, mae: 0.1209, huber: 0.0208, swd: 0.0215, target_std: 3.9637
    Epoch [15/50], Val Losses: mse: 0.3883, mae: 0.2487, huber: 0.0965, swd: 0.1647, target_std: 4.0274
    Epoch [15/50], Test Losses: mse: 0.4623, mae: 0.2663, huber: 0.1086, swd: 0.2169, target_std: 4.1166
      Epoch 15 composite train-obj: 0.020807
    Epoch [15/50], Test Losses: mse: 0.1117, mae: 0.2064, huber: 0.0501, swd: 0.0505, target_std: 4.1166
    Best round's Test MSE: 0.1117, MAE: 0.2064, SWD: 0.0505
    Best round's Validation MSE: 0.1001, MAE: 0.1918
    Best round's Test verification MSE : 0.1117, MAE: 0.2064, SWD: 0.0505
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.8749, mae: 0.9888, huber: 0.7118, swd: 3.4435, target_std: 3.9628
    Epoch [1/50], Val Losses: mse: 4.5753, mae: 0.9162, huber: 0.6397, swd: 3.0105, target_std: 4.0274
    Epoch [1/50], Test Losses: mse: 4.0182, mae: 0.9384, huber: 0.6503, swd: 2.5953, target_std: 4.1166
      Epoch 1 composite train-obj: 0.711777
            Val objective improved inf → 0.6397, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.5509, mae: 0.4847, huber: 0.2786, swd: 2.1232, target_std: 3.9634
    Epoch [2/50], Val Losses: mse: 3.1795, mae: 0.6522, huber: 0.3978, swd: 2.2887, target_std: 4.0274
    Epoch [2/50], Test Losses: mse: 2.7160, mae: 0.6622, huber: 0.3981, swd: 1.8629, target_std: 4.1166
      Epoch 2 composite train-obj: 0.278598
            Val objective improved 0.6397 → 0.3978, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.9581, mae: 0.3986, huber: 0.2106, swd: 1.5944, target_std: 3.9625
    Epoch [3/50], Val Losses: mse: 3.1242, mae: 0.7427, huber: 0.4800, swd: 1.7730, target_std: 4.0274
    Epoch [3/50], Test Losses: mse: 2.6263, mae: 0.7436, huber: 0.4687, swd: 1.4414, target_std: 4.1166
      Epoch 3 composite train-obj: 0.210627
            No improvement (0.4800), counter 1/5
    Epoch [4/50], Train Losses: mse: 1.5347, mae: 0.3701, huber: 0.1846, swd: 1.1952, target_std: 3.9607
    Epoch [4/50], Val Losses: mse: 1.6570, mae: 0.3978, huber: 0.1941, swd: 1.3278, target_std: 4.0274
    Epoch [4/50], Test Losses: mse: 1.3265, mae: 0.3998, huber: 0.1881, swd: 1.0264, target_std: 4.1166
      Epoch 4 composite train-obj: 0.184641
            Val objective improved 0.3978 → 0.1941, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 1.1153, mae: 0.2947, huber: 0.1355, swd: 0.8714, target_std: 3.9600
    Epoch [5/50], Val Losses: mse: 3.3906, mae: 0.6909, huber: 0.4542, swd: 1.4659, target_std: 4.0274
    Epoch [5/50], Test Losses: mse: 3.1823, mae: 0.6906, huber: 0.4506, swd: 1.4182, target_std: 4.1166
      Epoch 5 composite train-obj: 0.135456
            No improvement (0.4542), counter 1/5
    Epoch [6/50], Train Losses: mse: 1.1808, mae: 0.2985, huber: 0.1411, swd: 0.9374, target_std: 3.9635
    Epoch [6/50], Val Losses: mse: 1.0072, mae: 0.3948, huber: 0.1798, swd: 0.6178, target_std: 4.0274
    Epoch [6/50], Test Losses: mse: 0.7990, mae: 0.3993, huber: 0.1750, swd: 0.4425, target_std: 4.1166
      Epoch 6 composite train-obj: 0.141125
            Val objective improved 0.1941 → 0.1798, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.6333, mae: 0.2485, huber: 0.0977, swd: 0.4563, target_std: 3.9582
    Epoch [7/50], Val Losses: mse: 1.2143, mae: 0.3053, huber: 0.1438, swd: 0.9980, target_std: 4.0274
    Epoch [7/50], Test Losses: mse: 0.9687, mae: 0.3160, huber: 0.1447, swd: 0.7465, target_std: 4.1166
      Epoch 7 composite train-obj: 0.097722
            Val objective improved 0.1798 → 0.1438, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4316, mae: 0.2194, huber: 0.0765, swd: 0.2934, target_std: 3.9560
    Epoch [8/50], Val Losses: mse: 0.7530, mae: 0.2878, huber: 0.1143, swd: 0.5686, target_std: 4.0274
    Epoch [8/50], Test Losses: mse: 0.6685, mae: 0.3004, huber: 0.1183, swd: 0.4883, target_std: 4.1166
      Epoch 8 composite train-obj: 0.076469
            Val objective improved 0.1438 → 0.1143, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.3108, mae: 0.2023, huber: 0.0647, swd: 0.1847, target_std: 3.9588
    Epoch [9/50], Val Losses: mse: 0.5526, mae: 0.3612, huber: 0.1671, swd: 0.2231, target_std: 4.0274
    Epoch [9/50], Test Losses: mse: 0.6471, mae: 0.4136, huber: 0.2037, swd: 0.2599, target_std: 4.1166
      Epoch 9 composite train-obj: 0.064742
            No improvement (0.1671), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.1721, mae: 0.1776, huber: 0.0498, swd: 0.0814, target_std: 3.9622
    Epoch [10/50], Val Losses: mse: 0.0900, mae: 0.1459, huber: 0.0324, swd: 0.0325, target_std: 4.0274
    Epoch [10/50], Test Losses: mse: 0.0949, mae: 0.1559, huber: 0.0352, swd: 0.0366, target_std: 4.1166
      Epoch 10 composite train-obj: 0.049809
            Val objective improved 0.1143 → 0.0324, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0976, mae: 0.1428, huber: 0.0327, swd: 0.0425, target_std: 3.9638
    Epoch [11/50], Val Losses: mse: 0.2914, mae: 0.2205, huber: 0.0781, swd: 0.1737, target_std: 4.0274
    Epoch [11/50], Test Losses: mse: 0.2785, mae: 0.2393, huber: 0.0857, swd: 0.1556, target_std: 4.1166
      Epoch 11 composite train-obj: 0.032698
            No improvement (0.0781), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.0704, mae: 0.1295, huber: 0.0255, swd: 0.0302, target_std: 3.9646
    Epoch [12/50], Val Losses: mse: 0.1590, mae: 0.2116, huber: 0.0626, swd: 0.0358, target_std: 4.0274
    Epoch [12/50], Test Losses: mse: 0.1563, mae: 0.2201, huber: 0.0641, swd: 0.0371, target_std: 4.1166
      Epoch 12 composite train-obj: 0.025529
            No improvement (0.0626), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.0731, mae: 0.1361, huber: 0.0274, swd: 0.0302, target_std: 3.9621
    Epoch [13/50], Val Losses: mse: 0.2924, mae: 0.1968, huber: 0.0642, swd: 0.1484, target_std: 4.0274
    Epoch [13/50], Test Losses: mse: 0.2401, mae: 0.1960, huber: 0.0614, swd: 0.1293, target_std: 4.1166
      Epoch 13 composite train-obj: 0.027371
            No improvement (0.0642), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.0660, mae: 0.1321, huber: 0.0259, swd: 0.0280, target_std: 3.9586
    Epoch [14/50], Val Losses: mse: 0.0504, mae: 0.1091, huber: 0.0204, swd: 0.0208, target_std: 4.0274
    Epoch [14/50], Test Losses: mse: 0.0608, mae: 0.1192, huber: 0.0243, swd: 0.0264, target_std: 4.1166
      Epoch 14 composite train-obj: 0.025887
            Val objective improved 0.0324 → 0.0204, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.0521, mae: 0.1124, huber: 0.0200, swd: 0.0236, target_std: 3.9575
    Epoch [15/50], Val Losses: mse: 0.0568, mae: 0.1370, huber: 0.0251, swd: 0.0172, target_std: 4.0274
    Epoch [15/50], Test Losses: mse: 0.0529, mae: 0.1383, huber: 0.0241, swd: 0.0200, target_std: 4.1166
      Epoch 15 composite train-obj: 0.019999
            No improvement (0.0251), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.0505, mae: 0.1163, huber: 0.0203, swd: 0.0220, target_std: 3.9583
    Epoch [16/50], Val Losses: mse: 0.0939, mae: 0.1537, huber: 0.0372, swd: 0.0292, target_std: 4.0274
    Epoch [16/50], Test Losses: mse: 0.0920, mae: 0.1598, huber: 0.0378, swd: 0.0313, target_std: 4.1166
      Epoch 16 composite train-obj: 0.020307
            No improvement (0.0372), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.0455, mae: 0.1110, huber: 0.0184, swd: 0.0202, target_std: 3.9633
    Epoch [17/50], Val Losses: mse: 0.9395, mae: 0.2911, huber: 0.1353, swd: 0.4951, target_std: 4.0274
    Epoch [17/50], Test Losses: mse: 0.9050, mae: 0.3122, huber: 0.1480, swd: 0.4242, target_std: 4.1166
      Epoch 17 composite train-obj: 0.018424
            No improvement (0.1353), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.2911, mae: 0.1626, huber: 0.0517, swd: 0.1730, target_std: 3.9609
    Epoch [18/50], Val Losses: mse: 1.9510, mae: 0.6087, huber: 0.3766, swd: 1.0124, target_std: 4.0274
    Epoch [18/50], Test Losses: mse: 2.7893, mae: 0.7236, huber: 0.4800, swd: 1.5616, target_std: 4.1166
      Epoch 18 composite train-obj: 0.051693
            No improvement (0.3766), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.1819, mae: 0.1549, huber: 0.0435, swd: 0.0944, target_std: 3.9649
    Epoch [19/50], Val Losses: mse: 0.0598, mae: 0.1503, huber: 0.0273, swd: 0.0246, target_std: 4.0274
    Epoch [19/50], Test Losses: mse: 0.0729, mae: 0.1641, huber: 0.0326, swd: 0.0284, target_std: 4.1166
      Epoch 19 composite train-obj: 0.043488
    Epoch [19/50], Test Losses: mse: 0.0608, mae: 0.1192, huber: 0.0243, swd: 0.0264, target_std: 4.1166
    Best round's Test MSE: 0.0608, MAE: 0.1192, SWD: 0.0264
    Best round's Validation MSE: 0.0504, MAE: 0.1091
    Best round's Test verification MSE : 0.0608, MAE: 0.1192, SWD: 0.0264
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.8166, mae: 0.9839, huber: 0.7052, swd: 3.4981, target_std: 3.9630
    Epoch [1/50], Val Losses: mse: 3.9716, mae: 0.7765, huber: 0.5134, swd: 3.1065, target_std: 4.0274
    Epoch [1/50], Test Losses: mse: 3.6941, mae: 0.8283, huber: 0.5515, swd: 2.6522, target_std: 4.1166
      Epoch 1 composite train-obj: 0.705231
            Val objective improved inf → 0.5134, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.5652, mae: 0.4878, huber: 0.2819, swd: 2.2119, target_std: 3.9552
    Epoch [2/50], Val Losses: mse: 2.9405, mae: 0.6271, huber: 0.3608, swd: 2.4913, target_std: 4.0274
    Epoch [2/50], Test Losses: mse: 2.5008, mae: 0.6394, huber: 0.3637, swd: 2.0689, target_std: 4.1166
      Epoch 2 composite train-obj: 0.281865
            Val objective improved 0.5134 → 0.3608, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.9432, mae: 0.4087, huber: 0.2149, swd: 1.6333, target_std: 3.9622
    Epoch [3/50], Val Losses: mse: 6.4594, mae: 1.2646, huber: 0.9363, swd: 2.9165, target_std: 4.0274
    Epoch [3/50], Test Losses: mse: 5.9641, mae: 1.3117, huber: 0.9747, swd: 2.5898, target_std: 4.1166
      Epoch 3 composite train-obj: 0.214878
            No improvement (0.9363), counter 1/5
    Epoch [4/50], Train Losses: mse: 1.8363, mae: 0.4808, huber: 0.2610, swd: 1.3341, target_std: 3.9655
    Epoch [4/50], Val Losses: mse: 1.6181, mae: 0.4295, huber: 0.2086, swd: 1.2649, target_std: 4.0274
    Epoch [4/50], Test Losses: mse: 1.2858, mae: 0.4289, huber: 0.2018, swd: 0.9596, target_std: 4.1166
      Epoch 4 composite train-obj: 0.260990
            Val objective improved 0.3608 → 0.2086, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 1.0143, mae: 0.2947, huber: 0.1307, swd: 0.8209, target_std: 3.9634
    Epoch [5/50], Val Losses: mse: 1.0973, mae: 0.4472, huber: 0.2280, swd: 0.5811, target_std: 4.0274
    Epoch [5/50], Test Losses: mse: 0.9233, mae: 0.4373, huber: 0.2134, swd: 0.4582, target_std: 4.1166
      Epoch 5 composite train-obj: 0.130665
            No improvement (0.2280), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.6194, mae: 0.2610, huber: 0.1014, swd: 0.4596, target_std: 3.9626
    Epoch [6/50], Val Losses: mse: 1.6307, mae: 0.3659, huber: 0.1808, swd: 1.3492, target_std: 4.0274
    Epoch [6/50], Test Losses: mse: 1.2959, mae: 0.3641, huber: 0.1725, swd: 1.0025, target_std: 4.1166
      Epoch 6 composite train-obj: 0.101408
            Val objective improved 0.2086 → 0.1808, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.9284, mae: 0.2428, huber: 0.1081, swd: 0.7946, target_std: 3.9588
    Epoch [7/50], Val Losses: mse: 0.7301, mae: 0.2656, huber: 0.1044, swd: 0.5797, target_std: 4.0274
    Epoch [7/50], Test Losses: mse: 0.5739, mae: 0.2652, huber: 0.0977, swd: 0.4421, target_std: 4.1166
      Epoch 7 composite train-obj: 0.108133
            Val objective improved 0.1808 → 0.1044, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.5136, mae: 0.2333, huber: 0.0865, swd: 0.3813, target_std: 3.9667
    Epoch [8/50], Val Losses: mse: 1.3083, mae: 0.3767, huber: 0.1754, swd: 0.9922, target_std: 4.0274
    Epoch [8/50], Test Losses: mse: 1.1299, mae: 0.3953, huber: 0.1833, swd: 0.7467, target_std: 4.1166
      Epoch 8 composite train-obj: 0.086525
            No improvement (0.1754), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.5362, mae: 0.2275, huber: 0.0861, swd: 0.3655, target_std: 3.9625
    Epoch [9/50], Val Losses: mse: 1.1832, mae: 0.2865, huber: 0.1336, swd: 1.0367, target_std: 4.0274
    Epoch [9/50], Test Losses: mse: 0.9526, mae: 0.2858, huber: 0.1287, swd: 0.7967, target_std: 4.1166
      Epoch 9 composite train-obj: 0.086123
            No improvement (0.1336), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.4512, mae: 0.1926, huber: 0.0693, swd: 0.3473, target_std: 3.9555
    Epoch [10/50], Val Losses: mse: 0.4772, mae: 0.2408, huber: 0.0994, swd: 0.2767, target_std: 4.0274
    Epoch [10/50], Test Losses: mse: 0.5847, mae: 0.2537, huber: 0.1104, swd: 0.3592, target_std: 4.1166
      Epoch 10 composite train-obj: 0.069300
            Val objective improved 0.1044 → 0.0994, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.1577, mae: 0.1624, huber: 0.0437, swd: 0.0811, target_std: 3.9617
    Epoch [11/50], Val Losses: mse: 0.3193, mae: 0.1777, huber: 0.0609, swd: 0.1609, target_std: 4.0274
    Epoch [11/50], Test Losses: mse: 0.2501, mae: 0.1785, huber: 0.0562, swd: 0.1332, target_std: 4.1166
      Epoch 11 composite train-obj: 0.043737
            Val objective improved 0.0994 → 0.0609, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.1300, mae: 0.1559, huber: 0.0392, swd: 0.0632, target_std: 3.9635
    Epoch [12/50], Val Losses: mse: 0.6030, mae: 0.3528, huber: 0.1522, swd: 0.2752, target_std: 4.0274
    Epoch [12/50], Test Losses: mse: 0.5680, mae: 0.3833, huber: 0.1683, swd: 0.2750, target_std: 4.1166
      Epoch 12 composite train-obj: 0.039221
            No improvement (0.1522), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0993, mae: 0.1473, huber: 0.0339, swd: 0.0411, target_std: 3.9574
    Epoch [13/50], Val Losses: mse: 0.3953, mae: 0.3021, huber: 0.1301, swd: 0.1576, target_std: 4.0274
    Epoch [13/50], Test Losses: mse: 0.7021, mae: 0.3744, huber: 0.1913, swd: 0.2783, target_std: 4.1166
      Epoch 13 composite train-obj: 0.033930
            No improvement (0.1301), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.1208, mae: 0.1566, huber: 0.0390, swd: 0.0549, target_std: 3.9530
    Epoch [14/50], Val Losses: mse: 0.1667, mae: 0.1834, huber: 0.0541, swd: 0.0690, target_std: 4.0274
    Epoch [14/50], Test Losses: mse: 0.1734, mae: 0.2028, huber: 0.0621, swd: 0.0765, target_std: 4.1166
      Epoch 14 composite train-obj: 0.039039
            Val objective improved 0.0609 → 0.0541, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.0532, mae: 0.1149, huber: 0.0209, swd: 0.0221, target_std: 3.9662
    Epoch [15/50], Val Losses: mse: 0.1126, mae: 0.1822, huber: 0.0444, swd: 0.0453, target_std: 4.0274
    Epoch [15/50], Test Losses: mse: 0.1260, mae: 0.1975, huber: 0.0504, swd: 0.0514, target_std: 4.1166
      Epoch 15 composite train-obj: 0.020864
            Val objective improved 0.0541 → 0.0444, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 0.0631, mae: 0.1247, huber: 0.0243, swd: 0.0276, target_std: 3.9606
    Epoch [16/50], Val Losses: mse: 0.2213, mae: 0.2400, huber: 0.0732, swd: 0.0784, target_std: 4.0274
    Epoch [16/50], Test Losses: mse: 0.2216, mae: 0.2447, huber: 0.0746, swd: 0.0925, target_std: 4.1166
      Epoch 16 composite train-obj: 0.024272
            No improvement (0.0732), counter 1/5
    Epoch [17/50], Train Losses: mse: 0.0602, mae: 0.1241, huber: 0.0232, swd: 0.0219, target_std: 3.9552
    Epoch [17/50], Val Losses: mse: 0.1986, mae: 0.2532, huber: 0.0825, swd: 0.0478, target_std: 4.0274
    Epoch [17/50], Test Losses: mse: 0.3240, mae: 0.3025, huber: 0.1179, swd: 0.1201, target_std: 4.1166
      Epoch 17 composite train-obj: 0.023210
            No improvement (0.0825), counter 2/5
    Epoch [18/50], Train Losses: mse: 0.0652, mae: 0.1333, huber: 0.0266, swd: 0.0273, target_std: 3.9564
    Epoch [18/50], Val Losses: mse: 0.1966, mae: 0.2349, huber: 0.0701, swd: 0.0734, target_std: 4.0274
    Epoch [18/50], Test Losses: mse: 0.2356, mae: 0.2602, huber: 0.0859, swd: 0.0987, target_std: 4.1166
      Epoch 18 composite train-obj: 0.026568
            No improvement (0.0701), counter 3/5
    Epoch [19/50], Train Losses: mse: 0.0495, mae: 0.1177, huber: 0.0203, swd: 0.0214, target_std: 3.9631
    Epoch [19/50], Val Losses: mse: 0.1326, mae: 0.1626, huber: 0.0428, swd: 0.0727, target_std: 4.0274
    Epoch [19/50], Test Losses: mse: 0.1375, mae: 0.1738, huber: 0.0458, swd: 0.0838, target_std: 4.1166
      Epoch 19 composite train-obj: 0.020274
            Val objective improved 0.0444 → 0.0428, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 0.0418, mae: 0.1042, huber: 0.0176, swd: 0.0180, target_std: 3.9608
    Epoch [20/50], Val Losses: mse: 0.1484, mae: 0.2245, huber: 0.0591, swd: 0.0656, target_std: 4.0274
    Epoch [20/50], Test Losses: mse: 0.1626, mae: 0.2334, huber: 0.0632, swd: 0.0672, target_std: 4.1166
      Epoch 20 composite train-obj: 0.017559
            No improvement (0.0591), counter 1/5
    Epoch [21/50], Train Losses: mse: 0.0350, mae: 0.0967, huber: 0.0149, swd: 0.0153, target_std: 3.9623
    Epoch [21/50], Val Losses: mse: 1.5623, mae: 0.4773, huber: 0.2866, swd: 0.6076, target_std: 4.0274
    Epoch [21/50], Test Losses: mse: 1.2728, mae: 0.4769, huber: 0.2741, swd: 0.4914, target_std: 4.1166
      Epoch 21 composite train-obj: 0.014885
            No improvement (0.2866), counter 2/5
    Epoch [22/50], Train Losses: mse: 0.4605, mae: 0.1840, huber: 0.0690, swd: 0.3226, target_std: 3.9603
    Epoch [22/50], Val Losses: mse: 0.0901, mae: 0.1210, huber: 0.0274, swd: 0.0486, target_std: 4.0274
    Epoch [22/50], Test Losses: mse: 0.0788, mae: 0.1257, huber: 0.0274, swd: 0.0357, target_std: 4.1166
      Epoch 22 composite train-obj: 0.068989
            Val objective improved 0.0428 → 0.0274, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 0.0681, mae: 0.1198, huber: 0.0247, swd: 0.0288, target_std: 3.9583
    Epoch [23/50], Val Losses: mse: 0.1294, mae: 0.1689, huber: 0.0462, swd: 0.0448, target_std: 4.0274
    Epoch [23/50], Test Losses: mse: 0.1425, mae: 0.1833, huber: 0.0507, swd: 0.0537, target_std: 4.1166
      Epoch 23 composite train-obj: 0.024724
            No improvement (0.0462), counter 1/5
    Epoch [24/50], Train Losses: mse: 0.0422, mae: 0.1065, huber: 0.0179, swd: 0.0172, target_std: 3.9627
    Epoch [24/50], Val Losses: mse: 0.0539, mae: 0.1196, huber: 0.0220, swd: 0.0213, target_std: 4.0274
    Epoch [24/50], Test Losses: mse: 0.0583, mae: 0.1291, huber: 0.0239, swd: 0.0253, target_std: 4.1166
      Epoch 24 composite train-obj: 0.017859
            Val objective improved 0.0274 → 0.0220, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 0.0302, mae: 0.0903, huber: 0.0129, swd: 0.0131, target_std: 3.9562
    Epoch [25/50], Val Losses: mse: 0.1081, mae: 0.1285, huber: 0.0335, swd: 0.0689, target_std: 4.0274
    Epoch [25/50], Test Losses: mse: 0.1021, mae: 0.1331, huber: 0.0339, swd: 0.0678, target_std: 4.1166
      Epoch 25 composite train-obj: 0.012922
            No improvement (0.0335), counter 1/5
    Epoch [26/50], Train Losses: mse: 0.0296, mae: 0.0905, huber: 0.0129, swd: 0.0133, target_std: 3.9648
    Epoch [26/50], Val Losses: mse: 0.5308, mae: 0.2578, huber: 0.1031, swd: 0.2521, target_std: 4.0274
    Epoch [26/50], Test Losses: mse: 0.6280, mae: 0.2853, huber: 0.1216, swd: 0.2676, target_std: 4.1166
      Epoch 26 composite train-obj: 0.012896
            No improvement (0.1031), counter 2/5
    Epoch [27/50], Train Losses: mse: 0.1174, mae: 0.1194, huber: 0.0268, swd: 0.0516, target_std: 3.9605
    Epoch [27/50], Val Losses: mse: 0.0251, mae: 0.0960, huber: 0.0119, swd: 0.0101, target_std: 4.0274
    Epoch [27/50], Test Losses: mse: 0.0266, mae: 0.1031, huber: 0.0127, swd: 0.0107, target_std: 4.1166
      Epoch 27 composite train-obj: 0.026754
            Val objective improved 0.0220 → 0.0119, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 0.0353, mae: 0.0991, huber: 0.0153, swd: 0.0154, target_std: 3.9597
    Epoch [28/50], Val Losses: mse: 0.0625, mae: 0.1013, huber: 0.0236, swd: 0.0209, target_std: 4.0274
    Epoch [28/50], Test Losses: mse: 0.0589, mae: 0.1082, huber: 0.0239, swd: 0.0241, target_std: 4.1166
      Epoch 28 composite train-obj: 0.015314
            No improvement (0.0236), counter 1/5
    Epoch [29/50], Train Losses: mse: 0.0278, mae: 0.0909, huber: 0.0125, swd: 0.0121, target_std: 3.9632
    Epoch [29/50], Val Losses: mse: 0.1125, mae: 0.1647, huber: 0.0427, swd: 0.0503, target_std: 4.0274
    Epoch [29/50], Test Losses: mse: 0.1188, mae: 0.1744, huber: 0.0474, swd: 0.0510, target_std: 4.1166
      Epoch 29 composite train-obj: 0.012507
            No improvement (0.0427), counter 2/5
    Epoch [30/50], Train Losses: mse: 0.0249, mae: 0.0836, huber: 0.0111, swd: 0.0112, target_std: 3.9585
    Epoch [30/50], Val Losses: mse: 0.0394, mae: 0.0910, huber: 0.0152, swd: 0.0231, target_std: 4.0274
    Epoch [30/50], Test Losses: mse: 0.0373, mae: 0.0936, huber: 0.0149, swd: 0.0253, target_std: 4.1166
      Epoch 30 composite train-obj: 0.011067
            No improvement (0.0152), counter 3/5
    Epoch [31/50], Train Losses: mse: 0.0317, mae: 0.0954, huber: 0.0140, swd: 0.0142, target_std: 3.9633
    Epoch [31/50], Val Losses: mse: 0.0568, mae: 0.1393, huber: 0.0247, swd: 0.0259, target_std: 4.0274
    Epoch [31/50], Test Losses: mse: 0.0614, mae: 0.1486, huber: 0.0270, swd: 0.0294, target_std: 4.1166
      Epoch 31 composite train-obj: 0.013977
            No improvement (0.0247), counter 4/5
    Epoch [32/50], Train Losses: mse: 0.0265, mae: 0.0884, huber: 0.0120, swd: 0.0121, target_std: 3.9577
    Epoch [32/50], Val Losses: mse: 0.1879, mae: 0.2006, huber: 0.0661, swd: 0.0886, target_std: 4.0274
    Epoch [32/50], Test Losses: mse: 0.2014, mae: 0.2210, huber: 0.0755, swd: 0.1007, target_std: 4.1166
      Epoch 32 composite train-obj: 0.011960
    Epoch [32/50], Test Losses: mse: 0.0266, mae: 0.1031, huber: 0.0127, swd: 0.0107, target_std: 4.1166
    Best round's Test MSE: 0.0266, MAE: 0.1031, SWD: 0.0107
    Best round's Validation MSE: 0.0251, MAE: 0.0960
    Best round's Test verification MSE : 0.0266, MAE: 0.1031, SWD: 0.0107
    
    ==================================================
    Experiment Summary (ACL_rossler_seq96_pred720_20250430_1413)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0664 ± 0.0350
      mae: 0.1429 ± 0.0454
      huber: 0.0290 ± 0.0156
      swd: 0.0292 ± 0.0164
      target_std: 4.1166 ± 0.0000
      count: 35.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0585 ± 0.0311
      mae: 0.1323 ± 0.0424
      huber: 0.0257 ± 0.0140
      swd: 0.0229 ± 0.0115
      target_std: 4.0274 ± 0.0000
      count: 35.0000 ± 0.0000
    ==================================================
    
    Experiment complete: ACL_rossler_seq96_pred720_20250430_1413
    Model: ACL
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


```python

```

### Timemixer

#### pred=96

##### huber


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['rossler']['channels'],
    enc_in=data_mgr.datasets['rossler']['channels'],
    dec_in=data_mgr.datasets['rossler']['channels'],
    c_out=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_mixer_96_96 = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 7.0772, mae: 1.1023, huber: 0.8271, swd: 4.2984, ept: 87.7331
    Epoch [1/50], Val Losses: mse: 2.9710, mae: 0.5809, huber: 0.3859, swd: 2.1493, ept: 92.1416
    Epoch [1/50], Test Losses: mse: 2.7847, mae: 0.6110, huber: 0.4021, swd: 2.0363, ept: 91.8116
      Epoch 1 composite train-obj: 0.827071
            Val objective improved inf → 0.3859, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.4011, mae: 0.5996, huber: 0.3744, swd: 1.8219, ept: 92.6557
    Epoch [2/50], Val Losses: mse: 2.0313, mae: 0.4482, huber: 0.2717, swd: 1.6673, ept: 94.0336
    Epoch [2/50], Test Losses: mse: 1.9243, mae: 0.4660, huber: 0.2810, swd: 1.5677, ept: 93.5981
      Epoch 2 composite train-obj: 0.374425
            Val objective improved 0.3859 → 0.2717, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.9856, mae: 0.5083, huber: 0.3024, swd: 1.5808, ept: 93.4894
    Epoch [3/50], Val Losses: mse: 1.8579, mae: 0.4214, huber: 0.2426, swd: 1.5249, ept: 94.1202
    Epoch [3/50], Test Losses: mse: 1.7568, mae: 0.4391, huber: 0.2505, swd: 1.4330, ept: 93.7954
      Epoch 3 composite train-obj: 0.302394
            Val objective improved 0.2717 → 0.2426, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 1.8442, mae: 0.4625, huber: 0.2709, swd: 1.4847, ept: 93.7861
    Epoch [4/50], Val Losses: mse: 1.7547, mae: 0.4080, huber: 0.2270, swd: 1.4661, ept: 94.3334
    Epoch [4/50], Test Losses: mse: 1.6591, mae: 0.4252, huber: 0.2349, swd: 1.3744, ept: 94.0485
      Epoch 4 composite train-obj: 0.270888
            Val objective improved 0.2426 → 0.2270, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 1.7602, mae: 0.4308, huber: 0.2494, swd: 1.4279, ept: 93.9451
    Epoch [5/50], Val Losses: mse: 1.6745, mae: 0.3420, huber: 0.2018, swd: 1.4135, ept: 94.2960
    Epoch [5/50], Test Losses: mse: 1.5999, mae: 0.3627, huber: 0.2141, swd: 1.3397, ept: 93.9232
      Epoch 5 composite train-obj: 0.249376
            Val objective improved 0.2270 → 0.2018, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 1.6892, mae: 0.4078, huber: 0.2346, swd: 1.3742, ept: 94.0589
    Epoch [6/50], Val Losses: mse: 1.6685, mae: 0.3807, huber: 0.2090, swd: 1.3956, ept: 94.4243
    Epoch [6/50], Test Losses: mse: 1.6008, mae: 0.4062, huber: 0.2234, swd: 1.3271, ept: 93.9972
      Epoch 6 composite train-obj: 0.234556
            No improvement (0.2090), counter 1/5
    Epoch [7/50], Train Losses: mse: 1.6322, mae: 0.3917, huber: 0.2244, swd: 1.3302, ept: 94.1522
    Epoch [7/50], Val Losses: mse: 1.5892, mae: 0.3442, huber: 0.1937, swd: 1.3241, ept: 94.3710
    Epoch [7/50], Test Losses: mse: 1.5131, mae: 0.3661, huber: 0.2048, swd: 1.2506, ept: 94.0805
      Epoch 7 composite train-obj: 0.224372
            Val objective improved 0.2018 → 0.1937, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 1.5829, mae: 0.3806, huber: 0.2170, swd: 1.2915, ept: 94.2054
    Epoch [8/50], Val Losses: mse: 1.5164, mae: 0.3416, huber: 0.1843, swd: 1.2766, ept: 94.6559
    Epoch [8/50], Test Losses: mse: 1.4635, mae: 0.3701, huber: 0.2014, swd: 1.2234, ept: 94.2189
      Epoch 8 composite train-obj: 0.217024
            Val objective improved 0.1937 → 0.1843, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 1.5305, mae: 0.3686, huber: 0.2092, swd: 1.2437, ept: 94.2818
    Epoch [9/50], Val Losses: mse: 1.4277, mae: 0.2967, huber: 0.1684, swd: 1.1826, ept: 94.6387
    Epoch [9/50], Test Losses: mse: 1.3497, mae: 0.3132, huber: 0.1766, swd: 1.1085, ept: 94.3564
      Epoch 9 composite train-obj: 0.209208
            Val objective improved 0.1843 → 0.1684, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 1.5035, mae: 0.3595, huber: 0.2040, swd: 1.2227, ept: 94.3026
    Epoch [10/50], Val Losses: mse: 1.4806, mae: 0.3045, huber: 0.1781, swd: 1.2285, ept: 94.6860
    Epoch [10/50], Test Losses: mse: 1.4137, mae: 0.3279, huber: 0.1917, swd: 1.1558, ept: 94.2691
      Epoch 10 composite train-obj: 0.203993
            No improvement (0.1781), counter 1/5
    Epoch [11/50], Train Losses: mse: 1.4725, mae: 0.3507, huber: 0.1984, swd: 1.1956, ept: 94.3583
    Epoch [11/50], Val Losses: mse: 1.4168, mae: 0.2804, huber: 0.1661, swd: 1.2019, ept: 94.5429
    Epoch [11/50], Test Losses: mse: 1.3485, mae: 0.2990, huber: 0.1770, swd: 1.1325, ept: 94.1755
      Epoch 11 composite train-obj: 0.198389
            Val objective improved 0.1684 → 0.1661, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 1.4179, mae: 0.3450, huber: 0.1939, swd: 1.1457, ept: 94.3775
    Epoch [12/50], Val Losses: mse: 1.3043, mae: 0.2847, huber: 0.1553, swd: 1.0931, ept: 94.8700
    Epoch [12/50], Test Losses: mse: 1.2351, mae: 0.3042, huber: 0.1653, swd: 1.0222, ept: 94.5238
      Epoch 12 composite train-obj: 0.193864
            Val objective improved 0.1661 → 0.1553, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 1.3817, mae: 0.3372, huber: 0.1886, swd: 1.1132, ept: 94.4443
    Epoch [13/50], Val Losses: mse: 1.3262, mae: 0.2793, huber: 0.1565, swd: 1.1022, ept: 94.6270
    Epoch [13/50], Test Losses: mse: 1.2429, mae: 0.2953, huber: 0.1640, swd: 1.0226, ept: 94.3453
      Epoch 13 composite train-obj: 0.188618
            No improvement (0.1565), counter 1/5
    Epoch [14/50], Train Losses: mse: 1.3545, mae: 0.3289, huber: 0.1837, swd: 1.0898, ept: 94.4592
    Epoch [14/50], Val Losses: mse: 1.2495, mae: 0.3143, huber: 0.1604, swd: 1.0013, ept: 94.8254
    Epoch [14/50], Test Losses: mse: 1.1601, mae: 0.3283, huber: 0.1643, swd: 0.9190, ept: 94.6930
      Epoch 14 composite train-obj: 0.183665
            No improvement (0.1604), counter 2/5
    Epoch [15/50], Train Losses: mse: 1.3019, mae: 0.3195, huber: 0.1768, swd: 1.0451, ept: 94.5326
    Epoch [15/50], Val Losses: mse: 1.3525, mae: 0.3482, huber: 0.1793, swd: 1.1328, ept: 94.6378
    Epoch [15/50], Test Losses: mse: 1.2533, mae: 0.3668, huber: 0.1855, swd: 1.0394, ept: 94.4352
      Epoch 15 composite train-obj: 0.176817
            No improvement (0.1793), counter 3/5
    Epoch [16/50], Train Losses: mse: 1.2945, mae: 0.3184, huber: 0.1763, swd: 1.0400, ept: 94.5324
    Epoch [16/50], Val Losses: mse: 1.1890, mae: 0.3409, huber: 0.1712, swd: 0.9754, ept: 94.9044
    Epoch [16/50], Test Losses: mse: 1.1237, mae: 0.3674, huber: 0.1829, swd: 0.9088, ept: 94.6793
      Epoch 16 composite train-obj: 0.176260
            No improvement (0.1712), counter 4/5
    Epoch [17/50], Train Losses: mse: 1.2831, mae: 0.3189, huber: 0.1759, swd: 1.0244, ept: 94.5340
    Epoch [17/50], Val Losses: mse: 1.1202, mae: 0.2675, huber: 0.1450, swd: 0.9254, ept: 94.8836
    Epoch [17/50], Test Losses: mse: 1.0352, mae: 0.2837, huber: 0.1502, swd: 0.8438, ept: 94.7832
      Epoch 17 composite train-obj: 0.175888
            Val objective improved 0.1553 → 0.1450, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 1.1798, mae: 0.3055, huber: 0.1655, swd: 0.9400, ept: 94.6374
    Epoch [18/50], Val Losses: mse: 1.5376, mae: 0.3216, huber: 0.1787, swd: 1.3037, ept: 94.5027
    Epoch [18/50], Test Losses: mse: 1.4083, mae: 0.3361, huber: 0.1842, swd: 1.1887, ept: 94.1746
      Epoch 18 composite train-obj: 0.165493
            No improvement (0.1787), counter 1/5
    Epoch [19/50], Train Losses: mse: 1.1781, mae: 0.3009, huber: 0.1635, swd: 0.9345, ept: 94.6416
    Epoch [19/50], Val Losses: mse: 1.1716, mae: 0.2777, huber: 0.1526, swd: 0.9733, ept: 94.8255
    Epoch [19/50], Test Losses: mse: 1.0503, mae: 0.2864, huber: 0.1516, swd: 0.8612, ept: 94.6882
      Epoch 19 composite train-obj: 0.163467
            No improvement (0.1526), counter 2/5
    Epoch [20/50], Train Losses: mse: 1.1199, mae: 0.2979, huber: 0.1600, swd: 0.8796, ept: 94.6746
    Epoch [20/50], Val Losses: mse: 1.0113, mae: 0.2610, huber: 0.1392, swd: 0.8325, ept: 94.7400
    Epoch [20/50], Test Losses: mse: 0.9321, mae: 0.2746, huber: 0.1435, swd: 0.7582, ept: 94.5982
      Epoch 20 composite train-obj: 0.159955
            Val objective improved 0.1450 → 0.1392, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 1.0493, mae: 0.2879, huber: 0.1529, swd: 0.8236, ept: 94.7504
    Epoch [21/50], Val Losses: mse: 1.1232, mae: 0.2899, huber: 0.1536, swd: 0.9357, ept: 94.7353
    Epoch [21/50], Test Losses: mse: 1.0197, mae: 0.3042, huber: 0.1563, swd: 0.8418, ept: 94.5850
      Epoch 21 composite train-obj: 0.152865
            No improvement (0.1536), counter 1/5
    Epoch [22/50], Train Losses: mse: 1.1197, mae: 0.2953, huber: 0.1589, swd: 0.8887, ept: 94.6750
    Epoch [22/50], Val Losses: mse: 0.7609, mae: 0.2134, huber: 0.1093, swd: 0.6060, ept: 95.0719
    Epoch [22/50], Test Losses: mse: 0.6905, mae: 0.2207, huber: 0.1089, swd: 0.5410, ept: 95.1078
      Epoch 22 composite train-obj: 0.158864
            Val objective improved 0.1392 → 0.1093, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 1.0041, mae: 0.2837, huber: 0.1496, swd: 0.7835, ept: 94.7601
    Epoch [23/50], Val Losses: mse: 1.5216, mae: 0.3192, huber: 0.1834, swd: 1.2589, ept: 94.5099
    Epoch [23/50], Test Losses: mse: 1.4127, mae: 0.3418, huber: 0.1936, swd: 1.1646, ept: 94.0519
      Epoch 23 composite train-obj: 0.149619
            No improvement (0.1834), counter 1/5
    Epoch [24/50], Train Losses: mse: 0.9502, mae: 0.2741, huber: 0.1429, swd: 0.7433, ept: 94.8202
    Epoch [24/50], Val Losses: mse: 0.9843, mae: 0.3430, huber: 0.1638, swd: 0.8081, ept: 94.7560
    Epoch [24/50], Test Losses: mse: 0.8974, mae: 0.3604, huber: 0.1683, swd: 0.7313, ept: 94.6992
      Epoch 24 composite train-obj: 0.142885
            No improvement (0.1638), counter 2/5
    Epoch [25/50], Train Losses: mse: 0.9581, mae: 0.2764, huber: 0.1447, swd: 0.7430, ept: 94.7984
    Epoch [25/50], Val Losses: mse: 0.8432, mae: 0.3046, huber: 0.1503, swd: 0.6997, ept: 94.6594
    Epoch [25/50], Test Losses: mse: 0.7938, mae: 0.3259, huber: 0.1579, swd: 0.6543, ept: 94.6538
      Epoch 25 composite train-obj: 0.144693
            No improvement (0.1503), counter 3/5
    Epoch [26/50], Train Losses: mse: 0.8621, mae: 0.2701, huber: 0.1381, swd: 0.6634, ept: 94.8736
    Epoch [26/50], Val Losses: mse: 0.9269, mae: 0.3045, huber: 0.1494, swd: 0.7297, ept: 94.6960
    Epoch [26/50], Test Losses: mse: 0.8199, mae: 0.3136, huber: 0.1461, swd: 0.6379, ept: 94.8515
      Epoch 26 composite train-obj: 0.138089
            No improvement (0.1494), counter 4/5
    Epoch [27/50], Train Losses: mse: 0.8467, mae: 0.2636, huber: 0.1345, swd: 0.6551, ept: 94.8984
    Epoch [27/50], Val Losses: mse: 0.9966, mae: 0.2762, huber: 0.1505, swd: 0.8192, ept: 94.7719
    Epoch [27/50], Test Losses: mse: 0.9167, mae: 0.2937, huber: 0.1554, swd: 0.7528, ept: 94.6467
      Epoch 27 composite train-obj: 0.134486
    Epoch [27/50], Test Losses: mse: 0.6905, mae: 0.2207, huber: 0.1089, swd: 0.5410, ept: 95.1078
    Best round's Test MSE: 0.6905, MAE: 0.2207, SWD: 0.5410
    Best round's Validation MSE: 0.7609, MAE: 0.2134, SWD: 0.6060
    Best round's Test verification MSE : 0.6905, MAE: 0.2207, SWD: 0.5410
    Time taken: 224.54 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.4328, mae: 1.1166, huber: 0.8428, swd: 4.5285, ept: 87.9756
    Epoch [1/50], Val Losses: mse: 3.1189, mae: 0.6020, huber: 0.4020, swd: 2.1398, ept: 92.1708
    Epoch [1/50], Test Losses: mse: 2.9076, mae: 0.6331, huber: 0.4182, swd: 2.0221, ept: 91.6829
      Epoch 1 composite train-obj: 0.842785
            Val objective improved inf → 0.4020, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.6403, mae: 0.6203, huber: 0.3980, swd: 1.9687, ept: 92.4234
    Epoch [2/50], Val Losses: mse: 2.2213, mae: 0.4577, huber: 0.2966, swd: 1.7442, ept: 93.4088
    Epoch [2/50], Test Losses: mse: 2.1027, mae: 0.4859, huber: 0.3100, swd: 1.6439, ept: 93.2045
      Epoch 2 composite train-obj: 0.397956
            Val objective improved 0.4020 → 0.2966, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.0495, mae: 0.5284, huber: 0.3186, swd: 1.6165, ept: 93.4201
    Epoch [3/50], Val Losses: mse: 1.9436, mae: 0.4094, huber: 0.2479, swd: 1.6477, ept: 94.1546
    Epoch [3/50], Test Losses: mse: 1.8624, mae: 0.4318, huber: 0.2613, swd: 1.5642, ept: 93.6046
      Epoch 3 composite train-obj: 0.318556
            Val objective improved 0.2966 → 0.2479, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 1.8579, mae: 0.4748, huber: 0.2776, swd: 1.5016, ept: 93.8207
    Epoch [4/50], Val Losses: mse: 1.7401, mae: 0.3729, huber: 0.2221, swd: 1.4392, ept: 94.2396
    Epoch [4/50], Test Losses: mse: 1.6769, mae: 0.3984, huber: 0.2363, swd: 1.3698, ept: 93.8873
      Epoch 4 composite train-obj: 0.277637
            Val objective improved 0.2479 → 0.2221, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 1.7610, mae: 0.4424, huber: 0.2561, swd: 1.4373, ept: 93.9580
    Epoch [5/50], Val Losses: mse: 1.7277, mae: 0.3525, huber: 0.2106, swd: 1.4695, ept: 94.3584
    Epoch [5/50], Test Losses: mse: 1.6643, mae: 0.3795, huber: 0.2266, swd: 1.3956, ept: 93.8562
      Epoch 5 composite train-obj: 0.256110
            Val objective improved 0.2221 → 0.2106, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 1.6874, mae: 0.4152, huber: 0.2386, swd: 1.3862, ept: 94.1095
    Epoch [6/50], Val Losses: mse: 1.5533, mae: 0.3198, huber: 0.1884, swd: 1.3367, ept: 94.6360
    Epoch [6/50], Test Losses: mse: 1.5118, mae: 0.3443, huber: 0.2054, swd: 1.2775, ept: 94.0636
      Epoch 6 composite train-obj: 0.238585
            Val objective improved 0.2106 → 0.1884, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 1.6208, mae: 0.3950, huber: 0.2252, swd: 1.3392, ept: 94.1888
    Epoch [7/50], Val Losses: mse: 1.4968, mae: 0.3196, huber: 0.1860, swd: 1.2719, ept: 94.6209
    Epoch [7/50], Test Losses: mse: 1.4379, mae: 0.3412, huber: 0.1988, swd: 1.2027, ept: 94.3350
      Epoch 7 composite train-obj: 0.225177
            Val objective improved 0.1884 → 0.1860, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 1.5541, mae: 0.3790, huber: 0.2144, swd: 1.2897, ept: 94.2615
    Epoch [8/50], Val Losses: mse: 1.4476, mae: 0.3095, huber: 0.1766, swd: 1.2470, ept: 94.7228
    Epoch [8/50], Test Losses: mse: 1.3776, mae: 0.3322, huber: 0.1886, swd: 1.1707, ept: 94.3795
      Epoch 8 composite train-obj: 0.214439
            Val objective improved 0.1860 → 0.1766, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 1.4903, mae: 0.3648, huber: 0.2046, swd: 1.2471, ept: 94.3481
    Epoch [9/50], Val Losses: mse: 1.5414, mae: 0.3103, huber: 0.1801, swd: 1.3618, ept: 94.4516
    Epoch [9/50], Test Losses: mse: 1.4785, mae: 0.3330, huber: 0.1937, swd: 1.2865, ept: 93.9908
      Epoch 9 composite train-obj: 0.204582
            No improvement (0.1801), counter 1/5
    Epoch [10/50], Train Losses: mse: 1.4566, mae: 0.3529, huber: 0.1972, swd: 1.2192, ept: 94.3940
    Epoch [10/50], Val Losses: mse: 1.3580, mae: 0.2712, huber: 0.1567, swd: 1.1958, ept: 94.7428
    Epoch [10/50], Test Losses: mse: 1.2833, mae: 0.2876, huber: 0.1661, swd: 1.1147, ept: 94.3439
      Epoch 10 composite train-obj: 0.197227
            Val objective improved 0.1766 → 0.1567, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 1.3879, mae: 0.3438, huber: 0.1903, swd: 1.1676, ept: 94.4293
    Epoch [11/50], Val Losses: mse: 1.6990, mae: 0.3213, huber: 0.1918, swd: 1.4967, ept: 94.3395
    Epoch [11/50], Test Losses: mse: 1.6317, mae: 0.3447, huber: 0.2068, swd: 1.4211, ept: 93.7944
      Epoch 11 composite train-obj: 0.190322
            No improvement (0.1918), counter 1/5
    Epoch [12/50], Train Losses: mse: 1.3733, mae: 0.3344, huber: 0.1842, swd: 1.1513, ept: 94.5099
    Epoch [12/50], Val Losses: mse: 1.4240, mae: 0.3139, huber: 0.1693, swd: 1.2484, ept: 94.7061
    Epoch [12/50], Test Losses: mse: 1.3453, mae: 0.3363, huber: 0.1811, swd: 1.1637, ept: 94.3591
      Epoch 12 composite train-obj: 0.184186
            No improvement (0.1693), counter 2/5
    Epoch [13/50], Train Losses: mse: 1.2481, mae: 0.3220, huber: 0.1747, swd: 1.0471, ept: 94.5830
    Epoch [13/50], Val Losses: mse: 1.1849, mae: 0.3014, huber: 0.1613, swd: 1.0049, ept: 94.4989
    Epoch [13/50], Test Losses: mse: 1.0744, mae: 0.3136, huber: 0.1618, swd: 0.9048, ept: 94.5027
      Epoch 13 composite train-obj: 0.174669
            No improvement (0.1613), counter 3/5
    Epoch [14/50], Train Losses: mse: 1.2304, mae: 0.3160, huber: 0.1713, swd: 1.0298, ept: 94.6186
    Epoch [14/50], Val Losses: mse: 0.9855, mae: 0.2677, huber: 0.1383, swd: 0.8756, ept: 95.0520
    Epoch [14/50], Test Losses: mse: 0.8900, mae: 0.2726, huber: 0.1375, swd: 0.7797, ept: 94.9594
      Epoch 14 composite train-obj: 0.171298
            Val objective improved 0.1567 → 0.1383, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 1.1248, mae: 0.3075, huber: 0.1636, swd: 0.9392, ept: 94.6807
    Epoch [15/50], Val Losses: mse: 0.9250, mae: 0.2336, huber: 0.1262, swd: 0.8360, ept: 94.8630
    Epoch [15/50], Test Losses: mse: 0.8208, mae: 0.2356, huber: 0.1219, swd: 0.7349, ept: 94.8728
      Epoch 15 composite train-obj: 0.163624
            Val objective improved 0.1383 → 0.1262, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 1.1995, mae: 0.3108, huber: 0.1673, swd: 0.9957, ept: 94.6386
    Epoch [16/50], Val Losses: mse: 0.9693, mae: 0.2464, huber: 0.1347, swd: 0.8725, ept: 94.9262
    Epoch [16/50], Test Losses: mse: 0.8713, mae: 0.2514, huber: 0.1325, swd: 0.7776, ept: 94.8724
      Epoch 16 composite train-obj: 0.167314
            No improvement (0.1347), counter 1/5
    Epoch [17/50], Train Losses: mse: 1.1849, mae: 0.3087, huber: 0.1662, swd: 0.9832, ept: 94.6184
    Epoch [17/50], Val Losses: mse: 0.7896, mae: 0.2331, huber: 0.1265, swd: 0.6345, ept: 94.9154
    Epoch [17/50], Test Losses: mse: 0.8350, mae: 0.2552, huber: 0.1397, swd: 0.6492, ept: 94.6822
      Epoch 17 composite train-obj: 0.166244
            No improvement (0.1265), counter 2/5
    Epoch [18/50], Train Losses: mse: 1.0116, mae: 0.2900, huber: 0.1521, swd: 0.8310, ept: 94.7590
    Epoch [18/50], Val Losses: mse: 0.9984, mae: 0.2521, huber: 0.1419, swd: 0.8822, ept: 94.7691
    Epoch [18/50], Test Losses: mse: 0.9089, mae: 0.2617, huber: 0.1436, swd: 0.7993, ept: 94.5451
      Epoch 18 composite train-obj: 0.152060
            No improvement (0.1419), counter 3/5
    Epoch [19/50], Train Losses: mse: 0.9754, mae: 0.2849, huber: 0.1477, swd: 0.7996, ept: 94.8055
    Epoch [19/50], Val Losses: mse: 0.6738, mae: 0.2414, huber: 0.1127, swd: 0.5675, ept: 95.0933
    Epoch [19/50], Test Losses: mse: 0.6304, mae: 0.2515, huber: 0.1141, swd: 0.5182, ept: 95.0827
      Epoch 19 composite train-obj: 0.147698
            Val objective improved 0.1262 → 0.1127, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 0.9448, mae: 0.2801, huber: 0.1443, swd: 0.7693, ept: 94.8242
    Epoch [20/50], Val Losses: mse: 0.5705, mae: 0.1909, huber: 0.0941, swd: 0.4439, ept: 95.2708
    Epoch [20/50], Test Losses: mse: 0.5245, mae: 0.1958, huber: 0.0932, swd: 0.4071, ept: 95.2169
      Epoch 20 composite train-obj: 0.144342
            Val objective improved 0.1127 → 0.0941, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 0.8470, mae: 0.2712, huber: 0.1378, swd: 0.6859, ept: 94.8719
    Epoch [21/50], Val Losses: mse: 0.8139, mae: 0.2297, huber: 0.1245, swd: 0.7007, ept: 94.8487
    Epoch [21/50], Test Losses: mse: 0.7402, mae: 0.2412, huber: 0.1251, swd: 0.6336, ept: 94.8864
      Epoch 21 composite train-obj: 0.137813
            No improvement (0.1245), counter 1/5
    Epoch [22/50], Train Losses: mse: 0.9155, mae: 0.2760, huber: 0.1412, swd: 0.7427, ept: 94.8462
    Epoch [22/50], Val Losses: mse: 0.8151, mae: 0.2364, huber: 0.1253, swd: 0.6952, ept: 94.8727
    Epoch [22/50], Test Losses: mse: 0.7420, mae: 0.2438, huber: 0.1247, swd: 0.6292, ept: 94.6883
      Epoch 22 composite train-obj: 0.141237
            No improvement (0.1253), counter 2/5
    Epoch [23/50], Train Losses: mse: 0.7919, mae: 0.2656, huber: 0.1324, swd: 0.6337, ept: 94.9143
    Epoch [23/50], Val Losses: mse: 0.6131, mae: 0.2524, huber: 0.1152, swd: 0.4890, ept: 95.0969
    Epoch [23/50], Test Losses: mse: 0.5685, mae: 0.2600, huber: 0.1136, swd: 0.4453, ept: 95.2368
      Epoch 23 composite train-obj: 0.132436
            No improvement (0.1152), counter 3/5
    Epoch [24/50], Train Losses: mse: 0.8116, mae: 0.2671, huber: 0.1335, swd: 0.6538, ept: 94.9038
    Epoch [24/50], Val Losses: mse: 0.4608, mae: 0.1916, huber: 0.0874, swd: 0.3677, ept: 95.3223
    Epoch [24/50], Test Losses: mse: 0.4304, mae: 0.1998, huber: 0.0874, swd: 0.3412, ept: 95.3046
      Epoch 24 composite train-obj: 0.133541
            Val objective improved 0.0941 → 0.0874, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 0.7732, mae: 0.2585, huber: 0.1286, swd: 0.6172, ept: 94.9538
    Epoch [25/50], Val Losses: mse: 0.5159, mae: 0.2187, huber: 0.0980, swd: 0.4055, ept: 95.1442
    Epoch [25/50], Test Losses: mse: 0.4711, mae: 0.2247, huber: 0.0946, swd: 0.3679, ept: 95.2600
      Epoch 25 composite train-obj: 0.128593
            No improvement (0.0980), counter 1/5
    Epoch [26/50], Train Losses: mse: 0.8031, mae: 0.2626, huber: 0.1311, swd: 0.6425, ept: 94.9308
    Epoch [26/50], Val Losses: mse: 0.7100, mae: 0.2167, huber: 0.1155, swd: 0.5942, ept: 94.9497
    Epoch [26/50], Test Losses: mse: 0.6332, mae: 0.2194, huber: 0.1124, swd: 0.5294, ept: 94.9131
      Epoch 26 composite train-obj: 0.131070
            No improvement (0.1155), counter 2/5
    Epoch [27/50], Train Losses: mse: 0.7533, mae: 0.2515, huber: 0.1245, swd: 0.6058, ept: 94.9990
    Epoch [27/50], Val Losses: mse: 0.9851, mae: 0.2429, huber: 0.1370, swd: 0.8965, ept: 94.6057
    Epoch [27/50], Test Losses: mse: 0.9730, mae: 0.2641, huber: 0.1493, swd: 0.8715, ept: 94.2303
      Epoch 27 composite train-obj: 0.124478
            No improvement (0.1370), counter 3/5
    Epoch [28/50], Train Losses: mse: 0.7047, mae: 0.2486, huber: 0.1214, swd: 0.5589, ept: 95.0169
    Epoch [28/50], Val Losses: mse: 0.8170, mae: 0.2228, huber: 0.1227, swd: 0.6837, ept: 94.7923
    Epoch [28/50], Test Losses: mse: 0.7284, mae: 0.2271, huber: 0.1198, swd: 0.6087, ept: 94.7326
      Epoch 28 composite train-obj: 0.121419
            No improvement (0.1227), counter 4/5
    Epoch [29/50], Train Losses: mse: 0.7570, mae: 0.2528, huber: 0.1245, swd: 0.6073, ept: 94.9900
    Epoch [29/50], Val Losses: mse: 0.6552, mae: 0.2120, huber: 0.1121, swd: 0.5572, ept: 94.9342
    Epoch [29/50], Test Losses: mse: 0.6199, mae: 0.2208, huber: 0.1146, swd: 0.5262, ept: 94.7636
      Epoch 29 composite train-obj: 0.124491
    Epoch [29/50], Test Losses: mse: 0.4304, mae: 0.1998, huber: 0.0874, swd: 0.3412, ept: 95.3046
    Best round's Test MSE: 0.4304, MAE: 0.1998, SWD: 0.3412
    Best round's Validation MSE: 0.4608, MAE: 0.1916, SWD: 0.3677
    Best round's Test verification MSE : 0.4304, MAE: 0.1998, SWD: 0.3412
    Time taken: 222.90 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.4413, mae: 1.1044, huber: 0.8322, swd: 4.1860, ept: 88.1301
    Epoch [1/50], Val Losses: mse: 3.0704, mae: 0.6086, huber: 0.4077, swd: 2.0009, ept: 92.5738
    Epoch [1/50], Test Losses: mse: 2.8993, mae: 0.6419, huber: 0.4272, swd: 1.8950, ept: 92.1509
      Epoch 1 composite train-obj: 0.832160
            Val objective improved inf → 0.4077, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.7966, mae: 0.6330, huber: 0.4104, swd: 1.9252, ept: 92.4737
    Epoch [2/50], Val Losses: mse: 2.4885, mae: 0.4963, huber: 0.3219, swd: 1.8433, ept: 93.2159
    Epoch [2/50], Test Losses: mse: 2.3338, mae: 0.5193, huber: 0.3323, swd: 1.7270, ept: 92.9691
      Epoch 2 composite train-obj: 0.410388
            Val objective improved 0.4077 → 0.3219, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.2083, mae: 0.5425, huber: 0.3301, swd: 1.6309, ept: 93.4303
    Epoch [3/50], Val Losses: mse: 2.0381, mae: 0.4850, huber: 0.2871, swd: 1.5669, ept: 94.1605
    Epoch [3/50], Test Losses: mse: 1.9567, mae: 0.5116, huber: 0.3028, swd: 1.4879, ept: 93.7399
      Epoch 3 composite train-obj: 0.330066
            Val objective improved 0.3219 → 0.2871, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 1.9531, mae: 0.4930, huber: 0.2916, swd: 1.4542, ept: 93.8018
    Epoch [4/50], Val Losses: mse: 1.8796, mae: 0.4070, huber: 0.2433, swd: 1.4335, ept: 94.3126
    Epoch [4/50], Test Losses: mse: 1.8067, mae: 0.4323, huber: 0.2592, swd: 1.3607, ept: 93.8407
      Epoch 4 composite train-obj: 0.291637
            Val objective improved 0.2871 → 0.2433, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 1.7971, mae: 0.4560, huber: 0.2648, swd: 1.3383, ept: 94.0196
    Epoch [5/50], Val Losses: mse: 1.7864, mae: 0.4240, huber: 0.2373, swd: 1.3527, ept: 94.3798
    Epoch [5/50], Test Losses: mse: 1.7148, mae: 0.4486, huber: 0.2519, swd: 1.2798, ept: 93.9536
      Epoch 5 composite train-obj: 0.264844
            Val objective improved 0.2433 → 0.2373, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 1.6978, mae: 0.4303, huber: 0.2471, swd: 1.2690, ept: 94.1382
    Epoch [6/50], Val Losses: mse: 1.6114, mae: 0.3883, huber: 0.2107, swd: 1.2547, ept: 94.5682
    Epoch [6/50], Test Losses: mse: 1.5341, mae: 0.4107, huber: 0.2228, swd: 1.1850, ept: 94.1936
      Epoch 6 composite train-obj: 0.247061
            Val objective improved 0.2373 → 0.2107, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 1.5866, mae: 0.4015, huber: 0.2276, swd: 1.1888, ept: 94.2595
    Epoch [7/50], Val Losses: mse: 1.7644, mae: 0.3781, huber: 0.2210, swd: 1.3046, ept: 94.2530
    Epoch [7/50], Test Losses: mse: 1.6662, mae: 0.3969, huber: 0.2305, swd: 1.2286, ept: 93.8589
      Epoch 7 composite train-obj: 0.227564
            No improvement (0.2210), counter 1/5
    Epoch [8/50], Train Losses: mse: 1.5142, mae: 0.3844, huber: 0.2151, swd: 1.1299, ept: 94.3107
    Epoch [8/50], Val Losses: mse: 1.6486, mae: 0.3837, huber: 0.2189, swd: 1.2834, ept: 94.3073
    Epoch [8/50], Test Losses: mse: 1.5614, mae: 0.4057, huber: 0.2310, swd: 1.2048, ept: 93.9270
      Epoch 8 composite train-obj: 0.215130
            No improvement (0.2189), counter 2/5
    Epoch [9/50], Train Losses: mse: 1.4413, mae: 0.3709, huber: 0.2054, swd: 1.0731, ept: 94.3624
    Epoch [9/50], Val Losses: mse: 1.5400, mae: 0.3731, huber: 0.2116, swd: 1.1693, ept: 94.5457
    Epoch [9/50], Test Losses: mse: 1.4836, mae: 0.4025, huber: 0.2293, swd: 1.0947, ept: 94.2129
      Epoch 9 composite train-obj: 0.205416
            No improvement (0.2116), counter 3/5
    Epoch [10/50], Train Losses: mse: 1.3872, mae: 0.3568, huber: 0.1960, swd: 1.0325, ept: 94.4370
    Epoch [10/50], Val Losses: mse: 1.3245, mae: 0.3177, huber: 0.1725, swd: 1.0402, ept: 94.6048
    Epoch [10/50], Test Losses: mse: 1.2382, mae: 0.3374, huber: 0.1818, swd: 0.9550, ept: 94.3608
      Epoch 10 composite train-obj: 0.195977
            Val objective improved 0.2107 → 0.1725, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 1.3468, mae: 0.3452, huber: 0.1890, swd: 0.9997, ept: 94.4549
    Epoch [11/50], Val Losses: mse: 1.5459, mae: 0.3775, huber: 0.2092, swd: 1.2094, ept: 94.6025
    Epoch [11/50], Test Losses: mse: 1.4555, mae: 0.4038, huber: 0.2243, swd: 1.1190, ept: 94.2595
      Epoch 11 composite train-obj: 0.188977
            No improvement (0.2092), counter 1/5
    Epoch [12/50], Train Losses: mse: 1.2819, mae: 0.3382, huber: 0.1830, swd: 0.9493, ept: 94.5061
    Epoch [12/50], Val Losses: mse: 1.2755, mae: 0.3556, huber: 0.1899, swd: 1.0075, ept: 94.6589
    Epoch [12/50], Test Losses: mse: 1.1944, mae: 0.3783, huber: 0.2011, swd: 0.9287, ept: 94.5144
      Epoch 12 composite train-obj: 0.182962
            No improvement (0.1899), counter 2/5
    Epoch [13/50], Train Losses: mse: 1.2107, mae: 0.3237, huber: 0.1742, swd: 0.8883, ept: 94.5501
    Epoch [13/50], Val Losses: mse: 1.0588, mae: 0.2877, huber: 0.1490, swd: 0.7682, ept: 94.8508
    Epoch [13/50], Test Losses: mse: 0.9916, mae: 0.3030, huber: 0.1538, swd: 0.7090, ept: 94.7239
      Epoch 13 composite train-obj: 0.174178
            Val objective improved 0.1725 → 0.1490, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 1.1418, mae: 0.3136, huber: 0.1667, swd: 0.8322, ept: 94.6116
    Epoch [14/50], Val Losses: mse: 1.1524, mae: 0.3623, huber: 0.1856, swd: 0.8841, ept: 94.7149
    Epoch [14/50], Test Losses: mse: 1.1228, mae: 0.3940, huber: 0.2050, swd: 0.8406, ept: 94.3135
      Epoch 14 composite train-obj: 0.166728
            No improvement (0.1856), counter 1/5
    Epoch [15/50], Train Losses: mse: 1.0932, mae: 0.3098, huber: 0.1628, swd: 0.7917, ept: 94.6598
    Epoch [15/50], Val Losses: mse: 1.4210, mae: 0.3226, huber: 0.1843, swd: 1.1373, ept: 94.6809
    Epoch [15/50], Test Losses: mse: 1.3628, mae: 0.3574, huber: 0.2048, swd: 1.0741, ept: 94.1891
      Epoch 15 composite train-obj: 0.162791
            No improvement (0.1843), counter 2/5
    Epoch [16/50], Train Losses: mse: 1.0335, mae: 0.2982, huber: 0.1558, swd: 0.7444, ept: 94.6949
    Epoch [16/50], Val Losses: mse: 1.5819, mae: 0.3709, huber: 0.2112, swd: 1.2403, ept: 94.4439
    Epoch [16/50], Test Losses: mse: 1.5241, mae: 0.4046, huber: 0.2321, swd: 1.1749, ept: 94.0113
      Epoch 16 composite train-obj: 0.155784
            No improvement (0.2112), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.9839, mae: 0.2950, huber: 0.1520, swd: 0.6981, ept: 94.7445
    Epoch [17/50], Val Losses: mse: 1.4433, mae: 0.3497, huber: 0.1915, swd: 1.1352, ept: 94.4459
    Epoch [17/50], Test Losses: mse: 1.3983, mae: 0.3821, huber: 0.2107, swd: 1.0835, ept: 94.0161
      Epoch 17 composite train-obj: 0.151954
            No improvement (0.1915), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.9235, mae: 0.2849, huber: 0.1450, swd: 0.6537, ept: 94.7800
    Epoch [18/50], Val Losses: mse: 1.5186, mae: 0.3465, huber: 0.1970, swd: 1.1875, ept: 94.5182
    Epoch [18/50], Test Losses: mse: 1.4555, mae: 0.3785, huber: 0.2169, swd: 1.1266, ept: 94.1121
      Epoch 18 composite train-obj: 0.144952
    Epoch [18/50], Test Losses: mse: 0.9916, mae: 0.3030, huber: 0.1538, swd: 0.7090, ept: 94.7239
    Best round's Test MSE: 0.9916, MAE: 0.3030, SWD: 0.7090
    Best round's Validation MSE: 1.0588, MAE: 0.2877, SWD: 0.7682
    Best round's Test verification MSE : 0.9916, MAE: 0.3030, SWD: 0.7090
    Time taken: 160.82 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_rossler_seq96_pred96_20250513_0017)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.7042 ± 0.2293
      mae: 0.2412 ± 0.0446
      huber: 0.1167 ± 0.0276
      swd: 0.5304 ± 0.1503
      ept: 95.0454 ± 0.2411
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.7602 ± 0.2442
      mae: 0.2309 ± 0.0411
      huber: 0.1152 ± 0.0255
      swd: 0.5806 ± 0.1645
      ept: 95.0817 ± 0.1926
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 609.63 seconds
    
    Experiment complete: TimeMixer_rossler_seq96_pred96_20250513_0017
    Model: TimeMixer
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['rossler']['channels'],
    enc_in=data_mgr.datasets['rossler']['channels'],
    dec_in=data_mgr.datasets['rossler']['channels'],
    c_out=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 16.2527, mae: 1.9405, huber: 1.6176, swd: 9.6310, ept: 158.8868
    Epoch [1/50], Val Losses: mse: 8.6043, mae: 1.2510, huber: 0.9768, swd: 6.0323, ept: 175.2104
    Epoch [1/50], Test Losses: mse: 8.0749, mae: 1.3029, huber: 1.0110, swd: 5.4705, ept: 173.3087
      Epoch 1 composite train-obj: 1.617641
            Val objective improved inf → 0.9768, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.6421, mae: 1.1428, huber: 0.8545, swd: 4.7986, ept: 177.9137
    Epoch [2/50], Val Losses: mse: 5.3480, mae: 0.9916, huber: 0.7283, swd: 3.6244, ept: 181.2110
    Epoch [2/50], Test Losses: mse: 5.1927, mae: 1.0310, huber: 0.7538, swd: 3.3965, ept: 180.3861
      Epoch 2 composite train-obj: 0.854546
            Val objective improved 0.9768 → 0.7283, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.1399, mae: 1.0104, huber: 0.7359, swd: 3.6204, ept: 180.4320
    Epoch [3/50], Val Losses: mse: 4.5490, mae: 0.8743, huber: 0.6339, swd: 3.4191, ept: 183.5214
    Epoch [3/50], Test Losses: mse: 4.3741, mae: 0.8898, huber: 0.6400, swd: 3.1763, ept: 182.4064
      Epoch 3 composite train-obj: 0.735859
            Val objective improved 0.7283 → 0.6339, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6772, mae: 0.9258, huber: 0.6621, swd: 3.3992, ept: 182.0504
    Epoch [4/50], Val Losses: mse: 4.3931, mae: 0.7730, huber: 0.5590, swd: 3.5113, ept: 183.7403
    Epoch [4/50], Test Losses: mse: 4.2132, mae: 0.7904, huber: 0.5666, swd: 3.2269, ept: 183.1024
      Epoch 4 composite train-obj: 0.662061
            Val objective improved 0.6339 → 0.5590, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.2300, mae: 0.8503, huber: 0.5970, swd: 3.2084, ept: 183.6694
    Epoch [5/50], Val Losses: mse: 3.9974, mae: 0.7942, huber: 0.5535, swd: 3.1903, ept: 184.6838
    Epoch [5/50], Test Losses: mse: 3.7272, mae: 0.7987, huber: 0.5476, swd: 2.9440, ept: 184.0252
      Epoch 5 composite train-obj: 0.596960
            Val objective improved 0.5590 → 0.5535, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.9213, mae: 0.8045, huber: 0.5572, swd: 3.0556, ept: 184.6129
    Epoch [6/50], Val Losses: mse: 4.0044, mae: 0.7526, huber: 0.5311, swd: 3.2250, ept: 185.3635
    Epoch [6/50], Test Losses: mse: 3.8092, mae: 0.7752, huber: 0.5414, swd: 3.0032, ept: 184.5376
      Epoch 6 composite train-obj: 0.557198
            Val objective improved 0.5535 → 0.5311, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.7488, mae: 0.7704, huber: 0.5299, swd: 2.9708, ept: 185.3022
    Epoch [7/50], Val Losses: mse: 3.7557, mae: 0.6734, huber: 0.4742, swd: 3.0727, ept: 184.9359
    Epoch [7/50], Test Losses: mse: 3.5900, mae: 0.7008, huber: 0.4900, swd: 2.8657, ept: 184.4606
      Epoch 7 composite train-obj: 0.529904
            Val objective improved 0.5311 → 0.4742, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.6378, mae: 0.7505, huber: 0.5144, swd: 2.8996, ept: 185.6411
    Epoch [8/50], Val Losses: mse: 3.4050, mae: 0.6330, huber: 0.4385, swd: 2.9062, ept: 187.2568
    Epoch [8/50], Test Losses: mse: 3.2379, mae: 0.6447, huber: 0.4428, swd: 2.7072, ept: 186.0617
      Epoch 8 composite train-obj: 0.514360
            Val objective improved 0.4742 → 0.4385, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.5245, mae: 0.7291, huber: 0.4977, swd: 2.8380, ept: 186.0388
    Epoch [9/50], Val Losses: mse: 3.4657, mae: 0.6270, huber: 0.4371, swd: 2.9111, ept: 186.1892
    Epoch [9/50], Test Losses: mse: 3.2620, mae: 0.6453, huber: 0.4440, swd: 2.7027, ept: 185.6890
      Epoch 9 composite train-obj: 0.497667
            Val objective improved 0.4385 → 0.4371, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 3.4548, mae: 0.7135, huber: 0.4857, swd: 2.7987, ept: 186.2184
    Epoch [10/50], Val Losses: mse: 3.3468, mae: 0.6345, huber: 0.4396, swd: 2.8642, ept: 187.6494
    Epoch [10/50], Test Losses: mse: 3.1736, mae: 0.6493, huber: 0.4447, swd: 2.6626, ept: 186.2400
      Epoch 10 composite train-obj: 0.485698
            No improvement (0.4396), counter 1/5
    Epoch [11/50], Train Losses: mse: 3.3520, mae: 0.6983, huber: 0.4732, swd: 2.7336, ept: 186.5115
    Epoch [11/50], Val Losses: mse: 3.3250, mae: 0.5965, huber: 0.4171, swd: 2.8707, ept: 187.1084
    Epoch [11/50], Test Losses: mse: 3.1262, mae: 0.6133, huber: 0.4234, swd: 2.6767, ept: 186.2010
      Epoch 11 composite train-obj: 0.473234
            Val objective improved 0.4371 → 0.4171, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 3.2996, mae: 0.6882, huber: 0.4652, swd: 2.6982, ept: 186.7059
    Epoch [12/50], Val Losses: mse: 3.2651, mae: 0.6217, huber: 0.4224, swd: 2.7975, ept: 187.6016
    Epoch [12/50], Test Losses: mse: 3.1725, mae: 0.6506, huber: 0.4418, swd: 2.6351, ept: 186.3409
      Epoch 12 composite train-obj: 0.465247
            No improvement (0.4224), counter 1/5
    Epoch [13/50], Train Losses: mse: 3.2266, mae: 0.6759, huber: 0.4548, swd: 2.6578, ept: 186.9038
    Epoch [13/50], Val Losses: mse: 3.1383, mae: 0.5709, huber: 0.3953, swd: 2.7378, ept: 187.5848
    Epoch [13/50], Test Losses: mse: 2.9326, mae: 0.5806, huber: 0.3951, swd: 2.5226, ept: 186.7231
      Epoch 13 composite train-obj: 0.454831
            Val objective improved 0.4171 → 0.3953, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 3.1765, mae: 0.6679, huber: 0.4480, swd: 2.6133, ept: 187.0708
    Epoch [14/50], Val Losses: mse: 3.0176, mae: 0.5874, huber: 0.3925, swd: 2.6757, ept: 188.8590
    Epoch [14/50], Test Losses: mse: 2.8598, mae: 0.6037, huber: 0.3977, swd: 2.4770, ept: 187.3695
      Epoch 14 composite train-obj: 0.448042
            Val objective improved 0.3953 → 0.3925, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 3.0971, mae: 0.6513, huber: 0.4353, swd: 2.5792, ept: 187.2629
    Epoch [15/50], Val Losses: mse: 2.9620, mae: 0.6060, huber: 0.3951, swd: 2.6276, ept: 189.1846
    Epoch [15/50], Test Losses: mse: 2.8077, mae: 0.6245, huber: 0.4001, swd: 2.4380, ept: 188.0385
      Epoch 15 composite train-obj: 0.435330
            No improvement (0.3951), counter 1/5
    Epoch [16/50], Train Losses: mse: 3.0272, mae: 0.6394, huber: 0.4259, swd: 2.5254, ept: 187.4808
    Epoch [16/50], Val Losses: mse: 2.9745, mae: 0.5591, huber: 0.3780, swd: 2.5750, ept: 188.5974
    Epoch [16/50], Test Losses: mse: 2.8191, mae: 0.5682, huber: 0.3800, swd: 2.3845, ept: 187.2883
      Epoch 16 composite train-obj: 0.425907
            Val objective improved 0.3925 → 0.3780, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 2.9919, mae: 0.6340, huber: 0.4216, swd: 2.4976, ept: 187.6073
    Epoch [17/50], Val Losses: mse: 3.0764, mae: 0.5466, huber: 0.3732, swd: 2.6936, ept: 187.2780
    Epoch [17/50], Test Losses: mse: 2.9123, mae: 0.5736, huber: 0.3869, swd: 2.5121, ept: 186.2562
      Epoch 17 composite train-obj: 0.421590
            Val objective improved 0.3780 → 0.3732, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 2.9652, mae: 0.6286, huber: 0.4173, swd: 2.4779, ept: 187.7380
    Epoch [18/50], Val Losses: mse: 2.9321, mae: 0.5340, huber: 0.3640, swd: 2.5744, ept: 188.1483
    Epoch [18/50], Test Losses: mse: 2.7514, mae: 0.5481, huber: 0.3688, swd: 2.3938, ept: 187.2384
      Epoch 18 composite train-obj: 0.417350
            Val objective improved 0.3732 → 0.3640, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 2.9422, mae: 0.6205, huber: 0.4117, swd: 2.4568, ept: 187.8237
    Epoch [19/50], Val Losses: mse: 2.7942, mae: 0.5335, huber: 0.3564, swd: 2.4606, ept: 189.4095
    Epoch [19/50], Test Losses: mse: 2.6895, mae: 0.5554, huber: 0.3666, swd: 2.2724, ept: 188.1567
      Epoch 19 composite train-obj: 0.411726
            Val objective improved 0.3640 → 0.3564, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 2.8885, mae: 0.6142, huber: 0.4067, swd: 2.4074, ept: 187.9727
    Epoch [20/50], Val Losses: mse: 2.9822, mae: 0.6178, huber: 0.4104, swd: 2.6333, ept: 188.6290
    Epoch [20/50], Test Losses: mse: 2.8000, mae: 0.6319, huber: 0.4118, swd: 2.4325, ept: 187.1102
      Epoch 20 composite train-obj: 0.406735
            No improvement (0.4104), counter 1/5
    Epoch [21/50], Train Losses: mse: 2.8683, mae: 0.6090, huber: 0.4030, swd: 2.3910, ept: 187.9936
    Epoch [21/50], Val Losses: mse: 2.7280, mae: 0.5493, huber: 0.3631, swd: 2.3904, ept: 189.6126
    Epoch [21/50], Test Losses: mse: 2.5340, mae: 0.5565, huber: 0.3589, swd: 2.1974, ept: 188.3827
      Epoch 21 composite train-obj: 0.403027
            No improvement (0.3631), counter 2/5
    Epoch [22/50], Train Losses: mse: 2.8106, mae: 0.6022, huber: 0.3971, swd: 2.3431, ept: 188.1812
    Epoch [22/50], Val Losses: mse: 2.6646, mae: 0.5231, huber: 0.3558, swd: 2.3240, ept: 190.1307
    Epoch [22/50], Test Losses: mse: 2.5550, mae: 0.5456, huber: 0.3655, swd: 2.1536, ept: 188.9645
      Epoch 22 composite train-obj: 0.397124
            Val objective improved 0.3564 → 0.3558, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 2.8085, mae: 0.6003, huber: 0.3958, swd: 2.3411, ept: 188.1940
    Epoch [23/50], Val Losses: mse: 2.6501, mae: 0.5281, huber: 0.3456, swd: 2.3367, ept: 189.7375
    Epoch [23/50], Test Losses: mse: 2.4711, mae: 0.5346, huber: 0.3433, swd: 2.1674, ept: 188.7667
      Epoch 23 composite train-obj: 0.395790
            Val objective improved 0.3558 → 0.3456, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 2.7960, mae: 0.5940, huber: 0.3917, swd: 2.3306, ept: 188.2060
    Epoch [24/50], Val Losses: mse: 2.6371, mae: 0.5613, huber: 0.3619, swd: 2.3081, ept: 190.0349
    Epoch [24/50], Test Losses: mse: 2.4514, mae: 0.5703, huber: 0.3594, swd: 2.1161, ept: 189.0300
      Epoch 24 composite train-obj: 0.391662
            No improvement (0.3619), counter 1/5
    Epoch [25/50], Train Losses: mse: 2.7658, mae: 0.5930, huber: 0.3903, swd: 2.3079, ept: 188.3104
    Epoch [25/50], Val Losses: mse: 2.6820, mae: 0.5267, huber: 0.3530, swd: 2.3491, ept: 189.8553
    Epoch [25/50], Test Losses: mse: 2.5316, mae: 0.5357, huber: 0.3523, swd: 2.1465, ept: 188.6969
      Epoch 25 composite train-obj: 0.390336
            No improvement (0.3530), counter 2/5
    Epoch [26/50], Train Losses: mse: 2.7722, mae: 0.5873, huber: 0.3871, swd: 2.3093, ept: 188.2341
    Epoch [26/50], Val Losses: mse: 2.7560, mae: 0.5814, huber: 0.3787, swd: 2.3078, ept: 187.9161
    Epoch [26/50], Test Losses: mse: 2.5569, mae: 0.5939, huber: 0.3788, swd: 2.1299, ept: 187.6737
      Epoch 26 composite train-obj: 0.387105
            No improvement (0.3787), counter 3/5
    Epoch [27/50], Train Losses: mse: 2.7230, mae: 0.5892, huber: 0.3870, swd: 2.2665, ept: 188.3609
    Epoch [27/50], Val Losses: mse: 2.7151, mae: 0.6587, huber: 0.4103, swd: 2.3843, ept: 190.4191
    Epoch [27/50], Test Losses: mse: 2.5665, mae: 0.6837, huber: 0.4200, swd: 2.2064, ept: 189.5377
      Epoch 27 composite train-obj: 0.386981
            No improvement (0.4103), counter 4/5
    Epoch [28/50], Train Losses: mse: 2.6513, mae: 0.5826, huber: 0.3803, swd: 2.1895, ept: 188.5268
    Epoch [28/50], Val Losses: mse: 2.4530, mae: 0.4917, huber: 0.3233, swd: 2.1135, ept: 189.8380
    Epoch [28/50], Test Losses: mse: 2.3059, mae: 0.5037, huber: 0.3249, swd: 1.9346, ept: 188.6551
      Epoch 28 composite train-obj: 0.380349
            Val objective improved 0.3456 → 0.3233, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 2.5659, mae: 0.5642, huber: 0.3682, swd: 2.1254, ept: 188.6848
    Epoch [29/50], Val Losses: mse: 2.3452, mae: 0.4635, huber: 0.3093, swd: 2.0209, ept: 190.4855
    Epoch [29/50], Test Losses: mse: 2.1543, mae: 0.4641, huber: 0.3023, swd: 1.8480, ept: 189.5747
      Epoch 29 composite train-obj: 0.368203
            Val objective improved 0.3233 → 0.3093, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 2.5911, mae: 0.5687, huber: 0.3710, swd: 2.1448, ept: 188.6443
    Epoch [30/50], Val Losses: mse: 2.5601, mae: 0.4955, huber: 0.3231, swd: 2.1874, ept: 189.4307
    Epoch [30/50], Test Losses: mse: 2.3246, mae: 0.4951, huber: 0.3137, swd: 1.9621, ept: 189.0334
      Epoch 30 composite train-obj: 0.370980
            No improvement (0.3231), counter 1/5
    Epoch [31/50], Train Losses: mse: 2.5167, mae: 0.5568, huber: 0.3624, swd: 2.0739, ept: 188.6592
    Epoch [31/50], Val Losses: mse: 2.6447, mae: 0.5212, huber: 0.3463, swd: 2.2706, ept: 188.6786
    Epoch [31/50], Test Losses: mse: 2.4277, mae: 0.5283, huber: 0.3440, swd: 2.0553, ept: 188.3998
      Epoch 31 composite train-obj: 0.362398
            No improvement (0.3463), counter 2/5
    Epoch [32/50], Train Losses: mse: 2.4908, mae: 0.5543, huber: 0.3602, swd: 2.0567, ept: 188.8210
    Epoch [32/50], Val Losses: mse: 2.3799, mae: 0.4850, huber: 0.3190, swd: 2.0746, ept: 190.4929
    Epoch [32/50], Test Losses: mse: 2.1690, mae: 0.4825, huber: 0.3081, swd: 1.8794, ept: 189.4040
      Epoch 32 composite train-obj: 0.360231
            No improvement (0.3190), counter 3/5
    Epoch [33/50], Train Losses: mse: 2.4882, mae: 0.5547, huber: 0.3604, swd: 2.0374, ept: 188.8103
    Epoch [33/50], Val Losses: mse: 2.3660, mae: 0.4907, huber: 0.3170, swd: 2.0557, ept: 190.5720
    Epoch [33/50], Test Losses: mse: 2.1574, mae: 0.4868, huber: 0.3053, swd: 1.8614, ept: 189.4976
      Epoch 33 composite train-obj: 0.360368
            No improvement (0.3170), counter 4/5
    Epoch [34/50], Train Losses: mse: 2.4129, mae: 0.5442, huber: 0.3523, swd: 1.9748, ept: 188.9528
    Epoch [34/50], Val Losses: mse: 2.5162, mae: 0.5092, huber: 0.3325, swd: 2.1759, ept: 189.8153
    Epoch [34/50], Test Losses: mse: 2.3112, mae: 0.5113, huber: 0.3262, swd: 1.9707, ept: 188.2008
      Epoch 34 composite train-obj: 0.352301
    Epoch [34/50], Test Losses: mse: 2.1543, mae: 0.4641, huber: 0.3023, swd: 1.8480, ept: 189.5747
    Best round's Test MSE: 2.1543, MAE: 0.4641, SWD: 1.8480
    Best round's Validation MSE: 2.3452, MAE: 0.4635, SWD: 2.0209
    Best round's Test verification MSE : 2.1543, MAE: 0.4641, SWD: 1.8480
    Time taken: 255.12 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 16.6141, mae: 2.0481, huber: 1.7178, swd: 9.9056, ept: 157.6985
    Epoch [1/50], Val Losses: mse: 9.4899, mae: 1.1932, huber: 0.9365, swd: 6.9209, ept: 176.1459
    Epoch [1/50], Test Losses: mse: 8.7726, mae: 1.2411, huber: 0.9660, swd: 6.2798, ept: 174.6763
      Epoch 1 composite train-obj: 1.717754
            Val objective improved inf → 0.9365, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.0291, mae: 1.2048, huber: 0.9120, swd: 4.9605, ept: 176.5465
    Epoch [2/50], Val Losses: mse: 5.8160, mae: 0.9763, huber: 0.7263, swd: 4.0577, ept: 180.1348
    Epoch [2/50], Test Losses: mse: 5.5048, mae: 1.0086, huber: 0.7431, swd: 3.7835, ept: 179.7915
      Epoch 2 composite train-obj: 0.912021
            Val objective improved 0.9365 → 0.7263, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.4052, mae: 1.0467, huber: 0.7684, swd: 3.8254, ept: 180.0629
    Epoch [3/50], Val Losses: mse: 4.4992, mae: 0.8461, huber: 0.6151, swd: 3.3698, ept: 183.0738
    Epoch [3/50], Test Losses: mse: 4.3341, mae: 0.8783, huber: 0.6340, swd: 3.1799, ept: 182.5309
      Epoch 3 composite train-obj: 0.768433
            Val objective improved 0.7263 → 0.6151, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.5938, mae: 0.9329, huber: 0.6658, swd: 3.4511, ept: 182.3346
    Epoch [4/50], Val Losses: mse: 4.0258, mae: 0.7840, huber: 0.5399, swd: 3.2665, ept: 186.4754
    Epoch [4/50], Test Losses: mse: 3.9176, mae: 0.8180, huber: 0.5609, swd: 3.0737, ept: 184.8237
      Epoch 4 composite train-obj: 0.665828
            Val objective improved 0.6151 → 0.5399, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.0303, mae: 0.8315, huber: 0.5779, swd: 3.1716, ept: 184.0454
    Epoch [5/50], Val Losses: mse: 3.6529, mae: 0.7014, huber: 0.4877, swd: 3.0777, ept: 186.8387
    Epoch [5/50], Test Losses: mse: 3.5041, mae: 0.7275, huber: 0.5026, swd: 2.8966, ept: 185.3579
      Epoch 5 composite train-obj: 0.577898
            Val objective improved 0.5399 → 0.4877, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.7109, mae: 0.7784, huber: 0.5335, swd: 2.9962, ept: 185.0568
    Epoch [6/50], Val Losses: mse: 3.7308, mae: 0.6865, huber: 0.4652, swd: 3.1009, ept: 185.5397
    Epoch [6/50], Test Losses: mse: 3.5351, mae: 0.7199, huber: 0.4844, swd: 2.9076, ept: 185.0156
      Epoch 6 composite train-obj: 0.533501
            Val objective improved 0.4877 → 0.4652, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.6180, mae: 0.7537, huber: 0.5144, swd: 2.9396, ept: 185.5603
    Epoch [7/50], Val Losses: mse: 3.9155, mae: 0.7491, huber: 0.5089, swd: 3.2965, ept: 184.8647
    Epoch [7/50], Test Losses: mse: 3.7063, mae: 0.7873, huber: 0.5341, swd: 3.1028, ept: 184.4912
      Epoch 7 composite train-obj: 0.514353
            No improvement (0.5089), counter 1/5
    Epoch [8/50], Train Losses: mse: 3.4971, mae: 0.7319, huber: 0.4970, swd: 2.8799, ept: 185.9210
    Epoch [8/50], Val Losses: mse: 3.4450, mae: 0.6419, huber: 0.4345, swd: 2.9868, ept: 186.7903
    Epoch [8/50], Test Losses: mse: 3.2597, mae: 0.6650, huber: 0.4468, swd: 2.7923, ept: 185.7695
      Epoch 8 composite train-obj: 0.496988
            Val objective improved 0.4652 → 0.4345, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.3614, mae: 0.7059, huber: 0.4763, swd: 2.7976, ept: 186.3683
    Epoch [9/50], Val Losses: mse: 3.2483, mae: 0.6051, huber: 0.4075, swd: 2.8532, ept: 187.3229
    Epoch [9/50], Test Losses: mse: 3.0697, mae: 0.6240, huber: 0.4160, swd: 2.6586, ept: 186.2576
      Epoch 9 composite train-obj: 0.476321
            Val objective improved 0.4345 → 0.4075, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 3.2736, mae: 0.6872, huber: 0.4616, swd: 2.7490, ept: 186.6147
    Epoch [10/50], Val Losses: mse: 3.2419, mae: 0.5948, huber: 0.4072, swd: 2.8242, ept: 186.8599
    Epoch [10/50], Test Losses: mse: 3.1062, mae: 0.6229, huber: 0.4235, swd: 2.6418, ept: 186.2010
      Epoch 10 composite train-obj: 0.461590
            Val objective improved 0.4075 → 0.4072, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 3.1919, mae: 0.6713, huber: 0.4488, swd: 2.6926, ept: 186.8480
    Epoch [11/50], Val Losses: mse: 3.2274, mae: 0.5785, huber: 0.3930, swd: 2.8208, ept: 186.7639
    Epoch [11/50], Test Losses: mse: 3.0724, mae: 0.6041, huber: 0.4069, swd: 2.6268, ept: 185.9611
      Epoch 11 composite train-obj: 0.448759
            Val objective improved 0.4072 → 0.3930, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 3.1552, mae: 0.6594, huber: 0.4393, swd: 2.6653, ept: 187.0559
    Epoch [12/50], Val Losses: mse: 3.0429, mae: 0.6032, huber: 0.4007, swd: 2.7364, ept: 188.9314
    Epoch [12/50], Test Losses: mse: 2.9811, mae: 0.6351, huber: 0.4207, swd: 2.5632, ept: 187.2450
      Epoch 12 composite train-obj: 0.439297
            No improvement (0.4007), counter 1/5
    Epoch [13/50], Train Losses: mse: 3.0991, mae: 0.6459, huber: 0.4301, swd: 2.6399, ept: 187.1340
    Epoch [13/50], Val Losses: mse: 2.9822, mae: 0.5395, huber: 0.3667, swd: 2.6880, ept: 188.9392
    Epoch [13/50], Test Losses: mse: 2.9195, mae: 0.5636, huber: 0.3814, swd: 2.5086, ept: 186.8688
      Epoch 13 composite train-obj: 0.430139
            Val objective improved 0.3930 → 0.3667, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 3.0538, mae: 0.6331, huber: 0.4204, swd: 2.6007, ept: 187.3457
    Epoch [14/50], Val Losses: mse: 3.0144, mae: 0.5958, huber: 0.3961, swd: 2.6767, ept: 188.7049
    Epoch [14/50], Test Losses: mse: 2.9089, mae: 0.6143, huber: 0.4062, swd: 2.4891, ept: 187.2781
      Epoch 14 composite train-obj: 0.420397
            No improvement (0.3961), counter 1/5
    Epoch [15/50], Train Losses: mse: 3.0267, mae: 0.6253, huber: 0.4152, swd: 2.5877, ept: 187.4688
    Epoch [15/50], Val Losses: mse: 3.0893, mae: 0.5593, huber: 0.3753, swd: 2.7340, ept: 187.6731
    Epoch [15/50], Test Losses: mse: 2.9319, mae: 0.5758, huber: 0.3831, swd: 2.5391, ept: 186.8857
      Epoch 15 composite train-obj: 0.415225
            No improvement (0.3753), counter 2/5
    Epoch [16/50], Train Losses: mse: 3.0092, mae: 0.6170, huber: 0.4096, swd: 2.5720, ept: 187.5011
    Epoch [16/50], Val Losses: mse: 2.9324, mae: 0.5297, huber: 0.3567, swd: 2.6501, ept: 188.8934
    Epoch [16/50], Test Losses: mse: 2.7921, mae: 0.5507, huber: 0.3669, swd: 2.4713, ept: 187.3190
      Epoch 16 composite train-obj: 0.409561
            Val objective improved 0.3667 → 0.3567, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 2.9769, mae: 0.6131, huber: 0.4067, swd: 2.5462, ept: 187.6774
    Epoch [17/50], Val Losses: mse: 2.9741, mae: 0.5148, huber: 0.3543, swd: 2.6291, ept: 187.4649
    Epoch [17/50], Test Losses: mse: 2.8711, mae: 0.5371, huber: 0.3652, swd: 2.4379, ept: 186.8318
      Epoch 17 composite train-obj: 0.406749
            Val objective improved 0.3567 → 0.3543, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 2.9525, mae: 0.6045, huber: 0.4007, swd: 2.5271, ept: 187.7159
    Epoch [18/50], Val Losses: mse: 3.0748, mae: 0.6000, huber: 0.3867, swd: 2.7129, ept: 187.8975
    Epoch [18/50], Test Losses: mse: 2.9574, mae: 0.6229, huber: 0.3985, swd: 2.5211, ept: 186.8519
      Epoch 18 composite train-obj: 0.400684
            No improvement (0.3867), counter 1/5
    Epoch [19/50], Train Losses: mse: 2.9689, mae: 0.6021, huber: 0.3998, swd: 2.5396, ept: 187.6886
    Epoch [19/50], Val Losses: mse: 2.8996, mae: 0.5254, huber: 0.3582, swd: 2.6200, ept: 189.1136
    Epoch [19/50], Test Losses: mse: 2.7829, mae: 0.5466, huber: 0.3688, swd: 2.4361, ept: 187.7484
      Epoch 19 composite train-obj: 0.399824
            No improvement (0.3582), counter 2/5
    Epoch [20/50], Train Losses: mse: 2.9212, mae: 0.5959, huber: 0.3954, swd: 2.5088, ept: 187.8440
    Epoch [20/50], Val Losses: mse: 2.9112, mae: 0.5243, huber: 0.3535, swd: 2.6007, ept: 188.7056
    Epoch [20/50], Test Losses: mse: 2.7978, mae: 0.5472, huber: 0.3661, swd: 2.4216, ept: 187.4043
      Epoch 20 composite train-obj: 0.395435
            Val objective improved 0.3543 → 0.3535, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 2.9154, mae: 0.5933, huber: 0.3927, swd: 2.4983, ept: 187.8762
    Epoch [21/50], Val Losses: mse: 3.0425, mae: 0.5389, huber: 0.3595, swd: 2.6714, ept: 187.3936
    Epoch [21/50], Test Losses: mse: 2.8381, mae: 0.5551, huber: 0.3652, swd: 2.4741, ept: 187.1340
      Epoch 21 composite train-obj: 0.392744
            No improvement (0.3595), counter 1/5
    Epoch [22/50], Train Losses: mse: 2.8913, mae: 0.5862, huber: 0.3880, swd: 2.4853, ept: 187.9349
    Epoch [22/50], Val Losses: mse: 2.8073, mae: 0.5178, huber: 0.3429, swd: 2.5524, ept: 189.6686
    Epoch [22/50], Test Losses: mse: 2.6326, mae: 0.5262, huber: 0.3429, swd: 2.3692, ept: 187.8689
      Epoch 22 composite train-obj: 0.388039
            Val objective improved 0.3535 → 0.3429, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 2.8930, mae: 0.5868, huber: 0.3885, swd: 2.4932, ept: 188.0199
    Epoch [23/50], Val Losses: mse: 2.8499, mae: 0.5438, huber: 0.3598, swd: 2.5616, ept: 189.1543
    Epoch [23/50], Test Losses: mse: 2.6873, mae: 0.5620, huber: 0.3653, swd: 2.3750, ept: 187.9160
      Epoch 23 composite train-obj: 0.388454
            No improvement (0.3598), counter 1/5
    Epoch [24/50], Train Losses: mse: 2.8477, mae: 0.5780, huber: 0.3820, swd: 2.4526, ept: 188.0846
    Epoch [24/50], Val Losses: mse: 2.8246, mae: 0.4895, huber: 0.3359, swd: 2.5079, ept: 188.3976
    Epoch [24/50], Test Losses: mse: 2.6221, mae: 0.4941, huber: 0.3345, swd: 2.3096, ept: 187.8066
      Epoch 24 composite train-obj: 0.381976
            Val objective improved 0.3429 → 0.3359, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 2.8428, mae: 0.5771, huber: 0.3811, swd: 2.4418, ept: 188.1049
    Epoch [25/50], Val Losses: mse: 2.7778, mae: 0.5495, huber: 0.3579, swd: 2.5072, ept: 189.9141
    Epoch [25/50], Test Losses: mse: 2.6644, mae: 0.5678, huber: 0.3633, swd: 2.3174, ept: 188.5813
      Epoch 25 composite train-obj: 0.381134
            No improvement (0.3579), counter 1/5
    Epoch [26/50], Train Losses: mse: 2.8297, mae: 0.5746, huber: 0.3796, swd: 2.4377, ept: 188.1833
    Epoch [26/50], Val Losses: mse: 2.9333, mae: 0.5132, huber: 0.3507, swd: 2.5934, ept: 188.1639
    Epoch [26/50], Test Losses: mse: 2.7106, mae: 0.5275, huber: 0.3550, swd: 2.3954, ept: 187.6403
      Epoch 26 composite train-obj: 0.379570
            No improvement (0.3507), counter 2/5
    Epoch [27/50], Train Losses: mse: 2.8320, mae: 0.5797, huber: 0.3822, swd: 2.4358, ept: 188.2402
    Epoch [27/50], Val Losses: mse: 2.8047, mae: 0.5117, huber: 0.3401, swd: 2.5192, ept: 188.9825
    Epoch [27/50], Test Losses: mse: 2.6365, mae: 0.5211, huber: 0.3419, swd: 2.3254, ept: 187.8230
      Epoch 27 composite train-obj: 0.382182
            No improvement (0.3401), counter 3/5
    Epoch [28/50], Train Losses: mse: 2.8098, mae: 0.5708, huber: 0.3763, swd: 2.4151, ept: 188.2551
    Epoch [28/50], Val Losses: mse: 2.8550, mae: 0.5503, huber: 0.3564, swd: 2.5092, ept: 188.3727
    Epoch [28/50], Test Losses: mse: 2.6797, mae: 0.5627, huber: 0.3621, swd: 2.3218, ept: 187.6064
      Epoch 28 composite train-obj: 0.376272
            No improvement (0.3564), counter 4/5
    Epoch [29/50], Train Losses: mse: 2.7738, mae: 0.5678, huber: 0.3739, swd: 2.3966, ept: 188.3569
    Epoch [29/50], Val Losses: mse: 2.9142, mae: 0.5428, huber: 0.3546, swd: 2.6361, ept: 188.7678
    Epoch [29/50], Test Losses: mse: 2.6871, mae: 0.5503, huber: 0.3524, swd: 2.4288, ept: 187.5409
      Epoch 29 composite train-obj: 0.373870
    Epoch [29/50], Test Losses: mse: 2.6221, mae: 0.4941, huber: 0.3345, swd: 2.3096, ept: 187.8066
    Best round's Test MSE: 2.6221, MAE: 0.4941, SWD: 2.3096
    Best round's Validation MSE: 2.8246, MAE: 0.4895, SWD: 2.5079
    Best round's Test verification MSE : 2.6221, MAE: 0.4941, SWD: 2.3096
    Time taken: 217.03 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 16.7198, mae: 2.0526, huber: 1.7221, swd: 8.8699, ept: 158.3574
    Epoch [1/50], Val Losses: mse: 9.2140, mae: 1.3258, huber: 1.0225, swd: 6.1592, ept: 176.8385
    Epoch [1/50], Test Losses: mse: 8.7549, mae: 1.3806, huber: 1.0634, swd: 5.6004, ept: 174.8775
      Epoch 1 composite train-obj: 1.722116
            Val objective improved inf → 1.0225, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.8577, mae: 1.2359, huber: 0.9453, swd: 5.2134, ept: 176.7167
    Epoch [2/50], Val Losses: mse: 5.9067, mae: 1.0249, huber: 0.7692, swd: 3.7899, ept: 180.6255
    Epoch [2/50], Test Losses: mse: 5.5398, mae: 1.0536, huber: 0.7842, swd: 3.5006, ept: 180.4212
      Epoch 2 composite train-obj: 0.945305
            Val objective improved 1.0225 → 0.7692, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.5305, mae: 1.0658, huber: 0.7834, swd: 3.5927, ept: 180.8366
    Epoch [3/50], Val Losses: mse: 4.8618, mae: 0.8737, huber: 0.6434, swd: 3.2737, ept: 182.2172
    Epoch [3/50], Test Losses: mse: 4.7074, mae: 0.9027, huber: 0.6599, swd: 3.0520, ept: 181.7393
      Epoch 3 composite train-obj: 0.783446
            Val objective improved 0.7692 → 0.6434, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.8335, mae: 0.9769, huber: 0.7057, swd: 3.1314, ept: 182.3753
    Epoch [4/50], Val Losses: mse: 4.7595, mae: 0.8766, huber: 0.6335, swd: 3.1655, ept: 183.0343
    Epoch [4/50], Test Losses: mse: 4.7261, mae: 0.9193, huber: 0.6632, swd: 2.9582, ept: 182.1763
      Epoch 4 composite train-obj: 0.705676
            Val objective improved 0.6434 → 0.6335, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.5259, mae: 0.9227, huber: 0.6605, swd: 2.9998, ept: 183.3848
    Epoch [5/50], Val Losses: mse: 4.3328, mae: 0.8635, huber: 0.6105, swd: 3.1418, ept: 186.5376
    Epoch [5/50], Test Losses: mse: 4.4622, mae: 0.9177, huber: 0.6511, swd: 2.9886, ept: 184.4531
      Epoch 5 composite train-obj: 0.660481
            Val objective improved 0.6335 → 0.6105, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.1487, mae: 0.8672, huber: 0.6109, swd: 2.8684, ept: 184.4159
    Epoch [6/50], Val Losses: mse: 3.7832, mae: 0.7713, huber: 0.5408, swd: 2.8415, ept: 185.7046
    Epoch [6/50], Test Losses: mse: 3.6675, mae: 0.8005, huber: 0.5590, swd: 2.6866, ept: 184.7755
      Epoch 6 composite train-obj: 0.610921
            Val objective improved 0.6105 → 0.5408, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.8703, mae: 0.8156, huber: 0.5676, swd: 2.7490, ept: 185.1611
    Epoch [7/50], Val Losses: mse: 3.5401, mae: 0.6843, huber: 0.4716, swd: 2.7474, ept: 186.0445
    Epoch [7/50], Test Losses: mse: 3.4465, mae: 0.7163, huber: 0.4931, swd: 2.6045, ept: 185.1329
      Epoch 7 composite train-obj: 0.567586
            Val objective improved 0.5408 → 0.4716, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.6749, mae: 0.7796, huber: 0.5372, swd: 2.6787, ept: 185.6498
    Epoch [8/50], Val Losses: mse: 3.4348, mae: 0.6879, huber: 0.4681, swd: 2.7705, ept: 187.4299
    Epoch [8/50], Test Losses: mse: 3.3688, mae: 0.7250, huber: 0.4946, swd: 2.6273, ept: 185.8961
      Epoch 8 composite train-obj: 0.537212
            Val objective improved 0.4716 → 0.4681, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.5198, mae: 0.7499, huber: 0.5128, swd: 2.6027, ept: 186.1669
    Epoch [9/50], Val Losses: mse: 3.5286, mae: 0.7479, huber: 0.5055, swd: 2.7610, ept: 186.8833
    Epoch [9/50], Test Losses: mse: 3.4389, mae: 0.7806, huber: 0.5291, swd: 2.6254, ept: 185.8229
      Epoch 9 composite train-obj: 0.512800
            No improvement (0.5055), counter 1/5
    Epoch [10/50], Train Losses: mse: 3.3636, mae: 0.7194, huber: 0.4881, swd: 2.5132, ept: 186.6909
    Epoch [10/50], Val Losses: mse: 3.3379, mae: 0.6307, huber: 0.4284, swd: 2.5888, ept: 186.8104
    Epoch [10/50], Test Losses: mse: 3.1940, mae: 0.6589, huber: 0.4442, swd: 2.4259, ept: 186.0606
      Epoch 10 composite train-obj: 0.488054
            Val objective improved 0.4681 → 0.4284, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 3.3041, mae: 0.7039, huber: 0.4762, swd: 2.4820, ept: 186.8592
    Epoch [11/50], Val Losses: mse: 3.1700, mae: 0.6526, huber: 0.4319, swd: 2.5567, ept: 188.5231
    Epoch [11/50], Test Losses: mse: 3.0523, mae: 0.6739, huber: 0.4436, swd: 2.4212, ept: 186.9292
      Epoch 11 composite train-obj: 0.476196
            No improvement (0.4319), counter 1/5
    Epoch [12/50], Train Losses: mse: 3.1702, mae: 0.6840, huber: 0.4590, swd: 2.4069, ept: 187.2243
    Epoch [12/50], Val Losses: mse: 3.1497, mae: 0.6084, huber: 0.4062, swd: 2.5326, ept: 187.8705
    Epoch [12/50], Test Losses: mse: 3.0136, mae: 0.6299, huber: 0.4175, swd: 2.3618, ept: 186.4669
      Epoch 12 composite train-obj: 0.459005
            Val objective improved 0.4284 → 0.4062, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 3.1084, mae: 0.6689, huber: 0.4480, swd: 2.3596, ept: 187.3645
    Epoch [13/50], Val Losses: mse: 3.2781, mae: 0.6556, huber: 0.4406, swd: 2.6224, ept: 187.9537
    Epoch [13/50], Test Losses: mse: 3.1340, mae: 0.6744, huber: 0.4505, swd: 2.4484, ept: 185.9280
      Epoch 13 composite train-obj: 0.447992
            No improvement (0.4406), counter 1/5
    Epoch [14/50], Train Losses: mse: 3.0293, mae: 0.6532, huber: 0.4356, swd: 2.3097, ept: 187.5934
    Epoch [14/50], Val Losses: mse: 3.3276, mae: 0.6393, huber: 0.4249, swd: 2.6656, ept: 187.2145
    Epoch [14/50], Test Losses: mse: 3.1612, mae: 0.6654, huber: 0.4402, swd: 2.4940, ept: 185.6617
      Epoch 14 composite train-obj: 0.435597
            No improvement (0.4249), counter 2/5
    Epoch [15/50], Train Losses: mse: 2.9590, mae: 0.6448, huber: 0.4277, swd: 2.2645, ept: 187.8048
    Epoch [15/50], Val Losses: mse: 2.9410, mae: 0.6045, huber: 0.3972, swd: 2.3126, ept: 188.6849
    Epoch [15/50], Test Losses: mse: 2.7806, mae: 0.6269, huber: 0.4077, swd: 2.1384, ept: 187.9531
      Epoch 15 composite train-obj: 0.427689
            Val objective improved 0.4062 → 0.3972, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 2.8937, mae: 0.6260, huber: 0.4131, swd: 2.2234, ept: 187.9555
    Epoch [16/50], Val Losses: mse: 3.0777, mae: 0.6082, huber: 0.4119, swd: 2.5056, ept: 189.0190
    Epoch [16/50], Test Losses: mse: 2.9870, mae: 0.6372, huber: 0.4292, swd: 2.3458, ept: 186.7649
      Epoch 16 composite train-obj: 0.413077
            No improvement (0.4119), counter 1/5
    Epoch [17/50], Train Losses: mse: 2.8452, mae: 0.6151, huber: 0.4050, swd: 2.1868, ept: 187.9808
    Epoch [17/50], Val Losses: mse: 2.9913, mae: 0.6627, huber: 0.4307, swd: 2.3940, ept: 188.7017
    Epoch [17/50], Test Losses: mse: 2.8407, mae: 0.6808, huber: 0.4395, swd: 2.2146, ept: 187.5762
      Epoch 17 composite train-obj: 0.404957
            No improvement (0.4307), counter 2/5
    Epoch [18/50], Train Losses: mse: 2.7712, mae: 0.6006, huber: 0.3946, swd: 2.1270, ept: 188.1312
    Epoch [18/50], Val Losses: mse: 3.2764, mae: 0.6334, huber: 0.4183, swd: 2.5849, ept: 186.9991
    Epoch [18/50], Test Losses: mse: 3.0560, mae: 0.6550, huber: 0.4281, swd: 2.3908, ept: 185.8565
      Epoch 18 composite train-obj: 0.394557
            No improvement (0.4183), counter 3/5
    Epoch [19/50], Train Losses: mse: 2.7184, mae: 0.5918, huber: 0.3872, swd: 2.0953, ept: 188.2896
    Epoch [19/50], Val Losses: mse: 3.3967, mae: 0.6424, huber: 0.4340, swd: 2.6882, ept: 186.8924
    Epoch [19/50], Test Losses: mse: 3.1982, mae: 0.6608, huber: 0.4421, swd: 2.5068, ept: 185.6048
      Epoch 19 composite train-obj: 0.387207
            No improvement (0.4340), counter 4/5
    Epoch [20/50], Train Losses: mse: 2.6873, mae: 0.5878, huber: 0.3838, swd: 2.0733, ept: 188.3698
    Epoch [20/50], Val Losses: mse: 3.0139, mae: 0.5868, huber: 0.3881, swd: 2.4600, ept: 188.4101
    Epoch [20/50], Test Losses: mse: 2.8092, mae: 0.5953, huber: 0.3884, swd: 2.2720, ept: 186.8336
      Epoch 20 composite train-obj: 0.383827
            Val objective improved 0.3972 → 0.3881, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 2.5694, mae: 0.5726, huber: 0.3713, swd: 1.9656, ept: 188.5748
    Epoch [21/50], Val Losses: mse: 2.9595, mae: 0.6057, huber: 0.4019, swd: 2.3979, ept: 189.2935
    Epoch [21/50], Test Losses: mse: 2.7868, mae: 0.6220, huber: 0.4068, swd: 2.2153, ept: 187.1845
      Epoch 21 composite train-obj: 0.371263
            No improvement (0.4019), counter 1/5
    Epoch [22/50], Train Losses: mse: 2.5387, mae: 0.5660, huber: 0.3665, swd: 1.9413, ept: 188.6537
    Epoch [22/50], Val Losses: mse: 2.9903, mae: 0.6431, huber: 0.4195, swd: 2.4028, ept: 189.0244
    Epoch [22/50], Test Losses: mse: 2.8005, mae: 0.6580, huber: 0.4239, swd: 2.2267, ept: 187.1320
      Epoch 22 composite train-obj: 0.366508
            No improvement (0.4195), counter 2/5
    Epoch [23/50], Train Losses: mse: 2.4694, mae: 0.5575, huber: 0.3597, swd: 1.8772, ept: 188.8027
    Epoch [23/50], Val Losses: mse: 3.0877, mae: 0.6845, huber: 0.4479, swd: 2.5089, ept: 189.0681
    Epoch [23/50], Test Losses: mse: 2.8914, mae: 0.7006, huber: 0.4543, swd: 2.3440, ept: 187.3380
      Epoch 23 composite train-obj: 0.359730
            No improvement (0.4479), counter 3/5
    Epoch [24/50], Train Losses: mse: 2.4470, mae: 0.5551, huber: 0.3575, swd: 1.8616, ept: 188.7919
    Epoch [24/50], Val Losses: mse: 3.1881, mae: 0.6557, huber: 0.4343, swd: 2.6156, ept: 188.4921
    Epoch [24/50], Test Losses: mse: 2.9925, mae: 0.6732, huber: 0.4409, swd: 2.4523, ept: 186.6991
      Epoch 24 composite train-obj: 0.357522
            No improvement (0.4343), counter 4/5
    Epoch [25/50], Train Losses: mse: 2.4404, mae: 0.5521, huber: 0.3554, swd: 1.8607, ept: 188.7879
    Epoch [25/50], Val Losses: mse: 3.2958, mae: 0.6535, huber: 0.4437, swd: 2.6737, ept: 188.4280
    Epoch [25/50], Test Losses: mse: 3.1145, mae: 0.6799, huber: 0.4570, swd: 2.5137, ept: 186.8580
      Epoch 25 composite train-obj: 0.355427
    Epoch [25/50], Test Losses: mse: 2.8092, mae: 0.5953, huber: 0.3884, swd: 2.2720, ept: 186.8336
    Best round's Test MSE: 2.8092, MAE: 0.5953, SWD: 2.2720
    Best round's Validation MSE: 3.0139, MAE: 0.5868, SWD: 2.4600
    Best round's Test verification MSE : 2.8092, MAE: 0.5953, SWD: 2.2720
    Time taken: 212.29 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_rossler_seq96_pred196_20250513_1304)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 2.5285 ± 0.2754
      mae: 0.5179 ± 0.0561
      huber: 0.3417 ± 0.0355
      swd: 2.1432 ± 0.2093
      ept: 188.0716 ± 1.1346
      count: 39.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 2.7279 ± 0.2814
      mae: 0.5133 ± 0.0531
      huber: 0.3444 ± 0.0327
      swd: 2.3296 ± 0.2192
      ept: 189.0978 ± 0.9813
      count: 39.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 684.49 seconds
    
    Experiment complete: TimeMixer_rossler_seq96_pred196_20250513_1304
    Model: TimeMixer
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['rossler']['channels'],
    enc_in=data_mgr.datasets['rossler']['channels'],
    dec_in=data_mgr.datasets['rossler']['channels'],
    c_out=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 26.1457, mae: 2.7262, huber: 2.3707, swd: 13.9791, ept: 229.0973
    Epoch [1/50], Val Losses: mse: 19.2632, mae: 2.3370, huber: 1.9792, swd: 13.2302, ept: 257.4807
    Epoch [1/50], Test Losses: mse: 17.3000, mae: 2.3493, huber: 1.9811, swd: 10.7780, ept: 252.4893
      Epoch 1 composite train-obj: 2.370712
            Val objective improved inf → 1.9792, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 15.5450, mae: 1.8360, huber: 1.5116, swd: 10.7813, ept: 272.4168
    Epoch [2/50], Val Losses: mse: 13.2323, mae: 1.9618, huber: 1.6339, swd: 8.6212, ept: 264.8447
    Epoch [2/50], Test Losses: mse: 11.6700, mae: 1.9073, huber: 1.5713, swd: 7.1846, ept: 266.6964
      Epoch 2 composite train-obj: 1.511615
            Val objective improved 1.9792 → 1.6339, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.2883, mae: 1.6136, huber: 1.2967, swd: 7.4682, ept: 279.5852
    Epoch [3/50], Val Losses: mse: 10.5310, mae: 1.5991, huber: 1.2753, swd: 7.0294, ept: 286.2066
    Epoch [3/50], Test Losses: mse: 10.5746, mae: 1.6294, huber: 1.2967, swd: 6.3015, ept: 284.1903
      Epoch 3 composite train-obj: 1.296733
            Val objective improved 1.6339 → 1.2753, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.0075, mae: 1.3904, huber: 1.0820, swd: 5.2052, ept: 286.7195
    Epoch [4/50], Val Losses: mse: 7.1512, mae: 1.4312, huber: 1.0989, swd: 4.8961, ept: 297.5395
    Epoch [4/50], Test Losses: mse: 6.9447, mae: 1.4565, huber: 1.1146, swd: 4.5114, ept: 292.6460
      Epoch 4 composite train-obj: 1.082048
            Val objective improved 1.2753 → 1.0989, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.6385, mae: 1.2651, huber: 0.9646, swd: 4.3119, ept: 291.7531
    Epoch [5/50], Val Losses: mse: 5.8111, mae: 1.2476, huber: 0.9320, swd: 4.1112, ept: 301.5266
    Epoch [5/50], Test Losses: mse: 5.7144, mae: 1.2720, huber: 0.9473, swd: 4.0325, ept: 297.7113
      Epoch 5 composite train-obj: 0.964596
            Val objective improved 1.0989 → 0.9320, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.8824, mae: 1.1595, huber: 0.8691, swd: 3.8664, ept: 296.2028
    Epoch [6/50], Val Losses: mse: 5.5236, mae: 0.9953, huber: 0.7455, swd: 3.9179, ept: 297.4461
    Epoch [6/50], Test Losses: mse: 5.3685, mae: 1.0116, huber: 0.7550, swd: 3.6706, ept: 297.7663
      Epoch 6 composite train-obj: 0.869106
            Val objective improved 0.9320 → 0.7455, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.3953, mae: 1.0803, huber: 0.7975, swd: 3.6343, ept: 299.5060
    Epoch [7/50], Val Losses: mse: 5.5400, mae: 1.0579, huber: 0.7858, swd: 3.8140, ept: 298.6531
    Epoch [7/50], Test Losses: mse: 5.7852, mae: 1.0957, huber: 0.8186, swd: 3.7368, ept: 297.7428
      Epoch 7 composite train-obj: 0.797501
            No improvement (0.7858), counter 1/5
    Epoch [8/50], Train Losses: mse: 5.1944, mae: 1.0473, huber: 0.7694, swd: 3.5252, ept: 301.1201
    Epoch [8/50], Val Losses: mse: 5.9778, mae: 1.1088, huber: 0.8367, swd: 4.4440, ept: 299.4218
    Epoch [8/50], Test Losses: mse: 5.8619, mae: 1.1229, huber: 0.8448, swd: 4.1477, ept: 296.4450
      Epoch 8 composite train-obj: 0.769362
            No improvement (0.8367), counter 2/5
    Epoch [9/50], Train Losses: mse: 5.0891, mae: 1.0320, huber: 0.7563, swd: 3.4785, ept: 301.9752
    Epoch [9/50], Val Losses: mse: 4.8360, mae: 0.9596, huber: 0.6972, swd: 3.5096, ept: 301.8139
    Epoch [9/50], Test Losses: mse: 4.5570, mae: 0.9663, huber: 0.6964, swd: 3.2687, ept: 301.9685
      Epoch 9 composite train-obj: 0.756290
            Val objective improved 0.7455 → 0.6972, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 4.9697, mae: 1.0082, huber: 0.7365, swd: 3.4217, ept: 302.9407
    Epoch [10/50], Val Losses: mse: 4.6962, mae: 0.9076, huber: 0.6606, swd: 3.3709, ept: 303.3388
    Epoch [10/50], Test Losses: mse: 4.6090, mae: 0.9209, huber: 0.6666, swd: 3.1760, ept: 303.2222
      Epoch 10 composite train-obj: 0.736515
            Val objective improved 0.6972 → 0.6606, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 4.8060, mae: 0.9804, huber: 0.7120, swd: 3.3413, ept: 304.3704
    Epoch [11/50], Val Losses: mse: 5.4598, mae: 1.1526, huber: 0.8606, swd: 3.5723, ept: 307.4972
    Epoch [11/50], Test Losses: mse: 5.3510, mae: 1.1627, huber: 0.8647, swd: 3.3785, ept: 304.4831
      Epoch 11 composite train-obj: 0.711985
            No improvement (0.8606), counter 1/5
    Epoch [12/50], Train Losses: mse: 4.7286, mae: 0.9659, huber: 0.7000, swd: 3.2814, ept: 305.1797
    Epoch [12/50], Val Losses: mse: 4.4544, mae: 0.8676, huber: 0.6290, swd: 3.2829, ept: 306.0688
    Epoch [12/50], Test Losses: mse: 4.0906, mae: 0.8475, huber: 0.6020, swd: 3.0315, ept: 305.0160
      Epoch 12 composite train-obj: 0.699992
            Val objective improved 0.6606 → 0.6290, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 4.6846, mae: 0.9516, huber: 0.6885, swd: 3.2565, ept: 305.6500
    Epoch [13/50], Val Losses: mse: 4.8624, mae: 0.9048, huber: 0.6690, swd: 3.4879, ept: 305.7926
    Epoch [13/50], Test Losses: mse: 4.9401, mae: 0.9183, huber: 0.6765, swd: 3.4238, ept: 302.8265
      Epoch 13 composite train-obj: 0.688508
            No improvement (0.6690), counter 1/5
    Epoch [14/50], Train Losses: mse: 4.6942, mae: 0.9437, huber: 0.6830, swd: 3.2950, ept: 305.7715
    Epoch [14/50], Val Losses: mse: 4.3629, mae: 0.8457, huber: 0.6129, swd: 3.2294, ept: 309.7176
    Epoch [14/50], Test Losses: mse: 3.9614, mae: 0.8180, huber: 0.5804, swd: 2.9876, ept: 306.7617
      Epoch 14 composite train-obj: 0.682968
            Val objective improved 0.6290 → 0.6129, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 4.5545, mae: 0.9262, huber: 0.6675, swd: 3.1958, ept: 306.5423
    Epoch [15/50], Val Losses: mse: 5.1078, mae: 1.1044, huber: 0.8037, swd: 3.5165, ept: 310.3559
    Epoch [15/50], Test Losses: mse: 5.1632, mae: 1.1345, huber: 0.8246, swd: 3.4501, ept: 305.4654
      Epoch 15 composite train-obj: 0.667492
            No improvement (0.8037), counter 1/5
    Epoch [16/50], Train Losses: mse: 4.6588, mae: 0.9442, huber: 0.6833, swd: 3.2320, ept: 306.3310
    Epoch [16/50], Val Losses: mse: 4.4710, mae: 0.8552, huber: 0.6296, swd: 3.2328, ept: 305.8523
    Epoch [16/50], Test Losses: mse: 4.0240, mae: 0.8295, huber: 0.5977, swd: 2.9738, ept: 305.8594
      Epoch 16 composite train-obj: 0.683329
            No improvement (0.6296), counter 2/5
    Epoch [17/50], Train Losses: mse: 4.5174, mae: 0.9131, huber: 0.6579, swd: 3.1747, ept: 307.2371
    Epoch [17/50], Val Losses: mse: 4.4854, mae: 0.8735, huber: 0.6286, swd: 3.2389, ept: 310.5914
    Epoch [17/50], Test Losses: mse: 4.4433, mae: 0.8740, huber: 0.6231, swd: 3.1142, ept: 305.9724
      Epoch 17 composite train-obj: 0.657944
            No improvement (0.6286), counter 3/5
    Epoch [18/50], Train Losses: mse: 4.4751, mae: 0.9083, huber: 0.6531, swd: 3.1534, ept: 307.5340
    Epoch [18/50], Val Losses: mse: 4.2637, mae: 0.8484, huber: 0.5991, swd: 3.1627, ept: 311.1305
    Epoch [18/50], Test Losses: mse: 3.8715, mae: 0.8255, huber: 0.5694, swd: 2.9155, ept: 307.8479
      Epoch 18 composite train-obj: 0.653066
            Val objective improved 0.6129 → 0.5991, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 4.4323, mae: 0.9018, huber: 0.6478, swd: 3.1299, ept: 307.7754
    Epoch [19/50], Val Losses: mse: 4.5771, mae: 0.9687, huber: 0.6909, swd: 3.3005, ept: 313.3387
    Epoch [19/50], Test Losses: mse: 4.2099, mae: 0.9570, huber: 0.6719, swd: 3.0680, ept: 309.7757
      Epoch 19 composite train-obj: 0.647784
            No improvement (0.6909), counter 1/5
    Epoch [20/50], Train Losses: mse: 4.4566, mae: 0.9033, huber: 0.6503, swd: 3.1314, ept: 307.8269
    Epoch [20/50], Val Losses: mse: 4.3985, mae: 0.8430, huber: 0.6043, swd: 3.1983, ept: 306.4342
    Epoch [20/50], Test Losses: mse: 4.0788, mae: 0.8454, huber: 0.5974, swd: 2.9908, ept: 305.9157
      Epoch 20 composite train-obj: 0.650267
            No improvement (0.6043), counter 2/5
    Epoch [21/50], Train Losses: mse: 4.4015, mae: 0.8861, huber: 0.6362, swd: 3.1137, ept: 308.0980
    Epoch [21/50], Val Losses: mse: 4.2099, mae: 0.8310, huber: 0.5871, swd: 3.1413, ept: 311.2268
    Epoch [21/50], Test Losses: mse: 3.8479, mae: 0.8105, huber: 0.5596, swd: 2.9065, ept: 308.2369
      Epoch 21 composite train-obj: 0.636247
            Val objective improved 0.5991 → 0.5871, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 4.3817, mae: 0.8856, huber: 0.6360, swd: 3.1064, ept: 308.2424
    Epoch [22/50], Val Losses: mse: 4.5027, mae: 0.9117, huber: 0.6535, swd: 3.2041, ept: 309.7468
    Epoch [22/50], Test Losses: mse: 4.2914, mae: 0.9026, huber: 0.6379, swd: 3.0077, ept: 306.8239
      Epoch 22 composite train-obj: 0.635967
            No improvement (0.6535), counter 1/5
    Epoch [23/50], Train Losses: mse: 4.3651, mae: 0.8849, huber: 0.6346, swd: 3.0920, ept: 308.4359
    Epoch [23/50], Val Losses: mse: 4.6111, mae: 0.8810, huber: 0.6289, swd: 3.3409, ept: 304.5051
    Epoch [23/50], Test Losses: mse: 4.3036, mae: 0.8854, huber: 0.6233, swd: 3.1340, ept: 303.8360
      Epoch 23 composite train-obj: 0.634594
            No improvement (0.6289), counter 2/5
    Epoch [24/50], Train Losses: mse: 4.3660, mae: 0.8776, huber: 0.6293, swd: 3.0962, ept: 308.4786
    Epoch [24/50], Val Losses: mse: 4.2951, mae: 0.8818, huber: 0.6216, swd: 3.1593, ept: 312.0596
    Epoch [24/50], Test Losses: mse: 3.9565, mae: 0.8551, huber: 0.5893, swd: 2.9286, ept: 308.8621
      Epoch 24 composite train-obj: 0.629292
            No improvement (0.6216), counter 3/5
    Epoch [25/50], Train Losses: mse: 4.3254, mae: 0.8741, huber: 0.6266, swd: 3.0652, ept: 308.8485
    Epoch [25/50], Val Losses: mse: 4.8003, mae: 0.9178, huber: 0.6540, swd: 3.4176, ept: 304.9690
    Epoch [25/50], Test Losses: mse: 4.6588, mae: 0.9374, huber: 0.6666, swd: 3.2709, ept: 303.9770
      Epoch 25 composite train-obj: 0.626581
            No improvement (0.6540), counter 4/5
    Epoch [26/50], Train Losses: mse: 4.3520, mae: 0.8775, huber: 0.6294, swd: 3.0864, ept: 308.6128
    Epoch [26/50], Val Losses: mse: 4.7577, mae: 0.8840, huber: 0.6376, swd: 3.4068, ept: 304.1225
    Epoch [26/50], Test Losses: mse: 4.3526, mae: 0.8954, huber: 0.6389, swd: 3.1850, ept: 304.2961
      Epoch 26 composite train-obj: 0.629439
    Epoch [26/50], Test Losses: mse: 3.8479, mae: 0.8105, huber: 0.5596, swd: 2.9065, ept: 308.2369
    Best round's Test MSE: 3.8479, MAE: 0.8105, SWD: 2.9065
    Best round's Validation MSE: 4.2099, MAE: 0.8310, SWD: 3.1413
    Best round's Test verification MSE : 3.8479, MAE: 0.8105, SWD: 2.9065
    Time taken: 189.34 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 27.9406, mae: 2.8587, huber: 2.4995, swd: 15.2905, ept: 225.6485
    Epoch [1/50], Val Losses: mse: 19.4645, mae: 2.5462, huber: 2.1975, swd: 11.1372, ept: 244.4610
    Epoch [1/50], Test Losses: mse: 17.8786, mae: 2.5705, huber: 2.2114, swd: 9.5379, ept: 234.5734
      Epoch 1 composite train-obj: 2.499515
            Val objective improved inf → 2.1975, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 16.1351, mae: 1.9314, huber: 1.5997, swd: 11.1514, ept: 267.2122
    Epoch [2/50], Val Losses: mse: 17.7716, mae: 1.7301, huber: 1.4339, swd: 13.8778, ept: 276.8882
    Epoch [2/50], Test Losses: mse: 15.1406, mae: 1.6883, huber: 1.3783, swd: 10.9645, ept: 277.1157
      Epoch 2 composite train-obj: 1.599716
            Val objective improved 2.1975 → 1.4339, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.2054, mae: 1.6837, huber: 1.3605, swd: 8.2376, ept: 277.7970
    Epoch [3/50], Val Losses: mse: 13.5322, mae: 1.8643, huber: 1.5427, swd: 9.6405, ept: 276.7872
    Epoch [3/50], Test Losses: mse: 13.2950, mae: 1.9164, huber: 1.5815, swd: 8.5138, ept: 273.1089
      Epoch 3 composite train-obj: 1.360455
            No improvement (1.5427), counter 1/5
    Epoch [4/50], Train Losses: mse: 9.9623, mae: 1.5496, huber: 1.2326, swd: 6.5512, ept: 283.8672
    Epoch [4/50], Val Losses: mse: 7.6876, mae: 1.3350, huber: 1.0375, swd: 5.6224, ept: 291.2884
    Epoch [4/50], Test Losses: mse: 7.6428, mae: 1.3430, huber: 1.0379, swd: 5.0821, ept: 291.0839
      Epoch 4 composite train-obj: 1.232590
            Val objective improved 1.4339 → 1.0375, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.7007, mae: 1.3628, huber: 1.0542, swd: 5.1372, ept: 290.3489
    Epoch [5/50], Val Losses: mse: 8.1431, mae: 1.3631, huber: 1.0486, swd: 6.4139, ept: 290.7702
    Epoch [5/50], Test Losses: mse: 8.7454, mae: 1.4198, huber: 1.0955, swd: 6.7815, ept: 290.8324
      Epoch 5 composite train-obj: 1.054159
            No improvement (1.0486), counter 1/5
    Epoch [6/50], Train Losses: mse: 6.5888, mae: 1.2423, huber: 0.9418, swd: 4.3822, ept: 294.1800
    Epoch [6/50], Val Losses: mse: 6.0208, mae: 1.2911, huber: 0.9785, swd: 4.2427, ept: 296.8822
    Epoch [6/50], Test Losses: mse: 6.0054, mae: 1.3229, huber: 1.0024, swd: 4.0219, ept: 293.1244
      Epoch 6 composite train-obj: 0.941797
            Val objective improved 1.0375 → 0.9785, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 6.0679, mae: 1.1814, huber: 0.8863, swd: 4.0288, ept: 296.8076
    Epoch [7/50], Val Losses: mse: 7.0496, mae: 1.1887, huber: 0.9099, swd: 5.1243, ept: 291.8514
    Epoch [7/50], Test Losses: mse: 7.6786, mae: 1.2867, huber: 0.9923, swd: 5.3499, ept: 290.3452
      Epoch 7 composite train-obj: 0.886339
            Val objective improved 0.9785 → 0.9099, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.6824, mae: 1.1187, huber: 0.8315, swd: 3.8283, ept: 299.4286
    Epoch [8/50], Val Losses: mse: 5.4140, mae: 1.0023, huber: 0.7346, swd: 3.9328, ept: 298.7133
    Epoch [8/50], Test Losses: mse: 5.5657, mae: 1.0539, huber: 0.7754, swd: 4.0052, ept: 298.0810
      Epoch 8 composite train-obj: 0.831455
            Val objective improved 0.9099 → 0.7346, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 5.3715, mae: 1.0736, huber: 0.7913, swd: 3.6497, ept: 301.1899
    Epoch [9/50], Val Losses: mse: 6.6466, mae: 1.1528, huber: 0.8631, swd: 4.7336, ept: 295.2740
    Epoch [9/50], Test Losses: mse: 6.7398, mae: 1.1872, huber: 0.8885, swd: 4.6462, ept: 295.1139
      Epoch 9 composite train-obj: 0.791316
            No improvement (0.8631), counter 1/5
    Epoch [10/50], Train Losses: mse: 6.3303, mae: 1.1139, huber: 0.8302, swd: 4.4421, ept: 300.3904
    Epoch [10/50], Val Losses: mse: 5.3007, mae: 1.0299, huber: 0.7602, swd: 3.7394, ept: 302.7698
    Epoch [10/50], Test Losses: mse: 5.2969, mae: 1.0632, huber: 0.7827, swd: 3.6316, ept: 300.4788
      Epoch 10 composite train-obj: 0.830182
            No improvement (0.7602), counter 2/5
    Epoch [11/50], Train Losses: mse: 5.1637, mae: 1.0354, huber: 0.7592, swd: 3.5562, ept: 302.9036
    Epoch [11/50], Val Losses: mse: 5.0178, mae: 0.9801, huber: 0.7187, swd: 3.6601, ept: 305.2408
    Epoch [11/50], Test Losses: mse: 5.1888, mae: 1.0087, huber: 0.7428, swd: 3.7132, ept: 301.4312
      Epoch 11 composite train-obj: 0.759212
            Val objective improved 0.7346 → 0.7187, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 5.1293, mae: 1.0273, huber: 0.7530, swd: 3.5469, ept: 303.1751
    Epoch [12/50], Val Losses: mse: 5.0291, mae: 1.1196, huber: 0.8104, swd: 3.7541, ept: 311.3636
    Epoch [12/50], Test Losses: mse: 4.7749, mae: 1.1353, huber: 0.8176, swd: 3.5627, ept: 308.1218
      Epoch 12 composite train-obj: 0.752980
            No improvement (0.8104), counter 1/5
    Epoch [13/50], Train Losses: mse: 5.0119, mae: 1.0190, huber: 0.7458, swd: 3.4855, ept: 303.6703
    Epoch [13/50], Val Losses: mse: 4.5386, mae: 0.9418, huber: 0.6798, swd: 3.4519, ept: 310.6843
    Epoch [13/50], Test Losses: mse: 4.6740, mae: 0.9669, huber: 0.6943, swd: 3.3846, ept: 305.6998
      Epoch 13 composite train-obj: 0.745781
            Val objective improved 0.7187 → 0.6798, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 4.9076, mae: 0.9925, huber: 0.7228, swd: 3.4276, ept: 305.1004
    Epoch [14/50], Val Losses: mse: 4.6560, mae: 0.9119, huber: 0.6649, swd: 3.3949, ept: 305.3503
    Epoch [14/50], Test Losses: mse: 4.3846, mae: 0.9028, huber: 0.6515, swd: 3.2219, ept: 304.4121
      Epoch 14 composite train-obj: 0.722758
            Val objective improved 0.6798 → 0.6649, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 4.8240, mae: 0.9781, huber: 0.7109, swd: 3.3929, ept: 305.3660
    Epoch [15/50], Val Losses: mse: 4.7618, mae: 0.9521, huber: 0.6900, swd: 3.4991, ept: 307.0973
    Epoch [15/50], Test Losses: mse: 4.8150, mae: 0.9729, huber: 0.7034, swd: 3.4640, ept: 303.7267
      Epoch 15 composite train-obj: 0.710873
            No improvement (0.6900), counter 1/5
    Epoch [16/50], Train Losses: mse: 4.8133, mae: 0.9726, huber: 0.7060, swd: 3.3815, ept: 305.9610
    Epoch [16/50], Val Losses: mse: 4.9356, mae: 0.9595, huber: 0.7170, swd: 3.6301, ept: 304.9036
    Epoch [16/50], Test Losses: mse: 4.6968, mae: 0.9567, huber: 0.7072, swd: 3.4679, ept: 302.6712
      Epoch 16 composite train-obj: 0.706042
            No improvement (0.7170), counter 2/5
    Epoch [17/50], Train Losses: mse: 5.0263, mae: 0.9936, huber: 0.7269, swd: 3.4968, ept: 304.4658
    Epoch [17/50], Val Losses: mse: 4.7059, mae: 1.0409, huber: 0.7485, swd: 3.5861, ept: 313.6442
    Epoch [17/50], Test Losses: mse: 4.5909, mae: 1.0698, huber: 0.7681, swd: 3.4319, ept: 309.4554
      Epoch 17 composite train-obj: 0.726934
            No improvement (0.7485), counter 3/5
    Epoch [18/50], Train Losses: mse: 4.6826, mae: 0.9609, huber: 0.6952, swd: 3.3094, ept: 306.3344
    Epoch [18/50], Val Losses: mse: 5.3249, mae: 1.0167, huber: 0.7431, swd: 3.7194, ept: 303.6064
    Epoch [18/50], Test Losses: mse: 5.2733, mae: 1.0646, huber: 0.7786, swd: 3.6738, ept: 301.1813
      Epoch 18 composite train-obj: 0.695166
            No improvement (0.7431), counter 4/5
    Epoch [19/50], Train Losses: mse: 4.6438, mae: 0.9406, huber: 0.6795, swd: 3.2972, ept: 306.4018
    Epoch [19/50], Val Losses: mse: 4.7071, mae: 0.9295, huber: 0.6843, swd: 3.3923, ept: 305.0933
    Epoch [19/50], Test Losses: mse: 4.7442, mae: 0.9644, huber: 0.7083, swd: 3.2453, ept: 302.9259
      Epoch 19 composite train-obj: 0.679463
    Epoch [19/50], Test Losses: mse: 4.3846, mae: 0.9028, huber: 0.6515, swd: 3.2219, ept: 304.4121
    Best round's Test MSE: 4.3846, MAE: 0.9028, SWD: 3.2219
    Best round's Validation MSE: 4.6560, MAE: 0.9119, SWD: 3.3949
    Best round's Test verification MSE : 4.3846, MAE: 0.9028, SWD: 3.2219
    Time taken: 146.24 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 26.6613, mae: 2.7544, huber: 2.3995, swd: 14.0500, ept: 232.7183
    Epoch [1/50], Val Losses: mse: 19.1208, mae: 1.9294, huber: 1.6113, swd: 14.2914, ept: 275.9702
    Epoch [1/50], Test Losses: mse: 16.1805, mae: 1.8853, huber: 1.5585, swd: 11.3190, ept: 272.1213
      Epoch 1 composite train-obj: 2.399531
            Val objective improved inf → 1.6113, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 16.4182, mae: 1.8544, huber: 1.5304, swd: 11.7201, ept: 273.6743
    Epoch [2/50], Val Losses: mse: 17.2280, mae: 1.8099, huber: 1.5047, swd: 13.2335, ept: 281.1864
    Epoch [2/50], Test Losses: mse: 14.4613, mae: 1.7436, huber: 1.4294, swd: 10.5228, ept: 280.2619
      Epoch 2 composite train-obj: 1.530430
            Val objective improved 1.6113 → 1.5047, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 13.0903, mae: 1.7234, huber: 1.4039, swd: 9.0640, ept: 278.4686
    Epoch [3/50], Val Losses: mse: 12.4984, mae: 1.6816, huber: 1.3780, swd: 8.8876, ept: 282.0337
    Epoch [3/50], Test Losses: mse: 10.8416, mae: 1.6143, huber: 1.3024, swd: 7.2321, ept: 282.3753
      Epoch 3 composite train-obj: 1.403890
            Val objective improved 1.5047 → 1.3780, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 10.9413, mae: 1.6122, huber: 1.2992, swd: 7.2535, ept: 281.5697
    Epoch [4/50], Val Losses: mse: 10.5417, mae: 1.5371, huber: 1.2407, swd: 7.3487, ept: 282.1567
    Epoch [4/50], Test Losses: mse: 9.8404, mae: 1.5162, huber: 1.2108, swd: 6.3755, ept: 282.9136
      Epoch 4 composite train-obj: 1.299170
            Val objective improved 1.3780 → 1.2407, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 8.7461, mae: 1.4267, huber: 1.1210, swd: 5.8454, ept: 287.1293
    Epoch [5/50], Val Losses: mse: 10.9094, mae: 1.5038, huber: 1.2100, swd: 7.4724, ept: 284.6418
    Epoch [5/50], Test Losses: mse: 11.1305, mae: 1.6269, huber: 1.3187, swd: 6.9771, ept: 278.8068
      Epoch 5 composite train-obj: 1.121033
            Val objective improved 1.2407 → 1.2100, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.4530, mae: 1.2475, huber: 0.9510, swd: 5.1408, ept: 294.9398
    Epoch [6/50], Val Losses: mse: 7.8158, mae: 1.1949, huber: 0.9165, swd: 5.3121, ept: 294.4790
    Epoch [6/50], Test Losses: mse: 7.3903, mae: 1.2524, huber: 0.9594, swd: 4.7392, ept: 292.9468
      Epoch 6 composite train-obj: 0.950978
            Val objective improved 1.2100 → 0.9165, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.9298, mae: 1.1279, huber: 0.8398, swd: 4.0727, ept: 298.9415
    Epoch [7/50], Val Losses: mse: 4.9645, mae: 0.9526, huber: 0.6935, swd: 3.7487, ept: 304.0927
    Epoch [7/50], Test Losses: mse: 5.2172, mae: 0.9849, huber: 0.7195, swd: 3.6321, ept: 301.7209
      Epoch 7 composite train-obj: 0.839813
            Val objective improved 0.9165 → 0.6935, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.3233, mae: 1.0446, huber: 0.7658, swd: 3.7198, ept: 302.3995
    Epoch [8/50], Val Losses: mse: 4.9032, mae: 0.9262, huber: 0.6763, swd: 3.6352, ept: 304.4375
    Epoch [8/50], Test Losses: mse: 4.9639, mae: 0.9455, huber: 0.6914, swd: 3.4913, ept: 302.5506
      Epoch 8 composite train-obj: 0.765782
            Val objective improved 0.6935 → 0.6763, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 5.1860, mae: 1.0207, huber: 0.7459, swd: 3.6587, ept: 303.3153
    Epoch [9/50], Val Losses: mse: 5.3661, mae: 1.0138, huber: 0.7601, swd: 3.8548, ept: 308.6102
    Epoch [9/50], Test Losses: mse: 5.1275, mae: 1.0303, huber: 0.7669, swd: 3.6429, ept: 305.1838
      Epoch 9 composite train-obj: 0.745902
            No improvement (0.7601), counter 1/5
    Epoch [10/50], Train Losses: mse: 5.1118, mae: 1.0017, huber: 0.7305, swd: 3.6344, ept: 304.3583
    Epoch [10/50], Val Losses: mse: 5.1937, mae: 1.1493, huber: 0.8501, swd: 3.6356, ept: 307.9175
    Epoch [10/50], Test Losses: mse: 5.0644, mae: 1.1645, huber: 0.8591, swd: 3.4233, ept: 304.4970
      Epoch 10 composite train-obj: 0.730452
            No improvement (0.8501), counter 2/5
    Epoch [11/50], Train Losses: mse: 5.0348, mae: 0.9907, huber: 0.7211, swd: 3.6016, ept: 304.4464
    Epoch [11/50], Val Losses: mse: 5.8304, mae: 1.1694, huber: 0.8704, swd: 3.8451, ept: 301.5840
    Epoch [11/50], Test Losses: mse: 5.9716, mae: 1.1962, huber: 0.8889, swd: 3.7217, ept: 297.9743
      Epoch 11 composite train-obj: 0.721055
            No improvement (0.8704), counter 3/5
    Epoch [12/50], Train Losses: mse: 5.7671, mae: 1.0931, huber: 0.8139, swd: 3.9437, ept: 299.5828
    Epoch [12/50], Val Losses: mse: 5.1810, mae: 0.9417, huber: 0.6859, swd: 3.6888, ept: 302.3160
    Epoch [12/50], Test Losses: mse: 4.7652, mae: 0.9410, huber: 0.6757, swd: 3.4081, ept: 302.4242
      Epoch 12 composite train-obj: 0.813949
            No improvement (0.6859), counter 4/5
    Epoch [13/50], Train Losses: mse: 4.8532, mae: 0.9683, huber: 0.7015, swd: 3.4634, ept: 304.4757
    Epoch [13/50], Val Losses: mse: 5.0361, mae: 1.1120, huber: 0.8127, swd: 3.6969, ept: 310.5612
    Epoch [13/50], Test Losses: mse: 4.9638, mae: 1.1529, huber: 0.8420, swd: 3.5595, ept: 305.9948
      Epoch 13 composite train-obj: 0.701522
    Epoch [13/50], Test Losses: mse: 4.9639, mae: 0.9455, huber: 0.6914, swd: 3.4913, ept: 302.5506
    Best round's Test MSE: 4.9639, MAE: 0.9455, SWD: 3.4913
    Best round's Validation MSE: 4.9032, MAE: 0.9262, SWD: 3.6352
    Best round's Test verification MSE : 4.9639, MAE: 0.9455, SWD: 3.4913
    Time taken: 96.92 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_rossler_seq96_pred336_20250511_0404)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 4.3988 ± 0.4557
      mae: 0.8863 ± 0.0563
      huber: 0.6342 ± 0.0552
      swd: 3.2065 ± 0.2390
      ept: 305.0665 ± 2.3671
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 4.5897 ± 0.2869
      mae: 0.8897 ± 0.0419
      huber: 0.6428 ± 0.0396
      swd: 3.3905 ± 0.2017
      ept: 307.0049 ± 3.0085
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 432.55 seconds
    
    Experiment complete: TimeMixer_rossler_seq96_pred336_20250511_0404
    Model: TimeMixer
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['rossler']['channels'],
    enc_in=data_mgr.datasets['rossler']['channels'],
    dec_in=data_mgr.datasets['rossler']['channels'],
    c_out=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 26.1457, mae: 2.7262, huber: 2.3707, swd: 13.9791, target_std: 3.9435
    Epoch [1/50], Val Losses: mse: 19.2632, mae: 2.3370, huber: 1.9792, swd: 13.2302, target_std: 3.9670
    Epoch [1/50], Test Losses: mse: 17.3000, mae: 2.3493, huber: 1.9811, swd: 10.7780, target_std: 4.1272
      Epoch 1 composite train-obj: 2.370712
            Val objective improved inf → 1.9792, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 15.5450, mae: 1.8360, huber: 1.5116, swd: 10.7813, target_std: 3.9446
    Epoch [2/50], Val Losses: mse: 13.2323, mae: 1.9618, huber: 1.6339, swd: 8.6212, target_std: 3.9670
    Epoch [2/50], Test Losses: mse: 11.6700, mae: 1.9073, huber: 1.5713, swd: 7.1846, target_std: 4.1272
      Epoch 2 composite train-obj: 1.511615
            Val objective improved 1.9792 → 1.6339, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.2883, mae: 1.6136, huber: 1.2967, swd: 7.4682, target_std: 3.9461
    Epoch [3/50], Val Losses: mse: 10.5310, mae: 1.5991, huber: 1.2753, swd: 7.0294, target_std: 3.9670
    Epoch [3/50], Test Losses: mse: 10.5746, mae: 1.6294, huber: 1.2967, swd: 6.3015, target_std: 4.1272
      Epoch 3 composite train-obj: 1.296733
            Val objective improved 1.6339 → 1.2753, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.0075, mae: 1.3904, huber: 1.0820, swd: 5.2052, target_std: 3.9350
    Epoch [4/50], Val Losses: mse: 7.1512, mae: 1.4312, huber: 1.0989, swd: 4.8961, target_std: 3.9670
    Epoch [4/50], Test Losses: mse: 6.9447, mae: 1.4565, huber: 1.1146, swd: 4.5114, target_std: 4.1272
      Epoch 4 composite train-obj: 1.082048
            Val objective improved 1.2753 → 1.0989, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.6385, mae: 1.2651, huber: 0.9646, swd: 4.3119, target_std: 3.9383
    Epoch [5/50], Val Losses: mse: 5.8111, mae: 1.2476, huber: 0.9320, swd: 4.1112, target_std: 3.9670
    Epoch [5/50], Test Losses: mse: 5.7144, mae: 1.2720, huber: 0.9473, swd: 4.0325, target_std: 4.1272
      Epoch 5 composite train-obj: 0.964596
            Val objective improved 1.0989 → 0.9320, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.8824, mae: 1.1595, huber: 0.8691, swd: 3.8664, target_std: 3.9367
    Epoch [6/50], Val Losses: mse: 5.5236, mae: 0.9953, huber: 0.7455, swd: 3.9179, target_std: 3.9670
    Epoch [6/50], Test Losses: mse: 5.3685, mae: 1.0116, huber: 0.7550, swd: 3.6706, target_std: 4.1272
      Epoch 6 composite train-obj: 0.869106
            Val objective improved 0.9320 → 0.7455, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.3953, mae: 1.0803, huber: 0.7975, swd: 3.6343, target_std: 3.9430
    Epoch [7/50], Val Losses: mse: 5.5400, mae: 1.0579, huber: 0.7858, swd: 3.8140, target_std: 3.9670
    Epoch [7/50], Test Losses: mse: 5.7852, mae: 1.0957, huber: 0.8186, swd: 3.7368, target_std: 4.1272
      Epoch 7 composite train-obj: 0.797501
            No improvement (0.7858), counter 1/5
    Epoch [8/50], Train Losses: mse: 5.1944, mae: 1.0473, huber: 0.7694, swd: 3.5252, target_std: 3.9458
    Epoch [8/50], Val Losses: mse: 5.9778, mae: 1.1088, huber: 0.8367, swd: 4.4440, target_std: 3.9670
    Epoch [8/50], Test Losses: mse: 5.8619, mae: 1.1229, huber: 0.8448, swd: 4.1477, target_std: 4.1272
      Epoch 8 composite train-obj: 0.769362
            No improvement (0.8367), counter 2/5
    Epoch [9/50], Train Losses: mse: 5.0891, mae: 1.0320, huber: 0.7563, swd: 3.4785, target_std: 3.9434
    Epoch [9/50], Val Losses: mse: 4.8360, mae: 0.9596, huber: 0.6972, swd: 3.5096, target_std: 3.9670
    Epoch [9/50], Test Losses: mse: 4.5570, mae: 0.9663, huber: 0.6964, swd: 3.2687, target_std: 4.1272
      Epoch 9 composite train-obj: 0.756289
            Val objective improved 0.7455 → 0.6972, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 4.9697, mae: 1.0082, huber: 0.7365, swd: 3.4217, target_std: 3.9400
    Epoch [10/50], Val Losses: mse: 4.6962, mae: 0.9076, huber: 0.6606, swd: 3.3709, target_std: 3.9670
    Epoch [10/50], Test Losses: mse: 4.6090, mae: 0.9209, huber: 0.6666, swd: 3.1760, target_std: 4.1272
      Epoch 10 composite train-obj: 0.736515
            Val objective improved 0.6972 → 0.6606, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 4.8060, mae: 0.9804, huber: 0.7120, swd: 3.3413, target_std: 3.9464
    Epoch [11/50], Val Losses: mse: 5.4598, mae: 1.1526, huber: 0.8606, swd: 3.5723, target_std: 3.9670
    Epoch [11/50], Test Losses: mse: 5.3510, mae: 1.1627, huber: 0.8647, swd: 3.3785, target_std: 4.1272
      Epoch 11 composite train-obj: 0.711985
            No improvement (0.8606), counter 1/5
    Epoch [12/50], Train Losses: mse: 4.7286, mae: 0.9659, huber: 0.7000, swd: 3.2814, target_std: 3.9379
    Epoch [12/50], Val Losses: mse: 4.4544, mae: 0.8676, huber: 0.6290, swd: 3.2829, target_std: 3.9670
    Epoch [12/50], Test Losses: mse: 4.0906, mae: 0.8475, huber: 0.6020, swd: 3.0315, target_std: 4.1272
      Epoch 12 composite train-obj: 0.699992
            Val objective improved 0.6606 → 0.6290, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 4.6846, mae: 0.9516, huber: 0.6885, swd: 3.2565, target_std: 3.9372
    Epoch [13/50], Val Losses: mse: 4.8624, mae: 0.9048, huber: 0.6690, swd: 3.4879, target_std: 3.9670
    Epoch [13/50], Test Losses: mse: 4.9401, mae: 0.9183, huber: 0.6765, swd: 3.4238, target_std: 4.1272
      Epoch 13 composite train-obj: 0.688508
            No improvement (0.6690), counter 1/5
    Epoch [14/50], Train Losses: mse: 4.6942, mae: 0.9437, huber: 0.6830, swd: 3.2950, target_std: 3.9429
    Epoch [14/50], Val Losses: mse: 4.3629, mae: 0.8457, huber: 0.6129, swd: 3.2294, target_std: 3.9670
    Epoch [14/50], Test Losses: mse: 3.9614, mae: 0.8180, huber: 0.5804, swd: 2.9876, target_std: 4.1272
      Epoch 14 composite train-obj: 0.682968
            Val objective improved 0.6290 → 0.6129, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 4.5545, mae: 0.9262, huber: 0.6675, swd: 3.1958, target_std: 3.9410
    Epoch [15/50], Val Losses: mse: 5.1078, mae: 1.1044, huber: 0.8037, swd: 3.5165, target_std: 3.9670
    Epoch [15/50], Test Losses: mse: 5.1632, mae: 1.1346, huber: 0.8246, swd: 3.4501, target_std: 4.1272
      Epoch 15 composite train-obj: 0.667493
            No improvement (0.8037), counter 1/5
    Epoch [16/50], Train Losses: mse: 4.6588, mae: 0.9442, huber: 0.6833, swd: 3.2320, target_std: 3.9375
    Epoch [16/50], Val Losses: mse: 4.4710, mae: 0.8552, huber: 0.6296, swd: 3.2328, target_std: 3.9670
    Epoch [16/50], Test Losses: mse: 4.0240, mae: 0.8295, huber: 0.5977, swd: 2.9738, target_std: 4.1272
      Epoch 16 composite train-obj: 0.683329
            No improvement (0.6296), counter 2/5
    Epoch [17/50], Train Losses: mse: 4.5174, mae: 0.9131, huber: 0.6579, swd: 3.1747, target_std: 3.9379
    Epoch [17/50], Val Losses: mse: 4.4854, mae: 0.8735, huber: 0.6286, swd: 3.2389, target_std: 3.9670
    Epoch [17/50], Test Losses: mse: 4.4433, mae: 0.8740, huber: 0.6231, swd: 3.1142, target_std: 4.1272
      Epoch 17 composite train-obj: 0.657944
            No improvement (0.6286), counter 3/5
    Epoch [18/50], Train Losses: mse: 4.4751, mae: 0.9083, huber: 0.6531, swd: 3.1534, target_std: 3.9358
    Epoch [18/50], Val Losses: mse: 4.2637, mae: 0.8484, huber: 0.5991, swd: 3.1627, target_std: 3.9670
    Epoch [18/50], Test Losses: mse: 3.8715, mae: 0.8255, huber: 0.5694, swd: 2.9155, target_std: 4.1272
      Epoch 18 composite train-obj: 0.653066
            Val objective improved 0.6129 → 0.5991, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 4.4323, mae: 0.9018, huber: 0.6478, swd: 3.1299, target_std: 3.9390
    Epoch [19/50], Val Losses: mse: 4.5771, mae: 0.9687, huber: 0.6909, swd: 3.3005, target_std: 3.9670
    Epoch [19/50], Test Losses: mse: 4.2099, mae: 0.9570, huber: 0.6719, swd: 3.0680, target_std: 4.1272
      Epoch 19 composite train-obj: 0.647784
            No improvement (0.6909), counter 1/5
    Epoch [20/50], Train Losses: mse: 4.4566, mae: 0.9033, huber: 0.6503, swd: 3.1314, target_std: 3.9341
    Epoch [20/50], Val Losses: mse: 4.3985, mae: 0.8430, huber: 0.6043, swd: 3.1983, target_std: 3.9670
    Epoch [20/50], Test Losses: mse: 4.0788, mae: 0.8454, huber: 0.5974, swd: 2.9908, target_std: 4.1272
      Epoch 20 composite train-obj: 0.650267
            No improvement (0.6043), counter 2/5
    Epoch [21/50], Train Losses: mse: 4.4015, mae: 0.8861, huber: 0.6362, swd: 3.1137, target_std: 3.9356
    Epoch [21/50], Val Losses: mse: 4.2099, mae: 0.8310, huber: 0.5871, swd: 3.1413, target_std: 3.9670
    Epoch [21/50], Test Losses: mse: 3.8479, mae: 0.8105, huber: 0.5596, swd: 2.9064, target_std: 4.1272
      Epoch 21 composite train-obj: 0.636247
            Val objective improved 0.5991 → 0.5871, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 4.3817, mae: 0.8856, huber: 0.6360, swd: 3.1064, target_std: 3.9373
    Epoch [22/50], Val Losses: mse: 4.5027, mae: 0.9117, huber: 0.6535, swd: 3.2041, target_std: 3.9670
    Epoch [22/50], Test Losses: mse: 4.2914, mae: 0.9026, huber: 0.6379, swd: 3.0077, target_std: 4.1272
      Epoch 22 composite train-obj: 0.635967
            No improvement (0.6535), counter 1/5
    Epoch [23/50], Train Losses: mse: 4.3651, mae: 0.8849, huber: 0.6346, swd: 3.0920, target_std: 3.9384
    Epoch [23/50], Val Losses: mse: 4.6111, mae: 0.8810, huber: 0.6289, swd: 3.3409, target_std: 3.9670
    Epoch [23/50], Test Losses: mse: 4.3036, mae: 0.8854, huber: 0.6233, swd: 3.1340, target_std: 4.1272
      Epoch 23 composite train-obj: 0.634593
            No improvement (0.6289), counter 2/5
    Epoch [24/50], Train Losses: mse: 4.3660, mae: 0.8776, huber: 0.6293, swd: 3.0962, target_std: 3.9335
    Epoch [24/50], Val Losses: mse: 4.2951, mae: 0.8818, huber: 0.6216, swd: 3.1593, target_std: 3.9670
    Epoch [24/50], Test Losses: mse: 3.9565, mae: 0.8551, huber: 0.5893, swd: 2.9286, target_std: 4.1272
      Epoch 24 composite train-obj: 0.629292
            No improvement (0.6216), counter 3/5
    Epoch [25/50], Train Losses: mse: 4.3254, mae: 0.8741, huber: 0.6266, swd: 3.0652, target_std: 3.9420
    Epoch [25/50], Val Losses: mse: 4.8003, mae: 0.9178, huber: 0.6540, swd: 3.4176, target_std: 3.9670
    Epoch [25/50], Test Losses: mse: 4.6588, mae: 0.9374, huber: 0.6666, swd: 3.2709, target_std: 4.1272
      Epoch 25 composite train-obj: 0.626581
            No improvement (0.6540), counter 4/5
    Epoch [26/50], Train Losses: mse: 4.3520, mae: 0.8775, huber: 0.6294, swd: 3.0864, target_std: 3.9370
    Epoch [26/50], Val Losses: mse: 4.7578, mae: 0.8840, huber: 0.6376, swd: 3.4069, target_std: 3.9670
    Epoch [26/50], Test Losses: mse: 4.3526, mae: 0.8954, huber: 0.6389, swd: 3.1851, target_std: 4.1272
      Epoch 26 composite train-obj: 0.629439
    Epoch [26/50], Test Losses: mse: 3.8479, mae: 0.8105, huber: 0.5596, swd: 2.9064, target_std: 4.1272
    Best round's Test MSE: 3.8479, MAE: 0.8105, SWD: 2.9064
    Best round's Validation MSE: 4.2099, MAE: 0.8310
    Best round's Test verification MSE : 3.8479, MAE: 0.8105, SWD: 2.9064
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 27.9406, mae: 2.8587, huber: 2.4995, swd: 15.2905, target_std: 3.9337
    Epoch [1/50], Val Losses: mse: 19.4645, mae: 2.5462, huber: 2.1975, swd: 11.1372, target_std: 3.9670
    Epoch [1/50], Test Losses: mse: 17.8786, mae: 2.5705, huber: 2.2114, swd: 9.5379, target_std: 4.1272
      Epoch 1 composite train-obj: 2.499515
            Val objective improved inf → 2.1975, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 16.1351, mae: 1.9314, huber: 1.5997, swd: 11.1514, target_std: 3.9339
    Epoch [2/50], Val Losses: mse: 17.7716, mae: 1.7301, huber: 1.4339, swd: 13.8778, target_std: 3.9670
    Epoch [2/50], Test Losses: mse: 15.1406, mae: 1.6883, huber: 1.3783, swd: 10.9645, target_std: 4.1272
      Epoch 2 composite train-obj: 1.599716
            Val objective improved 2.1975 → 1.4339, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.2054, mae: 1.6837, huber: 1.3605, swd: 8.2376, target_std: 3.9392
    Epoch [3/50], Val Losses: mse: 13.5322, mae: 1.8643, huber: 1.5427, swd: 9.6404, target_std: 3.9670
    Epoch [3/50], Test Losses: mse: 13.2950, mae: 1.9164, huber: 1.5815, swd: 8.5138, target_std: 4.1272
      Epoch 3 composite train-obj: 1.360455
            No improvement (1.5427), counter 1/5
    Epoch [4/50], Train Losses: mse: 9.9623, mae: 1.5496, huber: 1.2326, swd: 6.5512, target_std: 3.9493
    Epoch [4/50], Val Losses: mse: 7.6876, mae: 1.3350, huber: 1.0375, swd: 5.6224, target_std: 3.9670
    Epoch [4/50], Test Losses: mse: 7.6427, mae: 1.3430, huber: 1.0379, swd: 5.0821, target_std: 4.1272
      Epoch 4 composite train-obj: 1.232590
            Val objective improved 1.4339 → 1.0375, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.7007, mae: 1.3628, huber: 1.0542, swd: 5.1372, target_std: 3.9393
    Epoch [5/50], Val Losses: mse: 8.1431, mae: 1.3631, huber: 1.0486, swd: 6.4139, target_std: 3.9670
    Epoch [5/50], Test Losses: mse: 8.7454, mae: 1.4198, huber: 1.0955, swd: 6.7815, target_std: 4.1272
      Epoch 5 composite train-obj: 1.054159
            No improvement (1.0486), counter 1/5
    Epoch [6/50], Train Losses: mse: 6.5888, mae: 1.2423, huber: 0.9418, swd: 4.3822, target_std: 3.9465
    Epoch [6/50], Val Losses: mse: 6.0208, mae: 1.2911, huber: 0.9785, swd: 4.2428, target_std: 3.9670
    Epoch [6/50], Test Losses: mse: 6.0054, mae: 1.3229, huber: 1.0024, swd: 4.0220, target_std: 4.1272
      Epoch 6 composite train-obj: 0.941797
            Val objective improved 1.0375 → 0.9785, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 6.0679, mae: 1.1814, huber: 0.8863, swd: 4.0288, target_std: 3.9419
    Epoch [7/50], Val Losses: mse: 7.0496, mae: 1.1887, huber: 0.9099, swd: 5.1243, target_std: 3.9670
    Epoch [7/50], Test Losses: mse: 7.6786, mae: 1.2867, huber: 0.9923, swd: 5.3499, target_std: 4.1272
      Epoch 7 composite train-obj: 0.886339
            Val objective improved 0.9785 → 0.9099, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.6824, mae: 1.1187, huber: 0.8315, swd: 3.8283, target_std: 3.9364
    Epoch [8/50], Val Losses: mse: 5.4141, mae: 1.0023, huber: 0.7346, swd: 3.9328, target_std: 3.9670
    Epoch [8/50], Test Losses: mse: 5.5657, mae: 1.0539, huber: 0.7754, swd: 4.0052, target_std: 4.1272
      Epoch 8 composite train-obj: 0.831454
            Val objective improved 0.9099 → 0.7346, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 5.3715, mae: 1.0736, huber: 0.7913, swd: 3.6497, target_std: 3.9333
    Epoch [9/50], Val Losses: mse: 6.6466, mae: 1.1528, huber: 0.8631, swd: 4.7336, target_std: 3.9670
    Epoch [9/50], Test Losses: mse: 6.7398, mae: 1.1872, huber: 0.8885, swd: 4.6462, target_std: 4.1272
      Epoch 9 composite train-obj: 0.791315
            No improvement (0.8631), counter 1/5
    Epoch [10/50], Train Losses: mse: 6.3303, mae: 1.1139, huber: 0.8302, swd: 4.4421, target_std: 3.9393
    Epoch [10/50], Val Losses: mse: 5.3006, mae: 1.0299, huber: 0.7601, swd: 3.7393, target_std: 3.9670
    Epoch [10/50], Test Losses: mse: 5.2969, mae: 1.0632, huber: 0.7827, swd: 3.6316, target_std: 4.1272
      Epoch 10 composite train-obj: 0.830182
            No improvement (0.7601), counter 2/5
    Epoch [11/50], Train Losses: mse: 5.1637, mae: 1.0354, huber: 0.7592, swd: 3.5562, target_std: 3.9354
    Epoch [11/50], Val Losses: mse: 5.0177, mae: 0.9801, huber: 0.7187, swd: 3.6600, target_std: 3.9670
    Epoch [11/50], Test Losses: mse: 5.1888, mae: 1.0087, huber: 0.7428, swd: 3.7132, target_std: 4.1272
      Epoch 11 composite train-obj: 0.759212
            Val objective improved 0.7346 → 0.7187, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 5.1293, mae: 1.0273, huber: 0.7530, swd: 3.5469, target_std: 3.9384
    Epoch [12/50], Val Losses: mse: 5.0291, mae: 1.1196, huber: 0.8104, swd: 3.7541, target_std: 3.9670
    Epoch [12/50], Test Losses: mse: 4.7749, mae: 1.1353, huber: 0.8176, swd: 3.5627, target_std: 4.1272
      Epoch 12 composite train-obj: 0.752980
            No improvement (0.8104), counter 1/5
    Epoch [13/50], Train Losses: mse: 5.0119, mae: 1.0190, huber: 0.7458, swd: 3.4855, target_std: 3.9446
    Epoch [13/50], Val Losses: mse: 4.5386, mae: 0.9418, huber: 0.6798, swd: 3.4519, target_std: 3.9670
    Epoch [13/50], Test Losses: mse: 4.6740, mae: 0.9669, huber: 0.6943, swd: 3.3846, target_std: 4.1272
      Epoch 13 composite train-obj: 0.745782
            Val objective improved 0.7187 → 0.6798, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 4.9076, mae: 0.9925, huber: 0.7228, swd: 3.4276, target_std: 3.9417
    Epoch [14/50], Val Losses: mse: 4.6560, mae: 0.9119, huber: 0.6649, swd: 3.3949, target_std: 3.9670
    Epoch [14/50], Test Losses: mse: 4.3846, mae: 0.9028, huber: 0.6515, swd: 3.2219, target_std: 4.1272
      Epoch 14 composite train-obj: 0.722757
            Val objective improved 0.6798 → 0.6649, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 4.8240, mae: 0.9781, huber: 0.7109, swd: 3.3929, target_std: 3.9390
    Epoch [15/50], Val Losses: mse: 4.7618, mae: 0.9521, huber: 0.6900, swd: 3.4991, target_std: 3.9670
    Epoch [15/50], Test Losses: mse: 4.8150, mae: 0.9729, huber: 0.7034, swd: 3.4640, target_std: 4.1272
      Epoch 15 composite train-obj: 0.710871
            No improvement (0.6900), counter 1/5
    Epoch [16/50], Train Losses: mse: 4.8133, mae: 0.9726, huber: 0.7060, swd: 3.3815, target_std: 3.9380
    Epoch [16/50], Val Losses: mse: 4.9348, mae: 0.9593, huber: 0.7169, swd: 3.6295, target_std: 3.9670
    Epoch [16/50], Test Losses: mse: 4.6959, mae: 0.9566, huber: 0.7071, swd: 3.4672, target_std: 4.1272
      Epoch 16 composite train-obj: 0.706041
            No improvement (0.7169), counter 2/5
    Epoch [17/50], Train Losses: mse: 5.0261, mae: 0.9936, huber: 0.7269, swd: 3.4967, target_std: 3.9434
    Epoch [17/50], Val Losses: mse: 4.7060, mae: 1.0409, huber: 0.7485, swd: 3.5861, target_std: 3.9670
    Epoch [17/50], Test Losses: mse: 4.5910, mae: 1.0698, huber: 0.7681, swd: 3.4319, target_std: 4.1272
      Epoch 17 composite train-obj: 0.726921
            No improvement (0.7485), counter 3/5
    Epoch [18/50], Train Losses: mse: 4.6825, mae: 0.9609, huber: 0.6952, swd: 3.3094, target_std: 3.9405
    Epoch [18/50], Val Losses: mse: 5.3270, mae: 1.0173, huber: 0.7435, swd: 3.7203, target_std: 3.9670
    Epoch [18/50], Test Losses: mse: 5.2759, mae: 1.0652, huber: 0.7791, swd: 3.6750, target_std: 4.1272
      Epoch 18 composite train-obj: 0.695165
            No improvement (0.7435), counter 4/5
    Epoch [19/50], Train Losses: mse: 4.6438, mae: 0.9406, huber: 0.6795, swd: 3.2972, target_std: 3.9387
    Epoch [19/50], Val Losses: mse: 4.7065, mae: 0.9293, huber: 0.6842, swd: 3.3921, target_std: 3.9670
    Epoch [19/50], Test Losses: mse: 4.7439, mae: 0.9643, huber: 0.7082, swd: 3.2452, target_std: 4.1272
      Epoch 19 composite train-obj: 0.679468
    Epoch [19/50], Test Losses: mse: 4.3846, mae: 0.9028, huber: 0.6515, swd: 3.2219, target_std: 4.1272
    Best round's Test MSE: 4.3846, MAE: 0.9028, SWD: 3.2219
    Best round's Validation MSE: 4.6560, MAE: 0.9119
    Best round's Test verification MSE : 4.3846, MAE: 0.9028, SWD: 3.2219
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 26.6613, mae: 2.7544, huber: 2.3995, swd: 14.0500, target_std: 3.9419
    Epoch [1/50], Val Losses: mse: 19.1208, mae: 1.9294, huber: 1.6113, swd: 14.2914, target_std: 3.9670
    Epoch [1/50], Test Losses: mse: 16.1805, mae: 1.8853, huber: 1.5585, swd: 11.3190, target_std: 4.1272
      Epoch 1 composite train-obj: 2.399531
            Val objective improved inf → 1.6113, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 16.4182, mae: 1.8544, huber: 1.5304, swd: 11.7201, target_std: 3.9386
    Epoch [2/50], Val Losses: mse: 17.2280, mae: 1.8099, huber: 1.5047, swd: 13.2335, target_std: 3.9670
    Epoch [2/50], Test Losses: mse: 14.4613, mae: 1.7436, huber: 1.4294, swd: 10.5228, target_std: 4.1272
      Epoch 2 composite train-obj: 1.530430
            Val objective improved 1.6113 → 1.5047, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 13.0903, mae: 1.7234, huber: 1.4039, swd: 9.0640, target_std: 3.9459
    Epoch [3/50], Val Losses: mse: 12.4984, mae: 1.6816, huber: 1.3780, swd: 8.8876, target_std: 3.9670
    Epoch [3/50], Test Losses: mse: 10.8416, mae: 1.6143, huber: 1.3024, swd: 7.2321, target_std: 4.1272
      Epoch 3 composite train-obj: 1.403890
            Val objective improved 1.5047 → 1.3780, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 10.9413, mae: 1.6122, huber: 1.2992, swd: 7.2535, target_std: 3.9361
    Epoch [4/50], Val Losses: mse: 10.5417, mae: 1.5371, huber: 1.2407, swd: 7.3487, target_std: 3.9670
    Epoch [4/50], Test Losses: mse: 9.8404, mae: 1.5162, huber: 1.2108, swd: 6.3755, target_std: 4.1272
      Epoch 4 composite train-obj: 1.299170
            Val objective improved 1.3780 → 1.2407, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 8.7461, mae: 1.4267, huber: 1.1210, swd: 5.8454, target_std: 3.9418
    Epoch [5/50], Val Losses: mse: 10.9094, mae: 1.5038, huber: 1.2100, swd: 7.4725, target_std: 3.9670
    Epoch [5/50], Test Losses: mse: 11.1305, mae: 1.6269, huber: 1.3187, swd: 6.9771, target_std: 4.1272
      Epoch 5 composite train-obj: 1.121033
            Val objective improved 1.2407 → 1.2100, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.4530, mae: 1.2475, huber: 0.9510, swd: 5.1408, target_std: 3.9408
    Epoch [6/50], Val Losses: mse: 7.8158, mae: 1.1949, huber: 0.9165, swd: 5.3121, target_std: 3.9670
    Epoch [6/50], Test Losses: mse: 7.3903, mae: 1.2524, huber: 0.9594, swd: 4.7392, target_std: 4.1272
      Epoch 6 composite train-obj: 0.950978
            Val objective improved 1.2100 → 0.9165, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.9298, mae: 1.1279, huber: 0.8398, swd: 4.0727, target_std: 3.9430
    Epoch [7/50], Val Losses: mse: 4.9645, mae: 0.9526, huber: 0.6935, swd: 3.7487, target_std: 3.9670
    Epoch [7/50], Test Losses: mse: 5.2172, mae: 0.9849, huber: 0.7195, swd: 3.6321, target_std: 4.1272
      Epoch 7 composite train-obj: 0.839813
            Val objective improved 0.9165 → 0.6935, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.3233, mae: 1.0446, huber: 0.7658, swd: 3.7198, target_std: 3.9422
    Epoch [8/50], Val Losses: mse: 4.9032, mae: 0.9262, huber: 0.6763, swd: 3.6352, target_std: 3.9670
    Epoch [8/50], Test Losses: mse: 4.9639, mae: 0.9455, huber: 0.6914, swd: 3.4913, target_std: 4.1272
      Epoch 8 composite train-obj: 0.765782
            Val objective improved 0.6935 → 0.6763, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 5.1860, mae: 1.0207, huber: 0.7459, swd: 3.6587, target_std: 3.9400
    Epoch [9/50], Val Losses: mse: 5.3661, mae: 1.0138, huber: 0.7601, swd: 3.8548, target_std: 3.9670
    Epoch [9/50], Test Losses: mse: 5.1275, mae: 1.0303, huber: 0.7669, swd: 3.6429, target_std: 4.1272
      Epoch 9 composite train-obj: 0.745902
            No improvement (0.7601), counter 1/5
    Epoch [10/50], Train Losses: mse: 5.1118, mae: 1.0017, huber: 0.7305, swd: 3.6344, target_std: 3.9486
    Epoch [10/50], Val Losses: mse: 5.1937, mae: 1.1493, huber: 0.8501, swd: 3.6356, target_std: 3.9670
    Epoch [10/50], Test Losses: mse: 5.0644, mae: 1.1645, huber: 0.8591, swd: 3.4233, target_std: 4.1272
      Epoch 10 composite train-obj: 0.730452
            No improvement (0.8501), counter 2/5
    Epoch [11/50], Train Losses: mse: 5.0348, mae: 0.9907, huber: 0.7211, swd: 3.6016, target_std: 3.9325
    Epoch [11/50], Val Losses: mse: 5.8304, mae: 1.1694, huber: 0.8704, swd: 3.8451, target_std: 3.9670
    Epoch [11/50], Test Losses: mse: 5.9716, mae: 1.1962, huber: 0.8889, swd: 3.7217, target_std: 4.1272
      Epoch 11 composite train-obj: 0.721055
            No improvement (0.8704), counter 3/5
    Epoch [12/50], Train Losses: mse: 5.7671, mae: 1.0931, huber: 0.8139, swd: 3.9437, target_std: 3.9336
    Epoch [12/50], Val Losses: mse: 5.1809, mae: 0.9417, huber: 0.6859, swd: 3.6888, target_std: 3.9670
    Epoch [12/50], Test Losses: mse: 4.7652, mae: 0.9410, huber: 0.6757, swd: 3.4081, target_std: 4.1272
      Epoch 12 composite train-obj: 0.813949
            No improvement (0.6859), counter 4/5
    Epoch [13/50], Train Losses: mse: 4.8532, mae: 0.9683, huber: 0.7015, swd: 3.4634, target_std: 3.9371
    Epoch [13/50], Val Losses: mse: 5.0361, mae: 1.1120, huber: 0.8127, swd: 3.6969, target_std: 3.9670
    Epoch [13/50], Test Losses: mse: 4.9638, mae: 1.1529, huber: 0.8420, swd: 3.5595, target_std: 4.1272
      Epoch 13 composite train-obj: 0.701522
    Epoch [13/50], Test Losses: mse: 4.9639, mae: 0.9455, huber: 0.6914, swd: 3.4913, target_std: 4.1272
    Best round's Test MSE: 4.9639, MAE: 0.9455, SWD: 3.4913
    Best round's Validation MSE: 4.9032, MAE: 0.9262
    Best round's Test verification MSE : 4.9639, MAE: 0.9455, SWD: 3.4913
    
    ==================================================
    Experiment Summary (TimeMixer_rossler_seq96_pred336_20250430_1253)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 4.3988 ± 0.4557
      mae: 0.8863 ± 0.0563
      huber: 0.6342 ± 0.0552
      swd: 3.2065 ± 0.2390
      target_std: 4.1272 ± 0.0000
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 4.5897 ± 0.2869
      mae: 0.8897 ± 0.0419
      huber: 0.6428 ± 0.0396
      swd: 3.3905 ± 0.2017
      target_std: 3.9670 ± 0.0000
      count: 38.0000 ± 0.0000
    ==================================================
    
    Experiment complete: TimeMixer_rossler_seq96_pred336_20250430_1253
    Model: TimeMixer
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['rossler']['channels'],
    enc_in=data_mgr.datasets['rossler']['channels'],
    dec_in=data_mgr.datasets['rossler']['channels'],
    c_out=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 279
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 20.5072, mae: 2.6311, huber: 2.2693, swd: 10.1608, ept: 369.4426
    Epoch [1/50], Val Losses: mse: 15.3772, mae: 2.1039, huber: 1.7665, swd: 9.6285, ept: 490.0199
    Epoch [1/50], Test Losses: mse: 13.6781, mae: 1.9732, huber: 1.6341, swd: 8.2152, ept: 477.4090
      Epoch 1 composite train-obj: 2.269327
            Val objective improved inf → 1.7665, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.5444, mae: 1.9219, huber: 1.5823, swd: 6.4606, ept: 484.9978
    Epoch [2/50], Val Losses: mse: 10.0956, mae: 1.7631, huber: 1.4405, swd: 5.7514, ept: 517.5613
    Epoch [2/50], Test Losses: mse: 8.7682, mae: 1.6303, huber: 1.3037, swd: 4.8617, ept: 522.5070
      Epoch 2 composite train-obj: 1.582320
            Val objective improved 1.7665 → 1.4405, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.6700, mae: 1.6580, huber: 1.3270, swd: 4.8182, ept: 525.7425
    Epoch [3/50], Val Losses: mse: 10.3752, mae: 1.9150, huber: 1.5634, swd: 6.0138, ept: 509.1049
    Epoch [3/50], Test Losses: mse: 8.9060, mae: 1.7774, huber: 1.4223, swd: 5.0073, ept: 510.8286
      Epoch 3 composite train-obj: 1.327010
            No improvement (1.5634), counter 1/5
    Epoch [4/50], Train Losses: mse: 8.5258, mae: 1.6049, huber: 1.2780, swd: 4.9304, ept: 534.7003
    Epoch [4/50], Val Losses: mse: 8.1813, mae: 1.5297, huber: 1.2232, swd: 4.8811, ept: 538.0265
    Epoch [4/50], Test Losses: mse: 6.8992, mae: 1.3781, huber: 1.0710, swd: 4.0706, ept: 552.2372
      Epoch 4 composite train-obj: 1.278032
            Val objective improved 1.4405 → 1.2232, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.4826, mae: 1.4886, huber: 1.1710, swd: 4.3256, ept: 552.2882
    Epoch [5/50], Val Losses: mse: 7.9884, mae: 1.5061, huber: 1.1939, swd: 4.8701, ept: 548.8801
    Epoch [5/50], Test Losses: mse: 6.8142, mae: 1.3725, huber: 1.0588, swd: 4.1154, ept: 560.2356
      Epoch 5 composite train-obj: 1.171015
            Val objective improved 1.2232 → 1.1939, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.2924, mae: 1.4602, huber: 1.1452, swd: 4.2723, ept: 558.0966
    Epoch [6/50], Val Losses: mse: 8.0360, mae: 1.5372, huber: 1.2230, swd: 4.7449, ept: 546.7603
    Epoch [6/50], Test Losses: mse: 6.8546, mae: 1.4008, huber: 1.0851, swd: 3.9892, ept: 557.7733
      Epoch 6 composite train-obj: 1.145186
            No improvement (1.2230), counter 1/5
    Epoch [7/50], Train Losses: mse: 7.1345, mae: 1.4322, huber: 1.1203, swd: 4.2032, ept: 563.7731
    Epoch [7/50], Val Losses: mse: 7.8822, mae: 1.5072, huber: 1.1940, swd: 4.6538, ept: 559.5087
    Epoch [7/50], Test Losses: mse: 6.4608, mae: 1.3538, huber: 1.0399, swd: 3.8285, ept: 572.5053
      Epoch 7 composite train-obj: 1.120293
            No improvement (1.1940), counter 2/5
    Epoch [8/50], Train Losses: mse: 6.9990, mae: 1.4112, huber: 1.1018, swd: 4.1444, ept: 566.9057
    Epoch [8/50], Val Losses: mse: 7.6164, mae: 1.4283, huber: 1.1327, swd: 4.6841, ept: 564.6691
    Epoch [8/50], Test Losses: mse: 6.6042, mae: 1.2918, huber: 0.9956, swd: 4.0256, ept: 574.5072
      Epoch 8 composite train-obj: 1.101821
            Val objective improved 1.1939 → 1.1327, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 6.9484, mae: 1.4001, huber: 1.0931, swd: 4.1421, ept: 570.0480
    Epoch [9/50], Val Losses: mse: 7.5423, mae: 1.4250, huber: 1.1314, swd: 4.6422, ept: 560.3998
    Epoch [9/50], Test Losses: mse: 6.1464, mae: 1.2630, huber: 0.9679, swd: 3.7944, ept: 575.7554
      Epoch 9 composite train-obj: 1.093057
            Val objective improved 1.1327 → 1.1314, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 6.8547, mae: 1.3825, huber: 1.0778, swd: 4.0951, ept: 572.9213
    Epoch [10/50], Val Losses: mse: 7.4078, mae: 1.4042, huber: 1.1158, swd: 4.5864, ept: 570.2655
    Epoch [10/50], Test Losses: mse: 6.2130, mae: 1.2650, huber: 0.9760, swd: 3.8402, ept: 579.3246
      Epoch 10 composite train-obj: 1.077782
            Val objective improved 1.1314 → 1.1158, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 6.7952, mae: 1.3744, huber: 1.0710, swd: 4.0623, ept: 575.2144
    Epoch [11/50], Val Losses: mse: 7.2350, mae: 1.3930, huber: 1.0998, swd: 4.4915, ept: 570.6702
    Epoch [11/50], Test Losses: mse: 5.9711, mae: 1.2349, huber: 0.9422, swd: 3.7052, ept: 582.5455
      Epoch 11 composite train-obj: 1.071010
            Val objective improved 1.1158 → 1.0998, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 6.7810, mae: 1.3680, huber: 1.0665, swd: 4.0561, ept: 575.1680
    Epoch [12/50], Val Losses: mse: 7.1298, mae: 1.4938, huber: 1.1730, swd: 4.4954, ept: 583.8852
    Epoch [12/50], Test Losses: mse: 5.7918, mae: 1.3366, huber: 1.0141, swd: 3.6673, ept: 583.8757
      Epoch 12 composite train-obj: 1.066474
            No improvement (1.1730), counter 1/5
    Epoch [13/50], Train Losses: mse: 6.7081, mae: 1.3658, huber: 1.0634, swd: 4.0225, ept: 576.6491
    Epoch [13/50], Val Losses: mse: 7.6481, mae: 1.4578, huber: 1.1552, swd: 4.5450, ept: 570.7884
    Epoch [13/50], Test Losses: mse: 6.4267, mae: 1.3114, huber: 1.0092, swd: 3.8426, ept: 579.7810
      Epoch 13 composite train-obj: 1.063376
            No improvement (1.1552), counter 2/5
    Epoch [14/50], Train Losses: mse: 6.6722, mae: 1.3522, huber: 1.0523, swd: 4.0042, ept: 579.2473
    Epoch [14/50], Val Losses: mse: 7.1739, mae: 1.4336, huber: 1.1272, swd: 4.4220, ept: 572.5949
    Epoch [14/50], Test Losses: mse: 6.2584, mae: 1.3104, huber: 1.0011, swd: 3.7597, ept: 579.1899
      Epoch 14 composite train-obj: 1.052306
            No improvement (1.1272), counter 3/5
    Epoch [15/50], Train Losses: mse: 6.6289, mae: 1.3482, huber: 1.0490, swd: 3.9717, ept: 580.2128
    Epoch [15/50], Val Losses: mse: 7.4984, mae: 1.4343, huber: 1.1312, swd: 4.6442, ept: 564.2714
    Epoch [15/50], Test Losses: mse: 6.3946, mae: 1.3083, huber: 1.0035, swd: 3.9977, ept: 574.5091
      Epoch 15 composite train-obj: 1.048986
            No improvement (1.1312), counter 4/5
    Epoch [16/50], Train Losses: mse: 6.5904, mae: 1.3417, huber: 1.0431, swd: 3.9606, ept: 581.0947
    Epoch [16/50], Val Losses: mse: 7.4176, mae: 1.4591, huber: 1.1500, swd: 4.3927, ept: 572.0977
    Epoch [16/50], Test Losses: mse: 7.1858, mae: 1.3541, huber: 1.0456, swd: 4.1275, ept: 579.1732
      Epoch 16 composite train-obj: 1.043077
    Epoch [16/50], Test Losses: mse: 5.9711, mae: 1.2349, huber: 0.9422, swd: 3.7052, ept: 582.5455
    Best round's Test MSE: 5.9711, MAE: 1.2349, SWD: 3.7052
    Best round's Validation MSE: 7.2350, MAE: 1.3930, SWD: 4.4915
    Best round's Test verification MSE : 5.9711, MAE: 1.2349, SWD: 3.7052
    Time taken: 137.12 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 22.1163, mae: 2.7116, huber: 2.3485, swd: 10.7652, ept: 368.0642
    Epoch [1/50], Val Losses: mse: 17.7930, mae: 2.3419, huber: 1.9961, swd: 10.3050, ept: 462.8886
    Epoch [1/50], Test Losses: mse: 15.8322, mae: 2.2085, huber: 1.8614, swd: 8.7269, ept: 441.2836
      Epoch 1 composite train-obj: 2.348509
            Val objective improved inf → 1.9961, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 14.7827, mae: 2.0584, huber: 1.7163, swd: 8.7137, ept: 474.7230
    Epoch [2/50], Val Losses: mse: 15.4962, mae: 2.1577, huber: 1.8145, swd: 9.2819, ept: 491.8875
    Epoch [2/50], Test Losses: mse: 13.6753, mae: 1.9991, huber: 1.6545, swd: 7.7818, ept: 492.7351
      Epoch 2 composite train-obj: 1.716317
            Val objective improved 1.9961 → 1.8145, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.6144, mae: 1.9474, huber: 1.6082, swd: 7.2201, ept: 492.4029
    Epoch [3/50], Val Losses: mse: 14.3489, mae: 1.9920, huber: 1.6589, swd: 8.8253, ept: 502.3614
    Epoch [3/50], Test Losses: mse: 12.4987, mae: 1.8183, huber: 1.4844, swd: 7.3825, ept: 511.5720
      Epoch 3 composite train-obj: 1.608154
            Val objective improved 1.8145 → 1.6589, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 10.8943, mae: 1.8357, huber: 1.4996, swd: 5.9949, ept: 506.4706
    Epoch [4/50], Val Losses: mse: 12.4738, mae: 1.8117, huber: 1.4873, swd: 7.7994, ept: 513.1206
    Epoch [4/50], Test Losses: mse: 11.7672, mae: 1.7346, huber: 1.4051, swd: 6.9510, ept: 516.7939
      Epoch 4 composite train-obj: 1.499597
            Val objective improved 1.6589 → 1.4873, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 9.2095, mae: 1.6830, huber: 1.3519, swd: 5.0987, ept: 527.3210
    Epoch [5/50], Val Losses: mse: 9.8217, mae: 1.6901, huber: 1.3734, swd: 5.9064, ept: 527.5883
    Epoch [5/50], Test Losses: mse: 9.0593, mae: 1.5935, huber: 1.2741, swd: 5.2606, ept: 527.6441
      Epoch 5 composite train-obj: 1.351943
            Val objective improved 1.4873 → 1.3734, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 8.4543, mae: 1.6050, huber: 1.2787, swd: 4.7389, ept: 538.5630
    Epoch [6/50], Val Losses: mse: 8.8774, mae: 1.6491, huber: 1.3223, swd: 5.2502, ept: 542.6857
    Epoch [6/50], Test Losses: mse: 8.2081, mae: 1.5339, huber: 1.2058, swd: 4.6919, ept: 547.8966
      Epoch 6 composite train-obj: 1.278698
            Val objective improved 1.3734 → 1.3223, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.9968, mae: 1.5584, huber: 1.2355, swd: 4.4738, ept: 543.0457
    Epoch [7/50], Val Losses: mse: 8.6756, mae: 1.6553, huber: 1.3265, swd: 4.9992, ept: 539.2710
    Epoch [7/50], Test Losses: mse: 8.1149, mae: 1.5620, huber: 1.2321, swd: 4.4655, ept: 540.9527
      Epoch 7 composite train-obj: 1.235477
            No improvement (1.3265), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.7224, mae: 1.5235, huber: 1.2033, swd: 4.3564, ept: 548.0183
    Epoch [8/50], Val Losses: mse: 7.6655, mae: 1.5674, huber: 1.2410, swd: 4.6468, ept: 566.6426
    Epoch [8/50], Test Losses: mse: 6.7950, mae: 1.4359, huber: 1.1085, swd: 3.9679, ept: 565.6322
      Epoch 8 composite train-obj: 1.203311
            Val objective improved 1.3223 → 1.2410, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.4669, mae: 1.4900, huber: 1.1724, swd: 4.2787, ept: 554.2879
    Epoch [9/50], Val Losses: mse: 8.5987, mae: 1.5880, huber: 1.2747, swd: 5.1153, ept: 551.7425
    Epoch [9/50], Test Losses: mse: 8.2176, mae: 1.5174, huber: 1.2011, swd: 4.7690, ept: 552.3492
      Epoch 9 composite train-obj: 1.172354
            No improvement (1.2747), counter 1/5
    Epoch [10/50], Train Losses: mse: 7.4171, mae: 1.4828, huber: 1.1664, swd: 4.2350, ept: 554.7284
    Epoch [10/50], Val Losses: mse: 7.6001, mae: 1.4842, huber: 1.1830, swd: 4.5939, ept: 553.4730
    Epoch [10/50], Test Losses: mse: 6.8296, mae: 1.3521, huber: 1.0496, swd: 4.0181, ept: 563.5899
      Epoch 10 composite train-obj: 1.166370
            Val objective improved 1.2410 → 1.1830, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 7.1355, mae: 1.4401, huber: 1.1284, swd: 4.1362, ept: 562.0296
    Epoch [11/50], Val Losses: mse: 8.1684, mae: 1.4914, huber: 1.1900, swd: 4.8088, ept: 552.8716
    Epoch [11/50], Test Losses: mse: 7.4790, mae: 1.3990, huber: 1.0948, swd: 4.3116, ept: 559.5528
      Epoch 11 composite train-obj: 1.128361
            No improvement (1.1900), counter 1/5
    Epoch [12/50], Train Losses: mse: 7.0743, mae: 1.4250, huber: 1.1151, swd: 4.1223, ept: 566.2696
    Epoch [12/50], Val Losses: mse: 7.5482, mae: 1.4843, huber: 1.1782, swd: 4.5083, ept: 568.8360
    Epoch [12/50], Test Losses: mse: 6.3492, mae: 1.3275, huber: 1.0208, swd: 3.7360, ept: 579.0533
      Epoch 12 composite train-obj: 1.115084
            Val objective improved 1.1830 → 1.1782, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 6.9181, mae: 1.4049, huber: 1.0970, swd: 4.0452, ept: 569.4497
    Epoch [13/50], Val Losses: mse: 7.4687, mae: 1.4576, huber: 1.1553, swd: 4.4949, ept: 569.0944
    Epoch [13/50], Test Losses: mse: 6.9158, mae: 1.3340, huber: 1.0300, swd: 4.0280, ept: 576.0664
      Epoch 13 composite train-obj: 1.097023
            Val objective improved 1.1782 → 1.1553, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 6.9486, mae: 1.4001, huber: 1.0932, swd: 4.0640, ept: 570.7279
    Epoch [14/50], Val Losses: mse: 8.6839, mae: 1.6888, huber: 1.3661, swd: 4.6400, ept: 531.7754
    Epoch [14/50], Test Losses: mse: 7.1881, mae: 1.5097, huber: 1.1871, swd: 3.7685, ept: 544.8695
      Epoch 14 composite train-obj: 1.093232
            No improvement (1.3661), counter 1/5
    Epoch [15/50], Train Losses: mse: 6.9466, mae: 1.4044, huber: 1.0972, swd: 4.0210, ept: 570.6231
    Epoch [15/50], Val Losses: mse: 7.2174, mae: 1.4082, huber: 1.1193, swd: 4.4016, ept: 577.7588
    Epoch [15/50], Test Losses: mse: 5.9191, mae: 1.2417, huber: 0.9515, swd: 3.6207, ept: 588.0714
      Epoch 15 composite train-obj: 1.097248
            Val objective improved 1.1553 → 1.1193, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 6.7234, mae: 1.3691, huber: 1.0658, swd: 3.9771, ept: 576.1029
    Epoch [16/50], Val Losses: mse: 7.6351, mae: 1.5387, huber: 1.2301, swd: 4.5463, ept: 557.1271
    Epoch [16/50], Test Losses: mse: 6.2956, mae: 1.3773, huber: 1.0675, swd: 3.7302, ept: 563.2846
      Epoch 16 composite train-obj: 1.065848
            No improvement (1.2301), counter 1/5
    Epoch [17/50], Train Losses: mse: 6.6896, mae: 1.3674, huber: 1.0642, swd: 3.9475, ept: 576.3788
    Epoch [17/50], Val Losses: mse: 7.3278, mae: 1.4840, huber: 1.1789, swd: 4.4024, ept: 568.8796
    Epoch [17/50], Test Losses: mse: 5.8821, mae: 1.3125, huber: 1.0041, swd: 3.5684, ept: 581.6690
      Epoch 17 composite train-obj: 1.064207
            No improvement (1.1789), counter 2/5
    Epoch [18/50], Train Losses: mse: 6.6061, mae: 1.3571, huber: 1.0546, swd: 3.9085, ept: 578.3329
    Epoch [18/50], Val Losses: mse: 7.2264, mae: 1.4593, huber: 1.1547, swd: 4.3886, ept: 568.9380
    Epoch [18/50], Test Losses: mse: 5.8364, mae: 1.2869, huber: 0.9811, swd: 3.5530, ept: 582.4296
      Epoch 18 composite train-obj: 1.054605
            No improvement (1.1547), counter 3/5
    Epoch [19/50], Train Losses: mse: 6.6705, mae: 1.3685, huber: 1.0644, swd: 3.9457, ept: 576.3776
    Epoch [19/50], Val Losses: mse: 8.4484, mae: 1.6306, huber: 1.3240, swd: 4.7377, ept: 536.9182
    Epoch [19/50], Test Losses: mse: 7.0339, mae: 1.4629, huber: 1.1564, swd: 3.9049, ept: 546.7461
      Epoch 19 composite train-obj: 1.064412
            No improvement (1.3240), counter 4/5
    Epoch [20/50], Train Losses: mse: 6.6774, mae: 1.3667, huber: 1.0645, swd: 3.9352, ept: 578.0158
    Epoch [20/50], Val Losses: mse: 7.4524, mae: 1.4436, huber: 1.1480, swd: 4.4096, ept: 571.8599
    Epoch [20/50], Test Losses: mse: 5.9883, mae: 1.2641, huber: 0.9704, swd: 3.5854, ept: 586.0034
      Epoch 20 composite train-obj: 1.064522
    Epoch [20/50], Test Losses: mse: 5.9191, mae: 1.2417, huber: 0.9515, swd: 3.6207, ept: 588.0714
    Best round's Test MSE: 5.9191, MAE: 1.2417, SWD: 3.6207
    Best round's Validation MSE: 7.2174, MAE: 1.4082, SWD: 4.4016
    Best round's Test verification MSE : 5.9191, MAE: 1.2417, SWD: 3.6207
    Time taken: 172.88 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 20.3647, mae: 2.5744, huber: 2.2167, swd: 10.1017, ept: 384.2166
    Epoch [1/50], Val Losses: mse: 16.4763, mae: 2.1482, huber: 1.8054, swd: 10.4922, ept: 492.0773
    Epoch [1/50], Test Losses: mse: 14.5516, mae: 1.9920, huber: 1.6498, swd: 8.8030, ept: 492.1897
      Epoch 1 composite train-obj: 2.216733
            Val objective improved inf → 1.8054, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 13.3799, mae: 1.9920, huber: 1.6527, swd: 7.9313, ept: 488.2731
    Epoch [2/50], Val Losses: mse: 13.0853, mae: 2.0228, huber: 1.6922, swd: 7.6378, ept: 485.3812
    Epoch [2/50], Test Losses: mse: 11.2154, mae: 1.8301, huber: 1.4992, swd: 6.3736, ept: 499.9999
      Epoch 2 composite train-obj: 1.652719
            Val objective improved 1.8054 → 1.6922, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.2547, mae: 1.8820, huber: 1.5448, swd: 6.3681, ept: 500.0521
    Epoch [3/50], Val Losses: mse: 11.2969, mae: 1.8852, huber: 1.5596, swd: 6.6444, ept: 502.0401
    Epoch [3/50], Test Losses: mse: 10.1543, mae: 1.7223, huber: 1.3975, swd: 5.7200, ept: 513.6026
      Epoch 3 composite train-obj: 1.544783
            Val objective improved 1.6922 → 1.5596, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 9.6082, mae: 1.7458, huber: 1.4133, swd: 5.4302, ept: 516.5274
    Epoch [4/50], Val Losses: mse: 10.0917, mae: 1.7382, huber: 1.4156, swd: 6.1489, ept: 519.8339
    Epoch [4/50], Test Losses: mse: 8.9782, mae: 1.5941, huber: 1.2735, swd: 5.2994, ept: 532.5833
      Epoch 4 composite train-obj: 1.413267
            Val objective improved 1.5596 → 1.4156, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 8.7293, mae: 1.6414, huber: 1.3145, swd: 5.0454, ept: 534.6427
    Epoch [5/50], Val Losses: mse: 8.9854, mae: 1.6508, huber: 1.3328, swd: 5.5142, ept: 536.0688
    Epoch [5/50], Test Losses: mse: 8.0801, mae: 1.5223, huber: 1.2041, swd: 4.8736, ept: 541.9754
      Epoch 5 composite train-obj: 1.314486
            Val objective improved 1.4156 → 1.3328, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 8.3933, mae: 1.5899, huber: 1.2668, swd: 4.9246, ept: 541.8444
    Epoch [6/50], Val Losses: mse: 9.2296, mae: 1.7842, huber: 1.4450, swd: 5.4667, ept: 530.2498
    Epoch [6/50], Test Losses: mse: 8.8263, mae: 1.6814, huber: 1.3416, swd: 4.9587, ept: 535.3638
      Epoch 6 composite train-obj: 1.266771
            No improvement (1.4450), counter 1/5
    Epoch [7/50], Train Losses: mse: 7.7751, mae: 1.5311, huber: 1.2123, swd: 4.5336, ept: 546.9786
    Epoch [7/50], Val Losses: mse: 8.4060, mae: 1.5448, huber: 1.2433, swd: 5.1291, ept: 542.4510
    Epoch [7/50], Test Losses: mse: 7.6246, mae: 1.4222, huber: 1.1192, swd: 4.5442, ept: 555.0702
      Epoch 7 composite train-obj: 1.212316
            Val objective improved 1.3328 → 1.2433, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 7.4567, mae: 1.4857, huber: 1.1711, swd: 4.3842, ept: 553.7915
    Epoch [8/50], Val Losses: mse: 8.1328, mae: 1.5496, huber: 1.2381, swd: 4.8781, ept: 546.1755
    Epoch [8/50], Test Losses: mse: 6.8587, mae: 1.3915, huber: 1.0796, swd: 4.0711, ept: 563.4093
      Epoch 8 composite train-obj: 1.171131
            Val objective improved 1.2433 → 1.2381, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.2906, mae: 1.4622, huber: 1.1494, swd: 4.3117, ept: 557.9858
    Epoch [9/50], Val Losses: mse: 8.0356, mae: 1.4818, huber: 1.1874, swd: 4.9476, ept: 552.4455
    Epoch [9/50], Test Losses: mse: 7.0581, mae: 1.3528, huber: 1.0586, swd: 4.3021, ept: 562.1030
      Epoch 9 composite train-obj: 1.149409
            Val objective improved 1.2381 → 1.1874, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.1449, mae: 1.4405, huber: 1.1300, swd: 4.2634, ept: 561.4881
    Epoch [10/50], Val Losses: mse: 8.4939, mae: 1.5975, huber: 1.2788, swd: 4.8759, ept: 550.6211
    Epoch [10/50], Test Losses: mse: 6.9296, mae: 1.4218, huber: 1.1030, swd: 3.9620, ept: 564.9445
      Epoch 10 composite train-obj: 1.130022
            No improvement (1.2788), counter 1/5
    Epoch [11/50], Train Losses: mse: 7.1085, mae: 1.4299, huber: 1.1209, swd: 4.2545, ept: 563.7072
    Epoch [11/50], Val Losses: mse: 7.5358, mae: 1.5541, huber: 1.2326, swd: 4.7163, ept: 572.1005
    Epoch [11/50], Test Losses: mse: 6.4248, mae: 1.4007, huber: 1.0801, swd: 3.9526, ept: 573.2715
      Epoch 11 composite train-obj: 1.120942
            No improvement (1.2326), counter 2/5
    Epoch [12/50], Train Losses: mse: 6.9892, mae: 1.4187, huber: 1.1101, swd: 4.2076, ept: 566.8418
    Epoch [12/50], Val Losses: mse: 9.9307, mae: 1.8584, huber: 1.5274, swd: 5.1584, ept: 500.6732
    Epoch [12/50], Test Losses: mse: 8.9353, mae: 1.7166, huber: 1.3849, swd: 4.3598, ept: 513.2729
      Epoch 12 composite train-obj: 1.110054
            No improvement (1.5274), counter 3/5
    Epoch [13/50], Train Losses: mse: 7.6661, mae: 1.5088, huber: 1.1928, swd: 4.5370, ept: 553.4609
    Epoch [13/50], Val Losses: mse: 7.7291, mae: 1.5943, huber: 1.2687, swd: 4.9365, ept: 555.8373
    Epoch [13/50], Test Losses: mse: 6.6995, mae: 1.4395, huber: 1.1130, swd: 4.1987, ept: 560.2349
      Epoch 13 composite train-obj: 1.192830
            No improvement (1.2687), counter 4/5
    Epoch [14/50], Train Losses: mse: 6.9079, mae: 1.4124, huber: 1.1038, swd: 4.1701, ept: 566.7591
    Epoch [14/50], Val Losses: mse: 7.4764, mae: 1.4164, huber: 1.1260, swd: 4.6815, ept: 562.5697
    Epoch [14/50], Test Losses: mse: 6.0586, mae: 1.2474, huber: 0.9577, swd: 3.8326, ept: 578.8060
      Epoch 14 composite train-obj: 1.103782
            Val objective improved 1.1874 → 1.1260, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 6.7977, mae: 1.3799, huber: 1.0756, swd: 4.1463, ept: 572.3875
    Epoch [15/50], Val Losses: mse: 7.2991, mae: 1.4550, huber: 1.1491, swd: 4.6011, ept: 572.2004
    Epoch [15/50], Test Losses: mse: 6.1073, mae: 1.2901, huber: 0.9840, swd: 3.8439, ept: 582.7784
      Epoch 15 composite train-obj: 1.075556
            No improvement (1.1491), counter 1/5
    Epoch [16/50], Train Losses: mse: 6.7046, mae: 1.3684, huber: 1.0659, swd: 4.0978, ept: 575.3620
    Epoch [16/50], Val Losses: mse: 7.4000, mae: 1.4169, huber: 1.1291, swd: 4.5847, ept: 567.5607
    Epoch [16/50], Test Losses: mse: 6.0947, mae: 1.2442, huber: 0.9577, swd: 3.7773, ept: 581.4919
      Epoch 16 composite train-obj: 1.065853
            No improvement (1.1291), counter 2/5
    Epoch [17/50], Train Losses: mse: 6.6874, mae: 1.3614, huber: 1.0601, swd: 4.0908, ept: 576.6835
    Epoch [17/50], Val Losses: mse: 7.2176, mae: 1.4087, huber: 1.1161, swd: 4.5537, ept: 574.4566
    Epoch [17/50], Test Losses: mse: 5.8209, mae: 1.2183, huber: 0.9285, swd: 3.6991, ept: 586.4585
      Epoch 17 composite train-obj: 1.060062
            Val objective improved 1.1260 → 1.1161, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 6.6520, mae: 1.3527, huber: 1.0530, swd: 4.0845, ept: 578.1494
    Epoch [18/50], Val Losses: mse: 7.7672, mae: 1.4453, huber: 1.1483, swd: 4.7503, ept: 562.4707
    Epoch [18/50], Test Losses: mse: 6.0892, mae: 1.2605, huber: 0.9608, swd: 3.8007, ept: 584.0989
      Epoch 18 composite train-obj: 1.052988
            No improvement (1.1483), counter 1/5
    Epoch [19/50], Train Losses: mse: 6.6446, mae: 1.3476, huber: 1.0484, swd: 4.0728, ept: 578.8774
    Epoch [19/50], Val Losses: mse: 9.2618, mae: 1.6850, huber: 1.3576, swd: 4.8371, ept: 536.0442
    Epoch [19/50], Test Losses: mse: 8.1703, mae: 1.5561, huber: 1.2263, swd: 3.9597, ept: 550.5936
      Epoch 19 composite train-obj: 1.048407
            No improvement (1.3576), counter 2/5
    Epoch [20/50], Train Losses: mse: 7.4224, mae: 1.4524, huber: 1.1447, swd: 4.3321, ept: 564.1531
    Epoch [20/50], Val Losses: mse: 7.6107, mae: 1.5710, huber: 1.2427, swd: 4.6798, ept: 567.3261
    Epoch [20/50], Test Losses: mse: 6.3868, mae: 1.4058, huber: 1.0776, swd: 3.8705, ept: 570.5502
      Epoch 20 composite train-obj: 1.144695
            No improvement (1.2427), counter 3/5
    Epoch [21/50], Train Losses: mse: 6.7271, mae: 1.3635, huber: 1.0628, swd: 4.0974, ept: 576.0312
    Epoch [21/50], Val Losses: mse: 7.2603, mae: 1.4154, huber: 1.1192, swd: 4.5755, ept: 568.1812
    Epoch [21/50], Test Losses: mse: 5.8415, mae: 1.2340, huber: 0.9384, swd: 3.7260, ept: 585.5086
      Epoch 21 composite train-obj: 1.062775
            No improvement (1.1192), counter 4/5
    Epoch [22/50], Train Losses: mse: 6.5786, mae: 1.3349, huber: 1.0382, swd: 4.0601, ept: 580.4327
    Epoch [22/50], Val Losses: mse: 7.1183, mae: 1.4150, huber: 1.1113, swd: 4.5527, ept: 574.5026
    Epoch [22/50], Test Losses: mse: 5.9698, mae: 1.2520, huber: 0.9505, swd: 3.8195, ept: 586.4332
      Epoch 22 composite train-obj: 1.038198
            Val objective improved 1.1161 → 1.1113, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 6.5515, mae: 1.3333, huber: 1.0365, swd: 4.0407, ept: 581.7228
    Epoch [23/50], Val Losses: mse: 7.1422, mae: 1.3975, huber: 1.1042, swd: 4.5394, ept: 576.9588
    Epoch [23/50], Test Losses: mse: 5.6911, mae: 1.2092, huber: 0.9180, swd: 3.6776, ept: 591.5620
      Epoch 23 composite train-obj: 1.036496
            Val objective improved 1.1113 → 1.1042, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 6.5591, mae: 1.3322, huber: 1.0360, swd: 4.0535, ept: 580.9132
    Epoch [24/50], Val Losses: mse: 7.2934, mae: 1.3908, huber: 1.0996, swd: 4.5404, ept: 569.4085
    Epoch [24/50], Test Losses: mse: 5.7357, mae: 1.1904, huber: 0.9019, swd: 3.6475, ept: 590.3477
      Epoch 24 composite train-obj: 1.035958
            Val objective improved 1.1042 → 1.0996, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 6.5626, mae: 1.3256, huber: 1.0308, swd: 4.0550, ept: 582.5551
    Epoch [25/50], Val Losses: mse: 7.3286, mae: 1.3939, huber: 1.1012, swd: 4.5756, ept: 570.0415
    Epoch [25/50], Test Losses: mse: 5.8612, mae: 1.2202, huber: 0.9275, swd: 3.7287, ept: 589.2712
      Epoch 25 composite train-obj: 1.030828
            No improvement (1.1012), counter 1/5
    Epoch [26/50], Train Losses: mse: 6.5386, mae: 1.3216, huber: 1.0276, swd: 4.0337, ept: 581.8328
    Epoch [26/50], Val Losses: mse: 7.1714, mae: 1.4734, huber: 1.1623, swd: 4.4668, ept: 579.2656
    Epoch [26/50], Test Losses: mse: 5.7572, mae: 1.2898, huber: 0.9806, swd: 3.6033, ept: 587.8683
      Epoch 26 composite train-obj: 1.027611
            No improvement (1.1623), counter 2/5
    Epoch [27/50], Train Losses: mse: 6.4314, mae: 1.3166, huber: 1.0222, swd: 3.9661, ept: 583.2227
    Epoch [27/50], Val Losses: mse: 7.1481, mae: 1.3826, huber: 1.0924, swd: 4.4798, ept: 572.8669
    Epoch [27/50], Test Losses: mse: 5.7539, mae: 1.1922, huber: 0.9047, swd: 3.6668, ept: 591.0955
      Epoch 27 composite train-obj: 1.022246
            Val objective improved 1.0996 → 1.0924, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 6.4452, mae: 1.3119, huber: 1.0189, swd: 3.9792, ept: 583.9679
    Epoch [28/50], Val Losses: mse: 6.9787, mae: 1.3584, huber: 1.0762, swd: 4.4207, ept: 576.9557
    Epoch [28/50], Test Losses: mse: 5.4099, mae: 1.1495, huber: 0.8721, swd: 3.5391, ept: 592.7410
      Epoch 28 composite train-obj: 1.018890
            Val objective improved 1.0924 → 1.0762, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 6.4693, mae: 1.3132, huber: 1.0202, swd: 3.9909, ept: 583.4145
    Epoch [29/50], Val Losses: mse: 8.2400, mae: 1.6240, huber: 1.3107, swd: 5.0076, ept: 543.5912
    Epoch [29/50], Test Losses: mse: 7.0963, mae: 1.4761, huber: 1.1606, swd: 4.2483, ept: 550.3974
      Epoch 29 composite train-obj: 1.020203
            No improvement (1.3107), counter 1/5
    Epoch [30/50], Train Losses: mse: 6.4742, mae: 1.3248, huber: 1.0295, swd: 3.9779, ept: 582.2525
    Epoch [30/50], Val Losses: mse: 7.8338, mae: 1.4914, huber: 1.1876, swd: 4.8717, ept: 554.1539
    Epoch [30/50], Test Losses: mse: 6.4464, mae: 1.3155, huber: 1.0112, swd: 3.9898, ept: 571.7576
      Epoch 30 composite train-obj: 1.029473
            No improvement (1.1876), counter 2/5
    Epoch [31/50], Train Losses: mse: 7.0022, mae: 1.3642, huber: 1.0647, swd: 4.3613, ept: 578.5362
    Epoch [31/50], Val Losses: mse: 7.4682, mae: 1.4711, huber: 1.1626, swd: 4.5969, ept: 563.0478
    Epoch [31/50], Test Losses: mse: 6.0698, mae: 1.2902, huber: 0.9823, swd: 3.7738, ept: 584.3493
      Epoch 31 composite train-obj: 1.064653
            No improvement (1.1626), counter 3/5
    Epoch [32/50], Train Losses: mse: 6.4294, mae: 1.3061, huber: 1.0134, swd: 3.9656, ept: 585.1438
    Epoch [32/50], Val Losses: mse: 7.1995, mae: 1.3802, huber: 1.0876, swd: 4.4876, ept: 570.4084
    Epoch [32/50], Test Losses: mse: 5.6800, mae: 1.1836, huber: 0.8927, swd: 3.6236, ept: 590.8839
      Epoch 32 composite train-obj: 1.013412
            No improvement (1.0876), counter 4/5
    Epoch [33/50], Train Losses: mse: 6.3572, mae: 1.2983, huber: 1.0068, swd: 3.9341, ept: 585.9084
    Epoch [33/50], Val Losses: mse: 7.3638, mae: 1.3669, huber: 1.0852, swd: 4.5950, ept: 571.8623
    Epoch [33/50], Test Losses: mse: 5.7911, mae: 1.1840, huber: 0.9049, swd: 3.7402, ept: 587.9279
      Epoch 33 composite train-obj: 1.006840
    Epoch [33/50], Test Losses: mse: 5.4099, mae: 1.1495, huber: 0.8721, swd: 3.5391, ept: 592.7410
    Best round's Test MSE: 5.4099, MAE: 1.1495, SWD: 3.5391
    Best round's Validation MSE: 6.9787, MAE: 1.3584, SWD: 4.4207
    Best round's Test verification MSE : 5.4099, MAE: 1.1495, SWD: 3.5391
    Time taken: 268.47 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_rossler_seq96_pred720_20250513_1315)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 5.7667 ± 0.2532
      mae: 1.2087 ± 0.0420
      huber: 0.9219 ± 0.0354
      swd: 3.6217 ± 0.0678
      ept: 587.7859 ± 4.1672
      count: 35.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.1437 ± 0.1169
      mae: 1.3865 ± 0.0208
      huber: 1.0984 ± 0.0176
      swd: 4.4379 ± 0.0387
      ept: 575.1282 ± 3.1693
      count: 35.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 578.52 seconds
    
    Experiment complete: TimeMixer_rossler_seq96_pred720_20250513_1315
    Model: TimeMixer
    Dataset: rossler
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
    channels=data_mgr.datasets['rossler']['channels'],
    enc_in=data_mgr.datasets['rossler']['channels'],
    dec_in=data_mgr.datasets['rossler']['channels'],
    c_out=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 9.4284, mae: 1.2200, huber: 0.9393, swd: 5.1202, ept: 73.2969
    Epoch [1/50], Val Losses: mse: 4.3067, mae: 0.7577, huber: 0.4976, swd: 2.9108, ept: 93.2728
    Epoch [1/50], Test Losses: mse: 4.1022, mae: 0.8004, huber: 0.5279, swd: 2.7313, ept: 92.9281
      Epoch 1 composite train-obj: 0.939265
            Val objective improved inf → 0.4976, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.3397, mae: 0.8584, huber: 0.6196, swd: 2.0577, ept: 73.6470
    Epoch [2/50], Val Losses: mse: 1.9330, mae: 0.5546, huber: 0.3168, swd: 1.5050, ept: 94.2436
    Epoch [2/50], Test Losses: mse: 1.9643, mae: 0.6081, huber: 0.3567, swd: 1.4897, ept: 93.7172
      Epoch 2 composite train-obj: 0.619550
            Val objective improved 0.4976 → 0.3168, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 3.6803, mae: 0.7593, huber: 0.5393, swd: 1.6329, ept: 73.7309
    Epoch [3/50], Val Losses: mse: 1.7140, mae: 0.4554, huber: 0.2522, swd: 1.3332, ept: 94.1692
    Epoch [3/50], Test Losses: mse: 1.6747, mae: 0.4977, huber: 0.2796, swd: 1.2885, ept: 93.7228
      Epoch 3 composite train-obj: 0.539310
            Val objective improved 0.3168 → 0.2522, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 3.5310, mae: 0.7255, huber: 0.5134, swd: 1.5439, ept: 73.7527
    Epoch [4/50], Val Losses: mse: 1.6196, mae: 0.3563, huber: 0.2066, swd: 1.2926, ept: 94.1230
    Epoch [4/50], Test Losses: mse: 1.5320, mae: 0.3901, huber: 0.2242, swd: 1.2184, ept: 93.8027
      Epoch 4 composite train-obj: 0.513362
            Val objective improved 0.2522 → 0.2066, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 3.4491, mae: 0.6994, huber: 0.4943, swd: 1.4892, ept: 73.7651
    Epoch [5/50], Val Losses: mse: 1.3353, mae: 0.3885, huber: 0.2023, swd: 1.0362, ept: 94.5982
    Epoch [5/50], Test Losses: mse: 1.2769, mae: 0.4241, huber: 0.2224, swd: 0.9640, ept: 94.4762
      Epoch 5 composite train-obj: 0.494296
            Val objective improved 0.2066 → 0.2023, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.3583, mae: 0.6818, huber: 0.4814, swd: 1.4220, ept: 73.8259
    Epoch [6/50], Val Losses: mse: 1.7768, mae: 0.4929, huber: 0.2714, swd: 1.4541, ept: 94.2740
    Epoch [6/50], Test Losses: mse: 1.7490, mae: 0.5423, huber: 0.3044, swd: 1.4165, ept: 93.7875
      Epoch 6 composite train-obj: 0.481376
            No improvement (0.2714), counter 1/5
    Epoch [7/50], Train Losses: mse: 3.3008, mae: 0.6678, huber: 0.4706, swd: 1.3702, ept: 73.8037
    Epoch [7/50], Val Losses: mse: 1.1779, mae: 0.3338, huber: 0.1609, swd: 0.9082, ept: 94.9003
    Epoch [7/50], Test Losses: mse: 1.1091, mae: 0.3651, huber: 0.1767, swd: 0.8475, ept: 94.6569
      Epoch 7 composite train-obj: 0.470599
            Val objective improved 0.2023 → 0.1609, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.2261, mae: 0.6560, huber: 0.4620, swd: 1.3141, ept: 73.8415
    Epoch [8/50], Val Losses: mse: 1.3898, mae: 0.3920, huber: 0.2104, swd: 1.0731, ept: 94.6014
    Epoch [8/50], Test Losses: mse: 1.2730, mae: 0.4137, huber: 0.2201, swd: 0.9715, ept: 94.4885
      Epoch 8 composite train-obj: 0.462025
            No improvement (0.2104), counter 1/5
    Epoch [9/50], Train Losses: mse: 3.1632, mae: 0.6432, huber: 0.4527, swd: 1.2617, ept: 73.8293
    Epoch [9/50], Val Losses: mse: 1.0874, mae: 0.3888, huber: 0.1876, swd: 0.8235, ept: 94.9543
    Epoch [9/50], Test Losses: mse: 1.1567, mae: 0.4326, huber: 0.2170, swd: 0.8672, ept: 94.5901
      Epoch 9 composite train-obj: 0.452724
            No improvement (0.1876), counter 2/5
    Epoch [10/50], Train Losses: mse: 3.1231, mae: 0.6406, huber: 0.4494, swd: 1.2296, ept: 73.8612
    Epoch [10/50], Val Losses: mse: 1.2826, mae: 0.4234, huber: 0.2138, swd: 0.9481, ept: 94.8036
    Epoch [10/50], Test Losses: mse: 1.3585, mae: 0.4678, huber: 0.2445, swd: 0.9845, ept: 94.5544
      Epoch 10 composite train-obj: 0.449422
            No improvement (0.2138), counter 3/5
    Epoch [11/50], Train Losses: mse: 3.0944, mae: 0.6326, huber: 0.4441, swd: 1.2010, ept: 73.8427
    Epoch [11/50], Val Losses: mse: 1.0994, mae: 0.3162, huber: 0.1654, swd: 0.8230, ept: 94.7871
    Epoch [11/50], Test Losses: mse: 1.1231, mae: 0.3470, huber: 0.1824, swd: 0.8321, ept: 94.6799
      Epoch 11 composite train-obj: 0.444095
            No improvement (0.1654), counter 4/5
    Epoch [12/50], Train Losses: mse: 3.0312, mae: 0.6262, huber: 0.4390, swd: 1.1417, ept: 73.8788
    Epoch [12/50], Val Losses: mse: 0.8772, mae: 0.2940, huber: 0.1394, swd: 0.6370, ept: 94.7833
    Epoch [12/50], Test Losses: mse: 0.8940, mae: 0.3232, huber: 0.1538, swd: 0.6341, ept: 94.7249
      Epoch 12 composite train-obj: 0.439026
            Val objective improved 0.1609 → 0.1394, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 3.0118, mae: 0.6192, huber: 0.4344, swd: 1.1271, ept: 73.8800
    Epoch [13/50], Val Losses: mse: 1.0135, mae: 0.3470, huber: 0.1670, swd: 0.7863, ept: 94.5786
    Epoch [13/50], Test Losses: mse: 0.9308, mae: 0.3671, huber: 0.1745, swd: 0.7071, ept: 94.5598
      Epoch 13 composite train-obj: 0.434442
            No improvement (0.1670), counter 1/5
    Epoch [14/50], Train Losses: mse: 3.0054, mae: 0.6235, huber: 0.4364, swd: 1.1238, ept: 73.8780
    Epoch [14/50], Val Losses: mse: 0.8265, mae: 0.3223, huber: 0.1451, swd: 0.6134, ept: 94.8946
    Epoch [14/50], Test Losses: mse: 0.7838, mae: 0.3445, huber: 0.1539, swd: 0.5665, ept: 94.9775
      Epoch 14 composite train-obj: 0.436426
            No improvement (0.1451), counter 2/5
    Epoch [15/50], Train Losses: mse: 2.9557, mae: 0.6143, huber: 0.4296, swd: 1.0699, ept: 73.9003
    Epoch [15/50], Val Losses: mse: 0.8458, mae: 0.2867, huber: 0.1332, swd: 0.6122, ept: 94.9917
    Epoch [15/50], Test Losses: mse: 0.8123, mae: 0.3098, huber: 0.1420, swd: 0.5784, ept: 94.9376
      Epoch 15 composite train-obj: 0.429551
            Val objective improved 0.1394 → 0.1332, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 2.9375, mae: 0.6092, huber: 0.4265, swd: 1.0673, ept: 73.8584
    Epoch [16/50], Val Losses: mse: 0.7092, mae: 0.2603, huber: 0.1227, swd: 0.5017, ept: 94.9932
    Epoch [16/50], Test Losses: mse: 0.6788, mae: 0.2809, huber: 0.1293, swd: 0.4668, ept: 95.0957
      Epoch 16 composite train-obj: 0.426477
            Val objective improved 0.1332 → 0.1227, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 2.9167, mae: 0.6047, huber: 0.4236, swd: 1.0435, ept: 73.8809
    Epoch [17/50], Val Losses: mse: 0.7873, mae: 0.3821, huber: 0.1735, swd: 0.5874, ept: 95.1740
    Epoch [17/50], Test Losses: mse: 0.7918, mae: 0.4064, huber: 0.1870, swd: 0.5791, ept: 95.1180
      Epoch 17 composite train-obj: 0.423578
            No improvement (0.1735), counter 1/5
    Epoch [18/50], Train Losses: mse: 2.8847, mae: 0.6066, huber: 0.4232, swd: 1.0127, ept: 73.9156
    Epoch [18/50], Val Losses: mse: 0.9045, mae: 0.4635, huber: 0.2259, swd: 0.6670, ept: 94.7134
    Epoch [18/50], Test Losses: mse: 0.9007, mae: 0.4977, huber: 0.2458, swd: 0.6554, ept: 94.8349
      Epoch 18 composite train-obj: 0.423174
            No improvement (0.2259), counter 2/5
    Epoch [19/50], Train Losses: mse: 2.8541, mae: 0.6008, huber: 0.4192, swd: 0.9820, ept: 73.8950
    Epoch [19/50], Val Losses: mse: 0.7473, mae: 0.3714, huber: 0.1651, swd: 0.5214, ept: 94.7923
    Epoch [19/50], Test Losses: mse: 0.7726, mae: 0.4037, huber: 0.1823, swd: 0.5311, ept: 94.8907
      Epoch 19 composite train-obj: 0.419170
            No improvement (0.1651), counter 3/5
    Epoch [20/50], Train Losses: mse: 2.8330, mae: 0.5964, huber: 0.4159, swd: 0.9608, ept: 73.8937
    Epoch [20/50], Val Losses: mse: 0.6023, mae: 0.2107, huber: 0.0992, swd: 0.4103, ept: 95.2195
    Epoch [20/50], Test Losses: mse: 0.6185, mae: 0.2350, huber: 0.1119, swd: 0.4271, ept: 95.0330
      Epoch 20 composite train-obj: 0.415943
            Val objective improved 0.1227 → 0.0992, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 2.7663, mae: 0.5900, huber: 0.4111, swd: 0.9163, ept: 73.9206
    Epoch [21/50], Val Losses: mse: 1.0979, mae: 0.3133, huber: 0.1611, swd: 0.8656, ept: 94.8258
    Epoch [21/50], Test Losses: mse: 1.0065, mae: 0.3319, huber: 0.1691, swd: 0.7829, ept: 94.6852
      Epoch 21 composite train-obj: 0.411113
            No improvement (0.1611), counter 1/5
    Epoch [22/50], Train Losses: mse: 2.7776, mae: 0.5905, huber: 0.4113, swd: 0.9221, ept: 73.9393
    Epoch [22/50], Val Losses: mse: 0.6040, mae: 0.3221, huber: 0.1347, swd: 0.4094, ept: 95.2212
    Epoch [22/50], Test Losses: mse: 0.6119, mae: 0.3506, huber: 0.1483, swd: 0.4126, ept: 95.1649
      Epoch 22 composite train-obj: 0.411331
            No improvement (0.1347), counter 2/5
    Epoch [23/50], Train Losses: mse: 2.7376, mae: 0.5863, huber: 0.4082, swd: 0.8791, ept: 73.9024
    Epoch [23/50], Val Losses: mse: 0.5921, mae: 0.2708, huber: 0.1132, swd: 0.4262, ept: 94.8493
    Epoch [23/50], Test Losses: mse: 0.5361, mae: 0.2831, huber: 0.1138, swd: 0.3761, ept: 95.0496
      Epoch 23 composite train-obj: 0.408207
            No improvement (0.1132), counter 3/5
    Epoch [24/50], Train Losses: mse: 2.7166, mae: 0.5847, huber: 0.4063, swd: 0.8684, ept: 73.9251
    Epoch [24/50], Val Losses: mse: 0.7029, mae: 0.3440, huber: 0.1606, swd: 0.4732, ept: 95.1394
    Epoch [24/50], Test Losses: mse: 0.8011, mae: 0.3856, huber: 0.1879, swd: 0.5406, ept: 95.0208
      Epoch 24 composite train-obj: 0.406263
            No improvement (0.1606), counter 4/5
    Epoch [25/50], Train Losses: mse: 2.6957, mae: 0.5804, huber: 0.4040, swd: 0.8407, ept: 73.9191
    Epoch [25/50], Val Losses: mse: 0.8377, mae: 0.3691, huber: 0.1809, swd: 0.6453, ept: 94.8630
    Epoch [25/50], Test Losses: mse: 0.7666, mae: 0.3812, huber: 0.1833, swd: 0.5804, ept: 95.0134
      Epoch 25 composite train-obj: 0.403977
    Epoch [25/50], Test Losses: mse: 0.6185, mae: 0.2350, huber: 0.1119, swd: 0.4271, ept: 95.0330
    Best round's Test MSE: 0.6185, MAE: 0.2350, SWD: 0.4271
    Best round's Validation MSE: 0.6023, MAE: 0.2107, SWD: 0.4103
    Best round's Test verification MSE : 0.6185, MAE: 0.2350, SWD: 0.4271
    Time taken: 82.64 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.5519, mae: 1.1916, huber: 0.9125, swd: 4.5151, ept: 73.2959
    Epoch [1/50], Val Losses: mse: 3.2966, mae: 0.6502, huber: 0.4220, swd: 2.1911, ept: 92.9923
    Epoch [1/50], Test Losses: mse: 3.5874, mae: 0.7080, huber: 0.4648, swd: 2.3676, ept: 92.7169
      Epoch 1 composite train-obj: 0.912507
            Val objective improved inf → 0.4220, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.0851, mae: 0.8384, huber: 0.6021, swd: 1.9701, ept: 73.6589
    Epoch [2/50], Val Losses: mse: 2.0971, mae: 0.5772, huber: 0.3490, swd: 1.7201, ept: 93.5029
    Epoch [2/50], Test Losses: mse: 2.0863, mae: 0.6294, huber: 0.3857, swd: 1.6843, ept: 93.2548
      Epoch 2 composite train-obj: 0.602114
            Val objective improved 0.4220 → 0.3490, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 3.6561, mae: 0.7608, huber: 0.5389, swd: 1.6870, ept: 73.7313
    Epoch [3/50], Val Losses: mse: 1.5106, mae: 0.3794, huber: 0.2041, swd: 1.2364, ept: 94.4639
    Epoch [3/50], Test Losses: mse: 1.4449, mae: 0.4149, huber: 0.2254, swd: 1.1646, ept: 94.1924
      Epoch 3 composite train-obj: 0.538853
            Val objective improved 0.3490 → 0.2041, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 3.5092, mae: 0.7177, huber: 0.5075, swd: 1.5809, ept: 73.7395
    Epoch [4/50], Val Losses: mse: 1.3648, mae: 0.4361, huber: 0.2193, swd: 1.1143, ept: 94.6762
    Epoch [4/50], Test Losses: mse: 1.2899, mae: 0.4633, huber: 0.2333, swd: 1.0489, ept: 94.5890
      Epoch 4 composite train-obj: 0.507542
            No improvement (0.2193), counter 1/5
    Epoch [5/50], Train Losses: mse: 3.4389, mae: 0.6955, huber: 0.4917, swd: 1.5307, ept: 73.8055
    Epoch [5/50], Val Losses: mse: 1.2298, mae: 0.3440, huber: 0.1679, swd: 1.0313, ept: 94.7168
    Epoch [5/50], Test Losses: mse: 1.1402, mae: 0.3705, huber: 0.1796, swd: 0.9479, ept: 94.5870
      Epoch 5 composite train-obj: 0.491700
            Val objective improved 0.2041 → 0.1679, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.3451, mae: 0.6783, huber: 0.4793, swd: 1.4569, ept: 73.8144
    Epoch [6/50], Val Losses: mse: 1.3942, mae: 0.3993, huber: 0.2007, swd: 1.1675, ept: 94.5402
    Epoch [6/50], Test Losses: mse: 1.3302, mae: 0.4342, huber: 0.2199, swd: 1.1067, ept: 94.2634
      Epoch 6 composite train-obj: 0.479294
            No improvement (0.2007), counter 1/5
    Epoch [7/50], Train Losses: mse: 3.2953, mae: 0.6638, huber: 0.4689, swd: 1.4256, ept: 73.8481
    Epoch [7/50], Val Losses: mse: 1.3337, mae: 0.3494, huber: 0.1808, swd: 1.0981, ept: 94.4632
    Epoch [7/50], Test Losses: mse: 1.2175, mae: 0.3740, huber: 0.1903, swd: 0.9981, ept: 94.4233
      Epoch 7 composite train-obj: 0.468895
            No improvement (0.1808), counter 2/5
    Epoch [8/50], Train Losses: mse: 3.2514, mae: 0.6580, huber: 0.4639, swd: 1.3882, ept: 73.8397
    Epoch [8/50], Val Losses: mse: 1.3088, mae: 0.4011, huber: 0.1958, swd: 1.0889, ept: 94.6660
    Epoch [8/50], Test Losses: mse: 1.2146, mae: 0.4285, huber: 0.2084, swd: 0.9998, ept: 94.5708
      Epoch 8 composite train-obj: 0.463853
            No improvement (0.1958), counter 3/5
    Epoch [9/50], Train Losses: mse: 3.2253, mae: 0.6501, huber: 0.4586, swd: 1.3590, ept: 73.8408
    Epoch [9/50], Val Losses: mse: 1.2325, mae: 0.3626, huber: 0.1744, swd: 1.0488, ept: 94.7969
    Epoch [9/50], Test Losses: mse: 1.2037, mae: 0.3990, huber: 0.1966, swd: 1.0090, ept: 94.4675
      Epoch 9 composite train-obj: 0.458622
            No improvement (0.1744), counter 4/5
    Epoch [10/50], Train Losses: mse: 3.1845, mae: 0.6364, huber: 0.4488, swd: 1.3271, ept: 73.8867
    Epoch [10/50], Val Losses: mse: 1.2772, mae: 0.3943, huber: 0.2014, swd: 1.0213, ept: 94.6990
    Epoch [10/50], Test Losses: mse: 1.1990, mae: 0.4222, huber: 0.2147, swd: 0.9430, ept: 94.5705
      Epoch 10 composite train-obj: 0.448782
    Epoch [10/50], Test Losses: mse: 1.1402, mae: 0.3705, huber: 0.1796, swd: 0.9479, ept: 94.5870
    Best round's Test MSE: 1.1402, MAE: 0.3705, SWD: 0.9479
    Best round's Validation MSE: 1.2298, MAE: 0.3440, SWD: 1.0313
    Best round's Test verification MSE : 1.1402, MAE: 0.3705, SWD: 0.9479
    Time taken: 33.97 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.5988, mae: 1.1893, huber: 0.9098, swd: 4.2116, ept: 73.3027
    Epoch [1/50], Val Losses: mse: 3.5971, mae: 0.9558, huber: 0.6573, swd: 2.5013, ept: 92.5769
    Epoch [1/50], Test Losses: mse: 3.6121, mae: 1.0211, huber: 0.7100, swd: 2.4701, ept: 92.1208
      Epoch 1 composite train-obj: 0.909830
            Val objective improved inf → 0.6573, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.0711, mae: 0.8444, huber: 0.6051, swd: 1.8074, ept: 73.6671
    Epoch [2/50], Val Losses: mse: 1.8027, mae: 0.4627, huber: 0.2707, swd: 1.3130, ept: 93.6240
    Epoch [2/50], Test Losses: mse: 1.8330, mae: 0.5096, huber: 0.3028, swd: 1.2976, ept: 93.3577
      Epoch 2 composite train-obj: 0.605118
            Val objective improved 0.6573 → 0.2707, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 3.7094, mae: 0.7659, huber: 0.5447, swd: 1.5839, ept: 73.6964
    Epoch [3/50], Val Losses: mse: 1.7780, mae: 0.4671, huber: 0.2658, swd: 1.2850, ept: 94.1824
    Epoch [3/50], Test Losses: mse: 1.6875, mae: 0.4981, huber: 0.2828, swd: 1.2002, ept: 94.0685
      Epoch 3 composite train-obj: 0.544687
            Val objective improved 0.2707 → 0.2658, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 3.5078, mae: 0.7265, huber: 0.5134, swd: 1.4473, ept: 73.7923
    Epoch [4/50], Val Losses: mse: 2.0570, mae: 0.6251, huber: 0.3611, swd: 1.5625, ept: 94.0916
    Epoch [4/50], Test Losses: mse: 2.0361, mae: 0.6766, huber: 0.3986, swd: 1.5312, ept: 93.6206
      Epoch 4 composite train-obj: 0.513412
            No improvement (0.3611), counter 1/5
    Epoch [5/50], Train Losses: mse: 3.4035, mae: 0.6995, huber: 0.4941, swd: 1.3824, ept: 73.7770
    Epoch [5/50], Val Losses: mse: 1.4580, mae: 0.4186, huber: 0.2179, swd: 1.1133, ept: 94.3696
    Epoch [5/50], Test Losses: mse: 1.4266, mae: 0.4549, huber: 0.2418, swd: 1.0791, ept: 93.9149
      Epoch 5 composite train-obj: 0.494142
            Val objective improved 0.2658 → 0.2179, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.2838, mae: 0.6778, huber: 0.4771, swd: 1.2903, ept: 73.7865
    Epoch [6/50], Val Losses: mse: 1.2163, mae: 0.4150, huber: 0.2107, swd: 0.8946, ept: 94.6716
    Epoch [6/50], Test Losses: mse: 1.2265, mae: 0.4571, huber: 0.2378, swd: 0.8766, ept: 94.3935
      Epoch 6 composite train-obj: 0.477101
            Val objective improved 0.2179 → 0.2107, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.3053, mae: 0.6744, huber: 0.4754, swd: 1.3128, ept: 73.8033
    Epoch [7/50], Val Losses: mse: 1.1035, mae: 0.3554, huber: 0.1761, swd: 0.7852, ept: 94.8022
    Epoch [7/50], Test Losses: mse: 1.0445, mae: 0.3809, huber: 0.1881, swd: 0.7342, ept: 94.8020
      Epoch 7 composite train-obj: 0.475422
            Val objective improved 0.2107 → 0.1761, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.1822, mae: 0.6624, huber: 0.4644, swd: 1.2082, ept: 73.8015
    Epoch [8/50], Val Losses: mse: 1.0516, mae: 0.3155, huber: 0.1591, swd: 0.7852, ept: 94.3824
    Epoch [8/50], Test Losses: mse: 0.9754, mae: 0.3389, huber: 0.1683, swd: 0.7181, ept: 94.4415
      Epoch 8 composite train-obj: 0.464370
            Val objective improved 0.1761 → 0.1591, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.1734, mae: 0.6525, huber: 0.4583, swd: 1.2070, ept: 73.8642
    Epoch [9/50], Val Losses: mse: 1.0710, mae: 0.3523, huber: 0.1760, swd: 0.7760, ept: 94.6444
    Epoch [9/50], Test Losses: mse: 0.9876, mae: 0.3749, huber: 0.1832, swd: 0.7026, ept: 94.7412
      Epoch 9 composite train-obj: 0.458258
            No improvement (0.1760), counter 1/5
    Epoch [10/50], Train Losses: mse: 3.1177, mae: 0.6458, huber: 0.4526, swd: 1.1632, ept: 73.8209
    Epoch [10/50], Val Losses: mse: 0.8634, mae: 0.2906, huber: 0.1426, swd: 0.6395, ept: 95.0713
    Epoch [10/50], Test Losses: mse: 0.8065, mae: 0.3093, huber: 0.1500, swd: 0.5866, ept: 95.0602
      Epoch 10 composite train-obj: 0.452561
            Val objective improved 0.1591 → 0.1426, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 3.0440, mae: 0.6338, huber: 0.4440, swd: 1.1024, ept: 73.8669
    Epoch [11/50], Val Losses: mse: 1.5332, mae: 0.4339, huber: 0.2320, swd: 1.1693, ept: 94.1897
    Epoch [11/50], Test Losses: mse: 1.4205, mae: 0.4592, huber: 0.2433, swd: 1.0798, ept: 93.9595
      Epoch 11 composite train-obj: 0.443960
            No improvement (0.2320), counter 1/5
    Epoch [12/50], Train Losses: mse: 2.9587, mae: 0.6211, huber: 0.4342, swd: 1.0259, ept: 73.8683
    Epoch [12/50], Val Losses: mse: 1.4859, mae: 0.4523, huber: 0.2378, swd: 1.1681, ept: 94.5264
    Epoch [12/50], Test Losses: mse: 1.3855, mae: 0.4781, huber: 0.2508, swd: 1.0893, ept: 94.1929
      Epoch 12 composite train-obj: 0.434245
            No improvement (0.2378), counter 2/5
    Epoch [13/50], Train Losses: mse: 2.9991, mae: 0.6322, huber: 0.4410, swd: 1.0584, ept: 73.8700
    Epoch [13/50], Val Losses: mse: 0.9832, mae: 0.3946, huber: 0.1849, swd: 0.7093, ept: 94.5299
    Epoch [13/50], Test Losses: mse: 0.9204, mae: 0.4193, huber: 0.1943, swd: 0.6572, ept: 94.6391
      Epoch 13 composite train-obj: 0.441016
            No improvement (0.1849), counter 3/5
    Epoch [14/50], Train Losses: mse: 2.9016, mae: 0.6139, huber: 0.4284, swd: 0.9801, ept: 73.8775
    Epoch [14/50], Val Losses: mse: 1.2318, mae: 0.3822, huber: 0.1962, swd: 0.9299, ept: 94.3985
    Epoch [14/50], Test Losses: mse: 1.1792, mae: 0.4105, huber: 0.2105, swd: 0.8829, ept: 94.1473
      Epoch 14 composite train-obj: 0.428442
            No improvement (0.1962), counter 4/5
    Epoch [15/50], Train Losses: mse: 2.8405, mae: 0.6039, huber: 0.4212, swd: 0.9311, ept: 73.8826
    Epoch [15/50], Val Losses: mse: 1.0209, mae: 0.3781, huber: 0.1797, swd: 0.7447, ept: 94.7256
    Epoch [15/50], Test Losses: mse: 0.9204, mae: 0.3943, huber: 0.1833, swd: 0.6608, ept: 94.7934
      Epoch 15 composite train-obj: 0.421212
    Epoch [15/50], Test Losses: mse: 0.8065, mae: 0.3093, huber: 0.1500, swd: 0.5866, ept: 95.0602
    Best round's Test MSE: 0.8065, MAE: 0.3093, SWD: 0.5866
    Best round's Validation MSE: 0.8634, MAE: 0.2906, SWD: 0.6395
    Best round's Test verification MSE : 0.8065, MAE: 0.3093, SWD: 0.5866
    Time taken: 47.27 seconds
    
    ==================================================
    Experiment Summary (PatchTST_rossler_seq96_pred96_20250513_1325)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.8551 ± 0.2157
      mae: 0.3049 ± 0.0554
      huber: 0.1472 ± 0.0277
      swd: 0.6538 ± 0.2179
      ept: 94.8934 ± 0.2169
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.8985 ± 0.2574
      mae: 0.2817 ± 0.0548
      huber: 0.1366 ± 0.0283
      swd: 0.6937 ± 0.2564
      ept: 95.0025 ± 0.2109
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 163.93 seconds
    
    Experiment complete: PatchTST_rossler_seq96_pred96_20250513_1325
    Model: PatchTST
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['rossler']['channels'],
    enc_in=data_mgr.datasets['rossler']['channels'],
    dec_in=data_mgr.datasets['rossler']['channels'],
    c_out=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 18.7758, mae: 2.2311, huber: 1.8947, swd: 9.4657, ept: 117.8713
    Epoch [1/50], Val Losses: mse: 24.0754, mae: 1.5592, huber: 1.2690, swd: 18.9152, ept: 175.6621
    Epoch [1/50], Test Losses: mse: 23.9660, mae: 1.7214, huber: 1.4134, swd: 18.1777, ept: 173.2847
      Epoch 1 composite train-obj: 1.894691
            Val objective improved inf → 1.2690, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.5110, mae: 1.5755, huber: 1.2695, swd: 5.3591, ept: 119.6200
    Epoch [2/50], Val Losses: mse: 5.1485, mae: 1.0429, huber: 0.7517, swd: 4.0423, ept: 182.4694
    Epoch [2/50], Test Losses: mse: 5.5531, mae: 1.1320, huber: 0.8279, swd: 3.9684, ept: 180.8102
      Epoch 2 composite train-obj: 1.269528
            Val objective improved 1.2690 → 0.7517, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.6562, mae: 1.3741, huber: 1.0855, swd: 4.2393, ept: 120.2380
    Epoch [3/50], Val Losses: mse: 4.8409, mae: 0.9899, huber: 0.6972, swd: 3.7401, ept: 183.2503
    Epoch [3/50], Test Losses: mse: 5.5934, mae: 1.1082, huber: 0.8006, swd: 3.7995, ept: 181.6272
      Epoch 3 composite train-obj: 1.085480
            Val objective improved 0.7517 → 0.6972, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.8327, mae: 1.2727, huber: 0.9938, swd: 3.7112, ept: 120.3967
    Epoch [4/50], Val Losses: mse: 10.6933, mae: 1.5438, huber: 1.2365, swd: 4.4397, ept: 173.5380
    Epoch [4/50], Test Losses: mse: 11.6516, mae: 1.6744, huber: 1.3536, swd: 4.3395, ept: 171.0490
      Epoch 4 composite train-obj: 0.993777
            No improvement (1.2365), counter 1/5
    Epoch [5/50], Train Losses: mse: 7.7366, mae: 1.2488, huber: 0.9725, swd: 3.6060, ept: 120.6561
    Epoch [5/50], Val Losses: mse: 3.9743, mae: 0.7619, huber: 0.5152, swd: 3.2180, ept: 184.6362
    Epoch [5/50], Test Losses: mse: 3.9745, mae: 0.8306, huber: 0.5680, swd: 3.0929, ept: 183.3672
      Epoch 5 composite train-obj: 0.972548
            Val objective improved 0.6972 → 0.5152, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.1537, mae: 1.1813, huber: 0.9113, swd: 3.3158, ept: 120.9785
    Epoch [6/50], Val Losses: mse: 3.7325, mae: 0.8091, huber: 0.5410, swd: 3.0208, ept: 186.3901
    Epoch [6/50], Test Losses: mse: 4.0954, mae: 0.8821, huber: 0.6014, swd: 3.1402, ept: 184.8513
      Epoch 6 composite train-obj: 0.911263
            No improvement (0.5410), counter 1/5
    Epoch [7/50], Train Losses: mse: 6.9813, mae: 1.1537, huber: 0.8879, swd: 3.2021, ept: 121.0836
    Epoch [7/50], Val Losses: mse: 3.8330, mae: 0.7389, huber: 0.5011, swd: 3.0092, ept: 185.8899
    Epoch [7/50], Test Losses: mse: 4.5194, mae: 0.8290, huber: 0.5762, swd: 3.4373, ept: 184.5943
      Epoch 7 composite train-obj: 0.887908
            Val objective improved 0.5152 → 0.5011, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 6.8304, mae: 1.1229, huber: 0.8621, swd: 3.1155, ept: 121.1908
    Epoch [8/50], Val Losses: mse: 3.6153, mae: 0.8551, huber: 0.5811, swd: 2.8315, ept: 187.3707
    Epoch [8/50], Test Losses: mse: 3.5323, mae: 0.9028, huber: 0.6183, swd: 2.6287, ept: 186.8359
      Epoch 8 composite train-obj: 0.862149
            No improvement (0.5811), counter 1/5
    Epoch [9/50], Train Losses: mse: 6.7748, mae: 1.1190, huber: 0.8583, swd: 3.0739, ept: 121.1879
    Epoch [9/50], Val Losses: mse: 3.3507, mae: 0.7998, huber: 0.5212, swd: 2.8011, ept: 187.6866
    Epoch [9/50], Test Losses: mse: 3.4390, mae: 0.8611, huber: 0.5706, swd: 2.6468, ept: 186.4134
      Epoch 9 composite train-obj: 0.858314
            No improvement (0.5212), counter 2/5
    Epoch [10/50], Train Losses: mse: 6.6752, mae: 1.1073, huber: 0.8467, swd: 3.0017, ept: 121.4212
    Epoch [10/50], Val Losses: mse: 3.5186, mae: 0.7279, huber: 0.4922, swd: 2.7816, ept: 187.0989
    Epoch [10/50], Test Losses: mse: 3.3680, mae: 0.7706, huber: 0.5217, swd: 2.6033, ept: 186.4388
      Epoch 10 composite train-obj: 0.846667
            Val objective improved 0.5011 → 0.4922, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 6.6132, mae: 1.0883, huber: 0.8326, swd: 2.9561, ept: 121.3397
    Epoch [11/50], Val Losses: mse: 3.5513, mae: 0.8013, huber: 0.5249, swd: 2.9722, ept: 183.4842
    Epoch [11/50], Test Losses: mse: 3.4982, mae: 0.8624, huber: 0.5702, swd: 2.8104, ept: 183.5687
      Epoch 11 composite train-obj: 0.832581
            No improvement (0.5249), counter 1/5
    Epoch [12/50], Train Losses: mse: 6.5404, mae: 1.0774, huber: 0.8231, swd: 2.9186, ept: 121.4394
    Epoch [12/50], Val Losses: mse: 3.3392, mae: 0.8110, huber: 0.5293, swd: 2.7967, ept: 187.3997
    Epoch [12/50], Test Losses: mse: 3.5876, mae: 0.8890, huber: 0.5925, swd: 2.6849, ept: 185.6192
      Epoch 12 composite train-obj: 0.823146
            No improvement (0.5293), counter 2/5
    Epoch [13/50], Train Losses: mse: 6.5340, mae: 1.0708, huber: 0.8187, swd: 2.9045, ept: 121.4347
    Epoch [13/50], Val Losses: mse: 3.2028, mae: 0.7496, huber: 0.4848, swd: 2.5937, ept: 187.9154
    Epoch [13/50], Test Losses: mse: 3.0880, mae: 0.7929, huber: 0.5161, swd: 2.4196, ept: 187.4827
      Epoch 13 composite train-obj: 0.818662
            Val objective improved 0.4922 → 0.4848, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 6.4575, mae: 1.0555, huber: 0.8065, swd: 2.8538, ept: 121.5048
    Epoch [14/50], Val Losses: mse: 3.4099, mae: 0.6372, huber: 0.4224, swd: 2.8123, ept: 186.7047
    Epoch [14/50], Test Losses: mse: 3.2936, mae: 0.6884, huber: 0.4569, swd: 2.6268, ept: 186.1509
      Epoch 14 composite train-obj: 0.806480
            Val objective improved 0.4848 → 0.4224, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 6.4270, mae: 1.0465, huber: 0.7991, swd: 2.8403, ept: 121.5596
    Epoch [15/50], Val Losses: mse: 3.4393, mae: 0.6927, huber: 0.4631, swd: 2.7590, ept: 187.0231
    Epoch [15/50], Test Losses: mse: 3.3078, mae: 0.7527, huber: 0.5064, swd: 2.5665, ept: 185.9084
      Epoch 15 composite train-obj: 0.799128
            No improvement (0.4631), counter 1/5
    Epoch [16/50], Train Losses: mse: 6.3917, mae: 1.0418, huber: 0.7951, swd: 2.8209, ept: 121.5884
    Epoch [16/50], Val Losses: mse: 3.1665, mae: 0.7417, huber: 0.4832, swd: 2.6830, ept: 187.5146
    Epoch [16/50], Test Losses: mse: 3.0300, mae: 0.7760, huber: 0.5054, swd: 2.4744, ept: 186.6663
      Epoch 16 composite train-obj: 0.795111
            No improvement (0.4832), counter 2/5
    Epoch [17/50], Train Losses: mse: 6.3551, mae: 1.0346, huber: 0.7895, swd: 2.7792, ept: 121.5181
    Epoch [17/50], Val Losses: mse: 3.5715, mae: 0.8194, huber: 0.5481, swd: 2.9663, ept: 187.2532
    Epoch [17/50], Test Losses: mse: 3.6096, mae: 0.8848, huber: 0.6009, swd: 2.8048, ept: 185.5004
      Epoch 17 composite train-obj: 0.789509
            No improvement (0.5481), counter 3/5
    Epoch [18/50], Train Losses: mse: 6.3917, mae: 1.0377, huber: 0.7921, swd: 2.8181, ept: 121.5566
    Epoch [18/50], Val Losses: mse: 3.1904, mae: 0.6961, huber: 0.4505, swd: 2.7187, ept: 187.8055
    Epoch [18/50], Test Losses: mse: 3.1408, mae: 0.7411, huber: 0.4831, swd: 2.5362, ept: 186.3380
      Epoch 18 composite train-obj: 0.792132
            No improvement (0.4505), counter 4/5
    Epoch [19/50], Train Losses: mse: 6.3223, mae: 1.0303, huber: 0.7853, swd: 2.7674, ept: 121.6539
    Epoch [19/50], Val Losses: mse: 3.5243, mae: 0.8107, huber: 0.5315, swd: 3.0355, ept: 186.9892
    Epoch [19/50], Test Losses: mse: 3.4077, mae: 0.8585, huber: 0.5667, swd: 2.8464, ept: 185.9637
      Epoch 19 composite train-obj: 0.785343
    Epoch [19/50], Test Losses: mse: 3.2936, mae: 0.6884, huber: 0.4569, swd: 2.6268, ept: 186.1509
    Best round's Test MSE: 3.2936, MAE: 0.6884, SWD: 2.6268
    Best round's Validation MSE: 3.4099, MAE: 0.6372, SWD: 2.8123
    Best round's Test verification MSE : 3.2936, MAE: 0.6884, SWD: 2.6268
    Time taken: 64.77 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 19.1059, mae: 2.2516, huber: 1.9151, swd: 10.0227, ept: 117.8750
    Epoch [1/50], Val Losses: mse: 7.8108, mae: 1.4672, huber: 1.1510, swd: 5.3359, ept: 176.2925
    Epoch [1/50], Test Losses: mse: 7.9598, mae: 1.5539, huber: 1.2251, swd: 5.1123, ept: 174.1354
      Epoch 1 composite train-obj: 1.915062
            Val objective improved inf → 1.1510, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.7046, mae: 1.6219, huber: 1.3127, swd: 5.6032, ept: 119.5006
    Epoch [2/50], Val Losses: mse: 6.1743, mae: 1.0884, huber: 0.8193, swd: 4.1908, ept: 180.1508
    Epoch [2/50], Test Losses: mse: 6.2538, mae: 1.1694, huber: 0.8854, swd: 3.9240, ept: 178.4154
      Epoch 2 composite train-obj: 1.312750
            Val objective improved 1.1510 → 0.8193, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.6477, mae: 1.4082, huber: 1.1153, swd: 4.2903, ept: 120.2706
    Epoch [3/50], Val Losses: mse: 4.4208, mae: 0.9291, huber: 0.6557, swd: 3.4312, ept: 183.9036
    Epoch [3/50], Test Losses: mse: 5.0091, mae: 1.0347, huber: 0.7453, swd: 3.5619, ept: 182.3454
      Epoch 3 composite train-obj: 1.115288
            Val objective improved 0.8193 → 0.6557, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.6596, mae: 1.2716, huber: 0.9922, swd: 3.6717, ept: 120.7594
    Epoch [4/50], Val Losses: mse: 3.9574, mae: 0.8314, huber: 0.5631, swd: 3.2592, ept: 185.0407
    Epoch [4/50], Test Losses: mse: 4.0890, mae: 0.9115, huber: 0.6273, swd: 3.1061, ept: 183.7640
      Epoch 4 composite train-obj: 0.992151
            Val objective improved 0.6557 → 0.5631, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.3648, mae: 1.2149, huber: 0.9417, swd: 3.5208, ept: 120.9386
    Epoch [5/50], Val Losses: mse: 3.6544, mae: 0.8027, huber: 0.5394, swd: 3.1047, ept: 185.7061
    Epoch [5/50], Test Losses: mse: 3.7668, mae: 0.8608, huber: 0.5849, swd: 2.9274, ept: 184.6978
      Epoch 5 composite train-obj: 0.941744
            Val objective improved 0.5631 → 0.5394, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.0061, mae: 1.1583, huber: 0.8939, swd: 3.2616, ept: 121.1681
    Epoch [6/50], Val Losses: mse: 3.7531, mae: 0.7420, huber: 0.4951, swd: 2.9718, ept: 185.5182
    Epoch [6/50], Test Losses: mse: 3.6923, mae: 0.8026, huber: 0.5419, swd: 2.8276, ept: 184.2836
      Epoch 6 composite train-obj: 0.893946
            Val objective improved 0.5394 → 0.4951, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 6.9233, mae: 1.1430, huber: 0.8801, swd: 3.2019, ept: 121.2662
    Epoch [7/50], Val Losses: mse: 3.5129, mae: 0.6927, huber: 0.4545, swd: 2.8881, ept: 186.5658
    Epoch [7/50], Test Losses: mse: 3.6834, mae: 0.7771, huber: 0.5225, swd: 2.7415, ept: 185.6501
      Epoch 7 composite train-obj: 0.880104
            Val objective improved 0.4951 → 0.4545, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 6.7723, mae: 1.1147, huber: 0.8564, swd: 3.1096, ept: 121.3197
    Epoch [8/50], Val Losses: mse: 3.6227, mae: 0.7962, huber: 0.5257, swd: 3.0639, ept: 186.5271
    Epoch [8/50], Test Losses: mse: 3.7530, mae: 0.8683, huber: 0.5848, swd: 2.9404, ept: 185.4656
      Epoch 8 composite train-obj: 0.856376
            No improvement (0.5257), counter 1/5
    Epoch [9/50], Train Losses: mse: 6.7384, mae: 1.1037, huber: 0.8477, swd: 3.0910, ept: 121.2927
    Epoch [9/50], Val Losses: mse: 3.6165, mae: 0.8867, huber: 0.5994, swd: 3.1204, ept: 187.6458
    Epoch [9/50], Test Losses: mse: 3.5473, mae: 0.9347, huber: 0.6342, swd: 2.9279, ept: 186.5982
      Epoch 9 composite train-obj: 0.847713
            No improvement (0.5994), counter 2/5
    Epoch [10/50], Train Losses: mse: 6.7112, mae: 1.1030, huber: 0.8460, swd: 3.0740, ept: 121.4014
    Epoch [10/50], Val Losses: mse: 3.5903, mae: 0.7809, huber: 0.5166, swd: 2.9228, ept: 187.0952
    Epoch [10/50], Test Losses: mse: 3.5874, mae: 0.8507, huber: 0.5697, swd: 2.7867, ept: 186.1948
      Epoch 10 composite train-obj: 0.845980
            No improvement (0.5166), counter 3/5
    Epoch [11/50], Train Losses: mse: 6.6421, mae: 1.0848, huber: 0.8314, swd: 3.0282, ept: 121.4549
    Epoch [11/50], Val Losses: mse: 3.2479, mae: 0.7249, huber: 0.4806, swd: 2.8097, ept: 188.1415
    Epoch [11/50], Test Losses: mse: 3.1933, mae: 0.7641, huber: 0.5079, swd: 2.6444, ept: 187.1423
      Epoch 11 composite train-obj: 0.831404
            No improvement (0.4806), counter 4/5
    Epoch [12/50], Train Losses: mse: 6.5923, mae: 1.0716, huber: 0.8214, swd: 3.0036, ept: 121.4048
    Epoch [12/50], Val Losses: mse: 3.2358, mae: 0.7065, huber: 0.4675, swd: 2.8086, ept: 187.5528
    Epoch [12/50], Test Losses: mse: 3.1716, mae: 0.7496, huber: 0.4984, swd: 2.6947, ept: 185.9458
      Epoch 12 composite train-obj: 0.821365
    Epoch [12/50], Test Losses: mse: 3.6834, mae: 0.7771, huber: 0.5225, swd: 2.7415, ept: 185.6501
    Best round's Test MSE: 3.6834, MAE: 0.7771, SWD: 2.7415
    Best round's Validation MSE: 3.5129, MAE: 0.6927, SWD: 2.8881
    Best round's Test verification MSE : 3.6834, MAE: 0.7771, SWD: 2.7415
    Time taken: 43.70 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.0244, mae: 2.1298, huber: 1.7961, swd: 7.8944, ept: 118.0022
    Epoch [1/50], Val Losses: mse: 7.4359, mae: 1.3285, huber: 1.0130, swd: 5.2660, ept: 178.6907
    Epoch [1/50], Test Losses: mse: 7.6998, mae: 1.4169, huber: 1.0898, swd: 5.0331, ept: 176.5768
      Epoch 1 composite train-obj: 1.796103
            Val objective improved inf → 1.0130, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.7626, mae: 1.5212, huber: 1.2175, swd: 4.6794, ept: 119.8366
    Epoch [2/50], Val Losses: mse: 4.9075, mae: 1.0851, huber: 0.7860, swd: 3.6015, ept: 182.4422
    Epoch [2/50], Test Losses: mse: 5.0856, mae: 1.1544, huber: 0.8427, swd: 3.3971, ept: 180.9160
      Epoch 2 composite train-obj: 1.217528
            Val objective improved 1.0130 → 0.7860, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.0774, mae: 1.3342, huber: 1.0464, swd: 3.6129, ept: 120.6051
    Epoch [3/50], Val Losses: mse: 4.4918, mae: 0.8412, huber: 0.5895, swd: 3.1677, ept: 182.8253
    Epoch [3/50], Test Losses: mse: 4.8320, mae: 0.9456, huber: 0.6754, swd: 3.0070, ept: 181.6686
      Epoch 3 composite train-obj: 1.046420
            Val objective improved 0.7860 → 0.5895, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.4373, mae: 1.2387, huber: 0.9616, swd: 3.2439, ept: 120.8702
    Epoch [4/50], Val Losses: mse: 3.9743, mae: 0.7577, huber: 0.5148, swd: 2.9675, ept: 184.2504
    Epoch [4/50], Test Losses: mse: 4.2416, mae: 0.8549, huber: 0.5952, swd: 2.9793, ept: 182.8027
      Epoch 4 composite train-obj: 0.961555
            Val objective improved 0.5895 → 0.5148, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.1392, mae: 1.1936, huber: 0.9216, swd: 3.0670, ept: 121.0066
    Epoch [5/50], Val Losses: mse: 3.6986, mae: 0.7224, huber: 0.4916, swd: 2.7831, ept: 185.5182
    Epoch [5/50], Test Losses: mse: 3.6374, mae: 0.7812, huber: 0.5361, swd: 2.6551, ept: 184.4754
      Epoch 5 composite train-obj: 0.921632
            Val objective improved 0.5148 → 0.4916, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.9712, mae: 1.1596, huber: 0.8929, swd: 2.9680, ept: 121.1867
    Epoch [6/50], Val Losses: mse: 3.6646, mae: 0.7875, huber: 0.5290, swd: 2.7491, ept: 184.9011
    Epoch [6/50], Test Losses: mse: 3.6844, mae: 0.8585, huber: 0.5840, swd: 2.6135, ept: 183.7934
      Epoch 6 composite train-obj: 0.892928
            No improvement (0.5290), counter 1/5
    Epoch [7/50], Train Losses: mse: 6.9021, mae: 1.1464, huber: 0.8819, swd: 2.9270, ept: 121.1985
    Epoch [7/50], Val Losses: mse: 4.2478, mae: 0.9822, huber: 0.6811, swd: 3.1845, ept: 183.3216
    Epoch [7/50], Test Losses: mse: 4.4536, mae: 1.0752, huber: 0.7588, swd: 3.0407, ept: 182.2947
      Epoch 7 composite train-obj: 0.881872
            No improvement (0.6811), counter 2/5
    Epoch [8/50], Train Losses: mse: 6.7807, mae: 1.1212, huber: 0.8609, swd: 2.8597, ept: 121.3291
    Epoch [8/50], Val Losses: mse: 3.4225, mae: 0.7798, huber: 0.5099, swd: 2.6338, ept: 186.5367
    Epoch [8/50], Test Losses: mse: 3.3741, mae: 0.8327, huber: 0.5494, swd: 2.4837, ept: 185.3945
      Epoch 8 composite train-obj: 0.860882
            No improvement (0.5099), counter 3/5
    Epoch [9/50], Train Losses: mse: 6.7751, mae: 1.1222, huber: 0.8612, swd: 2.8600, ept: 121.3540
    Epoch [9/50], Val Losses: mse: 3.4439, mae: 0.7650, huber: 0.5019, swd: 2.5774, ept: 187.2284
    Epoch [9/50], Test Losses: mse: 3.7260, mae: 0.8531, huber: 0.5751, swd: 2.4618, ept: 185.6624
      Epoch 9 composite train-obj: 0.861238
            No improvement (0.5019), counter 4/5
    Epoch [10/50], Train Losses: mse: 6.6609, mae: 1.0927, huber: 0.8376, swd: 2.7840, ept: 121.3797
    Epoch [10/50], Val Losses: mse: 3.4867, mae: 0.7332, huber: 0.4858, swd: 2.7079, ept: 186.7274
    Epoch [10/50], Test Losses: mse: 3.4324, mae: 0.7823, huber: 0.5197, swd: 2.5473, ept: 185.5552
      Epoch 10 composite train-obj: 0.837621
            Val objective improved 0.4916 → 0.4858, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 6.5667, mae: 1.0804, huber: 0.8267, swd: 2.7352, ept: 121.3937
    Epoch [11/50], Val Losses: mse: 3.8384, mae: 0.8040, huber: 0.5446, swd: 2.8824, ept: 185.6048
    Epoch [11/50], Test Losses: mse: 3.7799, mae: 0.8672, huber: 0.5938, swd: 2.7207, ept: 184.4751
      Epoch 11 composite train-obj: 0.826728
            No improvement (0.5446), counter 1/5
    Epoch [12/50], Train Losses: mse: 6.5676, mae: 1.0794, huber: 0.8260, swd: 2.7298, ept: 121.3698
    Epoch [12/50], Val Losses: mse: 3.4912, mae: 0.7431, huber: 0.4904, swd: 2.7555, ept: 187.2019
    Epoch [12/50], Test Losses: mse: 3.4507, mae: 0.7884, huber: 0.5250, swd: 2.6029, ept: 185.8472
      Epoch 12 composite train-obj: 0.825957
            No improvement (0.4904), counter 2/5
    Epoch [13/50], Train Losses: mse: 6.5376, mae: 1.0687, huber: 0.8174, swd: 2.7077, ept: 121.4132
    Epoch [13/50], Val Losses: mse: 3.4233, mae: 0.7606, huber: 0.5013, swd: 2.6434, ept: 186.5574
    Epoch [13/50], Test Losses: mse: 3.3551, mae: 0.8151, huber: 0.5398, swd: 2.4890, ept: 185.2801
      Epoch 13 composite train-obj: 0.817375
            No improvement (0.5013), counter 3/5
    Epoch [14/50], Train Losses: mse: 6.5223, mae: 1.0670, huber: 0.8160, swd: 2.6998, ept: 121.4786
    Epoch [14/50], Val Losses: mse: 3.2777, mae: 0.7734, huber: 0.5009, swd: 2.4667, ept: 187.1324
    Epoch [14/50], Test Losses: mse: 3.1923, mae: 0.8177, huber: 0.5325, swd: 2.3239, ept: 186.2411
      Epoch 14 composite train-obj: 0.816012
            No improvement (0.5009), counter 4/5
    Epoch [15/50], Train Losses: mse: 6.5225, mae: 1.0639, huber: 0.8138, swd: 2.7115, ept: 121.4410
    Epoch [15/50], Val Losses: mse: 3.1829, mae: 0.6980, huber: 0.4640, swd: 2.5527, ept: 187.9398
    Epoch [15/50], Test Losses: mse: 3.1985, mae: 0.7431, huber: 0.4975, swd: 2.3864, ept: 186.2301
      Epoch 15 composite train-obj: 0.813819
            Val objective improved 0.4858 → 0.4640, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 6.3997, mae: 1.0472, huber: 0.7993, swd: 2.6209, ept: 121.6045
    Epoch [16/50], Val Losses: mse: 3.9136, mae: 0.8866, huber: 0.6113, swd: 3.0910, ept: 186.5710
    Epoch [16/50], Test Losses: mse: 3.9142, mae: 0.9505, huber: 0.6636, swd: 2.9725, ept: 185.0325
      Epoch 16 composite train-obj: 0.799329
            No improvement (0.6113), counter 1/5
    Epoch [17/50], Train Losses: mse: 6.3687, mae: 1.0416, huber: 0.7944, swd: 2.6009, ept: 121.5990
    Epoch [17/50], Val Losses: mse: 3.3466, mae: 0.6968, huber: 0.4505, swd: 2.5243, ept: 185.9607
    Epoch [17/50], Test Losses: mse: 3.1444, mae: 0.7310, huber: 0.4729, swd: 2.3090, ept: 185.6795
      Epoch 17 composite train-obj: 0.794427
            Val objective improved 0.4640 → 0.4505, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 6.3827, mae: 1.0463, huber: 0.7974, swd: 2.6142, ept: 121.5337
    Epoch [18/50], Val Losses: mse: 3.3069, mae: 0.7719, huber: 0.5010, swd: 2.5127, ept: 187.1623
    Epoch [18/50], Test Losses: mse: 3.1249, mae: 0.8074, huber: 0.5241, swd: 2.3158, ept: 186.5855
      Epoch 18 composite train-obj: 0.797357
            No improvement (0.5010), counter 1/5
    Epoch [19/50], Train Losses: mse: 6.3486, mae: 1.0368, huber: 0.7909, swd: 2.5888, ept: 121.4535
    Epoch [19/50], Val Losses: mse: 3.2139, mae: 0.6923, huber: 0.4477, swd: 2.4488, ept: 187.2675
    Epoch [19/50], Test Losses: mse: 3.0976, mae: 0.7343, huber: 0.4768, swd: 2.2728, ept: 185.8636
      Epoch 19 composite train-obj: 0.790947
            Val objective improved 0.4505 → 0.4477, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 6.3258, mae: 1.0323, huber: 0.7865, swd: 2.5681, ept: 121.5809
    Epoch [20/50], Val Losses: mse: 3.1820, mae: 0.7171, huber: 0.4534, swd: 2.4903, ept: 187.4730
    Epoch [20/50], Test Losses: mse: 3.0672, mae: 0.7579, huber: 0.4808, swd: 2.2991, ept: 186.2886
      Epoch 20 composite train-obj: 0.786533
            No improvement (0.4534), counter 1/5
    Epoch [21/50], Train Losses: mse: 6.2799, mae: 1.0261, huber: 0.7808, swd: 2.5465, ept: 121.5371
    Epoch [21/50], Val Losses: mse: 3.8047, mae: 0.8427, huber: 0.5698, swd: 2.9372, ept: 187.1229
    Epoch [21/50], Test Losses: mse: 3.6182, mae: 0.8779, huber: 0.5928, swd: 2.7596, ept: 185.9358
      Epoch 21 composite train-obj: 0.780788
            No improvement (0.5698), counter 2/5
    Epoch [22/50], Train Losses: mse: 6.3143, mae: 1.0339, huber: 0.7873, swd: 2.5582, ept: 121.5373
    Epoch [22/50], Val Losses: mse: 3.3013, mae: 0.6254, huber: 0.4163, swd: 2.5406, ept: 187.0564
    Epoch [22/50], Test Losses: mse: 3.0709, mae: 0.6590, huber: 0.4343, swd: 2.3491, ept: 186.3003
      Epoch 22 composite train-obj: 0.787344
            Val objective improved 0.4477 → 0.4163, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 6.2009, mae: 1.0163, huber: 0.7732, swd: 2.4787, ept: 121.5872
    Epoch [23/50], Val Losses: mse: 3.4642, mae: 0.6651, huber: 0.4451, swd: 2.5532, ept: 186.8635
    Epoch [23/50], Test Losses: mse: 3.3577, mae: 0.7208, huber: 0.4849, swd: 2.3985, ept: 185.5225
      Epoch 23 composite train-obj: 0.773214
            No improvement (0.4451), counter 1/5
    Epoch [24/50], Train Losses: mse: 6.2287, mae: 1.0125, huber: 0.7713, swd: 2.4983, ept: 121.6250
    Epoch [24/50], Val Losses: mse: 3.0318, mae: 0.6318, huber: 0.4224, swd: 2.3597, ept: 188.1072
    Epoch [24/50], Test Losses: mse: 2.9380, mae: 0.6712, huber: 0.4503, swd: 2.1839, ept: 187.0602
      Epoch 24 composite train-obj: 0.771330
            No improvement (0.4224), counter 2/5
    Epoch [25/50], Train Losses: mse: 6.1648, mae: 1.0079, huber: 0.7666, swd: 2.4554, ept: 121.6245
    Epoch [25/50], Val Losses: mse: 2.7664, mae: 0.6674, huber: 0.4275, swd: 2.1341, ept: 188.7734
    Epoch [25/50], Test Losses: mse: 2.7598, mae: 0.7144, huber: 0.4607, swd: 1.9947, ept: 187.8356
      Epoch 25 composite train-obj: 0.766605
            No improvement (0.4275), counter 3/5
    Epoch [26/50], Train Losses: mse: 6.1252, mae: 1.0040, huber: 0.7628, swd: 2.4204, ept: 121.6320
    Epoch [26/50], Val Losses: mse: 2.8068, mae: 0.6375, huber: 0.4002, swd: 2.1267, ept: 188.2610
    Epoch [26/50], Test Losses: mse: 2.6497, mae: 0.6616, huber: 0.4130, swd: 1.9437, ept: 188.1190
      Epoch 26 composite train-obj: 0.762815
            Val objective improved 0.4163 → 0.4002, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 6.1070, mae: 0.9955, huber: 0.7573, swd: 2.3982, ept: 121.5828
    Epoch [27/50], Val Losses: mse: 3.1675, mae: 0.6248, huber: 0.4046, swd: 2.3783, ept: 187.4242
    Epoch [27/50], Test Losses: mse: 2.9533, mae: 0.6568, huber: 0.4234, swd: 2.1763, ept: 186.6265
      Epoch 27 composite train-obj: 0.757254
            No improvement (0.4046), counter 1/5
    Epoch [28/50], Train Losses: mse: 6.1207, mae: 0.9951, huber: 0.7571, swd: 2.4169, ept: 121.7165
    Epoch [28/50], Val Losses: mse: 3.3229, mae: 0.6245, huber: 0.4063, swd: 2.5184, ept: 186.8466
    Epoch [28/50], Test Losses: mse: 3.1760, mae: 0.6694, huber: 0.4368, swd: 2.3342, ept: 185.8117
      Epoch 28 composite train-obj: 0.757130
            No improvement (0.4063), counter 2/5
    Epoch [29/50], Train Losses: mse: 6.0705, mae: 0.9929, huber: 0.7546, swd: 2.3727, ept: 121.6492
    Epoch [29/50], Val Losses: mse: 3.1311, mae: 0.6337, huber: 0.4082, swd: 2.3814, ept: 187.2918
    Epoch [29/50], Test Losses: mse: 2.9639, mae: 0.6724, huber: 0.4325, swd: 2.1969, ept: 186.4898
      Epoch 29 composite train-obj: 0.754576
            No improvement (0.4082), counter 3/5
    Epoch [30/50], Train Losses: mse: 6.0574, mae: 0.9903, huber: 0.7530, swd: 2.3619, ept: 121.6657
    Epoch [30/50], Val Losses: mse: 2.9405, mae: 0.5961, huber: 0.3824, swd: 2.1691, ept: 188.4378
    Epoch [30/50], Test Losses: mse: 2.9191, mae: 0.6549, huber: 0.4250, swd: 2.0540, ept: 187.8961
      Epoch 30 composite train-obj: 0.753013
            Val objective improved 0.4002 → 0.3824, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 6.0427, mae: 0.9887, huber: 0.7515, swd: 2.3486, ept: 121.7152
    Epoch [31/50], Val Losses: mse: 2.9016, mae: 0.6277, huber: 0.4033, swd: 2.2388, ept: 187.0942
    Epoch [31/50], Test Losses: mse: 2.7381, mae: 0.6569, huber: 0.4201, swd: 2.0717, ept: 186.4348
      Epoch 31 composite train-obj: 0.751485
            No improvement (0.4033), counter 1/5
    Epoch [32/50], Train Losses: mse: 6.0298, mae: 0.9828, huber: 0.7476, swd: 2.3436, ept: 121.7258
    Epoch [32/50], Val Losses: mse: 2.5267, mae: 0.5950, huber: 0.3686, swd: 1.9545, ept: 188.9577
    Epoch [32/50], Test Losses: mse: 2.4609, mae: 0.6270, huber: 0.3876, swd: 1.7867, ept: 188.9542
      Epoch 32 composite train-obj: 0.747628
            Val objective improved 0.3824 → 0.3686, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 5.9872, mae: 0.9829, huber: 0.7459, swd: 2.3034, ept: 121.6455
    Epoch [33/50], Val Losses: mse: 3.3798, mae: 0.7571, huber: 0.4916, swd: 2.5109, ept: 186.9454
    Epoch [33/50], Test Losses: mse: 3.2675, mae: 0.7966, huber: 0.5200, swd: 2.3260, ept: 185.5490
      Epoch 33 composite train-obj: 0.745935
            No improvement (0.4916), counter 1/5
    Epoch [34/50], Train Losses: mse: 6.0425, mae: 0.9874, huber: 0.7505, swd: 2.3401, ept: 121.7254
    Epoch [34/50], Val Losses: mse: 3.1778, mae: 0.7164, huber: 0.4591, swd: 2.4436, ept: 187.5174
    Epoch [34/50], Test Losses: mse: 2.9821, mae: 0.7500, huber: 0.4784, swd: 2.2660, ept: 186.6618
      Epoch 34 composite train-obj: 0.750530
            No improvement (0.4591), counter 2/5
    Epoch [35/50], Train Losses: mse: 5.9597, mae: 0.9746, huber: 0.7403, swd: 2.2829, ept: 121.7232
    Epoch [35/50], Val Losses: mse: 3.5640, mae: 0.6517, huber: 0.4404, swd: 2.5726, ept: 186.5264
    Epoch [35/50], Test Losses: mse: 3.4169, mae: 0.7071, huber: 0.4786, swd: 2.4252, ept: 185.3943
      Epoch 35 composite train-obj: 0.740348
            No improvement (0.4404), counter 3/5
    Epoch [36/50], Train Losses: mse: 5.9833, mae: 0.9783, huber: 0.7432, swd: 2.2849, ept: 121.7039
    Epoch [36/50], Val Losses: mse: 3.3841, mae: 0.6774, huber: 0.4435, swd: 2.5949, ept: 186.4675
    Epoch [36/50], Test Losses: mse: 3.2294, mae: 0.7219, huber: 0.4740, swd: 2.4102, ept: 185.3886
      Epoch 36 composite train-obj: 0.743191
            No improvement (0.4435), counter 4/5
    Epoch [37/50], Train Losses: mse: 6.0055, mae: 0.9817, huber: 0.7457, swd: 2.3093, ept: 121.5777
    Epoch [37/50], Val Losses: mse: 3.4798, mae: 0.7805, huber: 0.5131, swd: 2.8021, ept: 187.1685
    Epoch [37/50], Test Losses: mse: 3.3565, mae: 0.8184, huber: 0.5395, swd: 2.6114, ept: 185.8570
      Epoch 37 composite train-obj: 0.745707
    Epoch [37/50], Test Losses: mse: 2.4609, mae: 0.6270, huber: 0.3876, swd: 1.7867, ept: 188.9542
    Best round's Test MSE: 2.4609, MAE: 0.6270, SWD: 1.7867
    Best round's Validation MSE: 2.5267, MAE: 0.5950, SWD: 1.9545
    Best round's Test verification MSE : 2.4609, MAE: 0.6270, SWD: 1.7867
    Time taken: 130.05 seconds
    
    ==================================================
    Experiment Summary (PatchTST_rossler_seq96_pred196_20250513_1328)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 3.1459 ± 0.5099
      mae: 0.6975 ± 0.0616
      huber: 0.4556 ± 0.0551
      swd: 2.3850 ± 0.4256
      ept: 186.9184 ± 1.4540
      count: 39.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 3.1498 ± 0.4426
      mae: 0.6416 ± 0.0400
      huber: 0.4152 ± 0.0354
      swd: 2.5516 ± 0.4234
      ept: 187.4094 ± 1.0963
      count: 39.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 238.57 seconds
    
    Experiment complete: PatchTST_rossler_seq96_pred196_20250513_1328
    Model: PatchTST
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['rossler']['channels'],
    enc_in=data_mgr.datasets['rossler']['channels'],
    dec_in=data_mgr.datasets['rossler']['channels'],
    c_out=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 26.9142, mae: 2.9954, huber: 2.6311, swd: 11.0698, ept: 156.5289
    Epoch [1/50], Val Losses: mse: 20.8846, mae: 2.7197, huber: 2.3550, swd: 8.6559, ept: 245.9946
    Epoch [1/50], Test Losses: mse: 22.2698, mae: 2.8403, huber: 2.4667, swd: 8.1669, ept: 239.3564
      Epoch 1 composite train-obj: 2.631112
            Val objective improved inf → 2.3550, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 16.6029, mae: 2.2056, huber: 1.8617, swd: 7.8230, ept: 158.4676
    Epoch [2/50], Val Losses: mse: 30.6932, mae: 2.7704, huber: 2.4111, swd: 20.8010, ept: 254.8288
    Epoch [2/50], Test Losses: mse: 34.3168, mae: 3.0524, huber: 2.6810, swd: 23.0517, ept: 242.6045
      Epoch 2 composite train-obj: 1.861694
            No improvement (2.4111), counter 1/5
    Epoch [3/50], Train Losses: mse: 15.6577, mae: 2.1352, huber: 1.7926, swd: 7.5016, ept: 159.5214
    Epoch [3/50], Val Losses: mse: 11.2507, mae: 1.9507, huber: 1.6085, swd: 6.1329, ept: 270.0220
    Epoch [3/50], Test Losses: mse: 13.2292, mae: 2.1058, huber: 1.7520, swd: 6.7309, ept: 264.4286
      Epoch 3 composite train-obj: 1.792616
            Val objective improved 2.3550 → 1.6085, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.1353, mae: 1.8129, huber: 1.4866, swd: 5.3461, ept: 160.5653
    Epoch [4/50], Val Losses: mse: 9.3933, mae: 1.4959, huber: 1.1880, swd: 4.3356, ept: 283.3431
    Epoch [4/50], Test Losses: mse: 9.9180, mae: 1.6171, huber: 1.2940, swd: 4.1938, ept: 278.2959
      Epoch 4 composite train-obj: 1.486621
            Val objective improved 1.6085 → 1.1880, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.5703, mae: 1.7280, huber: 1.4085, swd: 4.9541, ept: 161.2672
    Epoch [5/50], Val Losses: mse: 6.2747, mae: 1.2894, huber: 0.9779, swd: 4.0267, ept: 298.7832
    Epoch [5/50], Test Losses: mse: 6.4025, mae: 1.3606, huber: 1.0382, swd: 3.8325, ept: 294.6842
      Epoch 5 composite train-obj: 1.408469
            Val objective improved 1.1880 → 0.9779, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.5236, mae: 1.6170, huber: 1.3040, swd: 4.3498, ept: 162.5294
    Epoch [6/50], Val Losses: mse: 8.3584, mae: 1.5967, huber: 1.2497, swd: 4.6961, ept: 290.4807
    Epoch [6/50], Test Losses: mse: 9.0496, mae: 1.7129, huber: 1.3571, swd: 4.8947, ept: 285.6379
      Epoch 6 composite train-obj: 1.304020
            No improvement (1.2497), counter 1/5
    Epoch [7/50], Train Losses: mse: 10.2758, mae: 1.5800, huber: 1.2711, swd: 4.2470, ept: 162.3699
    Epoch [7/50], Val Losses: mse: 8.0493, mae: 1.4425, huber: 1.1212, swd: 4.9373, ept: 285.4697
    Epoch [7/50], Test Losses: mse: 8.6866, mae: 1.5523, huber: 1.2169, swd: 5.1988, ept: 278.4477
      Epoch 7 composite train-obj: 1.271140
            No improvement (1.1212), counter 2/5
    Epoch [8/50], Train Losses: mse: 10.5694, mae: 1.6209, huber: 1.3068, swd: 4.3866, ept: 162.3476
    Epoch [8/50], Val Losses: mse: 6.1065, mae: 1.2069, huber: 0.8931, swd: 4.1757, ept: 301.5551
    Epoch [8/50], Test Losses: mse: 6.6238, mae: 1.3090, huber: 0.9838, swd: 4.6163, ept: 297.0786
      Epoch 8 composite train-obj: 1.306784
            Val objective improved 0.9779 → 0.8931, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 9.9494, mae: 1.5284, huber: 1.2236, swd: 4.0631, ept: 162.7914
    Epoch [9/50], Val Losses: mse: 5.2220, mae: 1.0883, huber: 0.7870, swd: 3.6427, ept: 303.9272
    Epoch [9/50], Test Losses: mse: 5.2354, mae: 1.1505, huber: 0.8395, swd: 3.6646, ept: 300.3153
      Epoch 9 composite train-obj: 1.223553
            Val objective improved 0.8931 → 0.7870, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 9.8302, mae: 1.5084, huber: 1.2062, swd: 4.0010, ept: 162.6880
    Epoch [10/50], Val Losses: mse: 5.1163, mae: 1.0750, huber: 0.7821, swd: 3.5358, ept: 302.8771
    Epoch [10/50], Test Losses: mse: 5.0142, mae: 1.1257, huber: 0.8210, swd: 3.3595, ept: 300.0369
      Epoch 10 composite train-obj: 1.206171
            Val objective improved 0.7870 → 0.7821, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 9.6381, mae: 1.4907, huber: 1.1894, swd: 3.8725, ept: 163.2549
    Epoch [11/50], Val Losses: mse: 5.6475, mae: 1.2000, huber: 0.8900, swd: 3.6154, ept: 304.4798
    Epoch [11/50], Test Losses: mse: 5.6473, mae: 1.2509, huber: 0.9314, swd: 3.5167, ept: 301.8516
      Epoch 11 composite train-obj: 1.189439
            No improvement (0.8900), counter 1/5
    Epoch [12/50], Train Losses: mse: 9.5199, mae: 1.4711, huber: 1.1716, swd: 3.8140, ept: 162.9245
    Epoch [12/50], Val Losses: mse: 6.3440, mae: 1.2974, huber: 0.9815, swd: 3.9916, ept: 297.8496
    Epoch [12/50], Test Losses: mse: 6.4588, mae: 1.3717, huber: 1.0456, swd: 4.0090, ept: 291.8440
      Epoch 12 composite train-obj: 1.171590
            No improvement (0.9815), counter 2/5
    Epoch [13/50], Train Losses: mse: 9.4824, mae: 1.4662, huber: 1.1672, swd: 3.7980, ept: 162.7732
    Epoch [13/50], Val Losses: mse: 6.0570, mae: 1.1602, huber: 0.8525, swd: 3.9106, ept: 296.8801
    Epoch [13/50], Test Losses: mse: 6.2251, mae: 1.2409, huber: 0.9180, swd: 3.9545, ept: 293.4841
      Epoch 13 composite train-obj: 1.167159
            No improvement (0.8525), counter 3/5
    Epoch [14/50], Train Losses: mse: 9.8598, mae: 1.5306, huber: 1.2217, swd: 4.0307, ept: 163.0462
    Epoch [14/50], Val Losses: mse: 6.3425, mae: 1.2943, huber: 0.9875, swd: 3.7229, ept: 296.5671
    Epoch [14/50], Test Losses: mse: 6.2886, mae: 1.3575, huber: 1.0401, swd: 3.5154, ept: 292.3382
      Epoch 14 composite train-obj: 1.221716
            No improvement (0.9875), counter 4/5
    Epoch [15/50], Train Losses: mse: 9.5417, mae: 1.4713, huber: 1.1715, swd: 3.8350, ept: 162.8835
    Epoch [15/50], Val Losses: mse: 14.0072, mae: 1.9267, huber: 1.6002, swd: 7.5024, ept: 277.6540
    Epoch [15/50], Test Losses: mse: 14.7492, mae: 2.0721, huber: 1.7362, swd: 7.5899, ept: 270.3904
      Epoch 15 composite train-obj: 1.171482
    Epoch [15/50], Test Losses: mse: 5.0142, mae: 1.1257, huber: 0.8210, swd: 3.3595, ept: 300.0369
    Best round's Test MSE: 5.0142, MAE: 1.1257, SWD: 3.3595
    Best round's Validation MSE: 5.1163, MAE: 1.0750, SWD: 3.5358
    Best round's Test verification MSE : 5.0142, MAE: 1.1257, SWD: 3.3595
    Time taken: 53.33 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 27.3170, mae: 3.0269, huber: 2.6618, swd: 11.5441, ept: 156.6662
    Epoch [1/50], Val Losses: mse: 15.2358, mae: 2.2826, huber: 1.9376, swd: 8.1478, ept: 250.8973
    Epoch [1/50], Test Losses: mse: 14.8908, mae: 2.3425, huber: 1.9868, swd: 7.3272, ept: 244.6083
      Epoch 1 composite train-obj: 2.661792
            Val objective improved inf → 1.9376, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 16.3727, mae: 2.1947, huber: 1.8529, swd: 7.5797, ept: 159.0552
    Epoch [2/50], Val Losses: mse: 14.5063, mae: 1.8356, huber: 1.5002, swd: 10.5759, ept: 279.0530
    Epoch [2/50], Test Losses: mse: 13.0722, mae: 1.8447, huber: 1.5007, swd: 8.5914, ept: 278.4328
      Epoch 2 composite train-obj: 1.852873
            Val objective improved 1.9376 → 1.5002, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 13.9390, mae: 1.9547, huber: 1.6239, swd: 6.5473, ept: 160.5010
    Epoch [3/50], Val Losses: mse: 9.8110, mae: 1.7961, huber: 1.4443, swd: 5.3812, ept: 276.9411
    Epoch [3/50], Test Losses: mse: 10.8837, mae: 1.9160, huber: 1.5547, swd: 5.5275, ept: 273.7057
      Epoch 3 composite train-obj: 1.623865
            Val objective improved 1.5002 → 1.4443, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.5072, mae: 1.8113, huber: 1.4879, swd: 5.7823, ept: 160.8577
    Epoch [4/50], Val Losses: mse: 10.2483, mae: 1.6072, huber: 1.2716, swd: 7.2733, ept: 293.2550
    Epoch [4/50], Test Losses: mse: 9.6873, mae: 1.6709, huber: 1.3258, swd: 6.2297, ept: 288.9525
      Epoch 4 composite train-obj: 1.487921
            Val objective improved 1.4443 → 1.2716, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.6360, mae: 1.7187, huber: 1.3992, swd: 5.2629, ept: 161.3303
    Epoch [5/50], Val Losses: mse: 6.7377, mae: 1.2680, huber: 0.9619, swd: 4.5872, ept: 293.7506
    Epoch [5/50], Test Losses: mse: 7.0129, mae: 1.3411, huber: 1.0218, swd: 4.4250, ept: 290.7574
      Epoch 5 composite train-obj: 1.399212
            Val objective improved 1.2716 → 0.9619, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.1089, mae: 1.6613, huber: 1.3452, swd: 4.9123, ept: 161.3850
    Epoch [6/50], Val Losses: mse: 7.8371, mae: 1.4236, huber: 1.1026, swd: 5.3700, ept: 290.5150
    Epoch [6/50], Test Losses: mse: 8.8344, mae: 1.5424, huber: 1.2092, swd: 5.5416, ept: 283.4827
      Epoch 6 composite train-obj: 1.345216
            No improvement (1.1026), counter 1/5
    Epoch [7/50], Train Losses: mse: 10.6334, mae: 1.6140, huber: 1.3011, swd: 4.6093, ept: 162.5291
    Epoch [7/50], Val Losses: mse: 5.6190, mae: 1.1676, huber: 0.8560, swd: 4.0303, ept: 303.2759
    Epoch [7/50], Test Losses: mse: 6.2564, mae: 1.2475, huber: 0.9278, swd: 4.0980, ept: 300.3516
      Epoch 7 composite train-obj: 1.301091
            Val objective improved 0.9619 → 0.8560, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.2231, mae: 1.5630, huber: 1.2549, swd: 4.3580, ept: 162.5152
    Epoch [8/50], Val Losses: mse: 6.6152, mae: 1.2777, huber: 0.9538, swd: 4.3195, ept: 292.0271
    Epoch [8/50], Test Losses: mse: 6.7550, mae: 1.3598, huber: 1.0243, swd: 4.2034, ept: 288.3083
      Epoch 8 composite train-obj: 1.254907
            No improvement (0.9538), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.4858, mae: 1.6108, huber: 1.2967, swd: 4.4556, ept: 161.6469
    Epoch [9/50], Val Losses: mse: 31.6396, mae: 1.9859, huber: 1.6317, swd: 25.7309, ept: 289.3717
    Epoch [9/50], Test Losses: mse: 30.2471, mae: 2.1329, huber: 1.7686, swd: 23.9245, ept: 282.6867
      Epoch 9 composite train-obj: 1.296685
            No improvement (1.6317), counter 2/5
    Epoch [10/50], Train Losses: mse: 13.2286, mae: 1.7942, huber: 1.4704, swd: 6.3157, ept: 161.6212
    Epoch [10/50], Val Losses: mse: 10.5745, mae: 1.7597, huber: 1.4382, swd: 5.2070, ept: 284.3179
    Epoch [10/50], Test Losses: mse: 10.8213, mae: 1.8756, huber: 1.5438, swd: 5.0227, ept: 276.3725
      Epoch 10 composite train-obj: 1.470437
            No improvement (1.4382), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.3947, mae: 1.5958, huber: 1.2842, swd: 4.3668, ept: 162.8230
    Epoch [11/50], Val Losses: mse: 5.9265, mae: 1.2690, huber: 0.9440, swd: 4.0134, ept: 298.9812
    Epoch [11/50], Test Losses: mse: 5.9354, mae: 1.3197, huber: 0.9840, swd: 3.8747, ept: 297.6318
      Epoch 11 composite train-obj: 1.284200
            No improvement (0.9440), counter 4/5
    Epoch [12/50], Train Losses: mse: 9.8521, mae: 1.5224, huber: 1.2177, swd: 4.0907, ept: 162.4610
    Epoch [12/50], Val Losses: mse: 11.7848, mae: 1.9460, huber: 1.6005, swd: 5.1756, ept: 272.5903
    Epoch [12/50], Test Losses: mse: 12.2099, mae: 2.0715, huber: 1.7154, swd: 5.0976, ept: 264.4924
      Epoch 12 composite train-obj: 1.217746
    Epoch [12/50], Test Losses: mse: 6.2564, mae: 1.2475, huber: 0.9278, swd: 4.0980, ept: 300.3516
    Best round's Test MSE: 6.2564, MAE: 1.2475, SWD: 4.0980
    Best round's Validation MSE: 5.6190, MAE: 1.1676, SWD: 4.0303
    Best round's Test verification MSE : 6.2564, MAE: 1.2475, SWD: 4.0980
    Time taken: 42.64 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.2510, mae: 2.8323, huber: 2.4691, swd: 10.5534, ept: 156.8237
    Epoch [1/50], Val Losses: mse: 13.5768, mae: 2.1975, huber: 1.8456, swd: 7.1584, ept: 260.8658
    Epoch [1/50], Test Losses: mse: 13.8867, mae: 2.2438, huber: 1.8848, swd: 6.7326, ept: 257.2181
      Epoch 1 composite train-obj: 2.469088
            Val objective improved inf → 1.8456, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 15.0035, mae: 2.0524, huber: 1.7138, swd: 7.4762, ept: 159.3025
    Epoch [2/50], Val Losses: mse: 19.9900, mae: 2.3271, huber: 1.9596, swd: 14.3611, ept: 263.8271
    Epoch [2/50], Test Losses: mse: 19.3572, mae: 2.4625, huber: 2.0863, swd: 12.9272, ept: 256.4820
      Epoch 2 composite train-obj: 1.713795
            No improvement (1.9596), counter 1/5
    Epoch [3/50], Train Losses: mse: 12.9329, mae: 1.8747, huber: 1.5444, swd: 6.0514, ept: 160.3610
    Epoch [3/50], Val Losses: mse: 8.9589, mae: 1.6906, huber: 1.3573, swd: 4.6352, ept: 277.9932
    Epoch [3/50], Test Losses: mse: 10.8138, mae: 1.8572, huber: 1.5103, swd: 5.5923, ept: 272.8256
      Epoch 3 composite train-obj: 1.544419
            Val objective improved 1.8456 → 1.3573, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.6063, mae: 1.7424, huber: 1.4204, swd: 5.1186, ept: 161.1656
    Epoch [4/50], Val Losses: mse: 10.3661, mae: 1.7865, huber: 1.4493, swd: 6.8865, ept: 278.9169
    Epoch [4/50], Test Losses: mse: 10.3281, mae: 1.8687, huber: 1.5194, swd: 6.4671, ept: 271.3202
      Epoch 4 composite train-obj: 1.420423
            No improvement (1.4493), counter 1/5
    Epoch [5/50], Train Losses: mse: 11.6853, mae: 1.7542, huber: 1.4316, swd: 5.1904, ept: 160.8846
    Epoch [5/50], Val Losses: mse: 8.0000, mae: 1.5914, huber: 1.2624, swd: 4.5605, ept: 282.5287
    Epoch [5/50], Test Losses: mse: 9.1984, mae: 1.7452, huber: 1.4038, swd: 4.8805, ept: 276.8116
      Epoch 5 composite train-obj: 1.431612
            Val objective improved 1.3573 → 1.2624, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 10.9441, mae: 1.6654, huber: 1.3489, swd: 4.7063, ept: 162.0537
    Epoch [6/50], Val Losses: mse: 23.3177, mae: 3.1788, huber: 2.8119, swd: 6.8975, ept: 205.9035
    Epoch [6/50], Test Losses: mse: 24.6391, mae: 3.3315, huber: 2.9559, swd: 6.7655, ept: 196.1378
      Epoch 6 composite train-obj: 1.348877
            No improvement (2.8119), counter 1/5
    Epoch [7/50], Train Losses: mse: 12.4404, mae: 1.8593, huber: 1.5279, swd: 5.4501, ept: 161.4414
    Epoch [7/50], Val Losses: mse: 5.7801, mae: 1.2199, huber: 0.9085, swd: 4.0232, ept: 300.6268
    Epoch [7/50], Test Losses: mse: 6.0332, mae: 1.3147, huber: 0.9894, swd: 4.0161, ept: 296.9401
      Epoch 7 composite train-obj: 1.527884
            Val objective improved 1.2624 → 0.9085, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.2066, mae: 1.5840, huber: 1.2708, swd: 4.3447, ept: 162.7964
    Epoch [8/50], Val Losses: mse: 7.0568, mae: 1.4344, huber: 1.1027, swd: 4.2785, ept: 295.7190
    Epoch [8/50], Test Losses: mse: 7.7411, mae: 1.5449, huber: 1.2042, swd: 4.4202, ept: 291.7087
      Epoch 8 composite train-obj: 1.270764
            No improvement (1.1027), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.0377, mae: 1.5550, huber: 1.2446, swd: 4.2178, ept: 163.1241
    Epoch [9/50], Val Losses: mse: 5.8186, mae: 1.2625, huber: 0.9377, swd: 3.9413, ept: 303.8686
    Epoch [9/50], Test Losses: mse: 6.5445, mae: 1.3646, huber: 1.0311, swd: 4.1298, ept: 299.7991
      Epoch 9 composite train-obj: 1.244627
            No improvement (0.9377), counter 2/5
    Epoch [10/50], Train Losses: mse: 9.8847, mae: 1.5249, huber: 1.2178, swd: 4.1799, ept: 162.9624
    Epoch [10/50], Val Losses: mse: 5.0637, mae: 1.0696, huber: 0.7741, swd: 3.6781, ept: 305.1203
    Epoch [10/50], Test Losses: mse: 5.0909, mae: 1.1172, huber: 0.8111, swd: 3.5303, ept: 303.5315
      Epoch 10 composite train-obj: 1.217797
            Val objective improved 0.9085 → 0.7741, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 9.6843, mae: 1.4961, huber: 1.1921, swd: 4.0536, ept: 162.9822
    Epoch [11/50], Val Losses: mse: 12.5387, mae: 2.0353, huber: 1.6884, swd: 4.9923, ept: 266.7101
    Epoch [11/50], Test Losses: mse: 12.4745, mae: 2.1110, huber: 1.7547, swd: 4.8502, ept: 260.8789
      Epoch 11 composite train-obj: 1.192069
            No improvement (1.6884), counter 1/5
    Epoch [12/50], Train Losses: mse: 11.1408, mae: 1.7028, huber: 1.3821, swd: 4.7195, ept: 162.3185
    Epoch [12/50], Val Losses: mse: 6.0037, mae: 1.3258, huber: 1.0056, swd: 4.0248, ept: 299.9115
    Epoch [12/50], Test Losses: mse: 5.9938, mae: 1.3929, huber: 1.0607, swd: 3.8990, ept: 296.4994
      Epoch 12 composite train-obj: 1.382068
            No improvement (1.0056), counter 2/5
    Epoch [13/50], Train Losses: mse: 9.9304, mae: 1.5517, huber: 1.2410, swd: 4.1637, ept: 162.8629
    Epoch [13/50], Val Losses: mse: 5.5065, mae: 1.1471, huber: 0.8513, swd: 3.9488, ept: 300.9866
    Epoch [13/50], Test Losses: mse: 5.5468, mae: 1.2002, huber: 0.8921, swd: 3.9372, ept: 299.0414
      Epoch 13 composite train-obj: 1.240964
            No improvement (0.8513), counter 3/5
    Epoch [14/50], Train Losses: mse: 9.6473, mae: 1.4951, huber: 1.1910, swd: 4.0421, ept: 162.7891
    Epoch [14/50], Val Losses: mse: 5.8765, mae: 1.1204, huber: 0.8336, swd: 3.7942, ept: 299.1104
    Epoch [14/50], Test Losses: mse: 6.0049, mae: 1.2039, huber: 0.9034, swd: 3.7026, ept: 295.3671
      Epoch 14 composite train-obj: 1.190966
            No improvement (0.8336), counter 4/5
    Epoch [15/50], Train Losses: mse: 9.7082, mae: 1.5130, huber: 1.2054, swd: 4.0673, ept: 163.5636
    Epoch [15/50], Val Losses: mse: 5.1192, mae: 1.0080, huber: 0.7254, swd: 3.7194, ept: 303.2760
    Epoch [15/50], Test Losses: mse: 5.2486, mae: 1.0699, huber: 0.7731, swd: 3.7066, ept: 301.5483
      Epoch 15 composite train-obj: 1.205387
            Val objective improved 0.7741 → 0.7254, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 9.4601, mae: 1.4540, huber: 1.1546, swd: 3.9565, ept: 163.3059
    Epoch [16/50], Val Losses: mse: 6.5232, mae: 1.2722, huber: 0.9677, swd: 4.2767, ept: 300.9662
    Epoch [16/50], Test Losses: mse: 7.7503, mae: 1.4140, huber: 1.0956, swd: 5.1484, ept: 295.0771
      Epoch 16 composite train-obj: 1.154642
            No improvement (0.9677), counter 1/5
    Epoch [17/50], Train Losses: mse: 9.7283, mae: 1.4899, huber: 1.1872, swd: 4.1354, ept: 163.4224
    Epoch [17/50], Val Losses: mse: 9.6383, mae: 1.6983, huber: 1.3683, swd: 5.6827, ept: 289.3294
    Epoch [17/50], Test Losses: mse: 11.1729, mae: 1.8468, huber: 1.5055, swd: 6.4964, ept: 282.1103
      Epoch 17 composite train-obj: 1.187153
            No improvement (1.3683), counter 2/5
    Epoch [18/50], Train Losses: mse: 9.5916, mae: 1.4781, huber: 1.1770, swd: 3.9823, ept: 163.4675
    Epoch [18/50], Val Losses: mse: 5.3211, mae: 1.0683, huber: 0.7755, swd: 3.6407, ept: 303.9329
    Epoch [18/50], Test Losses: mse: 5.1834, mae: 1.1386, huber: 0.8306, swd: 3.5333, ept: 300.7463
      Epoch 18 composite train-obj: 1.176993
            No improvement (0.7755), counter 3/5
    Epoch [19/50], Train Losses: mse: 9.2515, mae: 1.4224, huber: 1.1269, swd: 3.8239, ept: 163.6627
    Epoch [19/50], Val Losses: mse: 5.3849, mae: 1.1299, huber: 0.8245, swd: 3.7505, ept: 304.8101
    Epoch [19/50], Test Losses: mse: 5.3101, mae: 1.1699, huber: 0.8538, swd: 3.6373, ept: 303.2966
      Epoch 19 composite train-obj: 1.126932
            No improvement (0.8245), counter 4/5
    Epoch [20/50], Train Losses: mse: 9.3924, mae: 1.4319, huber: 1.1368, swd: 3.8944, ept: 163.3389
    Epoch [20/50], Val Losses: mse: 9.6249, mae: 1.6010, huber: 1.2784, swd: 4.5348, ept: 282.4390
    Epoch [20/50], Test Losses: mse: 10.1544, mae: 1.7001, huber: 1.3650, swd: 4.5369, ept: 275.6811
      Epoch 20 composite train-obj: 1.136795
    Epoch [20/50], Test Losses: mse: 5.2486, mae: 1.0699, huber: 0.7731, swd: 3.7066, ept: 301.5483
    Best round's Test MSE: 5.2486, MAE: 1.0699, SWD: 3.7066
    Best round's Validation MSE: 5.1192, MAE: 1.0080, SWD: 3.7194
    Best round's Test verification MSE : 5.2486, MAE: 1.0699, SWD: 3.7066
    Time taken: 70.67 seconds
    
    ==================================================
    Experiment Summary (PatchTST_rossler_seq96_pred336_20250513_1332)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 5.5064 ± 0.5389
      mae: 1.1477 ± 0.0742
      huber: 0.8406 ± 0.0647
      swd: 3.7213 ± 0.3017
      ept: 300.6456 ± 0.6511
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.2849 ± 0.2363
      mae: 1.0835 ± 0.0654
      huber: 0.7878 ± 0.0535
      swd: 3.7618 ± 0.2041
      ept: 303.1430 ± 0.1880
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 166.70 seconds
    
    Experiment complete: PatchTST_rossler_seq96_pred336_20250513_1332
    Model: PatchTST
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['rossler']['channels'],
    enc_in=data_mgr.datasets['rossler']['channels'],
    dec_in=data_mgr.datasets['rossler']['channels'],
    c_out=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 279
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 21.7126, mae: 2.7784, huber: 2.4121, swd: 9.1205, ept: 222.9395
    Epoch [1/50], Val Losses: mse: 13.3071, mae: 2.1531, huber: 1.7975, swd: 6.4404, ept: 471.1586
    Epoch [1/50], Test Losses: mse: 13.5146, mae: 2.1232, huber: 1.7614, swd: 6.4714, ept: 464.0016
      Epoch 1 composite train-obj: 2.412059
            Val objective improved inf → 1.7975, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 14.1446, mae: 2.1791, huber: 1.8279, swd: 6.0353, ept: 227.3057
    Epoch [2/50], Val Losses: mse: 10.3074, mae: 1.8437, huber: 1.5020, swd: 5.2514, ept: 514.0230
    Epoch [2/50], Test Losses: mse: 9.6604, mae: 1.7746, huber: 1.4270, swd: 4.7674, ept: 508.5908
      Epoch 2 composite train-obj: 1.827879
            Val objective improved 1.7975 → 1.5020, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.0966, mae: 1.9850, huber: 1.6420, swd: 5.1200, ept: 229.2350
    Epoch [3/50], Val Losses: mse: 8.9062, mae: 1.7717, huber: 1.4236, swd: 5.1604, ept: 540.2859
    Epoch [3/50], Test Losses: mse: 7.8402, mae: 1.6582, huber: 1.3082, swd: 4.4189, ept: 544.6594
      Epoch 3 composite train-obj: 1.641981
            Val objective improved 1.5020 → 1.4236, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.3821, mae: 1.9148, huber: 1.5753, swd: 4.7870, ept: 230.7984
    Epoch [4/50], Val Losses: mse: 8.9944, mae: 1.6741, huber: 1.3518, swd: 5.2119, ept: 547.7082
    Epoch [4/50], Test Losses: mse: 7.9092, mae: 1.5907, huber: 1.2608, swd: 4.4459, ept: 544.5861
      Epoch 4 composite train-obj: 1.575320
            Val objective improved 1.4236 → 1.3518, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.1180, mae: 1.8767, huber: 1.5396, swd: 4.6686, ept: 231.9940
    Epoch [5/50], Val Losses: mse: 8.7100, mae: 1.6892, huber: 1.3540, swd: 4.7405, ept: 555.7799
    Epoch [5/50], Test Losses: mse: 7.5061, mae: 1.5832, huber: 1.2435, swd: 3.9378, ept: 552.2871
      Epoch 5 composite train-obj: 1.539600
            No improvement (1.3540), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.8157, mae: 1.8366, huber: 1.5027, swd: 4.5050, ept: 232.4574
    Epoch [6/50], Val Losses: mse: 10.8206, mae: 1.9142, huber: 1.5708, swd: 4.9172, ept: 516.0845
    Epoch [6/50], Test Losses: mse: 9.3285, mae: 1.7915, huber: 1.4445, swd: 4.1564, ept: 512.2974
      Epoch 6 composite train-obj: 1.502673
            No improvement (1.5708), counter 2/5
    Epoch [7/50], Train Losses: mse: 10.8711, mae: 1.8507, huber: 1.5156, swd: 4.4819, ept: 232.2335
    Epoch [7/50], Val Losses: mse: 8.3551, mae: 1.6906, huber: 1.3466, swd: 4.7273, ept: 552.1127
    Epoch [7/50], Test Losses: mse: 7.1678, mae: 1.5846, huber: 1.2359, swd: 3.9545, ept: 549.0085
      Epoch 7 composite train-obj: 1.515589
            Val objective improved 1.3518 → 1.3466, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.4909, mae: 1.7976, huber: 1.4660, swd: 4.3505, ept: 233.2547
    Epoch [8/50], Val Losses: mse: 8.6563, mae: 1.6829, huber: 1.3513, swd: 4.7099, ept: 540.5271
    Epoch [8/50], Test Losses: mse: 7.2969, mae: 1.5511, huber: 1.2170, swd: 3.8142, ept: 539.2248
      Epoch 8 composite train-obj: 1.466021
            No improvement (1.3513), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.5800, mae: 1.8036, huber: 1.4723, swd: 4.4028, ept: 233.8015
    Epoch [9/50], Val Losses: mse: 8.3750, mae: 1.5533, huber: 1.2346, swd: 4.6700, ept: 564.1335
    Epoch [9/50], Test Losses: mse: 7.2634, mae: 1.4620, huber: 1.1394, swd: 4.0328, ept: 562.9473
      Epoch 9 composite train-obj: 1.472302
            Val objective improved 1.3466 → 1.2346, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 10.3343, mae: 1.7643, huber: 1.4367, swd: 4.3086, ept: 233.3480
    Epoch [10/50], Val Losses: mse: 9.3869, mae: 1.7755, huber: 1.4350, swd: 4.9284, ept: 533.5660
    Epoch [10/50], Test Losses: mse: 8.1638, mae: 1.6989, huber: 1.3517, swd: 4.1648, ept: 525.5760
      Epoch 10 composite train-obj: 1.436686
            No improvement (1.4350), counter 1/5
    Epoch [11/50], Train Losses: mse: 10.4632, mae: 1.7929, huber: 1.4621, swd: 4.3301, ept: 233.4367
    Epoch [11/50], Val Losses: mse: 9.5124, mae: 1.8289, huber: 1.4831, swd: 4.7146, ept: 519.7346
    Epoch [11/50], Test Losses: mse: 8.2519, mae: 1.7314, huber: 1.3802, swd: 4.0049, ept: 520.7892
      Epoch 11 composite train-obj: 1.462075
            No improvement (1.4831), counter 2/5
    Epoch [12/50], Train Losses: mse: 10.2613, mae: 1.7640, huber: 1.4356, swd: 4.2719, ept: 233.4634
    Epoch [12/50], Val Losses: mse: 8.4667, mae: 1.6380, huber: 1.3105, swd: 4.6459, ept: 547.2045
    Epoch [12/50], Test Losses: mse: 7.1921, mae: 1.5286, huber: 1.1962, swd: 3.8268, ept: 547.8775
      Epoch 12 composite train-obj: 1.435593
            No improvement (1.3105), counter 3/5
    Epoch [13/50], Train Losses: mse: 10.3446, mae: 1.7739, huber: 1.4442, swd: 4.2982, ept: 233.6622
    Epoch [13/50], Val Losses: mse: 8.9955, mae: 1.6063, huber: 1.2842, swd: 4.7055, ept: 553.0672
    Epoch [13/50], Test Losses: mse: 7.9423, mae: 1.5290, huber: 1.2001, swd: 4.2212, ept: 546.6646
      Epoch 13 composite train-obj: 1.444175
            No improvement (1.2842), counter 4/5
    Epoch [14/50], Train Losses: mse: 10.2737, mae: 1.7494, huber: 1.4233, swd: 4.3323, ept: 233.5240
    Epoch [14/50], Val Losses: mse: 10.8913, mae: 1.9444, huber: 1.5973, swd: 4.8555, ept: 508.6198
    Epoch [14/50], Test Losses: mse: 9.8358, mae: 1.8757, huber: 1.5222, swd: 4.1606, ept: 499.3461
      Epoch 14 composite train-obj: 1.423282
    Epoch [14/50], Test Losses: mse: 7.2634, mae: 1.4620, huber: 1.1394, swd: 4.0328, ept: 562.9473
    Best round's Test MSE: 7.2634, MAE: 1.4620, SWD: 4.0328
    Best round's Validation MSE: 8.3750, MAE: 1.5533, SWD: 4.6700
    Best round's Test verification MSE : 7.2634, MAE: 1.4620, SWD: 4.0328
    Time taken: 52.08 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 21.2113, mae: 2.7706, huber: 2.4032, swd: 8.3423, ept: 222.0413
    Epoch [1/50], Val Losses: mse: 16.3961, mae: 2.2689, huber: 1.9129, swd: 8.9154, ept: 453.8560
    Epoch [1/50], Test Losses: mse: 16.1513, mae: 2.2201, huber: 1.8588, swd: 8.6546, ept: 440.8175
      Epoch 1 composite train-obj: 2.403213
            Val objective improved inf → 1.9129, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 14.3592, mae: 2.2144, huber: 1.8615, swd: 5.9521, ept: 227.2355
    Epoch [2/50], Val Losses: mse: 11.3493, mae: 2.0384, huber: 1.6836, swd: 5.5720, ept: 495.3594
    Epoch [2/50], Test Losses: mse: 10.7434, mae: 1.9669, huber: 1.6093, swd: 5.2610, ept: 485.3552
      Epoch 2 composite train-obj: 1.861510
            Val objective improved 1.9129 → 1.6836, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.4471, mae: 2.0331, huber: 1.6872, swd: 5.1617, ept: 229.1789
    Epoch [3/50], Val Losses: mse: 9.7984, mae: 1.9080, huber: 1.5593, swd: 5.0253, ept: 514.6861
    Epoch [3/50], Test Losses: mse: 8.6913, mae: 1.7911, huber: 1.4389, swd: 4.2594, ept: 512.0881
      Epoch 3 composite train-obj: 1.687210
            Val objective improved 1.6836 → 1.5593, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.5241, mae: 1.9332, huber: 1.5923, swd: 4.7324, ept: 230.8012
    Epoch [4/50], Val Losses: mse: 9.0269, mae: 1.7207, huber: 1.3803, swd: 5.0091, ept: 556.2692
    Epoch [4/50], Test Losses: mse: 7.9074, mae: 1.5995, huber: 1.2584, swd: 4.3090, ept: 554.4297
      Epoch 4 composite train-obj: 1.592331
            Val objective improved 1.5593 → 1.3803, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.0157, mae: 1.8696, huber: 1.5328, swd: 4.5304, ept: 231.1761
    Epoch [5/50], Val Losses: mse: 10.2389, mae: 1.8902, huber: 1.5501, swd: 5.2981, ept: 492.1367
    Epoch [5/50], Test Losses: mse: 9.7153, mae: 1.8289, huber: 1.4829, swd: 4.7496, ept: 489.2140
      Epoch 5 composite train-obj: 1.532755
            No improvement (1.5501), counter 1/5
    Epoch [6/50], Train Losses: mse: 10.8122, mae: 1.8427, huber: 1.5074, swd: 4.4119, ept: 232.1974
    Epoch [6/50], Val Losses: mse: 10.6132, mae: 1.8963, huber: 1.5551, swd: 6.2456, ept: 506.4060
    Epoch [6/50], Test Losses: mse: 9.4264, mae: 1.7976, huber: 1.4511, swd: 5.3213, ept: 507.9453
      Epoch 6 composite train-obj: 1.507360
            No improvement (1.5551), counter 2/5
    Epoch [7/50], Train Losses: mse: 11.2818, mae: 1.8962, huber: 1.5597, swd: 4.6701, ept: 231.9907
    Epoch [7/50], Val Losses: mse: 8.5853, mae: 1.6131, huber: 1.2828, swd: 4.6079, ept: 558.8831
    Epoch [7/50], Test Losses: mse: 7.2914, mae: 1.5028, huber: 1.1690, swd: 3.8409, ept: 556.3720
      Epoch 7 composite train-obj: 1.559678
            Val objective improved 1.3803 → 1.2828, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.4996, mae: 1.7945, huber: 1.4642, swd: 4.2980, ept: 233.5688
    Epoch [8/50], Val Losses: mse: 9.6352, mae: 1.7511, huber: 1.4257, swd: 4.8595, ept: 532.0661
    Epoch [8/50], Test Losses: mse: 8.2935, mae: 1.6477, huber: 1.3171, swd: 4.0694, ept: 519.9584
      Epoch 8 composite train-obj: 1.464209
            No improvement (1.4257), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.5064, mae: 1.7963, huber: 1.4653, swd: 4.2847, ept: 234.8626
    Epoch [9/50], Val Losses: mse: 8.6476, mae: 1.6183, huber: 1.2842, swd: 4.5920, ept: 557.9604
    Epoch [9/50], Test Losses: mse: 7.2499, mae: 1.5006, huber: 1.1616, swd: 3.8249, ept: 559.7762
      Epoch 9 composite train-obj: 1.465317
            No improvement (1.2842), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.3243, mae: 1.7686, huber: 1.4399, swd: 4.2292, ept: 233.1844
    Epoch [10/50], Val Losses: mse: 9.2993, mae: 1.6958, huber: 1.3715, swd: 4.8509, ept: 540.6083
    Epoch [10/50], Test Losses: mse: 8.3420, mae: 1.6310, huber: 1.2996, swd: 4.3428, ept: 535.8552
      Epoch 10 composite train-obj: 1.439912
            No improvement (1.3715), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.5979, mae: 1.8084, huber: 1.4768, swd: 4.3155, ept: 233.1002
    Epoch [11/50], Val Losses: mse: 9.4702, mae: 1.8214, huber: 1.4779, swd: 4.7691, ept: 534.1815
    Epoch [11/50], Test Losses: mse: 8.3725, mae: 1.7237, huber: 1.3762, swd: 4.0619, ept: 529.3218
      Epoch 11 composite train-obj: 1.476838
            No improvement (1.4779), counter 4/5
    Epoch [12/50], Train Losses: mse: 10.2811, mae: 1.7655, huber: 1.4368, swd: 4.1979, ept: 234.2632
    Epoch [12/50], Val Losses: mse: 8.4297, mae: 1.5636, huber: 1.2351, swd: 4.5080, ept: 567.9949
    Epoch [12/50], Test Losses: mse: 6.9891, mae: 1.4453, huber: 1.1117, swd: 3.7350, ept: 569.1323
      Epoch 12 composite train-obj: 1.436808
            Val objective improved 1.2828 → 1.2351, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 10.1324, mae: 1.7322, huber: 1.4072, swd: 4.1371, ept: 234.3497
    Epoch [13/50], Val Losses: mse: 9.9714, mae: 1.8623, huber: 1.5226, swd: 4.8472, ept: 513.0326
    Epoch [13/50], Test Losses: mse: 8.6781, mae: 1.7521, huber: 1.4071, swd: 4.0272, ept: 509.7277
      Epoch 13 composite train-obj: 1.407223
            No improvement (1.5226), counter 1/5
    Epoch [14/50], Train Losses: mse: 10.3827, mae: 1.7769, huber: 1.4485, swd: 4.2062, ept: 233.8607
    Epoch [14/50], Val Losses: mse: 8.3822, mae: 1.6638, huber: 1.3313, swd: 4.5121, ept: 553.1132
    Epoch [14/50], Test Losses: mse: 6.9828, mae: 1.5138, huber: 1.1790, swd: 3.7266, ept: 560.3012
      Epoch 14 composite train-obj: 1.448493
            No improvement (1.3313), counter 2/5
    Epoch [15/50], Train Losses: mse: 10.1369, mae: 1.7374, huber: 1.4115, swd: 4.1371, ept: 234.2725
    Epoch [15/50], Val Losses: mse: 8.1991, mae: 1.6110, huber: 1.2829, swd: 4.5109, ept: 561.5389
    Epoch [15/50], Test Losses: mse: 6.7639, mae: 1.4768, huber: 1.1444, swd: 3.7116, ept: 567.0727
      Epoch 15 composite train-obj: 1.411465
            No improvement (1.2829), counter 3/5
    Epoch [16/50], Train Losses: mse: 10.0139, mae: 1.7225, huber: 1.3977, swd: 4.0625, ept: 234.0195
    Epoch [16/50], Val Losses: mse: 9.4261, mae: 1.6900, huber: 1.3558, swd: 5.2970, ept: 543.5022
    Epoch [16/50], Test Losses: mse: 7.9902, mae: 1.5936, huber: 1.2515, swd: 4.3682, ept: 545.6687
      Epoch 16 composite train-obj: 1.397692
            No improvement (1.3558), counter 4/5
    Epoch [17/50], Train Losses: mse: 10.3340, mae: 1.7663, huber: 1.4378, swd: 4.2115, ept: 233.0148
    Epoch [17/50], Val Losses: mse: 8.0115, mae: 1.5375, huber: 1.2194, swd: 4.3570, ept: 569.5893
    Epoch [17/50], Test Losses: mse: 6.4777, mae: 1.3836, huber: 1.0641, swd: 3.5199, ept: 576.4465
      Epoch 17 composite train-obj: 1.437797
            Val objective improved 1.2351 → 1.2194, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 10.0019, mae: 1.7156, huber: 1.3916, swd: 4.0599, ept: 234.1037
    Epoch [18/50], Val Losses: mse: 8.6150, mae: 1.6509, huber: 1.3264, swd: 4.5055, ept: 552.6336
    Epoch [18/50], Test Losses: mse: 7.2403, mae: 1.5214, huber: 1.1937, swd: 3.6685, ept: 545.4947
      Epoch 18 composite train-obj: 1.391551
            No improvement (1.3264), counter 1/5
    Epoch [19/50], Train Losses: mse: 9.9815, mae: 1.7126, huber: 1.3896, swd: 4.0347, ept: 234.9294
    Epoch [19/50], Val Losses: mse: 8.6541, mae: 1.6169, huber: 1.2918, swd: 4.4223, ept: 564.6302
    Epoch [19/50], Test Losses: mse: 7.3978, mae: 1.5179, huber: 1.1883, swd: 3.7211, ept: 562.3471
      Epoch 19 composite train-obj: 1.389600
            No improvement (1.2918), counter 2/5
    Epoch [20/50], Train Losses: mse: 9.8902, mae: 1.6964, huber: 1.3746, swd: 4.0018, ept: 235.1530
    Epoch [20/50], Val Losses: mse: 8.1276, mae: 1.5050, huber: 1.1911, swd: 4.3743, ept: 574.4744
    Epoch [20/50], Test Losses: mse: 6.8046, mae: 1.3881, huber: 1.0713, swd: 3.6576, ept: 571.7303
      Epoch 20 composite train-obj: 1.374612
            Val objective improved 1.2194 → 1.1911, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 9.8860, mae: 1.6924, huber: 1.3715, swd: 3.9942, ept: 235.0499
    Epoch [21/50], Val Losses: mse: 8.2185, mae: 1.5866, huber: 1.2622, swd: 4.3480, ept: 568.8025
    Epoch [21/50], Test Losses: mse: 6.7444, mae: 1.4661, huber: 1.1353, swd: 3.5541, ept: 574.9411
      Epoch 21 composite train-obj: 1.371458
            No improvement (1.2622), counter 1/5
    Epoch [22/50], Train Losses: mse: 9.8562, mae: 1.6928, huber: 1.3713, swd: 3.9712, ept: 235.0223
    Epoch [22/50], Val Losses: mse: 8.1842, mae: 1.5439, huber: 1.2170, swd: 4.3602, ept: 573.1754
    Epoch [22/50], Test Losses: mse: 6.6205, mae: 1.4100, huber: 1.0788, swd: 3.5723, ept: 573.9237
      Epoch 22 composite train-obj: 1.371305
            No improvement (1.2170), counter 2/5
    Epoch [23/50], Train Losses: mse: 9.7846, mae: 1.6764, huber: 1.3571, swd: 3.9506, ept: 235.4587
    Epoch [23/50], Val Losses: mse: 7.4661, mae: 1.4630, huber: 1.1508, swd: 4.1868, ept: 574.5931
    Epoch [23/50], Test Losses: mse: 5.9215, mae: 1.2957, huber: 0.9829, swd: 3.3736, ept: 586.9151
      Epoch 23 composite train-obj: 1.357095
            Val objective improved 1.1911 → 1.1508, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 9.7758, mae: 1.6742, huber: 1.3551, swd: 3.9418, ept: 236.0753
    Epoch [24/50], Val Losses: mse: 7.7431, mae: 1.5211, huber: 1.1960, swd: 4.2435, ept: 574.3994
    Epoch [24/50], Test Losses: mse: 6.2454, mae: 1.3698, huber: 1.0424, swd: 3.4645, ept: 578.4049
      Epoch 24 composite train-obj: 1.355127
            No improvement (1.1960), counter 1/5
    Epoch [25/50], Train Losses: mse: 9.8019, mae: 1.6734, huber: 1.3550, swd: 3.9538, ept: 234.9535
    Epoch [25/50], Val Losses: mse: 8.1426, mae: 1.6292, huber: 1.3054, swd: 4.2617, ept: 557.5498
    Epoch [25/50], Test Losses: mse: 6.7411, mae: 1.4888, huber: 1.1631, swd: 3.4531, ept: 562.6028
      Epoch 25 composite train-obj: 1.355007
            No improvement (1.3054), counter 2/5
    Epoch [26/50], Train Losses: mse: 9.7553, mae: 1.6789, huber: 1.3589, swd: 3.9199, ept: 234.7380
    Epoch [26/50], Val Losses: mse: 7.6027, mae: 1.5230, huber: 1.1997, swd: 4.1248, ept: 565.5854
    Epoch [26/50], Test Losses: mse: 6.1434, mae: 1.3681, huber: 1.0419, swd: 3.3663, ept: 573.8832
      Epoch 26 composite train-obj: 1.358881
            No improvement (1.1997), counter 3/5
    Epoch [27/50], Train Losses: mse: 9.7915, mae: 1.6807, huber: 1.3605, swd: 3.9401, ept: 235.2736
    Epoch [27/50], Val Losses: mse: 7.9951, mae: 1.5131, huber: 1.1934, swd: 4.2375, ept: 571.9048
    Epoch [27/50], Test Losses: mse: 6.7659, mae: 1.3883, huber: 1.0676, swd: 3.5857, ept: 575.0913
      Epoch 27 composite train-obj: 1.360544
            No improvement (1.1934), counter 4/5
    Epoch [28/50], Train Losses: mse: 9.7074, mae: 1.6638, huber: 1.3462, swd: 3.8927, ept: 235.1921
    Epoch [28/50], Val Losses: mse: 7.4127, mae: 1.4800, huber: 1.1563, swd: 4.2164, ept: 574.2900
    Epoch [28/50], Test Losses: mse: 6.0055, mae: 1.3355, huber: 1.0084, swd: 3.4497, ept: 580.8140
      Epoch 28 composite train-obj: 1.346218
    Epoch [28/50], Test Losses: mse: 5.9215, mae: 1.2957, huber: 0.9829, swd: 3.3736, ept: 586.9151
    Best round's Test MSE: 5.9215, MAE: 1.2957, SWD: 3.3736
    Best round's Validation MSE: 7.4661, MAE: 1.4630, SWD: 4.1868
    Best round's Test verification MSE : 5.9215, MAE: 1.2957, SWD: 3.3736
    Time taken: 101.05 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 21.0714, mae: 2.7672, huber: 2.3997, swd: 8.7054, ept: 221.4135
    Epoch [1/50], Val Losses: mse: 16.4419, mae: 2.5804, huber: 2.2197, swd: 6.1354, ept: 371.8445
    Epoch [1/50], Test Losses: mse: 16.0850, mae: 2.5609, huber: 2.1921, swd: 5.4776, ept: 353.0067
      Epoch 1 composite train-obj: 2.399674
            Val objective improved inf → 2.2197, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 14.2146, mae: 2.2163, huber: 1.8621, swd: 5.9925, ept: 226.5817
    Epoch [2/50], Val Losses: mse: 10.4851, mae: 1.8645, huber: 1.5228, swd: 5.5841, ept: 530.9424
    Epoch [2/50], Test Losses: mse: 10.1367, mae: 1.8115, huber: 1.4624, swd: 5.2656, ept: 521.8093
      Epoch 2 composite train-obj: 1.862065
            Val objective improved 2.2197 → 1.5228, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.8925, mae: 1.9931, huber: 1.6478, swd: 4.9650, ept: 228.5360
    Epoch [3/50], Val Losses: mse: 10.2134, mae: 1.9364, huber: 1.5868, swd: 5.5960, ept: 488.5116
    Epoch [3/50], Test Losses: mse: 9.4001, mae: 1.8521, huber: 1.4981, swd: 4.9351, ept: 485.6993
      Epoch 3 composite train-obj: 1.647829
            No improvement (1.5868), counter 1/5
    Epoch [4/50], Train Losses: mse: 11.4691, mae: 1.9295, huber: 1.5886, swd: 4.8274, ept: 230.5726
    Epoch [4/50], Val Losses: mse: 12.6112, mae: 2.0543, huber: 1.7021, swd: 6.3956, ept: 487.2810
    Epoch [4/50], Test Losses: mse: 12.5549, mae: 2.0267, huber: 1.6687, swd: 6.3272, ept: 478.4131
      Epoch 4 composite train-obj: 1.588616
            No improvement (1.7021), counter 2/5
    Epoch [5/50], Train Losses: mse: 11.3637, mae: 1.9070, huber: 1.5675, swd: 4.8475, ept: 231.9108
    Epoch [5/50], Val Losses: mse: 11.6898, mae: 2.0537, huber: 1.7050, swd: 5.2289, ept: 476.7640
    Epoch [5/50], Test Losses: mse: 10.8345, mae: 1.9771, huber: 1.6236, swd: 4.6309, ept: 464.6704
      Epoch 5 composite train-obj: 1.567549
            No improvement (1.7050), counter 3/5
    Epoch [6/50], Train Losses: mse: 10.8506, mae: 1.8551, huber: 1.5196, swd: 4.5281, ept: 232.5854
    Epoch [6/50], Val Losses: mse: 10.1772, mae: 1.8391, huber: 1.5021, swd: 5.2234, ept: 531.5311
    Epoch [6/50], Test Losses: mse: 9.6475, mae: 1.8024, huber: 1.4597, swd: 4.7463, ept: 518.7802
      Epoch 6 composite train-obj: 1.519551
            Val objective improved 1.5228 → 1.5021, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 10.5849, mae: 1.8166, huber: 1.4831, swd: 4.4248, ept: 233.5282
    Epoch [7/50], Val Losses: mse: 8.3973, mae: 1.6240, huber: 1.2903, swd: 4.7421, ept: 559.1718
    Epoch [7/50], Test Losses: mse: 7.0043, mae: 1.4988, huber: 1.1611, swd: 3.8756, ept: 557.7866
      Epoch 7 composite train-obj: 1.483134
            Val objective improved 1.5021 → 1.2903, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 10.4211, mae: 1.7918, huber: 1.4609, swd: 4.3506, ept: 233.0670
    Epoch [8/50], Val Losses: mse: 9.2712, mae: 1.8028, huber: 1.4553, swd: 4.9059, ept: 535.7350
    Epoch [8/50], Test Losses: mse: 8.2756, mae: 1.7241, huber: 1.3714, swd: 4.1853, ept: 535.6782
      Epoch 8 composite train-obj: 1.460911
            No improvement (1.4553), counter 1/5
    Epoch [9/50], Train Losses: mse: 10.4310, mae: 1.7875, huber: 1.4573, swd: 4.3611, ept: 233.1089
    Epoch [9/50], Val Losses: mse: 13.2327, mae: 2.1542, huber: 1.8181, swd: 5.8108, ept: 465.8224
    Epoch [9/50], Test Losses: mse: 12.1508, mae: 2.0585, huber: 1.7187, swd: 5.0448, ept: 463.1057
      Epoch 9 composite train-obj: 1.457267
            No improvement (1.8181), counter 2/5
    Epoch [10/50], Train Losses: mse: 10.6112, mae: 1.8191, huber: 1.4864, swd: 4.4518, ept: 233.0795
    Epoch [10/50], Val Losses: mse: 8.6925, mae: 1.6333, huber: 1.3021, swd: 4.8732, ept: 563.2849
    Epoch [10/50], Test Losses: mse: 7.6426, mae: 1.5347, huber: 1.2005, swd: 4.3039, ept: 559.5621
      Epoch 10 composite train-obj: 1.486432
            No improvement (1.3021), counter 3/5
    Epoch [11/50], Train Losses: mse: 10.2985, mae: 1.7653, huber: 1.4366, swd: 4.3112, ept: 233.8499
    Epoch [11/50], Val Losses: mse: 11.8657, mae: 2.1078, huber: 1.7560, swd: 5.7881, ept: 485.9050
    Epoch [11/50], Test Losses: mse: 10.1169, mae: 1.9499, huber: 1.5951, swd: 4.7872, ept: 486.1736
      Epoch 11 composite train-obj: 1.436555
            No improvement (1.7560), counter 4/5
    Epoch [12/50], Train Losses: mse: 10.7149, mae: 1.8321, huber: 1.4987, swd: 4.5111, ept: 232.3842
    Epoch [12/50], Val Losses: mse: 12.9359, mae: 2.1647, huber: 1.8093, swd: 5.6309, ept: 485.4351
    Epoch [12/50], Test Losses: mse: 12.1335, mae: 2.1153, huber: 1.7546, swd: 4.9445, ept: 464.9330
      Epoch 12 composite train-obj: 1.498720
    Epoch [12/50], Test Losses: mse: 7.0043, mae: 1.4988, huber: 1.1611, swd: 3.8756, ept: 557.7866
    Best round's Test MSE: 7.0043, MAE: 1.4988, SWD: 3.8756
    Best round's Validation MSE: 8.3973, MAE: 1.6240, SWD: 4.7421
    Best round's Test verification MSE : 7.0043, MAE: 1.4988, SWD: 3.8756
    Time taken: 43.83 seconds
    
    ==================================================
    Experiment Summary (PatchTST_rossler_seq96_pred720_20250513_1334)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 6.7297 ± 0.5812
      mae: 1.4188 ± 0.0884
      huber: 1.0945 ± 0.0794
      swd: 3.7606 ± 0.2811
      ept: 569.2163 ± 12.6910
      count: 35.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 8.0794 ± 0.4338
      mae: 1.5468 ± 0.0659
      huber: 1.2252 ± 0.0573
      swd: 4.5329 ± 0.2465
      ept: 565.9662 ± 6.4277
      count: 35.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 197.03 seconds
    
    Experiment complete: PatchTST_rossler_seq96_pred720_20250513_1334
    Model: PatchTST
    Dataset: rossler
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
    channels=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 4.2109, mae: 0.8340, huber: 0.6046, swd: 3.2423, ept: 87.7475
    Epoch [1/50], Val Losses: mse: 3.1116, mae: 0.5854, huber: 0.3807, swd: 2.6562, ept: 92.1310
    Epoch [1/50], Test Losses: mse: 2.9098, mae: 0.6118, huber: 0.3971, swd: 2.4740, ept: 92.0895
      Epoch 1 composite train-obj: 0.604597
            Val objective improved inf → 0.3807, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.6982, mae: 0.5450, huber: 0.3493, swd: 2.2761, ept: 92.5877
    Epoch [2/50], Val Losses: mse: 2.7676, mae: 0.5297, huber: 0.3389, swd: 2.3291, ept: 92.6368
    Epoch [2/50], Test Losses: mse: 2.5838, mae: 0.5521, huber: 0.3524, swd: 2.1628, ept: 92.5861
      Epoch 2 composite train-obj: 0.349298
            Val objective improved 0.3807 → 0.3389, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.4621, mae: 0.5023, huber: 0.3172, swd: 2.0539, ept: 92.9410
    Epoch [3/50], Val Losses: mse: 2.5981, mae: 0.4910, huber: 0.3130, swd: 2.1665, ept: 92.9575
    Epoch [3/50], Test Losses: mse: 2.4264, mae: 0.5122, huber: 0.3256, swd: 2.0103, ept: 92.8436
      Epoch 3 composite train-obj: 0.317223
            Val objective improved 0.3389 → 0.3130, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 2.3273, mae: 0.4719, huber: 0.2960, swd: 1.9329, ept: 93.1516
    Epoch [4/50], Val Losses: mse: 2.4750, mae: 0.4712, huber: 0.2955, swd: 2.0594, ept: 93.1438
    Epoch [4/50], Test Losses: mse: 2.3096, mae: 0.4912, huber: 0.3072, swd: 1.9075, ept: 93.0293
      Epoch 4 composite train-obj: 0.296028
            Val objective improved 0.3130 → 0.2955, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 2.2391, mae: 0.4514, huber: 0.2819, swd: 1.8540, ept: 93.2993
    Epoch [5/50], Val Losses: mse: 2.4047, mae: 0.4448, huber: 0.2837, swd: 1.9923, ept: 93.2705
    Epoch [5/50], Test Losses: mse: 2.2436, mae: 0.4644, huber: 0.2952, swd: 1.8445, ept: 93.1449
      Epoch 5 composite train-obj: 0.281875
            Val objective improved 0.2955 → 0.2837, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 2.1814, mae: 0.4363, huber: 0.2721, swd: 1.8025, ept: 93.3892
    Epoch [6/50], Val Losses: mse: 2.3400, mae: 0.4395, huber: 0.2746, swd: 1.9434, ept: 93.3936
    Epoch [6/50], Test Losses: mse: 2.1805, mae: 0.4582, huber: 0.2850, swd: 1.7969, ept: 93.2511
      Epoch 6 composite train-obj: 0.272071
            Val objective improved 0.2837 → 0.2746, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 2.1376, mae: 0.4256, huber: 0.2647, swd: 1.7636, ept: 93.4715
    Epoch [7/50], Val Losses: mse: 2.2999, mae: 0.4246, huber: 0.2685, swd: 1.8922, ept: 93.4265
    Epoch [7/50], Test Losses: mse: 2.1421, mae: 0.4419, huber: 0.2786, swd: 1.7462, ept: 93.3003
      Epoch 7 composite train-obj: 0.264691
            Val objective improved 0.2746 → 0.2685, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 2.1014, mae: 0.4153, huber: 0.2584, swd: 1.7302, ept: 93.5167
    Epoch [8/50], Val Losses: mse: 2.2825, mae: 0.4097, huber: 0.2621, swd: 1.8883, ept: 93.4482
    Epoch [8/50], Test Losses: mse: 2.1265, mae: 0.4268, huber: 0.2722, swd: 1.7443, ept: 93.3013
      Epoch 8 composite train-obj: 0.258384
            Val objective improved 0.2685 → 0.2621, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 2.0717, mae: 0.4090, huber: 0.2538, swd: 1.7030, ept: 93.5675
    Epoch [9/50], Val Losses: mse: 2.2496, mae: 0.4044, huber: 0.2581, swd: 1.8593, ept: 93.5240
    Epoch [9/50], Test Losses: mse: 2.0934, mae: 0.4220, huber: 0.2675, swd: 1.7152, ept: 93.3770
      Epoch 9 composite train-obj: 0.253792
            Val objective improved 0.2621 → 0.2581, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 2.0436, mae: 0.4006, huber: 0.2493, swd: 1.6772, ept: 93.6071
    Epoch [10/50], Val Losses: mse: 2.2064, mae: 0.4204, huber: 0.2566, swd: 1.8174, ept: 93.6020
    Epoch [10/50], Test Losses: mse: 2.0544, mae: 0.4351, huber: 0.2654, swd: 1.6760, ept: 93.4239
      Epoch 10 composite train-obj: 0.249276
            Val objective improved 0.2581 → 0.2566, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 2.0186, mae: 0.3959, huber: 0.2456, swd: 1.6535, ept: 93.6420
    Epoch [11/50], Val Losses: mse: 2.1858, mae: 0.4014, huber: 0.2507, swd: 1.8003, ept: 93.6116
    Epoch [11/50], Test Losses: mse: 2.0317, mae: 0.4165, huber: 0.2592, swd: 1.6581, ept: 93.4730
      Epoch 11 composite train-obj: 0.245634
            Val objective improved 0.2566 → 0.2507, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 1.9962, mae: 0.3915, huber: 0.2427, swd: 1.6329, ept: 93.6676
    Epoch [12/50], Val Losses: mse: 2.1672, mae: 0.3949, huber: 0.2480, swd: 1.7734, ept: 93.6212
    Epoch [12/50], Test Losses: mse: 2.0168, mae: 0.4096, huber: 0.2565, swd: 1.6349, ept: 93.4548
      Epoch 12 composite train-obj: 0.242674
            Val objective improved 0.2507 → 0.2480, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 1.9806, mae: 0.3870, huber: 0.2400, swd: 1.6185, ept: 93.6853
    Epoch [13/50], Val Losses: mse: 2.1528, mae: 0.3953, huber: 0.2473, swd: 1.7791, ept: 93.6729
    Epoch [13/50], Test Losses: mse: 2.0013, mae: 0.4093, huber: 0.2553, swd: 1.6393, ept: 93.4965
      Epoch 13 composite train-obj: 0.240001
            Val objective improved 0.2480 → 0.2473, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 1.9631, mae: 0.3834, huber: 0.2375, swd: 1.6021, ept: 93.7147
    Epoch [14/50], Val Losses: mse: 2.1167, mae: 0.3914, huber: 0.2431, swd: 1.7352, ept: 93.6924
    Epoch [14/50], Test Losses: mse: 1.9671, mae: 0.4040, huber: 0.2508, swd: 1.5978, ept: 93.5216
      Epoch 14 composite train-obj: 0.237451
            Val objective improved 0.2473 → 0.2431, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 1.9462, mae: 0.3801, huber: 0.2351, swd: 1.5864, ept: 93.7356
    Epoch [15/50], Val Losses: mse: 2.1280, mae: 0.3813, huber: 0.2423, swd: 1.7330, ept: 93.6681
    Epoch [15/50], Test Losses: mse: 1.9807, mae: 0.3958, huber: 0.2510, swd: 1.5973, ept: 93.4926
      Epoch 15 composite train-obj: 0.235103
            Val objective improved 0.2431 → 0.2423, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 1.9360, mae: 0.3764, huber: 0.2331, swd: 1.5775, ept: 93.7511
    Epoch [16/50], Val Losses: mse: 2.1036, mae: 0.3781, huber: 0.2394, swd: 1.7169, ept: 93.6928
    Epoch [16/50], Test Losses: mse: 1.9559, mae: 0.3923, huber: 0.2473, swd: 1.5812, ept: 93.5290
      Epoch 16 composite train-obj: 0.233077
            Val objective improved 0.2423 → 0.2394, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 1.9258, mae: 0.3755, huber: 0.2318, swd: 1.5674, ept: 93.7589
    Epoch [17/50], Val Losses: mse: 2.0896, mae: 0.3744, huber: 0.2368, swd: 1.7052, ept: 93.7406
    Epoch [17/50], Test Losses: mse: 1.9409, mae: 0.3880, huber: 0.2447, swd: 1.5685, ept: 93.5900
      Epoch 17 composite train-obj: 0.231790
            Val objective improved 0.2394 → 0.2368, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 1.9144, mae: 0.3714, huber: 0.2297, swd: 1.5561, ept: 93.7784
    Epoch [18/50], Val Losses: mse: 2.0811, mae: 0.3814, huber: 0.2360, swd: 1.6978, ept: 93.7540
    Epoch [18/50], Test Losses: mse: 1.9332, mae: 0.3951, huber: 0.2436, swd: 1.5616, ept: 93.5885
      Epoch 18 composite train-obj: 0.229665
            Val objective improved 0.2368 → 0.2360, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 1.9075, mae: 0.3707, huber: 0.2287, swd: 1.5495, ept: 93.7863
    Epoch [19/50], Val Losses: mse: 2.0821, mae: 0.3683, huber: 0.2340, swd: 1.7023, ept: 93.7425
    Epoch [19/50], Test Losses: mse: 1.9350, mae: 0.3825, huber: 0.2422, swd: 1.5670, ept: 93.5581
      Epoch 19 composite train-obj: 0.228653
            Val objective improved 0.2360 → 0.2340, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 1.8982, mae: 0.3676, huber: 0.2270, swd: 1.5415, ept: 93.8005
    Epoch [20/50], Val Losses: mse: 2.0742, mae: 0.3624, huber: 0.2320, swd: 1.6887, ept: 93.7533
    Epoch [20/50], Test Losses: mse: 1.9290, mae: 0.3763, huber: 0.2403, swd: 1.5549, ept: 93.5915
      Epoch 20 composite train-obj: 0.226971
            Val objective improved 0.2340 → 0.2320, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 1.8928, mae: 0.3656, huber: 0.2258, swd: 1.5365, ept: 93.8161
    Epoch [21/50], Val Losses: mse: 2.0677, mae: 0.3834, huber: 0.2359, swd: 1.6962, ept: 93.8175
    Epoch [21/50], Test Losses: mse: 1.9226, mae: 0.3970, huber: 0.2437, swd: 1.5611, ept: 93.6231
      Epoch 21 composite train-obj: 0.225792
            No improvement (0.2359), counter 1/5
    Epoch [22/50], Train Losses: mse: 1.8841, mae: 0.3639, huber: 0.2245, swd: 1.5277, ept: 93.8167
    Epoch [22/50], Val Losses: mse: 2.0454, mae: 0.3694, huber: 0.2297, swd: 1.6670, ept: 93.7946
    Epoch [22/50], Test Losses: mse: 1.9001, mae: 0.3813, huber: 0.2371, swd: 1.5338, ept: 93.6184
      Epoch 22 composite train-obj: 0.224531
            Val objective improved 0.2320 → 0.2297, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 1.8786, mae: 0.3632, huber: 0.2240, swd: 1.5217, ept: 93.8230
    Epoch [23/50], Val Losses: mse: 2.0462, mae: 0.3660, huber: 0.2297, swd: 1.6745, ept: 93.8104
    Epoch [23/50], Test Losses: mse: 1.9017, mae: 0.3789, huber: 0.2373, swd: 1.5413, ept: 93.6232
      Epoch 23 composite train-obj: 0.223955
            Val objective improved 0.2297 → 0.2297, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 1.8722, mae: 0.3602, huber: 0.2224, swd: 1.5166, ept: 93.8400
    Epoch [24/50], Val Losses: mse: 2.0362, mae: 0.3689, huber: 0.2293, swd: 1.6611, ept: 93.8184
    Epoch [24/50], Test Losses: mse: 1.8904, mae: 0.3825, huber: 0.2368, swd: 1.5273, ept: 93.6368
      Epoch 24 composite train-obj: 0.222442
            Val objective improved 0.2297 → 0.2293, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 1.8682, mae: 0.3594, huber: 0.2218, swd: 1.5121, ept: 93.8423
    Epoch [25/50], Val Losses: mse: 2.0257, mae: 0.3699, huber: 0.2276, swd: 1.6547, ept: 93.8595
    Epoch [25/50], Test Losses: mse: 1.8823, mae: 0.3823, huber: 0.2349, swd: 1.5223, ept: 93.6522
      Epoch 25 composite train-obj: 0.221773
            Val objective improved 0.2293 → 0.2276, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 1.8603, mae: 0.3579, huber: 0.2206, swd: 1.5054, ept: 93.8501
    Epoch [26/50], Val Losses: mse: 2.0299, mae: 0.3617, huber: 0.2272, swd: 1.6566, ept: 93.8342
    Epoch [26/50], Test Losses: mse: 1.8847, mae: 0.3742, huber: 0.2344, swd: 1.5229, ept: 93.6577
      Epoch 26 composite train-obj: 0.220608
            Val objective improved 0.2276 → 0.2272, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 1.8573, mae: 0.3567, huber: 0.2201, swd: 1.5009, ept: 93.8530
    Epoch [27/50], Val Losses: mse: 2.0232, mae: 0.3563, huber: 0.2250, swd: 1.6481, ept: 93.8402
    Epoch [27/50], Test Losses: mse: 1.8802, mae: 0.3687, huber: 0.2325, swd: 1.5166, ept: 93.6344
      Epoch 27 composite train-obj: 0.220120
            Val objective improved 0.2272 → 0.2250, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 1.8515, mae: 0.3544, huber: 0.2190, swd: 1.4968, ept: 93.8635
    Epoch [28/50], Val Losses: mse: 2.0186, mae: 0.3580, huber: 0.2246, swd: 1.6449, ept: 93.8529
    Epoch [28/50], Test Losses: mse: 1.8755, mae: 0.3706, huber: 0.2322, swd: 1.5132, ept: 93.6607
      Epoch 28 composite train-obj: 0.218959
            Val objective improved 0.2250 → 0.2246, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 1.8486, mae: 0.3545, huber: 0.2186, swd: 1.4941, ept: 93.8644
    Epoch [29/50], Val Losses: mse: 2.0084, mae: 0.3602, huber: 0.2242, swd: 1.6267, ept: 93.8428
    Epoch [29/50], Test Losses: mse: 1.8642, mae: 0.3728, huber: 0.2315, swd: 1.4951, ept: 93.6878
      Epoch 29 composite train-obj: 0.218565
            Val objective improved 0.2246 → 0.2242, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 1.8414, mae: 0.3528, huber: 0.2176, swd: 1.4874, ept: 93.8740
    Epoch [30/50], Val Losses: mse: 2.0074, mae: 0.3567, huber: 0.2231, swd: 1.6299, ept: 93.8514
    Epoch [30/50], Test Losses: mse: 1.8653, mae: 0.3691, huber: 0.2305, swd: 1.4993, ept: 93.6645
      Epoch 30 composite train-obj: 0.217603
            Val objective improved 0.2242 → 0.2231, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 1.8397, mae: 0.3521, huber: 0.2172, swd: 1.4838, ept: 93.8727
    Epoch [31/50], Val Losses: mse: 2.0070, mae: 0.3540, huber: 0.2237, swd: 1.6297, ept: 93.8622
    Epoch [31/50], Test Losses: mse: 1.8644, mae: 0.3687, huber: 0.2314, swd: 1.4986, ept: 93.6840
      Epoch 31 composite train-obj: 0.217231
            No improvement (0.2237), counter 1/5
    Epoch [32/50], Train Losses: mse: 1.8355, mae: 0.3518, huber: 0.2168, swd: 1.4810, ept: 93.8842
    Epoch [32/50], Val Losses: mse: 1.9924, mae: 0.3615, huber: 0.2224, swd: 1.6167, ept: 93.8834
    Epoch [32/50], Test Losses: mse: 1.8500, mae: 0.3731, huber: 0.2291, swd: 1.4861, ept: 93.6995
      Epoch 32 composite train-obj: 0.216836
            Val objective improved 0.2231 → 0.2224, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 1.8290, mae: 0.3507, huber: 0.2159, swd: 1.4756, ept: 93.8897
    Epoch [33/50], Val Losses: mse: 1.9936, mae: 0.3555, huber: 0.2223, swd: 1.6103, ept: 93.8640
    Epoch [33/50], Test Losses: mse: 1.8508, mae: 0.3695, huber: 0.2296, swd: 1.4792, ept: 93.7058
      Epoch 33 composite train-obj: 0.215947
            Val objective improved 0.2224 → 0.2223, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 1.8274, mae: 0.3489, huber: 0.2154, swd: 1.4727, ept: 93.8913
    Epoch [34/50], Val Losses: mse: 1.9866, mae: 0.3526, huber: 0.2207, swd: 1.6103, ept: 93.8850
    Epoch [34/50], Test Losses: mse: 1.8454, mae: 0.3641, huber: 0.2279, swd: 1.4811, ept: 93.6823
      Epoch 34 composite train-obj: 0.215370
            Val objective improved 0.2223 → 0.2207, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 1.8223, mae: 0.3491, huber: 0.2149, swd: 1.4692, ept: 93.9068
    Epoch [35/50], Val Losses: mse: 1.9871, mae: 0.3514, huber: 0.2211, swd: 1.6021, ept: 93.8624
    Epoch [35/50], Test Losses: mse: 1.8456, mae: 0.3631, huber: 0.2285, swd: 1.4730, ept: 93.7010
      Epoch 35 composite train-obj: 0.214912
            No improvement (0.2211), counter 1/5
    Epoch [36/50], Train Losses: mse: 1.8206, mae: 0.3477, huber: 0.2144, swd: 1.4670, ept: 93.8983
    Epoch [36/50], Val Losses: mse: 1.9789, mae: 0.3529, huber: 0.2205, swd: 1.5975, ept: 93.8746
    Epoch [36/50], Test Losses: mse: 1.8368, mae: 0.3653, huber: 0.2277, swd: 1.4678, ept: 93.7106
      Epoch 36 composite train-obj: 0.214355
            Val objective improved 0.2207 → 0.2205, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 1.8165, mae: 0.3475, huber: 0.2141, swd: 1.4628, ept: 93.9058
    Epoch [37/50], Val Losses: mse: 1.9723, mae: 0.3570, huber: 0.2208, swd: 1.5879, ept: 93.8885
    Epoch [37/50], Test Losses: mse: 1.8311, mae: 0.3686, huber: 0.2279, swd: 1.4592, ept: 93.7295
      Epoch 37 composite train-obj: 0.214103
            No improvement (0.2208), counter 1/5
    Epoch [38/50], Train Losses: mse: 1.8148, mae: 0.3477, huber: 0.2140, swd: 1.4612, ept: 93.9068
    Epoch [38/50], Val Losses: mse: 1.9701, mae: 0.3564, huber: 0.2197, swd: 1.5942, ept: 93.8965
    Epoch [38/50], Test Losses: mse: 1.8287, mae: 0.3678, huber: 0.2263, swd: 1.4648, ept: 93.7216
      Epoch 38 composite train-obj: 0.213953
            Val objective improved 0.2205 → 0.2197, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 1.8085, mae: 0.3456, huber: 0.2128, swd: 1.4563, ept: 93.9169
    Epoch [39/50], Val Losses: mse: 1.9707, mae: 0.3477, huber: 0.2186, swd: 1.5933, ept: 93.8985
    Epoch [39/50], Test Losses: mse: 1.8303, mae: 0.3599, huber: 0.2256, swd: 1.4650, ept: 93.7139
      Epoch 39 composite train-obj: 0.212842
            Val objective improved 0.2197 → 0.2186, saving checkpoint.
    Epoch [40/50], Train Losses: mse: 1.8061, mae: 0.3448, huber: 0.2125, swd: 1.4543, ept: 93.9238
    Epoch [40/50], Val Losses: mse: 1.9644, mae: 0.3477, huber: 0.2184, swd: 1.5849, ept: 93.8912
    Epoch [40/50], Test Losses: mse: 1.8238, mae: 0.3591, huber: 0.2255, swd: 1.4568, ept: 93.7235
      Epoch 40 composite train-obj: 0.212453
            Val objective improved 0.2186 → 0.2184, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 1.8028, mae: 0.3431, huber: 0.2119, swd: 1.4507, ept: 93.9246
    Epoch [41/50], Val Losses: mse: 1.9550, mae: 0.3529, huber: 0.2178, swd: 1.5807, ept: 93.9148
    Epoch [41/50], Test Losses: mse: 1.8138, mae: 0.3648, huber: 0.2246, swd: 1.4523, ept: 93.7414
      Epoch 41 composite train-obj: 0.211865
            Val objective improved 0.2184 → 0.2178, saving checkpoint.
    Epoch [42/50], Train Losses: mse: 1.7984, mae: 0.3428, huber: 0.2114, swd: 1.4468, ept: 93.9354
    Epoch [42/50], Val Losses: mse: 1.9621, mae: 0.3443, huber: 0.2179, swd: 1.5838, ept: 93.8952
    Epoch [42/50], Test Losses: mse: 1.8223, mae: 0.3567, huber: 0.2252, swd: 1.4564, ept: 93.7335
      Epoch 42 composite train-obj: 0.211417
            No improvement (0.2179), counter 1/5
    Epoch [43/50], Train Losses: mse: 1.7954, mae: 0.3427, huber: 0.2113, swd: 1.4442, ept: 93.9356
    Epoch [43/50], Val Losses: mse: 1.9586, mae: 0.3428, huber: 0.2168, swd: 1.5877, ept: 93.9140
    Epoch [43/50], Test Losses: mse: 1.8176, mae: 0.3547, huber: 0.2236, swd: 1.4593, ept: 93.7399
      Epoch 43 composite train-obj: 0.211295
            Val objective improved 0.2178 → 0.2168, saving checkpoint.
    Epoch [44/50], Train Losses: mse: 1.7944, mae: 0.3430, huber: 0.2112, swd: 1.4424, ept: 93.9363
    Epoch [44/50], Val Losses: mse: 1.9616, mae: 0.3542, huber: 0.2194, swd: 1.5925, ept: 93.9441
    Epoch [44/50], Test Losses: mse: 1.8218, mae: 0.3666, huber: 0.2263, swd: 1.4633, ept: 93.7587
      Epoch 44 composite train-obj: 0.211239
            No improvement (0.2194), counter 1/5
    Epoch [45/50], Train Losses: mse: 1.7912, mae: 0.3417, huber: 0.2105, swd: 1.4404, ept: 93.9414
    Epoch [45/50], Val Losses: mse: 1.9565, mae: 0.3484, huber: 0.2173, swd: 1.5860, ept: 93.9294
    Epoch [45/50], Test Losses: mse: 1.8179, mae: 0.3604, huber: 0.2242, swd: 1.4595, ept: 93.7107
      Epoch 45 composite train-obj: 0.210504
            No improvement (0.2173), counter 2/5
    Epoch [46/50], Train Losses: mse: 1.7889, mae: 0.3411, huber: 0.2101, swd: 1.4386, ept: 93.9432
    Epoch [46/50], Val Losses: mse: 1.9490, mae: 0.3435, huber: 0.2159, swd: 1.5743, ept: 93.9174
    Epoch [46/50], Test Losses: mse: 1.8094, mae: 0.3555, huber: 0.2230, swd: 1.4473, ept: 93.7436
      Epoch 46 composite train-obj: 0.210063
            Val objective improved 0.2168 → 0.2159, saving checkpoint.
    Epoch [47/50], Train Losses: mse: 1.7856, mae: 0.3405, huber: 0.2097, swd: 1.4355, ept: 93.9425
    Epoch [47/50], Val Losses: mse: 1.9454, mae: 0.3406, huber: 0.2155, swd: 1.5709, ept: 93.9229
    Epoch [47/50], Test Losses: mse: 1.8051, mae: 0.3537, huber: 0.2224, swd: 1.4426, ept: 93.7616
      Epoch 47 composite train-obj: 0.209719
            Val objective improved 0.2159 → 0.2155, saving checkpoint.
    Epoch [48/50], Train Losses: mse: 1.7831, mae: 0.3403, huber: 0.2097, swd: 1.4325, ept: 93.9474
    Epoch [48/50], Val Losses: mse: 1.9435, mae: 0.3439, huber: 0.2150, swd: 1.5692, ept: 93.9255
    Epoch [48/50], Test Losses: mse: 1.8055, mae: 0.3558, huber: 0.2220, swd: 1.4428, ept: 93.7331
      Epoch 48 composite train-obj: 0.209667
            Val objective improved 0.2155 → 0.2150, saving checkpoint.
    Epoch [49/50], Train Losses: mse: 1.7811, mae: 0.3389, huber: 0.2091, swd: 1.4307, ept: 93.9450
    Epoch [49/50], Val Losses: mse: 1.9438, mae: 0.3404, huber: 0.2146, swd: 1.5717, ept: 93.9395
    Epoch [49/50], Test Losses: mse: 1.8058, mae: 0.3525, huber: 0.2215, swd: 1.4454, ept: 93.7230
      Epoch 49 composite train-obj: 0.209086
            Val objective improved 0.2150 → 0.2146, saving checkpoint.
    Epoch [50/50], Train Losses: mse: 1.7802, mae: 0.3395, huber: 0.2091, swd: 1.4293, ept: 93.9487
    Epoch [50/50], Val Losses: mse: 1.9387, mae: 0.3396, huber: 0.2143, swd: 1.5655, ept: 93.9381
    Epoch [50/50], Test Losses: mse: 1.8007, mae: 0.3514, huber: 0.2211, swd: 1.4396, ept: 93.7481
      Epoch 50 composite train-obj: 0.209061
            Val objective improved 0.2146 → 0.2143, saving checkpoint.
    Epoch [50/50], Test Losses: mse: 1.8007, mae: 0.3514, huber: 0.2211, swd: 1.4396, ept: 93.7481
    Best round's Test MSE: 1.8007, MAE: 0.3514, SWD: 1.4396
    Best round's Validation MSE: 1.9387, MAE: 0.3396, SWD: 1.5655
    Best round's Test verification MSE : 1.8007, MAE: 0.3514, SWD: 1.4396
    Time taken: 121.65 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.2871, mae: 0.8455, huber: 0.6149, swd: 3.2725, ept: 87.8637
    Epoch [1/50], Val Losses: mse: 3.1422, mae: 0.5778, huber: 0.3799, swd: 2.7135, ept: 92.1788
    Epoch [1/50], Test Losses: mse: 2.9416, mae: 0.6052, huber: 0.3974, swd: 2.5303, ept: 92.0943
      Epoch 1 composite train-obj: 0.614918
            Val objective improved inf → 0.3799, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.7019, mae: 0.5460, huber: 0.3497, swd: 2.3035, ept: 92.5660
    Epoch [2/50], Val Losses: mse: 2.7960, mae: 0.5284, huber: 0.3399, swd: 2.3743, ept: 92.6012
    Epoch [2/50], Test Losses: mse: 2.6138, mae: 0.5528, huber: 0.3546, swd: 2.2058, ept: 92.5440
      Epoch 2 composite train-obj: 0.349722
            Val objective improved 0.3799 → 0.3399, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.4678, mae: 0.5020, huber: 0.3176, swd: 2.0896, ept: 92.9174
    Epoch [3/50], Val Losses: mse: 2.6052, mae: 0.4894, huber: 0.3139, swd: 2.1960, ept: 92.8954
    Epoch [3/50], Test Losses: mse: 2.4340, mae: 0.5107, huber: 0.3268, swd: 2.0369, ept: 92.7853
      Epoch 3 composite train-obj: 0.317579
            Val objective improved 0.3399 → 0.3139, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 2.3263, mae: 0.4723, huber: 0.2965, swd: 1.9653, ept: 93.1364
    Epoch [4/50], Val Losses: mse: 2.4628, mae: 0.4752, huber: 0.2968, swd: 2.0936, ept: 93.1452
    Epoch [4/50], Test Losses: mse: 2.2984, mae: 0.4936, huber: 0.3079, swd: 1.9400, ept: 93.0247
      Epoch 4 composite train-obj: 0.296466
            Val objective improved 0.3139 → 0.2968, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 2.2378, mae: 0.4521, huber: 0.2824, swd: 1.8901, ept: 93.2965
    Epoch [5/50], Val Losses: mse: 2.4026, mae: 0.4424, huber: 0.2830, swd: 2.0367, ept: 93.2344
    Epoch [5/50], Test Losses: mse: 2.2398, mae: 0.4614, huber: 0.2941, swd: 1.8845, ept: 93.1392
      Epoch 5 composite train-obj: 0.282387
            Val objective improved 0.2968 → 0.2830, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 2.1775, mae: 0.4359, huber: 0.2719, swd: 1.8394, ept: 93.3868
    Epoch [6/50], Val Losses: mse: 2.3464, mae: 0.4323, huber: 0.2746, swd: 1.9824, ept: 93.3794
    Epoch [6/50], Test Losses: mse: 2.1874, mae: 0.4502, huber: 0.2853, swd: 1.8324, ept: 93.2227
      Epoch 6 composite train-obj: 0.271881
            Val objective improved 0.2830 → 0.2746, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 2.1351, mae: 0.4252, huber: 0.2646, swd: 1.8033, ept: 93.4665
    Epoch [7/50], Val Losses: mse: 2.3056, mae: 0.4210, huber: 0.2678, swd: 1.9548, ept: 93.4365
    Epoch [7/50], Test Losses: mse: 2.1478, mae: 0.4387, huber: 0.2780, swd: 1.8060, ept: 93.2824
      Epoch 7 composite train-obj: 0.264586
            Val objective improved 0.2746 → 0.2678, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 2.1024, mae: 0.4164, huber: 0.2587, swd: 1.7751, ept: 93.5190
    Epoch [8/50], Val Losses: mse: 2.2703, mae: 0.4143, huber: 0.2626, swd: 1.9261, ept: 93.4936
    Epoch [8/50], Test Losses: mse: 2.1141, mae: 0.4306, huber: 0.2722, swd: 1.7778, ept: 93.3359
      Epoch 8 composite train-obj: 0.258694
            Val objective improved 0.2678 → 0.2626, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 2.0697, mae: 0.4082, huber: 0.2536, swd: 1.7456, ept: 93.5664
    Epoch [9/50], Val Losses: mse: 2.2439, mae: 0.4073, huber: 0.2577, swd: 1.9102, ept: 93.5326
    Epoch [9/50], Test Losses: mse: 2.0904, mae: 0.4232, huber: 0.2673, swd: 1.7644, ept: 93.3554
      Epoch 9 composite train-obj: 0.253599
            Val objective improved 0.2626 → 0.2577, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 2.0438, mae: 0.4020, huber: 0.2496, swd: 1.7216, ept: 93.6060
    Epoch [10/50], Val Losses: mse: 2.2092, mae: 0.4072, huber: 0.2543, swd: 1.8685, ept: 93.5876
    Epoch [10/50], Test Losses: mse: 2.0553, mae: 0.4232, huber: 0.2634, swd: 1.7217, ept: 93.4361
      Epoch 10 composite train-obj: 0.249579
            Val objective improved 0.2577 → 0.2543, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 2.0174, mae: 0.3958, huber: 0.2456, swd: 1.6983, ept: 93.6376
    Epoch [11/50], Val Losses: mse: 2.1810, mae: 0.3977, huber: 0.2503, swd: 1.8481, ept: 93.6053
    Epoch [11/50], Test Losses: mse: 2.0275, mae: 0.4129, huber: 0.2589, swd: 1.7028, ept: 93.4620
      Epoch 11 composite train-obj: 0.245619
            Val objective improved 0.2543 → 0.2503, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 1.9960, mae: 0.3910, huber: 0.2425, swd: 1.6783, ept: 93.6747
    Epoch [12/50], Val Losses: mse: 2.1635, mae: 0.3982, huber: 0.2484, swd: 1.8360, ept: 93.6424
    Epoch [12/50], Test Losses: mse: 2.0124, mae: 0.4130, huber: 0.2567, swd: 1.6920, ept: 93.4504
      Epoch 12 composite train-obj: 0.242488
            Val objective improved 0.2503 → 0.2484, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 1.9770, mae: 0.3868, huber: 0.2397, swd: 1.6609, ept: 93.6891
    Epoch [13/50], Val Losses: mse: 2.1403, mae: 0.3868, huber: 0.2446, swd: 1.8037, ept: 93.6674
    Epoch [13/50], Test Losses: mse: 1.9897, mae: 0.4007, huber: 0.2528, swd: 1.6614, ept: 93.4966
      Epoch 13 composite train-obj: 0.239716
            Val objective improved 0.2484 → 0.2446, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 1.9625, mae: 0.3828, huber: 0.2372, swd: 1.6481, ept: 93.7119
    Epoch [14/50], Val Losses: mse: 2.1265, mae: 0.3971, huber: 0.2446, swd: 1.8006, ept: 93.7257
    Epoch [14/50], Test Losses: mse: 1.9756, mae: 0.4114, huber: 0.2524, swd: 1.6576, ept: 93.5570
      Epoch 14 composite train-obj: 0.237219
            No improvement (0.2446), counter 1/5
    Epoch [15/50], Train Losses: mse: 1.9455, mae: 0.3795, huber: 0.2349, swd: 1.6322, ept: 93.7306
    Epoch [15/50], Val Losses: mse: 2.1238, mae: 0.3796, huber: 0.2414, swd: 1.7916, ept: 93.6839
    Epoch [15/50], Test Losses: mse: 1.9749, mae: 0.3949, huber: 0.2498, swd: 1.6506, ept: 93.5189
      Epoch 15 composite train-obj: 0.234920
            Val objective improved 0.2446 → 0.2414, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 1.9339, mae: 0.3771, huber: 0.2332, swd: 1.6221, ept: 93.7564
    Epoch [16/50], Val Losses: mse: 2.1013, mae: 0.3842, huber: 0.2398, swd: 1.7800, ept: 93.7273
    Epoch [16/50], Test Losses: mse: 1.9533, mae: 0.3968, huber: 0.2473, swd: 1.6392, ept: 93.5537
      Epoch 16 composite train-obj: 0.233155
            Val objective improved 0.2414 → 0.2398, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 1.9235, mae: 0.3741, huber: 0.2312, swd: 1.6124, ept: 93.7657
    Epoch [17/50], Val Losses: mse: 2.1011, mae: 0.3787, huber: 0.2395, swd: 1.7528, ept: 93.6892
    Epoch [17/50], Test Losses: mse: 1.9539, mae: 0.3920, huber: 0.2478, swd: 1.6132, ept: 93.5432
      Epoch 17 composite train-obj: 0.231203
            Val objective improved 0.2398 → 0.2395, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 1.9163, mae: 0.3715, huber: 0.2298, swd: 1.6057, ept: 93.7702
    Epoch [18/50], Val Losses: mse: 2.0764, mae: 0.3751, huber: 0.2349, swd: 1.7457, ept: 93.7488
    Epoch [18/50], Test Losses: mse: 1.9285, mae: 0.3879, huber: 0.2426, swd: 1.6057, ept: 93.5958
      Epoch 18 composite train-obj: 0.229772
            Val objective improved 0.2395 → 0.2349, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 1.9040, mae: 0.3707, huber: 0.2285, swd: 1.5938, ept: 93.7890
    Epoch [19/50], Val Losses: mse: 2.0675, mae: 0.3725, huber: 0.2333, swd: 1.7410, ept: 93.7725
    Epoch [19/50], Test Losses: mse: 1.9215, mae: 0.3857, huber: 0.2411, swd: 1.6016, ept: 93.6119
      Epoch 19 composite train-obj: 0.228548
            Val objective improved 0.2349 → 0.2333, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 1.8970, mae: 0.3680, huber: 0.2270, swd: 1.5881, ept: 93.8036
    Epoch [20/50], Val Losses: mse: 2.0655, mae: 0.3678, huber: 0.2332, swd: 1.7326, ept: 93.7636
    Epoch [20/50], Test Losses: mse: 1.9208, mae: 0.3808, huber: 0.2412, swd: 1.5953, ept: 93.5651
      Epoch 20 composite train-obj: 0.226994
            Val objective improved 0.2333 → 0.2332, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 1.8913, mae: 0.3655, huber: 0.2257, swd: 1.5842, ept: 93.8192
    Epoch [21/50], Val Losses: mse: 2.0669, mae: 0.3778, huber: 0.2344, swd: 1.7504, ept: 93.7977
    Epoch [21/50], Test Losses: mse: 1.9212, mae: 0.3916, huber: 0.2420, swd: 1.6108, ept: 93.6118
      Epoch 21 composite train-obj: 0.225678
            No improvement (0.2344), counter 1/5
    Epoch [22/50], Train Losses: mse: 1.8842, mae: 0.3645, huber: 0.2249, swd: 1.5764, ept: 93.8140
    Epoch [22/50], Val Losses: mse: 2.0489, mae: 0.3649, huber: 0.2302, swd: 1.7172, ept: 93.7849
    Epoch [22/50], Test Losses: mse: 1.9031, mae: 0.3775, huber: 0.2378, swd: 1.5788, ept: 93.6187
      Epoch 22 composite train-obj: 0.224868
            Val objective improved 0.2332 → 0.2302, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 1.8769, mae: 0.3629, huber: 0.2237, swd: 1.5700, ept: 93.8252
    Epoch [23/50], Val Losses: mse: 2.0488, mae: 0.3608, huber: 0.2292, swd: 1.7271, ept: 93.8030
    Epoch [23/50], Test Losses: mse: 1.9022, mae: 0.3740, huber: 0.2368, swd: 1.5881, ept: 93.6292
      Epoch 23 composite train-obj: 0.223700
            Val objective improved 0.2302 → 0.2292, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 1.8733, mae: 0.3607, huber: 0.2227, swd: 1.5662, ept: 93.8373
    Epoch [24/50], Val Losses: mse: 2.0373, mae: 0.3674, huber: 0.2298, swd: 1.7177, ept: 93.8216
    Epoch [24/50], Test Losses: mse: 1.8921, mae: 0.3801, huber: 0.2370, swd: 1.5802, ept: 93.6331
      Epoch 24 composite train-obj: 0.222722
            No improvement (0.2298), counter 1/5
    Epoch [25/50], Train Losses: mse: 1.8683, mae: 0.3598, huber: 0.2219, swd: 1.5620, ept: 93.8404
    Epoch [25/50], Val Losses: mse: 2.0384, mae: 0.3621, huber: 0.2279, swd: 1.7122, ept: 93.8161
    Epoch [25/50], Test Losses: mse: 1.8939, mae: 0.3756, huber: 0.2357, swd: 1.5740, ept: 93.6427
      Epoch 25 composite train-obj: 0.221892
            Val objective improved 0.2292 → 0.2279, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 1.8630, mae: 0.3586, huber: 0.2211, swd: 1.5562, ept: 93.8473
    Epoch [26/50], Val Losses: mse: 2.0197, mae: 0.3618, huber: 0.2266, swd: 1.6940, ept: 93.8260
    Epoch [26/50], Test Losses: mse: 1.8756, mae: 0.3739, huber: 0.2339, swd: 1.5575, ept: 93.6517
      Epoch 26 composite train-obj: 0.221140
            Val objective improved 0.2279 → 0.2266, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 1.8554, mae: 0.3563, huber: 0.2200, swd: 1.5491, ept: 93.8553
    Epoch [27/50], Val Losses: mse: 2.0266, mae: 0.3616, huber: 0.2262, swd: 1.7129, ept: 93.8485
    Epoch [27/50], Test Losses: mse: 1.8834, mae: 0.3742, huber: 0.2337, swd: 1.5761, ept: 93.6538
      Epoch 27 composite train-obj: 0.219964
            Val objective improved 0.2266 → 0.2262, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 1.8533, mae: 0.3540, huber: 0.2188, swd: 1.5488, ept: 93.8639
    Epoch [28/50], Val Losses: mse: 2.0024, mae: 0.3658, huber: 0.2261, swd: 1.6793, ept: 93.8508
    Epoch [28/50], Test Losses: mse: 1.8588, mae: 0.3776, huber: 0.2328, swd: 1.5443, ept: 93.6576
      Epoch 28 composite train-obj: 0.218801
            Val objective improved 0.2262 → 0.2261, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 1.8460, mae: 0.3546, huber: 0.2186, swd: 1.5420, ept: 93.8681
    Epoch [29/50], Val Losses: mse: 2.0154, mae: 0.3629, huber: 0.2254, swd: 1.6987, ept: 93.8581
    Epoch [29/50], Test Losses: mse: 1.8716, mae: 0.3760, huber: 0.2327, swd: 1.5612, ept: 93.6865
      Epoch 29 composite train-obj: 0.218556
            Val objective improved 0.2261 → 0.2254, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 1.8423, mae: 0.3534, huber: 0.2179, swd: 1.5374, ept: 93.8757
    Epoch [30/50], Val Losses: mse: 2.0015, mae: 0.3629, huber: 0.2238, swd: 1.6877, ept: 93.8781
    Epoch [30/50], Test Losses: mse: 1.8575, mae: 0.3750, huber: 0.2307, swd: 1.5506, ept: 93.7026
      Epoch 30 composite train-obj: 0.217862
            Val objective improved 0.2254 → 0.2238, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 1.8387, mae: 0.3520, huber: 0.2171, swd: 1.5351, ept: 93.8758
    Epoch [31/50], Val Losses: mse: 2.0000, mae: 0.3620, huber: 0.2237, swd: 1.6852, ept: 93.8818
    Epoch [31/50], Test Losses: mse: 1.8571, mae: 0.3745, huber: 0.2308, swd: 1.5486, ept: 93.7094
      Epoch 31 composite train-obj: 0.217115
            Val objective improved 0.2238 → 0.2237, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 1.8347, mae: 0.3502, huber: 0.2162, swd: 1.5314, ept: 93.8901
    Epoch [32/50], Val Losses: mse: 1.9958, mae: 0.3581, huber: 0.2221, swd: 1.6734, ept: 93.8794
    Epoch [32/50], Test Losses: mse: 1.8543, mae: 0.3702, huber: 0.2293, swd: 1.5386, ept: 93.6705
      Epoch 32 composite train-obj: 0.216222
            Val objective improved 0.2237 → 0.2221, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 1.8292, mae: 0.3502, huber: 0.2158, swd: 1.5270, ept: 93.8889
    Epoch [33/50], Val Losses: mse: 1.9933, mae: 0.3563, huber: 0.2223, swd: 1.6670, ept: 93.8608
    Epoch [33/50], Test Losses: mse: 1.8506, mae: 0.3684, huber: 0.2295, swd: 1.5316, ept: 93.6880
      Epoch 33 composite train-obj: 0.215804
            No improvement (0.2223), counter 1/5
    Epoch [34/50], Train Losses: mse: 1.8256, mae: 0.3495, huber: 0.2153, swd: 1.5232, ept: 93.8947
    Epoch [34/50], Val Losses: mse: 1.9926, mae: 0.3511, huber: 0.2207, swd: 1.6763, ept: 93.8876
    Epoch [34/50], Test Losses: mse: 1.8505, mae: 0.3634, huber: 0.2280, swd: 1.5406, ept: 93.7080
      Epoch 34 composite train-obj: 0.215310
            Val objective improved 0.2221 → 0.2207, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 1.8235, mae: 0.3489, huber: 0.2149, swd: 1.5205, ept: 93.9080
    Epoch [35/50], Val Losses: mse: 1.9851, mae: 0.3535, huber: 0.2210, swd: 1.6714, ept: 93.8939
    Epoch [35/50], Test Losses: mse: 1.8443, mae: 0.3658, huber: 0.2282, swd: 1.5377, ept: 93.6776
      Epoch 35 composite train-obj: 0.214933
            No improvement (0.2210), counter 1/5
    Epoch [36/50], Train Losses: mse: 1.8207, mae: 0.3475, huber: 0.2143, swd: 1.5189, ept: 93.9017
    Epoch [36/50], Val Losses: mse: 1.9787, mae: 0.3509, huber: 0.2199, swd: 1.6574, ept: 93.8923
    Epoch [36/50], Test Losses: mse: 1.8364, mae: 0.3638, huber: 0.2271, swd: 1.5224, ept: 93.7200
      Epoch 36 composite train-obj: 0.214273
            Val objective improved 0.2207 → 0.2199, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 1.8149, mae: 0.3459, huber: 0.2135, swd: 1.5136, ept: 93.9093
    Epoch [37/50], Val Losses: mse: 1.9764, mae: 0.3525, huber: 0.2194, swd: 1.6613, ept: 93.9029
    Epoch [37/50], Test Losses: mse: 1.8345, mae: 0.3637, huber: 0.2263, swd: 1.5261, ept: 93.7218
      Epoch 37 composite train-obj: 0.213508
            Val objective improved 0.2199 → 0.2194, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 1.8139, mae: 0.3458, huber: 0.2134, swd: 1.5121, ept: 93.9133
    Epoch [38/50], Val Losses: mse: 1.9750, mae: 0.3571, huber: 0.2214, swd: 1.6414, ept: 93.8770
    Epoch [38/50], Test Losses: mse: 1.8336, mae: 0.3690, huber: 0.2286, swd: 1.5079, ept: 93.7121
      Epoch 38 composite train-obj: 0.213422
            No improvement (0.2214), counter 1/5
    Epoch [39/50], Train Losses: mse: 1.8085, mae: 0.3451, huber: 0.2128, swd: 1.5064, ept: 93.9169
    Epoch [39/50], Val Losses: mse: 1.9699, mae: 0.3496, huber: 0.2186, swd: 1.6582, ept: 93.9188
    Epoch [39/50], Test Losses: mse: 1.8295, mae: 0.3617, huber: 0.2256, swd: 1.5244, ept: 93.7177
      Epoch 39 composite train-obj: 0.212822
            Val objective improved 0.2194 → 0.2186, saving checkpoint.
    Epoch [40/50], Train Losses: mse: 1.8048, mae: 0.3438, huber: 0.2121, swd: 1.5046, ept: 93.9212
    Epoch [40/50], Val Losses: mse: 1.9616, mae: 0.3496, huber: 0.2178, swd: 1.6376, ept: 93.9067
    Epoch [40/50], Test Losses: mse: 1.8198, mae: 0.3616, huber: 0.2247, swd: 1.5031, ept: 93.7372
      Epoch 40 composite train-obj: 0.212130
            Val objective improved 0.2186 → 0.2178, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 1.8023, mae: 0.3444, huber: 0.2120, swd: 1.5014, ept: 93.9348
    Epoch [41/50], Val Losses: mse: 1.9591, mae: 0.3472, huber: 0.2176, swd: 1.6352, ept: 93.8988
    Epoch [41/50], Test Losses: mse: 1.8191, mae: 0.3587, huber: 0.2246, swd: 1.5026, ept: 93.7231
      Epoch 41 composite train-obj: 0.212043
            Val objective improved 0.2178 → 0.2176, saving checkpoint.
    Epoch [42/50], Train Losses: mse: 1.8008, mae: 0.3439, huber: 0.2119, swd: 1.5000, ept: 93.9336
    Epoch [42/50], Val Losses: mse: 1.9562, mae: 0.3620, huber: 0.2190, swd: 1.6451, ept: 93.9397
    Epoch [42/50], Test Losses: mse: 1.8154, mae: 0.3735, huber: 0.2256, swd: 1.5109, ept: 93.7526
      Epoch 42 composite train-obj: 0.211923
            No improvement (0.2190), counter 1/5
    Epoch [43/50], Train Losses: mse: 1.7952, mae: 0.3431, huber: 0.2112, swd: 1.4957, ept: 93.9369
    Epoch [43/50], Val Losses: mse: 1.9555, mae: 0.3425, huber: 0.2166, swd: 1.6368, ept: 93.9055
    Epoch [43/50], Test Losses: mse: 1.8157, mae: 0.3544, huber: 0.2234, swd: 1.5042, ept: 93.7274
      Epoch 43 composite train-obj: 0.211235
            Val objective improved 0.2176 → 0.2166, saving checkpoint.
    Epoch [44/50], Train Losses: mse: 1.7943, mae: 0.3427, huber: 0.2112, swd: 1.4938, ept: 93.9304
    Epoch [44/50], Val Losses: mse: 1.9620, mae: 0.3493, huber: 0.2187, swd: 1.6461, ept: 93.9292
    Epoch [44/50], Test Losses: mse: 1.8216, mae: 0.3625, huber: 0.2258, swd: 1.5125, ept: 93.7429
      Epoch 44 composite train-obj: 0.211151
            No improvement (0.2187), counter 1/5
    Epoch [45/50], Train Losses: mse: 1.7914, mae: 0.3419, huber: 0.2106, swd: 1.4914, ept: 93.9377
    Epoch [45/50], Val Losses: mse: 1.9530, mae: 0.3432, huber: 0.2168, swd: 1.6333, ept: 93.9204
    Epoch [45/50], Test Losses: mse: 1.8130, mae: 0.3557, huber: 0.2238, swd: 1.5005, ept: 93.7452
      Epoch 45 composite train-obj: 0.210636
            No improvement (0.2168), counter 2/5
    Epoch [46/50], Train Losses: mse: 1.7890, mae: 0.3398, huber: 0.2098, swd: 1.4899, ept: 93.9492
    Epoch [46/50], Val Losses: mse: 1.9395, mae: 0.3501, huber: 0.2155, swd: 1.6245, ept: 93.9451
    Epoch [46/50], Test Losses: mse: 1.8007, mae: 0.3605, huber: 0.2220, swd: 1.4932, ept: 93.7391
      Epoch 46 composite train-obj: 0.209824
            Val objective improved 0.2166 → 0.2155, saving checkpoint.
    Epoch [47/50], Train Losses: mse: 1.7847, mae: 0.3411, huber: 0.2098, swd: 1.4856, ept: 93.9491
    Epoch [47/50], Val Losses: mse: 1.9499, mae: 0.3395, huber: 0.2162, swd: 1.6278, ept: 93.9147
    Epoch [47/50], Test Losses: mse: 1.8104, mae: 0.3516, huber: 0.2230, swd: 1.4965, ept: 93.7197
      Epoch 47 composite train-obj: 0.209822
            No improvement (0.2162), counter 1/5
    Epoch [48/50], Train Losses: mse: 1.7841, mae: 0.3409, huber: 0.2098, swd: 1.4845, ept: 93.9482
    Epoch [48/50], Val Losses: mse: 1.9379, mae: 0.3421, huber: 0.2151, swd: 1.6254, ept: 93.9377
    Epoch [48/50], Test Losses: mse: 1.7983, mae: 0.3531, huber: 0.2218, swd: 1.4930, ept: 93.7693
      Epoch 48 composite train-obj: 0.209838
            Val objective improved 0.2155 → 0.2151, saving checkpoint.
    Epoch [49/50], Train Losses: mse: 1.7801, mae: 0.3391, huber: 0.2091, swd: 1.4814, ept: 93.9492
    Epoch [49/50], Val Losses: mse: 1.9382, mae: 0.3515, huber: 0.2164, swd: 1.6155, ept: 93.9368
    Epoch [49/50], Test Losses: mse: 1.7998, mae: 0.3649, huber: 0.2234, swd: 1.4840, ept: 93.7678
      Epoch 49 composite train-obj: 0.209065
            No improvement (0.2164), counter 1/5
    Epoch [50/50], Train Losses: mse: 1.7789, mae: 0.3386, huber: 0.2089, swd: 1.4801, ept: 93.9527
    Epoch [50/50], Val Losses: mse: 1.9282, mae: 0.3441, huber: 0.2144, swd: 1.6124, ept: 93.9462
    Epoch [50/50], Test Losses: mse: 1.7887, mae: 0.3558, huber: 0.2209, swd: 1.4806, ept: 93.7808
      Epoch 50 composite train-obj: 0.208859
            Val objective improved 0.2151 → 0.2144, saving checkpoint.
    Epoch [50/50], Test Losses: mse: 1.7887, mae: 0.3558, huber: 0.2209, swd: 1.4806, ept: 93.7808
    Best round's Test MSE: 1.7887, MAE: 0.3558, SWD: 1.4806
    Best round's Validation MSE: 1.9282, MAE: 0.3441, SWD: 1.6124
    Best round's Test verification MSE : 1.7887, MAE: 0.3558, SWD: 1.4806
    Time taken: 120.62 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.3016, mae: 0.8448, huber: 0.6154, swd: 2.9618, ept: 87.2631
    Epoch [1/50], Val Losses: mse: 3.1397, mae: 0.5821, huber: 0.3817, swd: 2.4725, ept: 92.1809
    Epoch [1/50], Test Losses: mse: 2.9389, mae: 0.6090, huber: 0.3990, swd: 2.3063, ept: 92.0600
      Epoch 1 composite train-obj: 0.615354
            Val objective improved inf → 0.3817, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.7070, mae: 0.5471, huber: 0.3504, swd: 2.1027, ept: 92.5295
    Epoch [2/50], Val Losses: mse: 2.7907, mae: 0.5256, huber: 0.3395, swd: 2.1595, ept: 92.6079
    Epoch [2/50], Test Losses: mse: 2.6074, mae: 0.5484, huber: 0.3535, swd: 2.0068, ept: 92.5415
      Epoch 2 composite train-obj: 0.350420
            Val objective improved 0.3817 → 0.3395, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.4686, mae: 0.5025, huber: 0.3174, swd: 1.9017, ept: 92.9224
    Epoch [3/50], Val Losses: mse: 2.5921, mae: 0.4971, huber: 0.3144, swd: 2.0129, ept: 92.9214
    Epoch [3/50], Test Losses: mse: 2.4185, mae: 0.5170, huber: 0.3264, swd: 1.8686, ept: 92.8457
      Epoch 3 composite train-obj: 0.317422
            Val objective improved 0.3395 → 0.3144, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 2.3280, mae: 0.4718, huber: 0.2961, swd: 1.7856, ept: 93.1421
    Epoch [4/50], Val Losses: mse: 2.4783, mae: 0.4670, huber: 0.2956, swd: 1.9010, ept: 93.1087
    Epoch [4/50], Test Losses: mse: 2.3132, mae: 0.4867, huber: 0.3072, swd: 1.7619, ept: 93.0001
      Epoch 4 composite train-obj: 0.296078
            Val objective improved 0.3144 → 0.2956, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 2.2383, mae: 0.4518, huber: 0.2821, swd: 1.7146, ept: 93.2814
    Epoch [5/50], Val Losses: mse: 2.4067, mae: 0.4523, huber: 0.2840, swd: 1.8588, ept: 93.2760
    Epoch [5/50], Test Losses: mse: 2.2438, mae: 0.4715, huber: 0.2950, swd: 1.7208, ept: 93.1439
      Epoch 5 composite train-obj: 0.282126
            Val objective improved 0.2956 → 0.2840, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 2.1846, mae: 0.4367, huber: 0.2722, swd: 1.6725, ept: 93.3857
    Epoch [6/50], Val Losses: mse: 2.3370, mae: 0.4327, huber: 0.2746, swd: 1.7946, ept: 93.3475
    Epoch [6/50], Test Losses: mse: 2.1755, mae: 0.4501, huber: 0.2848, swd: 1.6586, ept: 93.2385
      Epoch 6 composite train-obj: 0.272240
            Val objective improved 0.2840 → 0.2746, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 2.1390, mae: 0.4253, huber: 0.2648, swd: 1.6364, ept: 93.4627
    Epoch [7/50], Val Losses: mse: 2.3122, mae: 0.4246, huber: 0.2687, swd: 1.7822, ept: 93.4378
    Epoch [7/50], Test Losses: mse: 2.1547, mae: 0.4420, huber: 0.2790, swd: 1.6480, ept: 93.2804
      Epoch 7 composite train-obj: 0.264766
            Val objective improved 0.2746 → 0.2687, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 2.0996, mae: 0.4157, huber: 0.2583, swd: 1.6047, ept: 93.5245
    Epoch [8/50], Val Losses: mse: 2.2738, mae: 0.4148, huber: 0.2627, swd: 1.7441, ept: 93.5074
    Epoch [8/50], Test Losses: mse: 2.1182, mae: 0.4318, huber: 0.2726, swd: 1.6112, ept: 93.3544
      Epoch 8 composite train-obj: 0.258299
            Val objective improved 0.2687 → 0.2627, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 2.0707, mae: 0.4073, huber: 0.2534, swd: 1.5800, ept: 93.5670
    Epoch [9/50], Val Losses: mse: 2.2416, mae: 0.4100, huber: 0.2577, swd: 1.7227, ept: 93.5425
    Epoch [9/50], Test Losses: mse: 2.0877, mae: 0.4256, huber: 0.2672, swd: 1.5917, ept: 93.3599
      Epoch 9 composite train-obj: 0.253437
            Val objective improved 0.2627 → 0.2577, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 2.0429, mae: 0.4016, huber: 0.2493, swd: 1.5585, ept: 93.6070
    Epoch [10/50], Val Losses: mse: 2.2273, mae: 0.3928, huber: 0.2536, swd: 1.7072, ept: 93.5699
    Epoch [10/50], Test Losses: mse: 2.0737, mae: 0.4103, huber: 0.2632, swd: 1.5761, ept: 93.4067
      Epoch 10 composite train-obj: 0.249329
            Val objective improved 0.2577 → 0.2536, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 2.0201, mae: 0.3964, huber: 0.2460, swd: 1.5384, ept: 93.6332
    Epoch [11/50], Val Losses: mse: 2.1852, mae: 0.4044, huber: 0.2517, swd: 1.6820, ept: 93.6422
    Epoch [11/50], Test Losses: mse: 2.0336, mae: 0.4194, huber: 0.2603, swd: 1.5522, ept: 93.4526
      Epoch 11 composite train-obj: 0.246013
            Val objective improved 0.2536 → 0.2517, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 1.9977, mae: 0.3920, huber: 0.2429, swd: 1.5201, ept: 93.6683
    Epoch [12/50], Val Losses: mse: 2.1741, mae: 0.3933, huber: 0.2488, swd: 1.6722, ept: 93.6359
    Epoch [12/50], Test Losses: mse: 2.0230, mae: 0.4084, huber: 0.2575, swd: 1.5431, ept: 93.4641
      Epoch 12 composite train-obj: 0.242861
            Val objective improved 0.2517 → 0.2488, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 1.9801, mae: 0.3862, huber: 0.2397, swd: 1.5057, ept: 93.6946
    Epoch [13/50], Val Losses: mse: 2.1413, mae: 0.3947, huber: 0.2468, swd: 1.6260, ept: 93.6535
    Epoch [13/50], Test Losses: mse: 1.9918, mae: 0.4087, huber: 0.2549, swd: 1.4990, ept: 93.4872
      Epoch 13 composite train-obj: 0.239747
            Val objective improved 0.2488 → 0.2468, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 1.9601, mae: 0.3840, huber: 0.2374, swd: 1.4887, ept: 93.7110
    Epoch [14/50], Val Losses: mse: 2.1385, mae: 0.3792, huber: 0.2425, swd: 1.6315, ept: 93.6528
    Epoch [14/50], Test Losses: mse: 1.9884, mae: 0.3939, huber: 0.2508, swd: 1.5038, ept: 93.4944
      Epoch 14 composite train-obj: 0.237424
            Val objective improved 0.2468 → 0.2425, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 1.9482, mae: 0.3783, huber: 0.2347, swd: 1.4795, ept: 93.7329
    Epoch [15/50], Val Losses: mse: 2.1120, mae: 0.3833, huber: 0.2415, swd: 1.6010, ept: 93.6850
    Epoch [15/50], Test Losses: mse: 1.9639, mae: 0.3976, huber: 0.2498, swd: 1.4751, ept: 93.5404
      Epoch 15 composite train-obj: 0.234742
            Val objective improved 0.2425 → 0.2415, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 1.9350, mae: 0.3768, huber: 0.2331, swd: 1.4683, ept: 93.7518
    Epoch [16/50], Val Losses: mse: 2.0988, mae: 0.3822, huber: 0.2380, swd: 1.5989, ept: 93.7247
    Epoch [16/50], Test Losses: mse: 1.9515, mae: 0.3955, huber: 0.2458, swd: 1.4727, ept: 93.5553
      Epoch 16 composite train-obj: 0.233106
            Val objective improved 0.2415 → 0.2380, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 1.9237, mae: 0.3740, huber: 0.2312, swd: 1.4587, ept: 93.7684
    Epoch [17/50], Val Losses: mse: 2.0874, mae: 0.3770, huber: 0.2365, swd: 1.5892, ept: 93.7370
    Epoch [17/50], Test Losses: mse: 1.9392, mae: 0.3914, huber: 0.2444, swd: 1.4630, ept: 93.5754
      Epoch 17 composite train-obj: 0.231223
            Val objective improved 0.2380 → 0.2365, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 1.9164, mae: 0.3719, huber: 0.2298, swd: 1.4526, ept: 93.7784
    Epoch [18/50], Val Losses: mse: 2.0841, mae: 0.3848, huber: 0.2379, swd: 1.5919, ept: 93.7776
    Epoch [18/50], Test Losses: mse: 1.9343, mae: 0.3981, huber: 0.2452, swd: 1.4642, ept: 93.6225
      Epoch 18 composite train-obj: 0.229795
            No improvement (0.2379), counter 1/5
    Epoch [19/50], Train Losses: mse: 1.9062, mae: 0.3701, huber: 0.2286, swd: 1.4431, ept: 93.7885
    Epoch [19/50], Val Losses: mse: 2.0740, mae: 0.3719, huber: 0.2337, swd: 1.5827, ept: 93.7790
    Epoch [19/50], Test Losses: mse: 1.9269, mae: 0.3859, huber: 0.2416, swd: 1.4572, ept: 93.6019
      Epoch 19 composite train-obj: 0.228616
            Val objective improved 0.2365 → 0.2337, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 1.8982, mae: 0.3676, huber: 0.2271, swd: 1.4366, ept: 93.7961
    Epoch [20/50], Val Losses: mse: 2.0737, mae: 0.3691, huber: 0.2335, swd: 1.5788, ept: 93.7621
    Epoch [20/50], Test Losses: mse: 1.9279, mae: 0.3836, huber: 0.2415, swd: 1.4537, ept: 93.5946
      Epoch 20 composite train-obj: 0.227067
            Val objective improved 0.2337 → 0.2335, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 1.8911, mae: 0.3660, huber: 0.2259, swd: 1.4304, ept: 93.8078
    Epoch [21/50], Val Losses: mse: 2.0566, mae: 0.3699, huber: 0.2321, swd: 1.5582, ept: 93.7768
    Epoch [21/50], Test Losses: mse: 1.9122, mae: 0.3822, huber: 0.2397, swd: 1.4355, ept: 93.5909
      Epoch 21 composite train-obj: 0.225948
            Val objective improved 0.2335 → 0.2321, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 1.8845, mae: 0.3646, huber: 0.2247, swd: 1.4259, ept: 93.8205
    Epoch [22/50], Val Losses: mse: 2.0514, mae: 0.3670, huber: 0.2313, swd: 1.5538, ept: 93.7758
    Epoch [22/50], Test Losses: mse: 1.9063, mae: 0.3798, huber: 0.2391, swd: 1.4309, ept: 93.6101
      Epoch 22 composite train-obj: 0.224728
            Val objective improved 0.2321 → 0.2313, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 1.8798, mae: 0.3627, huber: 0.2238, swd: 1.4209, ept: 93.8261
    Epoch [23/50], Val Losses: mse: 2.0424, mae: 0.3702, huber: 0.2296, swd: 1.5541, ept: 93.8085
    Epoch [23/50], Test Losses: mse: 1.8985, mae: 0.3827, huber: 0.2370, swd: 1.4314, ept: 93.6001
      Epoch 23 composite train-obj: 0.223812
            Val objective improved 0.2313 → 0.2296, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 1.8712, mae: 0.3611, huber: 0.2226, swd: 1.4140, ept: 93.8396
    Epoch [24/50], Val Losses: mse: 2.0394, mae: 0.3693, huber: 0.2295, swd: 1.5432, ept: 93.8092
    Epoch [24/50], Test Losses: mse: 1.8963, mae: 0.3817, huber: 0.2372, swd: 1.4216, ept: 93.6231
      Epoch 24 composite train-obj: 0.222587
            Val objective improved 0.2296 → 0.2295, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 1.8684, mae: 0.3598, huber: 0.2219, swd: 1.4118, ept: 93.8402
    Epoch [25/50], Val Losses: mse: 2.0345, mae: 0.3596, huber: 0.2269, swd: 1.5478, ept: 93.8156
    Epoch [25/50], Test Losses: mse: 1.8897, mae: 0.3722, huber: 0.2343, swd: 1.4244, ept: 93.6366
      Epoch 25 composite train-obj: 0.221865
            Val objective improved 0.2295 → 0.2269, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 1.8621, mae: 0.3582, huber: 0.2208, swd: 1.4056, ept: 93.8511
    Epoch [26/50], Val Losses: mse: 2.0238, mae: 0.3628, huber: 0.2275, swd: 1.5357, ept: 93.8378
    Epoch [26/50], Test Losses: mse: 1.8773, mae: 0.3754, huber: 0.2347, swd: 1.4114, ept: 93.6900
      Epoch 26 composite train-obj: 0.220836
            No improvement (0.2275), counter 1/5
    Epoch [27/50], Train Losses: mse: 1.8568, mae: 0.3561, huber: 0.2198, swd: 1.4014, ept: 93.8557
    Epoch [27/50], Val Losses: mse: 2.0161, mae: 0.3597, huber: 0.2251, swd: 1.5290, ept: 93.8363
    Epoch [27/50], Test Losses: mse: 1.8731, mae: 0.3715, huber: 0.2326, swd: 1.4072, ept: 93.6555
      Epoch 27 composite train-obj: 0.219840
            Val objective improved 0.2269 → 0.2251, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 1.8511, mae: 0.3561, huber: 0.2193, swd: 1.3965, ept: 93.8656
    Epoch [28/50], Val Losses: mse: 2.0171, mae: 0.3597, huber: 0.2247, swd: 1.5350, ept: 93.8581
    Epoch [28/50], Test Losses: mse: 1.8732, mae: 0.3728, huber: 0.2320, swd: 1.4120, ept: 93.6755
      Epoch 28 composite train-obj: 0.219322
            Val objective improved 0.2251 → 0.2247, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 1.8464, mae: 0.3536, huber: 0.2183, swd: 1.3936, ept: 93.8672
    Epoch [29/50], Val Losses: mse: 2.0150, mae: 0.3650, huber: 0.2250, swd: 1.5308, ept: 93.8702
    Epoch [29/50], Test Losses: mse: 1.8716, mae: 0.3783, huber: 0.2324, swd: 1.4080, ept: 93.7044
      Epoch 29 composite train-obj: 0.218273
            No improvement (0.2250), counter 1/5
    Epoch [30/50], Train Losses: mse: 1.8452, mae: 0.3525, huber: 0.2177, swd: 1.3920, ept: 93.8758
    Epoch [30/50], Val Losses: mse: 2.0116, mae: 0.3665, huber: 0.2254, swd: 1.5327, ept: 93.8761
    Epoch [30/50], Test Losses: mse: 1.8690, mae: 0.3789, huber: 0.2326, swd: 1.4100, ept: 93.6848
      Epoch 30 composite train-obj: 0.217699
            No improvement (0.2254), counter 2/5
    Epoch [31/50], Train Losses: mse: 1.8379, mae: 0.3523, huber: 0.2170, swd: 1.3858, ept: 93.8817
    Epoch [31/50], Val Losses: mse: 1.9990, mae: 0.3611, huber: 0.2239, swd: 1.5083, ept: 93.8456
    Epoch [31/50], Test Losses: mse: 1.8557, mae: 0.3729, huber: 0.2309, swd: 1.3870, ept: 93.6895
      Epoch 31 composite train-obj: 0.217024
            Val objective improved 0.2247 → 0.2239, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 1.8342, mae: 0.3507, huber: 0.2163, swd: 1.3818, ept: 93.8805
    Epoch [32/50], Val Losses: mse: 1.9971, mae: 0.3501, huber: 0.2218, swd: 1.5083, ept: 93.8605
    Epoch [32/50], Test Losses: mse: 1.8548, mae: 0.3626, huber: 0.2292, swd: 1.3873, ept: 93.6966
      Epoch 32 composite train-obj: 0.216318
            Val objective improved 0.2239 → 0.2218, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 1.8315, mae: 0.3512, huber: 0.2163, swd: 1.3801, ept: 93.8887
    Epoch [33/50], Val Losses: mse: 1.9959, mae: 0.3548, huber: 0.2222, swd: 1.5047, ept: 93.8555
    Epoch [33/50], Test Losses: mse: 1.8553, mae: 0.3670, huber: 0.2296, swd: 1.3846, ept: 93.6723
      Epoch 33 composite train-obj: 0.216251
            No improvement (0.2222), counter 1/5
    Epoch [34/50], Train Losses: mse: 1.8266, mae: 0.3484, huber: 0.2151, swd: 1.3765, ept: 93.9010
    Epoch [34/50], Val Losses: mse: 1.9828, mae: 0.3553, huber: 0.2216, swd: 1.5017, ept: 93.8896
    Epoch [34/50], Test Losses: mse: 1.8409, mae: 0.3665, huber: 0.2284, swd: 1.3813, ept: 93.7135
      Epoch 34 composite train-obj: 0.215097
            Val objective improved 0.2218 → 0.2216, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 1.8225, mae: 0.3493, huber: 0.2149, swd: 1.3730, ept: 93.9023
    Epoch [35/50], Val Losses: mse: 1.9890, mae: 0.3532, huber: 0.2211, swd: 1.4991, ept: 93.8703
    Epoch [35/50], Test Losses: mse: 1.8488, mae: 0.3662, huber: 0.2285, swd: 1.3796, ept: 93.6681
      Epoch 35 composite train-obj: 0.214895
            Val objective improved 0.2216 → 0.2211, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 1.8193, mae: 0.3472, huber: 0.2143, swd: 1.3705, ept: 93.9015
    Epoch [36/50], Val Losses: mse: 1.9801, mae: 0.3538, huber: 0.2198, swd: 1.5009, ept: 93.9066
    Epoch [36/50], Test Losses: mse: 1.8379, mae: 0.3657, huber: 0.2267, swd: 1.3794, ept: 93.7296
      Epoch 36 composite train-obj: 0.214306
            Val objective improved 0.2211 → 0.2198, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 1.8146, mae: 0.3456, huber: 0.2134, swd: 1.3657, ept: 93.9047
    Epoch [37/50], Val Losses: mse: 1.9789, mae: 0.3499, huber: 0.2194, swd: 1.4998, ept: 93.8983
    Epoch [37/50], Test Losses: mse: 1.8373, mae: 0.3624, huber: 0.2266, swd: 1.3789, ept: 93.7234
      Epoch 37 composite train-obj: 0.213399
            Val objective improved 0.2198 → 0.2194, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 1.8136, mae: 0.3461, huber: 0.2133, swd: 1.3651, ept: 93.9154
    Epoch [38/50], Val Losses: mse: 1.9784, mae: 0.3536, huber: 0.2200, swd: 1.5046, ept: 93.9139
    Epoch [38/50], Test Losses: mse: 1.8370, mae: 0.3659, huber: 0.2271, swd: 1.3838, ept: 93.7302
      Epoch 38 composite train-obj: 0.213345
            No improvement (0.2200), counter 1/5
    Epoch [39/50], Train Losses: mse: 1.8085, mae: 0.3451, huber: 0.2127, swd: 1.3619, ept: 93.9175
    Epoch [39/50], Val Losses: mse: 1.9729, mae: 0.3533, huber: 0.2217, swd: 1.4783, ept: 93.8802
    Epoch [39/50], Test Losses: mse: 1.8331, mae: 0.3653, huber: 0.2290, swd: 1.3599, ept: 93.7110
      Epoch 39 composite train-obj: 0.212675
            No improvement (0.2217), counter 2/5
    Epoch [40/50], Train Losses: mse: 1.8044, mae: 0.3438, huber: 0.2123, swd: 1.3576, ept: 93.9303
    Epoch [40/50], Val Losses: mse: 1.9624, mae: 0.3494, huber: 0.2179, swd: 1.4857, ept: 93.9094
    Epoch [40/50], Test Losses: mse: 1.8220, mae: 0.3608, huber: 0.2248, swd: 1.3666, ept: 93.7066
      Epoch 40 composite train-obj: 0.212257
            Val objective improved 0.2194 → 0.2179, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 1.8022, mae: 0.3434, huber: 0.2119, swd: 1.3559, ept: 93.9212
    Epoch [41/50], Val Losses: mse: 1.9624, mae: 0.3459, huber: 0.2180, swd: 1.4818, ept: 93.9012
    Epoch [41/50], Test Losses: mse: 1.8236, mae: 0.3575, huber: 0.2251, swd: 1.3644, ept: 93.6960
      Epoch 41 composite train-obj: 0.211872
            No improvement (0.2180), counter 1/5
    Epoch [42/50], Train Losses: mse: 1.7997, mae: 0.3428, huber: 0.2115, swd: 1.3541, ept: 93.9322
    Epoch [42/50], Val Losses: mse: 1.9518, mae: 0.3508, huber: 0.2172, swd: 1.4745, ept: 93.9190
    Epoch [42/50], Test Losses: mse: 1.8109, mae: 0.3633, huber: 0.2240, swd: 1.3548, ept: 93.7526
      Epoch 42 composite train-obj: 0.211541
            Val objective improved 0.2179 → 0.2172, saving checkpoint.
    Epoch [43/50], Train Losses: mse: 1.7974, mae: 0.3435, huber: 0.2114, swd: 1.3514, ept: 93.9330
    Epoch [43/50], Val Losses: mse: 1.9532, mae: 0.3465, huber: 0.2164, swd: 1.4754, ept: 93.9138
    Epoch [43/50], Test Losses: mse: 1.8133, mae: 0.3589, huber: 0.2234, swd: 1.3567, ept: 93.7414
      Epoch 43 composite train-obj: 0.211422
            Val objective improved 0.2172 → 0.2164, saving checkpoint.
    Epoch [44/50], Train Losses: mse: 1.7937, mae: 0.3418, huber: 0.2109, swd: 1.3481, ept: 93.9425
    Epoch [44/50], Val Losses: mse: 1.9503, mae: 0.3493, huber: 0.2169, swd: 1.4752, ept: 93.9267
    Epoch [44/50], Test Losses: mse: 1.8113, mae: 0.3618, huber: 0.2238, swd: 1.3569, ept: 93.7472
      Epoch 44 composite train-obj: 0.210859
            No improvement (0.2169), counter 1/5
    Epoch [45/50], Train Losses: mse: 1.7901, mae: 0.3411, huber: 0.2103, swd: 1.3461, ept: 93.9394
    Epoch [45/50], Val Losses: mse: 1.9465, mae: 0.3499, huber: 0.2161, swd: 1.4700, ept: 93.9383
    Epoch [45/50], Test Losses: mse: 1.8063, mae: 0.3624, huber: 0.2229, swd: 1.3505, ept: 93.7699
      Epoch 45 composite train-obj: 0.210338
            Val objective improved 0.2164 → 0.2161, saving checkpoint.
    Epoch [46/50], Train Losses: mse: 1.7871, mae: 0.3417, huber: 0.2102, swd: 1.3437, ept: 93.9445
    Epoch [46/50], Val Losses: mse: 1.9516, mae: 0.3420, huber: 0.2164, swd: 1.4763, ept: 93.9257
    Epoch [46/50], Test Losses: mse: 1.8124, mae: 0.3542, huber: 0.2232, swd: 1.3579, ept: 93.7237
      Epoch 46 composite train-obj: 0.210152
            No improvement (0.2164), counter 1/5
    Epoch [47/50], Train Losses: mse: 1.7859, mae: 0.3403, huber: 0.2099, swd: 1.3419, ept: 93.9402
    Epoch [47/50], Val Losses: mse: 1.9466, mae: 0.3479, huber: 0.2176, swd: 1.4647, ept: 93.9195
    Epoch [47/50], Test Losses: mse: 1.8061, mae: 0.3614, huber: 0.2244, swd: 1.3457, ept: 93.7618
      Epoch 47 composite train-obj: 0.209938
            No improvement (0.2176), counter 2/5
    Epoch [48/50], Train Losses: mse: 1.7817, mae: 0.3398, huber: 0.2094, swd: 1.3384, ept: 93.9495
    Epoch [48/50], Val Losses: mse: 1.9409, mae: 0.3454, huber: 0.2154, swd: 1.4649, ept: 93.9395
    Epoch [48/50], Test Losses: mse: 1.8024, mae: 0.3576, huber: 0.2225, swd: 1.3475, ept: 93.7639
      Epoch 48 composite train-obj: 0.209397
            Val objective improved 0.2161 → 0.2154, saving checkpoint.
    Epoch [49/50], Train Losses: mse: 1.7802, mae: 0.3396, huber: 0.2092, swd: 1.3377, ept: 93.9493
    Epoch [49/50], Val Losses: mse: 1.9462, mae: 0.3376, huber: 0.2146, swd: 1.4743, ept: 93.9357
    Epoch [49/50], Test Losses: mse: 1.8081, mae: 0.3500, huber: 0.2217, swd: 1.3565, ept: 93.7482
      Epoch 49 composite train-obj: 0.209231
            Val objective improved 0.2154 → 0.2146, saving checkpoint.
    Epoch [50/50], Train Losses: mse: 1.7776, mae: 0.3385, huber: 0.2088, swd: 1.3354, ept: 93.9513
    Epoch [50/50], Val Losses: mse: 1.9399, mae: 0.3412, huber: 0.2144, swd: 1.4630, ept: 93.9440
    Epoch [50/50], Test Losses: mse: 1.8005, mae: 0.3530, huber: 0.2213, swd: 1.3444, ept: 93.7824
      Epoch 50 composite train-obj: 0.208799
            Val objective improved 0.2146 → 0.2144, saving checkpoint.
    Epoch [50/50], Test Losses: mse: 1.8005, mae: 0.3530, huber: 0.2213, swd: 1.3444, ept: 93.7824
    Best round's Test MSE: 1.8005, MAE: 0.3530, SWD: 1.3444
    Best round's Validation MSE: 1.9399, MAE: 0.3412, SWD: 1.4630
    Best round's Test verification MSE : 1.8005, MAE: 0.3530, SWD: 1.3444
    Time taken: 118.90 seconds
    
    ==================================================
    Experiment Summary (DLinear_rossler_seq96_pred96_20250513_1338)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 1.7967 ± 0.0056
      mae: 0.3534 ± 0.0018
      huber: 0.2211 ± 0.0002
      swd: 1.4215 ± 0.0570
      ept: 93.7704 ± 0.0158
      count: 40.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 1.9356 ± 0.0052
      mae: 0.3416 ± 0.0019
      huber: 0.2144 ± 0.0001
      swd: 1.5469 ± 0.0624
      ept: 93.9428 ± 0.0034
      count: 40.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 361.19 seconds
    
    Experiment complete: DLinear_rossler_seq96_pred96_20250513_1338
    Model: DLinear
    Dataset: rossler
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
    channels=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 283
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 6.4712, mae: 1.1479, huber: 0.8760, swd: 5.0951, ept: 168.5146
    Epoch [1/50], Val Losses: mse: 5.9043, mae: 0.9109, huber: 0.6652, swd: 5.0685, ept: 179.3854
    Epoch [1/50], Test Losses: mse: 5.4714, mae: 0.9518, huber: 0.6937, swd: 4.6889, ept: 178.9549
      Epoch 1 composite train-obj: 0.875994
            Val objective improved inf → 0.6652, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.0756, mae: 0.8726, huber: 0.6205, swd: 4.3521, ept: 181.3033
    Epoch [2/50], Val Losses: mse: 5.4900, mae: 0.8775, huber: 0.6329, swd: 4.6849, ept: 180.4089
    Epoch [2/50], Test Losses: mse: 5.0765, mae: 0.9104, huber: 0.6569, swd: 4.3213, ept: 180.0926
      Epoch 2 composite train-obj: 0.620531
            Val objective improved 0.6652 → 0.6329, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.8227, mae: 0.8404, huber: 0.5939, swd: 4.1281, ept: 182.0669
    Epoch [3/50], Val Losses: mse: 5.2205, mae: 0.8519, huber: 0.6085, swd: 4.4873, ept: 181.1800
    Epoch [3/50], Test Losses: mse: 4.8205, mae: 0.8823, huber: 0.6290, swd: 4.1403, ept: 181.1047
      Epoch 3 composite train-obj: 0.593934
            Val objective improved 0.6329 → 0.6085, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6587, mae: 0.8165, huber: 0.5749, swd: 3.9845, ept: 182.5897
    Epoch [4/50], Val Losses: mse: 5.0916, mae: 0.8304, huber: 0.5924, swd: 4.3600, ept: 181.5545
    Epoch [4/50], Test Losses: mse: 4.6997, mae: 0.8594, huber: 0.6125, swd: 4.0187, ept: 181.3952
      Epoch 4 composite train-obj: 0.574852
            Val objective improved 0.6085 → 0.5924, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.5830, mae: 0.8030, huber: 0.5644, swd: 3.9171, ept: 182.8337
    Epoch [5/50], Val Losses: mse: 5.0733, mae: 0.8169, huber: 0.5825, swd: 4.3102, ept: 181.7398
    Epoch [5/50], Test Losses: mse: 4.6911, mae: 0.8486, huber: 0.6043, swd: 3.9754, ept: 181.3398
      Epoch 5 composite train-obj: 0.564355
            Val objective improved 0.5924 → 0.5825, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.4801, mae: 0.7883, huber: 0.5522, swd: 3.8239, ept: 183.1463
    Epoch [6/50], Val Losses: mse: 4.9647, mae: 0.8019, huber: 0.5704, swd: 4.2231, ept: 182.1036
    Epoch [6/50], Test Losses: mse: 4.5889, mae: 0.8310, huber: 0.5912, swd: 3.8943, ept: 181.8466
      Epoch 6 composite train-obj: 0.552237
            Val objective improved 0.5825 → 0.5704, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.4329, mae: 0.7780, huber: 0.5447, swd: 3.7858, ept: 183.3351
    Epoch [7/50], Val Losses: mse: 4.8965, mae: 0.7922, huber: 0.5629, swd: 4.1704, ept: 182.3009
    Epoch [7/50], Test Losses: mse: 4.5223, mae: 0.8203, huber: 0.5827, swd: 3.8442, ept: 182.0633
      Epoch 7 composite train-obj: 0.544703
            Val objective improved 0.5704 → 0.5629, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 4.3682, mae: 0.7681, huber: 0.5365, swd: 3.7227, ept: 183.5253
    Epoch [8/50], Val Losses: mse: 4.8553, mae: 0.7815, huber: 0.5563, swd: 4.1611, ept: 182.4860
    Epoch [8/50], Test Losses: mse: 4.4837, mae: 0.8102, huber: 0.5755, swd: 3.8398, ept: 182.2780
      Epoch 8 composite train-obj: 0.536530
            Val objective improved 0.5629 → 0.5563, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 4.3512, mae: 0.7614, huber: 0.5324, swd: 3.7092, ept: 183.6128
    Epoch [9/50], Val Losses: mse: 4.7656, mae: 0.7964, huber: 0.5564, swd: 4.0721, ept: 182.8544
    Epoch [9/50], Test Losses: mse: 4.3988, mae: 0.8238, huber: 0.5746, swd: 3.7486, ept: 182.7172
      Epoch 9 composite train-obj: 0.532385
            No improvement (0.5564), counter 1/5
    Epoch [10/50], Train Losses: mse: 4.3032, mae: 0.7541, huber: 0.5260, swd: 3.6680, ept: 183.7834
    Epoch [10/50], Val Losses: mse: 4.8080, mae: 0.7729, huber: 0.5484, swd: 4.1340, ept: 182.7710
    Epoch [10/50], Test Losses: mse: 4.4404, mae: 0.7995, huber: 0.5668, swd: 3.8174, ept: 182.5598
      Epoch 10 composite train-obj: 0.526007
            Val objective improved 0.5563 → 0.5484, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 4.2898, mae: 0.7471, huber: 0.5214, swd: 3.6577, ept: 183.9051
    Epoch [11/50], Val Losses: mse: 4.7932, mae: 0.7667, huber: 0.5436, swd: 4.0993, ept: 182.7737
    Epoch [11/50], Test Losses: mse: 4.4231, mae: 0.7929, huber: 0.5618, swd: 3.7793, ept: 182.5433
      Epoch 11 composite train-obj: 0.521379
            Val objective improved 0.5484 → 0.5436, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 4.2875, mae: 0.7441, huber: 0.5193, swd: 3.6536, ept: 183.9452
    Epoch [12/50], Val Losses: mse: 4.7168, mae: 0.7675, huber: 0.5398, swd: 4.0364, ept: 183.1271
    Epoch [12/50], Test Losses: mse: 4.3483, mae: 0.7918, huber: 0.5565, swd: 3.7146, ept: 182.8802
      Epoch 12 composite train-obj: 0.519286
            Val objective improved 0.5436 → 0.5398, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 4.2555, mae: 0.7383, huber: 0.5144, swd: 3.6309, ept: 184.0721
    Epoch [13/50], Val Losses: mse: 4.7056, mae: 0.7576, huber: 0.5366, swd: 4.0085, ept: 183.1510
    Epoch [13/50], Test Losses: mse: 4.3384, mae: 0.7818, huber: 0.5538, swd: 3.6861, ept: 182.7998
      Epoch 13 composite train-obj: 0.514399
            Val objective improved 0.5398 → 0.5366, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 4.2479, mae: 0.7357, huber: 0.5129, swd: 3.6217, ept: 184.1249
    Epoch [14/50], Val Losses: mse: 4.6827, mae: 0.7554, huber: 0.5324, swd: 4.0018, ept: 183.2897
    Epoch [14/50], Test Losses: mse: 4.3134, mae: 0.7804, huber: 0.5488, swd: 3.6797, ept: 183.0363
      Epoch 14 composite train-obj: 0.512891
            Val objective improved 0.5366 → 0.5324, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 4.2249, mae: 0.7300, huber: 0.5084, swd: 3.6066, ept: 184.2707
    Epoch [15/50], Val Losses: mse: 4.6961, mae: 0.7533, huber: 0.5309, swd: 3.9932, ept: 183.3877
    Epoch [15/50], Test Losses: mse: 4.3285, mae: 0.7781, huber: 0.5480, swd: 3.6680, ept: 182.9512
      Epoch 15 composite train-obj: 0.508396
            Val objective improved 0.5324 → 0.5309, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 4.2213, mae: 0.7267, huber: 0.5060, swd: 3.6037, ept: 184.2766
    Epoch [16/50], Val Losses: mse: 4.6827, mae: 0.7494, huber: 0.5268, swd: 4.0087, ept: 183.5019
    Epoch [16/50], Test Losses: mse: 4.3139, mae: 0.7736, huber: 0.5431, swd: 3.6860, ept: 183.2059
      Epoch 16 composite train-obj: 0.505974
            Val objective improved 0.5309 → 0.5268, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 4.2026, mae: 0.7224, huber: 0.5025, swd: 3.5894, ept: 184.3888
    Epoch [17/50], Val Losses: mse: 4.6844, mae: 0.7392, huber: 0.5258, swd: 4.0038, ept: 183.4987
    Epoch [17/50], Test Losses: mse: 4.3158, mae: 0.7627, huber: 0.5423, swd: 3.6817, ept: 183.0793
      Epoch 17 composite train-obj: 0.502531
            Val objective improved 0.5268 → 0.5258, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 4.2016, mae: 0.7205, huber: 0.5015, swd: 3.5898, ept: 184.4113
    Epoch [18/50], Val Losses: mse: 4.7490, mae: 0.7393, huber: 0.5243, swd: 4.0894, ept: 183.4913
    Epoch [18/50], Test Losses: mse: 4.3831, mae: 0.7653, huber: 0.5423, swd: 3.7712, ept: 183.1123
      Epoch 18 composite train-obj: 0.501510
            Val objective improved 0.5258 → 0.5243, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 4.2235, mae: 0.7209, huber: 0.5022, swd: 3.6066, ept: 184.3891
    Epoch [19/50], Val Losses: mse: 4.6911, mae: 0.7653, huber: 0.5293, swd: 4.0459, ept: 183.7853
    Epoch [19/50], Test Losses: mse: 4.3270, mae: 0.7906, huber: 0.5460, swd: 3.7252, ept: 183.4702
      Epoch 19 composite train-obj: 0.502194
            No improvement (0.5293), counter 1/5
    Epoch [20/50], Train Losses: mse: 4.1825, mae: 0.7184, huber: 0.4985, swd: 3.5754, ept: 184.5430
    Epoch [20/50], Val Losses: mse: 4.7119, mae: 0.7299, huber: 0.5199, swd: 4.0285, ept: 183.5839
    Epoch [20/50], Test Losses: mse: 4.3432, mae: 0.7538, huber: 0.5372, swd: 3.7064, ept: 183.1086
      Epoch 20 composite train-obj: 0.498497
            Val objective improved 0.5243 → 0.5199, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 4.1967, mae: 0.7147, huber: 0.4972, swd: 3.5868, ept: 184.5598
    Epoch [21/50], Val Losses: mse: 4.6288, mae: 0.7484, huber: 0.5206, swd: 3.9789, ept: 183.8936
    Epoch [21/50], Test Losses: mse: 4.2601, mae: 0.7702, huber: 0.5355, swd: 3.6539, ept: 183.5804
      Epoch 21 composite train-obj: 0.497213
            No improvement (0.5206), counter 1/5
    Epoch [22/50], Train Losses: mse: 4.1716, mae: 0.7116, huber: 0.4939, swd: 3.5691, ept: 184.6497
    Epoch [22/50], Val Losses: mse: 4.6704, mae: 0.7304, huber: 0.5165, swd: 4.0048, ept: 183.7692
    Epoch [22/50], Test Losses: mse: 4.3001, mae: 0.7528, huber: 0.5323, swd: 3.6819, ept: 183.2999
      Epoch 22 composite train-obj: 0.493936
            Val objective improved 0.5199 → 0.5165, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 4.1795, mae: 0.7111, huber: 0.4941, swd: 3.5755, ept: 184.6157
    Epoch [23/50], Val Losses: mse: 4.6492, mae: 0.7391, huber: 0.5234, swd: 3.9419, ept: 183.7781
    Epoch [23/50], Test Losses: mse: 4.2829, mae: 0.7597, huber: 0.5395, swd: 3.6166, ept: 183.2665
      Epoch 23 composite train-obj: 0.494087
            No improvement (0.5234), counter 1/5
    Epoch [24/50], Train Losses: mse: 4.1736, mae: 0.7099, huber: 0.4928, swd: 3.5707, ept: 184.6875
    Epoch [24/50], Val Losses: mse: 4.6852, mae: 0.7264, huber: 0.5162, swd: 4.0009, ept: 183.7220
    Epoch [24/50], Test Losses: mse: 4.3151, mae: 0.7480, huber: 0.5328, swd: 3.6768, ept: 183.2705
      Epoch 24 composite train-obj: 0.492766
            Val objective improved 0.5165 → 0.5162, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 4.1729, mae: 0.7078, huber: 0.4916, swd: 3.5711, ept: 184.6694
    Epoch [25/50], Val Losses: mse: 4.6443, mae: 0.7223, huber: 0.5118, swd: 3.9791, ept: 183.8389
    Epoch [25/50], Test Losses: mse: 4.2766, mae: 0.7450, huber: 0.5279, swd: 3.6565, ept: 183.3315
      Epoch 25 composite train-obj: 0.491560
            Val objective improved 0.5162 → 0.5118, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 4.1715, mae: 0.7079, huber: 0.4916, swd: 3.5683, ept: 184.7106
    Epoch [26/50], Val Losses: mse: 4.6100, mae: 0.7354, huber: 0.5150, swd: 3.9502, ept: 184.0163
    Epoch [26/50], Test Losses: mse: 4.2458, mae: 0.7596, huber: 0.5313, swd: 3.6266, ept: 183.5578
      Epoch 26 composite train-obj: 0.491601
            No improvement (0.5150), counter 1/5
    Epoch [27/50], Train Losses: mse: 4.1630, mae: 0.7052, huber: 0.4896, swd: 3.5632, ept: 184.7184
    Epoch [27/50], Val Losses: mse: 4.6915, mae: 0.7282, huber: 0.5143, swd: 4.0130, ept: 183.7643
    Epoch [27/50], Test Losses: mse: 4.3212, mae: 0.7504, huber: 0.5311, swd: 3.6898, ept: 183.2815
      Epoch 27 composite train-obj: 0.489637
            No improvement (0.5143), counter 2/5
    Epoch [28/50], Train Losses: mse: 4.1633, mae: 0.7035, huber: 0.4883, swd: 3.5644, ept: 184.7780
    Epoch [28/50], Val Losses: mse: 4.6694, mae: 0.7189, huber: 0.5096, swd: 3.9994, ept: 183.8809
    Epoch [28/50], Test Losses: mse: 4.2998, mae: 0.7406, huber: 0.5259, swd: 3.6760, ept: 183.3734
      Epoch 28 composite train-obj: 0.488341
            Val objective improved 0.5118 → 0.5096, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 4.1580, mae: 0.7019, huber: 0.4869, swd: 3.5593, ept: 184.7764
    Epoch [29/50], Val Losses: mse: 4.6249, mae: 0.7216, huber: 0.5087, swd: 3.9461, ept: 183.9484
    Epoch [29/50], Test Losses: mse: 4.2560, mae: 0.7438, huber: 0.5244, swd: 3.6213, ept: 183.4828
      Epoch 29 composite train-obj: 0.486943
            Val objective improved 0.5096 → 0.5087, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 4.1436, mae: 0.7002, huber: 0.4859, swd: 3.5451, ept: 184.8138
    Epoch [30/50], Val Losses: mse: 4.6149, mae: 0.7366, huber: 0.5120, swd: 3.9729, ept: 184.1602
    Epoch [30/50], Test Losses: mse: 4.2468, mae: 0.7577, huber: 0.5269, swd: 3.6488, ept: 183.7705
      Epoch 30 composite train-obj: 0.485854
            No improvement (0.5120), counter 1/5
    Epoch [31/50], Train Losses: mse: 4.1459, mae: 0.6989, huber: 0.4849, swd: 3.5499, ept: 184.8294
    Epoch [31/50], Val Losses: mse: 4.6303, mae: 0.7216, huber: 0.5078, swd: 3.9907, ept: 184.0258
    Epoch [31/50], Test Losses: mse: 4.2599, mae: 0.7439, huber: 0.5231, swd: 3.6673, ept: 183.6974
      Epoch 31 composite train-obj: 0.484872
            Val objective improved 0.5087 → 0.5078, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 4.1886, mae: 0.7016, huber: 0.4879, swd: 3.5914, ept: 184.7666
    Epoch [32/50], Val Losses: mse: 4.5722, mae: 0.7292, huber: 0.5080, swd: 3.9238, ept: 184.1347
    Epoch [32/50], Test Losses: mse: 4.2057, mae: 0.7513, huber: 0.5229, swd: 3.5994, ept: 183.6262
      Epoch 32 composite train-obj: 0.487934
            No improvement (0.5080), counter 1/5
    Epoch [33/50], Train Losses: mse: 4.1325, mae: 0.6963, huber: 0.4828, swd: 3.5378, ept: 184.8703
    Epoch [33/50], Val Losses: mse: 4.6291, mae: 0.7209, huber: 0.5069, swd: 3.9868, ept: 184.1373
    Epoch [33/50], Test Losses: mse: 4.2620, mae: 0.7439, huber: 0.5228, swd: 3.6658, ept: 183.7739
      Epoch 33 composite train-obj: 0.482848
            Val objective improved 0.5078 → 0.5069, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 4.1437, mae: 0.6974, huber: 0.4838, swd: 3.5491, ept: 184.8712
    Epoch [34/50], Val Losses: mse: 4.6200, mae: 0.7136, huber: 0.5043, swd: 3.9562, ept: 184.0135
    Epoch [34/50], Test Losses: mse: 4.2504, mae: 0.7343, huber: 0.5197, swd: 3.6328, ept: 183.5588
      Epoch 34 composite train-obj: 0.483790
            Val objective improved 0.5069 → 0.5043, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 4.1456, mae: 0.6958, huber: 0.4828, swd: 3.5492, ept: 184.8558
    Epoch [35/50], Val Losses: mse: 4.6154, mae: 0.7207, huber: 0.5053, swd: 3.9625, ept: 184.1284
    Epoch [35/50], Test Losses: mse: 4.2489, mae: 0.7433, huber: 0.5212, swd: 3.6389, ept: 183.6533
      Epoch 35 composite train-obj: 0.482788
            No improvement (0.5053), counter 1/5
    Epoch [36/50], Train Losses: mse: 4.1300, mae: 0.6945, huber: 0.4817, swd: 3.5357, ept: 184.8989
    Epoch [36/50], Val Losses: mse: 4.6381, mae: 0.7102, huber: 0.5026, swd: 3.9744, ept: 184.0727
    Epoch [36/50], Test Losses: mse: 4.2696, mae: 0.7331, huber: 0.5191, swd: 3.6502, ept: 183.6341
      Epoch 36 composite train-obj: 0.481686
            Val objective improved 0.5043 → 0.5026, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 4.1321, mae: 0.6942, huber: 0.4814, swd: 3.5341, ept: 184.8813
    Epoch [37/50], Val Losses: mse: 4.6166, mae: 0.7143, huber: 0.5027, swd: 3.9464, ept: 184.0473
    Epoch [37/50], Test Losses: mse: 4.2465, mae: 0.7375, huber: 0.5185, swd: 3.6222, ept: 183.6286
      Epoch 37 composite train-obj: 0.481352
            No improvement (0.5027), counter 1/5
    Epoch [38/50], Train Losses: mse: 4.1187, mae: 0.6914, huber: 0.4795, swd: 3.5245, ept: 184.9145
    Epoch [38/50], Val Losses: mse: 4.7151, mae: 0.7063, huber: 0.5052, swd: 4.0371, ept: 183.9499
    Epoch [38/50], Test Losses: mse: 4.3460, mae: 0.7316, huber: 0.5232, swd: 3.7154, ept: 183.4630
      Epoch 38 composite train-obj: 0.479464
            No improvement (0.5052), counter 2/5
    Epoch [39/50], Train Losses: mse: 4.1367, mae: 0.6912, huber: 0.4799, swd: 3.5380, ept: 184.9078
    Epoch [39/50], Val Losses: mse: 4.5812, mae: 0.7142, huber: 0.5020, swd: 3.9262, ept: 184.1088
    Epoch [39/50], Test Losses: mse: 4.2128, mae: 0.7354, huber: 0.5170, swd: 3.6027, ept: 183.7189
      Epoch 39 composite train-obj: 0.479892
            Val objective improved 0.5026 → 0.5020, saving checkpoint.
    Epoch [40/50], Train Losses: mse: 4.1394, mae: 0.6922, huber: 0.4802, swd: 3.5406, ept: 184.9132
    Epoch [40/50], Val Losses: mse: 4.5805, mae: 0.7120, huber: 0.5007, swd: 3.9295, ept: 184.1768
    Epoch [40/50], Test Losses: mse: 4.2108, mae: 0.7330, huber: 0.5155, swd: 3.6058, ept: 183.7537
      Epoch 40 composite train-obj: 0.480179
            Val objective improved 0.5020 → 0.5007, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 4.1259, mae: 0.6899, huber: 0.4785, swd: 3.5321, ept: 184.9430
    Epoch [41/50], Val Losses: mse: 4.5991, mae: 0.7226, huber: 0.5028, swd: 3.9655, ept: 184.2569
    Epoch [41/50], Test Losses: mse: 4.2324, mae: 0.7445, huber: 0.5178, swd: 3.6442, ept: 183.8784
      Epoch 41 composite train-obj: 0.478486
            No improvement (0.5028), counter 1/5
    Epoch [42/50], Train Losses: mse: 4.1132, mae: 0.6891, huber: 0.4778, swd: 3.5191, ept: 184.9590
    Epoch [42/50], Val Losses: mse: 4.5899, mae: 0.7324, huber: 0.5079, swd: 3.9551, ept: 184.3408
    Epoch [42/50], Test Losses: mse: 4.2262, mae: 0.7551, huber: 0.5233, swd: 3.6350, ept: 183.9573
      Epoch 42 composite train-obj: 0.477824
            No improvement (0.5079), counter 2/5
    Epoch [43/50], Train Losses: mse: 4.1148, mae: 0.6892, huber: 0.4777, swd: 3.5218, ept: 184.9577
    Epoch [43/50], Val Losses: mse: 4.6141, mae: 0.7047, huber: 0.4985, swd: 3.9515, ept: 184.1321
    Epoch [43/50], Test Losses: mse: 4.2466, mae: 0.7268, huber: 0.5147, swd: 3.6295, ept: 183.6709
      Epoch 43 composite train-obj: 0.477685
            Val objective improved 0.5007 → 0.4985, saving checkpoint.
    Epoch [44/50], Train Losses: mse: 4.1022, mae: 0.6873, huber: 0.4763, swd: 3.5082, ept: 185.0149
    Epoch [44/50], Val Losses: mse: 4.6601, mae: 0.7032, huber: 0.4995, swd: 4.0095, ept: 184.1229
    Epoch [44/50], Test Losses: mse: 4.2934, mae: 0.7262, huber: 0.5166, swd: 3.6891, ept: 183.7086
      Epoch 44 composite train-obj: 0.476261
            No improvement (0.4995), counter 1/5
    Epoch [45/50], Train Losses: mse: 4.1302, mae: 0.6885, huber: 0.4780, swd: 3.5371, ept: 184.9510
    Epoch [45/50], Val Losses: mse: 4.5846, mae: 0.7120, huber: 0.5000, swd: 3.9069, ept: 184.1321
    Epoch [45/50], Test Losses: mse: 4.2174, mae: 0.7330, huber: 0.5158, swd: 3.5835, ept: 183.6928
      Epoch 45 composite train-obj: 0.478039
            No improvement (0.5000), counter 2/5
    Epoch [46/50], Train Losses: mse: 4.1099, mae: 0.6876, huber: 0.4765, swd: 3.5154, ept: 185.0084
    Epoch [46/50], Val Losses: mse: 4.6269, mae: 0.7089, huber: 0.5005, swd: 3.9849, ept: 184.1338
    Epoch [46/50], Test Losses: mse: 4.2582, mae: 0.7329, huber: 0.5164, swd: 3.6648, ept: 183.8279
      Epoch 46 composite train-obj: 0.476496
            No improvement (0.5005), counter 3/5
    Epoch [47/50], Train Losses: mse: 4.0997, mae: 0.6856, huber: 0.4752, swd: 3.5072, ept: 185.0418
    Epoch [47/50], Val Losses: mse: 4.6068, mae: 0.7030, huber: 0.4962, swd: 3.9411, ept: 184.2151
    Epoch [47/50], Test Losses: mse: 4.2416, mae: 0.7273, huber: 0.5129, swd: 3.6196, ept: 183.7076
      Epoch 47 composite train-obj: 0.475201
            Val objective improved 0.4985 → 0.4962, saving checkpoint.
    Epoch [48/50], Train Losses: mse: 4.1139, mae: 0.6859, huber: 0.4759, swd: 3.5197, ept: 185.0137
    Epoch [48/50], Val Losses: mse: 4.5521, mae: 0.7056, huber: 0.4973, swd: 3.9009, ept: 184.2630
    Epoch [48/50], Test Losses: mse: 4.1867, mae: 0.7271, huber: 0.5125, swd: 3.5795, ept: 183.8325
      Epoch 48 composite train-obj: 0.475873
            No improvement (0.4973), counter 1/5
    Epoch [49/50], Train Losses: mse: 4.1042, mae: 0.6847, huber: 0.4748, swd: 3.5111, ept: 185.0271
    Epoch [49/50], Val Losses: mse: 4.5692, mae: 0.7103, huber: 0.4971, swd: 3.9152, ept: 184.2678
    Epoch [49/50], Test Losses: mse: 4.2005, mae: 0.7306, huber: 0.5120, swd: 3.5919, ept: 183.8720
      Epoch 49 composite train-obj: 0.474789
            No improvement (0.4971), counter 2/5
    Epoch [50/50], Train Losses: mse: 4.0925, mae: 0.6831, huber: 0.4735, swd: 3.5012, ept: 185.0559
    Epoch [50/50], Val Losses: mse: 4.5628, mae: 0.7018, huber: 0.4957, swd: 3.9181, ept: 184.3100
    Epoch [50/50], Test Losses: mse: 4.1957, mae: 0.7233, huber: 0.5106, swd: 3.5962, ept: 183.8461
      Epoch 50 composite train-obj: 0.473472
            Val objective improved 0.4962 → 0.4957, saving checkpoint.
    Epoch [50/50], Test Losses: mse: 4.1957, mae: 0.7233, huber: 0.5106, swd: 3.5962, ept: 183.8461
    Best round's Test MSE: 4.1957, MAE: 0.7233, SWD: 3.5962
    Best round's Validation MSE: 4.5628, MAE: 0.7018, SWD: 3.9181
    Best round's Test verification MSE : 4.1957, MAE: 0.7233, SWD: 3.5962
    Time taken: 117.88 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.4068, mae: 1.1409, huber: 0.8681, swd: 5.1864, ept: 169.4003
    Epoch [1/50], Val Losses: mse: 5.8998, mae: 0.9079, huber: 0.6657, swd: 5.1308, ept: 179.6235
    Epoch [1/50], Test Losses: mse: 5.4711, mae: 0.9485, huber: 0.6950, swd: 4.7459, ept: 178.9731
      Epoch 1 composite train-obj: 0.868090
            Val objective improved inf → 0.6657, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.1100, mae: 0.8742, huber: 0.6229, swd: 4.4449, ept: 181.3071
    Epoch [2/50], Val Losses: mse: 5.4507, mae: 0.8864, huber: 0.6339, swd: 4.7683, ept: 180.4283
    Epoch [2/50], Test Losses: mse: 5.0369, mae: 0.9200, huber: 0.6562, swd: 4.4024, ept: 180.4162
      Epoch 2 composite train-obj: 0.622946
            Val objective improved 0.6657 → 0.6339, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.8294, mae: 0.8417, huber: 0.5949, swd: 4.1932, ept: 182.0699
    Epoch [3/50], Val Losses: mse: 5.2985, mae: 0.8443, huber: 0.6070, swd: 4.5936, ept: 181.1161
    Epoch [3/50], Test Losses: mse: 4.8969, mae: 0.8758, huber: 0.6291, swd: 4.2387, ept: 180.7885
      Epoch 3 composite train-obj: 0.594946
            Val objective improved 0.6339 → 0.6070, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6615, mae: 0.8181, huber: 0.5760, swd: 4.0409, ept: 182.5540
    Epoch [4/50], Val Losses: mse: 5.0964, mae: 0.8335, huber: 0.5921, swd: 4.4260, ept: 181.5758
    Epoch [4/50], Test Losses: mse: 4.7069, mae: 0.8637, huber: 0.6126, swd: 4.0805, ept: 181.4133
      Epoch 4 composite train-obj: 0.575992
            Val objective improved 0.6070 → 0.5921, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.5537, mae: 0.8016, huber: 0.5624, swd: 3.9442, ept: 182.9118
    Epoch [5/50], Val Losses: mse: 5.1015, mae: 0.8130, huber: 0.5796, swd: 4.4349, ept: 181.7851
    Epoch [5/50], Test Losses: mse: 4.7185, mae: 0.8438, huber: 0.6012, swd: 4.0979, ept: 181.5857
      Epoch 5 composite train-obj: 0.562450
            Val objective improved 0.5921 → 0.5796, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.4786, mae: 0.7864, huber: 0.5512, swd: 3.8750, ept: 183.1581
    Epoch [6/50], Val Losses: mse: 4.9478, mae: 0.8064, huber: 0.5706, swd: 4.2749, ept: 182.2021
    Epoch [6/50], Test Losses: mse: 4.5710, mae: 0.8353, huber: 0.5906, swd: 3.9420, ept: 181.8895
      Epoch 6 composite train-obj: 0.551192
            Val objective improved 0.5796 → 0.5706, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.4153, mae: 0.7753, huber: 0.5424, swd: 3.8180, ept: 183.3810
    Epoch [7/50], Val Losses: mse: 4.8780, mae: 0.8012, huber: 0.5647, swd: 4.2532, ept: 182.5015
    Epoch [7/50], Test Losses: mse: 4.5044, mae: 0.8288, huber: 0.5834, swd: 3.9249, ept: 182.3333
      Epoch 7 composite train-obj: 0.542394
            Val objective improved 0.5706 → 0.5647, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 4.3859, mae: 0.7688, huber: 0.5373, swd: 3.7899, ept: 183.5127
    Epoch [8/50], Val Losses: mse: 4.7990, mae: 0.7899, huber: 0.5568, swd: 4.1611, ept: 182.6541
    Epoch [8/50], Test Losses: mse: 4.4274, mae: 0.8172, huber: 0.5747, swd: 3.8340, ept: 182.4683
      Epoch 8 composite train-obj: 0.537348
            Val objective improved 0.5647 → 0.5568, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 4.3376, mae: 0.7605, huber: 0.5308, swd: 3.7469, ept: 183.6482
    Epoch [9/50], Val Losses: mse: 4.8133, mae: 0.7772, huber: 0.5506, swd: 4.1648, ept: 182.6615
    Epoch [9/50], Test Losses: mse: 4.4446, mae: 0.8039, huber: 0.5697, swd: 3.8383, ept: 182.4070
      Epoch 9 composite train-obj: 0.530813
            Val objective improved 0.5568 → 0.5506, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 4.3407, mae: 0.7551, huber: 0.5275, swd: 3.7514, ept: 183.7771
    Epoch [10/50], Val Losses: mse: 4.7880, mae: 0.7689, huber: 0.5460, swd: 4.1457, ept: 182.8691
    Epoch [10/50], Test Losses: mse: 4.4195, mae: 0.7961, huber: 0.5648, swd: 3.8197, ept: 182.5770
      Epoch 10 composite train-obj: 0.527463
            Val objective improved 0.5506 → 0.5460, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 4.3055, mae: 0.7491, huber: 0.5229, swd: 3.7191, ept: 183.8736
    Epoch [11/50], Val Losses: mse: 4.7583, mae: 0.7683, huber: 0.5437, swd: 4.0978, ept: 182.9337
    Epoch [11/50], Test Losses: mse: 4.3889, mae: 0.7945, huber: 0.5616, swd: 3.7700, ept: 182.6549
      Epoch 11 composite train-obj: 0.522878
            Val objective improved 0.5460 → 0.5437, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 4.2667, mae: 0.7418, huber: 0.5170, swd: 3.6844, ept: 184.0043
    Epoch [12/50], Val Losses: mse: 4.8168, mae: 0.7545, huber: 0.5389, swd: 4.1819, ept: 182.9245
    Epoch [12/50], Test Losses: mse: 4.4492, mae: 0.7818, huber: 0.5581, swd: 3.8592, ept: 182.6928
      Epoch 12 composite train-obj: 0.517039
            Val objective improved 0.5437 → 0.5389, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 4.2803, mae: 0.7407, huber: 0.5169, swd: 3.6962, ept: 184.0168
    Epoch [13/50], Val Losses: mse: 4.6880, mae: 0.7665, huber: 0.5378, swd: 4.0315, ept: 183.2458
    Epoch [13/50], Test Losses: mse: 4.3213, mae: 0.7912, huber: 0.5547, swd: 3.7041, ept: 182.8897
      Epoch 13 composite train-obj: 0.516925
            Val objective improved 0.5389 → 0.5378, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 4.2299, mae: 0.7331, huber: 0.5104, swd: 3.6520, ept: 184.1932
    Epoch [14/50], Val Losses: mse: 4.6750, mae: 0.7542, huber: 0.5334, swd: 4.0419, ept: 183.3231
    Epoch [14/50], Test Losses: mse: 4.3062, mae: 0.7783, huber: 0.5495, swd: 3.7142, ept: 182.9871
      Epoch 14 composite train-obj: 0.510405
            Val objective improved 0.5378 → 0.5334, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 4.2317, mae: 0.7305, huber: 0.5089, swd: 3.6551, ept: 184.2490
    Epoch [15/50], Val Losses: mse: 4.7488, mae: 0.7503, huber: 0.5305, swd: 4.1212, ept: 183.2948
    Epoch [15/50], Test Losses: mse: 4.3810, mae: 0.7756, huber: 0.5480, swd: 3.7981, ept: 182.9722
      Epoch 15 composite train-obj: 0.508933
            Val objective improved 0.5334 → 0.5305, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 4.2155, mae: 0.7277, huber: 0.5065, swd: 3.6400, ept: 184.2902
    Epoch [16/50], Val Losses: mse: 4.7242, mae: 0.7500, huber: 0.5284, swd: 4.1068, ept: 183.4407
    Epoch [16/50], Test Losses: mse: 4.3564, mae: 0.7753, huber: 0.5456, swd: 3.7825, ept: 183.1286
      Epoch 16 composite train-obj: 0.506538
            Val objective improved 0.5305 → 0.5284, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 4.2159, mae: 0.7246, huber: 0.5043, swd: 3.6391, ept: 184.3421
    Epoch [17/50], Val Losses: mse: 4.6719, mae: 0.7523, huber: 0.5268, swd: 4.0566, ept: 183.5984
    Epoch [17/50], Test Losses: mse: 4.3073, mae: 0.7772, huber: 0.5438, swd: 3.7316, ept: 183.2450
      Epoch 17 composite train-obj: 0.504305
            Val objective improved 0.5284 → 0.5268, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 4.2135, mae: 0.7210, huber: 0.5022, swd: 3.6396, ept: 184.3939
    Epoch [18/50], Val Losses: mse: 4.6934, mae: 0.7506, huber: 0.5255, swd: 4.0793, ept: 183.6729
    Epoch [18/50], Test Losses: mse: 4.3272, mae: 0.7763, huber: 0.5424, swd: 3.7528, ept: 183.2998
      Epoch 18 composite train-obj: 0.502152
            Val objective improved 0.5268 → 0.5255, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 4.1875, mae: 0.7180, huber: 0.4991, swd: 3.6200, ept: 184.4917
    Epoch [19/50], Val Losses: mse: 4.7267, mae: 0.7410, huber: 0.5215, swd: 4.0951, ept: 183.6297
    Epoch [19/50], Test Losses: mse: 4.3590, mae: 0.7667, huber: 0.5392, swd: 3.7689, ept: 183.1684
      Epoch 19 composite train-obj: 0.499117
            Val objective improved 0.5255 → 0.5215, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 4.1931, mae: 0.7164, huber: 0.4982, swd: 3.6240, ept: 184.5278
    Epoch [20/50], Val Losses: mse: 4.6987, mae: 0.7648, huber: 0.5308, swd: 4.0840, ept: 183.8360
    Epoch [20/50], Test Losses: mse: 4.3370, mae: 0.7904, huber: 0.5483, swd: 3.7605, ept: 183.4785
      Epoch 20 composite train-obj: 0.498176
            No improvement (0.5308), counter 1/5
    Epoch [21/50], Train Losses: mse: 4.1893, mae: 0.7159, huber: 0.4973, swd: 3.6199, ept: 184.5495
    Epoch [21/50], Val Losses: mse: 4.6998, mae: 0.7583, huber: 0.5241, swd: 4.0807, ept: 183.8896
    Epoch [21/50], Test Losses: mse: 4.3341, mae: 0.7850, huber: 0.5414, swd: 3.7537, ept: 183.5087
      Epoch 21 composite train-obj: 0.497297
            No improvement (0.5241), counter 2/5
    Epoch [22/50], Train Losses: mse: 4.2090, mae: 0.7150, huber: 0.4971, swd: 3.6430, ept: 184.5708
    Epoch [22/50], Val Losses: mse: 4.6123, mae: 0.7457, huber: 0.5200, swd: 3.9886, ept: 183.8988
    Epoch [22/50], Test Losses: mse: 4.2476, mae: 0.7689, huber: 0.5356, swd: 3.6638, ept: 183.3822
      Epoch 22 composite train-obj: 0.497115
            Val objective improved 0.5215 → 0.5200, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 4.1942, mae: 0.7153, huber: 0.4972, swd: 3.6244, ept: 184.5864
    Epoch [23/50], Val Losses: mse: 4.5836, mae: 0.7446, huber: 0.5208, swd: 3.9579, ept: 183.9522
    Epoch [23/50], Test Losses: mse: 4.2205, mae: 0.7666, huber: 0.5363, swd: 3.6332, ept: 183.4473
      Epoch 23 composite train-obj: 0.497170
            No improvement (0.5208), counter 1/5
    Epoch [24/50], Train Losses: mse: 4.1672, mae: 0.7097, huber: 0.4924, swd: 3.6049, ept: 184.6964
    Epoch [24/50], Val Losses: mse: 4.6570, mae: 0.7268, huber: 0.5143, swd: 4.0294, ept: 183.8669
    Epoch [24/50], Test Losses: mse: 4.2892, mae: 0.7511, huber: 0.5311, swd: 3.7026, ept: 183.3769
      Epoch 24 composite train-obj: 0.492421
            Val objective improved 0.5200 → 0.5143, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 4.1565, mae: 0.7055, huber: 0.4901, swd: 3.5936, ept: 184.6979
    Epoch [25/50], Val Losses: mse: 4.6765, mae: 0.7258, huber: 0.5137, swd: 4.0732, ept: 183.8383
    Epoch [25/50], Test Losses: mse: 4.3072, mae: 0.7487, huber: 0.5300, swd: 3.7484, ept: 183.4522
      Epoch 25 composite train-obj: 0.490066
            Val objective improved 0.5143 → 0.5137, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 4.1666, mae: 0.7053, huber: 0.4902, swd: 3.6025, ept: 184.7168
    Epoch [26/50], Val Losses: mse: 4.6060, mae: 0.7319, huber: 0.5123, swd: 3.9848, ept: 183.9802
    Epoch [26/50], Test Losses: mse: 4.2376, mae: 0.7528, huber: 0.5271, swd: 3.6568, ept: 183.5387
      Epoch 26 composite train-obj: 0.490211
            Val objective improved 0.5137 → 0.5123, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 4.1870, mae: 0.7063, huber: 0.4912, swd: 3.6241, ept: 184.6926
    Epoch [27/50], Val Losses: mse: 4.5341, mae: 0.7478, huber: 0.5181, swd: 3.9049, ept: 184.0522
    Epoch [27/50], Test Losses: mse: 4.1660, mae: 0.7666, huber: 0.5312, swd: 3.5774, ept: 183.6145
      Epoch 27 composite train-obj: 0.491156
            No improvement (0.5181), counter 1/5
    Epoch [28/50], Train Losses: mse: 4.1425, mae: 0.7034, huber: 0.4875, swd: 3.5828, ept: 184.8051
    Epoch [28/50], Val Losses: mse: 4.6703, mae: 0.7193, huber: 0.5109, swd: 4.0511, ept: 183.8772
    Epoch [28/50], Test Losses: mse: 4.2993, mae: 0.7404, huber: 0.5269, swd: 3.7246, ept: 183.4838
      Epoch 28 composite train-obj: 0.487507
            Val objective improved 0.5123 → 0.5109, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 4.1595, mae: 0.7015, huber: 0.4874, swd: 3.5956, ept: 184.7481
    Epoch [29/50], Val Losses: mse: 4.6149, mae: 0.7307, huber: 0.5119, swd: 4.0308, ept: 184.1333
    Epoch [29/50], Test Losses: mse: 4.2449, mae: 0.7525, huber: 0.5264, swd: 3.7030, ept: 183.8319
      Epoch 29 composite train-obj: 0.487358
            No improvement (0.5119), counter 1/5
    Epoch [30/50], Train Losses: mse: 4.1412, mae: 0.7008, huber: 0.4859, swd: 3.5810, ept: 184.8284
    Epoch [30/50], Val Losses: mse: 4.5836, mae: 0.7326, huber: 0.5139, swd: 3.9419, ept: 184.0813
    Epoch [30/50], Test Losses: mse: 4.2149, mae: 0.7528, huber: 0.5284, swd: 3.6116, ept: 183.5890
      Epoch 30 composite train-obj: 0.485891
            No improvement (0.5139), counter 2/5
    Epoch [31/50], Train Losses: mse: 4.1400, mae: 0.6983, huber: 0.4844, swd: 3.5809, ept: 184.8236
    Epoch [31/50], Val Losses: mse: 4.6671, mae: 0.7126, huber: 0.5061, swd: 4.0297, ept: 183.9249
    Epoch [31/50], Test Losses: mse: 4.2983, mae: 0.7360, huber: 0.5229, swd: 3.7036, ept: 183.4542
      Epoch 31 composite train-obj: 0.484422
            Val objective improved 0.5109 → 0.5061, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 4.1342, mae: 0.6970, huber: 0.4835, swd: 3.5740, ept: 184.8671
    Epoch [32/50], Val Losses: mse: 4.6914, mae: 0.7122, huber: 0.5069, swd: 4.0705, ept: 183.9154
    Epoch [32/50], Test Losses: mse: 4.3210, mae: 0.7357, huber: 0.5238, swd: 3.7440, ept: 183.4937
      Epoch 32 composite train-obj: 0.483520
            No improvement (0.5069), counter 1/5
    Epoch [33/50], Train Losses: mse: 4.1401, mae: 0.6966, huber: 0.4833, swd: 3.5800, ept: 184.8518
    Epoch [33/50], Val Losses: mse: 4.6758, mae: 0.7120, huber: 0.5067, swd: 4.0461, ept: 183.9104
    Epoch [33/50], Test Losses: mse: 4.3060, mae: 0.7345, huber: 0.5236, swd: 3.7203, ept: 183.4729
      Epoch 33 composite train-obj: 0.483293
            No improvement (0.5067), counter 2/5
    Epoch [34/50], Train Losses: mse: 4.1478, mae: 0.6969, huber: 0.4839, swd: 3.5842, ept: 184.8512
    Epoch [34/50], Val Losses: mse: 4.5961, mae: 0.7283, huber: 0.5068, swd: 3.9554, ept: 184.0983
    Epoch [34/50], Test Losses: mse: 4.2270, mae: 0.7493, huber: 0.5220, swd: 3.6248, ept: 183.6270
      Epoch 34 composite train-obj: 0.483896
            No improvement (0.5068), counter 3/5
    Epoch [35/50], Train Losses: mse: 4.1263, mae: 0.6946, huber: 0.4814, swd: 3.5647, ept: 184.9226
    Epoch [35/50], Val Losses: mse: 4.6038, mae: 0.7222, huber: 0.5056, swd: 4.0084, ept: 184.1350
    Epoch [35/50], Test Losses: mse: 4.2359, mae: 0.7439, huber: 0.5204, swd: 3.6829, ept: 183.7958
      Epoch 35 composite train-obj: 0.481365
            Val objective improved 0.5061 → 0.5056, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 4.1411, mae: 0.6942, huber: 0.4819, swd: 3.5790, ept: 184.8839
    Epoch [36/50], Val Losses: mse: 4.6437, mae: 0.7210, huber: 0.5058, swd: 4.0410, ept: 184.1124
    Epoch [36/50], Test Losses: mse: 4.2756, mae: 0.7439, huber: 0.5215, swd: 3.7164, ept: 183.7495
      Epoch 36 composite train-obj: 0.481919
            No improvement (0.5058), counter 1/5
    Epoch [37/50], Train Losses: mse: 4.1242, mae: 0.6932, huber: 0.4804, swd: 3.5635, ept: 184.8973
    Epoch [37/50], Val Losses: mse: 4.6165, mae: 0.7234, huber: 0.5061, swd: 4.0204, ept: 184.0924
    Epoch [37/50], Test Losses: mse: 4.2474, mae: 0.7461, huber: 0.5214, swd: 3.6952, ept: 183.7013
      Epoch 37 composite train-obj: 0.480398
            No improvement (0.5061), counter 2/5
    Epoch [38/50], Train Losses: mse: 4.1410, mae: 0.6946, huber: 0.4818, swd: 3.5809, ept: 184.8946
    Epoch [38/50], Val Losses: mse: 4.5753, mae: 0.7129, huber: 0.5014, swd: 3.9663, ept: 184.1778
    Epoch [38/50], Test Losses: mse: 4.2047, mae: 0.7337, huber: 0.5158, swd: 3.6384, ept: 183.7672
      Epoch 38 composite train-obj: 0.481828
            Val objective improved 0.5056 → 0.5014, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 4.1228, mae: 0.6918, huber: 0.4796, swd: 3.5621, ept: 184.9398
    Epoch [39/50], Val Losses: mse: 4.5807, mae: 0.7133, huber: 0.5012, swd: 3.9679, ept: 184.1685
    Epoch [39/50], Test Losses: mse: 4.2123, mae: 0.7340, huber: 0.5161, swd: 3.6425, ept: 183.7575
      Epoch 39 composite train-obj: 0.479574
            Val objective improved 0.5014 → 0.5012, saving checkpoint.
    Epoch [40/50], Train Losses: mse: 4.1319, mae: 0.6925, huber: 0.4805, swd: 3.5680, ept: 184.8852
    Epoch [40/50], Val Losses: mse: 4.5637, mae: 0.7298, huber: 0.5042, swd: 3.9274, ept: 184.2837
    Epoch [40/50], Test Losses: mse: 4.1988, mae: 0.7527, huber: 0.5200, swd: 3.5984, ept: 183.8218
      Epoch 40 composite train-obj: 0.480516
            No improvement (0.5042), counter 1/5
    Epoch [41/50], Train Losses: mse: 4.1121, mae: 0.6900, huber: 0.4783, swd: 3.5507, ept: 184.9755
    Epoch [41/50], Val Losses: mse: 4.5429, mae: 0.7282, huber: 0.5045, swd: 3.9049, ept: 184.3086
    Epoch [41/50], Test Losses: mse: 4.1771, mae: 0.7488, huber: 0.5192, swd: 3.5751, ept: 183.8181
      Epoch 41 composite train-obj: 0.478269
            No improvement (0.5045), counter 2/5
    Epoch [42/50], Train Losses: mse: 4.1243, mae: 0.6907, huber: 0.4792, swd: 3.5618, ept: 184.9258
    Epoch [42/50], Val Losses: mse: 4.5643, mae: 0.7271, huber: 0.5064, swd: 3.9768, ept: 184.3354
    Epoch [42/50], Test Losses: mse: 4.1987, mae: 0.7487, huber: 0.5210, swd: 3.6513, ept: 183.9902
      Epoch 42 composite train-obj: 0.479179
            No improvement (0.5064), counter 3/5
    Epoch [43/50], Train Losses: mse: 4.1073, mae: 0.6893, huber: 0.4775, swd: 3.5468, ept: 184.9848
    Epoch [43/50], Val Losses: mse: 4.6419, mae: 0.7028, huber: 0.4998, swd: 4.0018, ept: 184.0879
    Epoch [43/50], Test Losses: mse: 4.2743, mae: 0.7251, huber: 0.5167, swd: 3.6749, ept: 183.6128
      Epoch 43 composite train-obj: 0.477530
            Val objective improved 0.5012 → 0.4998, saving checkpoint.
    Epoch [44/50], Train Losses: mse: 4.1161, mae: 0.6880, huber: 0.4774, swd: 3.5529, ept: 184.9644
    Epoch [44/50], Val Losses: mse: 4.5914, mae: 0.7106, huber: 0.4979, swd: 3.9797, ept: 184.2052
    Epoch [44/50], Test Losses: mse: 4.2235, mae: 0.7323, huber: 0.5130, swd: 3.6538, ept: 183.8068
      Epoch 44 composite train-obj: 0.477371
            Val objective improved 0.4998 → 0.4979, saving checkpoint.
    Epoch [45/50], Train Losses: mse: 4.1142, mae: 0.6862, huber: 0.4760, swd: 3.5530, ept: 184.9898
    Epoch [45/50], Val Losses: mse: 4.5860, mae: 0.7058, huber: 0.4996, swd: 3.9527, ept: 184.1486
    Epoch [45/50], Test Losses: mse: 4.2184, mae: 0.7260, huber: 0.5150, swd: 3.6273, ept: 183.6619
      Epoch 45 composite train-obj: 0.476007
            No improvement (0.4996), counter 1/5
    Epoch [46/50], Train Losses: mse: 4.1109, mae: 0.6861, huber: 0.4758, swd: 3.5489, ept: 184.9906
    Epoch [46/50], Val Losses: mse: 4.5742, mae: 0.7056, huber: 0.4969, swd: 3.9650, ept: 184.2726
    Epoch [46/50], Test Losses: mse: 4.2061, mae: 0.7271, huber: 0.5119, swd: 3.6390, ept: 183.8567
      Epoch 46 composite train-obj: 0.475833
            Val objective improved 0.4979 → 0.4969, saving checkpoint.
    Epoch [47/50], Train Losses: mse: 4.1263, mae: 0.6883, huber: 0.4775, swd: 3.5623, ept: 184.9813
    Epoch [47/50], Val Losses: mse: 4.5726, mae: 0.7064, huber: 0.4973, swd: 3.9698, ept: 184.3092
    Epoch [47/50], Test Losses: mse: 4.2063, mae: 0.7279, huber: 0.5126, swd: 3.6458, ept: 183.8586
      Epoch 47 composite train-obj: 0.477516
            No improvement (0.4973), counter 1/5
    Epoch [48/50], Train Losses: mse: 4.1107, mae: 0.6862, huber: 0.4757, swd: 3.5496, ept: 185.0194
    Epoch [48/50], Val Losses: mse: 4.5292, mae: 0.7037, huber: 0.4966, swd: 3.9289, ept: 184.3225
    Epoch [48/50], Test Losses: mse: 4.1601, mae: 0.7233, huber: 0.5104, swd: 3.6016, ept: 183.9577
      Epoch 48 composite train-obj: 0.475724
            Val objective improved 0.4969 → 0.4966, saving checkpoint.
    Epoch [49/50], Train Losses: mse: 4.1145, mae: 0.6855, huber: 0.4754, swd: 3.5532, ept: 185.0301
    Epoch [49/50], Val Losses: mse: 4.5835, mae: 0.7040, huber: 0.4956, swd: 3.9695, ept: 184.3129
    Epoch [49/50], Test Losses: mse: 4.2169, mae: 0.7271, huber: 0.5113, swd: 3.6444, ept: 183.8865
      Epoch 49 composite train-obj: 0.475371
            Val objective improved 0.4966 → 0.4956, saving checkpoint.
    Epoch [50/50], Train Losses: mse: 4.0956, mae: 0.6842, huber: 0.4740, swd: 3.5355, ept: 185.0593
    Epoch [50/50], Val Losses: mse: 4.5800, mae: 0.7021, huber: 0.4978, swd: 3.9331, ept: 184.2580
    Epoch [50/50], Test Losses: mse: 4.2154, mae: 0.7234, huber: 0.5140, swd: 3.6075, ept: 183.7479
      Epoch 50 composite train-obj: 0.474027
            No improvement (0.4978), counter 1/5
    Epoch [50/50], Test Losses: mse: 4.2169, mae: 0.7271, huber: 0.5113, swd: 3.6444, ept: 183.8865
    Best round's Test MSE: 4.2169, MAE: 0.7271, SWD: 3.6444
    Best round's Validation MSE: 4.5835, MAE: 0.7040, SWD: 3.9695
    Best round's Test verification MSE : 4.2169, MAE: 0.7271, SWD: 3.6444
    Time taken: 118.54 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.4697, mae: 1.1505, huber: 0.8763, swd: 4.5863, ept: 168.5136
    Epoch [1/50], Val Losses: mse: 5.8133, mae: 0.9140, huber: 0.6639, swd: 4.5131, ept: 179.5492
    Epoch [1/50], Test Losses: mse: 5.3828, mae: 0.9535, huber: 0.6911, swd: 4.1697, ept: 179.1283
      Epoch 1 composite train-obj: 0.876304
            Val objective improved inf → 0.6639, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.0667, mae: 0.8715, huber: 0.6196, swd: 3.9258, ept: 181.3558
    Epoch [2/50], Val Losses: mse: 5.5253, mae: 0.8749, huber: 0.6314, swd: 4.2777, ept: 180.5878
    Epoch [2/50], Test Losses: mse: 5.1149, mae: 0.9114, huber: 0.6564, swd: 3.9510, ept: 180.1332
      Epoch 2 composite train-obj: 0.619611
            Val objective improved 0.6639 → 0.6314, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.8335, mae: 0.8414, huber: 0.5949, swd: 3.7420, ept: 182.0797
    Epoch [3/50], Val Losses: mse: 5.1850, mae: 0.8630, huber: 0.6116, swd: 4.0409, ept: 181.3078
    Epoch [3/50], Test Losses: mse: 4.7873, mae: 0.8942, huber: 0.6316, swd: 3.7253, ept: 181.2150
      Epoch 3 composite train-obj: 0.594850
            Val objective improved 0.6314 → 0.6116, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.6719, mae: 0.8190, huber: 0.5769, swd: 3.6188, ept: 182.5423
    Epoch [4/50], Val Losses: mse: 5.0778, mae: 0.8314, huber: 0.5927, swd: 3.9452, ept: 181.6274
    Epoch [4/50], Test Losses: mse: 4.6890, mae: 0.8619, huber: 0.6130, swd: 3.6355, ept: 181.4824
      Epoch 4 composite train-obj: 0.576854
            Val objective improved 0.6116 → 0.5927, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.5536, mae: 0.8000, huber: 0.5617, swd: 3.5282, ept: 182.9156
    Epoch [5/50], Val Losses: mse: 5.0713, mae: 0.8194, huber: 0.5804, swd: 3.9380, ept: 181.8601
    Epoch [5/50], Test Losses: mse: 4.6881, mae: 0.8503, huber: 0.6019, swd: 3.6341, ept: 181.6581
      Epoch 5 composite train-obj: 0.561703
            Val objective improved 0.5927 → 0.5804, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.4822, mae: 0.7862, huber: 0.5510, swd: 3.4741, ept: 183.1648
    Epoch [6/50], Val Losses: mse: 4.9565, mae: 0.8132, huber: 0.5742, swd: 3.8672, ept: 182.2748
    Epoch [6/50], Test Losses: mse: 4.5820, mae: 0.8427, huber: 0.5941, swd: 3.5707, ept: 182.1285
      Epoch 6 composite train-obj: 0.551036
            Val objective improved 0.5804 → 0.5742, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.4187, mae: 0.7775, huber: 0.5438, swd: 3.4241, ept: 183.4085
    Epoch [7/50], Val Losses: mse: 4.9347, mae: 0.7941, huber: 0.5663, swd: 3.7919, ept: 182.2976
    Epoch [7/50], Test Losses: mse: 4.5614, mae: 0.8230, huber: 0.5871, swd: 3.4924, ept: 181.9982
      Epoch 7 composite train-obj: 0.543766
            Val objective improved 0.5742 → 0.5663, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 4.3984, mae: 0.7693, huber: 0.5383, swd: 3.4072, ept: 183.4882
    Epoch [8/50], Val Losses: mse: 4.8925, mae: 0.7861, huber: 0.5573, swd: 3.7871, ept: 182.5142
    Epoch [8/50], Test Losses: mse: 4.5209, mae: 0.8159, huber: 0.5774, swd: 3.4917, ept: 182.1979
      Epoch 8 composite train-obj: 0.538293
            Val objective improved 0.5663 → 0.5573, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 4.3729, mae: 0.7616, huber: 0.5327, swd: 3.3925, ept: 183.6490
    Epoch [9/50], Val Losses: mse: 4.7508, mae: 0.7911, huber: 0.5562, swd: 3.6831, ept: 182.7042
    Epoch [9/50], Test Losses: mse: 4.3817, mae: 0.8171, huber: 0.5738, swd: 3.3892, ept: 182.4936
      Epoch 9 composite train-obj: 0.532732
            Val objective improved 0.5573 → 0.5562, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 4.3178, mae: 0.7553, huber: 0.5272, swd: 3.3449, ept: 183.7894
    Epoch [10/50], Val Losses: mse: 4.7888, mae: 0.7732, huber: 0.5471, swd: 3.7116, ept: 182.7725
    Epoch [10/50], Test Losses: mse: 4.4196, mae: 0.8000, huber: 0.5655, swd: 3.4183, ept: 182.4550
      Epoch 10 composite train-obj: 0.527235
            Val objective improved 0.5562 → 0.5471, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 4.2815, mae: 0.7476, huber: 0.5215, swd: 3.3192, ept: 183.9031
    Epoch [11/50], Val Losses: mse: 4.7439, mae: 0.7689, huber: 0.5437, swd: 3.6707, ept: 182.9331
    Epoch [11/50], Test Losses: mse: 4.3757, mae: 0.7946, huber: 0.5613, swd: 3.3764, ept: 182.6438
      Epoch 11 composite train-obj: 0.521455
            Val objective improved 0.5471 → 0.5437, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 4.2621, mae: 0.7408, huber: 0.5166, swd: 3.3051, ept: 183.9861
    Epoch [12/50], Val Losses: mse: 4.7364, mae: 0.7641, huber: 0.5390, swd: 3.6939, ept: 183.0803
    Epoch [12/50], Test Losses: mse: 4.3690, mae: 0.7886, huber: 0.5563, swd: 3.4034, ept: 182.8820
      Epoch 12 composite train-obj: 0.516573
            Val objective improved 0.5437 → 0.5390, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 4.2541, mae: 0.7389, huber: 0.5151, swd: 3.2977, ept: 184.0662
    Epoch [13/50], Val Losses: mse: 4.7280, mae: 0.7538, huber: 0.5352, swd: 3.6613, ept: 183.0966
    Epoch [13/50], Test Losses: mse: 4.3620, mae: 0.7784, huber: 0.5530, swd: 3.3688, ept: 182.7214
      Epoch 13 composite train-obj: 0.515088
            Val objective improved 0.5390 → 0.5352, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 4.2312, mae: 0.7329, huber: 0.5105, swd: 3.2820, ept: 184.1559
    Epoch [14/50], Val Losses: mse: 4.7156, mae: 0.7489, huber: 0.5323, swd: 3.6723, ept: 183.1936
    Epoch [14/50], Test Losses: mse: 4.3480, mae: 0.7739, huber: 0.5496, swd: 3.3804, ept: 182.8960
      Epoch 14 composite train-obj: 0.510542
            Val objective improved 0.5352 → 0.5323, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 4.2228, mae: 0.7299, huber: 0.5085, swd: 3.2755, ept: 184.2390
    Epoch [15/50], Val Losses: mse: 4.7019, mae: 0.7492, huber: 0.5303, swd: 3.6375, ept: 183.3001
    Epoch [15/50], Test Losses: mse: 4.3358, mae: 0.7742, huber: 0.5480, swd: 3.3450, ept: 182.9435
      Epoch 15 composite train-obj: 0.508548
            Val objective improved 0.5323 → 0.5303, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 4.2213, mae: 0.7274, huber: 0.5068, swd: 3.2748, ept: 184.2774
    Epoch [16/50], Val Losses: mse: 4.6531, mae: 0.7504, huber: 0.5278, swd: 3.6140, ept: 183.5844
    Epoch [16/50], Test Losses: mse: 4.2864, mae: 0.7739, huber: 0.5440, swd: 3.3198, ept: 183.1755
      Epoch 16 composite train-obj: 0.506794
            Val objective improved 0.5303 → 0.5278, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 4.2131, mae: 0.7245, huber: 0.5044, swd: 3.2724, ept: 184.3676
    Epoch [17/50], Val Losses: mse: 4.6565, mae: 0.7520, huber: 0.5279, swd: 3.5926, ept: 183.5853
    Epoch [17/50], Test Losses: mse: 4.2890, mae: 0.7744, huber: 0.5441, swd: 3.2961, ept: 183.1780
      Epoch 17 composite train-obj: 0.504373
            No improvement (0.5279), counter 1/5
    Epoch [18/50], Train Losses: mse: 4.1927, mae: 0.7201, huber: 0.5010, swd: 3.2561, ept: 184.4302
    Epoch [18/50], Val Losses: mse: 4.6629, mae: 0.7438, huber: 0.5231, swd: 3.6424, ept: 183.6351
    Epoch [18/50], Test Losses: mse: 4.2962, mae: 0.7670, huber: 0.5394, swd: 3.3505, ept: 183.2818
      Epoch 18 composite train-obj: 0.501040
            Val objective improved 0.5278 → 0.5231, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 4.1908, mae: 0.7187, huber: 0.4995, swd: 3.2571, ept: 184.4959
    Epoch [19/50], Val Losses: mse: 4.6963, mae: 0.7376, huber: 0.5226, swd: 3.6495, ept: 183.5163
    Epoch [19/50], Test Losses: mse: 4.3286, mae: 0.7602, huber: 0.5394, swd: 3.3582, ept: 183.0334
      Epoch 19 composite train-obj: 0.499533
            Val objective improved 0.5231 → 0.5226, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 4.1909, mae: 0.7178, huber: 0.4992, swd: 3.2562, ept: 184.5010
    Epoch [20/50], Val Losses: mse: 4.7503, mae: 0.7359, huber: 0.5207, swd: 3.7039, ept: 183.5936
    Epoch [20/50], Test Losses: mse: 4.3829, mae: 0.7625, huber: 0.5390, swd: 3.4115, ept: 183.2752
      Epoch 20 composite train-obj: 0.499153
            Val objective improved 0.5226 → 0.5207, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 4.1944, mae: 0.7156, huber: 0.4977, swd: 3.2605, ept: 184.5471
    Epoch [21/50], Val Losses: mse: 4.6584, mae: 0.7337, huber: 0.5193, swd: 3.6107, ept: 183.6671
    Epoch [21/50], Test Losses: mse: 4.2898, mae: 0.7562, huber: 0.5355, swd: 3.3161, ept: 183.2330
      Epoch 21 composite train-obj: 0.497713
            Val objective improved 0.5207 → 0.5193, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 4.1763, mae: 0.7130, huber: 0.4953, swd: 3.2452, ept: 184.6116
    Epoch [22/50], Val Losses: mse: 4.6523, mae: 0.7383, huber: 0.5203, swd: 3.6523, ept: 183.7960
    Epoch [22/50], Test Losses: mse: 4.2839, mae: 0.7609, huber: 0.5357, swd: 3.3595, ept: 183.4738
      Epoch 22 composite train-obj: 0.495333
            No improvement (0.5203), counter 1/5
    Epoch [23/50], Train Losses: mse: 4.1944, mae: 0.7130, huber: 0.4959, swd: 3.2591, ept: 184.5704
    Epoch [23/50], Val Losses: mse: 4.6394, mae: 0.7409, huber: 0.5167, swd: 3.6304, ept: 183.8962
    Epoch [23/50], Test Losses: mse: 4.2727, mae: 0.7646, huber: 0.5324, swd: 3.3370, ept: 183.4967
      Epoch 23 composite train-obj: 0.495887
            Val objective improved 0.5193 → 0.5167, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 4.1589, mae: 0.7080, huber: 0.4910, swd: 3.2350, ept: 184.6714
    Epoch [24/50], Val Losses: mse: 4.6624, mae: 0.7345, huber: 0.5228, swd: 3.5913, ept: 183.8128
    Epoch [24/50], Test Losses: mse: 4.2962, mae: 0.7558, huber: 0.5393, swd: 3.2950, ept: 183.2768
      Epoch 24 composite train-obj: 0.491049
            No improvement (0.5228), counter 1/5
    Epoch [25/50], Train Losses: mse: 4.1700, mae: 0.7077, huber: 0.4916, swd: 3.2423, ept: 184.6976
    Epoch [25/50], Val Losses: mse: 4.6400, mae: 0.7258, huber: 0.5118, swd: 3.6212, ept: 183.9244
    Epoch [25/50], Test Losses: mse: 4.2708, mae: 0.7484, huber: 0.5275, swd: 3.3262, ept: 183.4872
      Epoch 25 composite train-obj: 0.491562
            Val objective improved 0.5167 → 0.5118, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 4.1781, mae: 0.7072, huber: 0.4912, swd: 3.2489, ept: 184.6967
    Epoch [26/50], Val Losses: mse: 4.5892, mae: 0.7303, huber: 0.5115, swd: 3.5778, ept: 184.0016
    Epoch [26/50], Test Losses: mse: 4.2198, mae: 0.7517, huber: 0.5260, swd: 3.2821, ept: 183.5750
      Epoch 26 composite train-obj: 0.491241
            Val objective improved 0.5118 → 0.5115, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 4.1703, mae: 0.7048, huber: 0.4897, swd: 3.2440, ept: 184.7207
    Epoch [27/50], Val Losses: mse: 4.6198, mae: 0.7323, huber: 0.5117, swd: 3.6064, ept: 183.9977
    Epoch [27/50], Test Losses: mse: 4.2517, mae: 0.7549, huber: 0.5270, swd: 3.3126, ept: 183.6033
      Epoch 27 composite train-obj: 0.489679
            No improvement (0.5117), counter 1/5
    Epoch [28/50], Train Losses: mse: 4.1677, mae: 0.7052, huber: 0.4897, swd: 3.2427, ept: 184.7521
    Epoch [28/50], Val Losses: mse: 4.6462, mae: 0.7363, huber: 0.5160, swd: 3.5727, ept: 183.8797
    Epoch [28/50], Test Losses: mse: 4.2796, mae: 0.7571, huber: 0.5324, swd: 3.2779, ept: 183.3779
      Epoch 28 composite train-obj: 0.489655
            No improvement (0.5160), counter 2/5
    Epoch [29/50], Train Losses: mse: 4.1522, mae: 0.7026, huber: 0.4873, swd: 3.2302, ept: 184.8172
    Epoch [29/50], Val Losses: mse: 4.6645, mae: 0.7224, huber: 0.5105, swd: 3.6567, ept: 183.9199
    Epoch [29/50], Test Losses: mse: 4.2950, mae: 0.7452, huber: 0.5265, swd: 3.3643, ept: 183.5698
      Epoch 29 composite train-obj: 0.487254
            Val objective improved 0.5115 → 0.5105, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 4.1516, mae: 0.7012, huber: 0.4864, swd: 3.2304, ept: 184.8152
    Epoch [30/50], Val Losses: mse: 4.6561, mae: 0.7146, huber: 0.5072, swd: 3.6456, ept: 183.9221
    Epoch [30/50], Test Losses: mse: 4.2883, mae: 0.7381, huber: 0.5236, swd: 3.3526, ept: 183.5310
      Epoch 30 composite train-obj: 0.486440
            Val objective improved 0.5105 → 0.5072, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 4.1575, mae: 0.6991, huber: 0.4856, swd: 3.2340, ept: 184.7942
    Epoch [31/50], Val Losses: mse: 4.6808, mae: 0.7153, huber: 0.5071, swd: 3.6497, ept: 183.9663
    Epoch [31/50], Test Losses: mse: 4.3115, mae: 0.7392, huber: 0.5238, swd: 3.3548, ept: 183.4721
      Epoch 31 composite train-obj: 0.485592
            Val objective improved 0.5072 → 0.5071, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 4.1574, mae: 0.6988, huber: 0.4855, swd: 3.2333, ept: 184.7958
    Epoch [32/50], Val Losses: mse: 4.6298, mae: 0.7211, huber: 0.5066, swd: 3.6022, ept: 183.9847
    Epoch [32/50], Test Losses: mse: 4.2639, mae: 0.7437, huber: 0.5229, swd: 3.3107, ept: 183.4369
      Epoch 32 composite train-obj: 0.485474
            Val objective improved 0.5071 → 0.5066, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 4.1494, mae: 0.6984, huber: 0.4846, swd: 3.2275, ept: 184.8233
    Epoch [33/50], Val Losses: mse: 4.5599, mae: 0.7323, huber: 0.5119, swd: 3.5243, ept: 184.0849
    Epoch [33/50], Test Losses: mse: 4.1935, mae: 0.7516, huber: 0.5265, swd: 3.2292, ept: 183.6320
      Epoch 33 composite train-obj: 0.484595
            No improvement (0.5119), counter 1/5
    Epoch [34/50], Train Losses: mse: 4.1415, mae: 0.6972, huber: 0.4838, swd: 3.2208, ept: 184.8357
    Epoch [34/50], Val Losses: mse: 4.6051, mae: 0.7449, huber: 0.5159, swd: 3.6284, ept: 184.3279
    Epoch [34/50], Test Losses: mse: 4.2403, mae: 0.7670, huber: 0.5303, swd: 3.3361, ept: 183.9780
      Epoch 34 composite train-obj: 0.483847
            No improvement (0.5159), counter 2/5
    Epoch [35/50], Train Losses: mse: 4.1516, mae: 0.6983, huber: 0.4846, swd: 3.2309, ept: 184.8352
    Epoch [35/50], Val Losses: mse: 4.5977, mae: 0.7159, huber: 0.5035, swd: 3.5960, ept: 184.1037
    Epoch [35/50], Test Losses: mse: 4.2281, mae: 0.7379, huber: 0.5185, swd: 3.3006, ept: 183.6885
      Epoch 35 composite train-obj: 0.484563
            Val objective improved 0.5066 → 0.5035, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 4.1176, mae: 0.6937, huber: 0.4806, swd: 3.2039, ept: 184.9465
    Epoch [36/50], Val Losses: mse: 4.5889, mae: 0.7134, huber: 0.5040, swd: 3.5662, ept: 184.0942
    Epoch [36/50], Test Losses: mse: 4.2207, mae: 0.7337, huber: 0.5190, swd: 3.2715, ept: 183.6020
      Epoch 36 composite train-obj: 0.480617
            No improvement (0.5040), counter 1/5
    Epoch [37/50], Train Losses: mse: 4.1305, mae: 0.6922, huber: 0.4806, swd: 3.2124, ept: 184.8897
    Epoch [37/50], Val Losses: mse: 4.6237, mae: 0.7102, huber: 0.5017, swd: 3.6161, ept: 184.0815
    Epoch [37/50], Test Losses: mse: 4.2559, mae: 0.7330, huber: 0.5176, swd: 3.3234, ept: 183.6932
      Epoch 37 composite train-obj: 0.480552
            Val objective improved 0.5035 → 0.5017, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 4.1375, mae: 0.6933, huber: 0.4810, swd: 3.2209, ept: 184.8978
    Epoch [38/50], Val Losses: mse: 4.5882, mae: 0.7114, huber: 0.5017, swd: 3.5826, ept: 184.1163
    Epoch [38/50], Test Losses: mse: 4.2191, mae: 0.7326, huber: 0.5167, swd: 3.2891, ept: 183.6671
      Epoch 38 composite train-obj: 0.481031
            No improvement (0.5017), counter 1/5
    Epoch [39/50], Train Losses: mse: 4.1401, mae: 0.6930, huber: 0.4811, swd: 3.2215, ept: 184.8810
    Epoch [39/50], Val Losses: mse: 4.5702, mae: 0.7202, huber: 0.5028, swd: 3.5748, ept: 184.2043
    Epoch [39/50], Test Losses: mse: 4.2013, mae: 0.7416, huber: 0.5171, swd: 3.2801, ept: 183.7733
      Epoch 39 composite train-obj: 0.481087
            No improvement (0.5028), counter 2/5
    Epoch [40/50], Train Losses: mse: 4.1104, mae: 0.6899, huber: 0.4781, swd: 3.1969, ept: 184.9637
    Epoch [40/50], Val Losses: mse: 4.6186, mae: 0.7086, huber: 0.5010, swd: 3.5975, ept: 184.1054
    Epoch [40/50], Test Losses: mse: 4.2498, mae: 0.7303, huber: 0.5167, swd: 3.3046, ept: 183.6673
      Epoch 40 composite train-obj: 0.478071
            Val objective improved 0.5017 → 0.5010, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 4.1162, mae: 0.6901, huber: 0.4785, swd: 3.2029, ept: 184.9915
    Epoch [41/50], Val Losses: mse: 4.6038, mae: 0.7153, huber: 0.5006, swd: 3.5857, ept: 184.2182
    Epoch [41/50], Test Losses: mse: 4.2356, mae: 0.7378, huber: 0.5162, swd: 3.2906, ept: 183.7392
      Epoch 41 composite train-obj: 0.478484
            Val objective improved 0.5010 → 0.5006, saving checkpoint.
    Epoch [42/50], Train Losses: mse: 4.1301, mae: 0.6910, huber: 0.4796, swd: 3.2121, ept: 184.9302
    Epoch [42/50], Val Losses: mse: 4.5652, mae: 0.7150, huber: 0.4993, swd: 3.5581, ept: 184.1802
    Epoch [42/50], Test Losses: mse: 4.1977, mae: 0.7351, huber: 0.5142, swd: 3.2646, ept: 183.7582
      Epoch 42 composite train-obj: 0.479636
            Val objective improved 0.5006 → 0.4993, saving checkpoint.
    Epoch [43/50], Train Losses: mse: 4.1449, mae: 0.6925, huber: 0.4802, swd: 3.2238, ept: 184.9002
    Epoch [43/50], Val Losses: mse: 4.5926, mae: 0.7245, huber: 0.5020, swd: 3.5621, ept: 184.1263
    Epoch [43/50], Test Losses: mse: 4.2300, mae: 0.7486, huber: 0.5187, swd: 3.2712, ept: 183.6080
      Epoch 43 composite train-obj: 0.480248
            No improvement (0.5020), counter 1/5
    Epoch [44/50], Train Losses: mse: 4.1247, mae: 0.6904, huber: 0.4787, swd: 3.2069, ept: 184.9458
    Epoch [44/50], Val Losses: mse: 4.5418, mae: 0.7194, huber: 0.4994, swd: 3.5495, ept: 184.3524
    Epoch [44/50], Test Losses: mse: 4.1751, mae: 0.7407, huber: 0.5137, swd: 3.2547, ept: 183.8841
      Epoch 44 composite train-obj: 0.478675
            No improvement (0.4994), counter 2/5
    Epoch [45/50], Train Losses: mse: 4.1146, mae: 0.6880, huber: 0.4773, swd: 3.2009, ept: 184.9570
    Epoch [45/50], Val Losses: mse: 4.5322, mae: 0.7162, huber: 0.4995, swd: 3.5252, ept: 184.2860
    Epoch [45/50], Test Losses: mse: 4.1666, mae: 0.7361, huber: 0.5141, swd: 3.2303, ept: 183.8194
      Epoch 45 composite train-obj: 0.477251
            No improvement (0.4995), counter 3/5
    Epoch [46/50], Train Losses: mse: 4.1215, mae: 0.6880, huber: 0.4776, swd: 3.2042, ept: 184.9519
    Epoch [46/50], Val Losses: mse: 4.5472, mae: 0.7284, huber: 0.5015, swd: 3.5439, ept: 184.3819
    Epoch [46/50], Test Losses: mse: 4.1822, mae: 0.7515, huber: 0.5165, swd: 3.2477, ept: 184.0037
      Epoch 46 composite train-obj: 0.477570
            No improvement (0.5015), counter 4/5
    Epoch [47/50], Train Losses: mse: 4.1076, mae: 0.6862, huber: 0.4757, swd: 3.1972, ept: 184.9993
    Epoch [47/50], Val Losses: mse: 4.5800, mae: 0.7062, huber: 0.4978, swd: 3.5678, ept: 184.2331
    Epoch [47/50], Test Losses: mse: 4.2164, mae: 0.7298, huber: 0.5139, swd: 3.2769, ept: 183.7641
      Epoch 47 composite train-obj: 0.475658
            Val objective improved 0.4993 → 0.4978, saving checkpoint.
    Epoch [48/50], Train Losses: mse: 4.1113, mae: 0.6865, huber: 0.4762, swd: 3.1991, ept: 184.9957
    Epoch [48/50], Val Losses: mse: 4.5762, mae: 0.7161, huber: 0.4982, swd: 3.5521, ept: 184.2819
    Epoch [48/50], Test Losses: mse: 4.2125, mae: 0.7388, huber: 0.5144, swd: 3.2581, ept: 183.8023
      Epoch 48 composite train-obj: 0.476155
            No improvement (0.4982), counter 1/5
    Epoch [49/50], Train Losses: mse: 4.0974, mae: 0.6856, huber: 0.4751, swd: 3.1859, ept: 185.0459
    Epoch [49/50], Val Losses: mse: 4.5692, mae: 0.7117, huber: 0.4966, swd: 3.5484, ept: 184.2791
    Epoch [49/50], Test Losses: mse: 4.2036, mae: 0.7335, huber: 0.5121, swd: 3.2539, ept: 183.8143
      Epoch 49 composite train-obj: 0.475103
            Val objective improved 0.4978 → 0.4966, saving checkpoint.
    Epoch [50/50], Train Losses: mse: 4.1028, mae: 0.6839, huber: 0.4744, swd: 3.1908, ept: 185.0445
    Epoch [50/50], Val Losses: mse: 4.5783, mae: 0.7025, huber: 0.4949, swd: 3.5681, ept: 184.2642
    Epoch [50/50], Test Losses: mse: 4.2113, mae: 0.7236, huber: 0.5103, swd: 3.2746, ept: 183.7891
      Epoch 50 composite train-obj: 0.474415
            Val objective improved 0.4966 → 0.4949, saving checkpoint.
    Epoch [50/50], Test Losses: mse: 4.2113, mae: 0.7236, huber: 0.5103, swd: 3.2746, ept: 183.7891
    Best round's Test MSE: 4.2113, MAE: 0.7236, SWD: 3.2746
    Best round's Validation MSE: 4.5783, MAE: 0.7025, SWD: 3.5681
    Best round's Test verification MSE : 4.2113, MAE: 0.7236, SWD: 3.2746
    Time taken: 118.60 seconds
    
    ==================================================
    Experiment Summary (DLinear_rossler_seq96_pred196_20250513_1344)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 4.2080 ± 0.0090
      mae: 0.7247 ± 0.0017
      huber: 0.5107 ± 0.0004
      swd: 3.5051 ± 0.1642
      ept: 183.8405 ± 0.0399
      count: 39.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 4.5749 ± 0.0088
      mae: 0.7028 ± 0.0009
      huber: 0.4954 ± 0.0003
      swd: 3.8186 ± 0.1784
      ept: 184.2957 ± 0.0223
      count: 39.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 355.02 seconds
    
    Experiment complete: DLinear_rossler_seq96_pred196_20250513_1344
    Model: DLinear
    Dataset: rossler
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
    channels=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 7.7159, mae: 1.3246, huber: 1.0267, swd: 4.9163, ept: 269.8240
    Epoch [1/50], Val Losses: mse: 7.6088, mae: 1.1857, huber: 0.8912, swd: 5.1828, ept: 289.0890
    Epoch [1/50], Test Losses: mse: 6.9015, mae: 1.2099, huber: 0.9065, swd: 4.6886, ept: 288.5865
      Epoch 1 composite train-obj: 1.026710
            Val objective improved inf → 0.8912, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.4620, mae: 1.1015, huber: 0.8138, swd: 4.3890, ept: 294.7720
    Epoch [2/50], Val Losses: mse: 7.3843, mae: 1.1411, huber: 0.8589, swd: 5.0283, ept: 290.5998
    Epoch [2/50], Test Losses: mse: 6.6874, mae: 1.1608, huber: 0.8703, swd: 4.5522, ept: 290.3326
      Epoch 2 composite train-obj: 0.813784
            Val objective improved 0.8912 → 0.8589, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.3142, mae: 1.0909, huber: 0.8046, swd: 4.2844, ept: 295.2313
    Epoch [3/50], Val Losses: mse: 6.7029, mae: 1.1720, huber: 0.8631, swd: 4.6267, ept: 292.4572
    Epoch [3/50], Test Losses: mse: 6.0340, mae: 1.1769, huber: 0.8597, swd: 4.1613, ept: 293.0834
      Epoch 3 composite train-obj: 0.804621
            No improvement (0.8631), counter 1/5
    Epoch [4/50], Train Losses: mse: 6.1041, mae: 1.0777, huber: 0.7920, swd: 4.1429, ept: 296.4491
    Epoch [4/50], Val Losses: mse: 6.5372, mae: 1.1408, huber: 0.8435, swd: 4.4737, ept: 293.0571
    Epoch [4/50], Test Losses: mse: 5.8922, mae: 1.1474, huber: 0.8410, swd: 4.0376, ept: 294.1214
      Epoch 4 composite train-obj: 0.791969
            Val objective improved 0.8589 → 0.8435, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.8738, mae: 1.0525, huber: 0.7694, swd: 3.9914, ept: 297.7837
    Epoch [5/50], Val Losses: mse: 6.6960, mae: 1.1014, huber: 0.8207, swd: 4.5785, ept: 292.9877
    Epoch [5/50], Test Losses: mse: 6.0410, mae: 1.1076, huber: 0.8204, swd: 4.1430, ept: 293.2410
      Epoch 5 composite train-obj: 0.769370
            Val objective improved 0.8435 → 0.8207, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.8530, mae: 1.0414, huber: 0.7608, swd: 3.9694, ept: 298.0416
    Epoch [6/50], Val Losses: mse: 6.8311, mae: 1.0886, huber: 0.8136, swd: 4.6341, ept: 293.3954
    Epoch [6/50], Test Losses: mse: 6.1840, mae: 1.1021, huber: 0.8195, swd: 4.2108, ept: 293.4336
      Epoch 6 composite train-obj: 0.760846
            Val objective improved 0.8207 → 0.8136, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.8322, mae: 1.0372, huber: 0.7587, swd: 3.9535, ept: 298.1841
    Epoch [7/50], Val Losses: mse: 6.5381, mae: 1.0914, huber: 0.8119, swd: 4.4941, ept: 293.8840
    Epoch [7/50], Test Losses: mse: 5.9015, mae: 1.0953, huber: 0.8089, swd: 4.0696, ept: 294.2230
      Epoch 7 composite train-obj: 0.758659
            Val objective improved 0.8136 → 0.8119, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.7561, mae: 1.0269, huber: 0.7495, swd: 3.9062, ept: 298.6614
    Epoch [8/50], Val Losses: mse: 6.6356, mae: 1.0766, huber: 0.8000, swd: 4.4867, ept: 294.0197
    Epoch [8/50], Test Losses: mse: 6.0042, mae: 1.0856, huber: 0.8029, swd: 4.0736, ept: 294.2252
      Epoch 8 composite train-obj: 0.749522
            Val objective improved 0.8119 → 0.8000, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 5.7365, mae: 1.0201, huber: 0.7445, swd: 3.8872, ept: 298.8550
    Epoch [9/50], Val Losses: mse: 6.4824, mae: 1.0895, huber: 0.8025, swd: 4.3757, ept: 294.4857
    Epoch [9/50], Test Losses: mse: 5.8569, mae: 1.0946, huber: 0.8022, swd: 3.9656, ept: 294.6356
      Epoch 9 composite train-obj: 0.744514
            No improvement (0.8025), counter 1/5
    Epoch [10/50], Train Losses: mse: 5.7155, mae: 1.0164, huber: 0.7416, swd: 3.8704, ept: 299.0567
    Epoch [10/50], Val Losses: mse: 6.4851, mae: 1.0701, huber: 0.7970, swd: 4.4151, ept: 294.3331
    Epoch [10/50], Test Losses: mse: 5.8603, mae: 1.0763, huber: 0.7963, swd: 4.0135, ept: 294.6679
      Epoch 10 composite train-obj: 0.741552
            Val objective improved 0.8000 → 0.7970, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 5.7000, mae: 1.0108, huber: 0.7375, swd: 3.8612, ept: 299.1495
    Epoch [11/50], Val Losses: mse: 6.6250, mae: 1.0657, huber: 0.7924, swd: 4.4695, ept: 294.5295
    Epoch [11/50], Test Losses: mse: 6.0048, mae: 1.0748, huber: 0.7958, swd: 4.0708, ept: 294.7678
      Epoch 11 composite train-obj: 0.737467
            Val objective improved 0.7970 → 0.7924, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 5.7492, mae: 1.0187, huber: 0.7452, swd: 3.9019, ept: 298.3350
    Epoch [12/50], Val Losses: mse: 6.3707, mae: 1.1426, huber: 0.8296, swd: 4.2393, ept: 295.0682
    Epoch [12/50], Test Losses: mse: 5.7709, mae: 1.1449, huber: 0.8269, swd: 3.8399, ept: 295.1503
      Epoch 12 composite train-obj: 0.745246
            No improvement (0.8296), counter 1/5
    Epoch [13/50], Train Losses: mse: 5.6485, mae: 1.0140, huber: 0.7382, swd: 3.8202, ept: 299.3667
    Epoch [13/50], Val Losses: mse: 6.7337, mae: 1.0631, huber: 0.7938, swd: 4.5243, ept: 294.3840
    Epoch [13/50], Test Losses: mse: 6.1152, mae: 1.0742, huber: 0.7998, swd: 4.1284, ept: 294.4227
      Epoch 13 composite train-obj: 0.738221
            No improvement (0.7938), counter 2/5
    Epoch [14/50], Train Losses: mse: 5.6563, mae: 1.0014, huber: 0.7301, swd: 3.8292, ept: 299.5250
    Epoch [14/50], Val Losses: mse: 6.5134, mae: 1.0489, huber: 0.7846, swd: 4.4018, ept: 294.8377
    Epoch [14/50], Test Losses: mse: 5.8995, mae: 1.0596, huber: 0.7876, swd: 4.0095, ept: 294.9958
      Epoch 14 composite train-obj: 0.730130
            Val objective improved 0.7924 → 0.7846, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 5.6291, mae: 0.9993, huber: 0.7283, swd: 3.8132, ept: 299.6550
    Epoch [15/50], Val Losses: mse: 6.8240, mae: 1.0581, huber: 0.7929, swd: 4.5404, ept: 294.5201
    Epoch [15/50], Test Losses: mse: 6.2096, mae: 1.0695, huber: 0.8014, swd: 4.1467, ept: 293.9424
      Epoch 15 composite train-obj: 0.728259
            No improvement (0.7929), counter 1/5
    Epoch [16/50], Train Losses: mse: 5.6488, mae: 1.0012, huber: 0.7305, swd: 3.8283, ept: 299.4018
    Epoch [16/50], Val Losses: mse: 6.3432, mae: 1.0645, huber: 0.7867, swd: 4.3098, ept: 295.2623
    Epoch [16/50], Test Losses: mse: 5.7332, mae: 1.0721, huber: 0.7859, swd: 3.9180, ept: 295.7905
      Epoch 16 composite train-obj: 0.730480
            No improvement (0.7867), counter 2/5
    Epoch [17/50], Train Losses: mse: 5.7471, mae: 1.0108, huber: 0.7392, swd: 3.8877, ept: 299.1095
    Epoch [17/50], Val Losses: mse: 6.1991, mae: 1.1081, huber: 0.8172, swd: 4.2876, ept: 296.0085
    Epoch [17/50], Test Losses: mse: 5.6057, mae: 1.1080, huber: 0.8100, swd: 3.8965, ept: 296.9160
      Epoch 17 composite train-obj: 0.739224
            No improvement (0.8172), counter 3/5
    Epoch [18/50], Train Losses: mse: 5.5932, mae: 0.9971, huber: 0.7265, swd: 3.7942, ept: 299.9443
    Epoch [18/50], Val Losses: mse: 6.5339, mae: 1.0424, huber: 0.7782, swd: 4.4191, ept: 295.0154
    Epoch [18/50], Test Losses: mse: 5.9212, mae: 1.0516, huber: 0.7813, swd: 4.0329, ept: 295.3979
      Epoch 18 composite train-obj: 0.726474
            Val objective improved 0.7846 → 0.7782, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 5.6116, mae: 0.9917, huber: 0.7225, swd: 3.8072, ept: 299.9629
    Epoch [19/50], Val Losses: mse: 6.5926, mae: 1.0406, huber: 0.7768, swd: 4.4375, ept: 295.0485
    Epoch [19/50], Test Losses: mse: 5.9813, mae: 1.0497, huber: 0.7807, swd: 4.0471, ept: 295.3009
      Epoch 19 composite train-obj: 0.722469
            Val objective improved 0.7782 → 0.7768, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 5.7292, mae: 1.0010, huber: 0.7315, swd: 3.9150, ept: 299.6578
    Epoch [20/50], Val Losses: mse: 6.3391, mae: 1.1416, huber: 0.8318, swd: 4.1814, ept: 295.8832
    Epoch [20/50], Test Losses: mse: 5.7507, mae: 1.1421, huber: 0.8280, swd: 3.7904, ept: 295.6606
      Epoch 20 composite train-obj: 0.731540
            No improvement (0.8318), counter 1/5
    Epoch [21/50], Train Losses: mse: 5.6993, mae: 1.0092, huber: 0.7369, swd: 3.8501, ept: 299.1552
    Epoch [21/50], Val Losses: mse: 6.1505, mae: 1.0680, huber: 0.7888, swd: 4.2316, ept: 296.4003
    Epoch [21/50], Test Losses: mse: 5.5498, mae: 1.0691, huber: 0.7821, swd: 3.8395, ept: 297.0649
      Epoch 21 composite train-obj: 0.736938
            No improvement (0.7888), counter 2/5
    Epoch [22/50], Train Losses: mse: 5.6065, mae: 0.9960, huber: 0.7271, swd: 3.8032, ept: 299.7302
    Epoch [22/50], Val Losses: mse: 6.0973, mae: 1.1320, huber: 0.8186, swd: 4.1593, ept: 297.5038
    Epoch [22/50], Test Losses: mse: 5.5057, mae: 1.1284, huber: 0.8089, swd: 3.7565, ept: 297.3670
      Epoch 22 composite train-obj: 0.727067
            No improvement (0.8186), counter 3/5
    Epoch [23/50], Train Losses: mse: 5.5711, mae: 0.9933, huber: 0.7228, swd: 3.7747, ept: 300.4264
    Epoch [23/50], Val Losses: mse: 6.5434, mae: 1.0359, huber: 0.7720, swd: 4.3938, ept: 295.9365
    Epoch [23/50], Test Losses: mse: 5.9314, mae: 1.0442, huber: 0.7755, swd: 4.0071, ept: 295.7536
      Epoch 23 composite train-obj: 0.722792
            Val objective improved 0.7768 → 0.7720, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 5.5788, mae: 0.9842, huber: 0.7169, swd: 3.7840, ept: 300.3778
    Epoch [24/50], Val Losses: mse: 6.4473, mae: 1.0421, huber: 0.7735, swd: 4.3968, ept: 295.9018
    Epoch [24/50], Test Losses: mse: 5.8350, mae: 1.0479, huber: 0.7738, swd: 4.0075, ept: 296.1460
      Epoch 24 composite train-obj: 0.716851
            No improvement (0.7735), counter 1/5
    Epoch [25/50], Train Losses: mse: 5.5926, mae: 0.9839, huber: 0.7167, swd: 3.7938, ept: 300.4741
    Epoch [25/50], Val Losses: mse: 6.2177, mae: 1.0516, huber: 0.7746, swd: 4.2571, ept: 296.8832
    Epoch [25/50], Test Losses: mse: 5.6141, mae: 1.0533, huber: 0.7700, swd: 3.8657, ept: 296.9842
      Epoch 25 composite train-obj: 0.716674
            No improvement (0.7746), counter 2/5
    Epoch [26/50], Train Losses: mse: 5.5778, mae: 0.9838, huber: 0.7158, swd: 3.7862, ept: 300.6139
    Epoch [26/50], Val Losses: mse: 6.4423, mae: 1.0458, huber: 0.7742, swd: 4.4040, ept: 296.0210
    Epoch [26/50], Test Losses: mse: 5.8308, mae: 1.0542, huber: 0.7753, swd: 4.0170, ept: 296.2313
      Epoch 26 composite train-obj: 0.715826
            No improvement (0.7742), counter 3/5
    Epoch [27/50], Train Losses: mse: 5.5874, mae: 0.9822, huber: 0.7152, swd: 3.7944, ept: 300.6737
    Epoch [27/50], Val Losses: mse: 6.3402, mae: 1.0369, huber: 0.7693, swd: 4.3111, ept: 296.7632
    Epoch [27/50], Test Losses: mse: 5.7326, mae: 1.0402, huber: 0.7667, swd: 3.9185, ept: 296.5426
      Epoch 27 composite train-obj: 0.715203
            Val objective improved 0.7720 → 0.7693, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 5.5818, mae: 0.9816, huber: 0.7148, swd: 3.7898, ept: 300.7332
    Epoch [28/50], Val Losses: mse: 6.6619, mae: 1.0365, huber: 0.7769, swd: 4.4536, ept: 296.2148
    Epoch [28/50], Test Losses: mse: 6.0580, mae: 1.0469, huber: 0.7833, swd: 4.0701, ept: 295.4345
      Epoch 28 composite train-obj: 0.714775
            No improvement (0.7769), counter 1/5
    Epoch [29/50], Train Losses: mse: 5.5878, mae: 0.9811, huber: 0.7146, swd: 3.7960, ept: 300.7603
    Epoch [29/50], Val Losses: mse: 6.4104, mae: 1.0360, huber: 0.7664, swd: 4.3607, ept: 296.6397
    Epoch [29/50], Test Losses: mse: 5.8003, mae: 1.0410, huber: 0.7663, swd: 3.9725, ept: 296.4403
      Epoch 29 composite train-obj: 0.714588
            Val objective improved 0.7693 → 0.7664, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 5.5734, mae: 0.9790, huber: 0.7126, swd: 3.7860, ept: 300.8908
    Epoch [30/50], Val Losses: mse: 6.4743, mae: 1.0316, huber: 0.7660, swd: 4.4097, ept: 296.6081
    Epoch [30/50], Test Losses: mse: 5.8681, mae: 1.0403, huber: 0.7687, swd: 4.0251, ept: 296.5660
      Epoch 30 composite train-obj: 0.712595
            Val objective improved 0.7664 → 0.7660, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 5.5910, mae: 0.9797, huber: 0.7135, swd: 3.8024, ept: 300.8764
    Epoch [31/50], Val Losses: mse: 6.5674, mae: 1.0339, huber: 0.7725, swd: 4.4141, ept: 296.4260
    Epoch [31/50], Test Losses: mse: 5.9615, mae: 1.0449, huber: 0.7781, swd: 4.0317, ept: 295.6763
      Epoch 31 composite train-obj: 0.713483
            No improvement (0.7725), counter 1/5
    Epoch [32/50], Train Losses: mse: 5.5852, mae: 0.9785, huber: 0.7126, swd: 3.7959, ept: 300.9320
    Epoch [32/50], Val Losses: mse: 6.4695, mae: 1.0287, huber: 0.7648, swd: 4.3559, ept: 296.8641
    Epoch [32/50], Test Losses: mse: 5.8632, mae: 1.0371, huber: 0.7675, swd: 3.9661, ept: 296.5770
      Epoch 32 composite train-obj: 0.712646
            Val objective improved 0.7660 → 0.7648, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 5.5888, mae: 0.9778, huber: 0.7119, swd: 3.8027, ept: 300.9851
    Epoch [33/50], Val Losses: mse: 6.5487, mae: 1.0296, huber: 0.7682, swd: 4.3849, ept: 296.5768
    Epoch [33/50], Test Losses: mse: 5.9406, mae: 1.0387, huber: 0.7732, swd: 3.9991, ept: 296.0785
      Epoch 33 composite train-obj: 0.711942
            No improvement (0.7682), counter 1/5
    Epoch [34/50], Train Losses: mse: 5.7297, mae: 0.9918, huber: 0.7255, swd: 3.9164, ept: 300.0653
    Epoch [34/50], Val Losses: mse: 6.3899, mae: 1.1475, huber: 0.8242, swd: 4.2282, ept: 296.8699
    Epoch [34/50], Test Losses: mse: 5.8022, mae: 1.1541, huber: 0.8252, swd: 3.8404, ept: 296.6632
      Epoch 34 composite train-obj: 0.725540
            No improvement (0.8242), counter 2/5
    Epoch [35/50], Train Losses: mse: 5.5636, mae: 0.9861, huber: 0.7163, swd: 3.7754, ept: 300.9679
    Epoch [35/50], Val Losses: mse: 6.3942, mae: 1.0425, huber: 0.7674, swd: 4.3433, ept: 297.1404
    Epoch [35/50], Test Losses: mse: 5.7903, mae: 1.0475, huber: 0.7684, swd: 3.9575, ept: 297.1631
      Epoch 35 composite train-obj: 0.716324
            No improvement (0.7674), counter 3/5
    Epoch [36/50], Train Losses: mse: 5.5644, mae: 0.9743, huber: 0.7095, swd: 3.7809, ept: 301.1353
    Epoch [36/50], Val Losses: mse: 6.4209, mae: 1.0289, huber: 0.7627, swd: 4.3638, ept: 296.9846
    Epoch [36/50], Test Losses: mse: 5.8147, mae: 1.0370, huber: 0.7644, swd: 3.9784, ept: 296.9370
      Epoch 36 composite train-obj: 0.709538
            Val objective improved 0.7648 → 0.7627, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 5.5609, mae: 0.9741, huber: 0.7092, swd: 3.7778, ept: 301.1852
    Epoch [37/50], Val Losses: mse: 6.4565, mae: 1.0308, huber: 0.7629, swd: 4.4027, ept: 296.8972
    Epoch [37/50], Test Losses: mse: 5.8463, mae: 1.0356, huber: 0.7635, swd: 4.0133, ept: 296.8486
      Epoch 37 composite train-obj: 0.709193
            No improvement (0.7629), counter 1/5
    Epoch [38/50], Train Losses: mse: 5.5669, mae: 0.9742, huber: 0.7089, swd: 3.7800, ept: 301.1636
    Epoch [38/50], Val Losses: mse: 6.5518, mae: 1.0221, huber: 0.7627, swd: 4.4339, ept: 296.8227
    Epoch [38/50], Test Losses: mse: 5.9446, mae: 1.0319, huber: 0.7673, swd: 4.0520, ept: 296.4448
      Epoch 38 composite train-obj: 0.708881
            Val objective improved 0.7627 → 0.7627, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 5.6326, mae: 0.9806, huber: 0.7155, swd: 3.8285, ept: 300.7001
    Epoch [39/50], Val Losses: mse: 6.2749, mae: 1.0417, huber: 0.7724, swd: 4.3336, ept: 297.2414
    Epoch [39/50], Test Losses: mse: 5.6750, mae: 1.0460, huber: 0.7700, swd: 3.9463, ept: 297.2798
      Epoch 39 composite train-obj: 0.715529
            No improvement (0.7724), counter 1/5
    Epoch [40/50], Train Losses: mse: 5.5390, mae: 0.9753, huber: 0.7095, swd: 3.7708, ept: 301.2821
    Epoch [40/50], Val Losses: mse: 6.6578, mae: 1.0242, huber: 0.7671, swd: 4.5371, ept: 296.5796
    Epoch [40/50], Test Losses: mse: 6.0506, mae: 1.0349, huber: 0.7732, swd: 4.1575, ept: 296.1234
      Epoch 40 composite train-obj: 0.709530
            No improvement (0.7671), counter 2/5
    Epoch [41/50], Train Losses: mse: 5.5757, mae: 0.9724, huber: 0.7078, swd: 3.7887, ept: 301.2242
    Epoch [41/50], Val Losses: mse: 6.4930, mae: 1.0216, huber: 0.7613, swd: 4.3682, ept: 297.0540
    Epoch [41/50], Test Losses: mse: 5.8877, mae: 1.0282, huber: 0.7648, swd: 3.9839, ept: 296.6438
      Epoch 41 composite train-obj: 0.707782
            Val objective improved 0.7627 → 0.7613, saving checkpoint.
    Epoch [42/50], Train Losses: mse: 5.5556, mae: 0.9728, huber: 0.7076, swd: 3.7803, ept: 301.2538
    Epoch [42/50], Val Losses: mse: 6.5692, mae: 1.0290, huber: 0.7637, swd: 4.4483, ept: 297.0061
    Epoch [42/50], Test Losses: mse: 5.9638, mae: 1.0370, huber: 0.7683, swd: 4.0662, ept: 296.9382
      Epoch 42 composite train-obj: 0.707572
            No improvement (0.7637), counter 1/5
    Epoch [43/50], Train Losses: mse: 5.5658, mae: 0.9716, huber: 0.7073, swd: 3.7873, ept: 301.2892
    Epoch [43/50], Val Losses: mse: 6.3727, mae: 1.0504, huber: 0.7725, swd: 4.3126, ept: 297.1429
    Epoch [43/50], Test Losses: mse: 5.7696, mae: 1.0553, huber: 0.7708, swd: 3.9225, ept: 296.6782
      Epoch 43 composite train-obj: 0.707331
            No improvement (0.7725), counter 2/5
    Epoch [44/50], Train Losses: mse: 5.5640, mae: 0.9729, huber: 0.7077, swd: 3.7916, ept: 301.1407
    Epoch [44/50], Val Losses: mse: 6.5322, mae: 1.0196, huber: 0.7586, swd: 4.4090, ept: 297.1894
    Epoch [44/50], Test Losses: mse: 5.9283, mae: 1.0289, huber: 0.7630, swd: 4.0242, ept: 296.6281
      Epoch 44 composite train-obj: 0.707677
            Val objective improved 0.7613 → 0.7586, saving checkpoint.
    Epoch [45/50], Train Losses: mse: 5.5651, mae: 0.9681, huber: 0.7050, swd: 3.7914, ept: 301.3334
    Epoch [45/50], Val Losses: mse: 6.3941, mae: 1.0228, huber: 0.7577, swd: 4.3577, ept: 297.2581
    Epoch [45/50], Test Losses: mse: 5.7855, mae: 1.0290, huber: 0.7582, swd: 3.9703, ept: 297.0818
      Epoch 45 composite train-obj: 0.704961
            Val objective improved 0.7586 → 0.7577, saving checkpoint.
    Epoch [46/50], Train Losses: mse: 5.5533, mae: 0.9686, huber: 0.7050, swd: 3.7805, ept: 301.3913
    Epoch [46/50], Val Losses: mse: 6.4082, mae: 1.0228, huber: 0.7577, swd: 4.3775, ept: 297.3591
    Epoch [46/50], Test Losses: mse: 5.8029, mae: 1.0272, huber: 0.7576, swd: 3.9884, ept: 297.3305
      Epoch 46 composite train-obj: 0.704979
            Val objective improved 0.7577 → 0.7577, saving checkpoint.
    Epoch [47/50], Train Losses: mse: 5.5501, mae: 0.9686, huber: 0.7048, swd: 3.7799, ept: 301.4292
    Epoch [47/50], Val Losses: mse: 6.5590, mae: 1.0302, huber: 0.7638, swd: 4.3882, ept: 297.3960
    Epoch [47/50], Test Losses: mse: 5.9565, mae: 1.0363, huber: 0.7684, swd: 4.0022, ept: 296.9545
      Epoch 47 composite train-obj: 0.704776
            No improvement (0.7638), counter 1/5
    Epoch [48/50], Train Losses: mse: 5.5597, mae: 0.9683, huber: 0.7046, swd: 3.7883, ept: 301.4311
    Epoch [48/50], Val Losses: mse: 6.3697, mae: 1.0179, huber: 0.7591, swd: 4.3574, ept: 297.2180
    Epoch [48/50], Test Losses: mse: 5.7622, mae: 1.0235, huber: 0.7585, swd: 3.9686, ept: 296.7508
      Epoch 48 composite train-obj: 0.704554
            No improvement (0.7591), counter 2/5
    Epoch [49/50], Train Losses: mse: 5.5543, mae: 0.9677, huber: 0.7043, swd: 3.7831, ept: 301.4582
    Epoch [49/50], Val Losses: mse: 6.6460, mae: 1.0201, huber: 0.7631, swd: 4.4681, ept: 297.0811
    Epoch [49/50], Test Losses: mse: 6.0457, mae: 1.0315, huber: 0.7708, swd: 4.0908, ept: 296.4496
      Epoch 49 composite train-obj: 0.704274
            No improvement (0.7631), counter 3/5
    Epoch [50/50], Train Losses: mse: 5.5651, mae: 0.9679, huber: 0.7045, swd: 3.7911, ept: 301.4390
    Epoch [50/50], Val Losses: mse: 6.4678, mae: 1.0187, huber: 0.7578, swd: 4.3460, ept: 297.3920
    Epoch [50/50], Test Losses: mse: 5.8669, mae: 1.0247, huber: 0.7609, swd: 3.9588, ept: 296.8502
      Epoch 50 composite train-obj: 0.704527
            No improvement (0.7578), counter 4/5
    Epoch [50/50], Test Losses: mse: 5.8029, mae: 1.0272, huber: 0.7576, swd: 3.9884, ept: 297.3305
    Best round's Test MSE: 5.8029, MAE: 1.0272, SWD: 3.9884
    Best round's Validation MSE: 6.4082, MAE: 1.0228, SWD: 4.3775
    Best round's Test verification MSE : 5.8029, MAE: 1.0272, SWD: 3.9884
    Time taken: 122.40 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.6652, mae: 1.3130, huber: 1.0157, swd: 5.0041, ept: 271.5189
    Epoch [1/50], Val Losses: mse: 7.7724, mae: 1.1731, huber: 0.8885, swd: 5.3811, ept: 289.2974
    Epoch [1/50], Test Losses: mse: 7.0603, mae: 1.1989, huber: 0.9051, swd: 4.8774, ept: 288.2951
      Epoch 1 composite train-obj: 1.015672
            Val objective improved inf → 0.8885, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.4701, mae: 1.1044, huber: 0.8165, swd: 4.4747, ept: 294.5788
    Epoch [2/50], Val Losses: mse: 7.2640, mae: 1.1497, huber: 0.8612, swd: 5.0460, ept: 291.1303
    Epoch [2/50], Test Losses: mse: 6.5721, mae: 1.1685, huber: 0.8710, swd: 4.5648, ept: 290.6808
      Epoch 2 composite train-obj: 0.816462
            Val objective improved 0.8885 → 0.8612, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.1396, mae: 1.0761, huber: 0.7899, swd: 4.2506, ept: 296.4654
    Epoch [3/50], Val Losses: mse: 7.0985, mae: 1.1179, huber: 0.8392, swd: 4.8611, ept: 291.9001
    Epoch [3/50], Test Losses: mse: 6.4236, mae: 1.1321, huber: 0.8470, swd: 4.3989, ept: 291.3890
      Epoch 3 composite train-obj: 0.789917
            Val objective improved 0.8612 → 0.8392, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.0954, mae: 1.0672, huber: 0.7838, swd: 4.2396, ept: 296.9146
    Epoch [4/50], Val Losses: mse: 6.6432, mae: 1.1640, huber: 0.8590, swd: 4.5590, ept: 292.6272
    Epoch [4/50], Test Losses: mse: 5.9916, mae: 1.1721, huber: 0.8561, swd: 4.1080, ept: 292.6818
      Epoch 4 composite train-obj: 0.783838
            No improvement (0.8590), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.0047, mae: 1.0678, huber: 0.7834, swd: 4.1395, ept: 296.3320
    Epoch [5/50], Val Losses: mse: 6.5507, mae: 1.1672, huber: 0.8566, swd: 4.5870, ept: 293.9509
    Epoch [5/50], Test Losses: mse: 5.9125, mae: 1.1699, huber: 0.8529, swd: 4.1369, ept: 294.3743
      Epoch 5 composite train-obj: 0.783396
            No improvement (0.8566), counter 2/5
    Epoch [6/50], Train Losses: mse: 5.8297, mae: 1.0453, huber: 0.7636, swd: 4.0307, ept: 298.2491
    Epoch [6/50], Val Losses: mse: 6.8070, mae: 1.0889, huber: 0.8118, swd: 4.7135, ept: 293.3255
    Epoch [6/50], Test Losses: mse: 6.1515, mae: 1.0986, huber: 0.8155, swd: 4.2777, ept: 293.3674
      Epoch 6 composite train-obj: 0.763630
            Val objective improved 0.8392 → 0.8118, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.8107, mae: 1.0335, huber: 0.7551, swd: 4.0184, ept: 298.3491
    Epoch [7/50], Val Losses: mse: 6.8096, mae: 1.0890, huber: 0.8158, swd: 4.6338, ept: 293.4463
    Epoch [7/50], Test Losses: mse: 6.1735, mae: 1.1011, huber: 0.8217, swd: 4.2108, ept: 293.2088
      Epoch 7 composite train-obj: 0.755059
            No improvement (0.8158), counter 1/5
    Epoch [8/50], Train Losses: mse: 5.8055, mae: 1.0310, huber: 0.7541, swd: 4.0088, ept: 298.6047
    Epoch [8/50], Val Losses: mse: 6.4875, mae: 1.1227, huber: 0.8267, swd: 4.4256, ept: 294.2482
    Epoch [8/50], Test Losses: mse: 5.8645, mae: 1.1260, huber: 0.8245, swd: 4.0000, ept: 294.3234
      Epoch 8 composite train-obj: 0.754134
            No improvement (0.8267), counter 2/5
    Epoch [9/50], Train Losses: mse: 5.7436, mae: 1.0300, huber: 0.7519, swd: 3.9617, ept: 298.5406
    Epoch [9/50], Val Losses: mse: 6.3582, mae: 1.0965, huber: 0.8113, swd: 4.4558, ept: 294.4752
    Epoch [9/50], Test Losses: mse: 5.7354, mae: 1.1016, huber: 0.8072, swd: 4.0414, ept: 295.0044
      Epoch 9 composite train-obj: 0.751879
            Val objective improved 0.8118 → 0.8113, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 5.7090, mae: 1.0192, huber: 0.7437, swd: 3.9426, ept: 299.0251
    Epoch [10/50], Val Losses: mse: 6.7406, mae: 1.0720, huber: 0.7985, swd: 4.6257, ept: 294.1710
    Epoch [10/50], Test Losses: mse: 6.1099, mae: 1.0820, huber: 0.8033, swd: 4.2164, ept: 294.3851
      Epoch 10 composite train-obj: 0.743655
            Val objective improved 0.8113 → 0.7985, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 5.7134, mae: 1.0145, huber: 0.7401, swd: 3.9398, ept: 299.0878
    Epoch [11/50], Val Losses: mse: 6.5413, mae: 1.0710, huber: 0.7957, swd: 4.5418, ept: 294.5917
    Epoch [11/50], Test Losses: mse: 5.9204, mae: 1.0809, huber: 0.7981, swd: 4.1328, ept: 294.9308
      Epoch 11 composite train-obj: 0.740116
            Val objective improved 0.7985 → 0.7957, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 5.6957, mae: 1.0098, huber: 0.7370, swd: 3.9292, ept: 299.2125
    Epoch [12/50], Val Losses: mse: 6.7465, mae: 1.0741, huber: 0.7996, swd: 4.6217, ept: 294.3228
    Epoch [12/50], Test Losses: mse: 6.1212, mae: 1.0839, huber: 0.8045, swd: 4.2197, ept: 294.4265
      Epoch 12 composite train-obj: 0.737045
            No improvement (0.7996), counter 1/5
    Epoch [13/50], Train Losses: mse: 5.7557, mae: 1.0184, huber: 0.7451, swd: 3.9585, ept: 298.9070
    Epoch [13/50], Val Losses: mse: 6.3689, mae: 1.1710, huber: 0.8469, swd: 4.3446, ept: 295.2509
    Epoch [13/50], Test Losses: mse: 5.7696, mae: 1.1749, huber: 0.8443, swd: 3.9260, ept: 295.2497
      Epoch 13 composite train-obj: 0.745128
            No improvement (0.8469), counter 2/5
    Epoch [14/50], Train Losses: mse: 5.6591, mae: 1.0126, huber: 0.7376, swd: 3.8961, ept: 299.3915
    Epoch [14/50], Val Losses: mse: 6.5764, mae: 1.0520, huber: 0.7846, swd: 4.5184, ept: 294.5782
    Epoch [14/50], Test Losses: mse: 5.9532, mae: 1.0596, huber: 0.7872, swd: 4.1143, ept: 294.6519
      Epoch 14 composite train-obj: 0.737569
            Val objective improved 0.7957 → 0.7846, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 5.6538, mae: 1.0013, huber: 0.7306, swd: 3.8955, ept: 299.5254
    Epoch [15/50], Val Losses: mse: 6.5763, mae: 1.0649, huber: 0.7929, swd: 4.4907, ept: 294.8647
    Epoch [15/50], Test Losses: mse: 5.9655, mae: 1.0719, huber: 0.7948, swd: 4.0897, ept: 294.7481
      Epoch 15 composite train-obj: 0.730556
            No improvement (0.7929), counter 1/5
    Epoch [16/50], Train Losses: mse: 5.6698, mae: 1.0039, huber: 0.7330, swd: 3.9068, ept: 299.3966
    Epoch [16/50], Val Losses: mse: 6.2335, mae: 1.0772, huber: 0.7970, swd: 4.3656, ept: 295.4313
    Epoch [16/50], Test Losses: mse: 5.6305, mae: 1.0793, huber: 0.7922, swd: 3.9667, ept: 296.2616
      Epoch 16 composite train-obj: 0.732964
            No improvement (0.7970), counter 2/5
    Epoch [17/50], Train Losses: mse: 5.6090, mae: 0.9996, huber: 0.7285, swd: 3.8709, ept: 299.7951
    Epoch [17/50], Val Losses: mse: 6.4864, mae: 1.0506, huber: 0.7794, swd: 4.4458, ept: 295.2148
    Epoch [17/50], Test Losses: mse: 5.8749, mae: 1.0566, huber: 0.7804, swd: 4.0456, ept: 295.4376
      Epoch 17 composite train-obj: 0.728452
            Val objective improved 0.7846 → 0.7794, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 5.6263, mae: 0.9951, huber: 0.7253, swd: 3.8779, ept: 299.7640
    Epoch [18/50], Val Losses: mse: 6.5260, mae: 1.0397, huber: 0.7771, swd: 4.5069, ept: 295.1083
    Epoch [18/50], Test Losses: mse: 5.9157, mae: 1.0477, huber: 0.7800, swd: 4.1110, ept: 295.3428
      Epoch 18 composite train-obj: 0.725259
            Val objective improved 0.7794 → 0.7771, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 5.6268, mae: 0.9917, huber: 0.7231, swd: 3.8803, ept: 299.8246
    Epoch [19/50], Val Losses: mse: 6.4212, mae: 1.0478, huber: 0.7764, swd: 4.4540, ept: 295.4796
    Epoch [19/50], Test Losses: mse: 5.8124, mae: 1.0541, huber: 0.7769, swd: 4.0564, ept: 295.9332
      Epoch 19 composite train-obj: 0.723105
            Val objective improved 0.7771 → 0.7764, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 5.6120, mae: 0.9909, huber: 0.7221, swd: 3.8715, ept: 299.9177
    Epoch [20/50], Val Losses: mse: 6.6900, mae: 1.0378, huber: 0.7781, swd: 4.5625, ept: 295.1957
    Epoch [20/50], Test Losses: mse: 6.0838, mae: 1.0504, huber: 0.7861, swd: 4.1749, ept: 295.1361
      Epoch 20 composite train-obj: 0.722149
            No improvement (0.7781), counter 1/5
    Epoch [21/50], Train Losses: mse: 5.6463, mae: 0.9946, huber: 0.7263, swd: 3.8962, ept: 299.8385
    Epoch [21/50], Val Losses: mse: 6.2300, mae: 1.1003, huber: 0.7980, swd: 4.2987, ept: 296.6489
    Epoch [21/50], Test Losses: mse: 5.6280, mae: 1.1005, huber: 0.7924, swd: 3.8910, ept: 296.4830
      Epoch 21 composite train-obj: 0.726326
            No improvement (0.7980), counter 2/5
    Epoch [22/50], Train Losses: mse: 5.5800, mae: 0.9927, huber: 0.7222, swd: 3.8483, ept: 300.2186
    Epoch [22/50], Val Losses: mse: 6.6248, mae: 1.0375, huber: 0.7739, swd: 4.5580, ept: 295.6444
    Epoch [22/50], Test Losses: mse: 6.0165, mae: 1.0467, huber: 0.7790, swd: 4.1677, ept: 295.5948
      Epoch 22 composite train-obj: 0.722239
            Val objective improved 0.7764 → 0.7739, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 5.6017, mae: 0.9851, huber: 0.7178, swd: 3.8667, ept: 300.2141
    Epoch [23/50], Val Losses: mse: 6.4627, mae: 1.0409, huber: 0.7708, swd: 4.4334, ept: 296.2915
    Epoch [23/50], Test Losses: mse: 5.8579, mae: 1.0476, huber: 0.7732, swd: 4.0408, ept: 296.5519
      Epoch 23 composite train-obj: 0.717757
            Val objective improved 0.7739 → 0.7708, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 5.5966, mae: 0.9851, huber: 0.7175, swd: 3.8679, ept: 300.3299
    Epoch [24/50], Val Losses: mse: 6.6470, mae: 1.0359, huber: 0.7763, swd: 4.5415, ept: 295.9311
    Epoch [24/50], Test Losses: mse: 6.0414, mae: 1.0462, huber: 0.7831, swd: 4.1577, ept: 295.4159
      Epoch 24 composite train-obj: 0.717532
            No improvement (0.7763), counter 1/5
    Epoch [25/50], Train Losses: mse: 5.5908, mae: 0.9841, huber: 0.7166, swd: 3.8598, ept: 300.5128
    Epoch [25/50], Val Losses: mse: 6.6048, mae: 1.0408, huber: 0.7742, swd: 4.4957, ept: 296.3061
    Epoch [25/50], Test Losses: mse: 5.9999, mae: 1.0539, huber: 0.7809, swd: 4.1092, ept: 295.8945
      Epoch 25 composite train-obj: 0.716614
            No improvement (0.7742), counter 2/5
    Epoch [26/50], Train Losses: mse: 5.6073, mae: 0.9854, huber: 0.7180, swd: 3.8741, ept: 300.3469
    Epoch [26/50], Val Losses: mse: 6.1674, mae: 1.0566, huber: 0.7827, swd: 4.3263, ept: 296.8371
    Epoch [26/50], Test Losses: mse: 5.5680, mae: 1.0582, huber: 0.7762, swd: 3.9324, ept: 296.9429
      Epoch 26 composite train-obj: 0.718027
            No improvement (0.7827), counter 3/5
    Epoch [27/50], Train Losses: mse: 5.5939, mae: 0.9870, huber: 0.7189, swd: 3.8649, ept: 300.6342
    Epoch [27/50], Val Losses: mse: 6.2741, mae: 1.0706, huber: 0.7895, swd: 4.4242, ept: 296.9274
    Epoch [27/50], Test Losses: mse: 5.6729, mae: 1.0694, huber: 0.7816, swd: 4.0237, ept: 296.9663
      Epoch 27 composite train-obj: 0.718917
            No improvement (0.7895), counter 4/5
    Epoch [28/50], Train Losses: mse: 5.5786, mae: 0.9829, huber: 0.7151, swd: 3.8546, ept: 300.7939
    Epoch [28/50], Val Losses: mse: 6.5789, mae: 1.0455, huber: 0.7720, swd: 4.5205, ept: 296.4517
    Epoch [28/50], Test Losses: mse: 5.9686, mae: 1.0551, huber: 0.7767, swd: 4.1260, ept: 296.2566
      Epoch 28 composite train-obj: 0.715105
    Epoch [28/50], Test Losses: mse: 5.8579, mae: 1.0476, huber: 0.7732, swd: 4.0408, ept: 296.5519
    Best round's Test MSE: 5.8579, MAE: 1.0476, SWD: 4.0408
    Best round's Validation MSE: 6.4627, MAE: 1.0409, SWD: 4.4334
    Best round's Test verification MSE : 5.8579, MAE: 1.0476, SWD: 4.0408
    Time taken: 69.13 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.6973, mae: 1.3203, huber: 1.0225, swd: 4.9573, ept: 269.2638
    Epoch [1/50], Val Losses: mse: 7.5183, mae: 1.1865, huber: 0.8923, swd: 5.1435, ept: 289.4320
    Epoch [1/50], Test Losses: mse: 6.8141, mae: 1.2075, huber: 0.9055, swd: 4.6496, ept: 288.4211
      Epoch 1 composite train-obj: 1.022547
            Val objective improved inf → 0.8923, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.4470, mae: 1.1030, huber: 0.8143, swd: 4.4233, ept: 294.6743
    Epoch [2/50], Val Losses: mse: 7.3357, mae: 1.1548, huber: 0.8642, swd: 5.0605, ept: 290.9596
    Epoch [2/50], Test Losses: mse: 6.6381, mae: 1.1691, huber: 0.8726, swd: 4.5922, ept: 290.7474
      Epoch 2 composite train-obj: 0.814264
            Val objective improved 0.8923 → 0.8642, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 6.1561, mae: 1.0771, huber: 0.7908, swd: 4.2346, ept: 296.2279
    Epoch [3/50], Val Losses: mse: 7.2276, mae: 1.1357, huber: 0.8508, swd: 4.9183, ept: 291.6682
    Epoch [3/50], Test Losses: mse: 6.5472, mae: 1.1496, huber: 0.8591, swd: 4.4645, ept: 291.0423
      Epoch 3 composite train-obj: 0.790817
            Val objective improved 0.8642 → 0.8508, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.0145, mae: 1.0604, huber: 0.7770, swd: 4.1365, ept: 297.1288
    Epoch [4/50], Val Losses: mse: 7.0318, mae: 1.1145, huber: 0.8325, swd: 4.8115, ept: 292.2368
    Epoch [4/50], Test Losses: mse: 6.3609, mae: 1.1276, huber: 0.8392, swd: 4.3701, ept: 292.1196
      Epoch 4 composite train-obj: 0.776956
            Val objective improved 0.8508 → 0.8325, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.9064, mae: 1.0491, huber: 0.7669, swd: 4.0619, ept: 297.7342
    Epoch [5/50], Val Losses: mse: 6.9101, mae: 1.1011, huber: 0.8191, swd: 4.7185, ept: 293.1009
    Epoch [5/50], Test Losses: mse: 6.2572, mae: 1.1141, huber: 0.8257, swd: 4.2883, ept: 293.2703
      Epoch 5 composite train-obj: 0.766925
            Val objective improved 0.8325 → 0.8191, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.8428, mae: 1.0399, huber: 0.7600, swd: 4.0199, ept: 298.1967
    Epoch [6/50], Val Losses: mse: 6.6469, mae: 1.1029, huber: 0.8152, swd: 4.6113, ept: 293.7167
    Epoch [6/50], Test Losses: mse: 5.9987, mae: 1.1106, huber: 0.8150, swd: 4.1811, ept: 293.9528
      Epoch 6 composite train-obj: 0.759966
            Val objective improved 0.8191 → 0.8152, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.8773, mae: 1.0392, huber: 0.7606, swd: 4.0473, ept: 298.0168
    Epoch [7/50], Val Losses: mse: 6.3355, mae: 1.1054, huber: 0.8240, swd: 4.4509, ept: 294.5159
    Epoch [7/50], Test Losses: mse: 5.7104, mae: 1.1052, huber: 0.8165, swd: 4.0258, ept: 294.8653
      Epoch 7 composite train-obj: 0.760571
            No improvement (0.8240), counter 1/5
    Epoch [8/50], Train Losses: mse: 5.7580, mae: 1.0280, huber: 0.7506, swd: 3.9693, ept: 298.8052
    Epoch [8/50], Val Losses: mse: 6.9536, mae: 1.0810, huber: 0.8116, swd: 4.6962, ept: 293.6942
    Epoch [8/50], Test Losses: mse: 6.3236, mae: 1.0971, huber: 0.8224, swd: 4.2909, ept: 293.1897
      Epoch 8 composite train-obj: 0.750623
            Val objective improved 0.8152 → 0.8116, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 5.8047, mae: 1.0256, huber: 0.7500, swd: 4.0059, ept: 298.6136
    Epoch [9/50], Val Losses: mse: 6.3663, mae: 1.1322, huber: 0.8292, swd: 4.3695, ept: 294.8486
    Epoch [9/50], Test Losses: mse: 5.7469, mae: 1.1364, huber: 0.8235, swd: 3.9562, ept: 295.2513
      Epoch 9 composite train-obj: 0.750010
            No improvement (0.8292), counter 1/5
    Epoch [10/50], Train Losses: mse: 5.7091, mae: 1.0223, huber: 0.7452, swd: 3.9209, ept: 298.9307
    Epoch [10/50], Val Losses: mse: 6.6694, mae: 1.0679, huber: 0.8000, swd: 4.5437, ept: 294.1830
    Epoch [10/50], Test Losses: mse: 6.0520, mae: 1.0793, huber: 0.8049, swd: 4.1417, ept: 293.8078
      Epoch 10 composite train-obj: 0.745216
            Val objective improved 0.8116 → 0.8000, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 5.7145, mae: 1.0125, huber: 0.7394, swd: 3.9289, ept: 299.1805
    Epoch [11/50], Val Losses: mse: 6.3574, mae: 1.0797, huber: 0.7970, swd: 4.4346, ept: 295.0237
    Epoch [11/50], Test Losses: mse: 5.7387, mae: 1.0824, huber: 0.7928, swd: 4.0286, ept: 295.6060
      Epoch 11 composite train-obj: 0.739435
            Val objective improved 0.8000 → 0.7970, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 5.6926, mae: 1.0075, huber: 0.7350, swd: 3.9149, ept: 299.3192
    Epoch [12/50], Val Losses: mse: 6.7274, mae: 1.0668, huber: 0.7954, swd: 4.5574, ept: 294.5327
    Epoch [12/50], Test Losses: mse: 6.1130, mae: 1.0779, huber: 0.8018, swd: 4.1638, ept: 294.4174
      Epoch 12 composite train-obj: 0.735039
            Val objective improved 0.7970 → 0.7954, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 5.6828, mae: 1.0041, huber: 0.7327, swd: 3.9088, ept: 299.4965
    Epoch [13/50], Val Losses: mse: 6.6416, mae: 1.0526, huber: 0.7858, swd: 4.5375, ept: 294.6536
    Epoch [13/50], Test Losses: mse: 6.0224, mae: 1.0645, huber: 0.7915, swd: 4.1412, ept: 294.5266
      Epoch 13 composite train-obj: 0.732672
            Val objective improved 0.7954 → 0.7858, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 5.6642, mae: 1.0013, huber: 0.7305, swd: 3.8956, ept: 299.5545
    Epoch [14/50], Val Losses: mse: 6.6260, mae: 1.0602, huber: 0.7894, swd: 4.5443, ept: 294.8399
    Epoch [14/50], Test Losses: mse: 6.0069, mae: 1.0694, huber: 0.7930, swd: 4.1478, ept: 294.9171
      Epoch 14 composite train-obj: 0.730470
            No improvement (0.7894), counter 1/5
    Epoch [15/50], Train Losses: mse: 5.6795, mae: 1.0053, huber: 0.7339, swd: 3.9051, ept: 299.3286
    Epoch [15/50], Val Losses: mse: 6.1912, mae: 1.0987, huber: 0.8082, swd: 4.3448, ept: 296.0214
    Epoch [15/50], Test Losses: mse: 5.5925, mae: 1.0970, huber: 0.8001, swd: 3.9435, ept: 296.8511
      Epoch 15 composite train-obj: 0.733901
            No improvement (0.8082), counter 2/5
    Epoch [16/50], Train Losses: mse: 5.6320, mae: 1.0036, huber: 0.7318, swd: 3.8772, ept: 299.5719
    Epoch [16/50], Val Losses: mse: 6.3106, mae: 1.0605, huber: 0.7895, swd: 4.4048, ept: 295.4193
    Epoch [16/50], Test Losses: mse: 5.7022, mae: 1.0624, huber: 0.7835, swd: 4.0028, ept: 295.8540
      Epoch 16 composite train-obj: 0.731834
            No improvement (0.7895), counter 3/5
    Epoch [17/50], Train Losses: mse: 5.6295, mae: 0.9955, huber: 0.7256, swd: 3.8748, ept: 299.7858
    Epoch [17/50], Val Losses: mse: 6.4430, mae: 1.0464, huber: 0.7781, swd: 4.4245, ept: 295.4103
    Epoch [17/50], Test Losses: mse: 5.8316, mae: 1.0540, huber: 0.7798, swd: 4.0302, ept: 295.7221
      Epoch 17 composite train-obj: 0.725605
            Val objective improved 0.7858 → 0.7781, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 5.7464, mae: 1.0053, huber: 0.7360, swd: 3.9495, ept: 298.9908
    Epoch [18/50], Val Losses: mse: 6.2661, mae: 1.1246, huber: 0.8239, swd: 4.3741, ept: 296.4324
    Epoch [18/50], Test Losses: mse: 5.6784, mae: 1.1259, huber: 0.8194, swd: 3.9750, ept: 296.8919
      Epoch 18 composite train-obj: 0.736035
            No improvement (0.8239), counter 1/5
    Epoch [19/50], Train Losses: mse: 5.6077, mae: 0.9988, huber: 0.7274, swd: 3.8595, ept: 300.0910
    Epoch [19/50], Val Losses: mse: 6.7357, mae: 1.0431, huber: 0.7811, swd: 4.5789, ept: 295.1367
    Epoch [19/50], Test Losses: mse: 6.1271, mae: 1.0557, huber: 0.7889, swd: 4.1926, ept: 294.7558
      Epoch 19 composite train-obj: 0.727369
            No improvement (0.7811), counter 2/5
    Epoch [20/50], Train Losses: mse: 5.6239, mae: 0.9888, huber: 0.7207, swd: 3.8703, ept: 300.0213
    Epoch [20/50], Val Losses: mse: 6.6786, mae: 1.0398, huber: 0.7784, swd: 4.5659, ept: 295.1980
    Epoch [20/50], Test Losses: mse: 6.0667, mae: 1.0500, huber: 0.7841, swd: 4.1758, ept: 295.1097
      Epoch 20 composite train-obj: 0.720700
            No improvement (0.7784), counter 3/5
    Epoch [21/50], Train Losses: mse: 5.6139, mae: 0.9879, huber: 0.7198, swd: 3.8624, ept: 300.1078
    Epoch [21/50], Val Losses: mse: 6.6355, mae: 1.0342, huber: 0.7753, swd: 4.5304, ept: 295.3610
    Epoch [21/50], Test Losses: mse: 6.0242, mae: 1.0450, huber: 0.7815, swd: 4.1452, ept: 295.3087
      Epoch 21 composite train-obj: 0.719789
            Val objective improved 0.7781 → 0.7753, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 5.6170, mae: 0.9867, huber: 0.7190, swd: 3.8664, ept: 300.2444
    Epoch [22/50], Val Losses: mse: 6.6347, mae: 1.0385, huber: 0.7797, swd: 4.4871, ept: 296.1218
    Epoch [22/50], Test Losses: mse: 6.0310, mae: 1.0487, huber: 0.7854, swd: 4.0995, ept: 295.5513
      Epoch 22 composite train-obj: 0.718960
            No improvement (0.7797), counter 1/5
    Epoch [23/50], Train Losses: mse: 5.7114, mae: 0.9989, huber: 0.7307, swd: 3.9183, ept: 299.5620
    Epoch [23/50], Val Losses: mse: 6.3361, mae: 1.1557, huber: 0.8328, swd: 4.3453, ept: 297.1498
    Epoch [23/50], Test Losses: mse: 5.7455, mae: 1.1610, huber: 0.8316, swd: 3.9360, ept: 296.5672
      Epoch 23 composite train-obj: 0.730748
            No improvement (0.8328), counter 2/5
    Epoch [24/50], Train Losses: mse: 5.5909, mae: 0.9946, huber: 0.7230, swd: 3.8442, ept: 300.5988
    Epoch [24/50], Val Losses: mse: 6.4902, mae: 1.0498, huber: 0.7785, swd: 4.4817, ept: 296.3116
    Epoch [24/50], Test Losses: mse: 5.8864, mae: 1.0576, huber: 0.7809, swd: 4.0951, ept: 296.5359
      Epoch 24 composite train-obj: 0.723045
            No improvement (0.7785), counter 3/5
    Epoch [25/50], Train Losses: mse: 5.5972, mae: 0.9816, huber: 0.7152, swd: 3.8566, ept: 300.6062
    Epoch [25/50], Val Losses: mse: 6.5267, mae: 1.0298, huber: 0.7681, swd: 4.4861, ept: 296.1770
    Epoch [25/50], Test Losses: mse: 5.9161, mae: 1.0376, huber: 0.7706, swd: 4.0963, ept: 296.0133
      Epoch 25 composite train-obj: 0.715232
            Val objective improved 0.7753 → 0.7681, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 5.5812, mae: 0.9801, huber: 0.7140, swd: 3.8459, ept: 300.7768
    Epoch [26/50], Val Losses: mse: 6.4594, mae: 1.0303, huber: 0.7689, swd: 4.4728, ept: 296.3045
    Epoch [26/50], Test Losses: mse: 5.8491, mae: 1.0372, huber: 0.7694, swd: 4.0808, ept: 295.9527
      Epoch 26 composite train-obj: 0.714018
            No improvement (0.7689), counter 1/5
    Epoch [27/50], Train Losses: mse: 5.5980, mae: 0.9833, huber: 0.7167, swd: 3.8594, ept: 300.5012
    Epoch [27/50], Val Losses: mse: 6.3576, mae: 1.0571, huber: 0.7814, swd: 4.4111, ept: 296.6621
    Epoch [27/50], Test Losses: mse: 5.7534, mae: 1.0634, huber: 0.7804, swd: 4.0256, ept: 296.7384
      Epoch 27 composite train-obj: 0.716695
            No improvement (0.7814), counter 2/5
    Epoch [28/50], Train Losses: mse: 5.5621, mae: 0.9829, huber: 0.7150, swd: 3.8365, ept: 300.9753
    Epoch [28/50], Val Losses: mse: 6.6023, mae: 1.0307, huber: 0.7690, swd: 4.5179, ept: 296.3891
    Epoch [28/50], Test Losses: mse: 5.9925, mae: 1.0403, huber: 0.7741, swd: 4.1325, ept: 296.3773
      Epoch 28 composite train-obj: 0.715036
            No improvement (0.7690), counter 3/5
    Epoch [29/50], Train Losses: mse: 5.5866, mae: 0.9789, huber: 0.7127, swd: 3.8486, ept: 300.9047
    Epoch [29/50], Val Losses: mse: 6.4603, mae: 1.0299, huber: 0.7654, swd: 4.4337, ept: 296.7355
    Epoch [29/50], Test Losses: mse: 5.8541, mae: 1.0395, huber: 0.7684, swd: 4.0471, ept: 296.3264
      Epoch 29 composite train-obj: 0.712741
            Val objective improved 0.7681 → 0.7654, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 5.6825, mae: 0.9866, huber: 0.7202, swd: 3.9152, ept: 300.5593
    Epoch [30/50], Val Losses: mse: 6.2632, mae: 1.0552, huber: 0.7805, swd: 4.3774, ept: 297.1565
    Epoch [30/50], Test Losses: mse: 5.6623, mae: 1.0553, huber: 0.7741, swd: 3.9797, ept: 297.4630
      Epoch 30 composite train-obj: 0.720195
            No improvement (0.7805), counter 1/5
    Epoch [31/50], Train Losses: mse: 5.5728, mae: 0.9800, huber: 0.7130, swd: 3.8411, ept: 301.0440
    Epoch [31/50], Val Losses: mse: 6.6074, mae: 1.0289, huber: 0.7686, swd: 4.5410, ept: 296.5030
    Epoch [31/50], Test Losses: mse: 5.9979, mae: 1.0384, huber: 0.7732, swd: 4.1511, ept: 296.1591
      Epoch 31 composite train-obj: 0.712986
            No improvement (0.7686), counter 2/5
    Epoch [32/50], Train Losses: mse: 5.6110, mae: 0.9800, huber: 0.7143, swd: 3.8634, ept: 300.8111
    Epoch [32/50], Val Losses: mse: 6.0732, mae: 1.0892, huber: 0.7979, swd: 4.2315, ept: 297.3904
    Epoch [32/50], Test Losses: mse: 5.4766, mae: 1.0848, huber: 0.7870, swd: 3.8307, ept: 297.1476
      Epoch 32 composite train-obj: 0.714259
            No improvement (0.7979), counter 3/5
    Epoch [33/50], Train Losses: mse: 5.5708, mae: 0.9796, huber: 0.7128, swd: 3.8399, ept: 300.9402
    Epoch [33/50], Val Losses: mse: 6.9567, mae: 1.0431, huber: 0.7856, swd: 4.6755, ept: 296.3330
    Epoch [33/50], Test Losses: mse: 6.3528, mae: 1.0609, huber: 0.7987, swd: 4.3003, ept: 295.3462
      Epoch 33 composite train-obj: 0.712792
            No improvement (0.7856), counter 4/5
    Epoch [34/50], Train Losses: mse: 5.5866, mae: 0.9756, huber: 0.7105, swd: 3.8484, ept: 301.0679
    Epoch [34/50], Val Losses: mse: 6.3764, mae: 1.0305, huber: 0.7641, swd: 4.3889, ept: 297.0649
    Epoch [34/50], Test Losses: mse: 5.7687, mae: 1.0369, huber: 0.7649, swd: 4.0000, ept: 296.9153
      Epoch 34 composite train-obj: 0.710547
            Val objective improved 0.7654 → 0.7641, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 5.6853, mae: 0.9873, huber: 0.7218, swd: 3.9014, ept: 300.1552
    Epoch [35/50], Val Losses: mse: 6.1210, mae: 1.1664, huber: 0.8410, swd: 4.2831, ept: 298.5079
    Epoch [35/50], Test Losses: mse: 5.5385, mae: 1.1632, huber: 0.8316, swd: 3.8728, ept: 298.0727
      Epoch 35 composite train-obj: 0.721769
            No improvement (0.8410), counter 1/5
    Epoch [36/50], Train Losses: mse: 5.5782, mae: 0.9861, huber: 0.7172, swd: 3.8431, ept: 301.0951
    Epoch [36/50], Val Losses: mse: 6.5166, mae: 1.0299, huber: 0.7680, swd: 4.5097, ept: 296.8054
    Epoch [36/50], Test Losses: mse: 5.9065, mae: 1.0347, huber: 0.7687, swd: 4.1168, ept: 296.5571
      Epoch 36 composite train-obj: 0.717169
            No improvement (0.7680), counter 2/5
    Epoch [37/50], Train Losses: mse: 5.5658, mae: 0.9724, huber: 0.7080, swd: 3.8373, ept: 301.1972
    Epoch [37/50], Val Losses: mse: 6.3873, mae: 1.0248, huber: 0.7605, swd: 4.3882, ept: 297.2094
    Epoch [37/50], Test Losses: mse: 5.7797, mae: 1.0288, huber: 0.7603, swd: 3.9955, ept: 296.9814
      Epoch 37 composite train-obj: 0.708019
            Val objective improved 0.7641 → 0.7605, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 5.6377, mae: 0.9781, huber: 0.7135, swd: 3.9001, ept: 301.0106
    Epoch [38/50], Val Losses: mse: 6.3393, mae: 1.0465, huber: 0.7745, swd: 4.4275, ept: 297.3703
    Epoch [38/50], Test Losses: mse: 5.7390, mae: 1.0479, huber: 0.7700, swd: 4.0299, ept: 297.4072
      Epoch 38 composite train-obj: 0.713542
            No improvement (0.7745), counter 1/5
    Epoch [39/50], Train Losses: mse: 5.5610, mae: 0.9746, huber: 0.7091, swd: 3.8343, ept: 301.2662
    Epoch [39/50], Val Losses: mse: 6.3189, mae: 1.0373, huber: 0.7626, swd: 4.3486, ept: 297.6502
    Epoch [39/50], Test Losses: mse: 5.7136, mae: 1.0420, huber: 0.7616, swd: 3.9512, ept: 297.4183
      Epoch 39 composite train-obj: 0.709078
            No improvement (0.7626), counter 2/5
    Epoch [40/50], Train Losses: mse: 5.5580, mae: 0.9709, huber: 0.7065, swd: 3.8316, ept: 301.3098
    Epoch [40/50], Val Losses: mse: 6.4285, mae: 1.0268, huber: 0.7591, swd: 4.4206, ept: 297.2863
    Epoch [40/50], Test Losses: mse: 5.8212, mae: 1.0340, huber: 0.7605, swd: 4.0290, ept: 296.9455
      Epoch 40 composite train-obj: 0.706499
            Val objective improved 0.7605 → 0.7591, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 5.5621, mae: 0.9699, huber: 0.7058, swd: 3.8388, ept: 301.3344
    Epoch [41/50], Val Losses: mse: 6.4129, mae: 1.0224, huber: 0.7614, swd: 4.3976, ept: 297.2854
    Epoch [41/50], Test Losses: mse: 5.8104, mae: 1.0268, huber: 0.7626, swd: 4.0067, ept: 296.6559
      Epoch 41 composite train-obj: 0.705804
            No improvement (0.7614), counter 1/5
    Epoch [42/50], Train Losses: mse: 5.5744, mae: 0.9713, huber: 0.7068, swd: 3.8464, ept: 301.3261
    Epoch [42/50], Val Losses: mse: 6.6787, mae: 1.0254, huber: 0.7651, swd: 4.5544, ept: 296.9211
    Epoch [42/50], Test Losses: mse: 6.0711, mae: 1.0364, huber: 0.7722, swd: 4.1701, ept: 296.3841
      Epoch 42 composite train-obj: 0.706837
            No improvement (0.7651), counter 2/5
    Epoch [43/50], Train Losses: mse: 5.7164, mae: 0.9841, huber: 0.7195, swd: 3.9321, ept: 300.5447
    Epoch [43/50], Val Losses: mse: 6.1453, mae: 1.0666, huber: 0.7851, swd: 4.2930, ept: 298.0322
    Epoch [43/50], Test Losses: mse: 5.5578, mae: 1.0690, huber: 0.7807, swd: 3.9052, ept: 298.4109
      Epoch 43 composite train-obj: 0.719467
            No improvement (0.7851), counter 3/5
    Epoch [44/50], Train Losses: mse: 5.6603, mae: 0.9857, huber: 0.7201, swd: 3.9047, ept: 300.6464
    Epoch [44/50], Val Losses: mse: 6.1761, mae: 1.0679, huber: 0.7805, swd: 4.2924, ept: 298.1966
    Epoch [44/50], Test Losses: mse: 5.5810, mae: 1.0705, huber: 0.7761, swd: 3.8969, ept: 298.5182
      Epoch 44 composite train-obj: 0.720066
            No improvement (0.7805), counter 4/5
    Epoch [45/50], Train Losses: mse: 5.5411, mae: 0.9739, huber: 0.7086, swd: 3.8243, ept: 301.4655
    Epoch [45/50], Val Losses: mse: 6.3632, mae: 1.0249, huber: 0.7584, swd: 4.4015, ept: 297.3595
    Epoch [45/50], Test Losses: mse: 5.7543, mae: 1.0284, huber: 0.7568, swd: 4.0082, ept: 297.4493
      Epoch 45 composite train-obj: 0.708556
            Val objective improved 0.7591 → 0.7584, saving checkpoint.
    Epoch [46/50], Train Losses: mse: 5.5593, mae: 0.9691, huber: 0.7050, swd: 3.8352, ept: 301.4202
    Epoch [46/50], Val Losses: mse: 6.5581, mae: 1.0194, huber: 0.7606, swd: 4.4570, ept: 297.2079
    Epoch [46/50], Test Losses: mse: 5.9551, mae: 1.0293, huber: 0.7666, swd: 4.0754, ept: 296.7309
      Epoch 46 composite train-obj: 0.705049
            No improvement (0.7606), counter 1/5
    Epoch [47/50], Train Losses: mse: 5.5586, mae: 0.9680, huber: 0.7044, swd: 3.8351, ept: 301.4196
    Epoch [47/50], Val Losses: mse: 6.5654, mae: 1.0191, huber: 0.7624, swd: 4.4730, ept: 297.1205
    Epoch [47/50], Test Losses: mse: 5.9608, mae: 1.0272, huber: 0.7678, swd: 4.0888, ept: 296.4371
      Epoch 47 composite train-obj: 0.704441
            No improvement (0.7624), counter 2/5
    Epoch [48/50], Train Losses: mse: 5.5661, mae: 0.9677, huber: 0.7042, swd: 3.8424, ept: 301.4412
    Epoch [48/50], Val Losses: mse: 6.6185, mae: 1.0298, huber: 0.7691, swd: 4.5068, ept: 297.0965
    Epoch [48/50], Test Losses: mse: 6.0196, mae: 1.0431, huber: 0.7774, swd: 4.1315, ept: 296.5069
      Epoch 48 composite train-obj: 0.704247
            No improvement (0.7691), counter 3/5
    Epoch [49/50], Train Losses: mse: 5.5591, mae: 0.9671, huber: 0.7041, swd: 3.8384, ept: 301.4740
    Epoch [49/50], Val Losses: mse: 6.3681, mae: 1.0151, huber: 0.7546, swd: 4.3957, ept: 297.4241
    Epoch [49/50], Test Losses: mse: 5.7604, mae: 1.0217, huber: 0.7546, swd: 4.0036, ept: 297.1019
      Epoch 49 composite train-obj: 0.704057
            Val objective improved 0.7584 → 0.7546, saving checkpoint.
    Epoch [50/50], Train Losses: mse: 5.5573, mae: 0.9671, huber: 0.7035, swd: 3.8356, ept: 301.4819
    Epoch [50/50], Val Losses: mse: 6.4722, mae: 1.0132, huber: 0.7559, swd: 4.4263, ept: 297.1778
    Epoch [50/50], Test Losses: mse: 5.8641, mae: 1.0219, huber: 0.7598, swd: 4.0392, ept: 296.9183
      Epoch 50 composite train-obj: 0.703486
            No improvement (0.7559), counter 1/5
    Epoch [50/50], Test Losses: mse: 5.7604, mae: 1.0217, huber: 0.7546, swd: 4.0036, ept: 297.1019
    Best round's Test MSE: 5.7604, MAE: 1.0217, SWD: 4.0036
    Best round's Validation MSE: 6.3681, MAE: 1.0151, SWD: 4.3957
    Best round's Test verification MSE : 5.7604, MAE: 1.0217, SWD: 4.0036
    Time taken: 123.38 seconds
    
    ==================================================
    Experiment Summary (DLinear_rossler_seq96_pred336_20250513_1350)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 5.8070 ± 0.0399
      mae: 1.0322 ± 0.0111
      huber: 0.7618 ± 0.0081
      swd: 4.0110 ± 0.0220
      ept: 296.9948 ± 0.3268
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 6.4130 ± 0.0388
      mae: 1.0263 ± 0.0108
      huber: 0.7610 ± 0.0070
      swd: 4.4022 ± 0.0233
      ept: 297.0249 ± 0.5193
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 314.93 seconds
    
    Experiment complete: DLinear_rossler_seq96_pred336_20250513_1350
    Model: DLinear
    Dataset: rossler
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
    channels=data_mgr.datasets['rossler']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('rossler', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([36400, 3])
    Shape of validation data: torch.Size([5200, 3])
    Shape of testing data: torch.Size([10400, 3])
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([96, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 279
    Batch 0: Data shape torch.Size([128, 96, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 9.1040, mae: 1.7161, huber: 1.3810, swd: 5.2642, ept: 479.6871
    Epoch [1/50], Val Losses: mse: 9.4305, mae: 1.6795, huber: 1.3438, swd: 5.4987, ept: 539.8475
    Epoch [1/50], Test Losses: mse: 7.7290, mae: 1.5195, huber: 1.1870, swd: 4.5354, ept: 541.5589
      Epoch 1 composite train-obj: 1.381032
            Val objective improved inf → 1.3438, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.1813, mae: 1.5620, huber: 1.2274, swd: 4.7576, ept: 549.0254
    Epoch [2/50], Val Losses: mse: 8.8330, mae: 1.6925, huber: 1.3408, swd: 5.2616, ept: 537.5209
    Epoch [2/50], Test Losses: mse: 7.1993, mae: 1.5266, huber: 1.1768, swd: 4.3019, ept: 545.9513
      Epoch 2 composite train-obj: 1.227402
            Val objective improved 1.3438 → 1.3408, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.8368, mae: 1.5351, huber: 1.2001, swd: 4.5593, ept: 555.2291
    Epoch [3/50], Val Losses: mse: 8.9877, mae: 1.6497, huber: 1.3110, swd: 5.2468, ept: 542.6885
    Epoch [3/50], Test Losses: mse: 7.2926, mae: 1.4808, huber: 1.1466, swd: 4.2898, ept: 548.1864
      Epoch 3 composite train-obj: 1.200092
            Val objective improved 1.3408 → 1.3110, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.7353, mae: 1.5148, huber: 1.1827, swd: 4.4797, ept: 558.2531
    Epoch [4/50], Val Losses: mse: 8.9456, mae: 1.6336, huber: 1.2981, swd: 5.1594, ept: 547.4499
    Epoch [4/50], Test Losses: mse: 7.2373, mae: 1.4592, huber: 1.1286, swd: 4.2030, ept: 553.8417
      Epoch 4 composite train-obj: 1.182700
            Val objective improved 1.3110 → 1.2981, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.7512, mae: 1.5118, huber: 1.1810, swd: 4.4781, ept: 558.3079
    Epoch [5/50], Val Losses: mse: 8.6568, mae: 1.6741, huber: 1.3249, swd: 5.1360, ept: 536.9409
    Epoch [5/50], Test Losses: mse: 7.0736, mae: 1.5129, huber: 1.1667, swd: 4.2066, ept: 550.8287
      Epoch 5 composite train-obj: 1.180973
            No improvement (1.3249), counter 1/5
    Epoch [6/50], Train Losses: mse: 7.5957, mae: 1.5012, huber: 1.1699, swd: 4.3801, ept: 560.8190
    Epoch [6/50], Val Losses: mse: 8.7984, mae: 1.6335, huber: 1.2930, swd: 5.0661, ept: 548.0500
    Epoch [6/50], Test Losses: mse: 7.0696, mae: 1.4528, huber: 1.1182, swd: 4.1157, ept: 557.5186
      Epoch 6 composite train-obj: 1.169911
            Val objective improved 1.2981 → 1.2930, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.5795, mae: 1.4931, huber: 1.1634, swd: 4.3660, ept: 561.0116
    Epoch [7/50], Val Losses: mse: 8.7367, mae: 1.6474, huber: 1.3012, swd: 5.0040, ept: 542.9623
    Epoch [7/50], Test Losses: mse: 7.0810, mae: 1.4769, huber: 1.1351, swd: 4.0799, ept: 551.8695
      Epoch 7 composite train-obj: 1.163365
            No improvement (1.3012), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.6237, mae: 1.4968, huber: 1.1671, swd: 4.3813, ept: 560.2468
    Epoch [8/50], Val Losses: mse: 8.6584, mae: 1.6636, huber: 1.3063, swd: 5.0408, ept: 549.4144
    Epoch [8/50], Test Losses: mse: 6.9798, mae: 1.4926, huber: 1.1366, swd: 4.0678, ept: 558.2330
      Epoch 8 composite train-obj: 1.167097
            No improvement (1.3063), counter 2/5
    Epoch [9/50], Train Losses: mse: 7.5319, mae: 1.4871, huber: 1.1572, swd: 4.3295, ept: 562.8784
    Epoch [9/50], Val Losses: mse: 9.0682, mae: 1.6029, huber: 1.2761, swd: 5.1211, ept: 550.8130
    Epoch [9/50], Test Losses: mse: 7.3413, mae: 1.4381, huber: 1.1166, swd: 4.1701, ept: 556.3839
      Epoch 9 composite train-obj: 1.157165
            Val objective improved 1.2930 → 1.2761, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.5561, mae: 1.4765, huber: 1.1502, swd: 4.3462, ept: 563.2247
    Epoch [10/50], Val Losses: mse: 8.5738, mae: 1.6228, huber: 1.2841, swd: 5.0073, ept: 548.5953
    Epoch [10/50], Test Losses: mse: 6.8985, mae: 1.4475, huber: 1.1110, swd: 4.0750, ept: 560.4541
      Epoch 10 composite train-obj: 1.150176
            No improvement (1.2841), counter 1/5
    Epoch [11/50], Train Losses: mse: 7.5135, mae: 1.4740, huber: 1.1472, swd: 4.3095, ept: 564.3280
    Epoch [11/50], Val Losses: mse: 8.8414, mae: 1.5922, huber: 1.2649, swd: 5.0362, ept: 550.6519
    Epoch [11/50], Test Losses: mse: 7.1288, mae: 1.4231, huber: 1.1004, swd: 4.1040, ept: 558.2599
      Epoch 11 composite train-obj: 1.147220
            Val objective improved 1.2761 → 1.2649, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 7.5428, mae: 1.4712, huber: 1.1458, swd: 4.3313, ept: 563.7652
    Epoch [12/50], Val Losses: mse: 8.7742, mae: 1.6017, huber: 1.2680, swd: 4.9954, ept: 547.6702
    Epoch [12/50], Test Losses: mse: 7.0502, mae: 1.4293, huber: 1.1007, swd: 4.0472, ept: 558.7084
      Epoch 12 composite train-obj: 1.145776
            No improvement (1.2680), counter 1/5
    Epoch [13/50], Train Losses: mse: 7.5072, mae: 1.4678, huber: 1.1422, swd: 4.3004, ept: 563.5911
    Epoch [13/50], Val Losses: mse: 8.7158, mae: 1.6000, huber: 1.2696, swd: 5.0155, ept: 552.9781
    Epoch [13/50], Test Losses: mse: 6.9908, mae: 1.4231, huber: 1.0969, swd: 4.0723, ept: 562.7562
      Epoch 13 composite train-obj: 1.142248
            No improvement (1.2696), counter 2/5
    Epoch [14/50], Train Losses: mse: 7.5492, mae: 1.4716, huber: 1.1465, swd: 4.3209, ept: 563.3170
    Epoch [14/50], Val Losses: mse: 8.6353, mae: 1.5952, huber: 1.2653, swd: 4.9724, ept: 544.7845
    Epoch [14/50], Test Losses: mse: 6.9809, mae: 1.4320, huber: 1.1054, swd: 4.0211, ept: 553.6934
      Epoch 14 composite train-obj: 1.146483
            No improvement (1.2653), counter 3/5
    Epoch [15/50], Train Losses: mse: 7.4828, mae: 1.4665, huber: 1.1413, swd: 4.2893, ept: 564.0805
    Epoch [15/50], Val Losses: mse: 8.6926, mae: 1.5932, huber: 1.2624, swd: 4.9782, ept: 550.1699
    Epoch [15/50], Test Losses: mse: 6.9761, mae: 1.4155, huber: 1.0896, swd: 4.0467, ept: 560.4426
      Epoch 15 composite train-obj: 1.141257
            Val objective improved 1.2649 → 1.2624, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 7.4886, mae: 1.4614, huber: 1.1370, swd: 4.2918, ept: 564.7474
    Epoch [16/50], Val Losses: mse: 8.5641, mae: 1.5852, huber: 1.2549, swd: 4.9498, ept: 546.9388
    Epoch [16/50], Test Losses: mse: 6.9058, mae: 1.4128, huber: 1.0874, swd: 4.0168, ept: 557.5913
      Epoch 16 composite train-obj: 1.137019
            Val objective improved 1.2624 → 1.2549, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 7.5100, mae: 1.4634, huber: 1.1393, swd: 4.2987, ept: 563.4484
    Epoch [17/50], Val Losses: mse: 8.7838, mae: 1.5896, huber: 1.2640, swd: 5.0155, ept: 549.1068
    Epoch [17/50], Test Losses: mse: 7.0831, mae: 1.4226, huber: 1.1013, swd: 4.0540, ept: 557.0127
      Epoch 17 composite train-obj: 1.139329
            No improvement (1.2640), counter 1/5
    Epoch [18/50], Train Losses: mse: 7.4764, mae: 1.4602, huber: 1.1365, swd: 4.2798, ept: 564.4618
    Epoch [18/50], Val Losses: mse: 8.4031, mae: 1.6211, huber: 1.2762, swd: 4.8832, ept: 548.4312
    Epoch [18/50], Test Losses: mse: 6.7526, mae: 1.4425, huber: 1.1028, swd: 3.9556, ept: 560.6122
      Epoch 18 composite train-obj: 1.136477
            No improvement (1.2762), counter 2/5
    Epoch [19/50], Train Losses: mse: 7.4398, mae: 1.4587, huber: 1.1342, swd: 4.2625, ept: 565.1302
    Epoch [19/50], Val Losses: mse: 8.7027, mae: 1.5808, huber: 1.2538, swd: 5.0005, ept: 550.0570
    Epoch [19/50], Test Losses: mse: 6.9797, mae: 1.4079, huber: 1.0854, swd: 4.0704, ept: 560.8100
      Epoch 19 composite train-obj: 1.134214
            Val objective improved 1.2549 → 1.2538, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 7.4842, mae: 1.4560, huber: 1.1331, swd: 4.2888, ept: 564.2753
    Epoch [20/50], Val Losses: mse: 8.6366, mae: 1.6121, huber: 1.2726, swd: 4.9626, ept: 542.4247
    Epoch [20/50], Test Losses: mse: 6.9955, mae: 1.4478, huber: 1.1100, swd: 4.0179, ept: 555.6215
      Epoch 20 composite train-obj: 1.133116
            No improvement (1.2726), counter 1/5
    Epoch [21/50], Train Losses: mse: 7.5031, mae: 1.4632, huber: 1.1384, swd: 4.2969, ept: 562.4982
    Epoch [21/50], Val Losses: mse: 8.7977, mae: 1.5886, huber: 1.2625, swd: 4.9761, ept: 550.4536
    Epoch [21/50], Test Losses: mse: 7.0932, mae: 1.4166, huber: 1.0974, swd: 4.0289, ept: 556.9140
      Epoch 21 composite train-obj: 1.138418
            No improvement (1.2625), counter 2/5
    Epoch [22/50], Train Losses: mse: 7.4679, mae: 1.4555, huber: 1.1324, swd: 4.2708, ept: 564.9538
    Epoch [22/50], Val Losses: mse: 8.7321, mae: 1.5843, huber: 1.2553, swd: 5.0098, ept: 551.2211
    Epoch [22/50], Test Losses: mse: 6.9955, mae: 1.4083, huber: 1.0843, swd: 4.0628, ept: 560.9558
      Epoch 22 composite train-obj: 1.132427
            No improvement (1.2553), counter 3/5
    Epoch [23/50], Train Losses: mse: 7.4690, mae: 1.4528, huber: 1.1302, swd: 4.2800, ept: 564.7705
    Epoch [23/50], Val Losses: mse: 8.5303, mae: 1.5990, huber: 1.2616, swd: 4.9470, ept: 543.7631
    Epoch [23/50], Test Losses: mse: 6.8965, mae: 1.4360, huber: 1.0995, swd: 3.9951, ept: 555.9156
      Epoch 23 composite train-obj: 1.130161
            No improvement (1.2616), counter 4/5
    Epoch [24/50], Train Losses: mse: 7.4495, mae: 1.4550, huber: 1.1314, swd: 4.2636, ept: 564.3357
    Epoch [24/50], Val Losses: mse: 8.7015, mae: 1.5772, huber: 1.2493, swd: 4.9877, ept: 550.3737
    Epoch [24/50], Test Losses: mse: 6.9771, mae: 1.4057, huber: 1.0818, swd: 4.0410, ept: 559.4498
      Epoch 24 composite train-obj: 1.131384
            Val objective improved 1.2538 → 1.2493, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 7.4549, mae: 1.4493, huber: 1.1274, swd: 4.2643, ept: 565.9692
    Epoch [25/50], Val Losses: mse: 8.7574, mae: 1.5723, huber: 1.2479, swd: 5.0147, ept: 551.6364
    Epoch [25/50], Test Losses: mse: 7.0292, mae: 1.4007, huber: 1.0804, swd: 4.0648, ept: 561.5999
      Epoch 25 composite train-obj: 1.127405
            Val objective improved 1.2493 → 1.2479, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 7.4653, mae: 1.4519, huber: 1.1293, swd: 4.2751, ept: 565.2033
    Epoch [26/50], Val Losses: mse: 8.4885, mae: 1.5837, huber: 1.2537, swd: 4.9503, ept: 547.1141
    Epoch [26/50], Test Losses: mse: 6.8448, mae: 1.4124, huber: 1.0848, swd: 4.0246, ept: 557.8657
      Epoch 26 composite train-obj: 1.129288
            No improvement (1.2537), counter 1/5
    Epoch [27/50], Train Losses: mse: 7.4386, mae: 1.4514, huber: 1.1287, swd: 4.2663, ept: 566.1201
    Epoch [27/50], Val Losses: mse: 8.8924, mae: 1.5817, huber: 1.2567, swd: 5.0408, ept: 557.0291
    Epoch [27/50], Test Losses: mse: 7.1180, mae: 1.4072, huber: 1.0880, swd: 4.0783, ept: 564.6276
      Epoch 27 composite train-obj: 1.128734
            No improvement (1.2567), counter 2/5
    Epoch [28/50], Train Losses: mse: 7.4622, mae: 1.4495, huber: 1.1275, swd: 4.2700, ept: 566.1327
    Epoch [28/50], Val Losses: mse: 8.4156, mae: 1.6226, huber: 1.2749, swd: 4.8947, ept: 551.0467
    Epoch [28/50], Test Losses: mse: 6.7486, mae: 1.4438, huber: 1.1002, swd: 3.9524, ept: 561.5689
      Epoch 28 composite train-obj: 1.127549
            No improvement (1.2749), counter 3/5
    Epoch [29/50], Train Losses: mse: 7.4319, mae: 1.4523, huber: 1.1291, swd: 4.2525, ept: 566.0005
    Epoch [29/50], Val Losses: mse: 8.7452, mae: 1.5743, huber: 1.2472, swd: 4.9774, ept: 552.5972
    Epoch [29/50], Test Losses: mse: 7.0171, mae: 1.4033, huber: 1.0805, swd: 4.0451, ept: 562.8568
      Epoch 29 composite train-obj: 1.129144
            Val objective improved 1.2479 → 1.2472, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 7.4452, mae: 1.4456, huber: 1.1242, swd: 4.2650, ept: 566.8983
    Epoch [30/50], Val Losses: mse: 8.7762, mae: 1.5732, huber: 1.2493, swd: 5.0013, ept: 550.4579
    Epoch [30/50], Test Losses: mse: 7.0445, mae: 1.4019, huber: 1.0819, swd: 4.0696, ept: 558.5941
      Epoch 30 composite train-obj: 1.124162
            No improvement (1.2493), counter 1/5
    Epoch [31/50], Train Losses: mse: 7.4460, mae: 1.4466, huber: 1.1246, swd: 4.2664, ept: 566.7043
    Epoch [31/50], Val Losses: mse: 8.8025, mae: 1.5770, huber: 1.2505, swd: 5.0084, ept: 553.8573
    Epoch [31/50], Test Losses: mse: 7.0558, mae: 1.4007, huber: 1.0806, swd: 4.0671, ept: 562.7885
      Epoch 31 composite train-obj: 1.124580
            No improvement (1.2505), counter 2/5
    Epoch [32/50], Train Losses: mse: 7.4493, mae: 1.4458, huber: 1.1240, swd: 4.2680, ept: 566.7847
    Epoch [32/50], Val Losses: mse: 8.7367, mae: 1.5721, huber: 1.2476, swd: 4.9673, ept: 549.9031
    Epoch [32/50], Test Losses: mse: 7.0233, mae: 1.4007, huber: 1.0814, swd: 4.0237, ept: 558.8982
      Epoch 32 composite train-obj: 1.123952
            No improvement (1.2476), counter 3/5
    Epoch [33/50], Train Losses: mse: 7.4783, mae: 1.4459, huber: 1.1246, swd: 4.2845, ept: 567.1404
    Epoch [33/50], Val Losses: mse: 8.7578, mae: 1.5763, huber: 1.2491, swd: 4.9710, ept: 551.6011
    Epoch [33/50], Test Losses: mse: 7.0178, mae: 1.4050, huber: 1.0818, swd: 4.0406, ept: 560.5088
      Epoch 33 composite train-obj: 1.124552
            No improvement (1.2491), counter 4/5
    Epoch [34/50], Train Losses: mse: 7.4710, mae: 1.4465, huber: 1.1247, swd: 4.2811, ept: 567.0049
    Epoch [34/50], Val Losses: mse: 8.6404, mae: 1.5809, huber: 1.2504, swd: 4.9428, ept: 552.8208
    Epoch [34/50], Test Losses: mse: 6.9291, mae: 1.4075, huber: 1.0815, swd: 4.0012, ept: 560.4516
      Epoch 34 composite train-obj: 1.124735
    Epoch [34/50], Test Losses: mse: 7.0171, mae: 1.4033, huber: 1.0805, swd: 4.0451, ept: 562.8568
    Best round's Test MSE: 7.0171, MAE: 1.4033, SWD: 4.0451
    Best round's Validation MSE: 8.7452, MAE: 1.5743, SWD: 4.9774
    Best round's Test verification MSE : 7.0171, MAE: 1.4033, SWD: 4.0451
    Time taken: 86.91 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.0850, mae: 1.7127, huber: 1.3776, swd: 5.0949, ept: 483.4019
    Epoch [1/50], Val Losses: mse: 9.5818, mae: 1.6861, huber: 1.3533, swd: 5.3651, ept: 538.1034
    Epoch [1/50], Test Losses: mse: 7.8635, mae: 1.5262, huber: 1.1974, swd: 4.4244, ept: 536.1117
      Epoch 1 composite train-obj: 1.377554
            Val objective improved inf → 1.3533, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.0517, mae: 1.5504, huber: 1.2159, swd: 4.5612, ept: 551.2294
    Epoch [2/50], Val Losses: mse: 8.8811, mae: 1.6869, huber: 1.3364, swd: 5.1002, ept: 541.2109
    Epoch [2/50], Test Losses: mse: 7.1974, mae: 1.5072, huber: 1.1619, swd: 4.1612, ept: 549.4837
      Epoch 2 composite train-obj: 1.215932
            Val objective improved 1.3533 → 1.3364, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.8214, mae: 1.5305, huber: 1.1960, swd: 4.4330, ept: 556.2044
    Epoch [3/50], Val Losses: mse: 9.0447, mae: 1.6496, huber: 1.3118, swd: 5.1230, ept: 546.8564
    Epoch [3/50], Test Losses: mse: 7.3134, mae: 1.4741, huber: 1.1418, swd: 4.1769, ept: 552.0997
      Epoch 3 composite train-obj: 1.196035
            Val objective improved 1.3364 → 1.3118, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.7144, mae: 1.5109, huber: 1.1791, swd: 4.3722, ept: 559.1773
    Epoch [4/50], Val Losses: mse: 8.8701, mae: 1.6328, huber: 1.2938, swd: 4.9998, ept: 545.0594
    Epoch [4/50], Test Losses: mse: 7.1845, mae: 1.4644, huber: 1.1287, swd: 4.0648, ept: 552.6153
      Epoch 4 composite train-obj: 1.179052
            Val objective improved 1.3118 → 1.2938, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.6460, mae: 1.5009, huber: 1.1704, swd: 4.3269, ept: 560.4849
    Epoch [5/50], Val Losses: mse: 8.9281, mae: 1.6175, huber: 1.2850, swd: 5.0231, ept: 547.5063
    Epoch [5/50], Test Losses: mse: 7.2069, mae: 1.4459, huber: 1.1189, swd: 4.0935, ept: 555.2238
      Epoch 5 composite train-obj: 1.170369
            Val objective improved 1.2938 → 1.2850, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.6134, mae: 1.4921, huber: 1.1628, swd: 4.3013, ept: 561.7649
    Epoch [6/50], Val Losses: mse: 8.7952, mae: 1.6174, huber: 1.2826, swd: 4.9677, ept: 547.9004
    Epoch [6/50], Test Losses: mse: 7.0967, mae: 1.4454, huber: 1.1148, swd: 4.0544, ept: 557.9066
      Epoch 6 composite train-obj: 1.162775
            Val objective improved 1.2850 → 1.2826, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.5724, mae: 1.4865, huber: 1.1579, swd: 4.2762, ept: 562.2741
    Epoch [7/50], Val Losses: mse: 8.8143, mae: 1.6286, huber: 1.2878, swd: 4.9348, ept: 549.0603
    Epoch [7/50], Test Losses: mse: 7.0792, mae: 1.4512, huber: 1.1159, swd: 4.0001, ept: 559.4654
      Epoch 7 composite train-obj: 1.157947
            No improvement (1.2878), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.5761, mae: 1.4856, huber: 1.1569, swd: 4.2771, ept: 562.5840
    Epoch [8/50], Val Losses: mse: 8.8410, mae: 1.6133, huber: 1.2796, swd: 4.9573, ept: 550.7518
    Epoch [8/50], Test Losses: mse: 7.1242, mae: 1.4401, huber: 1.1116, swd: 4.0370, ept: 558.1600
      Epoch 8 composite train-obj: 1.156925
            Val objective improved 1.2826 → 1.2796, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.5424, mae: 1.4781, huber: 1.1510, swd: 4.2519, ept: 563.1994
    Epoch [9/50], Val Losses: mse: 8.9645, mae: 1.6022, huber: 1.2734, swd: 4.9944, ept: 551.2846
    Epoch [9/50], Test Losses: mse: 7.2391, mae: 1.4337, huber: 1.1098, swd: 4.0780, ept: 558.4406
      Epoch 9 composite train-obj: 1.150999
            Val objective improved 1.2796 → 1.2734, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.5560, mae: 1.4749, huber: 1.1487, swd: 4.2656, ept: 563.6825
    Epoch [10/50], Val Losses: mse: 8.7615, mae: 1.5943, huber: 1.2650, swd: 4.9381, ept: 545.9348
    Epoch [10/50], Test Losses: mse: 7.0732, mae: 1.4296, huber: 1.1040, swd: 4.0266, ept: 553.3178
      Epoch 10 composite train-obj: 1.148666
            Val objective improved 1.2734 → 1.2650, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 7.5241, mae: 1.4709, huber: 1.1452, swd: 4.2383, ept: 563.7543
    Epoch [11/50], Val Losses: mse: 8.8606, mae: 1.5909, huber: 1.2626, swd: 4.9403, ept: 552.3956
    Epoch [11/50], Test Losses: mse: 7.1330, mae: 1.4215, huber: 1.0974, swd: 4.0180, ept: 559.6429
      Epoch 11 composite train-obj: 1.145226
            Val objective improved 1.2650 → 1.2626, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 7.5209, mae: 1.4674, huber: 1.1422, swd: 4.2347, ept: 563.5653
    Epoch [12/50], Val Losses: mse: 8.6232, mae: 1.6134, huber: 1.2769, swd: 4.9229, ept: 554.7014
    Epoch [12/50], Test Losses: mse: 6.9147, mae: 1.4336, huber: 1.1006, swd: 3.9872, ept: 563.6446
      Epoch 12 composite train-obj: 1.142152
            No improvement (1.2769), counter 1/5
    Epoch [13/50], Train Losses: mse: 7.5213, mae: 1.4692, huber: 1.1439, swd: 4.2400, ept: 564.0062
    Epoch [13/50], Val Losses: mse: 8.6020, mae: 1.5874, huber: 1.2599, swd: 4.8493, ept: 543.6691
    Epoch [13/50], Test Losses: mse: 6.9736, mae: 1.4228, huber: 1.0999, swd: 3.9350, ept: 552.8471
      Epoch 13 composite train-obj: 1.143913
            Val objective improved 1.2626 → 1.2599, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 7.4894, mae: 1.4656, huber: 1.1407, swd: 4.2104, ept: 563.9132
    Epoch [14/50], Val Losses: mse: 8.9781, mae: 1.5862, huber: 1.2601, swd: 5.0121, ept: 551.7968
    Epoch [14/50], Test Losses: mse: 7.2101, mae: 1.4189, huber: 1.0970, swd: 4.0835, ept: 560.9880
      Epoch 14 composite train-obj: 1.140675
            No improvement (1.2601), counter 1/5
    Epoch [15/50], Train Losses: mse: 7.4993, mae: 1.4621, huber: 1.1377, swd: 4.2233, ept: 564.6691
    Epoch [15/50], Val Losses: mse: 8.9297, mae: 1.5910, huber: 1.2648, swd: 4.9784, ept: 550.5320
    Epoch [15/50], Test Losses: mse: 7.1854, mae: 1.4233, huber: 1.1004, swd: 4.0683, ept: 560.3197
      Epoch 15 composite train-obj: 1.137748
            No improvement (1.2648), counter 2/5
    Epoch [16/50], Train Losses: mse: 7.4831, mae: 1.4581, huber: 1.1344, swd: 4.2145, ept: 564.2463
    Epoch [16/50], Val Losses: mse: 8.7719, mae: 1.5851, huber: 1.2602, swd: 4.9184, ept: 551.9213
    Epoch [16/50], Test Losses: mse: 7.0337, mae: 1.4063, huber: 1.0872, swd: 3.9998, ept: 560.4273
      Epoch 16 composite train-obj: 1.134423
            No improvement (1.2602), counter 3/5
    Epoch [17/50], Train Losses: mse: 7.4805, mae: 1.4587, huber: 1.1348, swd: 4.2085, ept: 565.1059
    Epoch [17/50], Val Losses: mse: 8.7575, mae: 1.5811, huber: 1.2549, swd: 4.9080, ept: 552.9710
    Epoch [17/50], Test Losses: mse: 7.0369, mae: 1.4070, huber: 1.0866, swd: 3.9921, ept: 561.4110
      Epoch 17 composite train-obj: 1.134842
            Val objective improved 1.2599 → 1.2549, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 7.4819, mae: 1.4567, huber: 1.1332, swd: 4.2112, ept: 564.4583
    Epoch [18/50], Val Losses: mse: 8.4890, mae: 1.6077, huber: 1.2713, swd: 4.8443, ept: 548.6490
    Epoch [18/50], Test Losses: mse: 6.8167, mae: 1.4286, huber: 1.0954, swd: 3.9413, ept: 562.1024
      Epoch 18 composite train-obj: 1.133210
            No improvement (1.2713), counter 1/5
    Epoch [19/50], Train Losses: mse: 7.4547, mae: 1.4566, huber: 1.1330, swd: 4.1954, ept: 565.2050
    Epoch [19/50], Val Losses: mse: 8.7484, mae: 1.5818, huber: 1.2535, swd: 4.9076, ept: 551.7465
    Epoch [19/50], Test Losses: mse: 7.0166, mae: 1.4109, huber: 1.0860, swd: 3.9789, ept: 562.0469
      Epoch 19 composite train-obj: 1.133048
            Val objective improved 1.2549 → 1.2535, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 7.4625, mae: 1.4532, huber: 1.1301, swd: 4.2029, ept: 564.8540
    Epoch [20/50], Val Losses: mse: 8.8263, mae: 1.5792, huber: 1.2544, swd: 4.9303, ept: 551.5328
    Epoch [20/50], Test Losses: mse: 7.1232, mae: 1.4111, huber: 1.0912, swd: 4.0168, ept: 559.9944
      Epoch 20 composite train-obj: 1.130147
            No improvement (1.2544), counter 1/5
    Epoch [21/50], Train Losses: mse: 7.4646, mae: 1.4526, huber: 1.1295, swd: 4.2016, ept: 565.5469
    Epoch [21/50], Val Losses: mse: 8.7024, mae: 1.5811, huber: 1.2530, swd: 4.9049, ept: 551.8877
    Epoch [21/50], Test Losses: mse: 6.9789, mae: 1.4056, huber: 1.0823, swd: 3.9918, ept: 563.0610
      Epoch 21 composite train-obj: 1.129525
            Val objective improved 1.2535 → 1.2530, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 7.4631, mae: 1.4499, huber: 1.1277, swd: 4.2065, ept: 565.6661
    Epoch [22/50], Val Losses: mse: 8.7519, mae: 1.5797, huber: 1.2518, swd: 4.9080, ept: 551.9914
    Epoch [22/50], Test Losses: mse: 7.0230, mae: 1.4034, huber: 1.0813, swd: 3.9757, ept: 561.4012
      Epoch 22 composite train-obj: 1.127695
            Val objective improved 1.2530 → 1.2518, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 7.4783, mae: 1.4523, huber: 1.1294, swd: 4.2148, ept: 566.0175
    Epoch [23/50], Val Losses: mse: 8.8450, mae: 1.5850, huber: 1.2551, swd: 4.9422, ept: 550.0613
    Epoch [23/50], Test Losses: mse: 7.0875, mae: 1.4163, huber: 1.0899, swd: 4.0244, ept: 560.5057
      Epoch 23 composite train-obj: 1.129365
            No improvement (1.2551), counter 1/5
    Epoch [24/50], Train Losses: mse: 7.4552, mae: 1.4496, huber: 1.1269, swd: 4.1950, ept: 566.2083
    Epoch [24/50], Val Losses: mse: 8.9369, mae: 1.5760, huber: 1.2542, swd: 4.9691, ept: 552.4219
    Epoch [24/50], Test Losses: mse: 7.1896, mae: 1.4076, huber: 1.0908, swd: 4.0497, ept: 558.9078
      Epoch 24 composite train-obj: 1.126857
            No improvement (1.2542), counter 2/5
    Epoch [25/50], Train Losses: mse: 7.4630, mae: 1.4480, huber: 1.1262, swd: 4.2010, ept: 566.8720
    Epoch [25/50], Val Losses: mse: 8.5909, mae: 1.5755, huber: 1.2483, swd: 4.8704, ept: 551.9891
    Epoch [25/50], Test Losses: mse: 6.8821, mae: 1.3993, huber: 1.0762, swd: 3.9555, ept: 562.0553
      Epoch 25 composite train-obj: 1.126215
            Val objective improved 1.2518 → 1.2483, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 7.4778, mae: 1.4491, huber: 1.1273, swd: 4.2123, ept: 566.3283
    Epoch [26/50], Val Losses: mse: 8.5415, mae: 1.5810, huber: 1.2532, swd: 4.8147, ept: 546.4843
    Epoch [26/50], Test Losses: mse: 6.8925, mae: 1.4127, huber: 1.0901, swd: 3.8997, ept: 554.9083
      Epoch 26 composite train-obj: 1.127299
            No improvement (1.2532), counter 1/5
    Epoch [27/50], Train Losses: mse: 7.4451, mae: 1.4485, huber: 1.1264, swd: 4.1868, ept: 566.7755
    Epoch [27/50], Val Losses: mse: 8.6220, mae: 1.5771, huber: 1.2481, swd: 4.8557, ept: 553.4758
    Epoch [27/50], Test Losses: mse: 6.8864, mae: 1.3994, huber: 1.0755, swd: 3.9276, ept: 563.5766
      Epoch 27 composite train-obj: 1.126402
            Val objective improved 1.2483 → 1.2481, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 7.4468, mae: 1.4460, huber: 1.1242, swd: 4.1922, ept: 567.3408
    Epoch [28/50], Val Losses: mse: 8.6034, mae: 1.5683, huber: 1.2419, swd: 4.8313, ept: 549.9993
    Epoch [28/50], Test Losses: mse: 6.9253, mae: 1.4000, huber: 1.0780, swd: 3.9199, ept: 558.5424
      Epoch 28 composite train-obj: 1.124247
            Val objective improved 1.2481 → 1.2419, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 7.4404, mae: 1.4456, huber: 1.1240, swd: 4.1918, ept: 567.4643
    Epoch [29/50], Val Losses: mse: 8.8184, mae: 1.5816, huber: 1.2549, swd: 4.9432, ept: 554.1937
    Epoch [29/50], Test Losses: mse: 7.0876, mae: 1.4067, huber: 1.0869, swd: 3.9991, ept: 561.6223
      Epoch 29 composite train-obj: 1.123973
            No improvement (1.2549), counter 1/5
    Epoch [30/50], Train Losses: mse: 7.4772, mae: 1.4461, huber: 1.1249, swd: 4.2156, ept: 567.4307
    Epoch [30/50], Val Losses: mse: 8.7833, mae: 1.5741, huber: 1.2459, swd: 4.9167, ept: 553.5472
    Epoch [30/50], Test Losses: mse: 7.0517, mae: 1.4048, huber: 1.0811, swd: 3.9961, ept: 563.7682
      Epoch 30 composite train-obj: 1.124854
            No improvement (1.2459), counter 2/5
    Epoch [31/50], Train Losses: mse: 7.4572, mae: 1.4459, huber: 1.1238, swd: 4.1922, ept: 567.1331
    Epoch [31/50], Val Losses: mse: 8.8760, mae: 1.5673, huber: 1.2471, swd: 4.9555, ept: 553.4111
    Epoch [31/50], Test Losses: mse: 7.1231, mae: 1.3957, huber: 1.0808, swd: 4.0344, ept: 560.1651
      Epoch 31 composite train-obj: 1.123810
            No improvement (1.2471), counter 3/5
    Epoch [32/50], Train Losses: mse: 7.4531, mae: 1.4439, huber: 1.1227, swd: 4.1926, ept: 567.3512
    Epoch [32/50], Val Losses: mse: 8.6303, mae: 1.5699, huber: 1.2431, swd: 4.8783, ept: 552.9392
    Epoch [32/50], Test Losses: mse: 6.9168, mae: 1.3951, huber: 1.0721, swd: 3.9533, ept: 562.8202
      Epoch 32 composite train-obj: 1.122668
            No improvement (1.2431), counter 4/5
    Epoch [33/50], Train Losses: mse: 7.4385, mae: 1.4420, huber: 1.1210, swd: 4.1896, ept: 567.2062
    Epoch [33/50], Val Losses: mse: 8.6473, mae: 1.5898, huber: 1.2599, swd: 4.9111, ept: 554.2640
    Epoch [33/50], Test Losses: mse: 6.9370, mae: 1.4142, huber: 1.0878, swd: 4.0026, ept: 566.4084
      Epoch 33 composite train-obj: 1.120961
    Epoch [33/50], Test Losses: mse: 6.9253, mae: 1.4000, huber: 1.0780, swd: 3.9199, ept: 558.5424
    Best round's Test MSE: 6.9253, MAE: 1.4000, SWD: 3.9199
    Best round's Validation MSE: 8.6034, MAE: 1.5683, SWD: 4.8313
    Best round's Test verification MSE : 6.9253, MAE: 1.4000, SWD: 3.9199
    Time taken: 84.32 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.0683, mae: 1.7122, huber: 1.3773, swd: 5.2569, ept: 480.1367
    Epoch [1/50], Val Losses: mse: 9.5510, mae: 1.6812, huber: 1.3469, swd: 5.5732, ept: 539.6136
    Epoch [1/50], Test Losses: mse: 7.8186, mae: 1.5194, huber: 1.1892, swd: 4.5786, ept: 541.2990
      Epoch 1 composite train-obj: 1.377331
            Val objective improved inf → 1.3469, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.0414, mae: 1.5499, huber: 1.2156, swd: 4.7101, ept: 552.0150
    Epoch [2/50], Val Losses: mse: 9.2245, mae: 1.6579, huber: 1.3213, swd: 5.3799, ept: 542.2047
    Epoch [2/50], Test Losses: mse: 7.5185, mae: 1.4920, huber: 1.1599, swd: 4.3845, ept: 547.8504
      Epoch 2 composite train-obj: 1.215621
            Val objective improved 1.3469 → 1.3213, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.9274, mae: 1.5355, huber: 1.2020, swd: 4.6274, ept: 554.7158
    Epoch [3/50], Val Losses: mse: 8.8042, mae: 1.6574, huber: 1.3153, swd: 5.2388, ept: 542.5307
    Epoch [3/50], Test Losses: mse: 7.1507, mae: 1.4859, huber: 1.1468, swd: 4.2747, ept: 550.1324
      Epoch 3 composite train-obj: 1.202047
            Val objective improved 1.3213 → 1.3153, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.7110, mae: 1.5165, huber: 1.1837, swd: 4.5060, ept: 557.9407
    Epoch [4/50], Val Losses: mse: 8.8317, mae: 1.6311, huber: 1.2951, swd: 5.1996, ept: 544.5021
    Epoch [4/50], Test Losses: mse: 7.1587, mae: 1.4578, huber: 1.1266, swd: 4.2283, ept: 553.5447
      Epoch 4 composite train-obj: 1.183729
            Val objective improved 1.3153 → 1.2951, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.6869, mae: 1.5049, huber: 1.1742, swd: 4.4873, ept: 559.6611
    Epoch [5/50], Val Losses: mse: 8.7920, mae: 1.6304, huber: 1.2933, swd: 5.1250, ept: 543.5004
    Epoch [5/50], Test Losses: mse: 7.1289, mae: 1.4606, huber: 1.1286, swd: 4.1512, ept: 549.9648
      Epoch 5 composite train-obj: 1.174212
            Val objective improved 1.2951 → 1.2933, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.5920, mae: 1.4950, huber: 1.1648, swd: 4.4228, ept: 561.4677
    Epoch [6/50], Val Losses: mse: 8.8627, mae: 1.6158, huber: 1.2833, swd: 5.1376, ept: 548.2875
    Epoch [6/50], Test Losses: mse: 7.1700, mae: 1.4448, huber: 1.1174, swd: 4.1863, ept: 555.8580
      Epoch 6 composite train-obj: 1.164821
            Val objective improved 1.2933 → 1.2833, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.5810, mae: 1.4860, huber: 1.1578, swd: 4.4085, ept: 562.7207
    Epoch [7/50], Val Losses: mse: 8.8098, mae: 1.6356, huber: 1.2924, swd: 5.0884, ept: 552.2782
    Epoch [7/50], Test Losses: mse: 7.0786, mae: 1.4560, huber: 1.1187, swd: 4.1251, ept: 559.2876
      Epoch 7 composite train-obj: 1.157767
            No improvement (1.2924), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.5856, mae: 1.4901, huber: 1.1610, swd: 4.4017, ept: 561.8875
    Epoch [8/50], Val Losses: mse: 8.8204, mae: 1.6003, huber: 1.2724, swd: 5.1274, ept: 545.9243
    Epoch [8/50], Test Losses: mse: 7.1561, mae: 1.4379, huber: 1.1141, swd: 4.1480, ept: 552.2355
      Epoch 8 composite train-obj: 1.161011
            Val objective improved 1.2833 → 1.2724, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.6022, mae: 1.4859, huber: 1.1584, swd: 4.4067, ept: 562.0045
    Epoch [9/50], Val Losses: mse: 8.6167, mae: 1.6053, huber: 1.2721, swd: 5.0851, ept: 543.7788
    Epoch [9/50], Test Losses: mse: 6.9866, mae: 1.4364, huber: 1.1075, swd: 4.1221, ept: 553.2229
      Epoch 9 composite train-obj: 1.158386
            Val objective improved 1.2724 → 1.2721, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.5095, mae: 1.4761, huber: 1.1493, swd: 4.3628, ept: 563.4906
    Epoch [10/50], Val Losses: mse: 8.7485, mae: 1.5986, huber: 1.2661, swd: 5.0903, ept: 549.7863
    Epoch [10/50], Test Losses: mse: 7.0355, mae: 1.4271, huber: 1.0992, swd: 4.1233, ept: 558.3703
      Epoch 10 composite train-obj: 1.149269
            Val objective improved 1.2721 → 1.2661, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 7.5360, mae: 1.4721, huber: 1.1462, swd: 4.3674, ept: 563.6056
    Epoch [11/50], Val Losses: mse: 8.5870, mae: 1.6206, huber: 1.2795, swd: 5.0313, ept: 547.4388
    Epoch [11/50], Test Losses: mse: 6.9322, mae: 1.4560, huber: 1.1138, swd: 4.0657, ept: 558.8425
      Epoch 11 composite train-obj: 1.146168
            No improvement (1.2795), counter 1/5
    Epoch [12/50], Train Losses: mse: 7.4935, mae: 1.4713, huber: 1.1447, swd: 4.3464, ept: 562.4152
    Epoch [12/50], Val Losses: mse: 8.8941, mae: 1.5999, huber: 1.2713, swd: 5.1007, ept: 549.2211
    Epoch [12/50], Test Losses: mse: 7.1867, mae: 1.4285, huber: 1.1059, swd: 4.1563, ept: 556.2797
      Epoch 12 composite train-obj: 1.144674
            No improvement (1.2713), counter 2/5
    Epoch [13/50], Train Losses: mse: 7.5077, mae: 1.4658, huber: 1.1411, swd: 4.3492, ept: 564.5854
    Epoch [13/50], Val Losses: mse: 8.7469, mae: 1.5867, huber: 1.2590, swd: 5.0707, ept: 547.8652
    Epoch [13/50], Test Losses: mse: 7.0712, mae: 1.4200, huber: 1.0963, swd: 4.1148, ept: 558.3760
      Epoch 13 composite train-obj: 1.141084
            Val objective improved 1.2661 → 1.2590, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 7.5253, mae: 1.4657, huber: 1.1412, swd: 4.3614, ept: 564.4109
    Epoch [14/50], Val Losses: mse: 8.6546, mae: 1.5854, huber: 1.2569, swd: 5.0166, ept: 543.7598
    Epoch [14/50], Test Losses: mse: 7.0197, mae: 1.4254, huber: 1.1003, swd: 4.0648, ept: 554.0069
      Epoch 14 composite train-obj: 1.141249
            Val objective improved 1.2590 → 1.2569, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 7.5025, mae: 1.4631, huber: 1.1390, swd: 4.3347, ept: 564.1844
    Epoch [15/50], Val Losses: mse: 8.7585, mae: 1.5951, huber: 1.2629, swd: 5.1019, ept: 554.1456
    Epoch [15/50], Test Losses: mse: 7.0217, mae: 1.4181, huber: 1.0910, swd: 4.1202, ept: 563.2778
      Epoch 15 composite train-obj: 1.138967
            No improvement (1.2629), counter 1/5
    Epoch [16/50], Train Losses: mse: 7.4936, mae: 1.4600, huber: 1.1363, swd: 4.3387, ept: 564.7521
    Epoch [16/50], Val Losses: mse: 8.7840, mae: 1.5934, huber: 1.2634, swd: 5.0911, ept: 554.2454
    Epoch [16/50], Test Losses: mse: 7.0625, mae: 1.4183, huber: 1.0941, swd: 4.1122, ept: 562.2929
      Epoch 16 composite train-obj: 1.136271
            No improvement (1.2634), counter 2/5
    Epoch [17/50], Train Losses: mse: 7.5054, mae: 1.4608, huber: 1.1368, swd: 4.3544, ept: 564.6863
    Epoch [17/50], Val Losses: mse: 8.9881, mae: 1.5863, huber: 1.2623, swd: 5.1252, ept: 548.8796
    Epoch [17/50], Test Losses: mse: 7.2692, mae: 1.4244, huber: 1.1047, swd: 4.1715, ept: 556.4367
      Epoch 17 composite train-obj: 1.136759
            No improvement (1.2623), counter 3/5
    Epoch [18/50], Train Losses: mse: 7.5800, mae: 1.4659, huber: 1.1425, swd: 4.3711, ept: 562.7698
    Epoch [18/50], Val Losses: mse: 8.4743, mae: 1.5954, huber: 1.2605, swd: 5.0068, ept: 541.0278
    Epoch [18/50], Test Losses: mse: 6.8780, mae: 1.4327, huber: 1.0993, swd: 4.0640, ept: 552.0813
      Epoch 18 composite train-obj: 1.142487
            No improvement (1.2605), counter 4/5
    Epoch [19/50], Train Losses: mse: 7.4617, mae: 1.4593, huber: 1.1353, swd: 4.3210, ept: 564.2884
    Epoch [19/50], Val Losses: mse: 8.6860, mae: 1.5816, huber: 1.2531, swd: 5.0548, ept: 551.9031
    Epoch [19/50], Test Losses: mse: 6.9749, mae: 1.4081, huber: 1.0837, swd: 4.0901, ept: 562.1452
      Epoch 19 composite train-obj: 1.135282
            Val objective improved 1.2569 → 1.2531, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 7.4656, mae: 1.4543, huber: 1.1312, swd: 4.3215, ept: 565.4246
    Epoch [20/50], Val Losses: mse: 8.9766, mae: 1.5803, huber: 1.2566, swd: 5.1453, ept: 553.8253
    Epoch [20/50], Test Losses: mse: 7.2136, mae: 1.4102, huber: 1.0916, swd: 4.1611, ept: 560.2760
      Epoch 20 composite train-obj: 1.131198
            No improvement (1.2566), counter 1/5
    Epoch [21/50], Train Losses: mse: 7.4814, mae: 1.4531, huber: 1.1304, swd: 4.3287, ept: 565.0831
    Epoch [21/50], Val Losses: mse: 8.6854, mae: 1.5916, huber: 1.2584, swd: 5.0361, ept: 550.0581
    Epoch [21/50], Test Losses: mse: 6.9750, mae: 1.4183, huber: 1.0901, swd: 4.0609, ept: 558.8447
      Epoch 21 composite train-obj: 1.130398
            No improvement (1.2584), counter 2/5
    Epoch [22/50], Train Losses: mse: 7.5243, mae: 1.4606, huber: 1.1374, swd: 4.3480, ept: 563.4606
    Epoch [22/50], Val Losses: mse: 8.4442, mae: 1.6099, huber: 1.2726, swd: 5.0182, ept: 548.0885
    Epoch [22/50], Test Losses: mse: 6.7833, mae: 1.4335, huber: 1.0991, swd: 4.0629, ept: 559.7641
      Epoch 22 composite train-obj: 1.137379
            No improvement (1.2726), counter 3/5
    Epoch [23/50], Train Losses: mse: 7.4374, mae: 1.4543, huber: 1.1311, swd: 4.3083, ept: 565.3441
    Epoch [23/50], Val Losses: mse: 8.6273, mae: 1.6047, huber: 1.2655, swd: 4.9964, ept: 547.5993
    Epoch [23/50], Test Losses: mse: 6.9685, mae: 1.4443, huber: 1.1041, swd: 4.0318, ept: 559.4029
      Epoch 23 composite train-obj: 1.131073
            No improvement (1.2655), counter 4/5
    Epoch [24/50], Train Losses: mse: 7.5668, mae: 1.4653, huber: 1.1415, swd: 4.3649, ept: 562.6392
    Epoch [24/50], Val Losses: mse: 8.4652, mae: 1.5905, huber: 1.2601, swd: 4.9998, ept: 547.6632
    Epoch [24/50], Test Losses: mse: 6.8193, mae: 1.4197, huber: 1.0924, swd: 4.0654, ept: 560.1599
      Epoch 24 composite train-obj: 1.141489
    Epoch [24/50], Test Losses: mse: 6.9749, mae: 1.4081, huber: 1.0837, swd: 4.0901, ept: 562.1452
    Best round's Test MSE: 6.9749, MAE: 1.4081, SWD: 4.0901
    Best round's Validation MSE: 8.6860, MAE: 1.5816, SWD: 5.0548
    Best round's Test verification MSE : 6.9749, MAE: 1.4081, SWD: 4.0901
    Time taken: 61.38 seconds
    
    ==================================================
    Experiment Summary (DLinear_rossler_seq96_pred720_20250513_1355)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 6.9724 ± 0.0375
      mae: 1.4038 ± 0.0033
      huber: 1.0807 ± 0.0024
      swd: 4.0184 ± 0.0720
      ept: 561.1814 ± 1.8886
      count: 35.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 8.6782 ± 0.0581
      mae: 1.5747 ± 0.0054
      huber: 1.2474 ± 0.0045
      swd: 4.9545 ± 0.0927
      ept: 551.4999 ± 1.0982
      count: 35.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 232.64 seconds
    
    Experiment complete: DLinear_rossler_seq96_pred720_20250513_1355
    Model: DLinear
    Dataset: rossler
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    








