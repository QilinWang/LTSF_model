# data


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

# Initialize the data manager
data_mgr = DatasetManager(device='cuda')

# Load a synthetic dataset
data_mgr.load_trajectory('lorenz96', steps=18999, dt=1e-2) # 50399
# SCALE = False
# trajectory = utils.generate_trajectory('lorenz',steps=52200, dt=1e-2) 
# trajectory = utils.generate_hyperchaotic_rossler(steps=12000, dt=1e-3)
# trajectory_2 = utils.generate_henon(steps=52000) 
```



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
    channels=data_mgr.datasets['lorenz96']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([96, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([96, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 98
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 96, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 98
    Validation Batches: 9
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.2984, mae: 2.3328, huber: 1.8860, swd: 3.6799, ept: 44.0008
    Epoch [1/50], Val Losses: mse: 7.7761, mae: 2.0725, huber: 1.6355, swd: 1.8307, ept: 57.7672
    Epoch [1/50], Test Losses: mse: 8.1469, mae: 2.0908, huber: 1.6559, swd: 2.2358, ept: 57.8358
      Epoch 1 composite train-obj: 1.886001
            Val objective improved inf → 1.6355, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.1373, mae: 1.4282, huber: 1.0231, swd: 0.9751, ept: 75.1300
    Epoch [2/50], Val Losses: mse: 7.1328, mae: 1.9305, huber: 1.5026, swd: 1.4334, ept: 65.7005
    Epoch [2/50], Test Losses: mse: 7.3520, mae: 1.9283, huber: 1.5031, swd: 1.6488, ept: 65.1809
      Epoch 2 composite train-obj: 1.023134
            Val objective improved 1.6355 → 1.5026, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.4326, mae: 1.0717, huber: 0.6941, swd: 0.5005, ept: 84.9767
    Epoch [3/50], Val Losses: mse: 7.2324, mae: 1.9447, huber: 1.5169, swd: 1.7387, ept: 66.5177
    Epoch [3/50], Test Losses: mse: 7.3928, mae: 1.9075, huber: 1.4887, swd: 1.9193, ept: 65.9296
      Epoch 3 composite train-obj: 0.694067
            No improvement (1.5169), counter 1/5
    Epoch [4/50], Train Losses: mse: 1.5607, mae: 0.8524, huber: 0.4992, swd: 0.3050, ept: 89.8303
    Epoch [4/50], Val Losses: mse: 7.7127, mae: 1.9502, huber: 1.5303, swd: 1.4216, ept: 67.4865
    Epoch [4/50], Test Losses: mse: 7.4898, mae: 1.8991, huber: 1.4821, swd: 1.6012, ept: 67.8639
      Epoch 4 composite train-obj: 0.499177
            No improvement (1.5303), counter 2/5
    Epoch [5/50], Train Losses: mse: 1.1114, mae: 0.7242, huber: 0.3889, swd: 0.2051, ept: 92.2236
    Epoch [5/50], Val Losses: mse: 7.1731, mae: 1.8850, huber: 1.4670, swd: 1.4618, ept: 68.6042
    Epoch [5/50], Test Losses: mse: 7.7247, mae: 1.9063, huber: 1.4938, swd: 1.6884, ept: 67.6437
      Epoch 5 composite train-obj: 0.388948
            Val objective improved 1.5026 → 1.4670, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.8387, mae: 0.6335, huber: 0.3138, swd: 0.1457, ept: 93.6066
    Epoch [6/50], Val Losses: mse: 7.3263, mae: 1.8931, huber: 1.4750, swd: 1.4305, ept: 68.9546
    Epoch [6/50], Test Losses: mse: 7.7157, mae: 1.9018, huber: 1.4890, swd: 1.6240, ept: 68.4514
      Epoch 6 composite train-obj: 0.313819
            No improvement (1.4750), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.6484, mae: 0.5619, huber: 0.2568, swd: 0.1128, ept: 94.4049
    Epoch [7/50], Val Losses: mse: 6.9351, mae: 1.8464, huber: 1.4312, swd: 1.4081, ept: 69.8663
    Epoch [7/50], Test Losses: mse: 7.4064, mae: 1.8595, huber: 1.4495, swd: 1.6209, ept: 69.4712
      Epoch 7 composite train-obj: 0.256821
            Val objective improved 1.4670 → 1.4312, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.5259, mae: 0.5123, huber: 0.2183, swd: 0.0928, ept: 94.9856
    Epoch [8/50], Val Losses: mse: 7.0658, mae: 1.8704, huber: 1.4531, swd: 1.4994, ept: 69.6892
    Epoch [8/50], Test Losses: mse: 7.6951, mae: 1.8899, huber: 1.4805, swd: 1.6842, ept: 68.6249
      Epoch 8 composite train-obj: 0.218278
            No improvement (1.4531), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4329, mae: 0.4661, huber: 0.1851, swd: 0.0716, ept: 95.3000
    Epoch [9/50], Val Losses: mse: 7.2149, mae: 1.8673, huber: 1.4525, swd: 1.4193, ept: 70.5545
    Epoch [9/50], Test Losses: mse: 7.6259, mae: 1.8842, huber: 1.4741, swd: 1.6544, ept: 68.9761
      Epoch 9 composite train-obj: 0.185074
            No improvement (1.4525), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.3426, mae: 0.4178, huber: 0.1516, swd: 0.0559, ept: 95.5382
    Epoch [10/50], Val Losses: mse: 7.1233, mae: 1.8761, huber: 1.4590, swd: 1.3900, ept: 70.1217
    Epoch [10/50], Test Losses: mse: 7.5835, mae: 1.8784, huber: 1.4686, swd: 1.6331, ept: 69.5491
      Epoch 10 composite train-obj: 0.151613
            No improvement (1.4590), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.3117, mae: 0.4013, huber: 0.1404, swd: 0.0526, ept: 95.6533
    Epoch [11/50], Val Losses: mse: 7.2230, mae: 1.8751, huber: 1.4592, swd: 1.3270, ept: 70.0609
    Epoch [11/50], Test Losses: mse: 7.5367, mae: 1.8666, huber: 1.4585, swd: 1.5771, ept: 70.0379
      Epoch 11 composite train-obj: 0.140391
            No improvement (1.4592), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.2576, mae: 0.3662, huber: 0.1183, swd: 0.0431, ept: 95.7802
    Epoch [12/50], Val Losses: mse: 7.0568, mae: 1.8585, huber: 1.4440, swd: 1.3032, ept: 70.3578
    Epoch [12/50], Test Losses: mse: 7.3245, mae: 1.8428, huber: 1.4357, swd: 1.5467, ept: 70.5553
      Epoch 12 composite train-obj: 0.118259
    Epoch [12/50], Test Losses: mse: 7.4063, mae: 1.8595, huber: 1.4496, swd: 1.6210, ept: 69.4688
    Best round's Test MSE: 7.4064, MAE: 1.8595, SWD: 1.6209
    Best round's Validation MSE: 6.9351, MAE: 1.8464, SWD: 1.4081
    Best round's Test verification MSE : 7.4063, MAE: 1.8595, SWD: 1.6210
    Time taken: 36.22 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.4010, mae: 2.3449, huber: 1.8979, swd: 3.7598, ept: 45.0856
    Epoch [1/50], Val Losses: mse: 7.5915, mae: 2.0678, huber: 1.6283, swd: 1.7555, ept: 57.4746
    Epoch [1/50], Test Losses: mse: 8.4049, mae: 2.1027, huber: 1.6681, swd: 2.2511, ept: 59.0656
      Epoch 1 composite train-obj: 1.897885
            Val objective improved inf → 1.6283, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.4109, mae: 1.4806, huber: 1.0716, swd: 1.0820, ept: 74.0604
    Epoch [2/50], Val Losses: mse: 7.6642, mae: 1.9930, huber: 1.5622, swd: 1.4236, ept: 64.0020
    Epoch [2/50], Test Losses: mse: 7.9912, mae: 1.9886, huber: 1.5627, swd: 1.7157, ept: 63.8026
      Epoch 2 composite train-obj: 1.071606
            Val objective improved 1.6283 → 1.5622, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.5592, mae: 1.1068, huber: 0.7249, swd: 0.5519, ept: 84.2760
    Epoch [3/50], Val Losses: mse: 6.7358, mae: 1.8589, huber: 1.4342, swd: 1.3006, ept: 68.7080
    Epoch [3/50], Test Losses: mse: 7.4017, mae: 1.9003, huber: 1.4799, swd: 1.6784, ept: 67.0182
      Epoch 3 composite train-obj: 0.724887
            Val objective improved 1.5622 → 1.4342, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 1.6464, mae: 0.8901, huber: 0.5294, swd: 0.3299, ept: 89.2802
    Epoch [4/50], Val Losses: mse: 7.2326, mae: 1.9188, huber: 1.4937, swd: 1.4586, ept: 68.1508
    Epoch [4/50], Test Losses: mse: 7.2090, mae: 1.8669, huber: 1.4490, swd: 1.6981, ept: 67.7408
      Epoch 4 composite train-obj: 0.529368
            No improvement (1.4937), counter 1/5
    Epoch [5/50], Train Losses: mse: 1.1727, mae: 0.7539, huber: 0.4112, swd: 0.2221, ept: 91.9104
    Epoch [5/50], Val Losses: mse: 7.0832, mae: 1.8889, huber: 1.4662, swd: 1.3231, ept: 69.1517
    Epoch [5/50], Test Losses: mse: 7.4813, mae: 1.8913, huber: 1.4749, swd: 1.6150, ept: 67.7939
      Epoch 5 composite train-obj: 0.411179
            No improvement (1.4662), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.8701, mae: 0.6511, huber: 0.3262, swd: 0.1571, ept: 93.5044
    Epoch [6/50], Val Losses: mse: 7.2559, mae: 1.9101, huber: 1.4879, swd: 1.3369, ept: 69.1450
    Epoch [6/50], Test Losses: mse: 7.3361, mae: 1.8627, huber: 1.4482, swd: 1.5834, ept: 68.7396
      Epoch 6 composite train-obj: 0.326175
            No improvement (1.4879), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.7089, mae: 0.5914, huber: 0.2781, swd: 0.1240, ept: 94.2783
    Epoch [7/50], Val Losses: mse: 7.0389, mae: 1.8828, huber: 1.4625, swd: 1.3001, ept: 69.7107
    Epoch [7/50], Test Losses: mse: 7.3277, mae: 1.8675, huber: 1.4524, swd: 1.5987, ept: 68.7332
      Epoch 7 composite train-obj: 0.278148
            No improvement (1.4625), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.5468, mae: 0.5220, huber: 0.2250, swd: 0.0909, ept: 94.9227
    Epoch [8/50], Val Losses: mse: 6.9547, mae: 1.8715, huber: 1.4517, swd: 1.2624, ept: 70.0544
    Epoch [8/50], Test Losses: mse: 7.3487, mae: 1.8638, huber: 1.4507, swd: 1.6249, ept: 69.0464
      Epoch 8 composite train-obj: 0.225018
    Epoch [8/50], Test Losses: mse: 7.4024, mae: 1.9004, huber: 1.4800, swd: 1.6785, ept: 67.0243
    Best round's Test MSE: 7.4017, MAE: 1.9003, SWD: 1.6784
    Best round's Validation MSE: 6.7358, MAE: 1.8589, SWD: 1.3006
    Best round's Test verification MSE : 7.4024, MAE: 1.9004, SWD: 1.6785
    Time taken: 23.78 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.2928, mae: 2.3335, huber: 1.8868, swd: 3.3285, ept: 45.0792
    Epoch [1/50], Val Losses: mse: 7.4834, mae: 2.0364, huber: 1.5986, swd: 1.4650, ept: 59.7755
    Epoch [1/50], Test Losses: mse: 8.1231, mae: 2.0701, huber: 1.6358, swd: 1.8123, ept: 60.3285
      Epoch 1 composite train-obj: 1.886772
            Val objective improved inf → 1.5986, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.2722, mae: 1.4590, huber: 1.0521, swd: 0.9421, ept: 74.7716
    Epoch [2/50], Val Losses: mse: 6.9834, mae: 1.9211, huber: 1.4910, swd: 1.3379, ept: 65.9042
    Epoch [2/50], Test Losses: mse: 7.5223, mae: 1.9379, huber: 1.5136, swd: 1.6371, ept: 64.6761
      Epoch 2 composite train-obj: 1.052101
            Val objective improved 1.5986 → 1.4910, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.5809, mae: 1.1184, huber: 0.7357, swd: 0.5170, ept: 83.9301
    Epoch [3/50], Val Losses: mse: 7.1865, mae: 1.9188, huber: 1.4936, swd: 1.3149, ept: 66.9760
    Epoch [3/50], Test Losses: mse: 7.2188, mae: 1.8654, huber: 1.4492, swd: 1.5112, ept: 67.6605
      Epoch 3 composite train-obj: 0.735679
            No improvement (1.4936), counter 1/5
    Epoch [4/50], Train Losses: mse: 1.6613, mae: 0.8943, huber: 0.5336, swd: 0.3122, ept: 89.2393
    Epoch [4/50], Val Losses: mse: 8.0174, mae: 1.9918, huber: 1.5687, swd: 1.1716, ept: 66.2703
    Epoch [4/50], Test Losses: mse: 7.7392, mae: 1.9338, huber: 1.5138, swd: 1.3910, ept: 66.3559
      Epoch 4 composite train-obj: 0.533621
            No improvement (1.5687), counter 2/5
    Epoch [5/50], Train Losses: mse: 1.2128, mae: 0.7642, huber: 0.4213, swd: 0.2123, ept: 91.6453
    Epoch [5/50], Val Losses: mse: 7.4560, mae: 1.9124, huber: 1.4930, swd: 1.1812, ept: 68.3532
    Epoch [5/50], Test Losses: mse: 7.3151, mae: 1.8691, huber: 1.4530, swd: 1.3742, ept: 69.0318
      Epoch 5 composite train-obj: 0.421250
            No improvement (1.4930), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.9042, mae: 0.6614, huber: 0.3354, swd: 0.1525, ept: 93.3225
    Epoch [6/50], Val Losses: mse: 7.1217, mae: 1.8965, huber: 1.4757, swd: 1.1292, ept: 67.9137
    Epoch [6/50], Test Losses: mse: 7.4371, mae: 1.8729, huber: 1.4581, swd: 1.3281, ept: 69.3045
      Epoch 6 composite train-obj: 0.335365
            Val objective improved 1.4910 → 1.4757, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.7295, mae: 0.6006, huber: 0.2853, swd: 0.1189, ept: 94.2255
    Epoch [7/50], Val Losses: mse: 7.3474, mae: 1.8946, huber: 1.4751, swd: 1.2002, ept: 70.0743
    Epoch [7/50], Test Losses: mse: 7.2453, mae: 1.8475, huber: 1.4343, swd: 1.3317, ept: 69.8783
      Epoch 7 composite train-obj: 0.285345
            Val objective improved 1.4757 → 1.4751, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.5930, mae: 0.5449, huber: 0.2422, swd: 0.0975, ept: 94.8889
    Epoch [8/50], Val Losses: mse: 7.2121, mae: 1.8629, huber: 1.4464, swd: 1.1329, ept: 70.8170
    Epoch [8/50], Test Losses: mse: 7.2730, mae: 1.8428, huber: 1.4318, swd: 1.3606, ept: 70.3812
      Epoch 8 composite train-obj: 0.242239
            Val objective improved 1.4751 → 1.4464, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4591, mae: 0.4825, huber: 0.1956, swd: 0.0705, ept: 95.3222
    Epoch [9/50], Val Losses: mse: 7.1724, mae: 1.8732, huber: 1.4557, swd: 1.1327, ept: 70.8151
    Epoch [9/50], Test Losses: mse: 7.0994, mae: 1.8157, huber: 1.4068, swd: 1.3678, ept: 70.8826
      Epoch 9 composite train-obj: 0.195590
            No improvement (1.4557), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.3937, mae: 0.4512, huber: 0.1726, swd: 0.0638, ept: 95.5344
    Epoch [10/50], Val Losses: mse: 7.1750, mae: 1.8605, huber: 1.4433, swd: 1.1511, ept: 71.3434
    Epoch [10/50], Test Losses: mse: 7.1278, mae: 1.8256, huber: 1.4154, swd: 1.3422, ept: 70.7514
      Epoch 10 composite train-obj: 0.172617
            Val objective improved 1.4464 → 1.4433, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.3335, mae: 0.4158, huber: 0.1492, swd: 0.0505, ept: 95.6622
    Epoch [11/50], Val Losses: mse: 7.1121, mae: 1.8658, huber: 1.4484, swd: 1.1015, ept: 70.3291
    Epoch [11/50], Test Losses: mse: 7.1086, mae: 1.8212, huber: 1.4117, swd: 1.3444, ept: 70.5994
      Epoch 11 composite train-obj: 0.149168
            No improvement (1.4484), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.3177, mae: 0.4082, huber: 0.1438, swd: 0.0499, ept: 95.7448
    Epoch [12/50], Val Losses: mse: 7.0375, mae: 1.8494, huber: 1.4332, swd: 1.0901, ept: 71.0224
    Epoch [12/50], Test Losses: mse: 7.1028, mae: 1.8131, huber: 1.4050, swd: 1.3522, ept: 70.7164
      Epoch 12 composite train-obj: 0.143826
            Val objective improved 1.4433 → 1.4332, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.2647, mae: 0.3744, huber: 0.1221, swd: 0.0431, ept: 95.8340
    Epoch [13/50], Val Losses: mse: 7.0349, mae: 1.8445, huber: 1.4296, swd: 1.1254, ept: 71.0177
    Epoch [13/50], Test Losses: mse: 7.0989, mae: 1.8077, huber: 1.4008, swd: 1.3585, ept: 70.9285
      Epoch 13 composite train-obj: 0.122079
            Val objective improved 1.4332 → 1.4296, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.2301, mae: 0.3495, huber: 0.1074, swd: 0.0369, ept: 95.8789
    Epoch [14/50], Val Losses: mse: 6.8989, mae: 1.8377, huber: 1.4216, swd: 1.1080, ept: 71.3223
    Epoch [14/50], Test Losses: mse: 7.0950, mae: 1.8110, huber: 1.4039, swd: 1.3716, ept: 71.1980
      Epoch 14 composite train-obj: 0.107422
            Val objective improved 1.4296 → 1.4216, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.2384, mae: 0.3572, huber: 0.1117, swd: 0.0388, ept: 95.9195
    Epoch [15/50], Val Losses: mse: 7.2876, mae: 1.8683, huber: 1.4522, swd: 1.1317, ept: 71.2677
    Epoch [15/50], Test Losses: mse: 7.1566, mae: 1.8201, huber: 1.4119, swd: 1.3497, ept: 70.9472
      Epoch 15 composite train-obj: 0.111721
            No improvement (1.4522), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.2019, mae: 0.3289, huber: 0.0956, swd: 0.0347, ept: 95.9367
    Epoch [16/50], Val Losses: mse: 7.1842, mae: 1.8550, huber: 1.4389, swd: 1.1134, ept: 71.7563
    Epoch [16/50], Test Losses: mse: 7.0731, mae: 1.8040, huber: 1.3978, swd: 1.3348, ept: 71.2625
      Epoch 16 composite train-obj: 0.095617
            No improvement (1.4389), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.1818, mae: 0.3130, huber: 0.0869, swd: 0.0298, ept: 95.9691
    Epoch [17/50], Val Losses: mse: 6.9067, mae: 1.8383, huber: 1.4223, swd: 1.1254, ept: 71.7871
    Epoch [17/50], Test Losses: mse: 7.0337, mae: 1.7956, huber: 1.3903, swd: 1.3914, ept: 71.5626
      Epoch 17 composite train-obj: 0.086851
            No improvement (1.4223), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.1583, mae: 0.2936, huber: 0.0764, swd: 0.0273, ept: 95.9845
    Epoch [18/50], Val Losses: mse: 7.0243, mae: 1.8437, huber: 1.4284, swd: 1.1091, ept: 72.0378
    Epoch [18/50], Test Losses: mse: 7.0093, mae: 1.8003, huber: 1.3938, swd: 1.3448, ept: 71.3416
      Epoch 18 composite train-obj: 0.076375
            No improvement (1.4284), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.1546, mae: 0.2907, huber: 0.0748, swd: 0.0282, ept: 95.9883
    Epoch [19/50], Val Losses: mse: 7.0432, mae: 1.8471, huber: 1.4311, swd: 1.1320, ept: 72.2461
    Epoch [19/50], Test Losses: mse: 6.9435, mae: 1.7869, huber: 1.3817, swd: 1.3824, ept: 71.8627
      Epoch 19 composite train-obj: 0.074792
    Epoch [19/50], Test Losses: mse: 7.0950, mae: 1.8110, huber: 1.4039, swd: 1.3716, ept: 71.1957
    Best round's Test MSE: 7.0950, MAE: 1.8110, SWD: 1.3716
    Best round's Validation MSE: 6.8989, MAE: 1.8377, SWD: 1.1080
    Best round's Test verification MSE : 7.0950, MAE: 1.8110, SWD: 1.3716
    Time taken: 55.60 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz96_seq720_pred96_20250512_2241)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 7.3010 ± 0.1457
      mae: 1.8569 ± 0.0365
      huber: 1.4445 ± 0.0312
      swd: 1.5570 ± 0.1331
      ept: 69.2292 ± 1.7149
      count: 9.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 6.8566 ± 0.0867
      mae: 1.8477 ± 0.0087
      huber: 1.4290 ± 0.0054
      swd: 1.2722 ± 0.1241
      ept: 69.9655 ± 1.0696
      count: 9.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 115.70 seconds
    
    Experiment complete: ACL_lorenz96_seq720_pred96_20250512_2241
    Model: ACL
    Dataset: lorenz96
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
    channels=data_mgr.datasets['lorenz96']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([196, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([196, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 97
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 196, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 97
    Validation Batches: 8
    Test Batches: 23
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.1428, mae: 2.6357, huber: 2.1781, swd: 3.9533, ept: 46.6369
    Epoch [1/50], Val Losses: mse: 8.9655, mae: 2.3491, huber: 1.8963, swd: 2.4248, ept: 65.7723
    Epoch [1/50], Test Losses: mse: 10.9340, mae: 2.5459, huber: 2.0928, swd: 2.9200, ept: 61.7943
      Epoch 1 composite train-obj: 2.178070
            Val objective improved inf → 1.8963, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.0147, mae: 1.9513, huber: 1.5179, swd: 1.4359, ept: 95.1711
    Epoch [2/50], Val Losses: mse: 9.3668, mae: 2.3385, huber: 1.8922, swd: 1.6891, ept: 74.4586
    Epoch [2/50], Test Losses: mse: 10.8368, mae: 2.4569, huber: 2.0113, swd: 2.1937, ept: 78.1849
      Epoch 2 composite train-obj: 1.517852
            Val objective improved 1.8963 → 1.8922, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.9312, mae: 1.5777, huber: 1.1625, swd: 0.8174, ept: 122.1114
    Epoch [3/50], Val Losses: mse: 10.7243, mae: 2.4816, huber: 2.0320, swd: 1.3216, ept: 77.9039
    Epoch [3/50], Test Losses: mse: 10.8291, mae: 2.4199, huber: 1.9784, swd: 1.7424, ept: 83.4213
      Epoch 3 composite train-obj: 1.162550
            No improvement (2.0320), counter 1/5
    Epoch [4/50], Train Losses: mse: 3.5620, mae: 1.3232, huber: 0.9222, swd: 0.5165, ept: 140.1638
    Epoch [4/50], Val Losses: mse: 11.3011, mae: 2.5210, huber: 2.0734, swd: 1.1364, ept: 78.6537
    Epoch [4/50], Test Losses: mse: 11.3180, mae: 2.4576, huber: 2.0167, swd: 1.4323, ept: 85.6027
      Epoch 4 composite train-obj: 0.922166
            No improvement (2.0734), counter 2/5
    Epoch [5/50], Train Losses: mse: 2.5564, mae: 1.1152, huber: 0.7293, swd: 0.3398, ept: 155.6883
    Epoch [5/50], Val Losses: mse: 11.6074, mae: 2.5431, huber: 2.0965, swd: 1.1595, ept: 82.5164
    Epoch [5/50], Test Losses: mse: 11.3716, mae: 2.4503, huber: 2.0114, swd: 1.4015, ept: 87.1884
      Epoch 5 composite train-obj: 0.729346
            No improvement (2.0965), counter 3/5
    Epoch [6/50], Train Losses: mse: 1.9593, mae: 0.9790, huber: 0.6053, swd: 0.2420, ept: 165.8321
    Epoch [6/50], Val Losses: mse: 11.9271, mae: 2.5714, huber: 2.1246, swd: 1.0168, ept: 83.0942
    Epoch [6/50], Test Losses: mse: 11.4673, mae: 2.4536, huber: 2.0145, swd: 1.2786, ept: 89.6860
      Epoch 6 composite train-obj: 0.605337
            No improvement (2.1246), counter 4/5
    Epoch [7/50], Train Losses: mse: 1.5273, mae: 0.8673, huber: 0.5061, swd: 0.1840, ept: 173.4698
    Epoch [7/50], Val Losses: mse: 12.3033, mae: 2.6068, huber: 2.1595, swd: 1.0280, ept: 80.7344
    Epoch [7/50], Test Losses: mse: 11.6957, mae: 2.4742, huber: 2.0355, swd: 1.2335, ept: 88.9244
      Epoch 7 composite train-obj: 0.506055
    Epoch [7/50], Test Losses: mse: 10.8363, mae: 2.4568, huber: 2.0112, swd: 2.1936, ept: 78.1807
    Best round's Test MSE: 10.8368, MAE: 2.4569, SWD: 2.1937
    Best round's Validation MSE: 9.3668, MAE: 2.3385, SWD: 1.6891
    Best round's Test verification MSE : 10.8363, MAE: 2.4568, SWD: 2.1936
    Time taken: 21.10 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.9092, mae: 2.5948, huber: 2.1388, swd: 3.7377, ept: 49.4699
    Epoch [1/50], Val Losses: mse: 9.0343, mae: 2.3470, huber: 1.8961, swd: 2.5841, ept: 65.9547
    Epoch [1/50], Test Losses: mse: 11.0412, mae: 2.5496, huber: 2.0975, swd: 3.0470, ept: 63.6927
      Epoch 1 composite train-obj: 2.138751
            Val objective improved inf → 1.8961, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.7096, mae: 1.8903, huber: 1.4606, swd: 1.3279, ept: 101.8694
    Epoch [2/50], Val Losses: mse: 9.6346, mae: 2.3252, huber: 1.8816, swd: 1.1584, ept: 81.2272
    Epoch [2/50], Test Losses: mse: 11.3370, mae: 2.4802, huber: 2.0358, swd: 1.5754, ept: 78.9551
      Epoch 2 composite train-obj: 1.460580
            Val objective improved 1.8961 → 1.8816, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.5702, mae: 1.5052, huber: 1.0948, swd: 0.7546, ept: 128.7214
    Epoch [3/50], Val Losses: mse: 10.6320, mae: 2.4288, huber: 1.9818, swd: 1.0248, ept: 84.6750
    Epoch [3/50], Test Losses: mse: 10.8056, mae: 2.4054, huber: 1.9641, swd: 1.4395, ept: 86.0341
      Epoch 3 composite train-obj: 1.094810
            No improvement (1.9818), counter 1/5
    Epoch [4/50], Train Losses: mse: 3.1631, mae: 1.2379, huber: 0.8440, swd: 0.4618, ept: 147.3889
    Epoch [4/50], Val Losses: mse: 10.6951, mae: 2.4318, huber: 1.9860, swd: 1.0423, ept: 87.7720
    Epoch [4/50], Test Losses: mse: 10.8658, mae: 2.3828, huber: 1.9444, swd: 1.4214, ept: 89.9611
      Epoch 4 composite train-obj: 0.843977
            No improvement (1.9860), counter 2/5
    Epoch [5/50], Train Losses: mse: 2.2925, mae: 1.0531, huber: 0.6735, swd: 0.3050, ept: 160.4765
    Epoch [5/50], Val Losses: mse: 11.4540, mae: 2.5295, huber: 2.0814, swd: 1.0958, ept: 84.0439
    Epoch [5/50], Test Losses: mse: 11.3811, mae: 2.4483, huber: 2.0080, swd: 1.3869, ept: 89.2332
      Epoch 5 composite train-obj: 0.673510
            No improvement (2.0814), counter 3/5
    Epoch [6/50], Train Losses: mse: 1.7193, mae: 0.9139, huber: 0.5482, swd: 0.2197, ept: 170.8537
    Epoch [6/50], Val Losses: mse: 11.2001, mae: 2.4753, huber: 2.0300, swd: 1.0179, ept: 86.9402
    Epoch [6/50], Test Losses: mse: 11.1717, mae: 2.4137, huber: 1.9761, swd: 1.3292, ept: 91.8985
      Epoch 6 composite train-obj: 0.548222
            No improvement (2.0300), counter 4/5
    Epoch [7/50], Train Losses: mse: 1.3239, mae: 0.8054, huber: 0.4530, swd: 0.1577, ept: 178.1068
    Epoch [7/50], Val Losses: mse: 11.8204, mae: 2.5409, huber: 2.0957, swd: 1.0001, ept: 85.2715
    Epoch [7/50], Test Losses: mse: 11.3928, mae: 2.4291, huber: 1.9914, swd: 1.2607, ept: 92.1559
      Epoch 7 composite train-obj: 0.452987
    Epoch [7/50], Test Losses: mse: 11.3363, mae: 2.4801, huber: 2.0358, swd: 1.5750, ept: 78.9861
    Best round's Test MSE: 11.3370, MAE: 2.4802, SWD: 1.5754
    Best round's Validation MSE: 9.6346, MAE: 2.3252, SWD: 1.1584
    Best round's Test verification MSE : 11.3363, MAE: 2.4801, SWD: 1.5750
    Time taken: 21.30 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.8965, mae: 2.5957, huber: 2.1395, swd: 3.6092, ept: 49.6057
    Epoch [1/50], Val Losses: mse: 8.7477, mae: 2.3020, huber: 1.8516, swd: 2.0451, ept: 66.2545
    Epoch [1/50], Test Losses: mse: 10.8858, mae: 2.5218, huber: 2.0707, swd: 2.6071, ept: 66.3899
      Epoch 1 composite train-obj: 2.139535
            Val objective improved inf → 1.8516, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.7695, mae: 1.8988, huber: 1.4687, swd: 1.2957, ept: 99.7102
    Epoch [2/50], Val Losses: mse: 9.6543, mae: 2.3819, huber: 1.9331, swd: 1.4921, ept: 76.9387
    Epoch [2/50], Test Losses: mse: 10.5572, mae: 2.4162, huber: 1.9725, swd: 1.8761, ept: 79.6810
      Epoch 2 composite train-obj: 1.468702
            No improvement (1.9331), counter 1/5
    Epoch [3/50], Train Losses: mse: 4.5973, mae: 1.5090, huber: 1.0986, swd: 0.7377, ept: 126.7229
    Epoch [3/50], Val Losses: mse: 10.5557, mae: 2.4502, huber: 2.0030, swd: 1.1904, ept: 77.2680
    Epoch [3/50], Test Losses: mse: 10.8987, mae: 2.4149, huber: 1.9744, swd: 1.4301, ept: 84.0288
      Epoch 3 composite train-obj: 1.098556
            No improvement (2.0030), counter 2/5
    Epoch [4/50], Train Losses: mse: 3.1455, mae: 1.2330, huber: 0.8394, swd: 0.4514, ept: 146.5183
    Epoch [4/50], Val Losses: mse: 10.5786, mae: 2.4563, huber: 2.0077, swd: 1.0639, ept: 82.1074
    Epoch [4/50], Test Losses: mse: 10.8854, mae: 2.3969, huber: 1.9578, swd: 1.2219, ept: 86.6672
      Epoch 4 composite train-obj: 0.839350
            No improvement (2.0077), counter 3/5
    Epoch [5/50], Train Losses: mse: 2.2862, mae: 1.0512, huber: 0.6715, swd: 0.2967, ept: 160.0721
    Epoch [5/50], Val Losses: mse: 11.4866, mae: 2.5482, huber: 2.0995, swd: 0.9961, ept: 80.8827
    Epoch [5/50], Test Losses: mse: 11.1835, mae: 2.4147, huber: 1.9769, swd: 1.1282, ept: 88.8108
      Epoch 5 composite train-obj: 0.671537
            No improvement (2.0995), counter 4/5
    Epoch [6/50], Train Losses: mse: 1.7058, mae: 0.9102, huber: 0.5446, swd: 0.2085, ept: 170.5760
    Epoch [6/50], Val Losses: mse: 11.8990, mae: 2.5924, huber: 2.1446, swd: 0.9775, ept: 80.9676
    Epoch [6/50], Test Losses: mse: 11.3029, mae: 2.4246, huber: 1.9863, swd: 1.1583, ept: 91.1768
      Epoch 6 composite train-obj: 0.544623
    Epoch [6/50], Test Losses: mse: 10.8854, mae: 2.5217, huber: 2.0706, swd: 2.6064, ept: 66.4052
    Best round's Test MSE: 10.8858, MAE: 2.5218, SWD: 2.6071
    Best round's Validation MSE: 8.7477, MAE: 2.3020, SWD: 2.0451
    Best round's Test verification MSE : 10.8854, MAE: 2.5217, SWD: 2.6064
    Time taken: 18.44 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz96_seq720_pred196_20250512_2243)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.0199 ± 0.2251
      mae: 2.4863 ± 0.0269
      huber: 2.0393 ± 0.0244
      swd: 2.1254 ± 0.4240
      ept: 74.5100 ± 5.7504
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.2497 ± 0.3714
      mae: 2.3219 ± 0.0151
      huber: 1.8751 ± 0.0172
      swd: 1.6309 ± 0.3643
      ept: 73.9801 ± 6.1219
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 60.94 seconds
    
    Experiment complete: ACL_lorenz96_seq720_pred196_20250512_2243
    Model: ACL
    Dataset: lorenz96
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
    channels=data_mgr.datasets['lorenz96']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([336, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([336, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 96
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 336, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 96
    Validation Batches: 7
    Test Batches: 22
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.9352, mae: 2.7624, huber: 2.3009, swd: 3.8517, ept: 46.0090
    Epoch [1/50], Val Losses: mse: 10.3408, mae: 2.5749, huber: 2.1148, swd: 1.6389, ept: 51.1869
    Epoch [1/50], Test Losses: mse: 13.0808, mae: 2.8233, huber: 2.3633, swd: 2.0395, ept: 54.7388
      Epoch 1 composite train-obj: 2.300896
            Val objective improved inf → 2.1148, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.7711, mae: 2.2511, huber: 1.8052, swd: 1.6585, ept: 96.2882
    Epoch [2/50], Val Losses: mse: 11.0982, mae: 2.6403, huber: 2.1810, swd: 1.5154, ept: 60.4865
    Epoch [2/50], Test Losses: mse: 13.1582, mae: 2.7934, huber: 2.3363, swd: 1.7908, ept: 69.8782
      Epoch 2 composite train-obj: 1.805159
            No improvement (2.1810), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.7766, mae: 1.9088, huber: 1.4767, swd: 1.0540, ept: 127.6006
    Epoch [3/50], Val Losses: mse: 12.0336, mae: 2.7271, huber: 2.2679, swd: 1.3937, ept: 63.5978
    Epoch [3/50], Test Losses: mse: 13.9588, mae: 2.8600, huber: 2.4032, swd: 1.6213, ept: 73.6870
      Epoch 3 composite train-obj: 1.476691
            No improvement (2.2679), counter 2/5
    Epoch [4/50], Train Losses: mse: 5.2695, mae: 1.6438, huber: 1.2239, swd: 0.7095, ept: 153.6422
    Epoch [4/50], Val Losses: mse: 13.1907, mae: 2.8547, huber: 2.3940, swd: 1.2110, ept: 62.6121
    Epoch [4/50], Test Losses: mse: 14.5152, mae: 2.8934, huber: 2.4377, swd: 1.3520, ept: 79.8754
      Epoch 4 composite train-obj: 1.223862
            No improvement (2.3940), counter 3/5
    Epoch [5/50], Train Losses: mse: 4.2115, mae: 1.4475, huber: 1.0382, swd: 0.5074, ept: 178.0130
    Epoch [5/50], Val Losses: mse: 13.5867, mae: 2.8924, huber: 2.4309, swd: 1.2513, ept: 62.2534
    Epoch [5/50], Test Losses: mse: 14.3330, mae: 2.8767, huber: 2.4213, swd: 1.4298, ept: 81.5427
      Epoch 5 composite train-obj: 1.038167
            No improvement (2.4309), counter 4/5
    Epoch [6/50], Train Losses: mse: 3.3982, mae: 1.2906, huber: 0.8910, swd: 0.3814, ept: 198.8945
    Epoch [6/50], Val Losses: mse: 14.1780, mae: 2.9653, huber: 2.5026, swd: 0.9755, ept: 64.1912
    Epoch [6/50], Test Losses: mse: 14.9267, mae: 2.9259, huber: 2.4702, swd: 1.1727, ept: 81.9832
      Epoch 6 composite train-obj: 0.891002
    Epoch [6/50], Test Losses: mse: 13.0812, mae: 2.8234, huber: 2.3634, swd: 2.0393, ept: 54.7073
    Best round's Test MSE: 13.0808, MAE: 2.8233, SWD: 2.0395
    Best round's Validation MSE: 10.3408, MAE: 2.5749, SWD: 1.6389
    Best round's Test verification MSE : 13.0812, MAE: 2.8234, SWD: 2.0393
    Time taken: 18.16 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.9436, mae: 2.7639, huber: 2.3026, swd: 3.8366, ept: 46.1005
    Epoch [1/50], Val Losses: mse: 9.8470, mae: 2.5078, huber: 2.0485, swd: 2.6041, ept: 60.5945
    Epoch [1/50], Test Losses: mse: 12.2552, mae: 2.7444, huber: 2.2858, swd: 2.7756, ept: 60.7584
      Epoch 1 composite train-obj: 2.302575
            Val objective improved inf → 2.0485, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.6245, mae: 2.2219, huber: 1.7778, swd: 1.6216, ept: 100.8859
    Epoch [2/50], Val Losses: mse: 10.4397, mae: 2.5313, huber: 2.0759, swd: 1.6297, ept: 69.9971
    Epoch [2/50], Test Losses: mse: 12.7885, mae: 2.7454, huber: 2.2905, swd: 1.9726, ept: 75.3915
      Epoch 2 composite train-obj: 1.777793
            No improvement (2.0759), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.6579, mae: 1.8852, huber: 1.4547, swd: 1.0430, ept: 131.8146
    Epoch [3/50], Val Losses: mse: 11.7894, mae: 2.6937, huber: 2.2360, swd: 1.4986, ept: 69.3886
    Epoch [3/50], Test Losses: mse: 13.5085, mae: 2.7975, huber: 2.3433, swd: 1.7493, ept: 77.7859
      Epoch 3 composite train-obj: 1.454661
            No improvement (2.2360), counter 2/5
    Epoch [4/50], Train Losses: mse: 5.0978, mae: 1.6075, huber: 1.1902, swd: 0.6829, ept: 159.3467
    Epoch [4/50], Val Losses: mse: 12.5222, mae: 2.7594, huber: 2.3012, swd: 1.2913, ept: 72.6282
    Epoch [4/50], Test Losses: mse: 13.7486, mae: 2.8053, huber: 2.3519, swd: 1.5600, ept: 84.9467
      Epoch 4 composite train-obj: 1.190222
            No improvement (2.3012), counter 3/5
    Epoch [5/50], Train Losses: mse: 3.9945, mae: 1.4033, huber: 0.9971, swd: 0.4767, ept: 183.6754
    Epoch [5/50], Val Losses: mse: 13.4781, mae: 2.8629, huber: 2.4038, swd: 1.1114, ept: 70.3331
    Epoch [5/50], Test Losses: mse: 14.1872, mae: 2.8420, huber: 2.3887, swd: 1.3211, ept: 84.8263
      Epoch 5 composite train-obj: 0.997055
            No improvement (2.4038), counter 4/5
    Epoch [6/50], Train Losses: mse: 3.1358, mae: 1.2346, huber: 0.8394, swd: 0.3421, ept: 207.4675
    Epoch [6/50], Val Losses: mse: 13.7013, mae: 2.8821, huber: 2.4231, swd: 1.1802, ept: 72.4173
    Epoch [6/50], Test Losses: mse: 14.2824, mae: 2.8482, huber: 2.3950, swd: 1.3857, ept: 87.3293
      Epoch 6 composite train-obj: 0.839352
    Epoch [6/50], Test Losses: mse: 12.2548, mae: 2.7444, huber: 2.2858, swd: 2.7747, ept: 60.8261
    Best round's Test MSE: 12.2552, MAE: 2.7444, SWD: 2.7756
    Best round's Validation MSE: 9.8470, MAE: 2.5078, SWD: 2.6041
    Best round's Test verification MSE : 12.2548, MAE: 2.7444, SWD: 2.7747
    Time taken: 18.23 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.8672, mae: 2.7511, huber: 2.2902, swd: 3.8394, ept: 48.9065
    Epoch [1/50], Val Losses: mse: 9.8028, mae: 2.4904, huber: 2.0327, swd: 2.3104, ept: 62.3880
    Epoch [1/50], Test Losses: mse: 12.4057, mae: 2.7444, huber: 2.2870, swd: 2.5481, ept: 64.0047
      Epoch 1 composite train-obj: 2.290220
            Val objective improved inf → 2.0327, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.4717, mae: 2.1971, huber: 1.7538, swd: 1.5970, ept: 103.2395
    Epoch [2/50], Val Losses: mse: 10.7462, mae: 2.5776, huber: 2.1204, swd: 1.6105, ept: 70.6797
    Epoch [2/50], Test Losses: mse: 12.9219, mae: 2.7601, huber: 2.3052, swd: 1.8144, ept: 75.9282
      Epoch 2 composite train-obj: 1.753840
            No improvement (2.1204), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.4030, mae: 1.8427, huber: 1.4140, swd: 0.9815, ept: 133.7131
    Epoch [3/50], Val Losses: mse: 12.8676, mae: 2.8040, huber: 2.3450, swd: 1.3481, ept: 67.6769
    Epoch [3/50], Test Losses: mse: 14.1265, mae: 2.8574, huber: 2.4019, swd: 1.4768, ept: 77.7044
      Epoch 3 composite train-obj: 1.414034
            No improvement (2.3450), counter 2/5
    Epoch [4/50], Train Losses: mse: 4.8763, mae: 1.5687, huber: 1.1534, swd: 0.6521, ept: 162.1197
    Epoch [4/50], Val Losses: mse: 13.3125, mae: 2.8495, huber: 2.3898, swd: 1.4003, ept: 70.4502
    Epoch [4/50], Test Losses: mse: 14.2328, mae: 2.8530, huber: 2.3990, swd: 1.4713, ept: 83.3786
      Epoch 4 composite train-obj: 1.153382
            No improvement (2.3898), counter 3/5
    Epoch [5/50], Train Losses: mse: 3.7773, mae: 1.3621, huber: 0.9585, swd: 0.4583, ept: 188.8069
    Epoch [5/50], Val Losses: mse: 14.1788, mae: 2.9303, huber: 2.4706, swd: 1.2625, ept: 69.1054
    Epoch [5/50], Test Losses: mse: 14.8835, mae: 2.9028, huber: 2.4494, swd: 1.3213, ept: 85.6325
      Epoch 5 composite train-obj: 0.958532
            No improvement (2.4706), counter 4/5
    Epoch [6/50], Train Losses: mse: 2.9857, mae: 1.2078, huber: 0.8141, swd: 0.3285, ept: 212.0313
    Epoch [6/50], Val Losses: mse: 14.7465, mae: 2.9770, huber: 2.5169, swd: 1.1329, ept: 71.7693
    Epoch [6/50], Test Losses: mse: 15.1798, mae: 2.9284, huber: 2.4748, swd: 1.1955, ept: 87.1311
      Epoch 6 composite train-obj: 0.814140
    Epoch [6/50], Test Losses: mse: 12.4049, mae: 2.7443, huber: 2.2868, swd: 2.5475, ept: 63.9890
    Best round's Test MSE: 12.4057, MAE: 2.7444, SWD: 2.5481
    Best round's Validation MSE: 9.8028, MAE: 2.4904, SWD: 2.3104
    Best round's Test verification MSE : 12.4049, MAE: 2.7443, SWD: 2.5475
    Time taken: 18.06 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz96_seq720_pred336_20250512_2244)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.5806 ± 0.3590
      mae: 2.7707 ± 0.0372
      huber: 2.3120 ± 0.0363
      swd: 2.4544 ± 0.3077
      ept: 59.8340 ± 3.8388
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.9969 ± 0.2439
      mae: 2.5244 ± 0.0365
      huber: 2.0653 ± 0.0355
      swd: 2.1845 ± 0.4040
      ept: 58.0565 ± 4.9124
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 54.55 seconds
    
    Experiment complete: ACL_lorenz96_seq720_pred336_20250512_2244
    Model: ACL
    Dataset: lorenz96
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
    channels=data_mgr.datasets['lorenz96']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([720, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([720, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 720, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 4
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.7556, mae: 2.8949, huber: 2.4300, swd: 3.9050, ept: 39.6709
    Epoch [1/50], Val Losses: mse: 11.8516, mae: 2.7846, huber: 2.3195, swd: 3.0941, ept: 53.2953
    Epoch [1/50], Test Losses: mse: 14.2224, mae: 3.0211, huber: 2.5555, swd: 3.0640, ept: 51.2977
      Epoch 1 composite train-obj: 2.430004
            Val objective improved inf → 2.3195, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.4166, mae: 2.5275, huber: 2.0721, swd: 1.8834, ept: 82.3693
    Epoch [2/50], Val Losses: mse: 13.4085, mae: 2.9313, huber: 2.4670, swd: 1.9098, ept: 55.7592
    Epoch [2/50], Test Losses: mse: 15.3070, mae: 3.0858, huber: 2.6218, swd: 1.9368, ept: 61.5279
      Epoch 2 composite train-obj: 2.072142
            No improvement (2.4670), counter 1/5
    Epoch [3/50], Train Losses: mse: 8.5050, mae: 2.2100, huber: 1.7649, swd: 1.2085, ept: 104.5536
    Epoch [3/50], Val Losses: mse: 14.9533, mae: 3.0829, huber: 2.6177, swd: 1.8646, ept: 56.2307
    Epoch [3/50], Test Losses: mse: 16.0627, mae: 3.1407, huber: 2.6772, swd: 1.7324, ept: 68.5851
      Epoch 3 composite train-obj: 1.764889
            No improvement (2.6177), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.8761, mae: 1.9312, huber: 1.4970, swd: 0.8185, ept: 125.0256
    Epoch [4/50], Val Losses: mse: 15.8331, mae: 3.1845, huber: 2.7175, swd: 1.6406, ept: 55.4484
    Epoch [4/50], Test Losses: mse: 16.7036, mae: 3.1871, huber: 2.7238, swd: 1.4214, ept: 68.8778
      Epoch 4 composite train-obj: 1.496963
            No improvement (2.7175), counter 3/5
    Epoch [5/50], Train Losses: mse: 5.6034, mae: 1.7075, huber: 1.2834, swd: 0.5983, ept: 147.5427
    Epoch [5/50], Val Losses: mse: 16.4182, mae: 3.2388, huber: 2.7713, swd: 1.6451, ept: 53.0990
    Epoch [5/50], Test Losses: mse: 17.1459, mae: 3.2245, huber: 2.7612, swd: 1.3689, ept: 72.2318
      Epoch 5 composite train-obj: 1.283414
            No improvement (2.7713), counter 4/5
    Epoch [6/50], Train Losses: mse: 4.5691, mae: 1.5215, huber: 1.1067, swd: 0.4602, ept: 172.6787
    Epoch [6/50], Val Losses: mse: 17.0567, mae: 3.3011, huber: 2.8331, swd: 1.4613, ept: 56.7564
    Epoch [6/50], Test Losses: mse: 17.6188, mae: 3.2642, huber: 2.8007, swd: 1.2445, ept: 71.7642
      Epoch 6 composite train-obj: 1.106682
    Epoch [6/50], Test Losses: mse: 14.2223, mae: 3.0211, huber: 2.5554, swd: 3.0639, ept: 51.3857
    Best round's Test MSE: 14.2224, MAE: 3.0211, SWD: 3.0640
    Best round's Validation MSE: 11.8516, MAE: 2.7846, SWD: 3.0941
    Best round's Test verification MSE : 14.2223, MAE: 3.0211, SWD: 3.0639
    Time taken: 20.35 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.0556, mae: 2.9393, huber: 2.4736, swd: 4.3005, ept: 36.4158
    Epoch [1/50], Val Losses: mse: 11.8405, mae: 2.7960, huber: 2.3305, swd: 3.7213, ept: 55.3463
    Epoch [1/50], Test Losses: mse: 13.8668, mae: 2.9947, huber: 2.5291, swd: 3.7882, ept: 51.5218
      Epoch 1 composite train-obj: 2.473600
            Val objective improved inf → 2.3305, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.6818, mae: 2.5727, huber: 2.1162, swd: 2.0791, ept: 80.7519
    Epoch [2/50], Val Losses: mse: 13.0285, mae: 2.8792, huber: 2.4163, swd: 2.2494, ept: 63.7160
    Epoch [2/50], Test Losses: mse: 15.0913, mae: 3.0734, huber: 2.6094, swd: 2.2996, ept: 63.8705
      Epoch 2 composite train-obj: 2.116170
            No improvement (2.4163), counter 1/5
    Epoch [3/50], Train Losses: mse: 8.8406, mae: 2.2641, huber: 1.8174, swd: 1.3485, ept: 105.1443
    Epoch [3/50], Val Losses: mse: 14.6197, mae: 3.0408, huber: 2.5767, swd: 2.0243, ept: 58.8535
    Epoch [3/50], Test Losses: mse: 15.8169, mae: 3.1235, huber: 2.6600, swd: 1.9894, ept: 69.5744
      Epoch 3 composite train-obj: 1.817414
            No improvement (2.5767), counter 2/5
    Epoch [4/50], Train Losses: mse: 7.1394, mae: 1.9764, huber: 1.5404, swd: 0.9077, ept: 125.1660
    Epoch [4/50], Val Losses: mse: 16.1318, mae: 3.1955, huber: 2.7298, swd: 1.5412, ept: 59.0456
    Epoch [4/50], Test Losses: mse: 16.4235, mae: 3.1630, huber: 2.7002, swd: 1.5516, ept: 71.5071
      Epoch 4 composite train-obj: 1.540416
            No improvement (2.7298), counter 3/5
    Epoch [5/50], Train Losses: mse: 5.7635, mae: 1.7351, huber: 1.3098, swd: 0.6544, ept: 145.4405
    Epoch [5/50], Val Losses: mse: 16.8507, mae: 3.2696, huber: 2.8031, swd: 1.4989, ept: 57.3283
    Epoch [5/50], Test Losses: mse: 16.9287, mae: 3.1989, huber: 2.7366, swd: 1.5375, ept: 72.8572
      Epoch 5 composite train-obj: 1.309827
            No improvement (2.8031), counter 4/5
    Epoch [6/50], Train Losses: mse: 4.6616, mae: 1.5376, huber: 1.1221, swd: 0.4936, ept: 168.9185
    Epoch [6/50], Val Losses: mse: 17.4880, mae: 3.3277, huber: 2.8609, swd: 1.3508, ept: 59.6905
    Epoch [6/50], Test Losses: mse: 17.5391, mae: 3.2503, huber: 2.7875, swd: 1.4101, ept: 75.2985
      Epoch 6 composite train-obj: 1.122075
    Epoch [6/50], Test Losses: mse: 13.8675, mae: 2.9948, huber: 2.5292, swd: 3.7884, ept: 51.4476
    Best round's Test MSE: 13.8668, MAE: 2.9947, SWD: 3.7882
    Best round's Validation MSE: 11.8405, MAE: 2.7960, SWD: 3.7213
    Best round's Test verification MSE : 13.8675, MAE: 2.9948, SWD: 3.7884
    Time taken: 17.02 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.9901, mae: 2.9305, huber: 2.4649, swd: 4.2643, ept: 36.5829
    Epoch [1/50], Val Losses: mse: 11.7787, mae: 2.7822, huber: 2.3175, swd: 3.3949, ept: 59.5257
    Epoch [1/50], Test Losses: mse: 13.9373, mae: 2.9954, huber: 2.5303, swd: 3.3697, ept: 51.4915
      Epoch 1 composite train-obj: 2.464851
            Val objective improved inf → 2.3175, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.4931, mae: 2.5393, huber: 2.0839, swd: 1.9606, ept: 84.6731
    Epoch [2/50], Val Losses: mse: 12.8263, mae: 2.8710, huber: 2.4076, swd: 2.7696, ept: 65.1458
    Epoch [2/50], Test Losses: mse: 15.0406, mae: 3.0653, huber: 2.6019, swd: 2.5750, ept: 68.3999
      Epoch 2 composite train-obj: 2.083879
            No improvement (2.4076), counter 1/5
    Epoch [3/50], Train Losses: mse: 8.4759, mae: 2.2043, huber: 1.7596, swd: 1.2561, ept: 111.2031
    Epoch [3/50], Val Losses: mse: 14.6015, mae: 3.0682, huber: 2.6027, swd: 2.2495, ept: 60.3986
    Epoch [3/50], Test Losses: mse: 16.2331, mae: 3.1595, huber: 2.6961, swd: 2.0828, ept: 74.6537
      Epoch 3 composite train-obj: 1.759640
            No improvement (2.6027), counter 2/5
    Epoch [4/50], Train Losses: mse: 6.7297, mae: 1.9058, huber: 1.4729, swd: 0.8542, ept: 134.7662
    Epoch [4/50], Val Losses: mse: 15.4979, mae: 3.1577, huber: 2.6915, swd: 1.9121, ept: 62.4606
    Epoch [4/50], Test Losses: mse: 16.6937, mae: 3.1895, huber: 2.7263, swd: 1.6908, ept: 74.7426
      Epoch 4 composite train-obj: 1.472939
            No improvement (2.6915), counter 3/5
    Epoch [5/50], Train Losses: mse: 5.3529, mae: 1.6620, huber: 1.2404, swd: 0.6124, ept: 158.7713
    Epoch [5/50], Val Losses: mse: 16.9875, mae: 3.2997, huber: 2.8326, swd: 1.2772, ept: 59.7146
    Epoch [5/50], Test Losses: mse: 17.6118, mae: 3.2623, huber: 2.7990, swd: 1.2241, ept: 75.8658
      Epoch 5 composite train-obj: 1.240447
            No improvement (2.8326), counter 4/5
    Epoch [6/50], Train Losses: mse: 4.2321, mae: 1.4582, huber: 1.0471, swd: 0.4508, ept: 184.9126
    Epoch [6/50], Val Losses: mse: 17.6278, mae: 3.3528, huber: 2.8851, swd: 1.4932, ept: 63.3709
    Epoch [6/50], Test Losses: mse: 17.7540, mae: 3.2765, huber: 2.8134, swd: 1.3748, ept: 78.0840
      Epoch 6 composite train-obj: 1.047148
    Epoch [6/50], Test Losses: mse: 13.9374, mae: 2.9955, huber: 2.5303, swd: 3.3694, ept: 51.4511
    Best round's Test MSE: 13.9373, MAE: 2.9954, SWD: 3.3697
    Best round's Validation MSE: 11.7787, MAE: 2.7822, SWD: 3.3949
    Best round's Test verification MSE : 13.9374, MAE: 2.9955, SWD: 3.3694
    Time taken: 18.23 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz96_seq720_pred720_20250512_2245)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 14.0088 ± 0.1538
      mae: 3.0037 ± 0.0123
      huber: 2.5383 ± 0.0121
      swd: 3.4073 ± 0.2968
      ept: 51.4370 ± 0.0993
      count: 4.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 11.8236 ± 0.0321
      mae: 2.7876 ± 0.0060
      huber: 2.3225 ± 0.0057
      swd: 3.4035 ± 0.2561
      ept: 56.0558 ± 2.5926
      count: 4.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 56.22 seconds
    
    Experiment complete: ACL_lorenz96_seq720_pred720_20250512_2245
    Model: ACL
    Dataset: lorenz96
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
    channels=data_mgr.datasets['lorenz96']['channels'],
    enc_in=data_mgr.datasets['lorenz96']['channels'],
    dec_in=data_mgr.datasets['lorenz96']['channels'],
    c_out=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_mixer_96_96 = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([96, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([96, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 98
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 96, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 98
    Validation Batches: 9
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.0066, mae: 2.1243, huber: 1.6841, swd: 2.3337, ept: 51.1922
    Epoch [1/50], Val Losses: mse: 9.4924, mae: 2.3113, huber: 1.8669, swd: 2.1241, ept: 55.1325
    Epoch [1/50], Test Losses: mse: 9.6411, mae: 2.2771, huber: 1.8396, swd: 2.6438, ept: 56.5225
      Epoch 1 composite train-obj: 1.684104
            Val objective improved inf → 1.8669, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 3.3402, mae: 1.2859, huber: 0.8865, swd: 0.8811, ept: 79.5466
    Epoch [2/50], Val Losses: mse: 10.3811, mae: 2.3891, huber: 1.9458, swd: 2.1742, ept: 54.5422
    Epoch [2/50], Test Losses: mse: 10.3066, mae: 2.3097, huber: 1.8734, swd: 2.3046, ept: 58.5625
      Epoch 2 composite train-obj: 0.886485
            No improvement (1.9458), counter 1/5
    Epoch [3/50], Train Losses: mse: 1.4535, mae: 0.8454, huber: 0.4848, swd: 0.3719, ept: 90.9300
    Epoch [3/50], Val Losses: mse: 10.2451, mae: 2.3463, huber: 1.9070, swd: 2.1588, ept: 56.1983
    Epoch [3/50], Test Losses: mse: 10.3883, mae: 2.2913, huber: 1.8579, swd: 2.3401, ept: 59.7979
      Epoch 3 composite train-obj: 0.484797
            No improvement (1.9070), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.7340, mae: 0.6163, huber: 0.2905, swd: 0.1909, ept: 94.5168
    Epoch [4/50], Val Losses: mse: 10.3376, mae: 2.3561, huber: 1.9165, swd: 2.0206, ept: 56.8514
    Epoch [4/50], Test Losses: mse: 10.4150, mae: 2.3026, huber: 1.8701, swd: 2.3799, ept: 60.7138
      Epoch 4 composite train-obj: 0.290458
            No improvement (1.9165), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4430, mae: 0.4924, huber: 0.1950, swd: 0.1168, ept: 95.5371
    Epoch [5/50], Val Losses: mse: 10.4052, mae: 2.3579, huber: 1.9174, swd: 1.9657, ept: 58.0898
    Epoch [5/50], Test Losses: mse: 10.4561, mae: 2.3030, huber: 1.8709, swd: 2.3644, ept: 60.4906
      Epoch 5 composite train-obj: 0.194979
            No improvement (1.9174), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3095, mae: 0.4203, huber: 0.1447, swd: 0.0818, ept: 95.8723
    Epoch [6/50], Val Losses: mse: 10.3403, mae: 2.3552, huber: 1.9171, swd: 1.9386, ept: 58.1146
    Epoch [6/50], Test Losses: mse: 10.4024, mae: 2.3000, huber: 1.8677, swd: 2.3896, ept: 60.9994
      Epoch 6 composite train-obj: 0.144678
    Epoch [6/50], Test Losses: mse: 9.6411, mae: 2.2771, huber: 1.8396, swd: 2.6438, ept: 56.5225
    Best round's Test MSE: 9.6411, MAE: 2.2771, SWD: 2.6438
    Best round's Validation MSE: 9.4924, MAE: 2.3113, SWD: 2.1241
    Best round's Test verification MSE : 9.6411, MAE: 2.2771, SWD: 2.6438
    Time taken: 67.10 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.1795, mae: 2.2708, huber: 1.8266, swd: 2.1885, ept: 42.8139
    Epoch [1/50], Val Losses: mse: 9.1093, mae: 2.2550, huber: 1.8126, swd: 2.1280, ept: 57.1914
    Epoch [1/50], Test Losses: mse: 9.5051, mae: 2.2622, huber: 1.8251, swd: 2.5467, ept: 57.8062
      Epoch 1 composite train-obj: 1.826596
            Val objective improved inf → 1.8126, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 3.7966, mae: 1.3809, huber: 0.9754, swd: 1.0340, ept: 76.9467
    Epoch [2/50], Val Losses: mse: 9.2521, mae: 2.2472, huber: 1.8094, swd: 1.8965, ept: 57.9603
    Epoch [2/50], Test Losses: mse: 9.5484, mae: 2.2323, huber: 1.7990, swd: 2.4886, ept: 59.2674
      Epoch 2 composite train-obj: 0.975404
            Val objective improved 1.8126 → 1.8094, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.5922, mae: 0.8921, huber: 0.5248, swd: 0.4039, ept: 90.0592
    Epoch [3/50], Val Losses: mse: 9.6463, mae: 2.2759, huber: 1.8391, swd: 1.9207, ept: 58.3617
    Epoch [3/50], Test Losses: mse: 9.5767, mae: 2.2072, huber: 1.7767, swd: 2.4079, ept: 61.5277
      Epoch 3 composite train-obj: 0.524836
            No improvement (1.8391), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.7838, mae: 0.6445, huber: 0.3118, swd: 0.1938, ept: 94.3646
    Epoch [4/50], Val Losses: mse: 9.8736, mae: 2.3065, huber: 1.8679, swd: 1.9673, ept: 59.1426
    Epoch [4/50], Test Losses: mse: 9.3783, mae: 2.1911, huber: 1.7613, swd: 2.2992, ept: 62.2049
      Epoch 4 composite train-obj: 0.311775
            No improvement (1.8679), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4744, mae: 0.5150, huber: 0.2102, swd: 0.1212, ept: 95.4948
    Epoch [5/50], Val Losses: mse: 9.6544, mae: 2.2893, huber: 1.8500, swd: 2.0002, ept: 58.8190
    Epoch [5/50], Test Losses: mse: 9.3059, mae: 2.1830, huber: 1.7529, swd: 2.3630, ept: 62.6617
      Epoch 5 composite train-obj: 0.210215
            No improvement (1.8500), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3351, mae: 0.4404, huber: 0.1572, swd: 0.0878, ept: 95.9044
    Epoch [6/50], Val Losses: mse: 9.6021, mae: 2.2956, huber: 1.8549, swd: 1.9864, ept: 59.9808
    Epoch [6/50], Test Losses: mse: 9.1978, mae: 2.1776, huber: 1.7473, swd: 2.3540, ept: 63.1925
      Epoch 6 composite train-obj: 0.157152
            No improvement (1.8549), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.2532, mae: 0.3862, huber: 0.1222, swd: 0.0693, ept: 95.9870
    Epoch [7/50], Val Losses: mse: 9.6234, mae: 2.3107, huber: 1.8687, swd: 1.9156, ept: 59.5133
    Epoch [7/50], Test Losses: mse: 9.1315, mae: 2.1759, huber: 1.7447, swd: 2.3437, ept: 63.7240
      Epoch 7 composite train-obj: 0.122167
    Epoch [7/50], Test Losses: mse: 9.5484, mae: 2.2323, huber: 1.7990, swd: 2.4886, ept: 59.2674
    Best round's Test MSE: 9.5484, MAE: 2.2323, SWD: 2.4886
    Best round's Validation MSE: 9.2521, MAE: 2.2472, SWD: 1.8965
    Best round's Test verification MSE : 9.5484, MAE: 2.2323, SWD: 2.4886
    Time taken: 80.75 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.4992, mae: 2.1983, huber: 1.7561, swd: 2.4346, ept: 51.1518
    Epoch [1/50], Val Losses: mse: 9.1605, mae: 2.2731, huber: 1.8325, swd: 2.0126, ept: 57.5813
    Epoch [1/50], Test Losses: mse: 9.1169, mae: 2.2255, huber: 1.7883, swd: 2.3334, ept: 59.6838
      Epoch 1 composite train-obj: 1.756084
            Val objective improved inf → 1.8325, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.2673, mae: 1.4626, huber: 1.0538, swd: 1.0662, ept: 75.1852
    Epoch [2/50], Val Losses: mse: 9.1861, mae: 2.1957, huber: 1.7622, swd: 1.6103, ept: 60.9247
    Epoch [2/50], Test Losses: mse: 9.3778, mae: 2.1991, huber: 1.7692, swd: 2.0578, ept: 60.6781
      Epoch 2 composite train-obj: 1.053846
            Val objective improved 1.8325 → 1.7622, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.9398, mae: 0.9695, huber: 0.5963, swd: 0.4571, ept: 88.2300
    Epoch [3/50], Val Losses: mse: 10.0783, mae: 2.2814, huber: 1.8459, swd: 1.6907, ept: 60.1280
    Epoch [3/50], Test Losses: mse: 9.7456, mae: 2.2154, huber: 1.7858, swd: 1.9955, ept: 62.0141
      Epoch 3 composite train-obj: 0.596280
            No improvement (1.8459), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.9188, mae: 0.6839, huber: 0.3465, swd: 0.2123, ept: 93.7472
    Epoch [4/50], Val Losses: mse: 10.2460, mae: 2.3061, huber: 1.8705, swd: 1.6652, ept: 60.4421
    Epoch [4/50], Test Losses: mse: 9.5664, mae: 2.1889, huber: 1.7607, swd: 1.9201, ept: 63.2329
      Epoch 4 composite train-obj: 0.346467
            No improvement (1.8705), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5225, mae: 0.5314, huber: 0.2247, swd: 0.1229, ept: 95.3628
    Epoch [5/50], Val Losses: mse: 10.0487, mae: 2.2941, huber: 1.8589, swd: 1.7034, ept: 60.0954
    Epoch [5/50], Test Losses: mse: 9.5257, mae: 2.1791, huber: 1.7538, swd: 2.0056, ept: 64.1422
      Epoch 5 composite train-obj: 0.224738
            No improvement (1.8589), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3450, mae: 0.4408, huber: 0.1593, swd: 0.0839, ept: 95.8513
    Epoch [6/50], Val Losses: mse: 10.2122, mae: 2.3279, huber: 1.8893, swd: 1.6383, ept: 59.7269
    Epoch [6/50], Test Losses: mse: 9.4797, mae: 2.1853, huber: 1.7572, swd: 1.9666, ept: 64.1782
      Epoch 6 composite train-obj: 0.159340
            No improvement (1.8893), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.2479, mae: 0.3767, huber: 0.1185, swd: 0.0620, ept: 95.9615
    Epoch [7/50], Val Losses: mse: 10.2902, mae: 2.3416, huber: 1.9025, swd: 1.7392, ept: 59.7238
    Epoch [7/50], Test Losses: mse: 9.3783, mae: 2.1696, huber: 1.7427, swd: 2.0096, ept: 64.5787
      Epoch 7 composite train-obj: 0.118498
    Epoch [7/50], Test Losses: mse: 9.3778, mae: 2.1991, huber: 1.7692, swd: 2.0578, ept: 60.6781
    Best round's Test MSE: 9.3778, MAE: 2.1991, SWD: 2.0578
    Best round's Validation MSE: 9.1861, MAE: 2.1957, SWD: 1.6103
    Best round's Test verification MSE : 9.3778, MAE: 2.1991, SWD: 2.0578
    Time taken: 81.63 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz96_seq720_pred96_20250512_2246)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 9.5224 ± 0.1090
      mae: 2.2362 ± 0.0319
      huber: 1.8026 ± 0.0288
      swd: 2.3967 ± 0.2479
      ept: 58.8227 ± 1.7254
      count: 9.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.3102 ± 0.1316
      mae: 2.2514 ± 0.0473
      huber: 1.8128 ± 0.0428
      swd: 1.8770 ± 0.2102
      ept: 58.0058 ± 2.3649
      count: 9.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 229.63 seconds
    
    Experiment complete: TimeMixer_lorenz96_seq720_pred96_20250512_2246
    Model: TimeMixer
    Dataset: lorenz96
    Sequence Length: 720
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=720,
    pred_len=196,
    channels=data_mgr.datasets['lorenz96']['channels'],
    enc_in=data_mgr.datasets['lorenz96']['channels'],
    dec_in=data_mgr.datasets['lorenz96']['channels'],
    c_out=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_mixer_96_96 = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([196, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([196, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 97
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 196, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 97
    Validation Batches: 8
    Test Batches: 23
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.4493, mae: 2.6160, huber: 2.1604, swd: 2.3212, ept: 43.9876
    Epoch [1/50], Val Losses: mse: 11.2757, mae: 2.6170, huber: 2.1615, swd: 1.9935, ept: 58.2922
    Epoch [1/50], Test Losses: mse: 12.3034, mae: 2.6891, huber: 2.2358, swd: 2.5837, ept: 63.5222
      Epoch 1 composite train-obj: 2.160446
            Val objective improved inf → 2.1615, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.5832, mae: 1.8999, huber: 1.4665, swd: 1.3551, ept: 89.9971
    Epoch [2/50], Val Losses: mse: 13.2605, mae: 2.8014, huber: 2.3446, swd: 1.8320, ept: 57.3170
    Epoch [2/50], Test Losses: mse: 13.1721, mae: 2.7393, huber: 2.2882, swd: 2.2262, ept: 66.1714
      Epoch 2 composite train-obj: 1.466453
            No improvement (2.3446), counter 1/5
    Epoch [3/50], Train Losses: mse: 3.8382, mae: 1.4056, huber: 0.9947, swd: 0.7167, ept: 128.6742
    Epoch [3/50], Val Losses: mse: 14.0015, mae: 2.8917, huber: 2.4334, swd: 1.6356, ept: 57.0096
    Epoch [3/50], Test Losses: mse: 13.5050, mae: 2.7562, huber: 2.3053, swd: 2.0865, ept: 69.9968
      Epoch 3 composite train-obj: 0.994657
            No improvement (2.4334), counter 2/5
    Epoch [4/50], Train Losses: mse: 2.0900, mae: 1.0309, huber: 0.6460, swd: 0.3752, ept: 163.3181
    Epoch [4/50], Val Losses: mse: 13.9665, mae: 2.8800, huber: 2.4228, swd: 1.7088, ept: 57.8087
    Epoch [4/50], Test Losses: mse: 13.5178, mae: 2.7480, huber: 2.2978, swd: 2.0088, ept: 75.3577
      Epoch 4 composite train-obj: 0.646019
            No improvement (2.4228), counter 3/5
    Epoch [5/50], Train Losses: mse: 1.2110, mae: 0.7988, huber: 0.4384, swd: 0.2140, ept: 181.9837
    Epoch [5/50], Val Losses: mse: 14.1487, mae: 2.8993, huber: 2.4408, swd: 1.5763, ept: 57.8718
    Epoch [5/50], Test Losses: mse: 13.5681, mae: 2.7413, huber: 2.2915, swd: 1.8667, ept: 77.2832
      Epoch 5 composite train-obj: 0.438403
            No improvement (2.4408), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.7621, mae: 0.6459, huber: 0.3097, swd: 0.1312, ept: 189.9317
    Epoch [6/50], Val Losses: mse: 13.7714, mae: 2.8625, huber: 2.4049, swd: 1.7011, ept: 62.3593
    Epoch [6/50], Test Losses: mse: 13.3052, mae: 2.7139, huber: 2.2652, swd: 1.9132, ept: 79.4833
      Epoch 6 composite train-obj: 0.309724
    Epoch [6/50], Test Losses: mse: 12.3034, mae: 2.6891, huber: 2.2358, swd: 2.5837, ept: 63.5222
    Best round's Test MSE: 12.3034, MAE: 2.6891, SWD: 2.5837
    Best round's Validation MSE: 11.2757, MAE: 2.6170, SWD: 1.9935
    Best round's Test verification MSE : 12.3034, MAE: 2.6891, SWD: 2.5837
    Time taken: 71.20 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.4424, mae: 2.5894, huber: 2.1353, swd: 1.9917, ept: 43.0211
    Epoch [1/50], Val Losses: mse: 12.2793, mae: 2.7148, huber: 2.2583, swd: 1.5631, ept: 56.5014
    Epoch [1/50], Test Losses: mse: 12.6776, mae: 2.7228, huber: 2.2699, swd: 2.0340, ept: 61.7963
      Epoch 1 composite train-obj: 2.135340
            Val objective improved inf → 2.2583, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.2537, mae: 1.6681, huber: 1.2443, swd: 1.0219, ept: 107.4966
    Epoch [2/50], Val Losses: mse: 13.4300, mae: 2.7880, huber: 2.3342, swd: 1.9038, ept: 61.1744
    Epoch [2/50], Test Losses: mse: 13.3235, mae: 2.7370, huber: 2.2873, swd: 2.2056, ept: 68.7505
      Epoch 2 composite train-obj: 1.244250
            No improvement (2.3342), counter 1/5
    Epoch [3/50], Train Losses: mse: 2.5913, mae: 1.1450, huber: 0.7512, swd: 0.4562, ept: 152.7256
    Epoch [3/50], Val Losses: mse: 13.9299, mae: 2.8564, huber: 2.4005, swd: 1.7451, ept: 60.4130
    Epoch [3/50], Test Losses: mse: 13.5467, mae: 2.7482, huber: 2.2987, swd: 2.1060, ept: 73.5951
      Epoch 3 composite train-obj: 0.751176
            No improvement (2.4005), counter 2/5
    Epoch [4/50], Train Losses: mse: 1.3150, mae: 0.8278, huber: 0.4642, swd: 0.2233, ept: 179.4077
    Epoch [4/50], Val Losses: mse: 14.1093, mae: 2.8653, huber: 2.4095, swd: 1.7457, ept: 61.9755
    Epoch [4/50], Test Losses: mse: 13.4870, mae: 2.7320, huber: 2.2839, swd: 2.0376, ept: 77.6847
      Epoch 4 composite train-obj: 0.464238
            No improvement (2.4095), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.7647, mae: 0.6503, huber: 0.3130, swd: 0.1298, ept: 190.0091
    Epoch [5/50], Val Losses: mse: 14.1280, mae: 2.8660, huber: 2.4098, swd: 1.6434, ept: 62.7747
    Epoch [5/50], Test Losses: mse: 13.4974, mae: 2.7219, huber: 2.2750, swd: 2.0469, ept: 80.3553
      Epoch 5 composite train-obj: 0.313011
            No improvement (2.4098), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.5026, mae: 0.5376, huber: 0.2245, swd: 0.0866, ept: 194.0694
    Epoch [6/50], Val Losses: mse: 13.8287, mae: 2.8448, huber: 2.3883, swd: 1.7318, ept: 63.9865
    Epoch [6/50], Test Losses: mse: 13.3499, mae: 2.7019, huber: 2.2562, swd: 2.1385, ept: 82.6717
      Epoch 6 composite train-obj: 0.224549
    Epoch [6/50], Test Losses: mse: 12.6776, mae: 2.7228, huber: 2.2699, swd: 2.0340, ept: 61.7963
    Best round's Test MSE: 12.6776, MAE: 2.7228, SWD: 2.0340
    Best round's Validation MSE: 12.2793, MAE: 2.7148, SWD: 1.5631
    Best round's Test verification MSE : 12.6776, MAE: 2.7228, SWD: 2.0340
    Time taken: 70.72 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.2739, mae: 2.3384, huber: 1.8905, swd: 2.2144, ept: 66.0470
    Epoch [1/50], Val Losses: mse: 12.0913, mae: 2.6626, huber: 2.2092, swd: 1.6491, ept: 60.0878
    Epoch [1/50], Test Losses: mse: 12.6757, mae: 2.7135, huber: 2.2616, swd: 2.0283, ept: 60.4033
      Epoch 1 composite train-obj: 1.890456
            Val objective improved inf → 2.2092, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.4539, mae: 1.5054, huber: 1.0918, swd: 0.8294, ept: 123.0242
    Epoch [2/50], Val Losses: mse: 13.5582, mae: 2.7968, huber: 2.3419, swd: 1.5302, ept: 61.7683
    Epoch [2/50], Test Losses: mse: 13.0701, mae: 2.7032, huber: 2.2546, swd: 1.8214, ept: 73.1681
      Epoch 2 composite train-obj: 1.091781
            No improvement (2.3419), counter 1/5
    Epoch [3/50], Train Losses: mse: 1.6064, mae: 0.8948, huber: 0.5261, swd: 0.2749, ept: 174.6369
    Epoch [3/50], Val Losses: mse: 13.8611, mae: 2.8018, huber: 2.3476, swd: 1.4848, ept: 67.4762
    Epoch [3/50], Test Losses: mse: 13.2629, mae: 2.7011, huber: 2.2545, swd: 1.7271, ept: 79.3573
      Epoch 3 composite train-obj: 0.526129
            No improvement (2.3476), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.6831, mae: 0.6086, huber: 0.2819, swd: 0.1100, ept: 191.3753
    Epoch [4/50], Val Losses: mse: 13.9238, mae: 2.8137, huber: 2.3594, swd: 1.4828, ept: 68.6019
    Epoch [4/50], Test Losses: mse: 12.9647, mae: 2.6655, huber: 2.2205, swd: 1.7338, ept: 82.1186
      Epoch 4 composite train-obj: 0.281902
            No improvement (2.3594), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.3943, mae: 0.4731, huber: 0.1800, swd: 0.0650, ept: 195.0503
    Epoch [5/50], Val Losses: mse: 13.6991, mae: 2.7952, huber: 2.3403, swd: 1.5433, ept: 69.4796
    Epoch [5/50], Test Losses: mse: 12.7460, mae: 2.6428, huber: 2.1980, swd: 1.7876, ept: 84.0897
      Epoch 5 composite train-obj: 0.180037
            No improvement (2.3403), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2771, mae: 0.4008, huber: 0.1319, swd: 0.0490, ept: 195.7967
    Epoch [6/50], Val Losses: mse: 13.4858, mae: 2.7719, huber: 2.3180, swd: 1.5216, ept: 70.9539
    Epoch [6/50], Test Losses: mse: 12.5881, mae: 2.6242, huber: 2.1800, swd: 1.7733, ept: 85.2264
      Epoch 6 composite train-obj: 0.131852
    Epoch [6/50], Test Losses: mse: 12.6757, mae: 2.7135, huber: 2.2616, swd: 2.0283, ept: 60.4033
    Best round's Test MSE: 12.6757, MAE: 2.7135, SWD: 2.0283
    Best round's Validation MSE: 12.0913, MAE: 2.6626, SWD: 1.6491
    Best round's Test verification MSE : 12.6757, MAE: 2.7135, SWD: 2.0283
    Time taken: 71.64 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz96_seq720_pred196_20250512_2250)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 12.5522 ± 0.1760
      mae: 2.7085 ± 0.0142
      huber: 2.2557 ± 0.0145
      swd: 2.2153 ± 0.2605
      ept: 61.9073 ± 1.2757
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 11.8821 ± 0.4356
      mae: 2.6648 ± 0.0400
      huber: 2.2097 ± 0.0395
      swd: 1.7352 ± 0.1860
      ept: 58.2938 ± 1.4642
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 213.69 seconds
    
    Experiment complete: TimeMixer_lorenz96_seq720_pred196_20250512_2250
    Model: TimeMixer
    Dataset: lorenz96
    Sequence Length: 720
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=720,
    pred_len=336,
    channels=data_mgr.datasets['lorenz96']['channels'],
    enc_in=data_mgr.datasets['lorenz96']['channels'],
    dec_in=data_mgr.datasets['lorenz96']['channels'],
    c_out=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_mixer_96_96 = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([336, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([336, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 96
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 336, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 96
    Validation Batches: 7
    Test Batches: 22
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.0516, mae: 2.6095, huber: 2.1528, swd: 2.1927, ept: 53.1171
    Epoch [1/50], Val Losses: mse: 14.1213, mae: 2.9584, huber: 2.4960, swd: 1.5656, ept: 53.8440
    Epoch [1/50], Test Losses: mse: 14.6632, mae: 2.9838, huber: 2.5236, swd: 1.8097, ept: 63.6889
      Epoch 1 composite train-obj: 2.152769
            Val objective improved inf → 2.4960, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.2420, mae: 1.8490, huber: 1.4170, swd: 1.0367, ept: 115.7466
    Epoch [2/50], Val Losses: mse: 16.0559, mae: 3.1448, huber: 2.6817, swd: 1.4428, ept: 57.5621
    Epoch [2/50], Test Losses: mse: 15.5605, mae: 3.0089, huber: 2.5517, swd: 1.6870, ept: 69.8797
      Epoch 2 composite train-obj: 1.416951
            No improvement (2.6817), counter 1/5
    Epoch [3/50], Train Losses: mse: 3.0399, mae: 1.2512, huber: 0.8486, swd: 0.4309, ept: 205.8162
    Epoch [3/50], Val Losses: mse: 16.6460, mae: 3.2016, huber: 2.7379, swd: 1.4335, ept: 56.7293
    Epoch [3/50], Test Losses: mse: 16.3230, mae: 3.0610, huber: 2.6040, swd: 1.5577, ept: 80.7750
      Epoch 3 composite train-obj: 0.848625
            No improvement (2.7379), counter 2/5
    Epoch [4/50], Train Losses: mse: 1.5110, mae: 0.8887, huber: 0.5164, swd: 0.2011, ept: 281.4081
    Epoch [4/50], Val Losses: mse: 17.2108, mae: 3.2552, huber: 2.7911, swd: 1.4231, ept: 57.6660
    Epoch [4/50], Test Losses: mse: 16.3415, mae: 3.0575, huber: 2.6013, swd: 1.5129, ept: 86.9526
      Epoch 4 composite train-obj: 0.516413
            No improvement (2.7911), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.8780, mae: 0.6906, huber: 0.3454, swd: 0.1142, ept: 313.0831
    Epoch [5/50], Val Losses: mse: 17.1707, mae: 3.2506, huber: 2.7866, swd: 1.3569, ept: 58.4228
    Epoch [5/50], Test Losses: mse: 16.3217, mae: 3.0533, huber: 2.5970, swd: 1.4768, ept: 90.6760
      Epoch 5 composite train-obj: 0.345420
            No improvement (2.7866), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.5878, mae: 0.5742, huber: 0.2520, swd: 0.0768, ept: 324.9211
    Epoch [6/50], Val Losses: mse: 17.0902, mae: 3.2494, huber: 2.7851, swd: 1.4005, ept: 57.1979
    Epoch [6/50], Test Losses: mse: 16.2840, mae: 3.0564, huber: 2.5998, swd: 1.5418, ept: 90.4400
      Epoch 6 composite train-obj: 0.251951
    Epoch [6/50], Test Losses: mse: 14.6632, mae: 2.9838, huber: 2.5236, swd: 1.8097, ept: 63.6889
    Best round's Test MSE: 14.6632, MAE: 2.9838, SWD: 1.8097
    Best round's Validation MSE: 14.1213, MAE: 2.9584, SWD: 1.5656
    Best round's Test verification MSE : 14.6632, MAE: 2.9838, SWD: 1.8097
    Time taken: 72.19 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.0022, mae: 2.8115, huber: 2.3517, swd: 1.8482, ept: 33.4487
    Epoch [1/50], Val Losses: mse: 13.6338, mae: 2.9175, huber: 2.4542, swd: 1.7816, ept: 46.4222
    Epoch [1/50], Test Losses: mse: 14.3783, mae: 2.9610, huber: 2.5007, swd: 2.1603, ept: 56.1273
      Epoch 1 composite train-obj: 2.351695
            Val objective improved inf → 2.4542, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.1036, mae: 1.8243, huber: 1.3931, swd: 0.9882, ept: 112.9939
    Epoch [2/50], Val Losses: mse: 16.0822, mae: 3.1340, huber: 2.6702, swd: 1.4273, ept: 50.1530
    Epoch [2/50], Test Losses: mse: 15.9504, mae: 3.0537, huber: 2.5952, swd: 1.5520, ept: 64.8129
      Epoch 2 composite train-obj: 1.393069
            No improvement (2.6702), counter 1/5
    Epoch [3/50], Train Losses: mse: 2.7087, mae: 1.1758, huber: 0.7787, swd: 0.3696, ept: 215.1181
    Epoch [3/50], Val Losses: mse: 16.5184, mae: 3.1940, huber: 2.7294, swd: 1.4293, ept: 52.5576
    Epoch [3/50], Test Losses: mse: 16.4975, mae: 3.0810, huber: 2.6229, swd: 1.5905, ept: 78.0843
      Epoch 3 composite train-obj: 0.778717
            No improvement (2.7294), counter 2/5
    Epoch [4/50], Train Losses: mse: 1.2784, mae: 0.8213, huber: 0.4571, swd: 0.1625, ept: 291.4844
    Epoch [4/50], Val Losses: mse: 16.6424, mae: 3.2063, huber: 2.7417, swd: 1.4562, ept: 53.8770
    Epoch [4/50], Test Losses: mse: 16.2780, mae: 3.0591, huber: 2.6017, swd: 1.5449, ept: 84.8859
      Epoch 4 composite train-obj: 0.457143
            No improvement (2.7417), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.7529, mae: 0.6459, huber: 0.3084, swd: 0.0952, ept: 318.2941
    Epoch [5/50], Val Losses: mse: 16.5140, mae: 3.1905, huber: 2.7265, swd: 1.5175, ept: 54.2991
    Epoch [5/50], Test Losses: mse: 16.1299, mae: 3.0409, huber: 2.5839, swd: 1.5227, ept: 90.4010
      Epoch 5 composite train-obj: 0.308426
            No improvement (2.7265), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.5181, mae: 0.5451, huber: 0.2292, swd: 0.0662, ept: 328.2536
    Epoch [6/50], Val Losses: mse: 16.0966, mae: 3.1572, huber: 2.6929, swd: 1.5214, ept: 56.6253
    Epoch [6/50], Test Losses: mse: 15.9834, mae: 3.0300, huber: 2.5730, swd: 1.6343, ept: 90.9533
      Epoch 6 composite train-obj: 0.229219
    Epoch [6/50], Test Losses: mse: 14.3783, mae: 2.9610, huber: 2.5007, swd: 2.1603, ept: 56.1273
    Best round's Test MSE: 14.3783, MAE: 2.9610, SWD: 2.1603
    Best round's Validation MSE: 13.6338, MAE: 2.9175, SWD: 1.7816
    Best round's Test verification MSE : 14.3783, MAE: 2.9610, SWD: 2.1603
    Time taken: 72.43 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.8384, mae: 2.5816, huber: 2.1257, swd: 2.4803, ept: 57.0887
    Epoch [1/50], Val Losses: mse: 13.4461, mae: 2.8781, huber: 2.4157, swd: 1.7990, ept: 57.7930
    Epoch [1/50], Test Losses: mse: 14.3255, mae: 2.9543, huber: 2.4939, swd: 2.1482, ept: 61.0520
      Epoch 1 composite train-obj: 2.125705
            Val objective improved inf → 2.4157, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.7533, mae: 1.7569, huber: 1.3292, swd: 0.9481, ept: 127.9140
    Epoch [2/50], Val Losses: mse: 16.0923, mae: 3.1460, huber: 2.6826, swd: 1.7625, ept: 54.0960
    Epoch [2/50], Test Losses: mse: 16.1028, mae: 3.0707, huber: 2.6123, swd: 1.7072, ept: 68.4180
      Epoch 2 composite train-obj: 1.329223
            No improvement (2.6826), counter 1/5
    Epoch [3/50], Train Losses: mse: 2.2793, mae: 1.0721, huber: 0.6841, swd: 0.3103, ept: 240.3806
    Epoch [3/50], Val Losses: mse: 16.9765, mae: 3.2192, huber: 2.7547, swd: 1.5654, ept: 55.4222
    Epoch [3/50], Test Losses: mse: 16.4884, mae: 3.0844, huber: 2.6275, swd: 1.6901, ept: 85.0154
      Epoch 3 composite train-obj: 0.684097
            No improvement (2.7547), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.9416, mae: 0.7074, huber: 0.3610, swd: 0.1214, ept: 308.8337
    Epoch [4/50], Val Losses: mse: 16.8963, mae: 3.2070, huber: 2.7420, swd: 1.5865, ept: 56.9606
    Epoch [4/50], Test Losses: mse: 16.3783, mae: 3.0725, huber: 2.6158, swd: 1.6372, ept: 90.4344
      Epoch 4 composite train-obj: 0.361016
            No improvement (2.7420), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.5265, mae: 0.5425, huber: 0.2291, swd: 0.0695, ept: 327.1507
    Epoch [5/50], Val Losses: mse: 16.7415, mae: 3.2003, huber: 2.7353, swd: 1.5729, ept: 59.9488
    Epoch [5/50], Test Losses: mse: 16.1209, mae: 3.0468, huber: 2.5906, swd: 1.6752, ept: 92.7491
      Epoch 5 composite train-obj: 0.229128
            No improvement (2.7353), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.3553, mae: 0.4513, huber: 0.1640, swd: 0.0493, ept: 333.1418
    Epoch [6/50], Val Losses: mse: 16.4846, mae: 3.1774, huber: 2.7124, swd: 1.5825, ept: 61.1021
    Epoch [6/50], Test Losses: mse: 15.9561, mae: 3.0276, huber: 2.5718, swd: 1.6494, ept: 95.2615
      Epoch 6 composite train-obj: 0.163953
    Epoch [6/50], Test Losses: mse: 14.3255, mae: 2.9543, huber: 2.4939, swd: 2.1482, ept: 61.0520
    Best round's Test MSE: 14.3255, MAE: 2.9543, SWD: 2.1482
    Best round's Validation MSE: 13.4461, MAE: 2.8781, SWD: 1.7990
    Best round's Test verification MSE : 14.3255, MAE: 2.9543, SWD: 2.1482
    Time taken: 72.73 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz96_seq720_pred336_20250512_2254)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 14.4557 ± 0.1483
      mae: 2.9664 ± 0.0126
      huber: 2.5061 ± 0.0127
      swd: 2.0394 ± 0.1625
      ept: 60.2894 ± 3.1337
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 13.7337 ± 0.2846
      mae: 2.9180 ± 0.0328
      huber: 2.4553 ± 0.0328
      swd: 1.7154 ± 0.1062
      ept: 52.6864 ± 4.7137
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 217.49 seconds
    
    Experiment complete: TimeMixer_lorenz96_seq720_pred336_20250512_2254
    Model: TimeMixer
    Dataset: lorenz96
    Sequence Length: 720
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=720,
    pred_len=720,
    channels=data_mgr.datasets['lorenz96']['channels'],
    enc_in=data_mgr.datasets['lorenz96']['channels'],
    dec_in=data_mgr.datasets['lorenz96']['channels'],
    c_out=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp_mixer_96_96 = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([720, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([720, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 720, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 4
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.0307, mae: 2.8697, huber: 2.4069, swd: 2.3198, ept: 40.6562
    Epoch [1/50], Val Losses: mse: 15.1221, mae: 3.1169, huber: 2.6499, swd: 1.1554, ept: 62.3878
    Epoch [1/50], Test Losses: mse: 15.1240, mae: 3.0932, huber: 2.6283, swd: 2.3130, ept: 59.6414
      Epoch 1 composite train-obj: 2.406950
            Val objective improved inf → 2.6499, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.2641, mae: 2.3445, huber: 1.8945, swd: 1.4695, ept: 80.4412
    Epoch [2/50], Val Losses: mse: 19.0528, mae: 3.4870, huber: 3.0172, swd: 0.9058, ept: 46.1922
    Epoch [2/50], Test Losses: mse: 16.1869, mae: 3.1559, huber: 2.6920, swd: 1.6586, ept: 60.5717
      Epoch 2 composite train-obj: 1.894522
            No improvement (3.0172), counter 1/5
    Epoch [3/50], Train Losses: mse: 6.2402, mae: 1.8539, huber: 1.4203, swd: 0.8065, ept: 118.2460
    Epoch [3/50], Val Losses: mse: 20.8042, mae: 3.6351, huber: 3.1648, swd: 0.5949, ept: 47.3134
    Epoch [3/50], Test Losses: mse: 18.2254, mae: 3.3235, huber: 2.8590, swd: 1.1179, ept: 63.6631
      Epoch 3 composite train-obj: 1.420346
            No improvement (3.1648), counter 2/5
    Epoch [4/50], Train Losses: mse: 3.7928, mae: 1.4126, huber: 0.9988, swd: 0.4166, ept: 204.6501
    Epoch [4/50], Val Losses: mse: 22.1323, mae: 3.7565, huber: 3.2853, swd: 0.6613, ept: 46.0179
    Epoch [4/50], Test Losses: mse: 19.3834, mae: 3.4222, huber: 2.9569, swd: 1.0874, ept: 68.7390
      Epoch 4 composite train-obj: 0.998783
            No improvement (3.2853), counter 3/5
    Epoch [5/50], Train Losses: mse: 2.1761, mae: 1.0659, huber: 0.6751, swd: 0.2149, ept: 368.7022
    Epoch [5/50], Val Losses: mse: 22.3235, mae: 3.7720, huber: 3.3005, swd: 0.6857, ept: 47.8409
    Epoch [5/50], Test Losses: mse: 19.5896, mae: 3.4354, huber: 2.9703, swd: 1.0659, ept: 79.4025
      Epoch 5 composite train-obj: 0.675122
            No improvement (3.3005), counter 4/5
    Epoch [6/50], Train Losses: mse: 1.3385, mae: 0.8448, huber: 0.4759, swd: 0.1239, ept: 526.6075
    Epoch [6/50], Val Losses: mse: 22.2427, mae: 3.7664, huber: 3.2949, swd: 0.7167, ept: 46.4792
    Epoch [6/50], Test Losses: mse: 19.6196, mae: 3.4370, huber: 2.9719, swd: 1.0625, ept: 83.1607
      Epoch 6 composite train-obj: 0.475892
    Epoch [6/50], Test Losses: mse: 15.1240, mae: 3.0932, huber: 2.6283, swd: 2.3130, ept: 59.6414
    Best round's Test MSE: 15.1240, MAE: 3.0932, SWD: 2.3130
    Best round's Validation MSE: 15.1221, MAE: 3.1169, SWD: 1.1554
    Best round's Test verification MSE : 15.1240, MAE: 3.0932, SWD: 2.3130
    Time taken: 74.64 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 14.4088, mae: 2.9836, huber: 2.5198, swd: 2.2153, ept: 32.8314
    Epoch [1/50], Val Losses: mse: 14.9525, mae: 3.1015, huber: 2.6340, swd: 1.5743, ept: 51.5022
    Epoch [1/50], Test Losses: mse: 15.2193, mae: 3.1003, huber: 2.6357, swd: 2.6756, ept: 60.4398
      Epoch 1 composite train-obj: 2.519848
            Val objective improved inf → 2.6340, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.1877, mae: 2.3330, huber: 1.8832, swd: 1.4675, ept: 81.1217
    Epoch [2/50], Val Losses: mse: 18.3658, mae: 3.4137, huber: 2.9443, swd: 1.1147, ept: 53.3094
    Epoch [2/50], Test Losses: mse: 16.8435, mae: 3.2179, huber: 2.7537, swd: 1.7921, ept: 60.7477
      Epoch 2 composite train-obj: 1.883217
            No improvement (2.9443), counter 1/5
    Epoch [3/50], Train Losses: mse: 5.9942, mae: 1.8115, huber: 1.3794, swd: 0.7817, ept: 125.5123
    Epoch [3/50], Val Losses: mse: 20.2587, mae: 3.5722, huber: 3.1023, swd: 0.8742, ept: 49.4949
    Epoch [3/50], Test Losses: mse: 18.1946, mae: 3.3198, huber: 2.8557, swd: 1.3677, ept: 65.1169
      Epoch 3 composite train-obj: 1.379442
            No improvement (3.1023), counter 2/5
    Epoch [4/50], Train Losses: mse: 3.6083, mae: 1.3779, huber: 0.9659, swd: 0.3997, ept: 214.2826
    Epoch [4/50], Val Losses: mse: 21.5951, mae: 3.6765, huber: 3.2062, swd: 0.7923, ept: 52.2763
    Epoch [4/50], Test Losses: mse: 19.3280, mae: 3.4201, huber: 2.9552, swd: 1.1345, ept: 68.5003
      Epoch 4 composite train-obj: 0.965888
            No improvement (3.2062), counter 3/5
    Epoch [5/50], Train Losses: mse: 2.1316, mae: 1.0596, huber: 0.6688, swd: 0.2095, ept: 369.3385
    Epoch [5/50], Val Losses: mse: 22.0593, mae: 3.7091, huber: 3.2392, swd: 0.8113, ept: 53.3536
    Epoch [5/50], Test Losses: mse: 19.6622, mae: 3.4432, huber: 2.9786, swd: 1.1597, ept: 79.3011
      Epoch 5 composite train-obj: 0.668845
            No improvement (3.2392), counter 4/5
    Epoch [6/50], Train Losses: mse: 1.3418, mae: 0.8508, huber: 0.4807, swd: 0.1235, ept: 519.1952
    Epoch [6/50], Val Losses: mse: 22.2133, mae: 3.7221, huber: 3.2518, swd: 0.7411, ept: 52.4394
    Epoch [6/50], Test Losses: mse: 19.6858, mae: 3.4437, huber: 2.9790, swd: 1.0988, ept: 84.5369
      Epoch 6 composite train-obj: 0.480678
    Epoch [6/50], Test Losses: mse: 15.2193, mae: 3.1003, huber: 2.6357, swd: 2.6756, ept: 60.4398
    Best round's Test MSE: 15.2193, MAE: 3.1003, SWD: 2.6756
    Best round's Validation MSE: 14.9525, MAE: 3.1015, SWD: 1.5743
    Best round's Test verification MSE : 15.2193, MAE: 3.1003, SWD: 2.6756
    Time taken: 75.30 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.6454, mae: 2.9454, huber: 2.4815, swd: 2.6147, ept: 41.8130
    Epoch [1/50], Val Losses: mse: 14.4692, mae: 3.0738, huber: 2.6067, swd: 1.9874, ept: 66.2578
    Epoch [1/50], Test Losses: mse: 14.8964, mae: 3.0851, huber: 2.6195, swd: 3.3398, ept: 64.1586
      Epoch 1 composite train-obj: 2.481502
            Val objective improved inf → 2.6067, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.5125, mae: 2.5334, huber: 2.0785, swd: 1.9289, ept: 78.8976
    Epoch [2/50], Val Losses: mse: 17.5646, mae: 3.3654, huber: 2.8964, swd: 1.2877, ept: 53.2520
    Epoch [2/50], Test Losses: mse: 15.7083, mae: 3.1341, huber: 2.6697, swd: 2.2332, ept: 65.3529
      Epoch 2 composite train-obj: 2.078526
            No improvement (2.8964), counter 1/5
    Epoch [3/50], Train Losses: mse: 8.1022, mae: 2.1597, huber: 1.7156, swd: 1.2510, ept: 101.8640
    Epoch [3/50], Val Losses: mse: 20.1399, mae: 3.5812, huber: 3.1118, swd: 1.1689, ept: 52.5152
    Epoch [3/50], Test Losses: mse: 17.3377, mae: 3.2533, huber: 2.7894, swd: 1.6769, ept: 65.5607
      Epoch 3 composite train-obj: 1.715605
            No improvement (3.1118), counter 2/5
    Epoch [4/50], Train Losses: mse: 5.6546, mae: 1.7528, huber: 1.3233, swd: 0.7358, ept: 138.9690
    Epoch [4/50], Val Losses: mse: 21.3522, mae: 3.6935, huber: 3.2225, swd: 0.8902, ept: 47.8510
    Epoch [4/50], Test Losses: mse: 18.7327, mae: 3.3655, huber: 2.9015, swd: 1.2981, ept: 67.4885
      Epoch 4 composite train-obj: 1.323271
            No improvement (3.2225), counter 3/5
    Epoch [5/50], Train Losses: mse: 3.5360, mae: 1.3600, huber: 0.9494, swd: 0.3952, ept: 221.6696
    Epoch [5/50], Val Losses: mse: 21.7881, mae: 3.7370, huber: 3.2653, swd: 0.8167, ept: 45.1921
    Epoch [5/50], Test Losses: mse: 19.6498, mae: 3.4421, huber: 2.9773, swd: 1.1408, ept: 69.1127
      Epoch 5 composite train-obj: 0.949421
            No improvement (3.2653), counter 4/5
    Epoch [6/50], Train Losses: mse: 2.1101, mae: 1.0500, huber: 0.6607, swd: 0.2120, ept: 369.3263
    Epoch [6/50], Val Losses: mse: 22.1540, mae: 3.7778, huber: 3.3055, swd: 0.8210, ept: 47.5790
    Epoch [6/50], Test Losses: mse: 19.6622, mae: 3.4471, huber: 2.9819, swd: 1.1332, ept: 76.4374
      Epoch 6 composite train-obj: 0.660714
    Epoch [6/50], Test Losses: mse: 14.8964, mae: 3.0851, huber: 2.6195, swd: 3.3398, ept: 64.1586
    Best round's Test MSE: 14.8964, MAE: 3.0851, SWD: 3.3398
    Best round's Validation MSE: 14.4692, MAE: 3.0738, SWD: 1.9874
    Best round's Test verification MSE : 14.8964, MAE: 3.0851, SWD: 3.3398
    Time taken: 74.96 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz96_seq720_pred720_20250512_2257)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 15.0799 ± 0.1355
      mae: 3.0929 ± 0.0062
      huber: 2.6278 ± 0.0066
      swd: 2.7761 ± 0.4252
      ept: 61.4133 ± 1.9684
      count: 4.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 14.8479 ± 0.2766
      mae: 3.0974 ± 0.0178
      huber: 2.6302 ± 0.0178
      swd: 1.5724 ± 0.3397
      ept: 60.0493 ± 6.2468
      count: 4.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 225.07 seconds
    
    Experiment complete: TimeMixer_lorenz96_seq720_pred720_20250512_2257
    Model: TimeMixer
    Dataset: lorenz96
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
    channels=data_mgr.datasets['lorenz96']['channels'],
    enc_in=data_mgr.datasets['lorenz96']['channels'],
    dec_in=data_mgr.datasets['lorenz96']['channels'],
    c_out=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([96, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([96, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 98
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 96, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 98
    Validation Batches: 9
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 16.0178, mae: 2.9856, huber: 2.5258, swd: 4.1583, ept: 18.8505
    Epoch [1/50], Val Losses: mse: 11.4126, mae: 2.5719, huber: 2.1224, swd: 3.3593, ept: 44.2890
    Epoch [1/50], Test Losses: mse: 10.6105, mae: 2.4682, huber: 2.0203, swd: 3.2658, ept: 48.3277
      Epoch 1 composite train-obj: 2.525765
            Val objective improved inf → 2.1224, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.6680, mae: 2.3801, huber: 1.9317, swd: 2.4906, ept: 29.2203
    Epoch [2/50], Val Losses: mse: 10.1169, mae: 2.3996, huber: 1.9554, swd: 3.0302, ept: 49.7234
    Epoch [2/50], Test Losses: mse: 10.1113, mae: 2.4051, huber: 1.9606, swd: 3.2881, ept: 50.0491
      Epoch 2 composite train-obj: 1.931658
            Val objective improved 2.1224 → 1.9554, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.6129, mae: 2.2217, huber: 1.7785, swd: 2.0962, ept: 30.8338
    Epoch [3/50], Val Losses: mse: 10.0907, mae: 2.4093, huber: 1.9632, swd: 2.5095, ept: 49.1544
    Epoch [3/50], Test Losses: mse: 9.1277, mae: 2.2629, huber: 1.8215, swd: 2.4195, ept: 54.0114
      Epoch 3 composite train-obj: 1.778466
            No improvement (1.9632), counter 1/5
    Epoch [4/50], Train Losses: mse: 7.4920, mae: 2.0518, huber: 1.6143, swd: 1.7655, ept: 33.0137
    Epoch [4/50], Val Losses: mse: 8.9798, mae: 2.2087, huber: 1.7720, swd: 2.1681, ept: 57.5460
    Epoch [4/50], Test Losses: mse: 9.0364, mae: 2.2160, huber: 1.7792, swd: 2.4603, ept: 56.9231
      Epoch 4 composite train-obj: 1.614285
            Val objective improved 1.9554 → 1.7720, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.6101, mae: 1.9114, huber: 1.4791, swd: 1.5191, ept: 34.8903
    Epoch [5/50], Val Losses: mse: 8.5238, mae: 2.1878, huber: 1.7483, swd: 1.7941, ept: 57.7709
    Epoch [5/50], Test Losses: mse: 8.3289, mae: 2.1249, huber: 1.6910, swd: 2.1609, ept: 59.4390
      Epoch 5 composite train-obj: 1.479121
            Val objective improved 1.7720 → 1.7483, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.8573, mae: 1.7810, huber: 1.3547, swd: 1.3223, ept: 36.7181
    Epoch [6/50], Val Losses: mse: 8.0246, mae: 2.0956, huber: 1.6610, swd: 1.8445, ept: 61.7288
    Epoch [6/50], Test Losses: mse: 7.8419, mae: 2.0445, huber: 1.6137, swd: 2.2141, ept: 61.8267
      Epoch 6 composite train-obj: 1.354683
            Val objective improved 1.7483 → 1.6610, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.3402, mae: 1.6924, huber: 1.2697, swd: 1.1843, ept: 37.9105
    Epoch [7/50], Val Losses: mse: 7.6992, mae: 2.0613, huber: 1.6257, swd: 1.7839, ept: 63.7124
    Epoch [7/50], Test Losses: mse: 7.8089, mae: 2.0385, huber: 1.6074, swd: 2.3848, ept: 62.4609
      Epoch 7 composite train-obj: 1.269710
            Val objective improved 1.6610 → 1.6257, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 4.8919, mae: 1.6086, huber: 1.1904, swd: 1.0688, ept: 38.7944
    Epoch [8/50], Val Losses: mse: 7.7906, mae: 2.0308, huber: 1.6009, swd: 1.6306, ept: 65.1330
    Epoch [8/50], Test Losses: mse: 7.6146, mae: 1.9894, huber: 1.5617, swd: 1.8988, ept: 64.8266
      Epoch 8 composite train-obj: 1.190390
            Val objective improved 1.6257 → 1.6009, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 4.4731, mae: 1.5219, huber: 1.1096, swd: 0.9568, ept: 39.7104
    Epoch [9/50], Val Losses: mse: 8.4891, mae: 2.1434, huber: 1.7093, swd: 1.8090, ept: 61.4885
    Epoch [9/50], Test Losses: mse: 7.5995, mae: 1.9778, huber: 1.5535, swd: 1.7761, ept: 65.5494
      Epoch 9 composite train-obj: 1.109582
            No improvement (1.7093), counter 1/5
    Epoch [10/50], Train Losses: mse: 4.1892, mae: 1.4676, huber: 1.0578, swd: 0.8750, ept: 40.4542
    Epoch [10/50], Val Losses: mse: 7.9326, mae: 2.0439, huber: 1.6165, swd: 1.7076, ept: 64.2051
    Epoch [10/50], Test Losses: mse: 7.4436, mae: 1.9513, huber: 1.5277, swd: 1.9561, ept: 66.2089
      Epoch 10 composite train-obj: 1.057791
            No improvement (1.6165), counter 2/5
    Epoch [11/50], Train Losses: mse: 3.9089, mae: 1.4081, huber: 1.0020, swd: 0.8034, ept: 41.0403
    Epoch [11/50], Val Losses: mse: 7.5927, mae: 2.0059, huber: 1.5798, swd: 1.6359, ept: 65.2703
    Epoch [11/50], Test Losses: mse: 7.2960, mae: 1.9365, huber: 1.5116, swd: 1.9124, ept: 66.9860
      Epoch 11 composite train-obj: 1.002038
            Val objective improved 1.6009 → 1.5798, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 3.6699, mae: 1.3558, huber: 0.9536, swd: 0.7405, ept: 41.3974
    Epoch [12/50], Val Losses: mse: 6.9786, mae: 1.9277, huber: 1.5000, swd: 1.5330, ept: 67.5851
    Epoch [12/50], Test Losses: mse: 7.4234, mae: 1.9502, huber: 1.5252, swd: 2.0652, ept: 67.4288
      Epoch 12 composite train-obj: 0.953564
            Val objective improved 1.5798 → 1.5000, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 3.4897, mae: 1.3160, huber: 0.9165, swd: 0.6914, ept: 41.7394
    Epoch [13/50], Val Losses: mse: 7.5021, mae: 1.9879, huber: 1.5605, swd: 1.6249, ept: 67.0560
    Epoch [13/50], Test Losses: mse: 7.3634, mae: 1.9359, huber: 1.5117, swd: 1.9434, ept: 67.4690
      Epoch 13 composite train-obj: 0.916545
            No improvement (1.5605), counter 1/5
    Epoch [14/50], Train Losses: mse: 3.3478, mae: 1.2834, huber: 0.8861, swd: 0.6470, ept: 41.9338
    Epoch [14/50], Val Losses: mse: 7.3214, mae: 1.9631, huber: 1.5380, swd: 1.6324, ept: 67.5390
    Epoch [14/50], Test Losses: mse: 7.4362, mae: 1.9459, huber: 1.5218, swd: 2.0152, ept: 67.5834
      Epoch 14 composite train-obj: 0.886109
            No improvement (1.5380), counter 2/5
    Epoch [15/50], Train Losses: mse: 3.1807, mae: 1.2419, huber: 0.8480, swd: 0.5979, ept: 42.1280
    Epoch [15/50], Val Losses: mse: 7.4895, mae: 1.9686, huber: 1.5447, swd: 1.5496, ept: 68.4785
    Epoch [15/50], Test Losses: mse: 7.2982, mae: 1.9234, huber: 1.5008, swd: 1.8214, ept: 68.0076
      Epoch 15 composite train-obj: 0.847961
            No improvement (1.5447), counter 3/5
    Epoch [16/50], Train Losses: mse: 3.0668, mae: 1.2136, huber: 0.8217, swd: 0.5606, ept: 42.3509
    Epoch [16/50], Val Losses: mse: 7.3348, mae: 1.9727, huber: 1.5460, swd: 1.5367, ept: 67.4808
    Epoch [16/50], Test Losses: mse: 7.2039, mae: 1.9023, huber: 1.4796, swd: 1.7906, ept: 69.3848
      Epoch 16 composite train-obj: 0.821713
            No improvement (1.5460), counter 4/5
    Epoch [17/50], Train Losses: mse: 2.9440, mae: 1.1807, huber: 0.7924, swd: 0.5261, ept: 42.4954
    Epoch [17/50], Val Losses: mse: 7.4546, mae: 1.9721, huber: 1.5470, swd: 1.4604, ept: 68.3421
    Epoch [17/50], Test Losses: mse: 7.2843, mae: 1.9065, huber: 1.4870, swd: 1.8744, ept: 68.8143
      Epoch 17 composite train-obj: 0.792370
    Epoch [17/50], Test Losses: mse: 7.4234, mae: 1.9502, huber: 1.5252, swd: 2.0652, ept: 67.4288
    Best round's Test MSE: 7.4234, MAE: 1.9502, SWD: 2.0652
    Best round's Validation MSE: 6.9786, MAE: 1.9277, SWD: 1.5330
    Best round's Test verification MSE : 7.4234, MAE: 1.9502, SWD: 2.0652
    Time taken: 146.29 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 15.6809, mae: 2.9586, huber: 2.4992, swd: 4.0848, ept: 19.5021
    Epoch [1/50], Val Losses: mse: 10.9392, mae: 2.5321, huber: 2.0832, swd: 2.9071, ept: 43.7155
    Epoch [1/50], Test Losses: mse: 10.2032, mae: 2.4214, huber: 1.9742, swd: 3.0434, ept: 48.5498
      Epoch 1 composite train-obj: 2.499171
            Val objective improved inf → 2.0832, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.4867, mae: 2.3497, huber: 1.9020, swd: 2.3710, ept: 29.3503
    Epoch [2/50], Val Losses: mse: 10.5563, mae: 2.4354, huber: 1.9925, swd: 2.5230, ept: 49.4861
    Epoch [2/50], Test Losses: mse: 10.4755, mae: 2.4040, huber: 1.9605, swd: 2.7559, ept: 50.8023
      Epoch 2 composite train-obj: 1.901999
            Val objective improved 2.0832 → 1.9925, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 8.3418, mae: 2.1764, huber: 1.7348, swd: 1.9786, ept: 31.2893
    Epoch [3/50], Val Losses: mse: 9.6408, mae: 2.3588, huber: 1.9145, swd: 2.3097, ept: 51.2771
    Epoch [3/50], Test Losses: mse: 8.9268, mae: 2.2378, huber: 1.7976, swd: 2.7696, ept: 54.4149
      Epoch 3 composite train-obj: 1.734769
            Val objective improved 1.9925 → 1.9145, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.3413, mae: 2.0217, huber: 1.5856, swd: 1.7033, ept: 33.3457
    Epoch [4/50], Val Losses: mse: 8.6736, mae: 2.2206, huber: 1.7816, swd: 1.9644, ept: 55.8871
    Epoch [4/50], Test Losses: mse: 8.7869, mae: 2.2002, huber: 1.7623, swd: 2.5135, ept: 56.7452
      Epoch 4 composite train-obj: 1.585645
            Val objective improved 1.9145 → 1.7816, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.5717, mae: 1.8958, huber: 1.4649, swd: 1.5039, ept: 34.9962
    Epoch [5/50], Val Losses: mse: 9.2383, mae: 2.2610, huber: 1.8208, swd: 2.0026, ept: 56.9300
    Epoch [5/50], Test Losses: mse: 8.4730, mae: 2.1368, huber: 1.7020, swd: 2.2174, ept: 59.8923
      Epoch 5 composite train-obj: 1.464940
            No improvement (1.8208), counter 1/5
    Epoch [6/50], Train Losses: mse: 5.9162, mae: 1.7824, huber: 1.3570, swd: 1.3324, ept: 36.5310
    Epoch [6/50], Val Losses: mse: 8.9674, mae: 2.1960, huber: 1.7565, swd: 1.8618, ept: 59.6291
    Epoch [6/50], Test Losses: mse: 8.0246, mae: 2.0758, huber: 1.6422, swd: 2.1186, ept: 62.1684
      Epoch 6 composite train-obj: 1.356950
            Val objective improved 1.7816 → 1.7565, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.4093, mae: 1.6963, huber: 1.2747, swd: 1.2049, ept: 37.7549
    Epoch [7/50], Val Losses: mse: 9.2771, mae: 2.1877, huber: 1.7554, swd: 1.9907, ept: 61.5927
    Epoch [7/50], Test Losses: mse: 8.0740, mae: 2.0588, huber: 1.6300, swd: 2.0129, ept: 62.7083
      Epoch 7 composite train-obj: 1.274738
            Val objective improved 1.7565 → 1.7554, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 4.9270, mae: 1.6075, huber: 1.1906, swd: 1.0869, ept: 38.7761
    Epoch [8/50], Val Losses: mse: 9.1192, mae: 2.2408, huber: 1.7986, swd: 2.1676, ept: 60.1222
    Epoch [8/50], Test Losses: mse: 8.0531, mae: 2.0552, huber: 1.6252, swd: 2.0027, ept: 63.2205
      Epoch 8 composite train-obj: 1.190648
            No improvement (1.7986), counter 1/5
    Epoch [9/50], Train Losses: mse: 4.5201, mae: 1.5332, huber: 1.1193, swd: 0.9773, ept: 39.5899
    Epoch [9/50], Val Losses: mse: 8.2999, mae: 2.1040, huber: 1.6688, swd: 1.7441, ept: 65.1281
    Epoch [9/50], Test Losses: mse: 7.5185, mae: 1.9642, huber: 1.5389, swd: 2.0232, ept: 65.7160
      Epoch 9 composite train-obj: 1.119347
            Val objective improved 1.7554 → 1.6688, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 4.2044, mae: 1.4695, huber: 1.0602, swd: 0.8968, ept: 40.2105
    Epoch [10/50], Val Losses: mse: 8.7094, mae: 2.1612, huber: 1.7237, swd: 1.6867, ept: 61.8934
    Epoch [10/50], Test Losses: mse: 7.5497, mae: 1.9713, huber: 1.5451, swd: 1.9159, ept: 66.4060
      Epoch 10 composite train-obj: 1.060221
            No improvement (1.7237), counter 1/5
    Epoch [11/50], Train Losses: mse: 3.9367, mae: 1.4179, huber: 1.0106, swd: 0.8208, ept: 40.6902
    Epoch [11/50], Val Losses: mse: 8.6291, mae: 2.1548, huber: 1.7201, swd: 1.6649, ept: 62.4125
    Epoch [11/50], Test Losses: mse: 7.4281, mae: 1.9656, huber: 1.5392, swd: 1.7870, ept: 67.1045
      Epoch 11 composite train-obj: 1.010638
            No improvement (1.7201), counter 2/5
    Epoch [12/50], Train Losses: mse: 3.6966, mae: 1.3633, huber: 0.9605, swd: 0.7508, ept: 41.1492
    Epoch [12/50], Val Losses: mse: 8.0674, mae: 2.0556, huber: 1.6255, swd: 1.6298, ept: 65.6143
    Epoch [12/50], Test Losses: mse: 7.4459, mae: 1.9454, huber: 1.5222, swd: 1.9320, ept: 67.2269
      Epoch 12 composite train-obj: 0.960540
            Val objective improved 1.6688 → 1.6255, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 3.4654, mae: 1.3118, huber: 0.9125, swd: 0.6850, ept: 41.6821
    Epoch [13/50], Val Losses: mse: 8.2522, mae: 2.0914, huber: 1.6551, swd: 1.5925, ept: 64.6058
    Epoch [13/50], Test Losses: mse: 7.5062, mae: 1.9583, huber: 1.5325, swd: 1.9062, ept: 68.0596
      Epoch 13 composite train-obj: 0.912459
            No improvement (1.6551), counter 1/5
    Epoch [14/50], Train Losses: mse: 3.3288, mae: 1.2812, huber: 0.8838, swd: 0.6467, ept: 41.8814
    Epoch [14/50], Val Losses: mse: 7.9142, mae: 2.0288, huber: 1.6004, swd: 1.5485, ept: 66.4592
    Epoch [14/50], Test Losses: mse: 7.3884, mae: 1.9307, huber: 1.5068, swd: 1.9889, ept: 69.0050
      Epoch 14 composite train-obj: 0.883799
            Val objective improved 1.6255 → 1.6004, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 3.1642, mae: 1.2395, huber: 0.8460, swd: 0.6010, ept: 42.1181
    Epoch [15/50], Val Losses: mse: 8.2693, mae: 2.0633, huber: 1.6353, swd: 1.5528, ept: 65.8243
    Epoch [15/50], Test Losses: mse: 7.4724, mae: 1.9429, huber: 1.5222, swd: 1.9320, ept: 68.5942
      Epoch 15 composite train-obj: 0.846016
            No improvement (1.6353), counter 1/5
    Epoch [16/50], Train Losses: mse: 3.0365, mae: 1.2078, huber: 0.8170, swd: 0.5607, ept: 42.3047
    Epoch [16/50], Val Losses: mse: 7.7403, mae: 2.0083, huber: 1.5800, swd: 1.5470, ept: 67.2453
    Epoch [16/50], Test Losses: mse: 7.6493, mae: 1.9676, huber: 1.5444, swd: 2.1041, ept: 68.4699
      Epoch 16 composite train-obj: 0.817010
            Val objective improved 1.6004 → 1.5800, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 2.9540, mae: 1.1875, huber: 0.7984, swd: 0.5359, ept: 42.4538
    Epoch [17/50], Val Losses: mse: 8.1675, mae: 2.0693, huber: 1.6394, swd: 1.5535, ept: 65.3262
    Epoch [17/50], Test Losses: mse: 7.5078, mae: 1.9603, huber: 1.5358, swd: 1.9397, ept: 68.1425
      Epoch 17 composite train-obj: 0.798407
            No improvement (1.6394), counter 1/5
    Epoch [18/50], Train Losses: mse: 2.8398, mae: 1.1557, huber: 0.7698, swd: 0.5024, ept: 42.5002
    Epoch [18/50], Val Losses: mse: 7.8205, mae: 2.0374, huber: 1.6076, swd: 1.5327, ept: 66.5569
    Epoch [18/50], Test Losses: mse: 7.2855, mae: 1.9216, huber: 1.5003, swd: 1.9718, ept: 69.1478
      Epoch 18 composite train-obj: 0.769850
            No improvement (1.6076), counter 2/5
    Epoch [19/50], Train Losses: mse: 2.7675, mae: 1.1376, huber: 0.7530, swd: 0.4845, ept: 42.6981
    Epoch [19/50], Val Losses: mse: 8.1208, mae: 2.0670, huber: 1.6386, swd: 1.5723, ept: 65.2124
    Epoch [19/50], Test Losses: mse: 7.5000, mae: 1.9403, huber: 1.5190, swd: 2.0030, ept: 68.9656
      Epoch 19 composite train-obj: 0.753035
            No improvement (1.6386), counter 3/5
    Epoch [20/50], Train Losses: mse: 2.6785, mae: 1.1108, huber: 0.7292, swd: 0.4549, ept: 42.6727
    Epoch [20/50], Val Losses: mse: 7.6932, mae: 2.0325, huber: 1.6012, swd: 1.5411, ept: 66.5959
    Epoch [20/50], Test Losses: mse: 7.4967, mae: 1.9519, huber: 1.5288, swd: 2.0117, ept: 68.7482
      Epoch 20 composite train-obj: 0.729158
            No improvement (1.6012), counter 4/5
    Epoch [21/50], Train Losses: mse: 2.6266, mae: 1.0971, huber: 0.7168, swd: 0.4419, ept: 42.8107
    Epoch [21/50], Val Losses: mse: 7.7073, mae: 2.0259, huber: 1.5950, swd: 1.5360, ept: 66.8164
    Epoch [21/50], Test Losses: mse: 7.4923, mae: 1.9567, huber: 1.5312, swd: 1.9893, ept: 69.0617
      Epoch 21 composite train-obj: 0.716753
    Epoch [21/50], Test Losses: mse: 7.6493, mae: 1.9676, huber: 1.5444, swd: 2.1041, ept: 68.4699
    Best round's Test MSE: 7.6493, MAE: 1.9676, SWD: 2.1041
    Best round's Validation MSE: 7.7403, MAE: 2.0083, SWD: 1.5470
    Best round's Test verification MSE : 7.6493, MAE: 1.9676, SWD: 2.1041
    Time taken: 178.06 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 15.9371, mae: 2.9774, huber: 2.5178, swd: 3.9997, ept: 18.9490
    Epoch [1/50], Val Losses: mse: 10.6876, mae: 2.4794, huber: 2.0336, swd: 2.4685, ept: 47.0872
    Epoch [1/50], Test Losses: mse: 10.0031, mae: 2.3935, huber: 1.9486, swd: 2.7124, ept: 48.6754
      Epoch 1 composite train-obj: 2.517838
            Val objective improved inf → 2.0336, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.4675, mae: 2.3474, huber: 1.9002, swd: 2.1243, ept: 29.4673
    Epoch [2/50], Val Losses: mse: 10.9742, mae: 2.4805, huber: 2.0349, swd: 2.1999, ept: 47.8706
    Epoch [2/50], Test Losses: mse: 9.6853, mae: 2.3209, huber: 1.8795, swd: 2.3435, ept: 51.7386
      Epoch 2 composite train-obj: 1.900227
            No improvement (2.0349), counter 1/5
    Epoch [3/50], Train Losses: mse: 8.4200, mae: 2.1949, huber: 1.7523, swd: 1.8261, ept: 31.0340
    Epoch [3/50], Val Losses: mse: 9.3473, mae: 2.3619, huber: 1.9164, swd: 2.0649, ept: 49.5903
    Epoch [3/50], Test Losses: mse: 8.7144, mae: 2.2272, huber: 1.7850, swd: 2.2648, ept: 54.3055
      Epoch 3 composite train-obj: 1.752326
            Val objective improved 2.0336 → 1.9164, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.5477, mae: 2.0619, huber: 1.6242, swd: 1.6152, ept: 32.9853
    Epoch [4/50], Val Losses: mse: 10.2153, mae: 2.3664, huber: 1.9258, swd: 2.0402, ept: 53.2444
    Epoch [4/50], Test Losses: mse: 8.9497, mae: 2.2141, huber: 1.7783, swd: 2.1835, ept: 56.0274
      Epoch 4 composite train-obj: 1.624208
            No improvement (1.9258), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.8985, mae: 1.9560, huber: 1.5225, swd: 1.4648, ept: 34.3356
    Epoch [5/50], Val Losses: mse: 8.2545, mae: 2.1655, huber: 1.7266, swd: 1.7453, ept: 57.0056
    Epoch [5/50], Test Losses: mse: 8.1379, mae: 2.1020, huber: 1.6688, swd: 2.1628, ept: 59.7128
      Epoch 5 composite train-obj: 1.522500
            Val objective improved 1.9164 → 1.7266, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.2237, mae: 1.8433, huber: 1.4146, swd: 1.3207, ept: 35.8405
    Epoch [6/50], Val Losses: mse: 7.9999, mae: 2.1161, huber: 1.6797, swd: 1.6658, ept: 58.5482
    Epoch [6/50], Test Losses: mse: 8.4537, mae: 2.1268, huber: 1.6936, swd: 2.2527, ept: 60.3012
      Epoch 6 composite train-obj: 1.414609
            Val objective improved 1.7266 → 1.6797, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.6467, mae: 1.7445, huber: 1.3200, swd: 1.1737, ept: 37.1575
    Epoch [7/50], Val Losses: mse: 7.9173, mae: 2.0816, huber: 1.6490, swd: 1.6361, ept: 60.6311
    Epoch [7/50], Test Losses: mse: 7.8950, mae: 2.0313, huber: 1.6037, swd: 1.9840, ept: 62.7217
      Epoch 7 composite train-obj: 1.319976
            Val objective improved 1.6797 → 1.6490, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.1545, mae: 1.6543, huber: 1.2344, swd: 1.0526, ept: 38.3279
    Epoch [8/50], Val Losses: mse: 8.3173, mae: 2.1412, huber: 1.7027, swd: 1.5433, ept: 60.7740
    Epoch [8/50], Test Losses: mse: 7.8255, mae: 2.0199, huber: 1.5918, swd: 1.9024, ept: 63.7118
      Epoch 8 composite train-obj: 1.234361
            No improvement (1.7027), counter 1/5
    Epoch [9/50], Train Losses: mse: 4.7209, mae: 1.5736, huber: 1.1574, swd: 0.9545, ept: 39.2586
    Epoch [9/50], Val Losses: mse: 7.7843, mae: 2.0501, huber: 1.6157, swd: 1.3720, ept: 63.7056
    Epoch [9/50], Test Losses: mse: 7.9568, mae: 2.0220, huber: 1.5939, swd: 1.8751, ept: 64.4120
      Epoch 9 composite train-obj: 1.157371
            Val objective improved 1.6490 → 1.6157, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 4.3796, mae: 1.5080, huber: 1.0954, swd: 0.8694, ept: 40.0787
    Epoch [10/50], Val Losses: mse: 8.0065, mae: 2.0822, huber: 1.6461, swd: 1.5964, ept: 63.4892
    Epoch [10/50], Test Losses: mse: 7.8176, mae: 1.9969, huber: 1.5709, swd: 1.9318, ept: 66.2207
      Epoch 10 composite train-obj: 1.095450
            No improvement (1.6461), counter 1/5
    Epoch [11/50], Train Losses: mse: 4.0972, mae: 1.4514, huber: 1.0418, swd: 0.7981, ept: 40.6113
    Epoch [11/50], Val Losses: mse: 8.2260, mae: 2.1333, huber: 1.6928, swd: 1.5854, ept: 63.8403
    Epoch [11/50], Test Losses: mse: 7.7805, mae: 1.9972, huber: 1.5697, swd: 1.8753, ept: 67.1367
      Epoch 11 composite train-obj: 1.041830
            No improvement (1.6928), counter 2/5
    Epoch [12/50], Train Losses: mse: 3.8623, mae: 1.3986, huber: 0.9934, swd: 0.7337, ept: 41.0107
    Epoch [12/50], Val Losses: mse: 7.8127, mae: 2.0288, huber: 1.5997, swd: 1.4574, ept: 65.8245
    Epoch [12/50], Test Losses: mse: 7.8979, mae: 1.9902, huber: 1.5665, swd: 1.8949, ept: 66.8429
      Epoch 12 composite train-obj: 0.993357
            Val objective improved 1.6157 → 1.5997, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 3.6362, mae: 1.3512, huber: 0.9483, swd: 0.6821, ept: 41.5572
    Epoch [13/50], Val Losses: mse: 7.7459, mae: 2.0344, huber: 1.6044, swd: 1.4360, ept: 66.2345
    Epoch [13/50], Test Losses: mse: 7.6963, mae: 1.9619, huber: 1.5402, swd: 1.8240, ept: 67.9396
      Epoch 13 composite train-obj: 0.948254
            No improvement (1.6044), counter 1/5
    Epoch [14/50], Train Losses: mse: 3.4653, mae: 1.3112, huber: 0.9119, swd: 0.6347, ept: 41.7866
    Epoch [14/50], Val Losses: mse: 8.2110, mae: 2.1094, huber: 1.6746, swd: 1.5784, ept: 64.8025
    Epoch [14/50], Test Losses: mse: 7.8121, mae: 1.9812, huber: 1.5562, swd: 1.7639, ept: 68.0250
      Epoch 14 composite train-obj: 0.911901
            No improvement (1.6746), counter 2/5
    Epoch [15/50], Train Losses: mse: 3.2969, mae: 1.2721, huber: 0.8752, swd: 0.5857, ept: 42.1641
    Epoch [15/50], Val Losses: mse: 7.7868, mae: 2.0660, huber: 1.6296, swd: 1.4922, ept: 64.9038
    Epoch [15/50], Test Losses: mse: 7.8591, mae: 1.9830, huber: 1.5587, swd: 1.8416, ept: 68.1109
      Epoch 15 composite train-obj: 0.875206
            No improvement (1.6296), counter 3/5
    Epoch [16/50], Train Losses: mse: 3.1730, mae: 1.2417, huber: 0.8478, swd: 0.5506, ept: 42.1503
    Epoch [16/50], Val Losses: mse: 7.8807, mae: 2.0433, huber: 1.6117, swd: 1.4189, ept: 67.4496
    Epoch [16/50], Test Losses: mse: 7.8393, mae: 1.9664, huber: 1.5449, swd: 1.8449, ept: 68.5113
      Epoch 16 composite train-obj: 0.847751
            No improvement (1.6117), counter 4/5
    Epoch [17/50], Train Losses: mse: 3.0304, mae: 1.2058, huber: 0.8146, swd: 0.5144, ept: 42.3990
    Epoch [17/50], Val Losses: mse: 7.7309, mae: 2.0624, huber: 1.6255, swd: 1.4045, ept: 67.0153
    Epoch [17/50], Test Losses: mse: 7.8268, mae: 1.9874, huber: 1.5611, swd: 1.7966, ept: 68.4046
      Epoch 17 composite train-obj: 0.814648
    Epoch [17/50], Test Losses: mse: 7.8979, mae: 1.9902, huber: 1.5665, swd: 1.8949, ept: 66.8429
    Best round's Test MSE: 7.8979, MAE: 1.9902, SWD: 1.8949
    Best round's Validation MSE: 7.8127, MAE: 2.0288, SWD: 1.4574
    Best round's Test verification MSE : 7.8979, MAE: 1.9902, SWD: 1.8949
    Time taken: 146.11 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz96_seq720_pred96_20250512_2301)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 7.6568 ± 0.1938
      mae: 1.9693 ± 0.0164
      huber: 1.5454 ± 0.0169
      swd: 2.0214 ± 0.0909
      ept: 67.5805 ± 0.6729
      count: 9.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 7.5105 ± 0.3773
      mae: 1.9883 ± 0.0437
      huber: 1.5599 ± 0.0431
      swd: 1.5125 ± 0.0394
      ept: 66.8850 ± 0.7626
      count: 9.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 470.56 seconds
    
    Experiment complete: PatchTST_lorenz96_seq720_pred96_20250512_2301
    Model: PatchTST
    Dataset: lorenz96
    Sequence Length: 720
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=720,
    pred_len=196,
    channels=data_mgr.datasets['lorenz96']['channels'],
    enc_in=data_mgr.datasets['lorenz96']['channels'],
    dec_in=data_mgr.datasets['lorenz96']['channels'],
    c_out=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([196, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([196, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 97
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 196, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 97
    Validation Batches: 8
    Test Batches: 23
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.4138, mae: 3.1672, huber: 2.7032, swd: 3.7543, ept: 18.8721
    Epoch [1/50], Val Losses: mse: 13.9545, mae: 2.9296, huber: 2.4683, swd: 2.4578, ept: 44.5113
    Epoch [1/50], Test Losses: mse: 13.6330, mae: 2.8740, huber: 2.4147, swd: 2.5985, ept: 49.5514
      Epoch 1 composite train-obj: 2.703202
            Val objective improved inf → 2.4683, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.6612, mae: 2.6712, huber: 2.2142, swd: 2.2549, ept: 29.0006
    Epoch [2/50], Val Losses: mse: 13.6499, mae: 2.8660, huber: 2.4080, swd: 2.1354, ept: 49.7920
    Epoch [2/50], Test Losses: mse: 13.2187, mae: 2.8069, huber: 2.3508, swd: 2.1879, ept: 52.5552
      Epoch 2 composite train-obj: 2.214159
            Val objective improved 2.4683 → 2.4080, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 10.8553, mae: 2.5561, huber: 2.1022, swd: 2.0236, ept: 31.3271
    Epoch [3/50], Val Losses: mse: 11.3208, mae: 2.5898, huber: 2.1375, swd: 2.0828, ept: 58.0072
    Epoch [3/50], Test Losses: mse: 12.5321, mae: 2.7332, huber: 2.2783, swd: 2.6040, ept: 57.0115
      Epoch 3 composite train-obj: 2.102172
            Val objective improved 2.4080 → 2.1375, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 9.7306, mae: 2.3986, huber: 1.9488, swd: 1.7550, ept: 33.0662
    Epoch [4/50], Val Losses: mse: 11.3210, mae: 2.5914, huber: 2.1377, swd: 1.8460, ept: 56.9930
    Epoch [4/50], Test Losses: mse: 11.9713, mae: 2.6473, huber: 2.1943, swd: 2.2316, ept: 60.4288
      Epoch 4 composite train-obj: 1.948759
            No improvement (2.1377), counter 1/5
    Epoch [5/50], Train Losses: mse: 8.6530, mae: 2.2422, huber: 1.7969, swd: 1.4950, ept: 35.3805
    Epoch [5/50], Val Losses: mse: 11.2019, mae: 2.5648, huber: 2.1137, swd: 1.8395, ept: 64.8210
    Epoch [5/50], Test Losses: mse: 11.7287, mae: 2.6088, huber: 2.1588, swd: 2.3137, ept: 66.3679
      Epoch 5 composite train-obj: 1.796916
            Val objective improved 2.1375 → 2.1137, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.6998, mae: 2.0946, huber: 1.6541, swd: 1.3155, ept: 38.0631
    Epoch [6/50], Val Losses: mse: 11.0506, mae: 2.5599, huber: 2.1077, swd: 1.6833, ept: 65.7435
    Epoch [6/50], Test Losses: mse: 11.7588, mae: 2.6076, huber: 2.1570, swd: 2.3488, ept: 69.5099
      Epoch 6 composite train-obj: 1.654123
            Val objective improved 2.1137 → 2.1077, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 6.9473, mae: 1.9745, huber: 1.5381, swd: 1.1724, ept: 40.1298
    Epoch [7/50], Val Losses: mse: 12.1746, mae: 2.6728, huber: 2.2206, swd: 1.5549, ept: 66.9633
    Epoch [7/50], Test Losses: mse: 12.0359, mae: 2.6196, huber: 2.1704, swd: 1.9077, ept: 70.0179
      Epoch 7 composite train-obj: 1.538146
            No improvement (2.2206), counter 1/5
    Epoch [8/50], Train Losses: mse: 6.3234, mae: 1.8723, huber: 1.4397, swd: 1.0517, ept: 41.6318
    Epoch [8/50], Val Losses: mse: 11.9484, mae: 2.6302, huber: 2.1795, swd: 1.4062, ept: 72.3293
    Epoch [8/50], Test Losses: mse: 11.8189, mae: 2.5871, huber: 2.1398, swd: 1.9611, ept: 73.0591
      Epoch 8 composite train-obj: 1.439690
            No improvement (2.1795), counter 2/5
    Epoch [9/50], Train Losses: mse: 5.7927, mae: 1.7821, huber: 1.3531, swd: 0.9543, ept: 43.0562
    Epoch [9/50], Val Losses: mse: 11.8042, mae: 2.6119, huber: 2.1621, swd: 1.4037, ept: 74.1791
    Epoch [9/50], Test Losses: mse: 11.7746, mae: 2.5884, huber: 2.1400, swd: 2.1471, ept: 73.7520
      Epoch 9 composite train-obj: 1.353074
            No improvement (2.1621), counter 3/5
    Epoch [10/50], Train Losses: mse: 5.3332, mae: 1.7028, huber: 1.2771, swd: 0.8626, ept: 44.2908
    Epoch [10/50], Val Losses: mse: 11.8640, mae: 2.6163, huber: 2.1652, swd: 1.4457, ept: 75.2766
    Epoch [10/50], Test Losses: mse: 11.7200, mae: 2.5775, huber: 2.1299, swd: 2.1391, ept: 76.2627
      Epoch 10 composite train-obj: 1.277064
            No improvement (2.1652), counter 4/5
    Epoch [11/50], Train Losses: mse: 4.9151, mae: 1.6279, huber: 1.2052, swd: 0.7910, ept: 45.3056
    Epoch [11/50], Val Losses: mse: 12.4473, mae: 2.6849, huber: 2.2329, swd: 1.4006, ept: 73.3966
    Epoch [11/50], Test Losses: mse: 11.9461, mae: 2.5903, huber: 2.1426, swd: 1.8833, ept: 77.1691
      Epoch 11 composite train-obj: 1.205223
    Epoch [11/50], Test Losses: mse: 11.7588, mae: 2.6076, huber: 2.1570, swd: 2.3488, ept: 69.5099
    Best round's Test MSE: 11.7588, MAE: 2.6076, SWD: 2.3488
    Best round's Validation MSE: 11.0506, MAE: 2.5599, SWD: 1.6833
    Best round's Test verification MSE : 11.7588, MAE: 2.6076, SWD: 2.3488
    Time taken: 92.72 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.0009, mae: 3.1361, huber: 2.6725, swd: 3.6460, ept: 19.4411
    Epoch [1/50], Val Losses: mse: 12.9530, mae: 2.8096, huber: 2.3511, swd: 2.6876, ept: 47.4073
    Epoch [1/50], Test Losses: mse: 12.8102, mae: 2.7983, huber: 2.3395, swd: 3.1807, ept: 49.0199
      Epoch 1 composite train-obj: 2.672509
            Val objective improved inf → 2.3511, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.4085, mae: 2.6368, huber: 2.1805, swd: 2.1900, ept: 29.2545
    Epoch [2/50], Val Losses: mse: 13.1010, mae: 2.8126, huber: 2.3562, swd: 2.6084, ept: 51.4692
    Epoch [2/50], Test Losses: mse: 12.3620, mae: 2.7098, huber: 2.2557, swd: 2.7908, ept: 55.7931
      Epoch 2 composite train-obj: 2.180489
            No improvement (2.3562), counter 1/5
    Epoch [3/50], Train Losses: mse: 10.4653, mae: 2.5026, huber: 2.0497, swd: 1.9142, ept: 31.2833
    Epoch [3/50], Val Losses: mse: 11.0546, mae: 2.5945, huber: 2.1402, swd: 2.2897, ept: 52.7114
    Epoch [3/50], Test Losses: mse: 12.3369, mae: 2.7028, huber: 2.2496, swd: 2.7730, ept: 57.6046
      Epoch 3 composite train-obj: 2.049712
            Val objective improved 2.3511 → 2.1402, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 9.4970, mae: 2.3636, huber: 1.9148, swd: 1.6621, ept: 33.6284
    Epoch [4/50], Val Losses: mse: 11.6057, mae: 2.6075, huber: 2.1554, swd: 1.9010, ept: 62.8677
    Epoch [4/50], Test Losses: mse: 12.0781, mae: 2.6485, huber: 2.1972, swd: 2.3979, ept: 62.6207
      Epoch 4 composite train-obj: 1.914845
            No improvement (2.1554), counter 1/5
    Epoch [5/50], Train Losses: mse: 8.5648, mae: 2.2245, huber: 1.7802, swd: 1.4796, ept: 36.1548
    Epoch [5/50], Val Losses: mse: 11.3537, mae: 2.6127, huber: 2.1611, swd: 1.3734, ept: 61.8271
    Epoch [5/50], Test Losses: mse: 11.5810, mae: 2.6030, huber: 2.1522, swd: 1.8926, ept: 64.2576
      Epoch 5 composite train-obj: 1.780169
            No improvement (2.1611), counter 2/5
    Epoch [6/50], Train Losses: mse: 7.7844, mae: 2.1022, huber: 1.6618, swd: 1.3196, ept: 38.1715
    Epoch [6/50], Val Losses: mse: 11.0375, mae: 2.5612, huber: 2.1100, swd: 1.5145, ept: 67.1369
    Epoch [6/50], Test Losses: mse: 11.4392, mae: 2.5629, huber: 2.1132, swd: 2.0957, ept: 70.1259
      Epoch 6 composite train-obj: 1.661781
            Val objective improved 2.1402 → 2.1100, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.0908, mae: 1.9936, huber: 1.5567, swd: 1.1927, ept: 39.8625
    Epoch [7/50], Val Losses: mse: 11.3249, mae: 2.5568, huber: 2.1089, swd: 1.5048, ept: 69.5998
    Epoch [7/50], Test Losses: mse: 11.7163, mae: 2.5786, huber: 2.1308, swd: 2.1389, ept: 72.4138
      Epoch 7 composite train-obj: 1.556746
            Val objective improved 2.1100 → 2.1089, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 6.4733, mae: 1.8928, huber: 1.4597, swd: 1.0842, ept: 41.7815
    Epoch [8/50], Val Losses: mse: 12.0742, mae: 2.6508, huber: 2.1999, swd: 1.4844, ept: 68.9261
    Epoch [8/50], Test Losses: mse: 11.8513, mae: 2.5742, huber: 2.1274, swd: 1.8656, ept: 75.8291
      Epoch 8 composite train-obj: 1.459741
            No improvement (2.1999), counter 1/5
    Epoch [9/50], Train Losses: mse: 5.9451, mae: 1.8060, huber: 1.3760, swd: 0.9831, ept: 43.2213
    Epoch [9/50], Val Losses: mse: 12.3501, mae: 2.7094, huber: 2.2571, swd: 1.4226, ept: 69.2674
    Epoch [9/50], Test Losses: mse: 11.8447, mae: 2.5782, huber: 2.1305, swd: 1.8330, ept: 77.1270
      Epoch 9 composite train-obj: 1.375995
            No improvement (2.2571), counter 2/5
    Epoch [10/50], Train Losses: mse: 5.5185, mae: 1.7331, huber: 1.3061, swd: 0.9085, ept: 44.1823
    Epoch [10/50], Val Losses: mse: 12.1328, mae: 2.6611, huber: 2.2099, swd: 1.3853, ept: 69.6928
    Epoch [10/50], Test Losses: mse: 11.7999, mae: 2.5608, huber: 2.1145, swd: 1.8526, ept: 80.0409
      Epoch 10 composite train-obj: 1.306100
            No improvement (2.2099), counter 3/5
    Epoch [11/50], Train Losses: mse: 5.1305, mae: 1.6632, huber: 1.2395, swd: 0.8313, ept: 45.0427
    Epoch [11/50], Val Losses: mse: 12.8919, mae: 2.7396, huber: 2.2884, swd: 1.4766, ept: 71.6643
    Epoch [11/50], Test Losses: mse: 11.9988, mae: 2.5713, huber: 2.1256, swd: 1.8613, ept: 79.9955
      Epoch 11 composite train-obj: 1.239466
            No improvement (2.2884), counter 4/5
    Epoch [12/50], Train Losses: mse: 4.7975, mae: 1.6040, huber: 1.1826, swd: 0.7654, ept: 45.7969
    Epoch [12/50], Val Losses: mse: 13.1790, mae: 2.7741, huber: 2.3217, swd: 1.4486, ept: 72.5354
    Epoch [12/50], Test Losses: mse: 11.9593, mae: 2.5726, huber: 2.1259, swd: 1.7392, ept: 80.0191
      Epoch 12 composite train-obj: 1.182598
    Epoch [12/50], Test Losses: mse: 11.7163, mae: 2.5786, huber: 2.1308, swd: 2.1389, ept: 72.4138
    Best round's Test MSE: 11.7163, MAE: 2.5786, SWD: 2.1389
    Best round's Validation MSE: 11.3249, MAE: 2.5568, SWD: 1.5048
    Best round's Test verification MSE : 11.7163, MAE: 2.5786, SWD: 2.1389
    Time taken: 101.63 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.1163, mae: 3.1542, huber: 2.6902, swd: 3.3140, ept: 18.6108
    Epoch [1/50], Val Losses: mse: 12.6806, mae: 2.7740, huber: 2.3152, swd: 1.8519, ept: 48.6166
    Epoch [1/50], Test Losses: mse: 13.1958, mae: 2.8408, huber: 2.3816, swd: 2.7109, ept: 48.7160
      Epoch 1 composite train-obj: 2.690157
            Val objective improved inf → 2.3152, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.5985, mae: 2.6662, huber: 2.2091, swd: 2.0385, ept: 29.2283
    Epoch [2/50], Val Losses: mse: 12.3786, mae: 2.7076, huber: 2.2525, swd: 1.9250, ept: 51.6436
    Epoch [2/50], Test Losses: mse: 12.3892, mae: 2.7319, huber: 2.2762, swd: 2.3329, ept: 54.1040
      Epoch 2 composite train-obj: 2.209102
            Val objective improved 2.3152 → 2.2525, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 10.7840, mae: 2.5495, huber: 2.0958, swd: 1.8341, ept: 31.1389
    Epoch [3/50], Val Losses: mse: 12.6900, mae: 2.7496, huber: 2.2934, swd: 1.9153, ept: 52.4173
    Epoch [3/50], Test Losses: mse: 12.4082, mae: 2.7134, huber: 2.2581, swd: 1.9456, ept: 56.5636
      Epoch 3 composite train-obj: 2.095775
            No improvement (2.2934), counter 1/5
    Epoch [4/50], Train Losses: mse: 9.8867, mae: 2.4216, huber: 1.9713, swd: 1.6194, ept: 32.8577
    Epoch [4/50], Val Losses: mse: 11.9590, mae: 2.7109, huber: 2.2537, swd: 1.9527, ept: 51.1277
    Epoch [4/50], Test Losses: mse: 12.0359, mae: 2.6894, huber: 2.2343, swd: 2.4073, ept: 56.5944
      Epoch 4 composite train-obj: 1.971252
            No improvement (2.2537), counter 2/5
    Epoch [5/50], Train Losses: mse: 8.8975, mae: 2.2766, huber: 1.8305, swd: 1.4491, ept: 34.9889
    Epoch [5/50], Val Losses: mse: 11.9830, mae: 2.6701, huber: 2.2162, swd: 1.7922, ept: 58.3000
    Epoch [5/50], Test Losses: mse: 12.0712, mae: 2.6568, huber: 2.2047, swd: 2.3180, ept: 64.1637
      Epoch 5 composite train-obj: 1.830457
            Val objective improved 2.2525 → 2.2162, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.9372, mae: 2.1309, huber: 1.6893, swd: 1.2700, ept: 37.3238
    Epoch [6/50], Val Losses: mse: 11.6695, mae: 2.5977, huber: 2.1469, swd: 1.5729, ept: 63.9547
    Epoch [6/50], Test Losses: mse: 12.0223, mae: 2.6236, huber: 2.1741, swd: 2.0076, ept: 67.6419
      Epoch 6 composite train-obj: 1.689309
            Val objective improved 2.2162 → 2.1469, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.1571, mae: 2.0084, huber: 1.5707, swd: 1.1355, ept: 39.4425
    Epoch [7/50], Val Losses: mse: 11.8250, mae: 2.5887, huber: 2.1392, swd: 1.6599, ept: 68.9126
    Epoch [7/50], Test Losses: mse: 12.3735, mae: 2.6352, huber: 2.1866, swd: 1.9891, ept: 73.4862
      Epoch 7 composite train-obj: 1.570717
            Val objective improved 2.1469 → 2.1392, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 6.4461, mae: 1.8923, huber: 1.4589, swd: 1.0252, ept: 41.5195
    Epoch [8/50], Val Losses: mse: 11.8640, mae: 2.6118, huber: 2.1618, swd: 1.3890, ept: 69.3248
    Epoch [8/50], Test Losses: mse: 12.0345, mae: 2.5893, huber: 2.1418, swd: 1.7672, ept: 74.0131
      Epoch 8 composite train-obj: 1.458894
            No improvement (2.1618), counter 1/5
    Epoch [9/50], Train Losses: mse: 5.8616, mae: 1.7932, huber: 1.3637, swd: 0.9220, ept: 43.1892
    Epoch [9/50], Val Losses: mse: 12.0759, mae: 2.6511, huber: 2.1999, swd: 1.3952, ept: 70.3560
    Epoch [9/50], Test Losses: mse: 11.7630, mae: 2.5650, huber: 2.1168, swd: 1.7312, ept: 76.2261
      Epoch 9 composite train-obj: 1.363656
            No improvement (2.1999), counter 2/5
    Epoch [10/50], Train Losses: mse: 5.3797, mae: 1.7112, huber: 1.2849, swd: 0.8329, ept: 44.0557
    Epoch [10/50], Val Losses: mse: 11.9274, mae: 2.6008, huber: 2.1538, swd: 1.3988, ept: 76.5379
    Epoch [10/50], Test Losses: mse: 12.3228, mae: 2.6062, huber: 2.1599, swd: 1.7395, ept: 77.2008
      Epoch 10 composite train-obj: 1.284860
            No improvement (2.1538), counter 3/5
    Epoch [11/50], Train Losses: mse: 4.9560, mae: 1.6353, huber: 1.2124, swd: 0.7559, ept: 45.2567
    Epoch [11/50], Val Losses: mse: 12.7739, mae: 2.7172, huber: 2.2664, swd: 1.5514, ept: 69.6140
    Epoch [11/50], Test Losses: mse: 12.1695, mae: 2.5948, huber: 2.1473, swd: 1.8849, ept: 78.9639
      Epoch 11 composite train-obj: 1.212448
            No improvement (2.2664), counter 4/5
    Epoch [12/50], Train Losses: mse: 4.6165, mae: 1.5731, huber: 1.1528, swd: 0.6920, ept: 46.1968
    Epoch [12/50], Val Losses: mse: 12.6720, mae: 2.7002, huber: 2.2494, swd: 1.2685, ept: 74.3907
    Epoch [12/50], Test Losses: mse: 12.4579, mae: 2.6126, huber: 2.1661, swd: 1.6876, ept: 78.2264
      Epoch 12 composite train-obj: 1.152817
    Epoch [12/50], Test Losses: mse: 12.3735, mae: 2.6352, huber: 2.1866, swd: 1.9891, ept: 73.4862
    Best round's Test MSE: 12.3735, MAE: 2.6352, SWD: 1.9891
    Best round's Validation MSE: 11.8250, MAE: 2.5887, SWD: 1.6599
    Best round's Test verification MSE : 12.3735, MAE: 2.6352, SWD: 1.9891
    Time taken: 101.07 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz96_seq720_pred196_20250512_2309)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.9495 ± 0.3003
      mae: 2.6071 ± 0.0231
      huber: 2.1581 ± 0.0228
      swd: 2.1589 ± 0.1475
      ept: 71.8033 ± 1.6797
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 11.4002 ± 0.3206
      mae: 2.5684 ± 0.0144
      huber: 2.1186 ± 0.0145
      swd: 1.6160 ± 0.0792
      ept: 68.0853 ± 1.6795
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 295.52 seconds
    
    Experiment complete: PatchTST_lorenz96_seq720_pred196_20250512_2309
    Model: PatchTST
    Dataset: lorenz96
    Sequence Length: 720
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=720,
    pred_len=336,
    channels=data_mgr.datasets['lorenz96']['channels'],
    enc_in=data_mgr.datasets['lorenz96']['channels'],
    dec_in=data_mgr.datasets['lorenz96']['channels'],
    c_out=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([336, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([336, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 96
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 336, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 96
    Validation Batches: 7
    Test Batches: 22
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 18.1939, mae: 3.2774, huber: 2.8111, swd: 3.5577, ept: 18.2081
    Epoch [1/50], Val Losses: mse: 13.3980, mae: 2.9328, huber: 2.4674, swd: 2.1618, ept: 40.6063
    Epoch [1/50], Test Losses: mse: 14.7965, mae: 3.0484, huber: 2.5851, swd: 2.8201, ept: 47.4093
      Epoch 1 composite train-obj: 2.811073
            Val objective improved inf → 2.4674, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.7409, mae: 2.8288, huber: 2.3675, swd: 2.1653, ept: 28.0666
    Epoch [2/50], Val Losses: mse: 13.8660, mae: 2.9547, huber: 2.4911, swd: 2.0640, ept: 44.3027
    Epoch [2/50], Test Losses: mse: 14.4280, mae: 3.0084, huber: 2.5459, swd: 2.8916, ept: 50.8367
      Epoch 2 composite train-obj: 2.367477
            No improvement (2.4911), counter 1/5
    Epoch [3/50], Train Losses: mse: 12.1197, mae: 2.7410, huber: 2.2819, swd: 1.9876, ept: 30.0578
    Epoch [3/50], Val Losses: mse: 13.6910, mae: 2.8916, huber: 2.4310, swd: 1.5989, ept: 48.9838
    Epoch [3/50], Test Losses: mse: 14.8096, mae: 3.0214, huber: 2.5597, swd: 2.1846, ept: 49.5476
      Epoch 3 composite train-obj: 2.281870
            Val objective improved 2.4674 → 2.4310, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.0432, mae: 2.5901, huber: 2.1346, swd: 1.7213, ept: 32.0215
    Epoch [4/50], Val Losses: mse: 14.7234, mae: 3.0392, huber: 2.5757, swd: 1.8121, ept: 46.6393
    Epoch [4/50], Test Losses: mse: 14.6257, mae: 2.9841, huber: 2.5241, swd: 2.1788, ept: 55.3599
      Epoch 4 composite train-obj: 2.134555
            No improvement (2.5757), counter 1/5
    Epoch [5/50], Train Losses: mse: 9.8477, mae: 2.4215, huber: 1.9702, swd: 1.4732, ept: 34.4596
    Epoch [5/50], Val Losses: mse: 13.9393, mae: 2.9057, huber: 2.4460, swd: 1.5396, ept: 54.5131
    Epoch [5/50], Test Losses: mse: 14.8972, mae: 3.0047, huber: 2.5444, swd: 1.9955, ept: 59.9254
      Epoch 5 composite train-obj: 1.970186
            No improvement (2.4460), counter 2/5
    Epoch [6/50], Train Losses: mse: 8.7704, mae: 2.2647, huber: 1.8175, swd: 1.2707, ept: 37.0771
    Epoch [6/50], Val Losses: mse: 14.7581, mae: 2.9907, huber: 2.5294, swd: 1.4840, ept: 57.6894
    Epoch [6/50], Test Losses: mse: 15.1751, mae: 3.0150, huber: 2.5552, swd: 1.9302, ept: 63.4176
      Epoch 6 composite train-obj: 1.817526
            No improvement (2.5294), counter 3/5
    Epoch [7/50], Train Losses: mse: 7.8672, mae: 2.1295, huber: 1.6862, swd: 1.1122, ept: 39.3670
    Epoch [7/50], Val Losses: mse: 14.7994, mae: 2.9744, huber: 2.5141, swd: 1.4743, ept: 59.6810
    Epoch [7/50], Test Losses: mse: 15.1208, mae: 3.0014, huber: 2.5423, swd: 1.8841, ept: 65.8370
      Epoch 7 composite train-obj: 1.686169
            No improvement (2.5141), counter 4/5
    Epoch [8/50], Train Losses: mse: 7.0212, mae: 1.9973, huber: 1.5582, swd: 0.9719, ept: 41.5322
    Epoch [8/50], Val Losses: mse: 16.2458, mae: 3.1367, huber: 2.6746, swd: 1.4904, ept: 59.6773
    Epoch [8/50], Test Losses: mse: 15.0609, mae: 2.9855, huber: 2.5272, swd: 1.8339, ept: 68.2135
      Epoch 8 composite train-obj: 1.558184
    Epoch [8/50], Test Losses: mse: 14.8096, mae: 3.0214, huber: 2.5597, swd: 2.1846, ept: 49.5476
    Best round's Test MSE: 14.8096, MAE: 3.0214, SWD: 2.1846
    Best round's Validation MSE: 13.6910, MAE: 2.8916, SWD: 1.5989
    Best round's Test verification MSE : 14.8096, MAE: 3.0214, SWD: 2.1846
    Time taken: 70.21 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.9445, mae: 3.2543, huber: 2.7883, swd: 3.4087, ept: 18.9490
    Epoch [1/50], Val Losses: mse: 13.2731, mae: 2.9169, huber: 2.4524, swd: 2.2673, ept: 41.2953
    Epoch [1/50], Test Losses: mse: 14.5476, mae: 3.0191, huber: 2.5561, swd: 3.1544, ept: 48.5499
      Epoch 1 composite train-obj: 2.788300
            Val objective improved inf → 2.4524, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.7991, mae: 2.8332, huber: 2.3721, swd: 2.1566, ept: 28.5859
    Epoch [2/50], Val Losses: mse: 13.7212, mae: 2.9219, huber: 2.4592, swd: 1.9226, ept: 45.8927
    Epoch [2/50], Test Losses: mse: 14.7680, mae: 3.0286, huber: 2.5667, swd: 2.5365, ept: 51.3153
      Epoch 2 composite train-obj: 2.372145
            No improvement (2.4592), counter 1/5
    Epoch [3/50], Train Losses: mse: 12.0940, mae: 2.7373, huber: 2.2786, swd: 1.9807, ept: 30.4775
    Epoch [3/50], Val Losses: mse: 13.1946, mae: 2.8693, huber: 2.4083, swd: 2.0296, ept: 47.8354
    Epoch [3/50], Test Losses: mse: 14.8692, mae: 3.0309, huber: 2.5695, swd: 2.6024, ept: 53.5169
      Epoch 3 composite train-obj: 2.278564
            Val objective improved 2.4524 → 2.4083, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.0663, mae: 2.5957, huber: 2.1402, swd: 1.7083, ept: 32.0847
    Epoch [4/50], Val Losses: mse: 13.9707, mae: 2.9356, huber: 2.4750, swd: 2.0638, ept: 49.4430
    Epoch [4/50], Test Losses: mse: 14.6192, mae: 2.9920, huber: 2.5319, swd: 2.5948, ept: 56.9762
      Epoch 4 composite train-obj: 2.140203
            No improvement (2.4750), counter 1/5
    Epoch [5/50], Train Losses: mse: 9.9815, mae: 2.4432, huber: 1.9914, swd: 1.5193, ept: 34.6407
    Epoch [5/50], Val Losses: mse: 13.7715, mae: 2.8859, huber: 2.4269, swd: 1.3777, ept: 55.7255
    Epoch [5/50], Test Losses: mse: 14.6081, mae: 2.9636, huber: 2.5049, swd: 1.8271, ept: 63.2375
      Epoch 5 composite train-obj: 1.991432
            No improvement (2.4269), counter 2/5
    Epoch [6/50], Train Losses: mse: 8.9638, mae: 2.2958, huber: 1.8480, swd: 1.3268, ept: 36.6951
    Epoch [6/50], Val Losses: mse: 14.5196, mae: 2.9763, huber: 2.5156, swd: 1.3307, ept: 56.6608
    Epoch [6/50], Test Losses: mse: 14.5627, mae: 2.9506, huber: 2.4922, swd: 1.7438, ept: 64.9497
      Epoch 6 composite train-obj: 1.847978
            No improvement (2.5156), counter 3/5
    Epoch [7/50], Train Losses: mse: 8.0633, mae: 2.1584, huber: 1.7145, swd: 1.1601, ept: 38.8181
    Epoch [7/50], Val Losses: mse: 14.8946, mae: 3.0124, huber: 2.5519, swd: 1.5508, ept: 57.1938
    Epoch [7/50], Test Losses: mse: 14.7613, mae: 2.9592, huber: 2.5014, swd: 2.0353, ept: 67.4607
      Epoch 7 composite train-obj: 1.714518
            No improvement (2.5519), counter 4/5
    Epoch [8/50], Train Losses: mse: 7.3141, mae: 2.0439, huber: 1.6034, swd: 1.0296, ept: 40.7230
    Epoch [8/50], Val Losses: mse: 15.5058, mae: 3.0803, huber: 2.6192, swd: 1.4655, ept: 59.5144
    Epoch [8/50], Test Losses: mse: 15.0902, mae: 2.9817, huber: 2.5243, swd: 1.9020, ept: 71.9491
      Epoch 8 composite train-obj: 1.603419
    Epoch [8/50], Test Losses: mse: 14.8692, mae: 3.0309, huber: 2.5695, swd: 2.6024, ept: 53.5169
    Best round's Test MSE: 14.8692, MAE: 3.0309, SWD: 2.6024
    Best round's Validation MSE: 13.1946, MAE: 2.8693, SWD: 2.0296
    Best round's Test verification MSE : 14.8692, MAE: 3.0309, SWD: 2.6024
    Time taken: 67.05 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 17.7857, mae: 3.2439, huber: 2.7779, swd: 3.2789, ept: 18.8664
    Epoch [1/50], Val Losses: mse: 14.0367, mae: 2.9894, huber: 2.5235, swd: 2.1286, ept: 40.7796
    Epoch [1/50], Test Losses: mse: 14.6350, mae: 3.0239, huber: 2.5609, swd: 2.9174, ept: 47.0043
      Epoch 1 composite train-obj: 2.777859
            Val objective improved inf → 2.5235, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.8271, mae: 2.8361, huber: 2.3749, swd: 2.1386, ept: 28.6119
    Epoch [2/50], Val Losses: mse: 14.0589, mae: 2.9797, huber: 2.5164, swd: 1.9785, ept: 47.7454
    Epoch [2/50], Test Losses: mse: 14.0702, mae: 2.9677, huber: 2.5055, swd: 2.7065, ept: 50.3042
      Epoch 2 composite train-obj: 2.374945
            Val objective improved 2.5235 → 2.5164, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.1943, mae: 2.7528, huber: 2.2937, swd: 2.0211, ept: 30.4888
    Epoch [3/50], Val Losses: mse: 14.3064, mae: 2.9494, huber: 2.4889, swd: 1.3589, ept: 49.5460
    Epoch [3/50], Test Losses: mse: 14.3371, mae: 2.9680, huber: 2.5081, swd: 1.7262, ept: 52.8870
      Epoch 3 composite train-obj: 2.293720
            Val objective improved 2.5164 → 2.4889, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.4384, mae: 2.6479, huber: 2.1913, swd: 1.8216, ept: 32.1380
    Epoch [4/50], Val Losses: mse: 13.3469, mae: 2.8418, huber: 2.3828, swd: 1.3282, ept: 51.2149
    Epoch [4/50], Test Losses: mse: 14.6666, mae: 2.9814, huber: 2.5220, swd: 1.6902, ept: 56.5356
      Epoch 4 composite train-obj: 2.191329
            Val objective improved 2.4889 → 2.3828, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 10.3761, mae: 2.5016, huber: 2.0485, swd: 1.6175, ept: 34.0146
    Epoch [5/50], Val Losses: mse: 13.5692, mae: 2.9059, huber: 2.4440, swd: 1.4817, ept: 55.4005
    Epoch [5/50], Test Losses: mse: 14.2194, mae: 2.9352, huber: 2.4763, swd: 1.7337, ept: 60.8617
      Epoch 5 composite train-obj: 2.048479
            No improvement (2.4440), counter 1/5
    Epoch [6/50], Train Losses: mse: 9.3049, mae: 2.3464, huber: 1.8972, swd: 1.4031, ept: 36.5984
    Epoch [6/50], Val Losses: mse: 12.8340, mae: 2.7943, huber: 2.3358, swd: 1.6252, ept: 61.8659
    Epoch [6/50], Test Losses: mse: 14.0770, mae: 2.8958, huber: 2.4389, swd: 1.8603, ept: 66.0007
      Epoch 6 composite train-obj: 1.897230
            Val objective improved 2.3828 → 2.3358, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 8.2850, mae: 2.1950, huber: 1.7501, swd: 1.2219, ept: 38.9545
    Epoch [7/50], Val Losses: mse: 13.3564, mae: 2.8543, huber: 2.3946, swd: 1.3056, ept: 59.5175
    Epoch [7/50], Test Losses: mse: 14.2659, mae: 2.9092, huber: 2.4517, swd: 1.7178, ept: 68.9768
      Epoch 7 composite train-obj: 1.750065
            No improvement (2.3946), counter 1/5
    Epoch [8/50], Train Losses: mse: 7.4365, mae: 2.0650, huber: 1.6239, swd: 1.0719, ept: 40.8770
    Epoch [8/50], Val Losses: mse: 13.9821, mae: 2.9215, huber: 2.4606, swd: 1.5430, ept: 61.9777
    Epoch [8/50], Test Losses: mse: 14.4728, mae: 2.9166, huber: 2.4590, swd: 1.8179, ept: 72.1684
      Epoch 8 composite train-obj: 1.623885
            No improvement (2.4606), counter 2/5
    Epoch [9/50], Train Losses: mse: 6.7377, mae: 1.9535, huber: 1.5161, swd: 0.9524, ept: 42.5974
    Epoch [9/50], Val Losses: mse: 14.2165, mae: 2.9276, huber: 2.4674, swd: 1.3932, ept: 63.2789
    Epoch [9/50], Test Losses: mse: 14.4100, mae: 2.8991, huber: 2.4424, swd: 1.7768, ept: 76.6195
      Epoch 9 composite train-obj: 1.516125
            No improvement (2.4674), counter 3/5
    Epoch [10/50], Train Losses: mse: 6.1461, mae: 1.8568, huber: 1.4226, swd: 0.8530, ept: 43.7784
    Epoch [10/50], Val Losses: mse: 15.0222, mae: 3.0126, huber: 2.5514, swd: 1.3040, ept: 65.3864
    Epoch [10/50], Test Losses: mse: 15.0236, mae: 2.9470, huber: 2.4903, swd: 1.4695, ept: 77.8529
      Epoch 10 composite train-obj: 1.422619
            No improvement (2.5514), counter 4/5
    Epoch [11/50], Train Losses: mse: 5.6525, mae: 1.7719, huber: 1.3410, swd: 0.7676, ept: 45.1799
    Epoch [11/50], Val Losses: mse: 14.5974, mae: 2.9735, huber: 2.5134, swd: 1.3445, ept: 65.2586
    Epoch [11/50], Test Losses: mse: 14.9968, mae: 2.9543, huber: 2.4974, swd: 1.7963, ept: 77.9491
      Epoch 11 composite train-obj: 1.341000
    Epoch [11/50], Test Losses: mse: 14.0770, mae: 2.8958, huber: 2.4389, swd: 1.8603, ept: 66.0007
    Best round's Test MSE: 14.0770, MAE: 2.8958, SWD: 1.8603
    Best round's Validation MSE: 12.8340, MAE: 2.7943, SWD: 1.6252
    Best round's Test verification MSE : 14.0770, MAE: 2.8958, SWD: 1.8603
    Time taken: 92.96 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz96_seq720_pred336_20250512_2317)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 14.5853 ± 0.3602
      mae: 2.9827 ± 0.0616
      huber: 2.5227 ± 0.0594
      swd: 2.2158 ± 0.3037
      ept: 56.3551 ± 7.0103
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 13.2399 ± 0.3513
      mae: 2.8517 ± 0.0416
      huber: 2.3917 ± 0.0406
      swd: 1.7512 ± 0.1971
      ept: 52.8950 ± 6.3607
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 230.38 seconds
    
    Experiment complete: PatchTST_lorenz96_seq720_pred336_20250512_2317
    Model: PatchTST
    Dataset: lorenz96
    Sequence Length: 720
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=720,
    pred_len=720,
    channels=data_mgr.datasets['lorenz96']['channels'],
    enc_in=data_mgr.datasets['lorenz96']['channels'],
    dec_in=data_mgr.datasets['lorenz96']['channels'],
    c_out=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([720, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([720, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 720, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 4
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 19.0232, mae: 3.3671, huber: 2.8993, swd: 3.3369, ept: 17.3285
    Epoch [1/50], Val Losses: mse: 14.2250, mae: 3.0389, huber: 2.5728, swd: 1.7645, ept: 40.1254
    Epoch [1/50], Test Losses: mse: 15.5524, mae: 3.1740, huber: 2.7070, swd: 3.0901, ept: 45.1910
      Epoch 1 composite train-obj: 2.899304
            Val objective improved inf → 2.5728, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 13.6120, mae: 2.9504, huber: 2.4863, swd: 2.0737, ept: 27.1047
    Epoch [2/50], Val Losses: mse: 14.7252, mae: 3.0708, huber: 2.6046, swd: 1.3740, ept: 42.3003
    Epoch [2/50], Test Losses: mse: 15.8007, mae: 3.1808, huber: 2.7143, swd: 2.4669, ept: 48.0606
      Epoch 2 composite train-obj: 2.486314
            No improvement (2.6046), counter 1/5
    Epoch [3/50], Train Losses: mse: 13.3312, mae: 2.9079, huber: 2.4449, swd: 1.9751, ept: 29.0347
    Epoch [3/50], Val Losses: mse: 14.9755, mae: 3.1244, huber: 2.6577, swd: 1.5387, ept: 43.6778
    Epoch [3/50], Test Losses: mse: 15.2171, mae: 3.1136, huber: 2.6483, swd: 2.5463, ept: 53.3817
      Epoch 3 composite train-obj: 2.444904
            No improvement (2.6577), counter 2/5
    Epoch [4/50], Train Losses: mse: 12.6668, mae: 2.8177, huber: 2.3566, swd: 1.8207, ept: 30.6046
    Epoch [4/50], Val Losses: mse: 16.0788, mae: 3.2101, huber: 2.7432, swd: 1.3009, ept: 49.1793
    Epoch [4/50], Test Losses: mse: 15.5032, mae: 3.1390, huber: 2.6736, swd: 2.2445, ept: 53.5411
      Epoch 4 composite train-obj: 2.356621
            No improvement (2.7432), counter 3/5
    Epoch [5/50], Train Losses: mse: 11.5392, mae: 2.6654, huber: 2.2077, swd: 1.5898, ept: 32.8503
    Epoch [5/50], Val Losses: mse: 17.1168, mae: 3.2795, huber: 2.8121, swd: 1.4555, ept: 49.6666
    Epoch [5/50], Test Losses: mse: 16.4769, mae: 3.2218, huber: 2.7564, swd: 2.4355, ept: 58.2981
      Epoch 5 composite train-obj: 2.207654
            No improvement (2.8121), counter 4/5
    Epoch [6/50], Train Losses: mse: 10.4344, mae: 2.5119, huber: 2.0577, swd: 1.3751, ept: 35.2320
    Epoch [6/50], Val Losses: mse: 17.1562, mae: 3.2751, huber: 2.8081, swd: 1.0634, ept: 49.0979
    Epoch [6/50], Test Losses: mse: 16.9422, mae: 3.2590, huber: 2.7935, swd: 1.9461, ept: 61.6035
      Epoch 6 composite train-obj: 2.057695
    Epoch [6/50], Test Losses: mse: 15.5524, mae: 3.1740, huber: 2.7070, swd: 3.0901, ept: 45.1910
    Best round's Test MSE: 15.5524, MAE: 3.1740, SWD: 3.0901
    Best round's Validation MSE: 14.2250, MAE: 3.0389, SWD: 1.7645
    Best round's Test verification MSE : 15.5524, MAE: 3.1740, SWD: 3.0901
    Time taken: 51.06 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 18.6848, mae: 3.3420, huber: 2.8744, swd: 3.6461, ept: 18.1790
    Epoch [1/50], Val Losses: mse: 15.1161, mae: 3.1298, huber: 2.6629, swd: 1.7231, ept: 37.4267
    Epoch [1/50], Test Losses: mse: 15.2732, mae: 3.1331, huber: 2.6670, swd: 2.9095, ept: 47.9927
      Epoch 1 composite train-obj: 2.874426
            Val objective improved inf → 2.6629, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 13.6638, mae: 2.9577, huber: 2.4935, swd: 2.0906, ept: 27.7724
    Epoch [2/50], Val Losses: mse: 16.2661, mae: 3.2474, huber: 2.7791, swd: 1.5509, ept: 47.3263
    Epoch [2/50], Test Losses: mse: 15.8923, mae: 3.1942, huber: 2.7277, swd: 2.7910, ept: 51.1893
      Epoch 2 composite train-obj: 2.493501
            No improvement (2.7791), counter 1/5
    Epoch [3/50], Train Losses: mse: 13.3954, mae: 2.9189, huber: 2.4557, swd: 1.9893, ept: 29.5994
    Epoch [3/50], Val Losses: mse: 15.0096, mae: 3.0993, huber: 2.6337, swd: 1.4303, ept: 44.7595
    Epoch [3/50], Test Losses: mse: 15.5320, mae: 3.1458, huber: 2.6802, swd: 2.4929, ept: 53.0494
      Epoch 3 composite train-obj: 2.455710
            Val objective improved 2.6629 → 2.6337, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.5357, mae: 2.8017, huber: 2.3409, swd: 1.7780, ept: 30.9511
    Epoch [4/50], Val Losses: mse: 17.5525, mae: 3.3312, huber: 2.8633, swd: 1.0934, ept: 46.9641
    Epoch [4/50], Test Losses: mse: 16.0976, mae: 3.1892, huber: 2.7240, swd: 2.0513, ept: 54.5798
      Epoch 4 composite train-obj: 2.340948
            No improvement (2.8633), counter 1/5
    Epoch [5/50], Train Losses: mse: 11.4386, mae: 2.6515, huber: 2.1940, swd: 1.5249, ept: 33.1066
    Epoch [5/50], Val Losses: mse: 16.7405, mae: 3.2532, huber: 2.7859, swd: 0.8984, ept: 51.7995
    Epoch [5/50], Test Losses: mse: 16.2506, mae: 3.1847, huber: 2.7202, swd: 1.9580, ept: 61.4463
      Epoch 5 composite train-obj: 2.194023
            No improvement (2.7859), counter 2/5
    Epoch [6/50], Train Losses: mse: 10.2328, mae: 2.4826, huber: 2.0293, swd: 1.3378, ept: 35.9597
    Epoch [6/50], Val Losses: mse: 18.4676, mae: 3.4112, huber: 2.9428, swd: 0.9036, ept: 55.6542
    Epoch [6/50], Test Losses: mse: 16.9365, mae: 3.2362, huber: 2.7718, swd: 1.8137, ept: 63.7879
      Epoch 6 composite train-obj: 2.029294
            No improvement (2.9428), counter 3/5
    Epoch [7/50], Train Losses: mse: 9.1792, mae: 2.3308, huber: 1.8813, swd: 1.1622, ept: 38.4669
    Epoch [7/50], Val Losses: mse: 18.7040, mae: 3.4201, huber: 2.9521, swd: 0.9127, ept: 53.3212
    Epoch [7/50], Test Losses: mse: 16.9593, mae: 3.2481, huber: 2.7832, swd: 1.9086, ept: 67.4363
      Epoch 7 composite train-obj: 1.881282
            No improvement (2.9521), counter 4/5
    Epoch [8/50], Train Losses: mse: 8.2218, mae: 2.1898, huber: 1.7441, swd: 1.0103, ept: 40.7249
    Epoch [8/50], Val Losses: mse: 20.3680, mae: 3.5617, huber: 3.0931, swd: 0.9019, ept: 53.5799
    Epoch [8/50], Test Losses: mse: 17.3391, mae: 3.2647, huber: 2.8005, swd: 1.5792, ept: 72.5063
      Epoch 8 composite train-obj: 1.744064
    Epoch [8/50], Test Losses: mse: 15.5320, mae: 3.1458, huber: 2.6802, swd: 2.4929, ept: 53.0494
    Best round's Test MSE: 15.5320, MAE: 3.1458, SWD: 2.4929
    Best round's Validation MSE: 15.0096, MAE: 3.0993, SWD: 1.4303
    Best round's Test verification MSE : 15.5320, MAE: 3.1458, SWD: 2.4929
    Time taken: 68.33 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 18.5184, mae: 3.3337, huber: 2.8660, swd: 3.2426, ept: 17.8236
    Epoch [1/50], Val Losses: mse: 14.7438, mae: 3.0849, huber: 2.6181, swd: 2.0180, ept: 37.9355
    Epoch [1/50], Test Losses: mse: 15.5999, mae: 3.1776, huber: 2.7105, swd: 3.1420, ept: 43.4692
      Epoch 1 composite train-obj: 2.865957
            Val objective improved inf → 2.6181, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 13.7148, mae: 2.9651, huber: 2.5007, swd: 2.1356, ept: 26.7255
    Epoch [2/50], Val Losses: mse: 14.3304, mae: 3.0427, huber: 2.5771, swd: 1.5861, ept: 49.5331
    Epoch [2/50], Test Losses: mse: 15.3626, mae: 3.1388, huber: 2.6731, swd: 2.8946, ept: 48.9264
      Epoch 2 composite train-obj: 2.500736
            Val objective improved 2.6181 → 2.5771, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 13.3965, mae: 2.9199, huber: 2.4567, swd: 2.0301, ept: 28.9895
    Epoch [3/50], Val Losses: mse: 14.8353, mae: 3.0946, huber: 2.6282, swd: 1.4882, ept: 49.2358
    Epoch [3/50], Test Losses: mse: 15.9571, mae: 3.1944, huber: 2.7281, swd: 2.5863, ept: 54.1074
      Epoch 3 composite train-obj: 2.456703
            No improvement (2.6282), counter 1/5
    Epoch [4/50], Train Losses: mse: 12.8789, mae: 2.8468, huber: 2.3852, swd: 1.8923, ept: 30.7608
    Epoch [4/50], Val Losses: mse: 16.0628, mae: 3.2069, huber: 2.7394, swd: 1.5176, ept: 50.0176
    Epoch [4/50], Test Losses: mse: 15.7241, mae: 3.1768, huber: 2.7108, swd: 2.6775, ept: 54.7851
      Epoch 4 composite train-obj: 2.385214
            No improvement (2.7394), counter 2/5
    Epoch [5/50], Train Losses: mse: 11.9486, mae: 2.7212, huber: 2.2623, swd: 1.7089, ept: 32.9188
    Epoch [5/50], Val Losses: mse: 15.1103, mae: 3.0810, huber: 2.6162, swd: 1.1934, ept: 53.0403
    Epoch [5/50], Test Losses: mse: 15.5095, mae: 3.1438, huber: 2.6785, swd: 2.2738, ept: 59.1265
      Epoch 5 composite train-obj: 2.262295
            No improvement (2.6162), counter 3/5
    Epoch [6/50], Train Losses: mse: 10.9782, mae: 2.5881, huber: 2.1321, swd: 1.5197, ept: 34.9353
    Epoch [6/50], Val Losses: mse: 16.6304, mae: 3.2470, huber: 2.7798, swd: 0.9268, ept: 52.2216
    Epoch [6/50], Test Losses: mse: 15.9011, mae: 3.1526, huber: 2.6884, swd: 1.7458, ept: 64.3256
      Epoch 6 composite train-obj: 2.132144
            No improvement (2.7798), counter 4/5
    Epoch [7/50], Train Losses: mse: 9.9633, mae: 2.4442, huber: 1.9917, swd: 1.3481, ept: 37.1530
    Epoch [7/50], Val Losses: mse: 17.6875, mae: 3.3332, huber: 2.8660, swd: 0.9004, ept: 58.2735
    Epoch [7/50], Test Losses: mse: 16.1070, mae: 3.1682, huber: 2.7046, swd: 1.7500, ept: 67.7848
      Epoch 7 composite train-obj: 1.991725
    Epoch [7/50], Test Losses: mse: 15.3626, mae: 3.1388, huber: 2.6731, swd: 2.8946, ept: 48.9264
    Best round's Test MSE: 15.3626, MAE: 3.1388, SWD: 2.8946
    Best round's Validation MSE: 14.3304, MAE: 3.0427, SWD: 1.5861
    Best round's Test verification MSE : 15.3626, MAE: 3.1388, SWD: 2.8946
    Time taken: 60.31 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz96_seq720_pred720_20250512_2321)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 15.4823 ± 0.0850
      mae: 3.1529 ± 0.0152
      huber: 2.6868 ± 0.0146
      swd: 2.8259 ± 0.2486
      ept: 49.0556 ± 3.2095
      count: 4.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 14.5217 ± 0.3477
      mae: 3.0603 ± 0.0276
      huber: 2.5945 ± 0.0278
      swd: 1.5936 ± 0.1365
      ept: 44.8060 ± 3.8408
      count: 4.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 179.99 seconds
    
    Experiment complete: PatchTST_lorenz96_seq720_pred720_20250512_2321
    Model: PatchTST
    Dataset: lorenz96
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
    channels=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([96, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([96, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 98
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 96, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 98
    Validation Batches: 9
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.9421, mae: 2.7133, huber: 2.2558, swd: 3.8927, ept: 37.1143
    Epoch [1/50], Val Losses: mse: 11.4773, mae: 2.7421, huber: 2.2802, swd: 3.7408, ept: 43.4461
    Epoch [1/50], Test Losses: mse: 12.5005, mae: 2.7648, huber: 2.3061, swd: 4.1281, ept: 46.0494
      Epoch 1 composite train-obj: 2.255804
            Val objective improved inf → 2.2802, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.4866, mae: 2.4822, huber: 2.0348, swd: 3.5168, ept: 51.5904
    Epoch [2/50], Val Losses: mse: 11.0114, mae: 2.6115, huber: 2.1598, swd: 3.2296, ept: 49.0466
    Epoch [2/50], Test Losses: mse: 12.1616, mae: 2.6827, huber: 2.2317, swd: 3.7521, ept: 49.8786
      Epoch 2 composite train-obj: 2.034840
            Val objective improved 2.2802 → 2.1598, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 10.1969, mae: 2.4178, huber: 1.9771, swd: 3.3726, ept: 53.8596
    Epoch [3/50], Val Losses: mse: 11.2568, mae: 2.6107, huber: 2.1635, swd: 3.0129, ept: 48.7657
    Epoch [3/50], Test Losses: mse: 11.9127, mae: 2.6215, huber: 2.1766, swd: 3.4202, ept: 51.6599
      Epoch 3 composite train-obj: 1.977124
            No improvement (2.1635), counter 1/5
    Epoch [4/50], Train Losses: mse: 10.0738, mae: 2.3872, huber: 1.9506, swd: 3.2756, ept: 54.8203
    Epoch [4/50], Val Losses: mse: 10.6008, mae: 2.5320, huber: 2.0884, swd: 3.2618, ept: 50.9885
    Epoch [4/50], Test Losses: mse: 11.6813, mae: 2.5861, huber: 2.1446, swd: 3.6653, ept: 52.3983
      Epoch 4 composite train-obj: 1.950606
            Val objective improved 2.1598 → 2.0884, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 9.9857, mae: 2.3665, huber: 1.9326, swd: 3.2106, ept: 55.3333
    Epoch [5/50], Val Losses: mse: 10.6421, mae: 2.5393, huber: 2.0969, swd: 3.2711, ept: 51.3953
    Epoch [5/50], Test Losses: mse: 11.5163, mae: 2.5675, huber: 2.1271, swd: 3.6557, ept: 52.6587
      Epoch 5 composite train-obj: 1.932647
            No improvement (2.0969), counter 1/5
    Epoch [6/50], Train Losses: mse: 9.9064, mae: 2.3493, huber: 1.9177, swd: 3.1892, ept: 55.9805
    Epoch [6/50], Val Losses: mse: 10.7693, mae: 2.5466, huber: 2.1063, swd: 3.2593, ept: 51.0297
    Epoch [6/50], Test Losses: mse: 11.5516, mae: 2.5515, huber: 2.1145, swd: 3.5050, ept: 53.4075
      Epoch 6 composite train-obj: 1.917683
            No improvement (2.1063), counter 2/5
    Epoch [7/50], Train Losses: mse: 9.8359, mae: 2.3340, huber: 1.9043, swd: 3.1481, ept: 56.2948
    Epoch [7/50], Val Losses: mse: 10.2788, mae: 2.4813, huber: 2.0450, swd: 3.1881, ept: 52.4350
    Epoch [7/50], Test Losses: mse: 11.4319, mae: 2.5445, huber: 2.1082, swd: 3.5980, ept: 53.9522
      Epoch 7 composite train-obj: 1.904259
            Val objective improved 2.0884 → 2.0450, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 9.8090, mae: 2.3263, huber: 1.8977, swd: 3.1091, ept: 56.4653
    Epoch [8/50], Val Losses: mse: 10.5208, mae: 2.5050, huber: 2.0667, swd: 3.1291, ept: 52.6641
    Epoch [8/50], Test Losses: mse: 11.5089, mae: 2.5462, huber: 2.1099, swd: 3.4630, ept: 53.4339
      Epoch 8 composite train-obj: 1.897655
            No improvement (2.0667), counter 1/5
    Epoch [9/50], Train Losses: mse: 9.7401, mae: 2.3156, huber: 1.8880, swd: 3.0938, ept: 56.8136
    Epoch [9/50], Val Losses: mse: 10.0222, mae: 2.4423, huber: 2.0063, swd: 2.9361, ept: 53.5618
    Epoch [9/50], Test Losses: mse: 11.3029, mae: 2.5173, huber: 2.0832, swd: 3.3779, ept: 54.5798
      Epoch 9 composite train-obj: 1.888040
            Val objective improved 2.0450 → 2.0063, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 9.7139, mae: 2.3067, huber: 1.8804, swd: 3.0656, ept: 57.0513
    Epoch [10/50], Val Losses: mse: 10.1648, mae: 2.4439, huber: 2.0092, swd: 2.9402, ept: 54.6520
    Epoch [10/50], Test Losses: mse: 11.5368, mae: 2.5285, huber: 2.0965, swd: 3.3268, ept: 54.5173
      Epoch 10 composite train-obj: 1.880417
            No improvement (2.0092), counter 1/5
    Epoch [11/50], Train Losses: mse: 9.6955, mae: 2.3015, huber: 1.8760, swd: 3.0341, ept: 57.2180
    Epoch [11/50], Val Losses: mse: 9.8943, mae: 2.4223, huber: 1.9879, swd: 3.0985, ept: 54.1053
    Epoch [11/50], Test Losses: mse: 11.5220, mae: 2.5390, huber: 2.1052, swd: 3.5338, ept: 53.3939
      Epoch 11 composite train-obj: 1.876020
            Val objective improved 2.0063 → 1.9879, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 9.6675, mae: 2.2978, huber: 1.8726, swd: 3.0252, ept: 57.2234
    Epoch [12/50], Val Losses: mse: 10.1211, mae: 2.4359, huber: 2.0031, swd: 2.9452, ept: 54.4678
    Epoch [12/50], Test Losses: mse: 11.3380, mae: 2.5126, huber: 2.0809, swd: 3.3322, ept: 54.0191
      Epoch 12 composite train-obj: 1.872606
            No improvement (2.0031), counter 1/5
    Epoch [13/50], Train Losses: mse: 9.6202, mae: 2.2874, huber: 1.8631, swd: 3.0116, ept: 57.5211
    Epoch [13/50], Val Losses: mse: 9.9848, mae: 2.4179, huber: 1.9866, swd: 3.0705, ept: 53.7983
    Epoch [13/50], Test Losses: mse: 11.1534, mae: 2.4865, huber: 2.0559, swd: 3.4337, ept: 54.8897
      Epoch 13 composite train-obj: 1.863070
            Val objective improved 1.9879 → 1.9866, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 9.5848, mae: 2.2829, huber: 1.8591, swd: 2.9947, ept: 57.6743
    Epoch [14/50], Val Losses: mse: 10.2581, mae: 2.4669, huber: 2.0329, swd: 3.0096, ept: 53.8192
    Epoch [14/50], Test Losses: mse: 11.3701, mae: 2.5086, huber: 2.0777, swd: 3.2582, ept: 54.5048
      Epoch 14 composite train-obj: 1.859085
            No improvement (2.0329), counter 1/5
    Epoch [15/50], Train Losses: mse: 9.5837, mae: 2.2809, huber: 1.8573, swd: 2.9779, ept: 57.6875
    Epoch [15/50], Val Losses: mse: 10.1157, mae: 2.4367, huber: 2.0046, swd: 2.8425, ept: 53.6014
    Epoch [15/50], Test Losses: mse: 11.2058, mae: 2.4918, huber: 2.0613, swd: 3.2682, ept: 54.8369
      Epoch 15 composite train-obj: 1.857273
            No improvement (2.0046), counter 2/5
    Epoch [16/50], Train Losses: mse: 9.5590, mae: 2.2753, huber: 1.8525, swd: 2.9684, ept: 57.8712
    Epoch [16/50], Val Losses: mse: 10.2694, mae: 2.4513, huber: 2.0204, swd: 3.0894, ept: 53.9182
    Epoch [16/50], Test Losses: mse: 11.2455, mae: 2.4966, huber: 2.0679, swd: 3.3933, ept: 54.7954
      Epoch 16 composite train-obj: 1.852469
            No improvement (2.0204), counter 3/5
    Epoch [17/50], Train Losses: mse: 9.5222, mae: 2.2694, huber: 1.8472, swd: 2.9577, ept: 57.8510
    Epoch [17/50], Val Losses: mse: 9.9063, mae: 2.4011, huber: 1.9703, swd: 2.8967, ept: 54.6280
    Epoch [17/50], Test Losses: mse: 11.2038, mae: 2.4851, huber: 2.0559, swd: 3.3332, ept: 54.9636
      Epoch 17 composite train-obj: 1.847220
            Val objective improved 1.9866 → 1.9703, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 9.4992, mae: 2.2643, huber: 1.8426, swd: 2.9447, ept: 58.0415
    Epoch [18/50], Val Losses: mse: 9.9465, mae: 2.4028, huber: 1.9742, swd: 2.9375, ept: 54.1842
    Epoch [18/50], Test Losses: mse: 11.1992, mae: 2.4781, huber: 2.0500, swd: 3.3887, ept: 55.4167
      Epoch 18 composite train-obj: 1.842587
            No improvement (1.9742), counter 1/5
    Epoch [19/50], Train Losses: mse: 9.4999, mae: 2.2643, huber: 1.8427, swd: 2.9272, ept: 58.1311
    Epoch [19/50], Val Losses: mse: 9.7880, mae: 2.3786, huber: 1.9478, swd: 3.0160, ept: 55.1248
    Epoch [19/50], Test Losses: mse: 11.2109, mae: 2.4896, huber: 2.0602, swd: 3.3744, ept: 54.5617
      Epoch 19 composite train-obj: 1.842748
            Val objective improved 1.9703 → 1.9478, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 9.4420, mae: 2.2551, huber: 1.8339, swd: 2.9149, ept: 58.2471
    Epoch [20/50], Val Losses: mse: 10.0110, mae: 2.4232, huber: 1.9920, swd: 3.1840, ept: 54.4195
    Epoch [20/50], Test Losses: mse: 10.9179, mae: 2.4411, huber: 2.0135, swd: 3.3248, ept: 56.2356
      Epoch 20 composite train-obj: 1.833941
            No improvement (1.9920), counter 1/5
    Epoch [21/50], Train Losses: mse: 9.4441, mae: 2.2548, huber: 1.8341, swd: 2.9119, ept: 58.2931
    Epoch [21/50], Val Losses: mse: 10.1002, mae: 2.4272, huber: 1.9967, swd: 2.7950, ept: 54.4879
    Epoch [21/50], Test Losses: mse: 11.2157, mae: 2.4807, huber: 2.0517, swd: 3.2052, ept: 55.3776
      Epoch 21 composite train-obj: 1.834123
            No improvement (1.9967), counter 2/5
    Epoch [22/50], Train Losses: mse: 9.4300, mae: 2.2522, huber: 1.8316, swd: 2.8963, ept: 58.3384
    Epoch [22/50], Val Losses: mse: 9.9304, mae: 2.4029, huber: 1.9733, swd: 2.8758, ept: 55.8608
    Epoch [22/50], Test Losses: mse: 11.0141, mae: 2.4576, huber: 2.0297, swd: 3.2323, ept: 55.6104
      Epoch 22 composite train-obj: 1.831620
            No improvement (1.9733), counter 3/5
    Epoch [23/50], Train Losses: mse: 9.4358, mae: 2.2518, huber: 1.8312, swd: 2.8959, ept: 58.2564
    Epoch [23/50], Val Losses: mse: 9.9198, mae: 2.3949, huber: 1.9660, swd: 2.7586, ept: 54.4922
    Epoch [23/50], Test Losses: mse: 11.2094, mae: 2.4787, huber: 2.0517, swd: 3.2225, ept: 54.9276
      Epoch 23 composite train-obj: 1.831220
            No improvement (1.9660), counter 4/5
    Epoch [24/50], Train Losses: mse: 9.3945, mae: 2.2453, huber: 1.8255, swd: 2.8836, ept: 58.4676
    Epoch [24/50], Val Losses: mse: 9.6585, mae: 2.3582, huber: 1.9309, swd: 2.6521, ept: 55.8700
    Epoch [24/50], Test Losses: mse: 11.0708, mae: 2.4638, huber: 2.0370, swd: 3.2384, ept: 55.4134
      Epoch 24 composite train-obj: 1.825474
            Val objective improved 1.9478 → 1.9309, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 9.3806, mae: 2.2430, huber: 1.8233, swd: 2.8641, ept: 58.5339
    Epoch [25/50], Val Losses: mse: 9.6684, mae: 2.3726, huber: 1.9434, swd: 2.8047, ept: 55.9141
    Epoch [25/50], Test Losses: mse: 10.9171, mae: 2.4400, huber: 2.0142, swd: 3.2375, ept: 55.5762
      Epoch 25 composite train-obj: 1.823339
            No improvement (1.9434), counter 1/5
    Epoch [26/50], Train Losses: mse: 9.3542, mae: 2.2396, huber: 1.8202, swd: 2.8606, ept: 58.6591
    Epoch [26/50], Val Losses: mse: 9.6177, mae: 2.3512, huber: 1.9230, swd: 2.6027, ept: 56.0796
    Epoch [26/50], Test Losses: mse: 10.9460, mae: 2.4451, huber: 2.0188, swd: 3.0586, ept: 55.6121
      Epoch 26 composite train-obj: 1.820204
            Val objective improved 1.9309 → 1.9230, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 9.3820, mae: 2.2418, huber: 1.8223, swd: 2.8506, ept: 58.5538
    Epoch [27/50], Val Losses: mse: 9.7469, mae: 2.3685, huber: 1.9404, swd: 2.8650, ept: 55.3943
    Epoch [27/50], Test Losses: mse: 11.2093, mae: 2.4776, huber: 2.0513, swd: 3.3032, ept: 54.7828
      Epoch 27 composite train-obj: 1.822254
            No improvement (1.9404), counter 1/5
    Epoch [28/50], Train Losses: mse: 9.3772, mae: 2.2412, huber: 1.8221, swd: 2.8587, ept: 58.5922
    Epoch [28/50], Val Losses: mse: 9.5323, mae: 2.3417, huber: 1.9153, swd: 2.5829, ept: 56.6559
    Epoch [28/50], Test Losses: mse: 10.9664, mae: 2.4470, huber: 2.0217, swd: 3.0732, ept: 56.1470
      Epoch 28 composite train-obj: 1.822072
            Val objective improved 1.9230 → 1.9153, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 9.3343, mae: 2.2336, huber: 1.8149, swd: 2.8375, ept: 58.7250
    Epoch [29/50], Val Losses: mse: 9.9669, mae: 2.4015, huber: 1.9726, swd: 2.9945, ept: 56.1372
    Epoch [29/50], Test Losses: mse: 11.0143, mae: 2.4624, huber: 2.0358, swd: 3.4198, ept: 55.7450
      Epoch 29 composite train-obj: 1.814909
            No improvement (1.9726), counter 1/5
    Epoch [30/50], Train Losses: mse: 9.3226, mae: 2.2339, huber: 1.8151, swd: 2.8494, ept: 58.7318
    Epoch [30/50], Val Losses: mse: 10.1390, mae: 2.4153, huber: 1.9881, swd: 2.9729, ept: 55.8224
    Epoch [30/50], Test Losses: mse: 11.1194, mae: 2.4684, huber: 2.0423, swd: 3.2748, ept: 55.4292
      Epoch 30 composite train-obj: 1.815100
            No improvement (1.9881), counter 2/5
    Epoch [31/50], Train Losses: mse: 9.3166, mae: 2.2309, huber: 1.8124, swd: 2.8235, ept: 58.7312
    Epoch [31/50], Val Losses: mse: 9.9601, mae: 2.4043, huber: 1.9752, swd: 3.1697, ept: 54.8763
    Epoch [31/50], Test Losses: mse: 10.9613, mae: 2.4531, huber: 2.0270, swd: 3.4122, ept: 55.6794
      Epoch 31 composite train-obj: 1.812429
            No improvement (1.9752), counter 3/5
    Epoch [32/50], Train Losses: mse: 9.2645, mae: 2.2239, huber: 1.8058, swd: 2.8265, ept: 58.8416
    Epoch [32/50], Val Losses: mse: 9.6704, mae: 2.3589, huber: 1.9312, swd: 2.6458, ept: 57.1397
    Epoch [32/50], Test Losses: mse: 10.8124, mae: 2.4284, huber: 2.0029, swd: 3.1469, ept: 56.3688
      Epoch 32 composite train-obj: 1.805842
            No improvement (1.9312), counter 4/5
    Epoch [33/50], Train Losses: mse: 9.2880, mae: 2.2263, huber: 1.8080, swd: 2.8025, ept: 58.9165
    Epoch [33/50], Val Losses: mse: 9.6160, mae: 2.3617, huber: 1.9339, swd: 2.6718, ept: 56.5036
    Epoch [33/50], Test Losses: mse: 10.9036, mae: 2.4426, huber: 2.0168, swd: 3.1240, ept: 56.0617
      Epoch 33 composite train-obj: 1.808029
    Epoch [33/50], Test Losses: mse: 10.9664, mae: 2.4470, huber: 2.0217, swd: 3.0732, ept: 56.1470
    Best round's Test MSE: 10.9664, MAE: 2.4470, SWD: 3.0732
    Best round's Validation MSE: 9.5323, MAE: 2.3417, SWD: 2.5829
    Best round's Test verification MSE : 10.9664, MAE: 2.4470, SWD: 3.0732
    Time taken: 43.29 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.9145, mae: 2.7118, huber: 2.2540, swd: 3.8811, ept: 37.0234
    Epoch [1/50], Val Losses: mse: 11.4184, mae: 2.7270, huber: 2.2647, swd: 3.3895, ept: 43.5670
    Epoch [1/50], Test Losses: mse: 12.6437, mae: 2.7764, huber: 2.3179, swd: 3.9354, ept: 46.0291
      Epoch 1 composite train-obj: 2.254031
            Val objective improved inf → 2.2647, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.4939, mae: 2.4824, huber: 2.0351, swd: 3.5210, ept: 51.5752
    Epoch [2/50], Val Losses: mse: 11.0433, mae: 2.6154, huber: 2.1644, swd: 3.0541, ept: 47.9121
    Epoch [2/50], Test Losses: mse: 12.2546, mae: 2.6789, huber: 2.2285, swd: 3.5317, ept: 50.2169
      Epoch 2 composite train-obj: 2.035101
            Val objective improved 2.2647 → 2.1644, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 10.2161, mae: 2.4209, huber: 1.9800, swd: 3.3597, ept: 53.7035
    Epoch [3/50], Val Losses: mse: 10.4675, mae: 2.5392, huber: 2.0913, swd: 3.2522, ept: 51.1930
    Epoch [3/50], Test Losses: mse: 11.6796, mae: 2.5989, huber: 2.1540, swd: 3.6277, ept: 51.8947
      Epoch 3 composite train-obj: 1.979991
            Val objective improved 2.1644 → 2.0913, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 10.0767, mae: 2.3872, huber: 1.9507, swd: 3.2800, ept: 54.7893
    Epoch [4/50], Val Losses: mse: 10.8367, mae: 2.5597, huber: 2.1168, swd: 3.1194, ept: 50.2894
    Epoch [4/50], Test Losses: mse: 11.7604, mae: 2.5911, huber: 2.1497, swd: 3.4452, ept: 52.1986
      Epoch 4 composite train-obj: 1.950651
            No improvement (2.1168), counter 1/5
    Epoch [5/50], Train Losses: mse: 9.9864, mae: 2.3672, huber: 1.9333, swd: 3.2315, ept: 55.3997
    Epoch [5/50], Val Losses: mse: 10.8784, mae: 2.5642, huber: 2.1223, swd: 3.1778, ept: 51.8799
    Epoch [5/50], Test Losses: mse: 11.8569, mae: 2.5908, huber: 2.1524, swd: 3.4339, ept: 52.1552
      Epoch 5 composite train-obj: 1.933342
            No improvement (2.1223), counter 2/5
    Epoch [6/50], Train Losses: mse: 9.9264, mae: 2.3516, huber: 1.9199, swd: 3.1793, ept: 55.8457
    Epoch [6/50], Val Losses: mse: 10.6420, mae: 2.5290, huber: 2.0903, swd: 3.2185, ept: 51.0723
    Epoch [6/50], Test Losses: mse: 11.5486, mae: 2.5526, huber: 2.1148, swd: 3.5757, ept: 52.6184
      Epoch 6 composite train-obj: 1.919863
            Val objective improved 2.0913 → 2.0903, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 9.8448, mae: 2.3349, huber: 1.9051, swd: 3.1406, ept: 56.2739
    Epoch [7/50], Val Losses: mse: 10.1192, mae: 2.4650, huber: 2.0266, swd: 2.9760, ept: 52.1063
    Epoch [7/50], Test Losses: mse: 11.5375, mae: 2.5438, huber: 2.1080, swd: 3.3698, ept: 53.6190
      Epoch 7 composite train-obj: 1.905144
            Val objective improved 2.0903 → 2.0266, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 9.7983, mae: 2.3245, huber: 1.8959, swd: 3.1187, ept: 56.6686
    Epoch [8/50], Val Losses: mse: 10.0960, mae: 2.4587, huber: 2.0211, swd: 3.0728, ept: 52.8053
    Epoch [8/50], Test Losses: mse: 11.4838, mae: 2.5388, huber: 2.1042, swd: 3.4799, ept: 53.8965
      Epoch 8 composite train-obj: 1.895877
            Val objective improved 2.0266 → 2.0211, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 9.7519, mae: 2.3159, huber: 1.8883, swd: 3.1101, ept: 56.7690
    Epoch [9/50], Val Losses: mse: 10.1734, mae: 2.4557, huber: 2.0188, swd: 3.1003, ept: 53.4910
    Epoch [9/50], Test Losses: mse: 11.4062, mae: 2.5193, huber: 2.0851, swd: 3.4988, ept: 54.3473
      Epoch 9 composite train-obj: 1.888294
            Val objective improved 2.0211 → 2.0188, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 9.7234, mae: 2.3079, huber: 1.8814, swd: 3.0634, ept: 56.9790
    Epoch [10/50], Val Losses: mse: 10.6052, mae: 2.5086, huber: 2.0732, swd: 3.1912, ept: 52.6343
    Epoch [10/50], Test Losses: mse: 11.2945, mae: 2.5180, huber: 2.0847, swd: 3.4640, ept: 53.7939
      Epoch 10 composite train-obj: 1.881390
            No improvement (2.0732), counter 1/5
    Epoch [11/50], Train Losses: mse: 9.6667, mae: 2.3006, huber: 1.8749, swd: 3.0583, ept: 57.2244
    Epoch [11/50], Val Losses: mse: 10.1097, mae: 2.4284, huber: 1.9953, swd: 2.8948, ept: 54.1788
    Epoch [11/50], Test Losses: mse: 11.3047, mae: 2.5105, huber: 2.0781, swd: 3.4092, ept: 54.5183
      Epoch 11 composite train-obj: 1.874882
            Val objective improved 2.0188 → 1.9953, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 9.6322, mae: 2.2909, huber: 1.8663, swd: 3.0194, ept: 57.4249
    Epoch [12/50], Val Losses: mse: 10.3523, mae: 2.4714, huber: 2.0366, swd: 3.0543, ept: 53.4245
    Epoch [12/50], Test Losses: mse: 11.3414, mae: 2.5043, huber: 2.0733, swd: 3.3594, ept: 54.2347
      Epoch 12 composite train-obj: 1.866287
            No improvement (2.0366), counter 1/5
    Epoch [13/50], Train Losses: mse: 9.6023, mae: 2.2860, huber: 1.8618, swd: 3.0149, ept: 57.4485
    Epoch [13/50], Val Losses: mse: 10.0130, mae: 2.4222, huber: 1.9913, swd: 2.8271, ept: 53.7945
    Epoch [13/50], Test Losses: mse: 11.2588, mae: 2.4984, huber: 2.0675, swd: 3.2726, ept: 54.5725
      Epoch 13 composite train-obj: 1.861809
            Val objective improved 1.9953 → 1.9913, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 9.6145, mae: 2.2857, huber: 1.8619, swd: 2.9964, ept: 57.5910
    Epoch [14/50], Val Losses: mse: 10.0953, mae: 2.4195, huber: 1.9884, swd: 3.1613, ept: 55.0596
    Epoch [14/50], Test Losses: mse: 11.2512, mae: 2.4981, huber: 2.0666, swd: 3.5061, ept: 54.4397
      Epoch 14 composite train-obj: 1.861892
            Val objective improved 1.9913 → 1.9884, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 9.5846, mae: 2.2818, huber: 1.8582, swd: 3.0029, ept: 57.6316
    Epoch [15/50], Val Losses: mse: 10.0087, mae: 2.4181, huber: 1.9854, swd: 3.0310, ept: 54.9299
    Epoch [15/50], Test Losses: mse: 11.2103, mae: 2.4857, huber: 2.0546, swd: 3.2910, ept: 54.9115
      Epoch 15 composite train-obj: 1.858177
            Val objective improved 1.9884 → 1.9854, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 9.5501, mae: 2.2739, huber: 1.8510, swd: 2.9740, ept: 57.9133
    Epoch [16/50], Val Losses: mse: 10.2300, mae: 2.4456, huber: 2.0134, swd: 3.0247, ept: 54.2151
    Epoch [16/50], Test Losses: mse: 11.1830, mae: 2.4836, huber: 2.0550, swd: 3.3569, ept: 54.9577
      Epoch 16 composite train-obj: 1.851013
            No improvement (2.0134), counter 1/5
    Epoch [17/50], Train Losses: mse: 9.5409, mae: 2.2722, huber: 1.8498, swd: 2.9643, ept: 57.7963
    Epoch [17/50], Val Losses: mse: 9.7038, mae: 2.3851, huber: 1.9551, swd: 2.9417, ept: 54.8983
    Epoch [17/50], Test Losses: mse: 11.0622, mae: 2.4686, huber: 2.0396, swd: 3.3086, ept: 55.4256
      Epoch 17 composite train-obj: 1.849835
            Val objective improved 1.9854 → 1.9551, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 9.4966, mae: 2.2643, huber: 1.8423, swd: 2.9442, ept: 58.0182
    Epoch [18/50], Val Losses: mse: 10.0012, mae: 2.4027, huber: 1.9741, swd: 2.8722, ept: 54.3385
    Epoch [18/50], Test Losses: mse: 11.1429, mae: 2.4668, huber: 2.0390, swd: 3.1446, ept: 55.2550
      Epoch 18 composite train-obj: 1.842300
            No improvement (1.9741), counter 1/5
    Epoch [19/50], Train Losses: mse: 9.4798, mae: 2.2618, huber: 1.8401, swd: 2.9174, ept: 58.0523
    Epoch [19/50], Val Losses: mse: 10.0918, mae: 2.4242, huber: 1.9936, swd: 3.2736, ept: 54.8435
    Epoch [19/50], Test Losses: mse: 11.0189, mae: 2.4645, huber: 2.0358, swd: 3.4850, ept: 54.6944
      Epoch 19 composite train-obj: 1.840074
            No improvement (1.9936), counter 2/5
    Epoch [20/50], Train Losses: mse: 9.4647, mae: 2.2599, huber: 1.8386, swd: 2.9364, ept: 58.1437
    Epoch [20/50], Val Losses: mse: 9.9258, mae: 2.4027, huber: 1.9720, swd: 2.9650, ept: 54.4803
    Epoch [20/50], Test Losses: mse: 11.1464, mae: 2.4787, huber: 2.0493, swd: 3.4085, ept: 55.5474
      Epoch 20 composite train-obj: 1.838599
            No improvement (1.9720), counter 3/5
    Epoch [21/50], Train Losses: mse: 9.4476, mae: 2.2555, huber: 1.8345, swd: 2.9139, ept: 58.2745
    Epoch [21/50], Val Losses: mse: 9.9295, mae: 2.3924, huber: 1.9642, swd: 2.9983, ept: 54.9676
    Epoch [21/50], Test Losses: mse: 10.9467, mae: 2.4567, huber: 2.0281, swd: 3.2755, ept: 55.6083
      Epoch 21 composite train-obj: 1.834516
            No improvement (1.9642), counter 4/5
    Epoch [22/50], Train Losses: mse: 9.4344, mae: 2.2535, huber: 1.8327, swd: 2.9066, ept: 58.3159
    Epoch [22/50], Val Losses: mse: 9.9979, mae: 2.4168, huber: 1.9875, swd: 3.0149, ept: 54.9287
    Epoch [22/50], Test Losses: mse: 11.0731, mae: 2.4677, huber: 2.0406, swd: 3.3361, ept: 54.9246
      Epoch 22 composite train-obj: 1.832747
    Epoch [22/50], Test Losses: mse: 11.0622, mae: 2.4686, huber: 2.0396, swd: 3.3086, ept: 55.4256
    Best round's Test MSE: 11.0622, MAE: 2.4686, SWD: 3.3086
    Best round's Validation MSE: 9.7038, MAE: 2.3851, SWD: 2.9417
    Best round's Test verification MSE : 11.0622, MAE: 2.4686, SWD: 3.3086
    Time taken: 30.19 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.8880, mae: 2.7082, huber: 2.2507, swd: 3.5570, ept: 37.5074
    Epoch [1/50], Val Losses: mse: 11.8505, mae: 2.7641, huber: 2.3019, swd: 3.1794, ept: 44.6366
    Epoch [1/50], Test Losses: mse: 12.6133, mae: 2.7528, huber: 2.2968, swd: 3.5098, ept: 46.7947
      Epoch 1 composite train-obj: 2.250691
            Val objective improved inf → 2.3019, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 10.4867, mae: 2.4809, huber: 2.0339, swd: 3.1869, ept: 51.6921
    Epoch [2/50], Val Losses: mse: 11.4689, mae: 2.6909, huber: 2.2376, swd: 3.0408, ept: 47.3029
    Epoch [2/50], Test Losses: mse: 12.1643, mae: 2.6767, huber: 2.2271, swd: 3.3507, ept: 49.4312
      Epoch 2 composite train-obj: 2.033908
            Val objective improved 2.3019 → 2.2376, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 10.2100, mae: 2.4201, huber: 1.9792, swd: 3.0370, ept: 53.8193
    Epoch [3/50], Val Losses: mse: 10.8285, mae: 2.5863, huber: 2.1373, swd: 2.9380, ept: 50.2851
    Epoch [3/50], Test Losses: mse: 11.7246, mae: 2.6000, huber: 2.1551, swd: 3.3281, ept: 51.7655
      Epoch 3 composite train-obj: 1.979208
            Val objective improved 2.2376 → 2.1373, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 10.0669, mae: 2.3879, huber: 1.9511, swd: 2.9600, ept: 54.8394
    Epoch [4/50], Val Losses: mse: 10.3606, mae: 2.5040, huber: 2.0597, swd: 2.8652, ept: 51.9245
    Epoch [4/50], Test Losses: mse: 11.5077, mae: 2.5738, huber: 2.1318, swd: 3.3864, ept: 52.6903
      Epoch 4 composite train-obj: 1.951083
            Val objective improved 2.1373 → 2.0597, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 9.9956, mae: 2.3674, huber: 1.9335, swd: 2.9017, ept: 55.4079
    Epoch [5/50], Val Losses: mse: 10.4364, mae: 2.5265, huber: 2.0822, swd: 2.8842, ept: 51.9408
    Epoch [5/50], Test Losses: mse: 11.5006, mae: 2.5594, huber: 2.1199, swd: 3.2345, ept: 52.9083
      Epoch 5 composite train-obj: 1.933521
            No improvement (2.0822), counter 1/5
    Epoch [6/50], Train Losses: mse: 9.8971, mae: 2.3460, huber: 1.9146, swd: 2.8540, ept: 56.0119
    Epoch [6/50], Val Losses: mse: 10.3050, mae: 2.5026, huber: 2.0627, swd: 3.0045, ept: 51.7470
    Epoch [6/50], Test Losses: mse: 11.6366, mae: 2.5773, huber: 2.1384, swd: 3.3107, ept: 52.8445
      Epoch 6 composite train-obj: 1.914566
            No improvement (2.0627), counter 2/5
    Epoch [7/50], Train Losses: mse: 9.8502, mae: 2.3371, huber: 1.9071, swd: 2.8262, ept: 56.2010
    Epoch [7/50], Val Losses: mse: 10.1886, mae: 2.4542, huber: 2.0177, swd: 2.6973, ept: 53.3158
    Epoch [7/50], Test Losses: mse: 11.5540, mae: 2.5451, huber: 2.1089, swd: 3.1244, ept: 54.0693
      Epoch 7 composite train-obj: 1.907137
            Val objective improved 2.0597 → 2.0177, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 9.8065, mae: 2.3266, huber: 1.8979, swd: 2.7961, ept: 56.6808
    Epoch [8/50], Val Losses: mse: 10.6595, mae: 2.5094, huber: 2.0730, swd: 2.6276, ept: 52.6603
    Epoch [8/50], Test Losses: mse: 11.4807, mae: 2.5328, huber: 2.0981, swd: 2.9597, ept: 54.2264
      Epoch 8 composite train-obj: 1.897878
            No improvement (2.0730), counter 1/5
    Epoch [9/50], Train Losses: mse: 9.7586, mae: 2.3161, huber: 1.8887, swd: 2.7667, ept: 56.8269
    Epoch [9/50], Val Losses: mse: 10.2297, mae: 2.4680, huber: 2.0314, swd: 2.6837, ept: 52.1818
    Epoch [9/50], Test Losses: mse: 11.4088, mae: 2.5256, huber: 2.0914, swd: 3.0945, ept: 53.9406
      Epoch 9 composite train-obj: 1.888672
            No improvement (2.0314), counter 2/5
    Epoch [10/50], Train Losses: mse: 9.7101, mae: 2.3057, huber: 1.8793, swd: 2.7362, ept: 57.0909
    Epoch [10/50], Val Losses: mse: 9.8177, mae: 2.4135, huber: 1.9782, swd: 2.6782, ept: 54.7715
    Epoch [10/50], Test Losses: mse: 11.2580, mae: 2.5150, huber: 2.0817, swd: 3.1511, ept: 54.1161
      Epoch 10 composite train-obj: 1.879349
            Val objective improved 2.0177 → 1.9782, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 9.6655, mae: 2.2986, huber: 1.8730, swd: 2.7277, ept: 57.2536
    Epoch [11/50], Val Losses: mse: 10.0150, mae: 2.4223, huber: 1.9879, swd: 2.5916, ept: 55.2593
    Epoch [11/50], Test Losses: mse: 11.1255, mae: 2.4783, huber: 2.0465, swd: 2.9794, ept: 55.3218
      Epoch 11 composite train-obj: 1.872974
            No improvement (1.9879), counter 1/5
    Epoch [12/50], Train Losses: mse: 9.6483, mae: 2.2933, huber: 1.8685, swd: 2.7157, ept: 57.3409
    Epoch [12/50], Val Losses: mse: 10.1470, mae: 2.4369, huber: 2.0045, swd: 2.4855, ept: 53.8538
    Epoch [12/50], Test Losses: mse: 11.4171, mae: 2.5144, huber: 2.0831, swd: 2.9169, ept: 54.0342
      Epoch 12 composite train-obj: 1.868466
            No improvement (2.0045), counter 2/5
    Epoch [13/50], Train Losses: mse: 9.6199, mae: 2.2882, huber: 1.8637, swd: 2.6866, ept: 57.4871
    Epoch [13/50], Val Losses: mse: 10.0672, mae: 2.4490, huber: 2.0137, swd: 2.7277, ept: 54.2533
    Epoch [13/50], Test Losses: mse: 11.1662, mae: 2.4856, huber: 2.0546, swd: 3.0840, ept: 54.8083
      Epoch 13 composite train-obj: 1.863662
            No improvement (2.0137), counter 3/5
    Epoch [14/50], Train Losses: mse: 9.5832, mae: 2.2827, huber: 1.8586, swd: 2.6826, ept: 57.5690
    Epoch [14/50], Val Losses: mse: 10.2151, mae: 2.4468, huber: 2.0137, swd: 2.7085, ept: 52.7364
    Epoch [14/50], Test Losses: mse: 11.2286, mae: 2.4883, huber: 2.0585, swd: 2.9959, ept: 54.6879
      Epoch 14 composite train-obj: 1.858634
            No improvement (2.0137), counter 4/5
    Epoch [15/50], Train Losses: mse: 9.5688, mae: 2.2778, huber: 1.8547, swd: 2.6725, ept: 57.7141
    Epoch [15/50], Val Losses: mse: 10.3251, mae: 2.4540, huber: 2.0229, swd: 2.6885, ept: 53.8055
    Epoch [15/50], Test Losses: mse: 11.3556, mae: 2.5099, huber: 2.0795, swd: 3.0284, ept: 55.1743
      Epoch 15 composite train-obj: 1.854655
    Epoch [15/50], Test Losses: mse: 11.2580, mae: 2.5150, huber: 2.0817, swd: 3.1511, ept: 54.1161
    Best round's Test MSE: 11.2580, MAE: 2.5150, SWD: 3.1511
    Best round's Validation MSE: 9.8177, MAE: 2.4135, SWD: 2.6782
    Best round's Test verification MSE : 11.2580, MAE: 2.5150, SWD: 3.1511
    Time taken: 20.85 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz96_seq720_pred96_20250512_2153)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 11.0955 ± 0.1214
      mae: 2.4769 ± 0.0284
      huber: 2.0476 ± 0.0251
      swd: 3.1776 ± 0.0979
      ept: 55.2295 ± 0.8406
      count: 9.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 9.6846 ± 0.1173
      mae: 2.3801 ± 0.0295
      huber: 1.9495 ± 0.0260
      swd: 2.7343 ± 0.1517
      ept: 55.4419 ± 0.8600
      count: 9.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 94.37 seconds
    
    Experiment complete: DLinear_lorenz96_seq720_pred96_20250512_2153
    Model: DLinear
    Dataset: lorenz96
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
    channels=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([196, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([196, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 97
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 196, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 97
    Validation Batches: 8
    Test Batches: 23
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.6387, mae: 2.8196, huber: 2.3592, swd: 3.4402, ept: 42.0050
    Epoch [1/50], Val Losses: mse: 12.5189, mae: 2.8769, huber: 2.4128, swd: 3.2860, ept: 48.2196
    Epoch [1/50], Test Losses: mse: 14.1271, mae: 2.9740, huber: 2.5111, swd: 3.6237, ept: 49.8169
      Epoch 1 composite train-obj: 2.359221
            Val objective improved inf → 2.4128, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.6094, mae: 2.6656, huber: 2.2113, swd: 3.2543, ept: 60.6512
    Epoch [2/50], Val Losses: mse: 11.9727, mae: 2.7799, huber: 2.3202, swd: 2.9325, ept: 53.1155
    Epoch [2/50], Test Losses: mse: 13.9677, mae: 2.9316, huber: 2.4726, swd: 3.3797, ept: 54.0256
      Epoch 2 composite train-obj: 2.211265
            Val objective improved 2.4128 → 2.3202, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.4907, mae: 2.6363, huber: 2.1853, swd: 3.1427, ept: 63.2859
    Epoch [3/50], Val Losses: mse: 12.2004, mae: 2.8107, huber: 2.3529, swd: 3.1335, ept: 53.7472
    Epoch [3/50], Test Losses: mse: 13.7259, mae: 2.8983, huber: 2.4416, swd: 3.4299, ept: 56.4409
      Epoch 3 composite train-obj: 2.185256
            No improvement (2.3529), counter 1/5
    Epoch [4/50], Train Losses: mse: 11.4100, mae: 2.6184, huber: 2.1694, swd: 3.0927, ept: 64.7632
    Epoch [4/50], Val Losses: mse: 12.3974, mae: 2.8359, huber: 2.3778, swd: 3.0767, ept: 55.9169
    Epoch [4/50], Test Losses: mse: 13.7985, mae: 2.8984, huber: 2.4435, swd: 3.2910, ept: 57.4385
      Epoch 4 composite train-obj: 2.169363
            No improvement (2.3778), counter 2/5
    Epoch [5/50], Train Losses: mse: 11.3624, mae: 2.6072, huber: 2.1597, swd: 3.0444, ept: 65.2990
    Epoch [5/50], Val Losses: mse: 11.5897, mae: 2.7177, huber: 2.2634, swd: 2.8274, ept: 58.8555
    Epoch [5/50], Test Losses: mse: 13.6396, mae: 2.8716, huber: 2.4185, swd: 3.2945, ept: 57.3073
      Epoch 5 composite train-obj: 2.159715
            Val objective improved 2.3202 → 2.2634, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.3380, mae: 2.6002, huber: 2.1537, swd: 3.0246, ept: 65.8859
    Epoch [6/50], Val Losses: mse: 11.9492, mae: 2.7658, huber: 2.3116, swd: 2.8357, ept: 55.2179
    Epoch [6/50], Test Losses: mse: 13.6463, mae: 2.8722, huber: 2.4195, swd: 3.2084, ept: 59.8577
      Epoch 6 composite train-obj: 2.153681
            No improvement (2.3116), counter 1/5
    Epoch [7/50], Train Losses: mse: 11.2704, mae: 2.5880, huber: 2.1424, swd: 2.9841, ept: 66.4872
    Epoch [7/50], Val Losses: mse: 12.1457, mae: 2.7760, huber: 2.3232, swd: 3.0032, ept: 56.4436
    Epoch [7/50], Test Losses: mse: 13.6880, mae: 2.8735, huber: 2.4212, swd: 3.3486, ept: 58.2374
      Epoch 7 composite train-obj: 2.142404
            No improvement (2.3232), counter 2/5
    Epoch [8/50], Train Losses: mse: 11.2379, mae: 2.5836, huber: 2.1386, swd: 2.9703, ept: 66.7928
    Epoch [8/50], Val Losses: mse: 11.6510, mae: 2.7268, huber: 2.2737, swd: 2.6684, ept: 57.9472
    Epoch [8/50], Test Losses: mse: 13.4689, mae: 2.8420, huber: 2.3908, swd: 3.0461, ept: 59.0369
      Epoch 8 composite train-obj: 2.138634
            No improvement (2.2737), counter 3/5
    Epoch [9/50], Train Losses: mse: 11.2152, mae: 2.5773, huber: 2.1329, swd: 2.9532, ept: 67.0163
    Epoch [9/50], Val Losses: mse: 11.9190, mae: 2.7490, huber: 2.2962, swd: 3.0253, ept: 56.7802
    Epoch [9/50], Test Losses: mse: 13.3257, mae: 2.8359, huber: 2.3845, swd: 3.3072, ept: 59.6222
      Epoch 9 composite train-obj: 2.132935
            No improvement (2.2962), counter 4/5
    Epoch [10/50], Train Losses: mse: 11.1973, mae: 2.5723, huber: 2.1287, swd: 2.9227, ept: 67.4671
    Epoch [10/50], Val Losses: mse: 11.7598, mae: 2.7317, huber: 2.2798, swd: 2.7718, ept: 57.8783
    Epoch [10/50], Test Losses: mse: 13.5488, mae: 2.8501, huber: 2.3998, swd: 3.0363, ept: 60.5771
      Epoch 10 composite train-obj: 2.128658
    Epoch [10/50], Test Losses: mse: 13.6396, mae: 2.8716, huber: 2.4185, swd: 3.2945, ept: 57.3073
    Best round's Test MSE: 13.6396, MAE: 2.8716, SWD: 3.2945
    Best round's Validation MSE: 11.5897, MAE: 2.7177, SWD: 2.8274
    Best round's Test verification MSE : 13.6396, MAE: 2.8716, SWD: 3.2945
    Time taken: 13.91 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.6114, mae: 2.8153, huber: 2.3552, swd: 3.4553, ept: 41.9695
    Epoch [1/50], Val Losses: mse: 12.4933, mae: 2.8689, huber: 2.4044, swd: 3.2417, ept: 48.7892
    Epoch [1/50], Test Losses: mse: 13.9950, mae: 2.9523, huber: 2.4903, swd: 3.5713, ept: 49.6983
      Epoch 1 composite train-obj: 2.355173
            Val objective improved inf → 2.4044, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.5978, mae: 2.6649, huber: 2.2106, swd: 3.2722, ept: 60.7678
    Epoch [2/50], Val Losses: mse: 12.1110, mae: 2.8165, huber: 2.3551, swd: 3.4432, ept: 52.2344
    Epoch [2/50], Test Losses: mse: 13.7814, mae: 2.9192, huber: 2.4599, swd: 3.7669, ept: 53.5069
      Epoch 2 composite train-obj: 2.210566
            Val objective improved 2.4044 → 2.3551, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.4961, mae: 2.6367, huber: 2.1858, swd: 3.1819, ept: 63.2805
    Epoch [3/50], Val Losses: mse: 11.8951, mae: 2.7722, huber: 2.3135, swd: 3.1588, ept: 55.0100
    Epoch [3/50], Test Losses: mse: 13.8457, mae: 2.9111, huber: 2.4544, swd: 3.5056, ept: 56.0048
      Epoch 3 composite train-obj: 2.185759
            Val objective improved 2.3551 → 2.3135, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 11.4057, mae: 2.6175, huber: 2.1686, swd: 3.1062, ept: 64.6597
    Epoch [4/50], Val Losses: mse: 12.0954, mae: 2.7734, huber: 2.3184, swd: 3.1043, ept: 55.6218
    Epoch [4/50], Test Losses: mse: 13.6958, mae: 2.8909, huber: 2.4355, swd: 3.3922, ept: 55.7194
      Epoch 4 composite train-obj: 2.168576
            No improvement (2.3184), counter 1/5
    Epoch [5/50], Train Losses: mse: 11.3689, mae: 2.6082, huber: 2.1606, swd: 3.0693, ept: 65.3361
    Epoch [5/50], Val Losses: mse: 12.3050, mae: 2.8133, huber: 2.3578, swd: 3.1112, ept: 56.2871
    Epoch [5/50], Test Losses: mse: 13.6123, mae: 2.8723, huber: 2.4183, swd: 3.2653, ept: 58.2014
      Epoch 5 composite train-obj: 2.160604
            No improvement (2.3578), counter 2/5
    Epoch [6/50], Train Losses: mse: 11.3005, mae: 2.5956, huber: 2.1493, swd: 3.0446, ept: 66.0336
    Epoch [6/50], Val Losses: mse: 11.6331, mae: 2.7225, huber: 2.2695, swd: 3.0040, ept: 57.0537
    Epoch [6/50], Test Losses: mse: 13.4555, mae: 2.8526, huber: 2.4001, swd: 3.3678, ept: 58.6990
      Epoch 6 composite train-obj: 2.149336
            Val objective improved 2.3135 → 2.2695, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 11.2744, mae: 2.5899, huber: 2.1442, swd: 2.9932, ept: 66.4659
    Epoch [7/50], Val Losses: mse: 11.9450, mae: 2.7631, huber: 2.3103, swd: 3.0321, ept: 55.9597
    Epoch [7/50], Test Losses: mse: 13.6466, mae: 2.8787, huber: 2.4266, swd: 3.3336, ept: 57.7767
      Epoch 7 composite train-obj: 2.144248
            No improvement (2.3103), counter 1/5
    Epoch [8/50], Train Losses: mse: 11.2551, mae: 2.5838, huber: 2.1389, swd: 2.9816, ept: 66.8803
    Epoch [8/50], Val Losses: mse: 11.3544, mae: 2.6860, huber: 2.2343, swd: 2.7879, ept: 57.7905
    Epoch [8/50], Test Losses: mse: 13.5554, mae: 2.8479, huber: 2.3965, swd: 3.2560, ept: 61.1843
      Epoch 8 composite train-obj: 2.138938
            Val objective improved 2.2695 → 2.2343, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 11.2268, mae: 2.5794, huber: 2.1351, swd: 2.9608, ept: 67.2609
    Epoch [9/50], Val Losses: mse: 11.8724, mae: 2.7425, huber: 2.2893, swd: 3.0217, ept: 58.6425
    Epoch [9/50], Test Losses: mse: 13.6068, mae: 2.8608, huber: 2.4094, swd: 3.2046, ept: 59.0553
      Epoch 9 composite train-obj: 2.135080
            No improvement (2.2893), counter 1/5
    Epoch [10/50], Train Losses: mse: 11.1958, mae: 2.5737, huber: 2.1299, swd: 2.9356, ept: 67.5063
    Epoch [10/50], Val Losses: mse: 11.6764, mae: 2.7168, huber: 2.2653, swd: 2.9909, ept: 58.4754
    Epoch [10/50], Test Losses: mse: 13.3792, mae: 2.8363, huber: 2.3856, swd: 3.2477, ept: 59.9434
      Epoch 10 composite train-obj: 2.129872
            No improvement (2.2653), counter 2/5
    Epoch [11/50], Train Losses: mse: 11.1578, mae: 2.5669, huber: 2.1236, swd: 2.9241, ept: 67.7652
    Epoch [11/50], Val Losses: mse: 11.6116, mae: 2.7048, huber: 2.2537, swd: 3.1655, ept: 58.9937
    Epoch [11/50], Test Losses: mse: 13.5509, mae: 2.8523, huber: 2.4020, swd: 3.4220, ept: 59.3688
      Epoch 11 composite train-obj: 2.123606
            No improvement (2.2537), counter 3/5
    Epoch [12/50], Train Losses: mse: 11.1106, mae: 2.5599, huber: 2.1170, swd: 2.9130, ept: 67.9516
    Epoch [12/50], Val Losses: mse: 11.5522, mae: 2.7060, huber: 2.2547, swd: 3.0345, ept: 60.1126
    Epoch [12/50], Test Losses: mse: 13.3934, mae: 2.8286, huber: 2.3795, swd: 3.2675, ept: 60.2405
      Epoch 12 composite train-obj: 2.116960
            No improvement (2.2547), counter 4/5
    Epoch [13/50], Train Losses: mse: 11.1157, mae: 2.5584, huber: 2.1159, swd: 2.8929, ept: 68.1530
    Epoch [13/50], Val Losses: mse: 11.2796, mae: 2.6732, huber: 2.2231, swd: 3.0028, ept: 60.8315
    Epoch [13/50], Test Losses: mse: 13.3860, mae: 2.8392, huber: 2.3891, swd: 3.3477, ept: 60.2212
      Epoch 13 composite train-obj: 2.115929
            Val objective improved 2.2343 → 2.2231, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 11.1008, mae: 2.5570, huber: 2.1146, swd: 2.8836, ept: 68.0958
    Epoch [14/50], Val Losses: mse: 11.6696, mae: 2.7149, huber: 2.2639, swd: 2.7992, ept: 58.1017
    Epoch [14/50], Test Losses: mse: 13.3399, mae: 2.8242, huber: 2.3746, swd: 3.0694, ept: 60.6894
      Epoch 14 composite train-obj: 2.114647
            No improvement (2.2639), counter 1/5
    Epoch [15/50], Train Losses: mse: 11.0816, mae: 2.5528, huber: 2.1108, swd: 2.8457, ept: 68.3409
    Epoch [15/50], Val Losses: mse: 11.3400, mae: 2.6880, huber: 2.2374, swd: 2.9842, ept: 59.7559
    Epoch [15/50], Test Losses: mse: 13.1654, mae: 2.8157, huber: 2.3658, swd: 3.3647, ept: 60.4143
      Epoch 15 composite train-obj: 2.110767
            No improvement (2.2374), counter 2/5
    Epoch [16/50], Train Losses: mse: 11.0608, mae: 2.5500, huber: 2.1082, swd: 2.8474, ept: 68.4496
    Epoch [16/50], Val Losses: mse: 11.4820, mae: 2.6977, huber: 2.2470, swd: 2.9377, ept: 58.7231
    Epoch [16/50], Test Losses: mse: 13.1687, mae: 2.8156, huber: 2.3657, swd: 3.1835, ept: 60.4320
      Epoch 16 composite train-obj: 2.108195
            No improvement (2.2470), counter 3/5
    Epoch [17/50], Train Losses: mse: 11.0479, mae: 2.5469, huber: 2.1054, swd: 2.8326, ept: 68.6737
    Epoch [17/50], Val Losses: mse: 11.9965, mae: 2.7504, huber: 2.2998, swd: 2.8936, ept: 59.6490
    Epoch [17/50], Test Losses: mse: 13.5037, mae: 2.8467, huber: 2.3975, swd: 3.0163, ept: 59.5371
      Epoch 17 composite train-obj: 2.105376
            No improvement (2.2998), counter 4/5
    Epoch [18/50], Train Losses: mse: 11.0196, mae: 2.5440, huber: 2.1025, swd: 2.8140, ept: 68.8049
    Epoch [18/50], Val Losses: mse: 11.6638, mae: 2.7126, huber: 2.2618, swd: 2.8785, ept: 58.6123
    Epoch [18/50], Test Losses: mse: 13.2140, mae: 2.8061, huber: 2.3579, swd: 3.1151, ept: 60.8258
      Epoch 18 composite train-obj: 2.102512
    Epoch [18/50], Test Losses: mse: 13.3860, mae: 2.8392, huber: 2.3891, swd: 3.3477, ept: 60.2212
    Best round's Test MSE: 13.3860, MAE: 2.8392, SWD: 3.3477
    Best round's Validation MSE: 11.2796, MAE: 2.6732, SWD: 3.0028
    Best round's Test verification MSE : 13.3860, MAE: 2.8392, SWD: 3.3477
    Time taken: 23.21 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.6330, mae: 2.8178, huber: 2.3576, swd: 3.2473, ept: 41.7688
    Epoch [1/50], Val Losses: mse: 12.4463, mae: 2.8700, huber: 2.4056, swd: 3.0105, ept: 48.9130
    Epoch [1/50], Test Losses: mse: 14.2629, mae: 2.9919, huber: 2.5282, swd: 3.4191, ept: 50.3034
      Epoch 1 composite train-obj: 2.357566
            Val objective improved inf → 2.4056, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 11.6398, mae: 2.6708, huber: 2.2163, swd: 3.0605, ept: 60.2556
    Epoch [2/50], Val Losses: mse: 12.1252, mae: 2.8010, huber: 2.3404, swd: 2.7821, ept: 52.6294
    Epoch [2/50], Test Losses: mse: 13.9826, mae: 2.9277, huber: 2.4692, swd: 3.0856, ept: 54.3025
      Epoch 2 composite train-obj: 2.216252
            Val objective improved 2.4056 → 2.3404, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 11.4851, mae: 2.6362, huber: 2.1849, swd: 2.9608, ept: 63.2350
    Epoch [3/50], Val Losses: mse: 12.4212, mae: 2.8284, huber: 2.3698, swd: 3.1372, ept: 54.7001
    Epoch [3/50], Test Losses: mse: 13.8043, mae: 2.9131, huber: 2.4566, swd: 3.3344, ept: 55.0673
      Epoch 3 composite train-obj: 2.184932
            No improvement (2.3698), counter 1/5
    Epoch [4/50], Train Losses: mse: 11.4157, mae: 2.6199, huber: 2.1708, swd: 2.9097, ept: 64.4129
    Epoch [4/50], Val Losses: mse: 12.0355, mae: 2.7773, huber: 2.3209, swd: 2.8059, ept: 55.8758
    Epoch [4/50], Test Losses: mse: 13.7620, mae: 2.8882, huber: 2.4335, swd: 3.0437, ept: 56.4640
      Epoch 4 composite train-obj: 2.170813
            Val objective improved 2.3404 → 2.3209, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.3771, mae: 2.6093, huber: 2.1617, swd: 2.8525, ept: 65.4100
    Epoch [5/50], Val Losses: mse: 11.7589, mae: 2.7489, huber: 2.2943, swd: 2.6294, ept: 56.9818
    Epoch [5/50], Test Losses: mse: 13.5394, mae: 2.8617, huber: 2.4086, swd: 2.9662, ept: 57.3536
      Epoch 5 composite train-obj: 2.161715
            Val objective improved 2.3209 → 2.2943, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.3261, mae: 2.5993, huber: 2.1528, swd: 2.8322, ept: 66.1030
    Epoch [6/50], Val Losses: mse: 11.8698, mae: 2.7374, huber: 2.2831, swd: 2.9298, ept: 59.3388
    Epoch [6/50], Test Losses: mse: 13.8278, mae: 2.8965, huber: 2.4427, swd: 3.1833, ept: 57.7477
      Epoch 6 composite train-obj: 2.152796
            Val objective improved 2.2943 → 2.2831, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 11.2953, mae: 2.5913, huber: 2.1457, swd: 2.8020, ept: 66.5231
    Epoch [7/50], Val Losses: mse: 12.1327, mae: 2.7868, huber: 2.3322, swd: 2.9735, ept: 57.1407
    Epoch [7/50], Test Losses: mse: 13.6100, mae: 2.8708, huber: 2.4175, swd: 3.1151, ept: 59.4083
      Epoch 7 composite train-obj: 2.145705
            No improvement (2.3322), counter 1/5
    Epoch [8/50], Train Losses: mse: 11.2461, mae: 2.5845, huber: 2.1394, swd: 2.7840, ept: 66.8507
    Epoch [8/50], Val Losses: mse: 11.5609, mae: 2.7116, huber: 2.2586, swd: 2.7759, ept: 59.9419
    Epoch [8/50], Test Losses: mse: 13.4112, mae: 2.8480, huber: 2.3962, swd: 3.1342, ept: 59.2169
      Epoch 8 composite train-obj: 2.139444
            Val objective improved 2.2831 → 2.2586, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 11.2252, mae: 2.5782, huber: 2.1339, swd: 2.7585, ept: 67.4288
    Epoch [9/50], Val Losses: mse: 11.5044, mae: 2.7061, huber: 2.2551, swd: 2.5998, ept: 59.2241
    Epoch [9/50], Test Losses: mse: 13.3282, mae: 2.8330, huber: 2.3821, swd: 3.0414, ept: 59.5840
      Epoch 9 composite train-obj: 2.133917
            Val objective improved 2.2586 → 2.2551, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 11.2007, mae: 2.5736, huber: 2.1297, swd: 2.7481, ept: 67.5144
    Epoch [10/50], Val Losses: mse: 11.5333, mae: 2.7010, huber: 2.2507, swd: 2.5976, ept: 57.0603
    Epoch [10/50], Test Losses: mse: 13.3132, mae: 2.8239, huber: 2.3741, swd: 2.9205, ept: 59.1412
      Epoch 10 composite train-obj: 2.129723
            Val objective improved 2.2551 → 2.2507, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 11.1757, mae: 2.5696, huber: 2.1262, swd: 2.7314, ept: 67.4490
    Epoch [11/50], Val Losses: mse: 11.7085, mae: 2.7211, huber: 2.2705, swd: 2.5232, ept: 58.8406
    Epoch [11/50], Test Losses: mse: 13.3675, mae: 2.8357, huber: 2.3849, swd: 2.8768, ept: 60.2803
      Epoch 11 composite train-obj: 2.126191
            No improvement (2.2705), counter 1/5
    Epoch [12/50], Train Losses: mse: 11.1482, mae: 2.5644, huber: 2.1214, swd: 2.7050, ept: 67.9562
    Epoch [12/50], Val Losses: mse: 11.6481, mae: 2.7201, huber: 2.2698, swd: 2.4768, ept: 57.8252
    Epoch [12/50], Test Losses: mse: 13.3426, mae: 2.8249, huber: 2.3751, swd: 2.7507, ept: 59.7932
      Epoch 12 composite train-obj: 2.121422
            No improvement (2.2698), counter 2/5
    Epoch [13/50], Train Losses: mse: 11.1212, mae: 2.5607, huber: 2.1180, swd: 2.6995, ept: 68.0533
    Epoch [13/50], Val Losses: mse: 11.6469, mae: 2.7050, huber: 2.2555, swd: 2.5368, ept: 59.5367
    Epoch [13/50], Test Losses: mse: 13.3123, mae: 2.8191, huber: 2.3702, swd: 2.8932, ept: 61.3680
      Epoch 13 composite train-obj: 2.118017
            No improvement (2.2555), counter 3/5
    Epoch [14/50], Train Losses: mse: 11.0944, mae: 2.5553, huber: 2.1130, swd: 2.6845, ept: 68.2721
    Epoch [14/50], Val Losses: mse: 11.4059, mae: 2.6763, huber: 2.2262, swd: 2.6920, ept: 63.1447
    Epoch [14/50], Test Losses: mse: 13.3656, mae: 2.8269, huber: 2.3776, swd: 2.9562, ept: 59.6131
      Epoch 14 composite train-obj: 2.113030
            Val objective improved 2.2507 → 2.2262, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 11.0669, mae: 2.5518, huber: 2.1098, swd: 2.6786, ept: 68.4026
    Epoch [15/50], Val Losses: mse: 11.4160, mae: 2.6847, huber: 2.2337, swd: 2.4808, ept: 59.2094
    Epoch [15/50], Test Losses: mse: 13.3810, mae: 2.8379, huber: 2.3879, swd: 2.8035, ept: 59.9337
      Epoch 15 composite train-obj: 2.109784
            No improvement (2.2337), counter 1/5
    Epoch [16/50], Train Losses: mse: 11.0886, mae: 2.5540, huber: 2.1121, swd: 2.6580, ept: 68.4134
    Epoch [16/50], Val Losses: mse: 11.4586, mae: 2.6912, huber: 2.2410, swd: 2.5031, ept: 58.9321
    Epoch [16/50], Test Losses: mse: 13.2266, mae: 2.8066, huber: 2.3580, swd: 2.7521, ept: 60.6753
      Epoch 16 composite train-obj: 2.112113
            No improvement (2.2410), counter 2/5
    Epoch [17/50], Train Losses: mse: 11.0573, mae: 2.5492, huber: 2.1074, swd: 2.6333, ept: 68.5864
    Epoch [17/50], Val Losses: mse: 11.4111, mae: 2.6703, huber: 2.2204, swd: 2.6946, ept: 61.9494
    Epoch [17/50], Test Losses: mse: 13.4459, mae: 2.8397, huber: 2.3903, swd: 2.9308, ept: 59.9132
      Epoch 17 composite train-obj: 2.107432
            Val objective improved 2.2262 → 2.2204, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 11.0446, mae: 2.5467, huber: 2.1052, swd: 2.6339, ept: 68.7446
    Epoch [18/50], Val Losses: mse: 11.3128, mae: 2.6572, huber: 2.2084, swd: 2.4819, ept: 60.6489
    Epoch [18/50], Test Losses: mse: 13.3852, mae: 2.8275, huber: 2.3782, swd: 2.8893, ept: 60.8648
      Epoch 18 composite train-obj: 2.105188
            Val objective improved 2.2204 → 2.2084, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 11.0092, mae: 2.5416, huber: 2.1005, swd: 2.6293, ept: 68.7936
    Epoch [19/50], Val Losses: mse: 11.2615, mae: 2.6659, huber: 2.2169, swd: 2.6101, ept: 59.1727
    Epoch [19/50], Test Losses: mse: 13.2916, mae: 2.8218, huber: 2.3735, swd: 2.8862, ept: 61.0018
      Epoch 19 composite train-obj: 2.100481
            No improvement (2.2169), counter 1/5
    Epoch [20/50], Train Losses: mse: 11.0118, mae: 2.5416, huber: 2.1006, swd: 2.6288, ept: 68.9299
    Epoch [20/50], Val Losses: mse: 11.2988, mae: 2.6645, huber: 2.2154, swd: 2.4207, ept: 62.2459
    Epoch [20/50], Test Losses: mse: 13.2090, mae: 2.8037, huber: 2.3556, swd: 2.6987, ept: 61.5723
      Epoch 20 composite train-obj: 2.100578
            No improvement (2.2154), counter 2/5
    Epoch [21/50], Train Losses: mse: 11.0033, mae: 2.5382, huber: 2.0975, swd: 2.5937, ept: 69.2819
    Epoch [21/50], Val Losses: mse: 11.0138, mae: 2.6244, huber: 2.1764, swd: 2.5500, ept: 63.4403
    Epoch [21/50], Test Losses: mse: 13.2572, mae: 2.8143, huber: 2.3666, swd: 2.9476, ept: 61.5776
      Epoch 21 composite train-obj: 2.097497
            Val objective improved 2.2084 → 2.1764, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 10.9924, mae: 2.5385, huber: 2.0976, swd: 2.5912, ept: 69.2610
    Epoch [22/50], Val Losses: mse: 10.9523, mae: 2.6137, huber: 2.1676, swd: 2.4934, ept: 62.8368
    Epoch [22/50], Test Losses: mse: 13.2603, mae: 2.8098, huber: 2.3620, swd: 2.8845, ept: 61.2387
      Epoch 22 composite train-obj: 2.097648
            Val objective improved 2.1764 → 2.1676, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 10.9702, mae: 2.5344, huber: 2.0939, swd: 2.5918, ept: 69.1214
    Epoch [23/50], Val Losses: mse: 11.1843, mae: 2.6452, huber: 2.1981, swd: 2.4360, ept: 61.3794
    Epoch [23/50], Test Losses: mse: 13.1689, mae: 2.7919, huber: 2.3456, swd: 2.6920, ept: 62.2041
      Epoch 23 composite train-obj: 2.093933
            No improvement (2.1981), counter 1/5
    Epoch [24/50], Train Losses: mse: 10.9355, mae: 2.5292, huber: 2.0889, swd: 2.5751, ept: 69.5246
    Epoch [24/50], Val Losses: mse: 10.9199, mae: 2.6018, huber: 2.1558, swd: 2.4626, ept: 63.3907
    Epoch [24/50], Test Losses: mse: 13.4237, mae: 2.8328, huber: 2.3844, swd: 2.8442, ept: 60.5280
      Epoch 24 composite train-obj: 2.088883
            Val objective improved 2.1676 → 2.1558, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 10.9305, mae: 2.5285, huber: 2.0884, swd: 2.5703, ept: 69.5358
    Epoch [25/50], Val Losses: mse: 11.0291, mae: 2.6257, huber: 2.1785, swd: 2.6057, ept: 63.9396
    Epoch [25/50], Test Losses: mse: 13.2170, mae: 2.8107, huber: 2.3629, swd: 2.9664, ept: 61.1336
      Epoch 25 composite train-obj: 2.088363
            No improvement (2.1785), counter 1/5
    Epoch [26/50], Train Losses: mse: 10.9323, mae: 2.5281, huber: 2.0880, swd: 2.5630, ept: 69.5897
    Epoch [26/50], Val Losses: mse: 11.1522, mae: 2.6403, huber: 2.1934, swd: 2.5878, ept: 60.9508
    Epoch [26/50], Test Losses: mse: 13.1264, mae: 2.7987, huber: 2.3509, swd: 2.8398, ept: 61.1168
      Epoch 26 composite train-obj: 2.087998
            No improvement (2.1934), counter 2/5
    Epoch [27/50], Train Losses: mse: 10.9374, mae: 2.5282, huber: 2.0882, swd: 2.5482, ept: 69.7045
    Epoch [27/50], Val Losses: mse: 10.8796, mae: 2.6040, huber: 2.1575, swd: 2.4218, ept: 61.6923
    Epoch [27/50], Test Losses: mse: 13.1829, mae: 2.8056, huber: 2.3583, swd: 2.7636, ept: 60.8604
      Epoch 27 composite train-obj: 2.088226
            No improvement (2.1575), counter 3/5
    Epoch [28/50], Train Losses: mse: 10.9266, mae: 2.5261, huber: 2.0863, swd: 2.5297, ept: 69.4465
    Epoch [28/50], Val Losses: mse: 10.9868, mae: 2.6215, huber: 2.1735, swd: 2.6203, ept: 62.5588
    Epoch [28/50], Test Losses: mse: 13.2692, mae: 2.8163, huber: 2.3689, swd: 2.8518, ept: 61.9630
      Epoch 28 composite train-obj: 2.086279
            No improvement (2.1735), counter 4/5
    Epoch [29/50], Train Losses: mse: 10.9121, mae: 2.5241, huber: 2.0845, swd: 2.5380, ept: 69.5447
    Epoch [29/50], Val Losses: mse: 11.2604, mae: 2.6633, huber: 2.2147, swd: 2.4983, ept: 61.7539
    Epoch [29/50], Test Losses: mse: 13.1475, mae: 2.8057, huber: 2.3587, swd: 2.7604, ept: 61.1572
      Epoch 29 composite train-obj: 2.084474
    Epoch [29/50], Test Losses: mse: 13.4237, mae: 2.8328, huber: 2.3844, swd: 2.8442, ept: 60.5280
    Best round's Test MSE: 13.4237, MAE: 2.8328, SWD: 2.8442
    Best round's Validation MSE: 10.9199, MAE: 2.6018, SWD: 2.4626
    Best round's Test verification MSE : 13.4237, MAE: 2.8328, SWD: 2.8442
    Time taken: 37.40 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz96_seq720_pred196_20250512_2154)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 13.4831 ± 0.1117
      mae: 2.8479 ± 0.0170
      huber: 2.3973 ± 0.0151
      swd: 3.1621 ± 0.2258
      ept: 59.3522 ± 1.4514
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 11.2631 ± 0.2737
      mae: 2.6642 ± 0.0478
      huber: 2.2141 ± 0.0444
      swd: 2.7643 ± 0.2250
      ept: 61.0259 ± 1.8566
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 74.60 seconds
    
    Experiment complete: DLinear_lorenz96_seq720_pred196_20250512_2154
    Model: DLinear
    Dataset: lorenz96
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
    channels=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([336, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([336, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 96
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 336, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 96
    Validation Batches: 7
    Test Batches: 22
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.0045, mae: 2.8673, huber: 2.4061, swd: 3.0616, ept: 42.4197
    Epoch [1/50], Val Losses: mse: 13.1664, mae: 2.9592, huber: 2.4933, swd: 2.8927, ept: 44.9551
    Epoch [1/50], Test Losses: mse: 15.2621, mae: 3.1104, huber: 2.6455, swd: 3.4084, ept: 49.8490
      Epoch 1 composite train-obj: 2.406107
            Val objective improved inf → 2.4933, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.1568, mae: 2.7473, huber: 2.2904, swd: 2.9684, ept: 62.0465
    Epoch [2/50], Val Losses: mse: 13.1412, mae: 2.9414, huber: 2.4771, swd: 2.8301, ept: 47.7344
    Epoch [2/50], Test Losses: mse: 15.0743, mae: 3.0726, huber: 2.6100, swd: 3.2048, ept: 53.9234
      Epoch 2 composite train-obj: 2.290360
            Val objective improved 2.4933 → 2.4771, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.0771, mae: 2.7275, huber: 2.2726, swd: 2.8867, ept: 64.8854
    Epoch [3/50], Val Losses: mse: 13.2524, mae: 2.9498, huber: 2.4870, swd: 2.6889, ept: 50.9780
    Epoch [3/50], Test Losses: mse: 15.0690, mae: 3.0704, huber: 2.6089, swd: 3.1114, ept: 55.5041
      Epoch 3 composite train-obj: 2.272609
            No improvement (2.4870), counter 1/5
    Epoch [4/50], Train Losses: mse: 12.0382, mae: 2.7177, huber: 2.2640, swd: 2.8495, ept: 66.1910
    Epoch [4/50], Val Losses: mse: 13.2799, mae: 2.9497, huber: 2.4883, swd: 2.8523, ept: 49.9588
    Epoch [4/50], Test Losses: mse: 14.8130, mae: 3.0475, huber: 2.5861, swd: 3.1946, ept: 56.5495
      Epoch 4 composite train-obj: 2.264023
            No improvement (2.4883), counter 2/5
    Epoch [5/50], Train Losses: mse: 11.9884, mae: 2.7087, huber: 2.2560, swd: 2.8202, ept: 67.1246
    Epoch [5/50], Val Losses: mse: 12.8942, mae: 2.9027, huber: 2.4419, swd: 2.6510, ept: 52.5521
    Epoch [5/50], Test Losses: mse: 15.0766, mae: 3.0617, huber: 2.6018, swd: 3.0508, ept: 57.2302
      Epoch 5 composite train-obj: 2.255993
            Val objective improved 2.4771 → 2.4419, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 11.9596, mae: 2.7015, huber: 2.2495, swd: 2.7913, ept: 68.0020
    Epoch [6/50], Val Losses: mse: 13.1075, mae: 2.9233, huber: 2.4627, swd: 2.7076, ept: 55.8358
    Epoch [6/50], Test Losses: mse: 14.9287, mae: 3.0577, huber: 2.5969, swd: 3.1265, ept: 57.4252
      Epoch 6 composite train-obj: 2.249495
            No improvement (2.4627), counter 1/5
    Epoch [7/50], Train Losses: mse: 11.9162, mae: 2.6951, huber: 2.2437, swd: 2.7695, ept: 68.4071
    Epoch [7/50], Val Losses: mse: 12.9899, mae: 2.9069, huber: 2.4467, swd: 2.6847, ept: 54.5475
    Epoch [7/50], Test Losses: mse: 14.9476, mae: 3.0477, huber: 2.5892, swd: 3.0304, ept: 56.7432
      Epoch 7 composite train-obj: 2.243673
            No improvement (2.4467), counter 2/5
    Epoch [8/50], Train Losses: mse: 11.9204, mae: 2.6936, huber: 2.2425, swd: 2.7473, ept: 68.6099
    Epoch [8/50], Val Losses: mse: 12.9400, mae: 2.9073, huber: 2.4477, swd: 2.7284, ept: 53.3376
    Epoch [8/50], Test Losses: mse: 14.7612, mae: 3.0341, huber: 2.5746, swd: 3.1354, ept: 58.8041
      Epoch 8 composite train-obj: 2.242497
            No improvement (2.4477), counter 3/5
    Epoch [9/50], Train Losses: mse: 11.8859, mae: 2.6889, huber: 2.2381, swd: 2.7344, ept: 69.1089
    Epoch [9/50], Val Losses: mse: 12.8691, mae: 2.8959, huber: 2.4361, swd: 2.6312, ept: 54.9277
    Epoch [9/50], Test Losses: mse: 14.7756, mae: 3.0276, huber: 2.5690, swd: 3.0235, ept: 59.7113
      Epoch 9 composite train-obj: 2.238108
            Val objective improved 2.4419 → 2.4361, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 11.8720, mae: 2.6861, huber: 2.2354, swd: 2.7113, ept: 69.1117
    Epoch [10/50], Val Losses: mse: 12.5713, mae: 2.8604, huber: 2.4012, swd: 2.6922, ept: 55.4975
    Epoch [10/50], Test Losses: mse: 14.6891, mae: 3.0206, huber: 2.5618, swd: 3.1147, ept: 59.8665
      Epoch 10 composite train-obj: 2.235431
            Val objective improved 2.4361 → 2.4012, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 11.8496, mae: 2.6818, huber: 2.2317, swd: 2.7021, ept: 69.6849
    Epoch [11/50], Val Losses: mse: 12.8291, mae: 2.8883, huber: 2.4290, swd: 2.8657, ept: 54.8418
    Epoch [11/50], Test Losses: mse: 14.7733, mae: 3.0322, huber: 2.5740, swd: 3.1315, ept: 58.5245
      Epoch 11 composite train-obj: 2.231690
            No improvement (2.4290), counter 1/5
    Epoch [12/50], Train Losses: mse: 11.8395, mae: 2.6808, huber: 2.2307, swd: 2.6766, ept: 69.7220
    Epoch [12/50], Val Losses: mse: 12.7614, mae: 2.8767, huber: 2.4176, swd: 2.6711, ept: 55.8532
    Epoch [12/50], Test Losses: mse: 14.9069, mae: 3.0423, huber: 2.5842, swd: 3.0662, ept: 58.7614
      Epoch 12 composite train-obj: 2.230710
            No improvement (2.4176), counter 2/5
    Epoch [13/50], Train Losses: mse: 11.7984, mae: 2.6748, huber: 2.2250, swd: 2.6748, ept: 69.8856
    Epoch [13/50], Val Losses: mse: 12.6474, mae: 2.8770, huber: 2.4181, swd: 2.8439, ept: 56.0085
    Epoch [13/50], Test Losses: mse: 14.5728, mae: 3.0154, huber: 2.5573, swd: 3.1860, ept: 59.6513
      Epoch 13 composite train-obj: 2.225046
            No improvement (2.4181), counter 3/5
    Epoch [14/50], Train Losses: mse: 11.7942, mae: 2.6727, huber: 2.2231, swd: 2.6552, ept: 70.0119
    Epoch [14/50], Val Losses: mse: 12.4414, mae: 2.8385, huber: 2.3807, swd: 2.6164, ept: 56.5639
    Epoch [14/50], Test Losses: mse: 14.6820, mae: 3.0228, huber: 2.5647, swd: 3.0651, ept: 59.4271
      Epoch 14 composite train-obj: 2.223090
            Val objective improved 2.4012 → 2.3807, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 11.7727, mae: 2.6687, huber: 2.2195, swd: 2.6405, ept: 70.4138
    Epoch [15/50], Val Losses: mse: 12.6179, mae: 2.8723, huber: 2.4135, swd: 2.7797, ept: 58.4820
    Epoch [15/50], Test Losses: mse: 14.6257, mae: 3.0132, huber: 2.5558, swd: 3.0977, ept: 60.7775
      Epoch 15 composite train-obj: 2.219525
            No improvement (2.4135), counter 1/5
    Epoch [16/50], Train Losses: mse: 11.7592, mae: 2.6679, huber: 2.2186, swd: 2.6358, ept: 70.3175
    Epoch [16/50], Val Losses: mse: 12.4928, mae: 2.8573, huber: 2.3987, swd: 2.6602, ept: 57.3362
    Epoch [16/50], Test Losses: mse: 14.5040, mae: 2.9960, huber: 2.5392, swd: 2.9699, ept: 61.1491
      Epoch 16 composite train-obj: 2.218649
            No improvement (2.3987), counter 2/5
    Epoch [17/50], Train Losses: mse: 11.7401, mae: 2.6640, huber: 2.2150, swd: 2.6185, ept: 70.8239
    Epoch [17/50], Val Losses: mse: 12.3878, mae: 2.8302, huber: 2.3734, swd: 2.5551, ept: 58.3582
    Epoch [17/50], Test Losses: mse: 14.6406, mae: 3.0095, huber: 2.5518, swd: 2.9513, ept: 59.6610
      Epoch 17 composite train-obj: 2.215008
            Val objective improved 2.3807 → 2.3734, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 11.7307, mae: 2.6631, huber: 2.2143, swd: 2.6055, ept: 70.6579
    Epoch [18/50], Val Losses: mse: 12.5462, mae: 2.8605, huber: 2.4018, swd: 2.7096, ept: 56.5356
    Epoch [18/50], Test Losses: mse: 14.5824, mae: 3.0104, huber: 2.5528, swd: 3.0164, ept: 60.8419
      Epoch 18 composite train-obj: 2.214292
            No improvement (2.4018), counter 1/5
    Epoch [19/50], Train Losses: mse: 11.7209, mae: 2.6612, huber: 2.2124, swd: 2.6002, ept: 70.5379
    Epoch [19/50], Val Losses: mse: 12.8579, mae: 2.8837, huber: 2.4252, swd: 2.5387, ept: 56.4822
    Epoch [19/50], Test Losses: mse: 14.6745, mae: 3.0135, huber: 2.5562, swd: 2.7963, ept: 60.2275
      Epoch 19 composite train-obj: 2.212418
            No improvement (2.4252), counter 2/5
    Epoch [20/50], Train Losses: mse: 11.7171, mae: 2.6608, huber: 2.2120, swd: 2.5804, ept: 71.1097
    Epoch [20/50], Val Losses: mse: 12.7090, mae: 2.8656, huber: 2.4082, swd: 2.4813, ept: 60.8373
    Epoch [20/50], Test Losses: mse: 14.6500, mae: 3.0124, huber: 2.5549, swd: 2.8306, ept: 60.8684
      Epoch 20 composite train-obj: 2.211988
            No improvement (2.4082), counter 3/5
    Epoch [21/50], Train Losses: mse: 11.6884, mae: 2.6567, huber: 2.2082, swd: 2.5808, ept: 70.9695
    Epoch [21/50], Val Losses: mse: 12.3879, mae: 2.8371, huber: 2.3789, swd: 2.6055, ept: 57.2664
    Epoch [21/50], Test Losses: mse: 14.6379, mae: 3.0174, huber: 2.5597, swd: 2.9533, ept: 60.8836
      Epoch 21 composite train-obj: 2.208240
            No improvement (2.3789), counter 4/5
    Epoch [22/50], Train Losses: mse: 11.6975, mae: 2.6578, huber: 2.2093, swd: 2.5693, ept: 71.2353
    Epoch [22/50], Val Losses: mse: 12.5647, mae: 2.8444, huber: 2.3881, swd: 2.5236, ept: 58.6247
    Epoch [22/50], Test Losses: mse: 14.7356, mae: 3.0231, huber: 2.5660, swd: 2.7645, ept: 60.0463
      Epoch 22 composite train-obj: 2.209316
    Epoch [22/50], Test Losses: mse: 14.6406, mae: 3.0095, huber: 2.5518, swd: 2.9513, ept: 59.6610
    Best round's Test MSE: 14.6406, MAE: 3.0095, SWD: 2.9513
    Best round's Validation MSE: 12.3878, MAE: 2.8302, SWD: 2.5551
    Best round's Test verification MSE : 14.6406, MAE: 3.0095, SWD: 2.9513
    Time taken: 29.47 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.9791, mae: 2.8641, huber: 2.4030, swd: 3.0741, ept: 42.9843
    Epoch [1/50], Val Losses: mse: 13.2763, mae: 2.9893, huber: 2.5213, swd: 2.9693, ept: 44.4413
    Epoch [1/50], Test Losses: mse: 15.1386, mae: 3.0993, huber: 2.6343, swd: 3.4608, ept: 48.7529
      Epoch 1 composite train-obj: 2.403025
            Val objective improved inf → 2.5213, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.1813, mae: 2.7496, huber: 2.2926, swd: 2.9395, ept: 61.7326
    Epoch [2/50], Val Losses: mse: 13.1520, mae: 2.9499, huber: 2.4852, swd: 3.0189, ept: 48.9019
    Epoch [2/50], Test Losses: mse: 14.9686, mae: 3.0736, huber: 2.6106, swd: 3.4823, ept: 53.3065
      Epoch 2 composite train-obj: 2.292576
            Val objective improved 2.5213 → 2.4852, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.0870, mae: 2.7290, huber: 2.2741, swd: 2.8987, ept: 64.9518
    Epoch [3/50], Val Losses: mse: 13.2348, mae: 2.9480, huber: 2.4851, swd: 2.9211, ept: 51.3775
    Epoch [3/50], Test Losses: mse: 14.8887, mae: 3.0575, huber: 2.5956, swd: 3.2509, ept: 54.8443
      Epoch 3 composite train-obj: 2.274052
            Val objective improved 2.4852 → 2.4851, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.0068, mae: 2.7140, huber: 2.2605, swd: 2.8572, ept: 66.5699
    Epoch [4/50], Val Losses: mse: 13.0582, mae: 2.9294, huber: 2.4676, swd: 2.7333, ept: 51.4414
    Epoch [4/50], Test Losses: mse: 14.9779, mae: 3.0589, huber: 2.5987, swd: 3.2649, ept: 56.1489
      Epoch 4 composite train-obj: 2.260486
            Val objective improved 2.4851 → 2.4676, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.9837, mae: 2.7083, huber: 2.2555, swd: 2.8191, ept: 67.3227
    Epoch [5/50], Val Losses: mse: 13.1467, mae: 2.9323, huber: 2.4710, swd: 2.8646, ept: 53.2722
    Epoch [5/50], Test Losses: mse: 15.1028, mae: 3.0732, huber: 2.6128, swd: 3.1215, ept: 56.9152
      Epoch 5 composite train-obj: 2.255524
            No improvement (2.4710), counter 1/5
    Epoch [6/50], Train Losses: mse: 11.9534, mae: 2.7011, huber: 2.2490, swd: 2.7826, ept: 67.8202
    Epoch [6/50], Val Losses: mse: 13.3472, mae: 2.9482, huber: 2.4872, swd: 2.8423, ept: 53.8956
    Epoch [6/50], Test Losses: mse: 14.7057, mae: 3.0359, huber: 2.5756, swd: 3.1352, ept: 57.7618
      Epoch 6 composite train-obj: 2.248993
            No improvement (2.4872), counter 2/5
    Epoch [7/50], Train Losses: mse: 11.9293, mae: 2.6970, huber: 2.2454, swd: 2.7672, ept: 68.1800
    Epoch [7/50], Val Losses: mse: 13.0725, mae: 2.9227, huber: 2.4622, swd: 2.7803, ept: 52.5897
    Epoch [7/50], Test Losses: mse: 14.9494, mae: 3.0532, huber: 2.5928, swd: 3.0644, ept: 57.7616
      Epoch 7 composite train-obj: 2.245374
            Val objective improved 2.4676 → 2.4622, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 11.9097, mae: 2.6927, huber: 2.2417, swd: 2.7536, ept: 68.6339
    Epoch [8/50], Val Losses: mse: 12.7423, mae: 2.8799, huber: 2.4205, swd: 2.6478, ept: 54.5454
    Epoch [8/50], Test Losses: mse: 14.8169, mae: 3.0364, huber: 2.5776, swd: 3.0506, ept: 58.3802
      Epoch 8 composite train-obj: 2.241663
            Val objective improved 2.4622 → 2.4205, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 11.8764, mae: 2.6867, huber: 2.2361, swd: 2.7283, ept: 68.9593
    Epoch [9/50], Val Losses: mse: 12.8881, mae: 2.8904, huber: 2.4315, swd: 2.7786, ept: 55.3005
    Epoch [9/50], Test Losses: mse: 14.8492, mae: 3.0340, huber: 2.5756, swd: 3.0979, ept: 57.9369
      Epoch 9 composite train-obj: 2.236141
            No improvement (2.4315), counter 1/5
    Epoch [10/50], Train Losses: mse: 11.8673, mae: 2.6852, huber: 2.2347, swd: 2.7114, ept: 69.2739
    Epoch [10/50], Val Losses: mse: 12.6845, mae: 2.8679, huber: 2.4088, swd: 2.7879, ept: 55.9796
    Epoch [10/50], Test Losses: mse: 15.0036, mae: 3.0631, huber: 2.6034, swd: 3.2330, ept: 58.1486
      Epoch 10 composite train-obj: 2.234744
            Val objective improved 2.4205 → 2.4088, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 11.8598, mae: 2.6832, huber: 2.2329, swd: 2.6907, ept: 69.6906
    Epoch [11/50], Val Losses: mse: 12.8656, mae: 2.8932, huber: 2.4340, swd: 2.6819, ept: 54.8414
    Epoch [11/50], Test Losses: mse: 14.7630, mae: 3.0297, huber: 2.5707, swd: 3.0034, ept: 58.4889
      Epoch 11 composite train-obj: 2.232938
            No improvement (2.4340), counter 1/5
    Epoch [12/50], Train Losses: mse: 11.8255, mae: 2.6784, huber: 2.2285, swd: 2.6715, ept: 69.5297
    Epoch [12/50], Val Losses: mse: 12.6056, mae: 2.8618, huber: 2.4019, swd: 2.6570, ept: 55.5267
    Epoch [12/50], Test Losses: mse: 14.7984, mae: 3.0251, huber: 2.5676, swd: 3.0137, ept: 59.7294
      Epoch 12 composite train-obj: 2.228462
            Val objective improved 2.4088 → 2.4019, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 11.8253, mae: 2.6781, huber: 2.2282, swd: 2.6633, ept: 69.8219
    Epoch [13/50], Val Losses: mse: 12.6892, mae: 2.8803, huber: 2.4211, swd: 2.7122, ept: 54.0435
    Epoch [13/50], Test Losses: mse: 14.7793, mae: 3.0304, huber: 2.5724, swd: 3.0260, ept: 59.5214
      Epoch 13 composite train-obj: 2.228204
            No improvement (2.4211), counter 1/5
    Epoch [14/50], Train Losses: mse: 11.7886, mae: 2.6723, huber: 2.2228, swd: 2.6556, ept: 70.1243
    Epoch [14/50], Val Losses: mse: 12.6760, mae: 2.8766, huber: 2.4177, swd: 2.6464, ept: 55.0734
    Epoch [14/50], Test Losses: mse: 14.6082, mae: 3.0127, huber: 2.5547, swd: 2.9173, ept: 60.2606
      Epoch 14 composite train-obj: 2.222845
            No improvement (2.4177), counter 2/5
    Epoch [15/50], Train Losses: mse: 11.7685, mae: 2.6695, huber: 2.2202, swd: 2.6406, ept: 70.2085
    Epoch [15/50], Val Losses: mse: 12.5232, mae: 2.8473, huber: 2.3895, swd: 2.5704, ept: 57.8288
    Epoch [15/50], Test Losses: mse: 14.8335, mae: 3.0363, huber: 2.5783, swd: 2.9741, ept: 60.1779
      Epoch 15 composite train-obj: 2.220243
            Val objective improved 2.4019 → 2.3895, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 11.7654, mae: 2.6674, huber: 2.2183, swd: 2.6167, ept: 70.4494
    Epoch [16/50], Val Losses: mse: 12.8164, mae: 2.8848, huber: 2.4262, swd: 2.6782, ept: 55.7473
    Epoch [16/50], Test Losses: mse: 14.6086, mae: 3.0053, huber: 2.5479, swd: 3.0159, ept: 59.6600
      Epoch 16 composite train-obj: 2.218329
            No improvement (2.4262), counter 1/5
    Epoch [17/50], Train Losses: mse: 11.7559, mae: 2.6661, huber: 2.2171, swd: 2.6059, ept: 70.4708
    Epoch [17/50], Val Losses: mse: 12.5987, mae: 2.8635, huber: 2.4053, swd: 2.6562, ept: 55.6189
    Epoch [17/50], Test Losses: mse: 14.5928, mae: 3.0076, huber: 2.5503, swd: 2.9962, ept: 60.4285
      Epoch 17 composite train-obj: 2.217051
            No improvement (2.4053), counter 2/5
    Epoch [18/50], Train Losses: mse: 11.7288, mae: 2.6625, huber: 2.2137, swd: 2.6096, ept: 70.8067
    Epoch [18/50], Val Losses: mse: 12.5558, mae: 2.8434, huber: 2.3867, swd: 2.5647, ept: 58.1022
    Epoch [18/50], Test Losses: mse: 14.6682, mae: 3.0129, huber: 2.5559, swd: 2.9516, ept: 60.0178
      Epoch 18 composite train-obj: 2.213693
            Val objective improved 2.3895 → 2.3867, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 11.7256, mae: 2.6626, huber: 2.2136, swd: 2.5951, ept: 70.5800
    Epoch [19/50], Val Losses: mse: 12.3573, mae: 2.8290, huber: 2.3707, swd: 2.6449, ept: 55.9617
    Epoch [19/50], Test Losses: mse: 14.7075, mae: 3.0188, huber: 2.5613, swd: 2.9711, ept: 61.4329
      Epoch 19 composite train-obj: 2.213563
            Val objective improved 2.3867 → 2.3707, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 11.6954, mae: 2.6580, huber: 2.2093, swd: 2.5773, ept: 71.0853
    Epoch [20/50], Val Losses: mse: 12.4663, mae: 2.8447, huber: 2.3867, swd: 2.6079, ept: 56.9763
    Epoch [20/50], Test Losses: mse: 14.6300, mae: 3.0094, huber: 2.5520, swd: 2.8353, ept: 59.8291
      Epoch 20 composite train-obj: 2.209265
            No improvement (2.3867), counter 1/5
    Epoch [21/50], Train Losses: mse: 11.7151, mae: 2.6588, huber: 2.2104, swd: 2.5633, ept: 71.0243
    Epoch [21/50], Val Losses: mse: 12.7033, mae: 2.8759, huber: 2.4181, swd: 2.5590, ept: 55.9656
    Epoch [21/50], Test Losses: mse: 14.5963, mae: 3.0038, huber: 2.5471, swd: 2.8725, ept: 61.1380
      Epoch 21 composite train-obj: 2.210388
            No improvement (2.4181), counter 2/5
    Epoch [22/50], Train Losses: mse: 11.6807, mae: 2.6554, huber: 2.2070, swd: 2.5575, ept: 71.0435
    Epoch [22/50], Val Losses: mse: 12.4291, mae: 2.8320, huber: 2.3742, swd: 2.7597, ept: 57.8341
    Epoch [22/50], Test Losses: mse: 14.7488, mae: 3.0259, huber: 2.5686, swd: 2.9846, ept: 59.9691
      Epoch 22 composite train-obj: 2.207000
            No improvement (2.3742), counter 3/5
    Epoch [23/50], Train Losses: mse: 11.6624, mae: 2.6524, huber: 2.2042, swd: 2.5466, ept: 71.3029
    Epoch [23/50], Val Losses: mse: 12.5032, mae: 2.8459, huber: 2.3884, swd: 2.7593, ept: 57.9167
    Epoch [23/50], Test Losses: mse: 14.6456, mae: 3.0106, huber: 2.5541, swd: 2.9858, ept: 60.7832
      Epoch 23 composite train-obj: 2.204154
            No improvement (2.3884), counter 4/5
    Epoch [24/50], Train Losses: mse: 11.6535, mae: 2.6511, huber: 2.2030, swd: 2.5361, ept: 71.3554
    Epoch [24/50], Val Losses: mse: 12.6524, mae: 2.8574, huber: 2.4001, swd: 2.8895, ept: 55.1058
    Epoch [24/50], Test Losses: mse: 14.9206, mae: 3.0444, huber: 2.5870, swd: 3.1096, ept: 60.1086
      Epoch 24 composite train-obj: 2.202958
    Epoch [24/50], Test Losses: mse: 14.7075, mae: 3.0188, huber: 2.5613, swd: 2.9711, ept: 61.4329
    Best round's Test MSE: 14.7075, MAE: 3.0188, SWD: 2.9711
    Best round's Validation MSE: 12.3573, MAE: 2.8290, SWD: 2.6449
    Best round's Test verification MSE : 14.7075, MAE: 3.0188, SWD: 2.9711
    Time taken: 31.79 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 12.9872, mae: 2.8652, huber: 2.4040, swd: 3.0722, ept: 42.6229
    Epoch [1/50], Val Losses: mse: 13.2659, mae: 2.9735, huber: 2.5070, swd: 2.7276, ept: 44.2627
    Epoch [1/50], Test Losses: mse: 15.3329, mae: 3.1201, huber: 2.6549, swd: 3.2637, ept: 49.4618
      Epoch 1 composite train-obj: 2.404043
            Val objective improved inf → 2.5070, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.1537, mae: 2.7472, huber: 2.2903, swd: 2.9601, ept: 61.8401
    Epoch [2/50], Val Losses: mse: 13.1423, mae: 2.9394, huber: 2.4750, swd: 2.8408, ept: 48.2981
    Epoch [2/50], Test Losses: mse: 15.1531, mae: 3.0889, huber: 2.6257, swd: 3.2553, ept: 52.6300
      Epoch 2 composite train-obj: 2.290260
            Val objective improved 2.5070 → 2.4750, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.0934, mae: 2.7298, huber: 2.2748, swd: 2.8721, ept: 64.6723
    Epoch [3/50], Val Losses: mse: 13.3512, mae: 2.9661, huber: 2.5024, swd: 2.6573, ept: 50.4712
    Epoch [3/50], Test Losses: mse: 15.1144, mae: 3.0807, huber: 2.6185, swd: 3.0897, ept: 56.0124
      Epoch 3 composite train-obj: 2.274804
            No improvement (2.5024), counter 1/5
    Epoch [4/50], Train Losses: mse: 12.0198, mae: 2.7160, huber: 2.2623, swd: 2.8523, ept: 66.5276
    Epoch [4/50], Val Losses: mse: 12.9970, mae: 2.9198, huber: 2.4580, swd: 2.6046, ept: 52.1386
    Epoch [4/50], Test Losses: mse: 14.9213, mae: 3.0522, huber: 2.5916, swd: 3.0662, ept: 56.9393
      Epoch 4 composite train-obj: 2.262350
            Val objective improved 2.4750 → 2.4580, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 11.9898, mae: 2.7087, huber: 2.2559, swd: 2.8136, ept: 67.1123
    Epoch [5/50], Val Losses: mse: 13.0682, mae: 2.9301, huber: 2.4688, swd: 2.6815, ept: 52.3580
    Epoch [5/50], Test Losses: mse: 14.9045, mae: 3.0474, huber: 2.5872, swd: 3.0203, ept: 57.7010
      Epoch 5 composite train-obj: 2.255884
            No improvement (2.4688), counter 1/5
    Epoch [6/50], Train Losses: mse: 11.9649, mae: 2.7023, huber: 2.2503, swd: 2.7931, ept: 67.5922
    Epoch [6/50], Val Losses: mse: 12.7460, mae: 2.8857, huber: 2.4261, swd: 2.5882, ept: 53.4974
    Epoch [6/50], Test Losses: mse: 15.0199, mae: 3.0566, huber: 2.5970, swd: 3.0216, ept: 58.2419
      Epoch 6 composite train-obj: 2.250279
            Val objective improved 2.4580 → 2.4261, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 11.9201, mae: 2.6946, huber: 2.2432, swd: 2.7696, ept: 68.2701
    Epoch [7/50], Val Losses: mse: 13.1747, mae: 2.9368, huber: 2.4760, swd: 2.5967, ept: 50.8890
    Epoch [7/50], Test Losses: mse: 14.8392, mae: 3.0362, huber: 2.5768, swd: 2.9148, ept: 58.4581
      Epoch 7 composite train-obj: 2.243209
            No improvement (2.4760), counter 1/5
    Epoch [8/50], Train Losses: mse: 11.9207, mae: 2.6943, huber: 2.2431, swd: 2.7480, ept: 68.5278
    Epoch [8/50], Val Losses: mse: 12.8553, mae: 2.8989, huber: 2.4386, swd: 2.5895, ept: 56.1722
    Epoch [8/50], Test Losses: mse: 14.8895, mae: 3.0431, huber: 2.5841, swd: 3.0231, ept: 57.6173
      Epoch 8 composite train-obj: 2.243061
            No improvement (2.4386), counter 2/5
    Epoch [9/50], Train Losses: mse: 11.8911, mae: 2.6896, huber: 2.2386, swd: 2.7357, ept: 68.8736
    Epoch [9/50], Val Losses: mse: 13.0558, mae: 2.9246, huber: 2.4635, swd: 2.7857, ept: 53.0532
    Epoch [9/50], Test Losses: mse: 14.7861, mae: 3.0334, huber: 2.5744, swd: 3.1275, ept: 59.4324
      Epoch 9 composite train-obj: 2.238649
            No improvement (2.4635), counter 3/5
    Epoch [10/50], Train Losses: mse: 11.8883, mae: 2.6876, huber: 2.2370, swd: 2.7105, ept: 69.6092
    Epoch [10/50], Val Losses: mse: 12.8568, mae: 2.8945, huber: 2.4349, swd: 2.5951, ept: 56.2736
    Epoch [10/50], Test Losses: mse: 14.6172, mae: 3.0105, huber: 2.5529, swd: 2.9160, ept: 59.4346
      Epoch 10 composite train-obj: 2.237045
            No improvement (2.4349), counter 4/5
    Epoch [11/50], Train Losses: mse: 11.8262, mae: 2.6790, huber: 2.2289, swd: 2.7050, ept: 69.5641
    Epoch [11/50], Val Losses: mse: 12.6927, mae: 2.8728, huber: 2.4139, swd: 2.5489, ept: 54.8669
    Epoch [11/50], Test Losses: mse: 14.6125, mae: 3.0119, huber: 2.5537, swd: 2.9509, ept: 59.9385
      Epoch 11 composite train-obj: 2.228877
            Val objective improved 2.4261 → 2.4139, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 11.8341, mae: 2.6795, huber: 2.2295, swd: 2.6834, ept: 69.8204
    Epoch [12/50], Val Losses: mse: 12.9950, mae: 2.8982, huber: 2.4402, swd: 2.6749, ept: 57.4005
    Epoch [12/50], Test Losses: mse: 14.6970, mae: 3.0202, huber: 2.5619, swd: 2.9416, ept: 58.2678
      Epoch 12 composite train-obj: 2.229460
            No improvement (2.4402), counter 1/5
    Epoch [13/50], Train Losses: mse: 11.8147, mae: 2.6757, huber: 2.2260, swd: 2.6857, ept: 70.0370
    Epoch [13/50], Val Losses: mse: 12.4888, mae: 2.8522, huber: 2.3935, swd: 2.6101, ept: 54.6953
    Epoch [13/50], Test Losses: mse: 14.6551, mae: 3.0166, huber: 2.5586, swd: 2.9830, ept: 60.6553
      Epoch 13 composite train-obj: 2.225972
            Val objective improved 2.4139 → 2.3935, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 11.7827, mae: 2.6716, huber: 2.2221, swd: 2.6699, ept: 70.2850
    Epoch [14/50], Val Losses: mse: 12.8333, mae: 2.8860, huber: 2.4273, swd: 2.4512, ept: 56.0055
    Epoch [14/50], Test Losses: mse: 14.6727, mae: 3.0133, huber: 2.5560, swd: 2.7733, ept: 60.1751
      Epoch 14 composite train-obj: 2.222117
            No improvement (2.4273), counter 1/5
    Epoch [15/50], Train Losses: mse: 11.7680, mae: 2.6693, huber: 2.2199, swd: 2.6544, ept: 70.2504
    Epoch [15/50], Val Losses: mse: 12.5484, mae: 2.8525, huber: 2.3941, swd: 2.7917, ept: 57.8613
    Epoch [15/50], Test Losses: mse: 14.6857, mae: 3.0280, huber: 2.5698, swd: 3.0820, ept: 58.8456
      Epoch 15 composite train-obj: 2.219943
            No improvement (2.3941), counter 2/5
    Epoch [16/50], Train Losses: mse: 11.7571, mae: 2.6669, huber: 2.2177, swd: 2.6397, ept: 70.4695
    Epoch [16/50], Val Losses: mse: 12.6234, mae: 2.8637, huber: 2.4057, swd: 2.5740, ept: 55.2928
    Epoch [16/50], Test Losses: mse: 14.6303, mae: 3.0116, huber: 2.5539, swd: 2.9564, ept: 59.9448
      Epoch 16 composite train-obj: 2.217672
            No improvement (2.4057), counter 3/5
    Epoch [17/50], Train Losses: mse: 11.7491, mae: 2.6655, huber: 2.2164, swd: 2.6279, ept: 70.3922
    Epoch [17/50], Val Losses: mse: 12.7628, mae: 2.8790, huber: 2.4215, swd: 2.7047, ept: 55.4403
    Epoch [17/50], Test Losses: mse: 14.5052, mae: 2.9995, huber: 2.5421, swd: 2.9511, ept: 60.7091
      Epoch 17 composite train-obj: 2.216446
            No improvement (2.4215), counter 4/5
    Epoch [18/50], Train Losses: mse: 11.7487, mae: 2.6654, huber: 2.2165, swd: 2.6093, ept: 70.4999
    Epoch [18/50], Val Losses: mse: 12.1064, mae: 2.8062, huber: 2.3481, swd: 2.6270, ept: 56.8607
    Epoch [18/50], Test Losses: mse: 14.4531, mae: 2.9972, huber: 2.5402, swd: 3.0195, ept: 60.9939
      Epoch 18 composite train-obj: 2.216481
            Val objective improved 2.3935 → 2.3481, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 11.7228, mae: 2.6618, huber: 2.2131, swd: 2.6141, ept: 70.6835
    Epoch [19/50], Val Losses: mse: 12.3427, mae: 2.8258, huber: 2.3691, swd: 2.4814, ept: 59.0373
    Epoch [19/50], Test Losses: mse: 14.7536, mae: 3.0290, huber: 2.5712, swd: 2.8652, ept: 59.6307
      Epoch 19 composite train-obj: 2.213077
            No improvement (2.3691), counter 1/5
    Epoch [20/50], Train Losses: mse: 11.7111, mae: 2.6599, huber: 2.2113, swd: 2.5934, ept: 70.8908
    Epoch [20/50], Val Losses: mse: 12.3788, mae: 2.8307, huber: 2.3741, swd: 2.5428, ept: 58.9942
    Epoch [20/50], Test Losses: mse: 14.7697, mae: 3.0276, huber: 2.5703, swd: 2.8415, ept: 59.1607
      Epoch 20 composite train-obj: 2.211315
            No improvement (2.3741), counter 2/5
    Epoch [21/50], Train Losses: mse: 11.7123, mae: 2.6588, huber: 2.2102, swd: 2.5668, ept: 70.8959
    Epoch [21/50], Val Losses: mse: 12.6310, mae: 2.8664, huber: 2.4086, swd: 2.6364, ept: 56.2029
    Epoch [21/50], Test Losses: mse: 14.6700, mae: 3.0173, huber: 2.5598, swd: 2.8414, ept: 60.3409
      Epoch 21 composite train-obj: 2.210241
            No improvement (2.4086), counter 3/5
    Epoch [22/50], Train Losses: mse: 11.6987, mae: 2.6578, huber: 2.2094, swd: 2.5869, ept: 71.1059
    Epoch [22/50], Val Losses: mse: 12.2480, mae: 2.8131, huber: 2.3562, swd: 2.4912, ept: 59.3126
    Epoch [22/50], Test Losses: mse: 14.5476, mae: 3.0047, huber: 2.5476, swd: 2.8621, ept: 60.9751
      Epoch 22 composite train-obj: 2.209351
            No improvement (2.3562), counter 4/5
    Epoch [23/50], Train Losses: mse: 11.6804, mae: 2.6547, huber: 2.2063, swd: 2.5673, ept: 71.1698
    Epoch [23/50], Val Losses: mse: 12.3921, mae: 2.8383, huber: 2.3807, swd: 2.4935, ept: 57.3626
    Epoch [23/50], Test Losses: mse: 14.6542, mae: 3.0118, huber: 2.5547, swd: 2.7891, ept: 60.1227
      Epoch 23 composite train-obj: 2.206323
    Epoch [23/50], Test Losses: mse: 14.4531, mae: 2.9972, huber: 2.5402, swd: 3.0195, ept: 60.9939
    Best round's Test MSE: 14.4531, MAE: 2.9972, SWD: 3.0195
    Best round's Validation MSE: 12.1064, MAE: 2.8062, SWD: 2.6270
    Best round's Test verification MSE : 14.4531, MAE: 2.9972, SWD: 3.0195
    Time taken: 23.06 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz96_seq720_pred336_20250512_2156)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 14.6004 ± 0.1077
      mae: 3.0085 ± 0.0088
      huber: 2.5511 ± 0.0087
      swd: 2.9807 ± 0.0286
      ept: 60.6959 ± 0.7534
      count: 7.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 12.2838 ± 0.1261
      mae: 2.8218 ± 0.0110
      huber: 2.3640 ± 0.0113
      swd: 2.6090 ± 0.0388
      ept: 57.0602 ± 0.9885
      count: 7.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 84.41 seconds
    
    Experiment complete: DLinear_lorenz96_seq720_pred336_20250512_2156
    Model: DLinear
    Dataset: lorenz96
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
    channels=data_mgr.datasets['lorenz96']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('lorenz96', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([13300, 6])
    Shape of validation data: torch.Size([1900, 6])
    Shape of testing data: torch.Size([3800, 6])
    global_std.shape: torch.Size([6])
    Global Std for lorenz96: tensor([3.6750, 3.6678, 3.7240, 3.7347, 3.8038, 3.5588], device='cuda:0')
    Train set sample shapes: torch.Size([720, 6]), torch.Size([720, 6])
    Validation set sample shapes: torch.Size([720, 6]), torch.Size([720, 6])
    Test set data shapes: torch.Size([3800, 6]), torch.Size([3800, 6])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 720, 6]), Target shape torch.Size([128, 720, 6])
    
    ==================================================
    Data Preparation: lorenz96
    ==================================================
    Sequence Length: 720
    Prediction Length: 720
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 4
    Test Batches: 19
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.3342, mae: 2.9060, huber: 2.4441, swd: 2.8279, ept: 42.2006
    Epoch [1/50], Val Losses: mse: 14.9465, mae: 3.1526, huber: 2.6852, swd: 2.5076, ept: 42.9703
    Epoch [1/50], Test Losses: mse: 16.1307, mae: 3.2401, huber: 2.7712, swd: 3.2980, ept: 49.2586
      Epoch 1 composite train-obj: 2.444081
            Val objective improved inf → 2.6852, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.6334, mae: 2.8105, huber: 2.3515, swd: 2.7920, ept: 61.9106
    Epoch [2/50], Val Losses: mse: 14.6295, mae: 3.1158, huber: 2.6492, swd: 2.3502, ept: 49.7315
    Epoch [2/50], Test Losses: mse: 16.0486, mae: 3.2184, huber: 2.7517, swd: 3.3392, ept: 52.5089
      Epoch 2 composite train-obj: 2.351459
            Val objective improved 2.6852 → 2.6492, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.5971, mae: 2.8007, huber: 2.3428, swd: 2.7628, ept: 64.9139
    Epoch [3/50], Val Losses: mse: 14.6567, mae: 3.1152, huber: 2.6499, swd: 2.2927, ept: 47.6282
    Epoch [3/50], Test Losses: mse: 16.0099, mae: 3.2103, huber: 2.7438, swd: 3.0968, ept: 54.3344
      Epoch 3 composite train-obj: 2.342802
            No improvement (2.6499), counter 1/5
    Epoch [4/50], Train Losses: mse: 12.5701, mae: 2.7950, huber: 2.3376, swd: 2.7294, ept: 66.2647
    Epoch [4/50], Val Losses: mse: 14.6219, mae: 3.1013, huber: 2.6359, swd: 2.1824, ept: 52.1928
    Epoch [4/50], Test Losses: mse: 16.0996, mae: 3.2227, huber: 2.7564, swd: 3.0924, ept: 54.7260
      Epoch 4 composite train-obj: 2.337572
            Val objective improved 2.6492 → 2.6359, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 12.5626, mae: 2.7917, huber: 2.3348, swd: 2.7002, ept: 67.1755
    Epoch [5/50], Val Losses: mse: 14.9085, mae: 3.1382, huber: 2.6730, swd: 2.1890, ept: 51.6511
    Epoch [5/50], Test Losses: mse: 15.9756, mae: 3.2052, huber: 2.7391, swd: 3.0774, ept: 56.3298
      Epoch 5 composite train-obj: 2.334795
            No improvement (2.6730), counter 1/5
    Epoch [6/50], Train Losses: mse: 12.5627, mae: 2.7909, huber: 2.3342, swd: 2.6766, ept: 67.9995
    Epoch [6/50], Val Losses: mse: 14.5743, mae: 3.0906, huber: 2.6271, swd: 2.2527, ept: 51.1485
    Epoch [6/50], Test Losses: mse: 16.0542, mae: 3.2170, huber: 2.7512, swd: 3.1166, ept: 57.7391
      Epoch 6 composite train-obj: 2.334170
            Val objective improved 2.6359 → 2.6271, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 12.5232, mae: 2.7843, huber: 2.3280, swd: 2.6701, ept: 68.4950
    Epoch [7/50], Val Losses: mse: 14.9170, mae: 3.1304, huber: 2.6663, swd: 2.0110, ept: 53.7821
    Epoch [7/50], Test Losses: mse: 16.0352, mae: 3.2127, huber: 2.7469, swd: 2.8308, ept: 57.0062
      Epoch 7 composite train-obj: 2.327997
            No improvement (2.6663), counter 1/5
    Epoch [8/50], Train Losses: mse: 12.5052, mae: 2.7820, huber: 2.3259, swd: 2.6570, ept: 68.6452
    Epoch [8/50], Val Losses: mse: 14.7459, mae: 3.1120, huber: 2.6480, swd: 2.1309, ept: 52.4447
    Epoch [8/50], Test Losses: mse: 15.9924, mae: 3.2037, huber: 2.7385, swd: 3.0194, ept: 58.4194
      Epoch 8 composite train-obj: 2.325906
            No improvement (2.6480), counter 2/5
    Epoch [9/50], Train Losses: mse: 12.5018, mae: 2.7803, huber: 2.3243, swd: 2.6336, ept: 69.2527
    Epoch [9/50], Val Losses: mse: 14.3325, mae: 3.0692, huber: 2.6055, swd: 2.3047, ept: 54.3081
    Epoch [9/50], Test Losses: mse: 15.9686, mae: 3.2078, huber: 2.7426, swd: 3.1376, ept: 57.6937
      Epoch 9 composite train-obj: 2.324314
            Val objective improved 2.6271 → 2.6055, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 12.4999, mae: 2.7793, huber: 2.3235, swd: 2.6234, ept: 69.1368
    Epoch [10/50], Val Losses: mse: 14.7659, mae: 3.1229, huber: 2.6581, swd: 2.2884, ept: 54.6781
    Epoch [10/50], Test Losses: mse: 15.8619, mae: 3.1936, huber: 2.7286, swd: 3.1413, ept: 58.5102
      Epoch 10 composite train-obj: 2.323508
            No improvement (2.6581), counter 1/5
    Epoch [11/50], Train Losses: mse: 12.4737, mae: 2.7762, huber: 2.3206, swd: 2.6237, ept: 69.4148
    Epoch [11/50], Val Losses: mse: 15.0984, mae: 3.1539, huber: 2.6894, swd: 2.2683, ept: 55.1115
    Epoch [11/50], Test Losses: mse: 15.9702, mae: 3.2041, huber: 2.7392, swd: 3.0694, ept: 59.2495
      Epoch 11 composite train-obj: 2.320611
            No improvement (2.6894), counter 2/5
    Epoch [12/50], Train Losses: mse: 12.4541, mae: 2.7734, huber: 2.3179, swd: 2.6108, ept: 69.6350
    Epoch [12/50], Val Losses: mse: 14.4926, mae: 3.0783, huber: 2.6155, swd: 2.2169, ept: 53.8101
    Epoch [12/50], Test Losses: mse: 15.9066, mae: 3.1978, huber: 2.7328, swd: 2.9839, ept: 58.7778
      Epoch 12 composite train-obj: 2.317911
            No improvement (2.6155), counter 3/5
    Epoch [13/50], Train Losses: mse: 12.4475, mae: 2.7709, huber: 2.3157, swd: 2.5849, ept: 70.0462
    Epoch [13/50], Val Losses: mse: 14.6570, mae: 3.0968, huber: 2.6338, swd: 2.1919, ept: 54.6469
    Epoch [13/50], Test Losses: mse: 15.9204, mae: 3.1987, huber: 2.7337, swd: 3.0618, ept: 57.7763
      Epoch 13 composite train-obj: 2.315670
            No improvement (2.6338), counter 4/5
    Epoch [14/50], Train Losses: mse: 12.4405, mae: 2.7709, huber: 2.3156, swd: 2.5803, ept: 70.1513
    Epoch [14/50], Val Losses: mse: 14.7089, mae: 3.1041, huber: 2.6404, swd: 2.1561, ept: 57.6694
    Epoch [14/50], Test Losses: mse: 15.6525, mae: 3.1663, huber: 2.7019, swd: 2.9756, ept: 58.5508
      Epoch 14 composite train-obj: 2.315614
    Epoch [14/50], Test Losses: mse: 15.9686, mae: 3.2078, huber: 2.7426, swd: 3.1376, ept: 57.6937
    Best round's Test MSE: 15.9686, MAE: 3.2078, SWD: 3.1376
    Best round's Validation MSE: 14.3325, MAE: 3.0692, SWD: 2.3047
    Best round's Test verification MSE : 15.9686, MAE: 3.2078, SWD: 3.1376
    Time taken: 15.12 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.3459, mae: 2.9074, huber: 2.4455, swd: 2.8037, ept: 42.3709
    Epoch [1/50], Val Losses: mse: 14.7290, mae: 3.1275, huber: 2.6600, swd: 2.3080, ept: 43.6927
    Epoch [1/50], Test Losses: mse: 16.2002, mae: 3.2475, huber: 2.7789, swd: 3.2444, ept: 48.5974
      Epoch 1 composite train-obj: 2.445460
            Val objective improved inf → 2.6600, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.6441, mae: 2.8119, huber: 2.3529, swd: 2.7910, ept: 61.6472
    Epoch [2/50], Val Losses: mse: 15.1958, mae: 3.1759, huber: 2.7096, swd: 2.1675, ept: 49.1234
    Epoch [2/50], Test Losses: mse: 16.1029, mae: 3.2230, huber: 2.7560, swd: 3.0037, ept: 52.9227
      Epoch 2 composite train-obj: 2.352944
            No improvement (2.7096), counter 1/5
    Epoch [3/50], Train Losses: mse: 12.6030, mae: 2.8020, huber: 2.3439, swd: 2.7416, ept: 64.6477
    Epoch [3/50], Val Losses: mse: 14.8540, mae: 3.1294, huber: 2.6635, swd: 2.4124, ept: 52.3686
    Epoch [3/50], Test Losses: mse: 16.1650, mae: 3.2354, huber: 2.7683, swd: 3.3132, ept: 53.9136
      Epoch 3 composite train-obj: 2.343947
            No improvement (2.6635), counter 2/5
    Epoch [4/50], Train Losses: mse: 12.5679, mae: 2.7941, huber: 2.3368, swd: 2.7149, ept: 65.8610
    Epoch [4/50], Val Losses: mse: 14.7359, mae: 3.1179, huber: 2.6522, swd: 2.1669, ept: 50.5715
    Epoch [4/50], Test Losses: mse: 16.0178, mae: 3.2130, huber: 2.7467, swd: 3.0563, ept: 55.9060
      Epoch 4 composite train-obj: 2.336780
            Val objective improved 2.6600 → 2.6522, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 12.5557, mae: 2.7905, huber: 2.3337, swd: 2.6925, ept: 67.2338
    Epoch [5/50], Val Losses: mse: 14.9906, mae: 3.1466, huber: 2.6812, swd: 2.2413, ept: 51.9046
    Epoch [5/50], Test Losses: mse: 15.9051, mae: 3.1991, huber: 2.7330, swd: 3.1608, ept: 55.6381
      Epoch 5 composite train-obj: 2.333664
            No improvement (2.6812), counter 1/5
    Epoch [6/50], Train Losses: mse: 12.5556, mae: 2.7901, huber: 2.3333, swd: 2.6778, ept: 67.5905
    Epoch [6/50], Val Losses: mse: 14.6082, mae: 3.1054, huber: 2.6407, swd: 2.3109, ept: 50.7689
    Epoch [6/50], Test Losses: mse: 16.1234, mae: 3.2224, huber: 2.7567, swd: 3.1118, ept: 56.0007
      Epoch 6 composite train-obj: 2.333325
            Val objective improved 2.6522 → 2.6407, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 12.5115, mae: 2.7827, huber: 2.3265, swd: 2.6600, ept: 68.4276
    Epoch [7/50], Val Losses: mse: 15.0702, mae: 3.1515, huber: 2.6870, swd: 2.1805, ept: 53.3779
    Epoch [7/50], Test Losses: mse: 16.0714, mae: 3.2126, huber: 2.7470, swd: 2.9977, ept: 57.3081
      Epoch 7 composite train-obj: 2.326535
            No improvement (2.6870), counter 1/5
    Epoch [8/50], Train Losses: mse: 12.5167, mae: 2.7829, huber: 2.3268, swd: 2.6413, ept: 68.6649
    Epoch [8/50], Val Losses: mse: 14.6286, mae: 3.1100, huber: 2.6457, swd: 2.2263, ept: 53.6688
    Epoch [8/50], Test Losses: mse: 15.8213, mae: 3.1866, huber: 2.7211, swd: 3.0313, ept: 57.3726
      Epoch 8 composite train-obj: 2.326782
            No improvement (2.6457), counter 2/5
    Epoch [9/50], Train Losses: mse: 12.4973, mae: 2.7806, huber: 2.3246, swd: 2.6251, ept: 69.0446
    Epoch [9/50], Val Losses: mse: 14.2848, mae: 3.0633, huber: 2.5995, swd: 2.2746, ept: 55.0519
    Epoch [9/50], Test Losses: mse: 15.9516, mae: 3.2041, huber: 2.7390, swd: 3.1689, ept: 57.1176
      Epoch 9 composite train-obj: 2.324644
            Val objective improved 2.6407 → 2.5995, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 12.4839, mae: 2.7773, huber: 2.3216, swd: 2.6230, ept: 69.4545
    Epoch [10/50], Val Losses: mse: 14.6271, mae: 3.1039, huber: 2.6394, swd: 2.1892, ept: 53.4369
    Epoch [10/50], Test Losses: mse: 15.9740, mae: 3.2091, huber: 2.7436, swd: 2.9769, ept: 57.9931
      Epoch 10 composite train-obj: 2.321574
            No improvement (2.6394), counter 1/5
    Epoch [11/50], Train Losses: mse: 12.4567, mae: 2.7732, huber: 2.3178, swd: 2.6067, ept: 69.6905
    Epoch [11/50], Val Losses: mse: 14.3570, mae: 3.0727, huber: 2.6092, swd: 2.4371, ept: 54.3169
    Epoch [11/50], Test Losses: mse: 15.8277, mae: 3.1893, huber: 2.7247, swd: 3.2154, ept: 58.0165
      Epoch 11 composite train-obj: 2.317757
            No improvement (2.6092), counter 2/5
    Epoch [12/50], Train Losses: mse: 12.4418, mae: 2.7710, huber: 2.3156, swd: 2.6015, ept: 69.6543
    Epoch [12/50], Val Losses: mse: 14.6101, mae: 3.1047, huber: 2.6404, swd: 2.3346, ept: 54.2822
    Epoch [12/50], Test Losses: mse: 16.0279, mae: 3.2061, huber: 2.7409, swd: 3.1004, ept: 58.4169
      Epoch 12 composite train-obj: 2.315625
            No improvement (2.6404), counter 3/5
    Epoch [13/50], Train Losses: mse: 12.4670, mae: 2.7741, huber: 2.3187, swd: 2.5848, ept: 69.9279
    Epoch [13/50], Val Losses: mse: 14.6712, mae: 3.1037, huber: 2.6399, swd: 2.2327, ept: 54.7375
    Epoch [13/50], Test Losses: mse: 15.9325, mae: 3.1970, huber: 2.7320, swd: 2.9592, ept: 59.1984
      Epoch 13 composite train-obj: 2.318722
            No improvement (2.6399), counter 4/5
    Epoch [14/50], Train Losses: mse: 12.4417, mae: 2.7708, huber: 2.3155, swd: 2.5738, ept: 70.0781
    Epoch [14/50], Val Losses: mse: 14.5302, mae: 3.0841, huber: 2.6210, swd: 2.2418, ept: 55.0381
    Epoch [14/50], Test Losses: mse: 15.7802, mae: 3.1831, huber: 2.7185, swd: 3.0203, ept: 59.0294
      Epoch 14 composite train-obj: 2.315549
    Epoch [14/50], Test Losses: mse: 15.9516, mae: 3.2041, huber: 2.7390, swd: 3.1689, ept: 57.1176
    Best round's Test MSE: 15.9516, MAE: 3.2041, SWD: 3.1689
    Best round's Validation MSE: 14.2848, MAE: 3.0633, SWD: 2.2746
    Best round's Test verification MSE : 15.9516, MAE: 3.2041, SWD: 3.1689
    Time taken: 15.41 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 13.3557, mae: 2.9085, huber: 2.4465, swd: 2.8386, ept: 42.1363
    Epoch [1/50], Val Losses: mse: 14.9150, mae: 3.1507, huber: 2.6838, swd: 2.3360, ept: 42.9487
    Epoch [1/50], Test Losses: mse: 15.9666, mae: 3.2144, huber: 2.7464, swd: 3.2723, ept: 49.0128
      Epoch 1 composite train-obj: 2.446452
            Val objective improved inf → 2.6838, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 12.6404, mae: 2.8121, huber: 2.3531, swd: 2.8244, ept: 61.7877
    Epoch [2/50], Val Losses: mse: 14.9794, mae: 3.1493, huber: 2.6834, swd: 2.2382, ept: 49.1058
    Epoch [2/50], Test Losses: mse: 15.9957, mae: 3.2151, huber: 2.7479, swd: 3.1699, ept: 52.0944
      Epoch 2 composite train-obj: 2.353066
            Val objective improved 2.6838 → 2.6834, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 12.6069, mae: 2.8019, huber: 2.3439, swd: 2.7714, ept: 64.7226
    Epoch [3/50], Val Losses: mse: 14.5788, mae: 3.0990, huber: 2.6340, swd: 2.1604, ept: 49.7277
    Epoch [3/50], Test Losses: mse: 15.9641, mae: 3.2059, huber: 2.7395, swd: 3.0598, ept: 54.0236
      Epoch 3 composite train-obj: 2.343920
            Val objective improved 2.6834 → 2.6340, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 12.5693, mae: 2.7947, huber: 2.3374, swd: 2.7489, ept: 66.2531
    Epoch [4/50], Val Losses: mse: 14.9447, mae: 3.1361, huber: 2.6715, swd: 2.2911, ept: 50.1778
    Epoch [4/50], Test Losses: mse: 16.0872, mae: 3.2203, huber: 2.7539, swd: 3.0853, ept: 54.6037
      Epoch 4 composite train-obj: 2.337413
            No improvement (2.6715), counter 1/5
    Epoch [5/50], Train Losses: mse: 12.5650, mae: 2.7916, huber: 2.3347, swd: 2.7247, ept: 67.1935
    Epoch [5/50], Val Losses: mse: 14.6728, mae: 3.1131, huber: 2.6486, swd: 2.1832, ept: 50.6267
    Epoch [5/50], Test Losses: mse: 15.9443, mae: 3.1976, huber: 2.7318, swd: 3.0215, ept: 57.2853
      Epoch 5 composite train-obj: 2.334673
            No improvement (2.6486), counter 2/5
    Epoch [6/50], Train Losses: mse: 12.5283, mae: 2.7863, huber: 2.3298, swd: 2.7155, ept: 67.8526
    Epoch [6/50], Val Losses: mse: 14.9495, mae: 3.1364, huber: 2.6716, swd: 2.4025, ept: 51.5680
    Epoch [6/50], Test Losses: mse: 16.0700, mae: 3.2177, huber: 2.7518, swd: 3.1665, ept: 56.5431
      Epoch 6 composite train-obj: 2.329805
            No improvement (2.6716), counter 3/5
    Epoch [7/50], Train Losses: mse: 12.5368, mae: 2.7865, huber: 2.3301, swd: 2.6913, ept: 68.3922
    Epoch [7/50], Val Losses: mse: 14.5869, mae: 3.0936, huber: 2.6300, swd: 2.2688, ept: 51.5584
    Epoch [7/50], Test Losses: mse: 15.9961, mae: 3.2067, huber: 2.7410, swd: 3.0612, ept: 57.0430
      Epoch 7 composite train-obj: 2.330104
            Val objective improved 2.6340 → 2.6300, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 12.5079, mae: 2.7815, huber: 2.3255, swd: 2.6737, ept: 68.8099
    Epoch [8/50], Val Losses: mse: 14.5280, mae: 3.0915, huber: 2.6263, swd: 2.2967, ept: 52.8824
    Epoch [8/50], Test Losses: mse: 15.8181, mae: 3.1904, huber: 2.7250, swd: 3.0968, ept: 57.3095
      Epoch 8 composite train-obj: 2.325453
            Val objective improved 2.6300 → 2.6263, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 12.4917, mae: 2.7787, huber: 2.3229, swd: 2.6670, ept: 68.8744
    Epoch [9/50], Val Losses: mse: 14.8737, mae: 3.1297, huber: 2.6654, swd: 2.2838, ept: 54.2122
    Epoch [9/50], Test Losses: mse: 15.8789, mae: 3.1942, huber: 2.7290, swd: 3.1137, ept: 57.7652
      Epoch 9 composite train-obj: 2.322872
            No improvement (2.6654), counter 1/5
    Epoch [10/50], Train Losses: mse: 12.4924, mae: 2.7788, huber: 2.3229, swd: 2.6523, ept: 69.3741
    Epoch [10/50], Val Losses: mse: 14.7952, mae: 3.1264, huber: 2.6622, swd: 2.3572, ept: 52.0294
    Epoch [10/50], Test Losses: mse: 16.0730, mae: 3.2160, huber: 2.7503, swd: 3.0021, ept: 58.9907
      Epoch 10 composite train-obj: 2.322931
            No improvement (2.6622), counter 2/5
    Epoch [11/50], Train Losses: mse: 12.4729, mae: 2.7760, huber: 2.3204, swd: 2.6422, ept: 69.5775
    Epoch [11/50], Val Losses: mse: 14.7801, mae: 3.1113, huber: 2.6482, swd: 2.2683, ept: 54.1610
    Epoch [11/50], Test Losses: mse: 15.9593, mae: 3.2027, huber: 2.7374, swd: 2.9684, ept: 58.2773
      Epoch 11 composite train-obj: 2.320383
            No improvement (2.6482), counter 3/5
    Epoch [12/50], Train Losses: mse: 12.4527, mae: 2.7731, huber: 2.3176, swd: 2.6438, ept: 69.8445
    Epoch [12/50], Val Losses: mse: 14.4002, mae: 3.0686, huber: 2.6067, swd: 2.1560, ept: 53.4947
    Epoch [12/50], Test Losses: mse: 16.0450, mae: 3.2078, huber: 2.7432, swd: 2.9455, ept: 57.3249
      Epoch 12 composite train-obj: 2.317649
            Val objective improved 2.6263 → 2.6067, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 12.4477, mae: 2.7712, huber: 2.3159, swd: 2.6148, ept: 69.9763
    Epoch [13/50], Val Losses: mse: 14.8485, mae: 3.1273, huber: 2.6634, swd: 2.3202, ept: 54.4488
    Epoch [13/50], Test Losses: mse: 15.7802, mae: 3.1861, huber: 2.7210, swd: 3.0206, ept: 58.7661
      Epoch 13 composite train-obj: 2.315859
            No improvement (2.6634), counter 1/5
    Epoch [14/50], Train Losses: mse: 12.4355, mae: 2.7700, huber: 2.3147, swd: 2.6127, ept: 70.1664
    Epoch [14/50], Val Losses: mse: 14.6342, mae: 3.0965, huber: 2.6327, swd: 2.3246, ept: 54.6870
    Epoch [14/50], Test Losses: mse: 15.9185, mae: 3.1981, huber: 2.7330, swd: 2.9986, ept: 60.2063
      Epoch 14 composite train-obj: 2.314740
            No improvement (2.6327), counter 2/5
    Epoch [15/50], Train Losses: mse: 12.4552, mae: 2.7721, huber: 2.3168, swd: 2.5976, ept: 70.1827
    Epoch [15/50], Val Losses: mse: 14.7183, mae: 3.0974, huber: 2.6333, swd: 2.2512, ept: 56.3489
    Epoch [15/50], Test Losses: mse: 15.9463, mae: 3.1992, huber: 2.7342, swd: 2.9189, ept: 59.1775
      Epoch 15 composite train-obj: 2.316836
            No improvement (2.6333), counter 3/5
    Epoch [16/50], Train Losses: mse: 12.4267, mae: 2.7683, huber: 2.3132, swd: 2.5898, ept: 70.4941
    Epoch [16/50], Val Losses: mse: 14.3205, mae: 3.0602, huber: 2.5973, swd: 2.2484, ept: 55.8785
    Epoch [16/50], Test Losses: mse: 15.8811, mae: 3.1900, huber: 2.7256, swd: 2.9306, ept: 58.9233
      Epoch 16 composite train-obj: 2.313200
            Val objective improved 2.6067 → 2.5973, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 12.3962, mae: 2.7643, huber: 2.3095, swd: 2.5945, ept: 70.4872
    Epoch [17/50], Val Losses: mse: 14.4610, mae: 3.0723, huber: 2.6095, swd: 2.1286, ept: 56.8810
    Epoch [17/50], Test Losses: mse: 15.7878, mae: 3.1841, huber: 2.7196, swd: 2.8950, ept: 58.3989
      Epoch 17 composite train-obj: 2.309509
            No improvement (2.6095), counter 1/5
    Epoch [18/50], Train Losses: mse: 12.4026, mae: 2.7644, huber: 2.3095, swd: 2.5872, ept: 70.9034
    Epoch [18/50], Val Losses: mse: 14.2958, mae: 3.0553, huber: 2.5924, swd: 2.2325, ept: 54.0713
    Epoch [18/50], Test Losses: mse: 15.9142, mae: 3.1947, huber: 2.7303, swd: 2.9402, ept: 59.1586
      Epoch 18 composite train-obj: 2.309534
            Val objective improved 2.5973 → 2.5924, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 12.3938, mae: 2.7636, huber: 2.3087, swd: 2.5650, ept: 70.6548
    Epoch [19/50], Val Losses: mse: 14.5913, mae: 3.0843, huber: 2.6213, swd: 2.2057, ept: 57.0317
    Epoch [19/50], Test Losses: mse: 15.8755, mae: 3.1941, huber: 2.7294, swd: 2.9462, ept: 60.2020
      Epoch 19 composite train-obj: 2.308718
            No improvement (2.6213), counter 1/5
    Epoch [20/50], Train Losses: mse: 12.3936, mae: 2.7632, huber: 2.3084, swd: 2.5571, ept: 70.8600
    Epoch [20/50], Val Losses: mse: 14.5574, mae: 3.0933, huber: 2.6296, swd: 2.1084, ept: 56.7976
    Epoch [20/50], Test Losses: mse: 15.9489, mae: 3.2007, huber: 2.7359, swd: 2.8235, ept: 60.5424
      Epoch 20 composite train-obj: 2.308394
            No improvement (2.6296), counter 2/5
    Epoch [21/50], Train Losses: mse: 12.3886, mae: 2.7625, huber: 2.3078, swd: 2.5522, ept: 71.2245
    Epoch [21/50], Val Losses: mse: 14.9253, mae: 3.1271, huber: 2.6633, swd: 2.1954, ept: 56.5936
    Epoch [21/50], Test Losses: mse: 15.8163, mae: 3.1878, huber: 2.7233, swd: 2.9360, ept: 59.2098
      Epoch 21 composite train-obj: 2.307757
            No improvement (2.6633), counter 3/5
    Epoch [22/50], Train Losses: mse: 12.3565, mae: 2.7582, huber: 2.3036, swd: 2.5538, ept: 71.0546
    Epoch [22/50], Val Losses: mse: 14.4612, mae: 3.0766, huber: 2.6144, swd: 2.2990, ept: 54.6803
    Epoch [22/50], Test Losses: mse: 15.6955, mae: 3.1718, huber: 2.7079, swd: 3.0082, ept: 59.5132
      Epoch 22 composite train-obj: 2.303626
            No improvement (2.6144), counter 4/5
    Epoch [23/50], Train Losses: mse: 12.3539, mae: 2.7577, huber: 2.3032, swd: 2.5441, ept: 71.3670
    Epoch [23/50], Val Losses: mse: 14.0345, mae: 3.0233, huber: 2.5616, swd: 2.5053, ept: 57.0620
    Epoch [23/50], Test Losses: mse: 15.7681, mae: 3.1823, huber: 2.7181, swd: 3.0759, ept: 58.9597
      Epoch 23 composite train-obj: 2.303167
            Val objective improved 2.5924 → 2.5616, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 12.3573, mae: 2.7573, huber: 2.3028, swd: 2.5365, ept: 71.4790
    Epoch [24/50], Val Losses: mse: 14.0480, mae: 3.0219, huber: 2.5608, swd: 2.2377, ept: 57.0678
    Epoch [24/50], Test Losses: mse: 15.8468, mae: 3.1866, huber: 2.7227, swd: 2.8604, ept: 59.9996
      Epoch 24 composite train-obj: 2.302797
            Val objective improved 2.5616 → 2.5608, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 12.3738, mae: 2.7596, huber: 2.3051, swd: 2.5149, ept: 71.4880
    Epoch [25/50], Val Losses: mse: 14.3046, mae: 3.0478, huber: 2.5866, swd: 2.2826, ept: 54.8398
    Epoch [25/50], Test Losses: mse: 15.8308, mae: 3.1874, huber: 2.7230, swd: 2.8358, ept: 61.0614
      Epoch 25 composite train-obj: 2.305080
            No improvement (2.5866), counter 1/5
    Epoch [26/50], Train Losses: mse: 12.3303, mae: 2.7547, huber: 2.3003, swd: 2.5272, ept: 71.7249
    Epoch [26/50], Val Losses: mse: 14.3158, mae: 3.0542, huber: 2.5923, swd: 2.4065, ept: 58.4099
    Epoch [26/50], Test Losses: mse: 15.7450, mae: 3.1828, huber: 2.7183, swd: 2.9604, ept: 60.6729
      Epoch 26 composite train-obj: 2.300273
            No improvement (2.5923), counter 2/5
    Epoch [27/50], Train Losses: mse: 12.3127, mae: 2.7516, huber: 2.2974, swd: 2.5197, ept: 71.7508
    Epoch [27/50], Val Losses: mse: 14.0836, mae: 3.0354, huber: 2.5723, swd: 2.2479, ept: 56.1830
    Epoch [27/50], Test Losses: mse: 15.7213, mae: 3.1737, huber: 2.7097, swd: 2.8623, ept: 60.6438
      Epoch 27 composite train-obj: 2.297369
            No improvement (2.5723), counter 3/5
    Epoch [28/50], Train Losses: mse: 12.3214, mae: 2.7532, huber: 2.2989, swd: 2.5158, ept: 71.6876
    Epoch [28/50], Val Losses: mse: 14.1576, mae: 3.0404, huber: 2.5780, swd: 2.2674, ept: 56.2421
    Epoch [28/50], Test Losses: mse: 15.6756, mae: 3.1706, huber: 2.7065, swd: 2.9159, ept: 60.4319
      Epoch 28 composite train-obj: 2.298861
            No improvement (2.5780), counter 4/5
    Epoch [29/50], Train Losses: mse: 12.3254, mae: 2.7536, huber: 2.2993, swd: 2.5046, ept: 71.6796
    Epoch [29/50], Val Losses: mse: 14.0119, mae: 3.0264, huber: 2.5643, swd: 2.3165, ept: 56.4927
    Epoch [29/50], Test Losses: mse: 15.7241, mae: 3.1753, huber: 2.7112, swd: 2.9319, ept: 60.3720
      Epoch 29 composite train-obj: 2.299296
    Epoch [29/50], Test Losses: mse: 15.8468, mae: 3.1866, huber: 2.7227, swd: 2.8604, ept: 59.9996
    Best round's Test MSE: 15.8468, MAE: 3.1866, SWD: 2.8604
    Best round's Validation MSE: 14.0480, MAE: 3.0219, SWD: 2.2377
    Best round's Test verification MSE : 15.8468, MAE: 3.1866, SWD: 2.8604
    Time taken: 31.37 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz96_seq720_pred720_20250512_2157)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 15.9223 ± 0.0538
      mae: 3.1995 ± 0.0092
      huber: 2.7348 ± 0.0087
      swd: 3.0556 ± 0.1387
      ept: 58.2703 ± 1.2452
      count: 4.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 14.2218 ± 0.1244
      mae: 3.0515 ± 0.0211
      huber: 2.5886 ± 0.0198
      swd: 2.2724 ± 0.0274
      ept: 55.4759 ± 1.1659
      count: 4.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 62.03 seconds
    
    Experiment complete: DLinear_lorenz96_seq720_pred720_20250512_2157
    Model: DLinear
    Dataset: lorenz96
    Sequence Length: 720
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


