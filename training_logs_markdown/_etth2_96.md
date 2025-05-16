# ETTh2


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
data_mgr = DatasetManager(device='cuda')

# Load a synthetic dataset
data_mgr.load_csv('etth2', './etth2.csv')
```

    
    ==================================================
    Dataset: etth2 (csv)
    ==================================================
    Shape: torch.Size([17420, 7])
    Channels: 7
    Length: 17420
    Source: ./etth2.csv
    
    Sample data (first 2 rows):
    tensor([[41.1300, 12.4810, 36.5360,  9.3550,  4.4240,  1.3110, 38.6620],
            [37.5280, 10.1360, 33.9360,  7.5320,  4.4350,  1.2150, 37.1240]])
    ==================================================
    




    <data_manager.DatasetManager at 0x2a3850bb530>





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
    channels=data_mgr.datasets['etth2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.4033, mae: 0.3984, huber: 0.1576, swd: 0.2310, ept: 69.4499
    Epoch [1/50], Val Losses: mse: 0.3048, mae: 0.3886, huber: 0.1424, swd: 0.1615, ept: 64.3925
    Epoch [1/50], Test Losses: mse: 0.1971, mae: 0.3142, huber: 0.0945, swd: 0.0921, ept: 73.5153
      Epoch 1 composite train-obj: 0.157631
            Val objective improved inf → 0.1424, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2950, mae: 0.3323, huber: 0.1164, swd: 0.1324, ept: 76.7907
    Epoch [2/50], Val Losses: mse: 0.2957, mae: 0.3742, huber: 0.1380, swd: 0.1467, ept: 64.6343
    Epoch [2/50], Test Losses: mse: 0.1870, mae: 0.3030, huber: 0.0897, swd: 0.0790, ept: 74.3124
      Epoch 2 composite train-obj: 0.116373
            Val objective improved 0.1424 → 0.1380, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2714, mae: 0.3145, huber: 0.1072, swd: 0.1171, ept: 78.8191
    Epoch [3/50], Val Losses: mse: 0.2838, mae: 0.3708, huber: 0.1327, swd: 0.1469, ept: 67.1589
    Epoch [3/50], Test Losses: mse: 0.1837, mae: 0.3007, huber: 0.0879, swd: 0.0858, ept: 75.4206
      Epoch 3 composite train-obj: 0.107183
            Val objective improved 0.1380 → 0.1327, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2528, mae: 0.3033, huber: 0.1009, swd: 0.1078, ept: 80.3659
    Epoch [4/50], Val Losses: mse: 0.2912, mae: 0.3745, huber: 0.1354, swd: 0.1560, ept: 67.2829
    Epoch [4/50], Test Losses: mse: 0.1918, mae: 0.3064, huber: 0.0915, swd: 0.0968, ept: 73.8014
      Epoch 4 composite train-obj: 0.100866
            No improvement (0.1354), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.2400, mae: 0.2971, huber: 0.0970, swd: 0.0993, ept: 81.0618
    Epoch [5/50], Val Losses: mse: 0.2915, mae: 0.3767, huber: 0.1356, swd: 0.1610, ept: 68.0946
    Epoch [5/50], Test Losses: mse: 0.1982, mae: 0.3154, huber: 0.0945, swd: 0.1032, ept: 74.4919
      Epoch 5 composite train-obj: 0.097044
            No improvement (0.1356), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.2318, mae: 0.2934, huber: 0.0945, swd: 0.0952, ept: 81.4498
    Epoch [6/50], Val Losses: mse: 0.2682, mae: 0.3550, huber: 0.1265, swd: 0.1230, ept: 68.4310
    Epoch [6/50], Test Losses: mse: 0.1750, mae: 0.2899, huber: 0.0840, swd: 0.0740, ept: 76.2017
      Epoch 6 composite train-obj: 0.094525
            Val objective improved 0.1327 → 0.1265, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.2239, mae: 0.2900, huber: 0.0922, swd: 0.0921, ept: 81.7409
    Epoch [7/50], Val Losses: mse: 0.2892, mae: 0.3756, huber: 0.1352, swd: 0.1565, ept: 67.9082
    Epoch [7/50], Test Losses: mse: 0.1990, mae: 0.3149, huber: 0.0950, swd: 0.1054, ept: 74.2736
      Epoch 7 composite train-obj: 0.092153
            No improvement (0.1352), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.2182, mae: 0.2878, huber: 0.0905, swd: 0.0902, ept: 81.9042
    Epoch [8/50], Val Losses: mse: 0.2846, mae: 0.3709, huber: 0.1333, swd: 0.1415, ept: 67.9067
    Epoch [8/50], Test Losses: mse: 0.1869, mae: 0.3029, huber: 0.0895, swd: 0.0899, ept: 74.8689
      Epoch 8 composite train-obj: 0.090454
            No improvement (0.1333), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.2119, mae: 0.2844, huber: 0.0883, swd: 0.0867, ept: 82.0895
    Epoch [9/50], Val Losses: mse: 0.2813, mae: 0.3602, huber: 0.1313, swd: 0.1320, ept: 68.2338
    Epoch [9/50], Test Losses: mse: 0.1874, mae: 0.2927, huber: 0.0888, swd: 0.0810, ept: 75.0747
      Epoch 9 composite train-obj: 0.088307
            No improvement (0.1313), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.2072, mae: 0.2820, huber: 0.0866, swd: 0.0842, ept: 82.3443
    Epoch [10/50], Val Losses: mse: 0.2737, mae: 0.3592, huber: 0.1286, swd: 0.1300, ept: 68.4700
    Epoch [10/50], Test Losses: mse: 0.1920, mae: 0.3035, huber: 0.0913, swd: 0.0879, ept: 75.3262
      Epoch 10 composite train-obj: 0.086593
            No improvement (0.1286), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.2034, mae: 0.2794, huber: 0.0851, swd: 0.0818, ept: 82.4871
    Epoch [11/50], Val Losses: mse: 0.3059, mae: 0.3843, huber: 0.1422, swd: 0.1592, ept: 67.7881
    Epoch [11/50], Test Losses: mse: 0.2165, mae: 0.3263, huber: 0.1024, swd: 0.1080, ept: 74.1619
      Epoch 11 composite train-obj: 0.085057
    Epoch [11/50], Test Losses: mse: 0.1750, mae: 0.2899, huber: 0.0840, swd: 0.0740, ept: 76.2018
    Best round's Test MSE: 0.1750, MAE: 0.2899, SWD: 0.0740
    Best round's Validation MSE: 0.2682, MAE: 0.3550, SWD: 0.1230
    Best round's Test verification MSE : 0.1750, MAE: 0.2899, SWD: 0.0740
    Time taken: 31.44 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4005, mae: 0.3963, huber: 0.1563, swd: 0.2233, ept: 69.9008
    Epoch [1/50], Val Losses: mse: 0.2996, mae: 0.3815, huber: 0.1404, swd: 0.1419, ept: 63.3508
    Epoch [1/50], Test Losses: mse: 0.1954, mae: 0.3123, huber: 0.0937, swd: 0.0806, ept: 74.0064
      Epoch 1 composite train-obj: 0.156305
            Val objective improved inf → 0.1404, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2926, mae: 0.3309, huber: 0.1155, swd: 0.1279, ept: 76.8904
    Epoch [2/50], Val Losses: mse: 0.2849, mae: 0.3718, huber: 0.1335, swd: 0.1442, ept: 65.6677
    Epoch [2/50], Test Losses: mse: 0.1894, mae: 0.3058, huber: 0.0907, swd: 0.0835, ept: 73.6452
      Epoch 2 composite train-obj: 0.115452
            Val objective improved 0.1404 → 0.1335, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2663, mae: 0.3115, huber: 0.1055, swd: 0.1129, ept: 78.8604
    Epoch [3/50], Val Losses: mse: 0.2708, mae: 0.3574, huber: 0.1276, swd: 0.1280, ept: 67.1114
    Epoch [3/50], Test Losses: mse: 0.1787, mae: 0.2927, huber: 0.0854, swd: 0.0735, ept: 75.1323
      Epoch 3 composite train-obj: 0.105488
            Val objective improved 0.1335 → 0.1276, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2503, mae: 0.3035, huber: 0.1003, swd: 0.1056, ept: 80.1796
    Epoch [4/50], Val Losses: mse: 0.2792, mae: 0.3619, huber: 0.1304, swd: 0.1359, ept: 67.2643
    Epoch [4/50], Test Losses: mse: 0.1803, mae: 0.2927, huber: 0.0860, swd: 0.0762, ept: 74.9060
      Epoch 4 composite train-obj: 0.100328
            No improvement (0.1304), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.2375, mae: 0.2960, huber: 0.0962, swd: 0.0976, ept: 80.8624
    Epoch [5/50], Val Losses: mse: 0.2954, mae: 0.3793, huber: 0.1376, swd: 0.1496, ept: 66.6558
    Epoch [5/50], Test Losses: mse: 0.1889, mae: 0.3031, huber: 0.0900, swd: 0.0874, ept: 74.4823
      Epoch 5 composite train-obj: 0.096239
            No improvement (0.1376), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.2278, mae: 0.2932, huber: 0.0937, swd: 0.0926, ept: 81.2447
    Epoch [6/50], Val Losses: mse: 0.2671, mae: 0.3545, huber: 0.1260, swd: 0.1178, ept: 67.2462
    Epoch [6/50], Test Losses: mse: 0.1823, mae: 0.2924, huber: 0.0867, swd: 0.0752, ept: 75.2622
      Epoch 6 composite train-obj: 0.093651
            Val objective improved 0.1276 → 0.1260, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.2195, mae: 0.2879, huber: 0.0907, swd: 0.0890, ept: 81.6476
    Epoch [7/50], Val Losses: mse: 0.2991, mae: 0.3787, huber: 0.1391, swd: 0.1518, ept: 66.9110
    Epoch [7/50], Test Losses: mse: 0.2201, mae: 0.3295, huber: 0.1039, swd: 0.1135, ept: 72.8326
      Epoch 7 composite train-obj: 0.090701
            No improvement (0.1391), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.2146, mae: 0.2863, huber: 0.0892, swd: 0.0880, ept: 81.9089
    Epoch [8/50], Val Losses: mse: 0.2948, mae: 0.3761, huber: 0.1375, swd: 0.1486, ept: 67.6761
    Epoch [8/50], Test Losses: mse: 0.2056, mae: 0.3161, huber: 0.0976, swd: 0.1040, ept: 73.1327
      Epoch 8 composite train-obj: 0.089234
            No improvement (0.1375), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.2089, mae: 0.2828, huber: 0.0872, swd: 0.0848, ept: 82.0828
    Epoch [9/50], Val Losses: mse: 0.2905, mae: 0.3665, huber: 0.1349, swd: 0.1359, ept: 68.0196
    Epoch [9/50], Test Losses: mse: 0.2102, mae: 0.3143, huber: 0.0988, swd: 0.0984, ept: 73.3022
      Epoch 9 composite train-obj: 0.087162
            No improvement (0.1349), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.2038, mae: 0.2801, huber: 0.0854, swd: 0.0819, ept: 82.3143
    Epoch [10/50], Val Losses: mse: 0.2774, mae: 0.3607, huber: 0.1302, swd: 0.1293, ept: 69.1320
    Epoch [10/50], Test Losses: mse: 0.1929, mae: 0.2999, huber: 0.0913, swd: 0.0868, ept: 74.5068
      Epoch 10 composite train-obj: 0.085431
            No improvement (0.1302), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.1981, mae: 0.2761, huber: 0.0832, swd: 0.0786, ept: 82.6334
    Epoch [11/50], Val Losses: mse: 0.2740, mae: 0.3567, huber: 0.1287, swd: 0.1183, ept: 68.8179
    Epoch [11/50], Test Losses: mse: 0.2039, mae: 0.3067, huber: 0.0960, swd: 0.0865, ept: 74.3302
      Epoch 11 composite train-obj: 0.083151
    Epoch [11/50], Test Losses: mse: 0.1823, mae: 0.2924, huber: 0.0867, swd: 0.0752, ept: 75.2666
    Best round's Test MSE: 0.1823, MAE: 0.2924, SWD: 0.0752
    Best round's Validation MSE: 0.2671, MAE: 0.3545, SWD: 0.1178
    Best round's Test verification MSE : 0.1823, MAE: 0.2924, SWD: 0.0752
    Time taken: 31.00 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4088, mae: 0.4006, huber: 0.1593, swd: 0.2107, ept: 69.1742
    Epoch [1/50], Val Losses: mse: 0.2991, mae: 0.3813, huber: 0.1401, swd: 0.1406, ept: 63.8723
    Epoch [1/50], Test Losses: mse: 0.1957, mae: 0.3122, huber: 0.0938, swd: 0.0808, ept: 73.8886
      Epoch 1 composite train-obj: 0.159280
            Val objective improved inf → 0.1401, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2944, mae: 0.3310, huber: 0.1160, swd: 0.1222, ept: 76.7708
    Epoch [2/50], Val Losses: mse: 0.2825, mae: 0.3678, huber: 0.1327, swd: 0.1268, ept: 65.7860
    Epoch [2/50], Test Losses: mse: 0.1873, mae: 0.3035, huber: 0.0897, swd: 0.0722, ept: 74.6356
      Epoch 2 composite train-obj: 0.115982
            Val objective improved 0.1401 → 0.1327, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2699, mae: 0.3132, huber: 0.1066, swd: 0.1083, ept: 78.8293
    Epoch [3/50], Val Losses: mse: 0.2691, mae: 0.3549, huber: 0.1268, swd: 0.1150, ept: 67.3521
    Epoch [3/50], Test Losses: mse: 0.1786, mae: 0.2915, huber: 0.0853, swd: 0.0660, ept: 75.7892
      Epoch 3 composite train-obj: 0.106600
            Val objective improved 0.1327 → 0.1268, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2547, mae: 0.3044, huber: 0.1015, swd: 0.1013, ept: 80.0899
    Epoch [4/50], Val Losses: mse: 0.2848, mae: 0.3677, huber: 0.1330, swd: 0.1337, ept: 67.5154
    Epoch [4/50], Test Losses: mse: 0.1831, mae: 0.2989, huber: 0.0876, swd: 0.0747, ept: 75.1300
      Epoch 4 composite train-obj: 0.101510
            No improvement (0.1330), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.2417, mae: 0.2979, huber: 0.0976, swd: 0.0953, ept: 80.7835
    Epoch [5/50], Val Losses: mse: 0.2713, mae: 0.3552, huber: 0.1272, swd: 0.1192, ept: 67.8219
    Epoch [5/50], Test Losses: mse: 0.1806, mae: 0.2909, huber: 0.0859, swd: 0.0706, ept: 75.4301
      Epoch 5 composite train-obj: 0.097601
            No improvement (0.1272), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.2325, mae: 0.2945, huber: 0.0950, swd: 0.0902, ept: 81.0964
    Epoch [6/50], Val Losses: mse: 0.2661, mae: 0.3526, huber: 0.1252, swd: 0.1152, ept: 68.8578
    Epoch [6/50], Test Losses: mse: 0.1807, mae: 0.2930, huber: 0.0861, swd: 0.0702, ept: 76.0077
      Epoch 6 composite train-obj: 0.094955
            Val objective improved 0.1268 → 0.1252, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.2247, mae: 0.2902, huber: 0.0924, swd: 0.0871, ept: 81.5449
    Epoch [7/50], Val Losses: mse: 0.2595, mae: 0.3500, huber: 0.1228, swd: 0.1102, ept: 69.0532
    Epoch [7/50], Test Losses: mse: 0.1809, mae: 0.2918, huber: 0.0861, swd: 0.0692, ept: 76.2293
      Epoch 7 composite train-obj: 0.092443
            Val objective improved 0.1252 → 0.1228, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.2189, mae: 0.2880, huber: 0.0906, swd: 0.0849, ept: 81.7737
    Epoch [8/50], Val Losses: mse: 0.2715, mae: 0.3587, huber: 0.1280, swd: 0.1112, ept: 68.2064
    Epoch [8/50], Test Losses: mse: 0.1784, mae: 0.2927, huber: 0.0855, swd: 0.0689, ept: 75.7640
      Epoch 8 composite train-obj: 0.090614
            No improvement (0.1280), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.2145, mae: 0.2854, huber: 0.0890, swd: 0.0832, ept: 81.9692
    Epoch [9/50], Val Losses: mse: 0.2772, mae: 0.3592, huber: 0.1290, swd: 0.1174, ept: 68.8660
    Epoch [9/50], Test Losses: mse: 0.1845, mae: 0.2937, huber: 0.0880, swd: 0.0708, ept: 75.1283
      Epoch 9 composite train-obj: 0.089025
            No improvement (0.1290), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.2101, mae: 0.2829, huber: 0.0875, swd: 0.0810, ept: 82.2422
    Epoch [10/50], Val Losses: mse: 0.2660, mae: 0.3541, huber: 0.1254, swd: 0.1176, ept: 68.7013
    Epoch [10/50], Test Losses: mse: 0.1940, mae: 0.3043, huber: 0.0921, swd: 0.0811, ept: 75.1827
      Epoch 10 composite train-obj: 0.087491
            No improvement (0.1254), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.2041, mae: 0.2793, huber: 0.0852, swd: 0.0783, ept: 82.5096
    Epoch [11/50], Val Losses: mse: 0.2697, mae: 0.3544, huber: 0.1267, swd: 0.1119, ept: 68.6485
    Epoch [11/50], Test Losses: mse: 0.1909, mae: 0.2997, huber: 0.0910, swd: 0.0745, ept: 74.9138
      Epoch 11 composite train-obj: 0.085243
            No improvement (0.1267), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.2003, mae: 0.2774, huber: 0.0839, swd: 0.0757, ept: 82.6983
    Epoch [12/50], Val Losses: mse: 0.2836, mae: 0.3618, huber: 0.1317, swd: 0.1158, ept: 68.1006
    Epoch [12/50], Test Losses: mse: 0.1955, mae: 0.2986, huber: 0.0925, swd: 0.0766, ept: 74.6406
      Epoch 12 composite train-obj: 0.083856
    Epoch [12/50], Test Losses: mse: 0.1809, mae: 0.2918, huber: 0.0861, swd: 0.0692, ept: 76.2293
    Best round's Test MSE: 0.1809, MAE: 0.2918, SWD: 0.0692
    Best round's Validation MSE: 0.2595, MAE: 0.3500, SWD: 0.1102
    Best round's Test verification MSE : 0.1809, MAE: 0.2918, SWD: 0.0692
    Time taken: 34.17 seconds
    
    ==================================================
    Experiment Summary (ACL_etth2_seq96_pred96_20250510_2044)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.1794 ± 0.0031
      mae: 0.2914 ± 0.0011
      huber: 0.0856 ± 0.0012
      swd: 0.0728 ± 0.0026
      ept: 75.8978 ± 0.4495
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.2649 ± 0.0039
      mae: 0.3532 ± 0.0022
      huber: 0.1251 ± 0.0017
      swd: 0.1170 ± 0.0052
      ept: 68.2435 ± 0.7495
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 96.70 seconds
    
    Experiment complete: ACL_etth2_seq96_pred96_20250510_2044
    Model: ACL
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['etth2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 94
    Validation Batches: 13
    Test Batches: 26
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 104.5439, mae: 5.3031, huber: 4.8660, swd: 78.8662, ept: 55.2954
    Epoch [1/50], Val Losses: mse: 25.7177, mae: 3.2712, huber: 2.8419, swd: 9.9137, ept: 62.2779
    Epoch [1/50], Test Losses: mse: 18.5849, mae: 2.7978, huber: 2.3631, swd: 8.2631, ept: 71.3306
      Epoch 1 composite train-obj: 4.865982
            Val objective improved inf → 2.8419, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 24.4939, mae: 2.8512, huber: 2.4285, swd: 11.7182, ept: 74.8311
    Epoch [2/50], Val Losses: mse: 24.6260, mae: 3.1856, huber: 2.7596, swd: 10.3376, ept: 64.1520
    Epoch [2/50], Test Losses: mse: 16.7892, mae: 2.6327, huber: 2.2024, swd: 7.6791, ept: 73.0352
      Epoch 2 composite train-obj: 2.428538
            Val objective improved 2.8419 → 2.7596, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 23.1671, mae: 2.7411, huber: 2.3201, swd: 11.4325, ept: 76.7798
    Epoch [3/50], Val Losses: mse: 22.4973, mae: 3.0557, huber: 2.6298, swd: 9.1409, ept: 65.8833
    Epoch [3/50], Test Losses: mse: 16.0944, mae: 2.5716, huber: 2.1422, swd: 7.5560, ept: 73.6238
      Epoch 3 composite train-obj: 2.320132
            Val objective improved 2.7596 → 2.6298, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.0070, mae: 2.6280, huber: 2.2093, swd: 11.0092, ept: 78.5357
    Epoch [4/50], Val Losses: mse: 22.6274, mae: 3.0323, huber: 2.6067, swd: 9.6152, ept: 66.1752
    Epoch [4/50], Test Losses: mse: 16.2704, mae: 2.5794, huber: 2.1488, swd: 8.0361, ept: 73.6134
      Epoch 4 composite train-obj: 2.209257
            Val objective improved 2.6298 → 2.6067, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 21.4380, mae: 2.5914, huber: 2.1736, swd: 10.7012, ept: 79.0041
    Epoch [5/50], Val Losses: mse: 22.8172, mae: 3.0520, huber: 2.6250, swd: 9.5419, ept: 66.3843
    Epoch [5/50], Test Losses: mse: 18.1866, mae: 2.7439, huber: 2.3104, swd: 9.6860, ept: 70.4926
      Epoch 5 composite train-obj: 2.173585
            No improvement (2.6250), counter 1/5
    Epoch [6/50], Train Losses: mse: 21.3183, mae: 2.5757, huber: 2.1589, swd: 10.7022, ept: 79.1017
    Epoch [6/50], Val Losses: mse: 21.6137, mae: 2.9675, huber: 2.5422, swd: 8.9222, ept: 67.0595
    Epoch [6/50], Test Losses: mse: 16.0579, mae: 2.5470, huber: 2.1183, swd: 8.0156, ept: 73.5210
      Epoch 6 composite train-obj: 2.158875
            Val objective improved 2.6067 → 2.5422, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 20.5880, mae: 2.5103, huber: 2.0951, swd: 10.0811, ept: 79.6875
    Epoch [7/50], Val Losses: mse: 21.0425, mae: 2.9523, huber: 2.5255, swd: 8.6671, ept: 67.7406
    Epoch [7/50], Test Losses: mse: 15.2047, mae: 2.5004, huber: 2.0704, swd: 7.3045, ept: 74.4097
      Epoch 7 composite train-obj: 2.095076
            Val objective improved 2.5422 → 2.5255, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 20.7373, mae: 2.5435, huber: 2.1275, swd: 10.2409, ept: 79.3212
    Epoch [8/50], Val Losses: mse: 21.0915, mae: 2.9467, huber: 2.5211, swd: 8.8223, ept: 67.7306
    Epoch [8/50], Test Losses: mse: 15.1182, mae: 2.4900, huber: 2.0608, swd: 7.2552, ept: 74.8674
      Epoch 8 composite train-obj: 2.127513
            Val objective improved 2.5255 → 2.5211, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 20.3462, mae: 2.5040, huber: 2.0894, swd: 9.9113, ept: 79.7298
    Epoch [9/50], Val Losses: mse: 20.5943, mae: 2.9021, huber: 2.4782, swd: 8.4246, ept: 68.1738
    Epoch [9/50], Test Losses: mse: 15.1963, mae: 2.4629, huber: 2.0370, swd: 7.3243, ept: 74.4578
      Epoch 9 composite train-obj: 2.089380
            Val objective improved 2.5211 → 2.4782, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 20.1581, mae: 2.4901, huber: 2.0759, swd: 9.8249, ept: 79.7560
    Epoch [10/50], Val Losses: mse: 20.2096, mae: 2.9149, huber: 2.4882, swd: 8.3588, ept: 68.4675
    Epoch [10/50], Test Losses: mse: 14.9569, mae: 2.5100, huber: 2.0775, swd: 7.2478, ept: 75.5587
      Epoch 10 composite train-obj: 2.075862
            No improvement (2.4882), counter 1/5
    Epoch [11/50], Train Losses: mse: 20.0489, mae: 2.4909, huber: 2.0766, swd: 9.7544, ept: 79.8406
    Epoch [11/50], Val Losses: mse: 21.8976, mae: 3.0271, huber: 2.6015, swd: 10.2074, ept: 67.8453
    Epoch [11/50], Test Losses: mse: 15.4237, mae: 2.5998, huber: 2.1637, swd: 7.6649, ept: 75.1768
      Epoch 11 composite train-obj: 2.076623
            No improvement (2.6015), counter 2/5
    Epoch [12/50], Train Losses: mse: 19.9221, mae: 2.4855, huber: 2.0714, swd: 9.7041, ept: 79.8777
    Epoch [12/50], Val Losses: mse: 20.6231, mae: 2.9043, huber: 2.4815, swd: 8.6104, ept: 68.8920
    Epoch [12/50], Test Losses: mse: 15.0045, mae: 2.4639, huber: 2.0363, swd: 7.2643, ept: 74.6856
      Epoch 12 composite train-obj: 2.071379
            No improvement (2.4815), counter 3/5
    Epoch [13/50], Train Losses: mse: 19.4470, mae: 2.4386, huber: 2.0259, swd: 9.3127, ept: 80.2279
    Epoch [13/50], Val Losses: mse: 20.7091, mae: 2.9329, huber: 2.5119, swd: 8.5062, ept: 68.5581
    Epoch [13/50], Test Losses: mse: 15.5852, mae: 2.5147, huber: 2.0863, swd: 7.6847, ept: 73.7781
      Epoch 13 composite train-obj: 2.025909
            No improvement (2.5119), counter 4/5
    Epoch [14/50], Train Losses: mse: 19.2668, mae: 2.4397, huber: 2.0264, swd: 9.1549, ept: 80.2236
    Epoch [14/50], Val Losses: mse: 21.1564, mae: 2.9453, huber: 2.5236, swd: 9.1941, ept: 68.7373
    Epoch [14/50], Test Losses: mse: 15.1250, mae: 2.5146, huber: 2.0826, swd: 7.4593, ept: 75.1630
      Epoch 14 composite train-obj: 2.026446
    Epoch [14/50], Test Losses: mse: 15.1964, mae: 2.4630, huber: 2.0370, swd: 7.3245, ept: 74.4616
    Best round's Test MSE: 15.1963, MAE: 2.4629, SWD: 7.3243
    Best round's Validation MSE: 20.5943, MAE: 2.9021, SWD: 8.4246
    Best round's Test verification MSE : 15.1964, MAE: 2.4630, SWD: 7.3245
    Time taken: 38.26 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 103.7838, mae: 5.3114, huber: 4.8732, swd: 78.3474, ept: 55.9021
    Epoch [1/50], Val Losses: mse: 25.5700, mae: 3.2722, huber: 2.8426, swd: 9.6070, ept: 62.5731
    Epoch [1/50], Test Losses: mse: 18.3188, mae: 2.7861, huber: 2.3514, swd: 7.8047, ept: 71.2661
      Epoch 1 composite train-obj: 4.873162
            Val objective improved inf → 2.8426, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 25.1618, mae: 2.8864, huber: 2.4622, swd: 12.2857, ept: 74.4750
    Epoch [2/50], Val Losses: mse: 24.0666, mae: 3.1497, huber: 2.7223, swd: 8.8712, ept: 64.1230
    Epoch [2/50], Test Losses: mse: 17.9030, mae: 2.7334, huber: 2.3007, swd: 7.9810, ept: 71.2770
      Epoch 2 composite train-obj: 2.462231
            Val objective improved 2.8426 → 2.7223, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 23.1754, mae: 2.7076, huber: 2.2869, swd: 11.3213, ept: 77.3457
    Epoch [3/50], Val Losses: mse: 22.2256, mae: 3.0282, huber: 2.6019, swd: 8.3842, ept: 66.5321
    Epoch [3/50], Test Losses: mse: 16.8931, mae: 2.6283, huber: 2.1979, swd: 7.9579, ept: 72.5288
      Epoch 3 composite train-obj: 2.286884
            Val objective improved 2.7223 → 2.6019, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.3991, mae: 2.6479, huber: 2.2288, swd: 10.9714, ept: 78.4177
    Epoch [4/50], Val Losses: mse: 21.6743, mae: 2.9848, huber: 2.5597, swd: 8.3855, ept: 66.9590
    Epoch [4/50], Test Losses: mse: 15.6942, mae: 2.5261, huber: 2.0980, swd: 7.2483, ept: 73.7644
      Epoch 4 composite train-obj: 2.228848
            Val objective improved 2.6019 → 2.5597, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 21.7388, mae: 2.6004, huber: 2.1828, swd: 10.5320, ept: 78.9356
    Epoch [5/50], Val Losses: mse: 20.8286, mae: 2.9473, huber: 2.5232, swd: 7.9759, ept: 67.6605
    Epoch [5/50], Test Losses: mse: 15.1227, mae: 2.4827, huber: 2.0553, swd: 6.8990, ept: 74.8312
      Epoch 5 composite train-obj: 2.182822
            Val objective improved 2.5597 → 2.5232, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 21.5087, mae: 2.5925, huber: 2.1750, swd: 10.3684, ept: 79.0323
    Epoch [6/50], Val Losses: mse: 20.9081, mae: 2.9456, huber: 2.5222, swd: 8.1788, ept: 67.4246
    Epoch [6/50], Test Losses: mse: 15.0775, mae: 2.4835, huber: 2.0558, swd: 6.9936, ept: 75.0797
      Epoch 6 composite train-obj: 2.175004
            Val objective improved 2.5232 → 2.5222, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 20.9653, mae: 2.5536, huber: 2.1377, swd: 9.9665, ept: 79.3057
    Epoch [7/50], Val Losses: mse: 20.5509, mae: 2.9137, huber: 2.4892, swd: 7.7818, ept: 68.0788
    Epoch [7/50], Test Losses: mse: 15.3488, mae: 2.5271, huber: 2.0957, swd: 7.1938, ept: 74.2694
      Epoch 7 composite train-obj: 2.137747
            Val objective improved 2.5222 → 2.4892, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 20.7505, mae: 2.5393, huber: 2.1237, swd: 9.8571, ept: 79.3843
    Epoch [8/50], Val Losses: mse: 19.9690, mae: 2.8809, huber: 2.4582, swd: 7.3305, ept: 68.4113
    Epoch [8/50], Test Losses: mse: 15.2890, mae: 2.4742, huber: 2.0478, swd: 7.1368, ept: 74.9407
      Epoch 8 composite train-obj: 2.123657
            Val objective improved 2.4892 → 2.4582, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 20.4784, mae: 2.5124, huber: 2.0979, swd: 9.7120, ept: 79.6274
    Epoch [9/50], Val Losses: mse: 20.8327, mae: 2.9267, huber: 2.5026, swd: 8.1540, ept: 68.2590
    Epoch [9/50], Test Losses: mse: 15.2191, mae: 2.4763, huber: 2.0491, swd: 7.0664, ept: 73.9939
      Epoch 9 composite train-obj: 2.097859
            No improvement (2.5026), counter 1/5
    Epoch [10/50], Train Losses: mse: 20.5366, mae: 2.5254, huber: 2.1105, swd: 9.8899, ept: 79.4633
    Epoch [10/50], Val Losses: mse: 21.2485, mae: 2.9238, huber: 2.5015, swd: 8.4447, ept: 68.2743
    Epoch [10/50], Test Losses: mse: 15.9111, mae: 2.5170, huber: 2.0895, swd: 7.6555, ept: 73.8569
      Epoch 10 composite train-obj: 2.110470
            No improvement (2.5015), counter 2/5
    Epoch [11/50], Train Losses: mse: 20.2259, mae: 2.5001, huber: 2.0860, swd: 9.6361, ept: 79.7400
    Epoch [11/50], Val Losses: mse: 20.3407, mae: 2.9218, huber: 2.4992, swd: 7.7139, ept: 68.6448
    Epoch [11/50], Test Losses: mse: 15.2000, mae: 2.4689, huber: 2.0428, swd: 7.0004, ept: 74.3979
      Epoch 11 composite train-obj: 2.085958
            No improvement (2.4992), counter 3/5
    Epoch [12/50], Train Losses: mse: 19.6442, mae: 2.4515, huber: 2.0380, swd: 9.1911, ept: 80.1618
    Epoch [12/50], Val Losses: mse: 20.4976, mae: 2.8918, huber: 2.4703, swd: 7.8293, ept: 68.6621
    Epoch [12/50], Test Losses: mse: 16.1597, mae: 2.5477, huber: 2.1191, swd: 7.8638, ept: 72.6279
      Epoch 12 composite train-obj: 2.038040
            No improvement (2.4703), counter 4/5
    Epoch [13/50], Train Losses: mse: 19.6288, mae: 2.4570, huber: 2.0437, swd: 9.2119, ept: 80.0836
    Epoch [13/50], Val Losses: mse: 21.0756, mae: 2.9610, huber: 2.5398, swd: 8.6688, ept: 69.0148
    Epoch [13/50], Test Losses: mse: 14.6394, mae: 2.4681, huber: 2.0386, swd: 6.6894, ept: 75.4581
      Epoch 13 composite train-obj: 2.043669
    Epoch [13/50], Test Losses: mse: 15.2887, mae: 2.4741, huber: 2.0477, swd: 7.1366, ept: 74.9421
    Best round's Test MSE: 15.2890, MAE: 2.4742, SWD: 7.1368
    Best round's Validation MSE: 19.9690, MAE: 2.8809, SWD: 7.3305
    Best round's Test verification MSE : 15.2887, MAE: 2.4741, SWD: 7.1366
    Time taken: 36.28 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 115.7968, mae: 5.6072, huber: 5.1679, swd: 80.4340, ept: 54.1963
    Epoch [1/50], Val Losses: mse: 24.8181, mae: 3.2430, huber: 2.8142, swd: 8.9017, ept: 63.4163
    Epoch [1/50], Test Losses: mse: 17.7405, mae: 2.7611, huber: 2.3250, swd: 7.0599, ept: 72.1682
      Epoch 1 composite train-obj: 5.167939
            Val objective improved inf → 2.8142, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 24.8619, mae: 2.8602, huber: 2.4364, swd: 10.9206, ept: 74.6435
    Epoch [2/50], Val Losses: mse: 23.3287, mae: 3.1214, huber: 2.6945, swd: 7.8007, ept: 64.8112
    Epoch [2/50], Test Losses: mse: 17.3807, mae: 2.6916, huber: 2.2603, swd: 7.0541, ept: 72.3889
      Epoch 2 composite train-obj: 2.436436
            Val objective improved 2.8142 → 2.6945, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 23.3174, mae: 2.7365, huber: 2.3151, swd: 10.5117, ept: 76.8619
    Epoch [3/50], Val Losses: mse: 21.8589, mae: 3.0187, huber: 2.5934, swd: 8.2865, ept: 66.7107
    Epoch [3/50], Test Losses: mse: 15.6929, mae: 2.5439, huber: 2.1153, swd: 6.8377, ept: 74.8727
      Epoch 3 composite train-obj: 2.315112
            Val objective improved 2.6945 → 2.5934, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.3161, mae: 2.6570, huber: 2.2374, swd: 10.1376, ept: 78.2199
    Epoch [4/50], Val Losses: mse: 22.2784, mae: 3.0128, huber: 2.5866, swd: 8.0849, ept: 66.4577
    Epoch [4/50], Test Losses: mse: 16.7898, mae: 2.6238, huber: 2.1930, swd: 7.4957, ept: 72.8154
      Epoch 4 composite train-obj: 2.237357
            Val objective improved 2.5934 → 2.5866, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 21.4499, mae: 2.5728, huber: 2.1557, swd: 9.5898, ept: 79.2019
    Epoch [5/50], Val Losses: mse: 22.2454, mae: 3.0107, huber: 2.5856, swd: 8.1677, ept: 66.6828
    Epoch [5/50], Test Losses: mse: 17.1400, mae: 2.6251, huber: 2.1966, swd: 7.8963, ept: 72.5774
      Epoch 5 composite train-obj: 2.155706
            Val objective improved 2.5866 → 2.5856, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 21.4571, mae: 2.5947, huber: 2.1767, swd: 9.6465, ept: 78.7747
    Epoch [6/50], Val Losses: mse: 22.5728, mae: 3.0269, huber: 2.6021, swd: 8.5297, ept: 66.4831
    Epoch [6/50], Test Losses: mse: 17.3916, mae: 2.6532, huber: 2.2232, swd: 8.1545, ept: 71.8304
      Epoch 6 composite train-obj: 2.176664
            No improvement (2.6021), counter 1/5
    Epoch [7/50], Train Losses: mse: 20.9322, mae: 2.5469, huber: 2.1306, swd: 9.2937, ept: 79.2251
    Epoch [7/50], Val Losses: mse: 20.9636, mae: 2.9518, huber: 2.5269, swd: 8.0262, ept: 68.0310
    Epoch [7/50], Test Losses: mse: 14.7347, mae: 2.4765, huber: 2.0467, swd: 6.3410, ept: 75.8388
      Epoch 7 composite train-obj: 2.130626
            Val objective improved 2.5856 → 2.5269, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 20.7387, mae: 2.5405, huber: 2.1245, swd: 9.1560, ept: 79.2409
    Epoch [8/50], Val Losses: mse: 20.5033, mae: 2.9183, huber: 2.4940, swd: 7.4236, ept: 68.0356
    Epoch [8/50], Test Losses: mse: 14.8756, mae: 2.4609, huber: 2.0333, swd: 6.3069, ept: 75.0118
      Epoch 8 composite train-obj: 2.124472
            Val objective improved 2.5269 → 2.4940, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 20.4052, mae: 2.5088, huber: 2.0942, swd: 8.9640, ept: 79.5522
    Epoch [9/50], Val Losses: mse: 20.7841, mae: 2.9281, huber: 2.5030, swd: 7.5239, ept: 67.9853
    Epoch [9/50], Test Losses: mse: 15.4308, mae: 2.5113, huber: 2.0811, swd: 6.7315, ept: 73.6284
      Epoch 9 composite train-obj: 2.094180
            No improvement (2.5030), counter 1/5
    Epoch [10/50], Train Losses: mse: 20.1509, mae: 2.4874, huber: 2.0738, swd: 8.8239, ept: 79.7298
    Epoch [10/50], Val Losses: mse: 20.6293, mae: 2.9162, huber: 2.4918, swd: 7.3409, ept: 68.3702
    Epoch [10/50], Test Losses: mse: 15.5533, mae: 2.5095, huber: 2.0810, swd: 6.7813, ept: 73.7171
      Epoch 10 composite train-obj: 2.073806
            Val objective improved 2.4940 → 2.4918, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 19.8506, mae: 2.4647, huber: 2.0517, swd: 8.6369, ept: 80.0459
    Epoch [11/50], Val Losses: mse: 22.8456, mae: 3.0689, huber: 2.6426, swd: 8.6399, ept: 66.5951
    Epoch [11/50], Test Losses: mse: 17.6300, mae: 2.6798, huber: 2.2498, swd: 8.2235, ept: 70.4249
      Epoch 11 composite train-obj: 2.051670
            No improvement (2.6426), counter 1/5
    Epoch [12/50], Train Losses: mse: 20.1522, mae: 2.5018, huber: 2.0874, swd: 8.8678, ept: 79.5320
    Epoch [12/50], Val Losses: mse: 20.8180, mae: 2.9345, huber: 2.5097, swd: 7.3681, ept: 67.8412
    Epoch [12/50], Test Losses: mse: 16.1903, mae: 2.5602, huber: 2.1307, swd: 7.2666, ept: 72.8696
      Epoch 12 composite train-obj: 2.087437
            No improvement (2.5097), counter 2/5
    Epoch [13/50], Train Losses: mse: 19.6747, mae: 2.4604, huber: 2.0467, swd: 8.5432, ept: 80.0511
    Epoch [13/50], Val Losses: mse: 20.7508, mae: 2.9354, huber: 2.5117, swd: 7.4063, ept: 67.9476
    Epoch [13/50], Test Losses: mse: 15.5532, mae: 2.5006, huber: 2.0725, swd: 6.7433, ept: 74.1242
      Epoch 13 composite train-obj: 2.046736
            No improvement (2.5117), counter 3/5
    Epoch [14/50], Train Losses: mse: 19.3945, mae: 2.4359, huber: 2.0234, swd: 8.3372, ept: 80.2657
    Epoch [14/50], Val Losses: mse: 21.5551, mae: 2.9795, huber: 2.5584, swd: 7.9141, ept: 67.6720
    Epoch [14/50], Test Losses: mse: 16.4934, mae: 2.5504, huber: 2.1249, swd: 7.5442, ept: 72.5776
      Epoch 14 composite train-obj: 2.023446
            No improvement (2.5584), counter 4/5
    Epoch [15/50], Train Losses: mse: 19.2471, mae: 2.4362, huber: 2.0232, swd: 8.2440, ept: 80.2088
    Epoch [15/50], Val Losses: mse: 20.8704, mae: 2.9724, huber: 2.5505, swd: 7.5494, ept: 67.9366
    Epoch [15/50], Test Losses: mse: 15.1641, mae: 2.4774, huber: 2.0509, swd: 6.5372, ept: 74.4014
      Epoch 15 composite train-obj: 2.023182
    Epoch [15/50], Test Losses: mse: 15.5530, mae: 2.5095, huber: 2.0809, swd: 6.7810, ept: 73.7242
    Best round's Test MSE: 15.5533, MAE: 2.5095, SWD: 6.7813
    Best round's Validation MSE: 20.6293, MAE: 2.9162, SWD: 7.3409
    Best round's Test verification MSE : 15.5530, MAE: 2.5095, SWD: 6.7810
    Time taken: 41.27 seconds
    
    ==================================================
    Experiment Summary (ACL_etth2_seq96_pred96_20250511_1605)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 15.3462 ± 0.1512
      mae: 2.4822 ± 0.0199
      huber: 2.0552 ± 0.0187
      swd: 7.0808 ± 0.2252
      ept: 74.3719 ± 0.5032
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 20.3975 ± 0.3034
      mae: 2.8997 ± 0.0145
      huber: 2.4761 ± 0.0138
      swd: 7.6987 ± 0.5133
      ept: 68.3184 ± 0.1036
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 117.50 seconds
    
    Experiment complete: ACL_etth2_seq96_pred96_20250511_1605
    Model: ACL
    Dataset: etth2
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
    channels=data_mgr.datasets['etth2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.4720, mae: 0.4323, huber: 0.1807, swd: 0.2764, ept: 118.6528
    Epoch [1/50], Val Losses: mse: 0.3878, mae: 0.4462, huber: 0.1779, swd: 0.2263, ept: 98.7241
    Epoch [1/50], Test Losses: mse: 0.2292, mae: 0.3417, huber: 0.1100, swd: 0.1214, ept: 122.5559
      Epoch 1 composite train-obj: 0.180666
            Val objective improved inf → 0.1779, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3679, mae: 0.3718, huber: 0.1418, swd: 0.1803, ept: 132.6483
    Epoch [2/50], Val Losses: mse: 0.3599, mae: 0.4218, huber: 0.1663, swd: 0.1899, ept: 101.3173
    Epoch [2/50], Test Losses: mse: 0.2171, mae: 0.3296, huber: 0.1042, swd: 0.0973, ept: 126.1831
      Epoch 2 composite train-obj: 0.141831
            Val objective improved 0.1779 → 0.1663, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3389, mae: 0.3516, huber: 0.1305, swd: 0.1618, ept: 137.3802
    Epoch [3/50], Val Losses: mse: 0.3558, mae: 0.4167, huber: 0.1639, swd: 0.1941, ept: 107.6345
    Epoch [3/50], Test Losses: mse: 0.2124, mae: 0.3286, huber: 0.1021, swd: 0.0988, ept: 128.1095
      Epoch 3 composite train-obj: 0.130513
            Val objective improved 0.1663 → 0.1639, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3204, mae: 0.3403, huber: 0.1237, swd: 0.1514, ept: 142.1905
    Epoch [4/50], Val Losses: mse: 0.3478, mae: 0.4083, huber: 0.1606, swd: 0.1713, ept: 108.5758
    Epoch [4/50], Test Losses: mse: 0.2087, mae: 0.3217, huber: 0.1000, swd: 0.0910, ept: 130.5354
      Epoch 4 composite train-obj: 0.123730
            Val objective improved 0.1639 → 0.1606, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3116, mae: 0.3354, huber: 0.1206, swd: 0.1465, ept: 143.6657
    Epoch [5/50], Val Losses: mse: 0.3412, mae: 0.4079, huber: 0.1580, swd: 0.1708, ept: 110.6021
    Epoch [5/50], Test Losses: mse: 0.2128, mae: 0.3306, huber: 0.1025, swd: 0.0996, ept: 131.6593
      Epoch 5 composite train-obj: 0.120598
            Val objective improved 0.1606 → 0.1580, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3043, mae: 0.3309, huber: 0.1179, swd: 0.1419, ept: 144.7027
    Epoch [6/50], Val Losses: mse: 0.3497, mae: 0.4109, huber: 0.1616, swd: 0.1727, ept: 109.9336
    Epoch [6/50], Test Losses: mse: 0.2159, mae: 0.3308, huber: 0.1036, swd: 0.1007, ept: 131.2968
      Epoch 6 composite train-obj: 0.117898
            No improvement (0.1616), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.2987, mae: 0.3284, huber: 0.1161, swd: 0.1395, ept: 145.4220
    Epoch [7/50], Val Losses: mse: 0.3552, mae: 0.4124, huber: 0.1634, swd: 0.1770, ept: 110.2837
    Epoch [7/50], Test Losses: mse: 0.2216, mae: 0.3357, huber: 0.1062, swd: 0.1040, ept: 131.1141
      Epoch 7 composite train-obj: 0.116056
            No improvement (0.1634), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.2929, mae: 0.3250, huber: 0.1140, swd: 0.1364, ept: 145.9790
    Epoch [8/50], Val Losses: mse: 0.3524, mae: 0.4115, huber: 0.1620, swd: 0.1772, ept: 110.8876
    Epoch [8/50], Test Losses: mse: 0.2177, mae: 0.3312, huber: 0.1044, swd: 0.1011, ept: 130.9284
      Epoch 8 composite train-obj: 0.113996
            No improvement (0.1620), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.2882, mae: 0.3230, huber: 0.1125, swd: 0.1334, ept: 146.4367
    Epoch [9/50], Val Losses: mse: 0.3616, mae: 0.4170, huber: 0.1650, swd: 0.1923, ept: 112.2428
    Epoch [9/50], Test Losses: mse: 0.2338, mae: 0.3445, huber: 0.1113, swd: 0.1132, ept: 130.6534
      Epoch 9 composite train-obj: 0.112518
            No improvement (0.1650), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.2840, mae: 0.3215, huber: 0.1112, swd: 0.1314, ept: 146.7902
    Epoch [10/50], Val Losses: mse: 0.3539, mae: 0.4112, huber: 0.1626, swd: 0.1669, ept: 110.3166
    Epoch [10/50], Test Losses: mse: 0.2275, mae: 0.3362, huber: 0.1082, swd: 0.0991, ept: 130.1014
      Epoch 10 composite train-obj: 0.111205
    Epoch [10/50], Test Losses: mse: 0.2128, mae: 0.3306, huber: 0.1025, swd: 0.0996, ept: 131.6620
    Best round's Test MSE: 0.2128, MAE: 0.3306, SWD: 0.0996
    Best round's Validation MSE: 0.3412, MAE: 0.4079, SWD: 0.1708
    Best round's Test verification MSE : 0.2128, MAE: 0.3306, SWD: 0.0996
    Time taken: 27.70 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4678, mae: 0.4312, huber: 0.1797, swd: 0.2746, ept: 119.3058
    Epoch [1/50], Val Losses: mse: 0.3799, mae: 0.4404, huber: 0.1748, swd: 0.2154, ept: 99.3690
    Epoch [1/50], Test Losses: mse: 0.2271, mae: 0.3389, huber: 0.1089, swd: 0.1175, ept: 123.5997
      Epoch 1 composite train-obj: 0.179731
            Val objective improved inf → 0.1748, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3659, mae: 0.3699, huber: 0.1408, swd: 0.1796, ept: 132.6833
    Epoch [2/50], Val Losses: mse: 0.3722, mae: 0.4323, huber: 0.1715, swd: 0.2104, ept: 100.0716
    Epoch [2/50], Test Losses: mse: 0.2203, mae: 0.3359, huber: 0.1059, swd: 0.1082, ept: 124.1250
      Epoch 2 composite train-obj: 0.140825
            Val objective improved 0.1748 → 0.1715, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3397, mae: 0.3516, huber: 0.1307, swd: 0.1619, ept: 137.8078
    Epoch [3/50], Val Losses: mse: 0.3355, mae: 0.4033, huber: 0.1563, swd: 0.1619, ept: 107.2126
    Epoch [3/50], Test Losses: mse: 0.2095, mae: 0.3189, huber: 0.1002, swd: 0.0891, ept: 129.5461
      Epoch 3 composite train-obj: 0.130662
            Val objective improved 0.1715 → 0.1563, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3225, mae: 0.3400, huber: 0.1242, swd: 0.1531, ept: 141.9614
    Epoch [4/50], Val Losses: mse: 0.3348, mae: 0.4019, huber: 0.1558, swd: 0.1603, ept: 108.8980
    Epoch [4/50], Test Losses: mse: 0.2120, mae: 0.3267, huber: 0.1018, swd: 0.0972, ept: 130.1519
      Epoch 4 composite train-obj: 0.124177
            Val objective improved 0.1563 → 0.1558, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3148, mae: 0.3364, huber: 0.1215, swd: 0.1492, ept: 143.1537
    Epoch [5/50], Val Losses: mse: 0.3365, mae: 0.4045, huber: 0.1568, swd: 0.1601, ept: 109.2941
    Epoch [5/50], Test Losses: mse: 0.2110, mae: 0.3265, huber: 0.1015, swd: 0.0972, ept: 130.8361
      Epoch 5 composite train-obj: 0.121476
            No improvement (0.1568), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3071, mae: 0.3326, huber: 0.1189, swd: 0.1452, ept: 143.9013
    Epoch [6/50], Val Losses: mse: 0.3311, mae: 0.3999, huber: 0.1542, swd: 0.1592, ept: 110.0073
    Epoch [6/50], Test Losses: mse: 0.2136, mae: 0.3267, huber: 0.1024, swd: 0.1019, ept: 130.8138
      Epoch 6 composite train-obj: 0.118923
            Val objective improved 0.1558 → 0.1542, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3010, mae: 0.3288, huber: 0.1166, swd: 0.1415, ept: 144.7253
    Epoch [7/50], Val Losses: mse: 0.3481, mae: 0.4128, huber: 0.1607, swd: 0.1904, ept: 109.2871
    Epoch [7/50], Test Losses: mse: 0.2371, mae: 0.3484, huber: 0.1134, swd: 0.1290, ept: 127.2744
      Epoch 7 composite train-obj: 0.116603
            No improvement (0.1607), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.2958, mae: 0.3264, huber: 0.1149, swd: 0.1394, ept: 145.2292
    Epoch [8/50], Val Losses: mse: 0.3294, mae: 0.3986, huber: 0.1536, swd: 0.1555, ept: 109.3515
    Epoch [8/50], Test Losses: mse: 0.2303, mae: 0.3391, huber: 0.1098, swd: 0.1161, ept: 128.4004
      Epoch 8 composite train-obj: 0.114865
            Val objective improved 0.1542 → 0.1536, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.2901, mae: 0.3235, huber: 0.1130, swd: 0.1355, ept: 145.7442
    Epoch [9/50], Val Losses: mse: 0.3298, mae: 0.3983, huber: 0.1533, swd: 0.1613, ept: 111.7792
    Epoch [9/50], Test Losses: mse: 0.2244, mae: 0.3364, huber: 0.1074, swd: 0.1119, ept: 130.2072
      Epoch 9 composite train-obj: 0.113027
            Val objective improved 0.1536 → 0.1533, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.2843, mae: 0.3216, huber: 0.1114, swd: 0.1322, ept: 146.1459
    Epoch [10/50], Val Losses: mse: 0.3364, mae: 0.4031, huber: 0.1563, swd: 0.1607, ept: 110.1609
    Epoch [10/50], Test Losses: mse: 0.2270, mae: 0.3379, huber: 0.1088, swd: 0.1068, ept: 129.4120
      Epoch 10 composite train-obj: 0.111352
            No improvement (0.1563), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.2779, mae: 0.3178, huber: 0.1091, swd: 0.1276, ept: 146.7532
    Epoch [11/50], Val Losses: mse: 0.3494, mae: 0.4073, huber: 0.1609, swd: 0.1759, ept: 110.4202
    Epoch [11/50], Test Losses: mse: 0.2380, mae: 0.3439, huber: 0.1132, swd: 0.1202, ept: 126.1732
      Epoch 11 composite train-obj: 0.109093
            No improvement (0.1609), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.2742, mae: 0.3166, huber: 0.1080, swd: 0.1256, ept: 146.8298
    Epoch [12/50], Val Losses: mse: 0.3382, mae: 0.4024, huber: 0.1569, swd: 0.1571, ept: 109.9992
    Epoch [12/50], Test Losses: mse: 0.2326, mae: 0.3415, huber: 0.1109, swd: 0.1119, ept: 129.1630
      Epoch 12 composite train-obj: 0.108004
            No improvement (0.1569), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.2691, mae: 0.3142, huber: 0.1062, swd: 0.1223, ept: 147.2451
    Epoch [13/50], Val Losses: mse: 0.3397, mae: 0.4020, huber: 0.1567, swd: 0.1636, ept: 111.8893
    Epoch [13/50], Test Losses: mse: 0.2366, mae: 0.3467, huber: 0.1130, swd: 0.1223, ept: 128.0479
      Epoch 13 composite train-obj: 0.106209
            No improvement (0.1567), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.2640, mae: 0.3106, huber: 0.1043, swd: 0.1184, ept: 147.8868
    Epoch [14/50], Val Losses: mse: 0.3593, mae: 0.4087, huber: 0.1636, swd: 0.1787, ept: 111.4021
    Epoch [14/50], Test Losses: mse: 0.2536, mae: 0.3570, huber: 0.1201, swd: 0.1269, ept: 125.0749
      Epoch 14 composite train-obj: 0.104270
    Epoch [14/50], Test Losses: mse: 0.2244, mae: 0.3364, huber: 0.1074, swd: 0.1119, ept: 130.2025
    Best round's Test MSE: 0.2244, MAE: 0.3364, SWD: 0.1119
    Best round's Validation MSE: 0.3298, MAE: 0.3983, SWD: 0.1613
    Best round's Test verification MSE : 0.2244, MAE: 0.3364, SWD: 0.1119
    Time taken: 37.94 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4610, mae: 0.4274, huber: 0.1773, swd: 0.2382, ept: 120.1915
    Epoch [1/50], Val Losses: mse: 0.3784, mae: 0.4387, huber: 0.1742, swd: 0.1894, ept: 100.2016
    Epoch [1/50], Test Losses: mse: 0.2251, mae: 0.3359, huber: 0.1079, swd: 0.1030, ept: 124.3786
      Epoch 1 composite train-obj: 0.177303
            Val objective improved inf → 0.1742, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3683, mae: 0.3702, huber: 0.1415, swd: 0.1653, ept: 133.3900
    Epoch [2/50], Val Losses: mse: 0.3592, mae: 0.4225, huber: 0.1660, swd: 0.1787, ept: 103.3207
    Epoch [2/50], Test Losses: mse: 0.2192, mae: 0.3333, huber: 0.1052, swd: 0.0938, ept: 127.0942
      Epoch 2 composite train-obj: 0.141458
            Val objective improved 0.1742 → 0.1660, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3414, mae: 0.3524, huber: 0.1312, swd: 0.1512, ept: 138.0030
    Epoch [3/50], Val Losses: mse: 0.3465, mae: 0.4140, huber: 0.1606, swd: 0.1734, ept: 107.3756
    Epoch [3/50], Test Losses: mse: 0.2153, mae: 0.3304, huber: 0.1034, swd: 0.0963, ept: 127.8899
      Epoch 3 composite train-obj: 0.131203
            Val objective improved 0.1660 → 0.1606, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3225, mae: 0.3413, huber: 0.1245, swd: 0.1416, ept: 141.3521
    Epoch [4/50], Val Losses: mse: 0.3456, mae: 0.4104, huber: 0.1604, swd: 0.1579, ept: 107.9064
    Epoch [4/50], Test Losses: mse: 0.2109, mae: 0.3215, huber: 0.1010, swd: 0.0889, ept: 129.6310
      Epoch 4 composite train-obj: 0.124509
            Val objective improved 0.1606 → 0.1604, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3110, mae: 0.3348, huber: 0.1205, swd: 0.1357, ept: 143.1370
    Epoch [5/50], Val Losses: mse: 0.3465, mae: 0.4121, huber: 0.1603, swd: 0.1625, ept: 109.0270
    Epoch [5/50], Test Losses: mse: 0.2194, mae: 0.3325, huber: 0.1050, swd: 0.0985, ept: 128.4842
      Epoch 5 composite train-obj: 0.120496
            Val objective improved 0.1604 → 0.1603, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3043, mae: 0.3323, huber: 0.1183, swd: 0.1323, ept: 143.8295
    Epoch [6/50], Val Losses: mse: 0.3410, mae: 0.4086, huber: 0.1580, swd: 0.1596, ept: 110.7859
    Epoch [6/50], Test Losses: mse: 0.2220, mae: 0.3349, huber: 0.1063, swd: 0.0978, ept: 129.3587
      Epoch 6 composite train-obj: 0.118324
            Val objective improved 0.1603 → 0.1580, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.2969, mae: 0.3276, huber: 0.1156, swd: 0.1286, ept: 144.8545
    Epoch [7/50], Val Losses: mse: 0.3401, mae: 0.4047, huber: 0.1571, swd: 0.1542, ept: 109.2101
    Epoch [7/50], Test Losses: mse: 0.2273, mae: 0.3360, huber: 0.1085, swd: 0.1002, ept: 128.4383
      Epoch 7 composite train-obj: 0.115567
            Val objective improved 0.1580 → 0.1571, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.2913, mae: 0.3254, huber: 0.1138, swd: 0.1257, ept: 145.5963
    Epoch [8/50], Val Losses: mse: 0.3369, mae: 0.4049, huber: 0.1563, swd: 0.1495, ept: 109.5679
    Epoch [8/50], Test Losses: mse: 0.2280, mae: 0.3394, huber: 0.1091, swd: 0.1007, ept: 128.5422
      Epoch 8 composite train-obj: 0.113812
            Val objective improved 0.1571 → 0.1563, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.2851, mae: 0.3218, huber: 0.1116, swd: 0.1222, ept: 146.3503
    Epoch [9/50], Val Losses: mse: 0.3286, mae: 0.3973, huber: 0.1520, swd: 0.1440, ept: 111.1896
    Epoch [9/50], Test Losses: mse: 0.2402, mae: 0.3498, huber: 0.1146, swd: 0.1109, ept: 127.6608
      Epoch 9 composite train-obj: 0.111614
            Val objective improved 0.1563 → 0.1520, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.2805, mae: 0.3194, huber: 0.1100, swd: 0.1193, ept: 146.5645
    Epoch [10/50], Val Losses: mse: 0.3299, mae: 0.3989, huber: 0.1528, swd: 0.1414, ept: 111.0283
    Epoch [10/50], Test Losses: mse: 0.2354, mae: 0.3447, huber: 0.1120, swd: 0.1019, ept: 128.6204
      Epoch 10 composite train-obj: 0.110027
            No improvement (0.1528), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.2754, mae: 0.3172, huber: 0.1083, swd: 0.1162, ept: 147.0337
    Epoch [11/50], Val Losses: mse: 0.3346, mae: 0.3981, huber: 0.1539, swd: 0.1461, ept: 112.4017
    Epoch [11/50], Test Losses: mse: 0.2458, mae: 0.3511, huber: 0.1165, swd: 0.1042, ept: 127.8912
      Epoch 11 composite train-obj: 0.108350
            No improvement (0.1539), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.2698, mae: 0.3141, huber: 0.1064, swd: 0.1122, ept: 147.3495
    Epoch [12/50], Val Losses: mse: 0.3414, mae: 0.4032, huber: 0.1569, swd: 0.1435, ept: 112.1965
    Epoch [12/50], Test Losses: mse: 0.2469, mae: 0.3519, huber: 0.1170, swd: 0.1026, ept: 127.8560
      Epoch 12 composite train-obj: 0.106383
            No improvement (0.1569), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.2650, mae: 0.3120, huber: 0.1049, swd: 0.1091, ept: 147.6706
    Epoch [13/50], Val Losses: mse: 0.3598, mae: 0.4191, huber: 0.1647, swd: 0.1662, ept: 110.1830
    Epoch [13/50], Test Losses: mse: 0.2580, mae: 0.3618, huber: 0.1219, swd: 0.1154, ept: 125.6264
      Epoch 13 composite train-obj: 0.104851
            No improvement (0.1647), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.2609, mae: 0.3102, huber: 0.1035, swd: 0.1064, ept: 147.8319
    Epoch [14/50], Val Losses: mse: 0.3326, mae: 0.3969, huber: 0.1533, swd: 0.1321, ept: 112.8406
    Epoch [14/50], Test Losses: mse: 0.2392, mae: 0.3401, huber: 0.1133, swd: 0.0967, ept: 127.7996
      Epoch 14 composite train-obj: 0.103453
    Epoch [14/50], Test Losses: mse: 0.2402, mae: 0.3498, huber: 0.1146, swd: 0.1109, ept: 127.6608
    Best round's Test MSE: 0.2402, MAE: 0.3498, SWD: 0.1109
    Best round's Validation MSE: 0.3286, MAE: 0.3973, SWD: 0.1440
    Best round's Test verification MSE : 0.2402, MAE: 0.3498, SWD: 0.1109
    Time taken: 35.96 seconds
    
    ==================================================
    Experiment Summary (ACL_etth2_seq96_pred196_20250510_2045)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2258 ± 0.0112
      mae: 0.3390 ± 0.0080
      huber: 0.1082 ± 0.0050
      swd: 0.1075 ± 0.0056
      ept: 129.8424 ± 1.6526
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3332 ± 0.0057
      mae: 0.4012 ± 0.0048
      huber: 0.1545 ± 0.0026
      swd: 0.1587 ± 0.0111
      ept: 111.1903 ± 0.4805
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 101.67 seconds
    
    Experiment complete: ACL_etth2_seq96_pred196_20250510_2045
    Model: ACL
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['etth2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 12
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 108.8992, mae: 5.5415, huber: 5.1016, swd: 84.6621, ept: 95.1539
    Epoch [1/50], Val Losses: mse: 29.6508, mae: 3.6467, huber: 3.2092, swd: 11.6428, ept: 101.2346
    Epoch [1/50], Test Losses: mse: 19.9070, mae: 2.9366, huber: 2.5002, swd: 8.6701, ept: 121.3573
      Epoch 1 composite train-obj: 5.101642
            Val objective improved inf → 3.2092, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.6294, mae: 3.2204, huber: 2.7920, swd: 15.1531, ept: 127.3197
    Epoch [2/50], Val Losses: mse: 28.0133, mae: 3.5505, huber: 3.1130, swd: 10.1323, ept: 103.0101
    Epoch [2/50], Test Losses: mse: 19.6614, mae: 2.9139, huber: 2.4775, swd: 8.7590, ept: 121.0701
      Epoch 2 composite train-obj: 2.791988
            Val objective improved 3.2092 → 3.1130, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 30.5662, mae: 3.1557, huber: 2.7282, swd: 14.9235, ept: 129.5089
    Epoch [3/50], Val Losses: mse: 27.0637, mae: 3.5062, huber: 3.0688, swd: 10.5690, ept: 104.6074
    Epoch [3/50], Test Losses: mse: 18.8099, mae: 2.8561, huber: 2.4200, swd: 8.9689, ept: 124.1968
      Epoch 3 composite train-obj: 2.728205
            Val objective improved 3.1130 → 3.0688, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 29.2390, mae: 3.0493, huber: 2.6233, swd: 14.2302, ept: 133.5767
    Epoch [4/50], Val Losses: mse: 26.0108, mae: 3.4439, huber: 3.0067, swd: 9.4839, ept: 105.8558
    Epoch [4/50], Test Losses: mse: 18.5432, mae: 2.8408, huber: 2.4044, swd: 8.6598, ept: 124.2734
      Epoch 4 composite train-obj: 2.623299
            Val objective improved 3.0688 → 3.0067, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 27.8469, mae: 2.9302, huber: 2.5059, swd: 13.2114, ept: 137.5890
    Epoch [5/50], Val Losses: mse: 25.0682, mae: 3.3272, huber: 2.8919, swd: 8.7847, ept: 108.0650
    Epoch [5/50], Test Losses: mse: 18.0329, mae: 2.7504, huber: 2.3166, swd: 8.4460, ept: 126.5575
      Epoch 5 composite train-obj: 2.505880
            Val objective improved 3.0067 → 2.8919, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 27.5226, mae: 2.9090, huber: 2.4851, swd: 13.0719, ept: 138.6310
    Epoch [6/50], Val Losses: mse: 27.1734, mae: 3.4448, huber: 3.0078, swd: 10.2834, ept: 106.2548
    Epoch [6/50], Test Losses: mse: 19.5154, mae: 2.8372, huber: 2.4041, swd: 9.4955, ept: 120.2120
      Epoch 6 composite train-obj: 2.485093
            No improvement (3.0078), counter 1/5
    Epoch [7/50], Train Losses: mse: 27.2896, mae: 2.9045, huber: 2.4808, swd: 12.9653, ept: 138.4735
    Epoch [7/50], Val Losses: mse: 25.5320, mae: 3.3330, huber: 2.8987, swd: 9.0879, ept: 108.1890
    Epoch [7/50], Test Losses: mse: 18.4256, mae: 2.7779, huber: 2.3438, swd: 8.8173, ept: 124.3007
      Epoch 7 composite train-obj: 2.480789
            No improvement (2.8987), counter 2/5
    Epoch [8/50], Train Losses: mse: 26.7507, mae: 2.8549, huber: 2.4324, swd: 12.5612, ept: 139.8593
    Epoch [8/50], Val Losses: mse: 25.1214, mae: 3.3246, huber: 2.8903, swd: 9.2599, ept: 109.8831
    Epoch [8/50], Test Losses: mse: 17.5980, mae: 2.7396, huber: 2.3049, swd: 8.2586, ept: 125.5304
      Epoch 8 composite train-obj: 2.432397
            Val objective improved 2.8919 → 2.8903, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 26.7827, mae: 2.8649, huber: 2.4421, swd: 12.7075, ept: 139.5913
    Epoch [9/50], Val Losses: mse: 25.1235, mae: 3.3395, huber: 2.9051, swd: 9.8114, ept: 111.1040
    Epoch [9/50], Test Losses: mse: 17.0516, mae: 2.7235, huber: 2.2874, swd: 7.9427, ept: 128.2507
      Epoch 9 composite train-obj: 2.442061
            No improvement (2.9051), counter 1/5
    Epoch [10/50], Train Losses: mse: 26.9066, mae: 2.8821, huber: 2.4590, swd: 12.8558, ept: 139.0958
    Epoch [10/50], Val Losses: mse: 26.8326, mae: 3.3964, huber: 2.9625, swd: 10.0815, ept: 108.9137
    Epoch [10/50], Test Losses: mse: 19.6308, mae: 2.9009, huber: 2.4637, swd: 9.7048, ept: 118.5629
      Epoch 10 composite train-obj: 2.459007
            No improvement (2.9625), counter 2/5
    Epoch [11/50], Train Losses: mse: 26.2611, mae: 2.8180, huber: 2.3963, swd: 12.3231, ept: 140.5113
    Epoch [11/50], Val Losses: mse: 25.4879, mae: 3.3921, huber: 2.9570, swd: 10.4277, ept: 111.5701
    Epoch [11/50], Test Losses: mse: 17.7449, mae: 2.8341, huber: 2.3941, swd: 8.7431, ept: 126.2166
      Epoch 11 composite train-obj: 2.396344
            No improvement (2.9570), counter 3/5
    Epoch [12/50], Train Losses: mse: 26.1830, mae: 2.8127, huber: 2.3910, swd: 12.3412, ept: 140.8537
    Epoch [12/50], Val Losses: mse: 25.1226, mae: 3.3611, huber: 2.9262, swd: 9.8311, ept: 111.5171
    Epoch [12/50], Test Losses: mse: 17.1246, mae: 2.7321, huber: 2.2976, swd: 8.0906, ept: 128.4183
      Epoch 12 composite train-obj: 2.391031
            No improvement (2.9262), counter 4/5
    Epoch [13/50], Train Losses: mse: 25.6117, mae: 2.7599, huber: 2.3392, swd: 11.8051, ept: 141.9096
    Epoch [13/50], Val Losses: mse: 25.3138, mae: 3.3178, huber: 2.8823, swd: 9.0923, ept: 111.1447
    Epoch [13/50], Test Losses: mse: 18.1419, mae: 2.7827, huber: 2.3465, swd: 8.5873, ept: 122.8359
      Epoch 13 composite train-obj: 2.339155
            Val objective improved 2.8903 → 2.8823, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 25.6097, mae: 2.7675, huber: 2.3465, swd: 11.8248, ept: 141.5945
    Epoch [14/50], Val Losses: mse: 26.9615, mae: 3.3975, huber: 2.9648, swd: 10.6294, ept: 109.5388
    Epoch [14/50], Test Losses: mse: 19.1728, mae: 2.8331, huber: 2.3982, swd: 9.4082, ept: 120.9468
      Epoch 14 composite train-obj: 2.346539
            No improvement (2.9648), counter 1/5
    Epoch [15/50], Train Losses: mse: 25.8055, mae: 2.7858, huber: 2.3644, swd: 12.0702, ept: 141.3728
    Epoch [15/50], Val Losses: mse: 24.5270, mae: 3.2596, huber: 2.8273, swd: 8.7673, ept: 112.3836
    Epoch [15/50], Test Losses: mse: 17.3001, mae: 2.6837, huber: 2.2516, swd: 8.0240, ept: 126.6229
      Epoch 15 composite train-obj: 2.364424
            Val objective improved 2.8823 → 2.8273, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 25.2914, mae: 2.7364, huber: 2.3162, swd: 11.6256, ept: 142.4132
    Epoch [16/50], Val Losses: mse: 24.6330, mae: 3.3041, huber: 2.8724, swd: 9.3243, ept: 112.8462
    Epoch [16/50], Test Losses: mse: 17.2523, mae: 2.7403, huber: 2.3044, swd: 8.1001, ept: 126.6824
      Epoch 16 composite train-obj: 2.316189
            No improvement (2.8724), counter 1/5
    Epoch [17/50], Train Losses: mse: 25.1980, mae: 2.7314, huber: 2.3115, swd: 11.5412, ept: 142.7067
    Epoch [17/50], Val Losses: mse: 25.4049, mae: 3.3131, huber: 2.8800, swd: 9.0901, ept: 110.7063
    Epoch [17/50], Test Losses: mse: 18.4924, mae: 2.7656, huber: 2.3316, swd: 8.9765, ept: 125.0400
      Epoch 17 composite train-obj: 2.311481
            No improvement (2.8800), counter 2/5
    Epoch [18/50], Train Losses: mse: 25.1005, mae: 2.7245, huber: 2.3047, swd: 11.4586, ept: 142.5319
    Epoch [18/50], Val Losses: mse: 27.4427, mae: 3.4083, huber: 2.9740, swd: 10.8631, ept: 109.8834
    Epoch [18/50], Test Losses: mse: 19.7249, mae: 2.8862, huber: 2.4490, swd: 9.9762, ept: 120.6027
      Epoch 18 composite train-obj: 2.304737
            No improvement (2.9740), counter 3/5
    Epoch [19/50], Train Losses: mse: 25.1805, mae: 2.7372, huber: 2.3168, swd: 11.5389, ept: 142.3415
    Epoch [19/50], Val Losses: mse: 26.2647, mae: 3.3669, huber: 2.9320, swd: 9.8751, ept: 110.4108
    Epoch [19/50], Test Losses: mse: 18.6324, mae: 2.7578, huber: 2.3266, swd: 8.9815, ept: 122.4105
      Epoch 19 composite train-obj: 2.316793
            No improvement (2.9320), counter 4/5
    Epoch [20/50], Train Losses: mse: 25.3547, mae: 2.7495, huber: 2.3295, swd: 11.7622, ept: 142.1104
    Epoch [20/50], Val Losses: mse: 24.4447, mae: 3.2937, huber: 2.8628, swd: 9.3043, ept: 113.8116
    Epoch [20/50], Test Losses: mse: 17.2213, mae: 2.7207, huber: 2.2886, swd: 8.2049, ept: 127.6689
      Epoch 20 composite train-obj: 2.329452
    Epoch [20/50], Test Losses: mse: 17.2999, mae: 2.6837, huber: 2.2516, swd: 8.0237, ept: 126.6211
    Best round's Test MSE: 17.3001, MAE: 2.6837, SWD: 8.0240
    Best round's Validation MSE: 24.5270, MAE: 3.2596, SWD: 8.7673
    Best round's Test verification MSE : 17.2999, MAE: 2.6837, SWD: 8.0237
    Time taken: 54.58 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 108.6356, mae: 5.5253, huber: 5.0852, swd: 86.1934, ept: 96.0376
    Epoch [1/50], Val Losses: mse: 31.3746, mae: 3.7419, huber: 3.3045, swd: 14.2990, ept: 102.6726
    Epoch [1/50], Test Losses: mse: 20.0065, mae: 2.9614, huber: 2.5235, swd: 9.2462, ept: 123.5830
      Epoch 1 composite train-obj: 5.085176
            Val objective improved inf → 3.3045, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.3244, mae: 3.1947, huber: 2.7666, swd: 15.3770, ept: 129.2605
    Epoch [2/50], Val Losses: mse: 29.8853, mae: 3.6398, huber: 3.2032, swd: 11.5182, ept: 99.2547
    Epoch [2/50], Test Losses: mse: 20.6789, mae: 2.9618, huber: 2.5266, swd: 9.5377, ept: 117.5606
      Epoch 2 composite train-obj: 2.766609
            Val objective improved 3.3045 → 3.2032, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 30.1388, mae: 3.1164, huber: 2.6893, swd: 14.9717, ept: 131.8924
    Epoch [3/50], Val Losses: mse: 26.8338, mae: 3.4563, huber: 3.0208, swd: 10.2364, ept: 105.5845
    Epoch [3/50], Test Losses: mse: 18.6989, mae: 2.8329, huber: 2.3978, swd: 8.8561, ept: 123.8742
      Epoch 3 composite train-obj: 2.689258
            Val objective improved 3.2032 → 3.0208, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 28.8509, mae: 3.0052, huber: 2.5793, swd: 14.2993, ept: 136.1508
    Epoch [4/50], Val Losses: mse: 25.5002, mae: 3.3630, huber: 2.9278, swd: 9.3367, ept: 107.1551
    Epoch [4/50], Test Losses: mse: 18.3938, mae: 2.7835, huber: 2.3493, swd: 8.9549, ept: 127.3784
      Epoch 4 composite train-obj: 2.579306
            Val objective improved 3.0208 → 2.9278, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 28.4813, mae: 2.9830, huber: 2.5575, swd: 14.2430, ept: 137.7910
    Epoch [5/50], Val Losses: mse: 25.8840, mae: 3.3677, huber: 2.9322, swd: 9.6830, ept: 107.4029
    Epoch [5/50], Test Losses: mse: 18.7212, mae: 2.7719, huber: 2.3401, swd: 9.2926, ept: 125.4811
      Epoch 5 composite train-obj: 2.557454
            No improvement (2.9322), counter 1/5
    Epoch [6/50], Train Losses: mse: 27.9407, mae: 2.9242, huber: 2.4997, swd: 13.8483, ept: 139.4843
    Epoch [6/50], Val Losses: mse: 26.1335, mae: 3.3709, huber: 2.9356, swd: 9.7885, ept: 108.1523
    Epoch [6/50], Test Losses: mse: 19.0856, mae: 2.8028, huber: 2.3699, swd: 9.5826, ept: 124.3636
      Epoch 6 composite train-obj: 2.499729
            No improvement (2.9356), counter 2/5
    Epoch [7/50], Train Losses: mse: 27.7326, mae: 2.9218, huber: 2.4977, swd: 13.7385, ept: 139.2570
    Epoch [7/50], Val Losses: mse: 25.0956, mae: 3.3425, huber: 2.9067, swd: 9.4777, ept: 109.6283
    Epoch [7/50], Test Losses: mse: 17.6829, mae: 2.7595, huber: 2.3235, swd: 8.5833, ept: 127.0839
      Epoch 7 composite train-obj: 2.497741
            Val objective improved 2.9278 → 2.9067, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 27.2694, mae: 2.8880, huber: 2.4647, swd: 13.3621, ept: 139.7456
    Epoch [8/50], Val Losses: mse: 24.9161, mae: 3.3145, huber: 2.8801, swd: 9.4597, ept: 110.0105
    Epoch [8/50], Test Losses: mse: 17.5746, mae: 2.7568, huber: 2.3198, swd: 8.5448, ept: 128.2440
      Epoch 8 composite train-obj: 2.464714
            Val objective improved 2.9067 → 2.8801, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 27.0120, mae: 2.8651, huber: 2.4427, swd: 13.2274, ept: 140.3459
    Epoch [9/50], Val Losses: mse: 24.4574, mae: 3.2874, huber: 2.8551, swd: 9.0919, ept: 110.8757
    Epoch [9/50], Test Losses: mse: 17.4717, mae: 2.7285, huber: 2.2942, swd: 8.5424, ept: 129.8816
      Epoch 9 composite train-obj: 2.442683
            Val objective improved 2.8801 → 2.8551, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 26.9040, mae: 2.8595, huber: 2.4376, swd: 13.2470, ept: 140.4138
    Epoch [10/50], Val Losses: mse: 24.8138, mae: 3.3086, huber: 2.8757, swd: 9.5778, ept: 111.1225
    Epoch [10/50], Test Losses: mse: 17.3213, mae: 2.7250, huber: 2.2900, swd: 8.3835, ept: 128.9658
      Epoch 10 composite train-obj: 2.437646
            No improvement (2.8757), counter 1/5
    Epoch [11/50], Train Losses: mse: 26.4449, mae: 2.8211, huber: 2.4001, swd: 12.7844, ept: 141.0165
    Epoch [11/50], Val Losses: mse: 25.1453, mae: 3.3021, huber: 2.8723, swd: 9.7710, ept: 110.9323
    Epoch [11/50], Test Losses: mse: 17.2836, mae: 2.6608, huber: 2.2319, swd: 8.3204, ept: 128.4763
      Epoch 11 composite train-obj: 2.400062
            No improvement (2.8723), counter 2/5
    Epoch [12/50], Train Losses: mse: 26.4000, mae: 2.8182, huber: 2.3973, swd: 12.8133, ept: 140.8285
    Epoch [12/50], Val Losses: mse: 24.7441, mae: 3.2710, huber: 2.8420, swd: 9.4733, ept: 112.2062
    Epoch [12/50], Test Losses: mse: 17.1586, mae: 2.6809, huber: 2.2489, swd: 8.1303, ept: 129.8552
      Epoch 12 composite train-obj: 2.397257
            Val objective improved 2.8551 → 2.8420, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 26.1126, mae: 2.7968, huber: 2.3765, swd: 12.6004, ept: 141.3635
    Epoch [13/50], Val Losses: mse: 24.6611, mae: 3.2966, huber: 2.8674, swd: 9.4532, ept: 112.0466
    Epoch [13/50], Test Losses: mse: 17.3427, mae: 2.7276, huber: 2.2923, swd: 8.3951, ept: 127.7426
      Epoch 13 composite train-obj: 2.376535
            No improvement (2.8674), counter 1/5
    Epoch [14/50], Train Losses: mse: 25.7827, mae: 2.7691, huber: 2.3494, swd: 12.3797, ept: 141.6742
    Epoch [14/50], Val Losses: mse: 25.2409, mae: 3.2867, huber: 2.8597, swd: 9.9927, ept: 112.1690
    Epoch [14/50], Test Losses: mse: 17.6330, mae: 2.7107, huber: 2.2786, swd: 8.6504, ept: 126.5810
      Epoch 14 composite train-obj: 2.349450
            No improvement (2.8597), counter 2/5
    Epoch [15/50], Train Losses: mse: 25.7059, mae: 2.7597, huber: 2.3403, swd: 12.3086, ept: 141.8832
    Epoch [15/50], Val Losses: mse: 24.5420, mae: 3.2795, huber: 2.8517, swd: 9.2947, ept: 112.0948
    Epoch [15/50], Test Losses: mse: 17.4199, mae: 2.7286, huber: 2.2951, swd: 8.5574, ept: 127.1427
      Epoch 15 composite train-obj: 2.340345
            No improvement (2.8517), counter 3/5
    Epoch [16/50], Train Losses: mse: 25.6193, mae: 2.7608, huber: 2.3409, swd: 12.3028, ept: 142.1415
    Epoch [16/50], Val Losses: mse: 24.1656, mae: 3.2549, huber: 2.8269, swd: 9.3723, ept: 113.5617
    Epoch [16/50], Test Losses: mse: 16.9175, mae: 2.6674, huber: 2.2370, swd: 8.2004, ept: 129.3491
      Epoch 16 composite train-obj: 2.340950
            Val objective improved 2.8420 → 2.8269, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 25.4140, mae: 2.7413, huber: 2.3218, swd: 12.1332, ept: 142.3556
    Epoch [17/50], Val Losses: mse: 24.5116, mae: 3.2650, huber: 2.8354, swd: 9.6017, ept: 112.6646
    Epoch [17/50], Test Losses: mse: 17.1757, mae: 2.7072, huber: 2.2747, swd: 8.4344, ept: 127.5531
      Epoch 17 composite train-obj: 2.321845
            No improvement (2.8354), counter 1/5
    Epoch [18/50], Train Losses: mse: 25.3251, mae: 2.7368, huber: 2.3174, swd: 12.0915, ept: 142.6078
    Epoch [18/50], Val Losses: mse: 24.0660, mae: 3.2232, huber: 2.7934, swd: 8.9568, ept: 111.9107
    Epoch [18/50], Test Losses: mse: 17.5698, mae: 2.7019, huber: 2.2700, swd: 8.7429, ept: 126.4220
      Epoch 18 composite train-obj: 2.317449
            Val objective improved 2.8269 → 2.7934, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 25.1674, mae: 2.7228, huber: 2.3040, swd: 11.9501, ept: 142.8246
    Epoch [19/50], Val Losses: mse: 25.1191, mae: 3.2694, huber: 2.8416, swd: 9.7542, ept: 111.3799
    Epoch [19/50], Test Losses: mse: 18.1791, mae: 2.7558, huber: 2.3244, swd: 9.1350, ept: 122.1244
      Epoch 19 composite train-obj: 2.304000
            No improvement (2.8416), counter 1/5
    Epoch [20/50], Train Losses: mse: 25.3356, mae: 2.7464, huber: 2.3270, swd: 12.1171, ept: 142.2821
    Epoch [20/50], Val Losses: mse: 23.9977, mae: 3.2193, huber: 2.7932, swd: 9.1129, ept: 113.4230
    Epoch [20/50], Test Losses: mse: 17.2009, mae: 2.6444, huber: 2.2178, swd: 8.3595, ept: 126.3081
      Epoch 20 composite train-obj: 2.327033
            Val objective improved 2.7934 → 2.7932, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 24.9584, mae: 2.7052, huber: 2.2869, swd: 11.8196, ept: 143.3815
    Epoch [21/50], Val Losses: mse: 26.8622, mae: 3.3746, huber: 2.9456, swd: 11.0749, ept: 109.2911
    Epoch [21/50], Test Losses: mse: 19.9113, mae: 2.8611, huber: 2.4296, swd: 10.6116, ept: 119.1083
      Epoch 21 composite train-obj: 2.286854
            No improvement (2.9456), counter 1/5
    Epoch [22/50], Train Losses: mse: 25.0039, mae: 2.7112, huber: 2.2929, swd: 11.9023, ept: 143.0425
    Epoch [22/50], Val Losses: mse: 23.8450, mae: 3.2243, huber: 2.7977, swd: 8.9982, ept: 112.5727
    Epoch [22/50], Test Losses: mse: 17.3150, mae: 2.7116, huber: 2.2801, swd: 8.5677, ept: 126.4185
      Epoch 22 composite train-obj: 2.292910
            No improvement (2.7977), counter 2/5
    Epoch [23/50], Train Losses: mse: 24.6508, mae: 2.6721, huber: 2.2548, swd: 11.5678, ept: 143.8944
    Epoch [23/50], Val Losses: mse: 24.1760, mae: 3.2653, huber: 2.8387, swd: 9.4206, ept: 112.9287
    Epoch [23/50], Test Losses: mse: 17.5540, mae: 2.7150, huber: 2.2852, swd: 8.7216, ept: 126.3222
      Epoch 23 composite train-obj: 2.254777
            No improvement (2.8387), counter 3/5
    Epoch [24/50], Train Losses: mse: 24.6754, mae: 2.6805, huber: 2.2629, swd: 11.6402, ept: 143.9813
    Epoch [24/50], Val Losses: mse: 23.7647, mae: 3.2001, huber: 2.7744, swd: 9.0072, ept: 113.6063
    Epoch [24/50], Test Losses: mse: 17.3383, mae: 2.7262, huber: 2.2928, swd: 8.5237, ept: 125.4450
      Epoch 24 composite train-obj: 2.262919
            Val objective improved 2.7932 → 2.7744, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 24.5682, mae: 2.6681, huber: 2.2509, swd: 11.5804, ept: 143.9513
    Epoch [25/50], Val Losses: mse: 23.2553, mae: 3.1794, huber: 2.7538, swd: 8.6855, ept: 114.7107
    Epoch [25/50], Test Losses: mse: 16.9520, mae: 2.6908, huber: 2.2602, swd: 8.3241, ept: 127.3510
      Epoch 25 composite train-obj: 2.250932
            Val objective improved 2.7744 → 2.7538, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 24.5033, mae: 2.6685, huber: 2.2511, swd: 11.5420, ept: 144.0633
    Epoch [26/50], Val Losses: mse: 23.9941, mae: 3.2571, huber: 2.8279, swd: 9.1276, ept: 111.4707
    Epoch [26/50], Test Losses: mse: 17.4517, mae: 2.7640, huber: 2.3281, swd: 8.6507, ept: 125.7201
      Epoch 26 composite train-obj: 2.251146
            No improvement (2.8279), counter 1/5
    Epoch [27/50], Train Losses: mse: 24.6383, mae: 2.6938, huber: 2.2756, swd: 11.6897, ept: 143.5305
    Epoch [27/50], Val Losses: mse: 26.5318, mae: 3.3633, huber: 2.9327, swd: 10.8824, ept: 108.3974
    Epoch [27/50], Test Losses: mse: 20.4435, mae: 2.9659, huber: 2.5276, swd: 11.2007, ept: 115.4745
      Epoch 27 composite train-obj: 2.275633
            No improvement (2.9327), counter 2/5
    Epoch [28/50], Train Losses: mse: 24.6107, mae: 2.6802, huber: 2.2626, swd: 11.6206, ept: 143.5323
    Epoch [28/50], Val Losses: mse: 26.1873, mae: 3.3517, huber: 2.9255, swd: 10.6394, ept: 108.8983
    Epoch [28/50], Test Losses: mse: 19.6955, mae: 2.8720, huber: 2.4409, swd: 10.3658, ept: 117.8522
      Epoch 28 composite train-obj: 2.262642
            No improvement (2.9255), counter 3/5
    Epoch [29/50], Train Losses: mse: 24.2329, mae: 2.6427, huber: 2.2259, swd: 11.3414, ept: 144.6433
    Epoch [29/50], Val Losses: mse: 24.6984, mae: 3.3273, huber: 2.9010, swd: 10.1964, ept: 112.7705
    Epoch [29/50], Test Losses: mse: 17.1647, mae: 2.7295, huber: 2.2983, swd: 8.5020, ept: 126.3318
      Epoch 29 composite train-obj: 2.225898
            No improvement (2.9010), counter 4/5
    Epoch [30/50], Train Losses: mse: 24.1701, mae: 2.6456, huber: 2.2285, swd: 11.3151, ept: 144.6739
    Epoch [30/50], Val Losses: mse: 24.7235, mae: 3.2520, huber: 2.8288, swd: 9.6293, ept: 112.3092
    Epoch [30/50], Test Losses: mse: 17.8357, mae: 2.7178, huber: 2.2900, swd: 8.9012, ept: 124.8584
      Epoch 30 composite train-obj: 2.228546
    Epoch [30/50], Test Losses: mse: 16.9519, mae: 2.6908, huber: 2.2601, swd: 8.3239, ept: 127.3496
    Best round's Test MSE: 16.9520, MAE: 2.6908, SWD: 8.3241
    Best round's Validation MSE: 23.2553, MAE: 3.1794, SWD: 8.6855
    Best round's Test verification MSE : 16.9519, MAE: 2.6908, SWD: 8.3239
    Time taken: 84.33 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 110.8643, mae: 5.6210, huber: 5.1798, swd: 75.1049, ept: 93.8578
    Epoch [1/50], Val Losses: mse: 29.2738, mae: 3.6070, huber: 3.1703, swd: 9.6767, ept: 101.0582
    Epoch [1/50], Test Losses: mse: 20.4201, mae: 2.9804, huber: 2.5420, swd: 8.0264, ept: 120.9499
      Epoch 1 composite train-obj: 5.179752
            Val objective improved inf → 3.1703, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 31.7726, mae: 3.2166, huber: 2.7879, swd: 13.3730, ept: 126.6121
    Epoch [2/50], Val Losses: mse: 29.5597, mae: 3.6087, huber: 3.1714, swd: 8.9850, ept: 98.4633
    Epoch [2/50], Test Losses: mse: 22.4868, mae: 3.1348, huber: 2.6945, swd: 9.2110, ept: 114.7355
      Epoch 2 composite train-obj: 2.787884
            No improvement (3.1714), counter 1/5
    Epoch [3/50], Train Losses: mse: 30.5008, mae: 3.1275, huber: 2.7002, swd: 12.8016, ept: 129.6639
    Epoch [3/50], Val Losses: mse: 26.4497, mae: 3.4456, huber: 3.0087, swd: 8.0328, ept: 105.1886
    Epoch [3/50], Test Losses: mse: 18.9237, mae: 2.8391, huber: 2.4039, swd: 7.5234, ept: 122.6428
      Epoch 3 composite train-obj: 2.700165
            Val objective improved 3.1703 → 3.0087, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 28.9343, mae: 3.0102, huber: 2.5842, swd: 12.2168, ept: 134.3690
    Epoch [4/50], Val Losses: mse: 27.1182, mae: 3.4661, huber: 3.0290, swd: 8.7208, ept: 104.8639
    Epoch [4/50], Test Losses: mse: 18.8092, mae: 2.8244, huber: 2.3898, swd: 7.6512, ept: 122.0132
      Epoch 4 composite train-obj: 2.584240
            No improvement (3.0290), counter 1/5
    Epoch [5/50], Train Losses: mse: 28.0791, mae: 2.9415, huber: 2.5163, swd: 11.8576, ept: 137.0961
    Epoch [5/50], Val Losses: mse: 25.4327, mae: 3.3925, huber: 2.9550, swd: 8.2881, ept: 107.7581
    Epoch [5/50], Test Losses: mse: 17.8897, mae: 2.7756, huber: 2.3405, swd: 7.4074, ept: 126.6281
      Epoch 5 composite train-obj: 2.516301
            Val objective improved 3.0087 → 2.9550, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 27.5359, mae: 2.8988, huber: 2.4744, swd: 11.6307, ept: 138.7761
    Epoch [6/50], Val Losses: mse: 25.2839, mae: 3.3551, huber: 2.9188, swd: 7.9450, ept: 108.0011
    Epoch [6/50], Test Losses: mse: 17.8838, mae: 2.7553, huber: 2.3210, swd: 7.3531, ept: 126.3661
      Epoch 6 composite train-obj: 2.474363
            Val objective improved 2.9550 → 2.9188, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.5106, mae: 2.9000, huber: 2.4758, swd: 11.6793, ept: 138.4494
    Epoch [7/50], Val Losses: mse: 24.8045, mae: 3.3251, huber: 2.8868, swd: 7.9186, ept: 109.1320
    Epoch [7/50], Test Losses: mse: 17.9618, mae: 2.8369, huber: 2.3963, swd: 7.6002, ept: 128.0609
      Epoch 7 composite train-obj: 2.475840
            Val objective improved 2.9188 → 2.8868, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 26.9807, mae: 2.8629, huber: 2.4392, swd: 11.3315, ept: 139.5970
    Epoch [8/50], Val Losses: mse: 26.8990, mae: 3.4128, huber: 2.9763, swd: 9.1412, ept: 107.4087
    Epoch [8/50], Test Losses: mse: 18.4326, mae: 2.8105, huber: 2.3734, swd: 7.6830, ept: 123.6394
      Epoch 8 composite train-obj: 2.439165
            No improvement (2.9763), counter 1/5
    Epoch [9/50], Train Losses: mse: 26.9418, mae: 2.8565, huber: 2.4335, swd: 11.3158, ept: 139.5460
    Epoch [9/50], Val Losses: mse: 24.9553, mae: 3.3042, huber: 2.8705, swd: 7.9542, ept: 109.7624
    Epoch [9/50], Test Losses: mse: 17.5007, mae: 2.7208, huber: 2.2872, swd: 7.1546, ept: 128.2848
      Epoch 9 composite train-obj: 2.433479
            Val objective improved 2.8868 → 2.8705, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 26.7326, mae: 2.8361, huber: 2.4141, swd: 11.2434, ept: 140.2606
    Epoch [10/50], Val Losses: mse: 25.3035, mae: 3.3085, huber: 2.8762, swd: 8.1890, ept: 109.9843
    Epoch [10/50], Test Losses: mse: 17.5247, mae: 2.6979, huber: 2.2657, swd: 7.1672, ept: 127.5882
      Epoch 10 composite train-obj: 2.414140
            No improvement (2.8762), counter 1/5
    Epoch [11/50], Train Losses: mse: 26.5278, mae: 2.8219, huber: 2.4004, swd: 11.1457, ept: 140.6032
    Epoch [11/50], Val Losses: mse: 25.4449, mae: 3.3178, huber: 2.8858, swd: 8.1676, ept: 110.1942
    Epoch [11/50], Test Losses: mse: 18.0115, mae: 2.7208, huber: 2.2890, swd: 7.4360, ept: 126.1562
      Epoch 11 composite train-obj: 2.400425
            No improvement (2.8858), counter 2/5
    Epoch [12/50], Train Losses: mse: 26.1769, mae: 2.7972, huber: 2.3768, swd: 10.9144, ept: 140.8849
    Epoch [12/50], Val Losses: mse: 24.7632, mae: 3.2586, huber: 2.8279, swd: 8.0056, ept: 111.0640
    Epoch [12/50], Test Losses: mse: 17.5785, mae: 2.7388, huber: 2.3033, swd: 7.3381, ept: 127.5138
      Epoch 12 composite train-obj: 2.376765
            Val objective improved 2.8705 → 2.8279, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 25.8985, mae: 2.7726, huber: 2.3528, swd: 10.7261, ept: 141.3223
    Epoch [13/50], Val Losses: mse: 24.9746, mae: 3.2773, huber: 2.8467, swd: 8.2001, ept: 111.3750
    Epoch [13/50], Test Losses: mse: 17.2144, mae: 2.6858, huber: 2.2528, swd: 7.0409, ept: 129.7838
      Epoch 13 composite train-obj: 2.352828
            No improvement (2.8467), counter 1/5
    Epoch [14/50], Train Losses: mse: 25.9298, mae: 2.7837, huber: 2.3637, swd: 10.7924, ept: 140.8291
    Epoch [14/50], Val Losses: mse: 24.8692, mae: 3.2840, huber: 2.8516, swd: 8.1331, ept: 110.8735
    Epoch [14/50], Test Losses: mse: 17.3957, mae: 2.7501, huber: 2.3133, swd: 7.2478, ept: 127.9846
      Epoch 14 composite train-obj: 2.363671
            No improvement (2.8516), counter 2/5
    Epoch [15/50], Train Losses: mse: 25.5592, mae: 2.7536, huber: 2.3336, swd: 10.4988, ept: 141.5754
    Epoch [15/50], Val Losses: mse: 25.2023, mae: 3.2744, huber: 2.8430, swd: 8.3423, ept: 111.9673
    Epoch [15/50], Test Losses: mse: 17.6284, mae: 2.7214, huber: 2.2877, swd: 7.3701, ept: 128.3428
      Epoch 15 composite train-obj: 2.333631
            No improvement (2.8430), counter 3/5
    Epoch [16/50], Train Losses: mse: 25.4703, mae: 2.7487, huber: 2.3290, swd: 10.3886, ept: 141.5679
    Epoch [16/50], Val Losses: mse: 25.3259, mae: 3.3278, huber: 2.8947, swd: 8.2470, ept: 111.1120
    Epoch [16/50], Test Losses: mse: 17.3000, mae: 2.6899, huber: 2.2582, swd: 7.0423, ept: 126.9444
      Epoch 16 composite train-obj: 2.328993
            No improvement (2.8947), counter 4/5
    Epoch [17/50], Train Losses: mse: 25.4071, mae: 2.7435, huber: 2.3240, swd: 10.3881, ept: 141.6639
    Epoch [17/50], Val Losses: mse: 24.7592, mae: 3.2658, huber: 2.8358, swd: 7.8106, ept: 112.2787
    Epoch [17/50], Test Losses: mse: 18.3332, mae: 2.8027, huber: 2.3667, swd: 7.8094, ept: 124.9331
      Epoch 17 composite train-obj: 2.324017
    Epoch [17/50], Test Losses: mse: 17.5793, mae: 2.7388, huber: 2.3034, swd: 7.3385, ept: 127.5005
    Best round's Test MSE: 17.5785, MAE: 2.7388, SWD: 7.3381
    Best round's Validation MSE: 24.7632, MAE: 3.2586, SWD: 8.0056
    Best round's Test verification MSE : 17.5793, MAE: 2.7388, SWD: 7.3385
    Time taken: 49.85 seconds
    
    ==================================================
    Experiment Summary (ACL_etth2_seq96_pred196_20250511_1607)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 17.2769 ± 0.2563
      mae: 2.7044 ± 0.0244
      huber: 2.2717 ± 0.0226
      swd: 7.8954 ± 0.4126
      ept: 127.1625 ± 0.3873
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 24.1818 ± 0.6622
      mae: 3.2325 ± 0.0376
      huber: 2.8030 ± 0.0348
      swd: 8.4861 ± 0.3414
      ept: 112.7194 ± 1.5076
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 188.86 seconds
    
    Experiment complete: ACL_etth2_seq96_pred196_20250511_1607
    Model: ACL
    Dataset: etth2
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
    channels=data_mgr.datasets['etth2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.5387, mae: 0.4651, huber: 0.2032, swd: 0.3068, ept: 169.7649
    Epoch [1/50], Val Losses: mse: 0.4084, mae: 0.4600, huber: 0.1878, swd: 0.2116, ept: 138.8281
    Epoch [1/50], Test Losses: mse: 0.2407, mae: 0.3441, huber: 0.1148, swd: 0.1116, ept: 181.2342
      Epoch 1 composite train-obj: 0.203230
            Val objective improved inf → 0.1878, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4458, mae: 0.4108, huber: 0.1684, swd: 0.2236, ept: 189.3843
    Epoch [2/50], Val Losses: mse: 0.3982, mae: 0.4530, huber: 0.1834, swd: 0.2012, ept: 139.3757
    Epoch [2/50], Test Losses: mse: 0.2380, mae: 0.3484, huber: 0.1140, swd: 0.1060, ept: 183.2467
      Epoch 2 composite train-obj: 0.168405
            Val objective improved 0.1878 → 0.1834, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4170, mae: 0.3925, huber: 0.1571, swd: 0.2028, ept: 195.1437
    Epoch [3/50], Val Losses: mse: 0.4278, mae: 0.4679, huber: 0.1950, swd: 0.2280, ept: 143.3601
    Epoch [3/50], Test Losses: mse: 0.2566, mae: 0.3707, huber: 0.1232, swd: 0.1358, ept: 175.8120
      Epoch 3 composite train-obj: 0.157103
            No improvement (0.1950), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3929, mae: 0.3777, huber: 0.1478, swd: 0.1864, ept: 203.9305
    Epoch [4/50], Val Losses: mse: 0.3848, mae: 0.4377, huber: 0.1781, swd: 0.1685, ept: 155.3438
    Epoch [4/50], Test Losses: mse: 0.2421, mae: 0.3561, huber: 0.1160, swd: 0.1104, ept: 187.7170
      Epoch 4 composite train-obj: 0.147753
            Val objective improved 0.1834 → 0.1781, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3833, mae: 0.3719, huber: 0.1439, swd: 0.1810, ept: 207.2388
    Epoch [5/50], Val Losses: mse: 0.4010, mae: 0.4467, huber: 0.1849, swd: 0.1806, ept: 150.1221
    Epoch [5/50], Test Losses: mse: 0.2494, mae: 0.3547, huber: 0.1184, swd: 0.1167, ept: 185.9211
      Epoch 5 composite train-obj: 0.143945
            No improvement (0.1849), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3760, mae: 0.3675, huber: 0.1411, swd: 0.1775, ept: 209.0213
    Epoch [6/50], Val Losses: mse: 0.3801, mae: 0.4350, huber: 0.1762, swd: 0.1557, ept: 155.1954
    Epoch [6/50], Test Losses: mse: 0.2455, mae: 0.3519, huber: 0.1168, swd: 0.1056, ept: 189.3158
      Epoch 6 composite train-obj: 0.141067
            Val objective improved 0.1781 → 0.1762, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3695, mae: 0.3633, huber: 0.1385, swd: 0.1733, ept: 210.4194
    Epoch [7/50], Val Losses: mse: 0.4350, mae: 0.4686, huber: 0.1978, swd: 0.2387, ept: 152.5696
    Epoch [7/50], Test Losses: mse: 0.2500, mae: 0.3578, huber: 0.1195, swd: 0.1263, ept: 182.3191
      Epoch 7 composite train-obj: 0.138529
            No improvement (0.1978), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.3632, mae: 0.3597, huber: 0.1362, swd: 0.1699, ept: 211.3227
    Epoch [8/50], Val Losses: mse: 0.4018, mae: 0.4426, huber: 0.1838, swd: 0.1843, ept: 156.5348
    Epoch [8/50], Test Losses: mse: 0.2554, mae: 0.3577, huber: 0.1215, swd: 0.1153, ept: 183.8200
      Epoch 8 composite train-obj: 0.136205
            No improvement (0.1838), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.3591, mae: 0.3575, huber: 0.1347, swd: 0.1671, ept: 211.9645
    Epoch [9/50], Val Losses: mse: 0.3693, mae: 0.4284, huber: 0.1711, swd: 0.1608, ept: 160.1143
    Epoch [9/50], Test Losses: mse: 0.2661, mae: 0.3704, huber: 0.1268, swd: 0.1328, ept: 181.5215
      Epoch 9 composite train-obj: 0.134693
            Val objective improved 0.1762 → 0.1711, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.3544, mae: 0.3553, huber: 0.1332, swd: 0.1644, ept: 212.7931
    Epoch [10/50], Val Losses: mse: 0.3965, mae: 0.4416, huber: 0.1813, swd: 0.1844, ept: 158.6128
    Epoch [10/50], Test Losses: mse: 0.2633, mae: 0.3680, huber: 0.1254, swd: 0.1268, ept: 181.6281
      Epoch 10 composite train-obj: 0.133162
            No improvement (0.1813), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.3487, mae: 0.3522, huber: 0.1312, swd: 0.1608, ept: 214.1057
    Epoch [11/50], Val Losses: mse: 0.3909, mae: 0.4354, huber: 0.1795, swd: 0.1667, ept: 158.1822
    Epoch [11/50], Test Losses: mse: 0.2566, mae: 0.3553, huber: 0.1218, swd: 0.1159, ept: 182.9837
      Epoch 11 composite train-obj: 0.131150
            No improvement (0.1795), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.3452, mae: 0.3511, huber: 0.1302, swd: 0.1575, ept: 213.8183
    Epoch [12/50], Val Losses: mse: 0.4036, mae: 0.4412, huber: 0.1837, swd: 0.1847, ept: 158.0109
    Epoch [12/50], Test Losses: mse: 0.2730, mae: 0.3732, huber: 0.1296, swd: 0.1290, ept: 180.0056
      Epoch 12 composite train-obj: 0.130152
            No improvement (0.1837), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.3395, mae: 0.3482, huber: 0.1283, swd: 0.1534, ept: 214.6007
    Epoch [13/50], Val Losses: mse: 0.3839, mae: 0.4312, huber: 0.1766, swd: 0.1544, ept: 161.5091
    Epoch [13/50], Test Losses: mse: 0.2656, mae: 0.3625, huber: 0.1257, swd: 0.1183, ept: 183.5347
      Epoch 13 composite train-obj: 0.128266
            No improvement (0.1766), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.3342, mae: 0.3463, huber: 0.1267, swd: 0.1499, ept: 214.9561
    Epoch [14/50], Val Losses: mse: 0.4198, mae: 0.4514, huber: 0.1902, swd: 0.1871, ept: 157.4813
    Epoch [14/50], Test Losses: mse: 0.2800, mae: 0.3794, huber: 0.1327, swd: 0.1276, ept: 179.3242
      Epoch 14 composite train-obj: 0.126664
    Epoch [14/50], Test Losses: mse: 0.2661, mae: 0.3704, huber: 0.1268, swd: 0.1328, ept: 181.5250
    Best round's Test MSE: 0.2661, MAE: 0.3704, SWD: 0.1328
    Best round's Validation MSE: 0.3693, MAE: 0.4284, SWD: 0.1608
    Best round's Test verification MSE : 0.2661, MAE: 0.3704, SWD: 0.1328
    Time taken: 35.18 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5282, mae: 0.4601, huber: 0.2002, swd: 0.3067, ept: 171.6283
    Epoch [1/50], Val Losses: mse: 0.4165, mae: 0.4650, huber: 0.1911, swd: 0.2197, ept: 133.1969
    Epoch [1/50], Test Losses: mse: 0.2420, mae: 0.3461, huber: 0.1155, swd: 0.1155, ept: 179.6095
      Epoch 1 composite train-obj: 0.200207
            Val objective improved inf → 0.1911, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4382, mae: 0.4079, huber: 0.1660, swd: 0.2239, ept: 189.9770
    Epoch [2/50], Val Losses: mse: 0.3979, mae: 0.4521, huber: 0.1837, swd: 0.1939, ept: 139.3711
    Epoch [2/50], Test Losses: mse: 0.2383, mae: 0.3430, huber: 0.1133, swd: 0.1063, ept: 182.5305
      Epoch 2 composite train-obj: 0.166040
            Val objective improved 0.1911 → 0.1837, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4092, mae: 0.3896, huber: 0.1547, swd: 0.2043, ept: 196.0528
    Epoch [3/50], Val Losses: mse: 0.3960, mae: 0.4484, huber: 0.1826, swd: 0.1980, ept: 148.1858
    Epoch [3/50], Test Losses: mse: 0.2397, mae: 0.3530, huber: 0.1148, swd: 0.1158, ept: 182.3479
      Epoch 3 composite train-obj: 0.154680
            Val objective improved 0.1837 → 0.1826, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3905, mae: 0.3767, huber: 0.1470, swd: 0.1908, ept: 203.7410
    Epoch [4/50], Val Losses: mse: 0.3955, mae: 0.4457, huber: 0.1826, swd: 0.1883, ept: 154.0739
    Epoch [4/50], Test Losses: mse: 0.2386, mae: 0.3529, huber: 0.1145, swd: 0.1157, ept: 187.1570
      Epoch 4 composite train-obj: 0.147030
            No improvement (0.1826), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3806, mae: 0.3705, huber: 0.1431, swd: 0.1853, ept: 207.5247
    Epoch [5/50], Val Losses: mse: 0.3994, mae: 0.4455, huber: 0.1840, swd: 0.1921, ept: 154.1265
    Epoch [5/50], Test Losses: mse: 0.2396, mae: 0.3492, huber: 0.1144, swd: 0.1187, ept: 188.0006
      Epoch 5 composite train-obj: 0.143101
            No improvement (0.1840), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3741, mae: 0.3659, huber: 0.1404, swd: 0.1806, ept: 209.4084
    Epoch [6/50], Val Losses: mse: 0.3949, mae: 0.4426, huber: 0.1819, swd: 0.1920, ept: 157.6981
    Epoch [6/50], Test Losses: mse: 0.2526, mae: 0.3631, huber: 0.1208, swd: 0.1318, ept: 186.0844
      Epoch 6 composite train-obj: 0.140364
            Val objective improved 0.1826 → 0.1819, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3676, mae: 0.3625, huber: 0.1380, swd: 0.1778, ept: 210.9688
    Epoch [7/50], Val Losses: mse: 0.3870, mae: 0.4358, huber: 0.1788, swd: 0.1798, ept: 157.7330
    Epoch [7/50], Test Losses: mse: 0.2596, mae: 0.3655, huber: 0.1236, swd: 0.1328, ept: 186.1233
      Epoch 7 composite train-obj: 0.137957
            Val objective improved 0.1819 → 0.1788, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3627, mae: 0.3594, huber: 0.1361, swd: 0.1746, ept: 211.8462
    Epoch [8/50], Val Losses: mse: 0.4077, mae: 0.4476, huber: 0.1868, swd: 0.1982, ept: 155.8724
    Epoch [8/50], Test Losses: mse: 0.2650, mae: 0.3658, huber: 0.1254, swd: 0.1303, ept: 183.2743
      Epoch 8 composite train-obj: 0.136052
            No improvement (0.1868), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3591, mae: 0.3575, huber: 0.1347, swd: 0.1727, ept: 212.8855
    Epoch [9/50], Val Losses: mse: 0.3915, mae: 0.4375, huber: 0.1802, swd: 0.1754, ept: 156.1102
    Epoch [9/50], Test Losses: mse: 0.2728, mae: 0.3696, huber: 0.1282, swd: 0.1323, ept: 184.2384
      Epoch 9 composite train-obj: 0.134718
            No improvement (0.1802), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.3535, mae: 0.3546, huber: 0.1328, swd: 0.1691, ept: 213.6314
    Epoch [10/50], Val Losses: mse: 0.3871, mae: 0.4347, huber: 0.1786, swd: 0.1710, ept: 157.7330
    Epoch [10/50], Test Losses: mse: 0.2784, mae: 0.3780, huber: 0.1314, swd: 0.1386, ept: 184.1059
      Epoch 10 composite train-obj: 0.132758
            Val objective improved 0.1788 → 0.1786, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.3487, mae: 0.3518, huber: 0.1310, swd: 0.1658, ept: 214.9840
    Epoch [11/50], Val Losses: mse: 0.3816, mae: 0.4310, huber: 0.1758, swd: 0.1618, ept: 159.9504
    Epoch [11/50], Test Losses: mse: 0.2836, mae: 0.3829, huber: 0.1340, swd: 0.1378, ept: 184.2835
      Epoch 11 composite train-obj: 0.131029
            Val objective improved 0.1786 → 0.1758, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.3445, mae: 0.3499, huber: 0.1297, swd: 0.1627, ept: 215.4300
    Epoch [12/50], Val Losses: mse: 0.3994, mae: 0.4383, huber: 0.1822, swd: 0.1800, ept: 158.8723
    Epoch [12/50], Test Losses: mse: 0.2845, mae: 0.3828, huber: 0.1342, swd: 0.1392, ept: 183.7693
      Epoch 12 composite train-obj: 0.129695
            No improvement (0.1822), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.3388, mae: 0.3480, huber: 0.1281, swd: 0.1593, ept: 215.8079
    Epoch [13/50], Val Losses: mse: 0.3802, mae: 0.4297, huber: 0.1748, swd: 0.1557, ept: 160.9597
    Epoch [13/50], Test Losses: mse: 0.3007, mae: 0.3956, huber: 0.1416, swd: 0.1516, ept: 180.9877
      Epoch 13 composite train-obj: 0.128113
            Val objective improved 0.1758 → 0.1748, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.3334, mae: 0.3444, huber: 0.1261, swd: 0.1548, ept: 216.9323
    Epoch [14/50], Val Losses: mse: 0.3894, mae: 0.4377, huber: 0.1795, swd: 0.1552, ept: 158.5513
    Epoch [14/50], Test Losses: mse: 0.2927, mae: 0.3892, huber: 0.1383, swd: 0.1435, ept: 180.7234
      Epoch 14 composite train-obj: 0.126089
            No improvement (0.1795), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.3285, mae: 0.3429, huber: 0.1248, swd: 0.1513, ept: 216.9747
    Epoch [15/50], Val Losses: mse: 0.4010, mae: 0.4426, huber: 0.1836, swd: 0.1681, ept: 160.2839
    Epoch [15/50], Test Losses: mse: 0.2798, mae: 0.3737, huber: 0.1310, swd: 0.1319, ept: 183.5296
      Epoch 15 composite train-obj: 0.124780
            No improvement (0.1836), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.3231, mae: 0.3407, huber: 0.1230, swd: 0.1471, ept: 217.5499
    Epoch [16/50], Val Losses: mse: 0.4010, mae: 0.4434, huber: 0.1831, swd: 0.1666, ept: 159.5222
    Epoch [16/50], Test Losses: mse: 0.3036, mae: 0.3975, huber: 0.1427, swd: 0.1515, ept: 179.0213
      Epoch 16 composite train-obj: 0.123047
            No improvement (0.1831), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.3191, mae: 0.3390, huber: 0.1217, swd: 0.1436, ept: 217.6877
    Epoch [17/50], Val Losses: mse: 0.3979, mae: 0.4390, huber: 0.1815, swd: 0.1521, ept: 160.2928
    Epoch [17/50], Test Losses: mse: 0.2977, mae: 0.3920, huber: 0.1400, swd: 0.1425, ept: 181.5554
      Epoch 17 composite train-obj: 0.121738
            No improvement (0.1815), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.3146, mae: 0.3371, huber: 0.1203, swd: 0.1403, ept: 218.5887
    Epoch [18/50], Val Losses: mse: 0.3980, mae: 0.4439, huber: 0.1821, swd: 0.1417, ept: 158.2967
    Epoch [18/50], Test Losses: mse: 0.2876, mae: 0.3775, huber: 0.1336, swd: 0.1309, ept: 184.6683
      Epoch 18 composite train-obj: 0.120260
    Epoch [18/50], Test Losses: mse: 0.3006, mae: 0.3956, huber: 0.1415, swd: 0.1516, ept: 181.0300
    Best round's Test MSE: 0.3007, MAE: 0.3956, SWD: 0.1516
    Best round's Validation MSE: 0.3802, MAE: 0.4297, SWD: 0.1557
    Best round's Test verification MSE : 0.3006, MAE: 0.3956, SWD: 0.1516
    Time taken: 44.26 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5276, mae: 0.4592, huber: 0.1997, swd: 0.2893, ept: 171.5834
    Epoch [1/50], Val Losses: mse: 0.4067, mae: 0.4599, huber: 0.1870, swd: 0.2079, ept: 139.8471
    Epoch [1/50], Test Losses: mse: 0.2372, mae: 0.3433, huber: 0.1133, swd: 0.1043, ept: 182.6079
      Epoch 1 composite train-obj: 0.199669
            Val objective improved inf → 0.1870, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4385, mae: 0.4072, huber: 0.1660, swd: 0.2201, ept: 190.9873
    Epoch [2/50], Val Losses: mse: 0.4183, mae: 0.4667, huber: 0.1917, swd: 0.2319, ept: 137.6385
    Epoch [2/50], Test Losses: mse: 0.2394, mae: 0.3494, huber: 0.1146, swd: 0.1132, ept: 179.4876
      Epoch 2 composite train-obj: 0.165952
            No improvement (0.1917), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4080, mae: 0.3877, huber: 0.1541, swd: 0.1984, ept: 197.1307
    Epoch [3/50], Val Losses: mse: 0.3973, mae: 0.4460, huber: 0.1830, swd: 0.1832, ept: 149.0207
    Epoch [3/50], Test Losses: mse: 0.2363, mae: 0.3443, huber: 0.1125, swd: 0.1012, ept: 185.7626
      Epoch 3 composite train-obj: 0.154083
            Val objective improved 0.1870 → 0.1830, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3885, mae: 0.3750, huber: 0.1462, swd: 0.1853, ept: 204.4938
    Epoch [4/50], Val Losses: mse: 0.3793, mae: 0.4327, huber: 0.1756, swd: 0.1625, ept: 154.8328
    Epoch [4/50], Test Losses: mse: 0.2492, mae: 0.3603, huber: 0.1189, swd: 0.1127, ept: 187.6281
      Epoch 4 composite train-obj: 0.146205
            Val objective improved 0.1830 → 0.1756, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3792, mae: 0.3697, huber: 0.1426, swd: 0.1805, ept: 207.6060
    Epoch [5/50], Val Losses: mse: 0.3817, mae: 0.4374, huber: 0.1768, swd: 0.1657, ept: 156.6354
    Epoch [5/50], Test Losses: mse: 0.2483, mae: 0.3604, huber: 0.1189, swd: 0.1138, ept: 186.3892
      Epoch 5 composite train-obj: 0.142583
            No improvement (0.1768), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3708, mae: 0.3647, huber: 0.1394, swd: 0.1758, ept: 209.9918
    Epoch [6/50], Val Losses: mse: 0.4086, mae: 0.4580, huber: 0.1877, swd: 0.2055, ept: 151.8912
    Epoch [6/50], Test Losses: mse: 0.2777, mae: 0.3856, huber: 0.1329, swd: 0.1456, ept: 175.8147
      Epoch 6 composite train-obj: 0.139423
            No improvement (0.1877), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.3651, mae: 0.3616, huber: 0.1373, swd: 0.1732, ept: 211.0093
    Epoch [7/50], Val Losses: mse: 0.3824, mae: 0.4357, huber: 0.1772, swd: 0.1680, ept: 156.7582
    Epoch [7/50], Test Losses: mse: 0.2715, mae: 0.3741, huber: 0.1290, swd: 0.1279, ept: 182.8312
      Epoch 7 composite train-obj: 0.137291
            No improvement (0.1772), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.3581, mae: 0.3572, huber: 0.1348, swd: 0.1693, ept: 212.7226
    Epoch [8/50], Val Losses: mse: 0.3754, mae: 0.4331, huber: 0.1739, swd: 0.1635, ept: 158.6978
    Epoch [8/50], Test Losses: mse: 0.2700, mae: 0.3769, huber: 0.1288, swd: 0.1345, ept: 182.9494
      Epoch 8 composite train-obj: 0.134783
            Val objective improved 0.1756 → 0.1739, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.3542, mae: 0.3558, huber: 0.1334, swd: 0.1665, ept: 213.4139
    Epoch [9/50], Val Losses: mse: 0.3794, mae: 0.4357, huber: 0.1749, swd: 0.1696, ept: 158.2651
    Epoch [9/50], Test Losses: mse: 0.2838, mae: 0.3889, huber: 0.1354, swd: 0.1423, ept: 177.1431
      Epoch 9 composite train-obj: 0.133398
            No improvement (0.1749), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.3489, mae: 0.3533, huber: 0.1318, swd: 0.1631, ept: 213.9250
    Epoch [10/50], Val Losses: mse: 0.4120, mae: 0.4475, huber: 0.1880, swd: 0.1876, ept: 157.3897
    Epoch [10/50], Test Losses: mse: 0.2755, mae: 0.3766, huber: 0.1305, swd: 0.1254, ept: 181.3450
      Epoch 10 composite train-obj: 0.131796
            No improvement (0.1880), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.3420, mae: 0.3500, huber: 0.1295, swd: 0.1586, ept: 214.9325
    Epoch [11/50], Val Losses: mse: 0.4222, mae: 0.4566, huber: 0.1913, swd: 0.2080, ept: 157.2222
    Epoch [11/50], Test Losses: mse: 0.2964, mae: 0.3944, huber: 0.1401, swd: 0.1474, ept: 176.0969
      Epoch 11 composite train-obj: 0.129501
            No improvement (0.1913), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.3372, mae: 0.3474, huber: 0.1279, swd: 0.1553, ept: 215.4049
    Epoch [12/50], Val Losses: mse: 0.4156, mae: 0.4541, huber: 0.1889, swd: 0.1971, ept: 157.7257
    Epoch [12/50], Test Losses: mse: 0.3040, mae: 0.4029, huber: 0.1440, swd: 0.1567, ept: 171.8076
      Epoch 12 composite train-obj: 0.127899
            No improvement (0.1889), counter 4/5
    Epoch [13/50], Train Losses: mse: 0.3334, mae: 0.3458, huber: 0.1266, swd: 0.1519, ept: 216.0655
    Epoch [13/50], Val Losses: mse: 0.4141, mae: 0.4466, huber: 0.1881, swd: 0.1753, ept: 155.4802
    Epoch [13/50], Test Losses: mse: 0.2864, mae: 0.3814, huber: 0.1344, swd: 0.1286, ept: 181.6049
      Epoch 13 composite train-obj: 0.126590
    Epoch [13/50], Test Losses: mse: 0.2700, mae: 0.3769, huber: 0.1289, swd: 0.1345, ept: 182.9108
    Best round's Test MSE: 0.2700, MAE: 0.3769, SWD: 0.1345
    Best round's Validation MSE: 0.3754, MAE: 0.4331, SWD: 0.1635
    Best round's Test verification MSE : 0.2700, MAE: 0.3769, SWD: 0.1345
    Time taken: 32.32 seconds
    
    ==================================================
    Experiment Summary (ACL_etth2_seq96_pred336_20250510_2047)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2789 ± 0.0155
      mae: 0.3810 ± 0.0107
      huber: 0.1324 ± 0.0065
      swd: 0.1396 ± 0.0085
      ept: 181.8195 ± 0.8281
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3750 ± 0.0045
      mae: 0.4304 ± 0.0020
      huber: 0.1733 ± 0.0016
      swd: 0.1600 ± 0.0032
      ept: 159.9239 ± 0.9332
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 111.84 seconds
    
    Experiment complete: ACL_etth2_seq96_pred336_20250510_2047
    Model: ACL
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['etth2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 11
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 118.4013, mae: 5.8098, huber: 5.3677, swd: 89.1431, ept: 141.3680
    Epoch [1/50], Val Losses: mse: 31.4202, mae: 3.8186, huber: 3.3762, swd: 9.6569, ept: 139.8076
    Epoch [1/50], Test Losses: mse: 21.6811, mae: 3.0736, huber: 2.6340, swd: 9.0293, ept: 174.1085
      Epoch 1 composite train-obj: 5.367731
            Val objective improved inf → 3.3762, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.0979, mae: 3.5204, huber: 3.0879, swd: 18.3539, ept: 182.2493
    Epoch [2/50], Val Losses: mse: 30.1152, mae: 3.7745, huber: 3.3313, swd: 8.5415, ept: 143.8715
    Epoch [2/50], Test Losses: mse: 21.5496, mae: 3.0857, huber: 2.6442, swd: 9.1228, ept: 173.4751
      Epoch 2 composite train-obj: 3.087934
            Val objective improved 3.3762 → 3.3313, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.3061, mae: 3.4881, huber: 3.0561, swd: 18.1322, ept: 184.2394
    Epoch [3/50], Val Losses: mse: 33.8798, mae: 3.9674, huber: 3.5223, swd: 11.1551, ept: 134.6189
    Epoch [3/50], Test Losses: mse: 23.4925, mae: 3.2366, huber: 2.7929, swd: 10.4897, ept: 161.9219
      Epoch 3 composite train-obj: 3.056112
            No improvement (3.5223), counter 1/5
    Epoch [4/50], Train Losses: mse: 37.5659, mae: 3.4363, huber: 3.0044, swd: 17.9539, ept: 185.7961
    Epoch [4/50], Val Losses: mse: 29.1172, mae: 3.7488, huber: 3.3036, swd: 8.3752, ept: 145.8162
    Epoch [4/50], Test Losses: mse: 20.0570, mae: 2.9296, huber: 2.4928, swd: 8.5067, ept: 174.1605
      Epoch 4 composite train-obj: 3.004424
            Val objective improved 3.3313 → 3.3036, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 35.6785, mae: 3.2862, huber: 2.8559, swd: 16.7041, ept: 194.2276
    Epoch [5/50], Val Losses: mse: 28.5528, mae: 3.7002, huber: 3.2547, swd: 8.5801, ept: 149.2916
    Epoch [5/50], Test Losses: mse: 19.5573, mae: 2.9374, huber: 2.4974, swd: 8.5790, ept: 179.1437
      Epoch 5 composite train-obj: 2.855859
            Val objective improved 3.3036 → 3.2547, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 35.1456, mae: 3.2528, huber: 2.8226, swd: 16.5353, ept: 198.2073
    Epoch [6/50], Val Losses: mse: 29.4063, mae: 3.7234, huber: 3.2787, swd: 8.8972, ept: 149.0684
    Epoch [6/50], Test Losses: mse: 19.7224, mae: 2.9043, huber: 2.4674, swd: 8.5930, ept: 176.8176
      Epoch 6 composite train-obj: 2.822584
            No improvement (3.2787), counter 1/5
    Epoch [7/50], Train Losses: mse: 34.4630, mae: 3.2014, huber: 2.7719, swd: 16.0689, ept: 200.8642
    Epoch [7/50], Val Losses: mse: 28.2933, mae: 3.7160, huber: 3.2708, swd: 8.6851, ept: 153.0144
    Epoch [7/50], Test Losses: mse: 18.6200, mae: 2.8423, huber: 2.4062, swd: 8.0010, ept: 183.2437
      Epoch 7 composite train-obj: 2.771872
            No improvement (3.2708), counter 2/5
    Epoch [8/50], Train Losses: mse: 34.2168, mae: 3.2000, huber: 2.7704, swd: 15.9095, ept: 201.1461
    Epoch [8/50], Val Losses: mse: 27.7485, mae: 3.6138, huber: 3.1693, swd: 8.2003, ept: 154.7268
    Epoch [8/50], Test Losses: mse: 18.8470, mae: 2.8715, huber: 2.4326, swd: 8.2125, ept: 182.7637
      Epoch 8 composite train-obj: 2.770367
            Val objective improved 3.2547 → 3.1693, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 34.0334, mae: 3.1723, huber: 2.7432, swd: 15.8315, ept: 202.3954
    Epoch [9/50], Val Losses: mse: 28.8488, mae: 3.6695, huber: 3.2242, swd: 8.8894, ept: 152.6882
    Epoch [9/50], Test Losses: mse: 19.2523, mae: 2.9021, huber: 2.4622, swd: 8.3867, ept: 176.4590
      Epoch 9 composite train-obj: 2.743160
            No improvement (3.2242), counter 1/5
    Epoch [10/50], Train Losses: mse: 33.9201, mae: 3.1722, huber: 2.7429, swd: 15.8260, ept: 202.1168
    Epoch [10/50], Val Losses: mse: 27.5642, mae: 3.6401, huber: 3.1941, swd: 8.5735, ept: 156.1776
    Epoch [10/50], Test Losses: mse: 18.9494, mae: 2.9282, huber: 2.4860, swd: 8.6557, ept: 183.1909
      Epoch 10 composite train-obj: 2.742934
            No improvement (3.1941), counter 2/5
    Epoch [11/50], Train Losses: mse: 33.7726, mae: 3.1660, huber: 2.7367, swd: 15.8083, ept: 202.0836
    Epoch [11/50], Val Losses: mse: 26.9904, mae: 3.5619, huber: 3.1187, swd: 8.0589, ept: 156.2025
    Epoch [11/50], Test Losses: mse: 18.9565, mae: 2.9252, huber: 2.4828, swd: 8.5131, ept: 179.3080
      Epoch 11 composite train-obj: 2.736733
            Val objective improved 3.1693 → 3.1187, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 33.1608, mae: 3.1051, huber: 2.6771, swd: 15.2649, ept: 204.4376
    Epoch [12/50], Val Losses: mse: 27.8593, mae: 3.5831, huber: 3.1422, swd: 8.2615, ept: 154.1803
    Epoch [12/50], Test Losses: mse: 19.3079, mae: 2.8988, huber: 2.4591, swd: 8.4098, ept: 175.0442
      Epoch 12 composite train-obj: 2.677087
            No improvement (3.1422), counter 1/5
    Epoch [13/50], Train Losses: mse: 33.0443, mae: 3.1042, huber: 2.6761, swd: 15.1709, ept: 203.9407
    Epoch [13/50], Val Losses: mse: 28.1180, mae: 3.6173, huber: 3.1764, swd: 8.3990, ept: 154.7075
    Epoch [13/50], Test Losses: mse: 18.8728, mae: 2.8502, huber: 2.4122, swd: 8.1708, ept: 178.4978
      Epoch 13 composite train-obj: 2.676113
            No improvement (3.1764), counter 2/5
    Epoch [14/50], Train Losses: mse: 32.9078, mae: 3.0861, huber: 2.6587, swd: 15.0559, ept: 203.9999
    Epoch [14/50], Val Losses: mse: 28.3980, mae: 3.6285, huber: 3.1852, swd: 8.6510, ept: 153.2682
    Epoch [14/50], Test Losses: mse: 19.5429, mae: 2.9460, huber: 2.5025, swd: 8.6933, ept: 177.7996
      Epoch 14 composite train-obj: 2.658686
            No improvement (3.1852), counter 3/5
    Epoch [15/50], Train Losses: mse: 32.5923, mae: 3.0716, huber: 2.6442, swd: 14.8506, ept: 205.0263
    Epoch [15/50], Val Losses: mse: 28.8124, mae: 3.6305, huber: 3.1892, swd: 8.7550, ept: 152.5882
    Epoch [15/50], Test Losses: mse: 20.3222, mae: 2.9908, huber: 2.5487, swd: 9.3340, ept: 173.8300
      Epoch 15 composite train-obj: 2.644177
            No improvement (3.1892), counter 4/5
    Epoch [16/50], Train Losses: mse: 32.6881, mae: 3.0708, huber: 2.6437, swd: 14.9841, ept: 204.9550
    Epoch [16/50], Val Losses: mse: 27.6141, mae: 3.5724, huber: 3.1309, swd: 8.3664, ept: 156.5835
    Epoch [16/50], Test Losses: mse: 18.9239, mae: 2.8659, huber: 2.4263, swd: 8.2177, ept: 180.3204
      Epoch 16 composite train-obj: 2.643719
    Epoch [16/50], Test Losses: mse: 18.9565, mae: 2.9252, huber: 2.4828, swd: 8.5131, ept: 179.2869
    Best round's Test MSE: 18.9565, MAE: 2.9252, SWD: 8.5131
    Best round's Validation MSE: 26.9904, MAE: 3.5619, SWD: 8.0589
    Best round's Test verification MSE : 18.9565, MAE: 2.9252, SWD: 8.5131
    Time taken: 45.14 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 116.9037, mae: 5.7703, huber: 5.3278, swd: 93.4724, ept: 140.4390
    Epoch [1/50], Val Losses: mse: 30.9605, mae: 3.7995, huber: 3.3568, swd: 10.0943, ept: 142.0466
    Epoch [1/50], Test Losses: mse: 21.2504, mae: 3.0472, huber: 2.6075, swd: 9.1603, ept: 177.0750
      Epoch 1 composite train-obj: 5.327817
            Val objective improved inf → 3.3568, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.4509, mae: 3.5372, huber: 3.1042, swd: 19.8127, ept: 184.0710
    Epoch [2/50], Val Losses: mse: 31.0496, mae: 3.8055, huber: 3.3608, swd: 9.1572, ept: 139.4659
    Epoch [2/50], Test Losses: mse: 22.6148, mae: 3.1974, huber: 2.7528, swd: 10.1285, ept: 167.8571
      Epoch 2 composite train-obj: 3.104217
            No improvement (3.3608), counter 1/5
    Epoch [3/50], Train Losses: mse: 38.2646, mae: 3.4741, huber: 3.0414, swd: 18.9748, ept: 186.4523
    Epoch [3/50], Val Losses: mse: 29.8000, mae: 3.7673, huber: 3.3233, swd: 8.8995, ept: 144.4887
    Epoch [3/50], Test Losses: mse: 20.5664, mae: 2.9615, huber: 2.5249, swd: 8.9421, ept: 175.0989
      Epoch 3 composite train-obj: 3.041402
            Val objective improved 3.3568 → 3.3233, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.2630, mae: 3.4247, huber: 2.9924, swd: 18.5684, ept: 188.8282
    Epoch [4/50], Val Losses: mse: 29.6743, mae: 3.7513, huber: 3.3068, swd: 9.0548, ept: 144.5356
    Epoch [4/50], Test Losses: mse: 20.1119, mae: 2.9629, huber: 2.5234, swd: 8.8332, ept: 174.9799
      Epoch 4 composite train-obj: 2.992369
            Val objective improved 3.3233 → 3.3068, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 36.9102, mae: 3.3800, huber: 2.9481, swd: 18.6117, ept: 192.1750
    Epoch [5/50], Val Losses: mse: 29.0622, mae: 3.7373, huber: 3.2925, swd: 9.2805, ept: 148.6563
    Epoch [5/50], Test Losses: mse: 19.4751, mae: 2.9281, huber: 2.4885, swd: 8.8023, ept: 179.4827
      Epoch 5 composite train-obj: 2.948148
            Val objective improved 3.3068 → 3.2925, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 35.3306, mae: 3.2681, huber: 2.8376, swd: 17.4517, ept: 199.3115
    Epoch [6/50], Val Losses: mse: 28.3107, mae: 3.6704, huber: 3.2254, swd: 8.7335, ept: 150.0620
    Epoch [6/50], Test Losses: mse: 19.5673, mae: 2.9156, huber: 2.4765, swd: 9.0086, ept: 180.1661
      Epoch 6 composite train-obj: 2.837603
            Val objective improved 3.2925 → 3.2254, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 34.7351, mae: 3.2316, huber: 2.8017, swd: 16.9955, ept: 201.0237
    Epoch [7/50], Val Losses: mse: 29.0172, mae: 3.7003, huber: 3.2566, swd: 8.9355, ept: 148.6236
    Epoch [7/50], Test Losses: mse: 20.1488, mae: 2.9270, huber: 2.4898, swd: 9.2289, ept: 173.1230
      Epoch 7 composite train-obj: 2.801660
            No improvement (3.2566), counter 1/5
    Epoch [8/50], Train Losses: mse: 34.3290, mae: 3.2015, huber: 2.7722, swd: 16.6211, ept: 201.5586
    Epoch [8/50], Val Losses: mse: 27.9930, mae: 3.6812, huber: 3.2375, swd: 9.1548, ept: 152.5924
    Epoch [8/50], Test Losses: mse: 19.0335, mae: 2.8838, huber: 2.4455, swd: 8.9370, ept: 183.8787
      Epoch 8 composite train-obj: 2.772226
            No improvement (3.2375), counter 2/5
    Epoch [9/50], Train Losses: mse: 34.4296, mae: 3.2125, huber: 2.7833, swd: 16.8088, ept: 200.7243
    Epoch [9/50], Val Losses: mse: 29.2534, mae: 3.6543, huber: 3.2132, swd: 9.1516, ept: 149.4328
    Epoch [9/50], Test Losses: mse: 20.8009, mae: 2.9766, huber: 2.5380, swd: 9.9926, ept: 173.6990
      Epoch 9 composite train-obj: 2.783294
            Val objective improved 3.2254 → 3.2132, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 33.9025, mae: 3.1668, huber: 2.7381, swd: 16.4710, ept: 202.7902
    Epoch [10/50], Val Losses: mse: 27.1228, mae: 3.5707, huber: 3.1280, swd: 8.1980, ept: 154.9564
    Epoch [10/50], Test Losses: mse: 19.2162, mae: 2.9296, huber: 2.4873, swd: 8.9255, ept: 181.3777
      Epoch 10 composite train-obj: 2.738126
            Val objective improved 3.2132 → 3.1280, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 33.7239, mae: 3.1601, huber: 2.7316, swd: 16.4467, ept: 202.3837
    Epoch [11/50], Val Losses: mse: 27.9763, mae: 3.5900, huber: 3.1485, swd: 8.4970, ept: 154.1903
    Epoch [11/50], Test Losses: mse: 19.8601, mae: 2.9690, huber: 2.5260, swd: 9.2951, ept: 177.4096
      Epoch 11 composite train-obj: 2.731647
            No improvement (3.1485), counter 1/5
    Epoch [12/50], Train Losses: mse: 33.4399, mae: 3.1383, huber: 2.7103, swd: 16.2046, ept: 203.5388
    Epoch [12/50], Val Losses: mse: 27.6954, mae: 3.6008, huber: 3.1602, swd: 8.5437, ept: 155.2462
    Epoch [12/50], Test Losses: mse: 19.3871, mae: 2.9476, huber: 2.5059, swd: 9.0929, ept: 180.8420
      Epoch 12 composite train-obj: 2.710263
            No improvement (3.1602), counter 2/5
    Epoch [13/50], Train Losses: mse: 33.1061, mae: 3.1074, huber: 2.6799, swd: 15.9824, ept: 204.5821
    Epoch [13/50], Val Losses: mse: 26.9772, mae: 3.5451, huber: 3.1014, swd: 8.2986, ept: 156.7211
    Epoch [13/50], Test Losses: mse: 18.9013, mae: 2.8629, huber: 2.4240, swd: 8.8519, ept: 183.9173
      Epoch 13 composite train-obj: 2.679901
            Val objective improved 3.1280 → 3.1014, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 32.9871, mae: 3.1089, huber: 2.6814, swd: 15.9079, ept: 204.2369
    Epoch [14/50], Val Losses: mse: 28.1939, mae: 3.5904, huber: 3.1479, swd: 8.7009, ept: 153.8555
    Epoch [14/50], Test Losses: mse: 20.3681, mae: 3.0262, huber: 2.5815, swd: 9.6677, ept: 175.3330
      Epoch 14 composite train-obj: 2.681357
            No improvement (3.1479), counter 1/5
    Epoch [15/50], Train Losses: mse: 32.7320, mae: 3.0852, huber: 2.6578, swd: 15.6735, ept: 204.9083
    Epoch [15/50], Val Losses: mse: 26.8772, mae: 3.5197, huber: 3.0796, swd: 8.0574, ept: 156.9063
    Epoch [15/50], Test Losses: mse: 19.3331, mae: 2.9370, huber: 2.4952, swd: 9.1207, ept: 182.3700
      Epoch 15 composite train-obj: 2.657849
            Val objective improved 3.1014 → 3.0796, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 32.5695, mae: 3.0678, huber: 2.6409, swd: 15.5637, ept: 205.6798
    Epoch [16/50], Val Losses: mse: 28.5812, mae: 3.6426, huber: 3.1997, swd: 9.3648, ept: 154.4487
    Epoch [16/50], Test Losses: mse: 19.1246, mae: 2.9138, huber: 2.4716, swd: 8.6863, ept: 180.2647
      Epoch 16 composite train-obj: 2.640861
            No improvement (3.1997), counter 1/5
    Epoch [17/50], Train Losses: mse: 32.4382, mae: 3.0517, huber: 2.6253, swd: 15.4988, ept: 206.1341
    Epoch [17/50], Val Losses: mse: 27.5681, mae: 3.5712, huber: 3.1321, swd: 8.5162, ept: 155.7311
    Epoch [17/50], Test Losses: mse: 19.1329, mae: 2.8955, huber: 2.4558, swd: 8.8278, ept: 183.2264
      Epoch 17 composite train-obj: 2.625343
            No improvement (3.1321), counter 2/5
    Epoch [18/50], Train Losses: mse: 32.3814, mae: 3.0591, huber: 2.6327, swd: 15.5079, ept: 205.8828
    Epoch [18/50], Val Losses: mse: 26.9181, mae: 3.5038, huber: 3.0646, swd: 8.3185, ept: 157.2091
    Epoch [18/50], Test Losses: mse: 19.3760, mae: 2.9232, huber: 2.4828, swd: 9.0692, ept: 183.1794
      Epoch 18 composite train-obj: 2.632685
            Val objective improved 3.0796 → 3.0646, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 32.3102, mae: 3.0405, huber: 2.6147, swd: 15.3769, ept: 206.5630
    Epoch [19/50], Val Losses: mse: 27.7956, mae: 3.5938, huber: 3.1507, swd: 8.6546, ept: 155.4905
    Epoch [19/50], Test Losses: mse: 19.2328, mae: 2.8890, huber: 2.4493, swd: 8.9001, ept: 180.3516
      Epoch 19 composite train-obj: 2.614651
            No improvement (3.1507), counter 1/5
    Epoch [20/50], Train Losses: mse: 32.0085, mae: 3.0158, huber: 2.5902, swd: 15.1581, ept: 207.5189
    Epoch [20/50], Val Losses: mse: 27.1183, mae: 3.5308, huber: 3.0925, swd: 8.6616, ept: 158.2841
    Epoch [20/50], Test Losses: mse: 19.3338, mae: 2.9288, huber: 2.4885, swd: 9.1258, ept: 182.0850
      Epoch 20 composite train-obj: 2.590195
            No improvement (3.0925), counter 2/5
    Epoch [21/50], Train Losses: mse: 32.1698, mae: 3.0384, huber: 2.6126, swd: 15.3375, ept: 206.5379
    Epoch [21/50], Val Losses: mse: 27.5830, mae: 3.6116, huber: 3.1722, swd: 9.0916, ept: 157.7894
    Epoch [21/50], Test Losses: mse: 18.5074, mae: 2.8584, huber: 2.4202, swd: 8.4801, ept: 184.9621
      Epoch 21 composite train-obj: 2.612577
            No improvement (3.1722), counter 3/5
    Epoch [22/50], Train Losses: mse: 31.7865, mae: 3.0018, huber: 2.5766, swd: 15.0264, ept: 208.1163
    Epoch [22/50], Val Losses: mse: 26.7215, mae: 3.4941, huber: 3.0588, swd: 8.4268, ept: 157.3408
    Epoch [22/50], Test Losses: mse: 19.2470, mae: 2.8923, huber: 2.4540, swd: 9.1812, ept: 182.8351
      Epoch 22 composite train-obj: 2.576617
            Val objective improved 3.0646 → 3.0588, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 31.7899, mae: 3.0063, huber: 2.5810, swd: 15.0206, ept: 207.3919
    Epoch [23/50], Val Losses: mse: 27.4135, mae: 3.5746, huber: 3.1368, swd: 8.9068, ept: 157.1251
    Epoch [23/50], Test Losses: mse: 18.6070, mae: 2.8361, huber: 2.4002, swd: 8.5411, ept: 183.6744
      Epoch 23 composite train-obj: 2.581010
            No improvement (3.1368), counter 1/5
    Epoch [24/50], Train Losses: mse: 31.4554, mae: 2.9789, huber: 2.5543, swd: 14.7223, ept: 208.5852
    Epoch [24/50], Val Losses: mse: 27.0592, mae: 3.5017, huber: 3.0624, swd: 8.6808, ept: 157.5258
    Epoch [24/50], Test Losses: mse: 18.9731, mae: 2.8386, huber: 2.4019, swd: 8.8092, ept: 183.4107
      Epoch 24 composite train-obj: 2.554270
            No improvement (3.0624), counter 2/5
    Epoch [25/50], Train Losses: mse: 31.4282, mae: 2.9747, huber: 2.5502, swd: 14.7421, ept: 208.8667
    Epoch [25/50], Val Losses: mse: 26.9272, mae: 3.4982, huber: 3.0628, swd: 8.4983, ept: 157.2781
    Epoch [25/50], Test Losses: mse: 19.3071, mae: 2.8989, huber: 2.4605, swd: 9.0709, ept: 181.6701
      Epoch 25 composite train-obj: 2.550248
            No improvement (3.0628), counter 3/5
    Epoch [26/50], Train Losses: mse: 31.3227, mae: 2.9639, huber: 2.5398, swd: 14.6504, ept: 209.2213
    Epoch [26/50], Val Losses: mse: 26.9872, mae: 3.5056, huber: 3.0674, swd: 8.5568, ept: 158.4442
    Epoch [26/50], Test Losses: mse: 19.1848, mae: 2.8403, huber: 2.4063, swd: 8.9754, ept: 182.5280
      Epoch 26 composite train-obj: 2.539797
            No improvement (3.0674), counter 4/5
    Epoch [27/50], Train Losses: mse: 31.2489, mae: 2.9627, huber: 2.5386, swd: 14.5980, ept: 209.0966
    Epoch [27/50], Val Losses: mse: 27.7314, mae: 3.5384, huber: 3.0996, swd: 9.0473, ept: 156.9265
    Epoch [27/50], Test Losses: mse: 19.3605, mae: 2.8628, huber: 2.4267, swd: 8.9035, ept: 179.8952
      Epoch 27 composite train-obj: 2.538630
    Epoch [27/50], Test Losses: mse: 19.2470, mae: 2.8923, huber: 2.4540, swd: 9.1813, ept: 182.8169
    Best round's Test MSE: 19.2470, MAE: 2.8923, SWD: 9.1812
    Best round's Validation MSE: 26.7215, MAE: 3.4941, SWD: 8.4268
    Best round's Test verification MSE : 19.2470, MAE: 2.8923, SWD: 9.1813
    Time taken: 91.01 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 115.6907, mae: 5.7924, huber: 5.3506, swd: 81.3658, ept: 134.5163
    Epoch [1/50], Val Losses: mse: 31.1558, mae: 3.8116, huber: 3.3701, swd: 9.7092, ept: 139.2261
    Epoch [1/50], Test Losses: mse: 21.3326, mae: 3.0622, huber: 2.6221, swd: 8.4542, ept: 174.4745
      Epoch 1 composite train-obj: 5.350612
            Val objective improved inf → 3.3701, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 40.0584, mae: 3.5870, huber: 3.1537, swd: 18.9245, ept: 180.1823
    Epoch [2/50], Val Losses: mse: 30.6458, mae: 3.8018, huber: 3.3593, swd: 8.5717, ept: 141.2342
    Epoch [2/50], Test Losses: mse: 21.1448, mae: 3.0319, huber: 2.5928, swd: 8.1496, ept: 172.1425
      Epoch 2 composite train-obj: 3.153712
            Val objective improved 3.3701 → 3.3593, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.1056, mae: 3.4620, huber: 3.0298, swd: 17.4910, ept: 185.7015
    Epoch [3/50], Val Losses: mse: 29.6614, mae: 3.7869, huber: 3.3435, swd: 8.7997, ept: 143.9354
    Epoch [3/50], Test Losses: mse: 20.2036, mae: 2.9581, huber: 2.5206, swd: 8.1795, ept: 177.3781
      Epoch 3 composite train-obj: 3.029825
            Val objective improved 3.3593 → 3.3435, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.1563, mae: 3.3995, huber: 2.9675, swd: 17.3330, ept: 190.1612
    Epoch [4/50], Val Losses: mse: 29.1903, mae: 3.7755, huber: 3.3321, swd: 9.3667, ept: 148.5892
    Epoch [4/50], Test Losses: mse: 19.8869, mae: 2.9435, huber: 2.5064, swd: 8.7542, ept: 182.3182
      Epoch 4 composite train-obj: 2.967483
            Val objective improved 3.3435 → 3.3321, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 35.6903, mae: 3.2872, huber: 2.8564, swd: 16.4404, ept: 198.0162
    Epoch [5/50], Val Losses: mse: 28.1675, mae: 3.6829, huber: 3.2394, swd: 8.2581, ept: 148.9985
    Epoch [5/50], Test Losses: mse: 19.6019, mae: 2.9447, huber: 2.5053, swd: 8.3169, ept: 179.6491
      Epoch 5 composite train-obj: 2.856439
            Val objective improved 3.3321 → 3.2394, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 35.3257, mae: 3.2593, huber: 2.8290, swd: 16.2718, ept: 199.5239
    Epoch [6/50], Val Losses: mse: 29.2553, mae: 3.6867, huber: 3.2423, swd: 8.4565, ept: 148.3542
    Epoch [6/50], Test Losses: mse: 20.3452, mae: 2.9973, huber: 2.5551, swd: 8.5758, ept: 174.4306
      Epoch 6 composite train-obj: 2.828988
            No improvement (3.2423), counter 1/5
    Epoch [7/50], Train Losses: mse: 34.7726, mae: 3.2316, huber: 2.8018, swd: 15.9004, ept: 200.7111
    Epoch [7/50], Val Losses: mse: 29.9888, mae: 3.7260, huber: 3.2820, swd: 9.3151, ept: 149.1723
    Epoch [7/50], Test Losses: mse: 20.3162, mae: 2.9756, huber: 2.5350, swd: 8.6741, ept: 173.4193
      Epoch 7 composite train-obj: 2.801839
            No improvement (3.2820), counter 2/5
    Epoch [8/50], Train Losses: mse: 34.4998, mae: 3.2135, huber: 2.7840, swd: 15.7601, ept: 201.2915
    Epoch [8/50], Val Losses: mse: 28.5190, mae: 3.6223, huber: 3.1796, swd: 8.0675, ept: 151.9490
    Epoch [8/50], Test Losses: mse: 20.0270, mae: 2.9490, huber: 2.5086, swd: 8.4767, ept: 177.4097
      Epoch 8 composite train-obj: 2.783964
            Val objective improved 3.2394 → 3.1796, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 34.1249, mae: 3.1847, huber: 2.7560, swd: 15.4766, ept: 202.2900
    Epoch [9/50], Val Losses: mse: 30.5043, mae: 3.7072, huber: 3.2653, swd: 9.6990, ept: 149.2712
    Epoch [9/50], Test Losses: mse: 21.7526, mae: 3.1157, huber: 2.6699, swd: 9.8000, ept: 170.1370
      Epoch 9 composite train-obj: 2.756041
            No improvement (3.2653), counter 1/5
    Epoch [10/50], Train Losses: mse: 33.5766, mae: 3.1417, huber: 2.7140, swd: 15.0649, ept: 204.4295
    Epoch [10/50], Val Losses: mse: 28.1975, mae: 3.5736, huber: 3.1322, swd: 8.1908, ept: 152.3705
    Epoch [10/50], Test Losses: mse: 19.9718, mae: 2.9791, huber: 2.5352, swd: 8.5735, ept: 179.0600
      Epoch 10 composite train-obj: 2.714045
            Val objective improved 3.1796 → 3.1322, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 33.4308, mae: 3.1316, huber: 2.7042, swd: 15.0261, ept: 204.5640
    Epoch [11/50], Val Losses: mse: 29.9670, mae: 3.6512, huber: 3.2094, swd: 9.3855, ept: 151.9875
    Epoch [11/50], Test Losses: mse: 21.9639, mae: 3.1204, huber: 2.6736, swd: 10.1149, ept: 171.3303
      Epoch 11 composite train-obj: 2.704213
            No improvement (3.2094), counter 1/5
    Epoch [12/50], Train Losses: mse: 33.1837, mae: 3.1148, huber: 2.6879, swd: 14.8854, ept: 204.8752
    Epoch [12/50], Val Losses: mse: 29.5394, mae: 3.6560, huber: 3.2138, swd: 9.2156, ept: 152.6835
    Epoch [12/50], Test Losses: mse: 20.5267, mae: 2.9886, huber: 2.5469, swd: 8.8775, ept: 174.9905
      Epoch 12 composite train-obj: 2.687899
            No improvement (3.2138), counter 2/5
    Epoch [13/50], Train Losses: mse: 33.2891, mae: 3.1228, huber: 2.6958, swd: 15.0312, ept: 204.2888
    Epoch [13/50], Val Losses: mse: 28.0391, mae: 3.5565, huber: 3.1186, swd: 8.2486, ept: 154.4571
    Epoch [13/50], Test Losses: mse: 19.7275, mae: 2.9012, huber: 2.4641, swd: 8.2515, ept: 179.8498
      Epoch 13 composite train-obj: 2.695808
            Val objective improved 3.1322 → 3.1186, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 33.1436, mae: 3.1127, huber: 2.6857, swd: 15.0014, ept: 204.7255
    Epoch [14/50], Val Losses: mse: 28.8388, mae: 3.6307, huber: 3.1908, swd: 9.1292, ept: 153.5228
    Epoch [14/50], Test Losses: mse: 19.5378, mae: 2.9222, huber: 2.4826, swd: 8.2999, ept: 182.6608
      Epoch 14 composite train-obj: 2.685719
            No improvement (3.1908), counter 1/5
    Epoch [15/50], Train Losses: mse: 33.0579, mae: 3.1044, huber: 2.6775, swd: 14.9337, ept: 204.8698
    Epoch [15/50], Val Losses: mse: 28.5015, mae: 3.6103, huber: 3.1674, swd: 8.2534, ept: 153.1145
    Epoch [15/50], Test Losses: mse: 19.8466, mae: 2.9811, huber: 2.5358, swd: 8.3280, ept: 174.6512
      Epoch 15 composite train-obj: 2.677523
            No improvement (3.1674), counter 2/5
    Epoch [16/50], Train Losses: mse: 32.5720, mae: 3.0724, huber: 2.6460, swd: 14.4736, ept: 205.9285
    Epoch [16/50], Val Losses: mse: 26.9540, mae: 3.5061, huber: 3.0671, swd: 7.8845, ept: 156.1550
    Epoch [16/50], Test Losses: mse: 18.9100, mae: 2.8596, huber: 2.4215, swd: 7.9397, ept: 183.8922
      Epoch 16 composite train-obj: 2.646036
            Val objective improved 3.1186 → 3.0671, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 33.1602, mae: 3.1141, huber: 2.6872, swd: 15.0016, ept: 203.9307
    Epoch [17/50], Val Losses: mse: 27.5019, mae: 3.5664, huber: 3.1265, swd: 8.1766, ept: 156.0836
    Epoch [17/50], Test Losses: mse: 18.8968, mae: 2.8880, huber: 2.4480, swd: 7.7054, ept: 180.4933
      Epoch 17 composite train-obj: 2.687156
            No improvement (3.1265), counter 1/5
    Epoch [18/50], Train Losses: mse: 32.3498, mae: 3.0540, huber: 2.6279, swd: 14.4130, ept: 206.4913
    Epoch [18/50], Val Losses: mse: 27.9148, mae: 3.5542, huber: 3.1157, swd: 8.2812, ept: 156.4845
    Epoch [18/50], Test Losses: mse: 19.3496, mae: 2.8503, huber: 2.4146, swd: 8.0522, ept: 181.9258
      Epoch 18 composite train-obj: 2.627914
            No improvement (3.1157), counter 2/5
    Epoch [19/50], Train Losses: mse: 32.0829, mae: 3.0343, huber: 2.6085, swd: 14.2209, ept: 207.3286
    Epoch [19/50], Val Losses: mse: 29.4364, mae: 3.6184, huber: 3.1798, swd: 9.2010, ept: 153.9599
    Epoch [19/50], Test Losses: mse: 21.5458, mae: 3.0364, huber: 2.5986, swd: 9.6747, ept: 172.9984
      Epoch 19 composite train-obj: 2.608474
            No improvement (3.1798), counter 3/5
    Epoch [20/50], Train Losses: mse: 32.0813, mae: 3.0289, huber: 2.6033, swd: 14.2500, ept: 207.6636
    Epoch [20/50], Val Losses: mse: 28.4406, mae: 3.5877, huber: 3.1521, swd: 8.6683, ept: 155.2088
    Epoch [20/50], Test Losses: mse: 19.3829, mae: 2.8676, huber: 2.4333, swd: 8.0023, ept: 180.6191
      Epoch 20 composite train-obj: 2.603300
            No improvement (3.1521), counter 4/5
    Epoch [21/50], Train Losses: mse: 32.0352, mae: 3.0321, huber: 2.6066, swd: 14.2282, ept: 207.5160
    Epoch [21/50], Val Losses: mse: 28.8672, mae: 3.5753, huber: 3.1392, swd: 8.9615, ept: 155.0010
    Epoch [21/50], Test Losses: mse: 21.1806, mae: 2.9999, huber: 2.5629, swd: 9.3993, ept: 173.0850
      Epoch 21 composite train-obj: 2.606585
    Epoch [21/50], Test Losses: mse: 18.9102, mae: 2.8597, huber: 2.4215, swd: 7.9398, ept: 183.8806
    Best round's Test MSE: 18.9100, MAE: 2.8596, SWD: 7.9397
    Best round's Validation MSE: 26.9540, MAE: 3.5061, SWD: 7.8845
    Best round's Test verification MSE : 18.9102, MAE: 2.8597, SWD: 7.9398
    Time taken: 68.89 seconds
    
    ==================================================
    Experiment Summary (ACL_etth2_seq96_pred336_20250511_1610)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 19.0378 ± 0.1491
      mae: 2.8924 ± 0.0267
      huber: 2.4528 ± 0.0251
      swd: 8.5447 ± 0.5074
      ept: 182.0118 ± 1.9600
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 26.8886 ± 0.1191
      mae: 3.5207 ± 0.0295
      huber: 3.0815 ± 0.0265
      swd: 8.1234 ± 0.2261
      ept: 156.5661 ± 0.5481
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 205.11 seconds
    
    Experiment complete: ACL_etth2_seq96_pred336_20250511_1610
    Model: ACL
    Dataset: etth2
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
    channels=data_mgr.datasets['etth2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.6509, mae: 0.5265, huber: 0.2447, swd: 0.3769, ept: 260.9380
    Epoch [1/50], Val Losses: mse: 0.3933, mae: 0.4671, huber: 0.1841, swd: 0.1646, ept: 225.3072
    Epoch [1/50], Test Losses: mse: 0.2750, mae: 0.3729, huber: 0.1304, swd: 0.1254, ept: 301.1780
      Epoch 1 composite train-obj: 0.244689
            Val objective improved inf → 0.1841, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5678, mae: 0.4740, huber: 0.2109, swd: 0.3009, ept: 291.2636
    Epoch [2/50], Val Losses: mse: 0.4017, mae: 0.4633, huber: 0.1877, swd: 0.1553, ept: 232.1943
    Epoch [2/50], Test Losses: mse: 0.3069, mae: 0.4044, huber: 0.1455, swd: 0.1475, ept: 259.1643
      Epoch 2 composite train-obj: 0.210949
            No improvement (0.1877), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5348, mae: 0.4537, huber: 0.1976, swd: 0.2741, ept: 303.9037
    Epoch [3/50], Val Losses: mse: 0.4103, mae: 0.4728, huber: 0.1909, swd: 0.1801, ept: 242.3345
    Epoch [3/50], Test Losses: mse: 0.3334, mae: 0.4230, huber: 0.1594, swd: 0.1736, ept: 240.8326
      Epoch 3 composite train-obj: 0.197617
            No improvement (0.1909), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.5180, mae: 0.4427, huber: 0.1907, swd: 0.2633, ept: 311.8064
    Epoch [4/50], Val Losses: mse: 0.3923, mae: 0.4572, huber: 0.1833, swd: 0.1629, ept: 255.2270
    Epoch [4/50], Test Losses: mse: 0.3170, mae: 0.4149, huber: 0.1520, swd: 0.1637, ept: 245.0183
      Epoch 4 composite train-obj: 0.190654
            Val objective improved 0.1841 → 0.1833, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5115, mae: 0.4385, huber: 0.1879, swd: 0.2585, ept: 313.4977
    Epoch [5/50], Val Losses: mse: 0.3922, mae: 0.4510, huber: 0.1833, swd: 0.1586, ept: 261.3975
    Epoch [5/50], Test Losses: mse: 0.3221, mae: 0.4164, huber: 0.1538, swd: 0.1603, ept: 258.1491
      Epoch 5 composite train-obj: 0.187880
            Val objective improved 0.1833 → 0.1833, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5023, mae: 0.4337, huber: 0.1845, swd: 0.2541, ept: 315.9970
    Epoch [6/50], Val Losses: mse: 0.3616, mae: 0.4338, huber: 0.1707, swd: 0.1356, ept: 264.8814
    Epoch [6/50], Test Losses: mse: 0.3312, mae: 0.4259, huber: 0.1587, swd: 0.1687, ept: 251.6192
      Epoch 6 composite train-obj: 0.184505
            Val objective improved 0.1833 → 0.1707, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4953, mae: 0.4300, huber: 0.1820, swd: 0.2509, ept: 317.6434
    Epoch [7/50], Val Losses: mse: 0.3806, mae: 0.4456, huber: 0.1784, swd: 0.1424, ept: 266.1820
    Epoch [7/50], Test Losses: mse: 0.3234, mae: 0.4186, huber: 0.1550, swd: 0.1546, ept: 256.8421
      Epoch 7 composite train-obj: 0.181966
            No improvement (0.1784), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4891, mae: 0.4267, huber: 0.1797, swd: 0.2477, ept: 320.1403
    Epoch [8/50], Val Losses: mse: 0.3765, mae: 0.4390, huber: 0.1763, swd: 0.1276, ept: 273.5386
    Epoch [8/50], Test Losses: mse: 0.3037, mae: 0.4066, huber: 0.1451, swd: 0.1289, ept: 277.7378
      Epoch 8 composite train-obj: 0.179713
            No improvement (0.1763), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.4800, mae: 0.4223, huber: 0.1765, swd: 0.2423, ept: 321.5611
    Epoch [9/50], Val Losses: mse: 0.3798, mae: 0.4408, huber: 0.1776, swd: 0.1447, ept: 266.0600
    Epoch [9/50], Test Losses: mse: 0.3358, mae: 0.4339, huber: 0.1595, swd: 0.1603, ept: 259.4873
      Epoch 9 composite train-obj: 0.176513
            No improvement (0.1776), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.4726, mae: 0.4185, huber: 0.1738, swd: 0.2380, ept: 323.8208
    Epoch [10/50], Val Losses: mse: 0.3761, mae: 0.4374, huber: 0.1763, swd: 0.1220, ept: 271.3418
    Epoch [10/50], Test Losses: mse: 0.3252, mae: 0.4217, huber: 0.1546, swd: 0.1456, ept: 268.5640
      Epoch 10 composite train-obj: 0.173840
            No improvement (0.1763), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.4625, mae: 0.4139, huber: 0.1705, swd: 0.2302, ept: 326.0811
    Epoch [11/50], Val Losses: mse: 0.3667, mae: 0.4387, huber: 0.1721, swd: 0.1213, ept: 273.6358
    Epoch [11/50], Test Losses: mse: 0.3286, mae: 0.4275, huber: 0.1564, swd: 0.1472, ept: 265.4929
      Epoch 11 composite train-obj: 0.170534
    Epoch [11/50], Test Losses: mse: 0.3312, mae: 0.4259, huber: 0.1587, swd: 0.1687, ept: 251.6192
    Best round's Test MSE: 0.3312, MAE: 0.4259, SWD: 0.1687
    Best round's Validation MSE: 0.3616, MAE: 0.4338, SWD: 0.1356
    Best round's Test verification MSE : 0.3312, MAE: 0.4259, SWD: 0.1687
    Time taken: 28.40 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6486, mae: 0.5253, huber: 0.2440, swd: 0.3597, ept: 263.3417
    Epoch [1/50], Val Losses: mse: 0.3954, mae: 0.4701, huber: 0.1853, swd: 0.1562, ept: 227.0957
    Epoch [1/50], Test Losses: mse: 0.2789, mae: 0.3770, huber: 0.1322, swd: 0.1254, ept: 300.9165
      Epoch 1 composite train-obj: 0.243955
            Val objective improved inf → 0.1853, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5744, mae: 0.4787, huber: 0.2138, swd: 0.2962, ept: 290.1748
    Epoch [2/50], Val Losses: mse: 0.3797, mae: 0.4575, huber: 0.1788, swd: 0.1403, ept: 229.6511
    Epoch [2/50], Test Losses: mse: 0.2929, mae: 0.3989, huber: 0.1397, swd: 0.1269, ept: 274.0200
      Epoch 2 composite train-obj: 0.213790
            Val objective improved 0.1853 → 0.1788, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5393, mae: 0.4563, huber: 0.1993, swd: 0.2703, ept: 300.3112
    Epoch [3/50], Val Losses: mse: 0.4099, mae: 0.4695, huber: 0.1908, swd: 0.1764, ept: 238.7403
    Epoch [3/50], Test Losses: mse: 0.3102, mae: 0.4101, huber: 0.1487, swd: 0.1543, ept: 244.8679
      Epoch 3 composite train-obj: 0.199349
            No improvement (0.1908), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5197, mae: 0.4427, huber: 0.1909, swd: 0.2568, ept: 314.1552
    Epoch [4/50], Val Losses: mse: 0.3879, mae: 0.4510, huber: 0.1815, swd: 0.1537, ept: 254.4163
    Epoch [4/50], Test Losses: mse: 0.3060, mae: 0.4068, huber: 0.1463, swd: 0.1496, ept: 264.0498
      Epoch 4 composite train-obj: 0.190879
            No improvement (0.1815), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5107, mae: 0.4371, huber: 0.1872, swd: 0.2517, ept: 317.2420
    Epoch [5/50], Val Losses: mse: 0.3709, mae: 0.4344, huber: 0.1742, swd: 0.1293, ept: 265.9844
    Epoch [5/50], Test Losses: mse: 0.2955, mae: 0.3970, huber: 0.1407, swd: 0.1333, ept: 289.9686
      Epoch 5 composite train-obj: 0.187216
            Val objective improved 0.1788 → 0.1742, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.5031, mae: 0.4324, huber: 0.1842, swd: 0.2478, ept: 319.8671
    Epoch [6/50], Val Losses: mse: 0.3688, mae: 0.4467, huber: 0.1737, swd: 0.1371, ept: 263.2468
    Epoch [6/50], Test Losses: mse: 0.3090, mae: 0.4105, huber: 0.1484, swd: 0.1483, ept: 261.1260
      Epoch 6 composite train-obj: 0.184160
            Val objective improved 0.1742 → 0.1737, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4965, mae: 0.4299, huber: 0.1821, swd: 0.2449, ept: 320.0363
    Epoch [7/50], Val Losses: mse: 0.3778, mae: 0.4472, huber: 0.1775, swd: 0.1393, ept: 258.2819
    Epoch [7/50], Test Losses: mse: 0.3026, mae: 0.4031, huber: 0.1444, swd: 0.1396, ept: 267.5265
      Epoch 7 composite train-obj: 0.182055
            No improvement (0.1775), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4898, mae: 0.4267, huber: 0.1797, swd: 0.2414, ept: 320.7046
    Epoch [8/50], Val Losses: mse: 0.3670, mae: 0.4362, huber: 0.1723, swd: 0.1318, ept: 262.5196
    Epoch [8/50], Test Losses: mse: 0.3402, mae: 0.4324, huber: 0.1614, swd: 0.1619, ept: 251.9901
      Epoch 8 composite train-obj: 0.179729
            Val objective improved 0.1737 → 0.1723, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4830, mae: 0.4229, huber: 0.1772, swd: 0.2380, ept: 321.8321
    Epoch [9/50], Val Losses: mse: 0.3629, mae: 0.4340, huber: 0.1708, swd: 0.1239, ept: 267.1012
    Epoch [9/50], Test Losses: mse: 0.3169, mae: 0.4130, huber: 0.1499, swd: 0.1394, ept: 269.5528
      Epoch 9 composite train-obj: 0.177238
            Val objective improved 0.1723 → 0.1708, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.4751, mae: 0.4190, huber: 0.1745, swd: 0.2331, ept: 323.1795
    Epoch [10/50], Val Losses: mse: 0.3798, mae: 0.4401, huber: 0.1776, swd: 0.1124, ept: 278.0802
    Epoch [10/50], Test Losses: mse: 0.3259, mae: 0.4174, huber: 0.1532, swd: 0.1371, ept: 277.9015
      Epoch 10 composite train-obj: 0.174466
            No improvement (0.1776), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.4663, mae: 0.4149, huber: 0.1715, swd: 0.2275, ept: 324.6948
    Epoch [11/50], Val Losses: mse: 0.4072, mae: 0.4516, huber: 0.1873, swd: 0.1264, ept: 270.4894
    Epoch [11/50], Test Losses: mse: 0.3371, mae: 0.4295, huber: 0.1608, swd: 0.1346, ept: 260.9093
      Epoch 11 composite train-obj: 0.171520
            No improvement (0.1873), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.4610, mae: 0.4135, huber: 0.1701, swd: 0.2206, ept: 325.3893
    Epoch [12/50], Val Losses: mse: 0.3789, mae: 0.4374, huber: 0.1765, swd: 0.1083, ept: 280.5577
    Epoch [12/50], Test Losses: mse: 0.3317, mae: 0.4239, huber: 0.1565, swd: 0.1346, ept: 274.5203
      Epoch 12 composite train-obj: 0.170053
            No improvement (0.1765), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.4463, mae: 0.4087, huber: 0.1658, swd: 0.2077, ept: 326.8611
    Epoch [13/50], Val Losses: mse: 0.3917, mae: 0.4652, huber: 0.1830, swd: 0.1359, ept: 262.8262
    Epoch [13/50], Test Losses: mse: 0.3441, mae: 0.4379, huber: 0.1639, swd: 0.1577, ept: 254.7452
      Epoch 13 composite train-obj: 0.165791
            No improvement (0.1830), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.4332, mae: 0.4028, huber: 0.1613, swd: 0.1962, ept: 327.0241
    Epoch [14/50], Val Losses: mse: 0.4000, mae: 0.4514, huber: 0.1852, swd: 0.1112, ept: 279.1973
    Epoch [14/50], Test Losses: mse: 0.3390, mae: 0.4277, huber: 0.1588, swd: 0.1378, ept: 279.1260
      Epoch 14 composite train-obj: 0.161250
    Epoch [14/50], Test Losses: mse: 0.3169, mae: 0.4130, huber: 0.1499, swd: 0.1395, ept: 269.5262
    Best round's Test MSE: 0.3169, MAE: 0.4130, SWD: 0.1394
    Best round's Validation MSE: 0.3629, MAE: 0.4340, SWD: 0.1239
    Best round's Test verification MSE : 0.3169, MAE: 0.4130, SWD: 0.1395
    Time taken: 34.89 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6423, mae: 0.5216, huber: 0.2416, swd: 0.3807, ept: 264.7537
    Epoch [1/50], Val Losses: mse: 0.3906, mae: 0.4584, huber: 0.1825, swd: 0.1362, ept: 224.2236
    Epoch [1/50], Test Losses: mse: 0.2829, mae: 0.3741, huber: 0.1332, swd: 0.1172, ept: 300.3350
      Epoch 1 composite train-obj: 0.241622
            Val objective improved inf → 0.1825, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5654, mae: 0.4728, huber: 0.2102, swd: 0.3099, ept: 293.9491
    Epoch [2/50], Val Losses: mse: 0.4146, mae: 0.4713, huber: 0.1929, swd: 0.1438, ept: 248.1932
    Epoch [2/50], Test Losses: mse: 0.3024, mae: 0.4022, huber: 0.1426, swd: 0.1332, ept: 274.0129
      Epoch 2 composite train-obj: 0.210165
            No improvement (0.1929), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.5319, mae: 0.4512, huber: 0.1963, swd: 0.2830, ept: 306.3500
    Epoch [3/50], Val Losses: mse: 0.3837, mae: 0.4481, huber: 0.1804, swd: 0.1286, ept: 255.8533
    Epoch [3/50], Test Losses: mse: 0.3155, mae: 0.4137, huber: 0.1506, swd: 0.1506, ept: 261.7899
      Epoch 3 composite train-obj: 0.196314
            Val objective improved 0.1825 → 0.1804, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5170, mae: 0.4418, huber: 0.1902, swd: 0.2730, ept: 313.6342
    Epoch [4/50], Val Losses: mse: 0.3855, mae: 0.4448, huber: 0.1803, swd: 0.1388, ept: 264.6192
    Epoch [4/50], Test Losses: mse: 0.3076, mae: 0.4047, huber: 0.1467, swd: 0.1455, ept: 269.8761
      Epoch 4 composite train-obj: 0.190160
            Val objective improved 0.1804 → 0.1803, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5069, mae: 0.4354, huber: 0.1860, swd: 0.2672, ept: 317.9604
    Epoch [5/50], Val Losses: mse: 0.3854, mae: 0.4529, huber: 0.1803, swd: 0.1568, ept: 259.5495
    Epoch [5/50], Test Losses: mse: 0.3084, mae: 0.4085, huber: 0.1480, swd: 0.1534, ept: 265.3218
      Epoch 5 composite train-obj: 0.186038
            No improvement (0.1803), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.5015, mae: 0.4326, huber: 0.1840, swd: 0.2643, ept: 318.8210
    Epoch [6/50], Val Losses: mse: 0.3823, mae: 0.4445, huber: 0.1792, swd: 0.1397, ept: 268.7868
    Epoch [6/50], Test Losses: mse: 0.3361, mae: 0.4214, huber: 0.1582, swd: 0.1723, ept: 265.3471
      Epoch 6 composite train-obj: 0.183979
            Val objective improved 0.1803 → 0.1792, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4933, mae: 0.4280, huber: 0.1810, swd: 0.2603, ept: 320.1586
    Epoch [7/50], Val Losses: mse: 0.3724, mae: 0.4377, huber: 0.1746, swd: 0.1373, ept: 265.2739
    Epoch [7/50], Test Losses: mse: 0.3168, mae: 0.4147, huber: 0.1517, swd: 0.1569, ept: 262.6948
      Epoch 7 composite train-obj: 0.180973
            Val objective improved 0.1792 → 0.1746, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4856, mae: 0.4248, huber: 0.1785, swd: 0.2559, ept: 321.6646
    Epoch [8/50], Val Losses: mse: 0.3676, mae: 0.4335, huber: 0.1726, swd: 0.1280, ept: 268.7279
    Epoch [8/50], Test Losses: mse: 0.3393, mae: 0.4314, huber: 0.1625, swd: 0.1767, ept: 254.4583
      Epoch 8 composite train-obj: 0.178472
            Val objective improved 0.1746 → 0.1726, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.4805, mae: 0.4226, huber: 0.1767, swd: 0.2522, ept: 322.1315
    Epoch [9/50], Val Losses: mse: 0.3719, mae: 0.4349, huber: 0.1743, swd: 0.1228, ept: 272.6750
    Epoch [9/50], Test Losses: mse: 0.3259, mae: 0.4199, huber: 0.1540, swd: 0.1573, ept: 274.4068
      Epoch 9 composite train-obj: 0.176669
            No improvement (0.1743), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.4700, mae: 0.4168, huber: 0.1728, swd: 0.2439, ept: 324.7384
    Epoch [10/50], Val Losses: mse: 0.3732, mae: 0.4359, huber: 0.1746, swd: 0.1293, ept: 266.0795
    Epoch [10/50], Test Losses: mse: 0.3166, mae: 0.4147, huber: 0.1501, swd: 0.1557, ept: 272.5989
      Epoch 10 composite train-obj: 0.172830
            No improvement (0.1746), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.4588, mae: 0.4126, huber: 0.1695, swd: 0.2362, ept: 326.1782
    Epoch [11/50], Val Losses: mse: 0.3792, mae: 0.4374, huber: 0.1770, swd: 0.1086, ept: 273.3815
    Epoch [11/50], Test Losses: mse: 0.3267, mae: 0.4206, huber: 0.1542, swd: 0.1472, ept: 274.7776
      Epoch 11 composite train-obj: 0.169521
            No improvement (0.1770), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.4475, mae: 0.4086, huber: 0.1661, swd: 0.2226, ept: 326.7360
    Epoch [12/50], Val Losses: mse: 0.3918, mae: 0.4456, huber: 0.1813, swd: 0.1166, ept: 275.5329
    Epoch [12/50], Test Losses: mse: 0.3333, mae: 0.4261, huber: 0.1573, swd: 0.1490, ept: 262.3244
      Epoch 12 composite train-obj: 0.166056
            No improvement (0.1813), counter 4/5
    Epoch [13/50], Train Losses: mse: 0.4348, mae: 0.4027, huber: 0.1616, swd: 0.2111, ept: 328.8654
    Epoch [13/50], Val Losses: mse: 0.3748, mae: 0.4359, huber: 0.1750, swd: 0.1045, ept: 273.5358
    Epoch [13/50], Test Losses: mse: 0.3292, mae: 0.4150, huber: 0.1534, swd: 0.1585, ept: 291.1229
      Epoch 13 composite train-obj: 0.161642
    Epoch [13/50], Test Losses: mse: 0.3393, mae: 0.4314, huber: 0.1625, swd: 0.1767, ept: 254.4406
    Best round's Test MSE: 0.3393, MAE: 0.4314, SWD: 0.1767
    Best round's Validation MSE: 0.3676, MAE: 0.4335, SWD: 0.1280
    Best round's Test verification MSE : 0.3393, MAE: 0.4314, SWD: 0.1767
    Time taken: 31.46 seconds
    
    ==================================================
    Experiment Summary (ACL_etth2_seq96_pred720_20250510_2049)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.3291 ± 0.0092
      mae: 0.4234 ± 0.0077
      huber: 0.1571 ± 0.0053
      swd: 0.1616 ± 0.0160
      ept: 258.5434 ± 7.8706
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3641 ± 0.0026
      mae: 0.4338 ± 0.0002
      huber: 0.1713 ± 0.0009
      swd: 0.1292 ± 0.0048
      ept: 266.9035 ± 1.5766
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 94.86 seconds
    
    Experiment complete: ACL_etth2_seq96_pred720_20250510_2049
    Model: ACL
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['etth2']['channels'],# data_mgr.channels,              # ← number of features in your data
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
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 11
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 118.4013, mae: 5.8098, huber: 5.3677, swd: 89.1431, ept: 141.3680
    Epoch [1/50], Val Losses: mse: 31.4202, mae: 3.8186, huber: 3.3762, swd: 9.6569, ept: 139.8076
    Epoch [1/50], Test Losses: mse: 21.6811, mae: 3.0736, huber: 2.6340, swd: 9.0293, ept: 174.1085
      Epoch 1 composite train-obj: 5.367731
            Val objective improved inf → 3.3762, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.0979, mae: 3.5204, huber: 3.0879, swd: 18.3539, ept: 182.2493
    Epoch [2/50], Val Losses: mse: 30.1152, mae: 3.7745, huber: 3.3313, swd: 8.5415, ept: 143.8715
    Epoch [2/50], Test Losses: mse: 21.5496, mae: 3.0857, huber: 2.6442, swd: 9.1228, ept: 173.4751
      Epoch 2 composite train-obj: 3.087934
            Val objective improved 3.3762 → 3.3313, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.3061, mae: 3.4881, huber: 3.0561, swd: 18.1322, ept: 184.2394
    Epoch [3/50], Val Losses: mse: 33.8798, mae: 3.9674, huber: 3.5223, swd: 11.1551, ept: 134.6189
    Epoch [3/50], Test Losses: mse: 23.4925, mae: 3.2366, huber: 2.7929, swd: 10.4897, ept: 161.9219
      Epoch 3 composite train-obj: 3.056112
            No improvement (3.5223), counter 1/5
    Epoch [4/50], Train Losses: mse: 37.5659, mae: 3.4363, huber: 3.0044, swd: 17.9539, ept: 185.7961
    Epoch [4/50], Val Losses: mse: 29.1172, mae: 3.7488, huber: 3.3036, swd: 8.3752, ept: 145.8162
    Epoch [4/50], Test Losses: mse: 20.0570, mae: 2.9296, huber: 2.4928, swd: 8.5067, ept: 174.1605
      Epoch 4 composite train-obj: 3.004424
            Val objective improved 3.3313 → 3.3036, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 35.6785, mae: 3.2862, huber: 2.8559, swd: 16.7041, ept: 194.2276
    Epoch [5/50], Val Losses: mse: 28.5528, mae: 3.7002, huber: 3.2547, swd: 8.5801, ept: 149.2916
    Epoch [5/50], Test Losses: mse: 19.5573, mae: 2.9374, huber: 2.4974, swd: 8.5790, ept: 179.1437
      Epoch 5 composite train-obj: 2.855859
            Val objective improved 3.3036 → 3.2547, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 35.1456, mae: 3.2528, huber: 2.8226, swd: 16.5353, ept: 198.2073
    Epoch [6/50], Val Losses: mse: 29.4063, mae: 3.7234, huber: 3.2787, swd: 8.8972, ept: 149.0684
    Epoch [6/50], Test Losses: mse: 19.7224, mae: 2.9043, huber: 2.4674, swd: 8.5930, ept: 176.8176
      Epoch 6 composite train-obj: 2.822584
            No improvement (3.2787), counter 1/5
    Epoch [7/50], Train Losses: mse: 34.4630, mae: 3.2014, huber: 2.7719, swd: 16.0689, ept: 200.8642
    Epoch [7/50], Val Losses: mse: 28.2933, mae: 3.7160, huber: 3.2708, swd: 8.6851, ept: 153.0144
    Epoch [7/50], Test Losses: mse: 18.6200, mae: 2.8423, huber: 2.4062, swd: 8.0010, ept: 183.2437
      Epoch 7 composite train-obj: 2.771872
            No improvement (3.2708), counter 2/5
    Epoch [8/50], Train Losses: mse: 34.2168, mae: 3.2000, huber: 2.7704, swd: 15.9095, ept: 201.1461
    Epoch [8/50], Val Losses: mse: 27.7485, mae: 3.6138, huber: 3.1693, swd: 8.2003, ept: 154.7268
    Epoch [8/50], Test Losses: mse: 18.8470, mae: 2.8715, huber: 2.4326, swd: 8.2125, ept: 182.7637
      Epoch 8 composite train-obj: 2.770367
            Val objective improved 3.2547 → 3.1693, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 34.0334, mae: 3.1723, huber: 2.7432, swd: 15.8315, ept: 202.3954
    Epoch [9/50], Val Losses: mse: 28.8488, mae: 3.6695, huber: 3.2242, swd: 8.8894, ept: 152.6882
    Epoch [9/50], Test Losses: mse: 19.2523, mae: 2.9021, huber: 2.4622, swd: 8.3867, ept: 176.4590
      Epoch 9 composite train-obj: 2.743160
            No improvement (3.2242), counter 1/5
    Epoch [10/50], Train Losses: mse: 33.9201, mae: 3.1722, huber: 2.7429, swd: 15.8260, ept: 202.1168
    Epoch [10/50], Val Losses: mse: 27.5642, mae: 3.6401, huber: 3.1941, swd: 8.5735, ept: 156.1776
    Epoch [10/50], Test Losses: mse: 18.9494, mae: 2.9282, huber: 2.4860, swd: 8.6557, ept: 183.1909
      Epoch 10 composite train-obj: 2.742934
            No improvement (3.1941), counter 2/5
    Epoch [11/50], Train Losses: mse: 33.7726, mae: 3.1660, huber: 2.7367, swd: 15.8083, ept: 202.0836
    Epoch [11/50], Val Losses: mse: 26.9904, mae: 3.5619, huber: 3.1187, swd: 8.0589, ept: 156.2025
    Epoch [11/50], Test Losses: mse: 18.9565, mae: 2.9252, huber: 2.4828, swd: 8.5131, ept: 179.3080
      Epoch 11 composite train-obj: 2.736733
            Val objective improved 3.1693 → 3.1187, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 33.1608, mae: 3.1051, huber: 2.6771, swd: 15.2649, ept: 204.4376
    Epoch [12/50], Val Losses: mse: 27.8593, mae: 3.5831, huber: 3.1422, swd: 8.2615, ept: 154.1803
    Epoch [12/50], Test Losses: mse: 19.3079, mae: 2.8988, huber: 2.4591, swd: 8.4098, ept: 175.0442
      Epoch 12 composite train-obj: 2.677087
            No improvement (3.1422), counter 1/5
    Epoch [13/50], Train Losses: mse: 33.0443, mae: 3.1042, huber: 2.6761, swd: 15.1709, ept: 203.9407
    Epoch [13/50], Val Losses: mse: 28.1180, mae: 3.6173, huber: 3.1764, swd: 8.3990, ept: 154.7075
    Epoch [13/50], Test Losses: mse: 18.8728, mae: 2.8502, huber: 2.4122, swd: 8.1708, ept: 178.4978
      Epoch 13 composite train-obj: 2.676113
            No improvement (3.1764), counter 2/5
    Epoch [14/50], Train Losses: mse: 32.9078, mae: 3.0861, huber: 2.6587, swd: 15.0559, ept: 203.9999
    Epoch [14/50], Val Losses: mse: 28.3980, mae: 3.6285, huber: 3.1852, swd: 8.6510, ept: 153.2682
    Epoch [14/50], Test Losses: mse: 19.5429, mae: 2.9460, huber: 2.5025, swd: 8.6933, ept: 177.7996
      Epoch 14 composite train-obj: 2.658686
            No improvement (3.1852), counter 3/5
    Epoch [15/50], Train Losses: mse: 32.5923, mae: 3.0716, huber: 2.6442, swd: 14.8506, ept: 205.0263
    Epoch [15/50], Val Losses: mse: 28.8124, mae: 3.6305, huber: 3.1892, swd: 8.7550, ept: 152.5882
    Epoch [15/50], Test Losses: mse: 20.3222, mae: 2.9908, huber: 2.5487, swd: 9.3340, ept: 173.8300
      Epoch 15 composite train-obj: 2.644177
            No improvement (3.1892), counter 4/5
    Epoch [16/50], Train Losses: mse: 32.6881, mae: 3.0708, huber: 2.6437, swd: 14.9841, ept: 204.9550
    Epoch [16/50], Val Losses: mse: 27.6141, mae: 3.5724, huber: 3.1309, swd: 8.3664, ept: 156.5835
    Epoch [16/50], Test Losses: mse: 18.9239, mae: 2.8659, huber: 2.4263, swd: 8.2177, ept: 180.3204
      Epoch 16 composite train-obj: 2.643719
    Epoch [16/50], Test Losses: mse: 18.9565, mae: 2.9252, huber: 2.4828, swd: 8.5131, ept: 179.2869
    Best round's Test MSE: 18.9565, MAE: 2.9252, SWD: 8.5131
    Best round's Validation MSE: 26.9904, MAE: 3.5619, SWD: 8.0589
    Best round's Test verification MSE : 18.9565, MAE: 2.9252, SWD: 8.5131
    Time taken: 45.47 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 116.9037, mae: 5.7703, huber: 5.3278, swd: 93.4724, ept: 140.4390
    Epoch [1/50], Val Losses: mse: 30.9605, mae: 3.7995, huber: 3.3568, swd: 10.0943, ept: 142.0466
    Epoch [1/50], Test Losses: mse: 21.2504, mae: 3.0472, huber: 2.6075, swd: 9.1603, ept: 177.0750
      Epoch 1 composite train-obj: 5.327817
            Val objective improved inf → 3.3568, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.4509, mae: 3.5372, huber: 3.1042, swd: 19.8127, ept: 184.0710
    Epoch [2/50], Val Losses: mse: 31.0496, mae: 3.8055, huber: 3.3608, swd: 9.1572, ept: 139.4659
    Epoch [2/50], Test Losses: mse: 22.6148, mae: 3.1974, huber: 2.7528, swd: 10.1285, ept: 167.8571
      Epoch 2 composite train-obj: 3.104217
            No improvement (3.3608), counter 1/5
    Epoch [3/50], Train Losses: mse: 38.2646, mae: 3.4741, huber: 3.0414, swd: 18.9748, ept: 186.4523
    Epoch [3/50], Val Losses: mse: 29.8000, mae: 3.7673, huber: 3.3233, swd: 8.8995, ept: 144.4887
    Epoch [3/50], Test Losses: mse: 20.5664, mae: 2.9615, huber: 2.5249, swd: 8.9421, ept: 175.0989
      Epoch 3 composite train-obj: 3.041402
            Val objective improved 3.3568 → 3.3233, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.2630, mae: 3.4247, huber: 2.9924, swd: 18.5684, ept: 188.8282
    Epoch [4/50], Val Losses: mse: 29.6743, mae: 3.7513, huber: 3.3068, swd: 9.0548, ept: 144.5356
    Epoch [4/50], Test Losses: mse: 20.1119, mae: 2.9629, huber: 2.5234, swd: 8.8332, ept: 174.9799
      Epoch 4 composite train-obj: 2.992369
            Val objective improved 3.3233 → 3.3068, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 36.9102, mae: 3.3800, huber: 2.9481, swd: 18.6117, ept: 192.1750
    Epoch [5/50], Val Losses: mse: 29.0622, mae: 3.7373, huber: 3.2925, swd: 9.2805, ept: 148.6563
    Epoch [5/50], Test Losses: mse: 19.4751, mae: 2.9281, huber: 2.4885, swd: 8.8023, ept: 179.4827
      Epoch 5 composite train-obj: 2.948148
            Val objective improved 3.3068 → 3.2925, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 35.3306, mae: 3.2681, huber: 2.8376, swd: 17.4517, ept: 199.3115
    Epoch [6/50], Val Losses: mse: 28.3107, mae: 3.6704, huber: 3.2254, swd: 8.7335, ept: 150.0620
    Epoch [6/50], Test Losses: mse: 19.5673, mae: 2.9156, huber: 2.4765, swd: 9.0086, ept: 180.1661
      Epoch 6 composite train-obj: 2.837603
            Val objective improved 3.2925 → 3.2254, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 34.7351, mae: 3.2316, huber: 2.8017, swd: 16.9955, ept: 201.0237
    Epoch [7/50], Val Losses: mse: 29.0172, mae: 3.7003, huber: 3.2566, swd: 8.9355, ept: 148.6236
    Epoch [7/50], Test Losses: mse: 20.1488, mae: 2.9270, huber: 2.4898, swd: 9.2289, ept: 173.1230
      Epoch 7 composite train-obj: 2.801660
            No improvement (3.2566), counter 1/5
    Epoch [8/50], Train Losses: mse: 34.3290, mae: 3.2015, huber: 2.7722, swd: 16.6211, ept: 201.5586
    Epoch [8/50], Val Losses: mse: 27.9930, mae: 3.6812, huber: 3.2375, swd: 9.1548, ept: 152.5924
    Epoch [8/50], Test Losses: mse: 19.0335, mae: 2.8838, huber: 2.4455, swd: 8.9370, ept: 183.8787
      Epoch 8 composite train-obj: 2.772226
            No improvement (3.2375), counter 2/5
    Epoch [9/50], Train Losses: mse: 34.4296, mae: 3.2125, huber: 2.7833, swd: 16.8088, ept: 200.7243
    Epoch [9/50], Val Losses: mse: 29.2534, mae: 3.6543, huber: 3.2132, swd: 9.1516, ept: 149.4328
    Epoch [9/50], Test Losses: mse: 20.8009, mae: 2.9766, huber: 2.5380, swd: 9.9926, ept: 173.6990
      Epoch 9 composite train-obj: 2.783294
            Val objective improved 3.2254 → 3.2132, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 33.9025, mae: 3.1668, huber: 2.7381, swd: 16.4710, ept: 202.7902
    Epoch [10/50], Val Losses: mse: 27.1228, mae: 3.5707, huber: 3.1280, swd: 8.1980, ept: 154.9564
    Epoch [10/50], Test Losses: mse: 19.2162, mae: 2.9296, huber: 2.4873, swd: 8.9255, ept: 181.3777
      Epoch 10 composite train-obj: 2.738126
            Val objective improved 3.2132 → 3.1280, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 33.7239, mae: 3.1601, huber: 2.7316, swd: 16.4467, ept: 202.3837
    Epoch [11/50], Val Losses: mse: 27.9763, mae: 3.5900, huber: 3.1485, swd: 8.4970, ept: 154.1903
    Epoch [11/50], Test Losses: mse: 19.8601, mae: 2.9690, huber: 2.5260, swd: 9.2951, ept: 177.4096
      Epoch 11 composite train-obj: 2.731647
            No improvement (3.1485), counter 1/5
    Epoch [12/50], Train Losses: mse: 33.4399, mae: 3.1383, huber: 2.7103, swd: 16.2046, ept: 203.5388
    Epoch [12/50], Val Losses: mse: 27.6954, mae: 3.6008, huber: 3.1602, swd: 8.5437, ept: 155.2462
    Epoch [12/50], Test Losses: mse: 19.3871, mae: 2.9476, huber: 2.5059, swd: 9.0929, ept: 180.8420
      Epoch 12 composite train-obj: 2.710263
            No improvement (3.1602), counter 2/5
    Epoch [13/50], Train Losses: mse: 33.1061, mae: 3.1074, huber: 2.6799, swd: 15.9824, ept: 204.5821
    Epoch [13/50], Val Losses: mse: 26.9772, mae: 3.5451, huber: 3.1014, swd: 8.2986, ept: 156.7211
    Epoch [13/50], Test Losses: mse: 18.9013, mae: 2.8629, huber: 2.4240, swd: 8.8519, ept: 183.9173
      Epoch 13 composite train-obj: 2.679901
            Val objective improved 3.1280 → 3.1014, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 32.9871, mae: 3.1089, huber: 2.6814, swd: 15.9079, ept: 204.2369
    Epoch [14/50], Val Losses: mse: 28.1939, mae: 3.5904, huber: 3.1479, swd: 8.7009, ept: 153.8555
    Epoch [14/50], Test Losses: mse: 20.3681, mae: 3.0262, huber: 2.5815, swd: 9.6677, ept: 175.3330
      Epoch 14 composite train-obj: 2.681357
            No improvement (3.1479), counter 1/5
    Epoch [15/50], Train Losses: mse: 32.7320, mae: 3.0852, huber: 2.6578, swd: 15.6735, ept: 204.9083
    Epoch [15/50], Val Losses: mse: 26.8772, mae: 3.5197, huber: 3.0796, swd: 8.0574, ept: 156.9063
    Epoch [15/50], Test Losses: mse: 19.3331, mae: 2.9370, huber: 2.4952, swd: 9.1207, ept: 182.3700
      Epoch 15 composite train-obj: 2.657849
            Val objective improved 3.1014 → 3.0796, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 32.5695, mae: 3.0678, huber: 2.6409, swd: 15.5637, ept: 205.6798
    Epoch [16/50], Val Losses: mse: 28.5812, mae: 3.6426, huber: 3.1997, swd: 9.3648, ept: 154.4487
    Epoch [16/50], Test Losses: mse: 19.1246, mae: 2.9138, huber: 2.4716, swd: 8.6863, ept: 180.2647
      Epoch 16 composite train-obj: 2.640861
            No improvement (3.1997), counter 1/5
    Epoch [17/50], Train Losses: mse: 32.4382, mae: 3.0517, huber: 2.6253, swd: 15.4988, ept: 206.1341
    Epoch [17/50], Val Losses: mse: 27.5681, mae: 3.5712, huber: 3.1321, swd: 8.5162, ept: 155.7311
    Epoch [17/50], Test Losses: mse: 19.1329, mae: 2.8955, huber: 2.4558, swd: 8.8278, ept: 183.2264
      Epoch 17 composite train-obj: 2.625343
            No improvement (3.1321), counter 2/5
    Epoch [18/50], Train Losses: mse: 32.3814, mae: 3.0591, huber: 2.6327, swd: 15.5079, ept: 205.8828
    Epoch [18/50], Val Losses: mse: 26.9181, mae: 3.5038, huber: 3.0646, swd: 8.3185, ept: 157.2091
    Epoch [18/50], Test Losses: mse: 19.3760, mae: 2.9232, huber: 2.4828, swd: 9.0692, ept: 183.1794
      Epoch 18 composite train-obj: 2.632685
            Val objective improved 3.0796 → 3.0646, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 32.3102, mae: 3.0405, huber: 2.6147, swd: 15.3769, ept: 206.5630
    Epoch [19/50], Val Losses: mse: 27.7956, mae: 3.5938, huber: 3.1507, swd: 8.6546, ept: 155.4905
    Epoch [19/50], Test Losses: mse: 19.2328, mae: 2.8890, huber: 2.4493, swd: 8.9001, ept: 180.3516
      Epoch 19 composite train-obj: 2.614651
            No improvement (3.1507), counter 1/5
    Epoch [20/50], Train Losses: mse: 32.0085, mae: 3.0158, huber: 2.5902, swd: 15.1581, ept: 207.5189
    Epoch [20/50], Val Losses: mse: 27.1183, mae: 3.5308, huber: 3.0925, swd: 8.6616, ept: 158.2841
    Epoch [20/50], Test Losses: mse: 19.3338, mae: 2.9288, huber: 2.4885, swd: 9.1258, ept: 182.0850
      Epoch 20 composite train-obj: 2.590195
            No improvement (3.0925), counter 2/5
    Epoch [21/50], Train Losses: mse: 32.1698, mae: 3.0384, huber: 2.6126, swd: 15.3375, ept: 206.5379
    Epoch [21/50], Val Losses: mse: 27.5830, mae: 3.6116, huber: 3.1722, swd: 9.0916, ept: 157.7894
    Epoch [21/50], Test Losses: mse: 18.5074, mae: 2.8584, huber: 2.4202, swd: 8.4801, ept: 184.9621
      Epoch 21 composite train-obj: 2.612577
            No improvement (3.1722), counter 3/5
    Epoch [22/50], Train Losses: mse: 31.7865, mae: 3.0018, huber: 2.5766, swd: 15.0264, ept: 208.1163
    Epoch [22/50], Val Losses: mse: 26.7215, mae: 3.4941, huber: 3.0588, swd: 8.4268, ept: 157.3408
    Epoch [22/50], Test Losses: mse: 19.2470, mae: 2.8923, huber: 2.4540, swd: 9.1812, ept: 182.8351
      Epoch 22 composite train-obj: 2.576617
            Val objective improved 3.0646 → 3.0588, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 31.7899, mae: 3.0063, huber: 2.5810, swd: 15.0206, ept: 207.3919
    Epoch [23/50], Val Losses: mse: 27.4135, mae: 3.5746, huber: 3.1368, swd: 8.9068, ept: 157.1251
    Epoch [23/50], Test Losses: mse: 18.6070, mae: 2.8361, huber: 2.4002, swd: 8.5411, ept: 183.6744
      Epoch 23 composite train-obj: 2.581010
            No improvement (3.1368), counter 1/5
    Epoch [24/50], Train Losses: mse: 31.4554, mae: 2.9789, huber: 2.5543, swd: 14.7223, ept: 208.5852
    Epoch [24/50], Val Losses: mse: 27.0592, mae: 3.5017, huber: 3.0624, swd: 8.6808, ept: 157.5258
    Epoch [24/50], Test Losses: mse: 18.9731, mae: 2.8386, huber: 2.4019, swd: 8.8092, ept: 183.4107
      Epoch 24 composite train-obj: 2.554270
            No improvement (3.0624), counter 2/5
    Epoch [25/50], Train Losses: mse: 31.4282, mae: 2.9747, huber: 2.5502, swd: 14.7421, ept: 208.8667
    Epoch [25/50], Val Losses: mse: 26.9272, mae: 3.4982, huber: 3.0628, swd: 8.4983, ept: 157.2781
    Epoch [25/50], Test Losses: mse: 19.3071, mae: 2.8989, huber: 2.4605, swd: 9.0709, ept: 181.6701
      Epoch 25 composite train-obj: 2.550248
            No improvement (3.0628), counter 3/5
    Epoch [26/50], Train Losses: mse: 31.3227, mae: 2.9639, huber: 2.5398, swd: 14.6504, ept: 209.2213
    Epoch [26/50], Val Losses: mse: 26.9872, mae: 3.5056, huber: 3.0674, swd: 8.5568, ept: 158.4442
    Epoch [26/50], Test Losses: mse: 19.1848, mae: 2.8403, huber: 2.4063, swd: 8.9754, ept: 182.5280
      Epoch 26 composite train-obj: 2.539797
            No improvement (3.0674), counter 4/5
    Epoch [27/50], Train Losses: mse: 31.2489, mae: 2.9627, huber: 2.5386, swd: 14.5980, ept: 209.0966
    Epoch [27/50], Val Losses: mse: 27.7314, mae: 3.5384, huber: 3.0996, swd: 9.0473, ept: 156.9265
    Epoch [27/50], Test Losses: mse: 19.3605, mae: 2.8628, huber: 2.4267, swd: 8.9035, ept: 179.8952
      Epoch 27 composite train-obj: 2.538630
    Epoch [27/50], Test Losses: mse: 19.2470, mae: 2.8923, huber: 2.4540, swd: 9.1813, ept: 182.8169
    Best round's Test MSE: 19.2470, MAE: 2.8923, SWD: 9.1812
    Best round's Validation MSE: 26.7215, MAE: 3.4941, SWD: 8.4268
    Best round's Test verification MSE : 19.2470, MAE: 2.8923, SWD: 9.1813
    Time taken: 77.93 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 115.6907, mae: 5.7924, huber: 5.3506, swd: 81.3658, ept: 134.5163
    Epoch [1/50], Val Losses: mse: 31.1558, mae: 3.8116, huber: 3.3701, swd: 9.7092, ept: 139.2261
    Epoch [1/50], Test Losses: mse: 21.3326, mae: 3.0622, huber: 2.6221, swd: 8.4542, ept: 174.4745
      Epoch 1 composite train-obj: 5.350612
            Val objective improved inf → 3.3701, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 40.0584, mae: 3.5870, huber: 3.1537, swd: 18.9245, ept: 180.1823
    Epoch [2/50], Val Losses: mse: 30.6458, mae: 3.8018, huber: 3.3593, swd: 8.5717, ept: 141.2342
    Epoch [2/50], Test Losses: mse: 21.1448, mae: 3.0319, huber: 2.5928, swd: 8.1496, ept: 172.1425
      Epoch 2 composite train-obj: 3.153712
            Val objective improved 3.3701 → 3.3593, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.1056, mae: 3.4620, huber: 3.0298, swd: 17.4910, ept: 185.7015
    Epoch [3/50], Val Losses: mse: 29.6614, mae: 3.7869, huber: 3.3435, swd: 8.7997, ept: 143.9354
    Epoch [3/50], Test Losses: mse: 20.2036, mae: 2.9581, huber: 2.5206, swd: 8.1795, ept: 177.3781
      Epoch 3 composite train-obj: 3.029825
            Val objective improved 3.3593 → 3.3435, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.1563, mae: 3.3995, huber: 2.9675, swd: 17.3330, ept: 190.1612
    Epoch [4/50], Val Losses: mse: 29.1903, mae: 3.7755, huber: 3.3321, swd: 9.3667, ept: 148.5892
    Epoch [4/50], Test Losses: mse: 19.8869, mae: 2.9435, huber: 2.5064, swd: 8.7542, ept: 182.3182
      Epoch 4 composite train-obj: 2.967483
            Val objective improved 3.3435 → 3.3321, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 35.6903, mae: 3.2872, huber: 2.8564, swd: 16.4404, ept: 198.0162
    Epoch [5/50], Val Losses: mse: 28.1675, mae: 3.6829, huber: 3.2394, swd: 8.2581, ept: 148.9985
    Epoch [5/50], Test Losses: mse: 19.6019, mae: 2.9447, huber: 2.5053, swd: 8.3169, ept: 179.6491
      Epoch 5 composite train-obj: 2.856439
            Val objective improved 3.3321 → 3.2394, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 35.3257, mae: 3.2593, huber: 2.8290, swd: 16.2718, ept: 199.5239
    Epoch [6/50], Val Losses: mse: 29.2553, mae: 3.6867, huber: 3.2423, swd: 8.4565, ept: 148.3542
    Epoch [6/50], Test Losses: mse: 20.3452, mae: 2.9973, huber: 2.5551, swd: 8.5758, ept: 174.4306
      Epoch 6 composite train-obj: 2.828988
            No improvement (3.2423), counter 1/5
    Epoch [7/50], Train Losses: mse: 34.7726, mae: 3.2316, huber: 2.8018, swd: 15.9004, ept: 200.7111
    Epoch [7/50], Val Losses: mse: 29.9888, mae: 3.7260, huber: 3.2820, swd: 9.3151, ept: 149.1723
    Epoch [7/50], Test Losses: mse: 20.3162, mae: 2.9756, huber: 2.5350, swd: 8.6741, ept: 173.4193
      Epoch 7 composite train-obj: 2.801839
            No improvement (3.2820), counter 2/5
    Epoch [8/50], Train Losses: mse: 34.4998, mae: 3.2135, huber: 2.7840, swd: 15.7601, ept: 201.2915
    Epoch [8/50], Val Losses: mse: 28.5190, mae: 3.6223, huber: 3.1796, swd: 8.0675, ept: 151.9490
    Epoch [8/50], Test Losses: mse: 20.0270, mae: 2.9490, huber: 2.5086, swd: 8.4767, ept: 177.4097
      Epoch 8 composite train-obj: 2.783964
            Val objective improved 3.2394 → 3.1796, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 34.1249, mae: 3.1847, huber: 2.7560, swd: 15.4766, ept: 202.2900
    Epoch [9/50], Val Losses: mse: 30.5043, mae: 3.7072, huber: 3.2653, swd: 9.6990, ept: 149.2712
    Epoch [9/50], Test Losses: mse: 21.7526, mae: 3.1157, huber: 2.6699, swd: 9.8000, ept: 170.1370
      Epoch 9 composite train-obj: 2.756041
            No improvement (3.2653), counter 1/5
    Epoch [10/50], Train Losses: mse: 33.5766, mae: 3.1417, huber: 2.7140, swd: 15.0649, ept: 204.4295
    Epoch [10/50], Val Losses: mse: 28.1975, mae: 3.5736, huber: 3.1322, swd: 8.1908, ept: 152.3705
    Epoch [10/50], Test Losses: mse: 19.9718, mae: 2.9791, huber: 2.5352, swd: 8.5735, ept: 179.0600
      Epoch 10 composite train-obj: 2.714045
            Val objective improved 3.1796 → 3.1322, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 33.4308, mae: 3.1316, huber: 2.7042, swd: 15.0261, ept: 204.5640
    Epoch [11/50], Val Losses: mse: 29.9670, mae: 3.6512, huber: 3.2094, swd: 9.3855, ept: 151.9875
    Epoch [11/50], Test Losses: mse: 21.9639, mae: 3.1204, huber: 2.6736, swd: 10.1149, ept: 171.3303
      Epoch 11 composite train-obj: 2.704213
            No improvement (3.2094), counter 1/5
    Epoch [12/50], Train Losses: mse: 33.1837, mae: 3.1148, huber: 2.6879, swd: 14.8854, ept: 204.8752
    Epoch [12/50], Val Losses: mse: 29.5394, mae: 3.6560, huber: 3.2138, swd: 9.2156, ept: 152.6835
    Epoch [12/50], Test Losses: mse: 20.5267, mae: 2.9886, huber: 2.5469, swd: 8.8775, ept: 174.9905
      Epoch 12 composite train-obj: 2.687899
            No improvement (3.2138), counter 2/5
    Epoch [13/50], Train Losses: mse: 33.2891, mae: 3.1228, huber: 2.6958, swd: 15.0312, ept: 204.2888
    Epoch [13/50], Val Losses: mse: 28.0391, mae: 3.5565, huber: 3.1186, swd: 8.2486, ept: 154.4571
    Epoch [13/50], Test Losses: mse: 19.7275, mae: 2.9012, huber: 2.4641, swd: 8.2515, ept: 179.8498
      Epoch 13 composite train-obj: 2.695808
            Val objective improved 3.1322 → 3.1186, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 33.1436, mae: 3.1127, huber: 2.6857, swd: 15.0014, ept: 204.7255
    Epoch [14/50], Val Losses: mse: 28.8388, mae: 3.6307, huber: 3.1908, swd: 9.1292, ept: 153.5228
    Epoch [14/50], Test Losses: mse: 19.5378, mae: 2.9222, huber: 2.4826, swd: 8.2999, ept: 182.6608
      Epoch 14 composite train-obj: 2.685719
            No improvement (3.1908), counter 1/5
    Epoch [15/50], Train Losses: mse: 33.0579, mae: 3.1044, huber: 2.6775, swd: 14.9337, ept: 204.8698
    Epoch [15/50], Val Losses: mse: 28.5015, mae: 3.6103, huber: 3.1674, swd: 8.2534, ept: 153.1145
    Epoch [15/50], Test Losses: mse: 19.8466, mae: 2.9811, huber: 2.5358, swd: 8.3280, ept: 174.6512
      Epoch 15 composite train-obj: 2.677523
            No improvement (3.1674), counter 2/5
    Epoch [16/50], Train Losses: mse: 32.5720, mae: 3.0724, huber: 2.6460, swd: 14.4736, ept: 205.9285
    Epoch [16/50], Val Losses: mse: 26.9540, mae: 3.5061, huber: 3.0671, swd: 7.8845, ept: 156.1550
    Epoch [16/50], Test Losses: mse: 18.9100, mae: 2.8596, huber: 2.4215, swd: 7.9397, ept: 183.8922
      Epoch 16 composite train-obj: 2.646036
            Val objective improved 3.1186 → 3.0671, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 33.1602, mae: 3.1141, huber: 2.6872, swd: 15.0016, ept: 203.9307
    Epoch [17/50], Val Losses: mse: 27.5019, mae: 3.5664, huber: 3.1265, swd: 8.1766, ept: 156.0836
    Epoch [17/50], Test Losses: mse: 18.8968, mae: 2.8880, huber: 2.4480, swd: 7.7054, ept: 180.4933
      Epoch 17 composite train-obj: 2.687156
            No improvement (3.1265), counter 1/5
    Epoch [18/50], Train Losses: mse: 32.3498, mae: 3.0540, huber: 2.6279, swd: 14.4130, ept: 206.4913
    Epoch [18/50], Val Losses: mse: 27.9148, mae: 3.5542, huber: 3.1157, swd: 8.2812, ept: 156.4845
    Epoch [18/50], Test Losses: mse: 19.3496, mae: 2.8503, huber: 2.4146, swd: 8.0522, ept: 181.9258
      Epoch 18 composite train-obj: 2.627914
            No improvement (3.1157), counter 2/5
    Epoch [19/50], Train Losses: mse: 32.0829, mae: 3.0343, huber: 2.6085, swd: 14.2209, ept: 207.3286
    Epoch [19/50], Val Losses: mse: 29.4364, mae: 3.6184, huber: 3.1798, swd: 9.2010, ept: 153.9599
    Epoch [19/50], Test Losses: mse: 21.5458, mae: 3.0364, huber: 2.5986, swd: 9.6747, ept: 172.9984
      Epoch 19 composite train-obj: 2.608474
            No improvement (3.1798), counter 3/5
    Epoch [20/50], Train Losses: mse: 32.0813, mae: 3.0289, huber: 2.6033, swd: 14.2500, ept: 207.6636
    Epoch [20/50], Val Losses: mse: 28.4406, mae: 3.5877, huber: 3.1521, swd: 8.6683, ept: 155.2088
    Epoch [20/50], Test Losses: mse: 19.3829, mae: 2.8676, huber: 2.4333, swd: 8.0023, ept: 180.6191
      Epoch 20 composite train-obj: 2.603300
            No improvement (3.1521), counter 4/5
    Epoch [21/50], Train Losses: mse: 32.0352, mae: 3.0321, huber: 2.6066, swd: 14.2282, ept: 207.5160
    Epoch [21/50], Val Losses: mse: 28.8672, mae: 3.5753, huber: 3.1392, swd: 8.9615, ept: 155.0010
    Epoch [21/50], Test Losses: mse: 21.1806, mae: 2.9999, huber: 2.5629, swd: 9.3993, ept: 173.0850
      Epoch 21 composite train-obj: 2.606585
    Epoch [21/50], Test Losses: mse: 18.9102, mae: 2.8597, huber: 2.4215, swd: 7.9398, ept: 183.8806
    Best round's Test MSE: 18.9100, MAE: 2.8596, SWD: 7.9397
    Best round's Validation MSE: 26.9540, MAE: 3.5061, SWD: 7.8845
    Best round's Test verification MSE : 18.9102, MAE: 2.8597, SWD: 7.9398
    Time taken: 60.45 seconds
    
    ==================================================
    Experiment Summary (ACL_etth2_seq96_pred336_20250511_1613)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 19.0378 ± 0.1491
      mae: 2.8924 ± 0.0267
      huber: 2.4528 ± 0.0251
      swd: 8.5447 ± 0.5074
      ept: 182.0118 ± 1.9600
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 26.8886 ± 0.1191
      mae: 3.5207 ± 0.0295
      huber: 3.0815 ± 0.0265
      swd: 8.1234 ± 0.2261
      ept: 156.5661 ± 0.5481
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 183.95 seconds
    
    Experiment complete: ACL_etth2_seq96_pred336_20250511_1613
    Model: ACL
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### Timemixer

#### pred=96


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.3515, mae: 0.3481, huber: 0.1326, swd: 0.1209, target_std: 0.7879
    Epoch [1/50], Val Losses: mse: 0.2707, mae: 0.3524, huber: 0.1279, swd: 0.1135, target_std: 0.9798
    Epoch [1/50], Test Losses: mse: 0.1731, mae: 0.2850, huber: 0.0830, swd: 0.0634, target_std: 0.7349
      Epoch 1 composite train-obj: 0.132563
            Val objective improved inf → 0.1279, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2976, mae: 0.3138, huber: 0.1130, swd: 0.1112, target_std: 0.7878
    Epoch [2/50], Val Losses: mse: 0.2696, mae: 0.3506, huber: 0.1273, swd: 0.1132, target_std: 0.9798
    Epoch [2/50], Test Losses: mse: 0.1704, mae: 0.2819, huber: 0.0817, swd: 0.0623, target_std: 0.7349
      Epoch 2 composite train-obj: 0.112951
            Val objective improved 0.1279 → 0.1273, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2873, mae: 0.3084, huber: 0.1097, swd: 0.1092, target_std: 0.7879
    Epoch [3/50], Val Losses: mse: 0.2689, mae: 0.3507, huber: 0.1271, swd: 0.1126, target_std: 0.9798
    Epoch [3/50], Test Losses: mse: 0.1697, mae: 0.2827, huber: 0.0814, swd: 0.0626, target_std: 0.7349
      Epoch 3 composite train-obj: 0.109698
            Val objective improved 0.1273 → 0.1271, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2804, mae: 0.3056, huber: 0.1077, swd: 0.1068, target_std: 0.7878
    Epoch [4/50], Val Losses: mse: 0.2732, mae: 0.3542, huber: 0.1291, swd: 0.1120, target_std: 0.9798
    Epoch [4/50], Test Losses: mse: 0.1712, mae: 0.2848, huber: 0.0822, swd: 0.0617, target_std: 0.7349
      Epoch 4 composite train-obj: 0.107663
            No improvement (0.1291), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.2743, mae: 0.3032, huber: 0.1060, swd: 0.1045, target_std: 0.7879
    Epoch [5/50], Val Losses: mse: 0.2645, mae: 0.3475, huber: 0.1251, swd: 0.1115, target_std: 0.9798
    Epoch [5/50], Test Losses: mse: 0.1708, mae: 0.2825, huber: 0.0818, swd: 0.0647, target_std: 0.7349
      Epoch 5 composite train-obj: 0.106022
            Val objective improved 0.1271 → 0.1251, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.2673, mae: 0.3008, huber: 0.1041, swd: 0.1018, target_std: 0.7878
    Epoch [6/50], Val Losses: mse: 0.2675, mae: 0.3491, huber: 0.1263, swd: 0.1176, target_std: 0.9798
    Epoch [6/50], Test Losses: mse: 0.1717, mae: 0.2832, huber: 0.0821, swd: 0.0671, target_std: 0.7349
      Epoch 6 composite train-obj: 0.104143
            No improvement (0.1263), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.2606, mae: 0.2984, huber: 0.1023, swd: 0.0996, target_std: 0.7878
    Epoch [7/50], Val Losses: mse: 0.2764, mae: 0.3537, huber: 0.1300, swd: 0.1176, target_std: 0.9798
    Epoch [7/50], Test Losses: mse: 0.1718, mae: 0.2848, huber: 0.0824, swd: 0.0643, target_std: 0.7349
      Epoch 7 composite train-obj: 0.102283
            No improvement (0.1300), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.2544, mae: 0.2960, huber: 0.1005, swd: 0.0974, target_std: 0.7879
    Epoch [8/50], Val Losses: mse: 0.2665, mae: 0.3489, huber: 0.1259, swd: 0.1152, target_std: 0.9798
    Epoch [8/50], Test Losses: mse: 0.1732, mae: 0.2851, huber: 0.0829, swd: 0.0653, target_std: 0.7349
      Epoch 8 composite train-obj: 0.100471
            No improvement (0.1259), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.2497, mae: 0.2937, huber: 0.0990, swd: 0.0957, target_std: 0.7879
    Epoch [9/50], Val Losses: mse: 0.2784, mae: 0.3563, huber: 0.1308, swd: 0.1179, target_std: 0.9798
    Epoch [9/50], Test Losses: mse: 0.1746, mae: 0.2867, huber: 0.0836, swd: 0.0643, target_std: 0.7349
      Epoch 9 composite train-obj: 0.098967
            No improvement (0.1308), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.2434, mae: 0.2917, huber: 0.0972, swd: 0.0937, target_std: 0.7879
    Epoch [10/50], Val Losses: mse: 0.2670, mae: 0.3476, huber: 0.1257, swd: 0.1169, target_std: 0.9798
    Epoch [10/50], Test Losses: mse: 0.1713, mae: 0.2825, huber: 0.0820, swd: 0.0644, target_std: 0.7349
      Epoch 10 composite train-obj: 0.097245
    Epoch [10/50], Test Losses: mse: 0.1708, mae: 0.2825, huber: 0.0818, swd: 0.0647, target_std: 0.7349
    Best round's Test MSE: 0.1708, MAE: 0.2825, SWD: 0.0647
    Best round's Validation MSE: 0.2645, MAE: 0.3475
    Best round's Test verification MSE : 0.1708, MAE: 0.2825, SWD: 0.0647
    Time taken: 25.16 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.3566, mae: 0.3506, huber: 0.1342, swd: 0.1238, target_std: 0.7879
    Epoch [1/50], Val Losses: mse: 0.2797, mae: 0.3579, huber: 0.1317, swd: 0.1147, target_std: 0.9798
    Epoch [1/50], Test Losses: mse: 0.1733, mae: 0.2862, huber: 0.0832, swd: 0.0606, target_std: 0.7349
      Epoch 1 composite train-obj: 0.134225
            Val objective improved inf → 0.1317, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2993, mae: 0.3143, huber: 0.1135, swd: 0.1132, target_std: 0.7879
    Epoch [2/50], Val Losses: mse: 0.2706, mae: 0.3524, huber: 0.1278, swd: 0.1114, target_std: 0.9798
    Epoch [2/50], Test Losses: mse: 0.1700, mae: 0.2834, huber: 0.0816, swd: 0.0604, target_std: 0.7349
      Epoch 2 composite train-obj: 0.113461
            Val objective improved 0.1317 → 0.1278, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2903, mae: 0.3096, huber: 0.1104, swd: 0.1107, target_std: 0.7878
    Epoch [3/50], Val Losses: mse: 0.2808, mae: 0.3623, huber: 0.1328, swd: 0.1129, target_std: 0.9798
    Epoch [3/50], Test Losses: mse: 0.1741, mae: 0.2906, huber: 0.0838, swd: 0.0621, target_std: 0.7349
      Epoch 3 composite train-obj: 0.110433
            No improvement (0.1328), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.2840, mae: 0.3066, huber: 0.1086, swd: 0.1090, target_std: 0.7879
    Epoch [4/50], Val Losses: mse: 0.2695, mae: 0.3501, huber: 0.1272, swd: 0.1125, target_std: 0.9798
    Epoch [4/50], Test Losses: mse: 0.1684, mae: 0.2815, huber: 0.0808, swd: 0.0618, target_std: 0.7349
      Epoch 4 composite train-obj: 0.108592
            Val objective improved 0.1278 → 0.1272, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.2793, mae: 0.3051, huber: 0.1075, swd: 0.1074, target_std: 0.7879
    Epoch [5/50], Val Losses: mse: 0.2769, mae: 0.3596, huber: 0.1312, swd: 0.1107, target_std: 0.9798
    Epoch [5/50], Test Losses: mse: 0.1749, mae: 0.2913, huber: 0.0841, swd: 0.0623, target_std: 0.7349
      Epoch 5 composite train-obj: 0.107480
            No improvement (0.1312), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.2757, mae: 0.3036, huber: 0.1064, swd: 0.1060, target_std: 0.7880
    Epoch [6/50], Val Losses: mse: 0.2672, mae: 0.3498, huber: 0.1265, swd: 0.1097, target_std: 0.9798
    Epoch [6/50], Test Losses: mse: 0.1702, mae: 0.2837, huber: 0.0816, swd: 0.0623, target_std: 0.7349
      Epoch 6 composite train-obj: 0.106420
            Val objective improved 0.1272 → 0.1265, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.2726, mae: 0.3021, huber: 0.1055, swd: 0.1041, target_std: 0.7878
    Epoch [7/50], Val Losses: mse: 0.2681, mae: 0.3491, huber: 0.1267, swd: 0.1116, target_std: 0.9798
    Epoch [7/50], Test Losses: mse: 0.1700, mae: 0.2834, huber: 0.0815, swd: 0.0617, target_std: 0.7349
      Epoch 7 composite train-obj: 0.105526
            No improvement (0.1267), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.2683, mae: 0.3003, huber: 0.1043, swd: 0.1022, target_std: 0.7880
    Epoch [8/50], Val Losses: mse: 0.2707, mae: 0.3506, huber: 0.1278, swd: 0.1116, target_std: 0.9798
    Epoch [8/50], Test Losses: mse: 0.1708, mae: 0.2845, huber: 0.0820, swd: 0.0617, target_std: 0.7349
      Epoch 8 composite train-obj: 0.104278
            No improvement (0.1278), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.2663, mae: 0.2994, huber: 0.1036, swd: 0.1014, target_std: 0.7879
    Epoch [9/50], Val Losses: mse: 0.2690, mae: 0.3490, huber: 0.1269, swd: 0.1117, target_std: 0.9798
    Epoch [9/50], Test Losses: mse: 0.1702, mae: 0.2832, huber: 0.0817, swd: 0.0619, target_std: 0.7349
      Epoch 9 composite train-obj: 0.103639
            No improvement (0.1269), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.2615, mae: 0.2977, huber: 0.1024, swd: 0.1001, target_std: 0.7879
    Epoch [10/50], Val Losses: mse: 0.2639, mae: 0.3465, huber: 0.1247, swd: 0.1104, target_std: 0.9798
    Epoch [10/50], Test Losses: mse: 0.1723, mae: 0.2838, huber: 0.0825, swd: 0.0643, target_std: 0.7349
      Epoch 10 composite train-obj: 0.102391
            Val objective improved 0.1265 → 0.1247, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.2579, mae: 0.2969, huber: 0.1016, swd: 0.0997, target_std: 0.7879
    Epoch [11/50], Val Losses: mse: 0.2659, mae: 0.3463, huber: 0.1253, swd: 0.1122, target_std: 0.9798
    Epoch [11/50], Test Losses: mse: 0.1695, mae: 0.2814, huber: 0.0812, swd: 0.0616, target_std: 0.7349
      Epoch 11 composite train-obj: 0.101589
            No improvement (0.1253), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.2536, mae: 0.2949, huber: 0.1003, swd: 0.0982, target_std: 0.7879
    Epoch [12/50], Val Losses: mse: 0.2619, mae: 0.3441, huber: 0.1237, swd: 0.1105, target_std: 0.9798
    Epoch [12/50], Test Losses: mse: 0.1704, mae: 0.2817, huber: 0.0817, swd: 0.0625, target_std: 0.7349
      Epoch 12 composite train-obj: 0.100335
            Val objective improved 0.1247 → 0.1237, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.2497, mae: 0.2931, huber: 0.0991, swd: 0.0967, target_std: 0.7879
    Epoch [13/50], Val Losses: mse: 0.2622, mae: 0.3443, huber: 0.1238, swd: 0.1118, target_std: 0.9798
    Epoch [13/50], Test Losses: mse: 0.1696, mae: 0.2820, huber: 0.0815, swd: 0.0624, target_std: 0.7349
      Epoch 13 composite train-obj: 0.099059
            No improvement (0.1238), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.2455, mae: 0.2914, huber: 0.0979, swd: 0.0952, target_std: 0.7880
    Epoch [14/50], Val Losses: mse: 0.2610, mae: 0.3437, huber: 0.1232, swd: 0.1114, target_std: 0.9798
    Epoch [14/50], Test Losses: mse: 0.1717, mae: 0.2831, huber: 0.0823, swd: 0.0641, target_std: 0.7349
      Epoch 14 composite train-obj: 0.097918
            Val objective improved 0.1237 → 0.1232, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.2430, mae: 0.2902, huber: 0.0970, swd: 0.0947, target_std: 0.7882
    Epoch [15/50], Val Losses: mse: 0.2663, mae: 0.3455, huber: 0.1252, swd: 0.1150, target_std: 0.9798
    Epoch [15/50], Test Losses: mse: 0.1708, mae: 0.2822, huber: 0.0820, swd: 0.0640, target_std: 0.7349
      Epoch 15 composite train-obj: 0.096979
            No improvement (0.1252), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.2387, mae: 0.2888, huber: 0.0959, swd: 0.0931, target_std: 0.7880
    Epoch [16/50], Val Losses: mse: 0.2635, mae: 0.3454, huber: 0.1242, swd: 0.1127, target_std: 0.9798
    Epoch [16/50], Test Losses: mse: 0.1727, mae: 0.2846, huber: 0.0829, swd: 0.0647, target_std: 0.7349
      Epoch 16 composite train-obj: 0.095872
            No improvement (0.1242), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.2375, mae: 0.2882, huber: 0.0954, swd: 0.0922, target_std: 0.7881
    Epoch [17/50], Val Losses: mse: 0.2644, mae: 0.3448, huber: 0.1243, swd: 0.1139, target_std: 0.9798
    Epoch [17/50], Test Losses: mse: 0.1707, mae: 0.2823, huber: 0.0820, swd: 0.0635, target_std: 0.7349
      Epoch 17 composite train-obj: 0.095378
            No improvement (0.1243), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.2346, mae: 0.2866, huber: 0.0944, swd: 0.0912, target_std: 0.7878
    Epoch [18/50], Val Losses: mse: 0.2662, mae: 0.3483, huber: 0.1256, swd: 0.1142, target_std: 0.9798
    Epoch [18/50], Test Losses: mse: 0.1740, mae: 0.2878, huber: 0.0838, swd: 0.0640, target_std: 0.7349
      Epoch 18 composite train-obj: 0.094387
            No improvement (0.1256), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.2314, mae: 0.2855, huber: 0.0934, swd: 0.0897, target_std: 0.7879
    Epoch [19/50], Val Losses: mse: 0.2692, mae: 0.3490, huber: 0.1264, swd: 0.1147, target_std: 0.9798
    Epoch [19/50], Test Losses: mse: 0.1721, mae: 0.2850, huber: 0.0828, swd: 0.0641, target_std: 0.7349
      Epoch 19 composite train-obj: 0.093434
    Epoch [19/50], Test Losses: mse: 0.1717, mae: 0.2831, huber: 0.0823, swd: 0.0641, target_std: 0.7349
    Best round's Test MSE: 0.1717, MAE: 0.2831, SWD: 0.0641
    Best round's Validation MSE: 0.2610, MAE: 0.3437
    Best round's Test verification MSE : 0.1717, MAE: 0.2831, SWD: 0.0641
    Time taken: 45.86 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.3748, mae: 0.3611, huber: 0.1402, swd: 0.1210, target_std: 0.7879
    Epoch [1/50], Val Losses: mse: 0.2727, mae: 0.3543, huber: 0.1288, swd: 0.1021, target_std: 0.9798
    Epoch [1/50], Test Losses: mse: 0.1746, mae: 0.2870, huber: 0.0837, swd: 0.0579, target_std: 0.7349
      Epoch 1 composite train-obj: 0.140232
            Val objective improved inf → 0.1288, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2970, mae: 0.3130, huber: 0.1126, swd: 0.1031, target_std: 0.7877
    Epoch [2/50], Val Losses: mse: 0.2600, mae: 0.3456, huber: 0.1233, swd: 0.1008, target_std: 0.9798
    Epoch [2/50], Test Losses: mse: 0.1700, mae: 0.2817, huber: 0.0815, swd: 0.0587, target_std: 0.7349
      Epoch 2 composite train-obj: 0.112596
            Val objective improved 0.1288 → 0.1233, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2874, mae: 0.3088, huber: 0.1098, swd: 0.1020, target_std: 0.7878
    Epoch [3/50], Val Losses: mse: 0.2672, mae: 0.3504, huber: 0.1265, swd: 0.1011, target_std: 0.9798
    Epoch [3/50], Test Losses: mse: 0.1701, mae: 0.2839, huber: 0.0816, swd: 0.0583, target_std: 0.7349
      Epoch 3 composite train-obj: 0.109782
            No improvement (0.1265), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.2800, mae: 0.3060, huber: 0.1078, swd: 0.1001, target_std: 0.7878
    Epoch [4/50], Val Losses: mse: 0.2673, mae: 0.3496, huber: 0.1265, swd: 0.1007, target_std: 0.9798
    Epoch [4/50], Test Losses: mse: 0.1698, mae: 0.2831, huber: 0.0815, swd: 0.0579, target_std: 0.7349
      Epoch 4 composite train-obj: 0.107797
            No improvement (0.1265), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.2755, mae: 0.3038, huber: 0.1064, swd: 0.0985, target_std: 0.7880
    Epoch [5/50], Val Losses: mse: 0.2602, mae: 0.3432, huber: 0.1231, swd: 0.1018, target_std: 0.9798
    Epoch [5/50], Test Losses: mse: 0.1690, mae: 0.2806, huber: 0.0809, swd: 0.0594, target_std: 0.7349
      Epoch 5 composite train-obj: 0.106403
            Val objective improved 0.1233 → 0.1231, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.2716, mae: 0.3024, huber: 0.1054, swd: 0.0973, target_std: 0.7879
    Epoch [6/50], Val Losses: mse: 0.2704, mae: 0.3512, huber: 0.1278, swd: 0.1003, target_std: 0.9798
    Epoch [6/50], Test Losses: mse: 0.1711, mae: 0.2842, huber: 0.0821, swd: 0.0567, target_std: 0.7349
      Epoch 6 composite train-obj: 0.105407
            No improvement (0.1278), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.2676, mae: 0.3006, huber: 0.1043, swd: 0.0956, target_std: 0.7878
    Epoch [7/50], Val Losses: mse: 0.2664, mae: 0.3466, huber: 0.1257, swd: 0.1015, target_std: 0.9798
    Epoch [7/50], Test Losses: mse: 0.1703, mae: 0.2817, huber: 0.0815, swd: 0.0578, target_std: 0.7349
      Epoch 7 composite train-obj: 0.104306
            No improvement (0.1257), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.2637, mae: 0.2989, huber: 0.1032, swd: 0.0948, target_std: 0.7880
    Epoch [8/50], Val Losses: mse: 0.2706, mae: 0.3492, huber: 0.1276, swd: 0.1026, target_std: 0.9798
    Epoch [8/50], Test Losses: mse: 0.1699, mae: 0.2831, huber: 0.0815, swd: 0.0563, target_std: 0.7349
      Epoch 8 composite train-obj: 0.103222
            No improvement (0.1276), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.2601, mae: 0.2975, huber: 0.1022, swd: 0.0933, target_std: 0.7880
    Epoch [9/50], Val Losses: mse: 0.2622, mae: 0.3440, huber: 0.1237, swd: 0.1020, target_std: 0.9798
    Epoch [9/50], Test Losses: mse: 0.1723, mae: 0.2823, huber: 0.0823, swd: 0.0603, target_std: 0.7349
      Epoch 9 composite train-obj: 0.102175
            No improvement (0.1237), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.2557, mae: 0.2959, huber: 0.1010, swd: 0.0923, target_std: 0.7879
    Epoch [10/50], Val Losses: mse: 0.2680, mae: 0.3488, huber: 0.1266, swd: 0.1027, target_std: 0.9798
    Epoch [10/50], Test Losses: mse: 0.1722, mae: 0.2839, huber: 0.0825, swd: 0.0588, target_std: 0.7349
      Epoch 10 composite train-obj: 0.100989
    Epoch [10/50], Test Losses: mse: 0.1690, mae: 0.2806, huber: 0.0809, swd: 0.0594, target_std: 0.7349
    Best round's Test MSE: 0.1690, MAE: 0.2806, SWD: 0.0594
    Best round's Validation MSE: 0.2602, MAE: 0.3432
    Best round's Test verification MSE : 0.1690, MAE: 0.2806, SWD: 0.0594
    Time taken: 24.32 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth2_seq96_pred96_20250502_2123)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.1705 ± 0.0011
      mae: 0.2821 ± 0.0011
      huber: 0.0817 ± 0.0006
      swd: 0.0627 ± 0.0024
      target_std: 0.7349 ± 0.0000
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.2619 ± 0.0018
      mae: 0.3448 ± 0.0019
      huber: 0.1238 ± 0.0009
      swd: 0.1082 ± 0.0046
      target_std: 0.9798 ± 0.0000
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 95.41 seconds
    
    Experiment complete: TimeMixer_etth2_seq96_pred96_20250502_2123
    Model: TimeMixer
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 94
    Validation Batches: 13
    Test Batches: 26
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 27.5532, mae: 2.8856, huber: 2.4722, swd: 14.6452, ept: 71.7718
    Epoch [1/50], Val Losses: mse: 21.3557, mae: 2.9243, huber: 2.5045, swd: 9.3638, ept: 67.9312
    Epoch [1/50], Test Losses: mse: 14.9608, mae: 2.4227, huber: 2.0043, swd: 7.2809, ept: 75.6415
      Epoch 1 composite train-obj: 2.472163
            Val objective improved inf → 2.5045, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.1091, mae: 2.5767, huber: 2.1702, swd: 12.3900, ept: 78.8617
    Epoch [2/50], Val Losses: mse: 20.9834, mae: 2.8883, huber: 2.4696, swd: 9.3270, ept: 68.7220
    Epoch [2/50], Test Losses: mse: 14.5357, mae: 2.3843, huber: 1.9672, swd: 7.0803, ept: 76.3948
      Epoch 2 composite train-obj: 2.170235
            Val objective improved 2.5045 → 2.4696, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.4859, mae: 2.5308, huber: 2.1254, swd: 11.9943, ept: 79.6568
    Epoch [3/50], Val Losses: mse: 20.5436, mae: 2.8633, huber: 2.4452, swd: 9.0064, ept: 69.2530
    Epoch [3/50], Test Losses: mse: 14.4109, mae: 2.3769, huber: 1.9602, swd: 7.0535, ept: 77.1787
      Epoch 3 composite train-obj: 2.125411
            Val objective improved 2.4696 → 2.4452, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.0758, mae: 2.5056, huber: 2.1007, swd: 11.6895, ept: 80.0161
    Epoch [4/50], Val Losses: mse: 21.0730, mae: 2.8943, huber: 2.4762, swd: 9.7085, ept: 69.2520
    Epoch [4/50], Test Losses: mse: 14.3532, mae: 2.3775, huber: 1.9603, swd: 7.0607, ept: 77.3256
      Epoch 4 composite train-obj: 2.100686
            No improvement (2.4762), counter 1/5
    Epoch [5/50], Train Losses: mse: 21.7740, mae: 2.4879, huber: 2.0835, swd: 11.4764, ept: 80.2541
    Epoch [5/50], Val Losses: mse: 20.5514, mae: 2.8605, huber: 2.4430, swd: 9.0246, ept: 69.3299
    Epoch [5/50], Test Losses: mse: 14.5248, mae: 2.3759, huber: 1.9597, swd: 7.1717, ept: 77.1301
      Epoch 5 composite train-obj: 2.083545
            Val objective improved 2.4452 → 2.4430, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 21.3823, mae: 2.4672, huber: 2.0631, swd: 11.1748, ept: 80.4959
    Epoch [6/50], Val Losses: mse: 20.7753, mae: 2.8691, huber: 2.4525, swd: 9.2568, ept: 69.4709
    Epoch [6/50], Test Losses: mse: 14.6311, mae: 2.3760, huber: 1.9604, swd: 7.2275, ept: 77.2282
      Epoch 6 composite train-obj: 2.063144
            No improvement (2.4525), counter 1/5
    Epoch [7/50], Train Losses: mse: 21.0910, mae: 2.4503, huber: 2.0467, swd: 10.9828, ept: 80.5911
    Epoch [7/50], Val Losses: mse: 21.6246, mae: 2.9098, huber: 2.4938, swd: 10.1643, ept: 69.1869
    Epoch [7/50], Test Losses: mse: 14.5643, mae: 2.3820, huber: 1.9649, swd: 7.2987, ept: 77.3832
      Epoch 7 composite train-obj: 2.046685
            No improvement (2.4938), counter 2/5
    Epoch [8/50], Train Losses: mse: 20.6933, mae: 2.4323, huber: 2.0288, swd: 10.6956, ept: 80.7498
    Epoch [8/50], Val Losses: mse: 20.6812, mae: 2.8657, huber: 2.4494, swd: 9.0354, ept: 69.4637
    Epoch [8/50], Test Losses: mse: 14.8438, mae: 2.3862, huber: 1.9703, swd: 7.4055, ept: 76.8926
      Epoch 8 composite train-obj: 2.028819
            No improvement (2.4494), counter 3/5
    Epoch [9/50], Train Losses: mse: 20.4853, mae: 2.4184, huber: 2.0153, swd: 10.5574, ept: 80.8468
    Epoch [9/50], Val Losses: mse: 21.3149, mae: 2.9133, huber: 2.4956, swd: 9.6914, ept: 68.6166
    Epoch [9/50], Test Losses: mse: 14.8289, mae: 2.3955, huber: 1.9786, swd: 7.3676, ept: 76.7775
      Epoch 9 composite train-obj: 2.015329
            No improvement (2.4956), counter 4/5
    Epoch [10/50], Train Losses: mse: 20.0510, mae: 2.4040, huber: 2.0008, swd: 10.2366, ept: 80.8432
    Epoch [10/50], Val Losses: mse: 20.9581, mae: 2.8804, huber: 2.4632, swd: 9.2563, ept: 69.0053
    Epoch [10/50], Test Losses: mse: 14.8409, mae: 2.3848, huber: 1.9693, swd: 7.3724, ept: 76.4210
      Epoch 10 composite train-obj: 2.000783
    Epoch [10/50], Test Losses: mse: 14.5248, mae: 2.3759, huber: 1.9597, swd: 7.1717, ept: 77.1301
    Best round's Test MSE: 14.5248, MAE: 2.3759, SWD: 7.1717
    Best round's Validation MSE: 20.5514, MAE: 2.8605, SWD: 9.0246
    Best round's Test verification MSE : 14.5248, MAE: 2.3759, SWD: 7.1717
    Time taken: 26.59 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 28.1649, mae: 2.9179, huber: 2.5041, swd: 14.6486, ept: 71.5798
    Epoch [1/50], Val Losses: mse: 22.4442, mae: 2.9910, huber: 2.5705, swd: 9.9151, ept: 67.6632
    Epoch [1/50], Test Losses: mse: 14.8522, mae: 2.4378, huber: 2.0180, swd: 7.0368, ept: 76.4399
      Epoch 1 composite train-obj: 2.504149
            Val objective improved inf → 2.5705, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.2837, mae: 2.5856, huber: 2.1791, swd: 12.1207, ept: 78.9102
    Epoch [2/50], Val Losses: mse: 20.9989, mae: 2.8997, huber: 2.4797, swd: 8.9230, ept: 68.9612
    Epoch [2/50], Test Losses: mse: 14.5107, mae: 2.3960, huber: 1.9780, swd: 6.7761, ept: 76.7550
      Epoch 2 composite train-obj: 2.179066
            Val objective improved 2.5705 → 2.4797, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.6202, mae: 2.5364, huber: 2.1309, swd: 11.7110, ept: 79.5602
    Epoch [3/50], Val Losses: mse: 21.2106, mae: 2.8951, huber: 2.4769, swd: 9.2314, ept: 69.3895
    Epoch [3/50], Test Losses: mse: 14.4045, mae: 2.3785, huber: 1.9613, swd: 6.7841, ept: 77.0958
      Epoch 3 composite train-obj: 2.130940
            Val objective improved 2.4797 → 2.4769, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.2923, mae: 2.5152, huber: 2.1103, swd: 11.5165, ept: 79.9517
    Epoch [4/50], Val Losses: mse: 20.9713, mae: 2.8776, huber: 2.4603, swd: 8.9686, ept: 69.3863
    Epoch [4/50], Test Losses: mse: 14.4829, mae: 2.3738, huber: 1.9576, swd: 6.8196, ept: 77.1561
      Epoch 4 composite train-obj: 2.110330
            Val objective improved 2.4769 → 2.4603, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 21.9671, mae: 2.4976, huber: 2.0931, swd: 11.2733, ept: 80.1427
    Epoch [5/50], Val Losses: mse: 20.4311, mae: 2.8554, huber: 2.4383, swd: 8.6106, ept: 69.6145
    Epoch [5/50], Test Losses: mse: 14.4698, mae: 2.3804, huber: 1.9638, swd: 6.8757, ept: 77.4214
      Epoch 5 composite train-obj: 2.093092
            Val objective improved 2.4603 → 2.4383, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 21.7426, mae: 2.4860, huber: 2.0817, swd: 11.1133, ept: 80.3360
    Epoch [6/50], Val Losses: mse: 20.7761, mae: 2.8818, huber: 2.4639, swd: 9.0121, ept: 69.6261
    Epoch [6/50], Test Losses: mse: 14.4581, mae: 2.3888, huber: 1.9717, swd: 6.9298, ept: 77.9619
      Epoch 6 composite train-obj: 2.081719
            No improvement (2.4639), counter 1/5
    Epoch [7/50], Train Losses: mse: 21.6390, mae: 2.4768, huber: 2.0730, swd: 11.0948, ept: 80.4384
    Epoch [7/50], Val Losses: mse: 21.2622, mae: 2.8872, huber: 2.4714, swd: 9.2789, ept: 69.2825
    Epoch [7/50], Test Losses: mse: 14.5307, mae: 2.3713, huber: 1.9556, swd: 6.8980, ept: 77.1238
      Epoch 7 composite train-obj: 2.072985
            No improvement (2.4714), counter 2/5
    Epoch [8/50], Train Losses: mse: 21.4705, mae: 2.4655, huber: 2.0620, swd: 10.9550, ept: 80.5278
    Epoch [8/50], Val Losses: mse: 21.3387, mae: 2.9037, huber: 2.4870, swd: 9.5614, ept: 69.4657
    Epoch [8/50], Test Losses: mse: 14.4693, mae: 2.3894, huber: 1.9722, swd: 6.9439, ept: 77.6979
      Epoch 8 composite train-obj: 2.062027
            No improvement (2.4870), counter 3/5
    Epoch [9/50], Train Losses: mse: 21.3025, mae: 2.4552, huber: 2.0519, swd: 10.8435, ept: 80.6130
    Epoch [9/50], Val Losses: mse: 21.6329, mae: 2.9128, huber: 2.4965, swd: 9.7072, ept: 69.3361
    Epoch [9/50], Test Losses: mse: 14.4923, mae: 2.3853, huber: 1.9683, swd: 6.9801, ept: 77.7499
      Epoch 9 composite train-obj: 2.051944
            No improvement (2.4965), counter 4/5
    Epoch [10/50], Train Losses: mse: 21.0214, mae: 2.4416, huber: 2.0386, swd: 10.6039, ept: 80.7575
    Epoch [10/50], Val Losses: mse: 20.7080, mae: 2.8674, huber: 2.4509, swd: 8.7959, ept: 69.5484
    Epoch [10/50], Test Losses: mse: 14.4933, mae: 2.3734, huber: 1.9581, swd: 6.8966, ept: 77.5426
      Epoch 10 composite train-obj: 2.038558
    Epoch [10/50], Test Losses: mse: 14.4698, mae: 2.3804, huber: 1.9638, swd: 6.8757, ept: 77.4214
    Best round's Test MSE: 14.4698, MAE: 2.3804, SWD: 6.8757
    Best round's Validation MSE: 20.4311, MAE: 2.8554, SWD: 8.6106
    Best round's Test verification MSE : 14.4698, MAE: 2.3804, SWD: 6.8757
    Time taken: 26.68 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 29.6884, mae: 3.0091, huber: 2.5932, swd: 15.1762, ept: 70.9883
    Epoch [1/50], Val Losses: mse: 21.6624, mae: 2.9369, huber: 2.5166, swd: 8.6861, ept: 67.9823
    Epoch [1/50], Test Losses: mse: 15.0331, mae: 2.4336, huber: 2.0143, swd: 6.6391, ept: 75.5902
      Epoch 1 composite train-obj: 2.593157
            Val objective improved inf → 2.5166, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.1512, mae: 2.5784, huber: 2.1719, swd: 11.2744, ept: 79.0322
    Epoch [2/50], Val Losses: mse: 20.3088, mae: 2.8518, huber: 2.4333, swd: 7.7485, ept: 69.0221
    Epoch [2/50], Test Losses: mse: 14.8113, mae: 2.3993, huber: 1.9824, swd: 6.5555, ept: 76.0509
      Epoch 2 composite train-obj: 2.171938
            Val objective improved 2.5166 → 2.4333, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.5771, mae: 2.5384, huber: 2.1328, swd: 10.9229, ept: 79.6240
    Epoch [3/50], Val Losses: mse: 20.6634, mae: 2.8668, huber: 2.4488, swd: 8.2723, ept: 69.4270
    Epoch [3/50], Test Losses: mse: 14.5456, mae: 2.3842, huber: 1.9671, swd: 6.4322, ept: 76.7323
      Epoch 3 composite train-obj: 2.132769
            No improvement (2.4488), counter 1/5
    Epoch [4/50], Train Losses: mse: 22.1503, mae: 2.5113, huber: 2.1063, swd: 10.6357, ept: 79.9688
    Epoch [4/50], Val Losses: mse: 20.8244, mae: 2.8729, huber: 2.4555, swd: 8.4832, ept: 69.5078
    Epoch [4/50], Test Losses: mse: 14.5409, mae: 2.3791, huber: 1.9625, swd: 6.4485, ept: 76.6740
      Epoch 4 composite train-obj: 2.106320
            No improvement (2.4555), counter 2/5
    Epoch [5/50], Train Losses: mse: 21.8791, mae: 2.4934, huber: 2.0890, swd: 10.4656, ept: 80.2259
    Epoch [5/50], Val Losses: mse: 20.2598, mae: 2.8417, huber: 2.4244, swd: 7.9440, ept: 69.9227
    Epoch [5/50], Test Losses: mse: 14.5518, mae: 2.3763, huber: 1.9605, swd: 6.4448, ept: 77.2194
      Epoch 5 composite train-obj: 2.088977
            Val objective improved 2.4333 → 2.4244, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 21.6663, mae: 2.4799, huber: 2.0758, swd: 10.3269, ept: 80.3882
    Epoch [6/50], Val Losses: mse: 20.8424, mae: 2.8724, huber: 2.4553, swd: 8.5599, ept: 69.3067
    Epoch [6/50], Test Losses: mse: 14.5007, mae: 2.3777, huber: 1.9609, swd: 6.4606, ept: 77.0747
      Epoch 6 composite train-obj: 2.075803
            No improvement (2.4553), counter 1/5
    Epoch [7/50], Train Losses: mse: 21.5028, mae: 2.4704, huber: 2.0667, swd: 10.2331, ept: 80.4967
    Epoch [7/50], Val Losses: mse: 21.0910, mae: 2.8800, huber: 2.4640, swd: 8.6279, ept: 68.8576
    Epoch [7/50], Test Losses: mse: 14.6306, mae: 2.3756, huber: 1.9598, swd: 6.5193, ept: 76.6940
      Epoch 7 composite train-obj: 2.066667
            No improvement (2.4640), counter 2/5
    Epoch [8/50], Train Losses: mse: 21.3148, mae: 2.4573, huber: 2.0539, swd: 10.1092, ept: 80.6235
    Epoch [8/50], Val Losses: mse: 21.2397, mae: 2.8852, huber: 2.4695, swd: 8.8732, ept: 69.2846
    Epoch [8/50], Test Losses: mse: 14.4763, mae: 2.3771, huber: 1.9604, swd: 6.5037, ept: 77.2632
      Epoch 8 composite train-obj: 2.053908
            No improvement (2.4695), counter 3/5
    Epoch [9/50], Train Losses: mse: 21.1545, mae: 2.4485, huber: 2.0452, swd: 9.9984, ept: 80.7613
    Epoch [9/50], Val Losses: mse: 20.4668, mae: 2.8513, huber: 2.4342, swd: 8.0854, ept: 69.6696
    Epoch [9/50], Test Losses: mse: 14.7373, mae: 2.3828, huber: 1.9670, swd: 6.5914, ept: 76.9838
      Epoch 9 composite train-obj: 2.045186
            No improvement (2.4342), counter 4/5
    Epoch [10/50], Train Losses: mse: 20.9305, mae: 2.4387, huber: 2.0355, swd: 9.8499, ept: 80.8105
    Epoch [10/50], Val Losses: mse: 20.6765, mae: 2.8640, huber: 2.4472, swd: 8.3506, ept: 69.3976
    Epoch [10/50], Test Losses: mse: 14.6541, mae: 2.3834, huber: 1.9674, swd: 6.5958, ept: 77.1371
      Epoch 10 composite train-obj: 2.035534
    Epoch [10/50], Test Losses: mse: 14.5518, mae: 2.3763, huber: 1.9605, swd: 6.4448, ept: 77.2194
    Best round's Test MSE: 14.5518, MAE: 2.3763, SWD: 6.4448
    Best round's Validation MSE: 20.2598, MAE: 2.8417, SWD: 7.9440
    Best round's Test verification MSE : 14.5518, MAE: 2.3763, SWD: 6.4448
    Time taken: 26.66 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth2_seq96_pred96_20250511_1618)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 14.5155 ± 0.0341
      mae: 2.3775 ± 0.0020
      huber: 1.9613 ± 0.0018
      swd: 6.8307 ± 0.2984
      ept: 77.2570 ± 0.1219
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 20.4141 ± 0.1196
      mae: 2.8525 ± 0.0080
      huber: 2.4352 ± 0.0079
      swd: 8.5264 ± 0.4451
      ept: 69.6224 ± 0.2421
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 79.98 seconds
    
    Experiment complete: TimeMixer_etth2_seq96_pred96_20250511_1618
    Model: TimeMixer
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.4496, mae: 0.3984, huber: 0.1659, swd: 0.1672, target_std: 0.7894
    Epoch [1/50], Val Losses: mse: 0.3551, mae: 0.4157, huber: 0.1652, swd: 0.1424, target_std: 0.9804
    Epoch [1/50], Test Losses: mse: 0.2070, mae: 0.3147, huber: 0.0991, swd: 0.0762, target_std: 0.7299
      Epoch 1 composite train-obj: 0.165902
            Val objective improved inf → 0.1652, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3903, mae: 0.3609, huber: 0.1438, swd: 0.1525, target_std: 0.7895
    Epoch [2/50], Val Losses: mse: 0.3421, mae: 0.4069, huber: 0.1595, swd: 0.1415, target_std: 0.9804
    Epoch [2/50], Test Losses: mse: 0.2021, mae: 0.3107, huber: 0.0969, swd: 0.0758, target_std: 0.7299
      Epoch 2 composite train-obj: 0.143802
            Val objective improved 0.1652 → 0.1595, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3826, mae: 0.3568, huber: 0.1410, swd: 0.1514, target_std: 0.7894
    Epoch [3/50], Val Losses: mse: 0.3434, mae: 0.4060, huber: 0.1599, swd: 0.1438, target_std: 0.9804
    Epoch [3/50], Test Losses: mse: 0.2002, mae: 0.3096, huber: 0.0962, swd: 0.0765, target_std: 0.7299
      Epoch 3 composite train-obj: 0.140975
            No improvement (0.1599), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3747, mae: 0.3536, huber: 0.1385, swd: 0.1489, target_std: 0.7894
    Epoch [4/50], Val Losses: mse: 0.3414, mae: 0.4064, huber: 0.1594, swd: 0.1425, target_std: 0.9804
    Epoch [4/50], Test Losses: mse: 0.2011, mae: 0.3113, huber: 0.0967, swd: 0.0771, target_std: 0.7299
      Epoch 4 composite train-obj: 0.138459
            Val objective improved 0.1595 → 0.1594, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3694, mae: 0.3518, huber: 0.1371, swd: 0.1469, target_std: 0.7895
    Epoch [5/50], Val Losses: mse: 0.3464, mae: 0.4060, huber: 0.1610, swd: 0.1463, target_std: 0.9804
    Epoch [5/50], Test Losses: mse: 0.2003, mae: 0.3095, huber: 0.0964, swd: 0.0772, target_std: 0.7299
      Epoch 5 composite train-obj: 0.137081
            No improvement (0.1610), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3643, mae: 0.3496, huber: 0.1355, swd: 0.1453, target_std: 0.7895
    Epoch [6/50], Val Losses: mse: 0.3454, mae: 0.4065, huber: 0.1608, swd: 0.1444, target_std: 0.9804
    Epoch [6/50], Test Losses: mse: 0.2008, mae: 0.3104, huber: 0.0967, swd: 0.0763, target_std: 0.7299
      Epoch 6 composite train-obj: 0.135524
            No improvement (0.1608), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.3592, mae: 0.3479, huber: 0.1341, swd: 0.1439, target_std: 0.7894
    Epoch [7/50], Val Losses: mse: 0.3543, mae: 0.4110, huber: 0.1641, swd: 0.1488, target_std: 0.9804
    Epoch [7/50], Test Losses: mse: 0.2026, mae: 0.3111, huber: 0.0974, swd: 0.0770, target_std: 0.7299
      Epoch 7 composite train-obj: 0.134092
            No improvement (0.1641), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.3536, mae: 0.3458, huber: 0.1325, swd: 0.1416, target_std: 0.7894
    Epoch [8/50], Val Losses: mse: 0.3567, mae: 0.4149, huber: 0.1659, swd: 0.1492, target_std: 0.9804
    Epoch [8/50], Test Losses: mse: 0.2052, mae: 0.3166, huber: 0.0989, swd: 0.0792, target_std: 0.7299
      Epoch 8 composite train-obj: 0.132500
            No improvement (0.1659), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.3500, mae: 0.3443, huber: 0.1313, swd: 0.1414, target_std: 0.7894
    Epoch [9/50], Val Losses: mse: 0.3527, mae: 0.4119, huber: 0.1641, swd: 0.1475, target_std: 0.9804
    Epoch [9/50], Test Losses: mse: 0.2030, mae: 0.3142, huber: 0.0978, swd: 0.0768, target_std: 0.7299
      Epoch 9 composite train-obj: 0.131305
    Epoch [9/50], Test Losses: mse: 0.2011, mae: 0.3113, huber: 0.0967, swd: 0.0771, target_std: 0.7299
    Best round's Test MSE: 0.2011, MAE: 0.3113, SWD: 0.0771
    Best round's Validation MSE: 0.3414, MAE: 0.4064
    Best round's Test verification MSE : 0.2011, MAE: 0.3113, SWD: 0.0771
    Time taken: 21.44 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4379, mae: 0.3899, huber: 0.1612, swd: 0.1721, target_std: 0.7895
    Epoch [1/50], Val Losses: mse: 0.3563, mae: 0.4156, huber: 0.1657, swd: 0.1460, target_std: 0.9804
    Epoch [1/50], Test Losses: mse: 0.2073, mae: 0.3152, huber: 0.0992, swd: 0.0795, target_std: 0.7299
      Epoch 1 composite train-obj: 0.161185
            Val objective improved inf → 0.1657, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3880, mae: 0.3583, huber: 0.1425, swd: 0.1566, target_std: 0.7894
    Epoch [2/50], Val Losses: mse: 0.3462, mae: 0.4083, huber: 0.1614, swd: 0.1471, target_std: 0.9804
    Epoch [2/50], Test Losses: mse: 0.2024, mae: 0.3113, huber: 0.0972, swd: 0.0792, target_std: 0.7299
      Epoch 2 composite train-obj: 0.142504
            Val objective improved 0.1657 → 0.1614, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3760, mae: 0.3531, huber: 0.1386, swd: 0.1521, target_std: 0.7894
    Epoch [3/50], Val Losses: mse: 0.3407, mae: 0.4039, huber: 0.1589, swd: 0.1496, target_std: 0.9804
    Epoch [3/50], Test Losses: mse: 0.2011, mae: 0.3111, huber: 0.0968, swd: 0.0807, target_std: 0.7299
      Epoch 3 composite train-obj: 0.138629
            Val objective improved 0.1614 → 0.1589, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3675, mae: 0.3502, huber: 0.1361, swd: 0.1497, target_std: 0.7894
    Epoch [4/50], Val Losses: mse: 0.3414, mae: 0.4037, huber: 0.1590, swd: 0.1477, target_std: 0.9804
    Epoch [4/50], Test Losses: mse: 0.2015, mae: 0.3106, huber: 0.0969, swd: 0.0802, target_std: 0.7299
      Epoch 4 composite train-obj: 0.136105
            No improvement (0.1590), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3611, mae: 0.3475, huber: 0.1342, swd: 0.1468, target_std: 0.7894
    Epoch [5/50], Val Losses: mse: 0.3368, mae: 0.4026, huber: 0.1572, swd: 0.1435, target_std: 0.9804
    Epoch [5/50], Test Losses: mse: 0.2018, mae: 0.3119, huber: 0.0971, swd: 0.0798, target_std: 0.7299
      Epoch 5 composite train-obj: 0.134175
            Val objective improved 0.1589 → 0.1572, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3561, mae: 0.3456, huber: 0.1326, swd: 0.1456, target_std: 0.7894
    Epoch [6/50], Val Losses: mse: 0.3436, mae: 0.4059, huber: 0.1598, swd: 0.1489, target_std: 0.9804
    Epoch [6/50], Test Losses: mse: 0.2032, mae: 0.3132, huber: 0.0978, swd: 0.0821, target_std: 0.7299
      Epoch 6 composite train-obj: 0.132611
            No improvement (0.1598), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3486, mae: 0.3430, huber: 0.1304, swd: 0.1432, target_std: 0.7895
    Epoch [7/50], Val Losses: mse: 0.3427, mae: 0.4064, huber: 0.1596, swd: 0.1478, target_std: 0.9804
    Epoch [7/50], Test Losses: mse: 0.2043, mae: 0.3152, huber: 0.0984, swd: 0.0825, target_std: 0.7299
      Epoch 7 composite train-obj: 0.130433
            No improvement (0.1596), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3443, mae: 0.3412, huber: 0.1290, swd: 0.1424, target_std: 0.7894
    Epoch [8/50], Val Losses: mse: 0.3396, mae: 0.4039, huber: 0.1577, swd: 0.1474, target_std: 0.9804
    Epoch [8/50], Test Losses: mse: 0.2031, mae: 0.3132, huber: 0.0977, swd: 0.0816, target_std: 0.7299
      Epoch 8 composite train-obj: 0.129045
            No improvement (0.1577), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3378, mae: 0.3387, huber: 0.1270, swd: 0.1409, target_std: 0.7894
    Epoch [9/50], Val Losses: mse: 0.3461, mae: 0.4101, huber: 0.1614, swd: 0.1470, target_std: 0.9804
    Epoch [9/50], Test Losses: mse: 0.2038, mae: 0.3156, huber: 0.0982, swd: 0.0808, target_std: 0.7299
      Epoch 9 composite train-obj: 0.127005
            No improvement (0.1614), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.3324, mae: 0.3365, huber: 0.1254, swd: 0.1389, target_std: 0.7894
    Epoch [10/50], Val Losses: mse: 0.3411, mae: 0.4060, huber: 0.1588, swd: 0.1456, target_std: 0.9804
    Epoch [10/50], Test Losses: mse: 0.2023, mae: 0.3129, huber: 0.0973, swd: 0.0798, target_std: 0.7299
      Epoch 10 composite train-obj: 0.125383
    Epoch [10/50], Test Losses: mse: 0.2018, mae: 0.3119, huber: 0.0971, swd: 0.0798, target_std: 0.7299
    Best round's Test MSE: 0.2018, MAE: 0.3119, SWD: 0.0798
    Best round's Validation MSE: 0.3368, MAE: 0.4026
    Best round's Test verification MSE : 0.2018, MAE: 0.3119, SWD: 0.0798
    Time taken: 24.32 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4226, mae: 0.3811, huber: 0.1559, swd: 0.1510, target_std: 0.7894
    Epoch [1/50], Val Losses: mse: 0.3549, mae: 0.4128, huber: 0.1647, swd: 0.1283, target_std: 0.9804
    Epoch [1/50], Test Losses: mse: 0.2048, mae: 0.3118, huber: 0.0980, swd: 0.0665, target_std: 0.7299
      Epoch 1 composite train-obj: 0.155921
            Val objective improved inf → 0.1647, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3868, mae: 0.3575, huber: 0.1421, swd: 0.1406, target_std: 0.7894
    Epoch [2/50], Val Losses: mse: 0.3414, mae: 0.4056, huber: 0.1593, swd: 0.1238, target_std: 0.9804
    Epoch [2/50], Test Losses: mse: 0.2021, mae: 0.3110, huber: 0.0970, swd: 0.0658, target_std: 0.7299
      Epoch 2 composite train-obj: 0.142085
            Val objective improved 0.1647 → 0.1593, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3804, mae: 0.3541, huber: 0.1397, swd: 0.1384, target_std: 0.7894
    Epoch [3/50], Val Losses: mse: 0.3535, mae: 0.4100, huber: 0.1640, swd: 0.1295, target_std: 0.9804
    Epoch [3/50], Test Losses: mse: 0.2025, mae: 0.3119, huber: 0.0973, swd: 0.0657, target_std: 0.7299
      Epoch 3 composite train-obj: 0.139663
            No improvement (0.1640), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3740, mae: 0.3516, huber: 0.1376, swd: 0.1360, target_std: 0.7894
    Epoch [4/50], Val Losses: mse: 0.3415, mae: 0.4065, huber: 0.1595, swd: 0.1261, target_std: 0.9804
    Epoch [4/50], Test Losses: mse: 0.2009, mae: 0.3115, huber: 0.0966, swd: 0.0675, target_std: 0.7299
      Epoch 4 composite train-obj: 0.137616
            No improvement (0.1595), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3660, mae: 0.3490, huber: 0.1353, swd: 0.1332, target_std: 0.7894
    Epoch [5/50], Val Losses: mse: 0.3427, mae: 0.4071, huber: 0.1600, swd: 0.1271, target_std: 0.9804
    Epoch [5/50], Test Losses: mse: 0.2038, mae: 0.3157, huber: 0.0982, swd: 0.0693, target_std: 0.7299
      Epoch 5 composite train-obj: 0.135304
            No improvement (0.1600), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.3552, mae: 0.3455, huber: 0.1323, swd: 0.1306, target_std: 0.7894
    Epoch [6/50], Val Losses: mse: 0.3434, mae: 0.4052, huber: 0.1595, swd: 0.1312, target_std: 0.9804
    Epoch [6/50], Test Losses: mse: 0.2008, mae: 0.3099, huber: 0.0964, swd: 0.0692, target_std: 0.7299
      Epoch 6 composite train-obj: 0.132336
            No improvement (0.1595), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.3469, mae: 0.3419, huber: 0.1296, swd: 0.1289, target_std: 0.7894
    Epoch [7/50], Val Losses: mse: 0.3416, mae: 0.4048, huber: 0.1587, swd: 0.1273, target_std: 0.9804
    Epoch [7/50], Test Losses: mse: 0.2021, mae: 0.3130, huber: 0.0973, swd: 0.0684, target_std: 0.7299
      Epoch 7 composite train-obj: 0.129551
            Val objective improved 0.1593 → 0.1587, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3385, mae: 0.3385, huber: 0.1269, swd: 0.1279, target_std: 0.7894
    Epoch [8/50], Val Losses: mse: 0.3469, mae: 0.4079, huber: 0.1608, swd: 0.1311, target_std: 0.9804
    Epoch [8/50], Test Losses: mse: 0.2040, mae: 0.3142, huber: 0.0981, swd: 0.0682, target_std: 0.7299
      Epoch 8 composite train-obj: 0.126911
            No improvement (0.1608), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3315, mae: 0.3351, huber: 0.1245, swd: 0.1266, target_std: 0.7895
    Epoch [9/50], Val Losses: mse: 0.3502, mae: 0.4115, huber: 0.1627, swd: 0.1329, target_std: 0.9804
    Epoch [9/50], Test Losses: mse: 0.2032, mae: 0.3150, huber: 0.0980, swd: 0.0681, target_std: 0.7299
      Epoch 9 composite train-obj: 0.124479
            No improvement (0.1627), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.3257, mae: 0.3324, huber: 0.1226, swd: 0.1252, target_std: 0.7894
    Epoch [10/50], Val Losses: mse: 0.3485, mae: 0.4092, huber: 0.1615, swd: 0.1367, target_std: 0.9804
    Epoch [10/50], Test Losses: mse: 0.2012, mae: 0.3126, huber: 0.0969, swd: 0.0688, target_std: 0.7299
      Epoch 10 composite train-obj: 0.122601
            No improvement (0.1615), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.3216, mae: 0.3303, huber: 0.1211, swd: 0.1239, target_std: 0.7895
    Epoch [11/50], Val Losses: mse: 0.3599, mae: 0.4155, huber: 0.1659, swd: 0.1416, target_std: 0.9804
    Epoch [11/50], Test Losses: mse: 0.2032, mae: 0.3138, huber: 0.0978, swd: 0.0680, target_std: 0.7299
      Epoch 11 composite train-obj: 0.121126
            No improvement (0.1659), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.3173, mae: 0.3281, huber: 0.1197, swd: 0.1235, target_std: 0.7894
    Epoch [12/50], Val Losses: mse: 0.3549, mae: 0.4115, huber: 0.1634, swd: 0.1402, target_std: 0.9804
    Epoch [12/50], Test Losses: mse: 0.2041, mae: 0.3143, huber: 0.0981, swd: 0.0679, target_std: 0.7299
      Epoch 12 composite train-obj: 0.119653
    Epoch [12/50], Test Losses: mse: 0.2021, mae: 0.3130, huber: 0.0973, swd: 0.0684, target_std: 0.7299
    Best round's Test MSE: 0.2021, MAE: 0.3130, SWD: 0.0684
    Best round's Validation MSE: 0.3416, MAE: 0.4048
    Best round's Test verification MSE : 0.2021, MAE: 0.3130, SWD: 0.0684
    Time taken: 29.90 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth2_seq96_pred196_20250502_2145)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2017 ± 0.0005
      mae: 0.3120 ± 0.0007
      huber: 0.0971 ± 0.0003
      swd: 0.0751 ± 0.0049
      target_std: 0.7299 ± 0.0000
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3399 ± 0.0022
      mae: 0.4046 ± 0.0015
      huber: 0.1584 ± 0.0009
      swd: 0.1377 ± 0.0074
      target_std: 0.9804 ± 0.0000
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 75.70 seconds
    
    Experiment complete: TimeMixer_etth2_seq96_pred196_20250502_2145
    Model: TimeMixer
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 12
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 35.8134, mae: 3.3034, huber: 2.8830, swd: 17.7446, ept: 121.5065
    Epoch [1/50], Val Losses: mse: 27.4556, mae: 3.4437, huber: 3.0122, swd: 12.3305, ept: 108.4725
    Epoch [1/50], Test Losses: mse: 17.2682, mae: 2.6753, huber: 2.2515, swd: 8.3212, ept: 131.2117
      Epoch 1 composite train-obj: 2.883044
            Val objective improved inf → 3.0122, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 30.8821, mae: 2.9786, huber: 2.5644, swd: 16.1849, ept: 137.7952
    Epoch [2/50], Val Losses: mse: 26.1442, mae: 3.3445, huber: 2.9147, swd: 11.1936, ept: 110.8794
    Epoch [2/50], Test Losses: mse: 16.9732, mae: 2.6333, huber: 2.2111, swd: 8.0808, ept: 131.4153
      Epoch 2 composite train-obj: 2.564405
            Val objective improved 3.0122 → 2.9147, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 30.3180, mae: 2.9374, huber: 2.5238, swd: 15.8466, ept: 139.7783
    Epoch [3/50], Val Losses: mse: 26.5645, mae: 3.3551, huber: 2.9261, swd: 11.8021, ept: 112.2238
    Epoch [3/50], Test Losses: mse: 16.8537, mae: 2.6271, huber: 2.2048, swd: 8.1246, ept: 132.6998
      Epoch 3 composite train-obj: 2.523760
            No improvement (2.9261), counter 1/5
    Epoch [4/50], Train Losses: mse: 29.8685, mae: 2.9130, huber: 2.4998, swd: 15.5257, ept: 140.7226
    Epoch [4/50], Val Losses: mse: 26.1149, mae: 3.3370, huber: 2.9081, swd: 11.4229, ept: 112.4000
    Epoch [4/50], Test Losses: mse: 16.7865, mae: 2.6227, huber: 2.2010, swd: 8.0589, ept: 133.3760
      Epoch 4 composite train-obj: 2.499772
            Val objective improved 2.9147 → 2.9081, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 29.5587, mae: 2.9004, huber: 2.4873, swd: 15.2712, ept: 141.1613
    Epoch [5/50], Val Losses: mse: 26.7186, mae: 3.3469, huber: 2.9209, swd: 11.7560, ept: 112.0646
    Epoch [5/50], Test Losses: mse: 17.0127, mae: 2.6226, huber: 2.2015, swd: 8.2256, ept: 132.9340
      Epoch 5 composite train-obj: 2.487252
            No improvement (2.9209), counter 1/5
    Epoch [6/50], Train Losses: mse: 29.2561, mae: 2.8830, huber: 2.4702, swd: 15.0300, ept: 141.7642
    Epoch [6/50], Val Losses: mse: 26.7262, mae: 3.3547, huber: 2.9278, swd: 11.7587, ept: 112.4834
    Epoch [6/50], Test Losses: mse: 16.9481, mae: 2.6242, huber: 2.2033, swd: 8.1544, ept: 133.0548
      Epoch 6 composite train-obj: 2.470188
            No improvement (2.9278), counter 2/5
    Epoch [7/50], Train Losses: mse: 28.9656, mae: 2.8696, huber: 2.4570, swd: 14.8245, ept: 142.1957
    Epoch [7/50], Val Losses: mse: 27.4116, mae: 3.3814, huber: 2.9548, swd: 12.3189, ept: 111.5867
    Epoch [7/50], Test Losses: mse: 17.2097, mae: 2.6283, huber: 2.2077, swd: 8.3521, ept: 132.0568
      Epoch 7 composite train-obj: 2.456995
            No improvement (2.9548), counter 3/5
    Epoch [8/50], Train Losses: mse: 28.6480, mae: 2.8562, huber: 2.4439, swd: 14.5601, ept: 142.6022
    Epoch [8/50], Val Losses: mse: 27.2680, mae: 3.3845, huber: 2.9575, swd: 12.6020, ept: 113.0111
    Epoch [8/50], Test Losses: mse: 16.9787, mae: 2.6406, huber: 2.2185, swd: 8.3675, ept: 134.2238
      Epoch 8 composite train-obj: 2.443859
            No improvement (2.9575), counter 4/5
    Epoch [9/50], Train Losses: mse: 28.4423, mae: 2.8461, huber: 2.4338, swd: 14.4411, ept: 142.6235
    Epoch [9/50], Val Losses: mse: 26.5174, mae: 3.3420, huber: 2.9152, swd: 11.5985, ept: 112.7811
    Epoch [9/50], Test Losses: mse: 16.9939, mae: 2.6306, huber: 2.2096, swd: 8.2780, ept: 133.8098
      Epoch 9 composite train-obj: 2.433788
    Epoch [9/50], Test Losses: mse: 16.7865, mae: 2.6227, huber: 2.2010, swd: 8.0589, ept: 133.3760
    Best round's Test MSE: 16.7865, MAE: 2.6227, SWD: 8.0589
    Best round's Validation MSE: 26.1149, MAE: 3.3370, SWD: 11.4229
    Best round's Test verification MSE : 16.7865, MAE: 2.6227, SWD: 8.0589
    Time taken: 24.55 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 34.9305, mae: 3.2470, huber: 2.8278, swd: 19.3644, ept: 125.5122
    Epoch [1/50], Val Losses: mse: 27.6303, mae: 3.4427, huber: 3.0116, swd: 13.0189, ept: 109.5727
    Epoch [1/50], Test Losses: mse: 17.3114, mae: 2.6802, huber: 2.2563, swd: 8.6899, ept: 132.0479
      Epoch 1 composite train-obj: 2.827796
            Val objective improved inf → 3.0116, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 30.7153, mae: 2.9587, huber: 2.5451, swd: 16.7231, ept: 139.1438
    Epoch [2/50], Val Losses: mse: 26.4173, mae: 3.3512, huber: 2.9231, swd: 11.7783, ept: 111.4901
    Epoch [2/50], Test Losses: mse: 16.9647, mae: 2.6263, huber: 2.2048, swd: 8.4346, ept: 132.9876
      Epoch 2 composite train-obj: 2.545113
            Val objective improved 3.0116 → 2.9231, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 29.9750, mae: 2.9129, huber: 2.4999, swd: 16.1384, ept: 140.6452
    Epoch [3/50], Val Losses: mse: 25.9696, mae: 3.3289, huber: 2.9002, swd: 11.4432, ept: 112.5820
    Epoch [3/50], Test Losses: mse: 16.9081, mae: 2.6291, huber: 2.2072, swd: 8.4298, ept: 133.8521
      Epoch 3 composite train-obj: 2.499905
            Val objective improved 2.9231 → 2.9002, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 29.6135, mae: 2.8903, huber: 2.4774, swd: 15.8765, ept: 141.4077
    Epoch [4/50], Val Losses: mse: 26.1477, mae: 3.3294, huber: 2.9022, swd: 11.4641, ept: 112.3538
    Epoch [4/50], Test Losses: mse: 17.0411, mae: 2.6220, huber: 2.2007, swd: 8.4609, ept: 133.0429
      Epoch 4 composite train-obj: 2.477441
            No improvement (2.9022), counter 1/5
    Epoch [5/50], Train Losses: mse: 29.2374, mae: 2.8728, huber: 2.4603, swd: 15.5597, ept: 141.8271
    Epoch [5/50], Val Losses: mse: 25.7042, mae: 3.3071, huber: 2.8798, swd: 11.1626, ept: 112.8947
    Epoch [5/50], Test Losses: mse: 16.9580, mae: 2.6302, huber: 2.2083, swd: 8.4850, ept: 134.0091
      Epoch 5 composite train-obj: 2.460337
            Val objective improved 2.9002 → 2.8798, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 28.9964, mae: 2.8588, huber: 2.4466, swd: 15.3831, ept: 142.4034
    Epoch [6/50], Val Losses: mse: 26.2900, mae: 3.3337, huber: 2.9071, swd: 11.5925, ept: 112.9577
    Epoch [6/50], Test Losses: mse: 17.1194, mae: 2.6326, huber: 2.2106, swd: 8.6256, ept: 133.8474
      Epoch 6 composite train-obj: 2.446594
            No improvement (2.9071), counter 1/5
    Epoch [7/50], Train Losses: mse: 28.6693, mae: 2.8446, huber: 2.4325, swd: 15.1192, ept: 142.5991
    Epoch [7/50], Val Losses: mse: 26.5295, mae: 3.3507, huber: 2.9245, swd: 12.0539, ept: 112.9202
    Epoch [7/50], Test Losses: mse: 17.1780, mae: 2.6447, huber: 2.2223, swd: 8.6984, ept: 133.8688
      Epoch 7 composite train-obj: 2.432522
            No improvement (2.9245), counter 2/5
    Epoch [8/50], Train Losses: mse: 28.4046, mae: 2.8332, huber: 2.4213, swd: 14.9349, ept: 142.6300
    Epoch [8/50], Val Losses: mse: 26.2009, mae: 3.3415, huber: 2.9139, swd: 11.5295, ept: 112.4565
    Epoch [8/50], Test Losses: mse: 17.1472, mae: 2.6363, huber: 2.2145, swd: 8.5746, ept: 132.8579
      Epoch 8 composite train-obj: 2.421259
            No improvement (2.9139), counter 3/5
    Epoch [9/50], Train Losses: mse: 28.0702, mae: 2.8175, huber: 2.4055, swd: 14.6822, ept: 142.9380
    Epoch [9/50], Val Losses: mse: 26.7282, mae: 3.3787, huber: 2.9509, swd: 12.3272, ept: 112.5511
    Epoch [9/50], Test Losses: mse: 16.9725, mae: 2.6415, huber: 2.2191, swd: 8.5546, ept: 133.5801
      Epoch 9 composite train-obj: 2.405533
            No improvement (2.9509), counter 4/5
    Epoch [10/50], Train Losses: mse: 27.6594, mae: 2.8006, huber: 2.3888, swd: 14.3713, ept: 142.9522
    Epoch [10/50], Val Losses: mse: 26.5171, mae: 3.3560, huber: 2.9287, swd: 11.8227, ept: 112.5381
    Epoch [10/50], Test Losses: mse: 17.1492, mae: 2.6355, huber: 2.2137, swd: 8.5832, ept: 133.2107
      Epoch 10 composite train-obj: 2.388818
    Epoch [10/50], Test Losses: mse: 16.9580, mae: 2.6302, huber: 2.2083, swd: 8.4850, ept: 134.0091
    Best round's Test MSE: 16.9580, MAE: 2.6302, SWD: 8.4850
    Best round's Validation MSE: 25.7042, MAE: 3.3071, SWD: 11.1626
    Best round's Test verification MSE : 16.9580, MAE: 2.6302, SWD: 8.4850
    Time taken: 27.00 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 33.6863, mae: 3.1726, huber: 2.7548, swd: 16.1845, ept: 129.7884
    Epoch [1/50], Val Losses: mse: 27.1153, mae: 3.4011, huber: 2.9712, swd: 10.5675, ept: 110.0977
    Epoch [1/50], Test Losses: mse: 17.0771, mae: 2.6454, huber: 2.2225, swd: 7.2186, ept: 132.0648
      Epoch 1 composite train-obj: 2.754833
            Val objective improved inf → 2.9712, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 30.5865, mae: 2.9508, huber: 2.5373, swd: 14.3117, ept: 139.8561
    Epoch [2/50], Val Losses: mse: 26.0179, mae: 3.3295, huber: 2.9008, swd: 9.8779, ept: 111.3047
    Epoch [2/50], Test Losses: mse: 16.9537, mae: 2.6346, huber: 2.2128, swd: 7.1942, ept: 132.6258
      Epoch 2 composite train-obj: 2.537296
            Val objective improved 2.9712 → 2.9008, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 30.2426, mae: 2.9201, huber: 2.5070, swd: 14.0956, ept: 140.9510
    Epoch [3/50], Val Losses: mse: 26.6083, mae: 3.3513, huber: 2.9239, swd: 10.3636, ept: 111.6809
    Epoch [3/50], Test Losses: mse: 17.0091, mae: 2.6296, huber: 2.2083, swd: 7.3032, ept: 132.6866
      Epoch 3 composite train-obj: 2.506975
            No improvement (2.9239), counter 1/5
    Epoch [4/50], Train Losses: mse: 29.8730, mae: 2.8993, huber: 2.4865, swd: 13.8074, ept: 141.6880
    Epoch [4/50], Val Losses: mse: 25.4598, mae: 3.2906, huber: 2.8625, swd: 9.2611, ept: 112.5874
    Epoch [4/50], Test Losses: mse: 16.8368, mae: 2.6150, huber: 2.1944, swd: 7.0993, ept: 133.1258
      Epoch 4 composite train-obj: 2.486469
            Val objective improved 2.9008 → 2.8625, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 29.5970, mae: 2.8835, huber: 2.4707, swd: 13.5933, ept: 142.0356
    Epoch [5/50], Val Losses: mse: 25.4608, mae: 3.3015, huber: 2.8734, swd: 9.5518, ept: 113.7090
    Epoch [5/50], Test Losses: mse: 16.8901, mae: 2.6333, huber: 2.2113, swd: 7.2590, ept: 134.2432
      Epoch 5 composite train-obj: 2.470681
            No improvement (2.8734), counter 1/5
    Epoch [6/50], Train Losses: mse: 29.0133, mae: 2.8618, huber: 2.4492, swd: 13.1674, ept: 142.6539
    Epoch [6/50], Val Losses: mse: 25.9861, mae: 3.3225, huber: 2.8940, swd: 9.4910, ept: 111.6546
    Epoch [6/50], Test Losses: mse: 17.1906, mae: 2.6282, huber: 2.2074, swd: 7.2791, ept: 132.5001
      Epoch 6 composite train-obj: 2.449155
            No improvement (2.8940), counter 2/5
    Epoch [7/50], Train Losses: mse: 28.5641, mae: 2.8394, huber: 2.4270, swd: 12.8315, ept: 142.9295
    Epoch [7/50], Val Losses: mse: 25.1354, mae: 3.2685, huber: 2.8412, swd: 8.9357, ept: 113.1842
    Epoch [7/50], Test Losses: mse: 17.0758, mae: 2.6318, huber: 2.2103, swd: 7.3095, ept: 134.3058
      Epoch 7 composite train-obj: 2.427013
            Val objective improved 2.8625 → 2.8412, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 27.9374, mae: 2.8168, huber: 2.4045, swd: 12.3883, ept: 143.1263
    Epoch [8/50], Val Losses: mse: 25.8574, mae: 3.3088, huber: 2.8806, swd: 9.3548, ept: 112.6727
    Epoch [8/50], Test Losses: mse: 17.3574, mae: 2.6365, huber: 2.2155, swd: 7.4621, ept: 133.0790
      Epoch 8 composite train-obj: 2.404531
            No improvement (2.8806), counter 1/5
    Epoch [9/50], Train Losses: mse: 27.2574, mae: 2.7883, huber: 2.3763, swd: 11.9050, ept: 143.5409
    Epoch [9/50], Val Losses: mse: 25.4019, mae: 3.2920, huber: 2.8643, swd: 9.1293, ept: 113.4885
    Epoch [9/50], Test Losses: mse: 17.1611, mae: 2.6405, huber: 2.2188, swd: 7.4372, ept: 134.6140
      Epoch 9 composite train-obj: 2.376337
            No improvement (2.8643), counter 2/5
    Epoch [10/50], Train Losses: mse: 26.7963, mae: 2.7631, huber: 2.3515, swd: 11.6051, ept: 143.7792
    Epoch [10/50], Val Losses: mse: 25.5063, mae: 3.2982, huber: 2.8702, swd: 9.0819, ept: 113.6902
    Epoch [10/50], Test Losses: mse: 17.1370, mae: 2.6333, huber: 2.2119, swd: 7.3835, ept: 134.5662
      Epoch 10 composite train-obj: 2.351475
            No improvement (2.8702), counter 3/5
    Epoch [11/50], Train Losses: mse: 26.4138, mae: 2.7428, huber: 2.3314, swd: 11.3054, ept: 144.1188
    Epoch [11/50], Val Losses: mse: 25.3926, mae: 3.2922, huber: 2.8630, swd: 9.0326, ept: 113.9741
    Epoch [11/50], Test Losses: mse: 17.1309, mae: 2.6394, huber: 2.2182, swd: 7.3670, ept: 134.6315
      Epoch 11 composite train-obj: 2.331420
            No improvement (2.8630), counter 4/5
    Epoch [12/50], Train Losses: mse: 26.0707, mae: 2.7250, huber: 2.3138, swd: 11.0910, ept: 144.3814
    Epoch [12/50], Val Losses: mse: 25.1000, mae: 3.2745, huber: 2.8453, swd: 8.7457, ept: 113.8458
    Epoch [12/50], Test Losses: mse: 17.1799, mae: 2.6498, huber: 2.2281, swd: 7.4272, ept: 134.4054
      Epoch 12 composite train-obj: 2.313781
    Epoch [12/50], Test Losses: mse: 17.0758, mae: 2.6318, huber: 2.2103, swd: 7.3095, ept: 134.3058
    Best round's Test MSE: 17.0758, MAE: 2.6318, SWD: 7.3095
    Best round's Validation MSE: 25.1354, MAE: 3.2685, SWD: 8.9357
    Best round's Test verification MSE : 17.0758, MAE: 2.6318, SWD: 7.3095
    Time taken: 31.86 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth2_seq96_pred196_20250511_1616)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 16.9401 ± 0.1187
      mae: 2.6283 ± 0.0040
      huber: 2.2065 ± 0.0040
      swd: 7.9511 ± 0.4859
      ept: 133.8970 ± 0.3878
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 25.6515 ± 0.4016
      mae: 3.3042 ± 0.0280
      huber: 2.8764 ± 0.0274
      swd: 10.5071 ± 1.1162
      ept: 112.8263 ± 0.3238
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 83.48 seconds
    
    Experiment complete: TimeMixer_etth2_seq96_pred196_20250511_1616
    Model: TimeMixer
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.5627, mae: 0.4451, huber: 0.1999, swd: 0.1868, ept: 164.7956
    Epoch [1/50], Val Losses: mse: 0.3901, mae: 0.4419, huber: 0.1806, swd: 0.1437, ept: 153.0341
    Epoch [1/50], Test Losses: mse: 0.2316, mae: 0.3333, huber: 0.1101, swd: 0.0827, ept: 189.0485
      Epoch 1 composite train-obj: 0.199910
            Val objective improved inf → 0.1806, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4859, mae: 0.4042, huber: 0.1737, swd: 0.1841, ept: 197.7654
    Epoch [2/50], Val Losses: mse: 0.3787, mae: 0.4350, huber: 0.1759, swd: 0.1414, ept: 154.3223
    Epoch [2/50], Test Losses: mse: 0.2267, mae: 0.3305, huber: 0.1082, swd: 0.0812, ept: 191.5823
      Epoch 2 composite train-obj: 0.173661
            Val objective improved 0.1806 → 0.1759, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4770, mae: 0.3997, huber: 0.1705, swd: 0.1839, ept: 202.3262
    Epoch [3/50], Val Losses: mse: 0.3770, mae: 0.4330, huber: 0.1748, swd: 0.1481, ept: 154.7524
    Epoch [3/50], Test Losses: mse: 0.2214, mae: 0.3257, huber: 0.1058, swd: 0.0814, ept: 191.3380
      Epoch 3 composite train-obj: 0.170465
            Val objective improved 0.1759 → 0.1748, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4689, mae: 0.3964, huber: 0.1678, swd: 0.1831, ept: 204.5346
    Epoch [4/50], Val Losses: mse: 0.3895, mae: 0.4411, huber: 0.1805, swd: 0.1500, ept: 157.0060
    Epoch [4/50], Test Losses: mse: 0.2280, mae: 0.3331, huber: 0.1091, swd: 0.0836, ept: 192.1222
      Epoch 4 composite train-obj: 0.167799
            No improvement (0.1805), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4628, mae: 0.3940, huber: 0.1660, swd: 0.1812, ept: 205.8683
    Epoch [5/50], Val Losses: mse: 0.3797, mae: 0.4345, huber: 0.1759, swd: 0.1485, ept: 157.3627
    Epoch [5/50], Test Losses: mse: 0.2217, mae: 0.3268, huber: 0.1061, swd: 0.0829, ept: 192.7538
      Epoch 5 composite train-obj: 0.165991
            No improvement (0.1759), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4538, mae: 0.3908, huber: 0.1634, swd: 0.1793, ept: 206.7013
    Epoch [6/50], Val Losses: mse: 0.3872, mae: 0.4382, huber: 0.1788, swd: 0.1513, ept: 157.2049
    Epoch [6/50], Test Losses: mse: 0.2246, mae: 0.3291, huber: 0.1075, swd: 0.0838, ept: 193.3336
      Epoch 6 composite train-obj: 0.163378
            No improvement (0.1788), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.4457, mae: 0.3886, huber: 0.1614, swd: 0.1778, ept: 207.3549
    Epoch [7/50], Val Losses: mse: 0.3899, mae: 0.4400, huber: 0.1800, swd: 0.1512, ept: 157.3405
    Epoch [7/50], Test Losses: mse: 0.2249, mae: 0.3294, huber: 0.1076, swd: 0.0829, ept: 194.1064
      Epoch 7 composite train-obj: 0.161362
            No improvement (0.1800), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.4341, mae: 0.3845, huber: 0.1580, swd: 0.1753, ept: 208.1286
    Epoch [8/50], Val Losses: mse: 0.3964, mae: 0.4439, huber: 0.1827, swd: 0.1503, ept: 157.1217
    Epoch [8/50], Test Losses: mse: 0.2281, mae: 0.3327, huber: 0.1092, swd: 0.0820, ept: 194.1650
      Epoch 8 composite train-obj: 0.158028
    Epoch [8/50], Test Losses: mse: 0.2214, mae: 0.3257, huber: 0.1058, swd: 0.0814, ept: 191.3380
    Best round's Test MSE: 0.2214, MAE: 0.3257, SWD: 0.0814
    Best round's Validation MSE: 0.3770, MAE: 0.4330, SWD: 0.1481
    Best round's Test verification MSE : 0.2214, MAE: 0.3257, SWD: 0.0814
    Time taken: 21.32 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5211, mae: 0.4246, huber: 0.1861, swd: 0.2013, ept: 181.4534
    Epoch [1/50], Val Losses: mse: 0.3907, mae: 0.4414, huber: 0.1807, swd: 0.1507, ept: 153.6619
    Epoch [1/50], Test Losses: mse: 0.2297, mae: 0.3323, huber: 0.1095, swd: 0.0853, ept: 190.5572
      Epoch 1 composite train-obj: 0.186067
            Val objective improved inf → 0.1807, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4814, mae: 0.4010, huber: 0.1716, swd: 0.1932, ept: 202.6448
    Epoch [2/50], Val Losses: mse: 0.3788, mae: 0.4353, huber: 0.1758, swd: 0.1500, ept: 155.6436
    Epoch [2/50], Test Losses: mse: 0.2213, mae: 0.3259, huber: 0.1058, swd: 0.0844, ept: 192.8779
      Epoch 2 composite train-obj: 0.171551
            Val objective improved 0.1807 → 0.1758, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4744, mae: 0.3981, huber: 0.1692, swd: 0.1907, ept: 204.9657
    Epoch [3/50], Val Losses: mse: 0.3802, mae: 0.4344, huber: 0.1758, swd: 0.1584, ept: 156.2171
    Epoch [3/50], Test Losses: mse: 0.2173, mae: 0.3224, huber: 0.1041, swd: 0.0860, ept: 193.7381
      Epoch 3 composite train-obj: 0.169219
            No improvement (0.1758), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4698, mae: 0.3958, huber: 0.1674, swd: 0.1897, ept: 206.6896
    Epoch [4/50], Val Losses: mse: 0.3799, mae: 0.4357, huber: 0.1762, swd: 0.1520, ept: 157.9058
    Epoch [4/50], Test Losses: mse: 0.2184, mae: 0.3246, huber: 0.1047, swd: 0.0851, ept: 195.0107
      Epoch 4 composite train-obj: 0.167442
            No improvement (0.1762), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.4645, mae: 0.3944, huber: 0.1661, swd: 0.1883, ept: 207.1875
    Epoch [5/50], Val Losses: mse: 0.3806, mae: 0.4365, huber: 0.1768, swd: 0.1484, ept: 157.9439
    Epoch [5/50], Test Losses: mse: 0.2211, mae: 0.3274, huber: 0.1061, swd: 0.0844, ept: 194.3138
      Epoch 5 composite train-obj: 0.166083
            No improvement (0.1768), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.4580, mae: 0.3924, huber: 0.1645, swd: 0.1862, ept: 207.5332
    Epoch [6/50], Val Losses: mse: 0.3819, mae: 0.4386, huber: 0.1774, swd: 0.1462, ept: 157.8061
    Epoch [6/50], Test Losses: mse: 0.2219, mae: 0.3292, huber: 0.1066, swd: 0.0846, ept: 194.1530
      Epoch 6 composite train-obj: 0.164460
            No improvement (0.1774), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.4505, mae: 0.3905, huber: 0.1627, swd: 0.1846, ept: 207.4073
    Epoch [7/50], Val Losses: mse: 0.3850, mae: 0.4394, huber: 0.1785, swd: 0.1470, ept: 157.0427
    Epoch [7/50], Test Losses: mse: 0.2200, mae: 0.3270, huber: 0.1058, swd: 0.0829, ept: 193.9496
      Epoch 7 composite train-obj: 0.162746
    Epoch [7/50], Test Losses: mse: 0.2213, mae: 0.3259, huber: 0.1058, swd: 0.0844, ept: 192.8779
    Best round's Test MSE: 0.2213, MAE: 0.3259, SWD: 0.0844
    Best round's Validation MSE: 0.3788, MAE: 0.4353, SWD: 0.1500
    Best round's Test verification MSE : 0.2213, MAE: 0.3259, SWD: 0.0844
    Time taken: 18.19 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5084, mae: 0.4203, huber: 0.1830, swd: 0.1995, ept: 187.4770
    Epoch [1/50], Val Losses: mse: 0.3762, mae: 0.4337, huber: 0.1746, swd: 0.1418, ept: 155.4174
    Epoch [1/50], Test Losses: mse: 0.2241, mae: 0.3278, huber: 0.1069, swd: 0.0781, ept: 192.9550
      Epoch 1 composite train-obj: 0.183006
            Val objective improved inf → 0.1746, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4813, mae: 0.4013, huber: 0.1715, swd: 0.1895, ept: 203.2400
    Epoch [2/50], Val Losses: mse: 0.3831, mae: 0.4378, huber: 0.1775, swd: 0.1447, ept: 154.4727
    Epoch [2/50], Test Losses: mse: 0.2237, mae: 0.3286, huber: 0.1069, swd: 0.0782, ept: 192.7117
      Epoch 2 composite train-obj: 0.171497
            No improvement (0.1775), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4746, mae: 0.3984, huber: 0.1692, swd: 0.1871, ept: 205.3204
    Epoch [3/50], Val Losses: mse: 0.3823, mae: 0.4373, huber: 0.1775, swd: 0.1442, ept: 157.3144
    Epoch [3/50], Test Losses: mse: 0.2244, mae: 0.3292, huber: 0.1073, swd: 0.0788, ept: 195.0395
      Epoch 3 composite train-obj: 0.169244
            No improvement (0.1775), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4696, mae: 0.3961, huber: 0.1676, swd: 0.1854, ept: 206.2686
    Epoch [4/50], Val Losses: mse: 0.3793, mae: 0.4363, huber: 0.1762, swd: 0.1408, ept: 158.1056
    Epoch [4/50], Test Losses: mse: 0.2226, mae: 0.3283, huber: 0.1066, swd: 0.0782, ept: 194.8134
      Epoch 4 composite train-obj: 0.167618
            No improvement (0.1762), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4629, mae: 0.3939, huber: 0.1658, swd: 0.1832, ept: 207.2027
    Epoch [5/50], Val Losses: mse: 0.3852, mae: 0.4395, huber: 0.1788, swd: 0.1421, ept: 157.7068
    Epoch [5/50], Test Losses: mse: 0.2239, mae: 0.3291, huber: 0.1072, swd: 0.0775, ept: 195.0778
      Epoch 5 composite train-obj: 0.165780
            No improvement (0.1788), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.4551, mae: 0.3915, huber: 0.1638, swd: 0.1805, ept: 207.8737
    Epoch [6/50], Val Losses: mse: 0.3841, mae: 0.4375, huber: 0.1779, swd: 0.1435, ept: 159.3128
    Epoch [6/50], Test Losses: mse: 0.2227, mae: 0.3285, huber: 0.1068, swd: 0.0790, ept: 195.3112
      Epoch 6 composite train-obj: 0.163840
    Epoch [6/50], Test Losses: mse: 0.2241, mae: 0.3278, huber: 0.1069, swd: 0.0781, ept: 192.9550
    Best round's Test MSE: 0.2241, MAE: 0.3278, SWD: 0.0781
    Best round's Validation MSE: 0.3762, MAE: 0.4337, SWD: 0.1418
    Best round's Test verification MSE : 0.2241, MAE: 0.3278, SWD: 0.0781
    Time taken: 15.79 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth2_seq96_pred336_20250510_2041)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2222 ± 0.0013
      mae: 0.3265 ± 0.0010
      huber: 0.1062 ± 0.0005
      swd: 0.0813 ± 0.0026
      ept: 192.3903 ± 0.7448
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3773 ± 0.0011
      mae: 0.4340 ± 0.0010
      huber: 0.1751 ± 0.0005
      swd: 0.1466 ± 0.0035
      ept: 155.2711 ± 0.3782
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 55.41 seconds
    
    Experiment complete: TimeMixer_etth2_seq96_pred336_20250510_2041
    Model: TimeMixer
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 11
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 45.7842, mae: 3.7068, huber: 3.2791, swd: 21.9862, ept: 167.1396
    Epoch [1/50], Val Losses: mse: 29.3529, mae: 3.6286, huber: 3.1916, swd: 10.9768, ept: 153.3470
    Epoch [1/50], Test Losses: mse: 18.6702, mae: 2.8062, huber: 2.3793, swd: 8.5320, ept: 189.5178
      Epoch 1 composite train-obj: 3.279075
            Val objective improved inf → 3.1916, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.6517, mae: 3.3535, huber: 2.9324, swd: 20.6486, ept: 198.7579
    Epoch [2/50], Val Losses: mse: 28.2628, mae: 3.5540, huber: 3.1178, swd: 10.0565, ept: 154.5769
    Epoch [2/50], Test Losses: mse: 18.4516, mae: 2.7818, huber: 2.3562, swd: 8.3886, ept: 191.5362
      Epoch 2 composite train-obj: 2.932386
            Val objective improved 3.1916 → 3.1178, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.9488, mae: 3.3074, huber: 2.8869, swd: 20.1744, ept: 202.8363
    Epoch [3/50], Val Losses: mse: 27.9025, mae: 3.5287, huber: 3.0935, swd: 9.6912, ept: 154.8869
    Epoch [3/50], Test Losses: mse: 18.4149, mae: 2.7689, huber: 2.3437, swd: 8.3512, ept: 191.5686
      Epoch 3 composite train-obj: 2.886852
            Val objective improved 3.1178 → 3.0935, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 38.5475, mae: 3.2842, huber: 2.8638, swd: 19.8607, ept: 204.3963
    Epoch [4/50], Val Losses: mse: 28.4766, mae: 3.5645, huber: 3.1289, swd: 10.6464, ept: 157.8977
    Epoch [4/50], Test Losses: mse: 18.5935, mae: 2.7938, huber: 2.3676, swd: 8.6938, ept: 193.3416
      Epoch 4 composite train-obj: 2.863808
            No improvement (3.1289), counter 1/5
    Epoch [5/50], Train Losses: mse: 38.2109, mae: 3.2694, huber: 2.8493, swd: 19.6243, ept: 205.7660
    Epoch [5/50], Val Losses: mse: 28.1009, mae: 3.5379, huber: 3.1027, swd: 10.0261, ept: 158.0936
    Epoch [5/50], Test Losses: mse: 18.3199, mae: 2.7584, huber: 2.3335, swd: 8.3315, ept: 193.7788
      Epoch 5 composite train-obj: 2.849305
            No improvement (3.1027), counter 2/5
    Epoch [6/50], Train Losses: mse: 37.7643, mae: 3.2482, huber: 2.8284, swd: 19.2418, ept: 206.6021
    Epoch [6/50], Val Losses: mse: 28.4809, mae: 3.5667, huber: 3.1304, swd: 10.3055, ept: 158.7549
    Epoch [6/50], Test Losses: mse: 18.5698, mae: 2.7760, huber: 2.3507, swd: 8.5449, ept: 194.1330
      Epoch 6 composite train-obj: 2.828440
            No improvement (3.1304), counter 3/5
    Epoch [7/50], Train Losses: mse: 37.3177, mae: 3.2334, huber: 2.8138, swd: 18.8671, ept: 207.1919
    Epoch [7/50], Val Losses: mse: 28.4175, mae: 3.5612, huber: 3.1260, swd: 10.1029, ept: 158.1606
    Epoch [7/50], Test Losses: mse: 18.5204, mae: 2.7643, huber: 2.3397, swd: 8.5266, ept: 194.5488
      Epoch 7 composite train-obj: 2.813789
            No improvement (3.1260), counter 4/5
    Epoch [8/50], Train Losses: mse: 36.7349, mae: 3.2104, huber: 2.7910, swd: 18.3502, ept: 207.7760
    Epoch [8/50], Val Losses: mse: 28.7184, mae: 3.6011, huber: 3.1641, swd: 10.5106, ept: 157.9542
    Epoch [8/50], Test Losses: mse: 18.7126, mae: 2.7914, huber: 2.3660, swd: 8.6896, ept: 194.6240
      Epoch 8 composite train-obj: 2.790972
    Epoch [8/50], Test Losses: mse: 18.4149, mae: 2.7689, huber: 2.3437, swd: 8.3512, ept: 191.5686
    Best round's Test MSE: 18.4149, MAE: 2.7689, SWD: 8.3512
    Best round's Validation MSE: 27.9025, MAE: 3.5287, SWD: 9.6912
    Best round's Test verification MSE : 18.4149, MAE: 2.7689, SWD: 8.3512
    Time taken: 22.58 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 42.5063, mae: 3.5356, huber: 3.1104, swd: 22.8546, ept: 182.4382
    Epoch [1/50], Val Losses: mse: 29.0686, mae: 3.6045, huber: 3.1677, swd: 11.1061, ept: 152.7421
    Epoch [1/50], Test Losses: mse: 18.5491, mae: 2.7860, huber: 2.3602, swd: 8.7116, ept: 190.0081
      Epoch 1 composite train-obj: 3.110369
            Val objective improved inf → 3.1677, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.3496, mae: 3.3248, huber: 2.9037, swd: 21.4922, ept: 202.4038
    Epoch [2/50], Val Losses: mse: 28.2450, mae: 3.5525, huber: 3.1158, swd: 10.4419, ept: 156.6010
    Epoch [2/50], Test Losses: mse: 18.1341, mae: 2.7480, huber: 2.3234, swd: 8.4266, ept: 193.5033
      Epoch 2 composite train-obj: 2.903682
            Val objective improved 3.1677 → 3.1158, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.9000, mae: 3.2930, huber: 2.8721, swd: 21.1278, ept: 204.9218
    Epoch [3/50], Val Losses: mse: 28.3031, mae: 3.5450, huber: 3.1089, swd: 10.4577, ept: 157.3541
    Epoch [3/50], Test Losses: mse: 18.2094, mae: 2.7459, huber: 2.3216, swd: 8.4506, ept: 194.4759
      Epoch 3 composite train-obj: 2.872123
            Val objective improved 3.1158 → 3.1089, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 38.6242, mae: 3.2760, huber: 2.8552, swd: 20.8913, ept: 205.8440
    Epoch [4/50], Val Losses: mse: 27.9721, mae: 3.5335, huber: 3.0971, swd: 10.3584, ept: 158.5245
    Epoch [4/50], Test Losses: mse: 18.1274, mae: 2.7508, huber: 2.3260, swd: 8.4823, ept: 195.3166
      Epoch 4 composite train-obj: 2.855176
            Val objective improved 3.1089 → 3.0971, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 38.4069, mae: 3.2644, huber: 2.8438, swd: 20.7269, ept: 206.4938
    Epoch [5/50], Val Losses: mse: 28.2130, mae: 3.5549, huber: 3.1186, swd: 10.6377, ept: 158.8538
    Epoch [5/50], Test Losses: mse: 18.1803, mae: 2.7550, huber: 2.3304, swd: 8.5547, ept: 195.5211
      Epoch 5 composite train-obj: 2.843765
            No improvement (3.1186), counter 1/5
    Epoch [6/50], Train Losses: mse: 38.1099, mae: 3.2507, huber: 2.8305, swd: 20.4791, ept: 206.9343
    Epoch [6/50], Val Losses: mse: 27.9900, mae: 3.5504, huber: 3.1137, swd: 10.4483, ept: 159.5283
    Epoch [6/50], Test Losses: mse: 18.2372, mae: 2.7606, huber: 2.3359, swd: 8.5884, ept: 195.0973
      Epoch 6 composite train-obj: 2.830499
            No improvement (3.1137), counter 2/5
    Epoch [7/50], Train Losses: mse: 37.6710, mae: 3.2356, huber: 2.8156, swd: 20.1181, ept: 207.4148
    Epoch [7/50], Val Losses: mse: 27.9878, mae: 3.5475, huber: 3.1114, swd: 10.3219, ept: 159.5310
    Epoch [7/50], Test Losses: mse: 18.1815, mae: 2.7497, huber: 2.3253, swd: 8.5808, ept: 195.5437
      Epoch 7 composite train-obj: 2.815605
            No improvement (3.1114), counter 3/5
    Epoch [8/50], Train Losses: mse: 37.3585, mae: 3.2217, huber: 2.8020, swd: 19.8421, ept: 207.5436
    Epoch [8/50], Val Losses: mse: 28.4977, mae: 3.5823, huber: 3.1455, swd: 10.7308, ept: 158.4841
    Epoch [8/50], Test Losses: mse: 18.4079, mae: 2.7607, huber: 2.3364, swd: 8.6967, ept: 194.0666
      Epoch 8 composite train-obj: 2.801999
            No improvement (3.1455), counter 4/5
    Epoch [9/50], Train Losses: mse: 36.8878, mae: 3.2017, huber: 2.7821, swd: 19.4557, ept: 207.9366
    Epoch [9/50], Val Losses: mse: 28.6561, mae: 3.5930, huber: 3.1568, swd: 11.1400, ept: 158.6809
    Epoch [9/50], Test Losses: mse: 18.9705, mae: 2.8181, huber: 2.3924, swd: 9.3810, ept: 194.8324
      Epoch 9 composite train-obj: 2.782062
    Epoch [9/50], Test Losses: mse: 18.1274, mae: 2.7508, huber: 2.3260, swd: 8.4823, ept: 195.3166
    Best round's Test MSE: 18.1274, MAE: 2.7508, SWD: 8.4823
    Best round's Validation MSE: 27.9721, MAE: 3.5335, SWD: 10.3584
    Best round's Test verification MSE : 18.1274, MAE: 2.7508, SWD: 8.4823
    Time taken: 25.56 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 41.9613, mae: 3.5089, huber: 3.0851, swd: 21.4241, ept: 188.1358
    Epoch [1/50], Val Losses: mse: 28.2339, mae: 3.5608, huber: 3.1241, swd: 9.4487, ept: 154.4647
    Epoch [1/50], Test Losses: mse: 18.3826, mae: 2.7748, huber: 2.3491, swd: 7.7781, ept: 192.6972
      Epoch 1 composite train-obj: 3.085108
            Val objective improved inf → 3.1241, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.3526, mae: 3.3261, huber: 2.9050, swd: 19.9097, ept: 202.9749
    Epoch [2/50], Val Losses: mse: 28.7838, mae: 3.5735, huber: 3.1367, swd: 10.0591, ept: 155.2851
    Epoch [2/50], Test Losses: mse: 18.3516, mae: 2.7597, huber: 2.3346, swd: 7.7852, ept: 192.9551
      Epoch 2 composite train-obj: 2.905006
            No improvement (3.1367), counter 1/5
    Epoch [3/50], Train Losses: mse: 38.8920, mae: 3.2956, huber: 2.8748, swd: 19.5821, ept: 204.9155
    Epoch [3/50], Val Losses: mse: 28.4517, mae: 3.5588, huber: 3.1229, swd: 10.1543, ept: 158.1731
    Epoch [3/50], Test Losses: mse: 18.2909, mae: 2.7636, huber: 2.3386, swd: 7.8755, ept: 194.4545
      Epoch 3 composite train-obj: 2.874756
            Val objective improved 3.1241 → 3.1229, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 38.5190, mae: 3.2782, huber: 2.8575, swd: 19.2888, ept: 205.7625
    Epoch [4/50], Val Losses: mse: 27.9440, mae: 3.5345, huber: 3.0981, swd: 9.7305, ept: 159.1229
    Epoch [4/50], Test Losses: mse: 18.2559, mae: 2.7681, huber: 2.3428, swd: 7.8563, ept: 195.0960
      Epoch 4 composite train-obj: 2.857532
            Val objective improved 3.1229 → 3.0981, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 38.3007, mae: 3.2663, huber: 2.8458, swd: 19.1104, ept: 206.5591
    Epoch [5/50], Val Losses: mse: 28.5265, mae: 3.5622, huber: 3.1264, swd: 10.0585, ept: 158.2920
    Epoch [5/50], Test Losses: mse: 18.2553, mae: 2.7516, huber: 2.3274, swd: 7.7628, ept: 194.7420
      Epoch 5 composite train-obj: 2.845779
            No improvement (3.1264), counter 1/5
    Epoch [6/50], Train Losses: mse: 37.9036, mae: 3.2499, huber: 2.8298, swd: 18.8107, ept: 207.2374
    Epoch [6/50], Val Losses: mse: 28.2414, mae: 3.5458, huber: 3.1106, swd: 9.7902, ept: 159.2062
    Epoch [6/50], Test Losses: mse: 18.3479, mae: 2.7585, huber: 2.3339, swd: 7.8600, ept: 195.3574
      Epoch 6 composite train-obj: 2.829806
            No improvement (3.1106), counter 2/5
    Epoch [7/50], Train Losses: mse: 37.5579, mae: 3.2323, huber: 2.8126, swd: 18.5478, ept: 207.5289
    Epoch [7/50], Val Losses: mse: 27.8892, mae: 3.5322, huber: 3.0969, swd: 9.6082, ept: 159.4684
    Epoch [7/50], Test Losses: mse: 18.5471, mae: 2.7783, huber: 2.3536, swd: 8.1349, ept: 196.1364
      Epoch 7 composite train-obj: 2.812649
            Val objective improved 3.0981 → 3.0969, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 37.1093, mae: 3.2148, huber: 2.7954, swd: 18.1874, ept: 207.9642
    Epoch [8/50], Val Losses: mse: 28.2819, mae: 3.5501, huber: 3.1147, swd: 9.8802, ept: 158.7850
    Epoch [8/50], Test Losses: mse: 18.5517, mae: 2.7777, huber: 2.3526, swd: 8.1031, ept: 195.3104
      Epoch 8 composite train-obj: 2.795367
            No improvement (3.1147), counter 1/5
    Epoch [9/50], Train Losses: mse: 36.5013, mae: 3.1947, huber: 2.7753, swd: 17.7209, ept: 208.0908
    Epoch [9/50], Val Losses: mse: 28.8638, mae: 3.5781, huber: 3.1425, swd: 10.1018, ept: 158.1297
    Epoch [9/50], Test Losses: mse: 18.8671, mae: 2.7866, huber: 2.3615, swd: 8.2855, ept: 194.6107
      Epoch 9 composite train-obj: 2.775335
            No improvement (3.1425), counter 2/5
    Epoch [10/50], Train Losses: mse: 36.0199, mae: 3.1766, huber: 2.7573, swd: 17.3390, ept: 208.3551
    Epoch [10/50], Val Losses: mse: 27.9235, mae: 3.5467, huber: 3.1108, swd: 9.5334, ept: 159.1582
    Epoch [10/50], Test Losses: mse: 18.5102, mae: 2.7844, huber: 2.3595, swd: 8.1069, ept: 194.9389
      Epoch 10 composite train-obj: 2.757309
            No improvement (3.1108), counter 3/5
    Epoch [11/50], Train Losses: mse: 35.4537, mae: 3.1541, huber: 2.7348, swd: 16.9112, ept: 208.9455
    Epoch [11/50], Val Losses: mse: 28.9869, mae: 3.6034, huber: 3.1659, swd: 10.2118, ept: 158.9185
    Epoch [11/50], Test Losses: mse: 18.8770, mae: 2.7969, huber: 2.3716, swd: 8.3156, ept: 194.9933
      Epoch 11 composite train-obj: 2.734789
            No improvement (3.1659), counter 4/5
    Epoch [12/50], Train Losses: mse: 34.5649, mae: 3.1254, huber: 2.7061, swd: 16.1993, ept: 209.6224
    Epoch [12/50], Val Losses: mse: 28.1295, mae: 3.5553, huber: 3.1185, swd: 9.6023, ept: 160.2535
    Epoch [12/50], Test Losses: mse: 19.0150, mae: 2.8190, huber: 2.3933, swd: 8.4797, ept: 195.5136
      Epoch 12 composite train-obj: 2.706146
    Epoch [12/50], Test Losses: mse: 18.5471, mae: 2.7783, huber: 2.3536, swd: 8.1349, ept: 196.1364
    Best round's Test MSE: 18.5471, MAE: 2.7783, SWD: 8.1349
    Best round's Validation MSE: 27.8892, MAE: 3.5322, SWD: 9.6082
    Best round's Test verification MSE : 18.5471, MAE: 2.7783, SWD: 8.1349
    Time taken: 35.71 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth2_seq96_pred336_20250511_1619)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 18.3632 ± 0.1752
      mae: 2.7660 ± 0.0114
      huber: 2.3411 ± 0.0114
      swd: 8.3228 ± 0.1432
      ept: 194.3405 ± 1.9884
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 27.9213 ± 0.0364
      mae: 3.5315 ± 0.0021
      huber: 3.0958 ± 0.0017
      swd: 9.8859 ± 0.3358
      ept: 157.6266 ± 1.9752
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 83.91 seconds
    
    Experiment complete: TimeMixer_etth2_seq96_pred336_20250511_1619
    Model: TimeMixer
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.7587, mae: 0.5348, huber: 0.2640, swd: 0.2907, ept: 264.3375
    Epoch [1/50], Val Losses: mse: 0.4435, mae: 0.4807, huber: 0.2041, swd: 0.1080, ept: 258.9019
    Epoch [1/50], Test Losses: mse: 0.3122, mae: 0.3915, huber: 0.1457, swd: 0.1128, ept: 308.2006
      Epoch 1 composite train-obj: 0.264021
            Val objective improved inf → 0.2041, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6970, mae: 0.5005, huber: 0.2412, swd: 0.2762, ept: 306.0838
    Epoch [2/50], Val Losses: mse: 0.4013, mae: 0.4556, huber: 0.1869, swd: 0.1035, ept: 265.2568
    Epoch [2/50], Test Losses: mse: 0.2860, mae: 0.3726, huber: 0.1346, swd: 0.1025, ept: 310.3760
      Epoch 2 composite train-obj: 0.241244
            Val objective improved 0.2041 → 0.1869, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6872, mae: 0.4964, huber: 0.2379, swd: 0.2767, ept: 313.5471
    Epoch [3/50], Val Losses: mse: 0.4162, mae: 0.4639, huber: 0.1931, swd: 0.1075, ept: 267.2851
    Epoch [3/50], Test Losses: mse: 0.2915, mae: 0.3777, huber: 0.1371, swd: 0.1042, ept: 311.2150
      Epoch 3 composite train-obj: 0.237852
            No improvement (0.1931), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.6787, mae: 0.4931, huber: 0.2352, swd: 0.2748, ept: 315.8973
    Epoch [4/50], Val Losses: mse: 0.3971, mae: 0.4518, huber: 0.1851, swd: 0.1101, ept: 268.4688
    Epoch [4/50], Test Losses: mse: 0.2770, mae: 0.3669, huber: 0.1308, swd: 0.1003, ept: 313.4923
      Epoch 4 composite train-obj: 0.235241
            Val objective improved 0.1869 → 0.1851, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.6707, mae: 0.4904, huber: 0.2330, swd: 0.2720, ept: 317.5426
    Epoch [5/50], Val Losses: mse: 0.3884, mae: 0.4467, huber: 0.1815, swd: 0.0993, ept: 267.5109
    Epoch [5/50], Test Losses: mse: 0.2802, mae: 0.3692, huber: 0.1324, swd: 0.0981, ept: 309.9954
      Epoch 5 composite train-obj: 0.233024
            Val objective improved 0.1851 → 0.1815, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.6614, mae: 0.4872, huber: 0.2303, swd: 0.2693, ept: 318.4159
    Epoch [6/50], Val Losses: mse: 0.4029, mae: 0.4552, huber: 0.1874, swd: 0.0982, ept: 267.8015
    Epoch [6/50], Test Losses: mse: 0.2913, mae: 0.3787, huber: 0.1375, swd: 0.1050, ept: 315.2824
      Epoch 6 composite train-obj: 0.230265
            No improvement (0.1874), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.6488, mae: 0.4827, huber: 0.2267, swd: 0.2669, ept: 319.4953
    Epoch [7/50], Val Losses: mse: 0.3826, mae: 0.4439, huber: 0.1789, swd: 0.0977, ept: 267.7289
    Epoch [7/50], Test Losses: mse: 0.2786, mae: 0.3699, huber: 0.1319, swd: 0.0990, ept: 313.4161
      Epoch 7 composite train-obj: 0.226701
            Val objective improved 0.1815 → 0.1789, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.6386, mae: 0.4791, huber: 0.2234, swd: 0.2649, ept: 320.5243
    Epoch [8/50], Val Losses: mse: 0.3992, mae: 0.4536, huber: 0.1857, swd: 0.1001, ept: 268.3796
    Epoch [8/50], Test Losses: mse: 0.2920, mae: 0.3796, huber: 0.1377, swd: 0.1061, ept: 314.0118
      Epoch 8 composite train-obj: 0.223426
            No improvement (0.1857), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.6309, mae: 0.4758, huber: 0.2209, swd: 0.2639, ept: 321.2745
    Epoch [9/50], Val Losses: mse: 0.3751, mae: 0.4406, huber: 0.1760, swd: 0.0973, ept: 268.3579
    Epoch [9/50], Test Losses: mse: 0.2805, mae: 0.3718, huber: 0.1326, swd: 0.1016, ept: 313.0607
      Epoch 9 composite train-obj: 0.220913
            Val objective improved 0.1789 → 0.1760, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.6236, mae: 0.4726, huber: 0.2185, swd: 0.2627, ept: 322.5379
    Epoch [10/50], Val Losses: mse: 0.3880, mae: 0.4481, huber: 0.1811, swd: 0.1000, ept: 270.4513
    Epoch [10/50], Test Losses: mse: 0.2940, mae: 0.3813, huber: 0.1385, swd: 0.1091, ept: 312.7059
      Epoch 10 composite train-obj: 0.218454
            No improvement (0.1811), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.6173, mae: 0.4694, huber: 0.2161, swd: 0.2615, ept: 323.1725
    Epoch [11/50], Val Losses: mse: 0.3906, mae: 0.4503, huber: 0.1825, swd: 0.0988, ept: 266.8875
    Epoch [11/50], Test Losses: mse: 0.2925, mae: 0.3816, huber: 0.1383, swd: 0.1066, ept: 312.1024
      Epoch 11 composite train-obj: 0.216099
            No improvement (0.1825), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.6101, mae: 0.4665, huber: 0.2137, swd: 0.2598, ept: 323.4306
    Epoch [12/50], Val Losses: mse: 0.3875, mae: 0.4474, huber: 0.1808, swd: 0.0976, ept: 272.1013
    Epoch [12/50], Test Losses: mse: 0.2951, mae: 0.3833, huber: 0.1394, swd: 0.1086, ept: 312.5854
      Epoch 12 composite train-obj: 0.213684
            No improvement (0.1808), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.6068, mae: 0.4643, huber: 0.2122, swd: 0.2587, ept: 324.4918
    Epoch [13/50], Val Losses: mse: 0.3833, mae: 0.4452, huber: 0.1789, swd: 0.1027, ept: 273.0462
    Epoch [13/50], Test Losses: mse: 0.2851, mae: 0.3763, huber: 0.1349, swd: 0.1059, ept: 313.7812
      Epoch 13 composite train-obj: 0.212212
            No improvement (0.1789), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.6012, mae: 0.4616, huber: 0.2101, swd: 0.2575, ept: 325.4851
    Epoch [14/50], Val Losses: mse: 0.3802, mae: 0.4433, huber: 0.1776, swd: 0.1024, ept: 274.6267
    Epoch [14/50], Test Losses: mse: 0.2891, mae: 0.3779, huber: 0.1364, swd: 0.1084, ept: 312.0490
      Epoch 14 composite train-obj: 0.210070
    Epoch [14/50], Test Losses: mse: 0.2805, mae: 0.3718, huber: 0.1326, swd: 0.1016, ept: 313.0607
    Best round's Test MSE: 0.2805, MAE: 0.3718, SWD: 0.1016
    Best round's Validation MSE: 0.3751, MAE: 0.4406, SWD: 0.0973
    Best round's Test verification MSE : 0.2805, MAE: 0.3718, SWD: 0.1016
    Time taken: 43.27 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7307, mae: 0.5174, huber: 0.2528, swd: 0.2743, ept: 283.7228
    Epoch [1/50], Val Losses: mse: 0.4208, mae: 0.4662, huber: 0.1948, swd: 0.0978, ept: 264.5720
    Epoch [1/50], Test Losses: mse: 0.2984, mae: 0.3811, huber: 0.1399, swd: 0.0974, ept: 309.7861
      Epoch 1 composite train-obj: 0.252786
            Val objective improved inf → 0.1948, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6907, mae: 0.4968, huber: 0.2389, swd: 0.2641, ept: 313.3328
    Epoch [2/50], Val Losses: mse: 0.4256, mae: 0.4683, huber: 0.1968, swd: 0.1009, ept: 264.8432
    Epoch [2/50], Test Losses: mse: 0.2984, mae: 0.3820, huber: 0.1401, swd: 0.0975, ept: 311.1869
      Epoch 2 composite train-obj: 0.238852
            No improvement (0.1968), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.6808, mae: 0.4937, huber: 0.2361, swd: 0.2626, ept: 316.6937
    Epoch [3/50], Val Losses: mse: 0.4078, mae: 0.4587, huber: 0.1898, swd: 0.0992, ept: 266.4708
    Epoch [3/50], Test Losses: mse: 0.2837, mae: 0.3725, huber: 0.1338, swd: 0.0931, ept: 313.8944
      Epoch 3 composite train-obj: 0.236088
            Val objective improved 0.1948 → 0.1898, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.6739, mae: 0.4914, huber: 0.2341, swd: 0.2607, ept: 316.7995
    Epoch [4/50], Val Losses: mse: 0.4095, mae: 0.4589, huber: 0.1903, swd: 0.0920, ept: 266.9064
    Epoch [4/50], Test Losses: mse: 0.2905, mae: 0.3775, huber: 0.1370, swd: 0.0929, ept: 313.4117
      Epoch 4 composite train-obj: 0.234091
            No improvement (0.1903), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.6633, mae: 0.4873, huber: 0.2308, swd: 0.2573, ept: 319.0071
    Epoch [5/50], Val Losses: mse: 0.4106, mae: 0.4599, huber: 0.1906, swd: 0.0869, ept: 265.1575
    Epoch [5/50], Test Losses: mse: 0.2969, mae: 0.3832, huber: 0.1401, swd: 0.0951, ept: 313.7448
      Epoch 5 composite train-obj: 0.230844
            No improvement (0.1906), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.6532, mae: 0.4835, huber: 0.2278, swd: 0.2547, ept: 319.7077
    Epoch [6/50], Val Losses: mse: 0.4202, mae: 0.4656, huber: 0.1942, swd: 0.0898, ept: 264.2706
    Epoch [6/50], Test Losses: mse: 0.2998, mae: 0.3857, huber: 0.1413, swd: 0.0955, ept: 313.6494
      Epoch 6 composite train-obj: 0.227771
            No improvement (0.1942), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.6434, mae: 0.4804, huber: 0.2250, swd: 0.2527, ept: 320.1991
    Epoch [7/50], Val Losses: mse: 0.4061, mae: 0.4573, huber: 0.1883, swd: 0.0866, ept: 263.2265
    Epoch [7/50], Test Losses: mse: 0.2989, mae: 0.3848, huber: 0.1411, swd: 0.0956, ept: 312.5256
      Epoch 7 composite train-obj: 0.224992
            Val objective improved 0.1898 → 0.1883, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.6326, mae: 0.4757, huber: 0.2213, swd: 0.2502, ept: 320.9550
    Epoch [8/50], Val Losses: mse: 0.3852, mae: 0.4449, huber: 0.1791, swd: 0.0929, ept: 267.4708
    Epoch [8/50], Test Losses: mse: 0.2876, mae: 0.3746, huber: 0.1354, swd: 0.0954, ept: 312.5439
      Epoch 8 composite train-obj: 0.221293
            Val objective improved 0.1883 → 0.1791, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.6238, mae: 0.4721, huber: 0.2183, swd: 0.2492, ept: 321.7831
    Epoch [9/50], Val Losses: mse: 0.3760, mae: 0.4406, huber: 0.1757, swd: 0.0949, ept: 271.2171
    Epoch [9/50], Test Losses: mse: 0.2794, mae: 0.3700, huber: 0.1319, swd: 0.0934, ept: 314.7594
      Epoch 9 composite train-obj: 0.218337
            Val objective improved 0.1791 → 0.1757, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.6163, mae: 0.4687, huber: 0.2157, swd: 0.2478, ept: 322.5658
    Epoch [10/50], Val Losses: mse: 0.3765, mae: 0.4414, huber: 0.1758, swd: 0.0917, ept: 271.6427
    Epoch [10/50], Test Losses: mse: 0.2853, mae: 0.3740, huber: 0.1343, swd: 0.0956, ept: 313.4348
      Epoch 10 composite train-obj: 0.215666
            No improvement (0.1758), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.6100, mae: 0.4659, huber: 0.2135, swd: 0.2468, ept: 323.6505
    Epoch [11/50], Val Losses: mse: 0.3741, mae: 0.4396, huber: 0.1749, swd: 0.0894, ept: 271.6022
    Epoch [11/50], Test Losses: mse: 0.2886, mae: 0.3768, huber: 0.1361, swd: 0.0960, ept: 312.3730
      Epoch 11 composite train-obj: 0.213505
            Val objective improved 0.1757 → 0.1749, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.6054, mae: 0.4636, huber: 0.2117, swd: 0.2456, ept: 324.2472
    Epoch [12/50], Val Losses: mse: 0.3775, mae: 0.4411, huber: 0.1762, swd: 0.0916, ept: 274.0509
    Epoch [12/50], Test Losses: mse: 0.2932, mae: 0.3805, huber: 0.1382, swd: 0.0996, ept: 312.9784
      Epoch 12 composite train-obj: 0.211732
            No improvement (0.1762), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.5996, mae: 0.4603, huber: 0.2094, swd: 0.2443, ept: 325.4691
    Epoch [13/50], Val Losses: mse: 0.3786, mae: 0.4428, huber: 0.1768, swd: 0.0919, ept: 273.4245
    Epoch [13/50], Test Losses: mse: 0.2925, mae: 0.3813, huber: 0.1382, swd: 0.0970, ept: 314.3687
      Epoch 13 composite train-obj: 0.209448
            No improvement (0.1768), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.5945, mae: 0.4580, huber: 0.2077, swd: 0.2436, ept: 326.6782
    Epoch [14/50], Val Losses: mse: 0.3692, mae: 0.4379, huber: 0.1729, swd: 0.0973, ept: 271.8771
    Epoch [14/50], Test Losses: mse: 0.2872, mae: 0.3764, huber: 0.1351, swd: 0.0997, ept: 315.5189
      Epoch 14 composite train-obj: 0.207660
            Val objective improved 0.1749 → 0.1729, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.5913, mae: 0.4564, huber: 0.2065, swd: 0.2424, ept: 327.2344
    Epoch [15/50], Val Losses: mse: 0.3845, mae: 0.4465, huber: 0.1793, swd: 0.0963, ept: 274.0256
    Epoch [15/50], Test Losses: mse: 0.2972, mae: 0.3842, huber: 0.1399, swd: 0.1009, ept: 313.1956
      Epoch 15 composite train-obj: 0.206462
            No improvement (0.1793), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.5894, mae: 0.4552, huber: 0.2057, swd: 0.2415, ept: 327.3440
    Epoch [16/50], Val Losses: mse: 0.3822, mae: 0.4464, huber: 0.1786, swd: 0.0960, ept: 273.5381
    Epoch [16/50], Test Losses: mse: 0.2959, mae: 0.3840, huber: 0.1394, swd: 0.0992, ept: 313.8396
      Epoch 16 composite train-obj: 0.205682
            No improvement (0.1786), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.5848, mae: 0.4526, huber: 0.2038, swd: 0.2399, ept: 328.4399
    Epoch [17/50], Val Losses: mse: 0.3865, mae: 0.4486, huber: 0.1805, swd: 0.0948, ept: 273.9427
    Epoch [17/50], Test Losses: mse: 0.3013, mae: 0.3873, huber: 0.1418, swd: 0.0997, ept: 312.9655
      Epoch 17 composite train-obj: 0.203846
            No improvement (0.1805), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.5825, mae: 0.4515, huber: 0.2030, swd: 0.2393, ept: 328.9305
    Epoch [18/50], Val Losses: mse: 0.3896, mae: 0.4517, huber: 0.1820, swd: 0.0955, ept: 271.0121
    Epoch [18/50], Test Losses: mse: 0.3076, mae: 0.3925, huber: 0.1449, swd: 0.1014, ept: 313.2062
      Epoch 18 composite train-obj: 0.203034
            No improvement (0.1820), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.5792, mae: 0.4500, huber: 0.2019, swd: 0.2384, ept: 329.7776
    Epoch [19/50], Val Losses: mse: 0.3844, mae: 0.4480, huber: 0.1797, swd: 0.0987, ept: 273.8582
    Epoch [19/50], Test Losses: mse: 0.2988, mae: 0.3860, huber: 0.1408, swd: 0.0992, ept: 313.9808
      Epoch 19 composite train-obj: 0.201875
    Epoch [19/50], Test Losses: mse: 0.2872, mae: 0.3764, huber: 0.1351, swd: 0.0997, ept: 315.5189
    Best round's Test MSE: 0.2872, MAE: 0.3764, SWD: 0.0997
    Best round's Validation MSE: 0.3692, MAE: 0.4379, SWD: 0.0973
    Best round's Test verification MSE : 0.2872, MAE: 0.3764, SWD: 0.0997
    Time taken: 59.19 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7231, mae: 0.5163, huber: 0.2517, swd: 0.2980, ept: 283.7268
    Epoch [1/50], Val Losses: mse: 0.4107, mae: 0.4634, huber: 0.1912, swd: 0.0988, ept: 263.7835
    Epoch [1/50], Test Losses: mse: 0.2934, mae: 0.3780, huber: 0.1375, swd: 0.1080, ept: 309.8615
      Epoch 1 composite train-obj: 0.251749
            Val objective improved inf → 0.1912, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6905, mae: 0.4969, huber: 0.2389, swd: 0.2917, ept: 315.1264
    Epoch [2/50], Val Losses: mse: 0.4015, mae: 0.4554, huber: 0.1871, swd: 0.1059, ept: 266.0507
    Epoch [2/50], Test Losses: mse: 0.2810, mae: 0.3695, huber: 0.1324, swd: 0.1030, ept: 315.9387
      Epoch 2 composite train-obj: 0.238901
            Val objective improved 0.1912 → 0.1871, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6805, mae: 0.4933, huber: 0.2357, swd: 0.2891, ept: 319.1486
    Epoch [3/50], Val Losses: mse: 0.4153, mae: 0.4656, huber: 0.1931, swd: 0.0974, ept: 269.2884
    Epoch [3/50], Test Losses: mse: 0.2952, mae: 0.3823, huber: 0.1391, swd: 0.1059, ept: 313.9652
      Epoch 3 composite train-obj: 0.235695
            No improvement (0.1931), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.6699, mae: 0.4903, huber: 0.2328, swd: 0.2865, ept: 320.1930
    Epoch [4/50], Val Losses: mse: 0.4078, mae: 0.4598, huber: 0.1899, swd: 0.1025, ept: 268.8633
    Epoch [4/50], Test Losses: mse: 0.2836, mae: 0.3751, huber: 0.1345, swd: 0.1007, ept: 315.0869
      Epoch 4 composite train-obj: 0.232833
            No improvement (0.1899), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.6626, mae: 0.4879, huber: 0.2308, swd: 0.2844, ept: 320.2673
    Epoch [5/50], Val Losses: mse: 0.4102, mae: 0.4605, huber: 0.1906, swd: 0.1045, ept: 269.2835
    Epoch [5/50], Test Losses: mse: 0.2835, mae: 0.3753, huber: 0.1344, swd: 0.1012, ept: 315.8842
      Epoch 5 composite train-obj: 0.230824
            No improvement (0.1906), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.6561, mae: 0.4862, huber: 0.2292, swd: 0.2833, ept: 320.3493
    Epoch [6/50], Val Losses: mse: 0.4156, mae: 0.4633, huber: 0.1927, swd: 0.1013, ept: 268.5642
    Epoch [6/50], Test Losses: mse: 0.2860, mae: 0.3773, huber: 0.1357, swd: 0.1003, ept: 312.8801
      Epoch 6 composite train-obj: 0.229248
            No improvement (0.1927), counter 4/5
    Epoch [7/50], Train Losses: mse: 0.6488, mae: 0.4831, huber: 0.2269, swd: 0.2811, ept: 320.6052
    Epoch [7/50], Val Losses: mse: 0.4119, mae: 0.4607, huber: 0.1910, swd: 0.1024, ept: 269.0343
    Epoch [7/50], Test Losses: mse: 0.2811, mae: 0.3729, huber: 0.1334, swd: 0.0997, ept: 313.7732
      Epoch 7 composite train-obj: 0.226892
    Epoch [7/50], Test Losses: mse: 0.2810, mae: 0.3695, huber: 0.1324, swd: 0.1030, ept: 315.9387
    Best round's Test MSE: 0.2810, MAE: 0.3695, SWD: 0.1030
    Best round's Validation MSE: 0.4015, MAE: 0.4554, SWD: 0.1059
    Best round's Test verification MSE : 0.2810, MAE: 0.3695, SWD: 0.1030
    Time taken: 23.19 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth2_seq96_pred720_20250510_2042)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2829 ± 0.0031
      mae: 0.3726 ± 0.0029
      huber: 0.1334 ± 0.0012
      swd: 0.1014 ± 0.0013
      ept: 314.8394 ± 1.2694
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3819 ± 0.0141
      mae: 0.4446 ± 0.0077
      huber: 0.1787 ± 0.0061
      swd: 0.1001 ± 0.0041
      ept: 268.7619 ± 2.3957
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 125.69 seconds
    
    Experiment complete: TimeMixer_etth2_seq96_pred720_20250510_2042
    Model: TimeMixer
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
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
    
    Epoch [1/50], Train Losses: mse: 66.7969, mae: 4.5232, huber: 4.0836, swd: 36.3036, ept: 266.0532
    Epoch [1/50], Val Losses: mse: 32.3680, mae: 3.8665, huber: 3.4272, swd: 14.5852, ept: 261.9203
    Epoch [1/50], Test Losses: mse: 24.4900, mae: 3.2364, huber: 2.8024, swd: 12.3428, ept: 306.8206
      Epoch 1 composite train-obj: 4.083564
            Val objective improved inf → 3.4272, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 61.0834, mae: 4.2144, huber: 3.7791, swd: 33.0900, ept: 306.1841
    Epoch [2/50], Val Losses: mse: 29.3728, mae: 3.6968, huber: 3.2596, swd: 11.5529, ept: 266.5697
    Epoch [2/50], Test Losses: mse: 23.7163, mae: 3.1610, huber: 2.7283, swd: 11.6478, ept: 310.6429
      Epoch 2 composite train-obj: 3.779131
            Val objective improved 3.4272 → 3.2596, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 60.2019, mae: 4.1695, huber: 3.7342, swd: 32.4585, ept: 313.9173
    Epoch [3/50], Val Losses: mse: 30.3304, mae: 3.7475, huber: 3.3097, swd: 12.7767, ept: 268.7270
    Epoch [3/50], Test Losses: mse: 24.1070, mae: 3.1974, huber: 2.7641, swd: 12.0676, ept: 312.1073
      Epoch 3 composite train-obj: 3.734238
            No improvement (3.3097), counter 1/5
    Epoch [4/50], Train Losses: mse: 59.5625, mae: 4.1424, huber: 3.7071, swd: 32.0029, ept: 316.8078
    Epoch [4/50], Val Losses: mse: 29.1753, mae: 3.6751, huber: 3.2369, swd: 11.6412, ept: 271.9690
    Epoch [4/50], Test Losses: mse: 23.1420, mae: 3.1181, huber: 2.6863, swd: 11.1489, ept: 314.6329
      Epoch 4 composite train-obj: 3.707118
            Val objective improved 3.2596 → 3.2369, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 58.9343, mae: 4.1213, huber: 3.6861, swd: 31.5277, ept: 318.3052
    Epoch [5/50], Val Losses: mse: 27.7794, mae: 3.5964, huber: 3.1577, swd: 10.2825, ept: 271.6853
    Epoch [5/50], Test Losses: mse: 23.2270, mae: 3.1119, huber: 2.6813, swd: 11.0798, ept: 310.9248
      Epoch 5 composite train-obj: 3.686066
            Val objective improved 3.2369 → 3.1577, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 58.3583, mae: 4.0992, huber: 3.6641, swd: 31.0509, ept: 319.2640
    Epoch [6/50], Val Losses: mse: 29.2817, mae: 3.6870, huber: 3.2491, swd: 12.1079, ept: 272.5786
    Epoch [6/50], Test Losses: mse: 24.0679, mae: 3.1826, huber: 2.7501, swd: 12.2308, ept: 316.5352
      Epoch 6 composite train-obj: 3.664051
            No improvement (3.2491), counter 1/5
    Epoch [7/50], Train Losses: mse: 57.6454, mae: 4.0737, huber: 3.6387, swd: 30.4764, ept: 319.6856
    Epoch [7/50], Val Losses: mse: 28.4597, mae: 3.6370, huber: 3.1985, swd: 11.3693, ept: 273.1394
    Epoch [7/50], Test Losses: mse: 23.5919, mae: 3.1441, huber: 2.7125, swd: 11.7248, ept: 315.9260
      Epoch 7 composite train-obj: 3.638706
            No improvement (3.1985), counter 2/5
    Epoch [8/50], Train Losses: mse: 57.1108, mae: 4.0524, huber: 3.6173, swd: 30.0394, ept: 320.6057
    Epoch [8/50], Val Losses: mse: 29.5615, mae: 3.6944, huber: 3.2557, swd: 12.4619, ept: 272.4756
    Epoch [8/50], Test Losses: mse: 24.1921, mae: 3.1920, huber: 2.7594, swd: 12.2850, ept: 315.1868
      Epoch 8 composite train-obj: 3.617270
            No improvement (3.2557), counter 3/5
    Epoch [9/50], Train Losses: mse: 56.3952, mae: 4.0270, huber: 3.5919, swd: 29.4929, ept: 320.7185
    Epoch [9/50], Val Losses: mse: 27.7353, mae: 3.5858, huber: 3.1475, swd: 10.5165, ept: 270.6687
    Epoch [9/50], Test Losses: mse: 23.3555, mae: 3.1218, huber: 2.6900, swd: 11.3892, ept: 314.6204
      Epoch 9 composite train-obj: 3.591875
            Val objective improved 3.1577 → 3.1475, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 55.9413, mae: 4.0023, huber: 3.5676, swd: 29.1291, ept: 321.2801
    Epoch [10/50], Val Losses: mse: 28.6205, mae: 3.6392, huber: 3.2010, swd: 11.3627, ept: 271.4088
    Epoch [10/50], Test Losses: mse: 24.0182, mae: 3.1792, huber: 2.7463, swd: 11.9952, ept: 315.0118
      Epoch 10 composite train-obj: 3.567636
            No improvement (3.2010), counter 1/5
    Epoch [11/50], Train Losses: mse: 55.1922, mae: 3.9694, huber: 3.5348, swd: 28.4753, ept: 322.4674
    Epoch [11/50], Val Losses: mse: 29.3378, mae: 3.6910, huber: 3.2524, swd: 11.9183, ept: 268.7662
    Epoch [11/50], Test Losses: mse: 24.6467, mae: 3.2163, huber: 2.7831, swd: 12.4314, ept: 313.4210
      Epoch 11 composite train-obj: 3.534778
            No improvement (3.2524), counter 2/5
    Epoch [12/50], Train Losses: mse: 54.7314, mae: 3.9500, huber: 3.5155, swd: 28.1013, ept: 322.6845
    Epoch [12/50], Val Losses: mse: 27.8439, mae: 3.5807, huber: 3.1430, swd: 10.4260, ept: 272.9082
    Epoch [12/50], Test Losses: mse: 23.9637, mae: 3.1662, huber: 2.7336, swd: 11.9851, ept: 316.5968
      Epoch 12 composite train-obj: 3.515456
            Val objective improved 3.1475 → 3.1430, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 54.2529, mae: 3.9284, huber: 3.4940, swd: 27.6772, ept: 323.4360
    Epoch [13/50], Val Losses: mse: 28.3064, mae: 3.6146, huber: 3.1760, swd: 10.7254, ept: 271.7103
    Epoch [13/50], Test Losses: mse: 23.4986, mae: 3.1442, huber: 2.7115, swd: 11.3453, ept: 315.6917
      Epoch 13 composite train-obj: 3.493986
            No improvement (3.1760), counter 1/5
    Epoch [14/50], Train Losses: mse: 53.8218, mae: 3.9045, huber: 3.4704, swd: 27.2912, ept: 324.0686
    Epoch [14/50], Val Losses: mse: 28.6799, mae: 3.6457, huber: 3.2067, swd: 11.2297, ept: 272.4007
    Epoch [14/50], Test Losses: mse: 23.6705, mae: 3.1612, huber: 2.7280, swd: 11.6179, ept: 316.5864
      Epoch 14 composite train-obj: 3.470363
            No improvement (3.2067), counter 2/5
    Epoch [15/50], Train Losses: mse: 53.3423, mae: 3.8785, huber: 3.4446, swd: 26.8873, ept: 324.9649
    Epoch [15/50], Val Losses: mse: 28.2264, mae: 3.6033, huber: 3.1630, swd: 10.3258, ept: 274.3121
    Epoch [15/50], Test Losses: mse: 23.3292, mae: 3.1287, huber: 2.6952, swd: 11.0599, ept: 315.3899
      Epoch 15 composite train-obj: 3.444567
            No improvement (3.1630), counter 3/5
    Epoch [16/50], Train Losses: mse: 52.9613, mae: 3.8612, huber: 3.4273, swd: 26.5542, ept: 325.5423
    Epoch [16/50], Val Losses: mse: 29.3020, mae: 3.6825, huber: 3.2425, swd: 11.6940, ept: 271.8083
    Epoch [16/50], Test Losses: mse: 24.2997, mae: 3.2094, huber: 2.7754, swd: 12.0912, ept: 316.8316
      Epoch 16 composite train-obj: 3.427326
            No improvement (3.2425), counter 4/5
    Epoch [17/50], Train Losses: mse: 52.6324, mae: 3.8404, huber: 3.4068, swd: 26.2839, ept: 325.9927
    Epoch [17/50], Val Losses: mse: 28.8101, mae: 3.6559, huber: 3.2157, swd: 11.2179, ept: 270.7156
    Epoch [17/50], Test Losses: mse: 23.9443, mae: 3.1863, huber: 2.7521, swd: 11.8083, ept: 315.3744
      Epoch 17 composite train-obj: 3.406791
    Epoch [17/50], Test Losses: mse: 23.9637, mae: 3.1662, huber: 2.7336, swd: 11.9851, ept: 316.5968
    Best round's Test MSE: 23.9637, MAE: 3.1662, SWD: 11.9851
    Best round's Validation MSE: 27.8439, MAE: 3.5807, SWD: 10.4260
    Best round's Test verification MSE : 23.9637, MAE: 3.1662, SWD: 11.9851
    Time taken: 55.26 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 63.7849, mae: 4.3566, huber: 3.9194, swd: 31.5619, ept: 287.9244
    Epoch [1/50], Val Losses: mse: 30.1916, mae: 3.7363, huber: 3.2999, swd: 11.5612, ept: 267.3700
    Epoch [1/50], Test Losses: mse: 23.6630, mae: 3.1599, huber: 2.7273, swd: 10.7296, ept: 313.0497
      Epoch 1 composite train-obj: 3.919433
            Val objective improved inf → 3.2999, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.6028, mae: 4.1757, huber: 3.7412, swd: 30.1838, ept: 314.7864
    Epoch [2/50], Val Losses: mse: 30.3052, mae: 3.7363, huber: 3.2996, swd: 11.7432, ept: 268.0595
    Epoch [2/50], Test Losses: mse: 24.0368, mae: 3.1826, huber: 2.7499, swd: 11.0587, ept: 314.9891
      Epoch 2 composite train-obj: 3.741203
            Val objective improved 3.2999 → 3.2996, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.7902, mae: 4.1423, huber: 3.7078, swd: 29.6038, ept: 318.1237
    Epoch [3/50], Val Losses: mse: 29.4247, mae: 3.6941, huber: 3.2565, swd: 11.0803, ept: 270.1570
    Epoch [3/50], Test Losses: mse: 23.7957, mae: 3.1666, huber: 2.7337, swd: 10.8723, ept: 314.9175
      Epoch 3 composite train-obj: 3.707828
            Val objective improved 3.2996 → 3.2565, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 59.0731, mae: 4.1137, huber: 3.6793, swd: 29.0604, ept: 318.5308
    Epoch [4/50], Val Losses: mse: 28.2686, mae: 3.6214, huber: 3.1844, swd: 9.9472, ept: 272.4123
    Epoch [4/50], Test Losses: mse: 23.6399, mae: 3.1459, huber: 2.7142, swd: 10.5440, ept: 314.1588
      Epoch 4 composite train-obj: 3.679285
            Val objective improved 3.2565 → 3.1844, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 58.2943, mae: 4.0862, huber: 3.6518, swd: 28.4968, ept: 319.4501
    Epoch [5/50], Val Losses: mse: 28.1148, mae: 3.6076, huber: 3.1710, swd: 9.9418, ept: 272.5466
    Epoch [5/50], Test Losses: mse: 23.7494, mae: 3.1564, huber: 2.7241, swd: 10.7924, ept: 315.1236
      Epoch 5 composite train-obj: 3.651833
            Val objective improved 3.1844 → 3.1710, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 57.5590, mae: 4.0585, huber: 3.6241, swd: 27.9484, ept: 319.7900
    Epoch [6/50], Val Losses: mse: 29.3579, mae: 3.6827, huber: 3.2447, swd: 11.0211, ept: 271.5023
    Epoch [6/50], Test Losses: mse: 24.2014, mae: 3.1961, huber: 2.7627, swd: 11.1298, ept: 315.8075
      Epoch 6 composite train-obj: 3.624061
            No improvement (3.2447), counter 1/5
    Epoch [7/50], Train Losses: mse: 56.9747, mae: 4.0383, huber: 3.6038, swd: 27.5805, ept: 320.4763
    Epoch [7/50], Val Losses: mse: 29.9145, mae: 3.7139, huber: 3.2750, swd: 11.5086, ept: 269.5559
    Epoch [7/50], Test Losses: mse: 25.1292, mae: 3.2655, huber: 2.8313, swd: 11.7287, ept: 313.9255
      Epoch 7 composite train-obj: 3.603783
            No improvement (3.2750), counter 2/5
    Epoch [8/50], Train Losses: mse: 56.3112, mae: 4.0088, huber: 3.5744, swd: 27.0624, ept: 320.8419
    Epoch [8/50], Val Losses: mse: 28.4450, mae: 3.6122, huber: 3.1743, swd: 10.0388, ept: 271.1596
    Epoch [8/50], Test Losses: mse: 24.1857, mae: 3.1863, huber: 2.7525, swd: 11.0605, ept: 314.8180
      Epoch 8 composite train-obj: 3.574394
            No improvement (3.1743), counter 3/5
    Epoch [9/50], Train Losses: mse: 55.6473, mae: 3.9837, huber: 3.5493, swd: 26.5660, ept: 321.4628
    Epoch [9/50], Val Losses: mse: 28.5197, mae: 3.6131, huber: 3.1748, swd: 10.0471, ept: 271.1751
    Epoch [9/50], Test Losses: mse: 23.5509, mae: 3.1518, huber: 2.7172, swd: 10.3977, ept: 315.0746
      Epoch 9 composite train-obj: 3.549311
            No improvement (3.1748), counter 4/5
    Epoch [10/50], Train Losses: mse: 55.0369, mae: 3.9590, huber: 3.5246, swd: 26.0680, ept: 322.3114
    Epoch [10/50], Val Losses: mse: 27.7015, mae: 3.5742, huber: 3.1359, swd: 9.2744, ept: 272.8466
    Epoch [10/50], Test Losses: mse: 23.4532, mae: 3.1396, huber: 2.7056, swd: 10.3328, ept: 315.1433
      Epoch 10 composite train-obj: 3.524631
            Val objective improved 3.1710 → 3.1359, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 54.3629, mae: 3.9313, huber: 3.4971, swd: 25.5331, ept: 323.1879
    Epoch [11/50], Val Losses: mse: 28.4488, mae: 3.6216, huber: 3.1824, swd: 9.9683, ept: 270.5751
    Epoch [11/50], Test Losses: mse: 24.2247, mae: 3.1965, huber: 2.7625, swd: 10.9081, ept: 313.0451
      Epoch 11 composite train-obj: 3.497121
            No improvement (3.1824), counter 1/5
    Epoch [12/50], Train Losses: mse: 53.7492, mae: 3.9033, huber: 3.4692, swd: 25.0546, ept: 323.8831
    Epoch [12/50], Val Losses: mse: 28.6547, mae: 3.6251, huber: 3.1859, swd: 10.1038, ept: 273.0057
    Epoch [12/50], Test Losses: mse: 24.2496, mae: 3.2021, huber: 2.7673, swd: 11.0567, ept: 315.7741
      Epoch 12 composite train-obj: 3.469193
            No improvement (3.1859), counter 2/5
    Epoch [13/50], Train Losses: mse: 53.1786, mae: 3.8726, huber: 3.4386, swd: 24.6114, ept: 324.6256
    Epoch [13/50], Val Losses: mse: 28.2323, mae: 3.6168, huber: 3.1775, swd: 9.7883, ept: 273.1521
    Epoch [13/50], Test Losses: mse: 24.5665, mae: 3.2224, huber: 2.7877, swd: 11.2601, ept: 315.8700
      Epoch 13 composite train-obj: 3.438613
            No improvement (3.1775), counter 3/5
    Epoch [14/50], Train Losses: mse: 52.7901, mae: 3.8545, huber: 3.4206, swd: 24.3070, ept: 325.0882
    Epoch [14/50], Val Losses: mse: 28.3565, mae: 3.6061, huber: 3.1660, swd: 9.3564, ept: 274.0421
    Epoch [14/50], Test Losses: mse: 23.8782, mae: 3.1651, huber: 2.7305, swd: 10.6137, ept: 316.9366
      Epoch 14 composite train-obj: 3.420552
            No improvement (3.1660), counter 4/5
    Epoch [15/50], Train Losses: mse: 52.3487, mae: 3.8337, huber: 3.3998, swd: 23.9811, ept: 325.8301
    Epoch [15/50], Val Losses: mse: 28.5408, mae: 3.6364, huber: 3.1956, swd: 9.9661, ept: 273.0826
    Epoch [15/50], Test Losses: mse: 24.4593, mae: 3.2172, huber: 2.7824, swd: 11.1347, ept: 316.0957
      Epoch 15 composite train-obj: 3.399802
    Epoch [15/50], Test Losses: mse: 23.4532, mae: 3.1396, huber: 2.7056, swd: 10.3328, ept: 315.1433
    Best round's Test MSE: 23.4532, MAE: 3.1396, SWD: 10.3328
    Best round's Validation MSE: 27.7015, MAE: 3.5742, SWD: 9.2744
    Best round's Test verification MSE : 23.4532, MAE: 3.1396, SWD: 10.3328
    Time taken: 49.03 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 63.7245, mae: 4.3609, huber: 3.9240, swd: 36.3206, ept: 285.4214
    Epoch [1/50], Val Losses: mse: 30.8187, mae: 3.8011, huber: 3.3616, swd: 12.9867, ept: 264.9198
    Epoch [1/50], Test Losses: mse: 23.9158, mae: 3.2018, huber: 2.7679, swd: 12.2155, ept: 310.2214
      Epoch 1 composite train-obj: 3.924004
            Val objective improved inf → 3.3616, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.6231, mae: 4.1816, huber: 3.7468, swd: 34.7193, ept: 314.4502
    Epoch [2/50], Val Losses: mse: 28.9201, mae: 3.6651, huber: 3.2285, swd: 11.1003, ept: 267.8326
    Epoch [2/50], Test Losses: mse: 23.1700, mae: 3.1226, huber: 2.6906, swd: 11.5056, ept: 316.0625
      Epoch 2 composite train-obj: 3.746848
            Val objective improved 3.3616 → 3.2285, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.8417, mae: 4.1469, huber: 3.7122, swd: 34.0965, ept: 318.2572
    Epoch [3/50], Val Losses: mse: 29.0742, mae: 3.6870, huber: 3.2496, swd: 11.6325, ept: 269.4133
    Epoch [3/50], Test Losses: mse: 23.7235, mae: 3.1707, huber: 2.7378, swd: 12.0935, ept: 313.3785
      Epoch 3 composite train-obj: 3.712164
            No improvement (3.2496), counter 1/5
    Epoch [4/50], Train Losses: mse: 59.1963, mae: 4.1187, huber: 3.6842, swd: 33.5330, ept: 319.5835
    Epoch [4/50], Val Losses: mse: 28.2022, mae: 3.6214, huber: 3.1845, swd: 10.6326, ept: 272.6994
    Epoch [4/50], Test Losses: mse: 23.3575, mae: 3.1316, huber: 2.6992, swd: 11.6921, ept: 314.6012
      Epoch 4 composite train-obj: 3.684200
            Val objective improved 3.2285 → 3.1845, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 58.7527, mae: 4.1025, huber: 3.6681, swd: 33.1749, ept: 319.4373
    Epoch [5/50], Val Losses: mse: 28.9851, mae: 3.6678, huber: 3.2310, swd: 11.5018, ept: 272.6648
    Epoch [5/50], Test Losses: mse: 23.6914, mae: 3.1606, huber: 2.7278, swd: 12.0648, ept: 315.1558
      Epoch 5 composite train-obj: 3.668120
            No improvement (3.2310), counter 1/5
    Epoch [6/50], Train Losses: mse: 58.3388, mae: 4.0875, huber: 3.6532, swd: 32.8417, ept: 320.6607
    Epoch [6/50], Val Losses: mse: 28.9795, mae: 3.6596, huber: 3.2234, swd: 11.5426, ept: 270.6771
    Epoch [6/50], Test Losses: mse: 23.7990, mae: 3.1645, huber: 2.7322, swd: 12.1249, ept: 313.7775
      Epoch 6 composite train-obj: 3.653169
            No improvement (3.2234), counter 2/5
    Epoch [7/50], Train Losses: mse: 57.7194, mae: 4.0629, huber: 3.6286, swd: 32.2727, ept: 321.0191
    Epoch [7/50], Val Losses: mse: 29.1064, mae: 3.6561, huber: 3.2200, swd: 11.5603, ept: 270.3885
    Epoch [7/50], Test Losses: mse: 23.5597, mae: 3.1430, huber: 2.7111, swd: 11.8857, ept: 314.0407
      Epoch 7 composite train-obj: 3.628647
            No improvement (3.2200), counter 3/5
    Epoch [8/50], Train Losses: mse: 57.0257, mae: 4.0370, huber: 3.6028, swd: 31.6519, ept: 320.7209
    Epoch [8/50], Val Losses: mse: 30.5609, mae: 3.7334, huber: 3.2964, swd: 12.8644, ept: 268.0114
    Epoch [8/50], Test Losses: mse: 24.5381, mae: 3.2118, huber: 2.7790, swd: 12.6875, ept: 312.8979
      Epoch 8 composite train-obj: 3.602763
            No improvement (3.2964), counter 4/5
    Epoch [9/50], Train Losses: mse: 56.2895, mae: 4.0096, huber: 3.5753, swd: 31.0102, ept: 321.3023
    Epoch [9/50], Val Losses: mse: 29.1113, mae: 3.6606, huber: 3.2241, swd: 11.5224, ept: 268.4292
    Epoch [9/50], Test Losses: mse: 23.8240, mae: 3.1593, huber: 2.7271, swd: 12.1372, ept: 315.1551
      Epoch 9 composite train-obj: 3.575281
    Epoch [9/50], Test Losses: mse: 23.3575, mae: 3.1316, huber: 2.6992, swd: 11.6921, ept: 314.6012
    Best round's Test MSE: 23.3575, MAE: 3.1316, SWD: 11.6921
    Best round's Validation MSE: 28.2022, MAE: 3.6214, SWD: 10.6326
    Best round's Test verification MSE : 23.3575, MAE: 3.1316, SWD: 11.6921
    Time taken: 29.77 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_etth2_seq96_pred720_20250511_1621)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 23.5915 ± 0.2661
      mae: 3.1458 ± 0.0148
      huber: 2.7128 ± 0.0150
      swd: 11.3367 ± 0.7199
      ept: 315.4471 ± 0.8425
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 27.9158 ± 0.2107
      mae: 3.5921 ± 0.0209
      huber: 3.1545 ± 0.0215
      swd: 10.1110 ± 0.5975
      ept: 272.8180 ± 0.0876
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 134.11 seconds
    
    Experiment complete: TimeMixer_etth2_seq96_pred720_20250511_1621
    Model: TimeMixer
    Dataset: etth2
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
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.3185, mae: 0.3267, huber: 0.1205, swd: 0.1128, target_std: 0.7880
    Epoch [1/50], Val Losses: mse: 0.2587, mae: 0.3435, huber: 0.1223, swd: 0.1122, target_std: 0.9798
    Epoch [1/50], Test Losses: mse: 0.1700, mae: 0.2814, huber: 0.0814, swd: 0.0624, target_std: 0.7349
      Epoch 1 composite train-obj: 0.120517
            Val objective improved inf → 0.1223, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2996, mae: 0.3158, huber: 0.1142, swd: 0.1092, target_std: 0.7881
    Epoch [2/50], Val Losses: mse: 0.2600, mae: 0.3467, huber: 0.1231, swd: 0.1083, target_std: 0.9798
    Epoch [2/50], Test Losses: mse: 0.1745, mae: 0.2846, huber: 0.0833, swd: 0.0628, target_std: 0.7349
      Epoch 2 composite train-obj: 0.114240
            No improvement (0.1231), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.2883, mae: 0.3113, huber: 0.1112, swd: 0.1048, target_std: 0.7879
    Epoch [3/50], Val Losses: mse: 0.2672, mae: 0.3474, huber: 0.1255, swd: 0.1209, target_std: 0.9798
    Epoch [3/50], Test Losses: mse: 0.1727, mae: 0.2819, huber: 0.0823, swd: 0.0657, target_std: 0.7349
      Epoch 3 composite train-obj: 0.111202
            No improvement (0.1255), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2834, mae: 0.3093, huber: 0.1096, swd: 0.1042, target_std: 0.7879
    Epoch [4/50], Val Losses: mse: 0.2636, mae: 0.3451, huber: 0.1242, swd: 0.1140, target_std: 0.9798
    Epoch [4/50], Test Losses: mse: 0.1702, mae: 0.2827, huber: 0.0815, swd: 0.0611, target_std: 0.7349
      Epoch 4 composite train-obj: 0.109630
            No improvement (0.1242), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.2752, mae: 0.3060, huber: 0.1072, swd: 0.0999, target_std: 0.7879
    Epoch [5/50], Val Losses: mse: 0.2620, mae: 0.3464, huber: 0.1234, swd: 0.1189, target_std: 0.9798
    Epoch [5/50], Test Losses: mse: 0.1751, mae: 0.2837, huber: 0.0834, swd: 0.0681, target_std: 0.7349
      Epoch 5 composite train-obj: 0.107225
            No improvement (0.1234), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.2701, mae: 0.3040, huber: 0.1057, swd: 0.0981, target_std: 0.7878
    Epoch [6/50], Val Losses: mse: 0.2643, mae: 0.3472, huber: 0.1248, swd: 0.1135, target_std: 0.9798
    Epoch [6/50], Test Losses: mse: 0.1696, mae: 0.2844, huber: 0.0815, swd: 0.0608, target_std: 0.7349
      Epoch 6 composite train-obj: 0.105674
    Epoch [6/50], Test Losses: mse: 0.1700, mae: 0.2814, huber: 0.0814, swd: 0.0624, target_std: 0.7349
    Best round's Test MSE: 0.1700, MAE: 0.2814, SWD: 0.0624
    Best round's Validation MSE: 0.2587, MAE: 0.3435
    Best round's Test verification MSE : 0.1700, MAE: 0.2814, SWD: 0.0624
    Time taken: 8.46 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.3156, mae: 0.3258, huber: 0.1199, swd: 0.1104, target_std: 0.7879
    Epoch [1/50], Val Losses: mse: 0.2723, mae: 0.3510, huber: 0.1280, swd: 0.1128, target_std: 0.9798
    Epoch [1/50], Test Losses: mse: 0.1702, mae: 0.2825, huber: 0.0816, swd: 0.0586, target_std: 0.7349
      Epoch 1 composite train-obj: 0.119904
            Val objective improved inf → 0.1280, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3003, mae: 0.3160, huber: 0.1144, swd: 0.1092, target_std: 0.7878
    Epoch [2/50], Val Losses: mse: 0.2643, mae: 0.3456, huber: 0.1244, swd: 0.1136, target_std: 0.9798
    Epoch [2/50], Test Losses: mse: 0.1719, mae: 0.2810, huber: 0.0820, swd: 0.0629, target_std: 0.7349
      Epoch 2 composite train-obj: 0.114403
            Val objective improved 0.1280 → 0.1244, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2900, mae: 0.3115, huber: 0.1113, swd: 0.1050, target_std: 0.7880
    Epoch [3/50], Val Losses: mse: 0.2529, mae: 0.3396, huber: 0.1194, swd: 0.1115, target_std: 0.9798
    Epoch [3/50], Test Losses: mse: 0.1726, mae: 0.2819, huber: 0.0823, swd: 0.0642, target_std: 0.7349
      Epoch 3 composite train-obj: 0.111297
            Val objective improved 0.1244 → 0.1194, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2809, mae: 0.3087, huber: 0.1090, swd: 0.1024, target_std: 0.7878
    Epoch [4/50], Val Losses: mse: 0.2744, mae: 0.3516, huber: 0.1283, swd: 0.1231, target_std: 0.9798
    Epoch [4/50], Test Losses: mse: 0.1723, mae: 0.2817, huber: 0.0822, swd: 0.0649, target_std: 0.7349
      Epoch 4 composite train-obj: 0.108991
            No improvement (0.1283), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.2763, mae: 0.3062, huber: 0.1073, swd: 0.1004, target_std: 0.7878
    Epoch [5/50], Val Losses: mse: 0.2591, mae: 0.3474, huber: 0.1227, swd: 0.1088, target_std: 0.9798
    Epoch [5/50], Test Losses: mse: 0.1797, mae: 0.2888, huber: 0.0854, swd: 0.0665, target_std: 0.7349
      Epoch 5 composite train-obj: 0.107279
            No improvement (0.1227), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.2725, mae: 0.3050, huber: 0.1063, swd: 0.0984, target_std: 0.7880
    Epoch [6/50], Val Losses: mse: 0.2596, mae: 0.3442, huber: 0.1223, swd: 0.1166, target_std: 0.9798
    Epoch [6/50], Test Losses: mse: 0.1734, mae: 0.2824, huber: 0.0826, swd: 0.0652, target_std: 0.7349
      Epoch 6 composite train-obj: 0.106279
            No improvement (0.1223), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.2649, mae: 0.3016, huber: 0.1040, swd: 0.0968, target_std: 0.7878
    Epoch [7/50], Val Losses: mse: 0.2636, mae: 0.3463, huber: 0.1245, swd: 0.1131, target_std: 0.9798
    Epoch [7/50], Test Losses: mse: 0.1728, mae: 0.2853, huber: 0.0827, swd: 0.0611, target_std: 0.7349
      Epoch 7 composite train-obj: 0.104010
            No improvement (0.1245), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.2616, mae: 0.3000, huber: 0.1028, swd: 0.0948, target_std: 0.7880
    Epoch [8/50], Val Losses: mse: 0.2577, mae: 0.3450, huber: 0.1222, swd: 0.1114, target_std: 0.9798
    Epoch [8/50], Test Losses: mse: 0.1750, mae: 0.2870, huber: 0.0836, swd: 0.0624, target_std: 0.7349
      Epoch 8 composite train-obj: 0.102770
    Epoch [8/50], Test Losses: mse: 0.1726, mae: 0.2819, huber: 0.0823, swd: 0.0642, target_std: 0.7349
    Best round's Test MSE: 0.1726, MAE: 0.2819, SWD: 0.0642
    Best round's Validation MSE: 0.2529, MAE: 0.3396
    Best round's Test verification MSE : 0.1726, MAE: 0.2819, SWD: 0.0642
    Time taken: 10.96 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.3163, mae: 0.3254, huber: 0.1199, swd: 0.1023, target_std: 0.7876
    Epoch [1/50], Val Losses: mse: 0.2638, mae: 0.3479, huber: 0.1249, swd: 0.1004, target_std: 0.9798
    Epoch [1/50], Test Losses: mse: 0.1726, mae: 0.2843, huber: 0.0827, swd: 0.0570, target_std: 0.7349
      Epoch 1 composite train-obj: 0.119861
            Val objective improved inf → 0.1249, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2982, mae: 0.3152, huber: 0.1138, swd: 0.1003, target_std: 0.7879
    Epoch [2/50], Val Losses: mse: 0.2669, mae: 0.3482, huber: 0.1259, swd: 0.1046, target_std: 0.9798
    Epoch [2/50], Test Losses: mse: 0.1708, mae: 0.2825, huber: 0.0817, swd: 0.0564, target_std: 0.7349
      Epoch 2 composite train-obj: 0.113772
            No improvement (0.1259), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.2900, mae: 0.3113, huber: 0.1111, swd: 0.0978, target_std: 0.7880
    Epoch [3/50], Val Losses: mse: 0.2657, mae: 0.3498, huber: 0.1256, swd: 0.1009, target_std: 0.9798
    Epoch [3/50], Test Losses: mse: 0.1742, mae: 0.2859, huber: 0.0833, swd: 0.0575, target_std: 0.7349
      Epoch 3 composite train-obj: 0.111126
            No improvement (0.1256), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.2833, mae: 0.3086, huber: 0.1091, swd: 0.0957, target_std: 0.7878
    Epoch [4/50], Val Losses: mse: 0.2541, mae: 0.3391, huber: 0.1200, swd: 0.1045, target_std: 0.9798
    Epoch [4/50], Test Losses: mse: 0.1737, mae: 0.2816, huber: 0.0825, swd: 0.0598, target_std: 0.7349
      Epoch 4 composite train-obj: 0.109091
            Val objective improved 0.1249 → 0.1200, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.2763, mae: 0.3052, huber: 0.1067, swd: 0.0932, target_std: 0.7879
    Epoch [5/50], Val Losses: mse: 0.2636, mae: 0.3446, huber: 0.1241, swd: 0.1051, target_std: 0.9798
    Epoch [5/50], Test Losses: mse: 0.1713, mae: 0.2834, huber: 0.0819, swd: 0.0566, target_std: 0.7349
      Epoch 5 composite train-obj: 0.106746
            No improvement (0.1241), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.2715, mae: 0.3036, huber: 0.1054, swd: 0.0911, target_std: 0.7881
    Epoch [6/50], Val Losses: mse: 0.2631, mae: 0.3454, huber: 0.1242, swd: 0.1020, target_std: 0.9798
    Epoch [6/50], Test Losses: mse: 0.1696, mae: 0.2836, huber: 0.0814, swd: 0.0558, target_std: 0.7349
      Epoch 6 composite train-obj: 0.105420
            No improvement (0.1242), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.2648, mae: 0.3012, huber: 0.1036, swd: 0.0888, target_std: 0.7879
    Epoch [7/50], Val Losses: mse: 0.2665, mae: 0.3481, huber: 0.1253, swd: 0.1061, target_std: 0.9798
    Epoch [7/50], Test Losses: mse: 0.1728, mae: 0.2824, huber: 0.0824, swd: 0.0598, target_std: 0.7349
      Epoch 7 composite train-obj: 0.103561
            No improvement (0.1253), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.2629, mae: 0.3009, huber: 0.1033, swd: 0.0869, target_std: 0.7880
    Epoch [8/50], Val Losses: mse: 0.2561, mae: 0.3426, huber: 0.1214, swd: 0.1027, target_std: 0.9798
    Epoch [8/50], Test Losses: mse: 0.1755, mae: 0.2883, huber: 0.0840, swd: 0.0611, target_std: 0.7349
      Epoch 8 composite train-obj: 0.103297
            No improvement (0.1214), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.2561, mae: 0.2990, huber: 0.1016, swd: 0.0840, target_std: 0.7878
    Epoch [9/50], Val Losses: mse: 0.2619, mae: 0.3456, huber: 0.1237, swd: 0.1032, target_std: 0.9798
    Epoch [9/50], Test Losses: mse: 0.1721, mae: 0.2843, huber: 0.0824, swd: 0.0572, target_std: 0.7349
      Epoch 9 composite train-obj: 0.101582
    Epoch [9/50], Test Losses: mse: 0.1737, mae: 0.2816, huber: 0.0825, swd: 0.0598, target_std: 0.7349
    Best round's Test MSE: 0.1737, MAE: 0.2816, SWD: 0.0598
    Best round's Validation MSE: 0.2541, MAE: 0.3391
    Best round's Test verification MSE : 0.1737, MAE: 0.2816, SWD: 0.0598
    Time taken: 12.35 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth2_seq96_pred96_20250502_2125)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.1721 ± 0.0016
      mae: 0.2816 ± 0.0002
      huber: 0.0820 ± 0.0005
      swd: 0.0621 ± 0.0018
      target_std: 0.7349 ± 0.0000
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.2552 ± 0.0025
      mae: 0.3407 ± 0.0020
      huber: 0.1206 ± 0.0013
      swd: 0.1094 ± 0.0035
      target_std: 0.9798 ± 0.0000
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 31.82 seconds
    
    Experiment complete: PatchTST_etth2_seq96_pred96_20250502_2125
    Model: PatchTST
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 94
    Validation Batches: 13
    Test Batches: 26
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.7322, mae: 2.6940, huber: 2.2849, swd: 12.5637, ept: 74.5815
    Epoch [1/50], Val Losses: mse: 20.4026, mae: 2.8503, huber: 2.4328, swd: 8.8720, ept: 69.4387
    Epoch [1/50], Test Losses: mse: 14.6999, mae: 2.3925, huber: 1.9761, swd: 7.1911, ept: 76.2424
      Epoch 1 composite train-obj: 2.284898
            Val objective improved inf → 2.4328, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.5489, mae: 2.6020, huber: 2.1958, swd: 12.1157, ept: 76.3901
    Epoch [2/50], Val Losses: mse: 19.7241, mae: 2.8178, huber: 2.4005, swd: 8.2845, ept: 70.3847
    Epoch [2/50], Test Losses: mse: 14.7861, mae: 2.3941, huber: 1.9778, swd: 7.2872, ept: 76.8370
      Epoch 2 composite train-obj: 2.195753
            Val objective improved 2.4328 → 2.4005, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.8193, mae: 2.5655, huber: 2.1599, swd: 11.5802, ept: 76.6247
    Epoch [3/50], Val Losses: mse: 20.4902, mae: 2.8570, huber: 2.4392, swd: 8.7870, ept: 69.8865
    Epoch [3/50], Test Losses: mse: 14.7659, mae: 2.3916, huber: 1.9750, swd: 7.3862, ept: 77.0343
      Epoch 3 composite train-obj: 2.159900
            No improvement (2.4392), counter 1/5
    Epoch [4/50], Train Losses: mse: 22.5610, mae: 2.5514, huber: 2.1457, swd: 11.4617, ept: 76.7765
    Epoch [4/50], Val Losses: mse: 20.1339, mae: 2.8364, huber: 2.4185, swd: 8.4587, ept: 69.9920
    Epoch [4/50], Test Losses: mse: 14.8086, mae: 2.3924, huber: 1.9762, swd: 7.3595, ept: 76.3871
      Epoch 4 composite train-obj: 2.145727
            No improvement (2.4185), counter 2/5
    Epoch [5/50], Train Losses: mse: 22.0975, mae: 2.5254, huber: 2.1200, swd: 11.1089, ept: 76.9434
    Epoch [5/50], Val Losses: mse: 20.3200, mae: 2.8476, huber: 2.4283, swd: 8.5536, ept: 70.1452
    Epoch [5/50], Test Losses: mse: 14.7942, mae: 2.3863, huber: 1.9699, swd: 7.3572, ept: 76.3784
      Epoch 5 composite train-obj: 2.120047
            No improvement (2.4283), counter 3/5
    Epoch [6/50], Train Losses: mse: 21.7452, mae: 2.5049, huber: 2.0999, swd: 10.8962, ept: 77.0622
    Epoch [6/50], Val Losses: mse: 19.5372, mae: 2.8259, huber: 2.4051, swd: 8.0431, ept: 70.3774
    Epoch [6/50], Test Losses: mse: 14.6270, mae: 2.4051, huber: 1.9875, swd: 7.3230, ept: 77.5779
      Epoch 6 composite train-obj: 2.099903
            No improvement (2.4051), counter 4/5
    Epoch [7/50], Train Losses: mse: 21.5179, mae: 2.4873, huber: 2.0824, swd: 10.7670, ept: 77.1127
    Epoch [7/50], Val Losses: mse: 20.2858, mae: 2.8490, huber: 2.4293, swd: 8.6802, ept: 70.2552
    Epoch [7/50], Test Losses: mse: 14.7343, mae: 2.4019, huber: 1.9845, swd: 7.4145, ept: 77.5041
      Epoch 7 composite train-obj: 2.082361
    Epoch [7/50], Test Losses: mse: 14.7861, mae: 2.3941, huber: 1.9778, swd: 7.2872, ept: 76.8370
    Best round's Test MSE: 14.7861, MAE: 2.3941, SWD: 7.2872
    Best round's Validation MSE: 19.7241, MAE: 2.8178, SWD: 8.2845
    Best round's Test verification MSE : 14.7861, MAE: 2.3941, SWD: 7.2872
    Time taken: 10.56 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.7993, mae: 2.6971, huber: 2.2878, swd: 12.2492, ept: 74.6667
    Epoch [1/50], Val Losses: mse: 21.2425, mae: 2.8930, huber: 2.4742, swd: 9.0018, ept: 70.0115
    Epoch [1/50], Test Losses: mse: 14.5665, mae: 2.3829, huber: 1.9662, swd: 6.8238, ept: 76.7078
      Epoch 1 composite train-obj: 2.287847
            Val objective improved inf → 2.4742, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.5634, mae: 2.6032, huber: 2.1968, swd: 11.7329, ept: 76.3683
    Epoch [2/50], Val Losses: mse: 20.7379, mae: 2.8639, huber: 2.4468, swd: 8.5797, ept: 70.1042
    Epoch [2/50], Test Losses: mse: 14.7121, mae: 2.3811, huber: 1.9651, swd: 7.0079, ept: 76.3263
      Epoch 2 composite train-obj: 2.196793
            Val objective improved 2.4742 → 2.4468, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.8474, mae: 2.5646, huber: 2.1588, swd: 11.2607, ept: 76.6129
    Epoch [3/50], Val Losses: mse: 19.1642, mae: 2.7925, huber: 2.3743, swd: 7.3100, ept: 71.2230
    Epoch [3/50], Test Losses: mse: 14.8393, mae: 2.3977, huber: 1.9816, swd: 7.0843, ept: 76.9073
      Epoch 3 composite train-obj: 2.158773
            Val objective improved 2.4468 → 2.3743, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.3039, mae: 2.5397, huber: 2.1342, swd: 10.9649, ept: 76.7754
    Epoch [4/50], Val Losses: mse: 20.5349, mae: 2.8648, huber: 2.4457, swd: 8.4972, ept: 69.4516
    Epoch [4/50], Test Losses: mse: 14.7190, mae: 2.3844, huber: 1.9679, swd: 7.0166, ept: 76.1094
      Epoch 4 composite train-obj: 2.134221
            No improvement (2.4457), counter 1/5
    Epoch [5/50], Train Losses: mse: 21.9624, mae: 2.5183, huber: 2.1130, swd: 10.7370, ept: 76.9485
    Epoch [5/50], Val Losses: mse: 19.8668, mae: 2.8507, huber: 2.4288, swd: 7.7740, ept: 70.3098
    Epoch [5/50], Test Losses: mse: 14.9492, mae: 2.4178, huber: 1.9996, swd: 7.2019, ept: 77.3408
      Epoch 5 composite train-obj: 2.113024
            No improvement (2.4288), counter 2/5
    Epoch [6/50], Train Losses: mse: 21.7159, mae: 2.5042, huber: 2.0990, swd: 10.5522, ept: 77.0809
    Epoch [6/50], Val Losses: mse: 20.2673, mae: 2.8458, huber: 2.4264, swd: 8.1085, ept: 70.1670
    Epoch [6/50], Test Losses: mse: 14.7104, mae: 2.3905, huber: 1.9734, swd: 7.0120, ept: 77.0367
      Epoch 6 composite train-obj: 2.099041
            No improvement (2.4264), counter 3/5
    Epoch [7/50], Train Losses: mse: 21.2113, mae: 2.4809, huber: 2.0761, swd: 10.1884, ept: 77.1095
    Epoch [7/50], Val Losses: mse: 20.0719, mae: 2.8489, huber: 2.4295, swd: 8.1063, ept: 69.7273
    Epoch [7/50], Test Losses: mse: 14.8021, mae: 2.4061, huber: 1.9888, swd: 7.1256, ept: 77.2657
      Epoch 7 composite train-obj: 2.076062
            No improvement (2.4295), counter 4/5
    Epoch [8/50], Train Losses: mse: 21.0479, mae: 2.4651, huber: 2.0604, swd: 10.0643, ept: 77.2221
    Epoch [8/50], Val Losses: mse: 19.6642, mae: 2.8298, huber: 2.4085, swd: 7.6931, ept: 69.9615
    Epoch [8/50], Test Losses: mse: 14.8184, mae: 2.4276, huber: 2.0085, swd: 7.1959, ept: 77.7811
      Epoch 8 composite train-obj: 2.060418
    Epoch [8/50], Test Losses: mse: 14.8393, mae: 2.3977, huber: 1.9816, swd: 7.0843, ept: 76.9073
    Best round's Test MSE: 14.8393, MAE: 2.3977, SWD: 7.0843
    Best round's Validation MSE: 19.1642, MAE: 2.7925, SWD: 7.3100
    Best round's Test verification MSE : 14.8393, MAE: 2.3977, SWD: 7.0843
    Time taken: 11.65 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 24.6861, mae: 2.6892, huber: 2.2802, swd: 11.3756, ept: 74.8045
    Epoch [1/50], Val Losses: mse: 20.6458, mae: 2.8701, huber: 2.4514, swd: 8.2072, ept: 69.7887
    Epoch [1/50], Test Losses: mse: 14.7716, mae: 2.3999, huber: 1.9832, swd: 6.5994, ept: 76.6841
      Epoch 1 composite train-obj: 2.280221
            Val objective improved inf → 2.4514, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 23.3448, mae: 2.5952, huber: 2.1887, swd: 10.7406, ept: 76.4584
    Epoch [2/50], Val Losses: mse: 20.1250, mae: 2.8378, huber: 2.4197, swd: 7.4672, ept: 70.4701
    Epoch [2/50], Test Losses: mse: 14.8155, mae: 2.3838, huber: 1.9683, swd: 6.6264, ept: 76.2620
      Epoch 2 composite train-obj: 2.188703
            Val objective improved 2.4514 → 2.4197, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 22.9277, mae: 2.5695, huber: 2.1636, swd: 10.5532, ept: 76.6350
    Epoch [3/50], Val Losses: mse: 19.4731, mae: 2.8096, huber: 2.3912, swd: 7.1380, ept: 70.7098
    Epoch [3/50], Test Losses: mse: 14.8108, mae: 2.3948, huber: 1.9787, swd: 6.6518, ept: 76.6158
      Epoch 3 composite train-obj: 2.163581
            Val objective improved 2.4197 → 2.3912, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 22.4196, mae: 2.5388, huber: 2.1332, swd: 10.2496, ept: 76.9155
    Epoch [4/50], Val Losses: mse: 19.8561, mae: 2.8162, huber: 2.3983, swd: 7.3874, ept: 70.1942
    Epoch [4/50], Test Losses: mse: 14.9626, mae: 2.3962, huber: 1.9796, swd: 6.7915, ept: 76.5168
      Epoch 4 composite train-obj: 2.133214
            No improvement (2.3983), counter 1/5
    Epoch [5/50], Train Losses: mse: 22.1084, mae: 2.5164, huber: 2.1110, swd: 10.0752, ept: 76.9723
    Epoch [5/50], Val Losses: mse: 19.6514, mae: 2.8094, huber: 2.3921, swd: 7.2884, ept: 70.1669
    Epoch [5/50], Test Losses: mse: 14.8691, mae: 2.3991, huber: 1.9824, swd: 6.7777, ept: 77.0390
      Epoch 5 composite train-obj: 2.110995
            No improvement (2.3921), counter 2/5
    Epoch [6/50], Train Losses: mse: 21.7881, mae: 2.4939, huber: 2.0888, swd: 9.8670, ept: 76.9891
    Epoch [6/50], Val Losses: mse: 20.4207, mae: 2.8509, huber: 2.4322, swd: 7.7543, ept: 69.9099
    Epoch [6/50], Test Losses: mse: 14.9972, mae: 2.4037, huber: 1.9866, swd: 6.6944, ept: 75.9981
      Epoch 6 composite train-obj: 2.088781
            No improvement (2.4322), counter 3/5
    Epoch [7/50], Train Losses: mse: 21.5526, mae: 2.4793, huber: 2.0743, swd: 9.7725, ept: 77.1622
    Epoch [7/50], Val Losses: mse: 20.3181, mae: 2.8456, huber: 2.4265, swd: 7.7270, ept: 69.8260
    Epoch [7/50], Test Losses: mse: 15.0274, mae: 2.4050, huber: 1.9873, swd: 6.7289, ept: 76.4446
      Epoch 7 composite train-obj: 2.074317
            No improvement (2.4265), counter 4/5
    Epoch [8/50], Train Losses: mse: 21.2985, mae: 2.4665, huber: 2.0616, swd: 9.6114, ept: 77.2909
    Epoch [8/50], Val Losses: mse: 19.6625, mae: 2.8143, huber: 2.3954, swd: 7.2243, ept: 70.3004
    Epoch [8/50], Test Losses: mse: 15.0360, mae: 2.4268, huber: 2.0085, swd: 6.8527, ept: 76.9690
      Epoch 8 composite train-obj: 2.061646
    Epoch [8/50], Test Losses: mse: 14.8108, mae: 2.3948, huber: 1.9787, swd: 6.6518, ept: 76.6158
    Best round's Test MSE: 14.8108, MAE: 2.3948, SWD: 6.6518
    Best round's Validation MSE: 19.4731, MAE: 2.8096, SWD: 7.1380
    Best round's Test verification MSE : 14.8108, MAE: 2.3948, SWD: 6.6518
    Time taken: 11.55 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth2_seq96_pred96_20250511_1623)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 14.8121 ± 0.0217
      mae: 2.3955 ± 0.0016
      huber: 1.9794 ± 0.0016
      swd: 7.0078 ± 0.2650
      ept: 76.7867 ± 0.1242
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 19.4538 ± 0.2290
      mae: 2.8066 ± 0.0105
      huber: 2.3886 ± 0.0109
      swd: 7.5775 ± 0.5048
      ept: 70.7725 ± 0.3451
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 33.82 seconds
    
    Experiment complete: PatchTST_etth2_seq96_pred96_20250511_1623
    Model: PatchTST
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.4071, mae: 0.3708, huber: 0.1495, swd: 0.1521, target_std: 0.7894
    Epoch [1/50], Val Losses: mse: 0.3568, mae: 0.4110, huber: 0.1650, swd: 0.1493, target_std: 0.9804
    Epoch [1/50], Test Losses: mse: 0.2054, mae: 0.3107, huber: 0.0983, swd: 0.0773, target_std: 0.7299
      Epoch 1 composite train-obj: 0.149533
            Val objective improved inf → 0.1650, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3922, mae: 0.3619, huber: 0.1442, swd: 0.1506, target_std: 0.7895
    Epoch [2/50], Val Losses: mse: 0.3550, mae: 0.4150, huber: 0.1652, swd: 0.1416, target_std: 0.9804
    Epoch [2/50], Test Losses: mse: 0.2102, mae: 0.3204, huber: 0.1009, swd: 0.0763, target_std: 0.7299
      Epoch 2 composite train-obj: 0.144209
            No improvement (0.1652), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3819, mae: 0.3573, huber: 0.1408, swd: 0.1478, target_std: 0.7894
    Epoch [3/50], Val Losses: mse: 0.3383, mae: 0.4045, huber: 0.1578, swd: 0.1371, target_std: 0.9804
    Epoch [3/50], Test Losses: mse: 0.2026, mae: 0.3105, huber: 0.0970, swd: 0.0735, target_std: 0.7299
      Epoch 3 composite train-obj: 0.140811
            Val objective improved 0.1650 → 0.1578, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3697, mae: 0.3526, huber: 0.1372, swd: 0.1434, target_std: 0.7894
    Epoch [4/50], Val Losses: mse: 0.3373, mae: 0.4019, huber: 0.1570, swd: 0.1468, target_std: 0.9804
    Epoch [4/50], Test Losses: mse: 0.2036, mae: 0.3139, huber: 0.0979, swd: 0.0761, target_std: 0.7299
      Epoch 4 composite train-obj: 0.137212
            Val objective improved 0.1578 → 0.1570, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3623, mae: 0.3493, huber: 0.1347, swd: 0.1408, target_std: 0.7894
    Epoch [5/50], Val Losses: mse: 0.3466, mae: 0.4049, huber: 0.1603, swd: 0.1563, target_std: 0.9804
    Epoch [5/50], Test Losses: mse: 0.2040, mae: 0.3137, huber: 0.0979, swd: 0.0756, target_std: 0.7299
      Epoch 5 composite train-obj: 0.134722
            No improvement (0.1603), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.3564, mae: 0.3461, huber: 0.1325, swd: 0.1386, target_std: 0.7894
    Epoch [6/50], Val Losses: mse: 0.3338, mae: 0.3981, huber: 0.1552, swd: 0.1472, target_std: 0.9804
    Epoch [6/50], Test Losses: mse: 0.2047, mae: 0.3133, huber: 0.0983, swd: 0.0774, target_std: 0.7299
      Epoch 6 composite train-obj: 0.132482
            Val objective improved 0.1570 → 0.1552, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3511, mae: 0.3440, huber: 0.1309, swd: 0.1366, target_std: 0.7894
    Epoch [7/50], Val Losses: mse: 0.3330, mae: 0.4003, huber: 0.1554, swd: 0.1395, target_std: 0.9804
    Epoch [7/50], Test Losses: mse: 0.2052, mae: 0.3164, huber: 0.0987, swd: 0.0750, target_std: 0.7299
      Epoch 7 composite train-obj: 0.130928
            No improvement (0.1554), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.3471, mae: 0.3428, huber: 0.1299, swd: 0.1355, target_std: 0.7894
    Epoch [8/50], Val Losses: mse: 0.3411, mae: 0.4052, huber: 0.1588, swd: 0.1411, target_std: 0.9804
    Epoch [8/50], Test Losses: mse: 0.2044, mae: 0.3166, huber: 0.0984, swd: 0.0738, target_std: 0.7299
      Epoch 8 composite train-obj: 0.129907
            No improvement (0.1588), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.3423, mae: 0.3409, huber: 0.1285, swd: 0.1345, target_std: 0.7894
    Epoch [9/50], Val Losses: mse: 0.3508, mae: 0.4100, huber: 0.1626, swd: 0.1464, target_std: 0.9804
    Epoch [9/50], Test Losses: mse: 0.2060, mae: 0.3180, huber: 0.0992, swd: 0.0762, target_std: 0.7299
      Epoch 9 composite train-obj: 0.128538
            No improvement (0.1626), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.3402, mae: 0.3393, huber: 0.1274, swd: 0.1330, target_std: 0.7894
    Epoch [10/50], Val Losses: mse: 0.3335, mae: 0.3984, huber: 0.1549, swd: 0.1433, target_std: 0.9804
    Epoch [10/50], Test Losses: mse: 0.2093, mae: 0.3157, huber: 0.1000, swd: 0.0789, target_std: 0.7299
      Epoch 10 composite train-obj: 0.127370
            Val objective improved 0.1552 → 0.1549, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.3356, mae: 0.3377, huber: 0.1261, swd: 0.1303, target_std: 0.7894
    Epoch [11/50], Val Losses: mse: 0.3275, mae: 0.3964, huber: 0.1529, swd: 0.1349, target_std: 0.9804
    Epoch [11/50], Test Losses: mse: 0.2046, mae: 0.3144, huber: 0.0983, swd: 0.0750, target_std: 0.7299
      Epoch 11 composite train-obj: 0.126070
            Val objective improved 0.1549 → 0.1529, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.3339, mae: 0.3369, huber: 0.1254, swd: 0.1296, target_std: 0.7894
    Epoch [12/50], Val Losses: mse: 0.3424, mae: 0.4050, huber: 0.1592, swd: 0.1391, target_std: 0.9804
    Epoch [12/50], Test Losses: mse: 0.2061, mae: 0.3156, huber: 0.0989, swd: 0.0765, target_std: 0.7299
      Epoch 12 composite train-obj: 0.125413
            No improvement (0.1592), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.3295, mae: 0.3351, huber: 0.1242, swd: 0.1281, target_std: 0.7894
    Epoch [13/50], Val Losses: mse: 0.3393, mae: 0.4019, huber: 0.1573, swd: 0.1449, target_std: 0.9804
    Epoch [13/50], Test Losses: mse: 0.2085, mae: 0.3158, huber: 0.0998, swd: 0.0783, target_std: 0.7299
      Epoch 13 composite train-obj: 0.124151
            No improvement (0.1573), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.3266, mae: 0.3335, huber: 0.1229, swd: 0.1264, target_std: 0.7894
    Epoch [14/50], Val Losses: mse: 0.3472, mae: 0.4063, huber: 0.1604, swd: 0.1499, target_std: 0.9804
    Epoch [14/50], Test Losses: mse: 0.2077, mae: 0.3156, huber: 0.0995, swd: 0.0776, target_std: 0.7299
      Epoch 14 composite train-obj: 0.122906
            No improvement (0.1604), counter 3/5
    Epoch [15/50], Train Losses: mse: 0.3245, mae: 0.3329, huber: 0.1224, swd: 0.1260, target_std: 0.7895
    Epoch [15/50], Val Losses: mse: 0.3364, mae: 0.3999, huber: 0.1561, swd: 0.1448, target_std: 0.9804
    Epoch [15/50], Test Losses: mse: 0.2107, mae: 0.3173, huber: 0.1006, swd: 0.0777, target_std: 0.7299
      Epoch 15 composite train-obj: 0.122448
            No improvement (0.1561), counter 4/5
    Epoch [16/50], Train Losses: mse: 0.3215, mae: 0.3316, huber: 0.1215, swd: 0.1236, target_std: 0.7895
    Epoch [16/50], Val Losses: mse: 0.3488, mae: 0.4095, huber: 0.1617, swd: 0.1467, target_std: 0.9804
    Epoch [16/50], Test Losses: mse: 0.2096, mae: 0.3190, huber: 0.1005, swd: 0.0775, target_std: 0.7299
      Epoch 16 composite train-obj: 0.121480
    Epoch [16/50], Test Losses: mse: 0.2046, mae: 0.3144, huber: 0.0983, swd: 0.0750, target_std: 0.7299
    Best round's Test MSE: 0.2046, MAE: 0.3144, SWD: 0.0750
    Best round's Validation MSE: 0.3275, MAE: 0.3964
    Best round's Test verification MSE : 0.2046, MAE: 0.3144, SWD: 0.0750
    Time taken: 21.23 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4074, mae: 0.3700, huber: 0.1494, swd: 0.1554, target_std: 0.7894
    Epoch [1/50], Val Losses: mse: 0.3671, mae: 0.4196, huber: 0.1697, swd: 0.1500, target_std: 0.9804
    Epoch [1/50], Test Losses: mse: 0.2171, mae: 0.3237, huber: 0.1037, swd: 0.0836, target_std: 0.7299
      Epoch 1 composite train-obj: 0.149384
            Val objective improved inf → 0.1697, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3862, mae: 0.3595, huber: 0.1425, swd: 0.1520, target_std: 0.7894
    Epoch [2/50], Val Losses: mse: 0.3370, mae: 0.4025, huber: 0.1571, swd: 0.1399, target_std: 0.9804
    Epoch [2/50], Test Losses: mse: 0.2044, mae: 0.3136, huber: 0.0981, swd: 0.0782, target_std: 0.7299
      Epoch 2 composite train-obj: 0.142499
            Val objective improved 0.1697 → 0.1571, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3767, mae: 0.3562, huber: 0.1396, swd: 0.1476, target_std: 0.7894
    Epoch [3/50], Val Losses: mse: 0.3386, mae: 0.4039, huber: 0.1578, swd: 0.1473, target_std: 0.9804
    Epoch [3/50], Test Losses: mse: 0.2045, mae: 0.3130, huber: 0.0980, swd: 0.0800, target_std: 0.7299
      Epoch 3 composite train-obj: 0.139635
            No improvement (0.1578), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.3668, mae: 0.3519, huber: 0.1365, swd: 0.1443, target_std: 0.7894
    Epoch [4/50], Val Losses: mse: 0.3387, mae: 0.4027, huber: 0.1575, swd: 0.1514, target_std: 0.9804
    Epoch [4/50], Test Losses: mse: 0.2014, mae: 0.3112, huber: 0.0969, swd: 0.0766, target_std: 0.7299
      Epoch 4 composite train-obj: 0.136460
            No improvement (0.1575), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.3642, mae: 0.3499, huber: 0.1352, swd: 0.1432, target_std: 0.7895
    Epoch [5/50], Val Losses: mse: 0.3287, mae: 0.3962, huber: 0.1533, swd: 0.1468, target_std: 0.9804
    Epoch [5/50], Test Losses: mse: 0.2027, mae: 0.3111, huber: 0.0972, swd: 0.0788, target_std: 0.7299
      Epoch 5 composite train-obj: 0.135155
            Val objective improved 0.1571 → 0.1533, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3581, mae: 0.3472, huber: 0.1332, swd: 0.1420, target_std: 0.7894
    Epoch [6/50], Val Losses: mse: 0.3360, mae: 0.4001, huber: 0.1558, swd: 0.1509, target_std: 0.9804
    Epoch [6/50], Test Losses: mse: 0.2032, mae: 0.3103, huber: 0.0973, swd: 0.0811, target_std: 0.7299
      Epoch 6 composite train-obj: 0.133223
            No improvement (0.1558), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.3521, mae: 0.3447, huber: 0.1313, swd: 0.1397, target_std: 0.7894
    Epoch [7/50], Val Losses: mse: 0.3419, mae: 0.4050, huber: 0.1590, swd: 0.1479, target_std: 0.9804
    Epoch [7/50], Test Losses: mse: 0.2046, mae: 0.3141, huber: 0.0980, swd: 0.0780, target_std: 0.7299
      Epoch 7 composite train-obj: 0.131257
            No improvement (0.1590), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.3490, mae: 0.3438, huber: 0.1305, swd: 0.1387, target_std: 0.7894
    Epoch [8/50], Val Losses: mse: 0.3318, mae: 0.3961, huber: 0.1536, swd: 0.1504, target_std: 0.9804
    Epoch [8/50], Test Losses: mse: 0.2090, mae: 0.3144, huber: 0.0999, swd: 0.0850, target_std: 0.7299
      Epoch 8 composite train-obj: 0.130477
            No improvement (0.1536), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.3461, mae: 0.3421, huber: 0.1294, swd: 0.1376, target_std: 0.7894
    Epoch [9/50], Val Losses: mse: 0.3316, mae: 0.3989, huber: 0.1546, swd: 0.1450, target_std: 0.9804
    Epoch [9/50], Test Losses: mse: 0.2014, mae: 0.3117, huber: 0.0968, swd: 0.0774, target_std: 0.7299
      Epoch 9 composite train-obj: 0.129403
            No improvement (0.1546), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.3399, mae: 0.3397, huber: 0.1275, swd: 0.1363, target_std: 0.7894
    Epoch [10/50], Val Losses: mse: 0.3441, mae: 0.4071, huber: 0.1599, swd: 0.1524, target_std: 0.9804
    Epoch [10/50], Test Losses: mse: 0.2044, mae: 0.3131, huber: 0.0980, swd: 0.0803, target_std: 0.7299
      Epoch 10 composite train-obj: 0.127492
    Epoch [10/50], Test Losses: mse: 0.2027, mae: 0.3111, huber: 0.0972, swd: 0.0788, target_std: 0.7299
    Best round's Test MSE: 0.2027, MAE: 0.3111, SWD: 0.0788
    Best round's Validation MSE: 0.3287, MAE: 0.3962
    Best round's Test verification MSE : 0.2027, MAE: 0.3111, SWD: 0.0788
    Time taken: 13.15 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4064, mae: 0.3701, huber: 0.1493, swd: 0.1375, target_std: 0.7895
    Epoch [1/50], Val Losses: mse: 0.3307, mae: 0.3975, huber: 0.1542, swd: 0.1300, target_std: 0.9804
    Epoch [1/50], Test Losses: mse: 0.2030, mae: 0.3103, huber: 0.0973, swd: 0.0687, target_std: 0.7299
      Epoch 1 composite train-obj: 0.149282
            Val objective improved inf → 0.1542, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3873, mae: 0.3600, huber: 0.1428, swd: 0.1360, target_std: 0.7895
    Epoch [2/50], Val Losses: mse: 0.3486, mae: 0.4066, huber: 0.1615, swd: 0.1362, target_std: 0.9804
    Epoch [2/50], Test Losses: mse: 0.2012, mae: 0.3093, huber: 0.0966, swd: 0.0674, target_std: 0.7299
      Epoch 2 composite train-obj: 0.142811
            No improvement (0.1615), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.3741, mae: 0.3548, huber: 0.1386, swd: 0.1310, target_std: 0.7894
    Epoch [3/50], Val Losses: mse: 0.3268, mae: 0.3948, huber: 0.1518, swd: 0.1328, target_std: 0.9804
    Epoch [3/50], Test Losses: mse: 0.2026, mae: 0.3108, huber: 0.0972, swd: 0.0690, target_std: 0.7299
      Epoch 3 composite train-obj: 0.138646
            Val objective improved 0.1542 → 0.1518, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3655, mae: 0.3511, huber: 0.1359, swd: 0.1289, target_std: 0.7894
    Epoch [4/50], Val Losses: mse: 0.3392, mae: 0.4026, huber: 0.1577, swd: 0.1327, target_std: 0.9804
    Epoch [4/50], Test Losses: mse: 0.2005, mae: 0.3107, huber: 0.0963, swd: 0.0654, target_std: 0.7299
      Epoch 4 composite train-obj: 0.135856
            No improvement (0.1577), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3600, mae: 0.3484, huber: 0.1340, swd: 0.1270, target_std: 0.7895
    Epoch [5/50], Val Losses: mse: 0.3451, mae: 0.4100, huber: 0.1609, swd: 0.1289, target_std: 0.9804
    Epoch [5/50], Test Losses: mse: 0.2049, mae: 0.3162, huber: 0.0985, swd: 0.0662, target_std: 0.7299
      Epoch 5 composite train-obj: 0.133965
            No improvement (0.1609), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3559, mae: 0.3468, huber: 0.1325, swd: 0.1256, target_std: 0.7894
    Epoch [6/50], Val Losses: mse: 0.3383, mae: 0.4019, huber: 0.1569, swd: 0.1341, target_std: 0.9804
    Epoch [6/50], Test Losses: mse: 0.2006, mae: 0.3098, huber: 0.0963, swd: 0.0674, target_std: 0.7299
      Epoch 6 composite train-obj: 0.132491
            No improvement (0.1569), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3527, mae: 0.3450, huber: 0.1314, swd: 0.1256, target_std: 0.7894
    Epoch [7/50], Val Losses: mse: 0.3438, mae: 0.4061, huber: 0.1595, swd: 0.1350, target_std: 0.9804
    Epoch [7/50], Test Losses: mse: 0.2028, mae: 0.3111, huber: 0.0973, swd: 0.0696, target_std: 0.7299
      Epoch 7 composite train-obj: 0.131428
            No improvement (0.1595), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.3459, mae: 0.3421, huber: 0.1293, swd: 0.1239, target_std: 0.7894
    Epoch [8/50], Val Losses: mse: 0.3305, mae: 0.3995, huber: 0.1541, swd: 0.1259, target_std: 0.9804
    Epoch [8/50], Test Losses: mse: 0.2084, mae: 0.3183, huber: 0.1000, swd: 0.0700, target_std: 0.7299
      Epoch 8 composite train-obj: 0.129265
    Epoch [8/50], Test Losses: mse: 0.2026, mae: 0.3108, huber: 0.0972, swd: 0.0690, target_std: 0.7299
    Best round's Test MSE: 0.2026, MAE: 0.3108, SWD: 0.0690
    Best round's Validation MSE: 0.3268, MAE: 0.3948
    Best round's Test verification MSE : 0.2026, MAE: 0.3108, SWD: 0.0690
    Time taken: 10.59 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth2_seq96_pred196_20250502_2145)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2033 ± 0.0009
      mae: 0.3121 ± 0.0017
      huber: 0.0976 ± 0.0005
      swd: 0.0743 ± 0.0040
      target_std: 0.7299 ± 0.0000
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3277 ± 0.0008
      mae: 0.3958 ± 0.0007
      huber: 0.1527 ± 0.0006
      swd: 0.1382 ± 0.0062
      target_std: 0.9804 ± 0.0000
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 45.01 seconds
    
    Experiment complete: PatchTST_etth2_seq96_pred196_20250502_2145
    Model: PatchTST
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 12
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.2151, mae: 3.0666, huber: 2.6504, swd: 16.2441, ept: 129.3012
    Epoch [1/50], Val Losses: mse: 28.6794, mae: 3.4574, huber: 3.0289, swd: 13.3773, ept: 111.7916
    Epoch [1/50], Test Losses: mse: 17.7474, mae: 2.6658, huber: 2.2445, swd: 8.6568, ept: 130.8381
      Epoch 1 composite train-obj: 2.650426
            Val objective improved inf → 3.0289, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 30.8438, mae: 2.9808, huber: 2.5666, swd: 15.6128, ept: 133.1202
    Epoch [2/50], Val Losses: mse: 26.1511, mae: 3.3351, huber: 2.9071, swd: 11.1714, ept: 112.5148
    Epoch [2/50], Test Losses: mse: 17.0466, mae: 2.6391, huber: 2.2174, swd: 8.1935, ept: 130.7170
      Epoch 2 composite train-obj: 2.566614
            Val objective improved 3.0289 → 2.9071, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 30.1544, mae: 2.9466, huber: 2.5328, swd: 15.1256, ept: 133.9021
    Epoch [3/50], Val Losses: mse: 25.2090, mae: 3.2944, huber: 2.8653, swd: 10.1325, ept: 113.1301
    Epoch [3/50], Test Losses: mse: 17.0454, mae: 2.6312, huber: 2.2103, swd: 8.1508, ept: 130.9535
      Epoch 3 composite train-obj: 2.532757
            Val objective improved 2.9071 → 2.8653, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 29.4653, mae: 2.9105, huber: 2.4970, swd: 14.5790, ept: 134.2122
    Epoch [4/50], Val Losses: mse: 26.0151, mae: 3.3184, huber: 2.8895, swd: 10.5342, ept: 113.1987
    Epoch [4/50], Test Losses: mse: 17.7968, mae: 2.6637, huber: 2.2424, swd: 8.7075, ept: 132.6417
      Epoch 4 composite train-obj: 2.497031
            No improvement (2.8895), counter 1/5
    Epoch [5/50], Train Losses: mse: 28.7757, mae: 2.8774, huber: 2.4640, swd: 14.0427, ept: 134.7612
    Epoch [5/50], Val Losses: mse: 25.6658, mae: 3.2907, huber: 2.8618, swd: 10.3098, ept: 113.0388
    Epoch [5/50], Test Losses: mse: 17.8007, mae: 2.6681, huber: 2.2465, swd: 8.7448, ept: 132.3446
      Epoch 5 composite train-obj: 2.464029
            Val objective improved 2.8653 → 2.8618, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 28.4395, mae: 2.8575, huber: 2.4443, swd: 13.8086, ept: 134.7668
    Epoch [6/50], Val Losses: mse: 25.9465, mae: 3.3315, huber: 2.9017, swd: 10.7242, ept: 113.4126
    Epoch [6/50], Test Losses: mse: 17.9614, mae: 2.7015, huber: 2.2793, swd: 9.1053, ept: 133.0454
      Epoch 6 composite train-obj: 2.444336
            No improvement (2.9017), counter 1/5
    Epoch [7/50], Train Losses: mse: 28.2683, mae: 2.8426, huber: 2.4296, swd: 13.6699, ept: 135.0682
    Epoch [7/50], Val Losses: mse: 25.0935, mae: 3.2706, huber: 2.8420, swd: 10.1912, ept: 114.7395
    Epoch [7/50], Test Losses: mse: 17.6775, mae: 2.6905, huber: 2.2680, swd: 8.8648, ept: 133.3429
      Epoch 7 composite train-obj: 2.429564
            Val objective improved 2.8618 → 2.8420, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 27.8260, mae: 2.8206, huber: 2.4079, swd: 13.3117, ept: 135.2765
    Epoch [8/50], Val Losses: mse: 25.9564, mae: 3.3370, huber: 2.9063, swd: 10.9313, ept: 113.1184
    Epoch [8/50], Test Losses: mse: 17.4180, mae: 2.6880, huber: 2.2647, swd: 8.6382, ept: 134.0820
      Epoch 8 composite train-obj: 2.407869
            No improvement (2.9063), counter 1/5
    Epoch [9/50], Train Losses: mse: 27.6185, mae: 2.8071, huber: 2.3945, swd: 13.1890, ept: 135.5828
    Epoch [9/50], Val Losses: mse: 25.8882, mae: 3.3148, huber: 2.8855, swd: 10.5526, ept: 112.4026
    Epoch [9/50], Test Losses: mse: 17.4732, mae: 2.6782, huber: 2.2554, swd: 8.5475, ept: 133.7022
      Epoch 9 composite train-obj: 2.394523
            No improvement (2.8855), counter 2/5
    Epoch [10/50], Train Losses: mse: 27.4106, mae: 2.7892, huber: 2.3768, swd: 13.0148, ept: 135.7387
    Epoch [10/50], Val Losses: mse: 25.2919, mae: 3.2484, huber: 2.8210, swd: 10.0090, ept: 113.3074
    Epoch [10/50], Test Losses: mse: 18.1579, mae: 2.7026, huber: 2.2803, swd: 9.1440, ept: 132.1251
      Epoch 10 composite train-obj: 2.376844
            Val objective improved 2.8420 → 2.8210, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 27.0040, mae: 2.7702, huber: 2.3578, swd: 12.6724, ept: 135.9074
    Epoch [11/50], Val Losses: mse: 25.2489, mae: 3.2609, huber: 2.8329, swd: 10.0625, ept: 113.6687
    Epoch [11/50], Test Losses: mse: 17.5122, mae: 2.6767, huber: 2.2541, swd: 8.5204, ept: 133.7959
      Epoch 11 composite train-obj: 2.357789
            No improvement (2.8329), counter 1/5
    Epoch [12/50], Train Losses: mse: 26.8067, mae: 2.7559, huber: 2.3438, swd: 12.5085, ept: 136.0907
    Epoch [12/50], Val Losses: mse: 25.2813, mae: 3.2651, huber: 2.8354, swd: 10.1345, ept: 113.9425
    Epoch [12/50], Test Losses: mse: 18.1290, mae: 2.7248, huber: 2.3013, swd: 9.1610, ept: 132.8549
      Epoch 12 composite train-obj: 2.343759
            No improvement (2.8354), counter 2/5
    Epoch [13/50], Train Losses: mse: 26.5757, mae: 2.7401, huber: 2.3280, swd: 12.3544, ept: 136.1596
    Epoch [13/50], Val Losses: mse: 25.9747, mae: 3.3144, huber: 2.8844, swd: 10.6539, ept: 113.5063
    Epoch [13/50], Test Losses: mse: 17.7521, mae: 2.6974, huber: 2.2739, swd: 8.8001, ept: 134.9762
      Epoch 13 composite train-obj: 2.327973
            No improvement (2.8844), counter 3/5
    Epoch [14/50], Train Losses: mse: 26.3723, mae: 2.7275, huber: 2.3156, swd: 12.1643, ept: 136.4569
    Epoch [14/50], Val Losses: mse: 26.7815, mae: 3.3474, huber: 2.9185, swd: 11.3837, ept: 112.6280
    Epoch [14/50], Test Losses: mse: 18.2885, mae: 2.7248, huber: 2.3014, swd: 9.2185, ept: 133.3357
      Epoch 14 composite train-obj: 2.315623
            No improvement (2.9185), counter 4/5
    Epoch [15/50], Train Losses: mse: 26.1391, mae: 2.7134, huber: 2.3014, swd: 12.0017, ept: 136.5524
    Epoch [15/50], Val Losses: mse: 26.6896, mae: 3.3439, huber: 2.9128, swd: 10.9603, ept: 111.9319
    Epoch [15/50], Test Losses: mse: 17.8673, mae: 2.7006, huber: 2.2769, swd: 8.8400, ept: 133.9405
      Epoch 15 composite train-obj: 2.301367
    Epoch [15/50], Test Losses: mse: 18.1579, mae: 2.7026, huber: 2.2803, swd: 9.1440, ept: 132.1251
    Best round's Test MSE: 18.1579, MAE: 2.7026, SWD: 9.1440
    Best round's Validation MSE: 25.2919, MAE: 3.2484, SWD: 10.0090
    Best round's Test verification MSE : 18.1579, MAE: 2.7026, SWD: 9.1440
    Time taken: 22.40 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.2543, mae: 3.0645, huber: 2.6484, swd: 16.8862, ept: 129.4505
    Epoch [1/50], Val Losses: mse: 27.7158, mae: 3.4222, huber: 2.9941, swd: 13.2432, ept: 112.2213
    Epoch [1/50], Test Losses: mse: 17.5546, mae: 2.6867, huber: 2.2640, swd: 9.0369, ept: 132.4276
      Epoch 1 composite train-obj: 2.648421
            Val objective improved inf → 2.9941, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 30.7169, mae: 2.9740, huber: 2.5599, swd: 16.0392, ept: 133.1910
    Epoch [2/50], Val Losses: mse: 26.4535, mae: 3.3502, huber: 2.9223, swd: 11.5490, ept: 112.1629
    Epoch [2/50], Test Losses: mse: 17.1436, mae: 2.6449, huber: 2.2232, swd: 8.5190, ept: 132.1781
      Epoch 2 composite train-obj: 2.559909
            Val objective improved 2.9941 → 2.9223, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 30.0456, mae: 2.9450, huber: 2.5309, swd: 15.5801, ept: 133.8528
    Epoch [3/50], Val Losses: mse: 26.0164, mae: 3.3352, huber: 2.9067, swd: 11.0771, ept: 112.9759
    Epoch [3/50], Test Losses: mse: 17.3067, mae: 2.6511, huber: 2.2293, swd: 8.5798, ept: 131.1214
      Epoch 3 composite train-obj: 2.530927
            Val objective improved 2.9223 → 2.9067, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 29.2055, mae: 2.9067, huber: 2.4930, swd: 14.8682, ept: 134.0521
    Epoch [4/50], Val Losses: mse: 25.7091, mae: 3.3120, huber: 2.8837, swd: 10.9974, ept: 112.8079
    Epoch [4/50], Test Losses: mse: 17.3268, mae: 2.6510, huber: 2.2294, swd: 8.6331, ept: 132.5233
      Epoch 4 composite train-obj: 2.493041
            Val objective improved 2.9067 → 2.8837, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 28.8564, mae: 2.8873, huber: 2.4736, swd: 14.5786, ept: 134.5971
    Epoch [5/50], Val Losses: mse: 24.6858, mae: 3.2525, huber: 2.8252, swd: 10.0555, ept: 113.4074
    Epoch [5/50], Test Losses: mse: 17.2485, mae: 2.6486, huber: 2.2276, swd: 8.6281, ept: 132.5434
      Epoch 5 composite train-obj: 2.473624
            Val objective improved 2.8837 → 2.8252, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 28.5975, mae: 2.8678, huber: 2.4545, swd: 14.4189, ept: 134.6828
    Epoch [6/50], Val Losses: mse: 25.8285, mae: 3.2955, huber: 2.8669, swd: 10.5241, ept: 113.5200
    Epoch [6/50], Test Losses: mse: 18.0231, mae: 2.6697, huber: 2.2476, swd: 9.1805, ept: 131.9628
      Epoch 6 composite train-obj: 2.454524
            No improvement (2.8669), counter 1/5
    Epoch [7/50], Train Losses: mse: 28.0697, mae: 2.8398, huber: 2.4267, swd: 13.9375, ept: 135.1479
    Epoch [7/50], Val Losses: mse: 25.7343, mae: 3.3160, huber: 2.8869, swd: 10.8035, ept: 113.0932
    Epoch [7/50], Test Losses: mse: 17.2506, mae: 2.6497, huber: 2.2279, swd: 8.6428, ept: 133.2703
      Epoch 7 composite train-obj: 2.426747
            No improvement (2.8869), counter 2/5
    Epoch [8/50], Train Losses: mse: 27.8324, mae: 2.8302, huber: 2.4171, swd: 13.7680, ept: 135.1447
    Epoch [8/50], Val Losses: mse: 25.3826, mae: 3.2710, huber: 2.8434, swd: 10.2376, ept: 113.1882
    Epoch [8/50], Test Losses: mse: 17.8351, mae: 2.6729, huber: 2.2508, swd: 9.0273, ept: 131.7817
      Epoch 8 composite train-obj: 2.417075
            No improvement (2.8434), counter 3/5
    Epoch [9/50], Train Losses: mse: 27.7138, mae: 2.8094, huber: 2.3967, swd: 13.6644, ept: 135.3557
    Epoch [9/50], Val Losses: mse: 25.1318, mae: 3.2743, huber: 2.8454, swd: 10.1691, ept: 113.1478
    Epoch [9/50], Test Losses: mse: 17.3272, mae: 2.6537, huber: 2.2316, swd: 8.6530, ept: 134.0591
      Epoch 9 composite train-obj: 2.396726
            No improvement (2.8454), counter 4/5
    Epoch [10/50], Train Losses: mse: 27.3795, mae: 2.7959, huber: 2.3832, swd: 13.3979, ept: 135.7122
    Epoch [10/50], Val Losses: mse: 25.7133, mae: 3.3190, huber: 2.8893, swd: 10.7731, ept: 112.8214
    Epoch [10/50], Test Losses: mse: 17.0811, mae: 2.6520, huber: 2.2294, swd: 8.4492, ept: 134.2049
      Epoch 10 composite train-obj: 2.383231
    Epoch [10/50], Test Losses: mse: 17.2485, mae: 2.6486, huber: 2.2276, swd: 8.6281, ept: 132.5434
    Best round's Test MSE: 17.2485, MAE: 2.6486, SWD: 8.6281
    Best round's Validation MSE: 24.6858, MAE: 3.2525, SWD: 10.0555
    Best round's Test verification MSE : 17.2485, MAE: 2.6486, SWD: 8.6281
    Time taken: 14.94 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 32.1713, mae: 3.0661, huber: 2.6499, swd: 14.3365, ept: 129.1240
    Epoch [1/50], Val Losses: mse: 25.7916, mae: 3.3177, huber: 2.8887, swd: 9.2298, ept: 110.7974
    Epoch [1/50], Test Losses: mse: 17.3815, mae: 2.6443, huber: 2.2231, swd: 7.3396, ept: 128.7756
      Epoch 1 composite train-obj: 2.649892
            Val objective improved inf → 2.8887, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 30.8064, mae: 2.9789, huber: 2.5645, swd: 13.7913, ept: 132.9352
    Epoch [2/50], Val Losses: mse: 26.1413, mae: 3.3380, huber: 2.9084, swd: 9.4261, ept: 111.5764
    Epoch [2/50], Test Losses: mse: 17.3066, mae: 2.6426, huber: 2.2208, swd: 7.2447, ept: 130.5470
      Epoch 2 composite train-obj: 2.564518
            No improvement (2.9084), counter 1/5
    Epoch [3/50], Train Losses: mse: 29.9306, mae: 2.9363, huber: 2.5222, swd: 13.1875, ept: 133.6626
    Epoch [3/50], Val Losses: mse: 24.7454, mae: 3.2448, huber: 2.8154, swd: 8.2781, ept: 112.8890
    Epoch [3/50], Test Losses: mse: 17.2548, mae: 2.6490, huber: 2.2271, swd: 7.2647, ept: 131.8071
      Epoch 3 composite train-obj: 2.522177
            Val objective improved 2.8887 → 2.8154, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 29.2599, mae: 2.9018, huber: 2.4879, swd: 12.7623, ept: 134.1947
    Epoch [4/50], Val Losses: mse: 25.5876, mae: 3.3159, huber: 2.8843, swd: 8.9053, ept: 112.0337
    Epoch [4/50], Test Losses: mse: 17.4508, mae: 2.6577, huber: 2.2350, swd: 7.4416, ept: 132.0787
      Epoch 4 composite train-obj: 2.487899
            No improvement (2.8843), counter 1/5
    Epoch [5/50], Train Losses: mse: 28.8009, mae: 2.8753, huber: 2.4616, swd: 12.4773, ept: 134.6104
    Epoch [5/50], Val Losses: mse: 25.7470, mae: 3.3313, huber: 2.9011, swd: 9.4146, ept: 112.4409
    Epoch [5/50], Test Losses: mse: 17.1459, mae: 2.6577, huber: 2.2350, swd: 7.3280, ept: 133.0830
      Epoch 5 composite train-obj: 2.461574
            No improvement (2.9011), counter 2/5
    Epoch [6/50], Train Losses: mse: 28.4289, mae: 2.8570, huber: 2.4435, swd: 12.2428, ept: 134.9056
    Epoch [6/50], Val Losses: mse: 25.7984, mae: 3.3143, huber: 2.8842, swd: 8.9798, ept: 111.6494
    Epoch [6/50], Test Losses: mse: 17.2307, mae: 2.6494, huber: 2.2271, swd: 7.2437, ept: 132.4432
      Epoch 6 composite train-obj: 2.443475
            No improvement (2.8842), counter 3/5
    Epoch [7/50], Train Losses: mse: 28.1118, mae: 2.8379, huber: 2.4246, swd: 12.0134, ept: 134.9326
    Epoch [7/50], Val Losses: mse: 25.6001, mae: 3.2943, huber: 2.8654, swd: 9.0539, ept: 112.0393
    Epoch [7/50], Test Losses: mse: 17.5643, mae: 2.6747, huber: 2.2519, swd: 7.4696, ept: 132.0557
      Epoch 7 composite train-obj: 2.424581
            No improvement (2.8654), counter 4/5
    Epoch [8/50], Train Losses: mse: 27.8788, mae: 2.8220, huber: 2.4088, swd: 11.8576, ept: 135.2199
    Epoch [8/50], Val Losses: mse: 24.8466, mae: 3.2536, huber: 2.8250, swd: 8.3638, ept: 113.1308
    Epoch [8/50], Test Losses: mse: 17.4606, mae: 2.6702, huber: 2.2476, swd: 7.4693, ept: 132.3773
      Epoch 8 composite train-obj: 2.408826
    Epoch [8/50], Test Losses: mse: 17.2548, mae: 2.6490, huber: 2.2271, swd: 7.2647, ept: 131.8071
    Best round's Test MSE: 17.2548, MAE: 2.6490, SWD: 7.2647
    Best round's Validation MSE: 24.7454, MAE: 3.2448, SWD: 8.2781
    Best round's Test verification MSE : 17.2548, MAE: 2.6490, SWD: 7.2647
    Time taken: 11.81 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth2_seq96_pred196_20250511_1623)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 17.5537 ± 0.4272
      mae: 2.6667 ± 0.0254
      huber: 2.2450 ± 0.0250
      swd: 8.3456 ± 0.7928
      ept: 132.1585 ± 0.3015
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 24.9077 ± 0.2728
      mae: 3.2485 ± 0.0032
      huber: 2.8205 ± 0.0040
      swd: 9.4475 ± 0.8272
      ept: 113.2013 ± 0.2245
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 49.21 seconds
    
    Experiment complete: PatchTST_etth2_seq96_pred196_20250511_1623
    Model: PatchTST
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.5003, mae: 0.4127, huber: 0.1786, swd: 0.1831, ept: 185.7623
    Epoch [1/50], Val Losses: mse: 0.3755, mae: 0.4378, huber: 0.1753, swd: 0.1245, ept: 155.3961
    Epoch [1/50], Test Losses: mse: 0.2408, mae: 0.3444, huber: 0.1148, swd: 0.0859, ept: 189.7403
      Epoch 1 composite train-obj: 0.178558
            Val objective improved inf → 0.1753, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4824, mae: 0.4037, huber: 0.1725, swd: 0.1825, ept: 193.0328
    Epoch [2/50], Val Losses: mse: 0.3882, mae: 0.4391, huber: 0.1790, swd: 0.1475, ept: 157.9229
    Epoch [2/50], Test Losses: mse: 0.2321, mae: 0.3349, huber: 0.1107, swd: 0.0846, ept: 190.7971
      Epoch 2 composite train-obj: 0.172481
            No improvement (0.1790), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4693, mae: 0.3986, huber: 0.1686, swd: 0.1795, ept: 194.5762
    Epoch [3/50], Val Losses: mse: 0.4019, mae: 0.4507, huber: 0.1858, swd: 0.1366, ept: 158.1544
    Epoch [3/50], Test Losses: mse: 0.2481, mae: 0.3492, huber: 0.1178, swd: 0.0885, ept: 190.5246
      Epoch 3 composite train-obj: 0.168626
            No improvement (0.1858), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4597, mae: 0.3955, huber: 0.1661, swd: 0.1767, ept: 195.1220
    Epoch [4/50], Val Losses: mse: 0.3802, mae: 0.4353, huber: 0.1762, swd: 0.1466, ept: 157.9743
    Epoch [4/50], Test Losses: mse: 0.2284, mae: 0.3336, huber: 0.1092, swd: 0.0839, ept: 192.2609
      Epoch 4 composite train-obj: 0.166144
            No improvement (0.1762), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4502, mae: 0.3914, huber: 0.1632, swd: 0.1740, ept: 195.8152
    Epoch [5/50], Val Losses: mse: 0.3831, mae: 0.4311, huber: 0.1757, swd: 0.1519, ept: 158.0914
    Epoch [5/50], Test Losses: mse: 0.2304, mae: 0.3307, huber: 0.1095, swd: 0.0870, ept: 192.5703
      Epoch 5 composite train-obj: 0.163181
            No improvement (0.1757), counter 4/5
    Epoch [6/50], Train Losses: mse: 0.4447, mae: 0.3891, huber: 0.1614, swd: 0.1726, ept: 196.6698
    Epoch [6/50], Val Losses: mse: 0.3704, mae: 0.4282, huber: 0.1718, swd: 0.1376, ept: 160.3114
    Epoch [6/50], Test Losses: mse: 0.2267, mae: 0.3306, huber: 0.1082, swd: 0.0817, ept: 194.1953
      Epoch 6 composite train-obj: 0.161384
            Val objective improved 0.1753 → 0.1718, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4395, mae: 0.3868, huber: 0.1597, swd: 0.1716, ept: 196.6708
    Epoch [7/50], Val Losses: mse: 0.3941, mae: 0.4426, huber: 0.1822, swd: 0.1497, ept: 159.0842
    Epoch [7/50], Test Losses: mse: 0.2329, mae: 0.3359, huber: 0.1112, swd: 0.0818, ept: 192.4472
      Epoch 7 composite train-obj: 0.159710
            No improvement (0.1822), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4379, mae: 0.3859, huber: 0.1590, swd: 0.1708, ept: 196.9519
    Epoch [8/50], Val Losses: mse: 0.3903, mae: 0.4380, huber: 0.1792, swd: 0.1549, ept: 157.3056
    Epoch [8/50], Test Losses: mse: 0.2282, mae: 0.3335, huber: 0.1091, swd: 0.0823, ept: 193.5653
      Epoch 8 composite train-obj: 0.158988
            No improvement (0.1792), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.4316, mae: 0.3832, huber: 0.1570, swd: 0.1688, ept: 197.1968
    Epoch [9/50], Val Losses: mse: 0.3792, mae: 0.4330, huber: 0.1754, swd: 0.1476, ept: 159.3737
    Epoch [9/50], Test Losses: mse: 0.2278, mae: 0.3331, huber: 0.1092, swd: 0.0835, ept: 193.4430
      Epoch 9 composite train-obj: 0.156983
            No improvement (0.1754), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.4281, mae: 0.3812, huber: 0.1556, swd: 0.1680, ept: 198.0260
    Epoch [10/50], Val Losses: mse: 0.3891, mae: 0.4372, huber: 0.1788, swd: 0.1515, ept: 159.2970
    Epoch [10/50], Test Losses: mse: 0.2264, mae: 0.3323, huber: 0.1086, swd: 0.0844, ept: 194.2564
      Epoch 10 composite train-obj: 0.155600
            No improvement (0.1788), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.4258, mae: 0.3805, huber: 0.1550, swd: 0.1669, ept: 197.8293
    Epoch [11/50], Val Losses: mse: 0.3854, mae: 0.4368, huber: 0.1779, swd: 0.1476, ept: 159.0199
    Epoch [11/50], Test Losses: mse: 0.2285, mae: 0.3334, huber: 0.1094, swd: 0.0842, ept: 193.4971
      Epoch 11 composite train-obj: 0.155025
    Epoch [11/50], Test Losses: mse: 0.2267, mae: 0.3306, huber: 0.1082, swd: 0.0817, ept: 194.1953
    Best round's Test MSE: 0.2267, MAE: 0.3306, SWD: 0.0817
    Best round's Validation MSE: 0.3704, MAE: 0.4282, SWD: 0.1376
    Best round's Test verification MSE : 0.2267, MAE: 0.3306, SWD: 0.0817
    Time taken: 14.86 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5009, mae: 0.4128, huber: 0.1788, swd: 0.1900, ept: 185.9030
    Epoch [1/50], Val Losses: mse: 0.3716, mae: 0.4313, huber: 0.1728, swd: 0.1363, ept: 157.7774
    Epoch [1/50], Test Losses: mse: 0.2292, mae: 0.3310, huber: 0.1092, swd: 0.0884, ept: 190.1954
      Epoch 1 composite train-obj: 0.178828
            Val objective improved inf → 0.1728, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4776, mae: 0.4025, huber: 0.1716, swd: 0.1871, ept: 192.7945
    Epoch [2/50], Val Losses: mse: 0.4057, mae: 0.4483, huber: 0.1863, swd: 0.1628, ept: 157.6366
    Epoch [2/50], Test Losses: mse: 0.2290, mae: 0.3314, huber: 0.1090, swd: 0.0883, ept: 191.8145
      Epoch 2 composite train-obj: 0.171573
            No improvement (0.1863), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4651, mae: 0.3980, huber: 0.1679, swd: 0.1845, ept: 194.5491
    Epoch [3/50], Val Losses: mse: 0.3902, mae: 0.4395, huber: 0.1797, swd: 0.1566, ept: 156.7687
    Epoch [3/50], Test Losses: mse: 0.2280, mae: 0.3322, huber: 0.1090, swd: 0.0881, ept: 191.2425
      Epoch 3 composite train-obj: 0.167916
            No improvement (0.1797), counter 2/5
    Epoch [4/50], Train Losses: mse: 0.4551, mae: 0.3943, huber: 0.1650, swd: 0.1828, ept: 194.5688
    Epoch [4/50], Val Losses: mse: 0.3885, mae: 0.4392, huber: 0.1796, swd: 0.1542, ept: 157.1412
    Epoch [4/50], Test Losses: mse: 0.2277, mae: 0.3336, huber: 0.1090, swd: 0.0866, ept: 191.8470
      Epoch 4 composite train-obj: 0.165020
            No improvement (0.1796), counter 3/5
    Epoch [5/50], Train Losses: mse: 0.4489, mae: 0.3910, huber: 0.1626, swd: 0.1807, ept: 195.6087
    Epoch [5/50], Val Losses: mse: 0.3709, mae: 0.4299, huber: 0.1721, swd: 0.1435, ept: 159.0745
    Epoch [5/50], Test Losses: mse: 0.2357, mae: 0.3411, huber: 0.1128, swd: 0.0899, ept: 191.2781
      Epoch 5 composite train-obj: 0.162634
            Val objective improved 0.1728 → 0.1721, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4446, mae: 0.3890, huber: 0.1613, swd: 0.1795, ept: 196.1988
    Epoch [6/50], Val Losses: mse: 0.3872, mae: 0.4417, huber: 0.1800, swd: 0.1489, ept: 159.9202
    Epoch [6/50], Test Losses: mse: 0.2350, mae: 0.3393, huber: 0.1123, swd: 0.0885, ept: 192.2970
      Epoch 6 composite train-obj: 0.161251
            No improvement (0.1800), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.4363, mae: 0.3856, huber: 0.1586, swd: 0.1774, ept: 196.8672
    Epoch [7/50], Val Losses: mse: 0.3846, mae: 0.4398, huber: 0.1783, swd: 0.1444, ept: 159.2317
    Epoch [7/50], Test Losses: mse: 0.2402, mae: 0.3454, huber: 0.1150, swd: 0.0924, ept: 191.6288
      Epoch 7 composite train-obj: 0.158579
            No improvement (0.1783), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.4347, mae: 0.3855, huber: 0.1582, swd: 0.1772, ept: 196.8579
    Epoch [8/50], Val Losses: mse: 0.3951, mae: 0.4413, huber: 0.1813, swd: 0.1651, ept: 156.9347
    Epoch [8/50], Test Losses: mse: 0.2254, mae: 0.3289, huber: 0.1074, swd: 0.0860, ept: 193.5944
      Epoch 8 composite train-obj: 0.158240
            No improvement (0.1813), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.4290, mae: 0.3827, huber: 0.1562, swd: 0.1746, ept: 197.2122
    Epoch [9/50], Val Losses: mse: 0.3984, mae: 0.4449, huber: 0.1833, swd: 0.1618, ept: 158.4260
    Epoch [9/50], Test Losses: mse: 0.2292, mae: 0.3340, huber: 0.1096, swd: 0.0873, ept: 192.7732
      Epoch 9 composite train-obj: 0.156216
            No improvement (0.1833), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.4258, mae: 0.3810, huber: 0.1550, swd: 0.1745, ept: 197.7869
    Epoch [10/50], Val Losses: mse: 0.3903, mae: 0.4405, huber: 0.1799, swd: 0.1541, ept: 158.6066
    Epoch [10/50], Test Losses: mse: 0.2330, mae: 0.3368, huber: 0.1113, swd: 0.0885, ept: 192.7492
      Epoch 10 composite train-obj: 0.155004
    Epoch [10/50], Test Losses: mse: 0.2357, mae: 0.3411, huber: 0.1128, swd: 0.0899, ept: 191.2781
    Best round's Test MSE: 0.2357, MAE: 0.3411, SWD: 0.0899
    Best round's Validation MSE: 0.3709, MAE: 0.4299, SWD: 0.1435
    Best round's Test verification MSE : 0.2357, MAE: 0.3411, SWD: 0.0899
    Time taken: 13.11 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4995, mae: 0.4126, huber: 0.1783, swd: 0.1843, ept: 185.8268
    Epoch [1/50], Val Losses: mse: 0.3787, mae: 0.4340, huber: 0.1757, swd: 0.1375, ept: 158.2532
    Epoch [1/50], Test Losses: mse: 0.2287, mae: 0.3310, huber: 0.1090, swd: 0.0791, ept: 190.3570
      Epoch 1 composite train-obj: 0.178303
            Val objective improved inf → 0.1757, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4808, mae: 0.4026, huber: 0.1718, swd: 0.1837, ept: 193.5261
    Epoch [2/50], Val Losses: mse: 0.3885, mae: 0.4416, huber: 0.1802, swd: 0.1411, ept: 158.4413
    Epoch [2/50], Test Losses: mse: 0.2320, mae: 0.3352, huber: 0.1106, swd: 0.0798, ept: 190.7360
      Epoch 2 composite train-obj: 0.171805
            No improvement (0.1802), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4656, mae: 0.3977, huber: 0.1677, swd: 0.1800, ept: 194.9147
    Epoch [3/50], Val Losses: mse: 0.3745, mae: 0.4301, huber: 0.1732, swd: 0.1447, ept: 158.5463
    Epoch [3/50], Test Losses: mse: 0.2256, mae: 0.3302, huber: 0.1079, swd: 0.0784, ept: 192.9172
      Epoch 3 composite train-obj: 0.167719
            Val objective improved 0.1757 → 0.1732, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4576, mae: 0.3944, huber: 0.1653, swd: 0.1779, ept: 195.6456
    Epoch [4/50], Val Losses: mse: 0.3685, mae: 0.4277, huber: 0.1710, swd: 0.1409, ept: 158.0231
    Epoch [4/50], Test Losses: mse: 0.2251, mae: 0.3295, huber: 0.1075, swd: 0.0782, ept: 192.5866
      Epoch 4 composite train-obj: 0.165275
            Val objective improved 0.1732 → 0.1710, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4490, mae: 0.3908, huber: 0.1626, swd: 0.1765, ept: 196.1459
    Epoch [5/50], Val Losses: mse: 0.3747, mae: 0.4306, huber: 0.1734, swd: 0.1441, ept: 158.7316
    Epoch [5/50], Test Losses: mse: 0.2276, mae: 0.3332, huber: 0.1090, swd: 0.0777, ept: 192.7119
      Epoch 5 composite train-obj: 0.162569
            No improvement (0.1734), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4430, mae: 0.3880, huber: 0.1604, swd: 0.1747, ept: 196.8223
    Epoch [6/50], Val Losses: mse: 0.3811, mae: 0.4310, huber: 0.1751, swd: 0.1521, ept: 157.4065
    Epoch [6/50], Test Losses: mse: 0.2280, mae: 0.3298, huber: 0.1084, swd: 0.0802, ept: 193.7838
      Epoch 6 composite train-obj: 0.160440
            No improvement (0.1751), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.4385, mae: 0.3863, huber: 0.1591, swd: 0.1734, ept: 196.9529
    Epoch [7/50], Val Losses: mse: 0.3865, mae: 0.4379, huber: 0.1785, swd: 0.1506, ept: 158.9115
    Epoch [7/50], Test Losses: mse: 0.2274, mae: 0.3327, huber: 0.1087, swd: 0.0798, ept: 194.6049
      Epoch 7 composite train-obj: 0.159139
            No improvement (0.1785), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.4339, mae: 0.3843, huber: 0.1576, swd: 0.1718, ept: 197.1869
    Epoch [8/50], Val Losses: mse: 0.3791, mae: 0.4310, huber: 0.1747, swd: 0.1527, ept: 158.7663
    Epoch [8/50], Test Losses: mse: 0.2260, mae: 0.3302, huber: 0.1080, swd: 0.0798, ept: 194.2973
      Epoch 8 composite train-obj: 0.157592
            No improvement (0.1747), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.4322, mae: 0.3834, huber: 0.1569, swd: 0.1717, ept: 197.7152
    Epoch [9/50], Val Losses: mse: 0.3784, mae: 0.4330, huber: 0.1750, swd: 0.1423, ept: 159.0165
    Epoch [9/50], Test Losses: mse: 0.2319, mae: 0.3370, huber: 0.1108, swd: 0.0800, ept: 193.6221
      Epoch 9 composite train-obj: 0.156944
    Epoch [9/50], Test Losses: mse: 0.2251, mae: 0.3295, huber: 0.1075, swd: 0.0782, ept: 192.5866
    Best round's Test MSE: 0.2251, MAE: 0.3295, SWD: 0.0782
    Best round's Validation MSE: 0.3685, MAE: 0.4277, SWD: 0.1409
    Best round's Test verification MSE : 0.2251, MAE: 0.3295, SWD: 0.0782
    Time taken: 12.06 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth2_seq96_pred336_20250510_2050)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2292 ± 0.0047
      mae: 0.3337 ± 0.0053
      huber: 0.1095 ± 0.0024
      swd: 0.0833 ± 0.0049
      ept: 192.6867 ± 1.1931
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3699 ± 0.0010
      mae: 0.4286 ± 0.0009
      huber: 0.1716 ± 0.0005
      swd: 0.1407 ± 0.0024
      ept: 159.1363 ± 0.9352
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 40.09 seconds
    
    Experiment complete: PatchTST_etth2_seq96_pred336_20250510_2050
    Model: PatchTST
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 11
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 40.9733, mae: 3.4329, huber: 3.0098, swd: 20.7436, ept: 185.3596
    Epoch [1/50], Val Losses: mse: 27.8378, mae: 3.5583, huber: 3.1205, swd: 10.2247, ept: 158.0500
    Epoch [1/50], Test Losses: mse: 19.0957, mae: 2.8543, huber: 2.4268, swd: 9.2026, ept: 191.9080
      Epoch 1 composite train-obj: 3.009830
            Val objective improved inf → 3.1205, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.4449, mae: 3.3471, huber: 2.9252, swd: 19.8857, ept: 192.4183
    Epoch [2/50], Val Losses: mse: 28.4563, mae: 3.5725, huber: 3.1345, swd: 10.1607, ept: 159.4872
    Epoch [2/50], Test Losses: mse: 18.8526, mae: 2.8023, huber: 2.3768, swd: 8.6908, ept: 191.4895
      Epoch 2 composite train-obj: 2.925225
            No improvement (3.1345), counter 1/5
    Epoch [3/50], Train Losses: mse: 38.4110, mae: 3.3093, huber: 2.8876, swd: 19.0453, ept: 193.1692
    Epoch [3/50], Val Losses: mse: 29.3466, mae: 3.6374, huber: 3.1985, swd: 11.2065, ept: 159.7876
    Epoch [3/50], Test Losses: mse: 19.3902, mae: 2.8696, huber: 2.4421, swd: 9.0935, ept: 191.1581
      Epoch 3 composite train-obj: 2.887578
            No improvement (3.1985), counter 2/5
    Epoch [4/50], Train Losses: mse: 37.7362, mae: 3.2841, huber: 2.8624, swd: 18.5382, ept: 193.6205
    Epoch [4/50], Val Losses: mse: 27.7393, mae: 3.5432, huber: 3.1061, swd: 9.7041, ept: 159.5728
    Epoch [4/50], Test Losses: mse: 18.9157, mae: 2.8039, huber: 2.3785, swd: 8.8786, ept: 192.8506
      Epoch 4 composite train-obj: 2.862394
            Val objective improved 3.1205 → 3.1061, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 36.8967, mae: 3.2490, huber: 2.8275, swd: 17.8909, ept: 194.3107
    Epoch [5/50], Val Losses: mse: 28.2861, mae: 3.5277, huber: 3.0904, swd: 9.9101, ept: 157.5144
    Epoch [5/50], Test Losses: mse: 19.2010, mae: 2.8127, huber: 2.3873, swd: 9.0282, ept: 192.1946
      Epoch 5 composite train-obj: 2.827481
            Val objective improved 3.1061 → 3.0904, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 36.4342, mae: 3.2207, huber: 2.7994, swd: 17.5200, ept: 194.9659
    Epoch [6/50], Val Losses: mse: 28.0214, mae: 3.5354, huber: 3.0969, swd: 9.7776, ept: 159.2774
    Epoch [6/50], Test Losses: mse: 18.7356, mae: 2.8019, huber: 2.3758, swd: 8.5174, ept: 192.8641
      Epoch 6 composite train-obj: 2.799352
            No improvement (3.0969), counter 1/5
    Epoch [7/50], Train Losses: mse: 35.9394, mae: 3.1997, huber: 2.7787, swd: 17.1755, ept: 195.3820
    Epoch [7/50], Val Losses: mse: 28.9408, mae: 3.6115, huber: 3.1730, swd: 10.4663, ept: 158.2445
    Epoch [7/50], Test Losses: mse: 18.7689, mae: 2.7886, huber: 2.3632, swd: 8.6047, ept: 194.3645
      Epoch 7 composite train-obj: 2.778659
            No improvement (3.1730), counter 2/5
    Epoch [8/50], Train Losses: mse: 35.6454, mae: 3.1800, huber: 2.7592, swd: 16.9271, ept: 195.5695
    Epoch [8/50], Val Losses: mse: 28.3162, mae: 3.5259, huber: 3.0896, swd: 9.8374, ept: 158.3186
    Epoch [8/50], Test Losses: mse: 18.7596, mae: 2.7891, huber: 2.3641, swd: 8.6637, ept: 194.8948
      Epoch 8 composite train-obj: 2.759205
            Val objective improved 3.0904 → 3.0896, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 35.2402, mae: 3.1564, huber: 2.7358, swd: 16.6096, ept: 196.0725
    Epoch [9/50], Val Losses: mse: 28.6846, mae: 3.5717, huber: 3.1341, swd: 10.1737, ept: 158.1084
    Epoch [9/50], Test Losses: mse: 19.0586, mae: 2.8141, huber: 2.3884, swd: 8.9542, ept: 194.3695
      Epoch 9 composite train-obj: 2.735805
            No improvement (3.1341), counter 1/5
    Epoch [10/50], Train Losses: mse: 34.9707, mae: 3.1344, huber: 2.7142, swd: 16.3648, ept: 196.4830
    Epoch [10/50], Val Losses: mse: 28.5422, mae: 3.5586, huber: 3.1214, swd: 10.2025, ept: 157.7657
    Epoch [10/50], Test Losses: mse: 19.2048, mae: 2.8399, huber: 2.4135, swd: 9.1575, ept: 195.2965
      Epoch 10 composite train-obj: 2.714226
            No improvement (3.1214), counter 2/5
    Epoch [11/50], Train Losses: mse: 34.8397, mae: 3.1279, huber: 2.7078, swd: 16.2792, ept: 196.4574
    Epoch [11/50], Val Losses: mse: 29.4462, mae: 3.5868, huber: 3.1498, swd: 10.9143, ept: 157.6790
    Epoch [11/50], Test Losses: mse: 19.2814, mae: 2.8327, huber: 2.4069, swd: 9.1340, ept: 194.3716
      Epoch 11 composite train-obj: 2.707792
            No improvement (3.1498), counter 3/5
    Epoch [12/50], Train Losses: mse: 34.3198, mae: 3.0970, huber: 2.6771, swd: 15.8544, ept: 197.1560
    Epoch [12/50], Val Losses: mse: 28.5468, mae: 3.5596, huber: 3.1217, swd: 10.2877, ept: 157.1592
    Epoch [12/50], Test Losses: mse: 19.4965, mae: 2.8650, huber: 2.4381, swd: 9.2627, ept: 193.5592
      Epoch 12 composite train-obj: 2.677064
            No improvement (3.1217), counter 4/5
    Epoch [13/50], Train Losses: mse: 34.0986, mae: 3.0813, huber: 2.6616, swd: 15.6944, ept: 197.1559
    Epoch [13/50], Val Losses: mse: 29.8851, mae: 3.6294, huber: 3.1913, swd: 11.1296, ept: 157.4678
    Epoch [13/50], Test Losses: mse: 19.3959, mae: 2.8431, huber: 2.4165, swd: 9.1480, ept: 193.8606
      Epoch 13 composite train-obj: 2.661619
    Epoch [13/50], Test Losses: mse: 18.7596, mae: 2.7891, huber: 2.3641, swd: 8.6637, ept: 194.8948
    Best round's Test MSE: 18.7596, MAE: 2.7891, SWD: 8.6637
    Best round's Validation MSE: 28.3162, MAE: 3.5259, SWD: 9.8374
    Best round's Test verification MSE : 18.7596, MAE: 2.7891, SWD: 8.6637
    Time taken: 19.46 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 40.9974, mae: 3.4374, huber: 3.0143, swd: 21.7475, ept: 185.4106
    Epoch [1/50], Val Losses: mse: 27.4981, mae: 3.5199, huber: 3.0841, swd: 9.7076, ept: 157.7892
    Epoch [1/50], Test Losses: mse: 19.3603, mae: 2.8528, huber: 2.4262, swd: 9.3477, ept: 189.2247
      Epoch 1 composite train-obj: 3.014286
            Val objective improved inf → 3.0841, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.4415, mae: 3.3486, huber: 2.9269, swd: 20.9020, ept: 192.1624
    Epoch [2/50], Val Losses: mse: 28.8458, mae: 3.5847, huber: 3.1476, swd: 11.2066, ept: 159.3449
    Epoch [2/50], Test Losses: mse: 18.9642, mae: 2.8108, huber: 2.3853, swd: 9.1853, ept: 191.4887
      Epoch 2 composite train-obj: 2.926938
            No improvement (3.1476), counter 1/5
    Epoch [3/50], Train Losses: mse: 38.1803, mae: 3.3030, huber: 2.8816, swd: 19.7923, ept: 193.3838
    Epoch [3/50], Val Losses: mse: 28.9128, mae: 3.5773, huber: 3.1395, swd: 10.6452, ept: 157.3943
    Epoch [3/50], Test Losses: mse: 19.2492, mae: 2.8082, huber: 2.3825, swd: 9.2713, ept: 192.2873
      Epoch 3 composite train-obj: 2.881554
            No improvement (3.1395), counter 2/5
    Epoch [4/50], Train Losses: mse: 37.3710, mae: 3.2757, huber: 2.8543, swd: 19.0341, ept: 193.3554
    Epoch [4/50], Val Losses: mse: 28.2017, mae: 3.5435, huber: 3.1072, swd: 10.3284, ept: 159.3766
    Epoch [4/50], Test Losses: mse: 18.6781, mae: 2.7736, huber: 2.3487, swd: 8.7980, ept: 193.9461
      Epoch 4 composite train-obj: 2.854316
            No improvement (3.1072), counter 3/5
    Epoch [5/50], Train Losses: mse: 36.6666, mae: 3.2384, huber: 2.8175, swd: 18.4944, ept: 194.5066
    Epoch [5/50], Val Losses: mse: 28.1818, mae: 3.5359, huber: 3.1004, swd: 10.2458, ept: 158.4748
    Epoch [5/50], Test Losses: mse: 18.8661, mae: 2.7973, huber: 2.3719, swd: 9.0475, ept: 192.6665
      Epoch 5 composite train-obj: 2.817453
            No improvement (3.1004), counter 4/5
    Epoch [6/50], Train Losses: mse: 36.2390, mae: 3.2163, huber: 2.7955, swd: 18.1345, ept: 194.8803
    Epoch [6/50], Val Losses: mse: 28.1398, mae: 3.5565, huber: 3.1194, swd: 10.5891, ept: 158.7757
    Epoch [6/50], Test Losses: mse: 19.2857, mae: 2.8371, huber: 2.4110, swd: 9.4944, ept: 191.6692
      Epoch 6 composite train-obj: 2.795537
    Epoch [6/50], Test Losses: mse: 19.3603, mae: 2.8528, huber: 2.4262, swd: 9.3477, ept: 189.2247
    Best round's Test MSE: 19.3603, MAE: 2.8528, SWD: 9.3477
    Best round's Validation MSE: 27.4981, MAE: 3.5199, SWD: 9.7076
    Best round's Test verification MSE : 19.3603, MAE: 2.8528, SWD: 9.3477
    Time taken: 9.01 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 40.8602, mae: 3.4308, huber: 3.0076, swd: 20.0419, ept: 185.4080
    Epoch [1/50], Val Losses: mse: 28.6552, mae: 3.5699, huber: 3.1336, swd: 9.8541, ept: 156.3881
    Epoch [1/50], Test Losses: mse: 18.8339, mae: 2.7872, huber: 2.3624, swd: 8.0492, ept: 188.3052
      Epoch 1 composite train-obj: 3.007597
            Val objective improved inf → 3.1336, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 39.3919, mae: 3.3424, huber: 2.9209, swd: 19.2891, ept: 192.2762
    Epoch [2/50], Val Losses: mse: 29.1088, mae: 3.6161, huber: 3.1781, swd: 10.6699, ept: 157.2764
    Epoch [2/50], Test Losses: mse: 19.1392, mae: 2.8336, huber: 2.4074, swd: 8.4463, ept: 189.0160
      Epoch 2 composite train-obj: 2.920908
            No improvement (3.1781), counter 1/5
    Epoch [3/50], Train Losses: mse: 38.2095, mae: 3.3036, huber: 2.8822, swd: 18.3251, ept: 193.3845
    Epoch [3/50], Val Losses: mse: 27.2936, mae: 3.5142, huber: 3.0773, swd: 8.9217, ept: 158.4629
    Epoch [3/50], Test Losses: mse: 18.6232, mae: 2.7874, huber: 2.3622, swd: 8.0195, ept: 191.7156
      Epoch 3 composite train-obj: 2.882167
            Val objective improved 3.1336 → 3.0773, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.3431, mae: 3.2706, huber: 2.8492, swd: 17.6772, ept: 193.9292
    Epoch [4/50], Val Losses: mse: 27.1081, mae: 3.4856, huber: 3.0483, swd: 8.7006, ept: 158.5970
    Epoch [4/50], Test Losses: mse: 18.9996, mae: 2.8074, huber: 2.3820, swd: 8.2951, ept: 192.4739
      Epoch 4 composite train-obj: 2.849195
            Val objective improved 3.0773 → 3.0483, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 36.5975, mae: 3.2398, huber: 2.8185, swd: 17.1040, ept: 194.3459
    Epoch [5/50], Val Losses: mse: 27.8005, mae: 3.5145, huber: 3.0777, swd: 9.0566, ept: 158.8305
    Epoch [5/50], Test Losses: mse: 18.9738, mae: 2.7822, huber: 2.3577, swd: 8.1886, ept: 193.1934
      Epoch 5 composite train-obj: 2.818463
            No improvement (3.0777), counter 1/5
    Epoch [6/50], Train Losses: mse: 36.2926, mae: 3.2205, huber: 2.7994, swd: 16.9243, ept: 194.9049
    Epoch [6/50], Val Losses: mse: 28.1369, mae: 3.5235, huber: 3.0866, swd: 9.3139, ept: 158.0248
    Epoch [6/50], Test Losses: mse: 19.3720, mae: 2.8105, huber: 2.3856, swd: 8.5461, ept: 193.2244
      Epoch 6 composite train-obj: 2.799408
            No improvement (3.0866), counter 2/5
    Epoch [7/50], Train Losses: mse: 35.7847, mae: 3.1950, huber: 2.7741, swd: 16.5267, ept: 195.2408
    Epoch [7/50], Val Losses: mse: 27.8723, mae: 3.5223, huber: 3.0847, swd: 9.2610, ept: 158.8139
    Epoch [7/50], Test Losses: mse: 19.3013, mae: 2.8224, huber: 2.3972, swd: 8.4239, ept: 193.3653
      Epoch 7 composite train-obj: 2.774127
            No improvement (3.0847), counter 3/5
    Epoch [8/50], Train Losses: mse: 35.3397, mae: 3.1725, huber: 2.7520, swd: 16.1529, ept: 195.8086
    Epoch [8/50], Val Losses: mse: 28.0756, mae: 3.5300, huber: 3.0930, swd: 9.5972, ept: 158.3901
    Epoch [8/50], Test Losses: mse: 19.3033, mae: 2.8317, huber: 2.4057, swd: 8.5462, ept: 192.5466
      Epoch 8 composite train-obj: 2.751952
            No improvement (3.0930), counter 4/5
    Epoch [9/50], Train Losses: mse: 34.9485, mae: 3.1452, huber: 2.7249, swd: 15.8835, ept: 196.4949
    Epoch [9/50], Val Losses: mse: 28.3276, mae: 3.5403, huber: 3.1021, swd: 9.5904, ept: 158.2092
    Epoch [9/50], Test Losses: mse: 19.0190, mae: 2.8189, huber: 2.3927, swd: 8.1981, ept: 194.8481
      Epoch 9 composite train-obj: 2.724891
    Epoch [9/50], Test Losses: mse: 18.9996, mae: 2.8074, huber: 2.3820, swd: 8.2951, ept: 192.4739
    Best round's Test MSE: 18.9996, MAE: 2.8074, SWD: 8.2951
    Best round's Validation MSE: 27.1081, MAE: 3.4856, SWD: 8.7006
    Best round's Test verification MSE : 18.9996, MAE: 2.8074, SWD: 8.2951
    Time taken: 14.19 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth2_seq96_pred336_20250511_1624)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 19.0398 ± 0.2469
      mae: 2.8164 ± 0.0268
      huber: 2.3907 ± 0.0261
      swd: 8.7688 ± 0.4361
      ept: 192.1978 ± 2.3230
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 27.6408 ± 0.5034
      mae: 3.5105 ± 0.0177
      huber: 3.0740 ± 0.0183
      swd: 9.4152 ± 0.5081
      ept: 158.2349 ± 0.3351
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 42.72 seconds
    
    Experiment complete: PatchTST_etth2_seq96_pred336_20250511_1624
    Model: PatchTST
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.7117, mae: 0.5079, huber: 0.2460, swd: 0.2738, ept: 286.6974
    Epoch [1/50], Val Losses: mse: 0.4227, mae: 0.4720, huber: 0.1962, swd: 0.0992, ept: 270.1906
    Epoch [1/50], Test Losses: mse: 0.3009, mae: 0.3839, huber: 0.1410, swd: 0.1093, ept: 309.8539
      Epoch 1 composite train-obj: 0.246034
            Val objective improved inf → 0.1962, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6940, mae: 0.4996, huber: 0.2401, swd: 0.2743, ept: 299.5592
    Epoch [2/50], Val Losses: mse: 0.3968, mae: 0.4548, huber: 0.1851, swd: 0.1091, ept: 269.9595
    Epoch [2/50], Test Losses: mse: 0.2783, mae: 0.3670, huber: 0.1311, swd: 0.0990, ept: 310.6927
      Epoch 2 composite train-obj: 0.240145
            Val objective improved 0.1962 → 0.1851, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6836, mae: 0.4949, huber: 0.2368, swd: 0.2700, ept: 301.0930
    Epoch [3/50], Val Losses: mse: 0.4089, mae: 0.4624, huber: 0.1899, swd: 0.1150, ept: 271.7343
    Epoch [3/50], Test Losses: mse: 0.2840, mae: 0.3717, huber: 0.1335, swd: 0.1020, ept: 314.5572
      Epoch 3 composite train-obj: 0.236808
            No improvement (0.1899), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.6688, mae: 0.4895, huber: 0.2324, swd: 0.2660, ept: 302.0762
    Epoch [4/50], Val Losses: mse: 0.3884, mae: 0.4525, huber: 0.1819, swd: 0.1029, ept: 270.4390
    Epoch [4/50], Test Losses: mse: 0.2906, mae: 0.3778, huber: 0.1366, swd: 0.1019, ept: 308.6594
      Epoch 4 composite train-obj: 0.232357
            Val objective improved 0.1851 → 0.1819, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.6627, mae: 0.4861, huber: 0.2300, swd: 0.2639, ept: 303.3700
    Epoch [5/50], Val Losses: mse: 0.3963, mae: 0.4572, huber: 0.1852, swd: 0.1089, ept: 267.0707
    Epoch [5/50], Test Losses: mse: 0.2957, mae: 0.3817, huber: 0.1391, swd: 0.1054, ept: 312.2226
      Epoch 5 composite train-obj: 0.229968
            No improvement (0.1852), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.6542, mae: 0.4825, huber: 0.2272, swd: 0.2608, ept: 303.8159
    Epoch [6/50], Val Losses: mse: 0.4251, mae: 0.4722, huber: 0.1969, swd: 0.1209, ept: 270.9731
    Epoch [6/50], Test Losses: mse: 0.3067, mae: 0.3910, huber: 0.1441, swd: 0.1127, ept: 312.5919
      Epoch 6 composite train-obj: 0.227210
            No improvement (0.1969), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.6501, mae: 0.4807, huber: 0.2259, swd: 0.2598, ept: 304.0229
    Epoch [7/50], Val Losses: mse: 0.4066, mae: 0.4608, huber: 0.1890, swd: 0.1144, ept: 267.1096
    Epoch [7/50], Test Losses: mse: 0.2995, mae: 0.3835, huber: 0.1407, swd: 0.1072, ept: 311.8878
      Epoch 7 composite train-obj: 0.225933
            No improvement (0.1890), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.6428, mae: 0.4770, huber: 0.2232, swd: 0.2577, ept: 304.8487
    Epoch [8/50], Val Losses: mse: 0.3937, mae: 0.4541, huber: 0.1838, swd: 0.1129, ept: 269.8356
    Epoch [8/50], Test Losses: mse: 0.2922, mae: 0.3780, huber: 0.1374, swd: 0.1057, ept: 312.9778
      Epoch 8 composite train-obj: 0.223179
            No improvement (0.1838), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.6383, mae: 0.4753, huber: 0.2217, swd: 0.2569, ept: 305.8001
    Epoch [9/50], Val Losses: mse: 0.3882, mae: 0.4517, huber: 0.1815, swd: 0.1258, ept: 266.9878
    Epoch [9/50], Test Losses: mse: 0.2940, mae: 0.3796, huber: 0.1385, swd: 0.1067, ept: 314.0053
      Epoch 9 composite train-obj: 0.221739
            Val objective improved 0.1819 → 0.1815, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.6333, mae: 0.4729, huber: 0.2199, swd: 0.2551, ept: 305.4504
    Epoch [10/50], Val Losses: mse: 0.3858, mae: 0.4495, huber: 0.1803, swd: 0.1161, ept: 270.3221
    Epoch [10/50], Test Losses: mse: 0.2941, mae: 0.3805, huber: 0.1388, swd: 0.1093, ept: 313.6823
      Epoch 10 composite train-obj: 0.219876
            Val objective improved 0.1815 → 0.1803, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.6282, mae: 0.4705, huber: 0.2181, swd: 0.2531, ept: 305.9920
    Epoch [11/50], Val Losses: mse: 0.3987, mae: 0.4567, huber: 0.1856, swd: 0.1239, ept: 271.6614
    Epoch [11/50], Test Losses: mse: 0.2957, mae: 0.3820, huber: 0.1395, swd: 0.1107, ept: 314.5257
      Epoch 11 composite train-obj: 0.218141
            No improvement (0.1856), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.6231, mae: 0.4680, huber: 0.2163, swd: 0.2516, ept: 305.9345
    Epoch [12/50], Val Losses: mse: 0.3815, mae: 0.4463, huber: 0.1784, swd: 0.1190, ept: 270.9126
    Epoch [12/50], Test Losses: mse: 0.2946, mae: 0.3800, huber: 0.1388, swd: 0.1088, ept: 315.7448
      Epoch 12 composite train-obj: 0.216327
            Val objective improved 0.1803 → 0.1784, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.6184, mae: 0.4658, huber: 0.2145, swd: 0.2499, ept: 306.2564
    Epoch [13/50], Val Losses: mse: 0.4010, mae: 0.4564, huber: 0.1858, swd: 0.1226, ept: 270.2477
    Epoch [13/50], Test Losses: mse: 0.2957, mae: 0.3799, huber: 0.1389, swd: 0.1095, ept: 314.1836
      Epoch 13 composite train-obj: 0.214542
            No improvement (0.1858), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.6157, mae: 0.4645, huber: 0.2137, swd: 0.2487, ept: 306.6666
    Epoch [14/50], Val Losses: mse: 0.4137, mae: 0.4652, huber: 0.1916, swd: 0.1344, ept: 268.6256
    Epoch [14/50], Test Losses: mse: 0.2924, mae: 0.3806, huber: 0.1379, swd: 0.1072, ept: 317.0105
      Epoch 14 composite train-obj: 0.213697
            No improvement (0.1916), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.6117, mae: 0.4621, huber: 0.2119, swd: 0.2473, ept: 306.7749
    Epoch [15/50], Val Losses: mse: 0.4410, mae: 0.4778, huber: 0.2016, swd: 0.1386, ept: 271.1111
    Epoch [15/50], Test Losses: mse: 0.3346, mae: 0.4081, huber: 0.1560, swd: 0.1316, ept: 309.8843
      Epoch 15 composite train-obj: 0.211919
            No improvement (0.2016), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.6069, mae: 0.4602, huber: 0.2104, swd: 0.2456, ept: 306.2858
    Epoch [16/50], Val Losses: mse: 0.4096, mae: 0.4634, huber: 0.1901, swd: 0.1259, ept: 271.6083
    Epoch [16/50], Test Losses: mse: 0.3143, mae: 0.3951, huber: 0.1476, swd: 0.1162, ept: 315.3791
      Epoch 16 composite train-obj: 0.210354
            No improvement (0.1901), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.6008, mae: 0.4569, huber: 0.2079, swd: 0.2427, ept: 307.0523
    Epoch [17/50], Val Losses: mse: 0.4338, mae: 0.4738, huber: 0.1984, swd: 0.1427, ept: 270.5059
    Epoch [17/50], Test Losses: mse: 0.3152, mae: 0.3939, huber: 0.1472, swd: 0.1228, ept: 313.2207
      Epoch 17 composite train-obj: 0.207896
    Epoch [17/50], Test Losses: mse: 0.2946, mae: 0.3800, huber: 0.1388, swd: 0.1088, ept: 315.7448
    Best round's Test MSE: 0.2946, MAE: 0.3800, SWD: 0.1088
    Best round's Validation MSE: 0.3815, MAE: 0.4463, SWD: 0.1190
    Best round's Test verification MSE : 0.2946, MAE: 0.3800, SWD: 0.1088
    Time taken: 22.98 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7107, mae: 0.5074, huber: 0.2459, swd: 0.2566, ept: 287.3399
    Epoch [1/50], Val Losses: mse: 0.3981, mae: 0.4577, huber: 0.1861, swd: 0.0875, ept: 268.2673
    Epoch [1/50], Test Losses: mse: 0.2892, mae: 0.3765, huber: 0.1361, swd: 0.0946, ept: 307.3052
      Epoch 1 composite train-obj: 0.245879
            Val objective improved inf → 0.1861, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6907, mae: 0.4980, huber: 0.2391, swd: 0.2560, ept: 298.7860
    Epoch [2/50], Val Losses: mse: 0.3849, mae: 0.4499, huber: 0.1807, swd: 0.0954, ept: 265.9352
    Epoch [2/50], Test Losses: mse: 0.2842, mae: 0.3714, huber: 0.1339, swd: 0.0966, ept: 308.6300
      Epoch 2 composite train-obj: 0.239058
            Val objective improved 0.1861 → 0.1807, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6788, mae: 0.4938, huber: 0.2356, swd: 0.2538, ept: 300.2276
    Epoch [3/50], Val Losses: mse: 0.3905, mae: 0.4530, huber: 0.1826, swd: 0.1092, ept: 270.4964
    Epoch [3/50], Test Losses: mse: 0.2753, mae: 0.3648, huber: 0.1298, swd: 0.0917, ept: 312.6451
      Epoch 3 composite train-obj: 0.235563
            No improvement (0.1826), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.6660, mae: 0.4884, huber: 0.2314, swd: 0.2512, ept: 301.5519
    Epoch [4/50], Val Losses: mse: 0.3872, mae: 0.4529, huber: 0.1816, swd: 0.1046, ept: 266.9761
    Epoch [4/50], Test Losses: mse: 0.2749, mae: 0.3673, huber: 0.1301, swd: 0.0895, ept: 311.8065
      Epoch 4 composite train-obj: 0.231394
            No improvement (0.1816), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.6554, mae: 0.4841, huber: 0.2280, swd: 0.2489, ept: 301.9031
    Epoch [5/50], Val Losses: mse: 0.3887, mae: 0.4528, huber: 0.1821, swd: 0.1152, ept: 267.4463
    Epoch [5/50], Test Losses: mse: 0.2743, mae: 0.3656, huber: 0.1298, swd: 0.0912, ept: 315.0575
      Epoch 5 composite train-obj: 0.228044
            No improvement (0.1821), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.6530, mae: 0.4825, huber: 0.2269, swd: 0.2484, ept: 303.3681
    Epoch [6/50], Val Losses: mse: 0.3761, mae: 0.4436, huber: 0.1764, swd: 0.1104, ept: 267.2161
    Epoch [6/50], Test Losses: mse: 0.2759, mae: 0.3668, huber: 0.1304, swd: 0.0908, ept: 311.6864
      Epoch 6 composite train-obj: 0.226928
            Val objective improved 0.1807 → 0.1764, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.6459, mae: 0.4790, huber: 0.2244, swd: 0.2464, ept: 304.4811
    Epoch [7/50], Val Losses: mse: 0.3915, mae: 0.4528, huber: 0.1826, swd: 0.1142, ept: 266.8066
    Epoch [7/50], Test Losses: mse: 0.2783, mae: 0.3676, huber: 0.1312, swd: 0.0947, ept: 314.6629
      Epoch 7 composite train-obj: 0.224358
            No improvement (0.1826), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.6399, mae: 0.4770, huber: 0.2228, swd: 0.2456, ept: 304.0660
    Epoch [8/50], Val Losses: mse: 0.3968, mae: 0.4567, huber: 0.1849, swd: 0.1138, ept: 269.6663
    Epoch [8/50], Test Losses: mse: 0.2906, mae: 0.3791, huber: 0.1369, swd: 0.0972, ept: 312.7199
      Epoch 8 composite train-obj: 0.222777
            No improvement (0.1849), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.6369, mae: 0.4744, huber: 0.2210, swd: 0.2441, ept: 305.0960
    Epoch [9/50], Val Losses: mse: 0.4025, mae: 0.4600, huber: 0.1873, swd: 0.1139, ept: 271.1339
    Epoch [9/50], Test Losses: mse: 0.2825, mae: 0.3728, huber: 0.1335, swd: 0.0938, ept: 315.6136
      Epoch 9 composite train-obj: 0.220999
            No improvement (0.1873), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.6326, mae: 0.4726, huber: 0.2196, swd: 0.2423, ept: 305.6614
    Epoch [10/50], Val Losses: mse: 0.4129, mae: 0.4650, huber: 0.1911, swd: 0.1137, ept: 272.1918
    Epoch [10/50], Test Losses: mse: 0.2882, mae: 0.3759, huber: 0.1358, swd: 0.0969, ept: 311.8986
      Epoch 10 composite train-obj: 0.219558
            No improvement (0.1911), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.6272, mae: 0.4699, huber: 0.2176, swd: 0.2403, ept: 304.9739
    Epoch [11/50], Val Losses: mse: 0.4088, mae: 0.4628, huber: 0.1895, swd: 0.1153, ept: 271.2360
    Epoch [11/50], Test Losses: mse: 0.2936, mae: 0.3799, huber: 0.1380, swd: 0.1001, ept: 313.9288
      Epoch 11 composite train-obj: 0.217603
    Epoch [11/50], Test Losses: mse: 0.2759, mae: 0.3668, huber: 0.1304, swd: 0.0908, ept: 311.6864
    Best round's Test MSE: 0.2759, MAE: 0.3668, SWD: 0.0908
    Best round's Validation MSE: 0.3761, MAE: 0.4436, SWD: 0.1104
    Best round's Test verification MSE : 0.2759, MAE: 0.3668, SWD: 0.0908
    Time taken: 14.76 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.7127, mae: 0.5083, huber: 0.2464, swd: 0.2862, ept: 287.0532
    Epoch [1/50], Val Losses: mse: 0.4285, mae: 0.4752, huber: 0.1984, swd: 0.0978, ept: 267.4736
    Epoch [1/50], Test Losses: mse: 0.3062, mae: 0.3874, huber: 0.1432, swd: 0.1153, ept: 311.4488
      Epoch 1 composite train-obj: 0.246386
            Val objective improved inf → 0.1984, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.6888, mae: 0.4981, huber: 0.2388, swd: 0.2846, ept: 298.1652
    Epoch [2/50], Val Losses: mse: 0.4038, mae: 0.4626, huber: 0.1887, swd: 0.1085, ept: 266.1895
    Epoch [2/50], Test Losses: mse: 0.2820, mae: 0.3729, huber: 0.1332, swd: 0.1048, ept: 310.1786
      Epoch 2 composite train-obj: 0.238832
            Val objective improved 0.1984 → 0.1887, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.6759, mae: 0.4928, huber: 0.2348, swd: 0.2811, ept: 300.5689
    Epoch [3/50], Val Losses: mse: 0.3904, mae: 0.4546, huber: 0.1831, swd: 0.1083, ept: 265.5002
    Epoch [3/50], Test Losses: mse: 0.2809, mae: 0.3706, huber: 0.1324, swd: 0.1018, ept: 311.5250
      Epoch 3 composite train-obj: 0.234777
            Val objective improved 0.1887 → 0.1831, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.6659, mae: 0.4887, huber: 0.2315, swd: 0.2774, ept: 301.0536
    Epoch [4/50], Val Losses: mse: 0.3789, mae: 0.4467, huber: 0.1779, swd: 0.1090, ept: 267.8900
    Epoch [4/50], Test Losses: mse: 0.2778, mae: 0.3678, huber: 0.1310, swd: 0.0999, ept: 312.5075
      Epoch 4 composite train-obj: 0.231525
            Val objective improved 0.1831 → 0.1779, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.6574, mae: 0.4845, huber: 0.2284, swd: 0.2753, ept: 302.9608
    Epoch [5/50], Val Losses: mse: 0.3730, mae: 0.4419, huber: 0.1753, swd: 0.1134, ept: 271.4661
    Epoch [5/50], Test Losses: mse: 0.2698, mae: 0.3613, huber: 0.1275, swd: 0.1007, ept: 315.4913
      Epoch 5 composite train-obj: 0.228427
            Val objective improved 0.1779 → 0.1753, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.6518, mae: 0.4819, huber: 0.2266, swd: 0.2734, ept: 303.4464
    Epoch [6/50], Val Losses: mse: 0.3886, mae: 0.4529, huber: 0.1819, swd: 0.1096, ept: 270.6729
    Epoch [6/50], Test Losses: mse: 0.2900, mae: 0.3792, huber: 0.1370, swd: 0.1050, ept: 312.6429
      Epoch 6 composite train-obj: 0.226623
            No improvement (0.1819), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.6460, mae: 0.4792, huber: 0.2245, swd: 0.2725, ept: 304.4632
    Epoch [7/50], Val Losses: mse: 0.3847, mae: 0.4506, huber: 0.1803, swd: 0.1193, ept: 268.7273
    Epoch [7/50], Test Losses: mse: 0.2808, mae: 0.3708, huber: 0.1324, swd: 0.1052, ept: 316.0829
      Epoch 7 composite train-obj: 0.224459
            No improvement (0.1803), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.6401, mae: 0.4762, huber: 0.2224, swd: 0.2701, ept: 304.9475
    Epoch [8/50], Val Losses: mse: 0.3914, mae: 0.4538, huber: 0.1828, swd: 0.1212, ept: 271.4492
    Epoch [8/50], Test Losses: mse: 0.2876, mae: 0.3757, huber: 0.1354, swd: 0.1074, ept: 317.4736
      Epoch 8 composite train-obj: 0.222361
            No improvement (0.1828), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.6341, mae: 0.4735, huber: 0.2202, swd: 0.2674, ept: 305.0563
    Epoch [9/50], Val Losses: mse: 0.3934, mae: 0.4569, huber: 0.1841, swd: 0.1134, ept: 270.0200
    Epoch [9/50], Test Losses: mse: 0.2922, mae: 0.3822, huber: 0.1382, swd: 0.1077, ept: 315.5450
      Epoch 9 composite train-obj: 0.220234
            No improvement (0.1841), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.6292, mae: 0.4711, huber: 0.2185, swd: 0.2659, ept: 305.4809
    Epoch [10/50], Val Losses: mse: 0.3998, mae: 0.4601, huber: 0.1865, swd: 0.1189, ept: 270.8083
    Epoch [10/50], Test Losses: mse: 0.2915, mae: 0.3790, huber: 0.1373, swd: 0.1091, ept: 316.5810
      Epoch 10 composite train-obj: 0.218481
    Epoch [10/50], Test Losses: mse: 0.2698, mae: 0.3613, huber: 0.1275, swd: 0.1007, ept: 315.4913
    Best round's Test MSE: 0.2698, MAE: 0.3613, SWD: 0.1007
    Best round's Validation MSE: 0.3730, MAE: 0.4419, SWD: 0.1134
    Best round's Test verification MSE : 0.2698, MAE: 0.3613, SWD: 0.1007
    Time taken: 13.58 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth2_seq96_pred720_20250510_2051)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2801 ± 0.0106
      mae: 0.3694 ± 0.0078
      huber: 0.1322 ± 0.0048
      swd: 0.1001 ± 0.0074
      ept: 314.3075 ± 1.8563
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3769 ± 0.0035
      mae: 0.4439 ± 0.0018
      huber: 0.1767 ± 0.0013
      swd: 0.1143 ± 0.0036
      ept: 269.8650 ± 1.8866
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 51.37 seconds
    
    Experiment complete: PatchTST_etth2_seq96_pred720_20250510_2051
    Model: PatchTST
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['etth2']['channels'],
    enc_in=data_mgr.datasets['etth2']['channels'],
    dec_in=data_mgr.datasets['etth2']['channels'],
    c_out=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50,
    task_name='long_term_forecast',
    factor=3,
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
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
    
    Epoch [1/50], Train Losses: mse: 62.5725, mae: 4.2844, huber: 3.8480, swd: 33.4326, ept: 286.6956
    Epoch [1/50], Val Losses: mse: 27.6712, mae: 3.6223, huber: 3.1834, swd: 10.1683, ept: 271.2826
    Epoch [1/50], Test Losses: mse: 22.7348, mae: 3.0853, huber: 2.6547, swd: 10.7165, ept: 308.6064
      Epoch 1 composite train-obj: 3.848009
            Val objective improved inf → 3.1834, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.7507, mae: 4.1893, huber: 3.7536, swd: 32.3447, ept: 297.9279
    Epoch [2/50], Val Losses: mse: 26.8459, mae: 3.5764, huber: 3.1341, swd: 8.9945, ept: 270.8275
    Epoch [2/50], Test Losses: mse: 22.4876, mae: 3.0592, huber: 2.6276, swd: 10.6030, ept: 313.8597
      Epoch 2 composite train-obj: 3.753551
            Val objective improved 3.1834 → 3.1341, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.5248, mae: 4.1430, huber: 3.7071, swd: 31.2438, ept: 298.5798
    Epoch [3/50], Val Losses: mse: 29.3923, mae: 3.7305, huber: 3.2858, swd: 11.3532, ept: 273.1529
    Epoch [3/50], Test Losses: mse: 23.1592, mae: 3.1283, huber: 2.6955, swd: 10.7653, ept: 311.4187
      Epoch 3 composite train-obj: 3.707057
            No improvement (3.2858), counter 1/5
    Epoch [4/50], Train Losses: mse: 58.1727, mae: 4.0868, huber: 3.6513, swd: 30.0646, ept: 300.4151
    Epoch [4/50], Val Losses: mse: 27.4635, mae: 3.6209, huber: 3.1781, swd: 9.6518, ept: 271.4752
    Epoch [4/50], Test Losses: mse: 24.3515, mae: 3.2014, huber: 2.7684, swd: 11.9773, ept: 311.5092
      Epoch 4 composite train-obj: 3.651331
            No improvement (3.1781), counter 2/5
    Epoch [5/50], Train Losses: mse: 57.3037, mae: 4.0434, huber: 3.6084, swd: 29.3860, ept: 301.0898
    Epoch [5/50], Val Losses: mse: 27.1271, mae: 3.6019, huber: 3.1572, swd: 9.0790, ept: 264.4050
    Epoch [5/50], Test Losses: mse: 23.8595, mae: 3.1650, huber: 2.7318, swd: 11.5196, ept: 312.0369
      Epoch 5 composite train-obj: 3.608400
            No improvement (3.1572), counter 3/5
    Epoch [6/50], Train Losses: mse: 56.6783, mae: 4.0124, huber: 3.5776, swd: 28.8325, ept: 301.4054
    Epoch [6/50], Val Losses: mse: 26.8732, mae: 3.5952, huber: 3.1500, swd: 9.0182, ept: 267.8853
    Epoch [6/50], Test Losses: mse: 24.0035, mae: 3.1773, huber: 2.7444, swd: 11.6248, ept: 312.0693
      Epoch 6 composite train-obj: 3.577601
            No improvement (3.1500), counter 4/5
    Epoch [7/50], Train Losses: mse: 56.0651, mae: 3.9809, huber: 3.5464, swd: 28.3314, ept: 301.9967
    Epoch [7/50], Val Losses: mse: 27.4751, mae: 3.6156, huber: 3.1704, swd: 9.5643, ept: 268.6724
    Epoch [7/50], Test Losses: mse: 24.0995, mae: 3.1869, huber: 2.7530, swd: 11.8162, ept: 313.0267
      Epoch 7 composite train-obj: 3.546416
    Epoch [7/50], Test Losses: mse: 22.4876, mae: 3.0592, huber: 2.6276, swd: 10.6030, ept: 313.8597
    Best round's Test MSE: 22.4876, MAE: 3.0592, SWD: 10.6030
    Best round's Validation MSE: 26.8459, MAE: 3.5764, SWD: 8.9945
    Best round's Test verification MSE : 22.4876, MAE: 3.0592, SWD: 10.6030
    Time taken: 11.48 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 62.4187, mae: 4.2800, huber: 3.8435, swd: 30.5448, ept: 287.2565
    Epoch [1/50], Val Losses: mse: 29.1702, mae: 3.7036, huber: 3.2637, swd: 10.1496, ept: 270.2992
    Epoch [1/50], Test Losses: mse: 24.0993, mae: 3.1852, huber: 2.7533, swd: 10.6065, ept: 305.8520
      Epoch 1 composite train-obj: 3.843533
            Val objective improved inf → 3.2637, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.4337, mae: 4.1814, huber: 3.7455, swd: 29.4274, ept: 297.9186
    Epoch [2/50], Val Losses: mse: 27.3435, mae: 3.6025, huber: 3.1613, swd: 8.7357, ept: 272.4422
    Epoch [2/50], Test Losses: mse: 23.8800, mae: 3.1519, huber: 2.7204, swd: 10.6081, ept: 308.3837
      Epoch 2 composite train-obj: 3.745517
            Val objective improved 3.2637 → 3.1613, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.1608, mae: 4.1369, huber: 3.7009, swd: 28.3920, ept: 298.9817
    Epoch [3/50], Val Losses: mse: 26.6779, mae: 3.5735, huber: 3.1304, swd: 8.0521, ept: 269.9088
    Epoch [3/50], Test Losses: mse: 23.2872, mae: 3.1141, huber: 2.6824, swd: 10.1641, ept: 311.8710
      Epoch 3 composite train-obj: 3.700941
            Val objective improved 3.1613 → 3.1304, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 58.0444, mae: 4.0890, huber: 3.6536, swd: 27.5019, ept: 300.0047
    Epoch [4/50], Val Losses: mse: 26.8699, mae: 3.5894, huber: 3.1469, swd: 8.4533, ept: 265.5795
    Epoch [4/50], Test Losses: mse: 22.7280, mae: 3.0883, huber: 2.6565, swd: 9.7308, ept: 311.7912
      Epoch 4 composite train-obj: 3.653572
            No improvement (3.1469), counter 1/5
    Epoch [5/50], Train Losses: mse: 57.1877, mae: 4.0483, huber: 3.6131, swd: 26.8438, ept: 300.6225
    Epoch [5/50], Val Losses: mse: 27.4971, mae: 3.6347, huber: 3.1895, swd: 8.8175, ept: 261.7253
    Epoch [5/50], Test Losses: mse: 25.0301, mae: 3.2160, huber: 2.7835, swd: 11.4360, ept: 311.9935
      Epoch 5 composite train-obj: 3.613119
            No improvement (3.1895), counter 2/5
    Epoch [6/50], Train Losses: mse: 56.7685, mae: 4.0284, huber: 3.5933, swd: 26.5261, ept: 301.4359
    Epoch [6/50], Val Losses: mse: 26.7722, mae: 3.5727, huber: 3.1262, swd: 7.7087, ept: 268.2673
    Epoch [6/50], Test Losses: mse: 23.5816, mae: 3.1357, huber: 2.7026, swd: 10.0719, ept: 310.4266
      Epoch 6 composite train-obj: 3.593276
            Val objective improved 3.1304 → 3.1262, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 56.1132, mae: 3.9911, huber: 3.5564, swd: 26.0136, ept: 302.5607
    Epoch [7/50], Val Losses: mse: 27.6249, mae: 3.6218, huber: 3.1758, swd: 8.7907, ept: 267.7108
    Epoch [7/50], Test Losses: mse: 23.8769, mae: 3.1612, huber: 2.7282, swd: 10.5515, ept: 315.0006
      Epoch 7 composite train-obj: 3.556418
            No improvement (3.1758), counter 1/5
    Epoch [8/50], Train Losses: mse: 55.6847, mae: 3.9696, huber: 3.5351, swd: 25.6937, ept: 302.3641
    Epoch [8/50], Val Losses: mse: 28.3959, mae: 3.6799, huber: 3.2326, swd: 9.6106, ept: 265.2445
    Epoch [8/50], Test Losses: mse: 23.7301, mae: 3.1624, huber: 2.7288, swd: 10.3304, ept: 314.6860
      Epoch 8 composite train-obj: 3.535132
            No improvement (3.2326), counter 2/5
    Epoch [9/50], Train Losses: mse: 55.0776, mae: 3.9356, huber: 3.5012, swd: 25.2012, ept: 303.4723
    Epoch [9/50], Val Losses: mse: 28.1913, mae: 3.6633, huber: 3.2184, swd: 9.4226, ept: 267.2243
    Epoch [9/50], Test Losses: mse: 24.5035, mae: 3.2064, huber: 2.7734, swd: 10.7976, ept: 314.1875
      Epoch 9 composite train-obj: 3.501235
            No improvement (3.2184), counter 3/5
    Epoch [10/50], Train Losses: mse: 54.9226, mae: 3.9193, huber: 3.4852, swd: 25.1149, ept: 303.7469
    Epoch [10/50], Val Losses: mse: 27.9348, mae: 3.6565, huber: 3.2101, swd: 9.2007, ept: 267.5267
    Epoch [10/50], Test Losses: mse: 24.5716, mae: 3.2320, huber: 2.7975, swd: 10.9148, ept: 313.1839
      Epoch 10 composite train-obj: 3.485179
            No improvement (3.2101), counter 4/5
    Epoch [11/50], Train Losses: mse: 54.4128, mae: 3.8947, huber: 3.4609, swd: 24.7115, ept: 303.7319
    Epoch [11/50], Val Losses: mse: 28.0343, mae: 3.6657, huber: 3.2195, swd: 9.5286, ept: 269.3608
    Epoch [11/50], Test Losses: mse: 24.3961, mae: 3.2145, huber: 2.7812, swd: 11.0380, ept: 313.1080
      Epoch 11 composite train-obj: 3.460926
    Epoch [11/50], Test Losses: mse: 23.5816, mae: 3.1357, huber: 2.7026, swd: 10.0719, ept: 310.4266
    Best round's Test MSE: 23.5816, MAE: 3.1357, SWD: 10.0719
    Best round's Validation MSE: 26.7722, MAE: 3.5727, SWD: 7.7087
    Best round's Test verification MSE : 23.5816, MAE: 3.1357, SWD: 10.0719
    Time taken: 19.02 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 62.5565, mae: 4.2871, huber: 3.8504, swd: 35.2044, ept: 286.4705
    Epoch [1/50], Val Losses: mse: 30.0397, mae: 3.7550, huber: 3.3143, swd: 12.1439, ept: 268.8549
    Epoch [1/50], Test Losses: mse: 24.7675, mae: 3.2292, huber: 2.7968, swd: 12.7942, ept: 310.0403
      Epoch 1 composite train-obj: 3.850446
            Val objective improved inf → 3.3143, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.2870, mae: 4.1798, huber: 3.7434, swd: 33.6227, ept: 297.3480
    Epoch [2/50], Val Losses: mse: 26.9536, mae: 3.5844, huber: 3.1401, swd: 8.8426, ept: 267.7271
    Epoch [2/50], Test Losses: mse: 23.4784, mae: 3.1187, huber: 2.6871, swd: 11.7986, ept: 309.9255
      Epoch 2 composite train-obj: 3.743410
            Val objective improved 3.3143 → 3.1401, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 58.8785, mae: 4.1235, huber: 3.6874, swd: 32.3465, ept: 298.7347
    Epoch [3/50], Val Losses: mse: 26.1577, mae: 3.5466, huber: 3.1024, swd: 7.8277, ept: 269.9435
    Epoch [3/50], Test Losses: mse: 23.7227, mae: 3.1423, huber: 2.7100, swd: 11.6659, ept: 312.9216
      Epoch 3 composite train-obj: 3.687395
            Val objective improved 3.1401 → 3.1024, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 57.8823, mae: 4.0810, huber: 3.6453, swd: 31.5131, ept: 300.2721
    Epoch [4/50], Val Losses: mse: 26.6345, mae: 3.5564, huber: 3.1125, swd: 8.1783, ept: 268.8559
    Epoch [4/50], Test Losses: mse: 24.1864, mae: 3.1741, huber: 2.7420, swd: 11.9799, ept: 309.6016
      Epoch 4 composite train-obj: 3.645268
            No improvement (3.1125), counter 1/5
    Epoch [5/50], Train Losses: mse: 57.3107, mae: 4.0479, huber: 3.6126, swd: 31.0420, ept: 301.2797
    Epoch [5/50], Val Losses: mse: 25.7685, mae: 3.5103, huber: 3.0661, swd: 7.5850, ept: 270.2966
    Epoch [5/50], Test Losses: mse: 23.1096, mae: 3.1062, huber: 2.6744, swd: 11.1966, ept: 312.3621
      Epoch 5 composite train-obj: 3.612554
            Val objective improved 3.1024 → 3.0661, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 56.8261, mae: 4.0250, huber: 3.5899, swd: 30.6480, ept: 301.7452
    Epoch [6/50], Val Losses: mse: 26.7886, mae: 3.5807, huber: 3.1368, swd: 8.6653, ept: 267.8469
    Epoch [6/50], Test Losses: mse: 23.7634, mae: 3.1481, huber: 2.7156, swd: 11.7168, ept: 311.3250
      Epoch 6 composite train-obj: 3.589879
            No improvement (3.1368), counter 1/5
    Epoch [7/50], Train Losses: mse: 56.3295, mae: 3.9924, huber: 3.5575, swd: 30.2004, ept: 302.3561
    Epoch [7/50], Val Losses: mse: 26.1048, mae: 3.5175, huber: 3.0739, swd: 7.5843, ept: 269.7631
    Epoch [7/50], Test Losses: mse: 23.2667, mae: 3.1114, huber: 2.6787, swd: 11.0901, ept: 315.4636
      Epoch 7 composite train-obj: 3.557521
            No improvement (3.0739), counter 2/5
    Epoch [8/50], Train Losses: mse: 55.9508, mae: 3.9671, huber: 3.5326, swd: 29.8922, ept: 302.7949
    Epoch [8/50], Val Losses: mse: 27.2911, mae: 3.6169, huber: 3.1721, swd: 8.8351, ept: 267.2617
    Epoch [8/50], Test Losses: mse: 24.6915, mae: 3.2221, huber: 2.7877, swd: 12.3009, ept: 313.3405
      Epoch 8 composite train-obj: 3.532631
            No improvement (3.1721), counter 3/5
    Epoch [9/50], Train Losses: mse: 55.5459, mae: 3.9400, huber: 3.5058, swd: 29.5396, ept: 303.2673
    Epoch [9/50], Val Losses: mse: 26.8757, mae: 3.5878, huber: 3.1427, swd: 8.6661, ept: 266.9571
    Epoch [9/50], Test Losses: mse: 24.3897, mae: 3.2073, huber: 2.7738, swd: 12.2263, ept: 314.1447
      Epoch 9 composite train-obj: 3.505808
            No improvement (3.1427), counter 4/5
    Epoch [10/50], Train Losses: mse: 55.1437, mae: 3.9202, huber: 3.4862, swd: 29.1691, ept: 303.6871
    Epoch [10/50], Val Losses: mse: 28.2174, mae: 3.6554, huber: 3.2111, swd: 9.9563, ept: 268.4662
    Epoch [10/50], Test Losses: mse: 24.8435, mae: 3.2341, huber: 2.8001, swd: 12.2977, ept: 313.9712
      Epoch 10 composite train-obj: 3.486157
    Epoch [10/50], Test Losses: mse: 23.1096, mae: 3.1062, huber: 2.6744, swd: 11.1966, ept: 312.3621
    Best round's Test MSE: 23.1096, MAE: 3.1062, SWD: 11.1966
    Best round's Validation MSE: 25.7685, MAE: 3.5103, SWD: 7.5850
    Best round's Test verification MSE : 23.1096, MAE: 3.1062, SWD: 11.1966
    Time taken: 17.61 seconds
    
    ==================================================
    Experiment Summary (PatchTST_etth2_seq96_pred720_20250511_1625)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 23.0596 ± 0.4480
      mae: 3.1004 ± 0.0315
      huber: 2.6682 ± 0.0309
      swd: 10.6239 ± 0.4594
      ept: 312.2161 ± 1.4054
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 26.4622 ± 0.4915
      mae: 3.5531 ± 0.0303
      huber: 3.1088 ± 0.0304
      swd: 8.0961 ± 0.6373
      ept: 269.7971 ± 1.1033
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 48.20 seconds
    
    Experiment complete: PatchTST_etth2_seq96_pred720_20250511_1625
    Model: PatchTST
    Dataset: etth2
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
    channels=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.4117, mae: 0.3887, huber: 0.1555, swd: 0.1405, target_std: 0.7878
    Epoch [1/50], Val Losses: mse: 0.2844, mae: 0.3709, huber: 0.1335, swd: 0.1271, target_std: 0.9798
    Epoch [1/50], Test Losses: mse: 0.1838, mae: 0.2966, huber: 0.0879, swd: 0.0728, target_std: 0.7349
      Epoch 1 composite train-obj: 0.155500
            Val objective improved inf → 0.1335, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2951, mae: 0.3191, huber: 0.1134, swd: 0.1201, target_std: 0.7880
    Epoch [2/50], Val Losses: mse: 0.2715, mae: 0.3583, huber: 0.1276, swd: 0.1238, target_std: 0.9798
    Epoch [2/50], Test Losses: mse: 0.1755, mae: 0.2874, huber: 0.0840, swd: 0.0723, target_std: 0.7349
      Epoch 2 composite train-obj: 0.113439
            Val objective improved 0.1335 → 0.1276, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2841, mae: 0.3106, huber: 0.1091, swd: 0.1162, target_std: 0.7880
    Epoch [3/50], Val Losses: mse: 0.2644, mae: 0.3508, huber: 0.1248, swd: 0.1147, target_std: 0.9798
    Epoch [3/50], Test Losses: mse: 0.1698, mae: 0.2815, huber: 0.0813, swd: 0.0659, target_std: 0.7349
      Epoch 3 composite train-obj: 0.109083
            Val objective improved 0.1276 → 0.1248, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2798, mae: 0.3068, huber: 0.1073, swd: 0.1142, target_std: 0.7879
    Epoch [4/50], Val Losses: mse: 0.2640, mae: 0.3518, huber: 0.1246, swd: 0.1175, target_std: 0.9798
    Epoch [4/50], Test Losses: mse: 0.1700, mae: 0.2814, huber: 0.0813, swd: 0.0686, target_std: 0.7349
      Epoch 4 composite train-obj: 0.107257
            Val objective improved 0.1248 → 0.1246, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.2776, mae: 0.3052, huber: 0.1064, swd: 0.1129, target_std: 0.7879
    Epoch [5/50], Val Losses: mse: 0.2659, mae: 0.3527, huber: 0.1251, swd: 0.1208, target_std: 0.9798
    Epoch [5/50], Test Losses: mse: 0.1700, mae: 0.2818, huber: 0.0814, swd: 0.0703, target_std: 0.7349
      Epoch 5 composite train-obj: 0.106390
            No improvement (0.1251), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.2762, mae: 0.3048, huber: 0.1061, swd: 0.1126, target_std: 0.7878
    Epoch [6/50], Val Losses: mse: 0.2618, mae: 0.3473, huber: 0.1237, swd: 0.1133, target_std: 0.9798
    Epoch [6/50], Test Losses: mse: 0.1691, mae: 0.2814, huber: 0.0810, swd: 0.0675, target_std: 0.7349
      Epoch 6 composite train-obj: 0.106089
            Val objective improved 0.1246 → 0.1237, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.2755, mae: 0.3040, huber: 0.1057, swd: 0.1124, target_std: 0.7879
    Epoch [7/50], Val Losses: mse: 0.2639, mae: 0.3508, huber: 0.1244, swd: 0.1174, target_std: 0.9798
    Epoch [7/50], Test Losses: mse: 0.1686, mae: 0.2798, huber: 0.0807, swd: 0.0686, target_std: 0.7349
      Epoch 7 composite train-obj: 0.105687
            No improvement (0.1244), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.2752, mae: 0.3039, huber: 0.1056, swd: 0.1123, target_std: 0.7880
    Epoch [8/50], Val Losses: mse: 0.2641, mae: 0.3496, huber: 0.1244, swd: 0.1175, target_std: 0.9798
    Epoch [8/50], Test Losses: mse: 0.1701, mae: 0.2819, huber: 0.0814, swd: 0.0704, target_std: 0.7349
      Epoch 8 composite train-obj: 0.105557
            No improvement (0.1244), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.2749, mae: 0.3033, huber: 0.1054, swd: 0.1116, target_std: 0.7879
    Epoch [9/50], Val Losses: mse: 0.2620, mae: 0.3477, huber: 0.1235, swd: 0.1151, target_std: 0.9798
    Epoch [9/50], Test Losses: mse: 0.1713, mae: 0.2827, huber: 0.0820, swd: 0.0694, target_std: 0.7349
      Epoch 9 composite train-obj: 0.105394
            Val objective improved 0.1237 → 0.1235, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.2745, mae: 0.3034, huber: 0.1054, swd: 0.1118, target_std: 0.7881
    Epoch [10/50], Val Losses: mse: 0.2668, mae: 0.3527, huber: 0.1252, swd: 0.1243, target_std: 0.9798
    Epoch [10/50], Test Losses: mse: 0.1748, mae: 0.2885, huber: 0.0837, swd: 0.0763, target_std: 0.7349
      Epoch 10 composite train-obj: 0.105360
            No improvement (0.1252), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.2747, mae: 0.3031, huber: 0.1053, swd: 0.1115, target_std: 0.7879
    Epoch [11/50], Val Losses: mse: 0.2653, mae: 0.3539, huber: 0.1248, swd: 0.1217, target_std: 0.9798
    Epoch [11/50], Test Losses: mse: 0.1707, mae: 0.2826, huber: 0.0817, swd: 0.0715, target_std: 0.7349
      Epoch 11 composite train-obj: 0.105258
            No improvement (0.1248), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.2743, mae: 0.3032, huber: 0.1052, swd: 0.1116, target_std: 0.7877
    Epoch [12/50], Val Losses: mse: 0.2643, mae: 0.3513, huber: 0.1243, swd: 0.1199, target_std: 0.9798
    Epoch [12/50], Test Losses: mse: 0.1715, mae: 0.2824, huber: 0.0820, swd: 0.0722, target_std: 0.7349
      Epoch 12 composite train-obj: 0.105230
            No improvement (0.1243), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.2744, mae: 0.3031, huber: 0.1052, swd: 0.1115, target_std: 0.7879
    Epoch [13/50], Val Losses: mse: 0.2699, mae: 0.3568, huber: 0.1265, swd: 0.1294, target_std: 0.9798
    Epoch [13/50], Test Losses: mse: 0.1754, mae: 0.2866, huber: 0.0838, swd: 0.0780, target_std: 0.7349
      Epoch 13 composite train-obj: 0.105219
            No improvement (0.1265), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.2743, mae: 0.3033, huber: 0.1052, swd: 0.1119, target_std: 0.7879
    Epoch [14/50], Val Losses: mse: 0.2607, mae: 0.3469, huber: 0.1229, swd: 0.1135, target_std: 0.9798
    Epoch [14/50], Test Losses: mse: 0.1711, mae: 0.2822, huber: 0.0818, swd: 0.0685, target_std: 0.7349
      Epoch 14 composite train-obj: 0.105242
            Val objective improved 0.1235 → 0.1229, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.2746, mae: 0.3037, huber: 0.1054, swd: 0.1119, target_std: 0.7879
    Epoch [15/50], Val Losses: mse: 0.2620, mae: 0.3488, huber: 0.1239, swd: 0.1141, target_std: 0.9798
    Epoch [15/50], Test Losses: mse: 0.1694, mae: 0.2813, huber: 0.0810, swd: 0.0686, target_std: 0.7349
      Epoch 15 composite train-obj: 0.105363
            No improvement (0.1239), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.2742, mae: 0.3035, huber: 0.1053, swd: 0.1115, target_std: 0.7878
    Epoch [16/50], Val Losses: mse: 0.2621, mae: 0.3477, huber: 0.1233, swd: 0.1164, target_std: 0.9798
    Epoch [16/50], Test Losses: mse: 0.1694, mae: 0.2804, huber: 0.0811, swd: 0.0689, target_std: 0.7349
      Epoch 16 composite train-obj: 0.105274
            No improvement (0.1233), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.2746, mae: 0.3030, huber: 0.1052, swd: 0.1115, target_std: 0.7880
    Epoch [17/50], Val Losses: mse: 0.2638, mae: 0.3512, huber: 0.1243, swd: 0.1178, target_std: 0.9798
    Epoch [17/50], Test Losses: mse: 0.1679, mae: 0.2805, huber: 0.0804, swd: 0.0684, target_std: 0.7349
      Epoch 17 composite train-obj: 0.105243
            No improvement (0.1243), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.2740, mae: 0.3036, huber: 0.1053, swd: 0.1122, target_std: 0.7880
    Epoch [18/50], Val Losses: mse: 0.2644, mae: 0.3513, huber: 0.1248, swd: 0.1158, target_std: 0.9798
    Epoch [18/50], Test Losses: mse: 0.1664, mae: 0.2780, huber: 0.0797, swd: 0.0668, target_std: 0.7349
      Epoch 18 composite train-obj: 0.105272
            No improvement (0.1248), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.2743, mae: 0.3032, huber: 0.1052, swd: 0.1114, target_std: 0.7879
    Epoch [19/50], Val Losses: mse: 0.2632, mae: 0.3490, huber: 0.1241, swd: 0.1142, target_std: 0.9798
    Epoch [19/50], Test Losses: mse: 0.1655, mae: 0.2776, huber: 0.0793, swd: 0.0643, target_std: 0.7349
      Epoch 19 composite train-obj: 0.105178
    Epoch [19/50], Test Losses: mse: 0.1711, mae: 0.2822, huber: 0.0818, swd: 0.0685, target_std: 0.7349
    Best round's Test MSE: 0.1711, MAE: 0.2822, SWD: 0.0685
    Best round's Validation MSE: 0.2607, MAE: 0.3469
    Best round's Test verification MSE : 0.1711, MAE: 0.2822, SWD: 0.0685
    Time taken: 20.72 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4063, mae: 0.3860, huber: 0.1534, swd: 0.1399, target_std: 0.7879
    Epoch [1/50], Val Losses: mse: 0.2799, mae: 0.3666, huber: 0.1316, swd: 0.1187, target_std: 0.9798
    Epoch [1/50], Test Losses: mse: 0.1857, mae: 0.2982, huber: 0.0888, swd: 0.0708, target_std: 0.7349
      Epoch 1 composite train-obj: 0.153448
            Val objective improved inf → 0.1316, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2949, mae: 0.3190, huber: 0.1133, swd: 0.1192, target_std: 0.7878
    Epoch [2/50], Val Losses: mse: 0.2698, mae: 0.3564, huber: 0.1270, swd: 0.1173, target_std: 0.9798
    Epoch [2/50], Test Losses: mse: 0.1749, mae: 0.2872, huber: 0.0837, swd: 0.0688, target_std: 0.7349
      Epoch 2 composite train-obj: 0.113342
            Val objective improved 0.1316 → 0.1270, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2835, mae: 0.3103, huber: 0.1089, swd: 0.1148, target_std: 0.7877
    Epoch [3/50], Val Losses: mse: 0.2658, mae: 0.3521, huber: 0.1252, swd: 0.1156, target_std: 0.9798
    Epoch [3/50], Test Losses: mse: 0.1705, mae: 0.2828, huber: 0.0816, swd: 0.0661, target_std: 0.7349
      Epoch 3 composite train-obj: 0.108900
            Val objective improved 0.1270 → 0.1252, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2791, mae: 0.3070, huber: 0.1072, swd: 0.1131, target_std: 0.7878
    Epoch [4/50], Val Losses: mse: 0.2666, mae: 0.3513, huber: 0.1251, swd: 0.1194, target_std: 0.9798
    Epoch [4/50], Test Losses: mse: 0.1755, mae: 0.2868, huber: 0.0840, swd: 0.0726, target_std: 0.7349
      Epoch 4 composite train-obj: 0.107205
            Val objective improved 0.1252 → 0.1251, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.2775, mae: 0.3055, huber: 0.1065, swd: 0.1125, target_std: 0.7880
    Epoch [5/50], Val Losses: mse: 0.2616, mae: 0.3467, huber: 0.1234, swd: 0.1099, target_std: 0.9798
    Epoch [5/50], Test Losses: mse: 0.1690, mae: 0.2804, huber: 0.0809, swd: 0.0637, target_std: 0.7349
      Epoch 5 composite train-obj: 0.106481
            Val objective improved 0.1251 → 0.1234, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.2759, mae: 0.3043, huber: 0.1059, swd: 0.1113, target_std: 0.7877
    Epoch [6/50], Val Losses: mse: 0.2685, mae: 0.3551, huber: 0.1263, swd: 0.1205, target_std: 0.9798
    Epoch [6/50], Test Losses: mse: 0.1677, mae: 0.2786, huber: 0.0802, swd: 0.0666, target_std: 0.7349
      Epoch 6 composite train-obj: 0.105881
            No improvement (0.1263), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.2755, mae: 0.3040, huber: 0.1057, swd: 0.1110, target_std: 0.7878
    Epoch [7/50], Val Losses: mse: 0.2630, mae: 0.3488, huber: 0.1239, swd: 0.1146, target_std: 0.9798
    Epoch [7/50], Test Losses: mse: 0.1702, mae: 0.2817, huber: 0.0814, swd: 0.0670, target_std: 0.7349
      Epoch 7 composite train-obj: 0.105703
            No improvement (0.1239), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.2749, mae: 0.3036, huber: 0.1055, swd: 0.1111, target_std: 0.7880
    Epoch [8/50], Val Losses: mse: 0.2607, mae: 0.3477, huber: 0.1228, swd: 0.1136, target_std: 0.9798
    Epoch [8/50], Test Losses: mse: 0.1712, mae: 0.2826, huber: 0.0819, swd: 0.0686, target_std: 0.7349
      Epoch 8 composite train-obj: 0.105489
            Val objective improved 0.1234 → 0.1228, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.2748, mae: 0.3035, huber: 0.1054, swd: 0.1112, target_std: 0.7879
    Epoch [9/50], Val Losses: mse: 0.2632, mae: 0.3490, huber: 0.1238, swd: 0.1171, target_std: 0.9798
    Epoch [9/50], Test Losses: mse: 0.1724, mae: 0.2833, huber: 0.0825, swd: 0.0709, target_std: 0.7349
      Epoch 9 composite train-obj: 0.105438
            No improvement (0.1238), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.2751, mae: 0.3033, huber: 0.1054, swd: 0.1105, target_std: 0.7881
    Epoch [10/50], Val Losses: mse: 0.2663, mae: 0.3527, huber: 0.1253, swd: 0.1185, target_std: 0.9798
    Epoch [10/50], Test Losses: mse: 0.1653, mae: 0.2774, huber: 0.0792, swd: 0.0651, target_std: 0.7349
      Epoch 10 composite train-obj: 0.105418
            No improvement (0.1253), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.2744, mae: 0.3033, huber: 0.1053, swd: 0.1109, target_std: 0.7879
    Epoch [11/50], Val Losses: mse: 0.2624, mae: 0.3509, huber: 0.1239, swd: 0.1140, target_std: 0.9798
    Epoch [11/50], Test Losses: mse: 0.1673, mae: 0.2789, huber: 0.0801, swd: 0.0649, target_std: 0.7349
      Epoch 11 composite train-obj: 0.105271
            No improvement (0.1239), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.2745, mae: 0.3030, huber: 0.1052, swd: 0.1105, target_std: 0.7878
    Epoch [12/50], Val Losses: mse: 0.2603, mae: 0.3496, huber: 0.1231, swd: 0.1112, target_std: 0.9798
    Epoch [12/50], Test Losses: mse: 0.1687, mae: 0.2792, huber: 0.0806, swd: 0.0653, target_std: 0.7349
      Epoch 12 composite train-obj: 0.105217
            No improvement (0.1231), counter 4/5
    Epoch [13/50], Train Losses: mse: 0.2741, mae: 0.3034, huber: 0.1052, swd: 0.1108, target_std: 0.7880
    Epoch [13/50], Val Losses: mse: 0.2641, mae: 0.3502, huber: 0.1241, swd: 0.1174, target_std: 0.9798
    Epoch [13/50], Test Losses: mse: 0.1710, mae: 0.2821, huber: 0.0818, swd: 0.0694, target_std: 0.7349
      Epoch 13 composite train-obj: 0.105228
    Epoch [13/50], Test Losses: mse: 0.1712, mae: 0.2826, huber: 0.0819, swd: 0.0686, target_std: 0.7349
    Best round's Test MSE: 0.1712, MAE: 0.2826, SWD: 0.0686
    Best round's Validation MSE: 0.2607, MAE: 0.3477
    Best round's Test verification MSE : 0.1712, MAE: 0.2826, SWD: 0.0686
    Time taken: 14.09 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4054, mae: 0.3875, huber: 0.1540, swd: 0.1277, target_std: 0.7879
    Epoch [1/50], Val Losses: mse: 0.2825, mae: 0.3680, huber: 0.1326, swd: 0.1112, target_std: 0.9798
    Epoch [1/50], Test Losses: mse: 0.1854, mae: 0.2978, huber: 0.0886, swd: 0.0657, target_std: 0.7349
      Epoch 1 composite train-obj: 0.153996
            Val objective improved inf → 0.1326, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2956, mae: 0.3192, huber: 0.1135, swd: 0.1104, target_std: 0.7878
    Epoch [2/50], Val Losses: mse: 0.2750, mae: 0.3605, huber: 0.1289, swd: 0.1164, target_std: 0.9798
    Epoch [2/50], Test Losses: mse: 0.1771, mae: 0.2885, huber: 0.0847, swd: 0.0674, target_std: 0.7349
      Epoch 2 composite train-obj: 0.113466
            Val objective improved 0.1326 → 0.1289, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.2838, mae: 0.3105, huber: 0.1090, swd: 0.1069, target_std: 0.7878
    Epoch [3/50], Val Losses: mse: 0.2659, mae: 0.3520, huber: 0.1252, swd: 0.1068, target_std: 0.9798
    Epoch [3/50], Test Losses: mse: 0.1704, mae: 0.2825, huber: 0.0816, swd: 0.0626, target_std: 0.7349
      Epoch 3 composite train-obj: 0.108990
            Val objective improved 0.1289 → 0.1252, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2794, mae: 0.3069, huber: 0.1072, swd: 0.1053, target_std: 0.7879
    Epoch [4/50], Val Losses: mse: 0.2623, mae: 0.3491, huber: 0.1238, swd: 0.1040, target_std: 0.9798
    Epoch [4/50], Test Losses: mse: 0.1698, mae: 0.2815, huber: 0.0812, swd: 0.0617, target_std: 0.7349
      Epoch 4 composite train-obj: 0.107233
            Val objective improved 0.1252 → 0.1238, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.2773, mae: 0.3053, huber: 0.1064, swd: 0.1042, target_std: 0.7878
    Epoch [5/50], Val Losses: mse: 0.2628, mae: 0.3485, huber: 0.1237, swd: 0.1068, target_std: 0.9798
    Epoch [5/50], Test Losses: mse: 0.1725, mae: 0.2848, huber: 0.0826, swd: 0.0650, target_std: 0.7349
      Epoch 5 composite train-obj: 0.106389
            Val objective improved 0.1238 → 0.1237, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.2761, mae: 0.3044, huber: 0.1059, swd: 0.1039, target_std: 0.7878
    Epoch [6/50], Val Losses: mse: 0.2598, mae: 0.3457, huber: 0.1225, swd: 0.1029, target_std: 0.9798
    Epoch [6/50], Test Losses: mse: 0.1726, mae: 0.2850, huber: 0.0826, swd: 0.0640, target_std: 0.7349
      Epoch 6 composite train-obj: 0.105942
            Val objective improved 0.1237 → 0.1225, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.2759, mae: 0.3039, huber: 0.1057, swd: 0.1035, target_std: 0.7879
    Epoch [7/50], Val Losses: mse: 0.2619, mae: 0.3477, huber: 0.1236, swd: 0.1038, target_std: 0.9798
    Epoch [7/50], Test Losses: mse: 0.1697, mae: 0.2815, huber: 0.0812, swd: 0.0629, target_std: 0.7349
      Epoch 7 composite train-obj: 0.105741
            No improvement (0.1236), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.2748, mae: 0.3038, huber: 0.1055, swd: 0.1034, target_std: 0.7879
    Epoch [8/50], Val Losses: mse: 0.2627, mae: 0.3501, huber: 0.1242, swd: 0.1043, target_std: 0.9798
    Epoch [8/50], Test Losses: mse: 0.1662, mae: 0.2786, huber: 0.0796, swd: 0.0597, target_std: 0.7349
      Epoch 8 composite train-obj: 0.105459
            No improvement (0.1242), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.2746, mae: 0.3033, huber: 0.1053, swd: 0.1031, target_std: 0.7879
    Epoch [9/50], Val Losses: mse: 0.2633, mae: 0.3505, huber: 0.1241, swd: 0.1074, target_std: 0.9798
    Epoch [9/50], Test Losses: mse: 0.1710, mae: 0.2837, huber: 0.0819, swd: 0.0649, target_std: 0.7349
      Epoch 9 composite train-obj: 0.105348
            No improvement (0.1241), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.2748, mae: 0.3037, huber: 0.1055, swd: 0.1034, target_std: 0.7879
    Epoch [10/50], Val Losses: mse: 0.2676, mae: 0.3546, huber: 0.1258, swd: 0.1114, target_std: 0.9798
    Epoch [10/50], Test Losses: mse: 0.1701, mae: 0.2825, huber: 0.0814, swd: 0.0653, target_std: 0.7349
      Epoch 10 composite train-obj: 0.105492
            No improvement (0.1258), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.2751, mae: 0.3034, huber: 0.1054, swd: 0.1032, target_std: 0.7880
    Epoch [11/50], Val Losses: mse: 0.2624, mae: 0.3476, huber: 0.1236, swd: 0.1063, target_std: 0.9798
    Epoch [11/50], Test Losses: mse: 0.1724, mae: 0.2833, huber: 0.0824, swd: 0.0666, target_std: 0.7349
      Epoch 11 composite train-obj: 0.105415
    Epoch [11/50], Test Losses: mse: 0.1726, mae: 0.2850, huber: 0.0826, swd: 0.0640, target_std: 0.7349
    Best round's Test MSE: 0.1726, MAE: 0.2850, SWD: 0.0640
    Best round's Validation MSE: 0.2598, MAE: 0.3457
    Best round's Test verification MSE : 0.1726, MAE: 0.2850, SWD: 0.0640
    Time taken: 11.95 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth2_seq96_pred96_20250502_2125)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.1716 ± 0.0007
      mae: 0.2833 ± 0.0013
      huber: 0.0821 ± 0.0004
      swd: 0.0670 ± 0.0021
      target_std: 0.7349 ± 0.0000
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.2604 ± 0.0004
      mae: 0.3468 ± 0.0008
      huber: 0.1228 ± 0.0002
      swd: 0.1100 ± 0.0050
      target_std: 0.9798 ± 0.0000
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 46.78 seconds
    
    Experiment complete: DLinear_etth2_seq96_pred96_20250502_2125
    Model: DLinear
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=96,
    channels=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([96, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 94
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 96, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 96
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 94
    Validation Batches: 13
    Test Batches: 26
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 93.0737, mae: 4.6302, huber: 4.1998, swd: 27.4481, ept: 55.1876
    Epoch [1/50], Val Losses: mse: 23.9270, mae: 3.1699, huber: 2.7403, swd: 9.7228, ept: 60.4289
    Epoch [1/50], Test Losses: mse: 17.7157, mae: 2.6743, huber: 2.2491, swd: 8.6982, ept: 70.6616
      Epoch 1 composite train-obj: 4.199773
            Val objective improved inf → 2.7403, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 26.5558, mae: 2.7809, huber: 2.3666, swd: 14.6541, ept: 74.7911
    Epoch [2/50], Val Losses: mse: 22.4102, mae: 3.0464, huber: 2.6214, swd: 9.4406, ept: 64.3488
    Epoch [2/50], Test Losses: mse: 16.2649, mae: 2.5454, huber: 2.1235, swd: 8.0624, ept: 74.4164
      Epoch 2 composite train-obj: 2.366583
            Val objective improved 2.7403 → 2.6214, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 24.8570, mae: 2.6697, huber: 2.2585, swd: 13.8332, ept: 77.1927
    Epoch [3/50], Val Losses: mse: 21.8033, mae: 2.9860, huber: 2.5639, swd: 9.4103, ept: 66.7634
    Epoch [3/50], Test Losses: mse: 15.4729, mae: 2.4765, huber: 2.0561, swd: 7.6531, ept: 75.9507
      Epoch 3 composite train-obj: 2.258454
            Val objective improved 2.6214 → 2.5639, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 23.9365, mae: 2.6083, huber: 2.1985, swd: 13.3600, ept: 78.4106
    Epoch [4/50], Val Losses: mse: 21.5771, mae: 2.9557, huber: 2.5351, swd: 9.5482, ept: 67.9723
    Epoch [4/50], Test Losses: mse: 15.0213, mae: 2.4329, huber: 2.0143, swd: 7.3883, ept: 76.6268
      Epoch 4 composite train-obj: 2.198485
            Val objective improved 2.5639 → 2.5351, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 23.3519, mae: 2.5706, huber: 2.1617, swd: 13.0261, ept: 79.0269
    Epoch [5/50], Val Losses: mse: 21.5464, mae: 2.9346, huber: 2.5150, swd: 9.6407, ept: 68.6367
    Epoch [5/50], Test Losses: mse: 14.8957, mae: 2.4215, huber: 2.0023, swd: 7.3354, ept: 76.8115
      Epoch 5 composite train-obj: 2.161748
            Val objective improved 2.5351 → 2.5150, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 22.9880, mae: 2.5492, huber: 2.1407, swd: 12.7956, ept: 79.4207
    Epoch [6/50], Val Losses: mse: 21.5724, mae: 2.9310, huber: 2.5117, swd: 9.9030, ept: 69.2127
    Epoch [6/50], Test Losses: mse: 14.6491, mae: 2.4092, huber: 1.9898, swd: 7.2043, ept: 77.3408
      Epoch 6 composite train-obj: 2.140747
            Val objective improved 2.5150 → 2.5117, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 22.8124, mae: 2.5373, huber: 2.1290, swd: 12.6914, ept: 79.6118
    Epoch [7/50], Val Losses: mse: 21.7518, mae: 2.9364, huber: 2.5171, swd: 10.0842, ept: 69.3515
    Epoch [7/50], Test Losses: mse: 14.5773, mae: 2.3929, huber: 1.9750, swd: 7.1262, ept: 77.1271
      Epoch 7 composite train-obj: 2.128998
            No improvement (2.5171), counter 1/5
    Epoch [8/50], Train Losses: mse: 22.6843, mae: 2.5266, huber: 2.1185, swd: 12.6054, ept: 79.7203
    Epoch [8/50], Val Losses: mse: 21.7921, mae: 2.9304, huber: 2.5112, swd: 10.1158, ept: 69.5515
    Epoch [8/50], Test Losses: mse: 14.5851, mae: 2.3901, huber: 1.9721, swd: 7.1533, ept: 77.2399
      Epoch 8 composite train-obj: 2.118478
            Val objective improved 2.5117 → 2.5112, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 22.5983, mae: 2.5237, huber: 2.1155, swd: 12.5408, ept: 79.8087
    Epoch [9/50], Val Losses: mse: 22.0058, mae: 2.9480, huber: 2.5285, swd: 10.3901, ept: 69.3814
    Epoch [9/50], Test Losses: mse: 14.5882, mae: 2.3885, huber: 1.9706, swd: 7.1284, ept: 77.0654
      Epoch 9 composite train-obj: 2.115540
            No improvement (2.5285), counter 1/5
    Epoch [10/50], Train Losses: mse: 22.5645, mae: 2.5220, huber: 2.1137, swd: 12.5249, ept: 79.8676
    Epoch [10/50], Val Losses: mse: 21.9155, mae: 2.9305, huber: 2.5110, swd: 10.2845, ept: 69.6481
    Epoch [10/50], Test Losses: mse: 14.5518, mae: 2.3994, huber: 1.9795, swd: 7.1594, ept: 77.0747
      Epoch 10 composite train-obj: 2.113728
            Val objective improved 2.5112 → 2.5110, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 22.5275, mae: 2.5183, huber: 2.1099, swd: 12.4861, ept: 79.8904
    Epoch [11/50], Val Losses: mse: 22.0065, mae: 2.9362, huber: 2.5161, swd: 10.3553, ept: 69.6421
    Epoch [11/50], Test Losses: mse: 14.6330, mae: 2.3971, huber: 1.9780, swd: 7.1619, ept: 76.7593
      Epoch 11 composite train-obj: 2.109856
            No improvement (2.5161), counter 1/5
    Epoch [12/50], Train Losses: mse: 22.5335, mae: 2.5229, huber: 2.1143, swd: 12.4798, ept: 79.8842
    Epoch [12/50], Val Losses: mse: 22.1527, mae: 2.9472, huber: 2.5276, swd: 10.5825, ept: 69.5449
    Epoch [12/50], Test Losses: mse: 14.4872, mae: 2.3874, huber: 1.9696, swd: 7.0798, ept: 76.9513
      Epoch 12 composite train-obj: 2.114321
            No improvement (2.5276), counter 2/5
    Epoch [13/50], Train Losses: mse: 22.4419, mae: 2.5138, huber: 2.1053, swd: 12.4279, ept: 79.9360
    Epoch [13/50], Val Losses: mse: 22.0331, mae: 2.9345, huber: 2.5146, swd: 10.4249, ept: 69.7144
    Epoch [13/50], Test Losses: mse: 14.5031, mae: 2.3888, huber: 1.9704, swd: 7.1174, ept: 76.9732
      Epoch 13 composite train-obj: 2.105310
            No improvement (2.5146), counter 3/5
    Epoch [14/50], Train Losses: mse: 22.4171, mae: 2.5121, huber: 2.1037, swd: 12.4032, ept: 79.9849
    Epoch [14/50], Val Losses: mse: 22.2013, mae: 2.9465, huber: 2.5262, swd: 10.5171, ept: 69.6094
    Epoch [14/50], Test Losses: mse: 14.6228, mae: 2.3922, huber: 1.9731, swd: 7.1957, ept: 76.9881
      Epoch 14 composite train-obj: 2.103657
            No improvement (2.5262), counter 4/5
    Epoch [15/50], Train Losses: mse: 22.4194, mae: 2.5131, huber: 2.1045, swd: 12.4030, ept: 80.0161
    Epoch [15/50], Val Losses: mse: 22.2230, mae: 2.9477, huber: 2.5275, swd: 10.5783, ept: 69.5263
    Epoch [15/50], Test Losses: mse: 14.5062, mae: 2.3826, huber: 1.9645, swd: 7.1132, ept: 77.1739
      Epoch 15 composite train-obj: 2.104459
    Epoch [15/50], Test Losses: mse: 14.5518, mae: 2.3994, huber: 1.9795, swd: 7.1594, ept: 77.0747
    Best round's Test MSE: 14.5518, MAE: 2.3994, SWD: 7.1594
    Best round's Validation MSE: 21.9155, MAE: 2.9305, SWD: 10.2845
    Best round's Test verification MSE : 14.5518, MAE: 2.3994, SWD: 7.1594
    Time taken: 20.88 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 102.9085, mae: 4.7216, huber: 4.2916, swd: 25.7463, ept: 56.0256
    Epoch [1/50], Val Losses: mse: 23.9861, mae: 3.1728, huber: 2.7434, swd: 9.4179, ept: 60.5830
    Epoch [1/50], Test Losses: mse: 17.6318, mae: 2.6651, huber: 2.2404, swd: 8.4430, ept: 71.2565
      Epoch 1 composite train-obj: 4.291579
            Val objective improved inf → 2.7434, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 26.6605, mae: 2.7806, huber: 2.3664, swd: 14.2639, ept: 74.8535
    Epoch [2/50], Val Losses: mse: 22.5016, mae: 3.0520, huber: 2.6272, swd: 9.1177, ept: 64.3700
    Epoch [2/50], Test Losses: mse: 16.2389, mae: 2.5447, huber: 2.1225, swd: 7.8334, ept: 74.4426
      Epoch 2 composite train-obj: 2.366407
            Val objective improved 2.7434 → 2.6272, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 24.9707, mae: 2.6731, huber: 2.2617, swd: 13.4794, ept: 77.1238
    Epoch [3/50], Val Losses: mse: 21.7529, mae: 2.9820, huber: 2.5599, swd: 8.9123, ept: 66.8571
    Epoch [3/50], Test Losses: mse: 15.5563, mae: 2.4803, huber: 2.0598, swd: 7.4705, ept: 75.8755
      Epoch 3 composite train-obj: 2.261664
            Val objective improved 2.6272 → 2.5599, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 23.9734, mae: 2.6110, huber: 2.2013, swd: 12.9631, ept: 78.2537
    Epoch [4/50], Val Losses: mse: 21.5195, mae: 2.9499, huber: 2.5291, swd: 9.0078, ept: 68.1025
    Epoch [4/50], Test Losses: mse: 15.1349, mae: 2.4456, huber: 2.0254, swd: 7.2338, ept: 76.4446
      Epoch 4 composite train-obj: 2.201306
            Val objective improved 2.5599 → 2.5291, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 23.3892, mae: 2.5739, huber: 2.1648, swd: 12.6383, ept: 78.9836
    Epoch [5/50], Val Losses: mse: 21.5310, mae: 2.9425, huber: 2.5226, swd: 9.2101, ept: 68.6913
    Epoch [5/50], Test Losses: mse: 14.8412, mae: 2.4132, huber: 1.9950, swd: 7.0452, ept: 76.7792
      Epoch 5 composite train-obj: 2.164833
            Val objective improved 2.5291 → 2.5226, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 23.0393, mae: 2.5505, huber: 2.1420, swd: 12.4326, ept: 79.3571
    Epoch [6/50], Val Losses: mse: 21.5376, mae: 2.9323, huber: 2.5128, swd: 9.4277, ept: 69.1688
    Epoch [6/50], Test Losses: mse: 14.6800, mae: 2.4157, huber: 1.9969, swd: 7.0250, ept: 77.3320
      Epoch 6 composite train-obj: 2.142008
            Val objective improved 2.5226 → 2.5128, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 22.8359, mae: 2.5385, huber: 2.1300, swd: 12.3095, ept: 79.6274
    Epoch [7/50], Val Losses: mse: 21.7186, mae: 2.9367, huber: 2.5176, swd: 9.6415, ept: 69.2977
    Epoch [7/50], Test Losses: mse: 14.5375, mae: 2.3960, huber: 1.9778, swd: 6.8860, ept: 77.3257
      Epoch 7 composite train-obj: 2.129969
            No improvement (2.5176), counter 1/5
    Epoch [8/50], Train Losses: mse: 22.7498, mae: 2.5349, huber: 2.1262, swd: 12.2403, ept: 79.7154
    Epoch [8/50], Val Losses: mse: 21.6862, mae: 2.9238, huber: 2.5046, swd: 9.5453, ept: 69.7598
    Epoch [8/50], Test Losses: mse: 14.6037, mae: 2.3952, huber: 1.9766, swd: 6.9129, ept: 77.0439
      Epoch 8 composite train-obj: 2.126230
            Val objective improved 2.5128 → 2.5046, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 22.6196, mae: 2.5240, huber: 2.1157, swd: 12.1397, ept: 79.7747
    Epoch [9/50], Val Losses: mse: 21.9006, mae: 2.9413, huber: 2.5220, swd: 9.8151, ept: 69.4163
    Epoch [9/50], Test Losses: mse: 14.4782, mae: 2.3872, huber: 1.9692, swd: 6.8321, ept: 77.5666
      Epoch 9 composite train-obj: 2.115674
            No improvement (2.5220), counter 1/5
    Epoch [10/50], Train Losses: mse: 22.5580, mae: 2.5200, huber: 2.1118, swd: 12.1099, ept: 79.8963
    Epoch [10/50], Val Losses: mse: 21.8882, mae: 2.9281, huber: 2.5090, swd: 9.8127, ept: 69.6816
    Epoch [10/50], Test Losses: mse: 14.5144, mae: 2.3940, huber: 1.9753, swd: 6.8949, ept: 77.2429
      Epoch 10 composite train-obj: 2.111805
            No improvement (2.5090), counter 2/5
    Epoch [11/50], Train Losses: mse: 22.5414, mae: 2.5209, huber: 2.1124, swd: 12.0851, ept: 79.8520
    Epoch [11/50], Val Losses: mse: 21.9515, mae: 2.9323, huber: 2.5128, swd: 9.8416, ept: 69.5974
    Epoch [11/50], Test Losses: mse: 14.4856, mae: 2.3848, huber: 1.9670, swd: 6.8404, ept: 77.1878
      Epoch 11 composite train-obj: 2.112415
            No improvement (2.5128), counter 3/5
    Epoch [12/50], Train Losses: mse: 22.4837, mae: 2.5159, huber: 2.1076, swd: 12.0439, ept: 79.9274
    Epoch [12/50], Val Losses: mse: 22.0674, mae: 2.9426, huber: 2.5228, swd: 9.9741, ept: 69.3593
    Epoch [12/50], Test Losses: mse: 14.4555, mae: 2.3846, huber: 1.9668, swd: 6.8164, ept: 77.3895
      Epoch 12 composite train-obj: 2.107606
            No improvement (2.5228), counter 4/5
    Epoch [13/50], Train Losses: mse: 22.4165, mae: 2.5128, huber: 2.1044, swd: 11.9940, ept: 79.9289
    Epoch [13/50], Val Losses: mse: 22.0301, mae: 2.9344, huber: 2.5147, swd: 9.9339, ept: 69.6334
    Epoch [13/50], Test Losses: mse: 14.4722, mae: 2.3872, huber: 1.9690, swd: 6.8533, ept: 77.1550
      Epoch 13 composite train-obj: 2.104428
    Epoch [13/50], Test Losses: mse: 14.6037, mae: 2.3952, huber: 1.9766, swd: 6.9129, ept: 77.0439
    Best round's Test MSE: 14.6037, MAE: 2.3952, SWD: 6.9129
    Best round's Validation MSE: 21.6862, MAE: 2.9238, SWD: 9.5453
    Best round's Test verification MSE : 14.6037, MAE: 2.3952, SWD: 6.9129
    Time taken: 19.77 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 90.7801, mae: 4.5931, huber: 4.1628, swd: 23.7427, ept: 55.3825
    Epoch [1/50], Val Losses: mse: 24.0901, mae: 3.1811, huber: 2.7517, swd: 8.8554, ept: 60.7454
    Epoch [1/50], Test Losses: mse: 17.6707, mae: 2.6681, huber: 2.2432, swd: 7.8959, ept: 70.9186
      Epoch 1 composite train-obj: 4.162816
            Val objective improved inf → 2.7517, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 26.6810, mae: 2.7841, huber: 2.3697, swd: 13.3536, ept: 74.8418
    Epoch [2/50], Val Losses: mse: 22.5863, mae: 3.0563, huber: 2.6316, swd: 8.5762, ept: 64.4189
    Epoch [2/50], Test Losses: mse: 16.2692, mae: 2.5466, huber: 2.1247, swd: 7.3399, ept: 74.5662
      Epoch 2 composite train-obj: 2.369719
            Val objective improved 2.7517 → 2.6316, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 25.0085, mae: 2.6743, huber: 2.2630, swd: 12.6246, ept: 77.0886
    Epoch [3/50], Val Losses: mse: 21.8811, mae: 2.9932, huber: 2.5709, swd: 8.5305, ept: 66.6618
    Epoch [3/50], Test Losses: mse: 15.4445, mae: 2.4771, huber: 2.0568, swd: 6.9509, ept: 76.2208
      Epoch 3 composite train-obj: 2.262986
            Val objective improved 2.6316 → 2.5709, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 24.0162, mae: 2.6125, huber: 2.2026, swd: 12.1411, ept: 78.2913
    Epoch [4/50], Val Losses: mse: 21.6322, mae: 2.9633, huber: 2.5426, swd: 8.6359, ept: 68.0283
    Epoch [4/50], Test Losses: mse: 15.0103, mae: 2.4380, huber: 2.0188, swd: 6.7322, ept: 76.8044
      Epoch 4 composite train-obj: 2.202601
            Val objective improved 2.5709 → 2.5426, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 23.4122, mae: 2.5760, huber: 2.1667, swd: 11.8321, ept: 78.9690
    Epoch [5/50], Val Losses: mse: 21.5548, mae: 2.9422, huber: 2.5221, swd: 8.7110, ept: 68.7979
    Epoch [5/50], Test Losses: mse: 14.8214, mae: 2.4156, huber: 1.9963, swd: 6.6026, ept: 76.9157
      Epoch 5 composite train-obj: 2.166690
            Val objective improved 2.5426 → 2.5221, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 23.0341, mae: 2.5498, huber: 2.1414, swd: 11.6204, ept: 79.3710
    Epoch [6/50], Val Losses: mse: 21.5647, mae: 2.9252, huber: 2.5056, swd: 8.7689, ept: 69.2835
    Epoch [6/50], Test Losses: mse: 14.7809, mae: 2.4126, huber: 1.9934, swd: 6.5908, ept: 76.8996
      Epoch 6 composite train-obj: 2.141353
            Val objective improved 2.5221 → 2.5056, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 22.8495, mae: 2.5379, huber: 2.1296, swd: 11.5077, ept: 79.5745
    Epoch [7/50], Val Losses: mse: 21.7068, mae: 2.9325, huber: 2.5135, swd: 9.0598, ept: 69.3096
    Epoch [7/50], Test Losses: mse: 14.5542, mae: 2.3977, huber: 1.9793, swd: 6.4877, ept: 77.3270
      Epoch 7 composite train-obj: 2.129636
            No improvement (2.5135), counter 1/5
    Epoch [8/50], Train Losses: mse: 22.6883, mae: 2.5270, huber: 2.1190, swd: 11.4084, ept: 79.7184
    Epoch [8/50], Val Losses: mse: 21.8654, mae: 2.9394, huber: 2.5202, swd: 9.1573, ept: 69.5802
    Epoch [8/50], Test Losses: mse: 14.5733, mae: 2.3907, huber: 1.9726, swd: 6.4740, ept: 77.2167
      Epoch 8 composite train-obj: 2.119003
            No improvement (2.5202), counter 2/5
    Epoch [9/50], Train Losses: mse: 22.6059, mae: 2.5229, huber: 2.1147, swd: 11.3504, ept: 79.8607
    Epoch [9/50], Val Losses: mse: 21.9295, mae: 2.9401, huber: 2.5205, swd: 9.2552, ept: 69.6985
    Epoch [9/50], Test Losses: mse: 14.5723, mae: 2.3903, huber: 1.9721, swd: 6.4856, ept: 77.2230
      Epoch 9 composite train-obj: 2.114704
            No improvement (2.5205), counter 3/5
    Epoch [10/50], Train Losses: mse: 22.5715, mae: 2.5235, huber: 2.1149, swd: 11.3198, ept: 79.8515
    Epoch [10/50], Val Losses: mse: 21.9915, mae: 2.9402, huber: 2.5204, swd: 9.2687, ept: 69.5791
    Epoch [10/50], Test Losses: mse: 14.5987, mae: 2.3895, huber: 1.9710, swd: 6.4999, ept: 77.0282
      Epoch 10 composite train-obj: 2.114883
            No improvement (2.5204), counter 4/5
    Epoch [11/50], Train Losses: mse: 22.5523, mae: 2.5211, huber: 2.1127, swd: 11.3070, ept: 79.8289
    Epoch [11/50], Val Losses: mse: 22.0083, mae: 2.9368, huber: 2.5173, swd: 9.3224, ept: 69.7214
    Epoch [11/50], Test Losses: mse: 14.5264, mae: 2.3851, huber: 1.9674, swd: 6.4545, ept: 77.2179
      Epoch 11 composite train-obj: 2.112733
    Epoch [11/50], Test Losses: mse: 14.7809, mae: 2.4126, huber: 1.9934, swd: 6.5908, ept: 76.8996
    Best round's Test MSE: 14.7809, MAE: 2.4126, SWD: 6.5908
    Best round's Validation MSE: 21.5647, MAE: 2.9252, SWD: 8.7689
    Best round's Test verification MSE : 14.7809, MAE: 2.4126, SWD: 6.5908
    Time taken: 16.46 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth2_seq96_pred96_20250511_1626)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 14.6455 ± 0.0981
      mae: 2.4024 ± 0.0074
      huber: 1.9831 ± 0.0073
      swd: 6.8877 ± 0.2328
      ept: 77.0061 ± 0.0763
      count: 13.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 21.7221 ± 0.1455
      mae: 2.9265 ± 0.0029
      huber: 2.5071 ± 0.0028
      swd: 9.5329 ± 0.6188
      ept: 69.5638 ± 0.2034
      count: 13.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 57.15 seconds
    
    Experiment complete: DLinear_etth2_seq96_pred96_20250511_1626
    Model: DLinear
    Dataset: etth2
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
    channels=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.4747, mae: 0.4216, huber: 0.1772, swd: 0.1757, target_std: 0.7895
    Epoch [1/50], Val Losses: mse: 0.3596, mae: 0.4244, huber: 0.1666, swd: 0.1559, target_std: 0.9804
    Epoch [1/50], Test Losses: mse: 0.2163, mae: 0.3256, huber: 0.1034, swd: 0.0881, target_std: 0.7299
      Epoch 1 composite train-obj: 0.177218
            Val objective improved inf → 0.1666, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3736, mae: 0.3625, huber: 0.1406, swd: 0.1604, target_std: 0.7894
    Epoch [2/50], Val Losses: mse: 0.3527, mae: 0.4176, huber: 0.1636, swd: 0.1568, target_std: 0.9804
    Epoch [2/50], Test Losses: mse: 0.1991, mae: 0.3082, huber: 0.0955, swd: 0.0798, target_std: 0.7299
      Epoch 2 composite train-obj: 0.140557
            Val objective improved 0.1666 → 0.1636, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3642, mae: 0.3556, huber: 0.1368, swd: 0.1575, target_std: 0.7895
    Epoch [3/50], Val Losses: mse: 0.3436, mae: 0.4101, huber: 0.1591, swd: 0.1567, target_std: 0.9804
    Epoch [3/50], Test Losses: mse: 0.2058, mae: 0.3149, huber: 0.0986, swd: 0.0879, target_std: 0.7299
      Epoch 3 composite train-obj: 0.136848
            Val objective improved 0.1636 → 0.1591, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3606, mae: 0.3529, huber: 0.1354, swd: 0.1562, target_std: 0.7894
    Epoch [4/50], Val Losses: mse: 0.3469, mae: 0.4117, huber: 0.1607, swd: 0.1583, target_std: 0.9804
    Epoch [4/50], Test Losses: mse: 0.1987, mae: 0.3091, huber: 0.0954, swd: 0.0836, target_std: 0.7299
      Epoch 4 composite train-obj: 0.135436
            No improvement (0.1607), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.3589, mae: 0.3515, huber: 0.1347, swd: 0.1552, target_std: 0.7894
    Epoch [5/50], Val Losses: mse: 0.3458, mae: 0.4106, huber: 0.1601, swd: 0.1569, target_std: 0.9804
    Epoch [5/50], Test Losses: mse: 0.1972, mae: 0.3064, huber: 0.0946, swd: 0.0821, target_std: 0.7299
      Epoch 5 composite train-obj: 0.134695
            No improvement (0.1601), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.3579, mae: 0.3507, huber: 0.1343, swd: 0.1548, target_std: 0.7894
    Epoch [6/50], Val Losses: mse: 0.3444, mae: 0.4110, huber: 0.1596, swd: 0.1591, target_std: 0.9804
    Epoch [6/50], Test Losses: mse: 0.1986, mae: 0.3072, huber: 0.0952, swd: 0.0837, target_std: 0.7299
      Epoch 6 composite train-obj: 0.134295
            No improvement (0.1596), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.3569, mae: 0.3503, huber: 0.1340, swd: 0.1547, target_std: 0.7894
    Epoch [7/50], Val Losses: mse: 0.3390, mae: 0.4060, huber: 0.1576, swd: 0.1500, target_std: 0.9804
    Epoch [7/50], Test Losses: mse: 0.1994, mae: 0.3081, huber: 0.0956, swd: 0.0835, target_std: 0.7299
      Epoch 7 composite train-obj: 0.133975
            Val objective improved 0.1591 → 0.1576, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3571, mae: 0.3496, huber: 0.1338, swd: 0.1537, target_std: 0.7894
    Epoch [8/50], Val Losses: mse: 0.3430, mae: 0.4109, huber: 0.1590, swd: 0.1566, target_std: 0.9804
    Epoch [8/50], Test Losses: mse: 0.1987, mae: 0.3067, huber: 0.0953, swd: 0.0828, target_std: 0.7299
      Epoch 8 composite train-obj: 0.133786
            No improvement (0.1590), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3566, mae: 0.3498, huber: 0.1338, swd: 0.1544, target_std: 0.7895
    Epoch [9/50], Val Losses: mse: 0.3402, mae: 0.4079, huber: 0.1582, swd: 0.1520, target_std: 0.9804
    Epoch [9/50], Test Losses: mse: 0.1978, mae: 0.3063, huber: 0.0948, swd: 0.0818, target_std: 0.7299
      Epoch 9 composite train-obj: 0.133759
            No improvement (0.1582), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.3562, mae: 0.3495, huber: 0.1336, swd: 0.1540, target_std: 0.7895
    Epoch [10/50], Val Losses: mse: 0.3403, mae: 0.4073, huber: 0.1582, swd: 0.1524, target_std: 0.9804
    Epoch [10/50], Test Losses: mse: 0.1986, mae: 0.3071, huber: 0.0952, swd: 0.0825, target_std: 0.7299
      Epoch 10 composite train-obj: 0.133649
            No improvement (0.1582), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.3564, mae: 0.3497, huber: 0.1337, swd: 0.1542, target_std: 0.7895
    Epoch [11/50], Val Losses: mse: 0.3408, mae: 0.4054, huber: 0.1578, swd: 0.1561, target_std: 0.9804
    Epoch [11/50], Test Losses: mse: 0.2003, mae: 0.3103, huber: 0.0962, swd: 0.0853, target_std: 0.7299
      Epoch 11 composite train-obj: 0.133689
            No improvement (0.1578), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.3562, mae: 0.3495, huber: 0.1336, swd: 0.1543, target_std: 0.7894
    Epoch [12/50], Val Losses: mse: 0.3377, mae: 0.4057, huber: 0.1572, swd: 0.1467, target_std: 0.9804
    Epoch [12/50], Test Losses: mse: 0.1971, mae: 0.3064, huber: 0.0945, swd: 0.0801, target_std: 0.7299
      Epoch 12 composite train-obj: 0.133606
            Val objective improved 0.1576 → 0.1572, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.3558, mae: 0.3494, huber: 0.1335, swd: 0.1541, target_std: 0.7894
    Epoch [13/50], Val Losses: mse: 0.3443, mae: 0.4093, huber: 0.1594, swd: 0.1579, target_std: 0.9804
    Epoch [13/50], Test Losses: mse: 0.1996, mae: 0.3086, huber: 0.0957, swd: 0.0851, target_std: 0.7299
      Epoch 13 composite train-obj: 0.133483
            No improvement (0.1594), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.3563, mae: 0.3495, huber: 0.1336, swd: 0.1535, target_std: 0.7894
    Epoch [14/50], Val Losses: mse: 0.3482, mae: 0.4132, huber: 0.1612, swd: 0.1619, target_std: 0.9804
    Epoch [14/50], Test Losses: mse: 0.1934, mae: 0.3034, huber: 0.0929, swd: 0.0813, target_std: 0.7299
      Epoch 14 composite train-obj: 0.133568
            No improvement (0.1612), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.3560, mae: 0.3498, huber: 0.1336, swd: 0.1545, target_std: 0.7894
    Epoch [15/50], Val Losses: mse: 0.3401, mae: 0.4088, huber: 0.1582, swd: 0.1514, target_std: 0.9804
    Epoch [15/50], Test Losses: mse: 0.1995, mae: 0.3100, huber: 0.0956, swd: 0.0837, target_std: 0.7299
      Epoch 15 composite train-obj: 0.133629
            No improvement (0.1582), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.3558, mae: 0.3494, huber: 0.1335, swd: 0.1539, target_std: 0.7894
    Epoch [16/50], Val Losses: mse: 0.3458, mae: 0.4116, huber: 0.1599, swd: 0.1641, target_std: 0.9804
    Epoch [16/50], Test Losses: mse: 0.2021, mae: 0.3102, huber: 0.0968, swd: 0.0890, target_std: 0.7299
      Epoch 16 composite train-obj: 0.133488
            No improvement (0.1599), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.3560, mae: 0.3493, huber: 0.1335, swd: 0.1538, target_std: 0.7895
    Epoch [17/50], Val Losses: mse: 0.3480, mae: 0.4119, huber: 0.1608, swd: 0.1647, target_std: 0.9804
    Epoch [17/50], Test Losses: mse: 0.1979, mae: 0.3069, huber: 0.0949, swd: 0.0856, target_std: 0.7299
      Epoch 17 composite train-obj: 0.133517
    Epoch [17/50], Test Losses: mse: 0.1971, mae: 0.3064, huber: 0.0945, swd: 0.0801, target_std: 0.7299
    Best round's Test MSE: 0.1971, MAE: 0.3064, SWD: 0.0801
    Best round's Validation MSE: 0.3377, MAE: 0.4057
    Best round's Test verification MSE : 0.1971, MAE: 0.3064, SWD: 0.0801
    Time taken: 18.57 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4684, mae: 0.4191, huber: 0.1755, swd: 0.1792, target_std: 0.7894
    Epoch [1/50], Val Losses: mse: 0.3637, mae: 0.4268, huber: 0.1680, swd: 0.1679, target_std: 0.9804
    Epoch [1/50], Test Losses: mse: 0.2142, mae: 0.3237, huber: 0.1025, swd: 0.0910, target_std: 0.7299
      Epoch 1 composite train-obj: 0.175517
            Val objective improved inf → 0.1680, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3732, mae: 0.3622, huber: 0.1404, swd: 0.1628, target_std: 0.7894
    Epoch [2/50], Val Losses: mse: 0.3522, mae: 0.4169, huber: 0.1632, swd: 0.1637, target_std: 0.9804
    Epoch [2/50], Test Losses: mse: 0.2032, mae: 0.3124, huber: 0.0974, swd: 0.0872, target_std: 0.7299
      Epoch 2 composite train-obj: 0.140436
            Val objective improved 0.1680 → 0.1632, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3641, mae: 0.3557, huber: 0.1369, swd: 0.1600, target_std: 0.7894
    Epoch [3/50], Val Losses: mse: 0.3463, mae: 0.4108, huber: 0.1608, swd: 0.1551, target_std: 0.9804
    Epoch [3/50], Test Losses: mse: 0.1956, mae: 0.3054, huber: 0.0940, swd: 0.0805, target_std: 0.7299
      Epoch 3 composite train-obj: 0.136851
            Val objective improved 0.1632 → 0.1608, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3607, mae: 0.3530, huber: 0.1354, swd: 0.1587, target_std: 0.7894
    Epoch [4/50], Val Losses: mse: 0.3469, mae: 0.4111, huber: 0.1605, swd: 0.1656, target_std: 0.9804
    Epoch [4/50], Test Losses: mse: 0.2000, mae: 0.3089, huber: 0.0960, swd: 0.0885, target_std: 0.7299
      Epoch 4 composite train-obj: 0.135448
            Val objective improved 0.1608 → 0.1605, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3590, mae: 0.3514, huber: 0.1347, swd: 0.1575, target_std: 0.7895
    Epoch [5/50], Val Losses: mse: 0.3422, mae: 0.4086, huber: 0.1586, swd: 0.1575, target_std: 0.9804
    Epoch [5/50], Test Losses: mse: 0.1973, mae: 0.3087, huber: 0.0948, swd: 0.0844, target_std: 0.7299
      Epoch 5 composite train-obj: 0.134736
            Val objective improved 0.1605 → 0.1586, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3576, mae: 0.3509, huber: 0.1343, swd: 0.1578, target_std: 0.7894
    Epoch [6/50], Val Losses: mse: 0.3352, mae: 0.4034, huber: 0.1561, swd: 0.1482, target_std: 0.9804
    Epoch [6/50], Test Losses: mse: 0.2006, mae: 0.3085, huber: 0.0961, swd: 0.0844, target_std: 0.7299
      Epoch 6 composite train-obj: 0.134297
            Val objective improved 0.1586 → 0.1561, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3574, mae: 0.3504, huber: 0.1341, swd: 0.1569, target_std: 0.7894
    Epoch [7/50], Val Losses: mse: 0.3359, mae: 0.4039, huber: 0.1566, swd: 0.1472, target_std: 0.9804
    Epoch [7/50], Test Losses: mse: 0.2012, mae: 0.3085, huber: 0.0963, swd: 0.0854, target_std: 0.7299
      Epoch 7 composite train-obj: 0.134115
            No improvement (0.1566), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.3564, mae: 0.3501, huber: 0.1339, swd: 0.1572, target_std: 0.7894
    Epoch [8/50], Val Losses: mse: 0.3425, mae: 0.4077, huber: 0.1590, swd: 0.1565, target_std: 0.9804
    Epoch [8/50], Test Losses: mse: 0.1974, mae: 0.3064, huber: 0.0947, swd: 0.0844, target_std: 0.7299
      Epoch 8 composite train-obj: 0.133857
            No improvement (0.1590), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.3570, mae: 0.3499, huber: 0.1339, swd: 0.1568, target_std: 0.7894
    Epoch [9/50], Val Losses: mse: 0.3363, mae: 0.4042, huber: 0.1563, swd: 0.1553, target_std: 0.9804
    Epoch [9/50], Test Losses: mse: 0.2047, mae: 0.3116, huber: 0.0979, swd: 0.0922, target_std: 0.7299
      Epoch 9 composite train-obj: 0.133869
            No improvement (0.1563), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.3565, mae: 0.3496, huber: 0.1337, swd: 0.1567, target_std: 0.7895
    Epoch [10/50], Val Losses: mse: 0.3416, mae: 0.4069, huber: 0.1579, swd: 0.1608, target_std: 0.9804
    Epoch [10/50], Test Losses: mse: 0.2009, mae: 0.3113, huber: 0.0964, swd: 0.0898, target_std: 0.7299
      Epoch 10 composite train-obj: 0.133665
            No improvement (0.1579), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.3563, mae: 0.3495, huber: 0.1336, swd: 0.1561, target_std: 0.7894
    Epoch [11/50], Val Losses: mse: 0.3412, mae: 0.4090, huber: 0.1581, swd: 0.1608, target_std: 0.9804
    Epoch [11/50], Test Losses: mse: 0.2030, mae: 0.3120, huber: 0.0973, swd: 0.0913, target_std: 0.7299
      Epoch 11 composite train-obj: 0.133618
    Epoch [11/50], Test Losses: mse: 0.2006, mae: 0.3085, huber: 0.0961, swd: 0.0844, target_std: 0.7299
    Best round's Test MSE: 0.2006, MAE: 0.3085, SWD: 0.0844
    Best round's Validation MSE: 0.3352, MAE: 0.4034
    Best round's Test verification MSE : 0.2006, MAE: 0.3085, SWD: 0.0844
    Time taken: 11.02 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.4761, mae: 0.4219, huber: 0.1773, swd: 0.1589, target_std: 0.7894
    Epoch [1/50], Val Losses: mse: 0.3684, mae: 0.4285, huber: 0.1695, swd: 0.1548, target_std: 0.9804
    Epoch [1/50], Test Losses: mse: 0.2164, mae: 0.3249, huber: 0.1035, swd: 0.0824, target_std: 0.7299
      Epoch 1 composite train-obj: 0.177341
            Val objective improved inf → 0.1695, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3743, mae: 0.3626, huber: 0.1407, swd: 0.1475, target_std: 0.7894
    Epoch [2/50], Val Losses: mse: 0.3504, mae: 0.4151, huber: 0.1624, swd: 0.1406, target_std: 0.9804
    Epoch [2/50], Test Losses: mse: 0.2033, mae: 0.3131, huber: 0.0975, swd: 0.0755, target_std: 0.7299
      Epoch 2 composite train-obj: 0.140704
            Val objective improved 0.1695 → 0.1624, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.3644, mae: 0.3557, huber: 0.1369, swd: 0.1449, target_std: 0.7894
    Epoch [3/50], Val Losses: mse: 0.3465, mae: 0.4121, huber: 0.1605, swd: 0.1426, target_std: 0.9804
    Epoch [3/50], Test Losses: mse: 0.2004, mae: 0.3103, huber: 0.0962, swd: 0.0756, target_std: 0.7299
      Epoch 3 composite train-obj: 0.136946
            Val objective improved 0.1624 → 0.1605, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3607, mae: 0.3527, huber: 0.1354, swd: 0.1433, target_std: 0.7894
    Epoch [4/50], Val Losses: mse: 0.3455, mae: 0.4109, huber: 0.1604, swd: 0.1374, target_std: 0.9804
    Epoch [4/50], Test Losses: mse: 0.1951, mae: 0.3050, huber: 0.0937, swd: 0.0701, target_std: 0.7299
      Epoch 4 composite train-obj: 0.135375
            Val objective improved 0.1605 → 0.1604, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.3588, mae: 0.3513, huber: 0.1346, swd: 0.1426, target_std: 0.7894
    Epoch [5/50], Val Losses: mse: 0.3453, mae: 0.4105, huber: 0.1603, swd: 0.1404, target_std: 0.9804
    Epoch [5/50], Test Losses: mse: 0.1946, mae: 0.3043, huber: 0.0935, swd: 0.0715, target_std: 0.7299
      Epoch 5 composite train-obj: 0.134637
            Val objective improved 0.1604 → 0.1603, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.3577, mae: 0.3505, huber: 0.1342, swd: 0.1419, target_std: 0.7894
    Epoch [6/50], Val Losses: mse: 0.3422, mae: 0.4093, huber: 0.1586, swd: 0.1415, target_std: 0.9804
    Epoch [6/50], Test Losses: mse: 0.2030, mae: 0.3138, huber: 0.0974, swd: 0.0785, target_std: 0.7299
      Epoch 6 composite train-obj: 0.134247
            Val objective improved 0.1603 → 0.1586, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.3572, mae: 0.3504, huber: 0.1341, swd: 0.1423, target_std: 0.7894
    Epoch [7/50], Val Losses: mse: 0.3374, mae: 0.4053, huber: 0.1568, swd: 0.1337, target_std: 0.9804
    Epoch [7/50], Test Losses: mse: 0.1982, mae: 0.3062, huber: 0.0950, swd: 0.0735, target_std: 0.7299
      Epoch 7 composite train-obj: 0.134076
            Val objective improved 0.1586 → 0.1568, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.3574, mae: 0.3501, huber: 0.1340, swd: 0.1421, target_std: 0.7894
    Epoch [8/50], Val Losses: mse: 0.3408, mae: 0.4080, huber: 0.1579, swd: 0.1428, target_std: 0.9804
    Epoch [8/50], Test Losses: mse: 0.2026, mae: 0.3107, huber: 0.0970, swd: 0.0798, target_std: 0.7299
      Epoch 8 composite train-obj: 0.134015
            No improvement (0.1579), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.3563, mae: 0.3498, huber: 0.1337, swd: 0.1419, target_std: 0.7894
    Epoch [9/50], Val Losses: mse: 0.3413, mae: 0.4087, huber: 0.1581, swd: 0.1434, target_std: 0.9804
    Epoch [9/50], Test Losses: mse: 0.2052, mae: 0.3134, huber: 0.0982, swd: 0.0805, target_std: 0.7299
      Epoch 9 composite train-obj: 0.133726
            No improvement (0.1581), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.3565, mae: 0.3497, huber: 0.1337, swd: 0.1418, target_std: 0.7894
    Epoch [10/50], Val Losses: mse: 0.3454, mae: 0.4107, huber: 0.1598, swd: 0.1469, target_std: 0.9804
    Epoch [10/50], Test Losses: mse: 0.2008, mae: 0.3111, huber: 0.0963, swd: 0.0784, target_std: 0.7299
      Epoch 10 composite train-obj: 0.133710
            No improvement (0.1598), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.3565, mae: 0.3496, huber: 0.1337, swd: 0.1418, target_std: 0.7895
    Epoch [11/50], Val Losses: mse: 0.3401, mae: 0.4055, huber: 0.1575, swd: 0.1389, target_std: 0.9804
    Epoch [11/50], Test Losses: mse: 0.1993, mae: 0.3076, huber: 0.0956, swd: 0.0744, target_std: 0.7299
      Epoch 11 composite train-obj: 0.133685
            No improvement (0.1575), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.3562, mae: 0.3497, huber: 0.1337, swd: 0.1416, target_std: 0.7894
    Epoch [12/50], Val Losses: mse: 0.3481, mae: 0.4146, huber: 0.1609, swd: 0.1465, target_std: 0.9804
    Epoch [12/50], Test Losses: mse: 0.1993, mae: 0.3096, huber: 0.0956, swd: 0.0774, target_std: 0.7299
      Epoch 12 composite train-obj: 0.133679
    Epoch [12/50], Test Losses: mse: 0.1982, mae: 0.3062, huber: 0.0950, swd: 0.0735, target_std: 0.7299
    Best round's Test MSE: 0.1982, MAE: 0.3062, SWD: 0.0735
    Best round's Validation MSE: 0.3374, MAE: 0.4053
    Best round's Test verification MSE : 0.1982, MAE: 0.3062, SWD: 0.0735
    Time taken: 11.69 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth2_seq96_pred196_20250502_2144)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.1986 ± 0.0015
      mae: 0.3070 ± 0.0011
      huber: 0.0952 ± 0.0006
      swd: 0.0793 ± 0.0045
      target_std: 0.7299 ± 0.0000
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3367 ± 0.0011
      mae: 0.4048 ± 0.0010
      huber: 0.1567 ± 0.0004
      swd: 0.1429 ± 0.0065
      target_std: 0.9804 ± 0.0000
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 41.30 seconds
    
    Experiment complete: DLinear_etth2_seq96_pred196_20250502_2144
    Model: DLinear
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=196,
    channels=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([196, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 93
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 196, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 196
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 93
    Validation Batches: 12
    Test Batches: 25
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 106.1442, mae: 5.0062, huber: 4.5720, swd: 30.6946, ept: 95.0144
    Epoch [1/50], Val Losses: mse: 30.3281, mae: 3.6398, huber: 3.2030, swd: 12.9243, ept: 96.0452
    Epoch [1/50], Test Losses: mse: 19.4913, mae: 2.8561, huber: 2.4270, swd: 9.0874, ept: 119.9118
      Epoch 1 composite train-obj: 4.572016
            Val objective improved inf → 3.2030, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 33.4654, mae: 3.1235, huber: 2.7028, swd: 17.8994, ept: 130.6243
    Epoch [2/50], Val Losses: mse: 28.9970, mae: 3.5491, huber: 3.1154, swd: 12.8324, ept: 102.5836
    Epoch [2/50], Test Losses: mse: 18.3709, mae: 2.7555, huber: 2.3297, swd: 8.7546, ept: 126.0562
      Epoch 2 composite train-obj: 2.702782
            Val objective improved 3.2030 → 3.1154, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 32.0822, mae: 3.0401, huber: 2.6215, swd: 17.3340, ept: 134.9882
    Epoch [3/50], Val Losses: mse: 28.4510, mae: 3.5070, huber: 3.0750, swd: 12.8848, ept: 105.8294
    Epoch [3/50], Test Losses: mse: 17.7392, mae: 2.7038, huber: 2.2787, swd: 8.4841, ept: 129.5695
      Epoch 3 composite train-obj: 2.621523
            Val objective improved 3.1154 → 3.0750, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 31.2470, mae: 2.9930, huber: 2.5754, swd: 16.9281, ept: 137.5037
    Epoch [4/50], Val Losses: mse: 28.2526, mae: 3.4861, huber: 3.0550, swd: 13.0518, ept: 108.0898
    Epoch [4/50], Test Losses: mse: 17.3815, mae: 2.6794, huber: 2.2540, swd: 8.3807, ept: 131.3209
      Epoch 4 composite train-obj: 2.575404
            Val objective improved 3.0750 → 3.0550, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 30.7766, mae: 2.9645, huber: 2.5473, swd: 16.6690, ept: 138.9397
    Epoch [5/50], Val Losses: mse: 28.1238, mae: 3.4695, huber: 3.0388, swd: 13.1785, ept: 109.7616
    Epoch [5/50], Test Losses: mse: 17.2292, mae: 2.6626, huber: 2.2387, swd: 8.3623, ept: 131.9932
      Epoch 5 composite train-obj: 2.547298
            Val objective improved 3.0550 → 3.0388, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.4742, mae: 2.9480, huber: 2.5311, swd: 16.4976, ept: 139.6536
    Epoch [6/50], Val Losses: mse: 28.2419, mae: 3.4699, huber: 3.0392, swd: 13.2234, ept: 109.3031
    Epoch [6/50], Test Losses: mse: 17.2095, mae: 2.6501, huber: 2.2264, swd: 8.2603, ept: 131.9508
      Epoch 6 composite train-obj: 2.531063
            No improvement (3.0392), counter 1/5
    Epoch [7/50], Train Losses: mse: 30.2719, mae: 2.9388, huber: 2.5219, swd: 16.3399, ept: 140.2106
    Epoch [7/50], Val Losses: mse: 28.1014, mae: 3.4542, huber: 3.0237, swd: 13.2975, ept: 110.8865
    Epoch [7/50], Test Losses: mse: 17.1534, mae: 2.6507, huber: 2.2274, swd: 8.3940, ept: 133.1447
      Epoch 7 composite train-obj: 2.521899
            Val objective improved 3.0388 → 3.0237, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 30.1798, mae: 2.9320, huber: 2.5153, swd: 16.2916, ept: 140.3104
    Epoch [8/50], Val Losses: mse: 28.3845, mae: 3.4663, huber: 3.0354, swd: 13.5492, ept: 111.0855
    Epoch [8/50], Test Losses: mse: 16.9186, mae: 2.6353, huber: 2.2112, swd: 8.1450, ept: 132.5664
      Epoch 8 composite train-obj: 2.515347
            No improvement (3.0354), counter 1/5
    Epoch [9/50], Train Losses: mse: 30.0380, mae: 2.9233, huber: 2.5066, swd: 16.1839, ept: 140.7778
    Epoch [9/50], Val Losses: mse: 28.4988, mae: 3.4753, huber: 3.0441, swd: 13.6730, ept: 110.8604
    Epoch [9/50], Test Losses: mse: 16.8935, mae: 2.6263, huber: 2.2033, swd: 8.0739, ept: 132.6586
      Epoch 9 composite train-obj: 2.506601
            No improvement (3.0441), counter 2/5
    Epoch [10/50], Train Losses: mse: 29.9760, mae: 2.9204, huber: 2.5037, swd: 16.1283, ept: 140.7678
    Epoch [10/50], Val Losses: mse: 28.5574, mae: 3.4779, huber: 3.0463, swd: 13.7312, ept: 111.4903
    Epoch [10/50], Test Losses: mse: 16.8676, mae: 2.6251, huber: 2.2025, swd: 8.0886, ept: 132.3763
      Epoch 10 composite train-obj: 2.503701
            No improvement (3.0463), counter 3/5
    Epoch [11/50], Train Losses: mse: 29.9665, mae: 2.9244, huber: 2.5072, swd: 16.1124, ept: 140.6839
    Epoch [11/50], Val Losses: mse: 28.4094, mae: 3.4601, huber: 3.0287, swd: 13.6653, ept: 111.4879
    Epoch [11/50], Test Losses: mse: 16.9447, mae: 2.6404, huber: 2.2158, swd: 8.1683, ept: 132.7856
      Epoch 11 composite train-obj: 2.507211
            No improvement (3.0287), counter 4/5
    Epoch [12/50], Train Losses: mse: 29.8868, mae: 2.9183, huber: 2.5012, swd: 16.0594, ept: 140.9390
    Epoch [12/50], Val Losses: mse: 28.5065, mae: 3.4675, huber: 3.0358, swd: 13.6405, ept: 111.5884
    Epoch [12/50], Test Losses: mse: 16.9000, mae: 2.6233, huber: 2.2000, swd: 8.0700, ept: 132.6246
      Epoch 12 composite train-obj: 2.501208
    Epoch [12/50], Test Losses: mse: 17.1534, mae: 2.6507, huber: 2.2274, swd: 8.3940, ept: 133.1447
    Best round's Test MSE: 17.1534, MAE: 2.6507, SWD: 8.3940
    Best round's Validation MSE: 28.1014, MAE: 3.4542, SWD: 13.2975
    Best round's Test verification MSE : 17.1534, MAE: 2.6507, SWD: 8.3940
    Time taken: 19.23 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 96.9787, mae: 4.8537, huber: 4.4197, swd: 31.9745, ept: 95.1796
    Epoch [1/50], Val Losses: mse: 30.1303, mae: 3.6342, huber: 3.1973, swd: 13.2191, ept: 94.9829
    Epoch [1/50], Test Losses: mse: 19.5333, mae: 2.8587, huber: 2.4296, swd: 9.3262, ept: 119.0734
      Epoch 1 composite train-obj: 4.419729
            Val objective improved inf → 3.1973, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 33.3508, mae: 3.1214, huber: 2.7006, swd: 18.4325, ept: 130.3258
    Epoch [2/50], Val Losses: mse: 28.8542, mae: 3.5421, huber: 3.1086, swd: 13.1731, ept: 102.3267
    Epoch [2/50], Test Losses: mse: 18.3560, mae: 2.7564, huber: 2.3304, swd: 8.9838, ept: 126.1211
      Epoch 2 composite train-obj: 2.700569
            Val objective improved 3.1973 → 3.1086, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 31.9636, mae: 3.0357, huber: 2.6171, swd: 17.8457, ept: 135.1499
    Epoch [3/50], Val Losses: mse: 28.4067, mae: 3.5068, huber: 3.0749, swd: 13.4021, ept: 106.3893
    Epoch [3/50], Test Losses: mse: 17.7237, mae: 2.7053, huber: 2.2803, swd: 8.7109, ept: 129.8134
      Epoch 3 composite train-obj: 2.617138
            Val objective improved 3.1086 → 3.0749, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 31.1605, mae: 2.9893, huber: 2.5717, swd: 17.4382, ept: 137.6594
    Epoch [4/50], Val Losses: mse: 28.0116, mae: 3.4713, huber: 3.0403, swd: 13.1858, ept: 108.7247
    Epoch [4/50], Test Losses: mse: 17.5105, mae: 2.6760, huber: 2.2522, swd: 8.6933, ept: 131.3397
      Epoch 4 composite train-obj: 2.571726
            Val objective improved 3.0749 → 3.0403, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 30.7287, mae: 2.9635, huber: 2.5464, swd: 17.1832, ept: 139.0490
    Epoch [5/50], Val Losses: mse: 28.2638, mae: 3.4772, huber: 3.0467, swd: 13.7029, ept: 109.8397
    Epoch [5/50], Test Losses: mse: 17.1707, mae: 2.6630, huber: 2.2382, swd: 8.5299, ept: 132.5336
      Epoch 5 composite train-obj: 2.546420
            No improvement (3.0467), counter 1/5
    Epoch [6/50], Train Losses: mse: 30.4189, mae: 2.9460, huber: 2.5292, swd: 16.9939, ept: 139.8099
    Epoch [6/50], Val Losses: mse: 28.2384, mae: 3.4714, huber: 3.0409, swd: 13.7456, ept: 110.2031
    Epoch [6/50], Test Losses: mse: 17.0386, mae: 2.6435, huber: 2.2202, swd: 8.4311, ept: 132.6937
      Epoch 6 composite train-obj: 2.529191
            No improvement (3.0409), counter 2/5
    Epoch [7/50], Train Losses: mse: 30.2058, mae: 2.9353, huber: 2.5185, swd: 16.8553, ept: 140.3203
    Epoch [7/50], Val Losses: mse: 28.2936, mae: 3.4682, huber: 3.0375, swd: 13.8814, ept: 110.5517
    Epoch [7/50], Test Losses: mse: 16.9579, mae: 2.6389, huber: 2.2148, swd: 8.3729, ept: 133.3352
      Epoch 7 composite train-obj: 2.518484
            Val objective improved 3.0403 → 3.0375, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 30.1240, mae: 2.9318, huber: 2.5148, swd: 16.7950, ept: 140.3528
    Epoch [8/50], Val Losses: mse: 28.2448, mae: 3.4545, huber: 3.0240, swd: 13.9811, ept: 111.8894
    Epoch [8/50], Test Losses: mse: 17.0977, mae: 2.6576, huber: 2.2332, swd: 8.5709, ept: 133.1441
      Epoch 8 composite train-obj: 2.514785
            Val objective improved 3.0375 → 3.0240, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 30.0244, mae: 2.9248, huber: 2.5082, swd: 16.7173, ept: 140.5922
    Epoch [9/50], Val Losses: mse: 28.5686, mae: 3.4784, huber: 3.0473, swd: 14.1654, ept: 111.2072
    Epoch [9/50], Test Losses: mse: 16.9104, mae: 2.6252, huber: 2.2021, swd: 8.2917, ept: 132.7609
      Epoch 9 composite train-obj: 2.508180
            No improvement (3.0473), counter 1/5
    Epoch [10/50], Train Losses: mse: 29.9778, mae: 2.9215, huber: 2.5046, swd: 16.6748, ept: 140.7790
    Epoch [10/50], Val Losses: mse: 28.5022, mae: 3.4686, huber: 3.0373, swd: 14.0242, ept: 111.6375
    Epoch [10/50], Test Losses: mse: 16.9127, mae: 2.6295, huber: 2.2055, swd: 8.3302, ept: 132.9767
      Epoch 10 composite train-obj: 2.504625
            No improvement (3.0373), counter 2/5
    Epoch [11/50], Train Losses: mse: 29.9156, mae: 2.9181, huber: 2.5012, swd: 16.6255, ept: 140.8107
    Epoch [11/50], Val Losses: mse: 28.5847, mae: 3.4738, huber: 3.0423, swd: 14.2950, ept: 112.2609
    Epoch [11/50], Test Losses: mse: 16.8135, mae: 2.6274, huber: 2.2039, swd: 8.3437, ept: 133.5918
      Epoch 11 composite train-obj: 2.501176
            No improvement (3.0423), counter 3/5
    Epoch [12/50], Train Losses: mse: 29.9368, mae: 2.9216, huber: 2.5046, swd: 16.6295, ept: 140.8501
    Epoch [12/50], Val Losses: mse: 28.4473, mae: 3.4620, huber: 3.0299, swd: 13.9468, ept: 111.4364
    Epoch [12/50], Test Losses: mse: 17.0072, mae: 2.6294, huber: 2.2058, swd: 8.4202, ept: 132.0229
      Epoch 12 composite train-obj: 2.504555
            No improvement (3.0299), counter 4/5
    Epoch [13/50], Train Losses: mse: 29.8492, mae: 2.9165, huber: 2.4993, swd: 16.5770, ept: 140.8754
    Epoch [13/50], Val Losses: mse: 28.5993, mae: 3.4711, huber: 3.0391, swd: 14.1800, ept: 111.7653
    Epoch [13/50], Test Losses: mse: 16.8670, mae: 2.6236, huber: 2.2000, swd: 8.2914, ept: 132.4358
      Epoch 13 composite train-obj: 2.499328
    Epoch [13/50], Test Losses: mse: 17.0977, mae: 2.6576, huber: 2.2332, swd: 8.5709, ept: 133.1441
    Best round's Test MSE: 17.0977, MAE: 2.6576, SWD: 8.5709
    Best round's Validation MSE: 28.2448, MAE: 3.4545, SWD: 13.9811
    Best round's Test verification MSE : 17.0977, MAE: 2.6576, SWD: 8.5709
    Time taken: 19.83 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 102.5515, mae: 4.9336, huber: 4.4994, swd: 25.5006, ept: 94.8921
    Epoch [1/50], Val Losses: mse: 30.1188, mae: 3.6310, huber: 3.1940, swd: 11.2925, ept: 96.6226
    Epoch [1/50], Test Losses: mse: 19.6533, mae: 2.8594, huber: 2.4313, swd: 8.1024, ept: 119.3060
      Epoch 1 composite train-obj: 4.499408
            Val objective improved inf → 3.1940, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 33.5790, mae: 3.1285, huber: 2.7076, swd: 15.9539, ept: 130.5410
    Epoch [2/50], Val Losses: mse: 28.9269, mae: 3.5457, huber: 3.1121, swd: 11.3688, ept: 101.9681
    Epoch [2/50], Test Losses: mse: 18.4167, mae: 2.7614, huber: 2.3354, swd: 7.7442, ept: 126.3144
      Epoch 2 composite train-obj: 2.707589
            Val objective improved 3.1940 → 3.1121, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 32.1287, mae: 3.0410, huber: 2.6225, swd: 15.4178, ept: 135.1550
    Epoch [3/50], Val Losses: mse: 28.4595, mae: 3.5087, huber: 3.0769, swd: 11.4407, ept: 105.5312
    Epoch [3/50], Test Losses: mse: 17.6948, mae: 2.7039, huber: 2.2781, swd: 7.4677, ept: 129.6847
      Epoch 3 composite train-obj: 2.622501
            Val objective improved 3.1121 → 3.0769, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 31.2971, mae: 2.9929, huber: 2.5753, swd: 15.0535, ept: 137.4902
    Epoch [4/50], Val Losses: mse: 28.2133, mae: 3.4861, huber: 3.0552, swd: 11.5073, ept: 108.2147
    Epoch [4/50], Test Losses: mse: 17.3481, mae: 2.6690, huber: 2.2453, swd: 7.3515, ept: 131.1970
      Epoch 4 composite train-obj: 2.575348
            Val objective improved 3.0769 → 3.0552, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 30.7772, mae: 2.9642, huber: 2.5473, swd: 14.7871, ept: 138.8883
    Epoch [5/50], Val Losses: mse: 28.2349, mae: 3.4820, huber: 3.0514, swd: 11.6558, ept: 109.3411
    Epoch [5/50], Test Losses: mse: 17.1436, mae: 2.6501, huber: 2.2268, swd: 7.2346, ept: 132.4343
      Epoch 5 composite train-obj: 2.547256
            Val objective improved 3.0552 → 3.0514, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.4082, mae: 2.9433, huber: 2.5267, swd: 14.5907, ept: 139.7285
    Epoch [6/50], Val Losses: mse: 28.1857, mae: 3.4655, huber: 3.0348, swd: 11.7240, ept: 110.4205
    Epoch [6/50], Test Losses: mse: 17.0442, mae: 2.6529, huber: 2.2277, swd: 7.2619, ept: 132.8765
      Epoch 6 composite train-obj: 2.526700
            Val objective improved 3.0514 → 3.0348, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 30.2486, mae: 2.9354, huber: 2.5188, swd: 14.5035, ept: 140.2742
    Epoch [7/50], Val Losses: mse: 28.2531, mae: 3.4671, huber: 3.0358, swd: 11.8137, ept: 110.5779
    Epoch [7/50], Test Losses: mse: 17.0471, mae: 2.6422, huber: 2.2180, swd: 7.2378, ept: 132.6732
      Epoch 7 composite train-obj: 2.518817
            No improvement (3.0358), counter 1/5
    Epoch [8/50], Train Losses: mse: 30.1131, mae: 2.9295, huber: 2.5127, swd: 14.4078, ept: 140.4436
    Epoch [8/50], Val Losses: mse: 28.3508, mae: 3.4687, huber: 3.0378, swd: 11.7952, ept: 110.8597
    Epoch [8/50], Test Losses: mse: 16.9503, mae: 2.6282, huber: 2.2050, swd: 7.1678, ept: 132.6456
      Epoch 8 composite train-obj: 2.512653
            No improvement (3.0378), counter 2/5
    Epoch [9/50], Train Losses: mse: 30.0457, mae: 2.9243, huber: 2.5075, swd: 14.3656, ept: 140.6283
    Epoch [9/50], Val Losses: mse: 28.4778, mae: 3.4721, huber: 3.0413, swd: 12.0494, ept: 111.2362
    Epoch [9/50], Test Losses: mse: 16.8567, mae: 2.6321, huber: 2.2080, swd: 7.1276, ept: 133.1749
      Epoch 9 composite train-obj: 2.507512
            No improvement (3.0413), counter 3/5
    Epoch [10/50], Train Losses: mse: 29.9538, mae: 2.9188, huber: 2.5021, swd: 14.3044, ept: 140.7368
    Epoch [10/50], Val Losses: mse: 28.3110, mae: 3.4549, huber: 3.0239, swd: 11.9772, ept: 112.2284
    Epoch [10/50], Test Losses: mse: 16.9908, mae: 2.6514, huber: 2.2262, swd: 7.3393, ept: 133.5831
      Epoch 10 composite train-obj: 2.502114
            Val objective improved 3.0348 → 3.0239, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 29.9458, mae: 2.9206, huber: 2.5036, swd: 14.2917, ept: 140.7080
    Epoch [11/50], Val Losses: mse: 28.4976, mae: 3.4727, huber: 3.0408, swd: 12.0113, ept: 111.5923
    Epoch [11/50], Test Losses: mse: 16.8916, mae: 2.6228, huber: 2.2005, swd: 7.1437, ept: 132.5913
      Epoch 11 composite train-obj: 2.503570
            No improvement (3.0408), counter 1/5
    Epoch [12/50], Train Losses: mse: 29.8895, mae: 2.9176, huber: 2.5006, swd: 14.2434, ept: 140.8768
    Epoch [12/50], Val Losses: mse: 28.5917, mae: 3.4737, huber: 3.0420, swd: 12.2514, ept: 112.3084
    Epoch [12/50], Test Losses: mse: 16.8621, mae: 2.6314, huber: 2.2081, swd: 7.2040, ept: 133.6329
      Epoch 12 composite train-obj: 2.500553
            No improvement (3.0420), counter 2/5
    Epoch [13/50], Train Losses: mse: 29.8919, mae: 2.9188, huber: 2.5017, swd: 14.2542, ept: 140.8834
    Epoch [13/50], Val Losses: mse: 28.3374, mae: 3.4502, huber: 3.0187, swd: 11.9767, ept: 112.1484
    Epoch [13/50], Test Losses: mse: 16.9957, mae: 2.6417, huber: 2.2177, swd: 7.2944, ept: 133.2962
      Epoch 13 composite train-obj: 2.501706
            Val objective improved 3.0239 → 3.0187, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 29.8338, mae: 2.9154, huber: 2.4984, swd: 14.2194, ept: 141.0692
    Epoch [14/50], Val Losses: mse: 28.5220, mae: 3.4650, huber: 3.0333, swd: 12.0502, ept: 112.2708
    Epoch [14/50], Test Losses: mse: 16.8155, mae: 2.6228, huber: 2.1990, swd: 7.1251, ept: 132.9842
      Epoch 14 composite train-obj: 2.498356
            No improvement (3.0333), counter 1/5
    Epoch [15/50], Train Losses: mse: 29.8341, mae: 2.9143, huber: 2.4972, swd: 14.2052, ept: 140.9411
    Epoch [15/50], Val Losses: mse: 28.4783, mae: 3.4596, huber: 3.0274, swd: 11.9957, ept: 111.8703
    Epoch [15/50], Test Losses: mse: 16.8668, mae: 2.6258, huber: 2.2024, swd: 7.2013, ept: 132.7682
      Epoch 15 composite train-obj: 2.497191
            No improvement (3.0274), counter 2/5
    Epoch [16/50], Train Losses: mse: 29.8086, mae: 2.9158, huber: 2.4985, swd: 14.1830, ept: 140.9778
    Epoch [16/50], Val Losses: mse: 28.5904, mae: 3.4681, huber: 3.0359, swd: 12.0386, ept: 111.7791
    Epoch [16/50], Test Losses: mse: 16.8461, mae: 2.6231, huber: 2.1994, swd: 7.1043, ept: 132.9972
      Epoch 16 composite train-obj: 2.498524
            No improvement (3.0359), counter 3/5
    Epoch [17/50], Train Losses: mse: 29.7636, mae: 2.9110, huber: 2.4937, swd: 14.1578, ept: 141.0134
    Epoch [17/50], Val Losses: mse: 28.6844, mae: 3.4773, huber: 3.0447, swd: 12.0157, ept: 111.2867
    Epoch [17/50], Test Losses: mse: 16.9014, mae: 2.6190, huber: 2.1959, swd: 7.0930, ept: 131.7850
      Epoch 17 composite train-obj: 2.493706
            No improvement (3.0447), counter 4/5
    Epoch [18/50], Train Losses: mse: 29.7705, mae: 2.9142, huber: 2.4967, swd: 14.1418, ept: 141.0308
    Epoch [18/50], Val Losses: mse: 28.7803, mae: 3.4825, huber: 3.0500, swd: 12.2790, ept: 111.8951
    Epoch [18/50], Test Losses: mse: 16.7158, mae: 2.6159, huber: 2.1928, swd: 7.0704, ept: 132.7881
      Epoch 18 composite train-obj: 2.496694
    Epoch [18/50], Test Losses: mse: 16.9957, mae: 2.6417, huber: 2.2177, swd: 7.2944, ept: 133.2962
    Best round's Test MSE: 16.9957, MAE: 2.6417, SWD: 7.2944
    Best round's Validation MSE: 28.3374, MAE: 3.4502, SWD: 11.9767
    Best round's Test verification MSE : 16.9957, MAE: 2.6417, SWD: 7.2944
    Time taken: 26.24 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth2_seq96_pred196_20250511_1627)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 17.0823 ± 0.0653
      mae: 2.6500 ± 0.0065
      huber: 2.2261 ± 0.0064
      swd: 8.0864 ± 0.5647
      ept: 133.1950 ± 0.0716
      count: 12.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 28.2278 ± 0.0971
      mae: 3.4530 ± 0.0020
      huber: 3.0222 ± 0.0024
      swd: 13.0851 ± 0.8319
      ept: 111.6414 ± 0.5442
      count: 12.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 65.34 seconds
    
    Experiment complete: DLinear_etth2_seq96_pred196_20250511_1627
    Model: DLinear
    Dataset: etth2
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
    channels=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.5444, mae: 0.4553, huber: 0.2004, swd: 0.2023, ept: 157.5069
    Epoch [1/50], Val Losses: mse: 0.4094, mae: 0.4586, huber: 0.1872, swd: 0.1873, ept: 143.3072
    Epoch [1/50], Test Losses: mse: 0.2317, mae: 0.3400, huber: 0.1107, swd: 0.0943, ept: 175.5272
      Epoch 1 composite train-obj: 0.200409
            Val objective improved inf → 0.1872, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4509, mae: 0.4029, huber: 0.1668, swd: 0.1911, ept: 196.7295
    Epoch [2/50], Val Losses: mse: 0.3896, mae: 0.4455, huber: 0.1791, swd: 0.1738, ept: 153.3295
    Epoch [2/50], Test Losses: mse: 0.2239, mae: 0.3298, huber: 0.1070, swd: 0.0914, ept: 184.2616
      Epoch 2 composite train-obj: 0.166840
            Val objective improved 0.1872 → 0.1791, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4421, mae: 0.3973, huber: 0.1636, swd: 0.1895, ept: 201.8921
    Epoch [3/50], Val Losses: mse: 0.3909, mae: 0.4454, huber: 0.1794, swd: 0.1789, ept: 156.0831
    Epoch [3/50], Test Losses: mse: 0.2218, mae: 0.3278, huber: 0.1061, swd: 0.0931, ept: 186.4510
      Epoch 3 composite train-obj: 0.163633
            No improvement (0.1794), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.4388, mae: 0.3947, huber: 0.1622, swd: 0.1884, ept: 204.2148
    Epoch [4/50], Val Losses: mse: 0.3824, mae: 0.4396, huber: 0.1766, swd: 0.1621, ept: 158.5372
    Epoch [4/50], Test Losses: mse: 0.2155, mae: 0.3226, huber: 0.1032, swd: 0.0852, ept: 189.7297
      Epoch 4 composite train-obj: 0.162230
            Val objective improved 0.1791 → 0.1766, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4377, mae: 0.3933, huber: 0.1616, swd: 0.1875, ept: 205.3118
    Epoch [5/50], Val Losses: mse: 0.3934, mae: 0.4477, huber: 0.1807, swd: 0.1795, ept: 158.7998
    Epoch [5/50], Test Losses: mse: 0.2146, mae: 0.3213, huber: 0.1028, swd: 0.0882, ept: 188.5330
      Epoch 5 composite train-obj: 0.161612
            No improvement (0.1807), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.4363, mae: 0.3927, huber: 0.1612, swd: 0.1876, ept: 206.0056
    Epoch [6/50], Val Losses: mse: 0.4059, mae: 0.4559, huber: 0.1857, swd: 0.2002, ept: 157.5726
    Epoch [6/50], Test Losses: mse: 0.2203, mae: 0.3284, huber: 0.1054, swd: 0.0974, ept: 188.5001
      Epoch 6 composite train-obj: 0.161158
            No improvement (0.1857), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.4361, mae: 0.3923, huber: 0.1610, swd: 0.1870, ept: 206.2314
    Epoch [7/50], Val Losses: mse: 0.3820, mae: 0.4399, huber: 0.1761, swd: 0.1715, ept: 160.3881
    Epoch [7/50], Test Losses: mse: 0.2201, mae: 0.3272, huber: 0.1054, swd: 0.0926, ept: 189.1222
      Epoch 7 composite train-obj: 0.161008
            Val objective improved 0.1766 → 0.1761, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4359, mae: 0.3924, huber: 0.1610, swd: 0.1874, ept: 206.1400
    Epoch [8/50], Val Losses: mse: 0.4038, mae: 0.4544, huber: 0.1848, swd: 0.1954, ept: 157.3331
    Epoch [8/50], Test Losses: mse: 0.2155, mae: 0.3234, huber: 0.1033, swd: 0.0923, ept: 188.2869
      Epoch 8 composite train-obj: 0.160971
            No improvement (0.1848), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4351, mae: 0.3919, huber: 0.1607, swd: 0.1876, ept: 206.5795
    Epoch [9/50], Val Losses: mse: 0.3869, mae: 0.4436, huber: 0.1784, swd: 0.1695, ept: 159.6673
    Epoch [9/50], Test Losses: mse: 0.2135, mae: 0.3195, huber: 0.1022, swd: 0.0846, ept: 189.7936
      Epoch 9 composite train-obj: 0.160652
            No improvement (0.1784), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.4351, mae: 0.3918, huber: 0.1606, swd: 0.1870, ept: 206.3611
    Epoch [10/50], Val Losses: mse: 0.3907, mae: 0.4456, huber: 0.1797, swd: 0.1760, ept: 160.0416
    Epoch [10/50], Test Losses: mse: 0.2160, mae: 0.3231, huber: 0.1035, swd: 0.0893, ept: 190.0607
      Epoch 10 composite train-obj: 0.160614
            No improvement (0.1797), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.4352, mae: 0.3921, huber: 0.1607, swd: 0.1870, ept: 206.6824
    Epoch [11/50], Val Losses: mse: 0.3964, mae: 0.4481, huber: 0.1815, swd: 0.1901, ept: 160.0449
    Epoch [11/50], Test Losses: mse: 0.2227, mae: 0.3297, huber: 0.1065, swd: 0.0971, ept: 189.4862
      Epoch 11 composite train-obj: 0.160665
            No improvement (0.1815), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.4346, mae: 0.3915, huber: 0.1604, swd: 0.1868, ept: 206.7700
    Epoch [12/50], Val Losses: mse: 0.3857, mae: 0.4426, huber: 0.1777, swd: 0.1709, ept: 160.2245
    Epoch [12/50], Test Losses: mse: 0.2153, mae: 0.3237, huber: 0.1032, swd: 0.0873, ept: 190.9452
      Epoch 12 composite train-obj: 0.160434
    Epoch [12/50], Test Losses: mse: 0.2201, mae: 0.3272, huber: 0.1054, swd: 0.0926, ept: 189.1222
    Best round's Test MSE: 0.2201, MAE: 0.3272, SWD: 0.0926
    Best round's Validation MSE: 0.3820, MAE: 0.4399, SWD: 0.1715
    Best round's Test verification MSE : 0.2201, MAE: 0.3272, SWD: 0.0926
    Time taken: 12.05 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5409, mae: 0.4543, huber: 0.1995, swd: 0.2093, ept: 156.6829
    Epoch [1/50], Val Losses: mse: 0.4032, mae: 0.4554, huber: 0.1851, swd: 0.1809, ept: 144.7071
    Epoch [1/50], Test Losses: mse: 0.2257, mae: 0.3318, huber: 0.1079, swd: 0.0905, ept: 176.4331
      Epoch 1 composite train-obj: 0.199540
            Val objective improved inf → 0.1851, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4500, mae: 0.4027, huber: 0.1667, swd: 0.1971, ept: 196.4111
    Epoch [2/50], Val Losses: mse: 0.3965, mae: 0.4496, huber: 0.1821, swd: 0.1846, ept: 152.8941
    Epoch [2/50], Test Losses: mse: 0.2180, mae: 0.3254, huber: 0.1044, swd: 0.0911, ept: 184.6460
      Epoch 2 composite train-obj: 0.166675
            Val objective improved 0.1851 → 0.1821, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4424, mae: 0.3970, huber: 0.1636, swd: 0.1953, ept: 202.0598
    Epoch [3/50], Val Losses: mse: 0.3953, mae: 0.4490, huber: 0.1815, swd: 0.1869, ept: 156.2691
    Epoch [3/50], Test Losses: mse: 0.2161, mae: 0.3226, huber: 0.1035, swd: 0.0915, ept: 186.8332
      Epoch 3 composite train-obj: 0.163566
            Val objective improved 0.1821 → 0.1815, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4384, mae: 0.3946, huber: 0.1621, swd: 0.1943, ept: 204.0504
    Epoch [4/50], Val Losses: mse: 0.3911, mae: 0.4455, huber: 0.1796, swd: 0.1860, ept: 157.8305
    Epoch [4/50], Test Losses: mse: 0.2210, mae: 0.3279, huber: 0.1058, swd: 0.0989, ept: 187.7590
      Epoch 4 composite train-obj: 0.162139
            Val objective improved 0.1815 → 0.1796, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.4371, mae: 0.3934, huber: 0.1615, swd: 0.1940, ept: 205.1461
    Epoch [5/50], Val Losses: mse: 0.3884, mae: 0.4431, huber: 0.1784, swd: 0.1802, ept: 159.1936
    Epoch [5/50], Test Losses: mse: 0.2169, mae: 0.3234, huber: 0.1039, swd: 0.0923, ept: 189.3980
      Epoch 5 composite train-obj: 0.161470
            Val objective improved 0.1796 → 0.1784, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.4362, mae: 0.3926, huber: 0.1611, swd: 0.1935, ept: 205.8886
    Epoch [6/50], Val Losses: mse: 0.3865, mae: 0.4423, huber: 0.1780, swd: 0.1754, ept: 159.3940
    Epoch [6/50], Test Losses: mse: 0.2143, mae: 0.3219, huber: 0.1027, swd: 0.0897, ept: 190.6175
      Epoch 6 composite train-obj: 0.161117
            Val objective improved 0.1784 → 0.1780, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.4358, mae: 0.3921, huber: 0.1609, swd: 0.1929, ept: 206.2510
    Epoch [7/50], Val Losses: mse: 0.3888, mae: 0.4463, huber: 0.1791, swd: 0.1833, ept: 158.8656
    Epoch [7/50], Test Losses: mse: 0.2174, mae: 0.3246, huber: 0.1041, swd: 0.0945, ept: 189.1733
      Epoch 7 composite train-obj: 0.160864
            No improvement (0.1791), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.4356, mae: 0.3919, huber: 0.1608, swd: 0.1930, ept: 206.2103
    Epoch [8/50], Val Losses: mse: 0.3982, mae: 0.4501, huber: 0.1829, swd: 0.1889, ept: 158.9492
    Epoch [8/50], Test Losses: mse: 0.2107, mae: 0.3181, huber: 0.1011, swd: 0.0890, ept: 190.7419
      Epoch 8 composite train-obj: 0.160781
            No improvement (0.1829), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.4349, mae: 0.3918, huber: 0.1605, swd: 0.1930, ept: 206.5741
    Epoch [9/50], Val Losses: mse: 0.3928, mae: 0.4483, huber: 0.1805, swd: 0.1893, ept: 158.0736
    Epoch [9/50], Test Losses: mse: 0.2183, mae: 0.3252, huber: 0.1045, swd: 0.0977, ept: 188.3561
      Epoch 9 composite train-obj: 0.160543
            No improvement (0.1805), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.4349, mae: 0.3917, huber: 0.1606, swd: 0.1934, ept: 206.5685
    Epoch [10/50], Val Losses: mse: 0.3916, mae: 0.4468, huber: 0.1803, swd: 0.1854, ept: 160.1838
    Epoch [10/50], Test Losses: mse: 0.2130, mae: 0.3196, huber: 0.1020, swd: 0.0914, ept: 190.3104
      Epoch 10 composite train-obj: 0.160594
            No improvement (0.1803), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.4345, mae: 0.3915, huber: 0.1604, swd: 0.1927, ept: 206.6102
    Epoch [11/50], Val Losses: mse: 0.3957, mae: 0.4486, huber: 0.1816, swd: 0.1915, ept: 160.6439
    Epoch [11/50], Test Losses: mse: 0.2157, mae: 0.3238, huber: 0.1034, swd: 0.0949, ept: 189.4865
      Epoch 11 composite train-obj: 0.160424
    Epoch [11/50], Test Losses: mse: 0.2143, mae: 0.3219, huber: 0.1027, swd: 0.0897, ept: 190.6175
    Best round's Test MSE: 0.2143, MAE: 0.3219, SWD: 0.0897
    Best round's Validation MSE: 0.3865, MAE: 0.4423, SWD: 0.1754
    Best round's Test verification MSE : 0.2143, MAE: 0.3219, SWD: 0.0897
    Time taken: 10.76 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.5406, mae: 0.4539, huber: 0.1993, swd: 0.2018, ept: 156.7466
    Epoch [1/50], Val Losses: mse: 0.4129, mae: 0.4600, huber: 0.1887, swd: 0.1835, ept: 144.0463
    Epoch [1/50], Test Losses: mse: 0.2281, mae: 0.3359, huber: 0.1091, swd: 0.0879, ept: 175.4313
      Epoch 1 composite train-obj: 0.199298
            Val objective improved inf → 0.1887, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.4505, mae: 0.4029, huber: 0.1669, swd: 0.1923, ept: 196.4144
    Epoch [2/50], Val Losses: mse: 0.3910, mae: 0.4471, huber: 0.1798, swd: 0.1728, ept: 152.2528
    Epoch [2/50], Test Losses: mse: 0.2215, mae: 0.3273, huber: 0.1059, swd: 0.0871, ept: 183.5568
      Epoch 2 composite train-obj: 0.166851
            Val objective improved 0.1887 → 0.1798, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.4420, mae: 0.3972, huber: 0.1636, swd: 0.1911, ept: 201.7841
    Epoch [3/50], Val Losses: mse: 0.3828, mae: 0.4408, huber: 0.1767, swd: 0.1656, ept: 156.2960
    Epoch [3/50], Test Losses: mse: 0.2198, mae: 0.3256, huber: 0.1051, swd: 0.0867, ept: 186.8520
      Epoch 3 composite train-obj: 0.163577
            Val objective improved 0.1798 → 0.1767, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4387, mae: 0.3946, huber: 0.1622, swd: 0.1901, ept: 204.0192
    Epoch [4/50], Val Losses: mse: 0.3835, mae: 0.4412, huber: 0.1768, swd: 0.1669, ept: 158.6086
    Epoch [4/50], Test Losses: mse: 0.2202, mae: 0.3287, huber: 0.1054, swd: 0.0878, ept: 188.2164
      Epoch 4 composite train-obj: 0.162165
            No improvement (0.1768), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.4370, mae: 0.3932, huber: 0.1615, swd: 0.1894, ept: 205.1956
    Epoch [5/50], Val Losses: mse: 0.3956, mae: 0.4502, huber: 0.1815, swd: 0.1842, ept: 157.3510
    Epoch [5/50], Test Losses: mse: 0.2190, mae: 0.3284, huber: 0.1050, swd: 0.0888, ept: 188.1029
      Epoch 5 composite train-obj: 0.161456
            No improvement (0.1815), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.4363, mae: 0.3927, huber: 0.1611, swd: 0.1891, ept: 205.9612
    Epoch [6/50], Val Losses: mse: 0.3839, mae: 0.4413, huber: 0.1769, swd: 0.1696, ept: 159.6090
    Epoch [6/50], Test Losses: mse: 0.2203, mae: 0.3262, huber: 0.1053, swd: 0.0901, ept: 189.1392
      Epoch 6 composite train-obj: 0.161113
            No improvement (0.1769), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.4354, mae: 0.3920, huber: 0.1608, swd: 0.1892, ept: 206.3788
    Epoch [7/50], Val Losses: mse: 0.3793, mae: 0.4386, huber: 0.1751, swd: 0.1658, ept: 159.0111
    Epoch [7/50], Test Losses: mse: 0.2221, mae: 0.3266, huber: 0.1061, swd: 0.0897, ept: 188.0968
      Epoch 7 composite train-obj: 0.160767
            Val objective improved 0.1767 → 0.1751, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.4355, mae: 0.3921, huber: 0.1608, swd: 0.1886, ept: 206.1989
    Epoch [8/50], Val Losses: mse: 0.3807, mae: 0.4408, huber: 0.1756, swd: 0.1687, ept: 160.0739
    Epoch [8/50], Test Losses: mse: 0.2219, mae: 0.3276, huber: 0.1060, swd: 0.0907, ept: 189.1158
      Epoch 8 composite train-obj: 0.160806
            No improvement (0.1756), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.4348, mae: 0.3916, huber: 0.1605, swd: 0.1883, ept: 206.8251
    Epoch [9/50], Val Losses: mse: 0.3925, mae: 0.4480, huber: 0.1806, swd: 0.1780, ept: 159.0838
    Epoch [9/50], Test Losses: mse: 0.2157, mae: 0.3235, huber: 0.1033, swd: 0.0865, ept: 189.5366
      Epoch 9 composite train-obj: 0.160515
            No improvement (0.1806), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.4344, mae: 0.3915, huber: 0.1604, swd: 0.1888, ept: 206.6103
    Epoch [10/50], Val Losses: mse: 0.3943, mae: 0.4466, huber: 0.1808, swd: 0.1800, ept: 160.2808
    Epoch [10/50], Test Losses: mse: 0.2157, mae: 0.3242, huber: 0.1034, swd: 0.0868, ept: 189.6796
      Epoch 10 composite train-obj: 0.160434
            No improvement (0.1808), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.4347, mae: 0.3918, huber: 0.1605, swd: 0.1885, ept: 206.6074
    Epoch [11/50], Val Losses: mse: 0.3735, mae: 0.4341, huber: 0.1726, swd: 0.1597, ept: 160.9840
    Epoch [11/50], Test Losses: mse: 0.2257, mae: 0.3309, huber: 0.1077, swd: 0.0917, ept: 190.4422
      Epoch 11 composite train-obj: 0.160518
            Val objective improved 0.1751 → 0.1726, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.4352, mae: 0.3918, huber: 0.1607, swd: 0.1891, ept: 206.4581
    Epoch [12/50], Val Losses: mse: 0.3845, mae: 0.4403, huber: 0.1771, swd: 0.1708, ept: 159.9128
    Epoch [12/50], Test Losses: mse: 0.2228, mae: 0.3284, huber: 0.1065, swd: 0.0907, ept: 190.4804
      Epoch 12 composite train-obj: 0.160700
            No improvement (0.1771), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.4347, mae: 0.3918, huber: 0.1605, swd: 0.1888, ept: 206.9997
    Epoch [13/50], Val Losses: mse: 0.3826, mae: 0.4406, huber: 0.1763, swd: 0.1694, ept: 161.4960
    Epoch [13/50], Test Losses: mse: 0.2212, mae: 0.3280, huber: 0.1058, swd: 0.0906, ept: 190.2737
      Epoch 13 composite train-obj: 0.160493
            No improvement (0.1763), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.4349, mae: 0.3914, huber: 0.1605, swd: 0.1885, ept: 206.8638
    Epoch [14/50], Val Losses: mse: 0.3778, mae: 0.4372, huber: 0.1742, swd: 0.1647, ept: 161.5565
    Epoch [14/50], Test Losses: mse: 0.2220, mae: 0.3284, huber: 0.1061, swd: 0.0895, ept: 189.4269
      Epoch 14 composite train-obj: 0.160477
            No improvement (0.1742), counter 3/5
    Epoch [15/50], Train Losses: mse: 0.4345, mae: 0.3918, huber: 0.1606, swd: 0.1892, ept: 206.7004
    Epoch [15/50], Val Losses: mse: 0.3853, mae: 0.4419, huber: 0.1772, swd: 0.1728, ept: 160.7857
    Epoch [15/50], Test Losses: mse: 0.2215, mae: 0.3283, huber: 0.1059, swd: 0.0914, ept: 190.4196
      Epoch 15 composite train-obj: 0.160556
            No improvement (0.1772), counter 4/5
    Epoch [16/50], Train Losses: mse: 0.4345, mae: 0.3912, huber: 0.1603, swd: 0.1879, ept: 206.8763
    Epoch [16/50], Val Losses: mse: 0.4091, mae: 0.4574, huber: 0.1866, swd: 0.1990, ept: 159.9108
    Epoch [16/50], Test Losses: mse: 0.2182, mae: 0.3289, huber: 0.1046, swd: 0.0920, ept: 188.9349
      Epoch 16 composite train-obj: 0.160306
    Epoch [16/50], Test Losses: mse: 0.2257, mae: 0.3309, huber: 0.1077, swd: 0.0917, ept: 190.4422
    Best round's Test MSE: 0.2257, MAE: 0.3309, SWD: 0.0917
    Best round's Validation MSE: 0.3735, MAE: 0.4341, SWD: 0.1597
    Best round's Test verification MSE : 0.2257, MAE: 0.3309, SWD: 0.0917
    Time taken: 16.06 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth2_seq96_pred336_20250510_2052)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2200 ± 0.0046
      mae: 0.3267 ± 0.0037
      huber: 0.1053 ± 0.0020
      swd: 0.0913 ± 0.0012
      ept: 190.0607 ± 0.6674
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3806 ± 0.0054
      mae: 0.4388 ± 0.0034
      huber: 0.1756 ± 0.0022
      swd: 0.1689 ± 0.0067
      ept: 160.2554 ± 0.6559
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 38.90 seconds
    
    Experiment complete: DLinear_etth2_seq96_pred336_20250510_2052
    Model: DLinear
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=336,
    channels=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([336, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 92
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 336, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 336
    Batch Size: 128
    Scaling: No
    Train Split: 0.7
    Val Split: 0.8
    Training Batches: 92
    Validation Batches: 11
    Test Batches: 24
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 111.6802, mae: 5.2451, huber: 4.8066, swd: 33.3044, ept: 137.3492
    Epoch [1/50], Val Losses: mse: 31.2093, mae: 3.7622, huber: 3.3205, swd: 11.1361, ept: 138.4990
    Epoch [1/50], Test Losses: mse: 20.9299, mae: 3.0023, huber: 2.5688, swd: 9.3633, ept: 172.2536
      Epoch 1 composite train-obj: 4.806620
            Val objective improved inf → 3.3205, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 41.6704, mae: 3.4694, huber: 3.0424, swd: 22.0217, ept: 187.7282
    Epoch [2/50], Val Losses: mse: 30.0226, mae: 3.6943, huber: 3.2538, swd: 11.0324, ept: 146.6026
    Epoch [2/50], Test Losses: mse: 19.8150, mae: 2.8961, huber: 2.4659, swd: 8.9334, ept: 181.9002
      Epoch 2 composite train-obj: 3.042374
            Val objective improved 3.3205 → 3.2538, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.3773, mae: 3.3971, huber: 2.9716, swd: 21.5007, ept: 195.0583
    Epoch [3/50], Val Losses: mse: 29.5270, mae: 3.6609, huber: 3.2209, swd: 10.9605, ept: 151.3488
    Epoch [3/50], Test Losses: mse: 19.2285, mae: 2.8417, huber: 2.4129, swd: 8.6735, ept: 186.2918
      Epoch 3 composite train-obj: 2.971610
            Val objective improved 3.2538 → 3.2209, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 39.6157, mae: 3.3555, huber: 2.9310, swd: 21.1284, ept: 198.7920
    Epoch [4/50], Val Losses: mse: 29.1618, mae: 3.6347, huber: 3.1958, swd: 11.0382, ept: 155.9732
    Epoch [4/50], Test Losses: mse: 18.9742, mae: 2.8359, huber: 2.4065, swd: 8.7409, ept: 190.5133
      Epoch 4 composite train-obj: 2.931006
            Val objective improved 3.2209 → 3.1958, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 39.1798, mae: 3.3313, huber: 2.9072, swd: 20.9015, ept: 201.1728
    Epoch [5/50], Val Losses: mse: 29.4518, mae: 3.6572, huber: 3.2176, swd: 11.3812, ept: 156.5556
    Epoch [5/50], Test Losses: mse: 18.5697, mae: 2.8027, huber: 2.3742, swd: 8.3783, ept: 191.8559
      Epoch 5 composite train-obj: 2.907240
            No improvement (3.2176), counter 1/5
    Epoch [6/50], Train Losses: mse: 38.8698, mae: 3.3183, huber: 2.8941, swd: 20.7237, ept: 202.4897
    Epoch [6/50], Val Losses: mse: 29.1719, mae: 3.6246, huber: 3.1854, swd: 11.1343, ept: 158.5121
    Epoch [6/50], Test Losses: mse: 18.6264, mae: 2.8054, huber: 2.3764, swd: 8.5360, ept: 191.9997
      Epoch 6 composite train-obj: 2.894106
            Val objective improved 3.1958 → 3.1854, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 38.6530, mae: 3.3079, huber: 2.8838, swd: 20.5747, ept: 203.2795
    Epoch [7/50], Val Losses: mse: 29.3021, mae: 3.6353, huber: 3.1946, swd: 11.2204, ept: 158.6831
    Epoch [7/50], Test Losses: mse: 18.3634, mae: 2.7789, huber: 2.3501, swd: 8.2869, ept: 191.4925
      Epoch 7 composite train-obj: 2.883764
            No improvement (3.1946), counter 1/5
    Epoch [8/50], Train Losses: mse: 38.5546, mae: 3.3039, huber: 2.8796, swd: 20.4982, ept: 203.7084
    Epoch [8/50], Val Losses: mse: 29.3501, mae: 3.6347, huber: 3.1944, swd: 11.4664, ept: 160.6328
    Epoch [8/50], Test Losses: mse: 18.6409, mae: 2.8137, huber: 2.3844, swd: 8.6429, ept: 193.1138
      Epoch 8 composite train-obj: 2.879615
            No improvement (3.1944), counter 2/5
    Epoch [9/50], Train Losses: mse: 38.4827, mae: 3.3008, huber: 2.8764, swd: 20.4341, ept: 203.8995
    Epoch [9/50], Val Losses: mse: 29.3900, mae: 3.6387, huber: 3.1977, swd: 11.3472, ept: 159.6924
    Epoch [9/50], Test Losses: mse: 18.4862, mae: 2.7835, huber: 2.3558, swd: 8.4050, ept: 192.4178
      Epoch 9 composite train-obj: 2.876395
            No improvement (3.1977), counter 3/5
    Epoch [10/50], Train Losses: mse: 38.3757, mae: 3.2948, huber: 2.8704, swd: 20.3443, ept: 204.0064
    Epoch [10/50], Val Losses: mse: 29.1958, mae: 3.6205, huber: 3.1796, swd: 11.2711, ept: 159.7927
    Epoch [10/50], Test Losses: mse: 18.5191, mae: 2.7883, huber: 2.3599, swd: 8.4609, ept: 193.6876
      Epoch 10 composite train-obj: 2.870434
            Val objective improved 3.1854 → 3.1796, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 38.3214, mae: 3.2932, huber: 2.8685, swd: 20.3224, ept: 204.2988
    Epoch [11/50], Val Losses: mse: 29.3473, mae: 3.6329, huber: 3.1916, swd: 11.3955, ept: 159.7050
    Epoch [11/50], Test Losses: mse: 18.5247, mae: 2.7948, huber: 2.3657, swd: 8.4348, ept: 192.7589
      Epoch 11 composite train-obj: 2.868526
            No improvement (3.1916), counter 1/5
    Epoch [12/50], Train Losses: mse: 38.2441, mae: 3.2902, huber: 2.8656, swd: 20.2504, ept: 204.3669
    Epoch [12/50], Val Losses: mse: 29.1850, mae: 3.6174, huber: 3.1763, swd: 11.4309, ept: 161.6192
    Epoch [12/50], Test Losses: mse: 18.7189, mae: 2.8168, huber: 2.3869, swd: 8.7520, ept: 193.1695
      Epoch 12 composite train-obj: 2.865622
            Val objective improved 3.1796 → 3.1763, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 38.2325, mae: 3.2915, huber: 2.8668, swd: 20.2318, ept: 204.3633
    Epoch [13/50], Val Losses: mse: 29.4370, mae: 3.6352, huber: 3.1922, swd: 11.3138, ept: 160.6520
    Epoch [13/50], Test Losses: mse: 18.3935, mae: 2.7828, huber: 2.3532, swd: 8.4685, ept: 192.3186
      Epoch 13 composite train-obj: 2.866753
            No improvement (3.1922), counter 1/5
    Epoch [14/50], Train Losses: mse: 38.2420, mae: 3.2933, huber: 2.8684, swd: 20.2297, ept: 204.1346
    Epoch [14/50], Val Losses: mse: 29.5831, mae: 3.6467, huber: 3.2048, swd: 11.4785, ept: 159.7069
    Epoch [14/50], Test Losses: mse: 18.1000, mae: 2.7538, huber: 2.3266, swd: 8.1661, ept: 193.2995
      Epoch 14 composite train-obj: 2.868403
            No improvement (3.2048), counter 2/5
    Epoch [15/50], Train Losses: mse: 38.1929, mae: 3.2886, huber: 2.8637, swd: 20.1854, ept: 204.4945
    Epoch [15/50], Val Losses: mse: 29.4122, mae: 3.6370, huber: 3.1940, swd: 11.3255, ept: 160.5872
    Epoch [15/50], Test Losses: mse: 18.4464, mae: 2.7770, huber: 2.3486, swd: 8.4336, ept: 192.1632
      Epoch 15 composite train-obj: 2.863691
            No improvement (3.1940), counter 3/5
    Epoch [16/50], Train Losses: mse: 38.1100, mae: 3.2842, huber: 2.8593, swd: 20.1218, ept: 204.5469
    Epoch [16/50], Val Losses: mse: 29.4298, mae: 3.6370, huber: 3.1947, swd: 11.3349, ept: 160.9195
    Epoch [16/50], Test Losses: mse: 18.4239, mae: 2.7727, huber: 2.3457, swd: 8.4694, ept: 192.7425
      Epoch 16 composite train-obj: 2.859323
            No improvement (3.1947), counter 4/5
    Epoch [17/50], Train Losses: mse: 38.1328, mae: 3.2868, huber: 2.8618, swd: 20.1224, ept: 204.5794
    Epoch [17/50], Val Losses: mse: 29.3496, mae: 3.6288, huber: 3.1864, swd: 11.4415, ept: 160.6854
    Epoch [17/50], Test Losses: mse: 18.5683, mae: 2.8048, huber: 2.3748, swd: 8.5814, ept: 193.0549
      Epoch 17 composite train-obj: 2.861846
    Epoch [17/50], Test Losses: mse: 18.7189, mae: 2.8168, huber: 2.3869, swd: 8.7520, ept: 193.1695
    Best round's Test MSE: 18.7189, MAE: 2.8168, SWD: 8.7520
    Best round's Validation MSE: 29.1850, MAE: 3.6174, SWD: 11.4309
    Best round's Test verification MSE : 18.7189, MAE: 2.8168, SWD: 8.7520
    Time taken: 25.22 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 109.9695, mae: 5.2142, huber: 4.7757, swd: 35.1999, ept: 137.4376
    Epoch [1/50], Val Losses: mse: 31.3664, mae: 3.7737, huber: 3.3315, swd: 11.4624, ept: 139.6767
    Epoch [1/50], Test Losses: mse: 20.8714, mae: 2.9815, huber: 2.5498, swd: 9.7667, ept: 173.6344
      Epoch 1 composite train-obj: 4.775711
            Val objective improved inf → 3.3315, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 41.6600, mae: 3.4677, huber: 3.0406, swd: 23.1104, ept: 188.1119
    Epoch [2/50], Val Losses: mse: 30.0161, mae: 3.6886, huber: 3.2483, swd: 11.3486, ept: 148.3886
    Epoch [2/50], Test Losses: mse: 19.8240, mae: 2.8973, huber: 2.4673, swd: 9.4231, ept: 182.2312
      Epoch 2 composite train-obj: 3.040604
            Val objective improved 3.3315 → 3.2483, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.3917, mae: 3.3958, huber: 2.9705, swd: 22.5559, ept: 195.3854
    Epoch [3/50], Val Losses: mse: 29.5691, mae: 3.6614, huber: 3.2218, swd: 11.3875, ept: 152.5739
    Epoch [3/50], Test Losses: mse: 19.2238, mae: 2.8435, huber: 2.4151, swd: 9.1253, ept: 186.1264
      Epoch 3 composite train-obj: 2.970484
            Val objective improved 3.2483 → 3.2218, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 39.6433, mae: 3.3551, huber: 2.9305, swd: 22.1569, ept: 199.2401
    Epoch [4/50], Val Losses: mse: 29.3657, mae: 3.6470, huber: 3.2076, swd: 11.4988, ept: 155.4365
    Epoch [4/50], Test Losses: mse: 18.8288, mae: 2.8225, huber: 2.3935, swd: 8.9030, ept: 189.6149
      Epoch 4 composite train-obj: 2.930516
            Val objective improved 3.2218 → 3.2076, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 39.1889, mae: 3.3336, huber: 2.9093, swd: 21.9021, ept: 201.0290
    Epoch [5/50], Val Losses: mse: 29.2131, mae: 3.6344, huber: 3.1953, swd: 11.5765, ept: 157.5643
    Epoch [5/50], Test Losses: mse: 18.6406, mae: 2.8033, huber: 2.3749, swd: 8.7647, ept: 191.7058
      Epoch 5 composite train-obj: 2.909265
            Val objective improved 3.2076 → 3.1953, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 38.8698, mae: 3.3175, huber: 2.8934, swd: 21.6835, ept: 202.4521
    Epoch [6/50], Val Losses: mse: 29.1758, mae: 3.6286, huber: 3.1888, swd: 11.5665, ept: 158.7014
    Epoch [6/50], Test Losses: mse: 18.5513, mae: 2.7886, huber: 2.3604, swd: 8.7139, ept: 193.0001
      Epoch 6 composite train-obj: 2.893388
            Val objective improved 3.1953 → 3.1888, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 38.6858, mae: 3.3069, huber: 2.8829, swd: 21.5663, ept: 203.2116
    Epoch [7/50], Val Losses: mse: 29.2580, mae: 3.6300, huber: 3.1901, swd: 11.6465, ept: 158.6262
    Epoch [7/50], Test Losses: mse: 18.4590, mae: 2.7881, huber: 2.3592, swd: 8.5913, ept: 191.5538
      Epoch 7 composite train-obj: 2.882882
            No improvement (3.1901), counter 1/5
    Epoch [8/50], Train Losses: mse: 38.5708, mae: 3.3029, huber: 2.8788, swd: 21.4570, ept: 203.5134
    Epoch [8/50], Val Losses: mse: 29.2538, mae: 3.6380, huber: 3.1978, swd: 11.9457, ept: 160.3115
    Epoch [8/50], Test Losses: mse: 18.8304, mae: 2.8179, huber: 2.3898, swd: 9.0037, ept: 194.2369
      Epoch 8 composite train-obj: 2.878832
            No improvement (3.1978), counter 2/5
    Epoch [9/50], Train Losses: mse: 38.4367, mae: 3.2954, huber: 2.8712, swd: 21.3599, ept: 204.0038
    Epoch [9/50], Val Losses: mse: 29.1758, mae: 3.6216, huber: 3.1811, swd: 11.6454, ept: 160.9445
    Epoch [9/50], Test Losses: mse: 18.5946, mae: 2.7900, huber: 2.3621, swd: 8.9495, ept: 192.4499
      Epoch 9 composite train-obj: 2.871174
            Val objective improved 3.1888 → 3.1811, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 38.3557, mae: 3.2942, huber: 2.8697, swd: 21.2885, ept: 204.3434
    Epoch [10/50], Val Losses: mse: 29.2942, mae: 3.6246, huber: 3.1835, swd: 11.6814, ept: 160.6265
    Epoch [10/50], Test Losses: mse: 18.5699, mae: 2.7844, huber: 2.3569, swd: 8.8923, ept: 192.5734
      Epoch 10 composite train-obj: 2.869745
            No improvement (3.1835), counter 1/5
    Epoch [11/50], Train Losses: mse: 38.3033, mae: 3.2911, huber: 2.8667, swd: 21.2364, ept: 204.3579
    Epoch [11/50], Val Losses: mse: 29.2081, mae: 3.6226, huber: 3.1816, swd: 11.7177, ept: 160.6630
    Epoch [11/50], Test Losses: mse: 18.7557, mae: 2.7979, huber: 2.3702, swd: 8.9529, ept: 192.0573
      Epoch 11 composite train-obj: 2.866705
            No improvement (3.1816), counter 2/5
    Epoch [12/50], Train Losses: mse: 38.2499, mae: 3.2905, huber: 2.8659, swd: 21.1929, ept: 204.4767
    Epoch [12/50], Val Losses: mse: 29.3059, mae: 3.6252, huber: 3.1838, swd: 11.7364, ept: 160.2807
    Epoch [12/50], Test Losses: mse: 18.3983, mae: 2.7745, huber: 2.3465, swd: 8.7081, ept: 191.8297
      Epoch 12 composite train-obj: 2.865872
            No improvement (3.1838), counter 3/5
    Epoch [13/50], Train Losses: mse: 38.3011, mae: 3.2891, huber: 2.8645, swd: 21.2201, ept: 204.5924
    Epoch [13/50], Val Losses: mse: 29.3257, mae: 3.6315, huber: 3.1899, swd: 12.0440, ept: 161.1782
    Epoch [13/50], Test Losses: mse: 19.0033, mae: 2.8251, huber: 2.3975, swd: 9.3336, ept: 192.0480
      Epoch 13 composite train-obj: 2.864493
            No improvement (3.1899), counter 4/5
    Epoch [14/50], Train Losses: mse: 38.2406, mae: 3.2912, huber: 2.8665, swd: 21.1573, ept: 204.3147
    Epoch [14/50], Val Losses: mse: 29.4012, mae: 3.6369, huber: 3.1948, swd: 11.9542, ept: 159.4663
    Epoch [14/50], Test Losses: mse: 18.3627, mae: 2.7787, huber: 2.3506, swd: 8.6608, ept: 193.0845
      Epoch 14 composite train-obj: 2.866489
    Epoch [14/50], Test Losses: mse: 18.5946, mae: 2.7900, huber: 2.3621, swd: 8.9495, ept: 192.4499
    Best round's Test MSE: 18.5946, MAE: 2.7900, SWD: 8.9495
    Best round's Validation MSE: 29.1758, MAE: 3.6216, SWD: 11.6454
    Best round's Test verification MSE : 18.5946, MAE: 2.7900, SWD: 8.9495
    Time taken: 21.30 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 109.6192, mae: 5.2176, huber: 4.7792, swd: 31.7951, ept: 137.1282
    Epoch [1/50], Val Losses: mse: 31.2943, mae: 3.7644, huber: 3.3224, swd: 10.6126, ept: 138.5087
    Epoch [1/50], Test Losses: mse: 20.9305, mae: 2.9903, huber: 2.5578, swd: 8.8526, ept: 172.7248
      Epoch 1 composite train-obj: 4.779196
            Val objective improved inf → 3.3224, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 41.6838, mae: 3.4687, huber: 3.0415, swd: 21.3424, ept: 187.7696
    Epoch [2/50], Val Losses: mse: 30.0157, mae: 3.6945, huber: 3.2541, swd: 10.5959, ept: 146.4215
    Epoch [2/50], Test Losses: mse: 19.8109, mae: 2.8918, huber: 2.4625, swd: 8.4630, ept: 181.2630
      Epoch 2 composite train-obj: 3.041539
            Val objective improved 3.3224 → 3.2541, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.4334, mae: 3.3966, huber: 2.9712, swd: 20.8463, ept: 194.9304
    Epoch [3/50], Val Losses: mse: 29.3801, mae: 3.6554, huber: 3.2160, swd: 10.6684, ept: 153.5750
    Epoch [3/50], Test Losses: mse: 19.5174, mae: 2.8734, huber: 2.4446, swd: 8.5538, ept: 186.8488
      Epoch 3 composite train-obj: 2.971240
            Val objective improved 3.2541 → 3.2160, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 39.6415, mae: 3.3571, huber: 2.9325, swd: 20.4721, ept: 198.8802
    Epoch [4/50], Val Losses: mse: 29.1823, mae: 3.6333, huber: 3.1944, swd: 10.6048, ept: 156.3289
    Epoch [4/50], Test Losses: mse: 19.0164, mae: 2.8490, huber: 2.4182, swd: 8.2937, ept: 188.9033
      Epoch 4 composite train-obj: 2.932463
            Val objective improved 3.2160 → 3.1944, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 39.1957, mae: 3.3325, huber: 2.9083, swd: 20.2463, ept: 201.1923
    Epoch [5/50], Val Losses: mse: 29.0944, mae: 3.6239, huber: 3.1848, swd: 10.7112, ept: 158.2863
    Epoch [5/50], Test Losses: mse: 18.8844, mae: 2.8313, huber: 2.4015, swd: 8.2993, ept: 190.9212
      Epoch 5 composite train-obj: 2.908347
            Val objective improved 3.1944 → 3.1848, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 38.8968, mae: 3.3193, huber: 2.8950, swd: 20.0686, ept: 202.4968
    Epoch [6/50], Val Losses: mse: 29.1051, mae: 3.6255, huber: 3.1860, swd: 10.7193, ept: 159.0213
    Epoch [6/50], Test Losses: mse: 18.5896, mae: 2.7965, huber: 2.3682, swd: 8.0619, ept: 191.8746
      Epoch 6 composite train-obj: 2.895043
            No improvement (3.1860), counter 1/5
    Epoch [7/50], Train Losses: mse: 38.7023, mae: 3.3094, huber: 2.8851, swd: 19.9534, ept: 203.4392
    Epoch [7/50], Val Losses: mse: 29.3780, mae: 3.6435, huber: 3.2028, swd: 10.8423, ept: 158.2879
    Epoch [7/50], Test Losses: mse: 18.4991, mae: 2.7851, huber: 2.3569, swd: 7.8603, ept: 191.5012
      Epoch 7 composite train-obj: 2.885128
            No improvement (3.2028), counter 2/5
    Epoch [8/50], Train Losses: mse: 38.5301, mae: 3.3025, huber: 2.8781, swd: 19.8433, ept: 203.6224
    Epoch [8/50], Val Losses: mse: 29.3084, mae: 3.6357, huber: 3.1953, swd: 10.9698, ept: 159.9082
    Epoch [8/50], Test Losses: mse: 18.3898, mae: 2.7839, huber: 2.3557, swd: 7.9315, ept: 193.0441
      Epoch 8 composite train-obj: 2.878130
            No improvement (3.1953), counter 3/5
    Epoch [9/50], Train Losses: mse: 38.4537, mae: 3.2961, huber: 2.8718, swd: 19.7944, ept: 204.1592
    Epoch [9/50], Val Losses: mse: 29.3727, mae: 3.6385, huber: 3.1981, swd: 11.1848, ept: 160.0429
    Epoch [9/50], Test Losses: mse: 18.4369, mae: 2.7983, huber: 2.3698, swd: 8.0797, ept: 193.5619
      Epoch 9 composite train-obj: 2.871846
            No improvement (3.1981), counter 4/5
    Epoch [10/50], Train Losses: mse: 38.3799, mae: 3.2944, huber: 2.8700, swd: 19.7262, ept: 204.2587
    Epoch [10/50], Val Losses: mse: 29.2799, mae: 3.6280, huber: 3.1866, swd: 10.8534, ept: 159.3364
    Epoch [10/50], Test Losses: mse: 18.7824, mae: 2.8065, huber: 2.3777, swd: 8.1314, ept: 190.4946
      Epoch 10 composite train-obj: 2.870030
    Epoch [10/50], Test Losses: mse: 18.8844, mae: 2.8313, huber: 2.4015, swd: 8.2993, ept: 190.9212
    Best round's Test MSE: 18.8844, MAE: 2.8313, SWD: 8.2993
    Best round's Validation MSE: 29.0944, MAE: 3.6239, SWD: 10.7112
    Best round's Test verification MSE : 18.8844, MAE: 2.8313, SWD: 8.2993
    Time taken: 14.81 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth2_seq96_pred336_20250511_1628)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 18.7326 ± 0.1187
      mae: 2.8127 ± 0.0171
      huber: 2.3835 ± 0.0163
      swd: 8.6669 ± 0.2722
      ept: 192.1802 ± 0.9375
      count: 11.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 29.1517 ± 0.0407
      mae: 3.6210 ± 0.0027
      huber: 3.1807 ± 0.0035
      swd: 11.2625 ± 0.3996
      ept: 160.2833 ± 1.4387
      count: 11.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 61.37 seconds
    
    Experiment complete: DLinear_etth2_seq96_pred336_20250511_1628
    Model: DLinear
    Dataset: etth2
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
    channels=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=True)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([1.0258, 1.0527, 0.8852, 1.0967, 1.0979, 0.8402, 1.0425],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth2
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
    
    Epoch [1/50], Train Losses: mse: 0.6676, mae: 0.5248, huber: 0.2457, swd: 0.2795, ept: 236.6466
    Epoch [1/50], Val Losses: mse: 0.3820, mae: 0.4603, huber: 0.1794, swd: 0.1284, ept: 248.0774
    Epoch [1/50], Test Losses: mse: 0.2696, mae: 0.3671, huber: 0.1277, swd: 0.1038, ept: 287.7181
      Epoch 1 composite train-obj: 0.245712
            Val objective improved inf → 0.1794, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5879, mae: 0.4799, huber: 0.2163, swd: 0.2737, ept: 301.3085
    Epoch [2/50], Val Losses: mse: 0.3683, mae: 0.4511, huber: 0.1737, swd: 0.1199, ept: 258.4907
    Epoch [2/50], Test Losses: mse: 0.2623, mae: 0.3612, huber: 0.1244, swd: 0.1018, ept: 298.9962
      Epoch 2 composite train-obj: 0.216258
            Val objective improved 0.1794 → 0.1737, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5805, mae: 0.4748, huber: 0.2133, swd: 0.2734, ept: 310.2126
    Epoch [3/50], Val Losses: mse: 0.3800, mae: 0.4581, huber: 0.1783, swd: 0.1429, ept: 263.6865
    Epoch [3/50], Test Losses: mse: 0.2588, mae: 0.3591, huber: 0.1230, swd: 0.1059, ept: 302.7055
      Epoch 3 composite train-obj: 0.213296
            No improvement (0.1783), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5775, mae: 0.4725, huber: 0.2120, swd: 0.2717, ept: 314.2113
    Epoch [4/50], Val Losses: mse: 0.3844, mae: 0.4619, huber: 0.1801, swd: 0.1592, ept: 262.3114
    Epoch [4/50], Test Losses: mse: 0.2642, mae: 0.3641, huber: 0.1253, swd: 0.1154, ept: 306.8245
      Epoch 4 composite train-obj: 0.211999
            No improvement (0.1801), counter 2/5
    Epoch [5/50], Train Losses: mse: 0.5762, mae: 0.4718, huber: 0.2116, swd: 0.2728, ept: 316.0373
    Epoch [5/50], Val Losses: mse: 0.3744, mae: 0.4515, huber: 0.1760, swd: 0.1390, ept: 266.4982
    Epoch [5/50], Test Losses: mse: 0.2626, mae: 0.3621, huber: 0.1246, swd: 0.1108, ept: 307.5757
      Epoch 5 composite train-obj: 0.211585
            No improvement (0.1760), counter 3/5
    Epoch [6/50], Train Losses: mse: 0.5758, mae: 0.4710, huber: 0.2112, swd: 0.2715, ept: 317.5471
    Epoch [6/50], Val Losses: mse: 0.3611, mae: 0.4479, huber: 0.1704, swd: 0.1272, ept: 266.8978
    Epoch [6/50], Test Losses: mse: 0.2621, mae: 0.3620, huber: 0.1243, swd: 0.1083, ept: 307.1169
      Epoch 6 composite train-obj: 0.211227
            Val objective improved 0.1737 → 0.1704, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.5748, mae: 0.4707, huber: 0.2109, swd: 0.2717, ept: 317.7991
    Epoch [7/50], Val Losses: mse: 0.3778, mae: 0.4557, huber: 0.1773, swd: 0.1468, ept: 268.0071
    Epoch [7/50], Test Losses: mse: 0.2582, mae: 0.3578, huber: 0.1226, swd: 0.1068, ept: 309.3171
      Epoch 7 composite train-obj: 0.210924
            No improvement (0.1773), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.5748, mae: 0.4706, huber: 0.2109, swd: 0.2715, ept: 318.0207
    Epoch [8/50], Val Losses: mse: 0.3683, mae: 0.4509, huber: 0.1738, swd: 0.1323, ept: 266.1717
    Epoch [8/50], Test Losses: mse: 0.2594, mae: 0.3587, huber: 0.1231, swd: 0.1069, ept: 310.1737
      Epoch 8 composite train-obj: 0.210912
            No improvement (0.1738), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.5740, mae: 0.4703, huber: 0.2107, swd: 0.2716, ept: 318.6593
    Epoch [9/50], Val Losses: mse: 0.3669, mae: 0.4506, huber: 0.1733, swd: 0.1213, ept: 269.2278
    Epoch [9/50], Test Losses: mse: 0.2547, mae: 0.3536, huber: 0.1208, swd: 0.0997, ept: 309.6817
      Epoch 9 composite train-obj: 0.210717
            No improvement (0.1733), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.5738, mae: 0.4703, huber: 0.2106, swd: 0.2714, ept: 318.4821
    Epoch [10/50], Val Losses: mse: 0.3647, mae: 0.4503, huber: 0.1721, swd: 0.1306, ept: 269.0992
    Epoch [10/50], Test Losses: mse: 0.2596, mae: 0.3642, huber: 0.1235, swd: 0.1070, ept: 309.8277
      Epoch 10 composite train-obj: 0.210628
            No improvement (0.1721), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.5737, mae: 0.4699, huber: 0.2105, swd: 0.2709, ept: 318.8806
    Epoch [11/50], Val Losses: mse: 0.3708, mae: 0.4541, huber: 0.1746, swd: 0.1443, ept: 268.1672
    Epoch [11/50], Test Losses: mse: 0.2703, mae: 0.3694, huber: 0.1280, swd: 0.1201, ept: 309.1860
      Epoch 11 composite train-obj: 0.210489
    Epoch [11/50], Test Losses: mse: 0.2621, mae: 0.3620, huber: 0.1243, swd: 0.1083, ept: 307.1169
    Best round's Test MSE: 0.2621, MAE: 0.3620, SWD: 0.1083
    Best round's Validation MSE: 0.3611, MAE: 0.4479, SWD: 0.1272
    Best round's Test verification MSE : 0.2621, MAE: 0.3620, SWD: 0.1083
    Time taken: 11.17 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6671, mae: 0.5249, huber: 0.2456, swd: 0.2690, ept: 239.0493
    Epoch [1/50], Val Losses: mse: 0.4006, mae: 0.4741, huber: 0.1872, swd: 0.1433, ept: 247.1007
    Epoch [1/50], Test Losses: mse: 0.2696, mae: 0.3672, huber: 0.1276, swd: 0.1023, ept: 287.7482
      Epoch 1 composite train-obj: 0.245569
            Val objective improved inf → 0.1872, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5872, mae: 0.4798, huber: 0.2161, swd: 0.2657, ept: 300.4667
    Epoch [2/50], Val Losses: mse: 0.3984, mae: 0.4705, huber: 0.1860, swd: 0.1535, ept: 255.4584
    Epoch [2/50], Test Losses: mse: 0.2679, mae: 0.3668, huber: 0.1269, swd: 0.1087, ept: 299.9776
      Epoch 2 composite train-obj: 0.216069
            Val objective improved 0.1872 → 0.1860, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5803, mae: 0.4750, huber: 0.2133, swd: 0.2654, ept: 309.8550
    Epoch [3/50], Val Losses: mse: 0.3673, mae: 0.4477, huber: 0.1728, swd: 0.1182, ept: 266.5253
    Epoch [3/50], Test Losses: mse: 0.2624, mae: 0.3625, huber: 0.1246, swd: 0.0985, ept: 303.8337
      Epoch 3 composite train-obj: 0.213312
            Val objective improved 0.1860 → 0.1728, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.5778, mae: 0.4727, huber: 0.2122, swd: 0.2640, ept: 313.8285
    Epoch [4/50], Val Losses: mse: 0.3680, mae: 0.4468, huber: 0.1733, swd: 0.1192, ept: 267.8141
    Epoch [4/50], Test Losses: mse: 0.2642, mae: 0.3612, huber: 0.1252, swd: 0.1006, ept: 307.5936
      Epoch 4 composite train-obj: 0.212151
            No improvement (0.1733), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.5763, mae: 0.4717, huber: 0.2116, swd: 0.2640, ept: 315.8946
    Epoch [5/50], Val Losses: mse: 0.3740, mae: 0.4512, huber: 0.1758, swd: 0.1256, ept: 265.6905
    Epoch [5/50], Test Losses: mse: 0.2615, mae: 0.3632, huber: 0.1243, swd: 0.1014, ept: 305.8202
      Epoch 5 composite train-obj: 0.211622
            No improvement (0.1758), counter 2/5
    Epoch [6/50], Train Losses: mse: 0.5752, mae: 0.4710, huber: 0.2111, swd: 0.2632, ept: 317.2029
    Epoch [6/50], Val Losses: mse: 0.3860, mae: 0.4613, huber: 0.1807, swd: 0.1420, ept: 267.4558
    Epoch [6/50], Test Losses: mse: 0.2556, mae: 0.3577, huber: 0.1216, swd: 0.0992, ept: 308.4282
      Epoch 6 composite train-obj: 0.211135
            No improvement (0.1807), counter 3/5
    Epoch [7/50], Train Losses: mse: 0.5754, mae: 0.4709, huber: 0.2112, swd: 0.2639, ept: 318.0069
    Epoch [7/50], Val Losses: mse: 0.3921, mae: 0.4678, huber: 0.1837, swd: 0.1503, ept: 262.7658
    Epoch [7/50], Test Losses: mse: 0.2587, mae: 0.3602, huber: 0.1229, swd: 0.1046, ept: 305.4112
      Epoch 7 composite train-obj: 0.211171
            No improvement (0.1837), counter 4/5
    Epoch [8/50], Train Losses: mse: 0.5748, mae: 0.4706, huber: 0.2109, swd: 0.2637, ept: 318.0129
    Epoch [8/50], Val Losses: mse: 0.3792, mae: 0.4542, huber: 0.1777, swd: 0.1366, ept: 273.9648
    Epoch [8/50], Test Losses: mse: 0.2629, mae: 0.3620, huber: 0.1247, swd: 0.1048, ept: 310.7058
      Epoch 8 composite train-obj: 0.210902
    Epoch [8/50], Test Losses: mse: 0.2624, mae: 0.3625, huber: 0.1246, swd: 0.0985, ept: 303.8337
    Best round's Test MSE: 0.2624, MAE: 0.3625, SWD: 0.0985
    Best round's Validation MSE: 0.3673, MAE: 0.4477, SWD: 0.1182
    Best round's Test verification MSE : 0.2624, MAE: 0.3625, SWD: 0.0985
    Time taken: 8.10 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 0.6692, mae: 0.5251, huber: 0.2459, swd: 0.2929, ept: 236.0008
    Epoch [1/50], Val Losses: mse: 0.3811, mae: 0.4602, huber: 0.1790, swd: 0.1373, ept: 247.0185
    Epoch [1/50], Test Losses: mse: 0.2748, mae: 0.3728, huber: 0.1301, swd: 0.1137, ept: 289.5711
      Epoch 1 composite train-obj: 0.245900
            Val objective improved inf → 0.1790, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.5878, mae: 0.4801, huber: 0.2163, swd: 0.2886, ept: 300.2573
    Epoch [2/50], Val Losses: mse: 0.3757, mae: 0.4559, huber: 0.1766, swd: 0.1322, ept: 261.3776
    Epoch [2/50], Test Losses: mse: 0.2611, mae: 0.3606, huber: 0.1239, swd: 0.1073, ept: 301.8615
      Epoch 2 composite train-obj: 0.216271
            Val objective improved 0.1790 → 0.1766, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.5804, mae: 0.4748, huber: 0.2133, swd: 0.2878, ept: 309.8952
    Epoch [3/50], Val Losses: mse: 0.3893, mae: 0.4625, huber: 0.1818, swd: 0.1622, ept: 262.8847
    Epoch [3/50], Test Losses: mse: 0.2682, mae: 0.3683, huber: 0.1272, swd: 0.1193, ept: 304.2544
      Epoch 3 composite train-obj: 0.213295
            No improvement (0.1818), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.5780, mae: 0.4726, huber: 0.2122, swd: 0.2864, ept: 313.7631
    Epoch [4/50], Val Losses: mse: 0.3710, mae: 0.4510, huber: 0.1745, swd: 0.1322, ept: 268.3921
    Epoch [4/50], Test Losses: mse: 0.2618, mae: 0.3624, huber: 0.1243, swd: 0.1095, ept: 306.0362
      Epoch 4 composite train-obj: 0.212151
            Val objective improved 0.1766 → 0.1745, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.5765, mae: 0.4718, huber: 0.2117, swd: 0.2870, ept: 315.9247
    Epoch [5/50], Val Losses: mse: 0.3707, mae: 0.4502, huber: 0.1745, swd: 0.1300, ept: 266.5960
    Epoch [5/50], Test Losses: mse: 0.2586, mae: 0.3587, huber: 0.1228, swd: 0.1067, ept: 309.0553
      Epoch 5 composite train-obj: 0.211662
            No improvement (0.1745), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.5756, mae: 0.4710, huber: 0.2112, swd: 0.2853, ept: 316.9458
    Epoch [6/50], Val Losses: mse: 0.3718, mae: 0.4550, huber: 0.1750, swd: 0.1364, ept: 265.4922
    Epoch [6/50], Test Losses: mse: 0.2570, mae: 0.3577, huber: 0.1221, swd: 0.1076, ept: 308.4981
      Epoch 6 composite train-obj: 0.211223
            No improvement (0.1750), counter 2/5
    Epoch [7/50], Train Losses: mse: 0.5746, mae: 0.4704, huber: 0.2108, swd: 0.2862, ept: 317.8331
    Epoch [7/50], Val Losses: mse: 0.3769, mae: 0.4552, huber: 0.1772, swd: 0.1404, ept: 270.0724
    Epoch [7/50], Test Losses: mse: 0.2556, mae: 0.3583, huber: 0.1216, swd: 0.1065, ept: 309.2506
      Epoch 7 composite train-obj: 0.210821
            No improvement (0.1772), counter 3/5
    Epoch [8/50], Train Losses: mse: 0.5742, mae: 0.4703, huber: 0.2107, swd: 0.2861, ept: 318.4236
    Epoch [8/50], Val Losses: mse: 0.3958, mae: 0.4646, huber: 0.1848, swd: 0.1629, ept: 269.0108
    Epoch [8/50], Test Losses: mse: 0.2547, mae: 0.3555, huber: 0.1211, swd: 0.1092, ept: 310.0409
      Epoch 8 composite train-obj: 0.210719
            No improvement (0.1848), counter 4/5
    Epoch [9/50], Train Losses: mse: 0.5747, mae: 0.4706, huber: 0.2109, swd: 0.2861, ept: 318.2128
    Epoch [9/50], Val Losses: mse: 0.3823, mae: 0.4634, huber: 0.1799, swd: 0.1465, ept: 265.4741
    Epoch [9/50], Test Losses: mse: 0.2552, mae: 0.3575, huber: 0.1213, swd: 0.1097, ept: 308.0559
      Epoch 9 composite train-obj: 0.210869
    Epoch [9/50], Test Losses: mse: 0.2618, mae: 0.3624, huber: 0.1243, swd: 0.1095, ept: 306.0362
    Best round's Test MSE: 0.2618, MAE: 0.3624, SWD: 0.1095
    Best round's Validation MSE: 0.3710, MAE: 0.4510, SWD: 0.1322
    Best round's Test verification MSE : 0.2618, MAE: 0.3624, SWD: 0.1095
    Time taken: 9.03 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth2_seq96_pred720_20250510_2053)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.2621 ± 0.0002
      mae: 0.3623 ± 0.0002
      huber: 0.1244 ± 0.0001
      swd: 0.1054 ± 0.0049
      ept: 305.6623 ± 1.3662
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.3665 ± 0.0041
      mae: 0.4489 ± 0.0015
      huber: 0.1726 ± 0.0016
      swd: 0.1259 ± 0.0058
      ept: 267.2717 ± 0.8067
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 28.35 seconds
    
    Experiment complete: DLinear_etth2_seq96_pred720_20250510_2053
    Model: DLinear
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    


```python
importlib.reload(monotonic)
importlib.reload(train_config)
utils.reload_modules([utils])
cfg = train_config.FlatDLinearConfig(
    seq_len=96,
    pred_len=720,
    channels=data_mgr.datasets['etth2']['channels'],
    batch_size=128,
    learning_rate=9e-4,
    seeds=[1955, 7, 20],
    epochs=50, 
)
exp = execute_model_evaluation('etth2', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([12194, 7])
    Shape of validation data: torch.Size([1742, 7])
    Shape of testing data: torch.Size([3484, 7])
    global_std.shape: torch.Size([7])
    Global Std for etth2: tensor([10.2186,  6.0203, 13.0564,  4.3659,  6.1442,  6.0126, 11.8879],
           device='cuda:0')
    Train set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Validation set sample shapes: torch.Size([96, 7]), torch.Size([720, 7])
    Test set data shapes: torch.Size([3484, 7]), torch.Size([3484, 7])
    Number of batches in train_loader: 89
    Batch 0: Data shape torch.Size([128, 96, 7]), Target shape torch.Size([128, 720, 7])
    
    ==================================================
    Data Preparation: etth2
    ==================================================
    Sequence Length: 96
    Prediction Length: 720
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
    
    Epoch [1/50], Train Losses: mse: 135.8727, mae: 6.0063, huber: 5.5585, swd: 45.7708, ept: 219.8707
    Epoch [1/50], Val Losses: mse: 32.7059, mae: 3.8404, huber: 3.3973, swd: 12.9330, ept: 250.3483
    Epoch [1/50], Test Losses: mse: 26.5193, mae: 3.4135, huber: 2.9670, swd: 12.5864, ept: 283.1474
      Epoch 1 composite train-obj: 5.558502
            Val objective improved inf → 3.3973, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.8716, mae: 4.2328, huber: 3.7932, swd: 33.3251, ept: 292.1512
    Epoch [2/50], Val Losses: mse: 31.7994, mae: 3.7865, huber: 3.3449, swd: 13.2244, ept: 259.3802
    Epoch [2/50], Test Losses: mse: 25.6692, mae: 3.3595, huber: 2.9135, swd: 12.4887, ept: 294.9839
      Epoch 2 composite train-obj: 3.793189
            Val objective improved 3.3973 → 3.3449, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.7497, mae: 4.1771, huber: 3.7384, swd: 32.9321, ept: 302.7823
    Epoch [3/50], Val Losses: mse: 31.0865, mae: 3.7488, huber: 3.3072, swd: 13.2225, ept: 265.8439
    Epoch [3/50], Test Losses: mse: 25.3602, mae: 3.3292, huber: 2.8848, swd: 12.5804, ept: 302.4776
      Epoch 3 composite train-obj: 3.738357
            Val objective improved 3.3449 → 3.3072, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 58.9415, mae: 4.1396, huber: 3.7014, swd: 32.5515, ept: 308.8852
    Epoch [4/50], Val Losses: mse: 30.9682, mae: 3.7458, huber: 3.3031, swd: 13.6134, ept: 270.4180
    Epoch [4/50], Test Losses: mse: 25.2503, mae: 3.3165, huber: 2.8721, swd: 12.7093, ept: 307.7856
      Epoch 4 composite train-obj: 3.701426
            Val objective improved 3.3072 → 3.3031, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 58.5065, mae: 4.1221, huber: 3.6841, swd: 32.3438, ept: 311.9702
    Epoch [5/50], Val Losses: mse: 31.2725, mae: 3.7619, huber: 3.3202, swd: 13.8510, ept: 275.3993
    Epoch [5/50], Test Losses: mse: 24.9082, mae: 3.3041, huber: 2.8596, swd: 12.3041, ept: 308.0675
      Epoch 5 composite train-obj: 3.684068
            No improvement (3.3202), counter 1/5
    Epoch [6/50], Train Losses: mse: 58.2079, mae: 4.1082, huber: 3.6702, swd: 32.1661, ept: 314.0957
    Epoch [6/50], Val Losses: mse: 30.1286, mae: 3.6941, huber: 3.2503, swd: 12.9598, ept: 276.3452
    Epoch [6/50], Test Losses: mse: 24.6633, mae: 3.2869, huber: 2.8411, swd: 12.2609, ept: 310.3928
      Epoch 6 composite train-obj: 3.670195
            Val objective improved 3.3031 → 3.2503, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 57.9966, mae: 4.1009, huber: 3.6629, swd: 32.0234, ept: 315.1195
    Epoch [7/50], Val Losses: mse: 31.1861, mae: 3.7692, huber: 3.3241, swd: 14.1650, ept: 273.9975
    Epoch [7/50], Test Losses: mse: 24.0447, mae: 3.2307, huber: 2.7885, swd: 11.7406, ept: 313.2876
      Epoch 7 composite train-obj: 3.662895
            No improvement (3.3241), counter 1/5
    Epoch [8/50], Train Losses: mse: 57.7867, mae: 4.0932, huber: 3.6551, swd: 31.9074, ept: 316.0151
    Epoch [8/50], Val Losses: mse: 30.6323, mae: 3.7267, huber: 3.2796, swd: 13.5488, ept: 278.9458
    Epoch [8/50], Test Losses: mse: 24.7065, mae: 3.2598, huber: 2.8184, swd: 12.2933, ept: 314.6641
      Epoch 8 composite train-obj: 3.655052
            No improvement (3.2796), counter 2/5
    Epoch [9/50], Train Losses: mse: 57.6595, mae: 4.0885, huber: 3.6502, swd: 31.8116, ept: 316.3999
    Epoch [9/50], Val Losses: mse: 30.7930, mae: 3.7414, huber: 3.2946, swd: 13.5277, ept: 275.2039
    Epoch [9/50], Test Losses: mse: 23.7374, mae: 3.2187, huber: 2.7743, swd: 11.4554, ept: 312.4169
      Epoch 9 composite train-obj: 3.650215
            No improvement (3.2946), counter 3/5
    Epoch [10/50], Train Losses: mse: 57.5722, mae: 4.0831, huber: 3.6449, swd: 31.7184, ept: 316.9178
    Epoch [10/50], Val Losses: mse: 30.3175, mae: 3.7066, huber: 3.2597, swd: 13.3265, ept: 281.5956
    Epoch [10/50], Test Losses: mse: 24.3822, mae: 3.2499, huber: 2.8068, swd: 12.1053, ept: 312.4514
      Epoch 10 composite train-obj: 3.644888
            No improvement (3.2597), counter 4/5
    Epoch [11/50], Train Losses: mse: 57.4718, mae: 4.0812, huber: 3.6429, swd: 31.6587, ept: 316.8407
    Epoch [11/50], Val Losses: mse: 30.9873, mae: 3.7403, huber: 3.2905, swd: 13.8794, ept: 281.8217
    Epoch [11/50], Test Losses: mse: 24.6743, mae: 3.2646, huber: 2.8216, swd: 12.4416, ept: 314.4885
      Epoch 11 composite train-obj: 3.642879
    Epoch [11/50], Test Losses: mse: 24.6633, mae: 3.2869, huber: 2.8411, swd: 12.2609, ept: 310.3928
    Best round's Test MSE: 24.6633, MAE: 3.2869, SWD: 12.2609
    Best round's Validation MSE: 30.1286, MAE: 3.6941, SWD: 12.9598
    Best round's Test verification MSE : 24.6633, MAE: 3.2869, SWD: 12.2609
    Time taken: 17.02 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 130.0677, mae: 5.9095, huber: 5.4617, swd: 40.5146, ept: 218.9228
    Epoch [1/50], Val Losses: mse: 33.0979, mae: 3.8601, huber: 3.4174, swd: 12.3932, ept: 246.8942
    Epoch [1/50], Test Losses: mse: 26.8323, mae: 3.4555, huber: 3.0070, swd: 11.8541, ept: 283.4136
      Epoch 1 composite train-obj: 5.461748
            Val objective improved inf → 3.4174, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.8175, mae: 4.2328, huber: 3.7931, swd: 30.7951, ept: 291.4317
    Epoch [2/50], Val Losses: mse: 32.3595, mae: 3.8105, huber: 3.3690, swd: 12.7248, ept: 260.0901
    Epoch [2/50], Test Losses: mse: 26.0821, mae: 3.4018, huber: 2.9548, swd: 11.9027, ept: 293.7024
      Epoch 2 composite train-obj: 3.793122
            Val objective improved 3.4174 → 3.3690, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.7127, mae: 4.1802, huber: 3.7411, swd: 30.3905, ept: 302.0212
    Epoch [3/50], Val Losses: mse: 30.9156, mae: 3.7491, huber: 3.3072, swd: 12.0506, ept: 265.1083
    Epoch [3/50], Test Losses: mse: 25.7418, mae: 3.3529, huber: 2.9082, swd: 11.7556, ept: 302.3199
      Epoch 3 composite train-obj: 3.741114
            Val objective improved 3.3690 → 3.3072, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 58.9766, mae: 4.1428, huber: 3.7045, swd: 30.0580, ept: 308.2838
    Epoch [4/50], Val Losses: mse: 30.6763, mae: 3.7463, huber: 3.3030, swd: 11.8636, ept: 269.2474
    Epoch [4/50], Test Losses: mse: 25.1154, mae: 3.2896, huber: 2.8469, swd: 11.3033, ept: 306.8401
      Epoch 4 composite train-obj: 3.704521
            Val objective improved 3.3072 → 3.3030, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 58.4651, mae: 4.1215, huber: 3.6834, swd: 29.8324, ept: 311.6445
    Epoch [5/50], Val Losses: mse: 31.0880, mae: 3.7502, huber: 3.3079, swd: 12.5478, ept: 274.1546
    Epoch [5/50], Test Losses: mse: 24.4684, mae: 3.2805, huber: 2.8357, swd: 11.1286, ept: 309.6456
      Epoch 5 composite train-obj: 3.683411
            No improvement (3.3079), counter 1/5
    Epoch [6/50], Train Losses: mse: 58.1267, mae: 4.1055, huber: 3.6676, swd: 29.6412, ept: 313.8527
    Epoch [6/50], Val Losses: mse: 31.5193, mae: 3.7802, huber: 3.3362, swd: 13.0824, ept: 275.1168
    Epoch [6/50], Test Losses: mse: 24.0607, mae: 3.2484, huber: 2.8043, swd: 10.8456, ept: 311.9649
      Epoch 6 composite train-obj: 3.667628
            No improvement (3.3362), counter 2/5
    Epoch [7/50], Train Losses: mse: 57.9312, mae: 4.0973, huber: 3.6593, swd: 29.5348, ept: 315.4703
    Epoch [7/50], Val Losses: mse: 31.3710, mae: 3.7506, huber: 3.3059, swd: 13.1460, ept: 280.6879
    Epoch [7/50], Test Losses: mse: 25.5343, mae: 3.3427, huber: 2.8973, swd: 12.0766, ept: 312.1987
      Epoch 7 composite train-obj: 3.659313
            No improvement (3.3059), counter 3/5
    Epoch [8/50], Train Losses: mse: 57.8027, mae: 4.0952, huber: 3.6570, swd: 29.4487, ept: 315.6325
    Epoch [8/50], Val Losses: mse: 30.6850, mae: 3.7323, huber: 3.2861, swd: 12.4859, ept: 276.6331
    Epoch [8/50], Test Losses: mse: 23.7402, mae: 3.2014, huber: 2.7600, swd: 10.5068, ept: 314.5947
      Epoch 8 composite train-obj: 3.656969
            Val objective improved 3.3030 → 3.2861, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 57.6046, mae: 4.0866, huber: 3.6484, swd: 29.3133, ept: 316.2845
    Epoch [9/50], Val Losses: mse: 30.6029, mae: 3.7239, huber: 3.2774, swd: 12.4025, ept: 279.3517
    Epoch [9/50], Test Losses: mse: 23.8410, mae: 3.2278, huber: 2.7840, swd: 10.7764, ept: 313.4363
      Epoch 9 composite train-obj: 3.648395
            Val objective improved 3.2861 → 3.2774, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 57.5460, mae: 4.0827, huber: 3.6445, swd: 29.2916, ept: 317.2147
    Epoch [10/50], Val Losses: mse: 30.4081, mae: 3.7077, huber: 3.2597, swd: 12.2861, ept: 281.1925
    Epoch [10/50], Test Losses: mse: 23.8395, mae: 3.2114, huber: 2.7689, swd: 10.8173, ept: 314.3414
      Epoch 10 composite train-obj: 3.644522
            Val objective improved 3.2774 → 3.2597, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 57.4182, mae: 4.0769, huber: 3.6387, swd: 29.1796, ept: 317.3373
    Epoch [11/50], Val Losses: mse: 30.5356, mae: 3.7262, huber: 3.2761, swd: 12.2965, ept: 281.7009
    Epoch [11/50], Test Losses: mse: 23.9229, mae: 3.2143, huber: 2.7716, swd: 10.7629, ept: 312.8426
      Epoch 11 composite train-obj: 3.638735
            No improvement (3.2761), counter 1/5
    Epoch [12/50], Train Losses: mse: 57.3269, mae: 4.0758, huber: 3.6374, swd: 29.0948, ept: 317.0417
    Epoch [12/50], Val Losses: mse: 31.0696, mae: 3.7570, huber: 3.3069, swd: 12.7669, ept: 281.5222
    Epoch [12/50], Test Losses: mse: 24.5168, mae: 3.2509, huber: 2.8080, swd: 11.1589, ept: 311.9294
      Epoch 12 composite train-obj: 3.637420
            No improvement (3.3069), counter 2/5
    Epoch [13/50], Train Losses: mse: 57.2868, mae: 4.0775, huber: 3.6390, swd: 29.0801, ept: 317.4878
    Epoch [13/50], Val Losses: mse: 30.8121, mae: 3.7445, huber: 3.2936, swd: 12.5688, ept: 277.0389
    Epoch [13/50], Test Losses: mse: 23.5781, mae: 3.1849, huber: 2.7437, swd: 10.4063, ept: 314.4332
      Epoch 13 composite train-obj: 3.638974
            No improvement (3.2936), counter 3/5
    Epoch [14/50], Train Losses: mse: 57.2362, mae: 4.0765, huber: 3.6379, swd: 29.0495, ept: 317.2573
    Epoch [14/50], Val Losses: mse: 30.6519, mae: 3.7413, huber: 3.2909, swd: 12.2360, ept: 279.5035
    Epoch [14/50], Test Losses: mse: 24.3294, mae: 3.2578, huber: 2.8124, swd: 10.9505, ept: 313.6293
      Epoch 14 composite train-obj: 3.637899
            No improvement (3.2909), counter 4/5
    Epoch [15/50], Train Losses: mse: 57.1878, mae: 4.0724, huber: 3.6338, swd: 28.9971, ept: 317.3709
    Epoch [15/50], Val Losses: mse: 30.7036, mae: 3.7184, huber: 3.2679, swd: 12.6149, ept: 283.5549
    Epoch [15/50], Test Losses: mse: 24.4060, mae: 3.2602, huber: 2.8156, swd: 11.2564, ept: 312.3903
      Epoch 15 composite train-obj: 3.633831
    Epoch [15/50], Test Losses: mse: 23.8395, mae: 3.2114, huber: 2.7689, swd: 10.8173, ept: 314.3414
    Best round's Test MSE: 23.8395, MAE: 3.2114, SWD: 10.8173
    Best round's Validation MSE: 30.4081, MAE: 3.7077, SWD: 12.2861
    Best round's Test verification MSE : 23.8395, MAE: 3.2114, SWD: 10.8173
    Time taken: 24.05 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 134.5922, mae: 5.9816, huber: 5.5338, swd: 45.9490, ept: 218.5854
    Epoch [1/50], Val Losses: mse: 33.3037, mae: 3.8624, huber: 3.4197, swd: 13.7037, ept: 246.3180
    Epoch [1/50], Test Losses: mse: 27.4282, mae: 3.4980, huber: 3.0487, swd: 13.7405, ept: 283.1090
      Epoch 1 composite train-obj: 5.533812
            Val objective improved inf → 3.4197, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.9478, mae: 4.2348, huber: 3.7952, swd: 35.1329, ept: 292.6982
    Epoch [2/50], Val Losses: mse: 33.0330, mae: 3.8480, huber: 3.4072, swd: 14.5350, ept: 259.4935
    Epoch [2/50], Test Losses: mse: 26.0396, mae: 3.4032, huber: 2.9567, swd: 13.2219, ept: 294.4238
      Epoch 2 composite train-obj: 3.795151
            Val objective improved 3.4197 → 3.4072, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.7483, mae: 4.1760, huber: 3.7372, swd: 34.6568, ept: 302.6318
    Epoch [3/50], Val Losses: mse: 31.7198, mae: 3.7870, huber: 3.3455, swd: 13.9371, ept: 264.0978
    Epoch [3/50], Test Losses: mse: 25.3745, mae: 3.3482, huber: 2.9019, swd: 13.1725, ept: 301.9621
      Epoch 3 composite train-obj: 3.737232
            Val objective improved 3.4072 → 3.3455, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 58.9944, mae: 4.1416, huber: 3.7034, swd: 34.3202, ept: 308.9070
    Epoch [4/50], Val Losses: mse: 30.2929, mae: 3.7180, huber: 3.2759, swd: 12.5820, ept: 268.8221
    Epoch [4/50], Test Losses: mse: 24.3196, mae: 3.2608, huber: 2.8159, swd: 12.1718, ept: 306.6660
      Epoch 4 composite train-obj: 3.703448
            Val objective improved 3.3455 → 3.2759, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 58.5061, mae: 4.1218, huber: 3.6837, swd: 34.0685, ept: 312.5928
    Epoch [5/50], Val Losses: mse: 31.3161, mae: 3.7721, huber: 3.3275, swd: 13.8131, ept: 274.3684
    Epoch [5/50], Test Losses: mse: 25.0212, mae: 3.2984, huber: 2.8548, swd: 12.9618, ept: 309.2587
      Epoch 5 composite train-obj: 3.683713
            No improvement (3.3275), counter 1/5
    Epoch [6/50], Train Losses: mse: 58.2101, mae: 4.1080, huber: 3.6700, swd: 33.8924, ept: 314.0654
    Epoch [6/50], Val Losses: mse: 30.8251, mae: 3.7355, huber: 3.2919, swd: 13.6293, ept: 277.5129
    Epoch [6/50], Test Losses: mse: 24.9932, mae: 3.2990, huber: 2.8547, swd: 13.0442, ept: 311.9746
      Epoch 6 composite train-obj: 3.670036
            No improvement (3.2919), counter 2/5
    Epoch [7/50], Train Losses: mse: 58.0202, mae: 4.1025, huber: 3.6644, swd: 33.7888, ept: 315.5375
    Epoch [7/50], Val Losses: mse: 30.9694, mae: 3.7451, huber: 3.2999, swd: 13.5790, ept: 276.4931
    Epoch [7/50], Test Losses: mse: 24.1508, mae: 3.2517, huber: 2.8079, swd: 12.2530, ept: 311.6602
      Epoch 7 composite train-obj: 3.664388
            No improvement (3.2999), counter 3/5
    Epoch [8/50], Train Losses: mse: 57.7730, mae: 4.0910, huber: 3.6530, swd: 33.6096, ept: 316.4949
    Epoch [8/50], Val Losses: mse: 31.6196, mae: 3.7917, huber: 3.3451, swd: 14.3810, ept: 275.0473
    Epoch [8/50], Test Losses: mse: 24.4358, mae: 3.2631, huber: 2.8204, swd: 12.5320, ept: 312.8359
      Epoch 8 composite train-obj: 3.652977
            No improvement (3.3451), counter 4/5
    Epoch [9/50], Train Losses: mse: 57.6446, mae: 4.0863, huber: 3.6482, swd: 33.5176, ept: 316.1806
    Epoch [9/50], Val Losses: mse: 30.9897, mae: 3.7399, huber: 3.2935, swd: 13.8711, ept: 282.8173
    Epoch [9/50], Test Losses: mse: 24.3711, mae: 3.2710, huber: 2.8257, swd: 12.6875, ept: 313.4931
      Epoch 9 composite train-obj: 3.648169
    Epoch [9/50], Test Losses: mse: 24.3196, mae: 3.2608, huber: 2.8159, swd: 12.1718, ept: 306.6660
    Best round's Test MSE: 24.3196, MAE: 3.2608, SWD: 12.1718
    Best round's Validation MSE: 30.2929, MAE: 3.7180, SWD: 12.5820
    Best round's Test verification MSE : 24.3196, MAE: 3.2608, SWD: 12.1718
    Time taken: 13.72 seconds
    
    ==================================================
    Experiment Summary (DLinear_etth2_seq96_pred720_20250511_1629)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 24.2741 ± 0.3378
      mae: 3.2530 ± 0.0313
      huber: 2.8087 ± 0.0299
      swd: 11.7500 ± 0.6605
      ept: 310.4667 ± 3.1339
      count: 8.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 30.2766 ± 0.1147
      mae: 3.7066 ± 0.0098
      huber: 3.2619 ± 0.0106
      swd: 12.6093 ± 0.2757
      ept: 275.4533 ± 5.0894
      count: 8.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 54.85 seconds
    
    Experiment complete: DLinear_etth2_seq96_pred720_20250511_1629
    Model: DLinear
    Dataset: etth2
    Sequence Length: 96
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    




