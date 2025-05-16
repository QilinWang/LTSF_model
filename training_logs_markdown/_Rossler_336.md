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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 2.1727, mae: 0.5053, huber: 0.3101, swd: 1.8004, ept: 91.4372
    Epoch [1/50], Val Losses: mse: 0.9068, mae: 0.3953, huber: 0.1707, swd: 0.7564, ept: 94.6936
    Epoch [1/50], Test Losses: mse: 0.8480, mae: 0.4188, huber: 0.1838, swd: 0.6979, ept: 94.7616
      Epoch 1 composite train-obj: 0.310081
            Val objective improved inf → 0.1707, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3026, mae: 0.1773, huber: 0.0526, swd: 0.2291, ept: 95.4361
    Epoch [2/50], Val Losses: mse: 0.0775, mae: 0.1490, huber: 0.0301, swd: 0.0443, ept: 95.6707
    Epoch [2/50], Test Losses: mse: 0.0767, mae: 0.1558, huber: 0.0309, swd: 0.0442, ept: 95.7353
      Epoch 2 composite train-obj: 0.052604
            Val objective improved 0.1707 → 0.0301, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0316, mae: 0.0976, huber: 0.0136, swd: 0.0174, ept: 95.9485
    Epoch [3/50], Val Losses: mse: 0.3503, mae: 0.3126, huber: 0.1337, swd: 0.3084, ept: 95.9767
    Epoch [3/50], Test Losses: mse: 0.3712, mae: 0.3279, huber: 0.1431, swd: 0.3279, ept: 95.9755
      Epoch 3 composite train-obj: 0.013585
            No improvement (0.1337), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.0363, mae: 0.1009, huber: 0.0166, swd: 0.0221, ept: 95.9544
    Epoch [4/50], Val Losses: mse: 0.0204, mae: 0.0835, huber: 0.0101, swd: 0.0110, ept: 96.0000
    Epoch [4/50], Test Losses: mse: 0.0231, mae: 0.0907, huber: 0.0115, swd: 0.0125, ept: 96.0000
      Epoch 4 composite train-obj: 0.016586
            Val objective improved 0.0301 → 0.0101, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0133, mae: 0.0689, huber: 0.0065, swd: 0.0081, ept: 95.9970
    Epoch [5/50], Val Losses: mse: 0.1451, mae: 0.2069, huber: 0.0602, swd: 0.1137, ept: 95.5772
    Epoch [5/50], Test Losses: mse: 0.1592, mae: 0.2187, huber: 0.0675, swd: 0.1219, ept: 95.6477
      Epoch 5 composite train-obj: 0.006534
            No improvement (0.0602), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.0192, mae: 0.0781, huber: 0.0092, swd: 0.0118, ept: 95.9934
    Epoch [6/50], Val Losses: mse: 0.0178, mae: 0.0816, huber: 0.0087, swd: 0.0132, ept: 95.9990
    Epoch [6/50], Test Losses: mse: 0.0194, mae: 0.0867, huber: 0.0095, swd: 0.0147, ept: 95.9984
      Epoch 6 composite train-obj: 0.009242
            Val objective improved 0.0101 → 0.0087, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0151, mae: 0.0705, huber: 0.0073, swd: 0.0097, ept: 95.9962
    Epoch [7/50], Val Losses: mse: 0.0668, mae: 0.1516, huber: 0.0316, swd: 0.0553, ept: 95.9535
    Epoch [7/50], Test Losses: mse: 0.0768, mae: 0.1660, huber: 0.0368, swd: 0.0631, ept: 95.9815
      Epoch 7 composite train-obj: 0.007268
            No improvement (0.0316), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0112, mae: 0.0599, huber: 0.0055, swd: 0.0073, ept: 95.9988
    Epoch [8/50], Val Losses: mse: 0.0087, mae: 0.0611, huber: 0.0043, swd: 0.0054, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0089, mae: 0.0645, huber: 0.0045, swd: 0.0056, ept: 96.0000
      Epoch 8 composite train-obj: 0.005506
            Val objective improved 0.0087 → 0.0043, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0159, mae: 0.0687, huber: 0.0074, swd: 0.0107, ept: 95.9773
    Epoch [9/50], Val Losses: mse: 0.0182, mae: 0.0786, huber: 0.0088, swd: 0.0140, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0185, mae: 0.0815, huber: 0.0090, swd: 0.0141, ept: 95.9906
      Epoch 9 composite train-obj: 0.007369
            No improvement (0.0088), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.0071, mae: 0.0512, huber: 0.0035, swd: 0.0049, ept: 95.9991
    Epoch [10/50], Val Losses: mse: 0.0243, mae: 0.0925, huber: 0.0121, swd: 0.0174, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0264, mae: 0.0998, huber: 0.0132, swd: 0.0195, ept: 96.0000
      Epoch 10 composite train-obj: 0.003524
            No improvement (0.0121), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.0122, mae: 0.0589, huber: 0.0058, swd: 0.0085, ept: 95.9902
    Epoch [11/50], Val Losses: mse: 0.0732, mae: 0.1144, huber: 0.0295, swd: 0.0493, ept: 95.4592
    Epoch [11/50], Test Losses: mse: 0.0734, mae: 0.1223, huber: 0.0308, swd: 0.0496, ept: 95.6094
      Epoch 11 composite train-obj: 0.005766
            No improvement (0.0295), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.0104, mae: 0.0581, huber: 0.0051, swd: 0.0071, ept: 95.9951
    Epoch [12/50], Val Losses: mse: 0.0088, mae: 0.0614, huber: 0.0044, swd: 0.0059, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0093, mae: 0.0647, huber: 0.0047, swd: 0.0063, ept: 96.0000
      Epoch 12 composite train-obj: 0.005092
            No improvement (0.0044), counter 4/5
    Epoch [13/50], Train Losses: mse: 0.0099, mae: 0.0572, huber: 0.0049, swd: 0.0066, ept: 95.9980
    Epoch [13/50], Val Losses: mse: 0.0096, mae: 0.0666, huber: 0.0048, swd: 0.0066, ept: 96.0000
    Epoch [13/50], Test Losses: mse: 0.0101, mae: 0.0702, huber: 0.0051, swd: 0.0072, ept: 96.0000
      Epoch 13 composite train-obj: 0.004852
    Epoch [13/50], Test Losses: mse: 0.0089, mae: 0.0645, huber: 0.0045, swd: 0.0056, ept: 96.0000
    Best round's Test MSE: 0.0089, MAE: 0.0645, SWD: 0.0056
    Best round's Validation MSE: 0.0087, MAE: 0.0611, SWD: 0.0054
    Best round's Test verification MSE : 0.0089, MAE: 0.0645, SWD: 0.0056
    Time taken: 105.39 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.2464, mae: 0.5147, huber: 0.3168, swd: 1.9225, ept: 91.3080
    Epoch [1/50], Val Losses: mse: 0.9198, mae: 0.2912, huber: 0.1217, swd: 0.8236, ept: 94.4108
    Epoch [1/50], Test Losses: mse: 0.8243, mae: 0.3007, huber: 0.1255, swd: 0.7152, ept: 94.5278
      Epoch 1 composite train-obj: 0.316823
            Val objective improved inf → 0.1217, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3310, mae: 0.1706, huber: 0.0528, swd: 0.2889, ept: 95.2457
    Epoch [2/50], Val Losses: mse: 0.1876, mae: 0.2527, huber: 0.0761, swd: 0.1094, ept: 95.3465
    Epoch [2/50], Test Losses: mse: 0.1886, mae: 0.2627, huber: 0.0787, swd: 0.1112, ept: 95.4675
      Epoch 2 composite train-obj: 0.052807
            Val objective improved 0.1217 → 0.0761, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0426, mae: 0.1072, huber: 0.0173, swd: 0.0247, ept: 95.8819
    Epoch [3/50], Val Losses: mse: 0.3334, mae: 0.3930, huber: 0.1539, swd: 0.2824, ept: 95.7383
    Epoch [3/50], Test Losses: mse: 0.3734, mae: 0.4255, huber: 0.1736, swd: 0.3172, ept: 95.8402
      Epoch 3 composite train-obj: 0.017271
            No improvement (0.1539), counter 1/5
    Epoch [4/50], Train Losses: mse: 0.0237, mae: 0.0834, huber: 0.0111, swd: 0.0157, ept: 95.9750
    Epoch [4/50], Val Losses: mse: 0.0553, mae: 0.1219, huber: 0.0251, swd: 0.0183, ept: 96.0000
    Epoch [4/50], Test Losses: mse: 0.0589, mae: 0.1302, huber: 0.0271, swd: 0.0176, ept: 95.9988
      Epoch 4 composite train-obj: 0.011058
            Val objective improved 0.0761 → 0.0251, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0215, mae: 0.0817, huber: 0.0102, swd: 0.0139, ept: 95.9838
    Epoch [5/50], Val Losses: mse: 0.0193, mae: 0.0795, huber: 0.0094, swd: 0.0118, ept: 96.0000
    Epoch [5/50], Test Losses: mse: 0.0209, mae: 0.0839, huber: 0.0102, swd: 0.0128, ept: 96.0000
      Epoch 5 composite train-obj: 0.010175
            Val objective improved 0.0251 → 0.0094, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0118, mae: 0.0642, huber: 0.0058, swd: 0.0076, ept: 95.9984
    Epoch [6/50], Val Losses: mse: 0.0148, mae: 0.0727, huber: 0.0073, swd: 0.0122, ept: 96.0000
    Epoch [6/50], Test Losses: mse: 0.0155, mae: 0.0778, huber: 0.0077, swd: 0.0123, ept: 96.0000
      Epoch 6 composite train-obj: 0.005795
            Val objective improved 0.0094 → 0.0073, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0110, mae: 0.0608, huber: 0.0054, swd: 0.0074, ept: 95.9988
    Epoch [7/50], Val Losses: mse: 0.0211, mae: 0.0860, huber: 0.0105, swd: 0.0168, ept: 96.0000
    Epoch [7/50], Test Losses: mse: 0.0224, mae: 0.0920, huber: 0.0111, swd: 0.0179, ept: 96.0000
      Epoch 7 composite train-obj: 0.005387
            No improvement (0.0105), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0135, mae: 0.0698, huber: 0.0066, swd: 0.0093, ept: 95.9998
    Epoch [8/50], Val Losses: mse: 0.2408, mae: 0.2514, huber: 0.0777, swd: 0.2124, ept: 95.3035
    Epoch [8/50], Test Losses: mse: 0.2454, mae: 0.2633, huber: 0.0834, swd: 0.2153, ept: 95.3498
      Epoch 8 composite train-obj: 0.006641
            No improvement (0.0777), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.0549, mae: 0.0932, huber: 0.0162, swd: 0.0466, ept: 95.9136
    Epoch [9/50], Val Losses: mse: 0.0185, mae: 0.0832, huber: 0.0093, swd: 0.0158, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0205, mae: 0.0903, huber: 0.0102, swd: 0.0174, ept: 96.0000
      Epoch 9 composite train-obj: 0.016175
            No improvement (0.0093), counter 3/5
    Epoch [10/50], Train Losses: mse: 0.0081, mae: 0.0531, huber: 0.0040, swd: 0.0055, ept: 95.9999
    Epoch [10/50], Val Losses: mse: 0.0339, mae: 0.0918, huber: 0.0152, swd: 0.0300, ept: 95.9698
    Epoch [10/50], Test Losses: mse: 0.0366, mae: 0.0988, huber: 0.0168, swd: 0.0326, ept: 95.9704
      Epoch 10 composite train-obj: 0.004031
            No improvement (0.0152), counter 4/5
    Epoch [11/50], Train Losses: mse: 0.0122, mae: 0.0613, huber: 0.0059, swd: 0.0086, ept: 95.9948
    Epoch [11/50], Val Losses: mse: 0.7901, mae: 0.4117, huber: 0.2075, swd: 0.3935, ept: 93.8320
    Epoch [11/50], Test Losses: mse: 0.8708, mae: 0.4484, huber: 0.2344, swd: 0.4661, ept: 93.3306
      Epoch 11 composite train-obj: 0.005937
    Epoch [11/50], Test Losses: mse: 0.0155, mae: 0.0778, huber: 0.0077, swd: 0.0123, ept: 96.0000
    Best round's Test MSE: 0.0155, MAE: 0.0778, SWD: 0.0123
    Best round's Validation MSE: 0.0148, MAE: 0.0727, SWD: 0.0122
    Best round's Test verification MSE : 0.0155, MAE: 0.0778, SWD: 0.0123
    Time taken: 89.14 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 2.3053, mae: 0.5083, huber: 0.3150, swd: 1.7958, ept: 91.6034
    Epoch [1/50], Val Losses: mse: 1.0354, mae: 0.3436, huber: 0.1570, swd: 0.8673, ept: 94.8981
    Epoch [1/50], Test Losses: mse: 0.9469, mae: 0.3620, huber: 0.1662, swd: 0.7838, ept: 94.7910
      Epoch 1 composite train-obj: 0.315013
            Val objective improved inf → 0.1570, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3526, mae: 0.1838, huber: 0.0583, swd: 0.2720, ept: 95.3006
    Epoch [2/50], Val Losses: mse: 1.0069, mae: 0.4081, huber: 0.1804, swd: 0.8358, ept: 94.3760
    Epoch [2/50], Test Losses: mse: 0.9732, mae: 0.4272, huber: 0.1930, swd: 0.7934, ept: 94.0911
      Epoch 2 composite train-obj: 0.058323
            No improvement (0.1804), counter 1/5
    Epoch [3/50], Train Losses: mse: 0.4434, mae: 0.1818, huber: 0.0639, swd: 0.3666, ept: 95.2101
    Epoch [3/50], Val Losses: mse: 0.2140, mae: 0.1814, huber: 0.0589, swd: 0.1573, ept: 94.7211
    Epoch [3/50], Test Losses: mse: 0.2059, mae: 0.1910, huber: 0.0617, swd: 0.1482, ept: 94.9688
      Epoch 3 composite train-obj: 0.063946
            Val objective improved 0.1570 → 0.0589, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0482, mae: 0.1053, huber: 0.0178, swd: 0.0297, ept: 95.8533
    Epoch [4/50], Val Losses: mse: 0.0728, mae: 0.1892, huber: 0.0358, swd: 0.0477, ept: 95.9921
    Epoch [4/50], Test Losses: mse: 0.0786, mae: 0.2010, huber: 0.0388, swd: 0.0520, ept: 95.9917
      Epoch 4 composite train-obj: 0.017771
            Val objective improved 0.0589 → 0.0358, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0239, mae: 0.0835, huber: 0.0107, swd: 0.0155, ept: 95.9690
    Epoch [5/50], Val Losses: mse: 0.0716, mae: 0.1686, huber: 0.0350, swd: 0.0417, ept: 95.9914
    Epoch [5/50], Test Losses: mse: 0.0773, mae: 0.1789, huber: 0.0380, swd: 0.0437, ept: 95.9971
      Epoch 5 composite train-obj: 0.010735
            Val objective improved 0.0358 → 0.0350, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0191, mae: 0.0786, huber: 0.0092, swd: 0.0116, ept: 95.9906
    Epoch [6/50], Val Losses: mse: 0.1899, mae: 0.2213, huber: 0.0715, swd: 0.0692, ept: 94.4837
    Epoch [6/50], Test Losses: mse: 0.1862, mae: 0.2337, huber: 0.0744, swd: 0.0711, ept: 94.9090
      Epoch 6 composite train-obj: 0.009160
            No improvement (0.0715), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0304, mae: 0.0870, huber: 0.0130, swd: 0.0165, ept: 95.8894
    Epoch [7/50], Val Losses: mse: 0.1439, mae: 0.1976, huber: 0.0550, swd: 0.1181, ept: 95.3713
    Epoch [7/50], Test Losses: mse: 0.1694, mae: 0.2213, huber: 0.0651, swd: 0.1410, ept: 95.4278
      Epoch 7 composite train-obj: 0.012996
            No improvement (0.0550), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.0227, mae: 0.0734, huber: 0.0096, swd: 0.0155, ept: 95.9611
    Epoch [8/50], Val Losses: mse: 0.0378, mae: 0.1311, huber: 0.0189, swd: 0.0281, ept: 96.0000
    Epoch [8/50], Test Losses: mse: 0.0384, mae: 0.1342, huber: 0.0192, swd: 0.0288, ept: 96.0000
      Epoch 8 composite train-obj: 0.009554
            Val objective improved 0.0350 → 0.0189, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0127, mae: 0.0622, huber: 0.0061, swd: 0.0081, ept: 95.9945
    Epoch [9/50], Val Losses: mse: 0.0508, mae: 0.1212, huber: 0.0251, swd: 0.0347, ept: 96.0000
    Epoch [9/50], Test Losses: mse: 0.0518, mae: 0.1258, huber: 0.0257, swd: 0.0363, ept: 96.0000
      Epoch 9 composite train-obj: 0.006128
            No improvement (0.0251), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.0121, mae: 0.0600, huber: 0.0058, swd: 0.0079, ept: 95.9969
    Epoch [10/50], Val Losses: mse: 0.0064, mae: 0.0500, huber: 0.0032, swd: 0.0041, ept: 96.0000
    Epoch [10/50], Test Losses: mse: 0.0064, mae: 0.0521, huber: 0.0032, swd: 0.0040, ept: 96.0000
      Epoch 10 composite train-obj: 0.005823
            Val objective improved 0.0189 → 0.0032, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0136, mae: 0.0644, huber: 0.0066, swd: 0.0090, ept: 95.9955
    Epoch [11/50], Val Losses: mse: 0.0383, mae: 0.0995, huber: 0.0169, swd: 0.0259, ept: 95.9235
    Epoch [11/50], Test Losses: mse: 0.0351, mae: 0.1016, huber: 0.0161, swd: 0.0227, ept: 95.9530
      Epoch 11 composite train-obj: 0.006568
            No improvement (0.0169), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.0114, mae: 0.0599, huber: 0.0056, swd: 0.0072, ept: 95.9983
    Epoch [12/50], Val Losses: mse: 0.0189, mae: 0.0812, huber: 0.0094, swd: 0.0140, ept: 96.0000
    Epoch [12/50], Test Losses: mse: 0.0213, mae: 0.0891, huber: 0.0106, swd: 0.0157, ept: 96.0000
      Epoch 12 composite train-obj: 0.005556
            No improvement (0.0094), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.0092, mae: 0.0540, huber: 0.0045, swd: 0.0061, ept: 95.9987
    Epoch [13/50], Val Losses: mse: 0.0117, mae: 0.0651, huber: 0.0058, swd: 0.0077, ept: 96.0000
    Epoch [13/50], Test Losses: mse: 0.0135, mae: 0.0697, huber: 0.0067, swd: 0.0085, ept: 96.0000
      Epoch 13 composite train-obj: 0.004506
            No improvement (0.0058), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.0136, mae: 0.0617, huber: 0.0065, swd: 0.0083, ept: 95.9943
    Epoch [14/50], Val Losses: mse: 0.0098, mae: 0.0573, huber: 0.0049, swd: 0.0067, ept: 96.0000
    Epoch [14/50], Test Losses: mse: 0.0099, mae: 0.0610, huber: 0.0049, swd: 0.0068, ept: 96.0000
      Epoch 14 composite train-obj: 0.006536
            No improvement (0.0049), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.0118, mae: 0.0605, huber: 0.0058, swd: 0.0081, ept: 95.9991
    Epoch [15/50], Val Losses: mse: 0.0104, mae: 0.0639, huber: 0.0052, swd: 0.0063, ept: 96.0000
    Epoch [15/50], Test Losses: mse: 0.0104, mae: 0.0663, huber: 0.0052, swd: 0.0064, ept: 96.0000
      Epoch 15 composite train-obj: 0.005817
    Epoch [15/50], Test Losses: mse: 0.0064, mae: 0.0521, huber: 0.0032, swd: 0.0040, ept: 96.0000
    Best round's Test MSE: 0.0064, MAE: 0.0521, SWD: 0.0040
    Best round's Validation MSE: 0.0064, MAE: 0.0500, SWD: 0.0041
    Best round's Test verification MSE : 0.0064, MAE: 0.0521, SWD: 0.0040
    Time taken: 118.66 seconds
    
    ==================================================
    Experiment Summary (ACL_rossler_seq336_pred96_20250513_1243)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0103 ± 0.0038
      mae: 0.0648 ± 0.0105
      huber: 0.0051 ± 0.0019
      swd: 0.0073 ± 0.0036
      ept: 96.0000 ± 0.0000
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0100 ± 0.0035
      mae: 0.0613 ± 0.0093
      huber: 0.0049 ± 0.0017
      swd: 0.0072 ± 0.0036
      ept: 96.0000 ± 0.0000
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 313.27 seconds
    
    Experiment complete: ACL_rossler_seq336_pred96_20250513_1243
    Model: ACL
    Dataset: rossler
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 281
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 3.2397, mae: 0.6173, huber: 0.4033, swd: 2.9480, ept: 180.3137
    Epoch [1/50], Val Losses: mse: 2.2008, mae: 0.3333, huber: 0.1909, swd: 2.1300, ept: 187.9526
    Epoch [1/50], Test Losses: mse: 1.9491, mae: 0.3401, huber: 0.1941, swd: 1.8732, ept: 186.5203
      Epoch 1 composite train-obj: 0.403315
            Val objective improved inf → 0.1909, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 1.2037, mae: 0.2792, huber: 0.1318, swd: 1.0961, ept: 190.2501
    Epoch [2/50], Val Losses: mse: 0.1645, mae: 0.1600, huber: 0.0455, swd: 0.0559, ept: 191.1781
    Epoch [2/50], Test Losses: mse: 0.1485, mae: 0.1654, huber: 0.0450, swd: 0.0470, ept: 192.2450
      Epoch 2 composite train-obj: 0.131763
            Val objective improved 0.1909 → 0.0455, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0698, mae: 0.1311, huber: 0.0267, swd: 0.0294, ept: 194.1636
    Epoch [3/50], Val Losses: mse: 0.0587, mae: 0.1155, huber: 0.0226, swd: 0.0273, ept: 194.9153
    Epoch [3/50], Test Losses: mse: 0.0582, mae: 0.1235, huber: 0.0237, swd: 0.0282, ept: 195.1777
      Epoch 3 composite train-obj: 0.026663
            Val objective improved 0.0455 → 0.0226, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0319, mae: 0.1027, huber: 0.0147, swd: 0.0160, ept: 195.6917
    Epoch [4/50], Val Losses: mse: 0.0549, mae: 0.1421, huber: 0.0248, swd: 0.0346, ept: 195.1767
    Epoch [4/50], Test Losses: mse: 0.0548, mae: 0.1476, huber: 0.0251, swd: 0.0347, ept: 195.3554
      Epoch 4 composite train-obj: 0.014712
            No improvement (0.0248), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0180, mae: 0.0810, huber: 0.0088, swd: 0.0095, ept: 195.9545
    Epoch [5/50], Val Losses: mse: 0.0308, mae: 0.0969, huber: 0.0144, swd: 0.0184, ept: 195.4557
    Epoch [5/50], Test Losses: mse: 0.0307, mae: 0.1021, huber: 0.0146, swd: 0.0176, ept: 195.5543
      Epoch 5 composite train-obj: 0.008751
            Val objective improved 0.0226 → 0.0144, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0204, mae: 0.0872, huber: 0.0099, swd: 0.0111, ept: 195.9604
    Epoch [6/50], Val Losses: mse: 0.0153, mae: 0.0700, huber: 0.0075, swd: 0.0074, ept: 196.0000
    Epoch [6/50], Test Losses: mse: 0.0158, mae: 0.0745, huber: 0.0078, swd: 0.0078, ept: 196.0000
      Epoch 6 composite train-obj: 0.009944
            Val objective improved 0.0144 → 0.0075, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0165, mae: 0.0783, huber: 0.0081, swd: 0.0092, ept: 195.9902
    Epoch [7/50], Val Losses: mse: 0.0160, mae: 0.0792, huber: 0.0079, swd: 0.0092, ept: 196.0000
    Epoch [7/50], Test Losses: mse: 0.0168, mae: 0.0831, huber: 0.0083, swd: 0.0098, ept: 196.0000
      Epoch 7 composite train-obj: 0.008129
            No improvement (0.0079), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0207, mae: 0.0851, huber: 0.0101, swd: 0.0110, ept: 195.9656
    Epoch [8/50], Val Losses: mse: 0.0186, mae: 0.0704, huber: 0.0090, swd: 0.0127, ept: 196.0000
    Epoch [8/50], Test Losses: mse: 0.0201, mae: 0.0749, huber: 0.0098, swd: 0.0148, ept: 196.0000
      Epoch 8 composite train-obj: 0.010087
            No improvement (0.0090), counter 2/5
    Epoch [9/50], Train Losses: mse: 0.0167, mae: 0.0737, huber: 0.0082, swd: 0.0091, ept: 195.9969
    Epoch [9/50], Val Losses: mse: 0.0132, mae: 0.0610, huber: 0.0065, swd: 0.0069, ept: 196.0000
    Epoch [9/50], Test Losses: mse: 0.0128, mae: 0.0637, huber: 0.0064, swd: 0.0074, ept: 196.0000
      Epoch 9 composite train-obj: 0.008208
            Val objective improved 0.0075 → 0.0065, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.0123, mae: 0.0657, huber: 0.0060, swd: 0.0072, ept: 195.9932
    Epoch [10/50], Val Losses: mse: 0.0135, mae: 0.0647, huber: 0.0066, swd: 0.0089, ept: 196.0000
    Epoch [10/50], Test Losses: mse: 0.0139, mae: 0.0677, huber: 0.0068, swd: 0.0096, ept: 196.0000
      Epoch 10 composite train-obj: 0.006006
            No improvement (0.0066), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.0171, mae: 0.0768, huber: 0.0083, swd: 0.0097, ept: 195.9705
    Epoch [11/50], Val Losses: mse: 0.0307, mae: 0.0941, huber: 0.0140, swd: 0.0205, ept: 195.7409
    Epoch [11/50], Test Losses: mse: 0.0349, mae: 0.1029, huber: 0.0163, swd: 0.0243, ept: 195.7907
      Epoch 11 composite train-obj: 0.008350
            No improvement (0.0140), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.0099, mae: 0.0614, huber: 0.0049, swd: 0.0059, ept: 196.0000
    Epoch [12/50], Val Losses: mse: 0.0118, mae: 0.0591, huber: 0.0059, swd: 0.0054, ept: 196.0000
    Epoch [12/50], Test Losses: mse: 0.0120, mae: 0.0617, huber: 0.0059, swd: 0.0063, ept: 196.0000
      Epoch 12 composite train-obj: 0.004930
            Val objective improved 0.0065 → 0.0059, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.0135, mae: 0.0683, huber: 0.0066, swd: 0.0076, ept: 195.9888
    Epoch [13/50], Val Losses: mse: 0.0096, mae: 0.0574, huber: 0.0048, swd: 0.0044, ept: 196.0000
    Epoch [13/50], Test Losses: mse: 0.0096, mae: 0.0596, huber: 0.0048, swd: 0.0044, ept: 196.0000
      Epoch 13 composite train-obj: 0.006649
            Val objective improved 0.0059 → 0.0048, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0105, mae: 0.0612, huber: 0.0052, swd: 0.0065, ept: 195.9900
    Epoch [14/50], Val Losses: mse: 0.0141, mae: 0.0712, huber: 0.0070, swd: 0.0079, ept: 196.0000
    Epoch [14/50], Test Losses: mse: 0.0148, mae: 0.0751, huber: 0.0073, swd: 0.0086, ept: 196.0000
      Epoch 14 composite train-obj: 0.005172
            No improvement (0.0070), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.0149, mae: 0.0708, huber: 0.0074, swd: 0.0088, ept: 195.9929
    Epoch [15/50], Val Losses: mse: 0.0164, mae: 0.0767, huber: 0.0082, swd: 0.0103, ept: 196.0000
    Epoch [15/50], Test Losses: mse: 0.0169, mae: 0.0812, huber: 0.0084, swd: 0.0112, ept: 196.0000
      Epoch 15 composite train-obj: 0.007359
            No improvement (0.0082), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.0087, mae: 0.0558, huber: 0.0043, swd: 0.0052, ept: 196.0000
    Epoch [16/50], Val Losses: mse: 0.0205, mae: 0.0932, huber: 0.0102, swd: 0.0136, ept: 196.0000
    Epoch [16/50], Test Losses: mse: 0.0228, mae: 0.1011, huber: 0.0113, swd: 0.0153, ept: 196.0000
      Epoch 16 composite train-obj: 0.004339
            No improvement (0.0102), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.0092, mae: 0.0559, huber: 0.0046, swd: 0.0056, ept: 195.9984
    Epoch [17/50], Val Losses: mse: 0.0186, mae: 0.0824, huber: 0.0093, swd: 0.0101, ept: 196.0000
    Epoch [17/50], Test Losses: mse: 0.0204, mae: 0.0900, huber: 0.0102, swd: 0.0117, ept: 196.0000
      Epoch 17 composite train-obj: 0.004561
            No improvement (0.0093), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.0101, mae: 0.0600, huber: 0.0050, swd: 0.0062, ept: 195.9946
    Epoch [18/50], Val Losses: mse: 0.0148, mae: 0.0743, huber: 0.0074, swd: 0.0065, ept: 196.0000
    Epoch [18/50], Test Losses: mse: 0.0157, mae: 0.0790, huber: 0.0079, swd: 0.0070, ept: 196.0000
      Epoch 18 composite train-obj: 0.005008
    Epoch [18/50], Test Losses: mse: 0.0096, mae: 0.0596, huber: 0.0048, swd: 0.0044, ept: 196.0000
    Best round's Test MSE: 0.0096, MAE: 0.0596, SWD: 0.0044
    Best round's Validation MSE: 0.0096, MAE: 0.0574, SWD: 0.0044
    Best round's Test verification MSE : 0.0096, MAE: 0.0596, SWD: 0.0044
    Time taken: 143.60 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 3.2113, mae: 0.6160, huber: 0.4015, swd: 2.9009, ept: 180.1924
    Epoch [1/50], Val Losses: mse: 2.2289, mae: 0.3429, huber: 0.1942, swd: 2.1921, ept: 188.4863
    Epoch [1/50], Test Losses: mse: 1.9785, mae: 0.3467, huber: 0.1970, swd: 1.9342, ept: 187.9331
      Epoch 1 composite train-obj: 0.401468
            Val objective improved inf → 0.1942, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 1.3175, mae: 0.2835, huber: 0.1381, swd: 1.2299, ept: 190.5975
    Epoch [2/50], Val Losses: mse: 0.2979, mae: 0.2041, huber: 0.0699, swd: 0.1162, ept: 190.2216
    Epoch [2/50], Test Losses: mse: 0.2757, mae: 0.2063, huber: 0.0702, swd: 0.1102, ept: 190.5002
      Epoch 2 composite train-obj: 0.138117
            Val objective improved 0.1942 → 0.0699, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.1088, mae: 0.1453, huber: 0.0335, swd: 0.0500, ept: 193.4711
    Epoch [3/50], Val Losses: mse: 0.0344, mae: 0.0936, huber: 0.0147, swd: 0.0138, ept: 194.8417
    Epoch [3/50], Test Losses: mse: 0.0345, mae: 0.0981, huber: 0.0153, swd: 0.0139, ept: 195.0876
      Epoch 3 composite train-obj: 0.033481
            Val objective improved 0.0699 → 0.0147, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0360, mae: 0.1088, huber: 0.0164, swd: 0.0171, ept: 195.3820
    Epoch [4/50], Val Losses: mse: 0.0279, mae: 0.0926, huber: 0.0131, swd: 0.0116, ept: 195.9262
    Epoch [4/50], Test Losses: mse: 0.0273, mae: 0.0960, huber: 0.0129, swd: 0.0111, ept: 195.9453
      Epoch 4 composite train-obj: 0.016378
            Val objective improved 0.0147 → 0.0131, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0255, mae: 0.0943, huber: 0.0121, swd: 0.0132, ept: 195.8911
    Epoch [5/50], Val Losses: mse: 0.0329, mae: 0.1114, huber: 0.0159, swd: 0.0162, ept: 195.8792
    Epoch [5/50], Test Losses: mse: 0.0353, mae: 0.1191, huber: 0.0172, swd: 0.0169, ept: 195.8914
      Epoch 5 composite train-obj: 0.012125
            No improvement (0.0159), counter 1/5
    Epoch [6/50], Train Losses: mse: 0.0235, mae: 0.0900, huber: 0.0112, swd: 0.0126, ept: 195.9103
    Epoch [6/50], Val Losses: mse: 0.0108, mae: 0.0608, huber: 0.0054, swd: 0.0056, ept: 196.0000
    Epoch [6/50], Test Losses: mse: 0.0116, mae: 0.0647, huber: 0.0057, swd: 0.0060, ept: 196.0000
      Epoch 6 composite train-obj: 0.011194
            Val objective improved 0.0131 → 0.0054, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0172, mae: 0.0787, huber: 0.0084, swd: 0.0093, ept: 195.9862
    Epoch [7/50], Val Losses: mse: 0.0189, mae: 0.0637, huber: 0.0086, swd: 0.0101, ept: 196.0000
    Epoch [7/50], Test Losses: mse: 0.0199, mae: 0.0690, huber: 0.0092, swd: 0.0111, ept: 195.9845
      Epoch 7 composite train-obj: 0.008408
            No improvement (0.0086), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0177, mae: 0.0788, huber: 0.0086, swd: 0.0097, ept: 195.9652
    Epoch [8/50], Val Losses: mse: 0.0108, mae: 0.0625, huber: 0.0054, swd: 0.0053, ept: 196.0000
    Epoch [8/50], Test Losses: mse: 0.0114, mae: 0.0659, huber: 0.0057, swd: 0.0056, ept: 196.0000
      Epoch 8 composite train-obj: 0.008566
            Val objective improved 0.0054 → 0.0054, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0167, mae: 0.0766, huber: 0.0082, swd: 0.0096, ept: 195.9692
    Epoch [9/50], Val Losses: mse: 0.0069, mae: 0.0502, huber: 0.0034, swd: 0.0044, ept: 196.0000
    Epoch [9/50], Test Losses: mse: 0.0075, mae: 0.0531, huber: 0.0037, swd: 0.0048, ept: 196.0000
      Epoch 9 composite train-obj: 0.008181
            Val objective improved 0.0054 → 0.0034, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.0111, mae: 0.0627, huber: 0.0055, swd: 0.0064, ept: 195.9969
    Epoch [10/50], Val Losses: mse: 0.0222, mae: 0.0719, huber: 0.0102, swd: 0.0154, ept: 195.8649
    Epoch [10/50], Test Losses: mse: 0.0216, mae: 0.0754, huber: 0.0102, swd: 0.0153, ept: 195.8665
      Epoch 10 composite train-obj: 0.005490
            No improvement (0.0102), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.0211, mae: 0.0826, huber: 0.0102, swd: 0.0114, ept: 195.9688
    Epoch [11/50], Val Losses: mse: 0.0243, mae: 0.0899, huber: 0.0120, swd: 0.0128, ept: 196.0000
    Epoch [11/50], Test Losses: mse: 0.0267, mae: 0.0974, huber: 0.0132, swd: 0.0141, ept: 196.0000
      Epoch 11 composite train-obj: 0.010207
            No improvement (0.0120), counter 2/5
    Epoch [12/50], Train Losses: mse: 0.0119, mae: 0.0654, huber: 0.0059, swd: 0.0071, ept: 195.9977
    Epoch [12/50], Val Losses: mse: 0.0181, mae: 0.0728, huber: 0.0085, swd: 0.0076, ept: 195.9742
    Epoch [12/50], Test Losses: mse: 0.0182, mae: 0.0771, huber: 0.0086, swd: 0.0082, ept: 195.9696
      Epoch 12 composite train-obj: 0.005878
            No improvement (0.0085), counter 3/5
    Epoch [13/50], Train Losses: mse: 0.0114, mae: 0.0609, huber: 0.0056, swd: 0.0066, ept: 195.9931
    Epoch [13/50], Val Losses: mse: 0.0189, mae: 0.0844, huber: 0.0092, swd: 0.0129, ept: 195.9753
    Epoch [13/50], Test Losses: mse: 0.0203, mae: 0.0904, huber: 0.0099, swd: 0.0141, ept: 195.9784
      Epoch 13 composite train-obj: 0.005569
            No improvement (0.0092), counter 4/5
    Epoch [14/50], Train Losses: mse: 0.0161, mae: 0.0702, huber: 0.0078, swd: 0.0091, ept: 195.9728
    Epoch [14/50], Val Losses: mse: 0.0068, mae: 0.0476, huber: 0.0034, swd: 0.0034, ept: 196.0000
    Epoch [14/50], Test Losses: mse: 0.0066, mae: 0.0487, huber: 0.0033, swd: 0.0032, ept: 196.0000
      Epoch 14 composite train-obj: 0.007780
            Val objective improved 0.0034 → 0.0034, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.0089, mae: 0.0561, huber: 0.0044, swd: 0.0056, ept: 195.9971
    Epoch [15/50], Val Losses: mse: 0.0122, mae: 0.0632, huber: 0.0061, swd: 0.0071, ept: 196.0000
    Epoch [15/50], Test Losses: mse: 0.0124, mae: 0.0660, huber: 0.0062, swd: 0.0075, ept: 196.0000
      Epoch 15 composite train-obj: 0.004413
            No improvement (0.0061), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.0157, mae: 0.0696, huber: 0.0076, swd: 0.0084, ept: 195.9535
    Epoch [16/50], Val Losses: mse: 0.0145, mae: 0.0565, huber: 0.0069, swd: 0.0102, ept: 196.0000
    Epoch [16/50], Test Losses: mse: 0.0153, mae: 0.0606, huber: 0.0074, swd: 0.0113, ept: 196.0000
      Epoch 16 composite train-obj: 0.007638
            No improvement (0.0069), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.0093, mae: 0.0562, huber: 0.0046, swd: 0.0056, ept: 195.9932
    Epoch [17/50], Val Losses: mse: 0.0070, mae: 0.0558, huber: 0.0035, swd: 0.0037, ept: 196.0000
    Epoch [17/50], Test Losses: mse: 0.0077, mae: 0.0595, huber: 0.0038, swd: 0.0041, ept: 196.0000
      Epoch 17 composite train-obj: 0.004578
            No improvement (0.0035), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.0107, mae: 0.0592, huber: 0.0053, swd: 0.0064, ept: 195.9927
    Epoch [18/50], Val Losses: mse: 0.0129, mae: 0.0746, huber: 0.0065, swd: 0.0078, ept: 196.0000
    Epoch [18/50], Test Losses: mse: 0.0131, mae: 0.0770, huber: 0.0066, swd: 0.0081, ept: 196.0000
      Epoch 18 composite train-obj: 0.005284
            No improvement (0.0065), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.0110, mae: 0.0633, huber: 0.0055, swd: 0.0067, ept: 195.9933
    Epoch [19/50], Val Losses: mse: 0.0145, mae: 0.0694, huber: 0.0072, swd: 0.0107, ept: 196.0000
    Epoch [19/50], Test Losses: mse: 0.0154, mae: 0.0736, huber: 0.0076, swd: 0.0116, ept: 196.0000
      Epoch 19 composite train-obj: 0.005456
    Epoch [19/50], Test Losses: mse: 0.0066, mae: 0.0487, huber: 0.0033, swd: 0.0032, ept: 196.0000
    Best round's Test MSE: 0.0066, MAE: 0.0487, SWD: 0.0032
    Best round's Validation MSE: 0.0068, MAE: 0.0476, SWD: 0.0034
    Best round's Test verification MSE : 0.0066, MAE: 0.0487, SWD: 0.0032
    Time taken: 151.85 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 3.2029, mae: 0.6054, huber: 0.3995, swd: 2.6741, ept: 180.8992
    Epoch [1/50], Val Losses: mse: 2.1414, mae: 0.3222, huber: 0.1844, swd: 1.9611, ept: 188.7695
    Epoch [1/50], Test Losses: mse: 1.8954, mae: 0.3263, huber: 0.1866, swd: 1.7220, ept: 188.1310
      Epoch 1 composite train-obj: 0.399461
            Val objective improved inf → 0.1844, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 1.2832, mae: 0.2743, huber: 0.1333, swd: 1.1237, ept: 191.2831
    Epoch [2/50], Val Losses: mse: 0.3498, mae: 0.1885, huber: 0.0663, swd: 0.1857, ept: 192.3289
    Epoch [2/50], Test Losses: mse: 0.3128, mae: 0.1936, huber: 0.0669, swd: 0.1659, ept: 192.7044
      Epoch 2 composite train-obj: 0.133278
            Val objective improved 0.1844 → 0.0663, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.1195, mae: 0.1483, huber: 0.0350, swd: 0.0588, ept: 194.1914
    Epoch [3/50], Val Losses: mse: 0.0558, mae: 0.1378, huber: 0.0245, swd: 0.0244, ept: 195.1661
    Epoch [3/50], Test Losses: mse: 0.0546, mae: 0.1432, huber: 0.0246, swd: 0.0241, ept: 195.2720
      Epoch 3 composite train-obj: 0.035042
            Val objective improved 0.0663 → 0.0245, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0345, mae: 0.1052, huber: 0.0156, swd: 0.0152, ept: 195.6021
    Epoch [4/50], Val Losses: mse: 0.0392, mae: 0.0953, huber: 0.0165, swd: 0.0178, ept: 194.9500
    Epoch [4/50], Test Losses: mse: 0.0381, mae: 0.0990, huber: 0.0165, swd: 0.0179, ept: 195.1382
      Epoch 4 composite train-obj: 0.015594
            Val objective improved 0.0245 → 0.0165, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0341, mae: 0.1032, huber: 0.0156, swd: 0.0156, ept: 195.8086
    Epoch [5/50], Val Losses: mse: 0.0331, mae: 0.1080, huber: 0.0156, swd: 0.0153, ept: 195.8449
    Epoch [5/50], Test Losses: mse: 0.0319, mae: 0.1100, huber: 0.0152, swd: 0.0144, ept: 195.8650
      Epoch 5 composite train-obj: 0.015613
            Val objective improved 0.0165 → 0.0156, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0175, mae: 0.0774, huber: 0.0084, swd: 0.0087, ept: 195.9689
    Epoch [6/50], Val Losses: mse: 0.0267, mae: 0.0891, huber: 0.0127, swd: 0.0123, ept: 195.8809
    Epoch [6/50], Test Losses: mse: 0.0266, mae: 0.0935, huber: 0.0127, swd: 0.0125, ept: 195.8713
      Epoch 6 composite train-obj: 0.008427
            Val objective improved 0.0156 → 0.0127, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0255, mae: 0.0933, huber: 0.0122, swd: 0.0123, ept: 195.9412
    Epoch [7/50], Val Losses: mse: 0.0297, mae: 0.0942, huber: 0.0137, swd: 0.0108, ept: 195.9643
    Epoch [7/50], Test Losses: mse: 0.0295, mae: 0.0982, huber: 0.0139, swd: 0.0111, ept: 195.9817
      Epoch 7 composite train-obj: 0.012236
            No improvement (0.0137), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0118, mae: 0.0625, huber: 0.0057, swd: 0.0062, ept: 195.9924
    Epoch [8/50], Val Losses: mse: 0.0141, mae: 0.0652, huber: 0.0069, swd: 0.0074, ept: 195.9969
    Epoch [8/50], Test Losses: mse: 0.0152, mae: 0.0696, huber: 0.0075, swd: 0.0086, ept: 195.9975
      Epoch 8 composite train-obj: 0.005728
            Val objective improved 0.0127 → 0.0069, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0197, mae: 0.0821, huber: 0.0095, swd: 0.0101, ept: 195.9524
    Epoch [9/50], Val Losses: mse: 0.0202, mae: 0.0827, huber: 0.0099, swd: 0.0088, ept: 195.9054
    Epoch [9/50], Test Losses: mse: 0.0197, mae: 0.0849, huber: 0.0097, swd: 0.0089, ept: 195.9231
      Epoch 9 composite train-obj: 0.009510
            No improvement (0.0099), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.0184, mae: 0.0806, huber: 0.0090, swd: 0.0096, ept: 195.9823
    Epoch [10/50], Val Losses: mse: 0.0114, mae: 0.0554, huber: 0.0055, swd: 0.0072, ept: 196.0000
    Epoch [10/50], Test Losses: mse: 0.0117, mae: 0.0592, huber: 0.0057, swd: 0.0075, ept: 196.0000
      Epoch 10 composite train-obj: 0.009004
            Val objective improved 0.0069 → 0.0055, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0166, mae: 0.0739, huber: 0.0080, swd: 0.0090, ept: 195.9660
    Epoch [11/50], Val Losses: mse: 0.0084, mae: 0.0550, huber: 0.0042, swd: 0.0044, ept: 196.0000
    Epoch [11/50], Test Losses: mse: 0.0088, mae: 0.0589, huber: 0.0044, swd: 0.0046, ept: 196.0000
      Epoch 11 composite train-obj: 0.007981
            Val objective improved 0.0055 → 0.0042, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0160, mae: 0.0709, huber: 0.0076, swd: 0.0084, ept: 195.9492
    Epoch [12/50], Val Losses: mse: 0.0292, mae: 0.1110, huber: 0.0144, swd: 0.0170, ept: 195.9930
    Epoch [12/50], Test Losses: mse: 0.0320, mae: 0.1196, huber: 0.0158, swd: 0.0196, ept: 195.9924
      Epoch 12 composite train-obj: 0.007606
            No improvement (0.0144), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0118, mae: 0.0627, huber: 0.0058, swd: 0.0062, ept: 195.9935
    Epoch [13/50], Val Losses: mse: 0.0043, mae: 0.0399, huber: 0.0021, swd: 0.0022, ept: 196.0000
    Epoch [13/50], Test Losses: mse: 0.0044, mae: 0.0416, huber: 0.0022, swd: 0.0023, ept: 196.0000
      Epoch 13 composite train-obj: 0.005831
            Val objective improved 0.0042 → 0.0021, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0108, mae: 0.0616, huber: 0.0054, swd: 0.0059, ept: 195.9962
    Epoch [14/50], Val Losses: mse: 0.0105, mae: 0.0550, huber: 0.0052, swd: 0.0048, ept: 196.0000
    Epoch [14/50], Test Losses: mse: 0.0101, mae: 0.0574, huber: 0.0050, swd: 0.0047, ept: 196.0000
      Epoch 14 composite train-obj: 0.005374
            No improvement (0.0052), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.0169, mae: 0.0730, huber: 0.0082, swd: 0.0088, ept: 195.9671
    Epoch [15/50], Val Losses: mse: 0.0307, mae: 0.1105, huber: 0.0152, swd: 0.0113, ept: 196.0000
    Epoch [15/50], Test Losses: mse: 0.0299, mae: 0.1134, huber: 0.0148, swd: 0.0112, ept: 196.0000
      Epoch 15 composite train-obj: 0.008171
            No improvement (0.0152), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.0155, mae: 0.0703, huber: 0.0075, swd: 0.0079, ept: 195.9698
    Epoch [16/50], Val Losses: mse: 0.0275, mae: 0.0865, huber: 0.0127, swd: 0.0185, ept: 195.7239
    Epoch [16/50], Test Losses: mse: 0.0296, mae: 0.0948, huber: 0.0140, swd: 0.0200, ept: 195.7320
      Epoch 16 composite train-obj: 0.007504
            No improvement (0.0127), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.0073, mae: 0.0496, huber: 0.0036, swd: 0.0042, ept: 195.9878
    Epoch [17/50], Val Losses: mse: 0.0385, mae: 0.1304, huber: 0.0191, swd: 0.0203, ept: 196.0000
    Epoch [17/50], Test Losses: mse: 0.0415, mae: 0.1370, huber: 0.0207, swd: 0.0213, ept: 196.0000
      Epoch 17 composite train-obj: 0.003588
            No improvement (0.0191), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.0167, mae: 0.0770, huber: 0.0082, swd: 0.0089, ept: 195.9845
    Epoch [18/50], Val Losses: mse: 0.0098, mae: 0.0593, huber: 0.0049, swd: 0.0061, ept: 196.0000
    Epoch [18/50], Test Losses: mse: 0.0098, mae: 0.0611, huber: 0.0049, swd: 0.0063, ept: 196.0000
      Epoch 18 composite train-obj: 0.008175
    Epoch [18/50], Test Losses: mse: 0.0044, mae: 0.0416, huber: 0.0022, swd: 0.0023, ept: 196.0000
    Best round's Test MSE: 0.0044, MAE: 0.0416, SWD: 0.0023
    Best round's Validation MSE: 0.0043, MAE: 0.0399, SWD: 0.0022
    Best round's Test verification MSE : 0.0044, MAE: 0.0416, SWD: 0.0023
    Time taken: 144.36 seconds
    
    ==================================================
    Experiment Summary (ACL_rossler_seq336_pred196_20250513_1248)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0069 ± 0.0021
      mae: 0.0500 ± 0.0074
      huber: 0.0034 ± 0.0011
      swd: 0.0033 ± 0.0009
      ept: 196.0000 ± 0.0000
      count: 37.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0069 ± 0.0022
      mae: 0.0483 ± 0.0072
      huber: 0.0034 ± 0.0011
      swd: 0.0033 ± 0.0009
      ept: 196.0000 ± 0.0000
      count: 37.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 439.88 seconds
    
    Experiment complete: ACL_rossler_seq336_pred196_20250513_1248
    Model: ACL
    Dataset: rossler
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336

##### huber


```python
from monotonic import DynamicTanh
import torch.nn as nn

importlib.reload(monotonic)
importlib.reload(train_config) 
cfg = train_config.FlatACLConfig(  # original householder 
    seq_len=336,
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

##### do not use ablations: rotations (8,4)


```python
importlib.reload(monotonic)
importlib.reload(train_config)

cfg = train_config.FlatACLConfig( 
    seq_len=336,
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
    global_std.shape: torch.Size([3])
    Global Std for rossler: tensor([5.0964, 4.7939, 2.8266], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 280
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 3.2017, mae: 0.6333, huber: 0.4076, swd: 2.6055, ept: 300.8536
    Epoch [1/50], Val Losses: mse: 1.1065, mae: 0.3447, huber: 0.1600, swd: 0.7281, ept: 318.4954
    Epoch [1/50], Test Losses: mse: 0.9466, mae: 0.3453, huber: 0.1536, swd: 0.6067, ept: 316.4481
      Epoch 1 composite train-obj: 0.407577
            Val objective improved inf → 0.1600, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2324, mae: 0.2090, huber: 0.0614, swd: 0.1076, ept: 326.3840
    Epoch [2/50], Val Losses: mse: 0.0975, mae: 0.1736, huber: 0.0393, swd: 0.0333, ept: 328.7839
    Epoch [2/50], Test Losses: mse: 0.0955, mae: 0.1811, huber: 0.0404, swd: 0.0366, ept: 330.6555
      Epoch 2 composite train-obj: 0.061353
            Val objective improved 0.1600 → 0.0393, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0422, mae: 0.1241, huber: 0.0194, swd: 0.0164, ept: 334.8001
    Epoch [3/50], Val Losses: mse: 0.0421, mae: 0.1100, huber: 0.0181, swd: 0.0190, ept: 332.7940
    Epoch [3/50], Test Losses: mse: 0.0426, mae: 0.1165, huber: 0.0191, swd: 0.0204, ept: 333.7332
      Epoch 3 composite train-obj: 0.019371
            Val objective improved 0.0393 → 0.0181, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0261, mae: 0.0971, huber: 0.0123, swd: 0.0106, ept: 335.6404
    Epoch [4/50], Val Losses: mse: 0.0635, mae: 0.1418, huber: 0.0287, swd: 0.0233, ept: 334.3107
    Epoch [4/50], Test Losses: mse: 0.0652, mae: 0.1510, huber: 0.0304, swd: 0.0264, ept: 334.6904
      Epoch 4 composite train-obj: 0.012298
            No improvement (0.0287), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0320, mae: 0.1074, huber: 0.0154, swd: 0.0132, ept: 335.7580
    Epoch [5/50], Val Losses: mse: 0.0167, mae: 0.0784, huber: 0.0083, swd: 0.0068, ept: 336.0000
    Epoch [5/50], Test Losses: mse: 0.0173, mae: 0.0823, huber: 0.0086, swd: 0.0078, ept: 336.0000
      Epoch 5 composite train-obj: 0.015356
            Val objective improved 0.0181 → 0.0083, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0228, mae: 0.0914, huber: 0.0111, swd: 0.0099, ept: 335.8788
    Epoch [6/50], Val Losses: mse: 0.0256, mae: 0.0917, huber: 0.0122, swd: 0.0134, ept: 336.0000
    Epoch [6/50], Test Losses: mse: 0.0255, mae: 0.0938, huber: 0.0122, swd: 0.0137, ept: 336.0000
      Epoch 6 composite train-obj: 0.011061
            No improvement (0.0122), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0204, mae: 0.0857, huber: 0.0100, swd: 0.0086, ept: 335.9437
    Epoch [7/50], Val Losses: mse: 0.0152, mae: 0.0718, huber: 0.0074, swd: 0.0088, ept: 336.0000
    Epoch [7/50], Test Losses: mse: 0.0196, mae: 0.0761, huber: 0.0092, swd: 0.0130, ept: 335.7488
      Epoch 7 composite train-obj: 0.009987
            Val objective improved 0.0083 → 0.0074, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0218, mae: 0.0915, huber: 0.0107, swd: 0.0096, ept: 335.9364
    Epoch [8/50], Val Losses: mse: 0.0327, mae: 0.1018, huber: 0.0162, swd: 0.0126, ept: 336.0000
    Epoch [8/50], Test Losses: mse: 0.0351, mae: 0.1106, huber: 0.0174, swd: 0.0153, ept: 336.0000
      Epoch 8 composite train-obj: 0.010727
            No improvement (0.0162), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.0208, mae: 0.0863, huber: 0.0102, swd: 0.0092, ept: 335.9124
    Epoch [9/50], Val Losses: mse: 0.0397, mae: 0.0959, huber: 0.0157, swd: 0.0203, ept: 333.2427
    Epoch [9/50], Test Losses: mse: 0.0398, mae: 0.1019, huber: 0.0164, swd: 0.0226, ept: 334.0882
      Epoch 9 composite train-obj: 0.010210
            No improvement (0.0157), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.0115, mae: 0.0661, huber: 0.0057, swd: 0.0057, ept: 335.9563
    Epoch [10/50], Val Losses: mse: 0.0406, mae: 0.1088, huber: 0.0195, swd: 0.0106, ept: 335.3885
    Epoch [10/50], Test Losses: mse: 0.0420, mae: 0.1182, huber: 0.0204, swd: 0.0132, ept: 335.4284
      Epoch 10 composite train-obj: 0.005663
            No improvement (0.0195), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.0205, mae: 0.0829, huber: 0.0099, swd: 0.0090, ept: 335.9031
    Epoch [11/50], Val Losses: mse: 0.0194, mae: 0.0739, huber: 0.0092, swd: 0.0098, ept: 336.0000
    Epoch [11/50], Test Losses: mse: 0.0200, mae: 0.0790, huber: 0.0096, swd: 0.0112, ept: 335.9032
      Epoch 11 composite train-obj: 0.009945
            No improvement (0.0092), counter 4/5
    Epoch [12/50], Train Losses: mse: 0.0116, mae: 0.0638, huber: 0.0057, swd: 0.0057, ept: 335.9257
    Epoch [12/50], Val Losses: mse: 0.0134, mae: 0.0638, huber: 0.0066, swd: 0.0057, ept: 336.0000
    Epoch [12/50], Test Losses: mse: 0.0124, mae: 0.0648, huber: 0.0061, swd: 0.0055, ept: 336.0000
      Epoch 12 composite train-obj: 0.005671
            Val objective improved 0.0074 → 0.0066, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.0203, mae: 0.0845, huber: 0.0098, swd: 0.0098, ept: 335.8705
    Epoch [13/50], Val Losses: mse: 0.0166, mae: 0.0612, huber: 0.0076, swd: 0.0100, ept: 335.6437
    Epoch [13/50], Test Losses: mse: 0.0165, mae: 0.0631, huber: 0.0077, swd: 0.0108, ept: 335.7548
      Epoch 13 composite train-obj: 0.009842
            No improvement (0.0076), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.0120, mae: 0.0658, huber: 0.0059, swd: 0.0057, ept: 335.9757
    Epoch [14/50], Val Losses: mse: 0.0118, mae: 0.0624, huber: 0.0059, swd: 0.0053, ept: 336.0000
    Epoch [14/50], Test Losses: mse: 0.0141, mae: 0.0682, huber: 0.0070, swd: 0.0073, ept: 336.0000
      Epoch 14 composite train-obj: 0.005930
            Val objective improved 0.0066 → 0.0059, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.0126, mae: 0.0660, huber: 0.0062, swd: 0.0058, ept: 335.9651
    Epoch [15/50], Val Losses: mse: 0.0147, mae: 0.0689, huber: 0.0069, swd: 0.0077, ept: 335.5448
    Epoch [15/50], Test Losses: mse: 0.0151, mae: 0.0729, huber: 0.0072, swd: 0.0083, ept: 335.5376
      Epoch 15 composite train-obj: 0.006243
            No improvement (0.0069), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.0139, mae: 0.0711, huber: 0.0068, swd: 0.0066, ept: 335.9349
    Epoch [16/50], Val Losses: mse: 0.0151, mae: 0.0665, huber: 0.0075, swd: 0.0060, ept: 336.0000
    Epoch [16/50], Test Losses: mse: 0.0155, mae: 0.0714, huber: 0.0077, swd: 0.0066, ept: 336.0000
      Epoch 16 composite train-obj: 0.006827
            No improvement (0.0075), counter 2/5
    Epoch [17/50], Train Losses: mse: 0.0115, mae: 0.0621, huber: 0.0057, swd: 0.0058, ept: 335.9718
    Epoch [17/50], Val Losses: mse: 0.0391, mae: 0.1145, huber: 0.0194, swd: 0.0159, ept: 336.0000
    Epoch [17/50], Test Losses: mse: 0.0445, mae: 0.1237, huber: 0.0221, swd: 0.0212, ept: 336.0000
      Epoch 17 composite train-obj: 0.005684
            No improvement (0.0194), counter 3/5
    Epoch [18/50], Train Losses: mse: 0.0135, mae: 0.0682, huber: 0.0067, swd: 0.0068, ept: 335.9712
    Epoch [18/50], Val Losses: mse: 0.0310, mae: 0.0933, huber: 0.0155, swd: 0.0081, ept: 336.0000
    Epoch [18/50], Test Losses: mse: 0.0339, mae: 0.1011, huber: 0.0169, swd: 0.0102, ept: 336.0000
      Epoch 18 composite train-obj: 0.006673
            No improvement (0.0155), counter 4/5
    Epoch [19/50], Train Losses: mse: 0.0170, mae: 0.0717, huber: 0.0083, swd: 0.0077, ept: 335.8840
    Epoch [19/50], Val Losses: mse: 0.0318, mae: 0.1009, huber: 0.0148, swd: 0.0158, ept: 335.1353
    Epoch [19/50], Test Losses: mse: 0.0332, mae: 0.1066, huber: 0.0156, swd: 0.0181, ept: 335.0277
      Epoch 19 composite train-obj: 0.008311
    Epoch [19/50], Test Losses: mse: 0.0141, mae: 0.0682, huber: 0.0070, swd: 0.0073, ept: 336.0000
    Best round's Test MSE: 0.0141, MAE: 0.0682, SWD: 0.0073
    Best round's Validation MSE: 0.0118, MAE: 0.0624, SWD: 0.0053
    Best round's Test verification MSE : 0.0141, MAE: 0.0682, SWD: 0.0073
    Time taken: 225.09 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 3.2741, mae: 0.6370, huber: 0.4120, swd: 2.7207, ept: 302.0143
    Epoch [1/50], Val Losses: mse: 1.5351, mae: 0.3392, huber: 0.1727, swd: 1.1705, ept: 317.9606
    Epoch [1/50], Test Losses: mse: 1.3270, mae: 0.3412, huber: 0.1689, swd: 1.0053, ept: 314.2087
      Epoch 1 composite train-obj: 0.411986
            Val objective improved inf → 0.1727, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.3043, mae: 0.2089, huber: 0.0656, swd: 0.1677, ept: 324.8561
    Epoch [2/50], Val Losses: mse: 0.1505, mae: 0.2205, huber: 0.0645, swd: 0.0535, ept: 328.1567
    Epoch [2/50], Test Losses: mse: 0.1673, mae: 0.2417, huber: 0.0740, swd: 0.0634, ept: 329.6460
      Epoch 2 composite train-obj: 0.065637
            Val objective improved 0.1727 → 0.0645, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0555, mae: 0.1378, huber: 0.0248, swd: 0.0202, ept: 332.4101
    Epoch [3/50], Val Losses: mse: 0.0397, mae: 0.1146, huber: 0.0180, swd: 0.0117, ept: 332.9494
    Epoch [3/50], Test Losses: mse: 0.0409, mae: 0.1230, huber: 0.0189, swd: 0.0132, ept: 333.5543
      Epoch 3 composite train-obj: 0.024824
            Val objective improved 0.0645 → 0.0180, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0350, mae: 0.1138, huber: 0.0166, swd: 0.0135, ept: 334.9597
    Epoch [4/50], Val Losses: mse: 0.0417, mae: 0.1178, huber: 0.0193, swd: 0.0093, ept: 333.8151
    Epoch [4/50], Test Losses: mse: 0.0466, mae: 0.1283, huber: 0.0219, swd: 0.0123, ept: 334.5492
      Epoch 4 composite train-obj: 0.016601
            No improvement (0.0193), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0254, mae: 0.0961, huber: 0.0120, swd: 0.0108, ept: 335.5589
    Epoch [5/50], Val Losses: mse: 0.0233, mae: 0.0901, huber: 0.0113, swd: 0.0103, ept: 335.9198
    Epoch [5/50], Test Losses: mse: 0.0226, mae: 0.0924, huber: 0.0110, swd: 0.0102, ept: 335.8995
      Epoch 5 composite train-obj: 0.011970
            Val objective improved 0.0180 → 0.0113, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0208, mae: 0.0867, huber: 0.0101, swd: 0.0085, ept: 335.8975
    Epoch [6/50], Val Losses: mse: 0.1343, mae: 0.2129, huber: 0.0603, swd: 0.0567, ept: 331.5106
    Epoch [6/50], Test Losses: mse: 0.1580, mae: 0.2361, huber: 0.0716, swd: 0.0710, ept: 332.5688
      Epoch 6 composite train-obj: 0.010086
            No improvement (0.0603), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0345, mae: 0.1062, huber: 0.0161, swd: 0.0151, ept: 335.5018
    Epoch [7/50], Val Losses: mse: 0.0258, mae: 0.1010, huber: 0.0128, swd: 0.0101, ept: 336.0000
    Epoch [7/50], Test Losses: mse: 0.0279, mae: 0.1071, huber: 0.0138, swd: 0.0117, ept: 336.0000
      Epoch 7 composite train-obj: 0.016062
            No improvement (0.0128), counter 2/5
    Epoch [8/50], Train Losses: mse: 0.0203, mae: 0.0858, huber: 0.0099, swd: 0.0090, ept: 335.9014
    Epoch [8/50], Val Losses: mse: 0.0327, mae: 0.1048, huber: 0.0160, swd: 0.0130, ept: 336.0000
    Epoch [8/50], Test Losses: mse: 0.0365, mae: 0.1149, huber: 0.0179, swd: 0.0173, ept: 335.9980
      Epoch 8 composite train-obj: 0.009863
            No improvement (0.0160), counter 3/5
    Epoch [9/50], Train Losses: mse: 0.0172, mae: 0.0801, huber: 0.0084, swd: 0.0084, ept: 335.9186
    Epoch [9/50], Val Losses: mse: 0.0523, mae: 0.1077, huber: 0.0225, swd: 0.0162, ept: 333.1866
    Epoch [9/50], Test Losses: mse: 0.0499, mae: 0.1109, huber: 0.0217, swd: 0.0184, ept: 333.7635
      Epoch 9 composite train-obj: 0.008374
            No improvement (0.0225), counter 4/5
    Epoch [10/50], Train Losses: mse: 0.0134, mae: 0.0675, huber: 0.0064, swd: 0.0059, ept: 335.8329
    Epoch [10/50], Val Losses: mse: 0.0161, mae: 0.0671, huber: 0.0080, swd: 0.0070, ept: 336.0000
    Epoch [10/50], Test Losses: mse: 0.0176, mae: 0.0737, huber: 0.0088, swd: 0.0086, ept: 336.0000
      Epoch 10 composite train-obj: 0.006387
            Val objective improved 0.0113 → 0.0080, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0164, mae: 0.0773, huber: 0.0081, swd: 0.0077, ept: 335.9670
    Epoch [11/50], Val Losses: mse: 0.0176, mae: 0.0604, huber: 0.0077, swd: 0.0039, ept: 335.5006
    Epoch [11/50], Test Losses: mse: 0.0152, mae: 0.0613, huber: 0.0068, swd: 0.0039, ept: 335.6298
      Epoch 11 composite train-obj: 0.008086
            Val objective improved 0.0080 → 0.0077, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0209, mae: 0.0847, huber: 0.0102, swd: 0.0098, ept: 335.8829
    Epoch [12/50], Val Losses: mse: 0.0594, mae: 0.1452, huber: 0.0283, swd: 0.0234, ept: 335.1803
    Epoch [12/50], Test Losses: mse: 0.0670, mae: 0.1552, huber: 0.0318, swd: 0.0282, ept: 335.2719
      Epoch 12 composite train-obj: 0.010187
            No improvement (0.0283), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0132, mae: 0.0706, huber: 0.0065, swd: 0.0063, ept: 335.9550
    Epoch [13/50], Val Losses: mse: 0.0075, mae: 0.0530, huber: 0.0037, swd: 0.0036, ept: 336.0000
    Epoch [13/50], Test Losses: mse: 0.0082, mae: 0.0568, huber: 0.0041, swd: 0.0043, ept: 336.0000
      Epoch 13 composite train-obj: 0.006501
            Val objective improved 0.0077 → 0.0037, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0129, mae: 0.0661, huber: 0.0064, swd: 0.0059, ept: 335.9585
    Epoch [14/50], Val Losses: mse: 0.0141, mae: 0.0687, huber: 0.0070, swd: 0.0046, ept: 336.0000
    Epoch [14/50], Test Losses: mse: 0.0152, mae: 0.0731, huber: 0.0076, swd: 0.0053, ept: 336.0000
      Epoch 14 composite train-obj: 0.006350
            No improvement (0.0070), counter 1/5
    Epoch [15/50], Train Losses: mse: 0.0128, mae: 0.0681, huber: 0.0063, swd: 0.0062, ept: 335.9680
    Epoch [15/50], Val Losses: mse: 0.0137, mae: 0.0740, huber: 0.0068, swd: 0.0055, ept: 336.0000
    Epoch [15/50], Test Losses: mse: 0.0162, mae: 0.0816, huber: 0.0081, swd: 0.0070, ept: 336.0000
      Epoch 15 composite train-obj: 0.006332
            No improvement (0.0068), counter 2/5
    Epoch [16/50], Train Losses: mse: 0.0106, mae: 0.0624, huber: 0.0053, swd: 0.0053, ept: 335.9848
    Epoch [16/50], Val Losses: mse: 0.0085, mae: 0.0540, huber: 0.0042, swd: 0.0046, ept: 336.0000
    Epoch [16/50], Test Losses: mse: 0.0084, mae: 0.0557, huber: 0.0042, swd: 0.0045, ept: 336.0000
      Epoch 16 composite train-obj: 0.005254
            No improvement (0.0042), counter 3/5
    Epoch [17/50], Train Losses: mse: 0.0204, mae: 0.0810, huber: 0.0098, swd: 0.0100, ept: 335.8145
    Epoch [17/50], Val Losses: mse: 0.0199, mae: 0.0775, huber: 0.0095, swd: 0.0066, ept: 335.7272
    Epoch [17/50], Test Losses: mse: 0.0200, mae: 0.0814, huber: 0.0096, swd: 0.0074, ept: 335.8267
      Epoch 17 composite train-obj: 0.009847
            No improvement (0.0095), counter 4/5
    Epoch [18/50], Train Losses: mse: 0.0120, mae: 0.0653, huber: 0.0059, swd: 0.0059, ept: 335.9677
    Epoch [18/50], Val Losses: mse: 0.0119, mae: 0.0719, huber: 0.0059, swd: 0.0064, ept: 336.0000
    Epoch [18/50], Test Losses: mse: 0.0127, mae: 0.0765, huber: 0.0064, swd: 0.0070, ept: 336.0000
      Epoch 18 composite train-obj: 0.005893
    Epoch [18/50], Test Losses: mse: 0.0082, mae: 0.0568, huber: 0.0041, swd: 0.0043, ept: 336.0000
    Best round's Test MSE: 0.0082, MAE: 0.0568, SWD: 0.0043
    Best round's Validation MSE: 0.0075, MAE: 0.0530, SWD: 0.0036
    Best round's Test verification MSE : 0.0082, MAE: 0.0568, SWD: 0.0043
    Time taken: 212.68 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 3.2928, mae: 0.6448, huber: 0.4198, swd: 2.7044, ept: 301.1636
    Epoch [1/50], Val Losses: mse: 1.4607, mae: 0.3622, huber: 0.1804, swd: 1.0554, ept: 317.9495
    Epoch [1/50], Test Losses: mse: 1.2693, mae: 0.3746, huber: 0.1807, swd: 0.9088, ept: 314.1519
      Epoch 1 composite train-obj: 0.419761
            Val objective improved inf → 0.1804, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 0.2911, mae: 0.2179, huber: 0.0669, swd: 0.1526, ept: 325.7747
    Epoch [2/50], Val Losses: mse: 0.0734, mae: 0.1500, huber: 0.0295, swd: 0.0291, ept: 330.1330
    Epoch [2/50], Test Losses: mse: 0.0745, mae: 0.1590, huber: 0.0313, swd: 0.0312, ept: 330.8158
      Epoch 2 composite train-obj: 0.066944
            Val objective improved 0.1804 → 0.0295, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 0.0557, mae: 0.1381, huber: 0.0242, swd: 0.0208, ept: 334.0097
    Epoch [3/50], Val Losses: mse: 0.0519, mae: 0.1354, huber: 0.0242, swd: 0.0196, ept: 334.8777
    Epoch [3/50], Test Losses: mse: 0.0569, mae: 0.1463, huber: 0.0269, swd: 0.0219, ept: 335.0800
      Epoch 3 composite train-obj: 0.024245
            Val objective improved 0.0295 → 0.0242, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.0367, mae: 0.1159, huber: 0.0170, swd: 0.0149, ept: 335.4415
    Epoch [4/50], Val Losses: mse: 0.0572, mae: 0.1429, huber: 0.0270, swd: 0.0200, ept: 334.0064
    Epoch [4/50], Test Losses: mse: 0.0619, mae: 0.1529, huber: 0.0294, swd: 0.0243, ept: 334.3710
      Epoch 4 composite train-obj: 0.017015
            No improvement (0.0270), counter 1/5
    Epoch [5/50], Train Losses: mse: 0.0261, mae: 0.0970, huber: 0.0121, swd: 0.0111, ept: 335.6410
    Epoch [5/50], Val Losses: mse: 0.0320, mae: 0.1102, huber: 0.0158, swd: 0.0115, ept: 335.9983
    Epoch [5/50], Test Losses: mse: 0.0348, mae: 0.1186, huber: 0.0173, swd: 0.0140, ept: 335.9980
      Epoch 5 composite train-obj: 0.012149
            Val objective improved 0.0242 → 0.0158, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0255, mae: 0.0970, huber: 0.0123, swd: 0.0109, ept: 335.8622
    Epoch [6/50], Val Losses: mse: 0.1635, mae: 0.1989, huber: 0.0700, swd: 0.0420, ept: 335.1517
    Epoch [6/50], Test Losses: mse: 0.1761, mae: 0.2201, huber: 0.0778, swd: 0.0544, ept: 335.5639
      Epoch 6 composite train-obj: 0.012261
            No improvement (0.0700), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0261, mae: 0.0940, huber: 0.0124, swd: 0.0109, ept: 335.7951
    Epoch [7/50], Val Losses: mse: 0.0255, mae: 0.1020, huber: 0.0125, swd: 0.0156, ept: 336.0000
    Epoch [7/50], Test Losses: mse: 0.0270, mae: 0.1069, huber: 0.0133, swd: 0.0170, ept: 335.9811
      Epoch 7 composite train-obj: 0.012398
            Val objective improved 0.0158 → 0.0125, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0198, mae: 0.0869, huber: 0.0097, swd: 0.0089, ept: 335.9463
    Epoch [8/50], Val Losses: mse: 0.0588, mae: 0.1078, huber: 0.0221, swd: 0.0351, ept: 330.4365
    Epoch [8/50], Test Losses: mse: 0.0639, mae: 0.1133, huber: 0.0245, swd: 0.0426, ept: 330.9492
      Epoch 8 composite train-obj: 0.009654
            No improvement (0.0221), counter 1/5
    Epoch [9/50], Train Losses: mse: 0.0173, mae: 0.0769, huber: 0.0082, swd: 0.0085, ept: 335.8219
    Epoch [9/50], Val Losses: mse: 0.2848, mae: 0.2358, huber: 0.0929, swd: 0.1140, ept: 330.6793
    Epoch [9/50], Test Losses: mse: 0.2874, mae: 0.2571, huber: 0.1015, swd: 0.1172, ept: 331.7227
      Epoch 9 composite train-obj: 0.008177
            No improvement (0.0929), counter 2/5
    Epoch [10/50], Train Losses: mse: 0.0335, mae: 0.1012, huber: 0.0148, swd: 0.0152, ept: 335.5470
    Epoch [10/50], Val Losses: mse: 0.0310, mae: 0.1170, huber: 0.0154, swd: 0.0129, ept: 336.0000
    Epoch [10/50], Test Losses: mse: 0.0322, mae: 0.1219, huber: 0.0160, swd: 0.0138, ept: 336.0000
      Epoch 10 composite train-obj: 0.014838
            No improvement (0.0154), counter 3/5
    Epoch [11/50], Train Losses: mse: 0.0114, mae: 0.0627, huber: 0.0055, swd: 0.0052, ept: 335.9416
    Epoch [11/50], Val Losses: mse: 0.0086, mae: 0.0571, huber: 0.0043, swd: 0.0040, ept: 336.0000
    Epoch [11/50], Test Losses: mse: 0.0097, mae: 0.0622, huber: 0.0049, swd: 0.0048, ept: 336.0000
      Epoch 11 composite train-obj: 0.005527
            Val objective improved 0.0125 → 0.0043, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0176, mae: 0.0793, huber: 0.0086, swd: 0.0083, ept: 335.9392
    Epoch [12/50], Val Losses: mse: 0.0249, mae: 0.0741, huber: 0.0111, swd: 0.0111, ept: 335.5229
    Epoch [12/50], Test Losses: mse: 0.0232, mae: 0.0759, huber: 0.0106, swd: 0.0108, ept: 335.4769
      Epoch 12 composite train-obj: 0.008607
            No improvement (0.0111), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0157, mae: 0.0747, huber: 0.0077, swd: 0.0072, ept: 335.9470
    Epoch [13/50], Val Losses: mse: 0.0343, mae: 0.0897, huber: 0.0152, swd: 0.0174, ept: 334.9207
    Epoch [13/50], Test Losses: mse: 0.0352, mae: 0.0960, huber: 0.0160, swd: 0.0188, ept: 335.1302
      Epoch 13 composite train-obj: 0.007663
            No improvement (0.0152), counter 2/5
    Epoch [14/50], Train Losses: mse: 0.0124, mae: 0.0681, huber: 0.0061, swd: 0.0060, ept: 335.9511
    Epoch [14/50], Val Losses: mse: 0.0105, mae: 0.0651, huber: 0.0052, swd: 0.0044, ept: 336.0000
    Epoch [14/50], Test Losses: mse: 0.0099, mae: 0.0664, huber: 0.0049, swd: 0.0045, ept: 336.0000
      Epoch 14 composite train-obj: 0.006092
            No improvement (0.0052), counter 3/5
    Epoch [15/50], Train Losses: mse: 0.0133, mae: 0.0707, huber: 0.0066, swd: 0.0065, ept: 335.9673
    Epoch [15/50], Val Losses: mse: 0.0329, mae: 0.0939, huber: 0.0150, swd: 0.0187, ept: 334.9626
    Epoch [15/50], Test Losses: mse: 0.0352, mae: 0.1013, huber: 0.0164, swd: 0.0212, ept: 334.8923
      Epoch 15 composite train-obj: 0.006570
            No improvement (0.0150), counter 4/5
    Epoch [16/50], Train Losses: mse: 0.0167, mae: 0.0750, huber: 0.0080, swd: 0.0083, ept: 335.8675
    Epoch [16/50], Val Losses: mse: 0.0213, mae: 0.0874, huber: 0.0106, swd: 0.0090, ept: 335.8672
    Epoch [16/50], Test Losses: mse: 0.0223, mae: 0.0935, huber: 0.0111, swd: 0.0105, ept: 335.8827
      Epoch 16 composite train-obj: 0.008032
    Epoch [16/50], Test Losses: mse: 0.0097, mae: 0.0622, huber: 0.0049, swd: 0.0048, ept: 336.0000
    Best round's Test MSE: 0.0097, MAE: 0.0622, SWD: 0.0048
    Best round's Validation MSE: 0.0086, MAE: 0.0571, SWD: 0.0040
    Best round's Test verification MSE : 0.0097, MAE: 0.0622, SWD: 0.0048
    Time taken: 195.69 seconds
    
    ==================================================
    Experiment Summary (ACL_rossler_seq336_pred336_20250511_0326)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0107 ± 0.0025
      mae: 0.0624 ± 0.0047
      huber: 0.0053 ± 0.0012
      swd: 0.0055 ± 0.0013
      ept: 336.0000 ± 0.0000
      count: 36.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0093 ± 0.0018
      mae: 0.0575 ± 0.0038
      huber: 0.0046 ± 0.0009
      swd: 0.0043 ± 0.0008
      ept: 336.0000 ± 0.0000
      count: 36.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 633.61 seconds
    
    Experiment complete: ACL_rossler_seq336_pred336_20250511_0326
    Model: ACL
    Dataset: rossler
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 277
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 4.3458, mae: 0.8665, huber: 0.6013, swd: 3.2119, ept: 562.3547
    Epoch [1/50], Val Losses: mse: 3.3445, mae: 0.5221, huber: 0.3151, swd: 2.8756, ept: 631.7673
    Epoch [1/50], Test Losses: mse: 2.7634, mae: 0.5184, huber: 0.3014, swd: 2.3466, ept: 620.5020
      Epoch 1 composite train-obj: 0.601303
            Val objective improved inf → 0.3151, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.3485, mae: 0.4185, huber: 0.2341, swd: 1.9542, ept: 637.7356
    Epoch [2/50], Val Losses: mse: 2.5584, mae: 0.4392, huber: 0.2516, swd: 2.0473, ept: 637.1641
    Epoch [2/50], Test Losses: mse: 2.0734, mae: 0.4329, huber: 0.2345, swd: 1.6392, ept: 630.0838
      Epoch 2 composite train-obj: 0.234113
            Val objective improved 0.3151 → 0.2516, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.4169, mae: 0.3286, huber: 0.1630, swd: 1.0707, ept: 645.3501
    Epoch [3/50], Val Losses: mse: 0.9819, mae: 0.3016, huber: 0.1354, swd: 0.6548, ept: 650.8934
    Epoch [3/50], Test Losses: mse: 0.7618, mae: 0.2955, huber: 0.1233, swd: 0.4940, ept: 642.2796
      Epoch 3 composite train-obj: 0.163007
            Val objective improved 0.2516 → 0.1354, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.3904, mae: 0.2183, huber: 0.0727, swd: 0.2277, ept: 664.5629
    Epoch [4/50], Val Losses: mse: 0.1437, mae: 0.1663, huber: 0.0431, swd: 0.0435, ept: 667.6842
    Epoch [4/50], Test Losses: mse: 0.1327, mae: 0.1730, huber: 0.0437, swd: 0.0399, ept: 671.1304
      Epoch 4 composite train-obj: 0.072671
            Val objective improved 0.1354 → 0.0431, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0898, mae: 0.1648, huber: 0.0349, swd: 0.0283, ept: 689.4451
    Epoch [5/50], Val Losses: mse: 0.0912, mae: 0.1861, huber: 0.0404, swd: 0.0245, ept: 695.9207
    Epoch [5/50], Test Losses: mse: 0.1085, mae: 0.2095, huber: 0.0492, swd: 0.0298, ept: 701.0012
      Epoch 5 composite train-obj: 0.034856
            Val objective improved 0.0431 → 0.0404, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0456, mae: 0.1262, huber: 0.0202, swd: 0.0138, ept: 706.2898
    Epoch [6/50], Val Losses: mse: 0.0810, mae: 0.1650, huber: 0.0362, swd: 0.0277, ept: 698.5585
    Epoch [6/50], Test Losses: mse: 0.1081, mae: 0.1907, huber: 0.0478, swd: 0.0415, ept: 701.4010
      Epoch 6 composite train-obj: 0.020237
            Val objective improved 0.0404 → 0.0362, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0382, mae: 0.1197, huber: 0.0178, swd: 0.0118, ept: 711.6868
    Epoch [7/50], Val Losses: mse: 0.0388, mae: 0.1184, huber: 0.0181, swd: 0.0118, ept: 710.0414
    Epoch [7/50], Test Losses: mse: 0.0409, mae: 0.1257, huber: 0.0194, swd: 0.0129, ept: 710.7450
      Epoch 7 composite train-obj: 0.017844
            Val objective improved 0.0362 → 0.0181, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0218, mae: 0.0923, huber: 0.0102, swd: 0.0071, ept: 712.8540
    Epoch [8/50], Val Losses: mse: 0.0268, mae: 0.1012, huber: 0.0125, swd: 0.0087, ept: 713.2723
    Epoch [8/50], Test Losses: mse: 0.0251, mae: 0.1021, huber: 0.0118, swd: 0.0084, ept: 713.3767
      Epoch 8 composite train-obj: 0.010161
            Val objective improved 0.0181 → 0.0125, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0402, mae: 0.1148, huber: 0.0189, swd: 0.0128, ept: 713.1612
    Epoch [9/50], Val Losses: mse: 0.0220, mae: 0.0862, huber: 0.0101, swd: 0.0074, ept: 712.9827
    Epoch [9/50], Test Losses: mse: 0.0250, mae: 0.0940, huber: 0.0116, swd: 0.0086, ept: 713.5010
      Epoch 9 composite train-obj: 0.018914
            Val objective improved 0.0125 → 0.0101, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 0.0197, mae: 0.0866, huber: 0.0094, swd: 0.0070, ept: 713.9925
    Epoch [10/50], Val Losses: mse: 0.0291, mae: 0.1060, huber: 0.0142, swd: 0.0085, ept: 714.2227
    Epoch [10/50], Test Losses: mse: 0.0310, mae: 0.1127, huber: 0.0152, swd: 0.0108, ept: 714.8229
      Epoch 10 composite train-obj: 0.009382
            No improvement (0.0142), counter 1/5
    Epoch [11/50], Train Losses: mse: 0.0434, mae: 0.1220, huber: 0.0209, swd: 0.0148, ept: 714.2701
    Epoch [11/50], Val Losses: mse: 0.0162, mae: 0.0719, huber: 0.0076, swd: 0.0076, ept: 713.9189
    Epoch [11/50], Test Losses: mse: 0.0149, mae: 0.0727, huber: 0.0071, swd: 0.0067, ept: 715.2740
      Epoch 11 composite train-obj: 0.020897
            Val objective improved 0.0101 → 0.0076, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 0.0134, mae: 0.0717, huber: 0.0064, swd: 0.0048, ept: 715.0109
    Epoch [12/50], Val Losses: mse: 0.0341, mae: 0.1146, huber: 0.0164, swd: 0.0115, ept: 714.7791
    Epoch [12/50], Test Losses: mse: 0.0388, mae: 0.1251, huber: 0.0188, swd: 0.0130, ept: 716.3430
      Epoch 12 composite train-obj: 0.006440
            No improvement (0.0164), counter 1/5
    Epoch [13/50], Train Losses: mse: 0.0318, mae: 0.1065, huber: 0.0155, swd: 0.0114, ept: 715.3969
    Epoch [13/50], Val Losses: mse: 0.0132, mae: 0.0715, huber: 0.0065, swd: 0.0045, ept: 715.4297
    Epoch [13/50], Test Losses: mse: 0.0131, mae: 0.0735, huber: 0.0064, swd: 0.0048, ept: 716.9804
      Epoch 13 composite train-obj: 0.015495
            Val objective improved 0.0076 → 0.0065, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 0.0147, mae: 0.0752, huber: 0.0072, swd: 0.0055, ept: 716.3644
    Epoch [14/50], Val Losses: mse: 0.0099, mae: 0.0617, huber: 0.0048, swd: 0.0039, ept: 716.5582
    Epoch [14/50], Test Losses: mse: 0.0097, mae: 0.0643, huber: 0.0048, swd: 0.0038, ept: 717.6851
      Epoch 14 composite train-obj: 0.007207
            Val objective improved 0.0065 → 0.0048, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 0.0236, mae: 0.0952, huber: 0.0117, swd: 0.0088, ept: 717.5159
    Epoch [15/50], Val Losses: mse: 0.0368, mae: 0.1121, huber: 0.0182, swd: 0.0135, ept: 717.5174
    Epoch [15/50], Test Losses: mse: 0.0421, mae: 0.1235, huber: 0.0209, swd: 0.0169, ept: 718.4417
      Epoch 15 composite train-obj: 0.011710
            No improvement (0.0182), counter 1/5
    Epoch [16/50], Train Losses: mse: 0.0173, mae: 0.0800, huber: 0.0086, swd: 0.0066, ept: 718.7607
    Epoch [16/50], Val Losses: mse: 0.0082, mae: 0.0554, huber: 0.0041, swd: 0.0031, ept: 719.2664
    Epoch [16/50], Test Losses: mse: 0.0080, mae: 0.0567, huber: 0.0040, swd: 0.0032, ept: 719.3961
      Epoch 16 composite train-obj: 0.008566
            Val objective improved 0.0048 → 0.0041, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 0.0094, mae: 0.0607, huber: 0.0047, swd: 0.0038, ept: 719.7605
    Epoch [17/50], Val Losses: mse: 0.1263, mae: 0.1974, huber: 0.0578, swd: 0.0368, ept: 714.5278
    Epoch [17/50], Test Losses: mse: 0.1621, mae: 0.2222, huber: 0.0711, swd: 0.0628, ept: 716.2165
      Epoch 17 composite train-obj: 0.004655
            No improvement (0.0578), counter 1/5
    Epoch [18/50], Train Losses: mse: 0.0184, mae: 0.0809, huber: 0.0090, swd: 0.0070, ept: 719.7616
    Epoch [18/50], Val Losses: mse: 0.0162, mae: 0.0751, huber: 0.0081, swd: 0.0064, ept: 720.0000
    Epoch [18/50], Test Losses: mse: 0.0172, mae: 0.0801, huber: 0.0086, swd: 0.0072, ept: 720.0000
      Epoch 18 composite train-obj: 0.009011
            No improvement (0.0081), counter 2/5
    Epoch [19/50], Train Losses: mse: 0.0191, mae: 0.0808, huber: 0.0094, swd: 0.0072, ept: 719.9423
    Epoch [19/50], Val Losses: mse: 0.0135, mae: 0.0744, huber: 0.0067, swd: 0.0051, ept: 720.0000
    Epoch [19/50], Test Losses: mse: 0.0133, mae: 0.0765, huber: 0.0067, swd: 0.0054, ept: 720.0000
      Epoch 19 composite train-obj: 0.009423
            No improvement (0.0067), counter 3/5
    Epoch [20/50], Train Losses: mse: 0.0127, mae: 0.0697, huber: 0.0063, swd: 0.0053, ept: 719.9695
    Epoch [20/50], Val Losses: mse: 0.0197, mae: 0.0858, huber: 0.0098, swd: 0.0083, ept: 720.0000
    Epoch [20/50], Test Losses: mse: 0.0238, mae: 0.0944, huber: 0.0118, swd: 0.0101, ept: 720.0000
      Epoch 20 composite train-obj: 0.006336
            No improvement (0.0098), counter 4/5
    Epoch [21/50], Train Losses: mse: 0.0153, mae: 0.0716, huber: 0.0076, swd: 0.0061, ept: 719.9816
    Epoch [21/50], Val Losses: mse: 0.1422, mae: 0.1978, huber: 0.0670, swd: 0.0310, ept: 720.0000
    Epoch [21/50], Test Losses: mse: 0.1591, mae: 0.2150, huber: 0.0748, swd: 0.0470, ept: 720.0000
      Epoch 21 composite train-obj: 0.007609
    Epoch [21/50], Test Losses: mse: 0.0080, mae: 0.0567, huber: 0.0040, swd: 0.0032, ept: 719.3961
    Best round's Test MSE: 0.0080, MAE: 0.0567, SWD: 0.0032
    Best round's Validation MSE: 0.0082, MAE: 0.0554, SWD: 0.0031
    Best round's Test verification MSE : 0.0080, MAE: 0.0567, SWD: 0.0032
    Time taken: 178.11 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.2739, mae: 0.8575, huber: 0.5928, swd: 3.1168, ept: 564.8086
    Epoch [1/50], Val Losses: mse: 3.3737, mae: 0.5270, huber: 0.3203, swd: 2.9099, ept: 626.6874
    Epoch [1/50], Test Losses: mse: 2.7916, mae: 0.5253, huber: 0.3072, swd: 2.3725, ept: 617.7442
      Epoch 1 composite train-obj: 0.592775
            Val objective improved inf → 0.3203, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.3851, mae: 0.4252, huber: 0.2387, swd: 1.9978, ept: 637.6119
    Epoch [2/50], Val Losses: mse: 2.5558, mae: 0.3966, huber: 0.2325, swd: 2.0879, ept: 638.4698
    Epoch [2/50], Test Losses: mse: 2.0586, mae: 0.3793, huber: 0.2102, swd: 1.6668, ept: 631.0834
      Epoch 2 composite train-obj: 0.238666
            Val objective improved 0.3203 → 0.2325, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.4543, mae: 0.3397, huber: 0.1689, swd: 1.0823, ept: 646.3693
    Epoch [3/50], Val Losses: mse: 1.0616, mae: 0.3554, huber: 0.1599, swd: 0.6232, ept: 647.2890
    Epoch [3/50], Test Losses: mse: 0.8766, mae: 0.3663, huber: 0.1596, swd: 0.4984, ept: 641.2599
      Epoch 3 composite train-obj: 0.168928
            Val objective improved 0.2325 → 0.1599, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.4260, mae: 0.2147, huber: 0.0734, swd: 0.2240, ept: 657.4935
    Epoch [4/50], Val Losses: mse: 0.3241, mae: 0.2829, huber: 0.1078, swd: 0.0831, ept: 663.9161
    Epoch [4/50], Test Losses: mse: 0.3284, mae: 0.3066, huber: 0.1193, swd: 0.0941, ept: 655.1111
      Epoch 4 composite train-obj: 0.073368
            Val objective improved 0.1599 → 0.1078, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.1081, mae: 0.1648, huber: 0.0366, swd: 0.0288, ept: 675.6028
    Epoch [5/50], Val Losses: mse: 0.0848, mae: 0.1513, huber: 0.0312, swd: 0.0209, ept: 678.1384
    Epoch [5/50], Test Losses: mse: 0.0922, mae: 0.1634, huber: 0.0355, swd: 0.0224, ept: 679.3130
      Epoch 5 composite train-obj: 0.036621
            Val objective improved 0.1078 → 0.0312, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0495, mae: 0.1263, huber: 0.0205, swd: 0.0140, ept: 694.4511
    Epoch [6/50], Val Losses: mse: 0.1056, mae: 0.1787, huber: 0.0480, swd: 0.0172, ept: 696.0245
    Epoch [6/50], Test Losses: mse: 0.1003, mae: 0.1853, huber: 0.0468, swd: 0.0218, ept: 703.5572
      Epoch 6 composite train-obj: 0.020520
            No improvement (0.0480), counter 1/5
    Epoch [7/50], Train Losses: mse: 0.0394, mae: 0.1192, huber: 0.0181, swd: 0.0118, ept: 707.4525
    Epoch [7/50], Val Losses: mse: 0.0654, mae: 0.1345, huber: 0.0297, swd: 0.0182, ept: 709.8701
    Epoch [7/50], Test Losses: mse: 0.0743, mae: 0.1492, huber: 0.0347, swd: 0.0223, ept: 712.8801
      Epoch 7 composite train-obj: 0.018085
            Val objective improved 0.0312 → 0.0297, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 0.0312, mae: 0.1094, huber: 0.0149, swd: 0.0100, ept: 713.5185
    Epoch [8/50], Val Losses: mse: 0.0224, mae: 0.0918, huber: 0.0109, swd: 0.0068, ept: 713.7944
    Epoch [8/50], Test Losses: mse: 0.0256, mae: 0.1002, huber: 0.0125, swd: 0.0080, ept: 716.2880
      Epoch 8 composite train-obj: 0.014932
            Val objective improved 0.0297 → 0.0109, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0207, mae: 0.0908, huber: 0.0100, swd: 0.0073, ept: 716.7301
    Epoch [9/50], Val Losses: mse: 0.0236, mae: 0.0906, huber: 0.0114, swd: 0.0088, ept: 717.2060
    Epoch [9/50], Test Losses: mse: 0.0237, mae: 0.0923, huber: 0.0114, swd: 0.0089, ept: 717.9531
      Epoch 9 composite train-obj: 0.010027
            No improvement (0.0114), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.0269, mae: 0.1028, huber: 0.0132, swd: 0.0092, ept: 718.4542
    Epoch [10/50], Val Losses: mse: 0.0148, mae: 0.0766, huber: 0.0073, swd: 0.0056, ept: 719.0495
    Epoch [10/50], Test Losses: mse: 0.0140, mae: 0.0770, huber: 0.0069, swd: 0.0055, ept: 719.0414
      Epoch 10 composite train-obj: 0.013209
            Val objective improved 0.0109 → 0.0073, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 0.0268, mae: 0.0984, huber: 0.0131, swd: 0.0095, ept: 719.1922
    Epoch [11/50], Val Losses: mse: 0.0221, mae: 0.0936, huber: 0.0110, swd: 0.0076, ept: 720.0000
    Epoch [11/50], Test Losses: mse: 0.0244, mae: 0.1014, huber: 0.0121, swd: 0.0083, ept: 719.9925
      Epoch 11 composite train-obj: 0.013081
            No improvement (0.0110), counter 1/5
    Epoch [12/50], Train Losses: mse: 0.0325, mae: 0.1038, huber: 0.0157, swd: 0.0108, ept: 719.4536
    Epoch [12/50], Val Losses: mse: 0.1024, mae: 0.2051, huber: 0.0484, swd: 0.0294, ept: 715.3150
    Epoch [12/50], Test Losses: mse: 0.1210, mae: 0.2262, huber: 0.0573, swd: 0.0320, ept: 716.0124
      Epoch 12 composite train-obj: 0.015724
            No improvement (0.0484), counter 2/5
    Epoch [13/50], Train Losses: mse: 0.0199, mae: 0.0851, huber: 0.0098, swd: 0.0071, ept: 719.8366
    Epoch [13/50], Val Losses: mse: 0.0263, mae: 0.1043, huber: 0.0130, swd: 0.0112, ept: 720.0000
    Epoch [13/50], Test Losses: mse: 0.0284, mae: 0.1098, huber: 0.0140, swd: 0.0119, ept: 719.9938
      Epoch 13 composite train-obj: 0.009753
            No improvement (0.0130), counter 3/5
    Epoch [14/50], Train Losses: mse: 0.0167, mae: 0.0796, huber: 0.0083, swd: 0.0062, ept: 719.9189
    Epoch [14/50], Val Losses: mse: 0.0408, mae: 0.1229, huber: 0.0203, swd: 0.0105, ept: 719.9876
    Epoch [14/50], Test Losses: mse: 0.0462, mae: 0.1340, huber: 0.0229, swd: 0.0138, ept: 719.9813
      Epoch 14 composite train-obj: 0.008268
            No improvement (0.0203), counter 4/5
    Epoch [15/50], Train Losses: mse: 0.0208, mae: 0.0839, huber: 0.0102, swd: 0.0074, ept: 719.8839
    Epoch [15/50], Val Losses: mse: 0.0347, mae: 0.1181, huber: 0.0170, swd: 0.0115, ept: 719.7359
    Epoch [15/50], Test Losses: mse: 0.0361, mae: 0.1227, huber: 0.0177, swd: 0.0128, ept: 719.8428
      Epoch 15 composite train-obj: 0.010219
    Epoch [15/50], Test Losses: mse: 0.0140, mae: 0.0770, huber: 0.0069, swd: 0.0055, ept: 719.0414
    Best round's Test MSE: 0.0140, MAE: 0.0770, SWD: 0.0055
    Best round's Validation MSE: 0.0148, MAE: 0.0766, SWD: 0.0056
    Best round's Test verification MSE : 0.0140, MAE: 0.0770, SWD: 0.0055
    Time taken: 124.12 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.2354, mae: 0.8427, huber: 0.5817, swd: 3.1944, ept: 563.1058
    Epoch [1/50], Val Losses: mse: 3.2744, mae: 0.5071, huber: 0.3053, swd: 2.8904, ept: 621.6178
    Epoch [1/50], Test Losses: mse: 2.6968, mae: 0.5021, huber: 0.2907, swd: 2.3414, ept: 613.4845
      Epoch 1 composite train-obj: 0.581671
            Val objective improved inf → 0.3053, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.3338, mae: 0.4363, huber: 0.2420, swd: 1.9616, ept: 634.9585
    Epoch [2/50], Val Losses: mse: 2.4983, mae: 0.4326, huber: 0.2464, swd: 2.0463, ept: 636.8606
    Epoch [2/50], Test Losses: mse: 2.0238, mae: 0.4216, huber: 0.2283, swd: 1.6332, ept: 627.7362
      Epoch 2 composite train-obj: 0.242013
            Val objective improved 0.3053 → 0.2464, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.2532, mae: 0.3107, huber: 0.1494, swd: 0.9304, ept: 645.4458
    Epoch [3/50], Val Losses: mse: 0.6401, mae: 0.2574, huber: 0.1029, swd: 0.3513, ept: 655.0273
    Epoch [3/50], Test Losses: mse: 0.5154, mae: 0.2600, huber: 0.0962, swd: 0.2726, ept: 643.6882
      Epoch 3 composite train-obj: 0.149372
            Val objective improved 0.2464 → 0.1029, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 0.2558, mae: 0.2057, huber: 0.0608, swd: 0.1074, ept: 667.8105
    Epoch [4/50], Val Losses: mse: 0.1261, mae: 0.1705, huber: 0.0422, swd: 0.0280, ept: 672.3041
    Epoch [4/50], Test Losses: mse: 0.1129, mae: 0.1698, huber: 0.0397, swd: 0.0269, ept: 673.2095
      Epoch 4 composite train-obj: 0.060815
            Val objective improved 0.1029 → 0.0422, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 0.0726, mae: 0.1477, huber: 0.0290, swd: 0.0192, ept: 690.6181
    Epoch [5/50], Val Losses: mse: 0.0624, mae: 0.1497, huber: 0.0281, swd: 0.0125, ept: 695.2589
    Epoch [5/50], Test Losses: mse: 0.0649, mae: 0.1579, huber: 0.0296, swd: 0.0143, ept: 696.9790
      Epoch 5 composite train-obj: 0.028976
            Val objective improved 0.0422 → 0.0281, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 0.0459, mae: 0.1305, huber: 0.0213, swd: 0.0127, ept: 703.3875
    Epoch [6/50], Val Losses: mse: 0.0353, mae: 0.1136, huber: 0.0162, swd: 0.0126, ept: 702.4667
    Epoch [6/50], Test Losses: mse: 0.0374, mae: 0.1195, huber: 0.0175, swd: 0.0136, ept: 705.8497
      Epoch 6 composite train-obj: 0.021255
            Val objective improved 0.0281 → 0.0162, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 0.0288, mae: 0.1069, huber: 0.0139, swd: 0.0092, ept: 710.6848
    Epoch [7/50], Val Losses: mse: 0.0798, mae: 0.1760, huber: 0.0387, swd: 0.0240, ept: 711.1121
    Epoch [7/50], Test Losses: mse: 0.0837, mae: 0.1853, huber: 0.0408, swd: 0.0302, ept: 713.4470
      Epoch 7 composite train-obj: 0.013872
            No improvement (0.0387), counter 1/5
    Epoch [8/50], Train Losses: mse: 0.0347, mae: 0.1128, huber: 0.0169, swd: 0.0110, ept: 714.8175
    Epoch [8/50], Val Losses: mse: 0.0285, mae: 0.0994, huber: 0.0135, swd: 0.0095, ept: 710.3123
    Epoch [8/50], Test Losses: mse: 0.0262, mae: 0.0999, huber: 0.0125, swd: 0.0101, ept: 712.6928
      Epoch 8 composite train-obj: 0.016913
            Val objective improved 0.0162 → 0.0135, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 0.0322, mae: 0.1058, huber: 0.0157, swd: 0.0107, ept: 717.6737
    Epoch [9/50], Val Losses: mse: 0.1286, mae: 0.2113, huber: 0.0609, swd: 0.0320, ept: 718.0783
    Epoch [9/50], Test Losses: mse: 0.1814, mae: 0.2433, huber: 0.0815, swd: 0.0587, ept: 718.4345
      Epoch 9 composite train-obj: 0.015671
            No improvement (0.0609), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.0279, mae: 0.0989, huber: 0.0136, swd: 0.0093, ept: 718.9634
    Epoch [10/50], Val Losses: mse: 0.0562, mae: 0.1318, huber: 0.0241, swd: 0.0263, ept: 706.2208
    Epoch [10/50], Test Losses: mse: 0.0610, mae: 0.1365, huber: 0.0261, swd: 0.0265, ept: 709.3167
      Epoch 10 composite train-obj: 0.013640
            No improvement (0.0241), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.0192, mae: 0.0859, huber: 0.0095, swd: 0.0069, ept: 719.7025
    Epoch [11/50], Val Losses: mse: 0.0560, mae: 0.1418, huber: 0.0270, swd: 0.0218, ept: 719.7917
    Epoch [11/50], Test Losses: mse: 0.0633, mae: 0.1543, huber: 0.0307, swd: 0.0277, ept: 719.7365
      Epoch 11 composite train-obj: 0.009463
            No improvement (0.0270), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.0163, mae: 0.0810, huber: 0.0081, swd: 0.0061, ept: 719.9469
    Epoch [12/50], Val Losses: mse: 0.0187, mae: 0.0856, huber: 0.0091, swd: 0.0069, ept: 719.0541
    Epoch [12/50], Test Losses: mse: 0.0207, mae: 0.0910, huber: 0.0101, swd: 0.0075, ept: 719.1517
      Epoch 12 composite train-obj: 0.008092
            Val objective improved 0.0135 → 0.0091, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.0195, mae: 0.0841, huber: 0.0096, swd: 0.0069, ept: 719.9288
    Epoch [13/50], Val Losses: mse: 0.1987, mae: 0.2605, huber: 0.0883, swd: 0.0511, ept: 719.3341
    Epoch [13/50], Test Losses: mse: 0.2591, mae: 0.2939, huber: 0.1099, swd: 0.0817, ept: 718.9775
      Epoch 13 composite train-obj: 0.009599
            No improvement (0.0883), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.0259, mae: 0.0927, huber: 0.0127, swd: 0.0095, ept: 719.8377
    Epoch [14/50], Val Losses: mse: 0.0541, mae: 0.1280, huber: 0.0265, swd: 0.0173, ept: 720.0000
    Epoch [14/50], Test Losses: mse: 0.0669, mae: 0.1475, huber: 0.0330, swd: 0.0252, ept: 720.0000
      Epoch 14 composite train-obj: 0.012681
            No improvement (0.0265), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.0237, mae: 0.0926, huber: 0.0117, swd: 0.0090, ept: 719.7058
    Epoch [15/50], Val Losses: mse: 0.0102, mae: 0.0568, huber: 0.0050, swd: 0.0033, ept: 720.0000
    Epoch [15/50], Test Losses: mse: 0.0091, mae: 0.0571, huber: 0.0045, swd: 0.0033, ept: 719.9640
      Epoch 15 composite train-obj: 0.011703
            Val objective improved 0.0091 → 0.0050, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 0.0256, mae: 0.0948, huber: 0.0127, swd: 0.0091, ept: 719.8813
    Epoch [16/50], Val Losses: mse: 0.1623, mae: 0.2395, huber: 0.0747, swd: 0.0560, ept: 719.2037
    Epoch [16/50], Test Losses: mse: 0.2289, mae: 0.2736, huber: 0.0975, swd: 0.0905, ept: 719.5381
      Epoch 16 composite train-obj: 0.012670
            No improvement (0.0747), counter 1/5
    Epoch [17/50], Train Losses: mse: 0.0147, mae: 0.0711, huber: 0.0072, swd: 0.0058, ept: 719.9621
    Epoch [17/50], Val Losses: mse: 0.0120, mae: 0.0717, huber: 0.0060, swd: 0.0049, ept: 720.0000
    Epoch [17/50], Test Losses: mse: 0.0119, mae: 0.0728, huber: 0.0059, swd: 0.0051, ept: 720.0000
      Epoch 17 composite train-obj: 0.007239
            No improvement (0.0060), counter 2/5
    Epoch [18/50], Train Losses: mse: 0.0186, mae: 0.0797, huber: 0.0091, swd: 0.0070, ept: 719.6225
    Epoch [18/50], Val Losses: mse: 0.0279, mae: 0.0987, huber: 0.0139, swd: 0.0116, ept: 720.0000
    Epoch [18/50], Test Losses: mse: 0.0282, mae: 0.1044, huber: 0.0140, swd: 0.0130, ept: 720.0000
      Epoch 18 composite train-obj: 0.009090
            No improvement (0.0139), counter 3/5
    Epoch [19/50], Train Losses: mse: 0.0092, mae: 0.0598, huber: 0.0046, swd: 0.0039, ept: 719.9959
    Epoch [19/50], Val Losses: mse: 0.0254, mae: 0.0950, huber: 0.0125, swd: 0.0097, ept: 720.0000
    Epoch [19/50], Test Losses: mse: 0.0280, mae: 0.1020, huber: 0.0139, swd: 0.0106, ept: 719.7973
      Epoch 19 composite train-obj: 0.004582
            No improvement (0.0125), counter 4/5
    Epoch [20/50], Train Losses: mse: 0.0122, mae: 0.0674, huber: 0.0061, swd: 0.0048, ept: 719.9611
    Epoch [20/50], Val Losses: mse: 0.0055, mae: 0.0483, huber: 0.0027, swd: 0.0024, ept: 720.0000
    Epoch [20/50], Test Losses: mse: 0.0056, mae: 0.0508, huber: 0.0028, swd: 0.0025, ept: 720.0000
      Epoch 20 composite train-obj: 0.006077
            Val objective improved 0.0050 → 0.0027, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 0.0208, mae: 0.0844, huber: 0.0103, swd: 0.0078, ept: 719.9194
    Epoch [21/50], Val Losses: mse: 0.0151, mae: 0.0768, huber: 0.0075, swd: 0.0064, ept: 719.9996
    Epoch [21/50], Test Losses: mse: 0.0158, mae: 0.0815, huber: 0.0079, swd: 0.0065, ept: 720.0000
      Epoch 21 composite train-obj: 0.010286
            No improvement (0.0075), counter 1/5
    Epoch [22/50], Train Losses: mse: 0.0198, mae: 0.0819, huber: 0.0098, swd: 0.0076, ept: 719.8776
    Epoch [22/50], Val Losses: mse: 0.0144, mae: 0.0714, huber: 0.0071, swd: 0.0070, ept: 720.0000
    Epoch [22/50], Test Losses: mse: 0.0133, mae: 0.0717, huber: 0.0066, swd: 0.0067, ept: 720.0000
      Epoch 22 composite train-obj: 0.009783
            No improvement (0.0071), counter 2/5
    Epoch [23/50], Train Losses: mse: 0.0148, mae: 0.0723, huber: 0.0074, swd: 0.0060, ept: 719.9919
    Epoch [23/50], Val Losses: mse: 0.0188, mae: 0.0817, huber: 0.0092, swd: 0.0076, ept: 719.9885
    Epoch [23/50], Test Losses: mse: 0.0190, mae: 0.0849, huber: 0.0093, swd: 0.0076, ept: 719.9945
      Epoch 23 composite train-obj: 0.007351
            No improvement (0.0092), counter 3/5
    Epoch [24/50], Train Losses: mse: 0.0127, mae: 0.0683, huber: 0.0063, swd: 0.0050, ept: 719.9776
    Epoch [24/50], Val Losses: mse: 0.0216, mae: 0.0886, huber: 0.0108, swd: 0.0073, ept: 720.0000
    Epoch [24/50], Test Losses: mse: 0.0238, mae: 0.0943, huber: 0.0119, swd: 0.0089, ept: 720.0000
      Epoch 24 composite train-obj: 0.006304
            No improvement (0.0108), counter 4/5
    Epoch [25/50], Train Losses: mse: 0.0130, mae: 0.0708, huber: 0.0065, swd: 0.0054, ept: 719.9941
    Epoch [25/50], Val Losses: mse: 0.0158, mae: 0.0775, huber: 0.0078, swd: 0.0067, ept: 720.0000
    Epoch [25/50], Test Losses: mse: 0.0160, mae: 0.0801, huber: 0.0079, swd: 0.0076, ept: 720.0000
      Epoch 25 composite train-obj: 0.006500
    Epoch [25/50], Test Losses: mse: 0.0056, mae: 0.0508, huber: 0.0028, swd: 0.0025, ept: 720.0000
    Best round's Test MSE: 0.0056, MAE: 0.0508, SWD: 0.0025
    Best round's Validation MSE: 0.0055, MAE: 0.0483, SWD: 0.0024
    Best round's Test verification MSE : 0.0056, MAE: 0.0508, SWD: 0.0025
    Time taken: 209.44 seconds
    
    ==================================================
    Experiment Summary (ACL_rossler_seq336_pred720_20250513_1255)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 0.0092 ± 0.0035
      mae: 0.0615 ± 0.0112
      huber: 0.0046 ± 0.0017
      swd: 0.0037 ± 0.0013
      ept: 719.4792 ± 0.3957
      count: 33.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 0.0095 ± 0.0039
      mae: 0.0601 ± 0.0120
      huber: 0.0047 ± 0.0019
      swd: 0.0037 ± 0.0014
      ept: 719.4386 ± 0.4067
      count: 33.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 511.76 seconds
    
    Experiment complete: ACL_rossler_seq336_pred720_20250513_1255
    Model: ACL
    Dataset: rossler
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 4.7929, mae: 1.0037, huber: 0.7032, swd: 3.1898, ept: 87.8526
    Epoch [1/50], Val Losses: mse: 2.6326, mae: 0.5464, huber: 0.3354, swd: 2.2315, ept: 92.9971
    Epoch [1/50], Test Losses: mse: 2.4273, mae: 0.5739, huber: 0.3462, swd: 2.0539, ept: 92.7712
      Epoch 1 composite train-obj: 0.703182
            Val objective improved inf → 0.3354, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.1502, mae: 0.5481, huber: 0.3117, swd: 1.7953, ept: 93.4868
    Epoch [2/50], Val Losses: mse: 2.2210, mae: 0.4728, huber: 0.2778, swd: 1.8850, ept: 93.7545
    Epoch [2/50], Test Losses: mse: 2.0447, mae: 0.4909, huber: 0.2835, swd: 1.7320, ept: 93.4260
      Epoch 2 composite train-obj: 0.311676
            Val objective improved 0.3354 → 0.2778, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.9355, mae: 0.4826, huber: 0.2685, swd: 1.6131, ept: 93.7874
    Epoch [3/50], Val Losses: mse: 2.0164, mae: 0.4198, huber: 0.2406, swd: 1.7388, ept: 94.1783
    Epoch [3/50], Test Losses: mse: 1.8748, mae: 0.4304, huber: 0.2451, swd: 1.6087, ept: 93.5299
      Epoch 3 composite train-obj: 0.268457
            Val objective improved 0.2778 → 0.2406, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 1.8150, mae: 0.4464, huber: 0.2460, swd: 1.5006, ept: 93.9506
    Epoch [4/50], Val Losses: mse: 1.9668, mae: 0.5596, huber: 0.2965, swd: 1.6721, ept: 94.3324
    Epoch [4/50], Test Losses: mse: 1.8309, mae: 0.5792, huber: 0.3055, swd: 1.5541, ept: 94.0388
      Epoch 4 composite train-obj: 0.245963
            No improvement (0.2965), counter 1/5
    Epoch [5/50], Train Losses: mse: 1.7128, mae: 0.4279, huber: 0.2329, swd: 1.3997, ept: 94.0847
    Epoch [5/50], Val Losses: mse: 1.7953, mae: 0.5461, huber: 0.2963, swd: 1.4745, ept: 94.5969
    Epoch [5/50], Test Losses: mse: 1.7184, mae: 0.5786, huber: 0.3169, swd: 1.3922, ept: 94.0144
      Epoch 5 composite train-obj: 0.232937
            No improvement (0.2963), counter 2/5
    Epoch [6/50], Train Losses: mse: 1.5979, mae: 0.4104, huber: 0.2208, swd: 1.2890, ept: 94.2251
    Epoch [6/50], Val Losses: mse: 1.9641, mae: 0.5427, huber: 0.2886, swd: 1.5212, ept: 94.6687
    Epoch [6/50], Test Losses: mse: 1.7680, mae: 0.5530, huber: 0.2860, swd: 1.3503, ept: 94.3329
      Epoch 6 composite train-obj: 0.220825
            No improvement (0.2886), counter 3/5
    Epoch [7/50], Train Losses: mse: 1.6319, mae: 0.4038, huber: 0.2195, swd: 1.3200, ept: 94.1456
    Epoch [7/50], Val Losses: mse: 1.8439, mae: 0.4260, huber: 0.2339, swd: 1.5531, ept: 94.2778
    Epoch [7/50], Test Losses: mse: 1.6962, mae: 0.4415, huber: 0.2396, swd: 1.4247, ept: 93.6716
      Epoch 7 composite train-obj: 0.219523
            Val objective improved 0.2406 → 0.2339, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 1.4814, mae: 0.3815, huber: 0.2032, swd: 1.1740, ept: 94.3003
    Epoch [8/50], Val Losses: mse: 1.7848, mae: 0.4923, huber: 0.2599, swd: 1.5147, ept: 94.5114
    Epoch [8/50], Test Losses: mse: 1.7014, mae: 0.5198, huber: 0.2762, swd: 1.4323, ept: 94.0027
      Epoch 8 composite train-obj: 0.203160
            No improvement (0.2599), counter 1/5
    Epoch [9/50], Train Losses: mse: 1.4382, mae: 0.3719, huber: 0.1962, swd: 1.1446, ept: 94.3538
    Epoch [9/50], Val Losses: mse: 1.2711, mae: 0.4174, huber: 0.2111, swd: 1.0058, ept: 94.5995
    Epoch [9/50], Test Losses: mse: 1.0868, mae: 0.4145, huber: 0.1992, swd: 0.8493, ept: 94.5573
      Epoch 9 composite train-obj: 0.196233
            Val objective improved 0.2339 → 0.2111, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 1.3785, mae: 0.3700, huber: 0.1951, swd: 1.0982, ept: 94.3579
    Epoch [10/50], Val Losses: mse: 2.5641, mae: 0.6213, huber: 0.3871, swd: 1.8897, ept: 93.3013
    Epoch [10/50], Test Losses: mse: 2.4760, mae: 0.6584, huber: 0.4144, swd: 1.8176, ept: 93.1730
      Epoch 10 composite train-obj: 0.195129
            No improvement (0.3871), counter 1/5
    Epoch [11/50], Train Losses: mse: 1.6967, mae: 0.4310, huber: 0.2384, swd: 1.3560, ept: 94.0601
    Epoch [11/50], Val Losses: mse: 1.8812, mae: 0.5206, huber: 0.2828, swd: 1.4916, ept: 93.9070
    Epoch [11/50], Test Losses: mse: 1.8243, mae: 0.5531, huber: 0.3027, swd: 1.4487, ept: 93.3581
      Epoch 11 composite train-obj: 0.238350
            No improvement (0.2828), counter 2/5
    Epoch [12/50], Train Losses: mse: 1.1521, mae: 0.3497, huber: 0.1767, swd: 0.8795, ept: 94.5593
    Epoch [12/50], Val Losses: mse: 1.6755, mae: 0.5706, huber: 0.3055, swd: 1.3308, ept: 94.6935
    Epoch [12/50], Test Losses: mse: 1.5450, mae: 0.5871, huber: 0.3123, swd: 1.2298, ept: 94.5969
      Epoch 12 composite train-obj: 0.176739
            No improvement (0.3055), counter 3/5
    Epoch [13/50], Train Losses: mse: 1.1362, mae: 0.3420, huber: 0.1736, swd: 0.8789, ept: 94.5162
    Epoch [13/50], Val Losses: mse: 7.4953, mae: 1.2308, huber: 0.9143, swd: 5.4975, ept: 91.8651
    Epoch [13/50], Test Losses: mse: 6.9329, mae: 1.2904, huber: 0.9624, swd: 4.9934, ept: 90.9384
      Epoch 13 composite train-obj: 0.173622
            No improvement (0.9143), counter 4/5
    Epoch [14/50], Train Losses: mse: 1.9067, mae: 0.4505, huber: 0.2528, swd: 1.5314, ept: 94.0324
    Epoch [14/50], Val Losses: mse: 2.0009, mae: 0.5925, huber: 0.3269, swd: 1.7061, ept: 94.0780
    Epoch [14/50], Test Losses: mse: 1.9132, mae: 0.6217, huber: 0.3444, swd: 1.6349, ept: 93.6325
      Epoch 14 composite train-obj: 0.252799
    Epoch [14/50], Test Losses: mse: 1.0868, mae: 0.4145, huber: 0.1992, swd: 0.8493, ept: 94.5573
    Best round's Test MSE: 1.0868, MAE: 0.4145, SWD: 0.8493
    Best round's Validation MSE: 1.2711, MAE: 0.4174, SWD: 1.0058
    Best round's Test verification MSE : 1.0868, MAE: 0.4145, SWD: 0.8493
    Time taken: 117.81 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.2886, mae: 0.9639, huber: 0.6688, swd: 2.8875, ept: 87.2411
    Epoch [1/50], Val Losses: mse: 2.2905, mae: 0.4937, huber: 0.2815, swd: 1.9871, ept: 93.7948
    Epoch [1/50], Test Losses: mse: 2.0940, mae: 0.4976, huber: 0.2802, swd: 1.8107, ept: 93.3223
      Epoch 1 composite train-obj: 0.668776
            Val objective improved inf → 0.2815, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 1.9643, mae: 0.4977, huber: 0.2763, swd: 1.6307, ept: 93.7305
    Epoch [2/50], Val Losses: mse: 2.0178, mae: 0.4999, huber: 0.2760, swd: 1.6301, ept: 94.0173
    Epoch [2/50], Test Losses: mse: 1.8439, mae: 0.5110, huber: 0.2755, swd: 1.4782, ept: 93.8871
      Epoch 2 composite train-obj: 0.276332
            Val objective improved 0.2815 → 0.2760, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.7280, mae: 0.4442, huber: 0.2412, swd: 1.4082, ept: 94.0038
    Epoch [3/50], Val Losses: mse: 1.7291, mae: 0.4108, huber: 0.2318, swd: 1.5127, ept: 94.2913
    Epoch [3/50], Test Losses: mse: 1.6236, mae: 0.4284, huber: 0.2409, swd: 1.4014, ept: 93.7603
      Epoch 3 composite train-obj: 0.241180
            Val objective improved 0.2760 → 0.2318, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 1.5939, mae: 0.4099, huber: 0.2189, swd: 1.2973, ept: 94.2030
    Epoch [4/50], Val Losses: mse: 1.4098, mae: 0.4602, huber: 0.2270, swd: 1.1968, ept: 94.6008
    Epoch [4/50], Test Losses: mse: 1.2785, mae: 0.4702, huber: 0.2271, swd: 1.0730, ept: 94.3838
      Epoch 4 composite train-obj: 0.218854
            Val objective improved 0.2318 → 0.2270, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 1.7170, mae: 0.4143, huber: 0.2269, swd: 1.4353, ept: 94.0310
    Epoch [5/50], Val Losses: mse: 2.2661, mae: 0.6279, huber: 0.3807, swd: 1.7267, ept: 93.7089
    Epoch [5/50], Test Losses: mse: 2.1657, mae: 0.6638, huber: 0.4063, swd: 1.6381, ept: 93.5533
      Epoch 5 composite train-obj: 0.226929
            No improvement (0.3807), counter 1/5
    Epoch [6/50], Train Losses: mse: 1.7378, mae: 0.4301, huber: 0.2368, swd: 1.4197, ept: 94.0492
    Epoch [6/50], Val Losses: mse: 1.8631, mae: 0.5036, huber: 0.2797, swd: 1.5870, ept: 93.6630
    Epoch [6/50], Test Losses: mse: 1.7037, mae: 0.5213, huber: 0.2816, swd: 1.4368, ept: 93.6113
      Epoch 6 composite train-obj: 0.236796
            No improvement (0.2797), counter 2/5
    Epoch [7/50], Train Losses: mse: 1.5555, mae: 0.3936, huber: 0.2118, swd: 1.2893, ept: 94.1834
    Epoch [7/50], Val Losses: mse: 1.3651, mae: 0.4575, huber: 0.2291, swd: 1.1356, ept: 94.8265
    Epoch [7/50], Test Losses: mse: 1.2637, mae: 0.4764, huber: 0.2354, swd: 1.0362, ept: 94.7024
      Epoch 7 composite train-obj: 0.211816
            No improvement (0.2291), counter 3/5
    Epoch [8/50], Train Losses: mse: 1.2832, mae: 0.3581, huber: 0.1843, swd: 1.0275, ept: 94.4497
    Epoch [8/50], Val Losses: mse: 1.1610, mae: 0.3367, huber: 0.1703, swd: 0.9335, ept: 94.6249
    Epoch [8/50], Test Losses: mse: 1.0054, mae: 0.3379, huber: 0.1637, swd: 0.7930, ept: 94.4668
      Epoch 8 composite train-obj: 0.184319
            Val objective improved 0.2270 → 0.1703, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 1.0807, mae: 0.3329, huber: 0.1653, swd: 0.8628, ept: 94.6011
    Epoch [9/50], Val Losses: mse: 1.4961, mae: 0.3709, huber: 0.1964, swd: 1.0380, ept: 94.4043
    Epoch [9/50], Test Losses: mse: 1.3664, mae: 0.3733, huber: 0.1910, swd: 0.9477, ept: 94.3767
      Epoch 9 composite train-obj: 0.165317
            No improvement (0.1964), counter 1/5
    Epoch [10/50], Train Losses: mse: 0.8289, mae: 0.3124, huber: 0.1465, swd: 0.6394, ept: 94.7833
    Epoch [10/50], Val Losses: mse: 1.5205, mae: 0.3806, huber: 0.2073, swd: 1.3589, ept: 94.0270
    Epoch [10/50], Test Losses: mse: 1.4228, mae: 0.4007, huber: 0.2165, swd: 1.2638, ept: 93.5545
      Epoch 10 composite train-obj: 0.146540
            No improvement (0.2073), counter 2/5
    Epoch [11/50], Train Losses: mse: 0.8768, mae: 0.3109, huber: 0.1483, swd: 0.6967, ept: 94.7671
    Epoch [11/50], Val Losses: mse: 0.7867, mae: 0.3893, huber: 0.1797, swd: 0.6452, ept: 94.9546
    Epoch [11/50], Test Losses: mse: 0.7165, mae: 0.3961, huber: 0.1773, swd: 0.5825, ept: 95.0266
      Epoch 11 composite train-obj: 0.148338
            No improvement (0.1797), counter 3/5
    Epoch [12/50], Train Losses: mse: 0.5730, mae: 0.2744, huber: 0.1196, swd: 0.4286, ept: 95.0227
    Epoch [12/50], Val Losses: mse: 0.8162, mae: 0.3497, huber: 0.1667, swd: 0.6686, ept: 94.7860
    Epoch [12/50], Test Losses: mse: 0.7558, mae: 0.3572, huber: 0.1672, swd: 0.6239, ept: 94.6140
      Epoch 12 composite train-obj: 0.119630
            Val objective improved 0.1703 → 0.1667, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 0.5135, mae: 0.2606, huber: 0.1111, swd: 0.3793, ept: 95.1067
    Epoch [13/50], Val Losses: mse: 1.6863, mae: 0.6634, huber: 0.3783, swd: 1.4161, ept: 94.3129
    Epoch [13/50], Test Losses: mse: 1.6671, mae: 0.7053, huber: 0.4098, swd: 1.4006, ept: 93.8689
      Epoch 13 composite train-obj: 0.111051
            No improvement (0.3783), counter 1/5
    Epoch [14/50], Train Losses: mse: 0.5124, mae: 0.2642, huber: 0.1124, swd: 0.3828, ept: 95.1338
    Epoch [14/50], Val Losses: mse: 1.3827, mae: 0.5960, huber: 0.3220, swd: 1.1946, ept: 94.7374
    Epoch [14/50], Test Losses: mse: 1.3667, mae: 0.6259, huber: 0.3418, swd: 1.1827, ept: 94.4288
      Epoch 14 composite train-obj: 0.112394
            No improvement (0.3220), counter 2/5
    Epoch [15/50], Train Losses: mse: 0.4888, mae: 0.2618, huber: 0.1112, swd: 0.3591, ept: 95.1614
    Epoch [15/50], Val Losses: mse: 0.7143, mae: 0.5351, huber: 0.2701, swd: 0.5690, ept: 95.3192
    Epoch [15/50], Test Losses: mse: 0.8806, mae: 0.5816, huber: 0.3044, swd: 0.7338, ept: 95.0243
      Epoch 15 composite train-obj: 0.111188
            No improvement (0.2701), counter 3/5
    Epoch [16/50], Train Losses: mse: 0.5330, mae: 0.2741, huber: 0.1172, swd: 0.3959, ept: 95.1359
    Epoch [16/50], Val Losses: mse: 0.9796, mae: 0.4041, huber: 0.1977, swd: 0.7761, ept: 94.6373
    Epoch [16/50], Test Losses: mse: 0.8937, mae: 0.4171, huber: 0.2000, swd: 0.7044, ept: 94.5503
      Epoch 16 composite train-obj: 0.117154
            No improvement (0.1977), counter 4/5
    Epoch [17/50], Train Losses: mse: 0.4055, mae: 0.2392, huber: 0.0972, swd: 0.2913, ept: 95.2420
    Epoch [17/50], Val Losses: mse: 0.4914, mae: 0.4308, huber: 0.1898, swd: 0.3907, ept: 95.2880
    Epoch [17/50], Test Losses: mse: 0.5280, mae: 0.4564, huber: 0.2058, swd: 0.4273, ept: 95.4419
      Epoch 17 composite train-obj: 0.097228
    Epoch [17/50], Test Losses: mse: 0.7558, mae: 0.3572, huber: 0.1672, swd: 0.6239, ept: 94.6140
    Best round's Test MSE: 0.7558, MAE: 0.3572, SWD: 0.6239
    Best round's Validation MSE: 0.8162, MAE: 0.3497, SWD: 0.6686
    Best round's Test verification MSE : 0.7558, MAE: 0.3572, SWD: 0.6239
    Time taken: 143.57 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 4.6966, mae: 0.9685, huber: 0.6746, swd: 2.8208, ept: 87.4288
    Epoch [1/50], Val Losses: mse: 2.6468, mae: 0.5996, huber: 0.3606, swd: 1.9750, ept: 92.9880
    Epoch [1/50], Test Losses: mse: 2.4380, mae: 0.6326, huber: 0.3748, swd: 1.8089, ept: 92.8747
      Epoch 1 composite train-obj: 0.674555
            Val objective improved inf → 0.3606, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.0575, mae: 0.5221, huber: 0.2935, swd: 1.5679, ept: 93.6223
    Epoch [2/50], Val Losses: mse: 2.2652, mae: 0.6286, huber: 0.3579, swd: 1.7472, ept: 94.0059
    Epoch [2/50], Test Losses: mse: 2.1105, mae: 0.6502, huber: 0.3685, swd: 1.6251, ept: 93.7755
      Epoch 2 composite train-obj: 0.293479
            Val objective improved 0.3606 → 0.3579, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 1.8647, mae: 0.4631, huber: 0.2558, swd: 1.4185, ept: 93.8747
    Epoch [3/50], Val Losses: mse: 1.9308, mae: 0.4218, huber: 0.2355, swd: 1.5002, ept: 94.2066
    Epoch [3/50], Test Losses: mse: 1.7824, mae: 0.4340, huber: 0.2396, swd: 1.3747, ept: 93.6860
      Epoch 3 composite train-obj: 0.255763
            Val objective improved 0.3579 → 0.2355, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 1.7403, mae: 0.4313, huber: 0.2352, swd: 1.3149, ept: 94.0384
    Epoch [4/50], Val Losses: mse: 1.9337, mae: 0.4109, huber: 0.2363, swd: 1.5058, ept: 93.9532
    Epoch [4/50], Test Losses: mse: 1.8038, mae: 0.4304, huber: 0.2461, swd: 1.3932, ept: 93.4659
      Epoch 4 composite train-obj: 0.235205
            No improvement (0.2363), counter 1/5
    Epoch [5/50], Train Losses: mse: 1.6852, mae: 0.4154, huber: 0.2264, swd: 1.2689, ept: 94.0785
    Epoch [5/50], Val Losses: mse: 1.8006, mae: 0.4420, huber: 0.2387, swd: 1.3687, ept: 94.2732
    Epoch [5/50], Test Losses: mse: 1.6433, mae: 0.4485, huber: 0.2359, swd: 1.2402, ept: 94.0767
      Epoch 5 composite train-obj: 0.226364
            No improvement (0.2387), counter 2/5
    Epoch [6/50], Train Losses: mse: 1.5526, mae: 0.3978, huber: 0.2124, swd: 1.1492, ept: 94.1980
    Epoch [6/50], Val Losses: mse: 1.9993, mae: 0.4913, huber: 0.2851, swd: 1.4482, ept: 93.6943
    Epoch [6/50], Test Losses: mse: 1.8995, mae: 0.5326, huber: 0.3120, swd: 1.3551, ept: 93.4149
      Epoch 6 composite train-obj: 0.212380
            No improvement (0.2851), counter 3/5
    Epoch [7/50], Train Losses: mse: 1.8001, mae: 0.4145, huber: 0.2301, swd: 1.2660, ept: 94.1456
    Epoch [7/50], Val Losses: mse: 4.7015, mae: 1.2429, huber: 0.8963, swd: 3.2873, ept: 91.9048
    Epoch [7/50], Test Losses: mse: 4.7697, mae: 1.3244, huber: 0.9646, swd: 3.2756, ept: 90.9568
      Epoch 7 composite train-obj: 0.230116
            No improvement (0.8963), counter 4/5
    Epoch [8/50], Train Losses: mse: 2.2154, mae: 0.5414, huber: 0.3205, swd: 1.6727, ept: 93.2552
    Epoch [8/50], Val Losses: mse: 2.2545, mae: 0.5619, huber: 0.3211, swd: 1.6566, ept: 93.9188
    Epoch [8/50], Test Losses: mse: 2.0468, mae: 0.5722, huber: 0.3199, swd: 1.4978, ept: 93.7828
      Epoch 8 composite train-obj: 0.320503
    Epoch [8/50], Test Losses: mse: 1.7824, mae: 0.4340, huber: 0.2396, swd: 1.3747, ept: 93.6860
    Best round's Test MSE: 1.7824, MAE: 0.4340, SWD: 1.3747
    Best round's Validation MSE: 1.9308, MAE: 0.4218, SWD: 1.5002
    Best round's Test verification MSE : 1.7824, MAE: 0.4340, SWD: 1.3747
    Time taken: 67.73 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_rossler_seq336_pred96_20250513_1216)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 1.2084 ± 0.4278
      mae: 0.4019 ± 0.0326
      huber: 0.2020 ± 0.0296
      swd: 0.9493 ± 0.3146
      ept: 94.2858 ± 0.4247
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 1.3393 ± 0.4576
      mae: 0.3963 ± 0.0330
      huber: 0.2044 ± 0.0285
      swd: 1.0582 ± 0.3415
      ept: 94.5307 ± 0.2415
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 329.17 seconds
    
    Experiment complete: TimeMixer_rossler_seq336_pred96_20250513_1216
    Model: TimeMixer
    Dataset: rossler
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 281
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 6.1355, mae: 1.2860, huber: 0.9515, swd: 4.4852, ept: 166.9049
    Epoch [1/50], Val Losses: mse: 3.7053, mae: 0.7039, huber: 0.4671, swd: 3.2597, ept: 184.3087
    Epoch [1/50], Test Losses: mse: 3.2632, mae: 0.6797, huber: 0.4400, swd: 2.8943, ept: 184.2334
      Epoch 1 composite train-obj: 0.951547
            Val objective improved inf → 0.4671, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 3.2801, mae: 0.7545, huber: 0.4815, swd: 2.8445, ept: 185.7114
    Epoch [2/50], Val Losses: mse: 3.4384, mae: 0.6752, huber: 0.4360, swd: 3.0424, ept: 186.0878
    Epoch [2/50], Test Losses: mse: 3.0440, mae: 0.6553, huber: 0.4138, swd: 2.7104, ept: 185.6411
      Epoch 2 composite train-obj: 0.481539
            Val objective improved 0.4671 → 0.4360, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 3.0507, mae: 0.6771, huber: 0.4285, swd: 2.6786, ept: 186.6916
    Epoch [3/50], Val Losses: mse: 3.2124, mae: 0.6728, huber: 0.4253, swd: 2.9018, ept: 188.2872
    Epoch [3/50], Test Losses: mse: 2.8673, mae: 0.6610, huber: 0.4080, swd: 2.6059, ept: 187.0459
      Epoch 3 composite train-obj: 0.428462
            Val objective improved 0.4360 → 0.4253, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 2.9179, mae: 0.6330, huber: 0.3995, swd: 2.5642, ept: 187.1952
    Epoch [4/50], Val Losses: mse: 3.0442, mae: 0.5826, huber: 0.3772, swd: 2.7196, ept: 188.6270
    Epoch [4/50], Test Losses: mse: 2.6919, mae: 0.5636, huber: 0.3563, swd: 2.4106, ept: 187.3133
      Epoch 4 composite train-obj: 0.399458
            Val objective improved 0.4253 → 0.3772, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 2.8432, mae: 0.6083, huber: 0.3837, swd: 2.4826, ept: 187.4320
    Epoch [5/50], Val Losses: mse: 3.3831, mae: 0.7300, huber: 0.4693, swd: 3.1016, ept: 187.8549
    Epoch [5/50], Test Losses: mse: 3.0990, mae: 0.7363, huber: 0.4699, swd: 2.8428, ept: 186.0551
      Epoch 5 composite train-obj: 0.383743
            No improvement (0.4693), counter 1/5
    Epoch [6/50], Train Losses: mse: 2.7342, mae: 0.5931, huber: 0.3730, swd: 2.3807, ept: 187.7125
    Epoch [6/50], Val Losses: mse: 3.1264, mae: 0.6247, huber: 0.3984, swd: 2.7626, ept: 189.0455
    Epoch [6/50], Test Losses: mse: 2.7558, mae: 0.6104, huber: 0.3790, swd: 2.4368, ept: 187.1981
      Epoch 6 composite train-obj: 0.372968
            No improvement (0.3984), counter 2/5
    Epoch [7/50], Train Losses: mse: 2.6411, mae: 0.5789, huber: 0.3625, swd: 2.2820, ept: 187.8798
    Epoch [7/50], Val Losses: mse: 2.6437, mae: 0.6165, huber: 0.3753, swd: 2.3068, ept: 189.5258
    Epoch [7/50], Test Losses: mse: 2.2970, mae: 0.5955, huber: 0.3476, swd: 2.0073, ept: 189.1476
      Epoch 7 composite train-obj: 0.362476
            Val objective improved 0.3772 → 0.3753, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 2.5512, mae: 0.5714, huber: 0.3555, swd: 2.1694, ept: 187.9477
    Epoch [8/50], Val Losses: mse: 2.7267, mae: 0.5695, huber: 0.3544, swd: 2.3598, ept: 189.8246
    Epoch [8/50], Test Losses: mse: 2.3632, mae: 0.5485, huber: 0.3308, swd: 2.0310, ept: 188.3278
      Epoch 8 composite train-obj: 0.355537
            Val objective improved 0.3753 → 0.3544, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 2.2071, mae: 0.5455, huber: 0.3319, swd: 1.8034, ept: 188.3672
    Epoch [9/50], Val Losses: mse: 2.5306, mae: 0.5508, huber: 0.3371, swd: 2.1488, ept: 188.8495
    Epoch [9/50], Test Losses: mse: 2.1699, mae: 0.5291, huber: 0.3128, swd: 1.8247, ept: 187.4940
      Epoch 9 composite train-obj: 0.331905
            Val objective improved 0.3544 → 0.3371, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 1.9443, mae: 0.5200, huber: 0.3114, swd: 1.5547, ept: 188.5695
    Epoch [10/50], Val Losses: mse: 2.0044, mae: 0.5152, huber: 0.3069, swd: 1.6294, ept: 189.2069
    Epoch [10/50], Test Losses: mse: 1.7153, mae: 0.4917, huber: 0.2804, swd: 1.3861, ept: 188.3829
      Epoch 10 composite train-obj: 0.311411
            Val objective improved 0.3371 → 0.3069, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 1.7852, mae: 0.4989, huber: 0.2958, swd: 1.4009, ept: 188.8630
    Epoch [11/50], Val Losses: mse: 2.2919, mae: 0.5721, huber: 0.3427, swd: 1.9009, ept: 189.2827
    Epoch [11/50], Test Losses: mse: 1.9841, mae: 0.5546, huber: 0.3208, swd: 1.6414, ept: 187.3657
      Epoch 11 composite train-obj: 0.295787
            No improvement (0.3427), counter 1/5
    Epoch [12/50], Train Losses: mse: 1.7982, mae: 0.5028, huber: 0.2972, swd: 1.4488, ept: 189.0506
    Epoch [12/50], Val Losses: mse: 1.9786, mae: 0.5067, huber: 0.3205, swd: 1.6747, ept: 189.1292
    Epoch [12/50], Test Losses: mse: 1.7286, mae: 0.4867, huber: 0.2981, swd: 1.4639, ept: 189.4540
      Epoch 12 composite train-obj: 0.297220
            No improvement (0.3205), counter 2/5
    Epoch [13/50], Train Losses: mse: 1.6216, mae: 0.4788, huber: 0.2794, swd: 1.2761, ept: 189.2896
    Epoch [13/50], Val Losses: mse: 1.3996, mae: 0.5457, huber: 0.3014, swd: 1.0430, ept: 189.9652
    Epoch [13/50], Test Losses: mse: 1.2312, mae: 0.5316, huber: 0.2808, swd: 0.9146, ept: 190.3466
      Epoch 13 composite train-obj: 0.279382
            Val objective improved 0.3069 → 0.3014, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 1.5463, mae: 0.4640, huber: 0.2697, swd: 1.2211, ept: 189.4796
    Epoch [14/50], Val Losses: mse: 1.7012, mae: 0.5309, huber: 0.3189, swd: 1.2791, ept: 189.7767
    Epoch [14/50], Test Losses: mse: 1.8796, mae: 0.5495, huber: 0.3326, swd: 1.4749, ept: 188.7289
      Epoch 14 composite train-obj: 0.269661
            No improvement (0.3189), counter 1/5
    Epoch [15/50], Train Losses: mse: 1.5034, mae: 0.4614, huber: 0.2654, swd: 1.1968, ept: 189.7516
    Epoch [15/50], Val Losses: mse: 1.5652, mae: 0.5107, huber: 0.2937, swd: 1.2378, ept: 189.1853
    Epoch [15/50], Test Losses: mse: 1.4343, mae: 0.5014, huber: 0.2789, swd: 1.1546, ept: 188.6040
      Epoch 15 composite train-obj: 0.265352
            Val objective improved 0.3014 → 0.2937, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 1.3969, mae: 0.4517, huber: 0.2572, swd: 1.0857, ept: 189.9047
    Epoch [16/50], Val Losses: mse: 2.0231, mae: 0.4838, huber: 0.3007, swd: 1.6603, ept: 188.3211
    Epoch [16/50], Test Losses: mse: 1.7599, mae: 0.4652, huber: 0.2801, swd: 1.4495, ept: 187.0574
      Epoch 16 composite train-obj: 0.257155
            No improvement (0.3007), counter 1/5
    Epoch [17/50], Train Losses: mse: 1.2716, mae: 0.4315, huber: 0.2430, swd: 0.9643, ept: 190.2371
    Epoch [17/50], Val Losses: mse: 1.3568, mae: 0.5423, huber: 0.2996, swd: 1.0333, ept: 190.1583
    Epoch [17/50], Test Losses: mse: 1.1741, mae: 0.5215, huber: 0.2754, swd: 0.9077, ept: 189.4974
      Epoch 17 composite train-obj: 0.243017
            No improvement (0.2996), counter 2/5
    Epoch [18/50], Train Losses: mse: 1.3132, mae: 0.4378, huber: 0.2473, swd: 1.0140, ept: 190.2386
    Epoch [18/50], Val Losses: mse: 1.4540, mae: 0.4993, huber: 0.2875, swd: 1.1428, ept: 190.4676
    Epoch [18/50], Test Losses: mse: 1.2721, mae: 0.4881, huber: 0.2706, swd: 1.0133, ept: 189.2437
      Epoch 18 composite train-obj: 0.247262
            Val objective improved 0.2937 → 0.2875, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 1.1880, mae: 0.4229, huber: 0.2350, swd: 0.8872, ept: 190.4617
    Epoch [19/50], Val Losses: mse: 1.9865, mae: 0.5451, huber: 0.3416, swd: 1.7315, ept: 189.5213
    Epoch [19/50], Test Losses: mse: 1.8154, mae: 0.5474, huber: 0.3379, swd: 1.5839, ept: 188.4808
      Epoch 19 composite train-obj: 0.235004
            No improvement (0.3416), counter 1/5
    Epoch [20/50], Train Losses: mse: 1.2345, mae: 0.4196, huber: 0.2348, swd: 0.9590, ept: 190.5219
    Epoch [20/50], Val Losses: mse: 1.2201, mae: 0.4986, huber: 0.2786, swd: 0.9209, ept: 190.2056
    Epoch [20/50], Test Losses: mse: 1.1055, mae: 0.4952, huber: 0.2679, swd: 0.8406, ept: 188.7771
      Epoch 20 composite train-obj: 0.234781
            Val objective improved 0.2875 → 0.2786, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 1.0897, mae: 0.4074, huber: 0.2236, swd: 0.8001, ept: 190.9181
    Epoch [21/50], Val Losses: mse: 1.7537, mae: 0.5363, huber: 0.3110, swd: 1.4288, ept: 189.4061
    Epoch [21/50], Test Losses: mse: 1.5540, mae: 0.5239, huber: 0.2931, swd: 1.2727, ept: 188.4307
      Epoch 21 composite train-obj: 0.223554
            No improvement (0.3110), counter 1/5
    Epoch [22/50], Train Losses: mse: 1.1678, mae: 0.4126, huber: 0.2295, swd: 0.8763, ept: 190.6140
    Epoch [22/50], Val Losses: mse: 1.4899, mae: 0.5258, huber: 0.2996, swd: 1.2058, ept: 190.2860
    Epoch [22/50], Test Losses: mse: 1.3056, mae: 0.5117, huber: 0.2808, swd: 1.0615, ept: 189.1725
      Epoch 22 composite train-obj: 0.229461
            No improvement (0.2996), counter 2/5
    Epoch [23/50], Train Losses: mse: 1.0498, mae: 0.4045, huber: 0.2207, swd: 0.7669, ept: 191.0482
    Epoch [23/50], Val Losses: mse: 1.5059, mae: 0.5485, huber: 0.3195, swd: 1.2081, ept: 190.3448
    Epoch [23/50], Test Losses: mse: 1.3948, mae: 0.5445, huber: 0.3126, swd: 1.1204, ept: 189.7204
      Epoch 23 composite train-obj: 0.220674
            No improvement (0.3195), counter 3/5
    Epoch [24/50], Train Losses: mse: 0.9976, mae: 0.3947, huber: 0.2139, swd: 0.7197, ept: 191.2226
    Epoch [24/50], Val Losses: mse: 1.6940, mae: 0.5953, huber: 0.3554, swd: 1.2541, ept: 189.3468
    Epoch [24/50], Test Losses: mse: 1.6569, mae: 0.6084, huber: 0.3604, swd: 1.2335, ept: 187.6861
      Epoch 24 composite train-obj: 0.213860
            No improvement (0.3554), counter 4/5
    Epoch [25/50], Train Losses: mse: 1.1744, mae: 0.4012, huber: 0.2231, swd: 0.9166, ept: 191.0463
    Epoch [25/50], Val Losses: mse: 1.8180, mae: 0.5813, huber: 0.3578, swd: 1.4204, ept: 188.9053
    Epoch [25/50], Test Losses: mse: 1.7157, mae: 0.5897, huber: 0.3602, swd: 1.3244, ept: 187.6680
      Epoch 25 composite train-obj: 0.223072
    Epoch [25/50], Test Losses: mse: 1.1055, mae: 0.4952, huber: 0.2679, swd: 0.8406, ept: 188.7771
    Best round's Test MSE: 1.1055, MAE: 0.4952, SWD: 0.8406
    Best round's Validation MSE: 1.2201, MAE: 0.4986, SWD: 0.9209
    Best round's Test verification MSE : 1.1055, MAE: 0.4952, SWD: 0.8406
    Time taken: 211.84 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.0134, mae: 1.2394, huber: 0.9147, swd: 4.4496, ept: 169.3608
    Epoch [1/50], Val Losses: mse: 3.8364, mae: 0.7267, huber: 0.4856, swd: 3.3533, ept: 183.9619
    Epoch [1/50], Test Losses: mse: 3.4120, mae: 0.7252, huber: 0.4746, swd: 2.9901, ept: 184.1092
      Epoch 1 composite train-obj: 0.914708
            Val objective improved inf → 0.4856, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 3.1855, mae: 0.7173, huber: 0.4561, swd: 2.7857, ept: 185.9166
    Epoch [2/50], Val Losses: mse: 3.5850, mae: 0.7754, huber: 0.4982, swd: 3.1731, ept: 187.3438
    Epoch [2/50], Test Losses: mse: 3.1922, mae: 0.7652, huber: 0.4798, swd: 2.8295, ept: 186.4999
      Epoch 2 composite train-obj: 0.456066
            No improvement (0.4982), counter 1/5
    Epoch [3/50], Train Losses: mse: 2.9683, mae: 0.6511, huber: 0.4107, swd: 2.6129, ept: 186.8788
    Epoch [3/50], Val Losses: mse: 3.3384, mae: 0.6316, huber: 0.4138, swd: 2.9334, ept: 187.2792
    Epoch [3/50], Test Losses: mse: 2.9406, mae: 0.6154, huber: 0.3926, swd: 2.5970, ept: 186.7733
      Epoch 3 composite train-obj: 0.410666
            Val objective improved 0.4856 → 0.4138, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 2.9448, mae: 0.6258, huber: 0.3958, swd: 2.6132, ept: 187.0145
    Epoch [4/50], Val Losses: mse: 3.3855, mae: 0.7020, huber: 0.4605, swd: 3.0341, ept: 187.7178
    Epoch [4/50], Test Losses: mse: 3.1071, mae: 0.7000, huber: 0.4558, swd: 2.7910, ept: 186.5483
      Epoch 4 composite train-obj: 0.395778
            No improvement (0.4605), counter 1/5
    Epoch [5/50], Train Losses: mse: 2.8594, mae: 0.6111, huber: 0.3860, swd: 2.5268, ept: 187.2936
    Epoch [5/50], Val Losses: mse: 2.9172, mae: 0.5616, huber: 0.3644, swd: 2.6097, ept: 188.7279
    Epoch [5/50], Test Losses: mse: 2.5741, mae: 0.5493, huber: 0.3463, swd: 2.3099, ept: 187.3655
      Epoch 5 composite train-obj: 0.385978
            Val objective improved 0.4138 → 0.3644, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 2.7212, mae: 0.5902, huber: 0.3702, swd: 2.3816, ept: 187.7191
    Epoch [6/50], Val Losses: mse: 2.8859, mae: 0.5856, huber: 0.3804, swd: 2.5057, ept: 187.5656
    Epoch [6/50], Test Losses: mse: 2.5252, mae: 0.5625, huber: 0.3557, swd: 2.1995, ept: 187.7033
      Epoch 6 composite train-obj: 0.370158
            No improvement (0.3804), counter 1/5
    Epoch [7/50], Train Losses: mse: 2.4732, mae: 0.5634, huber: 0.3489, swd: 2.0949, ept: 188.0794
    Epoch [7/50], Val Losses: mse: 2.6804, mae: 0.5505, huber: 0.3625, swd: 2.3043, ept: 188.7646
    Epoch [7/50], Test Losses: mse: 2.3598, mae: 0.5379, huber: 0.3465, swd: 2.0164, ept: 186.7833
      Epoch 7 composite train-obj: 0.348894
            Val objective improved 0.3644 → 0.3625, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 2.1765, mae: 0.5376, huber: 0.3273, swd: 1.7791, ept: 188.2081
    Epoch [8/50], Val Losses: mse: 1.8892, mae: 0.5023, huber: 0.3064, swd: 1.4826, ept: 189.3557
    Epoch [8/50], Test Losses: mse: 1.6537, mae: 0.4884, huber: 0.2865, swd: 1.2847, ept: 188.8438
      Epoch 8 composite train-obj: 0.327270
            Val objective improved 0.3625 → 0.3064, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 1.9960, mae: 0.5187, huber: 0.3118, swd: 1.6064, ept: 188.4108
    Epoch [9/50], Val Losses: mse: 2.2602, mae: 0.5449, huber: 0.3414, swd: 1.8760, ept: 189.0099
    Epoch [9/50], Test Losses: mse: 2.0231, mae: 0.5342, huber: 0.3272, swd: 1.6694, ept: 187.5969
      Epoch 9 composite train-obj: 0.311818
            No improvement (0.3414), counter 1/5
    Epoch [10/50], Train Losses: mse: 1.7431, mae: 0.4920, huber: 0.2904, swd: 1.3843, ept: 189.0630
    Epoch [10/50], Val Losses: mse: 2.6112, mae: 0.6313, huber: 0.3827, swd: 2.2725, ept: 189.2387
    Epoch [10/50], Test Losses: mse: 2.3800, mae: 0.6333, huber: 0.3779, swd: 2.0734, ept: 187.5521
      Epoch 10 composite train-obj: 0.290428
            No improvement (0.3827), counter 2/5
    Epoch [11/50], Train Losses: mse: 1.6208, mae: 0.4806, huber: 0.2804, swd: 1.2697, ept: 189.1851
    Epoch [11/50], Val Losses: mse: 1.2910, mae: 0.4441, huber: 0.2546, swd: 0.8430, ept: 189.1679
    Epoch [11/50], Test Losses: mse: 1.1851, mae: 0.4342, huber: 0.2394, swd: 0.7926, ept: 188.4833
      Epoch 11 composite train-obj: 0.280430
            Val objective improved 0.3064 → 0.2546, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 1.5145, mae: 0.4661, huber: 0.2688, swd: 1.1829, ept: 189.5465
    Epoch [12/50], Val Losses: mse: 1.3230, mae: 0.6087, huber: 0.3396, swd: 0.9665, ept: 191.7423
    Epoch [12/50], Test Losses: mse: 1.2528, mae: 0.6172, huber: 0.3417, swd: 0.9467, ept: 190.9636
      Epoch 12 composite train-obj: 0.268763
            No improvement (0.3396), counter 1/5
    Epoch [13/50], Train Losses: mse: 1.4422, mae: 0.4503, huber: 0.2559, swd: 1.1321, ept: 189.9534
    Epoch [13/50], Val Losses: mse: 2.0588, mae: 0.5289, huber: 0.3145, swd: 1.7807, ept: 188.7929
    Epoch [13/50], Test Losses: mse: 1.7933, mae: 0.5098, huber: 0.2935, swd: 1.5582, ept: 188.1198
      Epoch 13 composite train-obj: 0.255851
            No improvement (0.3145), counter 2/5
    Epoch [14/50], Train Losses: mse: 1.3542, mae: 0.4389, huber: 0.2490, swd: 1.0687, ept: 190.1781
    Epoch [14/50], Val Losses: mse: 1.6368, mae: 0.6210, huber: 0.3549, swd: 1.2786, ept: 190.9449
    Epoch [14/50], Test Losses: mse: 1.4924, mae: 0.6208, huber: 0.3486, swd: 1.1687, ept: 189.6604
      Epoch 14 composite train-obj: 0.248991
            No improvement (0.3549), counter 3/5
    Epoch [15/50], Train Losses: mse: 1.3125, mae: 0.4401, huber: 0.2483, swd: 1.0084, ept: 190.1135
    Epoch [15/50], Val Losses: mse: 1.6074, mae: 0.6692, huber: 0.3931, swd: 1.2563, ept: 189.9131
    Epoch [15/50], Test Losses: mse: 1.5698, mae: 0.6887, huber: 0.4035, swd: 1.2439, ept: 189.8935
      Epoch 15 composite train-obj: 0.248325
            No improvement (0.3931), counter 4/5
    Epoch [16/50], Train Losses: mse: 1.1736, mae: 0.4206, huber: 0.2336, swd: 0.8756, ept: 190.5874
    Epoch [16/50], Val Losses: mse: 1.1455, mae: 0.4839, huber: 0.2635, swd: 0.8358, ept: 191.7488
    Epoch [16/50], Test Losses: mse: 1.0142, mae: 0.4774, huber: 0.2502, swd: 0.7532, ept: 191.1801
      Epoch 16 composite train-obj: 0.233574
    Epoch [16/50], Test Losses: mse: 1.1851, mae: 0.4342, huber: 0.2394, swd: 0.7926, ept: 188.4833
    Best round's Test MSE: 1.1851, MAE: 0.4342, SWD: 0.7926
    Best round's Validation MSE: 1.2910, MAE: 0.4441, SWD: 0.8430
    Best round's Test verification MSE : 1.1851, MAE: 0.4342, SWD: 0.7926
    Time taken: 138.97 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.5075, mae: 1.3122, huber: 0.9802, swd: 4.0401, ept: 166.1107
    Epoch [1/50], Val Losses: mse: 3.8449, mae: 0.7427, huber: 0.4927, swd: 3.0777, ept: 184.1194
    Epoch [1/50], Test Losses: mse: 3.4129, mae: 0.7292, huber: 0.4747, swd: 2.7384, ept: 183.8883
      Epoch 1 composite train-obj: 0.980237
            Val objective improved inf → 0.4927, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 3.2674, mae: 0.7362, huber: 0.4685, swd: 2.6227, ept: 185.5997
    Epoch [2/50], Val Losses: mse: 3.5906, mae: 0.6510, huber: 0.4350, swd: 2.9223, ept: 185.3155
    Epoch [2/50], Test Losses: mse: 3.1862, mae: 0.6417, huber: 0.4194, swd: 2.6032, ept: 184.9924
      Epoch 2 composite train-obj: 0.468526
            Val objective improved 0.4927 → 0.4350, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 3.0645, mae: 0.6696, huber: 0.4229, swd: 2.4921, ept: 186.5334
    Epoch [3/50], Val Losses: mse: 3.2992, mae: 0.6336, huber: 0.4161, swd: 2.7321, ept: 187.9409
    Epoch [3/50], Test Losses: mse: 2.9364, mae: 0.6208, huber: 0.3975, swd: 2.4420, ept: 186.8972
      Epoch 3 composite train-obj: 0.422889
            Val objective improved 0.4350 → 0.4161, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 2.9445, mae: 0.6327, huber: 0.3995, swd: 2.3979, ept: 187.1602
    Epoch [4/50], Val Losses: mse: 3.2978, mae: 0.6791, huber: 0.4348, swd: 2.6802, ept: 187.9804
    Epoch [4/50], Test Losses: mse: 2.9107, mae: 0.6550, huber: 0.4075, swd: 2.3797, ept: 187.0450
      Epoch 4 composite train-obj: 0.399454
            No improvement (0.4348), counter 1/5
    Epoch [5/50], Train Losses: mse: 2.8814, mae: 0.6152, huber: 0.3878, swd: 2.3380, ept: 187.4031
    Epoch [5/50], Val Losses: mse: 3.2565, mae: 0.5868, huber: 0.3950, swd: 2.7132, ept: 188.0000
    Epoch [5/50], Test Losses: mse: 2.8804, mae: 0.5624, huber: 0.3709, swd: 2.4136, ept: 186.6860
      Epoch 5 composite train-obj: 0.387758
            Val objective improved 0.4161 → 0.3950, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 2.8181, mae: 0.5947, huber: 0.3749, swd: 2.2808, ept: 187.5859
    Epoch [6/50], Val Losses: mse: 3.2454, mae: 0.5942, huber: 0.3915, swd: 2.6610, ept: 187.3073
    Epoch [6/50], Test Losses: mse: 2.8638, mae: 0.5771, huber: 0.3687, swd: 2.3604, ept: 186.5651
      Epoch 6 composite train-obj: 0.374905
            Val objective improved 0.3950 → 0.3915, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 2.7336, mae: 0.5821, huber: 0.3661, swd: 2.2020, ept: 187.8462
    Epoch [7/50], Val Losses: mse: 3.1583, mae: 0.6949, huber: 0.4486, swd: 2.5883, ept: 187.9323
    Epoch [7/50], Test Losses: mse: 2.8619, mae: 0.6962, huber: 0.4440, swd: 2.3525, ept: 186.8225
      Epoch 7 composite train-obj: 0.366086
            No improvement (0.4486), counter 1/5
    Epoch [8/50], Train Losses: mse: 2.4852, mae: 0.5562, huber: 0.3451, swd: 1.9576, ept: 188.3190
    Epoch [8/50], Val Losses: mse: 3.0526, mae: 0.5754, huber: 0.3826, swd: 2.4583, ept: 188.3074
    Epoch [8/50], Test Losses: mse: 2.6858, mae: 0.5598, huber: 0.3610, swd: 2.1593, ept: 186.8899
      Epoch 8 composite train-obj: 0.345061
            Val objective improved 0.3915 → 0.3826, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 2.4052, mae: 0.5450, huber: 0.3371, swd: 1.8716, ept: 188.3965
    Epoch [9/50], Val Losses: mse: 3.3767, mae: 0.6684, huber: 0.4385, swd: 2.6675, ept: 187.1798
    Epoch [9/50], Test Losses: mse: 3.0419, mae: 0.6559, huber: 0.4256, swd: 2.3958, ept: 185.8164
      Epoch 9 composite train-obj: 0.337052
            No improvement (0.4385), counter 1/5
    Epoch [10/50], Train Losses: mse: 2.1765, mae: 0.5423, huber: 0.3283, swd: 1.6397, ept: 188.6104
    Epoch [10/50], Val Losses: mse: 2.2176, mae: 0.5469, huber: 0.3359, swd: 1.7568, ept: 189.2211
    Epoch [10/50], Test Losses: mse: 1.8997, mae: 0.5168, huber: 0.3030, swd: 1.5131, ept: 189.2739
      Epoch 10 composite train-obj: 0.328327
            Val objective improved 0.3826 → 0.3359, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 2.0655, mae: 0.5227, huber: 0.3147, swd: 1.5825, ept: 188.9091
    Epoch [11/50], Val Losses: mse: 2.4378, mae: 0.5783, huber: 0.3602, swd: 1.9773, ept: 188.9306
    Epoch [11/50], Test Losses: mse: 2.1785, mae: 0.5676, huber: 0.3454, swd: 1.7712, ept: 187.3865
      Epoch 11 composite train-obj: 0.314674
            No improvement (0.3602), counter 1/5
    Epoch [12/50], Train Losses: mse: 1.8422, mae: 0.5000, huber: 0.2949, swd: 1.3756, ept: 189.3133
    Epoch [12/50], Val Losses: mse: 2.6755, mae: 0.5575, huber: 0.3529, swd: 2.1693, ept: 188.6571
    Epoch [12/50], Test Losses: mse: 2.3429, mae: 0.5338, huber: 0.3278, swd: 1.9102, ept: 187.2968
      Epoch 12 composite train-obj: 0.294915
            No improvement (0.3529), counter 2/5
    Epoch [13/50], Train Losses: mse: 1.8037, mae: 0.4847, huber: 0.2863, swd: 1.3813, ept: 189.5632
    Epoch [13/50], Val Losses: mse: 1.7929, mae: 0.5269, huber: 0.3086, swd: 1.3368, ept: 190.1618
    Epoch [13/50], Test Losses: mse: 1.5480, mae: 0.4994, huber: 0.2800, swd: 1.1605, ept: 190.7785
      Epoch 13 composite train-obj: 0.286303
            Val objective improved 0.3359 → 0.3086, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 1.6141, mae: 0.4671, huber: 0.2712, swd: 1.1901, ept: 189.8139
    Epoch [14/50], Val Losses: mse: 2.2713, mae: 0.5646, huber: 0.3613, swd: 1.7632, ept: 189.2738
    Epoch [14/50], Test Losses: mse: 2.0335, mae: 0.5547, huber: 0.3490, swd: 1.5770, ept: 187.5070
      Epoch 14 composite train-obj: 0.271210
            No improvement (0.3613), counter 1/5
    Epoch [15/50], Train Losses: mse: 1.5624, mae: 0.4633, huber: 0.2663, swd: 1.1555, ept: 190.0089
    Epoch [15/50], Val Losses: mse: 1.5595, mae: 0.4292, huber: 0.2626, swd: 1.1939, ept: 190.6945
    Epoch [15/50], Test Losses: mse: 1.3786, mae: 0.4161, huber: 0.2449, swd: 1.0651, ept: 189.5337
      Epoch 15 composite train-obj: 0.266291
            Val objective improved 0.3086 → 0.2626, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 1.4134, mae: 0.4451, huber: 0.2529, swd: 1.0108, ept: 190.2646
    Epoch [16/50], Val Losses: mse: 2.4012, mae: 0.5606, huber: 0.3597, swd: 1.8655, ept: 189.5810
    Epoch [16/50], Test Losses: mse: 2.1366, mae: 0.5565, huber: 0.3496, swd: 1.6538, ept: 187.8196
      Epoch 16 composite train-obj: 0.252880
            No improvement (0.3597), counter 1/5
    Epoch [17/50], Train Losses: mse: 1.4663, mae: 0.4400, huber: 0.2508, swd: 1.0945, ept: 190.5571
    Epoch [17/50], Val Losses: mse: 2.5833, mae: 0.6907, huber: 0.4588, swd: 1.9260, ept: 187.1938
    Epoch [17/50], Test Losses: mse: 2.4765, mae: 0.7078, huber: 0.4714, swd: 1.8193, ept: 185.6560
      Epoch 17 composite train-obj: 0.250799
            No improvement (0.4588), counter 2/5
    Epoch [18/50], Train Losses: mse: 1.3268, mae: 0.4270, huber: 0.2393, swd: 0.9653, ept: 190.8623
    Epoch [18/50], Val Losses: mse: 2.4166, mae: 0.6333, huber: 0.4090, swd: 1.8164, ept: 188.4217
    Epoch [18/50], Test Losses: mse: 2.2414, mae: 0.6378, huber: 0.4098, swd: 1.6808, ept: 187.1710
      Epoch 18 composite train-obj: 0.239261
            No improvement (0.4090), counter 3/5
    Epoch [19/50], Train Losses: mse: 1.3603, mae: 0.4352, huber: 0.2436, swd: 0.9981, ept: 190.8789
    Epoch [19/50], Val Losses: mse: 2.1551, mae: 0.6717, huber: 0.4126, swd: 1.6766, ept: 189.0918
    Epoch [19/50], Test Losses: mse: 2.0011, mae: 0.6811, huber: 0.4139, swd: 1.5580, ept: 188.1864
      Epoch 19 composite train-obj: 0.243635
            No improvement (0.4126), counter 4/5
    Epoch [20/50], Train Losses: mse: 1.3106, mae: 0.4209, huber: 0.2357, swd: 0.9548, ept: 190.9338
    Epoch [20/50], Val Losses: mse: 2.3734, mae: 0.7061, huber: 0.4492, swd: 1.7018, ept: 187.3892
    Epoch [20/50], Test Losses: mse: 2.2660, mae: 0.7211, huber: 0.4569, swd: 1.5969, ept: 185.5028
      Epoch 20 composite train-obj: 0.235675
    Epoch [20/50], Test Losses: mse: 1.3786, mae: 0.4161, huber: 0.2449, swd: 1.0651, ept: 189.5337
    Best round's Test MSE: 1.3786, MAE: 0.4161, SWD: 1.0651
    Best round's Validation MSE: 1.5595, MAE: 0.4292, SWD: 1.1939
    Best round's Test verification MSE : 1.3786, MAE: 0.4161, SWD: 1.0651
    Time taken: 171.15 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_rossler_seq336_pred196_20250513_1221)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 1.2231 ± 0.1147
      mae: 0.4485 ± 0.0339
      huber: 0.2507 ± 0.0123
      swd: 0.8994 ± 0.1188
      ept: 188.9314 ± 0.4425
      count: 37.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 1.3568 ± 0.1462
      mae: 0.4573 ± 0.0298
      huber: 0.2653 ± 0.0100
      swd: 0.9859 ± 0.1504
      ept: 190.0227 ± 0.6365
      count: 37.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 522.02 seconds
    
    Experiment complete: TimeMixer_rossler_seq336_pred196_20250513_1221
    Model: TimeMixer
    Dataset: rossler
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 280
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 7.0310, mae: 1.4236, huber: 1.0840, swd: 4.4729, ept: 266.3730
    Epoch [1/50], Val Losses: mse: 4.9421, mae: 0.9969, huber: 0.7090, swd: 3.7334, ept: 301.2304
    Epoch [1/50], Test Losses: mse: 4.2706, mae: 0.9412, huber: 0.6496, swd: 3.2658, ept: 302.0780
      Epoch 1 composite train-obj: 1.083972
            Val objective improved inf → 0.7090, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.1325, mae: 0.9174, huber: 0.6280, swd: 3.0858, ept: 304.7577
    Epoch [2/50], Val Losses: mse: 4.6937, mae: 0.9635, huber: 0.6757, swd: 3.5843, ept: 305.3199
    Epoch [2/50], Test Losses: mse: 4.0522, mae: 0.8998, huber: 0.6137, swd: 3.1356, ept: 304.5882
      Epoch 2 composite train-obj: 0.627959
            Val objective improved 0.7090 → 0.6757, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 3.8674, mae: 0.8435, huber: 0.5737, swd: 2.9485, ept: 307.7504
    Epoch [3/50], Val Losses: mse: 4.4038, mae: 0.8502, huber: 0.5964, swd: 3.4011, ept: 309.5179
    Epoch [3/50], Test Losses: mse: 3.7882, mae: 0.7973, huber: 0.5409, swd: 2.9593, ept: 307.1743
      Epoch 3 composite train-obj: 0.573722
            Val objective improved 0.6757 → 0.5964, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 3.7504, mae: 0.8044, huber: 0.5470, swd: 2.8715, ept: 309.3520
    Epoch [4/50], Val Losses: mse: 4.3468, mae: 0.8341, huber: 0.5897, swd: 3.3306, ept: 306.3713
    Epoch [4/50], Test Losses: mse: 3.7604, mae: 0.7891, huber: 0.5423, swd: 2.9159, ept: 305.6773
      Epoch 4 composite train-obj: 0.546962
            Val objective improved 0.5964 → 0.5897, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 3.6792, mae: 0.7862, huber: 0.5350, swd: 2.8134, ept: 310.0754
    Epoch [5/50], Val Losses: mse: 4.1910, mae: 0.7968, huber: 0.5626, swd: 3.2051, ept: 309.9312
    Epoch [5/50], Test Losses: mse: 3.6044, mae: 0.7505, huber: 0.5128, swd: 2.7945, ept: 308.3845
      Epoch 5 composite train-obj: 0.535012
            Val objective improved 0.5897 → 0.5626, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.5918, mae: 0.7663, huber: 0.5207, swd: 2.7358, ept: 310.9278
    Epoch [6/50], Val Losses: mse: 3.9583, mae: 0.7533, huber: 0.5360, swd: 3.1262, ept: 312.9865
    Epoch [6/50], Test Losses: mse: 3.4039, mae: 0.7067, huber: 0.4867, swd: 2.7046, ept: 310.5371
      Epoch 6 composite train-obj: 0.520724
            Val objective improved 0.5626 → 0.5360, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 3.5261, mae: 0.7592, huber: 0.5146, swd: 2.6729, ept: 311.3442
    Epoch [7/50], Val Losses: mse: 4.2592, mae: 0.8134, huber: 0.5714, swd: 3.2723, ept: 310.5357
    Epoch [7/50], Test Losses: mse: 3.6267, mae: 0.7570, huber: 0.5139, swd: 2.8293, ept: 308.1052
      Epoch 7 composite train-obj: 0.514594
            No improvement (0.5714), counter 1/5
    Epoch [8/50], Train Losses: mse: 3.4691, mae: 0.7436, huber: 0.5048, swd: 2.6276, ept: 311.3919
    Epoch [8/50], Val Losses: mse: 3.7949, mae: 0.7924, huber: 0.5500, swd: 2.8297, ept: 311.6168
    Epoch [8/50], Test Losses: mse: 3.3128, mae: 0.7503, huber: 0.5058, swd: 2.5095, ept: 309.3642
      Epoch 8 composite train-obj: 0.504766
            No improvement (0.5500), counter 2/5
    Epoch [9/50], Train Losses: mse: 3.2631, mae: 0.7252, huber: 0.4885, swd: 2.4207, ept: 312.3333
    Epoch [9/50], Val Losses: mse: 3.4587, mae: 0.7136, huber: 0.4982, swd: 2.5895, ept: 314.3223
    Epoch [9/50], Test Losses: mse: 2.9225, mae: 0.6549, huber: 0.4396, swd: 2.2161, ept: 312.3628
      Epoch 9 composite train-obj: 0.488530
            Val objective improved 0.5360 → 0.4982, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 3.0126, mae: 0.7121, huber: 0.4740, swd: 2.1375, ept: 312.8840
    Epoch [10/50], Val Losses: mse: 2.9246, mae: 0.6661, huber: 0.4647, swd: 1.7290, ept: 310.3278
    Epoch [10/50], Test Losses: mse: 2.4956, mae: 0.6178, huber: 0.4136, swd: 1.5223, ept: 310.4776
      Epoch 10 composite train-obj: 0.473960
            Val objective improved 0.4982 → 0.4647, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 2.6135, mae: 0.6790, huber: 0.4448, swd: 1.7335, ept: 313.4294
    Epoch [11/50], Val Losses: mse: 2.7220, mae: 0.6825, huber: 0.4533, swd: 1.7216, ept: 313.4970
    Epoch [11/50], Test Losses: mse: 2.2733, mae: 0.6235, huber: 0.3944, swd: 1.4619, ept: 313.0937
      Epoch 11 composite train-obj: 0.444838
            Val objective improved 0.4647 → 0.4533, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 2.4265, mae: 0.6520, huber: 0.4243, swd: 1.5819, ept: 314.3059
    Epoch [12/50], Val Losses: mse: 3.0091, mae: 0.7183, huber: 0.4868, swd: 1.9776, ept: 310.6968
    Epoch [12/50], Test Losses: mse: 2.5806, mae: 0.6726, huber: 0.4395, swd: 1.7256, ept: 308.8395
      Epoch 12 composite train-obj: 0.424280
            No improvement (0.4868), counter 1/5
    Epoch [13/50], Train Losses: mse: 2.2221, mae: 0.6308, huber: 0.4062, swd: 1.3998, ept: 315.1365
    Epoch [13/50], Val Losses: mse: 2.2404, mae: 0.6693, huber: 0.4322, swd: 1.3877, ept: 316.6508
    Epoch [13/50], Test Losses: mse: 1.8975, mae: 0.6167, huber: 0.3793, swd: 1.2045, ept: 315.0961
      Epoch 13 composite train-obj: 0.406179
            Val objective improved 0.4533 → 0.4322, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 2.0953, mae: 0.6234, huber: 0.3978, swd: 1.2831, ept: 315.7646
    Epoch [14/50], Val Losses: mse: 2.1027, mae: 0.6479, huber: 0.4193, swd: 1.2126, ept: 317.4834
    Epoch [14/50], Test Losses: mse: 1.8102, mae: 0.6131, huber: 0.3816, swd: 1.0686, ept: 316.7803
      Epoch 14 composite train-obj: 0.397803
            Val objective improved 0.4322 → 0.4193, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 2.1131, mae: 0.6222, huber: 0.3969, swd: 1.3340, ept: 316.0525
    Epoch [15/50], Val Losses: mse: 2.8647, mae: 0.6506, huber: 0.4481, swd: 1.9108, ept: 311.5524
    Epoch [15/50], Test Losses: mse: 2.4695, mae: 0.6065, huber: 0.4035, swd: 1.6914, ept: 315.1356
      Epoch 15 composite train-obj: 0.396916
            No improvement (0.4481), counter 1/5
    Epoch [16/50], Train Losses: mse: 1.9444, mae: 0.6014, huber: 0.3805, swd: 1.1799, ept: 316.8265
    Epoch [16/50], Val Losses: mse: 2.1047, mae: 0.6047, huber: 0.3963, swd: 1.2369, ept: 316.3362
    Epoch [16/50], Test Losses: mse: 1.7555, mae: 0.5590, huber: 0.3491, swd: 1.0527, ept: 314.4323
      Epoch 16 composite train-obj: 0.380458
            Val objective improved 0.4193 → 0.3963, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 1.8552, mae: 0.5934, huber: 0.3732, swd: 1.1092, ept: 317.6861
    Epoch [17/50], Val Losses: mse: 1.5740, mae: 0.5666, huber: 0.3644, swd: 0.7828, ept: 318.1479
    Epoch [17/50], Test Losses: mse: 1.3193, mae: 0.5243, huber: 0.3209, swd: 0.6691, ept: 316.7592
      Epoch 17 composite train-obj: 0.373154
            Val objective improved 0.3963 → 0.3644, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 1.7219, mae: 0.5814, huber: 0.3620, swd: 0.9872, ept: 318.1209
    Epoch [18/50], Val Losses: mse: 2.2267, mae: 0.6096, huber: 0.4073, swd: 1.4590, ept: 316.7035
    Epoch [18/50], Test Losses: mse: 1.8654, mae: 0.5625, huber: 0.3616, swd: 1.2387, ept: 314.6439
      Epoch 18 composite train-obj: 0.362046
            No improvement (0.4073), counter 1/5
    Epoch [19/50], Train Losses: mse: 1.6490, mae: 0.5697, huber: 0.3530, swd: 0.9374, ept: 318.6891
    Epoch [19/50], Val Losses: mse: 1.5747, mae: 0.5455, huber: 0.3492, swd: 0.7965, ept: 319.0965
    Epoch [19/50], Test Losses: mse: 1.2667, mae: 0.4911, huber: 0.2951, swd: 0.6500, ept: 320.0535
      Epoch 19 composite train-obj: 0.353030
            Val objective improved 0.3644 → 0.3492, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 1.5924, mae: 0.5602, huber: 0.3455, swd: 0.8840, ept: 319.4280
    Epoch [20/50], Val Losses: mse: 2.0107, mae: 0.6187, huber: 0.3953, swd: 1.2834, ept: 318.6568
    Epoch [20/50], Test Losses: mse: 2.5119, mae: 0.6157, huber: 0.3873, swd: 1.8437, ept: 315.4741
      Epoch 20 composite train-obj: 0.345491
            No improvement (0.3953), counter 1/5
    Epoch [21/50], Train Losses: mse: 1.5769, mae: 0.5540, huber: 0.3416, swd: 0.8782, ept: 319.8376
    Epoch [21/50], Val Losses: mse: 1.6920, mae: 0.5730, huber: 0.3648, swd: 0.9368, ept: 320.5920
    Epoch [21/50], Test Losses: mse: 1.3679, mae: 0.5184, huber: 0.3107, swd: 0.7739, ept: 322.1502
      Epoch 21 composite train-obj: 0.341649
            No improvement (0.3648), counter 2/5
    Epoch [22/50], Train Losses: mse: 1.6031, mae: 0.5534, huber: 0.3429, swd: 0.9103, ept: 319.8929
    Epoch [22/50], Val Losses: mse: 2.3656, mae: 0.6257, huber: 0.4211, swd: 1.3050, ept: 315.5933
    Epoch [22/50], Test Losses: mse: 1.9976, mae: 0.5916, huber: 0.3822, swd: 1.1254, ept: 318.6679
      Epoch 22 composite train-obj: 0.342850
            No improvement (0.4211), counter 3/5
    Epoch [23/50], Train Losses: mse: 1.4924, mae: 0.5384, huber: 0.3299, swd: 0.8026, ept: 320.9344
    Epoch [23/50], Val Losses: mse: 1.3953, mae: 0.5205, huber: 0.3259, swd: 0.6848, ept: 323.4856
    Epoch [23/50], Test Losses: mse: 1.3356, mae: 0.4836, huber: 0.2888, swd: 0.7354, ept: 324.8219
      Epoch 23 composite train-obj: 0.329884
            Val objective improved 0.3492 → 0.3259, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 1.4676, mae: 0.5362, huber: 0.3287, swd: 0.7854, ept: 320.9049
    Epoch [24/50], Val Losses: mse: 1.5867, mae: 0.5834, huber: 0.3659, swd: 0.8840, ept: 321.0376
    Epoch [24/50], Test Losses: mse: 1.2851, mae: 0.5315, huber: 0.3140, swd: 0.7323, ept: 319.5326
      Epoch 24 composite train-obj: 0.328747
            No improvement (0.3659), counter 1/5
    Epoch [25/50], Train Losses: mse: 1.3680, mae: 0.5257, huber: 0.3193, swd: 0.6897, ept: 322.1669
    Epoch [25/50], Val Losses: mse: 1.4849, mae: 0.5936, huber: 0.3690, swd: 0.7025, ept: 320.7019
    Epoch [25/50], Test Losses: mse: 1.2874, mae: 0.5528, huber: 0.3251, swd: 0.6580, ept: 321.0630
      Epoch 25 composite train-obj: 0.319348
            No improvement (0.3690), counter 2/5
    Epoch [26/50], Train Losses: mse: 1.4002, mae: 0.5278, huber: 0.3214, swd: 0.7257, ept: 321.9652
    Epoch [26/50], Val Losses: mse: 1.6746, mae: 0.5587, huber: 0.3630, swd: 0.9756, ept: 319.1650
    Epoch [26/50], Test Losses: mse: 1.4008, mae: 0.5208, huber: 0.3234, swd: 0.8257, ept: 318.0426
      Epoch 26 composite train-obj: 0.321394
            No improvement (0.3630), counter 3/5
    Epoch [27/50], Train Losses: mse: 1.5340, mae: 0.5515, huber: 0.3373, swd: 0.8420, ept: 321.0138
    Epoch [27/50], Val Losses: mse: 1.7587, mae: 0.5709, huber: 0.3649, swd: 0.9460, ept: 317.5295
    Epoch [27/50], Test Losses: mse: 1.4348, mae: 0.5124, huber: 0.3118, swd: 0.7999, ept: 317.0216
      Epoch 27 composite train-obj: 0.337308
            No improvement (0.3649), counter 4/5
    Epoch [28/50], Train Losses: mse: 1.3588, mae: 0.5193, huber: 0.3158, swd: 0.6890, ept: 322.4153
    Epoch [28/50], Val Losses: mse: 1.7947, mae: 0.5882, huber: 0.3724, swd: 0.9545, ept: 318.7767
    Epoch [28/50], Test Losses: mse: 1.4524, mae: 0.5281, huber: 0.3167, swd: 0.7904, ept: 318.2991
      Epoch 28 composite train-obj: 0.315842
    Epoch [28/50], Test Losses: mse: 1.3356, mae: 0.4836, huber: 0.2888, swd: 0.7354, ept: 324.8219
    Best round's Test MSE: 1.3356, MAE: 0.4836, SWD: 0.7354
    Best round's Validation MSE: 1.3953, MAE: 0.5205, SWD: 0.6848
    Best round's Test verification MSE : 1.3356, MAE: 0.4836, SWD: 0.7354
    Time taken: 248.49 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.4775, mae: 1.4916, huber: 1.1457, swd: 4.8396, ept: 261.4450
    Epoch [1/50], Val Losses: mse: 5.1658, mae: 1.0450, huber: 0.7429, swd: 3.8829, ept: 296.9128
    Epoch [1/50], Test Losses: mse: 4.4773, mae: 1.0028, huber: 0.6939, swd: 3.3873, ept: 299.3662
      Epoch 1 composite train-obj: 1.145741
            Val objective improved inf → 0.7429, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.1789, mae: 0.9372, huber: 0.6436, swd: 3.1393, ept: 304.4379
    Epoch [2/50], Val Losses: mse: 4.5278, mae: 0.8738, huber: 0.6248, swd: 3.5225, ept: 306.6968
    Epoch [2/50], Test Losses: mse: 3.8710, mae: 0.8088, huber: 0.5580, swd: 3.0603, ept: 305.8357
      Epoch 2 composite train-obj: 0.643650
            Val objective improved 0.7429 → 0.6248, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 3.8447, mae: 0.8475, huber: 0.5756, swd: 2.9686, ept: 307.7521
    Epoch [3/50], Val Losses: mse: 4.8249, mae: 0.8753, huber: 0.6287, swd: 3.5948, ept: 301.5543
    Epoch [3/50], Test Losses: mse: 4.1742, mae: 0.8353, huber: 0.5851, swd: 3.1440, ept: 302.2873
      Epoch 3 composite train-obj: 0.575618
            No improvement (0.6287), counter 1/5
    Epoch [4/50], Train Losses: mse: 3.7501, mae: 0.8151, huber: 0.5537, swd: 2.9096, ept: 309.0840
    Epoch [4/50], Val Losses: mse: 4.3445, mae: 0.7917, huber: 0.5662, swd: 3.3530, ept: 306.3759
    Epoch [4/50], Test Losses: mse: 3.7263, mae: 0.7370, huber: 0.5127, swd: 2.9106, ept: 304.9255
      Epoch 4 composite train-obj: 0.553694
            Val objective improved 0.6248 → 0.5662, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 3.6484, mae: 0.7872, huber: 0.5335, swd: 2.8306, ept: 309.8696
    Epoch [5/50], Val Losses: mse: 4.6026, mae: 0.8438, huber: 0.6098, swd: 3.5217, ept: 304.5280
    Epoch [5/50], Test Losses: mse: 4.0465, mae: 0.8113, huber: 0.5729, swd: 3.1230, ept: 302.8472
      Epoch 5 composite train-obj: 0.533532
            No improvement (0.6098), counter 1/5
    Epoch [6/50], Train Losses: mse: 3.5844, mae: 0.7711, huber: 0.5225, swd: 2.7723, ept: 310.3098
    Epoch [6/50], Val Losses: mse: 4.1012, mae: 0.8225, huber: 0.5738, swd: 3.1850, ept: 309.9519
    Epoch [6/50], Test Losses: mse: 3.5585, mae: 0.7813, huber: 0.5300, swd: 2.7845, ept: 308.9701
      Epoch 6 composite train-obj: 0.522461
            No improvement (0.5738), counter 2/5
    Epoch [7/50], Train Losses: mse: 3.4014, mae: 0.7492, huber: 0.5046, swd: 2.5734, ept: 311.0462
    Epoch [7/50], Val Losses: mse: 3.6673, mae: 0.7747, huber: 0.5297, swd: 2.6513, ept: 310.2993
    Epoch [7/50], Test Losses: mse: 3.1806, mae: 0.7267, huber: 0.4808, swd: 2.3301, ept: 308.3727
      Epoch 7 composite train-obj: 0.504563
            Val objective improved 0.5662 → 0.5297, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.2036, mae: 0.7353, huber: 0.4917, swd: 2.3234, ept: 311.1658
    Epoch [8/50], Val Losses: mse: 3.7218, mae: 0.7666, huber: 0.5287, swd: 2.8684, ept: 311.4192
    Epoch [8/50], Test Losses: mse: 3.2280, mae: 0.7262, huber: 0.4865, swd: 2.5171, ept: 309.0670
      Epoch 8 composite train-obj: 0.491652
            Val objective improved 0.5297 → 0.5287, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.1576, mae: 0.7236, huber: 0.4840, swd: 2.3048, ept: 311.2899
    Epoch [9/50], Val Losses: mse: 3.4604, mae: 0.7781, huber: 0.5253, swd: 2.6277, ept: 312.8298
    Epoch [9/50], Test Losses: mse: 2.9911, mae: 0.7290, huber: 0.4744, swd: 2.3078, ept: 310.5561
      Epoch 9 composite train-obj: 0.484047
            Val objective improved 0.5287 → 0.5253, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 2.9614, mae: 0.6989, huber: 0.4639, swd: 2.1187, ept: 312.3787
    Epoch [10/50], Val Losses: mse: 3.3953, mae: 0.7790, huber: 0.5302, swd: 2.4265, ept: 311.9001
    Epoch [10/50], Test Losses: mse: 2.9057, mae: 0.7221, huber: 0.4737, swd: 2.1144, ept: 309.2528
      Epoch 10 composite train-obj: 0.463901
            No improvement (0.5302), counter 1/5
    Epoch [11/50], Train Losses: mse: 2.7194, mae: 0.6797, huber: 0.4459, swd: 1.8486, ept: 312.6514
    Epoch [11/50], Val Losses: mse: 3.4196, mae: 0.7711, huber: 0.5210, swd: 2.4683, ept: 313.2259
    Epoch [11/50], Test Losses: mse: 2.9243, mae: 0.7222, huber: 0.4711, swd: 2.1444, ept: 310.7026
      Epoch 11 composite train-obj: 0.445879
            Val objective improved 0.5253 → 0.5210, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 2.5096, mae: 0.6679, huber: 0.4331, swd: 1.6687, ept: 313.4890
    Epoch [12/50], Val Losses: mse: 2.7373, mae: 0.7278, huber: 0.4935, swd: 1.6724, ept: 314.3111
    Epoch [12/50], Test Losses: mse: 2.3789, mae: 0.6840, huber: 0.4497, swd: 1.4527, ept: 313.2492
      Epoch 12 composite train-obj: 0.433147
            Val objective improved 0.5210 → 0.4935, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 2.3639, mae: 0.6490, huber: 0.4185, swd: 1.5192, ept: 313.8555
    Epoch [13/50], Val Losses: mse: 2.4961, mae: 0.6949, huber: 0.4497, swd: 1.5405, ept: 313.9840
    Epoch [13/50], Test Losses: mse: 2.1303, mae: 0.6469, huber: 0.4030, swd: 1.3244, ept: 311.3884
      Epoch 13 composite train-obj: 0.418513
            Val objective improved 0.4935 → 0.4497, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 2.1499, mae: 0.6279, huber: 0.4005, swd: 1.3198, ept: 315.0295
    Epoch [14/50], Val Losses: mse: 3.1571, mae: 0.7247, huber: 0.4875, swd: 2.1956, ept: 311.9878
    Epoch [14/50], Test Losses: mse: 2.7270, mae: 0.6861, huber: 0.4466, swd: 1.9302, ept: 308.6876
      Epoch 14 composite train-obj: 0.400500
            No improvement (0.4875), counter 1/5
    Epoch [15/50], Train Losses: mse: 2.0877, mae: 0.6142, huber: 0.3915, swd: 1.2831, ept: 315.3984
    Epoch [15/50], Val Losses: mse: 2.5643, mae: 0.6796, huber: 0.4463, swd: 1.7356, ept: 316.3670
    Epoch [15/50], Test Losses: mse: 2.1412, mae: 0.6307, huber: 0.3964, swd: 1.4538, ept: 313.5905
      Epoch 15 composite train-obj: 0.391477
            Val objective improved 0.4497 → 0.4463, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 2.2862, mae: 0.6307, huber: 0.4045, swd: 1.5133, ept: 315.3233
    Epoch [16/50], Val Losses: mse: 2.5057, mae: 0.6667, huber: 0.4443, swd: 1.7422, ept: 317.0226
    Epoch [16/50], Test Losses: mse: 2.1394, mae: 0.6256, huber: 0.4012, swd: 1.5005, ept: 315.1306
      Epoch 16 composite train-obj: 0.404522
            Val objective improved 0.4463 → 0.4443, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 1.9866, mae: 0.6032, huber: 0.3815, swd: 1.2179, ept: 316.0041
    Epoch [17/50], Val Losses: mse: 2.1398, mae: 0.7532, huber: 0.4898, swd: 1.2346, ept: 315.7588
    Epoch [17/50], Test Losses: mse: 1.9120, mae: 0.7208, huber: 0.4557, swd: 1.1197, ept: 313.6356
      Epoch 17 composite train-obj: 0.381451
            No improvement (0.4898), counter 1/5
    Epoch [18/50], Train Losses: mse: 1.7893, mae: 0.5830, huber: 0.3642, swd: 1.0499, ept: 317.1456
    Epoch [18/50], Val Losses: mse: 2.0814, mae: 0.6533, huber: 0.4139, swd: 1.2923, ept: 319.6997
    Epoch [18/50], Test Losses: mse: 1.7402, mae: 0.6078, huber: 0.3663, swd: 1.0884, ept: 318.9678
      Epoch 18 composite train-obj: 0.364196
            Val objective improved 0.4443 → 0.4139, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 1.8661, mae: 0.5783, huber: 0.3635, swd: 1.1458, ept: 317.0933
    Epoch [19/50], Val Losses: mse: 2.2352, mae: 0.6765, huber: 0.4437, swd: 1.3703, ept: 314.5908
    Epoch [19/50], Test Losses: mse: 1.9195, mae: 0.6410, huber: 0.4041, swd: 1.1832, ept: 314.6425
      Epoch 19 composite train-obj: 0.363511
            No improvement (0.4437), counter 1/5
    Epoch [20/50], Train Losses: mse: 1.7302, mae: 0.5689, huber: 0.3540, swd: 1.0144, ept: 317.8970
    Epoch [20/50], Val Losses: mse: 2.5992, mae: 0.8357, huber: 0.5598, swd: 1.5694, ept: 315.1407
    Epoch [20/50], Test Losses: mse: 2.4072, mae: 0.8177, huber: 0.5397, swd: 1.4432, ept: 310.1888
      Epoch 20 composite train-obj: 0.354033
            No improvement (0.5598), counter 2/5
    Epoch [21/50], Train Losses: mse: 1.6723, mae: 0.5644, huber: 0.3501, swd: 0.9696, ept: 317.9909
    Epoch [21/50], Val Losses: mse: 2.3540, mae: 0.7535, huber: 0.4819, swd: 1.5243, ept: 317.3934
    Epoch [21/50], Test Losses: mse: 2.0493, mae: 0.7223, huber: 0.4487, swd: 1.3446, ept: 315.7149
      Epoch 21 composite train-obj: 0.350052
            No improvement (0.4819), counter 3/5
    Epoch [22/50], Train Losses: mse: 1.6765, mae: 0.5515, huber: 0.3428, swd: 0.9868, ept: 318.4508
    Epoch [22/50], Val Losses: mse: 2.6976, mae: 0.7925, huber: 0.5244, swd: 1.7735, ept: 313.9855
    Epoch [22/50], Test Losses: mse: 2.3832, mae: 0.7682, huber: 0.4938, swd: 1.5826, ept: 312.9664
      Epoch 22 composite train-obj: 0.342802
            No improvement (0.5244), counter 4/5
    Epoch [23/50], Train Losses: mse: 1.7529, mae: 0.5597, huber: 0.3483, swd: 1.0622, ept: 319.1399
    Epoch [23/50], Val Losses: mse: 2.7773, mae: 0.7720, huber: 0.5031, swd: 1.9740, ept: 318.3772
    Epoch [23/50], Test Losses: mse: 2.3657, mae: 0.7249, huber: 0.4534, swd: 1.7155, ept: 315.1486
      Epoch 23 composite train-obj: 0.348313
    Epoch [23/50], Test Losses: mse: 1.7402, mae: 0.6078, huber: 0.3663, swd: 1.0884, ept: 318.9678
    Best round's Test MSE: 1.7402, MAE: 0.6078, SWD: 1.0884
    Best round's Validation MSE: 2.0814, MAE: 0.6533, SWD: 1.2923
    Best round's Test verification MSE : 1.7402, MAE: 0.6078, SWD: 1.0884
    Time taken: 206.25 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.0467, mae: 1.4168, huber: 1.0777, swd: 4.5860, ept: 268.2124
    Epoch [1/50], Val Losses: mse: 5.0783, mae: 1.0398, huber: 0.7437, swd: 3.8161, ept: 298.9943
    Epoch [1/50], Test Losses: mse: 4.4026, mae: 0.9877, huber: 0.6883, swd: 3.3379, ept: 300.7329
      Epoch 1 composite train-obj: 1.077664
            Val objective improved inf → 0.7437, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.1628, mae: 0.9272, huber: 0.6359, swd: 3.1352, ept: 304.4611
    Epoch [2/50], Val Losses: mse: 4.5146, mae: 0.9550, huber: 0.6700, swd: 3.5708, ept: 310.2663
    Epoch [2/50], Test Losses: mse: 3.9046, mae: 0.8962, huber: 0.6098, swd: 3.1310, ept: 308.7397
      Epoch 2 composite train-obj: 0.635922
            Val objective improved 0.7437 → 0.6700, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 3.8649, mae: 0.8519, huber: 0.5801, swd: 2.9759, ept: 308.0189
    Epoch [3/50], Val Losses: mse: 4.3837, mae: 0.8728, huber: 0.6134, swd: 3.4318, ept: 307.8630
    Epoch [3/50], Test Losses: mse: 3.7576, mae: 0.8173, huber: 0.5536, swd: 2.9773, ept: 306.8788
      Epoch 3 composite train-obj: 0.580070
            Val objective improved 0.6700 → 0.6134, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 3.7513, mae: 0.8132, huber: 0.5539, swd: 2.9097, ept: 309.2966
    Epoch [4/50], Val Losses: mse: 4.3006, mae: 0.9095, huber: 0.6296, swd: 3.4122, ept: 310.4390
    Epoch [4/50], Test Losses: mse: 3.7415, mae: 0.8663, huber: 0.5824, swd: 3.0027, ept: 307.1978
      Epoch 4 composite train-obj: 0.553907
            No improvement (0.6296), counter 1/5
    Epoch [5/50], Train Losses: mse: 3.6742, mae: 0.7885, huber: 0.5374, swd: 2.8610, ept: 310.0299
    Epoch [5/50], Val Losses: mse: 4.2337, mae: 0.8930, huber: 0.6123, swd: 3.3562, ept: 310.7405
    Epoch [5/50], Test Losses: mse: 3.6527, mae: 0.8387, huber: 0.5571, swd: 2.9308, ept: 308.4931
      Epoch 5 composite train-obj: 0.537408
            Val objective improved 0.6134 → 0.6123, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 3.5789, mae: 0.7659, huber: 0.5217, swd: 2.7883, ept: 310.7877
    Epoch [6/50], Val Losses: mse: 4.2316, mae: 0.9099, huber: 0.6300, swd: 3.3571, ept: 312.5242
    Epoch [6/50], Test Losses: mse: 3.6657, mae: 0.8586, huber: 0.5767, swd: 2.9507, ept: 308.9370
      Epoch 6 composite train-obj: 0.521686
            No improvement (0.6300), counter 1/5
    Epoch [7/50], Train Losses: mse: 3.5457, mae: 0.7594, huber: 0.5168, swd: 2.7548, ept: 311.0445
    Epoch [7/50], Val Losses: mse: 4.1181, mae: 0.8291, huber: 0.5853, swd: 3.2893, ept: 312.8692
    Epoch [7/50], Test Losses: mse: 3.5247, mae: 0.7736, huber: 0.5260, swd: 2.8471, ept: 309.8485
      Epoch 7 composite train-obj: 0.516812
            Val objective improved 0.6123 → 0.5853, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.4704, mae: 0.7460, huber: 0.5071, swd: 2.6851, ept: 311.6114
    Epoch [8/50], Val Losses: mse: 3.9156, mae: 0.7727, huber: 0.5503, swd: 3.1254, ept: 312.4739
    Epoch [8/50], Test Losses: mse: 3.3601, mae: 0.7187, huber: 0.4955, swd: 2.7068, ept: 310.8413
      Epoch 8 composite train-obj: 0.507099
            Val objective improved 0.5853 → 0.5503, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.3480, mae: 0.7305, huber: 0.4947, swd: 2.5662, ept: 312.4136
    Epoch [9/50], Val Losses: mse: 4.0522, mae: 0.7809, huber: 0.5540, swd: 3.0843, ept: 307.7823
    Epoch [9/50], Test Losses: mse: 3.4399, mae: 0.7265, huber: 0.4973, swd: 2.6530, ept: 306.9700
      Epoch 9 composite train-obj: 0.494734
            No improvement (0.5540), counter 1/5
    Epoch [10/50], Train Losses: mse: 3.2732, mae: 0.7288, huber: 0.4917, swd: 2.4751, ept: 312.1968
    Epoch [10/50], Val Losses: mse: 3.8626, mae: 0.9025, huber: 0.6168, swd: 2.8258, ept: 312.7878
    Epoch [10/50], Test Losses: mse: 3.3740, mae: 0.8586, huber: 0.5724, swd: 2.4973, ept: 310.1411
      Epoch 10 composite train-obj: 0.491699
            No improvement (0.6168), counter 2/5
    Epoch [11/50], Train Losses: mse: 2.9497, mae: 0.7063, huber: 0.4694, swd: 2.1243, ept: 313.0726
    Epoch [11/50], Val Losses: mse: 3.0338, mae: 0.7561, huber: 0.5113, swd: 2.1945, ept: 315.7870
    Epoch [11/50], Test Losses: mse: 2.6137, mae: 0.7179, huber: 0.4689, swd: 1.9073, ept: 314.4490
      Epoch 11 composite train-obj: 0.469375
            Val objective improved 0.5503 → 0.5113, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 2.5397, mae: 0.6696, huber: 0.4380, swd: 1.6936, ept: 313.9902
    Epoch [12/50], Val Losses: mse: 2.6182, mae: 0.7261, huber: 0.4740, swd: 1.6404, ept: 314.7427
    Epoch [12/50], Test Losses: mse: 2.2178, mae: 0.6687, huber: 0.4172, swd: 1.4179, ept: 316.1035
      Epoch 12 composite train-obj: 0.438000
            Val objective improved 0.5113 → 0.4740, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 2.2996, mae: 0.6486, huber: 0.4183, swd: 1.4755, ept: 314.6565
    Epoch [13/50], Val Losses: mse: 2.5410, mae: 0.7966, huber: 0.5269, swd: 1.6260, ept: 317.8240
    Epoch [13/50], Test Losses: mse: 2.2253, mae: 0.7602, huber: 0.4876, swd: 1.4491, ept: 315.6389
      Epoch 13 composite train-obj: 0.418265
            No improvement (0.5269), counter 1/5
    Epoch [14/50], Train Losses: mse: 2.2264, mae: 0.6421, huber: 0.4115, swd: 1.4497, ept: 315.4778
    Epoch [14/50], Val Losses: mse: 2.4935, mae: 0.7462, huber: 0.4834, swd: 1.6286, ept: 316.5972
    Epoch [14/50], Test Losses: mse: 2.1503, mae: 0.7077, huber: 0.4409, swd: 1.4387, ept: 315.0144
      Epoch 14 composite train-obj: 0.411483
            No improvement (0.4834), counter 2/5
    Epoch [15/50], Train Losses: mse: 2.2185, mae: 0.6336, huber: 0.4062, swd: 1.4724, ept: 315.7582
    Epoch [15/50], Val Losses: mse: 3.1285, mae: 0.7821, huber: 0.5352, swd: 2.1969, ept: 310.5951
    Epoch [15/50], Test Losses: mse: 2.7621, mae: 0.7476, huber: 0.4991, swd: 1.9729, ept: 308.5521
      Epoch 15 composite train-obj: 0.406225
            No improvement (0.5352), counter 3/5
    Epoch [16/50], Train Losses: mse: 2.1389, mae: 0.6166, huber: 0.3941, swd: 1.4129, ept: 316.3804
    Epoch [16/50], Val Losses: mse: 2.5359, mae: 0.8320, huber: 0.5556, swd: 1.4181, ept: 315.2970
    Epoch [16/50], Test Losses: mse: 2.3607, mae: 0.8071, huber: 0.5291, swd: 1.3658, ept: 313.3182
      Epoch 16 composite train-obj: 0.394131
            No improvement (0.5556), counter 4/5
    Epoch [17/50], Train Losses: mse: 1.8254, mae: 0.5914, huber: 0.3702, swd: 1.1079, ept: 317.5545
    Epoch [17/50], Val Losses: mse: 2.0073, mae: 0.6961, huber: 0.4434, swd: 1.1848, ept: 316.0749
    Epoch [17/50], Test Losses: mse: 1.7244, mae: 0.6564, huber: 0.4016, swd: 1.0544, ept: 314.0571
      Epoch 17 composite train-obj: 0.370220
            Val objective improved 0.4740 → 0.4434, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 2.0879, mae: 0.6093, huber: 0.3873, swd: 1.3922, ept: 317.0980
    Epoch [18/50], Val Losses: mse: 2.7434, mae: 0.7371, huber: 0.4990, swd: 1.7671, ept: 312.6839
    Epoch [18/50], Test Losses: mse: 2.3543, mae: 0.6944, huber: 0.4555, swd: 1.5451, ept: 311.0910
      Epoch 18 composite train-obj: 0.387306
            No improvement (0.4990), counter 1/5
    Epoch [19/50], Train Losses: mse: 1.6986, mae: 0.5774, huber: 0.3587, swd: 1.0007, ept: 318.4961
    Epoch [19/50], Val Losses: mse: 2.5332, mae: 0.7856, huber: 0.5317, swd: 1.4343, ept: 310.8239
    Epoch [19/50], Test Losses: mse: 2.2947, mae: 0.7564, huber: 0.5017, swd: 1.3194, ept: 309.5621
      Epoch 19 composite train-obj: 0.358711
            No improvement (0.5317), counter 2/5
    Epoch [20/50], Train Losses: mse: 1.7910, mae: 0.5829, huber: 0.3630, swd: 1.1076, ept: 318.6922
    Epoch [20/50], Val Losses: mse: 2.3586, mae: 0.7303, huber: 0.4842, swd: 1.4977, ept: 314.3322
    Epoch [20/50], Test Losses: mse: 2.0921, mae: 0.6988, huber: 0.4512, swd: 1.3557, ept: 313.3238
      Epoch 20 composite train-obj: 0.363002
            No improvement (0.4842), counter 3/5
    Epoch [21/50], Train Losses: mse: 1.6590, mae: 0.5616, huber: 0.3483, swd: 0.9831, ept: 319.3627
    Epoch [21/50], Val Losses: mse: 2.5259, mae: 0.7812, huber: 0.5230, swd: 1.5696, ept: 313.6580
    Epoch [21/50], Test Losses: mse: 2.2549, mae: 0.7531, huber: 0.4925, swd: 1.4101, ept: 311.2671
      Epoch 21 composite train-obj: 0.348308
            No improvement (0.5230), counter 4/5
    Epoch [22/50], Train Losses: mse: 1.5933, mae: 0.5569, huber: 0.3426, swd: 0.9288, ept: 319.9993
    Epoch [22/50], Val Losses: mse: 2.0515, mae: 0.7339, huber: 0.4714, swd: 1.1402, ept: 315.6204
    Epoch [22/50], Test Losses: mse: 1.7825, mae: 0.6943, huber: 0.4307, swd: 1.0181, ept: 314.7706
      Epoch 22 composite train-obj: 0.342581
    Epoch [22/50], Test Losses: mse: 1.7244, mae: 0.6564, huber: 0.4016, swd: 1.0544, ept: 314.0571
    Best round's Test MSE: 1.7244, MAE: 0.6564, SWD: 1.0544
    Best round's Validation MSE: 2.0073, MAE: 0.6961, SWD: 1.1848
    Best round's Test verification MSE : 1.7244, MAE: 0.6564, SWD: 1.0544
    Time taken: 197.51 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_rossler_seq336_pred336_20250511_0336)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 1.6001 ± 0.1871
      mae: 0.5826 ± 0.0727
      huber: 0.3522 ± 0.0471
      swd: 0.9594 ± 0.1590
      ept: 319.2823 ± 4.4003
      count: 36.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 1.8280 ± 0.3075
      mae: 0.6233 ± 0.0748
      huber: 0.3944 ± 0.0499
      swd: 1.0539 ± 0.2647
      ept: 319.7534 ± 3.0257
      count: 36.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 652.35 seconds
    
    Experiment complete: TimeMixer_rossler_seq336_pred336_20250511_0336
    Model: TimeMixer
    Dataset: rossler
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatTimeMixerConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 277
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 8.8553, mae: 1.7670, huber: 1.4100, swd: 4.9307, ept: 466.6849
    Epoch [1/50], Val Losses: mse: 7.6605, mae: 1.4798, huber: 1.1607, swd: 4.8330, ept: 539.2213
    Epoch [1/50], Test Losses: mse: 6.0232, mae: 1.2865, huber: 0.9683, swd: 3.8367, ept: 558.9928
      Epoch 1 composite train-obj: 1.410010
            Val objective improved inf → 1.1607, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.2711, mae: 1.3465, huber: 1.0217, swd: 3.9927, ept: 559.7175
    Epoch [2/50], Val Losses: mse: 7.0264, mae: 1.3414, huber: 1.0385, swd: 4.6700, ept: 560.4366
    Epoch [2/50], Test Losses: mse: 5.5199, mae: 1.1737, huber: 0.8689, swd: 3.6951, ept: 576.3042
      Epoch 2 composite train-obj: 1.021704
            Val objective improved 1.1607 → 1.0385, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.8504, mae: 1.2382, huber: 0.9302, swd: 3.8534, ept: 574.9558
    Epoch [3/50], Val Losses: mse: 6.7709, mae: 1.2740, huber: 0.9786, swd: 4.5353, ept: 571.8776
    Epoch [3/50], Test Losses: mse: 5.3654, mae: 1.1217, huber: 0.8264, swd: 3.6261, ept: 583.2699
      Epoch 3 composite train-obj: 0.930212
            Val objective improved 1.0385 → 0.9786, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.6089, mae: 1.1742, huber: 0.8771, swd: 3.7658, ept: 585.6614
    Epoch [4/50], Val Losses: mse: 7.4880, mae: 1.3441, huber: 1.0385, swd: 4.9337, ept: 576.4433
    Epoch [4/50], Test Losses: mse: 5.7736, mae: 1.1834, huber: 0.8759, swd: 3.8559, ept: 588.7724
      Epoch 4 composite train-obj: 0.877115
            No improvement (1.0385), counter 1/5
    Epoch [5/50], Train Losses: mse: 5.5024, mae: 1.1446, huber: 0.8528, swd: 3.7150, ept: 589.6101
    Epoch [5/50], Val Losses: mse: 6.3829, mae: 1.1932, huber: 0.9139, swd: 4.4337, ept: 579.2663
    Epoch [5/50], Test Losses: mse: 5.1604, mae: 1.0760, huber: 0.7938, swd: 3.6057, ept: 587.3349
      Epoch 5 composite train-obj: 0.852799
            Val objective improved 0.9786 → 0.9139, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.3541, mae: 1.1056, huber: 0.8221, swd: 3.6135, ept: 592.7200
    Epoch [6/50], Val Losses: mse: 6.0595, mae: 1.1449, huber: 0.8716, swd: 4.2123, ept: 586.4363
    Epoch [6/50], Test Losses: mse: 4.7942, mae: 0.9998, huber: 0.7283, swd: 3.3798, ept: 593.0031
      Epoch 6 composite train-obj: 0.822117
            Val objective improved 0.9139 → 0.8716, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.2341, mae: 1.0881, huber: 0.8071, swd: 3.4994, ept: 595.1123
    Epoch [7/50], Val Losses: mse: 5.7156, mae: 1.0950, huber: 0.8307, swd: 3.9415, ept: 590.0899
    Epoch [7/50], Test Losses: mse: 4.4890, mae: 0.9508, huber: 0.6902, swd: 3.1290, ept: 600.6704
      Epoch 7 composite train-obj: 0.807122
            Val objective improved 0.8716 → 0.8307, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.0464, mae: 1.0702, huber: 0.7910, swd: 3.2929, ept: 597.1356
    Epoch [8/50], Val Losses: mse: 6.1132, mae: 1.2291, huber: 0.9424, swd: 3.9471, ept: 579.1342
    Epoch [8/50], Test Losses: mse: 4.9997, mae: 1.0978, huber: 0.8123, swd: 3.2793, ept: 580.1763
      Epoch 8 composite train-obj: 0.790996
            No improvement (0.9424), counter 1/5
    Epoch [9/50], Train Losses: mse: 4.8005, mae: 1.0506, huber: 0.7728, swd: 3.0393, ept: 598.1826
    Epoch [9/50], Val Losses: mse: 5.1399, mae: 1.0458, huber: 0.7840, swd: 3.3090, ept: 597.1937
    Epoch [9/50], Test Losses: mse: 4.0134, mae: 0.9039, huber: 0.6472, swd: 2.6117, ept: 603.6974
      Epoch 9 composite train-obj: 0.772767
            Val objective improved 0.8307 → 0.7840, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 4.5781, mae: 1.0275, huber: 0.7512, swd: 2.8203, ept: 600.7766
    Epoch [10/50], Val Losses: mse: 5.3633, mae: 1.1589, huber: 0.8579, swd: 3.3878, ept: 595.2727
    Epoch [10/50], Test Losses: mse: 4.4597, mae: 1.0709, huber: 0.7663, swd: 2.7606, ept: 592.9530
      Epoch 10 composite train-obj: 0.751174
            No improvement (0.8579), counter 1/5
    Epoch [11/50], Train Losses: mse: 4.3353, mae: 0.9968, huber: 0.7241, swd: 2.6151, ept: 604.0126
    Epoch [11/50], Val Losses: mse: 5.2292, mae: 1.1103, huber: 0.8309, swd: 3.1733, ept: 588.5895
    Epoch [11/50], Test Losses: mse: 3.9701, mae: 0.9438, huber: 0.6713, swd: 2.4457, ept: 601.8907
      Epoch 11 composite train-obj: 0.724077
            No improvement (0.8309), counter 2/5
    Epoch [12/50], Train Losses: mse: 4.2648, mae: 0.9964, huber: 0.7233, swd: 2.5089, ept: 604.7179
    Epoch [12/50], Val Losses: mse: 4.6921, mae: 1.0554, huber: 0.7755, swd: 2.9800, ept: 603.6570
    Epoch [12/50], Test Losses: mse: 3.6769, mae: 0.9260, huber: 0.6484, swd: 2.3379, ept: 605.8739
      Epoch 12 composite train-obj: 0.723288
            Val objective improved 0.7840 → 0.7755, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 4.0556, mae: 0.9706, huber: 0.7001, swd: 2.3548, ept: 607.7498
    Epoch [13/50], Val Losses: mse: 4.3645, mae: 0.9671, huber: 0.7109, swd: 2.7375, ept: 603.5491
    Epoch [13/50], Test Losses: mse: 3.3905, mae: 0.8440, huber: 0.5914, swd: 2.1057, ept: 607.1643
      Epoch 13 composite train-obj: 0.700081
            Val objective improved 0.7755 → 0.7109, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 3.8965, mae: 0.9543, huber: 0.6860, swd: 2.2083, ept: 609.9316
    Epoch [14/50], Val Losses: mse: 4.1209, mae: 0.9844, huber: 0.7176, swd: 2.5167, ept: 606.9812
    Epoch [14/50], Test Losses: mse: 3.1219, mae: 0.8460, huber: 0.5832, swd: 1.9056, ept: 619.3887
      Epoch 14 composite train-obj: 0.686024
            No improvement (0.7176), counter 1/5
    Epoch [15/50], Train Losses: mse: 3.7445, mae: 0.9446, huber: 0.6744, swd: 2.0676, ept: 611.9450
    Epoch [15/50], Val Losses: mse: 4.4036, mae: 1.0615, huber: 0.7812, swd: 2.4261, ept: 599.8134
    Epoch [15/50], Test Losses: mse: 3.3722, mae: 0.9177, huber: 0.6407, swd: 1.8374, ept: 605.5005
      Epoch 15 composite train-obj: 0.674436
            No improvement (0.7812), counter 2/5
    Epoch [16/50], Train Losses: mse: 3.6622, mae: 0.9361, huber: 0.6675, swd: 1.9829, ept: 612.7776
    Epoch [16/50], Val Losses: mse: 4.1604, mae: 1.0433, huber: 0.7660, swd: 2.3230, ept: 601.6601
    Epoch [16/50], Test Losses: mse: 3.2312, mae: 0.9166, huber: 0.6439, swd: 1.7651, ept: 605.9563
      Epoch 16 composite train-obj: 0.667503
            No improvement (0.7660), counter 3/5
    Epoch [17/50], Train Losses: mse: 3.4979, mae: 0.9120, huber: 0.6479, swd: 1.8531, ept: 615.7741
    Epoch [17/50], Val Losses: mse: 3.7866, mae: 0.9912, huber: 0.7211, swd: 2.2191, ept: 609.7916
    Epoch [17/50], Test Losses: mse: 2.8731, mae: 0.8644, huber: 0.5959, swd: 1.6746, ept: 620.2073
      Epoch 17 composite train-obj: 0.647880
            No improvement (0.7211), counter 4/5
    Epoch [18/50], Train Losses: mse: 3.3812, mae: 0.9023, huber: 0.6387, swd: 1.7658, ept: 617.7505
    Epoch [18/50], Val Losses: mse: 3.6148, mae: 0.9767, huber: 0.6943, swd: 1.9998, ept: 608.0012
    Epoch [18/50], Test Losses: mse: 2.7873, mae: 0.8596, huber: 0.5788, swd: 1.4965, ept: 616.7341
      Epoch 18 composite train-obj: 0.638729
            Val objective improved 0.7109 → 0.6943, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 3.3615, mae: 0.9003, huber: 0.6360, swd: 1.7587, ept: 619.0420
    Epoch [19/50], Val Losses: mse: 4.3105, mae: 1.0767, huber: 0.7969, swd: 2.3844, ept: 599.1772
    Epoch [19/50], Test Losses: mse: 3.3922, mae: 0.9466, huber: 0.6685, swd: 1.8425, ept: 604.2455
      Epoch 19 composite train-obj: 0.635967
            No improvement (0.7969), counter 1/5
    Epoch [20/50], Train Losses: mse: 3.2003, mae: 0.8731, huber: 0.6150, swd: 1.6248, ept: 622.5624
    Epoch [20/50], Val Losses: mse: 3.5337, mae: 0.9889, huber: 0.6984, swd: 1.8047, ept: 623.7931
    Epoch [20/50], Test Losses: mse: 2.8364, mae: 0.8879, huber: 0.5967, swd: 1.4138, ept: 626.6323
      Epoch 20 composite train-obj: 0.614995
            No improvement (0.6984), counter 2/5
    Epoch [21/50], Train Losses: mse: 3.1942, mae: 0.8766, huber: 0.6168, swd: 1.6080, ept: 623.0483
    Epoch [21/50], Val Losses: mse: 3.2814, mae: 0.9193, huber: 0.6605, swd: 1.7925, ept: 618.0109
    Epoch [21/50], Test Losses: mse: 2.4708, mae: 0.7863, huber: 0.5316, swd: 1.3730, ept: 627.4912
      Epoch 21 composite train-obj: 0.616771
            Val objective improved 0.6943 → 0.6605, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 3.2027, mae: 0.8718, huber: 0.6134, swd: 1.6624, ept: 624.6467
    Epoch [22/50], Val Losses: mse: 3.3834, mae: 0.8972, huber: 0.6410, swd: 1.9226, ept: 623.9773
    Epoch [22/50], Test Losses: mse: 2.5171, mae: 0.7681, huber: 0.5160, swd: 1.4212, ept: 625.0422
      Epoch 22 composite train-obj: 0.613392
            Val objective improved 0.6605 → 0.6410, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 3.1532, mae: 0.8573, huber: 0.6034, swd: 1.6316, ept: 626.7734
    Epoch [23/50], Val Losses: mse: 3.7726, mae: 0.9387, huber: 0.6800, swd: 2.0353, ept: 611.2269
    Epoch [23/50], Test Losses: mse: 2.8277, mae: 0.8143, huber: 0.5591, swd: 1.4906, ept: 618.6236
      Epoch 23 composite train-obj: 0.603449
            No improvement (0.6800), counter 1/5
    Epoch [24/50], Train Losses: mse: 3.0918, mae: 0.8523, huber: 0.5987, swd: 1.5745, ept: 628.8983
    Epoch [24/50], Val Losses: mse: 4.0915, mae: 1.0693, huber: 0.7907, swd: 2.1859, ept: 604.1344
    Epoch [24/50], Test Losses: mse: 3.1937, mae: 0.9308, huber: 0.6586, swd: 1.7081, ept: 612.2591
      Epoch 24 composite train-obj: 0.598704
            No improvement (0.7907), counter 2/5
    Epoch [25/50], Train Losses: mse: 3.0445, mae: 0.8418, huber: 0.5909, swd: 1.5385, ept: 629.5043
    Epoch [25/50], Val Losses: mse: 3.8555, mae: 0.9972, huber: 0.7301, swd: 1.9607, ept: 610.0770
    Epoch [25/50], Test Losses: mse: 3.1574, mae: 0.8981, huber: 0.6335, swd: 1.5880, ept: 614.2030
      Epoch 25 composite train-obj: 0.590945
            No improvement (0.7301), counter 3/5
    Epoch [26/50], Train Losses: mse: 2.9716, mae: 0.8354, huber: 0.5846, swd: 1.4673, ept: 632.5629
    Epoch [26/50], Val Losses: mse: 3.7494, mae: 1.0519, huber: 0.7597, swd: 1.8725, ept: 617.4109
    Epoch [26/50], Test Losses: mse: 2.9504, mae: 0.9290, huber: 0.6406, swd: 1.4649, ept: 631.6781
      Epoch 26 composite train-obj: 0.584628
            No improvement (0.7597), counter 4/5
    Epoch [27/50], Train Losses: mse: 3.0166, mae: 0.8392, huber: 0.5882, swd: 1.4945, ept: 631.4469
    Epoch [27/50], Val Losses: mse: 3.9385, mae: 1.0314, huber: 0.7521, swd: 2.2150, ept: 612.8713
    Epoch [27/50], Test Losses: mse: 3.0619, mae: 0.9026, huber: 0.6280, swd: 1.7299, ept: 617.7493
      Epoch 27 composite train-obj: 0.588163
    Epoch [27/50], Test Losses: mse: 2.5171, mae: 0.7681, huber: 0.5160, swd: 1.4212, ept: 625.0422
    Best round's Test MSE: 2.5171, MAE: 0.7681, SWD: 1.4212
    Best round's Validation MSE: 3.3834, MAE: 0.8972, SWD: 1.9226
    Best round's Test verification MSE : 2.5171, MAE: 0.7681, SWD: 1.4212
    Time taken: 277.52 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.6883, mae: 1.7519, huber: 1.3949, swd: 4.7253, ept: 473.4408
    Epoch [1/50], Val Losses: mse: 7.3387, mae: 1.4643, huber: 1.1406, swd: 4.7683, ept: 537.9450
    Epoch [1/50], Test Losses: mse: 5.8461, mae: 1.2917, huber: 0.9669, swd: 3.8197, ept: 562.9261
      Epoch 1 composite train-obj: 1.394919
            Val objective improved inf → 1.1406, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.1607, mae: 1.3230, huber: 1.0008, swd: 3.8543, ept: 562.9180
    Epoch [2/50], Val Losses: mse: 8.0035, mae: 1.4813, huber: 1.1574, swd: 5.0708, ept: 558.2024
    Epoch [2/50], Test Losses: mse: 6.5869, mae: 1.3543, huber: 1.0282, swd: 4.2171, ept: 562.1543
      Epoch 2 composite train-obj: 1.000778
            No improvement (1.1574), counter 1/5
    Epoch [3/50], Train Losses: mse: 5.7975, mae: 1.2292, huber: 0.9213, swd: 3.7508, ept: 579.0122
    Epoch [3/50], Val Losses: mse: 6.5691, mae: 1.2369, huber: 0.9482, swd: 4.4332, ept: 581.3749
    Epoch [3/50], Test Losses: mse: 5.1069, mae: 1.0671, huber: 0.7799, swd: 3.4951, ept: 593.8070
      Epoch 3 composite train-obj: 0.921265
            Val objective improved 1.1406 → 0.9482, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.5686, mae: 1.1673, huber: 0.8703, swd: 3.6683, ept: 587.7131
    Epoch [4/50], Val Losses: mse: 7.1279, mae: 1.3711, huber: 1.0707, swd: 4.7765, ept: 565.9377
    Epoch [4/50], Test Losses: mse: 5.7499, mae: 1.2302, huber: 0.9267, swd: 3.8618, ept: 577.1910
      Epoch 4 composite train-obj: 0.870298
            No improvement (1.0707), counter 1/5
    Epoch [5/50], Train Losses: mse: 5.4151, mae: 1.1276, huber: 0.8379, swd: 3.6036, ept: 592.5680
    Epoch [5/50], Val Losses: mse: 7.1462, mae: 1.2842, huber: 0.9941, swd: 4.5842, ept: 570.9385
    Epoch [5/50], Test Losses: mse: 5.8389, mae: 1.1786, huber: 0.8834, swd: 3.7070, ept: 576.1054
      Epoch 5 composite train-obj: 0.837876
            No improvement (0.9941), counter 2/5
    Epoch [6/50], Train Losses: mse: 5.3218, mae: 1.1000, huber: 0.8162, swd: 3.5462, ept: 594.9777
    Epoch [6/50], Val Losses: mse: 6.3622, mae: 1.1890, huber: 0.9163, swd: 4.2941, ept: 575.6913
    Epoch [6/50], Test Losses: mse: 4.9193, mae: 1.0233, huber: 0.7500, swd: 3.3815, ept: 588.8861
      Epoch 6 composite train-obj: 0.816210
            Val objective improved 0.9482 → 0.9163, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.2096, mae: 1.0765, huber: 0.7977, swd: 3.4601, ept: 596.5268
    Epoch [7/50], Val Losses: mse: 5.8809, mae: 1.1154, huber: 0.8474, swd: 4.0926, ept: 586.8169
    Epoch [7/50], Test Losses: mse: 4.5942, mae: 0.9662, huber: 0.7010, swd: 3.2293, ept: 597.5537
      Epoch 7 composite train-obj: 0.797693
            Val objective improved 0.9163 → 0.8474, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.1264, mae: 1.0670, huber: 0.7893, swd: 3.3790, ept: 598.2452
    Epoch [8/50], Val Losses: mse: 6.1534, mae: 1.1309, huber: 0.8666, swd: 4.0557, ept: 588.3762
    Epoch [8/50], Test Losses: mse: 4.8768, mae: 0.9940, huber: 0.7303, swd: 3.2267, ept: 595.3644
      Epoch 8 composite train-obj: 0.789304
            No improvement (0.8666), counter 1/5
    Epoch [9/50], Train Losses: mse: 5.0264, mae: 1.0583, huber: 0.7808, swd: 3.2768, ept: 599.7937
    Epoch [9/50], Val Losses: mse: 6.6296, mae: 1.2091, huber: 0.9308, swd: 4.2073, ept: 579.3396
    Epoch [9/50], Test Losses: mse: 5.4104, mae: 1.1007, huber: 0.8173, swd: 3.4523, ept: 586.7679
      Epoch 9 composite train-obj: 0.780821
            No improvement (0.9308), counter 2/5
    Epoch [10/50], Train Losses: mse: 4.9014, mae: 1.0462, huber: 0.7701, swd: 3.1401, ept: 600.5585
    Epoch [10/50], Val Losses: mse: 5.6213, mae: 1.0838, huber: 0.8239, swd: 3.6367, ept: 590.7867
    Epoch [10/50], Test Losses: mse: 4.3121, mae: 0.9267, huber: 0.6700, swd: 2.8352, ept: 603.0768
      Epoch 10 composite train-obj: 0.770083
            Val objective improved 0.8474 → 0.8239, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 4.6911, mae: 1.0194, huber: 0.7473, swd: 2.9451, ept: 602.8598
    Epoch [11/50], Val Losses: mse: 5.0955, mae: 1.0563, huber: 0.7905, swd: 3.3495, ept: 599.3079
    Epoch [11/50], Test Losses: mse: 3.9246, mae: 0.9089, huber: 0.6442, swd: 2.6207, ept: 609.3014
      Epoch 11 composite train-obj: 0.747301
            Val objective improved 0.8239 → 0.7905, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 4.5331, mae: 1.0094, huber: 0.7363, swd: 2.7813, ept: 604.4484
    Epoch [12/50], Val Losses: mse: 5.4442, mae: 1.1402, huber: 0.8692, swd: 3.4843, ept: 586.3618
    Epoch [12/50], Test Losses: mse: 4.4315, mae: 1.0182, huber: 0.7490, swd: 2.8245, ept: 592.7543
      Epoch 12 composite train-obj: 0.736256
            No improvement (0.8692), counter 1/5
    Epoch [13/50], Train Losses: mse: 4.3181, mae: 0.9893, huber: 0.7177, swd: 2.5829, ept: 607.2535
    Epoch [13/50], Val Losses: mse: 5.2142, mae: 1.0710, huber: 0.8084, swd: 3.2887, ept: 591.7733
    Epoch [13/50], Test Losses: mse: 4.0778, mae: 0.9407, huber: 0.6786, swd: 2.5762, ept: 597.5079
      Epoch 13 composite train-obj: 0.717657
            No improvement (0.8084), counter 2/5
    Epoch [14/50], Train Losses: mse: 4.1641, mae: 0.9739, huber: 0.7036, swd: 2.4379, ept: 608.6326
    Epoch [14/50], Val Losses: mse: 4.8643, mae: 1.0836, huber: 0.8107, swd: 2.8762, ept: 592.9991
    Epoch [14/50], Test Losses: mse: 3.7399, mae: 0.9317, huber: 0.6623, swd: 2.2223, ept: 605.5406
      Epoch 14 composite train-obj: 0.703585
            No improvement (0.8107), counter 3/5
    Epoch [15/50], Train Losses: mse: 4.0090, mae: 0.9659, huber: 0.6942, swd: 2.2949, ept: 609.1083
    Epoch [15/50], Val Losses: mse: 4.4002, mae: 0.9852, huber: 0.7299, swd: 2.7144, ept: 605.6706
    Epoch [15/50], Test Losses: mse: 3.3519, mae: 0.8473, huber: 0.5957, swd: 2.0743, ept: 613.4489
      Epoch 15 composite train-obj: 0.694175
            Val objective improved 0.7905 → 0.7299, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 3.7850, mae: 0.9390, huber: 0.6713, swd: 2.0929, ept: 612.8299
    Epoch [16/50], Val Losses: mse: 4.2739, mae: 1.0033, huber: 0.7357, swd: 2.2965, ept: 603.7120
    Epoch [16/50], Test Losses: mse: 3.4476, mae: 0.8933, huber: 0.6275, swd: 1.8238, ept: 609.9378
      Epoch 16 composite train-obj: 0.671276
            No improvement (0.7357), counter 1/5
    Epoch [17/50], Train Losses: mse: 3.6338, mae: 0.9278, huber: 0.6603, swd: 1.9461, ept: 614.8519
    Epoch [17/50], Val Losses: mse: 4.2768, mae: 1.0168, huber: 0.7532, swd: 2.4643, ept: 597.6097
    Epoch [17/50], Test Losses: mse: 3.3331, mae: 0.8914, huber: 0.6301, swd: 1.8937, ept: 603.2907
      Epoch 17 composite train-obj: 0.660271
            No improvement (0.7532), counter 2/5
    Epoch [18/50], Train Losses: mse: 3.4966, mae: 0.9185, huber: 0.6514, swd: 1.8263, ept: 616.5514
    Epoch [18/50], Val Losses: mse: 4.4623, mae: 1.0658, huber: 0.7916, swd: 2.4024, ept: 598.5834
    Epoch [18/50], Test Losses: mse: 3.4799, mae: 0.9354, huber: 0.6610, swd: 1.8583, ept: 603.8709
      Epoch 18 composite train-obj: 0.651413
            No improvement (0.7916), counter 3/5
    Epoch [19/50], Train Losses: mse: 3.3294, mae: 0.8946, huber: 0.6302, swd: 1.7064, ept: 620.3495
    Epoch [19/50], Val Losses: mse: 3.3861, mae: 0.9077, huber: 0.6459, swd: 1.8079, ept: 621.0790
    Epoch [19/50], Test Losses: mse: 2.4704, mae: 0.7615, huber: 0.5051, swd: 1.3013, ept: 630.3673
      Epoch 19 composite train-obj: 0.630200
            Val objective improved 0.7299 → 0.6459, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 3.2931, mae: 0.8871, huber: 0.6245, swd: 1.6665, ept: 621.7226
    Epoch [20/50], Val Losses: mse: 3.2987, mae: 0.8717, huber: 0.6202, swd: 1.7053, ept: 623.3736
    Epoch [20/50], Test Losses: mse: 2.4381, mae: 0.7472, huber: 0.4993, swd: 1.2331, ept: 635.3976
      Epoch 20 composite train-obj: 0.624483
            Val objective improved 0.6459 → 0.6202, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 3.1807, mae: 0.8727, huber: 0.6119, swd: 1.5935, ept: 625.0922
    Epoch [21/50], Val Losses: mse: 3.4038, mae: 0.8906, huber: 0.6440, swd: 1.8500, ept: 623.5788
    Epoch [21/50], Test Losses: mse: 2.5605, mae: 0.7680, huber: 0.5265, swd: 1.3727, ept: 633.1248
      Epoch 21 composite train-obj: 0.611891
            No improvement (0.6440), counter 1/5
    Epoch [22/50], Train Losses: mse: 3.1335, mae: 0.8617, huber: 0.6042, swd: 1.5592, ept: 628.4616
    Epoch [22/50], Val Losses: mse: 4.8376, mae: 1.0710, huber: 0.8077, swd: 2.2275, ept: 606.1149
    Epoch [22/50], Test Losses: mse: 3.7878, mae: 0.9341, huber: 0.6759, swd: 1.7418, ept: 607.7187
      Epoch 22 composite train-obj: 0.604230
            No improvement (0.8077), counter 2/5
    Epoch [23/50], Train Losses: mse: 3.1371, mae: 0.8672, huber: 0.6073, swd: 1.5534, ept: 629.2296
    Epoch [23/50], Val Losses: mse: 3.5040, mae: 0.9064, huber: 0.6554, swd: 1.9411, ept: 623.2458
    Epoch [23/50], Test Losses: mse: 2.7559, mae: 0.7956, huber: 0.5506, swd: 1.5252, ept: 620.6840
      Epoch 23 composite train-obj: 0.607281
            No improvement (0.6554), counter 3/5
    Epoch [24/50], Train Losses: mse: 3.0566, mae: 0.8501, huber: 0.5951, swd: 1.5121, ept: 631.1496
    Epoch [24/50], Val Losses: mse: 3.2625, mae: 0.9026, huber: 0.6463, swd: 1.6364, ept: 630.6678
    Epoch [24/50], Test Losses: mse: 2.5506, mae: 0.7837, huber: 0.5326, swd: 1.2471, ept: 642.1313
      Epoch 24 composite train-obj: 0.595060
            No improvement (0.6463), counter 4/5
    Epoch [25/50], Train Losses: mse: 3.0479, mae: 0.8500, huber: 0.5940, swd: 1.5016, ept: 632.4664
    Epoch [25/50], Val Losses: mse: 3.2030, mae: 0.8526, huber: 0.6059, swd: 1.6175, ept: 630.4424
    Epoch [25/50], Test Losses: mse: 2.4517, mae: 0.7285, huber: 0.4911, swd: 1.2156, ept: 640.8956
      Epoch 25 composite train-obj: 0.594039
            Val objective improved 0.6202 → 0.6059, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 3.0206, mae: 0.8464, huber: 0.5914, swd: 1.4654, ept: 634.0013
    Epoch [26/50], Val Losses: mse: 3.7670, mae: 1.0149, huber: 0.7309, swd: 1.8885, ept: 615.2339
    Epoch [26/50], Test Losses: mse: 2.7247, mae: 0.8600, huber: 0.5784, swd: 1.3910, ept: 634.0668
      Epoch 26 composite train-obj: 0.591407
            No improvement (0.7309), counter 1/5
    Epoch [27/50], Train Losses: mse: 2.9555, mae: 0.8350, huber: 0.5821, swd: 1.4399, ept: 635.5621
    Epoch [27/50], Val Losses: mse: 3.8715, mae: 1.0110, huber: 0.7473, swd: 1.9945, ept: 613.1769
    Epoch [27/50], Test Losses: mse: 3.1022, mae: 0.8939, huber: 0.6333, swd: 1.5962, ept: 624.0133
      Epoch 27 composite train-obj: 0.582100
            No improvement (0.7473), counter 2/5
    Epoch [28/50], Train Losses: mse: 2.9350, mae: 0.8285, huber: 0.5781, swd: 1.4219, ept: 636.2819
    Epoch [28/50], Val Losses: mse: 3.3346, mae: 0.9149, huber: 0.6626, swd: 1.5610, ept: 628.3881
    Epoch [28/50], Test Losses: mse: 2.5920, mae: 0.7921, huber: 0.5452, swd: 1.2095, ept: 643.9832
      Epoch 28 composite train-obj: 0.578070
            No improvement (0.6626), counter 3/5
    Epoch [29/50], Train Losses: mse: 2.9009, mae: 0.8208, huber: 0.5716, swd: 1.4048, ept: 638.4564
    Epoch [29/50], Val Losses: mse: 4.2551, mae: 1.0505, huber: 0.7819, swd: 2.0763, ept: 611.1797
    Epoch [29/50], Test Losses: mse: 3.5203, mae: 0.9396, huber: 0.6753, swd: 1.6709, ept: 626.1286
      Epoch 29 composite train-obj: 0.571557
            No improvement (0.7819), counter 4/5
    Epoch [30/50], Train Losses: mse: 2.9097, mae: 0.8240, huber: 0.5742, swd: 1.3970, ept: 638.7282
    Epoch [30/50], Val Losses: mse: 3.4037, mae: 0.9672, huber: 0.6885, swd: 1.6565, ept: 631.9463
    Epoch [30/50], Test Losses: mse: 2.5325, mae: 0.8328, huber: 0.5559, swd: 1.2579, ept: 652.4911
      Epoch 30 composite train-obj: 0.574231
    Epoch [30/50], Test Losses: mse: 2.4517, mae: 0.7285, huber: 0.4911, swd: 1.2156, ept: 640.8956
    Best round's Test MSE: 2.4517, MAE: 0.7285, SWD: 1.2156
    Best round's Validation MSE: 3.2030, MAE: 0.8526, SWD: 1.6175
    Best round's Test verification MSE : 2.4517, MAE: 0.7285, SWD: 1.2156
    Time taken: 322.08 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.6258, mae: 1.7372, huber: 1.3821, swd: 4.7668, ept: 463.4043
    Epoch [1/50], Val Losses: mse: 7.4833, mae: 1.4777, huber: 1.1538, swd: 4.8585, ept: 531.5819
    Epoch [1/50], Test Losses: mse: 5.9488, mae: 1.3054, huber: 0.9786, swd: 3.8318, ept: 554.9711
      Epoch 1 composite train-obj: 1.382102
            Val objective improved inf → 1.1538, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.0805, mae: 1.3006, huber: 0.9828, swd: 3.9606, ept: 565.0103
    Epoch [2/50], Val Losses: mse: 6.8394, mae: 1.3156, huber: 1.0126, swd: 4.6872, ept: 574.4042
    Epoch [2/50], Test Losses: mse: 5.3303, mae: 1.1384, huber: 0.8357, swd: 3.7023, ept: 588.3877
      Epoch 2 composite train-obj: 0.982753
            Val objective improved 1.1538 → 1.0126, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.8482, mae: 1.2382, huber: 0.9306, swd: 3.8829, ept: 581.8178
    Epoch [3/50], Val Losses: mse: 6.6813, mae: 1.2656, huber: 0.9769, swd: 4.5924, ept: 578.0280
    Epoch [3/50], Test Losses: mse: 5.2157, mae: 1.1100, huber: 0.8190, swd: 3.5926, ept: 591.4916
      Epoch 3 composite train-obj: 0.930617
            Val objective improved 1.0126 → 0.9769, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.6200, mae: 1.1824, huber: 0.8857, swd: 3.7745, ept: 588.9057
    Epoch [4/50], Val Losses: mse: 6.4744, mae: 1.2968, huber: 0.9975, swd: 4.6362, ept: 578.5733
    Epoch [4/50], Test Losses: mse: 5.1097, mae: 1.1302, huber: 0.8343, swd: 3.6988, ept: 583.9957
      Epoch 4 composite train-obj: 0.885745
            No improvement (0.9975), counter 1/5
    Epoch [5/50], Train Losses: mse: 5.4813, mae: 1.1512, huber: 0.8597, swd: 3.7390, ept: 592.5317
    Epoch [5/50], Val Losses: mse: 6.6359, mae: 1.2463, huber: 0.9530, swd: 4.4747, ept: 581.2083
    Epoch [5/50], Test Losses: mse: 5.1849, mae: 1.0999, huber: 0.8057, swd: 3.4999, ept: 595.3855
      Epoch 5 composite train-obj: 0.859662
            Val objective improved 0.9769 → 0.9530, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.3390, mae: 1.1242, huber: 0.8370, swd: 3.6203, ept: 596.8662
    Epoch [6/50], Val Losses: mse: 6.0722, mae: 1.1854, huber: 0.9101, swd: 4.3597, ept: 589.3006
    Epoch [6/50], Test Losses: mse: 4.7848, mae: 1.0376, huber: 0.7625, swd: 3.4425, ept: 595.4476
      Epoch 6 composite train-obj: 0.836999
            Val objective improved 0.9530 → 0.9101, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.2172, mae: 1.1033, huber: 0.8198, swd: 3.5277, ept: 598.4580
    Epoch [7/50], Val Losses: mse: 6.0315, mae: 1.1587, huber: 0.8868, swd: 4.2512, ept: 589.0718
    Epoch [7/50], Test Losses: mse: 4.7578, mae: 1.0154, huber: 0.7435, swd: 3.3554, ept: 594.7177
      Epoch 7 composite train-obj: 0.819835
            Val objective improved 0.9101 → 0.8868, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.0512, mae: 1.0743, huber: 0.7954, swd: 3.3896, ept: 600.6967
    Epoch [8/50], Val Losses: mse: 6.5565, mae: 1.2256, huber: 0.9358, swd: 4.2828, ept: 585.6457
    Epoch [8/50], Test Losses: mse: 5.1296, mae: 1.0854, huber: 0.7919, swd: 3.3570, ept: 595.0464
      Epoch 8 composite train-obj: 0.795429
            No improvement (0.9358), counter 1/5
    Epoch [9/50], Train Losses: mse: 4.8557, mae: 1.0607, huber: 0.7802, swd: 3.1552, ept: 602.5040
    Epoch [9/50], Val Losses: mse: 5.7046, mae: 1.1723, huber: 0.8872, swd: 3.7289, ept: 594.2720
    Epoch [9/50], Test Losses: mse: 4.3758, mae: 1.0187, huber: 0.7328, swd: 2.8656, ept: 605.1437
      Epoch 9 composite train-obj: 0.780246
            No improvement (0.8872), counter 2/5
    Epoch [10/50], Train Losses: mse: 4.4948, mae: 1.0323, huber: 0.7519, swd: 2.7749, ept: 605.8854
    Epoch [10/50], Val Losses: mse: 4.8995, mae: 1.0819, huber: 0.8119, swd: 3.1009, ept: 601.8830
    Epoch [10/50], Test Losses: mse: 3.7115, mae: 0.9162, huber: 0.6509, swd: 2.3448, ept: 611.2434
      Epoch 10 composite train-obj: 0.751888
            Val objective improved 0.8868 → 0.8119, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 4.0690, mae: 0.9932, huber: 0.7158, swd: 2.3565, ept: 609.1264
    Epoch [11/50], Val Losses: mse: 4.2776, mae: 1.1003, huber: 0.8000, swd: 2.4773, ept: 604.2971
    Epoch [11/50], Test Losses: mse: 3.4058, mae: 0.9846, huber: 0.6847, swd: 1.9444, ept: 611.9955
      Epoch 11 composite train-obj: 0.715831
            Val objective improved 0.8119 → 0.8000, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 3.6600, mae: 0.9593, huber: 0.6824, swd: 2.0038, ept: 614.1618
    Epoch [12/50], Val Losses: mse: 4.4387, mae: 1.0838, huber: 0.8048, swd: 2.4613, ept: 599.4053
    Epoch [12/50], Test Losses: mse: 3.5314, mae: 0.9707, huber: 0.6923, swd: 1.9338, ept: 599.0474
      Epoch 12 composite train-obj: 0.682384
            No improvement (0.8048), counter 1/5
    Epoch [13/50], Train Losses: mse: 3.4612, mae: 0.9320, huber: 0.6586, swd: 1.8439, ept: 618.6666
    Epoch [13/50], Val Losses: mse: 4.1224, mae: 1.0219, huber: 0.7555, swd: 2.3665, ept: 610.1977
    Epoch [13/50], Test Losses: mse: 3.1539, mae: 0.8843, huber: 0.6234, swd: 1.8194, ept: 613.1319
      Epoch 13 composite train-obj: 0.658628
            Val objective improved 0.8000 → 0.7555, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 3.2935, mae: 0.9103, huber: 0.6406, swd: 1.7025, ept: 622.7669
    Epoch [14/50], Val Losses: mse: 4.7935, mae: 1.1838, huber: 0.8885, swd: 2.7103, ept: 594.8156
    Epoch [14/50], Test Losses: mse: 3.8176, mae: 1.0497, huber: 0.7580, swd: 2.1442, ept: 602.4307
      Epoch 14 composite train-obj: 0.640627
            No improvement (0.8885), counter 1/5
    Epoch [15/50], Train Losses: mse: 3.1513, mae: 0.8895, huber: 0.6238, swd: 1.6093, ept: 626.6349
    Epoch [15/50], Val Losses: mse: 4.4544, mae: 1.0570, huber: 0.7881, swd: 2.3389, ept: 607.3689
    Epoch [15/50], Test Losses: mse: 3.2772, mae: 0.8935, huber: 0.6311, swd: 1.7131, ept: 617.5229
      Epoch 15 composite train-obj: 0.623799
            No improvement (0.7881), counter 2/5
    Epoch [16/50], Train Losses: mse: 3.0915, mae: 0.8801, huber: 0.6152, swd: 1.5722, ept: 630.5453
    Epoch [16/50], Val Losses: mse: 4.0122, mae: 1.0287, huber: 0.7640, swd: 2.0731, ept: 612.2316
    Epoch [16/50], Test Losses: mse: 3.0975, mae: 0.8913, huber: 0.6320, swd: 1.5846, ept: 630.4546
      Epoch 16 composite train-obj: 0.615243
            No improvement (0.7640), counter 3/5
    Epoch [17/50], Train Losses: mse: 3.0049, mae: 0.8627, huber: 0.6020, swd: 1.5176, ept: 633.7305
    Epoch [17/50], Val Losses: mse: 4.7168, mae: 1.0954, huber: 0.8243, swd: 2.3840, ept: 610.6446
    Epoch [17/50], Test Losses: mse: 3.4817, mae: 0.9337, huber: 0.6670, swd: 1.7644, ept: 628.2843
      Epoch 17 composite train-obj: 0.602026
            No improvement (0.8243), counter 4/5
    Epoch [18/50], Train Losses: mse: 2.9723, mae: 0.8509, huber: 0.5931, swd: 1.5304, ept: 635.6055
    Epoch [18/50], Val Losses: mse: 5.2904, mae: 1.1949, huber: 0.9154, swd: 2.6340, ept: 593.2118
    Epoch [18/50], Test Losses: mse: 4.1292, mae: 1.0532, huber: 0.7754, swd: 2.0479, ept: 603.3739
      Epoch 18 composite train-obj: 0.593060
    Epoch [18/50], Test Losses: mse: 3.1539, mae: 0.8843, huber: 0.6234, swd: 1.8194, ept: 613.1319
    Best round's Test MSE: 3.1539, MAE: 0.8843, SWD: 1.8194
    Best round's Validation MSE: 4.1224, MAE: 1.0219, SWD: 2.3665
    Best round's Test verification MSE : 3.1539, MAE: 0.8843, SWD: 1.8194
    Time taken: 186.01 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_rossler_seq336_pred720_20250513_1230)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 2.7076 ± 0.3168
      mae: 0.7936 ± 0.0661
      huber: 0.5435 ± 0.0574
      swd: 1.4854 ± 0.2506
      ept: 626.3566 ± 11.3725
      count: 33.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 3.5696 ± 0.3978
      mae: 0.9239 ± 0.0717
      huber: 0.6674 ± 0.0639
      swd: 1.9689 ± 0.3075
      ept: 621.5391 ± 8.4428
      count: 33.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 785.68 seconds
    
    Experiment complete: TimeMixer_rossler_seq336_pred720_20250513_1230
    Model: TimeMixer
    Dataset: rossler
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 7.3591, mae: 1.4447, huber: 1.1071, swd: 2.9490, ept: 51.2661
    Epoch [1/50], Val Losses: mse: 22.4640, mae: 3.2054, huber: 2.8083, swd: 7.2035, ept: 55.9845
    Epoch [1/50], Test Losses: mse: 24.0162, mae: 3.3815, huber: 2.9721, swd: 7.1300, ept: 53.3593
      Epoch 1 composite train-obj: 1.107140
            Val objective improved inf → 2.8083, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 6.6946, mae: 1.2752, huber: 0.9581, swd: 2.5469, ept: 52.9188
    Epoch [2/50], Val Losses: mse: 5.0101, mae: 1.3686, huber: 1.0144, swd: 3.5252, ept: 89.3411
    Epoch [2/50], Test Losses: mse: 5.0077, mae: 1.4538, huber: 1.0810, swd: 3.4826, ept: 89.2876
      Epoch 2 composite train-obj: 0.958110
            Val objective improved 2.8083 → 1.0144, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.5331, mae: 1.0664, huber: 0.7746, swd: 1.9353, ept: 53.7026
    Epoch [3/50], Val Losses: mse: 4.2253, mae: 1.1419, huber: 0.8058, swd: 2.7630, ept: 92.6818
    Epoch [3/50], Test Losses: mse: 4.2671, mae: 1.2058, huber: 0.8603, swd: 2.7483, ept: 91.8070
      Epoch 3 composite train-obj: 0.774569
            Val objective improved 1.0144 → 0.8058, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.2516, mae: 1.0083, huber: 0.7255, swd: 1.7150, ept: 53.5912
    Epoch [4/50], Val Losses: mse: 6.8124, mae: 1.7840, huber: 1.4096, swd: 4.9873, ept: 82.8239
    Epoch [4/50], Test Losses: mse: 7.1892, mae: 1.8878, huber: 1.4997, swd: 5.1315, ept: 81.7402
      Epoch 4 composite train-obj: 0.725465
            No improvement (1.4096), counter 1/5
    Epoch [5/50], Train Losses: mse: 5.3017, mae: 1.0298, huber: 0.7452, swd: 1.7011, ept: 53.6474
    Epoch [5/50], Val Losses: mse: 1.9259, mae: 0.7384, huber: 0.4382, swd: 1.1468, ept: 94.2003
    Epoch [5/50], Test Losses: mse: 1.8373, mae: 0.7656, huber: 0.4542, swd: 1.0377, ept: 94.3409
      Epoch 5 composite train-obj: 0.745223
            Val objective improved 0.8058 → 0.4382, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.0036, mae: 0.9536, huber: 0.6844, swd: 1.4870, ept: 53.6426
    Epoch [6/50], Val Losses: mse: 4.0857, mae: 1.4368, huber: 1.0658, swd: 3.3774, ept: 93.0863
    Epoch [6/50], Test Losses: mse: 4.3137, mae: 1.5196, huber: 1.1381, swd: 3.5335, ept: 93.0199
      Epoch 6 composite train-obj: 0.684400
            No improvement (1.0658), counter 1/5
    Epoch [7/50], Train Losses: mse: 5.2327, mae: 1.0427, huber: 0.7543, swd: 1.6517, ept: 53.6211
    Epoch [7/50], Val Losses: mse: 2.3311, mae: 0.8358, huber: 0.5263, swd: 1.3765, ept: 93.9346
    Epoch [7/50], Test Losses: mse: 2.4173, mae: 0.9097, huber: 0.5859, swd: 1.4160, ept: 93.6195
      Epoch 7 composite train-obj: 0.754293
            No improvement (0.5263), counter 2/5
    Epoch [8/50], Train Losses: mse: 4.8661, mae: 0.9446, huber: 0.6750, swd: 1.3396, ept: 53.8820
    Epoch [8/50], Val Losses: mse: 1.5563, mae: 0.5100, huber: 0.2774, swd: 1.0859, ept: 94.2908
    Epoch [8/50], Test Losses: mse: 1.4367, mae: 0.5338, huber: 0.2893, swd: 0.9754, ept: 94.3543
      Epoch 8 composite train-obj: 0.674986
            Val objective improved 0.4382 → 0.2774, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 4.5908, mae: 0.8646, huber: 0.6149, swd: 1.1706, ept: 53.9405
    Epoch [9/50], Val Losses: mse: 0.9727, mae: 0.4105, huber: 0.1876, swd: 0.6629, ept: 94.6636
    Epoch [9/50], Test Losses: mse: 0.9378, mae: 0.4323, huber: 0.1985, swd: 0.6300, ept: 94.5886
      Epoch 9 composite train-obj: 0.614931
            Val objective improved 0.2774 → 0.1876, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 4.5726, mae: 0.8517, huber: 0.6084, swd: 1.1195, ept: 53.6673
    Epoch [10/50], Val Losses: mse: 2.6788, mae: 0.8481, huber: 0.5338, swd: 2.0145, ept: 93.7296
    Epoch [10/50], Test Losses: mse: 2.5729, mae: 0.8892, huber: 0.5609, swd: 1.8801, ept: 93.2967
      Epoch 10 composite train-obj: 0.608402
            No improvement (0.5338), counter 1/5
    Epoch [11/50], Train Losses: mse: 5.3822, mae: 1.0448, huber: 0.7590, swd: 1.7971, ept: 53.4757
    Epoch [11/50], Val Losses: mse: 2.8593, mae: 0.9925, huber: 0.6536, swd: 1.9193, ept: 93.5943
    Epoch [11/50], Test Losses: mse: 2.7101, mae: 1.0279, huber: 0.6733, swd: 1.7291, ept: 93.5184
      Epoch 11 composite train-obj: 0.759018
            No improvement (0.6536), counter 2/5
    Epoch [12/50], Train Losses: mse: 5.0847, mae: 1.0187, huber: 0.7329, swd: 1.5089, ept: 53.6368
    Epoch [12/50], Val Losses: mse: 3.8401, mae: 1.0577, huber: 0.7044, swd: 2.9783, ept: 92.9857
    Epoch [12/50], Test Losses: mse: 3.7498, mae: 1.1312, huber: 0.7561, swd: 2.8359, ept: 92.5681
      Epoch 12 composite train-obj: 0.732937
            No improvement (0.7044), counter 3/5
    Epoch [13/50], Train Losses: mse: 5.5418, mae: 1.0130, huber: 0.7389, swd: 1.8572, ept: 53.6181
    Epoch [13/50], Val Losses: mse: 1.6351, mae: 0.5719, huber: 0.3116, swd: 1.3225, ept: 94.3402
    Epoch [13/50], Test Losses: mse: 1.5085, mae: 0.5818, huber: 0.3148, swd: 1.1991, ept: 94.1966
      Epoch 13 composite train-obj: 0.738901
            No improvement (0.3116), counter 4/5
    Epoch [14/50], Train Losses: mse: 4.7817, mae: 0.8972, huber: 0.6392, swd: 1.3862, ept: 53.9546
    Epoch [14/50], Val Losses: mse: 1.2846, mae: 0.6061, huber: 0.3132, swd: 1.0135, ept: 94.8448
    Epoch [14/50], Test Losses: mse: 1.2437, mae: 0.6325, huber: 0.3296, swd: 0.9753, ept: 94.5689
      Epoch 14 composite train-obj: 0.639156
    Epoch [14/50], Test Losses: mse: 0.9378, mae: 0.4323, huber: 0.1985, swd: 0.6300, ept: 94.5886
    Best round's Test MSE: 0.9378, MAE: 0.4323, SWD: 0.6300
    Best round's Validation MSE: 0.9727, MAE: 0.4105, SWD: 0.6629
    Best round's Test verification MSE : 0.9378, MAE: 0.4323, SWD: 0.6300
    Time taken: 74.53 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.3189, mae: 1.4434, huber: 1.1057, swd: 2.9362, ept: 51.4320
    Epoch [1/50], Val Losses: mse: 7.0977, mae: 1.7336, huber: 1.3755, swd: 4.6817, ept: 77.2402
    Epoch [1/50], Test Losses: mse: 7.3543, mae: 1.8385, huber: 1.4681, swd: 4.6194, ept: 74.5908
      Epoch 1 composite train-obj: 1.105703
            Val objective improved inf → 1.3755, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.9328, mae: 1.1570, huber: 0.8517, swd: 2.1859, ept: 53.3325
    Epoch [2/50], Val Losses: mse: 7.4822, mae: 1.4842, huber: 1.1351, swd: 4.2491, ept: 85.8233
    Epoch [2/50], Test Losses: mse: 7.4616, mae: 1.5758, huber: 1.2100, swd: 4.2128, ept: 83.8872
      Epoch 2 composite train-obj: 0.851743
            Val objective improved 1.3755 → 1.1351, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.8680, mae: 1.0627, huber: 0.7773, swd: 2.0528, ept: 53.6268
    Epoch [3/50], Val Losses: mse: 2.1863, mae: 0.7104, huber: 0.4235, swd: 1.5107, ept: 93.8424
    Epoch [3/50], Test Losses: mse: 2.0573, mae: 0.7378, huber: 0.4382, swd: 1.3758, ept: 93.9121
      Epoch 3 composite train-obj: 0.777261
            Val objective improved 1.1351 → 0.4235, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.1207, mae: 0.9685, huber: 0.6969, swd: 1.6006, ept: 53.7570
    Epoch [4/50], Val Losses: mse: 2.2792, mae: 0.7382, huber: 0.4374, swd: 1.4445, ept: 93.4444
    Epoch [4/50], Test Losses: mse: 2.1819, mae: 0.7814, huber: 0.4640, swd: 1.3114, ept: 93.1668
      Epoch 4 composite train-obj: 0.696933
            No improvement (0.4374), counter 1/5
    Epoch [5/50], Train Losses: mse: 5.1177, mae: 0.9788, huber: 0.7037, swd: 1.5757, ept: 53.7572
    Epoch [5/50], Val Losses: mse: 1.9466, mae: 0.5290, huber: 0.2947, swd: 1.5232, ept: 93.7064
    Epoch [5/50], Test Losses: mse: 1.8221, mae: 0.5631, huber: 0.3149, swd: 1.4025, ept: 93.2818
      Epoch 5 composite train-obj: 0.703673
            Val objective improved 0.4235 → 0.2947, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.0322, mae: 0.9424, huber: 0.6783, swd: 1.5313, ept: 53.6264
    Epoch [6/50], Val Losses: mse: 8.7495, mae: 1.1650, huber: 0.8278, swd: 5.5010, ept: 93.5201
    Epoch [6/50], Test Losses: mse: 8.5271, mae: 1.2209, huber: 0.8672, swd: 5.3399, ept: 93.1090
      Epoch 6 composite train-obj: 0.678345
            No improvement (0.8278), counter 1/5
    Epoch [7/50], Train Losses: mse: 5.1553, mae: 0.9384, huber: 0.6776, swd: 1.5283, ept: 53.8062
    Epoch [7/50], Val Losses: mse: 4.1464, mae: 1.2323, huber: 0.8831, swd: 3.4848, ept: 92.5467
    Epoch [7/50], Test Losses: mse: 4.1826, mae: 1.2924, huber: 0.9318, swd: 3.4744, ept: 91.9105
      Epoch 7 composite train-obj: 0.677590
            No improvement (0.8831), counter 2/5
    Epoch [8/50], Train Losses: mse: 5.0203, mae: 0.9493, huber: 0.6833, swd: 1.4673, ept: 53.8350
    Epoch [8/50], Val Losses: mse: 3.0133, mae: 0.9040, huber: 0.5823, swd: 2.5903, ept: 93.7131
    Epoch [8/50], Test Losses: mse: 2.9303, mae: 0.9524, huber: 0.6207, swd: 2.5093, ept: 93.2807
      Epoch 8 composite train-obj: 0.683276
            No improvement (0.5823), counter 3/5
    Epoch [9/50], Train Losses: mse: 4.7270, mae: 0.8768, huber: 0.6281, swd: 1.2746, ept: 53.6804
    Epoch [9/50], Val Losses: mse: 3.8786, mae: 1.0959, huber: 0.7730, swd: 2.7140, ept: 92.6558
    Epoch [9/50], Test Losses: mse: 3.9718, mae: 1.1693, huber: 0.8355, swd: 2.6373, ept: 91.9719
      Epoch 9 composite train-obj: 0.628060
            No improvement (0.7730), counter 4/5
    Epoch [10/50], Train Losses: mse: 5.3467, mae: 1.0070, huber: 0.7310, swd: 1.7716, ept: 53.5405
    Epoch [10/50], Val Losses: mse: 2.4600, mae: 0.8026, huber: 0.4984, swd: 1.8205, ept: 93.3620
    Epoch [10/50], Test Losses: mse: 2.5212, mae: 0.8606, huber: 0.5384, swd: 1.8738, ept: 92.9998
      Epoch 10 composite train-obj: 0.731011
    Epoch [10/50], Test Losses: mse: 1.8221, mae: 0.5631, huber: 0.3149, swd: 1.4025, ept: 93.2818
    Best round's Test MSE: 1.8221, MAE: 0.5631, SWD: 1.4025
    Best round's Validation MSE: 1.9466, MAE: 0.5290, SWD: 1.5232
    Best round's Test verification MSE : 1.8221, MAE: 0.5631, SWD: 1.4025
    Time taken: 52.94 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 7.2913, mae: 1.4386, huber: 1.1021, swd: 2.6625, ept: 51.3565
    Epoch [1/50], Val Losses: mse: 4.7445, mae: 1.3003, huber: 0.9429, swd: 2.8676, ept: 88.6190
    Epoch [1/50], Test Losses: mse: 4.7976, mae: 1.3839, huber: 1.0108, swd: 2.8137, ept: 87.8958
      Epoch 1 composite train-obj: 1.102076
            Val objective improved inf → 0.9429, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.6499, mae: 1.1169, huber: 0.8143, swd: 1.8327, ept: 53.5937
    Epoch [2/50], Val Losses: mse: 4.0282, mae: 1.2446, huber: 0.8916, swd: 2.1555, ept: 89.8401
    Epoch [2/50], Test Losses: mse: 4.0322, mae: 1.2974, huber: 0.9339, swd: 2.0437, ept: 90.1196
      Epoch 2 composite train-obj: 0.814309
            Val objective improved 0.9429 → 0.8916, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.5395, mae: 1.0974, huber: 0.8003, swd: 1.7235, ept: 53.5927
    Epoch [3/50], Val Losses: mse: 3.9081, mae: 1.2008, huber: 0.8590, swd: 2.2049, ept: 92.5611
    Epoch [3/50], Test Losses: mse: 3.8848, mae: 1.2586, huber: 0.9034, swd: 2.0715, ept: 92.4509
      Epoch 3 composite train-obj: 0.800322
            Val objective improved 0.8916 → 0.8590, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.6866, mae: 1.1077, huber: 0.8091, swd: 1.8452, ept: 53.3871
    Epoch [4/50], Val Losses: mse: 3.4929, mae: 1.1456, huber: 0.8048, swd: 2.2330, ept: 92.6247
    Epoch [4/50], Test Losses: mse: 3.5450, mae: 1.2041, huber: 0.8539, swd: 2.1712, ept: 92.0792
      Epoch 4 composite train-obj: 0.809140
            Val objective improved 0.8590 → 0.8048, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.3567, mae: 1.0391, huber: 0.7536, swd: 1.5907, ept: 53.4548
    Epoch [5/50], Val Losses: mse: 5.8876, mae: 1.3215, huber: 0.9808, swd: 3.3399, ept: 87.2406
    Epoch [5/50], Test Losses: mse: 6.0468, mae: 1.3971, huber: 1.0456, swd: 3.4282, ept: 85.7231
      Epoch 5 composite train-obj: 0.753641
            No improvement (0.9808), counter 1/5
    Epoch [6/50], Train Losses: mse: 5.2634, mae: 1.0108, huber: 0.7334, swd: 1.4850, ept: 53.5498
    Epoch [6/50], Val Losses: mse: 1.5422, mae: 0.5526, huber: 0.2991, swd: 1.0543, ept: 94.2825
    Epoch [6/50], Test Losses: mse: 1.4369, mae: 0.5759, huber: 0.3104, swd: 0.9542, ept: 94.1449
      Epoch 6 composite train-obj: 0.733443
            Val objective improved 0.8048 → 0.2991, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.8216, mae: 0.9038, huber: 0.6457, swd: 1.2489, ept: 53.9568
    Epoch [7/50], Val Losses: mse: 2.6414, mae: 0.7283, huber: 0.4383, swd: 2.0155, ept: 93.4843
    Epoch [7/50], Test Losses: mse: 2.5029, mae: 0.7592, huber: 0.4561, swd: 1.8982, ept: 93.1399
      Epoch 7 composite train-obj: 0.645745
            No improvement (0.4383), counter 1/5
    Epoch [8/50], Train Losses: mse: 4.6453, mae: 0.8605, huber: 0.6150, swd: 1.0875, ept: 53.8332
    Epoch [8/50], Val Losses: mse: 5.9359, mae: 1.5899, huber: 1.2135, swd: 4.1968, ept: 90.1769
    Epoch [8/50], Test Losses: mse: 6.1800, mae: 1.6920, huber: 1.3015, swd: 4.3310, ept: 88.9507
      Epoch 8 composite train-obj: 0.614989
            No improvement (1.2135), counter 2/5
    Epoch [9/50], Train Losses: mse: 5.2018, mae: 1.0179, huber: 0.7347, swd: 1.4666, ept: 53.7669
    Epoch [9/50], Val Losses: mse: 3.4283, mae: 1.1916, huber: 0.8191, swd: 2.4161, ept: 92.7386
    Epoch [9/50], Test Losses: mse: 3.4701, mae: 1.2577, huber: 0.8700, swd: 2.4547, ept: 92.5101
      Epoch 9 composite train-obj: 0.734657
            No improvement (0.8191), counter 3/5
    Epoch [10/50], Train Losses: mse: 4.7429, mae: 0.8803, huber: 0.6303, swd: 1.1628, ept: 53.8303
    Epoch [10/50], Val Losses: mse: 4.1298, mae: 1.1991, huber: 0.8353, swd: 3.1921, ept: 93.4911
    Epoch [10/50], Test Losses: mse: 4.0853, mae: 1.2682, huber: 0.8879, swd: 3.1396, ept: 92.9996
      Epoch 10 composite train-obj: 0.630269
            No improvement (0.8353), counter 4/5
    Epoch [11/50], Train Losses: mse: 5.1267, mae: 0.9444, huber: 0.6832, swd: 1.4056, ept: 53.6276
    Epoch [11/50], Val Losses: mse: 2.2093, mae: 0.6784, huber: 0.3864, swd: 1.6536, ept: 94.1936
    Epoch [11/50], Test Losses: mse: 2.0636, mae: 0.7045, huber: 0.4005, swd: 1.5045, ept: 93.6616
      Epoch 11 composite train-obj: 0.683186
    Epoch [11/50], Test Losses: mse: 1.4369, mae: 0.5759, huber: 0.3104, swd: 0.9542, ept: 94.1449
    Best round's Test MSE: 1.4369, MAE: 0.5759, SWD: 0.9542
    Best round's Validation MSE: 1.5422, MAE: 0.5526, SWD: 1.0543
    Best round's Test verification MSE : 1.4369, MAE: 0.5759, SWD: 0.9542
    Time taken: 57.61 seconds
    
    ==================================================
    Experiment Summary (PatchTST_rossler_seq336_pred96_20250513_1150)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 1.3989 ± 0.3620
      mae: 0.5238 ± 0.0649
      huber: 0.2746 ± 0.0538
      swd: 0.9956 ± 0.3167
      ept: 94.0051 ± 0.5426
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 1.4872 ± 0.3995
      mae: 0.4973 ± 0.0622
      huber: 0.2605 ± 0.0516
      swd: 1.0801 ± 0.3517
      ept: 94.2175 ± 0.3935
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 185.16 seconds
    
    Experiment complete: PatchTST_rossler_seq336_pred96_20250513_1150
    Model: PatchTST
    Dataset: rossler
    Sequence Length: 336
    Prediction Length: 96
    Seeds: [1955, 7, 20]
    

#### pred=196


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 281
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 9.5338, mae: 1.7317, huber: 1.3725, swd: 4.6931, ept: 83.2104
    Epoch [1/50], Val Losses: mse: 5.1972, mae: 1.3139, huber: 0.9536, swd: 4.5086, ept: 181.3518
    Epoch [1/50], Test Losses: mse: 4.8572, mae: 1.3198, huber: 0.9503, swd: 4.1906, ept: 180.2401
      Epoch 1 composite train-obj: 1.372531
            Val objective improved inf → 0.9536, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.4513, mae: 1.3582, huber: 1.0284, swd: 3.4106, ept: 88.9921
    Epoch [2/50], Val Losses: mse: 4.0483, mae: 0.9646, huber: 0.6442, swd: 3.4031, ept: 184.1534
    Epoch [2/50], Test Losses: mse: 3.6640, mae: 0.9597, huber: 0.6338, swd: 3.0718, ept: 183.9535
      Epoch 2 composite train-obj: 1.028405
            Val objective improved 0.9536 → 0.6442, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.0699, mae: 1.2546, huber: 0.9410, swd: 3.1488, ept: 89.3357
    Epoch [3/50], Val Losses: mse: 3.8862, mae: 0.8041, huber: 0.5193, swd: 3.2253, ept: 185.0668
    Epoch [3/50], Test Losses: mse: 3.5108, mae: 0.8297, huber: 0.5308, swd: 2.9061, ept: 185.3399
      Epoch 3 composite train-obj: 0.941005
            Val objective improved 0.6442 → 0.5193, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.8524, mae: 1.1933, huber: 0.8910, swd: 2.9887, ept: 89.3813
    Epoch [4/50], Val Losses: mse: 3.7320, mae: 0.7524, huber: 0.4868, swd: 3.1016, ept: 184.9011
    Epoch [4/50], Test Losses: mse: 3.3536, mae: 0.7622, huber: 0.4896, swd: 2.7836, ept: 183.9116
      Epoch 4 composite train-obj: 0.890992
            Val objective improved 0.5193 → 0.4868, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.7335, mae: 1.1723, huber: 0.8729, swd: 2.8924, ept: 89.3590
    Epoch [5/50], Val Losses: mse: 3.5531, mae: 0.7438, huber: 0.4733, swd: 2.9370, ept: 186.1474
    Epoch [5/50], Test Losses: mse: 3.1762, mae: 0.7535, huber: 0.4748, swd: 2.6082, ept: 186.4350
      Epoch 5 composite train-obj: 0.872922
            Val objective improved 0.4868 → 0.4733, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.6141, mae: 1.1399, huber: 0.8477, swd: 2.7908, ept: 89.3779
    Epoch [6/50], Val Losses: mse: 3.5181, mae: 0.8940, huber: 0.5758, swd: 2.9876, ept: 187.3896
    Epoch [6/50], Test Losses: mse: 3.1798, mae: 0.8947, huber: 0.5687, swd: 2.6967, ept: 187.8412
      Epoch 6 composite train-obj: 0.847664
            No improvement (0.5758), counter 1/5
    Epoch [7/50], Train Losses: mse: 6.5704, mae: 1.1348, huber: 0.8431, swd: 2.7568, ept: 89.3948
    Epoch [7/50], Val Losses: mse: 3.2616, mae: 0.6707, huber: 0.4236, swd: 2.7781, ept: 186.9177
    Epoch [7/50], Test Losses: mse: 2.8582, mae: 0.6525, huber: 0.4048, swd: 2.4445, ept: 186.8382
      Epoch 7 composite train-obj: 0.843088
            Val objective improved 0.4733 → 0.4236, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 6.4507, mae: 1.0993, huber: 0.8170, swd: 2.6612, ept: 89.3760
    Epoch [8/50], Val Losses: mse: 3.0474, mae: 0.7157, huber: 0.4386, swd: 2.6056, ept: 186.8550
    Epoch [8/50], Test Losses: mse: 2.7184, mae: 0.7187, huber: 0.4314, swd: 2.3363, ept: 187.2536
      Epoch 8 composite train-obj: 0.817029
            No improvement (0.4386), counter 1/5
    Epoch [9/50], Train Losses: mse: 6.3891, mae: 1.0817, huber: 0.8040, swd: 2.6048, ept: 89.3525
    Epoch [9/50], Val Losses: mse: 3.1276, mae: 0.7762, huber: 0.4837, swd: 2.6105, ept: 186.8452
    Epoch [9/50], Test Losses: mse: 2.8158, mae: 0.7774, huber: 0.4781, swd: 2.3578, ept: 186.8098
      Epoch 9 composite train-obj: 0.804017
            No improvement (0.4837), counter 2/5
    Epoch [10/50], Train Losses: mse: 6.3539, mae: 1.0746, huber: 0.7991, swd: 2.5773, ept: 89.3443
    Epoch [10/50], Val Losses: mse: 3.2477, mae: 0.7423, huber: 0.4692, swd: 2.7243, ept: 187.0799
    Epoch [10/50], Test Losses: mse: 3.0069, mae: 0.7702, huber: 0.4837, swd: 2.5239, ept: 186.8224
      Epoch 10 composite train-obj: 0.799086
            No improvement (0.4692), counter 3/5
    Epoch [11/50], Train Losses: mse: 6.3441, mae: 1.0747, huber: 0.7982, swd: 2.5689, ept: 89.3857
    Epoch [11/50], Val Losses: mse: 3.0376, mae: 0.7506, huber: 0.4576, swd: 2.5983, ept: 187.5427
    Epoch [11/50], Test Losses: mse: 2.7075, mae: 0.7528, huber: 0.4511, swd: 2.3220, ept: 187.9416
      Epoch 11 composite train-obj: 0.798169
            No improvement (0.4576), counter 4/5
    Epoch [12/50], Train Losses: mse: 6.2487, mae: 1.0533, huber: 0.7819, swd: 2.4884, ept: 89.3561
    Epoch [12/50], Val Losses: mse: 3.4736, mae: 0.8882, huber: 0.5760, swd: 3.0408, ept: 187.2730
    Epoch [12/50], Test Losses: mse: 3.1232, mae: 0.8873, huber: 0.5690, swd: 2.7492, ept: 186.6891
      Epoch 12 composite train-obj: 0.781945
    Epoch [12/50], Test Losses: mse: 2.8582, mae: 0.6525, huber: 0.4048, swd: 2.4445, ept: 186.8382
    Best round's Test MSE: 2.8582, MAE: 0.6525, SWD: 2.4445
    Best round's Validation MSE: 3.2616, MAE: 0.6707, SWD: 2.7781
    Best round's Test verification MSE : 2.8582, MAE: 0.6525, SWD: 2.4445
    Time taken: 64.33 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.5733, mae: 1.7465, huber: 1.3849, swd: 4.7481, ept: 83.4327
    Epoch [1/50], Val Losses: mse: 5.4693, mae: 1.0889, huber: 0.7762, swd: 4.3843, ept: 177.5548
    Epoch [1/50], Test Losses: mse: 5.0883, mae: 1.1351, huber: 0.8101, swd: 4.0791, ept: 175.3895
      Epoch 1 composite train-obj: 1.384948
            Val objective improved inf → 0.7762, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.3871, mae: 1.3481, huber: 1.0185, swd: 3.3715, ept: 88.9991
    Epoch [2/50], Val Losses: mse: 4.0181, mae: 1.0019, huber: 0.6788, swd: 3.4766, ept: 186.7428
    Epoch [2/50], Test Losses: mse: 3.6550, mae: 0.9953, huber: 0.6663, swd: 3.1446, ept: 186.5352
      Epoch 2 composite train-obj: 1.018532
            Val objective improved 0.7762 → 0.6788, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.1046, mae: 1.2747, huber: 0.9563, swd: 3.1858, ept: 89.2084
    Epoch [3/50], Val Losses: mse: 3.8727, mae: 0.9201, huber: 0.6038, swd: 3.2967, ept: 186.4721
    Epoch [3/50], Test Losses: mse: 3.4897, mae: 0.9301, huber: 0.6016, swd: 2.9535, ept: 186.4438
      Epoch 3 composite train-obj: 0.956327
            Val objective improved 0.6788 → 0.6038, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.9128, mae: 1.2178, huber: 0.9098, swd: 3.0559, ept: 89.3450
    Epoch [4/50], Val Losses: mse: 3.5869, mae: 0.8002, huber: 0.5226, swd: 3.0731, ept: 185.7068
    Epoch [4/50], Test Losses: mse: 3.1800, mae: 0.7858, huber: 0.5046, swd: 2.7216, ept: 185.7396
      Epoch 4 composite train-obj: 0.909808
            Val objective improved 0.6038 → 0.5226, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 6.7533, mae: 1.1723, huber: 0.8737, swd: 2.9333, ept: 89.3460
    Epoch [5/50], Val Losses: mse: 3.5900, mae: 0.9102, huber: 0.5900, swd: 3.1651, ept: 186.7366
    Epoch [5/50], Test Losses: mse: 3.2195, mae: 0.9052, huber: 0.5780, swd: 2.8469, ept: 186.4883
      Epoch 5 composite train-obj: 0.873692
            No improvement (0.5900), counter 1/5
    Epoch [6/50], Train Losses: mse: 6.6597, mae: 1.1459, huber: 0.8538, swd: 2.8526, ept: 89.3821
    Epoch [6/50], Val Losses: mse: 3.7300, mae: 1.0217, huber: 0.6843, swd: 3.2854, ept: 187.6463
    Epoch [6/50], Test Losses: mse: 3.4286, mae: 1.0239, huber: 0.6826, swd: 3.0298, ept: 187.4730
      Epoch 6 composite train-obj: 0.853786
            No improvement (0.6843), counter 2/5
    Epoch [7/50], Train Losses: mse: 6.5678, mae: 1.1273, huber: 0.8390, swd: 2.7775, ept: 89.3266
    Epoch [7/50], Val Losses: mse: 3.2908, mae: 0.7089, huber: 0.4607, swd: 2.7923, ept: 187.1001
    Epoch [7/50], Test Losses: mse: 2.9201, mae: 0.7068, huber: 0.4498, swd: 2.4701, ept: 187.4333
      Epoch 7 composite train-obj: 0.838985
            Val objective improved 0.5226 → 0.4607, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 6.5246, mae: 1.1187, huber: 0.8318, swd: 2.7481, ept: 89.3955
    Epoch [8/50], Val Losses: mse: 3.2260, mae: 0.7754, huber: 0.4765, swd: 2.7412, ept: 187.2439
    Epoch [8/50], Test Losses: mse: 2.8870, mae: 0.7793, huber: 0.4715, swd: 2.4440, ept: 187.2716
      Epoch 8 composite train-obj: 0.831826
            No improvement (0.4765), counter 1/5
    Epoch [9/50], Train Losses: mse: 6.4183, mae: 1.0876, huber: 0.8090, swd: 2.6577, ept: 89.3440
    Epoch [9/50], Val Losses: mse: 3.1515, mae: 0.7060, huber: 0.4375, swd: 2.6958, ept: 186.3590
    Epoch [9/50], Test Losses: mse: 2.7797, mae: 0.7013, huber: 0.4263, swd: 2.3795, ept: 187.0156
      Epoch 9 composite train-obj: 0.809036
            Val objective improved 0.4607 → 0.4375, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 6.4041, mae: 1.0821, huber: 0.8046, swd: 2.6348, ept: 89.3300
    Epoch [10/50], Val Losses: mse: 3.1726, mae: 0.7433, huber: 0.4606, swd: 2.7088, ept: 188.3799
    Epoch [10/50], Test Losses: mse: 2.8405, mae: 0.7525, huber: 0.4581, swd: 2.4225, ept: 187.9713
      Epoch 10 composite train-obj: 0.804631
            No improvement (0.4606), counter 1/5
    Epoch [11/50], Train Losses: mse: 6.2925, mae: 1.0532, huber: 0.7837, swd: 2.5495, ept: 89.4147
    Epoch [11/50], Val Losses: mse: 3.0875, mae: 0.6721, huber: 0.4211, swd: 2.6865, ept: 187.1983
    Epoch [11/50], Test Losses: mse: 2.7193, mae: 0.6641, huber: 0.4086, swd: 2.3660, ept: 187.2249
      Epoch 11 composite train-obj: 0.783685
            Val objective improved 0.4375 → 0.4211, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 6.2706, mae: 1.0497, huber: 0.7812, swd: 2.5279, ept: 89.4476
    Epoch [12/50], Val Losses: mse: 3.1567, mae: 0.6687, huber: 0.4182, swd: 2.7243, ept: 186.5213
    Epoch [12/50], Test Losses: mse: 2.7909, mae: 0.6694, huber: 0.4107, swd: 2.4187, ept: 186.7314
      Epoch 12 composite train-obj: 0.781205
            Val objective improved 0.4211 → 0.4182, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 6.2338, mae: 1.0431, huber: 0.7758, swd: 2.4887, ept: 89.4003
    Epoch [13/50], Val Losses: mse: 3.0434, mae: 0.7318, huber: 0.4545, swd: 2.5669, ept: 187.1653
    Epoch [13/50], Test Losses: mse: 2.6935, mae: 0.7288, huber: 0.4440, swd: 2.2706, ept: 187.0045
      Epoch 13 composite train-obj: 0.775822
            No improvement (0.4545), counter 1/5
    Epoch [14/50], Train Losses: mse: 6.1674, mae: 1.0317, huber: 0.7677, swd: 2.4259, ept: 89.5281
    Epoch [14/50], Val Losses: mse: 3.4200, mae: 0.6863, huber: 0.4266, swd: 2.8647, ept: 186.8148
    Epoch [14/50], Test Losses: mse: 3.0369, mae: 0.6962, huber: 0.4296, swd: 2.5380, ept: 186.2334
      Epoch 14 composite train-obj: 0.767722
            No improvement (0.4266), counter 2/5
    Epoch [15/50], Train Losses: mse: 6.0876, mae: 1.0216, huber: 0.7586, swd: 2.3547, ept: 89.4668
    Epoch [15/50], Val Losses: mse: 2.9608, mae: 0.7208, huber: 0.4404, swd: 2.5334, ept: 187.3996
    Epoch [15/50], Test Losses: mse: 2.5954, mae: 0.7127, huber: 0.4261, swd: 2.2234, ept: 187.7388
      Epoch 15 composite train-obj: 0.758639
            No improvement (0.4404), counter 3/5
    Epoch [16/50], Train Losses: mse: 6.0849, mae: 1.0257, huber: 0.7615, swd: 2.3386, ept: 89.4677
    Epoch [16/50], Val Losses: mse: 3.1176, mae: 0.6332, huber: 0.4034, swd: 2.4896, ept: 185.7363
    Epoch [16/50], Test Losses: mse: 2.8738, mae: 0.6736, huber: 0.4260, swd: 2.2893, ept: 185.3872
      Epoch 16 composite train-obj: 0.761489
            Val objective improved 0.4182 → 0.4034, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 6.0452, mae: 1.0090, huber: 0.7509, swd: 2.3043, ept: 89.3737
    Epoch [17/50], Val Losses: mse: 3.4222, mae: 0.6747, huber: 0.4276, swd: 2.9380, ept: 185.8088
    Epoch [17/50], Test Losses: mse: 3.0514, mae: 0.6799, huber: 0.4254, swd: 2.6192, ept: 185.1081
      Epoch 17 composite train-obj: 0.750868
            No improvement (0.4276), counter 1/5
    Epoch [18/50], Train Losses: mse: 5.9750, mae: 1.0033, huber: 0.7453, swd: 2.2372, ept: 89.4639
    Epoch [18/50], Val Losses: mse: 2.6262, mae: 0.5591, huber: 0.3422, swd: 2.1750, ept: 188.4555
    Epoch [18/50], Test Losses: mse: 2.3195, mae: 0.5627, huber: 0.3402, swd: 1.9218, ept: 188.8144
      Epoch 18 composite train-obj: 0.745304
            Val objective improved 0.4034 → 0.3422, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 5.9578, mae: 0.9985, huber: 0.7415, swd: 2.2037, ept: 89.4537
    Epoch [19/50], Val Losses: mse: 2.6905, mae: 0.5952, huber: 0.3571, swd: 2.1756, ept: 187.5728
    Epoch [19/50], Test Losses: mse: 2.3588, mae: 0.5927, huber: 0.3492, swd: 1.9012, ept: 187.6864
      Epoch 19 composite train-obj: 0.741505
            No improvement (0.3571), counter 1/5
    Epoch [20/50], Train Losses: mse: 5.9042, mae: 0.9871, huber: 0.7336, swd: 2.1492, ept: 89.4133
    Epoch [20/50], Val Losses: mse: 2.8212, mae: 0.6137, huber: 0.3753, swd: 2.3330, ept: 187.9238
    Epoch [20/50], Test Losses: mse: 2.4644, mae: 0.6050, huber: 0.3620, swd: 2.0363, ept: 187.3880
      Epoch 20 composite train-obj: 0.733554
            No improvement (0.3753), counter 2/5
    Epoch [21/50], Train Losses: mse: 5.8211, mae: 0.9824, huber: 0.7281, swd: 2.0688, ept: 89.5957
    Epoch [21/50], Val Losses: mse: 2.6414, mae: 0.6493, huber: 0.3864, swd: 2.0413, ept: 187.3449
    Epoch [21/50], Test Losses: mse: 2.3463, mae: 0.6513, huber: 0.3805, swd: 1.7942, ept: 187.5370
      Epoch 21 composite train-obj: 0.728074
            No improvement (0.3864), counter 3/5
    Epoch [22/50], Train Losses: mse: 5.7944, mae: 0.9778, huber: 0.7254, swd: 2.0282, ept: 89.5474
    Epoch [22/50], Val Losses: mse: 2.3035, mae: 0.6536, huber: 0.3888, swd: 1.8836, ept: 189.5500
    Epoch [22/50], Test Losses: mse: 2.0381, mae: 0.6450, huber: 0.3762, swd: 1.6684, ept: 190.1094
      Epoch 22 composite train-obj: 0.725423
            No improvement (0.3888), counter 4/5
    Epoch [23/50], Train Losses: mse: 5.7098, mae: 0.9696, huber: 0.7180, swd: 1.9313, ept: 89.4768
    Epoch [23/50], Val Losses: mse: 2.7252, mae: 0.6806, huber: 0.4130, swd: 2.0594, ept: 187.5012
    Epoch [23/50], Test Losses: mse: 2.5245, mae: 0.7104, huber: 0.4281, swd: 1.9021, ept: 187.1602
      Epoch 23 composite train-obj: 0.717956
    Epoch [23/50], Test Losses: mse: 2.3195, mae: 0.5627, huber: 0.3402, swd: 1.9218, ept: 188.8144
    Best round's Test MSE: 2.3195, MAE: 0.5627, SWD: 1.9218
    Best round's Validation MSE: 2.6262, MAE: 0.5591, SWD: 2.1750
    Best round's Test verification MSE : 2.3195, MAE: 0.5627, SWD: 1.9218
    Time taken: 125.51 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 9.3971, mae: 1.7145, huber: 1.3561, swd: 4.1702, ept: 83.5074
    Epoch [1/50], Val Losses: mse: 4.2960, mae: 1.0361, huber: 0.7058, swd: 3.3575, ept: 184.2190
    Epoch [1/50], Test Losses: mse: 3.9031, mae: 1.0363, huber: 0.6981, swd: 3.0245, ept: 184.0486
      Epoch 1 composite train-obj: 1.356103
            Val objective improved inf → 0.7058, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.2695, mae: 1.3182, huber: 0.9933, swd: 3.0021, ept: 89.0210
    Epoch [2/50], Val Losses: mse: 4.3250, mae: 1.0994, huber: 0.7570, swd: 3.4366, ept: 185.8342
    Epoch [2/50], Test Losses: mse: 3.9833, mae: 1.1037, huber: 0.7530, swd: 3.1543, ept: 185.6418
      Epoch 2 composite train-obj: 0.993336
            No improvement (0.7570), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.0237, mae: 1.2439, huber: 0.9317, swd: 2.8707, ept: 89.3675
    Epoch [3/50], Val Losses: mse: 4.1427, mae: 0.7964, huber: 0.5296, swd: 3.0872, ept: 183.5507
    Epoch [3/50], Test Losses: mse: 3.7604, mae: 0.8116, huber: 0.5386, swd: 2.7909, ept: 182.7826
      Epoch 3 composite train-obj: 0.931730
            Val objective improved 0.7058 → 0.5296, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 6.8587, mae: 1.1954, huber: 0.8930, swd: 2.7578, ept: 89.2671
    Epoch [4/50], Val Losses: mse: 3.8156, mae: 0.8408, huber: 0.5446, swd: 2.9627, ept: 185.4533
    Epoch [4/50], Test Losses: mse: 3.4205, mae: 0.8573, huber: 0.5455, swd: 2.6437, ept: 184.8436
      Epoch 4 composite train-obj: 0.892979
            No improvement (0.5446), counter 1/5
    Epoch [5/50], Train Losses: mse: 6.7322, mae: 1.1550, huber: 0.8612, swd: 2.6735, ept: 89.3107
    Epoch [5/50], Val Losses: mse: 3.4545, mae: 0.7861, huber: 0.4986, swd: 2.6855, ept: 185.9352
    Epoch [5/50], Test Losses: mse: 3.0652, mae: 0.7703, huber: 0.4826, swd: 2.3864, ept: 186.3958
      Epoch 5 composite train-obj: 0.861217
            Val objective improved 0.5296 → 0.4986, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 6.6447, mae: 1.1338, huber: 0.8452, swd: 2.6039, ept: 89.3561
    Epoch [6/50], Val Losses: mse: 3.8544, mae: 0.7993, huber: 0.5245, swd: 2.8112, ept: 185.9663
    Epoch [6/50], Test Losses: mse: 3.5587, mae: 0.8344, huber: 0.5483, swd: 2.5687, ept: 185.4689
      Epoch 6 composite train-obj: 0.845164
            No improvement (0.5245), counter 1/5
    Epoch [7/50], Train Losses: mse: 6.5686, mae: 1.1159, huber: 0.8314, swd: 2.5479, ept: 89.3215
    Epoch [7/50], Val Losses: mse: 3.4516, mae: 0.8418, huber: 0.5392, swd: 2.7235, ept: 186.3255
    Epoch [7/50], Test Losses: mse: 3.0787, mae: 0.8322, huber: 0.5238, swd: 2.4274, ept: 185.7492
      Epoch 7 composite train-obj: 0.831438
            No improvement (0.5392), counter 2/5
    Epoch [8/50], Train Losses: mse: 6.4939, mae: 1.1046, huber: 0.8220, swd: 2.4883, ept: 89.4618
    Epoch [8/50], Val Losses: mse: 3.4293, mae: 0.8584, huber: 0.5523, swd: 2.6949, ept: 186.8761
    Epoch [8/50], Test Losses: mse: 3.1242, mae: 0.8622, huber: 0.5518, swd: 2.4603, ept: 186.1903
      Epoch 8 composite train-obj: 0.821961
            No improvement (0.5523), counter 3/5
    Epoch [9/50], Train Losses: mse: 6.4562, mae: 1.0891, huber: 0.8111, swd: 2.4510, ept: 89.3038
    Epoch [9/50], Val Losses: mse: 3.4759, mae: 0.9056, huber: 0.5849, swd: 2.7008, ept: 186.9217
    Epoch [9/50], Test Losses: mse: 3.1402, mae: 0.9055, huber: 0.5781, swd: 2.4341, ept: 186.6660
      Epoch 9 composite train-obj: 0.811105
            No improvement (0.5849), counter 4/5
    Epoch [10/50], Train Losses: mse: 6.4041, mae: 1.0811, huber: 0.8044, swd: 2.4173, ept: 89.4026
    Epoch [10/50], Val Losses: mse: 3.1757, mae: 0.7687, huber: 0.4853, swd: 2.4760, ept: 186.9512
    Epoch [10/50], Test Losses: mse: 2.8227, mae: 0.7551, huber: 0.4701, swd: 2.1986, ept: 187.2275
      Epoch 10 composite train-obj: 0.804432
            Val objective improved 0.4986 → 0.4853, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 6.3516, mae: 1.0659, huber: 0.7935, swd: 2.3766, ept: 89.3575
    Epoch [11/50], Val Losses: mse: 3.2408, mae: 0.8632, huber: 0.5428, swd: 2.5751, ept: 187.9531
    Epoch [11/50], Test Losses: mse: 2.8919, mae: 0.8573, huber: 0.5305, swd: 2.2939, ept: 188.1550
      Epoch 11 composite train-obj: 0.793486
            No improvement (0.5428), counter 1/5
    Epoch [12/50], Train Losses: mse: 6.2744, mae: 1.0559, huber: 0.7847, swd: 2.3118, ept: 89.3958
    Epoch [12/50], Val Losses: mse: 3.2576, mae: 0.8708, huber: 0.5563, swd: 2.5039, ept: 187.1922
    Epoch [12/50], Test Losses: mse: 2.9213, mae: 0.8654, huber: 0.5452, swd: 2.2232, ept: 187.3613
      Epoch 12 composite train-obj: 0.784718
            No improvement (0.5563), counter 2/5
    Epoch [13/50], Train Losses: mse: 6.2419, mae: 1.0474, huber: 0.7788, swd: 2.2869, ept: 89.3967
    Epoch [13/50], Val Losses: mse: 3.4573, mae: 0.6937, huber: 0.4411, swd: 2.5986, ept: 185.5851
    Epoch [13/50], Test Losses: mse: 3.1197, mae: 0.7212, huber: 0.4553, swd: 2.3271, ept: 185.3942
      Epoch 13 composite train-obj: 0.778798
            Val objective improved 0.4853 → 0.4411, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 6.1843, mae: 1.0379, huber: 0.7712, swd: 2.2426, ept: 89.5071
    Epoch [14/50], Val Losses: mse: 2.9974, mae: 0.7667, huber: 0.4729, swd: 2.3076, ept: 186.5113
    Epoch [14/50], Test Losses: mse: 2.6681, mae: 0.7677, huber: 0.4658, swd: 2.0415, ept: 187.1896
      Epoch 14 composite train-obj: 0.771224
            No improvement (0.4729), counter 1/5
    Epoch [15/50], Train Losses: mse: 6.1339, mae: 1.0180, huber: 0.7587, swd: 2.1875, ept: 89.3613
    Epoch [15/50], Val Losses: mse: 2.8802, mae: 0.6923, huber: 0.4206, swd: 2.1721, ept: 187.1050
    Epoch [15/50], Test Losses: mse: 2.5421, mae: 0.6897, huber: 0.4092, swd: 1.9039, ept: 187.9061
      Epoch 15 composite train-obj: 0.758707
            Val objective improved 0.4411 → 0.4206, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 6.0904, mae: 1.0168, huber: 0.7564, swd: 2.1481, ept: 89.3966
    Epoch [16/50], Val Losses: mse: 3.0067, mae: 0.8005, huber: 0.5112, swd: 2.3652, ept: 188.1881
    Epoch [16/50], Test Losses: mse: 2.7191, mae: 0.8044, huber: 0.5081, swd: 2.1361, ept: 187.0459
      Epoch 16 composite train-obj: 0.756431
            No improvement (0.5112), counter 1/5
    Epoch [17/50], Train Losses: mse: 6.0363, mae: 1.0127, huber: 0.7527, swd: 2.0966, ept: 89.4742
    Epoch [17/50], Val Losses: mse: 3.0640, mae: 0.6511, huber: 0.4006, swd: 2.3538, ept: 187.2387
    Epoch [17/50], Test Losses: mse: 2.7019, mae: 0.6513, huber: 0.3941, swd: 2.0700, ept: 186.4095
      Epoch 17 composite train-obj: 0.752720
            Val objective improved 0.4206 → 0.4006, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 6.0127, mae: 1.0011, huber: 0.7454, swd: 2.0769, ept: 89.4116
    Epoch [18/50], Val Losses: mse: 3.1373, mae: 0.7643, huber: 0.4809, swd: 2.4872, ept: 187.4298
    Epoch [18/50], Test Losses: mse: 2.7949, mae: 0.7641, huber: 0.4733, swd: 2.2154, ept: 186.3186
      Epoch 18 composite train-obj: 0.745364
            No improvement (0.4809), counter 1/5
    Epoch [19/50], Train Losses: mse: 6.0001, mae: 1.0034, huber: 0.7467, swd: 2.0546, ept: 89.4513
    Epoch [19/50], Val Losses: mse: 2.4856, mae: 0.6632, huber: 0.3914, swd: 1.9429, ept: 189.3265
    Epoch [19/50], Test Losses: mse: 2.1731, mae: 0.6517, huber: 0.3725, swd: 1.6979, ept: 189.9449
      Epoch 19 composite train-obj: 0.746738
            Val objective improved 0.4006 → 0.3914, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 5.9083, mae: 0.9920, huber: 0.7367, swd: 1.9840, ept: 89.5734
    Epoch [20/50], Val Losses: mse: 2.5840, mae: 0.5777, huber: 0.3461, swd: 1.9076, ept: 187.9025
    Epoch [20/50], Test Losses: mse: 2.2635, mae: 0.5797, huber: 0.3384, swd: 1.6681, ept: 188.3593
      Epoch 20 composite train-obj: 0.736744
            Val objective improved 0.3914 → 0.3461, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 5.8664, mae: 0.9802, huber: 0.7294, swd: 1.9362, ept: 89.4686
    Epoch [21/50], Val Losses: mse: 2.6547, mae: 0.5742, huber: 0.3489, swd: 2.0241, ept: 187.1997
    Epoch [21/50], Test Losses: mse: 2.3206, mae: 0.5713, huber: 0.3405, swd: 1.7640, ept: 186.6923
      Epoch 21 composite train-obj: 0.729426
            No improvement (0.3489), counter 1/5
    Epoch [22/50], Train Losses: mse: 5.8088, mae: 0.9735, huber: 0.7235, swd: 1.8861, ept: 89.5803
    Epoch [22/50], Val Losses: mse: 2.3178, mae: 0.5810, huber: 0.3322, swd: 1.6489, ept: 189.2242
    Epoch [22/50], Test Losses: mse: 2.0415, mae: 0.5804, huber: 0.3267, swd: 1.4395, ept: 189.7985
      Epoch 22 composite train-obj: 0.723465
            Val objective improved 0.3461 → 0.3322, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 5.7686, mae: 0.9698, huber: 0.7206, swd: 1.8408, ept: 89.4765
    Epoch [23/50], Val Losses: mse: 2.3489, mae: 0.5506, huber: 0.3311, swd: 1.6700, ept: 188.0976
    Epoch [23/50], Test Losses: mse: 2.0544, mae: 0.5503, huber: 0.3234, swd: 1.4437, ept: 189.1648
      Epoch 23 composite train-obj: 0.720620
            Val objective improved 0.3322 → 0.3311, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 5.7232, mae: 0.9685, huber: 0.7184, swd: 1.7876, ept: 89.5279
    Epoch [24/50], Val Losses: mse: 2.1392, mae: 0.6409, huber: 0.3759, swd: 1.5532, ept: 190.2560
    Epoch [24/50], Test Losses: mse: 1.9313, mae: 0.6430, huber: 0.3732, swd: 1.4052, ept: 189.6379
      Epoch 24 composite train-obj: 0.718368
            No improvement (0.3759), counter 1/5
    Epoch [25/50], Train Losses: mse: 5.6603, mae: 0.9617, huber: 0.7130, swd: 1.7292, ept: 89.4729
    Epoch [25/50], Val Losses: mse: 2.4226, mae: 0.6490, huber: 0.3808, swd: 1.7796, ept: 188.0844
    Epoch [25/50], Test Losses: mse: 2.1175, mae: 0.6417, huber: 0.3657, swd: 1.5445, ept: 187.4042
      Epoch 25 composite train-obj: 0.712959
            No improvement (0.3808), counter 2/5
    Epoch [26/50], Train Losses: mse: 5.6311, mae: 0.9589, huber: 0.7102, swd: 1.6916, ept: 89.5437
    Epoch [26/50], Val Losses: mse: 1.9643, mae: 0.4987, huber: 0.2865, swd: 1.4007, ept: 189.2656
    Epoch [26/50], Test Losses: mse: 1.6981, mae: 0.4891, huber: 0.2730, swd: 1.2094, ept: 189.9984
      Epoch 26 composite train-obj: 0.710178
            Val objective improved 0.3311 → 0.2865, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 5.5763, mae: 0.9522, huber: 0.7046, swd: 1.6378, ept: 89.4664
    Epoch [27/50], Val Losses: mse: 2.5492, mae: 0.6949, huber: 0.4205, swd: 1.8163, ept: 187.9015
    Epoch [27/50], Test Losses: mse: 2.2849, mae: 0.7013, huber: 0.4204, swd: 1.6180, ept: 187.2543
      Epoch 27 composite train-obj: 0.704645
            No improvement (0.4205), counter 1/5
    Epoch [28/50], Train Losses: mse: 5.5662, mae: 0.9551, huber: 0.7061, swd: 1.6184, ept: 89.4634
    Epoch [28/50], Val Losses: mse: 2.5408, mae: 0.6892, huber: 0.4109, swd: 1.8663, ept: 186.9141
    Epoch [28/50], Test Losses: mse: 2.2553, mae: 0.6985, huber: 0.4110, swd: 1.6509, ept: 186.9570
      Epoch 28 composite train-obj: 0.706069
            No improvement (0.4109), counter 2/5
    Epoch [29/50], Train Losses: mse: 5.4589, mae: 0.9421, huber: 0.6957, swd: 1.5200, ept: 89.5354
    Epoch [29/50], Val Losses: mse: 1.9162, mae: 0.6671, huber: 0.3762, swd: 1.3101, ept: 189.7180
    Epoch [29/50], Test Losses: mse: 1.6661, mae: 0.6531, huber: 0.3567, swd: 1.1265, ept: 190.5451
      Epoch 29 composite train-obj: 0.695712
            No improvement (0.3762), counter 3/5
    Epoch [30/50], Train Losses: mse: 5.4079, mae: 0.9362, huber: 0.6915, swd: 1.4771, ept: 89.4974
    Epoch [30/50], Val Losses: mse: 2.4797, mae: 0.8600, huber: 0.5323, swd: 1.8669, ept: 188.5626
    Epoch [30/50], Test Losses: mse: 2.2501, mae: 0.8640, huber: 0.5288, swd: 1.7006, ept: 187.7958
      Epoch 30 composite train-obj: 0.691510
            No improvement (0.5323), counter 4/5
    Epoch [31/50], Train Losses: mse: 5.3627, mae: 0.9317, huber: 0.6872, swd: 1.4260, ept: 89.4760
    Epoch [31/50], Val Losses: mse: 2.1402, mae: 0.7898, huber: 0.4803, swd: 1.2800, ept: 188.3302
    Epoch [31/50], Test Losses: mse: 1.9557, mae: 0.7966, huber: 0.4782, swd: 1.1339, ept: 189.0091
      Epoch 31 composite train-obj: 0.687179
    Epoch [31/50], Test Losses: mse: 1.6981, mae: 0.4891, huber: 0.2730, swd: 1.2094, ept: 189.9984
    Best round's Test MSE: 1.6981, MAE: 0.4891, SWD: 1.2094
    Best round's Validation MSE: 1.9643, MAE: 0.4987, SWD: 1.4007
    Best round's Test verification MSE : 1.6981, MAE: 0.4891, SWD: 1.2094
    Time taken: 168.30 seconds
    
    ==================================================
    Experiment Summary (PatchTST_rossler_seq336_pred196_20250513_1153)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 2.2919 ± 0.4740
      mae: 0.5681 ± 0.0668
      huber: 0.3393 ± 0.0538
      swd: 1.8586 ± 0.5062
      ept: 188.5503 ± 1.3036
      count: 37.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 2.6174 ± 0.5297
      mae: 0.5762 ± 0.0713
      huber: 0.3508 ± 0.0563
      swd: 2.1179 ± 0.5637
      ept: 188.2129 ± 0.9738
      count: 37.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 358.22 seconds
    
    Experiment complete: PatchTST_rossler_seq336_pred196_20250513_1153
    Model: PatchTST
    Dataset: rossler
    Sequence Length: 336
    Prediction Length: 196
    Seeds: [1955, 7, 20]
    

#### pred=336


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 280
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 10.5670, mae: 1.9265, huber: 1.5562, swd: 4.9753, ept: 114.4429
    Epoch [1/50], Val Losses: mse: 9.6809, mae: 1.6681, huber: 1.3261, swd: 7.0693, ept: 276.5746
    Epoch [1/50], Test Losses: mse: 9.3029, mae: 1.6780, huber: 1.3298, swd: 6.9071, ept: 271.7136
      Epoch 1 composite train-obj: 1.556239
            Val objective improved inf → 1.3261, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.0797, mae: 1.5009, huber: 1.1584, swd: 3.6101, ept: 125.3282
    Epoch [2/50], Val Losses: mse: 5.6772, mae: 1.1267, huber: 0.8052, swd: 3.8777, ept: 297.3684
    Epoch [2/50], Test Losses: mse: 4.9662, mae: 1.1050, huber: 0.7755, swd: 3.3775, ept: 297.2017
      Epoch 2 composite train-obj: 1.158399
            Val objective improved 1.3261 → 0.8052, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.6903, mae: 1.3949, huber: 1.0663, swd: 3.3989, ept: 126.0713
    Epoch [3/50], Val Losses: mse: 5.1838, mae: 1.1489, huber: 0.8190, swd: 3.8012, ept: 304.2477
    Epoch [3/50], Test Losses: mse: 4.5435, mae: 1.1080, huber: 0.7729, swd: 3.3555, ept: 304.6110
      Epoch 3 composite train-obj: 1.066262
            No improvement (0.8190), counter 1/5
    Epoch [4/50], Train Losses: mse: 7.4993, mae: 1.3461, huber: 1.0252, swd: 3.2891, ept: 126.3602
    Epoch [4/50], Val Losses: mse: 5.3179, mae: 1.1555, huber: 0.8177, swd: 3.6919, ept: 301.1585
    Epoch [4/50], Test Losses: mse: 4.6842, mae: 1.1335, huber: 0.7878, swd: 3.2730, ept: 302.8235
      Epoch 4 composite train-obj: 1.025235
            No improvement (0.8177), counter 2/5
    Epoch [5/50], Train Losses: mse: 7.3862, mae: 1.3165, huber: 1.0002, swd: 3.2314, ept: 126.4229
    Epoch [5/50], Val Losses: mse: 4.7673, mae: 1.0283, huber: 0.7166, swd: 3.4758, ept: 306.0220
    Epoch [5/50], Test Losses: mse: 4.1477, mae: 0.9920, huber: 0.6750, swd: 3.0451, ept: 306.0520
      Epoch 5 composite train-obj: 1.000239
            Val objective improved 0.8052 → 0.7166, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.2566, mae: 1.2788, huber: 0.9709, swd: 3.1377, ept: 126.4110
    Epoch [6/50], Val Losses: mse: 5.0931, mae: 1.0593, huber: 0.7388, swd: 3.4670, ept: 303.1810
    Epoch [6/50], Test Losses: mse: 4.4554, mae: 1.0410, huber: 0.7120, swd: 3.0312, ept: 304.1587
      Epoch 6 composite train-obj: 0.970864
            No improvement (0.7388), counter 1/5
    Epoch [7/50], Train Losses: mse: 7.1661, mae: 1.2535, huber: 0.9505, swd: 3.0723, ept: 126.4949
    Epoch [7/50], Val Losses: mse: 4.6481, mae: 0.9614, huber: 0.6594, swd: 3.3114, ept: 304.8289
    Epoch [7/50], Test Losses: mse: 4.0166, mae: 0.9270, huber: 0.6185, swd: 2.8793, ept: 306.8570
      Epoch 7 composite train-obj: 0.950480
            Val objective improved 0.7166 → 0.6594, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 7.1252, mae: 1.2402, huber: 0.9400, swd: 3.0482, ept: 126.3254
    Epoch [8/50], Val Losses: mse: 4.7281, mae: 1.0036, huber: 0.6881, swd: 3.3768, ept: 302.5961
    Epoch [8/50], Test Losses: mse: 4.1060, mae: 0.9661, huber: 0.6488, swd: 2.9509, ept: 303.2418
      Epoch 8 composite train-obj: 0.940019
            No improvement (0.6881), counter 1/5
    Epoch [9/50], Train Losses: mse: 7.0543, mae: 1.2229, huber: 0.9271, swd: 3.0068, ept: 126.5617
    Epoch [9/50], Val Losses: mse: 4.9549, mae: 1.0571, huber: 0.7269, swd: 3.3935, ept: 304.6071
    Epoch [9/50], Test Losses: mse: 4.3391, mae: 1.0412, huber: 0.7026, swd: 2.9747, ept: 305.3280
      Epoch 9 composite train-obj: 0.927123
            No improvement (0.7269), counter 2/5
    Epoch [10/50], Train Losses: mse: 7.0548, mae: 1.2231, huber: 0.9271, swd: 3.0044, ept: 126.1749
    Epoch [10/50], Val Losses: mse: 4.9821, mae: 0.8904, huber: 0.6242, swd: 3.4172, ept: 303.8348
    Epoch [10/50], Test Losses: mse: 4.3253, mae: 0.8755, huber: 0.6012, swd: 2.9965, ept: 304.4128
      Epoch 10 composite train-obj: 0.927100
            Val objective improved 0.6594 → 0.6242, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 6.9987, mae: 1.2045, huber: 0.9131, swd: 2.9637, ept: 126.4206
    Epoch [11/50], Val Losses: mse: 5.0804, mae: 1.0286, huber: 0.7158, swd: 3.6903, ept: 303.2875
    Epoch [11/50], Test Losses: mse: 4.6109, mae: 1.0172, huber: 0.6976, swd: 3.3902, ept: 303.6314
      Epoch 11 composite train-obj: 0.913138
            No improvement (0.7158), counter 1/5
    Epoch [12/50], Train Losses: mse: 6.9494, mae: 1.1931, huber: 0.9050, swd: 2.9290, ept: 126.4989
    Epoch [12/50], Val Losses: mse: 4.5274, mae: 0.8680, huber: 0.6051, swd: 3.3118, ept: 305.5121
    Epoch [12/50], Test Losses: mse: 3.9103, mae: 0.8402, huber: 0.5705, swd: 2.8823, ept: 305.1590
      Epoch 12 composite train-obj: 0.904987
            Val objective improved 0.6242 → 0.6051, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 6.8949, mae: 1.1836, huber: 0.8974, swd: 2.8894, ept: 126.4583
    Epoch [13/50], Val Losses: mse: 4.5085, mae: 1.0009, huber: 0.6879, swd: 3.1906, ept: 305.2826
    Epoch [13/50], Test Losses: mse: 3.9234, mae: 0.9650, huber: 0.6498, swd: 2.7991, ept: 307.2618
      Epoch 13 composite train-obj: 0.897425
            No improvement (0.6879), counter 1/5
    Epoch [14/50], Train Losses: mse: 6.8608, mae: 1.1735, huber: 0.8903, swd: 2.8585, ept: 126.6173
    Epoch [14/50], Val Losses: mse: 4.7055, mae: 0.8706, huber: 0.6064, swd: 3.1426, ept: 305.0407
    Epoch [14/50], Test Losses: mse: 4.0934, mae: 0.8512, huber: 0.5806, swd: 2.7411, ept: 305.2648
      Epoch 14 composite train-obj: 0.890330
            No improvement (0.6064), counter 2/5
    Epoch [15/50], Train Losses: mse: 6.8430, mae: 1.1781, huber: 0.8921, swd: 2.8446, ept: 126.6554
    Epoch [15/50], Val Losses: mse: 4.6576, mae: 0.9163, huber: 0.6273, swd: 3.2156, ept: 304.3887
    Epoch [15/50], Test Losses: mse: 4.0229, mae: 0.8900, huber: 0.5965, swd: 2.8070, ept: 305.4807
      Epoch 15 composite train-obj: 0.892140
            No improvement (0.6273), counter 3/5
    Epoch [16/50], Train Losses: mse: 6.7701, mae: 1.1636, huber: 0.8812, swd: 2.7937, ept: 126.6982
    Epoch [16/50], Val Losses: mse: 4.3741, mae: 0.8905, huber: 0.6080, swd: 3.1149, ept: 305.1714
    Epoch [16/50], Test Losses: mse: 3.7463, mae: 0.8539, huber: 0.5672, swd: 2.6892, ept: 306.3679
      Epoch 16 composite train-obj: 0.881185
            No improvement (0.6080), counter 4/5
    Epoch [17/50], Train Losses: mse: 6.7439, mae: 1.1546, huber: 0.8752, swd: 2.7682, ept: 126.6900
    Epoch [17/50], Val Losses: mse: 4.5325, mae: 0.8380, huber: 0.5839, swd: 3.1651, ept: 302.9476
    Epoch [17/50], Test Losses: mse: 3.9236, mae: 0.8158, huber: 0.5564, swd: 2.7644, ept: 303.8770
      Epoch 17 composite train-obj: 0.875197
            Val objective improved 0.6051 → 0.5839, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 6.6944, mae: 1.1452, huber: 0.8683, swd: 2.7221, ept: 126.6871
    Epoch [18/50], Val Losses: mse: 4.2226, mae: 0.8672, huber: 0.5946, swd: 2.9918, ept: 305.5786
    Epoch [18/50], Test Losses: mse: 3.6282, mae: 0.8306, huber: 0.5521, swd: 2.5701, ept: 307.5497
      Epoch 18 composite train-obj: 0.868259
            No improvement (0.5946), counter 1/5
    Epoch [19/50], Train Losses: mse: 6.6209, mae: 1.1405, huber: 0.8638, swd: 2.6547, ept: 126.5371
    Epoch [19/50], Val Losses: mse: 4.3291, mae: 0.8678, huber: 0.6025, swd: 2.9659, ept: 304.6369
    Epoch [19/50], Test Losses: mse: 3.8573, mae: 0.8570, huber: 0.5834, swd: 2.7010, ept: 305.5401
      Epoch 19 composite train-obj: 0.863783
            No improvement (0.6025), counter 2/5
    Epoch [20/50], Train Losses: mse: 6.5491, mae: 1.1359, huber: 0.8601, swd: 2.5638, ept: 126.6694
    Epoch [20/50], Val Losses: mse: 3.6538, mae: 0.8133, huber: 0.5495, swd: 2.5256, ept: 308.5568
    Epoch [20/50], Test Losses: mse: 3.2018, mae: 0.7861, huber: 0.5143, swd: 2.2548, ept: 309.5346
      Epoch 20 composite train-obj: 0.860112
            Val objective improved 0.5839 → 0.5495, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 6.4001, mae: 1.1194, huber: 0.8464, swd: 2.4233, ept: 126.6159
    Epoch [21/50], Val Losses: mse: 3.9162, mae: 0.8412, huber: 0.5677, swd: 2.7077, ept: 307.2773
    Epoch [21/50], Test Losses: mse: 3.4335, mae: 0.8187, huber: 0.5390, swd: 2.3984, ept: 308.3636
      Epoch 21 composite train-obj: 0.846449
            No improvement (0.5677), counter 1/5
    Epoch [22/50], Train Losses: mse: 6.4201, mae: 1.1219, huber: 0.8476, swd: 2.4491, ept: 126.7730
    Epoch [22/50], Val Losses: mse: 3.9486, mae: 0.8471, huber: 0.5649, swd: 2.5975, ept: 307.7350
    Epoch [22/50], Test Losses: mse: 3.4217, mae: 0.8183, huber: 0.5327, swd: 2.2767, ept: 308.3216
      Epoch 22 composite train-obj: 0.847617
            No improvement (0.5649), counter 2/5
    Epoch [23/50], Train Losses: mse: 6.3087, mae: 1.1080, huber: 0.8373, swd: 2.3312, ept: 126.5738
    Epoch [23/50], Val Losses: mse: 3.7641, mae: 0.9146, huber: 0.6156, swd: 2.5602, ept: 306.4404
    Epoch [23/50], Test Losses: mse: 3.2749, mae: 0.8733, huber: 0.5711, swd: 2.2437, ept: 306.0426
      Epoch 23 composite train-obj: 0.837324
            No improvement (0.6156), counter 3/5
    Epoch [24/50], Train Losses: mse: 6.2904, mae: 1.1065, huber: 0.8351, swd: 2.3372, ept: 126.6332
    Epoch [24/50], Val Losses: mse: 3.6813, mae: 0.8475, huber: 0.5648, swd: 2.3311, ept: 307.5516
    Epoch [24/50], Test Losses: mse: 3.2105, mae: 0.8176, huber: 0.5307, swd: 2.0342, ept: 308.5844
      Epoch 24 composite train-obj: 0.835062
            No improvement (0.5648), counter 4/5
    Epoch [25/50], Train Losses: mse: 6.2303, mae: 1.0995, huber: 0.8292, swd: 2.2667, ept: 126.6735
    Epoch [25/50], Val Losses: mse: 3.7727, mae: 0.8294, huber: 0.5542, swd: 2.4921, ept: 309.5211
    Epoch [25/50], Test Losses: mse: 3.2099, mae: 0.7943, huber: 0.5177, swd: 2.1261, ept: 309.6673
      Epoch 25 composite train-obj: 0.829244
    Epoch [25/50], Test Losses: mse: 3.2018, mae: 0.7861, huber: 0.5143, swd: 2.2548, ept: 309.5346
    Best round's Test MSE: 3.2018, MAE: 0.7861, SWD: 2.2548
    Best round's Validation MSE: 3.6538, MAE: 0.8133, SWD: 2.5256
    Best round's Test verification MSE : 3.2018, MAE: 0.7861, SWD: 2.2548
    Time taken: 140.51 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.2835, mae: 1.8723, huber: 1.5049, swd: 4.8566, ept: 115.7463
    Epoch [1/50], Val Losses: mse: 5.9902, mae: 1.3207, huber: 0.9794, swd: 4.1897, ept: 286.5048
    Epoch [1/50], Test Losses: mse: 5.2494, mae: 1.2635, huber: 0.9190, swd: 3.6453, ept: 287.8961
      Epoch 1 composite train-obj: 1.504902
            Val objective improved inf → 0.9794, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.9982, mae: 1.4838, huber: 1.1422, swd: 3.6213, ept: 125.6159
    Epoch [2/50], Val Losses: mse: 5.7515, mae: 1.1731, huber: 0.8463, swd: 4.0380, ept: 295.7239
    Epoch [2/50], Test Losses: mse: 5.0664, mae: 1.1550, huber: 0.8194, swd: 3.5219, ept: 295.4675
      Epoch 2 composite train-obj: 1.142203
            Val objective improved 0.9794 → 0.8463, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.6616, mae: 1.3833, huber: 1.0565, swd: 3.4478, ept: 126.2271
    Epoch [3/50], Val Losses: mse: 5.4789, mae: 1.1724, huber: 0.8406, swd: 3.8361, ept: 305.4057
    Epoch [3/50], Test Losses: mse: 4.8068, mae: 1.1428, huber: 0.8041, swd: 3.3826, ept: 304.6262
      Epoch 3 composite train-obj: 1.056458
            Val objective improved 0.8463 → 0.8406, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.5061, mae: 1.3414, huber: 1.0215, swd: 3.3557, ept: 126.2939
    Epoch [4/50], Val Losses: mse: 5.8328, mae: 1.4305, huber: 1.0640, swd: 4.2921, ept: 305.1591
    Epoch [4/50], Test Losses: mse: 5.4073, mae: 1.4296, huber: 1.0592, swd: 3.9754, ept: 305.2142
      Epoch 4 composite train-obj: 1.021454
            No improvement (1.0640), counter 1/5
    Epoch [5/50], Train Losses: mse: 7.4000, mae: 1.3114, huber: 0.9973, swd: 3.2948, ept: 126.3986
    Epoch [5/50], Val Losses: mse: 4.8609, mae: 0.9905, huber: 0.6870, swd: 3.5095, ept: 301.9398
    Epoch [5/50], Test Losses: mse: 4.2460, mae: 0.9612, huber: 0.6546, swd: 3.0644, ept: 302.5170
      Epoch 5 composite train-obj: 0.997349
            Val objective improved 0.8406 → 0.6870, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.2845, mae: 1.2869, huber: 0.9767, swd: 3.2287, ept: 126.4617
    Epoch [6/50], Val Losses: mse: 4.9204, mae: 1.0370, huber: 0.7212, swd: 3.5910, ept: 302.7882
    Epoch [6/50], Test Losses: mse: 4.2794, mae: 1.0028, huber: 0.6828, swd: 3.1523, ept: 303.6013
      Epoch 6 composite train-obj: 0.976651
            No improvement (0.7212), counter 1/5
    Epoch [7/50], Train Losses: mse: 7.1699, mae: 1.2515, huber: 0.9494, swd: 3.1512, ept: 126.3111
    Epoch [7/50], Val Losses: mse: 4.9693, mae: 1.0123, huber: 0.7027, swd: 3.5430, ept: 304.9148
    Epoch [7/50], Test Losses: mse: 4.3205, mae: 0.9888, huber: 0.6728, swd: 3.1141, ept: 305.4808
      Epoch 7 composite train-obj: 0.949449
            No improvement (0.7027), counter 2/5
    Epoch [8/50], Train Losses: mse: 7.1288, mae: 1.2431, huber: 0.9429, swd: 3.1261, ept: 126.4930
    Epoch [8/50], Val Losses: mse: 4.7124, mae: 0.9435, huber: 0.6609, swd: 3.5322, ept: 304.3177
    Epoch [8/50], Test Losses: mse: 4.0636, mae: 0.9066, huber: 0.6162, swd: 3.0689, ept: 305.3732
      Epoch 8 composite train-obj: 0.942939
            Val objective improved 0.6870 → 0.6609, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.0522, mae: 1.2168, huber: 0.9232, swd: 3.0754, ept: 126.6141
    Epoch [9/50], Val Losses: mse: 4.8767, mae: 0.9218, huber: 0.6408, swd: 3.5821, ept: 302.8684
    Epoch [9/50], Test Losses: mse: 4.2102, mae: 0.8968, huber: 0.6074, swd: 3.1065, ept: 302.0914
      Epoch 9 composite train-obj: 0.923190
            Val objective improved 0.6609 → 0.6408, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.0063, mae: 1.2070, huber: 0.9153, swd: 3.0487, ept: 126.5501
    Epoch [10/50], Val Losses: mse: 4.4614, mae: 0.9730, huber: 0.6689, swd: 3.3702, ept: 307.3700
    Epoch [10/50], Test Losses: mse: 3.8608, mae: 0.9335, huber: 0.6271, swd: 2.9292, ept: 307.4362
      Epoch 10 composite train-obj: 0.915277
            No improvement (0.6689), counter 1/5
    Epoch [11/50], Train Losses: mse: 6.9668, mae: 1.1985, huber: 0.9085, swd: 3.0201, ept: 126.7314
    Epoch [11/50], Val Losses: mse: 5.0324, mae: 1.0118, huber: 0.7058, swd: 3.6037, ept: 304.3107
    Epoch [11/50], Test Losses: mse: 4.3639, mae: 0.9777, huber: 0.6673, swd: 3.1597, ept: 303.8229
      Epoch 11 composite train-obj: 0.908537
            No improvement (0.7058), counter 2/5
    Epoch [12/50], Train Losses: mse: 6.9251, mae: 1.1865, huber: 0.9004, swd: 2.9860, ept: 126.7285
    Epoch [12/50], Val Losses: mse: 4.6854, mae: 0.9615, huber: 0.6636, swd: 3.3779, ept: 304.0983
    Epoch [12/50], Test Losses: mse: 4.0451, mae: 0.9285, huber: 0.6263, swd: 2.9299, ept: 305.4287
      Epoch 12 composite train-obj: 0.900356
            No improvement (0.6636), counter 3/5
    Epoch [13/50], Train Losses: mse: 6.8794, mae: 1.1853, huber: 0.8975, swd: 2.9432, ept: 126.7056
    Epoch [13/50], Val Losses: mse: 4.5400, mae: 0.9254, huber: 0.6460, swd: 3.3000, ept: 306.0271
    Epoch [13/50], Test Losses: mse: 3.9207, mae: 0.8896, huber: 0.6039, swd: 2.8658, ept: 307.7530
      Epoch 13 composite train-obj: 0.897477
            No improvement (0.6460), counter 4/5
    Epoch [14/50], Train Losses: mse: 6.8301, mae: 1.1751, huber: 0.8903, swd: 2.9010, ept: 126.6513
    Epoch [14/50], Val Losses: mse: 4.4021, mae: 0.9062, huber: 0.6303, swd: 3.1922, ept: 304.9367
    Epoch [14/50], Test Losses: mse: 3.8235, mae: 0.8729, huber: 0.5933, swd: 2.7677, ept: 306.3475
      Epoch 14 composite train-obj: 0.890333
            Val objective improved 0.6408 → 0.6303, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 6.7600, mae: 1.1636, huber: 0.8812, swd: 2.8278, ept: 126.6608
    Epoch [15/50], Val Losses: mse: 4.5269, mae: 0.9974, huber: 0.6871, swd: 3.1980, ept: 304.9149
    Epoch [15/50], Test Losses: mse: 3.9748, mae: 0.9659, huber: 0.6533, swd: 2.8566, ept: 305.6083
      Epoch 15 composite train-obj: 0.881207
            No improvement (0.6871), counter 1/5
    Epoch [16/50], Train Losses: mse: 6.7038, mae: 1.1618, huber: 0.8790, swd: 2.7635, ept: 126.5953
    Epoch [16/50], Val Losses: mse: 4.2173, mae: 0.9298, huber: 0.6335, swd: 2.9717, ept: 306.5387
    Epoch [16/50], Test Losses: mse: 3.8933, mae: 0.9097, huber: 0.6090, swd: 2.8529, ept: 307.4431
      Epoch 16 composite train-obj: 0.879011
            No improvement (0.6335), counter 2/5
    Epoch [17/50], Train Losses: mse: 6.5513, mae: 1.1360, huber: 0.8593, swd: 2.6226, ept: 126.8118
    Epoch [17/50], Val Losses: mse: 4.4761, mae: 1.0340, huber: 0.7104, swd: 3.0634, ept: 304.2866
    Epoch [17/50], Test Losses: mse: 3.8767, mae: 0.9931, huber: 0.6675, swd: 2.6726, ept: 304.6729
      Epoch 17 composite train-obj: 0.859304
            No improvement (0.7104), counter 3/5
    Epoch [18/50], Train Losses: mse: 6.4970, mae: 1.1399, huber: 0.8600, swd: 2.5550, ept: 126.7169
    Epoch [18/50], Val Losses: mse: 4.3410, mae: 0.9695, huber: 0.6623, swd: 3.0829, ept: 305.6095
    Epoch [18/50], Test Losses: mse: 3.8502, mae: 0.9397, huber: 0.6290, swd: 2.7790, ept: 304.6842
      Epoch 18 composite train-obj: 0.860016
            No improvement (0.6623), counter 4/5
    Epoch [19/50], Train Losses: mse: 6.4110, mae: 1.1274, huber: 0.8503, swd: 2.4676, ept: 126.9698
    Epoch [19/50], Val Losses: mse: 3.8214, mae: 0.8587, huber: 0.5729, swd: 2.5134, ept: 305.8899
    Epoch [19/50], Test Losses: mse: 3.3616, mae: 0.8298, huber: 0.5388, swd: 2.2609, ept: 306.9526
      Epoch 19 composite train-obj: 0.850287
            Val objective improved 0.6303 → 0.5729, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 6.3471, mae: 1.1246, huber: 0.8475, swd: 2.3902, ept: 126.8124
    Epoch [20/50], Val Losses: mse: 3.7957, mae: 0.8983, huber: 0.6042, swd: 2.4822, ept: 307.1548
    Epoch [20/50], Test Losses: mse: 3.3166, mae: 0.8657, huber: 0.5685, swd: 2.2012, ept: 306.2084
      Epoch 20 composite train-obj: 0.847532
            No improvement (0.6042), counter 1/5
    Epoch [21/50], Train Losses: mse: 6.2653, mae: 1.1172, huber: 0.8409, swd: 2.3017, ept: 126.7235
    Epoch [21/50], Val Losses: mse: 3.8516, mae: 0.9012, huber: 0.6072, swd: 2.4839, ept: 308.0263
    Epoch [21/50], Test Losses: mse: 3.3326, mae: 0.8680, huber: 0.5685, swd: 2.1552, ept: 307.7220
      Epoch 21 composite train-obj: 0.840930
            No improvement (0.6072), counter 2/5
    Epoch [22/50], Train Losses: mse: 6.1945, mae: 1.1044, huber: 0.8315, swd: 2.2210, ept: 126.8284
    Epoch [22/50], Val Losses: mse: 3.6613, mae: 0.9190, huber: 0.6185, swd: 2.4213, ept: 304.6676
    Epoch [22/50], Test Losses: mse: 3.2099, mae: 0.8860, huber: 0.5806, swd: 2.1487, ept: 306.2609
      Epoch 22 composite train-obj: 0.831498
            No improvement (0.6185), counter 3/5
    Epoch [23/50], Train Losses: mse: 6.0995, mae: 1.0917, huber: 0.8213, swd: 2.1289, ept: 126.7547
    Epoch [23/50], Val Losses: mse: 3.4451, mae: 0.9316, huber: 0.6164, swd: 2.2920, ept: 306.0054
    Epoch [23/50], Test Losses: mse: 2.9812, mae: 0.8897, huber: 0.5708, swd: 2.0174, ept: 308.5505
      Epoch 23 composite train-obj: 0.821289
            No improvement (0.6164), counter 4/5
    Epoch [24/50], Train Losses: mse: 6.1648, mae: 1.1068, huber: 0.8315, swd: 2.1922, ept: 126.6654
    Epoch [24/50], Val Losses: mse: 4.0387, mae: 0.9959, huber: 0.6929, swd: 2.5058, ept: 304.0010
    Epoch [24/50], Test Losses: mse: 3.6035, mae: 0.9713, huber: 0.6639, swd: 2.2236, ept: 304.8406
      Epoch 24 composite train-obj: 0.831478
    Epoch [24/50], Test Losses: mse: 3.3616, mae: 0.8298, huber: 0.5388, swd: 2.2609, ept: 306.9526
    Best round's Test MSE: 3.3616, MAE: 0.8298, SWD: 2.2609
    Best round's Validation MSE: 3.8214, MAE: 0.8587, SWD: 2.5134
    Best round's Test verification MSE : 3.3616, MAE: 0.8298, SWD: 2.2609
    Time taken: 135.59 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 10.3755, mae: 1.8925, huber: 1.5241, swd: 4.8712, ept: 115.1712
    Epoch [1/50], Val Losses: mse: 5.9426, mae: 1.2830, huber: 0.9370, swd: 4.2324, ept: 291.9143
    Epoch [1/50], Test Losses: mse: 5.2821, mae: 1.2631, huber: 0.9082, swd: 3.7245, ept: 294.8518
      Epoch 1 composite train-obj: 1.524146
            Val objective improved inf → 0.9370, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 8.0307, mae: 1.4832, huber: 1.1421, swd: 3.6522, ept: 125.5453
    Epoch [2/50], Val Losses: mse: 5.8100, mae: 1.3377, huber: 0.9816, swd: 4.1058, ept: 298.6203
    Epoch [2/50], Test Losses: mse: 5.1599, mae: 1.3051, huber: 0.9429, swd: 3.6440, ept: 300.2056
      Epoch 2 composite train-obj: 1.142132
            No improvement (0.9816), counter 1/5
    Epoch [3/50], Train Losses: mse: 7.6989, mae: 1.3944, huber: 1.0655, swd: 3.4707, ept: 126.2407
    Epoch [3/50], Val Losses: mse: 5.6627, mae: 1.0870, huber: 0.7854, swd: 4.0739, ept: 298.8749
    Epoch [3/50], Test Losses: mse: 5.0532, mae: 1.0835, huber: 0.7734, swd: 3.6563, ept: 296.8129
      Epoch 3 composite train-obj: 1.065474
            Val objective improved 0.9370 → 0.7854, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.5462, mae: 1.3527, huber: 1.0308, swd: 3.3942, ept: 126.2856
    Epoch [4/50], Val Losses: mse: 5.0585, mae: 1.1118, huber: 0.7844, swd: 3.7484, ept: 305.2890
    Epoch [4/50], Test Losses: mse: 4.4126, mae: 1.0694, huber: 0.7376, swd: 3.2941, ept: 306.0601
      Epoch 4 composite train-obj: 1.030820
            Val objective improved 0.7854 → 0.7844, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.4150, mae: 1.3166, huber: 1.0016, swd: 3.3164, ept: 126.4841
    Epoch [5/50], Val Losses: mse: 5.4586, mae: 1.1324, huber: 0.8060, swd: 3.8455, ept: 303.9863
    Epoch [5/50], Test Losses: mse: 4.8551, mae: 1.1230, huber: 0.7913, swd: 3.4624, ept: 304.2760
      Epoch 5 composite train-obj: 1.001556
            No improvement (0.8060), counter 1/5
    Epoch [6/50], Train Losses: mse: 7.3255, mae: 1.2926, huber: 0.9821, swd: 3.2616, ept: 126.5525
    Epoch [6/50], Val Losses: mse: 5.0421, mae: 1.0328, huber: 0.7283, swd: 3.7013, ept: 302.3150
    Epoch [6/50], Test Losses: mse: 4.4462, mae: 1.0162, huber: 0.7064, swd: 3.2867, ept: 303.5232
      Epoch 6 composite train-obj: 0.982079
            Val objective improved 0.7844 → 0.7283, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.2421, mae: 1.2673, huber: 0.9627, swd: 3.2075, ept: 126.5436
    Epoch [7/50], Val Losses: mse: 4.7261, mae: 0.9726, huber: 0.6684, swd: 3.5026, ept: 305.3890
    Epoch [7/50], Test Losses: mse: 4.0853, mae: 0.9437, huber: 0.6313, swd: 3.0446, ept: 306.4259
      Epoch 7 composite train-obj: 0.962747
            Val objective improved 0.7283 → 0.6684, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 7.1671, mae: 1.2525, huber: 0.9506, swd: 3.1663, ept: 126.4732
    Epoch [8/50], Val Losses: mse: 4.6959, mae: 1.0103, huber: 0.6981, swd: 3.4964, ept: 307.1398
    Epoch [8/50], Test Losses: mse: 4.0391, mae: 0.9589, huber: 0.6444, swd: 3.0384, ept: 308.8217
      Epoch 8 composite train-obj: 0.950625
            No improvement (0.6981), counter 1/5
    Epoch [9/50], Train Losses: mse: 7.1111, mae: 1.2349, huber: 0.9375, swd: 3.1247, ept: 126.5740
    Epoch [9/50], Val Losses: mse: 4.8296, mae: 0.9309, huber: 0.6526, swd: 3.5430, ept: 305.2374
    Epoch [9/50], Test Losses: mse: 4.1734, mae: 0.9029, huber: 0.6153, swd: 3.0935, ept: 305.7816
      Epoch 9 composite train-obj: 0.937472
            Val objective improved 0.6684 → 0.6526, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.0612, mae: 1.2288, huber: 0.9314, swd: 3.0923, ept: 126.5472
    Epoch [10/50], Val Losses: mse: 4.6864, mae: 1.0033, huber: 0.6884, swd: 3.4229, ept: 304.2489
    Epoch [10/50], Test Losses: mse: 4.0664, mae: 0.9740, huber: 0.6537, swd: 2.9803, ept: 305.9818
      Epoch 10 composite train-obj: 0.931412
            No improvement (0.6884), counter 1/5
    Epoch [11/50], Train Losses: mse: 6.9868, mae: 1.2143, huber: 0.9207, swd: 3.0313, ept: 126.5941
    Epoch [11/50], Val Losses: mse: 4.5209, mae: 0.9394, huber: 0.6566, swd: 3.3508, ept: 306.7497
    Epoch [11/50], Test Losses: mse: 3.8983, mae: 0.8967, huber: 0.6096, swd: 2.9234, ept: 307.1906
      Epoch 11 composite train-obj: 0.920728
            No improvement (0.6566), counter 2/5
    Epoch [12/50], Train Losses: mse: 6.8995, mae: 1.2043, huber: 0.9122, swd: 2.9460, ept: 126.6282
    Epoch [12/50], Val Losses: mse: 4.7944, mae: 0.9526, huber: 0.6552, swd: 3.3453, ept: 303.4385
    Epoch [12/50], Test Losses: mse: 4.2425, mae: 0.9510, huber: 0.6385, swd: 2.9856, ept: 303.1717
      Epoch 12 composite train-obj: 0.912202
            No improvement (0.6552), counter 3/5
    Epoch [13/50], Train Losses: mse: 6.7867, mae: 1.1904, huber: 0.9009, swd: 2.8314, ept: 126.5334
    Epoch [13/50], Val Losses: mse: 4.6372, mae: 1.1998, huber: 0.8439, swd: 3.3941, ept: 310.1969
    Epoch [13/50], Test Losses: mse: 4.1940, mae: 1.1844, huber: 0.8228, swd: 3.1207, ept: 309.7202
      Epoch 13 composite train-obj: 0.900912
            No improvement (0.8439), counter 4/5
    Epoch [14/50], Train Losses: mse: 6.6843, mae: 1.1785, huber: 0.8910, swd: 2.7248, ept: 126.7494
    Epoch [14/50], Val Losses: mse: 4.1625, mae: 0.8733, huber: 0.5904, swd: 2.8935, ept: 306.3686
    Epoch [14/50], Test Losses: mse: 3.6115, mae: 0.8468, huber: 0.5565, swd: 2.5299, ept: 307.8535
      Epoch 14 composite train-obj: 0.891044
            Val objective improved 0.6526 → 0.5904, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 6.5887, mae: 1.1710, huber: 0.8834, swd: 2.6140, ept: 126.6849
    Epoch [15/50], Val Losses: mse: 4.0424, mae: 0.9708, huber: 0.6643, swd: 2.8363, ept: 308.2258
    Epoch [15/50], Test Losses: mse: 3.5631, mae: 0.9391, huber: 0.6296, swd: 2.5558, ept: 309.8674
      Epoch 15 composite train-obj: 0.883402
            No improvement (0.6643), counter 1/5
    Epoch [16/50], Train Losses: mse: 6.4832, mae: 1.1562, huber: 0.8717, swd: 2.5058, ept: 126.7532
    Epoch [16/50], Val Losses: mse: 4.0242, mae: 1.0836, huber: 0.7481, swd: 2.8365, ept: 308.7635
    Epoch [16/50], Test Losses: mse: 3.5569, mae: 1.0441, huber: 0.7044, swd: 2.5437, ept: 310.0104
      Epoch 16 composite train-obj: 0.871693
            No improvement (0.7481), counter 2/5
    Epoch [17/50], Train Losses: mse: 6.4288, mae: 1.1505, huber: 0.8668, swd: 2.4509, ept: 126.7306
    Epoch [17/50], Val Losses: mse: 3.7281, mae: 0.9379, huber: 0.6347, swd: 2.4951, ept: 310.7697
    Epoch [17/50], Test Losses: mse: 3.2892, mae: 0.9125, huber: 0.6025, swd: 2.2476, ept: 310.1435
      Epoch 17 composite train-obj: 0.866776
            No improvement (0.6347), counter 3/5
    Epoch [18/50], Train Losses: mse: 6.3578, mae: 1.1415, huber: 0.8597, swd: 2.3887, ept: 126.6169
    Epoch [18/50], Val Losses: mse: 3.7518, mae: 0.9975, huber: 0.6827, swd: 2.6009, ept: 307.3659
    Epoch [18/50], Test Losses: mse: 3.4086, mae: 0.9685, huber: 0.6499, swd: 2.3810, ept: 307.5752
      Epoch 18 composite train-obj: 0.859700
            No improvement (0.6827), counter 4/5
    Epoch [19/50], Train Losses: mse: 6.3618, mae: 1.1358, huber: 0.8563, swd: 2.3862, ept: 126.6307
    Epoch [19/50], Val Losses: mse: 3.7832, mae: 1.0449, huber: 0.7184, swd: 2.5299, ept: 311.8485
    Epoch [19/50], Test Losses: mse: 3.4090, mae: 1.0268, huber: 0.6944, swd: 2.3172, ept: 312.6234
      Epoch 19 composite train-obj: 0.856276
    Epoch [19/50], Test Losses: mse: 3.6115, mae: 0.8468, huber: 0.5565, swd: 2.5299, ept: 307.8535
    Best round's Test MSE: 3.6115, MAE: 0.8468, SWD: 2.5299
    Best round's Validation MSE: 4.1625, MAE: 0.8733, SWD: 2.8935
    Best round's Test verification MSE : 3.6115, MAE: 0.8468, SWD: 2.5299
    Time taken: 108.52 seconds
    
    ==================================================
    Experiment Summary (PatchTST_rossler_seq336_pred336_20250511_0347)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 3.3916 ± 0.1686
      mae: 0.8209 ± 0.0255
      huber: 0.5365 ± 0.0173
      swd: 2.3486 ± 0.1283
      ept: 308.1135 ± 1.0700
      count: 36.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 3.8792 ± 0.2116
      mae: 0.8485 ± 0.0256
      huber: 0.5709 ± 0.0167
      swd: 2.6442 ± 0.1764
      ept: 306.9384 ± 1.1609
      count: 36.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 384.71 seconds
    
    Experiment complete: PatchTST_rossler_seq336_pred336_20250511_0347
    Model: PatchTST
    Dataset: rossler
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

#### pred=720


```python
utils.reload_modules([utils])
cfg = train_config.FlatPatchTSTConfig(
    seq_len=336,
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 277
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 11.5939, mae: 2.1447, huber: 1.7663, swd: 5.0265, ept: 168.6983
    Epoch [1/50], Val Losses: mse: 9.7739, mae: 1.7753, huber: 1.4179, swd: 5.3363, ept: 504.7681
    Epoch [1/50], Test Losses: mse: 8.0916, mae: 1.6417, huber: 1.2789, swd: 4.3859, ept: 499.7793
      Epoch 1 composite train-obj: 1.766336
            Val objective improved inf → 1.4179, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.5322, mae: 1.8085, huber: 1.4490, swd: 4.2223, ept: 185.2775
    Epoch [2/50], Val Losses: mse: 9.1723, mae: 1.6741, huber: 1.3250, swd: 5.1904, ept: 537.0382
    Epoch [2/50], Test Losses: mse: 7.9048, mae: 1.5834, huber: 1.2278, swd: 4.3411, ept: 534.6950
      Epoch 2 composite train-obj: 1.448964
            Val objective improved 1.4179 → 1.3250, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 9.1557, mae: 1.7218, huber: 1.3718, swd: 4.0958, ept: 187.1506
    Epoch [3/50], Val Losses: mse: 10.0408, mae: 1.6251, huber: 1.2960, swd: 5.6922, ept: 546.7694
    Epoch [3/50], Test Losses: mse: 8.4184, mae: 1.5354, huber: 1.1953, swd: 4.8342, ept: 536.9090
      Epoch 3 composite train-obj: 1.371816
            Val objective improved 1.3250 → 1.2960, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 9.0196, mae: 1.6866, huber: 1.3412, swd: 4.0317, ept: 187.4109
    Epoch [4/50], Val Losses: mse: 7.8841, mae: 1.4949, huber: 1.1686, swd: 4.7857, ept: 537.3062
    Epoch [4/50], Test Losses: mse: 6.5915, mae: 1.3919, huber: 1.0603, swd: 3.8886, ept: 547.5672
      Epoch 4 composite train-obj: 1.341161
            Val objective improved 1.2960 → 1.1686, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 8.8279, mae: 1.6413, huber: 1.3022, swd: 3.9328, ept: 187.8722
    Epoch [5/50], Val Losses: mse: 7.5711, mae: 1.5582, huber: 1.2115, swd: 4.6115, ept: 555.7786
    Epoch [5/50], Test Losses: mse: 6.0813, mae: 1.3939, huber: 1.0455, swd: 3.7027, ept: 570.8715
      Epoch 5 composite train-obj: 1.302228
            No improvement (1.2115), counter 1/5
    Epoch [6/50], Train Losses: mse: 8.6926, mae: 1.6197, huber: 1.2826, swd: 3.8652, ept: 188.2460
    Epoch [6/50], Val Losses: mse: 7.3489, mae: 1.4803, huber: 1.1447, swd: 4.5536, ept: 551.5807
    Epoch [6/50], Test Losses: mse: 6.0212, mae: 1.3560, huber: 1.0171, swd: 3.6957, ept: 571.0330
      Epoch 6 composite train-obj: 1.282605
            Val objective improved 1.1686 → 1.1447, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 8.5173, mae: 1.5872, huber: 1.2540, swd: 3.7570, ept: 188.7843
    Epoch [7/50], Val Losses: mse: 7.3956, mae: 1.3890, huber: 1.0746, swd: 4.3908, ept: 556.9214
    Epoch [7/50], Test Losses: mse: 5.9386, mae: 1.2523, huber: 0.9358, swd: 3.5177, ept: 564.2637
      Epoch 7 composite train-obj: 1.254004
            Val objective improved 1.1447 → 1.0746, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 8.3397, mae: 1.5615, huber: 1.2313, swd: 3.6382, ept: 189.0813
    Epoch [8/50], Val Losses: mse: 6.8121, mae: 1.3581, huber: 1.0434, swd: 4.0901, ept: 557.0838
    Epoch [8/50], Test Losses: mse: 5.3821, mae: 1.1925, huber: 0.8789, swd: 3.2485, ept: 573.6224
      Epoch 8 composite train-obj: 1.231348
            Val objective improved 1.0746 → 1.0434, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 8.0297, mae: 1.5281, huber: 1.2002, swd: 3.3697, ept: 189.2117
    Epoch [9/50], Val Losses: mse: 6.4334, mae: 1.3295, huber: 1.0131, swd: 3.6973, ept: 561.8142
    Epoch [9/50], Test Losses: mse: 5.3020, mae: 1.2111, huber: 0.8921, swd: 2.9919, ept: 575.5596
      Epoch 9 composite train-obj: 1.200237
            Val objective improved 1.0434 → 1.0131, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.7073, mae: 1.4970, huber: 1.1723, swd: 3.0849, ept: 189.2131
    Epoch [10/50], Val Losses: mse: 5.9295, mae: 1.3417, huber: 1.0068, swd: 3.3387, ept: 569.0550
    Epoch [10/50], Test Losses: mse: 4.6904, mae: 1.1950, huber: 0.8602, swd: 2.6146, ept: 591.2904
      Epoch 10 composite train-obj: 1.172268
            Val objective improved 1.0131 → 1.0068, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 7.4891, mae: 1.4701, huber: 1.1481, swd: 2.9181, ept: 189.3790
    Epoch [11/50], Val Losses: mse: 6.4472, mae: 1.3232, huber: 1.0020, swd: 3.4630, ept: 564.4366
    Epoch [11/50], Test Losses: mse: 5.2686, mae: 1.2065, huber: 0.8835, swd: 2.7307, ept: 581.5497
      Epoch 11 composite train-obj: 1.148098
            Val objective improved 1.0068 → 1.0020, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 7.3748, mae: 1.4581, huber: 1.1368, swd: 2.8050, ept: 189.2682
    Epoch [12/50], Val Losses: mse: 6.0874, mae: 1.3593, huber: 1.0312, swd: 3.2197, ept: 557.6291
    Epoch [12/50], Test Losses: mse: 4.9181, mae: 1.2346, huber: 0.9018, swd: 2.5742, ept: 582.5049
      Epoch 12 composite train-obj: 1.136773
            No improvement (1.0312), counter 1/5
    Epoch [13/50], Train Losses: mse: 7.2186, mae: 1.4335, huber: 1.1155, swd: 2.7062, ept: 189.3407
    Epoch [13/50], Val Losses: mse: 5.6864, mae: 1.2435, huber: 0.9284, swd: 2.9852, ept: 564.8193
    Epoch [13/50], Test Losses: mse: 4.7500, mae: 1.1457, huber: 0.8283, swd: 2.4372, ept: 584.3468
      Epoch 13 composite train-obj: 1.115500
            Val objective improved 1.0020 → 0.9284, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 7.1253, mae: 1.4163, huber: 1.0998, swd: 2.6465, ept: 189.2783
    Epoch [14/50], Val Losses: mse: 5.2469, mae: 1.1832, huber: 0.8775, swd: 2.7377, ept: 578.2592
    Epoch [14/50], Test Losses: mse: 4.2016, mae: 1.0710, huber: 0.7636, swd: 2.1563, ept: 599.9959
      Epoch 14 composite train-obj: 1.099788
            Val objective improved 0.9284 → 0.8775, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 7.0028, mae: 1.3983, huber: 1.0839, swd: 2.5521, ept: 189.1248
    Epoch [15/50], Val Losses: mse: 4.9860, mae: 1.2054, huber: 0.8882, swd: 2.6518, ept: 572.6829
    Epoch [15/50], Test Losses: mse: 4.1182, mae: 1.1018, huber: 0.7818, swd: 2.1083, ept: 584.3213
      Epoch 15 composite train-obj: 1.083943
            No improvement (0.8882), counter 1/5
    Epoch [16/50], Train Losses: mse: 6.9507, mae: 1.3904, huber: 1.0764, swd: 2.5037, ept: 189.1831
    Epoch [16/50], Val Losses: mse: 5.0459, mae: 1.3052, huber: 0.9637, swd: 2.8365, ept: 572.3777
    Epoch [16/50], Test Losses: mse: 4.1869, mae: 1.2095, huber: 0.8656, swd: 2.2955, ept: 592.8608
      Epoch 16 composite train-obj: 1.076430
            No improvement (0.9637), counter 2/5
    Epoch [17/50], Train Losses: mse: 6.8866, mae: 1.3842, huber: 1.0705, swd: 2.4614, ept: 189.3085
    Epoch [17/50], Val Losses: mse: 5.2723, mae: 1.1777, huber: 0.8783, swd: 2.5631, ept: 582.1864
    Epoch [17/50], Test Losses: mse: 4.3584, mae: 1.0845, huber: 0.7837, swd: 2.0499, ept: 593.7588
      Epoch 17 composite train-obj: 1.070533
            No improvement (0.8783), counter 3/5
    Epoch [18/50], Train Losses: mse: 6.7985, mae: 1.3664, huber: 1.0548, swd: 2.4040, ept: 189.3185
    Epoch [18/50], Val Losses: mse: 5.0115, mae: 1.1899, huber: 0.8768, swd: 2.5662, ept: 584.4154
    Epoch [18/50], Test Losses: mse: 4.0623, mae: 1.0751, huber: 0.7602, swd: 2.0573, ept: 597.0846
      Epoch 18 composite train-obj: 1.054793
            Val objective improved 0.8775 → 0.8768, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 6.7084, mae: 1.3494, huber: 1.0403, swd: 2.3435, ept: 189.4856
    Epoch [19/50], Val Losses: mse: 5.6997, mae: 1.2783, huber: 0.9475, swd: 2.5310, ept: 584.7635
    Epoch [19/50], Test Losses: mse: 4.7198, mae: 1.1963, huber: 0.8631, swd: 1.9859, ept: 599.8621
      Epoch 19 composite train-obj: 1.040314
            No improvement (0.9475), counter 1/5
    Epoch [20/50], Train Losses: mse: 6.7115, mae: 1.3499, huber: 1.0405, swd: 2.3333, ept: 189.5216
    Epoch [20/50], Val Losses: mse: 5.1582, mae: 1.2715, huber: 0.9416, swd: 2.9585, ept: 580.2334
    Epoch [20/50], Test Losses: mse: 4.2516, mae: 1.1719, huber: 0.8405, swd: 2.3920, ept: 594.4291
      Epoch 20 composite train-obj: 1.040517
            No improvement (0.9416), counter 2/5
    Epoch [21/50], Train Losses: mse: 6.6452, mae: 1.3383, huber: 1.0305, swd: 2.2917, ept: 189.3486
    Epoch [21/50], Val Losses: mse: 5.5141, mae: 1.2258, huber: 0.9128, swd: 2.5146, ept: 575.2124
    Epoch [21/50], Test Losses: mse: 4.6545, mae: 1.1443, huber: 0.8271, swd: 2.0264, ept: 586.2942
      Epoch 21 composite train-obj: 1.030455
            No improvement (0.9128), counter 3/5
    Epoch [22/50], Train Losses: mse: 6.5997, mae: 1.3286, huber: 1.0220, swd: 2.2565, ept: 189.3594
    Epoch [22/50], Val Losses: mse: 5.6258, mae: 1.2818, huber: 0.9569, swd: 2.6755, ept: 566.9070
    Epoch [22/50], Test Losses: mse: 4.5270, mae: 1.1901, huber: 0.8586, swd: 2.0720, ept: 576.3123
      Epoch 22 composite train-obj: 1.021998
            No improvement (0.9569), counter 4/5
    Epoch [23/50], Train Losses: mse: 6.5393, mae: 1.3176, huber: 1.0122, swd: 2.2202, ept: 189.6734
    Epoch [23/50], Val Losses: mse: 5.4860, mae: 1.2804, huber: 0.9501, swd: 2.4889, ept: 562.7803
    Epoch [23/50], Test Losses: mse: 4.5610, mae: 1.1973, huber: 0.8628, swd: 1.9919, ept: 571.2688
      Epoch 23 composite train-obj: 1.012231
    Epoch [23/50], Test Losses: mse: 4.0623, mae: 1.0751, huber: 0.7602, swd: 2.0573, ept: 597.0846
    Best round's Test MSE: 4.0623, MAE: 1.0751, SWD: 2.0573
    Best round's Validation MSE: 5.0115, MAE: 1.1899, SWD: 2.5662
    Best round's Test verification MSE : 4.0623, MAE: 1.0751, SWD: 2.0573
    Time taken: 136.66 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.2831, mae: 2.0963, huber: 1.7199, swd: 4.7171, ept: 169.4947
    Epoch [1/50], Val Losses: mse: 8.8890, mae: 1.8015, huber: 1.4366, swd: 4.9281, ept: 518.6779
    Epoch [1/50], Test Losses: mse: 7.2511, mae: 1.6418, huber: 1.2743, swd: 3.9481, ept: 534.3894
      Epoch 1 composite train-obj: 1.719903
            Val objective improved inf → 1.4366, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.4020, mae: 1.7797, huber: 1.4227, swd: 4.0999, ept: 185.5062
    Epoch [2/50], Val Losses: mse: 8.0082, mae: 1.6146, huber: 1.2640, swd: 4.8318, ept: 531.6974
    Epoch [2/50], Test Losses: mse: 6.5929, mae: 1.4769, huber: 1.1220, swd: 3.8931, ept: 548.4853
      Epoch 2 composite train-obj: 1.422741
            Val objective improved 1.4366 → 1.2640, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 9.0308, mae: 1.6966, huber: 1.3493, swd: 3.9697, ept: 187.4637
    Epoch [3/50], Val Losses: mse: 7.8433, mae: 1.5389, huber: 1.2001, swd: 4.6771, ept: 542.5812
    Epoch [3/50], Test Losses: mse: 6.4263, mae: 1.4071, huber: 1.0644, swd: 3.7787, ept: 557.6647
      Epoch 3 composite train-obj: 1.349287
            Val objective improved 1.2640 → 1.2001, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.8255, mae: 1.6469, huber: 1.3063, swd: 3.8828, ept: 187.8391
    Epoch [4/50], Val Losses: mse: 7.4961, mae: 1.4650, huber: 1.1317, swd: 4.4941, ept: 555.5443
    Epoch [4/50], Test Losses: mse: 6.0643, mae: 1.3333, huber: 0.9936, swd: 3.5959, ept: 568.3219
      Epoch 4 composite train-obj: 1.306345
            Val objective improved 1.2001 → 1.1317, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 8.6679, mae: 1.6096, huber: 1.2744, swd: 3.8016, ept: 187.8952
    Epoch [5/50], Val Losses: mse: 7.4093, mae: 1.4672, huber: 1.1377, swd: 4.5240, ept: 551.2516
    Epoch [5/50], Test Losses: mse: 6.0248, mae: 1.3415, huber: 1.0072, swd: 3.6737, ept: 571.6201
      Epoch 5 composite train-obj: 1.274440
            No improvement (1.1377), counter 1/5
    Epoch [6/50], Train Losses: mse: 8.5740, mae: 1.5884, huber: 1.2561, swd: 3.7498, ept: 188.0890
    Epoch [6/50], Val Losses: mse: 7.0876, mae: 1.4269, huber: 1.1020, swd: 4.4608, ept: 554.5904
    Epoch [6/50], Test Losses: mse: 5.6441, mae: 1.2648, huber: 0.9387, swd: 3.5481, ept: 578.1715
      Epoch 6 composite train-obj: 1.256121
            Val objective improved 1.1317 → 1.1020, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 8.4386, mae: 1.5607, huber: 1.2320, swd: 3.6689, ept: 188.4558
    Epoch [7/50], Val Losses: mse: 6.9700, mae: 1.3828, huber: 1.0551, swd: 4.3095, ept: 558.7914
    Epoch [7/50], Test Losses: mse: 5.6290, mae: 1.2588, huber: 0.9266, swd: 3.4531, ept: 573.4168
      Epoch 7 composite train-obj: 1.232042
            Val objective improved 1.1020 → 1.0551, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 8.3093, mae: 1.5373, huber: 1.2121, swd: 3.6073, ept: 189.0211
    Epoch [8/50], Val Losses: mse: 7.2566, mae: 1.3485, huber: 1.0413, swd: 4.2254, ept: 569.9922
    Epoch [8/50], Test Losses: mse: 5.9405, mae: 1.2519, huber: 0.9346, swd: 3.3527, ept: 573.0717
      Epoch 8 composite train-obj: 1.212136
            Val objective improved 1.0551 → 1.0413, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 8.2018, mae: 1.5200, huber: 1.1964, swd: 3.5331, ept: 189.1580
    Epoch [9/50], Val Losses: mse: 6.6773, mae: 1.3518, huber: 1.0282, swd: 4.0478, ept: 569.3077
    Epoch [9/50], Test Losses: mse: 5.3176, mae: 1.2095, huber: 0.8847, swd: 3.1601, ept: 581.8018
      Epoch 9 composite train-obj: 1.196429
            Val objective improved 1.0413 → 1.0282, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 8.0835, mae: 1.5029, huber: 1.1817, swd: 3.4558, ept: 189.3389
    Epoch [10/50], Val Losses: mse: 6.7320, mae: 1.3742, huber: 1.0440, swd: 4.0070, ept: 564.8543
    Epoch [10/50], Test Losses: mse: 5.4343, mae: 1.2498, huber: 0.9150, swd: 3.1793, ept: 582.4483
      Epoch 10 composite train-obj: 1.181667
            No improvement (1.0440), counter 1/5
    Epoch [11/50], Train Losses: mse: 7.8790, mae: 1.4822, huber: 1.1617, swd: 3.3056, ept: 189.0091
    Epoch [11/50], Val Losses: mse: 6.1707, mae: 1.2687, huber: 0.9606, swd: 3.5700, ept: 561.5825
    Epoch [11/50], Test Losses: mse: 5.0049, mae: 1.1542, huber: 0.8420, swd: 2.8392, ept: 574.1692
      Epoch 11 composite train-obj: 1.161692
            Val objective improved 1.0282 → 0.9606, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 7.5122, mae: 1.4567, huber: 1.1363, swd: 2.9508, ept: 189.2061
    Epoch [12/50], Val Losses: mse: 7.2338, mae: 1.4466, huber: 1.1189, swd: 3.7988, ept: 560.8505
    Epoch [12/50], Test Losses: mse: 6.9298, mae: 1.3880, huber: 1.0542, swd: 4.0145, ept: 558.7828
      Epoch 12 composite train-obj: 1.136308
            No improvement (1.1189), counter 1/5
    Epoch [13/50], Train Losses: mse: 7.2135, mae: 1.4282, huber: 1.1107, swd: 2.6875, ept: 189.4757
    Epoch [13/50], Val Losses: mse: 5.8006, mae: 1.2628, huber: 0.9463, swd: 3.0800, ept: 570.2815
    Epoch [13/50], Test Losses: mse: 4.8060, mae: 1.1543, huber: 0.8359, swd: 2.4677, ept: 579.6748
      Epoch 13 composite train-obj: 1.110733
            Val objective improved 0.9606 → 0.9463, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 7.0633, mae: 1.4078, huber: 1.0932, swd: 2.5538, ept: 189.6212
    Epoch [14/50], Val Losses: mse: 5.5595, mae: 1.2294, huber: 0.9177, swd: 2.6912, ept: 578.8963
    Epoch [14/50], Test Losses: mse: 4.5842, mae: 1.1175, huber: 0.8045, swd: 2.1446, ept: 590.3223
      Epoch 14 composite train-obj: 1.093213
            Val objective improved 0.9463 → 0.9177, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 6.9658, mae: 1.3942, huber: 1.0810, swd: 2.4679, ept: 189.3522
    Epoch [15/50], Val Losses: mse: 5.5195, mae: 1.2979, huber: 0.9704, swd: 2.7686, ept: 577.0049
    Epoch [15/50], Test Losses: mse: 4.5126, mae: 1.1938, huber: 0.8620, swd: 2.1893, ept: 577.2637
      Epoch 15 composite train-obj: 1.080965
            No improvement (0.9704), counter 1/5
    Epoch [16/50], Train Losses: mse: 6.8652, mae: 1.3762, huber: 1.0655, swd: 2.3963, ept: 189.4735
    Epoch [16/50], Val Losses: mse: 5.4562, mae: 1.2917, huber: 0.9624, swd: 2.6306, ept: 576.0465
    Epoch [16/50], Test Losses: mse: 4.5727, mae: 1.1956, huber: 0.8630, swd: 2.1056, ept: 587.3724
      Epoch 16 composite train-obj: 1.065487
            No improvement (0.9624), counter 2/5
    Epoch [17/50], Train Losses: mse: 6.8603, mae: 1.3791, huber: 1.0667, swd: 2.3936, ept: 189.2390
    Epoch [17/50], Val Losses: mse: 5.4605, mae: 1.3732, huber: 1.0275, swd: 2.8413, ept: 565.5413
    Epoch [17/50], Test Losses: mse: 4.5296, mae: 1.2530, huber: 0.9065, swd: 2.3021, ept: 589.2517
      Epoch 17 composite train-obj: 1.066690
            No improvement (1.0275), counter 3/5
    Epoch [18/50], Train Losses: mse: 6.7920, mae: 1.3673, huber: 1.0569, swd: 2.3399, ept: 189.0902
    Epoch [18/50], Val Losses: mse: 5.4279, mae: 1.2785, huber: 0.9543, swd: 2.8172, ept: 564.3115
    Epoch [18/50], Test Losses: mse: 4.5120, mae: 1.1823, huber: 0.8551, swd: 2.2831, ept: 576.6025
      Epoch 18 composite train-obj: 1.056925
            No improvement (0.9543), counter 4/5
    Epoch [19/50], Train Losses: mse: 6.7235, mae: 1.3545, huber: 1.0460, swd: 2.2959, ept: 189.2993
    Epoch [19/50], Val Losses: mse: 5.4629, mae: 1.2740, huber: 0.9400, swd: 2.7812, ept: 573.6269
    Epoch [19/50], Test Losses: mse: 4.5268, mae: 1.2041, huber: 0.8615, swd: 2.1830, ept: 567.4962
      Epoch 19 composite train-obj: 1.046038
    Epoch [19/50], Test Losses: mse: 4.5842, mae: 1.1175, huber: 0.8045, swd: 2.1446, ept: 590.3223
    Best round's Test MSE: 4.5842, MAE: 1.1175, SWD: 2.1446
    Best round's Validation MSE: 5.5595, MAE: 1.2294, SWD: 2.6912
    Best round's Test verification MSE : 4.5842, MAE: 1.1175, SWD: 2.1446
    Time taken: 112.91 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 11.3499, mae: 2.1108, huber: 1.7333, swd: 4.9034, ept: 169.0551
    Epoch [1/50], Val Losses: mse: 8.9204, mae: 1.7977, huber: 1.4337, swd: 5.1101, ept: 492.2041
    Epoch [1/50], Test Losses: mse: 7.3618, mae: 1.6501, huber: 1.2812, swd: 4.1120, ept: 500.4023
      Epoch 1 composite train-obj: 1.733305
            Val objective improved inf → 1.4337, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 9.4972, mae: 1.8003, huber: 1.4414, swd: 4.2332, ept: 185.2612
    Epoch [2/50], Val Losses: mse: 8.8212, mae: 1.6472, huber: 1.2963, swd: 5.0025, ept: 538.3034
    Epoch [2/50], Test Losses: mse: 7.2896, mae: 1.5310, huber: 1.1726, swd: 4.0123, ept: 538.9466
      Epoch 2 composite train-obj: 1.441361
            Val objective improved 1.4337 → 1.2963, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 9.1736, mae: 1.7236, huber: 1.3737, swd: 4.1272, ept: 187.4672
    Epoch [3/50], Val Losses: mse: 8.5324, mae: 1.5570, huber: 1.2222, swd: 5.1220, ept: 540.2375
    Epoch [3/50], Test Losses: mse: 7.2550, mae: 1.4636, huber: 1.1224, swd: 4.2518, ept: 544.6473
      Epoch 3 composite train-obj: 1.373700
            Val objective improved 1.2963 → 1.2222, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 8.9800, mae: 1.6757, huber: 1.3322, swd: 4.0461, ept: 187.9552
    Epoch [4/50], Val Losses: mse: 7.5777, mae: 1.4787, huber: 1.1524, swd: 4.7326, ept: 544.6471
    Epoch [4/50], Test Losses: mse: 6.1343, mae: 1.3358, huber: 1.0072, swd: 3.7580, ept: 560.3603
      Epoch 4 composite train-obj: 1.332178
            Val objective improved 1.2222 → 1.1524, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 8.7830, mae: 1.6364, huber: 1.2981, swd: 3.9357, ept: 188.4828
    Epoch [5/50], Val Losses: mse: 7.5209, mae: 1.5400, huber: 1.1906, swd: 4.6918, ept: 551.3495
    Epoch [5/50], Test Losses: mse: 6.1428, mae: 1.4009, huber: 1.0492, swd: 3.7687, ept: 563.3216
      Epoch 5 composite train-obj: 1.298147
            No improvement (1.1906), counter 1/5
    Epoch [6/50], Train Losses: mse: 8.6299, mae: 1.6065, huber: 1.2719, swd: 3.8497, ept: 188.7813
    Epoch [6/50], Val Losses: mse: 7.8115, mae: 1.4565, huber: 1.1354, swd: 4.7378, ept: 549.5071
    Epoch [6/50], Test Losses: mse: 6.4996, mae: 1.3666, huber: 1.0378, swd: 3.8689, ept: 556.6663
      Epoch 6 composite train-obj: 1.271903
            Val objective improved 1.1524 → 1.1354, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 8.4928, mae: 1.5803, huber: 1.2488, swd: 3.7574, ept: 188.9729
    Epoch [7/50], Val Losses: mse: 7.3695, mae: 1.5384, huber: 1.1859, swd: 4.5986, ept: 554.0671
    Epoch [7/50], Test Losses: mse: 6.0390, mae: 1.4103, huber: 1.0544, swd: 3.7345, ept: 570.7985
      Epoch 7 composite train-obj: 1.248845
            No improvement (1.1859), counter 1/5
    Epoch [8/50], Train Losses: mse: 8.3297, mae: 1.5588, huber: 1.2289, swd: 3.6481, ept: 189.0475
    Epoch [8/50], Val Losses: mse: 6.9358, mae: 1.4010, huber: 1.0776, swd: 4.3351, ept: 558.8117
    Epoch [8/50], Test Losses: mse: 5.6227, mae: 1.2716, huber: 0.9441, swd: 3.4403, ept: 574.4987
      Epoch 8 composite train-obj: 1.228887
            Val objective improved 1.1354 → 1.0776, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 8.1133, mae: 1.5286, huber: 1.2021, swd: 3.4905, ept: 189.4022
    Epoch [9/50], Val Losses: mse: 6.5634, mae: 1.3454, huber: 1.0230, swd: 3.8082, ept: 569.5740
    Epoch [9/50], Test Losses: mse: 5.2736, mae: 1.1965, huber: 0.8750, swd: 3.0271, ept: 585.1632
      Epoch 9 composite train-obj: 1.202139
            Val objective improved 1.0776 → 1.0230, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.8025, mae: 1.4981, huber: 1.1733, swd: 3.2119, ept: 189.6378
    Epoch [10/50], Val Losses: mse: 6.1295, mae: 1.2845, huber: 0.9664, swd: 3.4555, ept: 564.6811
    Epoch [10/50], Test Losses: mse: 5.0176, mae: 1.1702, huber: 0.8529, swd: 2.7285, ept: 583.7207
      Epoch 10 composite train-obj: 1.173251
            Val objective improved 1.0230 → 0.9664, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 7.5942, mae: 1.4789, huber: 1.1545, swd: 3.0038, ept: 189.4877
    Epoch [11/50], Val Losses: mse: 5.9390, mae: 1.2994, huber: 0.9732, swd: 3.2701, ept: 569.0895
    Epoch [11/50], Test Losses: mse: 4.8423, mae: 1.1834, huber: 0.8558, swd: 2.5878, ept: 589.4000
      Epoch 11 composite train-obj: 1.154550
            No improvement (0.9732), counter 1/5
    Epoch [12/50], Train Losses: mse: 7.4419, mae: 1.4578, huber: 1.1357, swd: 2.8792, ept: 189.3590
    Epoch [12/50], Val Losses: mse: 6.0355, mae: 1.3741, huber: 1.0412, swd: 3.3159, ept: 565.1209
    Epoch [12/50], Test Losses: mse: 4.8950, mae: 1.2367, huber: 0.9058, swd: 2.5621, ept: 579.0692
      Epoch 12 composite train-obj: 1.135704
            No improvement (1.0412), counter 2/5
    Epoch [13/50], Train Losses: mse: 7.3231, mae: 1.4434, huber: 1.1221, swd: 2.7862, ept: 189.4812
    Epoch [13/50], Val Losses: mse: 5.5914, mae: 1.3149, huber: 0.9765, swd: 2.9841, ept: 579.5215
    Epoch [13/50], Test Losses: mse: 4.4867, mae: 1.1821, huber: 0.8445, swd: 2.3150, ept: 595.2168
      Epoch 13 composite train-obj: 1.122076
            No improvement (0.9765), counter 3/5
    Epoch [14/50], Train Losses: mse: 7.2323, mae: 1.4293, huber: 1.1100, swd: 2.7059, ept: 189.6416
    Epoch [14/50], Val Losses: mse: 5.2758, mae: 1.2218, huber: 0.8996, swd: 2.8639, ept: 565.3591
    Epoch [14/50], Test Losses: mse: 4.2319, mae: 1.1030, huber: 0.7801, swd: 2.2188, ept: 580.1471
      Epoch 14 composite train-obj: 1.109962
            Val objective improved 0.9664 → 0.8996, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 7.0940, mae: 1.4106, huber: 1.0934, swd: 2.6085, ept: 189.4164
    Epoch [15/50], Val Losses: mse: 5.5387, mae: 1.3044, huber: 0.9670, swd: 2.8609, ept: 580.2941
    Epoch [15/50], Test Losses: mse: 4.4796, mae: 1.1833, huber: 0.8476, swd: 2.1753, ept: 593.0238
      Epoch 15 composite train-obj: 1.093396
            No improvement (0.9670), counter 1/5
    Epoch [16/50], Train Losses: mse: 7.0190, mae: 1.3986, huber: 1.0827, swd: 2.5532, ept: 189.4365
    Epoch [16/50], Val Losses: mse: 5.3720, mae: 1.2538, huber: 0.9214, swd: 2.6857, ept: 585.0636
    Epoch [16/50], Test Losses: mse: 4.3941, mae: 1.1394, huber: 0.8092, swd: 2.0754, ept: 598.9095
      Epoch 16 composite train-obj: 1.082699
            No improvement (0.9214), counter 2/5
    Epoch [17/50], Train Losses: mse: 6.9504, mae: 1.3883, huber: 1.0732, swd: 2.5000, ept: 189.3768
    Epoch [17/50], Val Losses: mse: 5.0588, mae: 1.2613, huber: 0.9314, swd: 2.6958, ept: 572.5945
    Epoch [17/50], Test Losses: mse: 4.0204, mae: 1.1359, huber: 0.8038, swd: 2.0915, ept: 600.9617
      Epoch 17 composite train-obj: 1.073167
            No improvement (0.9314), counter 3/5
    Epoch [18/50], Train Losses: mse: 6.8694, mae: 1.3766, huber: 1.0633, swd: 2.4382, ept: 189.6110
    Epoch [18/50], Val Losses: mse: 5.0003, mae: 1.1661, huber: 0.8549, swd: 2.5285, ept: 588.9768
    Epoch [18/50], Test Losses: mse: 4.0961, mae: 1.0517, huber: 0.7450, swd: 1.9497, ept: 598.7599
      Epoch 18 composite train-obj: 1.063282
            Val objective improved 0.8996 → 0.8549, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 6.8386, mae: 1.3703, huber: 1.0579, swd: 2.4078, ept: 189.5182
    Epoch [19/50], Val Losses: mse: 4.6531, mae: 1.1542, huber: 0.8415, swd: 2.4166, ept: 583.8763
    Epoch [19/50], Test Losses: mse: 3.6905, mae: 1.0349, huber: 0.7204, swd: 1.8479, ept: 599.0092
      Epoch 19 composite train-obj: 1.057936
            Val objective improved 0.8549 → 0.8415, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 6.7223, mae: 1.3529, huber: 1.0430, swd: 2.3421, ept: 189.3955
    Epoch [20/50], Val Losses: mse: 5.9165, mae: 1.2254, huber: 0.9142, swd: 2.8497, ept: 585.6598
    Epoch [20/50], Test Losses: mse: 4.6803, mae: 1.0996, huber: 0.7939, swd: 2.1181, ept: 586.0494
      Epoch 20 composite train-obj: 1.043025
            No improvement (0.9142), counter 1/5
    Epoch [21/50], Train Losses: mse: 6.6871, mae: 1.3502, huber: 1.0397, swd: 2.3010, ept: 189.1738
    Epoch [21/50], Val Losses: mse: 4.7560, mae: 1.2034, huber: 0.8732, swd: 2.2809, ept: 591.3288
    Epoch [21/50], Test Losses: mse: 3.8645, mae: 1.1033, huber: 0.7688, swd: 1.7556, ept: 607.0034
      Epoch 21 composite train-obj: 1.039684
            No improvement (0.8732), counter 2/5
    Epoch [22/50], Train Losses: mse: 6.6242, mae: 1.3357, huber: 1.0279, swd: 2.2653, ept: 189.4743
    Epoch [22/50], Val Losses: mse: 4.7219, mae: 1.1098, huber: 0.8113, swd: 2.2678, ept: 582.1005
    Epoch [22/50], Test Losses: mse: 3.7857, mae: 0.9903, huber: 0.6939, swd: 1.7409, ept: 602.2814
      Epoch 22 composite train-obj: 1.027858
            Val objective improved 0.8415 → 0.8113, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 6.6188, mae: 1.3360, huber: 1.0281, swd: 2.2529, ept: 189.3570
    Epoch [23/50], Val Losses: mse: 5.0521, mae: 1.1425, huber: 0.8359, swd: 2.4303, ept: 585.2599
    Epoch [23/50], Test Losses: mse: 4.1141, mae: 1.0424, huber: 0.7365, swd: 1.8909, ept: 596.3680
      Epoch 23 composite train-obj: 1.028115
            No improvement (0.8359), counter 1/5
    Epoch [24/50], Train Losses: mse: 6.6016, mae: 1.3327, huber: 1.0245, swd: 2.2347, ept: 189.5053
    Epoch [24/50], Val Losses: mse: 5.5011, mae: 1.2686, huber: 0.9328, swd: 2.4962, ept: 592.4910
    Epoch [24/50], Test Losses: mse: 4.4900, mae: 1.1770, huber: 0.8337, swd: 1.9565, ept: 598.8754
      Epoch 24 composite train-obj: 1.024476
            No improvement (0.9328), counter 2/5
    Epoch [25/50], Train Losses: mse: 6.5361, mae: 1.3186, huber: 1.0132, swd: 2.1915, ept: 189.2507
    Epoch [25/50], Val Losses: mse: 4.8703, mae: 1.2063, huber: 0.8840, swd: 2.3754, ept: 563.3652
    Epoch [25/50], Test Losses: mse: 3.9425, mae: 1.0916, huber: 0.7676, swd: 1.8137, ept: 587.0908
      Epoch 25 composite train-obj: 1.013184
            No improvement (0.8840), counter 3/5
    Epoch [26/50], Train Losses: mse: 6.4835, mae: 1.3155, huber: 1.0097, swd: 2.1536, ept: 189.5444
    Epoch [26/50], Val Losses: mse: 5.1062, mae: 1.2066, huber: 0.8836, swd: 2.3965, ept: 603.7207
    Epoch [26/50], Test Losses: mse: 4.0419, mae: 1.0848, huber: 0.7591, swd: 1.8247, ept: 613.1310
      Epoch 26 composite train-obj: 1.009750
            No improvement (0.8836), counter 4/5
    Epoch [27/50], Train Losses: mse: 6.4647, mae: 1.3059, huber: 1.0024, swd: 2.1533, ept: 189.7239
    Epoch [27/50], Val Losses: mse: 4.8534, mae: 1.1072, huber: 0.8065, swd: 2.3393, ept: 603.8257
    Epoch [27/50], Test Losses: mse: 3.8900, mae: 1.0058, huber: 0.7051, swd: 1.8564, ept: 603.7921
      Epoch 27 composite train-obj: 1.002450
            Val objective improved 0.8113 → 0.8065, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 6.4190, mae: 1.2997, huber: 0.9971, swd: 2.1165, ept: 189.7493
    Epoch [28/50], Val Losses: mse: 4.2195, mae: 1.1517, huber: 0.8253, swd: 2.0989, ept: 602.7849
    Epoch [28/50], Test Losses: mse: 3.4694, mae: 1.0607, huber: 0.7313, swd: 1.6455, ept: 612.4799
      Epoch 28 composite train-obj: 0.997056
            No improvement (0.8253), counter 1/5
    Epoch [29/50], Train Losses: mse: 6.4041, mae: 1.2981, huber: 0.9947, swd: 2.1005, ept: 189.7368
    Epoch [29/50], Val Losses: mse: 5.3620, mae: 1.2555, huber: 0.9291, swd: 2.3612, ept: 585.4052
    Epoch [29/50], Test Losses: mse: 4.5508, mae: 1.1733, huber: 0.8429, swd: 1.9493, ept: 589.1420
      Epoch 29 composite train-obj: 0.994748
            No improvement (0.9291), counter 2/5
    Epoch [30/50], Train Losses: mse: 6.4260, mae: 1.3022, huber: 0.9981, swd: 2.1038, ept: 189.5803
    Epoch [30/50], Val Losses: mse: 4.5325, mae: 1.1803, huber: 0.8521, swd: 2.1638, ept: 573.7540
    Epoch [30/50], Test Losses: mse: 3.7351, mae: 1.0999, huber: 0.7666, swd: 1.7322, ept: 575.0254
      Epoch 30 composite train-obj: 0.998051
            No improvement (0.8521), counter 3/5
    Epoch [31/50], Train Losses: mse: 6.3367, mae: 1.2879, huber: 0.9863, swd: 2.0485, ept: 189.4239
    Epoch [31/50], Val Losses: mse: 5.1398, mae: 1.3257, huber: 0.9797, swd: 2.3465, ept: 553.3429
    Epoch [31/50], Test Losses: mse: 4.3321, mae: 1.2498, huber: 0.8976, swd: 1.8242, ept: 553.4806
      Epoch 31 composite train-obj: 0.986343
            No improvement (0.9797), counter 4/5
    Epoch [32/50], Train Losses: mse: 6.2633, mae: 1.2747, huber: 0.9746, swd: 2.0159, ept: 189.6231
    Epoch [32/50], Val Losses: mse: 4.4345, mae: 1.1205, huber: 0.8080, swd: 2.1512, ept: 528.4740
    Epoch [32/50], Test Losses: mse: 3.6282, mae: 1.0376, huber: 0.7203, swd: 1.6548, ept: 525.5497
      Epoch 32 composite train-obj: 0.974643
    Epoch [32/50], Test Losses: mse: 3.8900, mae: 1.0058, huber: 0.7051, swd: 1.8564, ept: 603.7921
    Best round's Test MSE: 3.8900, MAE: 1.0058, SWD: 1.8564
    Best round's Validation MSE: 4.8534, MAE: 1.1072, SWD: 2.3393
    Best round's Test verification MSE : 3.8900, MAE: 1.0058, SWD: 1.8564
    Time taken: 186.84 seconds
    
    ==================================================
    Experiment Summary (PatchTST_rossler_seq336_pred720_20250513_1159)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 4.1788 ± 0.2951
      mae: 1.0662 ± 0.0460
      huber: 0.7566 ± 0.0407
      swd: 2.0194 ± 0.1207
      ept: 597.0664 ± 5.4990
      count: 33.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 5.1414 ± 0.3026
      mae: 1.1755 ± 0.0509
      huber: 0.8670 ± 0.0459
      swd: 2.5322 ± 0.1456
      ept: 589.0458 ± 10.6911
      count: 33.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 436.53 seconds
    
    Experiment complete: PatchTST_rossler_seq336_pred720_20250513_1159
    Model: PatchTST
    Dataset: rossler
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([96, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 282
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 96, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 3.7384, mae: 0.7961, huber: 0.5433, swd: 2.9740, ept: 89.3038
    Epoch [1/50], Val Losses: mse: 3.1726, mae: 0.7439, huber: 0.4694, swd: 2.6848, ept: 92.2651
    Epoch [1/50], Test Losses: mse: 2.9170, mae: 0.7531, huber: 0.4747, swd: 2.4588, ept: 92.2918
      Epoch 1 composite train-obj: 0.543299
            Val objective improved inf → 0.4694, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.5119, mae: 0.5680, huber: 0.3482, swd: 2.0821, ept: 92.8545
    Epoch [2/50], Val Losses: mse: 2.7803, mae: 0.6159, huber: 0.3790, swd: 2.2479, ept: 93.0071
    Epoch [2/50], Test Losses: mse: 2.5431, mae: 0.6181, huber: 0.3807, swd: 2.0348, ept: 92.8511
      Epoch 2 composite train-obj: 0.348185
            Val objective improved 0.4694 → 0.3790, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.3792, mae: 0.5197, huber: 0.3167, swd: 1.9651, ept: 93.1566
    Epoch [3/50], Val Losses: mse: 2.6812, mae: 0.6011, huber: 0.3749, swd: 2.2247, ept: 93.0066
    Epoch [3/50], Test Losses: mse: 2.4462, mae: 0.6113, huber: 0.3779, swd: 2.0130, ept: 92.9330
      Epoch 3 composite train-obj: 0.316735
            Val objective improved 0.3790 → 0.3749, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 2.1908, mae: 0.4869, huber: 0.2927, swd: 1.7962, ept: 93.3768
    Epoch [4/50], Val Losses: mse: 2.5160, mae: 0.5225, huber: 0.3175, swd: 2.0575, ept: 93.1210
    Epoch [4/50], Test Losses: mse: 2.2922, mae: 0.5257, huber: 0.3204, swd: 1.8615, ept: 93.0863
      Epoch 4 composite train-obj: 0.292739
            Val objective improved 0.3749 → 0.3175, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 2.1233, mae: 0.4629, huber: 0.2778, swd: 1.7395, ept: 93.4644
    Epoch [5/50], Val Losses: mse: 2.5444, mae: 0.4959, huber: 0.3196, swd: 2.0391, ept: 93.1645
    Epoch [5/50], Test Losses: mse: 2.3286, mae: 0.5096, huber: 0.3265, swd: 1.8501, ept: 93.0977
      Epoch 5 composite train-obj: 0.277753
            No improvement (0.3196), counter 1/5
    Epoch [6/50], Train Losses: mse: 2.0737, mae: 0.4505, huber: 0.2699, swd: 1.6946, ept: 93.5331
    Epoch [6/50], Val Losses: mse: 2.6554, mae: 0.7060, huber: 0.4230, swd: 2.1526, ept: 93.2652
    Epoch [6/50], Test Losses: mse: 2.4716, mae: 0.7290, huber: 0.4375, swd: 1.9894, ept: 93.1873
      Epoch 6 composite train-obj: 0.269850
            No improvement (0.4230), counter 2/5
    Epoch [7/50], Train Losses: mse: 2.0532, mae: 0.4490, huber: 0.2693, swd: 1.6711, ept: 93.5565
    Epoch [7/50], Val Losses: mse: 2.3084, mae: 0.4821, huber: 0.2860, swd: 1.8657, ept: 93.4733
    Epoch [7/50], Test Losses: mse: 2.0928, mae: 0.4799, huber: 0.2848, swd: 1.6745, ept: 93.3532
      Epoch 7 composite train-obj: 0.269335
            Val objective improved 0.3175 → 0.2860, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 2.0002, mae: 0.4289, huber: 0.2562, swd: 1.6304, ept: 93.6357
    Epoch [8/50], Val Losses: mse: 2.4406, mae: 0.4826, huber: 0.3072, swd: 1.9042, ept: 93.3867
    Epoch [8/50], Test Losses: mse: 2.2415, mae: 0.4952, huber: 0.3147, swd: 1.7269, ept: 93.2619
      Epoch 8 composite train-obj: 0.256177
            No improvement (0.3072), counter 1/5
    Epoch [9/50], Train Losses: mse: 1.9743, mae: 0.4209, huber: 0.2521, swd: 1.6063, ept: 93.6731
    Epoch [9/50], Val Losses: mse: 2.3256, mae: 0.5071, huber: 0.3047, swd: 1.8507, ept: 93.5376
    Epoch [9/50], Test Losses: mse: 2.1290, mae: 0.5131, huber: 0.3087, swd: 1.6740, ept: 93.3857
      Epoch 9 composite train-obj: 0.252087
            No improvement (0.3047), counter 2/5
    Epoch [10/50], Train Losses: mse: 1.9518, mae: 0.4173, huber: 0.2487, swd: 1.5862, ept: 93.7028
    Epoch [10/50], Val Losses: mse: 2.3151, mae: 0.4914, huber: 0.2967, swd: 1.8926, ept: 93.5637
    Epoch [10/50], Test Losses: mse: 2.1012, mae: 0.4971, huber: 0.2963, swd: 1.7027, ept: 93.4488
      Epoch 10 composite train-obj: 0.248736
            No improvement (0.2967), counter 3/5
    Epoch [11/50], Train Losses: mse: 1.9396, mae: 0.4132, huber: 0.2459, swd: 1.5757, ept: 93.6931
    Epoch [11/50], Val Losses: mse: 2.2372, mae: 0.4229, huber: 0.2661, swd: 1.8234, ept: 93.5169
    Epoch [11/50], Test Losses: mse: 2.0289, mae: 0.4271, huber: 0.2677, swd: 1.6407, ept: 93.4414
      Epoch 11 composite train-obj: 0.245896
            Val objective improved 0.2860 → 0.2661, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 1.9188, mae: 0.4064, huber: 0.2416, swd: 1.5580, ept: 93.7465
    Epoch [12/50], Val Losses: mse: 2.2751, mae: 0.4979, huber: 0.2958, swd: 1.8634, ept: 93.6893
    Epoch [12/50], Test Losses: mse: 2.0749, mae: 0.5038, huber: 0.2970, swd: 1.6823, ept: 93.5175
      Epoch 12 composite train-obj: 0.241645
            No improvement (0.2958), counter 1/5
    Epoch [13/50], Train Losses: mse: 1.9072, mae: 0.4024, huber: 0.2394, swd: 1.5472, ept: 93.7569
    Epoch [13/50], Val Losses: mse: 2.3053, mae: 0.4605, huber: 0.2867, swd: 1.8250, ept: 93.4633
    Epoch [13/50], Test Losses: mse: 2.1095, mae: 0.4718, huber: 0.2926, swd: 1.6531, ept: 93.3700
      Epoch 13 composite train-obj: 0.239397
            No improvement (0.2867), counter 2/5
    Epoch [14/50], Train Losses: mse: 1.8939, mae: 0.3985, huber: 0.2374, swd: 1.5342, ept: 93.7770
    Epoch [14/50], Val Losses: mse: 2.1900, mae: 0.4567, huber: 0.2669, swd: 1.7504, ept: 93.6251
    Epoch [14/50], Test Losses: mse: 1.9864, mae: 0.4595, huber: 0.2672, swd: 1.5686, ept: 93.5055
      Epoch 14 composite train-obj: 0.237376
            No improvement (0.2669), counter 3/5
    Epoch [15/50], Train Losses: mse: 1.8828, mae: 0.3966, huber: 0.2353, swd: 1.5248, ept: 93.7520
    Epoch [15/50], Val Losses: mse: 2.3103, mae: 0.4715, huber: 0.2907, swd: 1.8101, ept: 93.4947
    Epoch [15/50], Test Losses: mse: 2.1179, mae: 0.4878, huber: 0.2983, swd: 1.6412, ept: 93.4213
      Epoch 15 composite train-obj: 0.235324
            No improvement (0.2907), counter 4/5
    Epoch [16/50], Train Losses: mse: 1.8715, mae: 0.3913, huber: 0.2330, swd: 1.5144, ept: 93.8071
    Epoch [16/50], Val Losses: mse: 2.1765, mae: 0.4174, huber: 0.2591, swd: 1.7405, ept: 93.6450
    Epoch [16/50], Test Losses: mse: 1.9765, mae: 0.4234, huber: 0.2607, swd: 1.5655, ept: 93.5309
      Epoch 16 composite train-obj: 0.233036
            Val objective improved 0.2661 → 0.2591, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 1.8622, mae: 0.3888, huber: 0.2312, swd: 1.5061, ept: 93.8180
    Epoch [17/50], Val Losses: mse: 2.2708, mae: 0.4845, huber: 0.2883, swd: 1.8161, ept: 93.5630
    Epoch [17/50], Test Losses: mse: 2.0742, mae: 0.4971, huber: 0.2937, swd: 1.6447, ept: 93.4701
      Epoch 17 composite train-obj: 0.231175
            No improvement (0.2883), counter 1/5
    Epoch [18/50], Train Losses: mse: 1.8563, mae: 0.3883, huber: 0.2306, swd: 1.4991, ept: 93.8253
    Epoch [18/50], Val Losses: mse: 2.1356, mae: 0.4282, huber: 0.2532, swd: 1.7328, ept: 93.7104
    Epoch [18/50], Test Losses: mse: 1.9349, mae: 0.4323, huber: 0.2534, swd: 1.5549, ept: 93.5759
      Epoch 18 composite train-obj: 0.230603
            Val objective improved 0.2591 → 0.2532, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 1.8462, mae: 0.3850, huber: 0.2282, swd: 1.4919, ept: 93.8336
    Epoch [19/50], Val Losses: mse: 2.1781, mae: 0.4661, huber: 0.2716, swd: 1.7522, ept: 93.6540
    Epoch [19/50], Test Losses: mse: 1.9819, mae: 0.4727, huber: 0.2746, swd: 1.5791, ept: 93.5565
      Epoch 19 composite train-obj: 0.228231
            No improvement (0.2716), counter 1/5
    Epoch [20/50], Train Losses: mse: 1.8361, mae: 0.3825, huber: 0.2269, swd: 1.4835, ept: 93.8486
    Epoch [20/50], Val Losses: mse: 2.1722, mae: 0.4076, huber: 0.2570, swd: 1.7390, ept: 93.5914
    Epoch [20/50], Test Losses: mse: 1.9730, mae: 0.4152, huber: 0.2594, swd: 1.5642, ept: 93.5058
      Epoch 20 composite train-obj: 0.226892
            No improvement (0.2570), counter 2/5
    Epoch [21/50], Train Losses: mse: 1.8332, mae: 0.3808, huber: 0.2264, swd: 1.4800, ept: 93.8572
    Epoch [21/50], Val Losses: mse: 2.6590, mae: 0.7388, huber: 0.4515, swd: 1.9170, ept: 93.3276
    Epoch [21/50], Test Losses: mse: 2.5190, mae: 0.7744, huber: 0.4742, swd: 1.7781, ept: 93.2079
      Epoch 21 composite train-obj: 0.226374
            No improvement (0.4515), counter 3/5
    Epoch [22/50], Train Losses: mse: 1.8635, mae: 0.4001, huber: 0.2376, swd: 1.4834, ept: 93.7690
    Epoch [22/50], Val Losses: mse: 2.1157, mae: 0.3946, huber: 0.2476, swd: 1.7109, ept: 93.6436
    Epoch [22/50], Test Losses: mse: 1.9167, mae: 0.3967, huber: 0.2488, swd: 1.5375, ept: 93.5661
      Epoch 22 composite train-obj: 0.237650
            Val objective improved 0.2532 → 0.2476, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 1.8167, mae: 0.3742, huber: 0.2235, swd: 1.4647, ept: 93.8778
    Epoch [23/50], Val Losses: mse: 2.3872, mae: 0.5861, huber: 0.3509, swd: 1.7853, ept: 93.5511
    Epoch [23/50], Test Losses: mse: 2.2286, mae: 0.6097, huber: 0.3644, swd: 1.6321, ept: 93.4168
      Epoch 23 composite train-obj: 0.223484
            No improvement (0.3509), counter 1/5
    Epoch [24/50], Train Losses: mse: 1.8239, mae: 0.3845, huber: 0.2281, swd: 1.4634, ept: 93.8654
    Epoch [24/50], Val Losses: mse: 2.0816, mae: 0.3910, huber: 0.2438, swd: 1.6813, ept: 93.7572
    Epoch [24/50], Test Losses: mse: 1.8831, mae: 0.3923, huber: 0.2431, swd: 1.5082, ept: 93.6380
      Epoch 24 composite train-obj: 0.228056
            Val objective improved 0.2476 → 0.2438, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 1.8085, mae: 0.3723, huber: 0.2217, swd: 1.4585, ept: 93.8857
    Epoch [25/50], Val Losses: mse: 2.1451, mae: 0.4244, huber: 0.2588, swd: 1.7026, ept: 93.7032
    Epoch [25/50], Test Losses: mse: 1.9516, mae: 0.4336, huber: 0.2617, swd: 1.5334, ept: 93.5790
      Epoch 25 composite train-obj: 0.221732
            No improvement (0.2588), counter 1/5
    Epoch [26/50], Train Losses: mse: 1.8058, mae: 0.3734, huber: 0.2219, swd: 1.4553, ept: 93.8865
    Epoch [26/50], Val Losses: mse: 2.0737, mae: 0.4010, huber: 0.2408, swd: 1.6710, ept: 93.7372
    Epoch [26/50], Test Losses: mse: 1.8765, mae: 0.4004, huber: 0.2410, swd: 1.4985, ept: 93.6364
      Epoch 26 composite train-obj: 0.221884
            Val objective improved 0.2438 → 0.2408, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 1.8009, mae: 0.3735, huber: 0.2213, swd: 1.4501, ept: 93.8942
    Epoch [27/50], Val Losses: mse: 2.0782, mae: 0.3976, huber: 0.2426, swd: 1.6736, ept: 93.7361
    Epoch [27/50], Test Losses: mse: 1.8815, mae: 0.3997, huber: 0.2436, swd: 1.5000, ept: 93.6390
      Epoch 27 composite train-obj: 0.221300
            No improvement (0.2426), counter 1/5
    Epoch [28/50], Train Losses: mse: 1.7960, mae: 0.3698, huber: 0.2196, swd: 1.4475, ept: 93.8992
    Epoch [28/50], Val Losses: mse: 2.1330, mae: 0.4184, huber: 0.2562, swd: 1.7124, ept: 93.6787
    Epoch [28/50], Test Losses: mse: 1.9378, mae: 0.4300, huber: 0.2591, swd: 1.5415, ept: 93.5782
      Epoch 28 composite train-obj: 0.219569
            No improvement (0.2562), counter 2/5
    Epoch [29/50], Train Losses: mse: 1.8182, mae: 0.3765, huber: 0.2249, swd: 1.4614, ept: 93.8267
    Epoch [29/50], Val Losses: mse: 2.3985, mae: 0.6791, huber: 0.3967, swd: 1.8963, ept: 93.9120
    Epoch [29/50], Test Losses: mse: 2.2350, mae: 0.6979, huber: 0.4086, swd: 1.7385, ept: 93.7300
      Epoch 29 composite train-obj: 0.224935
            No improvement (0.3967), counter 3/5
    Epoch [30/50], Train Losses: mse: 1.8084, mae: 0.3831, huber: 0.2270, swd: 1.4470, ept: 93.9143
    Epoch [30/50], Val Losses: mse: 2.3089, mae: 0.5868, huber: 0.3450, swd: 1.7797, ept: 93.6830
    Epoch [30/50], Test Losses: mse: 2.1447, mae: 0.6080, huber: 0.3571, swd: 1.6317, ept: 93.5291
      Epoch 30 composite train-obj: 0.227004
            No improvement (0.3450), counter 4/5
    Epoch [31/50], Train Losses: mse: 1.7948, mae: 0.3798, huber: 0.2242, swd: 1.4377, ept: 93.9056
    Epoch [31/50], Val Losses: mse: 2.1894, mae: 0.5186, huber: 0.3102, swd: 1.8073, ept: 93.9281
    Epoch [31/50], Test Losses: mse: 1.9998, mae: 0.5253, huber: 0.3125, swd: 1.6332, ept: 93.7594
      Epoch 31 composite train-obj: 0.224158
    Epoch [31/50], Test Losses: mse: 1.8765, mae: 0.4004, huber: 0.2410, swd: 1.4985, ept: 93.6364
    Best round's Test MSE: 1.8765, MAE: 0.4004, SWD: 1.4985
    Best round's Validation MSE: 2.0737, MAE: 0.4010, SWD: 1.6710
    Best round's Test verification MSE : 1.8765, MAE: 0.4004, SWD: 1.4985
    Time taken: 63.91 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 3.6638, mae: 0.7853, huber: 0.5339, swd: 2.9451, ept: 89.4287
    Epoch [1/50], Val Losses: mse: 3.1294, mae: 0.6348, huber: 0.4094, swd: 2.6466, ept: 92.0552
    Epoch [1/50], Test Losses: mse: 2.8541, mae: 0.6385, huber: 0.4131, swd: 2.4061, ept: 92.1549
      Epoch 1 composite train-obj: 0.533889
            Val objective improved inf → 0.4094, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.5038, mae: 0.5587, huber: 0.3425, swd: 2.1007, ept: 92.8661
    Epoch [2/50], Val Losses: mse: 2.7326, mae: 0.6663, huber: 0.4011, swd: 2.3499, ept: 92.9994
    Epoch [2/50], Test Losses: mse: 2.4887, mae: 0.6744, huber: 0.4014, swd: 2.1313, ept: 92.9457
      Epoch 2 composite train-obj: 0.342498
            Val objective improved 0.4094 → 0.4011, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.3112, mae: 0.5160, huber: 0.3120, swd: 1.9317, ept: 93.1986
    Epoch [3/50], Val Losses: mse: 2.6257, mae: 0.5392, huber: 0.3379, swd: 2.2089, ept: 93.0154
    Epoch [3/50], Test Losses: mse: 2.3883, mae: 0.5440, huber: 0.3395, swd: 1.9999, ept: 92.9728
      Epoch 3 composite train-obj: 0.311970
            Val objective improved 0.4011 → 0.3379, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 2.1902, mae: 0.4800, huber: 0.2893, swd: 1.8290, ept: 93.3733
    Epoch [4/50], Val Losses: mse: 2.7412, mae: 0.5875, huber: 0.3671, swd: 2.2009, ept: 93.0016
    Epoch [4/50], Test Losses: mse: 2.5332, mae: 0.6068, huber: 0.3787, swd: 2.0122, ept: 92.9285
      Epoch 4 composite train-obj: 0.289314
            No improvement (0.3671), counter 1/5
    Epoch [5/50], Train Losses: mse: 2.1286, mae: 0.4668, huber: 0.2805, swd: 1.7715, ept: 93.4685
    Epoch [5/50], Val Losses: mse: 2.4787, mae: 0.5760, huber: 0.3471, swd: 2.1271, ept: 93.4659
    Epoch [5/50], Test Losses: mse: 2.2634, mae: 0.5837, huber: 0.3495, swd: 1.9269, ept: 93.3125
      Epoch 5 composite train-obj: 0.280519
            No improvement (0.3471), counter 2/5
    Epoch [6/50], Train Losses: mse: 2.0796, mae: 0.4534, huber: 0.2721, swd: 1.7317, ept: 93.4801
    Epoch [6/50], Val Losses: mse: 2.3734, mae: 0.4690, huber: 0.2928, swd: 2.0039, ept: 93.3355
    Epoch [6/50], Test Losses: mse: 2.1533, mae: 0.4712, huber: 0.2936, swd: 1.8067, ept: 93.2621
      Epoch 6 composite train-obj: 0.272082
            Val objective improved 0.3379 → 0.2928, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 2.0270, mae: 0.4357, huber: 0.2611, swd: 1.6892, ept: 93.5963
    Epoch [7/50], Val Losses: mse: 2.3736, mae: 0.5097, huber: 0.3094, swd: 2.0187, ept: 93.4698
    Epoch [7/50], Test Losses: mse: 2.1556, mae: 0.5154, huber: 0.3093, swd: 1.8185, ept: 93.3526
      Epoch 7 composite train-obj: 0.261074
            No improvement (0.3094), counter 1/5
    Epoch [8/50], Train Losses: mse: 2.0010, mae: 0.4300, huber: 0.2572, swd: 1.6667, ept: 93.6125
    Epoch [8/50], Val Losses: mse: 2.3071, mae: 0.4483, huber: 0.2773, swd: 1.9132, ept: 93.4457
    Epoch [8/50], Test Losses: mse: 2.0942, mae: 0.4498, huber: 0.2787, swd: 1.7222, ept: 93.3537
      Epoch 8 composite train-obj: 0.257228
            Val objective improved 0.2928 → 0.2773, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 1.9763, mae: 0.4229, huber: 0.2522, swd: 1.6467, ept: 93.6711
    Epoch [9/50], Val Losses: mse: 2.2608, mae: 0.4637, huber: 0.2816, swd: 1.9217, ept: 93.5489
    Epoch [9/50], Test Losses: mse: 2.0482, mae: 0.4653, huber: 0.2820, swd: 1.7279, ept: 93.4449
      Epoch 9 composite train-obj: 0.252233
            No improvement (0.2816), counter 1/5
    Epoch [10/50], Train Losses: mse: 1.9510, mae: 0.4159, huber: 0.2478, swd: 1.6240, ept: 93.6961
    Epoch [10/50], Val Losses: mse: 2.2409, mae: 0.4659, huber: 0.2739, swd: 1.8727, ept: 93.5578
    Epoch [10/50], Test Losses: mse: 2.0324, mae: 0.4670, huber: 0.2747, swd: 1.6844, ept: 93.4576
      Epoch 10 composite train-obj: 0.247762
            Val objective improved 0.2773 → 0.2739, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 1.9376, mae: 0.4117, huber: 0.2449, swd: 1.6139, ept: 93.7156
    Epoch [11/50], Val Losses: mse: 2.3458, mae: 0.4694, huber: 0.2929, swd: 1.9380, ept: 93.4577
    Epoch [11/50], Test Losses: mse: 2.1428, mae: 0.4814, huber: 0.2986, swd: 1.7553, ept: 93.3765
      Epoch 11 composite train-obj: 0.244871
            No improvement (0.2929), counter 1/5
    Epoch [12/50], Train Losses: mse: 1.9180, mae: 0.4043, huber: 0.2412, swd: 1.5964, ept: 93.7431
    Epoch [12/50], Val Losses: mse: 2.1888, mae: 0.4354, huber: 0.2611, swd: 1.8408, ept: 93.5974
    Epoch [12/50], Test Losses: mse: 1.9798, mae: 0.4322, huber: 0.2607, swd: 1.6513, ept: 93.5071
      Epoch 12 composite train-obj: 0.241166
            Val objective improved 0.2739 → 0.2611, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 1.9037, mae: 0.4008, huber: 0.2386, swd: 1.5854, ept: 93.7571
    Epoch [13/50], Val Losses: mse: 2.2439, mae: 0.4502, huber: 0.2737, swd: 1.8569, ept: 93.5088
    Epoch [13/50], Test Losses: mse: 2.0376, mae: 0.4578, huber: 0.2759, swd: 1.6702, ept: 93.4328
      Epoch 13 composite train-obj: 0.238580
            No improvement (0.2737), counter 1/5
    Epoch [14/50], Train Losses: mse: 1.8868, mae: 0.3968, huber: 0.2359, swd: 1.5707, ept: 93.7894
    Epoch [14/50], Val Losses: mse: 2.1627, mae: 0.4244, huber: 0.2554, swd: 1.8215, ept: 93.6478
    Epoch [14/50], Test Losses: mse: 1.9573, mae: 0.4246, huber: 0.2555, swd: 1.6345, ept: 93.5338
      Epoch 14 composite train-obj: 0.235876
            Val objective improved 0.2611 → 0.2554, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 1.8819, mae: 0.3928, huber: 0.2340, swd: 1.5665, ept: 93.7924
    Epoch [15/50], Val Losses: mse: 2.2333, mae: 0.4568, huber: 0.2743, swd: 1.8367, ept: 93.6134
    Epoch [15/50], Test Losses: mse: 2.0318, mae: 0.4651, huber: 0.2770, swd: 1.6547, ept: 93.4868
      Epoch 15 composite train-obj: 0.233997
            No improvement (0.2743), counter 1/5
    Epoch [16/50], Train Losses: mse: 1.8712, mae: 0.3921, huber: 0.2328, swd: 1.5560, ept: 93.8110
    Epoch [16/50], Val Losses: mse: 2.1867, mae: 0.4171, huber: 0.2595, swd: 1.8186, ept: 93.5567
    Epoch [16/50], Test Losses: mse: 1.9828, mae: 0.4219, huber: 0.2611, swd: 1.6349, ept: 93.4768
      Epoch 16 composite train-obj: 0.232779
            No improvement (0.2595), counter 2/5
    Epoch [17/50], Train Losses: mse: 1.8612, mae: 0.3897, huber: 0.2312, swd: 1.5475, ept: 93.8169
    Epoch [17/50], Val Losses: mse: 2.2058, mae: 0.4699, huber: 0.2835, swd: 1.8726, ept: 93.6276
    Epoch [17/50], Test Losses: mse: 1.9969, mae: 0.4741, huber: 0.2821, swd: 1.6791, ept: 93.5225
      Epoch 17 composite train-obj: 0.231249
            No improvement (0.2835), counter 3/5
    Epoch [18/50], Train Losses: mse: 1.8610, mae: 0.3913, huber: 0.2324, swd: 1.5481, ept: 93.7623
    Epoch [18/50], Val Losses: mse: 2.2922, mae: 0.5769, huber: 0.3286, swd: 1.8749, ept: 93.6562
    Epoch [18/50], Test Losses: mse: 2.1090, mae: 0.5928, huber: 0.3367, swd: 1.7068, ept: 93.5407
      Epoch 18 composite train-obj: 0.232361
            No improvement (0.3286), counter 4/5
    Epoch [19/50], Train Losses: mse: 1.8513, mae: 0.3889, huber: 0.2307, swd: 1.5368, ept: 93.8164
    Epoch [19/50], Val Losses: mse: 2.1696, mae: 0.4944, huber: 0.2916, swd: 1.8452, ept: 93.8226
    Epoch [19/50], Test Losses: mse: 1.9769, mae: 0.5011, huber: 0.2939, swd: 1.6667, ept: 93.6540
      Epoch 19 composite train-obj: 0.230702
    Epoch [19/50], Test Losses: mse: 1.9573, mae: 0.4246, huber: 0.2555, swd: 1.6345, ept: 93.5338
    Best round's Test MSE: 1.9573, MAE: 0.4246, SWD: 1.6345
    Best round's Validation MSE: 2.1627, MAE: 0.4244, SWD: 1.8215
    Best round's Test verification MSE : 1.9573, MAE: 0.4246, SWD: 1.6345
    Time taken: 38.39 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 3.6847, mae: 0.7890, huber: 0.5369, swd: 2.7025, ept: 89.1856
    Epoch [1/50], Val Losses: mse: 3.3132, mae: 0.8816, huber: 0.5551, swd: 2.3966, ept: 92.6605
    Epoch [1/50], Test Losses: mse: 3.0843, mae: 0.8983, huber: 0.5651, swd: 2.2061, ept: 92.4545
      Epoch 1 composite train-obj: 0.536947
            Val objective improved inf → 0.5551, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 2.5154, mae: 0.5787, huber: 0.3544, swd: 1.9092, ept: 92.8828
    Epoch [2/50], Val Losses: mse: 2.8789, mae: 0.5754, huber: 0.3718, swd: 2.1422, ept: 92.6992
    Epoch [2/50], Test Losses: mse: 2.6340, mae: 0.5886, huber: 0.3775, swd: 1.9455, ept: 92.6349
      Epoch 2 composite train-obj: 0.354443
            Val objective improved 0.5551 → 0.3718, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 2.3187, mae: 0.5140, huber: 0.3121, swd: 1.7605, ept: 93.1806
    Epoch [3/50], Val Losses: mse: 2.5758, mae: 0.5183, huber: 0.3230, swd: 1.9556, ept: 93.0269
    Epoch [3/50], Test Losses: mse: 2.3358, mae: 0.5147, huber: 0.3227, swd: 1.7640, ept: 92.9995
      Epoch 3 composite train-obj: 0.312080
            Val objective improved 0.3718 → 0.3230, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 2.2017, mae: 0.4861, huber: 0.2928, swd: 1.6674, ept: 93.3535
    Epoch [4/50], Val Losses: mse: 2.6343, mae: 0.6491, huber: 0.3977, swd: 1.9168, ept: 93.4160
    Epoch [4/50], Test Losses: mse: 2.4330, mae: 0.6561, huber: 0.4026, swd: 1.7499, ept: 93.2147
      Epoch 4 composite train-obj: 0.292789
            No improvement (0.3977), counter 1/5
    Epoch [5/50], Train Losses: mse: 2.1446, mae: 0.4730, huber: 0.2850, swd: 1.6174, ept: 93.4638
    Epoch [5/50], Val Losses: mse: 2.5200, mae: 0.5967, huber: 0.3607, swd: 1.9574, ept: 93.4057
    Epoch [5/50], Test Losses: mse: 2.3063, mae: 0.6087, huber: 0.3654, swd: 1.7775, ept: 93.2840
      Epoch 5 composite train-obj: 0.284968
            No improvement (0.3607), counter 2/5
    Epoch [6/50], Train Losses: mse: 2.0879, mae: 0.4560, huber: 0.2741, swd: 1.5771, ept: 93.4978
    Epoch [6/50], Val Losses: mse: 2.4176, mae: 0.4882, huber: 0.3048, swd: 1.8156, ept: 93.3564
    Epoch [6/50], Test Losses: mse: 2.1973, mae: 0.4951, huber: 0.3063, swd: 1.6380, ept: 93.2575
      Epoch 6 composite train-obj: 0.274074
            Val objective improved 0.3230 → 0.3048, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 2.0389, mae: 0.4387, huber: 0.2631, swd: 1.5403, ept: 93.5805
    Epoch [7/50], Val Losses: mse: 2.3766, mae: 0.5053, huber: 0.3033, swd: 1.7834, ept: 93.3482
    Epoch [7/50], Test Losses: mse: 2.1613, mae: 0.5138, huber: 0.3051, swd: 1.6092, ept: 93.2997
      Epoch 7 composite train-obj: 0.263110
            Val objective improved 0.3048 → 0.3033, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 2.0037, mae: 0.4320, huber: 0.2579, swd: 1.5129, ept: 93.6272
    Epoch [8/50], Val Losses: mse: 2.3309, mae: 0.4480, huber: 0.2849, swd: 1.7456, ept: 93.4196
    Epoch [8/50], Test Losses: mse: 2.1160, mae: 0.4519, huber: 0.2859, swd: 1.5728, ept: 93.3336
      Epoch 8 composite train-obj: 0.257876
            Val objective improved 0.3033 → 0.2849, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 1.9803, mae: 0.4240, huber: 0.2537, swd: 1.4952, ept: 93.6725
    Epoch [9/50], Val Losses: mse: 2.7378, mae: 0.7683, huber: 0.4769, swd: 1.8718, ept: 93.5667
    Epoch [9/50], Test Losses: mse: 2.5763, mae: 0.7924, huber: 0.4942, swd: 1.7197, ept: 93.3890
      Epoch 9 composite train-obj: 0.253692
            No improvement (0.4769), counter 1/5
    Epoch [10/50], Train Losses: mse: 1.9854, mae: 0.4338, huber: 0.2590, swd: 1.4869, ept: 93.6801
    Epoch [10/50], Val Losses: mse: 2.2845, mae: 0.4497, huber: 0.2780, swd: 1.7318, ept: 93.5059
    Epoch [10/50], Test Losses: mse: 2.0760, mae: 0.4574, huber: 0.2806, swd: 1.5621, ept: 93.4178
      Epoch 10 composite train-obj: 0.259003
            Val objective improved 0.2849 → 0.2780, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 1.9405, mae: 0.4101, huber: 0.2450, swd: 1.4656, ept: 93.7247
    Epoch [11/50], Val Losses: mse: 2.2866, mae: 0.4564, huber: 0.2820, swd: 1.6806, ept: 93.4947
    Epoch [11/50], Test Losses: mse: 2.0857, mae: 0.4656, huber: 0.2858, swd: 1.5170, ept: 93.4104
      Epoch 11 composite train-obj: 0.245000
            No improvement (0.2820), counter 1/5
    Epoch [12/50], Train Losses: mse: 1.9193, mae: 0.4049, huber: 0.2415, swd: 1.4471, ept: 93.7402
    Epoch [12/50], Val Losses: mse: 2.2141, mae: 0.4356, huber: 0.2650, swd: 1.6741, ept: 93.5654
    Epoch [12/50], Test Losses: mse: 2.0052, mae: 0.4354, huber: 0.2649, swd: 1.5048, ept: 93.4568
      Epoch 12 composite train-obj: 0.241473
            Val objective improved 0.2780 → 0.2650, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 1.9093, mae: 0.4027, huber: 0.2396, swd: 1.4395, ept: 93.7615
    Epoch [13/50], Val Losses: mse: 2.1723, mae: 0.4337, huber: 0.2588, swd: 1.6388, ept: 93.6417
    Epoch [13/50], Test Losses: mse: 1.9660, mae: 0.4320, huber: 0.2582, swd: 1.4702, ept: 93.5150
      Epoch 13 composite train-obj: 0.239560
            Val objective improved 0.2650 → 0.2588, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 1.8950, mae: 0.3980, huber: 0.2371, swd: 1.4272, ept: 93.7754
    Epoch [14/50], Val Losses: mse: 2.2873, mae: 0.5084, huber: 0.3055, swd: 1.7672, ept: 93.6173
    Epoch [14/50], Test Losses: mse: 2.0897, mae: 0.5207, huber: 0.3101, swd: 1.6000, ept: 93.4820
      Epoch 14 composite train-obj: 0.237128
            No improvement (0.3055), counter 1/5
    Epoch [15/50], Train Losses: mse: 1.8941, mae: 0.4011, huber: 0.2380, swd: 1.4269, ept: 93.7365
    Epoch [15/50], Val Losses: mse: 2.1587, mae: 0.4269, huber: 0.2563, swd: 1.6283, ept: 93.6228
    Epoch [15/50], Test Losses: mse: 1.9559, mae: 0.4305, huber: 0.2572, swd: 1.4632, ept: 93.5476
      Epoch 15 composite train-obj: 0.238034
            Val objective improved 0.2588 → 0.2563, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 1.8766, mae: 0.3941, huber: 0.2341, swd: 1.4113, ept: 93.7987
    Epoch [16/50], Val Losses: mse: 2.1616, mae: 0.4471, huber: 0.2673, swd: 1.6457, ept: 93.7032
    Epoch [16/50], Test Losses: mse: 1.9622, mae: 0.4528, huber: 0.2691, swd: 1.4818, ept: 93.5775
      Epoch 16 composite train-obj: 0.234126
            No improvement (0.2673), counter 1/5
    Epoch [17/50], Train Losses: mse: 1.8666, mae: 0.3910, huber: 0.2324, swd: 1.4052, ept: 93.8123
    Epoch [17/50], Val Losses: mse: 2.1675, mae: 0.4054, huber: 0.2553, swd: 1.6192, ept: 93.6105
    Epoch [17/50], Test Losses: mse: 1.9652, mae: 0.4094, huber: 0.2567, swd: 1.4560, ept: 93.5151
      Epoch 17 composite train-obj: 0.232448
            Val objective improved 0.2563 → 0.2553, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 1.8528, mae: 0.3860, huber: 0.2296, swd: 1.3950, ept: 93.8229
    Epoch [18/50], Val Losses: mse: 2.1773, mae: 0.4154, huber: 0.2597, swd: 1.6101, ept: 93.6454
    Epoch [18/50], Test Losses: mse: 1.9789, mae: 0.4238, huber: 0.2621, swd: 1.4488, ept: 93.5366
      Epoch 18 composite train-obj: 0.229591
            No improvement (0.2597), counter 1/5
    Epoch [19/50], Train Losses: mse: 1.9009, mae: 0.3919, huber: 0.2356, swd: 1.4302, ept: 93.7653
    Epoch [19/50], Val Losses: mse: 2.2492, mae: 0.5758, huber: 0.3332, swd: 1.6730, ept: 93.8804
    Epoch [19/50], Test Losses: mse: 2.0643, mae: 0.5814, huber: 0.3370, swd: 1.5127, ept: 93.7119
      Epoch 19 composite train-obj: 0.235596
            No improvement (0.3332), counter 2/5
    Epoch [20/50], Train Losses: mse: 1.8461, mae: 0.3911, huber: 0.2312, swd: 1.3842, ept: 93.8481
    Epoch [20/50], Val Losses: mse: 2.1210, mae: 0.4278, huber: 0.2536, swd: 1.5953, ept: 93.6782
    Epoch [20/50], Test Losses: mse: 1.9233, mae: 0.4321, huber: 0.2549, swd: 1.4352, ept: 93.5680
      Epoch 20 composite train-obj: 0.231171
            Val objective improved 0.2553 → 0.2536, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 1.8318, mae: 0.3799, huber: 0.2257, swd: 1.3776, ept: 93.8606
    Epoch [21/50], Val Losses: mse: 2.1170, mae: 0.4188, huber: 0.2512, swd: 1.6151, ept: 93.7418
    Epoch [21/50], Test Losses: mse: 1.9181, mae: 0.4186, huber: 0.2511, swd: 1.4514, ept: 93.5580
      Epoch 21 composite train-obj: 0.225704
            Val objective improved 0.2536 → 0.2512, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 1.8235, mae: 0.3781, huber: 0.2245, swd: 1.3707, ept: 93.8632
    Epoch [22/50], Val Losses: mse: 2.1119, mae: 0.4098, huber: 0.2482, swd: 1.5800, ept: 93.7050
    Epoch [22/50], Test Losses: mse: 1.9137, mae: 0.4105, huber: 0.2486, swd: 1.4188, ept: 93.5589
      Epoch 22 composite train-obj: 0.224541
            Val objective improved 0.2512 → 0.2482, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 1.8239, mae: 0.3795, huber: 0.2248, swd: 1.3712, ept: 93.8726
    Epoch [23/50], Val Losses: mse: 2.1082, mae: 0.4011, huber: 0.2471, swd: 1.5786, ept: 93.6804
    Epoch [23/50], Test Losses: mse: 1.9090, mae: 0.4024, huber: 0.2478, swd: 1.4170, ept: 93.5865
      Epoch 23 composite train-obj: 0.224811
            Val objective improved 0.2482 → 0.2471, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 1.8141, mae: 0.3764, huber: 0.2231, swd: 1.3630, ept: 93.8891
    Epoch [24/50], Val Losses: mse: 2.2096, mae: 0.5091, huber: 0.3024, swd: 1.6991, ept: 93.8564
    Epoch [24/50], Test Losses: mse: 2.0205, mae: 0.5201, huber: 0.3067, swd: 1.5387, ept: 93.6672
      Epoch 24 composite train-obj: 0.223144
            No improvement (0.3024), counter 1/5
    Epoch [25/50], Train Losses: mse: 1.8180, mae: 0.3802, huber: 0.2253, swd: 1.3651, ept: 93.8766
    Epoch [25/50], Val Losses: mse: 2.1410, mae: 0.4572, huber: 0.2671, swd: 1.6046, ept: 93.7610
    Epoch [25/50], Test Losses: mse: 1.9460, mae: 0.4636, huber: 0.2684, swd: 1.4433, ept: 93.6441
      Epoch 25 composite train-obj: 0.225265
            No improvement (0.2671), counter 2/5
    Epoch [26/50], Train Losses: mse: 1.8057, mae: 0.3739, huber: 0.2219, swd: 1.3548, ept: 93.8897
    Epoch [26/50], Val Losses: mse: 2.0754, mae: 0.4173, huber: 0.2456, swd: 1.5749, ept: 93.8121
    Epoch [26/50], Test Losses: mse: 1.8790, mae: 0.4204, huber: 0.2453, swd: 1.4135, ept: 93.6481
      Epoch 26 composite train-obj: 0.221895
            Val objective improved 0.2471 → 0.2456, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 1.7990, mae: 0.3721, huber: 0.2207, swd: 1.3505, ept: 93.9042
    Epoch [27/50], Val Losses: mse: 2.1188, mae: 0.4339, huber: 0.2582, swd: 1.6058, ept: 93.7882
    Epoch [27/50], Test Losses: mse: 1.9255, mae: 0.4390, huber: 0.2597, swd: 1.4462, ept: 93.6068
      Epoch 27 composite train-obj: 0.220657
            No improvement (0.2582), counter 1/5
    Epoch [28/50], Train Losses: mse: 1.7964, mae: 0.3723, huber: 0.2204, swd: 1.3476, ept: 93.9020
    Epoch [28/50], Val Losses: mse: 2.1199, mae: 0.4292, huber: 0.2570, swd: 1.5833, ept: 93.7582
    Epoch [28/50], Test Losses: mse: 1.9266, mae: 0.4371, huber: 0.2592, swd: 1.4274, ept: 93.6131
      Epoch 28 composite train-obj: 0.220429
            No improvement (0.2570), counter 2/5
    Epoch [29/50], Train Losses: mse: 1.7912, mae: 0.3707, huber: 0.2195, swd: 1.3437, ept: 93.9189
    Epoch [29/50], Val Losses: mse: 2.1056, mae: 0.4075, huber: 0.2511, swd: 1.5607, ept: 93.7226
    Epoch [29/50], Test Losses: mse: 1.9173, mae: 0.4147, huber: 0.2541, swd: 1.4076, ept: 93.5735
      Epoch 29 composite train-obj: 0.219514
            No improvement (0.2511), counter 3/5
    Epoch [30/50], Train Losses: mse: 1.7886, mae: 0.3694, huber: 0.2190, swd: 1.3413, ept: 93.9148
    Epoch [30/50], Val Losses: mse: 2.0502, mae: 0.3980, huber: 0.2386, swd: 1.5360, ept: 93.7791
    Epoch [30/50], Test Losses: mse: 1.8561, mae: 0.3988, huber: 0.2390, swd: 1.3788, ept: 93.6566
      Epoch 30 composite train-obj: 0.218993
            Val objective improved 0.2456 → 0.2386, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 1.8369, mae: 0.3753, huber: 0.2232, swd: 1.3856, ept: 93.8739
    Epoch [31/50], Val Losses: mse: 2.1862, mae: 0.5310, huber: 0.3053, swd: 1.6131, ept: 93.9214
    Epoch [31/50], Test Losses: mse: 2.0029, mae: 0.5411, huber: 0.3101, swd: 1.4515, ept: 93.7024
      Epoch 31 composite train-obj: 0.223234
            No improvement (0.3053), counter 1/5
    Epoch [32/50], Train Losses: mse: 1.8377, mae: 0.3825, huber: 0.2293, swd: 1.3613, ept: 93.8000
    Epoch [32/50], Val Losses: mse: 2.8624, mae: 0.8398, huber: 0.5442, swd: 1.8388, ept: 93.7006
    Epoch [32/50], Test Losses: mse: 2.7409, mae: 0.8727, huber: 0.5687, swd: 1.7398, ept: 93.5615
      Epoch 32 composite train-obj: 0.229288
            No improvement (0.5442), counter 2/5
    Epoch [33/50], Train Losses: mse: 1.8348, mae: 0.3944, huber: 0.2351, swd: 1.3584, ept: 93.8831
    Epoch [33/50], Val Losses: mse: 2.0685, mae: 0.4008, huber: 0.2443, swd: 1.5405, ept: 93.8065
    Epoch [33/50], Test Losses: mse: 1.8764, mae: 0.4050, huber: 0.2456, swd: 1.3846, ept: 93.6750
      Epoch 33 composite train-obj: 0.235121
            No improvement (0.2443), counter 3/5
    Epoch [34/50], Train Losses: mse: 1.7705, mae: 0.3614, huber: 0.2157, swd: 1.3290, ept: 93.9439
    Epoch [34/50], Val Losses: mse: 2.0544, mae: 0.3785, huber: 0.2380, swd: 1.5386, ept: 93.7687
    Epoch [34/50], Test Losses: mse: 1.8602, mae: 0.3832, huber: 0.2388, swd: 1.3815, ept: 93.6642
      Epoch 34 composite train-obj: 0.215716
            Val objective improved 0.2386 → 0.2380, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 1.7683, mae: 0.3620, huber: 0.2153, swd: 1.3273, ept: 93.9448
    Epoch [35/50], Val Losses: mse: 2.1092, mae: 0.4185, huber: 0.2542, swd: 1.5669, ept: 93.7443
    Epoch [35/50], Test Losses: mse: 1.9169, mae: 0.4286, huber: 0.2571, swd: 1.4111, ept: 93.6402
      Epoch 35 composite train-obj: 0.215309
            No improvement (0.2542), counter 1/5
    Epoch [36/50], Train Losses: mse: 1.7652, mae: 0.3629, huber: 0.2154, swd: 1.3231, ept: 93.9478
    Epoch [36/50], Val Losses: mse: 2.1084, mae: 0.4349, huber: 0.2579, swd: 1.5712, ept: 93.7623
    Epoch [36/50], Test Losses: mse: 1.9194, mae: 0.4451, huber: 0.2608, swd: 1.4183, ept: 93.6467
      Epoch 36 composite train-obj: 0.215406
            No improvement (0.2579), counter 2/5
    Epoch [37/50], Train Losses: mse: 1.7657, mae: 0.3644, huber: 0.2155, swd: 1.3234, ept: 93.9480
    Epoch [37/50], Val Losses: mse: 2.1065, mae: 0.4164, huber: 0.2581, swd: 1.5772, ept: 93.7364
    Epoch [37/50], Test Losses: mse: 1.9111, mae: 0.4235, huber: 0.2586, swd: 1.4170, ept: 93.6186
      Epoch 37 composite train-obj: 0.215463
            No improvement (0.2581), counter 3/5
    Epoch [38/50], Train Losses: mse: 1.7736, mae: 0.3671, huber: 0.2179, swd: 1.3312, ept: 93.8778
    Epoch [38/50], Val Losses: mse: 2.1328, mae: 0.4910, huber: 0.2847, swd: 1.5588, ept: 93.8032
    Epoch [38/50], Test Losses: mse: 1.9558, mae: 0.5043, huber: 0.2915, swd: 1.4110, ept: 93.6898
      Epoch 38 composite train-obj: 0.217946
            No improvement (0.2847), counter 4/5
    Epoch [39/50], Train Losses: mse: 1.7622, mae: 0.3640, huber: 0.2155, swd: 1.3196, ept: 93.9534
    Epoch [39/50], Val Losses: mse: 2.0598, mae: 0.3925, huber: 0.2414, swd: 1.5353, ept: 93.7491
    Epoch [39/50], Test Losses: mse: 1.8701, mae: 0.3988, huber: 0.2435, swd: 1.3817, ept: 93.6453
      Epoch 39 composite train-obj: 0.215489
    Epoch [39/50], Test Losses: mse: 1.8602, mae: 0.3832, huber: 0.2388, swd: 1.3815, ept: 93.6642
    Best round's Test MSE: 1.8602, MAE: 0.3832, SWD: 1.3815
    Best round's Validation MSE: 2.0544, MAE: 0.3785, SWD: 1.5386
    Best round's Test verification MSE : 1.8602, MAE: 0.3832, SWD: 1.3815
    Time taken: 81.99 seconds
    
    ==================================================
    Experiment Summary (DLinear_rossler_seq336_pred96_20250513_1208)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 1.8980 ± 0.0425
      mae: 0.4027 ± 0.0170
      huber: 0.2451 ± 0.0074
      swd: 1.5048 ± 0.1034
      ept: 93.6115 ± 0.0561
      count: 38.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 2.0969 ± 0.0471
      mae: 0.4013 ± 0.0188
      huber: 0.2447 ± 0.0076
      swd: 1.6770 ± 0.1156
      ept: 93.7179 ± 0.0512
      count: 38.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 184.32 seconds
    
    Experiment complete: DLinear_rossler_seq336_pred96_20250513_1208
    Model: DLinear
    Dataset: rossler
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([196, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 281
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 196, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 5.5777, mae: 1.0754, huber: 0.7866, swd: 4.3652, ept: 173.6182
    Epoch [1/50], Val Losses: mse: 5.2792, mae: 1.0081, huber: 0.7167, swd: 4.3181, ept: 178.6688
    Epoch [1/50], Test Losses: mse: 4.6554, mae: 0.9837, huber: 0.6923, swd: 3.7998, ept: 179.7924
      Epoch 1 composite train-obj: 0.786589
            Val objective improved inf → 0.7167, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.4495, mae: 0.8881, huber: 0.6156, swd: 3.6344, ept: 181.7832
    Epoch [2/50], Val Losses: mse: 5.0628, mae: 0.9381, huber: 0.6558, swd: 4.1465, ept: 180.4219
    Epoch [2/50], Test Losses: mse: 4.4668, mae: 0.9151, huber: 0.6355, swd: 3.6466, ept: 181.0422
      Epoch 2 composite train-obj: 0.615602
            Val objective improved 0.7167 → 0.6558, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.2523, mae: 0.8460, huber: 0.5831, swd: 3.4843, ept: 182.8165
    Epoch [3/50], Val Losses: mse: 4.7760, mae: 0.9105, huber: 0.6353, swd: 3.9462, ept: 181.5147
    Epoch [3/50], Test Losses: mse: 4.2047, mae: 0.8858, huber: 0.6124, swd: 3.4639, ept: 182.0724
      Epoch 3 composite train-obj: 0.583145
            Val objective improved 0.6558 → 0.6353, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.1652, mae: 0.8227, huber: 0.5660, swd: 3.4237, ept: 183.3181
    Epoch [4/50], Val Losses: mse: 4.9120, mae: 0.8782, huber: 0.6184, swd: 3.9815, ept: 181.5893
    Epoch [4/50], Test Losses: mse: 4.3520, mae: 0.8612, huber: 0.6047, swd: 3.5076, ept: 181.9852
      Epoch 4 composite train-obj: 0.566012
            Val objective improved 0.6353 → 0.6184, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.1068, mae: 0.8072, huber: 0.5543, swd: 3.3812, ept: 183.6152
    Epoch [5/50], Val Losses: mse: 4.7130, mae: 0.8696, huber: 0.6089, swd: 3.8188, ept: 182.2148
    Epoch [5/50], Test Losses: mse: 4.1628, mae: 0.8513, huber: 0.5909, swd: 3.3594, ept: 182.6118
      Epoch 5 composite train-obj: 0.554251
            Val objective improved 0.6184 → 0.6089, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.0533, mae: 0.7959, huber: 0.5463, swd: 3.3319, ept: 183.8692
    Epoch [6/50], Val Losses: mse: 4.5707, mae: 0.8594, huber: 0.5994, swd: 3.7900, ept: 182.6643
    Epoch [6/50], Test Losses: mse: 4.0273, mae: 0.8350, huber: 0.5771, swd: 3.3301, ept: 183.0608
      Epoch 6 composite train-obj: 0.546267
            Val objective improved 0.6089 → 0.5994, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.0271, mae: 0.7842, huber: 0.5381, swd: 3.3220, ept: 184.0301
    Epoch [7/50], Val Losses: mse: 4.6226, mae: 0.8377, huber: 0.5850, swd: 3.8318, ept: 182.5788
    Epoch [7/50], Test Losses: mse: 4.0747, mae: 0.8137, huber: 0.5658, swd: 3.3761, ept: 182.9631
      Epoch 7 composite train-obj: 0.538059
            Val objective improved 0.5994 → 0.5850, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 3.9833, mae: 0.7760, huber: 0.5313, swd: 3.2848, ept: 184.2400
    Epoch [8/50], Val Losses: mse: 4.9811, mae: 0.8215, huber: 0.5922, swd: 4.0445, ept: 182.1418
    Epoch [8/50], Test Losses: mse: 4.4440, mae: 0.8169, huber: 0.5888, swd: 3.5916, ept: 182.2167
      Epoch 8 composite train-obj: 0.531283
            No improvement (0.5922), counter 1/5
    Epoch [9/50], Train Losses: mse: 3.9773, mae: 0.7692, huber: 0.5269, swd: 3.2842, ept: 184.3304
    Epoch [9/50], Val Losses: mse: 4.5293, mae: 0.8322, huber: 0.5795, swd: 3.7963, ept: 182.8068
    Epoch [9/50], Test Losses: mse: 3.9834, mae: 0.8051, huber: 0.5579, swd: 3.3426, ept: 183.2653
      Epoch 9 composite train-obj: 0.526869
            Val objective improved 0.5850 → 0.5795, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 3.9492, mae: 0.7616, huber: 0.5213, swd: 3.2637, ept: 184.4316
    Epoch [10/50], Val Losses: mse: 4.5885, mae: 0.8152, huber: 0.5685, swd: 3.8283, ept: 182.7946
    Epoch [10/50], Test Losses: mse: 4.0494, mae: 0.7971, huber: 0.5533, swd: 3.3745, ept: 183.1428
      Epoch 10 composite train-obj: 0.521346
            Val objective improved 0.5795 → 0.5685, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 3.9452, mae: 0.7573, huber: 0.5181, swd: 3.2657, ept: 184.4641
    Epoch [11/50], Val Losses: mse: 4.4648, mae: 0.8371, huber: 0.5751, swd: 3.7477, ept: 183.3587
    Epoch [11/50], Test Losses: mse: 3.9313, mae: 0.8177, huber: 0.5553, swd: 3.2954, ept: 183.7823
      Epoch 11 composite train-obj: 0.518114
            No improvement (0.5751), counter 1/5
    Epoch [12/50], Train Losses: mse: 3.9186, mae: 0.7516, huber: 0.5143, swd: 3.2443, ept: 184.5722
    Epoch [12/50], Val Losses: mse: 4.6187, mae: 0.8153, huber: 0.5645, swd: 3.8394, ept: 183.1133
    Epoch [12/50], Test Losses: mse: 4.0786, mae: 0.7948, huber: 0.5491, swd: 3.3854, ept: 183.3885
      Epoch 12 composite train-obj: 0.514296
            Val objective improved 0.5685 → 0.5645, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 3.9155, mae: 0.7474, huber: 0.5107, swd: 3.2446, ept: 184.6546
    Epoch [13/50], Val Losses: mse: 4.5388, mae: 0.8022, huber: 0.5666, swd: 3.8161, ept: 183.1861
    Epoch [13/50], Test Losses: mse: 4.0055, mae: 0.7781, huber: 0.5491, swd: 3.3652, ept: 183.3170
      Epoch 13 composite train-obj: 0.510748
            No improvement (0.5666), counter 1/5
    Epoch [14/50], Train Losses: mse: 3.9088, mae: 0.7451, huber: 0.5090, swd: 3.2406, ept: 184.6919
    Epoch [14/50], Val Losses: mse: 4.6347, mae: 0.7899, huber: 0.5576, swd: 3.8816, ept: 182.9350
    Epoch [14/50], Test Losses: mse: 4.0958, mae: 0.7756, huber: 0.5454, swd: 3.4331, ept: 183.3182
      Epoch 14 composite train-obj: 0.509034
            Val objective improved 0.5645 → 0.5576, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 3.9046, mae: 0.7419, huber: 0.5070, swd: 3.2385, ept: 184.7206
    Epoch [15/50], Val Losses: mse: 4.6652, mae: 0.7910, huber: 0.5554, swd: 3.8621, ept: 183.0886
    Epoch [15/50], Test Losses: mse: 4.1294, mae: 0.7739, huber: 0.5440, swd: 3.4131, ept: 183.2421
      Epoch 15 composite train-obj: 0.507023
            Val objective improved 0.5576 → 0.5554, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 3.8997, mae: 0.7399, huber: 0.5050, swd: 3.2365, ept: 184.7808
    Epoch [16/50], Val Losses: mse: 4.6837, mae: 0.7767, huber: 0.5526, swd: 3.8869, ept: 183.0377
    Epoch [16/50], Test Losses: mse: 4.1517, mae: 0.7644, huber: 0.5444, swd: 3.4398, ept: 183.1472
      Epoch 16 composite train-obj: 0.504967
            Val objective improved 0.5554 → 0.5526, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 3.8910, mae: 0.7341, huber: 0.5019, swd: 3.2336, ept: 184.8104
    Epoch [17/50], Val Losses: mse: 4.4501, mae: 0.7964, huber: 0.5495, swd: 3.7180, ept: 183.5229
    Epoch [17/50], Test Losses: mse: 3.9195, mae: 0.7750, huber: 0.5322, swd: 3.2694, ept: 183.7683
      Epoch 17 composite train-obj: 0.501858
            Val objective improved 0.5526 → 0.5495, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 3.8939, mae: 0.7327, huber: 0.5006, swd: 3.2375, ept: 184.8467
    Epoch [18/50], Val Losses: mse: 4.5897, mae: 0.7861, huber: 0.5479, swd: 3.7752, ept: 183.2112
    Epoch [18/50], Test Losses: mse: 4.0603, mae: 0.7732, huber: 0.5377, swd: 3.3292, ept: 183.4057
      Epoch 18 composite train-obj: 0.500637
            Val objective improved 0.5495 → 0.5479, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 3.8783, mae: 0.7311, huber: 0.4986, swd: 3.2251, ept: 184.8961
    Epoch [19/50], Val Losses: mse: 4.5481, mae: 0.7705, huber: 0.5465, swd: 3.7956, ept: 183.2539
    Epoch [19/50], Test Losses: mse: 4.0167, mae: 0.7551, huber: 0.5338, swd: 3.3489, ept: 183.5060
      Epoch 19 composite train-obj: 0.498572
            Val objective improved 0.5479 → 0.5465, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 3.8863, mae: 0.7286, huber: 0.4975, swd: 3.2337, ept: 184.9258
    Epoch [20/50], Val Losses: mse: 4.4432, mae: 0.7881, huber: 0.5449, swd: 3.7140, ept: 183.6637
    Epoch [20/50], Test Losses: mse: 3.9115, mae: 0.7678, huber: 0.5274, swd: 3.2666, ept: 183.8636
      Epoch 20 composite train-obj: 0.497494
            Val objective improved 0.5465 → 0.5449, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 3.8701, mae: 0.7255, huber: 0.4954, swd: 3.2204, ept: 184.9486
    Epoch [21/50], Val Losses: mse: 4.5429, mae: 0.7784, huber: 0.5454, swd: 3.7410, ept: 183.4898
    Epoch [21/50], Test Losses: mse: 4.0134, mae: 0.7643, huber: 0.5338, swd: 3.2914, ept: 183.5985
      Epoch 21 composite train-obj: 0.495411
            No improvement (0.5454), counter 1/5
    Epoch [22/50], Train Losses: mse: 3.8791, mae: 0.7248, huber: 0.4948, swd: 3.2281, ept: 184.9506
    Epoch [22/50], Val Losses: mse: 4.6243, mae: 0.7729, huber: 0.5461, swd: 3.8479, ept: 183.4001
    Epoch [22/50], Test Losses: mse: 4.0907, mae: 0.7552, huber: 0.5335, swd: 3.4000, ept: 183.4275
      Epoch 22 composite train-obj: 0.494817
            No improvement (0.5461), counter 2/5
    Epoch [23/50], Train Losses: mse: 3.8707, mae: 0.7231, huber: 0.4936, swd: 3.2223, ept: 184.9737
    Epoch [23/50], Val Losses: mse: 4.6476, mae: 0.7674, huber: 0.5429, swd: 3.9039, ept: 183.2561
    Epoch [23/50], Test Losses: mse: 4.1143, mae: 0.7512, huber: 0.5323, swd: 3.4572, ept: 183.4963
      Epoch 23 composite train-obj: 0.493570
            Val objective improved 0.5449 → 0.5429, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 3.8799, mae: 0.7215, huber: 0.4926, swd: 3.2347, ept: 184.9872
    Epoch [24/50], Val Losses: mse: 4.5559, mae: 0.7776, huber: 0.5410, swd: 3.7975, ept: 183.6813
    Epoch [24/50], Test Losses: mse: 4.0259, mae: 0.7624, huber: 0.5286, swd: 3.3462, ept: 183.7579
      Epoch 24 composite train-obj: 0.492562
            Val objective improved 0.5429 → 0.5410, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 3.8607, mae: 0.7195, huber: 0.4901, swd: 3.2168, ept: 185.0428
    Epoch [25/50], Val Losses: mse: 4.5225, mae: 0.7695, huber: 0.5341, swd: 3.7527, ept: 183.6308
    Epoch [25/50], Test Losses: mse: 3.9929, mae: 0.7539, huber: 0.5224, swd: 3.3030, ept: 183.7785
      Epoch 25 composite train-obj: 0.490120
            Val objective improved 0.5410 → 0.5341, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 3.8629, mae: 0.7180, huber: 0.4898, swd: 3.2204, ept: 185.0439
    Epoch [26/50], Val Losses: mse: 4.5214, mae: 0.7659, huber: 0.5355, swd: 3.7656, ept: 183.6631
    Epoch [26/50], Test Losses: mse: 3.9944, mae: 0.7541, huber: 0.5247, swd: 3.3161, ept: 183.7806
      Epoch 26 composite train-obj: 0.489770
            No improvement (0.5355), counter 1/5
    Epoch [27/50], Train Losses: mse: 3.8737, mae: 0.7167, huber: 0.4893, swd: 3.2298, ept: 185.0168
    Epoch [27/50], Val Losses: mse: 4.4041, mae: 0.8086, huber: 0.5462, swd: 3.6685, ept: 184.0298
    Epoch [27/50], Test Losses: mse: 3.8780, mae: 0.7884, huber: 0.5287, swd: 3.2152, ept: 184.1803
      Epoch 27 composite train-obj: 0.489253
            No improvement (0.5462), counter 2/5
    Epoch [28/50], Train Losses: mse: 3.8663, mae: 0.7156, huber: 0.4881, swd: 3.2249, ept: 185.0818
    Epoch [28/50], Val Losses: mse: 4.4922, mae: 0.7798, huber: 0.5352, swd: 3.7686, ept: 183.6704
    Epoch [28/50], Test Losses: mse: 3.9585, mae: 0.7597, huber: 0.5205, swd: 3.3176, ept: 183.9616
      Epoch 28 composite train-obj: 0.488134
            No improvement (0.5352), counter 3/5
    Epoch [29/50], Train Losses: mse: 3.8704, mae: 0.7163, huber: 0.4882, swd: 3.2317, ept: 185.0829
    Epoch [29/50], Val Losses: mse: 4.5133, mae: 0.7752, huber: 0.5359, swd: 3.7466, ept: 183.7102
    Epoch [29/50], Test Losses: mse: 3.9875, mae: 0.7596, huber: 0.5241, swd: 3.3003, ept: 183.7317
      Epoch 29 composite train-obj: 0.488244
            No improvement (0.5359), counter 4/5
    Epoch [30/50], Train Losses: mse: 3.8492, mae: 0.7121, huber: 0.4851, swd: 3.2139, ept: 185.1337
    Epoch [30/50], Val Losses: mse: 4.5420, mae: 0.7597, huber: 0.5349, swd: 3.8195, ept: 183.5389
    Epoch [30/50], Test Losses: mse: 4.0075, mae: 0.7438, huber: 0.5223, swd: 3.3720, ept: 183.7016
      Epoch 30 composite train-obj: 0.485144
    Epoch [30/50], Test Losses: mse: 3.9929, mae: 0.7539, huber: 0.5224, swd: 3.3030, ept: 183.7785
    Best round's Test MSE: 3.9929, MAE: 0.7539, SWD: 3.3030
    Best round's Validation MSE: 4.5225, MAE: 0.7695, SWD: 3.7527
    Best round's Test verification MSE : 3.9929, MAE: 0.7539, SWD: 3.3030
    Time taken: 64.14 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 5.5592, mae: 1.0723, huber: 0.7843, swd: 4.4099, ept: 173.2077
    Epoch [1/50], Val Losses: mse: 5.4772, mae: 0.9768, huber: 0.7038, swd: 4.6158, ept: 178.7375
    Epoch [1/50], Test Losses: mse: 4.8526, mae: 0.9567, huber: 0.6866, swd: 4.0903, ept: 179.4894
      Epoch 1 composite train-obj: 0.784297
            Val objective improved inf → 0.7038, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.4666, mae: 0.8868, huber: 0.6149, swd: 3.7118, ept: 181.8721
    Epoch [2/50], Val Losses: mse: 5.2560, mae: 0.9208, huber: 0.6583, swd: 4.3612, ept: 179.9782
    Epoch [2/50], Test Losses: mse: 4.6625, mae: 0.9041, huber: 0.6461, swd: 3.8597, ept: 180.6619
      Epoch 2 composite train-obj: 0.614922
            Val objective improved 0.7038 → 0.6583, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.2815, mae: 0.8483, huber: 0.5844, swd: 3.5657, ept: 182.8016
    Epoch [3/50], Val Losses: mse: 4.9518, mae: 0.8923, huber: 0.6318, swd: 4.1564, ept: 181.1948
    Epoch [3/50], Test Losses: mse: 4.3719, mae: 0.8660, huber: 0.6116, swd: 3.6728, ept: 181.9329
      Epoch 3 composite train-obj: 0.584423
            Val objective improved 0.6583 → 0.6318, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.1648, mae: 0.8229, huber: 0.5664, swd: 3.4610, ept: 183.3526
    Epoch [4/50], Val Losses: mse: 5.0128, mae: 0.8683, huber: 0.6178, swd: 4.1418, ept: 181.0618
    Epoch [4/50], Test Losses: mse: 4.4436, mae: 0.8523, huber: 0.6070, swd: 3.6637, ept: 181.6027
      Epoch 4 composite train-obj: 0.566418
            Val objective improved 0.6318 → 0.6178, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.1218, mae: 0.8067, huber: 0.5545, swd: 3.4369, ept: 183.5839
    Epoch [5/50], Val Losses: mse: 4.7598, mae: 0.8573, huber: 0.6024, swd: 3.9641, ept: 181.8597
    Epoch [5/50], Test Losses: mse: 4.2044, mae: 0.8368, huber: 0.5860, swd: 3.4967, ept: 182.4168
      Epoch 5 composite train-obj: 0.554479
            Val objective improved 0.6178 → 0.6024, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.0465, mae: 0.7952, huber: 0.5455, swd: 3.3725, ept: 183.8837
    Epoch [6/50], Val Losses: mse: 4.9153, mae: 0.8398, huber: 0.6024, swd: 4.0447, ept: 181.9485
    Epoch [6/50], Test Losses: mse: 4.3753, mae: 0.8312, huber: 0.5951, swd: 3.5889, ept: 182.1282
      Epoch 6 composite train-obj: 0.545453
            Val objective improved 0.6024 → 0.6024, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.0284, mae: 0.7865, huber: 0.5395, swd: 3.3609, ept: 184.0512
    Epoch [7/50], Val Losses: mse: 4.6407, mae: 0.8342, huber: 0.5855, swd: 3.8891, ept: 182.5243
    Epoch [7/50], Test Losses: mse: 4.0934, mae: 0.8141, huber: 0.5677, swd: 3.4239, ept: 182.8287
      Epoch 7 composite train-obj: 0.539454
            Val objective improved 0.6024 → 0.5855, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 4.0127, mae: 0.7769, huber: 0.5332, swd: 3.3532, ept: 184.1591
    Epoch [8/50], Val Losses: mse: 4.6448, mae: 0.8565, huber: 0.5957, swd: 3.8759, ept: 182.7886
    Epoch [8/50], Test Losses: mse: 4.1011, mae: 0.8389, huber: 0.5781, swd: 3.4168, ept: 183.3552
      Epoch 8 composite train-obj: 0.533174
            No improvement (0.5957), counter 1/5
    Epoch [9/50], Train Losses: mse: 3.9775, mae: 0.7687, huber: 0.5266, swd: 3.3263, ept: 184.3566
    Epoch [9/50], Val Losses: mse: 4.5760, mae: 0.8294, huber: 0.5762, swd: 3.8494, ept: 182.9150
    Epoch [9/50], Test Losses: mse: 4.0308, mae: 0.8067, huber: 0.5581, swd: 3.3864, ept: 183.3755
      Epoch 9 composite train-obj: 0.526606
            Val objective improved 0.5855 → 0.5762, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 3.9443, mae: 0.7616, huber: 0.5206, swd: 3.3015, ept: 184.4396
    Epoch [10/50], Val Losses: mse: 4.6255, mae: 0.8182, huber: 0.5687, swd: 3.8617, ept: 182.7836
    Epoch [10/50], Test Losses: mse: 4.0840, mae: 0.7995, huber: 0.5542, swd: 3.4011, ept: 183.1705
      Epoch 10 composite train-obj: 0.520612
            Val objective improved 0.5762 → 0.5687, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 3.9474, mae: 0.7575, huber: 0.5184, swd: 3.3020, ept: 184.4773
    Epoch [11/50], Val Losses: mse: 4.5890, mae: 0.8091, huber: 0.5634, swd: 3.8629, ept: 183.0239
    Epoch [11/50], Test Losses: mse: 4.0526, mae: 0.7911, huber: 0.5488, swd: 3.4030, ept: 183.2967
      Epoch 11 composite train-obj: 0.518368
            Val objective improved 0.5687 → 0.5634, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 3.9370, mae: 0.7532, huber: 0.5151, swd: 3.2988, ept: 184.5571
    Epoch [12/50], Val Losses: mse: 4.4526, mae: 0.8201, huber: 0.5695, swd: 3.7772, ept: 183.3814
    Epoch [12/50], Test Losses: mse: 3.9201, mae: 0.7985, huber: 0.5482, swd: 3.3221, ept: 183.7019
      Epoch 12 composite train-obj: 0.515076
            No improvement (0.5695), counter 1/5
    Epoch [13/50], Train Losses: mse: 3.9273, mae: 0.7478, huber: 0.5114, swd: 3.2939, ept: 184.6261
    Epoch [13/50], Val Losses: mse: 4.5297, mae: 0.8214, huber: 0.5706, swd: 3.7765, ept: 183.2056
    Epoch [13/50], Test Losses: mse: 3.9959, mae: 0.8025, huber: 0.5526, swd: 3.3183, ept: 183.3287
      Epoch 13 composite train-obj: 0.511379
            No improvement (0.5706), counter 2/5
    Epoch [14/50], Train Losses: mse: 3.9166, mae: 0.7468, huber: 0.5096, swd: 3.2841, ept: 184.6719
    Epoch [14/50], Val Losses: mse: 4.5998, mae: 0.7943, huber: 0.5599, swd: 3.8638, ept: 183.0184
    Epoch [14/50], Test Losses: mse: 4.0597, mae: 0.7741, huber: 0.5448, swd: 3.4113, ept: 183.4039
      Epoch 14 composite train-obj: 0.509627
            Val objective improved 0.5634 → 0.5599, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 3.9075, mae: 0.7417, huber: 0.5069, swd: 3.2820, ept: 184.7182
    Epoch [15/50], Val Losses: mse: 4.4541, mae: 0.8045, huber: 0.5567, swd: 3.7670, ept: 183.3998
    Epoch [15/50], Test Losses: mse: 3.9168, mae: 0.7801, huber: 0.5369, swd: 3.3120, ept: 183.8515
      Epoch 15 composite train-obj: 0.506854
            Val objective improved 0.5599 → 0.5567, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 3.8995, mae: 0.7396, huber: 0.5052, swd: 3.2705, ept: 184.7810
    Epoch [16/50], Val Losses: mse: 4.5923, mae: 0.7872, huber: 0.5500, swd: 3.8181, ept: 183.1830
    Epoch [16/50], Test Losses: mse: 4.0611, mae: 0.7724, huber: 0.5388, swd: 3.3643, ept: 183.3545
      Epoch 16 composite train-obj: 0.505215
            Val objective improved 0.5567 → 0.5500, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 3.8863, mae: 0.7340, huber: 0.5014, swd: 3.2629, ept: 184.8294
    Epoch [17/50], Val Losses: mse: 4.7233, mae: 0.7732, huber: 0.5530, swd: 3.9646, ept: 183.0082
    Epoch [17/50], Test Losses: mse: 4.1900, mae: 0.7604, huber: 0.5448, swd: 3.5112, ept: 183.1792
      Epoch 17 composite train-obj: 0.501378
            No improvement (0.5530), counter 1/5
    Epoch [18/50], Train Losses: mse: 3.8779, mae: 0.7313, huber: 0.4996, swd: 3.2559, ept: 184.8645
    Epoch [18/50], Val Losses: mse: 4.4871, mae: 0.7891, huber: 0.5505, swd: 3.8211, ept: 183.5089
    Epoch [18/50], Test Losses: mse: 3.9595, mae: 0.7701, huber: 0.5348, swd: 3.3702, ept: 183.7524
      Epoch 18 composite train-obj: 0.499622
            No improvement (0.5505), counter 2/5
    Epoch [19/50], Train Losses: mse: 3.8839, mae: 0.7298, huber: 0.4986, swd: 3.2656, ept: 184.8865
    Epoch [19/50], Val Losses: mse: 4.5327, mae: 0.7810, huber: 0.5465, swd: 3.8337, ept: 183.4481
    Epoch [19/50], Test Losses: mse: 3.9951, mae: 0.7643, huber: 0.5318, swd: 3.3787, ept: 183.6880
      Epoch 19 composite train-obj: 0.498584
            Val objective improved 0.5500 → 0.5465, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 3.8755, mae: 0.7287, huber: 0.4972, swd: 3.2585, ept: 184.9492
    Epoch [20/50], Val Losses: mse: 4.5928, mae: 0.7738, huber: 0.5431, swd: 3.8310, ept: 183.3978
    Epoch [20/50], Test Losses: mse: 4.0641, mae: 0.7645, huber: 0.5341, swd: 3.3768, ept: 183.5062
      Epoch 20 composite train-obj: 0.497237
            Val objective improved 0.5465 → 0.5431, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 3.8830, mae: 0.7279, huber: 0.4968, swd: 3.2661, ept: 184.9324
    Epoch [21/50], Val Losses: mse: 4.3902, mae: 0.8178, huber: 0.5588, swd: 3.7454, ept: 183.9367
    Epoch [21/50], Test Losses: mse: 3.8574, mae: 0.7949, huber: 0.5374, swd: 3.2886, ept: 184.3136
      Epoch 21 composite train-obj: 0.496813
            No improvement (0.5588), counter 1/5
    Epoch [22/50], Train Losses: mse: 3.8658, mae: 0.7241, huber: 0.4937, swd: 3.2530, ept: 184.9801
    Epoch [22/50], Val Losses: mse: 4.6248, mae: 0.7739, huber: 0.5472, swd: 3.8655, ept: 183.4464
    Epoch [22/50], Test Losses: mse: 4.0964, mae: 0.7620, huber: 0.5370, swd: 3.4140, ept: 183.4717
      Epoch 22 composite train-obj: 0.493705
            No improvement (0.5472), counter 2/5
    Epoch [23/50], Train Losses: mse: 3.8740, mae: 0.7242, huber: 0.4936, swd: 3.2616, ept: 184.9850
    Epoch [23/50], Val Losses: mse: 4.6138, mae: 0.7794, huber: 0.5443, swd: 3.8847, ept: 183.3969
    Epoch [23/50], Test Losses: mse: 4.0770, mae: 0.7611, huber: 0.5319, swd: 3.4300, ept: 183.5106
      Epoch 23 composite train-obj: 0.493594
            No improvement (0.5443), counter 3/5
    Epoch [24/50], Train Losses: mse: 3.8566, mae: 0.7191, huber: 0.4906, swd: 3.2456, ept: 185.0439
    Epoch [24/50], Val Losses: mse: 4.5818, mae: 0.7586, huber: 0.5366, swd: 3.8734, ept: 183.4148
    Epoch [24/50], Test Losses: mse: 4.0520, mae: 0.7433, huber: 0.5266, swd: 3.4212, ept: 183.5905
      Epoch 24 composite train-obj: 0.490604
            Val objective improved 0.5431 → 0.5366, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 3.8695, mae: 0.7188, huber: 0.4902, swd: 3.2606, ept: 185.0240
    Epoch [25/50], Val Losses: mse: 4.5854, mae: 0.7646, huber: 0.5391, swd: 3.8933, ept: 183.5606
    Epoch [25/50], Test Losses: mse: 4.0565, mae: 0.7533, huber: 0.5293, swd: 3.4397, ept: 183.7044
      Epoch 25 composite train-obj: 0.490151
            No improvement (0.5391), counter 1/5
    Epoch [26/50], Train Losses: mse: 3.8572, mae: 0.7151, huber: 0.4876, swd: 3.2533, ept: 185.0457
    Epoch [26/50], Val Losses: mse: 4.6945, mae: 0.7576, huber: 0.5442, swd: 3.8961, ept: 183.3638
    Epoch [26/50], Test Losses: mse: 4.1698, mae: 0.7523, huber: 0.5402, swd: 3.4466, ept: 183.3792
      Epoch 26 composite train-obj: 0.487599
            No improvement (0.5442), counter 2/5
    Epoch [27/50], Train Losses: mse: 3.8727, mae: 0.7169, huber: 0.4896, swd: 3.2621, ept: 185.0534
    Epoch [27/50], Val Losses: mse: 4.4872, mae: 0.7696, huber: 0.5348, swd: 3.7603, ept: 183.6479
    Epoch [27/50], Test Losses: mse: 3.9573, mae: 0.7540, huber: 0.5218, swd: 3.3084, ept: 183.7758
      Epoch 27 composite train-obj: 0.489574
            Val objective improved 0.5366 → 0.5348, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 3.8623, mae: 0.7154, huber: 0.4876, swd: 3.2573, ept: 185.0843
    Epoch [28/50], Val Losses: mse: 4.5222, mae: 0.7740, huber: 0.5369, swd: 3.8052, ept: 183.8380
    Epoch [28/50], Test Losses: mse: 3.9952, mae: 0.7589, huber: 0.5252, swd: 3.3490, ept: 183.9143
      Epoch 28 composite train-obj: 0.487603
            No improvement (0.5369), counter 1/5
    Epoch [29/50], Train Losses: mse: 3.8606, mae: 0.7148, huber: 0.4870, swd: 3.2538, ept: 185.0803
    Epoch [29/50], Val Losses: mse: 4.4400, mae: 0.7762, huber: 0.5341, swd: 3.7081, ept: 183.8739
    Epoch [29/50], Test Losses: mse: 3.9192, mae: 0.7634, huber: 0.5217, swd: 3.2578, ept: 183.9810
      Epoch 29 composite train-obj: 0.486968
            Val objective improved 0.5348 → 0.5341, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 3.8630, mae: 0.7124, huber: 0.4859, swd: 3.2574, ept: 185.0987
    Epoch [30/50], Val Losses: mse: 4.6111, mae: 0.7639, huber: 0.5363, swd: 3.8265, ept: 183.7060
    Epoch [30/50], Test Losses: mse: 4.0845, mae: 0.7531, huber: 0.5278, swd: 3.3718, ept: 183.7198
      Epoch 30 composite train-obj: 0.485885
            No improvement (0.5363), counter 1/5
    Epoch [31/50], Train Losses: mse: 3.8626, mae: 0.7120, huber: 0.4858, swd: 3.2590, ept: 185.1414
    Epoch [31/50], Val Losses: mse: 4.6340, mae: 0.7581, huber: 0.5357, swd: 3.9204, ept: 183.4520
    Epoch [31/50], Test Losses: mse: 4.0984, mae: 0.7430, huber: 0.5256, swd: 3.4639, ept: 183.5451
      Epoch 31 composite train-obj: 0.485791
            No improvement (0.5357), counter 2/5
    Epoch [32/50], Train Losses: mse: 3.8475, mae: 0.7107, huber: 0.4845, swd: 3.2461, ept: 185.1411
    Epoch [32/50], Val Losses: mse: 4.5389, mae: 0.7638, huber: 0.5368, swd: 3.8721, ept: 183.5976
    Epoch [32/50], Test Losses: mse: 4.0015, mae: 0.7468, huber: 0.5238, swd: 3.4174, ept: 183.9467
      Epoch 32 composite train-obj: 0.484510
            No improvement (0.5368), counter 3/5
    Epoch [33/50], Train Losses: mse: 3.8592, mae: 0.7109, huber: 0.4847, swd: 3.2557, ept: 185.1531
    Epoch [33/50], Val Losses: mse: 4.4223, mae: 0.7829, huber: 0.5399, swd: 3.6841, ept: 183.8962
    Epoch [33/50], Test Losses: mse: 3.9046, mae: 0.7715, huber: 0.5278, swd: 3.2374, ept: 184.0064
      Epoch 33 composite train-obj: 0.484697
            No improvement (0.5399), counter 4/5
    Epoch [34/50], Train Losses: mse: 3.8479, mae: 0.7085, huber: 0.4829, swd: 3.2473, ept: 185.1719
    Epoch [34/50], Val Losses: mse: 4.4462, mae: 0.7557, huber: 0.5259, swd: 3.7660, ept: 183.8594
    Epoch [34/50], Test Losses: mse: 3.9207, mae: 0.7396, huber: 0.5135, swd: 3.3141, ept: 184.0750
      Epoch 34 composite train-obj: 0.482912
            Val objective improved 0.5341 → 0.5259, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 3.8319, mae: 0.7061, huber: 0.4807, swd: 3.2339, ept: 185.1935
    Epoch [35/50], Val Losses: mse: 4.4314, mae: 0.7591, huber: 0.5278, swd: 3.7517, ept: 183.8880
    Epoch [35/50], Test Losses: mse: 3.9019, mae: 0.7421, huber: 0.5135, swd: 3.2988, ept: 184.1943
      Epoch 35 composite train-obj: 0.480668
            No improvement (0.5278), counter 1/5
    Epoch [36/50], Train Losses: mse: 3.8428, mae: 0.7064, huber: 0.4811, swd: 3.2436, ept: 185.2200
    Epoch [36/50], Val Losses: mse: 4.4130, mae: 0.7728, huber: 0.5326, swd: 3.7288, ept: 183.9386
    Epoch [36/50], Test Losses: mse: 3.8865, mae: 0.7550, huber: 0.5172, swd: 3.2758, ept: 184.0854
      Epoch 36 composite train-obj: 0.481071
            No improvement (0.5326), counter 2/5
    Epoch [37/50], Train Losses: mse: 3.8654, mae: 0.7089, huber: 0.4836, swd: 3.2615, ept: 185.1652
    Epoch [37/50], Val Losses: mse: 4.4353, mae: 0.7645, huber: 0.5258, swd: 3.7402, ept: 184.0279
    Epoch [37/50], Test Losses: mse: 3.9127, mae: 0.7474, huber: 0.5130, swd: 3.2888, ept: 184.2017
      Epoch 37 composite train-obj: 0.483623
            Val objective improved 0.5259 → 0.5258, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 3.8377, mae: 0.7044, huber: 0.4796, swd: 3.2422, ept: 185.2348
    Epoch [38/50], Val Losses: mse: 4.4422, mae: 0.7715, huber: 0.5304, swd: 3.7845, ept: 183.9913
    Epoch [38/50], Test Losses: mse: 3.9098, mae: 0.7526, huber: 0.5146, swd: 3.3296, ept: 184.2628
      Epoch 38 composite train-obj: 0.479567
            No improvement (0.5304), counter 1/5
    Epoch [39/50], Train Losses: mse: 3.8388, mae: 0.7061, huber: 0.4808, swd: 3.2390, ept: 185.2442
    Epoch [39/50], Val Losses: mse: 4.5099, mae: 0.7642, huber: 0.5359, swd: 3.7807, ept: 183.8178
    Epoch [39/50], Test Losses: mse: 3.9903, mae: 0.7502, huber: 0.5246, swd: 3.3342, ept: 183.8133
      Epoch 39 composite train-obj: 0.480796
            No improvement (0.5359), counter 2/5
    Epoch [40/50], Train Losses: mse: 3.8303, mae: 0.7021, huber: 0.4779, swd: 3.2344, ept: 185.2261
    Epoch [40/50], Val Losses: mse: 4.5248, mae: 0.7435, huber: 0.5235, swd: 3.7891, ept: 183.8297
    Epoch [40/50], Test Losses: mse: 4.0006, mae: 0.7321, huber: 0.5147, swd: 3.3395, ept: 183.9001
      Epoch 40 composite train-obj: 0.477943
            Val objective improved 0.5258 → 0.5235, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 3.8314, mae: 0.7025, huber: 0.4784, swd: 3.2329, ept: 185.2884
    Epoch [41/50], Val Losses: mse: 4.4332, mae: 0.7622, huber: 0.5321, swd: 3.8255, ept: 184.0020
    Epoch [41/50], Test Losses: mse: 3.9041, mae: 0.7459, huber: 0.5173, swd: 3.3730, ept: 184.2473
      Epoch 41 composite train-obj: 0.478378
            No improvement (0.5321), counter 1/5
    Epoch [42/50], Train Losses: mse: 3.8407, mae: 0.7027, huber: 0.4785, swd: 3.2418, ept: 185.2550
    Epoch [42/50], Val Losses: mse: 4.5502, mae: 0.7412, huber: 0.5224, swd: 3.8602, ept: 183.7199
    Epoch [42/50], Test Losses: mse: 4.0198, mae: 0.7247, huber: 0.5119, swd: 3.4099, ept: 183.8513
      Epoch 42 composite train-obj: 0.478473
            Val objective improved 0.5235 → 0.5224, saving checkpoint.
    Epoch [43/50], Train Losses: mse: 3.8296, mae: 0.6999, huber: 0.4766, swd: 3.2343, ept: 185.2829
    Epoch [43/50], Val Losses: mse: 4.3510, mae: 0.7724, huber: 0.5283, swd: 3.6900, ept: 184.1662
    Epoch [43/50], Test Losses: mse: 3.8279, mae: 0.7523, huber: 0.5114, swd: 3.2390, ept: 184.2977
      Epoch 43 composite train-obj: 0.476576
            No improvement (0.5283), counter 1/5
    Epoch [44/50], Train Losses: mse: 3.8427, mae: 0.7035, huber: 0.4795, swd: 3.2432, ept: 185.2412
    Epoch [44/50], Val Losses: mse: 4.5287, mae: 0.7653, huber: 0.5310, swd: 3.7169, ept: 183.8754
    Epoch [44/50], Test Losses: mse: 4.0110, mae: 0.7596, huber: 0.5232, swd: 3.2691, ept: 183.8353
      Epoch 44 composite train-obj: 0.479467
            No improvement (0.5310), counter 2/5
    Epoch [45/50], Train Losses: mse: 3.8154, mae: 0.6985, huber: 0.4751, swd: 3.2225, ept: 185.3213
    Epoch [45/50], Val Losses: mse: 4.6462, mae: 0.7395, huber: 0.5317, swd: 3.8860, ept: 183.6304
    Epoch [45/50], Test Losses: mse: 4.1246, mae: 0.7308, huber: 0.5259, swd: 3.4414, ept: 183.5215
      Epoch 45 composite train-obj: 0.475073
            No improvement (0.5317), counter 3/5
    Epoch [46/50], Train Losses: mse: 3.8329, mae: 0.7006, huber: 0.4774, swd: 3.2361, ept: 185.2889
    Epoch [46/50], Val Losses: mse: 4.5017, mae: 0.7533, huber: 0.5257, swd: 3.7800, ept: 183.8070
    Epoch [46/50], Test Losses: mse: 3.9750, mae: 0.7381, huber: 0.5138, swd: 3.3323, ept: 183.8350
      Epoch 46 composite train-obj: 0.477364
            No improvement (0.5257), counter 4/5
    Epoch [47/50], Train Losses: mse: 3.8180, mae: 0.6976, huber: 0.4749, swd: 3.2252, ept: 185.3049
    Epoch [47/50], Val Losses: mse: 4.4493, mae: 0.7502, huber: 0.5204, swd: 3.7595, ept: 183.9540
    Epoch [47/50], Test Losses: mse: 3.9240, mae: 0.7339, huber: 0.5079, swd: 3.3111, ept: 184.1405
      Epoch 47 composite train-obj: 0.474933
            Val objective improved 0.5224 → 0.5204, saving checkpoint.
    Epoch [48/50], Train Losses: mse: 3.8215, mae: 0.7000, huber: 0.4760, swd: 3.2266, ept: 185.3233
    Epoch [48/50], Val Losses: mse: 4.5026, mae: 0.7467, huber: 0.5217, swd: 3.7987, ept: 183.9633
    Epoch [48/50], Test Losses: mse: 3.9752, mae: 0.7338, huber: 0.5111, swd: 3.3480, ept: 184.0108
      Epoch 48 composite train-obj: 0.476033
            No improvement (0.5217), counter 1/5
    Epoch [49/50], Train Losses: mse: 3.8278, mae: 0.6994, huber: 0.4761, swd: 3.2325, ept: 185.2973
    Epoch [49/50], Val Losses: mse: 4.4008, mae: 0.7479, huber: 0.5186, swd: 3.7463, ept: 184.0260
    Epoch [49/50], Test Losses: mse: 3.8768, mae: 0.7300, huber: 0.5050, swd: 3.2955, ept: 184.0900
      Epoch 49 composite train-obj: 0.476117
            Val objective improved 0.5204 → 0.5186, saving checkpoint.
    Epoch [50/50], Train Losses: mse: 3.8118, mae: 0.6950, huber: 0.4729, swd: 3.2212, ept: 185.3503
    Epoch [50/50], Val Losses: mse: 4.5539, mae: 0.7325, huber: 0.5221, swd: 3.8064, ept: 183.9148
    Epoch [50/50], Test Losses: mse: 4.0335, mae: 0.7230, huber: 0.5151, swd: 3.3606, ept: 183.9082
      Epoch 50 composite train-obj: 0.472878
            No improvement (0.5221), counter 1/5
    Epoch [50/50], Test Losses: mse: 3.8768, mae: 0.7300, huber: 0.5050, swd: 3.2955, ept: 184.0900
    Best round's Test MSE: 3.8768, MAE: 0.7300, SWD: 3.2955
    Best round's Validation MSE: 4.4008, MAE: 0.7479, SWD: 3.7463
    Best round's Test verification MSE : 3.8768, MAE: 0.7300, SWD: 3.2955
    Time taken: 112.03 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 5.5151, mae: 1.0654, huber: 0.7776, swd: 3.9292, ept: 173.2457
    Epoch [1/50], Val Losses: mse: 5.4841, mae: 0.9709, huber: 0.6970, swd: 4.0980, ept: 178.2159
    Epoch [1/50], Test Losses: mse: 4.8572, mae: 0.9529, huber: 0.6820, swd: 3.6266, ept: 179.3775
      Epoch 1 composite train-obj: 0.777562
            Val objective improved inf → 0.6970, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 4.4635, mae: 0.8853, huber: 0.6139, swd: 3.3258, ept: 181.8352
    Epoch [2/50], Val Losses: mse: 5.1105, mae: 0.9384, huber: 0.6552, swd: 3.7771, ept: 180.4305
    Epoch [2/50], Test Losses: mse: 4.5299, mae: 0.9206, huber: 0.6401, swd: 3.3283, ept: 181.0119
      Epoch 2 composite train-obj: 0.613927
            Val objective improved 0.6970 → 0.6552, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 4.2436, mae: 0.8452, huber: 0.5817, swd: 3.1685, ept: 182.8583
    Epoch [3/50], Val Losses: mse: 5.0007, mae: 0.8798, huber: 0.6319, swd: 3.7467, ept: 180.9923
    Epoch [3/50], Test Losses: mse: 4.4244, mae: 0.8593, huber: 0.6153, swd: 3.3144, ept: 181.4545
      Epoch 3 composite train-obj: 0.581659
            Val objective improved 0.6552 → 0.6319, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 4.1602, mae: 0.8227, huber: 0.5661, swd: 3.1124, ept: 183.3340
    Epoch [4/50], Val Losses: mse: 4.9765, mae: 0.8565, huber: 0.6168, swd: 3.7094, ept: 181.3745
    Epoch [4/50], Test Losses: mse: 4.4124, mae: 0.8389, huber: 0.6039, swd: 3.2815, ept: 181.7405
      Epoch 4 composite train-obj: 0.566130
            Val objective improved 0.6319 → 0.6168, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 4.0948, mae: 0.8064, huber: 0.5542, swd: 3.0675, ept: 183.6288
    Epoch [5/50], Val Losses: mse: 4.6604, mae: 0.8797, huber: 0.6144, swd: 3.5243, ept: 182.3367
    Epoch [5/50], Test Losses: mse: 4.1027, mae: 0.8568, huber: 0.5930, swd: 3.0971, ept: 183.0238
      Epoch 5 composite train-obj: 0.554213
            Val objective improved 0.6168 → 0.6144, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 4.0593, mae: 0.7949, huber: 0.5453, swd: 3.0506, ept: 183.8514
    Epoch [6/50], Val Losses: mse: 4.8304, mae: 0.8423, huber: 0.5957, swd: 3.5929, ept: 182.1249
    Epoch [6/50], Test Losses: mse: 4.2814, mae: 0.8254, huber: 0.5833, swd: 3.1692, ept: 182.4044
      Epoch 6 composite train-obj: 0.545340
            Val objective improved 0.6144 → 0.5957, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 4.0235, mae: 0.7843, huber: 0.5376, swd: 3.0264, ept: 184.0537
    Epoch [7/50], Val Losses: mse: 4.5769, mae: 0.8425, huber: 0.5879, swd: 3.4689, ept: 182.6544
    Epoch [7/50], Test Losses: mse: 4.0293, mae: 0.8202, huber: 0.5679, swd: 3.0507, ept: 183.2836
      Epoch 7 composite train-obj: 0.537593
            Val objective improved 0.5957 → 0.5879, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 4.0073, mae: 0.7752, huber: 0.5317, swd: 3.0189, ept: 184.1556
    Epoch [8/50], Val Losses: mse: 4.5571, mae: 0.8444, huber: 0.5811, swd: 3.4619, ept: 182.8775
    Epoch [8/50], Test Losses: mse: 4.0168, mae: 0.8247, huber: 0.5625, swd: 3.0431, ept: 183.2561
      Epoch 8 composite train-obj: 0.531748
            Val objective improved 0.5879 → 0.5811, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 3.9781, mae: 0.7671, huber: 0.5255, swd: 3.0019, ept: 184.2843
    Epoch [9/50], Val Losses: mse: 4.5963, mae: 0.8373, huber: 0.5801, swd: 3.4238, ept: 182.8157
    Epoch [9/50], Test Losses: mse: 4.0638, mae: 0.8253, huber: 0.5668, swd: 3.0071, ept: 183.0811
      Epoch 9 composite train-obj: 0.525466
            Val objective improved 0.5811 → 0.5801, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 3.9612, mae: 0.7616, huber: 0.5217, swd: 2.9915, ept: 184.4111
    Epoch [10/50], Val Losses: mse: 4.6328, mae: 0.8189, huber: 0.5700, swd: 3.5095, ept: 182.7527
    Epoch [10/50], Test Losses: mse: 4.0892, mae: 0.7960, huber: 0.5535, swd: 3.0962, ept: 183.2398
      Epoch 10 composite train-obj: 0.521733
            Val objective improved 0.5801 → 0.5700, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 3.9493, mae: 0.7593, huber: 0.5192, swd: 2.9811, ept: 184.4739
    Epoch [11/50], Val Losses: mse: 4.5364, mae: 0.8365, huber: 0.5749, swd: 3.4742, ept: 183.2374
    Epoch [11/50], Test Losses: mse: 3.9975, mae: 0.8145, huber: 0.5561, swd: 3.0590, ept: 183.6876
      Epoch 11 composite train-obj: 0.519223
            No improvement (0.5749), counter 1/5
    Epoch [12/50], Train Losses: mse: 3.9298, mae: 0.7525, huber: 0.5144, swd: 2.9687, ept: 184.5538
    Epoch [12/50], Val Losses: mse: 4.7146, mae: 0.8032, huber: 0.5659, swd: 3.5085, ept: 182.9782
    Epoch [12/50], Test Losses: mse: 4.1825, mae: 0.7946, huber: 0.5565, swd: 3.0959, ept: 182.9483
      Epoch 12 composite train-obj: 0.514432
            Val objective improved 0.5700 → 0.5659, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 3.9253, mae: 0.7484, huber: 0.5115, swd: 2.9716, ept: 184.6156
    Epoch [13/50], Val Losses: mse: 4.7065, mae: 0.8030, huber: 0.5671, swd: 3.5933, ept: 182.9110
    Epoch [13/50], Test Losses: mse: 4.1695, mae: 0.7857, huber: 0.5555, swd: 3.1814, ept: 183.2741
      Epoch 13 composite train-obj: 0.511489
            No improvement (0.5671), counter 1/5
    Epoch [14/50], Train Losses: mse: 3.9068, mae: 0.7439, huber: 0.5087, swd: 2.9571, ept: 184.7131
    Epoch [14/50], Val Losses: mse: 4.6107, mae: 0.8028, huber: 0.5633, swd: 3.5367, ept: 183.0754
    Epoch [14/50], Test Losses: mse: 4.0735, mae: 0.7918, huber: 0.5506, swd: 3.1242, ept: 183.4778
      Epoch 14 composite train-obj: 0.508729
            Val objective improved 0.5659 → 0.5633, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 3.9228, mae: 0.7423, huber: 0.5075, swd: 2.9723, ept: 184.7177
    Epoch [15/50], Val Losses: mse: 4.5426, mae: 0.8003, huber: 0.5569, swd: 3.5024, ept: 183.2516
    Epoch [15/50], Test Losses: mse: 4.0091, mae: 0.7822, huber: 0.5415, swd: 3.0922, ept: 183.5643
      Epoch 15 composite train-obj: 0.507519
            Val objective improved 0.5633 → 0.5569, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 3.8953, mae: 0.7386, huber: 0.5041, swd: 2.9525, ept: 184.7574
    Epoch [16/50], Val Losses: mse: 4.6427, mae: 0.7826, huber: 0.5552, swd: 3.5500, ept: 183.0631
    Epoch [16/50], Test Losses: mse: 4.1077, mae: 0.7647, huber: 0.5424, swd: 3.1412, ept: 183.3709
      Epoch 16 composite train-obj: 0.504112
            Val objective improved 0.5569 → 0.5552, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 3.9051, mae: 0.7355, huber: 0.5029, swd: 2.9626, ept: 184.7992
    Epoch [17/50], Val Losses: mse: 4.5514, mae: 0.7835, huber: 0.5519, swd: 3.4888, ept: 183.2437
    Epoch [17/50], Test Losses: mse: 4.0174, mae: 0.7642, huber: 0.5368, swd: 3.0796, ept: 183.5567
      Epoch 17 composite train-obj: 0.502869
            Val objective improved 0.5552 → 0.5519, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 3.8860, mae: 0.7327, huber: 0.5003, swd: 2.9477, ept: 184.8611
    Epoch [18/50], Val Losses: mse: 4.6576, mae: 0.7728, huber: 0.5506, swd: 3.5091, ept: 183.0058
    Epoch [18/50], Test Losses: mse: 4.1294, mae: 0.7621, huber: 0.5427, swd: 3.1036, ept: 183.3395
      Epoch 18 composite train-obj: 0.500288
            Val objective improved 0.5519 → 0.5506, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 3.8853, mae: 0.7293, huber: 0.4981, swd: 2.9491, ept: 184.8771
    Epoch [19/50], Val Losses: mse: 4.4860, mae: 0.7843, huber: 0.5457, swd: 3.4153, ept: 183.6656
    Epoch [19/50], Test Losses: mse: 3.9554, mae: 0.7676, huber: 0.5305, swd: 3.0041, ept: 183.6942
      Epoch 19 composite train-obj: 0.498077
            Val objective improved 0.5506 → 0.5457, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 3.8902, mae: 0.7273, huber: 0.4969, swd: 2.9583, ept: 184.9156
    Epoch [20/50], Val Losses: mse: 4.5187, mae: 0.7738, huber: 0.5422, swd: 3.4479, ept: 183.4856
    Epoch [20/50], Test Losses: mse: 3.9866, mae: 0.7547, huber: 0.5274, swd: 3.0378, ept: 183.5543
      Epoch 20 composite train-obj: 0.496916
            Val objective improved 0.5457 → 0.5422, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 3.8779, mae: 0.7274, huber: 0.4960, swd: 2.9453, ept: 184.9488
    Epoch [21/50], Val Losses: mse: 4.5065, mae: 0.7928, huber: 0.5461, swd: 3.4142, ept: 183.5243
    Epoch [21/50], Test Losses: mse: 3.9787, mae: 0.7735, huber: 0.5320, swd: 3.0048, ept: 183.6378
      Epoch 21 composite train-obj: 0.495954
            No improvement (0.5461), counter 1/5
    Epoch [22/50], Train Losses: mse: 3.8753, mae: 0.7255, huber: 0.4952, swd: 2.9428, ept: 184.9607
    Epoch [22/50], Val Losses: mse: 4.4397, mae: 0.7853, huber: 0.5419, swd: 3.3710, ept: 183.6833
    Epoch [22/50], Test Losses: mse: 3.9103, mae: 0.7684, huber: 0.5259, swd: 2.9613, ept: 183.9296
      Epoch 22 composite train-obj: 0.495245
            Val objective improved 0.5422 → 0.5419, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 3.8626, mae: 0.7210, huber: 0.4917, swd: 2.9386, ept: 185.0002
    Epoch [23/50], Val Losses: mse: 4.4458, mae: 0.7800, huber: 0.5402, swd: 3.4037, ept: 183.6773
    Epoch [23/50], Test Losses: mse: 3.9150, mae: 0.7620, huber: 0.5240, swd: 2.9931, ept: 183.8385
      Epoch 23 composite train-obj: 0.491745
            Val objective improved 0.5419 → 0.5402, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 3.8691, mae: 0.7216, huber: 0.4925, swd: 2.9404, ept: 185.0045
    Epoch [24/50], Val Losses: mse: 4.5698, mae: 0.7867, huber: 0.5481, swd: 3.3768, ept: 183.5669
    Epoch [24/50], Test Losses: mse: 4.0483, mae: 0.7766, huber: 0.5390, swd: 2.9689, ept: 183.6124
      Epoch 24 composite train-obj: 0.492467
            No improvement (0.5481), counter 1/5
    Epoch [25/50], Train Losses: mse: 3.8658, mae: 0.7191, huber: 0.4906, swd: 2.9413, ept: 185.0441
    Epoch [25/50], Val Losses: mse: 4.4700, mae: 0.7702, huber: 0.5368, swd: 3.4067, ept: 183.7394
    Epoch [25/50], Test Losses: mse: 3.9410, mae: 0.7509, huber: 0.5217, swd: 2.9960, ept: 183.8179
      Epoch 25 composite train-obj: 0.490622
            Val objective improved 0.5402 → 0.5368, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 3.8623, mae: 0.7175, huber: 0.4897, swd: 2.9392, ept: 185.0210
    Epoch [26/50], Val Losses: mse: 4.5603, mae: 0.7709, huber: 0.5408, swd: 3.3963, ept: 183.5438
    Epoch [26/50], Test Losses: mse: 4.0438, mae: 0.7632, huber: 0.5337, swd: 2.9911, ept: 183.5900
      Epoch 26 composite train-obj: 0.489657
            No improvement (0.5408), counter 1/5
    Epoch [27/50], Train Losses: mse: 3.8699, mae: 0.7158, huber: 0.4886, swd: 2.9474, ept: 185.0427
    Epoch [27/50], Val Losses: mse: 4.4013, mae: 0.7967, huber: 0.5493, swd: 3.3362, ept: 183.8177
    Epoch [27/50], Test Losses: mse: 3.8721, mae: 0.7771, huber: 0.5303, swd: 2.9284, ept: 184.0068
      Epoch 27 composite train-obj: 0.488561
            No improvement (0.5493), counter 2/5
    Epoch [28/50], Train Losses: mse: 3.8553, mae: 0.7147, huber: 0.4871, swd: 2.9355, ept: 185.0855
    Epoch [28/50], Val Losses: mse: 4.3986, mae: 0.7807, huber: 0.5395, swd: 3.3478, ept: 183.8539
    Epoch [28/50], Test Losses: mse: 3.8794, mae: 0.7652, huber: 0.5255, swd: 2.9409, ept: 184.0991
      Epoch 28 composite train-obj: 0.487127
            No improvement (0.5395), counter 3/5
    Epoch [29/50], Train Losses: mse: 3.8480, mae: 0.7126, huber: 0.4858, swd: 2.9319, ept: 185.1145
    Epoch [29/50], Val Losses: mse: 4.5664, mae: 0.7576, huber: 0.5328, swd: 3.5223, ept: 183.5849
    Epoch [29/50], Test Losses: mse: 4.0325, mae: 0.7395, huber: 0.5206, swd: 3.1102, ept: 183.8233
      Epoch 29 composite train-obj: 0.485756
            Val objective improved 0.5368 → 0.5328, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 3.8601, mae: 0.7144, huber: 0.4867, swd: 2.9400, ept: 185.1320
    Epoch [30/50], Val Losses: mse: 4.5057, mae: 0.7605, huber: 0.5307, swd: 3.4632, ept: 183.6334
    Epoch [30/50], Test Losses: mse: 3.9714, mae: 0.7413, huber: 0.5162, swd: 3.0529, ept: 183.8398
      Epoch 30 composite train-obj: 0.486694
            Val objective improved 0.5328 → 0.5307, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 3.8550, mae: 0.7111, huber: 0.4848, swd: 2.9388, ept: 185.1258
    Epoch [31/50], Val Losses: mse: 4.4577, mae: 0.7638, huber: 0.5305, swd: 3.4009, ept: 183.9274
    Epoch [31/50], Test Losses: mse: 3.9312, mae: 0.7497, huber: 0.5176, swd: 2.9902, ept: 184.0769
      Epoch 31 composite train-obj: 0.484836
            Val objective improved 0.5307 → 0.5305, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 3.8521, mae: 0.7117, huber: 0.4850, swd: 2.9365, ept: 185.1430
    Epoch [32/50], Val Losses: mse: 4.5057, mae: 0.7796, huber: 0.5408, swd: 3.3885, ept: 183.8839
    Epoch [32/50], Test Losses: mse: 3.9811, mae: 0.7674, huber: 0.5293, swd: 2.9744, ept: 183.9772
      Epoch 32 composite train-obj: 0.484992
            No improvement (0.5408), counter 1/5
    Epoch [33/50], Train Losses: mse: 3.8656, mae: 0.7134, huber: 0.4859, swd: 2.9450, ept: 185.1176
    Epoch [33/50], Val Losses: mse: 4.4671, mae: 0.7680, huber: 0.5301, swd: 3.4405, ept: 183.8714
    Epoch [33/50], Test Losses: mse: 3.9338, mae: 0.7485, huber: 0.5152, swd: 3.0274, ept: 184.1118
      Epoch 33 composite train-obj: 0.485929
            Val objective improved 0.5305 → 0.5301, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 3.8467, mae: 0.7081, huber: 0.4822, swd: 2.9350, ept: 185.1675
    Epoch [34/50], Val Losses: mse: 4.6182, mae: 0.7589, huber: 0.5334, swd: 3.4650, ept: 183.6959
    Epoch [34/50], Test Losses: mse: 4.0987, mae: 0.7541, huber: 0.5285, swd: 3.0595, ept: 183.6699
      Epoch 34 composite train-obj: 0.482165
            No improvement (0.5334), counter 1/5
    Epoch [35/50], Train Losses: mse: 3.8346, mae: 0.7073, huber: 0.4811, swd: 2.9225, ept: 185.2127
    Epoch [35/50], Val Losses: mse: 4.4867, mae: 0.7621, huber: 0.5326, swd: 3.3662, ept: 183.7428
    Epoch [35/50], Test Losses: mse: 3.9650, mae: 0.7543, huber: 0.5228, swd: 2.9597, ept: 183.7854
      Epoch 35 composite train-obj: 0.481130
            No improvement (0.5326), counter 2/5
    Epoch [36/50], Train Losses: mse: 3.8552, mae: 0.7064, huber: 0.4818, swd: 2.9396, ept: 185.1728
    Epoch [36/50], Val Losses: mse: 4.4751, mae: 0.7605, huber: 0.5264, swd: 3.3891, ept: 183.8740
    Epoch [36/50], Test Losses: mse: 3.9509, mae: 0.7472, huber: 0.5151, swd: 2.9807, ept: 183.9732
      Epoch 36 composite train-obj: 0.481834
            Val objective improved 0.5301 → 0.5264, saving checkpoint.
    Epoch [37/50], Train Losses: mse: 3.8383, mae: 0.7054, huber: 0.4805, swd: 2.9287, ept: 185.2096
    Epoch [37/50], Val Losses: mse: 4.4781, mae: 0.7756, huber: 0.5291, swd: 3.4114, ept: 183.9013
    Epoch [37/50], Test Losses: mse: 3.9475, mae: 0.7569, huber: 0.5154, swd: 3.0000, ept: 184.0229
      Epoch 37 composite train-obj: 0.480482
            No improvement (0.5291), counter 1/5
    Epoch [38/50], Train Losses: mse: 3.8510, mae: 0.7052, huber: 0.4804, swd: 2.9400, ept: 185.1939
    Epoch [38/50], Val Losses: mse: 4.3920, mae: 0.7761, huber: 0.5346, swd: 3.3944, ept: 184.1396
    Epoch [38/50], Test Losses: mse: 3.8702, mae: 0.7629, huber: 0.5206, swd: 2.9881, ept: 184.3749
      Epoch 38 composite train-obj: 0.480417
            No improvement (0.5346), counter 2/5
    Epoch [39/50], Train Losses: mse: 3.8412, mae: 0.7039, huber: 0.4802, swd: 2.9291, ept: 185.2340
    Epoch [39/50], Val Losses: mse: 4.4385, mae: 0.7706, huber: 0.5321, swd: 3.3509, ept: 184.0519
    Epoch [39/50], Test Losses: mse: 3.9155, mae: 0.7572, huber: 0.5194, swd: 2.9445, ept: 184.1907
      Epoch 39 composite train-obj: 0.480228
            No improvement (0.5321), counter 3/5
    Epoch [40/50], Train Losses: mse: 3.8306, mae: 0.7041, huber: 0.4796, swd: 2.9204, ept: 185.2354
    Epoch [40/50], Val Losses: mse: 4.4969, mae: 0.7554, huber: 0.5256, swd: 3.4102, ept: 183.8143
    Epoch [40/50], Test Losses: mse: 3.9702, mae: 0.7389, huber: 0.5137, swd: 3.0017, ept: 183.8389
      Epoch 40 composite train-obj: 0.479560
            Val objective improved 0.5264 → 0.5256, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 3.8385, mae: 0.7042, huber: 0.4799, swd: 2.9280, ept: 185.2618
    Epoch [41/50], Val Losses: mse: 4.3877, mae: 0.7597, huber: 0.5266, swd: 3.3813, ept: 183.9781
    Epoch [41/50], Test Losses: mse: 3.8617, mae: 0.7421, huber: 0.5119, swd: 2.9713, ept: 184.2913
      Epoch 41 composite train-obj: 0.479852
            No improvement (0.5266), counter 1/5
    Epoch [42/50], Train Losses: mse: 3.8302, mae: 0.7018, huber: 0.4779, swd: 2.9248, ept: 185.2695
    Epoch [42/50], Val Losses: mse: 4.4854, mae: 0.7446, huber: 0.5222, swd: 3.3927, ept: 183.8721
    Epoch [42/50], Test Losses: mse: 3.9617, mae: 0.7287, huber: 0.5117, swd: 2.9858, ept: 183.9449
      Epoch 42 composite train-obj: 0.477928
            Val objective improved 0.5256 → 0.5222, saving checkpoint.
    Epoch [43/50], Train Losses: mse: 3.8291, mae: 0.7017, huber: 0.4776, swd: 2.9200, ept: 185.2804
    Epoch [43/50], Val Losses: mse: 4.4496, mae: 0.7536, huber: 0.5234, swd: 3.3984, ept: 184.0427
    Epoch [43/50], Test Losses: mse: 3.9259, mae: 0.7399, huber: 0.5120, swd: 2.9879, ept: 184.1454
      Epoch 43 composite train-obj: 0.477649
            No improvement (0.5234), counter 1/5
    Epoch [44/50], Train Losses: mse: 3.8422, mae: 0.7020, huber: 0.4783, swd: 2.9358, ept: 185.2627
    Epoch [44/50], Val Losses: mse: 4.3641, mae: 0.7647, huber: 0.5273, swd: 3.3668, ept: 184.0695
    Epoch [44/50], Test Losses: mse: 3.8364, mae: 0.7446, huber: 0.5103, swd: 2.9590, ept: 184.3501
      Epoch 44 composite train-obj: 0.478262
            No improvement (0.5273), counter 2/5
    Epoch [45/50], Train Losses: mse: 3.8304, mae: 0.7001, huber: 0.4770, swd: 2.9232, ept: 185.2867
    Epoch [45/50], Val Losses: mse: 4.4786, mae: 0.7561, huber: 0.5235, swd: 3.4049, ept: 183.9395
    Epoch [45/50], Test Losses: mse: 3.9542, mae: 0.7426, huber: 0.5127, swd: 2.9983, ept: 184.0393
      Epoch 45 composite train-obj: 0.477025
            No improvement (0.5235), counter 3/5
    Epoch [46/50], Train Losses: mse: 3.8270, mae: 0.7010, huber: 0.4768, swd: 2.9195, ept: 185.2869
    Epoch [46/50], Val Losses: mse: 4.3806, mae: 0.7609, huber: 0.5275, swd: 3.3781, ept: 184.1816
    Epoch [46/50], Test Losses: mse: 3.8620, mae: 0.7459, huber: 0.5150, swd: 2.9719, ept: 184.4798
      Epoch 46 composite train-obj: 0.476790
            No improvement (0.5275), counter 4/5
    Epoch [47/50], Train Losses: mse: 3.8312, mae: 0.7000, huber: 0.4765, swd: 2.9244, ept: 185.2920
    Epoch [47/50], Val Losses: mse: 4.3609, mae: 0.7592, huber: 0.5234, swd: 3.3843, ept: 184.2472
    Epoch [47/50], Test Losses: mse: 3.8379, mae: 0.7425, huber: 0.5079, swd: 2.9766, ept: 184.3435
      Epoch 47 composite train-obj: 0.476510
    Epoch [47/50], Test Losses: mse: 3.9617, mae: 0.7287, huber: 0.5117, swd: 2.9858, ept: 183.9449
    Best round's Test MSE: 3.9617, MAE: 0.7287, SWD: 2.9858
    Best round's Validation MSE: 4.4854, MAE: 0.7446, SWD: 3.3927
    Best round's Test verification MSE : 3.9617, MAE: 0.7287, SWD: 2.9858
    Time taken: 102.73 seconds
    
    ==================================================
    Experiment Summary (DLinear_rossler_seq336_pred196_20250513_1211)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 3.9438 ± 0.0490
      mae: 0.7375 ± 0.0116
      huber: 0.5130 ± 0.0071
      swd: 3.1948 ± 0.1478
      ept: 183.9378 ± 0.1273
      count: 37.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 4.4696 ± 0.0509
      mae: 0.7540 ± 0.0110
      huber: 0.5250 ± 0.0066
      swd: 3.6306 ± 0.1682
      ept: 183.8430 ± 0.1627
      count: 37.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 278.92 seconds
    
    Experiment complete: DLinear_rossler_seq336_pred196_20250513_1211
    Model: DLinear
    Dataset: rossler
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 280
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 336, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 6.8865, mae: 1.2651, huber: 0.9547, swd: 4.3666, ept: 279.7297
    Epoch [1/50], Val Losses: mse: 7.1273, mae: 1.2564, huber: 0.9396, swd: 4.6624, ept: 285.8754
    Epoch [1/50], Test Losses: mse: 6.2602, mae: 1.2216, huber: 0.9063, swd: 4.1365, ept: 288.4946
      Epoch 1 composite train-obj: 0.954682
            Val objective improved inf → 0.9396, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.9471, mae: 1.1124, huber: 0.8107, swd: 3.8618, ept: 295.0428
    Epoch [2/50], Val Losses: mse: 7.1129, mae: 1.2046, huber: 0.8965, swd: 4.6004, ept: 289.1442
    Epoch [2/50], Test Losses: mse: 6.2410, mae: 1.1633, huber: 0.8633, swd: 4.0831, ept: 290.4371
      Epoch 2 composite train-obj: 0.810749
            Val objective improved 0.9396 → 0.8965, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.7012, mae: 1.0786, huber: 0.7824, swd: 3.7246, ept: 297.0403
    Epoch [3/50], Val Losses: mse: 6.4989, mae: 1.1869, huber: 0.8777, swd: 4.2523, ept: 292.2441
    Epoch [3/50], Test Losses: mse: 5.6493, mae: 1.1358, huber: 0.8311, swd: 3.7428, ept: 294.1535
      Epoch 3 composite train-obj: 0.782442
            Val objective improved 0.8965 → 0.8777, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.5492, mae: 1.0579, huber: 0.7660, swd: 3.6396, ept: 298.3065
    Epoch [4/50], Val Losses: mse: 6.6350, mae: 1.1460, huber: 0.8567, swd: 4.4040, ept: 291.7528
    Epoch [4/50], Test Losses: mse: 5.7911, mae: 1.1056, huber: 0.8183, swd: 3.9035, ept: 293.6718
      Epoch 4 composite train-obj: 0.766029
            Val objective improved 0.8777 → 0.8567, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.4962, mae: 1.0491, huber: 0.7599, swd: 3.6253, ept: 298.9006
    Epoch [5/50], Val Losses: mse: 6.4046, mae: 1.1550, huber: 0.8524, swd: 4.2266, ept: 293.2335
    Epoch [5/50], Test Losses: mse: 5.5465, mae: 1.1079, huber: 0.8044, swd: 3.7236, ept: 295.0704
      Epoch 5 composite train-obj: 0.759921
            Val objective improved 0.8567 → 0.8524, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.4158, mae: 1.0384, huber: 0.7506, swd: 3.5729, ept: 299.4708
    Epoch [6/50], Val Losses: mse: 6.5847, mae: 1.1322, huber: 0.8423, swd: 4.3241, ept: 292.6847
    Epoch [6/50], Test Losses: mse: 5.7319, mae: 1.0950, huber: 0.8051, swd: 3.8182, ept: 294.4336
      Epoch 6 composite train-obj: 0.750589
            Val objective improved 0.8524 → 0.8423, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.3923, mae: 1.0306, huber: 0.7452, swd: 3.5701, ept: 299.8336
    Epoch [7/50], Val Losses: mse: 6.2247, mae: 1.1425, huber: 0.8439, swd: 4.1423, ept: 294.4025
    Epoch [7/50], Test Losses: mse: 5.3699, mae: 1.0871, huber: 0.7906, swd: 3.6265, ept: 296.1355
      Epoch 7 composite train-obj: 0.745155
            No improvement (0.8439), counter 1/5
    Epoch [8/50], Train Losses: mse: 5.3671, mae: 1.0282, huber: 0.7433, swd: 3.5499, ept: 300.0991
    Epoch [8/50], Val Losses: mse: 6.1852, mae: 1.1335, huber: 0.8412, swd: 4.1677, ept: 294.3844
    Epoch [8/50], Test Losses: mse: 5.3438, mae: 1.0781, huber: 0.7879, swd: 3.6573, ept: 296.2544
      Epoch 8 composite train-obj: 0.743300
            Val objective improved 0.8423 → 0.8412, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 5.3428, mae: 1.0199, huber: 0.7369, swd: 3.5492, ept: 300.4297
    Epoch [9/50], Val Losses: mse: 6.5332, mae: 1.1150, huber: 0.8297, swd: 4.2819, ept: 292.9256
    Epoch [9/50], Test Losses: mse: 5.6438, mae: 1.0657, huber: 0.7843, swd: 3.7565, ept: 294.0041
      Epoch 9 composite train-obj: 0.736913
            Val objective improved 0.8412 → 0.8297, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 5.3200, mae: 1.0165, huber: 0.7341, swd: 3.5284, ept: 300.5902
    Epoch [10/50], Val Losses: mse: 6.5397, mae: 1.1117, huber: 0.8295, swd: 4.2464, ept: 293.9574
    Epoch [10/50], Test Losses: mse: 5.6592, mae: 1.0692, huber: 0.7885, swd: 3.7205, ept: 294.8597
      Epoch 10 composite train-obj: 0.734136
            Val objective improved 0.8297 → 0.8295, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 5.3206, mae: 1.0144, huber: 0.7330, swd: 3.5388, ept: 300.5873
    Epoch [11/50], Val Losses: mse: 5.9444, mae: 1.1486, huber: 0.8425, swd: 4.0505, ept: 296.2344
    Epoch [11/50], Test Losses: mse: 5.0975, mae: 1.0869, huber: 0.7804, swd: 3.5201, ept: 298.6434
      Epoch 11 composite train-obj: 0.733037
            No improvement (0.8425), counter 1/5
    Epoch [12/50], Train Losses: mse: 5.2995, mae: 1.0102, huber: 0.7294, swd: 3.5217, ept: 300.9219
    Epoch [12/50], Val Losses: mse: 6.3668, mae: 1.1023, huber: 0.8190, swd: 4.2383, ept: 294.9578
    Epoch [12/50], Test Losses: mse: 5.4931, mae: 1.0518, huber: 0.7718, swd: 3.7065, ept: 297.1751
      Epoch 12 composite train-obj: 0.729420
            Val objective improved 0.8295 → 0.8190, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 5.2894, mae: 1.0057, huber: 0.7263, swd: 3.5205, ept: 300.9694
    Epoch [13/50], Val Losses: mse: 6.4589, mae: 1.1336, huber: 0.8319, swd: 4.1660, ept: 294.6270
    Epoch [13/50], Test Losses: mse: 5.5559, mae: 1.0846, huber: 0.7842, swd: 3.6268, ept: 295.8591
      Epoch 13 composite train-obj: 0.726275
            No improvement (0.8319), counter 1/5
    Epoch [14/50], Train Losses: mse: 5.2656, mae: 1.0049, huber: 0.7255, swd: 3.5002, ept: 301.2182
    Epoch [14/50], Val Losses: mse: 6.6558, mae: 1.1010, huber: 0.8259, swd: 4.3705, ept: 292.4219
    Epoch [14/50], Test Losses: mse: 5.7748, mae: 1.0606, huber: 0.7894, swd: 3.8434, ept: 294.2388
      Epoch 14 composite train-obj: 0.725540
            No improvement (0.8259), counter 2/5
    Epoch [15/50], Train Losses: mse: 5.2731, mae: 1.0000, huber: 0.7213, swd: 3.5172, ept: 301.2270
    Epoch [15/50], Val Losses: mse: 6.4708, mae: 1.0976, huber: 0.8180, swd: 4.2950, ept: 294.2078
    Epoch [15/50], Test Losses: mse: 5.5858, mae: 1.0474, huber: 0.7728, swd: 3.7596, ept: 295.7670
      Epoch 15 composite train-obj: 0.721334
            Val objective improved 0.8190 → 0.8180, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 5.2626, mae: 0.9987, huber: 0.7204, swd: 3.5124, ept: 301.3245
    Epoch [16/50], Val Losses: mse: 6.2681, mae: 1.1088, huber: 0.8215, swd: 4.1481, ept: 296.0733
    Epoch [16/50], Test Losses: mse: 5.3989, mae: 1.0584, huber: 0.7725, swd: 3.6074, ept: 297.8786
      Epoch 16 composite train-obj: 0.720423
            No improvement (0.8215), counter 1/5
    Epoch [17/50], Train Losses: mse: 5.2800, mae: 1.0015, huber: 0.7234, swd: 3.5206, ept: 301.1823
    Epoch [17/50], Val Losses: mse: 6.1379, mae: 1.1231, huber: 0.8221, swd: 4.0893, ept: 295.7753
    Epoch [17/50], Test Losses: mse: 5.2604, mae: 1.0667, huber: 0.7669, swd: 3.5433, ept: 297.9757
      Epoch 17 composite train-obj: 0.723410
            No improvement (0.8221), counter 2/5
    Epoch [18/50], Train Losses: mse: 5.2521, mae: 0.9962, huber: 0.7188, swd: 3.5049, ept: 301.5234
    Epoch [18/50], Val Losses: mse: 6.3950, mae: 1.1040, huber: 0.8136, swd: 4.2398, ept: 295.1768
    Epoch [18/50], Test Losses: mse: 5.5037, mae: 1.0485, huber: 0.7648, swd: 3.6895, ept: 297.3605
      Epoch 18 composite train-obj: 0.718847
            Val objective improved 0.8180 → 0.8136, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 5.2576, mae: 0.9980, huber: 0.7199, swd: 3.5040, ept: 301.5130
    Epoch [19/50], Val Losses: mse: 6.2367, mae: 1.1041, huber: 0.8098, swd: 4.1723, ept: 295.7167
    Epoch [19/50], Test Losses: mse: 5.3621, mae: 1.0557, huber: 0.7617, swd: 3.6238, ept: 298.0574
      Epoch 19 composite train-obj: 0.719860
            Val objective improved 0.8136 → 0.8098, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 5.2548, mae: 0.9929, huber: 0.7164, swd: 3.5147, ept: 301.5892
    Epoch [20/50], Val Losses: mse: 6.4420, mae: 1.0866, huber: 0.8088, swd: 4.2446, ept: 294.5147
    Epoch [20/50], Test Losses: mse: 5.5482, mae: 1.0432, huber: 0.7656, swd: 3.6986, ept: 295.8918
      Epoch 20 composite train-obj: 0.716447
            Val objective improved 0.8098 → 0.8088, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 5.2390, mae: 0.9911, huber: 0.7150, swd: 3.5016, ept: 301.6544
    Epoch [21/50], Val Losses: mse: 6.2941, mae: 1.1079, huber: 0.8176, swd: 4.1863, ept: 295.1976
    Epoch [21/50], Test Losses: mse: 5.3997, mae: 1.0511, huber: 0.7660, swd: 3.6321, ept: 297.0401
      Epoch 21 composite train-obj: 0.714955
            No improvement (0.8176), counter 1/5
    Epoch [22/50], Train Losses: mse: 5.2441, mae: 0.9889, huber: 0.7135, swd: 3.5065, ept: 301.6568
    Epoch [22/50], Val Losses: mse: 6.2862, mae: 1.0891, huber: 0.8062, swd: 4.1614, ept: 295.4034
    Epoch [22/50], Test Losses: mse: 5.4019, mae: 1.0383, huber: 0.7575, swd: 3.6147, ept: 297.0168
      Epoch 22 composite train-obj: 0.713481
            Val objective improved 0.8088 → 0.8062, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 5.2309, mae: 0.9880, huber: 0.7120, swd: 3.4989, ept: 301.7657
    Epoch [23/50], Val Losses: mse: 6.3391, mae: 1.0806, huber: 0.8057, swd: 4.2217, ept: 295.8071
    Epoch [23/50], Test Losses: mse: 5.4522, mae: 1.0255, huber: 0.7564, swd: 3.6723, ept: 297.4995
      Epoch 23 composite train-obj: 0.711979
            Val objective improved 0.8062 → 0.8057, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 5.2367, mae: 0.9900, huber: 0.7147, swd: 3.4966, ept: 301.6450
    Epoch [24/50], Val Losses: mse: 6.0400, mae: 1.1138, huber: 0.8122, swd: 4.0811, ept: 297.0007
    Epoch [24/50], Test Losses: mse: 5.1730, mae: 1.0570, huber: 0.7555, swd: 3.5300, ept: 298.9356
      Epoch 24 composite train-obj: 0.714697
            No improvement (0.8122), counter 1/5
    Epoch [25/50], Train Losses: mse: 5.2273, mae: 0.9857, huber: 0.7105, swd: 3.4992, ept: 301.8852
    Epoch [25/50], Val Losses: mse: 6.4267, mae: 1.0700, huber: 0.8000, swd: 4.2814, ept: 295.5501
    Epoch [25/50], Test Losses: mse: 5.5544, mae: 1.0281, huber: 0.7594, swd: 3.7337, ept: 296.9259
      Epoch 25 composite train-obj: 0.710540
            Val objective improved 0.8057 → 0.8000, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 5.2382, mae: 0.9848, huber: 0.7108, swd: 3.5095, ept: 301.8073
    Epoch [26/50], Val Losses: mse: 6.0568, mae: 1.0920, huber: 0.8037, swd: 4.1041, ept: 297.0232
    Epoch [26/50], Test Losses: mse: 5.1879, mae: 1.0343, huber: 0.7483, swd: 3.5510, ept: 299.3027
      Epoch 26 composite train-obj: 0.710812
            No improvement (0.8037), counter 1/5
    Epoch [27/50], Train Losses: mse: 5.2237, mae: 0.9854, huber: 0.7104, swd: 3.5021, ept: 301.9495
    Epoch [27/50], Val Losses: mse: 6.1623, mae: 1.0891, huber: 0.8043, swd: 4.1032, ept: 296.0195
    Epoch [27/50], Test Losses: mse: 5.2942, mae: 1.0368, huber: 0.7540, swd: 3.5571, ept: 297.6939
      Epoch 27 composite train-obj: 0.710385
            No improvement (0.8043), counter 2/5
    Epoch [28/50], Train Losses: mse: 5.2281, mae: 0.9867, huber: 0.7111, swd: 3.5013, ept: 301.9364
    Epoch [28/50], Val Losses: mse: 6.4830, mae: 1.0804, huber: 0.8012, swd: 4.2406, ept: 295.2891
    Epoch [28/50], Test Losses: mse: 5.5852, mae: 1.0302, huber: 0.7572, swd: 3.6877, ept: 296.8477
      Epoch 28 composite train-obj: 0.711132
            No improvement (0.8012), counter 3/5
    Epoch [29/50], Train Losses: mse: 5.2256, mae: 0.9835, huber: 0.7094, swd: 3.4980, ept: 301.9370
    Epoch [29/50], Val Losses: mse: 6.1330, mae: 1.0873, huber: 0.8065, swd: 4.1091, ept: 296.6056
    Epoch [29/50], Test Losses: mse: 5.2625, mae: 1.0340, huber: 0.7534, swd: 3.5622, ept: 298.5783
      Epoch 29 composite train-obj: 0.709378
            No improvement (0.8065), counter 4/5
    Epoch [30/50], Train Losses: mse: 5.2192, mae: 0.9815, huber: 0.7081, swd: 3.4946, ept: 302.0914
    Epoch [30/50], Val Losses: mse: 6.1687, mae: 1.0809, huber: 0.7991, swd: 4.1648, ept: 296.2470
    Epoch [30/50], Test Losses: mse: 5.3164, mae: 1.0376, huber: 0.7560, swd: 3.6178, ept: 298.3729
      Epoch 30 composite train-obj: 0.708085
            Val objective improved 0.8000 → 0.7991, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 5.2226, mae: 0.9818, huber: 0.7079, swd: 3.5074, ept: 302.0583
    Epoch [31/50], Val Losses: mse: 6.1383, mae: 1.0892, huber: 0.8019, swd: 4.1327, ept: 296.1915
    Epoch [31/50], Test Losses: mse: 5.2642, mae: 1.0351, huber: 0.7491, swd: 3.5794, ept: 298.0684
      Epoch 31 composite train-obj: 0.707913
            No improvement (0.8019), counter 1/5
    Epoch [32/50], Train Losses: mse: 5.2095, mae: 0.9811, huber: 0.7073, swd: 3.4941, ept: 302.0892
    Epoch [32/50], Val Losses: mse: 6.2999, mae: 1.0802, huber: 0.8007, swd: 4.2123, ept: 296.0592
    Epoch [32/50], Test Losses: mse: 5.4092, mae: 1.0252, huber: 0.7501, swd: 3.6576, ept: 298.1983
      Epoch 32 composite train-obj: 0.707318
            No improvement (0.8007), counter 2/5
    Epoch [33/50], Train Losses: mse: 5.2185, mae: 0.9783, huber: 0.7055, swd: 3.5028, ept: 302.1058
    Epoch [33/50], Val Losses: mse: 6.1940, mae: 1.0767, huber: 0.7998, swd: 4.2163, ept: 296.9851
    Epoch [33/50], Test Losses: mse: 5.3326, mae: 1.0246, huber: 0.7511, swd: 3.6674, ept: 299.2336
      Epoch 33 composite train-obj: 0.705489
            No improvement (0.7998), counter 3/5
    Epoch [34/50], Train Losses: mse: 5.2045, mae: 0.9797, huber: 0.7062, swd: 3.4898, ept: 302.1783
    Epoch [34/50], Val Losses: mse: 6.2252, mae: 1.0788, huber: 0.7965, swd: 4.1560, ept: 296.2909
    Epoch [34/50], Test Losses: mse: 5.3480, mae: 1.0309, huber: 0.7484, swd: 3.6042, ept: 297.9471
      Epoch 34 composite train-obj: 0.706168
            Val objective improved 0.7991 → 0.7965, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 5.2112, mae: 0.9786, huber: 0.7055, swd: 3.4993, ept: 302.1178
    Epoch [35/50], Val Losses: mse: 6.6512, mae: 1.0827, huber: 0.8085, swd: 4.3782, ept: 295.2434
    Epoch [35/50], Test Losses: mse: 5.7503, mae: 1.0324, huber: 0.7679, swd: 3.8244, ept: 295.6788
      Epoch 35 composite train-obj: 0.705512
            No improvement (0.8085), counter 1/5
    Epoch [36/50], Train Losses: mse: 5.2319, mae: 0.9790, huber: 0.7064, swd: 3.5123, ept: 302.0181
    Epoch [36/50], Val Losses: mse: 6.2196, mae: 1.0959, huber: 0.8071, swd: 4.1904, ept: 296.5667
    Epoch [36/50], Test Losses: mse: 5.3519, mae: 1.0460, huber: 0.7590, swd: 3.6389, ept: 298.8164
      Epoch 36 composite train-obj: 0.706357
            No improvement (0.8071), counter 2/5
    Epoch [37/50], Train Losses: mse: 5.2003, mae: 0.9779, huber: 0.7052, swd: 3.4914, ept: 302.2181
    Epoch [37/50], Val Losses: mse: 6.3247, mae: 1.0918, huber: 0.8005, swd: 4.1798, ept: 296.1984
    Epoch [37/50], Test Losses: mse: 5.4485, mae: 1.0461, huber: 0.7554, swd: 3.6271, ept: 297.8803
      Epoch 37 composite train-obj: 0.705202
            No improvement (0.8005), counter 3/5
    Epoch [38/50], Train Losses: mse: 5.2126, mae: 0.9781, huber: 0.7051, swd: 3.4999, ept: 302.1541
    Epoch [38/50], Val Losses: mse: 6.6993, mae: 1.0676, huber: 0.8057, swd: 4.3537, ept: 294.5375
    Epoch [38/50], Test Losses: mse: 5.8035, mae: 1.0327, huber: 0.7705, swd: 3.8055, ept: 295.0200
      Epoch 38 composite train-obj: 0.705060
            No improvement (0.8057), counter 4/5
    Epoch [39/50], Train Losses: mse: 5.2076, mae: 0.9754, huber: 0.7035, swd: 3.4929, ept: 302.2028
    Epoch [39/50], Val Losses: mse: 6.2917, mae: 1.0758, huber: 0.7961, swd: 4.1909, ept: 296.2088
    Epoch [39/50], Test Losses: mse: 5.3982, mae: 1.0207, huber: 0.7454, swd: 3.6324, ept: 297.5357
      Epoch 39 composite train-obj: 0.703534
            Val objective improved 0.7965 → 0.7961, saving checkpoint.
    Epoch [40/50], Train Losses: mse: 5.1876, mae: 0.9744, huber: 0.7023, swd: 3.4833, ept: 302.3911
    Epoch [40/50], Val Losses: mse: 6.4005, mae: 1.0682, huber: 0.8005, swd: 4.2589, ept: 296.3133
    Epoch [40/50], Test Losses: mse: 5.5218, mae: 1.0198, huber: 0.7553, swd: 3.7091, ept: 297.4840
      Epoch 40 composite train-obj: 0.702346
            No improvement (0.8005), counter 1/5
    Epoch [41/50], Train Losses: mse: 5.1894, mae: 0.9726, huber: 0.7008, swd: 3.4847, ept: 302.2369
    Epoch [41/50], Val Losses: mse: 6.5817, mae: 1.0707, huber: 0.7992, swd: 4.3217, ept: 295.6285
    Epoch [41/50], Test Losses: mse: 5.7042, mae: 1.0296, huber: 0.7633, swd: 3.7775, ept: 297.0443
      Epoch 41 composite train-obj: 0.700844
            No improvement (0.7992), counter 2/5
    Epoch [42/50], Train Losses: mse: 5.1956, mae: 0.9735, huber: 0.7016, swd: 3.4889, ept: 302.4106
    Epoch [42/50], Val Losses: mse: 6.3723, mae: 1.0690, huber: 0.7963, swd: 4.2994, ept: 296.5958
    Epoch [42/50], Test Losses: mse: 5.5050, mae: 1.0260, huber: 0.7535, swd: 3.7462, ept: 298.0568
      Epoch 42 composite train-obj: 0.701587
            No improvement (0.7963), counter 3/5
    Epoch [43/50], Train Losses: mse: 5.2084, mae: 0.9747, huber: 0.7029, swd: 3.4992, ept: 302.3284
    Epoch [43/50], Val Losses: mse: 6.0705, mae: 1.0865, huber: 0.7958, swd: 4.1109, ept: 297.1670
    Epoch [43/50], Test Losses: mse: 5.2068, mae: 1.0364, huber: 0.7441, swd: 3.5553, ept: 299.5280
      Epoch 43 composite train-obj: 0.702936
            Val objective improved 0.7961 → 0.7958, saving checkpoint.
    Epoch [44/50], Train Losses: mse: 5.1871, mae: 0.9730, huber: 0.7009, swd: 3.4855, ept: 302.3411
    Epoch [44/50], Val Losses: mse: 6.2642, mae: 1.0789, huber: 0.7990, swd: 4.1728, ept: 296.6604
    Epoch [44/50], Test Losses: mse: 5.3864, mae: 1.0237, huber: 0.7503, swd: 3.6208, ept: 298.2990
      Epoch 44 composite train-obj: 0.700891
            No improvement (0.7990), counter 1/5
    Epoch [45/50], Train Losses: mse: 5.2010, mae: 0.9724, huber: 0.7011, swd: 3.4978, ept: 302.2567
    Epoch [45/50], Val Losses: mse: 6.0722, mae: 1.0873, huber: 0.7997, swd: 4.1638, ept: 297.3513
    Epoch [45/50], Test Losses: mse: 5.2253, mae: 1.0357, huber: 0.7500, swd: 3.6114, ept: 299.4793
      Epoch 45 composite train-obj: 0.701089
            No improvement (0.7997), counter 2/5
    Epoch [46/50], Train Losses: mse: 5.2014, mae: 0.9743, huber: 0.7022, swd: 3.4972, ept: 302.3956
    Epoch [46/50], Val Losses: mse: 6.2818, mae: 1.0637, huber: 0.7937, swd: 4.2723, ept: 296.6939
    Epoch [46/50], Test Losses: mse: 5.3992, mae: 1.0081, huber: 0.7439, swd: 3.7109, ept: 298.4891
      Epoch 46 composite train-obj: 0.702228
            Val objective improved 0.7958 → 0.7937, saving checkpoint.
    Epoch [47/50], Train Losses: mse: 5.2170, mae: 0.9732, huber: 0.7021, swd: 3.5067, ept: 302.3766
    Epoch [47/50], Val Losses: mse: 6.0432, mae: 1.0956, huber: 0.7994, swd: 4.0746, ept: 297.2389
    Epoch [47/50], Test Losses: mse: 5.1679, mae: 1.0329, huber: 0.7417, swd: 3.5143, ept: 299.1377
      Epoch 47 composite train-obj: 0.702095
            No improvement (0.7994), counter 1/5
    Epoch [48/50], Train Losses: mse: 5.1920, mae: 0.9730, huber: 0.7008, swd: 3.4895, ept: 302.4572
    Epoch [48/50], Val Losses: mse: 6.3576, mae: 1.0565, huber: 0.7879, swd: 4.2685, ept: 296.1997
    Epoch [48/50], Test Losses: mse: 5.4806, mae: 1.0088, huber: 0.7442, swd: 3.7183, ept: 297.8984
      Epoch 48 composite train-obj: 0.700761
            Val objective improved 0.7937 → 0.7879, saving checkpoint.
    Epoch [49/50], Train Losses: mse: 5.1866, mae: 0.9712, huber: 0.7000, swd: 3.4874, ept: 302.4587
    Epoch [49/50], Val Losses: mse: 6.5086, mae: 1.0912, huber: 0.8116, swd: 4.2067, ept: 295.8754
    Epoch [49/50], Test Losses: mse: 5.6245, mae: 1.0492, huber: 0.7716, swd: 3.6546, ept: 296.5543
      Epoch 49 composite train-obj: 0.700044
            No improvement (0.8116), counter 1/5
    Epoch [50/50], Train Losses: mse: 5.1975, mae: 0.9708, huber: 0.6998, swd: 3.4962, ept: 302.3735
    Epoch [50/50], Val Losses: mse: 6.2694, mae: 1.0637, huber: 0.7889, swd: 4.1560, ept: 297.0127
    Epoch [50/50], Test Losses: mse: 5.4068, mae: 1.0179, huber: 0.7450, swd: 3.6086, ept: 298.4724
      Epoch 50 composite train-obj: 0.699761
            No improvement (0.7889), counter 2/5
    Epoch [50/50], Test Losses: mse: 5.4806, mae: 1.0088, huber: 0.7442, swd: 3.7183, ept: 297.8984
    Best round's Test MSE: 5.4806, MAE: 1.0088, SWD: 3.7183
    Best round's Validation MSE: 6.3576, MAE: 1.0565, SWD: 4.2685
    Best round's Test verification MSE : 5.4806, MAE: 1.0088, SWD: 3.7183
    Time taken: 112.94 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.8832, mae: 1.2611, huber: 0.9514, swd: 4.4564, ept: 280.1434
    Epoch [1/50], Val Losses: mse: 7.4687, mae: 1.2469, huber: 0.9387, swd: 4.8728, ept: 284.6322
    Epoch [1/50], Test Losses: mse: 6.5762, mae: 1.2231, huber: 0.9136, swd: 4.3261, ept: 287.0180
      Epoch 1 composite train-obj: 0.951448
            Val objective improved inf → 0.9387, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.9547, mae: 1.1096, huber: 0.8084, swd: 3.9461, ept: 295.2072
    Epoch [2/50], Val Losses: mse: 6.9452, mae: 1.2027, huber: 0.8935, swd: 4.5658, ept: 289.2135
    Epoch [2/50], Test Losses: mse: 6.0652, mae: 1.1610, huber: 0.8563, swd: 4.0340, ept: 291.6233
      Epoch 2 composite train-obj: 0.808412
            Val objective improved 0.9387 → 0.8935, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.6897, mae: 1.0814, huber: 0.7842, swd: 3.7869, ept: 297.1572
    Epoch [3/50], Val Losses: mse: 6.8089, mae: 1.1803, huber: 0.8737, swd: 4.4696, ept: 290.4839
    Epoch [3/50], Test Losses: mse: 5.9235, mae: 1.1373, huber: 0.8339, swd: 3.9439, ept: 292.5902
      Epoch 3 composite train-obj: 0.784156
            Val objective improved 0.8935 → 0.8737, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.5566, mae: 1.0600, huber: 0.7679, swd: 3.7147, ept: 298.3230
    Epoch [4/50], Val Losses: mse: 6.6218, mae: 1.1668, huber: 0.8664, swd: 4.4012, ept: 291.6406
    Epoch [4/50], Test Losses: mse: 5.7368, mae: 1.1116, huber: 0.8175, swd: 3.8678, ept: 293.2952
      Epoch 4 composite train-obj: 0.767948
            Val objective improved 0.8737 → 0.8664, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.4809, mae: 1.0484, huber: 0.7590, swd: 3.6743, ept: 299.0066
    Epoch [5/50], Val Losses: mse: 6.5535, mae: 1.1659, huber: 0.8577, swd: 4.3845, ept: 292.0901
    Epoch [5/50], Test Losses: mse: 5.6810, mae: 1.1134, huber: 0.8096, swd: 3.8652, ept: 294.5653
      Epoch 5 composite train-obj: 0.758976
            Val objective improved 0.8664 → 0.8577, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.4364, mae: 1.0401, huber: 0.7524, swd: 3.6587, ept: 299.3645
    Epoch [6/50], Val Losses: mse: 6.6604, mae: 1.1401, huber: 0.8492, swd: 4.3696, ept: 292.4703
    Epoch [6/50], Test Losses: mse: 5.7694, mae: 1.0941, huber: 0.8059, swd: 3.8458, ept: 293.4772
      Epoch 6 composite train-obj: 0.752384
            Val objective improved 0.8577 → 0.8492, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.3974, mae: 1.0351, huber: 0.7489, swd: 3.6284, ept: 299.7785
    Epoch [7/50], Val Losses: mse: 6.4514, mae: 1.1318, huber: 0.8379, swd: 4.2783, ept: 294.0807
    Epoch [7/50], Test Losses: mse: 5.5867, mae: 1.0871, huber: 0.7944, swd: 3.7514, ept: 295.1729
      Epoch 7 composite train-obj: 0.748866
            Val objective improved 0.8492 → 0.8379, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.3524, mae: 1.0249, huber: 0.7402, swd: 3.6088, ept: 300.1412
    Epoch [8/50], Val Losses: mse: 6.2733, mae: 1.1262, huber: 0.8358, swd: 4.2769, ept: 294.2738
    Epoch [8/50], Test Losses: mse: 5.4134, mae: 1.0731, huber: 0.7834, swd: 3.7477, ept: 295.9205
      Epoch 8 composite train-obj: 0.740236
            Val objective improved 0.8379 → 0.8358, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 5.3392, mae: 1.0216, huber: 0.7382, swd: 3.6041, ept: 300.3313
    Epoch [9/50], Val Losses: mse: 6.4824, mae: 1.1096, huber: 0.8280, swd: 4.3563, ept: 294.5255
    Epoch [9/50], Test Losses: mse: 5.6145, mae: 1.0614, huber: 0.7838, swd: 3.8235, ept: 295.7455
      Epoch 9 composite train-obj: 0.738160
            Val objective improved 0.8358 → 0.8280, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 5.3197, mae: 1.0161, huber: 0.7340, swd: 3.5971, ept: 300.5131
    Epoch [10/50], Val Losses: mse: 6.5543, mae: 1.1109, huber: 0.8293, swd: 4.3783, ept: 292.5269
    Epoch [10/50], Test Losses: mse: 5.6741, mae: 1.0670, huber: 0.7889, swd: 3.8461, ept: 294.4819
      Epoch 10 composite train-obj: 0.734014
            No improvement (0.8293), counter 1/5
    Epoch [11/50], Train Losses: mse: 5.3037, mae: 1.0134, huber: 0.7319, swd: 3.5887, ept: 300.7429
    Epoch [11/50], Val Losses: mse: 6.4016, mae: 1.1053, huber: 0.8238, swd: 4.3399, ept: 293.8482
    Epoch [11/50], Test Losses: mse: 5.5221, mae: 1.0545, huber: 0.7759, swd: 3.8008, ept: 295.4152
      Epoch 11 composite train-obj: 0.731859
            Val objective improved 0.8280 → 0.8238, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 5.3023, mae: 1.0112, huber: 0.7304, swd: 3.5854, ept: 300.7992
    Epoch [12/50], Val Losses: mse: 6.0294, mae: 1.1323, huber: 0.8361, swd: 4.0961, ept: 295.3424
    Epoch [12/50], Test Losses: mse: 5.1744, mae: 1.0756, huber: 0.7795, swd: 3.5575, ept: 297.9898
      Epoch 12 composite train-obj: 0.730387
            No improvement (0.8361), counter 1/5
    Epoch [13/50], Train Losses: mse: 5.2918, mae: 1.0078, huber: 0.7276, swd: 3.5805, ept: 300.9171
    Epoch [13/50], Val Losses: mse: 6.1519, mae: 1.1111, huber: 0.8236, swd: 4.1917, ept: 295.3665
    Epoch [13/50], Test Losses: mse: 5.2792, mae: 1.0546, huber: 0.7690, swd: 3.6374, ept: 297.7847
      Epoch 13 composite train-obj: 0.727598
            Val objective improved 0.8238 → 0.8236, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 5.2766, mae: 1.0061, huber: 0.7269, swd: 3.5712, ept: 301.0754
    Epoch [14/50], Val Losses: mse: 6.4637, mae: 1.1142, huber: 0.8278, swd: 4.3185, ept: 295.2258
    Epoch [14/50], Test Losses: mse: 5.5713, mae: 1.0591, huber: 0.7787, swd: 3.7573, ept: 296.8780
      Epoch 14 composite train-obj: 0.726870
            No improvement (0.8278), counter 1/5
    Epoch [15/50], Train Losses: mse: 5.2673, mae: 1.0027, huber: 0.7241, swd: 3.5628, ept: 301.2265
    Epoch [15/50], Val Losses: mse: 6.2158, mae: 1.1137, huber: 0.8219, swd: 4.2640, ept: 295.3076
    Epoch [15/50], Test Losses: mse: 5.3679, mae: 1.0676, huber: 0.7755, swd: 3.7226, ept: 297.0651
      Epoch 15 composite train-obj: 0.724060
            Val objective improved 0.8236 → 0.8219, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 5.2583, mae: 0.9991, huber: 0.7207, swd: 3.5663, ept: 301.2676
    Epoch [16/50], Val Losses: mse: 6.3391, mae: 1.1017, huber: 0.8197, swd: 4.3524, ept: 295.4788
    Epoch [16/50], Test Losses: mse: 5.4725, mae: 1.0580, huber: 0.7741, swd: 3.7992, ept: 297.7772
      Epoch 16 composite train-obj: 0.720712
            Val objective improved 0.8219 → 0.8197, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 5.2715, mae: 0.9978, huber: 0.7203, swd: 3.5787, ept: 301.2010
    Epoch [17/50], Val Losses: mse: 6.1678, mae: 1.1032, huber: 0.8139, swd: 4.2318, ept: 296.0109
    Epoch [17/50], Test Losses: mse: 5.3039, mae: 1.0524, huber: 0.7622, swd: 3.6777, ept: 298.3928
      Epoch 17 composite train-obj: 0.720299
            Val objective improved 0.8197 → 0.8139, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 5.2462, mae: 0.9975, huber: 0.7195, swd: 3.5645, ept: 301.4755
    Epoch [18/50], Val Losses: mse: 6.3657, mae: 1.0944, huber: 0.8160, swd: 4.3303, ept: 296.1216
    Epoch [18/50], Test Losses: mse: 5.4900, mae: 1.0424, huber: 0.7675, swd: 3.7768, ept: 297.4877
      Epoch 18 composite train-obj: 0.719495
            No improvement (0.8160), counter 1/5
    Epoch [19/50], Train Losses: mse: 5.2407, mae: 0.9949, huber: 0.7182, swd: 3.5552, ept: 301.5121
    Epoch [19/50], Val Losses: mse: 6.3532, mae: 1.0937, huber: 0.8079, swd: 4.2797, ept: 295.1098
    Epoch [19/50], Test Losses: mse: 5.4722, mae: 1.0463, huber: 0.7633, swd: 3.7228, ept: 296.7046
      Epoch 19 composite train-obj: 0.718191
            Val objective improved 0.8139 → 0.8079, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 5.2549, mae: 0.9956, huber: 0.7186, swd: 3.5719, ept: 301.4646
    Epoch [20/50], Val Losses: mse: 6.1997, mae: 1.0949, huber: 0.8078, swd: 4.2333, ept: 296.3116
    Epoch [20/50], Test Losses: mse: 5.3298, mae: 1.0429, huber: 0.7576, swd: 3.6750, ept: 298.4526
      Epoch 20 composite train-obj: 0.718646
            Val objective improved 0.8079 → 0.8078, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 5.2431, mae: 0.9929, huber: 0.7165, swd: 3.5658, ept: 301.7928
    Epoch [21/50], Val Losses: mse: 6.5004, mae: 1.0952, huber: 0.8162, swd: 4.3823, ept: 294.9612
    Epoch [21/50], Test Losses: mse: 5.6007, mae: 1.0415, huber: 0.7692, swd: 3.8213, ept: 296.0899
      Epoch 21 composite train-obj: 0.716526
            No improvement (0.8162), counter 1/5
    Epoch [22/50], Train Losses: mse: 5.2442, mae: 0.9920, huber: 0.7159, swd: 3.5596, ept: 301.5895
    Epoch [22/50], Val Losses: mse: 6.3095, mae: 1.0938, huber: 0.8059, swd: 4.2182, ept: 295.3174
    Epoch [22/50], Test Losses: mse: 5.4356, mae: 1.0487, huber: 0.7622, swd: 3.6669, ept: 296.3881
      Epoch 22 composite train-obj: 0.715897
            Val objective improved 0.8078 → 0.8059, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 5.2304, mae: 0.9873, huber: 0.7119, swd: 3.5589, ept: 301.6734
    Epoch [23/50], Val Losses: mse: 6.2095, mae: 1.0859, huber: 0.8062, swd: 4.2093, ept: 295.9552
    Epoch [23/50], Test Losses: mse: 5.3393, mae: 1.0361, huber: 0.7573, swd: 3.6586, ept: 297.7244
      Epoch 23 composite train-obj: 0.711931
            No improvement (0.8062), counter 1/5
    Epoch [24/50], Train Losses: mse: 5.2358, mae: 0.9877, huber: 0.7125, swd: 3.5633, ept: 301.6985
    Epoch [24/50], Val Losses: mse: 6.0037, mae: 1.1220, huber: 0.8201, swd: 4.0865, ept: 297.0191
    Epoch [24/50], Test Losses: mse: 5.1410, mae: 1.0646, huber: 0.7631, swd: 3.5317, ept: 299.2931
      Epoch 24 composite train-obj: 0.712488
            No improvement (0.8201), counter 2/5
    Epoch [25/50], Train Losses: mse: 5.2394, mae: 0.9890, huber: 0.7139, swd: 3.5617, ept: 301.7636
    Epoch [25/50], Val Losses: mse: 6.2289, mae: 1.1153, huber: 0.8172, swd: 4.1948, ept: 295.6233
    Epoch [25/50], Test Losses: mse: 5.3411, mae: 1.0594, huber: 0.7646, swd: 3.6307, ept: 297.8215
      Epoch 25 composite train-obj: 0.713890
            No improvement (0.8172), counter 3/5
    Epoch [26/50], Train Losses: mse: 5.2311, mae: 0.9857, huber: 0.7108, swd: 3.5590, ept: 301.8215
    Epoch [26/50], Val Losses: mse: 6.2906, mae: 1.1040, huber: 0.8166, swd: 4.2220, ept: 294.4805
    Epoch [26/50], Test Losses: mse: 5.4148, mae: 1.0579, huber: 0.7713, swd: 3.6677, ept: 297.0882
      Epoch 26 composite train-obj: 0.710793
            No improvement (0.8166), counter 4/5
    Epoch [27/50], Train Losses: mse: 5.2178, mae: 0.9855, huber: 0.7105, swd: 3.5519, ept: 301.9357
    Epoch [27/50], Val Losses: mse: 6.4670, mae: 1.0788, huber: 0.8106, swd: 4.4041, ept: 295.4162
    Epoch [27/50], Test Losses: mse: 5.5810, mae: 1.0323, huber: 0.7672, swd: 3.8399, ept: 296.6152
      Epoch 27 composite train-obj: 0.710497
    Epoch [27/50], Test Losses: mse: 5.4356, mae: 1.0487, huber: 0.7622, swd: 3.6669, ept: 296.3881
    Best round's Test MSE: 5.4356, MAE: 1.0487, SWD: 3.6669
    Best round's Validation MSE: 6.3095, MAE: 1.0938, SWD: 4.2182
    Best round's Test verification MSE : 5.4356, MAE: 1.0487, SWD: 3.6669
    Time taken: 64.24 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 6.8850, mae: 1.2658, huber: 0.9556, swd: 4.4338, ept: 279.9133
    Epoch [1/50], Val Losses: mse: 7.2321, mae: 1.2451, huber: 0.9385, swd: 4.8160, ept: 285.1499
    Epoch [1/50], Test Losses: mse: 6.3577, mae: 1.2097, huber: 0.9056, swd: 4.2614, ept: 287.4886
      Epoch 1 composite train-obj: 0.955632
            Val objective improved inf → 0.9385, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 5.9316, mae: 1.1097, huber: 0.8086, swd: 3.9233, ept: 294.9570
    Epoch [2/50], Val Losses: mse: 7.0178, mae: 1.2020, huber: 0.8966, swd: 4.5699, ept: 287.4027
    Epoch [2/50], Test Losses: mse: 6.1470, mae: 1.1710, huber: 0.8664, swd: 4.0491, ept: 290.2471
      Epoch 2 composite train-obj: 0.808583
            Val objective improved 0.9385 → 0.8966, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 5.6973, mae: 1.0774, huber: 0.7817, swd: 3.7953, ept: 297.1692
    Epoch [3/50], Val Losses: mse: 7.1646, mae: 1.1911, huber: 0.8888, swd: 4.7167, ept: 290.0212
    Epoch [3/50], Test Losses: mse: 6.3031, mae: 1.1543, huber: 0.8579, swd: 4.1995, ept: 291.8365
      Epoch 3 composite train-obj: 0.781748
            Val objective improved 0.8966 → 0.8888, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 5.5648, mae: 1.0620, huber: 0.7688, swd: 3.7201, ept: 298.2875
    Epoch [4/50], Val Losses: mse: 6.5945, mae: 1.1693, huber: 0.8625, swd: 4.3924, ept: 291.9557
    Epoch [4/50], Test Losses: mse: 5.7390, mae: 1.1221, huber: 0.8195, swd: 3.8775, ept: 294.3652
      Epoch 4 composite train-obj: 0.768762
            Val objective improved 0.8888 → 0.8625, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 5.4763, mae: 1.0477, huber: 0.7586, swd: 3.6724, ept: 299.0297
    Epoch [5/50], Val Losses: mse: 6.4097, mae: 1.1471, huber: 0.8564, swd: 4.3546, ept: 292.4716
    Epoch [5/50], Test Losses: mse: 5.5526, mae: 1.0979, huber: 0.8070, swd: 3.8355, ept: 294.1465
      Epoch 5 composite train-obj: 0.758576
            Val objective improved 0.8625 → 0.8564, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 5.4043, mae: 1.0373, huber: 0.7500, swd: 3.6316, ept: 299.4726
    Epoch [6/50], Val Losses: mse: 6.7828, mae: 1.1436, huber: 0.8518, swd: 4.4740, ept: 291.5250
    Epoch [6/50], Test Losses: mse: 5.8874, mae: 1.0959, huber: 0.8104, swd: 3.9474, ept: 293.1262
      Epoch 6 composite train-obj: 0.749990
            Val objective improved 0.8564 → 0.8518, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 5.3789, mae: 1.0298, huber: 0.7443, swd: 3.6222, ept: 299.8701
    Epoch [7/50], Val Losses: mse: 6.3424, mae: 1.1225, huber: 0.8363, swd: 4.3180, ept: 294.1277
    Epoch [7/50], Test Losses: mse: 5.4859, mae: 1.0713, huber: 0.7872, swd: 3.7896, ept: 295.6366
      Epoch 7 composite train-obj: 0.744331
            Val objective improved 0.8518 → 0.8363, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 5.3607, mae: 1.0251, huber: 0.7407, swd: 3.6113, ept: 300.1381
    Epoch [8/50], Val Losses: mse: 6.8727, mae: 1.1164, huber: 0.8423, swd: 4.5293, ept: 290.9771
    Epoch [8/50], Test Losses: mse: 5.9762, mae: 1.0762, huber: 0.8069, swd: 4.0053, ept: 292.4234
      Epoch 8 composite train-obj: 0.740702
            No improvement (0.8423), counter 1/5
    Epoch [9/50], Train Losses: mse: 5.3479, mae: 1.0200, huber: 0.7377, swd: 3.6070, ept: 300.3279
    Epoch [9/50], Val Losses: mse: 6.4383, mae: 1.1186, huber: 0.8299, swd: 4.3666, ept: 294.3570
    Epoch [9/50], Test Losses: mse: 5.5796, mae: 1.0748, huber: 0.7864, swd: 3.8310, ept: 295.7623
      Epoch 9 composite train-obj: 0.737681
            Val objective improved 0.8363 → 0.8299, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 5.3332, mae: 1.0167, huber: 0.7347, swd: 3.6085, ept: 300.4065
    Epoch [10/50], Val Losses: mse: 6.5049, mae: 1.1104, huber: 0.8308, swd: 4.3864, ept: 293.5229
    Epoch [10/50], Test Losses: mse: 5.6302, mae: 1.0628, huber: 0.7861, swd: 3.8604, ept: 295.2341
      Epoch 10 composite train-obj: 0.734687
            No improvement (0.8308), counter 1/5
    Epoch [11/50], Train Losses: mse: 5.3192, mae: 1.0162, huber: 0.7337, swd: 3.5944, ept: 300.6623
    Epoch [11/50], Val Losses: mse: 6.2400, mae: 1.1147, huber: 0.8239, swd: 4.2520, ept: 295.1246
    Epoch [11/50], Test Losses: mse: 5.3654, mae: 1.0625, huber: 0.7712, swd: 3.7085, ept: 297.4307
      Epoch 11 composite train-obj: 0.733749
            Val objective improved 0.8299 → 0.8239, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 5.2952, mae: 1.0103, huber: 0.7297, swd: 3.5780, ept: 300.9297
    Epoch [12/50], Val Losses: mse: 6.2959, mae: 1.1091, huber: 0.8231, swd: 4.2387, ept: 294.4180
    Epoch [12/50], Test Losses: mse: 5.4245, mae: 1.0604, huber: 0.7747, swd: 3.7018, ept: 296.4367
      Epoch 12 composite train-obj: 0.729683
            Val objective improved 0.8239 → 0.8231, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 5.2776, mae: 1.0074, huber: 0.7268, swd: 3.5765, ept: 301.1391
    Epoch [13/50], Val Losses: mse: 6.6534, mae: 1.1138, huber: 0.8327, swd: 4.4392, ept: 293.8024
    Epoch [13/50], Test Losses: mse: 5.7505, mae: 1.0647, huber: 0.7891, swd: 3.8952, ept: 294.9210
      Epoch 13 composite train-obj: 0.726821
            No improvement (0.8327), counter 1/5
    Epoch [14/50], Train Losses: mse: 5.2839, mae: 1.0027, huber: 0.7244, swd: 3.5773, ept: 301.0375
    Epoch [14/50], Val Losses: mse: 6.4361, mae: 1.1144, huber: 0.8244, swd: 4.3171, ept: 294.4931
    Epoch [14/50], Test Losses: mse: 5.5381, mae: 1.0577, huber: 0.7745, swd: 3.7690, ept: 296.7679
      Epoch 14 composite train-obj: 0.724356
            No improvement (0.8244), counter 2/5
    Epoch [15/50], Train Losses: mse: 5.2771, mae: 1.0003, huber: 0.7219, swd: 3.5761, ept: 301.2885
    Epoch [15/50], Val Losses: mse: 6.3225, mae: 1.1126, huber: 0.8221, swd: 4.2033, ept: 295.0143
    Epoch [15/50], Test Losses: mse: 5.4372, mae: 1.0594, huber: 0.7727, swd: 3.6568, ept: 297.2663
      Epoch 15 composite train-obj: 0.721946
            Val objective improved 0.8231 → 0.8221, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 5.2621, mae: 1.0007, huber: 0.7224, swd: 3.5630, ept: 301.2610
    Epoch [16/50], Val Losses: mse: 6.2541, mae: 1.1118, huber: 0.8207, swd: 4.2767, ept: 295.5948
    Epoch [16/50], Test Losses: mse: 5.3918, mae: 1.0655, huber: 0.7731, swd: 3.7297, ept: 297.3486
      Epoch 16 composite train-obj: 0.722405
            Val objective improved 0.8221 → 0.8207, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 5.2550, mae: 1.0000, huber: 0.7214, swd: 3.5612, ept: 301.4278
    Epoch [17/50], Val Losses: mse: 6.3981, mae: 1.0942, huber: 0.8123, swd: 4.3199, ept: 294.9527
    Epoch [17/50], Test Losses: mse: 5.5223, mae: 1.0467, huber: 0.7691, swd: 3.7732, ept: 296.4959
      Epoch 17 composite train-obj: 0.721415
            Val objective improved 0.8207 → 0.8123, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 5.2635, mae: 0.9959, huber: 0.7187, swd: 3.5750, ept: 301.3965
    Epoch [18/50], Val Losses: mse: 6.4001, mae: 1.0885, huber: 0.8065, swd: 4.3415, ept: 295.5611
    Epoch [18/50], Test Losses: mse: 5.5248, mae: 1.0405, huber: 0.7619, swd: 3.7911, ept: 297.6829
      Epoch 18 composite train-obj: 0.718694
            Val objective improved 0.8123 → 0.8065, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 5.2703, mae: 0.9971, huber: 0.7194, swd: 3.5783, ept: 301.3906
    Epoch [19/50], Val Losses: mse: 6.2270, mae: 1.0932, huber: 0.8097, swd: 4.2295, ept: 295.5776
    Epoch [19/50], Test Losses: mse: 5.3544, mae: 1.0417, huber: 0.7608, swd: 3.6785, ept: 297.7687
      Epoch 19 composite train-obj: 0.719408
            No improvement (0.8097), counter 1/5
    Epoch [20/50], Train Losses: mse: 5.2540, mae: 0.9939, huber: 0.7168, swd: 3.5719, ept: 301.5916
    Epoch [20/50], Val Losses: mse: 6.4084, mae: 1.0885, huber: 0.8078, swd: 4.3204, ept: 295.1236
    Epoch [20/50], Test Losses: mse: 5.5103, mae: 1.0358, huber: 0.7604, swd: 3.7587, ept: 297.0849
      Epoch 20 composite train-obj: 0.716847
            No improvement (0.8078), counter 2/5
    Epoch [21/50], Train Losses: mse: 5.2363, mae: 0.9895, huber: 0.7137, swd: 3.5584, ept: 301.7313
    Epoch [21/50], Val Losses: mse: 6.7512, mae: 1.0975, huber: 0.8242, swd: 4.4982, ept: 293.3193
    Epoch [21/50], Test Losses: mse: 5.8565, mae: 1.0553, huber: 0.7882, swd: 3.9537, ept: 295.1151
      Epoch 21 composite train-obj: 0.713669
            No improvement (0.8242), counter 3/5
    Epoch [22/50], Train Losses: mse: 5.2392, mae: 0.9894, huber: 0.7138, swd: 3.5591, ept: 301.6988
    Epoch [22/50], Val Losses: mse: 6.4736, mae: 1.0864, huber: 0.8097, swd: 4.3572, ept: 295.1997
    Epoch [22/50], Test Losses: mse: 5.5889, mae: 1.0412, huber: 0.7671, swd: 3.8072, ept: 296.7792
      Epoch 22 composite train-obj: 0.713788
            No improvement (0.8097), counter 4/5
    Epoch [23/50], Train Losses: mse: 5.2498, mae: 0.9880, huber: 0.7135, swd: 3.5692, ept: 301.7169
    Epoch [23/50], Val Losses: mse: 6.1805, mae: 1.0857, huber: 0.8054, swd: 4.2225, ept: 296.3351
    Epoch [23/50], Test Losses: mse: 5.3112, mae: 1.0341, huber: 0.7533, swd: 3.6718, ept: 298.2197
      Epoch 23 composite train-obj: 0.713454
            Val objective improved 0.8065 → 0.8054, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 5.2395, mae: 0.9875, huber: 0.7126, swd: 3.5634, ept: 301.7804
    Epoch [24/50], Val Losses: mse: 6.3159, mae: 1.1008, huber: 0.8128, swd: 4.2344, ept: 296.2840
    Epoch [24/50], Test Losses: mse: 5.4274, mae: 1.0464, huber: 0.7631, swd: 3.6724, ept: 297.6536
      Epoch 24 composite train-obj: 0.712557
            No improvement (0.8128), counter 1/5
    Epoch [25/50], Train Losses: mse: 5.2334, mae: 0.9872, huber: 0.7121, swd: 3.5572, ept: 301.7780
    Epoch [25/50], Val Losses: mse: 6.1606, mae: 1.1040, huber: 0.8108, swd: 4.1974, ept: 296.6140
    Epoch [25/50], Test Losses: mse: 5.2770, mae: 1.0402, huber: 0.7548, swd: 3.6355, ept: 298.4296
      Epoch 25 composite train-obj: 0.712098
            No improvement (0.8108), counter 2/5
    Epoch [26/50], Train Losses: mse: 5.2223, mae: 0.9853, huber: 0.7107, swd: 3.5499, ept: 302.0118
    Epoch [26/50], Val Losses: mse: 6.1381, mae: 1.0976, huber: 0.8064, swd: 4.2062, ept: 296.5929
    Epoch [26/50], Test Losses: mse: 5.2606, mae: 1.0401, huber: 0.7525, swd: 3.6396, ept: 298.6419
      Epoch 26 composite train-obj: 0.710675
            No improvement (0.8064), counter 3/5
    Epoch [27/50], Train Losses: mse: 5.2078, mae: 0.9834, huber: 0.7088, swd: 3.5437, ept: 301.8460
    Epoch [27/50], Val Losses: mse: 6.2657, mae: 1.0849, huber: 0.8090, swd: 4.2846, ept: 296.0140
    Epoch [27/50], Test Losses: mse: 5.4005, mae: 1.0381, huber: 0.7632, swd: 3.7306, ept: 297.7030
      Epoch 27 composite train-obj: 0.708819
            No improvement (0.8090), counter 4/5
    Epoch [28/50], Train Losses: mse: 5.2290, mae: 0.9826, huber: 0.7089, swd: 3.5608, ept: 301.9698
    Epoch [28/50], Val Losses: mse: 6.1707, mae: 1.0895, huber: 0.8018, swd: 4.2372, ept: 297.0925
    Epoch [28/50], Test Losses: mse: 5.3130, mae: 1.0424, huber: 0.7539, swd: 3.6784, ept: 299.1646
      Epoch 28 composite train-obj: 0.708933
            Val objective improved 0.8054 → 0.8018, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 5.2355, mae: 0.9836, huber: 0.7098, swd: 3.5676, ept: 301.8446
    Epoch [29/50], Val Losses: mse: 6.2304, mae: 1.1004, huber: 0.8064, swd: 4.2509, ept: 296.0800
    Epoch [29/50], Test Losses: mse: 5.3563, mae: 1.0505, huber: 0.7576, swd: 3.6930, ept: 298.0329
      Epoch 29 composite train-obj: 0.709785
            No improvement (0.8064), counter 1/5
    Epoch [30/50], Train Losses: mse: 5.2192, mae: 0.9834, huber: 0.7093, swd: 3.5532, ept: 301.9525
    Epoch [30/50], Val Losses: mse: 6.2204, mae: 1.0829, huber: 0.8000, swd: 4.2718, ept: 296.8043
    Epoch [30/50], Test Losses: mse: 5.3571, mae: 1.0360, huber: 0.7523, swd: 3.7158, ept: 298.9141
      Epoch 30 composite train-obj: 0.709309
            Val objective improved 0.8018 → 0.8000, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 5.2102, mae: 0.9788, huber: 0.7059, swd: 3.5469, ept: 302.0155
    Epoch [31/50], Val Losses: mse: 6.3188, mae: 1.0751, huber: 0.7944, swd: 4.2836, ept: 295.8322
    Epoch [31/50], Test Losses: mse: 5.4393, mae: 1.0248, huber: 0.7494, swd: 3.7292, ept: 297.9796
      Epoch 31 composite train-obj: 0.705894
            Val objective improved 0.8000 → 0.7944, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 5.2110, mae: 0.9786, huber: 0.7056, swd: 3.5496, ept: 302.2003
    Epoch [32/50], Val Losses: mse: 6.4435, mae: 1.0702, huber: 0.7997, swd: 4.3804, ept: 296.3635
    Epoch [32/50], Test Losses: mse: 5.5720, mae: 1.0236, huber: 0.7568, swd: 3.8284, ept: 297.6837
      Epoch 32 composite train-obj: 0.705570
            No improvement (0.7997), counter 1/5
    Epoch [33/50], Train Losses: mse: 5.2263, mae: 0.9801, huber: 0.7071, swd: 3.5607, ept: 302.0963
    Epoch [33/50], Val Losses: mse: 6.1343, mae: 1.0728, huber: 0.7944, swd: 4.2207, ept: 297.4400
    Epoch [33/50], Test Losses: mse: 5.2720, mae: 1.0241, huber: 0.7443, swd: 3.6641, ept: 299.1448
      Epoch 33 composite train-obj: 0.707098
            No improvement (0.7944), counter 2/5
    Epoch [34/50], Train Losses: mse: 5.2148, mae: 0.9792, huber: 0.7059, swd: 3.5523, ept: 302.0980
    Epoch [34/50], Val Losses: mse: 6.0469, mae: 1.0807, huber: 0.7985, swd: 4.1896, ept: 297.1142
    Epoch [34/50], Test Losses: mse: 5.1845, mae: 1.0247, huber: 0.7437, swd: 3.6308, ept: 299.2683
      Epoch 34 composite train-obj: 0.705865
            No improvement (0.7985), counter 3/5
    Epoch [35/50], Train Losses: mse: 5.2071, mae: 0.9779, huber: 0.7049, swd: 3.5491, ept: 302.1341
    Epoch [35/50], Val Losses: mse: 6.1010, mae: 1.0894, huber: 0.7988, swd: 4.1883, ept: 296.4331
    Epoch [35/50], Test Losses: mse: 5.2300, mae: 1.0317, huber: 0.7453, swd: 3.6273, ept: 298.4263
      Epoch 35 composite train-obj: 0.704918
            No improvement (0.7988), counter 4/5
    Epoch [36/50], Train Losses: mse: 5.2108, mae: 0.9791, huber: 0.7056, swd: 3.5561, ept: 302.1788
    Epoch [36/50], Val Losses: mse: 6.6041, mae: 1.0843, huber: 0.8136, swd: 4.4830, ept: 295.3524
    Epoch [36/50], Test Losses: mse: 5.7196, mae: 1.0420, huber: 0.7746, swd: 3.9241, ept: 296.2244
      Epoch 36 composite train-obj: 0.705554
    Epoch [36/50], Test Losses: mse: 5.4393, mae: 1.0248, huber: 0.7494, swd: 3.7292, ept: 297.9796
    Best round's Test MSE: 5.4393, MAE: 1.0248, SWD: 3.7292
    Best round's Validation MSE: 6.3188, MAE: 1.0751, SWD: 4.2836
    Best round's Test verification MSE : 5.4393, MAE: 1.0248, SWD: 3.7292
    Time taken: 78.81 seconds
    
    ==================================================
    Experiment Summary (DLinear_rossler_seq336_pred336_20250511_0354)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 5.4518 ± 0.0204
      mae: 1.0274 ± 0.0164
      huber: 0.7519 ± 0.0076
      swd: 3.7048 ± 0.0272
      ept: 297.4221 ± 0.7319
      count: 36.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 6.3286 ± 0.0208
      mae: 1.0751 ± 0.0152
      huber: 0.7961 ± 0.0074
      swd: 4.2568 ± 0.0280
      ept: 295.7831 ± 0.3619
      count: 36.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 256.01 seconds
    
    Experiment complete: DLinear_rossler_seq336_pred336_20250511_0354
    Model: DLinear
    Dataset: rossler
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
    Train set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([720, 3])
    Test set data shapes: torch.Size([10400, 3]), torch.Size([10400, 3])
    Number of batches in train_loader: 277
    Batch 0: Data shape torch.Size([128, 336, 3]), Target shape torch.Size([128, 720, 3])
    
    ==================================================
    Data Preparation: rossler
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
    
    Epoch [1/50], Train Losses: mse: 8.1739, mae: 1.6156, huber: 1.2781, swd: 4.6006, ept: 501.3118
    Epoch [1/50], Val Losses: mse: 9.1567, mae: 1.6796, huber: 1.3320, swd: 5.1824, ept: 522.4643
    Epoch [1/50], Test Losses: mse: 7.2619, mae: 1.4755, huber: 1.1378, swd: 4.0831, ept: 538.9969
      Epoch 1 composite train-obj: 1.278082
            Val objective improved inf → 1.3320, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.4476, mae: 1.4979, huber: 1.1608, swd: 4.2127, ept: 544.3092
    Epoch [2/50], Val Losses: mse: 8.5563, mae: 1.6256, huber: 1.2756, swd: 4.9368, ept: 520.6684
    Epoch [2/50], Test Losses: mse: 6.8558, mae: 1.4348, huber: 1.0961, swd: 3.8998, ept: 542.5038
      Epoch 2 composite train-obj: 1.160844
            Val objective improved 1.3320 → 1.2756, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.2591, mae: 1.4723, huber: 1.1368, swd: 4.1261, ept: 550.4202
    Epoch [3/50], Val Losses: mse: 8.5456, mae: 1.6013, huber: 1.2632, swd: 4.9326, ept: 527.8710
    Epoch [3/50], Test Losses: mse: 6.8035, mae: 1.4108, huber: 1.0808, swd: 3.9089, ept: 549.2508
      Epoch 3 composite train-obj: 1.136842
            Val objective improved 1.2756 → 1.2632, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.1621, mae: 1.4569, huber: 1.1236, swd: 4.0857, ept: 554.1359
    Epoch [4/50], Val Losses: mse: 8.4327, mae: 1.6020, huber: 1.2589, swd: 4.9087, ept: 528.8190
    Epoch [4/50], Test Losses: mse: 6.6440, mae: 1.3979, huber: 1.0661, swd: 3.8588, ept: 550.5104
      Epoch 4 composite train-obj: 1.123638
            Val objective improved 1.2632 → 1.2589, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.1165, mae: 1.4450, huber: 1.1142, swd: 4.0626, ept: 556.5025
    Epoch [5/50], Val Losses: mse: 8.3913, mae: 1.5818, huber: 1.2464, swd: 4.8661, ept: 533.6816
    Epoch [5/50], Test Losses: mse: 6.6380, mae: 1.3837, huber: 1.0564, swd: 3.8416, ept: 553.1084
      Epoch 5 composite train-obj: 1.114169
            Val objective improved 1.2589 → 1.2464, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.0604, mae: 1.4382, huber: 1.1083, swd: 4.0354, ept: 557.4150
    Epoch [6/50], Val Losses: mse: 8.5023, mae: 1.5764, huber: 1.2400, swd: 4.8980, ept: 539.0629
    Epoch [6/50], Test Losses: mse: 6.7130, mae: 1.3758, huber: 1.0507, swd: 3.8637, ept: 556.8840
      Epoch 6 composite train-obj: 1.108287
            Val objective improved 1.2464 → 1.2400, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.0472, mae: 1.4330, huber: 1.1037, swd: 4.0302, ept: 558.9053
    Epoch [7/50], Val Losses: mse: 8.4357, mae: 1.5641, huber: 1.2306, swd: 4.8328, ept: 534.7042
    Epoch [7/50], Test Losses: mse: 6.7181, mae: 1.3846, huber: 1.0580, swd: 3.8217, ept: 555.3645
      Epoch 7 composite train-obj: 1.103736
            Val objective improved 1.2400 → 1.2306, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 7.0411, mae: 1.4301, huber: 1.1013, swd: 4.0214, ept: 559.1545
    Epoch [8/50], Val Losses: mse: 8.2005, mae: 1.5814, huber: 1.2372, swd: 4.8133, ept: 541.9396
    Epoch [8/50], Test Losses: mse: 6.4638, mae: 1.3793, huber: 1.0436, swd: 3.7942, ept: 561.3410
      Epoch 8 composite train-obj: 1.101279
            No improvement (1.2372), counter 1/5
    Epoch [9/50], Train Losses: mse: 7.0173, mae: 1.4260, huber: 1.0981, swd: 4.0141, ept: 559.5960
    Epoch [9/50], Val Losses: mse: 8.2415, mae: 1.5643, huber: 1.2301, swd: 4.8473, ept: 532.3745
    Epoch [9/50], Test Losses: mse: 6.5148, mae: 1.3682, huber: 1.0413, swd: 3.8369, ept: 556.2258
      Epoch 9 composite train-obj: 1.098084
            Val objective improved 1.2306 → 1.2301, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.0073, mae: 1.4239, huber: 1.0962, swd: 4.0055, ept: 559.6784
    Epoch [10/50], Val Losses: mse: 8.7451, mae: 1.5801, huber: 1.2481, swd: 4.9606, ept: 543.7495
    Epoch [10/50], Test Losses: mse: 6.8603, mae: 1.3823, huber: 1.0582, swd: 3.9245, ept: 561.1930
      Epoch 10 composite train-obj: 1.096170
            No improvement (1.2481), counter 1/5
    Epoch [11/50], Train Losses: mse: 6.9940, mae: 1.4197, huber: 1.0930, swd: 3.9971, ept: 560.2706
    Epoch [11/50], Val Losses: mse: 8.4896, mae: 1.5678, huber: 1.2369, swd: 4.8974, ept: 539.1709
    Epoch [11/50], Test Losses: mse: 6.6964, mae: 1.3636, huber: 1.0451, swd: 3.8802, ept: 555.0699
      Epoch 11 composite train-obj: 1.092970
            No improvement (1.2369), counter 2/5
    Epoch [12/50], Train Losses: mse: 6.9876, mae: 1.4190, huber: 1.0922, swd: 3.9976, ept: 560.4024
    Epoch [12/50], Val Losses: mse: 8.4418, mae: 1.5604, huber: 1.2273, swd: 4.8668, ept: 542.1044
    Epoch [12/50], Test Losses: mse: 6.6763, mae: 1.3684, huber: 1.0427, swd: 3.8529, ept: 559.7122
      Epoch 12 composite train-obj: 1.092246
            Val objective improved 1.2301 → 1.2273, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 6.9867, mae: 1.4155, huber: 1.0895, swd: 3.9987, ept: 561.2004
    Epoch [13/50], Val Losses: mse: 8.1945, mae: 1.5379, huber: 1.2076, swd: 4.7852, ept: 530.8356
    Epoch [13/50], Test Losses: mse: 6.5726, mae: 1.3706, huber: 1.0459, swd: 3.8090, ept: 556.6187
      Epoch 13 composite train-obj: 1.089469
            Val objective improved 1.2273 → 1.2076, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 6.9762, mae: 1.4149, huber: 1.0891, swd: 3.9919, ept: 560.9886
    Epoch [14/50], Val Losses: mse: 8.4200, mae: 1.5493, huber: 1.2189, swd: 4.8410, ept: 546.0536
    Epoch [14/50], Test Losses: mse: 6.6589, mae: 1.3570, huber: 1.0362, swd: 3.8378, ept: 561.8876
      Epoch 14 composite train-obj: 1.089084
            No improvement (1.2189), counter 1/5
    Epoch [15/50], Train Losses: mse: 6.9832, mae: 1.4139, huber: 1.0884, swd: 3.9968, ept: 560.8002
    Epoch [15/50], Val Losses: mse: 8.5838, mae: 1.5461, huber: 1.2222, swd: 4.9067, ept: 539.5044
    Epoch [15/50], Test Losses: mse: 6.8003, mae: 1.3567, huber: 1.0433, swd: 3.8912, ept: 556.5039
      Epoch 15 composite train-obj: 1.088413
            No improvement (1.2222), counter 2/5
    Epoch [16/50], Train Losses: mse: 6.9677, mae: 1.4117, huber: 1.0867, swd: 3.9866, ept: 561.2605
    Epoch [16/50], Val Losses: mse: 8.3504, mae: 1.5592, huber: 1.2255, swd: 4.8206, ept: 538.3560
    Epoch [16/50], Test Losses: mse: 6.5863, mae: 1.3604, huber: 1.0355, swd: 3.8046, ept: 556.5307
      Epoch 16 composite train-obj: 1.086733
            No improvement (1.2255), counter 3/5
    Epoch [17/50], Train Losses: mse: 6.9627, mae: 1.4102, huber: 1.0852, swd: 3.9861, ept: 561.6143
    Epoch [17/50], Val Losses: mse: 8.3287, mae: 1.5444, huber: 1.2135, swd: 4.8170, ept: 539.8275
    Epoch [17/50], Test Losses: mse: 6.6068, mae: 1.3539, huber: 1.0329, swd: 3.8153, ept: 558.3081
      Epoch 17 composite train-obj: 1.085229
            No improvement (1.2135), counter 4/5
    Epoch [18/50], Train Losses: mse: 6.9650, mae: 1.4084, huber: 1.0836, swd: 3.9913, ept: 562.1198
    Epoch [18/50], Val Losses: mse: 8.2374, mae: 1.5424, huber: 1.2097, swd: 4.7608, ept: 534.6112
    Epoch [18/50], Test Losses: mse: 6.5724, mae: 1.3628, huber: 1.0375, swd: 3.7763, ept: 554.7777
      Epoch 18 composite train-obj: 1.083577
    Epoch [18/50], Test Losses: mse: 6.5726, mae: 1.3706, huber: 1.0459, swd: 3.8090, ept: 556.6187
    Best round's Test MSE: 6.5726, MAE: 1.3706, SWD: 3.8090
    Best round's Validation MSE: 8.1945, MAE: 1.5379, SWD: 4.7852
    Best round's Test verification MSE : 6.5726, MAE: 1.3706, SWD: 3.8090
    Time taken: 42.37 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.1802, mae: 1.6153, huber: 1.2778, swd: 4.4806, ept: 500.3243
    Epoch [1/50], Val Losses: mse: 8.6368, mae: 1.6676, huber: 1.3172, swd: 5.0334, ept: 502.9313
    Epoch [1/50], Test Losses: mse: 6.9782, mae: 1.4904, huber: 1.1463, swd: 3.9981, ept: 532.0353
      Epoch 1 composite train-obj: 1.277839
            Val objective improved inf → 1.3172, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.4374, mae: 1.4983, huber: 1.1607, swd: 4.1336, ept: 543.7305
    Epoch [2/50], Val Losses: mse: 8.5326, mae: 1.6228, huber: 1.2798, swd: 4.9210, ept: 514.0852
    Epoch [2/50], Test Losses: mse: 6.8329, mae: 1.4331, huber: 1.1003, swd: 3.9081, ept: 540.9411
      Epoch 2 composite train-obj: 1.160749
            Val objective improved 1.3172 → 1.2798, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.2728, mae: 1.4713, huber: 1.1364, swd: 4.0758, ept: 551.2274
    Epoch [3/50], Val Losses: mse: 8.4081, mae: 1.6033, huber: 1.2603, swd: 4.7984, ept: 520.2394
    Epoch [3/50], Test Losses: mse: 6.7513, mae: 1.4231, huber: 1.0874, swd: 3.7834, ept: 543.1448
      Epoch 3 composite train-obj: 1.136399
            Val objective improved 1.2798 → 1.2603, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.1531, mae: 1.4571, huber: 1.1233, swd: 4.0195, ept: 554.7333
    Epoch [4/50], Val Losses: mse: 8.6527, mae: 1.5958, huber: 1.2587, swd: 4.8882, ept: 536.3168
    Epoch [4/50], Test Losses: mse: 6.8327, mae: 1.3939, huber: 1.0685, swd: 3.8573, ept: 553.6097
      Epoch 4 composite train-obj: 1.123256
            Val objective improved 1.2603 → 1.2587, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 7.1139, mae: 1.4462, huber: 1.1149, swd: 4.0067, ept: 556.7057
    Epoch [5/50], Val Losses: mse: 8.3203, mae: 1.5982, huber: 1.2540, swd: 4.8240, ept: 532.3876
    Epoch [5/50], Test Losses: mse: 6.5456, mae: 1.3907, huber: 1.0570, swd: 3.8032, ept: 556.0096
      Epoch 5 composite train-obj: 1.114871
            Val objective improved 1.2587 → 1.2540, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 7.0662, mae: 1.4387, huber: 1.1085, swd: 3.9810, ept: 557.8651
    Epoch [6/50], Val Losses: mse: 8.5668, mae: 1.5868, huber: 1.2549, swd: 4.8822, ept: 538.5384
    Epoch [6/50], Test Losses: mse: 6.7942, mae: 1.3969, huber: 1.0707, swd: 3.9052, ept: 558.8292
      Epoch 6 composite train-obj: 1.108465
            No improvement (1.2549), counter 1/5
    Epoch [7/50], Train Losses: mse: 7.0556, mae: 1.4336, huber: 1.1049, swd: 3.9857, ept: 557.9467
    Epoch [7/50], Val Losses: mse: 8.8391, mae: 1.5735, huber: 1.2451, swd: 4.9217, ept: 536.6514
    Epoch [7/50], Test Losses: mse: 7.0431, mae: 1.3982, huber: 1.0772, swd: 3.9182, ept: 547.8458
      Epoch 7 composite train-obj: 1.104909
            Val objective improved 1.2540 → 1.2451, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 7.0529, mae: 1.4319, huber: 1.1033, swd: 3.9758, ept: 558.6184
    Epoch [8/50], Val Losses: mse: 8.2629, mae: 1.5788, huber: 1.2430, swd: 4.7694, ept: 531.5448
    Epoch [8/50], Test Losses: mse: 6.5109, mae: 1.3779, huber: 1.0505, swd: 3.7811, ept: 554.2915
      Epoch 8 composite train-obj: 1.103268
            Val objective improved 1.2451 → 1.2430, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 7.0204, mae: 1.4265, huber: 1.0985, swd: 3.9623, ept: 559.1604
    Epoch [9/50], Val Losses: mse: 8.1829, mae: 1.5560, huber: 1.2213, swd: 4.7467, ept: 530.2365
    Epoch [9/50], Test Losses: mse: 6.5028, mae: 1.3710, huber: 1.0428, swd: 3.7743, ept: 554.6060
      Epoch 9 composite train-obj: 1.098489
            Val objective improved 1.2430 → 1.2213, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 6.9961, mae: 1.4218, huber: 1.0945, swd: 3.9541, ept: 560.4163
    Epoch [10/50], Val Losses: mse: 8.5357, mae: 1.5733, huber: 1.2414, swd: 4.8361, ept: 544.5663
    Epoch [10/50], Test Losses: mse: 6.6954, mae: 1.3643, huber: 1.0443, swd: 3.8331, ept: 561.6296
      Epoch 10 composite train-obj: 1.094454
            No improvement (1.2414), counter 1/5
    Epoch [11/50], Train Losses: mse: 7.0003, mae: 1.4204, huber: 1.0935, swd: 3.9569, ept: 560.4244
    Epoch [11/50], Val Losses: mse: 8.1782, mae: 1.5703, huber: 1.2324, swd: 4.7399, ept: 543.7324
    Epoch [11/50], Test Losses: mse: 6.4194, mae: 1.3638, huber: 1.0358, swd: 3.7476, ept: 564.6375
      Epoch 11 composite train-obj: 1.093468
            No improvement (1.2324), counter 2/5
    Epoch [12/50], Train Losses: mse: 6.9829, mae: 1.4165, huber: 1.0905, swd: 3.9518, ept: 561.0012
    Epoch [12/50], Val Losses: mse: 8.1593, mae: 1.5674, huber: 1.2307, swd: 4.7659, ept: 536.5234
    Epoch [12/50], Test Losses: mse: 6.4724, mae: 1.3756, huber: 1.0470, swd: 3.7965, ept: 559.9727
      Epoch 12 composite train-obj: 1.090466
            No improvement (1.2307), counter 3/5
    Epoch [13/50], Train Losses: mse: 6.9937, mae: 1.4172, huber: 1.0910, swd: 3.9568, ept: 560.8972
    Epoch [13/50], Val Losses: mse: 8.1642, mae: 1.5687, huber: 1.2317, swd: 4.7673, ept: 529.7338
    Epoch [13/50], Test Losses: mse: 6.4596, mae: 1.3740, huber: 1.0443, swd: 3.7849, ept: 555.6540
      Epoch 13 composite train-obj: 1.090990
            No improvement (1.2317), counter 4/5
    Epoch [14/50], Train Losses: mse: 6.9733, mae: 1.4146, huber: 1.0888, swd: 3.9451, ept: 561.3779
    Epoch [14/50], Val Losses: mse: 8.3519, mae: 1.5670, huber: 1.2336, swd: 4.7859, ept: 538.7708
    Epoch [14/50], Test Losses: mse: 6.5469, mae: 1.3585, huber: 1.0354, swd: 3.7899, ept: 560.1215
      Epoch 14 composite train-obj: 1.088753
    Epoch [14/50], Test Losses: mse: 6.5028, mae: 1.3710, huber: 1.0428, swd: 3.7743, ept: 554.6060
    Best round's Test MSE: 6.5028, MAE: 1.3710, SWD: 3.7743
    Best round's Validation MSE: 8.1829, MAE: 1.5560, SWD: 4.7467
    Best round's Test verification MSE : 6.5028, MAE: 1.3710, SWD: 3.7743
    Time taken: 33.87 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 8.1525, mae: 1.6110, huber: 1.2739, swd: 4.6281, ept: 501.5553
    Epoch [1/50], Val Losses: mse: 8.9721, mae: 1.6505, huber: 1.3105, swd: 5.2292, ept: 516.4947
    Epoch [1/50], Test Losses: mse: 7.2014, mae: 1.4627, huber: 1.1309, swd: 4.1272, ept: 532.8040
      Epoch 1 composite train-obj: 1.273905
            Val objective improved inf → 1.3105, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 7.4465, mae: 1.4993, huber: 1.1620, swd: 4.2652, ept: 544.0293
    Epoch [2/50], Val Losses: mse: 8.5221, mae: 1.6501, huber: 1.2995, swd: 5.0514, ept: 524.6953
    Epoch [2/50], Test Losses: mse: 6.7276, mae: 1.4429, huber: 1.0997, swd: 3.9766, ept: 548.3361
      Epoch 2 composite train-obj: 1.161951
            Val objective improved 1.3105 → 1.2995, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 7.2558, mae: 1.4714, huber: 1.1364, swd: 4.1826, ept: 550.5151
    Epoch [3/50], Val Losses: mse: 8.6693, mae: 1.6001, huber: 1.2625, swd: 5.0612, ept: 532.4063
    Epoch [3/50], Test Losses: mse: 6.8792, mae: 1.4032, huber: 1.0772, swd: 3.9942, ept: 550.7997
      Epoch 3 composite train-obj: 1.136389
            Val objective improved 1.2995 → 1.2625, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 7.1687, mae: 1.4557, huber: 1.1227, swd: 4.1477, ept: 553.9308
    Epoch [4/50], Val Losses: mse: 8.3069, mae: 1.6105, huber: 1.2635, swd: 4.9865, ept: 529.2263
    Epoch [4/50], Test Losses: mse: 6.5600, mae: 1.4116, huber: 1.0726, swd: 3.9305, ept: 553.7201
      Epoch 4 composite train-obj: 1.122724
            No improvement (1.2635), counter 1/5
    Epoch [5/50], Train Losses: mse: 7.1102, mae: 1.4458, huber: 1.1144, swd: 4.1192, ept: 556.3039
    Epoch [5/50], Val Losses: mse: 8.1236, mae: 1.6137, huber: 1.2675, swd: 4.9754, ept: 527.3437
    Epoch [5/50], Test Losses: mse: 6.5028, mae: 1.4219, huber: 1.0858, swd: 3.9501, ept: 555.7203
      Epoch 5 composite train-obj: 1.114412
            No improvement (1.2675), counter 2/5
    Epoch [6/50], Train Losses: mse: 7.0731, mae: 1.4402, huber: 1.1098, swd: 4.0989, ept: 557.3916
    Epoch [6/50], Val Losses: mse: 8.4284, mae: 1.5748, huber: 1.2378, swd: 4.9604, ept: 532.3439
    Epoch [6/50], Test Losses: mse: 6.6973, mae: 1.3868, huber: 1.0603, swd: 3.9269, ept: 552.1121
      Epoch 6 composite train-obj: 1.109760
            Val objective improved 1.2625 → 1.2378, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 7.0390, mae: 1.4343, huber: 1.1049, swd: 4.0832, ept: 558.4403
    Epoch [7/50], Val Losses: mse: 8.5451, mae: 1.5648, huber: 1.2338, swd: 4.9666, ept: 535.1884
    Epoch [7/50], Test Losses: mse: 6.8150, mae: 1.3830, huber: 1.0598, swd: 3.9346, ept: 551.3639
      Epoch 7 composite train-obj: 1.104935
            Val objective improved 1.2378 → 1.2338, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 7.0349, mae: 1.4299, huber: 1.1013, swd: 4.0758, ept: 559.5616
    Epoch [8/50], Val Losses: mse: 8.2640, mae: 1.5691, huber: 1.2340, swd: 4.9362, ept: 542.8337
    Epoch [8/50], Test Losses: mse: 6.5130, mae: 1.3653, huber: 1.0402, swd: 3.9089, ept: 562.0530
      Epoch 8 composite train-obj: 1.101271
            No improvement (1.2340), counter 1/5
    Epoch [9/50], Train Losses: mse: 7.0074, mae: 1.4245, huber: 1.0969, swd: 4.0657, ept: 560.0647
    Epoch [9/50], Val Losses: mse: 8.2310, mae: 1.5622, huber: 1.2296, swd: 4.9009, ept: 539.6921
    Epoch [9/50], Test Losses: mse: 6.5301, mae: 1.3703, huber: 1.0453, swd: 3.8798, ept: 557.7744
      Epoch 9 composite train-obj: 1.096884
            Val objective improved 1.2338 → 1.2296, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 7.0176, mae: 1.4246, huber: 1.0970, swd: 4.0652, ept: 560.0295
    Epoch [10/50], Val Losses: mse: 8.2541, mae: 1.5626, huber: 1.2316, swd: 4.9154, ept: 533.0348
    Epoch [10/50], Test Losses: mse: 6.5560, mae: 1.3723, huber: 1.0475, swd: 3.9030, ept: 554.5160
      Epoch 10 composite train-obj: 1.097013
            No improvement (1.2316), counter 1/5
    Epoch [11/50], Train Losses: mse: 7.0039, mae: 1.4206, huber: 1.0937, swd: 4.0667, ept: 560.5830
    Epoch [11/50], Val Losses: mse: 8.1369, mae: 1.5488, huber: 1.2124, swd: 4.8555, ept: 533.6958
    Epoch [11/50], Test Losses: mse: 6.4917, mae: 1.3624, huber: 1.0363, swd: 3.8458, ept: 555.8568
      Epoch 11 composite train-obj: 1.093729
            Val objective improved 1.2296 → 1.2124, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 6.9891, mae: 1.4183, huber: 1.0916, swd: 4.0569, ept: 560.8923
    Epoch [12/50], Val Losses: mse: 8.1968, mae: 1.5601, huber: 1.2253, swd: 4.8792, ept: 541.5699
    Epoch [12/50], Test Losses: mse: 6.4840, mae: 1.3679, huber: 1.0401, swd: 3.8727, ept: 561.7471
      Epoch 12 composite train-obj: 1.091624
            No improvement (1.2253), counter 1/5
    Epoch [13/50], Train Losses: mse: 7.0037, mae: 1.4192, huber: 1.0929, swd: 4.0591, ept: 560.5282
    Epoch [13/50], Val Losses: mse: 8.0412, mae: 1.5777, huber: 1.2361, swd: 4.8835, ept: 542.6843
    Epoch [13/50], Test Losses: mse: 6.3674, mae: 1.3807, huber: 1.0475, swd: 3.8635, ept: 568.3225
      Epoch 13 composite train-obj: 1.092867
            No improvement (1.2361), counter 2/5
    Epoch [14/50], Train Losses: mse: 6.9820, mae: 1.4158, huber: 1.0898, swd: 4.0511, ept: 560.2371
    Epoch [14/50], Val Losses: mse: 8.1750, mae: 1.5752, huber: 1.2337, swd: 4.8603, ept: 540.9475
    Epoch [14/50], Test Losses: mse: 6.4342, mae: 1.3752, huber: 1.0414, swd: 3.8273, ept: 562.9775
      Epoch 14 composite train-obj: 1.089833
            No improvement (1.2337), counter 3/5
    Epoch [15/50], Train Losses: mse: 6.9826, mae: 1.4153, huber: 1.0891, swd: 4.0478, ept: 561.3162
    Epoch [15/50], Val Losses: mse: 8.1803, mae: 1.5502, huber: 1.2146, swd: 4.8624, ept: 539.9605
    Epoch [15/50], Test Losses: mse: 6.4724, mae: 1.3561, huber: 1.0298, swd: 3.8481, ept: 559.1328
      Epoch 15 composite train-obj: 1.089129
            No improvement (1.2146), counter 4/5
    Epoch [16/50], Train Losses: mse: 6.9912, mae: 1.4139, huber: 1.0887, swd: 4.0564, ept: 561.2589
    Epoch [16/50], Val Losses: mse: 8.2379, mae: 1.5736, huber: 1.2337, swd: 4.8710, ept: 537.7432
    Epoch [16/50], Test Losses: mse: 6.4398, mae: 1.3645, huber: 1.0328, swd: 3.8226, ept: 560.1691
      Epoch 16 composite train-obj: 1.088726
    Epoch [16/50], Test Losses: mse: 6.4917, mae: 1.3624, huber: 1.0363, swd: 3.8458, ept: 555.8568
    Best round's Test MSE: 6.4917, MAE: 1.3624, SWD: 3.8458
    Best round's Validation MSE: 8.1369, MAE: 1.5488, SWD: 4.8555
    Best round's Test verification MSE : 6.4917, MAE: 1.3624, SWD: 3.8458
    Time taken: 40.55 seconds
    
    ==================================================
    Experiment Summary (DLinear_rossler_seq336_pred720_20250513_1206)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 6.5224 ± 0.0358
      mae: 1.3680 ± 0.0040
      huber: 1.0416 ± 0.0040
      swd: 3.8097 ± 0.0292
      ept: 555.6938 ± 0.8297
      count: 33.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 8.1714 ± 0.0248
      mae: 1.5476 ± 0.0074
      huber: 1.2138 ± 0.0057
      swd: 4.7958 ± 0.0450
      ept: 531.5893 ± 1.5095
      count: 33.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 116.83 seconds
    
    Experiment complete: DLinear_rossler_seq336_pred720_20250513_1206
    Model: DLinear
    Dataset: rossler
    Sequence Length: 336
    Prediction Length: 720
    Seeds: [1955, 7, 20]
    






